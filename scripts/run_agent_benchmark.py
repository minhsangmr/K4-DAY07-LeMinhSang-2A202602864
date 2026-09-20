"""Run the five canonical queries through KnowledgeBaseAgent with real providers.

Retrieval uses Gemini Embedding first. If its preflight request fails, Jina
Embeddings becomes the sole retrieval backend for the whole run. Generation
normally uses Gemini; it can use Jina VLM when Gemini is rate-limited. No API
keys or raw HTTP errors are written to the artifact.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bench
from src.agent import KnowledgeBaseAgent
from src.embeddings import GEMINI_EMBEDDING_MODEL, JINA_EMBEDDING_MODEL, GeminiEmbedder, JinaEmbedder
from src.store import EmbeddingStore


raw_output_path = os.getenv("AGENT_OUTPUT_PATH")
OUTPUT_PATH = Path(raw_output_path) if raw_output_path else ROOT / "ket_qua_agent_benchmark.txt"
GENERATION_MODEL = os.getenv("GEMINI_GENERATION_MODEL", "gemini-3.6-flash")
TOP_K = 3


class CachedEmbedder:
    """Avoid duplicate provider calls when agent retrieval repeats a query."""

    def __init__(self, backend: Callable[[str], list[float]]) -> None:
        self.backend = backend
        self._backend_name = getattr(backend, "_backend_name", backend.__class__.__name__)
        self._cache: dict[str, list[float]] = {}

    def __call__(self, text: str) -> list[float]:
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if key not in self._cache:
            self._cache[key] = self.backend(text)
        return self._cache[key]

    def preload(self, texts: list[str]) -> None:
        """Populate the cache with provider batch endpoints when available."""
        unique_texts = list(dict.fromkeys(texts))
        missing = [text for text in unique_texts if hashlib.sha256(text.encode("utf-8")).hexdigest() not in self._cache]
        if not missing:
            return

        if isinstance(self.backend, GeminiEmbedder):
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.backend.model_name}:batchEmbedContents"
            payload = {
                "requests": [
                    {
                        "model": f"models/{self.backend.model_name}",
                        "content": {"parts": [{"text": text}]},
                    }
                    for text in missing
                ]
            }
            response = requests.post(
                url,
                headers={"x-goog-api-key": self.backend.api_key, "Content-Type": "application/json"},
                json=payload,
                timeout=90,
            )
            if not response.ok:
                raise RuntimeError(f"Gemini batch embedding request failed with HTTP {response.status_code}")
            vectors = [item["values"] for item in response.json()["embeddings"]]
        elif isinstance(self.backend, JinaEmbedder):
            response = requests.post(
                "https://api.jina.ai/v1/embeddings",
                headers={"Authorization": f"Bearer {self.backend.api_key}", "Content-Type": "application/json"},
                json={"model": self.backend.model_name, "input": missing},
                timeout=90,
            )
            if not response.ok:
                raise RuntimeError(f"Jina batch embedding request failed with HTTP {response.status_code}")
            vectors = [item["embedding"] for item in sorted(response.json()["data"], key=lambda item: item["index"])]
        else:
            vectors = [self.backend(text) for text in missing]

        for text, vector in zip(missing, vectors, strict=True):
            self._cache[hashlib.sha256(text.encode("utf-8")).hexdigest()] = [float(value) for value in vector]


def choose_embedder() -> CachedEmbedder:
    """Preflight Gemini once; keep a single vector space for every stored chunk."""
    if os.getenv("AGENT_EMBEDDING_PROVIDER", "gemini").lower() == "jina":
        jina = JinaEmbedder(os.getenv("JINA_EMBEDDING_MODEL", JINA_EMBEDDING_MODEL))
        jina("provider preflight")
        return CachedEmbedder(jina)
    try:
        gemini = GeminiEmbedder(os.getenv("GEMINI_EMBEDDING_MODEL", GEMINI_EMBEDDING_MODEL))
        gemini("provider preflight")
        return CachedEmbedder(gemini)
    except Exception:
        jina = JinaEmbedder(os.getenv("JINA_EMBEDDING_MODEL", JINA_EMBEDDING_MODEL))
        jina("provider preflight")
        return CachedEmbedder(jina)


class GeminiGroundedLLM:
    """Minimal Gemini generateContent adapter used only after retrieval."""

    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is required for generation")
        self.model_name = GENERATION_MODEL

    def __call__(self, prompt: str) -> str:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent",
                    headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {"temperature": 0, "maxOutputTokens": 500},
                    },
                    timeout=60,
                )
                if not response.ok:
                    raise RuntimeError(f"Gemini generation request failed with HTTP {response.status_code}")
                parts = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
                answer = "".join(part.get("text", "") for part in parts).strip()
                if answer:
                    return answer
                raise RuntimeError("Gemini generation returned no text")
            except (requests.RequestException, RuntimeError) as error:
                last_error = error
                if attempt < 2:
                    time.sleep(1 + attempt)
        raise RuntimeError("Gemini generation failed after 3 attempts") from last_error

    def answer_batch(self, prompts: list[tuple[str, str]]) -> dict[str, str]:
        """Generate answers for independent agent prompts in one provider call."""
        task_text = "\n\n".join(
            f"=== {label} ===\n{prompt}" for label, prompt in prompts
        )
        batch_prompt = f"""You are running several independent knowledge-base-agent tasks.
For each labeled task below, obey the task's own grounding instruction and use only
that task's context. Return a JSON object only: each key must be exactly the task
label and each value must be its concise answer with that task's [n] citations.

{task_text}"""
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent",
                    headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
                    json={
                        "contents": [{"parts": [{"text": batch_prompt}]}],
                        "generationConfig": {
                            "temperature": 0,
                            "maxOutputTokens": 1400,
                            "responseMimeType": "application/json",
                        },
                    },
                    timeout=90,
                )
                if not response.ok:
                    raise RuntimeError(f"Gemini batch generation request failed with HTTP {response.status_code}")
                parts = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
                raw_text = "".join(part.get("text", "") for part in parts).strip()
                answers = json.loads(raw_text)
                if all(isinstance(answers.get(label), str) and answers[label].strip() for label, _ in prompts):
                    return {label: answers[label].strip() for label, _ in prompts}
                raise RuntimeError("Gemini batch generation returned incomplete JSON answers")
            except (requests.RequestException, RuntimeError, json.JSONDecodeError) as error:
                last_error = error
                if attempt < 2:
                    time.sleep(1 + attempt)
        raise RuntimeError("Gemini batch generation failed after 3 attempts") from last_error


class JinaGroundedLLM:
    """OpenAI-compatible Jina VLM chat fallback for grounded text answers."""

    def __init__(self) -> None:
        self.api_key = os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise RuntimeError("JINA_API_KEY is required for Jina generation")
        self.model_name = os.getenv("JINA_GENERATION_MODEL", "jina-vlm")

    def answer_batch(self, prompts: list[tuple[str, str]]) -> dict[str, str]:
        task_text = "\n\n".join(
            f"=== {label} ===\n{prompt}" for label, prompt in prompts
        )
        batch_prompt = f"""You are running several independent knowledge-base-agent tasks.
For each labeled task below, obey the task's own grounding instruction and use only
that task's context. Return a JSON object only: each key must be exactly the task
label and each value must be its concise answer with that task's [n] citations.

{task_text}"""
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = requests.post(
                    "https://api-beta-vlm.jina.ai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json={
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": batch_prompt}],
                        "temperature": 0,
                        "max_tokens": 1400,
                        "response_format": {"type": "json_object"},
                    },
                    timeout=90,
                )
                if not response.ok:
                    raise RuntimeError(f"Jina batch generation request failed with HTTP {response.status_code}")
                raw_text = response.json()["choices"][0]["message"]["content"].strip()
                answers = json.loads(raw_text)
                if all(isinstance(answers.get(label), str) and answers[label].strip() for label, _ in prompts):
                    return {label: answers[label].strip() for label, _ in prompts}
                raise RuntimeError("Jina batch generation returned incomplete JSON answers")
            except (requests.RequestException, RuntimeError, json.JSONDecodeError) as error:
                last_error = error
                if attempt < 2:
                    time.sleep(1 + attempt)
        raise RuntimeError("Jina batch generation failed after 3 attempts") from last_error


def choose_llm() -> GeminiGroundedLLM | JinaGroundedLLM:
    if os.getenv("AGENT_GENERATION_PROVIDER", "gemini").lower() == "jina":
        return JinaGroundedLLM()
    return GeminiGroundedLLM()


def preview(content: str, limit: int = 220) -> str:
    return re.sub(r"\s+", " ", content).strip()[:limit]


def run() -> str:
    load_dotenv(override=False)
    selected_keys = [key.strip() for key in os.getenv("AGENT_RUN_KEYS", "").split(",") if key.strip()]
    run_specs = [run_spec for run_spec in bench.RUNS if not selected_keys or run_spec.key in selected_keys]
    if selected_keys and len(run_specs) != len(selected_keys):
        raise ValueError("AGENT_RUN_KEYS contains an unknown benchmark label")
    if not run_specs:
        raise ValueError("No benchmark runs selected")

    embedder = choose_embedder()
    documents = bench.build_documents()
    embedder.preload([document.content for document in documents] + [run_spec.query for run_spec in run_specs])
    store = EmbeddingStore("sang_agent_gemini", embedding_fn=embedder)
    store.add_documents(documents)
    llm = choose_llm()
    agent = KnowledgeBaseAgent(store=store, llm_fn=llm)
    evaluations = [
        (
            run_spec,
            store.search_with_filter(
                run_spec.query, top_k=TOP_K, metadata_filter=run_spec.metadata_filter
            ),
        )
        for run_spec in run_specs
    ]
    prompts = [
        (run_spec.key, agent.build_prompt_from_results(run_spec.query, results))
        for run_spec, results in evaluations
    ]
    answers_by_key = llm.answer_batch(prompts)

    lines = [
        "KnowledgeBaseAgent benchmark — Sang",
        f"chunking=FixedSizeChunker(chunk_size={bench.CHUNK_SIZE}, overlap={bench.OVERLAP})",
        f"embedding_backend={embedder._backend_name}",
        f"generation_backend={llm.__class__.__name__}/{llm.model_name}",
        f"stored_chunks={store.get_collection_size()}; top_k={TOP_K}",
        "",
    ]
    for run_spec, results in evaluations:
        answer = answers_by_key[run_spec.key]
        result_doc_ids = [result["metadata"].get("doc_id", "") for result in results]
        gold_doc = any(doc_id in run_spec.gold_doc_ids for doc_id in result_doc_ids)
        marker = any(run_spec.gold_marker.lower() in result["content"].lower() for result in results)
        lines.extend((
            f"=== {run_spec.key} ===",
            f"query={run_spec.query}",
            f"metadata_filter={run_spec.metadata_filter or 'none'}",
            f"gold_doc_in_top3={gold_doc}; answer_span_in_top3={marker}; gold_marker={run_spec.gold_marker}",
        ))
        for rank, result in enumerate(results, start=1):
            metadata = result["metadata"]
            lines.append(
                f"top_{rank}: score={result['score']:.6f}; id={result['id']}; "
                f"doc_id={metadata.get('doc_id')}; audience={metadata.get('audience')}; "
                f"preview={preview(result['content'])}"
            )
        compact_answer = re.sub(r"\s+", " ", answer)
        lines.extend((f"agent_answer={compact_answer}", ""))
    return "\n".join(lines) + "\n"


def main() -> int:
    output = run()
    OUTPUT_PATH.write_text(output, encoding="utf-8")
    print(output, end="")
    print(f"Saved provider-backed agent output to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

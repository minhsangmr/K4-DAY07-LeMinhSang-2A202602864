from __future__ import annotations

import hashlib
import math
import os

# Multilingual model suitable for the Vietnamese corpora used in this Lab.
# The local backend remains optional; required checkpoints use MockEmbedder.
LOCAL_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
JINA_EMBEDDING_MODEL = "jina-embeddings-v3"
EMBEDDING_PROVIDER_ENV = "EMBEDDING_PROVIDER"


class MockEmbedder:
    """Deterministic embedding backend used by tests and default classroom runs."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self._backend_name = "mock embeddings fallback"

    def __call__(self, text: str) -> list[float]:
        digest = hashlib.md5(text.encode()).hexdigest()
        seed = int(digest, 16)
        vector = []
        for _ in range(self.dim):
            seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
            vector.append((seed / 0xFFFFFFFF) * 2 - 1)
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class LocalEmbedder:
    """Sentence Transformers-backed local embedder."""

    def __init__(self, model_name: str = LOCAL_EMBEDDING_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._backend_name = model_name
        self.model = SentenceTransformer(model_name)

    def __call__(self, text: str) -> list[float]:
        embedding = self.model.encode(text, normalize_embeddings=True)
        if hasattr(embedding, "tolist"):
            return embedding.tolist()
        return [float(value) for value in embedding]


class OpenAIEmbedder:
    """OpenAI embeddings API-backed embedder."""

    def __init__(self, model_name: str = OPENAI_EMBEDDING_MODEL) -> None:
        from openai import OpenAI

        self.model_name = model_name
        self._backend_name = model_name
        self.client = OpenAI()

    def __call__(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model_name, input=text)
        return [float(value) for value in response.data[0].embedding]


class GeminiEmbedder:
    """Google Gemini embeddings API-backed embedder.

    The SDK path is used when ``google-genai`` is installed.  The REST path
    keeps the lab runnable with only ``requests`` while using the same public
    Gemini embedContent endpoint.
    """

    def __init__(self, model_name: str = GEMINI_EMBEDDING_MODEL) -> None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is required for GeminiEmbedder")
        self.model_name = model_name
        self._backend_name = model_name
        self.api_key = api_key
        try:
            from google import genai

            self.client = genai.Client(api_key=api_key)
        except ImportError:
            self.client = None

    def __call__(self, text: str) -> list[float]:
        if self.client is not None:
            response = self.client.models.embed_content(model=self.model_name, contents=text)
            return [float(value) for value in response.embeddings[0].values]

        import requests

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:embedContent"
        response = requests.post(
            url,
            headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
            json={
                "model": f"models/{self.model_name}",
                "content": {"parts": [{"text": text}]},
            },
            timeout=45,
        )
        if not response.ok:
            raise RuntimeError(f"Gemini embedding request failed with HTTP {response.status_code}")
        return [float(value) for value in response.json()["embedding"]["values"]]


class JinaEmbedder:
    """Jina Embeddings REST API backend for an explicit provider fallback."""

    def __init__(self, model_name: str = JINA_EMBEDDING_MODEL) -> None:
        api_key = os.getenv("JINA_API_KEY")
        if not api_key:
            raise RuntimeError("JINA_API_KEY is required for JinaEmbedder")
        self.model_name = model_name
        self._backend_name = model_name
        self.api_key = api_key

    def __call__(self, text: str) -> list[float]:
        import requests

        response = requests.post(
            "https://api.jina.ai/v1/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.model_name, "input": [text]},
            timeout=45,
        )
        if not response.ok:
            raise RuntimeError(f"Jina embedding request failed with HTTP {response.status_code}")
        return [float(value) for value in response.json()["data"][0]["embedding"]]


_mock_embed = MockEmbedder()

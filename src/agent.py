from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def build_prompt_from_results(self, question: str, results: list[dict]) -> str:
        """Create the grounded prompt for already-retrieved search results."""
        context_blocks = []
        for index, result in enumerate(results, start=1):
            metadata = result.get("metadata", {})
            source = (
                metadata.get("source_url")
                or metadata.get("source")
                or metadata.get("doc_id")
                or result.get("id", "unknown source")
            )
            context_blocks.append(f"[{index}] Source: {source}\n{result['content']}")

        context = "\n\n".join(context_blocks)
        return f"""You are a knowledge-base assistant.
Answer the question using only the numbered context blocks below. Do not add facts
that are absent from the context. Cite the supporting block number(s), for example
[1], in the answer. If the context does not contain the answer, say so clearly.

Context:
{context}

Question: {question}

Answer:"""

    def answer(self, question: str, top_k: int = 3, metadata_filter: dict | None = None) -> str:
        """Answer from the top-k chunks, optionally restricted by metadata."""
        results = self.store.search_with_filter(
            question, top_k=top_k, metadata_filter=metadata_filter
        )
        if not results:
            return "I could not find relevant information in the knowledge base."
        return self.llm_fn(self.build_prompt_from_results(question, results))

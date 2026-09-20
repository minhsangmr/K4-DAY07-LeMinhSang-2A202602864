from __future__ import annotations

from typing import Any, Callable

from .chunking import compute_similarity
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Uses an in-memory store for the lab's deterministic test contract.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

    def _make_record(self, doc: Document) -> dict[str, Any]:
        """Embed one document and copy its metadata into a stored record."""
        metadata = dict(doc.metadata or {})
        # CP5 creates chunk ids such as ``buyer-open-case#2``. Preserve an
        # explicitly supplied source id; otherwise infer the parent document id.
        metadata["doc_id"] = metadata.get("doc_id") or doc.id.split("#", maxsplit=1)[0]
        record = {
            "id": doc.id,
            "content": doc.content,
            "metadata": metadata,
            "embedding": self._embedding_fn(doc.content),
            "index": self._next_index,
        }
        self._next_index += 1
        return record

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """Rank a supplied candidate set by descending cosine similarity."""
        if not records or top_k <= 0:
            return []

        query_embedding = self._embedding_fn(query)
        ranked = [
            {
                "id": record["id"],
                "content": record["content"],
                "metadata": dict(record["metadata"]),
                "score": compute_similarity(query_embedding, record["embedding"]),
            }
            for record in records
        ]
        ranked.sort(key=lambda result: result["score"], reverse=True)
        return ranked[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """Embed every input document and append exactly one in-memory record."""
        self._store.extend(self._make_record(doc) for doc in docs)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Find the top_k most similar stored records by cosine similarity."""
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        return len(self._store)

    def search_with_filter(
        self,
        query: str,
        top_k: int = 3,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if not metadata_filter:
            candidates = self._store
        else:
            candidates = [
                record
                for record in self._store
                if all(record["metadata"].get(key) == value for key, value in metadata_filter.items())
            ]
        return self._search_records(query, candidates, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        original_size = len(self._store)
        self._store = [
            record for record in self._store if record["metadata"].get("doc_id") != doc_id
        ]
        return len(self._store) != original_size

from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        """Split after sentence-ending punctuation without discarding it.

        This intentionally uses a lightweight rule for the lab: punctuation must
        be followed by whitespace (or the end of the input) to end a sentence.
        It does not try to solve language-specific cases such as abbreviations or
        decimal numbers; those are worth documenting when evaluating the strategy.
        """
        stripped_text = text.strip()
        if not stripped_text:
            return []

        # A lookbehind splits *after* . ! or ?, so the delimiter remains part of
        # the preceding sentence instead of producing incomplete chunks.
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", stripped_text)
            if sentence.strip()
        ]

        return [
            " ".join(sentences[index : index + self.max_sentences_per_chunk]).strip()
            for index in range(0, len(sentences), self.max_sentences_per_chunk)
        ]


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        # A positive limit keeps the hard-split fallback safe even when a caller
        # supplies 0 or a negative value.
        self.chunk_size = max(1, chunk_size)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        # _split preserves boundaries while it recurses. Strip only at the public
        # boundary so a separator retained by one recursion level is still
        # available when parent-level fragments are merged.
        return [piece.strip() for piece in self._split(text, self.separators) if piece.strip()]

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        """Split an oversized string using progressively smaller boundaries.

        A separator is kept with the preceding fragment. Consequently, merging
        small adjacent fragments neither drops punctuation nor removes the space
        or newline that separated the original text.
        """
        if len(current_text) <= self.chunk_size:
            return [current_text] if current_text else []

        if not remaining_separators:
            return [
                current_text[index : index + self.chunk_size]
                for index in range(0, len(current_text), self.chunk_size)
            ]

        separator = remaining_separators[0]
        later_separators = remaining_separators[1:]

        # The empty separator is the explicit final fallback: cut by character
        # count when no semantic boundary can make the text small enough.
        if separator == "":
            return [
                current_text[index : index + self.chunk_size]
                for index in range(0, len(current_text), self.chunk_size)
            ]

        # If this boundary is absent, do not manufacture fragments; try the next,
        # less-preferred separator instead.
        if separator not in current_text:
            return self._split(current_text, later_separators)

        raw_parts = current_text.split(separator)
        # Retain the separator after every part except the last one. This exactly
        # preserves the original boundary while allowing the merge step to pack
        # adjacent short fragments close to chunk_size.
        pieces = [part + separator for part in raw_parts[:-1]] + [raw_parts[-1]]
        fragments: list[str] = []
        for piece in pieces:
            if not piece:
                continue
            if len(piece) <= self.chunk_size:
                fragments.append(piece)
            else:
                fragments.extend(self._split(piece, later_separators))

        return self._merge_fragments(fragments)

    def _merge_fragments(self, fragments: list[str]) -> list[str]:
        """Pack adjacent, already-bounded fragments without exceeding the limit."""
        merged: list[str] = []
        current_chunk = ""
        for fragment in fragments:
            if not fragment:
                continue
            if current_chunk and len(current_chunk) + len(fragment) > self.chunk_size:
                merged.append(current_chunk)
                current_chunk = fragment
            else:
                current_chunk += fragment
        if current_chunk:
            merged.append(current_chunk)
        return merged


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    magnitude_a = math.sqrt(_dot(vec_a, vec_a))
    magnitude_b = math.sqrt(_dot(vec_b, vec_b))
    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0
    return _dot(vec_a, vec_b) / (magnitude_a * magnitude_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        safe_chunk_size = max(1, chunk_size)
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=safe_chunk_size, overlap=0),
            "by_sentences": SentenceChunker(),
            "recursive": RecursiveChunker(chunk_size=safe_chunk_size),
        }
        comparison: dict[str, dict[str, int | float | list[str]]] = {}
        for name, chunker in strategies.items():
            chunks = chunker.chunk(text)
            count = len(chunks)
            comparison[name] = {
                "count": count,
                "avg_length": (sum(len(chunk) for chunk in chunks) / count) if count else 0.0,
                "chunks": chunks,
            }
        return comparison

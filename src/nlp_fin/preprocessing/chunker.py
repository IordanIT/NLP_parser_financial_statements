"""Семантическая разбивка документа на чанки."""

from __future__ import annotations

from nlp_fin.ingestion.models import BlockType, Section
from nlp_fin.preprocessing.models import Chunk


class SemanticChunker:
    """Группирует секции в чанки с привязкой к заголовкам и лимитом длины."""

    def __init__(self, max_words: int = 150, overlap_ratio: float = 0.05) -> None:
        self._max_words = max_words
        self._overlap_words = max(1, round(max_words * overlap_ratio))

    def chunk(self, sections: list[Section]) -> list[Chunk]:
        chunks: list[Chunk] = []
        buffer: list[str] = []
        anchor: str | None = None
        page = 1
        words = 0
        chunk_id = 1

        for section in sections:
            text = section.text.strip()
            if not text:
                continue
            if section.block_type is BlockType.TITLE:
                if buffer:
                    chunks.append(self._make_chunk(buffer, anchor, page, chunk_id))
                    chunk_id += 1
                anchor = text
                page = section.page
                buffer = []
                words = 0
            for piece in self._slice_text(text):
                piece_words = len(piece.split())
                if buffer and words + piece_words > self._max_words:
                    chunks.append(self._make_chunk(buffer, anchor, page, chunk_id))
                    chunk_id += 1
                    buffer = []
                    words = 0
                buffer.append(piece)
                words += piece_words

        if buffer:
            chunks.append(self._make_chunk(buffer, anchor, page, chunk_id))

        return chunks

    def _slice_text(self, text: str) -> list[str]:
        words = text.split()
        if len(words) <= self._max_words:
            return [text]
        step = self._max_words - self._overlap_words
        slices: list[str] = []
        previous_tail: list[str] = []
        for start in range(0, len(words), step):
            segment = words[start : start + step]
            slices.append(" ".join(previous_tail + segment))
            previous_tail = segment[-self._overlap_words :]
        return slices

    @staticmethod
    def _make_chunk(
        buffer: list[str], anchor: str | None, page: int, chunk_id: int
    ) -> Chunk:
        return Chunk(chunk_id=chunk_id, text="\n".join(buffer), section_anchor=anchor, page=page)
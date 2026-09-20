"""Оркестрация предобработки документа."""

from __future__ import annotations

from nlp_fin.config import Settings, get_settings
from nlp_fin.ingestion.common import rows_to_text
from nlp_fin.ingestion.models import BlockType, ParsedDocument, Section
from nlp_fin.preprocessing.chunker import SemanticChunker
from nlp_fin.preprocessing.lemmatizer import Lemmatizer
from nlp_fin.preprocessing.models import DocumentChunks
from nlp_fin.preprocessing.noise_remover import NoiseRemover
from nlp_fin.preprocessing.normalizer import Normalizer


class PreprocessPipeline:
    """Преобразует распарсенный документ в нормализованные чанки."""

    def __init__(self, settings: Settings | None = None) -> None:
        config = settings or get_settings()
        self._normalizer = Normalizer()
        self._noise_remover = NoiseRemover(
            repeat_ratio=config.noise_repeat_ratio,
            header_zone=config.noise_header_zone,
            footer_zone=config.noise_footer_zone,
        )
        self._chunker = SemanticChunker(
            max_words=config.chunk_max_words, overlap_ratio=config.chunk_overlap_ratio
        )
        self._lemmatizer = Lemmatizer()

    def process(self, document: ParsedDocument) -> DocumentChunks:
        sections = [self._normalize_section(section) for section in document.sections]
        cleaned = self._noise_remover.remove(document.model_copy(update={"sections": sections}))
        chunks = self._chunker.chunk(cleaned.sections)
        for chunk in chunks:
            chunk.tokens = self._lemmatizer.lemmatize(chunk.text)
        normalized_text = "\n".join(section.text for section in cleaned.sections if section.text)
        return DocumentChunks(
            normalized_text=normalized_text,
            chunks=chunks,
            metadata={"source": cleaned.source.filename, "chunk_count": str(len(chunks))},
        )

    def _normalize_section(self, section: Section) -> Section:
        text = self._normalizer.normalize(section.text)
        rows = [[self._normalizer.normalize(cell) for cell in row] for row in section.rows]
        if section.block_type is BlockType.TABLE and rows:
            text = rows_to_text(rows)
        return section.model_copy(update={"text": text, "rows": rows})
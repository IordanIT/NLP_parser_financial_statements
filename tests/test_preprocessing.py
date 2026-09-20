"""Тесты лемматизации и пайплайна предобработки."""

from __future__ import annotations

import pytest

from nlp_fin.ingestion.models import (
    BlockType,
    DocumentType,
    ParsedDocument,
    Section,
    SourceInfo,
)
from nlp_fin.preprocessing.lemmatizer import Lemmatizer, model_available
from nlp_fin.preprocessing.models import DocumentChunks
from nlp_fin.preprocessing.pipeline import PreprocessPipeline

HAS_MODEL = model_available()


@pytest.mark.skipif(not HAS_MODEL, reason="spaCy-модель ru_core_news_md не установлена")
def test_lemmatizer_applies_financial_override() -> None:
    tokens = Lemmatizer().lemmatize("Выручку компании составили 123 млн")
    lemmas = [token.lemma for token in tokens]
    assert "выручка" in lemmas
    assert "млн" in lemmas


@pytest.mark.skipif(not HAS_MODEL, reason="spaCy-модель ru_core_news_md не установлена")
def test_lemmatizer_keeps_abbreviation() -> None:
    tokens = Lemmatizer().lemmatize("EBITDA компании выросла")
    lemmas = [token.lemma for token in tokens]
    assert "ebitda" in lemmas


@pytest.mark.skipif(not HAS_MODEL, reason="spaCy-модель ru_core_news_md не установлена")
def test_lemmatizer_lemmatizes_financial_terms() -> None:
    tokens = Lemmatizer().lemmatize("Задолженности и денежные потоки компании")
    lemmas = [token.lemma for token in tokens]
    assert "задолженность" in lemmas
    assert "поток" in lemmas
    assert "денежный" in lemmas


def _document() -> ParsedDocument:
    return ParsedDocument(
        source=SourceInfo(filename="otchet.pdf", document_type=DocumentType.PDF_TEXT),
        sections=[
            Section(page=1, block_type=BlockType.HEADER, text="Шапка отчёта"),
            Section(page=1, block_type=BlockType.TITLE, text="Годовой отчёт"),
            Section(
                page=1,
                block_type=BlockType.PARAGRAPH,
                text="Выручка составила 1 234 567,89 руб.",
            ),
            Section(
                page=1,
                block_type=BlockType.TABLE,
                text="Показатель | Значение | Выручка | 45 678",
                rows=[["Показатель", "Значение"], ["Выручка", "45 678"]],
            ),
            Section(page=1, block_type=BlockType.FOOTER, text="Страница 1"),
        ],
        raw_text="",
    )


def test_pipeline_removes_noise_and_normalizes() -> None:
    result: DocumentChunks = PreprocessPipeline().process(_document())
    assert "Страница" not in result.normalized_text
    assert "Шапка" not in result.normalized_text
    assert "1234567.89" in result.normalized_text
    assert result.metadata["chunk_count"] == "1"


def test_pipeline_builds_anchored_chunks_with_tokens() -> None:
    result: DocumentChunks = PreprocessPipeline().process(_document())
    assert result.chunks
    assert result.chunks[0].section_anchor == "Годовой отчет"
    assert all(chunk.tokens for chunk in result.chunks)
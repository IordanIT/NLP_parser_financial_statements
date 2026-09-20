"""Тесты удаления шума из документа."""

from __future__ import annotations

from nlp_fin.ingestion.models import (
    BlockType,
    DocumentType,
    ParsedDocument,
    Section,
    SourceInfo,
)
from nlp_fin.preprocessing.noise_remover import NoiseRemover


def _document(sections: list[Section]) -> ParsedDocument:
    return ParsedDocument(
        source=SourceInfo(filename="test.pdf", document_type=DocumentType.PDF_TEXT),
        sections=sections,
        raw_text="",
    )


def test_removes_header_and_footer_blocks() -> None:
    document = _document(
        [
            Section(page=1, block_type=BlockType.HEADER, text="Шапка отчёта"),
            Section(page=1, block_type=BlockType.PARAGRAPH, text="Выручка компании"),
            Section(page=1, block_type=BlockType.FOOTER, text="Страница 1"),
        ]
    )
    cleaned = NoiseRemover().remove(document)
    assert [section.text for section in cleaned.sections] == ["Выручка компании"]


def test_removes_regex_noise() -> None:
    document = _document(
        [
            Section(page=1, block_type=BlockType.PARAGRAPH, text="Подробности на стр. 5"),
            Section(page=1, block_type=BlockType.PARAGRAPH, text="ОГРН 1027700132195"),
            Section(
                page=1, block_type=BlockType.PARAGRAPH, text="выручка компании составила 12 млн"
            ),
        ]
    )
    cleaned = NoiseRemover().remove(document)
    assert [section.text for section in cleaned.sections] == ["выручка компании составила 12 млн"]


def test_removes_repeating_paragraphs() -> None:
    document = _document(
        [
            Section(page=1, block_type=BlockType.PARAGRAPH, text="Повторяющаяся строка"),
            Section(page=2, block_type=BlockType.PARAGRAPH, text="Повторяющаяся строка"),
            Section(page=1, block_type=BlockType.PARAGRAPH, text="Уникальная строка"),
        ]
    )
    cleaned = NoiseRemover(repeat_ratio=0.7).remove(document)
    assert [section.text for section in cleaned.sections] == ["Уникальная строка"]


def test_keeps_unique_and_titles() -> None:
    document = _document(
        [
            Section(page=1, block_type=BlockType.TITLE, text="Годовой отчёт"),
            Section(page=1, block_type=BlockType.PARAGRAPH, text="Единственное упоминание"),
        ]
    )
    cleaned = NoiseRemover().remove(document)
    assert len(cleaned.sections) == 2


def test_removes_geometry_header_and_footer() -> None:
    document = _document(
        [
            Section(
                page=1,
                block_type=BlockType.PARAGRAPH,
                text="Шапка",
                y_top=0.01,
                y_bottom=0.04,
            ),
            Section(
                page=1,
                block_type=BlockType.PARAGRAPH,
                text="Текст",
                y_top=0.2,
                y_bottom=0.8,
            ),
            Section(
                page=1,
                block_type=BlockType.PARAGRAPH,
                text="Подвал",
                y_top=0.96,
                y_bottom=0.99,
            ),
        ]
    )
    cleaned = NoiseRemover().remove(document)
    assert [section.text for section in cleaned.sections] == ["Текст"]


def test_keeps_sections_without_coordinates() -> None:
    document = _document(
        [Section(page=1, block_type=BlockType.PARAGRAPH, text="Текст без координат")]
    )
    cleaned = NoiseRemover().remove(document)
    assert [section.text for section in cleaned.sections] == ["Текст без координат"]
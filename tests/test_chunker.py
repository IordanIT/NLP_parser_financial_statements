"""Тесты семантического чанкинга."""

from __future__ import annotations

from nlp_fin.ingestion.models import BlockType, Section
from nlp_fin.preprocessing.chunker import SemanticChunker


def _section(block_type: BlockType, text: str, page: int = 1) -> Section:
    return Section(page=page, block_type=block_type, text=text)


def test_title_starts_new_anchored_chunk() -> None:
    sections = [
        _section(BlockType.TITLE, "Выручка"),
        _section(BlockType.PARAGRAPH, "Компания получила выручку за год"),
        _section(BlockType.TITLE, "Расходы"),
        _section(BlockType.PARAGRAPH, "Расходы выросли"),
    ]
    chunks = SemanticChunker().chunk(sections)
    assert len(chunks) == 2
    assert chunks[0].section_anchor == "Выручка"
    assert chunks[0].text.startswith("Выручка")
    assert chunks[1].section_anchor == "Расходы"


def test_respects_max_words_per_chunk() -> None:
    text = " ".join(str(index) for index in range(45))
    sections = [_section(BlockType.PARAGRAPH, text)]
    chunks = SemanticChunker(max_words=10, overlap_ratio=0.05).chunk(sections)
    assert len(chunks) > 1
    assert all(len(chunk.text.split()) <= 10 for chunk in chunks)


def test_overlap_on_long_text_slices() -> None:
    text = " ".join(f"w{index}" for index in range(30))
    sections = [_section(BlockType.PARAGRAPH, text)]
    chunks = SemanticChunker(max_words=10, overlap_ratio=0.3).chunk(sections)
    assert len(chunks) >= 2
    overlap = set(chunks[0].text.split()) & set(chunks[1].text.split())
    assert overlap


def test_section_without_title_keeps_no_anchor() -> None:
    sections = [_section(BlockType.PARAGRAPH, "Текст без заголовка")]
    chunks = SemanticChunker().chunk(sections)
    assert len(chunks) == 1
    assert chunks[0].section_anchor is None


def test_page_propagates_from_title() -> None:
    sections = [
        _section(BlockType.TITLE, "Баланс", page=3),
        _section(BlockType.PARAGRAPH, "Активы выросли"),
    ]
    chunks = SemanticChunker().chunk(sections)
    assert chunks[0].page == 3
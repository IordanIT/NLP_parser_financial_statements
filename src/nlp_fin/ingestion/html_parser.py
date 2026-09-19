"""Парсер HTML-документов."""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup, Tag

from nlp_fin.ingestion.base import ReportParser
from nlp_fin.ingestion.common import looks_like_title, rows_to_text
from nlp_fin.ingestion.models import (
    BlockType,
    DocumentType,
    ParsedDocument,
    Section,
    SourceInfo,
)

_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_TEXT_TAGS = _HEADING_TAGS | {"p", "li"}


class HtmlParser(ReportParser):
    """Извлекает заголовки, абзацы и таблицы из HTML в порядке следования."""

    def parse(self, path: Path) -> ParsedDocument:
        content = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(content, "html.parser")
        sections: list[Section] = []
        raw_parts: list[str] = []

        for tag in soup.find_all(_content_tags):
            if tag.name in _TEXT_TAGS:
                text = tag.get_text(" ", strip=True)
                block_type = (
                    BlockType.TITLE
                    if tag.name in _HEADING_TAGS or looks_like_title(text)
                    else BlockType.PARAGRAPH
                )
                if text:
                    sections.append(Section(page=1, block_type=block_type, text=text))
                    raw_parts.append(text)
            elif tag.name == "table":
                rows = _table_to_rows(tag)
                if rows:
                    sections.append(
                        Section(
                            page=1,
                            block_type=BlockType.TABLE,
                            text=rows_to_text(rows),
                            rows=rows,
                        )
                    )
                    raw_parts.append(rows_to_text(rows))

        return ParsedDocument(
            source=SourceInfo(filename=path.name, document_type=DocumentType.HTML),
            sections=sections,
            raw_text="\n".join(raw_parts),
        )


def _content_tags(tag: Tag) -> bool:
    """Пропускает только значимые блочные элементы вне вложенных таблиц."""
    if tag.name not in _TEXT_TAGS and tag.name != "table":
        return False
    return tag.find_parent("table") is None


def _table_to_rows(table: Tag) -> list[list[str]]:
    """Преобразует HTML-таблицу в списки строк."""
    rows: list[list[str]] = []
    for row in table.find_all("tr"):
        cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"])]
        if any(cells):
            rows.append(cells)
    return rows
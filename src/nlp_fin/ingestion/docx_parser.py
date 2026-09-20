"""Парсер документов Word."""

from __future__ import annotations

from pathlib import Path

from docx import Document as WordDocument
from docx.table import Table
from docx.text.paragraph import Paragraph

from nlp_fin.ingestion.base import ReportParser
from nlp_fin.ingestion.common import looks_like_title, rows_to_text
from nlp_fin.ingestion.models import (
    BlockType,
    DocumentType,
    ParsedDocument,
    Section,
    SourceInfo,
)


class DocxParser(ReportParser):
    """Извлекает абзацы и таблицы из документов Word в порядке следования."""

    def parse(self, path: Path) -> ParsedDocument:
        sections: list[Section] = []
        raw_parts: list[str] = []
        document = WordDocument(str(path))

        for element in document.iter_inner_content():
            if isinstance(element, Paragraph):
                text = element.text.strip()
                if text:
                    block_type = BlockType.TITLE if looks_like_title(text) else BlockType.PARAGRAPH
                    sections.append(Section(page=1, block_type=block_type, text=text))
                    raw_parts.append(text)
            elif isinstance(element, Table):
                rows = [[cell.text.strip() for cell in row.cells] for row in element.rows]
                text = rows_to_text(rows)
                sections.append(Section(page=1, block_type=BlockType.TABLE, text=text, rows=rows))
                raw_parts.append(text)

        return ParsedDocument(
            source=SourceInfo(filename=path.name, document_type=DocumentType.DOCX),
            sections=sections,
            raw_text="\n".join(raw_parts),
        )
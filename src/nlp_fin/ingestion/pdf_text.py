"""Парсер PDF-документов с текстовым слоем."""

from __future__ import annotations

from pathlib import Path

import pymupdf

from nlp_fin.ingestion.base import ReportParser
from nlp_fin.ingestion.common import looks_like_title, rows_to_text
from nlp_fin.ingestion.models import (
    BlockType,
    DocumentType,
    ParsedDocument,
    Section,
    SourceInfo,
)

HEADER_MARGIN = 0.05
FOOTER_MARGIN = 0.95


class PdfTextParser(ReportParser):
    """Извлекает текст, заголовки и таблицы из PDF с текстовым слоем."""

    def parse(self, path: Path) -> ParsedDocument:
        sections: list[Section] = []
        raw_parts: list[str] = []

        with pymupdf.open(path) as doc:
            for page_index, page in enumerate(doc, start=1):
                tables = page.find_tables().tables
                table_rects = [table.bbox for table in tables]

                for block in page.get_text("blocks", sort=True):
                    x0, y0, x1, y1, text, *_ = block
                    if not text.strip():
                        continue
                    rect = pymupdf.Rect(x0, y0, x1, y1)
                    if any(rect.intersects(table_rect) for table_rect in table_rects):
                        continue
                    cleaned = text.strip()
                    block_type = self._classify_block(page.rect.height, y0, y1, cleaned)
                    sections.append(Section(page=page_index, block_type=block_type, text=cleaned))
                    raw_parts.append(cleaned)

                for table in tables:
                    rows = [[(cell or "").strip() for cell in row] for row in table.extract()]
                    text = rows_to_text(rows)
                    sections.append(
                        Section(page=page_index, block_type=BlockType.TABLE, text=text, rows=rows)
                    )
                    raw_parts.append(text)

        return ParsedDocument(
            source=SourceInfo(filename=path.name, document_type=DocumentType.PDF_TEXT),
            sections=sections,
            raw_text="\n".join(raw_parts),
        )

    @staticmethod
    def _classify_block(page_height: float, y0: float, y1: float, text: str) -> BlockType:
        """Определяет тип блока по его положению на странице."""
        if y0 < page_height * HEADER_MARGIN:
            return BlockType.HEADER
        if y1 > page_height * FOOTER_MARGIN:
            return BlockType.FOOTER
        if looks_like_title(text):
            return BlockType.TITLE
        return BlockType.PARAGRAPH
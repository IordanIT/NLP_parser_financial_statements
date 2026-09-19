"""Модели данных результата парсинга документов."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class DocumentType(StrEnum):
    """Типы поддерживаемых документов."""

    PDF_TEXT = "pdf_text"
    PDF_SCAN = "pdf_scan"
    XLSX = "xlsx"
    DOCX = "docx"
    HTML = "html"


class BlockType(StrEnum):
    """Типы блоков внутри документа."""

    TITLE = "title"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    HEADER = "header"
    FOOTER = "footer"


class SourceInfo(BaseModel):
    """Сведения об исходном файле."""

    filename: str
    document_type: DocumentType


class Section(BaseModel):
    """Блок документа с текстом или таблицей."""

    page: int
    block_type: BlockType
    text: str = ""
    rows: list[list[str]] = Field(default_factory=list)


class ParsedDocument(BaseModel):
    """Результат извлечения структуры из документа."""

    source: SourceInfo
    sections: list[Section]
    raw_text: str
    metadata: dict[str, str] = Field(default_factory=dict)
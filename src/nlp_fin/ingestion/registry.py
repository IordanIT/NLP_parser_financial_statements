"""Фабрика парсеров и определение типа документа."""

from __future__ import annotations

from pathlib import Path

import pymupdf

from nlp_fin.config import OcrSettings, get_settings
from nlp_fin.ingestion.base import ReportParser
from nlp_fin.ingestion.docx_parser import DocxParser
from nlp_fin.ingestion.html_parser import HtmlParser
from nlp_fin.ingestion.models import DocumentType, ParsedDocument
from nlp_fin.ingestion.pdf_scan import PdfScanParser
from nlp_fin.ingestion.pdf_text import PdfTextParser
from nlp_fin.ingestion.xlsx_parser import XlsxParser

_EXTENSION_MAP: dict[str, DocumentType] = {
    ".xlsx": DocumentType.XLSX,
    ".xlsm": DocumentType.XLSX,
    ".docx": DocumentType.DOCX,
    ".html": DocumentType.HTML,
    ".htm": DocumentType.HTML,
}

_PARSERS: dict[DocumentType, type[ReportParser]] = {
    DocumentType.PDF_TEXT: PdfTextParser,
    DocumentType.PDF_SCAN: PdfScanParser,
    DocumentType.XLSX: XlsxParser,
    DocumentType.DOCX: DocxParser,
    DocumentType.HTML: HtmlParser,
}


def detect_document_type(path: str | Path) -> DocumentType:
    """Определяет тип документа по расширению и содержимому файла."""
    path = Path(path)
    extension = path.suffix.lower()
    if extension == ".pdf":
        return _detect_pdf_type(path)
    document_type = _EXTENSION_MAP.get(extension)
    if document_type is None:
        raise ValueError(f"Неподдерживаемый формат файла: {extension}")
    return document_type


def get_parser(path: str | Path) -> ReportParser:
    """Возвращает парсер для документа."""
    path = Path(path)
    document_type = detect_document_type(path)
    return _PARSERS[document_type]()


def parse_document(path: str | Path, ocr: OcrSettings | None = None) -> ParsedDocument:
    """Извлекает текст и структуру из документа."""
    path = Path(path)
    document_type = detect_document_type(path)
    if document_type is DocumentType.PDF_SCAN:
        parser = PdfScanParser(ocr or get_settings().ocr)
    else:
        parser = _PARSERS[document_type]()
    return parser.parse(path)


def _detect_pdf_type(path: Path) -> DocumentType:
    """Определяет, содержит ли PDF текстовый слой."""
    settings = get_settings().pdf
    with pymupdf.open(path) as doc:
        pages = min(settings.sample_pages, doc.page_count)
        text_chars = sum(len(doc[i].get_text().strip()) for i in range(pages)) if pages else 0
    average = text_chars / max(pages, 1)
    return DocumentType.PDF_TEXT if average >= settings.text_min_chars else DocumentType.PDF_SCAN
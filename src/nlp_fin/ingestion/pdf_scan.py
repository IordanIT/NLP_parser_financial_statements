"""Парсер отсканированных PDF-документов с применением OCR."""

from __future__ import annotations

import os
from pathlib import Path

import pymupdf
import pytesseract
from PIL import Image, ImageOps

from nlp_fin.config import OcrSettings
from nlp_fin.ingestion.base import ReportParser
from nlp_fin.ingestion.models import (
    BlockType,
    DocumentType,
    ParsedDocument,
    Section,
    SourceInfo,
)


class PdfScanParser(ReportParser):
    """Распознаёт текст отсканированных PDF через Tesseract."""

    def __init__(self, ocr_settings: OcrSettings | None = None) -> None:
        self._ocr = ocr_settings or OcrSettings()
        if self._ocr.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self._ocr.tesseract_cmd
        if self._ocr.tessdata_prefix:
            os.environ["TESSDATA_PREFIX"] = self._ocr.tessdata_prefix

    def parse(self, path: Path) -> ParsedDocument:
        sections: list[Section] = []
        raw_parts: list[str] = []
        scale = self._ocr.image_dpi / 72.0

        with pymupdf.open(path) as doc:
            for page_index, page in enumerate(doc, start=1):
                text = self._ocr_page(page, scale)
                if text.strip():
                    sections.append(
                        Section(page=page_index, block_type=BlockType.PARAGRAPH, text=text.strip())
                    )
                    raw_parts.append(text.strip())

        return ParsedDocument(
            source=SourceInfo(filename=path.name, document_type=DocumentType.PDF_SCAN),
            sections=sections,
            raw_text="\n".join(raw_parts),
        )

    def _ocr_page(self, page: pymupdf.Page, scale: float) -> str:
        """Распознаёт текст одной страницы."""
        pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        image = ImageOps.grayscale(image)
        image = ImageOps.autocontrast(image)
        language = self._resolve_language()
        config = f"--oem {self._ocr.oem} --psm {self._ocr.psm}"
        return pytesseract.image_to_string(image, lang=language, config=config)

    def _resolve_language(self) -> str:
        """Формирует список языков OCR с учётом латиницы."""
        language = self._ocr.tesseract_lang
        if "eng" not in language:
            return f"{language}+eng"
        return language
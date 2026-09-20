"""Удаление шумовых блоков из документа."""

from __future__ import annotations

import re
from collections import Counter

from nlp_fin.ingestion.models import BlockType, ParsedDocument, Section

_DEFAULT_PATTERNS: tuple[str, ...] = (
    r"стр\.?\s*\d+",
    r"страница\s*\d+",
    r"^\s*\d+\s*/\s*\d+\s*$",
    r"©",
    r"все права защищены",
    r"(огрн|инн|кпп)\s*[:\s]?\s*\d+",
    r"главный бухгалтер",
    r"генеральный директор",
    r"электронная подпись",
    r"ооо\s*[«\"][^»\"]+[»\"]",
)


class NoiseRemover:
    """Очищает документ от колонтитулов, подписей и служебных строк."""

    def __init__(
        self,
        patterns: tuple[str, ...] | None = None,
        repeat_ratio: float = 0.7,
        header_zone: float = 0.05,
        footer_zone: float = 0.95,
    ) -> None:
        default = patterns or _DEFAULT_PATTERNS
        self._patterns = tuple(re.compile(pattern, re.IGNORECASE) for pattern in default)
        self._repeat_ratio = repeat_ratio
        self._header_zone = header_zone
        self._footer_zone = footer_zone

    def remove(self, document: ParsedDocument) -> ParsedDocument:
        sections = self._drop_by_geometry(document.sections)
        sections = self._drop_headers_footers(sections)
        sections = [section for section in sections if not self._matches_noise(section)]
        sections = self._drop_repeating(sections)
        return document.model_copy(update={"sections": sections})

    def _drop_by_geometry(self, sections: list[Section]) -> list[Section]:
        return [
            section
            for section in sections
            if section.y_top is None
            or section.y_bottom is None
            or not (
                section.y_top < self._header_zone
                or section.y_bottom > self._footer_zone
            )
        ]

    def _matches_noise(self, section: Section) -> bool:
        return any(pattern.search(section.text) for pattern in self._patterns)

    @staticmethod
    def _drop_headers_footers(sections: list[Section]) -> list[Section]:
        return [
            section
            for section in sections
            if section.block_type not in (BlockType.HEADER, BlockType.FOOTER)
        ]

    def _drop_repeating(self, sections: list[Section]) -> list[Section]:
        pages = max((section.page for section in sections), default=1)
        counts = Counter(
            section.text
            for section in sections
            if section.block_type is BlockType.PARAGRAPH
        )
        threshold = max(2, round(self._repeat_ratio * pages))
        repeating = {text for text, count in counts.items() if count >= threshold}
        return [
            section
            for section in sections
            if not (
                section.block_type is BlockType.PARAGRAPH and section.text in repeating
            )
        ]
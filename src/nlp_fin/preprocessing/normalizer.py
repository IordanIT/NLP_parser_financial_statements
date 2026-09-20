"""Нормализация текста финансовых отчётов."""

from __future__ import annotations

import re
import unicodedata

_CYRILLIC = "а-яё"
_DASH_CHARS = "—–−‑"
_SOFT_HYPHEN = "\u00ad"
_LINE_BREAK_HYPHEN_RE = re.compile(
    r"(?<=[а-яёА-ЯЁ])-\s*(?:\r\n|\n)\s*(?=[а-яёА-ЯЁ])"
)
_NUMBER_SPACE_RE = re.compile(r"(?<=\d)[\s\u00a0]+(?=\d)")
_DECIMAL_COMMA_RE = re.compile(r"(?<=\d),(?=\d)")
_DATE_RE = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b")
_WHITESPACE_RE = re.compile(r"\s+")

_LOOKALIKE_MAP = {
    "a": "а", "c": "с", "e": "е", "k": "к", "o": "о", "p": "р", "x": "х",
}
_EMBEDDED_RE = re.compile(
    rf"(?<=[{_CYRILLIC}])[{''.join(_LOOKALIKE_MAP)}](?=[{_CYRILLIC}])"
)
_INITIAL_RE = re.compile(rf"(?<![\w])[co](?=[{_CYRILLIC}]{{2,}})", re.IGNORECASE)


class Normalizer:
    """Приводит текст к единому виду для последующего анализа."""

    def normalize(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text)
        normalized = normalized.replace(_SOFT_HYPHEN, "")
        normalized = re.sub(_LINE_BREAK_HYPHEN_RE, "", normalized)
        normalized = normalized.replace("ё", "е").replace("Ё", "Е")
        for character in _DASH_CHARS:
            normalized = normalized.replace(character, "-")
        normalized = re.sub(_EMBEDDED_RE, self._replace_lookalike, normalized)
        normalized = re.sub(_INITIAL_RE, self._replace_lookalike, normalized)
        normalized = re.sub(_NUMBER_SPACE_RE, "", normalized)
        normalized = re.sub(_DECIMAL_COMMA_RE, ".", normalized)
        normalized = re.sub(_DATE_RE, self._format_date, normalized)
        normalized = re.sub(_WHITESPACE_RE, " ", normalized)
        return normalized.strip()

    @staticmethod
    def _replace_lookalike(match: re.Match[str]) -> str:
        token = match.group(0)
        replacement = _LOOKALIKE_MAP.get(token.lower())
        if replacement is None:
            return token
        return replacement.upper() if token.isupper() else replacement

    @staticmethod
    def _format_date(match: re.Match[str]) -> str:
        day, month, year = match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"
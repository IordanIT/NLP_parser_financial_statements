"""Правиловая предразметка NER как baseline для спан-апа в Label Studio."""

from __future__ import annotations

import re

from nlp_fin.constants import (
    FINANCIAL_METRICS,
    METRIC_AMOUNT,
    METRIC_CURRENCY,
    METRIC_NAME,
    METRIC_PERIOD,
    METRIC_UNIT,
)
from nlp_fin.dataset.models import NERSpan

_METRIC_PATTERNS = tuple(
    re.compile(rf"(?<!\w){re.escape(metric.replace('_', ' '))}(?!\w)", re.IGNORECASE)
    for metric in sorted(
        (item.replace("_", " ") for item in FINANCIAL_METRICS), key=len, reverse=True
    )
)

_DATE_RE = re.compile(r"\d{1,2}\.\d{1,2}\.\d{2,4}")
_RANGE_RE = re.compile(r"(?:19|20)\d{2}\s*[—–-]\s*(?:19|20)\d{2}")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\s*год\w*")
_QUARTER_RE = re.compile(r"\b(?:\d+|[IVX]+)\s*квартал\w*")
_MONTHS_RE = re.compile(r"\b\d+\s*месяц\w*")

_PERIOD_PATTERNS = (_DATE_RE, _RANGE_RE, _MONTHS_RE, _QUARTER_RE, _YEAR_RE)

_CURRENCY_RE = re.compile(
    r"(?:руб(?:лей|ля|ль|л)?\.?|₽|RUB|USD|EUR|долл(?:ар\w*)?)", re.IGNORECASE
)

_UNIT_RE = re.compile(
    r"\b(?:тысяч\w*|миллион\w*|миллиард\w*|тыс|млн|млрд|штук|ед)\.?|%"
)

_AMOUNT_RE = re.compile(
    r"[+-]?\d+(?:[ \u00a0]\d{3})+(?:[.,]\d+)?|[+-]?\d+(?:[.,]\d+)?"
)


def _scan(
    text: str,
    pattern: re.Pattern[str],
    label: str,
    occupied: list[tuple[int, int]],
    spans: list[NERSpan],
) -> None:
    for match in pattern.finditer(text):
        if any(match.start() < end and match.end() > start for start, end in occupied):
            continue
        spans.append(
            NERSpan(text=match.group(0), label=label, start=match.start(), end=match.end())
        )
        occupied.append((match.start(), match.end()))


class RuleBasedAnnotator:
    """Предразмечает текст правилами: сначала метрики, затем период и значения."""

    def annotate(self, text: str) -> list[NERSpan]:
        occupied: list[tuple[int, int]] = []
        spans: list[NERSpan] = []
        for pattern in _METRIC_PATTERNS:
            _scan(text, pattern, METRIC_NAME, occupied, spans)
        for pattern in _PERIOD_PATTERNS:
            _scan(text, pattern, METRIC_PERIOD, occupied, spans)
        _scan(text, _UNIT_RE, METRIC_UNIT, occupied, spans)
        _scan(text, _CURRENCY_RE, METRIC_CURRENCY, occupied, spans)
        _scan(text, _AMOUNT_RE, METRIC_AMOUNT, occupied, spans)
        return sorted(spans, key=lambda span: (span.start, span.end - span.start))
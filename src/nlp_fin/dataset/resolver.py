"""Связывание метрики с соседними значениями в рамках чанка."""

from __future__ import annotations

from nlp_fin.constants import (
    METRIC_AMOUNT,
    METRIC_CURRENCY,
    METRIC_NAME,
    METRIC_PERIOD,
    METRIC_UNIT,
)
from nlp_fin.dataset.models import AnnotatedChunk, MetricSlot, NERSpan

_CONTEXT_RADIUS = 80


def _gap(first: NERSpan, second: NERSpan) -> int:
    return max(0, max(first.start, second.start) - min(first.end, second.end))


def _nearest(candidates: list[NERSpan], anchor: NERSpan) -> NERSpan | None:
    if not candidates:
        return None
    return min(candidates, key=lambda candidate: _gap(candidate, anchor))


def _context(text: str, anchor: NERSpan, value: NERSpan | None) -> str:
    core_start = anchor.start
    core_end = anchor.end
    if value is not None:
        core_start = min(core_start, value.start)
        core_end = max(core_end, value.end)
    return text[max(0, core_start - _CONTEXT_RADIUS) : core_end + _CONTEXT_RADIUS]


def resolve_metric_slots(chunk: AnnotatedChunk) -> list[MetricSlot]:
    """Собирает слоты метрик по соседним спанам одного чанка."""
    slots: list[MetricSlot] = []
    for anchor in chunk.spans:
        if anchor.label != METRIC_NAME:
            continue
        value = _nearest(
            [s for s in chunk.spans if s.label == METRIC_AMOUNT], anchor
        )
        period = _nearest(
            [s for s in chunk.spans if s.label == METRIC_PERIOD], anchor
        )
        currency = _nearest(
            [s for s in chunk.spans if s.label == METRIC_CURRENCY], anchor
        )
        unit = _nearest([s for s in chunk.spans if s.label == METRIC_UNIT], anchor)
        slots.append(
            MetricSlot(
                metric=anchor.text,
                value=value.text if value is not None else None,
                period=period.text if period is not None else None,
                currency=currency.text if currency is not None else None,
                unit=unit.text if unit is not None else None,
                context=_context(chunk.text, anchor, value),
            )
        )
    return slots
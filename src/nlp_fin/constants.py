"""Константы предметной области: финансовые метрики, сущности NER, классы рисков."""

from __future__ import annotations

FINANCIAL_METRICS: tuple[str, ...] = (
    "выручка",
    "себестоимость",
    "валовая_прибыль",
    "коммерческие_расходы",
    "управленческие_расходы",
    "прибыль_от_продаж",
    "ebitda",
    "ebit",
    "чистая_прибыль",
    "активы",
    "внеоборотные_активы",
    "оборотные_активы",
    "собственный_капитал",
    "обязательства",
    "долг",
    "денежные_средства",
    "рентабельность",
    "долг_к_ebitda",
    "операционный_денежный_поток",
)

ENTITY_TAGS: tuple[str, ...] = (
    "metric",
    "amount",
    "period",
    "currency",
    "unit",
)

RISK_CLASSES: tuple[str, ...] = (
    "low",
    "medium",
    "high",
)

PERIOD_GRANULARITY: tuple[str, ...] = (
    "quarter",
    "half_year",
    "nine_months",
    "year",
)
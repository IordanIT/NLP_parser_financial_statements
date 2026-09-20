"""Константы предметной области: финансовые метрики, сущности NER, классы рисков."""

from __future__ import annotations

from typing import Literal

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

METRIC_NAME: Literal["metric"] = "metric"
METRIC_AMOUNT: Literal["amount"] = "amount"
METRIC_PERIOD: Literal["period"] = "period"
METRIC_CURRENCY: Literal["currency"] = "currency"
METRIC_UNIT: Literal["unit"] = "unit"

NER_ENTITY_TYPES: tuple[str, ...] = (
    METRIC_NAME,
    METRIC_AMOUNT,
    METRIC_PERIOD,
    METRIC_CURRENCY,
    METRIC_UNIT,
)

NON_ENTITY = "O"

BIOES_LABELS: tuple[str, ...] = (NON_ENTITY,) + tuple(
    f"{edge}-{entity}" for entity in NER_ENTITY_TYPES for edge in "BIES"
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
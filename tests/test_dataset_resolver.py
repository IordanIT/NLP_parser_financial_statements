"""Тесты связывания метрики с соседними значениями в слот."""

from nlp_fin.constants import (
    METRIC_AMOUNT,
    METRIC_CURRENCY,
    METRIC_NAME,
    METRIC_PERIOD,
    METRIC_UNIT,
)
from nlp_fin.dataset.models import AnnotatedChunk, NERSpan
from nlp_fin.dataset.resolver import resolve_metric_slots


def test_resolve_metric_slot() -> None:
    chunk = AnnotatedChunk(
        chunk_id=0,
        text="По итогам 2023 год выручка 1 234 567 тыс. руб.",
        spans=[
            NERSpan(text="2023 год", label=METRIC_PERIOD, start=10, end=18),
            NERSpan(text="выручка", label=METRIC_NAME, start=19, end=26),
            NERSpan(text="1 234 567", label=METRIC_AMOUNT, start=27, end=36),
            NERSpan(text="тыс.", label=METRIC_UNIT, start=37, end=41),
            NERSpan(text="руб.", label=METRIC_CURRENCY, start=42, end=46),
        ],
    )
    slots = resolve_metric_slots(chunk)
    assert len(slots) == 1
    slot = slots[0]
    assert slot.metric == "выручка"
    assert slot.value == "1 234 567"
    assert slot.period == "2023 год"
    assert slot.unit == "тыс."
    assert slot.currency == "руб."
    assert "выручка" in slot.context


def test_nearest_amount_wins() -> None:
    chunk = AnnotatedChunk(
        chunk_id=0,
        text="себестоимость 100 руб. выручка 1 234 567 руб.",
        spans=[
            NERSpan(text="себестоимость", label=METRIC_NAME, start=0, end=13),
            NERSpan(text="100", label=METRIC_AMOUNT, start=14, end=17),
            NERSpan(text="выручка", label=METRIC_NAME, start=23, end=30),
            NERSpan(text="1 234 567", label=METRIC_AMOUNT, start=31, end=40),
        ],
    )
    slots = resolve_metric_slots(chunk)
    assert slots[0].metric == "себестоимость"
    assert slots[0].value == "100"
    assert slots[1].metric == "выручка"
    assert slots[1].value == "1 234 567"


def test_missing_values_left_none() -> None:
    chunk = AnnotatedChunk(
        chunk_id=0,
        text="Значение выручка: нет данных.",
        spans=[NERSpan(text="выручка", label=METRIC_NAME, start=9, end=16)],
    )
    slots = resolve_metric_slots(chunk)
    assert len(slots) == 1
    assert slots[0].value is None
    assert slots[0].period is None
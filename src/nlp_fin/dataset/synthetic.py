"""Генерация синтетических финансовых отчётов с NER-разметкой."""

from __future__ import annotations

import random

from pydantic import BaseModel, Field, model_validator

from nlp_fin.constants import (
    FINANCIAL_METRICS,
    METRIC_AMOUNT,
    METRIC_CURRENCY,
    METRIC_NAME,
    METRIC_PERIOD,
    METRIC_UNIT,
)
from nlp_fin.dataset.models import AnnotatedChunk, AnnotationRecord, NERSpan

_CURRENCIES: tuple[tuple[str | None, str | None], ...] = (
    ("тыс.", "руб."),
    ("млн", "руб."),
    ("млрд", "руб."),
    ("млн", "₽"),
    (None, "руб."),
    ("%", None),
)

_PERIODS: tuple[str, ...] = (
    "2023 год",
    "2023",
    "9 месяцев 2023 года",
    "I квартал 2023 года",
    "2022—2023",
)

_DATES: tuple[str, ...] = ("31.12.2023", "01.01.2023")

_DOCUMENT_TYPES: tuple[tuple[str, str], ...] = (
    ("Отчёт о финансовых результатах", "офр"),
    ("Бухгалтерский баланс", "баланс"),
    ("Пояснения к отчётности", "пояснения"),
)

_CONFUSIONS: dict[str, str] = {
    "о": "o",
    "а": "a",
    "е": "e",
    "с": "c",
    "к": "k",
    "р": "p",
    "0": "O",
    "1": "l",
    "3": "З",
}


class SyntheticOptions(BaseModel):
    """Параметры генератора синтетической разметки."""

    count: int = Field(default=500, ge=1)
    min_entities: int = Field(default=20, ge=1)
    max_entities: int = Field(default=80, ge=1)
    noise_level: float = Field(default=0.02, ge=0, le=1)
    seed: int = 42

    @model_validator(mode="after")
    def _check_bounds(self) -> SyntheticOptions:
        if self.max_entities < self.min_entities:
            raise ValueError("max_entities не может быть меньше min_entities")
        return self


def _format_amount(rng: random.Random) -> str:
    amount = rng.uniform(1e5, 5e10)
    if rng.random() < 0.35:
        return f"{amount:,.2f}".replace(",", " ").replace(".", ",")
    return f"{round(amount):,}".replace(",", " ")


def _ocr_noise(text: str, rng: random.Random, probability: float) -> str:
    chars = list(text)
    for index, char in enumerate(chars):
        replacement = _CONFUSIONS.get(char)
        if replacement is not None and rng.random() < probability:
            chars[index] = replacement
    return "".join(chars)


def _join(parts: list[tuple[str, str | None]]) -> tuple[str, list[NERSpan]]:
    text = "".join(part for part, _ in parts)
    spans: list[NERSpan] = []
    cursor = 0
    for part, label in parts:
        if part and label is not None:
            spans.append(
                NERSpan(text=part, label=label, start=cursor, end=cursor + len(part))
            )
        cursor += len(part)
    return text, spans


class SyntheticGenerator:
    """Порождает отчёты по шаблонам с отслеживанием смещений спанов."""

    def generate(self, options: SyntheticOptions) -> list[AnnotationRecord]:
        rng = random.Random(options.seed)
        return [
            self._report(rng, options, index) for index in range(options.count)
        ]

    def _report(
        self, rng: random.Random, options: SyntheticOptions, index: int
    ) -> AnnotationRecord:
        header, doc_type = rng.choice(_DOCUMENT_TYPES)
        chunks: list[AnnotatedChunk] = [AnnotatedChunk(chunk_id=0, text=header)]
        metrics = tuple(metric.replace("_", " ") for metric in FINANCIAL_METRICS)
        chunk_id = 1
        entities = 0
        while entities < options.min_entities and chunk_id < 200:
            text, spans = self._statement_parts(rng, rng.choice(metrics))
            if rng.random() < 0.2:
                text = _ocr_noise(text, rng, options.noise_level)
                spans = [
                    NERSpan(
                        text=text[span.start : span.end],
                        label=span.label,
                        start=span.start,
                        end=span.end,
                    )
                    for span in spans
                ]
            chunks.append(AnnotatedChunk(chunk_id=chunk_id, text=text, spans=spans))
            entities += len(spans)
            chunk_id += 1
        return AnnotationRecord(
            source=f"synthetic/{doc_type}/{index:04d}", chunks=chunks
        )

    def _statement_parts(
        self, rng: random.Random, metric: str
    ) -> tuple[str, list[NERSpan]]:
        unit, currency = rng.choice(_CURRENCIES)
        if unit == "%":
            amount = f"{rng.uniform(0.5, 60):.1f}".replace(".", ",")
        else:
            amount = _format_amount(rng)
        suffix: list[tuple[str, str | None]] = [
            (f" {unit}", METRIC_UNIT) if unit else ("", None),
            (f" {currency}", METRIC_CURRENCY) if currency else ("", None),
        ]
        variant = rng.random()
        if variant < 0.12:
            return _join(
                [("Значение ", None), (metric, METRIC_NAME), (": нет данных.", None)]
            )
        if variant < 0.35:
            parts: list[tuple[str, str | None]] = [
                ("Значение ", None),
                (metric, METRIC_NAME),
                (" за ", None),
                (rng.choice(_PERIODS), METRIC_PERIOD),
                (": ", None),
                (amount, METRIC_AMOUNT),
            ]
        elif variant < 0.7:
            parts = [
                ("По итогам ", None),
                (rng.choice(_PERIODS), METRIC_PERIOD),
                (" ", None),
                (metric, METRIC_NAME),
                (" — ", None),
                (amount, METRIC_AMOUNT),
            ]
        else:
            parts = [
                ("По состоянию на ", None),
                (rng.choice(_DATES), METRIC_PERIOD),
                (" ", None),
                (metric, METRIC_NAME),
                (": ", None),
                (amount, METRIC_AMOUNT),
            ]
        parts.extend(suffix)
        parts.append((".", None))
        return _join(parts)
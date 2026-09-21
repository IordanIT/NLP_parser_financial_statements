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
    RISK_CLASSES,
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

_AMOUNT_RANGES: dict[str, tuple[float, float]] = {
    "low": (1e6, 5e10),
    "medium": (1e4, 1e10),
    "high": (1e3, 5e9),
}

_NEGATIVE_PROB: dict[str, float] = {
    "low": 0.0,
    "medium": 0.25,
    "high": 0.65,
}

_PERCENT_RANGES: dict[str, tuple[float, float]] = {
    "low": (10.0, 60.0),
    "medium": (0.0, 40.0),
    "high": (-30.0, 15.0),
}

_SIGNALS: dict[str, tuple[str, ...]] = {
    "low": ("Рост ключевых показателей за 2023 год.",),
    "medium": ("Показатели компании стабильны за отчётный период.",),
    "high": ("Отмечено снижение финансовых результатов и рост долговой нагрузки.",),
}

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


def _amount_text(value: float, decimals: bool) -> str:
    absolute = abs(value)
    if decimals:
        text = f"{absolute:,.2f}".replace(",", " ").replace(".", ",")
    else:
        text = f"{round(absolute):,}".replace(",", " ")
    return f"-{text}" if value < 0 else text


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
        risk = rng.choice(RISK_CLASSES)
        header, doc_type = rng.choice(_DOCUMENT_TYPES)
        chunks: list[AnnotatedChunk] = [AnnotatedChunk(chunk_id=0, text=header)]
        metrics = tuple(metric.replace("_", " ") for metric in FINANCIAL_METRICS)
        chunk_id = 1
        entities = 0
        while entities < options.min_entities and chunk_id < 200:
            metric = self._scenario_metric(rng, metrics, risk)
            text, spans = self._statement_parts(rng, metric, risk)
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
            if rng.random() < 0.4:
                signal = rng.choice(_SIGNALS[risk])
                chunks.append(AnnotatedChunk(chunk_id=chunk_id, text=signal))
                chunk_id += 1
        return AnnotationRecord(
            source=f"synthetic/{doc_type}/{index:04d}",
            chunks=chunks,
            risk=risk,
        )

    def _scenario_metric(
        self, rng: random.Random, metrics: tuple[str, ...], risk: str
    ) -> str:
        preferred: tuple[str, str, str] = ("выручка", "ebitda", "чистая прибыль")
        if risk == "high":
            preferred = ("долг к ebitda", "чистая прибыль", "операционный денежный поток")
        elif risk == "low":
            preferred = ("выручка", "ebitda", "чистая прибыль")
        for metric in preferred:
            if metric in metrics and rng.random() < 0.8:
                return metric
        return rng.choice(metrics)

    def _statement_parts(
        self, rng: random.Random, metric: str, risk: str
    ) -> tuple[str, list[NERSpan]]:
        unit, currency = rng.choice(_CURRENCIES)
        if unit == "%":
            low, high = _PERCENT_RANGES[risk]
            amount = f"{rng.uniform(low, high):.1f}".replace(".", ",")
        else:
            low, high = _AMOUNT_RANGES[risk]
            value = rng.uniform(low, high)
            if rng.random() < _NEGATIVE_PROB[risk]:
                value = -value
            amount = _amount_text(value, rng.random() < 0.35)
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
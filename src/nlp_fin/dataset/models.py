"""Модели NER-разметки и слотов метрик."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

EntityLabel = Literal[
    "metric", "amount", "period", "currency", "unit"
]


class NERSpan(BaseModel):
    """Сущность в тексте чанка с границами в символах."""

    text: str
    label: EntityLabel
    start: int
    end: int


class AnnotatedChunk(BaseModel):
    """Чанк текста с размеченными сущностями."""

    chunk_id: int
    text: str
    spans: list[NERSpan] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_span_alignment(self) -> AnnotatedChunk:
        for span in self.spans:
            if self.text[span.start : span.end] != span.text:
                raise ValueError(f"Спан {span.text!r} не совпадает с текстом чанка")
        return self


class MetricSlot(BaseModel):
    """Слот метрики: ядро и связанные значения из одного чанка."""

    metric: str
    period: str | None = None
    currency: str | None = None
    unit: str | None = None
    value: str | None = None
    context: str = ""


class AnnotationRecord(BaseModel):
    """Запись разметки одного документа (одна строка JSONL)."""

    source: str
    chunks: list[AnnotatedChunk] = Field(default_factory=list)
"""Модели данных результата предобработки документа."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Token(BaseModel):
    """Токен текста с леммой."""

    text: str
    lemma: str


class Chunk(BaseModel):
    """Семантический отрывок документа для подачи в модель."""

    chunk_id: int
    text: str
    section_anchor: str | None = None
    page: int = 1
    tokens: list[Token] = Field(default_factory=list)


class DocumentChunks(BaseModel):
    """Результат предобработки: нормализованный текст и чанки документа."""

    normalized_text: str
    chunks: list[Chunk]
    metadata: dict[str, str] = Field(default_factory=dict)
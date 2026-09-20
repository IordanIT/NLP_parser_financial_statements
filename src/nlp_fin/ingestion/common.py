"""Вспомогательные функции для парсеров."""

from __future__ import annotations

TITLE_MAX_LENGTH = 120
_TITLE_STOP_CHARS = ".!?:,;"


def rows_to_text(rows: list[list[str]]) -> str:
    """Собирает текст таблицы через разделитель для плоского представления."""
    return " | ".join(" | ".join(row) for row in rows)


def looks_like_title(text: str) -> bool:
    """Определяет, похож ли блок текста на заголовок."""
    stripped = text.strip()
    if not stripped or len(stripped) > TITLE_MAX_LENGTH or "\n" in stripped:
        return False
    return not any(character in stripped for character in _TITLE_STOP_CHARS)
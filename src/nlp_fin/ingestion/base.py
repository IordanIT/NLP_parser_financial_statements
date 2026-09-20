"""Абстрактный интерфейс парсеров документов."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from nlp_fin.ingestion.models import ParsedDocument


class ReportParser(ABC):
    """Базовый класс парсера отчёта."""

    @abstractmethod
    def parse(self, path: Path) -> ParsedDocument:
        """Извлекает текст и структуру из документа."""
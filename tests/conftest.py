"""Общие фикстуры для тестов."""

from __future__ import annotations

import pytest

from nlp_fin.config import Settings, get_settings


@pytest.fixture
def settings() -> Settings:
    """Возвращает настройки приложения."""
    return get_settings()
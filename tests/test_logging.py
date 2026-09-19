"""Тесты настройки логирования."""

from __future__ import annotations

import logging

from nlp_fin.utils.logging import setup_logging


def test_setup_logging_sets_root_level() -> None:
    setup_logging("DEBUG")
    assert logging.getLogger().level == logging.DEBUG
"""Тесты нормализатора текста."""

from __future__ import annotations

from nlp_fin.preprocessing.normalizer import Normalizer


def test_collapses_spaces_and_strips() -> None:
    assert Normalizer().normalize("  Выручка    составила\n1 млн.  ") == "Выручка составила 1 млн."


def test_normalizes_dashes() -> None:
    assert Normalizer().normalize("прибыль — прибыль – прибыль - 2023–2024") == (
        "прибыль - прибыль - прибыль - 2023-2024"
    )


def test_normalizes_yo_to_e() -> None:
    assert Normalizer().normalize("счёт") == "счет"


def test_normalizes_thousands_separator() -> None:
    assert Normalizer().normalize("Выручка 1 234 567,89 руб.") == "Выручка 1234567.89 руб."
    assert Normalizer().normalize("5 000 000 руб.") == "5000000 руб."


def test_format_date() -> None:
    assert Normalizer().normalize("отчетный период 31.12.2023") == "отчетный период 2023-12-31"
    assert Normalizer().normalize("01/02/2024") == "2024-02-01"


def test_fixes_latin_lookalikes_inside_word() -> None:
    assert Normalizer().normalize("выручkа") == "выручка"
    assert Normalizer().normalize("расхoды") == "расходы"


def test_fixes_initial_latin_lookalike() -> None:
    assert Normalizer().normalize("Oтчет о прибылях") == "Отчет о прибылях"


def test_replaces_non_breaking_space() -> None:
    assert Normalizer().normalize("прибыль\u00a0до\u00a0налога") == "прибыль до налога"


def test_merges_hyphenated_words_over_line_break() -> None:
    assert Normalizer().normalize("выручка\nв размере\nприбы-\nль") == (
        "выручка в размере прибыль"
    )


def test_removes_soft_hyphen() -> None:
    assert Normalizer().normalize("выру\u00adчка за год") == "выручка за год"
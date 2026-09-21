"""Тесты генерации риск-сценариев в синтетической разметке."""

from __future__ import annotations

from nlp_fin.constants import RISK_CLASSES
from nlp_fin.dataset.models import RiskClass
from nlp_fin.dataset.synthetic import SyntheticGenerator, SyntheticOptions


def test_risk_field_is_set_on_records() -> None:
    options = SyntheticOptions(count=10, min_entities=5, max_entities=12, seed=5)
    records = SyntheticGenerator().generate(options)
    assert all(record.risk is not None for record in records)
    assert all(record.risk in RISK_CLASSES for record in records)


def test_risk_classes_match_literal() -> None:
    options = SyntheticOptions(count=3, min_entities=5, max_entities=12, seed=5)
    records = SyntheticGenerator().generate(options)
    risk: RiskClass | None = records[0].risk
    assert risk in {"low", "medium", "high"}


def test_all_risk_classes_are_present() -> None:
    options = SyntheticOptions(count=48, min_entities=5, max_entities=12, seed=5)
    records = SyntheticGenerator().generate(options)
    assert set(record.risk for record in records) == set(RISK_CLASSES)


def test_high_risk_reports_contain_negative_amounts() -> None:
    options = SyntheticOptions(count=48, min_entities=10, max_entities=20, seed=5)
    records = SyntheticGenerator().generate(options)
    high = [record for record in records if record.risk == "high"]
    assert high
    assert any(
        "-" in chunk.text for record in high for chunk in record.chunks
    )
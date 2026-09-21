"""Тесты метрик NER и риск-модели."""

from __future__ import annotations

import pytest

from nlp_fin.constants import BIOES_LABELS
from nlp_fin.training.metrics import (
    brier_multiclass,
    confusion_matrix,
    expected_calibration_error,
    macro_f1,
    ner_metrics,
    risk_metrics,
)

_ID_TO_LABEL = dict(enumerate(BIOES_LABELS))
_LABEL_TO_ID = {label: index for index, label in _ID_TO_LABEL.items()}


def test_macro_f1_perfect_and_wrong() -> None:
    assert macro_f1([0, 0, 1, 1], [0, 0, 1, 1], 2)["macro_f1"] == pytest.approx(1.0)
    score = macro_f1([0, 0, 1, 1], [0, 1, 1, 0], 2)["macro_f1"]
    assert score == pytest.approx(0.5)


def test_macro_f1_empty() -> None:
    result = macro_f1([], [], 3)
    assert result["macro_precision"] == 0.0
    assert result["macro_recall"] == 0.0
    assert result["macro_f1"] == 0.0


def test_brier_multiclass_perfect_and_bad() -> None:
    assert brier_multiclass([0, 1], [[1, 0], [0, 1]]) == pytest.approx(0.0)
    assert brier_multiclass([0, 1], [[1, 0], [1, 0]]) == pytest.approx(1.0)


def test_ece_calibrated_and_miscalibrated() -> None:
    calibrated = expected_calibration_error([0, 1, 0], [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])
    assert calibrated == pytest.approx(1 / 30)
    shifted = expected_calibration_error([0, 1, 1], [[1.0, 0.0], [0.9, 0.1], [0.9, 0.1]])
    assert shifted > calibrated


def test_confusion_matrix_counts() -> None:
    matrix = confusion_matrix([0, 0, 1, 2], [0, 1, 1, 2], 3)
    assert matrix[0][0] == 1
    assert matrix[0][1] == 1
    assert matrix[1][1] == 1
    assert matrix[2][2] == 1


def test_risk_metrics_fields() -> None:
    result = risk_metrics([0, 1, 2], [0, 1, 2], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 3)
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["macro_f1"] == pytest.approx(1.0)
    assert result["brier"] == pytest.approx(0.0)
    assert "confusion_matrix" in result
    assert "ece" in result


def test_ner_metrics_exact() -> None:
    gold = [
        ["B-metric", "I-metric", "E-metric", "O", "S-amount"],
        ["O", "B-amount", "E-amount"],
    ]
    gold_ids = [[_LABEL_TO_ID[label] for label in seq] for seq in gold]
    report = ner_metrics(gold_ids, gold_ids, _ID_TO_LABEL)
    assert report["macro avg"]["f1-score"] == pytest.approx(1.0)
    assert report["micro avg"]["f1-score"] == pytest.approx(1.0)


def test_ner_metrics_with_error() -> None:
    gold = [["B-metric", "I-metric", "E-metric", "O", "S-amount"]]
    pred = [["O", "I-metric", "E-metric", "O", "S-amount"]]
    gold_ids = [[_LABEL_TO_ID[label] for label in seq] for seq in gold]
    pred_ids = [[_LABEL_TO_ID[label] for label in seq] for seq in pred]
    report = ner_metrics(pred_ids, gold_ids, _ID_TO_LABEL)
    assert 0.0 < report["macro avg"]["f1-score"] < 1.0


def test_ner_metrics_skips_ignored() -> None:
    gold_ids = [[-100, _LABEL_TO_ID["S-amount"]]]
    pred_ids = [[-100, _LABEL_TO_ID["S-amount"]]]
    report = ner_metrics(pred_ids, gold_ids, _ID_TO_LABEL)
    assert report["macro avg"]["f1-score"] == pytest.approx(1.0)


def test_ner_metrics_is_json_serializable() -> None:
    import json

    gold_ids = [[-100, _LABEL_TO_ID["S-amount"]]]
    report = ner_metrics(gold_ids, gold_ids, _ID_TO_LABEL)
    json.dumps(report)
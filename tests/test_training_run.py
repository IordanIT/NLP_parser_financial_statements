"""Сквозные тесты обучения и оценки NER и риск-моделей на крошечной BERT."""

from __future__ import annotations

import json
from pathlib import Path

from tokenizer_helpers import make_records, make_tokenizer, tiny_config, write_records

from nlp_fin.constants import BIOES_LABELS, RISK_CLASSES
from nlp_fin.training.data import load_records, split_records
from nlp_fin.training.loop import TrainConfig
from nlp_fin.training.run import evaluate_ner, evaluate_risk, train_ner, train_risk


def _records() -> list:
    pairs = [
        (f"выручка {index}", [("выручка", "metric"), (str(index), "amount")])
        for index in range(6)
    ]
    records = make_records(pairs)
    for index, record in enumerate(records):
        record.risk = RISK_CLASSES[index % len(RISK_CLASSES)]
    return records


def test_train_and_evaluate_ner_end_to_end(tmp_path: Path) -> None:
    tokenizer = make_tokenizer(tmp_path, max_length=128)
    records = _records()
    path = write_records(tmp_path / "data", records)
    train, val = split_records(records, ratio=0.8, seed=7)
    out_dir = tmp_path / "out" / "ner"
    config = TrainConfig(epochs=2, batch_size=2, grad_accumulation=2, learning_rate=0.01)
    outcome = train_ner(
        train,
        val,
        out_dir,
        model_name=tiny_config(len(BIOES_LABELS)),
        num_labels=len(BIOES_LABELS),
        config=config,
        max_length=16,
        tokenizer=tokenizer,
    )
    assert outcome.best_epoch >= 1
    assert outcome.best_state
    assert (out_dir / "config.json").is_file()
    report = evaluate_ner(
        path,
        out_dir,
        tokenizer=tokenizer,
        num_labels=len(BIOES_LABELS),
        max_length=16,
    )
    assert 0.0 < report["macro avg"]["f1-score"] <= 1.0
    json.dumps(report)


def test_train_and_evaluate_risk_end_to_end(tmp_path: Path) -> None:
    tokenizer = make_tokenizer(tmp_path, max_length=128)
    records = _records()
    path = write_records(tmp_path / "data", records)
    train, val = split_records(records, ratio=0.8, seed=7)
    out_dir = tmp_path / "out" / "risk"
    config = TrainConfig(epochs=2, batch_size=2, grad_accumulation=2, learning_rate=0.01)
    outcome = train_risk(
        train,
        val,
        out_dir,
        model_name=tiny_config(len(RISK_CLASSES)),
        num_labels=len(RISK_CLASSES),
        config=config,
        max_length=32,
        dropout=0.3,
        tokenizer=tokenizer,
    )
    assert outcome.best_epoch >= 1
    assert (out_dir / "config.json").is_file()
    report = evaluate_risk(
        path,
        out_dir,
        tokenizer=tokenizer,
        num_labels=len(RISK_CLASSES),
        max_length=32,
    )
    assert report["n"] == len(records)
    assert report["macro_f1"] >= 0.0
    assert len(report["confusion_matrix"]) == len(RISK_CLASSES)
    json.dumps(report)


def test_load_records_after_roundtrip(tmp_path: Path) -> None:
    records = _records()
    path = write_records(tmp_path / "data", records)
    assert load_records(path) == records
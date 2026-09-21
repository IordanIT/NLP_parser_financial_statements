"""Тесты построения выборок и загрузки разметки для обучения."""

from __future__ import annotations

from pathlib import Path

from tokenizer_helpers import make_records, make_tokenizer, write_records

from nlp_fin.constants import RISK_CLASSES
from nlp_fin.training.data import (
    id_to_risk,
    load_records,
    ner_dataset,
    risk_dataset,
    risk_to_id,
    split_records,
)


def test_risk_id_mapping_roundtrip() -> None:
    for risk in RISK_CLASSES:
        assert id_to_risk(risk_to_id(risk)) == risk


def test_load_records_roundtrip(tmp_path: Path) -> None:
    records = make_records([("выручка 5", [("выручка", "metric"), ("5", "amount")])])
    path = write_records(tmp_path, records)
    assert load_records(path) == records


def test_split_records_splits_and_deterministic(tmp_path: Path) -> None:
    pairs = [
        (f"выручка {index}", [("выручка", "metric"), (str(index), "amount")])
        for index in range(10)
    ]
    records = make_records(pairs)
    train_one, val_one = split_records(records, ratio=0.8, seed=7)
    train_two, val_two = split_records(records, ratio=0.8, seed=7)
    assert len(train_one) == 8
    assert len(val_one) == 2
    assert train_one == train_two
    assert val_one == val_two


def test_split_records_different_seeds_reshuffle(tmp_path: Path) -> None:
    pairs = [
        (f"выручка {index}", [("выручка", "metric"), (str(index), "amount")])
        for index in range(10)
    ]
    records = make_records(pairs)
    train_one, _ = split_records(records, ratio=0.8, seed=1)
    train_two, _ = split_records(records, ratio=0.8, seed=99)
    assert train_one != train_two


def test_ner_dataset_shapes_and_alignment(tmp_path: Path) -> None:
    tokenizer = make_tokenizer(tmp_path, max_length=128)
    records = make_records([("выручка 5", [("выручка", "metric"), ("5", "amount")])])
    dataset = ner_dataset(records, tokenizer, max_length=16)
    input_ids, attention_mask, labels = dataset.tensors
    assert input_ids.shape == (1, 16)
    assert attention_mask.shape == (1, 16)
    assert labels.shape == (1, 16)
    assert labels[0, 0].item() == -100


def test_ner_dataset_uses_all_chunks(tmp_path: Path) -> None:
    tokenizer = make_tokenizer(tmp_path, max_length=128)
    records = make_records([("выручка 5", [("выручка", "metric"), ("5", "amount")])])
    records[0].chunks.append(records[0].chunks[0].model_copy())
    dataset = ner_dataset(records, tokenizer, max_length=16)
    assert dataset.tensors[0].shape[0] == 2


def test_risk_dataset_encodes_and_pads(tmp_path: Path) -> None:
    tokenizer = make_tokenizer(tmp_path, max_length=128)
    records = make_records([("выручка 5", [("выручка", "metric"), ("5", "amount")])])
    records[0].risk = "high"
    dataset = risk_dataset(records, tokenizer, max_length=32)
    input_ids, attention_mask, risk_ids = dataset.tensors
    assert input_ids.shape == (1, 32)
    assert attention_mask.sum() > 0
    assert risk_ids[0].item() == risk_to_id("high")


def test_risk_dataset_skips_missing_risk(tmp_path: Path) -> None:
    tokenizer = make_tokenizer(tmp_path, max_length=128)
    records = make_records([("выручка 5", [("выручка", "metric"), ("5", "amount")])])
    records[0].risk = None
    dataset = risk_dataset(records, tokenizer, max_length=32)
    assert dataset.tensors[0].shape[0] == 0
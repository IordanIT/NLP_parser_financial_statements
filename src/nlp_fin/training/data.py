"""Построение тензорных выборок NER и риска из разметки."""

from __future__ import annotations

import random
from pathlib import Path
from typing import cast

import torch
from torch.utils.data import TensorDataset
from transformers import PreTrainedTokenizerBase

from nlp_fin.constants import BIOES_LABELS, RISK_CLASSES
from nlp_fin.dataset.encode import NerEncoder
from nlp_fin.dataset.models import AnnotationRecord, RiskClass

_RISK_TO_ID = {risk: index for index, risk in enumerate(RISK_CLASSES)}


def load_records(path: Path) -> list[AnnotationRecord]:
    """Читает записи разметки из JSONL."""
    return [
        AnnotationRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def split_records(
    records: list[AnnotationRecord], ratio: float = 0.85, seed: int = 7
) -> tuple[list[AnnotationRecord], list[AnnotationRecord]]:
    """Перемешивает записи и делит на обучающую и валидационную части."""
    shuffled = list(records)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    cutoff = max(1, int(len(shuffled) * ratio))
    if cutoff >= len(shuffled):
        cutoff = max(0, len(shuffled) - 1)
    return shuffled[:cutoff], shuffled[cutoff:]


def ner_dataset(
    records: list[AnnotationRecord],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    label_set: tuple[str, ...] = BIOES_LABELS,
) -> TensorDataset:
    """Выборка NER: input_ids, attention_mask, BIOES-метки (с -100 у спецтокенов)."""
    encoder = NerEncoder(tokenizer, max_length=max_length, label_set=label_set)
    input_ids: list[list[int]] = []
    attention_masks: list[list[int]] = []
    labels: list[list[int]] = []
    for record in records:
        for chunk in record.chunks:
            item = encoder.encode_chunk(chunk)
            input_ids.append(item["input_ids"])
            attention_masks.append(item["attention_mask"])
            labels.append(item["labels"])
    return TensorDataset(
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_masks, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def risk_to_id(risk: RiskClass) -> int:
    return _RISK_TO_ID[risk]


def id_to_risk(risk_id: int) -> RiskClass:
    return cast(RiskClass, RISK_CLASSES[risk_id])


def risk_dataset(
    records: list[AnnotationRecord],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> TensorDataset:
    """Выборка риска: отчёт целиком в одну последовательность плюс метка класса."""
    input_ids: list[list[int]] = []
    attention_masks: list[list[int]] = []
    risk_ids: list[int] = []
    for record in records:
        if record.risk is None:
            continue
        text = "\n".join(chunk.text for chunk in record.chunks)
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        input_ids.append(encoded["input_ids"])
        attention_masks.append(encoded["attention_mask"])
        risk_ids.append(risk_to_id(record.risk))
    return TensorDataset(
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_masks, dtype=torch.long),
        torch.tensor(risk_ids, dtype=torch.long),
    )
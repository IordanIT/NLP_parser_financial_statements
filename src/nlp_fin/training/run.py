"""Оркестрация обучения и оценки NER и риск-моделей."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from nlp_fin.constants import BIOES_LABELS, RISK_CLASSES
from nlp_fin.dataset.models import AnnotationRecord
from nlp_fin.training.build import ModelSource, build_ner_model, build_risk_model
from nlp_fin.training.data import load_records, ner_dataset, risk_dataset
from nlp_fin.training.loop import TrainConfig, Trainer, TrainOutcome
from nlp_fin.training.metrics import macro_f1, ner_metrics, risk_metrics
from nlp_fin.training.weights import class_weights


def _tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_name)


def _eval_ner(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> float:
    torch_device = torch.device(device)
    model.eval()
    gold_ids: list[list[int]] = []
    pred_ids: list[list[int]] = []
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            logits = model(
                input_ids=input_ids.to(torch_device),
                attention_mask=attention_mask.to(torch_device),
            ).logits
            for gold, pred in zip(
                labels.tolist(), logits.argmax(dim=-1).tolist(), strict=True
            ):
                gold_ids.append(gold)
                pred_ids.append(pred)
    report = ner_metrics(pred_ids, gold_ids, dict(enumerate(BIOES_LABELS)))
    return float(report["macro avg"]["f1-score"])


def _eval_risk(model: nn.Module, loader: DataLoader, device: str) -> float:
    torch_device = torch.device(device)
    model.eval()
    true_ids: list[int] = []
    pred_ids: list[int] = []
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            logits = model(
                input_ids=input_ids.to(torch_device),
                attention_mask=attention_mask.to(torch_device),
            ).logits
            true_ids.extend(labels.tolist())
            pred_ids.extend(logits.argmax(dim=-1).tolist())
    return float(macro_f1(true_ids, pred_ids, len(RISK_CLASSES))["macro_f1"])


def train_ner(
    train_records: list[AnnotationRecord],
    val_records: list[AnnotationRecord],
    out_dir: Path,
    *,
    model_name: ModelSource,
    num_labels: int,
    config: TrainConfig,
    max_length: int,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> TrainOutcome:
    """Обучает NER-модель и сохраняет лучший чекпоинт в out_dir."""
    tok = tokenizer or _tokenizer(str(model_name))
    train_dataset = ner_dataset(train_records, tok, max_length, BIOES_LABELS)
    val_dataset = ner_dataset(val_records, tok, max_length, BIOES_LABELS)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    weights = class_weights(train_dataset.tensors[2], num_labels)
    loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
    model = build_ner_model(model_name, num_labels)

    trainer = Trainer(model, lambda holder: _eval_ner(holder, val_loader, config.device), config)
    outcome = trainer.run(train_loader, loss_fn)
    model.load_state_dict(outcome.best_state)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    return outcome


def train_risk(
    train_records: list[AnnotationRecord],
    val_records: list[AnnotationRecord],
    out_dir: Path,
    *,
    model_name: ModelSource,
    num_labels: int,
    config: TrainConfig,
    max_length: int,
    dropout: float,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> TrainOutcome:
    """Обучает риск-модель и сохраняет лучший чекпоинт в out_dir."""
    tok = tokenizer or _tokenizer(str(model_name))
    train_dataset = risk_dataset(train_records, tok, max_length)
    val_dataset = risk_dataset(val_records, tok, max_length)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    loss_fn = nn.CrossEntropyLoss()
    model = build_risk_model(model_name, num_labels, dropout)

    trainer = Trainer(model, lambda holder: _eval_risk(holder, val_loader, config.device), config)
    outcome = trainer.run(train_loader, loss_fn)
    model.load_state_dict(outcome.best_state)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    return outcome


def evaluate_ner(
    data_path: Path,
    model_path: Path,
    *,
    tokenizer: PreTrainedTokenizerBase,
    num_labels: int,
    max_length: int,
    batch_size: int = 32,
    device: str = "cpu",
) -> dict[str, Any]:
    """seqeval-отчёт NER по всем чанкам набора данных."""
    loader = DataLoader(
        ner_dataset(load_records(data_path), tokenizer, max_length, BIOES_LABELS),
        batch_size=batch_size,
    )
    model = build_ner_model(str(model_path), num_labels).to(torch.device(device)).eval()
    gold_ids: list[list[int]] = []
    pred_ids: list[list[int]] = []
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            logits = model(
                input_ids=input_ids.to(torch.device(device)),
                attention_mask=attention_mask.to(torch.device(device)),
            ).logits
            for gold, pred in zip(
                labels.tolist(), logits.argmax(dim=-1).tolist(), strict=True
            ):
                gold_ids.append(gold)
                pred_ids.append(pred)
    return ner_metrics(pred_ids, gold_ids, dict(enumerate(BIOES_LABELS)))


def evaluate_risk(
    data_path: Path,
    model_path: Path,
    *,
    tokenizer: PreTrainedTokenizerBase,
    num_labels: int,
    max_length: int,
    batch_size: int = 32,
    device: str = "cpu",
) -> dict[str, Any]:
    """Отчёт риск-модели: macro-F1, матрица ошибок, калибровка."""
    records = load_records(data_path)
    loader = DataLoader(risk_dataset(records, tokenizer, max_length), batch_size=batch_size)
    model = build_risk_model(str(model_path), num_labels, 0.0).to(torch.device(device)).eval()
    true_ids: list[int] = []
    pred_ids: list[int] = []
    probabilities: list[list[float]] = []
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            logits = model(
                input_ids=input_ids.to(torch.device(device)),
                attention_mask=attention_mask.to(torch.device(device)),
            ).logits
            probs = torch.softmax(logits, dim=-1)
            true_ids.extend(labels.tolist())
            pred_ids.extend(logits.argmax(dim=-1).tolist())
            probabilities.extend(probs.tolist())
    return risk_metrics(true_ids, pred_ids, probabilities, num_labels)
"""Тесты цикла обучения: планировщик, early stopping, сходимость."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers.modeling_outputs import SequenceClassifierOutput

from nlp_fin.training.loop import TrainConfig, Trainer, _linear_scheduler


class _DummyModel(nn.Module):
    """Минимальная модель с одним линейным слоем для циклов обучения."""

    def __init__(self, in_features: int = 1, num_classes: int = 2) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> SequenceClassifierOutput:
        return SequenceClassifierOutput(logits=self.linear(input_ids.float()))


def _loader(n: int) -> DataLoader:
    inputs = torch.linspace(0.0, 1.0, n).reshape(-1, 1)
    labels = (inputs[:, 0] >= 0.5).long()
    return DataLoader(TensorDataset(inputs, inputs.clone(), labels), batch_size=n)


def test_trainer_stops_on_patience() -> None:
    model = _DummyModel()
    scores: Iterator[float] = iter([0.5, 0.4, 0.3, 0.2, 0.1])

    def evaluate(_: nn.Module) -> float:
        return next(scores)

    config = TrainConfig(epochs=5, batch_size=8, grad_accumulation=8, patience=2)
    outcome = Trainer(model, evaluate, config).run(_loader(8), nn.CrossEntropyLoss())
    assert outcome.best_score == 0.5
    assert outcome.best_epoch == 1
    assert len(outcome.history) == 3


def test_trainer_keeps_late_best() -> None:
    model = _DummyModel()
    scores: Iterator[float] = iter([0.1, 0.9, 0.5])

    def evaluate(_: nn.Module) -> float:
        return next(scores)

    config = TrainConfig(epochs=4, batch_size=8, grad_accumulation=8, patience=1)
    outcome = Trainer(model, evaluate, config).run(_loader(8), nn.CrossEntropyLoss())
    assert outcome.best_epoch == 2
    assert outcome.best_score == 0.9
    assert list(outcome.best_state)


def test_trainer_converges_on_separable_task() -> None:
    torch.manual_seed(0)
    model = _DummyModel()
    loader = _loader(32)

    def score(holder: nn.Module) -> float:
        input_ids, _, labels = next(iter(loader))
        holder.eval()
        with torch.no_grad():
            logits = holder(input_ids).logits
        predictions = logits.argmax(dim=-1)
        return (predictions == labels).sum().item() / len(labels)

    config = TrainConfig(
        epochs=60,
        batch_size=32,
        grad_accumulation=32,
        learning_rate=0.2,
        patience=50,
    )
    outcome = Trainer(model, score, config).run(loader, nn.CrossEntropyLoss())
    assert outcome.best_score >= 0.9


def test_linear_scheduler_warmup_then_decay() -> None:
    model = _DummyModel()
    optimizer = AdamW(model.parameters(), lr=0.1)
    scheduler = _linear_scheduler(optimizer, warmup_steps=2, total_steps=6)
    rates: list[float] = []
    for _ in range(6):
        optimizer.step()
        scheduler.step()
        rates.append(float(scheduler.get_last_lr()[0]))
    assert rates[0] == pytest.approx(0.1)
    assert rates[1] == pytest.approx(0.1)
    assert all(rates[i] >= rates[i + 1] for i in range(1, len(rates) - 1))
    assert rates[-1] == pytest.approx(0.0)
    assert all(rate <= 0.1 + 1e-9 for rate in rates)


def test_linear_scheduler_no_warmup_decays_from_base() -> None:
    model = _DummyModel()
    optimizer = AdamW(model.parameters(), lr=0.1)
    scheduler = _linear_scheduler(optimizer, warmup_steps=0, total_steps=3)
    rates: list[float] = []
    for _ in range(3):
        optimizer.step()
        scheduler.step()
        rates.append(float(scheduler.get_last_lr()[0]))
    assert rates[0] > rates[1] > rates[2]
    assert rates[-1] == pytest.approx(0.0)
    assert all(rate <= 0.1 + 1e-9 for rate in rates)
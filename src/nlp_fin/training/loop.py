"""Цикл обучения модели с ранней остановкой."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

LabelLoss = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
Evaluator = Callable[[nn.Module], float]


@dataclass
class TrainConfig:
    """Гиперпараметры обучения."""

    epochs: int
    batch_size: int = 16
    learning_rate: float = 2e-5
    grad_accumulation: int = 4
    warmup_ratio: float = 0.06
    patience: int = 2
    device: str = "cpu"


@dataclass
class TrainOutcome:
    """Результат обучения."""

    best_score: float
    best_epoch: int
    best_state: dict[str, torch.Tensor]
    history: list[dict[str, float]] = field(default_factory=list)


def _linear_scheduler(
    optimizer: AdamW, warmup_steps: int, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        remainder = max(1, total_steps - warmup_steps)
        progress = (step - warmup_steps) / remainder
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _detach_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


class Trainer:
    """Тренирует модель: AdamW, linear-warmup, градиентная аккумуляция, early stopping."""

    def __init__(self, model: nn.Module, evaluate: Evaluator, config: TrainConfig) -> None:
        self._model = model
        self._evaluate = evaluate
        self._config = config

    def run(self, train_loader: DataLoader, loss_fn: LabelLoss) -> TrainOutcome:
        device = torch.device(self._config.device)
        model = self._model.to(device)
        optimizer = AdamW(model.parameters(), lr=self._config.learning_rate)
        steps_per_epoch = len(train_loader) // self._config.grad_accumulation
        if len(train_loader) % self._config.grad_accumulation:
            steps_per_epoch += 1
        total_steps = steps_per_epoch * self._config.epochs
        scheduler = _linear_scheduler(
            optimizer, int(total_steps * self._config.warmup_ratio), total_steps
        )
        best_score = float("-inf")
        best_epoch = 0
        best_state = _detach_state(model)
        history: list[dict[str, float]] = []
        no_improve = 0
        for epoch in range(1, self._config.epochs + 1):
            model.train()
            running = 0.0
            seen = 0
            optimizer.zero_grad()
            micro_steps = 0
            for batch in train_loader:
                input_ids, attention_mask, labels = (tensor.to(device) for tensor in batch)
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                loss = (
                    loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                    / self._config.grad_accumulation
                )
                loss.backward()
                running += float(loss.detach()) * self._config.grad_accumulation
                seen += 1
                micro_steps += 1
                if micro_steps % self._config.grad_accumulation == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            if micro_steps % self._config.grad_accumulation:
                optimizer.step()
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            score = self._evaluate(model)
            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": running / seen if seen else 0.0,
                    "score": score,
                }
            )
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = _detach_state(model)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self._config.patience:
                    break
        return TrainOutcome(
            best_score=best_score,
            best_epoch=best_epoch,
            best_state=best_state,
            history=history,
        )
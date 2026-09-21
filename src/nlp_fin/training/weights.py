"""Веса классов для CrossEntropyLoss с учётом дисбаланса фона."""

from __future__ import annotations

import torch


def class_weights(
    labels: torch.Tensor, num_classes: int, ignore_index: int = -100
) -> torch.Tensor:
    """Нормированные веса по обратной частоте меток (среднее по классам равно 1).

    Классы, отсутствующие в разметке, получают вес 1.0.
    """
    weights = torch.ones(num_classes)
    valid = labels[labels != ignore_index]
    if valid.numel() == 0:
        return weights
    counts = torch.bincount(valid, minlength=num_classes).float()
    present = counts > 0
    inverse = torch.zeros(num_classes)
    inverse[present] = 1.0 / counts[present]
    weights[present] = inverse[present] * int(present.sum()) / inverse[present].sum()
    return weights
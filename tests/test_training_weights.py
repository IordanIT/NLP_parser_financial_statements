"""Тесты весов классов для CrossEntropyLoss."""

from __future__ import annotations

import torch

from nlp_fin.training.weights import class_weights


def test_balanced_labels_give_unit_weights() -> None:
    labels = torch.tensor([[0, 1, 0, 1], [0, 1, 0, 1]])
    weights = class_weights(labels, num_classes=2)
    assert torch.allclose(weights, torch.ones(2))


def test_imbalanced_counts_invert_frequencies() -> None:
    labels = torch.tensor([0, 0, 0, 1, 1])
    weights = class_weights(labels, num_classes=2)
    assert weights[0] < weights[1]
    assert abs(float(weights.mean()) - 1.0) < 1e-6


def test_absent_class_keeps_weight_one() -> None:
    labels = torch.tensor([0, 0, 1])
    weights = class_weights(labels, num_classes=3)
    assert weights[2] == 1.0
    assert weights[0] < weights[1]
    assert abs(float(weights[:2].mean()) - 1.0) < 1e-6


def test_ignore_index_excluded() -> None:
    labels = torch.tensor([-100, -100, 0, 0, 1])
    weights = class_weights(labels, num_classes=2)
    assert weights[0] < 1.0
    assert weights[1] > 1.0
    assert abs(float(weights.mean()) - 1.0) < 1e-6


def test_all_ignored_returns_unit_weights() -> None:
    labels = torch.tensor([-100, -100])
    weights = class_weights(labels, num_classes=4)
    assert torch.allclose(weights, torch.ones(4))
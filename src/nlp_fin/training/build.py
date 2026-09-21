"""Конструирование моделей NER и риска на базе BERT."""

from __future__ import annotations

from copy import deepcopy

from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertForTokenClassification,
)

ModelSource = str | BertConfig


def build_ner_model(source: ModelSource, num_labels: int) -> BertForTokenClassification:
    """Создаёт NER-модель из имени предобученной модели или конфигурации."""
    if isinstance(source, str):
        return BertForTokenClassification.from_pretrained(source, num_labels=num_labels)
    config = deepcopy(source)
    config.num_labels = num_labels
    return BertForTokenClassification(config)


def build_risk_model(
    source: ModelSource, num_labels: int, dropout: float
) -> BertForSequenceClassification:
    """Создаёт риск-модель с головой на pooled-выводе и заданным dropout."""
    if isinstance(source, str):
        return BertForSequenceClassification.from_pretrained(
            source, num_labels=num_labels, classifier_dropout=dropout
        )
    config = deepcopy(source)
    config.num_labels = num_labels
    config.classifier_dropout = dropout
    return BertForSequenceClassification(config)
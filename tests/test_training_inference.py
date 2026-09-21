"""Тесты совместного инференса NER и риска (ReportAnalyzer)."""

from __future__ import annotations

from pathlib import Path

import torch
from tokenizer_helpers import make_tokenizer, save_tiny_models
from transformers.modeling_outputs import SequenceClassifierOutput

from nlp_fin.constants import BIOES_LABELS, RISK_CLASSES
from nlp_fin.training.inference import ReportAnalyzer


class _FakeModel:
    """Модель с фиксированной таблицей лог-итов по позиции токена."""

    def __init__(self, table: list[int], num_classes: int) -> None:
        self._table = table
        self._num_classes = num_classes

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> SequenceClassifierOutput:
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, self._num_classes)
        for index in range(length):
            logits[:, index, self._table[index]] = 4.0
        return SequenceClassifierOutput(logits=logits)

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> SequenceClassifierOutput:
        return self.forward(input_ids, attention_mask)

    def to(self, device: torch.device) -> _FakeModel:
        return self

    def eval(self) -> _FakeModel:
        return self


class _FakeRisk:
    """Модель риска с детерминированной распределённой логики по классам."""

    def __init__(self, class_id: int, num_classes: int) -> None:
        self._class_id = class_id
        self._num_classes = num_classes

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> SequenceClassifierOutput:
        logits = torch.zeros(input_ids.shape[0], self._num_classes)
        logits[:, self._class_id] = 4.0
        return SequenceClassifierOutput(logits=logits)

    def to(self, device: torch.device) -> _FakeRisk:
        return self

    def eval(self) -> _FakeRisk:
        return self


def _analyzer(tmp_path: Path, ner_table: list[int]) -> ReportAnalyzer:
    tokenizer = make_tokenizer(tmp_path, max_length=128)
    ner_dir, risk_dir = save_tiny_models(tmp_path, tokenizer)
    analyzer = ReportAnalyzer(str(ner_dir), str(risk_dir))
    analyzer._ner = _FakeModel(ner_table, len(BIOES_LABELS))
    analyzer._risk = _FakeRisk(RISK_CLASSES.index("high"), len(RISK_CLASSES))
    return analyzer


def test_report_analyzer_ner_on_text(tmp_path: Path) -> None:
    amount_id = BIOES_LABELS.index("S-amount")
    analyzer = _analyzer(tmp_path, [amount_id] * 64)
    spans = analyzer.ner_on_text("выручка 5")
    assert [(span["text"], span["label"]) for span in spans] == [
        ("выручка", "amount"),
        ("5", "amount"),
    ]
    assert all(span["confidence"] > 0.5 for span in spans)


def test_report_analyzer_risk_on_text(tmp_path: Path) -> None:
    amount_id = BIOES_LABELS.index("S-amount")
    analyzer = _analyzer(tmp_path, [amount_id] * 64)
    result = analyzer.risk_on_text("выручка 5")
    assert result["risk"] == "high"
    assert result["probabilities"]["high"] > 0.5
    assert set(result["probabilities"]) == set(RISK_CLASSES)


def test_report_analyzer_analyze_combines(tmp_path: Path) -> None:
    amount_id = BIOES_LABELS.index("S-amount")
    analyzer = _analyzer(tmp_path, [amount_id] * 64)
    result = analyzer.analyze("выручка 5")
    assert result["ner"]
    assert result["risk"]["risk"] == "high"
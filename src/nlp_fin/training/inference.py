"""Инференс NER и риска в одном процессе на общей базе BERT."""

from __future__ import annotations

from typing import Any

import torch
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    BertForTokenClassification,
)

from nlp_fin.constants import BIOES_LABELS, RISK_CLASSES


class ReportAnalyzer:
    """Анализирует текст отчёта: NER-сущности и класс риска."""

    def __init__(
        self,
        ner_model_path: str,
        risk_model_path: str,
        tokenizer_path: str | None = None,
        *,
        device: str = "cpu",
        quantize: bool = False,
    ) -> None:
        tokenizer_path = tokenizer_path or ner_model_path
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self._device = torch.device(device)
        self._ner = BertForTokenClassification.from_pretrained(ner_model_path)
        self._risk = BertForSequenceClassification.from_pretrained(risk_model_path)
        self._ner.to(self._device).eval()
        self._risk.to(self._device).eval()
        if quantize:
            self._quantize()

    def _quantize(self) -> None:
        if self._device.type != "cpu":
            return
        for attribute in ("_ner", "_risk"):
            model = getattr(self, attribute)
            try:
                quantized = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
                quantized.eval()
                setattr(self, attribute, quantized)
            except RuntimeError:
                continue

    def ner_on_text(self, text: str) -> list[dict[str, Any]]:
        """Извлекает NER-сущности списком словарей с текстом и меткой."""
        encoded = self._tokenizer(text, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self._device)
        attention_mask = encoded["attention_mask"].to(self._device)
        with torch.no_grad():
            logits = self._ner(input_ids=input_ids, attention_mask=attention_mask).logits
        probabilities = torch.softmax(logits, dim=-1)[0]
        predictions = logits.argmax(dim=-1)[0].tolist()

        word_ids = encoded.word_ids()
        raw_ids = input_ids[0].tolist()
        groups: dict[int, int] = {}
        for index, word_id in enumerate(word_ids):
            if word_id is not None and word_id not in groups:
                groups[word_id] = index
        ordered = sorted(groups)

        def word_tokens(word_id: int) -> list[int]:
            return [raw_ids[i] for i, wid in enumerate(word_ids) if wid == word_id]

        word_texts = [
            self._tokenizer.decode(word_tokens(word_id), skip_special_tokens=True)
            for word_id in ordered
        ]
        word_labels = [BIOES_LABELS[predictions[groups[word_id]]] for word_id in ordered]
        word_probs = [
            float(probabilities[groups[word_id]][predictions[groups[word_id]]].item())
            for word_id in ordered
        ]
        return self._build_spans(word_texts, word_labels, word_probs)

    @staticmethod
    def _span(
        word_texts: list[str], start: int, end: int, entity: str, confidence: float
    ) -> dict[str, Any]:
        return {
            "text": " ".join(word_texts[start:end]),
            "label": entity,
            "confidence": confidence,
        }

    def _build_spans(
        self, word_texts: list[str], word_labels: list[str], word_probs: list[float]
    ) -> list[dict[str, Any]]:
        spans: list[dict[str, Any]] = []
        start_index: int | None = None
        current_entity: str | None = None
        current_prob = 0.0
        for index, label in enumerate(word_labels):
            edge, _, entity = label.partition("-")
            if label == "O":
                if current_entity is not None and start_index is not None:
                    spans.append(
                        self._span(word_texts, start_index, index, current_entity, current_prob)
                    )
                    current_entity = None
                start_index = None
                continue
            if edge == "S":
                if current_entity is not None and start_index is not None:
                    spans.append(
                        self._span(word_texts, start_index, index, current_entity, current_prob)
                    )
                spans.append(
                    {"text": word_texts[index], "label": entity, "confidence": word_probs[index]}
                )
                current_entity = None
                start_index = None
            elif edge == "B":
                if current_entity is not None and start_index is not None:
                    spans.append(
                        self._span(word_texts, start_index, index, current_entity, current_prob)
                    )
                start_index = index
                current_entity = entity
                current_prob = word_probs[index]
            elif edge in ("I", "E"):
                if current_entity is None:
                    start_index = index
                    current_entity = entity
                    current_prob = word_probs[index]
                if edge == "E" and start_index is not None:
                    spans.append(
                        self._span(
                            word_texts, start_index, index + 1, current_entity, current_prob
                        )
                    )
                    current_entity = None
                    start_index = None
        if current_entity is not None and start_index is not None:
            spans.append(
                self._span(word_texts, start_index, len(word_texts), current_entity, current_prob)
            )
        return spans

    def risk_on_text(self, text: str) -> dict[str, Any]:
        encoded = self._tokenizer(text, max_length=256, truncation=True, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self._device)
        attention_mask = encoded["attention_mask"].to(self._device)
        with torch.no_grad():
            logits = self._risk(input_ids=input_ids, attention_mask=attention_mask).logits
        probabilities = torch.softmax(logits, dim=-1)[0]
        predicted = int(probabilities.argmax().item())
        return {
            "risk": RISK_CLASSES[predicted],
            "probabilities": {
                RISK_CLASSES[index]: float(value.item())
                for index, value in enumerate(probabilities)
            },
        }

    def analyze(self, text: str) -> dict[str, Any]:
        """Совместный инференс: NER и риск в одном процессе."""
        return {"ner": self.ner_on_text(text), "risk": self.risk_on_text(text)}
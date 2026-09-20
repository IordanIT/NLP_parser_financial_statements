"""Токенизация размеченных чанков в входы для обучения NER."""

from __future__ import annotations

from transformers import PreTrainedTokenizerBase

from nlp_fin.constants import BIOES_LABELS
from nlp_fin.dataset.models import AnnotatedChunk
from nlp_fin.dataset.tags import bioes_labels, split_words

_IGNORE_INDEX = -100


def _continuation_tag(word_label: str) -> str:
    prefix, sep, entity = word_label.partition("-")
    if sep and prefix in ("B", "E", "S"):
        return f"I-{entity}"
    return word_label


class NerEncoder:
    """Преобразует чанк с сущностями в input_ids, attention_mask и labels."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
        label_set: tuple[str, ...] = BIOES_LABELS,
    ) -> None:
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._label_to_id: dict[str, int] = {
            label: index for index, label in enumerate(label_set)
        }

    def encode_chunk(self, chunk: AnnotatedChunk) -> dict[str, list[int]]:
        words = split_words(chunk.text)
        word_labels = bioes_labels(chunk.spans, words)
        encoded = self._tokenizer(
            chunk.text,
            add_special_tokens=True,
            truncation=True,
            max_length=self._max_length,
            padding="max_length",
        )
        labels: list[int] = []
        piece_counts: dict[int, int] = {}
        for word_id in encoded.word_ids():
            if word_id is None:
                labels.append(_IGNORE_INDEX)
                continue
            piece_index = piece_counts.get(word_id, 0)
            piece_counts[word_id] = piece_index + 1
            word_label = word_labels[word_id]
            tag = word_label if piece_index == 0 else _continuation_tag(word_label)
            labels.append(self._label_to_id[tag])
        return {
            "input_ids": list(encoded["input_ids"]),
            "attention_mask": list(encoded["attention_mask"]),
            "labels": labels,
        }
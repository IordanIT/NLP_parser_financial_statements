"""Тесты BIOES-тегирования и разбиения на слова."""

import pytest

from nlp_fin.constants import METRIC_AMOUNT, METRIC_CURRENCY, METRIC_NAME
from nlp_fin.dataset.models import AnnotatedChunk, NERSpan
from nlp_fin.dataset.tags import (
    bioes_labels,
    spans_from_bioes,
    split_words,
)

TEXT = "чистая прибыль 1 234 567 руб."


def test_split_words_offsets() -> None:
    words = split_words(TEXT)
    assert [word.text for word in words] == ["чистая", "прибыль", "1", "234", "567", "руб."]
    assert [word.start for word in words] == [0, 7, 15, 17, 21, 25]
    assert all(word.end > word.start for word in words)


def test_bioes_labels_multiword_spans() -> None:
    chunk = AnnotatedChunk(
        chunk_id=0,
        text=TEXT,
        spans=[
            NERSpan(text="чистая прибыль", label=METRIC_NAME, start=0, end=14),
            NERSpan(text="1 234 567", label=METRIC_AMOUNT, start=15, end=24),
            NERSpan(text="руб.", label=METRIC_CURRENCY, start=25, end=29),
        ],
    )
    words = split_words(TEXT)
    labels = bioes_labels(chunk.spans, words)
    assert labels == [
        "B-metric",
        "E-metric",
        "B-amount",
        "I-amount",
        "E-amount",
        "S-currency",
    ]


def test_single_word_span_is_s() -> None:
    words = split_words("выручка")
    labels = bioes_labels([NERSpan(text="выручка", label=METRIC_NAME, start=0, end=7)], words)
    assert labels == ["S-metric"]


def test_roundtrip_spans() -> None:
    chunk = AnnotatedChunk(
        chunk_id=0,
        text=TEXT,
        spans=[
            NERSpan(text="чистая прибыль", label=METRIC_NAME, start=0, end=14),
            NERSpan(text="1 234 567", label=METRIC_AMOUNT, start=15, end=24),
            NERSpan(text="руб.", label=METRIC_CURRENCY, start=25, end=29),
        ],
    )
    words = split_words(TEXT)
    restored = spans_from_bioes(bioes_labels(chunk.spans, words), words)
    assert restored == chunk.spans


def test_span_overlap_skipped() -> None:
    words = split_words("выручка 100")
    spans = [NERSpan(text="выручка", label=METRIC_NAME, start=0, end=7)]
    labels = bioes_labels(spans, words)
    assert labels == ["S-metric", "O"]


def test_chunk_validates_span_alignment() -> None:
    with pytest.raises(ValueError):
        AnnotatedChunk(
            chunk_id=0,
            text="выручка",
            spans=[NERSpan(text="зарплата", label=METRIC_NAME, start=0, end=7)],
        )
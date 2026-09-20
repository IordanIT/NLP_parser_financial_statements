"""Тесты кодирования чанков в NER-входы (токены, штрайф-метки, -100)."""

from transformers import BertTokenizerFast

from nlp_fin.constants import (
    BIOES_LABELS,
    METRIC_AMOUNT,
    METRIC_NAME,
)
from nlp_fin.dataset.encode import NerEncoder
from nlp_fin.dataset.models import AnnotatedChunk, NERSpan

_VOCAB: list[str] = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    "вы",
    "##руч",
    "##ка",
    "чис",
    "##тая",
    "при",
    "##быль",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]

_LABEL_IDS = {label: index for index, label in enumerate(BIOES_LABELS)}


def _make_tokenizer(tmp_path) -> BertTokenizerFast:
    vocab_file = tmp_path / "vocab.txt"
    vocab_file.write_text("\n".join(_VOCAB) + "\n", encoding="utf-8")
    return BertTokenizerFast(
        vocab_file=str(vocab_file), do_lower_case=True, model_max_length=128
    )


def test_encode_single_word_span_with_pieces(tmp_path) -> None:
    encoder = NerEncoder(_make_tokenizer(tmp_path), max_length=16)
    chunk = AnnotatedChunk(
        chunk_id=0,
        text="прибыль 5",
        spans=[
            NERSpan(text="прибыль", label=METRIC_NAME, start=0, end=7),
            NERSpan(text="5", label=METRIC_AMOUNT, start=8, end=9),
        ],
    )
    encoded = encoder.encode_chunk(chunk)
    assert len(encoded["input_ids"]) == 16
    assert len(encoded["labels"]) == 16
    assert encoded["attention_mask"][:5] == [1, 1, 1, 1, 1]
    labels = encoded["labels"]
    assert labels[0] == -100
    assert labels[1] == _LABEL_IDS["S-metric"]
    assert labels[2] == _LABEL_IDS["I-metric"]
    assert labels[3] == _LABEL_IDS["S-amount"]
    assert labels[4] == -100
    assert all(label == -100 for label in labels[5:])


def test_encode_multiword_span_and_pieces(tmp_path) -> None:
    encoder = NerEncoder(_make_tokenizer(tmp_path), max_length=16)
    chunk = AnnotatedChunk(
        chunk_id=0,
        text="чистая прибыль 5",
        spans=[
            NERSpan(text="чистая прибыль", label=METRIC_NAME, start=0, end=14),
            NERSpan(text="5", label=METRIC_AMOUNT, start=15, end=16),
        ],
    )
    labels = encoder.encode_chunk(chunk)["labels"]
    assert labels[0] == -100
    assert labels[1] == _LABEL_IDS["B-metric"]
    assert labels[2] == _LABEL_IDS["I-metric"]
    assert labels[3] == _LABEL_IDS["E-metric"]
    assert labels[4] == _LABEL_IDS["I-metric"]
    assert labels[5] == _LABEL_IDS["S-amount"]
    assert labels[6] == -100


def test_truncation_caps_tokens(tmp_path) -> None:
    encoder = NerEncoder(_make_tokenizer(tmp_path), max_length=8)
    chunk = AnnotatedChunk(
        chunk_id=0,
        text="прибыль 5 6 7 8 9 0 1 2 3",
        spans=[
            NERSpan(text="прибыль", label=METRIC_NAME, start=0, end=7),
            NERSpan(text="5", label=METRIC_AMOUNT, start=8, end=9),
        ],
    )
    encoded = encoder.encode_chunk(chunk)
    assert len(encoded["input_ids"]) == 8
    assert len(encoded["labels"]) == 8
    assert sum(encoded["attention_mask"]) == 8
    assert encoded["labels"][0] == -100
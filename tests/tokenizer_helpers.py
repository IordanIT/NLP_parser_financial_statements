"""Помощники оффлайн-тестов: крошечный словарь BERT и записи разметки."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from transformers import BertConfig, BertTokenizerFast

from nlp_fin.constants import BIOES_LABELS
from nlp_fin.dataset.models import AnnotatedChunk, AnnotationRecord, EntityLabel, NERSpan

VOCAB: list[str] = [
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

LABEL_IDS: dict[str, int] = {label: index for index, label in enumerate(BIOES_LABELS)}


def make_tokenizer(tmp_path: Path, max_length: int = 128) -> BertTokenizerFast:
    vocab_file = tmp_path / "vocab.txt"
    vocab_file.write_text("\n".join(VOCAB) + "\n", encoding="utf-8")
    return BertTokenizerFast(
        vocab_file=str(vocab_file), do_lower_case=True, model_max_length=max_length
    )


def tiny_config(num_labels: int) -> BertConfig:
    return BertConfig(
        vocab_size=len(VOCAB),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        type_vocab_size=2,
        num_labels=num_labels,
    )


def make_records(
    pairs: list[tuple[str, list[tuple[str, str]]]]
) -> list[AnnotationRecord]:
    """Собирает записи из (текст, [(сущность, метка), ...]) с подсчётом смещений."""
    records: list[AnnotationRecord] = []
    for index, (text, entities) in enumerate(pairs):
        spans: list[NERSpan] = []
        search_from = 0
        for entity, label in entities:
            start = text.find(entity, search_from)
            if start == -1:
                raise ValueError(f"Сущность {entity!r} не найдена в тексте")
            spans.append(
                NERSpan(
                    text=entity,
                    label=cast(EntityLabel, label),
                    start=start,
                    end=start + len(entity),
                )
            )
            search_from = start + len(entity)
        records.append(
            AnnotationRecord(
                source=f"test/{index}",
                chunks=[AnnotatedChunk(chunk_id=0, text=text, spans=spans)],
            )
        )
    return records


def write_records(tmp_path: Path, records: list[AnnotationRecord]) -> Path:
    path = tmp_path / "annotations.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(record.model_dump_json() for record in records) + "\n", encoding="utf-8"
    )
    return path


def dump(records: list[AnnotationRecord]) -> list[dict[str, Any]]:
    return [record.model_dump() for record in records]


def assert_json_roundtrip(records: list[AnnotationRecord]) -> None:
    restored = json.loads("\n".join(record.model_dump_json() for record in records))
    assert restored == [record.model_dump() for record in records]


def save_tiny_models(tmp_path: Path, tokenizer: BertTokenizerFast) -> tuple[Path, Path]:
    from nlp_fin.constants import BIOES_LABELS, RISK_CLASSES
    from nlp_fin.training.build import build_ner_model, build_risk_model

    ner_dir = tmp_path / "models" / "ner"
    risk_dir = tmp_path / "models" / "risk"
    build_ner_model(tiny_config(len(BIOES_LABELS)), len(BIOES_LABELS)).save_pretrained(
        str(ner_dir)
    )
    build_risk_model(
        tiny_config(len(RISK_CLASSES)), len(RISK_CLASSES), dropout=0.3
    ).save_pretrained(str(risk_dir))
    tokenizer.save_pretrained(str(ner_dir))
    return ner_dir, risk_dir
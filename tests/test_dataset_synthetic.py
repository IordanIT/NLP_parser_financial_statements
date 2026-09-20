"""Тесты генератора синтетической разметки."""

import random

from nlp_fin.dataset.models import AnnotationRecord
from nlp_fin.dataset.synthetic import SyntheticGenerator, SyntheticOptions, _ocr_noise


def test_generate_count_and_min_entities() -> None:
    options = SyntheticOptions(count=3, min_entities=5, max_entities=12, seed=7)
    records = SyntheticGenerator().generate(options)
    assert len(records) == 3
    for record in records:
        entities = sum(len(chunk.spans) for chunk in record.chunks)
        assert entities >= options.min_entities
        assert record.source.startswith("synthetic/")


def test_span_integrity_after_ocr_noise() -> None:
    options = SyntheticOptions(count=2, min_entities=6, max_entities=10, noise_level=0.5, seed=3)
    records = SyntheticGenerator().generate(options)
    for record in records:
        for chunk in record.chunks:
            assert all(
                chunk.text[span.start : span.end] == span.text for span in chunk.spans
            )
            assert all(span.start < span.end for span in chunk.spans)


def test_deterministic_seed() -> None:
    options = SyntheticOptions(count=2, min_entities=6, max_entities=10, noise_level=0.1, seed=11)
    first = SyntheticGenerator().generate(options)
    second = SyntheticGenerator().generate(options)
    assert [record.model_dump() for record in first] == [
        record.model_dump() for record in second
    ]


def test_jsonl_roundtrip() -> None:
    options = SyntheticOptions(count=1, min_entities=4, max_entities=8, seed=1)
    record = SyntheticGenerator().generate(options)[0]
    restored = AnnotationRecord.model_validate_json(record.model_dump_json())
    assert restored == record


def test_ocr_noise_changes_confusable_chars() -> None:
    rng = random.Random(1)
    noised = _ocr_noise("выручка составила 3 000 000 руб.", rng, 1.0)
    assert noised != "выручка составила 3 000 000 руб."
    assert len(noised) == len("выручка составила 3 000 000 руб.")
"""Тесты правиловой предразметки NER."""

from nlp_fin.constants import (
    METRIC_AMOUNT,
    METRIC_CURRENCY,
    METRIC_NAME,
    METRIC_PERIOD,
    METRIC_UNIT,
)
from nlp_fin.dataset.baseline import RuleBasedAnnotator


def test_annotate_full_sentence() -> None:
    annotator = RuleBasedAnnotator()
    spans = annotator.annotate("По итогам 2023 года выручка составила 1 234 567 тыс. руб.")
    labels = {span.label for span in spans}
    assert labels == {
        METRIC_NAME,
        METRIC_AMOUNT,
        METRIC_PERIOD,
        METRIC_CURRENCY,
        METRIC_UNIT,
    }
    by_label = {span.label: span.text for span in spans}
    assert by_label[METRIC_NAME] == "выручка"
    assert by_label[METRIC_AMOUNT] == "1 234 567"
    assert by_label[METRIC_PERIOD] == "2023 года"
    assert by_label[METRIC_UNIT] == "тыс."
    assert by_label[METRIC_CURRENCY] == "руб."


def test_longest_metric_wins() -> None:
    annotator = RuleBasedAnnotator()
    spans = annotator.annotate("чистая прибыль компании составила 100 руб.")
    metric = [span for span in spans if span.label == METRIC_NAME]
    assert len(metric) == 1
    assert metric[0].text == "чистая прибыль"
    assert metric[0].start == 0


def test_latin_metric_and_decimal_amount() -> None:
    annotator = RuleBasedAnnotator()
    spans = annotator.annotate("EBITDA 12 500,00 млн руб.")
    by_label = {span.label: span.text for span in spans}
    assert by_label[METRIC_NAME] == "EBITDA"
    assert by_label[METRIC_AMOUNT] == "12 500,00"
    assert by_label[METRIC_UNIT] == "млн"
    assert by_label[METRIC_CURRENCY] == "руб."


def test_spans_sorted_and_non_overlapping() -> None:
    annotator = RuleBasedAnnotator()
    text = "За 2023 год выручка 1 234 567 руб., рентабельность продаж 12,5 %."
    spans = annotator.annotate(text)
    assert [span.start for span in spans] == sorted(span.start for span in spans)
    for index, span in enumerate(spans):
        for other in spans[:index]:
            assert span.start >= other.end or other.start >= span.end
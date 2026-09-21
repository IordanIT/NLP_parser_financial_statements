"""Метрики NER (seqeval) и оценки модели риска."""

from __future__ import annotations

from typing import Any

from seqeval.metrics import classification_report


def _to_plain(value: Any) -> Any:
    """Рекурсивно приводит numpy-скаляры отчётseqeval к простым типам Python."""
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    return value


def ner_metrics(
    pred_ids: list[list[int]],
    gold_ids: list[list[int]],
    id_to_label: dict[int, str],
    ignore_index: int = -100,
) -> dict[str, Any]:
    """seqeval-отчёт NER: агрегат и разбивка по сущностям."""
    gold_lists: list[list[str]] = []
    pred_lists: list[list[str]] = []
    for gold_seq, pred_seq in zip(gold_ids, pred_ids, strict=True):
        gold_words: list[str] = []
        pred_words: list[str] = []
        for gold_id, pred_id in zip(gold_seq, pred_seq, strict=True):
            if gold_id == ignore_index:
                continue
            gold_words.append(id_to_label[gold_id])
            pred_words.append(id_to_label[pred_id])
        gold_lists.append(gold_words)
        pred_lists.append(pred_words)
    return _to_plain(
        classification_report(
            gold_lists, pred_lists, digits=4, output_dict=True, zero_division=0
        )
    )


def confusion_matrix(
    true_ids: list[int], pred_ids: list[int], num_classes: int
) -> list[list[int]]:
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for true_id, pred_id in zip(true_ids, pred_ids, strict=True):
        matrix[true_id][pred_id] += 1
    return matrix


def macro_f1(
    true_ids: list[int], pred_ids: list[int], num_classes: int
) -> dict[str, float]:
    precision_sum = recall_sum = f1_sum = 0.0
    for cls in range(num_classes):
        tp = sum(1 for t, p in zip(true_ids, pred_ids, strict=True) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(true_ids, pred_ids, strict=True) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(true_ids, pred_ids, strict=True) if t == cls and p != cls)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precision_sum += precision
        recall_sum += recall
        f1_sum += f1
    return {
        "macro_precision": precision_sum / num_classes,
        "macro_recall": recall_sum / num_classes,
        "macro_f1": f1_sum / num_classes,
    }


def brier_multiclass(
    true_ids: list[int], probabilities: list[list[float]]
) -> float:
    """Мультиклассовый Brier score: средний квадрат ошибки вероятностей."""
    total = 0.0
    rows = 0
    for true_id, probs in zip(true_ids, probabilities, strict=True):
        total += sum(
            (prob - (1.0 if cls == true_id else 0.0)) ** 2
            for cls, prob in enumerate(probs)
        )
        rows += 1
    return total / rows if rows else 0.0


def expected_calibration_error(
    true_ids: list[int],
    probabilities: list[list[float]],
    bins: int = 10,
) -> float:
    """ECE: среднее отклонение уверенности от точности по бинам уверенности."""
    if not probabilities:
        return 0.0
    bin_accuracy = [0.0] * bins
    bin_confidence = [0.0] * bins
    bin_count = [0] * bins
    for true_id, probs in zip(true_ids, probabilities, strict=True):
        conf = max(probs)
        predicted = int(max(range(len(probs)), key=lambda i: probs[i]))
        index = min(int(conf * bins), bins - 1)
        bin_count[index] += 1
        bin_accuracy[index] += 1.0 if predicted == true_id else 0.0
        bin_confidence[index] += conf
    total = len(true_ids)
    error = 0.0
    for index in range(bins):
        if bin_count[index]:
            accuracy = bin_accuracy[index] / bin_count[index]
            confidence = bin_confidence[index] / bin_count[index]
            error += (bin_count[index] / total) * abs(accuracy - confidence)
    return error


def risk_metrics(
    true_ids: list[int],
    pred_ids: list[int],
    probabilities: list[list[float]],
    num_classes: int,
    bins: int = 10,
) -> dict[str, Any]:
    """Полный отчёт оценки риск-модели."""
    macro = macro_f1(true_ids, pred_ids, num_classes)
    correct = sum(1 for t, p in zip(true_ids, pred_ids, strict=True) if t == p)
    return {
        "n": len(true_ids),
        "accuracy": correct / len(true_ids) if true_ids else 0.0,
        **macro,
        "brier": brier_multiclass(true_ids, probabilities),
        "ece": expected_calibration_error(true_ids, probabilities, bins),
        "confusion_matrix": confusion_matrix(true_ids, pred_ids, num_classes),
    }
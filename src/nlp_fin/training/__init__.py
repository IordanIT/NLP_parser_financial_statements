"""Обучение и инференс NER и риск-моделей на базе BERT."""

from nlp_fin.training.build import build_ner_model, build_risk_model
from nlp_fin.training.data import load_records, ner_dataset, risk_dataset, split_records
from nlp_fin.training.inference import ReportAnalyzer
from nlp_fin.training.loop import TrainConfig, Trainer, TrainOutcome
from nlp_fin.training.metrics import brier_multiclass, expected_calibration_error, macro_f1
from nlp_fin.training.run import evaluate_ner, evaluate_risk, train_ner, train_risk
from nlp_fin.training.weights import class_weights

__all__ = [
    "TrainConfig",
    "TrainOutcome",
    "Trainer",
    "ReportAnalyzer",
    "brier_multiclass",
    "build_ner_model",
    "build_risk_model",
    "class_weights",
    "expected_calibration_error",
    "evaluate_ner",
    "evaluate_risk",
    "load_records",
    "macro_f1",
    "ner_dataset",
    "risk_dataset",
    "split_records",
    "train_ner",
    "train_risk",
]
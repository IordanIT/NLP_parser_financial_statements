"""CLI-оценка обученных NER и риск-моделей на размеченном JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from nlp_fin.config import get_settings
from nlp_fin.constants import BIOES_LABELS, RISK_CLASSES
from nlp_fin.training.run import evaluate_ner, evaluate_risk


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Оценка NER и риск-моделей")
    parser.add_argument(
        "--data", type=Path, default=settings.annotations_dir / "synthetic.jsonl"
    )
    parser.add_argument("--models-dir", type=Path, default=settings.models_dir)
    parser.add_argument("--tokenizer", type=str, default=settings.ner_tokenizer)
    parser.add_argument("--ner-max-length", type=int, default=settings.ner_max_length)
    parser.add_argument("--risk-max-length", type=int, default=settings.risk_max_length)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    ner_report = evaluate_ner(
        args.data,
        args.models_dir / "ner",
        tokenizer=tokenizer,
        num_labels=len(BIOES_LABELS),
        max_length=args.ner_max_length,
    )
    risk_report = evaluate_risk(
        args.data,
        args.models_dir / "risk",
        tokenizer=tokenizer,
        num_labels=len(RISK_CLASSES),
        max_length=args.risk_max_length,
    )
    ner_path = args.models_dir / "eval_NER.json"
    risk_path = args.models_dir / "eval_risk.json"
    ner_path.write_text(
        json.dumps(ner_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    risk_path.write_text(
        json.dumps(risk_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
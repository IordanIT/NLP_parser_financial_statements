"""CLI-обучение NER-модели на размеченном JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path

from nlp_fin.config import get_settings
from nlp_fin.constants import BIOES_LABELS
from nlp_fin.training.data import load_records, split_records
from nlp_fin.training.loop import TrainConfig
from nlp_fin.training.run import train_ner


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Обучение NER-модели")
    parser.add_argument(
        "--data", type=Path, default=settings.annotations_dir / "synthetic.jsonl"
    )
    parser.add_argument("--output", type=Path, default=settings.models_dir / "ner")
    parser.add_argument("--epochs", type=int, default=settings.ner_epochs)
    parser.add_argument("--batch-size", type=int, default=settings.ner_batch_size)
    parser.add_argument(
        "--grad-accumulation", type=int, default=settings.ner_grad_accumulation
    )
    parser.add_argument("--learning-rate", type=float, default=settings.ner_learning_rate)
    parser.add_argument("--warmup-ratio", type=float, default=settings.ner_warmup_ratio)
    parser.add_argument("--patience", type=int, default=settings.ner_patience)
    parser.add_argument("--max-length", type=int, default=settings.ner_max_length)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    records = load_records(args.data)
    train_records, val_records = split_records(records, args.val_ratio, args.seed)
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        grad_accumulation=args.grad_accumulation,
        warmup_ratio=args.warmup_ratio,
        patience=args.patience,
    )
    train_ner(
        train_records,
        val_records,
        args.output,
        model_name=settings.ner_tokenizer,
        num_labels=len(BIOES_LABELS),
        config=config,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
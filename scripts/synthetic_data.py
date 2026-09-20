"""CLI-генерация синтетической NER-разметки финансовых отчётов."""

from __future__ import annotations

import argparse
from pathlib import Path

from nlp_fin.dataset.synthetic import SyntheticGenerator, SyntheticOptions


def main() -> None:
    parser = argparse.ArgumentParser(description="Генерация синтетической разметки")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--min-entities", type=int, default=20)
    parser.add_argument("--max-entities", type=int, default=80)
    parser.add_argument("--noise-level", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/annotations/synthetic.jsonl"))
    args = parser.parse_args()

    options = SyntheticOptions(
        count=args.count,
        min_entities=args.min_entities,
        max_entities=args.max_entities,
        noise_level=args.noise_level,
        seed=args.seed,
    )
    records = SyntheticGenerator().generate(options)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(record.model_dump_json() for record in records) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
"""Обёртка над Label Studio: загрузка задач с предразметкой и экспорт разметки."""

from __future__ import annotations

import argparse
from pathlib import Path

from nlp_fin.dataset.baseline import RuleBasedAnnotator
from nlp_fin.dataset.labelstudio import LabelStudioClient, build_task_payloads
from nlp_fin.dataset.models import AnnotationRecord


def _read_records(path: Path) -> list[AnnotationRecord]:
    return [
        AnnotationRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Работа с разметкой в Label Studio")
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--project-id", required=True, type=int)
    commands = parser.add_subparsers(dest="command", required=True)
    import_command = commands.add_parser("import", help="Загрузить задачи с предразметкой")
    import_command.add_argument("--input", required=True, type=Path)
    export_command = commands.add_parser("export", help="Выгрузить разметку в JSONL")
    export_command.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    client = LabelStudioClient(args.url, args.api_key)
    if args.command == "import":
        annotator = RuleBasedAnnotator()
        for record in _read_records(args.input):
            for payload in build_task_payloads(record, annotator):
                client.create_task(payload)
    else:
        records = client.export_project(args.project_id)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "\n".join(
                record.model_dump_json() for record in records
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
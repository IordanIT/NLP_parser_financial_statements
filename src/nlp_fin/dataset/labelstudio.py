"""Клиент Label Studio: загрузка задач с предразметкой и экспорт разметки."""

from __future__ import annotations

import json
from typing import Any, cast
from urllib.request import Request, urlopen

from nlp_fin.dataset.baseline import RuleBasedAnnotator
from nlp_fin.dataset.models import (
    AnnotatedChunk,
    AnnotationRecord,
    EntityLabel,
    NERSpan,
)


def build_task_payloads(
    record: AnnotationRecord, annotator: RuleBasedAnnotator
) -> list[dict[str, Any]]:
    """Создаёт задачи Label Studio: по одной на чанк, с baseline-предразметкой."""
    payloads: list[dict[str, Any]] = []
    for chunk in record.chunks:
        payloads.append(
            {
                "data": {
                    "text": chunk.text,
                    "source": record.source,
                    "chunk_id": chunk.chunk_id,
                },
                "predictions": [
                    {
                        "model_version": "baseline",
                        "result": [
                            {
                                "from_name": "entities",
                                "to_name": "text",
                                "type": "labels",
                                "value": {
                                    "start": span.start,
                                    "end": span.end,
                                    "text": span.text,
                                    "labels": [span.label],
                                },
                            }
                            for span in annotator.annotate(chunk.text)
                        ],
                    }
                ],
            }
        )
    return payloads


def parse_export(export_payload: list[dict[str, Any]]) -> list[AnnotationRecord]:
    """Разбирает экспорт Label Studio в записи AnnotationRecord."""
    grouped: dict[str, list[AnnotatedChunk]] = {}
    for task in export_payload:
        data = task.get("data", {})
        source = str(data.get("source", "unknown"))
        chunk_id = int(data.get("chunk_id", 0))
        text = str(data.get("text", ""))
        spans: list[NERSpan] = []
        for annotation in task.get("annotations") or []:
            for result in annotation.get("result", []):
                value = result.get("value", {})
                labels = value.get("labels") or []
                if not labels:
                    continue
                spans.append(
                    NERSpan(
                        text=str(value.get("text", "")),
                        label=cast(EntityLabel, str(labels[0])),
                        start=int(value.get("start", 0)),
                        end=int(value.get("end", 0)),
                    )
                )
        grouped.setdefault(source, []).append(
            AnnotatedChunk(chunk_id=chunk_id, text=text, spans=spans)
        )
    return [
        AnnotationRecord(
            source=source, chunks=sorted(chunks, key=lambda chunk: chunk.chunk_id)
        )
        for source, chunks in grouped.items()
    ]


class LabelStudioClient:
    """Тонкий REST-клиент Label Studio."""

    def __init__(self, url: str, api_key: str) -> None:
        self._base_url = url.rstrip("/")
        self._api_key = api_key

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None
    ) -> Any:
        request = Request(
            f"{self._base_url}{path}",
            data=json.dumps(payload).encode("utf-8") if payload is not None else None,
            headers={
                "Authorization": f"Token {self._api_key}",
                "Content-Type": "application/json",
            },
            method=method,
        )
        with urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))

    def create_task(self, payload: dict[str, Any]) -> None:
        """Создаёт задачу с предразметкой."""
        self._request("POST", "/api/tasks", payload)

    def export_project(self, project_id: int) -> list[AnnotationRecord]:
        """Выгружает разметку проекта в записи AnnotationRecord."""
        payload = self._request(
            "GET", f"/api/projects/{project_id}/export?exportType=JSON", None
        )
        return parse_export(payload)
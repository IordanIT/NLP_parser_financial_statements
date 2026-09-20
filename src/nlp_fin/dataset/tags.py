"""Слова в тексте и BIOES-тегирование по спанам."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import cast

from nlp_fin.constants import NON_ENTITY
from nlp_fin.dataset.models import EntityLabel, NERSpan

_WORD_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class CharWord:
    """Слово в тексте чанка с границами в символах."""

    text: str
    start: int
    end: int


def split_words(text: str) -> list[CharWord]:
    """Разбивает текст на слова с позициями в символах."""
    return [
        CharWord(text=match.group(0), start=match.start(), end=match.end())
        for match in _WORD_RE.finditer(text)
    ]


def _word_covered(word: CharWord, span: NERSpan) -> bool:
    center = word.start + word.end
    return span.start * 2 <= center < span.end * 2


def bioes_labels(spans: list[NERSpan], words: list[CharWord]) -> list[str]:
    """Возвращает BIOES-метку для каждого слова чанка."""
    labels = [NON_ENTITY] * len(words)
    for span in spans:
        covered = [index for index, word in enumerate(words) if _word_covered(word, span)]
        if not covered:
            continue
        for position, index in enumerate(reversed(covered)):
            if position == len(covered) - 1:
                prefix = "B" if len(covered) > 1 else "S"
            elif position == 0:
                prefix = "E"
            else:
                prefix = "I"
            labels[index] = f"{prefix}-{span.label}"
    return labels


def spans_from_bioes(labels: list[str], words: list[CharWord]) -> list[NERSpan]:
    """Восстанавливает спаны из BIOES-меток слов."""
    spans: list[NERSpan] = []
    start: int | None = None
    label: EntityLabel | None = None
    for index, word_title in enumerate(labels):
        if word_title == NON_ENTITY:
            if start is not None and label is not None:
                spans.append(
                    NERSpan(
                        text=" ".join(w.text for w in words[start:index]),
                        label=label,
                        start=words[start].start,
                        end=words[index - 1].end,
                    )
                )
            start, label = None, None
            continue
        edge, _, entity = word_title.partition("-")
        entity_label = cast(EntityLabel, entity)
        if start is None:
            start = index
            label = entity_label
        elif edge == "B" or edge == "S":
            spans.append(
                NERSpan(
                    text=" ".join(w.text for w in words[start:index]),
                    label=label,
                    start=words[start].start,
                    end=words[index - 1].end,
                )
            )
            start, label = index, entity_label
        elif edge == "E" and label is not None:
            spans.append(
                NERSpan(
                    text=" ".join(w.text for w in words[start : index + 1]),
                    label=label,
                    start=words[start].start,
                    end=words[index].end,
                )
            )
            start, label = None, None
        elif edge == "I" and label is not None and entity_label != label:
            spans.append(
                NERSpan(
                    text=" ".join(w.text for w in words[start:index]),
                    label=label,
                    start=words[start].start,
                    end=words[index - 1].end,
                )
            )
            start, label = index, entity_label
    if start is not None and label is not None:
        spans.append(
            NERSpan(
                text=" ".join(w.text for w in words[start:]),
                label=label,
                start=words[start].start,
                end=words[-1].end,
            )
        )
    return spans
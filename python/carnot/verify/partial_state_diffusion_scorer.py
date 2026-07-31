"""Learned value head for partial DiffusionGemma denoising canvases.

The scorer is intentionally small and loadable: it trains a calibrated text
feature head over masked 256-cell token canvases and exposes
``score_partial_state(canvas, step)`` as an energy. The training data supplies
partial states and final correctness labels; the scorer never calls an
executable oracle at scoring time. The companion Exp 4292 runner performs the
load-bearing leak audit by masking answer-bearing cells before re-scoring.
"""

from __future__ import annotations

import json
import math
import pickle
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from carnot.serialization_safety import safe_pickle_load


DEFAULT_CANVAS_LEN = 256
DEFAULT_MASK_TOKEN_ID = 4
TOKEN_OFFSET = 10
ANSWER_PATTERNS = (
    re.compile(r"<<[^>]*>>\s*[-+]?\d[\d,]*(?:\.\d+)?"),
    re.compile(r"\\boxed\{[^}]+\}"),
    re.compile(r"boxed\{[^}]+\}"),
)


@dataclass(frozen=True)
class PartialStateRecord:
    """One masked denoising canvas paired with its final trajectory outcome."""

    task_id: str
    step: int
    canvas_ids: tuple[int, ...]
    answer_cell_indices: tuple[int, ...]
    label: bool
    source_text_sha256: str

    def to_preview(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "step": int(self.step),
            "revealed_cells": int(sum(token != DEFAULT_MASK_TOKEN_ID for token in self.canvas_ids)),
            "answer_cell_count": int(len(self.answer_cell_indices)),
            "label": bool(self.label),
            "source_text_sha256": self.source_text_sha256,
        }


@dataclass(frozen=True)
class MaskedCanvas:
    """A canvas with answer-bearing cells replaced by the mask token."""

    canvas_ids: tuple[int, ...]
    answer_cell_indices: tuple[int, ...]


class ByteCanvasEncoder:
    """Deterministic char-cell encoder for partial-canvas scorer fixtures.

    It maps each Unicode code point to a valid DiffusionGemma token-id range
    cell by adding a small offset, then masks unrevealed cells with the real
    DiffusionGemma mask token id. This gives tests and the build harness exact
    answer-cell offsets without pretending to be the GGUF tokenizer.
    """

    def __init__(
        self,
        *,
        canvas_len: int = DEFAULT_CANVAS_LEN,
        mask_token_id: int = DEFAULT_MASK_TOKEN_ID,
    ) -> None:
        if canvas_len <= 0:
            raise ValueError("canvas_len must be positive")
        self.canvas_len = int(canvas_len)
        self.mask_token_id = int(mask_token_id)

    def encode(
        self, text: str, *, visible_fraction: float
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if not 0.0 <= visible_fraction <= 1.0:
            raise ValueError("visible_fraction must be in [0, 1]")
        clipped = str(text)[: self.canvas_len]
        visible_cutoff = min(len(clipped), max(0, int(math.ceil(len(clipped) * visible_fraction))))
        canvas: list[int] = []
        for index, char in enumerate(clipped):
            token_id = ord(char) + TOKEN_OFFSET if index < visible_cutoff else self.mask_token_id
            canvas.append(int(token_id))
        while len(canvas) < self.canvas_len:
            canvas.append(self.mask_token_id)
        answer_indices = tuple(
            index
            for start, end in find_answer_spans(clipped)
            for index in range(max(0, start), min(end, self.canvas_len))
        )
        return tuple(canvas), answer_indices

    def decode_visible(self, canvas_ids: Sequence[int]) -> str:
        chars: list[str] = []
        for token_id in canvas_ids:
            value = int(token_id)
            if value == self.mask_token_id:
                continue
            codepoint = value - TOKEN_OFFSET
            if 0 <= codepoint <= 0x10FFFF:
                chars.append(chr(codepoint))
            else:
                chars.append(f" token_{value} ")
        return "".join(chars)


class PartialStateDiffusionScorer:
    """Small learned value head with a partial-canvas scoring API."""

    def __init__(
        self,
        *,
        random_seed: int = 4292,
        max_features: int = 5000,
        mask_token_id: int = DEFAULT_MASK_TOKEN_ID,
    ) -> None:
        self.random_seed = int(random_seed)
        self.max_features = int(max_features)
        self.mask_token_id = int(mask_token_id)
        self.vectorizer: TfidfVectorizer | None = None
        self.classifier: LogisticRegression | None = None
        self.is_fitted = False

    def fit(self, records: Sequence[PartialStateRecord]) -> "PartialStateDiffusionScorer":
        if not records:
            raise ValueError("at least one partial-state record is required")
        labels = np.asarray([bool(record.label) for record in records], dtype=bool)
        if len(set(bool(item) for item in labels)) != 2:
            raise ValueError("partial-state records must contain both labels")
        texts = [self._feature_text(record.canvas_ids, record.step) for record in records]
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=self.max_features,
            lowercase=True,
        )
        matrix = vectorizer.fit_transform(texts)
        classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=self.random_seed,
            solver="liblinear",
        )
        classifier.fit(matrix, labels)
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.is_fitted = True
        return self

    def predict_correct_proba(self, canvas_ids: Sequence[int], step: int) -> float:
        if not self.is_fitted or self.vectorizer is None or self.classifier is None:
            raise ValueError("PartialStateDiffusionScorer is not fitted")
        text = self._feature_text(canvas_ids, step)
        matrix = self.vectorizer.transform([text])
        probabilities = self.classifier.predict_proba(matrix)[0]
        class_to_index = {
            bool(value): index for index, value in enumerate(self.classifier.classes_)
        }
        return float(probabilities[class_to_index[True]])

    def score_partial_state(self, canvas_ids: Sequence[int], step: int) -> float:
        probability = min(max(self.predict_correct_proba(canvas_ids, step), 1e-12), 1.0)
        return float(-math.log(probability))

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> "PartialStateDiffusionScorer":
        """Load a pickled scorer from a trusted in-repo path.

        See ``DinaLRMPartialStateScorer.load`` for why the former
        ``pickle.load`` + ``isinstance`` pairing was not a security control (the
        payload runs during parsing, strictly before the type is inspected), and
        for why this loader warns rather than refuses on out-of-repo paths.
        """
        return safe_pickle_load(path, expected_type=cls, on_untrusted="warn")

    def _feature_text(self, canvas_ids: Sequence[int], step: int) -> str:
        visible_chars: list[str] = []
        token_features: list[str] = [f"step_{int(step)}"]
        for token_id in canvas_ids:
            value = int(token_id)
            if value == self.mask_token_id:
                continue
            token_features.append(f"tok_{value}")
            codepoint = value - TOKEN_OFFSET
            if 0 <= codepoint <= 0x10FFFF:
                visible_chars.append(chr(codepoint))
        return f"{' '.join(token_features)}\n{''.join(visible_chars)}"


def find_answer_spans(text: str) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    for pattern in ANSWER_PATTERNS:
        spans.extend(match.span() for match in pattern.finditer(str(text)))
    if not spans:
        return ()
    spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))
    return tuple(merged)


def build_partial_state_records(
    items: Iterable[dict[str, Any]],
    *,
    encoder: ByteCanvasEncoder | None = None,
    visible_fractions: Sequence[float] = (0.45, 0.7, 1.0),
) -> list[PartialStateRecord]:
    encoder = encoder or ByteCanvasEncoder()
    records: list[PartialStateRecord] = []
    for item in items:
        text = str(item.get("step_text") or item.get("text") or "")
        if not text:
            continue
        label_text = str(item.get("label", "")).lower()
        if label_text not in {"correct", "incorrect"}:
            continue
        task_id = str(item.get("question_id") or item.get("corpus_item_id") or len(records))
        source_hash = _sha256_text(text)
        for step, fraction in enumerate(visible_fractions):
            canvas, answer_indices = encoder.encode(text, visible_fraction=float(fraction))
            records.append(
                PartialStateRecord(
                    task_id=task_id,
                    step=int(step),
                    canvas_ids=canvas,
                    answer_cell_indices=answer_indices,
                    label=label_text == "correct",
                    source_text_sha256=source_hash,
                )
            )
    return records


def split_items_task_disjoint(
    items: Sequence[dict[str, Any]],
    *,
    heldout_fraction: float = 0.25,
    seed: int = 4292,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < heldout_fraction < 1.0:
        raise ValueError("heldout_fraction must be in (0, 1)")
    by_label: dict[str, list[str]] = {"correct": [], "incorrect": []}
    labels_by_task: dict[str, str] = {}
    for item in items:
        task_id = str(item.get("question_id") or item.get("corpus_item_id") or "")
        label = str(item.get("label", "")).lower()
        if task_id and label in by_label and task_id not in labels_by_task:
            labels_by_task[task_id] = label
            by_label[label].append(task_id)
    rng = random.Random(seed)
    heldout_ids: set[str] = set()
    for task_ids in by_label.values():
        rng.shuffle(task_ids)
        if task_ids:
            count = max(1, int(round(len(task_ids) * heldout_fraction)))
            heldout_ids.update(task_ids[:count])
    train: list[dict[str, Any]] = []
    heldout: list[dict[str, Any]] = []
    for item in items:
        task_id = str(item.get("question_id") or item.get("corpus_item_id") or "")
        if task_id in heldout_ids:
            heldout.append(dict(item))
        else:
            train.append(dict(item))
    if not train or not heldout:
        raise ValueError("task-disjoint split produced an empty train or heldout set")
    return train, heldout


def mask_answer_bearing_cells(
    record: PartialStateRecord,
    *,
    mask_token_id: int = DEFAULT_MASK_TOKEN_ID,
) -> MaskedCanvas:
    canvas = list(record.canvas_ids)
    for index in record.answer_cell_indices:
        if 0 <= int(index) < len(canvas):
            canvas[int(index)] = int(mask_token_id)
    return MaskedCanvas(canvas_ids=tuple(canvas), answer_cell_indices=record.answer_cell_indices)


def partial_state_auroc(
    scorer: PartialStateDiffusionScorer,
    records: Sequence[PartialStateRecord],
    *,
    mask_answer_cells: bool = False,
) -> float:
    if not records:
        raise ValueError("at least one held-out record is required")
    labels = [bool(record.label) for record in records]
    if len(set(labels)) != 2:
        raise ValueError("AUROC requires both positive and negative held-out labels")
    scores: list[float] = []
    for record in records:
        if mask_answer_cells:
            masked = mask_answer_bearing_cells(record, mask_token_id=scorer.mask_token_id)
            scores.append(scorer.predict_correct_proba(masked.canvas_ids, record.step))
        else:
            scores.append(scorer.predict_correct_proba(record.canvas_ids, record.step))
    return float(_rank_auroc(scores, labels))


def corpus_checksum(items: Sequence[dict[str, Any]]) -> str:
    encoded = json.dumps(list(items), sort_keys=True, ensure_ascii=True).encode("utf-8")
    return _sha256_bytes(encoded)


def _rank_auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    pairs = sorted(zip(scores, labels, strict=True), key=lambda item: item[0])
    n_pos = sum(1 for label in labels if label)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC requires both positive and negative labels")
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for _, label in pairs[index:end]:
            if label:
                rank_sum += average_rank
        index = end
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _sha256_text(text: str) -> str:
    return _sha256_bytes(str(text).encode("utf-8"))


def _sha256_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()

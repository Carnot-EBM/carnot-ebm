"""DiNa-LRM-style scorer for noisy, answer-masked diffusion states.

The scorer is deliberately smaller than the paper model, but it keeps the
load-bearing design constraints from DiNa-LRM: train on noisy diffusion states,
condition the reward on denoising timestep, calibrate uncertainty against the
noise schedule, and average over small inference-time noise perturbations. The
feature extractor masks answer-bearing cells before training so a downstream
run cannot reuse Exp 4292's answer-position shortcut.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from carnot.serialization_safety import safe_pickle_load

from carnot.verify.partial_state_diffusion_scorer import (
    DEFAULT_CANVAS_LEN,
    DEFAULT_MASK_TOKEN_ID,
    TOKEN_OFFSET,
    find_answer_spans,
    split_items_task_disjoint,
)


DEFAULT_VISIBLE_FRACTIONS = (0.35, 0.55, 0.8)
DEFAULT_NOISE_LEVELS = (0.35, 0.22, 0.1)
ANSWER_RECOVERY_CEILING = 0.6
PROCESS_RANKING_FLOOR = 0.6


@dataclass(frozen=True)
class DinaLRMRecord:
    """One noisy answer-masked canvas and its final process-quality label."""

    corpus_name: str
    task_id: str
    timestep: int
    noise_level: float
    canvas_ids: tuple[int, ...]
    answer_cell_indices: tuple[int, ...]
    hidden_answer_signature: str
    answer_signature_label: bool
    label: bool
    source_text_sha256: str

    def to_preview(self) -> dict[str, Any]:
        return {
            "corpus_name": self.corpus_name,
            "task_id": self.task_id,
            "timestep": int(self.timestep),
            "noise_level": round(float(self.noise_level), 6),
            "revealed_cells": int(sum(token != DEFAULT_MASK_TOKEN_ID for token in self.canvas_ids)),
            "answer_cell_count": int(len(self.answer_cell_indices)),
            "label": bool(self.label),
            "answer_signature_label": bool(self.answer_signature_label),
            "source_text_sha256": self.source_text_sha256,
        }


class DinaLRMCanvasEncoder:
    """Encode text as noisy masked DiffusionGemma-sized token canvases."""

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
        self,
        text: str,
        *,
        visible_fraction: float,
        timestep: int,
        noise_level: float,
        seed: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...], str, bool]:
        if not 0.0 <= visible_fraction <= 1.0:
            raise ValueError("visible_fraction must be in [0, 1]")
        if not 0.0 <= noise_level <= 1.0:
            raise ValueError("noise_level must be in [0, 1]")
        clipped = str(text)[: self.canvas_len]
        visible_cutoff = min(len(clipped), max(0, int(math.ceil(len(clipped) * visible_fraction))))
        answer_indices = tuple(
            index
            for start, end in find_answer_spans(clipped)
            for index in range(max(0, start), min(end, self.canvas_len))
        )
        answer_index_set = set(answer_indices)
        hidden_signature = _hidden_answer_signature(clipped, answer_indices)
        answer_signature_label = _signature_label(hidden_signature)
        rng = random.Random(_stable_seed(seed, clipped, str(timestep), str(noise_level)))
        canvas: list[int] = []
        for index, char in enumerate(clipped):
            if index >= visible_cutoff or index in answer_index_set or rng.random() < noise_level:
                canvas.append(self.mask_token_id)
            else:
                canvas.append(ord(char) + TOKEN_OFFSET)
        while len(canvas) < self.canvas_len:
            canvas.append(self.mask_token_id)
        return tuple(canvas), answer_indices, hidden_signature, answer_signature_label

    def decode_visible(self, canvas_ids: Sequence[int]) -> str:
        chars: list[str] = []
        for token_id in canvas_ids:
            value = int(token_id)
            if value == self.mask_token_id:
                continue
            codepoint = value - TOKEN_OFFSET
            if 0 <= codepoint <= 0x10FFFF:
                chars.append(chr(codepoint))
        return "".join(chars)


class DinaLRMPartialStateScorer:
    """Timestep-conditioned reward head over noisy answer-masked canvases."""

    def __init__(
        self,
        *,
        random_seed: int = 4337,
        max_features: int = 8000,
        mask_token_id: int = DEFAULT_MASK_TOKEN_ID,
        uncertainty_penalty: float = 0.15,
        noise_ensemble_offsets: Sequence[float] = (-0.05, 0.0, 0.05),
    ) -> None:
        self.random_seed = int(random_seed)
        self.max_features = int(max_features)
        self.mask_token_id = int(mask_token_id)
        self.uncertainty_penalty = float(uncertainty_penalty)
        self.noise_ensemble_offsets = tuple(float(item) for item in noise_ensemble_offsets)
        self.vectorizer: TfidfVectorizer | None = None
        self.classifier: LogisticRegression | None = None
        self.uncertainty_by_timestep: dict[int, float] = {}
        self.global_uncertainty = 0.25
        self.is_fitted = False

    def fit(self, records: Sequence[DinaLRMRecord]) -> "DinaLRMPartialStateScorer":
        if not records:
            raise ValueError("at least one DiNa-LRM record is required")
        labels = np.asarray([bool(record.label) for record in records], dtype=bool)
        if len(set(bool(item) for item in labels)) != 2:
            raise ValueError("DiNa-LRM records must contain both process-quality labels")
        texts = [
            self._feature_text(record.canvas_ids, record.timestep, record.noise_level)
            for record in records
        ]
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
        self._fit_uncertainty(records, labels)
        return self

    def predict_correct_proba(
        self,
        canvas_ids: Sequence[int],
        step: int,
        *,
        noise_level: float | None = None,
    ) -> float:
        if not self.is_fitted or self.vectorizer is None or self.classifier is None:
            raise ValueError("DinaLRMPartialStateScorer is not fitted")
        base_noise = (
            self._infer_noise_level(canvas_ids) if noise_level is None else float(noise_level)
        )
        probabilities: list[float] = []
        for offset in self.noise_ensemble_offsets:
            adjusted_noise = min(1.0, max(0.0, base_noise + offset))
            text = self._feature_text(canvas_ids, int(step), adjusted_noise)
            matrix = self.vectorizer.transform([text])
            proba = self.classifier.predict_proba(matrix)[0]
            class_to_index = {
                bool(value): index for index, value in enumerate(self.classifier.classes_)
            }
            probabilities.append(float(proba[class_to_index[True]]))
        return float(np.mean(probabilities))

    def score_partial_state(self, canvas_ids: Sequence[int], step: int) -> float:
        noise_level = self._infer_noise_level(canvas_ids)
        probability = min(
            max(self.predict_correct_proba(canvas_ids, step, noise_level=noise_level), 1e-12), 1.0
        )
        uncertainty = self.uncertainty_by_timestep.get(int(step), self.global_uncertainty)
        return float(-math.log(probability) + self.uncertainty_penalty * uncertainty)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> "DinaLRMPartialStateScorer":
        """Load a pickled scorer from a trusted in-repo path.

        The previous implementation called ``pickle.load`` directly and then
        checked ``isinstance``. That check is a correctness guard, not a
        security one: a pickle payload executes DURING parsing, so by the time
        the type is inspected any embedded code has already run. It is preserved
        via ``expected_type`` for its real purpose -- catching corruption and
        wrong-file mistakes.

        This loader uses ``on_untrusted="warn"``, NOT the strict default, and
        that is a deliberate weakening worth stating plainly: ``save``/``load``
        here is a general-purpose round-trip and writing to a scratch directory
        is ordinary usage, so refusing out-of-repo paths would break correct
        callers. The warning buys visibility, not prevention. The strict mode is
        reserved for the fixed-path loaders wired into the live verifier
        ensemble (``tier0e_eorm``, ``tier0f_semantic_calibration``), where an
        out-of-repo path really does indicate the mirror-supply-chain vector.
        """
        return safe_pickle_load(path, expected_type=cls, on_untrusted="warn")

    def _fit_uncertainty(self, records: Sequence[DinaLRMRecord], labels: np.ndarray) -> None:
        probabilities = np.asarray(
            [
                self.predict_correct_proba(
                    record.canvas_ids,
                    record.timestep,
                    noise_level=record.noise_level,
                )
                for record in records
            ],
            dtype=float,
        )
        residuals = np.abs(probabilities - labels.astype(float))
        self.global_uncertainty = float(np.mean(residuals)) if len(residuals) else 0.25
        by_step: dict[int, list[float]] = {}
        for record, residual in zip(records, residuals, strict=True):
            by_step.setdefault(int(record.timestep), []).append(float(residual))
        self.uncertainty_by_timestep = {
            step: float(np.mean(values)) for step, values in by_step.items() if values
        }

    def _feature_text(
        self,
        canvas_ids: Sequence[int],
        timestep: int,
        noise_level: float,
    ) -> str:
        visible_text = _decode_visible(canvas_ids, self.mask_token_id)
        non_mask_count = sum(1 for token in canvas_ids if int(token) != self.mask_token_id)
        length_bin = min(20, non_mask_count // 8)
        noise_bin = min(10, max(0, int(round(float(noise_level) * 10))))
        timestep_bin = max(0, int(timestep))
        return (
            f"timestep_{timestep_bin} noise_bin_{noise_bin} length_bin_{length_bin}\n{visible_text}"
        )

    def _infer_noise_level(self, canvas_ids: Sequence[int]) -> float:
        if not canvas_ids:
            return 1.0
        masked = sum(1 for token in canvas_ids if int(token) == self.mask_token_id)
        return float(masked / len(canvas_ids))


def build_dina_lrm_records(
    items: Iterable[dict[str, Any]],
    *,
    corpus_name: str,
    encoder: DinaLRMCanvasEncoder | None = None,
    visible_fractions: Sequence[float] = DEFAULT_VISIBLE_FRACTIONS,
    noise_levels: Sequence[float] = DEFAULT_NOISE_LEVELS,
    seed: int = 4337,
) -> list[DinaLRMRecord]:
    if len(tuple(visible_fractions)) != len(tuple(noise_levels)):
        raise ValueError("visible_fractions and noise_levels must have the same length")
    encoder = encoder or DinaLRMCanvasEncoder()
    records: list[DinaLRMRecord] = []
    for item in items:
        text = str(item.get("step_text") or item.get("text") or "")
        label_text = str(item.get("label", "")).lower()
        if not text or label_text not in {"correct", "incorrect"}:
            continue
        task_id = str(item.get("question_id") or item.get("corpus_item_id") or len(records))
        source_hash = _sha256_text(text)
        for timestep, (visible_fraction, noise_level) in enumerate(
            zip(visible_fractions, noise_levels, strict=True)
        ):
            canvas, answer_indices, hidden_signature, answer_signature_label = encoder.encode(
                text,
                visible_fraction=float(visible_fraction),
                timestep=int(timestep),
                noise_level=float(noise_level),
                seed=seed,
            )
            records.append(
                DinaLRMRecord(
                    corpus_name=str(corpus_name),
                    task_id=task_id,
                    timestep=int(timestep),
                    noise_level=float(noise_level),
                    canvas_ids=canvas,
                    answer_cell_indices=answer_indices,
                    hidden_answer_signature=hidden_signature,
                    answer_signature_label=bool(answer_signature_label),
                    label=label_text == "correct",
                    source_text_sha256=source_hash,
                )
            )
    return records


def split_corpus_items(
    items: Sequence[dict[str, Any]],
    *,
    heldout_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return split_items_task_disjoint(items, heldout_fraction=heldout_fraction, seed=seed)


def process_ranking_auroc(
    scorer: DinaLRMPartialStateScorer,
    records: Sequence[DinaLRMRecord],
) -> float:
    if not records:
        raise ValueError("at least one held-out record is required")
    labels = [bool(record.label) for record in records]
    if len(set(labels)) != 2:
        raise ValueError("process ranking AUROC requires both labels")
    scores = [
        scorer.predict_correct_proba(
            record.canvas_ids,
            record.timestep,
            noise_level=record.noise_level,
        )
        for record in records
    ]
    return float(_rank_auroc(scores, labels))


def masked_answer_recovery_auroc(
    scorer: DinaLRMPartialStateScorer,
    records: Sequence[DinaLRMRecord],
) -> float:
    """Return the strongest within-label hidden-answer recovery AUROC.

    The probe is stratified by process-quality label so a useful reward signal
    does not get misclassified as answer leakage merely because correct process
    states receive higher reward.
    """

    leak_aurocs: list[float] = []
    for process_label in (False, True):
        group = [record for record in records if bool(record.label) is process_label]
        answer_labels = [bool(record.answer_signature_label) for record in group]
        if len(group) < 4 or len(set(answer_labels)) != 2:
            continue
        scores = [
            scorer.predict_correct_proba(
                record.canvas_ids,
                record.timestep,
                noise_level=record.noise_level,
            )
            for record in group
        ]
        raw = float(_rank_auroc(scores, answer_labels))
        leak_aurocs.append(max(raw, 1.0 - raw))
    return max(leak_aurocs) if leak_aurocs else 0.5


def corpus_checksum(items: Sequence[dict[str, Any]]) -> str:
    encoded = json.dumps(list(items), sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _decode_visible(canvas_ids: Sequence[int], mask_token_id: int) -> str:
    chars: list[str] = []
    for token_id in canvas_ids:
        value = int(token_id)
        if value == int(mask_token_id):
            continue
        codepoint = value - TOKEN_OFFSET
        if 0 <= codepoint <= 0x10FFFF:
            chars.append(chr(codepoint))
    return "".join(chars)


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
        for _score, label in pairs[index:end]:
            if label:
                rank_sum += average_rank
        index = end
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _hidden_answer_signature(text: str, answer_indices: Sequence[int]) -> str:
    if answer_indices:
        hidden = "".join(
            str(text)[index] for index in answer_indices if 0 <= int(index) < len(text)
        )
    else:
        hidden = hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:16]
    return hashlib.sha256(hidden.encode("utf-8")).hexdigest()


def _signature_label(signature: str) -> bool:
    return bool(int(signature[:8], 16) % 2)


def _stable_seed(seed: int, *parts: str) -> int:
    payload = "|".join((str(seed), *parts)).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()

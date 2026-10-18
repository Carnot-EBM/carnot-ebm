#!/usr/bin/env python3
"""Evaluate one-pass option-logit readout on public JevBench tasks.

The module reimplements the required metrics. It never imports JevBench code.
It writes an artifact only when the caller supplies ``--output``.
Spec: REQ-AUTO-028, SCENARIO-AUTO-028-A, SCENARIO-AUTO-028-B.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

DEFAULT_TASKS_DIR = Path("/home/ianblenke/jevbench-src/datasets/public")
SOURCE_REPOSITORY = "https://github.com/fstandhartinger/jevbench"
MODEL_HF_ID = "unsloth/Qwen3.8-27B-GGUF"
PUBLIC_SUBSET_SIZE = 231
FULL_BENCHMARK_SIZE = 534
SPLIT_NAMES = ("easy", "hard", "original")
BOOTSTRAP_RESAMPLES = 1000


@dataclass(frozen=True)
class Option:
    """One stable option ID and its rubric text."""

    option_id: str
    description: str


@dataclass(frozen=True)
class DecisionTask:
    """The task fields needed for scoring and aggregate metrics."""

    task_id: str
    source_split: str
    family: str
    state: object
    question: str
    options: tuple[Option, ...]
    expected: str
    gold_probs: dict[str, float] | None


@dataclass(frozen=True)
class LoadError:
    """One source row that could not become a valid task."""

    file: str
    line_number: int
    message: str


@dataclass(frozen=True)
class LoadReport:
    """Loaded tasks plus visible errors and source hashes."""

    tasks: tuple[DecisionTask, ...]
    errors: tuple[LoadError, ...]
    file_sha256s: dict[str, str]


@dataclass(frozen=True)
class ScoredTask:
    """Original-order and reversed-order probabilities for one task."""

    task: DecisionTask
    original_probs: dict[str, float]
    reversed_probs: dict[str, float]


@dataclass(frozen=True)
class OptionOrderProbeResult:
    """Aggregate option-order results and their task-level inputs."""

    accuracy_original: float
    accuracy_reversed: float
    mean_absolute_probability_shift: float
    rows: tuple[ScoredTask, ...]


@dataclass(frozen=True)
class CalibrationFold:
    """One out-of-fold temperature fit receipt."""

    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    temperature: float


class OptionScorer(Protocol):
    """Score all declared options as one normalized distribution."""

    def score(
        self,
        state: object,
        question: str,
        options: Sequence[Option],
    ) -> dict[str, float]: ...


class StubScorer:
    """Return deterministic, order-invariant probabilities without inference."""

    def score(
        self,
        state: object,
        question: str,
        options: Sequence[Option],
    ) -> dict[str, float]:
        state_blob = json.dumps(state, ensure_ascii=False, sort_keys=True)
        weights: dict[str, float] = {}
        for option in options:
            blob = f"{state_blob}\n{question}\n{option.option_id}".encode()
            weights[option.option_id] = float(int(hashlib.sha256(blob).hexdigest()[:12], 16) + 1)
        total = sum(weights.values())
        return {option.option_id: weights[option.option_id] / total for option in options}


class LlamaCppLogitScorer:
    """Read next-token option logits from one llama.cpp evaluation call."""

    def __init__(self, model_path: Path, n_ctx: int = 32768) -> None:
        self.model_path = model_path
        self.n_ctx = n_ctx
        self._llm: Any = None
        self._backend: Any = None
        self._labels_by_count: dict[int, tuple[tuple[str, str, int], ...]] = {}
        self.load_duration_s: float | None = None
        self.gpu_offload_evidence: dict[str, object] | None = None

    def load(self) -> None:
        """Load llama.cpp lazily so tests and dry runs need no backend."""

        if self._llm is not None:
            return
        started = time.perf_counter()
        import llama_cpp  # noqa: PLC0415

        supports = getattr(llama_cpp, "llama_supports_gpu_offload", None)
        supports_gpu_offload = bool(supports()) if callable(supports) else False
        self._backend = llama_cpp
        self._llm = llama_cpp.Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_gpu_layers=-1,
            logits_all=True,
            verbose=False,
        )
        self.load_duration_s = time.perf_counter() - started
        self.gpu_offload_evidence = {
            "llama_supports_gpu_offload": supports_gpu_offload,
            "n_gpu_layers_requested": -1,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }

    def _tokenize(self, text: str, *, add_bos: bool) -> list[int]:
        try:
            return list(self._llm.tokenize(text.encode(), add_bos=add_bos, special=True))
        except TypeError:
            return list(self._llm.tokenize(text.encode(), add_bos=add_bos))

    def _option_labels(self, count: int) -> tuple[tuple[str, str, int], ...]:
        cached = self._labels_by_count.get(count)
        if cached is not None:
            return cached
        candidates = [chr(ord("A") + index) for index in range(26)]
        labels: list[tuple[str, str, int]] = []
        for display in candidates:
            token_text = f" {display}"
            token_ids = self._tokenize(token_text, add_bos=False)
            if len(token_ids) == 1:
                labels.append((display, token_text, token_ids[0]))
            if len(labels) == count:
                result = tuple(labels)
                self._labels_by_count[count] = result
                return result
        raise RuntimeError(f"GGUF tokenizer has fewer than {count} usable single-token labels")

    @staticmethod
    def _state_text(state: object) -> str:
        if isinstance(state, str):
            return state
        return json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2)

    def _build_prompt(
        self,
        state: object,
        question: str,
        options: Sequence[Option],
        labels: Sequence[tuple[str, str, int]],
    ) -> str:
        option_lines = [
            f"{label[0]}: {option.option_id} — {option.description}"
            for label, option in zip(labels, options, strict=True)
        ]
        return (
            "Select one declared option. Use the state and rubric.\n\n"
            f"State:\n{self._state_text(state)}\n\n"
            f"Question:\n{question}\n\n"
            "Options:\n" + "\n".join(option_lines) + "\n\nReturn only the option label.\nAnswer:"
        )

    def score(
        self,
        state: object,
        question: str,
        options: Sequence[Option],
    ) -> dict[str, float]:
        if self._llm is None:
            raise RuntimeError("load() must run before score()")
        labels = self._option_labels(len(options))
        prompt = self._build_prompt(state, question, options, labels)
        tokens = self._tokenize(prompt, add_bos=True)
        if len(tokens) >= self.n_ctx:
            raise ValueError(f"prompt has {len(tokens)} tokens but n_ctx is {self.n_ctx}")
        self._llm.reset()
        self._llm.eval(tokens)
        # `scores` is the whole (n_ctx x vocab) buffer. The last PROMPT row is n_tokens - 1;
        # scores[-1] is the buffer's final, never-written row (found 2026-09-20: it made every
        # readout uniform, so a whole run scored at chance and read as a finding).
        next_token_logits = self._llm.scores[len(tokens) - 1]
        selected = [float(next_token_logits[label[2]]) for label in labels]
        maximum = max(selected)
        weights = [math.exp(value - maximum) for value in selected]
        total = sum(weights)
        return {
            option.option_id: weight / total
            for option, weight in zip(options, weights, strict=True)
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _option_descriptions(
    question: Mapping[str, object], labels: tuple[str, ...]
) -> tuple[Option, ...]:
    criteria = question.get("criteria")
    question_type = question.get("type")
    descriptions: list[str] = []
    for index, label in enumerate(labels):
        description: object = label
        if isinstance(criteria, Mapping):
            description = criteria.get(label, label)
            if question_type == "noul":
                mapped = {"no": "false", "yes": "true"}.get(label)
                if mapped is not None:
                    description = criteria.get(mapped, description)
        elif isinstance(criteria, list) and index < len(criteria):
            description = criteria[index]
        descriptions.append(str(description) if description is not None else label)
    return tuple(
        Option(label, description) for label, description in zip(labels, descriptions, strict=True)
    )


def _task_from_row(row: object, source_split: str) -> DecisionTask:
    if not isinstance(row, Mapping):
        raise ValueError("row is not a JSON object")
    task_id = row.get("id")
    family = row.get("family")
    question = row.get("question")
    labels_value = row.get("labels")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("id must be a non-empty string")
    if not isinstance(family, str) or not family:
        raise ValueError("family must be a non-empty string")
    if not isinstance(question, Mapping):
        raise ValueError("question must be an object")
    question_type = question.get("type")
    if question_type not in {"noul", "choice", "score"}:
        raise ValueError(f"unsupported question type: {question_type!r}")
    instructions = question.get("instructions")
    if not isinstance(instructions, str) or not instructions:
        raise ValueError("question.instructions must be a non-empty string")
    if not isinstance(labels_value, list) or not labels_value:
        raise ValueError("labels must be a non-empty list")
    if any(not isinstance(label, str) or not label for label in labels_value):
        raise ValueError("each label must be a non-empty string")
    labels = tuple(labels_value)
    if len(set(labels)) != len(labels):
        raise ValueError("labels must be unique")
    expected_value = row.get("expected")
    expected = str(expected_value)
    if expected not in labels:
        raise ValueError(f"expected label {expected!r} is absent from labels")
    if row.get("split") != "public":
        raise ValueError("task split must be public")
    provenance = row.get("provenance")
    gold_probs: dict[str, float] | None = None
    if isinstance(provenance, Mapping) and provenance.get("gold_probs") is not None:
        gold_value = provenance["gold_probs"]
        if not isinstance(gold_value, Mapping):
            raise ValueError("provenance.gold_probs must be an object")
        gold_probs = {str(key): float(value) for key, value in gold_value.items()}
        _validate_distribution(gold_probs, labels)
    return DecisionTask(
        task_id=task_id,
        source_split=source_split,
        family=family,
        state=row.get("state"),
        question=instructions,
        options=_option_descriptions(question, labels),
        expected=expected,
        gold_probs=gold_probs,
    )


def load_public_tasks(tasks_dir: Path, splits: Sequence[str]) -> LoadReport:
    """Read selected JSONL files and report every malformed row."""

    tasks: list[DecisionTask] = []
    errors: list[LoadError] = []
    file_sha256s: dict[str, str] = {}
    seen_ids: set[str] = set()
    for split in splits:
        if split not in SPLIT_NAMES:
            raise ValueError(f"unknown split: {split}")
        path = tasks_dir / f"{split}.jsonl"
        if not path.is_file():
            errors.append(LoadError(str(path), 0, "source file is missing"))
            continue
        file_sha256s[path.name] = _sha256_file(path)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    errors.append(
                        LoadError(path.name, line_number, f"JSON decode error: {error.msg}")
                    )
                    continue
                try:
                    task = _task_from_row(row, split)
                    if task.task_id in seen_ids:
                        raise ValueError(f"duplicate task id: {task.task_id}")
                except (TypeError, ValueError, KeyError) as error:
                    errors.append(LoadError(path.name, line_number, str(error)))
                    continue
                seen_ids.add(task.task_id)
                tasks.append(task)
    return LoadReport(tuple(tasks), tuple(errors), file_sha256s)


def _validate_distribution(probabilities: Mapping[str, float], labels: Sequence[str]) -> None:
    if set(probabilities) != set(labels):
        raise ValueError("probability keys do not match declared option IDs")
    values = [float(probabilities[label]) for label in labels]
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("probabilities must be finite and non-negative")
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("probabilities must sum to 1")


def _predicted_label(probabilities: Mapping[str, float]) -> str:
    return max(probabilities, key=lambda label: probabilities[label])


def accuracy(probabilities: Sequence[Mapping[str, float]], expected: Sequence[str]) -> float:
    """Return argmax accuracy."""

    if len(probabilities) != len(expected) or not expected:
        raise ValueError("accuracy needs equal non-empty inputs")
    correct = sum(
        _predicted_label(distribution) == target
        for distribution, target in zip(probabilities, expected, strict=True)
    )
    return correct / len(expected)


def top_label_ece(
    probabilities: Sequence[Mapping[str, float]],
    expected: Sequence[str],
    n_bins: int = 10,
) -> float:
    """Return top-label ECE with equal-width confidence bins."""

    if len(probabilities) != len(expected) or not expected:
        raise ValueError("ECE needs equal non-empty inputs")
    counts = [0] * n_bins
    confidence_sums = [0.0] * n_bins
    correct_counts = [0] * n_bins
    for distribution, target in zip(probabilities, expected, strict=True):
        predicted = _predicted_label(distribution)
        confidence = min(max(float(distribution[predicted]), 0.0), 1.0)
        bin_index = min(int(confidence * n_bins), n_bins - 1)
        counts[bin_index] += 1
        confidence_sums[bin_index] += confidence
        correct_counts[bin_index] += int(predicted == target)
    ece = 0.0
    for count, confidence_sum, correct_count in zip(
        counts, confidence_sums, correct_counts, strict=True
    ):
        if count:
            ece += count / len(expected) * abs(correct_count / count - confidence_sum / count)
    return ece


def brier_score(
    probabilities: Mapping[str, float],
    expected: str,
    labels: Sequence[str],
) -> float:
    """Return the multiclass sum of squared probability errors."""

    return sum((float(probabilities[label]) - float(label == expected)) ** 2 for label in labels)


def total_variation_distance(
    predicted: Mapping[str, float],
    gold: Mapping[str, float],
    labels: Sequence[str],
) -> float:
    """Return half the L1 distance between two option distributions."""

    return 0.5 * sum(abs(float(predicted[label]) - float(gold[label])) for label in labels)


def jevbench_calibration_score(ece: float, mean_tvd: float | None) -> float:
    """Reimplement the v1.2 calibration-axis formula."""

    ece_component = max(0.0, 100.0 * (1.0 - ece / 0.5))
    if mean_tvd is None:
        return ece_component
    fidelity_component = 100.0 * (1.0 - mean_tvd)
    return (ece_component + fidelity_component) / 2.0


def run_option_order_probe(
    tasks: Sequence[DecisionTask],
    scorer: OptionScorer,
    *,
    progress_every: int = 25,
) -> OptionOrderProbeResult:
    """Score both option orders and map each result by stable option ID."""

    if not tasks:
        raise ValueError("option-order probe needs at least one task")
    rows: list[ScoredTask] = []
    shifts: list[float] = []
    original_correct = 0
    reversed_correct = 0
    for index, task in enumerate(tasks, start=1):
        option_ids = tuple(option.option_id for option in task.options)
        original = scorer.score(task.state, task.question, task.options)
        reversed_options = tuple(reversed(task.options))
        reversed_probs = scorer.score(task.state, task.question, reversed_options)
        _validate_distribution(original, option_ids)
        _validate_distribution(reversed_probs, option_ids)
        original_correct += int(_predicted_label(original) == task.expected)
        reversed_correct += int(_predicted_label(reversed_probs) == task.expected)
        shifts.extend(abs(original[label] - reversed_probs[label]) for label in option_ids)
        rows.append(ScoredTask(task, dict(original), dict(reversed_probs)))
        if progress_every > 0 and (index % progress_every == 0 or index == len(tasks)):
            print(f"progress: scored {index}/{len(tasks)} tasks in both option orders", flush=True)
    return OptionOrderProbeResult(
        accuracy_original=original_correct / len(tasks),
        accuracy_reversed=reversed_correct / len(tasks),
        mean_absolute_probability_shift=sum(shifts) / len(shifts),
        rows=tuple(rows),
    )


def kfold_indices(
    n_rows: int,
    folds: int,
    seed: int,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Build seeded folds whose training and test indices never overlap."""

    if n_rows < 1:
        raise ValueError("cross-validation needs at least one row")
    if folds < 2:
        raise ValueError("folds must be at least 2")
    fold_count = min(folds, n_rows)
    shuffled = list(range(n_rows))
    random.Random(seed).shuffle(shuffled)
    if fold_count == 1:
        return (((), (shuffled[0],)),)
    test_folds = [tuple(shuffled[index::fold_count]) for index in range(fold_count)]
    all_indices = set(range(n_rows))
    return tuple(
        (tuple(sorted(all_indices.difference(test))), tuple(sorted(test))) for test in test_folds
    )


def _temperature_scale(
    probabilities: Mapping[str, float],
    labels: Sequence[str],
    temperature: float,
) -> dict[str, float]:
    scaled_logits = [
        math.log(max(float(probabilities[label]), 1e-15)) / temperature for label in labels
    ]
    maximum = max(scaled_logits)
    weights = [math.exp(value - maximum) for value in scaled_logits]
    total = sum(weights)
    return {label: weight / total for label, weight in zip(labels, weights, strict=True)}


def _fit_temperature(
    probabilities: Sequence[Mapping[str, float]],
    expected: Sequence[str],
    labels: Sequence[Sequence[str]],
) -> float:
    if not probabilities:
        return 1.0

    def loss(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        total = 0.0
        for distribution, target, row_labels in zip(probabilities, expected, labels, strict=True):
            calibrated = _temperature_scale(distribution, row_labels, temperature)
            total -= math.log(max(calibrated[target], 1e-15))
        return total / len(probabilities)

    left = math.log(0.05)
    right = math.log(20.0)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = right - ratio * (right - left)
    x2 = left + ratio * (right - left)
    f1 = loss(x1)
    f2 = loss(x2)
    for _ in range(80):
        if f1 <= f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - ratio * (right - left)
            f1 = loss(x1)
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + ratio * (right - left)
            f2 = loss(x2)
    return math.exp((left + right) / 2.0)


def cross_validated_temperature_scale(
    probabilities: Sequence[Mapping[str, float]],
    expected: Sequence[str],
    labels: Sequence[Sequence[str]],
    *,
    folds: int,
    seed: int,
) -> tuple[list[dict[str, float]], tuple[CalibrationFold, ...]]:
    """Fit on k-1 folds and calibrate only each held-out fold."""

    if not (len(probabilities) == len(expected) == len(labels)):
        raise ValueError("calibration inputs must have equal lengths")
    output: list[dict[str, float] | None] = [None] * len(probabilities)
    receipts: list[CalibrationFold] = []
    for train_indices, test_indices in kfold_indices(len(probabilities), folds, seed):
        train_probabilities = [probabilities[index] for index in train_indices]
        train_expected = [expected[index] for index in train_indices]
        train_labels = [labels[index] for index in train_indices]
        temperature = _fit_temperature(train_probabilities, train_expected, train_labels)
        for index in test_indices:
            output[index] = _temperature_scale(probabilities[index], labels[index], temperature)
        receipts.append(CalibrationFold(train_indices, test_indices, temperature))
    if any(distribution is None for distribution in output):
        raise RuntimeError("cross-validation did not calibrate every row")
    return [distribution for distribution in output if distribution is not None], tuple(receipts)


def _metric_values(
    tasks: Sequence[DecisionTask],
    probabilities: Sequence[Mapping[str, float]],
) -> dict[str, float | None]:
    expected = [task.expected for task in tasks]
    ece = top_label_ece(probabilities, expected)
    brier_values = [
        brier_score(
            distribution,
            task.expected,
            tuple(option.option_id for option in task.options),
        )
        for task, distribution in zip(tasks, probabilities, strict=True)
    ]
    tvd_values = [
        total_variation_distance(
            distribution,
            task.gold_probs,
            tuple(option.option_id for option in task.options),
        )
        for task, distribution in zip(tasks, probabilities, strict=True)
        if task.gold_probs is not None
    ]
    mean_tvd = sum(tvd_values) / len(tvd_values) if tvd_values else None
    return {
        "accuracy": accuracy(probabilities, expected),
        "top_label_ece_10_bin": ece,
        "brier_score": sum(brier_values) / len(brier_values),
        "mean_total_variation_distance": mean_tvd,
        "probability_fidelity": 1.0 - mean_tvd if mean_tvd is not None else None,
        "jevbench_style_calibration_score": jevbench_calibration_score(ece, mean_tvd),
        "n": float(len(tasks)),
        "n_gold_distributions": float(len(tvd_values)),
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _metrics_with_bootstrap(
    tasks: Sequence[DecisionTask],
    probabilities: Sequence[Mapping[str, float]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, object]:
    values = _metric_values(tasks, probabilities)
    samples: dict[str, list[float]] = defaultdict(list)
    rng = random.Random(seed)
    for _ in range(resamples):
        indices = [rng.randrange(len(tasks)) for _ in tasks]
        sampled_tasks = [tasks[index] for index in indices]
        sampled_probabilities = [probabilities[index] for index in indices]
        sample_values = _metric_values(sampled_tasks, sampled_probabilities)
        for name, value in sample_values.items():
            if value is not None and name not in {"n", "n_gold_distributions"}:
                samples[name].append(value)
    report: dict[str, object] = {}
    for name, value in values.items():
        if name in {"n", "n_gold_distributions"}:
            report[name] = int(value or 0)
            continue
        interval_values = samples.get(name, [])
        report[name] = {
            "value": value,
            "ci95": (
                [_percentile(interval_values, 0.025), _percentile(interval_values, 0.975)]
                if interval_values
                else None
            ),
        }
    return report


def _read_git_head(repo_root: Path | None) -> str | None:
    if repo_root is None:
        return None
    git_path = repo_root / ".git"
    if git_path.is_file():
        pointer = git_path.read_text(encoding="utf-8").strip()
        if not pointer.startswith("gitdir: "):
            return None
        git_path = (repo_root / pointer.removeprefix("gitdir: ")).resolve()
    head_path = git_path / "HEAD"
    if not head_path.is_file():
        return None
    head = head_path.read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head if len(head) == 40 else None
    reference = head.removeprefix("ref: ")
    loose_ref = git_path / reference
    if loose_ref.is_file():
        return loose_ref.read_text(encoding="utf-8").strip()
    packed_refs = git_path / "packed-refs"
    if packed_refs.is_file():
        suffix = f" {reference}"
        for line in packed_refs.read_text(encoding="utf-8").splitlines():
            if line.endswith(suffix):
                return line.split(" ", 1)[0]
    return None


def _find_repo_root(path: Path) -> Path | None:
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _resolve_default_model() -> tuple[Path, dict[str, object]]:
    from carnot.inference.sota_models import cached_current_model, cached_sota_pair  # noqa: PLC0415

    pairs = cached_sota_pair(gpu_indices=(0, 1))
    selected = None
    if pairs is not None:
        selected = next((spec for spec in pairs if spec.get("hf_id") == MODEL_HF_ID), None)
    if selected is None:
        selected = cached_current_model(gpu_index=0)
    if selected is None or selected.get("hf_id") != MODEL_HF_ID:
        raise FileNotFoundError(f"cached GGUF not found for {MODEL_HF_ID}")
    model_path = Path(str(selected["model_path"]))
    return model_path, dict(selected)


def _option_probe_summary(rows: Sequence[ScoredTask]) -> dict[str, float | int]:
    shifts = [
        abs(row.original_probs[option.option_id] - row.reversed_probs[option.option_id])
        for row in rows
        for option in row.task.options
    ]
    return {
        "n": len(rows),
        "accuracy_original": sum(
            _predicted_label(row.original_probs) == row.task.expected for row in rows
        )
        / len(rows),
        "accuracy_reversed": sum(
            _predicted_label(row.reversed_probs) == row.task.expected for row in rows
        )
        / len(rows),
        "mean_absolute_probability_shift_per_option": sum(shifts) / len(shifts),
    }


def _reproducibility_checksum(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


_CONTROL_CASES: tuple[tuple[str, str, tuple[tuple[str, str], ...], str], ...] = (
    (
        "Which item is a fruit?",
        "Pick one.",
        (("apple", "An apple."), ("hammer", "A hammer.")),
        "apple",
    ),
    ("What is 2 + 2?", "Pick one.", (("four", "The number 4."), ("nine", "The number 9.")), "four"),
    (
        "Which city is the capital of France?",
        "Pick one.",
        (("paris", "Paris."), ("cairo", "Cairo.")),
        "paris",
    ),
    (
        "Which is a liquid at room temperature?",
        "Pick one.",
        (("water", "Water."), ("iron", "Iron.")),
        "water",
    ),
)


def readout_positive_control(scorer: OptionScorer) -> dict[str, object]:
    """Known-answer check that the readout can tell options apart at all.

    Runs trivial questions in both option orders. A readout that cannot pass this is broken,
    and a benchmark number from it is not a finding. Raises instead of returning a bad result.
    """

    correct = 0
    total = 0
    confidence_sum = 0.0
    for state, question, raw_options, expected in _CONTROL_CASES:
        for options_raw in (raw_options, tuple(reversed(raw_options))):
            options = tuple(Option(option_id=oid, description=text) for oid, text in options_raw)
            probs = scorer.score(state, question, options)
            total += 1
            if max(probs, key=lambda key: probs[key]) == expected:
                correct += 1
            confidence_sum += probs[expected]
    result: dict[str, object] = {
        "cases": total,
        "correct": correct,
        "mean_probability_of_correct_option": confidence_sum / total,
    }
    if correct < 7 or confidence_sum / total < 0.6:
        raise RuntimeError(
            f"readout failed its positive control ({correct}/{total} correct, "
            f"mean p(correct)={confidence_sum / total:.3f}); refusing to report benchmark numbers"
        )
    return result


def _uniform_fraction(distributions: Sequence[Mapping[str, float]]) -> float:
    if not distributions:
        return 0.0
    uniform = 0
    for dist in distributions:
        values = list(dist.values())
        if max(values) - min(values) < 1e-6:
            uniform += 1
    return uniform / len(distributions)


def run_evaluation(
    *,
    tasks_dir: Path,
    split: str,
    limit: int | None,
    dry_run: bool,
    model_gguf: Path | None,
    seed: int,
    folds: int,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, object]:
    """Run scoring and return an artifact without writing it."""

    started = time.perf_counter()
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be positive")
    selected_splits = SPLIT_NAMES if split == "all" else (split,)
    report = load_public_tasks(tasks_dir, selected_splits)
    tasks = report.tasks[:limit] if limit is not None else report.tasks
    if not tasks:
        raise ValueError("no valid tasks were loaded")

    preconditions_checked: dict[str, object] = {
        "tasks_dir_exists": tasks_dir.is_dir(),
        "selected_source_files_hashed": sorted(report.file_sha256s),
        "load_errors_reported": len(report.errors),
        "cpu_only_stub": dry_run,
    }
    gpu_evidence: dict[str, object] | None = None
    if dry_run:
        print("model load before: dry-run stub selected; no model will load", flush=True)
        scorer: OptionScorer = StubScorer()
        print("model load after: skipped for dry-run stub", flush=True)
        model_specs = [
            {
                "scorer": "StubScorer",
                "hf_id": MODEL_HF_ID,
                "gguf_path": None,
                "live_model_invoked": False,
            }
        ]
        inference_substrate = "stub_deterministic_no_live_inference"
    else:
        if model_gguf is None:
            resolved_path, resolved_spec = _resolve_default_model()
        else:
            resolved_path = model_gguf.expanduser().resolve()
            resolved_spec = {
                "name": "Qwen3.8 27B",
                "hf_id": MODEL_HF_ID,
                "model_path": str(resolved_path),
                "selection_role": "operator_override_path",
            }
        if not resolved_path.is_file():
            raise FileNotFoundError(f"GGUF does not exist: {resolved_path}")
        print(f"model load before: {resolved_path}", flush=True)
        live_scorer = LlamaCppLogitScorer(resolved_path)
        live_scorer.load()
        print(
            f"model load after: completed in {live_scorer.load_duration_s:.3f}s",
            flush=True,
        )
        scorer = live_scorer
        gpu_evidence = live_scorer.gpu_offload_evidence
        model_specs = [
            {
                "scorer": "LlamaCppLogitScorer",
                "hf_id": MODEL_HF_ID,
                "gguf_path": str(resolved_path),
                "selection_role": resolved_spec.get("selection_role"),
                "live_model_invoked": True,
            }
        ]
        inference_substrate = "live_llm_inference"
        preconditions_checked["model_gguf_exists"] = True
        preconditions_checked["llama_cpp_imported"] = True
        preconditions_checked["readout_positive_control"] = readout_positive_control(scorer)
        print(
            f"positive control passed: {preconditions_checked['readout_positive_control']}",
            flush=True,
        )

    probe = run_option_order_probe(tasks, scorer)
    raw_probabilities = [row.original_probs for row in probe.rows]
    uniform_fraction = _uniform_fraction(raw_probabilities)
    if not dry_run and uniform_fraction > 0.5:
        raise RuntimeError(
            f"degenerate readout: {uniform_fraction:.0%} of distributions are uniform; "
            "refusing to write an artifact"
        )
    expected = [task.expected for task in tasks]
    labels = [tuple(option.option_id for option in task.options) for task in tasks]
    calibrated_probabilities, calibration_folds = cross_validated_temperature_scale(
        raw_probabilities,
        expected,
        labels,
        folds=folds,
        seed=seed,
    )

    metrics_by_split: dict[str, object] = {}
    probe_by_split: dict[str, object] = {}
    for split_name in selected_splits:
        indices = [index for index, task in enumerate(tasks) if task.source_split == split_name]
        if not indices:
            continue
        split_tasks = [tasks[index] for index in indices]
        split_raw = [raw_probabilities[index] for index in indices]
        split_calibrated = [calibrated_probabilities[index] for index in indices]
        metrics_by_split[split_name] = {
            "raw": _metrics_with_bootstrap(
                split_tasks,
                split_raw,
                resamples=bootstrap_resamples,
                seed=seed + 101,
            ),
            "calibrated": _metrics_with_bootstrap(
                split_tasks,
                split_calibrated,
                resamples=bootstrap_resamples,
                seed=seed + 102,
            ),
        }
        probe_by_split[split_name] = _option_probe_summary([probe.rows[index] for index in indices])

    metrics = {
        "bootstrap_resamples": bootstrap_resamples,
        "overall": {
            "raw": _metrics_with_bootstrap(
                tasks,
                raw_probabilities,
                resamples=bootstrap_resamples,
                seed=seed + 1,
            ),
            "calibrated": _metrics_with_bootstrap(
                tasks,
                calibrated_probabilities,
                resamples=bootstrap_resamples,
                seed=seed + 2,
            ),
        },
        "by_split": metrics_by_split,
    }
    n_per_split = {name: sum(task.source_split == name for task in tasks) for name in SPLIT_NAMES}
    source_repo_root = _find_repo_root(tasks_dir.resolve())
    source = {
        "repository": SOURCE_REPOSITORY,
        "checkout": str(source_repo_root) if source_repo_root is not None else None,
        "commit": _read_git_head(source_repo_root),
        "file_sha256s": report.file_sha256s,
        "load_errors": [
            {
                "file": error.file,
                "line_number": error.line_number,
                "message": error.message,
            }
            for error in report.errors
        ],
    }
    fold_receipts = [
        {
            "train_indices": list(receipt.train_indices),
            "test_indices": list(receipt.test_indices),
            "train_test_disjoint": set(receipt.train_indices).isdisjoint(receipt.test_indices),
            "temperature": receipt.temperature,
        }
        for receipt in calibration_folds
    ]
    checksum = _reproducibility_checksum(
        {
            "source": source,
            "model_specs": model_specs,
            "seed": seed,
            "folds": folds,
            "task_ids": [task.task_id for task in tasks],
            "raw_probabilities": raw_probabilities,
            "calibrated_probabilities": calibrated_probabilities,
        }
    )
    verdict = (
        "complete_stub_evaluation_no_live_inference"
        if dry_run
        else "complete_live_option_readout_evaluation"
    )
    if report.errors:
        verdict += "_with_reported_load_errors"
    artifact: dict[str, object] = {
        "schema": "carnot.jevbench_readout_eval.v1",
        "run_utc": datetime.now(UTC).isoformat(),
        "inference_substrate": inference_substrate,
        "model_specs": model_specs,
        "random_seed": seed,
        "reproducibility_checksum": checksum,
        "duration_s": time.perf_counter() - started,
        "n_per_split": n_per_split,
        "n_total": len(tasks),
        "verifier_is_oracle": False,
        "honest_verdict": verdict,
        "preconditions_checked": preconditions_checked,
        "methodology_note": (
            "The public tasks are a subset of 231 of 534 tasks chosen by the benchmark "
            "authors. This evaluation makes no claim about the held-out tasks."
        ),
        "source": source,
        "metrics": metrics,
        "temperature_scaling": {
            "method": "seeded K-fold out-of-fold scalar temperature scaling",
            "requested_folds": folds,
            "folds": fold_receipts,
            "no_row_calibrates_itself": all(
                receipt["train_test_disjoint"] for receipt in fold_receipts
            ),
        },
        "option_order_probe": {
            "overall": _option_probe_summary(probe.rows),
            "by_split": probe_by_split,
            "probabilities_mapped_back_by_option_id": True,
        },
        "readout_method": {
            "forward_passes_per_score_call": 0 if dry_run else 1,
            "score_calls_per_task": 2,
            "generation_used": False,
            "json_response_parsing_used": False,
        },
    }
    if gpu_evidence is not None:
        artifact["llama_cpp_gpu_offload_evidence"] = gpu_evidence
    return artifact


def default_output_path() -> Path:
    """Return the opt-in result path for the current UTC date."""

    date = datetime.now(UTC).date().isoformat()
    return Path("results") / f"jevbench_readout_eval_{date}.json"


def write_artifact(path: Path, artifact: Mapping[str, object]) -> None:
    """Write JSON atomically after the caller explicitly selects a path."""

    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS_DIR)
    parser.add_argument("--split", choices=(*SPLIT_NAMES, "all"), default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output",
        nargs="?",
        const=str(default_output_path()),
        default=None,
        help="Write JSON. With no value, use the UTC-dated results path.",
    )
    parser.add_argument("--model-gguf", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and write only when ``--output`` is present."""

    args = _parser().parse_args(argv)
    artifact = run_evaluation(
        tasks_dir=args.tasks_dir,
        split=args.split,
        limit=args.limit,
        dry_run=args.dry_run,
        model_gguf=args.model_gguf,
        seed=args.seed,
        folds=args.folds,
    )
    if args.output is not None:
        output_path = Path(args.output)
        write_artifact(output_path, artifact)
        print(f"artifact written: {output_path}", flush=True)
    else:
        print("artifact not written: --output was not supplied", flush=True)
    print(
        json.dumps(
            {
                "honest_verdict": artifact["honest_verdict"],
                "inference_substrate": artifact["inference_substrate"],
                "n_per_split": artifact["n_per_split"],
                "duration_s": artifact["duration_s"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

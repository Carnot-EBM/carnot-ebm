"""Exp 4381: BiPRM-style detector localization and abstention on cached FoVer.

Spec refs: REQ-VERIFY-4381, SCENARIO-VERIFY-4381.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot.experiment_4375_verifier_as_detector_measurement import (
    _registry_has_fover_ensemble,
    compute_detector_auroc,
    read_labeled_fover_rows,
    score_fover_production_ensemble,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = ROOT / "results" / "experiment_4381_biprm_detector_localization_abstention.json"
DETECTOR_CORPUS_PATH = ROOT / "data" / "fover_corpus.jsonl"
STEP_CORPUS_PATH = ROOT / "data" / "step_level_prm_training.jsonl"
REGISTRY_PATH = ROOT / "ops" / "verifier_registry.yaml"
EXP4375_ARTIFACT_PATH = ROOT / "results" / "experiment_4375_verifier_as_detector_measurement.json"
RANDOM_SEED = 4381
RANDOM_SEEDS_USED = (4381,)
BOOTSTRAP_RESAMPLES = 2500
MIN_TRACES = 1000
SPEC_REFS = ["REQ-VERIFY-4381", "SCENARIO-VERIFY-4381"]
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
FUSION_METHOD = "mean_l2r_r2l"
RANDOM_CONTROL_REPLICATES = 128


FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A win (bidirectional fusion localizes the earliest "
        "error better than unidirectional AND the detection score enables a "
        "useful abstention operating point -- the detector graduates 'beats "
        "chance'->'actionable') and a CLEAN null (unidirectional is already as "
        "good / abstention gives no useful point) are BOTH decision-grade."
    ),
    "detector_localization_actionable": (
        "BARE bool: true iff bidirectional fusion improves first-error "
        "localization over unidirectional L2R (CI95-excl-0) AND the "
        "risk-coverage curve yields a useful selective-prediction operating "
        "point -- the oracle-distinct detector graduating from 'beats chance' "
        "to 'localizes + abstains usefully'."
    ),
    "localization_f1_by_direction": (
        "dict {unidirectional_l2r, bidirectional_fusion, causal_online} -> "
        "first-error-step localization accuracy/F1 -- the causal_online number "
        "is the in-loop-achievable one (R2L-free)."
    ),
    "localization_delta_ci95": (
        "Bootstrap CI95 (>=2000 resamples) of (bidirectional - unidirectional) "
        "first-error localization -- excluding 0 is the decision-grade gain."
    ),
    "abstention_curve": (
        "The accuracy-vs-coverage / risk-coverage points + precision@recall=0.9 "
        "+ the selective risk -- the 'I don't know' operating point that makes "
        "detection actionable."
    ),
    "online_vs_offline_separated": (
        "BARE bool=true -- bidirectional R2L uses future context and is reported "
        "as OFFLINE post-hoc; causal/L2R-only is the ONLINE-actionable number."
    ),
    "n_traces": "BARE int: scored trace count -- MUST be >= 1000 for the claim.",
    "verifier_is_oracle": (
        "BARE bool=false -- a learned/energy detection signal scored against "
        "cached candidates; the executable label defines correctness."
    ),
    "preconditions_checked": (
        "Records cached detector corpus + ensemble + step-labels + TRM "
        "stand-down; pre-empts silent missing-resource fabrication."
    ),
    "random_seed": "Determinism precondition for scoring order, fusion, and bootstrap.",
    "reproducibility_checksum": (
        "Hash of the corpus + ensemble config + fusion method + localization/"
        "abstention computation; lets a third party re-run."
    ),
    "model_specs": (
        "Verifier ensemble + cached step-labeled FoVer corpus + BiPRM fusion "
        "config + unidirectional baseline + n; oracle-distinct declaration."
    ),
}


class NoStepLabelsError(ValueError):
    """Raised when a candidate localization corpus has no usable step labels."""


@dataclass(frozen=True)
class PreconditionCheck:
    """One resource checked before any localization scoring."""

    resource: str
    available: bool
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "resource": self.resource,
            "available": bool(self.available),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ScoreBundle:
    """Flat scores returned by a verifier scorer."""

    scores: list[float]
    per_verifier_scores: dict[str, list[float]]


@dataclass(frozen=True)
class StepTrace:
    """One cached reasoning trace with per-step error labels."""

    trace_id: str
    steps: tuple[dict[str, Any], ...]
    labels: tuple[int, ...]

    @property
    def first_error_index(self) -> int | None:
        for idx, label in enumerate(self.labels):
            if int(label) == 1:
                return idx
        return None

    @property
    def has_error(self) -> bool:
        return self.first_error_index is not None

    @property
    def error_class(self) -> str:
        idx = self.first_error_index
        row = self.steps[idx or 0]
        return str(
            row.get("error_axis")
            or row.get("problem_type")
            or row.get("source_domain")
            or row.get("source")
            or "untyped"
        )


@dataclass(frozen=True)
class ScoredTrace:
    """One trace after causal, retrospective, and fused detector scoring."""

    trace_id: str
    labels: tuple[int, ...]
    l2r_scores: tuple[float, ...]
    r2l_scores: tuple[float, ...]
    fused_scores: tuple[float, ...]
    error_class: str

    @property
    def first_error_index(self) -> int | None:
        for idx, label in enumerate(self.labels):
            if int(label) == 1:
                return idx
        return None

    @property
    def has_error(self) -> bool:
        return self.first_error_index is not None


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime configuration for Exp 4381."""

    repo_root: Path = ROOT
    detector_corpus_path: Path = DETECTOR_CORPUS_PATH
    step_corpus_path: Path = STEP_CORPUS_PATH
    registry_path: Path = REGISTRY_PATH
    exp4375_artifact_path: Path = EXP4375_ARTIFACT_PATH
    artifact_path: Path = ARTIFACT_PATH
    min_traces: int = MIN_TRACES
    random_seed: int = RANDOM_SEED
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at


Scorer = Callable[[list[dict[str, Any]], Path], Any]
AdversarialVerifyRunner = Callable[[Path], dict[str, Any]]


def round_float(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return round(float(value), digits)


def _read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(dict(json.loads(line)))
        return rows

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        return [dict(row) for row in payload["items"]]
    if isinstance(payload, list):
        flattened: list[dict[str, Any]] = []
        for item in payload:
            if isinstance(item, dict) and isinstance(item.get("step_labels"), list):
                trace_id = str(item.get("question_id") or item.get("trace_id") or len(flattened))
                for step in item["step_labels"]:
                    step_row = dict(step)
                    step_row.setdefault("trace_id", trace_id)
                    step_row.setdefault("question_id", item.get("question_id"))
                    flattened.append(step_row)
            else:
                flattened.append(dict(item))
        return flattened
    raise ValueError(f"unsupported step corpus shape: {path}")


def _has_label_key(row: dict[str, Any]) -> bool:
    return any(key in row for key in ("step_label", "label", "violation_detected", "is_error"))


def label_to_error(value: Any) -> int:
    """Map a per-step label to 1 for error and 0 for correct."""

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return int(value != 0)
    normalized = str(value).strip().lower()
    if normalized in {"wrong", "incorrect", "error", "bad", "violated", "violation", "false"}:
        return 1
    if normalized in {"correct", "ok", "valid", "true", "clean", "pass", "passed"}:
        return 0
    raise ValueError(f"unsupported step label {value!r}")


def _row_error_label(row: dict[str, Any]) -> int:
    if "step_label" in row:
        return label_to_error(row["step_label"])
    if "label" in row:
        return label_to_error(row["label"])
    if "violation_detected" in row:
        return label_to_error(row["violation_detected"])
    if "is_error" in row:
        return label_to_error(row["is_error"])
    raise NoStepLabelsError("row lacks step_label/label/violation_detected/is_error")


def _step_index(row: dict[str, Any], fallback: int) -> int:
    if "step_index" in row:
        return int(row["step_index"])
    match = re.search(r":step_(\d+)$", str(row.get("question_id", "")))
    if match:
        return int(match.group(1))
    return fallback


def _step_text(row: dict[str, Any]) -> str:
    return str(row.get("partial_cot") or row.get("step_text") or row.get("text") or "")


def _explicit_trace_rows(rows: Sequence[dict[str, Any]]) -> list[StepTrace]:
    grouped: dict[str, list[tuple[int, int, dict[str, Any]]]] = {}
    for order, row in enumerate(rows):
        trace_id = str(row.get("trace_id") or row.get("trajectory_id") or row.get("candidate_id"))
        grouped.setdefault(trace_id, []).append((_step_index(row, order), order, dict(row)))

    traces: list[StepTrace] = []
    for trace_id, items in grouped.items():
        ordered = [row for _idx, _order, row in sorted(items)]
        labels = tuple(_row_error_label(row) for row in ordered)
        traces.append(StepTrace(trace_id=trace_id, steps=tuple(ordered), labels=labels))
    return traces


def _implicit_trace_rows(rows: Sequence[dict[str, Any]]) -> list[StepTrace]:
    traces: list[StepTrace] = []
    current: list[dict[str, Any]] = []
    last_qid: str | None = None
    last_fraction: float | None = None
    last_full_correct: Any = object()
    trace_counter = 0

    def flush() -> None:
        nonlocal current, trace_counter
        if not current:  # pragma: no cover - defensive guard for direct future calls.
            return
        qid = str(current[0].get("question_id") or f"trace_{trace_counter}")
        labels = tuple(_row_error_label(row) for row in current)
        traces.append(
            StepTrace(
                trace_id=f"{qid}#{trace_counter}",
                steps=tuple(current),
                labels=labels,
            )
        )
        trace_counter += 1
        current = []

    for row in rows:
        qid = str(row.get("question_id") or "")
        fraction_raw = row.get("prefix_fraction")
        fraction = float(fraction_raw) if fraction_raw is not None else None
        full_correct = row.get("full_cot_correct")
        starts_new = False
        if current:
            starts_new = qid != last_qid
            if fraction is not None and last_fraction is not None and fraction <= last_fraction:
                starts_new = True
            if full_correct is not None and full_correct != last_full_correct:
                starts_new = True
        if starts_new:
            flush()
        current.append(dict(row))
        last_qid = qid
        last_fraction = fraction
        last_full_correct = full_correct
    flush()
    return traces


def load_step_labeled_traces_from_rows(rows: Sequence[dict[str, Any]]) -> list[StepTrace]:
    """Group cached rows into traces and derive each first-error index."""

    if not rows or not any(_has_label_key(dict(row)) for row in rows):
        raise NoStepLabelsError("no per-step labels found")
    if any("trace_id" in row or "trajectory_id" in row or "candidate_id" in row for row in rows):
        traces = _explicit_trace_rows(rows)
    else:
        traces = _implicit_trace_rows(rows)
    if not traces or not any(trace.steps for trace in traces):  # pragma: no cover
        raise NoStepLabelsError("no trace steps found")
    return traces


def load_step_labeled_traces(path: Path) -> list[StepTrace]:
    return load_step_labeled_traces_from_rows(_read_json_or_jsonl(path))


def _score_rows_for_trace(trace: StepTrace, *, direction: str) -> list[dict[str, Any]]:
    full_trace_text = _step_text(trace.steps[-1]) or "\n".join(_step_text(row) for row in trace.steps)
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(trace.steps):
        label = int(trace.labels[idx])
        if direction == "l2r":
            text = _step_text(row)
            score_hint = row.get("l2r_score")
        else:
            text = str(
                row.get("r2l_text")
                or (
                    "Current step under retrospective verification:\n"
                    f"{_step_text(row)}\n\nFull trace including future context:\n{full_trace_text}"
                )
            )
            score_hint = row.get("r2l_score")
        score_row = {
            "question_id": trace.trace_id,
            "step_text": text,
            "label": "incorrect" if label else "correct",
            "source": str(row.get("source") or "cached_step_labeled_fover"),
            "problem_type": str(row.get("error_axis") or row.get("problem_type") or "step_error"),
        }
        if score_hint is not None:
            score_row["score_hint"] = float(score_hint)
        rows.append(score_row)
    return rows


def _coerce_scores(bundle: Any) -> list[float]:
    return [float(score) for score in bundle.scores]


def _split_scores(lengths: Sequence[int], scores: Sequence[float]) -> list[tuple[float, ...]]:
    if sum(lengths) != len(scores):
        raise ValueError(f"score count mismatch: expected {sum(lengths)}, got {len(scores)}")
    out: list[tuple[float, ...]] = []
    cursor = 0
    for length in lengths:
        out.append(tuple(float(score) for score in scores[cursor : cursor + length]))
        cursor += length
    return out


def score_traces_bidirectionally(
    traces: Sequence[StepTrace],
    repo_root: Path,
    *,
    scorer: Scorer = score_fover_production_ensemble,
) -> list[ScoredTrace]:
    """Score every step with causal L2R and offline suffix-aware R2L contexts."""

    lengths = [len(trace.steps) for trace in traces]
    l2r_rows = [row for trace in traces for row in _score_rows_for_trace(trace, direction="l2r")]
    r2l_rows = [row for trace in traces for row in _score_rows_for_trace(trace, direction="r2l")]
    l2r_scores_by_trace = _split_scores(lengths, _coerce_scores(scorer(l2r_rows, repo_root)))
    r2l_scores_by_trace = _split_scores(lengths, _coerce_scores(scorer(r2l_rows, repo_root)))

    scored: list[ScoredTrace] = []
    for trace, l2r_scores, r2l_scores in zip(
        traces, l2r_scores_by_trace, r2l_scores_by_trace, strict=True
    ):
        fused = tuple(
            (float(l2r_score) + float(r2l_score)) / 2.0
            for l2r_score, r2l_score in zip(l2r_scores, r2l_scores, strict=True)
        )
        scored.append(
            ScoredTrace(
                trace_id=trace.trace_id,
                labels=trace.labels,
                l2r_scores=l2r_scores,
                r2l_scores=r2l_scores,
                fused_scores=fused,
                error_class=trace.error_class,
            )
        )
    return scored


def _argmax_first(scores: Sequence[float]) -> int:
    return max(range(len(scores)), key=lambda idx: float(scores[idx]))


def _localization_successes(scored_traces: Sequence[ScoredTrace]) -> dict[str, list[int]]:
    successes = {
        "unidirectional_l2r": [],
        "bidirectional_fusion": [],
        "causal_online": [],
    }
    for trace in scored_traces:
        first_error = trace.first_error_index
        if first_error is None:
            continue
        l2r_success = int(_argmax_first(trace.l2r_scores) == first_error)
        fused_success = int(_argmax_first(trace.fused_scores) == first_error)
        successes["unidirectional_l2r"].append(l2r_success)
        successes["causal_online"].append(l2r_success)
        successes["bidirectional_fusion"].append(fused_success)
    return successes


def localization_f1_by_direction(scored_traces: Sequence[ScoredTrace]) -> dict[str, dict[str, Any]]:
    """Return exact first-error localization accuracy and equivalent micro-F1."""

    successes = _localization_successes(scored_traces)
    result: dict[str, dict[str, Any]] = {}
    for direction, values in successes.items():
        n_error = len(values)
        accuracy = sum(values) / n_error if n_error else None
        result[direction] = {
            "accuracy": round_float(accuracy),
            "f1": round_float(accuracy),
            "n_error_traces": n_error,
            "exact_match_count": int(sum(values)),
        }
    return result


def bootstrap_localization_delta_ci95(
    scored_traces: Sequence[ScoredTrace],
    *,
    seed: int,
    resamples: int,
) -> list[float | None]:
    """Bootstrap paired bidirectional-minus-L2R first-error localization."""

    successes = _localization_successes(scored_traces)
    l2r = successes["unidirectional_l2r"]
    fused = successes["bidirectional_fusion"]
    if not l2r or len(l2r) != len(fused) or resamples <= 0:
        return [None, None]
    deltas = [float(bi) - float(uni) for bi, uni in zip(fused, l2r, strict=True)]
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(resamples):
        sample = [deltas[rng.randrange(len(deltas))] for _idx in range(len(deltas))]
        values.append(sum(sample) / len(sample))
    values.sort()
    lo = int(0.025 * (len(values) - 1))
    hi = int(0.975 * (len(values) - 1))
    return [round_float(values[lo]), round_float(values[hi])]


def _precision_at_recall(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    recall_target: float,
) -> float | None:
    positives = sum(1 for label in labels if int(label) == 1)
    if positives == 0 or len(labels) != len(scores):
        return None
    tp = 0
    fp = 0
    best_precision: float | None = None
    for idx in sorted(range(len(scores)), key=lambda item: float(scores[item]), reverse=True):
        if int(labels[idx]) == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / positives
        if recall >= recall_target:
            precision = tp / max(1, tp + fp)
            best_precision = precision if best_precision is None else max(best_precision, precision)
    return round_float(best_precision)


def _random_score_auroc_control(labels: Sequence[int], *, seed: int) -> float | None:
    if len(set(int(label) for label in labels)) < 2:
        return None
    rng = random.Random(seed)
    values = [
        compute_detector_auroc(labels, [rng.random() for _label in labels])
        for _replicate in range(RANDOM_CONTROL_REPLICATES)
    ]
    return sum(values) / len(values)


def build_abstention_curve(scored_traces: Sequence[ScoredTrace], *, seed: int) -> dict[str, Any]:
    """Build risk-coverage points by retaining the least suspicious traces."""

    labels = [1 if trace.has_error else 0 for trace in scored_traces]
    scores = [max(trace.fused_scores) for trace in scored_traces]
    n = len(scored_traces)
    correct_flags = [1 - label for label in labels]
    base_rate = sum(correct_flags) / max(1, n)
    detector_auroc = None
    random_control = None
    if len(set(labels)) >= 2:
        detector_auroc = compute_detector_auroc(labels, scores)
        random_control = _random_score_auroc_control(labels, seed=seed)

    points: list[dict[str, Any]] = []
    retain_order = sorted(range(n), key=lambda idx: float(scores[idx]))
    for target_coverage in (1.0, 0.9, 0.75, 0.5, 0.25):
        n_retained = max(1, min(n, int(round(n * target_coverage))))
        retained = retain_order[:n_retained]
        retained_accuracy = sum(correct_flags[idx] for idx in retained) / n_retained
        actual_coverage = n_retained / max(1, n)
        points.append(
            {
                "coverage": round_float(actual_coverage),
                "retained_accuracy": round_float(retained_accuracy),
                "selective_risk": round_float(1.0 - retained_accuracy),
                "n_retained": n_retained,
            }
        )

    useful_candidates = [
        point
        for point in points
        if point["coverage"] is not None
        and point["coverage"] >= 0.5
        and point["retained_accuracy"] is not None
        and point["retained_accuracy"] >= round_float(base_rate + 0.05)
    ]
    useful = max(
        useful_candidates,
        key=lambda point: (float(point["retained_accuracy"]), float(point["coverage"])),
        default=None,
    )
    return {
        "score_orientation": "higher_error_score_more_suspicious_reject_highest_scores",
        "base_rate_fraction_correct": round_float(base_rate),
        "detector_auroc": round_float(detector_auroc),
        "random_score_auroc_control": round_float(random_control),
        "random_score_auroc_control_replicates": RANDOM_CONTROL_REPLICATES,
        "precision_at_recall_0_9": _precision_at_recall(labels, scores, recall_target=0.9),
        "recall_target": 0.9,
        "points": points,
        "useful_operating_point": useful,
        "useful_criterion": "retained_accuracy >= base_rate + 0.05 at coverage >= 0.5",
    }


def _missing_verifier_gaps(scored_traces: Sequence[ScoredTrace]) -> list[dict[str, Any]]:
    misses = [
        trace.error_class
        for trace in scored_traces
        if trace.first_error_index is not None
        and _argmax_first(trace.fused_scores) != trace.first_error_index
    ]
    gaps: list[dict[str, Any]] = []
    for error_class, count in sorted(Counter(misses).items()):
        gaps.append(
            {
                "gap_id": f"GAP-FOVER-BIPRM-LOCALIZATION-{error_class}",
                "status": "open",
                "error_class": error_class,
                "missed_first_error_traces": int(count),
                "missing_discriminator": (
                    "A verifier feature that separates the earliest causal error "
                    "from later downstream consequences for this class."
                ),
            }
        )
    return gaps


def hash_sources(source_paths: Sequence[Path], *, payload: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for path in sorted({Path(path) for path in source_paths}, key=lambda item: str(item)):
        digest.update(str(path).encode("utf-8"))
        if not path.exists():
            digest.update(b"\0MISSING\0")
            continue
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return "sha256:" + digest.hexdigest()


def _model_specs(
    *,
    n_traces: int,
    n_error_traces: int,
    source_paths: Sequence[Path],
    bootstrap_resamples: int,
    random_seed: int,
) -> dict[str, Any]:
    return {
        "verifier_ensemble_id": "fover_production_ensemble",
        "ensemble_components": [
            "tier0r_curry_howard",
            "tier0u_logical_consistency",
            "fr11_session_memory",
        ],
        "cached_detector_corpus": str(source_paths[0]) if source_paths else str(DETECTOR_CORPUS_PATH),
        "cached_step_labeled_fover_corpus": (
            str(source_paths[1]) if len(source_paths) > 1 else str(STEP_CORPUS_PATH)
        ),
        "verifier_registry_path": str(source_paths[2]) if len(source_paths) > 2 else str(REGISTRY_PATH),
        "exp4375_detector_artifact": (
            str(source_paths[3]) if len(source_paths) > 3 else str(EXP4375_ARTIFACT_PATH)
        ),
        "n": int(n_traces),
        "n_error_traces": int(n_error_traces),
        "l2r_pass": "causal_prefix_only",
        "r2l_pass": "offline_suffix_aware_future_context",
        "fusion_method": FUSION_METHOD,
        "unidirectional_baseline": "l2r_argmax_step_error_score",
        "online_actionable_direction": "causal_l2r_only",
        "offline_posthoc_direction": "bidirectional_fusion_l2r_plus_r2l",
        "bootstrap_resamples": int(bootstrap_resamples),
        "random_seed": int(random_seed),
        "random_seeds_used": list(RANDOM_SEEDS_USED),
        "trm_training": "stood_down_not_invoked",
        "live_generation": False,
        "verifier_is_oracle": False,
    }


def build_complete_artifact(
    *,
    scored_traces: Sequence[ScoredTrace],
    preconditions_checked: list[dict[str, Any]],
    source_paths: Sequence[Path],
    duration_s: float,
    random_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    localization = localization_f1_by_direction(scored_traces)
    delta_ci95 = bootstrap_localization_delta_ci95(
        scored_traces,
        seed=random_seed,
        resamples=bootstrap_resamples,
    )
    abstention = build_abstention_curve(scored_traces, seed=random_seed)
    ci_excludes_zero = bool(delta_ci95[0] is not None and float(delta_ci95[0]) > 0.0)
    useful_abstention = abstention.get("useful_operating_point") is not None
    actionable = bool(ci_excludes_zero and useful_abstention)
    n_error_traces = sum(1 for trace in scored_traces if trace.has_error)
    verdict = (
        "complete: detector_localization_actionable_bidirectional_gain_and_abstention"
        if actionable
        else "complete: clean_powered_null_bidirectional_not_actionable"
    )
    checksum_payload = {
        "localization_f1_by_direction": localization,
        "localization_delta_ci95": delta_ci95,
        "abstention_curve": abstention,
        "n_traces": len(scored_traces),
        "fusion_method": FUSION_METHOD,
        "random_seed": random_seed,
        "bootstrap_resamples": bootstrap_resamples,
    }
    return {
        "experiment": "experiment_4381_biprm_detector_localization_abstention",
        "schema": "carnot.biprm_detector_localization_abstention.v1",
        "honest_verdict": verdict,
        "detector_localization_actionable": actionable,
        "localization_f1_by_direction": localization,
        "localization_delta_ci95": delta_ci95,
        "abstention_curve": abstention,
        "online_vs_offline_separated": True,
        "n_traces": int(len(scored_traces)),
        "n_error_traces": int(n_error_traces),
        "verifier_is_oracle": False,
        "preconditions_checked": preconditions_checked,
        "random_seed": int(random_seed),
        "random_seeds_used": list(RANDOM_SEEDS_USED),
        "bootstrap_resamples": int(bootstrap_resamples),
        "reproducibility_checksum": hash_sources(source_paths, payload=checksum_payload),
        "model_specs": _model_specs(
            n_traces=len(scored_traces),
            n_error_traces=n_error_traces,
            source_paths=source_paths,
            bootstrap_resamples=bootstrap_resamples,
            random_seed=random_seed,
        ),
        "missing_verifier_gaps": _missing_verifier_gaps(scored_traces),
        "methodology_note": (
            "R2L scores use future context and are offline post-hoc only; the "
            "online-actionable number is the causal L2R baseline."
        ),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "duration_s": round_float(duration_s, 3),
        "inference_substrate": INFERENCE_SUBSTRATE,
    }


def build_blocked_artifact(
    *,
    honest_verdict: str,
    preconditions_checked: list[dict[str, Any]],
    source_paths: Sequence[Path],
    duration_s: float,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_4381_biprm_detector_localization_abstention",
        "schema": "carnot.biprm_detector_localization_abstention.v1",
        "honest_verdict": honest_verdict,
        "detector_localization_actionable": False,
        "localization_f1_by_direction": {},
        "localization_delta_ci95": [None, None],
        "abstention_curve": {},
        "online_vs_offline_separated": True,
        "n_traces": 0,
        "n_error_traces": 0,
        "verifier_is_oracle": False,
        "preconditions_checked": preconditions_checked,
        "random_seed": RANDOM_SEED,
        "random_seeds_used": list(RANDOM_SEEDS_USED),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "reproducibility_checksum": hash_sources(source_paths, payload={"blocked": honest_verdict}),
        "model_specs": {
            "verifier_ensemble_id": "fover_production_ensemble",
            "fusion_method": FUSION_METHOD,
            "online_actionable_direction": "causal_l2r_only",
            "trm_training": "stood_down_not_invoked",
            "live_generation": False,
            "verifier_is_oracle": False,
        },
        "missing_verifier_gaps": [],
        "methodology_note": "blocked before scoring; no localization metrics fabricated",
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "duration_s": round_float(duration_s, 3),
        "inference_substrate": INFERENCE_SUBSTRATE,
    }


def _detector_corpus_check(path: Path) -> PreconditionCheck:
    if not path.is_file():
        return PreconditionCheck("exp4375_cached_detector_corpus", False, "missing")
    try:
        rows = read_labeled_fover_rows(path)
    except Exception as exc:
        return PreconditionCheck("exp4375_cached_detector_corpus", False, f"unreadable: {exc}")
    if not rows:
        return PreconditionCheck("exp4375_cached_detector_corpus", False, "empty")
    return PreconditionCheck("exp4375_cached_detector_corpus", True, f"labeled_rows={len(rows)}")


def _step_corpus_checks(path: Path, min_traces: int) -> list[PreconditionCheck]:
    if not path.is_file():
        return [
            PreconditionCheck("cached_step_labeled_fover_corpus", False, "missing"),
            PreconditionCheck("step_labels_for_localization", False, "missing corpus"),
        ]
    try:
        traces = load_step_labeled_traces(path)
    except NoStepLabelsError as exc:
        return [
            PreconditionCheck("cached_step_labeled_fover_corpus", True, "readable"),
            PreconditionCheck("step_labels_for_localization", False, str(exc)),
        ]
    except Exception as exc:
        return [
            PreconditionCheck("cached_step_labeled_fover_corpus", False, f"unreadable: {exc}"),
            PreconditionCheck("step_labels_for_localization", False, "unreadable corpus"),
        ]
    n_error = sum(1 for trace in traces if trace.has_error)
    corpus_available = len(traces) >= min_traces
    return [
        PreconditionCheck(
            "cached_step_labeled_fover_corpus",
            corpus_available,
            f"traces={len(traces)}; required>={min_traces}; error_traces={n_error}",
        ),
        PreconditionCheck(
            "step_labels_for_localization",
            n_error > 0,
            f"first_error_indices_derivable={n_error > 0}; error_traces={n_error}",
        ),
    ]


def _scoring_path_loads() -> bool:
    score_fover_production_ensemble(
        [{"step_text": "Compute 1+1=2.", "label": "correct", "question_id": "probe"}],
        ROOT,
    )
    return True


def check_preconditions(
    *,
    detector_corpus_path: Path,
    step_corpus_path: Path,
    registry_path: Path,
    exp4375_artifact_path: Path,
    min_traces: int,
    scoring_path_checker: Callable[[], bool] = _scoring_path_loads,
) -> list[PreconditionCheck]:
    checks = [_detector_corpus_check(detector_corpus_path)]
    checks.append(
        PreconditionCheck(
            "exp4375_detector_artifact",
            exp4375_artifact_path.is_file(),
            str(exp4375_artifact_path) if exp4375_artifact_path.is_file() else "missing",
        )
    )
    registry_ok = _registry_has_fover_ensemble(registry_path)
    checks.append(
        PreconditionCheck(
            "verifier_registry",
            registry_ok,
            "fover_production_ensemble present" if registry_ok else "missing fover_production_ensemble",
        )
    )
    checks.extend(_step_corpus_checks(step_corpus_path, min_traces))
    try:
        scoring_ok = bool(scoring_path_checker())
        scoring_detail = "production FoVer scoring path imports and scores a probe row"
    except Exception as exc:
        scoring_ok = False
        scoring_detail = f"scoring path failed: {exc}"
    checks.append(PreconditionCheck("fover_scoring_path", scoring_ok, scoring_detail))
    checks.append(
        PreconditionCheck(
            "trm_training_stand_down",
            True,
            "not invoked; this experiment scores cached candidates only",
        )
    )
    return checks


def _blocked_reason(checks: Sequence[PreconditionCheck]) -> str | None:
    if all(check.available for check in checks):
        return None
    by_resource = {check.resource: check for check in checks}
    step_labels = by_resource.get("step_labels_for_localization")
    cached_step = by_resource.get("cached_step_labeled_fover_corpus")
    if (
        step_labels is not None
        and not step_labels.available
        and cached_step is not None
        and cached_step.available
    ):
        return "blocked_no_step_labels_for_localization"
    return "blocked_cached_step_labeled_corpus_unavailable"


def run_adversarial_verify(path: Path, repo_root: Path = ROOT) -> dict[str, Any]:
    script = repo_root / "scripts" / "adversarial_verify.py"
    if not script.is_file():
        return {"returncode": None, "flags": [], "stderr": "scripts/adversarial_verify.py missing"}
    proc = subprocess.run(
        [sys.executable, str(script), str(path)],
        cwd=str(repo_root),
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    return {
        "returncode": int(proc.returncode),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def _write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    scorer: Scorer = score_fover_production_ensemble,
    scoring_path_checker: Callable[[], bool] = _scoring_path_loads,
    adversarial_verify_runner: AdversarialVerifyRunner = run_adversarial_verify,
    write: bool = True,
) -> dict[str, Any]:
    cfg = config or ExperimentConfig()
    started = cfg.start_time()
    source_paths = [
        cfg.detector_corpus_path,
        cfg.step_corpus_path,
        cfg.registry_path,
        cfg.exp4375_artifact_path,
    ]
    checks = check_preconditions(
        detector_corpus_path=cfg.detector_corpus_path,
        step_corpus_path=cfg.step_corpus_path,
        registry_path=cfg.registry_path,
        exp4375_artifact_path=cfg.exp4375_artifact_path,
        min_traces=cfg.min_traces,
        scoring_path_checker=scoring_path_checker,
    )
    preconditions = [check.as_dict() for check in checks]
    blocked = _blocked_reason(checks)
    if blocked is not None:
        artifact = build_blocked_artifact(
            honest_verdict=blocked,
            preconditions_checked=preconditions,
            source_paths=source_paths,
            duration_s=cfg.clock() - started,
        )
        if write:
            _write_artifact(cfg.artifact_path, artifact)
        return artifact

    traces = load_step_labeled_traces(cfg.step_corpus_path)
    scored_traces = score_traces_bidirectionally(traces, cfg.repo_root, scorer=scorer)
    artifact = build_complete_artifact(
        scored_traces=scored_traces,
        preconditions_checked=preconditions,
        source_paths=source_paths,
        duration_s=cfg.clock() - started,
        random_seed=cfg.random_seed,
        bootstrap_resamples=cfg.bootstrap_resamples,
    )
    if write:
        _write_artifact(cfg.artifact_path, artifact)
        artifact["adversarial_verify"] = adversarial_verify_runner(cfg.artifact_path)
        _write_artifact(cfg.artifact_path, artifact)
    else:
        artifact["adversarial_verify"] = {"returncode": None, "skipped": True}
    return artifact


def main() -> int:
    artifact = run_experiment(write=True)
    print(
        "[exp4381] "
        f"{artifact['honest_verdict']} "
        f"actionable={artifact['detector_localization_actionable']} "
        f"n_traces={artifact['n_traces']} -> {ARTIFACT_PATH}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

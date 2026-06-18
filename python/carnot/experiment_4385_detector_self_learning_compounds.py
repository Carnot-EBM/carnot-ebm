"""Exp 4385: detector self-learning compounding curve on cached FoVer traces.

Spec refs: REQ-VERIFY-4385, SCENARIO-VERIFY-4385.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot.experiment_4375_verifier_as_detector_measurement import (
    compute_detector_auroc,
    read_labeled_fover_rows,
)
from carnot.experiment_4381_biprm_detector_localization_abstention import (
    EXP4375_ARTIFACT_PATH,
    DETECTOR_CORPUS_PATH,
    REGISTRY_PATH,
    STEP_CORPUS_PATH,
    NoStepLabelsError,
    ScoreBundle,
    ScoredTrace,
    StepTrace,
    _registry_has_fover_ensemble,
    _scoring_path_loads,
    load_step_labeled_traces,
    score_fover_production_ensemble,
    score_traces_bidirectionally,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = ROOT / "results" / "experiment_4385_detector_self_learning_compounds.json"
EXP4381_ARTIFACT_PATH = (
    ROOT / "results" / "experiment_4381_biprm_detector_localization_abstention.json"
)
RANDOM_SEED = 4385
RANDOM_SEEDS_USED = (4385,)
BOOTSTRAP_RESAMPLES = 2500
MIN_HELD_OUT_TRACES = 1000
HELD_OUT_FRACTION = 0.25
HEADROOM_EPSILON = 0.02
SPEC_REFS = ["REQ-VERIFY-4385", "SCENARIO-VERIFY-4385"]
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "detector_compounds",
    "learning_curve",
    "no_learning_baseline",
    "positive_control_passed",
    "compounding_delta_ci95",
    "verifier_is_oracle",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "model_specs",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A win (the detector COMPOUNDS -- held-out metric "
        "rises with data beyond the no-learning baseline) and a CLEAN null "
        "(saturated on FoVer, positive control confirms no headroom -> "
        "cross-domain is the fresh-headroom direction) are BOTH decision-grade."
    ),
    "detector_compounds": (
        "BARE bool: the capstone reads this; true iff held-out detection "
        "localization-F1 rises with accumulated-corpus size BEYOND the "
        "no-learning baseline (delta CI95-excl-0) AND the positive control "
        "confirms the curve is not flat-because-saturated."
    ),
    "learning_curve": (
        "list of {train_corpus_size, held_out_localization_f1, held_out_auroc, "
        "held_out_selective_risk} -- the compounding curve (the exp4364 "
        "discipline, on the detector)."
    ),
    "no_learning_baseline": (
        "BARE float: the fixed-threshold / equal-weight detector held-out "
        "localization-F1 -- the floor the learning curve must beat."
    ),
    "positive_control_passed": (
        "BARE bool: true iff the from-scratch full-corpus detector confirms "
        "non-trivial headroom over the no-learning floor."
    ),
    "compounding_delta_ci95": (
        "Bootstrap CI95 of the final-minus-initial held-out localization-F1 -- "
        "excluding 0 is the decision-grade compounding signal."
    ),
    "verifier_is_oracle": (
        "BARE bool=false -- the detector is a learned/energy signal, "
        "oracle-distinct."
    ),
    "preconditions_checked": (
        "Records the detector + corpus + ensemble + TRM-stand-down verified; "
        "pre-empts the silent-missing-resource fabrication mode."
    ),
    "random_seed": "Determinism precondition for the split, fitting, and bootstrap.",
    "reproducibility_checksum": (
        "Hash of the corpus split + fitted thresholds/weights + learning curve; "
        "lets a third party re-run."
    ),
    "model_specs": (
        "The verifier ensemble + corpus + train/held-out split + no-learning + "
        "positive-control baselines + n; required methodology."
    ),
}


@dataclass(frozen=True)
class PreconditionCheck:
    """One resource checked before detector fitting starts."""

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
class DetectorFit:
    """Fitted detector operating point for a corpus prefix."""

    weight_l2r: float
    weight_r2l: float
    threshold: float
    train_corpus_size: int
    train_localization_f1: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "weight_l2r": round_float(self.weight_l2r),
            "weight_r2l": round_float(self.weight_r2l),
            "threshold": round_float(self.threshold),
            "train_corpus_size": int(self.train_corpus_size),
            "train_localization_f1": round_float(self.train_localization_f1),
        }


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime configuration for Exp 4385."""

    repo_root: Path = ROOT
    detector_corpus_path: Path = DETECTOR_CORPUS_PATH
    step_corpus_path: Path = STEP_CORPUS_PATH
    registry_path: Path = REGISTRY_PATH
    exp4381_artifact_path: Path = EXP4381_ARTIFACT_PATH
    exp4375_artifact_path: Path = EXP4375_ARTIFACT_PATH
    artifact_path: Path = ARTIFACT_PATH
    min_held_out_traces: int = MIN_HELD_OUT_TRACES
    held_out_fraction: float = HELD_OUT_FRACTION
    random_seed: int = RANDOM_SEED
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at


Scorer = Callable[[list[dict[str, Any]], Path], ScoreBundle]
AdversarialVerifyRunner = Callable[[Path], dict[str, Any]]


def round_float(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return round(float(value), digits)


def no_learning_fit() -> DetectorFit:
    """Fixed-threshold equal-weight detector baseline."""

    return DetectorFit(
        weight_l2r=0.5,
        weight_r2l=0.5,
        threshold=0.5,
        train_corpus_size=0,
        train_localization_f1=None,
    )


def _fused_scores(trace: ScoredTrace, fit: DetectorFit) -> tuple[float, ...]:
    return tuple(
        fit.weight_l2r * float(l2r) + fit.weight_r2l * float(r2l)
        for l2r, r2l in zip(trace.l2r_scores, trace.r2l_scores, strict=True)
    )


def _first_error_index(labels: Sequence[int]) -> int | None:
    for idx, label in enumerate(labels):
        if int(label) == 1:
            return idx
    return None


def _predicted_index(scores: Sequence[float], threshold: float) -> int | None:
    for idx, score in enumerate(scores):
        if float(score) >= float(threshold):
            return idx
    return None


def _localization_counts(
    traces: Sequence[ScoredTrace],
    fit: DetectorFit,
) -> tuple[int, int, int]:
    tp = fp = fn = 0
    for trace in traces:
        first_error = _first_error_index(trace.labels)
        predicted = _predicted_index(_fused_scores(trace, fit), fit.threshold)
        if first_error is None:
            if predicted is not None:
                fp += 1
            continue
        if predicted is None:
            fn += 1
        elif predicted == first_error:
            tp += 1
        else:
            fp += 1
            fn += 1
    return tp, fp, fn


def _localization_f1(traces: Sequence[ScoredTrace], fit: DetectorFit) -> float:
    tp, fp, fn = _localization_counts(traces, fit)
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom else 0.0


def compute_auroc_safe(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    if len(labels) != len(scores) or len(set(int(label) for label in labels)) < 2:
        return None
    return float(compute_detector_auroc(labels, scores))


def selective_risk_at_threshold(
    traces: Sequence[ScoredTrace],
    fit: DetectorFit,
) -> float | None:
    retained: list[int] = []
    for trace in traces:
        score = max(_fused_scores(trace, fit))
        if score < fit.threshold:
            retained.append(1 if _first_error_index(trace.labels) is not None else 0)
    if not retained:
        return None
    return sum(retained) / len(retained)


def evaluate_detector_fit(
    traces: Sequence[ScoredTrace],
    fit: DetectorFit,
) -> dict[str, float | None]:
    labels = [1 if _first_error_index(trace.labels) is not None else 0 for trace in traces]
    scores = [max(_fused_scores(trace, fit)) for trace in traces]
    selective_risk = selective_risk_at_threshold(traces, fit)
    coverage = None
    if traces:
        retained = sum(1 for trace in traces if max(_fused_scores(trace, fit)) < fit.threshold)
        coverage = retained / len(traces)
    return {
        "localization_f1": round_float(_localization_f1(traces, fit)),
        "auroc": round_float(compute_auroc_safe(labels, scores)),
        "selective_risk": round_float(selective_risk),
        "coverage": round_float(coverage),
    }


def _candidate_thresholds(traces: Sequence[ScoredTrace], weight_l2r: float) -> list[float]:
    weight_r2l = 1.0 - weight_l2r
    values = sorted(
        {
            weight_l2r * float(l2r) + weight_r2l * float(r2l)
            for trace in traces
            for l2r, r2l in zip(trace.l2r_scores, trace.r2l_scores, strict=True)
        }
    )
    if not values:
        return [0.5]
    if len(values) > 25:
        indexes = {
            int(round(idx * (len(values) - 1) / 24))
            for idx in range(25)
        }
        values = [values[idx] for idx in sorted(indexes)]
    return values + [values[-1] + 1e-9]


def fit_detector_on_traces(traces: Sequence[ScoredTrace]) -> DetectorFit:
    """Fit BiPRM fusion weight and operating threshold on cached labels."""

    if not traces:
        return no_learning_fit()
    best_fit: DetectorFit | None = None
    best_key: tuple[float, float, float, float] | None = None
    for step in range(11):
        weight_l2r = step / 10.0
        probe_fit = DetectorFit(
            weight_l2r=weight_l2r,
            weight_r2l=1.0 - weight_l2r,
            threshold=0.0,
            train_corpus_size=len(traces),
        )
        labels = [1 if _first_error_index(trace.labels) is not None else 0 for trace in traces]
        scores = [max(_fused_scores(trace, probe_fit)) for trace in traces]
        auroc = float(compute_auroc_safe(labels, scores) or -1.0)
        for threshold in _candidate_thresholds(traces, weight_l2r):
            fit = DetectorFit(
                weight_l2r=weight_l2r,
                weight_r2l=1.0 - weight_l2r,
                threshold=float(threshold),
                train_corpus_size=len(traces),
            )
            f1 = _localization_f1(traces, fit)
            risk = selective_risk_at_threshold(traces, fit)
            risk_value = float(risk if risk is not None else 1.0)
            key = (f1, auroc, -risk_value, threshold)
            if best_key is None or key > best_key:
                best_key = key
                best_fit = DetectorFit(
                    weight_l2r=weight_l2r,
                    weight_r2l=1.0 - weight_l2r,
                    threshold=float(threshold),
                    train_corpus_size=len(traces),
                    train_localization_f1=f1,
                )
    return best_fit if best_fit is not None else no_learning_fit()


def _prefix_sizes(
    total: int,
    *,
    prefix_fractions: Sequence[float],
    min_prefix_size: int,
) -> list[int]:
    sizes: list[int] = []
    for fraction in prefix_fractions:
        size = max(int(min_prefix_size), int(round(total * float(fraction))))
        size = min(max(1, size), total)
        if size not in sizes:
            sizes.append(size)
    if total not in sizes:
        sizes.append(total)
    return sizes


def build_learning_curve(
    train_stream: Sequence[ScoredTrace],
    held_out: Sequence[ScoredTrace],
    *,
    prefix_fractions: Sequence[float] = (0.10, 0.25, 0.50, 1.0),
    min_prefix_size: int = 10,
) -> tuple[list[dict[str, Any]], list[DetectorFit]]:
    """Fit on accumulating train prefixes and evaluate fixed held-out traces."""

    curve: list[dict[str, Any]] = []
    fits: list[DetectorFit] = []
    for size in _prefix_sizes(
        len(train_stream),
        prefix_fractions=prefix_fractions,
        min_prefix_size=min_prefix_size,
    ):
        fit = fit_detector_on_traces(train_stream[:size])
        metrics = evaluate_detector_fit(held_out, fit)
        curve.append(
            {
                "train_corpus_size": int(size),
                "held_out_localization_f1": float(metrics["localization_f1"] or 0.0),
                "held_out_auroc": round_float(metrics["auroc"]),
                "held_out_selective_risk": float(
                    metrics["selective_risk"] if metrics["selective_risk"] is not None else 1.0
                ),
            }
        )
        fits.append(fit)
    return curve, fits


def bootstrap_compounding_delta_ci95(
    held_out: Sequence[ScoredTrace],
    initial_fit: DetectorFit,
    final_fit: DetectorFit,
    *,
    seed: int,
    resamples: int,
) -> list[float | None]:
    """Bootstrap final-minus-initial held-out localization-F1."""

    if not held_out or resamples <= 0:
        return [None, None]
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(resamples):
        sample = [held_out[rng.randrange(len(held_out))] for _idx in range(len(held_out))]
        delta = _localization_f1(sample, final_fit) - _localization_f1(sample, initial_fit)
        values.append(delta)
    values.sort()
    lo = int(0.025 * (len(values) - 1))
    hi = int(0.975 * (len(values) - 1))
    return [round_float(values[lo]), round_float(values[hi])]


def positive_control_summary(
    *,
    held_out_metric: float,
    no_learning_baseline: float,
    min_headroom: float = HEADROOM_EPSILON,
) -> dict[str, Any]:
    headroom = float(held_out_metric) - float(no_learning_baseline)
    return {
        "positive_control_passed": bool(headroom > min_headroom),
        "ceiling_held_out_localization_f1": round_float(held_out_metric),
        "headroom_over_no_learning": round_float(headroom),
        "minimum_nontrivial_headroom": float(min_headroom),
    }


def summarize_compounding_curve(
    learning_curve: Sequence[dict[str, Any]],
    *,
    no_learning_baseline: float,
    positive_control_passed: bool,
    compounding_delta_ci95: Sequence[float | None],
) -> dict[str, Any]:
    if not learning_curve:
        final_metric = 0.0
    else:
        final_metric = float(learning_curve[-1].get("held_out_localization_f1") or 0.0)
    ci_positive = bool(compounding_delta_ci95[0] is not None and float(compounding_delta_ci95[0]) > 0.0)
    final_beats_baseline = bool(final_metric > float(no_learning_baseline))
    return {
        "detector_compounds": bool(
            positive_control_passed and final_beats_baseline and ci_positive
        ),
        "final_beats_no_learning_baseline": final_beats_baseline,
        "compounding_ci95_excludes_zero": ci_positive,
    }


def split_train_heldout(
    scored_traces: Sequence[ScoredTrace],
    *,
    seed: int,
    min_held_out_traces: int,
    held_out_fraction: float = HELD_OUT_FRACTION,
) -> tuple[list[ScoredTrace], list[ScoredTrace]]:
    if len(scored_traces) < 2:
        raise ValueError("need at least two traces for train/held-out split")
    if len(scored_traces) <= min_held_out_traces:
        raise ValueError("need more traces than the minimum held-out size")
    traces = list(scored_traces)
    rng = random.Random(seed)
    rng.shuffle(traces)
    held_out_count = max(int(min_held_out_traces), int(round(len(traces) * held_out_fraction)))
    held_out_count = min(held_out_count, len(traces) - 1)
    return traces[held_out_count:], traces[:held_out_count]


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


def _json_artifact_check(path: Path, resource: str, required_key: str) -> PreconditionCheck:
    if not path.is_file():
        return PreconditionCheck(resource, False, "missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return PreconditionCheck(resource, False, f"unreadable: {exc}")
    return PreconditionCheck(
        resource,
        required_key in payload,
        f"{required_key} present" if required_key in payload else f"missing {required_key}",
    )


def _step_corpus_check(path: Path, min_held_out_traces: int) -> PreconditionCheck:
    if not path.is_file():
        return PreconditionCheck("cached_step_labeled_fover_corpus", False, "missing")
    try:
        traces = load_step_labeled_traces(path)
    except NoStepLabelsError as exc:
        return PreconditionCheck("cached_step_labeled_fover_corpus", False, str(exc))
    except Exception as exc:
        return PreconditionCheck("cached_step_labeled_fover_corpus", False, f"unreadable: {exc}")
    labels = [1 if trace.first_error_index is not None else 0 for trace in traces]
    required_total = int(min_held_out_traces) + 1
    if len(traces) < required_total:
        return PreconditionCheck(
            "cached_step_labeled_fover_corpus",
            False,
            f"traces={len(traces)}; required>={required_total}",
        )
    if len(set(labels)) < 2:
        return PreconditionCheck(
            "cached_step_labeled_fover_corpus",
            False,
            f"traces={len(traces)}; needs both clean and error traces",
        )
    return PreconditionCheck(
        "cached_step_labeled_fover_corpus",
        True,
        (
            f"traces={len(traces)}; held_out_min={min_held_out_traces}; "
            f"error_traces={sum(labels)}"
        ),
    )


def check_preconditions(
    *,
    detector_corpus_path: Path,
    step_corpus_path: Path,
    registry_path: Path,
    exp4381_artifact_path: Path,
    exp4375_artifact_path: Path,
    min_held_out_traces: int,
    scoring_path_checker: Callable[[], bool] = _scoring_path_loads,
) -> list[PreconditionCheck]:
    checks = [
        _detector_corpus_check(detector_corpus_path),
        _step_corpus_check(step_corpus_path, min_held_out_traces),
        _json_artifact_check(exp4381_artifact_path, "exp4381_detector_config", "model_specs"),
        _json_artifact_check(exp4375_artifact_path, "exp4375_detector_baseline", "detector_auroc"),
    ]
    registry_ok = _registry_has_fover_ensemble(registry_path)
    checks.append(
        PreconditionCheck(
            "verifier_registry",
            registry_ok,
            "fover_production_ensemble present" if registry_ok else "missing fover_production_ensemble",
        )
    )
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
            "not invoked; threshold/fusion fitting against cached candidates only",
        )
    )
    return checks


def _blocked_reason(checks: Sequence[PreconditionCheck]) -> str | None:
    if all(check.available for check in checks):
        return None
    return "blocked_detector_or_corpus_unavailable"


def _model_specs(
    *,
    source_paths: Sequence[Path],
    split_spec: dict[str, Any],
    fitted_configs: Sequence[DetectorFit],
    no_learning_metrics: dict[str, Any],
    positive_control: dict[str, Any],
    bootstrap_resamples: int,
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
        "exp4381_detector_config": (
            str(source_paths[3]) if len(source_paths) > 3 else str(EXP4381_ARTIFACT_PATH)
        ),
        "exp4375_detector_baseline": (
            str(source_paths[4]) if len(source_paths) > 4 else str(EXP4375_ARTIFACT_PATH)
        ),
        "split": split_spec,
        "train_trace_count": int(split_spec.get("train_trace_count", 0)),
        "held_out_trace_count": int(split_spec.get("held_out_trace_count", 0)),
        "held_out_minimum_required": int(split_spec.get("held_out_minimum_required", 0)),
        "fitted_detector_configs": [fit.as_dict() for fit in fitted_configs],
        "no_learning_baseline": {
            "fit": no_learning_fit().as_dict(),
            "metrics": no_learning_metrics,
        },
        "positive_control": positive_control,
        "primary_metric": "held_out_localization_f1",
        "auroc_reported_as_secondary_detection_metric": True,
        "selective_risk_reported_at_fit_threshold": True,
        "bootstrap_resamples": int(bootstrap_resamples),
        "random_seed": RANDOM_SEED,
        "random_seeds_used": list(RANDOM_SEEDS_USED),
        "trm_training": "stood_down_not_invoked",
        "live_generation": False,
        "verifier_is_oracle": False,
    }


def build_blocked_artifact(
    *,
    preconditions_checked: list[dict[str, Any]],
    source_paths: Sequence[Path],
    duration_s: float,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_4385_detector_self_learning_compounds",
        "schema": "carnot.detector_self_learning_compounds.v1",
        "honest_verdict": "blocked_detector_or_corpus_unavailable",
        "detector_compounds": False,
        "learning_curve": [],
        "no_learning_baseline": 0.0,
        "positive_control_passed": False,
        "compounding_delta_ci95": [None, None],
        "verifier_is_oracle": False,
        "preconditions_checked": preconditions_checked,
        "random_seed": RANDOM_SEED,
        "random_seeds_used": list(RANDOM_SEEDS_USED),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "reproducibility_checksum": hash_sources(
            source_paths,
            payload={"blocked": "blocked_detector_or_corpus_unavailable"},
        ),
        "model_specs": {
            "blocked_reason": "detector_or_corpus_unavailable",
            "trm_training": "stood_down_not_invoked",
            "live_generation": False,
            "verifier_is_oracle": False,
        },
        "fresh_headroom_direction": "cross_domain_detection_exp4386",
        "methodology_note": "blocked before detector fitting; no learning-curve metrics fabricated",
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "duration_s": round_float(duration_s, 3),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "adversarial_verify": {"status": "not_run_blocked_preconditions"},
    }


def build_complete_artifact(
    *,
    learning_curve: Sequence[dict[str, Any]],
    fitted_configs: Sequence[DetectorFit],
    no_learning_baseline: float,
    no_learning_metrics: dict[str, Any],
    positive_control: dict[str, Any],
    compounding_delta_ci95: Sequence[float | None],
    preconditions_checked: list[dict[str, Any]],
    source_paths: Sequence[Path],
    split_spec: dict[str, Any],
    duration_s: float,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    summary = summarize_compounding_curve(
        learning_curve,
        no_learning_baseline=no_learning_baseline,
        positive_control_passed=bool(positive_control.get("positive_control_passed")),
        compounding_delta_ci95=compounding_delta_ci95,
    )
    if summary["detector_compounds"]:
        verdict = "success: detector_compounds_heldout_localization_f1"
    elif not positive_control.get("positive_control_passed"):
        verdict = "complete: clean_saturated_null_fover_detector"
    else:
        verdict = "complete: detector_plateau_with_positive_control"
    checksum_payload = {
        "learning_curve": list(learning_curve),
        "fitted_configs": [fit.as_dict() for fit in fitted_configs],
        "no_learning_baseline": no_learning_baseline,
        "positive_control": positive_control,
        "compounding_delta_ci95": list(compounding_delta_ci95),
        "split_spec": split_spec,
        "random_seed": RANDOM_SEED,
    }
    return {
        "experiment": "experiment_4385_detector_self_learning_compounds",
        "schema": "carnot.detector_self_learning_compounds.v1",
        "honest_verdict": verdict,
        "detector_compounds": bool(summary["detector_compounds"]),
        "learning_curve": [dict(point) for point in learning_curve],
        "no_learning_baseline": float(no_learning_baseline),
        "positive_control_passed": bool(positive_control.get("positive_control_passed")),
        "compounding_delta_ci95": list(compounding_delta_ci95),
        "verifier_is_oracle": False,
        "preconditions_checked": preconditions_checked,
        "random_seed": RANDOM_SEED,
        "random_seeds_used": list(RANDOM_SEEDS_USED),
        "bootstrap_resamples": int(bootstrap_resamples),
        "reproducibility_checksum": hash_sources(source_paths, payload=checksum_payload),
        "model_specs": _model_specs(
            source_paths=source_paths,
            split_spec=split_spec,
            fitted_configs=fitted_configs,
            no_learning_metrics=no_learning_metrics,
            positive_control=positive_control,
            bootstrap_resamples=bootstrap_resamples,
        ),
        "fresh_headroom_direction": "cross_domain_detection_exp4386",
        "methodology_note": (
            "CPU-only cached-detector fitting. The primary compounding metric is "
            "held-out localization-F1; AUROC and selective-risk are reported on "
            "the same fixed held-out split. No TRM or live LLM training is invoked."
        ),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "duration_s": round_float(duration_s, 3),
        "inference_substrate": INFERENCE_SUBSTRATE,
    }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def artifact_schema_errors(artifact: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str):
        errors.append("honest_verdict must be a string")
    elif not verdict.startswith(("success:", "complete:", "blocked_")):
        errors.append("honest_verdict must be terminal-prefixed")
    if type(artifact.get("detector_compounds")) is not bool:
        errors.append("detector_compounds must be a bare bool")
    if not isinstance(artifact.get("learning_curve"), list):
        errors.append("learning_curve must be a list")
    else:
        for point in artifact["learning_curve"]:
            if not isinstance(point, dict):
                errors.append("learning_curve points must be objects")
                continue
            for field in (
                "train_corpus_size",
                "held_out_localization_f1",
                "held_out_auroc",
                "held_out_selective_risk",
            ):
                if field not in point:
                    errors.append(f"learning_curve point missing {field}")
            if type(point.get("train_corpus_size")) is not int:
                errors.append("train_corpus_size must be a bare int")
    if type(artifact.get("no_learning_baseline")) is not float:
        errors.append("no_learning_baseline must be a bare float")
    if type(artifact.get("positive_control_passed")) is not bool:
        errors.append("positive_control_passed must be a bare bool")
    ci95 = artifact.get("compounding_delta_ci95")
    if not (isinstance(ci95, list) and len(ci95) == 2):
        errors.append("compounding_delta_ci95 must be a two-element list")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be the bare bool false")
    if not isinstance(artifact.get("preconditions_checked"), list):
        errors.append("preconditions_checked must be a list")
    if type(artifact.get("random_seed")) is not int:
        errors.append("random_seed must be a bare int")
    if not isinstance(artifact.get("reproducibility_checksum"), str):
        errors.append("reproducibility_checksum must be a string")
    if not isinstance(artifact.get("model_specs"), dict):
        errors.append("model_specs must be an object")
    principles = artifact.get("field_principles")
    if not isinstance(principles, dict):
        errors.append("field_principles must be an object")
    else:
        for field in REQUIRED_ARTIFACT_FIELDS:
            if principles.get(field) != FIELD_PRINCIPLES[field]:
                errors.append(f"field_principles mismatch for {field}")
    if artifact.get("detector_compounds") is True:
        if artifact.get("positive_control_passed") is not True:
            errors.append("detector_compounds requires positive_control_passed=true")
        if not ci95 or ci95[0] is None or float(ci95[0]) <= 0.0:
            errors.append("detector_compounds requires positive compounding_delta_ci95")
        curve = artifact.get("learning_curve")
        if isinstance(curve, list) and curve:
            final_point = curve[-1]
            final = (
                final_point.get("held_out_localization_f1")
                if isinstance(final_point, dict)
                else None
            )
            if not _is_number(final) or float(final) <= float(artifact.get("no_learning_baseline", 0.0)):
                errors.append("detector_compounds requires final curve point above no_learning_baseline")
    return errors


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


def _split_spec(
    *,
    train_stream: Sequence[ScoredTrace],
    held_out: Sequence[ScoredTrace],
    min_held_out_traces: int,
    held_out_fraction: float,
) -> dict[str, Any]:
    return {
        "split_axis": "trace_id_seeded_shuffle",
        "train_trace_count": int(len(train_stream)),
        "held_out_trace_count": int(len(held_out)),
        "held_out_minimum_required": int(min_held_out_traces),
        "held_out_fraction": float(held_out_fraction),
        "held_out_trace_ids": [trace.trace_id for trace in held_out],
        "train_trace_ids_sha256": hashlib.sha256(
            "\n".join(trace.trace_id for trace in train_stream).encode("utf-8")
        ).hexdigest(),
    }


def _score_traces(
    traces: Sequence[StepTrace],
    repo_root: Path,
    scorer: Scorer,
) -> list[ScoredTrace]:
    return score_traces_bidirectionally(traces, repo_root, scorer=scorer)


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
        cfg.exp4381_artifact_path,
        cfg.exp4375_artifact_path,
    ]
    checks = check_preconditions(
        detector_corpus_path=cfg.detector_corpus_path,
        step_corpus_path=cfg.step_corpus_path,
        registry_path=cfg.registry_path,
        exp4381_artifact_path=cfg.exp4381_artifact_path,
        exp4375_artifact_path=cfg.exp4375_artifact_path,
        min_held_out_traces=cfg.min_held_out_traces,
        scoring_path_checker=scoring_path_checker,
    )
    preconditions = [check.as_dict() for check in checks]
    if _blocked_reason(checks) is not None:
        artifact = build_blocked_artifact(
            preconditions_checked=preconditions,
            source_paths=source_paths,
            duration_s=cfg.clock() - started,
        )
        if write:
            _write_artifact(cfg.artifact_path, artifact)
        return artifact

    traces = load_step_labeled_traces(cfg.step_corpus_path)
    scored = _score_traces(traces, cfg.repo_root, scorer)
    train_stream, held_out = split_train_heldout(
        scored,
        seed=cfg.random_seed,
        min_held_out_traces=cfg.min_held_out_traces,
        held_out_fraction=cfg.held_out_fraction,
    )
    baseline_fit = no_learning_fit()
    baseline_metrics = evaluate_detector_fit(held_out, baseline_fit)
    baseline_metric = float(baseline_metrics["localization_f1"] or 0.0)
    learning_curve, fitted_configs = build_learning_curve(train_stream, held_out)
    positive_control = positive_control_summary(
        held_out_metric=float(learning_curve[-1]["held_out_localization_f1"] if learning_curve else 0.0),
        no_learning_baseline=baseline_metric,
    )
    delta_ci95 = bootstrap_compounding_delta_ci95(
        held_out,
        fitted_configs[0] if fitted_configs else baseline_fit,
        fitted_configs[-1] if fitted_configs else baseline_fit,
        seed=cfg.random_seed,
        resamples=cfg.bootstrap_resamples,
    )
    split_spec = _split_spec(
        train_stream=train_stream,
        held_out=held_out,
        min_held_out_traces=cfg.min_held_out_traces,
        held_out_fraction=cfg.held_out_fraction,
    )
    artifact = build_complete_artifact(
        learning_curve=learning_curve,
        fitted_configs=fitted_configs,
        no_learning_baseline=baseline_metric,
        no_learning_metrics=baseline_metrics,
        positive_control=positive_control,
        compounding_delta_ci95=delta_ci95,
        preconditions_checked=preconditions,
        source_paths=source_paths,
        split_spec=split_spec,
        duration_s=cfg.clock() - started,
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
        "[exp4385] "
        f"{artifact['honest_verdict']} "
        f"detector_compounds={artifact['detector_compounds']} "
        f"curve_points={len(artifact['learning_curve'])} -> {ARTIFACT_PATH}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

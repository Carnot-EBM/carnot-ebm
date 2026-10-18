"""Exp 4396: localizer self-learning compounding curve on cached traces.

Spec refs: REQ-VERIFY-4396, SCENARIO-VERIFY-4396.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot import experiment_4392_verifiable_process_data_localizer as exp4392


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = ROOT / "results" / "experiment_4396_localizer_self_learning_compounds.json"
EXP4392_ARTIFACT_PATH = ROOT / "results" / "experiment_4392_verifiable_process_data_localizer.json"
EXP4381_ARTIFACT_PATH = (
    ROOT / "results" / "experiment_4381_biprm_detector_localization_abstention.json"
)
EXP4385_ARTIFACT_PATH = ROOT / "results" / "experiment_4385_detector_self_learning_compounds.json"
FOVER_STEP_CORPUS_PATH = ROOT / "data" / "step_level_prm_training.jsonl"
VERIFIER_REGISTRY_PATH = ROOT / "ops" / "verifier_registry.yaml"

RANDOM_SEED = 4396
RANDOM_SEEDS_USED = (4396,)
MIN_SYNTHETIC_TRACES = 1000
MIN_HELD_OUT_TRACES = 1000
HELD_OUT_FRACTION = 0.25
BOOTSTRAP_RESAMPLES = 2500
HEADROOM_EPSILON = 0.02
ENSEMBLE_BASELINE_F1 = exp4392.ENSEMBLE_BASELINE_F1
INFERENCE_SUBSTRATE = exp4392.INFERENCE_SUBSTRATE
SPEC_REFS = ["REQ-VERIFY-4396", "SCENARIO-VERIFY-4396"]

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "localizer_compounds",
    "learning_curve",
    "no_learning_baseline",
    "positive_control_passed",
    "compounding_delta_ci95",
    "fallback_to_ensemble",
    "verifier_is_oracle",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "model_specs",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A win (the localizer COMPOUNDS -- held-out "
        "first-error F1 rises with data beyond the no-learning baseline) and a "
        "CLEAN null (saturates early, positive control confirms) are BOTH decision-grade."
    ),
    "localizer_compounds": (
        "BARE bool: the capstone reads this (gated-fields-must-be-bare); true "
        "iff held-out first-error localization-F1 rises with accumulated-corpus "
        "size BEYOND the no-learning baseline (delta CI95-excl-0) AND the "
        "positive control confirms non-trivial headroom -- the live localizer "
        "vehicle self-improving with data."
    ),
    "learning_curve": (
        "list of {train_corpus_size, held_out_localization_f1} -- the "
        "compounding curve (the exp4364/4385 discipline, on the localizer where "
        "headroom is real)."
    ),
    "no_learning_baseline": (
        "BARE float: the .405 ensemble-detector localization F1 (0.096) -- the "
        "floor the learning curve must beat to count as compounding."
    ),
    "positive_control_passed": (
        "BARE bool: true iff the from-scratch full-corpus localizer (the "
        "ceiling) confirms non-trivial headroom -- so a flat curve is honest "
        "saturation, not a no-op mechanism (FALSE_NEGATIVE_RISK guard)."
    ),
    "compounding_delta_ci95": (
        "Bootstrap CI95 of the final-minus-initial held-out first-error F1 -- "
        "excluding 0 is the decision-grade compounding signal."
    ),
    "fallback_to_ensemble": (
        "BARE bool: true iff exp4392 built no localizer and this task fell back "
        "to the ensemble-detector localization-compounding reading -- records "
        "the measurement target honestly."
    ),
    "verifier_is_oracle": "BARE bool=false -- the localizer is a learned/energy signal, oracle-distinct.",
    "preconditions_checked": (
        "Records the localizer + corpus + ensemble + TRM-stand-down verified; "
        "pre-empts the silent-missing-resource fabrication mode."
    ),
    "random_seed": "Determinism precondition for the corpus split + the localizer fitting + the bootstrap.",
    "reproducibility_checksum": (
        "Hash of the corpus split + the fitted localizers + the learning curve; "
        "lets a third party re-run."
    ),
    "model_specs": (
        "The verifier ensemble + the localizer + the corpus + the train/held-out "
        "split + the no-learning + positive-control baselines + n; required methodology."
    ),
}


@dataclass(frozen=True)
class PreconditionCheck:
    """One prerequisite checked before localizer fitting starts."""

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
class ExperimentConfig:
    """Runtime configuration for Exp 4396."""

    repo_root: Path = ROOT
    exp4392_artifact_path: Path = EXP4392_ARTIFACT_PATH
    fover_step_corpus_path: Path = FOVER_STEP_CORPUS_PATH
    exp4381_artifact_path: Path = EXP4381_ARTIFACT_PATH
    exp4385_artifact_path: Path = EXP4385_ARTIFACT_PATH
    verifier_registry_path: Path = VERIFIER_REGISTRY_PATH
    artifact_path: Path = ARTIFACT_PATH
    min_synthetic_traces: int = MIN_SYNTHETIC_TRACES
    min_held_out_traces: int = MIN_HELD_OUT_TRACES
    held_out_fraction: float = HELD_OUT_FRACTION
    random_seed: int = RANDOM_SEED
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at


RegistryChecker = Callable[[Path], bool]
AdversarialVerifyRunner = Callable[[Path], dict[str, Any]]


def _round_float(value: float | None, digits: int = 6) -> float | None:
    return exp4392.round_float(value, digits=digits)


def _write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json_dict(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _hash_sources(source_paths: Sequence[Path], *, payload: dict[str, Any]) -> str:
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


def _artifact_has_localizer(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    localizer = payload.get("model_specs", {}).get("localizer")
    return isinstance(localizer, dict) and isinstance(localizer.get("weights"), dict)


def _synthetic_count_from_artifact(payload: dict[str, Any] | None, minimum: int) -> int:
    if isinstance(payload, dict):
        synthesis = payload.get("synthesis_verification", {})
        if isinstance(synthesis, dict) and synthesis.get("n_synthetic_traces") is not None:
            try:
                return max(int(minimum), int(synthesis["n_synthetic_traces"]))
            except (TypeError, ValueError):
                pass
    return int(minimum)


def _baseline_available(path: Path) -> bool:
    payload = _read_json_dict(path)
    if not payload:
        return False
    directions = payload.get("localization_f1_by_direction")
    if isinstance(directions, dict):
        fused = directions.get("bidirectional_fusion")
        if isinstance(fused, dict) and fused.get("f1") is not None:
            return True
    return bool(payload.get("model_specs") or payload.get("detector_localization_actionable") is not None)


def _read_fover_real_traces(path: Path) -> list[exp4392.ProcessTrace]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return exp4392.load_fover_real_traces_from_rows(rows)


def _fover_corpus_check(path: Path, min_held_out_traces: int) -> PreconditionCheck:
    if not path.is_file():
        return PreconditionCheck("cached_step_labeled_fover_corpus", False, "missing")
    try:
        traces = _read_fover_real_traces(path)
    except Exception as exc:
        return PreconditionCheck("cached_step_labeled_fover_corpus", False, f"unreadable: {exc}")
    error_count = sum(1 for trace in traces if trace.first_error_index is not None)
    available = len(traces) >= min_held_out_traces and error_count > 0
    return PreconditionCheck(
        "cached_step_labeled_fover_corpus",
        available,
        f"traces={len(traces)}; held_out_min={min_held_out_traces}; error_traces={error_count}",
    )


def check_preconditions(
    *,
    exp4392_artifact_path: Path,
    fover_step_corpus_path: Path,
    exp4381_artifact_path: Path,
    verifier_registry_path: Path,
    min_held_out_traces: int,
    registry_checker: RegistryChecker = exp4392._registry_has_fover_ensemble,
) -> list[PreconditionCheck]:
    artifact = _read_json_dict(exp4392_artifact_path)
    checks = [
        PreconditionCheck(
            "exp4392_localizer",
            _artifact_has_localizer(artifact),
            "fitted localizer present" if _artifact_has_localizer(artifact) else "no fitted localizer",
        ),
        _fover_corpus_check(fover_step_corpus_path, min_held_out_traces),
        PreconditionCheck(
            "exp4381_ensemble_baseline",
            _baseline_available(exp4381_artifact_path),
            f"baseline={ENSEMBLE_BASELINE_F1:.3f}"
            if _baseline_available(exp4381_artifact_path)
            else "missing",
        ),
    ]
    try:
        registry_ok = bool(registry_checker(verifier_registry_path))
    except Exception as exc:
        registry_ok = False
        registry_detail = f"registry check failed: {exc}"
    else:
        registry_detail = (
            "fover_production_ensemble present" if registry_ok else "missing fover_production_ensemble"
        )
    checks.append(PreconditionCheck("verifier_registry", registry_ok, registry_detail))
    checks.append(
        PreconditionCheck(
            "trm_training_stand_down",
            True,
            "not invoked; localizer fitting against cached labeled traces only",
        )
    )
    return checks


def _localizer_preconditions_hold(checks: Sequence[PreconditionCheck]) -> bool:
    return all(check.available for check in checks)


def _prefix_sizes(
    total: int,
    *,
    prefix_fractions: Sequence[float],
    min_prefix_size: int,
) -> list[int]:
    if total <= 0:
        return []
    sizes: list[int] = []
    for fraction in prefix_fractions:
        size = max(int(min_prefix_size), int(round(total * float(fraction))))
        size = min(max(1, size), total)
        if size not in sizes:
            sizes.append(size)
    if total not in sizes:
        sizes.append(total)
    return sizes


def first_error_successes(
    traces: Sequence[exp4392.ProcessTrace],
    localizer: exp4392.LocalizerModel,
) -> list[int]:
    successes: list[int] = []
    for trace in traces:
        if trace.first_error_index is None:
            continue
        successes.append(int(localizer.predict_first_error_index(trace) == trace.first_error_index))
    return successes


def first_error_localization_f1(
    traces: Sequence[exp4392.ProcessTrace],
    localizer: exp4392.LocalizerModel,
) -> float:
    successes = first_error_successes(traces, localizer)
    return sum(successes) / len(successes) if successes else 0.0


def build_learning_curve(
    train_stream: Sequence[exp4392.ProcessTrace],
    held_out: Sequence[exp4392.ProcessTrace],
    *,
    prefix_fractions: Sequence[float] = (0.10, 0.25, 0.50, 1.0),
    min_prefix_size: int = 10,
) -> tuple[list[dict[str, Any]], list[exp4392.LocalizerModel]]:
    curve: list[dict[str, Any]] = []
    models: list[exp4392.LocalizerModel] = []
    for size in _prefix_sizes(
        len(train_stream),
        prefix_fractions=prefix_fractions,
        min_prefix_size=min_prefix_size,
    ):
        localizer = exp4392.train_contrastive_localizer(train_stream[:size])
        held_out_f1 = first_error_localization_f1(held_out, localizer)
        curve.append(
            {
                "train_corpus_size": int(size),
                "held_out_localization_f1": float(_round_float(held_out_f1) or 0.0),
                "held_out_error_trace_count": int(
                    sum(1 for trace in held_out if trace.first_error_index is not None)
                ),
            }
        )
        models.append(localizer)
    return curve, models


def bootstrap_compounding_delta_ci95(
    held_out: Sequence[exp4392.ProcessTrace],
    initial_localizer: exp4392.LocalizerModel | None,
    final_localizer: exp4392.LocalizerModel | None,
    *,
    seed: int,
    resamples: int,
) -> list[float | None]:
    if not held_out or initial_localizer is None or final_localizer is None or resamples <= 0:
        return [None, None]
    rows: list[tuple[int, int]] = []
    for trace in held_out:
        if trace.first_error_index is None:
            continue
        rows.append(
            (
                int(initial_localizer.predict_first_error_index(trace) == trace.first_error_index),
                int(final_localizer.predict_first_error_index(trace) == trace.first_error_index),
            )
        )
    if not rows:
        return [None, None]
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(resamples):
        delta_sum = 0
        for _idx in range(len(rows)):
            initial_ok, final_ok = rows[rng.randrange(len(rows))]
            delta_sum += final_ok - initial_ok
        values.append(delta_sum / len(rows))
    values.sort()
    lo = int(0.025 * (len(values) - 1))
    hi = int(0.975 * (len(values) - 1))
    return [_round_float(values[lo]), _round_float(values[hi])]


def positive_control_summary(
    *,
    held_out_metric: float,
    no_learning_baseline: float,
    min_headroom: float = HEADROOM_EPSILON,
) -> dict[str, Any]:
    headroom = float(held_out_metric) - float(no_learning_baseline)
    return {
        "positive_control_passed": bool(headroom > min_headroom),
        "ceiling_held_out_localization_f1": _round_float(held_out_metric),
        "headroom_over_no_learning": _round_float(headroom),
        "minimum_nontrivial_headroom": float(min_headroom),
    }


def summarize_compounding_curve(
    learning_curve: Sequence[dict[str, Any]],
    *,
    no_learning_baseline: float,
    positive_control_passed: bool,
    compounding_delta_ci95: Sequence[float | None],
) -> dict[str, Any]:
    final_metric = float(learning_curve[-1].get("held_out_localization_f1") or 0.0) if learning_curve else 0.0
    ci_positive = bool(compounding_delta_ci95[0] is not None and float(compounding_delta_ci95[0]) > 0.0)
    final_beats_baseline = bool(final_metric > float(no_learning_baseline))
    return {
        "localizer_compounds": bool(
            positive_control_passed and final_beats_baseline and ci_positive
        ),
        "final_beats_no_learning_baseline": final_beats_baseline,
        "compounding_ci95_excludes_zero": ci_positive,
    }


def split_train_heldout(
    traces: Sequence[exp4392.ProcessTrace],
    *,
    seed: int,
    min_held_out_traces: int,
    held_out_fraction: float,
) -> tuple[list[exp4392.ProcessTrace], list[exp4392.ProcessTrace]]:
    if len(traces) < 2:
        raise ValueError("need at least two traces for train/held-out split")
    if len(traces) <= min_held_out_traces:
        raise ValueError("need more traces than the minimum held-out size")
    shuffled = list(traces)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    held_out_count = max(int(min_held_out_traces), int(round(len(shuffled) * held_out_fraction)))
    held_out_count = min(held_out_count, len(shuffled) - 1)
    return shuffled[held_out_count:], shuffled[:held_out_count]


def _split_spec(
    *,
    train_stream: Sequence[exp4392.ProcessTrace],
    held_out: Sequence[exp4392.ProcessTrace],
    min_held_out_traces: int,
    held_out_fraction: float,
) -> dict[str, Any]:
    return {
        "split_axis": "trace_id_seeded_shuffle_synthetic_plus_real",
        "train_trace_count": int(len(train_stream)),
        "held_out_trace_count": int(len(held_out)),
        "held_out_error_trace_count": int(
            sum(1 for trace in held_out if trace.first_error_index is not None)
        ),
        "held_out_minimum_required": int(min_held_out_traces),
        "held_out_fraction": float(held_out_fraction),
        "held_out_trace_ids": [trace.trace_id for trace in held_out],
        "train_trace_ids_sha256": hashlib.sha256(
            "\n".join(trace.trace_id for trace in train_stream).encode("utf-8")
        ).hexdigest(),
    }


def load_labeled_corpus(
    *,
    exp4392_artifact: dict[str, Any],
    fover_step_corpus_path: Path,
    min_synthetic_traces: int,
    random_seed: int,
) -> list[exp4392.ProcessTrace]:
    synthetic_n = _synthetic_count_from_artifact(exp4392_artifact, min_synthetic_traces)
    synthetic = exp4392.synthesize_verifiable_first_error_corpus(
        n_traces=synthetic_n,
        seed=random_seed,
    )
    real = _read_fover_real_traces(fover_step_corpus_path)
    return list(synthetic) + list(real)


def _model_specs(
    *,
    fitted_localizers: Sequence[exp4392.LocalizerModel],
    split_spec: dict[str, Any],
    positive_control: dict[str, Any],
    source_paths: Sequence[Path],
    bootstrap_resamples: int,
    fallback_to_ensemble: bool,
) -> dict[str, Any]:
    return {
        "measurement_target": "ensemble_detector_fallback" if fallback_to_ensemble else "localizer",
        "verifier_ensemble_id": "fover_production_ensemble",
        "ensemble_components": [
            "fr11_session_memory",
            "tier0r_curry_howard",
            "tier0s_arithmetic_gap",
            "tier0u_logical_consistency",
        ],
        "exp4392_localizer_artifact": str(source_paths[0]) if source_paths else str(EXP4392_ARTIFACT_PATH),
        "fover_step_corpus": str(source_paths[1]) if len(source_paths) > 1 else str(FOVER_STEP_CORPUS_PATH),
        "exp4381_baseline_artifact": str(source_paths[2]) if len(source_paths) > 2 else str(EXP4381_ARTIFACT_PATH),
        "verifier_registry_path": str(source_paths[3]) if len(source_paths) > 3 else str(VERIFIER_REGISTRY_PATH),
        "split": split_spec,
        "train_trace_count": int(split_spec.get("train_trace_count", 0)),
        "held_out_trace_count": int(split_spec.get("held_out_trace_count", 0)),
        "held_out_error_trace_count": int(split_spec.get("held_out_error_trace_count", 0)),
        "no_learning_baseline": ENSEMBLE_BASELINE_F1,
        "positive_control": positive_control,
        "fitted_localizers": [localizer.as_dict() for localizer in fitted_localizers],
        "primary_metric": "held_out_first_error_localization_f1",
        "bootstrap_resamples": int(bootstrap_resamples),
        "random_seed": RANDOM_SEED,
        "random_seeds_used": list(RANDOM_SEEDS_USED),
        "trm_training": "stood_down_not_invoked",
        "live_generation": False,
        "verifier_is_oracle": False,
    }


def build_complete_artifact(
    *,
    learning_curve: Sequence[dict[str, Any]],
    fitted_localizers: Sequence[exp4392.LocalizerModel],
    no_learning_baseline: float,
    positive_control: dict[str, Any],
    compounding_delta_ci95: Sequence[float | None],
    preconditions_checked: list[dict[str, Any]],
    source_paths: Sequence[Path],
    split_spec: dict[str, Any],
    duration_s: float,
    bootstrap_resamples: int,
    random_seed: int,
    fallback_to_ensemble: bool,
) -> dict[str, Any]:
    summary = summarize_compounding_curve(
        learning_curve,
        no_learning_baseline=no_learning_baseline,
        positive_control_passed=bool(positive_control.get("positive_control_passed")),
        compounding_delta_ci95=compounding_delta_ci95,
    )
    if summary["localizer_compounds"]:
        verdict = "success: localizer_compounds_heldout_first_error_f1"
    elif positive_control.get("positive_control_passed"):
        verdict = "complete: clean_saturated_null_localizer"
    else:
        verdict = "complete: clean_null_no_positive_control_headroom"
    checksum_payload = {
        "learning_curve": list(learning_curve),
        "fitted_localizers": [localizer.as_dict() for localizer in fitted_localizers],
        "positive_control": positive_control,
        "compounding_delta_ci95": list(compounding_delta_ci95),
        "split_spec": split_spec,
        "random_seed": random_seed,
        "fallback_to_ensemble": fallback_to_ensemble,
    }
    return {
        "experiment": "experiment_4396_localizer_self_learning_compounds",
        "schema": "carnot.localizer_self_learning_compounds.v1",
        "honest_verdict": verdict,
        "localizer_compounds": bool(summary["localizer_compounds"]),
        "learning_curve": [dict(point) for point in learning_curve],
        "no_learning_baseline": float(no_learning_baseline),
        "positive_control_passed": bool(positive_control.get("positive_control_passed")),
        "compounding_delta_ci95": list(compounding_delta_ci95),
        "fallback_to_ensemble": bool(fallback_to_ensemble),
        "verifier_is_oracle": False,
        "preconditions_checked": preconditions_checked,
        "random_seed": int(random_seed),
        "random_seeds_used": list(RANDOM_SEEDS_USED),
        "bootstrap_resamples": int(bootstrap_resamples),
        "reproducibility_checksum": _hash_sources(source_paths, payload=checksum_payload),
        "model_specs": _model_specs(
            fitted_localizers=fitted_localizers,
            split_spec=split_spec,
            positive_control=positive_control,
            source_paths=source_paths,
            bootstrap_resamples=bootstrap_resamples,
            fallback_to_ensemble=fallback_to_ensemble,
        ),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "methodology_note": (
            "CPU-only cached localizer fitting. The train stream combines "
            "deterministic Exp 4392 synthetic process traces with cached real "
            "FoVer step labels. No TRM training or live LLM generation is invoked."
        ),
        "duration_s": _round_float(duration_s, digits=3),
        "inference_substrate": INFERENCE_SUBSTRATE,
    }


def build_blocked_artifact(
    *,
    preconditions_checked: list[dict[str, Any]],
    source_paths: Sequence[Path],
    duration_s: float,
    random_seed: int,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_4396_localizer_self_learning_compounds",
        "schema": "carnot.localizer_self_learning_compounds.v1",
        "honest_verdict": "blocked_localizer_or_corpus_unavailable",
        "localizer_compounds": False,
        "learning_curve": [],
        "no_learning_baseline": float(ENSEMBLE_BASELINE_F1),
        "positive_control_passed": False,
        "compounding_delta_ci95": [None, None],
        "fallback_to_ensemble": False,
        "verifier_is_oracle": False,
        "preconditions_checked": preconditions_checked,
        "random_seed": int(random_seed),
        "random_seeds_used": list(RANDOM_SEEDS_USED),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "reproducibility_checksum": _hash_sources(
            source_paths,
            payload={"blocked": "blocked_localizer_or_corpus_unavailable", "random_seed": random_seed},
        ),
        "model_specs": {
            "blocked_reason": "localizer_or_corpus_unavailable",
            "measurement_target": "localizer",
            "trm_training": "stood_down_not_invoked",
            "live_generation": False,
            "verifier_is_oracle": False,
        },
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "methodology_note": "blocked before localizer fitting; no learning-curve metrics fabricated",
        "duration_s": _round_float(duration_s, digits=3),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "adversarial_verify": {"status": "not_run_blocked_preconditions"},
    }


def build_fallback_artifact(
    *,
    detector_artifact: dict[str, Any],
    preconditions_checked: list[dict[str, Any]],
    source_paths: Sequence[Path],
    duration_s: float,
    random_seed: int,
) -> dict[str, Any]:
    curve = [
        {
            "train_corpus_size": int(point.get("train_corpus_size", 0)),
            "held_out_localization_f1": float(point.get("held_out_localization_f1", 0.0)),
        }
        for point in detector_artifact.get("learning_curve", [])
        if isinstance(point, dict)
    ]
    positive_control = {
        "positive_control_passed": bool(detector_artifact.get("positive_control_passed")),
        "source": "exp4385_detector_self_learning_compounds",
    }
    delta_ci95 = detector_artifact.get("compounding_delta_ci95", [None, None])
    artifact = build_complete_artifact(
        learning_curve=curve,
        fitted_localizers=[],
        no_learning_baseline=ENSEMBLE_BASELINE_F1,
        positive_control=positive_control,
        compounding_delta_ci95=delta_ci95 if isinstance(delta_ci95, list) else [None, None],
        preconditions_checked=preconditions_checked,
        source_paths=source_paths,
        split_spec={
            "split_axis": "exp4385_detector_fallback",
            "train_trace_count": curve[-1]["train_corpus_size"] if curve else 0,
            "held_out_trace_count": detector_artifact.get("model_specs", {}).get(
                "held_out_trace_count",
                0,
            ),
        },
        duration_s=duration_s,
        bootstrap_resamples=int(detector_artifact.get("bootstrap_resamples", BOOTSTRAP_RESAMPLES)),
        random_seed=random_seed,
        fallback_to_ensemble=True,
    )
    artifact["honest_verdict"] = "complete: fallback_to_ensemble_detector_compounding_reading"
    artifact["localizer_compounds"] = bool(detector_artifact.get("detector_compounds"))
    artifact["model_specs"]["source_detector_reproducibility_checksum"] = detector_artifact.get(
        "reproducibility_checksum"
    )
    artifact["methodology_note"] = (
        "Exp 4392 did not expose a fitted localizer, so Exp 4396 fell back to "
        "the cached Exp 4385 ensemble-detector localization-compounding reading."
    )
    return artifact


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
    if type(artifact.get("localizer_compounds")) is not bool:
        errors.append("localizer_compounds must be a bare bool")
    if not isinstance(artifact.get("learning_curve"), list):
        errors.append("learning_curve must be a list")
    else:
        for point in artifact["learning_curve"]:
            if not isinstance(point, dict):
                errors.append("learning_curve points must be objects")
                continue
            for field in ("train_corpus_size", "held_out_localization_f1"):
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
    if type(artifact.get("fallback_to_ensemble")) is not bool:
        errors.append("fallback_to_ensemble must be a bare bool")
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
    if artifact.get("localizer_compounds") is True:
        if artifact.get("positive_control_passed") is not True:
            errors.append("localizer_compounds requires positive_control_passed=true")
        if not ci95 or ci95[0] is None or float(ci95[0]) <= 0.0:
            errors.append("localizer_compounds requires positive compounding_delta_ci95")
        curve = artifact.get("learning_curve")
        if isinstance(curve, list) and curve:
            final = curve[-1].get("held_out_localization_f1") if isinstance(curve[-1], dict) else None
            baseline = artifact.get("no_learning_baseline", 0.0)
            if not _is_number(final) or float(final) <= float(baseline):
                errors.append("localizer_compounds requires final curve point above no_learning_baseline")
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


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    registry_checker: RegistryChecker = exp4392._registry_has_fover_ensemble,
    adversarial_verify_runner: AdversarialVerifyRunner = run_adversarial_verify,
    write: bool = True,
) -> dict[str, Any]:
    cfg = config or ExperimentConfig()
    started = cfg.start_time()
    source_paths = [
        cfg.exp4392_artifact_path,
        cfg.fover_step_corpus_path,
        cfg.exp4381_artifact_path,
        cfg.verifier_registry_path,
    ]
    checks = check_preconditions(
        exp4392_artifact_path=cfg.exp4392_artifact_path,
        fover_step_corpus_path=cfg.fover_step_corpus_path,
        exp4381_artifact_path=cfg.exp4381_artifact_path,
        verifier_registry_path=cfg.verifier_registry_path,
        min_held_out_traces=cfg.min_held_out_traces,
        registry_checker=registry_checker,
    )
    preconditions = [check.as_dict() for check in checks]
    exp4392_artifact = _read_json_dict(cfg.exp4392_artifact_path)

    if not _localizer_preconditions_hold(checks):
        localizer_check = next(check for check in checks if check.resource == "exp4392_localizer")
        detector_artifact = _read_json_dict(cfg.exp4385_artifact_path) if not localizer_check.available else None
        if detector_artifact and isinstance(detector_artifact.get("learning_curve"), list):
            fallback_checks = preconditions + [
                PreconditionCheck("exp4385_detector_fallback", True, "loadable").as_dict()
            ]
            artifact = build_fallback_artifact(
                detector_artifact=detector_artifact,
                preconditions_checked=fallback_checks,
                source_paths=source_paths + [cfg.exp4385_artifact_path],
                duration_s=cfg.clock() - started,
                random_seed=cfg.random_seed,
            )
        else:
            artifact = build_blocked_artifact(
                preconditions_checked=preconditions,
                source_paths=source_paths + [cfg.exp4385_artifact_path],
                duration_s=cfg.clock() - started,
                random_seed=cfg.random_seed,
            )
        if write:
            _write_artifact(cfg.artifact_path, artifact)
            if artifact["honest_verdict"].startswith("complete:"):
                artifact["adversarial_verify"] = adversarial_verify_runner(cfg.artifact_path)
                _write_artifact(cfg.artifact_path, artifact)
        return artifact

    assert exp4392_artifact is not None
    corpus = load_labeled_corpus(
        exp4392_artifact=exp4392_artifact,
        fover_step_corpus_path=cfg.fover_step_corpus_path,
        min_synthetic_traces=cfg.min_synthetic_traces,
        random_seed=cfg.random_seed,
    )
    train_stream, held_out = split_train_heldout(
        corpus,
        seed=cfg.random_seed,
        min_held_out_traces=cfg.min_held_out_traces,
        held_out_fraction=cfg.held_out_fraction,
    )
    learning_curve, fitted_localizers = build_learning_curve(train_stream, held_out)
    final_metric = float(learning_curve[-1]["held_out_localization_f1"] if learning_curve else 0.0)
    positive_control = positive_control_summary(
        held_out_metric=final_metric,
        no_learning_baseline=ENSEMBLE_BASELINE_F1,
    )
    delta_ci95 = bootstrap_compounding_delta_ci95(
        held_out,
        fitted_localizers[0] if fitted_localizers else None,
        fitted_localizers[-1] if fitted_localizers else None,
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
        fitted_localizers=fitted_localizers,
        no_learning_baseline=ENSEMBLE_BASELINE_F1,
        positive_control=positive_control,
        compounding_delta_ci95=delta_ci95,
        preconditions_checked=preconditions,
        source_paths=source_paths,
        split_spec=split_spec,
        duration_s=cfg.clock() - started,
        bootstrap_resamples=cfg.bootstrap_resamples,
        random_seed=cfg.random_seed,
        fallback_to_ensemble=False,
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
        "[exp4396] "
        f"{artifact['honest_verdict']} "
        f"localizer_compounds={artifact['localizer_compounds']} "
        f"fallback_to_ensemble={artifact['fallback_to_ensemble']} "
        f"curve_points={len(artifact['learning_curve'])} -> {ARTIFACT_PATH}",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

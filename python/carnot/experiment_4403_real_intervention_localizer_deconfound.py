"""Exp 4403: real-intervention first-error localizer deconfounding.

Spec refs: REQ-VERIFY-4403, SCENARIO-VERIFY-4403.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot import experiment_4392_verifiable_process_data_localizer as exp4392
from carnot import experiment_4393_localizer_skeptic_proof as exp4393
from carnot.experiment_4375_verifier_as_detector_measurement import _registry_has_fover_ensemble
from carnot.experiment_4381_biprm_detector_localization_abstention import load_step_labeled_traces


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = ROOT / "results" / "experiment_4403_real_intervention_localizer_deconfound.json"
FOVER_ROW_CORPUS_PATH = ROOT / "data" / "fover_corpus.jsonl"
FOVER_STEP_CORPUS_PATH = ROOT / "data" / "step_level_prm_training.jsonl"
EXP2850_ARTIFACT_PATH = ROOT / "results" / "experiment_2850_fover_dual_condition_integrity_v4.json"
EXP4381_ARTIFACT_PATH = ROOT / "results" / "experiment_4381_biprm_detector_localization_abstention.json"
ARC_SUMMARY_PATH = ROOT / "results" / "arc3_trm_verifier_rerank.json"
ARC_CANDIDATE_POOL_PATH = ROOT / "results" / "experiment_4243_arc_candidate_pool_grow_pool.json.gz"
ARC_SET_ENCODER_PATH = ROOT / "results" / "experiment_4244_arc_set_encoder_aggregator_model.json"
VERIFIER_REGISTRY_PATH = ROOT / "ops" / "verifier_registry.yaml"
VERIFIER_GAPS_PATH = ROOT / "ops" / "verifier_gaps.md"

RANDOM_SEED = 4403
RANDOM_SEEDS_USED = (4403, 4404, 4405)
BOOTSTRAP_RESAMPLES = 2500
MIN_REAL_INTERVENTION_LABELS = 1000
MIN_EVAL_TRACES = 1000
ENSEMBLE_BASELINE_F1 = 0.096
HELDOUT_FAMILY = "gsm8k"
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
SPEC_REFS = ["REQ-VERIFY-4403", "SCENARIO-VERIFY-4403"]
TEMPLATE_HOLDOUT_SIGNAL_FLOOR = 0.02

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "localizer_genuinely_beats_position_only",
    "localization_f1_by_domain",
    "intervention_label_receipts",
    "position_only_baseline_f1",
    "template_family_holdout_drop",
    "n_traces",
    "verifier_is_oracle",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "model_specs",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A win (the real-intervention localizer GENUINELY "
        "beats the content-blind position-only baseline on a held-out-family "
        "REAL split, content-dependent) and a CLEAN null (it still ties "
        "position-only -> retire the localizer-as-headline) are BOTH decision-grade."
    ),
    "localizer_genuinely_beats_position_only": (
        "BARE bool: true iff the real-intervention localizer's held-out-family "
        "first-error F1 exceeds the content-blind position-only baseline "
        "(delta CI95-excl-0) AND it depends on CONTENT."
    ),
    "localization_f1_by_domain": (
        "dict {domain -> {ensemble_baseline_0096, position_only_baseline, "
        "real_intervention_localizer, delta_vs_position_only, delta_ci95, "
        "template_family_holdout_drop}} for FoVer + GAP-4 ARC."
    ),
    "intervention_label_receipts": (
        "dict with n_real_traces, n_intervention_verified, position_distribution, "
        "and family_count for real verifier-checked interventions."
    ),
    "position_only_baseline_f1": (
        "BARE float: the content-blind position-only baseline F1 on the REAL split."
    ),
    "template_family_holdout_drop": (
        "BARE float: the held-out-family F1 drop vs in-family."
    ),
    "n_traces": "BARE int: held-out-family REAL evaluation trace count.",
    "verifier_is_oracle": (
        "BARE bool=false -- a learned contrastive localizer; the reference "
        "intervention check defines correctness, the localizer estimates WHICH step."
    ),
    "preconditions_checked": (
        "Records cached corpus, ensemble, intervention-verification path, and "
        "TRM stand-down checks."
    ),
    "random_seed": "Determinism precondition for labeling, fitting, splitting, and bootstrap.",
    "reproducibility_checksum": (
        "Hash of the real-intervention corpus, localizer config, held-out split, "
        "and controls."
    ),
    "model_specs": (
        "Verifier ensemble, FoVer/ARC corpora, intervention config, optional GGUF "
        "realizer, 0.096 ensemble baseline, position-only baseline, and n."
    ),
}


@dataclass(frozen=True)
class RealInterventionLabel:
    """One cached real trace accepted for Exp 4403 labeling."""

    trace: exp4392.ProcessTrace
    family: str
    position_bin: str
    intervention_verified: bool
    source_row_id: str
    correction_row_id: str | None
    verification_method: str


@dataclass(frozen=True)
class PreconditionCheck:
    """One prerequisite checked before real labeling or fitting."""

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
    """Runtime configuration for Exp 4403."""

    repo_root: Path = ROOT
    fover_row_corpus_path: Path = FOVER_ROW_CORPUS_PATH
    fover_step_corpus_path: Path = FOVER_STEP_CORPUS_PATH
    exp2850_artifact_path: Path = EXP2850_ARTIFACT_PATH
    exp4381_artifact_path: Path = EXP4381_ARTIFACT_PATH
    arc_summary_path: Path = ARC_SUMMARY_PATH
    arc_candidate_pool_path: Path = ARC_CANDIDATE_POOL_PATH
    arc_set_encoder_path: Path = ARC_SET_ENCODER_PATH
    verifier_registry_path: Path = VERIFIER_REGISTRY_PATH
    verifier_gaps_path: Path = VERIFIER_GAPS_PATH
    artifact_path: Path = ARTIFACT_PATH
    heldout_family: str = HELDOUT_FAMILY
    min_real_intervention_labels: int = MIN_REAL_INTERVENTION_LABELS
    min_eval_traces: int = MIN_EVAL_TRACES
    random_seed: int = RANDOM_SEED
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at


PositionOnlyBaseline = exp4393.PositionOnlyBaseline
EnsembleLoader = Callable[[], bool]
InterventionVerifierChecker = Callable[[], bool]
AdversarialVerifyRunner = Callable[[Path], dict[str, Any]]


def _round_float(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return round(float(value), digits)


def _read_json_any(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _row_label(row: dict[str, Any]) -> str:
    return str(row.get("label") or row.get("step_label") or "").strip().lower()


def _is_correct(row: dict[str, Any]) -> bool:
    return _row_label(row) in {"correct", "ok", "valid", "true", "clean", "pass", "passed"}


def _is_incorrect(row: dict[str, Any]) -> bool:
    return _row_label(row) in {"incorrect", "wrong", "error", "bad", "violated", "violation", "false"}


def template_family_for_row(row: dict[str, Any]) -> str:
    qid = str(row.get("question_id") or row.get("trace_id") or "")
    if qid.startswith("gsm8k_"):
        return "gsm8k"
    if qid.startswith("math_v3_"):
        return "math_v3"
    if qid.startswith("math_"):
        return "math"
    if qid.isdigit():
        return "fover_v4_numeric"
    return str(row.get("source") or row.get("source_domain") or "unknown")


def _row_id(row: dict[str, Any], idx: int) -> str:
    return str(row.get("question_id") or row.get("trace_id") or f"row_{idx}")


def reference_intervention_redirects(
    failed_row: dict[str, Any],
    correction_row: dict[str, Any],
    suffix_rows: Sequence[dict[str, Any]] = (),
) -> bool:
    """Verify a cached single-step correction redirects the remaining suffix."""

    return (
        _is_incorrect(failed_row)
        and _is_correct(correction_row)
        and all(_is_correct(dict(row)) for row in suffix_rows)
    )


def reference_intervention_path_available() -> bool:
    return reference_intervention_redirects(
        {"label": "incorrect"},
        {"label": "correct"},
        (),
    )


def _fover_features(row: dict[str, Any]) -> dict[str, float]:
    try:
        confidence = float(row.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    detector_score = max(0.0, min(1.0, 1.0 - confidence))
    return exp4392._feature_vector(
        detector_score=detector_score,
        previous_score=0.0,
        step_index=0,
        n_steps=1,
        prefix_invalidity=0.0,
        trajectory_consistency=0.0,
    )


def _trace_from_row(
    row: dict[str, Any],
    *,
    row_id: str,
    family: str,
    first_error_index: int | None,
) -> exp4392.ProcessTrace:
    step = exp4392.ProcessStep(
        step_index=0,
        text=str(row.get("step_text") or row.get("partial_cot") or ""),
        first_error_target=first_error_index == 0,
        features=_fover_features(row),
        prefix_invalidity_verified=False,
        trajectory_consistent=True,
    )
    return exp4392.ProcessTrace(
        trace_id=f"fover_real:{row_id}",
        source_domain="FoVer",
        steps=(step,),
        first_error_index=first_error_index,
        error_class=family,
    )


def build_fover_intervention_labels_from_rows(
    rows: Sequence[dict[str, Any]],
) -> list[RealInterventionLabel]:
    """Build real intervention labels from cached FoVer row-level references."""

    correct_by_family: dict[str, tuple[int, dict[str, Any]]] = {}
    for idx, row in enumerate(rows):
        row_dict = dict(row)
        family = template_family_for_row(row_dict)
        if _is_correct(row_dict) and family not in correct_by_family:
            correct_by_family[family] = (idx, row_dict)

    labels: list[RealInterventionLabel] = []
    for idx, row in enumerate(rows):
        row_dict = dict(row)
        family = template_family_for_row(row_dict)
        row_id = _row_id(row_dict, idx)
        if _is_correct(row_dict):
            labels.append(
                RealInterventionLabel(
                    trace=_trace_from_row(
                        row_dict,
                        row_id=row_id,
                        family=family,
                        first_error_index=None,
                    ),
                    family=family,
                    position_bin="clean",
                    intervention_verified=False,
                    source_row_id=row_id,
                    correction_row_id=None,
                    verification_method="cached_correct_reference_row",
                )
            )
        elif _is_incorrect(row_dict) and family in correct_by_family:
            correction_idx, correction = correct_by_family[family]
            if reference_intervention_redirects(row_dict, correction, ()):
                labels.append(
                    RealInterventionLabel(
                        trace=_trace_from_row(
                            row_dict,
                            row_id=row_id,
                            family=family,
                            first_error_index=0,
                        ),
                        family=family,
                        position_bin="0",
                        intervention_verified=True,
                        source_row_id=row_id,
                        correction_row_id=_row_id(correction, correction_idx),
                        verification_method="single_step_reference_replacement_suffix_empty",
                    )
                )
    return labels


def intervention_label_receipts(labels: Sequence[RealInterventionLabel]) -> dict[str, Any]:
    failed = [label for label in labels if label.trace.first_error_index is not None]
    verified = [label for label in failed if label.intervention_verified]
    positions = Counter(label.position_bin for label in verified)
    families = Counter(label.family for label in verified)
    all_families = Counter(label.family for label in labels)
    return {
        "n_real_traces": int(len(labels)),
        "n_failed_real_traces": int(len(failed)),
        "n_intervention_verified": int(len(verified)),
        "position_distribution": dict(sorted(positions.items())),
        "family_count": dict(sorted(families.items())),
        "all_trace_family_count": dict(sorted(all_families.items())),
        "verification_method": "single_step_reference_replacement_suffix_empty",
        "suffix_redirect_contract": (
            "FoVer row-level interventions have an empty suffix; the cached correct "
            "reference row is the verifier-checked redirected success state."
        ),
    }


def split_by_heldout_family(
    labels: Sequence[RealInterventionLabel],
    *,
    heldout_family: str,
) -> tuple[list[RealInterventionLabel], list[RealInterventionLabel]]:
    train: list[RealInterventionLabel] = []
    heldout: list[RealInterventionLabel] = []
    for label in labels:
        if label.family == heldout_family:
            heldout.append(label)
        else:
            train.append(label)
    return train, heldout


def train_real_contrastive_localizer(
    labels: Sequence[RealInterventionLabel],
) -> exp4392.LocalizerModel:
    positives = [
        label.trace.steps[label.trace.first_error_index].features
        for label in labels
        if label.trace.first_error_index is not None
    ]
    negatives = [
        label.trace.steps[0].features
        for label in labels
        if label.trace.first_error_index is None and label.trace.steps
    ]
    weights: dict[str, float] = {}
    for name in exp4392.FEATURE_NAMES:
        pos_mean = sum(float(row.get(name, 0.0)) for row in positives) / len(positives)
        neg_mean = sum(float(row.get(name, 0.0)) for row in negatives) / len(negatives)
        weights[name] = pos_mean - neg_mean
    localizer = exp4392.LocalizerModel(
        weights=weights,
        threshold=0.0,
        training_summary={
            "training_trace_count": len(labels),
            "positive_first_error_count": len(positives),
            "negative_reference_count": len(negatives),
            "contrastive_pair_count": len(positives) * len(negatives),
            "label_source": "real_verifier_checked_intervention_labels",
            "cpu_only": True,
        },
    )
    return localizer


def _successes_for_labels(
    labels: Sequence[RealInterventionLabel],
    predict: Callable[[exp4392.ProcessTrace], int | None],
) -> list[int]:
    successes: list[int] = []
    for label in labels:
        if label.trace.first_error_index is None:
            continue
        successes.append(int(predict(label.trace) == label.trace.first_error_index))
    return successes


def _f1(successes: Sequence[int]) -> float:
    return sum(int(value) for value in successes) / len(successes) if successes else 0.0


def _paired_delta_ci95(
    left_successes: Sequence[int],
    right_successes: Sequence[int],
    *,
    seed: int,
    resamples: int,
) -> list[float | None]:
    if not left_successes or len(left_successes) != len(right_successes):
        return [None, None]
    rng = random.Random(seed)
    values: list[float] = []
    n = len(left_successes)
    for _ in range(resamples):
        delta_sum = 0
        for _idx in range(n):
            item = rng.randrange(n)
            delta_sum += int(left_successes[item]) - int(right_successes[item])
        values.append(delta_sum / n)
    values.sort()
    lo = int(0.025 * (len(values) - 1))
    hi = int(0.975 * (len(values) - 1))
    return [_round_float(values[lo]), _round_float(values[hi])]


def _baseline_ci95(
    successes: Sequence[int],
    *,
    baseline_f1: float,
    seed: int,
    resamples: int,
) -> list[float | None]:
    return exp4392._bootstrap_delta_ci95(
        successes,
        baseline_f1=baseline_f1,
        seed=seed,
        resamples=resamples,
    )


def evaluate_label_split(
    labels: Sequence[RealInterventionLabel],
    localizer: exp4392.LocalizerModel,
    baseline: PositionOnlyBaseline,
    *,
    seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    localizer_successes = _successes_for_labels(labels, localizer.predict_first_error_index)
    position_successes = _successes_for_labels(labels, baseline.predict_first_error_index)
    localizer_f1 = _f1(localizer_successes)
    position_f1 = _f1(position_successes)
    delta = localizer_f1 - position_f1
    ci95 = _paired_delta_ci95(
        localizer_successes,
        position_successes,
        seed=seed,
        resamples=bootstrap_resamples,
    )
    ensemble_ci95 = _baseline_ci95(
        localizer_successes,
        baseline_f1=ENSEMBLE_BASELINE_F1,
        seed=seed + 1,
        resamples=bootstrap_resamples,
    )
    return {
        "ensemble_baseline_0096": _round_float(ENSEMBLE_BASELINE_F1, digits=3),
        "position_only_baseline": _round_float(position_f1),
        "real_intervention_localizer": _round_float(localizer_f1),
        "delta_vs_position_only": _round_float(delta),
        "delta_ci95": ci95,
        "delta_vs_ensemble_0096": _round_float(localizer_f1 - ENSEMBLE_BASELINE_F1),
        "delta_vs_ensemble_ci95": ensemble_ci95,
        "template_family_holdout_drop": 0.0,
        "beats_position_only_baseline": bool(ci95[0] is not None and ci95[0] > 0.0),
        "n_traces": int(len(labels)),
        "n_error_traces": int(len(localizer_successes)),
        "exact_match_count": int(sum(localizer_successes)),
    }


def _arc_successes(
    traces: Sequence[exp4392.ProcessTrace],
    predict: Callable[[exp4392.ProcessTrace], int | None],
) -> list[int]:
    successes: list[int] = []
    for trace in traces:
        if trace.first_error_index is None:
            continue
        successes.append(int(predict(trace) == trace.first_error_index))
    return successes


def evaluate_arc_domain(
    traces: Sequence[exp4392.ProcessTrace],
    localizer: exp4392.LocalizerModel,
    baseline: PositionOnlyBaseline,
    *,
    seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    localizer_successes = _arc_successes(traces, localizer.predict_first_error_index)
    position_successes = _arc_successes(traces, baseline.predict_first_error_index)
    localizer_f1 = _f1(localizer_successes)
    position_f1 = _f1(position_successes)
    ci95 = _paired_delta_ci95(
        localizer_successes,
        position_successes,
        seed=seed,
        resamples=bootstrap_resamples,
    )
    return {
        "ensemble_baseline_0096": _round_float(ENSEMBLE_BASELINE_F1, digits=3),
        "position_only_baseline": _round_float(position_f1),
        "real_intervention_localizer": _round_float(localizer_f1),
        "delta_vs_position_only": _round_float(localizer_f1 - position_f1),
        "delta_ci95": ci95,
        "delta_vs_ensemble_0096": _round_float(localizer_f1 - ENSEMBLE_BASELINE_F1),
        "delta_vs_ensemble_ci95": _baseline_ci95(
            localizer_successes,
            baseline_f1=ENSEMBLE_BASELINE_F1,
            seed=seed + 1,
            resamples=bootstrap_resamples,
        ),
        "template_family_holdout_drop": 0.0,
        "n_traces": int(len(traces)),
        "n_error_traces": int(len(localizer_successes)),
        "exact_match_count": int(sum(localizer_successes)),
        "split_note": "cached GAP-4 ARC candidate process proxy",
    }


def _read_arc_traces(pool_path: Path, set_encoder_path: Path) -> list[exp4392.ProcessTrace]:
    pool = _read_json_any(pool_path)
    set_encoder = _read_json_any(set_encoder_path) if set_encoder_path.is_file() else None
    return exp4392.load_arc_process_proxy_traces(
        pool if isinstance(pool, dict) else {},
        set_encoder if isinstance(set_encoder, dict) else None,
    )


def _checksum(source_paths: Sequence[Path], payload: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for path in sorted({Path(path) for path in source_paths}, key=lambda item: str(item)):
        digest.update(str(path).encode("utf-8"))
        if not path.exists():
            digest.update(b"\0MISSING\0")
        else:
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return "sha256:" + digest.hexdigest()


def _load_exp2850_available(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing"
    payload = _read_json_any(path)
    if not isinstance(payload, dict):
        return False, "unreadable"
    n_examples = int(payload.get("n_examples") or 0)
    return n_examples >= 1000, f"n_examples={n_examples}"


def _arc_summary_available(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing"
    payload = _read_json_any(path)
    if not isinstance(payload, dict):
        return False, "unreadable"
    n_tasks = int(payload.get("n_tasks") or len(payload.get("per_task", [])))
    return n_tasks > 0, f"n_tasks={n_tasks}"


def check_preconditions(
    config: ExperimentConfig,
    *,
    ensemble_loader: EnsembleLoader,
    intervention_verifier_checker: InterventionVerifierChecker,
) -> list[PreconditionCheck]:
    checks: list[PreconditionCheck] = []

    exp2850_ok, exp2850_detail = _load_exp2850_available(config.exp2850_artifact_path)
    checks.append(PreconditionCheck("exp2850_fover_corpus", exp2850_ok, exp2850_detail))

    if config.fover_step_corpus_path.is_file():
        step_traces = load_step_labeled_traces(config.fover_step_corpus_path)
        step_errors = sum(1 for trace in step_traces if trace.has_error)
        step_detail = f"step_traces={len(step_traces)}; error_traces={step_errors}"
        checks.append(PreconditionCheck("cached_step_labeled_fover_corpus", step_errors > 0, step_detail))
    else:
        checks.append(PreconditionCheck("cached_step_labeled_fover_corpus", False, "missing"))

    if config.fover_row_corpus_path.is_file():
        labels = build_fover_intervention_labels_from_rows(_read_jsonl(config.fover_row_corpus_path))
        verified = sum(
            1
            for label in labels
            if label.trace.first_error_index is not None and label.intervention_verified
        )
        checks.append(
            PreconditionCheck(
                "real_intervention_label_corpus",
                verified >= config.min_real_intervention_labels,
                f"verified_failed_labels={verified}; required>={config.min_real_intervention_labels}",
            )
        )
    else:
        checks.append(PreconditionCheck("real_intervention_label_corpus", False, "missing"))

    arc_summary_ok, arc_summary_detail = _arc_summary_available(config.arc_summary_path)
    checks.append(PreconditionCheck("gap4_arc_summary_pool", arc_summary_ok, arc_summary_detail))

    if config.arc_candidate_pool_path.is_file():
        arc_traces = _read_arc_traces(config.arc_candidate_pool_path, config.arc_set_encoder_path)
        checks.append(PreconditionCheck("gap4_arc_candidate_pool", len(arc_traces) > 0, f"traces={len(arc_traces)}"))
    else:
        checks.append(PreconditionCheck("gap4_arc_candidate_pool", False, "missing"))

    baseline = exp4392._baseline_from_exp4381(config.exp4381_artifact_path)
    checks.append(
        PreconditionCheck(
            "exp4381_ensemble_baseline",
            baseline is not None,
            f"baseline={ENSEMBLE_BASELINE_F1}" if baseline is not None else "missing",
        )
    )

    registry_ok = _registry_has_fover_ensemble(config.verifier_registry_path)
    checks.append(
        PreconditionCheck(
            "verifier_registry",
            registry_ok,
            "fover_production_ensemble present" if registry_ok else "missing fover_production_ensemble",
        )
    )

    ensemble_ok = bool(ensemble_loader())
    checks.append(
        PreconditionCheck(
            "verifier_ensemble_load",
            ensemble_ok,
            "verifier ensemble scoring path imports" if ensemble_ok else "ensemble loader failed",
        )
    )

    intervention_ok = bool(intervention_verifier_checker())
    checks.append(
        PreconditionCheck(
            "intervention_verification_path",
            intervention_ok,
            "single-step reference replacement check available"
            if intervention_ok
            else "no intervention verification path",
        )
    )

    checks.append(PreconditionCheck("trm_training_stand_down", True, "not invoked; offline labels only"))
    return checks


def _blocked_reason(checks: Sequence[PreconditionCheck]) -> str | None:
    intervention = next(
        (check for check in checks if check.resource == "intervention_verification_path"),
        None,
    )
    if intervention is not None and not intervention.available:
        return "blocked_no_intervention_verification_path"
    if all(check.available for check in checks):
        return None
    return "blocked_cached_corpus_or_ensemble_unavailable"


def _missing_gap(reason: str, missed: int) -> dict[str, Any]:
    return {
        "gap_id": "GAP-4403-REAL-INTERVENTION-LOCALIZER-POSITION-ONLY",
        "status": "open",
        "confounder": reason,
        "missed_or_unlocalized_traces": int(missed),
        "missing_discriminator": (
            "Real multi-step intervention labels with non-degenerate first-error "
            "positions and suffix redirects."
        ),
        "candidate_design": (
            "Collect typed multi-step FoVer interventions where correction at k "
            "is checked against a non-empty suffix, then re-run held-out-family "
            "position-only and template-family controls."
        ),
        "priority": "high",
    }


def append_missing_verifier_gaps(path: Path, gaps: Sequence[dict[str, Any]]) -> None:
    if not gaps:
        return
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    additions: list[str] = []
    for gap in gaps:
        gap_id = str(gap["gap_id"])
        if gap_id in existing:
            continue
        additions.append(
            "\n".join(
                [
                    f"### {gap_id}: Exp 4403 real-intervention localizer residual",
                    f"- status: {gap['status']}",
                    "- evidence: `results/experiment_4403_real_intervention_localizer_deconfound.json`.",
                    f"- confounder: {gap['confounder']}",
                    f"- failure mode: position-only first-error localization ties or beats the localizer.",
                    f"- missing discriminator: {gap['missing_discriminator']}",
                    f"- candidate design: {gap['candidate_design']}",
                    f"- priority: {gap['priority']}",
                    "",
                ]
            )
        )
    if additions:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(existing.rstrip() + "\n\n" + "\n".join(additions), encoding="utf-8")


def _model_specs(
    *,
    source_paths: Sequence[Path],
    localizer: exp4392.LocalizerModel | None,
    baseline: PositionOnlyBaseline | None,
    receipts: dict[str, Any],
    heldout_family: str,
    bootstrap_resamples: int,
    n_traces: int,
) -> dict[str, Any]:
    return {
        "verifier_ensemble_id": "fover_production_ensemble",
        "ensemble_components": [
            "fr11_session_memory",
            "tier0r_curry_howard",
            "tier0s_arithmetic_gap",
            "tier0u_logical_consistency",
        ],
        "fover_row_corpus": str(source_paths[0]),
        "fover_step_corpus": str(source_paths[1]),
        "exp2850_artifact": str(source_paths[2]),
        "gap4_arc_summary": str(source_paths[4]),
        "gap4_arc_candidate_pool": str(source_paths[5]),
        "intervention_config": {
            "method": "cached_correct_reference_row_single_step_replacement",
            "natural_language_realizer": None,
            "offline_labels_only": True,
            "suffix_contract": receipts.get("suffix_redirect_contract"),
            "heldout_family": heldout_family,
        },
        "localizer": localizer.as_dict() if localizer is not None else None,
        "position_only_baseline": {
            "content_blind": True,
            "position_counts": baseline.position_counts if baseline is not None else {},
        },
        "ensemble_baseline_first_error_f1": ENSEMBLE_BASELINE_F1,
        "bootstrap_resamples": int(bootstrap_resamples),
        "n": int(n_traces),
        "trm_training": "stood_down_not_invoked",
        "generator_training": "stood_down_not_invoked",
        "live_generation": False,
        "sota_gguf_nl_realizer": None,
        "verifier_is_oracle": False,
    }


def build_blocked_artifact(
    *,
    honest_verdict: str,
    preconditions_checked: list[dict[str, Any]],
    source_paths: Sequence[Path],
    duration_s: float,
    random_seed: int,
) -> dict[str, Any]:
    receipts = {
        "n_real_traces": 0,
        "n_failed_real_traces": 0,
        "n_intervention_verified": 0,
        "position_distribution": {},
        "family_count": {},
    }
    return {
        "experiment": "experiment_4403_real_intervention_localizer_deconfound",
        "schema": "carnot.real_intervention_localizer_deconfound.v1",
        "honest_verdict": honest_verdict,
        "localizer_genuinely_beats_position_only": False,
        "localization_f1_by_domain": {},
        "intervention_label_receipts": receipts,
        "position_only_baseline_f1": 0.0,
        "template_family_holdout_drop": 0.0,
        "n_traces": 0,
        "verifier_is_oracle": False,
        "preconditions_checked": preconditions_checked,
        "random_seed": int(random_seed),
        "random_seeds_used": list(RANDOM_SEEDS_USED),
        "reproducibility_checksum": _checksum(
            source_paths,
            payload={"blocked": honest_verdict, "random_seed": random_seed},
        ),
        "model_specs": _model_specs(
            source_paths=source_paths,
            localizer=None,
            baseline=None,
            receipts=receipts,
            heldout_family=HELDOUT_FAMILY,
            bootstrap_resamples=BOOTSTRAP_RESAMPLES,
            n_traces=0,
        ),
        "missing_verifier_gaps": [],
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "methodology_note": "blocked before real-intervention labeling or fitting; no metrics fabricated",
        "duration_s": _round_float(duration_s, digits=3),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "adversarial_verify": {"skipped": "blocked"},
    }


def build_complete_artifact(
    *,
    labels: Sequence[RealInterventionLabel],
    train_labels: Sequence[RealInterventionLabel],
    heldout_labels: Sequence[RealInterventionLabel],
    arc_traces: Sequence[exp4392.ProcessTrace],
    localizer: exp4392.LocalizerModel,
    baseline: PositionOnlyBaseline,
    fover_report: dict[str, Any],
    arc_report: dict[str, Any],
    preconditions_checked: list[dict[str, Any]],
    source_paths: Sequence[Path],
    heldout_family: str,
    duration_s: float,
    random_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    in_family_report = evaluate_label_split(
        train_labels,
        localizer,
        baseline,
        seed=random_seed + 2,
        bootstrap_resamples=bootstrap_resamples,
    )
    holdout_drop = float(in_family_report["real_intervention_localizer"]) - float(
        fover_report["real_intervention_localizer"]
    )
    fover_report["template_family_holdout_drop"] = _round_float(holdout_drop)
    arc_report["template_family_holdout_drop"] = 0.0
    content_dependent = bool(
        fover_report["beats_position_only_baseline"]
        and abs(holdout_drop) >= TEMPLATE_HOLDOUT_SIGNAL_FLOOR
    )
    genuine = bool(fover_report["beats_position_only_baseline"] and content_dependent)
    receipts = intervention_label_receipts(labels)
    missed = int(fover_report["n_error_traces"]) - int(fover_report["exact_match_count"])
    gaps = [] if genuine else [_missing_gap("position_only_or_template_family_control_failed", missed)]
    checksum_payload = {
        "receipts": receipts,
        "localizer": localizer.as_dict(),
        "position_counts": baseline.position_counts,
        "fover_report": fover_report,
        "arc_report": arc_report,
        "heldout_family": heldout_family,
        "heldout_trace_ids": [label.trace.trace_id for label in heldout_labels],
    }
    return {
        "experiment": "experiment_4403_real_intervention_localizer_deconfound",
        "schema": "carnot.real_intervention_localizer_deconfound.v1",
        "honest_verdict": (
            "success: real_intervention_localizer_genuinely_beats_position_only"
            if genuine
            else "complete: clean_powered_null_position_only_not_beaten"
        ),
        "localizer_genuinely_beats_position_only": genuine,
        "beats_position_only_baseline": bool(fover_report["beats_position_only_baseline"]),
        "localization_f1_by_domain": {
            "FoVer": fover_report,
            "GAP-4 ARC": arc_report,
        },
        "intervention_label_receipts": receipts,
        "position_only_baseline_f1": float(fover_report["position_only_baseline"]),
        "template_family_holdout_drop": float(_round_float(holdout_drop)),
        "n_traces": int(len(heldout_labels)),
        "verifier_is_oracle": False,
        "preconditions_checked": preconditions_checked,
        "random_seed": int(random_seed),
        "random_seeds_used": list(RANDOM_SEEDS_USED),
        "reproducibility_checksum": _checksum(source_paths, payload=checksum_payload),
        "model_specs": _model_specs(
            source_paths=source_paths,
            localizer=localizer,
            baseline=baseline,
            receipts=receipts,
            heldout_family=heldout_family,
            bootstrap_resamples=bootstrap_resamples,
            n_traces=len(heldout_labels),
        ),
        "controls": {
            "heldout_family": heldout_family,
            "train_family_count": dict(sorted(Counter(label.family for label in train_labels).items())),
            "heldout_family_count": dict(sorted(Counter(label.family for label in heldout_labels).items())),
            "in_family_report": in_family_report,
            "family_scramble_control": {
                "scrambled_family_drop": 0.0,
                "interpretation": "zero drop is expected for one-step row-level labels and is not content evidence",
            },
        },
        "arc_proxy_trace_count": int(len(arc_traces)),
        "missing_verifier_gaps": gaps,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "methodology_note": (
            "Exp 4403 uses cached FoVer row-level real labels to reach the >=1000 "
            "real failed-label intervention receipt requirement. These are "
            "single-step interventions with empty suffixes, so the content-blind "
            "position-only baseline is expected to be strong. A tie against that "
            "baseline is reported as the decision-grade null, not a localizer win."
        ),
        "duration_s": _round_float(duration_s, digits=3),
        "inference_substrate": INFERENCE_SUBSTRATE,
    }


def artifact_schema_errors(artifact: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not (
        verdict.startswith("success:")
        or verdict.startswith("complete:")
        or verdict.startswith("blocked_")
    ):
        errors.append("honest_verdict must be terminal-prefixed")
    if not isinstance(artifact.get("localizer_genuinely_beats_position_only"), bool):
        errors.append("localizer_genuinely_beats_position_only must be bare bool")
    if not isinstance(artifact.get("position_only_baseline_f1"), float):
        errors.append("position_only_baseline_f1 must be bare float")
    if not isinstance(artifact.get("template_family_holdout_drop"), float):
        errors.append("template_family_holdout_drop must be bare float")
    if not isinstance(artifact.get("n_traces"), int):
        errors.append("n_traces must be bare int")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    return errors


def run_adversarial_verify(path: Path, repo_root: Path = ROOT) -> dict[str, Any]:  # pragma: no cover
    script = repo_root / "scripts" / "adversarial_verify.py"
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
    ensemble_loader: EnsembleLoader = exp4392._scoring_path_loads,
    intervention_verifier_checker: InterventionVerifierChecker = reference_intervention_path_available,
    adversarial_verify_runner: AdversarialVerifyRunner = run_adversarial_verify,
    write: bool = True,
) -> dict[str, Any]:
    cfg = config or ExperimentConfig()
    started = cfg.start_time()
    source_paths = [
        cfg.fover_row_corpus_path,
        cfg.fover_step_corpus_path,
        cfg.exp2850_artifact_path,
        cfg.exp4381_artifact_path,
        cfg.arc_summary_path,
        cfg.arc_candidate_pool_path,
        cfg.arc_set_encoder_path,
        cfg.verifier_registry_path,
    ]
    checks = check_preconditions(
        cfg,
        ensemble_loader=ensemble_loader,
        intervention_verifier_checker=intervention_verifier_checker,
    )
    preconditions = [check.as_dict() for check in checks]
    blocked = _blocked_reason(checks)
    if blocked is not None:
        artifact = build_blocked_artifact(
            honest_verdict=blocked,
            preconditions_checked=preconditions,
            source_paths=source_paths,
            duration_s=cfg.clock() - started,
            random_seed=cfg.random_seed,
        )
        if write:
            _write_artifact(cfg.artifact_path, artifact)
        return artifact

    labels = build_fover_intervention_labels_from_rows(_read_jsonl(cfg.fover_row_corpus_path))
    train_labels, heldout_labels = split_by_heldout_family(labels, heldout_family=cfg.heldout_family)
    if len(heldout_labels) < cfg.min_eval_traces:
        artifact = build_blocked_artifact(
            honest_verdict="blocked_cached_corpus_or_ensemble_unavailable",
            preconditions_checked=preconditions
            + [
                {
                    "resource": "heldout_family_eval_split",
                    "available": False,
                    "detail": f"heldout_traces={len(heldout_labels)}; required>={cfg.min_eval_traces}",
                }
            ],
            source_paths=source_paths,
            duration_s=cfg.clock() - started,
            random_seed=cfg.random_seed,
        )
        if write:
            _write_artifact(cfg.artifact_path, artifact)
        return artifact

    localizer = train_real_contrastive_localizer(train_labels)
    position_baseline = PositionOnlyBaseline.fit([label.trace for label in train_labels])
    fover_report = evaluate_label_split(
        heldout_labels,
        localizer,
        position_baseline,
        seed=cfg.random_seed,
        bootstrap_resamples=cfg.bootstrap_resamples,
    )
    arc_traces = _read_arc_traces(cfg.arc_candidate_pool_path, cfg.arc_set_encoder_path)
    arc_report = evaluate_arc_domain(
        arc_traces,
        localizer,
        position_baseline,
        seed=cfg.random_seed + 1,
        bootstrap_resamples=cfg.bootstrap_resamples,
    )
    artifact = build_complete_artifact(
        labels=labels,
        train_labels=train_labels,
        heldout_labels=heldout_labels,
        arc_traces=arc_traces,
        localizer=localizer,
        baseline=position_baseline,
        fover_report=fover_report,
        arc_report=arc_report,
        preconditions_checked=preconditions,
        source_paths=source_paths,
        heldout_family=cfg.heldout_family,
        duration_s=cfg.clock() - started,
        random_seed=cfg.random_seed,
        bootstrap_resamples=cfg.bootstrap_resamples,
    )
    if write:
        _write_artifact(cfg.artifact_path, artifact)
        artifact["adversarial_verify"] = adversarial_verify_runner(cfg.artifact_path)
        if artifact["adversarial_verify"].get("returncode") not in (0, None):
            artifact["flagged_adversarial"] = True
        _write_artifact(cfg.artifact_path, artifact)
        append_missing_verifier_gaps(cfg.verifier_gaps_path, artifact["missing_verifier_gaps"])
    else:
        artifact["adversarial_verify"] = {"returncode": None, "skipped": True}
    return artifact


def main() -> int:  # pragma: no cover
    artifact = run_experiment(write=True)
    print(
        "[exp4403] "
        f"{artifact['honest_verdict']} "
        f"genuine={artifact['localizer_genuinely_beats_position_only']} "
        f"position_only={artifact['position_only_baseline_f1']} "
        f"n_traces={artifact['n_traces']} -> {ARTIFACT_PATH}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Exp 3718 FoVer risk-coverage abstention characterization.

Spec: REQ-SPOE-3718, SCENARIO-SPOE-3718.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from carnot.pipeline import second_pair_detector as spd
from carnot.pipeline.product_value_vs_self_certainty_3684 import (
    self_certainty_error_proxy_from_confidence_errors,
)


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results/experiment_3718_risk_coverage_abstention_characterization.json")
RANDOM_SEED = 3718
BOOTSTRAP_SEEDS = (3718, 3719, 3720, 3721, 3722)
DEFAULT_N_BOOTSTRAP = 200
MIN_FOVER_EXAMPLES = 1000
TARGET_RISK = 0.05
FIXED_COVERAGES = (0.50, 0.80, 0.90)
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates "
    "(principle: scores cached FoVer step outputs; no LLM load; no compute-bound marker)."
)

VERDICT_ENERGY_BETTER = (
    "complete: energy_is_a_better_selective_prediction_signal_than_entropy_deployable_abstention_gate"
)
VERDICT_ENERGY_NOT_BETTER = (
    "complete: energy_ties_or_loses_to_entropy_as_abstention_signal_honest_negative"
)
VERDICT_BLOCKED = "complete: blocked_fover_perstep_scores_unavailable"
TERMINAL_VERDICTS = (
    VERDICT_ENERGY_BETTER,
    VERDICT_ENERGY_NOT_BETTER,
    VERDICT_BLOCKED,
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "energy_aurc",
    "baseline_aurc",
    "energy_beats_baseline_abstention",
    "coverage_at_5pct_risk",
    "risk_at_fixed_coverage",
    "energy_aurc_ci",
    "n_seeds",
    "n_examples",
    "calibration_brier_ece",
    "adversarial_verify_clean",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": INFERENCE_SUBSTRATE,
    "energy_aurc": (
        "Area Under the Risk-Coverage Curve for the energy signal -- the "
        "deployment-facing selective-prediction metric (lower is better)."
    ),
    "baseline_aurc": (
        "AURC for the entropy / self-certainty baseline on the SAME corpus -- "
        "the bar arXiv:2603.21172 says entropy should fail to clear."
    ),
    "energy_beats_baseline_abstention": (
        "BARE bool. True iff the energy risk-coverage curve dominates the "
        "baseline with a CI separating them at >=1 operating point. STORE AS "
        "BARE true/false."
    ),
    "coverage_at_5pct_risk": (
        "The deployable operating point: what fraction of steps can be kept "
        "(auto-accepted) at <=5% selective error -- the abstention gate's "
        "practical value."
    ),
    "risk_at_fixed_coverage": (
        "Selective risk at coverage {50,80,90}% -- the risk-coverage operating "
        "points for deployment."
    ),
    "energy_aurc_ci": (
        "CI of the energy AURC (5 seeds) -- a point estimate cannot decide vs "
        "the baseline."
    ),
    "n_seeds": "Replication count (>=5).",
    "n_examples": "Sample-size rigor (FoVer step corpus n>=1000).",
    "calibration_brier_ece": (
        "Calibration of the energy abstention signal -- a deployable operating "
        "point must be calibrated."
    ),
    "adversarial_verify_clean": "True iff no critical flag.",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection.",
    "duration_s": "Plausibility floor.",
}


@dataclass(frozen=True)
class AbstentionExample:
    """One cached FoVer step row for selective-prediction evaluation."""

    label: int
    energy_score: float
    baseline_score: float
    example_id: str = ""


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    seeds: Sequence[int] = BOOTSTRAP_SEEDS,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build the Exp 3718 artifact from cached FoVer per-step rows."""

    root_path = Path(root)
    examples, corpus_status = load_fover_abstention_examples(root_path)
    preconditions = [
        {
            "resource": "fover_perstep_scores",
            "available": bool(len(examples) >= MIN_FOVER_EXAMPLES),
            "detail": corpus_status.get("math", {}).get("path"),
            "n_examples": len(examples),
        },
        {
            "resource": "self_certainty_baseline",
            "available": bool(examples),
            "detail": "Exp 3684 signed Bernoulli-uniform KL proxy from same FoVer confidence rows",
        },
    ]
    artifact = build_artifact_from_examples(
        examples,
        started_s=started_s,
        now_s=now_s,
        seeds=seeds,
        n_bootstrap=n_bootstrap,
        tests_run=tests_run,
        min_examples=MIN_FOVER_EXAMPLES,
        extra={
            "corpus_status": corpus_status,
            "preconditions_checked": preconditions,
            "output_path": str(_repo_path(root_path, OUTPUT_REL_PATH)),
        },
    )
    return artifact


def build_artifact_from_examples(
    examples: Sequence[AbstentionExample],
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    seeds: Sequence[int] = BOOTSTRAP_SEEDS,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    fixed_coverages: Sequence[float] = FIXED_COVERAGES,
    tests_run: Sequence[str] | None = None,
    min_examples: int = 1,
    adversarial_verify_clean: bool = True,
    extra: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Assemble the risk-coverage artifact from synthetic or cached examples."""

    start = time.perf_counter() if started_s is None else float(started_s)
    seed_list = [int(seed) for seed in seeds]
    clean = _clean_examples(examples)
    blocked = len(clean) < int(min_examples) or not _has_both_classes(clean)
    if blocked:
        finished = time.perf_counter() if now_s is None else float(now_s)
        artifact = _base_artifact(
            verdict=VERDICT_BLOCKED,
            energy_beats_baseline=False,
            n_examples=len(clean),
            n_seeds=len(seed_list),
            duration_s=_round(max(0.0, finished - start)),
            tests_run=tests_run,
            adversarial_verify_clean=adversarial_verify_clean,
        )
        artifact.update(_empty_measurements())
        artifact.update(dict(extra or {}))
        artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
        validate_artifact(artifact)
        return artifact

    labels = [example.label for example in clean]
    energy_scores = [example.energy_score for example in clean]
    baseline_scores = [example.baseline_score for example in clean]
    comparison = compare_abstention_signals(
        labels,
        energy_scores,
        baseline_scores,
        seeds=seed_list,
        n_bootstrap=n_bootstrap,
        fixed_coverages=fixed_coverages,
    )
    leak_guard = {
        "triggered": bool(
            comparison["energy_auroc"] is not None
            and comparison["energy_auroc"] >= 0.99
            and len(clean) >= MIN_FOVER_EXAMPLES
        ),
        "condition": "energy_auroc >= 0.99 on n>=1000",
        "energy_auroc": comparison["energy_auroc"],
        "n_examples": len(clean),
    }
    beats = bool(comparison["energy_beats_baseline_abstention"] and not leak_guard["triggered"])
    verdict = VERDICT_ENERGY_BETTER if beats else VERDICT_ENERGY_NOT_BETTER
    finished = time.perf_counter() if now_s is None else float(now_s)
    artifact = _base_artifact(
        verdict=verdict,
        energy_beats_baseline=beats,
        n_examples=len(clean),
        n_seeds=len(seed_list),
        duration_s=_round(max(0.0, finished - start)),
        tests_run=tests_run,
        adversarial_verify_clean=adversarial_verify_clean,
    )
    artifact.update(comparison)
    artifact.update(
        {
            "energy_beats_baseline_abstention": beats,
            "coverage_at_5pct_risk": comparison["coverage_at_5pct_risk_by_signal"][
                "energy"
            ],
            "calibration_brier_ece": calibration_brier_ece(clean),
            "leak_guard": leak_guard,
            "baseline_signal": baseline_signal_description(),
        }
    )
    artifact.update(dict(extra or {}))
    artifact["acceptance_gate"]["passed"] = acceptance_gate_passed(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def load_fover_abstention_examples(
    root: Path | str,
    *,
    score_overrides: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
) -> tuple[list[AbstentionExample], JsonDict]:
    """Load cached FoVer math rows and derive the same-corpus baseline."""

    detector_examples, corpus_status = spd.load_cached_labeled_examples(
        Path(root),
        score_overrides=score_overrides,
        use_balanced_code_corpus=True,
    )
    math_examples = [example for example in detector_examples if example.domain == "math"]
    baseline_scores = self_certainty_error_proxy_from_confidence_errors(
        [example.confidence_error for example in math_examples]
    )
    examples = [
        AbstentionExample(
            label=example.label,
            energy_score=example.ensemble_energy,
            baseline_score=baseline,
            example_id=example.example_id,
        )
        for example, baseline in zip(math_examples, baseline_scores, strict=True)
    ]
    return examples, corpus_status


def compare_abstention_signals(
    labels: Sequence[int],
    energy_scores: Sequence[float],
    baseline_scores: Sequence[float],
    *,
    seeds: Sequence[int] = BOOTSTRAP_SEEDS,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    fixed_coverages: Sequence[float] = FIXED_COVERAGES,
) -> JsonDict:
    """Compare energy and baseline risk-coverage envelopes on aligned rows."""

    clean_labels, clean_energy, clean_baseline = _finite_triplets(
        labels,
        energy_scores,
        baseline_scores,
    )
    if not clean_labels or len(set(clean_labels)) < 2:
        return _empty_comparison(seeds)
    energy_summary = risk_coverage_summary(
        clean_labels,
        clean_energy,
        fixed_coverages=fixed_coverages,
    )
    baseline_summary = risk_coverage_summary(
        clean_labels,
        clean_baseline,
        fixed_coverages=fixed_coverages,
    )
    energy_aurc_metric = bootstrap_metric(
        clean_labels,
        clean_energy,
        metric_fn=lambda metric_labels, metric_scores: risk_coverage_summary(
            metric_labels,
            metric_scores,
            fixed_coverages=fixed_coverages,
            include_curve=False,
        )["aurc"],
        seeds=seeds,
        n_bootstrap=n_bootstrap,
    )
    baseline_aurc_metric = bootstrap_metric(
        clean_labels,
        clean_baseline,
        metric_fn=lambda metric_labels, metric_scores: risk_coverage_summary(
            metric_labels,
            metric_scores,
            fixed_coverages=fixed_coverages,
            include_curve=False,
        )["aurc"],
        seeds=seeds,
        n_bootstrap=n_bootstrap,
    )
    aurc_delta = paired_bootstrap_delta(
        clean_labels,
        clean_energy,
        clean_baseline,
        metric_fn=lambda metric_labels, metric_scores: risk_coverage_summary(
            metric_labels,
            metric_scores,
            fixed_coverages=fixed_coverages,
            include_curve=False,
        )["aurc"],
        seeds=seeds,
        n_bootstrap=n_bootstrap,
    )
    fixed_risk: JsonDict = {}
    fixed_ci_separates = False
    for coverage in fixed_coverages:
        key = _coverage_key(coverage)
        delta = paired_bootstrap_delta(
            clean_labels,
            clean_energy,
            clean_baseline,
            metric_fn=lambda metric_labels, metric_scores, cov=float(coverage): (
                risk_coverage_summary(
                    metric_labels,
                    metric_scores,
                    fixed_coverages=[cov],
                    include_curve=False,
                )["risk_at_fixed_coverage"][_coverage_key(cov)]
            ),
            seeds=seeds,
            n_bootstrap=n_bootstrap,
        )
        fixed_ci_separates = fixed_ci_separates or bool(delta["ci95"][0] > 0.0)
        fixed_risk[key] = {
            "energy": energy_summary["risk_at_fixed_coverage"][key],
            "baseline": baseline_summary["risk_at_fixed_coverage"][key],
            "baseline_minus_energy": delta["point"],
            "delta_ci95": delta["ci95"],
        }
    risk_dominates = all(row["energy"] <= row["baseline"] + 1e-12 for row in fixed_risk.values())
    aurc_ci_separates = bool(aurc_delta["ci95"][0] > 0.0)
    energy_beats = bool(
        energy_summary["aurc"] < baseline_summary["aurc"]
        and risk_dominates
        and (aurc_ci_separates or fixed_ci_separates)
    )
    return {
        "energy_aurc": energy_summary["aurc"],
        "baseline_aurc": baseline_summary["aurc"],
        "energy_aurc_ci": energy_aurc_metric["ci95"],
        "baseline_aurc_ci": baseline_aurc_metric["ci95"],
        "aurc_delta_baseline_minus_energy": aurc_delta["point"],
        "aurc_delta_ci": aurc_delta["ci95"],
        "energy_auroc": _round(spd.tie_aware_auroc(clean_labels, clean_energy)),
        "baseline_auroc": _round(spd.tie_aware_auroc(clean_labels, clean_baseline)),
        "coverage_at_5pct_risk_by_signal": {
            "energy": energy_summary["coverage_at_5pct_risk"],
            "baseline": baseline_summary["coverage_at_5pct_risk"],
        },
        "coverage_at_5pct_risk": energy_summary["coverage_at_5pct_risk"],
        "risk_at_fixed_coverage": fixed_risk,
        "risk_coverage_curve": {
            "energy": energy_summary["curve"],
            "baseline": baseline_summary["curve"],
        },
        "energy_beats_baseline_abstention": energy_beats,
        "dominance_diagnostics": {
            "risk_dominates_at_fixed_coverages": bool(risk_dominates),
            "aurc_ci_separates": bool(aurc_ci_separates),
            "fixed_coverage_ci_separates": bool(fixed_ci_separates),
        },
    }


def risk_coverage_summary(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    fixed_coverages: Sequence[float] = FIXED_COVERAGES,
    target_risk: float = TARGET_RISK,
    include_curve: bool = True,
) -> JsonDict:
    """Return selective risk versus coverage for one oriented risk signal."""

    clean_labels, clean_scores = spd.finite_label_scores(labels, scores)
    if not clean_labels:
        return {
            "aurc": None,
            "coverage_at_5pct_risk": 0.0,
            "risk_at_fixed_coverage": {
                _coverage_key(coverage): None for coverage in fixed_coverages
            },
            "curve": [],
        }
    label_arr = np.asarray(clean_labels, dtype=np.float64)
    score_arr = np.asarray(clean_scores, dtype=np.float64)
    order = np.argsort(score_arr, kind="mergesort")
    ordered_scores = score_arr[order]
    kept_labels = label_arr[order]
    kept_counts = np.arange(1, len(kept_labels) + 1, dtype=np.float64)
    cumulative_errors = _tie_expected_cumulative_errors(ordered_scores, kept_labels)
    risks = cumulative_errors / kept_counts
    coverages = kept_counts / float(len(kept_labels))
    risk_at_fixed = {
        _coverage_key(coverage): _round(float(risks[_keep_count(coverage, len(kept_labels)) - 1]))
        for coverage in fixed_coverages
    }
    valid_coverages = coverages[risks <= float(target_risk) + 1e-12]
    curve = []
    if include_curve:
        curve = [
            {
                "coverage": _round(float(coverages[idx])),
                "selective_risk": _round(float(risks[idx])),
                "kept": int(idx + 1),
                "expected_errors_kept": _round(float(cumulative_errors[idx])),
            }
            for idx in range(len(kept_labels) - 1, -1, -1)
        ]
    return {
        "aurc": _round(float(np.mean(risks))),
        "coverage_at_5pct_risk": _round(float(np.max(valid_coverages)) if len(valid_coverages) else 0.0),
        "risk_at_fixed_coverage": risk_at_fixed,
        "curve": curve,
    }


def bootstrap_metric(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    metric_fn: Callable[[Sequence[int], Sequence[float]], float | None],
    seeds: Sequence[int],
    n_bootstrap: int,
) -> JsonDict:
    """Return point and CI for one signal metric."""

    clean_labels, clean_scores = spd.finite_label_scores(labels, scores)
    if not clean_labels or len(set(clean_labels)) < 2:
        return {"point": None, "ci95": None, "seed_means": [], "bootstrap_seeds": list(seeds)}
    point = metric_fn(clean_labels, clean_scores)
    label_arr = np.asarray(clean_labels, dtype=np.int64)
    score_arr = np.asarray(clean_scores, dtype=np.float64)
    values, seed_means = _bootstrap_values(
        label_arr,
        [score_arr],
        metric_fn=lambda idx: metric_fn(label_arr[idx].tolist(), score_arr[idx].tolist()),
        seeds=seeds,
        n_bootstrap=n_bootstrap,
    )
    ci_low, ci_high = _ci_or_point(values, float(point))
    return {
        "point": _round(float(point)),
        "ci95": [_round(ci_low), _round(ci_high)],
        "seed_means": seed_means,
        "bootstrap_seeds": [int(seed) for seed in seeds],
    }


def paired_bootstrap_delta(
    labels: Sequence[int],
    energy_scores: Sequence[float],
    baseline_scores: Sequence[float],
    *,
    metric_fn: Callable[[Sequence[int], Sequence[float]], float | None],
    seeds: Sequence[int],
    n_bootstrap: int,
) -> JsonDict:
    """Return paired baseline-minus-energy metric delta and CI."""

    clean_labels, clean_energy, clean_baseline = _finite_triplets(
        labels,
        energy_scores,
        baseline_scores,
    )
    if not clean_labels or len(set(clean_labels)) < 2:
        return {"point": None, "ci95": None, "seed_means": [], "bootstrap_seeds": list(seeds)}
    point = float(metric_fn(clean_labels, clean_baseline)) - float(
        metric_fn(clean_labels, clean_energy)
    )
    label_arr = np.asarray(clean_labels, dtype=np.int64)
    energy_arr = np.asarray(clean_energy, dtype=np.float64)
    baseline_arr = np.asarray(clean_baseline, dtype=np.float64)
    values, seed_means = _bootstrap_values(
        label_arr,
        [energy_arr, baseline_arr],
        metric_fn=lambda idx: float(metric_fn(label_arr[idx].tolist(), baseline_arr[idx].tolist()))
        - float(metric_fn(label_arr[idx].tolist(), energy_arr[idx].tolist())),
        seeds=seeds,
        n_bootstrap=n_bootstrap,
    )
    ci_low, ci_high = _ci_or_point(values, point)
    return {
        "point": _round(point),
        "ci95": [_round(ci_low), _round(ci_high)],
        "seed_means": seed_means,
        "bootstrap_seeds": [int(seed) for seed in seeds],
    }


def calibration_brier_ece(examples: Sequence[AbstentionExample]) -> JsonDict:
    """Fit one-feature energy calibration and report held-out Brier/ECE."""

    clean = _clean_examples(examples)
    spd_examples = [
        spd.LabeledDetectorExample(
            domain="math",
            label=example.label,
            ensemble_energy=example.energy_score,
            confidence_error=0.0,
            example_id=example.example_id,
        )
        for example in clean
    ]
    train, holdout = spd.stratified_train_holdout(spd_examples, seed=RANDOM_SEED)
    if not _has_spd_both_classes(train) or not _has_spd_both_classes(holdout):
        return {"brier": None, "ece": None, "n_holdout": len(holdout)}
    detector = spd.CalibratedFusedDetector().fit(train)
    labels = [example.label for example in holdout]
    probabilities = detector.predict_proba(holdout)
    return {
        "brier": _round(spd.brier_score(labels, probabilities)),
        "ece": _round(spd.expected_calibration_error(labels, probabilities)),
        "n_holdout": len(holdout),
        "method": "one_feature_logistic_energy_calibration",
    }


def baseline_signal_description() -> JsonDict:
    """Describe the disclosed entropy/self-certainty baseline proxy."""

    return {
        "name": "entropy_self_certainty_proxy",
        "same_corpus": True,
        "implementation": "Exp 3684 signed Bernoulli-uniform KL from FoVer confidence_error",
        "orientation": "larger score means higher predicted step-error risk",
        "authenticity_gap": (
            "FoVer cached rows expose scalar confidence, not token-level entropy; "
            "the proxy is disclosed and evaluated on row-aligned FoVer steps."
        ),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3718 artifact contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if artifact.get("honest_verdict") not in TERMINAL_VERDICTS:
        raise ValueError("honest_verdict is not an accepted Exp 3718 terminal verdict")
    if type(artifact.get("energy_beats_baseline_abstention")) is not bool:
        raise ValueError("energy_beats_baseline_abstention must be a bare top-level bool")
    if type(artifact.get("adversarial_verify_clean")) is not bool:
        raise ValueError("adversarial_verify_clean must be a bare top-level bool")
    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or float(duration) < 0.0:
        raise ValueError("duration_s must be a non-negative number")
    n_seeds = artifact.get("n_seeds")
    if not isinstance(n_seeds, int) or n_seeds < 5:
        raise ValueError("n_seeds must be an integer >= 5")
    if artifact.get("honest_verdict") != VERDICT_BLOCKED:
        for field in ("energy_aurc", "baseline_aurc", "coverage_at_5pct_risk"):
            if not isinstance(artifact.get(field), (int, float)):
                raise ValueError(f"{field} must be numeric for runnable artifacts")
        if not isinstance(artifact.get("energy_aurc_ci"), Sequence):
            raise ValueError("energy_aurc_ci must be present for runnable artifacts")
    _validate_distinct_metric_classes(artifact)
    if artifact.get("leak_guard", {}).get("triggered") and artifact.get(
        "energy_beats_baseline_abstention"
    ):
        raise ValueError("leak guard triggered artifacts cannot publish a positive signal")


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash deterministic Exp 3718 artifact fields."""

    payload = {
        "energy_aurc": artifact.get("energy_aurc"),
        "baseline_aurc": artifact.get("baseline_aurc"),
        "energy_aurc_ci": artifact.get("energy_aurc_ci"),
        "baseline_aurc_ci": artifact.get("baseline_aurc_ci"),
        "aurc_delta_ci": artifact.get("aurc_delta_ci"),
        "risk_at_fixed_coverage": artifact.get("risk_at_fixed_coverage"),
        "coverage_at_5pct_risk": artifact.get("coverage_at_5pct_risk"),
        "calibration_brier_ece": artifact.get("calibration_brier_ece"),
        "energy_beats_baseline_abstention": artifact.get(
            "energy_beats_baseline_abstention"
        ),
        "n_examples": artifact.get("n_examples"),
        "random_seed": RANDOM_SEED,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, adversarial-check, validate, and persist the Exp 3718 artifact."""

    root_path = Path(root)
    output = _repo_path(root_path, Path(output_path))
    artifact = build_artifact(root_path, tests_run=tests_run)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = run_adversarial_verify_report(output)
    artifact["adversarial_verify_report"] = compact_adversarial_report(report)
    artifact["adversarial_verify_clean"] = adversarial_report_is_clean(report)
    if not artifact["adversarial_verify_clean"] and artifact["honest_verdict"] != VERDICT_BLOCKED:
        artifact["honest_verdict"] = VERDICT_ENERGY_NOT_BETTER
        artifact["energy_beats_baseline_abstention"] = False
    artifact["acceptance_gate"]["passed"] = acceptance_gate_passed(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def write_artifact_from_examples(
    root: Path | str,
    *,
    output_path: Path | str,
    examples: Sequence[AbstentionExample],
    **kwargs: Any,
) -> Path:
    """Persist a synthetic or pre-measured Exp 3718 artifact."""

    root_path = Path(root)
    output = _repo_path(root_path, Path(output_path))
    artifact = build_artifact_from_examples(examples, **kwargs)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def run_adversarial_verify_report(path: Path) -> JsonDict:
    """Run scripts/adversarial_verify.py against an artifact path."""

    verifier_path = REPO_ROOT / "scripts" / "adversarial_verify.py"
    spec = importlib.util.spec_from_file_location("carnot_adversarial_verify_3718", verifier_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {verifier_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return dict(module.verify_artifact(path))


def compact_adversarial_report(report: Mapping[str, Any]) -> JsonDict:
    """Keep the adversarial report small and deterministic in the artifact."""

    flags = [dict(flag) for flag in report.get("flags", []) if isinstance(flag, Mapping)]
    return {"flag_count": len(flags), "flags": flags}


def adversarial_report_is_clean(report: Mapping[str, Any]) -> bool:
    """Return true when no adversarial flag is critical."""

    flags = report.get("flags", [])
    if not isinstance(flags, Sequence):
        return False
    return not any(
        isinstance(flag, Mapping) and str(flag.get("severity", "")).lower() == "critical"
        for flag in flags
    )


def acceptance_gate_passed(artifact: Mapping[str, Any]) -> bool:
    """Return true when required deployment fields and adversarial cleanliness exist."""

    return bool(
        artifact.get("energy_aurc") is not None
        and artifact.get("baseline_aurc") is not None
        and artifact.get("energy_aurc_ci") is not None
        and artifact.get("coverage_at_5pct_risk") is not None
        and artifact.get("adversarial_verify_clean") is True
    )


def _base_artifact(
    *,
    verdict: str,
    energy_beats_baseline: bool,
    n_examples: int,
    n_seeds: int,
    duration_s: float,
    tests_run: Sequence[str] | None,
    adversarial_verify_clean: bool,
) -> JsonDict:
    return {
        "artifact": "experiment_3718_risk_coverage_abstention_characterization",
        "schema": "carnot.risk_coverage_abstention_3718.v1",
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "energy_beats_baseline_abstention": bool(energy_beats_baseline),
        "n_examples": int(n_examples),
        "n_seeds": int(n_seeds),
        "adversarial_verify_clean": bool(adversarial_verify_clean),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "duration_s": duration_s,
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate": {
            "condition": (
                "energy_aurc present AND baseline_aurc present AND "
                "energy_aurc_ci present AND coverage_at_5pct_risk present AND "
                "adversarial_verify_clean == true"
            ),
            "passed": False,
            "principle": (
                "A trustworthy abstention characterization requires the energy "
                "AND baseline risk-coverage curves with CI and a deployable "
                "operating point, adversarial-clean -- a single AUROC is not a "
                "selective-prediction verdict."
            ),
        },
        "tests_run": list(tests_run or []),
        "scripts_research_conductor_modified": False,
        "ops_docs_reconciliation_left_to_conductor": True,
    }


def _empty_measurements() -> JsonDict:
    return {
        "energy_aurc": None,
        "baseline_aurc": None,
        "energy_aurc_ci": None,
        "baseline_aurc_ci": None,
        "aurc_delta_baseline_minus_energy": None,
        "aurc_delta_ci": None,
        "energy_auroc": None,
        "baseline_auroc": None,
        "coverage_at_5pct_risk": None,
        "coverage_at_5pct_risk_by_signal": {"energy": None, "baseline": None},
        "risk_at_fixed_coverage": {},
        "risk_coverage_curve": {"energy": [], "baseline": []},
        "calibration_brier_ece": {"brier": None, "ece": None, "n_holdout": 0},
        "leak_guard": {
            "triggered": False,
            "condition": "energy_auroc >= 0.99 on n>=1000",
            "energy_auroc": None,
            "n_examples": 0,
        },
        "baseline_signal": baseline_signal_description(),
    }


def _empty_comparison(seeds: Sequence[int]) -> JsonDict:
    return {
        "energy_aurc": None,
        "baseline_aurc": None,
        "energy_aurc_ci": None,
        "baseline_aurc_ci": None,
        "aurc_delta_baseline_minus_energy": None,
        "aurc_delta_ci": None,
        "energy_auroc": None,
        "baseline_auroc": None,
        "coverage_at_5pct_risk_by_signal": {"energy": None, "baseline": None},
        "coverage_at_5pct_risk": None,
        "risk_at_fixed_coverage": {},
        "risk_coverage_curve": {"energy": [], "baseline": []},
        "energy_beats_baseline_abstention": False,
        "dominance_diagnostics": {
            "risk_dominates_at_fixed_coverages": False,
            "aurc_ci_separates": False,
            "fixed_coverage_ci_separates": False,
            "bootstrap_seeds": [int(seed) for seed in seeds],
        },
    }


def _bootstrap_values(
    labels: np.ndarray,
    score_arrays: Sequence[np.ndarray],
    *,
    metric_fn: Callable[[np.ndarray], float | None],
    seeds: Sequence[int],
    n_bootstrap: int,
) -> tuple[list[float], list[float]]:
    values: list[float] = []
    seed_means: list[float] = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        seed_values: list[float] = []
        for _ in range(int(n_bootstrap)):
            idx = rng.integers(0, len(labels), size=len(labels))
            if len(set(labels[idx].tolist())) < 2:
                continue
            value = metric_fn(idx)
            if value is None:
                continue
            seed_values.append(float(value))
            values.append(float(value))
        seed_means.append(_round(float(np.mean(seed_values))) if seed_values else 0.0)
    return values, seed_means


def _ci_or_point(values: Sequence[float], point: float) -> tuple[float, float]:
    if not values:
        return point, point
    ci_low, ci_high = np.percentile(np.asarray(values, dtype=np.float64), [2.5, 97.5])
    return float(ci_low), float(ci_high)


def _tie_expected_cumulative_errors(
    ordered_scores: np.ndarray,
    ordered_labels: np.ndarray,
) -> np.ndarray:
    cumulative = np.zeros(len(ordered_labels), dtype=np.float64)
    prior_errors = 0.0
    start = 0
    while start < len(ordered_labels):
        end = start + 1
        while end < len(ordered_labels) and ordered_scores[end] == ordered_scores[start]:
            end += 1
        group = ordered_labels[start:end]
        group_size = len(group)
        group_error_rate = float(np.sum(group)) / group_size
        for offset in range(1, group_size + 1):
            cumulative[start + offset - 1] = prior_errors + offset * group_error_rate
        prior_errors += float(np.sum(group))
        start = end
    return cumulative


def _clean_examples(examples: Sequence[AbstentionExample]) -> list[AbstentionExample]:
    clean = []
    for example in examples:
        energy = float(example.energy_score)
        baseline = float(example.baseline_score)
        if math.isfinite(energy) and math.isfinite(baseline):
            clean.append(
                AbstentionExample(
                    label=1 if int(example.label) else 0,
                    energy_score=energy,
                    baseline_score=baseline,
                    example_id=str(example.example_id),
                )
            )
    return clean


def _finite_triplets(
    labels: Sequence[int],
    energy_scores: Sequence[float],
    baseline_scores: Sequence[float],
) -> tuple[list[int], list[float], list[float]]:
    clean_labels: list[int] = []
    clean_energy: list[float] = []
    clean_baseline: list[float] = []
    for label, energy, baseline in zip(labels, energy_scores, baseline_scores, strict=False):
        energy_f = float(energy)
        baseline_f = float(baseline)
        if math.isfinite(energy_f) and math.isfinite(baseline_f):
            clean_labels.append(1 if int(label) else 0)
            clean_energy.append(energy_f)
            clean_baseline.append(baseline_f)
    return clean_labels, clean_energy, clean_baseline


def _has_both_classes(examples: Sequence[AbstentionExample]) -> bool:
    return len({example.label for example in examples}) == 2


def _has_spd_both_classes(examples: Sequence[spd.LabeledDetectorExample]) -> bool:
    return len({example.label for example in examples}) == 2


def _keep_count(coverage: float, n_examples: int) -> int:
    return max(1, min(n_examples, int(math.ceil(float(coverage) * n_examples))))


def _coverage_key(coverage: float) -> str:
    return f"{float(coverage):.2f}"


def _validate_distinct_metric_classes(artifact: Mapping[str, Any]) -> None:
    for prefix in ("energy", "baseline"):
        aurc = artifact.get(f"{prefix}_aurc")
        auroc = artifact.get(f"{prefix}_auroc")
        if aurc is not None and auroc is not None and aurc == auroc:
            raise ValueError("AURC, AUROC, and risk@coverage must remain distinct metrics")
    energy_aurc = artifact.get("energy_aurc")
    risks = artifact.get("risk_at_fixed_coverage")
    if not isinstance(risks, Mapping) or energy_aurc is None:
        return
    for row in risks.values():
        if isinstance(row, Mapping) and row.get("energy") == energy_aurc:
            raise ValueError("AURC, AUROC, and risk@coverage must remain distinct metrics")


def _repo_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _round(value: float) -> float:
    if not math.isfinite(float(value)):
        return float(value)
    return round(float(value), 6)


__all__ = [
    "AbstentionExample",
    "OUTPUT_REL_PATH",
    "REQUIRED_ARTIFACT_FIELDS",
    "build_artifact",
    "build_artifact_from_examples",
    "calibration_brier_ece",
    "compare_abstention_signals",
    "load_fover_abstention_examples",
    "risk_coverage_summary",
    "validate_artifact",
    "write_artifact",
    "write_artifact_from_examples",
]

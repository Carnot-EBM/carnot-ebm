"""Exp 4419: SteerConf cached-feature detector calibration repair.

Spec refs: REQ-VERIFY-4419, SCENARIO-VERIFY-4419.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # pragma: no cover - the repository test environment has numpy.
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

from carnot.experiment_4386_cross_domain_detection_generalization import (
    bootstrap_auroc_ci95,
    ci_includes_chance,
    ci_lower_beats_chance,
    compute_auroc,
    random_score_auroc_control,
    round_float,
)
from carnot.experiment_4397_cross_domain_detection_calibration import (
    expected_calibration_error,
    hash_sources,
    risk_coverage_curve,
    run_adversarial_verify,
)
from carnot.experiment_4408_cross_domain_detection_calibration_repair import (
    ARC_CANDIDATE_POOL_PATH,
    ARC_DETECTOR_MODEL_PATH,
    ARC_RERANK_PATH,
    CODE_DUAL_CONDITION_PATH,
    CODE_FULL_ENSEMBLE_PATH,
    CODE_POOL_PATH,
    CODE_REWARD_ARTIFACT_PATH,
    EXP4397_PATH,
    FOVER_BASELINE_PATH,
    FOVER_CORPUS_PATH,
    FOVER_DUAL_CONDITION_PATH,
    GSM8K_BASELINE_PATH,
    GSM8K_POOL_PATH,
    SCA_INGESTION_PATH,
    VERIFIER_GAPS_PATH,
    SCAResult,
    ScoredCandidate,
    _base_rate_separation,
    _cited_upstream_artifacts,
    _clip_probability,
    _write_artifact,
    append_missing_verifier_gaps,
    load_raw_domain_rows,
    pool_record,
    semantic_confidence_aggregation,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = ROOT / "results" / "experiment_4419_steerconf_code_detection_calibration_repair.json"

RANDOM_SEED = 4419
RANDOM_SEEDS_USED = (4419,)
BOOTSTRAP_RESAMPLES = 2500
RANDOM_CONTROL_REPLICATES = 128
CALIBRATION_STEPS = 500
CALIBRATION_LR = 0.08
MIN_POWERED_N = 300
SPEC_REFS = ["REQ-VERIFY-4419", "SCENARIO-VERIFY-4419"]
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"

STEERED_FEATURE_NAMES = (
    "verifier_score",
    "conservative_confidence",
    "optimistic_confidence",
    "confidence_consistency",
    "steer_width",
    "task_rank_fraction",
    "task_margin_confidence",
)
REQUIRED_DOMAINS = ("fover", "gap4_arc", "code_humaneval", "gsm8k")

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "detection_calibrated_multi_domain",
    "detection_by_domain",
    "verifier_is_oracle",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "model_specs",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A win (steered confidence rescues code -> deployable "
        "multi-domain detector) and a clean null (code stays at chance -> "
        "contract retired, domain-bound gap) are BOTH decision-grade."
    ),
    "detection_calibrated_multi_domain": (
        "BARE bool: the capstone reads this; true iff detection AUROC CI95-lower "
        "> 0.5 on >=2 non-FoVer domains AND leave-one-domain-out ECE below the "
        "uncalibrated baseline -- the deployable multi-domain detector contract."
    ),
    "detection_by_domain": (
        "list of {domain, detection_auroc, auroc_ci95, ece_uncalibrated, "
        "ece_lodo_calibrated, risk_coverage, random_score_control, "
        "steered_confidence_added_auroc, n} -- the per-domain calibration "
        "record showing whether steered confidence moved code off chance vs "
        "the .407 0.577."
    ),
    "verifier_is_oracle": (
        "BARE bool=false -- the verifier score + the steered-confidence features "
        "are oracle-distinct detectors (no executable oracle defines "
        "code-correctness detection here)."
    ),
    "preconditions_checked": (
        "Records the cached pools + the steering-signal path + TRM-stand-down "
        "verified; pre-empts the silent-missing-resource fabrication mode."
    ),
    "random_seed": (
        "Determinism precondition for the steering features + the "
        "leave-one-domain-out fit + the bootstrap."
    ),
    "reproducibility_checksum": (
        "Hash of the pools + the steered-confidence features + the calibration "
        "config; lets a third party re-run."
    ),
    "model_specs": (
        "The verifier ensemble + the de-confounded pools + the steering config "
        "+ the cached candidate sources + n per domain; required methodology + "
        "the oracle-distinct declaration."
    ),
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime configuration for Exp 4419."""

    repo_root: Path = ROOT
    artifact_path: Path = ARTIFACT_PATH
    fover_corpus_path: Path = FOVER_CORPUS_PATH
    fover_baseline_path: Path = FOVER_BASELINE_PATH
    fover_dual_condition_path: Path = FOVER_DUAL_CONDITION_PATH
    arc_detector_model_path: Path = ARC_DETECTOR_MODEL_PATH
    arc_candidate_pool_path: Path = ARC_CANDIDATE_POOL_PATH
    arc_rerank_path: Path = ARC_RERANK_PATH
    code_1999_path: Path = CODE_POOL_PATH
    code_2838_path: Path = CODE_FULL_ENSEMBLE_PATH
    code_2839_path: Path = CODE_DUAL_CONDITION_PATH
    code_reward_artifact_path: Path = CODE_REWARD_ARTIFACT_PATH
    gsm8k_pool_path: Path = GSM8K_POOL_PATH
    gsm8k_baseline_path: Path = GSM8K_BASELINE_PATH
    sca_ingestion_path: Path = SCA_INGESTION_PATH
    exp4397_path: Path = EXP4397_PATH
    verifier_gaps_path: Path = VERIFIER_GAPS_PATH
    min_powered_n: int = MIN_POWERED_N
    random_seed: int = RANDOM_SEED
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES
    random_control_replicates: int = RANDOM_CONTROL_REPLICATES
    calibration_steps: int = CALIBRATION_STEPS
    calibration_learning_rate: float = CALIBRATION_LR
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at


@dataclass(frozen=True)
class SteeredCandidate:
    """One post-SCA candidate with cached-feature confidence steering probes."""

    domain: str
    task_id: str
    candidate_id: str
    is_correct: bool
    verifier_score: float
    conservative_confidence: float
    optimistic_confidence: float
    confidence_consistency: float
    steer_width: float
    task_rank_fraction: float
    task_margin_confidence: float
    valid_output: bool = True
    source: str = ""
    semantic_key: str | None = None

    @property
    def feature_vector(self) -> tuple[float, ...]:
        return (
            float(self.verifier_score),
            float(self.conservative_confidence),
            float(self.optimistic_confidence),
            float(self.confidence_consistency),
            float(self.steer_width),
            float(self.task_rank_fraction),
            float(self.task_margin_confidence),
        )


@dataclass(frozen=True)
class SteeringFeatureBundle:
    """Steered rows plus the precondition evidence for the steering path."""

    rows: list[SteeredCandidate]
    available: bool
    feature_names: tuple[str, ...]
    summary: dict[str, Any]


@dataclass(frozen=True)
class FeatureLogisticCalibrator:
    """A deterministic logistic calibrator over verifier and steering features."""

    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    weights: tuple[float, ...]
    bias: float
    trained_on_domains: tuple[str, ...]
    n_train: int

    def predict_one(self, features: Sequence[float]) -> float:
        logit = float(self.bias)
        for value, mean, scale, weight in zip(
            features,
            self.means,
            self.scales,
            self.weights,
            strict=True,
        ):
            logit += float(weight) * ((float(value) - float(mean)) / float(scale))
        return _sigmoid(logit)

    def predict_many(self, rows: Sequence[SteeredCandidate]) -> list[float]:
        return [self.predict_one(row.feature_vector) for row in rows]

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "means": [round_float(value) for value in self.means],
            "scales": [round_float(value) for value in self.scales],
            "weights": [round_float(value) for value in self.weights],
            "bias": round_float(self.bias),
            "trained_on_domains": list(self.trained_on_domains),
            "n_train": int(self.n_train),
        }


AdversarialVerifyRunner = Callable[[Path], dict[str, Any]]


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _logit(probability: float) -> float:
    prob = _clip_probability(probability)
    return math.log(prob / (1.0 - prob))


def _stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _rank_fraction(scores: Sequence[float], idx: int) -> float:
    if len(scores) <= 1:
        return 0.5
    target = float(scores[idx])
    less = sum(1 for score in scores if float(score) < target)
    equal = sum(1 for score in scores if float(score) == target)
    average_rank = less + (equal - 1) / 2.0
    return _clip01(average_rank / (len(scores) - 1))


def _margin_confidence(scores: Sequence[float], idx: int) -> float:
    if len(scores) <= 1:
        return 0.5
    target = float(scores[idx])
    lo = min(float(score) for score in scores)
    hi = max(float(score) for score in scores)
    span = hi - lo
    if span <= 1e-12:
        return 0.5
    best_other = max(float(score) for offset, score in enumerate(scores) if offset != idx)
    return _clip01(0.5 + 0.5 * ((target - best_other) / span))


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _mean(values: Sequence[float]) -> float:
    return sum(float(value) for value in values) / len(values) if values else 0.0


def derive_steered_confidence_features(
    rows: Sequence[ScoredCandidate],
) -> SteeringFeatureBundle:
    """Derive conservative/optimistic confidence probes from cached features only."""

    by_task: dict[tuple[str, str], list[ScoredCandidate]] = defaultdict(list)
    for row in rows:
        by_task[(row.domain, row.task_id)].append(row)

    steered: list[SteeredCandidate] = []
    for (_domain, _task_id), members in sorted(by_task.items(), key=lambda item: item[0]):
        scores = [_clip_probability(float(member.verifier_score)) for member in members]
        for idx, member in enumerate(members):
            base = scores[idx]
            rank = _rank_fraction(scores, idx)
            margin = _margin_confidence(scores, idx)
            uncertainty = 1.0 - abs(base - 0.5) * 2.0
            support = 0.55 * rank + 0.45 * margin
            conservative = _clip_probability(base - 0.18 * uncertainty - 0.10 * (1.0 - support))
            optimistic = _clip_probability(base + 0.18 * uncertainty + 0.10 * support)
            width = _clip01(optimistic - conservative)
            consistency = _clip01(1.0 - width)
            steered.append(
                SteeredCandidate(
                    domain=member.domain,
                    task_id=member.task_id,
                    candidate_id=member.candidate_id,
                    is_correct=member.is_correct,
                    verifier_score=base,
                    conservative_confidence=conservative,
                    optimistic_confidence=optimistic,
                    confidence_consistency=consistency,
                    steer_width=width,
                    task_rank_fraction=rank,
                    task_margin_confidence=margin,
                    valid_output=member.valid_output,
                    source=member.source,
                    semantic_key=member.semantic_key,
                )
            )

    score_values = [float(row.verifier_score) for row in steered]
    feature_signatures = {row.feature_vector for row in steered}
    score_span = max(score_values) - min(score_values) if score_values else 0.0
    width_values = [row.steer_width for row in steered]
    available = bool(len(steered) > 0 and score_span > 1e-12 and len(feature_signatures) > 1)
    summary = {
        "available": available,
        "method": "cached_feature_conservative_optimistic_prompt_steer_proxy",
        "feature_names": list(STEERED_FEATURE_NAMES),
        "rows": int(len(steered)),
        "domains": sorted({row.domain for row in steered}),
        "score_span": round_float(score_span),
        "steer_width_mean": round_float(_mean(width_values)),
        "steer_width_min": round_float(min(width_values) if width_values else 0.0),
        "steer_width_max": round_float(max(width_values) if width_values else 0.0),
        "oracle_distinct_inputs": [
            "verifier_score",
            "within_task_score_rank",
            "within_task_score_margin",
            "score_uncertainty",
        ],
        "forbidden_inputs": [
            "hidden_pass",
            "executable_correctness_label",
            "live_model_logits_generated_during_exp4419",
            "trm_training_updates",
        ],
    }
    return SteeringFeatureBundle(
        rows=steered,
        available=available,
        feature_names=STEERED_FEATURE_NAMES,
        summary=summary,
    )


def _fit_feature_logistic_numpy(
    features: Sequence[Sequence[float]],
    labels: Sequence[int | bool],
    *,
    trained_on_domains: Sequence[str],
    n_steps: int,
    learning_rate: float,
) -> FeatureLogisticCalibrator:
    matrix = np.asarray(features, dtype=float)  # type: ignore[union-attr]
    label_array = np.asarray([int(label) for label in labels], dtype=float)  # type: ignore[union-attr]
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales = np.where(scales <= 1e-12, 1.0, scales)  # type: ignore[union-attr]
    z_matrix = (matrix - means) / scales
    base_rate = _clip_probability(float(label_array.mean()))
    bias = _logit(base_rate)
    weights = np.zeros(matrix.shape[1], dtype=float)  # type: ignore[union-attr]
    for _step in range(max(0, int(n_steps))):
        logits = bias + z_matrix @ weights
        logits = np.clip(logits, -40.0, 40.0)  # type: ignore[union-attr]
        preds = 1.0 / (1.0 + np.exp(-logits))  # type: ignore[union-attr]
        errors = preds - label_array
        grad_w = (z_matrix.T @ errors) / len(label_array)
        grad_b = float(errors.mean())
        weights -= float(learning_rate) * grad_w
        bias -= float(learning_rate) * grad_b
    return FeatureLogisticCalibrator(
        feature_names=STEERED_FEATURE_NAMES,
        means=tuple(float(value) for value in means.tolist()),
        scales=tuple(float(value) for value in scales.tolist()),
        weights=tuple(float(value) for value in weights.tolist()),
        bias=float(bias),
        trained_on_domains=tuple(sorted(set(trained_on_domains))),
        n_train=int(len(labels)),
    )


def fit_feature_logistic_calibrator(
    features: Sequence[Sequence[float]],
    labels: Sequence[int | bool],
    *,
    trained_on_domains: Sequence[str],
    n_steps: int = CALIBRATION_STEPS,
    learning_rate: float = CALIBRATION_LR,
) -> FeatureLogisticCalibrator:
    if len(features) != len(labels):
        raise ValueError("features and labels must have the same length")
    if not features:
        width = len(STEERED_FEATURE_NAMES)
        return FeatureLogisticCalibrator(
            feature_names=STEERED_FEATURE_NAMES,
            means=(0.0,) * width,
            scales=(1.0,) * width,
            weights=(0.0,) * width,
            bias=0.0,
            trained_on_domains=tuple(sorted(set(trained_on_domains))),
            n_train=0,
        )
    if np is not None:
        return _fit_feature_logistic_numpy(
            features,
            labels,
            trained_on_domains=trained_on_domains,
            n_steps=n_steps,
            learning_rate=learning_rate,
        )

    feature_count = len(features[0])
    columns = [[float(row[idx]) for row in features] for idx in range(feature_count)]
    means = tuple(_mean(column) for column in columns)
    scales = tuple(
        math.sqrt(_mean([(value - means[idx]) ** 2 for value in column])) or 1.0
        for idx, column in enumerate(columns)
    )
    z_rows = [
        [(float(value) - means[idx]) / scales[idx] for idx, value in enumerate(row)]
        for row in features
    ]
    label_values = [int(label) for label in labels]
    bias = _logit(sum(label_values) / len(label_values))
    weights = [0.0] * feature_count
    for _step in range(max(0, int(n_steps))):  # pragma: no cover - numpy path used in CI.
        grad_w = [0.0] * feature_count
        grad_b = 0.0
        for z_row, label in zip(z_rows, label_values, strict=True):
            pred = _sigmoid(bias + sum(w * z for w, z in zip(weights, z_row, strict=True)))
            error = pred - label
            grad_b += error
            for idx, z_value in enumerate(z_row):
                grad_w[idx] += error * z_value
        inv_n = 1.0 / len(z_rows)
        for idx in range(feature_count):
            weights[idx] -= learning_rate * grad_w[idx] * inv_n
        bias -= learning_rate * grad_b * inv_n
    return FeatureLogisticCalibrator(
        feature_names=STEERED_FEATURE_NAMES,
        means=means,
        scales=scales,
        weights=tuple(weights),
        bias=float(bias),
        trained_on_domains=tuple(sorted(set(trained_on_domains))),
        n_train=len(labels),
    )


def leave_one_domain_out_steered_calibration(
    domain_rows: Mapping[str, Sequence[SteeredCandidate]],
    *,
    n_steps: int,
    learning_rate: float,
) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for held_out in sorted(domain_rows):
        train_features: list[tuple[float, ...]] = []
        train_labels: list[int] = []
        train_domains: list[str] = []
        for domain, rows in domain_rows.items():
            if domain == held_out:
                continue
            train_features.extend(row.feature_vector for row in rows)
            train_labels.extend(1 if row.is_correct else 0 for row in rows)
            train_domains.extend([domain] * len(rows))
        held_rows = list(domain_rows[held_out])
        labels = [1 if row.is_correct else 0 for row in held_rows]
        baseline = [_clip_probability(row.verifier_score) for row in held_rows]
        calibrator = fit_feature_logistic_calibrator(
            train_features,
            train_labels,
            trained_on_domains=train_domains,
            n_steps=n_steps,
            learning_rate=learning_rate,
        )
        calibrated = calibrator.predict_many(held_rows)
        reports[held_out] = {
            "ece_uncalibrated": round_float(expected_calibration_error(labels, baseline)),
            "ece_lodo_calibrated": round_float(expected_calibration_error(labels, calibrated)),
            "risk_coverage": risk_coverage_curve(labels, calibrated),
            "calibrated_scores": calibrated,
            "feature_logistic_calibrator": calibrator.as_dict(),
        }
    return reports


def summarize_domain(
    domain: str,
    rows: Sequence[SteeredCandidate],
    *,
    calibration_report: Mapping[str, Any],
    seed: int,
    bootstrap_resamples: int,
    random_control_replicates: int,
    min_powered_n: int,
) -> dict[str, Any]:
    labels = [1 if row.is_correct else 0 for row in rows]
    baseline_scores = [row.verifier_score for row in rows]
    calibrated_scores = list(calibration_report.get("calibrated_scores", []))
    if len(calibrated_scores) != len(rows):
        calibrated_scores = baseline_scores
    baseline_auroc = compute_auroc(labels, baseline_scores)
    steered_auroc = compute_auroc(labels, calibrated_scores)
    return {
        "domain": domain,
        "detection_auroc": round_float(steered_auroc),
        "baseline_verifier_auroc": round_float(baseline_auroc),
        "auroc_ci95": bootstrap_auroc_ci95(
            labels,
            calibrated_scores,
            seed=seed,
            resamples=bootstrap_resamples,
        ),
        "ece_uncalibrated": calibration_report.get("ece_uncalibrated"),
        "ece_lodo_calibrated": calibration_report.get("ece_lodo_calibrated"),
        "risk_coverage": list(calibration_report.get("risk_coverage", [])),
        "random_score_control": random_score_auroc_control(
            labels,
            seed=seed,
            replicates=random_control_replicates,
        ),
        "steered_confidence_added_auroc": round_float(steered_auroc - baseline_auroc),
        "n": int(len(rows)),
        "base_rate": round_float(sum(labels) / max(1, len(labels))),
        "steering_probe_summary": {
            "conservative_mean": round_float(_mean([row.conservative_confidence for row in rows])),
            "optimistic_mean": round_float(_mean([row.optimistic_confidence for row in rows])),
            "consistency_mean": round_float(_mean([row.confidence_consistency for row in rows])),
            "steer_width_mean": round_float(_mean([row.steer_width for row in rows])),
        },
        "feature_logistic_calibrator": calibration_report.get("feature_logistic_calibrator", {}),
        "score_orientation": "higher_steered_calibrated_score_means_more_likely_correct",
        "claim_scope": (
            "proper_pool_n>=300"
            if len(rows) >= min_powered_n
            else f"report_n_only_scope_claim; n={len(rows)} < {min_powered_n}"
        ),
    }


def detection_calibrated_multi_domain(domain_results: Sequence[Mapping[str, Any]]) -> bool:
    powered = [
        result
        for result in domain_results
        if str(result.get("claim_scope")) == "proper_pool_n>=300"
    ]
    non_fover_wins = [
        result
        for result in powered
        if str(result.get("domain")) != "fover"
        and ci_lower_beats_chance(result.get("auroc_ci95", []))
    ]
    ece_transfers = all(
        result.get("ece_lodo_calibrated") is not None
        and result.get("ece_uncalibrated") is not None
        and float(result["ece_lodo_calibrated"]) < float(result["ece_uncalibrated"])
        for result in powered
    )
    return len(non_fover_wins) >= 2 and bool(ece_transfers)


def domains_at_chance(domain_results: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        str(result["domain"])
        for result in domain_results
        if ci_includes_chance(result.get("auroc_ci95", []))
    ]


def missing_gap_entries(domain_results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for result in domain_results:
        if not ci_includes_chance(result.get("auroc_ci95", [])):
            continue
        domain = str(result["domain"])
        slug = domain.upper().replace("_", "-")
        entries.append(
            {
                "gap_id": f"GAP-4419-{slug}-STEERCONF-DETECTOR-CHANCE",
                "status": "open",
                "domain": domain,
                "failure_mode": (
                    f"SteerConf cached-feature detection AUROC CI95 includes chance on "
                    f"{domain}; n={result.get('n')}, baseline_auroc="
                    f"{result.get('baseline_verifier_auroc')}, steered_delta="
                    f"{result.get('steered_confidence_added_auroc')}."
                ),
                "missing_discriminator": (
                    "A domain-native verifier feature beyond self-reported or "
                    "cached-feature confidence consistency that separates correct "
                    "outputs from plausible wrong outputs."
                ),
                "candidate_design": (
                    "Build a domain-specific oracle-distinct verifier feature, then "
                    "rerun Exp 4419's same cached-pool SteerConf and LODO calibration gate."
                ),
                "priority": "high" if domain == "code_humaneval" else "medium",
            }
        )
    return entries


def _required_min_n(domain: str, min_powered_n: int) -> int:
    return 1 if domain == "fover" else int(min_powered_n)


def _precondition_records(
    sca_by_domain: Mapping[str, SCAResult],
    unavailable_domains: Sequence[Mapping[str, Any]],
    steering: SteeringFeatureBundle | None,
    *,
    min_powered_n: int,
) -> tuple[list[dict[str, Any]], bool, bool]:
    records: list[dict[str, Any]] = []
    unavailable_by_domain = {str(item.get("domain")): item.get("reason") for item in unavailable_domains}
    pools_available = True
    for domain in REQUIRED_DOMAINS:
        result = sca_by_domain.get(domain)
        required_n = _required_min_n(domain, min_powered_n)
        n = len(result.rows) if result is not None else 0
        available = bool(result is not None and n >= required_n)
        if not available:
            pools_available = False
        records.append(
            {
                "resource": f"{domain}_proper_pool",
                "domain": domain,
                "available": available,
                "raw_n": int(result.metadata["raw_n"]) if result is not None else 0,
                "n": int(n),
                "required_n": int(required_n),
                "detail": (
                    f"cached_pool_loaded; n={n} >= {required_n}"
                    if available
                    else str(unavailable_by_domain.get(domain, f"n={n} < {required_n}"))
                ),
            }
        )
    steering_available = bool(steering is not None and steering.available)
    records.append(
        {
            "resource": "steering_signal_path",
            "available": steering_available,
            "detail": (
                "cached verifier-score rank/margin conservative-vs-optimistic "
                "SteerConf proxy available"
                if steering_available
                else "no non-degenerate cached-feature steering signal derived"
            ),
        }
    )
    records.append(
        {
            "resource": "trm_training_stand_down",
            "available": True,
            "detail": "no TRM training or live inference invoked; cached calibration measurement only",
        }
    )
    return records, pools_available, steering_available


def _steering_summary_by_domain(domain_rows: Mapping[str, Sequence[SteeredCandidate]]) -> dict[str, Any]:
    return {
        domain: {
            "n": len(rows),
            "conservative_mean": round_float(_mean([row.conservative_confidence for row in rows])),
            "optimistic_mean": round_float(_mean([row.optimistic_confidence for row in rows])),
            "consistency_mean": round_float(_mean([row.confidence_consistency for row in rows])),
            "steer_width_mean": round_float(_mean([row.steer_width for row in rows])),
        }
        for domain, rows in sorted(domain_rows.items())
    }


def _source_paths(config: ExperimentConfig, loaded_sources: Sequence[Path]) -> list[Path]:
    return list(
        dict.fromkeys(
            [
                config.exp4397_path,
                config.sca_ingestion_path,
                config.code_1999_path,
                config.code_2838_path,
                config.code_2839_path,
                config.code_reward_artifact_path,
                config.gsm8k_pool_path,
                config.gsm8k_baseline_path,
                config.fover_corpus_path,
                config.fover_baseline_path,
                config.fover_dual_condition_path,
                config.arc_detector_model_path,
                config.arc_candidate_pool_path,
                config.arc_rerank_path,
                *loaded_sources,
            ]
        )
    )


def _model_specs(
    *,
    domain_results: Sequence[Mapping[str, Any]],
    sca_by_domain: Mapping[str, SCAResult],
    pools_built: Sequence[Mapping[str, Any]],
    unavailable_domains: Sequence[Mapping[str, Any]],
    steering_summary: Mapping[str, Any],
    bootstrap_resamples: int,
    random_control_replicates: int,
) -> dict[str, Any]:
    return {
        "verifier_ensemble_id": "exp4419_steerconf_cached_detector_suite",
        "verifier_is_oracle": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "live_generation": False,
        "trm_training": "stood_down_not_invoked",
        "deconfounded_pool_source": "Exp 4408 loaders plus Semantic Confidence Aggregation",
        "steering_config": {
            "method": "cached_feature_conservative_optimistic_prompt_steer_proxy",
            "reference": "SteerConf arXiv:2503.02863 via Exp 4409",
            "feature_names": list(STEERED_FEATURE_NAMES),
            "no_live_logits": True,
            "no_hidden_label_features": True,
        },
        "calibration_method": "leave_one_domain_out_feature_logistic",
        "bootstrap_method": "stratified_candidate_bootstrap",
        "bootstrap_resamples": int(bootstrap_resamples),
        "random_score_control_replicates": int(random_control_replicates),
        "cached_pools": {
            str(result["domain"]): {
                "n": int(result["n"]),
                "base_rate": result["base_rate"],
                "claim_scope": result["claim_scope"],
                "baseline_verifier_auroc": result["baseline_verifier_auroc"],
                "steered_confidence_added_auroc": result["steered_confidence_added_auroc"],
            }
            for result in domain_results
        },
        "semantic_grouping": {
            domain: {
                "raw_n": result.metadata["raw_n"],
                "n": result.metadata["n"],
                "duplicate_group_count": result.metadata["duplicate_group_count"],
                "semantic_conflict_groups": result.metadata["semantic_conflict_groups"],
            }
            for domain, result in sorted(sca_by_domain.items())
        },
        "steering_summary_by_domain": dict(steering_summary),
        "pools_built": [dict(pool) for pool in pools_built],
        "unavailable_domains": [dict(item) for item in unavailable_domains],
        "random_seed": RANDOM_SEED,
        "random_seeds_used": list(RANDOM_SEEDS_USED),
    }


def build_complete_artifact(
    *,
    domain_results: Sequence[Mapping[str, Any]],
    sca_by_domain: Mapping[str, SCAResult],
    steered_by_domain: Mapping[str, Sequence[SteeredCandidate]],
    pools_built: Sequence[Mapping[str, Any]],
    unavailable_domains: Sequence[Mapping[str, Any]],
    preconditions_checked: Sequence[Mapping[str, Any]],
    cited_upstream_artifacts: Sequence[Mapping[str, Any]],
    source_paths: Sequence[Path],
    steering_summary: Mapping[str, Any],
    duration_s: float,
    bootstrap_resamples: int,
    random_control_replicates: int,
) -> dict[str, Any]:
    calibrated = detection_calibrated_multi_domain(domain_results)
    verdict = (
        "success: steered_confidence_rescues_multi_domain_detector"
        if calibrated
        else "complete: clean_null_steered_confidence_does_not_rescue_code_detector"
    )
    base_rate = _base_rate_separation(sca_by_domain, [d for d in REQUIRED_DOMAINS if d != "fover"])
    feature_payload = {
        domain: [row.feature_vector for row in rows]
        for domain, rows in sorted(steered_by_domain.items())
    }
    checksum_payload = {
        "detection_calibrated_multi_domain": calibrated,
        "detection_by_domain": [dict(result) for result in domain_results],
        "steering_features": feature_payload,
        "random_seed": RANDOM_SEED,
    }
    return {
        "experiment": "experiment_4419_steerconf_code_detection_calibration_repair",
        "schema": "carnot.steerconf_code_detection_calibration_repair.v1",
        "honest_verdict": verdict,
        "detection_calibrated_multi_domain": bool(calibrated),
        "detection_by_domain": [dict(result) for result in domain_results],
        "domains_at_chance": domains_at_chance(domain_results),
        "base_rate_separation": base_rate,
        "steering_feature_summary": dict(steering_summary),
        "verifier_is_oracle": False,
        "preconditions_checked": [dict(item) for item in preconditions_checked],
        "cited_upstream_artifacts": [dict(item) for item in cited_upstream_artifacts],
        "random_seed": RANDOM_SEED,
        "random_seeds_used": list(RANDOM_SEEDS_USED),
        "reproducibility_checksum": hash_sources(source_paths, payload=checksum_payload),
        "model_specs": _model_specs(
            domain_results=domain_results,
            sca_by_domain=sca_by_domain,
            pools_built=pools_built,
            unavailable_domains=unavailable_domains,
            steering_summary=_steering_summary_by_domain(steered_by_domain),
            bootstrap_resamples=bootstrap_resamples,
            random_control_replicates=random_control_replicates,
        ),
        "missing_verifier_gaps": missing_gap_entries(domain_results),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "duration_s": round_float(duration_s, 3),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "positive_control_passed": bool(
            sum(
                1
                for result in domain_results
                if str(result.get("domain")) != "fover"
                and ci_lower_beats_chance(result.get("auroc_ci95", []))
            )
            >= 2
        ),
    }


def build_blocked_artifact(
    *,
    honest_verdict: str,
    sca_by_domain: Mapping[str, SCAResult],
    pools_built: Sequence[Mapping[str, Any]],
    unavailable_domains: Sequence[Mapping[str, Any]],
    preconditions_checked: Sequence[Mapping[str, Any]],
    cited_upstream_artifacts: Sequence[Mapping[str, Any]],
    source_paths: Sequence[Path],
    steering_summary: Mapping[str, Any],
    duration_s: float,
) -> dict[str, Any]:
    base_rate = _base_rate_separation(sca_by_domain, [])
    return {
        "experiment": "experiment_4419_steerconf_code_detection_calibration_repair",
        "schema": "carnot.steerconf_code_detection_calibration_repair.v1",
        "honest_verdict": honest_verdict,
        "detection_calibrated_multi_domain": False,
        "detection_by_domain": [],
        "domains_at_chance": [],
        "base_rate_separation": base_rate,
        "steering_feature_summary": dict(steering_summary),
        "verifier_is_oracle": False,
        "preconditions_checked": [dict(item) for item in preconditions_checked],
        "cited_upstream_artifacts": [dict(item) for item in cited_upstream_artifacts],
        "random_seed": RANDOM_SEED,
        "random_seeds_used": list(RANDOM_SEEDS_USED),
        "reproducibility_checksum": hash_sources(
            source_paths,
            payload={
                "blocked": honest_verdict,
                "base_rate": base_rate,
                "steering_summary": dict(steering_summary),
            },
        ),
        "model_specs": {
            "blocked_reason": honest_verdict,
            "pools_built": [dict(pool) for pool in pools_built],
            "unavailable_domains": [dict(item) for item in unavailable_domains],
            "trm_training": "stood_down_not_invoked",
            "live_generation": False,
            "verifier_is_oracle": False,
            "steering_config": {
                "method": "cached_feature_conservative_optimistic_prompt_steer_proxy",
                "feature_names": list(STEERED_FEATURE_NAMES),
                "available": bool(steering_summary.get("available", False)),
            },
        },
        "missing_verifier_gaps": [],
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "duration_s": round_float(duration_s, 3),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "positive_control_passed": False,
        "adversarial_verify": {"status": "not_run_blocked_preconditions"},
    }


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors = [f"missing:{field}" for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if not isinstance(artifact.get("detection_calibrated_multi_domain"), bool):
        errors.append("invalid:detection_calibrated_multi_domain")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("invalid:verifier_is_oracle")
    if not isinstance(artifact.get("detection_by_domain"), list):
        errors.append("invalid:detection_by_domain")
    if not isinstance(artifact.get("preconditions_checked"), list):
        errors.append("invalid:preconditions_checked")
    if not isinstance(artifact.get("model_specs"), dict):
        errors.append("invalid:model_specs")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("invalid:inference_substrate")
    return errors


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    adversarial_verify_runner: AdversarialVerifyRunner = run_adversarial_verify,
    write: bool = True,
) -> dict[str, Any]:
    cfg = config or ExperimentConfig()
    started = cfg.start_time()
    raw_by_domain, pools_built, unavailable_domains, loaded_sources = load_raw_domain_rows(cfg)
    sca_by_domain = {
        domain: semantic_confidence_aggregation(rows)
        for domain, rows in sorted(raw_by_domain.items())
    }
    all_sca_rows = [row for result in sca_by_domain.values() for row in result.rows]
    steering = derive_steered_confidence_features(all_sca_rows)
    preconditions, pools_available, steering_available = _precondition_records(
        sca_by_domain,
        unavailable_domains,
        steering,
        min_powered_n=cfg.min_powered_n,
    )
    source_paths = _source_paths(cfg, loaded_sources)
    cited = _cited_upstream_artifacts(cfg, loaded_sources)
    if not pools_available:
        artifact = build_blocked_artifact(
            honest_verdict="blocked_cached_pools_unavailable",
            sca_by_domain=sca_by_domain,
            pools_built=pools_built,
            unavailable_domains=unavailable_domains,
            preconditions_checked=preconditions,
            cited_upstream_artifacts=cited,
            source_paths=source_paths,
            steering_summary=steering.summary,
            duration_s=cfg.clock() - started,
        )
        if write:
            _write_artifact(cfg.artifact_path, artifact)
        return artifact
    if not steering_available:
        artifact = build_blocked_artifact(
            honest_verdict="blocked_no_steering_signal_path",
            sca_by_domain=sca_by_domain,
            pools_built=pools_built,
            unavailable_domains=unavailable_domains,
            preconditions_checked=preconditions,
            cited_upstream_artifacts=cited,
            source_paths=source_paths,
            steering_summary=steering.summary,
            duration_s=cfg.clock() - started,
        )
        if write:
            _write_artifact(cfg.artifact_path, artifact)
        return artifact

    steered_by_domain: dict[str, list[SteeredCandidate]] = defaultdict(list)
    for row in steering.rows:
        steered_by_domain[row.domain].append(row)
    calibration = leave_one_domain_out_steered_calibration(
        steered_by_domain,
        n_steps=cfg.calibration_steps,
        learning_rate=cfg.calibration_learning_rate,
    )
    domain_results = [
        summarize_domain(
            domain,
            rows,
            calibration_report=calibration.get(domain, {}),
            seed=cfg.random_seed,
            bootstrap_resamples=cfg.bootstrap_resamples,
            random_control_replicates=cfg.random_control_replicates,
            min_powered_n=cfg.min_powered_n,
        )
        for domain, rows in sorted(steered_by_domain.items())
    ]
    artifact = build_complete_artifact(
        domain_results=domain_results,
        sca_by_domain=sca_by_domain,
        steered_by_domain=steered_by_domain,
        pools_built=pools_built,
        unavailable_domains=unavailable_domains,
        preconditions_checked=preconditions,
        cited_upstream_artifacts=cited,
        source_paths=source_paths,
        steering_summary=steering.summary,
        duration_s=cfg.clock() - started,
        bootstrap_resamples=cfg.bootstrap_resamples,
        random_control_replicates=cfg.random_control_replicates,
    )
    if write:
        if not artifact["detection_calibrated_multi_domain"]:
            append_missing_verifier_gaps(cfg.verifier_gaps_path, artifact["missing_verifier_gaps"])
        _write_artifact(cfg.artifact_path, artifact)
        artifact["adversarial_verify"] = adversarial_verify_runner(cfg.artifact_path)
        _write_artifact(cfg.artifact_path, artifact)
    else:
        artifact["adversarial_verify"] = {"returncode": None, "skipped": True}
    return artifact


def main() -> int:  # pragma: no cover - exercised through results/ CLI shim.
    artifact = run_experiment(write=True)
    print(
        "[exp4419] "
        f"{artifact['honest_verdict']} "
        f"detection_calibrated_multi_domain={artifact['detection_calibrated_multi_domain']} "
        f"domains={len(artifact['detection_by_domain'])} -> {ARTIFACT_PATH}",
        flush=True,
    )
    return 0

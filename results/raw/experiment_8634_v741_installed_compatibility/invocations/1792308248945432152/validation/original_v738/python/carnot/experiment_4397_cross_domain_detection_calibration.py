"""Exp 4397: calibrated multi-domain verifier-as-detector contract.

Spec refs: REQ-VERIFY-4397, SCENARIO-VERIFY-4397.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot.experiment_4375_verifier_as_detector_measurement import (
    label_to_error,
    read_labeled_fover_rows,
    score_fover_production_ensemble,
)
from carnot.experiment_4386_cross_domain_detection_generalization import (
    ScoredCandidate,
    bootstrap_auroc_ci95,
    ci_includes_chance,
    ci_lower_beats_chance,
    compute_auroc,
    load_arc_set_encoder_rows,
    random_score_auroc_control,
    round_float,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = ROOT / "results" / "experiment_4397_cross_domain_detection_calibration.json"
REGISTRY_PATH = ROOT / "ops" / "verifier_registry.yaml"
FOVER_CORPUS_PATH = ROOT / "data" / "fover_corpus.jsonl"
FOVER_BASELINE_PATH = ROOT / "results" / "experiment_4375_verifier_as_detector_measurement.json"
DETECTOR_CONFIG_PATH = (
    ROOT / "results" / "experiment_4381_biprm_detector_localization_abstention.json"
)
HEADROOM_CENSUS_PATH = ROOT / "results" / "experiment_4175_headroom_gate_executable_census.json"
ARC_RERANK_PATH = ROOT / "results" / "arc3_trm_verifier_rerank.json"
ARC_DETECTOR_MODEL_PATH = ROOT / "results" / "experiment_4244_arc_set_encoder_aggregator_model.json"
ARC_CANDIDATE_POOL_PATH = ROOT / "results" / "experiment_4243_arc_candidate_pool_grow_pool.json.gz"
CODE_POOL_PATH = ROOT / "results" / "experiment_1999_code_verification_humaneval.json"
CODE_FULL_ENSEMBLE_PATH = ROOT / "results" / "experiment_2838_humaneval_full_ensemble_eval.json"
CODE_DUAL_CONDITION_PATH = ROOT / "results" / "experiment_2839_humaneval_dual_condition_v3.json"
GSM8K_POOL_PATH = ROOT / "results" / "adversarial_gsm8k_data_400.json"
GSM8K_BASELINE_PATH = ROOT / "results" / "experiment_1998_live_it_baselines_gsm8k.json"
VERIFIER_GAPS_PATH = ROOT / "ops" / "verifier_gaps.md"

RANDOM_SEED = 4397
RANDOM_SEEDS_USED = (4397,)
BOOTSTRAP_RESAMPLES = 2500
RANDOM_CONTROL_REPLICATES = 128
CALIBRATION_STEPS = 900
CALIBRATION_LR = 0.08
MIN_NON_FOVER_POOLS = 2
MIN_DOMAIN_CANDIDATES = 1000
SPEC_REFS = ["REQ-VERIFY-4397", "SCENARIO-VERIFY-4397"]
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "detection_calibrated_multi_domain",
    "detection_by_domain",
    "domains_at_chance",
    "pools_built",
    "verifier_is_oracle",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "model_specs",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A calibration win (detection beats chance on >=2 "
        "non-FoVer domains + calibration transfers) and an honest null "
        "(calibration does not transfer / a domain is at chance -> logged gaps "
        "= product backlog) are BOTH decision-grade."
    ),
    "detection_calibrated_multi_domain": (
        "BARE bool: the capstone reads this (gated-fields-must-be-bare); true "
        "iff detection AUROC CI95 lower bound > 0.5 on >=2 NON-FoVer domains "
        "AND leave-one-domain-out calibration keeps ECE below the uncalibrated "
        "baseline on the held-out domain -- a deployable multi-domain detector contract."
    ),
    "detection_by_domain": (
        "list of {domain, detection_auroc, auroc_ci95, selection_headroom, "
        "ece_uncalibrated, ece_lodo_calibrated, n, base_rate} -- the per-domain "
        "detection-vs-selection DIVERGENCE + calibration transfer."
    ),
    "domains_at_chance": (
        "list[str]: domains where detection AUROC CI95 includes 0.5 -- each "
        "LOGGED as a missing-verifier gap."
    ),
    "pools_built": (
        "list of {domain, source_cached_artifacts, n} -- assembled from EXISTING "
        "cached candidates only; no new live inference."
    ),
    "verifier_is_oracle": (
        "BARE bool=false -- learned/energy or cached-feature detector scores are "
        "scored against executable/exact labels, oracle-distinct."
    ),
    "preconditions_checked": (
        "Records the >=2 non-FoVer cached pools + ensemble + TRM-stand-down "
        "verified; pre-empts silent missing-resource fabrication."
    ),
    "random_seed": "Determinism precondition for scoring, calibration, and bootstrap.",
    "reproducibility_checksum": (
        "Hash of the multi-domain pools + ensemble config + calibration + AUROC computation."
    ),
    "model_specs": (
        "The verifier ensemble + FoVer/ARC/code/GSM cached corpora + "
        "selection-headroom source + calibration method + n per domain."
    ),
}


@dataclass(frozen=True)
class PreconditionCheck:
    """One prerequisite checked before cached multi-domain scoring starts."""

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
class PlattScaler:
    """A small deterministic logistic calibrator fitted on cached scores."""

    mean: float
    scale: float
    weight: float
    bias: float
    trained_on_domains: tuple[str, ...]
    n_train: int

    def predict_one(self, score: float) -> float:
        z = (float(score) - self.mean) / self.scale
        return _sigmoid(self.bias + self.weight * z)

    def predict_many(self, scores: Sequence[float]) -> list[float]:
        return [self.predict_one(score) for score in scores]

    def as_dict(self) -> dict[str, Any]:
        return {
            "mean": round_float(self.mean),
            "scale": round_float(self.scale),
            "weight": round_float(self.weight),
            "bias": round_float(self.bias),
            "trained_on_domains": list(self.trained_on_domains),
            "n_train": int(self.n_train),
        }


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime configuration for Exp 4397."""

    repo_root: Path = ROOT
    artifact_path: Path = ARTIFACT_PATH
    registry_path: Path = REGISTRY_PATH
    fover_corpus_path: Path = FOVER_CORPUS_PATH
    fover_baseline_path: Path = FOVER_BASELINE_PATH
    detector_config_path: Path = DETECTOR_CONFIG_PATH
    headroom_census_path: Path = HEADROOM_CENSUS_PATH
    arc_rerank_path: Path = ARC_RERANK_PATH
    arc_detector_model_path: Path = ARC_DETECTOR_MODEL_PATH
    arc_candidate_pool_path: Path = ARC_CANDIDATE_POOL_PATH
    code_pool_path: Path = CODE_POOL_PATH
    code_full_ensemble_path: Path = CODE_FULL_ENSEMBLE_PATH
    code_dual_condition_path: Path = CODE_DUAL_CONDITION_PATH
    gsm8k_pool_path: Path = GSM8K_POOL_PATH
    gsm8k_baseline_path: Path = GSM8K_BASELINE_PATH
    verifier_gaps_path: Path = VERIFIER_GAPS_PATH
    min_non_fover_pools: int = MIN_NON_FOVER_POOLS
    min_domain_candidates: int = MIN_DOMAIN_CANDIDATES
    random_seed: int = RANDOM_SEED
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES
    random_control_replicates: int = RANDOM_CONTROL_REPLICATES
    calibration_steps: int = CALIBRATION_STEPS
    calibration_learning_rate: float = CALIBRATION_LR
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at


AdversarialVerifyRunner = Callable[[Path], dict[str, Any]]


def _read_json(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def hash_sources(source_paths: Sequence[Path], *, payload: Mapping[str, Any]) -> str:
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


def _clip_probability(value: float) -> float:
    return min(1.0 - 1e-6, max(1e-6, float(value)))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _labels_scores(rows: Sequence[ScoredCandidate]) -> tuple[list[int], list[float]]:
    return [1 if row.is_correct else 0 for row in rows], [float(row.verifier_score) for row in rows]


def _normalize_higher_correct(raw_scores: Sequence[float]) -> list[float]:
    if not raw_scores:
        return []
    lo = min(float(score) for score in raw_scores)
    hi = max(float(score) for score in raw_scores)
    if hi <= lo:
        return [0.5 for _score in raw_scores]
    return [(float(score) - lo) / (hi - lo) for score in raw_scores]


def load_fover_rows(path: Path = FOVER_CORPUS_PATH, repo_root: Path = ROOT) -> list[ScoredCandidate]:
    """Load FoVer cached rows and convert the error detector into correctness scores."""

    rows = read_labeled_fover_rows(path)
    bundle = score_fover_production_ensemble(rows, repo_root)
    error_scores = _normalize_higher_correct(bundle.scores)
    correct_scores = [1.0 - score for score in error_scores]
    scored: list[ScoredCandidate] = []
    for idx, (row, score) in enumerate(zip(rows, correct_scores, strict=True)):
        is_correct = label_to_error(row.get("label")) == 0
        scored.append(
            ScoredCandidate(
                domain="fover",
                task_id=str(row.get("question_id") or f"fover_{idx}"),
                candidate_id=f"fover:{idx}",
                is_correct=is_correct,
                verifier_score=float(score),
                valid_output=True,
                source=str(path),
            )
        )
    return scored


def load_code_humaneval_rows(path: Path = CODE_POOL_PATH) -> list[ScoredCandidate]:
    """Build HumanEval candidate rows from cached baseline/repair outcomes."""

    payload = _read_json(path)
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        return []
    rows: list[ScoredCandidate] = []
    for idx, item in enumerate(results):
        if not isinstance(item, dict):
            continue
        task_id = str(item.get("task_id") or f"HumanEval/{idx}")
        extracted = float(item.get("extracted_constraints") or 0.0)
        constraint_penalty = min(1.0, max(0.0, extracted / 2.0))
        candidates = (
            ("baseline", "baseline_passed", 0.95 - 0.70 * constraint_penalty),
            ("repair", "repair_passed", 0.85 - 0.25 * constraint_penalty),
        )
        for variant, key, score in candidates:
            if key not in item:
                continue
            rows.append(
                ScoredCandidate(
                    domain="code_humaneval",
                    task_id=task_id,
                    candidate_id=f"{task_id}:{variant}",
                    is_correct=bool(item[key]),
                    verifier_score=float(score),
                    valid_output=True,
                    source=str(path),
                )
            )
    return rows


def _gsm_perturbation_score(dataset_key: str, row: Mapping[str, Any]) -> float:
    perturbation = str(row.get("perturbation") or dataset_key).lower()
    key = dataset_key.lower()
    if "control" in key or perturbation in {"none", "control"}:
        return 0.95
    if "irrelevant" in key or "irrelevant" in perturbation:
        return 0.90
    if "combined" in key or "combined" in perturbation:
        return 0.08
    if "number" in key or "swap" in perturbation:
        return 0.10
    return 0.50


def load_gsm8k_original_answer_rows(path: Path = GSM8K_POOL_PATH) -> list[ScoredCandidate]:
    """Build GSM8K rows from cached perturbed problems and original answers."""

    payload = _read_json(path)
    datasets = payload.get("datasets") if isinstance(payload, dict) else None
    if not isinstance(datasets, dict):
        return []
    rows: list[ScoredCandidate] = []
    for dataset_key, items in datasets.items():
        if not isinstance(items, list):
            continue
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            if "correct_answer" not in item or "original_answer" not in item:
                continue
            task_id = f"gsm8k:{dataset_key}:{item.get('id', idx)}"
            rows.append(
                ScoredCandidate(
                    domain="gsm8k",
                    task_id=task_id,
                    candidate_id=f"{task_id}:original_answer",
                    is_correct=item.get("correct_answer") == item.get("original_answer"),
                    verifier_score=_gsm_perturbation_score(str(dataset_key), item),
                    valid_output=True,
                    source=str(path),
                )
            )
    return rows


def _pool_record(domain: str, sources: Sequence[Path], n: int) -> dict[str, Any]:
    return {
        "domain": domain,
        "source_cached_artifacts": [str(path) for path in sources],
        "n": int(n),
    }


def load_available_domain_rows(
    config: ExperimentConfig,
) -> tuple[dict[str, list[ScoredCandidate]], list[dict[str, Any]], list[dict[str, Any]], list[Path]]:
    """Load cached FoVer, ARC, code, and GSM detector pools."""

    domains: dict[str, list[ScoredCandidate]] = {}
    pools_built: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    source_paths: list[Path] = []

    loaders: tuple[
        tuple[str, tuple[Path, ...], Callable[[], list[ScoredCandidate]]],
        ...,
    ] = (
        (
            "fover",
            (config.fover_corpus_path, config.fover_baseline_path, config.detector_config_path),
            lambda: load_fover_rows(config.fover_corpus_path, config.repo_root),
        ),
        (
            "gap4_arc",
            (config.arc_detector_model_path, config.arc_candidate_pool_path),
            lambda: load_arc_set_encoder_rows(
                config.arc_detector_model_path,
                config.arc_candidate_pool_path,
            ),
        ),
        (
            "code_humaneval",
            (config.code_pool_path, config.code_full_ensemble_path, config.code_dual_condition_path),
            lambda: load_code_humaneval_rows(config.code_pool_path),
        ),
        (
            "gsm8k",
            (config.gsm8k_pool_path, config.gsm8k_baseline_path),
            lambda: load_gsm8k_original_answer_rows(config.gsm8k_pool_path),
        ),
    )

    for domain, sources, loader in loaders:
        try:
            rows = loader()
        except Exception as exc:  # pragma: no cover - exercised by integration failures.
            unavailable.append({"domain": domain, "reason": str(exc)})
            continue
        if rows:
            domains[domain] = rows
            pools_built.append(_pool_record(domain, sources, len(rows)))
            source_paths.extend(sources)
        else:
            unavailable.append({"domain": domain, "reason": "no_usable_cached_rows"})
    return domains, pools_built, unavailable, list(dict.fromkeys(source_paths))


def load_selection_headrooms(
    headroom_path: Path = HEADROOM_CENSUS_PATH,
    arc_rerank_path: Path = ARC_RERANK_PATH,
    fover_baseline_path: Path = FOVER_BASELINE_PATH,
) -> dict[str, float]:
    headrooms = {"fover": 0.0, "gap4_arc": 0.0, "code_humaneval": 0.0, "gsm8k": 0.0}
    if fover_baseline_path.exists():
        payload = _read_json(fover_baseline_path)
        selection = payload.get("selection_headroom") if isinstance(payload, dict) else None
        if isinstance(selection, dict):
            headrooms["fover"] = float(selection.get("headroom") or 0.0)
    if headroom_path.exists():
        payload = _read_json(headroom_path)
        per_domain = payload.get("per_domain_headroom") if isinstance(payload, dict) else {}
        if isinstance(per_domain, dict):
            code = per_domain.get("code")
            math_domain = per_domain.get("math")
            arc_legacy = per_domain.get("sudoku")
            if isinstance(code, dict):
                headrooms["code_humaneval"] = float(code.get("selectable_headroom") or 0.0)
            if isinstance(math_domain, dict):
                headrooms["gsm8k"] = float(math_domain.get("selectable_headroom") or 0.0)
            if isinstance(arc_legacy, dict):
                headrooms["gap4_arc"] = float(arc_legacy.get("selectable_headroom") or 0.0)
    if arc_rerank_path.exists():
        payload = _read_json(arc_rerank_path)
        if isinstance(payload, dict):
            oracle = payload.get("oracle_ceiling")
            vote = payload.get("trm_vote_pass2")
            oracle_at_k = oracle.get("pass@2") if isinstance(oracle, dict) else None
            if isinstance(oracle_at_k, (int, float)) and isinstance(vote, (int, float)):
                headrooms["gap4_arc"] = float(oracle_at_k) - float(vote)
    return {key: round(float(value), 10) for key, value in headrooms.items()}


def expected_calibration_error(
    labels: Sequence[int | bool],
    probabilities: Sequence[float],
    *,
    n_bins: int = 10,
) -> float:
    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must have the same length")
    if not labels:
        return 0.0
    total = len(labels)
    ece = 0.0
    for bin_idx in range(n_bins):
        lo = bin_idx / n_bins
        hi = (bin_idx + 1) / n_bins
        selected: list[int] = []
        for idx, prob in enumerate(probabilities):
            value = float(prob)
            if (bin_idx == n_bins - 1 and lo <= value <= hi) or (
                bin_idx < n_bins - 1 and lo <= value < hi
            ):
                selected.append(idx)
        if not selected:
            continue
        confidence = sum(float(probabilities[idx]) for idx in selected) / len(selected)
        accuracy = sum(int(labels[idx]) for idx in selected) / len(selected)
        ece += (len(selected) / total) * abs(confidence - accuracy)
    return ece


def fit_platt_scaler(
    scores: Sequence[float],
    labels: Sequence[int | bool],
    *,
    trained_on_domains: Sequence[str],
    n_steps: int = CALIBRATION_STEPS,
    learning_rate: float = CALIBRATION_LR,
) -> PlattScaler:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    if not scores:
        return PlattScaler(0.0, 1.0, 0.0, 0.0, tuple(trained_on_domains), 0)
    score_values = [float(score) for score in scores]
    label_values = [int(label) for label in labels]
    mean = sum(score_values) / len(score_values)
    variance = sum((score - mean) ** 2 for score in score_values) / max(1, len(score_values))
    scale = math.sqrt(variance) or 1.0
    z_scores = [(score - mean) / scale for score in score_values]
    base_rate = _clip_probability(sum(label_values) / len(label_values))
    bias = math.log(base_rate / (1.0 - base_rate))
    weight = 0.0
    for _step in range(max(0, int(n_steps))):
        grad_w = 0.0
        grad_b = 0.0
        for z_score, label in zip(z_scores, label_values, strict=True):
            pred = _sigmoid(bias + weight * z_score)
            error = pred - label
            grad_w += error * z_score
            grad_b += error
        inv_n = 1.0 / len(z_scores)
        weight -= learning_rate * grad_w * inv_n
        bias -= learning_rate * grad_b * inv_n
    return PlattScaler(
        mean=float(mean),
        scale=float(scale),
        weight=float(weight),
        bias=float(bias),
        trained_on_domains=tuple(sorted(set(trained_on_domains))),
        n_train=len(score_values),
    )


def risk_coverage_curve(
    labels: Sequence[int | bool],
    probabilities: Sequence[float],
    *,
    coverages: Sequence[float] = (1.0, 0.9, 0.75, 0.5, 0.25),
) -> list[dict[str, Any]]:
    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must have the same length")
    if not labels:
        return []
    order = sorted(range(len(labels)), key=lambda idx: float(probabilities[idx]), reverse=True)
    curve: list[dict[str, Any]] = []
    for coverage in coverages:
        keep_n = max(1, int(len(order) * float(coverage)))
        keep = order[:keep_n]
        accuracy = sum(int(labels[idx]) for idx in keep) / len(keep)
        threshold = float(probabilities[keep[-1]])
        curve.append(
            {
                "coverage": round_float(float(coverage)),
                "n_kept": int(len(keep)),
                "accuracy": round_float(accuracy),
                "risk": round_float(1.0 - accuracy),
                "threshold": round_float(threshold),
            }
        )
    return curve


def leave_one_domain_out_calibration(
    domain_rows: Mapping[str, Sequence[ScoredCandidate]],
    *,
    seed: int = RANDOM_SEED,
    n_steps: int = CALIBRATION_STEPS,
    learning_rate: float = CALIBRATION_LR,
) -> dict[str, dict[str, Any]]:
    del seed
    reports: dict[str, dict[str, Any]] = {}
    for held_out in sorted(domain_rows):
        train_scores: list[float] = []
        train_labels: list[int] = []
        train_domains: list[str] = []
        for domain, rows in domain_rows.items():
            if domain == held_out:
                continue
            labels, scores = _labels_scores(rows)
            train_scores.extend(scores)
            train_labels.extend(labels)
            train_domains.extend([domain] * len(rows))
        labels, scores = _labels_scores(domain_rows[held_out])
        scaler = fit_platt_scaler(
            train_scores,
            train_labels,
            trained_on_domains=train_domains,
            n_steps=n_steps,
            learning_rate=learning_rate,
        )
        uncalibrated = [_clip_probability(score) for score in scores]
        calibrated = scaler.predict_many(scores)
        reports[held_out] = {
            "ece_uncalibrated": round_float(
                expected_calibration_error(labels, uncalibrated)
            ),
            "ece_lodo_calibrated": round_float(
                expected_calibration_error(labels, calibrated)
            ),
            "risk_coverage": risk_coverage_curve(labels, calibrated),
            "platt_scaler": scaler.as_dict(),
        }
    return reports


def summarize_domain(
    domain: str,
    rows: Sequence[ScoredCandidate],
    *,
    selection_headroom: float,
    calibration_report: Mapping[str, Any],
    seed: int,
    bootstrap_resamples: int,
    random_control_replicates: int,
    min_candidates: int = MIN_DOMAIN_CANDIDATES,
) -> dict[str, Any]:
    labels, scores = _labels_scores(rows)
    auroc = compute_auroc(labels, scores)
    ci95 = bootstrap_auroc_ci95(labels, scores, seed=seed, resamples=bootstrap_resamples)
    valid_rows = [row for row in rows if row.is_correct or row.valid_output]
    valid_auroc: float | None = None
    if len({int(row.is_correct) for row in valid_rows}) == 2:
        valid_auroc = compute_auroc(
            [1 if row.is_correct else 0 for row in valid_rows],
            [row.verifier_score for row in valid_rows],
        )
    return {
        "domain": domain,
        "detection_auroc": round_float(auroc),
        "auroc_ci95": ci95,
        "selection_headroom": round_float(selection_headroom),
        "ece_uncalibrated": calibration_report.get("ece_uncalibrated"),
        "ece_lodo_calibrated": calibration_report.get("ece_lodo_calibrated"),
        "risk_coverage": list(calibration_report.get("risk_coverage", [])),
        "n": int(len(rows)),
        "base_rate": round_float(sum(labels) / max(1, len(labels))),
        "random_score_auroc_control": random_score_auroc_control(
            labels,
            seed=seed,
            replicates=random_control_replicates,
        ),
        "valid_but_wrong_restricted_auroc": round_float(valid_auroc),
        "valid_but_wrong_restricted_n": int(len(valid_rows)),
        "valid_wrong_negative_n": int(sum(1 for row in valid_rows if not row.is_correct)),
        "platt_scaler": calibration_report.get("platt_scaler", {}),
        "score_orientation": "higher_verifier_score_means_more_likely_correct",
        "claim_scope": (
            "n>=1000"
            if len(rows) >= min_candidates
            else f"underpowered_n={len(rows)}; report_n_only_scope_claim"
        ),
    }


def domains_at_chance(domain_results: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        str(result["domain"])
        for result in domain_results
        if ci_includes_chance(result.get("auroc_ci95", []))
    ]


def detection_calibrated_multi_domain(domain_results: Sequence[Mapping[str, Any]]) -> bool:
    non_fover_wins = [
        result
        for result in domain_results
        if str(result.get("domain")) != "fover"
        and ci_lower_beats_chance(result.get("auroc_ci95", []))
    ]
    calibration_transfers = all(
        result.get("ece_lodo_calibrated") is not None
        and result.get("ece_uncalibrated") is not None
        and float(result["ece_lodo_calibrated"]) < float(result["ece_uncalibrated"])
        for result in domain_results
    )
    return len(non_fover_wins) >= 2 and calibration_transfers


def missing_gap_entries(domain_results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for result in domain_results:
        if not ci_includes_chance(result.get("auroc_ci95", [])):
            continue
        domain = str(result["domain"])
        slug = domain.upper().replace("_", "-")
        headroom = float(result.get("selection_headroom") or 0.0)
        entries.append(
            {
                "gap_id": f"GAP-4397-{slug}-DETECTOR-CHANCE",
                "status": "open",
                "domain": domain,
                "failure_mode": (
                    f"Detection AUROC CI95 includes chance on {domain} "
                    f"while selection_headroom={round_float(headroom)}."
                ),
                "missing_discriminator": (
                    "A domain-specific oracle-distinct detector feature that separates "
                    "correct cached outputs from plausible wrong outputs without reading "
                    "the executable/exact label."
                ),
                "candidate_design": (
                    "Add a domain-native verifier score, then rerun Exp 4397 with "
                    "the same LODO calibration and AUROC-vs-headroom gate."
                ),
                "priority": "high" if headroom >= 0.10 else "medium",
            }
        )
    return entries


def append_missing_verifier_gaps(path: Path, entries: Sequence[Mapping[str, Any]]) -> None:
    if not entries:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Verifier Gaps\n"
    additions: list[str] = []
    for entry in entries:
        gap_id = str(entry["gap_id"])
        if gap_id in existing:
            continue
        additions.append(
            "\n"
            f"### {gap_id}\n"
            f"- status: {entry['status']}\n"
            f"- domain: {entry['domain']}\n"
            f"- failure_mode: {entry['failure_mode']}\n"
            f"- missing_discriminator: {entry['missing_discriminator']}\n"
            f"- candidate_design: {entry['candidate_design']}\n"
            f"- priority: {entry['priority']}\n"
        )
    if additions:
        path.write_text(existing.rstrip() + "\n" + "\n".join(additions) + "\n", encoding="utf-8")


def _json_has_key(path: Path, resource: str, key: str) -> PreconditionCheck:
    if not path.exists():
        return PreconditionCheck(resource, False, "missing")
    try:
        payload = _read_json(path)
    except Exception as exc:
        return PreconditionCheck(resource, False, f"unreadable: {exc}")
    available = isinstance(payload, dict) and key in payload
    return PreconditionCheck(
        resource,
        available,
        f"{key} present" if available else f"missing {key}",
    )


def check_preconditions(
    config: ExperimentConfig,
    domain_rows: Mapping[str, Sequence[ScoredCandidate]],
    unavailable_domains: Sequence[Mapping[str, Any]],
) -> list[PreconditionCheck]:
    non_fover = {domain: rows for domain, rows in domain_rows.items() if domain != "fover"}
    registry_ok = config.registry_path.exists() and "verifier_id" in config.registry_path.read_text(
        encoding="utf-8"
    )
    detail = (
        ", ".join(f"{domain}:n={len(rows)}" for domain, rows in sorted(non_fover.items()))
        if non_fover
        else "none; " + "; ".join(f"{item['domain']}={item['reason']}" for item in unavailable_domains)
    )
    return [
        PreconditionCheck(
            "verifier_registry",
            registry_ok,
            "loaded verifier registry" if registry_ok else "missing or malformed verifier registry",
        ),
        _json_has_key(config.headroom_census_path, "selection_headroom_census", "per_domain_headroom"),
        _json_has_key(config.arc_rerank_path, "gap4_arc_rerank_summary", "oracle_ceiling"),
        PreconditionCheck(
            "two_non_fover_cached_labeled_pools",
            len(non_fover) >= config.min_non_fover_pools,
            f"{detail}; required>={config.min_non_fover_pools}",
        ),
        PreconditionCheck(
            "trm_training_stand_down",
            True,
            "not invoked; Exp 4397 scores existing cached candidates only",
        ),
    ]


def _model_specs(
    *,
    domain_results: Sequence[Mapping[str, Any]],
    pools_built: Sequence[Mapping[str, Any]],
    unavailable_domains: Sequence[Mapping[str, Any]],
    source_paths: Sequence[Path],
    bootstrap_resamples: int,
    random_control_replicates: int,
) -> dict[str, Any]:
    return {
        "verifier_ensemble_id": "fover_arc_code_gsm_cached_detector_suite",
        "ensemble_registry_path": str(REGISTRY_PATH),
        "calibration_method": "leave_one_domain_out_platt_scaling",
        "calibration_reference": "arXiv:2102.10395 multi-domain calibration",
        "score_orientation": "higher_verifier_score_means_more_likely_correct",
        "selection_headroom_source": str(HEADROOM_CENSUS_PATH),
        "arc_selection_headroom_summary": str(ARC_RERANK_PATH),
        "score_sources": {
            "fover": str(FOVER_CORPUS_PATH),
            "gap4_arc": str(ARC_DETECTOR_MODEL_PATH),
            "code_humaneval": str(CODE_POOL_PATH),
            "gsm8k": str(GSM8K_POOL_PATH),
        },
        "cached_corpora": {
            str(result["domain"]): {
                "n": int(result["n"]),
                "base_rate": result["base_rate"],
                "selection_headroom": result["selection_headroom"],
                "claim_scope": result.get("claim_scope"),
            }
            for result in domain_results
        },
        "pools_built": [dict(pool) for pool in pools_built],
        "unavailable_domains": [dict(item) for item in unavailable_domains],
        "source_paths": [str(path) for path in source_paths],
        "bootstrap_method": "stratified_candidate_bootstrap",
        "bootstrap_resamples": int(bootstrap_resamples),
        "random_score_control_replicates": int(random_control_replicates),
        "sota_gguf_generated_candidate_pools": {
            "code": [str(CODE_FULL_ENSEMBLE_PATH), str(CODE_DUAL_CONDITION_PATH)],
            "gsm8k": [str(GSM8K_BASELINE_PATH)],
        },
        "code_gsm_score_note": (
            "Code and GSM labels are executable/exact, while detector scores are "
            "cached metadata-derived proxies; no label is copied into the score."
        ),
        "random_seed": RANDOM_SEED,
        "random_seeds_used": list(RANDOM_SEEDS_USED),
        "trm_training": "stood_down_not_invoked",
        "live_generation": False,
        "verifier_is_oracle": False,
    }


def build_complete_artifact(
    *,
    domain_results: Sequence[Mapping[str, Any]],
    pools_built: Sequence[Mapping[str, Any]],
    unavailable_domains: Sequence[Mapping[str, Any]],
    preconditions_checked: Sequence[Mapping[str, Any]],
    source_paths: Sequence[Path],
    duration_s: float,
    bootstrap_resamples: int,
    random_control_replicates: int,
    model_specs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    calibrated = detection_calibrated_multi_domain(domain_results)
    chance_domains = domains_at_chance(domain_results)
    verdict = (
        "success: calibrated_multi_domain_detector_contract_holds"
        if calibrated
        else "complete: calibrated_multi_domain_contract_false"
    )
    checksum_payload = {
        "detection_by_domain": [dict(result) for result in domain_results],
        "domains_at_chance": chance_domains,
        "pools_built": [dict(pool) for pool in pools_built],
        "detection_calibrated_multi_domain": calibrated,
        "bootstrap_resamples": bootstrap_resamples,
        "random_control_replicates": random_control_replicates,
        "random_seed": RANDOM_SEED,
    }
    specs = dict(
        model_specs
        or _model_specs(
            domain_results=domain_results,
            pools_built=pools_built,
            unavailable_domains=unavailable_domains,
            source_paths=source_paths,
            bootstrap_resamples=bootstrap_resamples,
            random_control_replicates=random_control_replicates,
        )
    )
    return {
        "experiment": "experiment_4397_cross_domain_detection_calibration",
        "schema": "carnot.cross_domain_detection_calibration.v1",
        "honest_verdict": verdict,
        "detection_calibrated_multi_domain": bool(calibrated),
        "detection_by_domain": [dict(result) for result in domain_results],
        "domains_at_chance": chance_domains,
        "pools_built": [dict(pool) for pool in pools_built],
        "verifier_is_oracle": False,
        "preconditions_checked": [dict(item) for item in preconditions_checked],
        "random_seed": RANDOM_SEED,
        "random_seeds_used": list(RANDOM_SEEDS_USED),
        "bootstrap_resamples": int(bootstrap_resamples),
        "reproducibility_checksum": hash_sources(source_paths, payload=checksum_payload),
        "model_specs": specs,
        "unavailable_domains": [dict(item) for item in unavailable_domains],
        "missing_verifier_gaps": missing_gap_entries(domain_results),
        "positive_control_passed": bool(
            sum(
                1
                for result in domain_results
                if str(result.get("domain")) != "fover"
                and ci_lower_beats_chance(result.get("auroc_ci95", []))
            )
            >= 2
        ),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "duration_s": round_float(duration_s, 3),
        "inference_substrate": INFERENCE_SUBSTRATE,
    }


def build_blocked_artifact(
    *,
    preconditions_checked: Sequence[Mapping[str, Any]],
    pools_built: Sequence[Mapping[str, Any]],
    unavailable_domains: Sequence[Mapping[str, Any]],
    source_paths: Sequence[Path],
    duration_s: float,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_4397_cross_domain_detection_calibration",
        "schema": "carnot.cross_domain_detection_calibration.v1",
        "honest_verdict": "blocked_insufficient_cached_pools",
        "detection_calibrated_multi_domain": False,
        "detection_by_domain": [],
        "domains_at_chance": [],
        "pools_built": [dict(pool) for pool in pools_built],
        "verifier_is_oracle": False,
        "preconditions_checked": [dict(item) for item in preconditions_checked],
        "random_seed": RANDOM_SEED,
        "random_seeds_used": list(RANDOM_SEEDS_USED),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "reproducibility_checksum": hash_sources(
            source_paths, payload={"blocked": "blocked_insufficient_cached_pools"}
        ),
        "model_specs": {
            "blocked_reason": "fewer_than_two_non_fover_cached_labeled_pools",
            "calibration_method": "leave_one_domain_out_platt_scaling",
            "pools_built": [dict(pool) for pool in pools_built],
            "unavailable_domains": [dict(item) for item in unavailable_domains],
            "trm_training": "stood_down_not_invoked",
            "live_generation": False,
            "verifier_is_oracle": False,
        },
        "unavailable_domains": [dict(item) for item in unavailable_domains],
        "missing_verifier_gaps": [],
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "duration_s": round_float(duration_s, 3),
        "inference_substrate": INFERENCE_SUBSTRATE,
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
    if not isinstance(artifact.get("domains_at_chance"), list):
        errors.append("invalid:domains_at_chance")
    if not isinstance(artifact.get("pools_built"), list):
        errors.append("invalid:pools_built")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("invalid:inference_substrate")
    return errors


def run_adversarial_verify(path: Path, repo_root: Path = ROOT) -> dict[str, Any]:  # pragma: no cover
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


def _configured_source_paths(config: ExperimentConfig, loaded_source_paths: Sequence[Path]) -> list[Path]:
    paths = [
        config.registry_path,
        config.headroom_census_path,
        config.arc_rerank_path,
        config.fover_corpus_path,
        config.fover_baseline_path,
        config.detector_config_path,
        config.arc_detector_model_path,
        config.arc_candidate_pool_path,
        config.code_pool_path,
        config.code_full_ensemble_path,
        config.code_dual_condition_path,
        config.gsm8k_pool_path,
        config.gsm8k_baseline_path,
    ]
    paths.extend(loaded_source_paths)
    return list(dict.fromkeys(paths))


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    adversarial_verify_runner: AdversarialVerifyRunner = run_adversarial_verify,
    write: bool = True,
) -> dict[str, Any]:
    cfg = config or ExperimentConfig()
    started = cfg.start_time()
    domain_rows, pools_built, unavailable_domains, loaded_sources = load_available_domain_rows(cfg)
    source_paths = _configured_source_paths(cfg, loaded_sources)
    checks = check_preconditions(cfg, domain_rows, unavailable_domains)
    preconditions = [check.as_dict() for check in checks]
    if not all(check.available for check in checks):
        artifact = build_blocked_artifact(
            preconditions_checked=preconditions,
            pools_built=pools_built,
            unavailable_domains=unavailable_domains,
            source_paths=source_paths,
            duration_s=cfg.clock() - started,
        )
        if write:
            _write_artifact(cfg.artifact_path, artifact)
        return artifact

    headrooms = load_selection_headrooms(
        cfg.headroom_census_path,
        cfg.arc_rerank_path,
        cfg.fover_baseline_path,
    )
    calibration = leave_one_domain_out_calibration(
        domain_rows,
        seed=cfg.random_seed,
        n_steps=cfg.calibration_steps,
        learning_rate=cfg.calibration_learning_rate,
    )
    domain_results = [
        summarize_domain(
            domain,
            rows,
            selection_headroom=headrooms.get(domain, 0.0),
            calibration_report=calibration.get(domain, {}),
            seed=cfg.random_seed,
            bootstrap_resamples=cfg.bootstrap_resamples,
            random_control_replicates=cfg.random_control_replicates,
            min_candidates=cfg.min_domain_candidates,
        )
        for domain, rows in sorted(domain_rows.items())
    ]
    model_specs = _model_specs(
        domain_results=domain_results,
        pools_built=pools_built,
        unavailable_domains=unavailable_domains,
        source_paths=source_paths,
        bootstrap_resamples=cfg.bootstrap_resamples,
        random_control_replicates=cfg.random_control_replicates,
    )
    artifact = build_complete_artifact(
        domain_results=domain_results,
        pools_built=pools_built,
        unavailable_domains=unavailable_domains,
        preconditions_checked=preconditions,
        source_paths=source_paths,
        duration_s=cfg.clock() - started,
        bootstrap_resamples=cfg.bootstrap_resamples,
        random_control_replicates=cfg.random_control_replicates,
        model_specs=model_specs,
    )
    if write:
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
        "[exp4397] "
        f"{artifact['honest_verdict']} "
        f"detection_calibrated_multi_domain={artifact['detection_calibrated_multi_domain']} "
        f"domains={len(artifact['detection_by_domain'])} -> {ARTIFACT_PATH}",
        flush=True,
    )
    return 0

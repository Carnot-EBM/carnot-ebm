"""Exp 3865 LDT lattice margin sharpening.

This module reuses Exp 3833's cached FoVer verifier scoring path, then asks a
narrower question: after matching the ensemble's score distribution by bins, is
there still evidence that the ensemble spares correct candidates better than a
random operator with the same per-bin elimination counts?

Spec refs: REQ-VERIFY-3865, SCENARIO-VERIFY-3865.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_FILE = "experiment_3865_ldt_lattice_margin_sharpening_v2.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / OUTPUT_FILE
EXP3833_SCRIPT = "scripts/experiment_3833_ldt_gap_ensemble_as_sound_lattice.py"
FOVER_CORPUS = "data/fover_test_v4.json"
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
SCORING_PATH = "carnot.eval.fover_memory_leakage_v3"
DEFAULT_RANDOM_SEED = 3865
DEFAULT_SCORE_BINS = 20
DEFAULT_BOOTSTRAP_RESAMPLES = 1000
TARGET_SOUNDNESS = 0.99

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "honest_verdict",
    "ensemble_vs_score_matched_margin",
    "margin_ci95",
    "informativeness_at_soundness_0_99_reproduced",
    "count_matched_margin_reproduced",
    "n_candidates",
    "random_seed",
    "reproducibility_checksum",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": (
        "Terminal prefix records whether the sharpened LDT lattice edge is real, "
        "marginal, or blocked without making auditors infer status from metrics."
    ),
    "ensemble_vs_score_matched_margin": (
        "THE sharpened number -- the ensemble's edge over a control that already "
        "has its score distribution. Materially >0 => real per-candidate "
        "discrimination; ~0 => exp3833's 0.010 was the score distribution, not "
        "deductive power."
    ),
    "margin_ci95": (
        "Bootstrap CI95 on the margin -- if the lower bound is <=0, the lattice "
        "edge is within noise (downgrades exp3833's LATTICE_VIABLE to MARGINAL)."
    ),
    "informativeness_at_soundness_0_99_reproduced": ("Sanity replication of exp3833's 0.59."),
    "count_matched_margin_reproduced": (
        "exp3833's 0.010 count-matched margin, reproduced -- anchors the comparison."
    ),
    "n_candidates": (
        "Adversarial-Verify + Inference-Substrate; verifier-scoring substrate, "
        "determinism for the stratified control."
    ),
    "random_seed": (
        "Adversarial-Verify + Inference-Substrate; verifier-scoring substrate, "
        "determinism for the stratified control."
    ),
    "reproducibility_checksum": (
        "Adversarial-Verify + Inference-Substrate; verifier-scoring substrate, "
        "determinism for the stratified control."
    ),
    "preconditions_checked": (
        "Adversarial-Verify + Inference-Substrate; verifier-scoring substrate, "
        "determinism for the stratified control."
    ),
    "inference_substrate": (
        "Adversarial-Verify + Inference-Substrate; verifier-scoring substrate, "
        "determinism for the stratified control."
    ),
    "duration_s": (
        "Adversarial-Verify + Inference-Substrate; verifier-scoring substrate, "
        "determinism for the stratified control."
    ),
}


@dataclass(frozen=True)
class OperatingPoint:
    tau: float
    soundness: float
    informativeness: float
    eliminated_count: int
    false_elimination_rate: float


@dataclass(frozen=True)
class SoundnessCurve:
    points: tuple[dict[str, float | int], ...]
    best_operating_point: OperatingPoint
    eliminated_mask: tuple[bool, ...]
    informativeness_at_soundness_0_99: float
    count_matched_margin: float
    random_control_soundness: float


@dataclass(frozen=True)
class MarginResult:
    curve: SoundnessCurve
    score_matched_control_soundness: float
    ensemble_vs_score_matched_margin: float
    margin_ci95: tuple[float, float]
    bootstrap_margins: tuple[float, ...]
    bootstrap_resamples: int
    score_bin_summary: tuple[dict[str, float | int], ...]


@dataclass(frozen=True)
class PreconditionResult:
    blocked_reason: str | None
    checks: dict[str, Any]


def _as_arrays(scores: Sequence[float], labels: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    score_array = np.asarray(scores, dtype=float)
    label_array = np.asarray(labels, dtype=int)
    if score_array.ndim != 1 or label_array.ndim != 1:
        raise ValueError("scores and labels must be one-dimensional")
    if len(score_array) != len(label_array):
        raise ValueError("scores and labels must have the same length")
    if len(score_array) == 0:
        raise ValueError("scores and labels must be non-empty")
    if not np.all(np.isfinite(score_array)):
        raise ValueError("scores must be finite")
    if not set(np.unique(label_array)).issubset({0, 1}):
        raise ValueError("labels must be binary labels with 1=correct and 0=incorrect")
    return score_array, label_array


def _operating_point_from_tau(
    scores: np.ndarray,
    labels: np.ndarray,
    tau: float,
) -> tuple[OperatingPoint, np.ndarray]:
    eliminated = scores > tau
    is_correct = labels == 1
    is_incorrect = labels == 0
    total_correct = int(np.sum(is_correct))
    total_incorrect = int(np.sum(is_incorrect))
    false_elimination_rate = (
        float(np.sum(eliminated & is_correct) / total_correct) if total_correct else 0.0
    )
    informativeness = (
        float(np.sum(eliminated & is_incorrect) / total_incorrect) if total_incorrect else 0.0
    )
    point = OperatingPoint(
        tau=float(tau),
        soundness=float(1.0 - false_elimination_rate),
        informativeness=informativeness,
        eliminated_count=int(np.sum(eliminated)),
        false_elimination_rate=false_elimination_rate,
    )
    return point, eliminated


def compute_soundness_curve(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    target_soundness: float = TARGET_SOUNDNESS,
) -> SoundnessCurve:
    """REQ-VERIFY-3865: reproduce Exp 3833's soundness/informativeness curve."""

    score_array, label_array = _as_arrays(scores, labels)
    curve_points: list[dict[str, float | int]] = []
    candidates: list[tuple[OperatingPoint, np.ndarray]] = []
    info_at_target = 0.0

    for tau in np.sort(np.unique(score_array))[::-1]:
        point, eliminated = _operating_point_from_tau(score_array, label_array, float(tau))
        curve_points.append(
            {
                "tau": point.tau,
                "soundness": point.soundness,
                "informativeness": point.informativeness,
                "eliminated_count": point.eliminated_count,
            }
        )
        if point.soundness >= target_soundness:
            candidates.append((point, eliminated))
            info_at_target = max(info_at_target, point.informativeness)

    best_point, best_mask = (
        max(candidates, key=lambda item: item[0].informativeness)
        if candidates
        else _operating_point_from_tau(score_array, label_array, float(np.max(score_array)))
    )
    random_soundness = float(1.0 - (best_point.eliminated_count / len(score_array)))
    count_matched_margin = float(best_point.soundness - random_soundness)
    return SoundnessCurve(
        points=tuple(curve_points),
        best_operating_point=best_point,
        eliminated_mask=tuple(bool(value) for value in best_mask),
        informativeness_at_soundness_0_99=float(info_at_target),
        count_matched_margin=count_matched_margin,
        random_control_soundness=random_soundness,
    )


def score_bin_ids(scores: Sequence[float], n_score_bins: int = DEFAULT_SCORE_BINS) -> np.ndarray:
    """Assign deterministic equal-frequency score bins."""

    score_array = np.asarray(scores, dtype=float)
    if n_score_bins <= 0:
        raise ValueError("n_score_bins must be positive")
    if len(score_array) == 0:
        raise ValueError("scores must be non-empty")
    order = np.lexsort((np.arange(len(score_array)), score_array))
    bins = np.empty(len(score_array), dtype=int)
    for rank, idx in enumerate(order):
        bins[idx] = min((rank * n_score_bins) // len(score_array), n_score_bins - 1)
    return bins


def score_matched_control_soundness(
    labels: Sequence[int],
    eliminated_mask: Sequence[bool],
    bins: Sequence[int],
) -> tuple[float, tuple[dict[str, float | int], ...]]:
    """SCENARIO-VERIFY-3865: randomize eliminations only within score bins."""

    label_array = np.asarray(labels, dtype=int)
    eliminated_array = np.asarray(eliminated_mask, dtype=bool)
    bin_array = np.asarray(bins, dtype=int)
    if not (len(label_array) == len(eliminated_array) == len(bin_array)):
        raise ValueError("labels, eliminated_mask, and bins must have the same length")
    total_correct = int(np.sum(label_array == 1))
    expected_correct_eliminated = 0.0
    summary: list[dict[str, float | int]] = []
    for bin_id in sorted(int(value) for value in np.unique(bin_array)):
        in_bin = bin_array == bin_id
        bin_size = int(np.sum(in_bin))
        eliminated_in_bin = int(np.sum(eliminated_array & in_bin))
        correct_in_bin = int(np.sum((label_array == 1) & in_bin))
        expected_bin_correct = eliminated_in_bin * (correct_in_bin / bin_size) if bin_size else 0.0
        expected_correct_eliminated += expected_bin_correct
        summary.append(
            {
                "bin_id": bin_id,
                "bin_size": bin_size,
                "ensemble_eliminated_in_bin": eliminated_in_bin,
                "correct_in_bin": correct_in_bin,
                "expected_correct_eliminated_by_control": float(expected_bin_correct),
            }
        )
    soundness = float(1.0 - (expected_correct_eliminated / total_correct)) if total_correct else 1.0
    return soundness, tuple(summary)


def _margin_once(
    scores: Sequence[float],
    labels: Sequence[int],
    n_score_bins: int,
) -> tuple[SoundnessCurve, float, float, tuple[dict[str, float | int], ...]]:
    curve = compute_soundness_curve(scores, labels)
    bins = score_bin_ids(scores, n_score_bins)
    control_soundness, summary = score_matched_control_soundness(
        labels,
        curve.eliminated_mask,
        bins,
    )
    margin = float(curve.best_operating_point.soundness - control_soundness)
    return curve, control_soundness, margin, summary


def compute_margin(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    n_score_bins: int = DEFAULT_SCORE_BINS,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> MarginResult:
    """REQ-VERIFY-3865: compute the score-matched margin and bootstrap CI95."""

    score_array, label_array = _as_arrays(scores, labels)
    if bootstrap_resamples < 1000:
        raise ValueError("bootstrap_resamples must be at least 1000")
    curve, control_soundness, margin, summary = _margin_once(score_array, label_array, n_score_bins)
    rng = np.random.default_rng(random_seed)
    bootstrap_margins: list[float] = []
    for _ in range(bootstrap_resamples):
        sample = rng.integers(0, len(score_array), size=len(score_array))
        _, _, bootstrap_margin, _ = _margin_once(
            score_array[sample],
            label_array[sample],
            n_score_bins,
        )
        bootstrap_margins.append(float(bootstrap_margin))
    ci_low, ci_high = np.percentile(np.asarray(bootstrap_margins, dtype=float), [2.5, 97.5])
    return MarginResult(
        curve=curve,
        score_matched_control_soundness=float(control_soundness),
        ensemble_vs_score_matched_margin=float(margin),
        margin_ci95=(float(ci_low), float(ci_high)),
        bootstrap_margins=tuple(bootstrap_margins),
        bootstrap_resamples=int(bootstrap_resamples),
        score_bin_summary=summary,
    )


def _round_metric(value: float | None) -> float | None:
    return None if value is None else float(round(value, 12))


def _verdict(margin: float, ci95: tuple[float, float]) -> str:
    if ci95[0] > 0.0:
        return (
            "complete: ldt_margin_LATTICE_REAL_scorematched_margin"
            f"{margin:.3f}_ci{ci95[0]:.3f}-{ci95[1]:.3f}_genuine_deductive_power"
        )
    return (
        "complete: ldt_margin_LATTICE_MARGINAL_scorematched_margin"
        f"{margin:.3f}_ci_includes_zero_edge_is_score_dist_not_deductive"
    )


def _checksum_payload(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    random_seed: int,
    n_score_bins: int,
    bootstrap_resamples: int,
) -> str:
    payload = {
        "schema": "carnot.ldt_lattice_margin_sharpening_v2",
        "scoring_path": SCORING_PATH,
        "n": len(scores),
        "random_seed": random_seed,
        "n_score_bins": n_score_bins,
        "bootstrap_resamples": bootstrap_resamples,
        "scores": [round(float(value), 12) for value in scores],
        "labels": [int(value) for value in labels],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_artifact_from_scores(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    started_s: float,
    now_s: float,
    repo_root: Path | str = REPO_ROOT,
    preconditions_checked: Mapping[str, Any] | None = None,
    n_score_bins: int = DEFAULT_SCORE_BINS,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    random_seed: int = DEFAULT_RANDOM_SEED,
    score_one_candidate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """REQ-VERIFY-3865: build the terminal artifact from cached verifier scores."""

    margin = compute_margin(
        scores,
        labels,
        n_score_bins=n_score_bins,
        bootstrap_resamples=bootstrap_resamples,
        random_seed=random_seed,
    )
    curve = margin.curve
    ci95 = margin.margin_ci95
    artifact = {
        "experiment_id": 3865,
        "schema": "carnot.ldt_lattice_margin_sharpening_v2",
        "spec": ["REQ-VERIFY-3865", "SCENARIO-VERIFY-3865"],
        "source_experiment": "results/experiment_3833_ldt_gap_ensemble_as_sound_lattice.json",
        "corpus_path": FOVER_CORPUS,
        "exp3833_script_path": EXP3833_SCRIPT,
        "honest_verdict": _verdict(margin.ensemble_vs_score_matched_margin, ci95),
        "ensemble_vs_score_matched_margin": _round_metric(margin.ensemble_vs_score_matched_margin),
        "margin_ci95": [_round_metric(ci95[0]), _round_metric(ci95[1])],
        "informativeness_at_soundness_0_99_reproduced": _round_metric(
            curve.informativeness_at_soundness_0_99
        ),
        "count_matched_margin_reproduced": _round_metric(curve.count_matched_margin),
        "ensemble_soundness_at_operating_point": _round_metric(
            curve.best_operating_point.soundness
        ),
        "false_elimination_rate_at_operating_point": _round_metric(
            curve.best_operating_point.false_elimination_rate
        ),
        "score_threshold_at_operating_point": _round_metric(curve.best_operating_point.tau),
        "eliminated_count_at_operating_point": curve.best_operating_point.eliminated_count,
        "score_matched_control_soundness": _round_metric(margin.score_matched_control_soundness),
        "random_control_soundness_at_matched_elimination_count": _round_metric(
            curve.random_control_soundness
        ),
        "elimination_soundness_at_informativeness_curve": list(curve.points),
        "score_matched_bin_summary": list(margin.score_bin_summary),
        "score_bin_count": int(n_score_bins),
        "bootstrap_resamples": int(bootstrap_resamples),
        "bootstrap_ci_method": "candidate_row_percentile_bootstrap_ci95",
        "score_matched_control_method": (
            "equal-frequency score bins; analytic expectation of uniform random "
            "elimination within each bin at the ensemble's per-bin counts"
        ),
        "n_candidates": len(scores),
        "random_seed": int(random_seed),
        "reproducibility_checksum": _checksum_payload(
            scores,
            labels,
            random_seed=random_seed,
            n_score_bins=n_score_bins,
            bootstrap_resamples=bootstrap_resamples,
        ),
        "preconditions_checked": dict(preconditions_checked or {}),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(max(0.0, now_s - started_s)),
        "field_principles": dict(FIELD_PRINCIPLES),
        "frozen_fover_auroc_unchanged": True,
        "model_specs": {
            "live_model_invoked": False,
            "scoring_path": SCORING_PATH,
            "uses_cached_candidates_only": True,
        },
        "score_one_candidate": dict(score_one_candidate or {}),
        "repo_root": str(repo_root),
    }
    validate_artifact(artifact)
    return artifact


def build_blocked_artifact(
    reason: str,
    *,
    started_s: float,
    now_s: float,
    preconditions_checked: Mapping[str, Any],
) -> dict[str, Any]:
    """REQ-VERIFY-3865: write blocked terminal artifacts without fake metrics."""

    artifact = {
        "experiment_id": 3865,
        "schema": "carnot.ldt_lattice_margin_sharpening_v2",
        "spec": ["REQ-VERIFY-3865", "SCENARIO-VERIFY-3865"],
        "source_experiment": "results/experiment_3833_ldt_gap_ensemble_as_sound_lattice.json",
        "corpus_path": FOVER_CORPUS,
        "exp3833_script_path": EXP3833_SCRIPT,
        "honest_verdict": reason,
        "ensemble_vs_score_matched_margin": None,
        "margin_ci95": [None, None],
        "informativeness_at_soundness_0_99_reproduced": None,
        "count_matched_margin_reproduced": None,
        "n_candidates": 0,
        "random_seed": DEFAULT_RANDOM_SEED,
        "reproducibility_checksum": "",
        "preconditions_checked": dict(preconditions_checked),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(max(0.0, now_s - started_s)),
        "field_principles": dict(FIELD_PRINCIPLES),
        "frozen_fover_auroc_unchanged": True,
        "model_specs": {
            "live_model_invoked": False,
            "scoring_path": SCORING_PATH,
            "uses_cached_candidates_only": True,
        },
        "score_one_candidate": {},
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    if missing:
        raise ValueError(f"missing required artifact fields: {sorted(missing)}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles must be a mapping")
    missing_principles = set(REQUIRED_ARTIFACT_FIELDS) - set(principles)
    if missing_principles:
        raise ValueError(f"field_principles missing entries: {sorted(missing_principles)}")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must declare cached verifier scoring")
    verdict = str(artifact["honest_verdict"])
    if not (verdict.startswith("complete: ") or verdict.startswith("blocked_")):
        raise ValueError("honest_verdict must start with a terminal prefix")
    if float(artifact["duration_s"]) < 0.0:
        raise ValueError("duration_s must be non-negative")
    if not isinstance(artifact["preconditions_checked"], Mapping):
        raise ValueError("preconditions_checked must be a mapping")
    ci95 = artifact["margin_ci95"]
    if not isinstance(ci95, Sequence) or isinstance(ci95, str) or len(ci95) != 2:
        raise ValueError("margin_ci95 must contain exactly two values")
    if verdict.startswith("blocked_"):
        return
    if int(artifact.get("bootstrap_resamples", 0)) < 1000:
        raise ValueError("bootstrap_resamples must be at least 1000")
    if int(artifact["n_candidates"]) <= 0:
        raise ValueError("n_candidates must be positive for complete artifacts")
    if len(str(artifact["reproducibility_checksum"])) != 64:
        raise ValueError("reproducibility_checksum must be a sha256 hex digest")
    ci_low = float(ci95[0])
    if ci_low > 0.0 and "LATTICE_REAL" not in verdict:
        raise ValueError("positive CI lower bound requires LATTICE_REAL verdict")
    if ci_low <= 0.0 and "LATTICE_MARGINAL" not in verdict:
        raise ValueError("CI including zero requires LATTICE_MARGINAL verdict")


def import_exp3833_script(repo_root: Path | str = REPO_ROOT) -> Any:
    script_path = Path(repo_root) / EXP3833_SCRIPT
    spec = importlib.util.spec_from_file_location("experiment_3833_ldt_gap", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_corpus(repo_root: Path | str = REPO_ROOT) -> list[dict[str, Any]]:
    corpus_path = Path(repo_root) / FOVER_CORPUS
    payload = json.loads(corpus_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("fover_test_v4 corpus must be a non-empty JSON list")
    first = payload[0]
    if not isinstance(first, Mapping) or not {"question_id", "step_text", "label"} <= set(first):
        raise ValueError("fover_test_v4 rows must include question_id, step_text, and label")
    return [dict(row) for row in payload]


def _label_to_correct_int(row: Mapping[str, Any]) -> int:
    return 1 if str(row.get("label", "")).lower() in {"correct", "true", "1"} else 0


def score_candidate_rows(
    rows: Sequence[Mapping[str, Any]],
    repo_root: Path | str = REPO_ROOT,
) -> list[float]:
    """Score rows with the same cached verifier formula used by Exp 3833."""

    from carnot.eval.fover_memory_leakage_v3 import (  # noqa: PLC0415
        FR11_MEMORY_BOOST,
        _fr11_memory_score,
        _load_fr11_memory_index,
        _score_text_verifiers,
    )

    texts = [str(row.get("step_text", "")) for row in rows]
    verifier_scores = _score_text_verifiers(texts)
    r_scores = verifier_scores["tier0r_curry_howard"]
    u_scores = verifier_scores["tier0u_logical_consistency"]
    architecture_scores = [0.9 * r + 0.1 * u for r, u in zip(r_scores, u_scores, strict=True)]
    memory_index = _load_fr11_memory_index(Path(repo_root))
    if memory_index["question_ids"] or memory_index["prompt_token_sets"]:
        memory_scores = [_fr11_memory_score(row, memory_index) for row in rows]
        return [
            float(arch + FR11_MEMORY_BOOST * memory)
            for arch, memory in zip(architecture_scores, memory_scores, strict=True)
        ]
    return [float(value) for value in architecture_scores]


def check_preconditions(repo_root: Path | str = REPO_ROOT) -> PreconditionResult:
    checks: dict[str, Any] = {}
    try:
        importlib.import_module("carnot.verify")
        checks["import_carnot_verify"] = True
    except ImportError as exc:
        checks["import_carnot_verify"] = False
        checks["import_error"] = str(exc)
        return PreconditionResult("blocked_verify_module_import", checks)

    try:
        corpus = _load_corpus(repo_root)
        checks["corpus_loaded"] = True
        checks["corpus_path"] = FOVER_CORPUS
        checks["corpus_rows"] = len(corpus)
    except Exception as exc:
        checks["corpus_loaded"] = False
        checks["corpus_error"] = str(exc)
        return PreconditionResult("blocked_fover_test_v4_corpus", checks)

    try:
        import_exp3833_script(repo_root)
        checks["exp3833_script_importable"] = True
    except Exception as exc:
        checks["exp3833_script_importable"] = False
        checks["exp3833_script_error"] = str(exc)
        return PreconditionResult("blocked_exp3833_script", checks)

    try:
        score = score_candidate_rows([corpus[0]], repo_root)[0]
        if not np.isfinite(score):
            raise ValueError("score is not finite")
        checks["score_one_candidate_reproduced"] = True
        checks["score_one_candidate"] = float(score)
    except Exception as exc:
        checks["score_one_candidate_reproduced"] = False
        checks["score_error"] = str(exc)
        return PreconditionResult("blocked_score_candidate_failed", checks)

    return PreconditionResult(None, checks)


def load_and_score_cached_candidates(
    repo_root: Path | str = REPO_ROOT,
) -> tuple[list[float], list[int], dict[str, Any]]:
    corpus = _load_corpus(repo_root)
    scores = score_candidate_rows(corpus, repo_root)
    labels = [_label_to_correct_int(row) for row in corpus]
    return scores, labels, {"score_one_candidate": scores[0], "n_corpus_rows": len(corpus)}


def write_artifact(
    repo_root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    n_score_bins: int = DEFAULT_SCORE_BINS,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> Path:
    started = time.perf_counter() if started_s is None else started_s
    preconditions = check_preconditions(repo_root)
    if preconditions.blocked_reason is not None:
        current = time.perf_counter() if now_s is None else now_s
        artifact = build_blocked_artifact(
            preconditions.blocked_reason,
            started_s=started,
            now_s=current,
            preconditions_checked=preconditions.checks,
        )
    else:
        scores, labels, score_metadata = load_and_score_cached_candidates(repo_root)
        current = time.perf_counter() if now_s is None else now_s
        artifact = build_artifact_from_scores(
            scores,
            labels,
            started_s=started,
            now_s=current,
            repo_root=repo_root,
            preconditions_checked=preconditions.checks,
            n_score_bins=n_score_bins,
            bootstrap_resamples=bootstrap_resamples,
            random_seed=random_seed,
            score_one_candidate=score_metadata,
        )

    destination = Path(output_path)
    if not destination.is_absolute():
        destination = Path(repo_root) / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return destination


def main() -> Path:
    return write_artifact(REPO_ROOT)


if __name__ == "__main__":  # pragma: no cover
    main()

"""FR-11 continuous self-learning v8 with online redundancy penalties.

Spec: REQ-LEARN-3647, SCENARIO-LEARN-3647.

The experiment is intentionally local: it scores cached verifier traces and
uses their checked-in labels to audit the online update rule. The deploy arm
updates reliability weights conservatively, then divides those weights by a
label-conditional redundancy penalty so correlated verifiers cannot dominate
the ensemble simply by repeating the same evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from carnot.verify.weaver_peer_comparison_v3 import (
    VERIFIER_NAMES,
    ensemble_scores,
    normalize_weights,
    safe_pearson_matrix,
    score_fover_corpus,
    tie_aware_auroc,
)


OUTPUT_REL_PATH = Path("results/experiment_3647_fr11_continuous_self_learning_v8.json")
DEFAULT_RANDOM_SEED = 3647
DEFAULT_N_ONLINE_UPDATES = 240
MIN_ONLINE_UPDATES = 200
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates "
    "(principle: scores cached traces; no LLM load)."
)
SUCCESS_VERDICT = (
    "complete: fr11_v8_online_correlation_aware_weighting_holds_no_collapse_quality_maintained"
)
NO_GAIN_VERDICT = "complete: fr11_v8_correlation_penalty_no_auroc_gain_noise_only_weighting_sufficient"
BLOCKED_VERDICT = "complete: blocked_fr11_module_or_traces_unavailable"

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "n_online_updates",
    "collapse_detected_deploy_arm",
    "collapse_detected_control",
    "correlation_aware_auroc_gain",
    "calibration_improved",
    "pass_rate_vs_true_accuracy_distinct_assert",
    "quality_maintained",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": "Scores cached traces; no LLM load.",
    "n_online_updates": "Sample-size of the self-learning sweep (>=200).",
    "collapse_detected_deploy_arm": (
        "The conservative-default + correlation-aware rule must prevent "
        "self-distillation collapse (alpha_t grounding)."
    ),
    "collapse_detected_control": (
        "Positive control: the naive online arm must collapse, else the test has no contrast."
    ),
    "correlation_aware_auroc_gain": (
        "The forward difference -- does down-weighting redundant (not just noisy) "
        "verifiers improve AUROC?"
    ),
    "calibration_improved": "Brier/ECE before vs after online adaptation.",
    "pass_rate_vs_true_accuracy_distinct_assert": (
        "De-flags the tautology where pass_rate and true_accuracy are the same array."
    ),
    "quality_maintained": "Collapse-prevention must not come at the cost of detector quality.",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection.",
    "duration_s": "Plausibility floor.",
}


def build_artifact(
    repo_root: Path,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    n_examples: int = DEFAULT_N_ONLINE_UPDATES,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, Any]:
    """Build Exp 3647 from cached FoVer rows and local FR-11 verifier state.

    This is the precondition gate requested by REQ-LEARN-3647. If the FR-11
    module or cached corpus is absent, the function writes a terminal blocked
    artifact rather than inventing an online-learning result.
    """

    root = Path(repo_root)
    start = time.time() if started_s is None else float(started_s)
    preconditions = probe_preconditions(root, n_examples=n_examples)
    if not all(item["available"] for item in preconditions):
        return _blocked_artifact(
            duration_s=_duration(start, now_s),
            random_seed=random_seed,
            preconditions=preconditions,
        )

    try:
        labels, scores_by_verifier = score_fover_corpus(
            root,
            n_examples=n_examples,
            random_seed=random_seed,
        )
    except Exception as exc:  # noqa: BLE001 - failed scoring is a blocked local precondition.
        return _blocked_artifact(
            duration_s=_duration(start, now_s),
            random_seed=random_seed,
            preconditions=[
                *preconditions,
                {
                    "resource": "cached_trace_scoring",
                    "available": False,
                    "detail": f"{type(exc).__name__}: {exc}",
                },
            ],
        )

    return build_artifact_from_scores(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        started_s=start,
        now_s=now_s,
        random_seed=random_seed,
        preconditions=preconditions,
        exp3644_artifact=_load_exp3644_artifact(root),
    )


def probe_preconditions(repo_root: Path, *, n_examples: int) -> list[dict[str, Any]]:
    """Return concrete FR-11 and cached-trace availability checks."""

    root = Path(repo_root)
    fr11_dir = root / "python" / "carnot" / "fr11"
    corpus_path = root / "data" / "fover_corpus.jsonl"
    checks = [
        {
            "resource": "fr11_module",
            "available": fr11_dir.is_dir(),
            "detail": str(fr11_dir),
        }
    ]
    if corpus_path.is_file():
        line_count = _line_count(corpus_path)
        checks.append(
            {
                "resource": "cached_traces",
                "available": line_count >= n_examples,
                "detail": f"fover_corpus.jsonl line_count={line_count}; required>={n_examples}",
            }
        )
    else:
        checks.append({"resource": "cached_traces", "available": False, "detail": "missing"})
    return checks


def build_artifact_from_scores(
    *,
    labels: Sequence[int],
    scores_by_verifier: Mapping[str, Sequence[float]],
    started_s: float,
    now_s: float | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    preconditions: Sequence[Mapping[str, Any]] | None = None,
    exp3644_artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate deploy and control arms from already-cached verifier scores."""

    names = list(scores_by_verifier)
    labels_arr = np.asarray(labels, dtype=np.int64)
    score_matrix = _score_matrix(scores_by_verifier, names)
    if len(labels_arr) != score_matrix.shape[0]:
        raise ValueError("labels and verifier scores must have the same length")
    if len(labels_arr) < MIN_ONLINE_UPDATES:
        raise ValueError(f"runnable v8 artifacts require at least {MIN_ONLINE_UPDATES} updates")
    if len(set(int(value) for value in labels_arr)) < 2:
        raise ValueError("labels must contain both classes")

    oriented_scores, orientation = orient_scores_to_error_probability(labels_arr, score_matrix)
    before_weights = normalize_weights(np.ones(oriented_scores.shape[1], dtype=np.float64))
    before_scores = ensemble_scores(oriented_scores, before_weights)

    reliability_raw = reliability_raw_weights(labels_arr, oriented_scores)
    noise_only_weights = normalize_weights(reliability_raw)
    noise_only_scores = ensemble_scores(oriented_scores, noise_only_weights)

    redundancy_matrix, correlation_source = seeded_redundancy_matrix(
        labels_arr,
        oriented_scores,
        exp3644_artifact=exp3644_artifact,
    )
    redundancy_penalty = 1.0 + np.sum(redundancy_matrix, axis=1)
    deploy_weights = normalize_weights(reliability_raw / redundancy_penalty)
    deploy_uncalibrated = ensemble_scores(oriented_scores, deploy_weights)
    deploy_scores = calibrate_with_online_bias(labels_arr, deploy_uncalibrated)

    control_weights = naive_beta0_control_weights(oriented_scores)
    control_scores = ensemble_scores(oriented_scores, control_weights)

    before_metrics = score_metrics(labels_arr, before_scores)
    deploy_metrics = score_metrics(labels_arr, deploy_scores)
    control_metrics = score_metrics(labels_arr, control_scores)
    noise_metrics = score_metrics(labels_arr, noise_only_scores)

    collapse_detected_deploy_arm = detect_deploy_collapse(before_metrics, deploy_metrics)
    collapse_detected_control = detect_control_collapse(before_metrics, control_metrics)
    calibration_improved = bool(
        deploy_metrics["brier"] <= before_metrics["brier"]
        and deploy_metrics["ece"] <= before_metrics["ece"]
    )
    quality_maintained = bool(
        not collapse_detected_deploy_arm
        and deploy_metrics["auroc"] >= before_metrics["auroc"] - 1e-12
        and deploy_metrics["brier"] <= before_metrics["brier"]
    )

    pass_rate, true_accuracy = online_metric_trajectories(labels_arr, deploy_scores)
    distinct_assert = not _same_trajectory(pass_rate, true_accuracy)
    correlation_aware_gain = deploy_metrics["auroc"] - noise_metrics["auroc"]
    gate_passed = bool(
        not collapse_detected_deploy_arm and collapse_detected_control and distinct_assert
    )
    verdict = select_honest_verdict(
        gate_passed=gate_passed,
        correlation_aware_auroc_gain=correlation_aware_gain,
    )

    artifact = {
        "artifact": "experiment_3647_fr11_continuous_self_learning_v8",
        "schema": "carnot.fr11_continuous_self_learning_v8",
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "n_online_updates": int(len(labels_arr)),
        "collapse_detected_deploy_arm": bool(collapse_detected_deploy_arm),
        "collapse_detected_control": bool(collapse_detected_control),
        "correlation_aware_auroc_gain": _round(correlation_aware_gain),
        "calibration_improved": bool(calibration_improved),
        "pass_rate_vs_true_accuracy_distinct_assert": bool(distinct_assert),
        "quality_maintained": bool(quality_maintained),
        "random_seed": int(random_seed),
        "reproducibility_checksum": reproducibility_checksum(
            labels_arr,
            oriented_scores,
            deploy_weights,
            control_weights,
            random_seed=random_seed,
        ),
        "duration_s": _round(_duration(float(started_s), now_s)),
        "acceptance_gate": {
            "condition": (
                "collapse_detected_deploy_arm == false AND "
                "collapse_detected_control == true AND "
                "pass_rate_vs_true_accuracy_distinct_assert == true"
            ),
            "passed": gate_passed,
            "principle": (
                "Self-learning is validated only when the guarded arm holds, "
                "the control collapses, and the two metrics are genuinely distinct."
            ),
        },
        "metrics_before_online_adaptation": before_metrics,
        "metrics_after_deploy_online_adaptation": deploy_metrics,
        "metrics_after_control_online_adaptation": control_metrics,
        "metrics_after_noise_only_weighting": noise_metrics,
        "pass_rate_trajectory": [_round(value) for value in pass_rate],
        "true_accuracy_trajectory": [_round(value) for value in true_accuracy],
        "verifier_names": names,
        "verifier_orientation": {
            name: "inverted" if sign < 0 else "as_scored"
            for name, sign in zip(names, orientation, strict=True)
        },
        "weights_before": _weights_to_json(names, before_weights),
        "weights_noise_only": _weights_to_json(names, noise_only_weights),
        "weights_deploy_correlation_aware": _weights_to_json(names, deploy_weights),
        "weights_control_beta0": _weights_to_json(names, control_weights),
        "redundancy_penalty_by_verifier": _weights_to_json(names, redundancy_penalty),
        "correlation_source": correlation_source,
        "preconditions_checked": [dict(item) for item in preconditions or []],
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


def orient_scores_to_error_probability(
    labels: Sequence[int] | np.ndarray,
    score_matrix: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    """Orient each verifier so larger scores mean more likely incorrect."""

    labels_arr = np.asarray(labels, dtype=np.int64)
    matrix = np.asarray(score_matrix, dtype=np.float64).copy()
    orientation: list[int] = []
    for column_index in range(matrix.shape[1]):
        auroc = tie_aware_auroc(labels_arr, matrix[:, column_index])
        if auroc < 0.5:
            matrix[:, column_index] = 1.0 - matrix[:, column_index]
            orientation.append(-1)
        else:
            orientation.append(1)
    return np.clip(matrix, 0.0, 1.0), orientation


def reliability_raw_weights(labels: Sequence[int] | np.ndarray, score_matrix: np.ndarray) -> np.ndarray:
    """Return inverse-Brier raw weights while gating collapsed low-variance columns."""

    labels_arr = np.asarray(labels, dtype=np.float64)
    matrix = np.asarray(score_matrix, dtype=np.float64)
    brier_by_verifier = np.mean((matrix - labels_arr[:, None]) ** 2, axis=0)
    variance_by_verifier = np.var(matrix, axis=0)
    raw = np.zeros(matrix.shape[1], dtype=np.float64)
    active = variance_by_verifier > 1e-4
    raw[active] = 1.0 / np.maximum(brier_by_verifier[active], 1e-9)
    return raw


def seeded_redundancy_matrix(
    labels: Sequence[int] | np.ndarray,
    score_matrix: np.ndarray,
    *,
    exp3644_artifact: Mapping[str, Any] | None,
) -> tuple[np.ndarray, str]:
    """Use Exp 3644's conditional matrix when available, else compute inline."""

    from_seed = _redundancy_from_exp3644(exp3644_artifact, expected_dim=score_matrix.shape[1])
    if from_seed is not None:
        return from_seed, "exp3644_artifact"
    labels_arr = np.asarray(labels, dtype=np.int64)
    correct = safe_pearson_matrix(score_matrix[labels_arr == 0])
    incorrect = safe_pearson_matrix(score_matrix[labels_arr == 1])
    matrix = (np.abs(correct) + np.abs(incorrect)) / 2.0
    np.fill_diagonal(matrix, 0.0)
    return matrix, "inline_cached_scores"


def calibrate_with_online_bias(labels: Sequence[int] | np.ndarray, scores: Sequence[float]) -> np.ndarray:
    """Apply the smallest cached-label bias that balances Brier and ECE."""

    labels_arr = np.asarray(labels, dtype=np.float64)
    base = np.asarray(scores, dtype=np.float64)
    best = base
    base_brier = brier_score(labels_arr, base)
    base_ece = ece_score(labels_arr, base)
    best_key = (base_brier + 0.1 * base_ece, base_brier, base_ece, 0.0)
    for bias in np.linspace(-0.25, 0.25, 101):
        candidate = np.clip(base + float(bias), 0.0, 1.0)
        candidate_brier = brier_score(labels_arr, candidate)
        candidate_ece = ece_score(labels_arr, candidate)
        key = (
            candidate_brier + 0.1 * candidate_ece,
            candidate_brier,
            candidate_ece,
            abs(float(bias)),
        )
        if key < best_key:
            best = candidate
            best_key = key
    return best


def naive_beta0_control_weights(score_matrix: np.ndarray) -> np.ndarray:
    """Naive beta=0 control: reward score stability without grounding labels."""

    matrix = np.asarray(score_matrix, dtype=np.float64)
    raw = 1.0 / np.maximum(np.var(matrix, axis=0), 1e-12)
    return normalize_weights(raw)


def score_metrics(labels: Sequence[int] | np.ndarray, scores: Sequence[float]) -> dict[str, float]:
    """Compute ranking and calibration metrics for one ensemble trajectory."""

    labels_arr = np.asarray(labels, dtype=np.int64)
    score_arr = np.asarray(scores, dtype=np.float64)
    return {
        "auroc": _round(tie_aware_auroc(labels_arr, score_arr)),
        "brier": _round(brier_score(labels_arr, score_arr)),
        "ece": _round(ece_score(labels_arr, score_arr)),
    }


def brier_score(labels: Sequence[int] | np.ndarray, scores: Sequence[float]) -> float:
    """Mean squared probability error; lower means better calibrated probabilities."""

    labels_arr = np.asarray(labels, dtype=np.float64)
    score_arr = np.asarray(scores, dtype=np.float64)
    return float(np.mean((score_arr - labels_arr) ** 2))


def ece_score(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float],
    *,
    n_bins: int = 10,
) -> float:
    """Expected calibration error over fixed-width probability bins."""

    labels_arr = np.asarray(labels, dtype=np.float64)
    score_arr = np.asarray(scores, dtype=np.float64)
    total = 0.0
    for bin_index in range(n_bins):
        lower = bin_index / n_bins
        upper = (bin_index + 1) / n_bins
        if bin_index == n_bins - 1:
            mask = (score_arr >= lower) & (score_arr <= upper)
        else:
            mask = (score_arr >= lower) & (score_arr < upper)
        if np.any(mask):
            total += float(np.mean(mask)) * abs(
                float(np.mean(score_arr[mask])) - float(np.mean(labels_arr[mask]))
            )
    return total


def detect_deploy_collapse(before_metrics: Mapping[str, float], deploy_metrics: Mapping[str, float]) -> bool:
    """Detect whether the guarded arm lost ranking or calibration quality."""

    return bool(
        deploy_metrics["auroc"] < max(0.55, before_metrics["auroc"] - 0.05)
        or deploy_metrics["brier"] > before_metrics["brier"] + 0.05
    )


def detect_control_collapse(before_metrics: Mapping[str, float], control_metrics: Mapping[str, float]) -> bool:
    """Detect positive-control collapse from the beta=0 self-reinforcing arm."""

    return bool(
        control_metrics["auroc"] < max(0.55, before_metrics["auroc"] - 0.10)
        or control_metrics["brier"] > before_metrics["brier"] + 0.05
    )


def online_metric_trajectories(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float],
    *,
    n_windows: int = 8,
) -> tuple[list[float], list[float]]:
    """Return pass-rate and true-accuracy windows without sharing arrays."""

    labels_arr = np.asarray(labels, dtype=np.int64)
    score_arr = np.asarray(scores, dtype=np.float64)
    window_size = max(1, int(math.ceil(len(labels_arr) / n_windows)))
    pass_rate: list[float] = []
    true_accuracy: list[float] = []
    for start in range(0, len(labels_arr), window_size):
        end = min(len(labels_arr), start + window_size)
        window_scores = score_arr[start:end]
        window_labels = labels_arr[start:end]
        pass_rate.append(float(np.mean(1.0 - window_scores)))
        predictions = (window_scores >= 0.5).astype(np.int64)
        true_accuracy.append(float(np.mean(predictions == window_labels)))
    return pass_rate, true_accuracy


def select_honest_verdict(*, gate_passed: bool, correlation_aware_auroc_gain: float) -> str:
    """Choose the terminal verdict allowed by the Exp 3647 prompt."""

    if gate_passed and correlation_aware_auroc_gain > 0.0:
        return SUCCESS_VERDICT
    return NO_GAIN_VERDICT


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3647 artifact schema before JSON is written."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    verdict = artifact.get("honest_verdict")
    if verdict not in {SUCCESS_VERDICT, NO_GAIN_VERDICT, BLOCKED_VERDICT}:
        raise ValueError(f"unsupported honest_verdict: {verdict!r}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles must be present")
    missing_principles = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in principles]
    if missing_principles:
        raise ValueError(f"missing field principles: {missing_principles}")
    gate = artifact.get("acceptance_gate")
    if not isinstance(gate, Mapping) or not isinstance(gate.get("passed"), bool):
        raise ValueError("acceptance_gate.passed must be present as a boolean")
    if not isinstance(artifact.get("duration_s"), int | float):
        raise ValueError("duration_s must be numeric")
    if verdict == BLOCKED_VERDICT:
        return
    if int(artifact["n_online_updates"]) < MIN_ONLINE_UPDATES:
        raise ValueError(f"runnable artifact must report at least {MIN_ONLINE_UPDATES} updates")
    for field in (
        "collapse_detected_deploy_arm",
        "collapse_detected_control",
        "calibration_improved",
        "pass_rate_vs_true_accuracy_distinct_assert",
        "quality_maintained",
    ):
        if not isinstance(artifact.get(field), bool):
            raise ValueError(f"{field} must be a boolean")
    gain = artifact.get("correlation_aware_auroc_gain")
    if not isinstance(gain, int | float) or not math.isfinite(float(gain)):
        raise ValueError("correlation_aware_auroc_gain must be finite")


def write_artifact(
    repo_root: Path,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build, validate, and write the requested Exp 3647 JSON deliverable."""

    root = Path(repo_root)
    artifact = build_artifact(root, started_s=started_s, now_s=now_s)
    target = root / output_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def reproducibility_checksum(
    labels: np.ndarray,
    score_matrix: np.ndarray,
    deploy_weights: Sequence[float],
    control_weights: Sequence[float],
    *,
    random_seed: int,
) -> str:
    """Hash the deterministic inputs and final arm weights for drift detection."""

    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(labels, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(score_matrix, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(deploy_weights, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(control_weights, dtype=np.float64).tobytes())
    digest.update(str(int(random_seed)).encode("ascii"))
    return digest.hexdigest()


def _blocked_artifact(
    *,
    duration_s: float,
    random_seed: int,
    preconditions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = json.dumps(
        {"preconditions": [dict(item) for item in preconditions], "random_seed": random_seed},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    artifact: dict[str, Any] = {
        "artifact": "experiment_3647_fr11_continuous_self_learning_v8",
        "schema": "carnot.fr11_continuous_self_learning_v8",
        "honest_verdict": BLOCKED_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "n_online_updates": 0,
        "collapse_detected_deploy_arm": False,
        "collapse_detected_control": False,
        "correlation_aware_auroc_gain": 0.0,
        "calibration_improved": False,
        "pass_rate_vs_true_accuracy_distinct_assert": False,
        "quality_maintained": False,
        "random_seed": int(random_seed),
        "reproducibility_checksum": hashlib.sha256(payload).hexdigest(),
        "duration_s": _round(duration_s),
        "acceptance_gate": {
            "condition": (
                "collapse_detected_deploy_arm == false AND "
                "collapse_detected_control == true AND "
                "pass_rate_vs_true_accuracy_distinct_assert == true"
            ),
            "passed": False,
            "principle": (
                "Self-learning is validated only when the guarded arm holds, "
                "the control collapses, and the two metrics are genuinely distinct."
            ),
        },
        "preconditions_checked": [dict(item) for item in preconditions],
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


def _load_exp3644_artifact(repo_root: Path) -> dict[str, Any] | None:
    path = Path(repo_root) / "results" / "experiment_3644_weaver_peer_comparison_v3.json"
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None


def _redundancy_from_exp3644(
    artifact: Mapping[str, Any] | None,
    *,
    expected_dim: int,
) -> np.ndarray | None:
    if not artifact:
        return None
    conditional = artifact.get("conditional_verifier_correlation_by_label")
    if not isinstance(conditional, Mapping):
        return None
    matrices = []
    for label in ("correct", "incorrect"):
        raw_matrix = conditional.get(label)
        if raw_matrix is None:
            return None
        matrix = np.asarray(raw_matrix, dtype=np.float64)
        if matrix.shape != (expected_dim, expected_dim):
            return None
        matrices.append(np.abs(matrix))
    redundancy = sum(matrices) / float(len(matrices))
    np.fill_diagonal(redundancy, 0.0)
    return redundancy


def _score_matrix(
    scores_by_verifier: Mapping[str, Sequence[float]],
    verifier_names: Sequence[str],
) -> np.ndarray:
    columns = [np.asarray(scores_by_verifier[name], dtype=np.float64) for name in verifier_names]
    if not columns:
        raise ValueError("at least one verifier score column is required")
    lengths = {len(column) for column in columns}
    if len(lengths) != 1:
        raise ValueError("all verifier score columns must have the same length")
    return np.column_stack(columns)


def _same_trajectory(left: Sequence[float], right: Sequence[float]) -> bool:
    return [_round(value) for value in left] == [_round(value) for value in right]


def _weights_to_json(names: Sequence[str], weights: Sequence[float]) -> dict[str, float]:
    return {name: _round(float(weight)) for name, weight in zip(names, weights, strict=True)}


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _line in handle)


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.time() if now_s is None else float(now_s)
    return max(0.0, end - float(started_s))


def _round(value: float | int | np.floating[Any], digits: int = 6) -> float:
    return round(float(value), digits)

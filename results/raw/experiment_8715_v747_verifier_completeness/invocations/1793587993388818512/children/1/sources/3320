"""FR-11 continuous self-learning v11 with drift-aware dependency re-estimation.

Spec: REQ-LEARN-3685, SCENARIO-LEARN-3685.

The v11 forward difference over v10 is explicit verifier-vote distribution
drift detection.  Cached verifier traces are ordered into a deterministic
two-slice stream whose vote distribution shifts mid-run; the deploy arm watches
that stream, re-estimates the dependency graph after drift, and keeps v10's
conservative default, uncertainty gate, and collapse guard.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from carnot.fr11 import continuous_self_learning_v10 as v10


JsonDict = dict[str, Any]

OUTPUT_REL_PATH = Path("results/experiment_3685_fr11_continuous_self_learning_v11.json")
DEFAULT_RANDOM_SEED = 3685
DEFAULT_CORPUS_RANDOM_SEED = 3673
DEFAULT_N_ONLINE_UPDATES = 1000
MIN_ONLINE_UPDATES = v10.MIN_ONLINE_UPDATES
DRIFT_QUANTILE = 0.35
DRIFT_DISTANCE_THRESHOLD = 0.30
DRIFT_WINDOW = 50
UPDATE_PERIOD = v10.UPDATE_PERIOD
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates "
    "(principle: scores cached traces; no LLM load)."
)
SUCCESS_VERDICT = (
    "complete: "
    "fr11_v11_drift_aware_online_dependency_aware_recovers_no_collapse_quality_maintained"
)
NO_GAIN_VERDICT = "complete: fr11_v11_no_gain_over_v10_fixed_structure_online_sufficient"
BLOCKED_VERDICT = "complete: blocked_fr11_module_or_traces_unavailable"
TERMINAL_VERDICTS = (SUCCESS_VERDICT, NO_GAIN_VERDICT, BLOCKED_VERDICT)
VERIFIER_NAMES = v10.VERIFIER_NAMES

score_fover_corpus = v10.score_fover_corpus
score_matrix = v10.score_matrix
online_metric_trajectories = v10.online_metric_trajectories
detect_weight_collapse = v10.detect_weight_collapse

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "n_online_updates",
    "drift_detected_deploy_arm",
    "collapse_detected_deploy_arm",
    "collapse_detected_control",
    "post_drift_auroc_gain_over_v10",
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
    "drift_detected_deploy_arm": (
        "The deploy arm must detect the injected distribution drift "
        "(the v11 forward difference)."
    ),
    "collapse_detected_deploy_arm": (
        "The conservative-default + uncertainty-gated rule must prevent weight "
        "collapse (alpha_t grounding)."
    ),
    "collapse_detected_control": (
        "Positive control: the naive arm must collapse or fail to adapt, else the test has no contrast."
    ),
    "post_drift_auroc_gain_over_v10": (
        "The forward difference -- does drift-aware re-estimation beat v10's "
        "fixed-structure online weighting after drift?"
    ),
    "pass_rate_vs_true_accuracy_distinct_assert": (
        "De-flags the tautology where pass_rate and true_accuracy are the same array."
    ),
    "quality_maintained": (
        "Collapse-prevention + drift adaptation must not come at the cost of ensemble quality."
    ),
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection (of the artifact, vs the corpus drift it measures).",
    "duration_s": "Plausibility floor.",
}


@dataclass(frozen=True)
class DriftStream:
    """Cached verifier rows ordered so the vote distribution shifts mid-stream."""

    labels: np.ndarray
    score_matrix: np.ndarray
    drift_point: int
    projection: str
    drift_distance: float
    pre_vote_mean: np.ndarray
    post_vote_mean: np.ndarray


@dataclass(frozen=True)
class DriftAwareOnlineFit:
    """Final deploy-arm weights plus the drift events that caused re-estimation."""

    weights: np.ndarray
    edges: list[JsonDict]
    drift_events: list[JsonDict]
    n_weight_updates: int
    n_structure_reestimates: int
    catch_utility: np.ndarray


def build_artifact(
    repo_root: Path | str,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    n_examples: int = DEFAULT_N_ONLINE_UPDATES,
    random_seed: int = DEFAULT_RANDOM_SEED,
    corpus_random_seed: int = DEFAULT_CORPUS_RANDOM_SEED,
) -> JsonDict:
    """Build Exp 3685 from cached FoVer rows and FR-11 verifier state."""

    root = Path(repo_root)
    start = time.time() if started_s is None else float(started_s)
    preconditions = [
        _fr11_precondition(root),
        *probe_cached_trace_preconditions(root, n_examples=n_examples),
    ]
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
            random_seed=corpus_random_seed,
        )
    except Exception as exc:  # noqa: BLE001 - cached scoring failure is terminal.
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
    )


def probe_cached_trace_preconditions(repo_root: Path | str, *, n_examples: int) -> list[JsonDict]:
    """Check cached trace scoring and the v11 drift-aware online functions."""

    checks = v10.probe_cached_trace_preconditions(Path(repo_root), n_examples=n_examples)
    checks.append(
        {
            "resource": "drift_aware_online_reestimation_implementation",
            "available": all(
                callable(func)
                for func in (
                    build_drifting_trace_stream,
                    drift_aware_streaming_dependency_weights,
                    fixed_structure_v10_weights,
                )
            ),
            "detail": "drift stream builder, online re-estimator, and v10 fixed baseline importable",
        }
    )
    return [dict(item) for item in checks]


def build_artifact_from_scores(
    *,
    labels: Sequence[int],
    scores_by_verifier: Mapping[str, Sequence[float]],
    started_s: float,
    now_s: float | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    preconditions: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Evaluate drift-aware deploy, naive control, v10 fixed, and Carnot baselines."""

    if not labels or not scores_by_verifier:
        return _blocked_artifact(
            duration_s=_duration(float(started_s), now_s),
            random_seed=random_seed,
            preconditions=preconditions or [_trace_precondition([], {})],
        )

    names = list(scores_by_verifier)
    matrix = score_matrix(scores_by_verifier, names)
    labels_arr = np.asarray(labels, dtype=np.int64)
    if matrix.shape[0] != len(labels_arr):
        raise ValueError("labels and verifier scores must have the same length")
    if not _runnable(labels_arr):
        return _blocked_artifact(
            duration_s=_duration(float(started_s), now_s),
            random_seed=random_seed,
            preconditions=preconditions or [_trace_precondition(labels_arr, scores_by_verifier)],
        )

    try:
        stream = build_drifting_trace_stream(
            labels=labels_arr,
            score_matrix=matrix,
            verifier_names=names,
            random_seed=random_seed,
        )
    except ValueError as exc:
        blocked_preconditions = [
            *(preconditions or [_trace_precondition(labels_arr, scores_by_verifier)]),
            {"resource": "distributionally_distinct_drift_slices", "available": False, "detail": str(exc)},
        ]
        return _blocked_artifact(
            duration_s=_duration(float(started_s), now_s),
            random_seed=random_seed,
            preconditions=blocked_preconditions,
        )

    deploy = drift_aware_streaming_dependency_weights(
        labels=stream.labels,
        score_matrix=stream.score_matrix,
        verifier_names=names,
    )
    control = v10.naive_online_dependency_weights(
        labels=stream.labels,
        score_matrix=stream.score_matrix,
        verifier_names=names,
    )
    v10_fixed_weights = fixed_structure_v10_weights(
        labels=stream.labels,
        score_matrix=stream.score_matrix,
        verifier_names=names,
        drift_point=stream.drift_point,
    )
    carnot_weights = v10.exp3644.carnot_current_weights(names)

    arm_scores = {
        "deploy": _signed_scores(stream.score_matrix, deploy.weights),
        "control": _signed_scores(stream.score_matrix, control["weights"]),
        "v10_fixed_structure_online": _signed_scores(stream.score_matrix, v10_fixed_weights),
        "static_carnot": _signed_scores(stream.score_matrix, carnot_weights),
    }
    pre_metrics = _arm_metrics(
        stream.labels[: stream.drift_point],
        {name: scores[: stream.drift_point] for name, scores in arm_scores.items()},
    )
    post_metrics = _arm_metrics(
        stream.labels[stream.drift_point :],
        {name: scores[stream.drift_point :] for name, scores in arm_scores.items()},
    )
    all_metrics = _arm_metrics(stream.labels, arm_scores)
    post_drift_gain_over_v10 = (
        post_metrics["deploy"]["auroc"] - post_metrics["v10_fixed_structure_online"]["auroc"]
    )
    quality_maintained = bool(
        post_metrics["deploy"]["auroc"] >= post_metrics["static_carnot"]["auroc"] - 1e-12
        and all_metrics["deploy"]["auroc"] >= all_metrics["static_carnot"]["auroc"] - 1e-12
    )
    collapse_detected_deploy_arm = detect_weight_collapse(deploy.weights)
    collapse_detected_control = detect_weight_collapse(control["weights"])
    drift_detected_deploy_arm = bool(deploy.drift_events)
    pass_rate, true_accuracy = online_metric_trajectories(stream.labels, arm_scores["deploy"])
    distinct_assert = [_round(value) for value in pass_rate] != [
        _round(value) for value in true_accuracy
    ]
    gate_passed = bool(
        drift_detected_deploy_arm
        and not collapse_detected_deploy_arm
        and collapse_detected_control
        and distinct_assert
    )
    verdict = select_honest_verdict(
        gate_passed=gate_passed,
        quality_maintained=quality_maintained,
        post_drift_auroc_gain_over_v10=post_drift_gain_over_v10,
    )

    artifact: JsonDict = {
        "artifact": "experiment_3685_fr11_continuous_self_learning_v11",
        "schema": "carnot.fr11_continuous_self_learning_v11",
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "n_online_updates": int(len(stream.labels)),
        "drift_detected_deploy_arm": drift_detected_deploy_arm,
        "collapse_detected_deploy_arm": bool(collapse_detected_deploy_arm),
        "collapse_detected_control": bool(collapse_detected_control),
        "post_drift_auroc_gain_over_v10": _round(post_drift_gain_over_v10),
        "pass_rate_vs_true_accuracy_distinct_assert": bool(distinct_assert),
        "quality_maintained": bool(quality_maintained),
        "random_seed": int(random_seed),
        "reproducibility_checksum": reproducibility_checksum(
            stream.labels,
            stream.score_matrix,
            names,
            deploy.weights,
            control["weights"],
            deploy.drift_events,
            random_seed=random_seed,
        ),
        "duration_s": _round(_duration(float(started_s), now_s)),
        "acceptance_gate": {
            "condition": (
                "drift_detected_deploy_arm == true AND "
                "collapse_detected_deploy_arm == false AND "
                "collapse_detected_control == true AND "
                "pass_rate_vs_true_accuracy_distinct_assert == true"
            ),
            "passed": gate_passed,
            "principle": (
                "Drift-aware self-learning is validated only when the deploy "
                "arm detects drift, does not collapse, the control collapses, "
                "and the two metrics are genuinely distinct (not a tautology)."
            ),
        },
        "drift_stream": {
            "drift_point": int(stream.drift_point),
            "projection": stream.projection,
            "vote_distribution_distance": _round(stream.drift_distance),
            "pre_vote_mean": [_round(value) for value in stream.pre_vote_mean],
            "post_vote_mean": [_round(value) for value in stream.post_vote_mean],
            "verifier_names": names,
        },
        "metrics_pre_drift": pre_metrics,
        "metrics_post_drift": post_metrics,
        "metrics_all": all_metrics,
        "post_drift_auroc_gain_over_static_carnot": _round(
            post_metrics["deploy"]["auroc"] - post_metrics["static_carnot"]["auroc"]
        ),
        "post_drift_window_metrics": post_drift_window_metrics(
            labels=stream.labels,
            scores_by_arm=arm_scores,
            drift_point=stream.drift_point,
        ),
        "weights_deploy_initial": _weights_to_json(
            names,
            np.ones(len(names), dtype=np.float64) / float(len(names)),
        ),
        "weights_deploy_final": _weights_to_json(names, deploy.weights),
        "weights_control_final": _weights_to_json(names, control["weights"]),
        "weights_v10_fixed_structure_online": _weights_to_json(names, v10_fixed_weights),
        "weights_static_carnot": _weights_to_json(names, carnot_weights),
        "deploy_dependency_edges_final": deploy.edges,
        "deploy_drift_events": deploy.drift_events,
        "deploy_n_weight_updates": int(deploy.n_weight_updates),
        "deploy_n_structure_reestimates": int(deploy.n_structure_reestimates),
        "control_selected_verifier": control["selected_verifier"],
        "observed_catch_utility_by_verifier": _weights_to_json(names, deploy.catch_utility),
        "pass_rate_trajectory": [_round(value) for value in pass_rate],
        "true_accuracy_trajectory": [_round(value) for value in true_accuracy],
        "verifier_names": names,
        "preconditions_checked": [dict(item) for item in preconditions or []],
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


def build_drifting_trace_stream(
    *,
    labels: Sequence[int] | np.ndarray,
    score_matrix: np.ndarray,
    verifier_names: Sequence[str],
    random_seed: int,
) -> DriftStream:
    """Select two cached slices with distinct verifier-vote distributions."""

    labels_arr = np.asarray(labels, dtype=np.int64)
    matrix = np.asarray(score_matrix, dtype=np.float64)
    _validate_scores(labels_arr, matrix, verifier_names)
    rng = np.random.default_rng(int(random_seed))
    candidates: list[tuple[float, str, np.ndarray, np.ndarray]] = []
    projections = [("mean_score", np.mean(matrix, axis=1))]
    projections.extend((f"{name}_score", matrix[:, index]) for index, name in enumerate(verifier_names))
    projections.extend((f"negative_{name}_score", -matrix[:, index]) for index, name in enumerate(verifier_names))
    for projection_name, projection in projections:
        lower = projection <= float(np.quantile(projection, DRIFT_QUANTILE))
        upper = projection >= float(np.quantile(projection, 1.0 - DRIFT_QUANTILE))
        low_idx = np.where(lower)[0]
        high_idx = np.where(upper)[0]
        n_each = min(len(low_idx), len(high_idx))
        if n_each < MIN_ONLINE_UPDATES // 2:
            continue
        rng.shuffle(low_idx)
        rng.shuffle(high_idx)
        low_idx = low_idx[:n_each]
        high_idx = high_idx[:n_each]
        if not _slice_has_binary_support(labels_arr[low_idx]) or not _slice_has_binary_support(
            labels_arr[high_idx]
        ):
            continue
        distance = vote_distribution_distance(matrix[low_idx], matrix[high_idx])
        candidates.append((distance, projection_name, low_idx, high_idx))

    if not candidates:
        raise ValueError("no two binary cached slices can support a drift stream")
    distance, projection_name, pre_idx, post_idx = max(
        candidates,
        key=lambda item: (item[0], item[1]),
    )
    if distance < DRIFT_DISTANCE_THRESHOLD:
        raise ValueError(
            f"vote-distribution distance {distance:.6f} below required {DRIFT_DISTANCE_THRESHOLD:.6f}"
        )
    ordered = np.concatenate([pre_idx, post_idx])
    drift_point = int(len(pre_idx))
    return DriftStream(
        labels=labels_arr[ordered],
        score_matrix=matrix[ordered],
        drift_point=drift_point,
        projection=projection_name,
        drift_distance=float(distance),
        pre_vote_mean=vote_mean(matrix[pre_idx]),
        post_vote_mean=vote_mean(matrix[post_idx]),
    )


def drift_aware_streaming_dependency_weights(
    *,
    labels: Sequence[int] | np.ndarray,
    score_matrix: np.ndarray,
    verifier_names: Sequence[str],
    drift_window: int = DRIFT_WINDOW,
) -> DriftAwareOnlineFit:
    """Learn graph-aware weights online and reset the training window on drift."""

    labels_arr = np.asarray(labels, dtype=np.int64)
    matrix = np.asarray(score_matrix, dtype=np.float64)
    names = list(verifier_names)
    _validate_scores(labels_arr, matrix, names)
    weights = np.ones(matrix.shape[1], dtype=np.float64) / float(matrix.shape[1])
    edges: list[JsonDict] = []
    drift_events: list[JsonDict] = []
    catch_utility = np.zeros(matrix.shape[1], dtype=np.float64)
    n_weight_updates = 0
    train_start = 0
    reference_window: np.ndarray | None = None

    for end in _update_points(len(labels_arr)):
        if end >= drift_window and reference_window is None:
            reference_window = matrix[:drift_window]
        drift_distance = 0.0
        drift_now = False
        if reference_window is not None and end >= 2 * drift_window:
            current_window = matrix[end - drift_window : end]
            drift_distance = vote_distribution_distance(reference_window, current_window)
            drift_now = detect_vote_distribution_drift(reference_window, current_window)
            if drift_now and (not drift_events or end - int(drift_events[-1]["end_index"]) >= drift_window):
                train_start = max(0, end - drift_window)
                reference_window = current_window
                drift_events.append(
                    {
                        "end_index": int(end),
                        "reestimate_from_index": int(train_start),
                        "vote_distribution_distance": _round(drift_distance),
                    }
                )

        fit_start = train_start if drift_events else 0
        seen_labels = labels_arr[fit_start:end]
        seen_matrix = matrix[fit_start:end]
        catch_utility = v10.balanced_catch_utilities(seen_labels, seen_matrix)
        if not v10.uncertainty_gate_cleared(seen_labels, catch_utility):
            continue
        fit = v10.fit_dependency_aware_weights(
            labels=seen_labels,
            score_matrix=seen_matrix,
            verifier_names=names,
        )
        candidate = v10.collapse_guarded_weights(fit.weights)
        if not detect_weight_collapse(candidate):
            weights = candidate
            edges = [dict(edge) for edge in fit.edges]
            n_weight_updates += 1
            if drift_now:
                drift_events[-1]["structure_reestimated"] = True

    for event in drift_events:
        event.setdefault("structure_reestimated", False)
    return DriftAwareOnlineFit(
        weights=weights,
        edges=edges,
        drift_events=drift_events,
        n_weight_updates=n_weight_updates,
        n_structure_reestimates=sum(1 for event in drift_events if event["structure_reestimated"]),
        catch_utility=catch_utility,
    )


def fixed_structure_v10_weights(
    *,
    labels: Sequence[int] | np.ndarray,
    score_matrix: np.ndarray,
    verifier_names: Sequence[str],
    drift_point: int,
) -> np.ndarray:
    """Fit v10's dependency structure before drift and hold it fixed afterward."""

    labels_arr = np.asarray(labels, dtype=np.int64)
    matrix = np.asarray(score_matrix, dtype=np.float64)
    split = max(MIN_ONLINE_UPDATES // 2, min(int(drift_point), len(labels_arr)))
    pre_labels = labels_arr[:split]
    pre_matrix = matrix[:split]
    fit = v10.fit_dependency_aware_weights(
        labels=pre_labels,
        score_matrix=pre_matrix,
        verifier_names=verifier_names,
    )
    return v10.collapse_guarded_weights(fit.weights)


def vote_mean(matrix: np.ndarray) -> np.ndarray:
    """Return per-verifier binary vote rates for a score matrix."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("score matrix must be two-dimensional")
    return np.mean(values >= 0.5, axis=0)


def vote_distribution_distance(reference: np.ndarray, current: np.ndarray) -> float:
    """Measure verifier-vote distribution drift with an L2 distance."""

    return float(np.linalg.norm(vote_mean(current) - vote_mean(reference)))


def detect_vote_distribution_drift(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    threshold: float = DRIFT_DISTANCE_THRESHOLD,
) -> bool:
    """Return true when two vote windows differ enough to trigger re-estimation."""

    return bool(vote_distribution_distance(reference, current) >= float(threshold))


def post_drift_window_metrics(
    *,
    labels: Sequence[int] | np.ndarray,
    scores_by_arm: Mapping[str, Sequence[float] | np.ndarray],
    drift_point: int,
    n_windows: int = 4,
) -> list[JsonDict]:
    """Measure post-drift recovery windows for each arm."""

    labels_arr = np.asarray(labels, dtype=np.int64)
    window_size = max(1, int(math.ceil((len(labels_arr) - int(drift_point)) / n_windows)))
    windows: list[JsonDict] = []
    for start in range(int(drift_point), len(labels_arr), window_size):
        end = min(len(labels_arr), start + window_size)
        window_scores = {
            name: np.asarray(scores, dtype=np.float64)[start:end]
            for name, scores in scores_by_arm.items()
        }
        metrics = _arm_metrics(labels_arr[start:end], window_scores)
        windows.append(
            {
                "start_index": int(start),
                "end_index": int(end),
                "deploy_beats_v10_and_static": bool(
                    metrics["deploy"]["auroc"]
                    >= max(
                        metrics["v10_fixed_structure_online"]["auroc"],
                        metrics["static_carnot"]["auroc"],
                    )
                ),
                "metrics": metrics,
            }
        )
    return windows


def select_honest_verdict(
    *,
    gate_passed: bool,
    quality_maintained: bool,
    post_drift_auroc_gain_over_v10: float,
) -> str:
    """Choose the allowed Exp 3685 terminal verdict."""

    if gate_passed and quality_maintained and post_drift_auroc_gain_over_v10 > 0.0:
        return SUCCESS_VERDICT
    return NO_GAIN_VERDICT


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3685 artifact schema before writing JSON."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    verdict = artifact.get("honest_verdict")
    if verdict not in set(TERMINAL_VERDICTS):
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
        "drift_detected_deploy_arm",
        "collapse_detected_deploy_arm",
        "collapse_detected_control",
        "pass_rate_vs_true_accuracy_distinct_assert",
        "quality_maintained",
    ):
        if not isinstance(artifact.get(field), bool):
            raise ValueError(f"{field} must be a boolean")
    gain = artifact.get("post_drift_auroc_gain_over_v10")
    if not isinstance(gain, int | float) or not math.isfinite(float(gain)):
        raise ValueError("post_drift_auroc_gain_over_v10 must be finite")


def write_artifact(
    repo_root: Path | str,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    labels: Sequence[int] | None = None,
    scores_by_verifier: Mapping[str, Sequence[float]] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build, validate, and persist the Exp 3685 JSON artifact."""

    root = Path(repo_root)
    if labels is None or scores_by_verifier is None:
        artifact = build_artifact(root, started_s=started_s, now_s=now_s)
    else:
        artifact = build_artifact_from_scores(
            labels=labels,
            scores_by_verifier=scores_by_verifier,
            started_s=time.time() if started_s is None else float(started_s),
            now_s=now_s,
        )
    output = root / output_path
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def reproducibility_checksum(
    labels: np.ndarray,
    score_matrix: np.ndarray,
    verifier_names: Sequence[str],
    deploy_weights: Sequence[float],
    control_weights: Sequence[float],
    drift_events: Sequence[Mapping[str, Any]],
    *,
    random_seed: int,
) -> str:
    """Hash deterministic inputs, drift detections, and final arm weights."""

    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(labels, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(score_matrix, dtype=np.float64).tobytes())
    digest.update(json.dumps(list(verifier_names), separators=(",", ":")).encode("utf-8"))
    digest.update(np.ascontiguousarray(deploy_weights, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(control_weights, dtype=np.float64).tobytes())
    digest.update(json.dumps([dict(event) for event in drift_events], sort_keys=True).encode("utf-8"))
    digest.update(str(int(random_seed)).encode("ascii"))
    return digest.hexdigest()


def _blocked_artifact(
    *,
    duration_s: float,
    random_seed: int,
    preconditions: Sequence[Mapping[str, Any]],
) -> JsonDict:
    payload = json.dumps(
        {"preconditions": [dict(item) for item in preconditions], "random_seed": random_seed},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    artifact: JsonDict = {
        "artifact": "experiment_3685_fr11_continuous_self_learning_v11",
        "schema": "carnot.fr11_continuous_self_learning_v11",
        "honest_verdict": BLOCKED_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "n_online_updates": 0,
        "drift_detected_deploy_arm": False,
        "collapse_detected_deploy_arm": False,
        "collapse_detected_control": False,
        "post_drift_auroc_gain_over_v10": 0.0,
        "pass_rate_vs_true_accuracy_distinct_assert": False,
        "quality_maintained": False,
        "random_seed": int(random_seed),
        "reproducibility_checksum": hashlib.sha256(payload).hexdigest(),
        "duration_s": _round(duration_s),
        "acceptance_gate": {
            "condition": (
                "drift_detected_deploy_arm == true AND "
                "collapse_detected_deploy_arm == false AND "
                "collapse_detected_control == true AND "
                "pass_rate_vs_true_accuracy_distinct_assert == true"
            ),
            "passed": False,
            "principle": (
                "Drift-aware self-learning is validated only when the deploy "
                "arm detects drift, does not collapse, the control collapses, "
                "and the two metrics are genuinely distinct (not a tautology)."
            ),
        },
        "preconditions_checked": [dict(item) for item in preconditions],
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


def _fr11_precondition(root: Path) -> JsonDict:
    fr11_dir = root / "python/carnot/fr11"
    return {
        "resource": "fr11_module",
        "available": fr11_dir.is_dir(),
        "detail": str(fr11_dir),
    }


def _trace_precondition(
    labels: Sequence[int] | np.ndarray,
    scores_by_verifier: Mapping[str, Sequence[float]],
) -> JsonDict:
    return {
        "resource": "cached_traces_with_per_verifier_scores_and_labels",
        "available": _runnable(np.asarray(labels, dtype=np.int64)),
        "detail": (
            f"n_examples={len(labels)}; labels={sorted(set(int(value) for value in labels))}; "
            f"n_verifiers={len(scores_by_verifier)}; required>={MIN_ONLINE_UPDATES}"
        ),
    }


def _runnable(labels: np.ndarray) -> bool:
    return len(labels) >= MIN_ONLINE_UPDATES and len(set(int(value) for value in labels)) == 2


def _validate_scores(labels: np.ndarray, matrix: np.ndarray, verifier_names: Sequence[str]) -> None:
    if matrix.ndim != 2:
        raise ValueError("score_matrix must be two-dimensional")
    if matrix.shape[0] != len(labels):
        raise ValueError("labels and verifier scores must have the same length")
    if matrix.shape[1] != len(verifier_names):
        raise ValueError("score_matrix column count must match verifier_names")
    if not np.isfinite(matrix).all():
        raise ValueError("verifier score matrix must be finite")
    if len(set(int(value) for value in labels)) != 2:
        raise ValueError("labels must contain both binary classes 0 and 1")


def _slice_has_binary_support(labels: np.ndarray) -> bool:
    counts = [int(np.sum(labels == label)) for label in (0, 1)]
    return min(counts) >= v10.MIN_CLASS_EXAMPLES


def _update_points(n_rows: int) -> list[int]:
    points = list(range(UPDATE_PERIOD, n_rows + 1, UPDATE_PERIOD))
    if not points or points[-1] != n_rows:
        points.append(n_rows)
    return points


def _arm_metrics(
    labels: Sequence[int] | np.ndarray,
    scores_by_arm: Mapping[str, Sequence[float] | np.ndarray],
) -> dict[str, dict[str, float]]:
    return {
        arm: v10.score_metrics(labels, np.asarray(scores, dtype=np.float64))
        for arm, scores in scores_by_arm.items()
    }


def _signed_scores(matrix: np.ndarray, weights: Sequence[float]) -> np.ndarray:
    return np.clip(np.asarray(matrix, dtype=np.float64) @ np.asarray(weights, dtype=np.float64), 0.0, 1.0)


def _weights_to_json(names: Sequence[str], weights: Sequence[float]) -> dict[str, float]:
    return {name: _round(float(weight)) for name, weight in zip(names, weights, strict=True)}


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.time() if now_s is None else float(now_s)
    return max(0.0, end - float(started_s))


def _round(value: float | int | np.floating[Any], digits: int = 6) -> float:
    return round(float(value), digits)

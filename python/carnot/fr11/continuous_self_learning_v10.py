"""FR-11 continuous self-learning v10 with online dependency-aware weighting.

Spec: REQ-LEARN-3673, SCENARIO-LEARN-3673.

This experiment keeps the Exp 3660 closed-loop discipline, but the learned
object is now the verifier ensemble itself.  The deploy arm starts from equal
weights, waits for enough observed catch-rate evidence, then learns the
label-conditional dependency graph and signed graph-aware weights from cached
verifier scores.  The guard leaves the default in place while evidence is
uncertain and rejects any fitted vector that would collapse onto a single
verifier.  The control arm intentionally has no guard: it picks the current
single best catch-rate verifier, so collapse is visible as a positive control.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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

_EXP3644_PATH = Path(__file__).resolve().parents[1] / "verify" / "weaver_peer_comparison_v3.py"
_EXP3644_SPEC = importlib.util.spec_from_file_location(
    "carnot_fr11_v10_weaver_peer_comparison_v3",
    _EXP3644_PATH,
)
if _EXP3644_SPEC is None or _EXP3644_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"could not load Exp 3644 module at {_EXP3644_PATH}")
exp3644 = importlib.util.module_from_spec(_EXP3644_SPEC)
sys.modules[_EXP3644_SPEC.name] = exp3644
_EXP3644_SPEC.loader.exec_module(exp3644)


JsonDict = dict[str, Any]

OUTPUT_REL_PATH = Path("results/experiment_3673_fr11_continuous_self_learning_v10.json")
DEFAULT_RANDOM_SEED = 3673
DEFAULT_CORPUS_RANDOM_SEED = 3673
DEFAULT_N_ONLINE_UPDATES = 1000
DEFAULT_CROSSFIT_FOLDS = 5
MIN_ONLINE_UPDATES = 200
MIN_GATE_EXAMPLES = 40
MIN_CLASS_EXAMPLES = 8
UPDATE_PERIOD = 25
DEPLOY_MAX_ABS_WEIGHT = 0.8
COLLAPSE_BOUNDARY = 0.95
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates "
    "(principle: scores cached traces; no LLM load)."
)
SUCCESS_VERDICT = (
    "complete: "
    "fr11_v10_online_dependency_aware_weighting_holds_no_collapse_quality_maintained"
)
NO_GAIN_VERDICT = "complete: fr11_v10_online_no_gain_fixed_weighting_sufficient"
BLOCKED_VERDICT = "complete: blocked_fr11_module_or_traces_unavailable"
TERMINAL_VERDICTS = (SUCCESS_VERDICT, NO_GAIN_VERDICT, BLOCKED_VERDICT)
VERIFIER_NAMES = exp3644.VERIFIER_NAMES

score_fover_corpus = exp3644.score_fover_corpus

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "n_online_updates",
    "collapse_detected_deploy_arm",
    "collapse_detected_control",
    "online_dependency_aware_auroc_gain",
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
        "The conservative-default + uncertainty-gated rule must prevent weight "
        "collapse (alpha_t grounding)."
    ),
    "collapse_detected_control": (
        "Positive control: the naive online arm must collapse, else the test has no contrast."
    ),
    "online_dependency_aware_auroc_gain": (
        "The forward difference -- does online dependency-aware weighting beat "
        "fixed dependency-aware and static Carnot?"
    ),
    "pass_rate_vs_true_accuracy_distinct_assert": (
        "De-flags the tautology where pass_rate and true_accuracy are the same array."
    ),
    "quality_maintained": "Collapse-prevention must not come at the cost of ensemble quality.",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection.",
    "duration_s": "Plausibility floor.",
}


@dataclass(frozen=True)
class DependencyAwareFit:
    """Learned graph-aware signed verifier weights for one online prefix."""

    weights: np.ndarray
    edges: list[JsonDict]


@dataclass(frozen=True)
class CrossfitResult:
    """Fixed dependency-aware scores used as the offline comparison."""

    scores: np.ndarray
    mean_weights: np.ndarray


def build_artifact(
    repo_root: Path | str,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    n_examples: int = DEFAULT_N_ONLINE_UPDATES,
    random_seed: int = DEFAULT_RANDOM_SEED,
    corpus_random_seed: int = DEFAULT_CORPUS_RANDOM_SEED,
) -> JsonDict:
    """Build Exp 3673 from cached FoVer rows and FR-11 verifier state."""

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
    """Check that cached rows, verifier score paths, and dependency code exist."""

    root = Path(repo_root)
    checks = exp3644.probe_preconditions(root, n_examples=n_examples)
    corpus = checks[0] if checks else {"available": False, "detail": "missing"}
    rewritten = [
        {
            "resource": "cached_traces_with_per_verifier_scores_and_labels",
            "available": bool(corpus.get("available")),
            "detail": str(corpus.get("detail", "missing")),
        },
        *[dict(item) for item in checks[1:]],
    ]
    dependency_functions = (
        fit_dependency_aware_weights,
        learn_dependency_graph,
        dependency_aware_crossfit_scores,
    )
    rewritten.append(
        {
            "resource": "dependency_aware_weighting_implementation",
            "available": all(callable(func) for func in dependency_functions),
            "detail": "online graph fit, graph learner, and fixed crossfit baseline importable",
        }
    )
    return rewritten


def score_matrix(
    scores_by_verifier: Mapping[str, Sequence[float]],
    verifier_names: Sequence[str],
) -> np.ndarray:
    """Convert verifier score columns to a finite matrix."""

    names = list(verifier_names)
    if len(names) < 2:
        raise ValueError("at least two verifier score columns are required")
    columns = [np.asarray(scores_by_verifier[name], dtype=np.float64) for name in names]
    lengths = {len(column) for column in columns}
    if len(lengths) != 1:
        raise ValueError("all verifier score columns must have the same length")
    matrix = np.column_stack(columns)
    if not np.isfinite(matrix).all():
        raise ValueError("verifier score matrix must be finite")
    return matrix


def dependency_aware_crossfit_scores(
    *,
    labels: np.ndarray,
    score_matrix: np.ndarray,
    verifier_names: Sequence[str],
    random_seed: int,
    n_folds: int,
) -> CrossfitResult:
    """Score held-out folds with graph-aware weights learned on train folds."""

    labels_arr = np.asarray(labels, dtype=np.int64)
    matrix = np.asarray(score_matrix, dtype=np.float64)
    _require_binary_labels(labels_arr)
    min_class_count = min(int(np.sum(labels_arr == 0)), int(np.sum(labels_arr == 1)))
    folds = max(2, min(int(n_folds), min_class_count))
    fold_ids = stratified_fold_ids(labels_arr, folds=folds, random_seed=random_seed)
    out = np.zeros(len(labels_arr), dtype=np.float64)
    fold_weights: list[np.ndarray] = []
    for fold in range(folds):
        train_idx = np.where(fold_ids != fold)[0]
        test_idx = np.where(fold_ids == fold)[0]
        fit = fit_dependency_aware_weights(
            labels=labels_arr[train_idx],
            score_matrix=matrix[train_idx],
            verifier_names=verifier_names,
        )
        out[test_idx] = matrix[test_idx] @ fit.weights
        fold_weights.append(fit.weights)
    return CrossfitResult(
        scores=out,
        mean_weights=np.mean(np.asarray(fold_weights, dtype=np.float64), axis=0),
    )


def stratified_fold_ids(labels: np.ndarray, *, folds: int, random_seed: int) -> np.ndarray:
    """Return deterministic stratified fold ids for binary labels."""

    labels_arr = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(int(random_seed))
    fold_ids = np.empty(len(labels_arr), dtype=np.int64)
    for label in (0, 1):
        indices = np.where(labels_arr == label)[0]
        rng.shuffle(indices)
        for offset, index in enumerate(indices):
            fold_ids[index] = offset % int(folds)
    return fold_ids


def fit_dependency_aware_weights(
    *,
    labels: np.ndarray,
    score_matrix: np.ndarray,
    verifier_names: Sequence[str],
    ridge: float = 1e-4,
) -> DependencyAwareFit:
    """Learn a label-conditional graph and signed graph-aware weights."""

    matrix = np.asarray(score_matrix, dtype=np.float64)
    names = list(verifier_names)
    _validate_fit_inputs(labels, matrix, names)
    labels_arr = np.asarray(labels, dtype=np.int64)
    edges = learn_dependency_graph(labels=labels_arr, score_matrix=matrix, verifier_names=names)
    means: dict[int, np.ndarray] = {}
    covariances: dict[int, np.ndarray] = {}
    class_counts: dict[int, int] = {}
    for label in (0, 1):
        rows = matrix[labels_arr == label]
        class_counts[label] = int(len(rows))
        means[label] = np.mean(rows, axis=0)
        centered = rows - means[label]
        covariances[label] = np.atleast_2d(np.cov(centered, rowvar=False, ddof=0)).astype(
            np.float64
        )
    pooled = (
        covariances[0] * (class_counts[0] / len(labels_arr))
        + covariances[1] * (class_counts[1] / len(labels_arr))
    )
    mask = np.eye(matrix.shape[1], dtype=np.float64)
    for edge in edges:
        left = names.index(edge["pair"][0])
        right = names.index(edge["pair"][1])
        mask[left, right] = 1.0
        mask[right, left] = 1.0
    graph_covariance = pooled * mask + np.eye(matrix.shape[1], dtype=np.float64) * float(ridge)
    direction = means[1] - means[0]
    raw_weights = np.linalg.pinv(graph_covariance) @ direction
    return DependencyAwareFit(weights=normalize_signed_weights(raw_weights), edges=edges)


def learn_dependency_graph(
    *,
    labels: np.ndarray,
    score_matrix: np.ndarray,
    verifier_names: Sequence[str],
) -> list[JsonDict]:
    """Learn a Chow-Liu tree from label-conditional Gaussian mutual information."""

    labels_arr = np.asarray(labels, dtype=np.int64)
    matrix = np.asarray(score_matrix, dtype=np.float64)
    names = list(verifier_names)
    edge_scores: list[tuple[float, int, int, dict[str, float]]] = []
    for i in range(matrix.shape[1]):
        for j in range(i + 1, matrix.shape[1]):
            by_label: dict[str, float] = {}
            score = 0.0
            for label in (0, 1):
                rows = matrix[labels_arr == label]
                corr = exp3644.safe_pearson(rows[:, i], rows[:, j])
                corr = max(min(corr, 0.999), -0.999)
                by_label[str(label)] = _round(corr)
                score += (len(rows) / len(labels_arr)) * (
                    -0.5 * math.log(max(1e-12, 1.0 - corr * corr))
                )
            edge_scores.append((float(score), i, j, by_label))
    parent = list(range(matrix.shape[1]))
    tree: list[JsonDict] = []
    for score, i, j, by_label in sorted(edge_scores, key=lambda item: (-item[0], item[1], item[2])):
        root_i = _find_parent(parent, i)
        root_j = _find_parent(parent, j)
        if root_i == root_j:
            continue
        parent[root_i] = root_j
        tree.append(
            {
                "pair": [names[i], names[j]],
                "conditional_mutual_information": _round(score),
                "by_label_correlation": by_label,
            }
        )
        if len(tree) == matrix.shape[1] - 1:
            break
    return tree


def normalize_signed_weights(raw_weights: Sequence[float]) -> np.ndarray:
    """Normalize signed weights by L1 magnitude, preserving anti-signal direction."""

    raw = np.asarray(raw_weights, dtype=np.float64)
    raw = np.where(np.isfinite(raw), raw, 0.0)
    total = float(np.sum(np.abs(raw)))
    if total <= 1e-12:
        return np.ones(len(raw), dtype=np.float64) / float(len(raw))
    weights = raw / total
    if len(weights):
        weights[-1] += math.copysign(1.0 - float(np.sum(np.abs(weights))), weights[-1])
    return weights


def build_artifact_from_scores(
    *,
    labels: Sequence[int],
    scores_by_verifier: Mapping[str, Sequence[float]],
    started_s: float,
    now_s: float | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    preconditions: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Evaluate guarded deploy and naive control from cached verifier scores."""

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

    ordered_labels, ordered_matrix = _online_order(labels_arr, matrix, random_seed=random_seed)
    carnot_weights = exp3644.carnot_current_weights(names)
    carnot_scores = exp3644.ensemble_scores(ordered_matrix, carnot_weights)
    fixed_dependency = dependency_aware_crossfit_scores(
        labels=ordered_labels,
        score_matrix=ordered_matrix,
        verifier_names=names,
        random_seed=random_seed,
        n_folds=DEFAULT_CROSSFIT_FOLDS,
    )
    fixed_dependency_scores = _signed_scores(ordered_matrix, fixed_dependency.mean_weights)
    fixed_dependency_crossfit_scores = _clip_scores(fixed_dependency.scores)

    deploy = guarded_online_dependency_weights(
        labels=ordered_labels,
        score_matrix=ordered_matrix,
        verifier_names=names,
    )
    control = naive_online_dependency_weights(
        labels=ordered_labels,
        score_matrix=ordered_matrix,
        verifier_names=names,
    )
    deploy_raw_scores = _signed_scores(ordered_matrix, deploy["weights"])
    deploy_scores, deploy_calibration = calibrate_logistic_scores(ordered_labels, deploy_raw_scores)
    control_scores = _signed_scores(ordered_matrix, control["weights"])

    before_metrics = score_metrics(ordered_labels, carnot_scores)
    fixed_metrics = score_metrics(ordered_labels, fixed_dependency_crossfit_scores)
    fixed_final_metrics = score_metrics(ordered_labels, fixed_dependency_scores)
    deploy_metrics = score_metrics(ordered_labels, deploy_scores)
    control_metrics = score_metrics(ordered_labels, control_scores)
    collapse_detected_deploy_arm = detect_weight_collapse(deploy["weights"])
    collapse_detected_control = detect_weight_collapse(control["weights"])
    online_gain = deploy_metrics["auroc"] - max(before_metrics["auroc"], fixed_metrics["auroc"])
    quality_maintained = bool(
        not collapse_detected_deploy_arm
        and deploy_metrics["auroc"] >= before_metrics["auroc"] - 1e-12
        and deploy_metrics["brier"] <= before_metrics["brier"] + 1e-12
    )
    pass_rate, true_accuracy = online_metric_trajectories(ordered_labels, deploy_scores)
    distinct_assert = [_round(value) for value in pass_rate] != [
        _round(value) for value in true_accuracy
    ]
    gate_passed = bool(
        not collapse_detected_deploy_arm and collapse_detected_control and distinct_assert
    )
    verdict = select_honest_verdict(
        gate_passed=gate_passed,
        online_dependency_aware_auroc_gain=online_gain,
    )
    artifact: JsonDict = {
        "artifact": "experiment_3673_fr11_continuous_self_learning_v10",
        "schema": "carnot.fr11_continuous_self_learning_v10",
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "n_online_updates": int(len(ordered_labels)),
        "collapse_detected_deploy_arm": bool(collapse_detected_deploy_arm),
        "collapse_detected_control": bool(collapse_detected_control),
        "online_dependency_aware_auroc_gain": _round(online_gain),
        "pass_rate_vs_true_accuracy_distinct_assert": bool(distinct_assert),
        "quality_maintained": bool(quality_maintained),
        "random_seed": int(random_seed),
        "reproducibility_checksum": reproducibility_checksum(
            ordered_labels,
            ordered_matrix,
            names,
            deploy["weights"],
            control["weights"],
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
                "the control collapses, and the two metrics are genuinely distinct "
                "(not a tautology)."
            ),
        },
        "metrics_before_online_adaptation": before_metrics,
        "metrics_fixed_dependency_aware": fixed_metrics,
        "metrics_fixed_dependency_aware_final_fit": fixed_final_metrics,
        "metrics_after_deploy_online_adaptation": deploy_metrics,
        "metrics_after_control_online_adaptation": control_metrics,
        "weights_carnot_static": _weights_to_json(names, carnot_weights),
        "weights_fixed_dependency_aware_mean": _weights_to_json(
            names,
            fixed_dependency.mean_weights,
        ),
        "weights_deploy_initial": _weights_to_json(
            names,
            np.ones(len(names), dtype=np.float64) / float(len(names)),
        ),
        "weights_deploy_final": _weights_to_json(names, deploy["weights"]),
        "weights_control_final": _weights_to_json(names, control["weights"]),
        "deploy_weight_bound": {
            "max_abs_weight": DEPLOY_MAX_ABS_WEIGHT,
            "principle": "Signed verifier weights are rejected if one verifier dominates.",
        },
        "deploy_logistic_calibration": deploy_calibration,
        "deploy_dependency_edges_final": deploy["edges"],
        "deploy_n_weight_updates": int(deploy["n_weight_updates"]),
        "control_selected_verifier": control["selected_verifier"],
        "observed_catch_utility_by_verifier": _weights_to_json(
            names,
            deploy["catch_utility"],
        ),
        "pass_rate_trajectory": [_round(value) for value in pass_rate],
        "true_accuracy_trajectory": [_round(value) for value in true_accuracy],
        "verifier_names": names,
        "preconditions_checked": [dict(item) for item in preconditions or []],
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


def guarded_online_dependency_weights(
    *,
    labels: Sequence[int] | np.ndarray,
    score_matrix: np.ndarray,
    verifier_names: Sequence[str],
) -> JsonDict:
    """Learn dependency-aware weights only after catch-rate evidence clears."""

    labels_arr = np.asarray(labels, dtype=np.int64)
    matrix = np.asarray(score_matrix, dtype=np.float64)
    names = list(verifier_names)
    weights = np.ones(matrix.shape[1], dtype=np.float64) / float(matrix.shape[1])
    edges: list[JsonDict] = []
    n_weight_updates = 0
    catch_utility = np.zeros(matrix.shape[1], dtype=np.float64)
    for end in range(1, len(labels_arr) + 1):
        if end % UPDATE_PERIOD != 0 and end != len(labels_arr):
            continue
        seen_labels = labels_arr[:end]
        seen_matrix = matrix[:end]
        catch_utility = balanced_catch_utilities(seen_labels, seen_matrix)
        if not uncertainty_gate_cleared(seen_labels, catch_utility):
            continue
        fit = fit_dependency_aware_weights(
            labels=seen_labels,
            score_matrix=seen_matrix,
            verifier_names=names,
        )
        candidate = collapse_guarded_weights(fit.weights)
        if not detect_weight_collapse(candidate):
            weights = candidate
            edges = [dict(edge) for edge in fit.edges]
            n_weight_updates += 1
    return {
        "weights": weights,
        "edges": edges,
        "n_weight_updates": n_weight_updates,
        "catch_utility": catch_utility,
    }


def naive_online_dependency_weights(
    *,
    labels: Sequence[int] | np.ndarray,
    score_matrix: np.ndarray,
    verifier_names: Sequence[str],
) -> JsonDict:
    """Positive-control update: pick one verifier by catch utility."""

    labels_arr = np.asarray(labels, dtype=np.int64)
    matrix = np.asarray(score_matrix, dtype=np.float64)
    utilities = balanced_catch_utilities(labels_arr, matrix)
    selected = int(np.argmax(utilities)) if len(utilities) else 0
    weights = np.zeros(matrix.shape[1], dtype=np.float64)
    if len(weights):
        weights[selected] = 1.0
    return {
        "weights": weights,
        "selected_verifier": list(verifier_names)[selected] if len(weights) else "",
        "catch_utility": utilities,
    }


def balanced_catch_utilities(
    labels: Sequence[int] | np.ndarray,
    score_matrix: np.ndarray,
) -> np.ndarray:
    """Compute TPR-minus-FPR catch utility for every verifier column."""

    labels_arr = np.asarray(labels, dtype=np.int64)
    matrix = np.asarray(score_matrix, dtype=np.float64)
    utilities = np.zeros(matrix.shape[1], dtype=np.float64)
    positives = labels_arr == 1
    negatives = labels_arr == 0
    if not np.any(positives) or not np.any(negatives):
        return utilities
    for column_index in range(matrix.shape[1]):
        caught = matrix[:, column_index] >= 0.5
        utilities[column_index] = float(np.mean(caught[positives]) - np.mean(caught[negatives]))
    return utilities


def uncertainty_gate_cleared(labels: Sequence[int] | np.ndarray, catch_utility: np.ndarray) -> bool:
    """Return true when sample size and utility separation justify an update."""

    labels_arr = np.asarray(labels, dtype=np.int64)
    if len(labels_arr) < MIN_GATE_EXAMPLES:
        return False
    class_counts = [int(np.sum(labels_arr == label)) for label in (0, 1)]
    if min(class_counts) < MIN_CLASS_EXAMPLES:
        return False
    spread = float(np.max(catch_utility) - np.min(catch_utility)) if len(catch_utility) else 0.0
    uncertainty = max(0.03, 1.0 / math.sqrt(float(len(labels_arr))))
    return bool(spread >= uncertainty)


def collapse_guarded_weights(raw_weights: Sequence[float]) -> np.ndarray:
    """Shrink fitted signed weights toward uniform until the collapse guard holds."""

    target = normalize_signed_weights(raw_weights)
    if _within_deploy_bounds(target):
        return target
    default = np.ones(len(target), dtype=np.float64) / float(len(target))
    for alpha in np.linspace(0.95, 0.0, 20):
        candidate = normalize_signed_weights((1.0 - float(alpha)) * default + alpha * target)
        if _within_deploy_bounds(candidate):
            return candidate
    return default  # pragma: no cover - the alpha grid always reaches the uniform default.


def detect_weight_collapse(weights: Mapping[str, float] | Sequence[float]) -> bool:
    """Detect concentration on one verifier or a boundary absolute weight."""

    values = np.asarray(list(weights.values()) if isinstance(weights, Mapping) else weights, dtype=float)
    if values.size == 0:
        return False
    abs_values = np.abs(values)
    return bool(np.max(abs_values) >= COLLAPSE_BOUNDARY or int(np.sum(abs_values > 1e-9)) <= 1)


def score_metrics(labels: Sequence[int] | np.ndarray, scores: Sequence[float]) -> dict[str, float]:
    """Return ranking and calibration metrics for an error-probability score."""

    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    labels_arr = np.asarray(labels, dtype=np.int64)
    score_arr = _clip_scores(scores)
    return {
        "auroc": _round(exp3644.tie_aware_auroc(labels_arr, score_arr)),
        "brier": _round(float(np.mean((score_arr - labels_arr.astype(np.float64)) ** 2))),
        "ece": _round(expected_calibration_error(labels_arr, score_arr)),
    }


def calibrate_logistic_scores(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float],
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit a monotone logistic map so signed weights become calibrated probabilities."""

    labels_arr = np.asarray(labels, dtype=np.float64)
    raw = _clip_scores(scores)
    best_scores = raw
    best_scale = 1.0
    best_bias = 0.0
    best_key = _calibration_key(labels_arr, raw, scale=best_scale, bias=best_bias)
    for scale in np.linspace(1.0, 40.0, 79):
        for bias in np.linspace(-12.0, 6.0, 181):
            z = np.clip(float(scale) * raw + float(bias), -60.0, 60.0)
            candidate = 1.0 / (1.0 + np.exp(-z))
            key = _calibration_key(labels_arr, candidate, scale=float(scale), bias=float(bias))
            if key < best_key:
                best_key = key
                best_scores = candidate
                best_scale = float(scale)
                best_bias = float(bias)
    return best_scores, {"scale": _round(best_scale), "bias": _round(best_bias)}


def expected_calibration_error(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Measure bin-wise probability calibration error for cached labels."""

    labels_arr = np.asarray(labels, dtype=np.float64)
    score_arr = _clip_scores(scores)
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


def online_metric_trajectories(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float],
    *,
    n_windows: int = 8,
) -> tuple[list[float], list[float]]:
    """Return pass-rate and true-accuracy windows without sharing arrays."""

    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    labels_arr = np.asarray(labels, dtype=np.int64)
    score_arr = _clip_scores(scores)
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


def select_honest_verdict(*, gate_passed: bool, online_dependency_aware_auroc_gain: float) -> str:
    """Choose the allowed Exp 3673 terminal verdict."""

    if gate_passed and online_dependency_aware_auroc_gain > 0.0:
        return SUCCESS_VERDICT
    return NO_GAIN_VERDICT


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3673 artifact schema before writing JSON."""

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
        "collapse_detected_deploy_arm",
        "collapse_detected_control",
        "pass_rate_vs_true_accuracy_distinct_assert",
        "quality_maintained",
    ):
        if not isinstance(artifact.get(field), bool):
            raise ValueError(f"{field} must be a boolean")
    gain = artifact.get("online_dependency_aware_auroc_gain")
    if not isinstance(gain, int | float) or not math.isfinite(float(gain)):
        raise ValueError("online_dependency_aware_auroc_gain must be finite")


def write_artifact(
    repo_root: Path | str,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    labels: Sequence[int] | None = None,
    scores_by_verifier: Mapping[str, Sequence[float]] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build, validate, and persist the Exp 3673 JSON artifact."""

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
    *,
    random_seed: int,
) -> str:
    """Hash deterministic inputs and final arm weights for drift detection."""

    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(labels, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(score_matrix, dtype=np.float64).tobytes())
    digest.update(json.dumps(list(verifier_names), separators=(",", ":")).encode("utf-8"))
    digest.update(np.ascontiguousarray(deploy_weights, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(control_weights, dtype=np.float64).tobytes())
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
        "artifact": "experiment_3673_fr11_continuous_self_learning_v10",
        "schema": "carnot.fr11_continuous_self_learning_v10",
        "honest_verdict": BLOCKED_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "n_online_updates": 0,
        "collapse_detected_deploy_arm": False,
        "collapse_detected_control": False,
        "online_dependency_aware_auroc_gain": 0.0,
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
                "the control collapses, and the two metrics are genuinely distinct "
                "(not a tautology)."
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


def _validate_fit_inputs(labels: np.ndarray, matrix: np.ndarray, names: Sequence[str]) -> None:
    if matrix.ndim != 2:
        raise ValueError("score_matrix must be two-dimensional")
    if matrix.shape[1] != len(names):
        raise ValueError("score_matrix column count must match verifier_names")
    if matrix.shape[1] < 2:
        raise ValueError("at least two verifier score columns are required")
    labels_arr = np.asarray(labels, dtype=np.int64)
    if matrix.shape[0] != len(labels_arr):
        raise ValueError("labels and verifier scores must have the same length")
    _require_binary_labels(labels_arr)


def _require_binary_labels(labels: np.ndarray) -> None:
    if set(int(value) for value in labels) != {0, 1}:
        raise ValueError("labels must contain both binary classes 0 and 1")


def _find_parent(parent: list[int], index: int) -> int:
    while parent[index] != index:
        parent[index] = parent[parent[index]]
        index = parent[index]
    return index


def _online_order(
    labels: np.ndarray,
    score_matrix: np.ndarray,
    *,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(random_seed))
    order = rng.permutation(len(labels))
    return labels[order], score_matrix[order]


def _signed_scores(score_matrix: np.ndarray, weights: Sequence[float]) -> np.ndarray:
    return _clip_scores(np.asarray(score_matrix, dtype=np.float64) @ np.asarray(weights, dtype=np.float64))


def _calibration_key(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    scale: float,
    bias: float,
) -> tuple[float, float, float, float]:
    brier = float(np.mean((scores - labels) ** 2))
    ece = expected_calibration_error(labels, scores)
    return (brier + 0.1 * ece, brier, ece, abs(float(scale) - 1.0) + abs(float(bias)))


def _clip_scores(scores: Sequence[float] | np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)


def _within_deploy_bounds(weights: Sequence[float]) -> bool:
    values = np.asarray(weights, dtype=np.float64)
    return not detect_weight_collapse(values) and bool(np.max(np.abs(values)) <= DEPLOY_MAX_ABS_WEIGHT)


def _weights_to_json(names: Sequence[str], weights: Sequence[float]) -> dict[str, float]:
    return {name: _round(float(weight)) for name, weight in zip(names, weights, strict=True)}


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.time() if now_s is None else float(now_s)
    return max(0.0, end - float(started_s))


def _round(value: float | int | np.floating[Any], digits: int = 6) -> float:
    return round(float(value), digits)

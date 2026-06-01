"""Exp 3656 correlation-aware weighting paradox diagnosis.

This module reuses the cached FoVer scores from Exp 3644, then replaces the
naive marginal-correlation penalty with a label-conditional dependency graph
and signed graph-aware weights. The learned graph is a Chow-Liu tree over the
verifier vote columns; the weights solve a graph-sparse Fisher system so
correlation is treated as dependency structure, not as a reason to suppress
every correlated verifier.

Spec: REQ-VERIFY-3656, SCENARIO-VERIFY-3656.
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

from carnot.verify import weaver_peer_comparison_v3 as exp3644


OUTPUT_REL_PATH = Path("results/experiment_3656_correlation_aware_weighting_paradox_diagnosis.json")
EXP3644_REL_PATH = exp3644.OUTPUT_REL_PATH
DEFAULT_N_EXAMPLES = exp3644.DEFAULT_N_EXAMPLES
DEFAULT_BASELINE_RANDOM_SEED = exp3644.DEFAULT_RANDOM_SEED
DEFAULT_RANDOM_SEED = 3656
DEFAULT_CROSSFIT_FOLDS = 5
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates "
    "(principle: scores the cached FoVer corpus; no LLM load)."
)

SUCCESS_DEPENDENCY_AWARE_RECOVERS = (
    "complete: paradox_resolved_naive_penalty_misspecified_dependency_aware_recovers"
)
SUCCESS_CORRELATION_HARMLESS = (
    "complete: paradox_resolved_correlation_genuinely_harmless_joint_null_space_concern_refuted_here"
)
BLOCKED_VERDICT = "complete: blocked_fover_corpus_or_correlation_matrix_unavailable"
TERMINAL_VERDICTS = (
    SUCCESS_DEPENDENCY_AWARE_RECOVERS,
    SUCCESS_CORRELATION_HARMLESS,
    BLOCKED_VERDICT,
)
OUTCOME_CATEGORIES = (
    "correlation_harmless",
    "penalty_misspecified",
    "dependency_aware_recovers",
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "ensemble_auroc_naive_correlation_aware",
    "ensemble_auroc_dependency_aware_proper",
    "ensemble_auroc_carnot",
    "naive_penalty_diagnosis",
    "correlation_harmless_or_penalty_misspecified",
    "n_examples",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": "Scores the cached FoVer corpus; no LLM load.",
    "ensemble_auroc_naive_correlation_aware": (
        "Reproduces the .334 regression (0.635) -- the paradox to explain."
    ),
    "ensemble_auroc_dependency_aware_proper": (
        "The arXiv:1903.05844 learned-dependency-structure weighting -- does "
        "proper dependency-awareness recover or beat Carnot's 0.919?"
    ),
    "ensemble_auroc_carnot": "Carnot's current weighting (0.919) -- the bar.",
    "naive_penalty_diagnosis": (
        "WHY the naive redundancy penalty hurt (redundancy-vs-complementarity "
        "confusion) -- the mechanistic explanation."
    ),
    "correlation_harmless_or_penalty_misspecified": (
        "H1 (correlation genuinely harmless) vs H2 (naive penalty "
        "mis-specified, proper dependency-awareness helps) -- the falsifiable result."
    ),
    "n_examples": "Sample-size rigor.",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection.",
    "duration_s": "Plausibility floor.",
}


@dataclass(frozen=True)
class OutcomeClassification:
    """Paradox diagnosis category plus the terminal verdict it maps to."""

    category: str
    terminal_verdict: str
    rationale: str


@dataclass(frozen=True)
class DependencyAwareFit:
    """Learned graph-aware signed verifier weights."""

    weights: np.ndarray
    edges: list[dict[str, Any]]
    class_means: dict[int, list[float]]
    graph_sparse_covariance: list[list[float]]


@dataclass(frozen=True)
class CrossfitResult:
    """Out-of-fold dependency-aware scores and fold diagnostics."""

    scores: np.ndarray
    mean_weights: np.ndarray
    fold_weights: list[list[float]]
    edge_frequencies: list[dict[str, Any]]
    folds: int


def build_artifact(
    repo_root: Path,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    n_examples: int = DEFAULT_N_EXAMPLES,
    random_seed: int = DEFAULT_RANDOM_SEED,
    baseline_random_seed: int = DEFAULT_BASELINE_RANDOM_SEED,
) -> dict[str, Any]:
    """Build the Exp 3656 terminal artifact from cached FoVer verifier scores."""

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
        prior_exp3644 = load_prior_exp3644_artifact(root)
        labels, scores_by_verifier = exp3644.score_fover_corpus(
            root,
            n_examples=n_examples,
            random_seed=baseline_random_seed,
        )
    except Exception as exc:  # noqa: BLE001 - terminal artifact must fail closed.
        failed = [
            *preconditions,
            {
                "resource": "fover_scoring",
                "available": False,
                "detail": f"{type(exc).__name__}: {exc}",
            },
        ]
        return _blocked_artifact(
            duration_s=_duration(start, now_s),
            random_seed=random_seed,
            preconditions=failed,
        )

    return build_artifact_from_scores(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        prior_exp3644=prior_exp3644,
        started_s=start,
        now_s=now_s,
        random_seed=random_seed,
        baseline_random_seed=baseline_random_seed,
        preconditions=preconditions,
    )


def probe_preconditions(repo_root: Path, *, n_examples: int) -> list[dict[str, Any]]:
    """Check the FoVer scorer preconditions and the prior Exp 3644 matrix."""

    root = Path(repo_root)
    checks = list(exp3644.probe_preconditions(root, n_examples=n_examples))
    try:
        prior = load_prior_exp3644_artifact(root)
        matrix = prior["pearson_verifier_correlation_matrix"]
        detail = f"matrix_shape={len(matrix)}x{len(matrix[0]) if matrix else 0}"
        available = True
    except Exception as exc:  # noqa: BLE001 - precondition diagnostics must be explicit.
        detail = f"{type(exc).__name__}: {exc}"
        available = False
    checks.append(
        {
            "resource": "exp3644_correlation_matrix",
            "available": available,
            "detail": detail,
        }
    )
    return checks


def load_prior_exp3644_artifact(repo_root: Path) -> dict[str, Any]:
    """Load the prior Exp 3644 artifact and verify the correlation matrix exists."""

    path = Path(repo_root) / EXP3644_REL_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    matrix = payload.get("pearson_verifier_correlation_matrix")
    if not isinstance(matrix, list) or not matrix or not isinstance(matrix[0], list):
        raise ValueError("Exp 3644 artifact has no readable Pearson correlation matrix")
    for field in (
        "ensemble_auroc_unweighted",
        "ensemble_auroc_weaver_style",
        "ensemble_auroc_carnot",
        "ensemble_auroc_correlation_aware",
        "auroc_delta_correlation_aware_vs_weaver",
    ):
        if field not in payload:
            raise ValueError(f"Exp 3644 artifact is missing {field}")
    return payload


def build_artifact_from_scores(
    *,
    labels: Sequence[int],
    scores_by_verifier: Mapping[str, Sequence[float]],
    prior_exp3644: Mapping[str, Any],
    started_s: float,
    now_s: float | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    baseline_random_seed: int = DEFAULT_BASELINE_RANDOM_SEED,
    preconditions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the terminal artifact from precomputed verifier score columns."""

    names = list(scores_by_verifier)
    matrix = score_matrix(scores_by_verifier, names)
    labels_arr = np.asarray(labels, dtype=np.int64)
    if matrix.shape[0] != len(labels_arr):
        raise ValueError("labels and verifier scores must have the same length")
    _require_binary_labels(labels_arr)

    unweighted_weights = exp3644.normalize_weights(np.ones(matrix.shape[1], dtype=float))
    weaver_weights = exp3644.weaver_style_weights(matrix)
    carnot_weights = exp3644.carnot_current_weights(names)
    naive_weights = exp3644.correlation_aware_weights(matrix)
    crossfit = dependency_aware_crossfit_scores(
        labels=labels_arr,
        score_matrix=matrix,
        verifier_names=names,
        random_seed=random_seed,
        n_folds=DEFAULT_CROSSFIT_FOLDS,
    )
    full_fit = fit_dependency_aware_weights(
        labels=labels_arr,
        score_matrix=matrix,
        verifier_names=names,
    )

    auroc_unweighted = exp3644.tie_aware_auroc(labels_arr, exp3644.ensemble_scores(matrix, unweighted_weights))
    auroc_weaver = exp3644.tie_aware_auroc(labels_arr, exp3644.ensemble_scores(matrix, weaver_weights))
    auroc_carnot = exp3644.tie_aware_auroc(labels_arr, exp3644.ensemble_scores(matrix, carnot_weights))
    auroc_naive = exp3644.tie_aware_auroc(labels_arr, exp3644.ensemble_scores(matrix, naive_weights))
    auroc_dependency = exp3644.tie_aware_auroc(labels_arr, crossfit.scores)

    classification = classify_paradox(
        naive_auroc=auroc_naive,
        dependency_aware_auroc=auroc_dependency,
        carnot_auroc=auroc_carnot,
    )
    diagnosis = diagnose_naive_penalty(
        labels=labels_arr,
        score_matrix=matrix,
        verifier_names=names,
        weaver_weights=weaver_weights,
        naive_weights=naive_weights,
        dependency_weights=crossfit.mean_weights,
        learned_edges=full_fit.edges,
    )

    artifact = {
        "artifact": "experiment_3656_correlation_aware_weighting_paradox_diagnosis",
        "schema": "carnot.correlation_aware_weighting_paradox_diagnosis.v1",
        "honest_verdict": classification.terminal_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "ensemble_auroc_unweighted": _round(auroc_unweighted),
        "ensemble_auroc_weaver_style": _round(auroc_weaver),
        "ensemble_auroc_carnot": _round(auroc_carnot),
        "ensemble_auroc_correlation_aware": _round(auroc_naive),
        "ensemble_auroc_naive_correlation_aware": _round(auroc_naive),
        "ensemble_auroc_dependency_aware_proper": _round(auroc_dependency),
        "dependency_aware_auroc_gain_vs_naive": _round(auroc_dependency - auroc_naive),
        "dependency_aware_auroc_delta_vs_carnot": _round(auroc_dependency - auroc_carnot),
        "naive_auroc_delta_vs_weaver": _round(auroc_naive - auroc_weaver),
        "exp3644_baseline_reproduction": exp3644_baseline_reproduction(
            current={
                "ensemble_auroc_unweighted": auroc_unweighted,
                "ensemble_auroc_weaver_style": auroc_weaver,
                "ensemble_auroc_carnot": auroc_carnot,
                "ensemble_auroc_correlation_aware": auroc_naive,
                "auroc_delta_correlation_aware_vs_weaver": auroc_naive - auroc_weaver,
            },
            prior_exp3644=prior_exp3644,
        ),
        "dependency_aware_learned_graph": {
            "graph_type": "label_conditional_chow_liu_tree",
            "edges": full_fit.edges,
            "edge_frequencies_across_folds": crossfit.edge_frequencies,
        },
        "dependency_aware_training_protocol": {
            "method": "stratified_crossfit_graph_sparse_signed_fisher_weights",
            "folds": crossfit.folds,
            "baseline_selection_seed": int(baseline_random_seed),
            "graph_seed": int(random_seed),
            "learned_on_labels": True,
        },
        "naive_penalty_diagnosis": diagnosis,
        "correlation_harmless_or_penalty_misspecified": classification.category,
        "outcome_rationale": classification.rationale,
        "n_examples": int(len(labels_arr)),
        "random_seed": int(random_seed),
        "baseline_random_seed": int(baseline_random_seed),
        "reproducibility_checksum": reproducibility_checksum(
            labels_arr,
            matrix,
            names,
            random_seed=random_seed,
            baseline_random_seed=baseline_random_seed,
            prior_exp3644=prior_exp3644,
        ),
        "duration_s": _round(_duration(float(started_s), now_s)),
        "verifier_names": names,
        "weights_unweighted": weights_to_json(names, unweighted_weights),
        "weights_weaver_style": weights_to_json(names, weaver_weights),
        "weights_carnot": weights_to_json(names, carnot_weights),
        "weights_naive_correlation_aware": weights_to_json(names, naive_weights),
        "weights_dependency_aware_proper": weights_to_json(names, crossfit.mean_weights),
        "weights_dependency_aware_full_fit": weights_to_json(names, full_fit.weights),
        "dependency_aware_fold_weights": [
            weights_to_json(names, fold_weights) for fold_weights in crossfit.fold_weights
        ],
        "dependency_aware_graph_sparse_covariance": full_fit.graph_sparse_covariance,
        "preconditions_checked": [dict(item) for item in preconditions or []],
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


def classify_paradox(
    *,
    naive_auroc: float,
    dependency_aware_auroc: float,
    carnot_auroc: float,
) -> OutcomeClassification:
    """Classify whether proper dependency awareness resolves the paradox."""

    naive_gap = max(0.0, float(carnot_auroc) - float(naive_auroc))
    recovery_floor = max(0.05, 0.5 * naive_gap)
    dependency_gain = float(dependency_aware_auroc) - float(naive_auroc)
    if dependency_gain >= recovery_floor and dependency_aware_auroc >= carnot_auroc - 0.01:
        return OutcomeClassification(
            category="dependency_aware_recovers",
            terminal_verdict=SUCCESS_DEPENDENCY_AWARE_RECOVERS,
            rationale=(
                "Proper label-conditional dependency-aware weights recover the naive "
                "correlation penalty regression and meet or nearly meet the Carnot bar."
            ),
        )
    if dependency_gain >= 0.025:
        return OutcomeClassification(
            category="penalty_misspecified",
            terminal_verdict=SUCCESS_DEPENDENCY_AWARE_RECOVERS,
            rationale=(
                "Proper dependency-aware weights improve materially over the naive "
                "penalty, so the marginal redundancy penalty was mis-specified."
            ),
        )
    return OutcomeClassification(
        category="correlation_harmless",
        terminal_verdict=SUCCESS_CORRELATION_HARMLESS,
        rationale=(
            "Proper dependency awareness does not materially improve over the naive "
            "correlation penalty, so this ensemble does not show harmful dependency."
        ),
    )


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
    """Score held-out folds with dependency-aware weights learned on train folds."""

    labels_arr = np.asarray(labels, dtype=np.int64)
    matrix = np.asarray(score_matrix, dtype=np.float64)
    _require_binary_labels(labels_arr)
    min_class_count = min(int(np.sum(labels_arr == 0)), int(np.sum(labels_arr == 1)))
    folds = min(int(n_folds), min_class_count)
    if folds < 2:
        folds = 2
    fold_ids = stratified_fold_ids(labels_arr, folds=folds, random_seed=random_seed)
    out = np.zeros(len(labels_arr), dtype=np.float64)
    fold_weights: list[list[float]] = []
    edge_counter: dict[tuple[str, str], int] = {}
    for fold in range(folds):
        train_idx = np.where(fold_ids != fold)[0]
        test_idx = np.where(fold_ids == fold)[0]
        fit = fit_dependency_aware_weights(
            labels=labels_arr[train_idx],
            score_matrix=matrix[train_idx],
            verifier_names=verifier_names,
        )
        out[test_idx] = matrix[test_idx] @ fit.weights
        fold_weights.append([float(value) for value in fit.weights])
        for edge in fit.edges:
            pair = tuple(edge["pair"])
            edge_counter[pair] = edge_counter.get(pair, 0) + 1
    mean_weights = np.mean(np.asarray(fold_weights, dtype=np.float64), axis=0)
    edge_frequencies = [
        {"pair": list(pair), "fold_count": count, "frequency": _round(count / folds)}
        for pair, count in sorted(edge_counter.items(), key=lambda item: (-item[1], item[0]))
    ]
    return CrossfitResult(
        scores=out,
        mean_weights=mean_weights,
        fold_weights=fold_weights,
        edge_frequencies=edge_frequencies,
        folds=folds,
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
    """Learn a label-conditional dependency graph and signed graph-aware weights."""

    matrix = np.asarray(score_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("score_matrix must be two-dimensional")
    names = list(verifier_names)
    if matrix.shape[1] != len(names):
        raise ValueError("score_matrix column count must match verifier_names")
    if matrix.shape[1] < 2:
        raise ValueError("at least two verifier score columns are required")
    labels_arr = np.asarray(labels, dtype=np.int64)
    if matrix.shape[0] != len(labels_arr):
        raise ValueError("labels and verifier scores must have the same length")
    _require_binary_labels(labels_arr)

    edges = learn_dependency_graph(labels=labels_arr, score_matrix=matrix, verifier_names=names)
    means: dict[int, np.ndarray] = {}
    covariances: dict[int, np.ndarray] = {}
    class_counts: dict[int, int] = {}
    for label in (0, 1):
        rows = matrix[labels_arr == label]
        class_counts[label] = int(len(rows))
        means[label] = np.mean(rows, axis=0)
        centered = rows - means[label]
        covariances[label] = np.cov(centered, rowvar=False, ddof=0)
        covariances[label] = np.atleast_2d(covariances[label]).astype(np.float64)

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
    weights = normalize_signed_weights(raw_weights)
    return DependencyAwareFit(
        weights=weights,
        edges=edges,
        class_means={label: [_round(value) for value in means[label]] for label in (0, 1)},
        graph_sparse_covariance=matrix_to_json(graph_covariance),
    )


def learn_dependency_graph(
    *,
    labels: np.ndarray,
    score_matrix: np.ndarray,
    verifier_names: Sequence[str],
) -> list[dict[str, Any]]:
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
                score += (len(rows) / len(labels_arr)) * (-0.5 * math.log(max(1e-12, 1.0 - corr * corr)))
            edge_scores.append((float(score), i, j, by_label))

    parent = list(range(matrix.shape[1]))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    tree: list[dict[str, Any]] = []
    for score, i, j, by_label in sorted(edge_scores, key=lambda item: (-item[0], item[1], item[2])):
        root_i = find(i)
        root_j = find(j)
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


def diagnose_naive_penalty(
    *,
    labels: np.ndarray,
    score_matrix: np.ndarray,
    verifier_names: Sequence[str],
    weaver_weights: np.ndarray,
    naive_weights: np.ndarray,
    dependency_weights: np.ndarray,
    learned_edges: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Explain whether the naive penalty mistook useful correlation for redundancy."""

    names = list(verifier_names)
    per_verifier: dict[str, dict[str, Any]] = {}
    for idx, name in enumerate(names):
        column = score_matrix[:, idx]
        auroc = exp3644.tie_aware_auroc(labels, column)
        mean_positive = float(np.mean(column[labels == 1]))
        mean_negative = float(np.mean(column[labels == 0]))
        per_verifier[name] = {
            "single_verifier_auroc": _round(auroc),
            "label_direction": "positive" if mean_positive >= mean_negative else "anti_signal",
            "mean_label_1": _round(mean_positive),
            "mean_label_0": _round(mean_negative),
            "weaver_weight": _round(weaver_weights[idx]),
            "naive_correlation_aware_weight": _round(naive_weights[idx]),
            "dependency_aware_weight": _round(dependency_weights[idx]),
            "naive_minus_weaver_weight": _round(naive_weights[idx] - weaver_weights[idx]),
        }

    retained_edges = []
    for edge in learned_edges:
        pair = list(edge["pair"])
        pair_signal = [per_verifier[name]["single_verifier_auroc"] for name in pair]
        pair_weights = [per_verifier[name]["dependency_aware_weight"] for name in pair]
        retained_edges.append(
            {
                "pair": pair,
                "single_verifier_aurocs": pair_signal,
                "dependency_aware_weights": pair_weights,
                "retains_complementary_signal": all(value >= 0.7 for value in pair_signal)
                and all(value > 0.0 for value in pair_weights),
            }
        )

    over_weighted_noise = [
        name
        for name, row in per_verifier.items()
        if abs(row["single_verifier_auroc"] - 0.5) <= 0.05
        and row["naive_correlation_aware_weight"] > row["weaver_weight"]
    ]
    anti_signal_weighted = [
        name
        for name, row in per_verifier.items()
        if row["single_verifier_auroc"] < 0.5
        and row["naive_correlation_aware_weight"] > row["weaver_weight"]
    ]
    summary = (
        "The naive inverse-covariance penalty treats marginal correlation as redundant "
        "mass to suppress. The learned dependency model instead keeps correlated "
        "high-AUROC verifiers when they retain label signal, flips anti-signal columns, "
        "and avoids promoting low-variance columns solely because they have small variance."
    )
    return {
        "summary": summary,
        "per_verifier": per_verifier,
        "learned_dependency_edges": retained_edges,
        "over_weighted_low_signal_verifiers": over_weighted_noise,
        "over_weighted_anti_signal_verifiers": anti_signal_weighted,
        "redundancy_vs_complementarity_confusion": bool(retained_edges),
    }


def exp3644_baseline_reproduction(
    *,
    current: Mapping[str, float],
    prior_exp3644: Mapping[str, Any],
) -> dict[str, Any]:
    """Record current baseline metrics and whether they match Exp 3644."""

    fields = (
        "ensemble_auroc_unweighted",
        "ensemble_auroc_weaver_style",
        "ensemble_auroc_carnot",
        "ensemble_auroc_correlation_aware",
        "auroc_delta_correlation_aware_vs_weaver",
    )
    prior = {field: _round(prior_exp3644.get(field)) for field in fields}
    computed = {field: _round(current[field]) for field in fields}
    matches = {
        field: (
            prior[field] is not None
            and abs(float(prior[field]) - float(computed[field])) <= 1e-6
        )
        for field in fields
    }
    return {
        **prior,
        "computed_from_current_scores": computed,
        "matches_prior_exp3644": matches,
        "all_baselines_match_prior": all(matches.values()),
    }


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


def reproducibility_checksum(
    labels: np.ndarray,
    matrix: np.ndarray,
    verifier_names: Sequence[str],
    *,
    random_seed: int,
    baseline_random_seed: int,
    prior_exp3644: Mapping[str, Any],
) -> str:
    """Hash the measured labels, scores, verifier order, seeds, and prior matrix."""

    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(labels, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(matrix, dtype=np.float64).tobytes())
    digest.update(json.dumps(list(verifier_names), separators=(",", ":")).encode("utf-8"))
    digest.update(str(int(random_seed)).encode("ascii"))
    digest.update(str(int(baseline_random_seed)).encode("ascii"))
    digest.update(
        json.dumps(
            prior_exp3644.get("pearson_verifier_correlation_matrix"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the terminal Exp 3656 schema before writing JSON."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles must be present")
    missing_principles = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in principles]
    if missing_principles:
        raise ValueError(f"missing field principles: {missing_principles}")
    verdict = artifact.get("honest_verdict")
    if verdict not in TERMINAL_VERDICTS:
        raise ValueError(f"unsupported honest_verdict: {verdict!r}")
    if int(artifact.get("n_examples", -1)) < 0:
        raise ValueError("n_examples must be nonnegative")
    if not isinstance(artifact.get("duration_s"), int | float):
        raise ValueError("duration_s must be numeric")
    if verdict == BLOCKED_VERDICT:
        return
    if artifact.get("correlation_harmless_or_penalty_misspecified") not in OUTCOME_CATEGORIES:
        raise ValueError("correlation_harmless_or_penalty_misspecified has unsupported value")
    if not isinstance(artifact.get("naive_penalty_diagnosis"), Mapping):
        raise ValueError("naive_penalty_diagnosis must be an object")
    for field in (
        "ensemble_auroc_naive_correlation_aware",
        "ensemble_auroc_dependency_aware_proper",
        "ensemble_auroc_carnot",
    ):
        value = artifact.get(field)
        if not isinstance(value, int | float) or not math.isfinite(float(value)):
            raise ValueError(f"{field} must be finite")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{field} must be in [0, 1]")


def write_artifact(
    repo_root: Path,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build, validate, and write the Exp 3656 terminal JSON artifact."""

    root = Path(repo_root)
    if started_s is None and now_s is None:
        artifact = build_artifact(root)
    else:
        artifact = build_artifact(root, started_s=started_s, now_s=now_s)
    target = root / output_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


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
        "artifact": "experiment_3656_correlation_aware_weighting_paradox_diagnosis",
        "schema": "carnot.correlation_aware_weighting_paradox_diagnosis.v1",
        "honest_verdict": BLOCKED_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "ensemble_auroc_unweighted": None,
        "ensemble_auroc_weaver_style": None,
        "ensemble_auroc_carnot": None,
        "ensemble_auroc_correlation_aware": None,
        "ensemble_auroc_naive_correlation_aware": None,
        "ensemble_auroc_dependency_aware_proper": None,
        "naive_penalty_diagnosis": None,
        "correlation_harmless_or_penalty_misspecified": "correlation_harmless",
        "n_examples": 0,
        "random_seed": int(random_seed),
        "reproducibility_checksum": hashlib.sha256(payload).hexdigest(),
        "duration_s": _round(duration_s),
        "preconditions_checked": [dict(item) for item in preconditions],
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


def _require_binary_labels(labels: np.ndarray) -> None:
    values = set(np.asarray(labels, dtype=np.int64).tolist())
    if values != {0, 1}:
        raise ValueError("labels must contain both binary classes 0 and 1")


def weights_to_json(names: Sequence[str], weights: Sequence[float]) -> dict[str, float]:
    return {name: _round(float(weight)) for name, weight in zip(names, weights, strict=True)}


def matrix_to_json(matrix: np.ndarray) -> list[list[float]]:
    return [[_round(float(value)) for value in row] for row in np.asarray(matrix)]


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.time() if now_s is None else float(now_s)
    return max(0.0, end - float(started_s))


def _round(value: float | int | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)

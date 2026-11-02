"""Exp 3667 clean dependency-aware weighting significance rerun.

This module reruns the Exp 3656 dependency-aware weighting lead without the
aliased AUROC fields that triggered the TAUTOLOGY flag. It scores the cached
FoVer corpus with the same four Exp 2837 verifiers, learns the
label-conditional dependency graph from Exp 3656, and adds bootstrap plus
paired DeLong evidence before any headline-candidate verdict is emitted.

Spec: REQ-VERIFY-3667, SCENARIO-VERIFY-3667.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from carnot.verify import correlation_aware_weighting_paradox_diagnosis as exp3656
from carnot.verify import weaver_peer_comparison_v3 as exp3644


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results/experiment_3667_dependency_aware_weighting_clean.json")
DEFAULT_N_EXAMPLES = exp3644.DEFAULT_N_EXAMPLES
DEFAULT_CORPUS_RANDOM_SEED = exp3644.DEFAULT_RANDOM_SEED
DEFAULT_RANDOM_SEED = 3667
DEFAULT_CROSSFIT_FOLDS = 5
DEFAULT_BOOTSTRAP_SEEDS = (3667, 3668, 3669, 3670, 3671)
DEFAULT_BOOTSTRAP_REPS = 200
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates "
    "(principle: scores the cached FoVer corpus; no LLM load)."
)

SUCCESS_SIGNIFICANT = (
    "complete: dependency_aware_weighting_beats_carnot_clean_significant_headline_candidate"
)
SUCCESS_NO_GAIN = (
    "complete: dependency_aware_weighting_no_significant_gain_clean_carnot_weighting_sufficient"
)
BLOCKED_VERDICT = "complete: blocked_fover_corpus_or_dependency_weighting_unavailable"
TERMINAL_VERDICTS = (SUCCESS_SIGNIFICANT, SUCCESS_NO_GAIN, BLOCKED_VERDICT)
OUTCOME_CATEGORIES = ("beats_carnot_significant", "no_significant_gain", "blocked")

AUROC_FIELDS = (
    "auroc_unweighted",
    "auroc_weaver_style",
    "auroc_carnot_current",
    "auroc_dependency_aware_proper",
)
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "auroc_unweighted",
    "auroc_weaver_style",
    "auroc_carnot_current",
    "auroc_dependency_aware_proper",
    "dependency_aware_vs_carnot_delta_ci",
    "delong_p_dependency_vs_carnot",
    "adversarial_verify_clean",
    "dependency_aware_beats_carnot",
    "n_examples",
    "n_seeds",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": (
        "verifier_ensemble_against_cached_candidates (principle: scores the "
        "cached FoVer corpus; no LLM load)."
    ),
    "auroc_unweighted": "Baseline -- one field, no alias.",
    "auroc_weaver_style": (
        "Weaver weak-supervision weighting (arXiv:2506.18203) -- one field, no alias."
    ),
    "auroc_carnot_current": (
        "Carnot's current weighting (~0.919) -- the bar; one field, no alias."
    ),
    "auroc_dependency_aware_proper": (
        "The arXiv:1903.05844 learned-dependency-structure weighting -- the "
        "candidate; one field, no alias."
    ),
    "dependency_aware_vs_carnot_delta_ci": (
        "Paired delta + bootstrap CI of dependency-aware minus Carnot -- the "
        "headline-advancement magnitude."
    ),
    "delong_p_dependency_vs_carnot": (
        "DeLong paired significance of the AUROC difference -- a point estimate "
        "alone is not enough to move a headline."
    ),
    "adversarial_verify_clean": (
        "True iff the artifact passes adversarial_verify with no "
        "TAUTOLOGY/critical flag -- de-flags the exp3656 finding."
    ),
    "dependency_aware_beats_carnot": (
        "BARE bool. True iff dependency-aware AUROC > Carnot AND the delta CI "
        "excludes 0 AND DeLong p<0.05 -- the headline-advancement-candidate gate."
    ),
    "n_examples": "Sample-size rigor (FoVer n>=1000).",
    "n_seeds": "Replication count (>=5).",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection.",
    "duration_s": "Plausibility floor.",
}


@dataclass(frozen=True)
class OutcomeClassification:
    """Terminal verdict and bare gate boolean for a measured outcome."""

    category: str
    terminal_verdict: str
    dependency_aware_beats_carnot: bool


def build_artifact(
    repo_root: Path,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    n_examples: int = DEFAULT_N_EXAMPLES,
    random_seed: int = DEFAULT_RANDOM_SEED,
    corpus_random_seed: int = DEFAULT_CORPUS_RANDOM_SEED,
    bootstrap_seeds: Sequence[int] = DEFAULT_BOOTSTRAP_SEEDS,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_REPS,
    adversarial_verify_clean: bool = False,
) -> dict[str, Any]:
    """Build the Exp 3667 artifact from cached FoVer verifier scores."""

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
        prior_exp3644 = exp3656.load_prior_exp3644_artifact(root)
        labels, scores_by_verifier = exp3644.score_fover_corpus(
            root,
            n_examples=n_examples,
            random_seed=corpus_random_seed,
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
        corpus_random_seed=corpus_random_seed,
        bootstrap_seeds=bootstrap_seeds,
        n_bootstrap=n_bootstrap,
        adversarial_verify_clean=adversarial_verify_clean,
        preconditions=preconditions,
    )


def probe_preconditions(repo_root: Path, *, n_examples: int) -> list[dict[str, Any]]:
    """Check FoVer scoring, Exp 3644 metadata, and dependency-aware implementation."""

    checks = list(exp3656.probe_preconditions(Path(repo_root), n_examples=n_examples))
    dependency_functions = (
        exp3656.dependency_aware_crossfit_scores,
        exp3656.fit_dependency_aware_weights,
        exp3656.learn_dependency_graph,
    )
    checks.append(
        {
            "resource": "exp3656_dependency_aware_implementation",
            "available": all(callable(func) for func in dependency_functions),
            "detail": "dependency graph, crossfit scores, and graph-aware weights importable",
        }
    )
    return checks


def build_artifact_from_scores(
    *,
    labels: Sequence[int],
    scores_by_verifier: Mapping[str, Sequence[float]],
    prior_exp3644: Mapping[str, Any],
    started_s: float,
    now_s: float | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    corpus_random_seed: int = DEFAULT_CORPUS_RANDOM_SEED,
    bootstrap_seeds: Sequence[int] = DEFAULT_BOOTSTRAP_SEEDS,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_REPS,
    adversarial_verify_clean: bool = False,
    preconditions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble the clean weighting panel and significance statistics."""

    labels_arr = np.asarray(labels, dtype=np.int64)
    names = list(scores_by_verifier)
    matrix = exp3656.score_matrix(scores_by_verifier, names)
    if matrix.shape[0] != len(labels_arr):
        raise ValueError("labels and verifier scores must have the same length")
    _require_binary_labels(labels_arr)

    scores = score_weighting_panel(
        labels=labels_arr,
        score_matrix=matrix,
        verifier_names=names,
        random_seed=random_seed,
    )
    aurocs = {
        "unweighted": exp3644.tie_aware_auroc(labels_arr, scores["unweighted"]),
        "weaver_style": exp3644.tie_aware_auroc(labels_arr, scores["weaver_style"]),
        "carnot_current": exp3644.tie_aware_auroc(labels_arr, scores["carnot_current"]),
        "dependency_aware_proper": exp3644.tie_aware_auroc(
            labels_arr,
            scores["dependency_aware_proper"],
        ),
    }
    delta_ci = paired_delta_ci(
        labels_arr,
        scores["dependency_aware_proper"],
        scores["carnot_current"],
        seeds=bootstrap_seeds,
        n_bootstrap=n_bootstrap,
    )
    delong = paired_delong_test(
        labels_arr,
        scores["dependency_aware_proper"],
        scores["carnot_current"],
    )
    classification = classify_outcome(
        blocked=False,
        adversarial_verify_clean=adversarial_verify_clean,
        dependency_aware_auroc=aurocs["dependency_aware_proper"],
        carnot_auroc=aurocs["carnot_current"],
        delta_ci=delta_ci,
        delong_p=delong["p_value"],
    )
    bootstrap = {
        key: bootstrap_auroc_ci(
            labels_arr,
            value,
            seeds=bootstrap_seeds,
            n_bootstrap=n_bootstrap,
        )
        for key, value in scores.items()
    }
    seed_panel = seed_replication_panel(
        labels=labels_arr,
        score_matrix=matrix,
        verifier_names=names,
        seeds=bootstrap_seeds,
    )

    artifact = {
        "artifact": "experiment_3667_dependency_aware_weighting_clean",
        "schema": "carnot.dependency_aware_weighting_clean.v1",
        "honest_verdict": classification.terminal_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "auroc_unweighted": _round_metric(aurocs["unweighted"]),
        "auroc_weaver_style": _round_metric(aurocs["weaver_style"]),
        "auroc_carnot_current": _round_metric(aurocs["carnot_current"]),
        "auroc_dependency_aware_proper": _round_metric(aurocs["dependency_aware_proper"]),
        "dependency_aware_vs_carnot_delta_ci": delta_ci,
        "delong_p_dependency_vs_carnot": _round_p(float(delong["p_value"])),
        "adversarial_verify_clean": bool(adversarial_verify_clean),
        "dependency_aware_beats_carnot": classification.dependency_aware_beats_carnot,
        "n_examples": int(len(labels_arr)),
        "n_seeds": int(len(tuple(bootstrap_seeds))),
        "random_seed": int(random_seed),
        "corpus_random_seed": int(corpus_random_seed),
        "bootstrap_seeds": [int(seed) for seed in bootstrap_seeds],
        "n_bootstrap_per_seed": int(n_bootstrap),
        "reproducibility_checksum": reproducibility_checksum(
            labels_arr,
            matrix,
            names,
            random_seed=random_seed,
            corpus_random_seed=corpus_random_seed,
            bootstrap_seeds=bootstrap_seeds,
            prior_exp3644=prior_exp3644,
        ),
        "duration_s": _round_metric(_duration(float(started_s), now_s)),
        "verifier_names": names,
        "auroc_bootstrap": bootstrap,
        "delong_dependency_vs_carnot": delong,
        "seed_panel": seed_panel,
        "dependency_aware_training_protocol": {
            "method": "stratified_crossfit_graph_sparse_signed_fisher_weights",
            "folds": DEFAULT_CROSSFIT_FOLDS,
            "learned_on_labels": True,
            "dependency_reference": "Learning Dependency Structures for Weak Supervision Models (arXiv:1903.05844)",
        },
        "exp3644_source": {
            "path": exp3656.EXP3644_REL_PATH.as_posix(),
            "baseline_reproduction": exp3656.exp3644_baseline_reproduction(
                current={
                    "ensemble_auroc_unweighted": aurocs["unweighted"],
                    "ensemble_auroc_weaver_style": aurocs["weaver_style"],
                    "ensemble_auroc_carnot": aurocs["carnot_current"],
                    "ensemble_auroc_correlation_aware": prior_exp3644.get(
                        "ensemble_auroc_correlation_aware",
                    ),
                    "auroc_delta_correlation_aware_vs_weaver": prior_exp3644.get(
                        "auroc_delta_correlation_aware_vs_weaver",
                    ),
                },
                prior_exp3644=prior_exp3644,
            ),
        },
        "de_tautology_note": (
            "Each conceptually distinct AUROC is stored under exactly one top-level "
            "field; no exp3656 correlation-aware alias is emitted."
        ),
        "acceptance_gate": {
            "condition": (
                "adversarial_verify_clean == true AND auroc_dependency_aware_proper "
                "present AND delong_p_dependency_vs_carnot present"
            ),
            "principle": (
                "A headline-advancement candidate requires a de-flagged artifact "
                "with measured AUROC and significance evidence."
            ),
            "passed": bool(
                adversarial_verify_clean
                and aurocs["dependency_aware_proper"] is not None
                and delong["p_value"] is not None
            ),
        },
        "preconditions_checked": [dict(item) for item in preconditions or []],
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


def score_weighting_panel(
    *,
    labels: np.ndarray,
    score_matrix: np.ndarray,
    verifier_names: Sequence[str],
    random_seed: int,
) -> dict[str, np.ndarray]:
    """Return one score vector per clean weighting scheme."""

    matrix = np.asarray(score_matrix, dtype=np.float64)
    names = list(verifier_names)
    crossfit = exp3656.dependency_aware_crossfit_scores(
        labels=np.asarray(labels, dtype=np.int64),
        score_matrix=matrix,
        verifier_names=names,
        random_seed=random_seed,
        n_folds=DEFAULT_CROSSFIT_FOLDS,
    )
    return {
        "unweighted": exp3644.ensemble_scores(
            matrix,
            exp3644.normalize_weights(np.ones(matrix.shape[1], dtype=float)),
        ),
        "weaver_style": exp3644.ensemble_scores(matrix, exp3644.weaver_style_weights(matrix)),
        "carnot_current": exp3644.ensemble_scores(matrix, exp3644.carnot_current_weights(names)),
        "dependency_aware_proper": crossfit.scores,
    }


def seed_replication_panel(
    *,
    labels: np.ndarray,
    score_matrix: np.ndarray,
    verifier_names: Sequence[str],
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    """Compute the four AUROCs across deterministic dependency crossfit seeds."""

    panel: list[dict[str, Any]] = []
    for seed in seeds:
        scores = score_weighting_panel(
            labels=labels,
            score_matrix=score_matrix,
            verifier_names=verifier_names,
            random_seed=int(seed),
        )
        panel.append(
            {
                "seed": int(seed),
                "auroc_unweighted": _round_metric(
                    exp3644.tie_aware_auroc(labels, scores["unweighted"]),
                ),
                "auroc_weaver_style": _round_metric(
                    exp3644.tie_aware_auroc(labels, scores["weaver_style"]),
                ),
                "auroc_carnot_current": _round_metric(
                    exp3644.tie_aware_auroc(labels, scores["carnot_current"]),
                ),
                "auroc_dependency_aware_proper": _round_metric(
                    exp3644.tie_aware_auroc(labels, scores["dependency_aware_proper"]),
                ),
            }
        )
    return panel


def bootstrap_auroc_ci(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    *,
    seeds: Sequence[int],
    n_bootstrap: int,
) -> dict[str, Any]:
    """Return a multi-seed percentile bootstrap CI for one AUROC."""

    label_arr, score_arr = checked_label_scores(labels, scores)
    point = exp3644.tie_aware_auroc(label_arr, score_arr)
    boot_values: list[float] = []
    seed_means: list[float] = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        values: list[float] = []
        for _ in range(int(n_bootstrap)):
            idx = rng.integers(0, len(label_arr), size=len(label_arr))
            if len(set(label_arr[idx].tolist())) < 2:
                continue
            value = exp3644.tie_aware_auroc(label_arr[idx], score_arr[idx])
            values.append(value)
            boot_values.append(value)
        seed_means.append(_round_metric(float(np.mean(values))) if values else _round_metric(point))
    ci_low, ci_high = percentile_ci_or_point(boot_values, point)
    return {
        "point": _round_metric(point),
        "ci95": [_round_metric(ci_low), _round_metric(ci_high)],
        "bootstrap_seeds": [int(seed) for seed in seeds],
        "seed_mean_aurocs": seed_means,
        "n_bootstrap_per_seed": int(n_bootstrap),
    }


def paired_delta_ci(
    labels: Sequence[int] | np.ndarray,
    dependency_scores: Sequence[float] | np.ndarray,
    carnot_scores: Sequence[float] | np.ndarray,
    *,
    seeds: Sequence[int],
    n_bootstrap: int,
) -> dict[str, Any]:
    """Return paired bootstrap CI for dependency-aware minus Carnot AUROC."""

    label_arr, dep_arr = checked_label_scores(labels, dependency_scores)
    _, carnot_arr = checked_label_scores(labels, carnot_scores)
    point = exp3644.tie_aware_auroc(label_arr, dep_arr) - exp3644.tie_aware_auroc(
        label_arr,
        carnot_arr,
    )
    boot_values: list[float] = []
    seed_means: list[float] = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        values: list[float] = []
        for _ in range(int(n_bootstrap)):
            idx = rng.integers(0, len(label_arr), size=len(label_arr))
            if len(set(label_arr[idx].tolist())) < 2:
                continue
            value = exp3644.tie_aware_auroc(label_arr[idx], dep_arr[idx]) - exp3644.tie_aware_auroc(
                label_arr[idx],
                carnot_arr[idx],
            )
            values.append(value)
            boot_values.append(value)
        seed_means.append(_round_metric(float(np.mean(values))) if values else _round_metric(point))
    ci_low, ci_high = percentile_ci_or_point(boot_values, point)
    return {
        "point": _round_metric(point),
        "ci95": [_round_metric(ci_low), _round_metric(ci_high)],
        "bootstrap_seeds": [int(seed) for seed in seeds],
        "seed_mean_deltas": seed_means,
        "n_bootstrap_per_seed": int(n_bootstrap),
    }


def paired_delong_test(
    labels: Sequence[int] | np.ndarray,
    dependency_scores: Sequence[float] | np.ndarray,
    carnot_scores: Sequence[float] | np.ndarray,
) -> dict[str, Any]:
    """Paired two-sided DeLong test for dependency-aware versus Carnot AUROC."""

    label_arr, dep_arr = checked_label_scores(labels, dependency_scores)
    _, carnot_arr = checked_label_scores(labels, carnot_scores)
    _require_binary_labels(label_arr)
    n_pos = int(np.sum(label_arr == 1))
    order = np.argsort(-label_arr)
    predictions = np.vstack([dep_arr, carnot_arr])[:, order]
    aucs, covariance = fast_delong(predictions, n_pos)
    diff = float(aucs[0] - aucs[1])
    variance = float(covariance[0, 0] + covariance[1, 1] - 2.0 * covariance[0, 1])
    standard_error = math.sqrt(max(variance, 0.0))
    if standard_error <= 1e-15:
        z_value = math.copysign(math.inf, diff) if diff != 0.0 else 0.0
        p_value = 0.0 if diff != 0.0 else 1.0
    else:
        z_value = diff / standard_error
        p_value = math.erfc(abs(z_value) / math.sqrt(2.0))
    return {
        "method": "paired_delong_auc_two_sided",
        "auc_dependency_aware_proper": _round_metric(float(aucs[0])),
        "auc_carnot_current": _round_metric(float(aucs[1])),
        "auc_difference": _round_metric(diff),
        "standard_error": _round_metric(standard_error),
        "z_value": _round_metric(z_value) if math.isfinite(z_value) else str(z_value),
        "p_value": _round_p(p_value),
    }


def fast_delong(predictions_sorted: np.ndarray, label_1_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Fast DeLong covariance for paired ROC curves."""

    m = int(label_1_count)
    n = predictions_sorted.shape[1] - m
    positive_examples = predictions_sorted[:, :m]
    negative_examples = predictions_sorted[:, m:]
    k = predictions_sorted.shape[0]
    tx = np.vstack([compute_midrank(positive_examples[row]) for row in range(k)])
    ty = np.vstack([compute_midrank(negative_examples[row]) for row in range(k)])
    tz = np.vstack([compute_midrank(predictions_sorted[row]) for row in range(k)])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = covariance_matrix(v01)
    sy = covariance_matrix(v10)
    return aucs, sx / m + sy / n


def compute_midrank(values: np.ndarray) -> np.ndarray:
    """Compute one-based midranks for tied DeLong scores."""

    order = np.argsort(values)
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_values = values[order]
    i = 0
    while i < values.shape[0]:
        j = i + 1
        while j < values.shape[0] and sorted_values[j] == sorted_values[i]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def covariance_matrix(values: np.ndarray) -> np.ndarray:
    """Return a two-dimensional covariance matrix for DeLong vectors."""

    if values.shape[1] < 2:
        return np.zeros((values.shape[0], values.shape[0]), dtype=np.float64)
    covariance = np.cov(values)
    if np.ndim(covariance) == 0:
        return np.asarray([[float(covariance)]], dtype=np.float64)
    return np.asarray(covariance, dtype=np.float64)


def classify_outcome(
    *,
    blocked: bool,
    adversarial_verify_clean: bool,
    dependency_aware_auroc: float | None,
    carnot_auroc: float | None,
    delta_ci: Mapping[str, Any] | None,
    delong_p: float | None,
) -> OutcomeClassification:
    """Map measured statistics onto the three allowed honest outcomes."""

    if blocked:
        return OutcomeClassification("blocked", BLOCKED_VERDICT, False)
    ci = list((delta_ci or {}).get("ci95") or [])
    ci_excludes_zero_positive = len(ci) == 2 and float(ci[0]) > 0.0
    measured_win = (
        dependency_aware_auroc is not None
        and carnot_auroc is not None
        and float(dependency_aware_auroc) > float(carnot_auroc)
        and ci_excludes_zero_positive
        and delong_p is not None
        and float(delong_p) < 0.05
    )
    if measured_win and adversarial_verify_clean:
        return OutcomeClassification("beats_carnot_significant", SUCCESS_SIGNIFICANT, True)
    return OutcomeClassification("no_significant_gain", SUCCESS_NO_GAIN, bool(measured_win))


def checked_label_scores(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert labels/scores to finite arrays with matching length."""

    label_arr = np.asarray(labels, dtype=np.int64)
    score_arr = np.asarray(scores, dtype=np.float64)
    if label_arr.shape[0] != score_arr.shape[0]:
        raise ValueError("labels and scores must have the same length")
    if not np.isfinite(score_arr).all():
        raise ValueError("scores must be finite")
    return label_arr, score_arr


def percentile_ci_or_point(values: Sequence[float], point: float) -> tuple[float, float]:
    """Return a 95% percentile interval, falling back to the point estimate."""

    if not values:
        return float(point), float(point)
    ci_low, ci_high = np.percentile(np.asarray(values, dtype=np.float64), [2.5, 97.5])
    return float(ci_low), float(ci_high)


def reproducibility_checksum(
    labels: np.ndarray,
    matrix: np.ndarray,
    verifier_names: Sequence[str],
    *,
    random_seed: int,
    corpus_random_seed: int,
    bootstrap_seeds: Sequence[int],
    prior_exp3644: Mapping[str, Any],
) -> str:
    """Hash the measured inputs, seeds, verifier order, and prior correlation matrix."""

    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(labels, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(matrix, dtype=np.float64).tobytes())
    digest.update(json.dumps(list(verifier_names), separators=(",", ":")).encode("utf-8"))
    digest.update(str(int(random_seed)).encode("ascii"))
    digest.update(str(int(corpus_random_seed)).encode("ascii"))
    digest.update(json.dumps([int(seed) for seed in bootstrap_seeds]).encode("ascii"))
    digest.update(
        json.dumps(
            prior_exp3644.get("pearson_verifier_correlation_matrix"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the terminal Exp 3667 artifact schema and anti-aliasing rule."""

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
    if type(artifact.get("dependency_aware_beats_carnot")) is not bool:
        raise ValueError("dependency_aware_beats_carnot must be a bare boolean")
    if type(artifact.get("adversarial_verify_clean")) is not bool:
        raise ValueError("adversarial_verify_clean must be a bare boolean")
    if not isinstance(artifact.get("duration_s"), int | float):
        raise ValueError("duration_s must be numeric")
    if verdict == BLOCKED_VERDICT:
        return
    if int(artifact.get("n_examples", 0)) < DEFAULT_N_EXAMPLES:
        raise ValueError(f"n_examples must be at least {DEFAULT_N_EXAMPLES}")
    if int(artifact.get("n_seeds", 0)) < 5:
        raise ValueError("n_seeds must be at least 5")
    _validate_auroc_fields(artifact)
    _validate_delta_ci(artifact.get("dependency_aware_vs_carnot_delta_ci"))
    delong_p = artifact.get("delong_p_dependency_vs_carnot")
    if not _is_finite_number(delong_p) or not 0.0 <= float(delong_p) <= 1.0:
        raise ValueError("delong_p_dependency_vs_carnot must be finite and in [0, 1]")


def _validate_auroc_fields(artifact: Mapping[str, Any]) -> None:
    seen: list[tuple[str, float]] = []
    for field in AUROC_FIELDS:
        value = artifact.get(field)
        if not _is_finite_number(value) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{field} must be finite and in [0, 1]")
        for prior_field, prior_value in seen:
            if _significant_digits_match(prior_value, float(value), 5):
                raise ValueError(f"aliased AUROC fields are forbidden: {prior_field}, {field}")
        seen.append((field, float(value)))


def _validate_delta_ci(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("dependency-aware delta CI must be an object")
    point = value.get("point")
    ci = value.get("ci95")
    if not _is_finite_number(point) or not isinstance(ci, list) or len(ci) != 2:
        raise ValueError("dependency-aware delta CI must include point and ci95")
    if not all(_is_finite_number(item) for item in ci):
        raise ValueError("dependency-aware delta CI bounds must be finite")
    if not float(ci[0]) <= float(point) <= float(ci[1]):
        raise ValueError("dependency-aware delta CI must contain its point estimate")


def write_artifact(
    repo_root: Path,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build, self-verify, and write the Exp 3667 terminal JSON artifact."""

    root = Path(repo_root)
    artifact = build_artifact(root, started_s=started_s, now_s=now_s)
    target = root / output_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if artifact["honest_verdict"] != BLOCKED_VERDICT:
        report = run_adversarial_verify_report(target)
        clean = adversarial_report_is_clean(report)
        classification = classify_outcome(
            blocked=False,
            adversarial_verify_clean=clean,
            dependency_aware_auroc=artifact["auroc_dependency_aware_proper"],
            carnot_auroc=artifact["auroc_carnot_current"],
            delta_ci=artifact["dependency_aware_vs_carnot_delta_ci"],
            delong_p=artifact["delong_p_dependency_vs_carnot"],
        )
        artifact["adversarial_verify_clean"] = clean
        artifact["dependency_aware_beats_carnot"] = classification.dependency_aware_beats_carnot
        artifact["honest_verdict"] = classification.terminal_verdict
        artifact["acceptance_gate"]["passed"] = bool(
            clean
            and artifact["auroc_dependency_aware_proper"] is not None
            and artifact["delong_p_dependency_vs_carnot"] is not None
        )
        artifact["adversarial_verify_report"] = {
            "flag_count": int(report.get("flag_count", 0)),
            "max_severity": report.get("max_severity"),
            "flags": list(report.get("flags") or []),
        }
        validate_artifact(artifact)
        target.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def run_adversarial_verify_report(path: Path) -> dict[str, Any]:  # pragma: no cover
    """Run the repository adversarial verifier and return its structured report."""

    script_path = REPO_ROOT / "scripts" / "adversarial_verify.py"
    spec = importlib.util.spec_from_file_location("carnot_adversarial_verify", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.verify_artifact(Path(path)))


def adversarial_report_is_clean(report: Mapping[str, Any]) -> bool:
    """True only when adversarial verification emitted no flags."""

    flags = list(report.get("flags") or [])
    return int(report.get("flag_count", len(flags))) == 0 and not flags


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
        "artifact": "experiment_3667_dependency_aware_weighting_clean",
        "schema": "carnot.dependency_aware_weighting_clean.v1",
        "honest_verdict": BLOCKED_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "auroc_unweighted": None,
        "auroc_weaver_style": None,
        "auroc_carnot_current": None,
        "auroc_dependency_aware_proper": None,
        "dependency_aware_vs_carnot_delta_ci": None,
        "delong_p_dependency_vs_carnot": None,
        "adversarial_verify_clean": False,
        "dependency_aware_beats_carnot": False,
        "n_examples": 0,
        "n_seeds": 0,
        "random_seed": int(random_seed),
        "reproducibility_checksum": hashlib.sha256(payload).hexdigest(),
        "duration_s": _round_metric(duration_s),
        "preconditions_checked": [dict(item) for item in preconditions],
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


def _require_binary_labels(labels: np.ndarray) -> None:
    values = set(np.asarray(labels, dtype=np.int64).tolist())
    if values != {0, 1}:
        raise ValueError("labels must contain both binary classes 0 and 1")


def _significant_digits_match(a: float, b: float, digits: int) -> bool:
    if a == b:
        return True
    if a == 0.0 or b == 0.0:
        return False
    return abs(a - b) / max(abs(a), abs(b)) < 10 ** (-digits)


def _is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(float(value))


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.time() if now_s is None else float(now_s)
    return max(0.0, end - float(started_s))


def _round_metric(value: float | int | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _round_p(value: float | int | None) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if numeric != 0.0 and abs(numeric) < 1e-6:
        return float(f"{numeric:.6g}")
    return round(numeric, 6)

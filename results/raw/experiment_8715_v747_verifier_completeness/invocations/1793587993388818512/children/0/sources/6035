"""Exp 3693 external de-entangled comparator for dependency-aware FoVer weighting.

This module compares the Exp 3680 dependency-aware re-freeze candidate against
a self-contained, published-style de-entangled / class-information-guided
baseline. The external baseline is intentionally not Carnot's graph model: it
learns class-information gain per verifier, penalizes unconditional behavioral
entanglement, orients anti-signals on train folds, and then scores held-out
FoVer rows.

Spec: REQ-VERIFY-3693, SCENARIO-VERIFY-3693.
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
from carnot.verify import dependency_aware_dual_condition_integrity as exp3680
from carnot.verify import dependency_aware_weighting_clean as exp3667
from carnot.verify import weaver_peer_comparison_v3 as exp3644


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results/experiment_3693_external_comparator_dependency_vs_deentangled.json")
EXP3667_REL_PATH = exp3667.OUTPUT_REL_PATH
EXP3680_REL_PATH = exp3680.OUTPUT_REL_PATH
DEFAULT_N_EXAMPLES = exp3680.DEFAULT_N_EXAMPLES
DEFAULT_RANDOM_SEED = 3693
DEFAULT_RANDOM_SEEDS = exp3680.DEFAULT_RANDOM_SEEDS
DEFAULT_BOOTSTRAP_REPS = exp3680.DEFAULT_BOOTSTRAP_REPS
DEFAULT_CROSSFIT_FOLDS = exp3667.DEFAULT_CROSSFIT_FOLDS
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates (principle: scores cached FoVer "
    "outputs; no LLM load; no compute-bound marker)."
)

SUCCESS_BEATS_EXTERNAL = (
    "complete: dependency_aware_candidate_beats_published_external_reweighting_baseline"
)
SUCCESS_TIES_OR_LOSES_EXTERNAL = (
    "complete: dependency_aware_candidate_ties_or_loses_external_baseline_refreeze_narrowed"
)
BLOCKED_VERDICT = "complete: blocked_fover_corpus_or_weightings_unavailable"
TERMINAL_VERDICTS = (SUCCESS_BEATS_EXTERNAL, SUCCESS_TIES_OR_LOSES_EXTERNAL, BLOCKED_VERDICT)
OUTCOME_CATEGORIES = ("candidate_beats_external", "candidate_ties_or_loses_external", "blocked")

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "dependency_aware_auroc",
    "external_comparator_auroc",
    "carnot_current_auroc",
    "weaver_style_baseline_auroc",
    "dependency_vs_external_delta_ci",
    "delong_p_dependency_vs_external",
    "external_comparator_implementation",
    "candidate_beats_external_comparator",
    "adversarial_verify_clean",
    "n_seeds",
    "n_examples",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": (
        "verifier_ensemble_against_cached_candidates (principle: scores cached "
        "FoVer outputs; no LLM load; no compute-bound marker)."
    ),
    "dependency_aware_auroc": (
        "The re-freeze candidate AUROC under the pooled protocol -- one field, no alias."
    ),
    "external_comparator_auroc": (
        "The published de-entangled/CIG-style reweighting baseline AUROC -- the "
        "external bar; a DISTINCT measurement, one field."
    ),
    "carnot_current_auroc": (
        "Carnot-current weighting under the identical protocol -- the internal bar."
    ),
    "weaver_style_baseline_auroc": (
        "The Weaver-style weak-verifier baseline (exp3644) for context."
    ),
    "dependency_vs_external_delta_ci": (
        "Paired delta + bootstrap CI of dependency-aware minus the external "
        "comparator -- the gap that decides candidate_beats_external."
    ),
    "delong_p_dependency_vs_external": (
        "DeLong paired significance vs the external baseline -- a point estimate "
        "cannot decide a head-to-head."
    ),
    "external_comparator_implementation": (
        "Honest description of the de-entangled/CIG-style baseline implemented "
        "(verifier authenticity: a real reweighting, not a relabeled copy of Carnot's)."
    ),
    "candidate_beats_external_comparator": (
        "BARE bool. True iff dependency-aware AUROC > external comparator AUROC "
        "with the delta CI excluding 0 -- whether the re-freeze candidate beats "
        "a published method, not just the internal prior. STORE AS BARE true/false."
    ),
    "adversarial_verify_clean": (
        "True iff the artifact passes adversarial_verify with no critical flag."
    ),
    "n_seeds": "Replication count.",
    "n_examples": "Sample-size rigor (FoVer n>=1000).",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection.",
    "duration_s": "Plausibility floor.",
}


@dataclass(frozen=True)
class OutcomeClassification:
    """Terminal verdict and bare comparator gate for one measured outcome."""

    category: str
    terminal_verdict: str
    candidate_beats_external_comparator: bool


@dataclass(frozen=True)
class CIGDeentangledFit:
    """Train-fold weights for the external CIG/de-entangled baseline."""

    weights: np.ndarray
    orientations: np.ndarray
    class_information_gain: list[float]
    entanglement_penalty: list[float]
    difficulty_weight_mean: float
    oriented_mean_by_label: dict[str, list[float]]


@dataclass(frozen=True)
class CIGCrossfitResult:
    """Held-out external-comparator scores and fold diagnostics."""

    scores: np.ndarray
    mean_weights: np.ndarray
    fold_weights: list[list[float]]
    fold_orientations: list[list[int]]
    mean_class_information_gain: list[float]
    mean_entanglement_penalty: list[float]
    folds: int


def classify_outcome(
    *,
    blocked: bool,
    dependency_aware_auroc: float | None,
    external_comparator_auroc: float | None,
    delta_ci: Mapping[str, Any] | None,
) -> OutcomeClassification:
    """Map measured external-comparator statistics onto allowed outcomes."""

    if blocked:
        return OutcomeClassification("blocked", BLOCKED_VERDICT, False)
    ci = list((delta_ci or {}).get("ci95") or [])
    ci_excludes_zero_positive = len(ci) == 2 and float(ci[0]) > 0.0
    beats = (
        dependency_aware_auroc is not None
        and external_comparator_auroc is not None
        and float(dependency_aware_auroc) > float(external_comparator_auroc)
        and ci_excludes_zero_positive
    )
    if beats:
        return OutcomeClassification("candidate_beats_external", SUCCESS_BEATS_EXTERNAL, True)
    return OutcomeClassification(
        "candidate_ties_or_loses_external",
        SUCCESS_TIES_OR_LOSES_EXTERNAL,
        False,
    )


def fit_cig_deentangled_weights(
    *,
    labels: Sequence[int] | np.ndarray,
    score_matrix: Sequence[Sequence[float]] | np.ndarray,
    verifier_names: Sequence[str],
) -> CIGDeentangledFit:
    """Fit the arXiv:2604.07650-style CIG/de-entangled train-fold baseline.

    The recipe is self-contained: difficulty-weighted class-information gain
    estimates verifier usefulness, train-fold score orientation handles
    anti-signals, and unconditional score correlation downweights redundant
    verifier axes. No dependency graph or Carnot weight vector is used.
    """

    labels_arr = np.asarray(labels, dtype=np.int64)
    matrix = _checked_score_matrix(score_matrix, verifier_names)
    if matrix.shape[0] != len(labels_arr):
        raise ValueError("labels and verifier scores must have the same length")
    _require_binary_labels(labels_arr)
    difficulty = difficulty_weights(matrix)
    k = matrix.shape[1]
    orientations = np.ones(k, dtype=np.int64)
    oriented = np.zeros_like(matrix, dtype=np.float64)
    cig_values: list[float] = []
    mean_by_label: dict[str, list[float]] = {"0": [], "1": []}
    direction_strength: list[float] = []

    for column_index in range(k):
        column = np.clip(matrix[:, column_index], 0.0, 1.0)
        mean_0 = _weighted_mean(column[labels_arr == 0], difficulty[labels_arr == 0])
        mean_1 = _weighted_mean(column[labels_arr == 1], difficulty[labels_arr == 1])
        if mean_1 < mean_0:
            orientations[column_index] = -1
            column = 1.0 - column
            mean_0, mean_1 = 1.0 - mean_0, 1.0 - mean_1
        oriented[:, column_index] = column
        mean_by_label["0"].append(_round_metric(mean_0))
        mean_by_label["1"].append(_round_metric(mean_1))
        cig = _weighted_class_information_gain(labels_arr, column, difficulty)
        cig_values.append(float(cig))
        direction_strength.append(abs(float(mean_1 - mean_0)))

    penalties = _entanglement_penalties(oriented)
    raw = (
        (np.asarray(cig_values, dtype=np.float64) + 1e-9)
        * (np.asarray(direction_strength, dtype=np.float64) + 1e-9)
        / (1.0 + np.asarray(penalties, dtype=np.float64))
    )
    if float(np.sum(raw)) <= 1e-12:
        raw = 1.0 / (1.0 + np.asarray(penalties, dtype=np.float64))
    weights = exp3644.normalize_weights(raw)
    return CIGDeentangledFit(
        weights=weights,
        orientations=orientations,
        class_information_gain=[_round_metric(value) for value in cig_values],
        entanglement_penalty=[_round_metric(value) for value in penalties],
        difficulty_weight_mean=_round_metric(float(np.mean(difficulty))),
        oriented_mean_by_label=mean_by_label,
    )


def apply_cig_deentangled_fit(
    score_matrix: Sequence[Sequence[float]] | np.ndarray,
    fit: CIGDeentangledFit,
) -> np.ndarray:
    """Score rows with a fitted CIG/de-entangled weight vector."""

    matrix = np.asarray(score_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("score_matrix must be two-dimensional")
    if matrix.shape[1] != len(fit.weights):
        raise ValueError("score_matrix column count must match fitted weights")
    oriented = np.clip(matrix, 0.0, 1.0).copy()
    for column_index, orientation in enumerate(fit.orientations):
        if int(orientation) < 0:
            oriented[:, column_index] = 1.0 - oriented[:, column_index]
    return exp3644.ensemble_scores(oriented, fit.weights)


def cig_deentangled_crossfit_scores(
    *,
    labels: Sequence[int] | np.ndarray,
    score_matrix: Sequence[Sequence[float]] | np.ndarray,
    verifier_names: Sequence[str],
    random_seed: int,
    n_folds: int,
) -> CIGCrossfitResult:
    """Return held-out scores from the external CIG/de-entangled comparator."""

    labels_arr = np.asarray(labels, dtype=np.int64)
    matrix = _checked_score_matrix(score_matrix, verifier_names)
    if matrix.shape[0] != len(labels_arr):
        raise ValueError("labels and verifier scores must have the same length")
    _require_binary_labels(labels_arr)
    min_class_count = min(int(np.sum(labels_arr == 0)), int(np.sum(labels_arr == 1)))
    if min_class_count < 2:
        raise ValueError("at least two examples per class are required for crossfit")
    folds = min(int(n_folds), min_class_count)
    if folds < 2:
        folds = 2
    fold_ids = exp3656.stratified_fold_ids(labels_arr, folds=folds, random_seed=random_seed)
    scores = np.zeros(len(labels_arr), dtype=np.float64)
    fold_weights: list[list[float]] = []
    fold_orientations: list[list[int]] = []
    fold_cig: list[list[float]] = []
    fold_penalties: list[list[float]] = []
    for fold in range(folds):
        train_idx = np.where(fold_ids != fold)[0]
        test_idx = np.where(fold_ids == fold)[0]
        fit = fit_cig_deentangled_weights(
            labels=labels_arr[train_idx],
            score_matrix=matrix[train_idx],
            verifier_names=verifier_names,
        )
        scores[test_idx] = apply_cig_deentangled_fit(matrix[test_idx], fit)
        fold_weights.append([float(value) for value in fit.weights])
        fold_orientations.append([int(value) for value in fit.orientations])
        fold_cig.append([float(value) for value in fit.class_information_gain])
        fold_penalties.append([float(value) for value in fit.entanglement_penalty])
    return CIGCrossfitResult(
        scores=scores,
        mean_weights=np.mean(np.asarray(fold_weights, dtype=np.float64), axis=0),
        fold_weights=fold_weights,
        fold_orientations=fold_orientations,
        mean_class_information_gain=[
            _round_metric(value) for value in np.mean(np.asarray(fold_cig, dtype=np.float64), axis=0)
        ],
        mean_entanglement_penalty=[
            _round_metric(value)
            for value in np.mean(np.asarray(fold_penalties, dtype=np.float64), axis=0)
        ],
        folds=folds,
    )


def difficulty_weights(score_matrix: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Weight ambiguous examples higher for class-information-gain fitting."""

    matrix = np.asarray(score_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("score_matrix must be two-dimensional")
    mean_score = np.mean(np.clip(matrix, 0.0, 1.0), axis=1)
    return 1.0 + 2.0 * np.clip(0.5 - np.abs(mean_score - 0.5), 0.0, 0.5)


def build_artifact(
    repo_root: Path,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    n_examples: int = DEFAULT_N_EXAMPLES,
    random_seed: int = DEFAULT_RANDOM_SEED,
    random_seeds: Sequence[int] = DEFAULT_RANDOM_SEEDS,
    bootstrap_seeds: Sequence[int] = DEFAULT_RANDOM_SEEDS,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_REPS,
    adversarial_verify_clean: bool = False,
) -> dict[str, Any]:
    """Build the Exp 3693 artifact from local FoVer rows or fail closed."""

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
        source_artifacts = load_source_artifacts(root)
        state_files = exp3680.discover_fr11_state_files(root)
        rows = [
            exp3680.score_dual_condition_rows(
                root,
                seed=int(seed),
                n_examples=n_examples,
                state_files=state_files,
            )
            for seed in random_seeds
        ]
    except Exception as exc:  # noqa: BLE001 - terminal artifact must fail closed.
        failed = [
            *preconditions,
            {
                "resource": "dual_condition_scoring",
                "available": False,
                "detail": f"{type(exc).__name__}: {exc}",
            },
        ]
        return _blocked_artifact(
            duration_s=_duration(start, now_s),
            random_seed=random_seed,
            preconditions=failed,
        )
    return build_artifact_from_condition_rows(
        rows,
        started_s=start,
        now_s=now_s,
        random_seed=random_seed,
        bootstrap_seeds=bootstrap_seeds,
        n_bootstrap=n_bootstrap,
        adversarial_verify_clean=adversarial_verify_clean,
        preconditions=preconditions,
        source_artifacts=source_artifacts,
    )


def probe_preconditions(repo_root: Path, *, n_examples: int) -> list[dict[str, Any]]:
    """Check FoVer, verifier scoring, and prior dependency-aware artifacts."""

    root = Path(repo_root)
    checks = list(exp3680.probe_preconditions(root, n_examples=n_examples))
    try:
        sources = load_source_artifacts(root)
        detail = (
            "exp3667_dependency="
            f"{sources['exp3667'].get('auroc_dependency_aware_proper')}; "
            "exp3680_dependency="
            f"{sources['exp3680'].get('production_auroc_dependency_aware')}"
        )
        available = True
    except Exception as exc:  # noqa: BLE001 - precondition diagnostics belong in artifact.
        detail = f"{type(exc).__name__}: {exc}"
        available = False
    checks.append(
        {
            "resource": "exp3667_exp3680_dependency_weighting_sources",
            "available": available,
            "detail": detail,
        }
    )
    checks.append(
        {
            "resource": "cig_deentangled_external_comparator_implementation",
            "available": all(
                callable(func)
                for func in (
                    fit_cig_deentangled_weights,
                    cig_deentangled_crossfit_scores,
                    classify_outcome,
                )
            ),
            "detail": "CIG/de-entangled crossfit scorer importable",
        }
    )
    return checks


def load_source_artifacts(repo_root: Path) -> dict[str, dict[str, Any]]:
    """Load prior Exp 3667 and Exp 3680 artifacts for provenance/preconditions."""

    root = Path(repo_root)
    exp3667_payload = json.loads((root / EXP3667_REL_PATH).read_text(encoding="utf-8"))
    exp3680_payload = json.loads((root / EXP3680_REL_PATH).read_text(encoding="utf-8"))
    for field in (
        "auroc_dependency_aware_proper",
        "auroc_carnot_current",
        "auroc_weaver_style",
        "adversarial_verify_clean",
    ):
        if field not in exp3667_payload:
            raise ValueError(f"Exp 3667 source artifact is missing {field}")
    for field in (
        "production_auroc_dependency_aware",
        "production_auroc_carnot_current",
        "adversarial_verify_clean",
        "leak_free",
        "n_seeds",
        "n_examples",
    ):
        if field not in exp3680_payload:
            raise ValueError(f"Exp 3680 source artifact is missing {field}")
    return {"exp3667": exp3667_payload, "exp3680": exp3680_payload}


def build_artifact_from_condition_rows(
    condition_rows: Sequence[exp3680.ConditionScoreRows],
    *,
    started_s: float,
    now_s: float | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    bootstrap_seeds: Sequence[int] = DEFAULT_RANDOM_SEEDS,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_REPS,
    adversarial_verify_clean: bool = False,
    preconditions: Sequence[Mapping[str, Any]] | None = None,
    source_artifacts: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble the external-comparator panel from already-scored split rows."""

    rows = list(condition_rows)
    if not rows:
        raise ValueError("at least one condition row panel is required")

    names = list(exp3644.VERIFIER_NAMES)
    per_seed: list[dict[str, Any]] = []
    pooled_labels: list[np.ndarray] = []
    pooled_dependency: list[np.ndarray] = []
    pooled_external: list[np.ndarray] = []
    pooled_carnot: list[np.ndarray] = []
    pooled_weaver: list[np.ndarray] = []
    external_fold_weights: list[dict[str, float]] = []
    external_fold_orientations: list[dict[str, int]] = []
    external_cig_values: list[list[float]] = []
    external_entanglement_values: list[list[float]] = []

    for row in rows:
        labels = np.asarray(row.labels, dtype=np.int64)
        production_matrix = exp3656.score_matrix(row.production_scores_by_verifier, names)
        if production_matrix.shape[0] != len(labels):
            raise ValueError("labels and verifier scores must have the same length")
        _require_binary_labels(labels)
        panel = exp3667.score_weighting_panel(
            labels=labels,
            score_matrix=production_matrix,
            verifier_names=names,
            random_seed=int(row.seed),
        )
        external = cig_deentangled_crossfit_scores(
            labels=labels,
            score_matrix=production_matrix,
            verifier_names=names,
            random_seed=int(row.seed),
            n_folds=DEFAULT_CROSSFIT_FOLDS,
        )
        pooled_labels.append(labels)
        pooled_dependency.append(np.asarray(panel["dependency_aware_proper"], dtype=np.float64))
        pooled_external.append(np.asarray(external.scores, dtype=np.float64))
        pooled_carnot.append(np.asarray(panel["carnot_current"], dtype=np.float64))
        pooled_weaver.append(np.asarray(panel["weaver_style"], dtype=np.float64))
        external_fold_weights.append(_weights_to_json(names, external.mean_weights))
        external_fold_orientations.append(
            {name: int(value) for name, value in zip(names, _mean_orientation(external.fold_orientations), strict=True)}
        )
        external_cig_values.append(external.mean_class_information_gain)
        external_entanglement_values.append(external.mean_entanglement_penalty)
        dep_auc = exp3644.tie_aware_auroc(labels, panel["dependency_aware_proper"])
        ext_auc = exp3644.tie_aware_auroc(labels, external.scores)
        per_seed.append(
            {
                "seed": int(row.seed),
                "n_examples": int(len(labels)),
                "dependency_aware_auroc": _round_metric(dep_auc),
                "external_comparator_auroc": _round_metric(ext_auc),
                "carnot_current_auroc": _round_metric(
                    exp3644.tie_aware_auroc(labels, panel["carnot_current"])
                ),
                "weaver_style_baseline_auroc": _round_metric(
                    exp3644.tie_aware_auroc(labels, panel["weaver_style"])
                ),
                "dependency_minus_external_auroc": _round_metric(dep_auc - ext_auc),
                "external_mean_weights": _weights_to_json(names, external.mean_weights),
                "external_mean_orientations": {
                    name: int(value)
                    for name, value in zip(names, _mean_orientation(external.fold_orientations), strict=True)
                },
                "subset_sha256": row.subset_sha256,
            }
        )

    pooled_label_arr = np.concatenate(pooled_labels)
    pooled_dependency_arr = np.concatenate(pooled_dependency)
    pooled_external_arr = np.concatenate(pooled_external)
    pooled_carnot_arr = np.concatenate(pooled_carnot)
    pooled_weaver_arr = np.concatenate(pooled_weaver)
    dependency_auroc = exp3644.tie_aware_auroc(pooled_label_arr, pooled_dependency_arr)
    external_auroc = exp3644.tie_aware_auroc(pooled_label_arr, pooled_external_arr)
    carnot_auroc = exp3644.tie_aware_auroc(pooled_label_arr, pooled_carnot_arr)
    weaver_auroc = exp3644.tie_aware_auroc(pooled_label_arr, pooled_weaver_arr)
    delta_ci = exp3667.paired_delta_ci(
        pooled_label_arr,
        pooled_dependency_arr,
        pooled_external_arr,
        seeds=bootstrap_seeds,
        n_bootstrap=n_bootstrap,
    )
    delong = exp3667.paired_delong_test(
        pooled_label_arr,
        pooled_dependency_arr,
        pooled_external_arr,
    )
    classification = classify_outcome(
        blocked=False,
        dependency_aware_auroc=dependency_auroc,
        external_comparator_auroc=external_auroc,
        delta_ci=delta_ci,
    )
    checksums = {
        "dependency_aware": vector_checksum(pooled_dependency_arr),
        "external_comparator": vector_checksum(pooled_external_arr),
        "carnot_current": vector_checksum(pooled_carnot_arr),
        "weaver_style_baseline": vector_checksum(pooled_weaver_arr),
    }
    source = {key: dict(value) for key, value in (source_artifacts or {}).items()}
    artifact = {
        "artifact": "experiment_3693_external_comparator_dependency_vs_deentangled",
        "schema": "carnot.external_comparator_dependency_vs_deentangled.v1",
        "honest_verdict": classification.terminal_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "dependency_aware_auroc": _round_metric(dependency_auroc),
        "external_comparator_auroc": _round_metric(external_auroc),
        "carnot_current_auroc": _round_metric(carnot_auroc),
        "weaver_style_baseline_auroc": _round_metric(weaver_auroc),
        "dependency_aware_auroc_ci": exp3667.bootstrap_auroc_ci(
            pooled_label_arr,
            pooled_dependency_arr,
            seeds=bootstrap_seeds,
            n_bootstrap=n_bootstrap,
        ),
        "external_comparator_auroc_ci": exp3667.bootstrap_auroc_ci(
            pooled_label_arr,
            pooled_external_arr,
            seeds=bootstrap_seeds,
            n_bootstrap=n_bootstrap,
        ),
        "dependency_vs_external_delta_ci": delta_ci,
        "delong_p_dependency_vs_external": _round_p(float(delong["p_value"])),
        "delong_dependency_vs_external": {
            "method": delong["method"],
            "auc_dependency_aware": delong["auc_dependency_aware_proper"],
            "auc_external_comparator": delong["auc_carnot_current"],
            "auc_difference": delong["auc_difference"],
            "standard_error": delong["standard_error"],
            "z_value": delong["z_value"],
            "p_value": delong["p_value"],
        },
        "external_comparator_implementation": {
            "reference": "arXiv:2604.07650",
            "method": "crossfit_difficulty_weighted_class_information_gain_deentangled",
            "description": (
                "For each train fold, verifier scores are oriented so higher means "
                "label=1, difficulty-weighted class-information gain estimates "
                "label utility, unconditional oriented-score correlations estimate "
                "behavioral entanglement, and normalized weights score held-out "
                "FoVer rows. This is not Carnot-current weighting and does not use "
                "the label-conditional dependency graph from arXiv:1903.05844."
            ),
            "learned_on_labels_for_weight_fit": True,
            "verifier_scores_use_gold_label": False,
            "folds": DEFAULT_CROSSFIT_FOLDS,
            "mean_weights_across_seeds": _weights_to_json(
                names,
                np.mean(
                    np.asarray(
                        [[weights[name] for name in names] for weights in external_fold_weights],
                        dtype=np.float64,
                    ),
                    axis=0,
                ),
            ),
            "mean_class_information_gain": _weights_to_json(
                names,
                np.mean(np.asarray(external_cig_values, dtype=np.float64), axis=0),
            ),
            "mean_entanglement_penalty": _weights_to_json(
                names,
                np.mean(np.asarray(external_entanglement_values, dtype=np.float64), axis=0),
            ),
        },
        "candidate_beats_external_comparator": (
            classification.candidate_beats_external_comparator
        ),
        "adversarial_verify_clean": bool(adversarial_verify_clean),
        "n_seeds": int(len(rows)),
        "n_examples": int(min(len(row.labels) for row in rows)),
        "n_pooled_examples": int(len(pooled_label_arr)),
        "random_seed": int(random_seed),
        "random_seeds_used": [int(row.seed) for row in rows],
        "bootstrap_seeds": [int(seed) for seed in bootstrap_seeds],
        "n_bootstrap_per_seed": int(n_bootstrap),
        "reproducibility_checksum": reproducibility_checksum(
            rows=rows,
            verifier_names=names,
            random_seed=random_seed,
            bootstrap_seeds=bootstrap_seeds,
            source_artifacts=source,
            external_scores=pooled_external_arr,
        ),
        "duration_s": _round_metric(_duration(float(started_s), now_s)),
        "verifier_names": names,
        "per_seed_results": per_seed,
        "external_comparator_mean_weights_by_seed": external_fold_weights,
        "external_comparator_mean_orientations_by_seed": external_fold_orientations,
        "score_vector_checksums": checksums,
        "exp3667_source": {
            "path": EXP3667_REL_PATH.as_posix(),
            "auroc_dependency_aware_proper": source.get("exp3667", {}).get(
                "auroc_dependency_aware_proper"
            ),
            "auroc_carnot_current": source.get("exp3667", {}).get("auroc_carnot_current"),
            "auroc_weaver_style": source.get("exp3667", {}).get("auroc_weaver_style"),
            "adversarial_verify_clean": source.get("exp3667", {}).get(
                "adversarial_verify_clean"
            ),
        },
        "exp3680_source": {
            "path": EXP3680_REL_PATH.as_posix(),
            "production_auroc_dependency_aware": source.get("exp3680", {}).get(
                "production_auroc_dependency_aware"
            ),
            "production_auroc_carnot_current": source.get("exp3680", {}).get(
                "production_auroc_carnot_current"
            ),
            "adversarial_verify_clean": source.get("exp3680", {}).get(
                "adversarial_verify_clean"
            ),
            "leak_free": source.get("exp3680", {}).get("leak_free"),
            "n_seeds": source.get("exp3680", {}).get("n_seeds"),
            "n_examples": source.get("exp3680", {}).get("n_examples"),
        },
        "de_tautology_note": (
            "The external comparator AUROC and dependency-aware AUROC are stored "
            "under exactly one field each; score_vector_checksums must differ."
        ),
        "acceptance_gate": {
            "condition": (
                "dependency_aware_auroc present AND external_comparator_auroc "
                "present AND dependency_vs_external_delta_ci present AND "
                "adversarial_verify_clean == true"
            ),
            "principle": (
                "An external-comparator verdict requires both AUROCs measured "
                "under the identical protocol and the paired delta CI, "
                "adversarial-clean -- comparing only to the internal prior would "
                "leave the credibility gap this task closes."
            ),
            "passed": bool(
                dependency_auroc is not None
                and external_auroc is not None
                and delta_ci is not None
                and adversarial_verify_clean
            ),
        },
        "preconditions_checked": [dict(item) for item in preconditions or []],
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


def reproducibility_checksum(
    *,
    rows: Sequence[exp3680.ConditionScoreRows],
    verifier_names: Sequence[str],
    random_seed: int,
    bootstrap_seeds: Sequence[int],
    source_artifacts: Mapping[str, Mapping[str, Any]],
    external_scores: np.ndarray,
) -> str:
    """Hash measured rows, method outputs, seeds, and source-artifact anchors."""

    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(int(row.seed)).encode("ascii"))
        digest.update(np.ascontiguousarray(row.labels, dtype=np.int64).tobytes())
        digest.update(
            np.ascontiguousarray(
                exp3656.score_matrix(row.production_scores_by_verifier, verifier_names),
                dtype=np.float64,
            ).tobytes()
        )
    digest.update(np.ascontiguousarray(external_scores, dtype=np.float64).tobytes())
    digest.update(json.dumps(list(verifier_names), separators=(",", ":")).encode("utf-8"))
    digest.update(str(int(random_seed)).encode("ascii"))
    digest.update(json.dumps([int(seed) for seed in bootstrap_seeds]).encode("ascii"))
    digest.update(
        json.dumps(
            {
                "exp3667": {
                    "auroc_dependency_aware_proper": source_artifacts.get("exp3667", {}).get(
                        "auroc_dependency_aware_proper"
                    ),
                    "adversarial_verify_clean": source_artifacts.get("exp3667", {}).get(
                        "adversarial_verify_clean"
                    ),
                },
                "exp3680": {
                    "production_auroc_dependency_aware": source_artifacts.get(
                        "exp3680", {}
                    ).get("production_auroc_dependency_aware"),
                    "adversarial_verify_clean": source_artifacts.get("exp3680", {}).get(
                        "adversarial_verify_clean"
                    ),
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()


def vector_checksum(values: Sequence[float] | np.ndarray) -> str:
    """Stable checksum for one full score vector."""

    arr = np.asarray(values, dtype=np.float64)
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float64).tobytes()).hexdigest()


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3693 schema, bare bool, and anti-copy discipline."""

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
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must use the cached-verifier sentinel")
    for field in ("candidate_beats_external_comparator", "adversarial_verify_clean"):
        if type(artifact.get(field)) is not bool:
            raise ValueError(f"{field} must be a bare boolean")
    if not isinstance(artifact.get("duration_s"), int | float):
        raise ValueError("duration_s must be numeric")
    if verdict == BLOCKED_VERDICT:
        return
    if int(artifact.get("n_seeds", 0)) < 5:
        raise ValueError("n_seeds must be at least 5")
    if int(artifact.get("n_examples", 0)) < DEFAULT_N_EXAMPLES:
        raise ValueError(f"n_examples must be at least {DEFAULT_N_EXAMPLES}")
    for field in (
        "dependency_aware_auroc",
        "external_comparator_auroc",
        "carnot_current_auroc",
        "weaver_style_baseline_auroc",
    ):
        _validate_auroc_field(artifact, field)
    _validate_ci(artifact.get("dependency_vs_external_delta_ci"), "dependency_vs_external_delta_ci")
    _validate_ci(artifact.get("dependency_aware_auroc_ci"), "dependency_aware_auroc_ci")
    _validate_ci(artifact.get("external_comparator_auroc_ci"), "external_comparator_auroc_ci")
    delong_p = artifact.get("delong_p_dependency_vs_external")
    if not _is_finite_number(delong_p) or not 0.0 <= float(delong_p) <= 1.0:
        raise ValueError("delong_p_dependency_vs_external must be finite and in [0, 1]")
    implementation = artifact.get("external_comparator_implementation")
    if not isinstance(implementation, Mapping) or implementation.get("reference") != "arXiv:2604.07650":
        raise ValueError("external_comparator_implementation must cite arXiv:2604.07650")
    checksums = artifact.get("score_vector_checksums")
    if not isinstance(checksums, Mapping):
        raise ValueError("score_vector_checksums must be present")
    if checksums.get("dependency_aware") == checksums.get("external_comparator"):
        raise ValueError("dependency-aware and external score vector checksums must differ")
    expected_bool = _candidate_beats_from_metrics(
        artifact.get("dependency_aware_auroc"),
        artifact.get("external_comparator_auroc"),
        artifact.get("dependency_vs_external_delta_ci"),
    )
    if artifact.get("candidate_beats_external_comparator") is not expected_bool:
        raise ValueError("candidate_beats_external_comparator does not match AUROC and CI")


def write_artifact(
    repo_root: Path,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build, adversarially verify, and write the Exp 3693 terminal artifact."""

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
            dependency_aware_auroc=artifact["dependency_aware_auroc"],
            external_comparator_auroc=artifact["external_comparator_auroc"],
            delta_ci=artifact["dependency_vs_external_delta_ci"],
        )
        artifact["adversarial_verify_clean"] = clean
        artifact["candidate_beats_external_comparator"] = (
            classification.candidate_beats_external_comparator
        )
        artifact["honest_verdict"] = classification.terminal_verdict
        artifact["acceptance_gate"]["passed"] = bool(
            artifact["dependency_aware_auroc"] is not None
            and artifact["external_comparator_auroc"] is not None
            and artifact["dependency_vs_external_delta_ci"] is not None
            and clean
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
    """True when adversarial verification emitted no critical or TAUTOLOGY flag."""

    for flag in list(report.get("flags") or []):
        item = dict(flag)
        if str(item.get("kind", "")) == "TAUTOLOGY" or str(item.get("severity", "")) == "critical":
            return False
    return True


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
        "artifact": "experiment_3693_external_comparator_dependency_vs_deentangled",
        "schema": "carnot.external_comparator_dependency_vs_deentangled.v1",
        "honest_verdict": BLOCKED_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "dependency_aware_auroc": None,
        "external_comparator_auroc": None,
        "carnot_current_auroc": None,
        "weaver_style_baseline_auroc": None,
        "dependency_vs_external_delta_ci": None,
        "delong_p_dependency_vs_external": None,
        "external_comparator_implementation": {
            "reference": "arXiv:2604.07650",
            "method": "blocked_before_scoring",
            "description": "FoVer corpus, verifier outputs, or dependency-aware weighting unavailable.",
        },
        "candidate_beats_external_comparator": False,
        "adversarial_verify_clean": False,
        "n_seeds": 0,
        "n_examples": 0,
        "random_seed": int(random_seed),
        "reproducibility_checksum": hashlib.sha256(payload).hexdigest(),
        "duration_s": _round_metric(duration_s),
        "acceptance_gate": {
            "condition": (
                "dependency_aware_auroc present AND external_comparator_auroc "
                "present AND dependency_vs_external_delta_ci present AND "
                "adversarial_verify_clean == true"
            ),
            "principle": (
                "An external-comparator verdict requires both AUROCs measured "
                "under the identical protocol and the paired delta CI, "
                "adversarial-clean -- comparing only to the internal prior would "
                "leave the credibility gap this task closes."
            ),
            "passed": False,
        },
        "preconditions_checked": [dict(item) for item in preconditions],
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


def _weighted_class_information_gain(
    labels: np.ndarray,
    scores: np.ndarray,
    sample_weight: np.ndarray,
) -> float:
    base_entropy = _weighted_binary_entropy(labels, sample_weight)
    if base_entropy <= 1e-12:
        return 0.0
    edges = np.unique(np.quantile(scores, [0.25, 0.50, 0.75]))
    bin_ids = np.digitize(scores, edges, right=False)
    total_weight = float(np.sum(sample_weight))
    conditional = 0.0
    for bin_id in sorted(set(bin_ids.tolist())):
        mask = bin_ids == bin_id
        weight = float(np.sum(sample_weight[mask]))
        if weight <= 0.0:
            continue
        conditional += (weight / total_weight) * _weighted_binary_entropy(
            labels[mask],
            sample_weight[mask],
        )
    return max(0.0, (base_entropy - conditional) / base_entropy)


def _weighted_binary_entropy(labels: np.ndarray, sample_weight: np.ndarray) -> float:
    total = float(np.sum(sample_weight))
    if total <= 0.0:
        return 0.0
    p_one = float(np.sum(sample_weight[labels == 1]) / total)
    if p_one <= 1e-12 or p_one >= 1.0 - 1e-12:
        return 0.0
    return float(-(p_one * math.log2(p_one) + (1.0 - p_one) * math.log2(1.0 - p_one)))


def _weighted_mean(values: np.ndarray, sample_weight: np.ndarray) -> float:
    total = float(np.sum(sample_weight))
    if total <= 0.0:
        return float(np.mean(values)) if len(values) else 0.0
    return float(np.sum(values * sample_weight) / total)


def _entanglement_penalties(oriented_matrix: np.ndarray) -> list[float]:
    penalties: list[float] = []
    for i in range(oriented_matrix.shape[1]):
        values = []
        for j in range(oriented_matrix.shape[1]):
            if i == j:
                continue
            values.append(abs(exp3644.safe_pearson(oriented_matrix[:, i], oriented_matrix[:, j])))
        penalties.append(float(np.mean(values)) if values else 0.0)
    return penalties


def _checked_score_matrix(
    score_matrix: Sequence[Sequence[float]] | np.ndarray,
    verifier_names: Sequence[str],
) -> np.ndarray:
    matrix = np.asarray(score_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("score_matrix must be two-dimensional")
    if matrix.shape[1] != len(list(verifier_names)):
        raise ValueError("score_matrix column count must match verifier_names")
    if not np.isfinite(matrix).all():
        raise ValueError("score_matrix must be finite")
    return matrix


def _mean_orientation(fold_orientations: Sequence[Sequence[int]]) -> list[int]:
    arr = np.asarray(fold_orientations, dtype=np.float64)
    means = np.mean(arr, axis=0)
    return [1 if value >= 0.0 else -1 for value in means]


def _candidate_beats_from_metrics(
    dependency_aware_auroc: Any,
    external_comparator_auroc: Any,
    delta_ci: Any,
) -> bool:
    if not _is_finite_number(dependency_aware_auroc) or not _is_finite_number(external_comparator_auroc):
        return False
    if not isinstance(delta_ci, Mapping):
        return False
    ci = delta_ci.get("ci95")
    return (
        isinstance(ci, list)
        and len(ci) == 2
        and _is_finite_number(ci[0])
        and float(dependency_aware_auroc) > float(external_comparator_auroc)
        and float(ci[0]) > 0.0
    )


def _require_binary_labels(labels: np.ndarray) -> None:
    values = set(np.asarray(labels, dtype=np.int64).tolist())
    if values != {0, 1}:
        raise ValueError("labels must contain both binary classes 0 and 1")


def _validate_auroc_field(artifact: Mapping[str, Any], field: str) -> None:
    value = artifact.get(field)
    if not _is_finite_number(value) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{field} must be finite and in [0, 1]")


def _validate_ci(value: Any, field: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    point = value.get("point")
    ci = value.get("ci95")
    if not _is_finite_number(point) or not isinstance(ci, list) or len(ci) != 2:
        raise ValueError(f"{field} must include point and ci95")
    if not all(_is_finite_number(item) for item in ci):
        raise ValueError(f"{field} bounds must be finite")
    if not float(ci[0]) <= float(point) <= float(ci[1]):
        raise ValueError(f"{field} must contain its point estimate")


def _weights_to_json(names: Sequence[str], weights: Sequence[float]) -> dict[str, float]:
    return {name: _round_metric(float(weight)) for name, weight in zip(names, weights, strict=True)}


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

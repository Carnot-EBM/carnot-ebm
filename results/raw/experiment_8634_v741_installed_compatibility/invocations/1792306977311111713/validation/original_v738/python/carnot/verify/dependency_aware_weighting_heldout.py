"""Exp 3668 held-out dependency-aware FoVer weighting.

This module answers the follow-up question from Exp 3667: whether the learned
dependency graph and signed weights still help when they are learned on TRAIN
rows and scored on disjoint TEST rows. The comparison keeps Carnot's current
weights on the same held-out rows, so a positive result cannot come from
changing the evaluation substrate.

Spec: REQ-VERIFY-3668, SCENARIO-VERIFY-3668.
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

from carnot.verify import correlation_aware_weighting_paradox_diagnosis as exp3656
from carnot.verify import dependency_aware_weighting_clean as exp3667
from carnot.verify import weaver_peer_comparison_v3 as exp3644


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results/experiment_3668_dependency_aware_weighting_heldout.json")
UPSTREAM_EXP3667_REL_PATH = exp3667.OUTPUT_REL_PATH
DEFAULT_N_EXAMPLES = exp3644.DEFAULT_N_EXAMPLES
DEFAULT_CORPUS_RANDOM_SEED = exp3644.DEFAULT_RANDOM_SEED
DEFAULT_RANDOM_SEED = 3668
DEFAULT_SPLIT_SEEDS = (3668, 3669, 3670, 3671, 3672)
DEFAULT_TEST_FRACTION = 0.20
DEFAULT_BOOTSTRAP_REPS = 200
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates "
    "(principle: scores held-out FoVer splits; no LLM load)."
)

SUCCESS_GENERALIZES = (
    "complete: dependency_aware_weighting_generalizes_heldout_headline_re_freeze_candidate_for_v337"
)
SUCCESS_OVERFIT = "complete: dependency_aware_weighting_overfit_train_only_heldout_win_evaporates"
BLOCKED_VERDICT = "complete: blocked_dependency_aware_weighting_not_confirmed_upstream"
TERMINAL_VERDICTS = (SUCCESS_GENERALIZES, SUCCESS_OVERFIT, BLOCKED_VERDICT)
OUTCOME_CATEGORIES = ("generalizes_heldout", "overfit_train_only", "blocked")

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "heldout_auroc_dependency_aware",
    "heldout_auroc_carnot",
    "heldout_delta_ci",
    "heldout_delong_p",
    "n_splits",
    "dependency_aware_generalizes_heldout",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": (
        "verifier_ensemble_against_cached_candidates (principle: scores "
        "held-out FoVer splits; no LLM load)."
    ),
    "heldout_auroc_dependency_aware": (
        "Dependency-aware AUROC on the disjoint TEST split (fit on TRAIN only) "
        "-- the generalization number."
    ),
    "heldout_auroc_carnot": (
        "Carnot weighting on the identical TEST split -- the held-out bar."
    ),
    "heldout_delta_ci": (
        "Paired held-out delta + bootstrap CI -- whether the win survives out-of-fit."
    ),
    "heldout_delong_p": "DeLong significance on the held-out split.",
    "n_splits": "Replication count of train/test splits (>=5).",
    "dependency_aware_generalizes_heldout": (
        "BARE bool. True iff the held-out delta favors dependency-aware with CI "
        "excluding 0 -- the de-risk gate before any headline re-freeze. STORE AS "
        "BARE true/false."
    ),
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection.",
    "duration_s": "Plausibility floor.",
}


@dataclass(frozen=True)
class OutcomeClassification:
    """Terminal verdict and bare gate boolean for a held-out outcome."""

    category: str
    terminal_verdict: str
    dependency_aware_generalizes_heldout: bool


@dataclass(frozen=True)
class StratifiedSplit:
    """Deterministic train/test index arrays for one held-out replication."""

    train_indices: np.ndarray
    test_indices: np.ndarray


@dataclass(frozen=True)
class HeldoutSplitResult:
    """Scores and diagnostics from fitting one train split and testing its holdout."""

    seed: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    test_labels: np.ndarray
    dependency_scores: np.ndarray
    carnot_scores: np.ndarray
    dependency_weights: np.ndarray
    learned_edges: list[dict[str, Any]]


def build_artifact(
    repo_root: Path,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    n_examples: int = DEFAULT_N_EXAMPLES,
    random_seed: int = DEFAULT_RANDOM_SEED,
    corpus_random_seed: int = DEFAULT_CORPUS_RANDOM_SEED,
    split_seeds: Sequence[int] = DEFAULT_SPLIT_SEEDS,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_REPS,
) -> dict[str, Any]:
    """Build the Exp 3668 artifact from cached FoVer verifier scores."""

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
        upstream = load_upstream_exp3667_artifact(root)
        labels, scores_by_verifier = exp3644.score_fover_corpus(
            root,
            n_examples=n_examples,
            random_seed=corpus_random_seed,
        )
    except Exception as exc:  # noqa: BLE001 - terminal artifacts fail closed.
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
        started_s=start,
        now_s=now_s,
        random_seed=random_seed,
        corpus_random_seed=corpus_random_seed,
        split_seeds=split_seeds,
        test_fraction=test_fraction,
        n_bootstrap=n_bootstrap,
        upstream_exp3667=upstream,
        preconditions=preconditions,
    )


def probe_preconditions(repo_root: Path, *, n_examples: int) -> list[dict[str, Any]]:
    """Check upstream Exp 3667 success and dependency-aware fitting availability."""

    root = Path(repo_root)
    checks = list(exp3656.probe_preconditions(root, n_examples=n_examples))
    try:
        upstream = load_upstream_exp3667_artifact(root)
        upstream_success = type(upstream.get("dependency_aware_beats_carnot")) is bool and bool(
            upstream["dependency_aware_beats_carnot"]
        )
        upstream_detail = (
            f"dependency_aware_beats_carnot={upstream.get('dependency_aware_beats_carnot')!r}"
        )
    except Exception as exc:  # noqa: BLE001 - precondition detail belongs in artifact.
        upstream_success = False
        upstream_detail = f"{type(exc).__name__}: {exc}"
    checks.append(
        {
            "resource": "exp3667_dependency_aware_beats_carnot",
            "available": upstream_success,
            "detail": upstream_detail,
        }
    )

    implementation_available = all(
        callable(func)
        for func in (
            exp3656.fit_dependency_aware_weights,
            exp3656.learn_dependency_graph,
            exp3656.dependency_aware_crossfit_scores,
        )
    )
    checks.append(
        {
            "resource": "exp3656_dependency_aware_implementation",
            "available": implementation_available,
            "detail": "dependency graph, crossfit scores, and graph-aware weights importable",
        }
    )
    return checks


def load_upstream_exp3667_artifact(repo_root: Path) -> dict[str, Any]:
    """Load the Exp 3667 artifact that authorizes a held-out generalization test."""

    path = Path(repo_root) / UPSTREAM_EXP3667_REL_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    if type(payload.get("dependency_aware_beats_carnot")) is not bool:
        raise ValueError("Exp 3667 artifact lacks bare dependency_aware_beats_carnot bool")
    return payload


def build_artifact_from_scores(
    *,
    labels: Sequence[int],
    scores_by_verifier: Mapping[str, Sequence[float]],
    started_s: float,
    now_s: float | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    corpus_random_seed: int = DEFAULT_CORPUS_RANDOM_SEED,
    split_seeds: Sequence[int] = DEFAULT_SPLIT_SEEDS,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_REPS,
    upstream_exp3667: Mapping[str, Any] | None = None,
    preconditions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble held-out split statistics from already-computed verifier columns."""

    upstream = dict(upstream_exp3667 or {})
    if upstream.get("dependency_aware_beats_carnot") is not True:
        failed = [
            *[dict(item) for item in preconditions or []],
            {
                "resource": "exp3667_dependency_aware_beats_carnot",
                "available": False,
                "detail": f"dependency_aware_beats_carnot={upstream.get('dependency_aware_beats_carnot')!r}",
            },
        ]
        return _blocked_artifact(
            duration_s=_duration(float(started_s), now_s),
            random_seed=random_seed,
            preconditions=failed,
        )

    labels_arr = np.asarray(labels, dtype=np.int64)
    names = list(scores_by_verifier)
    matrix = exp3656.score_matrix(scores_by_verifier, names)
    if matrix.shape[0] != len(labels_arr):
        raise ValueError("labels and verifier scores must have the same length")
    _require_binary_labels(labels_arr)

    split_results = [
        evaluate_heldout_split(
            labels=labels_arr,
            score_matrix=matrix,
            verifier_names=names,
            seed=int(seed),
            test_fraction=test_fraction,
        )
        for seed in split_seeds
    ]
    pooled_labels = np.concatenate([result.test_labels for result in split_results])
    pooled_dependency = np.concatenate([result.dependency_scores for result in split_results])
    pooled_carnot = np.concatenate([result.carnot_scores for result in split_results])

    heldout_dependency_auroc = exp3644.tie_aware_auroc(pooled_labels, pooled_dependency)
    heldout_carnot_auroc = exp3644.tie_aware_auroc(pooled_labels, pooled_carnot)
    delta_ci = exp3667.paired_delta_ci(
        pooled_labels,
        pooled_dependency,
        pooled_carnot,
        seeds=split_seeds,
        n_bootstrap=n_bootstrap,
    )
    delong = exp3667.paired_delong_test(pooled_labels, pooled_dependency, pooled_carnot)
    classification = classify_outcome(
        blocked=False,
        heldout_dependency_aware_auroc=heldout_dependency_auroc,
        heldout_carnot_auroc=heldout_carnot_auroc,
        delta_ci=delta_ci,
    )

    artifact = {
        "artifact": "experiment_3668_dependency_aware_weighting_heldout",
        "schema": "carnot.dependency_aware_weighting_heldout.v1",
        "honest_verdict": classification.terminal_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "heldout_auroc_dependency_aware": _round_metric(heldout_dependency_auroc),
        "heldout_auroc_carnot": _round_metric(heldout_carnot_auroc),
        "heldout_delta_ci": delta_ci,
        "heldout_delong_p": _round_p(float(delong["p_value"])),
        "n_splits": int(len(split_results)),
        "dependency_aware_generalizes_heldout": (
            classification.dependency_aware_generalizes_heldout
        ),
        "random_seed": int(random_seed),
        "corpus_random_seed": int(corpus_random_seed),
        "split_seeds": [int(seed) for seed in split_seeds],
        "test_fraction": _round_metric(float(test_fraction)),
        "n_examples": int(len(labels_arr)),
        "n_bootstrap_per_split_seed": int(n_bootstrap),
        "reproducibility_checksum": reproducibility_checksum(
            labels_arr,
            matrix,
            names,
            random_seed=random_seed,
            corpus_random_seed=corpus_random_seed,
            split_seeds=split_seeds,
            test_fraction=test_fraction,
            upstream_exp3667=upstream,
        ),
        "duration_s": _round_metric(_duration(float(started_s), now_s)),
        "verifier_names": names,
        "split_panel": [
            split_result_to_json(result, verifier_names=names) for result in split_results
        ],
        "heldout_delong": delong,
        "dependency_aware_training_protocol": {
            "method": "stratified_train_fit_graph_sparse_signed_fisher_weights_test_only",
            "learned_on": "TRAIN rows only",
            "evaluated_on": "disjoint TEST rows",
            "dependency_reference": (
                "Learning Dependency Structures for Weak Supervision Models "
                "(arXiv:1903.05844)"
            ),
        },
        "null_discipline": (
            "Carnot-current and dependency-aware AUROCs are computed on identical "
            "held-out TEST rows within every seeded split."
        ),
        "upstream_exp3667": {
            "path": UPSTREAM_EXP3667_REL_PATH.as_posix(),
            "dependency_aware_beats_carnot": bool(upstream.get("dependency_aware_beats_carnot")),
            "honest_verdict": upstream.get("honest_verdict"),
        },
        "acceptance_gate": {
            "condition": (
                "heldout_auroc_dependency_aware present AND heldout_auroc_carnot "
                "present AND n_splits >= 5"
            ),
            "principle": (
                "A generalization verdict requires both weightings evaluated on "
                "the same disjoint held-out splits, replicated -- a single-split "
                "or train-only number cannot de-risk a headline change."
            ),
            "passed": bool(
                heldout_dependency_auroc is not None
                and heldout_carnot_auroc is not None
                and len(split_results) >= 5
            ),
        },
        "preconditions_checked": [dict(item) for item in preconditions or []],
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


def evaluate_heldout_split(
    *,
    labels: np.ndarray,
    score_matrix: np.ndarray,
    verifier_names: Sequence[str],
    seed: int,
    test_fraction: float,
) -> HeldoutSplitResult:
    """Fit dependency-aware weights on TRAIN rows and score disjoint TEST rows."""

    split = stratified_train_test_indices(
        np.asarray(labels, dtype=np.int64),
        random_seed=int(seed),
        test_fraction=float(test_fraction),
    )
    matrix = np.asarray(score_matrix, dtype=np.float64)
    fit = exp3656.fit_dependency_aware_weights(
        labels=labels[split.train_indices],
        score_matrix=matrix[split.train_indices],
        verifier_names=verifier_names,
    )
    carnot_weights = exp3644.carnot_current_weights(verifier_names)
    return HeldoutSplitResult(
        seed=int(seed),
        train_indices=split.train_indices,
        test_indices=split.test_indices,
        test_labels=labels[split.test_indices],
        dependency_scores=matrix[split.test_indices] @ fit.weights,
        carnot_scores=exp3644.ensemble_scores(matrix[split.test_indices], carnot_weights),
        dependency_weights=fit.weights,
        learned_edges=fit.edges,
    )


def stratified_train_test_indices(
    labels: np.ndarray,
    *,
    random_seed: int,
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> StratifiedSplit:
    """Return one deterministic binary-label stratified train/test split."""

    labels_arr = np.asarray(labels, dtype=np.int64)
    _require_binary_labels(labels_arr)
    if not 0.0 < float(test_fraction) < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    rng = np.random.default_rng(int(random_seed))
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for label in (0, 1):
        indices = np.where(labels_arr == label)[0]
        if len(indices) < 2:
            raise ValueError("each binary class needs at least two rows")
        shuffled = np.array(indices, copy=True)
        rng.shuffle(shuffled)
        n_test = int(round(len(shuffled) * float(test_fraction)))
        n_test = min(max(1, n_test), len(shuffled) - 1)
        test_parts.append(shuffled[:n_test])
        train_parts.append(shuffled[n_test:])
    train_indices = np.sort(np.concatenate(train_parts))
    test_indices = np.sort(np.concatenate(test_parts))
    return StratifiedSplit(train_indices=train_indices, test_indices=test_indices)


def split_result_to_json(
    result: HeldoutSplitResult,
    *,
    verifier_names: Sequence[str],
) -> dict[str, Any]:
    """Serialize one split while keeping test-row identity auditable."""

    return {
        "seed": int(result.seed),
        "train_size": int(len(result.train_indices)),
        "test_size": int(len(result.test_indices)),
        "train_indices": [int(index) for index in result.train_indices],
        "test_indices": [int(index) for index in result.test_indices],
        "test_auroc_dependency_aware": _round_metric(
            exp3644.tie_aware_auroc(result.test_labels, result.dependency_scores)
        ),
        "test_auroc_carnot": _round_metric(
            exp3644.tie_aware_auroc(result.test_labels, result.carnot_scores)
        ),
        "test_delta_dependency_minus_carnot": _round_metric(
            exp3644.tie_aware_auroc(result.test_labels, result.dependency_scores)
            - exp3644.tie_aware_auroc(result.test_labels, result.carnot_scores)
        ),
        "dependency_aware_weights": exp3656.weights_to_json(
            verifier_names,
            result.dependency_weights,
        ),
        "learned_edges": [dict(edge) for edge in result.learned_edges],
    }


def classify_outcome(
    *,
    blocked: bool,
    heldout_dependency_aware_auroc: float | None,
    heldout_carnot_auroc: float | None,
    delta_ci: Mapping[str, Any] | None,
) -> OutcomeClassification:
    """Map held-out statistics onto the three allowed honest outcomes."""

    if blocked:
        return OutcomeClassification("blocked", BLOCKED_VERDICT, False)
    ci = list((delta_ci or {}).get("ci95") or [])
    ci_excludes_zero_positive = len(ci) == 2 and float(ci[0]) > 0.0
    generalizes = (
        heldout_dependency_aware_auroc is not None
        and heldout_carnot_auroc is not None
        and float(heldout_dependency_aware_auroc) > float(heldout_carnot_auroc)
        and ci_excludes_zero_positive
    )
    if generalizes:
        return OutcomeClassification("generalizes_heldout", SUCCESS_GENERALIZES, True)
    return OutcomeClassification("overfit_train_only", SUCCESS_OVERFIT, False)


def reproducibility_checksum(
    labels: np.ndarray,
    matrix: np.ndarray,
    verifier_names: Sequence[str],
    *,
    random_seed: int,
    corpus_random_seed: int,
    split_seeds: Sequence[int],
    test_fraction: float,
    upstream_exp3667: Mapping[str, Any],
) -> str:
    """Hash measured inputs, split seeds, verifier order, and upstream gate state."""

    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(labels, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(matrix, dtype=np.float64).tobytes())
    digest.update(json.dumps(list(verifier_names), separators=(",", ":")).encode("utf-8"))
    digest.update(str(int(random_seed)).encode("ascii"))
    digest.update(str(int(corpus_random_seed)).encode("ascii"))
    digest.update(json.dumps([int(seed) for seed in split_seeds]).encode("ascii"))
    digest.update(f"{float(test_fraction):.12f}".encode("ascii"))
    digest.update(
        json.dumps(
            {
                "dependency_aware_beats_carnot": upstream_exp3667.get(
                    "dependency_aware_beats_carnot"
                ),
                "honest_verdict": upstream_exp3667.get("honest_verdict"),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the terminal Exp 3668 schema before writing JSON."""

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
    if type(artifact.get("dependency_aware_generalizes_heldout")) is not bool:
        raise ValueError("dependency_aware_generalizes_heldout must be a bare boolean")
    if not isinstance(artifact.get("duration_s"), int | float):
        raise ValueError("duration_s must be numeric")
    if verdict == BLOCKED_VERDICT:
        return
    if int(artifact.get("n_splits", 0)) < 5:
        raise ValueError("n_splits must be at least 5")
    _validate_auroc_field(artifact, "heldout_auroc_dependency_aware")
    _validate_auroc_field(artifact, "heldout_auroc_carnot")
    _validate_delta_ci(artifact.get("heldout_delta_ci"))
    delong_p = artifact.get("heldout_delong_p")
    if not _is_finite_number(delong_p) or not 0.0 <= float(delong_p) <= 1.0:
        raise ValueError("heldout_delong_p must be finite and in [0, 1]")


def write_artifact(
    repo_root: Path,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and write the Exp 3668 terminal JSON artifact."""

    root = Path(repo_root)
    artifact = build_artifact(root, started_s=started_s, now_s=now_s)
    validate_artifact(artifact)
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
        "artifact": "experiment_3668_dependency_aware_weighting_heldout",
        "schema": "carnot.dependency_aware_weighting_heldout.v1",
        "honest_verdict": BLOCKED_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "heldout_auroc_dependency_aware": None,
        "heldout_auroc_carnot": None,
        "heldout_delta_ci": None,
        "heldout_delong_p": None,
        "n_splits": 0,
        "dependency_aware_generalizes_heldout": False,
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


def _validate_auroc_field(artifact: Mapping[str, Any], field: str) -> None:
    value = artifact.get(field)
    if not _is_finite_number(value) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{field} must be finite and in [0, 1]")


def _validate_delta_ci(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("heldout_delta_ci must be an object")
    point = value.get("point")
    ci = value.get("ci95")
    if not _is_finite_number(point) or not isinstance(ci, list) or len(ci) != 2:
        raise ValueError("heldout_delta_ci must include point and ci95")
    if not all(_is_finite_number(item) for item in ci):
        raise ValueError("heldout_delta_ci bounds must be finite")
    if not float(ci[0]) <= float(point) <= float(ci[1]):
        raise ValueError("heldout_delta_ci must contain its point estimate")


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

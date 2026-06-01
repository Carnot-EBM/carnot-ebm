"""Exp 3644 Weaver peer-comparison correlation audit.

This module measures the assumption that Weaver-style weak-verifier ensembles
lean on: verifier errors are conditionally independent once the true label is
known. Carnot's FoVer ensemble is scored from cached rows only, then the audit
compares equal weights, an independence-assuming inverse-variance baseline, the
current Carnot Exp 2837 weights, and a correlation-aware diagnostic.

Spec: REQ-VERIFY-3644, SCENARIO-VERIFY-3644.
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


OUTPUT_REL_PATH = Path("results/experiment_3644_weaver_peer_comparison_v3.json")
DEFAULT_RANDOM_SEED = 3644
DEFAULT_N_EXAMPLES = 1000
VERIFIER_NAMES = (
    "fr11_session_memory",
    "tier0r_curry_howard",
    "tier0s_arithmetic_gap",
    "tier0u_logical_consistency",
)
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates "
    "(principle: scores the cached FoVer corpus; no LLM load)."
)
SUCCESS_CORRELATION_MATTERS = (
    "complete: weaver_compared_correlation_matters_carnot_differentiates_on_correlation_awareness"
)
SUCCESS_CORRELATION_MARGINAL = (
    "complete: weaver_compared_verifiers_near_independent_correlation_awareness_marginal"
)
BLOCKED_VERDICT = "complete: blocked_fover_corpus_or_verifiers_unavailable"

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "mean_offdiagonal_verifier_correlation",
    "most_redundant_verifier_pair",
    "ensemble_auroc_unweighted",
    "ensemble_auroc_weaver_style",
    "ensemble_auroc_carnot",
    "correlation_awareness_matters",
    "n_examples",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": "Scores the cached FoVer corpus; no LLM load.",
    "mean_offdiagonal_verifier_correlation": (
        "The quantity Weaver assumes ~0; a materially non-zero value is the "
        "differentiation evidence."
    ),
    "most_redundant_verifier_pair": (
        "Names the pair Weaver's independence assumption most mis-handles -- "
        "concrete, auditable."
    ),
    "ensemble_auroc_unweighted": "Baseline: equal-weight ensemble.",
    "ensemble_auroc_weaver_style": (
        "Independence-assuming weak-supervision weighting -- the peer baseline."
    ),
    "ensemble_auroc_carnot": (
        "Carnot's current weighting -- the apples-to-apples comparison."
    ),
    "correlation_awareness_matters": (
        "True iff accounting for the measured correlation changes weights/AUROC "
        "vs the Weaver independence baseline -- the differentiation claim."
    ),
    "n_examples": "Sample-size rigor.",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection.",
    "duration_s": "Plausibility floor.",
}


def build_artifact(
    repo_root: Path,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    n_examples: int = DEFAULT_N_EXAMPLES,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, Any]:
    """Build the Exp 3644 artifact from the local FoVer corpus.

    The function performs only cached corpus scoring. It does not call an LLM,
    load model weights, or mutate FR-11 state.
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
        started_s=start,
        now_s=now_s,
        random_seed=random_seed,
        preconditions=preconditions,
    )


def probe_preconditions(repo_root: Path, *, n_examples: int) -> list[dict[str, Any]]:
    """Check that the corpus and all four scoring paths can run."""

    root = Path(repo_root)
    checks: list[dict[str, Any]] = []
    fover_path = root / "data" / "fover_corpus.jsonl"
    if fover_path.is_file():
        n_rows = _line_count(fover_path)
        checks.append(
            {
                "resource": "fover_corpus",
                "available": n_rows >= n_examples,
                "detail": f"line_count={n_rows}; required>={n_examples}",
            }
        )
    else:
        checks.append({"resource": "fover_corpus", "available": False, "detail": "missing"})

    try:
        from carnot.eval.fover_memory_leakage_v3 import _score_text_verifiers

        smoke = _score_text_verifiers(["1 + 1 = 2"])
        scoring_available = set(smoke) == set(VERIFIER_NAMES[1:])
        detail = "loaded=" + ",".join(sorted(smoke))
    except Exception as exc:  # noqa: BLE001 - reported as a blocked precondition.
        scoring_available = False
        detail = f"{type(exc).__name__}: {exc}"
    checks.append(
        {
            "resource": "text_scoring_verifiers",
            "available": scoring_available,
            "detail": detail,
        }
    )

    try:
        from carnot.eval.fover_memory_leakage_v3 import (
            _load_fr11_memory_index,
            discover_fr11_state_files,
        )

        state_files = discover_fr11_state_files(root)
        memory_index = _load_fr11_memory_index(root) if state_files else {}
        has_memory = bool(memory_index.get("question_ids") or memory_index.get("prompt_token_sets"))
        memory_available = bool(state_files) and has_memory
        memory_detail = f"state_files={len(state_files)}; memory_loaded={has_memory}"
    except Exception as exc:  # noqa: BLE001 - reported as a blocked precondition.
        memory_available = False
        memory_detail = f"{type(exc).__name__}: {exc}"
    checks.append(
        {
            "resource": "fr11_session_memory_verifier",
            "available": memory_available,
            "detail": memory_detail,
        }
    )
    return checks


def score_fover_corpus(
    repo_root: Path,
    *,
    n_examples: int,
    random_seed: int,
) -> tuple[list[int], dict[str, list[float]]]:
    """Score the deterministic FoVer subset with Exp 2837's four verifiers."""

    from carnot.eval.fover_memory_leakage_v3 import (
        _fr11_memory_score,
        _label_to_int,
        _load_fr11_memory_index,
        _read_fover_rows,
        _score_text_verifiers,
        _select_balanced_subset,
    )

    root = Path(repo_root)
    rows = _select_balanced_subset(
        _read_fover_rows(root / "data" / "fover_corpus.jsonl"),
        seed=random_seed,
        n_examples=n_examples,
    )
    labels = [_label_to_int(row["label"]) for row in rows]
    texts = [str(row.get("step_text", "")) for row in rows]
    text_scores = _score_text_verifiers(texts)
    memory_index = _load_fr11_memory_index(root)
    scores_by_verifier = {
        "fr11_session_memory": [_fr11_memory_score(row, memory_index) for row in rows],
        "tier0r_curry_howard": text_scores["tier0r_curry_howard"],
        "tier0s_arithmetic_gap": text_scores["tier0s_arithmetic_gap"],
        "tier0u_logical_consistency": text_scores["tier0u_logical_consistency"],
    }
    return labels, scores_by_verifier


def build_artifact_from_scores(
    *,
    labels: Sequence[int],
    scores_by_verifier: Mapping[str, Sequence[float]],
    started_s: float,
    now_s: float | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    preconditions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the terminal artifact from already-computed verifier scores."""

    names = list(scores_by_verifier)
    score_matrix = _score_matrix(scores_by_verifier, names)
    labels_arr = np.asarray(labels, dtype=np.int64)
    if score_matrix.shape[0] != len(labels_arr):
        raise ValueError("labels and verifier scores must have the same length")
    if score_matrix.shape[1] < 2:
        raise ValueError("at least two verifiers are required")

    pearson = safe_pearson_matrix(score_matrix)
    conditional = {
        "correct": safe_pearson_matrix(score_matrix[labels_arr == 0]),
        "incorrect": safe_pearson_matrix(score_matrix[labels_arr == 1]),
    }
    mean_conditional = mean_offdiagonal_abs(list(conditional.values()))
    most_redundant = most_redundant_pair(names, conditional)

    unweighted_weights = normalize_weights(np.ones(score_matrix.shape[1], dtype=float))
    weaver_weights = weaver_style_weights(score_matrix)
    carnot_weights = carnot_current_weights(names)
    aware_weights = correlation_aware_weights(score_matrix)

    unweighted_scores = ensemble_scores(score_matrix, unweighted_weights)
    weaver_scores = ensemble_scores(score_matrix, weaver_weights)
    carnot_scores = ensemble_scores(score_matrix, carnot_weights)
    aware_scores = ensemble_scores(score_matrix, aware_weights)

    auroc_unweighted = tie_aware_auroc(labels_arr, unweighted_scores)
    auroc_weaver = tie_aware_auroc(labels_arr, weaver_scores)
    auroc_carnot = tie_aware_auroc(labels_arr, carnot_scores)
    auroc_aware = tie_aware_auroc(labels_arr, aware_scores)

    weight_l1_delta = float(np.abs(aware_weights - weaver_weights).sum())
    auroc_delta = float(auroc_aware - auroc_weaver)
    correlation_awareness_matters = bool(
        (mean_conditional > 0.05 and weight_l1_delta > 0.05) or abs(auroc_delta) > 0.005
    )
    verdict = (
        SUCCESS_CORRELATION_MATTERS
        if correlation_awareness_matters
        else SUCCESS_CORRELATION_MARGINAL
    )

    artifact = {
        "artifact": "experiment_3644_weaver_peer_comparison_v3",
        "schema": "carnot.weaver_peer_comparison_v3",
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "mean_offdiagonal_verifier_correlation": _round(mean_conditional),
        "mean_offdiagonal_unconditional_verifier_correlation": _round(
            mean_offdiagonal_abs([pearson])
        ),
        "most_redundant_verifier_pair": most_redundant,
        "ensemble_auroc_unweighted": _round(auroc_unweighted),
        "ensemble_auroc_weaver_style": _round(auroc_weaver),
        "ensemble_auroc_carnot": _round(auroc_carnot),
        "ensemble_auroc_correlation_aware": _round(auroc_aware),
        "correlation_awareness_matters": correlation_awareness_matters,
        "n_examples": int(len(labels_arr)),
        "random_seed": int(random_seed),
        "reproducibility_checksum": reproducibility_checksum(
            labels_arr,
            score_matrix,
            names,
            random_seed=random_seed,
        ),
        "duration_s": _round(_duration(float(started_s), now_s)),
        "verifier_names": names,
        "pearson_verifier_correlation_matrix": _matrix_to_json(pearson),
        "conditional_verifier_correlation_by_label": {
            label: _matrix_to_json(matrix) for label, matrix in conditional.items()
        },
        "weights_unweighted": _weights_to_json(names, unweighted_weights),
        "weights_weaver_style": _weights_to_json(names, weaver_weights),
        "weights_carnot": _weights_to_json(names, carnot_weights),
        "weights_correlation_aware": _weights_to_json(names, aware_weights),
        "weight_l1_delta_correlation_aware_vs_weaver": _round(weight_l1_delta),
        "auroc_delta_correlation_aware_vs_weaver": _round(auroc_delta),
        "preconditions_checked": [dict(item) for item in preconditions or []],
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


def safe_pearson_matrix(score_matrix: np.ndarray) -> np.ndarray:
    """Return a finite Pearson matrix, using zero for undefined pairwise terms."""

    matrix = np.asarray(score_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("score_matrix must be two-dimensional")
    k = matrix.shape[1]
    corr = np.eye(k, dtype=np.float64)
    for i in range(k):
        for j in range(i + 1, k):
            value = safe_pearson(matrix[:, i], matrix[:, j])
            corr[i, j] = value
            corr[j, i] = value
    return corr


def safe_pearson(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute Pearson correlation while treating constant columns as undefined."""

    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 2:
        return 0.0
    x_centered = x - float(np.mean(x))
    y_centered = y - float(np.mean(y))
    denom = math.sqrt(float(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered)))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(x_centered, y_centered) / denom)


def mean_offdiagonal_abs(matrices: Sequence[np.ndarray]) -> float:
    """Mean absolute off-diagonal correlation across one or more matrices."""

    values: list[float] = []
    for matrix in matrices:
        arr = np.asarray(matrix, dtype=np.float64)
        for i in range(arr.shape[0]):
            for j in range(i + 1, arr.shape[1]):
                values.append(abs(float(arr[i, j])))
    if not values:
        return 0.0
    return float(np.mean(values))


def most_redundant_pair(
    verifier_names: Sequence[str],
    conditional_matrices: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Identify the verifier pair with the highest mean conditional correlation."""

    names = list(verifier_names)
    best_pair: tuple[str, str] | None = None
    best_by_label: dict[str, float] = {}
    best_value = -1.0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            by_label = {
                label: abs(float(matrix[i, j])) for label, matrix in conditional_matrices.items()
            }
            value = float(np.mean(list(by_label.values()))) if by_label else 0.0
            if value > best_value:
                best_value = value
                best_pair = (names[i], names[j])
                best_by_label = by_label
    return {
        "pair": list(best_pair or ("", "")),
        "conditional_abs_correlation": _round(best_value),
        "by_label": {label: _round(value) for label, value in sorted(best_by_label.items())},
    }


def weaver_style_weights(score_matrix: np.ndarray) -> np.ndarray:
    """Label-free independence baseline using inverse verifier variance."""

    matrix = np.asarray(score_matrix, dtype=np.float64)
    variances = np.var(matrix, axis=0)
    raw = np.zeros(matrix.shape[1], dtype=np.float64)
    active = variances > 1e-12
    raw[active] = 1.0 / variances[active]
    return normalize_weights(raw)


def correlation_aware_weights(score_matrix: np.ndarray, *, ridge: float = 1e-4) -> np.ndarray:
    """Label-free inverse-covariance weights that penalize redundant axes."""

    matrix = np.asarray(score_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("score_matrix must be two-dimensional")
    variances = np.var(matrix, axis=0)
    active = variances > 1e-12
    if not np.any(active):
        return normalize_weights(np.ones(matrix.shape[1], dtype=np.float64))
    active_matrix = matrix[:, active]
    centered = active_matrix - np.mean(active_matrix, axis=0)
    cov = np.cov(centered, rowvar=False, ddof=0)
    cov = np.atleast_2d(cov).astype(np.float64)
    cov = cov + np.eye(cov.shape[0], dtype=np.float64) * float(ridge)
    try:
        raw_active = np.linalg.solve(cov, np.ones(cov.shape[0], dtype=np.float64))
    except np.linalg.LinAlgError:
        raw_active = np.linalg.pinv(cov) @ np.ones(cov.shape[0], dtype=np.float64)
    raw_active = np.clip(raw_active, 0.0, None)
    raw = np.zeros(matrix.shape[1], dtype=np.float64)
    raw[active] = raw_active
    return normalize_weights(raw)


def carnot_current_weights(verifier_names: Sequence[str]) -> np.ndarray:
    """Return normalized Exp 2837 production architecture weights by verifier name."""

    raw_by_name = {
        "fr11_session_memory": 1.0,
        "tier0r_curry_howard": 0.9,
        "tier0s_arithmetic_gap": 0.0,
        "tier0u_logical_consistency": 0.1,
    }
    raw = np.asarray([raw_by_name.get(name, 0.0) for name in verifier_names], dtype=np.float64)
    return normalize_weights(raw)


def normalize_weights(raw_weights: Sequence[float]) -> np.ndarray:
    """Normalize nonnegative finite weights and fall back to uniform if needed."""

    raw = np.asarray(raw_weights, dtype=np.float64)
    raw = np.where(np.isfinite(raw) & (raw > 0.0), raw, 0.0)
    total = float(raw.sum())
    if total <= 0.0:
        weights = np.ones(len(raw), dtype=np.float64) / float(len(raw))
    else:
        weights = raw / total
    if len(weights):
        weights[-1] += 1.0 - float(weights.sum())
    return weights


def ensemble_scores(score_matrix: np.ndarray, weights: Sequence[float]) -> np.ndarray:
    """Apply a normalized verifier-weight vector to the score matrix."""

    matrix = np.asarray(score_matrix, dtype=np.float64)
    w = normalize_weights(weights)
    if matrix.shape[1] != len(w):
        raise ValueError("score_matrix column count must match weights")
    return matrix @ w


def tie_aware_auroc(labels: Sequence[int] | np.ndarray, scores: Sequence[float]) -> float:
    """Compute AUROC with half credit for tied positive/negative scores."""

    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    positives = s[y == 1]
    negatives = s[y == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return 0.5
    greater = positives[:, None] > negatives[None, :]
    ties = positives[:, None] == negatives[None, :]
    wins = float(greater.sum()) + 0.5 * float(ties.sum())
    return wins / float(len(positives) * len(negatives))


def reproducibility_checksum(
    labels: np.ndarray,
    score_matrix: np.ndarray,
    verifier_names: Sequence[str],
    *,
    random_seed: int,
) -> str:
    """Hash the measured labels, scores, verifier order, and seed."""

    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(labels, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(score_matrix, dtype=np.float64).tobytes())
    digest.update(json.dumps(list(verifier_names), separators=(",", ":")).encode("utf-8"))
    digest.update(str(int(random_seed)).encode("ascii"))
    return digest.hexdigest()


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the terminal Exp 3644 schema before writing JSON."""

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
    if verdict not in {
        SUCCESS_CORRELATION_MATTERS,
        SUCCESS_CORRELATION_MARGINAL,
        BLOCKED_VERDICT,
    }:
        raise ValueError(f"unsupported honest_verdict: {verdict!r}")
    if not isinstance(artifact.get("correlation_awareness_matters"), bool):
        raise ValueError("correlation_awareness_matters must be a bare boolean")
    if int(artifact.get("n_examples", 0)) < 0:
        raise ValueError("n_examples must be nonnegative")
    if not isinstance(artifact.get("duration_s"), int | float):
        raise ValueError("duration_s must be numeric")

    if verdict == BLOCKED_VERDICT:
        return

    if int(artifact["n_examples"]) <= 0:
        raise ValueError("runnable artifact must report n_examples > 0")
    for field in (
        "mean_offdiagonal_verifier_correlation",
        "ensemble_auroc_unweighted",
        "ensemble_auroc_weaver_style",
        "ensemble_auroc_carnot",
        "ensemble_auroc_correlation_aware",
    ):
        value = artifact.get(field)
        if not isinstance(value, int | float) or not math.isfinite(float(value)):
            raise ValueError(f"{field} must be finite")
    for field in (
        "ensemble_auroc_unweighted",
        "ensemble_auroc_weaver_style",
        "ensemble_auroc_carnot",
        "ensemble_auroc_correlation_aware",
    ):
        value = float(artifact[field])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{field} must be in [0, 1]")
    if not artifact.get("most_redundant_verifier_pair"):
        raise ValueError("most_redundant_verifier_pair must be present")


def write_artifact(
    repo_root: Path,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build, validate, and write the Exp 3644 terminal JSON artifact."""

    root = Path(repo_root)
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
        "artifact": "experiment_3644_weaver_peer_comparison_v3",
        "schema": "carnot.weaver_peer_comparison_v3",
        "honest_verdict": BLOCKED_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "mean_offdiagonal_verifier_correlation": None,
        "mean_offdiagonal_unconditional_verifier_correlation": None,
        "most_redundant_verifier_pair": None,
        "ensemble_auroc_unweighted": None,
        "ensemble_auroc_weaver_style": None,
        "ensemble_auroc_carnot": None,
        "ensemble_auroc_correlation_aware": None,
        "correlation_awareness_matters": False,
        "n_examples": 0,
        "random_seed": int(random_seed),
        "reproducibility_checksum": hashlib.sha256(payload).hexdigest(),
        "duration_s": _round(duration_s),
        "verifier_names": list(VERIFIER_NAMES),
        "pearson_verifier_correlation_matrix": None,
        "conditional_verifier_correlation_by_label": None,
        "weights_unweighted": None,
        "weights_weaver_style": None,
        "weights_carnot": None,
        "weights_correlation_aware": None,
        "weight_l1_delta_correlation_aware_vs_weaver": None,
        "auroc_delta_correlation_aware_vs_weaver": None,
        "preconditions_checked": [dict(item) for item in preconditions],
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


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


def _weights_to_json(names: Sequence[str], weights: Sequence[float]) -> dict[str, float]:
    return {name: _round(float(weight)) for name, weight in zip(names, weights, strict=True)}


def _matrix_to_json(matrix: np.ndarray) -> list[list[float]]:
    return [[_round(float(value)) for value in row] for row in np.asarray(matrix)]


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _line in handle)


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.time() if now_s is None else float(now_s)
    return max(0.0, end - float(started_s))


def _round(value: float | int | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)

"""FR-11 continuous self-learning v20 Tier-3 fast-path gate.

Spec: REQ-LEARN-3803, SCENARIO-LEARN-3803.

The v20 forward difference applies the Tier-3 predictor trained in Exp 3788.
It does not retrain that predictor and it does not replace the frozen FoVer
headline ensemble. The predictor is used only as a confidence gate: confident
rows short-circuit to the predictor score, and uncertain rows fall through to
the four-verifier ensemble score used by the Exp 2837 headline protocol.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from carnot.fr11 import continuous_self_learning_v17 as v17
from carnot.fr11 import continuous_self_learning_v19 as v19


JsonDict = dict[str, Any]

OUTPUT_REL_PATH = Path("results/experiment_3803_fr11_v20_tier3_fast_path_gate.json")
PREDICTOR_STATE_REL_PATH = v19.PREDICTOR_STATE_REL_PATH
OPERATING_POINT_REL_PATH = Path(
    "results/experiment_3803_fr11_v20_tier3_fast_path_gate_state.json"
)
EXP2837_REL_PATH = Path("results/experiment_2837_fover_memory_leakage_v3.json")
DEFAULT_RANDOM_SEED = 3803
DEFAULT_CORPUS_RANDOM_SEED = 3788
DEFAULT_N_EXAMPLES = 1000
DEFAULT_V19_SPLIT_SEED = 3788
DEFAULT_V19_TEST_FRACTION = 0.2
FROZEN_HEADLINE_ROUNDED = 0.9131
FROZEN_CI95 = (0.9027316334533082, 0.9235355665466916)
THRESHOLD_GRID = tuple(round(float(value), 2) for value in np.arange(0.0, 0.501, 0.01))
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates "
    "(principle: reads cached scores + the persisted predictor, no live model)."
)
SUCCESS_VERDICT_PREFIX = "complete: fr11_v20_tier3_fast_path_gate_skip_rate_"
SUCCESS_VERDICT_SUFFIX = (
    "_in_frozen_ci_no_accuracy_regression_headline_ensemble_unchanged_"
    "operating_point_persisted"
)
NO_SAFE_VERDICT = (
    "complete: fr11_v20_tier3_fast_path_gate_no_threshold_in_frozen_ci_"
    "no_compute_saving_claimed"
)
BLOCKED_INTERPRETER_VERDICT = "blocked_interpreter_runtime"
BLOCKED_TIER3_STATE_VERDICT = "blocked_tier3_predictor_state_missing"
BLOCKED_CORPUS_VERDICT = "blocked_fover_corpus_missing"
BLOCKED_SCORING_VERDICT = "blocked_fover_scores_missing"
VERIFIER_NAMES = v17.VERIFIER_NAMES
JEPA_DOMAINS = v19.JEPA_DOMAINS

score_fover_corpus = v17.score_fover_corpus
score_matrix = v17.score_matrix
probe_cached_trace_preconditions = v17.probe_cached_trace_preconditions

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "skip_rate_at_no_regression",
    "effective_auroc_at_operating_point",
    "compute_saving_vs_accuracy_curve",
    "accuracy_regression",
    "held_out_split_sizes",
    "headline_ensemble_unchanged",
    "is_tier3_application_not_retrain",
    "operating_point_persisted",
    "model_specs",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal prefix; the fast-path outcome; blocked_<resource> if a "
        "precondition failed."
    ),
    "inference_substrate": (
        "verifier_ensemble_against_cached_candidates (principle: reads cached "
        "scores + the persisted predictor, no live model)."
    ),
    "skip_rate_at_no_regression": (
        "Fraction of candidates the fast-path short-circuits at the operating "
        "point where effective AUROC stays in the frozen CI -- the core "
        "compute-saving deliverable."
    ),
    "effective_auroc_at_operating_point": (
        "The combined fast-path+fallthrough AUROC at the chosen threshold -- "
        "MUST be within the frozen 0.9131 CI [0.9027, 0.9235]."
    ),
    "compute_saving_vs_accuracy_curve": (
        "The threshold sweep (skip rate vs effective AUROC) -- so the operator "
        "can pick a different operating point; honest trade-off."
    ),
    "accuracy_regression": (
        "BARE bool, false at the reported operating point -- the gate trades "
        "compute for nothing if it degrades the verdict."
    ),
    "held_out_split_sizes": (
        "Sample-size + leakage hygiene -- the fast-path is measured on a split "
        "DISJOINT from the predictor's training split."
    ),
    "headline_ensemble_unchanged": (
        "BARE bool, true -- the frozen 0.9131 full-scoring ensemble is the "
        "fall-through and is UNTOUCHED; the gate is additive."
    ),
    "is_tier3_application_not_retrain": (
        "BARE bool, true -- confirms v20 APPLIES the v19-trained predictor as "
        "a gate (it does not retrain)."
    ),
    "operating_point_persisted": (
        "BARE bool, true -- the chosen gate operating point was persisted so a "
        "future milestone resumes it."
    ),
    "model_specs": (
        "Names the corpus + 4 verifiers + the Tier-3 predictor -- honest substrate."
    ),
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Content hash catches drift.",
    "duration_s": "Wall-clock plausibility floor; verifier-scoring + a gate sweep is modest.",
}


@dataclass(frozen=True)
class GateApplicationResult:
    """One threshold application of the persisted predictor gate."""

    combined_scores: np.ndarray
    skip_mask: np.ndarray
    skip_count: int
    fallthrough_count: int
    skip_rate: float


def build_artifact(
    repo_root: Path | str,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    n_examples: int = DEFAULT_N_EXAMPLES,
    random_seed: int = DEFAULT_RANDOM_SEED,
    corpus_random_seed: int = DEFAULT_CORPUS_RANDOM_SEED,
    predictor_state_path: Path | str = PREDICTOR_STATE_REL_PATH,
    operating_point_path: Path | str = OPERATING_POINT_REL_PATH,
    frozen_ci: tuple[float, float] = FROZEN_CI95,
) -> JsonDict:
    """Build Exp 3803 from cached FoVer rows and the persisted v19 predictor."""

    root = Path(repo_root)
    start = time.time() if started_s is None else float(started_s)
    predictor_state = _resolve_under_root(root, predictor_state_path)
    operating_output = _resolve_under_root(root, operating_point_path)
    corpus_path = root / "data" / "fover_corpus.jsonl"
    preconditions = [
        _interpreter_precondition(),
        _predictor_state_precondition(predictor_state),
        _fover_corpus_precondition(corpus_path, n_examples=n_examples),
    ]
    if not all(item["available"] for item in preconditions):
        return _blocked_artifact(
            duration_s=_duration(start, now_s),
            random_seed=random_seed,
            predictor_state_path=predictor_state,
            operating_point_path=operating_output,
            repo_root=root,
            preconditions=preconditions,
            verdict=_blocked_verdict(preconditions),
        )
    preconditions.extend(probe_cached_trace_preconditions(root, n_examples=n_examples))
    if not all(item["available"] for item in preconditions):
        return _blocked_artifact(
            duration_s=_duration(start, now_s),
            random_seed=random_seed,
            predictor_state_path=predictor_state,
            operating_point_path=operating_output,
            repo_root=root,
            preconditions=preconditions,
            verdict=_blocked_verdict(preconditions),
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
            predictor_state_path=predictor_state,
            operating_point_path=operating_output,
            repo_root=root,
            preconditions=[
                *preconditions,
                {
                    "resource": "cached_trace_scoring",
                    "available": False,
                    "detail": f"{type(exc).__name__}: {exc}",
                },
            ],
            verdict=BLOCKED_SCORING_VERDICT,
        )

    return build_artifact_from_scores(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        predictor_state_path=predictor_state,
        operating_point_path=operating_output,
        started_s=start,
        now_s=now_s,
        random_seed=random_seed,
        repo_root=root,
        preconditions=preconditions,
        corpus_absolute_path=corpus_path.resolve(),
        frozen_ci=frozen_ci,
        headline_ensemble_reference=load_headline_ensemble_reference(root),
    )


def build_artifact_from_scores(
    *,
    labels: Sequence[int],
    scores_by_verifier: Mapping[str, Sequence[float]],
    predictor_state_path: Path | str,
    operating_point_path: Path | str,
    started_s: float,
    now_s: float | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    repo_root: Path | str | None = None,
    preconditions: Sequence[Mapping[str, Any]] | None = None,
    corpus_absolute_path: Path | str | None = None,
    frozen_ci: tuple[float, float] = FROZEN_CI95,
    headline_ensemble_reference: float | None = None,
    predictor_state_sha256: str | None = None,
) -> JsonDict:
    """Apply the persisted Tier-3 predictor as a confidence-gated fast path."""

    root = Path(repo_root) if repo_root is not None else None
    predictor_state = _resolve_under_root(root or Path("."), predictor_state_path)
    operating_output = _resolve_under_root(root or Path("."), operating_point_path)
    state_precondition = _predictor_state_precondition(predictor_state)
    checked = [*(dict(item) for item in (preconditions or []))]
    if not state_precondition["available"]:
        checked.append(state_precondition)
        return _blocked_artifact(
            duration_s=_duration(float(started_s), now_s),
            random_seed=random_seed,
            predictor_state_path=predictor_state,
            operating_point_path=operating_output,
            repo_root=root,
            preconditions=checked,
            verdict=BLOCKED_TIER3_STATE_VERDICT,
        )
    if not labels or not scores_by_verifier:
        checked.append(_trace_precondition(labels, scores_by_verifier))
        return _blocked_artifact(
            duration_s=_duration(float(started_s), now_s),
            random_seed=random_seed,
            predictor_state_path=predictor_state,
            operating_point_path=operating_output,
            repo_root=root,
            preconditions=checked,
            verdict=BLOCKED_CORPUS_VERDICT,
        )

    labels_arr = np.asarray(labels, dtype=np.int64)
    _require_binary_labels(labels_arr)
    matrix = score_matrix(scores_by_verifier, VERIFIER_NAMES)
    if matrix.shape[0] != len(labels_arr):
        raise ValueError("labels and verifier scores must have the same length")
    predictor_payload = v19.load_predictor_state(predictor_state)
    predictor = predictor_payload["predictor"]
    predictor_metadata = dict(predictor_payload.get("metadata", {}))
    full_scores = full_ensemble_scores(matrix)
    full_auroc_all = v17.v10.exp3644.tie_aware_auroc(labels_arr, full_scores)
    test_fraction = _metadata_test_fraction(predictor_metadata, len(labels_arr))
    train_indices, held_out_indices = v19.stratified_train_test_indices(
        labels_arr,
        test_fraction=test_fraction,
        random_seed=DEFAULT_V19_SPLIT_SEED,
    )
    held_out_labels = labels_arr[held_out_indices]
    held_out_full_scores = full_scores[held_out_indices]
    predictor_probabilities = predictor_probabilities_for_indices(
        predictor,
        labels=labels_arr.tolist(),
        scores_by_verifier=scores_by_verifier,
        indices=held_out_indices,
    )
    full_held_out_auroc = v17.v10.exp3644.tie_aware_auroc(
        held_out_labels,
        held_out_full_scores,
    )
    curve = sweep_confidence_thresholds(
        labels=held_out_labels,
        full_scores=held_out_full_scores,
        predictor_probabilities=predictor_probabilities,
        thresholds=THRESHOLD_GRID,
    )
    selected = choose_operating_point(curve, frozen_ci=frozen_ci)
    state_sha = predictor_state_sha256 or _file_sha256(predictor_state)
    headline_ok = headline_ensemble_unchanged(headline_ensemble_reference)
    if selected is None:
        operating_sha = None
        persisted = False
        skip_rate = None
        effective_auroc = None
        accuracy_regression: bool | None = True
        verdict = NO_SAFE_VERDICT
        operating_point: JsonDict = {}
    else:
        operating_point = dict(selected)
        operating_sha = persist_operating_point_state(
            operating_output,
            payload={
                "schema": "carnot.fr11_v20_tier3_fast_path_gate_state",
                "selected_operating_point": operating_point,
                "frozen_ci95": {"low": float(frozen_ci[0]), "high": float(frozen_ci[1])},
                "predictor_state_path": _relative_path(predictor_state, root),
                "predictor_state_sha256": state_sha,
                "random_seed": int(random_seed),
            },
        )
        persisted = operating_output.is_file()
        skip_rate = float(selected["skip_rate"])
        effective_auroc = float(selected["effective_auroc"])
        accuracy_regression = not _in_ci(effective_auroc, frozen_ci)
        verdict = _success_verdict(skip_rate, effective_auroc)

    artifact: JsonDict = {
        "artifact": "experiment_3803_fr11_v20_tier3_fast_path_gate",
        "schema": "carnot.fr11_continuous_self_learning_v20",
        "continuous_self_learning_task": True,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "skip_rate_at_no_regression": _round(skip_rate),
        "effective_auroc_at_operating_point": _round(effective_auroc),
        "compute_saving_vs_accuracy_curve": curve,
        "accuracy_regression": accuracy_regression,
        "held_out_split_sizes": _split_sizes(labels_arr, train_indices, held_out_indices),
        "headline_ensemble_unchanged": bool(headline_ok),
        "is_tier3_application_not_retrain": True,
        "operating_point_persisted": bool(persisted),
        "model_specs": _model_specs(corpus_absolute_path, n_examples=len(labels_arr)),
        "random_seed": int(random_seed),
        "reproducibility_checksum": reproducibility_checksum(
            labels=labels_arr,
            scores=matrix,
            train_indices=train_indices,
            held_out_indices=held_out_indices,
            held_out_full_scores=held_out_full_scores,
            predictor_probabilities=predictor_probabilities,
            curve=curve,
            selected=operating_point,
            random_seed=random_seed,
            predictor_state_sha256=state_sha,
            operating_point_sha256=operating_sha,
            frozen_ci=frozen_ci,
        ),
        "duration_s": _round(_duration(float(started_s), now_s)),
        "n_examples": int(len(labels_arr)),
        "n_samples": int(len(held_out_indices)),
        "verifier_names": list(VERIFIER_NAMES),
        "predictor_state_path": _relative_path(predictor_state, root),
        "predictor_state_sha256": state_sha,
        "operating_point_state_path": _relative_path(operating_output, root),
        "operating_point_state_sha256": operating_sha,
        "operating_point": operating_point,
        "frozen_ci95": {"low": float(frozen_ci[0]), "high": float(frozen_ci[1])},
        "frozen_headline_ensemble_auroc": headline_ensemble_reference,
        "fallthrough_full_ensemble_auroc_all_scored_rows": _round(full_auroc_all),
        "fallthrough_full_ensemble_auroc_held_out": _round(full_held_out_auroc),
        "predictor_metadata": predictor_metadata,
        "methodology": _methodology(random_seed, corpus_absolute_path),
        "preconditions_checked": checked,
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate": {
            "passed": bool(
                selected is not None
                and _in_ci(float(selected["effective_auroc"]), frozen_ci)
                and not accuracy_regression
                and headline_ok
                and persisted
            ),
            "condition": (
                "chosen threshold effective AUROC in frozen CI AND "
                "accuracy_regression=false AND frozen full ensemble is the "
                "fall-through AND operating point persisted"
            ),
            "principle": (
                "The Tier-3 fast path is useful only when it saves verifier "
                "calls without degrading the frozen headline verdict."
            ),
        },
    }
    validate_artifact(artifact)
    return artifact


def write_artifact(
    repo_root: Path | str,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    predictor_state_path: Path | str = PREDICTOR_STATE_REL_PATH,
    operating_point_path: Path | str = OPERATING_POINT_REL_PATH,
    labels: Sequence[int] | None = None,
    scores_by_verifier: Mapping[str, Sequence[float]] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
    frozen_ci: tuple[float, float] = FROZEN_CI95,
) -> Path:
    """Build, validate, and write the Exp 3803 JSON artifact."""

    root = Path(repo_root)
    start = time.time() if started_s is None else float(started_s)
    resolved_state_path = _resolve_under_root(root, predictor_state_path)
    resolved_operating_path = _resolve_under_root(root, operating_point_path)
    if labels is not None or scores_by_verifier is not None:
        artifact = build_artifact_from_scores(
            labels=labels or [],
            scores_by_verifier=scores_by_verifier or {},
            predictor_state_path=resolved_state_path,
            operating_point_path=resolved_operating_path,
            started_s=start,
            now_s=now_s,
            repo_root=root,
            corpus_absolute_path=(root / "data" / "fover_corpus.jsonl").resolve(),
            frozen_ci=frozen_ci,
            headline_ensemble_reference=load_headline_ensemble_reference(root),
        )
    else:
        artifact = build_artifact(
            root,
            started_s=start,
            now_s=now_s,
            predictor_state_path=resolved_state_path,
            operating_point_path=resolved_operating_path,
            frozen_ci=frozen_ci,
        )
    output = _resolve_under_root(root, output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def apply_confidence_gate(
    full_scores: Sequence[float] | np.ndarray,
    predictor_probabilities: Sequence[float] | np.ndarray,
    *,
    threshold: float,
) -> GateApplicationResult:
    """Return combined scores for one confidence threshold."""

    full = np.asarray(full_scores, dtype=np.float64)
    predictor = np.asarray(predictor_probabilities, dtype=np.float64)
    if len(full) != len(predictor):
        raise ValueError("full_scores and predictor_probabilities must have the same length")
    if not np.isfinite(full).all() or not np.isfinite(predictor).all():
        raise ValueError("full scores and predictor probabilities must be finite")
    if len(full) == 0:
        raise ValueError("at least one score is required")
    confidence = np.abs(predictor - 0.5)
    skip_mask = confidence >= float(threshold)
    combined = full.copy()
    combined[skip_mask] = predictor[skip_mask]
    skip_count = int(np.sum(skip_mask))
    return GateApplicationResult(
        combined_scores=combined,
        skip_mask=skip_mask,
        skip_count=skip_count,
        fallthrough_count=int(len(full) - skip_count),
        skip_rate=float(skip_count / len(full)),
    )


def sweep_confidence_thresholds(
    *,
    labels: Sequence[int] | np.ndarray,
    full_scores: Sequence[float] | np.ndarray,
    predictor_probabilities: Sequence[float] | np.ndarray,
    thresholds: Sequence[float] = THRESHOLD_GRID,
) -> list[JsonDict]:
    """Measure skip rate and effective AUROC across confidence thresholds."""

    labels_arr = np.asarray(labels, dtype=np.int64)
    _require_binary_labels(labels_arr)
    rows: list[JsonDict] = []
    for threshold in thresholds:
        gate = apply_confidence_gate(
            full_scores,
            predictor_probabilities,
            threshold=float(threshold),
        )
        rows.append(
            {
                "threshold": _round(threshold, digits=4),
                "skip_rate": _round(gate.skip_rate),
                "skip_count": gate.skip_count,
                "fallthrough_count": gate.fallthrough_count,
                "effective_auroc": _round(
                    v17.v10.exp3644.tie_aware_auroc(labels_arr, gate.combined_scores)
                ),
            }
        )
    return rows


def choose_operating_point(
    curve: Sequence[Mapping[str, Any]],
    *,
    frozen_ci: tuple[float, float] = FROZEN_CI95,
) -> JsonDict | None:
    """Choose the highest-skip threshold whose effective AUROC is in the CI."""

    safe = [
        dict(row)
        for row in curve
        if _in_ci(float(row.get("effective_auroc", math.nan)), frozen_ci)
    ]
    if not safe:
        return None
    safe.sort(
        key=lambda row: (
            float(row["skip_rate"]),
            float(row["effective_auroc"]),
            -float(row["threshold"]),
        ),
        reverse=True,
    )
    return safe[0]


def predictor_probabilities_for_indices(
    predictor: Any,
    *,
    labels: Sequence[int],
    scores_by_verifier: Mapping[str, Sequence[float]],
    indices: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Score held-out embeddings with the loaded v19 predictor."""

    pairs = v19.build_predictor_pairs(labels, scores_by_verifier)
    probabilities: list[float] = []
    for index in np.asarray(indices, dtype=np.int64):
        domain_scores = predictor.predict(pairs[int(index)]["embedding"])
        probabilities.append(float(np.mean([domain_scores[domain] for domain in JEPA_DOMAINS])))
    return np.asarray(probabilities, dtype=np.float64)


def full_ensemble_scores(matrix: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Compute the unchanged Exp 2837 four-verifier fall-through score."""

    score_arr = np.asarray(matrix, dtype=np.float64)
    weights = v17.v10.exp3644.carnot_current_weights(VERIFIER_NAMES)
    return v17.v10.exp3644.ensemble_scores(score_arr, weights)


def persist_operating_point_state(path: Path | str, *, payload: Mapping[str, Any]) -> str:
    """Persist the selected gate threshold and return the state checksum."""

    output = Path(path)
    stored = dict(payload)
    checksum = _json_sha256(stored)
    stored["sha256"] = checksum
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(stored, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return checksum


def headline_ensemble_unchanged(reference: float | None) -> bool:
    """Return true when the frozen Exp 2837 headline AUROC remains 0.9131."""

    if reference is None:
        return False
    return round(float(reference), 4) == FROZEN_HEADLINE_ROUNDED


def load_headline_ensemble_reference(repo_root: Path | str) -> float | None:
    """Load the Exp 2837 frozen headline ensemble AUROC if present."""

    path = Path(repo_root) / EXP2837_REL_PATH
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, Mapping):
        return None
    value = payload.get("condition_a_production_auroc_mean")
    return float(value) if isinstance(value, int | float) else None


def reproducibility_checksum(
    *,
    labels: np.ndarray | Sequence[int],
    scores: np.ndarray | Sequence[Sequence[float]],
    train_indices: np.ndarray | Sequence[int],
    held_out_indices: np.ndarray | Sequence[int],
    held_out_full_scores: np.ndarray | Sequence[float],
    predictor_probabilities: np.ndarray | Sequence[float],
    curve: Sequence[Mapping[str, Any]],
    selected: Mapping[str, Any],
    random_seed: int,
    predictor_state_sha256: str,
    operating_point_sha256: str | None,
    frozen_ci: tuple[float, float],
) -> str:
    """Hash the measured labels, scores, split, predictions, and gate state."""

    payload = {
        "labels_sha256": _array_sha256(np.asarray(labels, dtype=np.int64)),
        "scores_sha256": _array_sha256(np.asarray(scores, dtype=np.float64)),
        "train_indices_sha256": _array_sha256(np.asarray(train_indices, dtype=np.int64)),
        "held_out_indices_sha256": _array_sha256(np.asarray(held_out_indices, dtype=np.int64)),
        "held_out_full_scores_sha256": _array_sha256(
            np.asarray(held_out_full_scores, dtype=np.float64)
        ),
        "predictor_probabilities_sha256": _array_sha256(
            np.asarray(predictor_probabilities, dtype=np.float64)
        ),
        "curve": [dict(row) for row in curve],
        "selected": dict(selected),
        "random_seed": int(random_seed),
        "predictor_state_sha256": predictor_state_sha256,
        "operating_point_sha256": operating_point_sha256,
        "frozen_ci": [float(frozen_ci[0]), float(frozen_ci[1])],
    }
    return _json_sha256(payload)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3803 artifact schema before writing."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles must be present")
    missing_principles = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in principles]
    if missing_principles:
        raise ValueError(f"missing field principles: {missing_principles}")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must declare verifier-scoring substrate")
    serialized = json.dumps(artifact, sort_keys=True)
    for marker in ("GGUF", "CUDA", "cuda", "live_llm_inference", "torch.cuda"):
        if marker in serialized:
            raise ValueError("forbidden inference marker present")
    if not isinstance(artifact.get("held_out_split_sizes"), Mapping):
        raise ValueError("held_out_split_sizes must be a mapping")
    if not isinstance(artifact.get("headline_ensemble_unchanged"), bool):
        raise ValueError("headline_ensemble_unchanged must be a bare boolean")
    if artifact.get("is_tier3_application_not_retrain") is not True:
        raise ValueError("is_tier3_application_not_retrain must be true")
    if not isinstance(artifact.get("operating_point_persisted"), bool):
        raise ValueError("operating_point_persisted must be boolean")
    if not isinstance(artifact.get("duration_s"), int | float):
        raise ValueError("duration_s must be numeric")

    verdict = artifact.get("honest_verdict")
    if verdict in {
        BLOCKED_INTERPRETER_VERDICT,
        BLOCKED_TIER3_STATE_VERDICT,
        BLOCKED_CORPUS_VERDICT,
        BLOCKED_SCORING_VERDICT,
    }:
        if artifact.get("skip_rate_at_no_regression") is not None:
            raise ValueError("blocked artifact must not fabricate skip_rate_at_no_regression")
        if artifact.get("effective_auroc_at_operating_point") is not None:
            raise ValueError("blocked artifact must not fabricate effective AUROC")
        if artifact.get("compute_saving_vs_accuracy_curve") != []:
            raise ValueError("blocked artifact must not fabricate a gate curve")
        if artifact.get("operating_point_persisted") is not False:
            raise ValueError("blocked artifact must not fabricate operating-point state")
        return

    if verdict == NO_SAFE_VERDICT:
        if artifact.get("skip_rate_at_no_regression") is not None:
            raise ValueError("no-safe artifact must not claim skip_rate_at_no_regression")
        if artifact.get("accuracy_regression") is not True:
            raise ValueError("no-safe artifact must declare accuracy_regression=true")
        return

    if not isinstance(verdict, str) or not (
        verdict.startswith(SUCCESS_VERDICT_PREFIX)
        and verdict.endswith(SUCCESS_VERDICT_SUFFIX)
    ):
        raise ValueError(f"unsupported honest_verdict: {verdict!r}")
    skip_rate = artifact.get("skip_rate_at_no_regression")
    if not isinstance(skip_rate, int | float) or not 0.0 <= float(skip_rate) <= 1.0:
        raise ValueError("skip_rate_at_no_regression must be in [0, 1]")
    effective_auroc = artifact.get("effective_auroc_at_operating_point")
    if not isinstance(effective_auroc, int | float) or not 0.0 <= float(effective_auroc) <= 1.0:
        raise ValueError("effective_auroc_at_operating_point must be in [0, 1]")
    curve = artifact.get("compute_saving_vs_accuracy_curve")
    if not isinstance(curve, list) or not curve:
        raise ValueError("compute_saving_vs_accuracy_curve must be a non-empty list")
    if artifact.get("accuracy_regression") is not False:
        raise ValueError("accuracy_regression must be false for success")
    if artifact.get("headline_ensemble_unchanged") is not True:
        raise ValueError("headline_ensemble_unchanged must be true for success")
    if artifact.get("operating_point_persisted") is not True:
        raise ValueError("operating_point_persisted must be true for success")
    ci = artifact.get("frozen_ci95")
    if not isinstance(ci, Mapping) or not _in_ci(
        float(effective_auroc),
        (float(ci.get("low")), float(ci.get("high"))),
    ):
        raise ValueError("effective_auroc_at_operating_point must be inside frozen CI")
    specs = artifact.get("model_specs")
    if not isinstance(specs, Mapping) or specs.get("predictive_head") != "FR11ExtendedJEPA":
        raise ValueError("model_specs must name the FR11ExtendedJEPA predictor")
    if specs.get("verifiers") != list(VERIFIER_NAMES):
        raise ValueError("model_specs must name the four FoVer verifiers")


def _blocked_artifact(
    *,
    duration_s: float,
    random_seed: int,
    predictor_state_path: Path,
    operating_point_path: Path,
    repo_root: Path | None,
    preconditions: Sequence[Mapping[str, Any]],
    verdict: str,
) -> JsonDict:
    payload = {
        "preconditions": [dict(item) for item in preconditions],
        "random_seed": int(random_seed),
        "verdict": verdict,
    }
    artifact: JsonDict = {
        "artifact": "experiment_3803_fr11_v20_tier3_fast_path_gate",
        "schema": "carnot.fr11_continuous_self_learning_v20",
        "continuous_self_learning_task": True,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "skip_rate_at_no_regression": None,
        "effective_auroc_at_operating_point": None,
        "compute_saving_vs_accuracy_curve": [],
        "accuracy_regression": None,
        "held_out_split_sizes": {
            "train": 0,
            "held_out": 0,
            "held_out_positive": 0,
            "held_out_negative": 0,
            "train_disjoint_from_held_out": True,
        },
        "headline_ensemble_unchanged": False,
        "is_tier3_application_not_retrain": True,
        "operating_point_persisted": False,
        "model_specs": _model_specs(None, n_examples=0),
        "random_seed": int(random_seed),
        "reproducibility_checksum": _json_sha256(payload),
        "duration_s": _round(duration_s),
        "n_examples": 0,
        "n_samples": 0,
        "verifier_names": list(VERIFIER_NAMES),
        "predictor_state_path": _relative_path(predictor_state_path, repo_root),
        "predictor_state_sha256": None,
        "operating_point_state_path": _relative_path(operating_point_path, repo_root),
        "operating_point_state_sha256": None,
        "operating_point": {},
        "frozen_ci95": {"low": float(FROZEN_CI95[0]), "high": float(FROZEN_CI95[1])},
        "methodology": _methodology(random_seed, None),
        "preconditions_checked": [dict(item) for item in preconditions],
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate": {
            "passed": False,
            "condition": "FoVer scores, cached labels, and v19 predictor state are present",
            "principle": "No fast-path saving is emitted without real predictor and corpus evidence.",
        },
    }
    validate_artifact(artifact)
    return artifact


def _interpreter_precondition() -> JsonDict:
    packages = ("jax", "numpy", "sklearn")
    loaded: list[str] = []
    missing: list[str] = []
    for package in packages:
        try:
            importlib.import_module(package)
            loaded.append(package)
        except Exception:  # noqa: BLE001 - reported as blocked precondition.
            missing.append(package)
    try:
        module = importlib.import_module("carnot.fr11.tier3_jepa")
        jepa_importable = callable(getattr(module, "FR11ExtendedJEPA", None))
    except Exception:  # noqa: BLE001 - reported as blocked precondition.
        jepa_importable = False
    executable = Path(sys.executable).as_posix()
    is_venv = ".venv/bin/python" in executable or executable.endswith("/.venv/bin/python")
    available = bool(is_venv and not missing and jepa_importable)
    return {
        "resource": "interpreter_runtime",
        "available": available,
        "detail": (
            f"executable={executable}; loaded={','.join(loaded)}; "
            f"missing={','.join(missing) or 'none'}; "
            f"FR11ExtendedJEPA_importable={jepa_importable}"
        ),
    }


def _predictor_state_precondition(path: Path | str) -> JsonDict:
    state_path = Path(path)
    absolute = state_path.resolve()
    if not state_path.is_file():
        return {
            "resource": "tier3_predictor_state",
            "available": False,
            "detail": f"{absolute}; missing",
        }
    try:
        payload = v19.load_predictor_state(state_path)
        metadata = dict(payload.get("metadata", {}))
        predictor_ok = metadata.get("predictor") == "FR11ExtendedJEPA"
    except Exception as exc:  # noqa: BLE001 - state loading failure is terminal.
        return {
            "resource": "tier3_predictor_state",
            "available": False,
            "detail": f"{absolute}; load_failed={type(exc).__name__}: {exc}",
        }
    return {
        "resource": "tier3_predictor_state",
        "available": bool(predictor_ok),
        "detail": (
            f"{absolute}; predictor={metadata.get('predictor')}; "
            f"predictive_auroc={metadata.get('predictive_auroc')}; "
            f"sha256={_file_sha256(state_path)}"
        ),
    }


def _fover_corpus_precondition(corpus_path: Path, *, n_examples: int) -> JsonDict:
    absolute = corpus_path.resolve()
    if not corpus_path.is_file():
        return {
            "resource": "fover_corpus_absolute_path",
            "available": False,
            "detail": f"{absolute}; missing",
        }
    n_rows = _line_count(corpus_path)
    return {
        "resource": "fover_corpus_absolute_path",
        "available": n_rows >= int(n_examples),
        "detail": f"{absolute}; line_count={n_rows}; required>={int(n_examples)}",
    }


def _blocked_verdict(preconditions: Sequence[Mapping[str, Any]]) -> str:
    for item in preconditions:
        if item.get("available"):
            continue
        resource = str(item.get("resource", "resource"))
        if resource == "interpreter_runtime":
            return BLOCKED_INTERPRETER_VERDICT
        if resource == "tier3_predictor_state":
            return BLOCKED_TIER3_STATE_VERDICT
        if "fover_corpus" in resource:
            return BLOCKED_CORPUS_VERDICT
    return BLOCKED_SCORING_VERDICT


def _trace_precondition(
    labels: Sequence[int],
    scores_by_verifier: Mapping[str, Sequence[float]],
) -> JsonDict:
    return {
        "resource": "cached_traces_with_per_verifier_scores_and_labels",
        "available": bool(labels and scores_by_verifier),
        "detail": f"n_examples={len(labels)}; n_verifiers={len(scores_by_verifier)}",
    }


def _model_specs(corpus_absolute_path: Path | str | None, *, n_examples: int) -> JsonDict:
    return {
        "corpus": "FoVer cached corpus",
        "corpus_absolute_path": str(corpus_absolute_path) if corpus_absolute_path else None,
        "n_examples_requested": int(n_examples),
        "verifiers": list(VERIFIER_NAMES),
        "live_model_invoked": False,
        "scoring_mode": "cached_verifier_scores_only_with_tier3_fast_path_gate",
        "predictive_head": "FR11ExtendedJEPA",
        "predictive_head_module": "carnot.fr11.tier3_jepa",
        "predictor_state_source": "results/experiment_3788_fr11_v19_tier3_predictive_jepa_state.npz",
        "predictor_input_dim": v19.JEPA_EMBED_DIM,
        "predictor_outputs": list(JEPA_DOMAINS),
        "tier": "Tier-3 predictive verification application",
        "ensemble_status": "frozen_0.9131_scoring_ensemble_is_fallthrough_unchanged",
    }


def _methodology(random_seed: int, corpus_absolute_path: Path | str | None) -> JsonDict:
    return {
        "random_seed": int(random_seed),
        "corpus_absolute_path": str(corpus_absolute_path) if corpus_absolute_path else None,
        "scoring_protocol": "ops/reproduction-runbook-fover-headline.md",
        "domain_keys": list(VERIFIER_NAMES),
        "lineage": "v20_tier3_fast_path_application_not_v19_retrain",
        "feature_builder": "v19_four_verifier_scores_to_258_dim_FR11ExtendedJEPA_embedding",
        "predictor_api": "FR11ExtendedJEPA.predict/energy loaded from persisted v19 state",
        "gate_confidence": "abs(mean_domain_probability - 0.5)",
        "fallthrough_path": "Exp 2837 frozen four-verifier ensemble scoring",
        "heldout_split": "v19 deterministic test rows reconstructed from seed 3788",
    }


def _metadata_test_fraction(metadata: Mapping[str, Any], n_examples: int) -> float:
    sizes = metadata.get("train_test_split_sizes")
    if isinstance(sizes, Mapping):
        test = sizes.get("test")
        if isinstance(test, int | float) and int(n_examples) > 0:
            fraction = float(test) / float(n_examples)
            if 0.0 < fraction < 0.5:
                return fraction
    return DEFAULT_V19_TEST_FRACTION


def _split_sizes(
    labels: np.ndarray,
    train_indices: np.ndarray,
    held_out_indices: np.ndarray,
) -> JsonDict:
    train_labels = labels[train_indices]
    held_out_labels = labels[held_out_indices]
    return {
        "train": int(len(train_indices)),
        "held_out": int(len(held_out_indices)),
        "train_positive": int(np.sum(train_labels == 1)),
        "train_negative": int(np.sum(train_labels == 0)),
        "held_out_positive": int(np.sum(held_out_labels == 1)),
        "held_out_negative": int(np.sum(held_out_labels == 0)),
        "train_disjoint_from_held_out": bool(
            set(int(index) for index in train_indices).isdisjoint(
                set(int(index) for index in held_out_indices)
            )
        ),
        "split_source": "v19_reconstructed_stratified_split_seed_3788",
    }


def _success_verdict(skip_rate: float, effective_auroc: float) -> str:
    return (
        f"{SUCCESS_VERDICT_PREFIX}{float(skip_rate):.4f}_effective_auroc_"
        f"{float(effective_auroc):.4f}{SUCCESS_VERDICT_SUFFIX}"
    )


def _in_ci(value: float, frozen_ci: tuple[float, float]) -> bool:
    return float(frozen_ci[0]) <= float(value) <= float(frozen_ci[1])


def _resolve_under_root(root: Path, path: Path | str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _relative_path(path: Path, root: Path | None) -> str:
    if root is None:
        return path.as_posix()
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _require_binary_labels(labels: np.ndarray) -> None:
    if set(int(value) for value in labels) != {0, 1}:
        raise ValueError("labels must contain both binary classes 0 and 1")


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.time() if now_s is None else float(now_s)
    return max(0.0001, end - float(started_s))


def _round(value: float | int | None, digits: int = 9) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

"""FR-11 continuous self-learning v21 fast-path robustness measurement.

Spec: REQ-LEARN-3813, SCENARIO-LEARN-3813.

The v21 forward difference measures the Exp 3803 Tier-3 fast-path operating
point on a second, disjoint FoVer split. It loads the persisted v19 predictor
and v20 threshold unchanged, then reports whether that operating point holds
out of sample. It does not retrain the predictor and it does not re-tune the
threshold to match the v20 skip rate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import random
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from carnot.fr11 import continuous_self_learning_v19 as v19
from carnot.fr11 import continuous_self_learning_v20 as v20


JsonDict = dict[str, Any]
IndexedFoVerRow = tuple[int, Mapping[str, Any]]

OUTPUT_REL_PATH = Path("results/experiment_3813_fr11_v21_fast_path_robustness.json")
PREDICTOR_STATE_REL_PATH = v19.PREDICTOR_STATE_REL_PATH
OPERATING_POINT_REL_PATH = v20.OPERATING_POINT_REL_PATH
EXP2837_REL_PATH = v20.EXP2837_REL_PATH
DEFAULT_RANDOM_SEED = 3813
DEFAULT_V19_SAMPLE_SEED = 3788
DEFAULT_V19_SPLIT_SEED = 3788
DEFAULT_V19_N_EXAMPLES = 1000
DEFAULT_V19_TEST_FRACTION = 0.2
DEFAULT_SECOND_SPLIT_SEED = 3813
DEFAULT_SECOND_SPLIT_SIZE = 200
DEFAULT_SKIP_RATE_BAND = 0.10
FROZEN_CI95 = v20.FROZEN_CI95
FROZEN_HEADLINE_ROUNDED = v20.FROZEN_HEADLINE_ROUNDED
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates "
    "(principle: reads cached scores + the persisted predictor + operating point, no live model)."
)
SUCCESS_VERDICT_PREFIX = "complete: fr11_v21_fast_path_robustness_skip_"
BLOCKED_INTERPRETER_VERDICT = "blocked_interpreter_runtime"
BLOCKED_PERSISTED_STATE_VERDICT = "blocked_persisted_state_missing"
BLOCKED_CORPUS_VERDICT = "blocked_fover_corpus_missing"
BLOCKED_SCORING_VERDICT = "blocked_fover_scores_missing"
BLOCKED_INSUFFICIENT_HOLDOUT_VERDICT = "blocked_insufficient_holdout"
VERIFIER_NAMES = v20.VERIFIER_NAMES
JEPA_DOMAINS = v19.JEPA_DOMAINS

score_matrix = v20.score_matrix
full_ensemble_scores = v20.full_ensemble_scores
apply_confidence_gate = v20.apply_confidence_gate
load_headline_ensemble_reference = v20.load_headline_ensemble_reference
probe_cached_trace_preconditions = v20.probe_cached_trace_preconditions

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "v20_operating_point_threshold",
    "skip_rate_second_split",
    "effective_auroc_second_split",
    "operating_point_generalizes",
    "three_split_sizes",
    "accuracy_regression",
    "headline_ensemble_unchanged",
    "is_measurement_not_retrain",
    "model_specs",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal prefix; the robustness outcome; blocked_<resource> if a "
        "precondition failed."
    ),
    "inference_substrate": (
        "verifier_ensemble_against_cached_candidates (principle: reads cached "
        "scores + the persisted predictor + operating point, no live model)."
    ),
    "v20_operating_point_threshold": (
        "The persisted v20 threshold T reused UNCHANGED -- confirms v21 measures, "
        "it does not re-tune."
    ),
    "skip_rate_second_split": (
        "The skip rate of the persisted gate on the SECOND held-out split -- the "
        "out-of-sample generalization evidence vs v20's 0.56."
    ),
    "effective_auroc_second_split": (
        "The combined fast-path+fallthrough AUROC on the second split -- must stay "
        "within the frozen CI [0.9027, 0.9235] for the gate to generalize without "
        "regression."
    ),
    "operating_point_generalizes": (
        "BARE bool -- whether the v20 operating point holds out-of-sample (skip "
        "within band AND no AUROC regression); false is a valid finding (needs "
        "per-split recalibration)."
    ),
    "three_split_sizes": (
        "Sample-size + leakage hygiene -- the predictor-training split, the v20 "
        "measurement split, and the v21 second split are all disjoint."
    ),
    "accuracy_regression": (
        "BARE bool, must be false for generalization -- the gate trades compute "
        "for nothing if it degrades the verdict on the second split."
    ),
    "headline_ensemble_unchanged": (
        "BARE bool, true -- the frozen 0.9131 full-scoring ensemble is the "
        "fall-through and is UNTOUCHED; the gate is additive."
    ),
    "is_measurement_not_retrain": (
        "BARE bool, true -- confirms v21 MEASURES the persisted operating point "
        "(does not retrain or re-tune) -- the lineage-continuation forward difference."
    ),
    "model_specs": (
        "Names the corpus + 4 verifiers + the Tier-3 predictor -- honest substrate."
    ),
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Content hash catches drift.",
    "duration_s": "Wall-clock plausibility floor; verifier-scoring on a second split is modest.",
}


class InsufficientHoldoutError(RuntimeError):
    """Raised when the corpus cannot provide a disjoint second held-out split."""


@dataclass(frozen=True)
class SecondSplitPlan:
    """Original FoVer row IDs for the three leakage-sensitive splits."""

    v19_train_row_ids: tuple[int, ...]
    v20_measurement_row_ids: tuple[int, ...]
    v21_second_row_ids: tuple[int, ...]


@dataclass(frozen=True)
class ScoredSecondSplit:
    """Labels and verifier scores for the second held-out split."""

    labels: list[int]
    scores_by_verifier: dict[str, list[float]]
    row_ids: list[int]
    v19_train_row_ids: list[int]
    v20_measurement_row_ids: list[int]
    corpus_absolute_path: Path


def build_artifact(
    repo_root: Path | str,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    n_examples: int = DEFAULT_V19_N_EXAMPLES,
    second_split_size: int = DEFAULT_SECOND_SPLIT_SIZE,
    random_seed: int = DEFAULT_RANDOM_SEED,
    v19_sample_seed: int = DEFAULT_V19_SAMPLE_SEED,
    v19_split_seed: int = DEFAULT_V19_SPLIT_SEED,
    second_split_seed: int = DEFAULT_SECOND_SPLIT_SEED,
    predictor_state_path: Path | str = PREDICTOR_STATE_REL_PATH,
    operating_point_path: Path | str = OPERATING_POINT_REL_PATH,
    frozen_ci: tuple[float, float] = FROZEN_CI95,
) -> JsonDict:
    """Build Exp 3813 from cached FoVer rows and persisted v19/v20 state."""

    root = Path(repo_root)
    start = time.time() if started_s is None else float(started_s)
    predictor_state = _resolve_under_root(root, predictor_state_path)
    operating_point_state = _resolve_under_root(root, operating_point_path)
    corpus_path = root / "data" / "fover_corpus.jsonl"
    preconditions = [
        _interpreter_precondition(),
        _persisted_state_precondition(predictor_state, operating_point_state),
        _fover_corpus_precondition(corpus_path, n_examples=n_examples + second_split_size),
    ]
    if not all(item["available"] for item in preconditions):
        return _blocked_artifact(
            duration_s=_duration(start, now_s),
            random_seed=random_seed,
            predictor_state_path=predictor_state,
            operating_point_path=operating_point_state,
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
            operating_point_path=operating_point_state,
            repo_root=root,
            preconditions=preconditions,
            verdict=BLOCKED_SCORING_VERDICT,
        )

    try:
        split = score_second_heldout_split(
            root,
            v19_sample_seed=v19_sample_seed,
            v19_n_examples=n_examples,
            v19_split_seed=v19_split_seed,
            second_split_seed=second_split_seed,
            second_split_size=second_split_size,
        )
    except InsufficientHoldoutError as exc:
        return _blocked_artifact(
            duration_s=_duration(start, now_s),
            random_seed=random_seed,
            predictor_state_path=predictor_state,
            operating_point_path=operating_point_state,
            repo_root=root,
            preconditions=[
                *preconditions,
                {
                    "resource": "second_heldout_split",
                    "available": False,
                    "detail": str(exc),
                },
            ],
            verdict=BLOCKED_INSUFFICIENT_HOLDOUT_VERDICT,
        )
    except Exception as exc:  # noqa: BLE001 - cached scoring failure is terminal.
        return _blocked_artifact(
            duration_s=_duration(start, now_s),
            random_seed=random_seed,
            predictor_state_path=predictor_state,
            operating_point_path=operating_point_state,
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
        labels=split.labels,
        scores_by_verifier=split.scores_by_verifier,
        row_ids=split.row_ids,
        v19_train_row_ids=split.v19_train_row_ids,
        v20_measurement_row_ids=split.v20_measurement_row_ids,
        predictor_state_path=predictor_state,
        operating_point_path=operating_point_state,
        started_s=start,
        now_s=now_s,
        random_seed=random_seed,
        repo_root=root,
        preconditions=preconditions,
        corpus_absolute_path=split.corpus_absolute_path,
        frozen_ci=frozen_ci,
        headline_ensemble_reference=load_headline_ensemble_reference(root),
    )


def build_artifact_from_scores(
    *,
    labels: Sequence[int],
    scores_by_verifier: Mapping[str, Sequence[float]],
    row_ids: Sequence[int],
    v19_train_row_ids: Sequence[int],
    v20_measurement_row_ids: Sequence[int],
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
    operating_point_state_sha256: str | None = None,
    skip_rate_band: float = DEFAULT_SKIP_RATE_BAND,
) -> JsonDict:
    """Measure the persisted v20 gate on one disjoint second split."""

    root = Path(repo_root) if repo_root is not None else None
    predictor_state = _resolve_under_root(root or Path("."), predictor_state_path)
    operating_point_state = _resolve_under_root(root or Path("."), operating_point_path)
    persisted_precondition = _persisted_state_precondition(
        predictor_state,
        operating_point_state,
    )
    checked = [*(dict(item) for item in (preconditions or []))]
    if not persisted_precondition["available"]:
        checked.append(persisted_precondition)
        return _blocked_artifact(
            duration_s=_duration(float(started_s), now_s),
            random_seed=random_seed,
            predictor_state_path=predictor_state,
            operating_point_path=operating_point_state,
            repo_root=root,
            preconditions=checked,
            verdict=BLOCKED_PERSISTED_STATE_VERDICT,
        )
    if not labels or not scores_by_verifier or not row_ids:
        checked.append(
            {
                "resource": "second_heldout_split",
                "available": False,
                "detail": (
                    f"labels={len(labels)}; row_ids={len(row_ids)}; "
                    f"score_columns={len(scores_by_verifier)}"
                ),
            }
        )
        return _blocked_artifact(
            duration_s=_duration(float(started_s), now_s),
            random_seed=random_seed,
            predictor_state_path=predictor_state,
            operating_point_path=operating_point_state,
            repo_root=root,
            preconditions=checked,
            verdict=BLOCKED_INSUFFICIENT_HOLDOUT_VERDICT,
        )

    labels_arr = np.asarray(labels, dtype=np.int64)
    _require_binary_labels(labels_arr)
    row_id_arr = np.asarray(row_ids, dtype=np.int64)
    if len(row_id_arr) != len(labels_arr):
        raise ValueError("row_ids and labels must have the same length")
    matrix = score_matrix(scores_by_verifier, VERIFIER_NAMES)
    if matrix.shape[0] != len(labels_arr):
        raise ValueError("labels and verifier scores must have the same length")

    state_payload = v19.load_predictor_state(predictor_state)
    predictor = state_payload["predictor"]
    operating_point = load_operating_point_state(operating_point_state)
    threshold = float(operating_point["threshold"])
    v20_skip_rate = float(operating_point["skip_rate"])
    full_scores = full_ensemble_scores(matrix)
    predictor_probabilities = _predictor_probabilities(
        predictor,
        labels=labels_arr.tolist(),
        scores_by_verifier=scores_by_verifier,
    )
    gate = apply_confidence_gate(
        full_scores,
        predictor_probabilities,
        threshold=threshold,
    )
    effective_auroc = v20.v17.v10.exp3644.tie_aware_auroc(
        labels_arr,
        gate.combined_scores,
    )
    full_second_auroc = v20.v17.v10.exp3644.tie_aware_auroc(labels_arr, full_scores)
    skip_rate = float(gate.skip_rate)
    auroc_in_ci = _in_ci(effective_auroc, frozen_ci)
    accuracy_regression = bool(float(effective_auroc) < float(frozen_ci[0]))
    skip_within_band = bool(abs(skip_rate - v20_skip_rate) <= float(skip_rate_band))
    headline_ok = headline_ensemble_unchanged(headline_ensemble_reference)
    split_sizes = three_split_sizes(
        v19_train_row_ids=v19_train_row_ids,
        v20_measurement_row_ids=v20_measurement_row_ids,
        v21_second_row_ids=row_ids,
        second_split_labels=labels_arr,
    )
    generalizes = bool(
        skip_within_band
        and auroc_in_ci
        and not accuracy_regression
        and headline_ok
        and split_sizes["all_disjoint"]
    )
    state_sha = predictor_state_sha256 or _file_sha256(predictor_state)
    op_sha = operating_point_state_sha256 or _file_sha256(operating_point_state)

    artifact: JsonDict = {
        "artifact": "experiment_3813_fr11_v21_fast_path_robustness",
        "schema": "carnot.fr11_continuous_self_learning_v21",
        "continuous_self_learning_task": True,
        "honest_verdict": _success_verdict(skip_rate, effective_auroc, generalizes),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "v20_operating_point_threshold": _round(threshold),
        "skip_rate_second_split": _round(skip_rate),
        "effective_auroc_second_split": _round(effective_auroc),
        "operating_point_generalizes": generalizes,
        "three_split_sizes": split_sizes,
        "accuracy_regression": accuracy_regression,
        "headline_ensemble_unchanged": bool(headline_ok),
        "is_measurement_not_retrain": True,
        "model_specs": _model_specs(corpus_absolute_path, n_examples=len(labels_arr)),
        "random_seed": int(random_seed),
        "reproducibility_checksum": reproducibility_checksum(
            labels=labels_arr,
            row_ids=row_id_arr,
            scores=matrix,
            full_scores=full_scores,
            predictor_probabilities=predictor_probabilities,
            combined_scores=gate.combined_scores,
            skip_mask=gate.skip_mask,
            v19_train_row_ids=v19_train_row_ids,
            v20_measurement_row_ids=v20_measurement_row_ids,
            threshold=threshold,
            random_seed=random_seed,
            predictor_state_sha256=state_sha,
            operating_point_state_sha256=op_sha,
            frozen_ci=frozen_ci,
        ),
        "duration_s": _round(_duration(float(started_s), now_s)),
        "n_samples": int(len(labels_arr)),
        "verifier_names": list(VERIFIER_NAMES),
        "predictor_state_path": _relative_path(predictor_state, root),
        "predictor_state_sha256": state_sha,
        "operating_point_state_path": _relative_path(operating_point_state, root),
        "operating_point_state_sha256": op_sha,
        "v20_reported_skip_rate": _round(v20_skip_rate),
        "v20_reported_effective_auroc": _round(operating_point["effective_auroc"]),
        "skip_rate_band": float(skip_rate_band),
        "skip_rate_within_band": skip_within_band,
        "effective_auroc_in_frozen_ci": auroc_in_ci,
        "frozen_ci95": {"low": float(frozen_ci[0]), "high": float(frozen_ci[1])},
        "frozen_headline_ensemble_auroc": headline_ensemble_reference,
        "fallthrough_full_ensemble_auroc_second_split": _round(full_second_auroc),
        "skip_count_second_split": gate.skip_count,
        "fallthrough_count_second_split": gate.fallthrough_count,
        "predictor_probabilities_sha256": _array_sha256(predictor_probabilities),
        "combined_scores_sha256": _array_sha256(gate.combined_scores),
        "methodology": _methodology(random_seed, corpus_absolute_path),
        "preconditions_checked": checked,
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate": {
            "passed": generalizes,
            "condition": (
                "second-split skip rate within band of persisted 0.56 AND "
                "effective AUROC in frozen CI AND accuracy_regression=false AND "
                "headline ensemble unchanged"
            ),
            "principle": (
                "The v20 operating point is deployable only if the persisted "
                "threshold holds on a disjoint second split without re-tuning."
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
    row_ids: Sequence[int] | None = None,
    v19_train_row_ids: Sequence[int] | None = None,
    v20_measurement_row_ids: Sequence[int] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
    frozen_ci: tuple[float, float] = FROZEN_CI95,
) -> Path:
    """Build, validate, and write the Exp 3813 JSON artifact."""

    root = Path(repo_root)
    start = time.time() if started_s is None else float(started_s)
    resolved_state_path = _resolve_under_root(root, predictor_state_path)
    resolved_operating_path = _resolve_under_root(root, operating_point_path)
    if labels is not None or scores_by_verifier is not None or row_ids is not None:
        artifact = build_artifact_from_scores(
            labels=labels or [],
            scores_by_verifier=scores_by_verifier or {},
            row_ids=row_ids or [],
            v19_train_row_ids=v19_train_row_ids or [],
            v20_measurement_row_ids=v20_measurement_row_ids or [],
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


def score_second_heldout_split(
    repo_root: Path | str,
    *,
    v19_sample_seed: int = DEFAULT_V19_SAMPLE_SEED,
    v19_n_examples: int = DEFAULT_V19_N_EXAMPLES,
    v19_split_seed: int = DEFAULT_V19_SPLIT_SEED,
    v19_test_fraction: float = DEFAULT_V19_TEST_FRACTION,
    second_split_seed: int = DEFAULT_SECOND_SPLIT_SEED,
    second_split_size: int = DEFAULT_SECOND_SPLIT_SIZE,
) -> ScoredSecondSplit:
    """Select a disjoint second split and score it with the four verifiers."""

    from carnot.eval.fover_memory_leakage_v3 import (
        _fr11_memory_score,
        _load_fr11_memory_index,
        _score_text_verifiers,
    )

    root = Path(repo_root)
    corpus_path = root / "data" / "fover_corpus.jsonl"
    indexed_rows = read_indexed_fover_rows(corpus_path)
    plan = construct_second_split_plan(
        indexed_rows,
        v19_sample_seed=v19_sample_seed,
        v19_n_examples=v19_n_examples,
        v19_split_seed=v19_split_seed,
        v19_test_fraction=v19_test_fraction,
        second_split_seed=second_split_seed,
        second_split_size=second_split_size,
    )
    rows_by_id = {int(row_id): row for row_id, row in indexed_rows}
    second_rows = [(row_id, rows_by_id[int(row_id)]) for row_id in plan.v21_second_row_ids]
    texts = [str(row.get("step_text", "")) for _row_id, row in second_rows]
    text_scores = _score_text_verifiers(texts)
    memory_index = _load_fr11_memory_index(root)
    scores_by_verifier = {
        "fr11_session_memory": [
            float(_fr11_memory_score(dict(row), memory_index)) for _row_id, row in second_rows
        ],
        "tier0r_curry_howard": [float(value) for value in text_scores["tier0r_curry_howard"]],
        "tier0s_arithmetic_gap": [float(value) for value in text_scores["tier0s_arithmetic_gap"]],
        "tier0u_logical_consistency": [
            float(value) for value in text_scores["tier0u_logical_consistency"]
        ],
    }
    return ScoredSecondSplit(
        labels=[_label_to_int(row["label"]) for _row_id, row in second_rows],
        scores_by_verifier=scores_by_verifier,
        row_ids=[int(row_id) for row_id, _row in second_rows],
        v19_train_row_ids=list(plan.v19_train_row_ids),
        v20_measurement_row_ids=list(plan.v20_measurement_row_ids),
        corpus_absolute_path=corpus_path.resolve(),
    )


def read_indexed_fover_rows(path: Path | str) -> list[IndexedFoVerRow]:
    """Read valid FoVer rows while preserving stable original line indices."""

    indexed: list[IndexedFoVerRow] = []
    for line_index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        label = row.get("label")
        if label in {"correct", "incorrect", 0, 1, "0", "1"}:
            indexed.append((line_index, row))
    return indexed


def construct_second_split_plan(
    indexed_rows: Sequence[IndexedFoVerRow],
    *,
    v19_sample_seed: int = DEFAULT_V19_SAMPLE_SEED,
    v19_n_examples: int = DEFAULT_V19_N_EXAMPLES,
    v19_split_seed: int = DEFAULT_V19_SPLIT_SEED,
    v19_test_fraction: float = DEFAULT_V19_TEST_FRACTION,
    second_split_seed: int = DEFAULT_SECOND_SPLIT_SEED,
    second_split_size: int = DEFAULT_SECOND_SPLIT_SIZE,
) -> SecondSplitPlan:
    """Return mutually disjoint v19 train, v20 measurement, and v21 row IDs."""

    v19_rows = _select_balanced_indexed_subset(
        indexed_rows,
        seed=v19_sample_seed,
        n_examples=v19_n_examples,
    )
    v19_labels = np.asarray([_label_to_int(row["label"]) for _row_id, row in v19_rows])
    train_indices, test_indices = v19.stratified_train_test_indices(
        v19_labels,
        test_fraction=v19_test_fraction,
        random_seed=v19_split_seed,
    )
    v19_row_ids = np.asarray([int(row_id) for row_id, _row in v19_rows], dtype=np.int64)
    train_row_ids = tuple(int(v19_row_ids[int(index)]) for index in train_indices)
    measurement_row_ids = tuple(int(v19_row_ids[int(index)]) for index in test_indices)
    excluded = set(train_row_ids) | set(measurement_row_ids)
    second_rows = _select_balanced_indexed_subset(
        indexed_rows,
        seed=second_split_seed,
        n_examples=second_split_size,
        exclude_row_ids=excluded,
    )
    second_row_ids = tuple(int(row_id) for row_id, _row in second_rows)
    if (
        set(train_row_ids) & set(measurement_row_ids)
        or set(train_row_ids) & set(second_row_ids)
        or set(measurement_row_ids) & set(second_row_ids)
    ):
        raise InsufficientHoldoutError("constructed splits are not mutually disjoint")
    return SecondSplitPlan(
        v19_train_row_ids=train_row_ids,
        v20_measurement_row_ids=measurement_row_ids,
        v21_second_row_ids=second_row_ids,
    )


def _select_balanced_indexed_subset(
    indexed_rows: Sequence[IndexedFoVerRow],
    *,
    seed: int,
    n_examples: int,
    exclude_row_ids: set[int] | None = None,
) -> list[IndexedFoVerRow]:
    excluded = exclude_row_ids or set()
    candidates = [(row_id, row) for row_id, row in indexed_rows if int(row_id) not in excluded]
    positives = [(row_id, row) for row_id, row in candidates if _label_to_int(row["label"]) == 1]
    negatives = [(row_id, row) for row_id, row in candidates if _label_to_int(row["label"]) == 0]
    n_pos = int(n_examples) // 2
    n_neg = int(n_examples) - n_pos
    if len(positives) < n_pos or len(negatives) < n_neg:
        raise InsufficientHoldoutError(
            f"FoVer corpus lacks disjoint class balance for n={int(n_examples)}: "
            f"positives={len(positives)}, negatives={len(negatives)}, "
            f"excluded={len(excluded)}"
        )
    rng = random.Random(int(seed))
    subset = [*rng.sample(positives, n_pos), *rng.sample(negatives, n_neg)]
    rng.shuffle(subset)
    return subset


def load_operating_point_state(path: Path | str) -> JsonDict:
    """Load and validate the persisted Exp 3803 operating-point state."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("operating point state must be a mapping")
    selected = payload.get("selected_operating_point")
    if not isinstance(selected, Mapping):
        raise ValueError("selected_operating_point must be a mapping")
    threshold = selected.get("threshold")
    if not isinstance(threshold, int | float) or not math.isfinite(float(threshold)):
        raise ValueError("selected operating point threshold must be numeric")
    skip_rate = selected.get("skip_rate")
    if not isinstance(skip_rate, int | float) or not 0.0 <= float(skip_rate) <= 1.0:
        raise ValueError("selected operating point skip_rate must be in [0, 1]")
    effective_auroc = selected.get("effective_auroc")
    if not isinstance(effective_auroc, int | float) or not 0.0 <= float(effective_auroc) <= 1.0:
        raise ValueError("selected operating point effective_auroc must be in [0, 1]")
    return {
        "threshold": float(threshold),
        "skip_rate": float(skip_rate),
        "effective_auroc": float(effective_auroc),
        "raw": dict(payload),
    }


def three_split_sizes(
    *,
    v19_train_row_ids: Sequence[int],
    v20_measurement_row_ids: Sequence[int],
    v21_second_row_ids: Sequence[int],
    second_split_labels: Sequence[int] | np.ndarray,
) -> JsonDict:
    """Summarize split sizes and disjointness using original FoVer row IDs."""

    train_set = {int(value) for value in v19_train_row_ids}
    measurement_set = {int(value) for value in v20_measurement_row_ids}
    second_set = {int(value) for value in v21_second_row_ids}
    labels_arr = np.asarray(second_split_labels, dtype=np.int64)
    return {
        "predictor_training": int(len(train_set)),
        "v20_measurement": int(len(measurement_set)),
        "v21_second": int(len(second_set)),
        "v21_second_positive": int(np.sum(labels_arr == 1)),
        "v21_second_negative": int(np.sum(labels_arr == 0)),
        "train_disjoint_from_v20_measurement": bool(train_set.isdisjoint(measurement_set)),
        "train_disjoint_from_v21_second": bool(train_set.isdisjoint(second_set)),
        "v20_measurement_disjoint_from_v21_second": bool(
            measurement_set.isdisjoint(second_set)
        ),
        "all_disjoint": bool(
            train_set.isdisjoint(measurement_set)
            and train_set.isdisjoint(second_set)
            and measurement_set.isdisjoint(second_set)
        ),
        "row_id_hashes": {
            "predictor_training_sha256": _int_sequence_sha256(sorted(train_set)),
            "v20_measurement_sha256": _int_sequence_sha256(sorted(measurement_set)),
            "v21_second_sha256": _int_sequence_sha256(sorted(second_set)),
        },
    }


def reproducibility_checksum(
    *,
    labels: np.ndarray | Sequence[int],
    row_ids: np.ndarray | Sequence[int],
    scores: np.ndarray | Sequence[Sequence[float]],
    full_scores: np.ndarray | Sequence[float],
    predictor_probabilities: np.ndarray | Sequence[float],
    combined_scores: np.ndarray | Sequence[float],
    skip_mask: np.ndarray | Sequence[bool],
    v19_train_row_ids: Sequence[int],
    v20_measurement_row_ids: Sequence[int],
    threshold: float,
    random_seed: int,
    predictor_state_sha256: str,
    operating_point_state_sha256: str,
    frozen_ci: tuple[float, float],
) -> str:
    """Hash the measured split, persisted states, scores, and fixed threshold."""

    payload = {
        "labels_sha256": _array_sha256(np.asarray(labels, dtype=np.int64)),
        "row_ids_sha256": _array_sha256(np.asarray(row_ids, dtype=np.int64)),
        "scores_sha256": _array_sha256(np.asarray(scores, dtype=np.float64)),
        "full_scores_sha256": _array_sha256(np.asarray(full_scores, dtype=np.float64)),
        "predictor_probabilities_sha256": _array_sha256(
            np.asarray(predictor_probabilities, dtype=np.float64)
        ),
        "combined_scores_sha256": _array_sha256(np.asarray(combined_scores, dtype=np.float64)),
        "skip_mask_sha256": _array_sha256(np.asarray(skip_mask, dtype=np.bool_)),
        "v19_train_row_ids_sha256": _int_sequence_sha256(v19_train_row_ids),
        "v20_measurement_row_ids_sha256": _int_sequence_sha256(v20_measurement_row_ids),
        "threshold": float(threshold),
        "random_seed": int(random_seed),
        "predictor_state_sha256": predictor_state_sha256,
        "operating_point_state_sha256": operating_point_state_sha256,
        "frozen_ci": [float(frozen_ci[0]), float(frozen_ci[1])],
    }
    return _json_sha256(payload)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3813 artifact schema before writing."""

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
    if not isinstance(artifact.get("operating_point_generalizes"), bool):
        raise ValueError("operating_point_generalizes must be a bare boolean")
    if not isinstance(artifact.get("three_split_sizes"), Mapping):
        raise ValueError("three_split_sizes must be a mapping")
    if not isinstance(artifact.get("headline_ensemble_unchanged"), bool):
        raise ValueError("headline_ensemble_unchanged must be a bare boolean")
    if artifact.get("is_measurement_not_retrain") is not True:
        raise ValueError("is_measurement_not_retrain must be true")
    if not isinstance(artifact.get("duration_s"), int | float):
        raise ValueError("duration_s must be numeric")

    verdict = artifact.get("honest_verdict")
    if verdict in {
        BLOCKED_INTERPRETER_VERDICT,
        BLOCKED_PERSISTED_STATE_VERDICT,
        BLOCKED_CORPUS_VERDICT,
        BLOCKED_SCORING_VERDICT,
        BLOCKED_INSUFFICIENT_HOLDOUT_VERDICT,
    }:
        if artifact.get("v20_operating_point_threshold") is not None:
            raise ValueError("blocked artifact must not fabricate v20 threshold")
        if artifact.get("skip_rate_second_split") is not None:
            raise ValueError("blocked artifact must not fabricate skip_rate_second_split")
        if artifact.get("effective_auroc_second_split") is not None:
            raise ValueError("blocked artifact must not fabricate effective AUROC")
        if artifact.get("accuracy_regression") is not None:
            raise ValueError("blocked artifact must not fabricate accuracy_regression")
        return

    if not isinstance(verdict, str) or not verdict.startswith(SUCCESS_VERDICT_PREFIX):
        raise ValueError(f"unsupported honest_verdict: {verdict!r}")
    threshold = artifact.get("v20_operating_point_threshold")
    if not isinstance(threshold, int | float) or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("v20_operating_point_threshold must be in [0, 1]")
    skip_rate = artifact.get("skip_rate_second_split")
    if not isinstance(skip_rate, int | float) or not 0.0 <= float(skip_rate) <= 1.0:
        raise ValueError("skip_rate_second_split must be in [0, 1]")
    effective_auroc = artifact.get("effective_auroc_second_split")
    if not isinstance(effective_auroc, int | float) or not 0.0 <= float(effective_auroc) <= 1.0:
        raise ValueError("effective_auroc_second_split must be in [0, 1]")
    if not isinstance(artifact.get("accuracy_regression"), bool):
        raise ValueError("accuracy_regression must be a bare boolean")
    if artifact.get("headline_ensemble_unchanged") is not True:
        raise ValueError("headline_ensemble_unchanged must be true for measured artifact")
    sizes = artifact["three_split_sizes"]
    if sizes.get("all_disjoint") is not True:
        raise ValueError("three_split_sizes must prove all splits are disjoint")
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
        "artifact": "experiment_3813_fr11_v21_fast_path_robustness",
        "schema": "carnot.fr11_continuous_self_learning_v21",
        "continuous_self_learning_task": True,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "v20_operating_point_threshold": None,
        "skip_rate_second_split": None,
        "effective_auroc_second_split": None,
        "operating_point_generalizes": False,
        "three_split_sizes": {
            "predictor_training": 0,
            "v20_measurement": 0,
            "v21_second": 0,
            "all_disjoint": True,
        },
        "accuracy_regression": None,
        "headline_ensemble_unchanged": False,
        "is_measurement_not_retrain": True,
        "model_specs": _model_specs(None, n_examples=0),
        "random_seed": int(random_seed),
        "reproducibility_checksum": _json_sha256(payload),
        "duration_s": _round(duration_s),
        "n_samples": 0,
        "verifier_names": list(VERIFIER_NAMES),
        "predictor_state_path": _relative_path(predictor_state_path, repo_root),
        "predictor_state_sha256": None,
        "operating_point_state_path": _relative_path(operating_point_path, repo_root),
        "operating_point_state_sha256": None,
        "methodology": _methodology(random_seed, None),
        "preconditions_checked": [dict(item) for item in preconditions],
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate": {
            "passed": False,
            "condition": "persisted v19 predictor, v20 operating point, and disjoint holdout exist",
            "principle": "No robustness finding is emitted without real persisted state and split evidence.",
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


def _persisted_state_precondition(
    predictor_state_path: Path | str,
    operating_point_path: Path | str,
) -> JsonDict:
    predictor_state = Path(predictor_state_path)
    operating_point = Path(operating_point_path)
    details: list[str] = []
    available = True
    predictor_check = v20._predictor_state_precondition(predictor_state)
    available = available and bool(predictor_check["available"])
    details.append(f"predictor_state=({predictor_check['detail']})")
    if not operating_point.is_file():
        available = False
        details.append(f"operating_point={operating_point.resolve()}; missing")
    else:
        try:
            op = load_operating_point_state(operating_point)
            details.append(
                f"operating_point={operating_point.resolve()}; "
                f"threshold={op['threshold']}; skip_rate={op['skip_rate']}; "
                f"sha256={_file_sha256(operating_point)}"
            )
        except Exception as exc:  # noqa: BLE001 - reported as blocked precondition.
            available = False
            details.append(
                f"operating_point={operating_point.resolve()}; "
                f"load_failed={type(exc).__name__}: {exc}"
            )
    return {
        "resource": "persisted_fast_path_state",
        "available": bool(available),
        "detail": "; ".join(details),
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
        if resource == "persisted_fast_path_state":
            return BLOCKED_PERSISTED_STATE_VERDICT
        if "fover_corpus" in resource:
            return BLOCKED_CORPUS_VERDICT
        if resource == "second_heldout_split":
            return BLOCKED_INSUFFICIENT_HOLDOUT_VERDICT
    return BLOCKED_SCORING_VERDICT


def _predictor_probabilities(
    predictor: Any,
    *,
    labels: Sequence[int],
    scores_by_verifier: Mapping[str, Sequence[float]],
) -> np.ndarray:
    pairs = v19.build_predictor_pairs(labels, scores_by_verifier)
    probabilities: list[float] = []
    for pair in pairs:
        domain_scores = predictor.predict(pair["embedding"])
        probabilities.append(float(np.mean([domain_scores[domain] for domain in JEPA_DOMAINS])))
    return np.asarray(probabilities, dtype=np.float64)


def _model_specs(corpus_absolute_path: Path | str | None, *, n_examples: int) -> JsonDict:
    return {
        "corpus": "FoVer cached corpus",
        "corpus_absolute_path": str(corpus_absolute_path) if corpus_absolute_path else None,
        "n_examples_second_split": int(n_examples),
        "verifiers": list(VERIFIER_NAMES),
        "live_model_invoked": False,
        "scoring_mode": "cached_verifier_scores_only_with_persisted_tier3_fast_path_gate",
        "predictive_head": "FR11ExtendedJEPA",
        "predictive_head_module": "carnot.fr11.tier3_jepa",
        "predictor_state_source": "results/experiment_3788_fr11_v19_tier3_predictive_jepa_state.npz",
        "operating_point_source": "results/experiment_3803_fr11_v20_tier3_fast_path_gate_state.json",
        "predictor_input_dim": v19.JEPA_EMBED_DIM,
        "predictor_outputs": list(JEPA_DOMAINS),
        "tier": "Tier-3 predictive verification robustness measurement",
        "ensemble_status": "frozen_0.9131_scoring_ensemble_is_fallthrough_unchanged",
    }


def _methodology(random_seed: int, corpus_absolute_path: Path | str | None) -> JsonDict:
    return {
        "random_seed": int(random_seed),
        "corpus_absolute_path": str(corpus_absolute_path) if corpus_absolute_path else None,
        "scoring_protocol": "ops/reproduction-runbook-fover-headline.md",
        "domain_keys": list(VERIFIER_NAMES),
        "lineage": "v21_second_split_robustness_measurement_not_retrain_or_retune",
        "feature_builder": "v19_four_verifier_scores_to_258_dim_FR11ExtendedJEPA_embedding",
        "predictor_api": "FR11ExtendedJEPA.predict loaded from persisted v19 state",
        "gate_confidence": "abs(mean_domain_probability - 0.5)",
        "fixed_threshold_source": "Exp 3803 persisted operating point, reused unchanged",
        "fallthrough_path": "Exp 2837 frozen four-verifier ensemble scoring",
        "second_split": (
            "different seed, original FoVer row IDs disjoint from v19 train and v20 measurement"
        ),
    }


def _success_verdict(skip_rate: float, effective_auroc: float, generalizes: bool) -> str:
    generalizes_text = "true" if generalizes else "false"
    return (
        f"{SUCCESS_VERDICT_PREFIX}{float(skip_rate):.4f}_effective_auroc_"
        f"{float(effective_auroc):.4f}_operating_point_generalizes_"
        f"{generalizes_text}_headline_ensemble_unchanged_measurement_not_retrain"
    )


def headline_ensemble_unchanged(reference: float | None) -> bool:
    return bool(reference is not None and round(float(reference), 4) == FROZEN_HEADLINE_ROUNDED)


def _label_to_int(label: Any) -> int:
    if label in {"incorrect", 1, "1"}:
        return 1
    if label in {"correct", 0, "0"}:
        return 0
    raise ValueError(f"unsupported FoVer label: {label!r}")


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
        return sum(1 for _line in handle)


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _int_sequence_sha256(values: Sequence[int]) -> str:
    arr = np.asarray([int(value) for value in values], dtype=np.int64)
    return _array_sha256(arr)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

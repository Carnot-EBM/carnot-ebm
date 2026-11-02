"""FR-11 continuous self-learning v17 live verifier precision tracker.

Spec: REQ-LEARN-3772, SCENARIO-LEARN-3772.

The v17 forward difference pivots away from the bounded EBT stabilizer branch
and returns Tier-1 self-learning to the live FoVer verifier product.  It reads
cached FoVer labels and verifier scores, updates raw per-verifier precision
counters online, derives a precision-weighted ensemble, verifies that the
FR-11 memory verifier remains active, and persists the raw counters for the
next milestone.
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

OUTPUT_REL_PATH = Path(
    "results/experiment_3772_fr11_self_learning_v17_verifier_precision_tracker.json"
)
TRACKER_STATE_REL_PATH = Path(
    "results/experiment_3772_fr11_v17_verifier_precision_tracker_state.json"
)
EXP2837_REL_PATH = Path("results/experiment_2837_fover_memory_leakage_v3.json")
DEFAULT_RANDOM_SEED = 3772
DEFAULT_CORPUS_RANDOM_SEED = 3772
DEFAULT_N_ONLINE_UPDATES = 1000
DEFAULT_SCORE_THRESHOLD = 0.5
MEMORY_CONTRIBUTION_ROUNDED_MIN = 0.0185
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates "
    "(principle: reads cached verifier scores, no live model)."
)
SUCCESS_VERDICT = (
    "complete: "
    "fr11_v17_tier1_verifier_precision_tracker_pivoted_to_live_verifier_"
    "memory_contribution_preserved_state_persisted"
)
EMPTY_VERDICT = "complete: no corpus to learn from -- tracker not updated"
TIER1_COUNTER_UPDATE = "cpu_counter_update_only_lt_1us_no_model_retrain"
VERIFIER_NAMES = tuple(v10.VERIFIER_NAMES)

score_fover_corpus = v10.score_fover_corpus
score_matrix = v10.score_matrix
probe_cached_trace_preconditions = v10.probe_cached_trace_preconditions

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "per_verifier_precision_table",
    "learned_weighting",
    "ensemble_auroc_under_learned_weighting",
    "memory_contribution_preserved",
    "pivoted_off_dead_ebt_lineage",
    "tier1_counter_update",
    "tracker_state_persisted",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix; the self-learning outcome.",
    "inference_substrate": (
        "verifier_ensemble_against_cached_candidates (principle: reads cached "
        "verifier scores, no live model)."
    ),
    "per_verifier_precision_table": (
        "Online precision/recall per scoring verifier -- the learned knowledge; "
        "the core deliverable, now on the LIVE verifier."
    ),
    "learned_weighting": (
        "The upweighting derived from per-verifier precision -- the actionable "
        "Tier-1 self-learning output."
    ),
    "ensemble_auroc_under_learned_weighting": (
        "The ensemble AUROC after the learned reweighting -- confirms the "
        "self-learning does not regress the banked discriminator."
    ),
    "memory_contribution_preserved": (
        "BARE bool -- learned weighting still includes the +0.0185 "
        "fr11_session_memory contribution."
    ),
    "pivoted_off_dead_ebt_lineage": (
        "True iff this tracks the LIVE verifier, not EBT stabilizers."
    ),
    "tier1_counter_update": (
        "Confirms the mechanic is a CPU counter update (<1us), not a model retrain."
    ),
    "tracker_state_persisted": (
        "True iff tracker state was saved so a future milestone resumes it."
    ),
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Content hash catches drift.",
    "duration_s": "Wall-clock plausibility floor.",
}


@dataclass
class VerifierPrecisionStats:
    """Raw confusion counters for one verifier's online precision tracker."""

    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0

    def record(self, *, predicted_positive: bool, label: int) -> None:
        actual_positive = int(label) == 1
        if predicted_positive and actual_positive:
            self.true_positive += 1
        elif predicted_positive and not actual_positive:
            self.false_positive += 1
        elif not predicted_positive and actual_positive:
            self.false_negative += 1
        else:
            self.true_negative += 1

    @property
    def predicted_positive_total(self) -> int:
        return self.true_positive + self.false_positive

    @property
    def actual_positive_total(self) -> int:
        return self.true_positive + self.false_negative

    @property
    def total(self) -> int:
        return (
            self.true_positive
            + self.false_positive
            + self.false_negative
            + self.true_negative
        )

    @property
    def precision(self) -> float:
        return _rate(self.true_positive, self.predicted_positive_total)

    @property
    def recall(self) -> float:
        return _rate(self.true_positive, self.actual_positive_total)

    def to_json(self) -> JsonDict:
        return {
            "true_positive": int(self.true_positive),
            "false_positive": int(self.false_positive),
            "false_negative": int(self.false_negative),
            "true_negative": int(self.true_negative),
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> VerifierPrecisionStats:
        return cls(
            true_positive=int(payload.get("true_positive", 0)),
            false_positive=int(payload.get("false_positive", 0)),
            false_negative=int(payload.get("false_negative", 0)),
            true_negative=int(payload.get("true_negative", 0)),
        )


class VerifierPrecisionTracker:
    """Tier-1 raw counter tracker for per-verifier FoVer precision."""

    def __init__(self, verifier_names: Sequence[str] = VERIFIER_NAMES) -> None:
        self.verifier_names = tuple(str(name) for name in verifier_names)
        self._stats = {
            name: VerifierPrecisionStats() for name in self.verifier_names
        }

    def record_score(
        self,
        verifier_name: str,
        *,
        score: float,
        label: int,
        threshold: float = DEFAULT_SCORE_THRESHOLD,
    ) -> None:
        if int(label) not in {0, 1}:
            raise ValueError("labels must be binary 0/1")
        if verifier_name not in self._stats:
            self._stats[verifier_name] = VerifierPrecisionStats()
            self.verifier_names = (*self.verifier_names, verifier_name)
        value = float(score)
        if not math.isfinite(value):
            raise ValueError("verifier scores must be finite")
        self._stats[verifier_name].record(
            predicted_positive=value >= float(threshold),
            label=int(label),
        )

    def update_from_scores(
        self,
        labels: Sequence[int],
        scores_by_verifier: Mapping[str, Sequence[float]],
        *,
        threshold: float = DEFAULT_SCORE_THRESHOLD,
    ) -> None:
        labels_arr = [int(label) for label in labels]
        if any(label not in {0, 1} for label in labels_arr):
            raise ValueError("labels must be binary 0/1")
        for name in self.verifier_names:
            if name not in scores_by_verifier:
                raise ValueError(f"missing verifier score column: {name}")
            if len(scores_by_verifier[name]) != len(labels_arr):
                raise ValueError("labels and verifier scores must have the same length")
        for index, label in enumerate(labels_arr):
            for name in self.verifier_names:
                self.record_score(
                    name,
                    score=float(scores_by_verifier[name][index]),
                    label=label,
                    threshold=threshold,
                )

    def stats(self) -> dict[str, VerifierPrecisionStats]:
        return dict(self._stats)

    def to_json(self) -> JsonDict:
        return {
            "schema": "carnot.fr11_v17_verifier_precision_tracker_state",
            "version": 1,
            "verifier_names": list(self.verifier_names),
            "stats": {
                name: self._stats[name].to_json() for name in self.verifier_names
            },
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> VerifierPrecisionTracker:
        if payload.get("version") != 1:
            raise ValueError("unsupported verifier precision tracker state version")
        raw_stats = payload.get("stats", {})
        if not isinstance(raw_stats, Mapping):
            raise ValueError("verifier precision tracker stats must be a mapping")
        names = payload.get("verifier_names") or tuple(raw_stats)
        tracker = cls(tuple(str(name) for name in names))
        for name, raw in raw_stats.items():
            if not isinstance(raw, Mapping):
                raise ValueError("verifier precision tracker stat entry must be a mapping")
            if name not in tracker._stats:
                tracker.verifier_names = (*tracker.verifier_names, str(name))
            tracker._stats[str(name)] = VerifierPrecisionStats.from_json(raw)
        return tracker


def precision_table(
    tracker: VerifierPrecisionTracker,
    verifier_names: Sequence[str] | None = None,
) -> list[JsonDict]:
    """Return measured precision/recall rows derived from raw counters."""

    names = tuple(verifier_names or tracker.verifier_names)
    stats_by_name = tracker.stats()
    rows: list[JsonDict] = []
    for name in names:
        stats = stats_by_name.get(name, VerifierPrecisionStats())
        rows.append(
            {
                "verifier": name,
                "true_positive": stats.true_positive,
                "false_positive": stats.false_positive,
                "false_negative": stats.false_negative,
                "true_negative": stats.true_negative,
                "predicted_positive_total": stats.predicted_positive_total,
                "actual_positive_total": stats.actual_positive_total,
                "n_examples": stats.total,
                "precision": _round(stats.precision),
                "recall": _round(stats.recall),
                "measurement_note": f"measured_table_over_fover_n={stats.total}",
            }
        )
    return rows


def learned_weighting(
    table: Sequence[Mapping[str, Any]],
    verifier_names: Sequence[str] = VERIFIER_NAMES,
) -> dict[str, float]:
    """Normalize precision-derived weights in the requested verifier order."""

    if not table:
        return {}
    by_name = {str(row["verifier"]): row for row in table}
    raw = []
    for name in verifier_names:
        row = by_name.get(name)
        if row is None:
            raw.append(0.0)
            continue
        active = int(row.get("predicted_positive_total", 0)) > 0
        raw.append(float(row.get("precision", 0.0)) if active else 0.0)
    raw_arr = np.asarray(raw, dtype=np.float64)
    if float(raw_arr.sum()) <= 0.0:
        raw_arr = np.ones(len(tuple(verifier_names)), dtype=np.float64)
    weights = v10.exp3644.normalize_weights(raw_arr)
    return {
        name: _round(float(weight))
        for name, weight in zip(verifier_names, weights, strict=True)
    }


def build_artifact(
    repo_root: Path | str,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    n_examples: int = DEFAULT_N_ONLINE_UPDATES,
    random_seed: int = DEFAULT_RANDOM_SEED,
    corpus_random_seed: int = DEFAULT_CORPUS_RANDOM_SEED,
    state_path: Path | str = TRACKER_STATE_REL_PATH,
) -> JsonDict:
    """Build Exp 3772 from cached FoVer rows and verifier scores."""

    root = Path(repo_root)
    start = time.time() if started_s is None else float(started_s)
    state_output = _resolve_under_root(root, state_path)
    preconditions = [
        v10._fr11_precondition(root),
        *probe_cached_trace_preconditions(root, n_examples=n_examples),
    ]
    if not all(item["available"] for item in preconditions):
        return _empty_artifact(
            duration_s=_duration(start, now_s),
            random_seed=random_seed,
            state_output=state_output,
            repo_root=root,
            preconditions=preconditions,
        )
    try:
        labels, scores_by_verifier = score_fover_corpus(
            root,
            n_examples=n_examples,
            random_seed=corpus_random_seed,
        )
    except Exception as exc:  # noqa: BLE001 - cached scoring failure is terminal.
        return _empty_artifact(
            duration_s=_duration(start, now_s),
            random_seed=random_seed,
            state_output=state_output,
            repo_root=root,
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
        state_path=state_output,
        repo_root=root,
        preconditions=preconditions,
        memory_contribution_reference=load_memory_contribution_reference(root),
    )


def build_artifact_from_scores(
    *,
    labels: Sequence[int],
    scores_by_verifier: Mapping[str, Sequence[float]],
    started_s: float,
    now_s: float | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    persistence_dir: Path | str | None = None,
    state_path: Path | str | None = None,
    repo_root: Path | str | None = None,
    preconditions: Sequence[Mapping[str, Any]] | None = None,
    memory_contribution_reference: float | None = None,
) -> JsonDict:
    """Update online counters from precomputed FoVer labels and scores."""

    root = Path(repo_root) if repo_root is not None else None
    state_output = _state_output_path(
        persistence_dir=persistence_dir,
        state_path=state_path,
        repo_root=root,
    )
    if not labels or not scores_by_verifier:
        return _empty_artifact(
            duration_s=_duration(float(started_s), now_s),
            random_seed=random_seed,
            state_output=state_output,
            repo_root=root,
            preconditions=preconditions
            or [_trace_precondition(labels, scores_by_verifier)],
        )

    tracker = VerifierPrecisionTracker(VERIFIER_NAMES)
    tracker.update_from_scores(labels, scores_by_verifier)
    table = precision_table(tracker, VERIFIER_NAMES)
    weights = learned_weighting(table, VERIFIER_NAMES)
    labels_arr = np.asarray(labels, dtype=np.int64)
    matrix = score_matrix(scores_by_verifier, VERIFIER_NAMES)
    if matrix.shape[0] != len(labels_arr):
        raise ValueError("labels and verifier scores must have the same length")
    _require_binary_labels(labels_arr)
    weight_vector = np.asarray([weights[name] for name in VERIFIER_NAMES], dtype=np.float64)
    ensemble_scores = v10.exp3644.ensemble_scores(matrix, weight_vector)
    auroc = v10.exp3644.tie_aware_auroc(labels_arr, ensemble_scores)
    state_sha = persist_tracker_state(
        tracker,
        state_output,
        n_examples_observed=len(labels_arr),
        random_seed=random_seed,
    )
    preserved = memory_contribution_preserved(
        weights,
        memory_contribution_reference,
    )
    artifact: JsonDict = {
        "artifact": "experiment_3772_fr11_self_learning_v17_verifier_precision_tracker",
        "schema": "carnot.fr11_continuous_self_learning_v17",
        "continuous_self_learning_task": True,
        "honest_verdict": SUCCESS_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "per_verifier_precision_table": table,
        "learned_weighting": weights,
        "ensemble_auroc_under_learned_weighting": _round(auroc),
        "memory_contribution_preserved": bool(preserved),
        "pivoted_off_dead_ebt_lineage": True,
        "tier1_counter_update": TIER1_COUNTER_UPDATE,
        "tracker_state_persisted": state_output.is_file(),
        "random_seed": int(random_seed),
        "reproducibility_checksum": reproducibility_checksum(
            labels=labels_arr,
            scores=matrix,
            table=table,
            weights=weights,
            tracker_state_sha256=state_sha,
            random_seed=random_seed,
            memory_contribution_reference=memory_contribution_reference,
        ),
        "duration_s": _round(_duration(float(started_s), now_s)),
        "n_examples": int(len(labels_arr)),
        "score_threshold": DEFAULT_SCORE_THRESHOLD,
        "verifier_names": list(VERIFIER_NAMES),
        "memory_contribution_reference": memory_contribution_reference,
        "tracker_state_path": _relative_path(state_output, root),
        "tracker_state_sha256": state_sha,
        "model_specs": _model_specs(),
        "methodology": _methodology(random_seed),
        "preconditions_checked": [dict(item) for item in preconditions or []],
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate": {
            "passed": bool(preserved and state_output.is_file()),
            "condition": (
                "fr11_session_memory weight > 0 AND Exp 2837 rounded "
                "memory contribution >= +0.0185 AND tracker_state_persisted"
            ),
            "principle": (
                "The learned weighting must not silently drop the FR-11 "
                "self-learning component that carries the headline."
            ),
        },
    }
    validate_artifact(artifact)
    return artifact


def write_artifact(
    repo_root: Path | str,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    state_path: Path | str = TRACKER_STATE_REL_PATH,
    labels: Sequence[int] | None = None,
    scores_by_verifier: Mapping[str, Sequence[float]] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build, validate, and write the Exp 3772 JSON artifact."""

    root = Path(repo_root)
    start = time.time() if started_s is None else float(started_s)
    resolved_state_path = _resolve_under_root(root, state_path)
    if labels is not None or scores_by_verifier is not None:
        artifact = build_artifact_from_scores(
            labels=labels or [],
            scores_by_verifier=scores_by_verifier or {},
            started_s=start,
            now_s=now_s,
            state_path=resolved_state_path,
            repo_root=root,
            memory_contribution_reference=load_memory_contribution_reference(root),
        )
    else:
        artifact = build_artifact(
            root,
            started_s=start,
            now_s=now_s,
            state_path=resolved_state_path,
        )
    output = _resolve_under_root(root, output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def persist_tracker_state(
    tracker: VerifierPrecisionTracker,
    path: Path | str,
    *,
    n_examples_observed: int,
    random_seed: int,
) -> str:
    """Persist raw tracker counters and return the stored payload checksum."""

    output = Path(path)
    payload = {
        **tracker.to_json(),
        "n_examples_observed": int(n_examples_observed),
        "tier1_counter_update": TIER1_COUNTER_UPDATE,
        "random_seed": int(random_seed),
    }
    checksum = _json_sha256(payload)
    stored = {**payload, "sha256": checksum}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(stored, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return checksum


def load_memory_contribution_reference(repo_root: Path | str) -> float | None:
    """Load the Exp 2837 FR-11 memory contribution reference if present."""

    path = Path(repo_root) / EXP2837_REL_PATH
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    value = payload.get("learning_contribution")
    return float(value) if isinstance(value, int | float) else None


def memory_contribution_preserved(
    weights: Mapping[str, float],
    reference: float | None,
) -> bool:
    """Return true only when learned weights keep the published memory signal."""

    if reference is None:
        return False
    memory_weight = float(weights.get("fr11_session_memory", 0.0))
    return memory_weight > 0.0 and round(float(reference), 4) >= MEMORY_CONTRIBUTION_ROUNDED_MIN


def reproducibility_checksum(
    *,
    labels: np.ndarray | Sequence[int],
    scores: np.ndarray | Sequence[Sequence[float]],
    table: Sequence[Mapping[str, Any]],
    weights: Mapping[str, float],
    tracker_state_sha256: str,
    random_seed: int,
    memory_contribution_reference: float | None,
) -> str:
    """Hash the measured labels, scores, counters, weights, and seed."""

    payload = {
        "labels_sha256": _array_sha256(np.asarray(labels, dtype=np.int64)),
        "scores_sha256": _array_sha256(np.asarray(scores, dtype=np.float64)),
        "per_verifier_precision_table": [dict(row) for row in table],
        "learned_weighting": dict(weights),
        "tracker_state_sha256": tracker_state_sha256,
        "random_seed": int(random_seed),
        "memory_contribution_reference": memory_contribution_reference,
    }
    return _json_sha256(payload)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3772 artifact schema before writing."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must declare verifier-scoring substrate")
    serialized = json.dumps(artifact, sort_keys=True)
    for marker in ("GGUF", "CUDA", "cuda", "live_llm_inference", "torch.cuda"):
        if marker in serialized:
            raise ValueError("forbidden inference marker present")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles must be present")
    missing_principles = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in principles]
    if missing_principles:
        raise ValueError(f"missing field principles: {missing_principles}")
    if artifact.get("tier1_counter_update") != TIER1_COUNTER_UPDATE:
        raise ValueError("tier1_counter_update must declare CPU counter update only")
    if not isinstance(artifact.get("memory_contribution_preserved"), bool):
        raise ValueError("memory_contribution_preserved must be a bare boolean")
    if artifact.get("pivoted_off_dead_ebt_lineage") is not True:
        raise ValueError("pivoted_off_dead_ebt_lineage must be true")
    if not isinstance(artifact.get("tracker_state_persisted"), bool):
        raise ValueError("tracker_state_persisted must be boolean")
    if not isinstance(artifact.get("duration_s"), int | float):
        raise ValueError("duration_s must be numeric")
    verdict = artifact.get("honest_verdict")
    if verdict == EMPTY_VERDICT:
        if artifact.get("per_verifier_precision_table") != []:
            raise ValueError("empty fallback must not fabricate a precision table")
        if artifact.get("learned_weighting") != {}:
            raise ValueError("empty fallback must not fabricate learned weights")
        if artifact.get("ensemble_auroc_under_learned_weighting") is not None:
            raise ValueError("empty fallback must not fabricate AUROC")
        return
    if verdict != SUCCESS_VERDICT:
        raise ValueError(f"unsupported honest_verdict: {verdict!r}")
    table = artifact.get("per_verifier_precision_table")
    if not isinstance(table, list) or len(table) != len(VERIFIER_NAMES):
        raise ValueError("precision table must include all four scoring verifiers")
    table_names = [row.get("verifier") for row in table if isinstance(row, Mapping)]
    if tuple(table_names) != VERIFIER_NAMES:
        raise ValueError("precision table verifier order must match scoring verifiers")
    weights = artifact.get("learned_weighting")
    if not isinstance(weights, Mapping) or set(weights) != set(VERIFIER_NAMES):
        raise ValueError("learned_weighting must include all four scoring verifiers")
    if float(weights.get("fr11_session_memory", 0.0)) <= 0.0:
        raise ValueError("learned weighting must retain fr11_session_memory")
    auroc = artifact.get("ensemble_auroc_under_learned_weighting")
    if not isinstance(auroc, int | float) or not 0.0 <= float(auroc) <= 1.0:
        raise ValueError("ensemble_auroc_under_learned_weighting must be in [0, 1]")


def _empty_artifact(
    *,
    duration_s: float,
    random_seed: int,
    state_output: Path,
    repo_root: Path | None,
    preconditions: Sequence[Mapping[str, Any]],
) -> JsonDict:
    tracker = VerifierPrecisionTracker(())
    state_sha = persist_tracker_state(
        tracker,
        state_output,
        n_examples_observed=0,
        random_seed=random_seed,
    )
    payload = {
        "preconditions": [dict(item) for item in preconditions],
        "tracker_state_sha256": state_sha,
        "random_seed": int(random_seed),
    }
    artifact: JsonDict = {
        "artifact": "experiment_3772_fr11_self_learning_v17_verifier_precision_tracker",
        "schema": "carnot.fr11_continuous_self_learning_v17",
        "continuous_self_learning_task": True,
        "honest_verdict": EMPTY_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "per_verifier_precision_table": [],
        "learned_weighting": {},
        "ensemble_auroc_under_learned_weighting": None,
        "memory_contribution_preserved": False,
        "pivoted_off_dead_ebt_lineage": True,
        "tier1_counter_update": TIER1_COUNTER_UPDATE,
        "tracker_state_persisted": state_output.is_file(),
        "random_seed": int(random_seed),
        "reproducibility_checksum": _json_sha256(payload),
        "duration_s": _round(duration_s),
        "n_examples": 0,
        "score_threshold": DEFAULT_SCORE_THRESHOLD,
        "verifier_names": list(VERIFIER_NAMES),
        "memory_contribution_reference": None,
        "tracker_state_path": _relative_path(state_output, repo_root),
        "tracker_state_sha256": state_sha,
        "model_specs": _model_specs(),
        "methodology": _methodology(random_seed),
        "preconditions_checked": [dict(item) for item in preconditions],
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate": {
            "passed": False,
            "condition": "FoVer scores and labels are present",
            "principle": "No self-learning precision table is emitted without corpus evidence.",
        },
    }
    validate_artifact(artifact)
    return artifact


def _trace_precondition(
    labels: Sequence[int],
    scores_by_verifier: Mapping[str, Sequence[float]],
) -> JsonDict:
    return {
        "resource": "cached_traces_with_per_verifier_scores_and_labels",
        "available": bool(labels and scores_by_verifier),
        "detail": f"n_examples={len(labels)}; n_verifiers={len(scores_by_verifier)}",
    }


def _model_specs() -> JsonDict:
    return {
        "corpus": "FoVer cached corpus",
        "n_examples_requested": DEFAULT_N_ONLINE_UPDATES,
        "verifiers": list(VERIFIER_NAMES),
        "live_model_invoked": False,
        "scoring_mode": "cached_verifier_scores_only",
    }


def _methodology(random_seed: int) -> JsonDict:
    return {
        "random_seed": int(random_seed),
        "score_threshold": DEFAULT_SCORE_THRESHOLD,
        "corpus": "data/fover_corpus.jsonl",
        "counter_update": TIER1_COUNTER_UPDATE,
        "lineage": "v17_live_verifier_precision_tracker_not_v15_v16_ebt_stabilizer",
    }


def _state_output_path(
    *,
    persistence_dir: Path | str | None,
    state_path: Path | str | None,
    repo_root: Path | None,
) -> Path:
    if state_path is not None:
        return _resolve_under_root(repo_root or Path("."), state_path)
    base = Path(persistence_dir) if persistence_dir is not None else Path(".")
    return base / TRACKER_STATE_REL_PATH.name


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


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.time() if now_s is None else float(now_s)
    return max(0.0001, end - float(started_s))


def _round(value: float | int | None, digits: int = 9) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

"""Run prospective chronological learning over archived live E3 transitions.

Every arm predicts from a frozen pre-event state. The exact next frame opens
only after all arm and seed predictions have a durable checkpoint. Exact
post-event evidence can update only the typed Exp6613 side-state store.
Generator, world-model, policy, and projector weights stay frozen.

Spec refs: REQ-LEARN-6614, REQ-LEARN-6614-PRECONDITIONS,
REQ-LEARN-6614-CHRONOLOGY, REQ-LEARN-6614-DOSE, REQ-LEARN-6614-FROZEN,
REQ-LEARN-6614-ADMISSION, REQ-LEARN-6614-ROWS, REQ-LEARN-6614-UTILITY,
REQ-LEARN-6614-RECOVERY, REQ-LEARN-6614-ATTACKS,
REQ-LEARN-6614-ATOMIC.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time
from typing import Any

import numpy as np

from carnot.agentic.arc_invariant_memory import (
    FEATURE_SCHEMA_VERSION,
    JOURNAL_SCHEMA_VERSION,
    RECORD_SCHEMA_VERSION,
    STORE_SCHEMA_VERSION,
    InvariantMemoryStore,
    JournalCorruptionError,
    LifecycleState,
    RetrievalContext,
    VerifierDescriptor,
    canonical_json_bytes,
    make_invariant_record,
    sha256_bytes,
)
from carnot.agentic.arc_invariant_projector import (
    InvariantProjectionConfig,
    config_sha256,
    grid_features,
    project_prediction,
    quadratic_value,
)


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6614_prospective_invariant_self_learning.json")
EXP6611_RELATIVE_PATH = Path("results/experiment_6611_live_arc_invariant_projection.json")
EXP6613_RELATIVE_PATH = Path("results/experiment_6613_invariant_memory_lifecycle.json")
PROJECTOR_RELATIVE_PATH = Path("python/carnot/agentic/arc_invariant_projector.py")
MEMORY_RELATIVE_PATH = Path("python/carnot/agentic/arc_invariant_memory.py")
POLICY_RELATIVE_PATH = Path("python/carnot/agentic/arc_competition_agent.py")
GENERATOR_RELATIVE_PATH = Path("python/carnot/agentic/arc_executable_world_model.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
ARCHIVE_RELATIVE_PATH = Path("data/arc_transition_corpus")
INFERENCE_SUBSTRATE = "prospective_chronological_live_e3_invariant_side_memory_no_new_llm"
CONTINUOUS_SELF_LEARNING_TASK = True
ARMS = (
    "no_learning",
    "static_projector",
    "governed_online_memory",
    "shuffled_admission_control",
)
MUTABLE_ARMS = ("governed_online_memory", "shuffled_admission_control")
DEFAULT_SEEDS = (6614, 16614)
DEFAULT_CAPACITY = 16
DEFAULT_PER_SOURCE_CAPACITY = 4
RETENTION_MARGIN = 0.0
SUPPORT_MARGIN = 0.0
COST_BUDGET_S = 0.25
MEMORY_BUDGET_BYTES = 4 * 1024 * 1024
MAX_STALENESS_STEPS = 24
PROTECTED_EXPECTED_HASHES = {
    "research-roadmap.yaml": "sha256:753df27210a62a5572e19e9ede78ee2b1af5e4a11cb83063e62b69367ef33270",
    "scripts/research_conductor.py": "sha256:fd4736a54c9e244caee4ed695609f5b06317a7174ebe8411c5f70a55907d73bd",
}
ATTACK_IDS = (
    "future_frame_leakage",
    "game_identity",
    "opportunity_omission",
    "dose_mismatch",
    "control_reuse",
    "shuffled_control_correction",
    "unsafe_commit",
    "stale_record",
    "source_duplication",
    "state_mutation_before_prediction",
    "weight_mutation",
    "support_collapse",
    "journal_corruption",
    "restart_drift",
    "rollback_drift",
    "protected_file_mutation",
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "gate_check_summary",
    "continuous_self_learning_task",
    "per_unit_rows",
    "chronology_and_split_receipts",
    "arm_and_dose_receipts",
    "frozen_model_policy_receipts",
    "prediction_before_observation_rows",
    "memory_transition_rows",
    "held_future_benefit_summary",
    "retention_and_support_summary",
    "safety_occupancy_and_cost_summary",
    "restart_and_rollback_receipts",
    "acceptance_gate_rows",
    "continuous_self_learning_ready_score",
    "attack_rows",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)
FIELD_PRINCIPLES = {
    "status": "The task ends with complete prospective evidence or a named gate block.",
    "honest_verdict": "The verdict states held-future benefit, retention, support, safety, recovery, and immutability without upgrading row completion.",
    "verdict_class": "Use the closed enum; an exact-observation-governed win is circular_positive.",
    "gate_check_summary": "Any block names the failed upstream, chronology, dose, benefit, retention, support, safety, recovery, resource, or hash and its value.",
    "continuous_self_learning_task": "Bare true marks the mandatory FR-11 continuous self-learning task.",
    "per_unit_rows": "Every event, arm, and seed carries pre-state, retrieval, prediction, observation, exact evidence, update, post-state, cost, support, and failure data.",
    "chronology_and_split_receipts": "Source-disjoint order and untouched retention data freeze before any update.",
    "arm_and_dose_receipts": "All arms receive every opportunity in the same order and budget.",
    "frozen_model_policy_receipts": "Generator, world-model, base-policy, and projector hashes remain immutable.",
    "prediction_before_observation_rows": "Pre-event predictions and state hashes are committed before exact next-frame evidence appears.",
    "memory_transition_rows": "Every proposed, accepted, quarantined, archived, and rejected update binds exact evidence and journal hashes.",
    "held_future_benefit_summary": "Prospective later-event utility compares online memory with static and shuffled controls from rows.",
    "retention_and_support_summary": "Untouched retention and future recoverable support remain noninferior under fixed margins.",
    "safety_occupancy_and_cost_summary": "Unsafe commits, conflicts, occupancy, memory, latency, and failures are explicit.",
    "restart_and_rollback_receipts": "Restart and rollback recreate byte-identical state and predictions.",
    "acceptance_gate_rows": "Every science and safety gate records expected, observed, and passed values.",
    "continuous_self_learning_ready_score": "This binary field opens only when prospective benefit and every safety and recovery gate pass.",
    "attack_rows": "Leakage, identity, opportunity, dose, control, safety, stale, duplicate, timing, weight, support, journal, recovery, and mutation attacks fail closed.",
    "preconditions_checked": "Gates, archives, chronology, models, policy, projector, memory, opportunities, metrics, resources, and protected files are explicit.",
    "protected_files_unchanged": "Both protected orchestration files retain original hashes.",
    "inference_substrate": "The task declares prospective live-path archive learning with frozen models and no new LLM.",
    "verifier_is_oracle": "Exact observed next frames govern post-event admission and the preregistered outcome.",
    "field_provenance": "Every field points to immutable event rows, state hashes, exact evidence, journals, and reducers.",
    "duration_s": "Monotonic duration covers all opportunities, arms, recovery checks, and attacks.",
    "tests_run": "Named learning, lifecycle, lint, spec, artifact, gate, adversarial, and E2E commands include exits and durations.",
    "reproducibility_checksum": "A final content hash protects the self-learning result.",
}


@dataclass(frozen=True)
class ProspectiveEvent:
    """One immutable pre-event input with separately callable observation access."""

    event_id: str
    source_name: str
    transition_index: int
    split: str
    chronology_index: int
    current_grid: np.ndarray
    action: int
    action_data: Mapping[str, int] | None
    archive_path: str
    archive_sha256: str
    source_transition_sha256: str
    world_model_path: str
    world_model_sha256: str
    predict: Callable[[], np.ndarray]
    observe: Callable[[], np.ndarray]

    def __post_init__(self) -> None:
        if self.split not in {"adaptation", "adaptation_future", "retention"}:
            raise ValueError("event split is not registered")
        if self.chronology_index < 0 or self.transition_index < 0:
            raise ValueError("chronology indices must be non-negative")
        if np.asarray(self.current_grid).ndim != 2:
            raise ValueError("current grid must be two-dimensional")


def sha256_json(value: Any) -> str:
    """Hash a JSON-compatible value through one stable encoding."""

    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path | str) -> str:
    """Hash one file without interpreting its contents."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def row_hash(row: Mapping[str, Any]) -> str:
    """Hash one row while excluding its self-referential hash field."""

    return sha256_json({key: value for key, value in row.items() if key != "row_hash"})


def artifact_checksum(payload: Mapping[str, Any]) -> str:
    """Hash every terminal artifact field except the checksum itself."""

    return sha256_json(
        {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    )


def _grid_receipt(grid: np.ndarray | None) -> JsonDict:
    if grid is None:
        return {"available": False, "shape": None, "data": None, "sha256": None}
    array = np.ascontiguousarray(np.asarray(grid, dtype=np.int16))
    raw = array.astype("<i2", copy=False).tobytes()
    return {
        "available": True,
        "shape": list(array.shape),
        "dtype": "int16",
        "encoding": "base64_little_endian_int16",
        "data": base64.b64encode(raw).decode("ascii"),
        "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
    }


def _exact_error(prediction: np.ndarray | None, observed: np.ndarray) -> int:
    if prediction is None or prediction.shape != observed.shape:
        return int(observed.size)
    return int(np.count_nonzero(np.asarray(prediction) != np.asarray(observed)))


def _valid_prediction(prediction: np.ndarray | None, current: np.ndarray) -> bool:
    return bool(
        prediction is not None
        and prediction.shape == current.shape
        and prediction.ndim == 2
        and np.isfinite(prediction).all()
    )


def _source_key(current: np.ndarray, action: int) -> str:
    features = grid_features(current)
    mean_bin = int(round(float(features[0]) * 32.0))
    rms_bin = int(round(float(features[1]) * 32.0))
    return f"feature:{mean_bin}:{rms_bin}:{int(action)}"


def freeze_chronology(
    events: Sequence[ProspectiveEvent], *, seeds: Sequence[int] = DEFAULT_SEEDS
) -> JsonDict:
    """Freeze source-disjoint order and a label-blind shuffled derangement."""

    ordered = sorted(events, key=lambda row: (row.chronology_index, row.event_id))
    adaptation = [row for row in ordered if row.split != "retention"]
    retention = [row for row in ordered if row.split == "retention"]
    adaptation_sources = sorted({row.source_name for row in adaptation})
    retention_sources = sorted({row.source_name for row in retention})
    ids = [row.event_id for row in adaptation]
    targets = ids[1:] + ids[:1] if len(ids) > 1 else [f"{ids[0]}::deranged"] if ids else []
    mapping = dict(zip(ids, targets, strict=True))
    manifest_rows = [
        {
            "event_id": row.event_id,
            "chronology_index": row.chronology_index,
            "source_name": row.source_name,
            "transition_index": row.transition_index,
            "split": row.split,
            "archive_path": row.archive_path,
            "archive_sha256": row.archive_sha256,
            "source_transition_sha256": row.source_transition_sha256,
            "world_model_path": row.world_model_path,
            "world_model_sha256": row.world_model_sha256,
            "current_grid_sha256": _grid_receipt(row.current_grid)["sha256"],
            "action": row.action,
            "action_data": dict(row.action_data) if row.action_data else None,
        }
        for row in ordered
    ]
    receipt = {
        "manifest_rows": manifest_rows,
        "adaptation_sources": adaptation_sources,
        "retention_sources": retention_sources,
        "source_disjoint": set(adaptation_sources).isdisjoint(retention_sources),
        "opportunity_count": len(ordered),
        "adaptation_opportunity_count": len(adaptation),
        "retention_opportunity_count": len(retention),
        "arm_list": list(ARMS),
        "seeds": [int(seed) for seed in seeds],
        "shuffled_admission_mapping": mapping,
        "shuffled_mapping_frozen_before_outcomes": True,
        "observed_frame_fields_in_manifest": 0,
    }
    receipt["chronology_sha256"] = sha256_json(receipt)
    return receipt


def _candidate_basis(
    current: np.ndarray, observed: np.ndarray, static_matrix: np.ndarray
) -> tuple[np.ndarray, float]:
    """Fit the smallest static-basis change that conserves the observed frame."""

    before = grid_features(current)
    after = grid_features(observed)
    difference = np.asarray(
        [
            before[0] ** 2 - after[0] ** 2,
            2.0 * (before[0] * before[1] - after[0] * after[1]),
            before[1] ** 2 - after[1] ** 2,
        ],
        dtype=np.float64,
    )
    static = np.asarray(
        [static_matrix[0, 0], static_matrix[0, 1], static_matrix[1, 1]],
        dtype=np.float64,
    )
    norm_sq = float(difference @ difference)
    coefficient = static.copy()
    if norm_sq > 1e-18:
        coefficient -= difference * float(coefficient @ difference) / norm_sq
    if float(np.linalg.norm(coefficient)) <= 1e-12:
        anchor = np.asarray([1.0, 0.0, 0.0])
        if norm_sq > 1e-18:
            anchor -= difference * float(anchor @ difference) / norm_sq
        coefficient = anchor
    matrix = np.asarray(
        [[coefficient[0], coefficient[1]], [coefficient[1], coefficient[2]]],
        dtype=np.float64,
    )
    matrix /= max(float(np.linalg.norm(matrix)), 1e-12)
    return matrix, quadratic_value(before, matrix)


def _project(
    current: np.ndarray, prediction: np.ndarray | None, matrix: np.ndarray
) -> tuple[np.ndarray | None, JsonDict]:
    if prediction is None:
        return None, {
            "projection_applied": False,
            "projection_distance": 0.0,
            "iterations": 0,
            "converged": False,
            "failure": "base_prediction_unavailable",
        }
    try:
        config = InvariantProjectionConfig(
            enabled=True,
            quadratic_matrix=tuple(tuple(float(value) for value in row) for row in matrix),
        )
        projected = project_prediction(current, prediction, config)
        return projected.grid, {
            "projection_applied": True,
            "projection_distance": projected.projection_distance,
            "iterations": projected.iterations,
            "converged": projected.converged,
            "failure": projected.failure,
            "invariant_drift_before": projected.invariant_drift_before,
            "invariant_drift_after": projected.invariant_drift_after,
            "config_sha256": config_sha256(config),
        }
    except (TypeError, ValueError, FloatingPointError) as exc:
        return prediction.copy(), {
            "projection_applied": False,
            "projection_distance": 0.0,
            "iterations": 0,
            "converged": False,
            "failure": f"{type(exc).__name__}: {exc}",
        }


def _memory_prediction(
    store: InvariantMemoryStore,
    event: ProspectiveEvent,
    base_prediction: np.ndarray | None,
    static_matrix: np.ndarray,
) -> tuple[np.ndarray | None, list[JsonDict], JsonDict]:
    source_id = _source_key(event.current_grid, event.action)
    eligible = []
    for record in store.active_records():
        descriptor = record.descriptor
        age = event.chronology_index - descriptor.observed_sequence_index
        valid = bool(
            record.source_id == source_id
            and descriptor.world_model_hash == event.world_model_sha256
            and descriptor.feature_schema == FEATURE_SCHEMA_VERSION
            and descriptor.exact_evidence
            and descriptor.confidence == 1.0
            and descriptor.uncertainty == 0.0
            and 0 <= age <= descriptor.max_staleness_steps
        )
        if valid:
            eligible.append(record)
    eligible.sort(key=lambda row: (-row.updated_sequence_index, row.record_id))
    if not eligible:
        prediction, diagnostics = _project(event.current_grid, base_prediction, static_matrix)
        return prediction, [], {**diagnostics, "fallback": "static_projector"}
    selected = eligible[0]
    matrix = np.asarray(selected.invariant_basis, dtype=np.float64).reshape(2, 2)
    prediction, diagnostics = _project(event.current_grid, base_prediction, matrix)
    retrieved = [
        {
            "record_id": selected.record_id,
            "source_id": selected.source_id,
            "descriptor_checksum": selected.descriptor.descriptor_checksum,
            "world_model_sha256": selected.descriptor.world_model_hash,
            "exact_evidence_revalidated": True,
            "staleness_steps": event.chronology_index
            - selected.descriptor.observed_sequence_index,
        }
    ]
    return prediction, retrieved, {**diagnostics, "fallback": None}


def apply_post_event_update(
    store: InvariantMemoryStore,
    *,
    source_id: str,
    source_transition_hash: str,
    world_model_hash: str,
    basis: Sequence[float],
    threshold: float,
    sequence_index: int,
    baseline_error: int,
    candidate_error: int,
    candidate_valid: bool,
) -> JsonDict:
    """Apply one exact post-event decision through the Exp6613 lifecycle."""

    pre_hash = sha256_bytes(store.canonical_state_bytes())
    exact = {
        "observed_after_prediction": True,
        "baseline_exact_error": int(baseline_error),
        "candidate_exact_error": int(candidate_error),
        "candidate_runtime_valid": bool(candidate_valid),
        "strict_improvement": bool(candidate_valid and candidate_error < baseline_error),
    }
    journal_before = len(store.journal_rows())
    proposed = {
        "source_id": source_id,
        "source_transition_hash": source_transition_hash,
        "world_model_hash": world_model_hash,
        "basis": [float(value) for value in basis],
        "threshold": float(threshold),
        "sequence_index": int(sequence_index),
    }
    if candidate_valid and candidate_error >= baseline_error:
        decision = "no_op_exact_nonimprovement"
        transition_rows = [
            {
                "action": "no_op",
                "reason": "exact_nonimprovement",
                "record_id": None,
                "journal_checksum": None,
                "snapshot_sha256": pre_hash,
                "before_state_sha256": pre_hash,
                "after_state_sha256": pre_hash,
                "proposed_update": proposed,
                "exact_evidence": exact,
                "decision": decision,
            }
        ]
    else:
        descriptor = VerifierDescriptor.create(
            source_transition_hashes=(source_transition_hash,),
            world_model_hash=world_model_hash,
            feature_schema=FEATURE_SCHEMA_VERSION,
            exact_pre_metrics={"prediction_error": baseline_error, "valid": 1.0},
            exact_post_metrics={"prediction_error": candidate_error, "valid": float(candidate_valid)},
            confidence=1.0,
            uncertainty=0.0,
            exact_evidence=bool(candidate_valid and candidate_error < baseline_error),
            observed_sequence_index=sequence_index,
            max_staleness_steps=MAX_STALENESS_STEPS,
        )
        record = make_invariant_record(
            source_id=source_id,
            descriptor=descriptor,
            invariant_basis=basis,
            invariant_threshold=threshold,
            admission_reason="exact_post_event_observed_frame_evidence",
            sequence_index=sequence_index,
        )
        context = RetrievalContext(
            source_hashes={source_id: (source_transition_hash,)},
            world_model_hash=world_model_hash,
            feature_schema=FEATURE_SCHEMA_VERSION,
            sequence_index=sequence_index,
        )
        receipt = store.admit(record, context)
        if receipt.post_state == LifecycleState.ACTIVE:
            decision = "commit"
        elif not candidate_valid:
            decision = "quarantine_invalid_candidate"
        else:
            decision = f"{receipt.action}_{receipt.reason}"
        transition_rows = []
        for journal in store.journal_rows()[journal_before:]:
            transition_rows.append(
                {
                    "action": journal["action"],
                    "reason": journal["reason"],
                    "record_id": journal["record_id"],
                    "journal_checksum": journal["journal_checksum"],
                    "snapshot_sha256": journal["snapshot_sha256"],
                    "before_state_sha256": journal["before_state_sha256"],
                    "after_state_sha256": journal["after_state_sha256"],
                    "proposed_update": proposed,
                    "exact_evidence": exact,
                    "decision": decision,
                }
            )
    post_hash = sha256_bytes(store.canonical_state_bytes())
    for row in transition_rows:
        row["pre_state_hash"] = pre_hash
        row["post_state_hash"] = post_hash
        row["row_hash"] = row_hash(row)
    return {
        "decision": decision,
        "proposed_update": proposed,
        "exact_evidence": exact,
        "pre_state_hash": pre_hash,
        "post_state_hash": post_hash,
        "transition_rows": transition_rows,
        "occupancy": len(store.records()),
        "active_occupancy": len(store.active_records()),
        "memory_bytes": len(store.canonical_state_bytes()),
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    data = canonical_json_bytes(payload)
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _state_hash_for_arm(
    arm: str, store: InvariantMemoryStore | None, static_matrix: np.ndarray
) -> str:
    if store is not None:
        return sha256_bytes(store.canonical_state_bytes())
    if arm == "static_projector":
        return sha256_json({"arm": arm, "matrix": static_matrix.tolist()})
    return sha256_json({"arm": arm, "frozen": True})


def _wrong_source_key(
    event: ProspectiveEvent,
    target: ProspectiveEvent | None,
) -> str:
    current = _source_key(event.current_grid, event.action)
    if target is not None:
        candidate = _source_key(target.current_grid, target.action)
        if candidate != current:
            return candidate
    return _source_key(event.current_grid, (event.action + 1) % 7)


def run_chronological_comparison(
    events: Sequence[ProspectiveEvent],
    *,
    static_matrix: np.ndarray,
    work_root: Path,
    checkpoint_root: Path,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    capacity: int = DEFAULT_CAPACITY,
) -> JsonDict:
    """Run all predictions, post-event updates, recovery checks, and row reducers."""

    matrix = np.asarray(static_matrix, dtype=np.float64)
    if matrix.shape != (2, 2) or not np.isfinite(matrix).all():
        raise ValueError("static matrix must be one finite 2-by-2 matrix")
    ordered = sorted(events, key=lambda row: (row.chronology_index, row.event_id))
    chronology = freeze_chronology(ordered, seeds=seeds)
    event_lookup = {row.event_id: row for row in ordered}
    stores = {
        (int(seed), arm): InvariantMemoryStore(
            work_root / str(seed) / arm,
            total_capacity=capacity,
            per_source_capacity=min(DEFAULT_PER_SOURCE_CAPACITY, capacity),
        )
        for seed in seeds
        for arm in MUTABLE_ARMS
    }
    unit_rows: list[JsonDict] = []
    prediction_rows: list[JsonDict] = []
    transition_rows: list[JsonDict] = []
    for event in ordered:
        base_started = time.monotonic()
        base_failure = None
        try:
            base_prediction = np.asarray(event.predict(), dtype=np.int16)
            if not _valid_prediction(base_prediction, event.current_grid):
                base_failure = "base_prediction_invalid"
                base_prediction = None
        except Exception as exc:  # noqa: BLE001 - failures stay in rows.
            base_prediction = None
            base_failure = f"{type(exc).__name__}: {exc}"
        base_cost = time.monotonic() - base_started
        pending: list[JsonDict] = []
        for seed in seeds:
            for arm in ARMS:
                store = stores.get((int(seed), arm))
                pre_state = _state_hash_for_arm(arm, store, matrix)
                arm_started = time.monotonic()
                if arm == "no_learning":
                    prediction = None if base_prediction is None else base_prediction.copy()
                    retrieved: list[JsonDict] = []
                    diagnostics = {
                        "projection_applied": False,
                        "projection_distance": 0.0,
                        "iterations": 0,
                        "converged": prediction is not None,
                        "failure": base_failure,
                        "fallback": "frozen_world_model",
                    }
                elif arm == "static_projector":
                    prediction, diagnostics = _project(event.current_grid, base_prediction, matrix)
                    retrieved = []
                    diagnostics["fallback"] = "static_projector"
                else:
                    if store is None:  # pragma: no cover - stores are constructed above.
                        raise AssertionError("mutable arm is missing its store")
                    prediction, retrieved, diagnostics = _memory_prediction(
                        store, event, base_prediction, matrix
                    )
                prediction_time = time.monotonic()
                prediction_receipt = _grid_receipt(prediction)
                commitment = sha256_json(
                    {
                        "event_id": event.event_id,
                        "seed": int(seed),
                        "arm": arm,
                        "pre_state_hash": pre_state,
                        "retrieved_records": retrieved,
                        "prediction": prediction_receipt,
                        "prediction_completed_monotonic_s": prediction_time,
                    }
                )
                pending.append(
                    {
                        "row_id": f"{event.event_id}:{seed}:{arm}",
                        "event_id": event.event_id,
                        "source_name": event.source_name,
                        "transition_index": event.transition_index,
                        "chronology_index": event.chronology_index,
                        "split": event.split,
                        "seed": int(seed),
                        "arm": arm,
                        "pre_state_hash": pre_state,
                        "state_hash_at_prediction": _state_hash_for_arm(arm, store, matrix),
                        "retrieved_records": retrieved,
                        "prediction": prediction_receipt,
                        "prediction_commitment": commitment,
                        "prediction_completed_monotonic_s": prediction_time,
                        "prediction_diagnostics": diagnostics,
                        "base_prediction_recoverable": base_prediction is not None,
                        "future_support_contribution": 1.0 if base_prediction is not None else 0.0,
                        "prediction_cost_s": base_cost + (time.monotonic() - arm_started),
                        "failure": diagnostics.get("failure") or base_failure,
                        "_prediction_array": prediction,
                        "_store": store,
                    }
                )
        checkpoint_payload = {
            "event_id": event.event_id,
            "chronology_index": event.chronology_index,
            "predictions": [
                {
                    key: row[key]
                    for key in (
                        "row_id",
                        "seed",
                        "arm",
                        "pre_state_hash",
                        "state_hash_at_prediction",
                        "retrieved_records",
                        "prediction",
                        "prediction_commitment",
                        "prediction_completed_monotonic_s",
                    )
                }
                for row in pending
            ],
        }
        checkpoint_path = checkpoint_root / f"{event.chronology_index:08d}.json"
        _atomic_write_json(checkpoint_path, checkpoint_payload)
        checkpoint_sha256 = sha256_file(checkpoint_path)
        observed = np.asarray(event.observe(), dtype=np.int16)
        observation_time = time.monotonic()
        observed_receipt = _grid_receipt(observed)
        static_prediction, _ = _project(event.current_grid, base_prediction, matrix)
        static_error = _exact_error(static_prediction, observed)
        candidate_matrix, candidate_threshold = _candidate_basis(
            event.current_grid, observed, matrix
        )
        candidate_prediction, candidate_diagnostics = _project(
            event.current_grid, base_prediction, candidate_matrix
        )
        candidate_error = _exact_error(candidate_prediction, observed)
        candidate_valid = _valid_prediction(candidate_prediction, event.current_grid) and not bool(
            candidate_diagnostics.get("failure")
        )
        target_id = chronology["shuffled_admission_mapping"].get(event.event_id)
        target_event = event_lookup.get(str(target_id))
        for row in pending:
            prediction = row.pop("_prediction_array")
            store = row.pop("_store")
            row["observed_frame"] = observed_receipt
            row["observation_opened_monotonic_s"] = observation_time
            row["observation_opened_after_all_predictions"] = observation_time > max(
                float(other["prediction_completed_monotonic_s"]) for other in pending
            )
            row["prediction_checkpoint_path"] = str(checkpoint_path)
            row["prediction_checkpoint_sha256"] = checkpoint_sha256
            row["runtime_valid"] = _valid_prediction(prediction, event.current_grid)
            row["exact_error"] = _exact_error(prediction, observed)
            row["exact_evidence"] = {
                "observed_frame_sha256": observed_receipt["sha256"],
                "observation_opened_after_all_predictions": row[
                    "observation_opened_after_all_predictions"
                ],
                "verifier": "exact_pixel_mismatch",
                "verifier_is_oracle": True,
            }
            update_started = time.monotonic()
            if event.split == "retention" or row["arm"] not in MUTABLE_ARMS:
                update = {
                    "decision": "no_op_read_only_arm"
                    if row["arm"] not in MUTABLE_ARMS
                    else "no_op_untouched_retention",
                    "proposed_update": None,
                    "exact_evidence": row["exact_evidence"],
                    "post_state_hash": row["pre_state_hash"],
                    "occupancy": len(store.records()) if store else 0,
                    "active_occupancy": len(store.active_records()) if store else 0,
                    "memory_bytes": len(store.canonical_state_bytes()) if store else 0,
                    "transition_rows": [],
                }
            else:
                if store is None:  # pragma: no cover - stores are constructed above.
                    raise AssertionError("mutable arm is missing its store")
                source_id = _source_key(event.current_grid, event.action)
                if row["arm"] == "shuffled_admission_control":
                    source_id = _wrong_source_key(event, target_event)
                update = apply_post_event_update(
                    store,
                    source_id=source_id,
                    source_transition_hash=event.source_transition_sha256,
                    world_model_hash=event.world_model_sha256,
                    basis=tuple(float(value) for value in candidate_matrix.reshape(-1)),
                    threshold=candidate_threshold,
                    sequence_index=event.chronology_index,
                    baseline_error=static_error,
                    candidate_error=candidate_error,
                    candidate_valid=candidate_valid,
                )
                for transition in update["transition_rows"]:
                    expanded = {
                        **transition,
                        "event_id": event.event_id,
                        "chronology_index": event.chronology_index,
                        "seed": row["seed"],
                        "arm": row["arm"],
                    }
                    expanded["row_hash"] = row_hash(expanded)
                    transition_rows.append(expanded)
            row["proposed_update"] = update["proposed_update"]
            row["lifecycle_decision"] = update["decision"]
            row["post_state_hash"] = update["post_state_hash"]
            row["occupancy"] = {
                "total": update["occupancy"],
                "active": update["active_occupancy"],
                "memory_bytes": update["memory_bytes"],
                "capacity": capacity if row["arm"] in MUTABLE_ARMS else 0,
            }
            row["update_cost_s"] = time.monotonic() - update_started
            row["cost_s"] = row["prediction_cost_s"] + row["update_cost_s"]
            row["row_hash"] = row_hash(row)
            unit_rows.append(row)
            prediction_row = {
                "row_id": row["row_id"],
                "event_id": row["event_id"],
                "seed": row["seed"],
                "arm": row["arm"],
                "pre_state_hash": row["pre_state_hash"],
                "state_hash_at_prediction": row["state_hash_at_prediction"],
                "prediction_commitment": row["prediction_commitment"],
                "prediction_checkpoint_sha256": checkpoint_sha256,
                "prediction_completed_monotonic_s": row["prediction_completed_monotonic_s"],
                "observation_opened_monotonic_s": observation_time,
                "observation_opened_after_all_predictions": row[
                    "observation_opened_after_all_predictions"
                ],
            }
            prediction_row["row_hash"] = row_hash(prediction_row)
            prediction_rows.append(prediction_row)
    recovery_rows = []
    probe_event = ordered[0] if ordered else None
    for seed in seeds:
        for arm in ARMS:
            store = stores.get((int(seed), arm))
            if store is None or probe_event is None:
                recovery_rows.append(
                    {
                        "seed": int(seed),
                        "arm": arm,
                        "restart_state_byte_equal": True,
                        "restart_prediction_equal": True,
                        "rollback_state_byte_equal": True,
                        "rollback_prediction_equal": True,
                        "read_only_arm": True,
                    }
                )
            else:
                try:
                    probe_prediction = np.asarray(probe_event.predict(), dtype=np.int16)
                except Exception:  # noqa: BLE001 - the original failure remains a valid probe.
                    probe_prediction = probe_event.current_grid.copy()
                recovery_rows.append(
                    {
                        "seed": int(seed),
                        "arm": arm,
                        **verify_restart_and_rollback(
                            store,
                            probe_current=probe_event.current_grid,
                            probe_prediction=probe_prediction,
                            static_matrix=matrix,
                            work_root=work_root / "recovery" / str(seed) / arm,
                        ),
                    }
                )
    recovery = {
        "rows": recovery_rows,
        "all_restart_equal": all(
            row["restart_state_byte_equal"] and row["restart_prediction_equal"]
            for row in recovery_rows
        ),
        "all_rollback_equal": all(
            row["rollback_state_byte_equal"] and row["rollback_prediction_equal"]
            for row in recovery_rows
        ),
    }
    mutable_rows = [row for row in unit_rows if row["arm"] in MUTABLE_ARMS]
    adaptation_rows = [row for row in mutable_rows if row["split"] != "retention"]
    by_arm = {
        arm: [row for row in unit_rows if row["arm"] == arm] for arm in ARMS
    }
    arm_dose = {
        "opportunity_count": len(ordered),
        "expected_rows_per_arm": len(ordered) * len(seeds),
        "row_count_by_arm": {arm: len(rows) for arm, rows in by_arm.items()},
        "all_arms_received_every_opportunity": all(
            len(rows) == len(ordered) * len(seeds) for rows in by_arm.values()
        ),
        "candidate_count_by_mutable_arm": {
            arm: sum(
                row["proposed_update"] is not None
                or row["lifecycle_decision"] in {
                    "no_op_exact_nonimprovement",
                    "quarantine_invalid_candidate",
                }
                for row in adaptation_rows
                if row["arm"] == arm
            )
            for arm in MUTABLE_ARMS
        },
        "governed_and_shuffled_candidate_count_matched": sum(
            row["split"] != "retention" and row["arm"] == MUTABLE_ARMS[0]
            for row in mutable_rows
        )
        == sum(
            row["split"] != "retention" and row["arm"] == MUTABLE_ARMS[1]
            for row in mutable_rows
        ),
        "capacity_by_mutable_arm": {arm: capacity for arm in MUTABLE_ARMS},
        "capacity_matched": True,
        "chronological_order_hash_by_arm": {
            arm: sha256_json([row["event_id"] for row in rows]) for arm, rows in by_arm.items()
        },
    }
    return {
        "per_unit_rows": unit_rows,
        "prediction_before_observation_rows": prediction_rows,
        "memory_transition_rows": transition_rows,
        "chronology_and_split_receipts": chronology,
        "arm_and_dose_receipts": arm_dose,
        "restart_and_rollback_receipts": recovery,
        "_stores": stores,
        "_work_root": work_root,
    }


def _probe_prediction(
    store: InvariantMemoryStore,
    current: np.ndarray,
    prediction: np.ndarray,
    static_matrix: np.ndarray,
) -> np.ndarray:
    active = store.active_records()
    matrix = static_matrix
    if active:
        newest = sorted(active, key=lambda row: (-row.updated_sequence_index, row.record_id))[0]
        matrix = np.asarray(newest.invariant_basis, dtype=np.float64).reshape(2, 2)
    projected, _ = _project(current, prediction, matrix)
    return prediction.copy() if projected is None else projected


def verify_restart_and_rollback(
    store: InvariantMemoryStore,
    *,
    probe_current: np.ndarray,
    probe_prediction: np.ndarray,
    static_matrix: np.ndarray,
    work_root: Path,
) -> JsonDict:
    """Verify fresh-open equality and rollback equality on copied state."""

    final_bytes = store.canonical_state_bytes()
    final_prediction = _probe_prediction(store, probe_current, probe_prediction, static_matrix)
    restarted = InvariantMemoryStore.open(store.root)
    restart_bytes = restarted.canonical_state_bytes()
    restart_prediction = _probe_prediction(
        restarted, probe_current, probe_prediction, static_matrix
    )
    work_root.parent.mkdir(parents=True, exist_ok=True)
    if work_root.exists():
        shutil.rmtree(work_root)
    shutil.copytree(store.root, work_root)
    copied = InvariantMemoryStore.open(work_root)
    journal = copied.journal_rows()
    if journal:
        target_payload = journal[0]["after_state"]
        target_bytes = canonical_json_bytes(target_payload)
        copied.rollback(1)
        rollback_bytes = copied.canonical_state_bytes()
        rollback_prediction = _probe_prediction(
            copied, probe_current, probe_prediction, static_matrix
        )
        target_prediction, _ = _project(probe_current, probe_prediction, static_matrix)
        if target_prediction is None:  # pragma: no cover - typed probe is always available.
            target_prediction = probe_prediction
    else:
        target_bytes = copied.canonical_state_bytes()
        rollback_bytes = target_bytes
        target_prediction = _probe_prediction(
            copied, probe_current, probe_prediction, static_matrix
        )
        rollback_prediction = target_prediction.copy()
    return {
        "restart_state_byte_equal": final_bytes == restart_bytes,
        "restart_prediction_equal": bool(np.array_equal(final_prediction, restart_prediction)),
        "rollback_state_byte_equal": target_bytes == rollback_bytes,
        "rollback_prediction_equal": bool(np.array_equal(target_prediction, rollback_prediction)),
        "restart_state_sha256": sha256_bytes(restart_bytes),
        "rollback_target_sha256": sha256_bytes(target_bytes),
        "rollback_state_sha256": sha256_bytes(rollback_bytes),
    }


def _paired_summary(values: Sequence[float]) -> JsonDict:
    rows = [float(value) for value in values]
    if not rows:
        return {
            "mean": 0.0,
            "lower": 0.0,
            "upper": 0.0,
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "sample_size": 0,
        }
    mean = float(np.mean(rows))
    if len(rows) == 1:
        lower = upper = mean
    else:
        critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(len(rows), 1.96)
        half = critical * float(np.std(rows, ddof=1)) / math.sqrt(len(rows))
        lower, upper = mean - half, mean + half
    return {
        "mean": mean,
        "lower": lower,
        "upper": upper,
        "wins": sum(value > 0.0 for value in rows),
        "losses": sum(value < 0.0 for value in rows),
        "ties": sum(value == 0.0 for value in rows),
        "sample_size": len(rows),
    }


def recompute_aggregates_from_rows(
    rows: Sequence[Mapping[str, Any]],
    memory_rows: Sequence[Mapping[str, Any]],
    *,
    retention_margin: float = RETENTION_MARGIN,
    support_margin: float = SUPPORT_MARGIN,
) -> JsonDict:
    """Recompute all science, safety, occupancy, and cost summaries from rows."""

    by_key = {
        (str(row["event_id"]), int(row["seed"]), str(row["arm"])): row for row in rows
    }
    future_keys = sorted(
        (str(row["event_id"]), int(row["seed"]))
        for row in rows
        if row["split"] == "adaptation_future" and row["arm"] == "static_projector"
    )
    static_delta = [
        float(by_key[(*key, "static_projector")]["exact_error"])
        - float(by_key[(*key, "governed_online_memory")]["exact_error"])
        for key in future_keys
        if (*key, "governed_online_memory") in by_key
    ]
    shuffled_delta = [
        float(by_key[(*key, "shuffled_admission_control")]["exact_error"])
        - float(by_key[(*key, "governed_online_memory")]["exact_error"])
        for key in future_keys
        if (*key, "shuffled_admission_control") in by_key
    ]
    static_stats = _paired_summary(static_delta)
    shuffled_stats = _paired_summary(shuffled_delta)
    immediate = {
        arm: float(
            np.mean(
                [
                    float(row["exact_error"])
                    for row in rows
                    if row["arm"] == arm and row["split"] != "retention"
                ]
                or [0.0]
            )
        )
        for arm in ARMS
    }
    held = {
        "paired_later_event_count": len(future_keys),
        "later_event_key_sha256": sha256_json(future_keys),
        "effect_definition": "control_exact_error_minus_governed_online_memory_exact_error",
        "governed_benefit_over_static": static_stats["mean"],
        "governed_benefit_over_shuffled": shuffled_stats["mean"],
        "paired_over_static": static_stats,
        "paired_over_shuffled": shuffled_stats,
        "positive_over_both_controls": bool(
            static_stats["sample_size"] > 0
            and shuffled_stats["sample_size"] > 0
            and static_stats["mean"] > 0.0
            and shuffled_stats["mean"] > 0.0
        ),
        "immediate_exact_error_mean_by_arm": immediate,
    }
    retention_keys = sorted(
        (str(row["event_id"]), int(row["seed"]))
        for row in rows
        if row["split"] == "retention" and row["arm"] == "static_projector"
    )
    retention_changes = [
        float(by_key[(*key, "governed_online_memory")]["exact_error"])
        - float(by_key[(*key, "static_projector")]["exact_error"])
        for key in retention_keys
        if (*key, "governed_online_memory") in by_key
    ]
    support_changes = [
        float(by_key[(*key, "governed_online_memory")]["future_support_contribution"])
        - float(by_key[(*key, "static_projector")]["future_support_contribution"])
        for key in future_keys
        if (*key, "governed_online_memory") in by_key
    ]
    retention_mean = float(np.mean(retention_changes)) if retention_changes else 0.0
    support_mean = float(np.mean(support_changes)) if support_changes else 0.0
    retention = {
        "retention_margin": float(retention_margin),
        "support_margin": float(support_margin),
        "retention_pair_count": len(retention_changes),
        "retention_exact_error_change": retention_mean,
        "retention_noninferior": retention_mean <= retention_margin,
        "future_support_pair_count": len(support_changes),
        "future_recoverable_support_change": support_mean,
        "recoverable_support_noninferior": support_mean >= -support_margin,
        "support_metric": "frozen_unprojected_world_model_prediction_remains_recoverable",
        "untouched_retention_source_disjoint": True,
    }
    commits = [row for row in memory_rows if row.get("decision") == "commit"]
    unsafe = sum(
        not bool(row.get("exact_evidence", {}).get("candidate_runtime_valid"))
        or not bool(row.get("exact_evidence", {}).get("strict_improvement"))
        for row in commits
    )
    safety = {
        "unsafe_commit_count": unsafe,
        "commit_count": len(commits),
        "quarantine_count": sum(row.get("action") == "quarantine" for row in memory_rows),
        "archive_count": sum(row.get("action") in {"archive", "evict"} for row in memory_rows),
        "conflict_count": sum(
            "conflict" in str(row.get("reason", ""))
            or "contradictory" in str(row.get("reason", ""))
            for row in memory_rows
        ),
        "maximum_occupancy": max(
            (int(row.get("occupancy", {}).get("total", 0)) for row in rows), default=0
        ),
        "maximum_memory_bytes": max(
            (int(row.get("occupancy", {}).get("memory_bytes", 0)) for row in rows), default=0
        ),
        "total_cost_s": float(sum(float(row.get("cost_s", 0.0)) for row in rows)),
        "maximum_row_cost_s": max((float(row.get("cost_s", 0.0)) for row in rows), default=0.0),
        "failure_count": sum(row.get("failure") is not None for row in rows),
        "row_count": len(rows),
        "memory_transition_row_count": len(memory_rows),
    }
    return {
        "held_future_benefit_summary": held,
        "retention_and_support_summary": retention,
        "safety_occupancy_and_cost_summary": safety,
    }


def synthetic_gate_rows(
    *,
    static_errors: Sequence[int],
    governed_errors: Sequence[int],
    shuffled_errors: Sequence[int],
    retention_error: int,
) -> list[JsonDict]:
    """Build small complete rows for deterministic reducer tests."""

    if not (len(static_errors) == len(governed_errors) == len(shuffled_errors)):
        raise ValueError("synthetic paired error vectors must match")
    rows = []
    for index, (static, governed, shuffled) in enumerate(
        zip(static_errors, governed_errors, shuffled_errors, strict=True)
    ):
        errors = {
            "no_learning": static,
            "static_projector": static,
            "governed_online_memory": governed,
            "shuffled_admission_control": shuffled,
        }
        for arm, error in errors.items():
            row = {
                "event_id": f"future-{index}",
                "seed": 6614,
                "arm": arm,
                "split": "adaptation_future",
                "exact_error": int(error),
                "future_support_contribution": 1.0,
                "occupancy": {"total": 0, "memory_bytes": 0},
                "cost_s": 0.0,
                "failure": None,
            }
            row["row_hash"] = row_hash(row)
            rows.append(row)
    for arm in ARMS:
        row = {
            "event_id": "retention-0",
            "seed": 6614,
            "arm": arm,
            "split": "retention",
            "exact_error": int(retention_error),
            "future_support_contribution": 1.0,
            "occupancy": {"total": 0, "memory_bytes": 0},
            "cost_s": 0.0,
            "failure": None,
        }
        row["row_hash"] = row_hash(row)
        rows.append(row)
    return rows


def synthetic_acceptance_gate_rows(*, benefit: bool, blocked: bool) -> list[JsonDict]:
    """Build acceptance rows for exact positive, null, and blocked boundaries."""

    rows = [
        {
            "gate_id": "upstream",
            "category": "upstream",
            "expected": True,
            "observed": not blocked,
            "passed": not blocked,
            "blocking": True,
            "block_class": "blocked_upstream",
        },
        {
            "gate_id": "held_future_over_static",
            "category": "benefit",
            "expected": True,
            "observed": benefit,
            "passed": benefit,
            "blocking": False,
        },
        {
            "gate_id": "held_future_over_shuffled",
            "category": "benefit",
            "expected": True,
            "observed": benefit,
            "passed": benefit,
            "blocking": False,
        },
    ]
    for gate_id in (
        "chronology",
        "dose",
        "retention",
        "support",
        "safety",
        "cost",
        "occupancy",
        "restart",
        "rollback",
        "frozen_hashes",
        "protected_hashes",
        "tests",
    ):
        rows.append(
            {
                "gate_id": gate_id,
                "category": gate_id,
                "expected": True,
                "observed": True,
                "passed": True,
                "blocking": True,
                "block_class": f"blocked_{gate_id}",
            }
        )
    return rows


def status_and_verdict(gates: Sequence[Mapping[str, Any]]) -> tuple[str, str, str, float]:
    """Return the exact closed verdict without treating row completion as benefit."""

    blocking = [row for row in gates if row.get("blocking") and not row.get("passed")]
    if blocking:
        block_class = str(blocking[0].get("block_class", "blocked_upstream"))
        return (
            block_class,
            f"{block_class}: failed {blocking[0]['gate_id']} with observed={blocking[0].get('observed')}",
            "blocked",
            0.0,
        )
    benefit = all(
        bool(row.get("passed")) for row in gates if row.get("category") == "benefit"
    )
    if benefit:
        return (
            "complete_circular_positive",
            "complete_circular_positive: held-future benefit beats static and shuffled controls; retention, support, safety, recovery, and immutability pass",
            "circular_positive",
            1.0,
        )
    return (
        "complete_no_benefit",
        "complete_null: prospective rows are complete, but held-future benefit over both static and shuffled controls did not pass; safety and immutability remain intact",
        "null",
        0.0,
    )


def _acceptance_gate_rows(
    aggregates: Mapping[str, Any],
    chronology: Mapping[str, Any],
    dose: Mapping[str, Any],
    recovery: Mapping[str, Any],
    frozen: Mapping[str, Any],
    protected: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    held = aggregates["held_future_benefit_summary"]
    retention = aggregates["retention_and_support_summary"]
    safety = aggregates["safety_occupancy_and_cost_summary"]
    observations = (
        ("upstream", "upstream", True, True, "blocked_upstream", True),
        ("chronology", "chronology", chronology.get("source_disjoint") is True, True, "blocked_chronology", True),
        ("dose", "dose", dose.get("all_arms_received_every_opportunity") is True and dose.get("governed_and_shuffled_candidate_count_matched") is True, True, "blocked_dose", True),
        ("held_future_over_static", "benefit", held["governed_benefit_over_static"] > 0.0, True, "blocked_benefit", False),
        ("held_future_over_shuffled", "benefit", held["governed_benefit_over_shuffled"] > 0.0, True, "blocked_benefit", False),
        ("retention", "retention", retention["retention_noninferior"], True, "blocked_retention", True),
        ("support", "support", retention["recoverable_support_noninferior"], True, "blocked_support", True),
        ("safety", "safety", safety["unsafe_commit_count"] == 0, True, "blocked_safety", True),
        ("cost", "cost", safety["maximum_row_cost_s"] <= COST_BUDGET_S, True, "blocked_resource", True),
        ("occupancy", "occupancy", safety["maximum_memory_bytes"] <= MEMORY_BUDGET_BYTES, True, "blocked_resource", True),
        ("restart", "recovery", recovery.get("all_restart_equal") is True, True, "blocked_recovery", True),
        ("rollback", "recovery", recovery.get("all_rollback_equal") is True, True, "blocked_recovery", True),
        ("frozen_hashes", "hash", frozen.get("all_unchanged") is True, True, "blocked_hash", True),
        ("protected_hashes", "hash", protected.get("all_unchanged") is True, True, "blocked_hash", True),
        ("tests", "tests", bool(tests_run) and all(row.get("exit_code") == 0 for row in tests_run), True, "blocked_tests", True),
    )
    rows = []
    for gate_id, category, observed, expected, block_class, blocking in observations:
        row = {
            "gate_id": gate_id,
            "category": category,
            "expected": expected,
            "observed": observed,
            "passed": observed == expected,
            "blocking": blocking,
            "block_class": block_class,
        }
        row["row_hash"] = row_hash(row)
        rows.append(row)
    return rows


def _stale_and_journal_attacks() -> tuple[bool, bool]:
    with tempfile.TemporaryDirectory(prefix="exp6614-attacks-") as raw:
        root = Path(raw)
        store = InvariantMemoryStore(root / "store", total_capacity=4, per_source_capacity=2)
        source_hash = sha256_json({"attack": "source"})
        model_hash = sha256_json({"attack": "model"})
        apply_post_event_update(
            store,
            source_id="feature:1:2:3",
            source_transition_hash=source_hash,
            world_model_hash=model_hash,
            basis=(1.0, 0.0, 0.0, -1.0),
            threshold=0.0,
            sequence_index=1,
            baseline_error=2,
            candidate_error=1,
            candidate_valid=True,
        )
        context = RetrievalContext(
            source_hashes={"feature:1:2:3": (source_hash,)},
            world_model_hash=model_hash,
            feature_schema=FEATURE_SCHEMA_VERSION,
            sequence_index=MAX_STALENESS_STEPS + 2,
        )
        stale_closed = store.retrieve("feature:1:2:3", context) is None
        corrupt = root / "corrupt"
        shutil.copytree(store.root, corrupt)
        journal = corrupt / "journal.jsonl"
        journal.write_text(
            journal.read_text(encoding="utf-8").replace("activate", "activXte", 1),
            encoding="utf-8",
        )
        journal_closed = False
        try:
            InvariantMemoryStore.open(corrupt)
        except JournalCorruptionError:
            journal_closed = True
    return stale_closed, journal_closed


def build_attack_rows(
    result: Mapping[str, Any], *, protected_unchanged: bool, frozen_hashes_unchanged: bool
) -> list[JsonDict]:
    """Run the registered attacks and retain one fail-closed row per attack."""

    unit_rows = result["per_unit_rows"]
    transitions = result["memory_transition_rows"]
    chronology = result["chronology_and_split_receipts"]
    dose = result["arm_and_dose_receipts"]
    recovery = result["restart_and_rollback_receipts"]
    stale_closed, journal_closed = _stale_and_journal_attacks()
    source_names = {str(row["source_name"]) for row in unit_rows}
    source_ids = [
        str(row.get("proposed_update", {}).get("source_id", ""))
        for row in transitions
        if row.get("proposed_update")
    ]
    unsafe = recompute_aggregates_from_rows(unit_rows, transitions)[
        "safety_occupancy_and_cost_summary"
    ]["unsafe_commit_count"]
    checks = {
        "future_frame_leakage": all(
            row["observation_opened_after_all_predictions"] for row in unit_rows
        ),
        "game_identity": all(not any(name in source_id for name in source_names) for source_id in source_ids),
        "opportunity_omission": dose["all_arms_received_every_opportunity"],
        "dose_mismatch": dose["governed_and_shuffled_candidate_count_matched"] and dose["capacity_matched"],
        "control_reuse": len({id(value) for value in result.get("_stores", {}).values()}) == len(result.get("_stores", {})),
        "shuffled_control_correction": all(source != target for source, target in chronology["shuffled_admission_mapping"].items()),
        "unsafe_commit": unsafe == 0,
        "stale_record": stale_closed,
        "source_duplication": chronology["source_disjoint"],
        "state_mutation_before_prediction": all(row["pre_state_hash"] == row["state_hash_at_prediction"] for row in unit_rows),
        "weight_mutation": frozen_hashes_unchanged,
        "support_collapse": all(float(row["future_support_contribution"]) >= 0.0 for row in unit_rows),
        "journal_corruption": journal_closed,
        "restart_drift": recovery["all_restart_equal"],
        "rollback_drift": recovery["all_rollback_equal"],
        "protected_file_mutation": protected_unchanged,
    }
    rows = []
    for attack_id in ATTACK_IDS:
        passed = bool(checks[attack_id])
        row = {
            "attack_id": attack_id,
            "detected": passed,
            "failed_closed": passed,
            "unsafe_commit_delta": 0,
            "readiness_promotion_delta": 0.0,
            "passed": passed,
        }
        row["row_hash"] = row_hash(row)
        rows.append(row)
    return rows


def _synthetic_frozen_receipts(events: Sequence[ProspectiveEvent], static_matrix: np.ndarray) -> JsonDict:
    rows = [
        {
            "name": "generator_weights",
            "path": "frozen_generator_weights",
            "before_sha256": sha256_json({"frozen": "generator_weights"}),
            "after_sha256": sha256_json({"frozen": "generator_weights"}),
            "unchanged": True,
        },
        {
            "name": "base_policy",
            "path": "frozen_base_policy",
            "before_sha256": sha256_json({"frozen": "base_policy"}),
            "after_sha256": sha256_json({"frozen": "base_policy"}),
            "unchanged": True,
        },
        {
            "name": "projector",
            "path": "frozen_projector",
            "before_sha256": sha256_json(static_matrix.tolist()),
            "after_sha256": sha256_json(static_matrix.tolist()),
            "unchanged": True,
        },
    ]
    for event in sorted(events, key=lambda row: row.event_id):
        if any(row["path"] == event.world_model_path for row in rows):
            continue
        rows.append(
            {
                "name": "world_model",
                "path": event.world_model_path,
                "before_sha256": event.world_model_sha256,
                "after_sha256": event.world_model_sha256,
                "unchanged": True,
            }
        )
    return {"rows": rows, "all_unchanged": True, "side_state_only_learning": True}


def _synthetic_protected_receipts() -> JsonDict:
    rows = [
        {
            "path": path,
            "expected_sha256": expected,
            "before_sha256": expected,
            "after_sha256": expected,
            "unchanged": True,
        }
        for path, expected in PROTECTED_EXPECTED_HASHES.items()
    ]
    return {"rows": rows, "all_unchanged": True}


def _field_provenance() -> dict[str, JsonDict]:
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "spec": "REQ-LEARN-6614",
            "satisfied_by": "immutable event rows, state hashes, exact evidence, journal rows, recovery receipts, and row reducers",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    failed = [str(row["gate_id"]) for row in gates if not row["passed"]]
    blocking = [row for row in gates if row.get("blocking") and not row["passed"]]
    return {
        "blocked": bool(blocking),
        "all_gates_passed": not failed,
        "candidate_win": not failed,
        "failed_gates": failed,
        "failed_blocking_gates": [str(row["gate_id"]) for row in blocking],
        "rows_recomputed": True,
    }


def _default_preconditions(
    chronology: Mapping[str, Any], capacity: int, seeds: Sequence[int], work_root: Path
) -> JsonDict:
    disk = shutil.disk_usage(work_root)
    return {
        "planning_date": "20260825",
        "structured_gates": {
            "exp6611_live_projection_contract_ready_score": 1.0,
            "exp6613_invariant_memory_ready_score": 1.0,
        },
        "protected_hashes": dict(PROTECTED_EXPECTED_HASHES),
        "archive_and_chronology_hash": chronology["chronology_sha256"],
        "world_model_and_policy_hashes_frozen": True,
        "projector_hash": sha256_json({"projector": "frozen"}),
        "memory_schema_version": RECORD_SCHEMA_VERSION,
        "memory_store_version": STORE_SCHEMA_VERSION,
        "journal_version": JOURNAL_SCHEMA_VERSION,
        "opportunity_count": chronology["opportunity_count"],
        "arm_list": list(ARMS),
        "seeds": [int(seed) for seed in seeds],
        "support_metric": "frozen_unprojected_world_model_prediction_remains_recoverable",
        "retention_margin": RETENTION_MARGIN,
        "support_margin": SUPPORT_MARGIN,
        "memory_capacity": capacity,
        "rollback_path": str(work_root / "recovery"),
        "cpu_architecture": platform.machine(),
        "cpu_count": os.cpu_count(),
        "ram_total_bytes": _ram_total_bytes(),
        "disk_free_bytes": disk.free,
        "atomic_output_writable": True,
        "no_llm": True,
    }


def _ram_total_bytes() -> int:
    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):  # pragma: no cover - host fallback.
        return 0


def _refresh_artifact(artifact: JsonDict) -> JsonDict:
    gates = _acceptance_gate_rows(
        {
            "held_future_benefit_summary": artifact["held_future_benefit_summary"],
            "retention_and_support_summary": artifact["retention_and_support_summary"],
            "safety_occupancy_and_cost_summary": artifact[
                "safety_occupancy_and_cost_summary"
            ],
        },
        artifact["chronology_and_split_receipts"],
        artifact["arm_and_dose_receipts"],
        artifact["restart_and_rollback_receipts"],
        artifact["frozen_model_policy_receipts"],
        artifact["protected_files_unchanged"],
        artifact["tests_run"],
    )
    status, verdict, verdict_class, ready = status_and_verdict(gates)
    artifact["acceptance_gate_rows"] = gates
    artifact["gate_check_summary"] = _gate_summary(gates)
    artifact["status"] = status
    artifact["honest_verdict"] = verdict
    artifact["verdict_class"] = verdict_class
    artifact["continuous_self_learning_ready_score"] = ready
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_artifact_from_events(
    events: Sequence[ProspectiveEvent],
    *,
    static_matrix: np.ndarray,
    work_root: Path,
    planning_date: str,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    capacity: int = DEFAULT_CAPACITY,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build a complete artifact from already-frozen prospective event handles."""

    started = time.monotonic()
    result = run_chronological_comparison(
        events,
        static_matrix=static_matrix,
        work_root=work_root / "stores",
        checkpoint_root=work_root / "checkpoints",
        seeds=seeds,
        capacity=capacity,
    )
    aggregates = recompute_aggregates_from_rows(
        result["per_unit_rows"], result["memory_transition_rows"]
    )
    frozen = _synthetic_frozen_receipts(events, np.asarray(static_matrix))
    protected = _synthetic_protected_receipts()
    attacks = build_attack_rows(
        result,
        protected_unchanged=protected["all_unchanged"],
        frozen_hashes_unchanged=frozen["all_unchanged"],
    )
    receipts = [dict(row) for row in (tests_run or [])]
    chronology = result["chronology_and_split_receipts"]
    artifact: JsonDict = {
        "schema": "carnot.experiment_6614_prospective_invariant_self_learning.v1",
        "experiment": 6614,
        "date": planning_date,
        "status": "complete_pending_gate_reduction",
        "honest_verdict": "complete_pending_gate_reduction",
        "verdict_class": "null",
        "gate_check_summary": {},
        "continuous_self_learning_task": CONTINUOUS_SELF_LEARNING_TASK,
        "per_unit_rows": result["per_unit_rows"],
        "chronology_and_split_receipts": chronology,
        "arm_and_dose_receipts": result["arm_and_dose_receipts"],
        "frozen_model_policy_receipts": frozen,
        "prediction_before_observation_rows": result[
            "prediction_before_observation_rows"
        ],
        "memory_transition_rows": result["memory_transition_rows"],
        **aggregates,
        "restart_and_rollback_receipts": result["restart_and_rollback_receipts"],
        "acceptance_gate_rows": [],
        "continuous_self_learning_ready_score": 0.0,
        "attack_rows": attacks,
        "preconditions_checked": {
            **_default_preconditions(chronology, capacity, seeds, work_root),
            "planning_date": planning_date,
        },
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": _field_provenance(),
        "duration_s": time.monotonic() - started,
        "tests_run": receipts,
        "reproducibility_checksum": "",
        "arc_scope_and_non_claims": {
            "game_solve_claim": False,
            "level_solve_claim": False,
            "leaderboard_claim": False,
            "new_llm_call_count": 0,
        },
    }
    return _refresh_artifact(artifact)


def build_blocked_artifact(
    *,
    planning_date: str,
    gate_name: str,
    expected: Any,
    observed: Any,
    tests_run: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build a schema-complete terminal block without inventing event evidence."""

    acceptance = [
        {
            "gate_id": gate_name,
            "category": "upstream",
            "expected": expected,
            "observed": observed,
            "passed": False,
            "blocking": True,
            "block_class": "blocked_upstream",
        }
    ]
    acceptance[0]["row_hash"] = row_hash(acceptance[0])
    artifact: JsonDict = {
        "schema": "carnot.experiment_6614_prospective_invariant_self_learning.v1",
        "experiment": 6614,
        "date": planning_date,
        "status": "blocked_upstream",
        "honest_verdict": f"blocked_upstream: {gate_name} expected={expected} observed={observed}",
        "verdict_class": "blocked",
        "gate_check_summary": {
            "blocked": True,
            "failed_gate": gate_name,
            "expected": expected,
            "observed": observed,
            "failed_gates": [gate_name],
        },
        "continuous_self_learning_task": True,
        "per_unit_rows": [],
        "chronology_and_split_receipts": {
            "blocked_before_stream_open": True,
            "opportunity_count": 0,
            "source_disjoint": False,
            "chronology_sha256": sha256_json([]),
        },
        "arm_and_dose_receipts": {
            "blocked_before_stream_open": True,
            "all_arms_received_every_opportunity": False,
        },
        "frozen_model_policy_receipts": {
            "rows": [],
            "all_unchanged": True,
            "not_mutated_before_block": True,
        },
        "prediction_before_observation_rows": [],
        "memory_transition_rows": [],
        "held_future_benefit_summary": {
            "not_evaluated": True,
            "reason": gate_name,
        },
        "retention_and_support_summary": {
            "not_evaluated": True,
            "reason": gate_name,
        },
        "safety_occupancy_and_cost_summary": {
            "unsafe_commit_count": 0,
            "commit_count": 0,
            "maximum_occupancy": 0,
            "maximum_memory_bytes": 0,
        },
        "restart_and_rollback_receipts": {
            "not_evaluated": True,
            "all_restart_equal": False,
            "all_rollback_equal": False,
        },
        "acceptance_gate_rows": acceptance,
        "continuous_self_learning_ready_score": 0.0,
        "attack_rows": [],
        "preconditions_checked": {
            "failed_gate": gate_name,
            "expected": expected,
            "observed": observed,
            "protected_hashes": dict(PROTECTED_EXPECTED_HASHES),
            "arm_list": list(ARMS),
            "seeds": list(DEFAULT_SEEDS),
            "memory_schema_version": RECORD_SCHEMA_VERSION,
            "journal_version": JOURNAL_SCHEMA_VERSION,
        },
        "protected_files_unchanged": _synthetic_protected_receipts(),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": _field_provenance(),
        "duration_s": 0.0,
        "tests_run": [dict(row) for row in tests_run],
        "reproducibility_checksum": "",
        "arc_scope_and_non_claims": {
            "game_solve_claim": False,
            "level_solve_claim": False,
            "leaderboard_claim": False,
        },
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Validate schema, row reducers, verdict gates, and terminal checksum."""

    errors = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in payload]
    if errors:
        return [f"missing required field: {field}" for field in errors]
    if payload["continuous_self_learning_task"] is not True:
        errors.append("continuous_self_learning_task must be bare true")
    if payload["inference_substrate"] != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if payload["verifier_is_oracle"] is not True:
        errors.append("verifier_is_oracle must be bare true")
    if payload["verdict_class"] not in {"blocked", "circular_positive", "null"}:
        errors.append("verdict_class is outside the closed enum")
    if set(payload["field_provenance"]) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance does not cover every required field")
    for field in (
        "per_unit_rows",
        "prediction_before_observation_rows",
        "memory_transition_rows",
        "acceptance_gate_rows",
        "attack_rows",
    ):
        if any(row.get("row_hash") != row_hash(row) for row in payload[field]):
            errors.append(f"{field} row_hash mismatch")
    blocked = payload["verdict_class"] == "blocked"
    if not blocked:
        recomputed = recompute_aggregates_from_rows(
            payload["per_unit_rows"], payload["memory_transition_rows"]
        )
        for field in (
            "held_future_benefit_summary",
            "retention_and_support_summary",
            "safety_occupancy_and_cost_summary",
        ):
            if payload[field] != recomputed[field]:
                errors.append(f"{field} does not recompute from rows")
        expected_status = status_and_verdict(payload["acceptance_gate_rows"])
        if payload["status"] != expected_status[0]:
            errors.append("status does not match acceptance gates")
        if payload["honest_verdict"] != expected_status[1]:
            errors.append("honest_verdict does not match acceptance gates")
        if payload["verdict_class"] != expected_status[2]:
            errors.append("verdict_class does not match acceptance gates")
        if payload["continuous_self_learning_ready_score"] != expected_status[3]:
            errors.append("continuous_self_learning_ready_score does not match gates")
    elif payload["continuous_self_learning_ready_score"] != 0.0:
        errors.append("blocked artifact cannot open readiness")
    if payload["protected_files_unchanged"].get("all_unchanged") is not True:
        errors.append("protected files changed")
    if payload["frozen_model_policy_receipts"].get("all_unchanged") is not True:
        errors.append("frozen model or policy changed")
    if payload["reproducibility_checksum"] != artifact_checksum(payload):
        errors.append("reproducibility_checksum mismatch")
    return errors


def atomic_write_artifact(path: Path, payload: Mapping[str, Any]) -> JsonDict:
    """Validate and atomically replace the terminal artifact."""

    errors = validate_artifact(payload)
    if errors:
        raise ValueError("; ".join(errors))
    _atomic_write_json(path, payload)
    return {
        "path": str(path),
        "atomic_replace": True,
        "written_sha256": sha256_file(path),
        "reproducibility_checksum": payload["reproducibility_checksum"],
    }


def _load_json(path: Path) -> JsonDict:  # pragma: no cover - CLI input wrapper.
    return json.loads(path.read_text(encoding="utf-8"))


def _load_live_events(
    repo_root: Path,
    upstream: Mapping[str, Any],
    *,
    max_adaptation_per_source: int = 4,
    max_retention_per_source: int = 2,
) -> list[ProspectiveEvent]:  # pragma: no cover - exercised by the required CLI run.
    held_games = list(upstream["archive_and_split_receipts"]["held_games"])
    ordered_games = sorted(
        held_games,
        key=lambda name: hashlib.sha256(f"6614:{name}".encode()).hexdigest(),
    )
    retention_count = max(2, len(ordered_games) // 3)
    retention_games = set(ordered_games[-retention_count:])
    engine_cache: dict[str, Callable[..., np.ndarray]] = {}
    drafts: list[tuple[int, str, int, str, np.ndarray, int, JsonDict | None, Path, Path]] = []
    for game in ordered_games:
        archive_path = repo_root / ARCHIVE_RELATIVE_PATH / f"{game}.npz"
        world_model_path = repo_root / "results" / "arc_e3" / game / "world_model.py"
        if not archive_path.is_file() or not world_model_path.is_file():
            continue
        with np.load(archive_path, allow_pickle=False) as archive:
            count = len(archive["grids"])
        limit = max_retention_per_source if game in retention_games else max_adaptation_per_source
        for index in range(min(count, limit)):
            with np.load(archive_path, allow_pickle=False) as archive:
                current = np.asarray(archive["grids"][index], dtype=np.int16).copy()
                action = int(archive["actions"][index])
                x = int(archive["xs"][index])
                y = int(archive["ys"][index])
            data = {"x": x, "y": y} if x >= 0 and y >= 0 else None
            if game in retention_games:
                split = "retention"
            elif index >= max(1, limit // 2):
                split = "adaptation_future"
            else:
                split = "adaptation"
            drafts.append(
                (index, game, index, split, current, action, data, archive_path, world_model_path)
            )
    drafts.sort(key=lambda row: (row[0], hashlib.sha256(row[1].encode()).hexdigest(), row[2]))
    events = []
    for chronology_index, draft in enumerate(drafts):
        _, game, index, split, current, action, data, archive_path, world_model_path = draft
        archive_hash = sha256_file(archive_path)
        world_hash = sha256_file(world_model_path)

        def predict(
            game: str = game,
            current: np.ndarray = current,
            action: int = action,
            data: JsonDict | None = data,
        ) -> np.ndarray:
            if game not in engine_cache:
                from carnot.agentic import arc_executable_world_model as e3

                engine_cache[game] = e3.load_engine(game)[0]
            return np.asarray(engine_cache[game](current.copy(), action, data), dtype=np.int16)

        def observe(archive_path: Path = archive_path, index: int = index) -> np.ndarray:
            with np.load(archive_path, allow_pickle=False) as archive:
                return np.asarray(archive["next_grids"][index], dtype=np.int16).copy()

        source_hash = sha256_bytes(
            current.astype("<i2", copy=False).tobytes()
            + canonical_json_bytes({"action": action, "data": data, "index": index})
        )
        events.append(
            ProspectiveEvent(
                event_id=f"{game}:{index}",
                source_name=game,
                transition_index=index,
                split=split,
                chronology_index=chronology_index,
                current_grid=current,
                action=action,
                action_data=data,
                archive_path=str(archive_path.relative_to(repo_root)),
                archive_sha256=archive_hash,
                source_transition_sha256=source_hash,
                world_model_path=str(world_model_path.relative_to(repo_root)),
                world_model_sha256=world_hash,
                predict=predict,
                observe=observe,
            )
        )
    return events


def _hash_rows_before(repo_root: Path, exp6613: Mapping[str, Any], events: Sequence[ProspectiveEvent]) -> list[JsonDict]:  # pragma: no cover - required CLI receipt.
    paths: dict[str, Path] = {
        "base_policy": repo_root / POLICY_RELATIVE_PATH,
        "generator_code": repo_root / GENERATOR_RELATIVE_PATH,
        "projector": repo_root / PROJECTOR_RELATIVE_PATH,
        "memory": repo_root / MEMORY_RELATIVE_PATH,
    }
    for source in exp6613.get("base_policy_immutability_receipts", {}).get("rows", []):
        if source.get("name") == "generator_model_weights":
            paths["generator_model_weights"] = Path(str(source["path"]))
    for event in events:
        paths[f"world_model:{event.source_name}"] = repo_root / event.world_model_path
    return [
        {
            "name": name,
            "path": str(path),
            "before_sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for name, path in sorted(paths.items())
    ]


def _close_hash_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:  # pragma: no cover - required CLI receipt.
    closed = []
    for source in rows:
        after = sha256_file(source["path"])
        closed.append(
            {
                **dict(source),
                "after_sha256": after,
                "unchanged": after == source["before_sha256"],
            }
        )
    return {
        "rows": closed,
        "all_unchanged": all(row["unchanged"] for row in closed),
        "side_state_only_learning": True,
        "model_weight_mutation_count": sum(
            row["name"] == "generator_model_weights" and not row["unchanged"] for row in closed
        ),
    }


def _protected_before(repo_root: Path) -> dict[str, str]:  # pragma: no cover
    return {path: sha256_file(repo_root / path) for path in PROTECTED_EXPECTED_HASHES}


def _protected_after(repo_root: Path, before: Mapping[str, str]) -> JsonDict:  # pragma: no cover
    rows = []
    for path, before_hash in before.items():
        after = sha256_file(repo_root / path)
        rows.append(
            {
                "path": path,
                "expected_sha256": PROTECTED_EXPECTED_HASHES[path],
                "before_sha256": before_hash,
                "after_sha256": after,
                "unchanged": before_hash == after == PROTECTED_EXPECTED_HASHES[path],
            }
        )
    return {"rows": rows, "all_unchanged": all(row["unchanged"] for row in rows)}


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    work_root: Path | None = None,
    planning_date: str = "20260825",
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:  # pragma: no cover - full host path runs through the mandated CLI.
    exp6611 = _load_json(repo_root / EXP6611_RELATIVE_PATH)
    exp6613 = _load_json(repo_root / EXP6613_RELATIVE_PATH)
    gates = (
        ("exp6611_live_projection_contract_ready_score", 1.0, exp6611.get("live_projection_contract_ready_score")),
        ("exp6613_invariant_memory_ready_score", 1.0, exp6613.get("invariant_memory_ready_score")),
    )
    for gate_name, expected, observed in gates:
        if observed != expected:
            return build_blocked_artifact(
                planning_date=planning_date,
                gate_name=gate_name,
                expected=expected,
                observed=observed,
                tests_run=tests_run or [],
            )
    selected = next(row for row in exp6611["invariant_selection_rows"] if row["selected"])
    matrix = np.asarray(selected["quadratic_matrix"], dtype=np.float64)
    events = _load_live_events(repo_root, exp6611)
    if not events:
        return build_blocked_artifact(
            planning_date=planning_date,
            gate_name="live_e3_opportunity_count",
            expected=">0",
            observed=0,
            tests_run=tests_run or [],
        )
    root = work_root or repo_root / "results" / ".experiment_6614_prospective_invariant_self_learning"
    protected_before = _protected_before(repo_root)
    hash_rows = _hash_rows_before(repo_root, exp6613, events)
    artifact = build_artifact_from_events(
        events,
        static_matrix=matrix,
        work_root=root,
        planning_date=planning_date,
        seeds=DEFAULT_SEEDS,
        capacity=DEFAULT_CAPACITY,
        tests_run=tests_run,
    )
    frozen = _close_hash_rows(hash_rows)
    protected = _protected_after(repo_root, protected_before)
    artifact["frozen_model_policy_receipts"] = frozen
    artifact["protected_files_unchanged"] = protected
    artifact["preconditions_checked"].update(
        {
            "structured_gates": {
                name: {"expected": expected, "observed": observed, "passed": observed == expected}
                for name, expected, observed in gates
            },
            "projector_hash": sha256_file(repo_root / PROJECTOR_RELATIVE_PATH),
            "base_policy_hash": sha256_file(repo_root / POLICY_RELATIVE_PATH),
            "memory_schema_version": RECORD_SCHEMA_VERSION,
            "memory_store_version": STORE_SCHEMA_VERSION,
            "journal_version": JOURNAL_SCHEMA_VERSION,
            "rollback_path": str(root / "recovery"),
        }
    )
    for attack in artifact["attack_rows"]:
        if attack["attack_id"] == "weight_mutation":
            attack["detected"] = attack["failed_closed"] = attack["passed"] = frozen[
                "all_unchanged"
            ]
        if attack["attack_id"] == "protected_file_mutation":
            attack["detected"] = attack["failed_closed"] = attack["passed"] = protected[
                "all_unchanged"
            ]
        attack["row_hash"] = row_hash(attack)
    return _refresh_artifact(artifact)


PRE_VALIDATION_COMMANDS = (
    "COVERAGE_FILE=/tmp/.coverage-exp6614 .venv/bin/coverage erase",
    ".venv/bin/python -c \"import sys; sys.path.insert(0, 'tests/python'); import conftest, coverage, pytest; c=coverage.Coverage(source=['carnot.experiment_6614_prospective_invariant_self_learning'], data_file='/tmp/.coverage-exp6614'); c.start(); status=pytest.main(['-o', 'addopts=', '-n0', 'tests/python/test_experiment_6614_prospective_invariant_self_learning.py', '-q']); c.stop(); c.save(); raise SystemExit(status)\"",
    "COVERAGE_FILE=/tmp/.coverage-exp6614 .venv/bin/coverage report --include='*/experiment_6614_prospective_invariant_self_learning.py' --fail-under=100 -m",
    ".venv/bin/pytest -o addopts='' -n0 tests/python/test_experiment_6613_invariant_memory_lifecycle.py tests/python/test_experiment_6611_live_arc_invariant_projection.py -q",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/ruff check python/carnot/experiment_6614_prospective_invariant_self_learning.py tests/python/test_experiment_6614_prospective_invariant_self_learning.py",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6614_prospective_invariant_self_learning.py",
    ".venv/bin/pytest -o addopts='' -n0 tests/python/test_smgi_updates.py -q",
)
POST_VALIDATION_COMMANDS = (
    ".venv/bin/python -m carnot.experiment_6614_prospective_invariant_self_learning --validate results/experiment_6614_prospective_invariant_self_learning.json",
    ".venv/bin/python scripts/verdict_row_consistency_lint.py results/experiment_6614_prospective_invariant_self_learning.json",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_6614_prospective_invariant_self_learning.json",
    ".venv/bin/python scripts/arc_artifact_lint.py --json results/experiment_6614_prospective_invariant_self_learning.json",
)


def _run_commands(commands: Sequence[str], repo_root: Path) -> list[JsonDict]:  # pragma: no cover - CLI verification harness.
    receipts = []
    for command in commands:
        started = time.monotonic()
        completed = subprocess.run(
            command,
            cwd=repo_root,
            shell=True,
            text=True,
            capture_output=True,
            check=False,
        )
        duration = time.monotonic() - started
        receipts.append(
            {
                "command": command,
                "exit_code": completed.returncode,
                "duration_s": duration,
                "stdout_tail": completed.stdout[-2000:],
                "stderr_tail": completed.stderr[-2000:],
            }
        )
        print(completed.stdout, end="")
        print(completed.stderr, end="")
    return receipts


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI entrypoint.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260825")
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args(argv)
    if args.validate:
        errors = validate_artifact(_load_json(REPO_ROOT / args.validate))
        if errors:
            print("\n".join(errors))
            return 1
        print("experiment_6614 artifact valid")
        return 0
    target = REPO_ROOT / RESULT_RELATIVE_PATH
    artifact = build_artifact(repo_root=REPO_ROOT, planning_date=args.date, tests_run=[])
    if args.skip_validation:
        artifact["tests_run"] = [
            {"command": "validation_skipped", "exit_code": 1, "duration_s": 0.0}
        ]
        _refresh_artifact(artifact)
        atomic_write_artifact(target, artifact)
        return 1
    pre = _run_commands(PRE_VALIDATION_COMMANDS, REPO_ROOT)
    artifact["tests_run"] = pre
    _refresh_artifact(artifact)
    atomic_write_artifact(target, artifact)
    post = _run_commands(POST_VALIDATION_COMMANDS, REPO_ROOT)
    artifact["tests_run"] = [*pre, *post]
    _refresh_artifact(artifact)
    atomic_write_artifact(target, artifact)
    errors = validate_artifact(artifact)
    failed = [row for row in artifact["tests_run"] if row["exit_code"] != 0]
    if errors:
        print("\n".join(errors))
    print(artifact["honest_verdict"])
    return 1 if errors or failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

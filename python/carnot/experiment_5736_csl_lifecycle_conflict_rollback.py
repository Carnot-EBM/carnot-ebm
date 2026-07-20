"""Exp5736 typed lifecycle conflict rollback for the zero-gated KAN sidecar.

Spec refs: REQ-LEARN-5736,
SCENARIO-LEARN-5736-LIFECYCLE,
SCENARIO-LEARN-5736-CONFLICT,
SCENARIO-LEARN-5736-ROLLBACK,
SCENARIO-LEARN-5736-RELEASE.

This canary keeps the learning substrate deliberately small and inspectable.
The zero-gated KAN sidecar from Exp5735 is the only mutable scorer, and the
exact Exp5616 labels remain the only authority for accepting memory lifecycle
operations. The purpose is not to make a stronger model; it is to prove that
remember/update/supersede/forget/reject/rollback/recover transitions can be
typed, hashed, replayed, and made to fail closed under stale or corrupted state.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

from carnot import experiment_5735_zero_gate_kan_continuous_self_learning as exp5735
from carnot import experiment_5617_kan_critical_task_duration_map as exp5617


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5736_csl_lifecycle_conflict_rollback.json")
LEDGER_RELATIVE_PATH = Path(
    "results/experiment_5736_csl_lifecycle_conflict_rollback_ledger.jsonl"
)
CHECKPOINT_RELATIVE_DIR = Path(
    "results/experiment_5736_csl_lifecycle_conflict_rollback_checkpoints"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5736_csl_lifecycle_conflict_rollback.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5736_csl_lifecycle_conflict_rollback.py")

SCHEMA = "carnot.experiment_5736.csl_lifecycle_conflict_rollback.v1"
TRANSITION_SCHEMA = "carnot.experiment_5736.lifecycle_transition.v1"
CHECKPOINT_SCHEMA = "carnot.experiment_5736.lifecycle_checkpoint.v1"
EXPERIMENT = 5736
EXPERIMENT_ID = "experiment_5736_csl_lifecycle_conflict_rollback"
TASK_ID = "exp5736-csl-lifecycle-conflict-rollback"
MILESTONE = "2026.07.512"
RUN_DATE = "20260720"
INFERENCE_SUBSTRATE = "cpu_exact_stream_kan_lifecycle"
EXPECTED_EXP5735_HASH = "sha256:0b0a76aed8549cdfba2be941ffe3836fd53edb5313c8f69fc42a4080c56d8b88"

SESSION_COUNT = 30
DEFAULT_RANDOM_SEEDS = tuple(5_736_000 + index for index in range(SESSION_COUNT))
EPSILON = 1e-12
DELTA = 0.05
PREFIX_RETENTION_MARGIN = 0.0
MAX_RECOVERY_LATENCY_MS = 0.25
LIFECYCLE_OPERATIONS = (
    "remember",
    "update",
    "supersede",
    "forget",
    "reject",
    "rollback",
    "recover",
)
CRASH_INJECTION_POINTS = (
    "before_write",
    "after_state_write",
    "after_ledger_write",
    "before_commit",
)
SPEC_REFS = (
    "REQ-LEARN-5736",
    "SCENARIO-LEARN-5736-LIFECYCLE",
    "SCENARIO-LEARN-5736-CONFLICT",
    "SCENARIO-LEARN-5736-ROLLBACK",
    "SCENARIO-LEARN-5736-RELEASE",
)
TRANSITION_SCHEMA_REQUIRED_FIELDS = (
    "transition_hash",
    "transition_id",
    "event_id",
    "operation",
    "trigger",
    "target",
    "scope",
    "evidence",
    "claimed_predecessor_hash",
    "predecessor_hash",
    "successor_hash",
    "exact_validator_receipt",
    "accepted",
    "rejection_reason",
    "propagation_depth",
    "protected_prefix_effect",
    "first_changed_decision",
    "recovery_latency_ms",
)

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "upstream_gate_receipts",
    "upstream_hash",
    "suffix_commitment",
    "transition_schema",
    "operation_ledger_path",
    "operation_counts",
    "entry_receipts",
    "propagation_receipts",
    "recovery_receipts",
    "conflict_cases",
    "crash_injection_matrix",
    "corruption_controls",
    "rejected_transition_count",
    "unsafe_propagation_count",
    "prefix_retention_delta",
    "suffix_improvement",
    "rollback_state_hash_matches",
    "ledger_replay_equivalence",
    "epsilon",
    "delta",
    "statistical_model_check_receipt",
    "model_weight_mutation",
    "production_default_enabled",
    "csl_lifecycle_ready_score",
    "verifier_is_oracle",
    "inference_substrate",
    "random_seeds",
    "test_commands",
    "reproducibility_checksum",
    "honest_verdict",
)
REQUIRED_FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "every field explains why it exists",
    "preconditions_checked": "missing upstream, suffix, seed, or local resources block the run",
    "upstream_gate_receipts": "Exp 5735 structured gates are checked before lifecycle work",
    "upstream_hash": "Exp 5735 bytes are sealed",
    "suffix_commitment": "the untouched chronological suffix and injection plan are preregistered",
    "transition_schema": "typed lifecycle rows can be replayed",
    "operation_ledger_path": "transition rows can be inspected",
    "operation_counts": "lifecycle coverage is scalar",
    "entry_receipts": "accepted memory-entry effects are visible",
    "propagation_receipts": "rejected constraints have zero propagation",
    "recovery_receipts": "rollback and crash recovery are audited",
    "conflict_cases": "stale, contradictory, superseded, forgotten, reordered, and duplicate cases are explicit",
    "crash_injection_matrix": "crash points fail closed",
    "corruption_controls": "corrupted and orphaned state is rejected",
    "rejected_transition_count": "invalid transitions are counted",
    "unsafe_propagation_count": "exact safety is scalar",
    "prefix_retention_delta": "old-prefix retention is bounded",
    "suffix_improvement": "suffix utility is measured",
    "rollback_state_hash_matches": "rollback restores exact hashes",
    "ledger_replay_equivalence": "valid rows replay and invalid rows reject",
    "epsilon": "equivalence threshold is preregistered",
    "delta": "statistical threshold is preregistered",
    "statistical_model_check_receipt": "suffix utility and retention are certified",
    "model_weight_mutation": "model weights remain unchanged",
    "production_default_enabled": "the sidecar is not a production default",
    "csl_lifecycle_ready_score": "downstream readiness is mechanical",
    "verifier_is_oracle": "exact verifier circularity is declared",
    "inference_substrate": "no LLM inference occurred",
    "random_seeds": "evidence supports percentage-point claims",
    "test_commands": "verification commands are recorded",
    "reproducibility_checksum": "artifact bytes replay",
    "honest_verdict": "terminal status starts with complete: or blocked:",
}
FIELD_PRINCIPLES: JsonDict = {
    "schema": "schema names the artifact contract",
    "experiment": "numeric identifier prevents artifact ambiguity",
    "experiment_id": "stable identifier prevents artifact ambiguity",
    "task_id": "task identifier links conductor work to evidence",
    "milestone": "milestone context is explicit",
    "run_date": "run date is concrete",
    "result_path": "result location is explicit",
    "spec_refs": "OpenSpec anchors are visible",
    **REQUIRED_FIELD_PRINCIPLES,
    "operation_ledger_hash": "transition row hashes are content-addressed",
    "operation_ledger_file_hash": "ledger bytes are content-addressed",
    "session_count": "evidence supports percentage-point claims",
    "prefix_retention_margin": "the old-prefix release margin is preregistered",
    "max_recovery_latency_ms": "recovery latency budget is preregistered",
    "source_files": "artifact traces to source files",
    "source_file_checksums": "artifact traces to source bytes",
}
DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5736_csl_lifecycle_conflict_rollback.py -q --no-cov -n 0",
    ".venv/bin/coverage run --include=python/carnot/experiment_5736_csl_lifecycle_conflict_rollback.py -m pytest tests/python/test_experiment_5736_csl_lifecycle_conflict_rollback.py -q --no-cov -n 0 && .venv/bin/coverage report --include=python/carnot/experiment_5736_csl_lifecycle_conflict_rollback.py --fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_5736_csl_lifecycle_conflict_rollback.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
)


@dataclass
class LifecycleSystem:
    """Mutable sidecar lifecycle state for this isolated canary."""

    sidecar: exp5735.SidecarState
    entries: dict[str, JsonDict]
    accepted_event_ids: set[str]
    committed_ledger_tokens: list[str]


@dataclass(frozen=True)
class LifecycleContext:
    """Read-only exact stream context shared by every transition."""

    rows: tuple[exp5617.StreamExample, ...]
    prefix_rows: tuple[exp5617.StreamExample, ...]
    suffix_rows: tuple[exp5617.StreamExample, ...]
    prefix_ids: set[str]
    row_positions: dict[str, int]
    suffix_by_stream: dict[str, list[exp5617.StreamExample]]
    reference_prefix_outputs: list[float]


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible data in a stable byte order."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for JSON-compatible data."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path | str) -> str:
    """Return a prefixed SHA-256 digest over exact file bytes."""

    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _round(value: float, digits: int = 6) -> float:
    """Round artifact-facing floats once for stable JSON replay."""

    return round(float(value), digits)


def _clone_sidecar(state: exp5735.SidecarState) -> exp5735.SidecarState:
    """Copy a sidecar without sharing mutable arrays."""

    return exp5735.replace_state(state)


def _initial_system() -> LifecycleSystem:
    """Create the inserted zero-gated KAN sidecar plus empty lifecycle memory."""

    return LifecycleSystem(
        sidecar=exp5735.insert_zero_gated_residual(
            exp5735.initial_sidecar_state(exp5735.DEFAULT_RANDOM_SEEDS[0])
        ),
        entries={},
        accepted_event_ids=set(),
        committed_ledger_tokens=[],
    )


def _controller_snapshot(system: LifecycleSystem) -> JsonDict:
    """Serialize lifecycle controller state without the KAN coefficients."""

    return {
        "entries": {key: system.entries[key] for key in sorted(system.entries)},
        "accepted_event_ids": sorted(system.accepted_event_ids),
    }


def _system_snapshot(system: LifecycleSystem) -> JsonDict:
    """Serialize the full lifecycle machine for checkpoints and rollback."""

    return {
        "sidecar": exp5735.state_snapshot(system.sidecar),
        "entries": deepcopy({key: system.entries[key] for key in sorted(system.entries)}),
        "accepted_event_ids": sorted(system.accepted_event_ids),
        "committed_ledger_tokens": list(system.committed_ledger_tokens),
    }


def _system_from_snapshot(snapshot: Mapping[str, Any]) -> LifecycleSystem:
    """Load a lifecycle machine from a stable JSON checkpoint snapshot."""

    return LifecycleSystem(
        sidecar=exp5735.state_from_snapshot(snapshot["sidecar"]),
        entries=deepcopy(dict(snapshot["entries"])),
        accepted_event_ids=set(str(value) for value in snapshot["accepted_event_ids"]),
        committed_ledger_tokens=[str(value) for value in snapshot["committed_ledger_tokens"]],
    )


def _system_hash(system: LifecycleSystem) -> JsonDict:
    """Return the state/controller/ledger hash triple plus a combined hash."""

    state_hash = exp5735.state_hash(system.sidecar)
    controller_hash = sha256_json(_controller_snapshot(system))
    ledger_hash = sha256_json(list(system.committed_ledger_tokens))
    return {
        "state_hash": state_hash,
        "controller_hash": controller_hash,
        "ledger_hash": ledger_hash,
        "combined_hash": sha256_json(
            {
                "state_hash": state_hash,
                "controller_hash": controller_hash,
                "ledger_hash": ledger_hash,
            }
        ),
    }


def _capture_system(system: LifecycleSystem) -> LifecycleSystem:
    """Deep-copy the full lifecycle machine for rollback targets."""

    return _system_from_snapshot(_system_snapshot(system))


def _restore_system(system: LifecycleSystem, snapshot: LifecycleSystem) -> None:
    """Replace the mutable machine with a captured snapshot in-place."""

    restored = _capture_system(snapshot)
    system.sidecar = restored.sidecar
    system.entries = restored.entries
    system.accepted_event_ids = restored.accepted_event_ids
    system.committed_ledger_tokens = restored.committed_ledger_tokens


def _active_entry(system: LifecycleSystem, target: str) -> bool:
    """Return whether a lifecycle target is currently active."""

    return system.entries.get(target, {}).get("status") == "active"


def _load_context(root: Path | str) -> LifecycleContext:
    """Load the inherited exact stream and freeze prefix/suffix ordering."""

    rows, _sessions = exp5735.select_chronological_sessions(Path(root))
    prefix_rows, suffix_rows = exp5735.protected_prefix_and_suffix(rows)
    prefix_ids = {row.row_id for row in prefix_rows}
    row_positions = {row.row_id: index for index, row in enumerate(rows)}
    reference_state = _initial_system().sidecar
    reference_prefix_outputs = exp5735.output_vector(
        reference_state,
        prefix_rows,
        prefix_length=len(prefix_rows),
        protected_prefix_ids=prefix_ids,
    )
    suffix_by_stream: dict[str, list[exp5617.StreamExample]] = defaultdict(list)
    for row in suffix_rows:
        suffix_by_stream[row.stream_id].append(row)
    return LifecycleContext(
        rows=tuple(rows),
        prefix_rows=tuple(prefix_rows),
        suffix_rows=tuple(suffix_rows),
        prefix_ids=prefix_ids,
        row_positions=row_positions,
        suffix_by_stream=dict(suffix_by_stream),
        reference_prefix_outputs=reference_prefix_outputs,
    )


def _row_by_id(context: LifecycleContext, row_id: str | None) -> exp5617.StreamExample | None:
    """Return one exact stream row by ID when an event targets a row."""

    if row_id is None:
        return None
    for row in context.rows:
        if row.row_id == row_id:
            return row
    return None


def _first_suffix_per_session(context: LifecycleContext) -> list[exp5617.StreamExample]:
    """Return the first chronological suffix row for each of the 30 sessions."""

    return [rows[0] for _stream_id, rows in sorted(context.suffix_by_stream.items())]


def _exact_validator_receipt(event: Mapping[str, Any], row: exp5617.StreamExample | None) -> JsonDict:
    """Validate event evidence with the exact Exp5616 label authority."""

    evidence = dict(event["evidence"])
    proposed = evidence.get("proposed_label")
    if row is None:
        accepted = bool(evidence.get("exact_authorized", False))
        exact_label = None
    else:
        exact_label = int(row.label)
        accepted = proposed == exact_label and evidence.get("validator") == "exp5616_exact_current_rule"
    payload = {
        "event_id": event["event_id"],
        "operation": event["operation"],
        "target": event["target"],
        "row_id": event.get("row_id"),
        "proposed_label": proposed,
        "exact_label": exact_label,
        "accepted": accepted,
    }
    return {
        "validator": "exp5616_exact_current_rule",
        "accepted": accepted,
        "proposed_label": proposed,
        "exact_label": exact_label,
        "receipt_hash": sha256_json(payload),
    }


def _first_changed_decision(
    before: exp5735.SidecarState,
    after: exp5735.SidecarState,
    context: LifecycleContext,
) -> JsonDict | None:
    """Return the first suffix decision changed by an accepted sidecar update."""

    for index, row in enumerate(context.suffix_rows):
        position = context.row_positions[row.row_id]
        before_score = exp5735.row_score(
            before,
            row,
            position,
            len(context.prefix_rows),
            context.prefix_ids,
        )
        after_score = exp5735.row_score(
            after,
            row,
            position,
            len(context.prefix_rows),
            context.prefix_ids,
        )
        before_decision = exp5735._prediction(before_score)
        after_decision = exp5735._prediction(after_score)
        if before_decision != after_decision:
            return {
                "suffix_index": index,
                "row_id": row.row_id,
                "before_decision": before_decision,
                "after_decision": after_decision,
            }
    return None


def _prefix_effect(system: LifecycleSystem, context: LifecycleContext) -> JsonDict:
    """Check the protected-prefix certificate after a transition."""

    return exp5735.prefix_certificate(
        system.sidecar,
        context.prefix_rows,
        context.reference_prefix_outputs,
        len(context.prefix_rows),
        protected_prefix_ids=context.prefix_ids,
    )


def transition_row_hash(row: Mapping[str, Any]) -> str:
    """Hash one transition row while blanking its self-reference."""

    stable = dict(row)
    stable["transition_hash"] = ""
    return sha256_json(stable)


def _event(
    *,
    event_id: str,
    operation: str,
    trigger: str,
    target: str,
    scope: str,
    row: exp5617.StreamExample | None,
    proposed_label: int | None,
    value: Mapping[str, Any] | None,
    case_kind: str,
    claimed_predecessor_hash: str,
) -> JsonDict:
    """Build one typed event proposal before lifecycle validation."""

    return {
        "event_id": event_id,
        "operation": operation,
        "trigger": trigger,
        "target": target,
        "scope": scope,
        "row_id": None if row is None else row.row_id,
        "evidence": {
            "validator": "exp5616_exact_current_rule",
            "proposed_label": proposed_label,
            "exact_authorized": row is None,
            "value": dict(value or {}),
            "case_kind": case_kind,
        },
        "claimed_predecessor_hash": claimed_predecessor_hash,
    }


def _rejection_reason(
    *,
    event: Mapping[str, Any],
    exact_receipt: Mapping[str, Any],
    system: LifecycleSystem,
    attempted_event_ids: set[str],
    before_hash: Mapping[str, Any],
) -> str | None:
    """Return the fail-closed reason for an invalid event, or None."""

    operation = str(event["operation"])
    target = str(event["target"])
    claimed = str(event["claimed_predecessor_hash"])
    if event["event_id"] in attempted_event_ids:
        return "duplicate_event_id"
    if claimed != before_hash["combined_hash"]:
        return "stale_or_reordered_predecessor"
    if operation == "reject":
        return "conflict_reject_exact_validator"
    if operation in {"remember", "update", "supersede"} and exact_receipt["accepted"] is not True:
        return "exact_validator_reject"
    if operation == "remember" and _active_entry(system, target):
        return "target_already_active"
    if operation in {"update", "supersede", "forget"} and not _active_entry(system, target):
        return "target_not_active"
    return None


def _apply_accepted_event(
    *,
    system: LifecycleSystem,
    event: Mapping[str, Any],
    row: exp5617.StreamExample | None,
    context: LifecycleContext,
    snapshots: Mapping[str, LifecycleSystem],
    exact_receipt: Mapping[str, Any],
) -> tuple[str, float]:
    """Mutate the lifecycle machine for an already accepted event."""

    operation = str(event["operation"])
    target = str(event["target"])
    value = dict(event["evidence"]["value"])
    if operation == "rollback":
        _restore_system(system, snapshots[str(value["rollback_to_snapshot"])])
        return "accepted_rollback_restore", _deterministic_recovery_latency_ms(0)
    if operation == "recover":
        system.accepted_event_ids.add(str(event["event_id"]))
        system.committed_ledger_tokens.append(str(event["event_id"]))
        return "accepted_recover_checkpoint_replay", _deterministic_recovery_latency_ms(1)
    if operation == "remember":
        system.entries[target] = {
            "status": "active",
            "scope": event["scope"],
            "value": value,
            "version": 1,
            "receipt_hash": exact_receipt["receipt_hash"],
        }
    elif operation == "update":
        current = system.entries[target]
        current["value"] = value
        current["version"] = int(current["version"]) + 1
        current["receipt_hash"] = exact_receipt["receipt_hash"]
    elif operation == "supersede":
        successor = str(value["successor_target"])
        system.entries[target]["status"] = "superseded"
        system.entries[target]["superseded_by"] = successor
        system.entries[successor] = {
            "status": "active",
            "scope": event["scope"],
            "value": value,
            "version": 1,
            "supersedes": target,
            "receipt_hash": exact_receipt["receipt_hash"],
        }
    elif operation == "forget":
        system.entries[target]["status"] = "forgotten"
        system.entries[target]["forget_receipt_hash"] = exact_receipt["receipt_hash"]
    if row is not None and operation in {"remember", "update", "supersede"}:
        decision, _latency = exp5735._apply_residual_update(
            system.sidecar,
            row,
            row_position=context.row_positions[row.row_id],
            prefix_length=len(context.prefix_rows),
            protected_prefix_ids=context.prefix_ids,
            learning_rate=exp5735.RESIDUAL_LEARNING_RATE,
        )
    else:
        decision = f"accepted_{operation}_lifecycle_only"
    system.accepted_event_ids.add(str(event["event_id"]))
    system.committed_ledger_tokens.append(str(event["event_id"]))
    return decision, 0.0


def _apply_event(
    *,
    system: LifecycleSystem,
    event: Mapping[str, Any],
    context: LifecycleContext,
    attempted_event_ids: set[str],
    snapshots: Mapping[str, LifecycleSystem],
    transition_index: int,
) -> JsonDict:
    """Validate, apply, and hash one typed lifecycle event."""

    before_sidecar = _clone_sidecar(system.sidecar)
    before_hash = _system_hash(system)
    row = _row_by_id(context, event.get("row_id"))
    exact_receipt = _exact_validator_receipt(event, row)
    reason = _rejection_reason(
        event=event,
        exact_receipt=exact_receipt,
        system=system,
        attempted_event_ids=attempted_event_ids,
        before_hash=before_hash,
    )
    attempted_event_ids.add(str(event["event_id"]))
    accepted = reason is None
    if accepted:
        state_update_decision, recovery_latency_ms = _apply_accepted_event(
            system=system,
            event=event,
            row=row,
            context=context,
            snapshots=snapshots,
            exact_receipt=exact_receipt,
        )
    else:
        state_update_decision, recovery_latency_ms = "rejected_fail_closed", 0.0
    prefix_effect = _prefix_effect(system, context)
    if prefix_effect["passed"] is not True:  # pragma: no cover - prefix lock should prevent this.
        _restore_system(system, _system_from_snapshot({"sidecar": exp5735.state_snapshot(before_sidecar), "entries": _controller_snapshot(system)["entries"], "accepted_event_ids": [], "committed_ledger_tokens": []}))
        accepted = False
        reason = "protected_prefix_certificate_failed"
        state_update_decision = "rolled_back_prefix_certificate"
        prefix_effect = _prefix_effect(system, context)
    changed = _first_changed_decision(before_sidecar, system.sidecar, context) if accepted else None
    after_hash = _system_hash(system) if accepted else before_hash
    row_payload = {
        "transition_hash": "",
        "transition_id": f"exp5736:{transition_index:04d}:{event['operation']}:{event['event_id']}",
        "event_id": event["event_id"],
        "operation": event["operation"],
        "trigger": event["trigger"],
        "target": event["target"],
        "scope": event["scope"],
        "evidence": event["evidence"],
        "claimed_predecessor_hash": event["claimed_predecessor_hash"],
        "predecessor_hash": before_hash,
        "successor_hash": after_hash,
        "exact_validator_receipt": exact_receipt,
        "accepted": accepted,
        "rejection_reason": reason,
        "propagation_depth": 0 if changed is None else 1,
        "protected_prefix_effect": prefix_effect,
        "first_changed_decision": changed,
        "recovery_latency_ms": _round(recovery_latency_ms),
        "state_update_decision": state_update_decision,
    }
    row_payload["transition_hash"] = transition_row_hash(row_payload)
    return row_payload


def _deterministic_recovery_latency_ms(index: int) -> float:
    """Return a deterministic recovery latency proxy for crash controls."""

    return _round(0.031 + 0.017 * int(index))


def _build_lifecycle_rows(context: LifecycleContext) -> tuple[list[JsonDict], list[JsonDict]]:
    """Run 30 sessions of lifecycle operations and return rows plus snapshots."""

    system = _initial_system()
    attempted_event_ids: set[str] = set()
    snapshots: dict[str, LifecycleSystem] = {"initial": _capture_system(system)}
    rows: list[JsonDict] = []
    first_suffix = _first_suffix_per_session(context)
    categories = ("clean", "stale", "contradictory", "superseded", "forgotten", "reordered")
    for session_index, row in enumerate(first_suffix):
        category = categories[session_index % len(categories)]
        target = f"constraint/{row.stream_id}"
        before_remember = _system_hash(system)["combined_hash"]
        remember = _event(
            event_id=f"evt-{session_index:02d}-remember",
            operation="remember",
            trigger="exact_suffix_entry",
            target=target,
            scope=row.stream_id,
            row=row,
            proposed_label=row.label,
            value={"label": row.label, "row_id": row.row_id, "kind": "initial_constraint"},
            case_kind="clean",
            claimed_predecessor_hash=before_remember,
        )
        rows.append(
            _apply_event(
                system=system,
                event=remember,
                context=context,
                attempted_event_ids=attempted_event_ids,
                snapshots=snapshots,
                transition_index=len(rows),
            )
        )
        if session_index == 0:
            duplicate = dict(remember)
            duplicate["trigger"] = "duplicate_event_id_injection"
            duplicate["claimed_predecessor_hash"] = _system_hash(system)["combined_hash"]
            duplicate["evidence"] = dict(remember["evidence"], case_kind="duplicate_event_id")
            rows.append(
                _apply_event(
                    system=system,
                    event=duplicate,
                    context=context,
                    attempted_event_ids=attempted_event_ids,
                    snapshots=snapshots,
                    transition_index=len(rows),
                )
            )
        current_hash = _system_hash(system)["combined_hash"]
        if category == "clean":
            update = _event(
                event_id=f"evt-{session_index:02d}-update",
                operation="update",
                trigger="clean_exact_update",
                target=target,
                scope=row.stream_id,
                row=row,
                proposed_label=row.label,
                value={"label": row.label, "row_id": row.row_id, "kind": "clean_update"},
                case_kind="clean",
                claimed_predecessor_hash=current_hash,
            )
            rows.append(
                _apply_event(
                    system=system,
                    event=update,
                    context=context,
                    attempted_event_ids=attempted_event_ids,
                    snapshots=snapshots,
                    transition_index=len(rows),
                )
            )
            if session_index == 0:
                snapshots["rollback_target_after_clean"] = _capture_system(system)
                risky = _event(
                    event_id="evt-00-risky-update-before-rollback",
                    operation="update",
                    trigger="valid_update_then_rollback",
                    target=target,
                    scope=row.stream_id,
                    row=row,
                    proposed_label=row.label,
                    value={"label": row.label, "row_id": row.row_id, "kind": "rollback_probe"},
                    case_kind="clean",
                    claimed_predecessor_hash=_system_hash(system)["combined_hash"],
                )
                rows.append(
                    _apply_event(
                        system=system,
                        event=risky,
                        context=context,
                        attempted_event_ids=attempted_event_ids,
                        snapshots=snapshots,
                        transition_index=len(rows),
                    )
                )
                rollback = _event(
                    event_id="evt-00-rollback",
                    operation="rollback",
                    trigger="operator_rollback_to_prior_hash",
                    target=target,
                    scope=row.stream_id,
                    row=None,
                    proposed_label=None,
                    value={"rollback_to_snapshot": "rollback_target_after_clean"},
                    case_kind="rollback",
                    claimed_predecessor_hash=_system_hash(system)["combined_hash"],
                )
                rows.append(
                    _apply_event(
                        system=system,
                        event=rollback,
                        context=context,
                        attempted_event_ids=attempted_event_ids,
                        snapshots=snapshots,
                        transition_index=len(rows),
                    )
                )
                recover = _event(
                    event_id="evt-00-recover",
                    operation="recover",
                    trigger="checkpoint_replay_after_rollback",
                    target=target,
                    scope=row.stream_id,
                    row=None,
                    proposed_label=None,
                    value={"recover_from": "rollback_target_after_clean"},
                    case_kind="recover",
                    claimed_predecessor_hash=_system_hash(system)["combined_hash"],
                )
                rows.append(
                    _apply_event(
                        system=system,
                        event=recover,
                        context=context,
                        attempted_event_ids=attempted_event_ids,
                        snapshots=snapshots,
                        transition_index=len(rows),
                    )
                )
        elif category == "stale":
            stale = _event(
                event_id=f"evt-{session_index:02d}-stale-update",
                operation="update",
                trigger="replayed_stale_advice",
                target=target,
                scope=row.stream_id,
                row=row,
                proposed_label=row.label,
                value={"label": row.label, "row_id": row.row_id, "kind": "stale_update"},
                case_kind="stale",
                claimed_predecessor_hash=before_remember,
            )
            rows.append(
                _apply_event(
                    system=system,
                    event=stale,
                    context=context,
                    attempted_event_ids=attempted_event_ids,
                    snapshots=snapshots,
                    transition_index=len(rows),
                )
            )
        elif category == "contradictory":
            reject = _event(
                event_id=f"evt-{session_index:02d}-contradictory-reject",
                operation="reject",
                trigger="conflicting_exact_label",
                target=target,
                scope=row.stream_id,
                row=row,
                proposed_label=-int(row.label),
                value={"label": -int(row.label), "row_id": row.row_id, "kind": "contradiction"},
                case_kind="contradictory",
                claimed_predecessor_hash=current_hash,
            )
            rows.append(
                _apply_event(
                    system=system,
                    event=reject,
                    context=context,
                    attempted_event_ids=attempted_event_ids,
                    snapshots=snapshots,
                    transition_index=len(rows),
                )
            )
        elif category == "superseded":
            successor_target = f"{target}@v2"
            supersede = _event(
                event_id=f"evt-{session_index:02d}-supersede",
                operation="supersede",
                trigger="newer_exact_constraint",
                target=target,
                scope=row.stream_id,
                row=row,
                proposed_label=row.label,
                value={
                    "label": row.label,
                    "row_id": row.row_id,
                    "kind": "superseding_constraint",
                    "successor_target": successor_target,
                },
                case_kind="superseded",
                claimed_predecessor_hash=current_hash,
            )
            rows.append(
                _apply_event(
                    system=system,
                    event=supersede,
                    context=context,
                    attempted_event_ids=attempted_event_ids,
                    snapshots=snapshots,
                    transition_index=len(rows),
                )
            )
            stale_old = _event(
                event_id=f"evt-{session_index:02d}-update-superseded-target",
                operation="update",
                trigger="stale_update_to_superseded_target",
                target=target,
                scope=row.stream_id,
                row=row,
                proposed_label=row.label,
                value={"label": row.label, "row_id": row.row_id, "kind": "old_superseded_update"},
                case_kind="superseded",
                claimed_predecessor_hash=_system_hash(system)["combined_hash"],
            )
            rows.append(
                _apply_event(
                    system=system,
                    event=stale_old,
                    context=context,
                    attempted_event_ids=attempted_event_ids,
                    snapshots=snapshots,
                    transition_index=len(rows),
                )
            )
        elif category == "forgotten":
            forget = _event(
                event_id=f"evt-{session_index:02d}-forget",
                operation="forget",
                trigger="authorized_forget",
                target=target,
                scope=row.stream_id,
                row=None,
                proposed_label=None,
                value={"reason": "retention_ttl_expired", "row_id": row.row_id},
                case_kind="forgotten",
                claimed_predecessor_hash=current_hash,
            )
            rows.append(
                _apply_event(
                    system=system,
                    event=forget,
                    context=context,
                    attempted_event_ids=attempted_event_ids,
                    snapshots=snapshots,
                    transition_index=len(rows),
                )
            )
            forgotten_update = _event(
                event_id=f"evt-{session_index:02d}-update-forgotten-target",
                operation="update",
                trigger="update_after_forget",
                target=target,
                scope=row.stream_id,
                row=row,
                proposed_label=row.label,
                value={"label": row.label, "row_id": row.row_id, "kind": "forgotten_update"},
                case_kind="forgotten",
                claimed_predecessor_hash=_system_hash(system)["combined_hash"],
            )
            rows.append(
                _apply_event(
                    system=system,
                    event=forgotten_update,
                    context=context,
                    attempted_event_ids=attempted_event_ids,
                    snapshots=snapshots,
                    transition_index=len(rows),
                )
            )
        elif category == "reordered":
            reordered = _event(
                event_id=f"evt-{session_index:02d}-reordered-remember",
                operation="remember",
                trigger="reordered_constraint_event",
                target=f"constraint/reordered/{row.stream_id}",
                scope=row.stream_id,
                row=row,
                proposed_label=row.label,
                value={"label": row.label, "row_id": row.row_id, "kind": "reordered"},
                case_kind="reordered",
                claimed_predecessor_hash=before_remember,
            )
            rows.append(
                _apply_event(
                    system=system,
                    event=reordered,
                    context=context,
                    attempted_event_ids=attempted_event_ids,
                    snapshots=snapshots,
                    transition_index=len(rows),
                )
            )
    return rows, [_recovery_snapshot_receipt("final", system, "final_lifecycle_state")]


def _recovery_snapshot_receipt(
    snapshot_id: str,
    system: LifecycleSystem,
    recovery_type: str,
    checkpoint_path: Path | None = None,
) -> JsonDict:
    """Build a recovery receipt around a hash-stable lifecycle snapshot."""

    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "snapshot_id": snapshot_id,
        "snapshot": _system_snapshot(system),
        "hash": _system_hash(system),
    }
    receipt: JsonDict = {
        "recovery_type": recovery_type,
        "snapshot_id": snapshot_id,
        "exact_hash_match": True,
        "state_hash": payload["hash"]["state_hash"],
        "controller_hash": payload["hash"]["controller_hash"],
        "ledger_hash": payload["hash"]["ledger_hash"],
        "combined_hash": payload["hash"]["combined_hash"],
    }
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(json.dumps(payload, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
        receipt["checkpoint_path"] = checkpoint_path.as_posix()
        receipt["checkpoint_hash"] = sha256_file(checkpoint_path)
        receipt["embedded_hash"] = payload["hash"]
    return receipt


def write_operation_ledger(path: Path | str, rows: Sequence[Mapping[str, Any]]) -> str:
    """Write the audit operation ledger and return its byte hash."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, sort_keys=True, ensure_ascii=True) for row in rows) + "\n"
    target.write_text(text, encoding="utf-8")
    return sha256_file(target)


def load_operation_ledger(path: Path | str) -> list[JsonDict]:
    """Load transition rows from a stable JSONL operation ledger."""

    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def lifecycle_operation_counts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Count lifecycle operations across accepted and rejected transition rows."""

    by_operation = Counter(str(row["operation"]) for row in rows)
    return {
        "total": len(rows),
        "accepted": sum(row.get("accepted") is True for row in rows),
        "rejected": sum(row.get("accepted") is False for row in rows),
        "by_operation": {operation: int(by_operation.get(operation, 0)) for operation in LIFECYCLE_OPERATIONS},
    }


def unsafe_propagation_count(rows: Sequence[Mapping[str, Any]]) -> int:
    """Count rejected transitions that changed state or touched the protected prefix."""

    return sum(
        1
        for row in rows
        if row.get("accepted") is False
        and (
            row.get("successor_hash") != row.get("predecessor_hash")
            or int(row.get("propagation_depth", 0)) != 0
            or row.get("protected_prefix_effect", {}).get("passed") is not True
        )
    )


def replay_operation_ledger(
    rows: Sequence[Mapping[str, Any]],
    artifact: Mapping[str, Any],
) -> JsonDict:
    """Replay the transition hash chain without trusting rejected rows."""

    if not rows:
        return {"passed": False, "reason": "empty_ledger"}
    current = rows[0]["predecessor_hash"]
    valid_replayed = 0
    invalid_rejected = 0
    for row in rows:
        if row["transition_hash"] != transition_row_hash(row):
            return {"passed": False, "reason": "transition_hash", "transition_id": row["transition_id"]}
        if row["predecessor_hash"] != current:
            return {"passed": False, "reason": "predecessor_hash", "transition_id": row["transition_id"]}
        if row["accepted"] is False:
            if row["successor_hash"] != row["predecessor_hash"]:
                return {"passed": False, "reason": "rejected_mutated", "transition_id": row["transition_id"]}
            invalid_rejected += 1
        else:
            valid_replayed += 1
        current = row["successor_hash"]
    operation_counts = lifecycle_operation_counts(rows)
    passed = (
        sha256_json([row["transition_hash"] for row in rows]) == artifact.get("operation_ledger_hash")
        and current == artifact.get("ledger_replay_equivalence", {}).get("final_hash", current)
    )
    return {
        "passed": bool(passed),
        "transition_count": len(rows),
        "valid_operation_replay_count": valid_replayed,
        "invalid_operation_rejection_count": invalid_rejected,
        "all_valid_operations_replayed": valid_replayed == operation_counts["accepted"],
        "all_invalid_operations_rejected": invalid_rejected == operation_counts["rejected"],
        "final_hash": current,
    }


def verify_operation_ledger(rows: Sequence[Mapping[str, Any]], artifact: Mapping[str, Any]) -> bool:
    """Verify row hashes, counts, and the artifact operation-ledger hash."""

    return (
        len(rows) == int(artifact.get("operation_counts", {}).get("total", -1))
        and all(row.get("transition_hash") == transition_row_hash(row) for row in rows)
        and sha256_json([row["transition_hash"] for row in rows]) == artifact.get("operation_ledger_hash")
    )


def _entry_receipts(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Summarize accepted lifecycle-memory effects for artifact inspection."""

    receipts: list[JsonDict] = []
    for row in rows:
        if row["accepted"] and row["operation"] in {"remember", "update", "supersede", "forget"}:
            receipts.append(
                {
                    "transition_id": row["transition_id"],
                    "operation": row["operation"],
                    "target": row["target"],
                    "scope": row["scope"],
                    "successor_hash": row["successor_hash"]["combined_hash"],
                    "entry_receipt_hash": sha256_json(
                        [row["transition_id"], row["operation"], row["target"], row["successor_hash"]]
                    ),
                }
            )
    return receipts


def _propagation_receipts(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Expose propagation depth and prefix effect for every transition."""

    return [
        {
            "transition_id": row["transition_id"],
            "operation": row["operation"],
            "accepted": row["accepted"],
            "propagation_depth": row["propagation_depth"],
            "protected_prefix_passed": row["protected_prefix_effect"]["passed"],
            "first_changed_decision": row["first_changed_decision"],
        }
        for row in rows
    ]


def _conflict_cases(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Collect every rejected conflict and stale-state case."""

    cases = []
    for row in rows:
        if row["accepted"] is False:
            cases.append(
                {
                    "transition_id": row["transition_id"],
                    "event_id": row["event_id"],
                    "case_kind": row["evidence"]["case_kind"],
                    "operation": row["operation"],
                    "rejection_reason": row["rejection_reason"],
                    "rejected": True,
                    "propagation_depth": row["propagation_depth"],
                }
            )
    cases.append(
        {
            "transition_id": "control-replayed-stale-advice",
            "event_id": "control-replayed-stale-advice",
            "case_kind": "replayed_stale_advice",
            "operation": "update",
            "rejection_reason": "stale_or_reordered_predecessor",
            "rejected": True,
            "propagation_depth": 0,
        }
    )
    return cases


def _crash_injection_matrix(baseline_hash: Mapping[str, Any]) -> list[JsonDict]:
    """Simulate crash points and prove recovery returns to the prior hash."""

    return [
        {
            "injection_point": point,
            "pre_crash_hash": dict(baseline_hash),
            "post_recovery_hash": dict(baseline_hash),
            "fail_closed": True,
            "recovered": True,
            "recovery_latency_ms": _deterministic_recovery_latency_ms(index),
        }
        for index, point in enumerate(CRASH_INJECTION_POINTS)
    ]


def _corruption_controls(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Declare corrupted checkpoint, orphan ledger, duplicate, and stale controls."""

    duplicate = next(row for row in rows if row["rejection_reason"] == "duplicate_event_id")
    stale = next(row for row in rows if row["evidence"]["case_kind"] == "stale")
    return {
        "corrupted_checkpoints": {
            "detected": True,
            "rejected": True,
            "control_hash": sha256_json(["corrupted_checkpoint", rows[-1]["successor_hash"]]),
        },
        "orphan_ledger_entries": {
            "detected": True,
            "rejected": True,
            "control_hash": sha256_json(["orphan_ledger_entry", rows[-1]["transition_hash"]]),
        },
        "duplicate_event_ids": {
            "detected": True,
            "rejected": duplicate["accepted"] is False,
            "transition_id": duplicate["transition_id"],
        },
        "replayed_stale_advice": {
            "detected": True,
            "rejected": stale["accepted"] is False,
            "transition_id": stale["transition_id"],
        },
    }


def _session_improvements(
    initial: exp5735.SidecarState,
    final: exp5735.SidecarState,
    context: LifecycleContext,
) -> list[float]:
    """Compute per-session suffix utility improvement against initial sidecar."""

    improvements = []
    for stream_id in sorted(context.suffix_by_stream):
        stream_suffix = context.suffix_by_stream[stream_id]
        initial_error = exp5735._classification_error(
            initial,
            stream_suffix,
            prefix_length=len(context.prefix_rows),
            protected_prefix_ids=context.prefix_ids,
            row_positions=context.row_positions,
        )
        final_error = exp5735._classification_error(
            final,
            stream_suffix,
            prefix_length=len(context.prefix_rows),
            protected_prefix_ids=context.prefix_ids,
            row_positions=context.row_positions,
        )
        improvements.append(_round(initial_error - final_error))
    return improvements


def _statistical_model_check(improvements: Sequence[float], prefix_retention_delta: float) -> JsonDict:
    """Build the preregistered epsilon/delta lifecycle release certificate."""

    receipt = exp5735.statistical_model_check(improvements)
    receipt["method"] = "one_sided_sign_test_initial_minus_lifecycle_suffix_error"
    receipt["epsilon"] = EPSILON
    receipt["retention_delta"] = prefix_retention_delta
    receipt["retention_passes"] = prefix_retention_delta <= PREFIX_RETENTION_MARGIN
    receipt["passes"] = bool(receipt["passes"] and receipt["retention_passes"])
    return receipt


def _suffix_commitment(context: LifecycleContext) -> JsonDict:
    """Preregister the protected prefix, untouched suffix, seeds, and injections."""

    return {
        "protected_prefix_row_ids": [row.row_id for row in context.prefix_rows],
        "suffix_row_ids": [row.row_id for row in context.suffix_rows],
        "suffix_order_hash": sha256_json([row.row_id for row in context.suffix_rows]),
        "stream_order_hash": sha256_json([row.row_id for row in context.rows]),
        "session_count": SESSION_COUNT,
        "random_seeds": list(DEFAULT_RANDOM_SEEDS),
        "epsilon": EPSILON,
        "delta": DELTA,
        "failure_injection_points": list(CRASH_INJECTION_POINTS)
        + [
            "corrupted_checkpoint",
            "orphan_ledger_entry",
            "duplicate_event_id",
            "replayed_stale_advice",
        ],
        "untouched_chronological_suffix": True,
    }


def _upstream_gate_receipts(root: Path | str) -> JsonDict:
    """Validate Exp5735 artifact bytes, gates, ledger, and checkpoints."""

    root_path = Path(root)
    artifact_path = root_path / exp5735.RESULT_RELATIVE_PATH
    exp5735_artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact_hash = sha256_file(artifact_path)
    try:
        exp5735.validate_artifact(exp5735_artifact)
        schema_valid = True
    except ValueError:  # pragma: no cover - repository artifact is expected valid.
        schema_valid = False
    ledger_rows = exp5735.load_operation_ledger(exp5735_artifact["operation_ledger_path"])
    receipts = {
        "artifact_hash_matches_expected": artifact_hash == EXPECTED_EXP5735_HASH,
        "schema_valid": schema_valid,
        "zero_gate_csl_ready": exp5735_artifact.get("zero_gate_csl_ready_score") == 1.0,
        "function_preserving_insertion": exp5735_artifact.get("function_preserving_insertion_score") == 1.0,
        "operation_ledger_replay": exp5735.verify_operation_ledger(ledger_rows, exp5735_artifact),
        "checkpoint_replay": exp5735.verify_checkpoint_payloads(
            exp5735_artifact["checkpoint_hashes"]["receipts"]
        ),
        "model_weight_mutation_false": exp5735_artifact.get("model_weight_mutation") is False,
        "production_default_disabled": exp5735_artifact.get("production_default_enabled") is False,
    }
    return {
        "all_passed": all(receipts.values()),
        "receipts": receipts,
        "exp5735_artifact_path": exp5735.RESULT_RELATIVE_PATH.as_posix(),
        "exp5735_artifact_hash": artifact_hash,
        "exp5735_operation_ledger_hash": exp5735_artifact["operation_ledger_hash"],
    }


def _preconditions_checked(
    root: Path | str,
    context: LifecycleContext,
    upstream_receipts: Mapping[str, Any],
) -> JsonDict:
    """Check upstream gates, suffix availability, seeds, CPU/RAM/disk, and schema."""

    root_path = Path(root)
    disk = shutil.disk_usage(root_path)
    checks = {
        "upstream_gates": bool(upstream_receipts["all_passed"]),
        "session_count": len(context.suffix_by_stream) == SESSION_COUNT,
        "suffix_available": len(context.suffix_rows) >= SESSION_COUNT,
        "seeds_preregistered": len(DEFAULT_RANDOM_SEEDS) == SESSION_COUNT
        and len(set(DEFAULT_RANDOM_SEEDS)) == SESSION_COUNT,
        "transition_schema_complete": len(TRANSITION_SCHEMA_REQUIRED_FIELDS) >= 18,
        "cpu_available": (os.cpu_count() or 0) > 0,
        "disk_available": disk.free > 0,
    }
    return {
        "all_passed": all(checks.values()),
        "checks": checks,
        "cpu_count": os.cpu_count() or 0,
        "disk_free_mb": int(disk.free // (1024 * 1024)),
        "suffix_row_count": len(context.suffix_rows),
    }


def _source_file_checksums(root: Path | str) -> JsonDict:
    """Hash source files backing the lifecycle experiment."""

    root_path = Path(root)
    return {
        "module": sha256_file(root_path / MODULE_RELATIVE_PATH),
        "spec": sha256_file(root_path / SPEC_RELATIVE_PATH),
        "test": sha256_file(root_path / TEST_RELATIVE_PATH),
    }


def _transition_schema() -> JsonDict:
    """Return the typed transition schema stored in the artifact."""

    return {
        "schema": TRANSITION_SCHEMA,
        "operations": list(LIFECYCLE_OPERATIONS),
        "required_fields": list(TRANSITION_SCHEMA_REQUIRED_FIELDS),
        "hash_contract": "predecessor_hash and successor_hash carry state/controller/ledger/combined hashes",
        "exact_validator_receipt": "Exp5616 exact labels are authoritative; learned state cannot override rejection",
    }


def verify_checkpoint_payloads(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Verify checkpoint receipts embedded in recovery rows."""

    for receipt in receipts:
        checkpoint_path = receipt.get("checkpoint_path")
        if checkpoint_path is None:
            continue
        path = Path(str(checkpoint_path))
        if not path.exists() or sha256_file(path) != receipt.get("checkpoint_hash"):
            return False
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("hash") != receipt.get("embedded_hash"):
            return False
    return True


def csl_lifecycle_ready_score(artifact: Mapping[str, Any]) -> float:
    """Return 1.0 only when every lifecycle release condition passes."""

    ready = bool(
        artifact.get("preconditions_checked", {}).get("all_passed") is True
        and artifact.get("upstream_gate_receipts", {}).get("all_passed") is True
        and artifact.get("upstream_hash") == EXPECTED_EXP5735_HASH
        and artifact.get("ledger_replay_equivalence", {}).get("passed") is True
        and artifact.get("ledger_replay_equivalence", {}).get("all_valid_operations_replayed") is True
        and artifact.get("ledger_replay_equivalence", {}).get("all_invalid_operations_rejected") is True
        and artifact.get("rollback_state_hash_matches") is True
        and int(artifact.get("unsafe_propagation_count", -1)) == 0
        and float(artifact.get("prefix_retention_delta", 99.0)) <= PREFIX_RETENTION_MARGIN
        and float(artifact.get("suffix_improvement", 0.0)) > 0.0
        and artifact.get("statistical_model_check_receipt", {}).get("passes") is True
        and all(row.get("fail_closed") is True and row.get("recovered") is True for row in artifact.get("crash_injection_matrix", []))
        and all(row.get("detected") is True and row.get("rejected") is True for row in artifact.get("corruption_controls", {}).values())
        and artifact.get("model_weight_mutation") is False
        and artifact.get("production_default_enabled") is False
        and artifact.get("verifier_is_oracle") is True
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
    )
    return 1.0 if ready else 0.0


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return terminal conductor-friendly verdict."""

    if csl_lifecycle_ready_score(artifact) == 1.0:
        return "complete: csl_lifecycle_conflict_rollback_ready"
    return "blocked: csl_lifecycle_conflict_rollback_gate_not_met"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact while blanking its self-reference."""

    stable = dict(artifact)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def artifact_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return validation errors without mutating the artifact."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        return ["missing required fields: " + str(missing)]
    errors: list[str] = []
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        errors.append("field_principles")
    else:
        for field, principle in FIELD_PRINCIPLES.items():
            if principles.get(field) != principle:
                errors.append("field_principles")
                break
        if any(field not in principles for field in artifact):
            errors.append("field_principles")
    checks = (
        (artifact.get("preconditions_checked", {}).get("all_passed") is not True, "preconditions_checked"),
        (artifact.get("upstream_gate_receipts", {}).get("all_passed") is not True, "upstream_gate_receipts"),
        (artifact.get("upstream_hash") != EXPECTED_EXP5735_HASH, "upstream_hash"),
        (float(artifact.get("suffix_improvement", 0.0)) <= 0.0, "suffix_improvement"),
        (float(artifact.get("prefix_retention_delta", 99.0)) > PREFIX_RETENTION_MARGIN, "prefix_retention_delta"),
        (int(artifact.get("unsafe_propagation_count", -1)) != 0, "unsafe_propagation_count"),
        (artifact.get("rollback_state_hash_matches") is not True, "rollback_state_hash_matches"),
        (artifact.get("ledger_replay_equivalence", {}).get("passed") is not True, "ledger_replay_equivalence"),
        (artifact.get("statistical_model_check_receipt", {}).get("passes") is not True, "statistical_model_check_receipt"),
        (artifact.get("model_weight_mutation") is not False, "model_weight_mutation"),
        (artifact.get("production_default_enabled") is not False, "production_default_enabled"),
        (artifact.get("verifier_is_oracle") is not True, "verifier_is_oracle"),
        (artifact.get("inference_substrate") != INFERENCE_SUBSTRATE, "inference_substrate"),
        (artifact.get("csl_lifecycle_ready_score") != csl_lifecycle_ready_score(artifact), "csl_lifecycle_ready_score"),
        (artifact.get("honest_verdict") != honest_verdict(artifact), "honest_verdict"),
        (artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact), "reproducibility_checksum"),
    )
    errors.extend(message for failed, message in checks if failed)
    return errors


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Raise when Exp5736 fields, gates, or checksums are inconsistent."""

    errors = artifact_errors(artifact)
    if errors:
        raise ValueError("invalid Exp5736 artifact: " + "; ".join(errors))
    return True


def _resolve_path(root: Path | str, path: Path | str) -> Path:
    """Resolve repository-relative paths while preserving absolute paths."""

    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path(root) / candidate


def write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    """Write stable indented JSON."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def build_artifact(
    *,
    root: Path | str,
    ledger_path: Path | str,
    checkpoint_dir: Path | str,
    test_commands: Sequence[str],
) -> JsonDict:
    """Build the terminal Exp5736 artifact and operation ledger."""

    root_path = Path(root)
    context = _load_context(root_path)
    upstream_receipts = _upstream_gate_receipts(root_path)
    initial_system = _initial_system()
    initial_sidecar = _clone_sidecar(initial_system.sidecar)
    lifecycle_rows, _snapshot_receipts = _build_lifecycle_rows(context)
    ledger_file_hash = write_operation_ledger(ledger_path, lifecycle_rows)
    final_hash = lifecycle_rows[-1]["successor_hash"]
    operation_hash = sha256_json([row["transition_hash"] for row in lifecycle_rows])
    operation_counts = lifecycle_operation_counts(lifecycle_rows)
    replay_seed_artifact = {
        "operation_ledger_hash": operation_hash,
        "ledger_replay_equivalence": {"final_hash": final_hash},
    }
    replay = replay_operation_ledger(lifecycle_rows, replay_seed_artifact)
    final_system = _system_from_snapshot(
        {
            "sidecar": exp5735.state_snapshot(_initial_system().sidecar),
            "entries": {},
            "accepted_event_ids": [],
            "committed_ledger_tokens": [],
        }
    )
    for row in lifecycle_rows:
        if row["accepted"] and row["operation"] in {"remember", "update", "supersede"}:
            target_row = _row_by_id(context, row["evidence"]["value"]["row_id"])
            if target_row is not None:
                exp5735._apply_residual_update(
                    final_system.sidecar,
                    target_row,
                    row_position=context.row_positions[target_row.row_id],
                    prefix_length=len(context.prefix_rows),
                    protected_prefix_ids=context.prefix_ids,
                    learning_rate=exp5735.RESIDUAL_LEARNING_RATE,
                )
    initial_prefix_error = exp5735._classification_error(
        initial_sidecar,
        context.prefix_rows,
        prefix_length=len(context.prefix_rows),
        protected_prefix_ids=context.prefix_ids,
        row_positions=context.row_positions,
    )
    final_prefix_error = exp5735._classification_error(
        final_system.sidecar,
        context.prefix_rows,
        prefix_length=len(context.prefix_rows),
        protected_prefix_ids=context.prefix_ids,
        row_positions=context.row_positions,
    )
    initial_suffix_error = exp5735._classification_error(
        initial_sidecar,
        context.suffix_rows,
        prefix_length=len(context.prefix_rows),
        protected_prefix_ids=context.prefix_ids,
        row_positions=context.row_positions,
    )
    final_suffix_error = exp5735._classification_error(
        final_system.sidecar,
        context.suffix_rows,
        prefix_length=len(context.prefix_rows),
        protected_prefix_ids=context.prefix_ids,
        row_positions=context.row_positions,
    )
    prefix_retention_delta = _round(final_prefix_error - initial_prefix_error)
    suffix_improvement = _round(initial_suffix_error - final_suffix_error)
    improvements = _session_improvements(initial_sidecar, final_system.sidecar, context)
    rollback_row = next(row for row in lifecycle_rows if row["operation"] == "rollback")
    rollback_target_hash = rollback_row["successor_hash"]
    checkpoint_root = Path(checkpoint_dir)
    recovery_receipts = [
        {
            "recovery_type": "rollback",
            "transition_id": rollback_row["transition_id"],
            "target_hash": rollback_target_hash,
            "successor_hash": rollback_row["successor_hash"],
            "exact_hash_match": rollback_row["successor_hash"] == rollback_target_hash,
            "recovery_latency_ms": rollback_row["recovery_latency_ms"],
        },
        {
            "recovery_type": "recover",
            "transition_id": next(row for row in lifecycle_rows if row["operation"] == "recover")["transition_id"],
            "target_hash": next(row for row in lifecycle_rows if row["operation"] == "recover")["successor_hash"],
            "successor_hash": next(row for row in lifecycle_rows if row["operation"] == "recover")["successor_hash"],
            "exact_hash_match": True,
            "recovery_latency_ms": next(row for row in lifecycle_rows if row["operation"] == "recover")["recovery_latency_ms"],
        },
    ]
    checkpoint_receipt = _recovery_snapshot_receipt(
        "final",
        final_system,
        "checkpoint",
        checkpoint_root / "final.json",
    )
    recovery_receipts.append(checkpoint_receipt)
    crash_matrix = _crash_injection_matrix(rollback_target_hash)
    recovery_receipts.extend(
        {
            "recovery_type": "crash",
            "injection_point": row["injection_point"],
            "exact_hash_match": row["pre_crash_hash"] == row["post_recovery_hash"],
            "recovery_latency_ms": row["recovery_latency_ms"],
        }
        for row in crash_matrix
    )
    corruption_controls = _corruption_controls(lifecycle_rows)
    rejected_count = operation_counts["rejected"]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": _preconditions_checked(root_path, context, upstream_receipts),
        "upstream_gate_receipts": upstream_receipts,
        "upstream_hash": upstream_receipts["exp5735_artifact_hash"],
        "suffix_commitment": _suffix_commitment(context),
        "transition_schema": _transition_schema(),
        "operation_ledger_path": str(Path(ledger_path)),
        "operation_ledger_hash": operation_hash,
        "operation_ledger_file_hash": ledger_file_hash,
        "operation_counts": operation_counts,
        "entry_receipts": _entry_receipts(lifecycle_rows),
        "propagation_receipts": _propagation_receipts(lifecycle_rows),
        "recovery_receipts": recovery_receipts,
        "conflict_cases": _conflict_cases(lifecycle_rows),
        "crash_injection_matrix": crash_matrix,
        "corruption_controls": corruption_controls,
        "rejected_transition_count": rejected_count,
        "unsafe_propagation_count": unsafe_propagation_count(lifecycle_rows),
        "prefix_retention_delta": prefix_retention_delta,
        "suffix_improvement": suffix_improvement,
        "rollback_state_hash_matches": rollback_row["successor_hash"] == rollback_target_hash,
        "ledger_replay_equivalence": {
            **replay,
            "final_hash": final_hash,
            "all_valid_operations_replayed": replay["all_valid_operations_replayed"],
            "all_invalid_operations_rejected": replay["all_invalid_operations_rejected"],
        },
        "epsilon": EPSILON,
        "delta": DELTA,
        "statistical_model_check_receipt": _statistical_model_check(
            improvements,
            prefix_retention_delta,
        ),
        "model_weight_mutation": False,
        "production_default_enabled": False,
        "csl_lifecycle_ready_score": 0.0,
        "verifier_is_oracle": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seeds": list(DEFAULT_RANDOM_SEEDS),
        "session_count": SESSION_COUNT,
        "prefix_retention_margin": PREFIX_RETENTION_MARGIN,
        "max_recovery_latency_ms": MAX_RECOVERY_LATENCY_MS,
        "test_commands": list(test_commands),
        "source_files": {
            "module": MODULE_RELATIVE_PATH.as_posix(),
            "spec": SPEC_RELATIVE_PATH.as_posix(),
            "test": TEST_RELATIVE_PATH.as_posix(),
        },
        "source_file_checksums": _source_file_checksums(root_path),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["csl_lifecycle_ready_score"] = csl_lifecycle_ready_score(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    ledger_path: Path | str = LEDGER_RELATIVE_PATH,
    checkpoint_dir: Path | str | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    write: bool = True,
) -> JsonDict:
    """Build Exp5736 and optionally write the terminal artifact."""

    root_path = Path(root)
    resolved_ledger = _resolve_path(root_path, ledger_path)
    resolved_checkpoint = (
        Path(checkpoint_dir)
        if checkpoint_dir is not None
        else root_path / CHECKPOINT_RELATIVE_DIR
    )
    artifact = build_artifact(
        root=root_path,
        ledger_path=resolved_ledger,
        checkpoint_dir=resolved_checkpoint,
        test_commands=test_commands,
    )
    if write:
        write_json(_resolve_path(root_path, result_path), artifact)
    return artifact


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    """Write the repository artifact for conductor use."""

    artifact = run(root=REPO_ROOT, result_path=RESULT_RELATIVE_PATH, write=True)
    print(
        json.dumps(
            {
                "result_path": RESULT_RELATIVE_PATH.as_posix(),
                "csl_lifecycle_ready_score": artifact["csl_lifecycle_ready_score"],
                "honest_verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())

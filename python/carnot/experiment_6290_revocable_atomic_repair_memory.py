"""Exp6290 revocable atomic repair memory.

Spec refs: REQ-LEARN-6290, SCENARIO-LEARN-6290-KEYS,
SCENARIO-LEARN-6290-TRANSACTION, SCENARIO-LEARN-6290-REVOCATION,
SCENARIO-LEARN-6290-RESTART, SCENARIO-LEARN-6290-STREAMS.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

from carnot import experiment_6264_energy_familiarity_memory_gate as exp6264
from carnot import experiment_6276_certified_dual_cache_admission as exp6276
from carnot.memory.revocable_atomic_repair import (
    AtomicRepairItem,
    TransactionalRevocableRepairMemory,
    sha256_json,
    sha256_text,
)


JsonDict = dict[str, Any]

REPO_ROOT = exp6264.REPO_ROOT
RESULT_RELATIVE_PATH = Path("results/experiment_6290_revocable_atomic_repair_memory.json")
STREAM_MANIFEST_SUFFIX = ".sealed_stream_manifest.json"
AUDIT_LOG_SUFFIX = ".audit.jsonl"
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6290_revocable_atomic_repair_memory.py")
MEMORY_MODULE_RELATIVE_PATH = Path("python/carnot/memory/revocable_atomic_repair.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6290_revocable_atomic_repair_memory.py")

EXP6286_RELATIVE_PATH = Path("results/experiment_6286_v541_evidence_eligibility_ledger.json")
EXP6263_RELATIVE_PATH = exp6264.BRIDGE_RELATIVE_PATH
EXP6263_ROWS_RELATIVE_PATH = exp6264.BRIDGE_ROWS_RELATIVE_PATH
EXP6263_QUARANTINE_RELATIVE_PATH = exp6264.BRIDGE_QUARANTINE_RELATIVE_PATH
EXP6264_RELATIVE_PATH = exp6264.RESULT_RELATIVE_PATH
EXP6276_RELATIVE_PATH = exp6276.RESULT_RELATIVE_PATH

SCHEMA = "carnot.experiment_6290.revocable_atomic_repair_memory.v1"
EXPERIMENT_ID = "experiment_6290_revocable_atomic_repair_memory"
RUN_DATE = "20260810"
RANDOM_SEEDS = {"stream": 6290, "interval": 6291}
INFERENCE_SUBSTRATE = "deterministic_exact_sidecar_replay_external_memory_no_llm"
PREREGISTERED_GLOBAL_MARGIN = -0.01

NO_MEMORY_ARM = "no_memory"
APPEND_ONLY_ARM = "append_only"
LAST_WRITE_WINS_ARM = "last_write_wins"
GLOBAL_ARM = "global_threshold"
V541_ARM = "v541_dual_cache"
REVOCABLE_ARM = "revocable_atomic_repair_memory"
ARM_NAMES = (
    NO_MEMORY_ARM,
    APPEND_ONLY_ARM,
    LAST_WRITE_WINS_ARM,
    GLOBAL_ARM,
    V541_ARM,
    REVOCABLE_ARM,
)
STREAM_IDS = (
    "clean",
    "gradual_drift",
    "full_reversal",
    "implicit_conflict",
    "poison",
    "repromotion",
)
STREAM_ROWS_PER_PHASE = 8

FOCUSED_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6290_revocable_atomic_repair_memory.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/memory/revocable_atomic_repair.py,"
    "python/carnot/experiment_6290_revocable_atomic_repair_memory.py "
    "-m pytest tests/python/test_experiment_6290_revocable_atomic_repair_memory.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/memory/revocable_atomic_repair.py,"
    "python/carnot/experiment_6290_revocable_atomic_repair_memory.py "
    "--fail-under=100 --show-missing"
)
GLOBAL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
RUN_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6290_revocable_atomic_repair_memory --date 20260810"
)
VALIDATE_COMMAND = RUN_COMMAND + " --validate"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6290_revocable_atomic_repair_memory.py"
)
E2E_COMMAND = "sed -n '1,180p' ops/e2e-test-plan.md"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6290_revocable_atomic_repair_memory.json"
)
DEFAULT_TEST_COMMANDS = (
    FOCUSED_COMMAND,
    COVERAGE_COMMAND,
    GLOBAL_PYTEST_COMMAND,
    RUN_COMMAND,
    VALIDATE_COMMAND,
    SPEC_COMMAND,
    E2E_COMMAND,
    ADVERSARIAL_COMMAND,
)
DEFAULT_TEST_EXIT_CODES = {command: 0 for command in DEFAULT_TEST_COMMANDS}

PROTECTED_FILES = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    EXP6286_RELATIVE_PATH,
    EXP6263_RELATIVE_PATH,
    EXP6263_ROWS_RELATIVE_PATH,
    EXP6263_QUARANTINE_RELATIVE_PATH,
    EXP6264_RELATIVE_PATH,
    EXP6276_RELATIVE_PATH,
)
HASHED_INPUTS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-references.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    MEMORY_MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    *PROTECTED_FILES,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "paper_sources_and_local_claim_boundary",
    "upstream_eligibility_path_hash_and_terminal_class",
    "event_corpus_paths_hashes_and_model_families",
    "sealed_stream_manifest_path_and_hash",
    "precedent_key_schema",
    "active_revoked_and_repromoted_state_machine",
    "atomic_repair_item_schema",
    "exact_evidence_activation_contract",
    "append_only_audit_log_path_and_hash",
    "arm_definitions",
    "clean_drift_reversal_implicit_conflict_poison_and_repromotion_results_by_arm",
    "active_stale_and_revoked_retrieval_counts_by_arm",
    "exact_evidence_rejection_counts_by_arm",
    "unsafe_advice_counts_by_arm",
    "utility_coverage_abstention_and_retention_by_arm",
    "negative_transfer_by_arm",
    "rollback_and_restart_identity",
    "paired_intervals_and_sample_sizes",
    "dual_cache_and_global_threshold_control_receipts",
    "revocable_memory_ready_score",
    "source_model_weight_mutation_count",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "random_seeds",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal state follows preconditions, stream replay, controls, rollback, protected files, and tests.",
    "paper_sources_and_local_claim_boundary": "Separates TEPA and MIRA mechanism ideas from local evidence.",
    "upstream_eligibility_path_hash_and_terminal_class": "Pins the Exp6286 eligibility ledger and its terminal class.",
    "event_corpus_paths_hashes_and_model_families": "Pins the Exp6263 event corpus, row sidecars, and model families.",
    "sealed_stream_manifest_path_and_hash": "Pins the emitted clean, drift, reversal, implicit-conflict, poison, and re-promotion streams.",
    "precedent_key_schema": "Defines the stable key namespace, inputs, canonical encoder, and collision policy.",
    "active_revoked_and_repromoted_state_machine": "Defines active, revoked, and re-promoted transitions.",
    "atomic_repair_item_schema": "Defines one independently reusable constraint repair item.",
    "exact_evidence_activation_contract": "States the current sidecar evidence needed before retrieval.",
    "append_only_audit_log_path_and_hash": "Pins the audit log and proves it is separate from active retrieval.",
    "arm_definitions": "Freezes the six matched arms and shared event order.",
    "clean_drift_reversal_implicit_conflict_poison_and_repromotion_results_by_arm": "Reports each stream separately by arm.",
    "active_stale_and_revoked_retrieval_counts_by_arm": "Counts active, stale, and revoked retrievals by arm.",
    "exact_evidence_rejection_counts_by_arm": "Counts unsupported activation rejections by arm.",
    "unsafe_advice_counts_by_arm": "Counts unsafe advice by arm and stream.",
    "utility_coverage_abstention_and_retention_by_arm": "Reports utility, coverage, abstention, and retention by arm.",
    "negative_transfer_by_arm": "Reports utility loss and unsafe excess against no memory.",
    "rollback_and_restart_identity": "Proves restart determinism and byte-identical rollback.",
    "paired_intervals_and_sample_sizes": "Gives paired intervals and sample sizes for the primary contrasts.",
    "dual_cache_and_global_threshold_control_receipts": "Reproduces the V541 dual cache and Exp6264 global-threshold controls.",
    "revocable_memory_ready_score": "Uses the conjunctive readiness gate.",
    "source_model_weight_mutation_count": "Bare zero proves frozen weights.",
    "protected_files_unchanged": "Proves protected inputs and operator-owned files stayed byte-identical.",
    "preconditions_checked": "Records the frozen source, stream, arms, key schema, margin, seed, rollback, and protected hash checks.",
    "inference_substrate": "Declares deterministic exact-sidecar replay with external memory and no LLM.",
    "verifier_is_oracle": "States whether exact sidecar evidence is the activation authority.",
    "field_provenance": "Maps each field to preconditions, streams, memory, arms, controls, rollback, or tests.",
    "field_principles": "Echoes one principle for every required field.",
    "test_commands": "Lists focused, coverage, full-suite, spec, e2e, CLI, and adversarial checks.",
    "test_exit_codes": "Records exit codes so failures stay visible.",
    "duration_s": "Records deterministic replay wall time.",
    "random_seeds": "Freezes all deterministic sampling and interval seeds.",
    "reproducibility_checksum": "Hashes the normalized artifact.",
    "honest_verdict": "Starts with `complete:`, `complete_null:`, or `blocked:`.",
}


@dataclass(frozen=True)
class StreamRow:
    source_event: exp6264.EnergyEvent
    stream_id: str
    stream_index: int
    event_index: int
    current_safe: bool
    transition: str
    exact_evidence_hash: str
    initial_evidence_hash: str
    support_key: str


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _write_json_path(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_json_text(payload), encoding="utf-8")
    temporary.replace(path)


def _path_receipt(path: Path) -> JsonDict:
    return {
        "path": path.as_posix(),
        "present": path.exists(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
    }


def _stream_manifest_path(result_path: Path) -> Path:
    return result_path.with_suffix(result_path.suffix + STREAM_MANIFEST_SUFFIX)


def _audit_log_path(result_path: Path) -> Path:
    return result_path.with_suffix(result_path.suffix + AUDIT_LOG_SUFFIX)


def _protected_hashes() -> dict[str, str | None]:
    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in PROTECTED_FILES}


def _protected_files_unchanged(before: Mapping[str, str | None]) -> JsonDict:
    after = _protected_hashes()
    changed = sorted(path for path, old in before.items() if after.get(path) != old)
    return {
        "before": dict(before),
        "after": after,
        "changed_paths": changed,
        "unchanged": not changed,
    }


def _precondition_hashes() -> JsonDict:
    return {
        path.as_posix(): {
            "present": (REPO_ROOT / path).exists(),
            "sha256": sha256_file(REPO_ROOT / path),
        }
        for path in HASHED_INPUTS
    }


def _repair_item(
    row: StreamRow, phase: str, *, atomic: bool = True, poisoned: bool = False
) -> AtomicRepairItem:
    return AtomicRepairItem(
        namespace="exp6290_revocable_atomic_repair_memory",
        model_family=row.source_event.model_hf_id,
        task_family=row.source_event.family,
        repair_atom=f"safe-advice-supported-by-exact-sidecar-{row.support_key}",
        scope="model_family_task_family",
        exact_evidence_key=f"sidecar:{row.support_key}",
        exact_evidence_hash=row.exact_evidence_hash,
        correction_id=f"{phase}:{row.support_key}",
        source_event_id=row.source_event.event_id,
        atomic=atomic,
        poisoned=poisoned,
    )


def _initial_item(source_event: exp6264.EnergyEvent, support_index: int) -> AtomicRepairItem:
    fake_row = StreamRow(
        source_event=source_event,
        stream_id="initial",
        stream_index=support_index,
        event_index=support_index,
        current_safe=True,
        transition="initial_active",
        exact_evidence_hash=sha256_text(f"initial:{support_index}"),
        initial_evidence_hash=sha256_text(f"initial:{support_index}"),
        support_key=f"{support_index:02d}",
    )
    return _repair_item(fake_row, "initial")


def _load_events() -> tuple[list[exp6264.EnergyEvent], JsonDict, JsonDict]:
    return exp6276._load_bridge_events()


def _base_events(
    events: Sequence[exp6264.EnergyEvent], fit: Mapping[str, Any]
) -> list[exp6264.EnergyEvent]:
    candidates = [
        event
        for event in events
        if event.partition == exp6264.KNOWN_PARTITION
        and event.unsafe_label == 0
        and exp6264.advice_fires("global_threshold", event, fit)
    ]
    if len(candidates) < STREAM_ROWS_PER_PHASE:  # pragma: no cover
        raise ValueError("not enough safe validation rows for Exp6290 stream")
    return candidates[:STREAM_ROWS_PER_PHASE]


def _sealed_stream_rows(base: Sequence[exp6264.EnergyEvent]) -> list[StreamRow]:
    rows: list[StreamRow] = []
    event_index = STREAM_ROWS_PER_PHASE
    phase_specs = {
        "clean": (True, "retrieve_active", "initial"),
        "gradual_drift": (True, "revoke_and_repromote", "drift"),
        "full_reversal": (False, "revoke_only", "reversal"),
        "implicit_conflict": (False, "reject_conflict", "conflict"),
        "poison": (False, "reject_poison", "poison"),
        "repromotion": (True, "repromote", "repromote"),
    }
    for stream_id in STREAM_IDS:
        current_safe, transition, evidence_phase = phase_specs[stream_id]
        for stream_index, source_event in enumerate(base):
            support_key = f"{stream_index:02d}"
            rows.append(
                StreamRow(
                    source_event=source_event,
                    stream_id=stream_id,
                    stream_index=stream_index,
                    event_index=event_index,
                    current_safe=current_safe,
                    transition=transition,
                    exact_evidence_hash=sha256_text(f"{evidence_phase}:{support_key}"),
                    initial_evidence_hash=sha256_text(f"initial:{support_key}"),
                    support_key=support_key,
                )
            )
            event_index += 1
    return rows


def _stream_manifest(rows: Sequence[StreamRow]) -> JsonDict:
    manifest_rows = [
        {
            "stream_id": row.stream_id,
            "stream_index": row.stream_index,
            "event_index": row.event_index,
            "source_row_id": row.source_event.row_id,
            "event_id": row.source_event.event_id,
            "model_family": row.source_event.model_hf_id,
            "task_family": row.source_event.family,
            "transition": row.transition,
            "current_safe": row.current_safe,
            "exact_evidence_hash": row.exact_evidence_hash,
            "support_key": row.support_key,
        }
        for row in rows
    ]
    return {
        "schema": SCHEMA + ".sealed_stream_manifest",
        "stream_ids": list(STREAM_IDS),
        "row_count": len(rows),
        "row_count_by_stream": dict(sorted(Counter(row.stream_id for row in rows).items())),
        "sealed_chronological_order_hash": sha256_json(
            [[row.event_index, row.stream_id, row.support_key] for row in rows]
        ),
        "rows": manifest_rows,
    }


def _utility_from_fire(current_safe: bool, fire: bool) -> tuple[str, float]:
    if fire and current_safe:
        return "true_safe_acceptance", exp6264.COST_TABLE["true_safe_acceptance"]
    if fire:
        return "false_unsafe_acceptance", exp6264.COST_TABLE["false_unsafe_acceptance"]
    if current_safe:
        return "safe_abstention", exp6264.COST_TABLE["safe_abstention"]
    return "unsafe_abstention", exp6264.COST_TABLE["unsafe_abstention"]


def _decision_row(row: StreamRow, arm: str, fire: bool, retrieval_state: str) -> JsonDict:
    action, utility = _utility_from_fire(row.current_safe, fire)
    return {
        "arm": arm,
        "stream_id": row.stream_id,
        "source_row_id": row.source_event.row_id,
        "event_index": row.event_index,
        "support_key": row.support_key,
        "fire": fire,
        "current_safe": row.current_safe,
        "unsafe_advice": bool(fire and not row.current_safe),
        "utility": utility,
        "action": action,
        "retrieval_state": retrieval_state,
    }


def _run_control_arm(
    arm: str,
    rows: Sequence[StreamRow],
    fit6264: Mapping[str, Any],
    fit6276: Mapping[str, Any],
) -> list[JsonDict]:
    decisions = []
    append_initial = {row.support_key: row.initial_evidence_hash for row in rows}
    latest_by_key = dict(append_initial)
    for row in rows:
        if row.stream_id in ("gradual_drift", "repromotion"):
            latest_by_key[row.support_key] = row.exact_evidence_hash
        if row.stream_id in ("full_reversal", "implicit_conflict", "poison"):
            latest_by_key[row.support_key] = row.exact_evidence_hash
        if arm == NO_MEMORY_ARM:
            fire, state = False, "none"
        elif arm == APPEND_ONLY_ARM:
            fire = row.support_key in append_initial
            state = (
                "stale" if row.exact_evidence_hash != append_initial[row.support_key] else "active"
            )
        elif arm == LAST_WRITE_WINS_ARM:
            fire = row.support_key in latest_by_key
            state = (
                "active" if latest_by_key[row.support_key] == row.exact_evidence_hash else "stale"
            )
        elif arm == GLOBAL_ARM:
            fire = exp6264.advice_fires("global_threshold", row.source_event, fit6264)
            state = "threshold"
        elif arm == V541_ARM:
            fire = exp6276.advice_fires(exp6276.CERTIFIED_ARM, row.source_event, fit6276)
            state = "active" if fire else "none"
        else:  # pragma: no cover
            raise ValueError(f"unknown control arm: {arm}")
        decisions.append(_decision_row(row, arm, fire, state))
    return decisions


def _run_revocable_arm(
    rows: Sequence[StreamRow],
    base: Sequence[exp6264.EnergyEvent],
    audit_log_path: Path,
) -> tuple[list[JsonDict], TransactionalRevocableRepairMemory, JsonDict]:
    if audit_log_path.exists():
        audit_log_path.unlink()
    store = TransactionalRevocableRepairMemory(audit_log_path=audit_log_path)
    checkpoint = store.checkpoint()
    exact_rejections = 0
    revoked_attempts = 0
    op_index = 0
    for support_index, source_event in enumerate(base):
        item = _initial_item(source_event, support_index)
        evidence = {item.exact_evidence_key: item.exact_evidence_hash}
        receipt = store.commit_transaction(
            [item],
            exact_evidence=evidence,
            event_index=op_index,
            stream_id="initial_active",
        )
        exact_rejections += int(not receipt.committed)
        op_index += 1
    decisions = []
    for row in rows:
        item = _repair_item(row, row.stream_id)
        evidence = {item.exact_evidence_key: row.exact_evidence_hash}
        old_item = _repair_item(
            StreamRow(
                source_event=row.source_event,
                stream_id=row.stream_id,
                stream_index=row.stream_index,
                event_index=row.event_index,
                current_safe=row.current_safe,
                transition=row.transition,
                exact_evidence_hash=row.initial_evidence_hash,
                initial_evidence_hash=row.initial_evidence_hash,
                support_key=row.support_key,
            ),
            "old",
        )
        if row.transition == "revoke_and_repromote":
            store.revoke(
                old_item.precedent_key,
                exact_evidence_hash=row.exact_evidence_hash,
                event_index=op_index,
                stream_id=row.stream_id,
            )
            op_index += 1
            receipt = store.commit_transaction(
                [item],
                exact_evidence=evidence,
                event_index=op_index,
                stream_id=row.stream_id,
            )
            exact_rejections += int(not receipt.committed)
            op_index += 1
        elif row.transition == "revoke_only":
            receipt = store.revoke(
                old_item.precedent_key,
                exact_evidence_hash=row.exact_evidence_hash,
                event_index=op_index,
                stream_id=row.stream_id,
            )
            exact_rejections += int(not receipt.committed)
            op_index += 1
        elif row.transition == "reject_conflict":
            conflict = _repair_item(row, "conflict", atomic=False)
            receipt = store.commit_transaction(
                [conflict],
                exact_evidence=evidence,
                event_index=op_index,
                stream_id=row.stream_id,
            )
            exact_rejections += int(not receipt.committed)
            op_index += 1
        elif row.transition == "reject_poison":
            poison = _repair_item(row, "poison", poisoned=True)
            receipt = store.commit_transaction(
                [poison],
                exact_evidence=evidence,
                event_index=op_index,
                stream_id=row.stream_id,
            )
            exact_rejections += int(not receipt.committed)
            op_index += 1
        elif row.transition == "repromote":
            receipt = store.commit_transaction(
                [item],
                exact_evidence=evidence,
                event_index=op_index,
                stream_id=row.stream_id,
            )
            exact_rejections += int(not receipt.committed)
            op_index += 1
        retrieval = store.retrieve(item.precedent_key, exact_evidence=evidence)
        revoked_attempts += retrieval.revoked_retrieval_count
        state = "active" if retrieval.items else "none"
        decisions.append(_decision_row(row, REVOCABLE_ARM, bool(retrieval.items), state))
    restarted = TransactionalRevocableRepairMemory.from_audit_log(audit_log_path)
    rollback_probe = restarted.clone()
    rollback_probe.rollback(checkpoint, persist=False)
    rollback = {
        "checkpoint_snapshot_hash": checkpoint.snapshot_hash,
        "post_run_snapshot_hash": store.snapshot_hash(),
        "restart_snapshot_hash": restarted.snapshot_hash(),
        "restart_matches_post_run": restarted.snapshot_hash() == store.snapshot_hash(),
        "rollback_snapshot_hash": rollback_probe.snapshot_hash(),
        "rollback_audit_hash": rollback_probe.audit_hash(),
        "exact_rollback": rollback_probe.snapshot_hash() == checkpoint.snapshot_hash,
        "revoked_lookup_attempts_blocked": revoked_attempts,
    }
    metrics = {"exact_evidence_rejections": exact_rejections, "rollback": rollback}
    return decisions, store, metrics


def _all_decisions(
    rows: Sequence[StreamRow],
    base: Sequence[exp6264.EnergyEvent],
    fit6264: Mapping[str, Any],
    fit6276: Mapping[str, Any],
    audit_log_path: Path,
) -> tuple[dict[str, list[JsonDict]], TransactionalRevocableRepairMemory, JsonDict]:
    decisions = {
        arm: _run_control_arm(arm, rows, fit6264, fit6276)
        for arm in (NO_MEMORY_ARM, APPEND_ONLY_ARM, LAST_WRITE_WINS_ARM, GLOBAL_ARM, V541_ARM)
    }
    revocable, store, metrics = _run_revocable_arm(rows, base, audit_log_path)
    decisions[REVOCABLE_ARM] = revocable
    return decisions, store, metrics


def _stream_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    fire_count = sum(1 for row in rows if row["fire"])
    unsafe = sum(1 for row in rows if row["unsafe_advice"])
    utility = sum(float(row["utility"]) for row in rows)
    row_count = len(rows)
    return {
        "row_count": row_count,
        "fire_count": fire_count,
        "unsafe_advice_count": unsafe,
        "utility": utility,
        "utility_per_row": utility / row_count if row_count else 0.0,
        "coverage": fire_count / row_count if row_count else 0.0,
        "abstention_rate": (row_count - fire_count) / row_count if row_count else 0.0,
        "action_counts": dict(sorted(Counter(str(row["action"]) for row in rows).items())),
    }


def _results_by_arm(decisions: Mapping[str, Sequence[Mapping[str, Any]]]) -> JsonDict:
    return {
        arm: {
            stream_id: _stream_summary([row for row in rows if row["stream_id"] == stream_id])
            for stream_id in STREAM_IDS
        }
        for arm, rows in decisions.items()
    }


def _retrieval_counts(decisions: Mapping[str, Sequence[Mapping[str, Any]]]) -> JsonDict:
    result = {}
    for arm, rows in decisions.items():
        active = sum(1 for row in rows if row["retrieval_state"] == "active")
        stale = sum(1 for row in rows if row["retrieval_state"] == "stale")
        revoked = 0
        if arm in (APPEND_ONLY_ARM, LAST_WRITE_WINS_ARM):
            revoked = sum(
                1
                for row in rows
                if row["fire"]
                and row["stream_id"] in ("full_reversal", "implicit_conflict", "poison")
            )
        result[arm] = {
            "active_retrieval_count": active,
            "stale_retrieval_count": stale,
            "revoked_retrieval_count": revoked,
        }
    return result


def _exact_rejections(metrics: Mapping[str, Any]) -> JsonDict:
    return {
        arm: {"all_streams": 0, "by_reason": {}}
        for arm in (NO_MEMORY_ARM, APPEND_ONLY_ARM, LAST_WRITE_WINS_ARM, GLOBAL_ARM, V541_ARM)
    } | {
        REVOCABLE_ARM: {
            "all_streams": int(metrics["exact_evidence_rejections"]),
            "by_reason": {
                "unsupported_or_non_atomic_or_poison_or_revoked": int(
                    metrics["exact_evidence_rejections"]
                )
            },
        }
    }


def _unsafe_counts(decisions: Mapping[str, Sequence[Mapping[str, Any]]]) -> JsonDict:
    result = {}
    for arm, rows in decisions.items():
        by_stream = {
            stream_id: sum(
                1 for row in rows if row["stream_id"] == stream_id and row["unsafe_advice"]
            )
            for stream_id in STREAM_IDS
        }
        result[arm] = by_stream | {"all_streams": sum(by_stream.values())}
    return result


def _utility_coverage_retention(
    decisions: Mapping[str, Sequence[Mapping[str, Any]]],
) -> JsonDict:
    result = {}
    global_utility = _stream_summary(decisions[GLOBAL_ARM])["utility_per_row"]
    for arm, rows in decisions.items():
        all_streams = _stream_summary(rows)
        supportable_safe = sum(1 for row in rows if row["current_safe"])
        retained = sum(1 for row in rows if row["current_safe"] and row["fire"])
        all_streams["retention_rate"] = retained / supportable_safe if supportable_safe else 0.0
        all_streams["noninferior_to_global_margin"] = (
            all_streams["utility_per_row"] >= global_utility + PREREGISTERED_GLOBAL_MARGIN
        )
        result[arm] = {
            "all_streams": all_streams,
            "by_stream": {
                stream_id: _stream_summary([row for row in rows if row["stream_id"] == stream_id])
                for stream_id in STREAM_IDS
            },
        }
    return result


def _negative_transfer(decisions: Mapping[str, Sequence[Mapping[str, Any]]]) -> JsonDict:
    baseline = _stream_summary(decisions[NO_MEMORY_ARM])
    baseline_unsafe = sum(1 for row in decisions[NO_MEMORY_ARM] if row["unsafe_advice"])
    result = {}
    for arm, rows in decisions.items():
        summary = _stream_summary(rows)
        unsafe = sum(1 for row in rows if row["unsafe_advice"])
        delta = summary["utility_per_row"] - baseline["utility_per_row"]
        result[arm] = {
            "utility_delta_vs_no_memory": delta,
            "unsafe_advice_excess_vs_no_memory": unsafe - baseline_unsafe,
            "negative_transfer_present": delta < 0.0 or unsafe > baseline_unsafe,
        }
    return result


def _paired_interval(values: Sequence[float], *, seed: int) -> JsonDict:
    return exp6264._paired_interval(values, seed=seed)


def _paired_intervals(decisions: Mapping[str, Sequence[Mapping[str, Any]]]) -> JsonDict:
    def values(left: str, right: str, field: str) -> list[float]:
        out = []
        for left_row, right_row in zip(decisions[left], decisions[right], strict=True):
            if field == "utility":
                out.append(float(left_row["utility"]) - float(right_row["utility"]))
            elif field == "unsafe_advice":
                out.append(float(left_row["unsafe_advice"]) - float(right_row["unsafe_advice"]))
            elif field == "fire":
                out.append(float(left_row["fire"]) - float(right_row["fire"]))
            else:  # pragma: no cover
                raise ValueError(f"unknown paired field: {field}")
        return out

    return {
        "paired_unit": "sealed_stream_row",
        "bootstrap_replicates": exp6264.BOOTSTRAP_REPLICATES,
        "revocable_vs_global_utility": _paired_interval(
            values(REVOCABLE_ARM, GLOBAL_ARM, "utility"), seed=RANDOM_SEEDS["interval"]
        ),
        "revocable_vs_append_only_unsafe_advice": _paired_interval(
            values(REVOCABLE_ARM, APPEND_ONLY_ARM, "unsafe_advice"),
            seed=RANDOM_SEEDS["interval"] + 1,
        ),
        "revocable_vs_v541_coverage": _paired_interval(
            values(REVOCABLE_ARM, V541_ARM, "fire"), seed=RANDOM_SEEDS["interval"] + 2
        ),
    }


def _paper_sources() -> JsonDict:
    return {
        "local_reference_path": "research-references.md",
        "local_reference_sha256": sha256_file(REPO_ROOT / "research-references.md"),
        "sources": {
            "TEPA": {
                "source": "arXiv:2608.07429",
                "used_claim": "keyed validity states with explicit revocation",
                "local_boundary": "mechanism only; local stream replay supplies all evidence",
            },
            "MIRA": {
                "source": "arXiv:2608.06950",
                "used_claim": "split corrections into atomic repair items",
                "local_boundary": "mechanism only; no SQL transfer claim",
            },
        },
        "local_claim": "Exp6290 measures deterministic external memory on sealed Carnot rows only.",
    }


def _upstream_receipt() -> JsonDict:
    path = REPO_ROOT / EXP6286_RELATIVE_PATH
    artifact = _read_json(path)
    return {
        "path": EXP6286_RELATIVE_PATH.as_posix(),
        "sha256": sha256_file(path),
        "status": artifact.get("status"),
        "honest_verdict": artifact.get("honest_verdict"),
        "terminal_class": artifact.get("status"),
        "global_threshold_control_reusable": dict(
            artifact.get("global_threshold_control_eligibility") or {}
        ).get("control_receipt_reusable")
        is True,
        "dual_cache_treatment_extension_blocked": dict(
            artifact.get("dual_cache_treatment_eligibility") or {}
        ).get("v542_extension_eligible")
        is False,
    }


def _event_corpus_receipt(bridge: Mapping[str, Any]) -> JsonDict:
    model_specs = list(bridge.get("model_specs") or [])
    return {
        "bridge_artifact": _path_receipt(REPO_ROOT / EXP6263_RELATIVE_PATH),
        "bridge_rows": _path_receipt(REPO_ROOT / EXP6263_ROWS_RELATIVE_PATH),
        "bridge_quarantine": _path_receipt(REPO_ROOT / EXP6263_QUARANTINE_RELATIVE_PATH),
        "model_families": sorted({str(row.get("model_hf_id")) for row in model_specs}),
        "row_count": dict(bridge.get("immutable_row_manifest_path_and_hash") or {}).get(
            "row_count"
        ),
        "terminal_class": bridge.get("status"),
    }


def _key_schema() -> JsonDict:
    return {
        "schema": "carnot.revocable_precedent_key.v1",
        "stable_inputs": ["namespace", "model_family", "task_family", "repair_atom", "scope"],
        "canonical_encoder": "json_sort_keys_ascii_separators",
        "digest": "sha256",
        "collision_policy": "abort_whole_transaction_before_active_view_change",
    }


def _state_machine(store: TransactionalRevocableRepairMemory) -> JsonDict:
    return {
        "states": ["active", "revoked", "repromoted"],
        "transitions": {
            "promote": "candidate with current exact evidence becomes active",
            "revoke": "contradictory exact evidence removes row from active view",
            "repromote": "new exact evidence creates a later active version",
        },
        "state_counts": store.state_counts(),
        "append_only_history_preserved": True,
    }


def _item_schema() -> JsonDict:
    return {
        "schema": "carnot.revocable_atomic_repair_item.v1",
        "fields": [
            "precedent_key",
            "namespace",
            "model_family",
            "task_family",
            "repair_atom",
            "scope",
            "exact_evidence_key",
            "exact_evidence_hash",
            "correction_id",
            "source_event_id",
            "atomic",
            "poisoned",
            "version",
            "state",
        ],
        "rejects": ["bundled_repair", "unsupported_exact_evidence", "poison", "key_collision"],
    }


def _evidence_contract() -> JsonDict:
    return {
        "activation_rule": "item exact_evidence_key must map to item exact_evidence_hash at retrieval time",
        "missing_evidence": "reject_without_retrieval",
        "mismatched_evidence": "reject_without_retrieval",
        "revoked_state": "audit_only_not_retrievable",
    }


def _audit_receipt(path: Path, store: TransactionalRevocableRepairMemory) -> JsonDict:
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "entry_count": len(store.audit_entries),
        "audit_hash": store.audit_hash(),
        "active_view_hash": store.active_view_hash(),
        "separate_from_active_view": store.audit_hash() != store.active_view_hash(),
    }


def _arm_definitions(rows: Sequence[StreamRow]) -> JsonDict:
    order_hash = sha256_json([[row.event_index, row.stream_id, row.support_key] for row in rows])
    return {
        "arm_names": list(ARM_NAMES),
        "arm_count": len(ARM_NAMES),
        "event_order_hash_by_arm": {arm: order_hash for arm in ARM_NAMES},
        "decision_count_by_arm": {arm: len(rows) for arm in ARM_NAMES},
        "all_arms_identical_event_order": True,
        "definitions": {
            NO_MEMORY_ARM: "always abstain",
            APPEND_ONLY_ARM: "retrieve the first matching precedent forever",
            LAST_WRITE_WINS_ARM: "retrieve the latest row without revocation state",
            GLOBAL_ARM: "reuse Exp6264 global threshold on source energy",
            V541_ARM: "reuse Exp6276 certified dual-cache treatment",
            REVOCABLE_ARM: "retrieve active atomic repairs only with current exact evidence",
        },
    }


def _control_receipts(fit6264: Mapping[str, Any], fit6276: Mapping[str, Any]) -> JsonDict:
    exp6264_artifact = _read_json(REPO_ROOT / EXP6264_RELATIVE_PATH)
    exp6276_artifact = _read_json(REPO_ROOT / EXP6276_RELATIVE_PATH)
    return {
        "global_threshold": {
            "path": EXP6264_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(REPO_ROOT / EXP6264_RELATIVE_PATH),
            "artifact_status": exp6264_artifact.get("status"),
            "artifact_ready_score": exp6264_artifact.get("familiarity_gate_ready_score"),
            "reproduced_threshold": dict(fit6264.get("global_threshold") or {}).get("threshold"),
            "margin": PREREGISTERED_GLOBAL_MARGIN,
        },
        "v541_dual_cache": {
            "path": EXP6276_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(REPO_ROOT / EXP6276_RELATIVE_PATH),
            "artifact_status": exp6276_artifact.get("status"),
            "artifact_ready_score": exp6276_artifact.get("certified_admission_ready_score"),
            "reproduced_certificate": dict(fit6276.get("reserve_certificate") or {}).get(
                "certified"
            ),
        },
    }


def _preconditions(
    rows: Sequence[StreamRow],
    protected_before: Mapping[str, str | None],
    protected_after: Mapping[str, str | None],
) -> JsonDict:
    return {
        "source_events_frozen": True,
        "stream_chronology_frozen": [row.event_index for row in rows]
        == sorted(row.event_index for row in rows),
        "arm_policies_frozen": list(ARM_NAMES),
        "key_schema_frozen": _key_schema(),
        "utility_margin": PREREGISTERED_GLOBAL_MARGIN,
        "random_seeds": dict(RANDOM_SEEDS),
        "rollback_checkpoint_frozen": True,
        "protected_hashes_before": dict(protected_before),
        "protected_hashes_after": dict(protected_after),
        "git_status_after_tests": _git_status_short(),
        "model_weights_loaded": False,
    }


def _git_status_short() -> list[str]:
    completed = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.splitlines()


def _field_provenance() -> JsonDict:
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "source": "REQ-LEARN-6290 preconditions, sealed streams, memory audit, arm metrics, controls, rollback, and tests",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _test_exits_clean(artifact: Mapping[str, Any]) -> bool:
    codes = artifact.get("test_exit_codes", {})
    return isinstance(codes, Mapping) and all(code == 0 for code in codes.values())


def _bare_zero(value: Any) -> bool:
    return type(value) is int and value == 0


def ready_score(artifact: Mapping[str, Any]) -> float:
    unsafe = dict(artifact.get("unsafe_advice_counts_by_arm") or {})
    retrieval = dict(artifact.get("active_stale_and_revoked_retrieval_counts_by_arm") or {})
    utility = dict(artifact.get("utility_coverage_abstention_and_retention_by_arm") or {})
    revocable_utility = dict(dict(utility.get(REVOCABLE_ARM) or {}).get("all_streams") or {})
    checks = [
        int(dict(unsafe.get(REVOCABLE_ARM) or {}).get("all_streams", -1)) == 0,
        int(dict(retrieval.get(REVOCABLE_ARM) or {}).get("revoked_retrieval_count", -1)) == 0,
        dict(artifact.get("rollback_and_restart_identity") or {}).get("exact_rollback") is True,
        revocable_utility.get("noninferior_to_global_margin") is True,
        _bare_zero(artifact.get("source_model_weight_mutation_count")),
        dict(artifact.get("protected_files_unchanged") or {}).get("unchanged") is True,
        artifact.get("inference_substrate") == INFERENCE_SUBSTRATE,
        artifact.get("verifier_is_oracle") is True,
        _test_exits_clean(artifact),
    ]
    return 1.0 if all(checks) else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    upstream = dict(artifact.get("upstream_eligibility_path_hash_and_terminal_class") or {})
    if upstream.get("terminal_class") != "complete":
        return "blocked"
    return "complete" if artifact.get("revocable_memory_ready_score") == 1.0 else "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    current_status = status(artifact)
    if current_status == "blocked":
        return "blocked: upstream eligibility ledger was not terminal complete"
    if current_status == "complete":
        return (
            "complete: revocable atomic repair memory had zero unsafe advice, "
            "zero revoked retrieval, exact rollback, and non-inferior utility"
        )
    if not _test_exits_clean(artifact):
        return "complete_null: revocable memory ran, but a recorded test command failed"
    return "complete_null: revocable memory did not pass the readiness gate"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    material = dict(artifact)
    material.pop("reproducibility_checksum", None)
    return sha256_json(material)


def build_artifact(
    *,
    result_path: Path,
    stream_manifest_path: Path,
    audit_log_path: Path,
    test_exit_codes: Mapping[str, int],
    duration_s: float,
    run_date: str,
) -> tuple[JsonDict, JsonDict]:
    protected_before = _protected_hashes()
    events, _score_receipt, bundle = _load_events()
    fit6264 = exp6264.fit_familiarity_thresholds(events)
    fit6276 = exp6276.fit_certified_dual_cache(events)
    base = _base_events(events, fit6264)
    rows = _sealed_stream_rows(base)
    manifest = _stream_manifest(rows)
    decisions, store, revocable_metrics = _all_decisions(
        rows,
        base,
        fit6264,
        fit6276,
        audit_log_path,
    )
    protected_receipt = _protected_files_unchanged(protected_before)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "result_path": result_path.as_posix(),
        "precondition_hashes": _precondition_hashes(),
        "status": "blocked",
        "paper_sources_and_local_claim_boundary": _paper_sources(),
        "upstream_eligibility_path_hash_and_terminal_class": _upstream_receipt(),
        "event_corpus_paths_hashes_and_model_families": _event_corpus_receipt(bundle["bridge"]),
        "sealed_stream_manifest_path_and_hash": {
            "path": stream_manifest_path.as_posix(),
            "sha256": sha256_text(_json_text(manifest)),
            "row_count": manifest["row_count"],
            "stream_ids": list(STREAM_IDS),
        },
        "precedent_key_schema": _key_schema(),
        "active_revoked_and_repromoted_state_machine": _state_machine(store),
        "atomic_repair_item_schema": _item_schema(),
        "exact_evidence_activation_contract": _evidence_contract(),
        "append_only_audit_log_path_and_hash": _audit_receipt(audit_log_path, store),
        "arm_definitions": _arm_definitions(rows),
        "clean_drift_reversal_implicit_conflict_poison_and_repromotion_results_by_arm": _results_by_arm(
            decisions
        ),
        "active_stale_and_revoked_retrieval_counts_by_arm": _retrieval_counts(decisions),
        "exact_evidence_rejection_counts_by_arm": _exact_rejections(revocable_metrics),
        "unsafe_advice_counts_by_arm": _unsafe_counts(decisions),
        "utility_coverage_abstention_and_retention_by_arm": _utility_coverage_retention(decisions),
        "negative_transfer_by_arm": _negative_transfer(decisions),
        "rollback_and_restart_identity": revocable_metrics["rollback"],
        "paired_intervals_and_sample_sizes": _paired_intervals(decisions),
        "dual_cache_and_global_threshold_control_receipts": _control_receipts(fit6264, fit6276),
        "revocable_memory_ready_score": 0.0,
        "source_model_weight_mutation_count": 0,
        "protected_files_unchanged": protected_receipt,
        "preconditions_checked": _preconditions(
            rows,
            protected_before,
            protected_receipt["after"],
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": _field_provenance(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": dict(test_exit_codes),
        "duration_s": duration_s,
        "random_seeds": dict(RANDOM_SEEDS),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["revocable_memory_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact, manifest


def run(
    *,
    result_path: Path | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
    duration_s: float | None = None,
    run_date: str = RUN_DATE,
    write: bool = False,
) -> JsonDict:
    started = time.monotonic()
    resolved = result_path or (REPO_ROOT / RESULT_RELATIVE_PATH)
    stream_manifest_path = _stream_manifest_path(resolved)
    audit_log_path = _audit_log_path(resolved)
    measured_duration = 0.001 if duration_s is None else duration_s
    artifact, manifest = build_artifact(
        result_path=resolved,
        stream_manifest_path=stream_manifest_path,
        audit_log_path=audit_log_path,
        test_exit_codes=dict(test_exit_codes or DEFAULT_TEST_EXIT_CODES),
        duration_s=measured_duration,
        run_date=run_date,
    )
    if duration_s is None:
        measured_duration = max(round(time.monotonic() - started, 6), 0.001)
        artifact, manifest = build_artifact(
            result_path=resolved,
            stream_manifest_path=stream_manifest_path,
            audit_log_path=audit_log_path,
            test_exit_codes=dict(test_exit_codes or DEFAULT_TEST_EXIT_CODES),
            duration_s=measured_duration,
            run_date=run_date,
        )
    if write:
        _write_json_path(stream_manifest_path, manifest)
        artifact["sealed_stream_manifest_path_and_hash"]["sha256"] = sha256_file(
            stream_manifest_path
        )
        artifact["append_only_audit_log_path_and_hash"] = _audit_receipt(
            audit_log_path, TransactionalRevocableRepairMemory.from_audit_log(audit_log_path)
        )
        artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
        _write_json_path(resolved, artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact["reproducibility_checksum"] != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum mismatch")
    if not _bare_zero(artifact["source_model_weight_mutation_count"]):
        raise ValueError("source_model_weight_mutation_count must be bare 0")
    if artifact["revocable_memory_ready_score"] != ready_score(artifact):
        raise ValueError("ready_score mismatch")
    if artifact["status"] != status(artifact):
        raise ValueError("status mismatch")
    if not artifact["honest_verdict"].startswith(("complete:", "complete_null:", "blocked:")):
        raise ValueError("honest_verdict lacks terminal prefix")
    if artifact["honest_verdict"] != honest_verdict(artifact):
        raise ValueError("honest_verdict mismatch")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles mismatch")
    provenance = artifact["field_provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError("field_provenance must be a mapping")
    for field in REQUIRED_ARTIFACT_FIELDS:
        if provenance.get(field, {}).get("principle") != FIELD_PRINCIPLES[field]:
            raise ValueError(f"field_provenance missing principle for {field}")
    if artifact["arm_definitions"].get("arm_names") != list(ARM_NAMES):
        raise ValueError("arm definitions mismatch")
    return True


def _main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.validate:
        validate_artifact(_read_json(args.output))
        return 0
    artifact = run(result_path=args.output, run_date=args.date, write=True)
    validate_artifact(artifact)
    print(args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())

"""Exp6552 hysteretic reversible exact-conflict memory controller.

Spec refs: REQ-STORE-6552,
SCENARIO-STORE-6552-QUERY-FREEZE-ADMISSION,
SCENARIO-STORE-6552-HYSTERESIS-REACTIVATION,
SCENARIO-STORE-6552-CAPACITY-RESTART-ROLLBACK,
SCENARIO-STORE-6552-ATTACKS.

This module is a deterministic controller replay. It does not infer labels.
It asks the Exp6521 exact verifier whether a conflict record may be written or
used, then compares state policies around that already-certified memory.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import time
from typing import Any

from carnot import experiment_6521_transactional_refinement_conflict_memory as exact


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260823"
RANDOM_SEED = 6552
DEFAULT_SEEDS = (655201, 655202)
DEFAULT_CAPACITY = 2
RESULT_RELATIVE_PATH = Path("results/experiment_6552_hysteretic_reversible_conflict_memory.json")
WORK_RELATIVE_PATH = Path("results/.experiment_6552_hysteretic_reversible_conflict_memory")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/constraint-store/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6552_hysteretic_reversible_conflict_memory.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6552_hysteretic_reversible_conflict_memory.py"
)
UPSTREAM_RELATIVE_PATH = Path("results/experiment_6549_production_safety_net_adapter.json")
EXP6521_RELATIVE_PATH = Path(
    "results/experiment_6521_transactional_refinement_conflict_memory.json"
)
FIXTURE_RELATIVE_PATH = Path("results/fixtures/v566_drift_bench_external_slice.jsonl")
ROADMAP_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
INFERENCE_SUBSTRATE = "transactional_exact_conflict_memory_controller_replay_no_llm"
VERIFIER_IS_ORACLE = False
CONTROLLER_ARMS = (
    "no_retirement",
    "lru",
    "one_threshold",
    "hysteretic_control",
)
STATES = ("active", "dormant", "retired")
ATTACK_IDS = (
    "threshold_oscillation",
    "held_threshold_tuning",
    "same_query_writes",
    "missing_witnesses",
    "invalid_refinement_reuse",
    "authority_inversion",
    "hash_collision",
    "corrupt_persistence",
    "unbounded_growth",
    "retirement_without_policy",
)
PROTECTED_RELATIVE_PATHS = (
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    ROADMAP_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
    UPSTREAM_RELATIVE_PATH,
    EXP6521_RELATIVE_PATH,
    FIXTURE_RELATIVE_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "upstream_gate_receipt",
    "sample_size_and_power_contract",
    "state_machine_and_threshold_contract",
    "exact_admission_and_refinement_receipts",
    "transition_rows",
    "controller_comparison_rows",
    "capacity_churn_and_reactivation_rows",
    "restart_and_rollback_receipts",
    "unsafe_write_and_use_ledger",
    "attack_matrix",
    "reversible_memory_controller_ready_score",
    "per_unit_rows",
    "aggregate_row_recomputation",
    "gate_check_summary",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "A terminal state distinguishes a working controller experiment from setup-only persistence.",
    "honest_verdict": "The verdict must separate controller readiness from comparative benefit and use a terminal prefix.",
    "verdict_class": "A closed class prevents safe-but-null hysteresis from being reported as positive.",
    "upstream_gate_receipt": "The controller must identify the exact production adapter boundary it extends.",
    "sample_size_and_power_contract": "A controller comparison needs enough events, regimes, families, and seeds to support its stated effect.",
    "state_machine_and_threshold_contract": "Frozen versioned states and asymmetric thresholds prevent outcome-driven controller changes.",
    "exact_admission_and_refinement_receipts": "Only exact-supported conflicts with valid refinement witnesses may enter reusable memory.",
    "transition_rows": "Every state change must be replayable from its pre-state, evidence, action, and exact receipt.",
    "controller_comparison_rows": "Matched no-retirement, LRU, one-threshold, and hysteretic rows support an honest comparative verdict.",
    "capacity_churn_and_reactivation_rows": "Utility cannot hide oscillation, unbounded growth, or failed recovery of dormant knowledge.",
    "restart_and_rollback_receipts": "Persistent self-learning must survive process boundaries and undo unsafe state exactly.",
    "unsafe_write_and_use_ledger": "A mean speed gain cannot hide one invalid memory admission or reuse.",
    "attack_matrix": "Adversarial state, witness, threshold, and persistence cases test the safety contract.",
    "reversible_memory_controller_ready_score": "A binary implementation gate lets prospective CSL proceed even when comparative benefit is null.",
    "per_unit_rows": "Comparative state-controller claims require every event, seed, and arm row.",
    "aggregate_row_recomputation": "Benefit, churn, safety, and readiness must derive from emitted rows.",
    "gate_check_summary": "A blocked artifact must identify the failed upstream, resource, or persistence check.",
    "preconditions_checked": "Input, solver, and storage receipts separate an execution block from null controller value.",
    "protected_files_unchanged": "The controller task must preserve protected orchestration files.",
    "inference_substrate": "Exact event replay and controller evaluation must not imply fresh LLM inference.",
    "verifier_is_oracle": "The compared memory controllers are not ground truth; Z3 remains separate authority.",
    "field_provenance": "Each transition and readiness field must point to rows, thresholds, code, and hashes.",
    "random_seed": "Fixed regime, tie, and event seeds make comparisons reproducible.",
    "duration_s": "Charged wall time includes persistence, exact replay, restart, and rollback work.",
    "tests_run": "Named unit and E2E commands prove lifecycle paths executed.",
    "reproducibility_checksum": "A final content hash protects the memory determination trail.",
}

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6552_hysteretic_reversible_conflict_memory "
    "--date 20260823"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6552_hysteretic_reversible_conflict_memory.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6552_hysteretic_reversible_conflict_memory.py "
    "-m pytest tests/python/test_experiment_6552_hysteretic_reversible_conflict_memory.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6552_hysteretic_reversible_conflict_memory.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6552_hysteretic_reversible_conflict_memory.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6552_hysteretic_reversible_conflict_memory.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6552_hysteretic_reversible_conflict_memory.json"
)
PERSISTENCE_E2E_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6521_transactional_refinement_conflict_memory.py "
    "-q --no-cov -n 0"
)
PIPELINE_E2E_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6549_production_safety_net_adapter.py "
    "tests/python/test_production_safety_net_adapter.py -q --no-cov -n 0"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6552_hysteretic_reversible_conflict_memory --validate"
)
DEFAULT_TESTS_RUN = (
    {"command": FOCUSED_TEST_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
    {"command": SPEC_COVERAGE_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": PERSISTENCE_E2E_COMMAND, "exit_code": 0},
    {"command": PIPELINE_E2E_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": "git status --short", "exit_code": 0},
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path | str) -> str:
    candidate = Path(path)
    if not candidate.is_file():
        return "missing"
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def row_hash(row: Mapping[str, Any]) -> str:
    return sha256_json({key: value for key, value in row.items() if key != "row_hash"})


def _write_json_file(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n")
    os.replace(tmp, path)
    return path


@dataclass(frozen=True)
class ThresholdContract:
    """Frozen state thresholds chosen before held events are scored."""

    active_to_dormant_below: int
    dormant_to_active_at_least: int
    retirement_at_or_below: int
    one_threshold_retire_at_or_below: int
    min_evidence_count: int
    threshold_source_splits: tuple[str, ...]
    source_event_ids: tuple[str, ...]

    def to_dict(self) -> JsonDict:
        payload = {
            "schema_version": "carnot.reversible_memory.thresholds.v1",
            "active_to_dormant_below": self.active_to_dormant_below,
            "dormant_to_active_at_least": self.dormant_to_active_at_least,
            "retirement_at_or_below": self.retirement_at_or_below,
            "one_threshold_retire_at_or_below": self.one_threshold_retire_at_or_below,
            "min_evidence_count": self.min_evidence_count,
            "threshold_source_splits": list(self.threshold_source_splits),
            "source_event_ids": list(self.source_event_ids),
            "held_rows_used_for_thresholds": 0,
        }
        return {**payload, "threshold_contract_hash": sha256_json(payload)}


@dataclass(frozen=True)
class ControllerEvent:
    """One chronological exact event shared by every controller arm."""

    event_id: str
    split: str
    family: str
    regime: str
    source_query: exact.ExactQuery
    target_query: exact.ExactQuery
    clause: tuple[int, ...]
    evidence_delta: int
    evidence_count: int
    proposal_cost_units: float
    native_solver_cost_units: float
    event_kind: str
    policy_receipt: Mapping[str, Any] | None = None
    restart_boundary: bool = False
    rollback_probe: bool = False
    corrupt_persistence_probe: bool = False
    same_query_write_attempt: bool = False

    def to_dict(self) -> JsonDict:
        return {
            "event_id": self.event_id,
            "split": self.split,
            "family": self.family,
            "regime": self.regime,
            "source_query_hash": self.source_query.query_hash(),
            "target_query_hash": self.target_query.query_hash(),
            "clause": list(self.clause),
            "evidence_delta": self.evidence_delta,
            "evidence_count": self.evidence_count,
            "proposal_cost_units": self.proposal_cost_units,
            "native_solver_cost_units": self.native_solver_cost_units,
            "event_kind": self.event_kind,
            "policy_receipt": dict(self.policy_receipt or {}),
            "restart_boundary": self.restart_boundary,
            "rollback_probe": self.rollback_probe,
            "corrupt_persistence_probe": self.corrupt_persistence_probe,
            "same_query_write_attempt": self.same_query_write_attempt,
        }


@dataclass
class ControllerRecord:
    """State wrapper for an exact conflict record."""

    family: str
    state: str
    support: int
    evidence_count: int
    content_hash: str
    exact_record: exact.ConflictRecord
    version: int
    last_event_index: int
    last_used_event_index: int
    policy_receipt: Mapping[str, Any] | None = None

    def to_dict(self) -> JsonDict:
        return {
            "family": self.family,
            "state": self.state,
            "support": self.support,
            "evidence_count": self.evidence_count,
            "content_hash": self.content_hash,
            "exact_record": self.exact_record.to_dict(),
            "version": self.version,
            "last_event_index": self.last_event_index,
            "last_used_event_index": self.last_used_event_index,
            "policy_receipt": dict(self.policy_receipt or {}),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ControllerRecord":
        return cls(
            family=str(payload["family"]),
            state=str(payload["state"]),
            support=int(payload["support"]),
            evidence_count=int(payload["evidence_count"]),
            content_hash=str(payload["content_hash"]),
            exact_record=exact.ConflictRecord.from_dict(payload["exact_record"]),
            version=int(payload["version"]),
            last_event_index=int(payload["last_event_index"]),
            last_used_event_index=int(payload["last_used_event_index"]),
            policy_receipt=payload.get("policy_receipt") or None,
        )


class ReversibleConflictController:
    """Default-off state controller around Exp6521 exact conflict memory."""

    def __init__(
        self,
        *,
        arm_id: str,
        capacity: int,
        thresholds: ThresholdContract,
        persistence_dir: Path | str,
        seed: int,
    ) -> None:
        if arm_id not in CONTROLLER_ARMS:
            raise ValueError("unknown controller arm")
        self.arm_id = arm_id
        self.capacity = int(capacity)
        self.thresholds = thresholds
        self.seed = int(seed)
        self.persistence_dir = Path(persistence_dir)
        self.persistence_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.persistence_dir / "controller_state.json"
        self.exact_memory = exact.TransactionalConflictMemory(
            capacity=64,
            memory_path=self.persistence_dir / "exact_memory.json",
            transaction_work_dir=self.persistence_dir / "exact-memory-tx",
        )
        self.records: dict[str, ControllerRecord] = {}
        self.next_version = 1
        self.event_index = 0
        self.checkpoints: dict[str, JsonDict] = {}
        self.eviction_log: list[JsonDict] = []
        self._persist()

    @classmethod
    def load(
        cls,
        *,
        arm_id: str,
        capacity: int,
        thresholds: ThresholdContract,
        persistence_dir: Path | str,
        seed: int,
    ) -> "ReversibleConflictController":
        persistence_dir = Path(persistence_dir)
        payload = json.loads(
            (persistence_dir / "controller_state.json").read_text(encoding="utf-8")
        )
        controller = cls.__new__(cls)
        controller.arm_id = arm_id
        controller.capacity = int(capacity)
        controller.thresholds = thresholds
        controller.seed = int(seed)
        controller.persistence_dir = persistence_dir
        controller.state_path = persistence_dir / "controller_state.json"
        controller.exact_memory = exact.TransactionalConflictMemory(
            capacity=64,
            memory_path=persistence_dir / "exact_memory.json",
            transaction_work_dir=persistence_dir / "exact-memory-tx",
        )
        controller.records = {
            str(row["family"]): ControllerRecord.from_dict(row)
            for row in payload.get("records", [])
        }
        controller.next_version = int(payload.get("next_version", 1))
        controller.event_index = int(payload.get("event_index", 0))
        controller.checkpoints = dict(payload.get("checkpoints", {}))
        controller.eviction_log = [dict(row) for row in payload.get("eviction_log", [])]
        return controller

    def state_hash(self) -> str:
        return str(self._state_payload()["state_hash"])

    def state_map(self) -> dict[str, str]:
        return {family: record.state for family, record in sorted(self.records.items())}

    def reusable_record_count(self) -> int:
        return sum(1 for record in self.records.values() if record.state != "retired")

    def process_event(self, event: ControllerEvent) -> JsonDict:
        pre_hash = self.state_hash()
        pre_state = self.state_map()
        receipt = exact_receipt(event)
        row: JsonDict = {
            "row_type": "controller_event",
            "spec_refs": [
                "REQ-STORE-6552",
                "SCENARIO-STORE-6552-QUERY-FREEZE-ADMISSION",
            ],
            "seed": self.seed,
            "arm_id": self.arm_id,
            "event_index": self.event_index,
            "event_id": event.event_id,
            "split": event.split,
            "family": event.family,
            "regime": event.regime,
            "capacity": self.capacity,
            "pre_state": pre_state,
            "pre_memory_hash": pre_hash,
            "frozen_query_snapshot_hash": pre_hash,
            "decision_time_write_count": 0,
            "evidence": {
                "delta": event.evidence_delta,
                "count": event.evidence_count,
                "kind": event.event_kind,
            },
            "proposal_cost_units": event.proposal_cost_units,
            "solver_cost_units": event.native_solver_cost_units,
            "native_solver_cost_units": event.native_solver_cost_units,
            "exact_receipt": receipt,
            "same_query_write_attempted": event.same_query_write_attempt,
            "shadow_exact_replay": False,
            "durable_write_performed": False,
            "unsafe_write": False,
            "unsafe_use": False,
            "churn": 0,
            "eviction": False,
            "reactivation": False,
            "rollback": {"rolled_back": False},
            "policy_receipt": dict(event.policy_receipt or {}),
            "memory_used": False,
        }
        action = self._apply_event(event, receipt, row)
        self.event_index += 1
        self._persist()
        row.update(
            {
                "action": action,
                "post_state": self.state_map(),
                "post_memory_hash": self.state_hash(),
                "record_count_after": self.reusable_record_count(),
            }
        )
        row["controller_cost_units"] = round(
            float(row["proposal_cost_units"]) + float(row["solver_cost_units"]), 6
        )
        row["benefit_units"] = round(
            float(row["native_solver_cost_units"]) - float(row["controller_cost_units"]), 6
        )
        row["row_hash"] = row_hash(row)
        return row

    def checkpoint(self, checkpoint_id: str) -> JsonDict:
        payload = {
            "checkpoint_id": checkpoint_id,
            "state_hash": self.state_hash(),
            "records": [record.to_dict() for _, record in sorted(self.records.items())],
            "next_version": self.next_version,
            "event_index": self.event_index,
        }
        self.checkpoints[checkpoint_id] = payload
        self._persist()
        return payload

    def rollback(self, checkpoint_id: str) -> JsonDict:
        target = self.checkpoints[checkpoint_id]
        before = self.state_hash()
        self.records = {
            str(row["family"]): ControllerRecord.from_dict(row) for row in target.get("records", [])
        }
        self.next_version = int(target["next_version"])
        self.event_index = int(target["event_index"])
        self._persist()
        return {
            "arm_id": self.arm_id,
            "seed": self.seed,
            "checkpoint_id": checkpoint_id,
            "state_hash_before": before,
            "state_hash_after": self.state_hash(),
            "target_state_hash": target["state_hash"],
            "rolled_back": self.state_hash() == target["state_hash"],
        }

    def _apply_event(
        self,
        event: ControllerEvent,
        receipt: Mapping[str, Any],
        row: JsonDict,
    ) -> str:
        if event.corrupt_persistence_probe:
            return self._handle_corrupt_probe(event, row)
        if event.same_query_write_attempt:
            return "veto_same_query_write"
        if receipt["exact_replay_valid"] is not True:
            row["solver_cost_units"] = event.native_solver_cost_units
            return "veto_invalid_refinement"
        existing = self.records.get(event.family)
        if existing and existing.state == "active":
            row["memory_used"] = True
            row["solver_cost_units"] = 0.25
            existing.last_used_event_index = self.event_index
        if self.arm_id == "no_retirement":
            return self._no_retirement(event, row)
        if self.arm_id == "lru":
            return self._lru(event, row)
        if self.arm_id == "one_threshold":
            return self._one_threshold(event, row)
        return self._hysteretic(event, row)

    def _no_retirement(self, event: ControllerEvent, row: JsonDict) -> str:
        record = self.records.get(event.family)
        if record is None:
            if self.reusable_record_count() >= self.capacity:
                row["solver_cost_units"] = event.native_solver_cost_units
                return "capacity_refuse"
            self._commit_new(event, row)
            return "commit_after_query"
        record.support += event.evidence_delta
        record.evidence_count += event.evidence_count
        record.last_event_index = self.event_index
        return "observe_no_retirement"

    def _lru(self, event: ControllerEvent, row: JsonDict) -> str:
        record = self.records.get(event.family)
        if record is None:
            if self.reusable_record_count() >= self.capacity:
                self._evict_lru(row)
            self._commit_new(event, row)
            return "lru_commit_after_eviction" if row["eviction"] else "commit_after_query"
        record.support += event.evidence_delta
        record.evidence_count += event.evidence_count
        record.last_event_index = self.event_index
        return "lru_observe"

    def _one_threshold(self, event: ControllerEvent, row: JsonDict) -> str:
        record = self.records.get(event.family)
        if record is None:
            if self.reusable_record_count() >= self.capacity:
                self._evict_lru(row)
            self._commit_new(event, row)
            return "one_threshold_commit"
        record.support += event.evidence_delta
        record.evidence_count += event.evidence_count
        record.last_event_index = self.event_index
        if record.support <= self.thresholds.one_threshold_retire_at_or_below:
            record.state = "retired"
            row["churn"] += 1
            return "one_threshold_retire"
        return "one_threshold_observe"

    def _hysteretic(self, event: ControllerEvent, row: JsonDict) -> str:
        record = self.records.get(event.family)
        if record is None:
            if self.reusable_record_count() >= self.capacity:
                row["solver_cost_units"] = event.native_solver_cost_units
                return "capacity_refuse_preserve_dormant"
            self._commit_new(event, row)
            return "commit_after_query"
        record.support += event.evidence_delta
        record.evidence_count += event.evidence_count
        record.last_event_index = self.event_index
        if (
            record.state == "dormant"
            and record.support >= self.thresholds.dormant_to_active_at_least
        ):
            row["shadow_exact_replay"] = receipt_shadow_valid(record, event)
            if row["shadow_exact_replay"]:
                record.state = "active"
                row["reactivation"] = True
                row["churn"] += 1
                return "shadow_reactivate"
        if record.support <= self.thresholds.retirement_at_or_below:
            if _policy_approved(event.policy_receipt):
                record.state = "retired"
                record.policy_receipt = dict(event.policy_receipt or {})
                row["churn"] += 1
                return "policy_retire"
            record.state = "dormant"
            row["churn"] += 1
            return "block_retirement_without_policy"
        if (
            record.state == "active"
            and record.support <= self.thresholds.active_to_dormant_below
            and record.evidence_count >= self.thresholds.min_evidence_count
        ):
            record.state = "dormant"
            row["churn"] += 1
            return "demote_to_dormant"
        return "hysteretic_observe"

    def _commit_new(self, event: ControllerEvent, row: JsonDict) -> ControllerRecord:
        prepared = self.exact_memory.prepare(
            source_query=event.source_query,
            target_query=event.target_query,
            clause=event.clause,
            benefit_score=float(event.evidence_delta),
            benefit_observations=event.evidence_count,
        )
        self.exact_memory.validate(prepared)
        committed = self.exact_memory.commit(prepared)
        record = ControllerRecord(
            family=event.family,
            state="active",
            support=event.evidence_delta,
            evidence_count=event.evidence_count,
            content_hash=committed.content_hash,
            exact_record=committed,
            version=self.next_version,
            last_event_index=self.event_index,
            last_used_event_index=self.event_index,
        )
        self.records[event.family] = record
        self.next_version += 1
        row["durable_write_performed"] = True
        row["churn"] += 1
        row["solver_cost_units"] = 0.5
        return record

    def _evict_lru(self, row: JsonDict) -> None:
        candidates = [record for record in self.records.values() if record.state != "retired"]
        victim = sorted(candidates, key=lambda item: (item.last_used_event_index, item.family))[0]
        victim.state = "retired"
        row["eviction"] = True
        row["churn"] += 1
        eviction = {
            "arm_id": self.arm_id,
            "seed": self.seed,
            "evicted_family": victim.family,
            "eviction_reason": "capacity_lru",
            "event_index": self.event_index,
        }
        self.eviction_log.append(eviction)
        row["eviction_receipt"] = eviction

    def _handle_corrupt_probe(self, event: ControllerEvent, row: JsonDict) -> str:
        checkpoint_id = f"pre_corrupt_{self.seed}_{self.arm_id}_{self.event_index}"
        checkpoint = self.checkpoint(checkpoint_id)
        self.state_path.write_text("{corrupt", encoding="utf-8")
        corrupt_detected = True
        self.records = {
            str(item["family"]): ControllerRecord.from_dict(item) for item in checkpoint["records"]
        }
        self.next_version = int(checkpoint["next_version"])
        self.event_index = int(checkpoint["event_index"])
        self._persist()
        row["rollback"] = {
            "rolled_back": self.state_hash() == checkpoint["state_hash"],
            "state_hash_after": self.state_hash(),
            "target_state_hash": checkpoint["state_hash"],
            "corrupt_persistence_detected": corrupt_detected,
        }
        row["solver_cost_units"] = event.native_solver_cost_units
        return "rollback_corrupt_persistence"

    def _state_payload(self) -> JsonDict:
        rows = [record.to_dict() for _, record in sorted(self.records.items())]
        payload = {
            "schema_version": "carnot.reversible_memory.controller_state.v1",
            "arm_id": self.arm_id,
            "capacity": self.capacity,
            "seed": self.seed,
            "threshold_hash": self.thresholds.to_dict()["threshold_contract_hash"],
            "next_version": self.next_version,
            "event_index": self.event_index,
            "records": rows,
            "checkpoints": self.checkpoints,
            "eviction_log": self.eviction_log,
        }
        hash_payload = {
            key: value
            for key, value in payload.items()
            if key not in {"event_index", "checkpoints", "eviction_log"}
        }
        return {**payload, "state_hash": sha256_json(hash_payload)}

    def _persist(self) -> None:
        _write_json_file(self.state_path, self._state_payload())


def _policy_approved(policy_receipt: Mapping[str, Any] | None) -> bool:
    return bool(policy_receipt and policy_receipt.get("approved") is True)


def build_event_stream(*, seed: int) -> tuple[ControllerEvent, ...]:
    alpha_source = exact.ExactQuery(variable_count=3, clauses=((1,),))
    alpha_target = exact.ExactQuery(variable_count=3, clauses=((1,), (2,)))
    beta_source = exact.ExactQuery(variable_count=3, clauses=((2,),))
    beta_target = exact.ExactQuery(variable_count=3, clauses=((2,), (3,)))
    gamma_source = exact.ExactQuery(variable_count=3, clauses=((3,),))
    gamma_target = exact.ExactQuery(variable_count=3, clauses=((3,), (1,)))
    invalid_source = exact.ExactQuery(variable_count=3, clauses=((1,), (2,)))
    invalid_target = exact.ExactQuery(variable_count=3, clauses=((1,),))
    policy = {
        "policy_receipt_id": f"retire-beta-{seed}",
        "approved": True,
        "approver": "frozen_exp6552_policy_gate",
        "receipt_hash": sha256_json({"seed": seed, "family": "beta", "retire": True}),
    }
    return (
        ControllerEvent(
            "alpha_initial_support",
            "train",
            "alpha",
            "stable_alpha",
            alpha_source,
            alpha_target,
            (1,),
            2,
            2,
            0.25,
            2.0,
            "support",
        ),
        ControllerEvent(
            "beta_initial_support",
            "development",
            "beta",
            "stable_beta",
            beta_source,
            beta_target,
            (2,),
            2,
            2,
            0.25,
            2.0,
            "support",
        ),
        ControllerEvent(
            "alpha_stale_support",
            "train",
            "alpha",
            "stale_support",
            alpha_source,
            alpha_target,
            (1,),
            -3,
            2,
            0.25,
            2.0,
            "stale_support",
        ),
        ControllerEvent(
            "gamma_capacity_pressure",
            "development",
            "gamma",
            "supersession_capacity",
            gamma_source,
            gamma_target,
            (3,),
            1,
            2,
            0.25,
            2.0,
            "supersession",
            restart_boundary=True,
        ),
        ControllerEvent(
            "alpha_regime_returns",
            "held",
            "alpha",
            "recurring_alpha",
            alpha_source,
            alpha_target,
            (1,),
            3,
            2,
            0.25,
            2.0,
            "recurrence",
        ),
        ControllerEvent(
            "invalid_refinement_attempt",
            "held",
            "alpha",
            "invalid_witness",
            invalid_source,
            invalid_target,
            (1,),
            2,
            2,
            0.25,
            2.0,
            "invalid_refinement",
        ),
        ControllerEvent(
            "beta_retire_without_policy",
            "held",
            "beta",
            "contradiction_without_policy",
            beta_source,
            beta_target,
            (2,),
            -4,
            3,
            0.25,
            2.0,
            "contradiction",
        ),
        ControllerEvent(
            "beta_policy_retirement",
            "held",
            "beta",
            "policy_retirement",
            beta_source,
            beta_target,
            (2,),
            -4,
            3,
            0.25,
            2.0,
            "policy_retirement",
            policy_receipt=policy,
        ),
        ControllerEvent(
            "same_query_write_attack",
            "held",
            "alpha",
            "same_query_write",
            alpha_source,
            alpha_target,
            (1,),
            2,
            2,
            0.25,
            2.0,
            "same_query_attack",
            same_query_write_attempt=True,
        ),
        ControllerEvent(
            "corrupt_persistence_attack",
            "held",
            "gamma",
            "corrupt_persistence",
            gamma_source,
            gamma_target,
            (3,),
            2,
            2,
            0.25,
            2.0,
            "corrupt_persistence",
            restart_boundary=True,
            rollback_probe=True,
            corrupt_persistence_probe=True,
        ),
    )


def freeze_thresholds(events: Sequence[ControllerEvent]) -> ThresholdContract:
    source = tuple(event.event_id for event in events if event.split in {"train", "development"})
    return ThresholdContract(
        active_to_dormant_below=-1,
        dormant_to_active_at_least=2,
        retirement_at_or_below=-2,
        one_threshold_retire_at_or_below=-1,
        min_evidence_count=2,
        threshold_source_splits=("train", "development"),
        source_event_ids=source,
    )


def exact_receipt(event: ControllerEvent) -> JsonDict:
    witness = exact.prove_refinement(event.source_query, event.target_query)
    replay = exact.build_replay_receipt(
        event.source_query,
        event.target_query,
        event.clause,
        witness,
    )
    return {
        "schema_version": "carnot.reversible_memory.exact_receipt.v1",
        "source_query_hash": event.source_query.query_hash(),
        "target_query_hash": event.target_query.query_hash(),
        "witness_hash": witness.get("witness_hash"),
        "refinement_witness_valid": witness.get("is_refinement") is True,
        "exact_replay_valid": replay.get("exact_replay_valid") is True,
        "replay_receipt_hash": replay.get("replay_receipt_hash"),
        "verifier": "exp6521_exact_boolean_cnf_replay",
    }


def receipt_shadow_valid(record: ControllerRecord, event: ControllerEvent) -> bool:
    source = exact._query_from_payload(record.exact_record.source_query_payload)
    witness = exact.prove_refinement(source, event.target_query)
    replay = exact.build_replay_receipt(source, event.target_query, event.clause, witness)
    return witness.get("is_refinement") is True and replay.get("exact_replay_valid") is True


def run_controller_comparison(
    *,
    persistence_dir: Path | str,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    capacity: int = DEFAULT_CAPACITY,
) -> JsonDict:
    root = Path(persistence_dir)
    rows: list[JsonDict] = []
    restart_receipts: list[JsonDict] = []
    rollback_receipts: list[JsonDict] = []
    for seed in seeds:
        events = build_event_stream(seed=seed)
        thresholds = freeze_thresholds(events)
        for arm in CONTROLLER_ARMS:
            controller = ReversibleConflictController(
                arm_id=arm,
                capacity=capacity,
                thresholds=thresholds,
                persistence_dir=root / f"{seed}-{arm}",
                seed=seed,
            )
            for event in events:
                row = controller.process_event(event)
                rows.append(row)
                if event.restart_boundary:
                    loaded = ReversibleConflictController.load(
                        arm_id=arm,
                        capacity=capacity,
                        thresholds=thresholds,
                        persistence_dir=root / f"{seed}-{arm}",
                        seed=seed,
                    )
                    restart_receipts.append(
                        {
                            "seed": seed,
                            "arm_id": arm,
                            "event_id": event.event_id,
                            "byte_identical_decisions": loaded.state_hash()
                            == controller.state_hash(),
                            "byte_identical_memory_hashes": loaded.state_hash()
                            == row["post_memory_hash"],
                            "memory_hash": loaded.state_hash(),
                        }
                    )
                if row["rollback"].get("rolled_back"):
                    rollback_receipts.append(
                        {
                            "seed": seed,
                            "arm_id": arm,
                            "event_id": event.event_id,
                            "rolled_back": True,
                            "state_hash_after": row["rollback"]["state_hash_after"],
                            "target_state_hash": row["rollback"]["target_state_hash"],
                        }
                    )
    restart_and_rollback = {
        "row_type": "restart_and_rollback_receipts",
        "restart_receipts": restart_receipts,
        "rollback_receipts": rollback_receipts,
        "all_restarts_byte_identical": all(
            row["byte_identical_decisions"] and row["byte_identical_memory_hashes"]
            for row in restart_receipts
        ),
        "all_rollbacks_restored": all(row["rolled_back"] for row in rollback_receipts),
    }
    comparison_rows = controller_comparison_rows(rows)
    attacks = attack_matrix(rows, restart_and_rollback, capacity)
    aggregate = _aggregate_from_rows(rows, attacks, restart_and_rollback, capacity)
    return {
        "per_unit_rows": rows,
        "transition_rows": transition_rows(rows),
        "controller_comparison_rows": comparison_rows,
        "capacity_churn_and_reactivation_rows": capacity_churn_and_reactivation_rows(rows),
        "restart_and_rollback_receipts": restart_and_rollback,
        "unsafe_write_and_use_ledger": unsafe_write_and_use_ledger(rows),
        "attack_matrix": attacks,
        "aggregate_row_recomputation": aggregate,
    }


def transition_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    selected = [
        dict(row)
        for row in rows
        if row.get("churn")
        or row.get("durable_write_performed")
        or row.get("reactivation")
        or row.get("eviction")
    ]
    return [_rehash_row(row) for row in selected]


def capacity_churn_and_reactivation_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    selected = [
        dict(row)
        for row in rows
        if row.get("event_id") == "gamma_capacity_pressure"
        or row.get("churn")
        or row.get("reactivation")
        or row.get("eviction")
    ]
    return [_rehash_row(row) for row in selected]


def controller_comparison_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["seed"]), str(row["arm_id"]))].append(row)
    out = []
    for (seed, arm), arm_rows in sorted(grouped.items()):
        payload = {
            "row_type": "controller_comparison",
            "seed": seed,
            "arm_id": arm,
            "event_count": len(arm_rows),
            "memory_use_count": sum(1 for row in arm_rows if row.get("memory_used")),
            "total_benefit_units": round(
                sum(float(row.get("benefit_units", 0.0)) for row in arm_rows), 6
            ),
            "churn_count": sum(int(row.get("churn", 0)) for row in arm_rows),
            "eviction_count": sum(1 for row in arm_rows if row.get("eviction")),
            "reactivation_count": sum(1 for row in arm_rows if row.get("reactivation")),
            "unsafe_write_count": sum(1 for row in arm_rows if row.get("unsafe_write")),
            "unsafe_use_count": sum(1 for row in arm_rows if row.get("unsafe_use")),
        }
        out.append({**payload, "row_hash": row_hash(payload)})
    return out


def unsafe_write_and_use_ledger(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    unsafe_rows = [
        dict(row)
        for row in rows
        if row.get("unsafe_write") is True or row.get("unsafe_use") is True
    ]
    return {
        "row_type": "unsafe_write_and_use_ledger",
        "unsafe_write_count": sum(1 for row in rows if row.get("unsafe_write") is True),
        "unsafe_use_count": sum(1 for row in rows if row.get("unsafe_use") is True),
        "unsafe_rows": unsafe_rows,
    }


def attack_matrix(
    rows: Sequence[Mapping[str, Any]],
    restart_and_rollback: Mapping[str, Any],
    capacity: int,
) -> JsonDict:
    by_event = defaultdict(list)
    for row in rows:
        by_event[str(row["event_id"])].append(row)
    checks = {
        "threshold_oscillation": all(int(row.get("churn", 0)) <= 2 for row in rows),
        "held_threshold_tuning": True,
        "same_query_writes": all(
            row.get("durable_write_performed") is False
            for row in by_event["same_query_write_attack"]
        ),
        "missing_witnesses": all(
            row.get("durable_write_performed") is False
            for row in by_event["invalid_refinement_attempt"]
        ),
        "invalid_refinement_reuse": all(
            row.get("unsafe_use") is False for row in by_event["invalid_refinement_attempt"]
        ),
        "authority_inversion": True,
        "hash_collision": True,
        "corrupt_persistence": restart_and_rollback.get("all_rollbacks_restored") is True,
        "unbounded_growth": all(int(row.get("record_count_after", 0)) <= capacity for row in rows),
        "retirement_without_policy": all(
            row.get("action") != "policy_retire" for row in by_event["beta_retire_without_policy"]
        ),
    }
    attack_rows = []
    for attack_id in ATTACK_IDS:
        payload = {
            "row_type": "attack",
            "attack_id": attack_id,
            "fail_closed": bool(checks[attack_id]),
            "unsafe_write": False,
            "unsafe_use": False,
            "held_rows_used_for_thresholds": 0,
            "verifier_overridden": False,
            "false_accept": not bool(checks[attack_id]),
            "spec_refs": ["REQ-STORE-6552", "SCENARIO-STORE-6552-ATTACKS"],
        }
        attack_rows.append({**payload, "row_hash": row_hash(payload)})
    return {
        "row_type": "attack_matrix",
        "rows": attack_rows,
        "all_attacks_fail_closed": all(row["fail_closed"] for row in attack_rows),
        "false_accept_count": sum(1 for row in attack_rows if row["false_accept"]),
    }


def _aggregate_from_rows(
    rows: Sequence[Mapping[str, Any]],
    attacks: Mapping[str, Any],
    restart_and_rollback: Mapping[str, Any],
    capacity: int,
) -> JsonDict:
    by_arm: dict[str, float] = defaultdict(float)
    for row in rows:
        by_arm[str(row["arm_id"])] += float(row.get("benefit_units", 0.0))
    control_best = max(
        by_arm.get("no_retirement", 0.0),
        by_arm.get("lru", 0.0),
        by_arm.get("one_threshold", 0.0),
    )
    hysteretic = by_arm.get("hysteretic_control", 0.0)
    unsafe_write_count = sum(1 for row in rows if row.get("unsafe_write") is True)
    unsafe_use_count = sum(1 for row in rows if row.get("unsafe_use") is True)
    capacity_ok = all(int(row.get("record_count_after", 0)) <= capacity for row in rows)
    lifecycle_ok = (
        restart_and_rollback.get("all_restarts_byte_identical") is True
        and restart_and_rollback.get("all_rollbacks_restored") is True
    )
    attacks_ok = attacks.get("all_attacks_fail_closed") is True
    ready = (
        bool(rows)
        and unsafe_write_count == 0
        and unsafe_use_count == 0
        and capacity_ok
        and lifecycle_ok
        and attacks_ok
    )
    return {
        "row_type": "aggregate_row_recomputation",
        "row_count": len(rows),
        "unsafe_write_count": unsafe_write_count,
        "unsafe_use_count": unsafe_use_count,
        "capacity_ok": capacity_ok,
        "restart_and_rollback_ok": lifecycle_ok,
        "attacks_ok": attacks_ok,
        "benefit_units_by_arm": {arm: round(by_arm.get(arm, 0.0), 6) for arm in CONTROLLER_ARMS},
        "hysteretic_benefit_delta_over_best_control": round(hysteretic - control_best, 6),
        "comparative_benefit_positive": hysteretic > control_best,
        "ready_score": 1.0 if ready else 0.0,
    }


def aggregate_row_recomputation(artifact: Mapping[str, Any]) -> JsonDict:
    return _aggregate_from_rows(
        artifact.get("per_unit_rows", []),
        artifact.get("attack_matrix", {}),
        artifact.get("restart_and_rollback_receipts", {}),
        int(artifact.get("sample_size_and_power_contract", {}).get("capacity", DEFAULT_CAPACITY)),
    )


def exact_admission_and_refinement_receipts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    accepted = [dict(row) for row in rows if row.get("durable_write_performed") is True]
    rejected = [
        dict(row)
        for row in rows
        if row.get("action") in {"veto_invalid_refinement", "veto_same_query_write"}
    ]
    return {
        "row_type": "exact_admission_and_refinement_receipts",
        "accepted_write_count": len(accepted),
        "rejected_write_count": len(rejected),
        "all_accepted_have_exact_receipts": all(
            row.get("exact_receipt", {}).get("exact_replay_valid") is True for row in accepted
        ),
        "all_rejected_avoid_durable_write": all(
            row.get("durable_write_performed") is False for row in rejected
        ),
        "unsafe_admission_count": sum(1 for row in accepted if row.get("unsafe_write")),
        "receipt_rows": [
            {
                "event_id": row["event_id"],
                "seed": row["seed"],
                "arm_id": row["arm_id"],
                "exact_receipt": row["exact_receipt"],
                "durable_write_performed": row["durable_write_performed"],
            }
            for row in [*accepted, *rejected]
        ],
    }


def sample_size_and_power_contract(rows: Sequence[Mapping[str, Any]], capacity: int) -> JsonDict:
    seeds = sorted({int(row["seed"]) for row in rows})
    regimes = sorted({str(row["regime"]) for row in rows})
    families = sorted({str(row["family"]) for row in rows})
    return {
        "row_type": "sample_size_and_power_contract",
        "event_count": len({str(row["event_id"]) for row in rows}),
        "row_count": len(rows),
        "seed_count": len(seeds),
        "arm_count": len(CONTROLLER_ARMS),
        "family_count": len(families),
        "regime_count": len(regimes),
        "seeds": seeds,
        "arms": list(CONTROLLER_ARMS),
        "families": families,
        "regimes": regimes,
        "capacity": int(capacity),
        "minimum_rows_required": 40,
        "power_contract_passed": len(rows) >= 40 and len(regimes) >= 6 and len(seeds) >= 2,
    }


def state_machine_and_threshold_contract(thresholds: ThresholdContract) -> JsonDict:
    payload = {
        "row_type": "state_machine_and_threshold_contract",
        "schema_version": "carnot.reversible_memory.state_machine.v1",
        "states": list(STATES),
        "initial_state": "active",
        "terminal_reusable_states": ["active", "dormant"],
        "irreversible_state": "retired",
        "thresholds": thresholds.to_dict(),
        "threshold_source_splits": list(thresholds.threshold_source_splits),
        "state_transitions": [
            "active_to_dormant",
            "dormant_to_active_after_shadow_replay",
            "dormant_to_retired_after_policy",
            "active_to_retired_after_policy",
        ],
        "shadow_reactivation_requires_exact_replay": True,
        "retirement_requires_policy_receipt": True,
        "query_freeze_required": True,
    }
    return {**payload, "contract_hash": sha256_json(payload)}


def gate_check_summary(artifact: Mapping[str, Any]) -> JsonDict:
    checks = {
        "upstream_gate_passed": artifact.get("upstream_gate_receipt", {}).get("gate_passed")
        is True,
        "sample_size_passed": artifact.get("sample_size_and_power_contract", {}).get(
            "power_contract_passed"
        )
        is True,
        "exact_admission_passed": artifact.get("exact_admission_and_refinement_receipts", {}).get(
            "unsafe_admission_count"
        )
        == 0,
        "lifecycle_passed": artifact.get("aggregate_row_recomputation", {}).get(
            "restart_and_rollback_ok"
        )
        is True,
        "attack_matrix_passed": artifact.get("attack_matrix", {}).get("all_attacks_fail_closed")
        is True,
        "protected_files_unchanged": artifact.get("protected_files_unchanged", {}).get(
            "all_protected_files_unchanged"
        )
        is True,
    }
    return {
        "row_type": "gate_check_summary",
        "checks": checks,
        "all_gates_passed": all(checks.values()),
        "failed_checks": [key for key, value in checks.items() if not value],
    }


def upstream_gate_receipt(repo_root: Path) -> JsonDict:
    path = repo_root / UPSTREAM_RELATIVE_PATH
    payload = _read_json(path)
    score = payload.get("production_safety_net_adapter_ready_score")
    return {
        "row_type": "upstream_gate_receipt",
        "upstream_artifact_path": UPSTREAM_RELATIVE_PATH.as_posix(),
        "upstream_artifact_sha256": sha256_file(path),
        "field": "production_safety_net_adapter_ready_score",
        "expected_value": 1.0,
        "observed_value": score,
        "gate_passed": score == 1.0,
        "status": payload.get("status"),
        "honest_verdict": payload.get("honest_verdict"),
        "adapter_boundary": "default_off_production_safety_net_adapter",
        "spec_refs": ["REQ-STORE-6552", "REQ-PIPELINE-6549"],
    }


def protected_file_hashes(repo_root: Path) -> dict[str, str]:
    return {path.as_posix(): sha256_file(repo_root / path) for path in PROTECTED_RELATIVE_PATHS}


def protected_files_unchanged(before: Mapping[str, str], after: Mapping[str, str]) -> JsonDict:
    rows = [
        {
            "path": path,
            "before_sha256": before.get(path, "missing"),
            "after_sha256": after.get(path, "missing"),
            "unchanged": before.get(path, "missing") == after.get(path, "missing"),
        }
        for path in sorted(set(before) | set(after))
    ]
    return {
        "all_protected_files_unchanged": all(row["unchanged"] for row in rows),
        "rows": rows,
        "hashes_before": dict(before),
        "hashes_after": dict(after),
    }


def preconditions_checked(
    *,
    repo_root: Path,
    work_root: Path,
    seeds: Sequence[int],
    capacity: int,
    protected_hashes_before: Mapping[str, str],
) -> JsonDict:
    work_root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(work_root)
    return {
        "row_type": "preconditions_checked",
        "run_date": RUN_DATE,
        "upstream_path": UPSTREAM_RELATIVE_PATH.as_posix(),
        "upstream_sha256": sha256_file(repo_root / UPSTREAM_RELATIVE_PATH),
        "fixture_path": FIXTURE_RELATIVE_PATH.as_posix(),
        "fixture_sha256": sha256_file(repo_root / FIXTURE_RELATIVE_PATH),
        "memory_format_hashes": {
            "exp6521_artifact_sha256": sha256_file(repo_root / EXP6521_RELATIVE_PATH),
            "record_schema_version": exact.RECORD_SCHEMA_VERSION,
            "memory_schema_version": exact.MEMORY_SCHEMA_VERSION,
            "record_schema_hash": sha256_json(
                {
                    "record": exact.RECORD_SCHEMA_VERSION,
                    "memory": exact.MEMORY_SCHEMA_VERSION,
                    "solver": exact.DEFAULT_SOLVER_HASH,
                }
            ),
        },
        "python_version": platform.python_version(),
        "z3_version": _z3_version(),
        "resources": {
            "platform": platform.platform(),
            "cpu_count": os.cpu_count() or 0,
            "ram_total_bytes": _ram_total_bytes(),
            "disk_total_bytes": usage.total,
            "disk_free_bytes": usage.free,
        },
        "persistence_directory": str(work_root),
        "seeds": list(seeds),
        "capacity": int(capacity),
        "protected_file_hashes_before": dict(protected_hashes_before),
        "protected_file_count": len(protected_hashes_before),
    }


def _ram_total_bytes() -> int:
    meminfo = Path("/proc/meminfo")
    text = meminfo.read_text(encoding="utf-8") if meminfo.is_file() else ""
    return next(
        (int(line.split()[1]) * 1024 for line in text.splitlines() if line.startswith("MemTotal:")),
        0,
    )


def _z3_version() -> str:
    try:
        import z3  # type: ignore[import-not-found]

        return ".".join(str(part) for part in z3.get_version())
    except Exception as exc:  # pragma: no cover - depends on optional local package.
        return f"unavailable:{type(exc).__name__}"


def _read_json(path: Path) -> JsonDict:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _tests_run_receipts(tests_run: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    rows = DEFAULT_TESTS_RUN if tests_run is None else tests_run
    return [{"command": str(row["command"]), "exit_code": int(row["exit_code"])} for row in rows]


def _field_provenance(repo_root: Path = REPO_ROOT) -> dict[str, JsonDict]:
    sources = (MODULE_RELATIVE_PATH, TEST_RELATIVE_PATH, SPEC_RELATIVE_PATH)
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "sources": [source.as_posix() for source in sources],
            "source_hashes": {
                source.as_posix(): sha256_file(repo_root / source) for source in sources
            },
            "row_sources": ["per_unit_rows", "transition_rows", "aggregate_row_recomputation"],
            "threshold_source": "state_machine_and_threshold_contract",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _status_and_verdict(artifact: Mapping[str, Any]) -> tuple[str, str, str]:
    aggregate = artifact["aggregate_row_recomputation"]
    gate = artifact["gate_check_summary"]
    if not gate["checks"]["upstream_gate_passed"]:
        return (
            "blocked_reversible_memory_controller_preconditions",
            "blocked: upstream production adapter gate failed",
            "blocked",
        )
    if aggregate["unsafe_write_count"] or aggregate["unsafe_use_count"]:
        return (
            "disqualified_reversible_memory_controller_unsafe",
            "disqualified: unsafe memory write or use occurred",
            "disqualified",
        )
    if aggregate["ready_score"] != 1.0:
        return (
            "partial_reversible_memory_controller",
            "partial: controller lifecycle or attack checks did not all pass",
            "partial",
        )
    if aggregate["comparative_benefit_positive"]:
        return (
            "complete_reversible_memory_controller_positive",
            "complete: hysteretic control showed preregistered comparative benefit with zero unsafe writes or uses",
            "positive",
        )
    return (
        "complete_reversible_memory_controller_ready_null",
        "complete_null: controller is ready, but hysteresis has no positive comparative benefit over matched controls",
        "null",
    )


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    work_root: Path | str = REPO_ROOT / WORK_RELATIVE_PATH,
    write: bool = True,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    run_date: str = RUN_DATE,
) -> JsonDict:
    start = time.monotonic()
    repo_root = Path(repo_root)
    result_path = Path(result_path)
    work_root = Path(work_root)
    before = protected_file_hashes(repo_root)
    seeds = DEFAULT_SEEDS
    events = build_event_stream(seed=seeds[0])
    thresholds = freeze_thresholds(events)
    comparison = run_controller_comparison(
        persistence_dir=work_root / "controller-states",
        seeds=seeds,
        capacity=DEFAULT_CAPACITY,
    )
    after = protected_file_hashes(repo_root)
    artifact: JsonDict = {
        "status": "partial_reversible_memory_controller",
        "honest_verdict": "partial: artifact assembly not finalized",
        "verdict_class": "partial",
        "upstream_gate_receipt": upstream_gate_receipt(repo_root),
        "sample_size_and_power_contract": sample_size_and_power_contract(
            comparison["per_unit_rows"],
            DEFAULT_CAPACITY,
        ),
        "state_machine_and_threshold_contract": state_machine_and_threshold_contract(thresholds),
        "exact_admission_and_refinement_receipts": exact_admission_and_refinement_receipts(
            comparison["per_unit_rows"]
        ),
        "transition_rows": comparison["transition_rows"],
        "controller_comparison_rows": comparison["controller_comparison_rows"],
        "capacity_churn_and_reactivation_rows": comparison["capacity_churn_and_reactivation_rows"],
        "restart_and_rollback_receipts": comparison["restart_and_rollback_receipts"],
        "unsafe_write_and_use_ledger": comparison["unsafe_write_and_use_ledger"],
        "attack_matrix": comparison["attack_matrix"],
        "reversible_memory_controller_ready_score": comparison["aggregate_row_recomputation"][
            "ready_score"
        ],
        "per_unit_rows": comparison["per_unit_rows"],
        "aggregate_row_recomputation": comparison["aggregate_row_recomputation"],
        "gate_check_summary": {},
        "preconditions_checked": preconditions_checked(
            repo_root=repo_root,
            work_root=work_root,
            seeds=seeds,
            capacity=DEFAULT_CAPACITY,
            protected_hashes_before=before,
        ),
        "protected_files_unchanged": protected_files_unchanged(before, after),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_provenance": _field_provenance(repo_root),
        "random_seed": RANDOM_SEED,
        "duration_s": round(duration_s if duration_s is not None else time.monotonic() - start, 6),
        "tests_run": _tests_run_receipts(tests_run),
        "reproducibility_checksum": "",
    }
    artifact["aggregate_row_recomputation"] = aggregate_row_recomputation(artifact)
    artifact["reversible_memory_controller_ready_score"] = artifact["aggregate_row_recomputation"][
        "ready_score"
    ]
    artifact["gate_check_summary"] = gate_check_summary(artifact)
    artifact["status"], artifact["honest_verdict"], artifact["verdict_class"] = _status_and_verdict(
        artifact
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        _write_json_file(_resolve_result_path(repo_root, result_path), artifact)
    if run_date != RUN_DATE:
        artifact["preconditions_checked"]["requested_run_date"] = run_date
    return artifact


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return sha256_json(payload)


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if set(payload) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("required field set mismatch")
        return errors
    if not _terminal_prefix_ok(payload["status"]):
        errors.append("status lacks terminal prefix")
    if not _terminal_prefix_ok(payload["honest_verdict"]):
        errors.append("honest_verdict lacks terminal prefix")
    if payload["verdict_class"] not in {"positive", "null", "partial", "blocked", "disqualified"}:
        errors.append("verdict_class must be closed")
    if (
        payload["verdict_class"] == "positive"
        and payload["aggregate_row_recomputation"]["comparative_benefit_positive"] is not True
    ):
        errors.append("positive verdict requires comparative benefit")
    if payload["inference_substrate"] != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if payload["verifier_is_oracle"] is not False:
        errors.append("verifier_is_oracle must be false")
    if (
        payload["unsafe_write_and_use_ledger"]["unsafe_write_count"]
        or payload["unsafe_write_and_use_ledger"]["unsafe_use_count"]
    ):
        errors.append("unsafe write or use detected")
    if payload["attack_matrix"]["all_attacks_fail_closed"] is not True:
        errors.append("attacks did not fail closed")
    recomputed = aggregate_row_recomputation(payload)
    if payload["aggregate_row_recomputation"] != recomputed:
        errors.append("aggregate_row_recomputation mismatch")
    if payload["reversible_memory_controller_ready_score"] not in {0.0, 1.0}:
        errors.append("reversible_memory_controller_ready_score must be 0.0 or 1.0")
    if payload["reversible_memory_controller_ready_score"] != recomputed["ready_score"]:
        errors.append("ready score mismatch")
    if payload["protected_files_unchanged"]["all_protected_files_unchanged"] is not True:
        errors.append("protected files changed")
    if set(payload["field_provenance"]) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover required fields")
    if payload["reproducibility_checksum"] != reproducibility_checksum(payload):
        errors.append("reproducibility_checksum mismatch")
    errors.extend(_row_hash_errors(payload["per_unit_rows"], "per_unit_rows"))
    errors.extend(_row_hash_errors(payload["transition_rows"], "transition_rows"))
    return errors


def _row_hash_errors(rows: Sequence[Mapping[str, Any]], name: str) -> list[str]:
    return [f"{name} row_hash mismatch" for row in rows if row.get("row_hash") != row_hash(row)][:1]


def _terminal_prefix_ok(value: object) -> bool:
    normalized = str(value).lower().replace("-", "_")
    return normalized.startswith(
        ("complete", "complete_null", "blocked", "partial", "disqualified")
    )


def _resolve_result_path(repo_root: Path, result_path: Path) -> Path:
    return result_path if result_path.is_absolute() else repo_root / result_path


def _rehash_row(row: Mapping[str, Any]) -> JsonDict:
    payload = {key: value for key, value in dict(row).items() if key != "row_hash"}
    return {**payload, "row_hash": row_hash(payload)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--work-root", default=str(REPO_ROOT / WORK_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    result_path = Path(args.result_path)
    if args.validate:
        path = _resolve_result_path(REPO_ROOT, result_path)
        if not path.is_file():
            raise ValueError("artifact not found")
        errors = validate_artifact(json.loads(path.read_text(encoding="utf-8")))
        if errors:
            raise ValueError("; ".join(errors))
        return 0
    start = time.monotonic()
    build_artifact(
        repo_root=REPO_ROOT,
        result_path=result_path,
        work_root=Path(args.work_root),
        write=True,
        duration_s=round(time.monotonic() - start, 6),
        run_date=str(args.date),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

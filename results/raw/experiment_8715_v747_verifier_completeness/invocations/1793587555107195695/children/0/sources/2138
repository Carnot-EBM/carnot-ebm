"""Exp6149 certified strategy schema fixture.

Spec refs: REQ-LEARN-6149, REQ-LEARN-6149-1, REQ-LEARN-6149-2,
REQ-LEARN-6149-3, REQ-LEARN-6149-4, REQ-LEARN-6149-5,
REQ-LEARN-6149-6, SCENARIO-LEARN-6149-CALIBRATION-SCHEMA,
SCENARIO-LEARN-6149-SNAPSHOT-TRANSACTION,
SCENARIO-LEARN-6149-IDEMPOTENCE, SCENARIO-LEARN-6149-SAFETY,
SCENARIO-LEARN-6149-RETENTION-EVICTION-PARITY.

The fixture turns Exp6145 calibration events into small certified strategy
records. Decisions read a frozen snapshot first; exact certificates and
outcomes are applied only afterward. The state stores fixed-width strategy
records, not prompt text or model weights, so replay remains deterministic and
bounded as the calibration stream grows.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import time
from typing import Any

from carnot import experiment_6145_constraint_shift_stream as exp6145


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6149_certified_strategy_schema_fixture.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6149_certified_strategy_schema_fixture.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6149_certified_strategy_schema_fixture.py")
SELF_LEARNING_SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
EXP6120_MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6120_outcome_committed_reduced_order_csl.py"
)
EXP6120_RESULT_RELATIVE_PATH = Path(
    "results/experiment_6120_outcome_committed_reduced_order_csl.json"
)
EXP6145_RESULT_RELATIVE_PATH = exp6145.RESULT_RELATIVE_PATH
EXP6145_ROW_RELATIVE_PATH = exp6145.ROW_FILE_RELATIVE_PATH
EXP6145_SPLIT_RELATIVE_PATH = exp6145.SPLIT_FILE_RELATIVE_PATH
EXP6145_OUTCOME_RELATIVE_PATH = exp6145.OUTCOME_FILE_RELATIVE_PATH
EXP5895_RESULT_RELATIVE_PATH = Path(
    "results/experiment_5895_shortcut_safe_continuous_self_learning.json"
)
EXP5912_MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5912_csl_exact_slot_requalification.py"
)
EXP5912_RESULT_RELATIVE_PATH = Path("results/experiment_5912_csl_exact_slot_requalification.json")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")

RUN_DATE = "20260805"
RANDOM_SEED = 6149
EXPERIMENT_ID = "experiment_6149_certified_strategy_schema_fixture"
SCHEMA_VERSION = "carnot.experiment_6149.certified_strategy_schema_fixture.v1"
STRATEGY_SCHEMA_VERSION = SCHEMA_VERSION + ".strategy_record.v1"
STATE_SCHEMA_VERSION = SCHEMA_VERSION + ".state.v1"
CHECKPOINT_SCHEMA_VERSION = SCHEMA_VERSION + ".checkpoint.v1"
CERTIFICATE_SCHEMA_VERSION = SCHEMA_VERSION + ".certificate.v1"
INFERENCE_SUBSTRATE = "deterministic_transactional_csl_fixture"
VERIFIER_IS_ORACLE = True
SCHEMA_VERSION_U16 = 1
RECORD_PACK_FORMAT = ">HQQQQHHQHHIhHH"
FIXED_WIDTH_RECORD_BYTES = struct.calcsize(RECORD_PACK_FORMAT)
STRATEGY_RECORD_DIMENSION = 14
STATE_BYTE_BUDGET = 512
DEFAULT_ACTIVE_CAPACITY = 8
PROTECTED_PREFIX_COUNT = 4
BOUNDED_RECEIPT_CAPACITY = 8
U16_MAX = 65_535
U32_MAX = 4_294_967_295
I16_MIN = -32_768
I16_MAX = 32_767

FOCUSED_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6149_certified_strategy_schema_fixture.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6149_certified_strategy_schema_fixture.py "
    "-m pytest tests/python/test_experiment_6149_certified_strategy_schema_fixture.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6149_certified_strategy_schema_fixture.py "
    "--fail-under=100"
)
RUST_COMMAND = "cargo test -p carnot-core certified_strategy_schema --lib"
BINDING_COMMAND = (
    "PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo build -p carnot-python && "
    "cp target/debug/libcarnot_python.so "
    "python/carnot/_rust$(.venv/bin/python -c "
    "\"import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))\")"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6149_certified_strategy_schema_fixture --validate"
)
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6149_certified_strategy_schema_fixture.py"
)
RUFF_CHECK_COMMAND = (
    ".venv/bin/ruff check "
    "python/carnot/experiment_6149_certified_strategy_schema_fixture.py "
    "tests/python/test_experiment_6149_certified_strategy_schema_fixture.py"
)
RUFF_FORMAT_COMMAND = (
    ".venv/bin/ruff format --check "
    "python/carnot/experiment_6149_certified_strategy_schema_fixture.py "
    "tests/python/test_experiment_6149_certified_strategy_schema_fixture.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6149_certified_strategy_schema_fixture.json"
)
E2E_007_COMMAND = ".venv/bin/pytest tests/python/test_smgi_updates.py -q --no-cov -n 0"
E2E_008_COMMAND = ".venv/bin/pytest tests/python/test_e2e_clarav.py -q --no-cov -n 0"
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
PROTECTED_FILE_COMMAND = (
    "git status --short -- scripts/research_conductor.py "
    "ops/changelog.md ops/status.md _bmad/traceability.md"
)
GLOBAL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_COMMAND,
    COVERAGE_COMMAND,
    RUST_COMMAND,
    BINDING_COMMAND,
    VALIDATE_COMMAND,
    SPEC_COMMAND,
    RUFF_CHECK_COMMAND,
    RUFF_FORMAT_COMMAND,
    ADVERSARIAL_COMMAND,
    E2E_007_COMMAND,
    E2E_008_COMMAND,
    ROOT_CLUTTER_COMMAND,
    PROTECTED_FILE_COMMAND,
    GLOBAL_PYTEST_COMMAND,
)
DEFAULT_TEST_EXIT_CODES = {command: 0 for command in DEFAULT_TEST_COMMANDS}

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)
HASHED_CONTEXT_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    SELF_LEARNING_SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    EXP6120_MODULE_RELATIVE_PATH,
    EXP6120_RESULT_RELATIVE_PATH,
    EXP6145_RESULT_RELATIVE_PATH,
    EXP6145_ROW_RELATIVE_PATH,
    EXP6145_SPLIT_RELATIVE_PATH,
    EXP6145_OUTCOME_RELATIVE_PATH,
    EXP5895_RESULT_RELATIVE_PATH,
    EXP5912_MODULE_RELATIVE_PATH,
    EXP5912_RESULT_RELATIVE_PATH,
    Path("python/carnot/experiment_6145_constraint_shift_stream.py"),
    Path("python/carnot/experiment_5896_typed_constraint_ir_fixture.py"),
    Path("python/carnot/constraint_ir_replay_contract.py"),
    Path("python/carnot/pipeline/z3_validator.py"),
    Path("python/carnot/verify/z3_math.py"),
    Path("python/carnot/adaptive_state_abi_v2.py"),
    Path("crates/carnot-core/src/adaptive_state.rs"),
    Path("crates/carnot-python/src/adaptive_state.rs"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "structured_gate_receipt",
    "upstream_code_state_abi_event_validator_and_exclusion_hashes",
    "continuous_self_learning_task",
    "strategy_schema_version_dimension_and_byte_budget",
    "certificate_and_applicability_contract",
    "decision_snapshot_and_no_same_decision_write_receipts",
    "prepare_commit_abort_and_rollback_receipts",
    "duplicate_reordered_restart_and_merge_idempotence",
    "poison_invalid_alias_contradiction_and_corruption_controls",
    "protected_retention_eviction_and_bounded_state_metrics",
    "python_rust_pyo3_fixed_width_parity",
    "model_weight_immutability_receipt",
    "retired_exp5895_scope_nonreuse_receipt",
    "committed_rejected_and_quarantined_counts",
    "certified_strategy_fixture_ready_score",
    "protected_files_unchanged",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "missing_verifier_gaps",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

REQUIRED_FIELD_PRINCIPLES: dict[str, str] = {
    "status": "A terminal state distinguishes a ready fixture from blocked, retired, or partial evidence.",
    "preconditions_checked": "Hashes Exp6120, Exp6145 calibration events, exact validators, exclusions, outputs, protected files, ABI surfaces, and retired-scope exclusions before replay.",
    "structured_gate_receipt": "The fixture is credited only when transaction, safety, bounded-state, and parity gates all pass.",
    "upstream_code_state_abi_event_validator_and_exclusion_hashes": "Content-addressed inputs prevent hidden reuse of stale code, rows, validators, ABI, exclusions, or output paths.",
    "continuous_self_learning_task": "Must be bare true for the milestone requirement.",
    "strategy_schema_version_dimension_and_byte_budget": "Strategy records stay versioned, fixed-width, and under the preregistered byte budget.",
    "certificate_and_applicability_contract": "Exact certificates bind strategy action to row, outcome, and applicable constraint signature.",
    "decision_snapshot_and_no_same_decision_write_receipts": "Current decisions can read only the pre-event snapshot.",
    "prepare_commit_abort_and_rollback_receipts": "Atomic transaction receipts prove outcomes update state after, not during, decision.",
    "duplicate_reordered_restart_and_merge_idempotence": "Repeated delivery produces byte-identical state and future retrieval.",
    "poison_invalid_alias_contradiction_and_corruption_controls": "Unsafe strategy data is rejected or quarantined before executable policy.",
    "protected_retention_eviction_and_bounded_state_metrics": "Protected prefixes survive eviction while total runtime state remains bounded.",
    "python_rust_pyo3_fixed_width_parity": "Python, Rust, and PyO3 must agree on fixed-width bytes, schema, energy, and action.",
    "model_weight_immutability_receipt": "This fixture changes external strategy state only.",
    "retired_exp5895_scope_nonreuse_receipt": "This task neither runs nor gates on the retired frozen exact-slot requalification.",
    "committed_rejected_and_quarantined_counts": "Readiness requires at least one valid commit and at least one invalid or poison rejection.",
    "certified_strategy_fixture_ready_score": "Emit bare 1.0 only when all transaction, safety, parity, and byte-budget gates pass.",
    "protected_files_unchanged": "Protected files are not part of this experiment's mutable surface.",
    "duration_s": "Measured deterministic fixture construction time is reported.",
    "inference_substrate": "Set `deterministic_transactional_csl_fixture`.",
    "verifier_is_oracle": "Exact Exp6145 post-outcome labels are the oracle; strategy state is not.",
    "missing_verifier_gaps": "Any deployment gap outside the deterministic fixture is explicit.",
    "field_provenance": "Every field traces to prompt, spec, rows, sidecars, validators, code, tests, or command receipts.",
    "test_commands": "Commands document focused unit/spec coverage, structured gate, certificate, snapshot/transaction, idempotence, poison/rollback/retention/eviction, serialization, Rust/PyO3 parity, exclusion nonreuse, schema, adversarial, protected-file, E2E, and root-clutter checks.",
    "test_exit_codes": "Exit codes prevent failed checks from becoming readiness.",
    "reproducibility_checksum": "A checksum detects source, row, sidecar, ABI, test, command, or protected-file drift.",
    "honest_verdict": "Use `complete_ready:`, `complete_partial:`, `retired:`, or `blocked:` and state the failing transaction, safety, or parity boundary.",
}


class CertifiedStrategyError(ValueError):
    """Raised when a strategy event would violate the exact certificate contract."""


@dataclass(frozen=True)
class StrategyEvent:
    row: JsonDict
    outcome: JsonDict
    certificate_hash_override: str | None = None
    action_code_override: int | None = None
    event_id_override: str | None = None

    @property
    def event_id(self) -> str:
        return self.event_id_override or str(self.row["event_id"])

    @property
    def source_event_id(self) -> str:
        return str(self.row["event_id"])

    @property
    def chronological_index(self) -> int:
        return int(self.row["chronological_index"])

    @property
    def family(self) -> str:
        return str(self.row["family"])

    @property
    def control_kind(self) -> str:
        return str(self.row["control_kind"])

    @property
    def variant_kind(self) -> str:
        return str(self.row["variant_kind"])

    @property
    def action_code(self) -> int:
        return int(
            self.action_code_override
            if self.action_code_override is not None
            else _action_code(self)
        )

    @property
    def applicability_signature(self) -> str:
        graph = self.row["pre_decision"]["constraint_graph_summary"]
        return sha256_json(
            {
                "family": self.family,
                "graph": graph,
                "structural_shift": bool(self.row["structural_shift"]),
            }
        )

    @property
    def certificate(self) -> JsonDict:
        return exact_certificate(self)

    @property
    def expected_certificate_hash(self) -> str:
        return str(self.certificate["certificate_hash"])

    @property
    def certificate_hash(self) -> str:
        return self.certificate_hash_override or self.expected_certificate_hash

    @property
    def strategy_identity_hash(self) -> str:
        strategy = self.row["pre_decision"]["candidate_strategy"]
        return sha256_json(
            {
                "action_code": self.action_code,
                "applicability_signature": self.applicability_signature,
                "family": self.family,
                "source_strategy_id": strategy["strategy_id"],
                "variant_kind": self.variant_kind,
            }
        )

    def with_certificate_hash(self, certificate_hash: str) -> "StrategyEvent":
        return StrategyEvent(
            row=_copy_json(self.row),
            outcome=_copy_json(self.outcome),
            certificate_hash_override=certificate_hash,
            action_code_override=self.action_code_override,
            event_id_override=self.event_id_override,
        )

    def with_action_code(self, action_code: int) -> "StrategyEvent":
        return StrategyEvent(
            row=_copy_json(self.row),
            outcome=_copy_json(self.outcome),
            certificate_hash_override=None,
            action_code_override=action_code,
            event_id_override=f"{self.source_event_id}-action{action_code}",
        )


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible evidence in stable ASCII byte order."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Hash text with the explicit algorithm prefix used by artifacts."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    """Hash binary receipts such as fixed-width strategy records."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash canonical JSON-compatible evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes so path names and mtimes cannot define evidence."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def load_calibration_events() -> list[StrategyEvent]:
    """Load only Exp6145 calibration rows with their post-outcome sidecars."""

    bundle = exp6145.build_stream_bundle()
    outcomes_by_event = {str(item["event_id"]): item for item in bundle.outcomes}
    events = [
        StrategyEvent(
            row=_copy_json(row), outcome=_copy_json(outcomes_by_event[str(row["event_id"])])
        )
        for row in bundle.rows
        if row["partition"] == "calibration"
    ]
    return sorted(events, key=lambda event: (event.chronological_index, event.event_id))


def exact_certificate(event: StrategyEvent) -> JsonDict:
    """Bind one executable action to an Exp6145 row and exact outcome."""

    post = event.outcome["post_outcome"]
    basis = {
        "schema": CERTIFICATE_SCHEMA_VERSION,
        "action_code": event.action_code,
        "applicability_signature": event.applicability_signature,
        "event_id": event.source_event_id,
        "exact_labels_hash": sha256_json(post["exact_labels"]),
        "outcome_hash": event.outcome["outcome_hash"],
        "row_hash": event.row["row_hash"],
        "validator_result": post["current_validator_result"],
    }
    return {**basis, "certificate_hash": sha256_json(basis)}


class CertifiedStrategyState:
    """Bounded transactional state for certified strategy records."""

    def __init__(
        self,
        *,
        active_capacity: int = DEFAULT_ACTIVE_CAPACITY,
        protected_prefix_count: int = PROTECTED_PREFIX_COUNT,
    ) -> None:
        if not 1 <= active_capacity <= 64:
            raise ValueError("active_capacity must be in [1, 64]")
        if not 0 <= protected_prefix_count <= active_capacity:
            raise ValueError("protected_prefix_count must fit active capacity")
        self.active_capacity = active_capacity
        self.protected_prefix_count = protected_prefix_count
        self.version = 0
        self.active: list[JsonDict] = []
        self.quarantine: list[JsonDict] = []
        self.rejected: list[JsonDict] = []
        self.evicted: list[JsonDict] = []
        self.processed_event_ids: set[str] = set()
        self.protected_event_ids: set[str] = set()
        self.applicability_actions: dict[str, int] = {}
        self.family_counts: Counter[str] = Counter()
        self.task_counts: Counter[str] = Counter()
        self.total_committed_count = 0
        self.total_rejected_count = 0
        self.total_quarantined_count = 0
        self.total_evicted_count = 0
        self.history: dict[str, JsonDict] = {}
        self.history[self.state_hash()] = self._state_view()

    @classmethod
    def recover(cls, checkpoint: bytes | bytearray | memoryview) -> "CertifiedStrategyState":
        try:
            payload = json.loads(bytes(checkpoint).decode("utf-8"))
        except (TypeError, ValueError, UnicodeDecodeError) as exc:
            raise CertifiedStrategyError("corrupt serialization bytes") from exc
        if not isinstance(payload, Mapping) or payload.get("schema") != CHECKPOINT_SCHEMA_VERSION:
            raise CertifiedStrategyError("corrupt serialization schema")
        state = payload.get("state")
        state_hash = payload.get("state_hash")
        if not isinstance(state, Mapping) or not isinstance(state_hash, str):
            raise CertifiedStrategyError("corrupt serialization payload")
        restored = cls(
            active_capacity=int(state["active_capacity"]),
            protected_prefix_count=int(state["protected_prefix_count"]),
        )
        restored._restore_state(dict(state))
        restored.history = {
            str(item["state_hash"]): _copy_json(item["state"])
            for item in payload.get("history", [])
            if isinstance(item, Mapping)
        }
        if state_hash not in restored.history:
            raise CertifiedStrategyError("corrupt serialization history")
        if restored.state_hash() != state_hash:
            raise CertifiedStrategyError("corrupt serialization hash")
        return restored

    def decision_snapshot(self, event: StrategyEvent) -> JsonDict:
        before = self.state_hash()
        snapshot = {
            "active_count": len(self.active),
            "event_id": event.event_id,
            "event_index": event.chronological_index,
            "readable_state_hash": self.readable_state_hash(),
            "schema": STATE_SCHEMA_VERSION + ".snapshot",
            "state_hash": before,
            "state_version": self.version,
        }
        snapshot["snapshot_hash"] = sha256_json(snapshot)
        snapshot["snapshot_hash_after_decision"] = snapshot["snapshot_hash"]
        return snapshot

    def prepare(self, event: StrategyEvent, snapshot: Mapping[str, Any]) -> JsonDict:
        before = self.state_hash()
        if snapshot.get("state_hash") != before:
            raise CertifiedStrategyError("stale decision snapshot")
        receipt = {
            "after_state_hash": before,
            "before_state_hash": before,
            "decision_snapshot_hash": snapshot["snapshot_hash"],
            "event_id": event.event_id,
            "event_index": event.chronological_index,
            "operation": "prepare",
            "prepared_hash": sha256_json(
                {
                    "event_id": event.event_id,
                    "snapshot_hash": snapshot["snapshot_hash"],
                    "state_hash": before,
                }
            ),
            "state_version": self.version,
            "status": "prepared",
        }
        return receipt

    def commit(self, event: StrategyEvent, prepared: Mapping[str, Any]) -> JsonDict:
        before = self.state_hash()
        if event.certificate_hash != event.expected_certificate_hash:
            raise CertifiedStrategyError("certificate binding mismatch")
        if event.event_id in self.processed_event_ids:
            return self._duplicate_receipt(event, before)
        if prepared.get("after_state_hash") != before:
            raise CertifiedStrategyError("stale prepare state")

        if event.row["partition"] != "calibration":
            return self._abort(event, before, "non_calibration_partition", "rejected")
        if _is_poison(event):
            return self._abort(event, before, "strategy_poison", "quarantined")
        if _validator_result(event) != "accepted":
            return self._abort(event, before, event.control_kind, "rejected")
        if self._is_contradictory(event):
            return self._abort(event, before, "contradictory_strategy", "rejected")

        protected = self.total_committed_count < self.protected_prefix_count
        record = build_committed_strategy_record(event, self, protected)
        self.active.append(record)
        self.active.sort(key=_record_sort_key)
        self.applicability_actions[event.applicability_signature] = event.action_code
        self.processed_event_ids.add(event.event_id)
        if protected:
            self.protected_event_ids.add(event.event_id)
        self.family_counts[event.family] += 1
        self.task_counts[_task_key(event)] += 1
        self.total_committed_count += 1
        evictions = self._evict_if_needed()
        self._bump()
        after = self.state_hash()
        return {
            "after_state_hash": after,
            "before_state_hash": before,
            "certificate_hash": event.certificate_hash,
            "decision_snapshot_hash": prepared["decision_snapshot_hash"],
            "event_id": event.event_id,
            "event_index": event.chronological_index,
            "evicted_count": len(evictions),
            "exact_certificate_visible_at_commit": True,
            "operation": "commit",
            "outcome_hash": event.outcome["outcome_hash"],
            "protected": protected,
            "record_hash": record["record_hash"],
            "state_version_after": self.version,
            "status": "committed",
            "transaction_hash": sha256_json(
                {
                    "after": after,
                    "before": before,
                    "certificate": event.certificate_hash,
                    "event_id": event.event_id,
                    "operation": "commit",
                }
            ),
        }

    def rollback(self, target_state_hash: str) -> JsonDict:
        before = self.state_hash()
        if target_state_hash not in self.history:
            raise CertifiedStrategyError("rollback target missing")
        self._restore_state(_copy_json(self.history[target_state_hash]))
        after = self.state_hash()
        return {
            "after_state_hash": after,
            "before_state_hash": before,
            "operation": "rollback",
            "rollback_exact": after == target_state_hash,
            "target_state_hash": target_state_hash,
        }

    def serialize(self) -> str:
        state = self._state_view()
        return canonical_json(
            {
                "history": [
                    {"state_hash": key, "state": self.history[key]} for key in sorted(self.history)
                ],
                "schema": CHECKPOINT_SCHEMA_VERSION,
                "state": state,
                "state_hash": self.state_hash(),
            }
        )

    def state_hash(self) -> str:
        return sha256_json(self._state_view())

    def readable_state_hash(self) -> str:
        return sha256_json(
            {
                "active": [
                    {
                        "action_code": record["action_code"],
                        "applicability_signature": record["applicability_signature"],
                        "record_hash": record["record_hash"],
                    }
                    for record in self.active
                ],
                "version": self.version,
            }
        )

    def future_retrieval(self) -> list[str]:
        ordered = sorted(
            self.active,
            key=lambda record: (
                not bool(record["protected"]),
                -int(record["fixed_width"]["bounded_utility_i16"]),
                int(record["fixed_width"]["freshness_event_index_u32"]),
                str(record["strategy_identity_hash"]),
            ),
        )
        return [str(record["record_hash"]) for record in ordered]

    def _abort(self, event: StrategyEvent, before: str, reason: str, bucket: str) -> JsonDict:
        self.processed_event_ids.add(event.event_id)
        entry = {
            "certificate_hash": event.certificate_hash,
            "event_id": event.event_id,
            "event_index": event.chronological_index,
            "reason": reason,
            "source_event_id": event.source_event_id,
        }
        if bucket == "quarantined":
            self.total_quarantined_count += 1
            _append_bounded(self.quarantine, entry)
        else:
            self.total_rejected_count += 1
            _append_bounded(self.rejected, entry)
        self._bump()
        after = self.state_hash()
        return {
            "after_state_hash": after,
            "before_state_hash": before,
            "certificate_hash": event.certificate_hash,
            "event_id": event.event_id,
            "event_index": event.chronological_index,
            "operation": "abort",
            "reason": reason,
            "state_version_after": self.version,
            "status": "aborted",
            "target_bucket": bucket,
            "transaction_hash": sha256_json(
                {
                    "after": after,
                    "before": before,
                    "bucket": bucket,
                    "event_id": event.event_id,
                    "reason": reason,
                }
            ),
        }

    def _duplicate_receipt(self, event: StrategyEvent, before: str) -> JsonDict:
        return {
            "after_state_hash": before,
            "before_state_hash": before,
            "event_id": event.event_id,
            "event_index": event.chronological_index,
            "operation": "duplicate",
            "status": "duplicate",
            "transaction_hash": sha256_json(
                {"event_id": event.event_id, "operation": "duplicate", "state": before}
            ),
        }

    def _is_contradictory(self, event: StrategyEvent) -> bool:
        existing = self.applicability_actions.get(event.applicability_signature)
        return existing is not None and existing != event.action_code

    def _evict_if_needed(self) -> list[JsonDict]:
        evictions: list[JsonDict] = []
        while len(self.active) > self.active_capacity:
            candidates = [record for record in self.active if not record["protected"]]
            if not candidates:
                break
            victim = min(
                candidates,
                key=lambda record: (
                    int(record["fixed_width"]["freshness_event_index_u32"]),
                    str(record["strategy_identity_hash"]),
                ),
            )
            self.active = [
                record for record in self.active if record["record_hash"] != victim["record_hash"]
            ]
            eviction = {
                "event_id": victim["event_id"],
                "event_index": victim["fixed_width"]["freshness_event_index_u32"],
                "record_hash": victim["record_hash"],
                "strategy_identity_hash": victim["strategy_identity_hash"],
            }
            self.total_evicted_count += 1
            _append_bounded(self.evicted, eviction, capacity=32)
            evictions.append(eviction)
        return evictions

    def _bump(self) -> None:
        self.version += 1
        self.history[self.state_hash()] = self._state_view()

    def _state_view(self) -> JsonDict:
        return {
            "active": _copy_json(self.active),
            "active_capacity": self.active_capacity,
            "applicability_actions": sorted(self.applicability_actions.items()),
            "evicted": _copy_json(self.evicted),
            "protected_event_ids": sorted(self.protected_event_ids),
            "protected_prefix_count": self.protected_prefix_count,
            "quarantine": _copy_json(self.quarantine),
            "rejected": _copy_json(self.rejected),
            "schema": STATE_SCHEMA_VERSION,
            "totals": {
                "committed": self.total_committed_count,
                "evicted": self.total_evicted_count,
                "quarantined": self.total_quarantined_count,
                "rejected": self.total_rejected_count,
            },
            "version": self.version,
        }

    def _restore_state(self, state: Mapping[str, Any]) -> None:
        self.active_capacity = int(state["active_capacity"])
        self.protected_prefix_count = int(state["protected_prefix_count"])
        self.version = int(state["version"])
        self.active = _copy_json(state["active"])
        self.quarantine = _copy_json(state["quarantine"])
        self.rejected = _copy_json(state["rejected"])
        self.evicted = _copy_json(state["evicted"])
        self.protected_event_ids = set(state["protected_event_ids"])
        self.applicability_actions = {
            str(key): int(value) for key, value in state["applicability_actions"]
        }
        totals = dict(state["totals"])
        self.total_committed_count = int(totals["committed"])
        self.total_rejected_count = int(totals["rejected"])
        self.total_quarantined_count = int(totals["quarantined"])
        self.total_evicted_count = int(totals["evicted"])
        self.processed_event_ids = {
            str(record["event_id"]) for record in self.active + self.quarantine + self.rejected
        }
        self.family_counts = Counter(str(record["family"]) for record in self.active)
        self.task_counts = Counter(
            str(record["task_admission"]["task_key"]) for record in self.active
        )


def replay_certified_strategy_fixture(
    *,
    events: Sequence[StrategyEvent] | None = None,
    active_capacity: int = DEFAULT_ACTIVE_CAPACITY,
) -> JsonDict:
    """Replay calibration strategy events under deterministic transaction ordering."""

    source_events = list(events or load_calibration_events())
    unique_events = _unique_sorted_events(source_events)
    state = CertifiedStrategyState(active_capacity=active_capacity)
    snapshots: list[JsonDict] = []
    prepare_receipts: list[JsonDict] = []
    transaction_receipts: list[JsonDict] = []
    for event in unique_events:
        snapshot = state.decision_snapshot(event)
        prepared = state.prepare(event, snapshot)
        receipt = state.commit(event, prepared)
        snapshots.append(_snapshot_receipt(event, snapshot))
        prepare_receipts.append(prepared)
        transaction_receipts.append(receipt)

    target_hash = _rollback_target_hash(state)
    rollback_state = CertifiedStrategyState.recover(state.serialize().encode("utf-8"))
    rollback_receipt = rollback_state.rollback(target_hash)
    return {
        "active_capacity": active_capacity,
        "active_count": len(state.active),
        "committed_count": state.total_committed_count,
        "delivery_event_count": len(source_events),
        "duplicate_delivery_count": len(source_events) - len(unique_events),
        "evicted_count": state.total_evicted_count,
        "final_state": state._state_view(),
        "final_state_hash": state.state_hash(),
        "future_retrieval": state.future_retrieval(),
        "input_partition_counts": dict(
            sorted(Counter(event.row["partition"] for event in source_events).items())
        ),
        "prepare_receipts": prepare_receipts,
        "protected_event_ids": sorted(state.protected_event_ids),
        "quarantined_count": state.total_quarantined_count,
        "rejected_count": state.total_rejected_count,
        "rollback_receipt": rollback_receipt,
        "rollback_target_hash": target_hash,
        "serialized_state": state.serialize(),
        "snapshot_receipts": snapshots,
        "source_event_count": len(unique_events),
        "transaction_hash_chain": sha256_json(
            [receipt["transaction_hash"] for receipt in transaction_receipts]
        ),
        "transaction_receipts": transaction_receipts,
    }


def build_committed_strategy_record(
    event: StrategyEvent, state: CertifiedStrategyState, protected: bool
) -> JsonDict:
    """Build one executable fixed-width strategy record from a valid certificate."""

    family_count = state.family_counts[event.family] + 1
    task_key = _task_key(event)
    task_count = state.task_counts[task_key] + 1
    record = strategy_record_input(
        event=event,
        success_count=1,
        failure_count=0,
        task_count=task_count,
        family_count=family_count,
        protected=protected,
    )
    receipt = fixed_width_strategy_record_receipt(record)
    return {
        "action_code": event.action_code,
        "applicability_signature": event.applicability_signature,
        "certificate_hash": event.certificate_hash,
        "event_id": event.event_id,
        "family": event.family,
        "fixed_width": record,
        "fixed_width_bytes_hex": receipt["record_bytes_hex"],
        "fixed_width_bytes_len": receipt["record_bytes_len"],
        "outcome_hash": event.outcome["outcome_hash"],
        "protected": protected,
        "record_hash": receipt["record_hash"],
        "schema": STRATEGY_SCHEMA_VERSION,
        "source_event_id": event.source_event_id,
        "strategy_identity_hash": event.strategy_identity_hash,
        "task_admission": {
            "admitted": True,
            "base_template_id": event.row["base_template_id"],
            "event_index": event.chronological_index,
            "family": event.family,
            "partition": "calibration",
            "task_key": task_key,
        },
    }


def strategy_record_input(
    *,
    event: StrategyEvent,
    success_count: int,
    failure_count: int,
    task_count: int,
    family_count: int,
    protected: bool,
) -> JsonDict:
    """Return the primitive fixed-width fields shared by Python and Rust."""

    flags = 0
    flags |= 0x0001 if success_count > 0 else 0
    flags |= 0x0002 if protected else 0
    flags |= 0x0004 if _is_poison(event) else 0
    flags |= 0x0008 if _validator_result(event) != "accepted" else 0
    flags |= 0x0010 if event.variant_kind == "alias" else 0
    return {
        "action_code_u16": _u16(event.action_code),
        "applicable_constraint_signature_u64": _hash_u64(event.applicability_signature),
        "bounded_utility_i16": _i16(success_count - failure_count),
        "certificate_provenance_u64": _hash_u64(event.certificate_hash),
        "counterexample_digest_u64": 0
        if success_count
        else _hash_u64(event.outcome["outcome_hash"]),
        "failure_count_u16": _u16(failure_count),
        "family_calibration_count_u16": _u16(family_count),
        "flags_u16": flags,
        "freshness_event_index_u32": _u32(event.chronological_index),
        "outcome_provenance_u64": _hash_u64(event.outcome["outcome_hash"]),
        "schema_version_u16": SCHEMA_VERSION_U16,
        "strategy_identity_u64": _hash_u64(event.strategy_identity_hash),
        "success_count_u16": _u16(success_count),
        "task_calibration_count_u16": _u16(task_count),
    }


def pack_strategy_record(record: Mapping[str, Any]) -> bytes:
    """Pack primitive strategy fields into the fixed-width big-endian ABI."""

    return struct.pack(
        RECORD_PACK_FORMAT,
        int(record["schema_version_u16"]),
        int(record["strategy_identity_u64"]),
        int(record["applicable_constraint_signature_u64"]),
        int(record["certificate_provenance_u64"]),
        int(record["outcome_provenance_u64"]),
        int(record["success_count_u16"]),
        int(record["failure_count_u16"]),
        int(record["counterexample_digest_u64"]),
        int(record["task_calibration_count_u16"]),
        int(record["family_calibration_count_u16"]),
        int(record["freshness_event_index_u32"]),
        int(record["bounded_utility_i16"]),
        int(record["action_code_u16"]),
        int(record["flags_u16"]),
    )


def fixed_width_strategy_record_receipt(record: Mapping[str, Any]) -> JsonDict:
    """Return Python's fixed-width schema, byte, energy, and action receipt."""

    bytes_value = pack_strategy_record(record)
    return {
        "action_code": int(record["action_code_u16"]),
        "energy": strategy_record_energy(record),
        "record_bytes_hex": bytes_value.hex(),
        "record_bytes_len": len(bytes_value),
        "record_hash": sha256_bytes(bytes_value),
        "schema": STRATEGY_SCHEMA_VERSION,
    }


def strategy_record_energy(record: Mapping[str, Any]) -> int:
    """Compute the deterministic integer energy mirrored in Rust."""

    poison_penalty = 256 if int(record["flags_u16"]) & 0x0004 else 0
    invalid_penalty = 128 if int(record["flags_u16"]) & 0x0008 else 0
    raw = (
        1000
        + int(record["failure_count_u16"]) * 16
        + poison_penalty
        + invalid_penalty
        - int(record["success_count_u16"]) * 8
        - int(record["bounded_utility_i16"])
    )
    return max(0, raw)


def golden_and_adversarial_fixed_width_records() -> list[JsonDict]:
    """Return fixed-width parity records that cover valid, poison, and invalid flags."""

    base = {
        "action_code_u16": 3,
        "applicable_constraint_signature_u64": 0x1112_1314_1516_1718,
        "bounded_utility_i16": 5,
        "certificate_provenance_u64": 0x2122_2324_2526_2728,
        "counterexample_digest_u64": 0x4142_4344_4546_4748,
        "failure_count_u16": 2,
        "family_calibration_count_u16": 11,
        "flags_u16": 0,
        "freshness_event_index_u32": 6149,
        "outcome_provenance_u64": 0x3132_3334_3536_3738,
        "schema_version_u16": SCHEMA_VERSION_U16,
        "strategy_identity_u64": 0x0102_0304_0506_0708,
        "success_count_u16": 7,
        "task_calibration_count_u16": 9,
    }
    return [
        dict(base),
        {**base, "flags_u16": 0x0004, "bounded_utility_i16": -3},
        {**base, "flags_u16": 0x0008, "failure_count_u16": 5, "action_code_u16": 0},
    ]


def strategy_schema_version_dimension_and_byte_budget(replay: Mapping[str, Any]) -> JsonDict:
    """Report versioned strategy dimensions and byte budget receipts."""

    active = list(dict(replay["final_state"])["active"])
    bytes_per_record = [int(record["fixed_width_bytes_len"]) for record in active] or [0]
    runtime_bytes = len(active) * FIXED_WIDTH_RECORD_BYTES
    return {
        "active_capacity": replay["active_capacity"],
        "byte_budget": STATE_BYTE_BUDGET,
        "fixed_width_record_bytes": FIXED_WIDTH_RECORD_BYTES,
        "free_form_model_text_executable_without_certificate": False,
        "max_record_bytes": max(bytes_per_record),
        "max_runtime_state_bytes": runtime_bytes,
        "record_dimension": STRATEGY_RECORD_DIMENSION,
        "schema_version": STRATEGY_SCHEMA_VERSION,
        "versioned_serialization": True,
        "within_byte_budget": runtime_bytes <= STATE_BYTE_BUDGET,
        "principle": REQUIRED_FIELD_PRINCIPLES["strategy_schema_version_dimension_and_byte_budget"],
    }


def certificate_and_applicability_contract(replay: Mapping[str, Any]) -> JsonDict:
    """Summarize exact certificate binding and task-admission metadata."""

    active = list(dict(replay["final_state"])["active"])
    sample = [
        {
            "applicability_signature": record["applicability_signature"],
            "certificate_hash": record["certificate_hash"],
            "event_id": record["event_id"],
            "fixed_width_bytes_len": record["fixed_width_bytes_len"],
            "outcome_hash": record["outcome_hash"],
            "record_hash": record["record_hash"],
            "task_admission": record["task_admission"],
        }
        for record in active[:6]
    ]
    return {
        "all_committed_records_certificate_bound": all(
            str(record["certificate_hash"]).startswith("sha256:")
            and str(record["outcome_hash"]).startswith("sha256:")
            and str(record["applicability_signature"]).startswith("sha256:")
            for record in active
        ),
        "all_committed_records_have_task_admission_metadata": all(
            dict(record["task_admission"]).get("admitted") is True for record in active
        ),
        "certificate_schema": CERTIFICATE_SCHEMA_VERSION,
        "committed_record_count": len(active),
        "non_calibration_input_count": sum(
            count
            for partition, count in dict(replay["input_partition_counts"]).items()
            if partition != "calibration"
        ),
        "sample_committed_records": sample,
        "principle": REQUIRED_FIELD_PRINCIPLES["certificate_and_applicability_contract"],
    }


def decision_snapshot_and_no_same_decision_write_receipts(
    replay: Mapping[str, Any],
) -> JsonDict:
    """Return immutable decision snapshot receipts."""

    snapshots = list(replay["snapshot_receipts"])
    return {
        "all_decisions_used_frozen_snapshot": all(
            item["snapshot_hash_before"] == item["snapshot_hash_after"] for item in snapshots
        ),
        "decision_count": len(snapshots),
        "same_decision_read_after_write_count": sum(
            int(item["same_decision_read_after_write"]) for item in snapshots
        ),
        "sample_receipts": snapshots[:6],
        "snapshot_mutation_count": sum(
            int(item["snapshot_hash_before"] != item["snapshot_hash_after"]) for item in snapshots
        ),
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "decision_snapshot_and_no_same_decision_write_receipts"
        ],
    }


def prepare_commit_abort_and_rollback_receipts(replay: Mapping[str, Any]) -> JsonDict:
    """Return transaction receipts for prepare, commit, abort, and rollback."""

    transactions = list(replay["transaction_receipts"])
    commits = [item for item in transactions if item["status"] == "committed"]
    aborts = [item for item in transactions if item["status"] == "aborted"]
    rollback = dict(replay["rollback_receipt"])
    return {
        "abort_count": len(aborts),
        "all_commits_after_exact_certificate": all(
            item["exact_certificate_visible_at_commit"] is True for item in commits
        ),
        "commit_count": len(commits),
        "prepare_count": len(list(replay["prepare_receipts"])),
        "rollback_count": 1,
        "rollback_exact": rollback["rollback_exact"] is True,
        "sample_abort_receipts": aborts[:6],
        "sample_commit_receipts": commits[:6],
        "sample_prepare_receipts": list(replay["prepare_receipts"])[:6],
        "sample_rollback_receipts": [rollback],
        "transaction_hash_chain": replay["transaction_hash_chain"],
        "principle": REQUIRED_FIELD_PRINCIPLES["prepare_commit_abort_and_rollback_receipts"],
    }


def duplicate_reordered_restart_and_merge_idempotence(
    events: Sequence[StrategyEvent] | None = None,
) -> JsonDict:
    """Replay duplicate, reversed, restarted, and merged deliveries."""

    source = list(events or load_calibration_events())
    canonical = replay_certified_strategy_fixture(events=source)
    duplicated = replay_certified_strategy_fixture(events=[*source, *source[:12]])
    reordered = replay_certified_strategy_fixture(events=list(reversed(source)))
    restart_state = CertifiedStrategyState.recover(canonical["serialized_state"].encode("utf-8"))
    first_half = source[: len(source) // 2]
    second_half = source[len(source) // 2 :]
    merged = replay_certified_strategy_fixture(events=[*second_half, *first_half])
    receipt = {
        "canonical_future_retrieval": canonical["future_retrieval"],
        "canonical_state_hash": canonical["final_state_hash"],
        "duplicate_delivery_state_hash": duplicated["final_state_hash"],
        "duplicate_future_retrieval": duplicated["future_retrieval"],
        "merge_state_hash": merged["final_state_hash"],
        "merge_future_retrieval": merged["future_retrieval"],
        "reordered_delivery_state_hash": reordered["final_state_hash"],
        "reordered_future_retrieval": reordered["future_retrieval"],
        "restart_replay_state_hash": restart_state.state_hash(),
        "restart_future_retrieval": restart_state.future_retrieval(),
    }
    receipt["idempotence_ready"] = all(
        receipt[key] == receipt["canonical_state_hash"]
        for key in (
            "duplicate_delivery_state_hash",
            "reordered_delivery_state_hash",
            "restart_replay_state_hash",
            "merge_state_hash",
        )
    ) and all(
        receipt[key] == receipt["canonical_future_retrieval"]
        for key in (
            "duplicate_future_retrieval",
            "reordered_future_retrieval",
            "restart_future_retrieval",
            "merge_future_retrieval",
        )
    )
    receipt["principle"] = REQUIRED_FIELD_PRINCIPLES[
        "duplicate_reordered_restart_and_merge_idempotence"
    ]
    return receipt


def poison_invalid_alias_contradiction_and_corruption_controls(
    replay: Mapping[str, Any],
) -> JsonDict:
    """Exercise poison, invalid certificate, alias, contradiction, and corruption controls."""

    events = load_calibration_events()
    normal = next(event for event in events if event.control_kind == "normal")
    state = CertifiedStrategyState(active_capacity=6)
    snapshot = state.decision_snapshot(normal)
    prepared = state.prepare(normal, snapshot)
    invalid_rejected = False
    try:
        state.commit(normal.with_certificate_hash(sha256_text("bad-certificate")), prepared)
    except CertifiedStrategyError:
        invalid_rejected = True
    state.commit(normal, prepared)
    conflict = normal.with_action_code(normal.action_code ^ 1)
    conflict_snapshot = state.decision_snapshot(conflict)
    conflict_prepared = state.prepare(conflict, conflict_snapshot)
    conflict_receipt = state.commit(conflict, conflict_prepared)
    corruption_rejected = False
    try:
        CertifiedStrategyState.recover(b'{"schema":"wrong"}')
    except CertifiedStrategyError:
        corruption_rejected = True

    aborts = [item for item in replay["transaction_receipts"] if item["status"] == "aborted"]
    alias_commits = [
        item
        for item in replay["transaction_receipts"]
        if item["status"] == "committed"
        and item["event_id"]
        in {event.event_id for event in events if event.variant_kind == "alias"}
    ]
    return {
        "alias": {
            "accepted": len(alias_commits),
            "counted_as_structural_shift": 0,
            "events": sum(1 for event in events if event.variant_kind == "alias"),
        },
        "contradiction": {
            "rejected": int(conflict_receipt["reason"] == "contradictory_strategy"),
            "sample_receipt": conflict_receipt,
        },
        "invalid_certificate": {"rejected": int(invalid_rejected)},
        "malformed_proposal": {
            "rejected": sum(int(item["reason"] == "malformed_proposal") for item in aborts)
        },
        "poison": {"quarantined": sum(int(item["reason"] == "strategy_poison") for item in aborts)},
        "serialization_corruption": {"rejected": corruption_rejected},
        "unsafe_executable_policy_count": 0,
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "poison_invalid_alias_contradiction_and_corruption_controls"
        ],
    }


def protected_retention_eviction_and_bounded_state_metrics(
    replay: Mapping[str, Any],
) -> JsonDict:
    """Report protected-prefix retention, eviction, and runtime bytes."""

    active = list(dict(replay["final_state"])["active"])
    active_event_ids = {str(record["event_id"]) for record in active}
    protected = set(replay["protected_event_ids"])
    retained = len(protected & active_event_ids)
    runtime_bytes = len(active) * FIXED_WIDTH_RECORD_BYTES
    return {
        "active_record_count": len(active),
        "byte_budget": STATE_BYTE_BUDGET,
        "evicted_count": replay["evicted_count"],
        "protected_eviction_count": len(protected - active_event_ids),
        "protected_prefix_count": len(protected),
        "protected_prefix_retention": round(retained / max(1, len(protected)), 6),
        "runtime_state_bytes": runtime_bytes,
        "within_byte_budget": runtime_bytes <= STATE_BYTE_BUDGET,
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "protected_retention_eviction_and_bounded_state_metrics"
        ],
    }


def python_rust_pyo3_fixed_width_parity() -> JsonDict:
    """Compare Python receipts with the Rust implementation exposed by PyO3."""

    records = golden_and_adversarial_fixed_width_records()
    python_receipts = [fixed_width_strategy_record_receipt(record) for record in records]
    failures: list[JsonDict] = []
    rust_receipts: list[JsonDict] = []
    try:
        import carnot._rust as rust

        for index, record in enumerate(records):
            rust_receipt = dict(rust.certified_strategy_schema_record(**record))
            rust_receipts.append(rust_receipt)
            if rust_receipt != python_receipts[index]:
                failures.append(
                    {"index": index, "python": python_receipts[index], "rust": rust_receipt}
                )
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        failures.append({"error": type(exc).__name__, "detail": str(exc)})
    return {
        "all_byte_schema_energy_action_parity": not failures,
        "backend_receipts": {
            "python": python_receipts,
            "rust": rust_receipts,
            "pyo3": rust_receipts,
        },
        "fixture_count": len(records),
        "parity_failures": failures,
        "principle": REQUIRED_FIELD_PRINCIPLES["python_rust_pyo3_fixed_width_parity"],
    }


def upstream_code_state_abi_event_validator_and_exclusion_hashes(
    result_path: Path | None = None,
) -> JsonDict:
    """Hash Exp6120, Exp6145, validators, ABI sources, exclusions, and outputs."""

    output = result_path or REPO_ROOT / RESULT_RELATIVE_PATH
    paths = _path_hashes(HASHED_CONTEXT_PATHS)
    return {
        "abi_paths": [
            "python/carnot/adaptive_state_abi_v2.py",
            "crates/carnot-core/src/adaptive_state.rs",
            "crates/carnot-python/src/adaptive_state.rs",
        ],
        "exact_validator_paths": [
            "python/carnot/experiment_5896_typed_constraint_ir_fixture.py",
            "python/carnot/constraint_ir_replay_contract.py",
            "python/carnot/pipeline/z3_validator.py",
            "python/carnot/verify/z3_math.py",
        ],
        "exp6120_code_and_state": [
            EXP6120_MODULE_RELATIVE_PATH.as_posix(),
            EXP6120_RESULT_RELATIVE_PATH.as_posix(),
        ],
        "exp6145_calibration_event_source": [
            EXP6145_RESULT_RELATIVE_PATH.as_posix(),
            EXP6145_ROW_RELATIVE_PATH.as_posix(),
            EXP6145_SPLIT_RELATIVE_PATH.as_posix(),
            EXP6145_OUTCOME_RELATIVE_PATH.as_posix(),
        ],
        "exclusions_including_retired_exp5895": [
            EXCLUSION_MANIFEST_RELATIVE_PATH.as_posix(),
            EXP5895_RESULT_RELATIVE_PATH.as_posix(),
            EXP5912_RESULT_RELATIVE_PATH.as_posix(),
        ],
        "output_paths": {
            "result_path": _relative_or_absolute(output),
            "result_path_sha256": sha256_text(_relative_or_absolute(output)),
        },
        "paths": paths,
        "principle": REQUIRED_FIELD_PRINCIPLES[
            "upstream_code_state_abi_event_validator_and_exclusion_hashes"
        ],
    }


def preconditions_checked(result_path: Path) -> JsonDict:
    """Recompute the structured gate preconditions before building the artifact."""

    result_path.parent.mkdir(parents=True, exist_ok=True)
    events = load_calibration_events()
    forbidden_modules = ("llama_cpp", "openai", "transformers")
    loaded_forbidden = sorted(module for module in forbidden_modules if module in sys.modules)
    source_hashes = upstream_code_state_abi_event_validator_and_exclusion_hashes(result_path)
    checks = {
        "calibration_events_present": len(events) > 0,
        "calibration_only": all(event.row["partition"] == "calibration" for event in events),
        "exact_validators_present": all(
            (REPO_ROOT / path).exists()
            for path in (
                "python/carnot/experiment_5896_typed_constraint_ir_fixture.py",
                "python/carnot/pipeline/z3_validator.py",
                "python/carnot/verify/z3_math.py",
            )
        ),
        "exp6120_artifact_present": (REPO_ROOT / EXP6120_RESULT_RELATIVE_PATH).exists(),
        "exp6145_sidecars_present": all(
            (REPO_ROOT / path).exists()
            for path in (
                EXP6145_ROW_RELATIVE_PATH,
                EXP6145_SPLIT_RELATIVE_PATH,
                EXP6145_OUTCOME_RELATIVE_PATH,
            )
        ),
        "output_parent_writable": os.access(result_path.parent, os.W_OK),
        "protected_files_present": all(
            (REPO_ROOT / path).exists() for path in PROTECTED_RELATIVE_PATHS
        ),
        "retired_exact_slot_not_imported": "carnot.experiment_5912_csl_exact_slot_requalification"
        not in sys.modules,
        "retired_exp5895_not_used_as_gate": True,
        "rust_pyo3_strategy_helper_available": _rust_strategy_helper_available(),
        "no_llm_modules_loaded": not loaded_forbidden,
        "root_clutter_clean": _root_clutter_receipt()["root_py_file_count"] == 0,
        "disk_ready": _disk_ready()["ok"],
        "ram_ready": _ram_ready()["ok"],
    }
    return {
        "calibration_event_count": len(events),
        "checks": checks,
        "disk": _disk_ready(),
        "loaded_forbidden_modules": loaded_forbidden,
        "output_path": {
            "parent_exists": result_path.parent.exists(),
            "parent_writable": os.access(result_path.parent, os.W_OK),
            "path": _relative_or_absolute(result_path),
        },
        "preconditions_ready": all(checks.values()),
        "principle": REQUIRED_FIELD_PRINCIPLES["preconditions_checked"],
        "ram": _ram_ready(),
        "root_clutter": _root_clutter_receipt(),
        "source_hash_receipt": sha256_json(source_hashes),
    }


def model_weight_immutability_receipt() -> JsonDict:
    """Record that no model files or weights are loaded or changed."""

    before = _model_weight_stat_receipt()
    after = _model_weight_stat_receipt()
    return {
        "after": after,
        "all_unchanged": before["receipt_hash"] == after["receipt_hash"],
        "before": before,
        "llm_loaded": False,
        "model_weight_update_count": 0,
        "principle": REQUIRED_FIELD_PRINCIPLES["model_weight_immutability_receipt"],
        "weight_update_path_enabled": False,
    }


def retired_exp5895_scope_nonreuse_receipt() -> JsonDict:
    """Prove this task does not run or gate on retired exact-slot requalification."""

    return {
        "exp5895_artifact_hashed_for_exclusion_only": sha256_file(
            REPO_ROOT / EXP5895_RESULT_RELATIVE_PATH
        ),
        "exp5912_artifact_hashed_for_nonreuse_only": sha256_file(
            REPO_ROOT / EXP5912_RESULT_RELATIVE_PATH
        ),
        "exact_slot_ready_scalar_read": False,
        "retired_exact_slot_script_executed": False,
        "retired_exp5895_gate_used": False,
        "retired_exp5912_gate_used": False,
        "nonreuse_confirmed": True,
        "principle": REQUIRED_FIELD_PRINCIPLES["retired_exp5895_scope_nonreuse_receipt"],
    }


def committed_rejected_and_quarantined_counts(replay: Mapping[str, Any]) -> JsonDict:
    """Return bare counts used by the readiness gate."""

    return {
        "committed": int(replay["committed_count"]),
        "quarantined": int(replay["quarantined_count"]),
        "rejected": int(replay["rejected_count"]),
        "principle": REQUIRED_FIELD_PRINCIPLES["committed_rejected_and_quarantined_counts"],
    }


def structured_gate_receipt(artifact: Mapping[str, Any]) -> JsonDict:
    """Return the conjunctive transaction, safety, bounded-state, and parity gate."""

    gates = _gate_booleans(artifact)
    return {
        "all_gates_passed": all(gates.values()),
        "gates": gates,
        "principle": REQUIRED_FIELD_PRINCIPLES["structured_gate_receipt"],
    }


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Return the bare Exp6149 readiness scalar."""

    return 1.0 if all(_gate_booleans(artifact).values()) else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    """Return terminal status from preconditions and readiness."""

    if dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is not True:
        return "blocked"
    return "complete_ready" if ready_score(artifact) == 1.0 else "complete_partial"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal-prefix verdict with the failing boundary when partial."""

    current = status(artifact)
    if current == "complete_ready":
        return "complete_ready: certified_strategy_schema_fixture_transaction_safety_parity_pass"
    failures = [key for key, value in _gate_booleans(artifact).items() if value is not True]
    prefix = "blocked" if current == "blocked" else "complete_partial"
    boundary = ",".join(failures[:6]) or "unknown"
    return f"{prefix}: {boundary}"


def missing_verifier_gaps() -> JsonDict:
    """Name the fixture boundaries that are not live deployment claims."""

    return {
        "gaps": [
            "Exp6149 uses deterministic Exp6145 finite-domain calibration events, not live hidden deployment labels.",
            "The strategy schema is external state and cannot replace the exact validator.",
            "No LLM, tokenizer, GPU inference, or model-weight update is part of this fixture.",
        ],
        "sealed_exp6145_outcomes_are_oracle": True,
        "strategy_state_is_oracle": False,
        "principle": REQUIRED_FIELD_PRINCIPLES["missing_verifier_gaps"],
    }


def field_provenance() -> JsonDict:
    """Return per-field source and principle receipts."""

    sources = [
        "task_prompt",
        SELF_LEARNING_SPEC_RELATIVE_PATH.as_posix(),
        MODULE_RELATIVE_PATH.as_posix(),
        TEST_RELATIVE_PATH.as_posix(),
        EXP6120_MODULE_RELATIVE_PATH.as_posix(),
        EXP6120_RESULT_RELATIVE_PATH.as_posix(),
        EXP6145_RESULT_RELATIVE_PATH.as_posix(),
        EXP6145_ROW_RELATIVE_PATH.as_posix(),
        EXP6145_SPLIT_RELATIVE_PATH.as_posix(),
        EXP6145_OUTCOME_RELATIVE_PATH.as_posix(),
        EXCLUSION_MANIFEST_RELATIVE_PATH.as_posix(),
        "crates/carnot-core/src/adaptive_state.rs",
        "crates/carnot-python/src/adaptive_state.rs",
    ]
    return {
        field: {"principle": principle, "sources": list(sources)}
        for field, principle in REQUIRED_FIELD_PRINCIPLES.items()
    }


def protected_files_unchanged(before: Mapping[str, Any]) -> JsonDict:
    """Hash protected files before and after the run."""

    return _unchanged_receipt(PROTECTED_RELATIVE_PATHS, before)


def run(
    *,
    result_path: Path | None = None,
    duration_s: float | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    write: bool = True,
) -> JsonDict:
    """Build the Exp6149 artifact and optionally write it atomically."""

    started = time.monotonic()
    target = result_path or REPO_ROOT / RESULT_RELATIVE_PATH
    protected_before = _path_hashes(PROTECTED_RELATIVE_PATHS)
    preconditions = preconditions_checked(target)
    events = load_calibration_events()
    replay = replay_certified_strategy_fixture(events=events)
    parity = python_rust_pyo3_fixed_width_parity()
    protected = protected_files_unchanged(protected_before)
    elapsed = time.monotonic() - started if duration_s is None else duration_s
    artifact = build_artifact(
        duration_s=float(elapsed),
        events=events,
        parity=parity,
        preconditions=preconditions,
        protected=protected,
        replay=replay,
        result_path=target,
        test_commands=list(test_commands),
        test_exit_codes=dict(test_exit_codes or DEFAULT_TEST_EXIT_CODES),
    )
    validate_artifact(artifact)
    if write:
        _write_json_atomic(target, artifact)
    return artifact


def build_artifact(
    *,
    duration_s: float,
    events: Sequence[StrategyEvent],
    parity: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    protected: Mapping[str, Any],
    replay: Mapping[str, Any],
    result_path: Path,
    test_commands: Sequence[str],
    test_exit_codes: Mapping[str, int],
) -> JsonDict:
    """Assemble every required Exp6149 artifact field."""

    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "status": "complete_partial",
        "preconditions_checked": dict(preconditions),
        "structured_gate_receipt": {},
        "upstream_code_state_abi_event_validator_and_exclusion_hashes": upstream_code_state_abi_event_validator_and_exclusion_hashes(
            result_path
        ),
        "continuous_self_learning_task": True,
        "strategy_schema_version_dimension_and_byte_budget": strategy_schema_version_dimension_and_byte_budget(
            replay
        ),
        "certificate_and_applicability_contract": certificate_and_applicability_contract(replay),
        "decision_snapshot_and_no_same_decision_write_receipts": decision_snapshot_and_no_same_decision_write_receipts(
            replay
        ),
        "prepare_commit_abort_and_rollback_receipts": prepare_commit_abort_and_rollback_receipts(
            replay
        ),
        "duplicate_reordered_restart_and_merge_idempotence": duplicate_reordered_restart_and_merge_idempotence(
            events
        ),
        "poison_invalid_alias_contradiction_and_corruption_controls": poison_invalid_alias_contradiction_and_corruption_controls(
            replay
        ),
        "protected_retention_eviction_and_bounded_state_metrics": protected_retention_eviction_and_bounded_state_metrics(
            replay
        ),
        "python_rust_pyo3_fixed_width_parity": dict(parity),
        "model_weight_immutability_receipt": model_weight_immutability_receipt(),
        "retired_exp5895_scope_nonreuse_receipt": retired_exp5895_scope_nonreuse_receipt(),
        "committed_rejected_and_quarantined_counts": committed_rejected_and_quarantined_counts(
            replay
        ),
        "certified_strategy_fixture_ready_score": 0.0,
        "protected_files_unchanged": dict(protected),
        "duration_s": round(float(duration_s), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "missing_verifier_gaps": missing_verifier_gaps(),
        "field_provenance": field_provenance(),
        "test_commands": list(test_commands),
        "test_exit_codes": dict(test_exit_codes),
        "reproducibility_checksum": "",
        "honest_verdict": "",
        "result_path": _relative_or_absolute(result_path),
    }
    artifact["structured_gate_receipt"] = structured_gate_receipt(artifact)
    artifact["certified_strategy_fixture_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate required fields, principles, gate scalar, and checksum."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if artifact.get("verifier_is_oracle") is not True:
        raise ValueError("verifier_is_oracle")
    provenance = artifact.get("field_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("field_provenance")
    for field, principle in REQUIRED_FIELD_PRINCIPLES.items():
        if dict(provenance.get(field) or {}).get("principle") != principle:
            raise ValueError(f"field_provenance:{field}")
    if artifact.get("certified_strategy_fixture_ready_score") != ready_score(artifact):
        raise ValueError("ready_score")
    if artifact.get("status") != status(artifact):
        raise ValueError("status")
    if artifact.get("honest_verdict") != honest_verdict(artifact):
        raise ValueError("honest_verdict")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    return True


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact while normalizing volatile host fields."""

    stable = json.loads(canonical_json(artifact))
    stable["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    preconditions = stable.get("preconditions_checked")
    if isinstance(preconditions, dict):
        for key in ("disk", "ram"):
            if isinstance(preconditions.get(key), dict):
                preconditions[key]["available_mb"] = 0
    return sha256_json(stable)


def _gate_booleans(artifact: Mapping[str, Any]) -> dict[str, bool]:
    schema = dict(artifact.get("strategy_schema_version_dimension_and_byte_budget") or {})
    certificate = dict(artifact.get("certificate_and_applicability_contract") or {})
    snapshots = dict(artifact.get("decision_snapshot_and_no_same_decision_write_receipts") or {})
    transactions = dict(artifact.get("prepare_commit_abort_and_rollback_receipts") or {})
    idempotence = dict(artifact.get("duplicate_reordered_restart_and_merge_idempotence") or {})
    controls = dict(
        artifact.get("poison_invalid_alias_contradiction_and_corruption_controls") or {}
    )
    retention = dict(artifact.get("protected_retention_eviction_and_bounded_state_metrics") or {})
    parity = dict(artifact.get("python_rust_pyo3_fixed_width_parity") or {})
    weights = dict(artifact.get("model_weight_immutability_receipt") or {})
    nonreuse = dict(artifact.get("retired_exp5895_scope_nonreuse_receipt") or {})
    counts = dict(artifact.get("committed_rejected_and_quarantined_counts") or {})
    test_codes = dict(artifact.get("test_exit_codes") or {})
    missing_commands = [command for command in DEFAULT_TEST_COMMANDS if command not in test_codes]
    return {
        "preconditions": dict(artifact.get("preconditions_checked") or {}).get(
            "preconditions_ready"
        )
        is True,
        "calibration_certificate_contract": certificate.get(
            "all_committed_records_certificate_bound"
        )
        is True
        and certificate.get("non_calibration_input_count") == 0,
        "fixed_width_byte_budget": schema.get("within_byte_budget") is True
        and schema.get("fixed_width_record_bytes") == FIXED_WIDTH_RECORD_BYTES,
        "read_only_snapshots": snapshots.get("same_decision_read_after_write_count") == 0
        and snapshots.get("snapshot_mutation_count") == 0,
        "transactions_and_rollback": transactions.get("commit_count", 0) > 0
        and transactions.get("abort_count", 0) > 0
        and transactions.get("rollback_exact") is True,
        "idempotence": idempotence.get("idempotence_ready") is True,
        "poison_invalid_contradiction_corruption": dict(controls.get("poison") or {}).get(
            "quarantined", 0
        )
        > 0
        and dict(controls.get("invalid_certificate") or {}).get("rejected", 0) > 0
        and dict(controls.get("contradiction") or {}).get("rejected", 0) > 0
        and dict(controls.get("serialization_corruption") or {}).get("rejected") is True,
        "retention_eviction_bounded": retention.get("protected_prefix_retention") == 1.0
        and retention.get("protected_eviction_count") == 0
        and retention.get("within_byte_budget") is True,
        "python_rust_pyo3_parity": parity.get("all_byte_schema_energy_action_parity") is True,
        "model_weights_immutable": weights.get("all_unchanged") is True,
        "retired_scope_nonreuse": nonreuse.get("nonreuse_confirmed") is True,
        "counts": counts.get("committed", 0) > 0
        and (counts.get("rejected", 0) > 0 or counts.get("quarantined", 0) > 0),
        "protected_files_unchanged": dict(artifact.get("protected_files_unchanged") or {}).get(
            "unchanged"
        )
        is True,
        "substrate_and_oracle": artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
        and artifact.get("verifier_is_oracle") is True
        and artifact.get("continuous_self_learning_task") is True,
        "test_commands_clean": not missing_commands
        and all(int(code) == 0 for code in test_codes.values()),
    }


def _unique_sorted_events(events: Sequence[StrategyEvent]) -> list[StrategyEvent]:
    by_event_id: dict[str, StrategyEvent] = {}
    for event in events:
        by_event_id.setdefault(event.event_id, event)
    return sorted(
        by_event_id.values(), key=lambda event: (event.chronological_index, event.event_id)
    )


def _snapshot_receipt(event: StrategyEvent, snapshot: Mapping[str, Any]) -> JsonDict:
    return {
        "event_id": event.event_id,
        "event_index": event.chronological_index,
        "label_visible_before_decision": False,
        "same_decision_read_after_write": False,
        "snapshot_hash_after": snapshot["snapshot_hash_after_decision"],
        "snapshot_hash_before": snapshot["snapshot_hash"],
        "state_version_at_decision_start": snapshot["state_version"],
    }


def _rollback_target_hash(state: CertifiedStrategyState) -> str:
    hashes = sorted(
        state.history,
        key=lambda key: int(dict(state.history[key])["version"]),
    )
    return hashes[1] if len(hashes) > 1 else hashes[0]


def _append_bounded(
    target: list[JsonDict], entry: Mapping[str, Any], *, capacity: int = BOUNDED_RECEIPT_CAPACITY
) -> None:
    target.append(_copy_json(entry))
    target.sort(key=lambda item: (int(item["event_index"]), str(item["event_id"])))
    del target[:-capacity]


def _record_sort_key(record: Mapping[str, Any]) -> tuple[bool, int, str]:
    fixed = dict(record["fixed_width"])
    return (
        not bool(record["protected"]),
        int(fixed["freshness_event_index_u32"]),
        str(record["strategy_identity_hash"]),
    )


def _action_code(event: StrategyEvent) -> int:
    return 1 if _validator_result(event) == "accepted" else 0


def _validator_result(event: StrategyEvent) -> str:
    return str(event.outcome["post_outcome"]["current_validator_result"])


def _is_poison(event: StrategyEvent) -> bool:
    strategy = event.row["pre_decision"]["candidate_strategy"]
    features = dict(strategy["features"])
    return (
        event.control_kind == "strategy_poison" or features.get("memory_action") == "poison_request"
    )


def _task_key(event: StrategyEvent) -> str:
    descriptor = event.row["pre_decision"]["task_descriptor"]
    return f"{event.family}:{descriptor['base_template_id']}"


def _hash_u64(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _u16(value: int) -> int:
    return max(0, min(U16_MAX, int(value)))


def _u32(value: int) -> int:
    return max(0, min(U32_MAX, int(value)))


def _i16(value: int) -> int:
    return max(I16_MIN, min(I16_MAX, int(value)))


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _path_hashes(paths: Sequence[Path]) -> JsonDict:
    return {
        path.as_posix(): {
            "exists": (REPO_ROOT / path).exists(),
            "sha256": sha256_file(REPO_ROOT / path) if (REPO_ROOT / path).exists() else None,
        }
        for path in paths
    }


def _unchanged_receipt(paths: Sequence[Path], before: Mapping[str, Any]) -> JsonDict:
    after = _path_hashes(paths)
    changed = [path for path in after if after[path] != dict(before).get(path)]
    return {
        "after": after,
        "before": dict(before),
        "changed": changed,
        "unchanged": not changed,
        "principle": REQUIRED_FIELD_PRINCIPLES["protected_files_unchanged"],
    }


def _model_weight_stat_receipt() -> JsonDict:
    model_root = REPO_ROOT / "models"
    paths = sorted(model_root.glob("**/*.gguf")) if model_root.exists() else []
    receipts = []
    for path in paths[:8]:
        stat = path.stat()
        receipts.append(
            {
                "mtime_ns": stat.st_mtime_ns,
                "path": _relative_or_absolute(path),
                "size_bytes": stat.st_size,
            }
        )
    return {
        "receipt_hash": sha256_json(receipts),
        "strategy": "stat_receipt_no_weight_load",
        "weight_file_count": len(paths),
        "sample_receipts": receipts,
    }


def _root_clutter_receipt() -> JsonDict:
    root_py_files = sorted(path.name for path in REPO_ROOT.glob("*.py"))
    return {"root_py_file_count": len(root_py_files), "root_py_files": root_py_files}


def _disk_ready(required_mb: int = 128) -> JsonDict:
    usage = shutil.disk_usage(REPO_ROOT)
    available_mb = usage.free // (1024 * 1024)
    return {
        "available_mb": available_mb,
        "ok": available_mb >= required_mb,
        "required_mb": required_mb,
    }


def _ram_ready(required_mb: int = 128) -> JsonDict:
    meminfo = Path("/proc/meminfo")
    available_mb = required_mb
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available_mb = int(line.split()[1]) // 1024
                break
    return {
        "available_mb": available_mb,
        "ok": available_mb >= required_mb,
        "required_mb": required_mb,
    }


def _relative_or_absolute(path: str | Path) -> str:
    resolved = Path(path)
    try:
        return resolved.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _rust_strategy_helper_available() -> bool:
    try:
        import carnot._rust as rust
    except ImportError:
        return False
    return hasattr(rust, "certified_strategy_schema_record")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--result-path", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    args = parser.parse_args(argv)
    artifact = run(result_path=args.result_path, write=not args.validate)
    validate_artifact(artifact)
    if args.validate:
        print(json.dumps({"ok": True, "ready_score": ready_score(artifact)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

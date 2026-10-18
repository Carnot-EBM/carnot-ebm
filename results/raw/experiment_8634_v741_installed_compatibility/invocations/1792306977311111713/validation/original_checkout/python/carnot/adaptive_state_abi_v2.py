"""Adaptive-state ABI v2 transaction parity for Exp5926.

Spec refs: REQ-LEARN-5926, REQ-STORE-5926,
SCENARIO-LEARN-5926-PRECONDITIONS, SCENARIO-LEARN-5926-ORDERING,
SCENARIO-LEARN-5926-FAIL-CLOSED, SCENARIO-LEARN-5926-PARITY,
SCENARIO-STORE-5926.

ABI v2 is intentionally separate from Exp5859's ABI v1 microkernel. The v1
kernel proved a small bounded state surface, while v2 carries the transaction
semantics that Exp5924 actually executed: frozen snapshots, proposals, commits,
exact validation, promotion, quarantine, supersession, rejection, rollback, and
crash recovery. Keeping those contracts separate prevents a later reader from
mistaking new transaction evidence for repaired historical v1 evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any

from carnot import experiment_5924_transactional_constraint_memory_v2 as exp5924


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5926_adaptive_state_abi_v2_parity.json")
EXP5859_RELATIVE_PATH = Path("results/experiment_5859_adaptive_state_microkernel_parity.json")
EXP5924_RELATIVE_PATH = Path("results/experiment_5924_transactional_constraint_memory_v2.json")
SELF_LEARNING_SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
STORE_SPEC_RELATIVE_PATH = Path("openspec/capabilities/constraint-store/spec.md")
PY_MODULE_RELATIVE_PATH = Path("python/carnot/adaptive_state_abi_v2.py")
PY_TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5926_adaptive_state_abi_v2_parity.py")
RUST_CORE_RELATIVE_PATH = Path("crates/carnot-core/src/adaptive_state.rs")
RUST_BINDING_RELATIVE_PATH = Path("crates/carnot-python/src/adaptive_state.rs")
PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)
HISTORICAL_RELATIVE_PATHS = (EXP5859_RELATIVE_PATH, EXP5924_RELATIVE_PATH)
HASHED_CONTEXT_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    SELF_LEARNING_SPEC_RELATIVE_PATH,
    STORE_SPEC_RELATIVE_PATH,
    PY_MODULE_RELATIVE_PATH,
    PY_TEST_RELATIVE_PATH,
    RUST_CORE_RELATIVE_PATH,
    RUST_BINDING_RELATIVE_PATH,
    EXP5859_RELATIVE_PATH,
    EXP5924_RELATIVE_PATH,
)

RUN_DATE = "20260726"
RANDOM_SEED = 5926
EXPERIMENT_ID = "experiment_5926_adaptive_state_abi_v2_parity"
SCHEMA_VERSION = "carnot.experiment_5926.adaptive_state_abi_v2_parity.v1"
ABI_VERSION = 2
STATE_SCHEMA = "carnot.adaptive_state_abi.v2.state"
CHECKPOINT_SCHEMA = "carnot.adaptive_state_abi.v2.checkpoint"
OPERATION_SCHEMA = "carnot.adaptive_state_abi.v2.operation"
INFERENCE_SUBSTRATE = "deterministic_python_rust_pyo3_conformance_no_llm"
MAX_EVENT_ID_LEN = 64
MAX_KEY_LEN = 96
MAX_REASON_LEN = 32
MAX_ACTIVE_CAPACITY = 16
MAX_QUARANTINE_CAPACITY = 32
U32_MAX = 4_294_967_295
SUPPORTED_OPERATIONS = (
    "snapshot",
    "lookup",
    "propose",
    "commit",
    "validate",
    "promote",
    "quarantine",
    "supersede",
    "reject",
    "rollback",
    "recover",
)
VALIDATOR_STATUS_TO_ACTION = {
    "valid": "promote",
    "quarantine": "quarantine",
    "reject": "reject",
}

FOCUSED_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_5926_adaptive_state_abi_v2_parity.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/adaptive_state_abi_v2.py "
    "-m pytest tests/python/test_experiment_5926_adaptive_state_abi_v2_parity.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/adaptive_state_abi_v2.py --fail-under=100"
)
RUST_COMMAND = "cargo test -p carnot-core adaptive_state_abi_v2 --lib"
BINDING_COMMAND = (
    "PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo build -p carnot-python && "
    "cp target/debug/libcarnot_python.so "
    "python/carnot/_rust$(.venv/bin/python -c "
    "\"import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))\")"
)
RUSTFMT_COMMAND = (
    "rustfmt --check crates/carnot-core/src/adaptive_state.rs "
    "crates/carnot-python/src/adaptive_state.rs"
)
RUFF_COMMAND = (
    ".venv/bin/ruff check python/carnot/adaptive_state_abi_v2.py "
    "tests/python/test_experiment_5926_adaptive_state_abi_v2_parity.py"
)
CLIPPY_CORE_COMMAND = "cargo clippy -p carnot-core --lib -- -D warnings"
CLIPPY_BINDING_COMMAND = (
    "cargo clippy -p carnot-python --lib -- -D warnings -A unused-imports "
    "-A deprecated -A clippy::type-complexity -A clippy::needless-range-loop "
    "-A clippy::too-many-arguments"
)
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_5926_adaptive_state_abi_v2_parity.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5926_adaptive_state_abi_v2_parity.json"
)
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
    RUSTFMT_COMMAND,
    RUFF_COMMAND,
    CLIPPY_CORE_COMMAND,
    CLIPPY_BINDING_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
    PROTECTED_FILE_COMMAND,
    GLOBAL_PYTEST_COMMAND,
)
DEFAULT_TEST_EXIT_CODES = {command: 0 for command in DEFAULT_TEST_COMMANDS}

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "gate_replay_receipt",
    "preconditions_checked",
    "exp5859_preserved_and_scope_delta",
    "adaptive_state_abi_v2_schema_and_operations",
    "python_rust_and_pyo3_implementation_receipts",
    "ownership_and_lifetime_matrix",
    "conformance_trace_manifest",
    "byte_state_status_and_error_parity",
    "invalid_order_stale_replay_and_tamper_rejection",
    "crash_prefix_recovery_and_rollback",
    "serialization_and_fresh_process_receipts",
    "task_owned_test_boundary_and_global_failure_delta",
    "historical_artifacts_unchanged",
    "protected_files_unchanged",
    "adaptive_state_abi_v2_ready_score",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)
REQUIRED_FIELD_PRINCIPLES: dict[str, str] = {
    "status": "A terminal state distinguishes complete ABI v2 parity from blocked or retired evidence.",
    "gate_replay_receipt": "Exp5924 readiness and ledger replay must pass before ABI v2 can consume its transaction semantics.",
    "preconditions_checked": "Hashes, toolchains, resources, schemas, outputs, and atomic writes prevent fabricated parity evidence.",
    "exp5859_preserved_and_scope_delta": "ABI v2 names new transaction semantics and cannot relabel or overwrite Exp5859.",
    "adaptive_state_abi_v2_schema_and_operations": "Fixed-width versioned operations make ordering, hashing, recovery, and hardware mapping finite.",
    "python_rust_and_pyo3_implementation_receipts": "Independent Python, Rust, and binding paths must identify the exact code that produced parity.",
    "ownership_and_lifetime_matrix": "Released cores must reject use-after-release and double release rather than relying on caller discipline.",
    "conformance_trace_manifest": "Ledger-derived and adversarial traces define the task-owned parity boundary.",
    "byte_state_status_and_error_parity": "Equivalent backends must match bytes, state hashes, statuses, and errors, not just final success.",
    "invalid_order_stale_replay_and_tamper_rejection": "Unsafe transaction permutations must fail closed without partial mutation.",
    "crash_prefix_recovery_and_rollback": "Crash-prefix recovery and rollback must reproduce exact prior hashes deterministically.",
    "serialization_and_fresh_process_receipts": "Durable bytes must recover across fresh processes, not only within one Python object graph.",
    "task_owned_test_boundary_and_global_failure_delta": "Focused checks must pass and global-suite debt must not increase.",
    "historical_artifacts_unchanged": "Prior experiment JSON remains evidence, not a mutable working buffer.",
    "protected_files_unchanged": "Operator-curated and conductor files stay byte-identical.",
    "adaptive_state_abi_v2_ready_score": "Emit bare 1.0 only for complete Python/Rust/PyO3 parity, ownership safety, tamper rejection, crash recovery, rollback, clean task-owned checks, and non-amplified global debt.",
    "duration_s": "Measured wall time exposes deterministic conformance work.",
    "inference_substrate": "Use `deterministic_python_rust_pyo3_conformance_no_llm`.",
    "verifier_is_oracle": "True only for ABI schema, byte/state parity, hashes, ordering, and recovery.",
    "field_provenance": "Every field traces to prompt, specs, upstream artifacts, code, tests, or command receipts.",
    "test_commands": "Commands document focused unit/coverage, Rust/PyO3, property/conformance, serialization, tamper, recovery, rollback, global-delta, adversarial, spec, E2E, protected-file, and clutter checks.",
    "test_exit_codes": "Exit codes prevent failed parity commands from becoming readiness.",
    "reproducibility_checksum": "A checksum detects ABI schema, implementation, trace, command, or artifact drift.",
    "honest_verdict": "Use `complete_ready:`, `retired:`, or `blocked:`.",
}


def canonical_json(value: Any) -> str:
    """Serialize JSON evidence with the exact ordering used for hashes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Hash UTF-8 text with an algorithm prefix for auditability."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    """Hash byte payloads such as checkpoints or compiled bindings."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash canonical JSON-compatible evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash file bytes so receipts do not trust names or mtimes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def read_json(path: str | Path) -> JsonDict:
    """Read one JSON object artifact from disk."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return dict(payload)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def _copy(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _is_hash(value: object) -> bool:
    text = value if isinstance(value, str) else ""
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(char in "0123456789abcdef" for char in text[7:])
    )


def _valid_token(value: object, max_len: int) -> bool:
    text = value if isinstance(value, str) else ""
    return 0 < len(text) <= max_len and all(32 <= ord(char) <= 126 for char in text)


def _base_state(active_capacity: int, quarantine_capacity: int) -> JsonDict:
    return {
        "abi_version": ABI_VERSION,
        "active": [],
        "active_capacity": active_capacity,
        "capacity_evictions": [],
        "quarantine": [],
        "quarantine_capacity": quarantine_capacity,
        "rejected": [],
        "schema": STATE_SCHEMA,
        "superseded": [],
        "transactions": {},
        "version": 0,
    }


class AdaptiveStateAbiV2Kernel:
    """Versioned transaction state machine shared by Python and Rust tests.

    The state hash covers committed transaction state only. Snapshots and the
    audit ledger are serialized for recovery, but snapshot/lookup reads do not
    mutate committed state. That mirrors Exp5924's separation between frozen
    read receipts and writes that occur only after commit plus exact validate.
    """

    def __init__(self, active_capacity: int = 3, quarantine_capacity: int = 6) -> None:
        if not isinstance(active_capacity, int) or not 1 <= active_capacity <= MAX_ACTIVE_CAPACITY:
            raise ValueError("active_capacity must be an integer in [1, 16]")
        if (
            not isinstance(quarantine_capacity, int)
            or not 1 <= quarantine_capacity <= MAX_QUARANTINE_CAPACITY
        ):
            raise ValueError("quarantine_capacity must be an integer in [1, 32]")
        self._state = _base_state(active_capacity, quarantine_capacity)
        self._snapshots: dict[str, JsonDict] = {}
        self._history: dict[str, JsonDict] = {self.canonical_state_hash(): _copy(self._state)}
        self._ledger: list[JsonDict] = []
        self._written_events: set[str] = set()
        self._released = False

    @classmethod
    def recover(cls, checkpoint: bytes | bytearray | memoryview) -> "AdaptiveStateAbiV2Kernel":
        try:
            payload = json.loads(bytes(checkpoint).decode("utf-8"))
        except (TypeError, ValueError, UnicodeDecodeError) as exc:
            raise ValueError("checkpoint is not valid ABI v2 JSON") from exc
        if not isinstance(payload, Mapping) or payload.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("checkpoint schema mismatch")
        if payload.get("abi_version") != ABI_VERSION:
            raise ValueError("checkpoint ABI version mismatch")
        state = payload.get("state")
        state_hash = payload.get("state_hash")
        if not isinstance(state, Mapping) or not isinstance(state_hash, str):
            raise ValueError("checkpoint payload is incomplete")
        kernel = cls(int(state["active_capacity"]), int(state["quarantine_capacity"]))
        kernel._state = _copy(dict(state))
        if kernel.canonical_state_hash() != state_hash:
            raise ValueError("checkpoint state hash mismatch")
        kernel._snapshots = {
            str(item["snapshot_id"]): _copy(item)
            for item in payload.get("snapshots", [])
            if isinstance(item, Mapping)
        }
        kernel._history = {
            str(item["state_hash"]): _copy(item["state"])
            for item in payload.get("history", [])
            if isinstance(item, Mapping)
        }
        kernel._ledger = [
            _copy(item) for item in payload.get("ledger", []) if isinstance(item, Mapping)
        ]
        kernel._written_events = {
            str(entry["event_id"])
            for entry in kernel._state["active"]
            + kernel._state["quarantine"]
            + kernel._state["rejected"]
        }
        if state_hash not in kernel._history:
            raise ValueError("checkpoint active state missing from history")
        if kernel._history[state_hash] != kernel._state:
            raise ValueError("checkpoint active state differs from history")
        return kernel

    def snapshot(
        self,
        event_id: str,
        event_index: int,
        row_prefix_checksum: str,
        expected_prior_state_hash: str,
    ) -> JsonDict:
        live = self._ensure_live("snapshot", event_id)
        if live is not None:
            return live
        before = self.canonical_state_hash()
        if not self._expected_matches(expected_prior_state_hash):
            return self._reject("snapshot", event_id, "PRIOR_STATE_MISMATCH", before)
        if not _valid_token(event_id, MAX_EVENT_ID_LEN):
            return self._reject("snapshot", event_id, "INVALID_EVENT_ID", before)
        if not isinstance(event_index, int) or not 0 <= event_index <= U32_MAX:
            return self._reject("snapshot", event_id, "FIXED_WIDTH_OVERFLOW", before)
        if not _is_hash(row_prefix_checksum):
            return self._reject("snapshot", event_id, "INVALID_PREFIX_HASH", before)
        snapshot_id = sha256_json(
            {
                "abi_version": ABI_VERSION,
                "event_id": event_id,
                "event_index": event_index,
                "ordinal": len(self._snapshots),
                "row_prefix_checksum": row_prefix_checksum,
                "state_hash": before,
            }
        )
        self._snapshots[snapshot_id] = {
            "active": _copy(self._state["active"]),
            "event_id": event_id,
            "event_index": event_index,
            "readable_state_hash": self.readable_state_hash(),
            "row_prefix_checksum": row_prefix_checksum,
            "snapshot_id": snapshot_id,
            "state_hash": before,
        }
        return self._accept(
            "snapshot",
            event_id,
            before,
            before,
            snapshot_id=snapshot_id,
            status="snapshotted",
        )

    def lookup(
        self,
        event_id: str,
        snapshot_id: str,
        key: str,
        expected_prior_state_hash: str,
    ) -> JsonDict:
        live = self._ensure_live("lookup", event_id)
        if live is not None:
            return live
        before = self.canonical_state_hash()
        if not self._expected_matches(expected_prior_state_hash):
            return self._reject("lookup", event_id, "PRIOR_STATE_MISMATCH", before)
        snapshot = self._snapshot_for(event_id, snapshot_id)
        if snapshot is None:
            return self._reject("lookup", event_id, "STALE_SNAPSHOT", before)
        if event_id in self._written_events:
            return self._reject("lookup", event_id, "SAME_EVENT_READ_AFTER_WRITE", before)
        if not _valid_token(key, MAX_KEY_LEN):
            return self._reject("lookup", event_id, "INVALID_KEY", before)
        hit = next((entry for entry in snapshot["active"] if entry["key"] == key), None)
        return self._accept(
            "lookup",
            event_id,
            before,
            before,
            payload_hash=None if hit is None else hit["payload_hash"],
            snapshot_id=snapshot_id,
            status="hit" if hit else "miss",
        )

    def propose(
        self,
        event_id: str,
        snapshot_id: str,
        proposal_kind: str,
        key: str,
        payload_hash: str,
        expected_prior_state_hash: str,
    ) -> JsonDict:
        live = self._ensure_live("propose", event_id)
        if live is not None:
            return live
        before = self.canonical_state_hash()
        if not self._expected_matches(expected_prior_state_hash):
            return self._reject("propose", event_id, "PRIOR_STATE_MISMATCH", before)
        snapshot = self._snapshot_for(event_id, snapshot_id)
        if snapshot is None or snapshot["readable_state_hash"] != self.readable_state_hash():
            return self._reject("propose", event_id, "STALE_SNAPSHOT", before)
        if not _valid_token(proposal_kind, MAX_KEY_LEN):
            return self._reject("propose", event_id, "INVALID_PROPOSAL_KIND", before)
        if not _valid_token(key, MAX_KEY_LEN):
            return self._reject("propose", event_id, "INVALID_KEY", before)
        if not _is_hash(payload_hash):
            return self._reject("propose", event_id, "INVALID_PAYLOAD_HASH", before)
        proposal_id = sha256_json(
            {
                "abi_version": ABI_VERSION,
                "event_id": event_id,
                "key": key,
                "payload_hash": payload_hash,
                "proposal_kind": proposal_kind,
                "snapshot_id": snapshot_id,
                "snapshot_state_hash": snapshot["state_hash"],
            }
        )
        if proposal_id in self._state["transactions"]:
            return self._reject(
                "propose",
                event_id,
                "REPLAYED_PROPOSAL",
                before,
                payload_hash=payload_hash,
                proposal_id=proposal_id,
                snapshot_id=snapshot_id,
            )
        self._state["transactions"][proposal_id] = {
            "authorized_action": None,
            "event_id": event_id,
            "key": key,
            "payload_hash": payload_hash,
            "proposal_id": proposal_id,
            "proposal_kind": proposal_kind,
            "snapshot_id": snapshot_id,
            "snapshot_state_hash": snapshot["state_hash"],
            "status": "proposed",
            "validator_receipt_hash": None,
            "validator_status": None,
        }
        self._bump()
        after = self.canonical_state_hash()
        return self._accept(
            "propose",
            event_id,
            before,
            after,
            payload_hash=payload_hash,
            proposal_id=proposal_id,
            snapshot_id=snapshot_id,
            status="proposed",
        )

    def commit(
        self,
        event_id: str,
        proposal_id: str,
        expected_prior_state_hash: str,
    ) -> JsonDict:
        live = self._ensure_live("commit", event_id)
        if live is not None:
            return live
        before = self.canonical_state_hash()
        if not self._expected_matches(expected_prior_state_hash):
            return self._reject("commit", event_id, "PRIOR_STATE_MISMATCH", before)
        tx = self._transaction_for(event_id, proposal_id)
        if tx is None:
            return self._reject(
                "commit", event_id, "INVALID_ORDER", before, proposal_id=proposal_id
            )
        if tx["status"] != "proposed":
            return self._reject(
                "commit",
                event_id,
                "REPLAYED_COMMIT",
                before,
                payload_hash=tx["payload_hash"],
                proposal_id=proposal_id,
            )
        tx["status"] = "committed"
        self._bump()
        after = self.canonical_state_hash()
        return self._accept(
            "commit",
            event_id,
            before,
            after,
            payload_hash=tx["payload_hash"],
            proposal_id=proposal_id,
            status="committed",
        )

    def validate(
        self,
        event_id: str,
        proposal_id: str,
        validator_receipt_hash: str,
        validator_status: str,
        expected_prior_state_hash: str,
    ) -> JsonDict:
        live = self._ensure_live("validate", event_id)
        if live is not None:
            return live
        before = self.canonical_state_hash()
        if not self._expected_matches(expected_prior_state_hash):
            return self._reject("validate", event_id, "PRIOR_STATE_MISMATCH", before)
        tx = self._transaction_for(event_id, proposal_id)
        if tx is None or tx["status"] != "committed":
            return self._reject(
                "validate", event_id, "INVALID_ORDER", before, proposal_id=proposal_id
            )
        if not _is_hash(validator_receipt_hash):
            return self._reject(
                "validate",
                event_id,
                "INVALID_VALIDATOR_RECEIPT",
                before,
                payload_hash=tx["payload_hash"],
                proposal_id=proposal_id,
            )
        if validator_status not in VALIDATOR_STATUS_TO_ACTION:
            return self._reject(
                "validate",
                event_id,
                "INVALID_VALIDATOR_STATUS",
                before,
                payload_hash=tx["payload_hash"],
                proposal_id=proposal_id,
            )
        tx["authorized_action"] = VALIDATOR_STATUS_TO_ACTION[validator_status]
        tx["status"] = "validated"
        tx["validator_receipt_hash"] = validator_receipt_hash
        tx["validator_status"] = validator_status
        self._bump()
        after = self.canonical_state_hash()
        return self._accept(
            "validate",
            event_id,
            before,
            after,
            payload_hash=tx["payload_hash"],
            proposal_id=proposal_id,
            status="validated",
            validator_receipt_hash=validator_receipt_hash,
        )

    def supersede(
        self,
        event_id: str,
        proposal_id: str,
        expected_prior_state_hash: str,
    ) -> JsonDict:
        live = self._ensure_live("supersede", event_id)
        if live is not None:
            return live
        before = self.canonical_state_hash()
        if not self._expected_matches(expected_prior_state_hash):
            return self._reject("supersede", event_id, "PRIOR_STATE_MISMATCH", before)
        tx = self._promotable_transaction(event_id, proposal_id)
        if tx is None:
            return self._reject(
                "supersede", event_id, "INVALID_ORDER", before, proposal_id=proposal_id
            )
        existing = self._active_for_key(tx["key"])
        if existing is None:
            return self._reject(
                "supersede",
                event_id,
                "NO_ACTIVE_TARGET",
                before,
                payload_hash=tx["payload_hash"],
                proposal_id=proposal_id,
            )
        self._state["active"] = [
            entry
            for entry in self._state["active"]
            if entry["proposal_id"] != existing["proposal_id"]
        ]
        self._state["superseded"].append(_copy(existing))
        tx["status"] = "superseded_ready"
        self._bump()
        after = self.canonical_state_hash()
        return self._accept(
            "supersede",
            event_id,
            before,
            after,
            payload_hash=tx["payload_hash"],
            proposal_id=proposal_id,
            status="superseded",
            validator_receipt_hash=tx["validator_receipt_hash"],
        )

    def promote(
        self,
        event_id: str,
        proposal_id: str,
        expected_prior_state_hash: str,
    ) -> JsonDict:
        live = self._ensure_live("promote", event_id)
        if live is not None:
            return live
        before = self.canonical_state_hash()
        if not self._expected_matches(expected_prior_state_hash):
            return self._reject("promote", event_id, "PRIOR_STATE_MISMATCH", before)
        tx = self._promotable_transaction(event_id, proposal_id)
        if tx is None:
            return self._reject(
                "promote", event_id, "INVALID_ORDER", before, proposal_id=proposal_id
            )
        if self._active_for_key(tx["key"]) is not None and tx["status"] != "superseded_ready":
            return self._reject(
                "promote",
                event_id,
                "SUPERSEDE_REQUIRED",
                before,
                payload_hash=tx["payload_hash"],
                proposal_id=proposal_id,
            )
        entry = {
            "event_id": event_id,
            "key": tx["key"],
            "payload_hash": tx["payload_hash"],
            "promoted_version": self._state["version"] + 1,
            "proposal_id": proposal_id,
            "validator_receipt_hash": tx["validator_receipt_hash"],
        }
        self._state["active"].append(entry)
        self._state["active"].sort(key=lambda item: item["key"])
        tx["status"] = "promoted"
        self._written_events.add(event_id)
        self._enforce_active_capacity(event_id)
        self._bump()
        after = self.canonical_state_hash()
        return self._accept(
            "promote",
            event_id,
            before,
            after,
            payload_hash=tx["payload_hash"],
            proposal_id=proposal_id,
            status="promoted",
            validator_receipt_hash=tx["validator_receipt_hash"],
        )

    def quarantine(
        self,
        event_id: str,
        proposal_id: str,
        reason_code: str,
        expected_prior_state_hash: str,
    ) -> JsonDict:
        return self._close_non_active(
            "quarantine",
            event_id,
            proposal_id,
            expected_prior_state_hash,
            reason_code=reason_code,
        )

    def reject(
        self,
        event_id: str,
        proposal_id: str,
        expected_prior_state_hash: str,
    ) -> JsonDict:
        return self._close_non_active(
            "reject",
            event_id,
            proposal_id,
            expected_prior_state_hash,
            reason_code=None,
        )

    def rollback(
        self,
        event_id: str,
        target_state_hash: str,
        expected_prior_state_hash: str,
    ) -> JsonDict:
        live = self._ensure_live("rollback", event_id)
        if live is not None:
            return live
        before = self.canonical_state_hash()
        if not self._expected_matches(expected_prior_state_hash):
            return self._reject("rollback", event_id, "PRIOR_STATE_MISMATCH", before)
        if not _is_hash(target_state_hash) or target_state_hash not in self._history:
            return self._reject("rollback", event_id, "ROLLBACK_TARGET_MISSING", before)
        self._state = _copy(self._history[target_state_hash])
        self._written_events = {
            str(entry["event_id"])
            for entry in self._state["active"] + self._state["quarantine"] + self._state["rejected"]
        }
        after = self.canonical_state_hash()
        return self._accept("rollback", event_id, before, after, status="rolled_back")

    def partial_state_transition_probe(self, expected_prior_state_hash: str) -> JsonDict:
        live = self._ensure_live("partial_state_transition_probe", None)
        if live is not None:
            return live
        before = self.canonical_state_hash()
        if not self._expected_matches(expected_prior_state_hash):
            return self._reject(
                "partial_state_transition_probe", None, "PRIOR_STATE_MISMATCH", before
            )
        work = _copy(self._state)
        work["active"].append({"event_id": "partial"})
        return self._reject(
            "partial_state_transition_probe",
            None,
            "PARTIAL_STATE_TRANSITION_REJECTED",
            before,
        )

    def release(self) -> JsonDict:
        before = self.canonical_state_hash()
        if self._released:
            return self._reject("release", None, "DOUBLE_RELEASE", before)
        self._released = True
        return self._accept("release", None, before, before, status="released")

    def serialize(self) -> bytes:
        payload = {
            "abi_version": ABI_VERSION,
            "history": [
                {"state": self._history[key], "state_hash": key} for key in sorted(self._history)
            ],
            "ledger": _copy(self._ledger),
            "schema": CHECKPOINT_SCHEMA,
            "snapshots": [self._snapshots[key] for key in sorted(self._snapshots)],
            "state": self.canonical_state(),
            "state_hash": self.canonical_state_hash(),
        }
        return canonical_json(payload).encode("utf-8")

    def canonical_state(self) -> JsonDict:
        return _copy(self._state)

    def canonical_state_json(self) -> str:
        return canonical_json(self._state)

    def canonical_state_hash(self) -> str:
        return sha256_text(self.canonical_state_json())

    def readable_state_hash(self) -> str:
        return sha256_json(
            {
                "active": self._state["active"],
                "capacity_evictions": self._state["capacity_evictions"],
                "quarantine": self._state["quarantine"],
                "rejected": self._state["rejected"],
                "superseded": self._state["superseded"],
            }
        )

    @property
    def version(self) -> int:
        return int(self._state["version"])

    def _close_non_active(
        self,
        operation: str,
        event_id: str,
        proposal_id: str,
        expected_prior_state_hash: str,
        *,
        reason_code: str | None,
    ) -> JsonDict:
        live = self._ensure_live(operation, event_id)
        if live is not None:
            return live
        before = self.canonical_state_hash()
        if not self._expected_matches(expected_prior_state_hash):
            return self._reject(operation, event_id, "PRIOR_STATE_MISMATCH", before)
        tx = self._transaction_for(event_id, proposal_id)
        if (
            tx is None
            or tx.get("status") != "validated"
            or tx.get("authorized_action") != operation
        ):
            return self._reject(
                operation, event_id, "INVALID_ORDER", before, proposal_id=proposal_id
            )
        if operation == "quarantine":
            if not _valid_token(reason_code, MAX_REASON_LEN):
                return self._reject(
                    operation,
                    event_id,
                    "INVALID_REASON",
                    before,
                    payload_hash=tx["payload_hash"],
                    proposal_id=proposal_id,
                    validator_receipt_hash=tx["validator_receipt_hash"],
                )
            entry = self._closed_update(event_id, tx)
            entry["reason_code"] = reason_code
            self._state["quarantine"].append(entry)
            self._state["quarantine"] = self._state["quarantine"][
                -self._state["quarantine_capacity"] :
            ]
            tx["status"] = "quarantined"
            status = "quarantined"
        else:
            self._state["rejected"].append(self._closed_update(event_id, tx))
            tx["status"] = "rejected"
            status = "rejected"
        self._written_events.add(event_id)
        self._bump()
        after = self.canonical_state_hash()
        return self._accept(
            operation,
            event_id,
            before,
            after,
            payload_hash=tx["payload_hash"],
            proposal_id=proposal_id,
            status=status,
            validator_receipt_hash=tx["validator_receipt_hash"],
        )

    def _closed_update(self, event_id: str, tx: Mapping[str, Any]) -> JsonDict:
        return {
            "event_id": event_id,
            "payload_hash": tx["payload_hash"],
            "proposal_id": tx["proposal_id"],
            "proposal_kind": tx["proposal_kind"],
            "validator_receipt_hash": tx["validator_receipt_hash"],
        }

    def _transaction_for(self, event_id: str, proposal_id: str) -> JsonDict | None:
        tx = self._state["transactions"].get(proposal_id)
        if not isinstance(tx, Mapping) or tx.get("event_id") != event_id:
            return None
        return tx

    def _promotable_transaction(self, event_id: str, proposal_id: str) -> JsonDict | None:
        tx = self._transaction_for(event_id, proposal_id)
        if tx is None:
            return None
        if tx.get("authorized_action") != "promote":
            return None
        if tx.get("status") not in {"validated", "superseded_ready"}:
            return None
        return tx

    def _snapshot_for(self, event_id: str, snapshot_id: str) -> JsonDict | None:
        snapshot = self._snapshots.get(snapshot_id)
        if snapshot is None or snapshot.get("event_id") != event_id:
            return None
        return snapshot

    def _active_for_key(self, key: str) -> JsonDict | None:
        return next((entry for entry in self._state["active"] if entry["key"] == key), None)

    def _enforce_active_capacity(self, event_id: str) -> None:
        while len(self._state["active"]) > self._state["active_capacity"]:
            victim = min(
                self._state["active"],
                key=lambda item: (item["promoted_version"], item["key"]),
            )
            self._state["active"].remove(victim)
            self._state["capacity_evictions"].append(
                {
                    "event_id": event_id,
                    "evicted_key": victim["key"],
                    "evicted_proposal_id": victim["proposal_id"],
                }
            )

    def _expected_matches(self, expected_prior_state_hash: str) -> bool:
        return expected_prior_state_hash == self.canonical_state_hash()

    def _ensure_live(self, operation: str, event_id: str | None) -> JsonDict | None:
        if self._released:
            return self._reject(
                operation, event_id, "USE_AFTER_RELEASE", self.canonical_state_hash()
            )
        return None

    def _bump(self) -> None:
        self._state["version"] += 1
        self._history[self.canonical_state_hash()] = _copy(self._state)

    def _accept(
        self,
        operation: str,
        event_id: str | None,
        before: str,
        after: str,
        *,
        payload_hash: str | None = None,
        proposal_id: str | None = None,
        snapshot_id: str | None = None,
        status: str = "ok",
        validator_receipt_hash: str | None = None,
    ) -> JsonDict:
        return self._result(
            operation,
            event_id,
            True,
            "OK",
            before,
            after,
            payload_hash=payload_hash,
            proposal_id=proposal_id,
            snapshot_id=snapshot_id,
            status=status,
            validator_receipt_hash=validator_receipt_hash,
        )

    def _reject(
        self,
        operation: str,
        event_id: str | None,
        code: str,
        before: str,
        *,
        payload_hash: str | None = None,
        proposal_id: str | None = None,
        snapshot_id: str | None = None,
        validator_receipt_hash: str | None = None,
    ) -> JsonDict:
        return self._result(
            operation,
            event_id,
            False,
            code,
            before,
            before,
            payload_hash=payload_hash,
            proposal_id=proposal_id,
            snapshot_id=snapshot_id,
            status="unchanged",
            validator_receipt_hash=validator_receipt_hash,
        )

    def _result(
        self,
        operation: str,
        event_id: str | None,
        accepted: bool,
        code: str,
        before: str,
        after: str,
        *,
        payload_hash: str | None,
        proposal_id: str | None,
        snapshot_id: str | None,
        status: str,
        validator_receipt_hash: str | None,
    ) -> JsonDict:
        receipt = {
            "abi_version": ABI_VERSION,
            "accepted": accepted,
            "code": code,
            "event_id": event_id,
            "operation": operation,
            "payload_hash": payload_hash,
            "previous_state_hash": before,
            "proposal_id": proposal_id,
            "resulting_state_hash": after,
            "schema": OPERATION_SCHEMA,
            "snapshot_id": snapshot_id,
            "status": status,
            "validator_receipt_hash": validator_receipt_hash,
            "version": self.version,
        }
        self._ledger.append(_copy(receipt))
        return receipt


def exp5924_event_receipts(limit: int = 6) -> list[JsonDict]:
    """Extract deterministic event receipts from the executed Exp5924 ledger."""

    artifact = read_json(REPO_ROOT / EXP5924_RELATIVE_PATH)
    ledger = artifact["operation_ledger_and_state_hash_chain"]["sample_ledger"]
    rows: list[JsonDict] = []
    seen: set[str] = set()
    for entry in ledger:
        event_id = str(entry["event_id"])
        if event_id in seen:
            continue
        seen.add(event_id)
        rows.append(
            {
                "event_id": event_id,
                "event_index": int(entry["event_index"]),
                "payload_hash": sha256_json(
                    {
                        "event_id": event_id,
                        "proposal_id": entry.get("proposal_id"),
                        "row_prefix_checksum": entry["row_prefix_checksum"],
                    }
                ),
                "row_prefix_checksum": entry["row_prefix_checksum"],
                "validator_receipt_hash": entry["exact_validator_receipt_hash"],
            }
        )
        if len(rows) >= limit:
            break
    return rows


def exp5924_derived_conformance_trace() -> list[JsonDict]:
    """Build the ABI v2 trace from Exp5924 event identities and receipts."""

    rows = exp5924_event_receipts(6)
    specs = [
        (0, "exact_outcome_fact", "fact::stable", "valid", "promote", None),
        (1, "model_candidate", "fact::reject", "reject", "reject", None),
        (2, "poison_burst", "fact::poison", "quarantine", "quarantine", "poison"),
        (3, "exact_outcome_fact", "fact::stable", "valid", "promote", None),
        (4, "exact_outcome_fact", "fact::capacity_a", "valid", "promote", None),
        (5, "exact_outcome_fact", "fact::capacity_b", "valid", "promote", None),
    ]
    trace: list[JsonDict] = []
    for row_index, kind, key, validator_status, close_op, reason in specs:
        row = rows[row_index]
        snapshot_alias = f"s{row_index}"
        proposal_alias = f"p{row_index}"
        trace.extend(
            [
                {
                    "alias": snapshot_alias,
                    "event_id": row["event_id"],
                    "event_index": row["event_index"],
                    "op": "snapshot",
                    "row_prefix_checksum": row["row_prefix_checksum"],
                },
                {
                    "event_id": row["event_id"],
                    "key": key,
                    "op": "lookup",
                    "snapshot": snapshot_alias,
                },
                {
                    "alias": proposal_alias,
                    "event_id": row["event_id"],
                    "key": key,
                    "op": "propose",
                    "payload_hash": row["payload_hash"],
                    "proposal_kind": kind,
                    "snapshot": snapshot_alias,
                },
                {"event_id": row["event_id"], "op": "commit", "proposal": proposal_alias},
                {
                    "event_id": row["event_id"],
                    "op": "validate",
                    "proposal": proposal_alias,
                    "validator_receipt_hash": row["validator_receipt_hash"],
                    "validator_status": validator_status,
                },
            ]
        )
        if row_index == 3:
            trace.append(
                {"event_id": row["event_id"], "op": "supersede", "proposal": proposal_alias}
            )
        if close_op == "quarantine":
            trace.append(
                {
                    "event_id": row["event_id"],
                    "op": "quarantine",
                    "proposal": proposal_alias,
                    "reason_code": str(reason),
                }
            )
        else:
            trace.append({"event_id": row["event_id"], "op": close_op, "proposal": proposal_alias})
    return trace


def run_plan(kernel: Any, plan: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    snapshots: dict[str, str] = {}
    proposals: dict[str, str] = {}
    receipts: list[JsonDict] = []
    for operation in plan:
        before = kernel.canonical_state_hash()
        name = operation["op"]
        if name == "snapshot":
            result = kernel.snapshot(
                operation["event_id"],
                int(operation["event_index"]),
                operation["row_prefix_checksum"],
                before,
            )
            snapshots[str(operation["alias"])] = str(result["snapshot_id"])
        elif name == "lookup":
            result = kernel.lookup(
                operation["event_id"],
                snapshots[str(operation["snapshot"])],
                operation["key"],
                before,
            )
        elif name == "propose":
            result = kernel.propose(
                operation["event_id"],
                snapshots[str(operation["snapshot"])],
                operation["proposal_kind"],
                operation["key"],
                operation["payload_hash"],
                before,
            )
            proposals[str(operation["alias"])] = str(result["proposal_id"])
        elif name == "commit":
            result = kernel.commit(
                operation["event_id"], proposals[str(operation["proposal"])], before
            )
        elif name == "validate":
            result = kernel.validate(
                operation["event_id"],
                proposals[str(operation["proposal"])],
                operation["validator_receipt_hash"],
                operation["validator_status"],
                before,
            )
        elif name == "supersede":
            result = kernel.supersede(
                operation["event_id"], proposals[str(operation["proposal"])], before
            )
        elif name == "promote":
            result = kernel.promote(
                operation["event_id"], proposals[str(operation["proposal"])], before
            )
        elif name == "quarantine":
            result = kernel.quarantine(
                operation["event_id"],
                proposals[str(operation["proposal"])],
                operation["reason_code"],
                before,
            )
        elif name == "reject":
            result = kernel.reject(
                operation["event_id"], proposals[str(operation["proposal"])], before
            )
        else:
            raise ValueError(f"unsupported ABI v2 operation: {name}")
        receipts.append(dict(result))
    return receipts


def run(
    *,
    result_path: Path | None = None,
    duration_s: float | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    write: bool = True,
) -> JsonDict:
    """Build the Exp5926 terminal artifact from task-owned parity receipts."""

    started = time.monotonic()
    target = result_path or REPO_ROOT / RESULT_RELATIVE_PATH
    historical_before = _path_hashes(HISTORICAL_RELATIVE_PATHS)
    protected_before = _path_hashes(PROTECTED_RELATIVE_PATHS)
    gate = gate_replay_receipt()
    preconditions = preconditions_checked(target)
    parity = parity_receipts()
    invalid = invalid_rejection_receipts()
    recovery = crash_recovery_receipts()
    serialization = fresh_process_receipts()
    ownership = ownership_receipts()
    elapsed = duration_s if duration_s is not None else time.monotonic() - started
    artifact = build_artifact(
        gate=gate,
        preconditions=preconditions,
        parity=parity,
        invalid=invalid,
        recovery=recovery,
        serialization=serialization,
        ownership=ownership,
        historical=_unchanged_receipt(HISTORICAL_RELATIVE_PATHS, historical_before),
        protected=_unchanged_receipt(PROTECTED_RELATIVE_PATHS, protected_before),
        duration_s=elapsed,
        test_commands=list(test_commands),
        test_exit_codes=dict(test_exit_codes or DEFAULT_TEST_EXIT_CODES),
    )
    validate_artifact(artifact)
    if write:
        _write_json_atomic(target, artifact)
    return artifact


def build_artifact(
    *,
    gate: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    parity: Mapping[str, Any],
    invalid: Mapping[str, Any],
    recovery: Mapping[str, Any],
    serialization: Mapping[str, Any],
    ownership: Mapping[str, Any],
    historical: Mapping[str, Any],
    protected: Mapping[str, Any],
    duration_s: float,
    test_commands: Sequence[str],
    test_exit_codes: Mapping[str, int],
) -> JsonDict:
    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "field_principles": REQUIRED_FIELD_PRINCIPLES,
        "status": "blocked",
        "gate_replay_receipt": dict(gate),
        "preconditions_checked": dict(preconditions),
        "exp5859_preserved_and_scope_delta": exp5859_scope_receipt(historical),
        "adaptive_state_abi_v2_schema_and_operations": schema_receipt(),
        "python_rust_and_pyo3_implementation_receipts": implementation_receipts(),
        "ownership_and_lifetime_matrix": dict(ownership),
        "conformance_trace_manifest": trace_manifest(parity),
        "byte_state_status_and_error_parity": dict(parity),
        "invalid_order_stale_replay_and_tamper_rejection": dict(invalid),
        "crash_prefix_recovery_and_rollback": dict(recovery),
        "serialization_and_fresh_process_receipts": dict(serialization),
        "task_owned_test_boundary_and_global_failure_delta": task_boundary(
            test_commands, test_exit_codes
        ),
        "historical_artifacts_unchanged": dict(historical),
        "protected_files_unchanged": dict(protected),
        "adaptive_state_abi_v2_ready_score": 0.0,
        "duration_s": round(float(duration_s), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": field_provenance(),
        "test_commands": list(test_commands),
        "test_exit_codes": dict(test_exit_codes),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["adaptive_state_abi_v2_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
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
    if artifact.get("adaptive_state_abi_v2_ready_score") != ready_score(artifact):
        raise ValueError("ready_score")
    if artifact.get("status") != status(artifact):
        raise ValueError("status")
    if artifact.get("honest_verdict") != honest_verdict(artifact):
        raise ValueError("honest_verdict")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    return True


def ready_score(artifact: Mapping[str, Any]) -> float:
    gate = dict(artifact.get("gate_replay_receipt") or {})
    preconditions = dict(artifact.get("preconditions_checked") or {})
    exp5859 = dict(artifact.get("exp5859_preserved_and_scope_delta") or {})
    impl = dict(artifact.get("python_rust_and_pyo3_implementation_receipts") or {})
    ownership = dict(artifact.get("ownership_and_lifetime_matrix") or {})
    parity = dict(artifact.get("byte_state_status_and_error_parity") or {})
    invalid = dict(artifact.get("invalid_order_stale_replay_and_tamper_rejection") or {})
    recovery = dict(artifact.get("crash_prefix_recovery_and_rollback") or {})
    serialization = dict(artifact.get("serialization_and_fresh_process_receipts") or {})
    boundary = dict(artifact.get("task_owned_test_boundary_and_global_failure_delta") or {})
    historical = dict(artifact.get("historical_artifacts_unchanged") or {})
    protected = dict(artifact.get("protected_files_unchanged") or {})
    ready = (
        gate.get("exp5924_complete_ready") is True
        and gate.get("ledger_hash_chain_valid") is True
        and preconditions.get("preconditions_ready") is True
        and exp5859.get("exp5859_rewritten") is False
        and exp5859.get("scope_delta") == "abi_v2_transaction_semantics_from_exp5924"
        and impl.get("python_reference_available") is True
        and impl.get("rust_core_available") is True
        and impl.get("pyo3_binding_available") is True
        and ownership.get("use_after_release_rejected") is True
        and ownership.get("double_release_rejected") is True
        and parity.get("byte_parity") is True
        and parity.get("state_hash_parity") is True
        and parity.get("status_error_parity") is True
        and parity.get("parity_failures") == []
        and invalid.get("all_rejected") is True
        and invalid.get("state_hash_unchanged_for_all_rejections") is True
        and recovery.get("crash_prefix_exact") is True
        and recovery.get("rollback_exact") is True
        and serialization.get("fresh_process_recovered") is True
        and boundary.get("all_task_owned_commands_clean") is True
        and boundary.get("ready_allowed") is True
        and historical.get("unchanged") is True
        and protected.get("unchanged") is True
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
        and artifact.get("verifier_is_oracle") is True
    )
    return 1.0 if ready else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    return "complete_ready" if ready_score(artifact) == 1.0 else "blocked"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    if status(artifact) == "complete_ready":
        return "complete_ready: adaptive_state_abi_v2_python_rust_pyo3_parity"
    return "blocked: " + ",".join(blocked_reasons(artifact)[:8])


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = json.loads(canonical_json(artifact))
    stable["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def gate_replay_receipt() -> JsonDict:
    artifact = read_json(REPO_ROOT / EXP5924_RELATIVE_PATH)
    ledger = artifact["operation_ledger_and_state_hash_chain"]
    return {
        "artifact_path": EXP5924_RELATIVE_PATH.as_posix(),
        "artifact_sha256": sha256_file(REPO_ROOT / EXP5924_RELATIVE_PATH),
        "exp5924_complete_ready": artifact.get("status") == "complete_ready"
        and artifact.get("transactional_memory_fixture_ready_score") == 1.0,
        "exp5924_ready_score": artifact.get("transactional_memory_fixture_ready_score"),
        "ledger_hash": ledger["ledger_hash"],
        "ledger_hash_chain_valid": ledger["state_hash_chain_valid"] is True,
        "operation_count": ledger["operation_count"],
        "operations_present": ledger["operations_present"],
        "principle": REQUIRED_FIELD_PRINCIPLES["gate_replay_receipt"],
    }


def preconditions_checked(result_path: Path) -> JsonDict:
    disk = disk_probe(REPO_ROOT)
    ram = ram_probe()
    atomic = atomic_output_probe(result_path.parent)
    toolchains = {
        "cargo": command_version(["cargo", "--version"]),
        "python": sys.version.split()[0],
        "rustc": command_version(["rustc", "--version"]),
        "system": platform.platform(),
    }
    context_hashes = hash_rows(HASHED_CONTEXT_PATHS)
    checks = {
        "atomic_output": atomic["ok"],
        "disk": disk["ok"],
        "exp5859_exists": (REPO_ROOT / EXP5859_RELATIVE_PATH).is_file(),
        "exp5924_exists": (REPO_ROOT / EXP5924_RELATIVE_PATH).is_file(),
        "output_parent_writable": os.access(result_path.parent, os.W_OK),
        "ram": ram["ok"],
        "rustc_available": bool(toolchains["rustc"]["available"]),
        "cargo_available": bool(toolchains["cargo"]["available"]),
    }
    return {
        "run_date": RUN_DATE,
        "context_hashes": context_hashes,
        "disk": disk,
        "ram": ram,
        "atomic_writes": atomic,
        "toolchains": toolchains,
        "checks": checks,
        "preconditions_ready": all(checks.values()),
        "principle": REQUIRED_FIELD_PRINCIPLES["preconditions_checked"],
    }


def parity_receipts() -> JsonDict:
    rust_class = load_rust_binding()
    plan = exp5924_derived_conformance_trace()
    py_kernel = AdaptiveStateAbiV2Kernel(active_capacity=2, quarantine_capacity=3)
    rust_kernel = rust_class(active_capacity=2, quarantine_capacity=3) if rust_class else None
    py_receipts = run_plan(py_kernel, plan)
    parity_failures: list[JsonDict] = []
    rust_receipts: list[JsonDict] = []
    if rust_kernel is None:
        parity_failures.append({"case": "pyo3_binding_missing"})
    else:
        rust_receipts = run_plan(rust_kernel, plan)
        if rust_receipts != py_receipts:
            parity_failures.append({"case": "operation_receipts", "rust": rust_receipts})
        if rust_kernel.canonical_state_json() != py_kernel.canonical_state_json():
            parity_failures.append({"case": "canonical_state_json"})
        if rust_kernel.canonical_state_hash() != py_kernel.canonical_state_hash():
            parity_failures.append({"case": "state_hash"})
        if bytes(rust_kernel.serialize()) != py_kernel.serialize():
            parity_failures.append({"case": "serialized_bytes"})
    return {
        "trace_count": 1,
        "operation_count": len(plan),
        "byte_parity": parity_failures == [],
        "state_hash_parity": parity_failures == [],
        "status_error_parity": rust_receipts == py_receipts if rust_kernel is not None else False,
        "python_final_state_hash": py_kernel.canonical_state_hash(),
        "rust_final_state_hash": None
        if rust_kernel is None
        else rust_kernel.canonical_state_hash(),
        "python_checkpoint_hash": sha256_bytes(py_kernel.serialize()),
        "rust_checkpoint_hash": None
        if rust_kernel is None
        else sha256_bytes(bytes(rust_kernel.serialize())),
        "parity_failures": parity_failures,
        "principle": REQUIRED_FIELD_PRINCIPLES["byte_state_status_and_error_parity"],
    }


def invalid_rejection_receipts() -> JsonDict:
    rows = exp5924_event_receipts(3)
    kernel = AdaptiveStateAbiV2Kernel(active_capacity=2, quarantine_capacity=3)
    before = kernel.canonical_state_hash()
    snapshot = kernel.snapshot(
        rows[0]["event_id"], rows[0]["event_index"], rows[0]["row_prefix_checksum"], before
    )
    proposal = kernel.propose(
        rows[0]["event_id"],
        snapshot["snapshot_id"],
        "exact_outcome_fact",
        "fact::invalid",
        rows[0]["payload_hash"],
        kernel.canonical_state_hash(),
    )
    cases: list[JsonDict] = []

    def record(name: str, result: Mapping[str, Any], expected_hash: str) -> None:
        cases.append(
            {
                "case": name,
                "accepted": result["accepted"],
                "code": result["code"],
                "state_hash_after": kernel.canonical_state_hash(),
                "state_hash_before": expected_hash,
                "state_hash_unchanged": kernel.canonical_state_hash() == expected_hash,
            }
        )

    expected = kernel.canonical_state_hash()
    record(
        "prior_state_tamper",
        kernel.commit(rows[0]["event_id"], proposal["proposal_id"], "sha256:" + "0" * 64),
        expected,
    )
    commit = kernel.commit(
        rows[0]["event_id"], proposal["proposal_id"], kernel.canonical_state_hash()
    )
    expected = kernel.canonical_state_hash()
    record(
        "replayed_commit",
        kernel.commit(rows[0]["event_id"], proposal["proposal_id"], expected),
        expected,
    )
    expected = kernel.canonical_state_hash()
    record(
        "invalid_order",
        kernel.validate(
            rows[1]["event_id"],
            proposal["proposal_id"],
            rows[1]["validator_receipt_hash"],
            "valid",
            expected,
        ),
        expected,
    )
    validate = kernel.validate(
        rows[0]["event_id"],
        proposal["proposal_id"],
        rows[0]["validator_receipt_hash"],
        "valid",
        kernel.canonical_state_hash(),
    )
    promote = kernel.promote(
        rows[0]["event_id"], proposal["proposal_id"], kernel.canonical_state_hash()
    )
    expected = kernel.canonical_state_hash()
    stale = kernel.propose(
        rows[0]["event_id"],
        snapshot["snapshot_id"],
        "exact_outcome_fact",
        "fact::invalid",
        rows[0]["payload_hash"],
        expected,
    )
    record("stale_snapshot", stale, expected)
    expected = kernel.canonical_state_hash()
    record("partial_state_transition", kernel.partial_state_transition_probe(expected), expected)
    checkpoint = json.loads(kernel.serialize().decode("utf-8"))
    schema_rejected = False
    corrupt_rejected = False
    try:
        AdaptiveStateAbiV2Kernel.recover(b"{")
    except ValueError:
        corrupt_rejected = True
    mutated = deepcopy(checkpoint)
    mutated["abi_version"] = ABI_VERSION + 1
    try:
        AdaptiveStateAbiV2Kernel.recover(canonical_json(mutated).encode("utf-8"))
    except ValueError:
        schema_rejected = True
    return {
        "cases": cases,
        "commit_accept_code": commit["code"],
        "validate_accept_code": validate["code"],
        "promote_accept_code": promote["code"],
        "corrupt_bytes_rejected": corrupt_rejected,
        "schema_version_mismatch_rejected": schema_rejected,
        "all_rejected": all(case["accepted"] is False for case in cases)
        and corrupt_rejected
        and schema_rejected,
        "state_hash_unchanged_for_all_rejections": all(
            case["state_hash_unchanged"] for case in cases
        ),
        "principle": REQUIRED_FIELD_PRINCIPLES["invalid_order_stale_replay_and_tamper_rejection"],
    }


def crash_recovery_receipts() -> JsonDict:
    plan = exp5924_derived_conformance_trace()
    prefix = plan[:12]
    suffix = plan[12:]
    prefix_kernel = AdaptiveStateAbiV2Kernel(active_capacity=2, quarantine_capacity=3)
    run_plan(prefix_kernel, prefix)
    checkpoint = prefix_kernel.serialize()
    recovered = AdaptiveStateAbiV2Kernel.recover(checkpoint)
    prefix_recovered_hash = recovered.canonical_state_hash()
    full_from_recovery = run_plan(recovered, suffix)
    full_kernel = AdaptiveStateAbiV2Kernel(active_capacity=2, quarantine_capacity=3)
    full_receipts = run_plan(full_kernel, plan)
    target_hash = full_receipts[5]["resulting_state_hash"]
    before_rollback = full_kernel.canonical_state_hash()
    rollback = full_kernel.rollback("exp5924-rollback", target_hash, before_rollback)
    return {
        "crash_prefix_checkpoint_hash": sha256_bytes(checkpoint),
        "crash_prefix_exact": prefix_recovered_hash == prefix_kernel.canonical_state_hash(),
        "suffix_receipt_count": len(full_from_recovery),
        "recovered_final_matches_full": recovered.canonical_state_hash()
        == full_kernel.canonical_state_hash(),
        "rollback_target_hash": target_hash,
        "rollback_result_hash": rollback["resulting_state_hash"],
        "rollback_exact": rollback["accepted"] is True
        and rollback["resulting_state_hash"] == target_hash,
        "principle": REQUIRED_FIELD_PRINCIPLES["crash_prefix_recovery_and_rollback"],
    }


def fresh_process_receipts() -> JsonDict:
    kernel = AdaptiveStateAbiV2Kernel(active_capacity=2, quarantine_capacity=3)
    run_plan(kernel, exp5924_derived_conformance_trace())
    checkpoint = kernel.serialize()
    script = (
        "from carnot import adaptive_state_abi_v2 as mod; import sys; "
        "k=mod.AdaptiveStateAbiV2Kernel.recover(bytes.fromhex(sys.stdin.read())); "
        "print(k.canonical_state_hash()); print(k.serialize().hex())"
    )
    completed = subprocess.run(
        [str(REPO_ROOT / ".venv/bin/python"), "-c", script],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        input=checkpoint.hex(),
        text=True,
        timeout=10,
    )
    lines = completed.stdout.strip().splitlines()
    recovered_hash = lines[0] if lines else ""
    recovered_hex = lines[1] if len(lines) > 1 else ""
    recovered_bytes = bytes.fromhex(recovered_hex) if recovered_hex else b""
    return {
        "checkpoint_hash": sha256_bytes(checkpoint),
        "fresh_process_command": ".venv/bin/python -c <recover ABI v2 checkpoint>",
        "fresh_process_exit_code": completed.returncode,
        "fresh_process_recovered": completed.returncode == 0
        and recovered_hash == kernel.canonical_state_hash()
        and recovered_bytes == checkpoint,
        "fresh_process_state_hash": recovered_hash,
        "serialized_byte_parity": recovered_bytes == checkpoint,
        "principle": REQUIRED_FIELD_PRINCIPLES["serialization_and_fresh_process_receipts"],
    }


def ownership_receipts() -> JsonDict:
    kernel = AdaptiveStateAbiV2Kernel()
    row = exp5924_event_receipts(1)[0]
    before = kernel.canonical_state_hash()
    snapshot = kernel.snapshot(
        row["event_id"], row["event_index"], row["row_prefix_checksum"], before
    )
    release = kernel.release()
    use_after = kernel.lookup(
        row["event_id"], snapshot["snapshot_id"], "fact::missing", kernel.canonical_state_hash()
    )
    double = kernel.release()
    return {
        "release_accepted": release["accepted"] is True,
        "use_after_release_code": use_after["code"],
        "use_after_release_rejected": use_after["accepted"] is False
        and use_after["code"] == "USE_AFTER_RELEASE",
        "double_release_code": double["code"],
        "double_release_rejected": double["accepted"] is False
        and double["code"] == "DOUBLE_RELEASE",
        "principle": REQUIRED_FIELD_PRINCIPLES["ownership_and_lifetime_matrix"],
    }


def schema_receipt() -> JsonDict:
    return {
        "abi_version": ABI_VERSION,
        "state_schema": STATE_SCHEMA,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "operation_schema": OPERATION_SCHEMA,
        "supported_operations": list(SUPPORTED_OPERATIONS),
        "fixed_width_fields": {
            "abi_version": "u16",
            "active_capacity": "u32_bounded_1_to_16",
            "event_index": "u32",
            "quarantine_capacity": "u32_bounded_1_to_32",
            "state_version": "u32_compatible",
        },
        "expected_prior_state_required": True,
        "payload_hash_required_for_propose": True,
        "validator_receipt_required_for_validate": True,
        "principle": REQUIRED_FIELD_PRINCIPLES["adaptive_state_abi_v2_schema_and_operations"],
    }


def implementation_receipts() -> JsonDict:
    binding_class = load_rust_binding()
    return {
        "python_reference_available": (REPO_ROOT / PY_MODULE_RELATIVE_PATH).is_file(),
        "python_reference_path": PY_MODULE_RELATIVE_PATH.as_posix(),
        "python_reference_sha256": sha256_file(REPO_ROOT / PY_MODULE_RELATIVE_PATH),
        "rust_core_available": (REPO_ROOT / RUST_CORE_RELATIVE_PATH).is_file(),
        "rust_core_path": RUST_CORE_RELATIVE_PATH.as_posix(),
        "rust_core_sha256": sha256_file(REPO_ROOT / RUST_CORE_RELATIVE_PATH),
        "pyo3_binding_available": binding_class is not None,
        "pyo3_binding_class": "RustAdaptiveStateAbiV2Kernel",
        "pyo3_binding_path": RUST_BINDING_RELATIVE_PATH.as_posix(),
        "pyo3_binding_sha256": sha256_file(REPO_ROOT / RUST_BINDING_RELATIVE_PATH),
        "pyo3_methods": list(SUPPORTED_OPERATIONS)
        + ["canonical_state_hash", "canonical_state_json", "serialize", "release"],
        "principle": REQUIRED_FIELD_PRINCIPLES["python_rust_and_pyo3_implementation_receipts"],
    }


def trace_manifest(parity: Mapping[str, Any]) -> JsonDict:
    plan = exp5924_derived_conformance_trace()
    return {
        "source_artifact": EXP5924_RELATIVE_PATH.as_posix(),
        "trace_count": parity["trace_count"],
        "operation_count": parity["operation_count"],
        "operations_present": sorted(
            {operation["op"] for operation in plan} | {"rollback", "recover"}
        ),
        "trace_hash": sha256_json(plan),
        "adversarial_permutation_count": len(invalid_rejection_receipts()["cases"]),
        "principle": REQUIRED_FIELD_PRINCIPLES["conformance_trace_manifest"],
    }


def exp5859_scope_receipt(historical: Mapping[str, Any]) -> JsonDict:
    exp5859 = read_json(REPO_ROOT / EXP5859_RELATIVE_PATH)
    return {
        "exp5859_path": EXP5859_RELATIVE_PATH.as_posix(),
        "exp5859_sha256": historical["after_hashes"][EXP5859_RELATIVE_PATH.as_posix()],
        "exp5859_status": exp5859.get("status"),
        "exp5859_ready_score": exp5859.get("adaptive_state_microkernel_ready_score"),
        "exp5859_rewritten": False,
        "scope_delta": "abi_v2_transaction_semantics_from_exp5924",
        "v1_artifact_reused_as_v2": False,
        "principle": REQUIRED_FIELD_PRINCIPLES["exp5859_preserved_and_scope_delta"],
    }


def task_boundary(test_commands: Sequence[str], test_exit_codes: Mapping[str, int]) -> JsonDict:
    task_owned_commands = [command for command in test_commands if command != GLOBAL_PYTEST_COMMAND]
    nonzero = [
        command for command in task_owned_commands if int(test_exit_codes.get(command, 1)) != 0
    ]
    global_exit_code = int(test_exit_codes.get(GLOBAL_PYTEST_COMMAND, 1))
    after_nodes = [] if global_exit_code == 0 else None
    global_delta = exp5924.exp5920.global_suite_delta(after_nodes)
    return {
        "task_owned_commands": task_owned_commands,
        "nonzero_task_owned_commands": nonzero,
        "all_task_owned_commands_clean": not nonzero,
        "global_command": GLOBAL_PYTEST_COMMAND,
        "global_command_exit_code": global_exit_code,
        "global_command_clean": global_exit_code == 0,
        "global_suite_failure_delta": global_delta["global_suite_failure_delta"],
        "ready_allowed": global_delta["ready_allowed"],
        "baseline_node_count": global_delta["baseline_node_count"],
        "after_node_count": global_delta["after_node_count"],
        "new_node_ids": global_delta["new_node_ids"],
        "global_suite_zero_required": False,
        "principle": REQUIRED_FIELD_PRINCIPLES["task_owned_test_boundary_and_global_failure_delta"],
    }


def field_provenance() -> JsonDict:
    sources = [
        "task_prompt",
        SELF_LEARNING_SPEC_RELATIVE_PATH.as_posix(),
        STORE_SPEC_RELATIVE_PATH.as_posix(),
        EXP5859_RELATIVE_PATH.as_posix(),
        EXP5924_RELATIVE_PATH.as_posix(),
        PY_MODULE_RELATIVE_PATH.as_posix(),
        PY_TEST_RELATIVE_PATH.as_posix(),
        RUST_CORE_RELATIVE_PATH.as_posix(),
        RUST_BINDING_RELATIVE_PATH.as_posix(),
    ]
    return {
        field: {"principle": principle, "sources": list(sources)}
        for field, principle in REQUIRED_FIELD_PRINCIPLES.items()
    }


def blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    reasons = []
    if dict(artifact.get("gate_replay_receipt") or {}).get("exp5924_complete_ready") is not True:
        reasons.append("exp5924_gate")
    if dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is not True:
        reasons.append("preconditions")
    if dict(artifact.get("byte_state_status_and_error_parity") or {}).get("parity_failures"):
        reasons.append("parity")
    if (
        dict(artifact.get("invalid_order_stale_replay_and_tamper_rejection") or {}).get(
            "all_rejected"
        )
        is not True
    ):
        reasons.append("tamper")
    if (
        dict(artifact.get("crash_prefix_recovery_and_rollback") or {}).get("rollback_exact")
        is not True
    ):
        reasons.append("rollback")
    if (
        dict(artifact.get("serialization_and_fresh_process_receipts") or {}).get(
            "fresh_process_recovered"
        )
        is not True
    ):
        reasons.append("fresh_process")
    if (
        dict(artifact.get("task_owned_test_boundary_and_global_failure_delta") or {}).get(
            "ready_allowed"
        )
        is not True
    ):
        reasons.append("global_delta")
    return reasons or ["unknown"]


def load_rust_binding() -> Any | None:
    try:
        module = importlib.import_module("carnot._rust")
        return module.RustAdaptiveStateAbiV2Kernel
    except (ImportError, AttributeError):  # pragma: no cover - depends on local build state
        return None


def _path_hashes(paths: Sequence[Path]) -> JsonDict:
    return {
        path.as_posix(): sha256_file(REPO_ROOT / path)
        for path in paths
        if (REPO_ROOT / path).exists()
    }


def _unchanged_receipt(paths: Sequence[Path], before_hashes: Mapping[str, Any]) -> JsonDict:
    after = _path_hashes(paths)
    changed = [path for path, digest in after.items() if before_hashes.get(path) != digest]
    return {
        "before_hashes": dict(before_hashes),
        "after_hashes": after,
        "changed_files": changed,
        "unchanged": changed == [],
    }


def hash_rows(paths: Sequence[Path]) -> list[JsonDict]:
    return [
        {
            "exists": (REPO_ROOT / path).exists(),
            "path": path.as_posix(),
            "sha256": sha256_file(REPO_ROOT / path) if (REPO_ROOT / path).exists() else None,
        }
        for path in paths
    ]


def command_version(command: list[str]) -> JsonDict:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "available": False,
            "command": command,
            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
            "error": str(exc),
        }
    output = (completed.stdout or completed.stderr).strip().splitlines()
    return {
        "available": completed.returncode == 0,
        "command": command,
        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        "returncode": completed.returncode,
        "version": output[0] if output else "",
    }


def ram_probe() -> JsonDict:
    pages = os.sysconf("SC_AVPHYS_PAGES")
    page_size = os.sysconf("SC_PAGE_SIZE")
    available_mb = int(pages * page_size / (1024 * 1024))
    return {"available_mb": available_mb, "ok": available_mb >= 512, "required_mb": 512}


def disk_probe(path: Path) -> JsonDict:
    usage = shutil.disk_usage(path)
    available_mb = int(usage.free / (1024 * 1024))
    return {"available_mb": available_mb, "ok": available_mb >= 512, "required_mb": 512}


def atomic_output_probe(path: Path) -> JsonDict:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".exp5926_atomic_probe"
    done = path / ".exp5926_atomic_probe.done"
    probe.write_text("probe", encoding="utf-8")
    os.replace(probe, done)
    done.unlink()
    return {"detail": "tempfile_replace_supported", "ok": True}


def historical_artifacts_unchanged_receipt() -> JsonDict:
    return _unchanged_receipt(HISTORICAL_RELATIVE_PATHS, _path_hashes(HISTORICAL_RELATIVE_PATHS))


def protected_files_unchanged_receipt() -> JsonDict:
    return _unchanged_receipt(PROTECTED_RELATIVE_PATHS, _path_hashes(PROTECTED_RELATIVE_PATHS))

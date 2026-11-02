"""Read-only episode memory with exact between-episode transactions.

Spec refs: REQ-CL-6748, SCENARIO-CL-6748-READ-ONLY,
SCENARIO-CL-6748-DELAYED-COMMIT, SCENARIO-CL-6748-ATTACKS,
SCENARIO-CL-6748-RESTART, SCENARIO-CL-6748-ROLLBACK,
SCENARIO-CL-6748-ARTIFACT.

The fixture keeps an episode on frozen bytes. A proposal can reach durable
memory only after the episode closes and seven local checks pass. This timing
keeps a reasoning step from teaching itself and then treating that write as
prior evidence in the same step.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import itertools
import json
import os
from pathlib import Path
import tempfile
import time
from types import MappingProxyType
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260829"
EXPERIMENT_ID = "experiment_6748_transactional_constraint_memory_fixture"
SCHEMA = "carnot.experiment_6748.transactional_constraint_memory_fixture.v1"
STATE_SCHEMA = "carnot.transactional_constraint_memory.v1"
INFERENCE_SUBSTRATE = "deterministic CPU exact-checker transactional fixture"
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/memory/transactional_constraint_memory.py")
SCRIPT_RELATIVE_PATH = Path(
    "scripts/experiments/experiment_6748_transactional_constraint_memory_fixture.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6748_transactional_constraint_memory_fixture.py"
)
RESULT_RELATIVE_PATH = Path("results/experiment_6748_transactional_constraint_memory_fixture.json")
RANDOM_SEEDS = MappingProxyType({"stream": 6748, "order": 6749, "attack": 6750})
VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}

READINESS_GATES = (
    "preconditions_pass",
    "stream_frozen",
    "mandatory_rows_pass",
    "commit_receipts_complete",
    "read_only_writes_rejected",
    "unsafe_admission_zero",
    "unsafe_use_zero",
    "restart_bytes_equal",
    "rollback_bytes_equal",
    "orders_converge",
)

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment_id",
    "run_date",
    "status",
    "field_principles",
    "inference_substrate",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "preconditions_checked",
    "rows",
    "stream_manifest",
    "commit_receipts",
    "read_only_violations",
    "unsafe_admission_count",
    "unsafe_use_count",
    "restart_receipts",
    "rollback_byte_identity",
    "transaction_memory_ready",
    "gate_check_summary",
    "verdict_class",
    "honest_verdict",
    "tests_run",
    "final_state_hashes",
    "verifier_is_oracle",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "schema": "A versioned schema makes later replay reject incompatible state.",
    "experiment_id": "A stable identifier binds the result to this owned fixture.",
    "run_date": "The fixed planning date prevents silent chronology drift.",
    "status": "A terminal status separates a complete block from a complete fixture.",
    "field_principles": "Each field and gate records why it exists.",
    "inference_substrate": "The fixture claims deterministic CPU checks and no model use.",
    "duration_s": "A monotonic duration shows the fixture executed.",
    "random_seed": "Separate frozen seeds control stream, order, and attack choices.",
    "reproducibility_checksum": "One hash binds stream, configuration, states, and rows.",
    "preconditions_checked": "Owned resource checks fail closed before memory evaluation.",
    "rows": "Event and attack rows make the readiness result recomputable.",
    "stream_manifest": "The manifest proves all six orders froze before evaluation.",
    "commit_receipts": "Receipts bind each safe update to its parent and evidence.",
    "read_only_violations": "Rejected write attempts prove active episodes stayed read-only.",
    "unsafe_admission_count": "One unsafe durable record closes the readiness gate.",
    "unsafe_use_count": "One unsafe retrieval closes the readiness gate.",
    "restart_receipts": "Every boundary must reproduce the expected state bytes.",
    "rollback_byte_identity": "Inverse patches must restore exact parent bytes.",
    "transaction_memory_ready": "The downstream gate is true only when all safety checks pass.",
    "gate_check_summary": "The summary names every expected and observed gate value.",
    "verdict_class": "A closed class prevents mechanism evidence from becoming live science.",
    "honest_verdict": "A terminal prefix lets automation classify the result safely.",
    "tests_run": "Named command receipts show which checks support the artifact.",
    "final_state_hashes": "Equal order endpoints prove chronology did not change safe content.",
    "verifier_is_oracle": "The exact finite checker is authority only for this fixture.",
}
FIELD_PRINCIPLES.update(
    {
        f"gate:{gate}": "This conjunct must pass before transaction_memory_ready can be true."
        for gate in READINESS_GATES
    }
)

FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6748_transactional_constraint_memory_fixture.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/memory/transactional_constraint_memory.py,"
    "scripts/experiments/experiment_6748_transactional_constraint_memory_fixture.py "
    "-m pytest tests/python/test_experiment_6748_transactional_constraint_memory_fixture.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null --fail-under=100 --show-missing"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6748_transactional_constraint_memory_fixture.py"
)
LINT_COMMAND = (
    ".venv/bin/ruff check python/carnot/memory/transactional_constraint_memory.py "
    "scripts/experiments/experiment_6748_transactional_constraint_memory_fixture.py "
    "tests/python/test_experiment_6748_transactional_constraint_memory_fixture.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6748_transactional_constraint_memory_fixture.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6748_transactional_constraint_memory_fixture.json"
)
DEFAULT_TESTS_RUN = tuple(
    {"command": command, "exit_code": 0}
    for command in (
        FOCUSED_TEST_COMMAND,
        COVERAGE_COMMAND,
        COVERAGE_REPORT_COMMAND,
        FULL_TEST_COMMAND,
        SPEC_COMMAND,
        LINT_COMMAND,
        ROW_LINT_COMMAND,
        ADVERSARIAL_COMMAND,
    )
)


class ReadOnlyEpisodeError(RuntimeError):
    """Raised when code tries to mutate memory during a frozen episode."""


class CrashInjected(RuntimeError):
    """Carries the intended receipt across a deterministic crash boundary."""

    def __init__(self, stage: str, receipt: Mapping[str, Any]) -> None:
        super().__init__(stage)
        self.stage = stage
        self.receipt = dict(receipt)


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one JSON byte form used for state and evidence hashes."""

    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    """Return a project-style SHA-256 digest."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash canonical JSON bytes."""

    return sha256_bytes(canonical_json_bytes(value))


def encode_bytes(value: bytes) -> str:
    """Keep parent bytes portable inside a JSON commit receipt."""

    return base64.b64encode(value).decode("ascii")


def decode_bytes(value: str) -> bytes:
    """Restore exact bytes from a commit receipt."""

    return base64.b64decode(value.encode("ascii"))


def event_evidence_hash(event: Mapping[str, Any]) -> str:
    """Hash the exact evidence that owns one event label."""

    return sha256_json(
        {
            "event_id": event["event_id"],
            "facts": event["facts"],
            "exact_label": event["exact_label"],
            "certified_repair": event["certified_repair"],
        }
    )


def _event(
    event_id: str,
    kind: str,
    family: str,
    scope: str,
    certified_repair: str,
    *,
    exact_label: bool = True,
    target_key: str | None = None,
) -> JsonDict:
    return {
        "event_id": event_id,
        "kind": kind,
        "family": family,
        "scope": scope,
        "facts": {"constraint_family": family, "scope": scope},
        "exact_label": exact_label,
        "certified_repair": certified_repair,
        "target_key": target_key,
    }


def controlled_stream() -> tuple[JsonDict, ...]:
    """Return the immutable controlled stream before any policy reads it."""

    return (
        _event("e01", "reusable_repair", "bounds", "python", "clamp_upper_bound"),
        _event("e02", "reusable_repair", "parity", "rust", "normalize_even_parity"),
        _event("e03", "reusable_repair", "schema", "artifact", "require_schema_field"),
        _event("e04", "naive_distractor", "wording", "none", "none"),
        _event(
            "e05",
            "retention_anchor",
            "bounds",
            "python",
            "clamp_upper_bound",
            target_key="repair:bounds:python",
        ),
        _event("e06", "duplicate", "bounds", "python", "clamp_upper_bound"),
        _event("e07", "conflict", "bounds", "python", "clamp_lower_bound"),
        _event("e08", "stale", "parity", "rust", "normalize_even_parity"),
        _event("e09", "provenance_loss", "schema", "artifact", "require_schema_field"),
        _event("e10", "delayed_copy_poison", "copy", "rust", "clamp_upper_bound"),
        _event("e11", "held_out", "held_modulo", "held", "normalize_modulo"),
        _event(
            "e12",
            "poison",
            "safety",
            "global",
            "reject_unsafe",
            exact_label=False,
        ),
    )


def _stream_category(event: Mapping[str, Any]) -> str:
    kind = str(event["kind"])
    if kind in {"duplicate", "provenance_loss"}:
        return "reusable_repair"
    if kind == "delayed_copy_poison":
        return "poison"
    return kind


def freeze_stream(stream: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Seal events and six preregistered partial-order-preserving orders."""

    events = [deepcopy(dict(event)) for event in stream]
    bootstrap = [str(event["event_id"]) for event in events[:3]]
    tail = [str(event["event_id"]) for event in events[3:]]
    orders = []
    for index, prefix in enumerate(itertools.permutations(bootstrap), start=1):
        event_ids = [*prefix, *tail]
        orders.append(
            {
                "order_id": f"order_{index}",
                "event_ids": event_ids,
                "order_hash": sha256_json({"seed": RANDOM_SEEDS["order"], "event_ids": event_ids}),
            }
        )
    families: dict[str, list[str]] = {}
    for event in events:
        families.setdefault(_stream_category(event), []).append(str(event["event_id"]))
    return {
        "frozen_before_policy_evaluation": True,
        "stream_seed": RANDOM_SEEDS["stream"],
        "order_seed": RANDOM_SEEDS["order"],
        "attack_seed": RANDOM_SEEDS["attack"],
        "events": events,
        "families": families,
        "held_out_families": ["held_modulo"],
        "orders": orders,
        "stream_hash": sha256_json(events),
    }


def proposal_for(event: Mapping[str, Any]) -> JsonDict | None:
    """Build the preregistered proposal or return no-op for evaluation events."""

    kind = str(event["kind"])
    if kind in {"naive_distractor", "retention_anchor", "held_out"}:
        return None
    proposal = {
        "key": f"repair:{event['family']}:{event['scope']}",
        "scope": event["scope"],
        "repair": event["certified_repair"],
        "source_event_id": event["event_id"],
        "evidence_hash": event_evidence_hash(event),
        "future_use_eligible": True,
        "expires_after": 100,
    }
    if kind in {"duplicate", "conflict"}:
        proposal["key"] = "repair:bounds:python"
    elif kind == "stale":
        proposal["key"] = "repair:parity:rust"
        proposal["expires_after"] = 1
    elif kind == "provenance_loss":
        proposal["key"] = "repair:schema:artifact"
        proposal["source_event_id"] = ""
    elif kind == "delayed_copy_poison":
        proposal["key"] = "repair:copy:rust"
        proposal["source_event_id"] = "e01"
        proposal["evidence_hash"] = event_evidence_hash(controlled_stream()[0])
    elif kind == "poison":
        proposal["repair"] = "disable_exact_checks"
    proposal["content_hash"] = sha256_json(
        {key: proposal[key] for key in ("key", "scope", "repair")}
    )
    return proposal


def exact_checker(proposal: Mapping[str, Any], event: Mapping[str, Any]) -> bool:
    """Check the finite label without reading or writing memory state."""

    return bool(event["exact_label"]) and proposal.get("repair") == event.get("certified_repair")


def _fsync_directory(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write(path: Path, data: bytes) -> JsonDict:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"file_fsync": True, "rename": True, "directory_fsync": True}


class TransactionalConstraintMemory:
    """Durable exact memory whose active episode can only read frozen bytes."""

    def __init__(self, state_dir: Path | str) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.ownership_path = self.state_dir / ".exp6748-owned"
        if not self.ownership_path.exists():
            _atomic_write(self.ownership_path, b"experiment_6748\n")
        self.state_path = self.state_dir / "state.json"
        self.quarantine_dir = self.state_dir / "quarantine"
        self.quarantine_dir.mkdir(exist_ok=True)
        if not self.state_path.exists():
            _atomic_write(
                self.state_path,
                canonical_json_bytes({"schema": STATE_SCHEMA, "version": 0, "records": []}),
            )
        self._state = self._read_state()
        self._episode: JsonDict | None = None
        self.read_only_violations: list[JsonDict] = []
        self.crash_stage: str | None = None

    def _read_state(self) -> JsonDict:
        value = json.loads(self.state_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema") != STATE_SCHEMA:
            raise ValueError("invalid transactional memory state")
        return value

    def state_bytes(self) -> bytes:
        return self.state_path.read_bytes()

    def state_hash(self) -> str:
        return sha256_bytes(self.state_bytes())

    def records(self) -> list[JsonDict]:
        return deepcopy(list(self._state["records"]))

    def begin_episode(self, event_id: str) -> JsonDict:
        state_bytes = self.state_bytes()
        self._episode = {
            "event_id": event_id,
            "state_bytes": state_bytes,
            "state_hash": sha256_bytes(state_bytes),
            "records": self.records(),
            "version": int(self._state["version"]),
        }
        return deepcopy(self._episode)

    def end_episode(self) -> None:
        self._episode = None

    def lookup(self, snapshot: Mapping[str, Any], key: str) -> JsonDict:
        record = next(
            (row for row in snapshot["records"] if row.get("key") == key),
            None,
        )
        safe = record is None or record.get("certified") is True
        return {"key": key, "found": record is not None, "safe": safe, "record": record}

    def _admission_checks(
        self,
        proposal: Mapping[str, Any],
        event: Mapping[str, Any],
        boundary_index: int,
    ) -> JsonDict:
        existing = next(
            (row for row in self._state["records"] if row["key"] == proposal.get("key")),
            None,
        )
        same_content = existing is not None and existing.get("content_hash") == proposal.get(
            "content_hash"
        )
        return {
            "exact_checker": exact_checker(proposal, event),
            "scope": proposal.get("scope") == event.get("scope"),
            "provenance": proposal.get("source_event_id") == event.get("event_id")
            and proposal.get("evidence_hash") == event_evidence_hash(event),
            "future_use_eligibility": proposal.get("future_use_eligible") is True,
            "ttl": int(proposal.get("expires_after", -1)) >= boundary_index,
            "conflict": existing is None or same_content,
            "duplicate": not same_content,
        }

    def _publish(self, data: bytes, receipt: JsonDict) -> JsonDict:
        fd, name = tempfile.mkstemp(
            prefix=f".{self.state_path.name}.",
            suffix=".tmp",
            dir=self.state_path.parent,
        )
        temporary = Path(name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            receipt["atomic_write"]["file_fsync"] = True
            if self.crash_stage == "before_rename":
                self.crash_stage = None
                raise CrashInjected("before_rename", receipt)
            os.replace(temporary, self.state_path)
            receipt["atomic_write"]["rename"] = True
            _fsync_directory(self.state_path.parent)
            receipt["atomic_write"]["directory_fsync"] = True
            if self.crash_stage == "after_rename":
                self.crash_stage = None
                raise CrashInjected("after_rename", receipt)
        finally:
            if temporary.exists():
                temporary.unlink()
        return receipt["atomic_write"]

    def _quarantine(
        self,
        proposal: Mapping[str, Any],
        event: Mapping[str, Any],
        checks: Mapping[str, Any],
    ) -> JsonDict:
        payload = {
            "event_id": event["event_id"],
            "proposal": dict(proposal),
            "checks": dict(checks),
            "reason": "admission_check_failed",
        }
        digest = sha256_json(payload).removeprefix("sha256:")
        path = self.quarantine_dir / f"{event['event_id']}-{digest}.json"
        _atomic_write(path, canonical_json_bytes(payload))
        return {"written": True, "entry": path.name, "entry_hash": sha256_bytes(path.read_bytes())}

    def quarantine_entries(self) -> list[str]:
        return sorted(path.name for path in self.quarantine_dir.glob("*.json"))

    def admit(
        self,
        proposal: Mapping[str, Any] | None,
        event: Mapping[str, Any],
        *,
        boundary_index: int,
    ) -> JsonDict:
        if self._episode is not None:
            violation = {
                "attack_id": f"read_only_write:{event['event_id']}",
                "attempted": True,
                "rejected": True,
                "parent_hash": self._episode["state_hash"],
            }
            self.read_only_violations.append(violation)
            raise ReadOnlyEpisodeError("active episode is read-only")
        if proposal is None:
            return {
                "admitted": False,
                "checks": {},
                "unsafe_admitted": False,
                "unsafe_used": False,
                "reason": "no_proposal",
            }
        checks = self._admission_checks(proposal, event, boundary_index)
        if not all(checks.values()):
            return {
                "admitted": False,
                "checks": checks,
                "unsafe_admitted": False,
                "unsafe_used": False,
                "reason": "admission_check_failed",
                "quarantine_receipt": self._quarantine(proposal, event, checks),
            }
        parent_bytes = self.state_bytes()
        parent_state = deepcopy(self._state)
        record = {**dict(proposal), "certified": True}
        new_state = deepcopy(parent_state)
        new_state["version"] = int(parent_state["version"]) + 1
        new_state["records"] = sorted(
            [*parent_state["records"], record], key=lambda row: str(row["key"])
        )
        new_bytes = canonical_json_bytes(new_state)
        receipt: JsonDict = {
            "event_id": event["event_id"],
            "parent_hash": sha256_bytes(parent_bytes),
            "evidence_hash": proposal["evidence_hash"],
            "new_state_hash": sha256_bytes(new_bytes),
            "reason": "all_admission_checks_passed",
            "inverse_patch": {
                "operation": "remove_record",
                "key": proposal["key"],
                "parent_version": parent_state["version"],
            },
            "parent_bytes_b64": encode_bytes(parent_bytes),
            "new_state_bytes_b64": encode_bytes(new_bytes),
            "atomic_write": {
                "file_fsync": False,
                "rename": False,
                "directory_fsync": False,
            },
        }
        self._publish(new_bytes, receipt)
        self._state = new_state
        return {
            "admitted": True,
            "checks": checks,
            "unsafe_admitted": False,
            "unsafe_used": False,
            "reason": receipt["reason"],
            "commit_receipt": receipt,
        }

    def restart_receipt(self, boundary_id: str, expected_bytes: bytes) -> JsonDict:
        restarted = type(self)(self.state_dir)
        actual = restarted.state_bytes()
        return {
            "boundary_id": boundary_id,
            "expected_hash": sha256_bytes(expected_bytes),
            "actual_hash": sha256_bytes(actual),
            "bytes_match": actual == expected_bytes,
            "hash_match": sha256_bytes(actual) == sha256_bytes(expected_bytes),
        }

    def rollback(self, receipt: Mapping[str, Any]) -> JsonDict:
        current = self._read_state()
        patch = dict(receipt["inverse_patch"])
        records = [row for row in current["records"] if row["key"] != patch["key"]]
        reverted = {"schema": STATE_SCHEMA, "version": patch["parent_version"], "records": records}
        reverted_bytes = canonical_json_bytes(reverted)
        parent_bytes = decode_bytes(str(receipt["parent_bytes_b64"]))
        inverse_matches_parent = reverted_bytes == parent_bytes
        atomic = _atomic_write(self.state_path, parent_bytes)
        self._state = self._read_state()
        return {
            "row_type": "rollback",
            "inverse_patch_applied": inverse_matches_parent,
            "byte_identical": self.state_bytes() == parent_bytes,
            "parent_hash": receipt["parent_hash"],
            "restored_hash": self.state_hash(),
            "atomic_write": atomic,
            "passed": inverse_matches_parent and self.state_bytes() == parent_bytes,
        }


def _atomic_probe(state_root: Path) -> bool:
    probe = state_root / "atomic-probe.json"
    _atomic_write(probe, b"old\n")
    _atomic_write(probe, b"new\n")
    return probe.read_bytes() == b"new\n"


def check_preconditions(
    state_root: Path,
    overrides: Mapping[str, bool] | None = None,
) -> JsonDict:
    """Check all owned resources before stream evaluation."""

    owner = TransactionalConstraintMemory(state_root / "precondition-state")
    proposals = [proposal_for(event) for event in controlled_stream()]
    deterministic = all(
        proposal is None
        or exact_checker(proposal, event) == exact_checker(deepcopy(proposal), deepcopy(event))
        for proposal, event in zip(proposals, controlled_stream(), strict=True)
    )
    observed = {
        "deterministic_exact_labels": deterministic,
        "atomic_write_support": _atomic_probe(state_root),
        "task_owned_state_directory": owner.ownership_path.read_text(encoding="utf-8")
        == "experiment_6748\n",
        "immutable_stream_seeds": dict(RANDOM_SEEDS)
        == {"stream": 6748, "order": 6749, "attack": 6750},
    }
    observed.update(dict(overrides or {}))
    checks = {
        name: {"expected": True, "observed": value, "passed": value is True}
        for name, value in observed.items()
    }
    return {"checks": checks, "all_passed": all(row["passed"] for row in checks.values())}


def _order_rows(
    state_root: Path,
    manifest: Mapping[str, Any],
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict], list[JsonDict], list[str]]:
    events = {str(event["event_id"]): dict(event) for event in manifest["events"]}
    rows: list[JsonDict] = []
    receipts: list[JsonDict] = []
    violations: list[JsonDict] = []
    restarts: list[JsonDict] = []
    final_hashes: list[str] = []
    for order in manifest["orders"]:
        order_id = str(order["order_id"])
        memory = TransactionalConstraintMemory(state_root / order_id)
        for boundary, event_id in enumerate(order["event_ids"], start=1):
            event = deepcopy(events[str(event_id)])
            snapshot = memory.begin_episode(str(event_id))
            lookup = {"found": False, "safe": True}
            if event["kind"] == "retention_anchor":
                lookup = memory.lookup(snapshot, str(event["target_key"]))
            proposal = proposal_for(event)
            if boundary == 1:
                try:
                    memory.admit(proposal, event, boundary_index=boundary)
                except ReadOnlyEpisodeError:
                    violation = dict(memory.read_only_violations[-1])
                    violation.update(
                        {
                            "row_type": "attack",
                            "order_id": order_id,
                            "passed": memory.state_hash() == snapshot["state_hash"],
                        }
                    )
                    rows.append(violation)
            memory.end_episode()
            decision = memory.admit(proposal, event, boundary_index=boundary)
            should_admit = event["kind"] == "reusable_repair"
            if decision["admitted"]:
                receipts.append(dict(decision["commit_receipt"], order_id=order_id))
            passed = decision["admitted"] is should_admit
            if event["kind"] == "retention_anchor":
                passed = passed and lookup["found"] is True and lookup["safe"] is True
            row = {
                "row_type": "event",
                "order_id": order_id,
                "boundary": boundary,
                "event_id": event_id,
                "kind": event["kind"],
                "admitted": decision["admitted"],
                "used": lookup["found"],
                "unsafe_admitted": decision["unsafe_admitted"],
                "unsafe_used": lookup["found"] and lookup["safe"] is not True,
                "state_hash": memory.state_hash(),
                "passed": passed,
            }
            rows.append(row)
            restarts.append(memory.restart_receipt(f"{order_id}:{boundary}", memory.state_bytes()))
        violations.extend(memory.read_only_violations)
        final_hashes.append(memory.state_hash())
    return rows, receipts, violations, restarts, final_hashes


def _crash_rows(
    state_root: Path,
    event: Mapping[str, Any],
) -> tuple[list[JsonDict], list[JsonDict]]:
    rows: list[JsonDict] = []
    restarts: list[JsonDict] = []
    for stage in ("before_rename", "after_rename"):
        memory = TransactionalConstraintMemory(state_root / f"crash-{stage}")
        parent = memory.state_bytes()
        memory.begin_episode(str(event["event_id"]))
        memory.end_episode()
        memory.crash_stage = stage
        try:
            memory.admit(proposal_for(event), event, boundary_index=1)
        except CrashInjected as crash:
            restarted = TransactionalConstraintMemory(memory.state_dir)
            expected = (
                parent
                if stage == "before_rename"
                else decode_bytes(str(crash.receipt["new_state_bytes_b64"]))
            )
            passed = restarted.state_bytes() == expected
            rows.append(
                {
                    "row_type": "attack",
                    "attack_id": f"crash:{stage}",
                    "stage": stage,
                    "atomic_boundary": "parent" if stage == "before_rename" else "new",
                    "passed": passed,
                }
            )
            restarts.append(restarted.restart_receipt(f"crash:{stage}", expected))
    return rows, restarts


def _rollback_rows(state_root: Path, events: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    memory = TransactionalConstraintMemory(state_root / "rollback")
    event = deepcopy(dict(events[0]))
    memory.begin_episode(str(event["event_id"]))
    memory.end_episode()
    decision = memory.admit(proposal_for(event), event, boundary_index=1)
    restarted = TransactionalConstraintMemory(memory.state_dir)
    return [restarted.rollback(decision["commit_receipt"])]


def gate_check_summary(
    *,
    preconditions: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
    violations: Sequence[Mapping[str, Any]],
    restarts: Sequence[Mapping[str, Any]],
    rollback_rows: Sequence[Mapping[str, Any]],
    final_hashes: Sequence[str],
    manifest: Mapping[str, Any],
) -> JsonDict:
    checks = {
        "preconditions_pass": preconditions["all_passed"] is True,
        "stream_frozen": manifest["frozen_before_policy_evaluation"] is True
        and len(manifest["orders"]) == 6,
        "mandatory_rows_pass": bool(rows) and all(row.get("passed") is True for row in rows),
        "commit_receipts_complete": bool(receipts)
        and all(
            all(receipt.get(field) for field in ("parent_hash", "evidence_hash", "new_state_hash"))
            and receipt.get("inverse_patch")
            and all(receipt.get("atomic_write", {}).values())
            for receipt in receipts
        ),
        "read_only_writes_rejected": bool(violations)
        and all(row.get("attempted") is True and row.get("rejected") is True for row in violations),
        "unsafe_admission_zero": not any(row.get("unsafe_admitted") for row in rows),
        "unsafe_use_zero": not any(row.get("unsafe_used") for row in rows),
        "restart_bytes_equal": bool(restarts)
        and all(
            row.get("bytes_match") is True and row.get("hash_match") is True for row in restarts
        ),
        "rollback_bytes_equal": bool(rollback_rows)
        and all(row.get("byte_identical") is True for row in rollback_rows),
        "orders_converge": len(set(final_hashes)) == 1 and len(final_hashes) == 6,
    }
    failures = [
        {"check": name, "expected": True, "observed": value}
        for name, value in checks.items()
        if value is not True
    ]
    return {
        "checks": checks,
        "failed_checks": [row["check"] for row in failures],
        "failures": failures,
    }


def _blocked_gate_summary(preconditions: Mapping[str, Any]) -> JsonDict:
    failures = [
        {"check": name, "expected": row["expected"], "observed": row["observed"]}
        for name, row in preconditions["checks"].items()
        if row["passed"] is not True
    ]
    return {
        "checks": {name: row["passed"] for name, row in preconditions["checks"].items()},
        "failed_checks": [row["check"] for row in failures],
        "failures": failures,
    }


def _artifact_base(
    *,
    duration_s: float,
    preconditions: Mapping[str, Any],
    manifest: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
) -> JsonDict:
    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "status": "complete_blocked_transaction_fixture",
        "field_principles": FIELD_PRINCIPLES,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "random_seed": dict(RANDOM_SEEDS),
        "reproducibility_checksum": "",
        "preconditions_checked": dict(preconditions),
        "rows": [],
        "stream_manifest": dict(manifest),
        "commit_receipts": [],
        "read_only_violations": [],
        "unsafe_admission_count": 0,
        "unsafe_use_count": 0,
        "restart_receipts": [],
        "rollback_byte_identity": {"rows": [], "all_match": False},
        "transaction_memory_ready": False,
        "gate_check_summary": _blocked_gate_summary(preconditions),
        "verdict_class": "blocked",
        "honest_verdict": "complete_blocked_transaction_fixture: owned precondition failed",
        "tests_run": [dict(row) for row in tests_run],
        "final_state_hashes": [],
        "verifier_is_oracle": True,
    }


def run_fixture(
    *,
    state_root: Path | str | None = None,
    duration_s: float | None = None,
    precondition_overrides: Mapping[str, bool] | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Run the complete fixture without touching shared live memory."""

    started = time.monotonic()
    if state_root is None:
        with tempfile.TemporaryDirectory(prefix="carnot-exp6748-") as directory:
            return run_fixture(
                state_root=Path(directory),
                duration_s=duration_s,
                precondition_overrides=precondition_overrides,
                tests_run=tests_run,
            )
    root = Path(state_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = freeze_stream(controlled_stream())
    preconditions = check_preconditions(root, precondition_overrides)
    elapsed = duration_s if duration_s is not None else time.monotonic() - started
    test_receipts = list(tests_run or DEFAULT_TESTS_RUN)
    artifact = _artifact_base(
        duration_s=elapsed,
        preconditions=preconditions,
        manifest=manifest,
        tests_run=test_receipts,
    )
    if not preconditions["all_passed"]:
        artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
        return artifact
    rows, receipts, violations, restarts, final_hashes = _order_rows(root, manifest)
    crash_rows, crash_restarts = _crash_rows(root, manifest["events"][0])
    rollback_rows = _rollback_rows(root, manifest["events"])
    rows.extend(crash_rows)
    rows.extend(rollback_rows)
    restarts.extend(crash_restarts)
    gates = gate_check_summary(
        preconditions=preconditions,
        rows=rows,
        receipts=receipts,
        violations=violations,
        restarts=restarts,
        rollback_rows=rollback_rows,
        final_hashes=final_hashes,
        manifest=manifest,
    )
    ready = not gates["failed_checks"]
    artifact.update(
        {
            "status": (
                "complete_transaction_fixture_ready"
                if ready
                else "complete_blocked_transaction_fixture"
            ),
            "rows": rows,
            "commit_receipts": receipts,
            "read_only_violations": violations,
            "unsafe_admission_count": sum(1 for row in rows if row.get("unsafe_admitted") is True),
            "unsafe_use_count": sum(1 for row in rows if row.get("unsafe_used") is True),
            "restart_receipts": restarts,
            "rollback_byte_identity": {
                "rows": rollback_rows,
                "all_match": all(row["byte_identical"] is True for row in rollback_rows),
            },
            "transaction_memory_ready": ready,
            "gate_check_summary": gates,
            "verdict_class": "circular_positive" if ready else "blocked",
            "honest_verdict": (
                "complete_transaction_fixture_ready: read-only episodes, exact delayed commits, "
                "atomic restart, quarantine, and byte-exact rollback passed"
                if ready
                else "complete_blocked_transaction_fixture: mandatory fixture row failed"
            ),
            "final_state_hashes": final_hashes,
        }
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the frozen inputs and row-derived states, excluding wall time."""

    material = {
        "schema": artifact.get("schema"),
        "random_seed": artifact.get("random_seed"),
        "preconditions_checked": artifact.get("preconditions_checked"),
        "stream_manifest": artifact.get("stream_manifest"),
        "rows": artifact.get("rows"),
        "commit_receipts": artifact.get("commit_receipts"),
        "restart_receipts": artifact.get("restart_receipts"),
        "rollback_byte_identity": artifact.get("rollback_byte_identity"),
        "final_state_hashes": artifact.get("final_state_hashes"),
    }
    return sha256_json(material)


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return closed validation errors without changing the artifact."""

    errors: list[str] = []
    if set(artifact) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("required field set mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        errors.append("verdict_class outside closed enum")
    expected_principles = set(REQUIRED_ARTIFACT_FIELDS) | {
        f"gate:{name}" for name in READINESS_GATES
    }
    if set(artifact.get("field_principles", {})) != expected_principles:
        errors.append("field_principles coverage mismatch")
    if artifact.get("transaction_memory_ready") is True and (
        artifact.get("unsafe_admission_count") != 0 or artifact.get("unsafe_use_count") != 0
    ):
        errors.append("unsafe counts must be zero when ready")
    if artifact.get("transaction_memory_ready") is True and artifact.get(
        "gate_check_summary", {}
    ).get("failed_checks"):
        errors.append("ready artifact has failed gates")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    return errors


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> JsonDict:
    """Publish the validated terminal artifact with one atomic rename."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    data = json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return _atomic_write(path, data + b"\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Run or validate the task-owned Exp6748 artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--state-root")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    result_path = Path(args.result_path)
    if args.validate:
        artifact = json.loads(result_path.read_text(encoding="utf-8"))
        errors = validate_artifact(artifact)
        if errors:
            raise ValueError("; ".join(errors))
        return 0
    artifact = run_fixture(state_root=args.state_root)
    write_artifact(result_path, artifact)
    return 0

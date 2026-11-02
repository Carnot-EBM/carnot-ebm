"""Bounded adaptive-state microkernel for Exp5859.

Spec refs: REQ-LEARN-5859, SCENARIO-LEARN-5859-PRECONDITIONS,
SCENARIO-LEARN-5859-OPERATION-PARITY,
SCENARIO-LEARN-5859-STATE-HASH-ROUNDTRIP,
SCENARIO-LEARN-5859-FAIL-CLOSED.

This module is deliberately small and deterministic. It translates only the
Exp5858 accepted operation classes into a bounded state ABI that can be replayed
by Python, Rust, and later board-facing code. It does not learn new science:
the kernel only stores, rejects, rolls back, serializes, restores, and hashes
external adaptive state.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5859_adaptive_state_microkernel_parity.json")
EXP5858_RESULT_RELATIVE_PATH = Path(
    "results/experiment_5858_reduced_oracle_continuous_self_learning.json"
)
EXP5858_ROWS_RELATIVE_PATH = Path(
    "results/experiment_5858_reduced_oracle_continuous_self_learning.rows.jsonl"
)
SELF_LEARNING_SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
PY_MODULE_RELATIVE_PATH = Path("python/carnot/adaptive_state.py")
PY_TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_5859_adaptive_state_microkernel_parity.py"
)
RUST_CORE_RELATIVE_PATH = Path("crates/carnot-core/src/adaptive_state.rs")
RUST_BINDING_RELATIVE_PATH = Path("crates/carnot-python/src/adaptive_state.rs")
PYPROJECT_RELATIVE_PATH = Path("pyproject.toml")
CARGO_WORKSPACE_RELATIVE_PATH = Path("Cargo.toml")
CARGO_LOCK_RELATIVE_PATH = Path("Cargo.lock")
CARGO_CORE_RELATIVE_PATH = Path("crates/carnot-core/Cargo.toml")
CARGO_PYTHON_RELATIVE_PATH = Path("crates/carnot-python/Cargo.toml")
PROTECTED_FILE_RELATIVE_PATH = Path("scripts/research_conductor.py")

SCHEMA = "carnot.experiment_5859.adaptive_state_microkernel_parity.v1"
STATE_SCHEMA = "carnot.adaptive_state_microkernel.v1.state"
CHECKPOINT_SCHEMA = "carnot.adaptive_state_microkernel.v1.checkpoint"
ABI_VERSION = 1
MAX_CAPACITY = 64
MAX_HISTORY_CAPACITY = 128
MAX_REPLAY_LIMIT = 64
MAX_EVENT_ID_LEN = 64
MAX_REASON_LEN = 32
U16_MAX = 65_535
U32_MAX = 4_294_967_295
INFERENCE_SUBSTRATE = "deterministic_cross_language_state_execution_no_llm"
ABI_OPERATIONS = (
    "apply_event",
    "acquire_core",
    "quarantine",
    "promote",
    "select_replay",
    "roll_back",
    "serialize",
    "restore",
    "canonical_state_hash",
)
QUALIFIED_EXP5858_CHANGES = ("addition", "supersession", "recurrence")
EVENT_FIELDS = (
    "event_id",
    "chronology_index",
    "change",
    "signature_hash",
    "payload_hash",
    "confidence_q16",
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "qualified_operation_mapping",
    "abi_schema_and_bounds",
    "python_implementation_receipt",
    "rust_implementation_receipt",
    "binding_receipt",
    "cross_language_operation_parity",
    "canonical_state_and_hash_parity",
    "serialization_restart_and_rollback_parity",
    "invalid_input_and_capacity_controls",
    "per_operation_latency_receipts",
    "adaptive_state_microkernel_ready_score",
    "duration_s",
    "inference_substrate",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)
REQUIRED_FIELD_PRINCIPLES: dict[str, str] = {
    "status": "A terminal conformance state distinguishes a usable kernel from partial bindings.",
    "preconditions_checked": (
        "Gate, schema, toolchain, manifests, resources, and outputs prevent incompatible builds."
    ),
    "qualified_operation_mapping": "Only scientifically accepted Exp5858 operations enter the kernel.",
    "abi_schema_and_bounds": (
        "Fixed types, capacities, ordering, and versions make hardware mapping finite."
    ),
    "python_implementation_receipt": "The Python reference owns readable state semantics.",
    "rust_implementation_receipt": "The Rust path owns deterministic deployable execution.",
    "binding_receipt": "The cross-language call surface must be explicit and tested.",
    "cross_language_operation_parity": "Every operation must accept, reject, and transition identically.",
    "canonical_state_and_hash_parity": "Equivalent states must produce identical canonical hashes.",
    "serialization_restart_and_rollback_parity": (
        "Durable state must round-trip and restore identically."
    ),
    "invalid_input_and_capacity_controls": "Malformed or unbounded inputs fail closed.",
    "per_operation_latency_receipts": "Measured timing is descriptive and cannot imply speedup.",
    "adaptive_state_microkernel_ready_score": "EMIT BARE scalar for optional board parity.",
    "duration_s": "Measured build/test time exposes bootstrap-only work.",
    "inference_substrate": (
        "`deterministic_cross_language_state_execution_no_llm` declares the true path."
    ),
    "field_provenance": (
        "Every result traces to operation traces, implementations, toolchains, and hashes."
    ),
    "test_commands": "Commands document Python, Rust, binding, property, and E2E checks.",
    "test_exit_codes": "Exit codes prevent failed conformance becoming readiness.",
    "reproducibility_checksum": (
        "A checksum detects ABI, implementation, trace, or toolchain drift."
    ),
    "honest_verdict": "A terminal prefix states parity, mismatch, or blocked outcome.",
}


def canonical_json(value: Any) -> str:
    """Serialize JSON evidence exactly once before hashing or byte comparison."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Hash text with an explicit algorithm prefix so receipts are self-describing."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    """Hash exact bytes with the same prefix used in result artifacts."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash canonical JSON-compatible evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash file bytes rather than trusting path names or modification times."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def read_json(path: str | Path) -> JsonDict:
    """Read a JSON object artifact."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _is_hash(value: object) -> bool:
    text = value if isinstance(value, str) else ""
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(char in "0123456789abcdef" for char in text[7:])
    )


def _valid_short_token(value: object, max_len: int) -> bool:
    text = value if isinstance(value, str) else ""
    return 0 < len(text) <= max_len and all(32 <= ord(char) <= 126 for char in text)


def _base_state(capacity: int, history_capacity: int) -> JsonDict:
    return {
        "abi_version": ABI_VERSION,
        "applied": [],
        "capacity": capacity,
        "core": [],
        "evicted": [],
        "history_capacity": history_capacity,
        "last_chronology": -1,
        "promoted": [],
        "quarantine": [],
        "schema": STATE_SCHEMA,
        "version_id": 0,
    }


def _result(accepted: bool, code: str, version_id: int, state_hash: str, **extra: Any) -> JsonDict:
    payload: JsonDict = {
        "accepted": accepted,
        "code": code,
        "state_hash": state_hash,
        "version_id": version_id,
    }
    payload.update(extra)
    return payload


def _find_entry(entries: list[JsonDict], event_id: str) -> JsonDict | None:
    for entry in entries:
        if entry["event_id"] == event_id:
            return entry
    return None


def _without_entry(entries: list[JsonDict], event_id: str) -> list[JsonDict]:
    return [entry for entry in entries if entry["event_id"] != event_id]


def _event_from_input(event: object) -> tuple[JsonDict | None, str]:
    if not isinstance(event, dict):
        return None, "MALFORMED_EVENT"
    if set(EVENT_FIELDS) != set(event):
        return None, "MALFORMED_EVENT"
    event_id = event["event_id"]
    chronology_index = event["chronology_index"]
    change = event["change"]
    signature_hash = event["signature_hash"]
    payload_hash = event["payload_hash"]
    confidence_q16 = event["confidence_q16"]
    if not _valid_short_token(event_id, MAX_EVENT_ID_LEN):
        return None, "INVALID_EVENT_ID"
    if not isinstance(chronology_index, int) or not 0 <= chronology_index <= U32_MAX:
        return None, "FIXED_WIDTH_OVERFLOW"
    if change not in QUALIFIED_EXP5858_CHANGES:
        return None, "UNQUALIFIED_OPERATION"
    if not _is_hash(signature_hash) or not _is_hash(payload_hash):
        return None, "INVALID_HASH"
    if not isinstance(confidence_q16, int) or not 0 <= confidence_q16 <= U16_MAX:
        return None, "FIXED_WIDTH_OVERFLOW"
    clean = {
        "change": change,
        "chronology_index": chronology_index,
        "confidence_q16": confidence_q16,
        "event_id": event_id,
        "payload_hash": payload_hash,
        "signature_hash": signature_hash,
    }
    return clean, "OK"


class AdaptiveStateKernel:
    """Small bounded state machine shared by the Python and Rust references."""

    def __init__(self, capacity: int = 8, history_capacity: int = 32) -> None:
        if not isinstance(capacity, int) or not 1 <= capacity <= MAX_CAPACITY:
            raise ValueError("capacity must be an integer in [1, 64]")
        if (
            not isinstance(history_capacity, int)
            or not 2 <= history_capacity <= MAX_HISTORY_CAPACITY
        ):
            raise ValueError("history_capacity must be an integer in [2, 128]")
        self._state = _base_state(capacity, history_capacity)
        self._history: dict[int, JsonDict] = {0: _copy_json(self._state)}

    @classmethod
    def restore(cls, checkpoint: bytes | bytearray | memoryview) -> "AdaptiveStateKernel":
        try:
            payload = json.loads(bytes(checkpoint).decode("utf-8"))
        except (TypeError, ValueError, UnicodeDecodeError) as exc:
            raise ValueError("checkpoint is not valid adaptive-state JSON") from exc
        if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("checkpoint schema mismatch")
        if payload.get("abi_version") != ABI_VERSION:
            raise ValueError("checkpoint ABI version mismatch")
        active = payload.get("active")
        history = payload.get("history")
        if not isinstance(active, dict) or not isinstance(history, list) or not history:
            raise ValueError("checkpoint payload is incomplete")
        kernel = cls(int(active["capacity"]), int(active["history_capacity"]))
        kernel._state = _copy_json(active)
        kernel._history = {int(item["version_id"]): _copy_json(item) for item in history}
        if int(kernel._state["version_id"]) not in kernel._history:
            raise ValueError("checkpoint active version missing from history")
        if kernel._history[int(kernel._state["version_id"])] != kernel._state:
            raise ValueError("checkpoint active state differs from history")
        return kernel

    def apply_event(self, event: object) -> JsonDict:
        before = self.canonical_state_hash()
        clean, code = _event_from_input(event)
        if clean is None:
            return _result(False, code, self.version_id, before)
        if _find_entry(self._state["applied"], clean["event_id"]) is not None:
            return _result(False, "DUPLICATE_EVENT", self.version_id, before)
        if clean["chronology_index"] <= self._state["last_chronology"]:
            return _result(False, "OUT_OF_ORDER_EVENT", self.version_id, before)
        self._state["applied"].append(clean)
        self._state["applied"].sort(key=lambda item: item["event_id"])
        self._state["last_chronology"] = clean["chronology_index"]
        self._bump_version()
        return _result(True, "OK", self.version_id, self.canonical_state_hash())

    def acquire_core(self, event_id: str) -> JsonDict:
        before = self.canonical_state_hash()
        if not _valid_short_token(event_id, MAX_EVENT_ID_LEN):
            return _result(False, "INVALID_EVENT_ID", self.version_id, before)
        event = _find_entry(self._state["applied"], event_id)
        if event is None:
            return _result(False, "UNKNOWN_EVENT", self.version_id, before)
        if event["change"] != "addition":
            return _result(False, "UNQUALIFIED_OPERATION", self.version_id, before)
        if _find_entry(self._state["core"], event_id) is not None:
            return _result(False, "DUPLICATE_CORE", self.version_id, before)
        self._state["core"].append(_copy_json(event))
        self._state["core"].sort(key=lambda item: item["event_id"])
        self._bump_version()
        return _result(True, "OK", self.version_id, self.canonical_state_hash())

    def quarantine(self, event_id: str, reason_code: str) -> JsonDict:
        before = self.canonical_state_hash()
        if not _valid_short_token(event_id, MAX_EVENT_ID_LEN):
            return _result(False, "INVALID_EVENT_ID", self.version_id, before)
        if not _valid_short_token(reason_code, MAX_REASON_LEN):
            return _result(False, "INVALID_REASON", self.version_id, before)
        if _find_entry(self._state["applied"], event_id) is None:
            return _result(False, "UNKNOWN_EVENT", self.version_id, before)
        if _find_entry(self._state["quarantine"], event_id) is not None:
            return _result(False, "DUPLICATE_QUARANTINE", self.version_id, before)
        self._state["core"] = _without_entry(self._state["core"], event_id)
        self._state["quarantine"].append(
            {
                "event_id": event_id,
                "reason_code": reason_code,
                "version_id": self.version_id + 1,
            }
        )
        self._state["quarantine"].sort(key=lambda item: item["event_id"])
        self._bump_version()
        return _result(True, "OK", self.version_id, self.canonical_state_hash())

    def promote(self, event_id: str) -> JsonDict:
        before = self.canonical_state_hash()
        if not _valid_short_token(event_id, MAX_EVENT_ID_LEN):
            return _result(False, "INVALID_EVENT_ID", self.version_id, before)
        if _find_entry(self._state["quarantine"], event_id) is not None:
            return _result(False, "QUARANTINED_EVENT", self.version_id, before)
        event = _find_entry(self._state["core"], event_id)
        if event is None:
            known = _find_entry(self._state["applied"], event_id) is not None
            code = "NOT_IN_CORE" if known else "UNKNOWN_EVENT"
            return _result(False, code, self.version_id, before)
        if _find_entry(self._state["promoted"], event_id) is not None:
            return _result(False, "DUPLICATE_PROMOTION", self.version_id, before)
        promoted = _copy_json(event)
        promoted["promoted_version"] = self.version_id + 1
        self._state["promoted"].append(promoted)
        self._evict_if_needed(promoted["promoted_version"])
        self._state["promoted"].sort(key=lambda item: item["event_id"])
        self._bump_version()
        return _result(True, "OK", self.version_id, self.canonical_state_hash())

    def select_replay(self, limit: int) -> JsonDict:
        before = self.canonical_state_hash()
        if not isinstance(limit, int) or limit < 0:
            return _result(False, "INVALID_REPLAY_LIMIT", self.version_id, before)
        if limit > MAX_REPLAY_LIMIT:
            return _result(False, "REPLAY_LIMIT_EXCEEDED", self.version_id, before)
        ordered = sorted(
            self._state["promoted"],
            key=lambda item: (-item["confidence_q16"], item["chronology_index"], item["event_id"]),
        )
        selected = [item["event_id"] for item in ordered[:limit]]
        return _result(True, "OK", self.version_id, before, selected_replay=selected)

    def roll_back(self, version_id: int) -> JsonDict:
        before = self.canonical_state_hash()
        if not isinstance(version_id, int) or version_id < 0:
            return _result(False, "ROLLBACK_PAST_ROOT", self.version_id, before)
        if version_id not in self._history:
            return _result(False, "ROLLBACK_VERSION_MISSING", self.version_id, before)
        self._state = _copy_json(self._history[version_id])
        keep_versions = [version for version in self._history if version <= version_id]
        self._history = {version: self._history[version] for version in keep_versions}
        return _result(True, "OK", self.version_id, self.canonical_state_hash())

    def serialize(self) -> bytes:
        payload = {
            "abi_version": ABI_VERSION,
            "active": self.canonical_state(),
            "history": [self._history[key] for key in sorted(self._history)],
            "schema": CHECKPOINT_SCHEMA,
        }
        return canonical_json(payload).encode("utf-8")

    def canonical_state(self) -> JsonDict:
        return _copy_json(self._state)

    def canonical_state_json(self) -> str:
        return canonical_json(self._state)

    def canonical_state_hash(self) -> str:
        return sha256_text(self.canonical_state_json())

    @property
    def version_id(self) -> int:
        return int(self._state["version_id"])

    def _bump_version(self) -> None:
        self._state["version_id"] += 1
        self._history[self.version_id] = _copy_json(self._state)
        min_kept = max(1, self.version_id - self._state["history_capacity"] + 2)
        self._history = {
            version: state
            for version, state in self._history.items()
            if version == 0 or version >= min_kept
        }

    def _evict_if_needed(self, version_id: int) -> None:
        while len(self._state["promoted"]) > self._state["capacity"]:
            victim = min(
                self._state["promoted"],
                key=lambda item: (item["promoted_version"], item["event_id"]),
            )
            self._state["promoted"] = _without_entry(self._state["promoted"], victim["event_id"])
            self._state["evicted"].append({"event_id": victim["event_id"], "version_id": version_id})
            self._state["evicted"].sort(key=lambda item: (item["version_id"], item["event_id"]))


def make_event(event_id: str, chronology_index: int, change: str, confidence_q16: int) -> JsonDict:
    """Create a bounded fixture event without using any science labels."""

    basis = f"{event_id}:{chronology_index}:{change}:{confidence_q16}"
    return {
        "change": change,
        "chronology_index": chronology_index,
        "confidence_q16": confidence_q16,
        "event_id": event_id,
        "payload_hash": sha256_text("payload:" + basis),
        "signature_hash": sha256_text("signature:" + basis),
    }


def deterministic_fixture_trace() -> list[JsonDict]:
    """Return the fixed trace used by Python, Rust, binding, and artifact checks."""

    events = [
        make_event("evt-0001", 0, "addition", 50_000),
        make_event("evt-0002", 1, "supersession", 1_000),
        make_event("evt-0003", 2, "recurrence", 40_000),
        make_event("evt-0004", 3, "addition", 60_000),
        make_event("evt-0005", 4, "addition", 65_000),
    ]
    return [
        {"event": events[0], "op": "apply_event"},
        {"event_id": "evt-0001", "op": "acquire_core"},
        {"event_id": "evt-0001", "op": "promote"},
        {"event": events[1], "op": "apply_event"},
        {"event_id": "evt-0002", "op": "quarantine", "reason_code": "superseded"},
        {"event": events[2], "op": "apply_event"},
        {"limit": 2, "op": "select_replay"},
        {"event": events[3], "op": "apply_event"},
        {"event_id": "evt-0004", "op": "acquire_core"},
        {"event_id": "evt-0004", "op": "promote"},
        {"event": events[4], "op": "apply_event"},
        {"event_id": "evt-0005", "op": "acquire_core"},
        {"event_id": "evt-0005", "op": "promote"},
    ]


def randomized_operation_traces(
    seed: int,
    trace_count: int,
    events_per_trace: int,
) -> list[list[JsonDict]]:
    """Generate bounded valid traces from operation fixtures, not science labels."""

    rng = random.Random(seed)
    traces: list[list[JsonDict]] = []
    for trace_index in range(trace_count):
        trace: list[JsonDict] = []
        for event_index in range(events_per_trace):
            event_id = f"r{trace_index:02d}-{event_index:04d}"
            change = rng.choice(QUALIFIED_EXP5858_CHANGES)
            confidence_q16 = rng.randint(0, U16_MAX)
            event = make_event(event_id, event_index, change, confidence_q16)
            trace.append({"event": event, "op": "apply_event"})
            if change == "addition":
                trace.append({"event_id": event_id, "op": "acquire_core"})
                if rng.choice((True, False)):
                    trace.append({"event_id": event_id, "op": "promote"})
            elif change == "supersession":
                trace.append(
                    {"event_id": event_id, "op": "quarantine", "reason_code": "superseded"}
                )
            else:
                trace.append({"limit": rng.randint(0, 3), "op": "select_replay"})
        traces.append(trace)
    return traces


def _dispatch(kernel: object, operation: JsonDict) -> JsonDict:
    name = operation["op"]
    if name == "apply_event":
        return kernel.apply_event(operation["event"])  # type: ignore[attr-defined]
    if name == "acquire_core":
        return kernel.acquire_core(operation["event_id"])  # type: ignore[attr-defined]
    if name == "quarantine":
        return kernel.quarantine(operation["event_id"], operation["reason_code"])  # type: ignore[attr-defined]
    if name == "promote":
        return kernel.promote(operation["event_id"])  # type: ignore[attr-defined]
    if name == "select_replay":
        return kernel.select_replay(operation["limit"])  # type: ignore[attr-defined]
    if name == "roll_back":
        return kernel.roll_back(operation["version_id"])  # type: ignore[attr-defined]
    raise ValueError(f"unsupported operation {name}")


def _command_version(command: list[str]) -> JsonDict:
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


def _ram_receipt() -> JsonDict:
    pages = os.sysconf("SC_AVPHYS_PAGES")
    page_size = os.sysconf("SC_PAGE_SIZE")
    available_mb = int(pages * page_size / (1024 * 1024))
    return {"available_mb": available_mb, "ok": available_mb >= 512, "required_mb": 512}


def _disk_receipt(path: Path) -> JsonDict:
    usage = shutil.disk_usage(path)
    available_mb = int(usage.free / (1024 * 1024))
    return {"available_mb": available_mb, "ok": available_mb >= 512, "required_mb": 512}


def _atomic_receipt(path: Path) -> JsonDict:
    path.parent.mkdir(parents=True, exist_ok=True)
    probe = path.with_suffix(path.suffix + ".atomic_probe")
    probe.write_text("probe", encoding="utf-8")
    os.replace(probe, probe.with_suffix(".done"))
    probe.with_suffix(".done").unlink()
    return {"ok": True, "path": str(path)}


def _read_jsonl(path: Path) -> list[JsonDict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def collect_preconditions(result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH) -> JsonDict:
    """Replay the Exp5858 gate and hash manifests before writing readiness."""

    result_path = Path(result_path)
    exp5858 = read_json(REPO_ROOT / EXP5858_RESULT_RELATIVE_PATH)
    rows = _read_jsonl(REPO_ROOT / EXP5858_ROWS_RELATIVE_PATH)
    row_changes = sorted({row["change"] for row in rows})
    manifest_paths = [
        EXP5858_RESULT_RELATIVE_PATH,
        EXP5858_ROWS_RELATIVE_PATH,
        SELF_LEARNING_SPEC_RELATIVE_PATH,
        PY_MODULE_RELATIVE_PATH,
        PY_TEST_RELATIVE_PATH,
        RUST_CORE_RELATIVE_PATH,
        RUST_BINDING_RELATIVE_PATH,
        PYPROJECT_RELATIVE_PATH,
        CARGO_WORKSPACE_RELATIVE_PATH,
        CARGO_LOCK_RELATIVE_PATH,
        CARGO_CORE_RELATIVE_PATH,
        CARGO_PYTHON_RELATIVE_PATH,
        PROTECTED_FILE_RELATIVE_PATH,
    ]
    file_hashes = {
        path.as_posix(): sha256_file(REPO_ROOT / path)
        for path in manifest_paths
        if (REPO_ROOT / path).exists()
    }
    binding_paths = sorted((REPO_ROOT / "python/carnot").glob("_rust*.so"))
    binding_receipts = [
        {"path": path.relative_to(REPO_ROOT).as_posix(), "sha256": sha256_file(path)}
        for path in binding_paths
    ]
    exp5858_ready = (
        exp5858.get("status") == "ready"
        and exp5858.get("continuous_self_learning_ready_score") == 1.0
    )
    resources = {"disk": _disk_receipt(REPO_ROOT), "ram": _ram_receipt()}
    preconditions_ready = (
        exp5858_ready
        and set(row_changes) == set(QUALIFIED_EXP5858_CHANGES)
        and resources["disk"]["ok"]
        and resources["ram"]["ok"]
    )
    return {
        "atomic_output": _atomic_receipt(result_path),
        "binding_hashes": binding_receipts,
        "exp5858_gate": {
            "continuous_self_learning_ready_score": exp5858.get(
                "continuous_self_learning_ready_score"
            ),
            "honest_verdict": exp5858.get("honest_verdict"),
            "ok": exp5858_ready,
            "status": exp5858.get("status"),
        },
        "exp5858_operation_schema": {
            "qualified_changes": row_changes,
            "row_count": len(rows),
            "schema_hash": sha256_json({"changes": row_changes, "fields": sorted(rows[0])}),
        },
        "file_hashes": file_hashes,
        "preconditions_ready": preconditions_ready,
        "resources": resources,
        "serialization_versions": {
            "abi_version": ABI_VERSION,
            "checkpoint_schema": CHECKPOINT_SCHEMA,
            "state_schema": STATE_SCHEMA,
        },
        "toolchains": {
            "cargo": _command_version(["cargo", "--version"]),
            "python": sys.version.split()[0],
            "rustc": _command_version(["rustc", "--version"]),
            "system": platform.platform(),
        },
    }


def _load_rust_binding() -> Any | None:
    try:
        module = importlib.import_module("carnot._rust")
        return module.RustAdaptiveStateKernel
    except (ImportError, AttributeError):  # pragma: no cover - exercised only when binding absent
        return None


def _run_parity_receipts() -> JsonDict:
    rust_class = _load_rust_binding()
    traces = [deterministic_fixture_trace()]
    traces.extend(randomized_operation_traces(seed=5859, trace_count=4, events_per_trace=6))
    operation_count = 0
    parity_failures: list[JsonDict] = []
    latency_ns: dict[str, list[int]] = {name: [] for name in ABI_OPERATIONS}
    py_final_hashes: list[str] = []
    rust_final_hashes: list[str] = []
    serialization_parity = False
    rollback_parity = False
    restart_parity = False

    for trace_index, trace in enumerate(traces):
        py_kernel = AdaptiveStateKernel(capacity=3, history_capacity=32)
        rust_kernel = rust_class(capacity=3, history_capacity=32) if rust_class is not None else None
        rollback_version = 0
        for operation in trace:
            started = time.perf_counter_ns()
            py_result = _dispatch(py_kernel, operation)
            latency_ns[operation["op"]].append(time.perf_counter_ns() - started)
            rust_result = None if rust_kernel is None else _dispatch(rust_kernel, operation)
            operation_count += 1
            if operation["op"] == "promote":
                rollback_version = int(py_result["version_id"])
            if rust_result != py_result:
                parity_failures.append(
                    {
                        "operation": operation["op"],
                        "py_result": py_result,
                        "rust_result": rust_result,
                        "trace_index": trace_index,
                    }
                )
            if rust_kernel is not None:
                state_match = rust_kernel.canonical_state_json() == py_kernel.canonical_state_json()
                hash_match = rust_kernel.canonical_state_hash() == py_kernel.canonical_state_hash()
                bytes_match = bytes(rust_kernel.serialize()) == py_kernel.serialize()
                if not state_match or not hash_match or not bytes_match:
                    parity_failures.append(
                        {
                            "bytes_match": bytes_match,
                            "hash_match": hash_match,
                            "operation": operation["op"],
                            "state_match": state_match,
                            "trace_index": trace_index,
                        }
                    )
        py_final_hashes.append(py_kernel.canonical_state_hash())
        if rust_kernel is not None:
            rust_final_hashes.append(rust_kernel.canonical_state_hash())
            py_restored = AdaptiveStateKernel.restore(py_kernel.serialize())
            rust_restored = rust_class.restore(rust_kernel.serialize())
            serialization_parity = (
                py_restored.canonical_state_json() == rust_restored.canonical_state_json()
                and py_restored.canonical_state_hash() == rust_restored.canonical_state_hash()
                and py_restored.serialize() == bytes(rust_restored.serialize())
            )
            rollback_parity = py_restored.roll_back(rollback_version) == rust_restored.roll_back(
                rollback_version
            )
            restart_event = make_event(f"restart-{trace_index:02d}", 100 + trace_index, "addition", 1)
            for operation in (
                {"event": restart_event, "op": "apply_event"},
                {"event_id": restart_event["event_id"], "op": "acquire_core"},
                {"event_id": restart_event["event_id"], "op": "promote"},
            ):
                _dispatch(py_restored, operation)
                _dispatch(rust_restored, operation)
            restart_parity = (
                py_restored.canonical_state_json() == rust_restored.canonical_state_json()
            )

    invalid_receipt = _invalid_input_receipt()
    latency_summary = {
        name: {
            "count": len(values),
            "max_ns": max(values) if values else 0,
            "mean_ns": round(sum(values) / len(values), 3) if values else 0.0,
            "min_ns": min(values) if values else 0,
        }
        for name, values in latency_ns.items()
    }
    binding_available = rust_class is not None
    return {
        "binding_receipt": {
            "binding_available": binding_available,
            "class_name": "RustAdaptiveStateKernel",
            "methods": list(ABI_OPERATIONS),
        },
        "canonical_state_and_hash_parity": {
            "canonical_form_parity": binding_available and not parity_failures,
            "hash_parity": binding_available and py_final_hashes == rust_final_hashes,
            "py_final_hashes": py_final_hashes,
            "rust_final_hashes": rust_final_hashes,
        },
        "cross_language_operation_parity": {
            "accept_reject_parity": binding_available and not parity_failures,
            "operation_count": operation_count,
            "parity_failures": parity_failures,
            "trace_count": len(traces),
        },
        "invalid_input_and_capacity_controls": invalid_receipt,
        "per_operation_latency_receipts": {
            "claim": "descriptive_only_no_speedup_claim",
            "python_reference_ns": latency_summary,
        },
        "serialization_restart_and_rollback_parity": {
            "restart_parity": binding_available and restart_parity,
            "rollback_parity": binding_available and rollback_parity,
            "round_trip_parity": binding_available and serialization_parity,
        },
    }


def _invalid_input_receipt() -> JsonDict:
    kernel = AdaptiveStateKernel(capacity=2, history_capacity=8)
    event = make_event("invalid-base", 0, "addition", 1)
    accepted = kernel.apply_event(event)
    baseline_hash = kernel.canonical_state_hash()
    invalid_cases = [
        ("duplicate_event", lambda: kernel.apply_event(event), "DUPLICATE_EVENT"),
        (
            "out_of_order",
            lambda: kernel.apply_event(make_event("invalid-old", 0, "addition", 1)),
            "OUT_OF_ORDER_EVENT",
        ),
        (
            "overflow",
            lambda: kernel.apply_event(make_event("invalid-overflow", 2, "addition", U16_MAX + 1)),
            "FIXED_WIDTH_OVERFLOW",
        ),
        (
            "corrupted_checkpoint",
            lambda: AdaptiveStateKernel.restore(b"broken"),
            "checkpoint is not valid adaptive-state JSON",
        ),
    ]
    receipts = []
    for name, call, code in invalid_cases:
        try:
            result = call()
            observed_code = result["code"]
            preserved = kernel.canonical_state_hash() == baseline_hash
        except ValueError as exc:
            observed_code = str(exc)
            preserved = kernel.canonical_state_hash() == baseline_hash
        receipts.append(
            {
                "case": name,
                "expected_code": code,
                "observed_code": observed_code,
                "state_preserved": preserved,
            }
        )
    return {
        "accepted_control_event": accepted["accepted"],
        "capacity": kernel.canonical_state()["capacity"],
        "fail_closed": all(
            item["expected_code"] == item["observed_code"] and item["state_preserved"]
            for item in receipts
        ),
        "invalid_case_count": len(receipts),
        "receipts": receipts,
    }


def qualified_operation_mapping() -> JsonDict:
    """Map only Exp5858 accepted operation classes into the ABI."""

    return {
        "exp5858_changes": list(QUALIFIED_EXP5858_CHANGES),
        "mapping": {
            "addition": ["apply_event", "acquire_core", "promote"],
            "recurrence": ["apply_event", "select_replay"],
            "supersession": ["apply_event", "quarantine"],
        },
        "operations": list(ABI_OPERATIONS),
        "source": EXP5858_ROWS_RELATIVE_PATH.as_posix(),
    }


def abi_schema_and_bounds() -> JsonDict:
    """Expose every finite field bound needed by later hardware mapping."""

    return {
        "abi_version": ABI_VERSION,
        "canonical_encoding": "json_sort_keys_ascii_no_spaces_utf8",
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "deterministic_eviction": "oldest_promoted_version_then_event_id",
        "event_fields": list(EVENT_FIELDS),
        "fixed_width_fields": {
            "capacity": f"u32_bounded_1_to_{MAX_CAPACITY}",
            "chronology_index": "u32",
            "confidence_q16": "u16_fixed_point",
            "history_capacity": f"u32_bounded_2_to_{MAX_HISTORY_CAPACITY}",
            "version_id": "u32",
        },
        "max_event_id_len": MAX_EVENT_ID_LEN,
        "max_reason_len": MAX_REASON_LEN,
        "max_replay_limit": MAX_REPLAY_LIMIT,
        "state_schema": STATE_SCHEMA,
        "stable_ordering": "lexicographic_event_id_and_replay_confidence_desc",
    }


def implementation_receipts() -> tuple[JsonDict, JsonDict]:
    """Hash the Python and Rust implementation files."""

    py_path = REPO_ROOT / PY_MODULE_RELATIVE_PATH
    rust_path = REPO_ROOT / RUST_CORE_RELATIVE_PATH
    return (
        {
            "path": PY_MODULE_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(py_path),
            "state_owner": "readable_reference_semantics",
        },
        {
            "path": RUST_CORE_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(rust_path),
            "state_owner": "deterministic_deployable_execution",
        },
    )


def validate_artifact(artifact: JsonDict) -> bool:
    """Check terminal Exp5859 readiness without trusting prose fields."""

    required = set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    exits_ok = all(code == 0 for code in artifact.get("test_exit_codes", {}).values())
    return (
        required
        and artifact["adaptive_state_microkernel_ready_score"] in (0.0, 1.0)
        and artifact["inference_substrate"] == INFERENCE_SUBSTRATE
        and artifact["reproducibility_checksum"] == reproducibility_checksum(artifact)
        and (
            artifact["adaptive_state_microkernel_ready_score"] == 0.0
            or (
                artifact["status"] == "ready"
                and exits_ok
                and artifact["preconditions_checked"]["preconditions_ready"]
                and artifact["cross_language_operation_parity"]["accept_reject_parity"]
                and artifact["canonical_state_and_hash_parity"]["hash_parity"]
                and artifact["serialization_restart_and_rollback_parity"]["round_trip_parity"]
                and artifact["invalid_input_and_capacity_controls"]["fail_closed"]
            )
        )
    )


def reproducibility_checksum(artifact: JsonDict) -> str:
    """Hash every readiness field except the checksum slot itself."""

    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return sha256_json(payload)


def run(
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    duration_s: float | None = None,
    test_commands: list[str] | None = None,
    test_exit_codes: dict[str, int] | None = None,
    write: bool = True,
) -> JsonDict:
    """Build the terminal Exp5859 conformance artifact."""

    started = time.perf_counter()
    result_path = Path(result_path)
    test_commands = list(test_commands or [])
    test_exit_codes = dict(test_exit_codes or {})
    preconditions = collect_preconditions(result_path)
    parity = _run_parity_receipts()
    py_receipt, rust_receipt = implementation_receipts()
    exits_ok = all(code == 0 for code in test_exit_codes.values()) if test_exit_codes else False
    ready = (
        preconditions["preconditions_ready"]
        and exits_ok
        and parity["cross_language_operation_parity"]["accept_reject_parity"]
        and parity["canonical_state_and_hash_parity"]["hash_parity"]
        and parity["serialization_restart_and_rollback_parity"]["round_trip_parity"]
        and parity["serialization_restart_and_rollback_parity"]["rollback_parity"]
        and parity["serialization_restart_and_rollback_parity"]["restart_parity"]
        and parity["invalid_input_and_capacity_controls"]["fail_closed"]
    )
    artifact: JsonDict = {
        "abi_schema_and_bounds": abi_schema_and_bounds(),
        "adaptive_state_microkernel_ready_score": 1.0 if ready else 0.0,
        "binding_receipt": parity["binding_receipt"],
        "canonical_state_and_hash_parity": parity["canonical_state_and_hash_parity"],
        "cross_language_operation_parity": parity["cross_language_operation_parity"],
        "duration_s": round(duration_s if duration_s is not None else time.perf_counter() - started, 6),
        "field_provenance": {
            "field_principles": REQUIRED_FIELD_PRINCIPLES,
            "operation_trace_hash": sha256_json(
                [deterministic_fixture_trace()]
                + randomized_operation_traces(seed=5859, trace_count=4, events_per_trace=6)
            ),
            "result_path": RESULT_RELATIVE_PATH.as_posix(),
            "source_hashes": preconditions["file_hashes"],
        },
        "honest_verdict": (
            "parity: adaptive_state_microkernel_ready"
            if ready
            else "blocked: adaptive_state_microkernel_conformance_incomplete"
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "invalid_input_and_capacity_controls": parity["invalid_input_and_capacity_controls"],
        "per_operation_latency_receipts": parity["per_operation_latency_receipts"],
        "preconditions_checked": preconditions,
        "python_implementation_receipt": py_receipt,
        "qualified_operation_mapping": qualified_operation_mapping(),
        "rust_implementation_receipt": rust_receipt,
        "serialization_restart_and_rollback_parity": parity[
            "serialization_restart_and_rollback_parity"
        ],
        "status": "ready" if ready else "blocked",
        "test_commands": test_commands,
        "test_exit_codes": test_exit_codes,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    if write:
        _write_json_atomic(result_path, artifact)
    return artifact

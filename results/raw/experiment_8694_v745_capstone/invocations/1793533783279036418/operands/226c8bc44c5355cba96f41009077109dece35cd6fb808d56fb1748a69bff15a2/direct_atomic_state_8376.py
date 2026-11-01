"""REQ-VERIFY-8376: direct predictions and feedback share one durable version.

Immutable files let readers finish an older request while one writer publishes
the next complete state. These files establish process recovery only.
"""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Callable
import fcntl
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import continuous_local_learning_8348 as optimizer
from carnot.verify import spline_table_fidelity_8352 as evaluator
from carnot.verify.threshold_guard_8362 import validate

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = "openspec/change-proposals/v722-direct-service-protocol.json"
VERSION = "v722-direct-state-v1"
MAX_EVENTS = 256
MAX_BYTES = 2 * 1024 * 1024
BARRIERS = ("temporary_write", "file_fsync", "pointer_replace", "acknowledgment")


def initial(head: Json) -> Json:
    """Reject table metadata so this durable service has no approximation dependency."""
    if set(head) != {"arm", "coefficients", "temperature", "knots", "degree"}:
        raise ValueError("direct_head_fields")
    validate(head)
    if head["knots"] != evaluator.kernel.KNOTS or head["degree"] != 3:
        raise ValueError("frozen_basis")
    return dict(
        schema=VERSION,
        version=0,
        head=deepcopy(head),
        issued={},
        pending=[],
        release_cursor=0,
        applied={},
        events=[],
    )


def transition(state: Json, event: Json) -> tuple[Json, Json]:
    """A replayed feedback ID returns its saved result before any coefficient write."""
    kind = event["kind"]
    required = (
        {"kind", "id", "x", "slot"}
        if kind == "issue"
        else {"kind", "id", "issue_id", "issued_version", "clock", "y"}
    )
    if kind not in {"issue", "feedback"} or set(event) != required:
        raise ValueError("event_fields")
    identity = event["id"]
    if not isinstance(identity, str) or not 1 <= len(identity) <= 64:
        raise ValueError("event_identity")
    ledger = state["issued"] if kind == "issue" else state["applied"]
    if identity in ledger:
        prior = ledger[identity]
        if prior["event"] != event:
            raise ValueError("duplicate_conflict")
        return state, deepcopy(prior)
    if len(state["events"]) >= MAX_EVENTS:
        raise ValueError("event_budget")
    new = deepcopy(state)
    new["version"] += 1
    if kind == "issue":
        x = None if event["x"] is None else np.asarray(event["x"], dtype=np.float64)
        validate(new["head"], x)
        if type(event["slot"]) is not int or event["slot"] != len(new["issued"]) + 1:
            raise ValueError("issue_order")
        p = None if x is None else float(evaluator.direct(new["head"], x[None])[0])
        result = dict(
            event=deepcopy(event),
            version=new["version"],
            probability=p,
            action="escalate" if p is None else evaluator.kernel.action(p),
            head_hash=canonical_hash(new["head"]),
        )
        new["issued"][identity] = result
        new["pending"].append(identity)
    else:
        issue = new["issued"].get(event["issue_id"])
        if (
            issue is None
            or event["issue_id"] not in new["pending"]
            or event["issued_version"] != issue["version"]
            or event["clock"] != issue["event"]["slot"] + 8
            or issue["event"]["slot"] != new["release_cursor"] + 1
            or (
                event["y"] is not None and (type(event["y"]) is not int or event["y"] not in (0, 1))
            )
        ):
            raise ValueError("stale_or_invalid_feedback")
        change = optimizer.learn(new["head"], issue["event"]["x"], event["y"], "online_sparse")
        new["head"]["coefficients"] = change["coefficients"]
        new["pending"].remove(event["issue_id"])
        new["release_cursor"] += 1
        result = dict(
            event=deepcopy(event),
            version=new["version"],
            head_hash=canonical_hash(new["head"]),
            changed=change["changed"],
        )
        new["applied"][identity] = result
    new["events"].append(deepcopy(event))
    return new, deepcopy(result)


def trace(seed: int, head: Json) -> Json:
    """Constructed delayed labels qualify transactions without reopening utility studies."""
    rng = random.Random(seed)
    events: list[Json] = []
    versions: dict[int, int] = {}
    for clock in range(1, 25):
        if clock <= 16:
            versions[clock] = len(events) + 1
            events.append(
                dict(
                    kind="issue",
                    id=f"issue-{clock}",
                    slot=clock,
                    x=[rng.uniform(-2, 2), *[rng.random() for _ in range(4)]],
                )
            )
        if clock > 8:
            slot = clock - 8
            events.append(
                dict(
                    kind="feedback",
                    id=f"feedback-{slot}",
                    issue_id=f"issue-{slot}",
                    issued_version=versions[slot],
                    clock=clock,
                    y=slot % 2,
                )
            )
    return dict(seed=seed, head=deepcopy(head), events=events, schema=VERSION, crash_event_index=15)


def fold(frozen: Json, stop: int | None = None) -> Json:
    """Reconstruct causal meaning independently of checkpoint hashes and process timing."""
    state = initial(frozen["head"])
    for event in frozen["events"][:stop]:
        state, _ = transition(state, event)
    return state


def sync_directory(path: Path) -> None:
    """Directory metadata must reach storage after each rename, before acknowledgment."""
    descriptor = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class Store:
    """One writer replaces a pointer; readers keep their complete immutable snapshot."""

    def __init__(self, path: Path, frozen: Json):
        self.path, self.frozen = path, frozen
        path.mkdir(parents=True, exist_ok=True, mode=0o700)

    def read(self) -> Json:
        """Replaying issued events detects corruption even when a checksum is repaired."""
        pointer = json.loads((self.path / "current.json").read_bytes())
        if Path(pointer["file"]).name != pointer["file"]:
            raise ValueError("pointer_path")
        snapshot = self.path / pointer["file"]
        if snapshot.stat().st_size > MAX_BYTES:
            raise ValueError("state_byte_budget")
        envelope = json.loads(snapshot.read_bytes())
        state: Json = envelope["state"]
        if envelope["sha256"] != canonical_hash(state) or pointer["sha256"] != envelope["sha256"]:
            raise ValueError("state_checksum")
        if (
            state != fold(self.frozen, len(state["events"]))
            or state["events"] != self.frozen["events"][: len(state["events"])]
        ):
            raise ValueError("state_semantics")
        return state

    def commit(self, state: Json, hook: Callable[[str], None] = lambda phase: None) -> None:
        """The acknowledgment follows file and directory durability of the whole state."""
        digest = canonical_hash(state)
        encoded = json.dumps(dict(state=state, sha256=digest), sort_keys=True).encode()
        if len(encoded) > MAX_BYTES:
            raise ValueError("state_byte_budget")
        target = self.path / ("version-" + digest[7:] + ".json")
        temporary = self.path / ".state.tmp"
        with temporary.open("wb") as stream:
            stream.write(encoded)
            hook("temporary_write")
            stream.flush()
            os.fsync(stream.fileno())
            hook("file_fsync")
        temporary.replace(target)
        sync_directory(self.path)
        atomic_json(self.path / "current.json", dict(file=target.name, sha256=digest))
        sync_directory(self.path)
        hook("pointer_replace")
        hook("acknowledgment")

    def initialize(self) -> None:
        """Initialization is explicit; disappearance of an existing pointer is an error."""
        with (self.path / "writer.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not (self.path / "current.json").exists():
                if any(self.path.glob("version-*.json")):
                    raise FileNotFoundError("missing_durable_pointer")
                self.commit(initial(self.frozen["head"]))

    def apply(self, event: Json, hook: Callable[[str], None] = lambda phase: None) -> Json:
        """The writer lock covers reading, learning and durable publication together."""
        with (self.path / "writer.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            before = self.read()
            after, result = transition(before, event)
            if after != before:
                if event != self.frozen["events"][len(before["events"])]:
                    raise ValueError("trace_order")
                self.commit(after, hook)
            return result

    def cleanup(self, *, interrupt: bool = False) -> int:
        """Only abandoned temporary writes are removed; pinned versions stay readable."""
        removed = 0
        with (self.path / "writer.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for path in sorted(self.path.glob(".*.tmp")):
                path.unlink()
                removed += 1
                if interrupt:
                    raise RuntimeError("interrupted_cleanup")
            sync_directory(self.path)
        return removed

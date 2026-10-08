"""REQ-VERIFY-8263: one role-bound server lifetime preserves every intended slot.

The peer is supplied by later GPU tasks. Qualification uses scripted peers with
real serialized requests and parsed replies; it never pretends to acquire a model.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
import json
from pathlib import Path
import time
from typing import Any

from carnot.reporting.current_work_receipt import canonical_hash, atomic_json
from carnot.verify.calibrated_memory_trajectory_8211 import append as append, journal as journal
from carnot.verify.focal_protocol_8263 import parse

Json = dict[str, Any]


def capture(
    slots: list[Json], path: Path, server: AbstractContextManager[Callable[[str], str]]
) -> Json:
    """Bind exact slot/role/request bytes before dispatch and checkpoint eight sources."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plan_hash = canonical_hash(slots)
    events = journal(path)
    if events and (
        not path.read_bytes().endswith(b"\n") or any(e["plan_hash"] != plan_hash for e in events)
    ):
        raise ValueError("capture_resume_key")
    done = {e["slot"]: e["row"] for e in events if e["kind"] == "reply"}
    issued = {e["slot"]: e for e in events if e["kind"] == "issue"}
    if len(done) != sum(e["kind"] == "reply" for e in events):
        raise ValueError("duplicate_reply")
    start = time.monotonic_ns()
    with server as dispatch:
        for index, slot in enumerate(slots):
            if index in done:
                continue
            key = canonical_hash(
                dict(role=slot["role"], unit_id=slot["unit_id"], view=slot["view"])
            )
            if index in issued and issued[index]["resume_key"] != key:
                raise ValueError("source_role_drift")
            if index not in issued:
                append(path, dict(kind="issue", slot=index, plan_hash=plan_hash, resume_key=key))
            before = time.monotonic_ns()
            print(
                f"[exp8263] phase=before_generation completed={len(done)} pending={len(slots) - len(done)}",
                flush=True,
            )
            error = None
            transcript = None
            try:
                wire = json.dumps(
                    dict(
                        resume_key=key,
                        role=slot["role"],
                        mode=slot.get("mode", "valid"),
                        request=slot["view"]["request"],
                    ),
                    separators=(",", ":"),
                )
                reply = json.loads(dispatch(wire))
                transcript = reply["text"]
                if reply["resume_key"] != key or reply["role"] != slot["role"]:
                    raise ValueError("reply_binding")
                parsed = parse(slot["view"], reply["text"], canonical_hash(slot["view"]["request"]))
            except (TimeoutError, OSError, ValueError, KeyError) as exc:
                parsed, error = dict(status="escalated", rows=[], semantic_gold=False), str(exc)
            row = dict(
                slot=index,
                unit_id=slot["unit_id"],
                role=slot["role"],
                status=parsed["status"],
                parsed=parsed,
                error=error,
                transcript=transcript,
                resume_key=key,
                started_monotonic_ns=before,
                ended_monotonic_ns=time.monotonic_ns(),
            )
            append(path, dict(kind="reply", slot=index, plan_hash=plan_hash, row=row))
            done[index] = row
            if (index + 1) % 8 == 0 or index + 1 == len(slots):
                atomic_json(
                    path.with_suffix(".checkpoint.json"),
                    dict(plan_hash=plan_hash, rows=[done[k] for k in sorted(done)]),
                )
            print(
                f"[exp8263] phase=after_generation completed={len(done)} pending={len(slots) - len(done)}",
                flush=True,
            )
    return dict(
        rows=[done[k] for k in sorted(done)],
        intended_count=len(slots),
        completed_count=sum(r["status"] == "completed" for r in done.values()),
        failed_count=sum(r["status"] != "completed" for r in done.values()),
        server_lifetimes=1,
        intended_slots=slots,
        plan_hash=plan_hash,
        phase_spans=[
            dict(
                phase="capture", started_monotonic_ns=start, ended_monotonic_ns=time.monotonic_ns()
            )
        ],
    )


class PipePeer:
    """Bound scripted child replies and clean their process group after one lifetime."""

    def __init__(self, argv: list[str], deadline: float = 10):
        self.argv, self.deadline = argv, deadline
        self.first = True

    def __enter__(self) -> Callable[[str], str]:
        import subprocess

        self.child = subprocess.Popen(
            self.argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )
        return self.dispatch

    def dispatch(self, wire: str) -> str:
        import select

        assert self.child.stdin is not None and self.child.stdout is not None
        self.child.stdin.write(wire + "\n")
        self.child.stdin.flush()
        deadline = 20 if self.first else self.deadline
        self.first = False
        if not select.select([self.child.stdout], [], [], deadline)[0]:
            raise TimeoutError("peer_deadline")
        return str(self.child.stdout.readline())

    def __exit__(self, *args: Any) -> None:
        import os
        import signal
        import subprocess

        if self.child.poll() is None:
            os.killpg(self.child.pid, signal.SIGTERM)
        try:
            self.child.wait(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(self.child.pid, signal.SIGKILL)
            self.child.wait(timeout=2)
        for stream in [self.child.stdin, self.child.stdout]:
            if stream is not None:
                stream.close()


def peer() -> int:
    """Serve private protocol controls; these responses have no model provenance."""
    import sys
    import os

    for line in sys.stdin:
        value = json.loads(line)
        mode = value.get("mode", "valid")
        if mode == "drop":
            return 17
        if mode == "timeout":
            # The parent deliberately tests a bounded unresponsive child.
            os.read(0, 1)
        focal = value["request"]["sentence_indices"][0]
        text = f"{focal}|B|0.50|[]"
        if mode == "partial":
            text = f"{focal}|B|"
        if mode == "duplicate":
            text += "\n" + text
        print(
            json.dumps(
                dict(
                    resume_key=value["resume_key"],
                    role="drift" if mode == "role" else value["role"],
                    text=text,
                )
            ),
            flush=True,
        )
    return 0

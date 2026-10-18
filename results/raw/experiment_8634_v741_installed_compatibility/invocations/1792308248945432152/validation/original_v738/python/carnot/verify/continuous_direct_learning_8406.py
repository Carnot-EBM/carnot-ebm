"""REQ-VERIFY-8406: delayed labels change only the shipped sparse numeric head."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import time
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.direct_state_qualification_8404 import QualifiedStore
from carnot.verify import direct_atomic_state_8376 as s

from carnot.verify.continuous_local_learning_8348 import Labels, probability as probability

Json = dict[str, Any]
VARIANTS = ("in_memory_uninterrupted", "durable_uninterrupted", "durable_crash_resume")
SEEDS = (11, 22, 33)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush counts so a supervisor can distinguish progress from a stalled worker."""
    print(f"[exp8406] phase={phase} completed={completed} pending={pending}", flush=True)


def read_steps(directory: Path) -> list[Json]:
    """Complete per-slot states make timing and semantic evidence independently replayable."""
    return [json.loads(p.read_bytes()) for p in sorted((directory / "steps").glob("*.json"))]


def issue(bundle: Json, slot: int) -> Json:
    """Source identity is separate from the short transaction identity used by the store."""
    return dict(kind="issue", id=f"issue-{slot}", slot=slot, x=bundle["slots"][slot - 1]["x"])


def feedback(state: Json, slot: int, y: int | None) -> Json:
    """The receipt names the exact immutable issue version and its due clock."""
    return dict(
        kind="feedback",
        id=f"feedback-{slot}",
        issue_id=f"issue-{slot}",
        issued_version=state["issued"][f"issue-{slot}"]["version"],
        clock=slot + 8,
        y=y,
    )


def check_steps(bundle: Json, steps: list[Json]) -> Json:
    """Reconstruct every prediction and update from sealed operands, not reported totals."""
    state = s.initial(bundle["head"])
    vault = Labels(Path(bundle["labels"]["path"]), bundle["slots"])
    for slot, step in enumerate(steps, 1):
        state, receipt = s.transition(state, issue(bundle, slot))
        if step["slot"] != slot or step["issue_state"] != state or step["issue"] != receipt:
            raise ValueError("issued_state_drift")
        if slot > 8:
            y = vault.release(slot - 8, slot, sealed=True)["y"]
            state, _ = s.transition(state, feedback(state, slot - 8, y))
        if state != step["state"] or step["update_latency_ns"] < 0:
            raise ValueError("primitive_transition_drift")
    return dict(state)


def run(bundle: Json, directory: Path, variant: str, *, crash: int = 0, stop: int = 96) -> Json:
    """The original store acknowledges whole versions before feedback or process exit."""
    if variant not in VARIANTS or crash not in (0, 32, 64) or not 1 <= stop <= 96:
        raise ValueError("service_configuration")
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    steps = read_steps(directory)
    state = check_steps(bundle, steps)
    frozen = dict(head=bundle["head"], events=deepcopy(state["events"]))
    store = None if variant == VARIANTS[0] else QualifiedStore(directory / "store", frozen)
    if store is not None:
        store.initialize()
        if store.read() != state:
            raise ValueError("restart_state_drift")
    vault = Labels(Path(bundle["labels"]["path"]), bundle["slots"])
    began = time.monotonic()

    def apply(event: Json) -> Json:
        nonlocal state
        frozen["events"].append(event)
        if store is None:
            state, result = s.transition(state, event)
        else:
            result = store.apply(event)
            # The acknowledged result and the same shipped transition advance
            # the local view without an extra disk replay for each event.
            state, expected = s.transition(state, event)
            if result != expected:
                raise ValueError("acknowledged_state_drift")
        return dict(result)

    for slot in range(len(steps) + 1, stop + 1):
        if time.monotonic() - began > 600:
            raise TimeoutError("worker_deadline")
        issued = apply(issue(bundle, slot))
        issue_state = deepcopy(state)
        started = time.monotonic_ns()
        if slot > 8:
            target = vault.release(slot - 8, slot, sealed=True)
            apply(feedback(state, slot - 8, target["y"]))
        elapsed = time.monotonic_ns() - started
        if state["issued"][f"issue-{slot}"] != issued:
            raise ValueError("issued_prediction_mutation")
        atomic_json(
            directory / "steps" / f"{slot:03}.json",
            dict(
                slot=slot,
                issue=issued,
                issue_state=issue_state,
                state=state,
                update_latency_ns=elapsed,
            ),
        )
        if slot % 16 == 0:
            progress("service_" + variant, slot, stop - slot)
        if slot == crash:
            progress("durable_slot_then_genuine_exit73", slot, stop - slot)
            os._exit(73)
    atomic_json(directory / "final.json", state)
    return state


def controls(bundle: Json, state: Json) -> list[Json]:
    """Invalid feedback cannot write; an identical retry returns its saved result."""
    rows = []
    vault = Labels(Path(bundle["labels"]["path"]), bundle["slots"])
    for name, args in [
        ("future", dict(slot=1, clock=8, sealed=True)),
        ("unsealed", dict(slot=1, clock=9, sealed=False)),
    ]:
        try:
            vault.release(**args)
        except ValueError:
            rows.append(
                dict(name=name, attempted_access=args, passed=True, rejected_before_decode=True)
            )
    first = state["applied"]["feedback-1"]["event"]
    after, saved = s.transition(state, first)
    rows.append(
        dict(
            name="identical_retry",
            event=first,
            before_hash=canonical_hash(state),
            after_hash=canonical_hash(after),
            passed=after == state and saved == state["applied"]["feedback-1"],
        )
    )
    for name, event in [
        ("duplicate_conflict", dict(first, y=1 - (first["y"] or 0))),
        ("stale", dict(first, id="stale")),
        ("reordered", feedback(state, len(state["applied"]) + 2, 0)),
    ]:
        before = canonical_hash(state)
        try:
            s.transition(state, event)
        except ValueError as error:
            rows.append(
                dict(
                    name=name,
                    event=event,
                    before_hash=before,
                    after_hash=canonical_hash(state),
                    observed_error=str(error),
                    passed=canonical_hash(state) == before,
                    rejected=True,
                )
            )
    if len(rows) != 6:
        raise ValueError("tripwire_did_not_fire")
    return rows

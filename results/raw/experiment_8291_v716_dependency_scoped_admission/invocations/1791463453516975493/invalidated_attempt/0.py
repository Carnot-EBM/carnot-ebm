"""REQ-VERIFY-8291: exact Boolean fixtures test mechanics, not GRACE soundness.

Implications derive a consequent from any true antecedent; equality propagates
truth both ways. Boolean constraints protect fixed values or require a Boolean.
These explicit equations need no semantic extraction, model, or learned judge.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import random
import signal
import time
from typing import Any

from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]
ARMS = ["full", "scoped", "one_hop"]


def manifest() -> Json:
    """Freeze topology and delayed labels before costs can influence the roster."""
    graphs = []
    for seed in [7161, 7162, 7163]:
        for size in [32, 128]:
            for topology in ["chain", "sparse_dag", "cyclic", "dense"]:
                key = f"{seed}-{size}-{topology}"
                rng = random.Random(seed + size)
                width = size // 4 if topology in ["chain", "sparse_dag"] else size - 7
                constraints = []
                for i in range(width - 1):
                    reads = [f"x{i}"]
                    if topology == "sparse_dag" and i > 1:
                        reads.append(f"x{rng.randrange(i)}")
                    if topology == "dense":
                        reads = [f"x{j}" for j in range(width) if j != i + 1]
                    equality = topology == "cyclic"
                    constraints.append(
                        dict(
                            id=f"edge{i}",
                            kind="equality" if equality else "implication",
                            reads=reads + [f"x{i + 1}"] if equality else reads,
                            writes=[f"x{i}", f"x{i + 1}"] if equality else [f"x{i + 1}"],
                        )
                    )
                constraints.append(
                    dict(
                        id="goal",
                        kind="boolean",
                        reads=[f"x{width - 1}"],
                        writes=[],
                        expected=False,
                    )
                )
                constraints.extend(
                    [
                        dict(id="safe0", kind="implication", reads=["z0"], writes=["z1"]),
                        dict(id="safe1", kind="equality", reads=["z1", "z2"], writes=["z1", "z2"]),
                        dict(id="safe2", kind="boolean", reads=["z2"], writes=[], expected=None),
                    ]
                )
                for name in ["y0", "y1"]:
                    constraints.append(
                        dict(id=name, kind="boolean", reads=[name], writes=[], expected=None)
                    )
                for i in range(size - len(constraints)):
                    constraints.append(
                        dict(
                            id=f"retention{i}",
                            kind="boolean",
                            reads=[f"r{i}"],
                            writes=[],
                            expected=False,
                        )
                    )
                events = []
                for t in range(64):
                    kind = t % 8
                    source = f"{key}:{t - 1 if kind == 5 else t}"
                    var = "x0" if kind == 0 else "z0" if kind in [1, 2, 3, 6] else "y1"
                    events.append(
                        dict(
                            id=t,
                            source=source,
                            op="invalidate" if kind == 3 else "add",
                            variable=var,
                            value=kind != 2,
                            target=f"{key}:1",
                            reads=None if kind == 4 else [var],
                            writes=None if kind == 4 else [var],
                            version=-1 if kind == 6 else None,
                            label=int(kind != 7),
                            eligible_time=t + 8,
                        )
                    )
                graphs.append(
                    equations(
                        dict(
                            id=key,
                            seed=seed,
                            size=size,
                            topology=topology,
                            constraints=constraints,
                            events=events,
                        )
                    )
                )
    return dict(
        seeds=[7161, 7162, 7163],
        graphs=graphs,
        delay=8,
        repetitions=5,
        warmups=1,
        alternating_order=True,
        cpu_deadline_s=600,
        validation_deadline_s=900,
        constraint_fraction_threshold=0.5,
        time_ratio_threshold=1.0,
        H3="zero scoped/full event disagreements; exact crash "
        "parity; planted transitive conflict detected",
        generator_weight_updates=0,
    )


def variables(graph: Json) -> list[str]:
    """Equations retain their operands even when declared metadata is absent."""
    return sorted({v for c in graph["constraints"] for v in c["operands"]})


def equations(graph: Json) -> Json:
    """Separate semantic operands from metadata so missing metadata stays conservative."""
    result = deepcopy(graph)
    for c in result["constraints"]:
        c.setdefault("operands", list(c["reads"] or []) + list(c["writes"] or []))
    return result


def reference(graph: Json, roots: Json) -> tuple[Json, list[str], int]:
    """Independently rescan every equation until no truth value can change."""
    values = dict.fromkeys(variables(graph), False) | roots
    count = 0
    changed = True
    while changed:
        changed = False
        for c in graph["constraints"]:
            count += 1
            operands = c["operands"]
            targets = operands if c["kind"] == "equality" else operands[-1:]
            enabled = any(
                values[v] for v in (operands if c["kind"] == "equality" else operands[:-1])
            )
            if c["kind"] != "boolean" and enabled:
                for target in targets:
                    if not values[target]:
                        values[target] = True
                        changed = True
    conflicts = []
    for c in graph["constraints"]:
        count += 1
        operands = c["operands"]
        valid = (
            (
                all(type(values[v]) is bool for v in operands)
                and (c["expected"] is None or values[operands[0]] == c["expected"])
            )
            if c["kind"] == "boolean"
            else (len({values[v] for v in operands}) == 1)
            if c["kind"] == "equality"
            else (not any(values[v] for v in operands[:-1]) or values[operands[-1]])
        )
        if not valid:
            conflicts.append(c["id"])
    return values, conflicts, count


def check(graph: Json, roots: Json, event: Json, arm: str, committed: Json | None = None) -> Json:
    """Closure follows read/write edges to a fixed point; one-hop is intentionally unsafe."""
    if arm not in ARMS:
        raise ValueError("arm")
    start = time.monotonic_ns()
    graph = equations(graph)
    fallback = (
        "missing_event_metadata"
        if event["reads"] is None or event["writes"] is None
        else (
            "missing_constraint_metadata"
            if any(c["reads"] is None or c["writes"] is None for c in graph["constraints"])
            else None
        )
    )
    touched = set(event["writes"] or [])
    selected: set[str] = set()
    changed = True
    while changed:
        changed = False
        for c in graph["constraints"]:
            if touched.intersection(c["reads"] or []) and c["id"] not in selected:
                selected.add(c["id"])
                if arm != "one_hop":
                    touched.update(c["writes"] or [])
                    changed = True
        if arm == "one_hop":
            break
    if arm == "full" or fallback:
        selected = {c["id"] for c in graph["constraints"]}
    closure_end = time.monotonic_ns()
    # Scoped derivation scans selected rules; untouched components keep their roots.
    if arm == "full" or fallback or arm == "one_hop":
        values, all_conflicts, scans = reference(graph, roots)
        conflicts = [c for c in all_conflicts if c in selected]
    else:
        values = dict(committed or dict.fromkeys(variables(graph), False))
        for variable in touched:
            values[variable] = roots.get(variable, False)
        values.update(roots)
        scans = 0
        changed = True
        while changed:
            changed = False
            for c in graph["constraints"]:
                if c["id"] in selected and c["kind"] != "boolean":
                    scans += 1
                    operands = c["operands"]
                    targets = c["writes"]
                    enabled = any(values[v] for v in c["reads"])
                    if enabled:
                        for target in targets:
                            if not values[target]:
                                values[target] = True
                                changed = True
        # Closure proves untouched components keep their already committed values.
        conflicts = []
        for c in graph["constraints"]:
            if c["id"] in selected:
                scans += 1
                operands = c["operands"]
                invalid = (
                    (
                        c["expected"] is not None
                        and values[operands[0]] != c["expected"]
                        or any(type(values[v]) is not bool for v in operands)
                    )
                    if c["kind"] == "boolean"
                    else (
                        any(values[v] for v in c["reads"])
                        and not all(values[v] for v in c["writes"])
                    )
                )
                if invalid:
                    conflicts.append(c["id"])
    end = time.monotonic_ns()
    return dict(
        values=values,
        conflicts=conflicts,
        dependencies=sorted(touched),
        selected=sorted(selected),
        closure_size=len(selected),
        fallback_reason=fallback,
        evaluated_constraint_count=len(selected),
        scan_count=scans,
        closure_ns=closure_end - start,
        replay_ns=end - closure_end,
    )


def initial(graph: Json) -> Json:
    """The journal, rather than a checkpoint, owns admitted and pending state."""
    return dict(
        soft={},
        values=reference(equations(graph), {})[0],
        seen=[],
        version=0,
        issues=[],
        releases=[],
    )


def snapshot(state: Json) -> str:
    """Bind actual committed state while keeping implementation costs separate."""
    return canonical_hash({k: state[k] for k in ["soft", "values", "seen", "version"]})


def release(graph: Json, state: Json, event: Json, now: int, arm: str) -> Json:
    """Only delayed eligible unique feedback may admit or retract a soft fact."""
    start = time.monotonic_ns()
    soft = deepcopy(state["soft"])
    reason = "eligible"
    retracted = []
    if now != event["eligible_time"] or len(state["issues"]) <= now:
        raise ValueError("feedback_clock")
    if event["source"] in state["seen"]:
        reason = "duplicate_source"
    elif event["version"] is not None and event["version"] != state["version"]:
        reason = "stale_feedback"
    elif event["label"] != 1:
        reason = "negative_feedback"
    elif event["op"] == "invalidate":
        if event["target"] in soft:
            retracted = [event["target"]]
            del soft[event["target"]]
        else:
            reason = "absent_invalidation"
    elif any(
        v["variable"] == event["variable"] and v["value"] != event["value"] for v in soft.values()
    ):
        reason = "contradictory_soft_addition"
    else:
        soft[event["source"]] = dict(variable=event["variable"], value=event["value"])
    roots = {v["variable"]: v["value"] for v in soft.values()}
    checked = check(graph, roots, event, arm, state["values"])
    accepted = reason == "eligible" and not checked["conflicts"]
    if accepted:
        state["soft"], state["values"] = soft, checked["values"]
        state["version"] += 1
    else:
        retracted = []
    if event["source"] not in state["seen"]:
        state["seen"].append(event["source"])
    return dict(
        checked,
        id=event["id"],
        source=event["source"],
        eligible_time=now,
        accepted=accepted,
        reason=reason,
        retracted=retracted,
        admitted_set=sorted(state["soft"]),
        post_state_sha256=snapshot(state),
        compute_ns=time.monotonic_ns() - start,
    )


def semantic(state: Json) -> Json:
    """Compare causal decisions and state without comparing clocks or arm cost."""
    keys = [
        "id",
        "source",
        "eligible_time",
        "accepted",
        "reason",
        "retracted",
        "admitted_set",
        "post_state_sha256",
        "conflicts",
    ]
    return dict(
        soft=state["soft"],
        values=state["values"],
        seen=state["seen"],
        version=state["version"],
        issues=state["issues"],
        releases=[{k: r[k] for k in keys} for r in state["releases"]],
    )


def issue(graph: Json, state: Json, slot: int) -> Json:
    """A later issue can use only state committed by prior eligible feedback."""
    event = graph["events"][slot] if slot < 64 else dict(id=slot, source="retention")
    return dict(
        id=slot,
        source=event["source"],
        event_sha256=canonical_hash({k: v for k, v in event.items() if k != "label"}),
        state_sha256=snapshot(state),
        decision=canonical_hash(
            [state["values"]["z2"], state["values"]["y1"], sorted(state["soft"])]
        ),
    )


def append(path: Path, kind: str, row: Json) -> None:
    """Fsync each issue before due labels can alter committed memory."""
    record = dict(kind=kind, row=row)
    record["sha256"] = canonical_hash(record)
    with path.open("ab") as stream:
        stream.write(json.dumps(record, sort_keys=True).encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def load(graph: Json, arm: str, path: Path) -> Json:
    """Rebuild decisions from immutable equations, rejecting even rehashed drift."""
    state = initial(graph)
    data = path.read_bytes() if path.exists() else b""
    if data and not data.endswith(b"\n"):
        raise ValueError("partial_record")
    for line in data.splitlines():
        record = json.loads(line)
        row = record["row"]
        if record["sha256"] != canonical_hash({k: v for k, v in record.items() if k != "sha256"}):
            raise ValueError("journal_drift")
        if record["kind"] == "issue":
            expected = issue(graph, state, len(state["issues"]))
            if expected != row:
                raise ValueError("journal_drift")
            state["issues"].append(row)
        elif record["kind"] == "release":
            expected = release(
                graph, state, graph["events"][len(state["releases"])], len(state["issues"]) - 1, arm
            )
            ignored = {
                "closure_ns",
                "replay_ns",
                "compute_ns",
                "persistence_ns",
                "total_ns",
                "started_monotonic_ns",
                "ended_monotonic_ns",
            }
            if {k: v for k, v in expected.items() if k not in ignored} != {
                k: v for k, v in row.items() if k not in ignored
            }:
                raise ValueError("journal_drift")
            state["releases"].append(row)
        else:
            raise ValueError("journal_kind")
    return state


def execute(graph: Json, arm: str, path: Path, crash: int = -1) -> Json:
    """Resume an issue killed before feedback, then release it before the next issue."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = load(graph, arm, path)

    def due(now: int, transaction_start: int = 0, issue_persistence: int = 0) -> None:
        origin = now - 8
        if 0 <= origin < 64 and origin == len(state["releases"]):
            start = transaction_start or time.monotonic_ns()
            row = release(graph, state, graph["events"][origin], now, arm)
            persist_start = time.monotonic_ns()
            append(path, "release", row)
            row.update(
                persistence_ns=issue_persistence + time.monotonic_ns() - persist_start,
                started_monotonic_ns=start,
                ended_monotonic_ns=time.monotonic_ns(),
            )
            row["total_ns"] = row["ended_monotonic_ns"] - start
            # The timing receipt is separately fsynced, so transaction cost includes its work.
            append(path.with_suffix(path.suffix + ".costs"), "cost", row)
            state["releases"].append(row)

    if state["issues"]:
        due(len(state["issues"]) - 1)
    for slot in range(len(state["issues"]), 72):
        transaction_start = time.monotonic_ns()
        row = issue(graph, state, slot)
        issue_persist_start = time.monotonic_ns()
        append(path, "issue", row)
        issue_persistence = time.monotonic_ns() - issue_persist_start
        state["issues"].append(row)
        if slot < 8:
            end = time.monotonic_ns()
            append(
                path.with_suffix(path.suffix + ".prefix.costs"),
                "prefix",
                dict(
                    id=slot,
                    started_monotonic_ns=transaction_start,
                    ended_monotonic_ns=end,
                    total_ns=end - transaction_start,
                    persistence_ns=issue_persistence,
                ),
            )
        if slot == crash:
            import coverage

            current = coverage.Coverage.current()
            if current is not None:
                current.save()
            print(
                f"[exp8291] phase=kill_boundary completed={slot + 1} pending={64 - len(state['releases'])}",
                flush=True,
            )
            os.kill(os.getpid(), signal.SIGKILL)
        due(slot, transaction_start, issue_persistence)
        if (slot + 1) % 24 == 0:
            print(
                f"[exp8291] phase=events completed={slot + 1} pending={72 - slot - 1}", flush=True
            )
    return state

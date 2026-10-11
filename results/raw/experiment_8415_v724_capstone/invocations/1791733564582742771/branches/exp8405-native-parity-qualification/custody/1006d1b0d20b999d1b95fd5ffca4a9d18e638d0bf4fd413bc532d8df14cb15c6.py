"""REQ-KAN-8306 / REQ-VERIFY-8306: local support needs exact cache bookkeeping.

The head is ordinary logistic regression on fixed spline features. Constructed
faults qualify numeric writes and recovery, without testing natural learning.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import random
import signal
import time
from typing import Any

from carnot.experiment_7425_v651_spline_prototype import cubic_basis, sigmoid
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8306_v717_local_update_isolation"
CLI = "scripts/experiments/" + NAME + ".py"
KNOTS = [0.0, 0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0]
ARMS = ["full", "indexed", "truncated"]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush counts so an external supervisor can observe each real boundary."""
    print(f"[exp8306] phase={phase} completed={completed} pending={pending}", flush=True)


@lru_cache(maxsize=4096)
def local_basis(value: float) -> tuple[float, ...]:
    """Reuse the tested numeric kernel; memoization changes no coefficients."""
    return tuple(float(v) for v in cubic_basis(value, KNOTS).values)


def design(x: list[float]) -> list[float]:
    """Keep holistic logit unbounded and reject invalid natural local features."""
    if len(x) != 5 or not all(math.isfinite(v) for v in x) or any(not 0 <= v <= 1 for v in x[1:]):
        raise ValueError("features")
    return [x[0], 1.0, *[v for feature in x[1:] for v in local_basis(feature)]]


def scalar_design(x: list[float]) -> list[float]:
    """Independent scalar recursion detects ordering and endpoint kernel errors."""

    def basis(i: int, degree: int, v: float) -> float:
        if degree == 0:
            return float(KNOTS[i] <= v < KNOTS[i + 1])
        left = KNOTS[i + degree] - KNOTS[i]
        right = KNOTS[i + degree + 1] - KNOTS[i + 1]
        return ((v - KNOTS[i]) / left * basis(i, degree - 1, v) if left else 0.0) + (
            (KNOTS[i + degree + 1] - v) / right * basis(i + 1, degree - 1, v) if right else 0.0
        )

    return [
        x[0],
        1.0,
        *[float(i == 7) if v == 1 else basis(i, 3, v) for v in x[1:] for i in range(8)],
    ]


def logit(coefficients: list[float], x: list[float], arm: str = "indexed") -> float:
    """Full dot product and active-only sum use the same ordered feature vector."""
    d = design(x)
    return sum(c * v for c, v in zip(coefficients, d, strict=True) if arm == "full" or v)


def action(p: float) -> str:
    """Threshold ties escalate because neither confident action is permitted."""
    return "accept" if p < 0.25 else "reject" if p > 0.75 else "escalate"


def static_fit() -> list[float]:
    """Train fixture slope before freezing it; this is an optimizer control only."""
    c = [1.0, 0.0, *([0.0] * 32)]
    for _ in range(20):
        gradient = [0.0] * 34
        for label in [0, 1]:
            x = [float(2 * label - 1), *([float(label)] * 4)]
            d = design(x)
            residual = sigmoid(logit(c, x)) - label
            gradient = [g + residual * v / 2 for g, v in zip(gradient, d, strict=True)]
        c = [min(4.0, max(-4.0, v - 0.01 * g)) for v, g in zip(c, gradient, strict=True)]
        c[0] = min(2.0, max(0.0, c[0]))
    return c


def manifest() -> Json:
    """Freeze all 48 oracle trajectories before timing or fault outcomes open."""
    trajectories = []
    patterns = [
        [0.0, 1.0],
        [0.2, 0.4, 0.6, 0.8],
        [0.199999, 0.200001, 0.399999, 0.400001],
        [0.05, 0.15, 0.25, 0.35, 0.65, 0.85],
    ]
    for seed in [7171, 7172, 7173]:
        for pattern, values in enumerate(patterns):
            for fault in range(4):
                rng = random.Random(seed + pattern * 10 + fault)
                events = []
                for t in range(64):
                    kind = (t + fault) % 8
                    x = [rng.uniform(-1, 1), *[values[(t + j) % len(values)] for j in range(4)]]
                    events.append(
                        dict(
                            id=str(t - 1 if kind == 5 else t),
                            x=x,
                            y=t % 2,
                            stale=kind == 6,
                            metadata=None if kind == 4 else -1 if kind == 7 else "current",
                            global_update=kind == 2,
                            rate=0.0 if kind == 3 else 0.01,
                        )
                    )
                c = static_fit()
                if pattern == 2:
                    c[2:] = [3.999 if i % 2 else -3.999 for i in range(32)]
                trajectories.append(
                    dict(
                        id=f"{seed}-{pattern}-{fault}",
                        seed=seed,
                        pattern=pattern,
                        fault=fault,
                        coefficients=c,
                        cache_x=[
                            [0.0, *([v] * 4)]
                            for v in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                        ],
                        events=events,
                    )
                )
    return dict(
        trajectories=trajectories,
        knots=KNOTS,
        parameter_count=34,
        delay=8,
        repetitions=5,
        warmups=1,
        measurement_deadline_s=600,
        online_decay=0,
        online_slope_frozen=True,
    )


def initial(trajectory: Json, arm: str = "indexed") -> Json:
    """Persist reverse dependencies alongside pending issues and deduplication IDs."""
    cache = {
        str(i): dict(x=x, p=sigmoid(logit(trajectory["coefficients"], x)), valid=True)
        for i, x in enumerate(trajectory["cache_x"])
    }
    index = {
        str(i): [key for key, row in cache.items() if design(row["x"])[i] != 0] for i in range(34)
    }
    return dict(
        coefficients=list(trajectory["coefficients"]),
        cache=cache,
        index=index,
        index_version=0,
        version=0,
        applied=[],
        pending=[],
        issues=[],
        releases=[],
        stale_cache_count=0,
        arm=arm,
        costs=[],
    )


def invalidate(state: Json, changed: list[int], arm: str) -> Json:
    """A global write costs every cache entry; untrusted metadata costs a full scan."""
    global_change = any(i < 2 for i in changed)
    fallback = state["index_version"] != state["version"]
    touched = 0
    selected: set[str] = set()
    if changed:
        if global_change or fallback:
            selected = set(state["cache"])
            touched = len(selected)
        elif arm == "full":
            for key, row in state["cache"].items():
                touched += 1
                if any(design(row["x"])[i] != 0 for i in changed):
                    selected.add(key)
        else:
            for i in changed:
                entries = state["index"][str(i)]
                touched += len(entries)
                selected.update(entries)
            if arm == "truncated" and selected:
                selected.remove(min(selected, key=int))
    for key in selected:
        state["cache"][key]["valid"] = False
    state["version"] += int(bool(changed))
    state["index_version"] = state["version"]
    return dict(
        invalidated=sorted(selected, key=int),
        invalidation_touches=touched,
        global_change=global_change,
        fallback=fallback,
    )


def update(state: Json, event: Json, arm: str) -> Json:
    """Both arms use the same gradient norm and projection, without global decay."""
    if arm not in ARMS:
        raise ValueError("arm")
    reason = (
        "duplicate"
        if event["id"] in state["applied"]
        else "stale"
        if event.get("stale")
        else "applied"
    )
    changed: list[int] = []
    projected = False
    visits = 0
    if reason == "applied":
        d = design(event["x"])
        residual = sigmoid(logit(state["coefficients"], event["x"], arm)) - event["y"]
        gradient = [
            0.0,
            residual if event.get("global_update") else 0.0,
            *[residual * v for v in d[2:]],
        ]
        scale = min(1.0, 1.0 / math.sqrt(sum(g * g for g in gradient))) if any(gradient) else 1.0
        for i, g in enumerate(gradient):
            if arm == "full" or g != 0:
                visits += 1
                proposed = state["coefficients"][i] - event.get("rate", 0.01) * g * scale
                clipped = min(4.0, max(-4.0, proposed))
                projected = projected or clipped != proposed
                if clipped != state["coefficients"][i]:
                    changed.append(i)
                    state["coefficients"][i] = clipped
        state["applied"].append(event["id"])
    if event.get("metadata", "current") != "current":
        state["index_version"] = -1
    result = invalidate(state, changed, arm)
    return dict(
        result,
        id=event["id"],
        reason=reason,
        changed=changed,
        coefficient_visits=visits,
        projection=projected,
        coefficients=list(state["coefficients"]),
    )


def issue(state: Json, slot: int) -> Json:
    """Refresh invalid cache values while preserving already issued probabilities."""
    key = str(slot % len(state["cache"]))
    row = state["cache"][key]
    expected = sigmoid(logit(state["coefficients"], row["x"]))
    if not row["valid"]:
        row["p"], row["valid"] = expected, True
    if abs(row["p"] - expected) > 1e-12:
        state["stale_cache_count"] += 1
    result = dict(
        slot=slot,
        cache_id=key,
        p=row["p"],
        action=action(row["p"]),
        coefficients=list(state["coefficients"]),
    )
    state["issues"].append(result)
    state["pending"].append(slot)
    return result


def release(trajectory: Json, state: Json, slot: int, arm: str) -> None:
    """Only due feedback follows its durable issue, and IDs update at most once."""
    origin = slot - 8
    if 0 <= origin < 64 and origin == len(state["releases"]):
        state["releases"].append(update(state, trajectory["events"][origin], arm))
        state["pending"].remove(origin)


def semantic(state: Json) -> Json:
    """Exclude clocks and traversal costs when comparing transaction meaning."""
    return {
        k: deepcopy(state[k])
        for k in ["coefficients", "applied", "pending", "issues", "stale_cache_count"]
    } | dict(
        releases=[
            {k: r[k] for k in ["id", "reason", "changed", "projection", "coefficients"]}
            for r in state["releases"]
        ]
    )


def load(trajectory: Json, path: Path) -> Json:
    """Recompute causal state so even a rehashed checkpoint mutation is rejected."""
    if not path.is_file():
        return initial(trajectory)
    record = json.loads(path.read_text())
    state = json.loads(record.get("encoded_state", "{}"))
    if record.get("sha256") != canonical_hash(state):
        raise ValueError("checkpoint_hash")
    expected = initial(trajectory, state["arm"])
    for slot in range(len(state["issues"])):
        issue(expected, slot)
        if slot < len(state["issues"]) - 1 or len(state["releases"]) > max(0, slot - 8):
            release(trajectory, expected, slot, state["arm"])
    fields = [
        "coefficients",
        "cache",
        "index",
        "index_version",
        "version",
        "applied",
        "pending",
        "issues",
        "releases",
        "stale_cache_count",
        "arm",
    ]
    normalized = lambda s: (
        {k: s[k] for k in fields if k != "releases"}
        | dict(
            releases=[
                {
                    k: v
                    for k, v in r.items()
                    if k not in ["invalidation_touches", "coefficient_visits"]
                }
                for r in s["releases"]
            ]
        )
    )
    if normalized(expected) != normalized(state):
        raise ValueError("checkpoint_semantics")
    return dict(state)


def save(path: Path, state: Json) -> None:
    """Keep complete state in one encoded value to avoid recursive writer overhead."""
    atomic_json(
        path,
        dict(
            encoded_state=json.dumps(state, sort_keys=True, separators=(",", ":")),
            sha256=canonical_hash(state),
        ),
    )


def execute(trajectory: Json, arm: str, path: Path, crash: int = -1, pause: int = -1) -> Json:
    """Atomic issue then release snapshots survive death between the two writes."""
    state = load(trajectory, path) if path.is_file() else initial(trajectory, arm)
    if state["issues"]:
        release(trajectory, state, len(state["issues"]) - 1, arm)
    for slot in range(len(state["issues"]), 72):
        began = time.monotonic_ns()
        issue(state, slot)
        issue_done = time.monotonic_ns()
        save(path, state)
        issued = time.monotonic_ns()
        if slot == pause:
            return state
        if slot == crash:
            import coverage

            current = coverage.Coverage.current()
            if current is not None:
                current.save()
            progress("kill_boundary", slot + 1, 72 - slot - 1)
            os.kill(os.getpid(), signal.SIGKILL)
        release(trajectory, state, slot, arm)
        release_done = time.monotonic_ns()
        save(path, state)
        ended = time.monotonic_ns()
        state["costs"].append(
            dict(
                slot=slot,
                started_monotonic_ns=began,
                ended_monotonic_ns=ended,
                issue_ns=issue_done - began,
                release_ns=release_done - issued,
                checkpoint_ns=(issued - issue_done) + (ended - release_done),
                issue_checkpoint_ns=issued - began,
                release_checkpoint_ns=ended - issued,
                total_ns=ended - began,
            )
        )
        if (slot + 1) % 24 == 0:
            progress("events", slot + 1, 72 - slot - 1)
    save(path, state)
    return state


def numeric_audit() -> Json:
    """Use scalar basis values and centered differences without calling updates."""
    c = static_fit()
    maximum = 0.0
    basis_error = 0.0
    for value in [0.0, 1e-12, 0.2, 0.4, 0.6, 0.8, 1.0 - 1e-12, 1.0]:
        x = [0.3, *([value] * 4)]
        d, oracle = design(x), scalar_design(x)
        basis_error = max(basis_error, max(abs(a - b) for a, b in zip(d, oracle, strict=True)))
        eta = sum(a * b for a, b in zip(c, oracle, strict=True))
        for i in range(34):
            high, low = list(c), list(c)
            high[i] += 1e-5
            low[i] -= 1e-5
            loss = lambda weights: math.log1p(
                math.exp(sum(a * b for a, b in zip(weights, oracle, strict=True)))
            )
            finite = (loss(high) - loss(low)) / 2e-5
            maximum = max(maximum, abs(finite - sigmoid(eta) * d[i]))
    return dict(
        basis_error_max=basis_error,
        finite_difference_error_max=maximum,
        boundary_actions=[action(0.25), action(0.75)],
        static_slope=c[0],
    )


def negative_control() -> Json:
    """Plant a cache probability across an action boundary; a truncated index misses it."""
    trajectory = dict(
        coefficients=[0.0, math.log(0.249 / 0.751), *([0.0] * 32)],
        cache_x=[[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0, 1.0]],
    )
    good, bad = initial(trajectory), initial(trajectory, "truncated")
    event = dict(id="planted", x=[0.0, 0.0, 0.0, 0.0, 0.0], y=1)
    update(good, event, "indexed")
    update(bad, event, "truncated")
    before = deepcopy(bad["cache"])
    issue(good, 0)
    issue(bad, 0)
    immutable = deepcopy(good["issues"])
    update(good, dict(event, id="global", global_update=True), "indexed")
    assert good["issues"] == immutable
    return dict(
        detected=bad["stale_cache_count"] > 0,
        action_changed=good["issues"][0]["action"] != bad["issues"][0]["action"],
        stale_probability=before["0"]["p"],
        fresh_probability=good["issues"][0]["p"],
        unsafe_action=bad["issues"][0]["action"],
        reference_action=good["issues"][0]["action"],
    )

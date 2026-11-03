"""REQ-VERIFY-8263: delayed released labels adjust allowed typed decisions.

Source hashes assign negative controls only. They are never learned lookup
features. All arms share feedback, and private retention labels remain outside
updates, so a gain here qualifies causal mechanics rather than natural science.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from carnot.reporting.current_work_receipt import canonical_hash, atomic_json
from carnot.verify.calibrated_memory_trajectory_8211 import append, journal
from carnot.verify.evidence_view_kernel_8249 import mix

Json = dict[str, Any]
GROUPS = [f"{i:03b}" for i in range(8)]
ARMS = ["global", "group", "random101", "random102", "random103"]


def group(features: Json | None) -> str | None:
    """Only cached relation and the two frozen delta predicates define groups."""
    if features is None:
        return None
    if features["relation"] not in {"E", "C", "B"} or any(
        not math.isfinite(features[k]) for k in ["selected_delta", "control_delta"]
    ):
        raise ValueError("features")
    return f"{int(features['relation'] == 'E')}{int(features['selected_delta'] > 0.1)}{int(abs(features['control_delta']) > 0.1)}"


def random_group(source: str, seed: int) -> str:
    """Freeze canonical UTF-8 source-hash assignment before any label is accessed."""
    data = json.dumps([source, seed], ensure_ascii=False, separators=(",", ":")).encode()
    return GROUPS[int.from_bytes(hashlib.sha256(data).digest(), "big") % 8]


def action(p: float | None, accepted: bool, fallback: str = "escalate") -> str:
    """Minimize actual allowed action costs, with exact boundary ties escalated."""
    if p is None:
        return fallback
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("probability")
    costs = dict(reject=1 - p, escalate=0.5)
    if accepted:
        costs["accept"] = 5 * p
    minimum = min(costs.values())
    winners = [k for k, v in costs.items() if abs(v - minimum) <= 1e-12]
    return winners[0] if len(winners) == 1 else "escalate"


def cost(decision: str, y: int) -> float:
    """Use the frozen typed scientific endpoint instead of threshold loss."""
    return float({"accept": 5 * y, "reject": 1 - y, "escalate": 0.5}[decision])


def initial() -> Json:
    """Store explicit counters and pending feedback for exact causal recovery."""
    return dict(
        global_counts=dict(n=0, unsupported=0),
        groups={a: {g: dict(n=0, unsupported=0) for g in GROUPS} for a in ARMS[1:]},
        issued=[],
        released=[],
        pending=[],
        events=[],
        admission_events=[],
    )


def predict(state: Json, row: Json) -> Json:
    """Missing feature rows share frozen fallback and cannot create group updates."""
    key = group(row["features"]) if row["p_static"] is not None else None
    keys = {
        a: key if a == "group" else random_group(row["original_source_sha256"], int(a[6:]))
        for a in ARMS[1:]
    }
    if key is None:
        keys = dict.fromkeys(ARMS[1:])
    ps = {}
    for arm in ARMS:
        p = row["p_static"]
        if p is not None and key is not None:
            p = mix(p, state["global_counts"])
            if arm != "global" and state["groups"][arm][keys[arm]]["n"] >= 8:
                p = mix(p, state["groups"][arm][keys[arm]])
        ps[arm] = p
    return dict(
        row,
        keys=keys,
        predictions=ps,
        actions={
            a: action(p, row["baseline_accepted"], row["fallback_action"]) for a, p in ps.items()
        },
    )


def transition(state: Json, event: Json) -> None:
    """Recompute issues and reject any release without its prior durable clock."""
    r = event["record"]
    if event["kind"] == "issue":
        if r["slot"] != len(state["issued"]) or r["role"] not in {"stream", "retention"}:
            raise ValueError("issue_order")
        row = {k: v for k, v in r.items() if k not in {"keys", "predictions", "actions"}}
        if predict(state, row) != r:
            raise ValueError("prediction_drift")
        state["issued"].append(r)
        if r["role"] == "stream":
            state["pending"].append(r["slot"])
    elif event["kind"] == "release":
        origin, now = r["origin"], r["now"]
        if not 0 <= origin < len(state["issued"]):
            raise ValueError("unissued")
        if now != origin + 8 or now >= len(state["issued"]):
            raise ValueError("future_feedback")
        issued = state["issued"][origin]
        source = issued["original_source_sha256"]
        if issued["role"] == "retention":
            raise ValueError("retention")
        if source in state["released"]:
            raise ValueError("duplicate_source")
        if type(r["label"]) is not int or r["label"] not in [0, 1]:
            raise ValueError("label")
        counters = [state["global_counts"]]
        for arm, key in issued["keys"].items():
            if key is not None:
                counts = state["groups"][arm][key]
                counters.append(counts)
                if counts["n"] == 7:
                    state["admission_events"].append(
                        dict(arm=arm, group=key, release_slot=now, effective_issue_slot=now + 1)
                    )
        for counts in counters:
            counts["n"] += 1
            counts["unsupported"] += r["label"]
        state["released"].append(source)
        state["pending"].remove(origin)
    else:
        raise ValueError("event_kind")
    state["events"].append(event)


def load(path: Path) -> Json:
    """Reject torn records and rebuild state from complete fsynced transitions."""
    if path.exists() and path.read_bytes() and not path.read_bytes().endswith(b"\n"):
        raise ValueError("partial_record")
    state = initial()
    for event in journal(path):
        transition(state, event)
    return state


def roster(kind: str) -> list[Json]:
    """Specify private labels independently of runtime predictions before running.

    Natural-shaped missingness is a feasibility control with no gain demand.
    The deterministic learnable labels intentionally make fixture benefit circular.
    """
    size = 384 if kind in {"learnable", "no_signal"} else 96
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    rows = []
    for t in range(size + 32):
        g = GROUPS[t % 8]
        retained = t >= size
        missing = kind == "zero" or (kind == "natural" and t % 3 == 0) or retained
        features = (
            None
            if missing
            else dict(
                relation="E" if g[0] == "1" else "B",
                selected_delta=0.2 if g[1] == "1" else 0,
                control_delta=-0.2 if g[2] == "1" else 0,
            )
        )
        y = t % 2 if retained else (t // 8) % 2 if kind == "no_signal" else labels[t % 8]
        rows.append(
            dict(
                slot=t,
                original_source_sha256=canonical_hash(["private8263", kind, t]),
                role="retention" if retained else "stream",
                p_static=float(y)
                if retained
                else None
                if kind == "natural" and t % 9 == 0
                else 0.45,
                features=features,
                baseline_accepted=retained and y == 0,
                fallback_action="escalate",
                label=y,
            )
        )
    return rows


def run(rows: list[Json], path: Path, *, stop: int = 0, crash_slot: int = -1) -> Json:
    """Finish pending releases before the next issue after a genuine hard exit.

    Issues are fsynced before feedback is read. Checkpoints each eight sources
    bind the complete frozen roster and pending list, while the journal is authority.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    state = load(path)
    checkpoint = path.with_suffix(".checkpoint.json")
    if checkpoint.exists() and json.loads(checkpoint.read_bytes())[
        "roster_sha256"
    ] != canonical_hash(rows):
        raise ValueError("resume_roster")
    if any(
        {k: v for k, v in rows[r["slot"]].items() if k != "label"}
        != {k: v for k, v in r.items() if k not in {"keys", "predictions", "actions"}}
        for r in state["issued"]
    ):
        raise ValueError("resume_roster")

    def release_due(now: int) -> None:
        origin = now - 8
        if origin in state["pending"]:
            event = dict(
                kind="release", record=dict(origin=origin, now=now, label=rows[origin]["label"])
            )
            transition(state, event)
            append(path, event)

    if state["issued"]:
        release_due(len(state["issued"]) - 1)
    for row in rows[len(state["issued"]) : stop or len(rows)]:
        current = {k: v for k, v in row.items() if k != "label"}
        event = dict(kind="issue", record=predict(state, current))
        transition(state, event)
        append(path, event)
        if row["slot"] == crash_slot:
            atomic_json(path.with_suffix(".pending.json"), state)
            print(
                f"[exp8263] phase=hard_exit completed={row['slot'] + 1} pending={len(state['pending'])}",
                flush=True,
            )
            os._exit(73)
        release_due(row["slot"])
        if (row["slot"] + 1) % 8 == 0:
            atomic_json(
                path.with_suffix(".checkpoint.json"),
                dict(
                    roster_sha256=canonical_hash(rows),
                    state_sha256=canonical_hash(state),
                    pending=state["pending"],
                    global_counts=state["global_counts"],
                    groups=state["groups"],
                    issued_count=len(state["issued"]),
                ),
            )
        if (row["slot"] + 1) % 64 == 0:
            print(
                f"[exp8263] phase=typed_control completed={row['slot'] + 1} pending={len(rows) - row['slot'] - 1}",
                flush=True,
            )
    return state


def score(state: Json, rows: list[Json], condition: str) -> list[Json]:
    """Keep every source and arm with actual typed costs and metric denominators."""
    return [
        dict(
            unit_id=r["original_source_sha256"],
            source_cluster_id=r["original_source_sha256"],
            slot=r["slot"],
            role=r["role"],
            condition=condition,
            arm=arm,
            action=r["actions"][arm],
            label=rows[r["slot"]]["label"],
            p=r["predictions"][arm],
            actual_cost=cost(r["actions"][arm], rows[r["slot"]]["label"]),
            numerator=cost(r["actions"][arm], rows[r["slot"]]["label"]),
            denominator=1,
            status="completed",
            independent_source_count=0,
        )
        for r in state["issued"]
        for arm in ARMS
    ]


def controls(private: Path) -> Json:
    """Freeze all private rosters first and score this same causal typed runtime."""
    private.mkdir(parents=True, exist_ok=True)
    rosters = {name: roster(name) for name in ["natural", "learnable", "no_signal", "zero"]}
    atomic_json(private / "rosters.json", rosters)
    states, scored = {}, {}
    for name, rows in rosters.items():
        states[name] = run(rows, private / (name + ".jsonl"))
        scored[name] = score(states[name], rows, name)
    later = [r for r in scored["learnable"] if r["role"] == "stream" and r["slot"] >= 192]
    costs = {a: sum(r["actual_cost"] for r in later if r["arm"] == a) / 192 for a in ARMS}
    paired = {a: {r["slot"]: r for r in later if r["arm"] == a} for a in ARMS}
    improvements = sum(
        paired["group"][t]["actual_cost"] < r["actual_cost"] for t, r in paired["global"].items()
    )
    false_accepts = {
        a: sum(r["action"] == "accept" and r["label"] == 1 for r in later if r["arm"] == a)
        for a in ARMS
    }
    retained = [r for r in scored["learnable"] if r["role"] == "retention"]
    retention = {
        a: dict(
            cost=sum(r["actual_cost"] for r in retained if r["arm"] == a) / 32,
            brier=sum((r["p"] - r["label"]) ** 2 for r in retained if r["arm"] == a) / 32,
        )
        for a in ARMS
    }
    positive = (
        costs["global"] - costs["group"] > 0.02
        and improvements >= 5
        and false_accepts["group"] <= false_accepts["global"]
        and retention["group"]["cost"] - retention["global"]["cost"] <= 0.02
        and retention["group"]["brier"] - retention["global"]["brier"] <= 0.01
    )
    return dict(
        rosters=rosters,
        states=states,
        natural_shape_control_rows=scored["natural"],
        learnable_control_rows=scored["learnable"],
        decision_control_rows=scored["no_signal"] + scored["zero"],
        admission_events={n: s["admission_events"] for n, s in states.items()},
        state_hashes={n: canonical_hash(s) for n, s in states.items()},
        later_costs=costs,
        improvements=improvements,
        false_accepts=false_accepts,
        retention=retention,
        zero_admission_count=len(states["zero"]["admission_events"]),
        positive_control_passed=positive,
        fixture_scope="circular_positive mechanics only",
    )

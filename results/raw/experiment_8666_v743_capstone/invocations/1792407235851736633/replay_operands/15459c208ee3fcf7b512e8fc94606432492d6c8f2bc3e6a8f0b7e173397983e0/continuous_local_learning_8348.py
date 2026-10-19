"""REQ-SELF-8348: delayed numeric adaptation preserves issued decisions.

The original spline basis and conservative cache invalidation remain shared.
Only already released labels can change the small numeric coefficients.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import local_update_isolation_8306 as kernel
from carnot.verify.calibrated_memory_trajectory_8211 import append, journal
from carnot.verify.independent_online_memory_8116 import LabelVault

Json = dict[str, Any]
ARMS = [
    "frozen_spline",
    "online_sparse",
    "online_dense",
    "calibration_only",
    "shuffled_due_feedback",
]
SEED = 7178310


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush measured counts so a parent can distinguish work from silence."""
    print(f"[exp8348] phase={phase} completed={completed} pending={pending}", flush=True)


def probability(head: Json, x: list[float] | None) -> float | None:
    """Missing features cannot be turned into an invented negative decision."""
    return (
        None
        if x is None
        else float(kernel.sigmoid(kernel.logit(head["coefficients"], x) / head["temperature"]))
    )


def learn(head: Json, x: list[float] | None, y: int | None, arm: str) -> Json:
    """One deployed-logit gradient step freezes every nonlocal parameter.

    Temperature divides the residual before norm clipping. The dense control
    visits zero coordinates too, but uses exactly the same arithmetic order.
    """
    if arm not in ARMS:
        raise ValueError("arm")
    c = list(head["coefficients"])
    reason = (
        "missing_features"
        if x is None
        else "missing_label"
        if y is None
        else "frozen"
        if arm == "frozen_spline"
        else "applied"
    )
    support = [] if x is None else [i for i, v in enumerate(kernel.design(x)) if i >= 2 and v]
    gradient = [0.0] * 34
    if reason == "applied":
        assert x is not None and y is not None
        residual = (float(probability(head, x)) - y) / head["temperature"]
        phi = kernel.design(x)
        gradient = (
            [0.0, residual, *([0.0] * 32)]
            if arm == "calibration_only"
            else [0.0, 0.0, *[residual * v for v in phi[2:]]]
        )
        norm = math.sqrt(sum(g * g for g in gradient))
        scale = min(1.0, 1.0 / norm) if norm else 1.0
        for i, g in enumerate(gradient):
            if g or arm == "online_dense":
                c[i] = min(4.0, max(-4.0, c[i] - 0.01 * g * scale))
    delta = [a - b for a, b in zip(c, head["coefficients"], strict=True)]
    return dict(
        coefficients=c,
        changed=[i for i, d in enumerate(delta) if d],
        coefficient_delta=delta,
        basis_support=support,
        reason=reason,
        gradient_norm=math.sqrt(sum(g * g for g in gradient)),
    )


class Labels(LabelVault):
    """Decode exactly one due stream fragment; retention targets stay opaque."""

    def release(self, slot: int, clock: int, *, sealed: bool, retention: bool = False) -> Json:
        if retention or not 1 <= slot <= 88 or clock != slot + 8 or not sealed:
            raise ValueError("due_label_barrier")
        row: Json = json.loads(self.fragments[slot - 1])
        public = self.rows[slot - 1]
        if (row["slot"], row["unit_id"], row["source_cluster_id"]) != (
            slot,
            public["unit_id"],
            public["source_cluster_id"],
        ) or row["y"] not in (0, 1, None):
            raise ValueError("label_identity")
        return row


def initial(bundle: Json) -> Json:
    """All arms start from the same frozen head and exact coefficient dependencies."""
    arms = {}
    for arm in ARMS:
        head = deepcopy(bundle["head"])
        cache = {
            str(r["slot"]): dict(x=r["x"], p=probability(head, r["x"]), valid=True)
            for r in bundle["slots"][:96]
        }
        index = {
            str(i): [
                key for key, r in cache.items() if r["x"] is not None and kernel.design(r["x"])[i]
            ]
            for i in range(34)
        }
        arms[arm] = dict(
            head,
            cache=cache,
            index=index,
            index_hash=canonical_hash(index),
            index_version=0,
            version=0,
        )
    return dict(
        arms=arms,
        cursor=0,
        pending=[],
        feedback_ids=[],
        released_pool=[],
        shuffle_draws=0,
        issued=[],
        updates=[],
        releases=[],
        retention=[],
    )


def release(state: Json, bundle: Json, target: Json, clock: int) -> None:
    """Exactly-once feedback changes only future cached predictions."""
    slot, identity = target["slot"], target["unit_id"]
    if identity in state["feedback_ids"] or clock != slot + 8 or slot not in state["pending"]:
        raise ValueError("duplicate_or_stale_feedback")
    row = bundle["slots"][slot - 1]
    if row["unit_id"] != identity or row["source_cluster_id"] != target["source_cluster_id"]:
        raise ValueError("feedback_identity")
    y = target["y"]
    if y is not None:
        state["released_pool"].append(y)
    rng = random.Random(SEED)
    for n in range(1, len(state["released_pool"])):
        rng.randrange(n)
    shuffled = rng.choice(state["released_pool"]) if y is not None else None
    state["shuffle_draws"] += int(y is not None)
    for arm in ARMS:
        head = state["arms"][arm]
        change = learn(head, row["x"], shuffled if arm == "shuffled_due_feedback" else y, arm)
        head["coefficients"] = change["coefficients"]
        if (
            head["index_hash"] != canonical_hash(head["index"])
            or head["index_version"] != head["version"]
        ):
            head["index_version"] = -1
            head["index"] = {
                str(i): [
                    key
                    for key, r in head["cache"].items()
                    if r["x"] is not None and kernel.design(r["x"])[i]
                ]
                for i in range(34)
            }
            head["index_hash"] = canonical_hash(head["index"])
        cached = head["cache"]
        if arm == "online_dense":
            head["cache"] = {key: r for key, r in cached.items() if r["x"] is not None}
        invalidation = kernel.invalidate(
            head, change["changed"], "full" if arm == "online_dense" else "indexed"
        )
        head["cache"] = cached
        state["updates"].append(
            dict(
                change,
                **invalidation,
                arm=arm,
                label_slot=slot,
                release_slot=clock,
                unit_id=identity,
                source_cluster_id=row["source_cluster_id"],
                observed_y=y,
                used_y=shuffled if arm == "shuffled_due_feedback" else y,
            )
        )
    state["feedback_ids"].append(identity)
    state["pending"].remove(slot)
    state["releases"].append(dict(target, label_slot=slot, release_slot=clock))


def retention(state: Json, bundle: Json, window: int) -> None:
    """Fixed feature-only shadows seal every window without evaluator labels."""
    for row in bundle["slots"][96:]:
        for arm in ARMS:
            head = state["arms"][arm]
            p = probability(head, row["x"])
            state["retention"].append(
                dict(
                    window=window,
                    slot=row["slot"],
                    arm=arm,
                    unit_id=row["unit_id"],
                    source_cluster_id=row["source_cluster_id"],
                    p=p,
                    action="escalate" if p is None else kernel.action(p),
                    state_hash=canonical_hash(head["coefficients"]),
                    targets_opened=False,
                )
            )


def run(bundle: Json, raw: Path, *, crash: int = 0, mutate_from: int = 0, stop: int = 96) -> Json:
    """Save issued state before release and save each complete update before ack.

    Owned workers exit only after a durable update. Timing stays outside the
    semantic state so a restart compares numeric decisions, not wall clocks.
    """
    raw.mkdir(parents=True, exist_ok=True)
    checkpoint = raw / "state.json"
    events = raw / "events.jsonl"
    if checkpoint.exists():
        state = json.loads(checkpoint.read_bytes())
        records = journal(events)
        if not records or records[-1].get("state_hash") != canonical_hash(state):
            raise ValueError("checkpoint_journal_drift")
    else:
        state = initial(bundle)
        retention(state, bundle, 0)
    vault = Labels(Path(bundle["labels"]["path"]), bundle["slots"])
    began = time.monotonic()
    for slot in range(state["cursor"] + 1, stop + 1):
        if time.monotonic() - began > 120:
            raise TimeoutError("trajectory_deadline")
        row = bundle["slots"][slot - 1]
        issues = []
        for arm in ARMS:
            head = state["arms"][arm]
            cached = head["cache"].pop(str(slot))
            p = cached["p"] if cached["valid"] else probability(head, row["x"])
            if p != probability(head, row["x"]):
                raise ValueError("stale_cache")
            for entries in head["index"].values():
                if str(slot) in entries:
                    entries.remove(str(slot))
            head["index_hash"] = canonical_hash(head["index"])
            issues.append(
                dict(
                    slot=slot,
                    issue_slot=slot,
                    due_slot=slot + 8,
                    unit_id=row["unit_id"],
                    source_cluster_id=row["source_cluster_id"],
                    arm=arm,
                    p=p,
                    action="escalate" if p is None else kernel.action(p),
                    state_hash=canonical_hash(
                        dict(
                            coefficients=head["coefficients"],
                            temperature=head["temperature"],
                            feedback_ids=state["feedback_ids"],
                            pending=state["pending"],
                            index_version=head["index_version"],
                        )
                    ),
                    status="excluded" if p is None else "completed",
                )
            )
        state["issued"].extend(issues)
        state["pending"].append(slot)
        append(events, dict(kind="issue", slot=slot, rows=issues))
        atomic_json(checkpoint, state)
        started = time.monotonic_ns()
        if slot > 8:
            target = vault.release(slot - 8, slot, sealed=True)
            if mutate_from and target["slot"] >= mutate_from and target["y"] is not None:
                target = dict(target, y=1 - target["y"])
            release(state, bundle, target, slot)
            append(
                events,
                dict(kind="release", slot=slot, target=target, updates=state["updates"][-5:]),
            )
        state["cursor"] = slot
        if slot in (32, 64, 96):
            retention(state, bundle, slot)
            atomic_json(raw / f"checkpoint-{slot}.json", state)
        atomic_json(checkpoint, state)
        append(events, dict(kind="commit", slot=slot, state_hash=canonical_hash(state)))
        append(
            raw / "costs.jsonl",
            dict(
                slot=slot,
                durable_update_latency_ns=time.monotonic_ns() - started,
                memory_bytes=len(json.dumps(state).encode()),
                checkpoint_bytes=checkpoint.stat().st_size,
            ),
        )
        if slot % 16 == 0:
            progress("trajectory", slot, 96 - slot)
        if slot == crash:
            progress("durable_update_then_hard_exit", slot, 96 - slot)
            os._exit(73)
    atomic_json(raw / "final.json", state)
    return dict(state)


def control(head: Json) -> Json:
    """A constructed fit-only input tests the exact budget without method retuning."""
    local = [0.0] * 4
    phi = kernel.design([0.0, *local])
    offset = sum(c * v for c, v in zip(head["coefficients"][1:], phi[1:], strict=True))
    x = [(head["temperature"] * (math.log(3) - 0.05) - offset) / head["coefficients"][0], *local]
    current = deepcopy(head)
    before = probability(current, x)
    for _ in range(88):
        current["coefficients"] = learn(current, x, 1, "online_sparse")["coefficients"]
    after = probability(current, x)
    assert before is not None and after is not None
    return dict(
        updates=88,
        budget=88,
        features=x,
        initial_p=before,
        final_p=after,
        initial_action=kernel.action(before),
        final_action=kernel.action(after),
        passed=kernel.action(before) != kernel.action(after),
        scope="constructed_fit_only_control",
        verdict_class="circular_positive",
    )


def reachability(bundle: Json) -> list[Json]:
    """Norm bounds identify actions that this fixed update budget cannot reach."""
    rows = []
    for row in bundle["slots"]:
        x = row["x"]
        head = bundle["head"]
        p = probability(head, x)
        z = None if x is None else kernel.logit(head["coefficients"], x) / head["temperature"]
        bound = (
            None
            if x is None
            else 0.88 * math.sqrt(sum(v * v for v in kernel.design(x)[2:])) / head["temperature"]
        )
        margin = None if z is None else min(abs(z - math.log(3)), abs(z + math.log(3)))
        rows.append(
            dict(
                slot=row["slot"],
                unit_id=row["unit_id"],
                initial_p=p,
                starting_logit=z,
                starting_margin=margin,
                attainable_logit_change_bound=bound,
                action_boundary_reachable=None if margin is None else margin <= bound,
            )
        )
    return rows


def attribution(bundle: Json, state: Json) -> list[Json]:
    """Each admitted delta is isolated on later distinct sources, without targets."""
    rows = []
    for update in state["updates"]:
        if not update["changed"]:
            continue
        after = dict(bundle["head"], coefficients=update["coefficients"])
        before = dict(
            after,
            coefficients=[
                c - d
                for c, d in zip(update["coefficients"], update["coefficient_delta"], strict=True)
            ],
        )
        later = []
        for row in bundle["slots"][update["release_slot"] : 96]:
            if row["x"] is None or row["source_cluster_id"] == update["source_cluster_id"]:
                continue
            p0, p1 = probability(before, row["x"]), probability(after, row["x"])
            if p0 != p1:
                assert p0 is not None and p1 is not None
                later.append(
                    dict(
                        slot=row["slot"],
                        unit_id=row["unit_id"],
                        p_before=p0,
                        p_after=p1,
                        action_changed=kernel.action(p0) != kernel.action(p1),
                    )
                )
        rows.append(
            dict(
                update_source=update["unit_id"],
                release_slot=update["release_slot"],
                arm=update["arm"],
                basis_support=update["basis_support"],
                coefficient_delta=update["coefficient_delta"],
                later_distinct_sources=later,
                later_probability_changed=bool(later),
                later_action_changed=any(v["action_changed"] for v in later),
            )
        )
    return rows


def numeric_proof(work: Json) -> Json:
    """Independent scalar recursion and a deliberate wrong coefficient test exact zeros."""

    def error(state: Json) -> float:
        expected = {
            a: list(work["bundle"]["head"]["coefficients"])
            for a in ["online_sparse", "online_dense"]
        }
        errors = []
        for row in state["updates"]:
            arm = row["arm"]
            if arm not in expected:
                continue
            x = work["bundle"]["slots"][row["label_slot"] - 1]["x"]
            c = expected[arm]
            if x is not None and row["used_y"] is not None:
                d = kernel.scalar_design(x)
                temperature = work["bundle"]["head"]["temperature"]
                z = sum(a * b for a, b in zip(c, d, strict=True)) / temperature
                residual = (kernel.sigmoid(z) - row["used_y"]) / temperature
                g = [0.0, 0.0, *[residual * v for v in d[2:]]]
                norm = math.sqrt(sum(v * v for v in g))
                scale = min(1.0, 1.0 / norm) if norm else 1.0
                expected[arm] = [
                    min(4, max(-4, v - 0.01 * w * scale)) if w else v
                    for v, w in zip(c, g, strict=True)
                ]
            errors.extend(
                abs(a - b) for a, b in zip(expected[arm], row["coefficients"], strict=True)
            )
        return max(errors)

    if not work["state"]:
        return dict(recomputed=False, deliberate_error_rejected=False)
    measured = error(work["state"])
    bad = deepcopy(work["state"])
    next(u for u in bad["updates"] if u["arm"] == "online_dense")["coefficients"][2] += 0.1
    return dict(
        recomputed=measured <= 1e-10,
        deliberate_error_rejected=error(bad) > 1e-10,
        independent_scalar_error_max=measured,
        scope="arithmetic_only_not_learning_benefit",
    )

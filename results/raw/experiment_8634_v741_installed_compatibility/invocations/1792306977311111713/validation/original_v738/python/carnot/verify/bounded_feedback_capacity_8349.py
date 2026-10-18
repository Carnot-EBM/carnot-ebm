"""REQ-VERIFY-8349: finite tracking is a constructed systems experiment.

The scheduler sees features and independent priorities, never future delays or
labels. The separate evaluator can score labels permanently lost by admission.
This adapts finite tracking, without implementing paper scheduling or DW-FTRL.
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

Json = dict[str, Any]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual counts so supervision does not mistake progress for a stall."""
    print(f"[exp8349] phase={phase} completed={completed} pending={pending}", flush=True)


def manifest() -> Json:
    """Freeze evaluator traces and scheduler rules before any observation opens."""
    traces = []
    for family in range(3):
        events = []
        for t in range(1, 513):
            x = [0.0] * 12 + [((17 * t + 13 * j) % 101) / 100 for j in range(4)]
            delay = (
                8
                if family == 0
                else [1, 8, 32, 64][(t - 1) % 4]
                if family == 1
                else 64
                if 129 <= t <= 256
                else 8
            )
            y = int((1 if t <= 256 else -1) * (2 * x[12] - x[13] + x[14] - 2 * x[15] + 0.1) >= 0)
            events.append(dict(slot=t, x=x, y=y, delay=delay))
        traces.append(dict(family=family, events=events, label_scope="constructed"))
    units = [
        dict(id=f"{f}-{seed}-{policy}-{cap}", family=f, seed=seed, policy=policy, capacity=cap)
        for f in range(3)
        for seed in [11, 22, 33]
        for policy, cap in [
            ("unlimited", None),
            *[(p, c) for p in ["first", "random"] for c in [4, 16, 64]],
        ]
    ]
    return dict(
        traces=traces,
        units=units,
        step=0.01,
        gradient_norm_cap=1,
        knots=kernel.KNOTS,
        ordering="issue_admit_then_due",
        inverse_propensity_weighting=False,
        random_rule="independent uniform continuous priority; retain C smallest",
        independent_count=3,
    )


class Vault:
    """Only the evaluator knows delays; learner feedback opens at expiration."""

    def __init__(self, trace: Json):
        self._rows = trace["events"]

    def due(self, clock: int) -> list[int]:
        """Release identifiers only when the hidden delay actually expires."""
        return [v["slot"] for v in self._rows if v["slot"] + v["delay"] == clock]

    def release(self, slot: int, clock: int, state: Json) -> Json:
        """Loss and future-label barriers prevent feedback from being resurrected."""
        row = self._rows[slot - 1]
        if (
            clock != slot + row["delay"]
            or slot in state["lost"]
            or slot not in [v["slot"] for v in state["pending"]]
        ):
            raise ValueError("future_or_lost_label")
        return dict(slot=slot, x=row["x"], y=row["y"], delay=row["delay"])


def initial(unit: Json) -> Json:
    """Store no future labels or delays in the learner's persistent state."""
    if unit["policy"] not in ["unlimited", "first", "random"]:
        raise ValueError("scheduler")
    return dict(
        unit=unit,
        cursor=0,
        heads={a: [1.0, 0.0, *([0.0] * 32)] for a in ["sparse", "dense"]},
        temperature=1,
        pending=[],
        lost=[],
        applied=[],
        events=[],
        maximum_pending_count=0,
        scheduler_rng_state=json.loads(json.dumps(random.Random(unit["seed"]).getstate())),
        metrics={
            a: dict(coefficient_touches=0, bytes_written=0, full_update_time_s=0.0)
            for a in ["sparse", "dense"]
        },
    )


def invariant(state: Json) -> None:
    """Admission must satisfy the bound before any due item frees a slot."""
    cap = state["unit"]["capacity"]
    if cap is not None and len(state["pending"]) > cap:
        raise ValueError("capacity_overflow")


def semantic(state: Json) -> Json:
    """Clocks and measured write costs do not alter predictions or RNG meaning."""
    return {k: v for k, v in state.items() if k not in ["metrics", "checksum"]}


def simulate(
    trace: Json,
    unit: Json,
    *,
    state: Json | None = None,
    until: int = 576,
    directory: Path | None = None,
) -> Json:
    """Issue first, permanently drop excess items, then learn from due survivors."""
    state = initial(unit) if state is None else state
    rng = random.Random()
    saved = state["scheduler_rng_state"]
    rng.setstate((saved[0], tuple(saved[1]), saved[2]))
    vault = Vault(trace)
    for clock in range(state["cursor"] + 1, until + 1):
        if clock <= 512:
            row = trace["events"][clock - 1]
            x = [row["x"][0], *row["x"][12:16]]
            p = {a: float(kernel.sigmoid(kernel.logit(c, x))) for a, c in state["heads"].items()}
            priority = rng.random()
            state["events"].append(
                dict(kind="issue", clock=clock, slot=clock, p=p, priority=priority)
            )
            item = dict(slot=clock, priority=priority)
            candidates = state["pending"] + [item]
            cap, policy = unit["capacity"], unit["policy"]
            selected = (
                candidates
                if cap is None
                else (
                    sorted(candidates, key=lambda v: v["priority"])[:cap]
                    if policy == "random"
                    else candidates[:cap]
                )
            )
            for dropped in candidates:
                if dropped not in selected:
                    state["lost"].append(dropped["slot"])
                    state["events"].append(dict(kind="drop", clock=clock, **dropped))
            state["pending"] = selected
            invariant(state)
            state["maximum_pending_count"] = max(state["maximum_pending_count"], len(selected))
            state["events"].append(dict(kind="admit", clock=clock, members=deepcopy(selected)))
        for slot in vault.due(clock):
            retained = slot not in state["lost"]
            state["events"].append(dict(kind="due", clock=clock, slot=slot, retained=retained))
            if retained:
                feedback = vault.release(slot, clock, state)
                for arm in ["sparse", "dense"]:
                    began = time.perf_counter()
                    head = kernel.initial(dict(coefficients=state["heads"][arm], cache_x=[]))
                    change = kernel.update(
                        head,
                        dict(
                            id=str(slot), x=[0.0, *feedback["x"][12:16]], y=feedback["y"], rate=0.01
                        ),
                        "indexed" if arm == "sparse" else "full",
                    )
                    state["heads"][arm] = head["coefficients"]
                    event = dict(
                        kind="update",
                        clock=clock,
                        slot=slot,
                        arm=arm,
                        y=feedback["y"],
                        delay=feedback["delay"],
                        coefficients=change["coefficients"],
                        coefficient_touches=change["coefficient_visits"],
                    )
                    state["events"].append(event)
                    encoded = (json.dumps(event, sort_keys=True) + "\n").encode()
                    if directory is not None:
                        with (directory / "updates.jsonl").open("ab") as stream:
                            stream.write(encoded)
                            stream.flush()
                            os.fsync(stream.fileno())
                    metric = state["metrics"][arm]
                    metric["bytes_written"] += len(encoded) if directory is not None else 0
                    metric["coefficient_touches"] += change["coefficient_visits"]
                    metric["full_update_time_s"] += time.perf_counter() - began
                state["applied"].append(slot)
                state["pending"] = [v for v in state["pending"] if v["slot"] != slot]
        state["cursor"] = clock
        state["scheduler_rng_state"] = json.loads(json.dumps(rng.getstate()))
        if clock % 128 == 0:
            progress("ticks", clock, max(0, until - clock))
    state["feedback_retained"] = len(state["applied"])
    state["feedback_lost"] = len(state["lost"])
    return state


def worker(bundle: Json, directory: Path, crash: int = 0) -> Json:
    """Hard exit happens only after a complete causal state and RNG are durable."""
    progress("worker_before_benchmark")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "state.json"
    trace, unit = bundle["trace"], bundle["unit"]
    state = json.loads(path.read_bytes()) if path.exists() else initial(unit)
    if path.exists() and semantic(state) != semantic(simulate(trace, unit, until=state["cursor"])):
        raise ValueError("rehashed_checkpoint_drift")
    state = simulate(trace, unit, state=state, until=crash or 576, directory=directory)
    state["checksum"] = canonical_hash(semantic(state))
    atomic_json(path, state)
    if crash:
        atomic_json(directory / f"checkpoint-{crash}.json", state)
        progress("durable_before_exit", crash, 512 - crash)
        os._exit(73)
    atomic_json(directory / "final.json", state)
    progress("worker_after_benchmark", 512, 0)
    return state


def audit(trace: Json, state: Json) -> bool:
    """Recompute all admissions and feedback, so rehashing cannot legitimize tampering."""
    return semantic(state) == semantic(simulate(trace, state["unit"], until=state["cursor"]))


def controls() -> Json:
    """Each deliberate violation must trip the real learner boundary or capacity guard."""
    state = initial(dict(policy="first", capacity=4, seed=11))
    vault = Vault(manifest()["traces"][0])
    outcomes = {}
    for name, slot, clock, lost, pending in [
        ("future_leakage_rejected", 1, 1, [], [dict(slot=1, priority=0.1)]),
        ("lost_resurrection_rejected", 1, 9, [1], []),
        ("overflow_rejected", 1, 9, [], [dict(slot=i, priority=0.1) for i in range(5)]),
    ]:
        state.update(lost=lost, pending=pending)
        try:
            invariant(state)
            vault.release(slot, clock, state)
            outcomes[name] = False
        except ValueError:
            outcomes[name] = True
    return outcomes


def numeric_proof(states: list[Json]) -> Json:
    """Use independently recursive spline values, and corrupt a coefficient as control."""
    errors = []
    for state in states:
        for issue in [r for r in state["events"] if r["kind"] == "issue"]:
            t = issue["slot"]
            basis = kernel.scalar_design(
                [0.0, *[((17 * t + 13 * j) % 101) / 100 for j in range(4)]]
            )
            errors.append(
                abs(
                    sum(a * b for a, b in zip(state["heads"]["sparse"], basis, strict=True))
                    - sum(a * b for a, b in zip(state["heads"]["dense"], basis, strict=True))
                )
            )
    if not states:
        return dict(recomputed=False, deliberate_error_rejected=False)
    corrupted = list(states[0]["heads"]["dense"])
    corrupted[1] += 0.125
    basis = kernel.scalar_design([0.0, 0.3, 0.4, 0.5, 0.6])
    wrong = abs(
        sum(
            (a - b) * v
            for a, b, v in zip(corrupted, states[0]["heads"]["sparse"], basis, strict=True)
        )
    )
    return dict(
        recomputed=max(errors) == 0,
        deliberate_error_rejected=wrong != 0,
        observed_error=max(errors),
        deliberate_error=wrong,
    )

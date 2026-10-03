"""REQ-VERIFY-8000: delayed labels update the state that issued a set.

The clipped binary protocol is a timing adaptation of arXiv:2609.07251 section
4. Issue-before-release makes the saved recurrence spacing delay+1. These finite
observations do not establish a deployment coverage guarantee.
"""

from __future__ import annotations

from collections.abc import Callable
import json
import math
from typing import Any

import numpy as np

from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]
ARMS = ("frozen", "scalar", "interleaved")
CONFIG = dict(
    alpha=0.10,
    gamma=0.01,
    clip=[0.01, 0.50],
    pool=64,
    delays=[20, 24, 36],
    temperatures=[0.5, 1.0, 2.0],
    windows=[[65, 128], [129, 192], [193, 256]],
    minimum_groups=160,
    minimum_blocks=8,
    block_length=20,
    draws=10000,
    random_seed=69300,
    restart_slot=128,
)
RECURRENCE = dict(
    source="https://arxiv.org/html/2609.07251v1",
    section="4, equations 3 and 8",
    paper="alpha[t+tau] = alpha[t] + gamma*(target-error[t])",
    interleaved="alpha[t+1] = clip(issue_alpha[t-delay] + .01*(.10-error[t-delay]))",
    scalar="alpha[t+1] = clip(current_alpha[t] + .01*(.10-error[t-delay]))",
    timing="issue then release; effective phase spacing delay+1",
    adaptations="binary nonconformity, finite rank, trailing64, clipping; no theorem transfer",
    frozen="fixed calibration cutoff and alpha; adaptive arms use due trailing64 scores",
)


def transform(p: float, temperature: float) -> float:
    """Temperature changes odds, while preserving exact zero and one inputs."""
    return p if p in (0.0, 1.0) else 1 / (1 + math.exp(-math.log(p / (1 - p)) / temperature))


def calibrate(rows: list[Json]) -> tuple[float, list[float]]:
    """Ascending ties prevent choosing a temperature after stream labels arrive."""
    losses = [
        sum((transform(r["p"], t) - r["y"]) ** 2 for r in rows) / len(rows)
        for t in CONFIG["temperatures"]
    ]
    t = float(CONFIG["temperatures"][int(np.argmin(losses))])
    return t, [transform(r["p"], t) if r["y"] == 0 else 1 - transform(r["p"], t) for r in rows]


def cutoff(scores: list[float], alpha: float) -> float | None:
    """None encodes infinity in strict JSON when the corrected rank exceeds n."""
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    return None if rank > len(scores) else float(sorted(scores)[rank - 1])


def prediction_set(p: float, threshold: float | None) -> list[int]:
    """Label zero means supported, so its nonconformity is unsupported probability."""
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("probability")
    return [y for y, score in ((0, p), (1, 1 - p)) if threshold is None or score <= threshold]


def action(labels: list[int]) -> str:
    """Ambiguous and empty sets both need a human decision."""
    return "accept" if labels == [0] else ("reject" if labels == [1] else "escalate")


def initial(scores: list[float], delay: int, arm: str) -> Json:
    """Pending rows and the score pool form the complete restartable state."""
    return dict(
        alpha=0.1,
        delay=delay,
        arm=arm,
        pool=scores[-64:],
        fixed_cutoff=cutoff(scores, 0.1),
        pending=[],
        issued=[],
        feedback=[],
    )


def step(state: Json, row: Json, slot: int, target: Callable[[str], int | None]) -> None:
    """Seal this issue before opening any due outcome, including excluded slots."""
    usable = row["p"] is not None and row["status"] == "completed"
    alpha = 0.1 if state["arm"] == "frozen" else state["alpha"]
    q = state["fixed_cutoff"] if state["arm"] == "frozen" else cutoff(state["pool"], alpha)
    labels = prediction_set(row["p"], q) if usable else []
    issued = dict(
        row,
        slot=slot,
        issue_slot=slot,
        due_slot=slot + state["delay"],
        phase=(slot - 1) % (state["delay"] + 1),
        issue_alpha=alpha,
        cutoff=q,
        prediction_set=labels,
        action=action(labels),
        arm=state["arm"],
        delay=state["delay"],
        seed=69300,
        eligibility=usable,
        numerator=int(usable),
        denominator=1,
        failure_status=row["status"] == "failed",
        censor_status=row["status"] == "censored",
    )
    state["issued"].append(issued)
    state["pending"].append(issued)
    due = [r for r in state["pending"] if r["due_slot"] == slot]
    for old in due:
        y = target(old["family_id"]) if old["eligibility"] else None
        eligible = old["eligibility"] and y in (0, 1)
        error = int(y not in old["prediction_set"]) if eligible else None
        base = old["issue_alpha"] if state["arm"] == "interleaved" else state["alpha"]
        after = max(0.01, min(0.50, base + 0.01 * (0.1 - error))) if eligible else base
        if state["arm"] != "frozen":
            state["alpha"] = after
        if eligible:
            state["pool"] = (state["pool"] + [old["p"] if y == 0 else 1 - old["p"]])[-64:]
        state["feedback"].append(
            dict(
                old,
                y=y,
                error=error,
                release_slot=slot,
                feedback_identity=old["family_id"],
                base_alpha=base,
                alpha_after=state["alpha"],
                eligibility=eligible,
                numerator=error if eligible else 0,
                censor_status=old["censor_status"] or y is None,
            )
        )
    state["pending"] = [r for r in state["pending"] if r["due_slot"] > slot]


def resume(bundle: Json, state: Json, start: int) -> Json:
    """A saved state consumes only later slots, retaining pending feedback identity."""
    t, _ = calibrate(bundle["calibration"])
    for slot in range(start, len(bundle["stream"]) + 1):
        original = bundle["stream"][slot - 1]
        row = dict(original, p=transform(original["p"], t) if original["p"] is not None else None)
        step(state, row, slot, bundle["targets"].__getitem__)
        if slot % 64 == 0:
            print(
                f"[exp8000] replay arm={state['arm']} delay={state['delay']} slot={slot}",
                flush=True,
            )
    return state


def run(bundle: Json, arm: str, delay: int, restart: bool = False) -> Json:
    """The persisted midpoint is sufficient for a separate process to resume."""
    t, scores = calibrate(bundle["calibration"])
    state = initial(scores, delay, arm)
    for slot, original in enumerate(bundle["stream"][:128], 1):
        row = dict(original, p=transform(original["p"], t) if original["p"] is not None else None)
        step(state, row, slot, bundle["targets"].__getitem__)
    checkpoint = json.loads(json.dumps(state))
    state = json.loads(json.dumps(checkpoint)) if restart else state
    state = resume(bundle, state, 129)
    state["checkpoint128"] = checkpoint
    return state


def controls() -> Json:
    """Controlled stale state detects a wrong update base; a constant stream stays stable."""
    row = dict(family_id="control", source_cluster_id="control", p=0.99, status="completed")
    states = [initial(list(np.linspace(0.01, 0.99, 64)), 20, a) for a in ("scalar", "interleaved")]
    for state in states:
        step(state, row, 1, lambda _: 0)
        state["alpha"] = 0.49
        step(state, dict(row, family_id="later"), 21, lambda _: 0)
    a, b = (s["feedback"][0] for s in states)
    responsive = math.isclose(b["alpha_after"], 0.091) and math.isclose(a["alpha_after"], 0.481)
    thresholds = [cutoff(s["pool"], s["alpha"]) for s in states]
    changed = prediction_set(0.7, thresholds[0]) != prediction_set(0.7, thresholds[1])
    stable = []
    for arm in ("scalar", "interleaved"):
        state = initial([0.1] * 64, 20, arm)
        for slot in range(1, 257):
            step(state, dict(row, family_id=str(slot), p=0.1), slot, lambda _: 0)
        stable.append(all(r["prediction_set"] == [0] for r in state["issued"]))
    return dict(
        passed=responsive and changed and all(stable),
        known_responsive=dict(passed=responsive and changed, scalar=a, interleaved=b),
        no_shift=dict(passed=all(stable)),
        verdict_class="circular_positive",
        scope="protocol fixtures only; no independent natural evidence",
    )


def comparison(states: Json, delay: int) -> Json:
    """Paired complete slot blocks preserve local dependence in this finite replay."""
    arms = {
        a: [
            r
            for r in states[f"{a}-{delay}"]["feedback"]
            if r["eligibility"] and 65 <= r["slot"] <= 256
        ]
        for a in ARMS
    }
    deviations, windows = {}, []
    for arm, rows in arms.items():
        values = []
        for lo, hi in CONFIG["windows"]:
            selected = [r for r in rows if lo <= r["slot"] <= hi]
            n, errors = len(selected), sum(r["error"] for r in selected)
            dev = abs(errors / n - 0.1) if n else None
            windows.append(
                dict(
                    arm=arm,
                    delay=delay,
                    start=lo,
                    end=hi,
                    numerator=errors,
                    denominator=n,
                    deviation=dev,
                    eligibility=n > 0,
                    failure_status=False,
                    censor_status=n < hi - lo + 1,
                )
            )
            if n:
                values.append(dev)
        deviations[arm] = float(np.mean(values)) if values else None
    maps = {a: {r["slot"]: r for r in arms[a]} for a in ("scalar", "interleaved")}
    blocks, block_windows = [], []
    for lo in range(65, 237, 20):
        slots = list(range(lo, lo + 20))
        if all(t in maps["scalar"] and t in maps["interleaved"] for t in slots):
            blocks.append(
                [sum(maps[a][t]["error"] for t in slots) / 20 for a in ("scalar", "interleaved")]
            )
            block_windows.append(
                [
                    [
                        sum(maps[a][t]["error"] for t in slots if wlo <= t <= whi)
                        for a in ("scalar", "interleaved")
                    ]
                    + [sum(wlo <= t <= whi for t in slots)]
                    for wlo, whi in CONFIG["windows"]
                ]
            )
    groups = len({r["source_cluster_id"] for r in arms["scalar"]})
    lower = None
    if blocks:
        rng = np.random.default_rng(CONFIG["random_seed"])
        matrix = np.asarray(block_windows, dtype=float)
        samples = []
        for start in range(0, 10000, 1000):
            draw = matrix[rng.integers(0, len(blocks), (1000, len(blocks)))]
            totals = draw.sum(axis=1)
            denominators = totals[:, :, 2]
            valid = denominators > 0
            rates = totals[:, :, :2] / np.maximum(denominators[:, :, None], 1)
            dev = (np.abs(rates - 0.1) * valid[:, :, None]).sum(axis=1) / np.maximum(
                valid.sum(axis=1)[:, None], 1
            )
            samples.extend((dev[:, 0] - dev[:, 1]).tolist())
            print(f"[exp8000] bootstrap delay={delay} draws={start + 1000}", flush=True)
        lower = float(np.quantile(samples, 0.025))
    scalar, inter = arms["scalar"], arms["interleaved"]
    gain = deviations["scalar"] - deviations["interleaved"] if scalar else None
    size = (
        float(
            np.mean([len(r["prediction_set"]) for r in inter])
            - np.mean([len(r["prediction_set"]) for r in scalar])
        )
        if scalar
        else None
    )
    false = {a: sum(r["action"] == "accept" and r["y"] == 1 for r in arms[a]) for a in ARMS}
    gates = dict(
        support=groups >= 160 and len(blocks) >= 8,
        deviation_gain=gain is not None and gain >= 0.02,
        lower_positive=lower is not None and lower > 0,
        set_size=size is not None and size <= 0.10,
        false_accepts=false["interleaved"] <= false["scalar"],
    )
    return dict(
        delay=delay,
        deviations=deviations,
        gain=gain,
        lower95=lower,
        set_size_increase=size,
        false_accepts=false,
        eligible_groups=groups,
        complete_blocks=len(blocks),
        block_error_rates=blocks,
        bootstrap_scope="Complete20-slot blocks, recomputed fixed-window deviations; conditional on observed order",
        gates=gates,
        benefit=all(gates.values()),
        windows=windows,
    )


def measure(bundle: Json) -> Json:
    """All policies share sealed point probabilities; repeats do not increase support."""
    states, restart = {}, []
    for delay in CONFIG["delays"]:
        for arm in ARMS:
            key = f"{arm}-{delay}"
            states[key] = run(bundle, arm, delay)
            replayed = run(bundle, arm, delay, restart=True)
            restart.append(
                dict(
                    arm=arm,
                    delay=delay,
                    slot=128,
                    passed=canonical_hash(states[key]) == canonical_hash(replayed),
                )
            )
    contrasts = [comparison(states, d) for d in CONFIG["delays"]]
    rows = [r for s in states.values() for r in s["feedback"]]
    issued = [r for s in states.values() for r in s["issued"]]
    briers = {
        key: sum((r["p"] - r["y"]) ** 2 for r in s["feedback"] if r["eligibility"])
        / max(1, sum(r["eligibility"] for r in s["feedback"]))
        for key, s in states.items()
    }
    parity = all(len({briers[f"{a}-{d}"] for a in ARMS}) == 1 for d in CONFIG["delays"])
    return dict(
        issued_state_rows=issued,
        rows=rows,
        pending_feedback={k: s["pending"] for k, s in states.items()},
        coverage_windows=[w for c in contrasts for w in c["windows"]],
        restart_rows=restart,
        restart_states={k: s["checkpoint128"] for k, s in states.items()},
        delay_sensitivity=contrasts,
        point_brier_parity=dict(passed=parity, values=briers),
        point_probability_hash=canonical_hash(
            [(r["family_id"], r["p"]) for r in states["scalar-20"]["issued"]]
        ),
        acceptance_gate_results=contrasts[0]["gates"],
        confidence_benefit_score=int(contrasts[0]["benefit"]),
        positive_control_results=controls(),
        genuine_headroom=dict(
            errors=sum(r["error"] for r in states["scalar-20"]["feedback"] if r["eligibility"]),
            scope="observed errors permit oracle improvement; no optimality claim",
        ),
    )

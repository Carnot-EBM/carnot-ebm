"""REQ-VERIFY-7999: derive replay from primitives without producer decision code.

The small cubic spline is rebuilt here so a shared acceptance function cannot
silently make both the producer and its auditor agree on the same error.
"""

from __future__ import annotations

import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline  # type: ignore[import-untyped]
from scipy.special import expit  # type: ignore[import-untyped]

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash

Json = dict[str, Any]
Array = NDArray[np.float64]
ARMS = ("targeted_ipw", "targeted_unweighted", "uniform_ipw", "full_feedback", "frozen_no_write")
CONTROLS = ("frozen_no_write", "uniform_ipw", "targeted_unweighted")
CONFIG = dict(
    random_seed=69399,
    suffix=[41, 256],
    bootstrap_draws=10000,
    block_length=20,
    sensitivity=[10, 40],
    minimum=160,
    per_class=20,
    minimum_blocks=8,
    retention_minimum=48,
    retention_per_class=8,
)


class Crash(Exception):
    """Interrupt at an atomic boundary so recovery must use durable bytes."""


def geometry(head: Json, sources: list[Json]) -> tuple[Array, Array]:
    """Use only public observations and the initial head's frozen scaler."""
    raw = np.array(
        [
            [r["q"], *r["features"]]
            if r["q"] is not None and r["features"] is not None
            else [0.0] * 9
            for r in sources
        ],
        dtype=float,
    )
    low, high = np.array(head["scaler"]["minimum"]), np.array(head["scaler"]["maximum"])
    x = np.clip((raw - low) / np.where(high > low, high - low, 1), 0, 1)
    knots = [0.0] * 4 + [i / 9 for i in range(1, 9)] + [1.0] * 4
    matrix = np.column_stack(
        [BSpline.design_matrix(x[:, j], knots, 3).toarray() for j in range(9)] + [np.ones(len(x))]
    )
    q = np.clip(raw[:, 0], 1e-4, 1 - 1e-4)
    return matrix, np.log(q / (1 - q))


def probability(head: Json, matrix: Array, offset: float) -> float:
    """Normalize the two energies directly, preserving the producer arithmetic."""
    theta = np.asarray(head["parameters"]) * head["decay_scale"]
    z = (offset + (matrix[None, :] @ theta)[0]) / head["temperature"]
    energies = np.array([[0.0, -z]])
    weights = np.exp(-energies - np.max(-energies, axis=1, keepdims=True))
    return float(weights[0, 1] / weights.sum(axis=1)[0])


def action(p: float | None) -> str:
    """Choose the cheapest registered action with the original fixed tie order."""
    if p is None:
        return "escalate"
    return min(((5 * p, 1, "accept"), (1 - p, 2, "reject"), (0.25, 0, "escalate")))[2]


def acquire(head: Json, sources: list[Json], seed: int, arm: str) -> list[Json]:
    """Rebuild source-keyed sampling and equal expected uniform budgets."""
    if any(
        set(r) != {"family_id", "source_cluster_id", "q", "features", "status"} for r in sources
    ):
        raise ValueError("public_fields")
    matrix, offsets = geometry(head, sources)
    ps = [
        probability(head, matrix[i], float(offsets[i]))
        if r["status"] == "completed" and r["q"] is not None and r["features"] is not None
        else None
        for i, r in enumerate(sources)
    ]
    targeted = [0.0 if p is None else (0.5 if 0.05 <= p <= 0.75 else 0.125) for p in ps]
    uniform = sum(targeted) / max(1, sum(p is not None for p in ps))
    rows = []
    for slot, (r, p, target) in enumerate(zip(sources, ps, targeted, strict=True)):
        draw = (
            int(
                canonical_hash(dict(source=r["source_cluster_id"], seed=seed)).split(":")[1][:13],
                16,
            )
            / 16**13
        )
        pi = (
            target
            if arm.startswith("targeted")
            else (uniform if arm == "uniform_ipw" else float(arm == "full_feedback"))
        )
        pi = pi if p is not None else 0.0
        rows.append(
            dict(
                r,
                arm=arm,
                seed=seed,
                slot=slot,
                pi=pi,
                draw=draw,
                initial_probability=p,
                selected=draw < pi,
                numerator=pi,
                denominator=1,
                eligibility=p is not None,
                failure_status=r["status"] == "failed",
                censor_status=r["status"] == "censored",
            )
        )
    return rows


def execute(
    head: Json,
    sources: list[Json],
    schedule: list[Json],
    targets: Json,
    directory: Path | None = None,
    crash: str | None = None,
) -> Json:
    """Issue each prediction before its due update, restoring only atomic commits."""
    arm, seed = schedule[0]["arm"], schedule[0]["seed"]
    matrix, offsets = geometry(head, sources)
    state = dict(
        head=copy.deepcopy(head),
        next_slot=0,
        seen_label_ids=[],
        rng_state=dict(algorithm="sha256_source_seed_counter", seed=seed, next_slot=0),
        pending_feedback=[
            dict(family_id=r["family_id"], origin_slot=r["slot"], due_slot=r["slot"] + 20)
            for r in schedule
            if r["selected"]
        ],
    )
    records: list[Json] = []
    if directory:
        directory.mkdir(parents=True, exist_ok=True)
        for path in sorted(directory.glob("committed-*.json")):
            records.append(json.loads(path.read_text()))
        if records:
            state = records[-1]["state"]
        marker = directory / "state.json"
        if (records and records[-1]["checksum"] != canonical_hash(state)) or (
            marker.exists() and json.loads(marker.read_text())["checksum"] != canonical_hash(state)
        ):
            raise ValueError("checkpoint_checksum")
    for slot in range(state["next_slot"], len(sources)):
        r, selected = sources[slot], schedule[slot]
        p = (
            probability(state["head"], matrix[slot], float(offsets[slot]))
            if selected["eligibility"]
            else None
        )
        prediction = dict(
            family_id=r["family_id"],
            source_cluster_id=r["source_cluster_id"],
            slot=slot,
            arm=arm,
            seed=seed,
            probability=p,
            action=action(p),
            numerator=p,
            denominator=int(p is not None),
            eligibility=p is not None,
            failure_status=selected["failure_status"],
            censor_status=selected["censor_status"],
            head_checksum=canonical_hash(state["head"]),
        )
        if directory:
            atomic_json(
                directory / f"issued-{slot:04d}.json",
                dict(prediction=prediction, state_checksum=canonical_hash(state)),
            )
        reveals, updates, gradients = [], [], []
        origin = slot - 20
        if origin >= 0 and schedule[origin]["selected"]:
            old = schedule[origin]
            receipt = dict(
                family_id=old["family_id"],
                source_cluster_id=old["source_cluster_id"],
                origin_slot=origin,
                due_slot=slot,
                arm=arm,
                seed=seed,
                receipt_id=f"{arm}:{seed}:{origin}",
                pi=old["pi"],
                weight=1 / old["pi"] if arm.endswith("ipw") else 1.0,
                y=targets[old["family_id"]],
                numerator=1,
                denominator=1,
                eligibility=True,
                failure_status=False,
                censor_status=False,
            )
            reveals.append(receipt)
            state["seen_label_ids"].append(receipt["receipt_id"])
            if receipt["y"] is not None:
                h = state["head"]
                ids = np.array([108, *np.flatnonzero(matrix[origin, :108])], dtype=int)
                values = matrix[origin, ids]
                local = np.array([h["parameters"][int(i)] for i in ids]) * h["decay_scale"]
                z = offsets[origin] + values @ local
                gradient = values * (expit(z / h["temperature"]) - receipt["y"]) / h["temperature"]
                scale = h["decay_scale"] * (1 - 0.002 * 0.01)
                for i, g in zip(ids, gradient, strict=True):
                    h["parameters"][int(i)] -= 0.01 * receipt["weight"] * float(g) / scale
                h["decay_scale"] = scale
                updates.append(
                    dict(
                        receipt,
                        coefficient_touches=len(ids),
                        global_decay_writes=1,
                        logical_decay_coefficients=109,
                        numerator=len(ids),
                        denominator=109,
                        device="cpu",
                    )
                )
                gradients.append(
                    dict(
                        receipt,
                        coefficient_ids=ids.tolist(),
                        data_gradient=gradient.tolist(),
                        gradient_error=0.0,
                        head_after=copy.deepcopy(h),
                    )
                )
            state["pending_feedback"] = [
                r for r in state["pending_feedback"] if r["origin_slot"] != origin
            ]
        state["next_slot"] = slot + 1
        state["rng_state"]["next_slot"] = slot + 1
        record = dict(
            state=copy.deepcopy(state),
            checksum=canonical_hash(state),
            prediction=prediction,
            reveal_rows=reveals,
            update_rows=updates,
            gradient_rows=gradients,
        )
        if directory:
            if slot == 128 and crash == "before":
                raise Crash("before")
            atomic_json(directory / f"committed-{slot:04d}.json", record)
            if slot == 128 and crash == "after":
                raise Crash("after")
        records.append(record)
        if slot % 64 == 0:
            print(f"[exp7999] arm={arm} seed={seed} slots={slot + 1}/{len(sources)}", flush=True)
    if directory:
        atomic_json(directory / "state.json", dict(checksum=canonical_hash(state)))
    return dict(
        final_state=state,
        issued_predictions=[r["prediction"] for r in records],
        reveal_rows=[v for r in records for v in r["reveal_rows"]],
        update_rows=[v for r in records for v in r["update_rows"]],
        gradient_rows=[v for r in records for v in r["gradient_rows"]],
        state_checksums=[r["checksum"] for r in records],
    )


def reduce(
    head: Json, sources: list[Json], schedule: list[Json], saved: Json, targets: Json
) -> Json:
    """Reject primitive drift and compare gradients with saved coefficient changes."""
    if schedule != acquire(head, sources, schedule[0]["seed"], schedule[0]["arm"]):
        raise ValueError("acquisition_drift")
    got = execute(head, sources, schedule, targets)
    original = saved["trajectory"]
    for field in ("issued_predictions", "reveal_rows", "update_rows", "final_state"):
        if got[field] != original[field]:
            raise ValueError(field + "_drift")
    if got["state_checksums"] != [r["checksum"] for r in original["checkpoint_rows"]]:
        raise ValueError("state_drift")
    directory = Path(saved["state_directory"])
    for row in got["gradient_rows"]:
        record = json.loads((directory / f"committed-{row['due_slot']:04d}.json").read_text())
        if record["checksum"] != canonical_hash(record["state"]):
            raise ValueError("saved_checkpoint")
        previous = json.loads(
            (directory / f"committed-{row['due_slot'] - 1:04d}.json").read_text()
        )["state"]["head"]
        current = record["state"]["head"]
        ids = row["coefficient_ids"]
        observed = (
            (np.asarray(previous["parameters"])[ids] - np.asarray(current["parameters"])[ids])
            * current["decay_scale"]
            / (0.01 * row["weight"])
        )
        row["gradient_error"] = float(np.max(np.abs(observed - row["data_gradient"])))
        if row["gradient_error"] > 1e-10 or current != row.pop("head_after"):
            raise ValueError("saved_gradient")
    matrix, offsets = geometry(head, sources)
    changed = any(
        r["probability"] is not None
        and abs(r["probability"] - probability(head, matrix[i], float(offsets[i]))) > 1e-12
        for i, r in enumerate(got["issued_predictions"])
    )
    rows = []
    for p in got["issued_predictions"]:
        y = targets[p["family_id"]]
        eligible = p["probability"] is not None and y is not None
        cost = (
            0.25
            if p["action"] == "escalate"
            else (
                5 * y
                if p["action"] == "accept" and y is not None
                else (1 - y if y is not None else None)
            )
        )
        rows.append(
            dict(
                p,
                y=y,
                eligibility=eligible,
                cost=cost,
                brier=(p["probability"] - y) ** 2 if eligible else None,
                false_accept=int(p["action"] == "accept" and y == 1),
                numerator=cost,
                denominator=int(eligible),
            )
        )
    return dict(got, rows=rows, prediction_changed=changed)


def bootstrap(diff: Array, length: int) -> Json:
    """Paired moving blocks diagnose dependence within this single adaptive stream."""
    if not np.isfinite(diff).any():
        return dict(gain=None, interval=[None, None], raw_p=1.0)
    rng = np.random.default_rng(CONFIG["random_seed"])
    length = min(length, len(diff))
    starts = rng.integers(0, len(diff) - length + 1, (10000, int(np.ceil(len(diff) / length))))
    indices = (starts[:, :, None] + np.arange(length)).reshape(10000, -1)[:, : len(diff)]
    samples = diff[indices]
    counts = np.isfinite(samples).sum(axis=1)
    sampled = np.nansum(samples[counts > 0], axis=1) / counts[counts > 0]
    centered = sampled - np.nanmean(diff)
    return dict(
        gain=float(np.nanmean(diff)),
        interval=np.quantile(sampled, [0.025, 0.975]).tolist(),
        raw_p=float((1 + np.sum(centered >= np.nanmean(diff))) / 10001),
        intended_draws=10000,
        completed_draws=len(sampled),
        censored_draws=10000 - len(sampled),
    )


def holm(values: dict[str, float]) -> dict[str, float]:
    """Correct all registered contrasts together so selection cannot hide a loser."""
    adjusted, previous = {}, 0.0
    for i, (name, p) in enumerate(sorted(values.items(), key=lambda r: r[1])):
        previous = max(previous, min(1.0, (len(values) - i) * p))
        adjusted[name] = previous
    return adjusted


def grouped(rows: list[Json]) -> list[Json]:
    """Average schedules inside each source and arm before any resampling."""
    buckets: Json = defaultdict(list)
    for r in rows:
        if r["eligibility"]:
            buckets[(r["source_cluster_id"], r["arm"])].append(r)
    return [
        dict(
            v[0],
            **{k: float(np.mean([r[k] for r in v])) for k in ("cost", "brier", "false_accept")},
        )
        for v in buckets.values()
    ]


def support(rows: list[Json], minimum: int, per_class: int) -> Json:
    """Count source groups, with class support independent of schedule count."""
    ids = {r["source_cluster_id"] for r in rows if r["eligibility"]}
    counts = {
        str(y): len({r["source_cluster_id"] for r in rows if r["eligibility"] and r["y"] == y})
        for y in (0, 1)
    }
    return dict(
        independent=len(ids),
        class_counts=counts,
        minimum=minimum,
        per_class=per_class,
        passed=len(ids) >= minimum and min(counts.values()) >= per_class,
    )


def compare(rows: list[Json]) -> Json:
    """Reduce the fixed suffix using three controls and a single frozen gate."""
    averaged = grouped([r for r in rows if 40 <= r["slot"] <= 255])
    by_arm = {a: {r["source_cluster_id"]: r for r in averaged if r["arm"] == a} for a in ARMS}
    ids = sorted(
        set.intersection(*(set(by_arm[a]) for a in ("targeted_ipw", *CONTROLS))),
        key=lambda identity: by_arm["targeted_ipw"][identity]["slot"],
    )
    selected = [by_arm["targeted_ipw"][identity] for identity in ids]
    floors = support(selected, 160, 20)
    slots = {r["slot"] for r in selected}
    blocks = sum(set(range(start, start + 20)) <= slots for start in range(40, 237, 20))
    comparisons = {}
    for control in CONTROLS:
        diff = np.array(
            [by_arm[control][cid]["cost"] - by_arm["targeted_ipw"][cid]["cost"] for cid in ids]
        )
        brier = np.array(
            [by_arm["targeted_ipw"][cid]["brier"] - by_arm[control][cid]["brier"] for cid in ids]
        )
        if ids:
            slot_diff, slot_brier = np.full(216, np.nan), np.full(216, np.nan)
            for i, cid in enumerate(ids):
                slot = by_arm["targeted_ipw"][cid]["slot"] - 40
                slot_diff[slot], slot_brier[slot] = diff[i], brier[i]
            diff, brier = slot_diff, slot_brier
        result = bootstrap(diff, 20)
        result.update(
            brier_degradation=bootstrap(brier, 20),
            independent=len(ids),
            false_accept_difference=sum(
                by_arm["targeted_ipw"][cid]["false_accept"] - by_arm[control][cid]["false_accept"]
                for cid in ids
            ),
            sensitivity={str(b): bootstrap(diff, b) for b in (10, 40)},
        )
        comparisons[control] = result
        print(f"[exp7999] bootstrap_complete control={control} groups={len(ids)}", flush=True)
    adjusted = holm({k: v["raw_p"] for k, v in comparisons.items()})
    for name, result in comparisons.items():
        result["adjusted_p"] = adjusted[name]
    benefit = (
        floors["passed"]
        and blocks >= 8
        and all(
            r["gain"] is not None
            and r["gain"] >= 0.02
            and r["interval"][0] > 0
            and r["adjusted_p"] < 0.05
            and r["false_accept_difference"] <= 0
            and r["brier_degradation"]["interval"][1] <= 0.01
            for r in comparisons.values()
        )
    )
    frozen = by_arm["frozen_no_write"]
    selectable = {
        a: sum(abs(by_arm[a][cid]["cost"] - frozen[cid]["cost"]) > 1e-12 for cid in ids)
        for a in ARMS
    }
    return dict(
        paired_comparisons=comparisons,
        support_by_class=floors,
        effective_blocks=blocks,
        benefit=bool(benefit),
        headroom_diagnostics=dict(
            initial_cost=sum(frozen[cid]["cost"] for cid in ids),
            selectable_changes=selectable,
            natural_null_inconclusive=not any(selectable.values()),
        ),
        dependence_limits="Moving-block intervals are dependence diagnostics for one adaptive development stream; they do not guarantee population coverage or deployment generalization.",
    )


def retention(predictions: list[Json], targets: Json) -> Json:
    """Evaluate sealed predictions; there is no update or threshold selection here."""
    rows = []
    for r in predictions:
        p, y = r["probability"], targets[r["family_id"]]
        eligible = p is not None and y is not None
        a = action(p)
        cost = (
            0.25
            if a == "escalate"
            else (5 * y if a == "accept" and y is not None else (1 - y if y is not None else None))
        )
        rows.append(
            dict(
                r,
                y=y,
                action=a,
                eligibility=eligible,
                cost=cost,
                brier=(p - y) ** 2 if eligible else None,
                false_accept=int(a == "accept" and y == 1),
                numerator=cost,
                denominator=int(eligible),
                failure_status=False,
                censor_status=y is None,
            )
        )
    averaged = grouped(rows)
    floors = support(averaged, 48, 8)
    baseline = {r["source_cluster_id"]: r for r in averaged if r["arm"] == "initial"}
    comparisons = {}
    for arm in ("targeted_ipw", *CONTROLS):
        current = [r for r in averaged if r["arm"] == arm]
        cost = bootstrap(
            np.array([r["cost"] - baseline[r["source_cluster_id"]]["cost"] for r in current]), 1
        )
        brier = bootstrap(
            np.array([r["brier"] - baseline[r["source_cluster_id"]]["brier"] for r in current]), 1
        )
        comparisons[arm] = dict(
            cost_degradation=cost,
            brier_degradation=brier,
            false_accepts=sum(r["false_accept"] for r in current),
            initial_false_accepts=sum(
                baseline[r["source_cluster_id"]]["false_accept"] for r in current
            ),
        )
    passed = floors["passed"] and all(
        r["cost_degradation"]["interval"][1] is not None
        and r["cost_degradation"]["interval"][1] <= 0.01
        for r in comparisons.values()
    )
    return dict(
        rows=rows,
        support=floors,
        comparisons=comparisons,
        passed=bool(passed),
        weights_updated=False,
        thresholds_updated=False,
    )


def controls() -> Json:
    """A circular fixture has known headroom and must change future predictions."""
    head = dict(
        parameters=[0.0] * 109,
        temperature=1.0,
        decay_scale=1.0,
        scaler=dict(minimum=[0.0] * 9, maximum=[1.0] * 9),
        arm="spline",
        seed=17,
    )
    results: Json = {}
    for name, q, y in (("known_benefit", 0.5, 1), ("no_headroom", 0.001, 0)):
        sources = [
            dict(
                family_id=f"fixture-{i}",
                source_cluster_id=f"fixture-{i}",
                q=q,
                features=[0.5] * 8,
                status="completed",
            )
            for i in range(256)
        ]
        result = execute(
            head,
            sources,
            acquire(head, sources, 101, "full_feedback"),
            {r["family_id"]: y for r in sources},
        )
        first, last = (
            result["issued_predictions"][0]["probability"],
            result["issued_predictions"][-1]["probability"],
        )
        results[name] = dict(
            future_prediction_changed=abs(first - last) > 1e-6,
            initial_brier=(first - y) ** 2,
            final_brier=(last - y) ** 2,
            genuine_headroom=name == "known_benefit",
            benefit=name == "known_benefit" and (last - y) ** 2 < (first - y) ** 2,
            verdict_class="circular_positive",
            verifier_is_oracle=True,
            independent_natural_evidence=False,
            initial_cost=0.25
            if action(first) == "escalate"
            else (5 * y if action(first) == "accept" else 1 - y),
            final_cost=0.25
            if action(last) == "escalate"
            else (5 * y if action(last) == "accept" else 1 - y),
        )
    results["passed"] = (
        results["known_benefit"]["benefit"]
        and results["known_benefit"]["future_prediction_changed"]
        and not results["no_headroom"]["benefit"]
    )
    return results

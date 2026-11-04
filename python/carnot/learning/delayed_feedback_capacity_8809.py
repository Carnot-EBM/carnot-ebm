"""REQ-VERIFY-8809: overflow discards pending feedback, never a scored prediction.

Both eviction rules are unweighted heuristics. The predictor sees public graphs
and released predecessor labels through the existing numerical energy learner.
"""

from copy import deepcopy
import random
import time
from typing import Any

import numpy as np
from scipy.stats import t

from carnot.learning import causal_template_streams_8804 as s
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]


def manifest(streams: int = 30, queries: int = 64, seeds: list[int] | None = None) -> Json:
    """Freeze release-before-arrival so delay equal to capacity is a true negative control."""
    return dict(
        streams=streams,
        queries=queries,
        seeds=seeds or [11, 23, 37],
        delays=[4, 16, 32],
        capacities=[4, 8],
        policies=["FIFO", "uniform"],
        event_order="release_then_arrival_then_eviction",
        template_bound=32,
        paid_checks=5,
        update_rule="Exp8802 SGD0.03 Brier+0.1NCE, unweighted",
        random_seed=8809,
        retention_queries=12,
    )


def arms() -> list[tuple[int, int | None, str]]:
    """Each delay has one unlimited reference, shared across both capacities."""
    return [
        (d, c, p)
        for d in (4, 16, 32)
        for c, p in [(None, "unlimited"), (4, "FIFO"), (4, "uniform"), (8, "FIFO"), (8, "uniform")]
    ]


def meta(stream: int, seed: int, delay: int, capacity: int | None, policy: str) -> Json:
    """Pair conditions within reset streams; seeds never become independent clusters."""
    return dict(
        cluster_id=f"stream-{stream:02}",
        seed=seed,
        delay=delay,
        capacity=capacity,
        policy=policy,
        arm=f"d{delay}-c{capacity}-{policy}",
    )


def census(plan: Json, reason: str) -> list[Json]:
    """Allocate missing outcomes without converting unavailable measurements to zero."""
    return [
        dict(
            meta(i, seed, d, c, p),
            unit_id=f"capacity-{i}-{q}",
            ordinal=q,
            metric=None,
            denominator=1,
            status="censored",
            missing_reason=reason,
        )
        for i in range(plan["streams"])
        for seed in plan["seeds"]
        for d, c, p in arms()
        for q in range(plan["queries"])
    ]


def cell(
    stream: int, seed: int, delay: int, capacity: int | None, policy: str, head: Json, queries: int
) -> Json:
    """Seal each prediction before grading, and apply only retained feedback when it arrives.

    The release loop considers only the current clock. Eviction consults pending
    identifiers and arrival order, never labels or future expiry information.
    """
    common = meta(stream, seed, delay, capacity, policy)
    rng = random.Random(8809 + stream)
    labels = [s.m.LABELS[i % 3] for i in range(queries)]
    rng.shuffle(labels)
    public = [s.m.world(f"capacity-{stream}-{i}-", label) for i, label in enumerate(labels)]
    eviction = random.Random(8809 + 1009 * stream + seed)
    params, templates = deepcopy(head["parameters"]), []
    pending: dict[int, Json] = {}
    result: Json = dict(
        common,
        head=deepcopy(head),
        queries=queries,
        per_query_rows=[],
        admission_rows=[],
        feedback_loss_rows=[],
        release_rows=[],
        applied_rows=[],
        overflow_event_rows=[],
        retention_rows=[],
        queue_cost_rows=[],
    )
    started = time.monotonic_ns()
    queue_ns = 0
    peak = 0

    def retain(checkpoint: str) -> None:
        state_hash = canonical_hash([params, templates])
        for i in range(12):
            sample = s.m.world(f"retention-{stream}-{i}-", s.m.LABELS[i % 3])
            observation = s.m.observe(sample, templates)
            p = s.probabilities(params, observation["counts"])
            result["retention_rows"].append(
                dict(
                    common,
                    unit_id=f"retention-{stream}-{i}",
                    checkpoint=checkpoint,
                    prediction=p,
                    metric=s.brier(p, str(s.m.decide(sample)["decision"])),
                    state_hash=state_hash,
                    denominator=1,
                    status="completed",
                    missing_reason=None,
                )
            )
        assert state_hash == canonical_hash([params, templates])

    retain("initial")
    tail_start = started
    for now in range(queries + delay):
        if now == queries:
            tail_start = time.monotonic_ns()
        issued = now - delay
        if issued >= 0:
            qstart = time.monotonic_ns()
            saved = pending.pop(issued, None)
            queue_ns += time.monotonic_ns() - qstart
            event = dict(
                common,
                event_id=f"e{issued}",
                issued_time=issued,
                time=now,
                tail=now >= queries,
                applied=saved is not None,
            )
            if saved is not None:
                label = str(s.m.decide(public[issued])["decision"])
                before = canonical_hash([params, templates])
                params, gradient = s.update(params, saved["counts"], label)
                template = s.m.normalize(public[issued])
                if template not in templates and len(templates) < 32:
                    templates.append(template)
                result["applied_rows"].append(
                    dict(
                        event,
                        label=label,
                        **gradient,
                        before_hash=before,
                        after_hash=canonical_hash([params, templates]),
                        weight=1,
                    )
                )
            result["release_rows"].append(event)
        if now < queries:
            observation = s.m.observe(public[now], templates)
            p = s.probabilities(params, observation["counts"])
            sealed = time.monotonic_ns()
            row = dict(
                common,
                unit_id=f"capacity-{stream}-{now}",
                ordinal=now,
                prediction=p,
                counts=observation["counts"],
                paid_checks=5,
                state_hash=canonical_hash([params, templates]),
                sealed_ns=sealed,
                denominator=1,
                status="completed",
                missing_reason=None,
            )
            truth = str(s.m.decide(public[now])["decision"])
            row.update(truth=truth, metric=s.brier(p, truth), scored_ns=time.monotonic_ns())
            result["per_query_rows"].append(row)
            qstart = time.monotonic_ns()
            pending[now] = observation
            attempted = len(pending)
            peak = max(peak, attempted)
            dropped = None
            if capacity is not None and attempted > capacity:
                dropped = (
                    next(iter(pending)) if policy == "FIFO" else eviction.choice(list(pending))
                )
                pending.pop(dropped)
                result["overflow_event_rows"].append(
                    dict(
                        common,
                        time=now,
                        pending_count=attempted,
                        retained_count=len(pending),
                        dropped_event=f"e{dropped}",
                    )
                )
                result["feedback_loss_rows"].append(
                    dict(
                        common,
                        time=now,
                        event_id=f"e{dropped}",
                        issued_time=dropped,
                        missing_reason="queue_eviction",
                    )
                )
            result["admission_rows"].append(
                dict(
                    common,
                    time=now,
                    event_id=f"e{now}",
                    admitted=dropped != now,
                    evicted_event=None if dropped is None else f"e{dropped}",
                    pending_count=attempted,
                    retained_count=len(pending),
                )
            )
            queue_ns += time.monotonic_ns() - qstart
    tail_end = time.monotonic_ns()
    retain("final")
    result["final_state_hash"] = canonical_hash([params, templates])
    result["queue_cost_rows"].append(
        dict(
            common,
            started_ns=started,
            ended_ns=tail_end,
            duration_s=(tail_end - started) / 1e9,
            queue_service_s=queue_ns / 1e9,
            tail_started_ns=tail_start,
            tail_ended_ns=tail_end,
            tail_drain_s=(tail_end - tail_start) / 1e9,
            tail_applied=sum(e["tail"] for e in result["applied_rows"]),
            peak_pending_count=peak,
            pending_after_drain=len(pending),
            template_count=len(templates),
        )
    )
    return result


def controls(cells: list[Json]) -> Json:
    """A capacity effect needs actual loss of feedback and a matching no-overflow reference."""
    indexed = {
        (c["cluster_id"], c["seed"], c["delay"], c["capacity"], c["policy"]): c for c in cells
    }
    overloaded = [c for c in cells if c["capacity"] is not None and c["delay"] > c["capacity"]]
    negative = [c for c in cells if c["delay"] == 4 and c["capacity"] == 8]
    different = False
    for c in cells:
        if c["policy"] == "FIFO":
            peer = indexed[(c["cluster_id"], c["seed"], c["delay"], c["capacity"], "uniform")]
            different |= [e["event_id"] for e in c["applied_rows"]] != [
                e["event_id"] for e in peer["applied_rows"]
            ]
    equal = True
    for c in negative:
        peer = indexed[(c["cluster_id"], c["seed"], 4, None, "unlimited")]
        equal &= c["final_state_hash"] == peer["final_state_hash"]
        equal &= [q["prediction"] for q in c["per_query_rows"]] == [
            q["prediction"] for q in peer["per_query_rows"]
        ]
        equal &= [e["after_hash"] for e in c["applied_rows"]] == [
            e["after_hash"] for e in peer["applied_rows"]
        ]
        equal &= not c["feedback_loss_rows"]
    return dict(
        overflow_positive=bool(overloaded) and all(c["overflow_event_rows"] for c in overloaded),
        overloaded_peaks=bool(overloaded)
        and all(c["queue_cost_rows"][0]["peak_pending_count"] > c["capacity"] for c in overloaded),
        differing_admissions=bool(different),
        negative_control=bool(negative) and bool(equal),
    )


def interval(values: list[float], alpha: float) -> Json:
    """Use paired stream means; nested seeds and query siblings cannot narrow the interval."""
    mean = float(np.mean(values)) if values else None
    radius = (
        float(t.ppf(1 - alpha / 2, len(values) - 1) * np.std(values, ddof=1) / np.sqrt(len(values)))
        if len(values) > 1
        else None
    )
    if len(values) > 1 and min(values) == max(values):
        radius = 0.0
    return dict(
        mean=mean,
        lower=None if radius is None else mean - radius,
        upper=None if radius is None else mean + radius,
        confidence=1 - alpha,
        independent_count=len(values),
        method="paired Student t over reset-stream means after averaging nested seeds",
    )


def summarize(cells: list[Json]) -> Json:
    """Freeze six primary contrasts; other comparisons remain descriptive stream intervals."""
    means = {
        (c["cluster_id"], c["seed"], c["delay"], c["capacity"], c["policy"]): (
            float(np.mean([q["metric"] for q in c["per_query_rows"]])),
            float(
                np.mean([q["metric"] for q in c["retention_rows"] if q["checkpoint"] == "final"])
            ),
        )
        for c in cells
    }
    streams = sorted({c["cluster_id"] for c in cells})
    seeds = sorted({c["seed"] for c in cells})

    def contrast(d: int, c: int, policy: str, peer: str, metric: int, alpha: float) -> Json:
        values = [
            float(
                np.mean(
                    [
                        means[(cluster, seed, d, c, policy)][metric]
                        - means[(cluster, seed, d, None if peer == "unlimited" else c, peer)][
                            metric
                        ]
                        for seed in seeds
                    ]
                )
            )
            for cluster in streams
        ]
        return interval(values, alpha)

    simultaneous, effects = [], []
    valid = all(controls(cells).values())
    for d in (4, 16, 32):
        for c in (4, 8):
            primary = dict(
                delay=d,
                capacity=c,
                contrast="FIFO_minus_uniform",
                **contrast(d, c, "FIFO", "uniform", 0, 0.05 / 6),
            )
            retention = contrast(d, c, "FIFO", "uniform", 1, 0.05 / 6)
            primary.update(
                retention_interval=retention,
                family_size=6,
                interpretable=valid,
                useful=valid
                and primary["upper"] is not None
                and primary["upper"] < -0.01
                and retention["upper"] <= 0.02,
            )
            simultaneous.append(primary)
            for policy in ("FIFO", "uniform"):
                effects.append(
                    dict(
                        delay=d,
                        capacity=c,
                        policy=policy,
                        brier_minus_unlimited=contrast(d, c, policy, "unlimited", 0, 0.05),
                        retention_minus_unlimited=contrast(d, c, policy, "unlimited", 1, 0.05),
                    )
                )
    return dict(
        simultaneous_intervals=simultaneous,
        per_cell_effects=effects,
        useful_pending_feedback_policy_score=int(any(r["useful"] for r in simultaneous)),
    )


def semantic(value: Any) -> Any:
    """Reconstruction ignores wall-clock variation while checking every scientific operand."""
    if isinstance(value, dict):
        return {
            k: semantic(v)
            for k, v in value.items()
            if not k.endswith(("_ns", "_s")) and k != "queue_cost_rows"
        }
    if isinstance(value, list):
        return [semantic(v) for v in value]
    return value

"""REQ-VERIFY-8810: audit sealed probabilities without using producer reductions.

The same primitive verifier checks graph labels and active predicates. This
module reconstructs the census, state identities and statistical comparisons.
No gradient is evaluated and no model is loaded by this audit.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from carnot.memory import causal_memory_8802 as primitive
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8810_v754_causal_learning_audit"
UPSTREAM = "results/experiment_8804_v754_causal_template_streams.json"
PIN = "sha256:d4881be7f883ed52c87517d82587443dcd9260762233f83db95415d90993dab5"
PROTOTYPE = "results/experiment_8802_v754_causal_memory_prototype.json"
PROTOTYPE_PIN = "sha256:ddbdbb66aa7118e6a03bb702d9f51e91a346337d1d89fa7befc9469ef1b8d497"
MODEL_SPECS: list[Json] = []


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual work counts so a long read cannot resemble an abandoned run."""
    print(f"[exp8810] phase={phase} completed={completed} pending={pending}", flush=True)


def reference(path: Path) -> Json:
    """Bind operands to bytes, including files outside the current checkout."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path))


def load(ref: Json) -> Json:
    """An outer summary cannot authorize changed sealed input bytes."""
    path = Path(ref["path"])
    if sha256_file(path) != ref["sha256"]:
        raise ValueError("operand_hash:" + str(path))
    value: Json = json.loads(path.read_bytes())
    return value


def require(passed: bool, reason: str) -> None:
    """Use an explicit exception so checks remain active under optimized Python."""
    if not passed:
        raise ValueError(reason)


def score(p: list[float], truth: str, threshold: float) -> Json:
    """Score a proper three-class loss; abstention remains separate from rejection."""
    require(
        len(p) == 3 and all(math.isfinite(x) and 0 <= x <= 1 for x in p) and abs(sum(p) - 1) < 1e-6,
        "probability_distribution",
    )
    action = primitive.action(p, threshold)
    return dict(
        metric=sum((x - int(i == primitive.LABELS.index(truth))) ** 2 for i, x in enumerate(p)),
        action=action,
        false_accept=int(action == "accept" and truth != "support"),
        false_reject=int(action == "reject" and truth == "support"),
        covered=int(action != "escalate"),
    )


def census(frozen: Json, reason: str) -> list[Json]:
    """Schedule every identity before opening producer rows, including absent slots."""
    return [
        dict(
            unit_id=recipe["unit_id"] + (f":step{step}" if step is not None else ""),
            world_id=recipe["unit_id"],
            cluster_id=stream["stream_id"],
            seed=seed,
            arm=arm,
            split=split,
            checkpoint=step,
            ordinal=i,
            denominator=1,
            metric=None,
            status="censored",
            missing_reason=reason,
        )
        for stream in frozen["streams"]
        for seed in frozen["seeds"]
        for arm in frozen["arms"]
        for split, key, steps in [
            ("online", "worlds", [None]),
            ("retention", "retention", frozen["retention_checkpoints"]),
        ]
        for step in steps
        for i, recipe in enumerate(stream[key])
    ]


def identity(row: Json) -> tuple[Any, ...]:
    """Sibling conditions and seeds are paired cells, not independent observations."""
    return row["cluster_id"], row["seed"], row["arm"], row["unit_id"]


def release(state: Json, row: Json, payload: Json, arm: str) -> bool:
    """Rebuild symbolic transaction fields; coefficients are authenticated snapshots, not refits."""
    event = row["event_id"]
    pending = state["pending"].pop(event)
    label = str(primitive.decide(payload)["decision"])
    prefix = state["released"] + [label]
    ordinal = pending["ordinal"] + 8
    training = prefix[(ordinal * 7) % len(prefix)] if arm == "shuffled_feedback" else label
    template = primitive.normalize(payload)
    add = (
        arm in {"additions_only", "combined", "shuffled_feedback"}
        and training == label
        and template not in state["templates"]
        and len(state["templates"]) < 32
    )
    require(
        row["feedback_ordinal"] == ordinal
        and row["prediction_ordinal"] == pending["ordinal"]
        and row["feedback_label"] == training
        and row["independently_verified_label"] == label
        and row["released_prefix"] == prefix
        and row["template_added"] == add
        and row["template"] == template
        and row["coefficient_committed"]
        == (arm in {"calibration_only", "combined", "shuffled_feedback"}),
        "release_boundary",
    )
    if add:
        state["templates"].append(template)
    state["acks"].append(event)
    state["released"].append(label)
    return training != label


def state_chain(
    source: Json, group: list[Json], seals: dict[tuple[Any, ...], Json], head: Json
) -> Json:
    """Hash every complete serving state and link it to the preceding committed boundary.

    Tail coefficient updates have no intermediate snapshots. Their final complete
    identity and symbolic transitions are checked; numerical gradients are not
    re-executed and therefore cannot count as audit training or extra evidence.
    """
    first = group[0]
    key = (first["cluster_id"], first["seed"], first["arm"])
    arm = first["arm"]
    selected = lambda r: (r["cluster_id"], r["seed"], r["arm"]) == key
    releases = {r["prediction_ordinal"]: r for r in source["release_order_rows"] if selected(r)}
    costs = {r["unit_id"]: r for r in source["complete_learning_cost_rows"] if selected(r)}
    online = sorted((r for r in group if r["split"] == "online"), key=lambda r: r["ordinal"])
    retention = {
        (r["checkpoint"], r["ordinal"]): seals[identity(r)]
        for r in group
        if r["split"] == "retention"
    }
    state = dict(templates=[], parameters=head["parameters"], pending={}, acks=[], released=[])
    snapshots = {0: deepcopy(state)}
    shuffled = 0
    for i, row in enumerate(online):
        seal = seals[identity(row)]
        if i >= 9:
            if arm not in {"read_only", "additions_only"}:
                state["parameters"] = seal["pre_state"]["parameters"]
            require(canonical_hash(state) == releases[i - 9]["checkpoint_hash"], "commit_identity")
        require(seal["pre_state"] == state, "complete_state_chain")
        if i in (0, 16):
            snapshots[i] = deepcopy(state)
        state["pending"][row["event_id"]] = seal["sealed"]
        if i >= 8:
            shuffled += release(
                state, releases[i - 8], seals[identity(online[i - 8])]["payload"], arm
            )
            require(
                costs[online[i - 8]["unit_id"]]["checkpoint_hash"]
                == releases[i - 8]["checkpoint_hash"],
                "cost_commit",
            )
    if arm not in {"read_only", "additions_only"}:
        state["parameters"] = retention[(32, 0)]["parameters"]
    require(canonical_hash(state) == releases[23]["checkpoint_hash"], "retention_commit")
    snapshots[32] = deepcopy(state)
    for row in group:
        if row["split"] == "retention":
            seal = seals[identity(row)]
            expected = snapshots[row["checkpoint"]]
            require(
                seal["state_hash"] == canonical_hash(expected)
                and seal["parameters"] == expected["parameters"]
                and seal["templates"] == expected["templates"]
                and row["learning_state_unchanged"],
                "retention_state",
            )
    for i in range(24, 32):
        shuffled += release(state, releases[i], seals[identity(online[i])]["payload"], arm)
    final = load(next(r for r in source["final_state_refs"] if selected(r)))
    if arm not in {"read_only", "additions_only"}:
        state["parameters"] = final["parameters"]
    require(
        state == final and canonical_hash(final) == releases[31]["checkpoint_hash"], "final_state"
    )
    return dict(
        complete_state_identities=len(online) + len(retention) + 1, shuffled_label_changes=shuffled
    )


def reduce(source: Json, frozen: Json) -> Json:
    """Authenticate each planned slot and recompute losses without producer aggregation."""
    supplied = source["per_query_rows"] + source["retention_rows"]
    by_id = {identity(r): r for r in supplied}
    require(len(by_id) == len(supplied), "duplicate_identity")
    rows = census(frozen, "missing_row")
    require(set(by_id) <= {identity(r) for r in rows}, "unscheduled_identity")
    heads = {h["seed"]: h for h in source["heads"]}
    cost_rows = {identity(r): r for r in source["complete_learning_cost_rows"]}
    seals: dict[tuple[Any, ...], Json] = {}
    public = {
        s["stream_id"]: load(ref)
        for s, ref in zip(frozen["streams"], source["input_refs"], strict=True)
    }
    changed, snapshots, shuffled = 0, 0, 0
    groups: dict[tuple[Any, ...], list[Json]] = defaultdict(list)
    began = time.monotonic()
    for index, target in enumerate(rows):
        row = by_id.get(identity(target))
        if row is None or row["status"] != "completed":
            target["missing_reason"] = (
                "missing_row" if row is None else row.get("missing_reason", "producer_incomplete")
            )
            continue
        try:
            seal = load(row["seal_ref"])
        except FileNotFoundError:
            target["missing_reason"] = "missing_seal"
            continue
        payload = public[row["cluster_id"]][row["split"]][row["ordinal"]]
        require(seal["payload"] == payload, "public_input")
        truth = str(primitive.decide(payload)["decision"])
        require(
            row["truth"] == truth and row["sealed_ns"] < row["truth_recorded_ns"], "label_order"
        )
        if row["split"] == "online":
            state = seal["pre_state"]
            observation = primitive.observe(payload, state["templates"])
            require(
                seal["pre_state_hash"]
                == canonical_hash(state)
                == row["state_hash"]
                == row["pre_update_state_hash"],
                "state_identity",
            )
            require(seal["sealed"]["evidence_hash"] == canonical_hash(payload), "evidence_identity")
            require(all(row[k] == v for k, v in seal["sealed"].items()), "sealed_fields")
            prediction = seal["sealed"]["prediction"]
            observed_changed = observation["counts"] != primitive.observe(payload, [])["counts"]
            require(observed_changed == row["changed_later_predicates"], "changed_predicates")
            changed += int(row["arm"] == "combined" and row["ordinal"] >= 9 and observed_changed)
            costs = cost_rows[identity(row)]
            require(
                row["sealed_ns"] < costs["release_started_ns"] and costs["restart_equal"],
                "feedback_clock",
            )
        else:
            observation = primitive.observe(payload, seal["templates"])
            require(observation == seal["observation"], "retention_observation")
            prediction = seal["prediction"]
        scored = score(prediction, truth, heads[row["seed"]]["threshold"])
        require(
            scored["metric"] == row["metric"]
            and scored["action"] == row["action"]
            and prediction == row["prediction"]
            and row["paid_checks"] == 5,
            "scored_fields",
        )
        if row["split"] == "online":
            require(
                observation["counts"] == row["counts"]
                and observation["decision"] == row["decision"],
                "active_predicates",
            )
        target.update(
            scored, truth=truth, prediction=prediction, status="completed", missing_reason=None
        )
        seals[identity(row)] = seal
        groups[(row["cluster_id"], row["seed"], row["arm"])].append(row)
        if (index + 1) % 1000 == 0 or time.monotonic() - began >= 30:
            progress("authenticate_slots", index + 1, len(rows) - index - 1)
            began = time.monotonic()
    complete = all(r["status"] == "completed" for r in rows)
    if complete:
        for group in groups.values():
            checked = state_chain(source, group, seals, heads[group[0]["seed"]])
            snapshots += checked["complete_state_identities"]
            shuffled += checked["shuffled_label_changes"]
    checks = dict(
        complete_denominators=complete,
        prediction_before_label=True,
        complete_state_identities=complete,
        fixed_check_budget=True,
        actual_future_changed_check_use=changed > 0,
        valid_shuffled_control=shuffled > 0,
    )
    progress(
        "authenticated_slots",
        sum(r["status"] == "completed" for r in rows),
        sum(r["status"] != "completed" for r in rows),
    )
    return dict(
        rows=rows,
        leakage_checks=dict(
            checks=checks,
            complete_state_identities=snapshots,
            future_changed_checks=changed,
            shuffled_label_changes=shuffled,
            tail_state_scope="All full snapshots and final tail identity; intermediate tail coefficients are not refitted.",
        ),
    )


def summarize(rows: list[Json]) -> Json:
    """Average nested seeds before whole-stream resampling across the frozen family."""
    require(all(r["status"] == "completed" for r in rows), "incomplete_denominators")
    future = [r for r in rows if r["split"] == "online" and r["ordinal"] >= 9]
    streams = sorted({r["cluster_id"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})

    def mean(
        cluster: str, arm: str, metric: str, split: str = "online", step: int | None = None
    ) -> float:
        selected = future if split == "online" else rows
        return float(
            np.mean(
                [
                    np.mean(
                        [
                            r[metric]
                            for r in selected
                            if r["cluster_id"] == cluster
                            and r["seed"] == seed
                            and r["arm"] == arm
                            and r["split"] == split
                            and r["checkpoint"] == step
                        ]
                    )
                    for seed in seeds
                ]
            )
        )

    names = ["future_combined_minus_read_only", "future_combined_minus_calibration_only"]
    specs = [
        ("metric", "combined", arm, "online", None, None)
        for arm in ["read_only", "calibration_only"]
    ]
    for arm in ["additions_only", "calibration_only", "combined"]:
        for step in [16, 32]:
            names.append(f"retention_{arm}_step{step}_minus_step0")
            specs.append(("metric", arm, arm, "retention", step, 0))
    for metric in ["false_accept", "false_reject"]:
        names.append("error_combined_minus_read_only_" + metric)
        specs.append((metric, "combined", "read_only", "online", None, None))
    deltas = np.asarray(
        [
            [
                mean(c, arm, metric, split, step) - mean(c, comparator, metric, split, baseline)
                for metric, arm, comparator, split, step, baseline in specs
            ]
            for c in streams
        ]
    )
    rng = np.random.default_rng(8810)
    progress("bootstrap_before", 0, 10000)
    draws = np.empty((10000, 10))
    for offset in range(0, 10000, 1000):
        indices = rng.integers(0, len(streams), size=(1000, len(streams)))
        draws[offset : offset + 1000] = deltas[indices].mean(axis=1)
        progress("bootstrap_draws", offset + 1000, 9000 - offset)
    bounds = np.quantile(draws, [0.00025, 0.99975], axis=0)
    intervals = [
        dict(
            comparison=name,
            estimate=float(deltas[:, i].mean()),
            lower=float(bounds[0, i]),
            upper=float(bounds[1, i]),
            draws=10000,
            independent_count=len(streams),
            simultaneous_confidence=0.995,
            tail_probability=0.00025,
        )
        for i, name in enumerate(names)
    ]
    coverage = []
    for arm in primitive.ARMS:
        local = [r for r in future if r["arm"] == arm]
        coverage.append(
            dict(
                arm=arm,
                denominator=len(local),
                accepted=sum(r["action"] == "accept" for r in local),
                rejected=sum(r["action"] == "reject" for r in local),
                covered=sum(r["covered"] for r in local),
                coverage=sum(r["covered"] for r in local) / len(local),
                false_accepts=sum(r["false_accept"] for r in local),
                false_rejects=sum(r["false_reject"] for r in local),
                support_denominator=sum(r["truth"] == "support" for r in local),
                nonsupport_denominator=sum(r["truth"] != "support" for r in local),
                brier=float(np.mean([r["metric"] for r in local])),
            )
        )
    return dict(
        familywise_intervals=intervals,
        coverage_rows=coverage,
        stream_comparison_rows=[
            dict(cluster_id=c, differences=deltas[i].tolist()) for i, c in enumerate(streams)
        ],
        retention_checks=dict(comparisons=intervals[2:8], maximum_allowed_regression=0.02),
        headroom_checks=dict(
            exact_oracle_brier=0,
            exact_oracle_accuracy=1,
            scope="Exact symbolic labels define the ceiling; observed coverage is reported separately.",
        ),
    )


def benefit_gates(reduced: Json, summary: Json) -> Json:
    """Execution readiness alone cannot pass causal benefit or retention gates."""
    bounds = summary["familywise_intervals"]
    return dict(
        future_brier=all(r["upper"] < -0.01 for r in bounds[:2]),
        retention=all(r["upper"] <= 0.02 for r in bounds[2:8]),
        errors=all(r["upper"] <= 0.02 for r in bounds[8:]),
        **reduced["leakage_checks"]["checks"],
    )

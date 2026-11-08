"""REQ-REPORT-8933: freeze replay methods using development evidence only.

Eight paired streams test the causal implementation. Fresh evaluation recipes
remain unopened, so development improvements cannot grant a benefit claim.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from carnot.memory import compatible_replay_8933 as replay
from carnot.memory import signed_memory_8892 as signed
from carnot.reporting import guarded_update_methods_8905 as base
from carnot.reporting import topology_memory_methods_8881 as old
from carnot.reporting.causal_learning_audit_8810 import require
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v748_collection_contract import namespace

__all__ = ["signed", "old"]
Json = dict[str, Any]
ROOT, SEED = base.ROOT, 764000
NAME, TASK = "experiment_8933_v764_compatible_replay_methods", "exp8933-compatible-replay-methods"
CLI = "scripts/experiments/" + NAME + ".py"
ARMS = replay.ARMS
OWNED = [
    "python/carnot/memory/compatible_replay_8933.py",
    "python/carnot/reporting/compatible_replay_methods_8933.py",
    "python/carnot/reporting/compatible_replay_runner_8933.py",
    CLI,
]
TESTS = ["tests/python/test_compatible_replay_methods_8933.py"]
anchors = base.anchors


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush boundaries and completed counts so supervision can distinguish stalled work."""
    print(f"[exp8933] phase={phase} completed={completed} pending={pending}", flush=True)


def protocol() -> Json:
    """Freeze eight development streams and24 disjoint recipes without their labels."""
    p = base.protocol()
    runtime = namespace(old, "compatible8933_plan")
    runtime.SEED = SEED
    for key, indices, pilot in (
        ("pilot_manifest", range(100, 108), True),
        ("frozen_evaluation_manifest", range(200, 224), False),
    ):
        plans = []
        for index in indices:
            plan = runtime.plan(index, pilot=pilot)
            plan.update(
                initial_head_seed=plan["seed"],
                seed=SEED + index,
                stream_id=f"v764-{'development' if pilot else 'evaluation'}-{index}",
                delays=[8] * 48,
            )
            for role in ("online", "retention"):
                for i, unit in enumerate(plan[role]):
                    unit["unit_id"] = f"{plan['stream_id']}/{role}/{i}"
            plans.append(plan)
        p[key] = plans
    bounds = [
        dict(
            split="online",
            endpoint="brier",
            ordinals=[25, 48],
            against=arm,
            upper=-0.01 if arm == "frozen_verified" else 0,
            strict=arm != "frozen_verified",
        )
        for arm in ARMS
        if arm != "compatible_replay"
    ]
    bounds += [
        dict(
            split="retention",
            checkpoint=c,
            endpoint=e,
            against="frozen_verified",
            upper=0.01,
            strict=False,
        )
        for c in (0, 24, 48)
        for e in ("brier", "error")
    ]
    p.update(
        schema=replay.SCHEMA,
        arms=list(ARMS),
        expected_row_counts=dict(development=15360, evaluation=46080),
        gradients_per_train_release=2,
        replay_rule="minimum squared signed.vector distance to latest visible query; stable unit id ties; prior TRAIN buffer; empty repeats current",
        permutation="fixed within released TRAIN block; guard and template truth unchanged",
        optimizer_equations=[
            "p=softmax(W*x)",
            "W1=W-0.1*grad(-log(p[y_released]))",
            "W2=W1-0.1*grad(-log(softmax(W1*x_replay)[y_replay]))",
        ],
        evaluation_plan=dict(
            candidate_arm="compatible_replay",
            family_intervals=10,
            family_alpha=0.01,
            alpha_per_bound=0.001,
            bootstrap_resamples=10000,
            bootstrap_unit="whole paired stream",
            seed=764200,
            bounds=bounds,
        ),
        initial_states="unchanged authenticated signed development heads; paired per stream",
        selector_precedent="Exp5857 visible signatures and bounded prior replay; its oracle structural scores are not signed-head Brier evidence",
    )
    return p


def run_stream(raw: Path, plan: Json, params: Json, *, restart: bool = True) -> Json:
    """Reuse prediction, delayed block release, retention and restart orchestration."""
    runtime = namespace(base, "compatible8933_stream")

    def store(path: Path, head: Json, arm: str, anchor: list[Json]) -> replay.Store:
        return replay.Store(path, head, arm, anchor, seed=plan["seed"])

    runtime.g = SimpleNamespace(Store=store, ARMS=ARMS)
    runtime.progress, runtime.anchors = progress, anchors
    result: Json = runtime.run_stream(raw, plan, params, restart=restart)
    states = result["states"]
    result["control_rows"] = [
        r for r in result["control_rows"] if r["check"] != "matched_training_work"
    ]
    trainable = [states[a] for a in ARMS[1:]]
    result["control_rows"] += [
        dict(
            check="compatible_selection_parity",
            passed=[r["selected_id"] for r in states["compatible_replay"]["selection_rows"]]
            == [r["selected_id"] for r in states["compatible_permuted"]["selection_rows"]],
        ),
        dict(
            check="two_gradient_budget",
            passed=len({s["training_calls"] for s in trainable}) == 1
            and all(
                s["training_calls"]
                == 2 * sum(r["split"] == "train" for r in s["released"])
                == len(s["gradient_ledger"])
                for s in trainable
            )
            and states[ARMS[0]]["training_calls"] == 0,
        ),
        dict(check="unchanged_frozen_head", passed=states[ARMS[0]]["committed"] == params),
        dict(
            check="guard_truth_unchanged",
            passed=all(
                r["label"] == r["true_label"]
                for s in states.values()
                for r in s["released"]
                if r["split"] == "guard"
            ),
        ),
        dict(
            check="equal_true_feedback",
            passed=all(
                [(r["group_id"], r["true_label"]) for r in s["released"]]
                == [(r["group_id"], r["true_label"]) for r in states[ARMS[0]]["released"]]
                for s in states.values()
            ),
        ),
    ]
    rows = []
    for r in result["rows"]:
        for metric, value in (("brier", r["metric"]), ("error", r["error"])):
            rows.append(dict(r, metric=metric, value=value, numerator=value))
    result["rows"] = rows
    return result


def census(reason: str) -> Json:
    """Keep every scheduled development cell when external evidence is unavailable."""
    rows = []
    for plan in protocol()["pilot_manifest"]:
        for role, checkpoint in [("online", None)] + [("retention", c) for c in (0, 24, 48)]:
            for i, unit in enumerate(plan[role]):
                for arm in ARMS:
                    for metric in ("brier", "error"):
                        rows.append(
                            dict(
                                unit_id=unit["unit_id"]
                                + (f"/checkpoint{checkpoint}" if checkpoint is not None else ""),
                                cluster_id=plan["stream_id"],
                                seed=plan["seed"],
                                arm=arm,
                                split=role,
                                checkpoint=checkpoint,
                                ordinal=i + 1,
                                metric=metric,
                                value=None,
                                numerator=None,
                                denominator=1,
                                status="censored",
                                missing_reason=reason,
                            )
                        )
    return dict(
        rows=rows,
        control_rows=[],
        promotion_rows=[],
        numerical_training_calls=0,
        replay_selection_rows=[],
        gradient_ledger=[],
        buffer_state_hashes=[],
        development_rows=[],
    )


def development_summary(rows: list[Json]) -> list[Json]:
    """Recompute descriptive paired deltas from original per-metric rows only."""
    summaries = []
    for bound in protocol()["evaluation_plan"]["bounds"]:
        deltas = []
        for cluster in sorted({r["cluster_id"] for r in rows}):
            means = {}
            for arm in ("compatible_replay", bound["against"]):
                selected = [
                    r["value"]
                    for r in rows
                    if r["cluster_id"] == cluster
                    and r["arm"] == arm
                    and r["metric"] == bound["endpoint"]
                    and r["split"] == bound["split"]
                    and r["checkpoint"] == bound.get("checkpoint")
                    and (bound["split"] != "online" or r["ordinal"] >= 25)
                ]
                means[arm] = sum(selected) / len(selected)
            deltas.append(means["compatible_replay"] - means[bound["against"]])
        summaries.append(
            dict(
                bound,
                paired_stream_deltas=deltas,
                mean=sum(deltas) / len(deltas),
                scope="development descriptive only; no readiness or benefit gate",
            )
        )
    return summaries


def recover(raw: Path, reason: str) -> Json:
    """Retain all completed seals and charge durable updates after an owned failure."""
    result = census(reason)
    completed = {}
    failed = set()
    for plan in protocol()["pilot_manifest"]:
        attempt = raw / ("attempt-" + plan["stream_id"] + ".json")
        if attempt.exists() and json.loads(attempt.read_bytes())["status"] == "failed":
            failed.add(plan["stream_id"])
        path = raw / (plan["stream_id"] + ".json")
        if path.exists():
            stream = json.loads(path.read_bytes())
            result["control_rows"] += stream["control_rows"]
            completed.update({(r["unit_id"], r["arm"], r["metric"]): r for r in stream["rows"]})
        else:
            partial = raw / "states" / plan["stream_id"] / "rows.json"
            if partial.exists():
                for row in json.loads(partial.read_bytes())["rows"]:
                    for metric, value in (("brier", row["metric"]), ("error", row["error"])):
                        r = dict(row, metric=metric, value=value, numerator=value)
                        completed[r["unit_id"], r["arm"], metric] = r
        for arm in ARMS:
            state_path = raw / "states" / plan["stream_id"] / (arm + ".json")
            if not state_path.exists():
                continue
            state = json.loads(state_path.read_bytes())
            result["numerical_training_calls"] += state["training_calls"]
            for target, source in (
                ("promotion_rows", "promotions"),
                ("replay_selection_rows", "selection_rows"),
                ("gradient_ledger", "gradient_ledger"),
            ):
                result[target] += [
                    dict(r, arm=arm, cluster_id=plan["stream_id"], seed=plan["seed"])
                    for r in state[source]
                ]
            result["buffer_state_hashes"].append(
                dict(
                    arm=arm,
                    cluster_id=plan["stream_id"],
                    buffer_hash=canonical_hash(state["buffer"]),
                    rng_hash=canonical_hash(state["rng_state"]),
                    state_hash=canonical_hash(state),
                )
            )
    result["rows"] = [
        completed.get(
            (r["unit_id"], r["arm"], r["metric"]),
            dict(r, status="failed") if r["cluster_id"] in failed else r,
        )
        for r in result["rows"]
    ]
    if all(r["status"] == "complete" for r in result["rows"]):
        result["development_rows"] = development_summary(result["rows"])
    return result


def measure(raw: Path, heads: Json) -> Json:
    """Run eight paired development streams, then leave every evaluation world unopened."""
    atomic_json(raw / "protocol.json", protocol())
    atomic_json(raw / "heads.json", heads)
    for i, plan in enumerate(protocol()["pilot_manifest"]):
        progress("development_before", i, 8 - i)
        params = heads[str(plan["initial_head_seed"])]
        attempt = raw / ("attempt-" + plan["stream_id"] + ".json")
        atomic_json(attempt, dict(status="attempted"))
        try:
            result = run_stream(raw / "states" / plan["stream_id"], plan, params)
        except (OSError, ValueError, RuntimeError, KeyError):
            atomic_json(attempt, dict(status="failed"))
            raise
        result["control_rows"] += replay.selector_controls(params)
        atomic_json(raw / (plan["stream_id"] + ".json"), result)
        atomic_json(attempt, dict(status="complete"))
        require(all(r["passed"] for r in result["control_rows"]), "owned_control_failure")
        progress("development_after", i + 1, 7 - i)
    return recover(raw, "unfinished")

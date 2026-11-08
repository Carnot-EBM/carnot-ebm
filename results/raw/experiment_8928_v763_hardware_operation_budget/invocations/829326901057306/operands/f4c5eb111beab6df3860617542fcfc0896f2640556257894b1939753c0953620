"""REQ-REPORT-8905: qualify a fixed development-only update-admission rule.

Pilot outcomes cannot change thresholds or open the fresh evaluation manifest.
The constructed verifier supplies oracle labels, not independent generalization.
"""

from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any

from carnot.memory import guarded_memory_8905 as g
from carnot.memory import signed_memory_8892 as signed
from carnot.memory import causal_memory_8802 as primitive
from carnot.reporting import evidence_set_methods_8768 as symbolic
from carnot.reporting import topology_memory_methods_8881 as old
from carnot.reporting.causal_learning_audit_8810 import require, score
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v748_collection_contract import namespace

__all__ = ["g", "signed", "old"]

Json = dict[str, Any]
ROOT = old.ROOT
NAME, TASK = "experiment_8905_v762_guarded_update_methods", "exp8905-guarded-update-methods"
CLI = "scripts/experiments/" + NAME + ".py"
OWNED = [
    "python/carnot/memory/guarded_memory_8905.py",
    "python/carnot/reporting/guarded_update_methods_8905.py",
    "python/carnot/reporting/guarded_update_runner_8905.py",
    CLI,
]
TESTS = ["tests/python/test_guarded_update_methods_8905.py"]
SEED = g.SEED


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real counts without extending the fixed execution deadline."""
    print(f"[exp8905] phase={phase} completed={completed} pending={pending}", flush=True)


def protocol() -> Json:
    """Freeze all recipes and thresholds without generating fresh evaluation worlds."""
    runtime = namespace(old, "guarded8905_protocol")
    runtime.SEED = SEED

    def plans(indices: range, pilot: bool) -> list[Json]:
        rows = []
        for index in indices:
            plan = runtime.plan(index, pilot=pilot)
            plan["stream_id"] = f"v762-{'development' if pilot else 'evaluation'}-{index}"
            for role in ("online", "retention"):
                for i, row in enumerate(plan[role]):
                    row["unit_id"] = f"{plan['stream_id']}/{role}/{i}"
            rows.append(plan)
        return rows

    return dict(
        schema=g.SCHEMA,
        arms=list(g.ARMS),
        pilot_manifest=plans(range(100, 108), True),
        frozen_evaluation_manifest=plans(range(200, 224), False),
        expected_row_counts=dict(development=6144, evaluation=18432),
        feedback_blocks=[list(range(i + 1, i + 9)) for i in range(0, 48, 8)],
        split_rule="int(SHA256(group_id),16)%4 == 0: guard; otherwise train",
        training_window=32,
        guard_window=8,
        minimum_guard_groups=4,
        fit_feedback_counts=list(g.SCHEDULE),
        anchor_groups=16,
        recent_brier_improvement=0.001,
        anchor_brier_degradation=0.005,
        anchor_error_increase=0,
        development_future_brier_max=-0.01,
        retention_brier_error_max=0.01,
        retention_checkpoints=[0, 24, 48],
        objective="linear categorical log loss",
        optimizer="SGD",
        learning_rate=0.1,
        feature_schema=signed.SCHEMA,
        topology_manifest=old.graphs(),
        evaluation_opened=False,
        heuristic_scope="Repeated operational guards are not an independent risk certificate.",
    )


def anchors() -> list[Json]:
    """Fix sixteen distinct development groups before measuring any pilot outcome."""
    rows = []
    for i in range(16):
        group = f"v762-anchor/{i}"
        value = old.world(old.graphs()["development"][i % 8], group, old.ERRORS[i % 6], SEED + i)
        rows.append(
            dict(
                group_id=group,
                counts=signed.observe(value, [old.topology(value["query"])])["counts"],
                label=str(symbolic.decide(value)["decision"]),
                evidence_hash=canonical_hash(value),
            )
        )
    return rows


def scored(
    plan: Json, arm: str, unit: Json, sealed: Json, truth: str, role: str, checkpoint: int | None
) -> Json:
    """Every headline keeps its original query and paired-stream denominator."""
    return dict(
        unit_id=unit["unit_id"] + (f"/checkpoint{checkpoint}" if checkpoint is not None else ""),
        cluster_id=plan["stream_id"],
        seed=plan["seed"],
        arm=arm,
        split=role,
        checkpoint=checkpoint,
        ordinal=sealed["ordinal"],
        prediction=sealed["prediction"],
        truth=truth,
        denominator=1,
        status="complete",
        missing_reason=None,
        **score(sealed["prediction"], truth, 0.5),
        error=int(primitive.LABELS[max(range(3), key=sealed["prediction"].__getitem__)] != truth),
    )


def run_stream(raw: Path, plan: Json, params: Json, *, restart: bool = True) -> Json:
    """Keep retention reads outside the store and release complete due blocks in order."""
    anchor = anchors()
    stores = {a: g.Store(raw / (a + ".json"), params, a, anchor) for a in g.ARMS}
    values = [
        old.world(
            old.graphs()["development"][u["template"]],
            u["unit_id"] + "/",
            u["error_type"],
            u["world_seed"],
        )
        for u in plan["online"]
    ]
    rows: list[Json] = []
    due = [i + 1 + delay for i, delay in enumerate(plan["delays"])]
    released = 0
    restart_checks = 0

    def retention(checkpoint: int) -> None:
        for i, unit in enumerate(plan["retention"]):
            value = old.world(
                old.graphs()["development"][unit["template"]],
                unit["unit_id"] + "/",
                unit["error_type"],
                unit["world_seed"],
            )
            for arm, store in stores.items():
                observation = signed.observe(value, store.state["templates"])
                prediction = signed.probabilities(store.state["committed"], observation["counts"])
                rows.append(
                    scored(
                        plan,
                        arm,
                        unit,
                        dict(ordinal=i + 1, prediction=prediction),
                        str(symbolic.decide(value)["decision"]),
                        "retention",
                        checkpoint,
                    )
                )
        atomic_json(raw / "rows.json", dict(rows=rows))

    retention(0)
    for tick in range(1, max(due) + 1):
        if tick <= 48:
            for store in stores.values():
                store.predict(
                    plan["online"][tick - 1]["unit_id"], values[tick - 1], tick, due[tick - 1]
                )
        while released < 48 and tick >= max(due[released : released + 8]):
            batch = [
                (plan["online"][i]["unit_id"], values[i]) for i in range(released, released + 8)
            ]
            for arm, store in stores.items():
                store.release(batch, tick)
                for record, unit in zip(
                    store.state["released"][-8:],
                    plan["online"][released : released + 8],
                    strict=True,
                ):
                    rows.append(
                        scored(plan, arm, unit, record, record["true_label"], "online", None)
                    )
            released += 8
            atomic_json(raw / "rows.json", dict(rows=rows))
        if tick == 24:
            retention(24)
        if restart:
            for arm, store in list(stores.items()):
                resumed = g.Store(store.path, {}, arm, anchor)
                require(resumed.state == store.state, "restart_drift")
                stores[arm] = resumed
                restart_checks += 1
        if tick % 16 == 0:
            progress("pilot_" + plan["stream_id"], tick, max(due) - tick)
    retention(48)
    states = {arm: s.state for arm, s in stores.items()}
    controls = [
        dict(check="durable_restart", passed=not restart or restart_checks > 0),
        dict(
            check="matched_training_work",
            passed=len({s["training_calls"] for s in states.values()}) == 1,
        ),
        dict(
            check="matched_guard_work",
            passed=len({s["guard_scores"] for s in states.values()}) == 1,
        ),
        dict(
            check="true_template_admission",
            passed=all(s["templates"] == states[g.ARMS[0]]["templates"] for s in states.values()),
        ),
        dict(
            check="guard_train_disjoint",
            passed=all(
                not set(r["training_groups"]) & set(r["guard_groups"])
                for s in states.values()
                for r in s["promotions"]
            ),
        ),
        dict(
            check="release_complete",
            passed=all(s["cursor"] == 48 and not s["pending"] for s in states.values()),
        ),
    ]
    return dict(
        rows=rows,
        control_rows=controls,
        states=states,
        promotion_rows=[
            dict(r, arm=a, cluster_id=plan["stream_id"], seed=plan["seed"])
            for a, s in states.items()
            for r in s["promotions"]
        ],
        numerical_training_calls=sum(s["training_calls"] for s in states.values()),
    )


def feasibility(rows: list[Json], promotions: list[Json]) -> Json:
    """Apply the predeclared development conjunction without selecting new thresholds."""
    reductions = []
    for role, checkpoint, endpoint in [("online", None, "metric")] + [
        ("retention", c, e) for c in (0, 24, 48) for e in ("metric", "error")
    ]:
        paired = []
        for cluster in sorted({r["cluster_id"] for r in rows}):
            means = {}
            for arm in ("guarded_true", "frozen_verified"):
                selected = [
                    r
                    for r in rows
                    if r["cluster_id"] == cluster
                    and r["arm"] == arm
                    and r["split"] == role
                    and r["checkpoint"] == checkpoint
                    and (role != "online" or r["ordinal"] >= 25)
                ]
                means[arm] = sum(r[endpoint] for r in selected) / len(selected)
            paired.append(means["guarded_true"] - means["frozen_verified"])
        mean = sum(paired) / len(paired)
        threshold = -0.01 if role == "online" else 0.01
        reductions.append(
            dict(
                split=role,
                checkpoint=checkpoint,
                endpoint=endpoint,
                paired_stream_deltas=paired,
                mean=mean,
                threshold=threshold,
                passed=mean <= threshold,
            )
        )
    promoted = any(r["promoted"] and r["arm"] == "guarded_true" for r in promotions)
    return dict(
        development_rows=reductions,
        candidate_ready_score=int(promoted and all(r["passed"] for r in reductions)),
    )


def measure(raw: Path, heads: Json) -> Json:
    """Run exactly eight frozen pilots and retain every completed stream privately."""
    p = protocol()
    atomic_json(raw / "protocol.json", p)
    atomic_json(raw / "anchors.json", dict(rows=anchors()))
    began = time.monotonic()
    for index, plan in enumerate(p["pilot_manifest"]):
        require(time.monotonic() - began < 1800, "pilot_deadline")
        progress("pilot_before", index, 8 - index)
        params = heads if "weights" in heads else heads[str(plan["seed"])]
        attempt = raw / ("attempt-" + plan["stream_id"] + ".json")
        atomic_json(attempt, dict(status="attempted"))
        try:
            result = run_stream(raw / "states" / plan["stream_id"], plan, params)
            atomic_json(raw / (plan["stream_id"] + ".json"), result)
        except (OSError, ValueError, RuntimeError, KeyError):
            atomic_json(attempt, dict(status="failed"))
            raise
        atomic_json(attempt, dict(status="complete"))
        require(all(r["passed"] for r in result["control_rows"]), "owned_control_failure")
        progress("pilot_after", index + 1, 7 - index)
    results = [
        json.loads((raw / (s["stream_id"] + ".json")).read_bytes()) for s in p["pilot_manifest"]
    ]
    rows = [r for s in results for r in s["rows"]]
    promotions = [r for s in results for r in s["promotion_rows"]]
    return dict(
        rows=rows,
        promotion_rows=promotions,
        control_rows=[r for s in results for r in s["control_rows"]],
        numerical_training_calls=sum(s["numerical_training_calls"] for s in results),
        **feasibility(rows, promotions),
    )

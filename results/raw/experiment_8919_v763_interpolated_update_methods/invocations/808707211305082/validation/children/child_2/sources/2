"""REQ-REPORT-8919: freeze one candidate rule using the existing causal pilot runner.

Only development worlds are constructed here. Evaluation entries are recipes,
and their retention labels never reach the store or the promotion guard.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from carnot.memory import guarded_memory_8905 as g
from carnot.memory import interpolated_memory_8919 as opt
from carnot.memory import signed_memory_8892 as signed
from carnot.reporting import guarded_update_methods_8905 as base
from carnot.reporting import guarded_benefit_audit_8911 as audit
from carnot.reporting import topology_memory_methods_8881 as old
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v748_collection_contract import namespace

__all__ = ["g", "signed", "old", "opt"]
Json = dict[str, Any]
ROOT, SEED = base.ROOT, 763100
NAME, TASK = (
    "experiment_8919_v763_interpolated_update_methods",
    "exp8919-interpolated-update-methods",
)
CLI = "scripts/experiments/" + NAME + ".py"
ARMS = ("frozen_verified", "guarded_true", "interpolated_true", "interpolated_permuted")
OWNED = [
    "python/carnot/memory/interpolated_memory_8919.py",
    "python/carnot/memory/guarded_memory_8905.py",
    "python/carnot/reporting/interpolated_update_methods_8919.py",
    "python/carnot/reporting/interpolated_update_runner_8919.py",
    CLI,
]
TESTS = [
    "tests/python/test_interpolated_update_methods_8919.py",
    "tests/python/test_guarded_update_methods_8905.py",
]
anchors = base.anchors


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush phase boundaries and bounded loop counts so supervision sees real work."""
    print(f"[exp8919] phase={phase} completed={completed} pending={pending}", flush=True)


def protocol() -> Json:
    """Reuse the frozen guard settings while sealing disjoint stream identities and seeds."""
    p = base.protocol()
    runtime = namespace(old, "interpolated8919_plan")
    runtime.SEED = 763000
    for key, indices, pilot in (
        ("pilot_manifest", range(100, 108), True),
        ("frozen_evaluation_manifest", range(200, 224), False),
    ):
        plans = []
        for index in indices:
            plan = runtime.plan(index, pilot=pilot)
            plan.update(
                initial_head_seed=plan["seed"],
                seed=763000 + index,
                stream_id=f"v763-{'development' if pilot else 'evaluation'}-{index}",
                delays=[8] * 48,
            )
            for role in ("online", "retention"):
                for i, unit in enumerate(plan[role]):
                    unit["unit_id"] = f"{plan['stream_id']}/{role}/{i}"
            plans.append(plan)
        p[key] = plans
    evaluation = dict(
        audit.GATES,
        seed=763200,
        future_contrasts=["frozen_verified", "interpolated_permuted", "guarded_true"],
        candidate_arm="interpolated_true",
    )
    evaluation["future_guarded_upper_strict"] = evaluation.pop("future_ordinary_upper_strict")
    evaluation["bounds"] = [
        dict(
            split="online",
            endpoint="metric",
            ordinals=[25, 48],
            against=arm,
            upper=-0.01 if arm == "frozen_verified" else 0,
            strict=arm != "frozen_verified",
        )
        for arm in ("frozen_verified", "interpolated_permuted", "guarded_true")
    ] + [
        dict(
            split="retention",
            checkpoint=c,
            endpoint=e,
            against="frozen_verified",
            upper=0.01,
            strict=False,
        )
        for c in (0, 24, 48)
        for e in ("metric", "error")
    ]
    p.update(
        schema=opt.SCHEMA,
        arms=list(ARMS),
        optimizer="SPIN-style fixed interpolation",
        optimizer_constants=dict(alpha=opt.ALPHA, beta=opt.BETA, eta=opt.ETA),
        optimizer_equations=[
            "x=(1-beta)*fast+beta*slow",
            "fast_next=fast-eta*grad(loss(x))",
            "slow_next=slow+alpha*(fast_next-slow)",
        ],
        evaluation_plan=evaluation,
        permutation="released training labels within release block; guard truth unchanged",
        state_sequences="continuous across scheduled fits; slow submitted to unchanged guard",
    )
    return p


def feasibility(rows: list[Json], promotions: list[Json]) -> Json:
    """Reduce development streams with the original thresholds and the new candidate arm."""
    mapped = [
        dict(r, arm="guarded_true" if r["arm"] == "interpolated_true" else r["arm"])
        for r in rows
        if r["arm"] != "guarded_true"
    ]
    mapped_promotions = [
        dict(r, arm="guarded_true") for r in promotions if r["arm"] == "interpolated_true"
    ]
    return dict(base.feasibility(mapped, mapped_promotions))


def run_stream(raw: Path, plan: Json, params: Json, *, restart: bool = True) -> Json:
    """Keep query work, releases, templates and guard scoring in the original runner."""
    runtime = namespace(base, "interpolated8919_stream")

    def store(path: Path, head: Json, arm: str, anchor: list[Json]) -> g.Store:
        return g.Store(
            path,
            head,
            arm,
            anchor,
            optimizer=opt.SCHEMA if arm.startswith("interpolated_") else "sgd",
        )

    runtime.g = SimpleNamespace(Store=store, ARMS=ARMS)
    runtime.progress = progress
    result: Json = runtime.run_stream(raw, plan, params, restart=restart)
    states = result["states"]
    true, interpolated = states["guarded_true"], states["interpolated_true"]
    result["control_rows"] += [
        dict(
            check="equal_causal_information",
            passed=[
                [r[k] for k in ("group_id", "split", "label", "true_label", "counts")]
                for r in true["released"]
            ]
            == [
                [r[k] for k in ("group_id", "split", "label", "true_label", "counts")]
                for r in interpolated["released"]
            ],
        ),
        dict(
            check="equal_gradient_budget",
            passed=true["training_calls"]
            == interpolated["training_calls"]
            == interpolated["optimizer"]["update_count"],
        ),
        dict(
            check="training_only_permutation",
            passed=all(
                r["label"] == r["true_label"]
                for r in states["interpolated_permuted"]["released"]
                if r["split"] == "guard"
            ),
        ),
        dict(
            check="finite_changed_candidate",
            passed=all(
                canonical_hash(states[a]["optimizer"]["slow"]) != canonical_hash(params)
                and states[a]["optimizer"]["update_count"] > 0
                for a in ARMS[2:]
            ),
        ),
    ]
    return result


def measure(raw: Path, heads: Json) -> Json:
    """Run eight paired pilots through the shared loop; never construct evaluation worlds."""
    runtime = namespace(base, "interpolated8919_measure")
    runtime.protocol, runtime.run_stream = protocol, run_stream
    runtime.progress, runtime.feasibility = progress, feasibility
    if "weights" not in heads:
        heads = {
            str(p["seed"]): heads[str(p["initial_head_seed"])] for p in protocol()["pilot_manifest"]
        }
    result: Json = runtime.measure(raw, heads)
    result["gradient_call_ledger"] = [
        dict(
            cluster_id=p["stream_id"],
            arm=a,
            calls=state["training_calls"],
            guard_scores=state["guard_scores"],
            update_count=state.get("optimizer", {}).get("update_count", 0),
            promotions=state["promotions"],
        )
        for p in protocol()["pilot_manifest"]
        for a, state in json.loads((raw / (p["stream_id"] + ".json")).read_bytes())[
            "states"
        ].items()
    ]
    atomic_json(raw / "gradient_ledger.json", dict(rows=result["gradient_call_ledger"]))
    return result

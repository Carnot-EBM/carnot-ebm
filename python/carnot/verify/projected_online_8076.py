"""REQ-REPORT-8076: causal constraint learning on cached development sources.

Known released labels define finite inequalities. Their satisfaction is an
empirical check and grants no guarantee for future labels or environments.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from carnot import experiment_8072_v699_sealed_methods as methods
from carnot.experiment_8058_v698_sealed_evidence_methods import partition
from carnot.experiment_8021_v695_typed_decision_test import action, loss
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.evidence_features_custody_7980 import checked
from carnot.verify import constraint_projection_8075 as kernel
from carnot.verify import fresh_feedback_8064 as old
from carnot.verify import causal_online_8025 as basis

Json = dict[str, Any]
Array = NDArray[np.float64]
CONFIG = methods.methods()["learning"]
ARMS = CONFIG["arms"]
GUARDED = ("ray_fresh", "projected_fresh")
START = time.monotonic()
FIELDS = dict(
    issue="issued_prediction_rows",
    release="feedback_release_rows",
    gradient="gradient_rows",
    addition="constraint_addition_rows",
    eviction="constraint_eviction_rows",
    projection="projection_rows",
    fallback="fallback_rows",
    raw="raw_candidate_rows",
    candidate="candidate_commit_rows",
    consume="admission_consumption_rows",
    alpha="alpha_check_rows",
    commit="durable_commit_rows",
    pending="pending_update_rows",
    seal="final_head_seals",
    budget="per_arm_gradient_label_operation_budgets",
    block="admission_block_rows",
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Actual counts and elapsed time keep long work visible without padding."""
    print(
        f"[exp8076] {phase} elapsed_s={time.monotonic() - START:.3f} completed={completed} pending={pending}",
        flush=True,
    )


def feasible(theta: Array, initial: Array, rows: list[Json]) -> float:
    """Check the entire active memory and box rather than only sampled rows."""
    point, w0 = np.append(theta, 1.0), np.append(initial, 1.0)
    matrix = np.asarray([r["normal"] for r in rows]).reshape(len(rows), len(point))
    low, high = w0 - 0.5, w0 + 0.5
    low[-1] = high[-1] = 1.0
    return float(
        np.max(kernel.residuals(point, matrix, np.asarray([r["rhs"] for r in rows]), low, high))
    )


def admit(
    head: Json,
    incumbent: Array,
    endpoint: Array,
    initial: Array,
    x: Array,
    y: Array,
    rows: list[Json],
    projected: bool,
) -> Json:
    """Keep the qualified empirical guard, adding only full-memory feasibility."""
    checks, _ = old.guard(head, incumbent, endpoint, initial, x, y)
    for row in checks:
        point = incumbent + row["alpha"] * (endpoint - incumbent)
        row["max_residual"] = feasible(point, initial, rows) if projected else None
        if projected and row["max_residual"] > kernel.TOL:
            row["reasons"].append("current_memory")
            row["passed"] = False
    passing = [r["alpha"] for r in checks if r["passed"]]
    alpha = float(passing[0]) if passing else None
    fallback = "none" if alpha is not None else "incumbent"
    point = incumbent + alpha * (endpoint - incumbent) if alpha is not None else incumbent.copy()
    if alpha is None and projected and feasible(point, initial, rows) > kernel.TOL:
        point, fallback = initial.copy(), "initial"
    return dict(
        checks=checks,
        alpha=alpha,
        parameters=point.tolist(),
        fallback=fallback,
        all_rejected=bool(checks) and not passing,
        changed=bool(np.any(point != incumbent)),
    )


def trajectory(data: Json, seed: int, emit: Any, release: Any, costs: list[Json]) -> None:
    """Advance original slots; all label access occurs through delayed release."""
    sources, head = data["sources"], data["head"]
    initial = np.asarray(head["parameters"], dtype=float)
    x = np.asarray(
        [basis.design(head, r) if r["public_eligible"] else np.zeros(len(initial)) for r in sources]
    )
    phi, w0 = kernel.calibrated(x, initial, head["calibration"])
    states = {arm: initial.copy() for arm in ARMS}
    memories = {arm: dict(initial=w0.tolist(), events=[], rows=[]) for arm in GUARDED}
    released: dict[int, Any] = {}
    updates: list[int] = []
    budgets = {arm: dict(gradients=0, label_operations=0) for arm in ARMS}
    pending: Json = {}
    intercept, slope = head["calibration"]

    def write(kind: str, row: Json) -> None:
        emit(kind, deepcopy(dict(seed=seed, **row)))

    def labels(ids: list[int]) -> Array:
        return np.asarray([released[i] for i in ids], dtype=float)

    for slot, source in enumerate(sources):
        began = time.process_time_ns()
        for arm in ARMS:
            p = (
                float(expit(intercept + slope * float(x[slot] @ states[arm])))
                if source["public_eligible"]
                else None
            )
            write(
                "issue",
                dict(
                    slot=slot,
                    source=source["source_cluster_id"],
                    family_id=source["family_id"],
                    arm=arm,
                    probability=p,
                    action=action(p),
                    head_hash=canonical_hash(states[arm].tolist()),
                ),
            )
        costs.append(
            dict(
                seed=seed,
                slot=slot,
                component="query_and_issue_storage",
                cpu_ns=time.process_time_ns() - began,
                units=4,
            )
        )
        if slot in CONFIG["attempt_slots"]:
            if pending:
                write(
                    "pending",
                    dict(
                        slot=slot,
                        arm="shared",
                        status="censored",
                        reason="next_attempt",
                        candidate_slot=pending["slot"],
                        consumed_slots=pending["observed"],
                    ),
                )
                pending = {}
            ids = updates[-32:]
            if not old.adequate(labels(ids), 16):
                write(
                    "pending",
                    dict(slot=slot, arm="shared", status="deferred", reason="update_support"),
                )
            else:
                pending = dict(slot=slot, observed=[], candidates={}, update_slots=ids)
                for arm in ARMS[1:]:
                    began = time.process_time_ns()
                    theta = states[arm].copy()
                    for step in range(4):
                        before = theta.copy()
                        gradient = (
                            slope
                            * (
                                x[ids].T
                                @ (expit(intercept + slope * (x[ids] @ theta)) - labels(ids))
                            )
                            / len(ids)
                            + 0.002 * theta
                        )
                        theta -= 0.01 * gradient
                        write(
                            "gradient",
                            dict(
                                slot=slot,
                                arm=arm,
                                step=step,
                                update_slots=ids,
                                before=before.tolist(),
                                gradient=gradient.tolist(),
                                after=theta.tolist(),
                            ),
                        )
                    budgets[arm]["gradients"] += 4
                    budgets[arm]["label_operations"] += 4 * len(ids)
                    costs.append(
                        dict(
                            seed=seed,
                            slot=slot,
                            arm=arm,
                            component="gradient_and_storage",
                            cpu_ns=time.process_time_ns() - began,
                            units=4 * len(ids),
                        )
                    )
                    write(
                        "raw",
                        dict(
                            slot=slot,
                            arm=arm,
                            parameters=theta.tolist(),
                            head_hash=canonical_hash(theta.tolist()),
                        ),
                    )
                    if arm == "projected_fresh":
                        projection_began = time.process_time_ns()
                        projection = kernel.project(
                            np.append(theta, 1.0),
                            memories[arm]["rows"],
                            w0,
                            np.append(states[arm], 1.0),
                            seed=seed + slot,
                            frozen_last=True,
                        )
                        cost = projection["cost"]
                        costs.append(
                            dict(
                                seed=seed,
                                slot=slot,
                                arm=arm,
                                component="projection",
                                cpu_ns=time.process_time_ns() - projection_began,
                                wall_ns=cost["total_ns"],
                                fallback_ns=cost["fallback_ns"],
                                units=cost["row_dot_products"],
                            )
                        )
                        projection["cost"] = {
                            k: v for k, v in cost.items() if not k.endswith("_ns")
                        }
                        write(
                            "projection",
                            dict(
                                slot=slot,
                                arm=arm,
                                constraints=deepcopy(memories[arm]["rows"]),
                                **projection,
                            ),
                        )
                        if projection["fallback"] != "none":
                            write(
                                "fallback",
                                dict(
                                    slot=slot,
                                    arm=arm,
                                    reason="projection",
                                    fallback=projection["fallback"],
                                    parameters=projection["point"][:-1],
                                ),
                            )
                        theta = np.asarray(projection["point"][:-1])
                    pending["candidates"][arm] = theta.tolist()
                    write(
                        "candidate",
                        dict(
                            slot=slot,
                            arm=arm,
                            parameters=theta.tolist(),
                            endpoint_hash=canonical_hash(theta.tolist()),
                            incumbent=states[arm].tolist(),
                            incumbent_hash=canonical_hash(states[arm].tolist()),
                            constraints=deepcopy(memories[arm]["rows"]) if arm in GUARDED else [],
                            release_frontier=slot - 1,
                            update_slots=ids,
                        ),
                    )
                write(
                    "block",
                    dict(
                        slot=slot,
                        arm="shared",
                        selection="next12 eligible admission releases strictly after commitment",
                        next_opportunity=min(slot + 64, len(sources)),
                        requested=12,
                    ),
                )
        origin = slot - 20
        if origin >= 0:
            y = release(origin)
            if y is not None and (type(y) is not int or y not in (0, 1)):
                raise ValueError("label_contract")
            released[origin] = y
            row = sources[origin]
            valid = bool(row["public_eligible"] and row.get("eligible", True) and y is not None)
            role = "admission" if partition(row["source_cluster_id"]) == 0 else "update"
            write(
                "release",
                dict(
                    slot=origin,
                    release_slot=slot,
                    source=row["source_cluster_id"],
                    family_id=row["family_id"],
                    arm="shared",
                    y=y,
                    eligible=valid,
                    feedback_role=role,
                ),
            )
            if valid and role == "update":
                updates.append(origin)
                for arm in GUARDED:
                    began = time.process_time_ns()
                    receipt = dict(
                        source_id=row["source_cluster_id"],
                        role="update",
                        eligible=True,
                        release_slot=slot,
                        observed_slot=slot,
                        phi=phi[origin].tolist(),
                        y=y,
                    )
                    status = kernel.Memory._apply(memories[arm], receipt)
                    if status == "added":
                        event = memories[arm]["events"][-1]
                        write(
                            "addition",
                            dict(
                                slot=origin,
                                arm=arm,
                                receipt=receipt,
                                constraint=event["addition"],
                                active_count=len(memories[arm]["rows"]),
                            ),
                        )
                        for identity in event["evictions"]:
                            write(
                                "eviction",
                                dict(slot=origin, release_slot=slot, arm=arm, source_id=identity),
                            )
                    costs.append(
                        dict(
                            seed=seed,
                            slot=slot,
                            arm=arm,
                            component="constraint_construction_and_storage",
                            cpu_ns=time.process_time_ns() - began,
                            units=int(status == "added"),
                        )
                    )
            if pending and slot > pending["slot"] and valid and role == "admission":
                pending["observed"].append(origin)
                write(
                    "consume",
                    dict(
                        slot=origin,
                        release_slot=slot,
                        arm="shared",
                        candidate_slot=pending["slot"],
                        source=row["source_cluster_id"],
                        y=y,
                    ),
                )
            if pending and len(pending["observed"]) == 12:
                ids = pending["observed"]
                for arm in ARMS:
                    began = time.process_time_ns()
                    decision: Json = dict(
                        alpha=None,
                        fallback="none",
                        checks=[],
                        all_rejected=False,
                        changed=False,
                        parameters=states[arm].tolist(),
                    )
                    if arm == "unconditional":
                        decision.update(
                            alpha=1.0,
                            parameters=pending["candidates"][arm],
                            changed=pending["candidates"][arm] != states[arm].tolist(),
                        )
                    if arm in GUARDED:
                        decision = admit(
                            head,
                            states[arm],
                            np.asarray(pending["candidates"][arm]),
                            initial,
                            x[ids],
                            labels(ids),
                            memories[arm]["rows"],
                            arm == "projected_fresh",
                        )
                        for check in decision["checks"]:
                            write(
                                "alpha",
                                dict(
                                    slot=slot,
                                    candidate_slot=pending["slot"],
                                    arm=arm,
                                    guard_slots=ids,
                                    **check,
                                ),
                            )
                        if decision["fallback"] != "none":
                            write(
                                "fallback",
                                dict(
                                    slot=slot,
                                    arm=arm,
                                    reason="admission",
                                    fallback=decision["fallback"],
                                    parameters=decision["parameters"],
                                ),
                            )
                    states[arm] = np.asarray(decision["parameters"])
                    write(
                        "commit",
                        dict(
                            slot=slot,
                            candidate_slot=pending["slot"],
                            arm=arm,
                            alpha=decision["alpha"],
                            all_rejected=decision["all_rejected"],
                            changed=decision["changed"],
                            fallback=decision["fallback"],
                            status="frozen"
                            if arm == "frozen"
                            else "deferred"
                            if decision["alpha"] is None
                            else "zero"
                            if decision["alpha"] == 0
                            else "accepted",
                            parameters=states[arm].tolist(),
                            head_hash=canonical_hash(states[arm].tolist()),
                        ),
                    )
                    costs.append(
                        dict(
                            seed=seed,
                            slot=slot,
                            arm=arm,
                            component="admission_full_validation_fallback_and_storage",
                            cpu_ns=time.process_time_ns() - began,
                            units=len(decision["checks"]) * 12,
                        )
                    )
                pending = {}
    if pending:
        write(
            "pending",
            dict(
                slot=len(sources),
                arm="shared",
                status="censored",
                reason="stream_end",
                candidate_slot=pending["slot"],
                consumed_slots=pending["observed"],
            ),
        )
    for arm in ARMS:
        write(
            "budget",
            dict(arm=arm, **budgets[arm], cap=12, extra_projection_work=arm == "projected_fresh"),
        )
        write(
            "seal",
            dict(
                arm=arm,
                parameters=states[arm].tolist(),
                head_hash=canonical_hash(states[arm].tolist()),
                retention_labels_opened=False,
            ),
        )
    costs.append(
        dict(
            seed=seed,
            component="active_memory_bytes",
            units=sum(len(json.dumps(memories[a]["rows"]).encode()) for a in GUARDED)
            + sum(s.nbytes for s in states.values())
            + x.nbytes,
        )
    )


def measure(data: Json, raw: Path, *, budget_s: float = 1200) -> Json:
    """A durable deterministic prefix resumes before any retention labels open."""
    inputs = {k: data[k] for k in ("head", "sources", "seeds")}
    raw.mkdir(parents=True, exist_ok=True)
    path = raw / "inputs.json"
    if path.exists() and json.loads(path.read_text()) != inputs:
        raise ValueError("input_drift")
    atomic_json(path, inputs)
    vault = data.get("labels")
    deadline = time.monotonic() + budget_s
    costs: list[Json] = []
    progress("benchmark_before", 0, len(inputs["seeds"]))
    for index, seed in enumerate(inputs["seeds"]):
        journal = old.Journal(raw / f"seed-{seed}")
        heartbeat = time.monotonic()

        def emit(kind: str, row: Json) -> None:
            nonlocal heartbeat
            if time.monotonic() > deadline:
                raise TimeoutError("numerical_budget")
            journal.emit(kind, row)
            if os.environ.get("CARNOT_8076_CRASH_EVENT") == kind:
                os._exit(73)
            if time.monotonic() - heartbeat >= 30:
                progress(
                    "trajectory_events",
                    journal.index,
                    max(0, len(inputs["sources"]) - row.get("slot", 0)),
                )
                heartbeat = time.monotonic()

        def release(origin: int) -> Any:
            nonlocal vault
            if vault is None:
                vault = {
                    r["family_id"]: r["eligible_y"]
                    for r in json.loads(checked(data["target_reference"]).read_text())["rows"]
                }
            return vault[inputs["sources"][origin]["family_id"]]

        try:
            trajectory(inputs, seed, emit, release, costs)
            if journal.index < len(journal.prefix):
                raise ValueError("trailing_events")
        finally:
            journal.close()
        progress("seed_complete", index + 1, len(inputs["seeds"]) - index - 1)
    if not (raw / "cpu_costs.json").exists():
        atomic_json(raw / "cpu_costs.json", dict(rows=costs))
    result = reduce(raw)
    atomic_json(raw / "final_head_seals.json", dict(rows=result["final_head_seals"]))
    progress("benchmark_after", len(inputs["seeds"]), 0)
    return result


def reduce(raw: Path) -> Json:
    """Reconstruct each equation and reduction from immutable primitive events."""
    data = json.loads((raw / "inputs.json").read_text())
    result: Json = {f: [] for f in FIELDS.values()}
    for index, seed in enumerate(data["seeds"]):
        with sqlite3.connect(
            f"file:{raw / f'seed-{seed}' / 'ledger.sqlite'}?mode=ro", uri=True
        ) as db:
            events = [
                (kind, json.loads(payload))
                for kind, payload in db.execute("SELECT kind,payload FROM events ORDER BY seq")
            ]
        labels = {r["slot"]: r["y"] for k, r in events if k == "release"}
        expected: list[tuple[str, Json]] = []
        trajectory(data, seed, lambda k, r: expected.append((k, r)), lambda i: labels[i], [])
        if expected != events:
            raise ValueError("event_order_or_operand_drift")
        for kind, row in events:
            result[FIELDS[kind]].append(row)
        progress("cold_seed_reduced", index + 1, len(data["seeds"]) - index - 1)
    feedback = {(r["seed"], r["slot"]): r for r in result["feedback_release_rows"]}
    rows = []
    for row in result["issued_prediction_rows"]:
        released = feedback.get((row["seed"], row["slot"]))
        valid = bool(released and released["eligible"])
        y = released["y"] if released and valid else None
        rows.append(
            dict(
                row,
                unit=f"stream/{row['slot']}",
                condition="original_delayed_stream",
                y=y,
                numerator=loss(row["action"], y) if valid else None,
                denominator=int(valid),
                brier=(row["probability"] - y) ** 2 if valid else None,
                status="completed" if valid else "censored" if released is None else "excluded",
                exclusion_reason=None
                if valid
                else "unreleased_tail"
                if released is None
                else "ineligible_complete_target",
            )
        )
    first = [r for r in result["feedback_release_rows"] if r["seed"] == data["seeds"][0]]
    completed = sum(r["eligible"] for r in first)
    counts = dict(
        intended_count=len(data["sources"]),
        eligible_count=completed,
        independent_count=len({r["source"] for r in first if r["eligible"]}),
        completed_count=completed,
        censored_count=len(data["sources"]) - len(first),
        excluded_count=len(first) - completed,
        failed_count=0,
    )
    result.update(rows=rows, **counts)
    result["sample_size_budget"] = dict(counts, seeds_are_independent=False, independent_datasets=1)
    costs = json.loads((raw / "cpu_costs.json").read_text())["rows"]
    result["cpu_update_costs"] = [r for r in costs if "cpu_ns" in r]
    result["memory_bytes"] = dict(
        active_peak=max(r["units"] for r in costs if r["component"] == "active_memory_bytes"),
        audit_journal=sum(p.stat().st_size for p in raw.rglob("ledger.sqlite")),
        scope="active vectors, heads, serialized constraints; audit journal separately charged",
    )
    commits = result["durable_commit_rows"]
    result["behavior_counts"] = dict(
        noop=sum(not r["changed"] for r in commits if r["arm"] != "frozen"),
        all_rejected=sum(r["all_rejected"] for r in commits),
        reset=sum(r["fallback"] == "initial" for r in result["fallback_rows"]),
        accepted_nonzero=sum(
            r["alpha"] is not None and r["alpha"] > 0 and r["changed"] for r in commits
        ),
        benefit_credit=0,
    )
    return result

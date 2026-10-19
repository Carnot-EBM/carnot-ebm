"""REQ-REPORT-8077: cold equations keep later labels out of issued decisions.

This reader uses the qualified independent V698 basis and empirical guard.
It imports no producer transition, projection implementation or aggregate reducer.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from carnot.reporting.current_work_receipt import canonical_hash
from carnot.verify import fresh_learning_audit_8065 as prior
from carnot.verify import learning_retention_audit_8026 as math

Json = dict[str, Any]
Array = NDArray[np.float64]
ARMS = ["frozen", "unconditional", "ray_fresh", "projected_fresh"]
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
CONFIG: Json = dict(seed=6998077, draws=10000, blocks=[32, 16, 64], margin=0.02, timeline=256)


def residual(point: Array, rows: list[Json], initial: Array) -> Array:
    """Check every inequality and box face, since sampled success is insufficient."""
    low, high = initial - 0.5, initial + 0.5
    low[-1] = high[-1] = 1.0
    matrix = np.asarray([r["normal"] for r in rows]).reshape(len(rows), len(initial))
    return np.concatenate(
        (
            np.maximum(np.asarray([r["rhs"] for r in rows]) - matrix @ point, 0),
            np.maximum(low - point, 0),
            np.maximum(point - high, 0),
        )
    )


def projection(
    observed: Json,
    rows: list[Json],
    initial: Array,
    incumbent: Array,
    seed: int,
    *,
    budget: int = 256,
) -> Json:
    """Rebuild every seeded correction and separately validate budget fallbacks."""
    raw = np.asarray(observed["proposal"], dtype=float)
    point = np.clip(raw, initial - 0.5, initial + 0.5)
    point[-1] = 1.0
    rng = np.random.default_rng(seed)
    steps = []
    full_checks = 1
    for _ in range(budget):
        full_checks += 1
        if np.max(residual(point, rows, initial)) <= 1e-8:
            break
        sample = rng.choice(len(rows), min(8, len(rows)), replace=False).tolist()
        chosen = min(
            sample,
            key=lambda i: (
                -float(rows[i]["rhs"] - np.asarray(rows[i]["normal"]) @ point),
                rows[i]["source_id"],
            ),
        )
        normal = np.asarray(rows[chosen]["normal"])
        violation = max(0.0, float(rows[chosen]["rhs"] - normal @ point))
        before = point.copy()
        if violation > 0:
            point = np.clip(
                point + violation * normal / np.sum(normal * normal), initial - 0.5, initial + 0.5
            )
            point[-1] = 1.0
        steps.append(
            dict(
                sample=sample,
                chosen=chosen,
                violation=violation,
                before=before.tolist(),
                after=point.tolist(),
            )
        )
    full_checks += 2
    candidate = residual(point, rows, initial)
    feasible = bool(np.isfinite(point).all() and np.max(candidate) <= 1e-8)
    fallback = "none"
    if not feasible:
        full_checks += 1
        point, fallback = (
            (incumbent.copy(), "incumbent")
            if np.max(residual(incumbent, rows, initial)) <= 1e-8
            else (initial.copy(), "initial")
        )
    result = dict(
        point=point.tolist(),
        candidate_residuals=candidate.tolist(),
        residuals=residual(point, rows, initial).tolist(),
        max_residual=float(np.max(residual(point, rows, initial))),
        fallback=fallback,
        projection_rows=steps,
        projection_steps=len(steps),
        candidate_feasible=feasible,
        feasible=True,
        termination="residual" if feasible else "budget_exhausted",
        distance=float(np.linalg.norm(point - raw)),
        cost=dict(
            full_residual_checks=full_checks + 1,
            row_dot_products=(full_checks + 1) * len(rows)
            + sum(len(r["sample"]) + 1 for r in steps),
            row_norm_evaluations=len(rows),
            coefficient_writes=(
                1 + sum(r["violation"] > 0 for r in steps) + int(fallback != "none")
            )
            * len(initial),
            cpu_calls=1,
            rust_calls=0,
            gpu_calls=0,
        ),
    )
    for key, expected in result.items():
        actual = (
            {k: v for k, v in observed[key].items() if not k.endswith("_ns")}
            if key == "cost"
            else observed[key]
        )
        math.equal("projection." + key, expected, actual)
    return result


def admission(
    head: Json,
    incumbent: Array,
    endpoint: Array,
    initial: Array,
    x: Array,
    y: Array,
    rows: list[Json],
    projected: bool,
) -> Json:
    """Keep the independent empirical guard and validate all current memory rows."""
    checks, _ = prior.guard(head, incumbent, endpoint, initial, x, y)
    for check in checks:
        point = incumbent + check["alpha"] * (endpoint - incumbent)
        check["max_residual"] = (
            float(np.max(residual(np.append(point, 1.0), rows, np.append(initial, 1.0))))
            if projected
            else None
        )
        if projected and check["max_residual"] > 1e-8:
            check["reasons"].append("current_memory")
            check["passed"] = False
    passing = [r["alpha"] for r in checks if r["passed"]]
    alpha = float(passing[0]) if passing else None
    fallback = "none" if alpha is not None else "incumbent"
    point = incumbent + alpha * (endpoint - incumbent) if alpha is not None else incumbent.copy()
    if (
        alpha is None
        and projected
        and np.max(residual(np.append(point, 1.0), rows, np.append(initial, 1.0))) > 1e-8
    ):
        point, fallback = initial.copy(), "initial"
    return dict(
        checks=checks,
        alpha=alpha,
        parameters=point.tolist(),
        fallback=fallback,
        all_rejected=bool(checks) and not passing,
        changed=bool(np.any(point != incumbent)),
    )


def reconstruct(raw: Path, targets: Json, *, budget_s: float = 900) -> Json:
    """Generate each expected event causally and compare exact journal operands."""
    deadline = time.monotonic() + budget_s
    data = json.loads((raw / "inputs.json").read_text())
    head, sources = data["head"], data["sources"]
    initial = np.asarray(head["parameters"], dtype=float)
    x = np.asarray(
        [math.design(head, r) if r["public_eligible"] else np.zeros(len(initial)) for r in sources]
    )
    intercept, slope = head["calibration"]
    phi = np.column_stack((slope * x, np.full(len(x), intercept)))
    w0 = np.append(initial, 1.0)
    result: Json = {field: [] for field in FIELDS.values()}
    result["reconstructed_projection_residuals"] = []
    for seed in data["seeds"]:
        with sqlite3.connect(
            f"file:{raw / f'seed-{seed}' / 'ledger.sqlite'}?mode=ro", uri=True
        ) as db:
            events = db.execute("SELECT seq,kind,payload FROM events ORDER BY seq").fetchall()
        cursor = 0
        states = {arm: initial.copy() for arm in ARMS}
        memory: Json = {arm: [] for arm in ARMS[2:]}
        released: dict[int, Any] = {}
        updates: list[int] = []
        pending: Json = {}
        budgets = {arm: dict(gradients=0, label_operations=0) for arm in ARMS}

        def emit(kind: str, **row: Any) -> None:
            nonlocal cursor
            if time.monotonic() > deadline:
                raise TimeoutError("independent_reduction_budget")
            expected = dict(seed=seed, **row)
            seq, actual_kind, payload = events[cursor]
            math.equal(
                f"event.{seed}.{cursor}",
                (cursor, kind, expected),
                (seq, actual_kind, json.loads(payload)),
            )
            result[FIELDS[kind]].append(deepcopy(expected))
            cursor += 1

        def ys(ids: list[int]) -> Array:
            return np.asarray([released[i] for i in ids], dtype=float)

        for slot, source in enumerate(sources):
            for arm in ARMS:
                p = (
                    float(expit(intercept + slope * float(x[slot] @ states[arm])))
                    if source["public_eligible"]
                    else None
                )
                emit(
                    "issue",
                    slot=slot,
                    source=source["source_cluster_id"],
                    family_id=source["family_id"],
                    arm=arm,
                    probability=p,
                    action=math.action(p),
                    head_hash=canonical_hash(states[arm].tolist()),
                )
            if slot in (64, 128, 192):
                if pending:
                    emit(
                        "pending",
                        slot=slot,
                        arm="shared",
                        status="censored",
                        reason="next_attempt",
                        candidate_slot=pending["slot"],
                        consumed_slots=pending["observed"],
                    )
                    pending = {}
                ids = updates[-32:]
                if not prior.adequate(ys(ids), 16):
                    emit(
                        "pending",
                        slot=slot,
                        arm="shared",
                        status="deferred",
                        reason="update_support",
                    )
                else:
                    pending = dict(slot=slot, observed=[], candidates={}, update_slots=ids)
                    for arm in ARMS[1:]:
                        theta = states[arm].copy()
                        for step in range(4):
                            before = theta.copy()
                            gradient = (
                                slope
                                * (
                                    x[ids].T
                                    @ (expit(intercept + slope * (x[ids] @ theta)) - ys(ids))
                                )
                                / len(ids)
                                + 0.002 * theta
                            )
                            theta -= 0.01 * gradient
                            emit(
                                "gradient",
                                slot=slot,
                                arm=arm,
                                step=step,
                                update_slots=ids,
                                before=before.tolist(),
                                gradient=gradient.tolist(),
                                after=theta.tolist(),
                            )
                        budgets[arm]["gradients"] += 4
                        budgets[arm]["label_operations"] += 4 * len(ids)
                        emit(
                            "raw",
                            slot=slot,
                            arm=arm,
                            parameters=theta.tolist(),
                            head_hash=canonical_hash(theta.tolist()),
                        )
                        if arm == "projected_fresh":
                            observed = json.loads(events[cursor][2])
                            math.equal("projection.memory", memory[arm], observed["constraints"])
                            math.equal(
                                "projection.identity",
                                (seed, slot, arm),
                                (observed["seed"], observed["slot"], observed["arm"]),
                            )
                            math.equal(
                                "projection.proposal",
                                np.append(theta, 1.0).tolist(),
                                observed["proposal"],
                            )
                            rebuilt = projection(
                                observed, memory[arm], w0, np.append(states[arm], 1.0), seed + slot
                            )
                            result["reconstructed_projection_residuals"].append(
                                dict(seed=seed, slot=slot, **rebuilt)
                            )
                            emit("projection", **{k: v for k, v in observed.items() if k != "seed"})
                            if rebuilt["fallback"] != "none":
                                emit(
                                    "fallback",
                                    slot=slot,
                                    arm=arm,
                                    reason="projection",
                                    fallback=rebuilt["fallback"],
                                    parameters=rebuilt["point"][:-1],
                                )
                            theta = np.asarray(rebuilt["point"][:-1])
                        pending["candidates"][arm] = theta.tolist()
                        emit(
                            "candidate",
                            slot=slot,
                            arm=arm,
                            parameters=theta.tolist(),
                            endpoint_hash=canonical_hash(theta.tolist()),
                            incumbent=states[arm].tolist(),
                            incumbent_hash=canonical_hash(states[arm].tolist()),
                            constraints=memory[arm] if arm in ARMS[2:] else [],
                            release_frontier=slot - 1,
                            update_slots=ids,
                        )
                    emit(
                        "block",
                        slot=slot,
                        arm="shared",
                        selection="next12 eligible admission releases strictly after commitment",
                        next_opportunity=min(slot + 64, len(sources)),
                        requested=12,
                    )
            origin = slot - 20
            if origin < 0:
                continue
            row = sources[origin]
            y = targets[row["family_id"]]
            math.equal("label_contract", True, y is None or type(y) is int and y in (0, 1))
            released[origin] = y
            valid = bool(row["public_eligible"] and row.get("eligible", True) and y is not None)
            feedback_role = prior.role(row)
            emit(
                "release",
                slot=origin,
                release_slot=slot,
                source=row["source_cluster_id"],
                family_id=row["family_id"],
                arm="shared",
                y=y,
                eligible=valid,
                feedback_role=feedback_role,
            )
            if valid and feedback_role == "update":
                updates.append(origin)
                for arm in ARMS[2:]:
                    receipt = dict(
                        source_id=row["source_cluster_id"],
                        role="update",
                        eligible=True,
                        release_slot=slot,
                        observed_slot=slot,
                        phi=phi[origin].tolist(),
                        y=y,
                    )
                    normal = (2 * y - 1) * phi[origin]
                    constraint = dict(
                        source_id=row["source_cluster_id"],
                        normal=normal.tolist(),
                        rhs=min(
                            float((2 * y - 1) * (phi[origin] @ w0)), 0.0 if y else float(np.log(9))
                        ),
                        release_slot=slot,
                        release_receipt=receipt,
                    )
                    ordered = sorted(
                        [*memory[arm], constraint],
                        key=lambda r: (r["release_slot"], r["source_id"]),
                    )
                    memory[arm] = ordered[-64:]
                    emit(
                        "addition",
                        slot=origin,
                        arm=arm,
                        receipt=receipt,
                        constraint=constraint,
                        active_count=len(memory[arm]),
                    )
                    for evicted in ordered[:-64]:
                        emit(
                            "eviction",
                            slot=origin,
                            release_slot=slot,
                            arm=arm,
                            source_id=evicted["source_id"],
                        )
            if pending and slot > pending["slot"] and valid and feedback_role == "admission":
                pending["observed"].append(origin)
                emit(
                    "consume",
                    slot=origin,
                    release_slot=slot,
                    arm="shared",
                    candidate_slot=pending["slot"],
                    source=row["source_cluster_id"],
                    y=y,
                )
            if pending and len(pending["observed"]) == 12:
                ids = pending["observed"]
                for arm in ARMS:
                    alpha: float | None = None
                    fallback = "none"
                    checks: list[Json] = []
                    all_rejected = changed = False
                    point = states[arm].copy()
                    if arm == "unconditional":
                        alpha, point = 1.0, np.asarray(pending["candidates"][arm])
                        changed = bool(np.any(point != states[arm]))
                    if arm in ARMS[2:]:
                        decision = admission(
                            head,
                            states[arm],
                            np.asarray(pending["candidates"][arm]),
                            initial,
                            x[ids],
                            ys(ids),
                            memory[arm],
                            arm == "projected_fresh",
                        )
                        alpha, fallback = decision["alpha"], decision["fallback"]
                        point = np.asarray(decision["parameters"])
                        all_rejected, changed = decision["all_rejected"], decision["changed"]
                        for check in decision["checks"]:
                            emit(
                                "alpha",
                                slot=slot,
                                candidate_slot=pending["slot"],
                                arm=arm,
                                guard_slots=ids,
                                **check,
                            )
                        if fallback != "none":
                            emit(
                                "fallback",
                                slot=slot,
                                arm=arm,
                                reason="admission",
                                fallback=fallback,
                                parameters=point.tolist(),
                            )
                    states[arm] = point
                    emit(
                        "commit",
                        slot=slot,
                        candidate_slot=pending["slot"],
                        arm=arm,
                        alpha=alpha,
                        all_rejected=all_rejected,
                        changed=changed,
                        fallback=fallback,
                        status="frozen"
                        if arm == "frozen"
                        else "deferred"
                        if alpha is None
                        else "zero"
                        if alpha == 0
                        else "accepted",
                        parameters=point.tolist(),
                        head_hash=canonical_hash(point.tolist()),
                    )
                pending = {}
        if pending:
            emit(
                "pending",
                slot=len(sources),
                arm="shared",
                status="censored",
                reason="stream_end",
                candidate_slot=pending["slot"],
                consumed_slots=pending["observed"],
            )
        for arm in ARMS:
            emit(
                "budget",
                arm=arm,
                **budgets[arm],
                cap=12,
                extra_projection_work=arm == "projected_fresh",
            )
            emit(
                "seal",
                arm=arm,
                parameters=states[arm].tolist(),
                head_hash=canonical_hash(states[arm].tolist()),
                retention_labels_opened=False,
            )
        math.equal("complete_journal", len(events), cursor)
        print(
            f"[exp8077] cold_seed_reconstructed completed={data['seeds'].index(seed) + 1} pending={len(data['seeds']) - data['seeds'].index(seed) - 1}",
            flush=True,
        )
    released_rows = {(r["seed"], r["slot"]): r for r in result["feedback_release_rows"]}
    result["rows"] = []
    for row in result["issued_prediction_rows"]:
        feedback = released_rows.get((row["seed"], row["slot"]))
        valid = bool(feedback and feedback["eligible"])
        y = feedback["y"] if feedback and valid else None
        result["rows"].append(
            dict(
                row,
                unit=f"stream/{row['slot']}",
                condition="original_delayed_stream",
                y=y,
                numerator=math.cost(row["action"], int(y)) if valid and y is not None else None,
                denominator=int(valid),
                brier=(row["probability"] - y) ** 2 if valid else None,
                status="completed" if valid else "censored" if feedback is None else "excluded",
                exclusion_reason=None
                if valid
                else "unreleased_tail"
                if feedback is None
                else "ineligible_complete_target",
            )
        )
    commits = result["durable_commit_rows"]
    cutoff = min((r["slot"] for r in commits), default=256)
    result["first_shared_opportunity_slot"] = cutoff
    result["later_source_rows"] = [
        r
        for r in result["rows"]
        if r["slot"] > cutoff and prior.role(sources[r["slot"]]) == "update"
    ]
    result["qualified_constraint_additions"] = len(result["constraint_addition_rows"])
    result["resets"] = sum(r["fallback"] == "initial" for r in result["fallback_rows"])
    result["unchanged_sources"] = len({r["source"] for r in result["later_source_rows"]})
    return result


def bootstrap(diff: list[float], block: int) -> Json:
    """Retain all timeline masks and count only finite sampled source means."""
    x = np.asarray(diff, dtype=float)
    result: Json = dict(
        gain=None,
        raw_p=1.0,
        margin=0.02,
        block_length=block,
        draws=10000,
        slot_count=256,
        completed_draws=0,
        censored_draws=10000,
        lower=None,
        interval=[None, None],
    )
    if not np.isfinite(x).any():
        return result
    rng = np.random.default_rng(CONFIG["seed"])
    starts = rng.integers(0, 257 - block, (10000, int(np.ceil(256 / block))))
    ids = (starts[:, :, None] + np.arange(block)).reshape(10000, -1)[:, :256]
    samples = x[ids]
    n = np.isfinite(samples).sum(axis=1)
    draws = np.nansum(samples[n > 0], axis=1) / n[n > 0]
    gain = float(np.nanmean(x))
    errors = draws - gain
    result.update(
        gain=gain,
        raw_p=float((1 + sum(errors >= gain - 0.02 - 1e-15)) / (1 + len(errors))),
        completed_draws=len(errors),
        censored_draws=10000 - len(errors),
        lower=gain - float(np.quantile(errors, 0.95)),
        interval=(gain - np.quantile(errors, [0.975, 0.025])).tolist(),
    )
    return result


def comparisons(rows: list[Json], retained: list[Json]) -> Json:
    """Reuse independent paired safety arithmetic, with the V699 contrast and seed."""
    mapping = dict(zip(ARMS, prior.ARMS, strict=True))
    reverse = {v: k for k, v in mapping.items()}
    reduced = prior.comparisons(
        [dict(r, arm=mapping[r["arm"]]) for r in rows],
        [dict(r, arm=mapping[r["arm"]]) for r in retained],
    )
    h = reduced["primary_hypothesis_results"][0]
    cells = {(r["source"], r["seed"], r["arm"]): r for r in rows if r["denominator"]}
    seeds = sorted({r["seed"] for r in rows})
    common = [
        s
        for s in sorted({r["source"] for r in rows})
        if seeds and all((s, seed, arm) in cells for seed in seeds for arm in ARMS)
    ]
    diff = [float("nan")] * 256
    for source in common:
        diff[cells[source, seeds[0], "frozen"]["slot"]] = float(
            np.mean(
                [
                    cells[source, seed, "ray_fresh"]["numerator"]
                    - cells[source, seed, "projected_fresh"]["numerator"]
                    for seed in seeds
                ]
            )
        )
    h.update(
        hypothesis="H2",
        comparison="ray_fresh cost minus projected_fresh cost",
        tests=[bootstrap(diff, block) for block in CONFIG["blocks"]],
        multiplicity="Raw H2 and safety/support to capstone Holm .05 over exactly H1/H2",
        uncertainty_scope="Conditional exposed development trajectory; seeds averaged within sources; no unseen-environment generalization",
    )
    h["qualified_benefit"] = bool(
        h["support_passed"]
        and h["safety_passed"]
        and h["beneficial_changed_sources"] >= 5
        and h["tests"][0]["gain"] is not None
        and h["tests"][0]["gain"] >= 0.02
        and h["tests"][0]["raw_p"] < 0.05
    )
    h["capstone_family_p"] = (
        h["tests"][0]["raw_p"] if h["support_passed"] and h["safety_passed"] else 1.0
    )
    for r in h["retention_checks"] + reduced["per_seed_false_accept_rows"]:
        r["arm"] = reverse[r["arm"]]
    return dict(
        H2=h,
        primary_hypothesis_results=[h],
        per_seed_false_accept_rows=reduced["per_seed_false_accept_rows"],
    )

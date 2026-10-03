"""REQ-REPORT-8051: accept small-head motion using only released guard labels.

A guard checks a finite exposed sample. It cannot certify future safety.
Durable issues keep their original decisions after feedback changes the head.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sqlite3
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from carnot import experiment_8046_v697_branch_protocols as protocol
from carnot.experiment_8021_v695_typed_decision_test import action, loss
from carnot.experiment_8019_v695_eligible_targets import shard
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.evidence_features_custody_7980 import checked
from carnot.verify import causal_online_8025 as old
from carnot.verify.windowed_online_8038 import equal

Json = dict[str, Any]
Array = NDArray[np.float64]
CONFIG = dict(
    protocol.METHODS["learning"],
    statistics="descriptive per original source; no benefit test or retention access",
)
ARMS = CONFIG["arms"]
BEGAN = time.monotonic()


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Report real counts so a stalled numerical child remains visible."""
    print(
        f"[exp8051] phase={phase} elapsed_s={time.monotonic() - BEGAN:.3f} completed={completed} pending={pending}",
        flush=True,
    )


def guard(
    head: Json, theta: Array, delta: Array, initial: Array, x: Array, y: Array, arm: str
) -> Json:
    """Check calibrated probabilities against the initial head on every released guard."""
    ready = len(y) >= 8 and min(int(sum(y == c)) for c in (0, 1)) >= 2
    diagnostics = []
    if ready:

        def operands(parameters: Array) -> Json:
            a, b = head["calibration"]
            ps = np.asarray(old.expit(a + b * (x @ parameters)), dtype=float).tolist()
            acts = [action(p) for p in ps]
            return dict(
                probabilities=ps,
                actions=acts,
                brier=float(np.mean((np.array(ps) - y) ** 2)),
                typed_cost=float(np.mean([loss(a, int(v)) for a, v in zip(acts, y, strict=True)])),
                false_accepts=[
                    i
                    for i, (a, v) in enumerate(zip(acts, y, strict=True))
                    if a == "accept" and v == 1
                ],
            )

        baseline = operands(initial)
        for alpha in protocol.METHODS["guard"]["alphas"]:
            candidate = operands(theta + alpha * delta)
            reasons = []
            if candidate["brier"] > baseline["brier"] + 1e-12:
                reasons.append("brier")
            if candidate["typed_cost"] > baseline["typed_cost"] + 1e-12:
                reasons.append("typed_cost")
            added = sorted(set(candidate["false_accepts"]) - set(baseline["false_accepts"]))
            if added:
                reasons.append("new_false_accept")
            diagnostics.append(
                dict(
                    alpha=alpha,
                    baseline=baseline,
                    candidate=candidate,
                    new_false_accepts=added,
                    reasons=reasons,
                    admissible=not reasons,
                    numerator=len(y),
                    denominator=len(y),
                )
            )
    passing = [r["alpha"] for r in diagnostics if r["admissible"]]
    alpha = 1 if arm == "unconstrained" else passing[0] if passing else 0 if not ready else None
    return dict(
        alpha=alpha,
        parameters=(initial if alpha is None else theta + alpha * delta).tolist(),
        diagnostics=diagnostics,
        reset=alpha is None,
        rejected=alpha != 1,
        status="waiting_guard" if not ready else "reset" if alpha is None else "commit",
    )


def partition_rows(sources: list[Json]) -> list[Json]:
    """Public identity fixes the role before any label is opened."""
    return [
        dict(
            family_id=r["family_id"],
            source_cluster_id=r["source_cluster_id"],
            original_slot=i,
            bucket=protocol.partition(r["source_cluster_id"]),
            feedback_role="guard" if protocol.partition(r["source_cluster_id"]) == 0 else "update",
        )
        for i, r in enumerate(sources)
    ]


def selection(rows: list[Json], seed: int, block: int) -> list[Json]:
    """Both adaptive arms receive the same cumulative, label-independent draw."""
    ordered = sorted(rows, key=lambda r: canonical_hash(dict(seed=seed, identity=r["family_id"])))
    return protocol.prior.select(ordered, "cumulative", seed, block)


def measure(data: Json, raw: Path, *, budget_s: float = 900) -> Json:
    """Commit each issue before asking the evaluator for its single due label."""
    raw.mkdir(parents=True, exist_ok=True)
    atomic_json(raw / "inputs.json", {k: data[k] for k in ("head", "sources", "seeds")})
    atomic_json(
        raw / "methods.json",
        dict(
            config=CONFIG,
            guard=protocol.METHODS["guard"],
            optimizer=old.CONFIG,
            partition=partition_rows(data["sources"]),
        ),
    )
    sources = data["sources"]
    partition_ref = shard(raw, "partitions", dict(rows=partition_rows(sources)))
    vectors = [old.design(data["head"], r) if r["public_eligible"] else None for r in sources]
    ledgers = []
    deadline = time.monotonic() + budget_s
    total = len(sources) * len(ARMS) * len(data["seeds"])
    done = 0
    labels: Json = {}
    progress("benchmark_before", done, total)
    try:
        for seed in data["seeds"]:
            ledger = old.Ledger(raw / f"seed-{seed}")
            ledgers.append(ledger)
            for arm in ARMS:
                progress("small_head_load_before", done, total - done)
                head = copy.deepcopy(data["head"])
                initial = old.coefficients(head)
                progress("small_head_load_after", done, total - done)
                released: list[Json] = []
                guards: list[Json] = []
                attempts = 0
                for slot, source in enumerate(sources):
                    if time.monotonic() > deadline:
                        raise TimeoutError("numerical_budget")
                    p = old.probability(head, vectors[slot]) if vectors[slot] is not None else None
                    ledger.write(
                        "issue",
                        f"issue/{arm}/{seed}/{slot}",
                        dict(
                            arm=arm,
                            seed=seed,
                            slot=slot,
                            family_id=source["family_id"],
                            probability=p,
                            action=action(p),
                            head_hash=canonical_hash(head),
                        ),
                    )
                    origin = slot - 20
                    if origin >= 0:
                        if not labels:
                            labels.update(data.get("labels", {}))
                            if not labels:
                                labels.update(
                                    {
                                        r["family_id"]: r["eligible_y"]
                                        for r in json.loads(
                                            checked(data["target_reference"]).read_text()
                                        )["rows"]
                                    }
                                )
                        y = labels[sources[origin]["family_id"]]
                        equal(y is None or (type(y) is int and y in (0, 1)), True, "label_contract")
                        eligible = y is not None and vectors[origin] is not None
                        feedback = dict(
                            arm=arm,
                            seed=seed,
                            origin_slot=origin,
                            due_slot=slot,
                            family_id=sources[origin]["family_id"],
                            y=y,
                            eligible=eligible,
                            feedback_role="guard"
                            if protocol.partition(sources[origin]["source_cluster_id"]) == 0
                            else "update",
                        )
                        ledger.write("release", f"release/{arm}/{seed}/{origin}", feedback)
                        if eligible:
                            pool = guards if feedback["feedback_role"] == "guard" else released
                            pool.append(feedback)
                            chosen = (
                                selection(released, seed, len(released) // 16)
                                if pool is released
                                and len(released) % 16 == 0
                                and arm != "frozen_no_write"
                                else []
                            )
                            for index, selected in enumerate(chosen if attempts < 64 else []):
                                candidate = copy.deepcopy(head)
                                info = old.update(
                                    candidate, vectors[selected["origin_slot"]], selected["y"]
                                )
                                ledger.write(
                                    "gradient",
                                    f"gradient/{arm}/{seed}/{attempts}",
                                    dict(
                                        arm=arm,
                                        seed=seed,
                                        slot=slot,
                                        family_id=selected["family_id"],
                                        origin_slot=selected["origin_slot"],
                                        y=selected["y"],
                                        before_hash=canonical_hash(head),
                                        proposed_coefficients=old.coefficients(candidate).tolist(),
                                        info=info,
                                    ),
                                )
                                commit(
                                    ledger,
                                    head,
                                    initial,
                                    vectors,
                                    guards,
                                    old.coefficients(candidate) - old.coefficients(head),
                                    arm,
                                    seed,
                                    slot,
                                    f"gradient/{attempts}",
                                )
                                attempts += 1
                            if pool is guards and arm != "frozen_no_write":
                                commit(
                                    ledger,
                                    head,
                                    initial,
                                    vectors,
                                    guards,
                                    np.zeros_like(initial),
                                    arm,
                                    seed,
                                    slot,
                                    "guard_expansion",
                                )
                            if pool is released and len(released) % 16 == 0:
                                ledger.write(
                                    "checkpoint",
                                    f"checkpoint/{arm}/{seed}/{slot}",
                                    dict(
                                        arm=arm,
                                        seed=seed,
                                        slot=slot,
                                        head=shard(raw, "heads", head),
                                        partition=partition_ref,
                                        guard_ids=[r["family_id"] for r in guards],
                                        update_ids=[r["family_id"] for r in released],
                                    ),
                                )
                    done += 1
                ledger.write(
                    "checkpoint",
                    f"final/{arm}/{seed}",
                    dict(
                        arm=arm,
                        seed=seed,
                        slot=len(sources),
                        head=shard(raw, "heads", head),
                        partition=partition_ref,
                        guard_ids=[r["family_id"] for r in guards],
                        update_ids=[r["family_id"] for r in released],
                    ),
                )
                progress("arm_complete", done, total - done)
    finally:
        for ledger in ledgers:
            ledger.close()
    progress("benchmark_after", done, total - done)
    return reduce(raw)


def commit(
    ledger: old.Ledger,
    head: Json,
    initial: Array,
    vectors: list[Any],
    guards: list[Json],
    delta: Array,
    arm: str,
    seed: int,
    slot: int,
    reason: str,
) -> None:
    """Keep rejected proposals and reset operands in the same synchronous journal."""
    before = old.coefficients(head)
    x = np.array([vectors[r["origin_slot"]] for r in guards])
    y = np.array([r["y"] for r in guards])
    result = guard(head, before, delta, initial, x, y, arm)
    head.update(parameters=result["parameters"], decay_scale=1.0)
    ledger.write(
        "acceptance",
        f"acceptance/{arm}/{seed}/{slot}/{reason}",
        dict(
            arm=arm,
            seed=seed,
            slot=slot,
            reason=reason,
            guard_ids=[r["family_id"] for r in guards],
            before_coefficients=before.tolist(),
            proposed_coefficients=(before + delta).tolist(),
            head_hash=canonical_hash(head),
            **result,
        ),
    )


def reduce(raw: Path) -> Json:
    """Rebuild issued states with dense gradient equations, without trusting aggregates."""
    data = json.loads((raw / "inputs.json").read_text())
    sources = data["sources"]
    partitions = partition_rows(sources)
    methods = json.loads((raw / "methods.json").read_text())
    equal(
        methods,
        dict(
            config=CONFIG,
            guard=protocol.METHODS["guard"],
            optimizer=old.CONFIG,
            partition=partitions,
        ),
        "methods_drift",
    )
    vectors = [old.design(data["head"], r) if r["public_eligible"] else None for r in sources]
    databases = [sqlite3.connect(p) for p in sorted(raw.glob("seed-*/ledger.sqlite"))]
    events = (
        (kind, json.loads(text))
        for db in databases
        for kind, text in db.execute("SELECT kind,payload FROM events ORDER BY seq")
    )
    issues: Json = {}
    states: Json = {}
    released: Json = {}
    guards: Json = {}
    attempts: Json = {}
    pending: Json = {}
    result: Json = dict(
        guard_partition_rows=partitions,
        issue_release_rows=[],
        candidate_update_rows=[],
        accepted_update_rows=[],
        rejection_rows=[],
        reset_rows=[],
        guard_check_rows=[],
        head_checkpoints=[],
        rows=[],
        attempted_gradient_counts=[],
        committed_update_counts=[],
    )
    initial = old.coefficients(data["head"])
    for kind, row in events:
        key = f"{row['arm']}/{row['seed']}"
        if key not in states:
            states[key] = copy.deepcopy(data["head"])
            released[key] = []
            guards[key] = []
            attempts[key] = 0
        head = states[key]
        arm = row["arm"]
        seed = row["seed"]
        if kind == "issue":
            slot = row["slot"]
            x = vectors[slot]
            p = old.probability(head, x) if x is not None else None
            equal(
                row,
                dict(
                    arm=arm,
                    seed=seed,
                    slot=slot,
                    family_id=sources[slot]["family_id"],
                    probability=p,
                    action=action(p),
                    head_hash=canonical_hash(head),
                ),
                "issued_state_drift",
            )
            equal(slot, len(issues.get(key, [])), "issue_order")
            issues.setdefault(key, []).append(row)
        elif kind == "release":
            origin = row["origin_slot"]
            issued = issues[key][origin]
            y = row["y"]
            equal(len(issues[key]) - 1, origin + 20, "release_clock")
            equal(
                row["eligible"],
                y is not None and vectors[origin] is not None,
                "release_eligibility",
            )
            equal(
                row["feedback_role"],
                "guard"
                if protocol.partition(sources[origin]["source_cluster_id"]) == 0
                else "update",
                "partition_drift",
            )
            if row["eligible"]:
                (guards[key] if row["feedback_role"] == "guard" else released[key]).append(row)
            record = dict(
                issued,
                source_cluster_id=sources[origin]["source_cluster_id"],
                y=y,
                feedback_role=row["feedback_role"],
                due_slot=row["due_slot"],
                eligible=row["eligible"],
                exclusion_reason=None if row["eligible"] else "unknown_or_public_unavailable",
                brier=(issued["probability"] - y) ** 2 if row["eligible"] else None,
                typed_cost=loss(issued["action"], y) if row["eligible"] else None,
                numerator=int(row["eligible"]),
                denominator=1,
                status="completed" if row["eligible"] else "excluded",
            )
            result["issue_release_rows"].append(record)
        elif kind == "gradient":
            chosen = selection(released[key], seed, len(released[key]) // 16)
            equal(row["family_id"], chosen[attempts[key] % 4]["family_id"], "selected_id_drift")
            equal(row["before_hash"], canonical_hash(head), "gradient_before")
            x = vectors[row["origin_slot"]]
            before = old.coefficients(head)
            gradient = (old.probability(head, x) - row["y"]) * head["calibration"][
                1
            ] * x + 0.002 * before
            proposed = before - 0.01 * gradient
            equal(
                bool(np.allclose(proposed, row["proposed_coefficients"], atol=1e-14, rtol=0)),
                True,
                "gradient_equation",
            )
            equal(
                row["origin_slot"] + 20 <= row["slot"]
                and protocol.partition(sources[row["origin_slot"]]["source_cluster_id"]) != 0,
                True,
                "gradient_access",
            )
            pending[key] = np.asarray(row["proposed_coefficients"]) - before
            attempts[key] += 1
            result["candidate_update_rows"].append(
                {k: v for k, v in row.items() if k not in {"proposed_coefficients", "info"}}
                | dict(
                    ledger_identity=f"gradient/{arm}/{seed}/{attempts[key] - 1}",
                    gradient_norm=row["info"]["logical_gradient_norm"],
                    hot_update_ns=row["info"]["hot_update_ns"],
                    bytes_written=row["info"]["bytes_written"],
                )
            )
        elif kind == "acceptance":
            current = old.coefficients(head)
            delta = (
                pending.pop(key)
                if row["reason"].startswith("gradient/")
                else np.zeros_like(initial)
            )
            x = np.array([vectors[r["origin_slot"]] for r in guards[key]])
            y = np.array([r["y"] for r in guards[key]])
            expected = guard(head, current, delta, initial, x, y, arm)
            equal({k: row[k] for k in expected}, expected, "guard_drift")
            equal(row["guard_ids"], [r["family_id"] for r in guards[key]], "guard_operands")
            equal(row["before_coefficients"], current.tolist(), "acceptance_before")
            equal(row["proposed_coefficients"], (current + delta).tolist(), "acceptance_proposal")
            head.update(parameters=expected["parameters"], decay_scale=1.0)
            equal(row["head_hash"], canonical_hash(head), "accepted_head")
            summary = {
                k: row[k]
                for k in (
                    "arm",
                    "seed",
                    "slot",
                    "reason",
                    "alpha",
                    "reset",
                    "rejected",
                    "status",
                    "head_hash",
                )
            }
            summary.update(
                guard_count=len(row["guard_ids"]),
                ledger_identity=f"acceptance/{arm}/{seed}/{row['slot']}/{row['reason']}",
                diagnostics=[
                    {
                        k: d[k]
                        for k in (
                            "alpha",
                            "admissible",
                            "reasons",
                            "new_false_accepts",
                            "numerator",
                            "denominator",
                        )
                    }
                    | dict(
                        brier=d["candidate"]["brier"],
                        typed_cost=d["candidate"]["typed_cost"],
                        baseline_brier=d["baseline"]["brier"],
                        baseline_cost=d["baseline"]["typed_cost"],
                    )
                    for d in row["diagnostics"]
                ],
            )
            result["guard_check_rows"].append(summary)
            if row["reason"].startswith("gradient/") and row["parameters"] != current.tolist():
                result["accepted_update_rows"].append(summary)
            if row["rejected"]:
                result["rejection_rows"].append(summary)
            if row["reset"]:
                result["reset_rows"].append(summary)
        elif kind == "checkpoint":
            equal(json.loads(checked(row["head"]).read_text()), head, "checkpoint_state")
            equal(
                json.loads(checked(row["partition"]).read_text())["rows"],
                partitions,
                "checkpoint_partition",
            )
            equal(row["guard_ids"], [r["family_id"] for r in guards[key]], "checkpoint_guard")
            equal(row["update_ids"], [r["family_id"] for r in released[key]], "checkpoint_update")
            result["head_checkpoints"].append(row)
        else:
            raise ValueError("unknown_event")
    for db in databases:
        db.close()
    for key, issued_rows in issues.items():
        equal(len(issued_rows), len(sources), "unfinished_arm")
        arm, seed_text = key.split("/")
        seed = int(seed_text)
        for row in issued_rows[max(0, len(sources) - 20) :]:
            result["issue_release_rows"].append(
                dict(
                    row,
                    source_cluster_id=sources[row["slot"]]["source_cluster_id"],
                    y=None,
                    feedback_role=partitions[row["slot"]]["feedback_role"],
                    due_slot=row["slot"] + 20,
                    eligible=False,
                    exclusion_reason="pending_tail_release",
                    brier=None,
                    typed_cost=None,
                    numerator=0,
                    denominator=1,
                    status="censored",
                )
            )
        eligible = [
            r
            for r in result["issue_release_rows"]
            if r["arm"] == arm and r["seed"] == seed and r["eligible"]
        ]
        later = [
            r
            for r in eligible
            if r["feedback_role"] == "update"
            and r["slot"]
            > min(
                [v["slot"] for v in result["candidate_update_rows"] if v["seed"] == seed]
                or [len(sources)]
            )
        ]
        for condition, rows in [("all_released", eligible), ("later_update_role", later)]:
            result["rows"].append(
                dict(
                    arm=arm,
                    seed=seed,
                    condition=condition,
                    numerator=len(rows),
                    denominator=len(sources),
                    brier_numerator=sum(r["brier"] for r in rows),
                    brier_denominator=len(rows),
                    typed_cost_numerator=sum(r["typed_cost"] for r in rows),
                    typed_cost_denominator=len(rows),
                    brier=float(np.mean([r["brier"] for r in rows])) if rows else None,
                    typed_cost=float(np.mean([r["typed_cost"] for r in rows])) if rows else None,
                    false_accepts=sum(r["action"] == "accept" and r["y"] == 1 for r in rows),
                    independent_count=len({r["source_cluster_id"] for r in rows}),
                )
            )
        expected_attempts = 0 if arm == "frozen_no_write" else min(64, len(released[key]) // 16 * 4)
        equal(attempts[key], expected_attempts, "attempt_budget")
        result["attempted_gradient_counts"].append(
            dict(arm=arm, seed=seed, count=attempts[key], denominator=64)
        )
        result["committed_update_counts"].append(
            dict(
                arm=arm,
                seed=seed,
                count=sum(
                    r["arm"] == arm
                    and r["seed"] == seed
                    and r["reason"].startswith("gradient/")
                    and not r["reset"]
                    for r in result["accepted_update_rows"]
                ),
                denominator=attempts[key],
            )
        )
    equal(
        set(issues), {f"{arm}/{seed}" for seed in data["seeds"] for arm in ARMS}, "scheduled_arms"
    )
    first = [
        r
        for r in result["issue_release_rows"]
        if r["arm"] == ARMS[0] and r["seed"] == data["seeds"][0]
    ]
    counts = dict(
        intended=len(sources),
        eligible=sum(r["eligible"] for r in first),
        completed=sum(r["status"] == "completed" for r in first),
        excluded=sum(r["status"] == "excluded" for r in first),
        failed=0,
        censored=sum(r["status"] == "censored" for r in first),
        independent=len({r["source_cluster_id"] for r in first if r["eligible"]}),
    )
    result.update(
        sample_size_budget=dict(counts, seeds_are_independent=False, independent_datasets=1),
        **{k + "_count": v for k, v in counts.items()},
    )
    return result


def controls() -> Json:
    """Synthetic sign controls establish guard sensitivity, not natural learning benefit."""
    head = dict(calibration=[0.0, 1.0])
    initial = np.zeros(1)
    x = np.tile([[-1.0], [1.0]], (4, 1))
    y = np.tile([0.0, 1.0], 4)
    good = guard(head, initial, np.ones(1), initial, x, y, "feedback_constrained")
    bad = guard(head, initial, -10 * np.ones(1), initial, x, y, "feedback_constrained")
    return dict(
        passed=good["alpha"] == 1 and bad["alpha"] == 0,
        beneficial_alpha=good["alpha"],
        destructive_alpha=bad["alpha"],
        scope="artificial sign control; no deployment or benefit credit",
    )

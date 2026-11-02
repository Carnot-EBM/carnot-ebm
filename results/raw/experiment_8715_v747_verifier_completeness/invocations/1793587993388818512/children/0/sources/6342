"""REQ-REPORT-8038: vary replay support while keeping the optimizer fixed.

Each issued state reaches synchronous storage before due feedback is returned.
Only a completed release block can spend the shared gradient budget.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sqlite3
import time
from typing import Any

import numpy as np

from carnot import experiment_8032_v696_sealed_methods as sealed
from carnot.experiment_8021_v695_typed_decision_test import action, loss
from carnot.experiment_8019_v695_eligible_targets import shard
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.verify import causal_online_8025 as old

Json = dict[str, Any]
ARMS = sealed.LEARNING_ARMS
CONFIG = dict(
    sealed.METHODS["learning"],
    slots=256,
    example_seed=101,
    numerical_budget_s=900,
    sparse_dense_tolerance=1e-10,
    selection="hashed source order then seeded uniform sampling; newest16 prior uniform",
    retention_labels_opened=False,
)
BEGAN = time.monotonic()


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Real counts and flushed elapsed time make bounded work visible."""
    print(
        f"[exp8038] phase={phase} elapsed_s={time.monotonic() - BEGAN:.3f} "
        f"completed={completed} pending={pending}",
        flush=True,
    )


def select(released: list[Json], arm: str, seed: int, block: int) -> tuple[list[Json], list[Json]]:
    """Canonical identity order avoids outcome-dependent ties in uniform draws."""
    pool = (
        released[-64:] if arm == "recent64" else released if arm == "cumulative" else released[-16:]
    )
    if arm in {"frozen_no_write", "newest16"}:
        return pool, [] if arm == "frozen_no_write" else old.select(pool, "uniform", seed)
    ordered = sorted(pool, key=lambda r: canonical_hash(dict(seed=seed, identity=r["family_id"])))
    return pool, sealed.select(ordered, "cumulative", seed, block)


def equal(actual: Any, expected: Any, name: str) -> None:
    """Reject mismatched evidence instead of repairing it during reduction."""
    if actual != expected:
        raise ValueError(name)


def measure(data: Json, raw: Path, *, budget_s: float = 900) -> Json:
    """Journal each causal event and seal block checkpoints before reduction.

    The evaluator closure returns one due label. It never supplies a retention
    vault or unreleased loss to selection. Private fixtures use the same path.
    """
    raw.mkdir(parents=True, exist_ok=True)
    atomic_json(raw / "inputs.json", {k: data[k] for k in ("head", "sources", "seeds")})
    atomic_json(raw / "methods.json", dict(config=CONFIG, optimizer=old.CONFIG))
    ledger = old.Ledger(raw)
    vault: Json = {}
    sources = data["sources"]
    vectors = [old.design(data["head"], r) if r["public_eligible"] else None for r in sources]
    deadline = time.monotonic() + budget_s
    total = len(sources) * len(ARMS) * len(data["seeds"])
    done = 0

    def write(kind: str, identity: str, row: Json) -> None:
        timing = ledger.write(kind, identity, row)
        ledger.write("timing", "timing/" + identity, dict(timing, kind=kind, identity=identity))

    progress("benchmark_before", done, total)
    try:
        for seed in data["seeds"]:
            for arm in ARMS:
                progress("small_head_load_before", done, total - done)
                head = copy.deepcopy(data["head"])
                checkpoint = shard(raw, "checkpoints", head)
                progress("small_head_load_after", done, total - done)
                issued, released, used = [], [], set()
                updates = 0
                for slot, source in enumerate(sources):
                    if time.monotonic() > deadline:
                        raise TimeoutError("numerical_budget")
                    x = vectors[slot]
                    p = old.probability(head, x) if x is not None else None
                    row = dict(
                        arm=arm,
                        seed=seed,
                        slot=slot,
                        family_id=source["family_id"],
                        source_cluster_id=source["source_cluster_id"],
                        probability=p,
                        action=action(p),
                        head_hash=canonical_hash(head),
                        eligibility=p is not None,
                        exclusion_reason=source.get("exclusion_reason"),
                    )
                    write("issue", f"issue/{arm}/{seed}/{slot}", row)
                    issued.append(row)
                    origin = slot - CONFIG["delay"]
                    if origin >= 0:
                        if not vault:
                            vault.update(data.get("labels", {}))
                            if not vault:
                                document = json.loads(checked(data["target_reference"]).read_text())
                                vault.update(
                                    {r["family_id"]: r["eligible_y"] for r in document["rows"]}
                                )
                        y = vault[sources[origin]["family_id"]]
                        equal(
                            y is None or (type(y) is int and y in (0, 1)), True, "target_contract"
                        )
                        prior = issued[origin]
                        valid = y is not None and prior["probability"] is not None
                        feedback = dict(
                            arm=arm,
                            seed=seed,
                            origin_slot=origin,
                            due_slot=slot,
                            family_id=prior["family_id"],
                            y=y,
                            eligibility=valid,
                            issued_probability=prior["probability"],
                            issued_action=prior["action"],
                            issued_decision_loss=loss(prior["action"], y) if valid else None,
                            issued_brier_loss=(prior["probability"] - y) ** 2 if valid else None,
                            exclusion_reason=None if valid else "unknown_or_public_unavailable",
                        )
                        write("release", f"release/{arm}/{seed}/{origin}", feedback)
                        if valid:
                            released.append(feedback)
                            if len(released) % 16 == 0:
                                block = len(released) // 16
                                pool, chosen = select(released, arm, seed, block)
                                buffer = dict(
                                    arm=arm,
                                    seed=seed,
                                    slot=slot,
                                    block=block,
                                    eligible_released_count=len(released),
                                    pool=[
                                        dict(
                                            family_id=r["family_id"],
                                            origin_slot=r["origin_slot"],
                                            release_age=slot - r["due_slot"],
                                            origin_age=slot - r["origin_slot"],
                                        )
                                        for r in pool
                                    ],
                                    selected_ids=[r["family_id"] for r in chosen],
                                )
                                write("buffer", f"buffer/{arm}/{seed}/{block}", buffer)
                                gradients = []
                                before_hash = canonical_hash(head)
                                for selected in chosen if updates < 64 else []:
                                    began = time.process_time_ns()
                                    info = old.update(
                                        head, vectors[selected["origin_slot"]], selected["y"]
                                    )
                                    info.update(
                                        update_cpu_ns=time.process_time_ns() - began,
                                        family_id=selected["family_id"],
                                        origin_slot=selected["origin_slot"],
                                        y=selected["y"],
                                        issued_brier_loss=selected["issued_brier_loss"],
                                        issued_decision_loss=selected["issued_decision_loss"],
                                        repeated_id=selected["family_id"] in used,
                                        update_index=updates,
                                        after_head_hash=canonical_hash(head),
                                    )
                                    gradients.append(info)
                                    used.add(selected["family_id"])
                                    updates += 1
                                checkpoint = shard(raw, "checkpoints", head)
                                write(
                                    "commit",
                                    f"commit/{arm}/{seed}/{block}",
                                    dict(
                                        arm=arm,
                                        seed=seed,
                                        slot=slot,
                                        block=block,
                                        before_head_hash=before_hash,
                                        head_hash=canonical_hash(head),
                                        checkpoint=checkpoint,
                                        gradients=gradients,
                                        actual_gradient_count=updates,
                                    ),
                                )
                                progress(f"block_{arm}_seed_{seed}", done, total - done)
                    done += 1
                    if (slot + 1) % 64 == 0:
                        progress(f"stream_{arm}_seed_{seed}", done, total - done)
                write(
                    "final",
                    f"final/{arm}/{seed}",
                    dict(
                        arm=arm,
                        seed=seed,
                        head_hash=canonical_hash(head),
                        checkpoint=checkpoint,
                        actual_gradient_count=updates,
                        pending_tail=min(20, len(sources)),
                    ),
                )
    finally:
        ledger.close()
    atomic_json(
        raw / "seal.json",
        dict(
            sealed=True,
            retention_read_permitted_after_seal=True,
            references=[reference(p) for p in sorted(raw.rglob("*")) if p.is_file()],
        ),
    )
    progress("benchmark_after_trajectories_sealed", done, 0)
    return reduce(raw)


def reduce(raw: Path) -> Json:
    """Replay committed states exactly once and derive only issued-state metrics.

    This reader uses recorded due feedback, never the original target vault.
    Arithmetic parity is checked against the dense equation at every update.
    """
    seal = json.loads((raw / "seal.json").read_text())
    equal(seal["sealed"], True, "unsealed_trajectory")
    for ref in seal["references"]:
        checked(ref)
    data = json.loads((raw / "inputs.json").read_text())
    vectors = [
        old.design(data["head"], r) if r["public_eligible"] else None for r in data["sources"]
    ]
    equal(json.loads((raw / "methods.json").read_text())["config"], CONFIG, "methods_drift")
    db = sqlite3.connect(f"file:{raw / 'ledger.sqlite'}?mode=ro", uri=True)
    events = [
        (seq, kind, identity, json.loads(payload))
        for seq, kind, identity, payload in db.execute(
            "SELECT seq,kind,identity,payload FROM events ORDER BY seq"
        )
    ]
    db.close()
    equal(len({i for _, _, i, _ in events}), len(events), "duplicate_commit")
    heads: Json = {}
    releases: Json = {}
    seen: Json = {}
    used: Json = {}
    buffers: Json = {}
    counts: Json = {}
    issues, feedback, gradients, buffer_rows, commits, finals, timings = [], [], [], [], [], [], []
    max_error = 0.0
    for seq, kind, identity, r in events:
        if kind == "timing":
            timings.append(dict(r, durable_commit_id=seq))
            continue
        key = f"{r['arm']}/{r['seed']}"
        if key not in heads:
            heads[key] = copy.deepcopy(data["head"])
        head = heads[key]
        eligible = releases.setdefault(key, [])
        count = counts.setdefault(key, 0)
        ids = used.setdefault(key, set())
        if kind == "issue":
            source = data["sources"][r["slot"]]
            x = vectors[r["slot"]]
            p = old.probability(head, x) if x is not None else None
            equal(
                (r["probability"], r["action"], r["head_hash"], r["family_id"]),
                (p, action(p), canonical_hash(head), source["family_id"]),
                "issued_state_drift",
            )
            seen[f"{key}/{r['slot']}"] = r
            issues.append(dict(r, durable_commit_id=seq))
        elif kind == "release":
            prior = seen[f"{key}/{r['origin_slot']}"]
            valid = r["y"] is not None and prior["probability"] is not None
            equal(
                (
                    r["due_slot"],
                    r["issued_probability"],
                    r["issued_action"],
                    r["eligibility"],
                    r["issued_decision_loss"],
                    r["issued_brier_loss"],
                    f"{key}/{r['due_slot']}" in seen,
                ),
                (
                    r["origin_slot"] + 20,
                    prior["probability"],
                    prior["action"],
                    valid,
                    loss(prior["action"], r["y"]) if valid else None,
                    (prior["probability"] - r["y"]) ** 2 if valid else None,
                    True,
                ),
                "release_order_or_loss",
            )
            feedback.append(dict(r, durable_commit_id=seq))
            if valid:
                eligible.append(r)
        elif kind == "buffer":
            pool, chosen = select(eligible, r["arm"], r["seed"], r["block"])
            expected = [
                dict(
                    family_id=z["family_id"],
                    origin_slot=z["origin_slot"],
                    release_age=r["slot"] - z["due_slot"],
                    origin_age=r["slot"] - z["origin_slot"],
                )
                for z in pool
            ]
            equal(
                (r["pool"], r["selected_ids"], r["eligible_released_count"], r["block"] * 16),
                (expected, [z["family_id"] for z in chosen], len(eligible), len(eligible)),
                "buffer_selection",
            )
            buffers[key] = chosen
            buffer_rows.append(dict(r, durable_commit_id=seq))
        elif kind == "commit":
            equal(r["before_head_hash"], canonical_hash(head), "commit_start")
            chosen = buffers[key] if count < 64 else []
            equal(len(r["gradients"]), len(chosen), "gradient_budget")
            for g, selected in zip(r["gradients"], chosen, strict=True):
                x = vectors[selected["origin_slot"]]
                before = old.coefficients(head)
                dense = before - 0.01 * (
                    (old.probability(head, x) - selected["y"]) * head["calibration"][1] * x
                    + 0.002 * before
                )
                audit = old.update(head, x, selected["y"])
                error = float(np.max(np.abs(old.coefficients(head) - dense)))
                max_error = max(max_error, error)
                equal(error <= CONFIG["sparse_dense_tolerance"], True, "sparse_dense_parity")
                equal(
                    (
                        g["family_id"],
                        g["y"],
                        g["update_index"],
                        g["after_coefficients"],
                        g["active_basis_indices"],
                        g["after_head_hash"],
                        g["repeated_id"],
                    ),
                    (
                        selected["family_id"],
                        selected["y"],
                        count,
                        audit["after_coefficients"],
                        audit["active_basis_indices"],
                        canonical_hash(head),
                        selected["family_id"] in ids,
                    ),
                    "gradient_equation",
                )
                gradients.append(
                    dict(
                        g,
                        arm=r["arm"],
                        seed=r["seed"],
                        slot=r["slot"],
                        block=r["block"],
                        durable_commit_id=seq,
                        release_age=r["slot"] - selected["due_slot"],
                        active_spline_indices=[i for i in g["active_basis_indices"] if i >= 2],
                        active_knot_operations=3 * len(g["active_basis_indices"]) + 1,
                    )
                )
                ids.add(selected["family_id"])
                count += 1
            counts[key] = count
            equal(
                (
                    r["head_hash"],
                    json.loads(checked(r["checkpoint"]).read_text()),
                    r["actual_gradient_count"],
                ),
                (canonical_hash(head), head, count),
                "commit_checkpoint",
            )
            commits.append(dict(r, durable_commit_id=seq))
        elif kind == "final":
            expected_count = (
                0 if r["arm"] == "frozen_no_write" else min(64, len(eligible) // 16 * 4)
            )
            equal(
                (
                    count,
                    r["actual_gradient_count"],
                    r["head_hash"],
                    json.loads(checked(r["checkpoint"]).read_text()),
                ),
                (expected_count, expected_count, canonical_hash(head), head),
                "final_budget_or_state",
            )
            finals.append(dict(r, durable_commit_id=seq))
    expected_keys = {f"{arm}/{seed}" for seed in data["seeds"] for arm in ARMS}
    equal({f"{r['arm']}/{r['seed']}" for r in finals}, expected_keys, "missing_final")
    labels = {(r["arm"], r["seed"], r["origin_slot"]): r for r in feedback}
    rows, budgets = [], []
    for r in issues:
        label = labels.get((r["arm"], r["seed"], r["slot"]))
        known = label is not None and label["eligibility"]
        rows.append(
            dict(
                r,
                numerator=label["issued_decision_loss"] if known else None,
                denominator=int(known),
                brier=label["issued_brier_loss"] if known else None,
                y=label["y"] if label else None,
                eligibility=known,
                censor_reason="pending_delay" if label is None else None,
                exclusion_reason=None
                if known
                else "pending_delay"
                if label is None
                else label["exclusion_reason"],
                failure_reason=None,
            )
        )
    for final in finals:
        arm, seed = final["arm"], final["seed"]
        subset = [r for r in rows if r["arm"] == arm and r["seed"] == seed]
        eligible = [r for r in subset if r["eligibility"]]
        later = [r for r in eligible if r["slot"] >= 36]
        selected = [r for r in gradients if r["arm"] == arm and r["seed"] == seed]
        budget = dict(
            intended=len(data["sources"]),
            eligible=len(eligible),
            started=len(subset),
            completed=len(subset),
            failed=0,
            censored=sum(r["censor_reason"] is not None for r in subset),
            excluded=sum(not r["eligibility"] and r["censor_reason"] is None for r in subset),
            independent=len({r["source_cluster_id"] for r in eligible}),
        )
        budgets.append(
            dict(
                budget,
                arm=arm,
                seed=seed,
                update_opportunities=len(eligible) // 16 * 4,
                actual_gradient_count=len(selected),
                matched_learning_budget=arm != "frozen_no_write",
                cap=64,
                repeated_id_count=sum(r["repeated_id"] for r in selected),
                later_cost_numerator=sum(r["numerator"] for r in later),
                later_cost_denominator=len(later),
                later_brier_numerator=sum(r["brier"] for r in later),
            )
        )
    primary = next(
        r for r in budgets if r["arm"] == "frozen_no_write" and r["seed"] == data["seeds"][0]
    )
    sample = {
        k: primary[k]
        for k in (
            "intended",
            "eligible",
            "started",
            "completed",
            "excluded",
            "failed",
            "censored",
            "independent",
        )
    }
    sample.update(seeds_are_independent=False, independent_datasets=1)
    cpu = [
        dict(
            arm=r["arm"],
            seed=r["seed"],
            update_index=r["update_index"],
            cpu_ns=r["update_cpu_ns"],
            arithmetic_ns=r["hot_update_ns"],
            bytes_written=r["bytes_written"],
            active_knot_operations=r["active_knot_operations"],
        )
        for r in gradients
    ]
    arithmetic = sum(r["arithmetic_ns"] for r in cpu)
    total = sum(r["cpu_ns"] for r in cpu) + sum(r["durable_transaction_ns"] for r in timings)
    fraction = arithmetic / total if total else 0.0
    return dict(
        rows=rows,
        issued_prediction_rows=issues,
        feedback_release_rows=feedback,
        replay_buffer_rows=buffer_rows,
        gradient_rows=gradients,
        update_budget_rows=budgets,
        checkpoint_references=list(
            {r["checkpoint"]["sha256"]: r["checkpoint"] for r in commits + finals}.values()
        ),
        final_checkpoint_rows=finals,
        durable_commit_rows=[{k: v for k, v in r.items() if k != "gradients"} for r in commits],
        cpu_update_costs=cpu,
        transaction_cost_rows=timings,
        sample_size_budget=sample,
        sparse_dense_max_error=max_error,
        journal_exactly_once=True,
        retained_labels_opened=False,
        trajectory_seal=reference(raw / "seal.json"),
        hardware_path=dict(
            current="Tier1/2 CPU and RAM",
            prospective="Rust/SIMD sparse gather/dot/update; FPGA",
            operation_count_scope="three sparse coefficient operations per active index plus global scale; dense diagnostics excluded",
            arithmetic_speedup_target=100,
            measured_speedup=False,
            observed_arithmetic_fraction=fraction,
            amdahl_transaction_inclusive_target=1 / (1 - fraction + fraction / 100),
            total_cost_ns=total,
            total_scope="update CPU plus synchronous transaction wall time; checkpoints and runner overhead excluded",
        ),
    )


def controls() -> Json:
    """An artificial boundary crossing checks action sensitivity, not benefit."""
    head = dict(parameters=[0.0] * 110, decay_scale=1.0, calibration=[0.0, 1.0])
    head["parameters"][0] = float(np.log(0.0999 / 0.9001))
    x = np.zeros(110)
    x[0] = 1
    before = action(old.probability(head, x))
    old.update(head, x, 1)
    after = action(old.probability(head, x))
    return dict(
        scope="artificial CPU control; no natural benefit credit",
        before=before,
        after=after,
        passed=before != after,
        natural_benefit=False,
    )

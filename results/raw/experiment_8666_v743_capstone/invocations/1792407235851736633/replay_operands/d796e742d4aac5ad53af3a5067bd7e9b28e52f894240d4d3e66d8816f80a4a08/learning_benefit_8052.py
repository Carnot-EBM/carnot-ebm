"""REQ-REPORT-8052: independent causal equations qualify finite exposed evidence."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import random
import sqlite3
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.evidence_features_custody_7980 import checked
from carnot.verify import learning_retention_audit_8026 as math
from carnot.verify.learning_benefit_8039 import bootstrap, controls

Json = dict[str, Any]
Array = NDArray[np.float64]
ARMS = ("unconstrained", "feedback_constrained", "frozen_no_write")
CONFIG = dict(
    seed=6978052,
    draws=10000,
    blocks=[32, 16, 64],
    margin=0.02,
    later_minimum=120,
    per_class=15,
    changed_minimum=5,
    retention_minimum=48,
    retention_per_class=8,
    brier_drift=0.01,
    cost_drift=0.02,
    budget_s=900,
    timeline=256,
    independent_streams=1,
)
BEGAN = time.monotonic()
equal = math.equal


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Expose actual work counts so a quiet child does not hide a stalled audit."""
    print(
        f"[exp8052] phase={phase} elapsed_s={time.monotonic() - BEGAN:.3f} completed={completed} pending={pending}",
        flush=True,
    )


def role(source: Json) -> str:
    """Public source identity fixes the feedback role before target access."""
    return (
        "guard"
        if int(canonical_hash(source["source_cluster_id"]).split(":")[1], 16) % 4 == 0
        else "update"
    )


def select(rows: list[Json], seed: int, block: int) -> list[Json]:
    """Recreate uniform cumulative draws without importing the learner selector."""
    ordered = sorted(rows, key=lambda r: canonical_hash(dict(seed=seed, identity=r["family_id"])))
    rng = random.Random(int(canonical_hash(dict(seed=seed, block=block)).split(":")[1], 16))
    return rng.sample(ordered, 4)


def guard(
    head: Json, theta: Array, delta: Array, initial: Array, x: Array, y: Array, arm: str
) -> Json:
    """Independent calibrated guard equations keep rejection and reset operands."""
    ready = len(y) >= 8 and min(int(sum(y == c)) for c in (0, 1)) >= 2
    diagnostics = []

    def operands(parameters: Array) -> Json:
        a, b = head["calibration"]
        ps = np.asarray(math.expit(a + b * (x @ parameters)), dtype=float).tolist()
        acts = [math.action(p) for p in ps]
        return dict(
            probabilities=ps,
            actions=acts,
            brier=float(np.mean((np.array(ps) - y) ** 2)),
            typed_cost=float(np.mean([math.cost(a, int(v)) for a, v in zip(acts, y, strict=True)])),
            false_accepts=[
                i for i, (a, v) in enumerate(zip(acts, y, strict=True)) if a == "accept" and v == 1
            ],
        )

    if ready:
        baseline = operands(initial)
        for alpha in [1, 0.5, 0.25, 0.125, 0]:
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
    alpha = 1 if arm == ARMS[0] else passing[0] if passing else 0 if not ready else None
    return dict(
        alpha=alpha,
        parameters=(initial if alpha is None else theta + alpha * delta).tolist(),
        diagnostics=diagnostics,
        reset=alpha is None,
        rejected=alpha != 1,
        status="waiting_guard" if not ready else "reset" if alpha is None else "commit",
    )


def replay(raw: Path, labels: Json, *, budget_s: float = 900) -> Json:
    """Cold-rebuild every event from public vectors and original released labels."""
    progress("independent_head_load_before")
    data = json.loads((raw / "inputs.json").read_text())
    methods = json.loads((raw / "methods.json").read_text())
    sources, head, seeds = data["sources"], data["head"], data["seeds"]
    for field, expected in dict(
        delay=20,
        block_update_releases=16,
        step=0.01,
        ridge=0.001,
        cap_steps=64,
        guard_gradients=False,
        terminal_flush=False,
    ).items():
        equal("methods." + field, expected, methods["config"][field])
    vectors = [math.design(head, r) if r["public_eligible"] else None for r in sources]
    initial = np.asarray(head["parameters"]) * head["decay_scale"]
    progress("independent_head_load_after")
    states, rows, independent, checks, first, counts = {}, {}, [], [], [], []
    deadline = time.monotonic() + budget_s
    for seed in seeds:
        db = sqlite3.connect(f"file:{raw / f'seed-{seed}' / 'ledger.sqlite'}?mode=ro", uri=True)
        events = db.execute("select seq,kind,identity,payload from events order by seq").fetchall()
        db.close()
        for arm in ARMS:
            key = f"{arm}/{seed}"
            state = copy.deepcopy(head)
            issues, released, guards = [], [], []
            attempts = releases = acceptance_count = checkpoint_count = 0
            pending = None
            seen = set()
            final = False
            for seq, kind, identity, text in events:
                r = json.loads(text)
                if r["arm"] != arm:
                    continue
                equal("budget", True, time.monotonic() <= deadline)
                equal("seed", seed, r["seed"])
                equal("duplicate_identity", False, identity in seen)
                seen.add(identity)
                if kind == "issue":
                    slot = r["slot"]
                    equal("issue_order", len(issues), slot)
                    equal("pending_decision", None, pending)
                    equal("issue_identity", f"issue/{key}/{slot}", identity)
                    equal("issue_source", sources[slot]["family_id"], r["family_id"])
                    p = (
                        math.probability(state, vectors[slot])
                        if vectors[slot] is not None
                        else None
                    )
                    equal("probability", p, r["probability"])
                    equal("action", math.action(p), r["action"])
                    equal("head_hash", canonical_hash(state), r["head_hash"])
                    issues.append(r)
                elif kind == "release":
                    origin = r["origin_slot"]
                    equal("release_order", releases, origin)
                    equal("release_identity", f"release/{key}/{origin}", identity)
                    equal("release_due", origin + 20, r["due_slot"])
                    equal("release_clock", origin + 20, len(issues) - 1)
                    equal("release_source", sources[origin]["family_id"], r["family_id"])
                    y = labels[r["family_id"]]
                    equal("released_target", y, r["y"])
                    equal("target_contract", True, y is None or (type(y) is int and y in (0, 1)))
                    equal("release_role", role(sources[origin]), r["feedback_role"])
                    valid = y is not None and vectors[origin] is not None
                    equal("release_eligible", valid, r["eligible"])
                    releases += 1
                    if valid:
                        (guards if role(sources[origin]) == "guard" else released).append(r)
                    rows[(key, origin)] = dict(
                        issues[origin],
                        source_cluster_id=sources[origin]["source_cluster_id"],
                        y=y,
                        eligible=valid,
                        feedback_role=role(sources[origin]),
                        typed_cost=math.cost(issues[origin]["action"], y) if valid else None,
                        brier=(issues[origin]["probability"] - y) ** 2 if valid else None,
                        false_accept=int(issues[origin]["action"] == "accept" and y == 1)
                        if valid
                        else None,
                        status="completed" if valid else "excluded",
                        numerator=int(valid),
                        denominator=1,
                    )
                elif kind == "gradient":
                    equal("adaptive_arm", True, arm != ARMS[2])
                    equal("gradient_identity", f"gradient/{key}/{attempts}", identity)
                    equal("gradient_budget", True, attempts < 64 and len(released) % 16 == 0)
                    chosen = select(released, seed, len(released) // 16)[attempts % 4]
                    equal(
                        "gradient_source",
                        (chosen["family_id"], chosen["origin_slot"], chosen["y"]),
                        (r["family_id"], r["origin_slot"], r["y"]),
                    )
                    equal("gradient_clock", len(issues) - 1, r["slot"])
                    equal("gradient_before", canonical_hash(state), r["before_hash"])
                    candidate = copy.deepcopy(state)
                    math.update(candidate, vectors[r["origin_slot"]], r["y"])
                    proposed = np.asarray(candidate["parameters"]) * candidate["decay_scale"]
                    equal("candidate_coefficients", proposed.tolist(), r["proposed_coefficients"])
                    before = np.asarray(state["parameters"]) * state["decay_scale"]
                    dense = before - 0.01 * (
                        (math.probability(state, vectors[r["origin_slot"]]) - r["y"])
                        * state["calibration"][1]
                        * vectors[r["origin_slot"]]
                        + 0.002 * before
                    )
                    equal("dense_gradient", True, bool(np.max(np.abs(dense - proposed)) <= 1e-14))
                    pending = proposed - before
                    independent.append(
                        dict(
                            arm=arm,
                            seed=seed,
                            slot=r["slot"],
                            source_cluster_id=sources[r["origin_slot"]]["source_cluster_id"],
                            origin_slot=r["origin_slot"],
                            candidate_hash=canonical_hash(candidate),
                            coefficient_error=float(np.max(np.abs(dense - proposed))),
                            numerator=1,
                            denominator=1,
                        )
                    )
                    if attempts == 0:
                        first.append(r["slot"])
                    attempts += 1
                elif kind == "acceptance":
                    acceptance_count += 1
                    equal("acceptance_clock", len(issues) - 1, r["slot"])
                    equal(
                        "acceptance_identity",
                        f"acceptance/{key}/{r['slot']}/{r['reason']}",
                        identity,
                    )
                    delta = (
                        pending if r["reason"].startswith("gradient/") else np.zeros_like(initial)
                    )
                    equal("candidate_present", True, delta is not None)
                    current = np.asarray(state["parameters"]) * state["decay_scale"]
                    x = np.array([vectors[z["origin_slot"]] for z in guards])
                    y = np.array([z["y"] for z in guards])
                    expected = guard(state, current, delta, initial, x, y, arm)
                    equal("guard_ids", [z["family_id"] for z in guards], r["guard_ids"])
                    equal("guard_before", current.tolist(), r["before_coefficients"])
                    equal("guard_proposed", (current + delta).tolist(), r["proposed_coefficients"])
                    equal("alpha_diagnostics", expected, {k: r[k] for k in expected})
                    state.update(parameters=expected["parameters"], decay_scale=1.0)
                    equal("acceptance_hash", canonical_hash(state), r["head_hash"])
                    pending = None
                    checks.append(
                        dict(
                            arm=arm,
                            seed=seed,
                            slot=r["slot"],
                            alpha=expected["alpha"],
                            reset=expected["reset"],
                            rejected=expected["rejected"],
                            reason=r["reason"],
                            head_hash=r["head_hash"],
                            guard_count=len(guards),
                            numerator=1,
                            denominator=1,
                        )
                    )
                elif kind == "checkpoint":
                    checkpoint_count += 1
                    ref = r["head"]
                    local = raw / "heads" / Path(ref["path"]).name
                    observed = json.loads(
                        checked(
                            dict(ref, path=str(local if local.exists() else Path(ref["path"])))
                        ).read_text()
                    )
                    equal("checkpoint", state, observed)
                    equal("checkpoint_guard", [z["family_id"] for z in guards], r["guard_ids"])
                    equal("checkpoint_update", [z["family_id"] for z in released], r["update_ids"])
                    part_ref = r["partition"]
                    part_local = raw / "partitions" / Path(part_ref["path"]).name
                    part = json.loads(checked(dict(part_ref, path=str(part_local))).read_text())[
                        "rows"
                    ]
                    equal(
                        "checkpoint_roles",
                        [role(s) for s in sources],
                        [z["feedback_role"] for z in part],
                    )
                    final = identity == f"final/{key}" or final
                else:
                    raise ValueError("unknown_event:" + kind)
            equal("final_checkpoint", True, final)
            equal("issue_count", len(sources), len(issues))
            equal("release_count", max(0, len(sources) - 20), releases)
            equal(
                "attempt_count", 0 if arm == ARMS[2] else min(64, len(released) // 16 * 4), attempts
            )
            equal("unfinished_candidate", None, pending)
            equal(
                "acceptance_count",
                0 if arm == ARMS[2] else attempts + len(guards),
                acceptance_count,
            )
            equal("checkpoint_count", len(released) // 16 + 1, checkpoint_count)
            for r in issues[releases:]:
                rows[(key, r["slot"])] = dict(
                    r,
                    source_cluster_id=sources[r["slot"]]["source_cluster_id"],
                    y=None,
                    eligible=False,
                    feedback_role=role(sources[r["slot"]]),
                    typed_cost=None,
                    brier=None,
                    false_accept=None,
                    status="censored",
                    numerator=0,
                    denominator=1,
                )
            states[key] = state
            counts.append(
                dict(
                    arm=arm,
                    seed=seed,
                    attempt_count=attempts,
                    issue_count=len(issues),
                    release_count=releases,
                    guard_count=len(guards),
                    update_count=len(released),
                )
            )
            progress("replayed_arm", len(states), len(seeds) * 3 - len(states))
    start = max(first or [len(sources)])
    output = [
        dict(
            r,
            post_first_update=r["slot"] > start,
            exclusion_reason=None if r["eligible"] else r["status"],
        )
        for r in rows.values()
    ]
    first_rows = [r for r in output if r["arm"] == ARMS[0] and r["seed"] == seeds[0]]
    sample = dict(
        intended=len(sources),
        eligible=sum(r["eligible"] for r in first_rows),
        completed=sum(r["eligible"] for r in first_rows),
        excluded=sum(r["status"] == "excluded" for r in first_rows),
        censored=sum(r["status"] == "censored" for r in first_rows),
        failed=0,
        independent=len({r["source_cluster_id"] for r in first_rows if r["eligible"]}),
        seeds_are_independent=False,
        independent_datasets=1,
    )
    return dict(
        rows=output,
        independent_reduction_rows=independent,
        guard_reconstruction_rows=checks,
        final_states=states,
        initial_head=head,
        count_rows=counts,
        sample_size_budget=sample,
        numerical_agreement=dict(
            candidate_count=len(independent),
            issue_count=len(output),
            checkpoint_agreement=True,
            probability_agreement=True,
            head_hash_agreement=True,
        ),
    )


def retention(bundle: Json, replayed: Json, seal: Path, *, cold: bool = False) -> list[Json]:
    """Seal old-task predictions and terminal states before releasing vault targets."""
    predictions = []
    for key, head in replayed["final_states"].items():
        arm, seed = key.split("/")
        for r in bundle["retention_public"]:
            x = math.design(head, r) if r["public_eligible"] else None
            p = math.probability(head, x) if x is not None else None
            p0 = math.probability(replayed["initial_head"], x) if x is not None else None
            predictions.append(
                dict(
                    arm=arm,
                    seed=int(seed),
                    source_cluster_id=r["source_cluster_id"],
                    family_id=r["family_id"],
                    slot=r["slot"],
                    probability=p,
                    initial_probability=p0,
                    action=math.action(p),
                    initial_action=math.action(p0),
                )
            )
    value = dict(
        predictions=predictions,
        final_head_hashes={k: canonical_hash(h) for k, h in replayed["final_states"].items()},
        retention_labels_opened=False,
    )
    if cold:
        equal("retention_seal", value, json.loads(seal.read_text()))
    else:
        atomic_json(seal, value)
    progress("terminal_heads_and_retention_predictions_sealed", len(predictions), 0)
    labels = json.loads(checked(bundle["retention_target"]).read_text())["rows"]
    target = {r["family_id"]: r["eligible_y"] for r in labels}
    equal("retention_ids", sorted({r["family_id"] for r in predictions}), sorted(target))
    rows = []
    for r in predictions:
        y = target[r["family_id"]]
        equal("retention_target", True, y is None or (type(y) is int and y in (0, 1)))
        valid = y is not None and r["probability"] is not None
        cost = math.cost(r["action"], y) if valid else None
        brier = (r["probability"] - y) ** 2 if valid else None
        rows.append(
            dict(
                r,
                y=y,
                eligible=valid,
                typed_cost=cost,
                brier=brier,
                cost_drift=cost - math.cost(r["initial_action"], y) if valid else None,
                brier_drift=brier - (r["initial_probability"] - y) ** 2 if valid else None,
                numerator=int(valid),
                denominator=1,
                exclusion_reason=None if valid else "unknown_or_public_unavailable",
            )
        )
    return rows


def compare(rows: list[Json], retained: list[Json]) -> Json:
    """Pair all arms on original slots; seed repetitions add no independent sources."""
    seeds = sorted({r["seed"] for r in rows})
    indexed = {(r["arm"], r["seed"], r["slot"]): r for r in rows}
    slots = range(CONFIG["timeline"])
    eligible = []
    for slot in slots:
        group = [indexed.get((arm, seed, slot)) for arm in ARMS for seed in seeds]
        if group and all(
            r and r["eligible"] and r["post_first_update"] and r["feedback_role"] == "update"
            for r in group
        ):
            eligible.append(slot)
    unique = {
        indexed[(ARMS[0], seeds[0], i)]["source_cluster_id"]: indexed[(ARMS[0], seeds[0], i)]
        for i in eligible
    }
    support = len(unique) >= 120 and all(
        sum(r["y"] == y for r in unique.values()) >= 15 for y in (0, 1)
    )
    retained_unique = {r["source_cluster_id"]: r for r in retained if r["eligible"]}
    retention_support = len(retained_unique) >= 48 and all(
        sum(r["y"] == y for r in retained_unique.values()) >= 8 for y in (0, 1)
    )
    drift = []
    for arm in ARMS[:2]:
        for seed in seeds:
            group = [r for r in retained if r["arm"] == arm and r["seed"] == seed and r["eligible"]]
            c = sum(r["cost_drift"] for r in group)
            b = sum(r["brier_drift"] for r in group)
            n = len(group)
            drift.append(
                dict(
                    arm=arm,
                    seed=seed,
                    cost_numerator=c,
                    brier_numerator=b,
                    denominator=n,
                    cost_drift=c / n if n else None,
                    brier_drift=b / n if n else None,
                    passed=bool(n and c / n <= 0.02 and b / n <= 0.01),
                )
            )
    retention_passed = retention_support and all(r["passed"] for r in drift)
    later, false_rows, hypotheses = [], [], []
    for comparator in (ARMS[0], ARMS[2]):
        diff = [float("nan")] * 256
        changed = set()
        for slot in eligible:
            pairs = [(indexed[(ARMS[1], s, slot)], indexed[(comparator, s, slot)]) for s in seeds]
            diff[slot] = float(np.mean([b["typed_cost"] - a["typed_cost"] for a, b in pairs]))
            if diff[slot] > 0 and any(a["action"] != b["action"] for a, b in pairs):
                changed.add(pairs[0][0]["source_cluster_id"])
            for a, b in pairs:
                later.append(
                    dict(
                        a,
                        comparator=comparator,
                        paired_gain=b["typed_cost"] - a["typed_cost"],
                        comparator_cost=b["typed_cost"],
                    )
                )
        for seed in seeds:
            treatment = [indexed[(ARMS[1], seed, i)] for i in eligible]
            baseline = [indexed[(comparator, seed, i)] for i in eligible]
            t = sum(r["false_accept"] for r in treatment)
            b = sum(r["false_accept"] for r in baseline)
            g = sum(
                y["typed_cost"] - x["typed_cost"] for x, y in zip(treatment, baseline, strict=True)
            )
            false_rows.append(
                dict(
                    seed=seed,
                    treatment=ARMS[1],
                    comparator=comparator,
                    treatment_numerator=t,
                    comparator_numerator=b,
                    denominator=len(eligible),
                    difference=t - b,
                    passed=t <= b,
                    gain_numerator=g,
                    gain=g / len(eligible) if eligible else None,
                )
            )
        tests = [bootstrap(diff, n) for n in CONFIG["blocks"]]
        gates = dict(
            support=support,
            retention=retention_passed,
            beneficial_changes=len(changed) >= 5,
            no_added_false_accepts=all(
                r["passed"] for r in false_rows if r["comparator"] == comparator
            ),
            gain_margin=tests[0]["gain"] is not None and tests[0]["gain"] >= 0.02,
            per_seed_gain=all(
                r["gain"] is not None and r["gain"] >= 0.02
                for r in false_rows
                if r["comparator"] == comparator
            ),
            margin_test=tests[0]["raw_p"] < 0.05
            and tests[0]["lower"] is not None
            and tests[0]["lower"] > 0.02,
        )
        hypotheses.append(
            dict(
                tests[0],
                hypothesis="H3" if comparator == ARMS[0] else "secondary_frozen",
                comparator=comparator,
                treatment=ARMS[1],
                gates=gates,
                beneficial_changed_groups=len(changed),
                block_sensitivity=tests[1:],
                local_passed=all(gates.values()),
                capstone_family_p=tests[0]["raw_p"] if all(gates.values()) else 1.0,
                capstone_family=["H1", "H2", "H3"],
                family_credit=False,
            )
        )
    safety = all(r["passed"] for r in false_rows) and retention_passed
    if not safety:
        hypotheses[0]["capstone_family_p"] = 1.0
    secondary = []
    summaries = []
    for arm in ARMS:
        for seed in seeds:
            for condition in ("later_update_role", "later_guard_role"):
                group = [
                    r
                    for r in rows
                    if r["arm"] == arm
                    and r["seed"] == seed
                    and r["eligible"]
                    and r["post_first_update"]
                    and r["feedback_role"]
                    == ("guard" if condition == "later_guard_role" else "update")
                ]
                n = len(group)
                c = sum(r["typed_cost"] for r in group)
                b = sum(r["brier"] for r in group)
                summary = dict(
                    arm=arm,
                    seed=seed,
                    condition=condition,
                    numerator=n,
                    denominator=256,
                    typed_cost_numerator=c,
                    brier_numerator=b,
                    metric_denominator=n,
                    typed_cost=c / n if n else None,
                    brier=b / n if n else None,
                    false_accepts=sum(r["false_accept"] for r in group),
                    independent_count=len({r["source_cluster_id"] for r in group}),
                )
                summaries.append(summary)
                if condition == "later_guard_role":
                    secondary.append(summary)
    return dict(
        primary_hypothesis_results=hypotheses,
        later_source_rows=later,
        per_seed_false_accept_rows=false_rows,
        retention_drift_rows=drift,
        guard_role_secondary_results=secondary,
        condition_metric_rows=summaries,
        support_passed=support,
        retention_support_passed=retention_support,
        retention_passed=retention_passed,
        later_support=dict(
            independent_count=len(unique),
            class_counts={str(y): sum(r["y"] == y for r in unique.values()) for y in (0, 1)},
            minimum=120,
            per_class=15,
            eligible_slots=eligible,
        ),
        retention_support=dict(
            independent_count=len(retained_unique),
            class_counts={
                str(y): sum(r["y"] == y for r in retained_unique.values()) for y in (0, 1)
            },
            minimum=48,
            per_class=8,
        ),
        benefit_ready_score=int(bool(support and safety and hypotheses[0]["local_passed"])),
    )

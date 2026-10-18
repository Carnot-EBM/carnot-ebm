"""REQ-REPORT-8065: independent arithmetic on primitive chronological journals.

Issued states, rather than final-head rescoring, determine every later outcome.
The reader imports no learner transition, optimizer, guard or role selector.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np
from numpy.typing import NDArray

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import learning_retention_audit_8026 as math

Json = dict[str, Any]
Array = NDArray[np.float64]
ARMS = ["frozen", "unconditional", "reused_guard", "fresh_admission"]
FIELDS = dict(
    issue="issued_prediction_rows",
    release="feedback_release_rows",
    candidate="candidate_commit_rows",
    consume="admission_consumption_rows",
    alpha="alpha_check_rows",
    commit="durable_commit_rows",
    pending="pending_update_rows",
    budget="update_budget_rows",
    seal="final_head_seals",
)
CONFIG = dict(seed=6988065, draws=10000, blocks=[32, 16, 64], margin=0.02, timeline=256)


def role(row: Json) -> str:
    """Literal public source bytes assign a role before targets are available."""
    return (
        "admission"
        if int(hashlib.sha256(row["source_cluster_id"].encode()).hexdigest(), 16) % 4 == 0
        else "update"
    )


def adequate(y: Array, minimum: int) -> bool:
    """A missing class defers the attempt without outcome-selected replacements."""
    return len(y) >= minimum and all(int(sum(y == c)) >= 2 for c in (0, 1))


def operands(head: Json, theta: Array, x: Array, y: Array) -> Json:
    """Keep false accepts as row identities so averages cannot hide added harm."""
    a, b = head["calibration"]
    ps = math.expit(a + b * (x @ theta))
    acts = [math.action(float(p)) for p in ps]
    return dict(
        probabilities=ps.tolist(),
        brier=float(np.mean((ps - y) ** 2)),
        typed_cost=float(np.mean([math.cost(v, int(t)) for v, t in zip(acts, y, strict=True)])),
        false_accepts=[
            i for i, (v, t) in enumerate(zip(acts, y, strict=True)) if v == "accept" and t == 1
        ],
    )


def guard(
    head: Json, current: Array, candidate: Array, initial: Array, x: Array, y: Array
) -> tuple[list[Json], float | None]:
    """Rebuild all alpha operands independently, including rejection and zero."""
    checks = []
    if not adequate(y, 4):
        return checks, None
    original, incumbent = operands(head, initial, x, y), operands(head, current, x, y)
    for alpha in [1, 0.5, 0.25, 0.125, 0]:
        proposed = operands(head, current + alpha * (candidate - current), x, y)
        reasons = []
        for base, name, margins in [
            (original, "initial", (0.01, 0.02)),
            (incumbent, "incumbent", (0.0, 0.0)),
        ]:
            for metric, margin in zip(("brier", "typed_cost"), margins, strict=True):
                if proposed[metric] > base[metric] + margin:
                    reasons.append(name + "." + metric)
            if set(proposed["false_accepts"]) - set(base["false_accepts"]):
                reasons.append(name + ".false_accept")
        checks.append(
            dict(
                alpha=alpha,
                initial=original,
                incumbent=incumbent,
                candidate=proposed,
                passed=not reasons,
                reasons=reasons,
                numerator=len(y),
                denominator=len(y),
            )
        )
    passing = [r["alpha"] for r in checks if r["passed"]]
    return checks, float(passing[0]) if passing else None


def reconstruct(raw: Path, targets: Json) -> Json:
    """Read every journal in order and require the independent transition bytes."""
    data = json.loads((raw / "inputs.json").read_text())
    head, sources = data["head"], data["sources"]
    x = np.asarray(
        [
            math.design(head, r) if r["public_eligible"] else np.zeros(len(head["parameters"]))
            for r in sources
        ]
    )
    result: Json = {v: [] for v in FIELDS.values()}
    initial = np.asarray(head["parameters"], dtype=float)
    a, b = head["calibration"]
    for seed in data["seeds"]:
        db = sqlite3.connect(f"file:{raw / f'seed-{seed}' / 'ledger.sqlite'}?mode=ro", uri=True)
        events = db.execute("SELECT seq,kind,payload FROM events ORDER BY seq").fetchall()
        db.close()
        index = 0
        states = {arm: initial.copy() for arm in ARMS}
        released: dict[int, Any] = {}
        updates: list[int] = []
        admissions: list[int] = []
        consumed: set[int] = set()
        pending: Json = {}
        counts = {arm: 0 for arm in ARMS}

        def emit(kind: str, **row: Any) -> None:
            nonlocal index
            expected = dict(seed=seed, **row)
            seq, observed_kind, payload = events[index]
            math.equal(
                f"event.{seed}.{index}",
                (index, kind, expected),
                (seq, observed_kind, json.loads(payload)),
            )
            result[FIELDS[kind]].append(deepcopy(expected))
            index += 1

        def eligible(i: int) -> bool:
            return bool(sources[i]["public_eligible"] and sources[i].get("eligible", True))

        def ys(ids: list[int]) -> Array:
            return np.asarray([released[i] for i in ids], dtype=float)

        for slot, source in enumerate(sources):
            for arm in ARMS:
                p = (
                    float(math.expit(a + b * float(x[slot] @ states[arm])))
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
                if not adequate(ys(ids), 16):
                    emit(
                        "pending",
                        slot=slot,
                        arm="shared",
                        status="deferred",
                        reason="update_support",
                    )
                else:
                    candidates = {}
                    for arm in ARMS[1:]:
                        theta = states[arm].copy()
                        for _ in range(4):
                            theta -= 0.01 * (
                                b
                                * (x[ids].T @ (math.expit(a + b * (x[ids] @ theta)) - ys(ids)))
                                / len(ids)
                                + 0.002 * theta
                            )
                        candidates[arm] = theta.tolist()
                        counts[arm] += 4
                    fresh = [
                        i
                        for i, r in enumerate(sources)
                        if i + 20 > slot
                        and eligible(i)
                        and role(r) == "admission"
                        and i not in consumed
                    ][:12]
                    pending = dict(
                        slot=slot,
                        candidates=candidates,
                        incumbents={arm: states[arm].tolist() for arm in ARMS},
                        candidate_hashes={arm: canonical_hash(t) for arm, t in candidates.items()},
                        incumbent_hashes={
                            arm: canonical_hash(states[arm].tolist()) for arm in ARMS
                        },
                        update_slots=ids,
                        reused_slots=list(admissions),
                        fresh_slots=fresh,
                        observed=[],
                    )
                    emit("candidate", arm="shared", **pending)
            origin = slot - 20
            if origin < 0:
                continue
            y = targets[sources[origin]["family_id"]]
            math.equal("label_contract", True, y is None or type(y) is int and y in (0, 1))
            released[origin] = y
            valid = eligible(origin) and y is not None
            feedback_role = role(sources[origin])
            emit(
                "release",
                slot=origin,
                release_slot=slot,
                source=sources[origin]["source_cluster_id"],
                family_id=sources[origin]["family_id"],
                arm="shared",
                y=y,
                eligible=valid,
                feedback_role=feedback_role,
            )
            if valid:
                (admissions if feedback_role == "admission" else updates).append(origin)
            if pending and origin in pending["fresh_slots"] and valid:
                math.equal("admission_reuse", False, origin in consumed)
                consumed.add(origin)
                pending["observed"].append(origin)
                emit(
                    "consume",
                    slot=origin,
                    release_slot=slot,
                    arm="fresh_admission",
                    candidate_slot=pending["slot"],
                    source=sources[origin]["source_cluster_id"],
                    y=y,
                )
            if pending and len(pending["observed"]) == 12:
                fresh, reused = pending["fresh_slots"], pending["reused_slots"]
                ready = adequate(ys(fresh), 4) and adequate(ys(reused), 4)
                for arm in ARMS:
                    alpha: float | None = 1.0 if arm == "unconditional" else None
                    if arm in ARMS[2:] and ready:
                        ids = reused if arm == "reused_guard" else fresh
                        checks, alpha = guard(
                            head,
                            states[arm],
                            np.asarray(pending["candidates"][arm]),
                            initial,
                            x[ids],
                            ys(ids),
                        )
                        for check in checks:
                            emit(
                                "alpha",
                                slot=slot,
                                candidate_slot=pending["slot"],
                                arm=arm,
                                guard_slots=ids,
                                **check,
                            )
                    if alpha is not None:
                        states[arm] += alpha * (
                            np.asarray(pending["candidates"][arm]) - states[arm]
                        )
                    emit(
                        "commit",
                        slot=slot,
                        candidate_slot=pending["slot"],
                        arm=arm,
                        alpha=alpha,
                        status="frozen"
                        if arm == "frozen"
                        else "deferred"
                        if alpha is None
                        else "zero"
                        if alpha == 0
                        else "accepted",
                        parameters=states[arm].tolist(),
                        head_hash=canonical_hash(states[arm].tolist()),
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
                gradients=counts[arm],
                cap=12,
                numerator=counts[arm],
                denominator=12,
            )
            emit(
                "seal",
                arm=arm,
                parameters=states[arm].tolist(),
                head_hash=canonical_hash(states[arm].tolist()),
            )
        math.equal("complete_journal", len(events), index)
        print(f"[exp8065] independent seed={seed} events={index}", flush=True)
    feedback = {
        r["slot"]: r for r in result["feedback_release_rows"] if r["seed"] == data["seeds"][0]
    }
    result["rows"] = []
    for r in result["issued_prediction_rows"]:
        release = feedback.get(r["slot"])
        valid = bool(release and release["eligible"])
        y = release["y"] if valid else None
        result["rows"].append(
            dict(
                r,
                unit=f"stream/{r['slot']}",
                numerator=math.cost(r["action"], y) if valid else None,
                denominator=int(valid),
                brier=(r["probability"] - y) ** 2 if valid else None,
                y=y,
                status="completed" if valid else "censored" if release is None else "excluded",
                exclusion_reason=None
                if valid
                else "unreleased_tail"
                if release is None
                else "ineligible_complete_target",
            )
        )
    clocks = {
        seed: min(
            (r["slot"] for r in result["durable_commit_rows"] if r["seed"] == seed), default=256
        )
        for seed in data["seeds"]
    }
    cutoff = max(clocks.values())
    result["later_source_rows"] = [
        r
        for r in result["rows"]
        if r["denominator"]
        and r["slot"] > cutoff
        and feedback[r["slot"]]["feedback_role"] == "update"
    ]
    result["first_shared_opportunity_slot"] = cutoff
    result["admission_reuse_count"] = 0
    return result


def retention(data: Json, seals: list[Json], raw: Path, targets: Any) -> list[Json]:
    """Persist all predictions before a separate target accessor is invoked."""
    predictions = []
    for seal in seals:
        h = dict(data["head"], parameters=seal["parameters"], decay_scale=1.0)
        for r in data["retention"]:
            p = math.probability(h, math.design(h, r)) if r["public_eligible"] else None
            predictions.append(
                dict(
                    seed=seal["seed"],
                    arm=seal["arm"],
                    slot=r["slot"],
                    source=r["source_cluster_id"],
                    family_id=r["family_id"],
                    probability=p,
                    action=math.action(p),
                    head_hash=seal["head_hash"],
                )
            )
    atomic_json(
        raw / "retention_prediction_seal.json",
        dict(rows=predictions, final_head_seals=seals, labels_opened=False),
    )
    labels = targets()
    rows = []
    for r in predictions:
        y = labels[r["family_id"]]
        valid = y is not None and r["probability"] is not None
        rows.append(
            dict(
                r,
                y=y,
                unit=f"retention/{r['slot']}",
                numerator=math.cost(r["action"], y) if valid else None,
                denominator=int(valid),
                brier=(r["probability"] - y) ** 2 if valid else None,
                status="completed" if valid else "excluded",
                exclusion_reason=None if valid else "ineligible_complete_target",
            )
        )
    return rows


def bootstrap(diff: list[float], block: int) -> Json:
    """Invert a nonzero-margin test without compressing missing timeline slots."""
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
    mean = float(np.nanmean(x))
    errors = draws - mean
    result.update(
        gain=mean,
        raw_p=float((1 + sum(errors >= mean - 0.02 - 1e-15)) / (1 + len(errors))),
        completed_draws=len(errors),
        censored_draws=10000 - len(errors),
        lower=mean - float(np.quantile(errors, 0.95)),
        interval=(mean - np.quantile(errors, [0.975, 0.025])).tolist(),
    )
    return result


def comparisons(rows: list[Json], retained: list[Json]) -> Json:
    """Pair complete source/seed/arm cells before counting independent support."""
    seeds = sorted({r["seed"] for r in rows})
    cells = {(r["source"], r["seed"], r["arm"]): r for r in rows if r["denominator"]}
    sources = sorted({r["source"] for r in rows})
    common = (
        [s for s in sources if all((s, seed, arm) in cells for seed in seeds for arm in ARMS)]
        if seeds
        else []
    )
    diff = [float("nan")] * 256
    changed = 0
    costs = {arm: [] for arm in ARMS}
    classes = [0, 0]
    for source in common:
        group = {arm: [cells[source, seed, arm] for seed in seeds] for arm in ARMS}
        first = group["frozen"][0]
        classes[first["y"]] += 1
        means = {
            arm: float(np.mean([r["numerator"] / r["denominator"] for r in group[arm]]))
            for arm in ARMS
        }
        gain = means["reused_guard"] - means["fresh_admission"]
        diff[first["slot"]] = gain
        changed += int(
            gain > 0
            and any(
                f["action"] != r["action"]
                for f, r in zip(group["fresh_admission"], group["reused_guard"], strict=True)
            )
        )
        for arm in ARMS:
            costs[arm].append(means[arm])
    false_accepts = [
        dict(
            seed=seed,
            arm=arm,
            numerator=sum(
                cells[s, seed, arm]["action"] == "accept" and cells[s, seed, arm]["y"] == 1
                for s in common
            ),
            denominator=len(common),
        )
        for seed in seeds
        for arm in ARMS
    ]
    fa = {(r["seed"], r["arm"]): r["numerator"] for r in false_accepts}
    false_accept_safe = all(
        fa[seed, "fresh_admission"] <= fa[seed, arm] for seed in seeds for arm in ARMS[:3]
    )
    noninferiority = {
        arm: (float(np.mean(costs["fresh_admission"])) - float(np.mean(costs[arm])))
        if common
        else None
        for arm in ARMS[:2]
    }
    rcells = {(r["source"], r["seed"], r["arm"]): r for r in retained if r["denominator"]}
    retained_sources = sorted({r["source"] for r in retained})
    retention_common = (
        [
            s
            for s in retained_sources
            if all((s, seed, arm) in rcells for seed in seeds for arm in ["frozen", *ARMS[2:]])
        ]
        if seeds
        else []
    )
    retention_classes = (
        [sum(rcells[s, seeds[0], "frozen"]["y"] == c for s in retention_common) for c in (0, 1)]
        if seeds
        else [0, 0]
    )
    retention_checks = []
    for arm in ARMS[2:]:
        drifts = {}
        for metric in ("brier", "numerator"):
            drifts[metric] = (
                float(
                    np.mean(
                        [
                            rcells[s, seed, arm][metric] - rcells[s, seed, "frozen"][metric]
                            for s in retention_common
                            for seed in seeds
                        ]
                    )
                )
                if retention_common
                else None
            )
        retention_checks.append(
            dict(
                arm=arm,
                brier_drift=drifts["brier"],
                cost_drift=drifts["numerator"],
                passed=bool(
                    retention_common and drifts["brier"] <= 0.01 and drifts["numerator"] <= 0.02
                ),
            )
        )
    support = (
        len(common) >= 80
        and min(classes) >= 10
        and len(retention_common) >= 48
        and min(retention_classes) >= 8
    )
    safety = (
        false_accept_safe
        and all(v is not None and v <= 0.02 for v in noninferiority.values())
        and all(r["passed"] for r in retention_checks)
    )
    tests = [bootstrap(diff, block) for block in CONFIG["blocks"]]
    benefit = (
        support
        and safety
        and changed >= 5
        and tests[0]["gain"] is not None
        and tests[0]["gain"] >= 0.02
        and tests[0]["raw_p"] < 0.05
        and tests[0]["lower"] > 0.02
    )
    h = dict(
        hypothesis="H3",
        comparison="fresh_admission versus reused_guard",
        support_count=len(common),
        class_counts=classes,
        retention_support_count=len(retention_common),
        retention_class_counts=retention_classes,
        support_passed=support,
        safety_passed=safety,
        beneficial_changed_sources=changed,
        noninferiority_cost_differences=noninferiority,
        per_seed_false_accept_safe=false_accept_safe,
        retention_checks=retention_checks,
        tests=tests,
        qualified_benefit=benefit,
        capstone_family_p=tests[0]["raw_p"] if support and safety else 1.0,
        multiplicity="Holm .05 over exactly H1/H2/H3 in Exp8069; no local family substitution",
        uncertainty_scope="Conditional historically exposed single development trajectory; seeds are averaged within sources",
    )
    return dict(primary_hypothesis_results=[h], per_seed_false_accept_rows=false_accepts)

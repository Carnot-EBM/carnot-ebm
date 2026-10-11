"""REQ-VERIFY-8351: independently reduce exposed, sealed decision evidence.

Small numeric coefficient changes do not count as learning benefit. Costs are
computed on the original source roster, including every missing-input escalation.
"""

from __future__ import annotations

from copy import deepcopy
import math
import random
from typing import Any, cast

import numpy as np

from carnot.reporting.current_work_receipt import canonical_hash
from carnot.verify import continuous_local_learning_8348 as original
from carnot.verify.local_update_isolation_8306 import scalar_design
from carnot.experiment_7425_v651_spline_prototype import sigmoid
from carnot.verify.sentence_spline_fit_8334 import action
from carnot.verify.static_benefit_audit_8350 import cost

Json = dict[str, Any]
ARMS = original.ARMS
WINDOWS = [0, 32, 64, 96]
CONFIG = dict(
    seed=7178312,
    draws=10000,
    block_length=8,
    alpha=0.025,
    intended=88,
    minimum_sources=64,
    minimum_per_label=8,
    mean_gain_min=0.02,
    retention_minimum_sources=20,
    retention_minimum_per_label=4,
    retention_cost_degradation_max=0.02,
    retention_brier_degradation_max=0.01,
)
run, numeric_proof = original.run, original.numeric_proof


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual completed counts because a silent child can conceal a stall."""
    print(f"[exp8351] phase={phase} completed={completed} pending={pending}", flush=True)


def scalar(
    head: Json, x: list[float] | None, y: int | None = None, arm: str = "frozen_spline"
) -> Json:
    """Independent recursion checks deployed temperature, clipping and frozen terms."""
    c = list(head["coefficients"])
    phi = None if x is None else scalar_design(x)
    p = (
        None
        if phi is None
        else float(sigmoid(sum(a * b for a, b in zip(c, phi, strict=True)) / head["temperature"]))
    )
    if phi is not None and y is not None and arm != "frozen_spline":
        assert p is not None
        residual = (float(p) - y) / head["temperature"]
        g = (
            [0.0, residual, *([0.0] * 32)]
            if arm == "calibration_only"
            else [0.0, 0.0, *[residual * v for v in phi[2:]]]
        )
        norm = math.sqrt(sum(v * v for v in g))
        scale = min(1.0, 1.0 / norm) if norm else 1.0
        c = [
            min(4.0, max(-4.0, a - 0.01 * b * scale)) if b else a for a, b in zip(c, g, strict=True)
        ]
    return dict(p=p, coefficients=c)


def reconstruct(
    bundle: Json, state: Json, events: list[Json], checkpoints: Json, future: Json
) -> Json:
    """Recompute each issue and update from ordered releases, never claimed gains.

    Checkpoint states and issue-before-release journals preserve the actual causal
    order. Scalar arithmetic and changed-coefficient controls detect false parity.
    """
    slots = bundle["slots"]
    if (
        len({r["source_cluster_id"] for r in slots}) != 128
        or len({r["unit_id"] for r in slots}) != 128
    ):
        raise ValueError("source_identity")
    fit = {r["source_cluster_id"] for r in bundle["fit"]}
    separation = not fit.intersection(r["source_cluster_id"] for r in slots)
    separation = separation and not {r["source_cluster_id"] for r in slots[:96]}.intersection(
        r["source_cluster_id"] for r in slots[96:]
    )
    heads = {a: deepcopy(bundle["head"]) for a in ARMS}
    pool, issued, updates, snapshots = [], [], [], {0: deepcopy(heads)}
    probability_error, coefficient_error = 0.0, 0.0
    rng_seed = original.SEED
    releases = {r["release_slot"]: r for r in state["releases"]}
    for slot in range(1, 97):
        rows = [r for r in state["issued"] if r["slot"] == slot]
        if {r["arm"] for r in rows} != set(ARMS) or len(rows) != 5:
            raise ValueError("issued_roster")
        for row in rows:
            p = scalar(heads[row["arm"]], slots[slot - 1]["x"])["p"]
            if (p is None) != (row["p"] is None) or row["action"] != action(p):
                raise ValueError("scalar_issue")
            probability_error = max(probability_error, abs(p - row["p"]) if p is not None else 0.0)
        issued.extend(rows)
        if slot > 8:
            target = releases[slot]
            source = slots[slot - 9]
            if target["label_slot"] != slot - 8 or any(
                target[f] != source[f] for f in ("unit_id", "source_cluster_id")
            ):
                raise ValueError("release_identity")
            y = target["y"]
            if y is not None:
                pool.append(y)
            rng = random.Random(rng_seed)
            for n in range(1, len(pool)):
                rng.randrange(n)
            shuffled = rng.choice(pool) if y is not None else None
            for arm in ARMS:
                u = next(
                    r for r in state["updates"] if r["release_slot"] == slot and r["arm"] == arm
                )
                used = shuffled if arm == "shuffled_due_feedback" else y
                if u["used_y"] != used:
                    raise ValueError("released_shuffle")
                computed = scalar(heads[arm], source["x"], used, arm)["coefficients"]
                coefficient_error = max(
                    coefficient_error,
                    max(abs(a - b) for a, b in zip(computed, u["coefficients"], strict=True)),
                )
                heads[arm]["coefficients"] = computed
                updates.append(u)
        if slot in WINDOWS:
            snapshots[slot] = deepcopy(heads)
            checkpoint = checkpoints[str(slot)]
            if checkpoint["issued"] != issued or checkpoint["updates"] != updates:
                raise ValueError("checkpoint_prefix")
        if slot % 32 == 0:
            progress("scalar_reconstruction", slot, 96 - slot)
    for row in state["retention"]:
        head = snapshots[row["window"]][row["arm"]]
        p = scalar(head, slots[row["slot"] - 1]["x"])["p"]
        if (p is None) != (row["p"] is None) or row["action"] != action(p):
            raise ValueError("scalar_retention")
        probability_error = max(probability_error, abs(p - row["p"]) if p is not None else 0.0)
    if probability_error > 1e-12 or coefficient_error > 1e-12:
        raise ValueError("scalar_reconstruction_drift")
    kinds = [(r["kind"], r["slot"]) for r in events]
    commits = {r["slot"]: r["state_hash"] for r in events if r["kind"] == "commit"}
    if commits.get(96) != canonical_hash(state) or any(
        commits.get(w) != canonical_hash(checkpoints[str(w)]) for w in (32, 64, 96)
    ):
        raise ValueError("checkpoint_journal_drift")
    causal = all(kinds.index(("issue", s)) < kinds.index(("release", s)) for s in range(9, 97))
    restart = checkpoints["crash32"] == state == checkpoints["crash64"]
    future_ok = (
        state["issued"][:285] == future["issued"][:285]
        and state["issued"][285:] != future["issued"][285:]
    )
    rejected = []
    for target, clock in [(state["releases"][0], 9), (state["releases"][1], 9)]:
        control_state = deepcopy(state)
        if target["label_slot"] == 2:
            control_state["feedback_ids"].remove(target["unit_id"])
            control_state["pending"].append(2)
        try:
            original.release(control_state, bundle, target, clock)
        except ValueError:
            rejected.append(True)
    proof = original.numeric_proof(dict(bundle=bundle, state=state))
    attribution = original.attribution(bundle, state)
    first = next(
        r for r in attribution if r["arm"] == "online_sparse" and r["later_distinct_sources"]
    )
    update = next(
        u
        for u in state["updates"]
        if u["arm"] == "online_sparse" and u["release_slot"] == first["release_slot"]
    )
    after = dict(bundle["head"], coefficients=update["coefficients"])
    before = dict(
        after,
        coefficients=[
            c - d for c, d in zip(update["coefficients"], update["coefficient_delta"], strict=True)
        ],
    )
    substitution_error = max(
        abs(scalar(h, slots[r["slot"] - 1]["x"])["p"] - r[f])
        for r in first["later_distinct_sources"]
        for h, f in [(before, "p_before"), (after, "p_after")]
    )
    substitution = dict(
        passed=substitution_error <= 1e-12,
        later_distinct_source_count=len(first["later_distinct_sources"]),
        scalar_error_max=substitution_error,
        before_update_state=before,
        after_update_state=after,
        release_slot=first["release_slot"],
        update_source=first["update_source"],
        later_source_rows=first["later_distinct_sources"],
    )
    reach = reachability(bundle, state)
    checks = dict(
        issue_before_release=causal,
        restart_parity=restart,
        future_label_invariance=future_ok,
        duplicate_stale_rejection=len(rejected) == 2,
        feedback_rejection_controls=["duplicate", "stale"] if len(rejected) == 2 else [],
        source_separation=separation,
        scalar_update=proof,
        before_update_substitution=substitution,
        probability_error_max=probability_error,
        coefficient_error_max=coefficient_error,
        update_budget_control=original.control(bundle["head"]),
        later_source_attribution=attribution,
        **reach,
    )
    checks["passed"] = all(
        [
            causal,
            restart,
            future_ok,
            len(rejected) == 2,
            separation,
            proof["recomputed"],
            proof["deliberate_error_rejected"],
            checks["update_budget_control"]["passed"],
            substitution["passed"],
        ]
    )
    return checks


def reachability(bundle: Json, state: Json) -> Json:
    """Bound every allowed label history under the actual deployed gradient rule.

    Nonnegative spline bases allow a coefficient box to enclose every possible
    update. All-zero/all-one histories provide reachable witnesses on the same
    distinct admitted source schedule; they are controls, never natural labels.
    """
    head = bundle["head"]
    lo, hi = list(head["coefficients"]), list(head["coefficients"])
    lower, upper = deepcopy(head), deepcopy(head)
    admitted = {
        u["release_slot"]: u
        for u in state["updates"]
        if u["arm"] == "online_sparse" and u["reason"] == "applied"
    }
    rows = []
    for slot in range(1, 97):
        x = bundle["slots"][slot - 1]["x"]
        if slot >= 9:
            phi = None if x is None else scalar_design(x)
            zlo = (
                None
                if phi is None
                else sum(a * b for a, b in zip(lo, phi, strict=True)) / head["temperature"]
            )
            zhi = (
                None
                if phi is None
                else sum(a * b for a, b in zip(hi, phi, strict=True)) / head["temperature"]
            )
            p0, p1 = scalar(lower, x)["p"], scalar(upper, x)["p"]
            frozen = scalar(head, x)["p"]
            witness = action(p0) != action(frozen) or action(p1) != action(frozen)
            unreachable = (
                zlo is not None
                and zhi is not None
                and not any(zlo <= boundary <= zhi for boundary in (-math.log(3), math.log(3)))
            )
            rows.append(
                dict(
                    slot=slot,
                    unit_id=bundle["slots"][slot - 1]["unit_id"],
                    initial_p=frozen,
                    logit_lower=zlo,
                    logit_upper=zhi,
                    reachable_witness=witness,
                    certified_unreachable=unreachable,
                    all_zero_probability=p0,
                    all_one_probability=p1,
                )
            )
        if slot in admitted:
            source_x = bundle["slots"][admitted[slot]["label_slot"] - 1]["x"]
            phi = scalar_design(source_x)
            pmin = scalar(dict(head, coefficients=lo), source_x)["p"]
            pmax = scalar(dict(head, coefficients=hi), source_x)["p"]
            lo = [
                a if i < 2 else max(-4.0, a - 0.01 * pmax * v / head["temperature"])
                for i, (a, v) in enumerate(zip(lo, phi, strict=True))
            ]
            hi = [
                a if i < 2 else min(4.0, a + 0.01 * (1 - pmin) * v / head["temperature"])
                for i, (a, v) in enumerate(zip(hi, phi, strict=True))
            ]
            lower["coefficients"] = scalar(lower, source_x, 0, "online_sparse")["coefficients"]
            upper["coefficients"] = scalar(upper, source_x, 1, "online_sparse")["coefficients"]
    return dict(
        reachable_count=sum(r["reachable_witness"] for r in rows),
        certified_unreachable_count=sum(r["certified_unreachable"] for r in rows),
        feature_qualified_count=sum(r["initial_p"] is not None for r in rows),
        rows=rows,
        method="coefficient_interval_enclosure_and_actual_rule_extreme_history_witnesses",
        control_verdict_class="circular_positive",
        update_rule="four_features_fixed_cubic_knots_lr.01_temperature_scaled_clipped_gradient",
    )


def bootstrap(rows: list[Json]) -> Json:
    """Preserve serial order with fixed moving blocks; exposed intervals are descriptive."""
    progress("before_benchmark_moving_block_bootstrap", 0, 10000)
    values = np.asarray([[r["gain_lower"], r["gain_upper"]] for r in rows])
    starts = np.random.default_rng(7178312).integers(0, 81, (10000, 11))
    indices = (starts[:, :, None] + np.arange(8)).reshape(10000, 88)
    means = values[indices].mean(axis=1)
    known = bool(np.all(values[:, 0] == values[:, 1]))
    result = dict(
        requested_draws=10000,
        valid_draws=10000,
        block_length=8,
        random_seed=7178312,
        nominal_one_sided_level=0.975,
        mean_gain=float(values[:, 0].mean()) if known else None,
        mean_gain_lower=float(values[:, 0].mean()),
        mean_gain_upper=float(values[:, 1].mean()),
        lower_one_sided_975=float(np.quantile(means[:, 0], 0.025)),
        upper_descriptive_975=float(np.quantile(means[:, 1], 0.975)),
        bootstrap_checksum=canonical_hash(means.tolist()),
        scope="fixed_exposed_trajectory_descriptive_no_IID_or_conformal_guarantee",
    )
    progress("after_benchmark_moving_block_bootstrap", 10000, 0)
    return result


def reduce(state: Json, targets: list[Json], support: Json) -> Json:
    """Join each original source once, keeping unavailable labels and all windows."""
    labels = {t["slot"]: t for t in targets}
    if (
        len(labels) != 128
        or len(targets) != 128
        or set(labels) != set(range(1, 129))
        or len({t["source_cluster_id"] for t in targets}) != 128
    ):
        raise ValueError("target_roster")
    scored = []
    keys = set()
    for p in state["issued"] + state["retention"]:
        window = p.get("window")
        key = (p["slot"], p["arm"], window)
        t = labels[p["slot"]]
        if (
            key in keys
            or p["arm"] not in ARMS
            or p["action"] != action(p["p"])
            or t["y"] not in (0, 1, None)
            or any(p[f] != t[f] for f in ("unit_id", "source_cluster_id"))
        ):
            raise ValueError("prediction_identity_or_action")
        keys.add(key)
        costs = [
            cast(float, cost(p["action"], y)) for y in ([t["y"]] if t["y"] is not None else [0, 1])
        ]
        scored.append(
            dict(
                p,
                y=t["y"],
                qualified=p["p"] is not None and t["y"] is not None,
                cost=cost(p["action"], t["y"]),
                cost_lower=min(costs),
                cost_upper=max(costs),
                brier=(p["p"] - t["y"]) ** 2 if p["p"] is not None and t["y"] is not None else None,
            )
        )
    expected = {(s, a, None) for s in range(1, 97) for a in ARMS} | {
        (s, a, w) for s in range(97, 129) for a in ARMS for w in WINDOWS
    }
    if keys != expected:
        raise ValueError("all_arm_window_roster")
    lookup = {(r["slot"], r["arm"], r.get("window")): r for r in scored}
    paired = []
    for s in range(9, 97):
        f, o = lookup[s, "frozen_spline", None], lookup[s, "online_sparse", None]
        y = labels[s]["y"]
        gains = [
            cast(float, cost(f["action"], z)) - cast(float, cost(o["action"], z))
            for z in ([y] if y is not None else [0, 1])
        ]
        paired.append(
            dict(
                slot=s,
                unit_id=f["unit_id"],
                source_cluster_id=f["source_cluster_id"],
                y=y,
                qualified=all(lookup[s, a, None]["qualified"] for a in ARMS),
                frozen_cost=f["cost"],
                online_cost=o["cost"],
                gain_lower=min(gains),
                gain_upper=max(gains),
                probability_changed=f["p"] != o["p"],
                action_changed=f["action"] != o["action"],
            )
        )
    qualified = [r for r in paired if r["qualified"]]
    classes = {str(y): sum(r["y"] == y for r in qualified) for y in (0, 1)}
    arm_results = []
    for arm in ARMS:
        rows = [lookup[s, arm, None] for s in range(9, 97)]
        complete = [r for r in rows if r["qualified"]]
        accepted = [r for r in complete if r["action"] == "accept"]
        arm_results.append(
            dict(
                arm=arm,
                intended_count=88,
                qualified_count=len(complete),
                cost_mean=sum(r["cost_lower"] for r in rows) / 88,
                cost_mean_upper=sum(r["cost_upper"] for r in rows) / 88,
                brier_mean=sum(r["brier"] for r in complete) / len(complete) if complete else None,
                wrong_accept_rate=sum(r["y"] == 1 for r in accepted) / len(accepted)
                if accepted
                else None,
                wrong_accept_count=sum(r["y"] == 1 and r["action"] == "accept" for r in complete),
                coverage=sum(r["action"] != "escalate" for r in rows) / 88,
                probability_changes=sum(
                    r["p"] != lookup[r["slot"], "frozen_spline", None]["p"] for r in rows
                ),
                action_changes=sum(
                    r["action"] != lookup[r["slot"], "frozen_spline", None]["action"] for r in rows
                ),
            )
        )
    retained = [r for r in scored if r.get("window") is not None]
    windows = retention_summary(retained, lookup)
    interval = bootstrap(paired)
    parity = all(
        lookup[s, "online_dense", None]["p"] == lookup[s, "online_sparse", None]["p"]
        and lookup[s, "online_dense", None]["action"] == lookup[s, "online_sparse", None]["action"]
        for s in range(9, 97)
    )
    parity = parity and all(
        lookup[s, "online_dense", w]["p"] == lookup[s, "online_sparse", w]["p"]
        for s in range(97, 129)
        for w in WINDOWS
    )
    enough = len(qualified) >= 64 and min(classes.values()) >= 8
    oracle = sum(float(r["frozen_cost"]) for r in qualified) / len(qualified) if qualified else 0.0
    informative = (
        enough
        and oracle > 0
        and support["passed"]
        and support["reachable_count"] > 0
        and any(u["arm"] == "online_sparse" and u["reason"] == "applied" for u in state["updates"])
    )
    final = windows[-1]
    retention_ok = (
        final["qualified_count"] >= 20
        and min(final["class_support"].values()) >= 4
        and final["online_cost_degradation"] <= 0.02
        and final["online_brier_degradation"] <= 0.01
    )
    signal = int(
        informative
        and interval["mean_gain_lower"] >= 0.02
        and interval["lower_one_sided_975"] > 0
        and retention_ok
    )
    unreachable = (
        support.get("certified_unreachable_count", support.get("feature_qualified_count", 1))
        == support.get("feature_qualified_count", 1)
        and support["reachable_count"] == 0
    )
    disposition = (
        "positive_delayed_decision_benefit"
        if signal
        else "null_insufficient_support"
        if not enough
        else "null_update_budget_no_headroom"
        if unreachable
        else "null_delayed_decision_benefit"
        if informative
        else "null_insufficient_support"
    )
    return dict(
        paired_cost_rows=paired,
        block_bootstrap_summary=interval,
        arm_results=arm_results,
        qualified_count=len(qualified),
        class_support=classes,
        missing_bounds=[r for r in paired if not r["qualified"]],
        retention_rows=retained,
        retention_window_bounds=windows,
        dense_sparse_exact_parity=parity,
        permissible_action_oracle_headroom=oracle,
        utility_null_informative=informative,
        h2_development_signal_score=signal,
        science_disposition=disposition,
        retention_qualified=retention_ok,
        typed_action_control=dict(
            passed=cost("accept", 1) == 1
            and cost("reject", 0) == 1
            and cost("escalate", None) == 0.5
            and action(0.25) == "escalate"
            and action(0.75) == "escalate",
            verdict_class="circular_positive",
        ),
        calibration_only_comparison_scope="descriptive_frozen_win_does_not_establish_locality_specific_advantage",
    )


def retention_summary(retained: list[Json], lookup: dict[tuple[Any, ...], Json]) -> list[Json]:
    """All fixed windows and missing bounds remain visible on the small panel."""
    result = []
    for window in WINDOWS:
        frozen = [lookup[s, "frozen_spline", window] for s in range(97, 129)]
        qualified = [
            r for r in frozen if all(lookup[r["slot"], a, window]["qualified"] for a in ARMS)
        ]
        arms = []
        for arm in ARMS:
            rows = [lookup[s, arm, window] for s in range(97, 129)]
            complete = [r for r in rows if r["qualified"]]
            pairs = [(r, lookup[r["slot"], "frozen_spline", window]) for r in complete]
            cost_delta = (
                sum(r["cost"] - f["cost"] for r, f in pairs) / len(pairs) if pairs else None
            )
            brier_delta = (
                sum(r["brier"] - f["brier"] for r, f in pairs) / len(pairs) if pairs else None
            )
            bounds = []
            for r, f in zip(rows, frozen, strict=True):
                ys = [r["y"]] if r["y"] is not None else [0, 1]
                deltas = [
                    cast(float, cost(r["action"], y)) - cast(float, cost(f["action"], y))
                    for y in ys
                ]
                bd = (
                    [(r["p"] - y) ** 2 - (f["p"] - y) ** 2 for y in ys]
                    if r["p"] is not None and f["p"] is not None
                    else [-1.0, 1.0]
                )
                bounds.append(
                    dict(
                        slot=r["slot"],
                        missing=not r["qualified"],
                        cost_degradation_lower=min(deltas),
                        cost_degradation_upper=max(deltas),
                        brier_degradation_lower=min(bd),
                        brier_degradation_upper=max(bd),
                    )
                )
            arms.append(
                dict(
                    arm=arm,
                    intended_count=32,
                    qualified_count=len(complete),
                    cost_degradation=cost_delta,
                    brier_degradation=brier_delta,
                    cost_mean_lower=sum(r["cost_lower"] for r in rows) / 32,
                    cost_mean_upper=sum(r["cost_upper"] for r in rows) / 32,
                    brier_mean=sum(r["brier"] for r in complete) / len(complete)
                    if complete
                    else None,
                    intended_cost_degradation_lower=sum(r["cost_degradation_lower"] for r in bounds)
                    / 32,
                    intended_cost_degradation_upper=sum(r["cost_degradation_upper"] for r in bounds)
                    / 32,
                    missing_bounds=[r for r in bounds if r["missing"]],
                )
            )
        online = next(a for a in arms if a["arm"] == "online_sparse")
        result.append(
            dict(
                window=window,
                intended_count=32,
                qualified_count=len(qualified),
                class_support={str(y): sum(r["y"] == y for r in qualified) for y in (0, 1)},
                missing_slots=[r["slot"] for r in frozen if not r["qualified"]],
                arm_results=arms,
                online_cost_degradation=online["cost_degradation"],
                online_brier_degradation=online["brier_degradation"],
                scope="small_exploratory_exposed_panel_no_replacement_no_window_selection",
            )
        )
    return result

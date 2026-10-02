"""REQ-REPORT-8026: reconstruct calibrated decisions without producer reduction.

Independent equations make an altered issue state or selected target visible.
Retention remains descriptive when the audit itself has prior target exposure.
"""

from __future__ import annotations

from collections import defaultdict
import copy
import json
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline  # type: ignore[import-untyped]
from scipy.special import expit  # type: ignore[import-untyped]

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.evidence_features_custody_7980 import checked
from carnot.verify.causal_online_8025 import ARMS, progress

Json = dict[str, Any]
Array = NDArray[np.float64]
CONFIG = dict(
    seed=69526,
    draws=10000,
    blocks=[32, 16, 64],
    later_start=36,
    retention_minimum=48,
    per_class=8,
    gain=0.02,
    cost_drift=0.01,
    brier_drift=0.005,
    overlap="public-fit support prevalence; exclude shared columns",
    hypotheses=["decision_loss", "brier_loss", "periodic"],
    alpha=0.05,
)


class AuditFailure(ValueError):
    """Keep both operands so a rejected trajectory can be diagnosed exactly."""

    def __init__(self, field: str, expected: Any, observed: Any):
        self.operand = dict(
            artifact_field=field, expected=expected, observed=observed, passed=False
        )
        super().__init__(json.dumps(self.operand, sort_keys=True))


def equal(field: str, expected: Any, observed: Any) -> None:
    """Fail at the original field instead of hiding errors in a combined gate."""
    if expected != observed:
        raise AuditFailure(field, expected, observed)


def design(head: Json, row: Json) -> Array:
    """Rebuild the frozen cubic geometry from the original public feature bytes."""
    geo = head["geometry"]
    raw = np.array([[row["q"], *row["features"]]], dtype=float)
    low, high = np.array(geo["scaler"]["minimum"]), np.array(geo["scaler"]["maximum"])
    scaled = np.clip((raw - low) / np.where(high > low, high - low, 1.0), 0, 1)
    knots = [0.0] * 4 + [i / 9 for i in range(1, 9)] + [1.0] * 4
    q = np.clip(raw[:, 0], 0.0001, 0.9999)
    z = (np.log(q / (1 - q)) - geo["logit_center"]) / geo["logit_scale"]
    return np.column_stack(
        [np.ones(1), z]
        + [BSpline.design_matrix(scaled[:, j], knots, 3).toarray() for j in range(9)]
    )[0]


def probability(head: Json, x: Array) -> float:
    """The fixed affine calibrator is part of every prediction's primitive state."""
    a, b = head["calibration"]
    return float(expit(a + b * float(x @ (np.asarray(head["parameters"]) * head["decay_scale"]))))


def action(p: float | None) -> str:
    """The original cost matrix gives escalation every exact threshold tie."""
    return "escalate" if p is None else "accept" if p < 0.1 else "reject" if p > 0.5 else "escalate"


def cost(decision: str, y: int) -> float:
    """Unsupported accepts retain their original five-unit penalty."""
    return float(5 * y if decision == "accept" else 1 - y if decision == "reject" else 0.5)


def update(head: Json, x: Array, y: int) -> None:
    """Recompute sparse BCE descent and global decay without calling the learner."""
    ids = np.flatnonzero(x)
    gradient = (probability(head, x) - y) * head["calibration"][1] * x[ids]
    scale = head["decay_scale"] * (1 - 2 * 0.001 * 0.01)
    for i, g in zip(ids, gradient, strict=True):
        head["parameters"][int(i)] -= 0.01 * float(g) / scale
    head["decay_scale"] = scale


def selected(block: list[Json], arm: str, seed: int) -> list[Json]:
    """Source-keyed ties keep every registered arm on the same four-update budget."""

    def tie(r: Json) -> str:
        return canonical_hash(dict(seed=seed, identity=r["family_id"]))

    if arm == "periodic":
        return [block[i] for i in (0, 4, 8, 12)]
    if arm == "uniform":
        return sorted(block, key=tie)[:4]
    metric = "actual_cost" if arm == "decision_loss" else "brier"
    return sorted(block, key=lambda r: (-r[metric], tie(r)))[:4]


def bootstrap(diff: Array, length: int, alpha: float) -> Json:
    """Missing slots stay in place; blocks describe one trajectory's uncertainty."""
    rng = np.random.default_rng(CONFIG["seed"])
    length = min(length, len(diff))
    starts = rng.integers(0, len(diff) - length + 1, (10000, int(np.ceil(len(diff) / length))))
    ids = (starts[:, :, None] + np.arange(length)).reshape(10000, -1)[:, : len(diff)]
    samples = diff[ids]
    n = np.isfinite(samples).sum(axis=1)
    draws = np.nansum(samples[n > 0], axis=1) / n[n > 0]
    mean = float(np.nanmean(diff)) if np.isfinite(diff).any() else None
    return dict(
        gain=mean,
        interval=np.quantile(draws, [alpha / 2, 1 - alpha / 2]).tolist()
        if len(draws)
        else [None, None],
        raw_p=float((1 + np.sum(draws - mean >= mean)) / (len(draws) + 1))
        if mean is not None
        else 1.0,
        draws=10000,
        completed=len(draws),
        censored=10000 - len(draws),
        slot_count=len(diff),
        block_length=length,
        alpha=alpha,
    )


def controls() -> Json:
    """Private decision fixtures ensure true improvement and zero headroom differ."""
    gain = cost(action(0.09), 1) - cost(action(0.11), 1)
    zero = cost(action(0.01), 0) - cost(action(0.02), 0)
    return dict(
        known_headroom=dict(gain=gain, passed=gain == 4.5),
        zero_headroom=dict(gain=zero, passed=zero == 0),
        scope="artificial reducer controls; no natural benefit credit",
    )


def reduce(bundle: Json, seal: Path | None = None) -> Json:
    """Replay original rows in order, freezing retention predictions before labels."""
    raw = Path(bundle["trajectory"])
    inputs = json.loads((raw / "inputs.json").read_text())
    head, sources, seeds = inputs["head"], inputs["sources"], inputs["seeds"]
    vectors = [design(head, r) if r["public_eligible"] else None for r in sources]
    states = {(arm, seed): copy.deepcopy(head) for arm in ARMS for seed in seeds}
    issued: dict[tuple[str, int, int], Json] = {}
    issue_counts: dict[tuple[str, int], int] = defaultdict(int)
    blocks: dict[tuple[str, int], list[Json]] = defaultdict(list)
    counts: dict[tuple[str, int], int] = defaultdict(int)
    finals: Json = {}
    checkpoint_cache: Json = {}
    checkpoints: Json = {}
    rows, releases = [], []
    labels: Json = {}
    db = sqlite3.connect(f"file:{raw / 'ledger.sqlite'}?mode=ro", uri=True)
    progress("independent_benchmark_before", 0, len(sources) * len(states))
    for kind, payload in db.execute("select kind,payload from events order by seq"):
        if kind == "timing":
            continue
        r = json.loads(payload)
        key = (r["arm"], r["seed"])
        state = states[key]
        prefix = f"{kind}/{r['arm']}/{r['seed']}/{r.get('slot', r.get('origin_slot'))}"
        if kind in ("issue", "update"):
            ref = r["checkpoint"]
            if "checkpoint_directory" in bundle:
                ref = dict(
                    ref, path=str(Path(bundle["checkpoint_directory"]) / Path(ref["path"]).name)
                )
            digest = ref["sha256"]
            if digest not in checkpoint_cache:
                checkpoint_cache[digest] = json.loads(checked(ref).read_text())
            checkpoints[digest] = ref
        if kind == "issue":
            equal(prefix + "/slot", issue_counts[key], r["slot"])
            equal(prefix + "/checkpoint_state", state, checkpoint_cache[digest])
            equal(prefix + "/head_hash", canonical_hash(state), r["head_hash"])
            source, x = sources[r["slot"]], vectors[r["slot"]]
            equal(prefix + "/family_id", source["family_id"], r["family_id"])
            p = probability(state, x) if x is not None else None
            equal(prefix + "/probability", p, r["probability"])
            equal(prefix + "/action", action(p), r["action"])
            row = dict(
                arm=r["arm"],
                seed=r["seed"],
                slot=r["slot"],
                family_id=source["family_id"],
                source_cluster_id=source["source_cluster_id"],
                probability=p,
                action=action(p),
                head_hash=r["head_hash"],
                cost=None,
                brier=None,
                y=None,
                false_accept=None,
                numerator=None,
                denominator=0,
                eligibility=False,
                exclusion_reason=source.get("exclusion_reason"),
                failure_reason=None,
                censor_reason="pending_delay",
                update_count=counts[key],
            )
            issued[(*key, r["slot"])] = row
            rows.append(row)
            issue_counts[key] += 1
            if len(rows) % 256 == 0:
                progress("independent_issues", len(rows), len(sources) * len(states) - len(rows))
        elif kind == "release":
            equal(prefix + "/issue_before_release", True, (*key, r["due_slot"]) in issued)
            equal(prefix + "/due_slot", r["origin_slot"] + 20, r["due_slot"])
            old = issued[(*key, r["origin_slot"])]
            if not labels:
                document = json.loads(checked(bundle["stream_target"]).read_text())
                labels.update({t["family_id"]: t["eligible_y"] for t in document["rows"]})
            y = labels[old["family_id"]]
            equal(prefix + "/y", y, r["y"])
            equal(prefix + "/family_id", old["family_id"], r["family_id"])
            equal(prefix + "/issued_probability", old["probability"], r["issued_probability"])
            equal(prefix + "/issued_action", old["action"], r["issued_action"])
            valid = y is not None and old["probability"] is not None
            equal(prefix + "/eligibility", valid, r["eligibility"])
            c = cost(old["action"], y) if valid else None
            b = (old["probability"] - y) ** 2 if valid else None
            equal(prefix + "/actual_cost", c, r["actual_cost"])
            equal(prefix + "/brier", b, r["brier"])
            old.update(
                y=y,
                cost=c,
                brier=b,
                false_accept=int(old["action"] == "accept" and y == 1) if valid else None,
                numerator=c,
                denominator=int(valid),
                eligibility=valid,
                censor_reason=None,
                exclusion_reason=None if valid else "unknown_or_unavailable_target",
            )
            release = dict(r, actual_cost=c, brier=b)
            releases.append(release)
            if valid:
                blocks[key].append(release)
        elif kind == "update":
            index = counts[key]
            equal(prefix + "/update_index", index, r["update_index"])
            block = blocks[key][index // 4 * 16 : (index // 4 + 1) * 16]
            equal(prefix + "/complete_block", 16, len(block))
            choice = selected(block, r["arm"], r["seed"])[index % 4]
            for field in ("family_id", "origin_slot", "due_slot", "y"):
                equal(prefix + "/" + field, choice[field], r[field])
            equal(
                prefix + "/selection_block_ids",
                [t["family_id"] for t in block],
                r["selection_block_ids"],
            )
            equal(prefix + "/before_head_hash", canonical_hash(state), r["before_head_hash"])
            equal(
                prefix + "/before_coefficients",
                (np.asarray(state["parameters"]) * state["decay_scale"]).tolist(),
                r["before_coefficients"],
            )
            equal(prefix + "/release_due", True, r["due_slot"] <= r["slot"])
            update(state, vectors[r["origin_slot"]], r["y"])
            equal(
                prefix + "/after_coefficients",
                (np.asarray(state["parameters"]) * state["decay_scale"]).tolist(),
                r["after_coefficients"],
            )
            equal(prefix + "/after_head_hash", canonical_hash(state), r["after_head_hash"])
            equal(prefix + "/checkpoint_state", state, checkpoint_cache[digest])
            counts[key] += 1
            finals[key] = ref
    db.close()
    budgets = []
    for (arm, seed), state in states.items():
        equal(
            f"issue/{arm}/{seed}/count",
            len(sources),
            sum(r["arm"] == arm and r["seed"] == seed for r in rows),
        )
        expected = min(64, len(blocks[(arm, seed)]) // 16 * 4) if arm != "frozen_no_write" else 0
        equal(f"budget/{arm}/{seed}/updates", expected, counts[(arm, seed)])
        budgets.append(
            dict(
                arm=arm,
                seed=seed,
                updates=counts[(arm, seed)],
                expected=expected,
                numerator=counts[(arm, seed)],
                denominator=expected,
                passed=True,
                later_cost_numerator=sum(
                    r["cost"]
                    for r in rows
                    if r["arm"] == arm
                    and r["seed"] == seed
                    and r["slot"] >= 36
                    and r["eligibility"]
                ),
                later_cost_denominator=sum(
                    r["arm"] == arm and r["seed"] == seed and r["slot"] >= 36 and r["eligibility"]
                    for r in rows
                ),
                source_transition_count=len(
                    {
                        r["source_cluster_id"]
                        for r in rows
                        if r["arm"] == arm
                        and r["seed"] == seed
                        and r["eligibility"]
                        and r["action"] != issued[("frozen_no_write", seed, r["slot"])]["action"]
                    }
                ),
            )
        )
    progress("independent_benchmark_after", len(rows), 0)
    retention, overlap, support, drift = retention_measure(bundle, head, states, seal)
    intervals, benefits = comparisons(rows, retention, support, seeds)
    return dict(
        independent_issue_rows=rows,
        budget_comparison_rows=budgets,
        retention_rows=retention,
        overlap_strata=overlap,
        retention_support=support,
        retention_drift_rows=drift,
        block_bootstrap_intervals=intervals,
        benefit_rows=benefits,
        checkpoint_references=list(checkpoints.values()),
        positive_control_results=controls(),
        independent_release_rows=releases,
        frozen_final_head_hashes={f"{a}/{s}": canonical_hash(h) for (a, s), h in states.items()},
    )


def retention_measure(
    bundle: Json, initial: Json, states: dict[tuple[str, int], Json], seal: Path | None
) -> tuple[list[Json], list[Json], Json, list[Json]]:
    """Public fit support fixes quartiles; labels cannot select slots or strata."""
    public = bundle["retention_public"]
    equal("retention/original_slots", list(range(64)), [r["slot"] for r in public])
    fit = [design(initial, r)[2:] != 0 for r in bundle["fit_public"] if r["public_eligible"]]
    prevalence = np.mean(fit, axis=0)

    def overlap(x: Array) -> float:
        active = x[2:] != 0
        return float(np.mean(prevalence[active]))

    cuts = np.quantile([float(np.mean(prevalence[x])) for x in fit], [0.25, 0.5, 0.75])
    predictions = []
    for (arm, seed), h in states.items():
        for r in public:
            x = design(initial, r) if r["public_eligible"] else None
            p = probability(h, x) if x is not None else None
            p0 = probability(initial, x) if x is not None else None
            ov = overlap(x) if x is not None else None
            predictions.append(
                dict(
                    arm=arm,
                    seed=seed,
                    slot=r["slot"],
                    family_id=r["family_id"],
                    source_cluster_id=r["source_cluster_id"],
                    probability=p,
                    initial_probability=p0,
                    action=action(p),
                    initial_action=action(p0),
                    overlap=ov,
                    quartile=int(np.searchsorted(cuts, ov, side="right"))
                    if ov is not None
                    else None,
                )
            )
    if seal:
        atomic_json(
            seal,
            dict(
                cutpoints=cuts.tolist(),
                predictions=predictions,
                final_states={f"{a}/{s}": canonical_hash(h) for (a, s), h in states.items()},
                retention_labels_opened=False,
            ),
        )
    progress("retention_prediction_seal_before_target_access", len(predictions), 0)
    labels = {
        r["family_id"]: r["eligible_y"]
        for r in json.loads(checked(bundle["retention_target"]).read_text())["rows"]
    }
    for r in predictions:
        y = labels[r["family_id"]]
        valid = y is not None and r["probability"] is not None
        c, c0 = (cost(r["action"], y), cost(r["initial_action"], y)) if valid else (None, None)
        b = (r["probability"] - y) ** 2 if valid else None
        b0 = (r["initial_probability"] - y) ** 2 if valid else None
        r.update(
            y=y,
            cost=c,
            brier=b,
            initial_cost=c0,
            initial_brier=b0,
            cost_drift=c - c0 if valid else None,
            brier_drift=b - b0 if valid else None,
            false_accept=int(r["action"] == "accept" and y == 1) if valid else None,
            initial_false_accept=int(r["initial_action"] == "accept" and y == 1) if valid else None,
            changed_action=r["action"] != r["initial_action"],
            eligibility=valid,
            numerator=c,
            denominator=int(valid),
            failure_reason=None,
            censor_reason=None,
            exclusion_reason=None if valid else "unknown_or_unavailable_target",
        )
    unique = {r["source_cluster_id"]: r for r in predictions if r["eligibility"]}
    classes = {str(y): sum(r["y"] == y for r in unique.values()) for y in (0, 1)}
    support = dict(
        intended=64,
        completed=64,
        eligible=len(unique),
        independent=len(unique),
        class_counts=classes,
        minimum=48,
        per_class=8,
        excluded=64 - len(unique),
        passed=len(unique) >= 48 and min(classes.values()) >= 8,
        cutpoints=cuts.tolist(),
        cutpoint_source="original public fit features only",
        prior_exposure="historical development; no independent deployment",
    )
    drift = []
    for arm, seed in states:
        rs = [r for r in predictions if r["arm"] == arm and r["seed"] == seed and r["eligibility"]]
        drift.append(
            dict(
                arm=arm,
                seed=seed,
                denominator=len(rs),
                cost_drift=float(np.mean([r["cost_drift"] for r in rs])) if rs else None,
                brier_drift=float(np.mean([r["brier_drift"] for r in rs])) if rs else None,
                false_accepts=sum(r["false_accept"] for r in rs),
                initial_false_accepts=sum(r["initial_false_accept"] for r in rs),
                source_transition_count=len(
                    {r["source_cluster_id"] for r in rs if r["changed_action"]}
                ),
            )
        )
    strata = []
    for arm in ARMS:
        for q in range(4):
            rs = [
                r
                for r in predictions
                if r["arm"] == arm and r["quartile"] == q and r["eligibility"]
            ]
            strata.append(
                dict(
                    arm=arm,
                    quartile=q,
                    independent=len({r["source_cluster_id"] for r in rs}),
                    denominator=len(rs),
                    cost_drift=float(np.mean([r["cost_drift"] for r in rs])) if rs else None,
                    brier_drift=float(np.mean([r["brier_drift"] for r in rs])) if rs else None,
                    source_transition_count=len(
                        {r["source_cluster_id"] for r in rs if r["changed_action"]}
                    ),
                    scope="descriptive; shared intercept, logit and global decay can affect disjoint spline support",
                )
            )
    return predictions, strata, support, drift


def comparisons(
    rows: list[Json], retention: list[Json], support: Json, seeds: list[int]
) -> tuple[list[Json], list[Json]]:
    """Pair slots and average seeds before the three frozen comparisons are tested."""
    by = {(r["arm"], r["seed"], r["slot"]): r for r in rows}
    slots = sorted({r["slot"] for r in rows if r["slot"] >= 36})
    diffs = {}
    for arm in CONFIG["hypotheses"]:
        diff = []
        for slot in slots:
            rs = [(by[(arm, s, slot)], by[("uniform", s, slot)]) for s in seeds]
            diff.append(
                float(np.mean([b["cost"] - a["cost"] for a, b in rs]))
                if all(a["eligibility"] and b["eligibility"] for a, b in rs)
                else np.nan
            )
        diffs[arm] = np.asarray(diff)
    raw = {a: bootstrap(d, 32, 0.05)["raw_p"] for a, d in diffs.items()}
    order = sorted(raw, key=lambda a: raw[a])
    intervals, benefits = [], []
    previous = 0.0
    for rank, arm in enumerate(order):
        adjusted = max(previous, min(1.0, raw[arm] * (3 - rank)))
        previous = adjusted
        ci = bootstrap(diffs[arm], 32, 0.05 / (3 - rank))
        for length in (32, 16, 64):
            intervals.append(
                dict(
                    arm=arm,
                    registered_block=length,
                    **bootstrap(diffs[arm], length, 0.05 / (3 - rank)),
                    holm_adjusted_p=adjusted,
                    method="Holm step-down bootstrap intervals; seeds averaged per original slot",
                    scope="one development trajectory; no independent environments",
                )
            )
        rs = [r for r in rows if r["arm"] == arm and r["slot"] >= 36 and r["eligibility"]]
        base = [by[("uniform", r["seed"], r["slot"])] for r in rs]
        held = [r for r in retention if r["arm"] == arm and r["eligibility"]]
        fa = sum(r["false_accept"] for r in rs) - sum(r["false_accept"] for r in base)
        cd = float(np.mean([r["cost_drift"] for r in held])) if held else None
        bd = float(np.mean([r["brier_drift"] for r in held])) if held else None
        gates = dict(
            mean_gain=ci["gain"] is not None and ci["gain"] >= 0.02,
            adjusted_ci=ci["interval"][0] is not None and ci["interval"][0] > 0 and adjusted < 0.05,
            false_accepts=fa <= 0,
            retention_support=support["passed"],
            retention_cost=cd is not None and cd <= 0.01,
            retention_brier=bd is not None and bd <= 0.005,
        )
        benefits.append(
            dict(
                arm=arm,
                mean_later_gain=ci["gain"],
                false_accept_delta=fa / len(seeds),
                retention_cost_drift=cd,
                retention_brier_drift=bd,
                acceptance_gates=gates,
                benefit_passed=all(gates.values()),
                denominator=int(np.isfinite(diffs[arm]).sum()),
            )
        )
    return intervals, benefits

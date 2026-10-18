"""REQ-REPORT-8039: replay primitive states with separate audit equations.

The independent cubic design and gradient equations come from the earlier audit,
not the learner. Public fit support fixes retention strata before labels open.
"""

from __future__ import annotations

from collections import defaultdict
import copy
import json
from pathlib import Path
import random
import sqlite3
import time
from typing import Any

import numpy as np

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.verify import learning_retention_audit_8026 as math

Json = dict[str, Any]
ARMS = ("recent64", "cumulative", "newest16", "frozen_no_write")
CONFIG = dict(
    seed=6968039,
    draws=10000,
    blocks=[32, 16, 64],
    margin=0.02,
    cost_drift=0.02,
    brier_drift=0.01,
    stream_minimum=192,
    stream_per_class=20,
    later_minimum=160,
    changed_minimum=5,
    retention_minimum=48,
    retention_per_class=8,
    tolerance=1e-10,
    budget_s=900,
    effective_independent_streams=1,
    holm_family=["H1", "H2", "H3"],
)
BEGAN = time.monotonic()


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Real elapsed time and counts expose progress without adding artificial work."""
    print(
        f"[exp8039] phase={phase} elapsed_s={time.monotonic() - BEGAN:.3f} "
        f"completed={completed} pending={pending}",
        flush=True,
    )


def equal(field: str, expected: Any, observed: Any) -> None:
    """Keep failed operands instead of silently converting invalid data to zero."""
    math.equal(field, expected, observed)


def local_ref(raw: Path, ref: Json) -> Json:
    """Copied producer bytes retain their hash while becoming durable task inputs."""
    p = Path(ref["path"])
    return dict(
        ref, path=str(raw / ("checkpoints" if p.parent.name == "checkpoints" else "") / p.name)
    )


def selected(
    released: list[Json], arm: str, seed: int, block: int
) -> tuple[list[Json], list[Json]]:
    """Reproduce registered uniform draws using only public identities and age."""
    pool = (
        released[-64:] if arm == "recent64" else released if arm == "cumulative" else released[-16:]
    )
    tie = lambda r: canonical_hash(dict(seed=seed, identity=r["family_id"]))
    if arm == "frozen_no_write":
        return pool, []
    if arm == "newest16":
        return pool, sorted(pool, key=tie)[:4]
    ordered = sorted(pool, key=tie)
    rng = random.Random(int(canonical_hash(dict(seed=seed, block=block)).split(":")[1], 16))
    return pool, rng.sample(ordered, 4)


def replay_trajectory(raw: Path) -> Json:
    """Check every causal event, update and final checkpoint without producer reduction."""
    seal = json.loads((raw / "seal.json").read_text())
    equal("trajectory_sealed", True, seal["sealed"])
    for ref in seal["references"]:
        checked(local_ref(raw, ref))
    data = json.loads((raw / "inputs.json").read_text())
    methods = json.loads((raw / "methods.json").read_text())["config"]
    for field, expected in dict(
        delay=20,
        block=16,
        updates_per_block=4,
        maximum_updates=64,
        learning_rate=0.01,
        l2=0.001,
        terminal_flush=False,
        retention_labels_opened=False,
    ).items():
        equal("producer_methods." + field, expected, methods[field])
    head, sources, seeds = data["head"], data["sources"], data["seeds"]
    progress("independent_small_head_load_before")
    vectors = [math.design(head, r) if r["public_eligible"] else None for r in sources]
    states = {(arm, seed): copy.deepcopy(head) for seed in seeds for arm in ARMS}
    progress("independent_small_head_load_after")
    issued: dict[tuple[str, int, int], Json] = {}
    releases: dict[tuple[str, int], list[Json]] = defaultdict(list)
    release_counts: dict[tuple[str, int], int] = defaultdict(int)
    issue_counts: dict[tuple[str, int], int] = defaultdict(int)
    counts: dict[tuple[str, int], int] = defaultdict(int)
    buffers: dict[tuple[str, int], list[Json]] = {}
    labels: dict[tuple[str, int, int], Json] = {}
    first: dict[tuple[str, int], int] = {}
    finals, updates, commits, seen = [], [], [], set()
    coefficient_error = probability_error = 0.0
    db = sqlite3.connect(f"file:{raw / 'ledger.sqlite'}?mode=ro", uri=True)
    events = db.execute("SELECT seq,kind,identity,payload FROM events ORDER BY seq").fetchall()
    db.close()
    deadline = time.monotonic() + CONFIG["budget_s"]
    progress("independent_benchmark_before", 0, len(events))
    for index, (seq, kind, identity, payload) in enumerate(events):
        if (index + 1) % 2048 == 0:
            progress("independent_events", index + 1, len(events) - index - 1)
        equal("numerical_budget", True, time.monotonic() <= deadline)
        equal("duplicate_commit_identity", False, identity in seen)
        seen.add(identity)
        if kind == "timing":
            continue
        r = json.loads(payload)
        key = (r["arm"], r["seed"])
        state = states[key]
        if kind == "issue":
            slot = r["slot"]
            equal("issue_order", issue_counts[key], slot)
            equal("issue_identity", f"issue/{key[0]}/{key[1]}/{slot}", identity)
            equal("issue_source", sources[slot]["family_id"], r["family_id"])
            p = math.probability(state, vectors[slot]) if vectors[slot] is not None else None
            error = (
                abs(p - r["probability"]) if p is not None and r["probability"] is not None else 0.0
            )
            equal("probability_presence", p is None, r["probability"] is None)
            equal("probability_tolerance", True, error <= CONFIG["tolerance"])
            probability_error = max(probability_error, error)
            equal("action", math.action(p), r["action"])
            equal("issue_state", canonical_hash(state), r["head_hash"])
            issued[(*key, slot)] = dict(r, durable_commit_id=seq, probability=p)
            issue_counts[key] += 1
        elif kind == "release":
            slot = r["origin_slot"]
            equal("feedback_order", release_counts[key], slot)
            equal("release_identity", f"release/{key[0]}/{key[1]}/{slot}", identity)
            equal("feedback_delay", slot + 20, r["due_slot"])
            equal("durable_issue_before_feedback", True, (*key, r["due_slot"]) in issued)
            prior = issued[(*key, slot)]
            equal("feedback_source", prior["family_id"], r["family_id"])
            equal(
                "label_contract", True, r["y"] is None or (type(r["y"]) is int and r["y"] in (0, 1))
            )
            valid = r["y"] is not None and prior["probability"] is not None
            equal("release_eligibility", valid, r["eligibility"])
            equal(
                "issued_loss",
                math.cost(prior["action"], r["y"]) if valid else None,
                r["issued_decision_loss"],
            )
            equal(
                "issued_brier",
                (prior["probability"] - r["y"]) ** 2 if valid else None,
                r["issued_brier_loss"],
            )
            labels[(*key, slot)] = r
            release_counts[key] += 1
            if valid:
                releases[key].append(r)
        elif kind == "buffer":
            equal("buffer_block", len(releases[key]), r["block"] * 16)
            pool, chosen = selected(releases[key], *key, r["block"])
            equal(
                "buffer_pool", [z["family_id"] for z in pool], [z["family_id"] for z in r["pool"]]
            )
            equal("buffer_selection", [z["family_id"] for z in chosen], r["selected_ids"])
            buffers[key] = chosen
        elif kind == "commit":
            equal("commit_identity", f"commit/{key[0]}/{key[1]}/{r['block']}", identity)
            equal("commit_start", canonical_hash(state), r["before_head_hash"])
            chosen = buffers.pop(key) if counts[key] < 64 else []
            equal("update_budget", len(chosen), len(r["gradients"]))
            for g, z in zip(r["gradients"], chosen, strict=True):
                equal(
                    "update_source",
                    (z["family_id"], z["origin_slot"], z["y"], counts[key]),
                    (g["family_id"], g["origin_slot"], g["y"], g["update_index"]),
                )
                before = np.asarray(state["parameters"]) * state["decay_scale"]
                equal(
                    "before_coefficients",
                    True,
                    bool(np.max(np.abs(before - g["before_coefficients"])) <= CONFIG["tolerance"]),
                )
                math.update(state, vectors[z["origin_slot"]], z["y"])
                after = np.asarray(state["parameters"]) * state["decay_scale"]
                error = float(np.max(np.abs(after - g["after_coefficients"])))
                equal("coefficient_tolerance", True, error <= CONFIG["tolerance"])
                coefficient_error = max(coefficient_error, error)
                equal("update_state", canonical_hash(state), g["after_head_hash"])
                counts[key] += 1
                first.setdefault(key, r["slot"])
                updates.append(
                    dict(
                        arm=key[0],
                        seed=key[1],
                        family_id=z["family_id"],
                        slot=r["slot"],
                        update_index=counts[key] - 1,
                        commit_identity=identity,
                        durable_commit_id=seq,
                        coefficient_error=error,
                        passed=True,
                    )
                )
            equal("commit_count", counts[key], r["actual_gradient_count"])
            equal(
                "commit_checkpoint",
                state,
                json.loads(checked(local_ref(raw, r["checkpoint"])).read_text()),
            )
            equal("commit_head_hash", canonical_hash(state), r["head_hash"])
            commits.append(
                dict(
                    identity=identity,
                    durable_commit_id=seq,
                    arm=key[0],
                    seed=key[1],
                    checkpoint=local_ref(raw, r["checkpoint"]),
                    passed=True,
                )
            )
        elif kind == "final":
            equal("final_identity", f"final/{key[0]}/{key[1]}", identity)
            expected = 0 if key[0] == "frozen_no_write" else min(64, len(releases[key]) // 16 * 4)
            equal(
                "final_budget",
                (expected, len(sources), max(0, len(sources) - 20)),
                (counts[key], issue_counts[key], release_counts[key]),
            )
            equal(
                "final_checkpoint",
                state,
                json.loads(checked(local_ref(raw, r["checkpoint"])).read_text()),
            )
            equal(
                "final_state",
                (canonical_hash(state), counts[key]),
                (r["head_hash"], r["actual_gradient_count"]),
            )
            finals.append(dict(r, checkpoint=local_ref(raw, r["checkpoint"])))
        else:
            equal("event_kind", "issue/release/buffer/commit/final", kind)
    equal("complete_final_set", sorted(states), sorted((r["arm"], r["seed"]) for r in finals))
    rows = []
    for key, r in issued.items():
        label = labels.get(key)
        valid = label is not None and label["eligibility"]
        y = label["y"] if label else None
        c = math.cost(r["action"], y) if valid else None
        rows.append(
            dict(
                r,
                y=y,
                cost=c,
                numerator=c,
                denominator=int(valid),
                brier=(r["probability"] - y) ** 2 if valid else None,
                eligibility=valid,
                post_first_update=r["slot"] > first.get((r["arm"], r["seed"]), len(sources)),
                false_accept=int(r["action"] == "accept" and y == 1) if valid else None,
                censor_reason="pending_delay" if label is None else None,
                exclusion_reason=None
                if valid
                else "pending_delay"
                if label is None
                else "unknown_or_public_unavailable",
                failure_reason=None,
            )
        )
    unique = {
        r["source_cluster_id"]: r
        for r in rows
        if r["arm"] == "recent64" and r["seed"] == seeds[0] and r["eligibility"]
    }
    sample = dict(
        intended=len(sources),
        completed=len(sources),
        started=len(sources),
        eligible=len(unique),
        independent=len(unique),
        excluded=sum(
            r["exclusion_reason"] != "pending_delay" and not r["eligibility"]
            for r in rows[: len(sources)]
        ),
        censored=min(20, len(sources)),
        failed=0,
        seeds_are_independent=False,
        independent_datasets=1,
        class_counts={str(y): sum(z["y"] == y for z in unique.values()) for y in (0, 1)},
    )
    progress("independent_benchmark_after", len(events), 0)
    return dict(
        rows=rows,
        independent_replay_rows=updates,
        durable_commit_rows=commits,
        final_checkpoint_rows=finals,
        checkpoint_references=[r["checkpoint"] for r in finals],
        initial_head=head,
        final_states={f"{arm}/{seed}": h for (arm, seed), h in states.items()},
        sample_size_budget=sample,
        numerical_agreement=dict(
            coefficient_max_error=coefficient_error,
            probability_max_error=probability_error,
            measured=True,
            update_count=len(updates),
            issued_count=len(rows),
            scope="Separate audit equations versus producer bytes; exact zero is possible for deterministic arithmetic, not classifier perfection.",
        ),
    )


def retention(bundle: Json, replay: Json, seal: Path, *, cold: bool = False) -> Json:
    """Seal label-free retention predictions and fit-only strata before target access."""
    protocol = json.loads(checked(bundle["audit_protocol"]).read_text())
    equal(
        "protocol_before_targets",
        (True, CONFIG, 8039),
        (protocol["frozen"], protocol["config"], protocol["identity"]),
    )
    initial = replay["initial_head"]
    fit = [math.design(initial, r)[2:] != 0 for r in bundle["fit_public"] if r["public_eligible"]]
    prevalence = np.mean(fit, axis=0)
    cuts = np.quantile([float(np.mean(prevalence[x])) for x in fit], [0.25, 0.5, 0.75])
    predictions = []
    progress("retention_benchmark_before")
    for key, head in replay["final_states"].items():
        progress(
            "retention_small_head_load_before", len(predictions), len(bundle["retention_public"])
        )
        arm, seed = key.split("/")
        for r in bundle["retention_public"]:
            x = math.design(initial, r) if r["public_eligible"] else None
            p = math.probability(head, x) if x is not None else None
            p0 = math.probability(initial, x) if x is not None else None
            overlap = float(np.mean(prevalence[x[2:] != 0])) if x is not None else None
            predictions.append(
                dict(
                    arm=arm,
                    seed=int(seed),
                    slot=r["slot"],
                    family_id=r["family_id"],
                    source_cluster_id=r["source_cluster_id"],
                    probability=p,
                    initial_probability=p0,
                    action=math.action(p),
                    initial_action=math.action(p0),
                    overlap=overlap,
                    quartile=int(np.searchsorted(cuts, overlap, side="right"))
                    if overlap is not None
                    else None,
                )
            )
        progress("retention_small_head_load_after", len(predictions), 0)
    frozen = dict(
        predictions=predictions,
        cutpoints=cuts.tolist(),
        retention_labels_opened=False,
        final_states={k: canonical_hash(h) for k, h in replay["final_states"].items()},
    )
    if cold:
        equal("retention_prediction_seal", frozen, json.loads(seal.read_text()))
    else:
        atomic_json(seal, frozen)
    progress("retention_predictions_sealed_before_targets", len(predictions), 0)
    targets = json.loads(checked(bundle["retention_target"]).read_text())["rows"]
    labels = {r["family_id"]: r["eligible_y"] for r in targets}
    equal(
        "retention_target_ids",
        sorted(r["family_id"] for r in bundle["retention_public"]),
        sorted(labels),
    )
    rows = []
    for r in predictions:
        y = labels[r["family_id"]]
        equal("retention_target_contract", True, y is None or (type(y) is int and y in (0, 1)))
        valid = y is not None and r["probability"] is not None
        c = math.cost(r["action"], y) if valid else None
        c0 = math.cost(r["initial_action"], y) if valid else None
        b = (r["probability"] - y) ** 2 if valid else None
        b0 = (r["initial_probability"] - y) ** 2 if valid else None
        rows.append(
            dict(
                r,
                y=y,
                eligibility=valid,
                cost=c,
                initial_cost=c0,
                brier=b,
                initial_brier=b0,
                cost_drift=c - c0 if valid else None,
                brier_drift=b - b0 if valid else None,
                numerator=c,
                denominator=int(valid),
                censor_reason=None,
                failure_reason=None,
                exclusion_reason=None if valid else "unknown_or_public_unavailable",
            )
        )
    unique = {r["source_cluster_id"]: r for r in rows if r["eligibility"]}
    classes = {str(y): sum(r["y"] == y for r in unique.values()) for y in (0, 1)}
    support = dict(
        intended=len(bundle["retention_public"]),
        eligible=len(unique),
        independent=len(unique),
        completed=len(bundle["retention_public"]),
        class_counts=classes,
        excluded=len(bundle["retention_public"]) - len(unique),
        censored=0,
        failed=0,
        passed=len(unique) >= CONFIG["retention_minimum"]
        and min(classes.values()) >= CONFIG["retention_per_class"],
    )
    strata = []
    for key in replay["final_states"]:
        arm, seed = key.split("/")
        for q in range(4):
            group = [
                r
                for r in rows
                if r["arm"] == arm
                and r["seed"] == int(seed)
                and r["quartile"] == q
                and r["eligibility"]
            ]
            strata.append(
                dict(
                    arm=arm,
                    seed=int(seed),
                    quartile=q,
                    numerator=sum(r["cost_drift"] for r in group),
                    denominator=len(group),
                    cost_drift=sum(r["cost_drift"] for r in group) / len(group) if group else None,
                    brier_drift=sum(r["brier_drift"] for r in group) / len(group)
                    if group
                    else None,
                    independent_count=len({r["source_cluster_id"] for r in group}),
                    descriptive_only=True,
                )
            )
    checks = [
        dict(
            check_name="retention_prediction_and_final_checkpoint_seal",
            artifact_field="retention_labels_opened",
            expected=False,
            observed=frozen["retention_labels_opened"],
            passed=True,
            path=str(seal),
            sha256=reference(seal)["sha256"],
            upstream_id="exp8039-learning-benefit-audit",
        )
    ]
    progress("retention_benchmark_after", len(rows), 0)
    return dict(
        retention_rows=rows,
        retention_support=support,
        overlap_strata_rows=strata,
        overlap_cutpoints=cuts.tolist(),
        prefreeze_access_checks=checks,
    )


def bootstrap(diff: list[float], block: int) -> Json:
    """Invert the paired .02 margin test on one fixed chronological trajectory."""
    x = np.asarray(diff, dtype=float)
    result: Json = dict(
        gain=None,
        raw_p=1.0,
        margin=CONFIG["margin"],
        block_length=min(block, len(x)),
        draws=CONFIG["draws"],
        completed_draws=0,
        censored_draws=CONFIG["draws"],
        lower=None,
        interval=[None, None],
        slot_count=len(x),
        eligible_slots=int(np.isfinite(x).sum()),
        uncertainty_scope="conditional finite-trajectory moving-block uncertainty; seeds are algorithm repetitions",
    )
    if not np.isfinite(x).any():
        return result
    length = min(block, len(x))
    rng = np.random.default_rng(CONFIG["seed"])
    starts = rng.integers(0, len(x) - length + 1, (CONFIG["draws"], int(np.ceil(len(x) / length))))
    ids = (starts[:, :, None] + np.arange(length)).reshape(CONFIG["draws"], -1)[:, : len(x)]
    samples = x[ids]
    n = np.isfinite(samples).sum(axis=1)
    draws = np.nansum(samples[n > 0], axis=1) / n[n > 0]
    mean = float(np.nanmean(x))
    errors = draws - mean
    result.update(
        gain=mean,
        raw_p=float((1 + np.sum(errors >= mean - CONFIG["margin"] - 1e-15)) / (1 + len(errors))),
        completed_draws=len(errors),
        censored_draws=CONFIG["draws"] - len(errors),
        lower=mean - float(np.quantile(errors, 0.95)),
        interval=(mean - np.quantile(errors, [0.975, 0.025])).tolist(),
    )
    return result


def compare(rows: list[Json], retained: list[Json]) -> Json:
    """Keep support, retention and local benefit gates separate from capstone Holm."""
    seeds = sorted({r["seed"] for r in rows})
    indexed = {(r["arm"], r["seed"], r["slot"]): r for r in rows}
    slots = sorted({r["slot"] for r in rows})
    unique = {
        r["source_cluster_id"]: r
        for r in rows
        if r["arm"] == "recent64" and r["seed"] == seeds[0] and r["eligibility"]
    }
    support = len(unique) >= CONFIG["stream_minimum"] and all(
        sum(r["y"] == y for r in unique.values()) >= CONFIG["stream_per_class"] for y in (0, 1)
    )
    retention_unique = {r["source_cluster_id"]: r for r in retained if r["eligibility"]}
    retention_support = len(retention_unique) >= CONFIG["retention_minimum"] and all(
        sum(r["y"] == y for r in retention_unique.values()) >= CONFIG["retention_per_class"]
        for y in (0, 1)
    )
    drift = []
    for arm in ARMS:
        for seed in seeds:
            group = [
                r for r in retained if r["arm"] == arm and r["seed"] == seed and r["eligibility"]
            ]
            c = sum(r["cost_drift"] for r in group) / len(group) if group else None
            b = sum(r["brier_drift"] for r in group) / len(group) if group else None
            drift.append(
                dict(
                    arm=arm,
                    seed=seed,
                    cost_drift=c,
                    brier_drift=b,
                    numerator=sum(r["cost_drift"] for r in group),
                    denominator=len(group),
                    passed=c is not None
                    and c <= CONFIG["cost_drift"]
                    and b <= CONFIG["brier_drift"],
                )
            )
    hypotheses, later_rows = [], []
    for comparator in ("cumulative", "newest16", "frozen_no_write"):
        diff, changed = [], set()
        false_accept_difference = {str(seed): 0 for seed in seeds}
        for slot in slots:
            pairs = [
                (indexed[("recent64", s, slot)], indexed[(comparator, s, slot)]) for s in seeds
            ]
            post = all(
                a["post_first_update"]
                and (b["post_first_update"] or comparator == "frozen_no_write")
                for a, b in pairs
            )
            if not post:
                continue
            valid = all(a["eligibility"] and b["eligibility"] for a, b in pairs)
            diff.append(
                float(np.mean([b["cost"] - a["cost"] for a, b in pairs])) if valid else float("nan")
            )
            if valid:
                for a, b in pairs:
                    false_accept_difference[str(a["seed"])] += a["false_accept"] - b["false_accept"]
                if diff[-1] > 0 and any(a["action"] != b["action"] for a, b in pairs):
                    changed.add(pairs[0][0]["source_cluster_id"])
            for a, b in pairs:
                later_rows.append(
                    dict(
                        a,
                        comparator=comparator,
                        comparator_cost=b["cost"],
                        paired_gain=b["cost"] - a["cost"] if valid else None,
                    )
                )
        tests = [bootstrap(diff, block) for block in CONFIG["blocks"]]
        retention_gate = retention_support and all(
            r["passed"] for r in drift if r["arm"] in ("recent64", comparator)
        )
        gates = dict(
            stream_support=support,
            later_support=tests[0]["eligible_slots"] >= CONFIG["later_minimum"],
            changed_natural_decisions=len(changed) >= CONFIG["changed_minimum"],
            no_additional_false_accepts=all(n <= 0 for n in false_accept_difference.values()),
            retention=retention_gate,
            gain_margin=tests[0]["gain"] is not None and tests[0]["gain"] >= CONFIG["margin"],
            conditional_margin_test=tests[0]["raw_p"] < 0.05
            and tests[0]["lower"] is not None
            and tests[0]["lower"] > CONFIG["margin"],
        )
        hypotheses.append(
            dict(
                tests[0],
                hypothesis="H3" if comparator == "cumulative" else "secondary_" + comparator,
                comparator=comparator,
                treatment="recent64",
                gates=gates,
                local_passed=all(gates.values()),
                beneficial_changed_groups=len(changed),
                false_accept_difference_by_seed=false_accept_difference,
                capstone_family_p=tests[0]["raw_p"] if all(gates.values()) else 1.0,
                block_sensitivity=tests[1:],
                capstone_family=["H1", "H2", "H3"],
                holm_alpha=0.05,
                family_credit=False,
                capstone_instruction="Combine H3 margin p with H1/H2 using Holm .05; failed science gates supply p=1.",
            )
        )
    return dict(
        primary_hypothesis_results=hypotheses,
        later_loss_rows=later_rows,
        retention_drift_rows=drift,
        effective_independent_streams=1,
        learning_benefit_score=int(hypotheses[0]["local_passed"]),
    )


def controls() -> Json:
    """Known improvement and zero headroom check the reducer without natural credit."""
    values = math.controls()
    return dict(values, passed=all(r["passed"] for r in values.values() if isinstance(r, dict)))

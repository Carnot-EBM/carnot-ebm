"""REQ-VERIFY-8109: source-level decisions prevent seeds from creating evidence."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]


def hypothesis(rows: list[Json], name: str) -> Json:
    """Keep original missing slots in resampling so selecting successes cannot improve a bound."""
    groups: dict[str, list[Json]] = defaultdict(list)
    masks = []
    for row in rows:
        valid = row.get("status") == "completed" and all(
            k in row
            for k in (
                "source_id",
                "slot",
                "label",
                "control_cost",
                "treatment_cost",
                "beneficial",
                "extra_false_accept",
                "brier_increase",
                "equality_control",
                "other_control_increase",
            )
        )
        masks.append(
            dict(
                source_id=row.get("source_id"),
                slot=row.get("slot"),
                eligible=valid,
                exclusion_reason=None if valid else "missing_or_invalid_pair",
            )
        )
        if valid:
            groups[str(row["source_id"])].append(row)
    units = list(groups.values())
    support = {str(label): sum(g[0]["label"] == label for g in units) for label in (0, 1)}
    gains = np.array([np.mean([r["control_cost"] - r["treatment_cost"] for r in g]) for g in units])
    blocks = [
        [i for i, g in enumerate(units) if start <= g[0]["slot"] < start + 16]
        for start in range(65, 257, 16)
    ]
    equality = all(r["equality_control"] for g in units for r in g)
    supported = (
        len(units) >= (96 if name == "H1" else 128)
        and min(support.values()) >= (12 if name == "H1" else 8)
        and (name == "H1" or sum(bool(b) for b in blocks) >= 8)
    )
    bound = None
    valid_draws = 0
    sensitivities: Json = {}
    if supported and equality:
        rng = np.random.default_rng(701)
        if name == "H1":
            draws = gains[rng.integers(len(units), size=(10000, len(units)))].mean(axis=1)
            valid_draws = 10000
            bound = float(np.quantile(draws, 0.025))
        else:
            # Each window retains absent slots. Empty resamples remain invalid draws.
            for width in (8, 16, 32):
                slot_gains = {g[0]["slot"]: gains[i] for i, g in enumerate(units)}
                windows = [
                    [slot_gains[s] for s in range(start, start + width) if s in slot_gains]
                    for start in range(65, 257 - width + 1)
                ]
                totals = np.array([sum(w) for w in windows])
                counts = np.array([len(w) for w in windows])
                choices = rng.integers(len(windows), size=(10000, 192 // width))
                denominators = counts[choices].sum(axis=1)
                selected = denominators > 0
                draw_values = totals[choices].sum(axis=1)[selected] / denominators[selected]
                sensitivities[str(width)] = dict(
                    valid_draws=int(selected.sum()),
                    lower_bound=float(np.quantile(draw_values, 0.025)),
                )
            valid_draws = sensitivities["16"]["valid_draws"]
            bound = sensitivities["16"]["lower_bound"] if valid_draws >= 9500 else None
    benefit = sum(all(r["beneficial"] for r in g) for g in units)
    harms = sum(any(r["extra_false_accept"] for r in g) for g in units)
    calibration = all(r["brier_increase"] <= 0.01 for g in units for r in g)
    controls = all(r["other_control_increase"] <= 0.02 for g in units for r in g)
    signal = int(
        bound is not None
        and bound > 0.02
        and benefit >= 5
        and harms == 0
        and calibration
        and controls
    )
    return dict(
        hypothesis=name,
        status="completed" if supported and equality else "blocked",
        source_count=len(units),
        independent_source_count=len(units),
        class_support=support,
        sample_masks=masks,
        excluded_count=sum(not m["eligible"] for m in masks),
        observed_gain=float(gains.mean()) if len(gains) else None,
        lower_bound=bound,
        valid_draws=valid_draws,
        bootstrap_draws=10000,
        block_sensitivities=sensitivities,
        beneficial_sources=benefit,
        extra_false_accepts=harms,
        equality_control_passed=equality,
        development_signal_score=signal,
        exposure="previously exposed development",
        precision_limit="Nominal bootstrap coverage is not independent confirmation.",
    )


def reduce(data: Json) -> Json:
    """A finished accounting task may be blocked scientifically without scheduling retries."""
    tasks, dispositions, primaries = data["tasks"], data["dispositions"], data["primaries"]
    h1 = hypothesis(
        primaries.get(tasks[4]["id"], {}).get("decision_rows", [])
        if dispositions[4]["eligible"]
        else [],
        "H1",
    )
    learning = primaries.get(tasks[7]["id"], {}) if dispositions[7]["eligible"] else {}
    h2 = hypothesis(learning.get("decision_rows", []), "H2")
    retention = learning.get("retention_rows", [])
    retention_groups = {
        str(row["source_id"]): row for row in retention if row.get("status") == "completed"
    }
    retention_support = {
        str(i): sum(row["label"] == i for row in retention_groups.values()) for i in (0, 1)
    }
    retention_safe = (
        len(retention_groups) >= 48
        and min(retention_support.values()) >= 8
        and all(
            row["retention_cost_increase"] <= 0.02
            and row["retention_brier_increase"] <= 0.01
            and not row["extra_false_accept"]
            for row in retention_groups.values()
        )
    )
    h2["retention"] = dict(
        source_count=len(retention_groups),
        class_support=retention_support,
        safe=retention_safe,
        precision="64 exposed sources give descriptive precision only.",
    )
    h2["development_signal_score"] *= int(retention_safe)
    blocked = (
        bool(data["failures"])
        or any(not row["eligible"] for row in dispositions)
        or any(h["status"] == "blocked" for h in (h1, h2))
    )
    state = (
        "blocked"
        if blocked
        else "positive"
        if h1["development_signal_score"] or h2["development_signal_score"]
        else "null"
    )
    own = dict(
        task_id=tasks[-1]["id"],
        source_id="owned_capstone",
        unit_id=tasks[-1]["id"],
        arm="task_disposition",
        condition="independent_accounting",
        issued_state="complete_" + state,
        metric="science_inputs_available",
        numerator=int(not blocked),
        denominator=1,
        raw_numerator=int(not blocked),
        raw_denominator=1,
        status="completed",
        eligible=not blocked,
        excluded=blocked,
        completed=True,
        failed=False,
        censored=False,
        honest_verdict="complete_" + state + "_v701_capstone",
        verdict_class=state,
        exclusion_reason="upstream_science_unavailable" if blocked else None,
    )
    rows = [*dispositions, own]
    retirements = []
    for task, row in zip(tasks, rows, strict=True):
        for prior in task["prior_failures"]:
            same = prior["verdict"] == row["honest_verdict"]
            retirements.append(
                dict(
                    prior,
                    task_id=task["id"],
                    observed_verdict=row["honest_verdict"],
                    same_verdict=same,
                    scope=task["id"],
                    input_path=row.get("path"),
                    input_sha256=row.get("sha256"),
                    task_config_sha256=canonical_hash(task),
                    stated_addressed_by=prior["addressed_by"],
                    remaining_data_requirements="Independent human targets, current paired decisions, retention and matched complete-service costs.",
                    retire_exact_configuration=bool(
                        same
                        and prior["retire_if_same_verdict"]
                        and row["eligible"]
                        and row["verdict_class"] == "null"
                    ),
                    environmental_block_retires_method_family=False,
                    reopen_condition="Changed mechanism or separately acquired independent human-labeled evidence.",
                )
            )
    next_evidence = (
        "resolve upstream validation and absent outputs once; no unchanged scientific retry"
        if blocked
        else "acquire separately an independent human-labeled cohort before generalization"
        if state == "positive"
        else "retire this exact unchanged radial configuration; preserve its null"
    )
    return dict(
        honest_verdict=own["honest_verdict"],
        verdict_class=state,
        rows=rows,
        task_dispositions=rows,
        task_contract=tasks,
        retirement_candidates=retirements,
        H1=h1,
        H2=h2,
        h1_development_signal_score=h1["development_signal_score"],
        h2_development_signal_score=h2["development_signal_score"],
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        verifier_is_oracle=0,
        claim_scope=0,
        exposure_scope=0,
        capstone_ready_score=0,
        science_ready_score=int(not blocked),
        intended_count=13,
        eligible_count=sum(row["eligible"] for row in rows),
        independent_count=0,
        completed_count=13,
        excluded_count=sum(row["excluded"] for row in rows),
        censored_count=0,
        failed_count=sum(row["failed"] for row in rows),
        sample_size_budget=dict(tasks=13, H1=128, H2=192, retention=64, seeds_are_sources=False),
        gate_check_summary=data["failures"],
        preconditions_checked=data["preconditions"],
        independent_reductions=data["independent_reductions"],
        gap_decisions={
            name: dict(closed=False, requirements=reqs, reopen_condition=condition)
            for name, reqs, condition in (
                (
                    "independent_verifier",
                    ["FR-06", "FR-12"],
                    "Acquire independent human-labeled sources and qualify useful typed decisions.",
                ),
                (
                    "lifelong_learning",
                    ["FR-11"],
                    "Show later causal benefit and retained safety on independent sources.",
                ),
                (
                    "whole_service_deployment",
                    ["FR-05", "FR-08", "FR-09", "FR-10", "NFR-01"],
                    "Measure matched deployed service including acquisition, feedback, transport and recovery; require >=10x.",
                ),
            )
        },
        whole_service_deployment=dict(
            ready=False,
            reason="Host costs and fixture arithmetic cannot establish deployed useful service.",
        ),
        scope_reduction_compliance=dict(
            next_evidence_decision=next_evidence,
            data_pivot_is_headline=False,
            original_independent_gaps_preserved=True,
            repaired_fitted_baseline="Immutable nonzero fitted weights, zero padding, empirical cost/Brier guards.",
            unmatched_effective_capacity="Equal installed center counts do not imply equal effective rank or capacity.",
            publication_corrigenda_preserved=True,
        ),
    )

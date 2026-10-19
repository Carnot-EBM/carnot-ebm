"""REQ-REPORT-8498: a complete inventory grants no missing scientific benefit."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import v726_capstone_report as old
from carnot.reporting import v729_capstone_report as prior
from carnot.reporting import v730_capstone_evidence as e
from carnot.reporting import v730_contract_methods as methods

Json = dict[str, Any]
_BUILD = old.build
CLAIMS = {
    key: (
        number + 14,
        decision,
        condition.replace("V729", "V730").replace("Exp8475/8476", "Exp8489/8490"),
    )
    for key, (number, decision, condition) in prior.CLAIMS.items()
    if key != "objective_learning"
}
CLAIMS["objective_learning"] = (
    8491,
    "continue_after_current_receipt_qualification",
    "Repair current Exp8489/8490 bound receipts, then measure later Brier/typed losses versus frozen and shuffled controls and all five simultaneous retention upper95<=0.01 windows. Missing prerequisites do not refute learning.",
)
CLAIMS["source_information"] = (
    8488,
    "continue_only_with_disjoint_independent_labels",
    "Preserve positive corrected paired Brier bounds and typed-loss noninferiority; independently adjudicate disjoint questions with resolved exposure before claiming generalization.",
)
CLAIMS["append_service"] = (
    8494,
    "repair_owned_validation_before_speed_claim",
    "Repair the exact strict-mypy failure, retain primary paired clocks and original thresholds, and independently qualify the terminal artifact. Require PRIMARY lower95>1 and p95<=1.05; tenfold lower95>=10 is separate. Matched Rust/Python NFR-01 remains open.",
)


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Preserve measured historical nulls and report unqualified current clocks as descriptive only."""
    with patch.multiple(old, e=e, CLAIMS=CLAIMS):
        value = dict(_BUILD(work, receipts, raw, output))
    value.update(
        experiment_id=8498,
        run_date="20261014",
        random_seed=7308498,
        honest_verdict=value["honest_verdict"].replace("v726", "v730"),
        invocation_argv=work["invocation_argv"],
        canonical_tasks_sha256=work["authority"].get("canonical_tasks_sha256"),
        observed_active_tasks_sha256=work["authority"].get("observed_active_tasks_sha256"),
        methodology_note="Read-only aggregation with independent shipped primitive replay in bounded CPU workers. Public-label source gains are exposed retrospective evidence. Later Brier learning remains unmeasured, disqualified append clocks are descriptive, and historical journal timing remains a separate null. No model loads, generations, device calls or generator weight changes. Generalization and matched Rust/Python NFR-01 remain open.",
    )
    value["rows"][-1]["experiment_id"] = 8498
    value["acceptance_gates"]["all_slots_accounted"] = len(value["rows"]) == 14
    value["scientific_claims"]["objective_learning"]["retention_windows"] = [
        dict(
            window=i,
            metric=None,
            missing_reason="Exp8491 pre-gated; later Brier/typed-loss comparison unmeasured",
        )
        for i in range(5)
    ]
    for name, metric in [
        ("source_information", "source_information_benefit_score"),
        ("objective_learning", "exploratory_learning_benefit_score"),
        ("append_service", "pooled_speedup"),
    ]:
        claim = value["scientific_claims"][name]
        claim["qualified_retrospective_benefit"] = (
            claim["metrics"].get(metric) if claim["qualified_producer"] else None
        )
        claim["missing_reason"] = (
            None if claim["qualified_producer"] else "required_producer_absent_or_unqualified"
        )
    value["scientific_claims"]["append_service"].update(
        PRIMARY_only=True,
        NFR_01_status="open_matched_Rust_Python_required",
        FR_05_08_status="open_matched_Rust_parity_required",
    )
    value["scientific_claims"]["V726_preserved"] = {
        str(r["experiment_id"]): dict(
            qualified=r["eligible"],
            original_verdict=r["honest_verdict"],
            metrics=r["selected"],
            primary=r["primary_reference"],
        )
        for r in work["historical_slots"]
        if r["experiment_id"] < 8457
    }
    predecessor = e.load(
        next(r for r in work["predecessors"] if Path(r["path"]).name.startswith("experiment_8470_"))
    )
    journal = predecessor["scientific_claims"]["durable_service"]
    value["scientific_claims"]["historical_journal_null"] = dict(
        producer_experiment_id=8466,
        complete_journal_speedup=journal["complete_journal_speedup"],
        speedup_ci95=journal["metrics"]["speedup_ci95"],
        journal_speed_benefit_score=journal["metrics"]["journal_speed_benefit_score"],
        source=next(
            r for r in work["predecessors"] if Path(r["path"]).name.startswith("experiment_8470_")
        ),
        decision="retire_unchanged_journal_timing; current_append_requires_qualification",
    )
    value["scientific_claims"]["historical_service_reductions"] = {
        str(r["experiment_id"]): dict(
            qualified=r["eligible"],
            original_verdict=r["honest_verdict"],
            metrics=r["selected"],
            primary=r["primary_reference"],
        )
        for r in work["historical_slots"]
        if r["experiment_id"] >= 8457
    }
    value["scientific_claims"]["objective_stream"] = {
        str(r["experiment_id"]): dict(
            qualified=r["eligible"],
            original_verdict=r["honest_verdict"],
            metrics=r["selected"],
            primary=r["primary_reference"],
            missing_reason="later_learning_comparison_not_measured",
        )
        for r in work["slots"]
        if r["experiment_id"] == 8490
    }
    v729ref = next(
        r for r in work["predecessors"] if Path(r["path"]).name.startswith("experiment_8484_")
    )
    v729 = e.load(v729ref)
    value["scientific_claims"]["V729_preserved"] = dict(
        source=v729ref,
        producer_count=v729["producer_count"],
        cascade_skip_count=v729["cascade_skip_count"],
        original_verdict=v729["honest_verdict"],
        decision="preserve_missing_authority_and_unmeasured_science; no_hypothesis_retirement",
    )
    value["scientific_claims"]["V728_preserved"] = dict(
        source=next(
            r for r in work["predecessors"] if Path(r["path"]).name.startswith("experiment_8470_")
        ),
        journal_equivalence=predecessor["scientific_claims"]["journal_durability"],
        decision="preserve_qualified_journal_equivalence_and_separate_service_null",
    )
    value["qualification_repair_audit"] = [
        dict(
            producer=r["task_id"],
            primary=r["primary_reference"],
            qualified=r["eligible"],
            disposition=r["disposition"],
            original_rejection=methods.protocol()["validation_repairs"][key],
            metrics=r["selected"],
        )
        for number, key in [
            (8486, "original_producer_pin_rejection"),
            (8489, "changed_reference_rejection"),
            (8496, "panel_registry_rejection"),
        ]
        for r in work["slots"]
        if r["experiment_id"] == number
    ]
    for row in [*work["slots"], *work["historical_slots"]]:
        if row["verdict_class"] in {"blocked", "disqualified", "null"} or not row["eligible"]:
            value["retirement_recommendations"].append(
                dict(
                    task_id=row["task_id"],
                    predecessor=row["primary_reference"],
                    exact_verdict=row["honest_verdict"],
                    recommendation="retire_exact_unchanged_attempt",
                    hypothesis_retired=False,
                    reopening_condition="Repair the named qualification failure, supply changed external evidence or change the measured mechanism; preserve original thresholds.",
                )
            )
    value["cited_upstream_artifacts"] += [
        dict(r["primary_reference"], producer=r["task_id"], imported_fields=list(r["selected"]))
        for r in work["historical_slots"]
    ]
    value["field_principles"].update(
        {
            k: "Bind actual invocation and producer hashes; administrative completeness grants no scientific benefit."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["reproducibility_checksum"] = e.checksum(value)
    return value

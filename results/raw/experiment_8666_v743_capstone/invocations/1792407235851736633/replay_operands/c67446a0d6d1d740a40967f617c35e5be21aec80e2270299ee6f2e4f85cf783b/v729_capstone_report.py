"""REQ-REPORT-8484: qualified reconciliation is separate from measured scientific benefit."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import v726_capstone_report as old
from carnot.reporting import v729_capstone_evidence as e
from carnot.reporting import v729_contract_methods as methods

Json = dict[str, Any]
CLAIMS = {
    "source_qualification": (
        8472,
        "repair_exact_missing_pin_rejection",
        "Exercise original Exp8458 line554 producer-pin rejection without changing its thresholds; authenticate complete independent V729 authority.",
    ),
    "scalar_source_fit": (
        8473,
        "continue_only_with_qualified_controls",
        "Freeze matched logistic/spline scalar, full, no-source and permuted-source decisions under the unchanged source roles.",
    ),
    "source_information": (
        8474,
        "continue_changed_qualified_information_comparison",
        "Both corrected paired Brier lower95 bounds>0 and typed-loss noninferiority; independent claims additionally require disjoint independently adjudicated questions and resolved exposure.",
    ),
    "objective_qualification": (
        8475,
        "repair_exact_changed_reference_rejection",
        "Exercise original Exp8461 line388 changed-reference rejection; retain derivatives and durable transitions and all historical assertions in bounded exact shards.",
    ),
    "objective_learning": (
        8477,
        "continue_only_after_qualified_objective_learning",
        "Qualified Exp8475/8476; matched later Brier/log-loss/frozen/shuffled comparisons with both Brier lower95>0, no typed-loss increase, simultaneous retention upper95<=0.01 in all five windows.",
    ),
    "publication_cost_model": (
        8478,
        "continue_changed_publication_mechanism",
        "Use disjoint measured publication clocks; stable append mechanism must preserve file/namespace barriers, not merely remove required synchronization.",
    ),
    "append_durability": (
        8479,
        "qualify_new_append_protocol",
        "Require exact acknowledged-prefix recovery, idempotence and equivalent transitions before append speed measurements; process exits do not certify physical power loss.",
    ),
    "append_service": (
        8480,
        "continue_only_after_append_qualification",
        "PRIMARY matched complete wall lower95>1 and p95 ratio<=1.05; separate tenfold lower95>=10. Require matched Rust/Python implementations for NFR-01 and FR-05/08.",
    ),
    "current_qwen": (
        8481,
        "blocked_optional_runtime_and_source_prerequisites",
        "Authenticate changed CUDA context/copy repair and qualified source controls before bounded optional Qwen invocations. CPU science is independent.",
    ),
    "arc_cross_game_supervision": (
        8482,
        "repair_remaining_owned_qualification_before_panel",
        "Preserve original Exp8468 line350 rejection and all assertions; three overlapping adapter-withheld games and five resolved firings per shared arm cell before refinement. Keep censoring and zero solve credit.",
    ),
    "board_operations": (
        8483,
        "blocked_installed_append_execution",
        "KV260 quadratic Ising k<=5; PolarFire board-local Linux CPU hashes; GateMate exact missing source bytes, dated cable/port/power change, IDCODE0x20000001, n16 flash and sample/hash parity. Installed append and complete transfer clocks remain required.",
    ),
}
_BUILD = old.build


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Missing prerequisites retire an unchanged attempt; they do not retire unmeasured hypotheses."""
    with patch.multiple(old, e=e, CLAIMS=CLAIMS):
        value = dict(_BUILD(work, receipts, raw, output))
    value.update(
        experiment_id=8484,
        run_date="20261013",
        random_seed=7298484,
        honest_verdict=value["honest_verdict"].replace("v726", "v729"),
        invocation_argv=work["invocation_argv"],
        canonical_tasks_sha256=work["authority"].get("canonical_tasks_sha256"),
        observed_active_tasks_sha256=work["authority"].get("observed_active_tasks_sha256"),
        methodology_note="Read-only aggregation and independent primitive replay. Public-label science is retrospective. Missing source/Brier/append comparisons remain unmeasured. Zero model or device calls; frozen generator weights. Historical journal null and current append evidence are separate; generalization and matched Rust/Python NFR-01 remain open.",
    )
    value["rows"][-1]["experiment_id"] = 8484
    value["acceptance_gates"]["all_slots_accounted"] = len(value["rows"]) == 14
    for claim in value["scientific_claims"].values():
        if "qualified_producer" in claim:
            claim["missing_reason"] = (
                None if claim["qualified_producer"] else "required_producer_absent_or_unqualified"
            )
    learning = value["scientific_claims"]["objective_learning"]
    learning["retention_windows"] = learning["metrics"].get(
        "per_window_retention",
        [
            dict(window=i, metric=None, missing_reason="Exp8477 retention comparison not measured")
            for i in range(5)
        ],
    )
    for name, metric in [
        ("source_information", "source_information_benefit_score"),
        ("objective_learning", "exploratory_learning_benefit_score"),
        ("append_service", "pooled_speedup"),
    ]:
        claim = value["scientific_claims"][name]
        claim["qualified_retrospective_benefit"] = (
            claim["metrics"].get(metric) if claim["qualified_producer"] else None
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
            decision="preserve_retrospective_positive_learning_null_and_arithmetic_ceiling",
        )
        for r in work["historical_slots"]
        if r["experiment_id"] < 8457
    }
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
    value["exposed_question_roles"] = methods.protocol()["partition"]
    value["qualification_repair_audit"] = [
        dict(
            producer=r["task_id"],
            primary=r["primary_reference"],
            disposition=r["disposition"],
            qualified=r["eligible"],
            original_rejection=methods.protocol()["validation_repairs"][key],
            missing_reason=None if r["eligible"] else "current_qualification_absent_or_failed",
        )
        for number, key in [
            (8472, "original_producer_pin_rejection"),
            (8475, "changed_reference_rejection"),
            (8482, "panel_registry_rejection"),
        ]
        for r in work["slots"]
        if r["experiment_id"] == number
    ]
    predecessor = e.load(work["predecessors"][-1])
    journal = predecessor["scientific_claims"]["durable_service"]
    value["scientific_claims"]["historical_journal_null"] = dict(
        source=work["predecessors"][-1],
        producer_experiment_id=8466,
        complete_journal_speedup=journal["complete_journal_speedup"],
        speedup_ci95=journal["metrics"]["speedup_ci95"],
        journal_speed_benefit_score=journal["metrics"]["journal_speed_benefit_score"],
        decision="retire_unchanged_journal_timing; no_current_append_speed_measurement",
    )
    value["scientific_claims"]["V728_preserved"] = dict(
        source=work["predecessors"][-1],
        outcomes=[
            dict(
                experiment_id=r["experiment_id"],
                disposition=r["disposition"],
                honest_verdict=r.get("honest_verdict"),
                primary=r.get("primary_reference"),
            )
            for r in predecessor["rows"]
        ],
        decision="retain_qualification_failures_source_objective_absence_and_directory_sync_cost",
    )
    value["cited_upstream_artifacts"] += [
        dict(r["primary_reference"], producer=r["task_id"], imported_fields=list(r["selected"]))
        for r in work["historical_slots"]
    ]
    value["historical_model_provenance"] += [
        dict(
            producer=r["task_id"],
            provenance=methods.history.model_provenance(
                e.load(r["primary_reference"]), r["primary_reference"]
            ),
            scope="cached original evidence",
        )
        for r in work["historical_slots"]
    ]
    for row in [*work["slots"], *work["historical_slots"]]:
        if (row.get("producer_executed") or row["disposition"] == "pre_gate_receipt") and row[
            "verdict_class"
        ] in {
            "blocked",
            "disqualified",
            "null",
        }:
            value["retirement_recommendations"].append(
                dict(
                    task_id=row["task_id"],
                    predecessor=row["primary_reference"],
                    exact_verdict=row["honest_verdict"],
                    recommendation="retire_exact_unchanged_attempt",
                    hypothesis_retired=False,
                    reopening_condition="Repair the named qualification or authority defect, supply changed external evidence, or change the measured mechanism; preserve original thresholds.",
                )
            )
    value["field_principles"].update(
        {
            k: "Bind actual invocation and original producer hashes; missing is distinct from zero and administrative completeness grants no science benefit."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["reproducibility_checksum"] = e.checksum(value)
    return value

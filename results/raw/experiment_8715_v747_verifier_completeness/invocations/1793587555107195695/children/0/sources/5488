"""REQ-REPORT-8470: complete accounting does not imply favorable scientific evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import v726_capstone_report as old
from carnot.reporting import v728_capstone_evidence as e

Json = dict[str, Any]
CLAIMS = {
    "source_information": (
        8460,
        "continue_after_source_qualification",
        "Repair Exp8458 qualification; freeze scalar logistic/spline, full, no-source and permuted-source controls. Both corrected Brier lower bounds>0 and typed-loss noninferiority are required; independent generalization needs a disjoint adjudicated cohort.",
    ),
    "objective_qualification": (
        8461,
        "repair_derivative_and_durable_transition_qualification",
        "Repair the owned Exp8461 validation defect. Recompute Brier derivatives and durable transitions independently before learning.",
    ),
    "objective_learning": (
        8463,
        "continue_changed_objective_after_qualification",
        "Require qualified Exp8461 and matched Brier/log-loss/frozen/shuffled later-question losses. Both later Brier lower bounds>0, no typed-loss increase, simultaneous retention upper95<=0.01 across all five windows.",
    ),
    "durable_service": (
        8466,
        "retire_exact_unchanged_cost_repeat",
        "Conditional engineering benefit requires complete paired wall lower95>1 and journal/snapshot p95<=1.05; tenfold lower95>=10 is separate. Change measured service work before repeating; Rust/Python NFR-01 and FR-05/08 parity need independent matched implementations.",
    ),
    "journal_durability": (
        8465,
        "retain_process_exit_durability_scope",
        "Retain acknowledged idempotent recovery and causal snapshot/journal equivalence. Physical power-loss durability is not certified by process exits.",
    ),
    "current_qwen": (
        8467,
        "continue_only_after_qualified_runtime_repair",
        "Require authenticated CUDA repair, context/copy parity and source controls before bounded optional current-Qwen calls.",
    ),
    "arc_cross_game_supervision": (
        8468,
        "repair_panel_qualification_before_refinement",
        "Qualify the frozen fresh four-game panel; require three overlapping adapter-withheld games and five resolved firings per shared arm cell. Keep censored outcomes and zero solve credit.",
    ),
    "board_operations": (
        8469,
        "blocked_execution_evidence",
        "KV260 quadratic Ising k<=5; PolarFire authenticated board-local Linux CPU dispatch. GateMate requires exact source bytes, dated cable/port/power change, IDCODE0x20000001, n16 flash and sample/hash parity. Complete board/transport clocks are required.",
    ),
}


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Original reducers retain thresholds; missing producers never receive zero scientific effect."""
    with patch.multiple(old, e=e, CLAIMS=CLAIMS):
        value = dict(_BUILD(work, receipts, raw, output))
    value.update(
        experiment_id=8470,
        run_date="20261013",
        random_seed=7288470,
        honest_verdict=value["honest_verdict"].replace("v726", "v728"),
        invocation_argv=work["invocation_argv"],
        methodology_note="Read-only aggregation with pinned fresh-process primitive replay. Public-label effects remain retrospective; no model, board or game is invoked. Missing source/objective science is not zero. Journal service is engineering evidence only; FR-05/08 and NFR-01 remain open. Historical FoVer gates do not establish V728 success.",
    )
    value["rows"][-1]["experiment_id"] = 8470
    value["acceptance_gates"]["all_slots_accounted"] = len(value["rows"]) == 14
    for claim in value["scientific_claims"].values():
        if "qualified_producer" in claim:
            claim["missing_reason"] = (
                None if claim["qualified_producer"] else "required_producer_not_qualified"
            )
    for name in ["source_information", "objective_learning"]:
        claim = value["scientific_claims"][name]
        claim.update(
            qualified_retrospective_benefit=None, missing_reason="required_producer_not_qualified"
        )
    value["scientific_claims"]["objective_learning"]["retention_windows"] = [
        dict(window=i, metric=None, missing_reason="Exp8463 objective audit not measured")
        for i in range(5)
    ]
    service = value["scientific_claims"]["durable_service"]
    metric = service["metrics"]
    service.update(
        complete_journal_speedup=metric.get("pooled_speedup")
        if service["qualified_producer"]
        else None,
        durability_qualified=value["scientific_claims"]["journal_durability"]["qualified_producer"],
        FR_05_08_status="open_matched_Rust_parity_required",
        NFR_01_status="open_Rust_versus_Python_required",
    )
    value["scientific_claims"]["V726_preserved"] = {
        str(row["experiment_id"]): dict(
            qualified=row["eligible"],
            original_verdict=row["honest_verdict"],
            metrics=row["selected"],
            primary=row["primary_reference"],
            decision="keep_retrospective_positive_learning_null_and_arithmetic_ceiling",
        )
        for row in work["historical_slots"]
    }
    predecessor = e.load(work["predecessors"][3])
    value["scientific_claims"]["V727_preserved"] = dict(
        source=work["predecessors"][3],
        producer_count=predecessor["producer_count"],
        skipped_count=predecessor["pregate_count"] + predecessor["cascade_skip_count"],
        verdict_class=predecessor["verdict_class"],
        decision="no_source_objective_or_journal_science_measured",
    )
    value["cited_upstream_artifacts"] += [
        dict(r["primary_reference"], producer=r["task_id"], imported_fields=list(r["selected"]))
        for r in work["historical_slots"]
    ]
    value["historical_model_provenance"] += [
        dict(
            producer=r["task_id"],
            provenance=r.get("historical_model_provenance", []),
            scope="sealed cached outputs only",
        )
        for r in work["historical_slots"]
    ]
    for row in [*work["slots"], *work["historical_slots"]]:
        if row.get("producer_executed") and (
            row["verdict_class"] in {"blocked", "disqualified", "null"}
            or row["experiment_id"] == 8437
        ):
            value["retirement_recommendations"].append(
                dict(
                    task_id=row["task_id"],
                    predecessor=row["primary_reference"],
                    exact_verdict=row["honest_verdict"],
                    recommendation="retire_exact_unchanged_repeat",
                    hypothesis_retired=False,
                    reopening_condition="Repair named qualification defect or change the measured mechanism; preserve independent operands and all original thresholds.",
                )
            )
    value["field_principles"].update(
        {
            k: "Bind actual invocation and sealed original producer scope; administrative completeness grants no scientific benefit."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["reproducibility_checksum"] = e.checksum(value)
    return value


_BUILD = old.build

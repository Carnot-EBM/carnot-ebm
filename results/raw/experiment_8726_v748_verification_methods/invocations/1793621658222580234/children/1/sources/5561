"""REQ-REPORT-8564: scientific nulls survive blocked milestones and owned failures."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import v726_capstone_report as shipped
from carnot.reporting import v735_capstone_evidence as e
from carnot.reporting.v734_capstone_report import CLAIMS as PRIOR_CLAIMS

Json = dict[str, Any]
CLAIMS = {
    name: (number + delta, decision, condition)
    for (name, (number, decision, condition)), delta in zip(
        PRIOR_CLAIMS.items(), [13, 13, 13, 13], strict=True
    )
}


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Only qualified producer bytes can support their specific claim and release decision."""
    with patch.multiple(shipped, e=e, CLAIMS=CLAIMS):
        value = dict(shipped.build(work, receipts, raw, output))
    prior = e.load(work["predecessors"][0])
    qualified = [r for r in work["slots"] if r["eligible"] and r["verdict_class"] == "null"]
    release = next(r for r in work["slots"] if r["experiment_id"] == 8553)
    value.update(
        experiment_id=8564,
        run_date="20261016",
        random_seed=7358564,
        honest_verdict=f"complete_{value['verdict_class']}_v735_capstone",
        intended_count=13,
        task_count=13,
        censored_count=13 - value["completed_count"],
        failed_count=value["disqualified_producer_count"],
        administrative_complete=len(value["rows"]) == 13,
        milestone_status="scheduling_accounted_scientific_milestone_unqualified",
        canonical_tasks_sha256=work["contract_digest"],
        task_contract_rows=work["authority"]["contract_rows"],
        preserved_nulls=prior["preserved_nulls"],
        qualified_scientific_results=prior["preserved_nulls"] + qualified,
        quarantined_inputs=prior["quarantined_inputs"]
        + [r for r in work["slots"] if r["verdict_class"] == "disqualified"],
        blocked_inputs=[r for r in work["slots"] if r["verdict_class"] == "blocked"],
        release_admissibility_decision=release["selected"].get("admission_decision")
        if release["eligible"]
        else None,
        release_admissibility_source=release["primary_reference"],
        conditional_exposed_benefits=dict(
            status="no_new_qualified_learning_or_mask_benefit", generalization_credit=0
        ),
        deterministic_claim_samples=work["samples"],
        publication_authorized=False,
        publication_gate_result=work["publication"],
        g_gate_rows=[dict(gate=k, result=v) for k, v in work["publication"]["gates"].items()],
        next_evidence_requirements=value["next_evidence_conditions"]
        | dict(
            fresh_data="Disjoint independently adjudicated human-label question clusters; HALLMARK constructed labels and FaithBench are exposed.",
            runtime="Causal library/environment repair with bounded context/copy evidence.",
            physical_hardware="Compatible installed operations, complete transport clocks and physical GateMate source/cable/IDCODE/flash/smoke evidence.",
        ),
        architecture_boundary_notes=prior["architecture_boundary_notes"],
        research_program_priorities=[
            "Repair the owned canonical-counter validation defect before learning measurements.",
            "Retain qualified complete native costs and speed null; require a changed nonkernel path before repeating.",
            "Qualify exact ARC execution receipts before any live mask claim.",
            "Acquire independent human labels only under a separately authorized study.",
        ],
        methodology_note="No-model aggregation from exact sealed V734 and V735 terminal bytes. Thirteen scheduling rows do not establish scientific success. Small original learning and ARC samples plus stored complete-cost operands are diagnostic; no experiment or historical validation runner is rerun. Native cost qualification does not prove acceleration. G1-G4 remain separate and grant no publication authority.",
    )
    value["sample_size_budget"]["intended_scheduling_slots"] = 13
    value["acceptance_gates"]["all_slots_accounted"] = len(value["rows"]) == 13
    value["rows"][-1].update(
        experiment_id=8564,
        honest_verdict=value["honest_verdict"],
        verdict_class=value["verdict_class"],
        terminal_seal="exact_final_primary_bytes_in_external_bound_sidecar",
    )
    for name, claim in value["scientific_claims"].items():
        claim.update(qualified_retrospective_benefit=None, generalized_learning_benefit_score=0)
        if name == "complete_native_cost" and claim["qualified_producer"]:
            claim["decision"] = "qualified_complete_cost_speed_null_retire_unchanged_repeat"
    value["retirement_recommendations"] += [
        dict(
            scope="unchanged_exposed_source_head",
            recommendation="retire_raw_scalar_permuted_refits_and_unchanged_inventories",
            hypothesis_retired=False,
            reopening_condition="New source information or disjoint independently adjudicated questions; preserve Exp8529 qualified source null.",
        )
    ]
    value["field_principles"].update(
        {
            k: "Bind original source hashes and scope; scheduling and qualification grant no independent scientific promotion."
            for k in [*value, "field_principles", "reproducibility_checksum"]
        }
    )
    value["reproducibility_checksum"] = e.checksum(value)
    return value

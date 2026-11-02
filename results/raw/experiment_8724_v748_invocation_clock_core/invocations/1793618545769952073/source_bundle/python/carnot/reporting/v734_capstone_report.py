"""REQ-REPORT-8551: qualification, scientific benefit and scheduling stay separate."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import v726_capstone_report as shipped
from carnot.reporting import v734_capstone_evidence as e

Json = dict[str, Any]
CLAIMS = {
    "continuous_learning": (
        8544,
        "unavailable_prerequisite",
        "Qualify original learning families and causal coordinate accounting; then test real versus shuffled feedback and simultaneous retention on the registered stream.",
    ),
    "complete_native_cost": (
        8546,
        "unavailable_prerequisite",
        "Qualify the actual nine-input109-coefficient native interface, then measure disjoint complete paired request clocks; five-input service timing does not transfer.",
    ),
    "actual_capability_mask": (
        8549,
        "unavailable_prerequisite",
        "Qualify terminal replay and complete controls before measuring actual mask behavior; require a changed discovery mechanism or resolved direct effects for benefit.",
    ),
    "physical_operations": (
        8550,
        "blocked_compatible_execution",
        "Supply compatible installed operations and complete device/transport clocks. GateMate requires exact sources, physical cable/port/power change, IDCODE, flash and smoke evidence.",
    ),
}


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Reuse the shipped schema while keeping original nulls and failures authoritative."""
    with patch.multiple(shipped, e=e, CLAIMS=CLAIMS):
        value = dict(shipped.build(work, receipts, raw, output))
    prior = e.load(work["predecessors"][0])
    nulls = [
        {
            k: r[k]
            for k in [
                "experiment_id",
                "task_id",
                "honest_verdict",
                "verdict_class",
                "primary_reference",
                "eligible",
            ]
        }
        | {"primary_contrast_rows": r["selected"]["primary_contrast_rows"]}
        for r in prior["rows"]
        if r["experiment_id"] == 8529
    ]
    quarantines = [
        {
            k: r.get(k)
            for k in [
                "experiment_id",
                "task_id",
                "honest_verdict",
                "verdict_class",
                "primary_reference",
                "upstream_flagged_adversarial",
                "original_required_checks_passed",
                "upstream_adversarial_findings",
                "upstream_failed_receipts",
            ]
        }
        for r in [*prior["rows"], *work["slots"]]
        if r["verdict_class"] == "disqualified"
    ]
    value.update(
        experiment_id=8551,
        run_date="20261015",
        random_seed=7348551,
        honest_verdict=f"complete_{value['verdict_class']}_v734_capstone",
        intended_count=12,
        task_count=12,
        censored_count=12 - value["completed_count"],
        failed_count=value["disqualified_producer_count"],
        administrative_complete=len(value["rows"]) == 12,
        milestone_status="scheduling_accounted_scientific_milestone_unqualified",
        canonical_tasks_sha256=work["contract_digest"],
        task_contract_rows=work["authority"]["contract_rows"],
        qualified_scientific_results=nulls,
        preserved_nulls=nulls,
        quarantined_inputs=quarantines,
        blocked_inputs=[r for r in work["slots"] if r["verdict_class"] == "blocked"],
        conditional_exposed_benefits=dict(
            status="no_new_qualified_benefit",
            generalization_credit=0,
            scope="exposed public labels and constructed oracle controls only",
        ),
        deterministic_claim_samples=work["samples"],
        historical_exposure_and_quarantines=e.methods.protocol()["retirement_ledger"],
        publication_authorized=False,
        publication_gate_result=work["publication"],
        g_gate_rows=[dict(gate=k, result=v) for k, v in work["publication"]["gates"].items()],
        next_evidence_requirements=value["next_evidence_conditions"]
        | {
            "fresh_data": "Disjoint independently adjudicated new question clusters with resolved exposure; FaithBench is already exposed.",
            "runtime": "Authenticated causal library/environment repair and successful bounded CUDA context/copy evidence before current-Qwen generation.",
        },
        architecture_boundary_notes=[
            "The current calibrated verifier consumes cached exposed answers; no generator weights change.",
            "Learning event counts and changed coordinates have distinct meanings; neither proves later-block benefit.",
            "Native parity, complete service cost and compatible physical execution are separate prerequisites.",
            "Stored ARC replay checks execution mechanics and grants no hidden-game capability benefit.",
        ],
        methodology_note="No-model aggregation from original sealed terminal evidence. Twelve scheduling slots do not imply twelve successful studies. One original learning event and one ARC episode are diagnostic samples; no experiment or full historical family is rerun. Missing native cost remains missing. G1-G4 govern publication independently and grant no publication authority.",
    )
    value["sample_size_budget"]["intended_scheduling_slots"] = 12
    value["acceptance_gates"].update(all_slots_accounted=len(value["rows"]) == 12)
    value["rows"][-1].update(
        experiment_id=8551,
        honest_verdict=value["honest_verdict"],
        verdict_class=value["verdict_class"],
        terminal_seal="exact_final_primary_bytes_in_external_bound_sidecar",
    )
    for claim in value["scientific_claims"].values():
        claim.update(qualified_retrospective_benefit=None, generalized_learning_benefit_score=0)
    value["retirement_recommendations"].append(
        dict(
            scope="unchanged_exposed_source_head",
            recommendation="retire_raw_scalar_permuted_refits_and_unchanged_inventories",
            hypothesis_retired=False,
            reopening_condition="Require new source information or disjoint independently adjudicated questions; preserve Exp8529 qualified source null.",
        )
    )
    value["field_principles"].update(
        {
            k: "Bind original producer hashes and scope; administrative completion grants no scientific promotion."
            for k in value
        }
    )
    value["reproducibility_checksum"] = e.checksum(value)
    return value

"""REQ-REPORT-8525: branch evidence, qualification and continuation have separate meanings."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import v726_capstone_report as shipped
from carnot.reporting import v731_capstone_report as previous
from carnot.reporting import v731_capstone_evidence as history_evidence
from carnot.reporting import v732_capstone_evidence as e

Json = dict[str, Any]
CLAIMS = {
    name: (number, decision, previous.CLAIMS[name][2])
    for name, number, decision in [
        ("source_information", 8515, "unmeasured_three_source_contrasts"),
        ("objective_learning", 8518, "disqualified_stream_unmeasured_benefit_audit"),
        ("append_service", 8520, "qualified_five_input_host_service_only"),
        ("current_qwen", 8521, "unmeasured_runtime_pre_gate"),
        ("arc_cross_game_supervision", 8523, "qualified_direct_effect_null"),
        ("board_operations", 8524, "blocked_installed_operation_and_physical_receipts"),
    ]
}
CLAIMS["objective_learning"] = (
    8518,
    CLAIMS["objective_learning"][1],
    "Repair the exact Exp8517 validation and stale sidecar failures, then audit real later Brier gains versus frozen/log-loss and simultaneous old-behavior retention upper95<=.01; retain the original V726 initial head.",
)
CLAIMS["append_service"] = (
    8520,
    CLAIMS["append_service"][1],
    "Preserve qualified paired PRIMARY complete-clock lower95>1 and p95<=1.05. Measure nine-input inference and actual Rust/Python NFR-01 separately; five-input host storage and process-exit recovery do not prove those claims or physical power-loss durability.",
)


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Preserve unqualified observations without turning administrative completion into benefit."""
    with patch.multiple(shipped, e=e, CLAIMS=CLAIMS):
        value = dict(shipped.build(work, receipts, raw, output))
    history = [e.load(ref) for ref in work["predecessors"]]
    v731, v730 = history
    value.update(
        experiment_id=8525,
        run_date="20261014",
        random_seed=7328525,
        honest_verdict=f"complete_{value['verdict_class']}_v732_capstone",
        canonical_tasks_sha256=work["authority"].get("canonical_tasks_sha256"),
        invocation_argv=work["invocation_argv"],
        administrative_complete=len(value["rows"]) == 14,
        scientific_eligibility="blocked_required_upstream_science_unavailable"
        if value["verdict_class"] == "blocked"
        else value["verdict_class"],
        publication_authorized=False,
        historical_reductions=work["historical_slots"],
        V731_history=dict(
            producer_count=v731["producer_count"],
            pregate_count=v731["pregate_count"],
            cascade_skip_count=v731["cascade_skip_count"],
            capstone_count=1,
            task_count=13,
            rows=v731["rows"],
            primary=work["predecessors"][0],
            preserved_incomplete_design=e.historical_refs()[2],
            authority_retroactively_repaired=False,
        ),
        V730_history=dict(
            primary=work["predecessors"][1],
            scientific_claims=v730["scientific_claims"],
            paper_ready=v730["paper_ready"],
            publication_readiness_grants_branch_benefit=False,
        ),
        inherited_adversarial_findings=[
            dict(
                producer=r["task_id"],
                primary=r["primary_reference"],
                flagged_adversarial=r.get("upstream_flagged_adversarial"),
                findings=r.get("upstream_adversarial_findings", []),
            )
            for r in [*work["slots"], *work["historical_slots"]]
            if r.get("upstream_flagged_adversarial") or r.get("upstream_adversarial_findings")
        ],
    )
    value["rows"][-1].update(
        experiment_id=8525,
        disposition="current_reconciliation",
        honest_verdict=value["honest_verdict"],
        verdict_class=value["verdict_class"],
    )
    value["acceptance_gates"]["all_slots_accounted"] = len(value["rows"]) == 14
    value["failed_count"] = value["disqualified_producer_count"]
    for name, (number, _, _) in CLAIMS.items():
        row = next(r for r in value["rows"] if r["experiment_id"] == number)
        value["scientific_claims"][name].update(
            qualified_retrospective_benefit=None,
            missing_reason=None
            if row["eligible"]
            else "required_scientific_producer_absent_or_disqualified",
        )
    value["scientific_claims"]["source_information"]["primary_contrasts"] = [
        dict(control=c, gain=None, missing_reason="Exp8514 pre-gated; Exp8515 cascade-skipped")
        for c in ["raw_p2", "scalar", "source_permutation"]
    ]
    value["scientific_claims"]["objective_learning"]["retention_windows"] = [
        dict(window=i, metric=None, missing_reason="Exp8517 disqualified; Exp8518 pre-gated")
        for i in range(5)
    ]
    append = next(r for r in value["rows"] if r["experiment_id"] == 8520)
    value["scientific_claims"]["append_service"].update(
        qualified_retrospective_benefit=append["selected"].get("append_speed_benefit_score")
        if append["eligible"]
        else None,
        scope="five-input host service; original paired PRIMARY complete clocks only",
    )
    value["scientific_claims"]["arc_cross_game_supervision"].update(
        qualified_retrospective_benefit=False,
        scope="direct local state effects; all redirects censored; no transferable benefit or solve credit",
    )
    pins: Json = {}
    for ancestor in history:
        for row in [*ancestor["rows"], *ancestor.get("historical_reductions", [])]:
            if row.get("primary_reference"):
                pins[row["task_id"]] = row["primary_reference"]
        for ref in ancestor["source_artifact_hashes"]:
            name = Path(ref["path"]).name
            if (
                name.startswith("experiment_")
                and int(name.split("_")[1]) in history_evidence.PREDECESSOR_PINS
            ):
                number = int(name.split("_")[1])
                if ref["sha256"] != history_evidence.PREDECESSOR_PINS[number]:
                    raise ValueError("predecessor_pin")
                older = e.load(ref)
                pins[older["task_id"]] = ref
                pins.update(
                    {
                        r["task_id"]: r["primary_reference"]
                        for r in older["rows"]
                        if r.get("primary_reference")
                    }
                )
    pins.update(
        {
            ancestor["task_id"]: ref
            for ancestor, ref in zip(history, work["predecessors"], strict=True)
        }
    )
    value["prior_failure_comparisons"] = []
    for task in work["tasks"]:
        for prior in task.get("prior_failures", []):
            ref = pins.get(prior["experiment_id"])
            original = e.load(ref) if ref and ref.get("exists", True) else None
            value["prior_failure_comparisons"].append(
                dict(
                    task_id=task["id"],
                    predecessor=prior["experiment_id"],
                    reference=ref,
                    expected_verdict=prior["verdict"],
                    observed_verdict=original.get("honest_verdict") if original else None,
                    exact_match=original.get("honest_verdict") == prior["verdict"]
                    if original
                    else None,
                    retire_if_same_verdict=prior.get("retire_if_same_verdict"),
                    missing_reason=None
                    if original
                    else "authenticated_predecessor_bytes_unavailable",
                )
            )
    for name, (number, _, condition) in CLAIMS.items():
        row = next(r for r in value["rows"] if r["experiment_id"] == number)
        value["retirement_recommendations"].append(
            dict(
                task_id=row["task_id"],
                predecessor=row["primary_reference"],
                exact_verdict=row["honest_verdict"],
                recommendation="retire_exact_unchanged_attempt",
                hypothesis_retired=False,
                scope=name,
                reopening_condition=condition,
            )
        )
    value["methodology_note"] = (
        "Read-only no-model aggregation under exact original authority. Independent workers cold-reduce primitive source, learning, retention, host clocks and direct ARC effects when available. Exposed history grants no generalization. Missing science stays missing, failed producers stay disqualified, and external absence blocks continuation. Publication G1–G4 is separate from milestone science and grants no submission permission."
    )
    value["field_principles"].update(
        {
            k: "Bind exact producer bytes and this invocation; administrative completion grants no scientific benefit."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["reproducibility_checksum"] = e.checksum(value)
    return value

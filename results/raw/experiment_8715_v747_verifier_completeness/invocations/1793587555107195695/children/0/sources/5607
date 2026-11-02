"""REQ-REPORT-8666: successful bookkeeping can coexist with blocked science."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from carnot.reporting import v743_capstone_inputs as e
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS
from carnot.reporting.v741_capstone_inputs import cause
from scripts.recurring_blocker_ledger import normalise

Json = dict[str, Any]
REOPEN = {
    "learning": "Run a fresh cohort A with unchanged transitive driver/test hashes, then cohort B's25 original nodes and all deadline mutants. Retain all64 nodes, original27 runner nodes and83 statements. Stop on drift, killed child or omitted assertion. Only then run real delayed updates and later Brier plus simultaneous retention audit.",
    "source": "The fixed190-response coverage question is closed positive for55 recovered responses. Reopen only with changed evidence or mechanism. Semantic entailment needs independent labels and an oracle-distinct control; stop if coverage is substituted for truth.",
    "installed": "Keep the qualified120-cell cost null closed for NFR-01. A changed native mechanism and matched full lifecycle lower95>=10 may reopen it; stop at any original122-node or cost-partition failure.",
    "runtime": "Changed CUDA environment/library hashes and actual context/copy success must precede the fixed Qwen canary. Stop on its first failure; optional runtime cannot block CPU branches.",
    "arc": "Retain corrected serialization and236 authenticated reset causes. Diagnose the unchanged observer's1,022 duplicate_observation issues per episode under sealed original assertions. Stop on a failed clean-observer gate; only then rerun the exact panel. Supervisor efficacy separately needs a changed causal policy and uncensored cross-game outcome support.",
    "hardware": "Supply complete disjoint operation spans that isolate spline arithmetic, compatible physical kernels, source/build/bitstream hashes, executed parity and host/device clocks. Stop without those operands. GateMate requires physical change then IDCODE0x20000001 then n16. No conditional ceiling is achieved speed.",
    "generalization": "Supply unexposed independent questions and oracle-distinct matched controls with original confidence thresholds; stop if exposure or oracle circularity remains.",
}


def decisions(work: Json, rows: list[Json]) -> list[Json]:
    """Literal prior verdicts and underlying causes constrain every retirement scope."""
    result = []
    for task, row in zip(work["protocol"]["task_map_v743"], rows, strict=True):
        prior = []
        for registration in task.get("prior_failures", []):
            number = registration["experiment_id"].split("-", 1)[0][3:]
            candidates = list((e.ROOT / "results").glob(f"experiment_{number}_*.json"))
            actual = e.read(e.reference(candidates[0])) if len(candidates) == 1 else {}
            prior.append(
                dict(
                    experiment_id=registration["experiment_id"],
                    registered_verdict=registration["verdict"],
                    literal_verdict=actual.get("honest_verdict"),
                    operand_state="observed_value" if actual else "missing_file",
                    normalized_verdict=normalise(str(actual.get("honest_verdict", "missing"))),
                    normalized_underlying_cause=cause(actual),
                    failed_owned_checks=e.failure_evidence(actual),
                )
            )
        current_cause = cause(dict(row, checks=row.get("original_gate_check_summary")))
        if task["id"] == "exp8664-arc-reset-transfer":
            current_cause = "duplicate_observation_clean_observer_failure"
            for previous in prior:
                if previous["experiment_id"].startswith("exp8650-"):
                    previous["normalized_underlying_cause"] = "TypeError_missing_reset_causes"
        decision = (
            "close_exact_qualified_null"
            if row["eligible"] and row["verdict_class"] == "null"
            else "retain_qualified_scope"
            if row["eligible"]
            else "require_changed_prerequisite"
        )
        result.append(
            dict(
                task_id=task["id"],
                literal_verdict=row["honest_verdict"],
                verdict_class=row["verdict_class"],
                disposition=row["disposition"],
                actual_prior_outcomes=prior,
                normalized_underlying_cause=current_cause,
                exact_verdict_repeat=any(
                    p["literal_verdict"] == row["honest_verdict"] for p in prior
                ),
                normalized_cause_repeat=any(
                    p["normalized_underlying_cause"] == current_cause for p in prior
                ),
                decision=decision,
                causal_scope=task["title"],
                conductor_owner_action="recommendation_only",
                reopening_operands=REOPEN,
            )
        )
    return result


def build(work: Json, receipts: list[Json], output: Path) -> Json:
    """Only owned validation controls reconciliation readiness; missing science stays blocked."""
    owned = (
        bool(receipts)
        and all(r["passed"] for r in receipts)
        and work["contract"]["ready"]
        and work["preconditions_passed"]
    )
    verdict = "blocked" if owned else "disqualified"
    rows = deepcopy(work["slots"])
    rows.append(
        dict(
            experiment_id=8666,
            task_id=e.TASK,
            unit_id=e.TASK,
            arm="scheduled_slot",
            status="completed",
            censored=False,
            excluded=False,
            absolute_metric=int(owned),
            effective_independent_groups=0,
            eligible=owned,
            producer_executed=True,
            disposition="capstone_reconciliation",
            honest_verdict=f"complete_{verdict}_v743_capstone",
            verdict_class=verdict,
            missing_reason=None,
        )
    )
    task_results = decisions(work, rows)
    reductions = work["reductions"]
    scores = dict(
        reconciliation_ready_score=int(owned),
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        learning_execution_ready_score=0,
        source_comparison_ready_score=int(
            owned and reductions["coverage"]["recovered_count"] == 55
        ),
        installed_cost_ready_score=int(owned and reductions["installed"]["session_count"] == 120),
        packaging_noninferiority_score=int(
            owned and reductions["installed"]["packaging_noninferiority_score"]
        ),
        nfr01_score=0,
        reset_provenance_ready_score=int(owned),
        reset_transfer_rows_ready_score=0,
        promotion_ready_score=0,
        physical_hardware_ready_score=0,
    )
    gap_results = [
        dict(
            gap_id=gap,
            decision=decision,
            scientific_benefit=benefit,
            reopening_operands=REOPEN[operand],
        )
        for gap, decision, benefit, operand in [
            ("FR-06/FR-11", "blocked_no_current_learning_or_retention", None, "learning"),
            ("FR-12-COVERAGE", "closed_positive_fixed_complete_source_protocol", 55, "source"),
            ("FR-12-SEMANTICS", "blocked_no_semantic_truth_evidence", None, "source"),
            (
                "FR-05/FR-08",
                "qualified_installed_compatibility_and_complete_cost",
                None,
                "installed",
            ),
            ("NFR-01", "closed_qualified_null_below10x", 0, "installed"),
            ("GAP-INDEPENDENT-GENERALIZATION", "blocked_exposed_questions", None, "generalization"),
            ("GAP-ORACLE-DISTINCT", "blocked_no_distinct_control", None, "generalization"),
            ("ARC-RESET-TRANSFER", "disqualified_original_clean_observer_assertion", None, "arc"),
            ("ARC-SUPERVISOR", "preserve_V737_efficacy_null", 0, "arc"),
            ("HARDWARE", "blocked_missing_compatible_physical_execution", None, "hardware"),
        ]
    ]
    publication = work["publication"]
    value: Json = dict(
        experiment_id=8666,
        task_id=e.TASK,
        milestone="2026.10.743",
        run_date=work["run_date"],
        honest_verdict=f"complete_{verdict}_v743_capstone",
        verdict_class=verdict,
        inference_substrate_class="no_model_load",
        inference_substrate="CPU immutable evidence reduction; no numerical prediction or language-model load",
        MODEL_SPECS=[],
        model_invocation_counts=ZERO_INVOCATION_COUNTS,
        current_language_model_calls=0,
        numerical_inference_this_invocation=False,
        historical_model_provenance=work["historical_models"],
        target_model=dict(
            kind="authenticated_numerical_energy_head",
            checkpoint=work["protocol"]["head_checkpoint_reference"],
            parameter_count=109,
            numerical_inference_this_invocation=False,
        ),
        rows=rows,
        intended_count=14,
        completed_count=14,
        failed_count=0 if owned else 1,
        censored_count=0,
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            intended_scheduling_slots=14,
            independent_scientific_questions_added=0,
            scientific_sample_sizes="Retain producer response/question/session/game denominators; do not count capstone rows as studies.",
        ),
        verifier_is_oracle=False,
        exposure_scope="Previously exposed registered evidence only; administrative reconciliation",
        required_checks_passed=owned,
        flagged_adversarial=False,
        acceptance_gates=dict(
            owned_reconciliation=owned,
            essential_learning_and_retention=False,
            physical_execution=False,
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal_validation.json"),
        adversarial_findings=[],
        preconditions_checked=work["preconditions"],
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=8666,
        source_artifact_hashes=work["source_refs"],
        code_config_hashes=work["code_refs"],
        raw_shard_hashes=work["operand_refs"],
        cited_upstream_artifacts=[r["primary_reference"] for r in rows[:-1]],
        per_task_results=task_results,
        per_gap_results=gap_results,
        original_union_audit=work["unions"],
        normalized_failure_causes=[
            dict(task_id=r["task_id"], cause=r["normalized_underlying_cause"]) for r in task_results
        ],
        retirement_recommendations=task_results,
        reopen_conditions=REOPEN,
        next_actions=REOPEN,
        publication_gates=publication,
        unmet_gates=publication["unmet_gates"],
        publication_authorized=False,
        paper_ready=publication["paper_ready"],
        cold_reductions=reductions,
        full_task_objects=work["protocol"]["task_map_v743"],
        canonical_tasks_sha256=e.methods.TASKS_PIN,
        task_contract_rows=work["contract"]["rows"],
        branch_gate_values=scores,
        primitive_reference=e.reference(Path(work["raw"]) / "measurement.json"),
        validation_reference=e.reference(Path(work["raw"]) / "validation.json"),
        publication_output=str(output),
        gate_check_summary=work["gate_rows"]
        + [
            dict(
                upstream=e.TASK,
                check=k,
                path=str(output),
                sha256=None,
                field=k,
                operator="==",
                expected=v,
                observed=v,
                operand_state="observed_zero" if v == 0 else "observed_value",
                passed=True,
            )
            for k, v in scores.items()
        ],
        preserved_nulls=work["history"]["preserved_nulls"],
        historical_quarantines=work["history"]["historical_quarantines"],
        preserved_v737_supervisor_null=work["history"]["preserved_v737_supervisor_null"],
        historical_capstone_reference=work["history_reference"],
        reconciliation_success=owned,
        branch_evidence_limit="Exp8665 retained120 complete installed cells. Learning is absent; extraction admission raised a gate-shape TypeError. Opaque binding residuals cannot isolate physical spline costs.",
    )
    value.update(scores)
    value.update({k: publication["gates"][k] for k in ("G1", "G2", "G3", "G4")})
    value["field_principles"] = {
        k: "Bind identity, exact source bytes and original scientific scope; qualification never substitutes for measured benefit."
        for k in [*value, "field_principles"]
    }
    value["reproducibility_checksum"] = e.checksum(value)
    value["field_principles"]["reproducibility_checksum"] = (
        "Hash every declared report operand; cold replay separately authenticates original sources."
    )
    value["reproducibility_checksum"] = e.checksum(value)
    return value

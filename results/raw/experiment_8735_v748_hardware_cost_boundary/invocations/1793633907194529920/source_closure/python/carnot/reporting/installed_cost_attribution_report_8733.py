"""REQ-REPORT-8733: qualify attribution separately from batching and speed benefit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

Json = dict[str, Any]


def build(work: Json, output: Path) -> Json:
    """Missing data never becomes a zero cost; excessive profiler overhead disqualifies."""
    from carnot.reporting import installed_cost_attribution_8733 as a

    measured, validation = work["measurement"], work["validation"]
    rows = measured["process_rows"]
    summary = a.aggregate(rows)
    owned = work["private_control"] or (
        validation["coverage"]["complete"]
        and validation["custody"]["passed"]
        and all(r["passed"] for r in validation["receipts"])
    )
    blocked = bool(work["inputs"]["gate_check_summary"])
    valid = bool(owned and not blocked and not work["owned_error"] and summary["ready"])
    ready = valid and not work["private_control"]
    verdict = (
        "disqualified"
        if not owned or work["owned_error"]
        else "blocked"
        if blocked
        else "disqualified"
        if not summary["ready"]
        else "null"
        if work["private_control"]
        else "positive"
    )
    reason = (
        "external_operand"
        if blocked
        else "instrumentation_overhead"
        if summary["complete"] and not summary["ready"]
        else "installed_cost_attribution"
    )
    fractions = summary["fractions"] if valid else []
    evidence = sorted(
        [r for r in fractions if r["clock_domain"] == "child_serve"],
        key=lambda r: r["estimate"],
        reverse=True,
    )[:2]
    value: Json = dict(
        experiment_id=8733,
        task_id=a.e.TASK,
        milestone="2026.11.748",
        run_date="20261102",
        invocation_id=work["invocation_id"],
        honest_verdict=f"complete_{verdict}_{reason}",
        verdict_class=verdict,
        gate_check_summary=work["inputs"]["gate_check_summary"],
        inference_substrate="verifier_ensemble_against_cached_candidates"
        if summary["complete"]
        else "aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(language_model_calls=0, model_loads=0),
        current_language_model_calls=0,
        rows=rows,
        intended_count=40,
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=sum(r["status"] == "failed" for r in rows),
        censored_count=sum(r["status"] == "censored" for r in rows),
        excluded_count=0,
        independent_count=0,
        verifier_is_oracle=False,
        exposure_scope="fixed exposed original64 panel; runtime diagnostic only",
        independent_generalization_score=0,
        acceptance_gates=[
            dict(name="complete_processes", passed=summary["complete"]),
            dict(name="low_overhead_attribution", passed=valid),
        ],
        required_checks_passed=bool(owned),
        flagged_adversarial=bool(not owned or work["owned_error"]),
        validation_receipts=validation["receipts"] + measured["receipts"],
        owned_coverage=validation["coverage"],
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal_validation.json"),
        preconditions_checked=work["preconditions"],
        duration_s=work["duration_s"],
        started_at_utc=work["started_at_utc"],
        ended_at_utc=work["ended_at_utc"],
        phase_spans=work["phase_spans"],
        random_seed=7488733,
        source_artifact_hashes=work["inputs"]["references"],
        code_config_hashes=work["sources"],
        raw_shard_hashes=work["raw_references"] + validation["raw_shard_hashes"],
        cited_upstream_artifacts=work["inputs"]["references"],
        installed_attribution_ready_score=int(ready),
        branch_gate_values=dict(installed_attribution_ready_score=int(ready)),
        process_rows=rows,
        component_span_rows=measured["complete_cost_rows"],
        instrumentation_overhead_interval=summary["overhead"],
        component_fraction_intervals=fractions,
        diagnostic_fraction_intervals=summary["fractions"],
        unexplained_residual=[
            dict(
                process_id=r["unit_id"],
                child_s=r.get("child_unexplained_s"),
                end_to_end_parent_s=r.get("end_to_end_residual_s"),
            )
            for r in rows
        ],
        next_optimization_evidence=dict(
            qualified=valid,
            largest_components=evidence,
            recommendation="Remove redundant feature extraction in a future parity-tested change."
            if evidence
            else "No production optimization selected from disqualified or absent attribution.",
            measured_improvement=False,
        ),
        task_contract_rows=work["inputs"]["contract"]["rows"],
        canonical_tasks_sha256=a.contract.m.PIN,
        original_scientific_method=work["inputs"].get("method"),
        preserved_8717_intervals=work["inputs"].get("historical_paired_intervals", []),
        batching_benefit_claim_this_invocation=False,
        same_batch_10x_claim_this_invocation=False,
        physical_device_execution_score=0,
        generalized_learning_benefit_score=0,
        sample_size_budget=dict(
            process_cells=40,
            paired_repeats=10,
            panels_per_process=16,
            items_per_panel=64,
            item_evaluations=40960,
            independent_semantic_questions=0,
        ),
        primitive_reference=a.e.reference(Path(work["raw"]) / "work.json"),
        owned_error=work["owned_error"],
        adversarial_findings=work.get("adversarial_findings", []),
        methodology="Exact Exp8716 installed wheel and extension outside checkout. Forty B16 process cells, ten paired repeats per Python/native arm; sixteen original64 panels. Same-arm paired requests and scientific outputs are exact; original cross-arm numerical tolerance is retained. Child exclusive spans have their own measured parent. End-to-end parent residual excludes only disjoint parent spans. Ten process clusters and10000 bootstrap draws, seed7488733; overhead upper95<=5% required. No speed improvement, durability, learning, generalization or physical acceleration claim.",
    )
    value["field_principles"] = {
        k: "Bind exact evidence and current invocation; missing costs differ from zero and attribution differs from improvement."
        for k in value
    }
    value["reproducibility_checksum"] = a.e.canonical_hash(value)
    return value

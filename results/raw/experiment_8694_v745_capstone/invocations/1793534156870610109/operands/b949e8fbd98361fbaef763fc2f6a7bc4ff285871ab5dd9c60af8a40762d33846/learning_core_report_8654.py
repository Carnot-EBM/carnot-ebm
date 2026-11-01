"""REQ-REPORT-8654: report core readiness separately from complete qualification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from carnot.reporting import learning_core_8654 as e
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]
REQUIRED = {
    "owned_tests",
    "ruff_check",
    "ruff_format",
    "mypy",
    "spec",
    "private_e2e018",
    "private_e2e021",
}


def build(work: Json, output: Path) -> Json:
    """A planned complete core partition cannot grant training or scientific benefit."""
    partition = work["partition"]
    cohorts = {r["cohort"]: r for r in partition.get("cohorts", [])}
    intended = cohorts.get("A", {}).get("node_ids", [])
    outcomes = work["outcomes"]
    seen = [r["node_id"] for r in outcomes]
    checks = dict(
        inputs=not work["input_failures"],
        source_identity_unchanged=not work.get("source_reuse_refusals"),
        cohort_a_nodes=bool(intended)
        and sorted(seen) == sorted(intended)
        and all(r["outcome"] == "passed" for r in outcomes),
        retained_children=bool(work["children"]) and not work.get("child_errors"),
        original_runner_observed=work.get("original_coverage", {}).get("observed_statements", 0)
        > 0,
        owned_statements=work.get("owned_coverage", {}).get("complete", False),
        launch_budgets=bool(work["launch_budget_rows"])
        and all(r["admitted"] for r in work["launch_budget_rows"]),
        validation=REQUIRED <= {r["name"] for r in work["receipts"] if r["passed"]}
        and all(r["passed"] for r in work["receipts"]),
    )
    ready = all(checks.values())
    verdict = (
        "blocked" if work["input_failures"] else "circular_positive" if ready else "disqualified"
    )
    rows = [
        dict(
            node_id=n,
            unit=n,
            outcome=next((r["outcome"] for r in outcomes if r["node_id"] == n), "unstarted"),
            missing_reason=None if n in seen else "not_executed_or_no_terminal_receipt",
            effective_independent_groups=0,
        )
        for n in intended
    ]
    failures = list(work["input_failures"]) + work.get("source_reuse_refusals", [])
    if not work["input_failures"]:
        failures += [
            e.methods.gate(Path(work["raw"]) / "work.json", k, True, v, e.TASK)
            for k, v in checks.items()
            if not v
        ]
    value = dict(
        experiment_id=8654,
        task_id=e.TASK,
        milestone="2026.10.743",
        run_date=work["run_date"],
        honest_verdict="complete_" + verdict + "_v743_learning_core_execution",
        verdict_class=verdict,
        gate_check_summary=failures,
        inference_substrate="deterministic_cpu_small_energy_head_no_model_load",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(current_llm_calls=0, model_loads=0, generations=0),
        historical_model_provenance=work["upstream"].get("historical_model_provenance", []),
        target_model=work["numerical_inference"],
        small_head_numerical_inference=work["numerical_inference"],
        rows=rows,
        intended_count=len(intended),
        completed_count=len(outcomes),
        failed_count=sum(r["outcome"] != "passed" for r in outcomes),
        censored_count=sum(n not in seen for n in intended),
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            unit="original assertion node", intended=len(intended), independent_questions=0
        ),
        verifier_is_oracle=True,
        exposure_scope="exposed historical mechanics and original assertions; core_only execution qualification",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=ready,
        flagged_adversarial=bool(work.get("adversarial_findings")),
        acceptance_gates={
            k: dict(
                passed=v,
                principle="Real original node execution and sealed operands are required; core readiness cannot authorize training.",
            )
            for k, v in checks.items()
        },
        validation_receipts=work["receipts"],
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal.json"),
        adversarial_findings=work.get("adversarial_findings", []),
        preconditions_checked=work["preconditions"],
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=7438654,
        source_artifact_hashes=work["input_references"],
        code_config_hashes=work["sources"],
        raw_shard_hashes=[r["reference"] for m in work["children"] for r in m["files"]],
        cited_upstream_artifacts=work["input_references"],
        learning_core_ready_score=int(ready),
        cohort_a_nodes=intended,
        cohort_b_pending_nodes=cohorts.get("B", {}).get("node_ids", []),
        original_full_node_union=partition.get("original_node_ids", []),
        child_execution_receipts=[m["receipt"] for m in work["children"]],
        child_shard_manifest=work["children"],
        transitive_source_hashes=work["sources"],
        mechanics_fixture_reference=work.get("mechanics_fixture"),
        original_statement_union=work.get("original_coverage", {}),
        owned_statement_coverage=work.get("owned_coverage", {}),
        grouping_rows=work["grouping_rows"],
        launch_budget_rows=work["launch_budget_rows"],
        prior_failure_diagnosis=work["prior_failure"],
        source_reuse_refusals=work.get("source_reuse_refusals", []),
        historical_execution_source_hashes=work.get("historical_execution_sources", []),
        scope="core_only",
        complete_learning_qualification=False,
        training_eligible=False,
        branch_gate_values=dict(learning_core_ready_score=int(ready)),
        primitive_reference=e.c.ref(Path(work["raw"]) / "work.json"),
        invocation_seal_reference=work["invocation_seal_reference"],
        publication_output=str(output),
        future_dependencies=[
            dict(experiment_id=n, status="future_dependency_not_observation")
            for n in range(8655, 8667)
        ],
        historical_quarantines=work["upstream"].get("historical_quarantines", {}),
        methodology_note="Registered module fixtures execute once per child. Immutable coverage and JUnit bind exact original assertions. Exp8655 must authenticate unchanged core sources/config/operands and execute B before training is eligible.",
    )
    value["field_principles"] = {
        k: "Bind this invocation to immutable operands and actual bounded children; qualification differs from benefit."
        for k in value
    }
    value["reproducibility_checksum"] = canonical_hash(value)
    return value

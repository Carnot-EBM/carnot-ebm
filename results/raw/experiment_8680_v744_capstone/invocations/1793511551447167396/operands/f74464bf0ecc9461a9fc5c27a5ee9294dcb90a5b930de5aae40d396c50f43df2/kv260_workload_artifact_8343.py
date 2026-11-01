"""REQ-REPORT-8343: arithmetic and complete service costs have different scopes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from carnot.reporting import kv260_arithmetic_8343 as p
from carnot.reporting import kv260_workload_cost_8343 as e
from carnot.reporting import kv260_workload_primitives_8329 as durable
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, canonical_hash

Json = dict[str, Any]


def build(work: Json, raw: Path, receipts: list[Json]) -> Json:
    """Count each constructed vector once per arm; five clocks add no sources."""
    owned = bool(receipts) and all(r["passed"] for r in receipts) and not work["owned_failure"]
    clocks = work["timing_rows"]
    arithmetic = len(clocks) == 10 and all(
        {r["repetition"] for r in clocks if r["arm"] == arm} == set(range(5)) for arm in p.ARMS
    )
    rows = [
        dict(
            branch="arithmetic",
            source_id=f"constructed:{i + 1}",
            arm=arm,
            condition="arithmetic:" + arm,
            intended=1,
            completed=arithmetic,
            failed=work["owned_failure"],
            censored=not arithmetic and not work["owned_failure"],
            excluded=False,
            independent=0,
            numerator=int(arithmetic),
            denominator=1,
            evidence_status="constructed_CPU_arithmetic"
            if arithmetic
            else "unavailable_arithmetic",
        )
        for i in range(128)
        for arm in p.ARMS
    ]
    arithmetic_rows = list(rows)
    for branch in e.SOURCES:
        detail = work["branches"].get(branch, dict(eligible=False, source_count=None))
        identifiers = sorted(
            {r["source_id"] for r in work["durable_rows"] if r["branch"] == branch}
        )
        for identity in identifiers or [branch + ":unavailable"]:
            for arm in durable.ARMS:
                matches = [
                    r
                    for r in work["durable_rows"]
                    if (r["branch"], r["source_id"], r["arm"]) == (branch, identity, arm)
                ]
                complete = detail["eligible"] and len(matches) == 5
                rows.append(
                    dict(
                        branch=branch,
                        source_id=identity,
                        arm=arm,
                        condition=branch + ":" + arm,
                        intended=1,
                        completed=complete,
                        failed=False,
                        censored=not complete,
                        excluded=False,
                        independent=int(complete and branch == "natural"),
                        numerator=int(complete),
                        denominator=1,
                        evidence_status="measured_complete_CPU_transaction"
                        if complete
                        else "unavailable_external_operand",
                    )
                )
    failures = [r for r in work["checks"] if not r["passed"]]
    kind = "disqualified" if not owned else "blocked" if failures else "circular_positive"
    ready = int(owned and arithmetic)
    durable_ready = int(owned and any(r["completed"] and r["branch"] != "arithmetic" for r in rows))
    natural_ready = int(owned and any(r["completed"] and r["branch"] == "natural" for r in rows))
    operations = [
        dict(
            operation=op,
            assigned_substrate="host_CPU",
            kv260_supported=False,
            measured_ns=[r["operation_ns"][op] for r in clocks],
            reason="quadratic_Ising_overlay_has_no_spline_or_persistence_kernel",
        )
        for op in ["basis_evaluation", "gradient", "coefficient_update"]
    ]
    value: Json = dict(
        experiment_id=8343,
        task_id="exp8343-kv260-workload-cost",
        milestone="2026.10.719",
        run_date="20261009",
        honest_verdict="complete_"
        + kind
        + "_"
        + (
            "owned_checks"
            if not owned
            else failures[0]["upstream"]
            if failures
            else "constructed_arithmetic"
        ),
        verdict_class=kind,
        gate_check_summary=work["checks"],
        inference_substrate="cached_candidate_scoring",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        historical_model_provenance=work["historical_model_provenance"],
        rows=rows,
        arithmetic_rows=arithmetic_rows,
        intended_count=len(rows),
        completed_count=sum(int(r["completed"]) for r in rows),
        failed_count=sum(int(r["failed"]) for r in rows),
        censored_count=sum(int(r["censored"]) for r in rows),
        excluded_count=0,
        independent_count=len({r["source_id"] for r in rows if r["independent"]}),
        sample_size_budget=dict(
            constructed_vectors=128,
            natural_sources=None,
            warmups=1,
            repetitions=5,
            arms=2,
            repetitions_are_independent=False,
        ),
        verifier_is_oracle=True,
        exposure_scope="exposed_cached_development",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=owned,
        flagged_adversarial=bool(work.get("adversarial_findings")),
        acceptance_gates=dict(
            owned_checks=owned,
            arithmetic_cost=bool(ready),
            durable_cost=bool(durable_ready),
            natural_cost=bool(natural_ready),
            board_execution=False,
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str((raw / "terminal_validation.json").absolute()),
        adversarial_findings=work.get("adversarial_findings", []),
        finding_dispositions=work.get("finding_dispositions", []),
        preconditions_checked=True,
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=7198343,
        source_artifact_hashes=work["refs"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=[e.reference(raw / "measurement.json"), work["arithmetic_reference"]],
        measurement_reference=e.reference(raw / "measurement.json"),
        cited_upstream_artifacts=[
            dict(
                branch=b,
                path=str(e.ROOT / "results" / (n + ".json")),
                imported_fields=[f, "workload_reference", "terminal_validation_sidecar_path"],
                sha256=next(
                    (
                        r["sha256"]
                        for r in work["refs"]
                        if Path(r["original_path"]).name == n + ".json"
                    ),
                    None,
                ),
            )
            for b, (n, f) in e.SOURCES.items()
        ],
        cpu_cost_ready_score=max(ready, durable_ready),
        arithmetic_cost_ready_score=ready,
        durable_cost_ready_score=durable_ready,
        natural_cost_ready_score=natural_ready,
        kv260_execution_ready_score=0,
        cpu_cost_scope="constructed_spline_arithmetic"
        if not durable_ready
        else "constructed_arithmetic_and_qualified_complete_CPU_transactions",
        operation_rows=operations
        + (
            durable.boundary(work["durable_rows"])["operation_rows"] if work["durable_rows"] else []
        ),
        timing_rows=clocks + work["durable_rows"],
        numeric_audit=work.get("numeric_audit"),
        unmeasured_operations=[] if durable_ready else durable.OPERATIONS[2:],
        board_obligations=dict(
            kv260=dict(
                status="complete_blocked_compatible_spline_kernel",
                current_execution=False,
                access=["ssh", "kria"],
                k_max=5,
                overlay="carnot_ising_v2_n64/quadratic_Ising_only",
                historical=work["kv260_history"],
                unsupported_operations=[
                    "spline_basis",
                    "coefficient_update",
                    "database_persistence",
                ],
                next_requirement="Authenticated compatible kernel and identical inputs/outputs in bounded SSH transcript with transfer and CPU cost",
            ),
            polarfire=dict(
                status="authenticated_CPU_dispatch"
                if work["polarfire_graduation"]["polarfire_workload_validated"]
                else "complete_blocked_polarfire_terminal_authentication"
            ),
        ),
        polarfire_graduation=work["polarfire_graduation"],
        polarfire_workload_validated=work["polarfire_graduation"]["polarfire_workload_validated"],
        polarfire_terminal_evidence_hashes=[
            r for r in work["refs"] if "8259" in r["original_path"]
        ],
        compatible_fraction=None,
        amdahl_upper_bound=None,
        full_service_speedup=None,
        nfr01_met=False,
        accelerator_benefit="unproved_no_compatible_operation",
        current_device_execution_count=0,
        transfer_cost_ns=None,
        nfr01_status="requires_complete_empirical_tenfold_service_cost",
        tier1_100x_status="unproved_requires_complete_empirical_costs",
        science_parameters=p.SCIENCE,
        invocation_argv=work.get("invocation_argv", []),
        execution_manifest_reference=work.get("execution_manifest_reference"),
        owned_coverage_reference=work.get("owned_coverage_reference"),
        methodology=dict(
            description="Independent constructed spline operation timing with SciPy and finite-difference qualification",
            current_model_work=False,
            no_fixture_substitution=True,
            timing_repetitions_are_not_sources=True,
            denominator="CPU wall time; future host transfer and CPU portions remain included",
        ),
        arithmetic_scientific_class="circular_positive" if ready else "unqualified",
    )
    value["field_principles"] = {
        key: "Bind measured evidence and preserve unavailable external observations separately."
        for key in value
    }
    value["field_principles"].update(
        field_principles="Explain the evidentiary purpose of every field.",
        reproducibility_checksum="Bind the full reduction excluding this checksum.",
        arithmetic_cost_ready_score="Constructed oracle arithmetic has no natural or service claim.",
        durable_cost_ready_score="Complete persistence/recovery costs require their own qualified producer.",
        natural_cost_ready_score="Natural source readiness cannot follow from constructed inputs.",
        cpu_cost_scope="Name the measured CPU branch that grants readiness.",
        compatible_fraction="Unsupported operations leave service fractions unavailable.",
        timing_rows="Keep five paired clocks with repetition IDs; never count them as sources.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value

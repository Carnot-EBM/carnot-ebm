"""REQ-REPORT-8770: diagnose residual cost without changing the qualified service.

Process repeats measure runtime variation on exposed requests. They cannot add
independent scientific cases or support a new throughput headline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from carnot.reporting import feature_reuse_cost_8757 as previous
from carnot.reporting.primary_publication import read_bound_sidecar

Json = dict[str, Any]
ROOT = previous.ROOT
NAME, TASK = "experiment_8770_v751_optimized_cost_attribution", "exp8770-optimized-cost-attribution"
CLI = f"scripts/experiments/{NAME}.py"
TESTS = ["tests/python/test_optimized_cost_attribution_8770.py"]
OWNED = [
    f"python/carnot/reporting/optimized_cost_attribution{s}_8770.py"
    for s in ["", "_measure", "_runner"]
] + ["python/carnot/inference/optimized_profile_8770.py", CLI]
COST = "results/experiment_8757_v750_feature_reuse_cost.json"
COST_PIN = "sha256:668218f9dffdaafa2a886789af61b26b9ab2f9743f56cf05d2092e610a144258"
SEED = 7518770
reference, load, atomic_json, canonical_hash = (
    previous.reference,
    previous.load,
    previous.atomic_json,
    previous.canonical_hash,
)


def method() -> Json:
    """Freeze overhead and reconciliation limits before seeing any timing result."""
    return dict(
        process_cells=12,
        paired_repeats=3,
        panels=16,
        requests_per_panel=64,
        batch_size=16,
        overhead_limit=0.05,
        reconciliation_absolute_s=0.001,
        reconciliation_relative=0.01,
        random_seed=SEED,
        uncertainty="range of three paired processes; no population CI",
        unchanged_workload=previous.method(),
    )


def inputs(root: Path) -> Json:
    """Require both qualified primaries and their byte-bound installed operands."""
    value: Json = previous.inputs(root)
    checked = previous.prior.authenticate(
        dict(path=str(root / COST), sha256=COST_PIN),
        dict(
            experiment_id=8757,
            task_id="exp8757-feature-reuse-cost",
            installed_cost_ready_score=1,
            required_checks_passed=True,
            flagged_adversarial=False,
        ),
    )
    value["references"] += checked["references"]
    value["gate_check_summary"] += checked["failures"]
    if checked["passed"]:
        try:
            cost = checked["value"]
            wrapper = reference(Path(cost["terminal_validation_sidecar_path"]))
            side = Path(load(wrapper)["publication"]["sidecar_path"])
            if read_bound_sidecar(root / COST, side)["report"]["passed"] is not True:
                raise ValueError("terminal_rejected")
            primitive = load(cost["primitive_reference"])
            if load(primitive["seal_reference"]) != {
                k: v for k, v in primitive.items() if k != "seal_reference"
            }:
                raise ValueError("cost_seal")
            if value.get("passed") and primitive["inputs"]["workload"] != value["workload"]:
                raise ValueError("optimized_workload_identity")
            value["references"] += [
                wrapper,
                reference(side),
                cost["primitive_reference"],
                primitive["seal_reference"],
            ]
        except (OSError, ValueError, KeyError, TypeError) as error:
            value["gate_check_summary"].append(
                dict(
                    upstream="exp8757-feature-reuse-cost",
                    path=str(root / COST),
                    sha256=COST_PIN,
                    check="qualified_terminal",
                    field="terminal_validation_sidecar_path",
                    operator="==",
                    expected=True,
                    observed=str(error),
                    observed_state="rejected_producer",
                )
            )
    value["passed"] = not value["gate_check_summary"]
    if value["passed"]:
        value["workload"]["configuration_template"] = value["workload"]["configuration_templates"][
            "reuse"
        ]
    value["references"] += [
        reference(ROOT / path)
        for path in [
            "python/carnot/inference/installed_profile_8733.py",
            "python/carnot/reporting/installed_cost_attribution_measure_8733.py",
            "python/carnot/reporting/installed_cost_attribution_8733.py",
            "python/carnot/reporting/feature_reuse_cost_8757.py",
            "python/carnot/reporting/feature_reuse_cost_runner_8757.py",
            "python/carnot/reporting/v748_collection_contract.py",
            "ops/exclusion_manifest.yaml",
        ]
    ]
    return value


def roster() -> list[Json]:
    """Reverse pair order on the middle repeat to expose order effects."""
    return [
        dict(
            unit_id=f"R{repeat}-{arm}-{int(on)}",
            repeat=repeat,
            arm=arm,
            instrumented=on,
            mode="reuse",
            batch=16,
            panels=16,
            item_evaluations=1024,
            status="censored",
            passed=False,
        )
        for repeat in range(3)
        for arm in (["python", "native"] if repeat % 2 == 0 else ["native", "python"])
        for on in ([False, True] if repeat % 2 == 0 else [True, False])
    ]


def aggregate(rows: list[Json]) -> Json:
    """Use complete process clocks for bounds; unknown residuals are never free."""
    complete = len(rows) == 12 and all(r["status"] == "completed" and r["passed"] for r in rows)
    result: Json = dict(
        complete=complete,
        ready=False,
        component_cost_rows=[],
        observer_overhead_rows=[],
        reconciliation_rows=[],
        ideal_speed_bounds=[],
        dominant_component=None,
        next_optimization_decision=dict(
            selected_target=None,
            reason="No supported target without complete low-overhead attribution.",
        ),
    )
    if not complete:
        return result
    lookup = {(r["repeat"], r["arm"], r["instrumented"]): r for r in rows}
    if len(lookup) != 12 or set(lookup) != {
        (s["repeat"], s["arm"], s["instrumented"]) for s in roster()
    }:
        raise ValueError("process_roster")
    for arm in ["python", "native"]:
        off, on = [[lookup[i, arm, state] for i in range(3)] for state in [False, True]]
        if any(
            a["request_digest"] != b["request_digest"] or a["output_digest"] != b["output_digest"]
            for a, b in zip(off, on, strict=True)
        ):
            raise ValueError("paired_exact_parity")
        overhead = [b["duration_s"] / a["duration_s"] - 1 for a, b in zip(off, on, strict=True)]
        result["observer_overhead_rows"].append(
            dict(
                arm=arm,
                paired_repeats=3,
                estimate=sum(r["duration_s"] for r in on) / sum(r["duration_s"] for r in off) - 1,
                minimum=min(overhead),
                maximum=max(overhead),
                pair_fractions=overhead,
            )
        )
        for row in on:
            for domain, costs, duration in [
                (
                    "child_serve",
                    dict(row["components"], unattributed_residual=row["child_unexplained_s"]),
                    row["child_duration_s"],
                ),
                (
                    "parent_process",
                    row.get(
                        "complete_components",
                        dict(
                            row["components"],
                            process_startup=row["startup_s"],
                            process_shutdown=row["shutdown_s"],
                            unattributed_residual=row["duration_s"]
                            - row["startup_s"]
                            - row["shutdown_s"]
                            - sum(row["components"].values()),
                        ),
                    ),
                    row["duration_s"],
                ),
            ]:
                accounted = sum(v for _, v in sorted(costs.items()))
                error = abs(accounted - duration)
                passed = min(costs.values()) >= 0 and error <= max(0.001, 0.01 * duration)
                if not passed:
                    raise ValueError("cost_reconciliation")
                result["reconciliation_rows"].append(
                    dict(
                        process_id=row["unit_id"],
                        clock_domain=domain,
                        measured_s=duration,
                        accounted_s=accounted,
                        error_s=error,
                        passed=passed,
                    )
                )
                result["component_cost_rows"] += [
                    dict(
                        process_id=row["unit_id"],
                        arm=arm,
                        operation=k,
                        duration_s=v,
                        clock_domain=domain,
                        parent_duration_s=duration,
                        fraction=v / duration,
                    )
                    for k, v in sorted(costs.items())
                ]
        parent = [
            r
            for r in result["component_cost_rows"]
            if r["arm"] == arm and r["clock_domain"] == "parent_process"
        ]
        total = sum(r["duration_s"] for r in on)
        for operation in sorted({r["operation"] for r in parent}):
            fraction = sum(r["duration_s"] for r in parent if r["operation"] == operation) / total
            result["ideal_speed_bounds"].append(
                dict(
                    arm=arm,
                    operation=operation,
                    clock_domain="parent_process",
                    limiting_fraction=fraction,
                    ideal_speedup=1 / (1 - fraction) if fraction < 1 else None,
                    assumption="Remove this entire measured component; all other costs, including unknowns, remain.",
                )
            )
    result["ready"] = (
        max(r["maximum"] for r in result["observer_overhead_rows"]) <= method()["overhead_limit"]
    )
    candidates = [
        r
        for r in result["ideal_speed_bounds"]
        if r["arm"] == "native"
        and r["operation"]
        not in {
            "unattributed_residual",
            "process_startup",
            "process_shutdown",
            "transport_read",
            "transport_write",
        }
    ]
    dominant = max(candidates, key=lambda r: r["limiting_fraction"], default=None)
    result["dominant_component"] = dominant
    if result["ready"] and dominant and dominant["limiting_fraction"] >= 0.1:
        result["next_optimization_decision"] = dict(
            selected_target=dominant["operation"],
            reason="Investigate only this measured limiting component in a future parity-tested task.",
            implemented=False,
            ideal_speedup=dominant["ideal_speedup"],
        )
    return result


def build(work: Json, output: Path) -> Json:
    """A valid overhead limit is a diagnostic null, while owned failures disqualify."""
    measured, validation = work["measurement"], work["validation"]
    summary = aggregate(measured["process_rows"])
    owned = bool(
        work["private_control"]
        or (
            validation["coverage"]["complete"]
            and validation["custody"]["passed"]
            and all(r["passed"] for r in validation["receipts"])
        )
    )
    failures = work["inputs"]["gate_check_summary"] + measured["gate_check_summary"]
    execution = bool(owned and not work["owned_error"] and not failures and summary["complete"])
    ready = execution and summary["ready"] and not work["private_control"]
    verdict = (
        "disqualified"
        if not owned or work["owned_error"]
        else "blocked"
        if failures
        else "positive"
        if ready
        else "null"
    )
    rows = measured["process_rows"]
    workload = work["inputs"].get("workload", {})
    value: Json = dict(
        experiment_id=8770,
        task_id=TASK,
        milestone="2026.11.751",
        run_date="20261103",
        invocation_id=work["invocation_id"],
        honest_verdict=f"complete_{verdict}_optimized_cost_attribution",
        verdict_class=verdict,
        gate_check_summary=failures,
        inference_substrate="verifier_ensemble_against_cached_candidates"
        if summary["complete"]
        else "aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(language_model_calls=0, model_loads=0),
        numerical_training_calls=0,
        rows=rows,
        intended_count=12,
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=sum(r["status"] == "failed" for r in rows),
        censored_count=sum(r["status"] == "censored" for r in rows),
        excluded_count=0,
        independent_count=0,
        independent_process_repeat_count=3 if summary["complete"] else 0,
        verifier_is_oracle=False,
        exposure_scope="original64 exposed requests; installed process diagnostic",
        independent_generalization_score=0,
        residual_cost_ready_score=int(ready),
        acceptance_gates=[
            dict(name="complete_optimized_diagnostic_execution", passed=execution),
            dict(name="low_observer_overhead", passed=summary["ready"]),
        ],
        scientific_promotion_gates=[dict(name="new_speed_headline", passed=False)],
        required_checks_passed=owned and not work["owned_error"],
        flagged_adversarial=bool(
            not owned or work["owned_error"] or work.get("adversarial_findings")
        ),
        validation_receipts=validation["receipts"] + measured["receipts"],
        owned_coverage=validation["coverage"],
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal_validation.json"),
        preconditions_checked=work["preconditions"],
        started_at_utc=work["started_at_utc"],
        ended_at_utc=work["ended_at_utc"],
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=SEED,
        source_artifact_hashes=work["inputs"]["references"],
        code_config_hashes=work["sources"],
        raw_shard_hashes=work["raw_references"] + validation["raw_shard_hashes"],
        cited_upstream_artifacts=work["inputs"]["references"],
        registered_method=method(),
        repository_health_reference=dict(
            historical=work["inputs"].get("repository_health_reference"),
            current_diagnostic=work["preconditions"].get("repository_health_reference"),
            proves_complete_full_suite_pass=False,
        ),
        scoped_validation_scope=dict(
            test_files=TESTS,
            e2e=["E2E-003", "E2E-004", "E2E-018", "E2E-021"],
            repository_health_is_dependency=False,
        ),
        component_cost_rows=summary["component_cost_rows"],
        observer_overhead_rows=summary["observer_overhead_rows"],
        reconciliation_rows=summary["reconciliation_rows"],
        wheel_reference=workload.get("wheel"),
        loaded_extension_reference=dict(
            qualified=workload.get("extension"),
            observed=[
                dict(process_id=r["unit_id"], reference=r.get("loaded_extension_reference"))
                for r in rows
                if r["arm"] == "native"
            ],
            retained_copy=measured["references"][1] if len(measured["references"]) >= 2 else None,
        ),
        dominant_component=summary["dominant_component"],
        ideal_speed_bounds=summary["ideal_speed_bounds"],
        next_optimization_decision=summary["next_optimization_decision"],
        request_latency_rows=measured["request_rows"],
        parity_rows=measured["parity_rows"],
        primitive_reference=reference(Path(work["raw"]) / "work.json"),
        owned_error=work["owned_error"],
        adversarial_findings=work.get("adversarial_findings", []),
        same_batch_10x_claim_this_invocation=False,
        physical_device_execution_score=0,
        sample_size_budget=dict(
            process_cells=12,
            paired_repeats=3,
            panels_per_process=16,
            items_per_panel=64,
            item_evaluations=12288,
            independent_semantic_questions=0,
        ),
        limitations=[
            "Three process pairs support observed ranges, not precise population confidence intervals.",
            "Native crossing includes native residual arithmetic; Python timing cannot separate them.",
            "Observer overhead is measured separately and is not a new speed headline.",
            "Unattributed residuals remain in every optimistic Amdahl denominator.",
        ],
        methodology="Exact qualified Exp8756 wheel/extension and Exp8757 optimized B16 workload. Three counterbalanced process pairs per backend and profiler condition; twelve processes,16 panels of64 requests each. Exclusive leaf spans and explicit residuals reconcile child and complete-parent clocks separately. Parent complete clocks include launch and shutdown; steady request p95 is separate. Instrumented timing is diagnostic only. Observed three-pair overhead ranges suppress target selection above5%. Optimistic complete-process Amdahl bounds remove one entire measured component while retaining all other and unknown costs. No LLM, learning, generalization, physical acceleration or new service speed claim.",
    )
    value["field_principles"] = dict.fromkeys(
        [*value, "field_principles", "reproducibility_checksum"],
        "Bind exact current invocation evidence; distinguish unknown costs and execution acceptance from scientific promotion.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value

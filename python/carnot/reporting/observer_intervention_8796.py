"""REQ-REPORT-8796: preserve missing observer evidence when authority is rejected.

This invocation closes at preflight. Its frozen design does not authorize the
older, one-sided observer gate or invent timing evidence for absent windows.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

from carnot.reporting import durable_cost_components_8784 as previous
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash

Json = dict[str, Any]
__all__ = ["atomic_json", "canonical_hash"]
ROOT = previous.ROOT
NAME, TASK = "experiment_8796_v753_observer_intervention", "exp8796-observer-intervention"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/observer_intervention_8796.py",
    "python/carnot/reporting/observer_intervention_runner_8796.py",
    CLI,
]
TESTS = ["tests/python/test_observer_intervention_8796.py"]
SEED = 7538796
MODEL_SPECS: list[Json] = []
ARMS, OFFSETS, CONDITIONS = (
    ["python", "resident_native"],
    [0, 2032, 4064],
    ["plain", "sham", "clock"],
)
UPSTREAM = ROOT / "results/experiment_8784_v752_durable_cost_components.json"
PIN = "sha256:1078b2e2b97e58f7fc14a4252730faab07c4c1153fc8c303e5d699f7573cb794"
reference, read, verify = previous.reference, previous.read, previous.verify
METHOD = dict(
    arms=ARMS,
    offsets=OFFSETS,
    repeats=2,
    conditions=CONDITIONS,
    updates=32,
    restart_after=16,
    windows=36,
    paired_triples=12,
    maximum_absolute_disturbance=0.05,
    maximum_accounting_discrepancy=0.02,
    maximum_projected_execution_s=2400,
    validation_reserve_s=900,
    buffer="preallocated_block_boundaries",
    sham="identical_dispatch_and_buffers_without_timestamp_reads",
    formatting="after_timed_window",
    independent_workloads=1,
    confidence_scope="this_constructed_workload_only",
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush counts so blocked admission and validation remain visible to the parent."""
    print(f"[exp8796] phase={phase} completed={completed} pending={pending}", flush=True)


def failure(ref: Json, field: str, expected: Any, observed: Any, state: str) -> Json:
    """Retain the rejected operand, including the difference between absence and zero."""
    return dict(
        previous.failure(ref, field, expected, observed, state),
        upstream=previous.TASK,
        check="original_observer_evidence_admission",
    )


def roster() -> list[Json]:
    """All six observer orders occur once per arm before any timing is visible."""
    orders = list(itertools.permutations(CONDITIONS))
    rows: list[Json] = []
    for repeat in range(2):
        for index, offset in enumerate(OFFSETS):
            for arm in ARMS:
                for order, condition in enumerate(orders[3 * repeat + index]):
                    rows.append(
                        dict(
                            unit_id=f"{repeat}:{offset}:{arm}:{condition}",
                            repeat=repeat,
                            offset=offset,
                            arm=arm,
                            condition=condition,
                            order=order,
                            process_order=len(rows),
                            updates=32,
                        )
                    )
    return rows


def inputs() -> Json:
    """Original producer assertions remain authoritative even when the roadmap moves."""
    result: Json = previous.inputs()
    ref = dict(path=str(UPSTREAM), sha256=PIN)
    field, expected = "durable_cost_rows_ready_score", 1
    try:
        verify(ref)
        prior = json.loads(Path(ref["path"]).read_bytes())
        if not isinstance(prior, dict):
            raise ValueError("producer_object_required")
        if field not in prior or prior[field] != 1:
            result["failures"].append(
                failure(
                    ref,
                    field,
                    1,
                    prior.get(field),
                    "missing_field" if field not in prior else "observed_zero",
                )
            )
            return result
        if (
            prior["experiment_id"] != 8784
            or prior["task_id"] != previous.TASK
            or not prior["required_checks_passed"]
            or prior["flagged_adversarial"]
        ):
            raise ValueError("rejected_producer")
        terminal_ref = reference(Path(prior["terminal_validation_sidecar_path"]))
        terminal = read(terminal_ref)
        if not terminal["passed"] or terminal["publication"]["primary_sha256"] != PIN:
            raise ValueError("upstream_terminal_binding")
        primitive = read(prior["primitive_reference"])
        if len(prior["rows"]) != 36 or prior["rows"] != primitive["window_rows"]:
            raise ValueError("original_timing_rows")
        result["refs"] += [ref, terminal_ref, prior["primitive_reference"]]
        result["historical_observer_diagnostics"] = [
            dict(
                row,
                historical_one_sided_passed=row["passed"],
                absolute_disturbance_passed=abs(row["overhead_fraction"]) <= 0.05,
            )
            for row in prior.get("observer_overhead_rows", [])
        ]
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["failures"].append(
            failure(
                ref,
                field,
                expected,
                str(error),
                "missing_file" if isinstance(error, FileNotFoundError) else "rejected_producer",
            )
        )
    return result


def pairs(rows: list[Json]) -> list[Json]:
    """A large negative difference fails the same absolute gate as a positive one."""
    lookup = {(r["repeat"], r["offset"], r["arm"], r["condition"]): r for r in rows}
    result = []
    for repeat, offset, arm in itertools.product(range(2), OFFSETS, ARMS):
        triple = {c: lookup.get((repeat, offset, arm, c), {}) for c in CONDITIONS}
        complete = all(
            r.get("status") == "completed" and r.get("complete_ns", 0) > 0 for r in triple.values()
        )
        row: Json = dict(
            repeat=repeat,
            offset=offset,
            arm=arm,
            independent_workloads=1,
            status="completed" if complete else "censored",
            numerator=None,
            denominator=None,
            missingness=None if complete else "admission_blocked",
            raw_times_ns={c: r.get("complete_ns") for c, r in triple.items()},
        )
        for condition in ["sham", "clock"]:
            signed = (
                triple[condition]["complete_ns"] / triple["plain"]["complete_ns"] - 1
                if complete
                else None
            )
            row[condition + "_signed_fraction"] = signed
            row[condition + "_absolute_fraction"] = abs(signed) if signed is not None else None
        row["disturbance_passed"] = complete and all(
            row[c + "_absolute_fraction"] <= 0.05 for c in ["sham", "clock"]
        )
        result.append(row)
    return result


def build(work: Json, receipts: list[Json], output: Path) -> Json:
    """A blocked result preserves all missing costs and cannot acquire readiness."""
    if not work["failures"]:
        raise ValueError("failed_admission_required")
    raw = Path(work["raw"])
    owned = bool(receipts) and all(r["passed"] for r in receipts)
    covered = any(r["name"] == "coverage_report" and r["passed"] for r in receipts)
    bad = not owned or (not covered and not work["private_control"])
    verdict = "disqualified" if bad else "blocked"
    rows = [
        dict(
            r,
            status="censored",
            censored=True,
            excluded=False,
            passed=False,
            complete_ns=None,
            numerator=None,
            denominator=32,
            missingness="admission_blocked",
            observed_updates=0,
            segments=[],
            commands=[],
            update_rows=[],
            world_count=1,
            workload_count=1,
        )
        for r in roster()
    ]
    value = dict(
        experiment_id=8796,
        task_id=TASK,
        milestone="2026.11.753",
        run_date=work["run_date"],
        invocation_id=raw.name,
        honest_verdict="complete_" + verdict + "_upstream_contract",
        verdict_class=verdict,
        gate_check_summary=work["failures"],
        inference_substrate="deterministic_verifier",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(language_model_calls=0, model_loads=0, generation_calls=0),
        numerical_training_calls=0,
        verifier_is_oracle=True,
        exposure_scope="constructed_durable_observer_preflight_only",
        independent_generalization_score=0,
        rows=rows,
        intended_count=36,
        completed_count=0,
        failed_count=0,
        censored_count=36,
        excluded_count=0,
        independent_count=0,
        n_samples=0,
        observer_rows_ready_score=0,
        component_cost_ready_score=0,
        per_window_rows=rows,
        process_order_rows=roster(),
        paired_observer_rows=pairs(rows),
        positive_control_rows=[],
        component_cost_rows=[],
        unassigned_cost_rows=[],
        parent_clock_reconciliation_rows=[],
        restart_rows=[],
        confidence_limits=dict(
            status="unavailable_no_measurement",
            lower95=None,
            upper95=None,
            scope="this_one_constructed_workload_only",
        ),
        acceptance_gates=dict(
            owned_validation=owned,
            owned_statement_coverage=covered,
            authenticated_sources=False,
            measurement_executed=False,
            observer_disturbance=False,
            component_attribution=False,
        ),
        required_checks_passed=not bad,
        flagged_adversarial=bad,
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        primitive_reference=reference(raw / "measurement.json"),
        validation_reference=reference(raw / "validation.json"),
        publication_output=str(output),
        private_control=work["private_control"],
        frozen_methods=METHOD,
        source_artifact_hashes=work["source_refs"],
        cited_upstream_artifacts=work["source_refs"],
        code_config_hashes=work["code_refs"],
        raw_shard_hashes=work["raw_refs"],
        repository_health_reference=work.get("repository_health_reference", {}),
        scoped_validation_scope=dict(
            owned=OWNED,
            tests=TESTS,
            named_e2es=["E2E-018", "E2E-021"],
            native_e2e_applicable=False,
            full_suite_pass_claimed=False,
        ),
        historical_observer_diagnostics=work.get("historical_observer_diagnostics", []),
        methodology_note="Admission only: original upstream contract rejected; no observer controls or durable windows executed. All costs remain missing. Historical negative overhead is instability, not native acceleration. No population throughput, stage-local gain or complete-service benefit is established.",
    )
    for key in [
        "preconditions_checked",
        "duration_s",
        "started_at_utc",
        "ended_at_utc",
        "phase_spans",
        "resource_projection",
    ]:
        value[key] = work[key]
    value["random_seed"] = SEED
    value["reproducibility_checksum"] = canonical_hash(
        dict(method=METHOD, sources=work["source_refs"], code=work["code_refs"], roster=roster())
    )
    value["field_principles"] = {
        k: "Bind exact invocation and sealed operands; missing measurements remain null and repeated timings describe at most one workload."
        for k in value
    }
    return value

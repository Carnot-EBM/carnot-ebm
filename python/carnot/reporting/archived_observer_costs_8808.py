"""REQ-REPORT-8808: immutable authority qualifies costs, never a speedup claim."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any
from unittest.mock import patch

from carnot.reporting import observer_intervention_8796 as previous
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash

Json = dict[str, Any]
__all__ = ["atomic_json", "canonical_hash"]
ROOT = previous.ROOT
NAME, TASK = "experiment_8808_v754_archived_observer_costs", "exp8808-archived-observer-costs"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/archived_observer_costs_8808.py",
    "python/carnot/reporting/archived_observer_runner_8808.py",
    CLI,
]
TESTS = ["tests/python/test_archived_observer_costs_8808.py"]
SEED = 7548808
MODEL_SPECS: list[Json] = []
ARMS, OFFSETS, CONDITIONS = previous.ARMS, previous.OFFSETS, previous.CONDITIONS
METHOD = dict(
    previous.METHOD, random_seed=SEED, descriptive_only=True, maximum_independent_groups=6
)
historical = previous.previous.historical
request, check_window, starting_states = (
    historical.request,
    historical.check_window,
    historical.starting_states,
)
reference, read, verify, failure = (
    previous.reference,
    previous.read,
    previous.verify,
    previous.previous.failure,
)
AUTHORITIES = {
    "751": dict(
        path=str(ROOT / "openspec/change-proposals/research-roadmap-v751-preserved-20261103.md"),
        sha256="sha256:da649a19cccc6296825a7b650bcab3dcbf5dc9f04e5e9b8e7d0f817bf8036c58",
    ),
    "752": dict(
        path=str(ROOT / "openspec/change-proposals/research-roadmap-v752-preserved-20261103.md"),
        sha256="sha256:9643522778637bc54e622a7a5dc62faf1279944464c6df2a5028d590478d9872",
    ),
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Counts expose actual work so a waiting parent can detect a stopped child."""
    if phase in {"sources_plan_frozen", "window_before", "window_after"}:
        pending = max(0, len(roster()) - completed)
    print(f"[exp8808] phase={phase} completed={completed} pending={pending}", flush=True)


def inputs() -> Json:
    """Change only design lookup; every original producer assertion still executes."""
    refs = list(AUTHORITIES.values())
    for ref in refs:
        try:
            verify(ref)
        except (OSError, ValueError) as error:
            return dict(
                refs=[],
                failures=[
                    failure(
                        ref,
                        "immutable_design_bytes",
                        ref["sha256"],
                        str(error),
                        "missing_file"
                        if isinstance(error, FileNotFoundError)
                        else "rejected_producer",
                    )
                ],
            )
    with patch.object(previous.previous.identity.contract, "DESIGN", AUTHORITIES["752"]["path"]):
        value: Json = previous.inputs()
    value["refs"] += refs
    value["immutable_authority_refs"] = refs
    return value


def roster() -> list[Json]:
    """Freeze process order before timing while retaining all six observer orders."""
    rows = previous.roster()
    groups = [(rep, off, arm) for rep in range(2) for off in OFFSETS for arm in ARMS]
    random.Random(SEED).shuffle(groups)
    rows.sort(key=lambda r: (groups.index((r["repeat"], r["offset"], r["arm"])), r["order"]))
    return [
        dict(r, process_order=i, cluster_id=f"{r['repeat']}:{r['offset']}", seed=SEED)
        for i, r in enumerate(rows)
    ]


def projection(latencies: list[int], complete_ns: int, *, one_time_s: float = 0) -> Json:
    """The cheap probe estimates every scheduled process before execution is admitted."""
    value: Json = previous.previous.projection(latencies, complete_ns)
    value["validation_reserve_s"] = 900
    value["projected_execution_s"] = value["projected_with_reserve_s"]
    value["one_time_authority_custody_s"] = one_time_s
    value["projected_with_reserve_s"] += 900 + one_time_s
    value["admitted"] = value["projected_execution_s"] <= 2400
    return value


def pairs(rows: list[Json]) -> list[Json]:
    """Clock dispatch must also remain stable against its matched sham dispatch."""
    result: list[Json] = previous.pairs(rows)
    for row in result:
        times = row["raw_times_ns"]
        signed = times["clock"] / times["sham"] - 1 if row["status"] == "completed" else None
        row.update(
            clock_vs_sham_signed_fraction=signed,
            clock_vs_sham_absolute_fraction=abs(signed) if signed is not None else None,
        )
        row["disturbance_passed"] = (
            row["disturbance_passed"] and signed is not None and abs(signed) <= 0.05
        )
    return result


def controls(rows: list[Json]) -> tuple[list[Json], list[Json], list[Json]]:
    """Compare complete checkpoints and outputs before attributing any measured cost."""
    parity, reconcile, residual = [], [], []
    baselines = {
        (r["repeat"], r["offset"]): r
        for r in rows
        if r["condition"] == "plain" and r["arm"] == "python"
    }
    for row in rows:
        accounted_total = 0
        baseline = baselines.get((row["repeat"], row["offset"]), {})
        states = [s["checkpoint"] for s in row["segments"]]
        expected = [s["checkpoint"] for s in baseline.get("segments", [])]

        def semantic(r: Json) -> list[Json]:
            """Timing fields cannot affect exact output equality."""
            return [
                {
                    k: u[k]
                    for k in (
                        "update_index",
                        "probability_before",
                        "probability_after",
                        "decision_before",
                        "decision_after",
                        "state_hash",
                        "serialized_hash",
                    )
                }
                for u in r.get("update_rows", [])
                if u["status"] == "completed"
            ]

        parity.append(
            dict(
                unit_id=row["unit_id"],
                passed=row["status"] == "completed"
                and row["restart_verified"]
                and len(states) == 2
                and states == expected
                and semantic(row) == semantic(baseline),
            )
        )
        for segment, command in zip(row["segments"], row["commands"], strict=True):
            part = segment["components"]
            parent = command["ended_monotonic_ns"] - command["started_monotonic_ns"]
            startup = segment["begin_ns"] - command["started_monotonic_ns"]
            shutdown = command["ended_monotonic_ns"] - segment["end_ns"]
            accounted = sum(part["costs_ns"].values()) + part["unassigned_ns"]
            accounted_total += accounted
            discrepancy = abs(parent - accounted) / parent
            reconcile.append(
                dict(
                    unit_id=row["unit_id"],
                    start=segment["start"],
                    parent_ns=parent,
                    startup_ns=startup,
                    shutdown_ns=shutdown,
                    accounted_ns=accounted,
                    discrepancy_fraction=discrepancy,
                    passed=discrepancy <= 0.02,
                )
            )
            residual.append(
                dict(
                    unit_id=row["unit_id"],
                    start=segment["start"],
                    cost_ns=part["unassigned_ns"],
                    measured_gap=True,
                )
            )
        if row["status"] == "completed":
            orchestration = row["complete_ns"] - sum(
                c["ended_monotonic_ns"] - c["started_monotonic_ns"] for c in row["commands"]
            )
            accounted_window = accounted_total + orchestration
            discrepancy = abs(row["complete_ns"] - accounted_window) / row["complete_ns"]
            residual.append(
                dict(
                    unit_id=row["unit_id"],
                    start="parent_orchestration",
                    cost_ns=orchestration,
                    measured_gap=True,
                )
            )
            reconcile.append(
                dict(
                    unit_id=row["unit_id"],
                    start="full_window",
                    parent_ns=row["complete_ns"],
                    accounted_ns=accounted_window,
                    discrepancy_fraction=discrepancy,
                    passed=orchestration >= 0 and discrepancy <= 0.02,
                )
            )
    return parity, reconcile, residual


def build(work: Json, receipts: list[Json], output: Path) -> Json:
    """Complete but unstable observer costs remain a descriptive null result."""
    raw, rows = Path(work["raw"]), work["window_rows"]
    parity, reconcile, residual = controls(rows)
    paired = pairs(rows)
    owned = bool(receipts) and all(r["passed"] for r in receipts)
    covered = any(r["name"] == "coverage_report" and r["passed"] for r in receipts)
    complete = sum(r["status"] == "completed" for r in rows)
    execution = (
        complete == 36
        and all(r["passed"] for r in parity)
        and all(u["passed"] for r in rows for u in r["update_rows"])
        and all(len(r["update_rows"]) == 32 for r in rows)
    )
    positive = work.get("positive_control_rows", [])
    control_pass = bool(positive) and all(r["passed"] for r in positive)
    bad = (
        not owned
        or (not covered and not work["private_control"])
        or (
            not work["failures"]
            and (not execution or not control_pass)
            and not work["private_control"]
        )
    )
    ready = (
        owned
        and covered
        and execution
        and control_pass
        and not work["failures"]
        and not work["private_control"]
    )
    component = (
        ready
        and all(p["disturbance_passed"] for p in paired)
        and len(reconcile) == 108
        and all(r["passed"] for r in reconcile)
    )
    verdict = "disqualified" if bad else "blocked" if work["failures"] else "null"
    source_refs = work["source_refs"]
    value = dict(
        experiment_id=8808,
        task_id=TASK,
        milestone="2026.11.754",
        run_date=work["run_date"],
        invocation_id=raw.name,
        honest_verdict="complete_" + verdict + "_archived_observer_costs",
        verdict_class=verdict,
        inference_substrate="deterministic_verifier",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(language_model_calls=0, model_loads=0, generation_calls=0),
        numerical_training_calls=0,
        verifier_is_oracle=True,
        exposure_scope="one_constructed_ordinary_durable_workload",
        independent_generalization_score=0,
        rows=rows,
        per_window_rows=rows,
        intended_count=len(work["roster"]),
        completed_count=complete,
        censored_count=sum(r["status"] == "censored" for r in rows),
        failed_count=sum(r["status"] == "failed" for r in rows),
        excluded_count=0,
        independent_count=min(
            6, len({r["cluster_id"] for r in rows if r["status"] == "completed"})
        ),
        observer_rows_ready_score=int(ready),
        component_cost_ready_score=int(component),
        state_parity_rows=parity,
        paired_observer_rows=paired,
        parent_clock_reconciliation_rows=reconcile,
        positive_control_rows=positive,
        unassigned_cost_rows=residual,
        gate_check_summary=[
            dict(r, artifact_field=r.get("artifact_field", r["field"])) for r in work["failures"]
        ],
        acceptance_gates=dict(
            owned_validation=owned,
            owned_statement_coverage=covered,
            authenticated_sources=not work["failures"],
            full_operation_state_controls=execution,
            positive_control=control_pass,
            observer_disturbance=all(r["disturbance_passed"] for r in paired),
            parent_accounting=len(reconcile) == 108 and all(r["passed"] for r in reconcile),
        ),
        required_checks_passed=not bad,
        flagged_adversarial=bad,
        validation_receipts=receipts,
        owned_statement_coverage=dict(
            percent=100 if covered else None,
            zero_exclusions=True,
            reference=reference(raw / "coverage.json")
            if (raw / "coverage.json").exists()
            else None,
        ),
        immutable_authority_refs=work.get("immutable_authority_refs", []),
        source_artifact_hashes=source_refs,
        code_config_hashes=work["code_refs"],
        raw_shard_hashes=work["raw_refs"],
        random_seed=SEED,
        reproducibility_checksum=canonical_hash(
            dict(method=METHOD, sources=source_refs, code=work["code_refs"], roster=work["roster"])
        ),
        preconditions_checked=work["preconditions"],
        phase_spans=work["phase_spans"],
        duration_s=work["duration_s"],
        started_at_utc=work["started_at_utc"],
        ended_at_utc=work["ended_at_utc"],
        resource_projection=work["resource_projection"],
        primitive_reference=reference(raw / "measurement.json"),
        validation_reference=reference(raw / "validation.json"),
        publication_output=str(output),
        private_control=work["private_control"],
        frozen_methods=METHOD,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        repository_health_reference=work.get("repository_health_reference", {}),
        process_order_rows=work["roster"],
        ordinary_starting_state_rows=work["starting_state_rows"],
        prefix_preparation_cost_rows=work["excluded_prefix_cost_rows"],
        historical_observer_diagnostics=work.get("historical_observer_diagnostics", []),
        physical_device_execution=False,
        scoped_validation_scope=dict(
            owned=OWNED,
            tests=TESTS,
            named_e2es=["E2E-018", "E2E-021"],
            native_e2e_applicable=False,
            full_suite_pass_claimed=False,
        ),
        methodology_note="Counterbalanced plain parent timing, sham dispatch and buffered clocks. Full parent windows include real restart, acknowledgement and durable barriers. Startup, imports, custody and prefix costs are recorded separately and are never multiplied by updates. Negative overhead is instability. Six process/state groups describe one constructed workload, with no population speedup or physical-device claim.",
    )
    value["cited_upstream_artifacts"] = [
        dict(experiment_id=n, fields_imported=fields, **ref)
        for n, fields, ref in work.get("citations", [])
    ]
    value["field_principles"] = {
        k: "Exact sealed operands and measured boundaries; missing costs remain null, repeated windows do not establish population benefit."
        for k in value
    }
    return value

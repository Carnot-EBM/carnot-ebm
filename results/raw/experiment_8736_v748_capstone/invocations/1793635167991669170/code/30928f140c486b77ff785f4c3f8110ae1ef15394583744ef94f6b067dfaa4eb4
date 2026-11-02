"""REQ-REPORT-8371: hardware names cannot substitute for compatible operations.

Historical CPU and Ising receipts remain useful evidence. They cannot establish
spline execution or a service speedup without the missing operation costs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Any, cast
from unittest.mock import patch

from carnot.reporting import kv260_local_cost_boundary_8315 as base
from carnot.reporting import kv260_workload_cost_8329 as boards
from carnot.reporting import kv260_table_cost_8356 as tables
from carnot.reporting import v721_contract_methods as authority
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
)

Json = dict[str, Any]
ROOT = base.ROOT
NAME = "experiment_8371_v721_hardware_operation_boundary"
TASK, MILESTONE = "exp8371-hardware-operation-boundary", "2026.10.721"
CLI = "scripts/experiments/" + NAME + ".py"
TEST = "tests/python/test_hardware_operation_boundary_8371.py"
OWNED = [
    "python/carnot/reporting/hardware_operation_boundary_8371.py",
    "python/carnot/reporting/hardware_operation_runner_8371.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
PIN = "sha256:0caf392aa696e79a1aa8ebbaa14b89deb40974466fbd4bb538813a7cd293e9e9"
TASK_PIN = "sha256:d562b49e9305156a5ce1919bcd42ef2259a1fa5afa4725be9084f5bcdb258571"
HISTORY = "results/experiment_8356_v720_kv260_workload_cost.json"
SOURCES = [
    "experiment_8364_v721_continuous_table_learning",
    "experiment_8365_v721_local_service_cost",
    "experiment_8367_v721_native_service_cost",
]
OPERATIONS = [
    "coefficient_reads",
    "int16_table_reads",
    "interpolation",
    "derivative_bound_refresh",
    "direct_fallback",
    "state_persistence",
    "transport",
]
REUSED = [
    "python/carnot/reporting/primary_publication.py",
    "scripts/adversarial_verify.py",
    "scripts/verdict_row_consistency_lint.py",
    "python/carnot/reporting/v709_execution.py",
    "python/carnot/reporting/v717_contract_runner.py",
    "python/carnot/reporting/v718_replay_runner.py",
    "python/carnot/reporting/v721_contract_methods.py",
    "python/carnot/reporting/kv260_table_cost_8356.py",
    "ops/exclusion_manifest.yaml",
]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush counts so evidence authentication cannot look like stalled computation."""
    print(f"[exp8371] phase={phase} completed={completed} pending={pending}", flush=True)


def gate(path: Path, field: str, expected: Any, observed: Any) -> Json:
    """Missing operands remain null so no reader can confuse absence with zero."""
    return dict(
        upstream=path.stem,
        path=str(path),
        hash=base.sha256_file(path) if path.is_file() else None,
        artifact_field=field,
        op="==",
        expected=expected,
        observed=observed,
        passed=observed == expected,
    )


def operation_map(rows: list[Json]) -> tuple[list[Json], float | None, list[str]]:
    """A complete typed denominator is required even when no installed operation fits."""
    costs: Json = {}
    for row in rows:
        op, cost = row.get("operation"), row.get("cost_ns")
        if (
            op not in OPERATIONS
            or op in costs
            or type(cost) not in (int, float)
            or not math.isfinite(cast(float, cost))
            or cast(float, cost) < 0
        ):
            raise ValueError("operation_trace_schema")
        costs[op] = cost
    mapped = [
        dict(
            operation=op,
            assigned_substrate="host_CPU",
            kv260_supported=False,
            cost_ns=costs.get(op),
            cost_status="measured" if op in costs else "absent",
            reason="authenticated_quadratic_Ising_k_max_le_5_has_no_spline_or_durable_file_protocol",
        )
        for op in OPERATIONS
    ]
    missing = [op for op in OPERATIONS if op not in costs]
    return mapped, 0.0 if not missing and sum(costs.values()) > 0 else None, missing


def board_history(root: Path, raw: Path, work: Json) -> Json:
    """Recheck the old terminal and physical dispatch; do not repeat board timings."""
    path = root / HISTORY
    if base.sha256_file(path) != PIN:
        raise ValueError("historical_primary_hash")
    value = base.probe(path, raw, work, None)
    if value is None:
        raise ValueError("historical_terminal")
    if not tables.complete(value["table_timing_rows"]):
        raise ValueError("historical_five_distinct_repeats")
    for ref in value["polarfire_terminal_evidence_hashes"]:
        base.pin(base.checked(ref), raw, work["refs"])
    graduation = boards.polarfire(root, raw, work)
    if graduation != value["polarfire_graduation"]:
        raise ValueError("polarfire_graduation_changed")
    kv = value["board_obligations"]["kv260"]["historical"]["historical"]
    for location, digest in [
        (kv["source_transcript"], kv["source_transcript_sha256"]),
        (str(root / kv["source_path"]), kv["source_hash"]),
    ]:
        base.pin(base.checked(dict(path=location, sha256=digest)), raw, work["refs"])
    return dict(
        primary_sha256=PIN,
        graduation=graduation,
        kv260=kv,
        historical_model_provenance=value["historical_model_provenance"],
        historical_cpu_operation_rows=value["operation_rows"],
    )


def traces(root: Path, raw: Path, work: Json) -> list[Json]:
    """Only the declared operation reference supplies costs; aggregates are insufficient."""
    imported = []
    for index, name in enumerate(SOURCES):
        path = root / "results" / (name + ".json")
        progress("before_source_" + name, index, len(SOURCES) - index)
        branch: Json = dict(path=str(path), present=path.is_file(), reference=None, rows=[])
        value = base.probe(path, raw, work, None)
        if value is not None:
            try:
                if value["milestone"] != MILESTONE or value["experiment_id"] != int(
                    name.split("_")[1]
                ):
                    raise ValueError("producer_identity")
                ref = value.get("operation_level_workload_reference")
                branch["reference"] = ref
                if not isinstance(ref, dict):
                    raise ValueError("absent_operation_level_workload_reference")
                trace = json.loads(base.pin(base.checked(ref), raw, work["refs"]).read_bytes())
                operation_map(trace["operation_rows"])
                branch["rows"] = trace["operation_rows"]
            except (OSError, ValueError, KeyError, TypeError) as error:
                work["checks"].append(
                    gate(
                        path,
                        "operation_level_workload_reference",
                        "authenticated_typed_operation_rows",
                        str(error),
                    )
                )
        imported.append(branch)
        progress("after_source_" + name, index + 1, len(SOURCES) - index - 1)
    return imported


def measure(root: Path, raw: Path) -> Json:
    """Authenticate authority and each branch independently before mapping costs."""
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True)
    work: Json = dict(root=str(root), checks=[], refs=[], history={}, contract={}, branches=[])
    progress("authority_before")
    try:
        contract = authority.authority(root, raw / "authority")
        task = next(t for t in contract["tasks"] if t["id"] == TASK)
        work["contract"] = dict(
            activated=contract["activated"],
            canonical_tasks_sha256=contract["canonical_tasks_sha256"],
            task_sha256=canonical_hash(task),
        )
        if not contract["activated"] or canonical_hash(task) != TASK_PIN:
            raise ValueError("activated_task_authority")
        for name in [authority.DESIGN, authority.ACTIVE, authority.legacy.PROTOCOL]:
            base.pin(root / name, raw, work["refs"])
        if base.sha256_file(root / authority.legacy.PROTOCOL) != authority.legacy.base.PIN:
            raise ValueError("V717_protocol_changed")
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        work["checks"].append(
            gate(root / authority.DESIGN, "activated_task_authority", TASK_PIN, str(error))
        )
    progress("authority_after_history_before")
    try:
        work["history"] = board_history(root, raw, work)
    except (OSError, ValueError, KeyError, TypeError) as error:
        work["checks"].append(
            gate(root / HISTORY, "historical_board_terminal", "valid", str(error))
        )
    progress("history_after_operation_traces_before")
    work["branches"] = traces(root, raw, work)
    for key in ["compatible_spline_kernel", "complete_request_transfer_protocol"]:
        work["checks"].append(gate(root / HISTORY, key, "authenticated", None))
    work.update(
        started_monotonic_ns=start,
        ended_monotonic_ns=time.monotonic_ns(),
        code_config_hashes=[base.reference(ROOT / p) for p in [*OWNED, TEST, *REUSED]],
    )
    atomic_json(raw / "measurement.json", work)
    progress("operation_map_after", len(work["branches"]), 0)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Execution qualification preserves missing costs and historical CPU graduation."""
    owned = (
        bool(receipts)
        and all(r["passed"] for r in receipts)
        and work.get("validation_preflight_complete", True)
    )
    history = work["history"]
    rows: list[Json] = []
    operations: list[Json] = []
    fractions: list[float | None] = []
    missing: set[str] = set()
    for branch in work["branches"]:
        mapped, fraction, absent = operation_map(branch["rows"])
        operations.extend(
            dict(
                r,
                upstream=Path(branch["path"]).stem,
                operation_level_workload_reference=branch["reference"],
            )
            for r in mapped
        )
        fractions.append(fraction)
        missing.update(absent)
        ready = bool(branch["rows"] and not absent)
        rows.append(
            dict(
                source_id=Path(branch["path"]).stem,
                arm="operation_trace",
                intended=1,
                completed=ready,
                failed=False,
                censored=not ready,
                excluded=False,
                independent=0,
                numerator=int(ready),
                denominator=1,
            )
        )
    for board in ["kv260", "polarfire"]:
        ready = bool(history)
        rows.append(
            dict(
                source_id=board,
                arm="historical_accounting",
                intended=1,
                completed=ready,
                failed=False,
                censored=not ready,
                excluded=False,
                independent=0,
                numerator=int(ready),
                denominator=1,
            )
        )
    failed = any(not r["passed"] for r in receipts)
    kind = (
        "disqualified"
        if failed
        else "blocked"
        if any(not c["passed"] for c in work["checks"])
        else "circular_positive"
    )
    graduation = history.get("graduation", {})
    value: Json = dict(
        experiment_id=8371,
        task_id=TASK,
        milestone=MILESTONE,
        run_date="20261010",
        honest_verdict="complete_" + kind + "_hardware_operation_boundary",
        verdict_class=kind,
        gate_check_summary=work["checks"],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        no_model_load=True,
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        historical_model_provenance=history.get("historical_model_provenance", []),
        rows=rows,
        intended_count=len(rows),
        completed_count=sum(r["completed"] for r in rows),
        failed_count=0,
        censored_count=sum(r["censored"] for r in rows),
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            branches=3, historical_boards=2, timing_repeats_are_independent=False
        ),
        verifier_is_oracle=True,
        exposure_scope="exposed_cached_development",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=owned,
        flagged_adversarial=bool(work.get("adversarial_findings")),
        acceptance_gates=dict(
            owned_checks=owned,
            authority=work["contract"].get("activated", False),
            historical_boards=bool(history),
            compatible_kernel=False,
            complete_transport=False,
            scientific_benefit=False,
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(
            output.parent / "raw" / output.stem / "terminal_validation.json"
        ),
        adversarial_findings=work.get("adversarial_findings", []),
        preconditions_checked=True,
        duration_s=(work["ended_monotonic_ns"] - work["started_monotonic_ns"]) / 1e9,
        phase_spans=[
            dict(
                phase="authenticate_and_map",
                started_monotonic_ns=work["started_monotonic_ns"],
                ended_monotonic_ns=work["ended_monotonic_ns"],
            )
        ],
        random_seed=7218371,
        source_artifact_hashes=work["refs"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=[base.reference(raw / "measurement.json")],
        cited_upstream_artifacts=[
            dict(r, imported_fields=["terminal history or declared operation reference"])
            for r in work["refs"]
        ],
        operation_level_workload_reference=[
            dict(path=b["path"], present=b["present"], reference=b["reference"])
            for b in work["branches"]
        ],
        operation_rows=operations,
        compatible_fraction=0.0 if all(f is not None for f in fractions) else None,
        compatible_fraction_status="measured"
        if all(f is not None for f in fractions)
        else "unmeasured",
        unmeasured_operations=[op for op in OPERATIONS if op in missing],
        kv260_execution_ready_score=0,
        board_obligations=dict(
            kv260=dict(
                k_max=5,
                historical=history.get("kv260"),
                current_execution=False,
                missing=["compatible_spline_kernel", "complete_request_transfer_protocol"],
                access_precondition=[
                    "ssh",
                    "-o",
                    "ConnectTimeout=5",
                    "-o",
                    "BatchMode=yes",
                    "kria",
                    "true",
                ],
                further_prerequisites=[
                    "exclusive_lease",
                    "exact_firmware_identity",
                    "owned_reversible_job",
                    "dispatch_output_hashes",
                    "transfer_clocks",
                ],
            ),
            polarfire=dict(
                status="authenticated_CPU_dispatch"
                if graduation
                else "blocked_terminal_authentication",
                scope="board_local_Linux_CPU_only",
                reopened=False,
            ),
            gatemate=dict(
                task_id="exp8372-gatemate-missing-evidence", status="separate_physical_obligation"
            ),
        ),
        polarfire_workload_validated=graduation.get("polarfire_workload_validated", False),
        polarfire_graduation=graduation,
        polarfire_terminal_evidence_hashes=[
            r for r in work["refs"] if "8259" in r["original_path"]
        ],
        actual_substrate="host_CPU_aggregation",
        current_device_execution_count=0,
        full_service_speedup=None,
        acquisition_relevance="defer_no_measured_compatible_service_benefit_no_V721_acquisition",
        NPU_qualified=False,
        TSU_qualified=False,
        historical_cpu_operation_rows=history.get("historical_cpu_operation_rows", []),
        current_contract_ready_score=int(owned and work["contract"].get("activated", False)),
        work_reference=base.reference(raw / "measurement.json"),
        publication_output=str(output),
        execution_manifest_reference=work.get("execution_manifest_reference"),
        owned_coverage_reference=work.get("owned_coverage_reference"),
        methodology_note="Read authenticated upstream artifacts only. Existing quadratic Ising firmware is incompatible with these local spline operations. CPU10x cannot establish fabric or total LLM service savings.",
    )
    value["field_principles"] = {
        k: "Bind exact authority and immutable evidence; absent costs are null and CPU qualification grants no fabric or generalization credit."
        for k in [*value, "field_principles", "reproducibility_checksum"]
    }
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Rehashed summaries must still match pinned authority, source bytes and dispatch."""
    try:
        value = json.loads(path.read_bytes())
        for ref in [
            value["work_reference"],
            *value["source_artifact_hashes"],
            *value["code_config_hashes"],
        ]:
            base.checked(ref)
        work = json.loads(base.checked(value["work_reference"]).read_bytes())
        if work["refs"] != value["source_artifact_hashes"]:
            return False
        by_original = {r["original_path"]: r for r in work["refs"]}
        for ref in work["refs"]:
            if base.sha256_file(Path(ref["original_path"])) != ref["sha256"]:
                return False
        for field in ["execution_manifest_reference", "owned_coverage_reference"]:
            if work.get(field):
                base.checked(work[field])
        for receipt in value["validation_receipts"]:
            for stream in ["stdout", "stderr"]:
                if stream + "_path" in receipt:
                    base.checked(
                        dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
                    )
        with TemporaryDirectory(prefix="exp8371-cold-") as directory:
            private = Path(directory)
            root = Path(work["root"])
            if str(root / authority.DESIGN) in by_original:
                for name in [authority.DESIGN, authority.ACTIVE]:
                    target = private / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(base.checked(by_original[str(root / name)]).read_bytes())
                checked = authority.authority(private, private / "authority")
                task = next(t for t in checked["tasks"] if t["id"] == TASK)
                if (
                    dict(
                        activated=checked["activated"],
                        canonical_tasks_sha256=checked["canonical_tasks_sha256"],
                        task_sha256=canonical_hash(task),
                    )
                    != work["contract"]
                    or canonical_hash(task) != TASK_PIN
                ):
                    return False
                protocol = base.checked(by_original[str(root / authority.legacy.PROTOCOL)])
                if base.sha256_file(protocol) != authority.legacy.base.PIN:
                    return False
            if work["history"]:
                actual = board_history(root, private / "boards", dict(checks=[], refs=[]))
                if actual != work["history"]:
                    return False
            actual_branches = traces(root, private / "sources", dict(checks=[], refs=[]))
            if actual_branches != work["branches"]:
                return False
        rebuilt = build(
            work,
            value["validation_receipts"],
            Path(value["work_reference"]["path"]).parent,
            Path(value["publication_output"]),
        )
        return bool(rebuilt == value)
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return False

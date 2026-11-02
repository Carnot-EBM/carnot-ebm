"""REQ-REPORT-8454: installed operations constrain the new journal service.

Original reader bytes and authority remain separate from current admission.
A host representation change cannot install a missing FPGA operation.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import board_reader_qualification_8440 as historical
from carnot.reporting import v727_contract_methods as authority
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
)
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
ROOT, base = historical.ROOT, historical.base
NAME, TASK, MILESTONE = (
    "experiment_8454_v727_board_operation_boundary",
    "exp8454-board-operation-boundary",
    "2026.10.727",
)
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_board_operation_boundary_8454.py"
OWNED = [
    "python/carnot/reporting/board_operation_boundary_8454.py",
    "python/carnot/reporting/board_operation_runner_8454.py",
    CLI,
]
READER = "results/experiment_8440_v726_board_reader_qualification.json"
READER_PIN = "sha256:f1fe951405765fc39318a440f90c3baa2cb34d713f75945de638aa3b74a887bb"
COST = "results/experiment_8451_v727_journal_service_cost.json"
OPERATIONS = ["serialization", "journal_append", "fsync", "checkpoint", "arithmetic", "transport"]
MODEL_SPECS: list[Json] = []
reference, failure = base.reference, historical.gate


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual phase counts so cached reductions never appear stalled."""
    print(f"[exp8454] phase={phase} completed={completed} pending={pending}", flush=True)


def checksum(value: Json) -> str:
    """Rehashing alone does not grant authority; replay also rebuilds operands."""
    return str(canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"}))


def reader(root: Path, raw: Path) -> Json:
    """The shipped reader replays only under its sealed original source aliases."""
    path = root / READER
    result: Json = dict(
        qualified=False, checks=[], refs=[], models=[], kv260={}, polarfire={}, host_bottleneck={}
    )
    try:
        value = json.loads(
            base.pin(
                historical.old.authority.checked(path, READER_PIN), raw, result["refs"]
            ).read_bytes()
        )
        work = json.loads(base.checked(value["work_reference"]).read_bytes())
        with historical.old.prior.prior.source_aliases(
            value["source_artifact_hashes"], work["absent"]
        ):
            valid = historical.replay(path)
        if (
            not valid
            or value["board_reader_ready_score"] != 1
            or not value["required_checks_passed"]
            or value["flagged_adversarial"]
        ):
            raise ValueError("original_reader_qualification")
        result.update(
            qualified=True,
            kv260=value["board_obligations"]["kv260"]["historical"],
            polarfire=value["polarfire_graduation"],
            models=value["historical_model_provenance"],
            host_bottleneck=value["optional_current_cost_evidence"],
        )
        for ref in value["source_artifact_hashes"]:
            base.pin(base.checked(ref), raw, result["refs"])
        result["refs"].append(reference(base.checked(value["work_reference"])))
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["qualified"] = False
        result["checks"].append(
            failure(path, "board_reader_ready_score", 1, None if not path.is_file() else str(error))
        )
    return result


def costs(root: Path, raw: Path) -> Json:
    """Only qualified primitive clocks are imported; a blocked branch keeps its exact field."""
    path = root / COST
    result: Json = dict(
        qualified=False,
        rows=[],
        refs=[],
        checks=[],
        observed=None,
        missing_field="service_cost_ready_score",
    )
    try:
        value = json.loads(base.pin(path, raw, result["refs"]).read_bytes())
        result["observed"] = value.get("service_cost_ready_score")
        if result["observed"] != 1:
            raise ValueError("service_cost_ready_score")
        checked = base.probe(
            path,
            raw,
            dict(refs=result["refs"], checks=result["checks"]),
            "service_cost_ready_score",
        )
        receipt = child(
            "optional_cost_cold",
            [
                str(ROOT / ".venv/bin/python"),
                "-u",
                str(ROOT / "scripts/experiments/experiment_8451_v727_journal_service_cost.py"),
                "--cold-replay",
                str(path),
            ],
            raw / "cost-reader",
            deadline=60,
            scope="upstream",
        )
        if (
            checked is None
            or not receipt["passed"]
            or value["verdict_class"] not in {"positive", "circular_positive", "null"}
        ):
            raise ValueError("cost_primitive_replay")
        result["missing_field"] = "paired_repeat_rows[].stage_ns"
        rows = [
            dict(operation=op, cost_ns=ns, arm=row.get("arm"), repeat=row.get("repeat"))
            for row in value["paired_repeat_rows"]
            for op, ns in row.get("stage_ns", {}).items()
        ]
        if not rows:
            raise ValueError("paired_repeat_rows[].stage_ns")
        operation_mapping(rows, result["missing_field"])
        result.update(qualified=True, rows=rows, missing_field=None, replay_receipt=receipt)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["checks"].append(
            failure(
                path,
                result["missing_field"],
                1
                if result["missing_field"] == "service_cost_ready_score"
                else "measured_primitive_clocks",
                result["observed"]
                if result["missing_field"] == "service_cost_ready_score"
                else str(error),
            )
        )
    return result


def operation_mapping(rows: list[Json], missing: str | None) -> list[Json]:
    """Each operation keeps its host clock; board CPU dispatch grants no new kernel."""
    for row in rows:
        if type(row["cost_ns"]) is not int or row["cost_ns"] < 0:
            raise ValueError("invalid_primitive_clock")
    intended = rows or [dict(operation=op, cost_ns=None) for op in OPERATIONS]
    return [
        dict(
            row,
            board=board,
            assigned_substrate="host_CPU",
            device_supported=False,
            compatibility="host_only_no_installed_compatible_operation",
            board_cpu_scope="historical_Linux_CPU_dispatch_only" if board == "PolarFire" else None,
            missing_reason=missing if row["cost_ns"] is None else None,
            device_clock_ns=None,
            transport_clock_ns=None,
        )
        for board in ["KV260", "PolarFire"]
        for row in intended
    ]


def resources(private: Path) -> Json:
    """Private disk and memory are checked without probing optional boards or CUDA."""
    mounts = [line.split() for line in Path("/proc/self/mountinfo").read_text().splitlines()]
    mount = max(
        (m for m in mounts if private.resolve().is_relative_to(Path(m[4]))), key=lambda m: len(m[4])
    )
    filesystem = mount[mount.index("-") + 1]
    mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    (private / "write_probe").write_bytes(b"private disk scratch")
    return dict(
        disk_backed=filesystem not in {"tmpfs", "ramfs"},
        filesystem=filesystem,
        memory_available_bytes=int(mem["MemAvailable"].split()[0]) * 1024,
        free_disk_bytes=shutil.disk_usage(private).free,
        tools_available=all(
            shutil.which(str(ROOT / ".venv/bin" / n))
            for n in ["python", "pytest", "coverage", "ruff", "mypy"]
        ),
        private_scratch=str(private),
        measurement_cap_s=3600,
        task_cap_s=4800,
        validation_reserve_s=900,
        no_model_load=True,
    )


def contract(root: Path, raw: Path) -> Json:
    """Keep the full task digest visible even when the independent design is absent."""
    value = authority.authority(root, raw)
    return {
        k: value[k]
        for k in [
            "activated",
            "canonical_tasks_sha256",
            "observed_count",
            "observed_active_tasks_sha256",
            "gate_check_summary",
        ]
    } | dict(
        task_sha256=canonical_hash(next((t for t in value["tasks"] if t["id"] == TASK), {})),
        expected_tasks_sha256=authority.TASKS_PIN,
    )


def measure(root: Path, raw: Path, private: Path | None = None) -> Json:
    """Seal invocation operands before tests; optional costs never gate history."""
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    progress("preconditions", 0, 3)
    pre = resources(private or raw)
    current = contract(root, raw / "authority")
    history = reader(root, raw / "reader")
    progress("historical_reader_sealed", 1, 2)
    optional = costs(root, raw / "cost")
    refs = [*history["refs"], *optional["refs"]]
    names = [authority.DESIGN, authority.ACTIVE, authority.STAGED, READER, COST]
    absent = [str((root / n).absolute()) for n in names if not (root / n).is_file()]
    for name in names[:3]:
        if (root / name).is_file():
            base.pin(root / name, raw, refs)
    checks = [
        failure(Path(pre["private_scratch"]), key, True, pre[key])
        for key in ["disk_backed", "tools_available"]
    ]
    checks += [
        failure(
            raw, "available_memory_at_least_1GiB", True, pre["memory_available_bytes"] >= 1024**3
        ),
        failure(raw, "available_disk_at_least_1GiB", True, pre["free_disk_bytes"] >= 1024**3),
    ]
    work: Json = dict(
        root=str(root.absolute()),
        contract=current,
        reader=history,
        cost=optional,
        refs=refs,
        absent=absent,
        checks=checks,
        preconditions=pre,
        started_monotonic_ns=start,
        ended_monotonic_ns=time.monotonic_ns(),
        phase_spans=[],
        invocation_argv=list(__import__("sys").argv),
        code_config_hashes=[
            reference(ROOT / p)
            for p in [
                *OWNED,
                TEST,
                "python/carnot/reporting/board_reader_qualification_8440.py",
                "python/carnot/reporting/primary_publication.py",
                "scripts/adversarial_verify.py",
                "scripts/verdict_row_consistency_lint.py",
                "scripts/experiment_template.py",
                "ops/exclusion_manifest.yaml",
            ]
        ],
    )
    work["phase_spans"] = [
        dict(
            name="sealed_evidence_reduction", duration_s=(work["ended_monotonic_ns"] - start) / 1e9
        )
    ]
    atomic_json(raw / "measurement.json", work)
    progress("measurement_complete", 3, 0)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Continuity readiness records evidence accounting and grants no acceleration."""
    history, cost = work["reader"], work["cost"]
    owned = (
        bool(receipts)
        and all(r["passed"] for r in receipts)
        and all(r["passed"] for r in work["checks"])
        and not work.get("adversarial_findings")
    )
    ready = owned and history["qualified"] and work["contract"]["activated"]
    checks = [
        *work["checks"],
        *work["contract"]["gate_check_summary"],
        *history["checks"],
        *cost["checks"],
    ]
    checks += [
        failure(
            Path(work["root"]) / READER,
            "installed_compatible_spline_journal_operations",
            True,
            False,
        ),
        failure(
            Path(work["root"]) / READER, "complete_transfer_and_device_clocks", "measured", None
        ),
    ]
    checks = [
        dict(g, field=g["artifact_field"], operator=g["op"], check=g["artifact_field"])
        for g in checks
    ]
    rows = [
        dict(
            source_id=unit,
            arm="evidence_only_continuity",
            intended=1,
            completed=bool(qualified),
            failed=False,
            censored=not qualified,
            excluded=False,
            independent=0,
            numerator=int(qualified),
            denominator=1,
            missing_reason=None if qualified else missing,
        )
        for unit, qualified, missing in [
            ("KV260", history["qualified"], "board_reader_ready_score"),
            ("PolarFire", history["qualified"], "board_reader_ready_score"),
            ("Exp8451", cost["qualified"], cost["missing_field"]),
        ]
    ]
    klass = "blocked" if owned else "disqualified"
    value: Json = dict(
        experiment_id=8454,
        task_id=TASK,
        milestone=MILESTONE,
        run_date="20261012",
        honest_verdict=f"complete_{klass}_v727_board_operation_boundary",
        verdict_class=klass,
        gate_check_summary=checks,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS, current_llm_invocations=0),
        historical_model_provenance=history["models"],
        rows=rows,
        intended_count=len(rows),
        completed_count=sum(r["completed"] for r in rows),
        failed_count=0,
        censored_count=sum(r["censored"] for r in rows),
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            historical_boards=2, optional_cost_branches=1, independent_questions=0
        ),
        verifier_is_oracle=False,
        exposure_scope="historical_public_labels_retrospective_overlap_unknown; engineering_continuity_only",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=bool(owned),
        flagged_adversarial=not bool(owned),
        acceptance_gates=dict(
            owned_validation=bool(owned),
            original_reader=history["qualified"],
            current_contract=work["contract"]["activated"],
            compatible_operations=False,
            complete_transfer_device_clocks=False,
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(
            output.parent / "raw" / NAME / "terminal_validation.json"
        ),
        adversarial_findings=work.get("adversarial_findings", []),
        preconditions_checked=work["preconditions"],
        duration_s=(work["ended_monotonic_ns"] - work["started_monotonic_ns"]) / 1e9,
        phase_spans=work["phase_spans"],
        random_seed=7278454,
        source_artifact_hashes=work["refs"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=[reference(raw / "measurement.json"), *work.get("coverage_shards", [])],
        work_reference=reference(raw / "measurement.json"),
        publication_output=str(output),
        invocation_argv=work["invocation_argv"],
        cited_upstream_artifacts=[
            dict(
                producer=8440,
                producer_hash=READER_PIN,
                imported_fields=[
                    "board_obligations",
                    "polarfire_graduation",
                    "historical_model_provenance",
                    "optional_current_cost_evidence",
                ],
            )
        ],
        board_boundary_recorded_score=int(ready),
        board_reader_ready_score=int(owned and history["qualified"]),
        current_contract_ready_score=int(owned and work["contract"]["activated"]),
        board_operation_rows=operation_mapping(cost["rows"], cost["missing_field"]),
        kv260_supported_operations=[
            dict(operation="quadratic_Ising_energy", k_max=5, evidence_sha256=READER_PIN)
        ]
        if history["qualified"]
        else [],
        polarfire_graduation=history["polarfire"],
        unsupported_operations=OPERATIONS,
        next_device_evidence=[
            "exclusive_board_lease",
            "installed_compatible_kernel_and_firmware_hash",
            "matched_input_output_parity",
            "complete_request_response_bytes_transfer_and_device_clocks",
            "equivalent_durable_transaction_fsync_and_checkpoint_clocks",
        ],
        device_commands_executed=[],
        newly_owned_coverage=work.get("owned_coverage"),
        optional_cost_evidence=dict(
            qualified=cost["qualified"],
            observed=cost["observed"],
            missing_field=cost["missing_field"],
        ),
        board_terminal_states=[
            dict(
                board="KV260",
                state="historical_quadratic_Ising_k_max_le_5"
                if history["qualified"]
                else "blocked_reader",
                current_service_state="unsupported_spline_journal",
                full_service_acceleration=False,
            ),
            dict(
                board="PolarFire",
                state="historical_board_local_Linux_CPU_dispatch"
                if history["qualified"]
                else "blocked_reader",
                current_service_state="unmeasured_journal_CPU_dispatch",
                full_service_acceleration=False,
            ),
        ],
        no_purchase_implication="Retain existing CPU/storage; historical persistence bottlenecks provide no evidence for purchasing arithmetic accelerators.",
        historical_host_cost_evidence=history.get("host_bottleneck", {}),
        closed_historical_findings=dict(H1=-0.00390625, H2=0),
        methodology_note="Reuse the qualified original reader with sealed authority. Journal serialization, append, fsync, checkpoint, spline arithmetic and transport require installed compatible operations and full clocks. Public labels remain retrospective; independent factuality, current Qwen quality and hidden-game transfer remain open. Zero model calls and zero device commands.",
    )
    value["field_principles"] = {
        k: "Bind exact invocation and immutable producer evidence; missing clocks remain null and continuity grants no scientific or acceleration benefit."
        for k in [*value, "field_principles", "reproducibility_checksum"]
    }
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Rebuild summaries from original operands, defeating self-consistent rehashing."""
    try:
        value = json.loads(path.read_bytes())
        work = json.loads(base.checked(value["work_reference"]).read_bytes())
        for ref in [
            *value["source_artifact_hashes"],
            *value["code_config_hashes"],
            *value["raw_shard_hashes"],
        ]:
            base.checked(ref)
        for receipt in value["validation_receipts"]:
            for stream in ["stdout", "stderr"]:
                if stream + "_path" in receipt:
                    base.checked(
                        dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
                    )
        with TemporaryDirectory(prefix="exp8454-replay-", dir="/var/tmp") as folder:
            private = Path(folder)
            with historical.old.prior.prior.source_aliases(work["refs"], work["absent"]):
                current = contract(Path(work["root"]), private / "authority")
                history = reader(Path(work["root"]), private / "reader")
                cost = costs(Path(work["root"]), private / "cost")
            if (
                current != work["contract"]
                or any(
                    history[k] != work["reader"][k]
                    for k in [
                        "qualified",
                        "kv260",
                        "polarfire",
                        "models",
                        "checks",
                        "host_bottleneck",
                    ]
                )
                or any(
                    cost[k] != work["cost"][k]
                    for k in ["qualified", "rows", "checks", "observed", "missing_field"]
                )
            ):
                return False
        return bool(
            value
            == build(
                work,
                value["validation_receipts"],
                Path(value["work_reference"]["path"]).parent,
                Path(value["publication_output"]),
            )
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False

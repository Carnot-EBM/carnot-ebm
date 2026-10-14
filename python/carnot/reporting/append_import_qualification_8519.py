"""REQ-REPORT-8519: import readiness cannot qualify historical speed measurements."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import append_service_cost_8494 as reader
from carnot.reporting import append_store_qualification_8493 as store_reader
from carnot.reporting import v732_contract_methods as methods
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.verify import append_service_cost_8494 as kernel
from carnot.verify.append_store_8493 import Store

Json = dict[str, Any]
ROOT = reader.ROOT
NAME = "experiment_8519_v732_append_import_qualification"
TASK = "exp8519-append-import-qualification"
CLI, TEST = (
    f"scripts/experiments/{NAME}.py",
    "tests/python/test_append_import_qualification_8519.py",
)
OWNED = [
    "python/carnot/reporting/append_import_history_8519.py",
    "python/carnot/reporting/append_import_qualification_8519.py",
    "python/carnot/reporting/append_import_runner_8519.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
REQUIRED = {
    "owned_tests",
    "coverage_report",
    "repair_line_coverage",
    "ruff_check",
    "ruff_format",
    "strict_mypy",
    "spec_coverage",
    "historical_union",
    "private_cli",
    "missing",
    "error",
    "tamper",
}
PINS = {
    8493: reader.UPSTREAM_PIN,
    8494: "sha256:e1584c837751142964bb9ce59ba1bfa887df1c6ac331f2c658f2e70b482ab742",
}
HISTORICAL = ROOT / "results/raw" / NAME / "historical"
reference, read = reader.reference, reader.read


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """True counts keep CPU readers and their children visible to the supervisor."""
    print(f"[exp8519] phase={phase} completed={completed} pending={pending}", flush=True)


def checksum(value: Json) -> str:
    """A stable digest detects changed summaries before independent reconstruction."""
    return str(canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"}))


def gate(path: Path, field: str, expected: Any, observed: Any, upstream: str = TASK) -> Json:
    """Missing inputs retain None so an absent measurement cannot become zero."""
    return dict(
        check=field,
        upstream=upstream,
        path=str(path),
        sha256=sha256_file(path) if path.is_file() else None,
        field=field,
        operator="==",
        expected=expected,
        observed=observed,
        missing=not path.is_file(),
        passed=False,
    )


def preflight(root: Path, raw: Path) -> Json:
    """Check tools, resources and exact activated task before touching historical operands."""
    progress("preconditions")
    raw.mkdir(parents=True, exist_ok=True)
    tools = {
        n: shutil.which(str(ROOT / ".venv/bin" / n))
        for n in ("python", "pytest", "coverage", "ruff", "mypy")
    }
    memory = int(Path("/proc/meminfo").read_text().split("MemAvailable:")[1].split()[0]) * 1024
    with TemporaryDirectory(dir="/var/tmp", prefix="exp8519-preflight-") as d:
        scratch = Path(d)
        probe = scratch / "readback"
        probe.write_bytes(b"private_disk_probe")
        conditions = dict(
            tools=tools,
            memory_available_bytes=memory,
            disk_free_bytes=shutil.disk_usage(scratch).free,
            scratch_filesystem=store_reader.upstream.filesystem(scratch),
            scratch_readback=probe.read_bytes() == b"private_disk_probe",
            measurement_cap_s=3600,
            validation_reserve_s=900,
            task_cap_s=4800,
            optional_cuda_and_boards="not_required",
        )
    failures = []
    if (
        not all(tools.values())
        or memory < 256 * 1024**2
        or conditions["disk_free_bytes"] < 256 * 1024**2
        or not conditions["scratch_filesystem"]["disk_backed"]
    ):
        failures.append(gate(raw, "tools_memory_disk", True, conditions))
    refs: list[Json] = []
    try:
        authority = methods.authority(root, raw / "authority")
        if not authority["activated"]:
            failures.extend(authority["gate_check_summary"])
        task = next(t for t in authority["tasks"] if t["id"] == TASK)
        if task["deliverable"] != "results/" + NAME + ".json":
            failures.append(
                gate(
                    root / methods.ACTIVE,
                    "deliverable",
                    "results/" + NAME + ".json",
                    task["deliverable"],
                )
            )
        upstream = root / "results/experiment_8512_v732_contract_methods.json"
        upstream_pin = "sha256:d8aad6fcd8662fe8abc98acaa95add49df44b3b5359700ab09bbaafee3114eec"
        value = read(dict(path=str(upstream), sha256=upstream_pin))
        refs.append(reference(upstream))
        side = read(reference(Path(value["terminal_validation_sidecar_path"])))
        bound = read_bound_sidecar(upstream, Path(side["publication"]["sidecar_path"]))
        if (
            not bound["report"]["passed"]
            or not value["required_checks_passed"]
            or value["flagged_adversarial"]
        ):
            failures.append(gate(upstream, "qualified_methods_consumer", True, False, methods.TASK))
        for field in ("methods_ready_score", "service_inputs_ready_score"):
            if value.get(field) != 1:
                failures.append(gate(upstream, field, 1, value.get(field), methods.TASK))
        refs += [reference(Path(methods.protocol()["v732_activation_reference"]["path"]))]
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        failed_path = root / methods.ACTIVE
        failures.append(
            gate(
                failed_path,
                "activated_task_inputs",
                True,
                str(error) if failed_path.is_file() else None,
            )
        )
    for number, name, producer in [
        (8493, store_reader.NAME, store_reader.TASK),
        (8494, reader.NAME, reader.TASK),
    ]:
        path = root / "results" / (name + ".json")
        observed = sha256_file(path) if path.is_file() else None
        if observed != PINS[number]:
            failures.append(gate(path, "sha256", PINS[number], observed, producer))
        else:
            refs.append(reference(path))
    return dict(
        root=str(root),
        raw=str(raw),
        start_ns=time.monotonic_ns(),
        preconditions_checked=conditions,
        gate_check_summary=failures,
        source_artifact_hashes=refs,
        history={},
        phase_spans=[],
    )


def history(root: Path, raw: Path) -> Json:
    """Cold-reduce sealed crash and clock operands without running any benchmark."""
    raw.mkdir(parents=True, exist_ok=True)
    rows: list[Json] = []
    refs: list[Json] = []
    provenance: list[Json] = []
    protocol = methods.protocol()
    for ref in protocol["v730_authority_references"]:
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("historical_authority_hash")
        refs.append(reference(Path(ref["path"])))
    tapes = [
        read(ref)
        for ref in read(reference(ROOT / store_reader.UPSTREAM))["transition_fixture"][
            "tape_references"
        ]
    ]
    for number in (8493, 8494):
        path = next((root / "results").glob(f"experiment_{number}_*.json"))
        value = read(dict(path=str(path), sha256=PINS[number]))
        if value["run_date"] != "20261013":
            raise ValueError("historical_date")
        refs.append(reference(path))
        terminal = read(reference(Path(value["terminal_validation_sidecar_path"])))
        side = read_bound_sidecar(path, Path(terminal["publication"]["sidecar_path"]))
        if not side["report"]["passed"]:
            raise ValueError("historical_publication")
        for ref in value["code_config_hashes"] + value["source_artifact_hashes"]:
            original = Path(ref["path"])
            if original == ROOT / "python/carnot/reporting/append_service_cost_8494.py":
                original = HISTORICAL / "append_service_cost_8494.py"
            if sha256_file(original) != ref["sha256"]:
                raise ValueError("historical_source_hash")
        work = read(value["work_reference"])
        if (
            store_reader.upstream.methods.tasks_digest(work["authority"]["tasks"])
            != store_reader.upstream.methods.TASKS_PIN
        ):
            raise ValueError("historical_task_contract")
        for row in work["rows"]:
            progress("historical_primitive", len(rows), 40 - len(rows))
            primitive = read(row["primitive_reference"])
            frozen = read(primitive["frozen_reference"])
            if frozen not in tapes:
                raise ValueError("historical_tape")
            for ref in primitive["store_references"]:
                if sha256_file(Path(ref["path"])) != ref["sha256"]:
                    raise ValueError("historical_store_bytes")
            if number == 8493:
                if not store_reader.reduce(primitive):
                    raise ValueError("historical_store_replay")
                metrics: Json = dict(
                    actual_exit=primitive["actual_exit"],
                    recovered_state_hash=primitive["recovered_state_hash"],
                )
            else:
                metrics = kernel.reduce(frozen, primitive)
                with TemporaryDirectory(dir="/var/tmp", prefix="exp8519-store-") as d:
                    target = Path(d) / "store"
                    shutil.copytree(Path(primitive["directory"]) / "store", target)
                    store = (kernel.base.Snapshot if row["arm"] == "snapshot" else Store)(
                        target, frozen
                    )
                    if kernel.base.encode(store.read()) != kernel.base.encode(primitive["final"]):
                        raise ValueError("historical_disk_replay")
            rows.append(
                dict(
                    unit_id=f"{number}:" + row["unit_id"],
                    status="completed",
                    passed=True,
                    absolute_metric=1,
                    metrics=metrics,
                    primitive_reference=row["primitive_reference"],
                    missing_reason=None,
                )
            )
        refs.append(value["work_reference"])
        provenance.extend(value["historical_model_provenance"])
    return dict(
        rows=rows,
        references=refs,
        historical_model_provenance=provenance,
        historical_clock_eligibility="historical_only_disqualified_8494",
    )


def build(work: Json, receipts: list[Json], output: Path) -> Json:
    """Engineering readiness needs every required check and never changes historical eligibility."""
    raw = Path(work["raw"])
    atomic_json(raw / "work.json", work)
    history_result = work["history"]
    passed = REQUIRED <= {r["name"] for r in receipts if r["passed"]}
    failures = (
        any(not r["passed"] for r in receipts)
        or bool(work.get("owned_failure"))
        or bool(work.get("adversarial_findings"))
    )
    verdict = (
        "disqualified"
        if failures
        else "blocked"
        if work["gate_check_summary"]
        else "circular_positive"
        if passed and len(history_result.get("rows", [])) == 40
        else "disqualified"
    )
    ready = int(verdict == "circular_positive")
    rows = history_result.get(
        "rows",
        [
            dict(
                unit_id=f"unavailable:{i}",
                status="censored",
                passed=None,
                absolute_metric=None,
                missing_reason="required_inputs_missing",
            )
            for i in range(40)
        ],
    )
    code = [reference(ROOT / p) for p in [*OWNED, TEST]]
    value: Json = dict(
        schema="carnot.append_import_qualification.v1",
        experiment_id=8519,
        task_id=TASK,
        milestone="2026.10.732",
        run_date="20261014",
        output_path=str(output),
        honest_verdict="complete_" + verdict + "_append_import_qualification",
        verdict_class=verdict,
        append_runner_ready_score=ready,
        required_checks_passed=passed and not failures,
        gate_check_summary=work["gate_check_summary"],
        flagged_adversarial=bool(work.get("adversarial_findings")),
        adversarial_findings=work.get("adversarial_findings", []),
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        historical_model_provenance=history_result.get("historical_model_provenance", []),
        rows=rows,
        intended_count=40,
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=sum(r["status"] == "failed" for r in rows),
        censored_count=sum(r["status"] == "censored" for r in rows),
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            historical_store_rows=16, historical_clock_rows=24, independent_science_questions=0
        ),
        verifier_is_oracle=True,
        exposure_scope="sealed historical engineering fixtures",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        acceptance_gates=dict(
            required_checks=passed,
            historical_replay=bool(history_result),
            no_benchmark=True,
            no_current_llm_calls=True,
            owned_coverage="coverage_report" in {r["name"] for r in receipts if r["passed"]},
        ),
        validation_receipts=receipts,
        owned_coverage=work.get("owned_coverage", {}),
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        preconditions_checked=work["preconditions_checked"],
        duration_s=work.get("duration_s", 0.001),
        phase_spans=work["phase_spans"],
        random_seed=7328519,
        source_artifact_hashes=work["source_artifact_hashes"]
        + history_result.get("references", []),
        code_config_hashes=code,
        raw_shard_hashes=[reference(raw / "work.json"), *work.get("validation_references", [])],
        work_reference=reference(raw / "work.json"),
        typing_failure_reference=reference(HISTORICAL / "strict_mypy_before.receipt.json"),
        import_repair_diff_reference=reference(HISTORICAL / "import_repair.diff"),
        historical_store_receipt=dict(
            original_date="20261013",
            original_sha256=PINS[8493],
            replayed_rows=16 if history_result else 0,
        ),
        measurement_kernel_hash=reference(
            ROOT / "python/carnot/verify/append_service_cost_8494.py"
        ),
        replay_reader_hash=reference(ROOT / "python/carnot/reporting/append_service_cost_8494.py"),
        validation_repair_hash=canonical_hash(code),
        historical_clock_eligibility="historical_only_disqualified_8494",
        durability_scope="unchanged Linux process-exit fixtures; no physical power-loss certification",
        methodology_note="Cold replay authenticates original crash/equivalence and complete-clock operands. Explicit import repair only; zero new benchmark measurements and zero model calls. Historical Exp8494 remains disqualified.",
    )
    value["cited_upstream_artifacts"] = [
        dict(r, fields_imported="original authority and historical primitive replay only")
        for r in value["source_artifact_hashes"]
    ]
    value["field_principles"] = {
        k: "Bind current qualification to sealed evidence; import readiness is not speed or scientific benefit."
        for k in value
    }
    value["field_principles"]["append_runner_ready_score"] = (
        "Require exact current strict typing, owned coverage, private standalone replay and unchanged consumers."
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Independent pinned primitives reject even a self-consistently rehashed summary."""
    try:
        value = json.loads(path.read_bytes())
        if value["experiment_id"] != 8519 or value["reproducibility_checksum"] != checksum(value):
            return False
        for ref in [
            *value["source_artifact_hashes"],
            *value["code_config_hashes"],
            *value["raw_shard_hashes"],
            value["measurement_kernel_hash"],
            value["replay_reader_hash"],
            value["typing_failure_reference"],
            value["import_repair_diff_reference"],
        ]:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        work = read(value["work_reference"])
        if work["history"]:
            with TemporaryDirectory(dir="/var/tmp", prefix="exp8519-cold-") as d:
                if history(Path(work["root"]), Path(d)) != work["history"]:
                    return False
        return bool(value == build(work, value["validation_receipts"], Path(value["output_path"])))
    except (OSError, ValueError, KeyError, TypeError, IndexError, StopIteration):
        return False

"""REQ-REPORT-8436: independently pinned operands distinguish attribution from speed.

The producer's primary binds the original clock bytes. A candidate cannot qualify
by changing both its timing summaries and their newly computed hashes.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import time
from typing import Any

import yaml

from carnot.reporting import v726_contract_methods as authority
from carnot.reporting import native_parity_8405 as native
from carnot.reporting import v718_replay_history as consumer
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    sha256_file,
    ZERO_INVOCATION_COUNTS,
)
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.reporting.v709_execution import child
from carnot.verify import python_transaction_cost_8407 as source
from carnot.verify import transaction_attribution_8436 as k

Json = dict[str, Any]
ROOT = authority.ROOT
NAME, TASK = "experiment_8436_v726_transaction_attribution", "exp8436-transaction-attribution"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_transaction_attribution_8436.py"
OWNED = [
    "python/carnot/verify/transaction_attribution_8436.py",
    "python/carnot/reporting/transaction_attribution_8436.py",
    "python/carnot/reporting/transaction_attribution_runner_8436.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
PINS = {
    "results/experiment_8407_v724_python_transaction_cost.json": "sha256:033cdbbe98be7790889354e50646f49c24a9e270cbb5c3c83b8d708c540c293f",
    "results/experiment_8405_v724_native_parity_qualification.json": "sha256:b94b031f975ec6415e466fdf4c88df73180ac6b3e3afb7016fa2f47c75515270",
}
TERMINAL_PIN = "sha256:9deaa6be7f82dd1046823029ed324f6974fe700c50498dc0dab5925d87af1ae2"
SIDECAR_PIN = "sha256:0b5ed1674eb9a27d9f5cc7964a80f76647300604fab8392faae6926b0cd8c2c4"
GUARDS = [
    "python/carnot/reporting/v718_replay_history.py",
    "python/carnot/reporting/primary_publication.py",
    "scripts/adversarial_verify.py",
    "scripts/verdict_row_consistency_lint.py",
]
REQUIRED = {
    "owned_tests",
    "private_E2E018_consumers",
    "private_E2E021",
    "coverage_combine",
    "coverage_report",
    "coverage_json",
    "ruff_check",
    "ruff_format",
    "strict_mypy",
    "spec_coverage",
}
reference = authority.reference


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real work counts so an audit remains observable without model calls."""
    print(f"[exp8436] phase={phase} completed={completed} pending={pending}", flush=True)


def checksum(value: Json) -> str:
    """Exclude only the checksum itself so every declared field stays bound."""
    return str(
        canonical_hash(
            {key: item for key, item in value.items() if key != "reproducibility_checksum"}
        )
    )


def read(ref: Json) -> Json:
    """Reject altered bytes before decoding, even when the JSON still looks valid."""
    path = Path(ref["path"])
    if sha256_file(path) != ref["sha256"]:
        raise ValueError("input_hash:" + str(path))
    return dict(json.loads(path.read_bytes()))


def gate(path: Path, field: str, expected: Any, observed: Any) -> Json:
    """Missing operands stay null, distinct from a measured zero clock or error."""
    return dict(
        check=field,
        upstream=TASK,
        path=str(path),
        sha256=sha256_file(path) if path.is_file() else None,
        field=field,
        operator="==",
        expected=expected,
        observed=observed,
        passed=False,
    )


def reduce(root: Path, refs: list[Json]) -> tuple[list[Json], list[Json], Json]:
    """Match all intended slots to producer-pinned repeats; invalid rows stay visible."""
    primary = read(dict(path=str(root / next(iter(PINS))), sha256=next(iter(PINS.values()))))
    work = read(primary["work_reference"])
    originals = work["results"]
    if len(originals) != 162 or len(refs) != 162:
        raise ValueError("repeat_roster")
    rows: list[Json] = []
    warmups: list[Json] = []
    for index, (ref, original) in enumerate(zip(refs, originals, strict=True)):
        if index % 18 == 0:
            progress("primitive_reduction", index, 162 - index)
        cell, repeat = source.CELLS[index // 6], index % 6
        slot: Json = dict(
            cell=cell,
            repeat=repeat,
            repeat_id=f"{cell['workload']}-b{cell['batch']}-r{cell['predictions_per_update']}-rep{repeat}",
        )
        try:
            if ref["sha256"] != original["sha256"] or ref["warmup"] != (repeat == 0):
                raise ValueError("producer_clock_binding")
            operand = read(ref)
            if operand["cell"] != cell or operand["repeat"] != repeat:
                raise ValueError("slot_binding")
            tape = source.cost_tape(cell["predictions_per_update"])
            if len(operand["rows"]) != len(tape) or any(
                {key: row[key] for key in event} != event
                for row, event in zip(operand["rows"], tape, strict=True)
            ):
                raise ValueError("original_work_counts")
            slot.update(k.attribute(operand), primitive_reference=ref)
        except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
            slot.update(
                status="failed",
                missing_reason=str(error),
                complete_wall_s=None,
                kernel_s=None,
                conversion_s=None,
                persistence_stage_s=None,
                unattributed_s=None,
                interval_checks=dict(valid=False),
            )
        (warmups if repeat == 0 else rows).append(slot)
    return rows, warmups, primary


def native_diagnosis(root: Path, raw: Path) -> Json:
    """Reproduce the metric-name rejection using paired proof and actual private CLI."""
    path, pin = list(PINS.items())[1]
    value = read(dict(path=str(root / path), sha256=pin))
    proof = native.coefficient_reduction(value["rows"])
    candidate = raw / "native_private.json"
    candidate.write_bytes((root / path).read_bytes())
    receipt = child(
        "native_adversarial",
        [
            str(ROOT / ".venv/bin/python"),
            "-u",
            "scripts/adversarial_verify.py",
            "--json",
            str(candidate),
        ],
        raw / "native",
        expected=1,
        deadline=60,
    )
    report = json.loads(Path(receipt["stdout_path"]).read_bytes())
    report.update(candidate_sha256=sha256_file(candidate), verifier_sha256=consumer.verifier_hash())
    disposition = consumer.consume(report, candidate, receipt["exit_code"], proof)
    atomic_json(raw / "native_report.json", report)
    return dict(
        reproduced=receipt["passed"]
        and not disposition["passed"]
        and disposition["findings"] == value["adversarial_findings"]
        and proof == value["coefficient_reduction"],
        consumer=disposition,
        proof=proof,
        receipt=receipt,
        candidate=reference(candidate),
        report=reference(raw / "native_report.json"),
        supplied_metric="max_coefficient_error",
        supported_metric="dense_sparse_error_max",
        changed_diagnosis="Paired coefficient proof succeeds; unchanged consumer rejects the metric name. The previous native measurement retry is dropped.",
        native_measurement_retried=False,
    )


def measure(root: Path, raw: Path, private: Path) -> Json:
    """Check authority, disk and producer custody before reading historical clocks."""
    progress("preconditions_before")
    began = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    probe = private / "disk_probe"
    probe.write_bytes(b"private disk scratch")
    mounts = [line.split(" - ") for line in Path("/proc/self/mountinfo").read_text().splitlines()]
    mounted = max(
        (line for line in mounts if private.resolve().is_relative_to(Path(line[0].split()[4]))),
        key=lambda line: len(line[0].split()[4]),
    )
    fs = mounted[1].split()[0]
    memory = int(Path("/proc/meminfo").read_text().split("MemAvailable:")[1].split()[0]) * 1024
    resources: Json = dict(
        disk_backed=fs not in ("tmpfs", "ramfs"),
        filesystem_type=fs,
        available_memory_bytes=memory,
        memory_available=memory > 256_000_000,
        disk_free=shutil.disk_usage(private).free > 256_000_000,
        scratch_readback=probe.read_bytes() == b"private disk scratch",
        task_cap_s=4800,
        validation_reserve_s=900,
        scratch_path=str(private),
        tools={
            name: (ROOT / ".venv/bin" / name).is_file()
            for name in ("python", "pytest", "coverage", "ruff", "mypy")
        },
    )
    active = authority.authority(root, raw / "authority")
    gates = (
        [
            gate(
                root / authority.ACTIVE,
                "activated_complete_V726_authority",
                True,
                active["activated"],
            )
        ]
        if not active["activated"]
        else []
    )
    for name in ("disk_backed", "memory_available", "disk_free", "scratch_readback"):
        if not resources[name]:
            gates.append(gate(private, name, True, resources[name]))
    for name, available in resources["tools"].items():
        if not available:
            gates.append(gate(ROOT / ".venv/bin" / name, "tool_available", True, None))
    refs, operands, diagnosis = [], [], {}
    for path, pin in PINS.items():
        observed = sha256_file(root / path) if (root / path).is_file() else None
        if observed != pin:
            gates.append(gate(root / path, "pinned_primary_sha256", pin, observed))
        else:
            refs.append(reference(root / path))
    progress("preconditions_after")
    if (root / next(iter(PINS))).is_file() and sha256_file(root / next(iter(PINS))) == next(
        iter(PINS.values())
    ):
        try:
            primary = read(refs[0])
            terminal = Path(primary["terminal_validation_sidecar_path"])
            if sha256_file(terminal) != TERMINAL_PIN:
                raise ValueError("terminal_receipt_hash")
            receipt = json.loads(terminal.read_bytes())
            if sha256_file(Path(receipt["sidecar_path"])) != SIDECAR_PIN:
                raise ValueError("terminal_sidecar_hash")
            sidecar = read_bound_sidecar(root / next(iter(PINS)), Path(receipt["sidecar_path"]))
            if (
                not sidecar["report"]["passed"]
                or receipt["primary_sha256"] != refs[0]["sha256"]
                or primary["python_cost_ready_score"] != 1
            ):
                raise ValueError("unqualified_producer")
            work = read(primary["work_reference"])
            refs.extend(
                [
                    reference(terminal),
                    reference(Path(receipt["sidecar_path"])),
                    primary["work_reference"],
                ]
            )
            for ref in [*work["refs"], *work["code_config_hashes"]]:
                read_bytes = Path(ref["path"]).read_bytes()
                if sha256_file(Path(ref["path"])) != ref["sha256"]:
                    raise ValueError("historical_source_or_code_hash")
                refs.append(ref)
                # Read exact historical authority bytes, never require its producer to rerun.
                if not read_bytes:
                    raise ValueError("empty_historical_operand")
            for index, ref in enumerate(work["results"]):
                if index % 18 == 0:
                    progress("snapshot_primitive_operands", index, 162 - index)
                primitive_path = Path(ref["path"])
                if sha256_file(primitive_path) != ref["sha256"]:
                    raise ValueError("primitive_hash")
                destination = raw / "operands" / f"{index:03}.json"
                destination.parent.mkdir(exist_ok=True)
                shutil.copyfile(primitive_path, destination)
                operands.append(
                    dict(
                        reference(destination),
                        warmup=ref["warmup"],
                        source_path=str(primitive_path),
                    )
                )
        except (OSError, ValueError, KeyError, TypeError) as error:
            gates.append(
                gate(
                    root / next(iter(PINS)), "authenticated_terminal_and_operands", True, str(error)
                )
            )
    rows: list[Json] = []
    warmups: list[Json] = []
    primary = {}
    if len(operands) == 162:
        rows, warmups, primary = reduce(root, operands)
    else:
        rows = [
            dict(
                cell=cell,
                repeat=repeat,
                repeat_id=f"{cell['workload']}-b{cell['batch']}-r{cell['predictions_per_update']}-rep{repeat}",
                status="censored",
                missing_reason="required_primitive_operands_unavailable",
                complete_wall_s=None,
                kernel_s=None,
                conversion_s=None,
                persistence_stage_s=None,
                unattributed_s=None,
                interval_checks=dict(valid=False),
            )
            for cell in source.CELLS
            for repeat in range(1, 6)
        ]
    if len(refs) >= 2 and refs[1]["sha256"] == list(PINS.values())[1]:
        diagnosis = native_diagnosis(root, raw)
    atomic_json(raw / "attribution_rows.json", dict(rows=rows, warmups=warmups))
    work = dict(
        root=str(root),
        raw=str(raw),
        rows=rows,
        warmups=warmups,
        operands=operands,
        refs=refs,
        gate_check_summary=gates,
        authority=active,
        preconditions_checked=resources,
        native_consumer_failure=diagnosis,
        upstream_work_counts={
            key: primary.get(key)
            for key in (
                "operation_counts",
                "feedback_conservation",
                "intended_count",
                "completed_count",
                "failed_count",
                "censored_count",
                "excluded_count",
                "independent_count",
            )
        },
        historical_model_provenance=primary.get("historical_model_provenance", []),
        unchanged_guard_hashes=[reference(ROOT / path) for path in GUARDS],
        code_config_hashes=[reference(ROOT / path) for path in [*OWNED, TEST]],
        started_monotonic_ns=began,
        duration_s=(time.monotonic_ns() - began) / 1e9,
        phase_spans=[
            dict(phase="authenticate_and_attribute", start_ns=began, end_ns=time.monotonic_ns())
        ],
    )
    atomic_json(raw / "measurement.json", work)
    progress(
        "attribution_complete",
        sum(row["status"] == "completed" for row in rows),
        sum(row["status"] != "completed" for row in rows),
    )
    return work


def build(work: Json, receipts: list[Json], output: Path) -> Json:
    """Readiness requires authentic complete operands and all owned checks, not gain."""
    rows = work["rows"]
    complete = len(rows) == 135 and all(row["status"] == "completed" for row in rows)
    passed = bool(receipts) and all(row["passed"] for row in receipts)
    required = REQUIRED <= {row.get("name") for row in receipts if row["passed"]}
    native_ready = work["native_consumer_failure"].get("reproduced", False)
    verdict = (
        "disqualified"
        if not passed
        else "blocked"
        if work["gate_check_summary"]
        else "null"
        if complete and passed and required and native_ready
        else "disqualified"
    )
    valid = [row for row in rows if row["status"] == "completed"]
    totals = {
        key: sum(row[key] for row in valid) if valid else None
        for key in (
            "complete_wall_s",
            "kernel_s",
            "conversion_s",
            "persistence_stage_s",
            "unattributed_s",
        )
    }
    value: Json = dict(
        experiment_id=8436,
        task_id=TASK,
        milestone="2026.10.726",
        run_date="20261012",
        honest_verdict="complete_" + verdict + "_transaction_attribution",
        verdict_class=verdict,
        gate_check_summary=work["gate_check_summary"],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS, current_llm_invocations=0),
        historical_model_provenance=work["historical_model_provenance"],
        rows=rows,
        intended_count=135,
        completed_count=len(valid),
        failed_count=sum(row["status"] == "failed" for row in rows),
        censored_count=sum(row["status"] == "censored" for row in rows),
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            cells=27,
            measured_repeats_per_cell=5,
            warmups_per_cell=1,
            repeats_add_independent_questions=0,
        ),
        verifier_is_oracle=False,
        exposure_scope="retrospective_exposed_cached_cost_operands",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=passed and (required or verdict == "blocked"),
        flagged_adversarial=not passed or verdict == "disqualified",
        acceptance_gates=dict(
            authentic_inputs=not work["gate_check_summary"],
            complete_135_clocks=complete,
            owned_validation=passed,
            scoped_validation=required,
            native_rejection_reproduced=native_ready,
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal_validation.json"),
        adversarial_findings=work.get("adversarial_findings", []),
        preconditions_checked=work["preconditions_checked"],
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=7268436,
        source_artifact_hashes=work["refs"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=work["operands"] + work.get("coverage_shards", []),
        cited_upstream_artifacts=[
            dict(
                ref,
                imported_fields="historical authority, terminal custody, primitive timing and original work counts",
            )
            for ref in work["refs"]
        ],
        clock_attribution_ready_score=int(verdict == "null"),
        attribution_rows_path=str(Path(work["raw"]) / "attribution_rows.json"),
        attribution_rows_sha256=sha256_file(Path(work["raw"]) / "attribution_rows.json"),
        primary_cell=dict(workload="natural", batch=1, predictions_per_update=8),
        measured_repeat_count=len(valid),
        interval_checks=dict(
            complete_roster=complete,
            valid_rows=len(valid),
            disjoint_partition="initialization plus contained stages plus unallocated gaps; tail is a separate overlapping wall subinterval",
        ),
        native_parity_ready_score=0,
        native_consumer_failure=work["native_consumer_failure"],
        unchanged_guard_hashes=work["unchanged_guard_hashes"],
        upstream_work_counts=work["upstream_work_counts"],
        warmups=work["warmups"],
        work_reference=reference(Path(work["raw"]) / "measurement.json"),
        owned_coverage=work.get("owned_coverage", {}),
        closed_exposed_utility=dict(H1=-0.00390625, H2=0, reopened=False),
        pointer_publication_scope="Composite snapshot rename and directory fsync, then complete pointer JSON serialization, temporary write, file fsync and pointer rename. Not bare rename or disk-only timing.",
        methodology_note="Primitive retrospective attribution only, with complete wall denominators and unchanged native consumer rejection. No new timing, LLM acquisition, current-Qwen skill, factuality, hidden-game generalization or speed benefit.",
        **totals,
    )
    value["field_principles"] = {
        key: "Bind the field to exact historical or invocation evidence; missing differs from zero and qualification establishes no speed or generalization benefit."
        for key in value
    }
    value["field_principles"]["acceptance_gates"] = (
        "Authentic complete 135-repeat clocks and passing owned validation establish administrative attribution readiness, independent of measured gain. Native rejection is a required diagnostic, never native qualification."
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Cold readers reconstruct primitive timings against independently pinned sources."""
    try:
        value = json.loads(path.read_bytes())
        if value["reproducibility_checksum"] != checksum(value):
            return False
        work = read(value["work_reference"])
        if work["authority"]["activated"]:
            snapshot = work["authority"]["authority_snapshots"]["active"]
            active = yaml.safe_load(Path(snapshot["snapshot_path"]).read_bytes())
            if authority.tasks_digest(active["tasks"]) != authority.TASKS_PIN:
                return False
        for ref in [
            *value["source_artifact_hashes"],
            *value["code_config_hashes"],
            *value["unchanged_guard_hashes"],
            *value["raw_shard_hashes"],
        ]:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        if work["operands"]:
            rows, warmups, _ = reduce(Path(work["root"]), work["operands"])
            if rows != work["rows"] or warmups != work["warmups"]:
                return False
        timing = read(
            dict(path=value["attribution_rows_path"], sha256=value["attribution_rows_sha256"])
        )
        if timing != dict(rows=work["rows"], warmups=work["warmups"]):
            return False
        diagnosis = work["native_consumer_failure"]
        if diagnosis:
            candidate = Path(diagnosis["candidate"]["path"])
            if sha256_file(candidate) != list(PINS.values())[1]:
                return False
            proof = native.coefficient_reduction(read(diagnosis["candidate"])["rows"])
            disposition = consumer.consume(
                read(diagnosis["report"]), candidate, diagnosis["receipt"]["exit_code"], proof
            )
            expected_findings = read(diagnosis["candidate"])["adversarial_findings"]
            if (
                disposition != diagnosis["consumer"]
                or proof != diagnosis["proof"]
                or diagnosis["reproduced"]
                != (
                    diagnosis["receipt"]["passed"]
                    and not disposition["passed"]
                    and disposition["findings"] == expected_findings
                )
            ):
                return False
        return bool(value == build(work, value["validation_receipts"], path))
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return False

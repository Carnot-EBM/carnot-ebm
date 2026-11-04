"""REQ-VERIFY-8783: seal the shared descriptor before any durable child runs."""

from __future__ import annotations

from datetime import datetime, UTC
import json
import shutil
from pathlib import Path
import sys
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import durable_producer_identity_8783 as e
from carnot.reporting import ordinary_durable_runner_8772 as historical
from carnot.reporting import v752_branch_methods as contract
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
runtime = namespace(historical, "identity8783_runner")
runtime.e = e.measurement
runtime.E2E = ["tests/python/test_primary_publication_7928.py", *historical.PRIVATE_E2E]
base_measure, base_worker, base_replay = runtime.measure, runtime.worker, runtime.replay


def worker(request: Json, folder: Path) -> Json:
    """Reject foreign requests before invoking the unchanged durable store worker."""
    producer = request["producer_descriptor"]
    e.exact_identity(producer, producer)
    result: Json = base_worker(request, folder)
    result["producer_descriptor"] = producer
    atomic_json(folder / "result.json", result)
    return result


def consumer_controls(value: Json, raw: Path) -> list[Json]:
    """Use both shipped readers and exact field gates on freshly hashed private copies."""
    rows = []
    for case, task in [
        ("exact", e.TASK),
        ("same_number_alias", "exp8783-ordinary-update-cost"),
        ("historical_alias", "exp8772-ordinary-update-cost"),
    ]:
        folder = raw / case
        path = folder / (e.NAME + ".json")
        candidate = dict(value, task_id=task)
        atomic_json(path, candidate)
        ref = e.reference(path)
        atomic_json(
            folder / "raw" / "rehash.json",
            dict(candidate=ref, semantic_sha256=canonical_hash(candidate)),
        )
        try:
            e.exact_identity(
                candidate, value.get("producer_descriptor", e.descriptor(value["invocation_id"]))
            )
            authenticated = True
        except ValueError:
            authenticated = False
        identity = reader_receipt(e.TASK, folder, field="task_id", expected=e.TASK)
        readiness = reader_receipt(e.TASK, folder, field="durable_producer_ready_score")
        rows.append(
            dict(
                case=case,
                candidate_reference=ref,
                identity_reader=identity,
                readiness_reader=readiness,
                descriptor_authenticated=authenticated,
                passed=identity["passed"] == (case == "exact")
                and readiness["passed"]
                and authenticated == (case == "exact"),
            )
        )
    return rows


def measure(raw: Path, date: str, *, private: bool = False) -> Json:
    """Keep original clocks local and freeze descriptor, authorities and exact nodes first."""
    start, wall = time.monotonic_ns(), datetime.now(UTC).isoformat()
    e.progress("authenticate_exact_sources_before")
    admitted = e.inputs()
    producer = e.descriptor(raw.name)
    producer_path = raw.with_name(raw.name + "-producer_descriptor.json")
    atomic_json(producer_path, producer)
    frozen_contract = raw.with_name(raw.name + "-task-contract.md")
    if not admitted["failures"]:
        shutil.copyfile(e.ROOT / contract.DESIGN, frozen_contract)
        admitted["refs"] = [
            dict(ref, path=str(frozen_contract))
            if ref["path"] == str(e.ROOT / contract.DESIGN)
            else ref
            for ref in admitted["refs"]
        ]
    dependencies = [
        e.reference(e.ROOT / relative)
        for relative in [
            "python/carnot/reporting/ordinary_durable_cost_8772.py",
            "python/carnot/reporting/ordinary_durable_runner_8772.py",
            "python/carnot/reporting/lifecycle_archive_authority_8771.py",
            "python/carnot/reporting/v752_branch_methods.py",
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
            "scripts/conductor_gates.py",
            "scripts/in_process_doc_reconcile.py",
            "pyproject.toml",
            "tests/python/conftest.py",
        ]
    ]
    admitted["refs"] += dependencies
    old_refs = [
        e.reference(next((e.ROOT / "results").glob(f"experiment_{n}_*.json"))) for n in [8772, 8774]
    ]
    admitted["refs"] += old_refs
    old = e.read(old_refs[0])
    diagnostics = [
        dict(
            upstream="exp8772-ordinary-durable-cost",
            **old_refs[0],
            check="exact_historical_producer_identity",
            field="task_id",
            operator="==",
            expected="exp8772-ordinary-durable-cost",
            observed=old["task_id"],
            passed=False,
            operand_state="rejected_producer",
            admitted_current_hardware_evidence=False,
        )
    ]
    admitted["producer_descriptor"] = producer
    admitted["refs"].append(e.reference(producer_path))
    with patch.object(runtime.e, "inputs", lambda: admitted):
        work: Json = base_measure(raw, date)
    work.update(
        producer_descriptor=producer,
        dependency_code_refs=dependencies,
        historical_identity_diagnostics=diagnostics,
        producer_descriptor_reference=e.reference(producer_path),
        private_control=private,
        expected_trace=admitted.get("trace", [])[2032:2064],
        original_authority_rows=admitted.get("original_authority_rows", []),
        retained_controls=admitted.get("retained_controls", {}),
    )
    work["consumer_admission_rows"] = consumer_controls(
        dict(
            producer,
            honest_verdict="complete_null_identity_fixture",
            verdict_class="null",
            durable_producer_ready_score=1,
            required_checks_passed=True,
            flagged_adversarial=False,
        ),
        raw / "consumer_controls",
    )
    end = time.monotonic_ns()
    work.update(
        start_ns=start,
        end_ns=end,
        duration_s=(end - start) / 1e9,
        started_at_utc=wall,
        ended_at_utc=datetime.now(UTC).isoformat(),
        phase_spans=[
            dict(
                phase="authenticate_and_measure_identity_windows",
                start_ns=start,
                end_ns=end,
                duration_s=(end - start) / 1e9,
            )
        ],
    )
    work["raw_refs"] += [
        e.reference(p) for p in sorted((raw / "consumer_controls").rglob("*.json"))
    ]
    seal(work)
    return work


def seal(work: Json) -> None:
    """Independent primitive seals prevent a candidate from blessing edited summaries."""
    raw = Path(work["raw"])
    atomic_json(raw / "measurement.json", work)
    atomic_json(
        raw / "measurement_seal.json", dict(measurement=e.reference(raw / "measurement.json"))
    )


def replay(path: Path) -> None:
    """Check exact identity before the original seal, raw-byte and reduction checks."""
    value = json.loads(path.read_bytes())
    producer = value["producer_descriptor"]
    e.exact_identity(value, producer)
    work = e.read(value["primitive_reference"])
    if producer != work["producer_descriptor"] or producer != e.read(
        work["producer_descriptor_reference"]
    ):
        raise ValueError("producer_descriptor_drift")
    base_replay(path)


def run(output: Path, date: str, *, private: bool = False) -> int:
    """Publish exact checked bytes and retain the live consumer's selected hash."""
    if private and (
        output.absolute().is_relative_to(e.ROOT) or output.resolve().is_relative_to(e.ROOT)
    ):
        raise ValueError("private_output")
    output = output.resolve()
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    work = measure(raw, date, private=private)
    receipts = (
        [dict(name="private_control_only", passed=True)] if private else runtime.validate(raw)
    )
    if not private:
        e.progress("repository_health_before")
        health = child(
            "full_python_suite",
            [str(e.ROOT / ".venv/bin/pytest"), "tests/python", "-q"],
            raw / "repository_health",
            deadline=180,
            scope="repository_health",
        )
        work["repository_health_reference"] = dict(
            current_receipt=health, full_suite_pass_claimed=False
        )
        seal(work)
        e.progress("repository_health_after")
    atomic_json(raw / "validation.json", dict(receipts=receipts))
    atomic_json(raw / "validation_seal.json", dict(validation=e.reference(raw / "validation.json")))
    value = e.build(work, receipts, output)
    e.exact_identity(value, work["producer_descriptor"])
    publication = publish_primary(
        output, value, lambda path: runtime.terminal(path, raw / "terminal")
    )
    reader = reader_receipt(e.TASK, output.parent, field="task_id", expected=e.TASK)
    readiness = reader_receipt(
        e.TASK,
        output.parent,
        field="durable_producer_ready_score",
        expected=value["durable_producer_ready_score"],
    )
    atomic_json(
        raw / "terminal_validation.json",
        dict(
            publication=publication,
            consumer=reader,
            readiness_consumer=readiness,
            passed=reader["passed"] and readiness["passed"],
        ),
    )
    e.progress(value["honest_verdict"], value["completed_count"], value["censored_count"])
    return 0 if value["required_checks_passed"] and reader["passed"] and readiness["passed"] else 1


runtime.worker = worker
main = runtime.main
runtime.run, runtime.measure, runtime.replay = run, measure, replay

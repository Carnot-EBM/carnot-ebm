"""REQ-REPORT-8771: reuse lifecycle arithmetic with immutable V747 test authority."""

from __future__ import annotations

from datetime import datetime, UTC
from pathlib import Path
import os
import sys
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import lifecycle_event_8758 as previous
from carnot.reporting import lifecycle_event_runner_8758 as previous_runner
from carnot.reporting import lifecycle_archive_authority_8771 as authority
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
e = namespace(previous, "lifecycle8771_report")
runtime = namespace(previous_runner, "lifecycle8771_runner")
e.NAME, e.TASK = (
    "experiment_8771_v751_lifecycle_archive_qualification",
    "exp8771-lifecycle-archive-qualification",
)
e.CLI = f"scripts/experiments/{e.NAME}.py"
e.OWNED = [
    "python/carnot/reporting/lifecycle_archive_authority_8771.py",
    "python/carnot/reporting/lifecycle_archive_8771.py",
    e.CLI,
]
e.TESTS = ["tests/python/test_lifecycle_archive_8771.py"]
runtime.e = e
runtime.E2E = (
    previous.TESTS
    + previous_runner.E2E
    + ["tests/python/test_experiment_7891_v685_authority_lifecycle.py"]
)
e.METHOD = dict(previous.METHOD, durable_cost_successor="exp8772-ordinary-update-cost")
base_build, base_measure, base_validate, base_inputs = (
    e.build,
    runtime.measure,
    runtime.validate,
    e.inputs,
)


def inputs() -> Json:
    """Use the actual current opt-in binary only under the original numerical witness hash."""
    value: Json = base_inputs()
    if not value["failures"]:
        actual = e.reference(Path(e.resident.original.DEFAULT_EXTENSION))
        value["refs"].append(actual)
        if actual["sha256"] != value["extension"]["sha256"]:
            value["failures"].append(
                dict(
                    upstream="current_sparse_native_runtime",
                    **actual,
                    check="loaded_binary_identity",
                    field="current_native_binary",
                    operator="==",
                    expected=value["extension"]["sha256"],
                    observed=actual["sha256"],
                    passed=False,
                    operand_state="missing_file" if actual["sha256"] is None else "changed_bytes",
                )
            )
        else:
            value["extension"] = dict(actual, actual_loaded=True)
    return value


def measure(raw: Path, date: str) -> Json:
    """Freeze the original failure and historical dependencies before numerical replay."""
    start, wall = time.monotonic_ns(), datetime.now(UTC).isoformat()
    e.progress("v747_authority_before")
    fixture_root = raw.with_name(raw.name + "-authority")
    failure, refs, nodes = authority.reproduce(fixture_root / "failure")
    root, manifest = authority.fixture(fixture_root / "root")
    historical = authority.verify_fixture(root, manifest["sha256"])
    historical["manifest_reference"] = manifest
    refs += [manifest, e.reference(root / authority.HISTORICAL_CLI)]
    refs += e.read(manifest)["references"]
    refs += [e.reference(e.ROOT / p) for p in previous.OWNED + previous.TESTS]
    refs += [e.reference(Path(os.__file__)), e.reference(Path(sys.executable))]
    admitted = e.inputs()
    admitted["refs"] += refs
    frozen = time.monotonic_ns()
    e.progress("immutable_v747_frozen_before_current_lifecycle")
    with patch.object(e, "inputs", lambda: admitted):
        work: Json = base_measure(raw, date)
    end = time.monotonic_ns()
    work.update(
        historical_authority_rows=[historical],
        historical_failure_reproduction=failure,
        original_compiled_nodes=nodes,
        historical_fixture_manifest=manifest,
        native_runtime_reference=admitted["extension"],
        start_ns=start,
        end_ns=end,
        started_at_utc=wall,
        ended_at_utc=datetime.now(UTC).isoformat(),
        duration_s=(end - start) / 1e9,
        phase_spans=[
            dict(
                phase="reproduce_and_freeze_authority",
                start_ns=start,
                end_ns=frozen,
                duration_s=(frozen - start) / 1e9,
            ),
            dict(
                phase="current_lifecycle_replay",
                start_ns=frozen,
                end_ns=end,
                duration_s=(end - frozen) / 1e9,
            ),
        ],
    )
    atomic_json(raw / "measurement.json", work)
    atomic_json(
        raw / "measurement_seal.json", dict(measurement=e.reference(raw / "measurement.json"))
    )
    return work


def build(work: Json, receipts: list[Json], output: Path) -> Json:
    """Historical qualification and current acceptance must both precede readiness."""
    value: Json = base_build(work, receipts, output)
    union = next((r for r in receipts if r["name"] == "original_disjoint_union"), {})
    historical_ok = bool(work["historical_authority_rows"]) and all(
        r["passed"] for r in work["historical_authority_rows"]
    )
    union_ok = value["private_control"] or (
        union.get("passed", False)
        and set(work["original_compiled_nodes"]) <= set(union.get("executed_node_ids", []))
    )
    ready = historical_ok and union_ok
    value.update(
        experiment_id=8771,
        milestone="2026.11.751",
        random_seed=7518771,
        honest_verdict=value["honest_verdict"].replace("lifecycle_event", "lifecycle_archive"),
        historical_authority_rows=work["historical_authority_rows"],
        historical_failure_reproduction=work["historical_failure_reproduction"],
        original_node_union=union,
        ordinary_window_ready_score=int(ready and value["ordinary_window_ready_score"]),
        threshold_fixture_ready_score=int(ready and value["threshold_fixture_ready_score"]),
        inference_substrate="deterministic_verifier",
    )
    value["acceptance_gates"].update(
        immutable_v747_authority=historical_ok, original_union=bool(union_ok)
    )
    value["scoped_validation_scope"].update(
        consumers=runtime.E2E,
        historical_producers_requalified=False,
        named_e2es=["E2E-003", "E2E-004", "E2E-018", "E2E-021"],
    )
    value["reproducibility_checksum"] = canonical_hash(
        dict(
            method=e.METHOD,
            code=work["code_refs"],
            sources=work["source_refs"],
            authority=work["historical_fixture_manifest"],
        )
    )
    value["field_principles"].update(
        {
            k: "Bind immutable historical authority and current execution separately; no natural transfer or speed claim."
            for k in value
        }
    )
    return value


def validate(raw: Path) -> list[Json]:
    """Bind only the original compiled-sparse shard to its private historical root."""
    work = e.read(e.reference(raw / "measurement.json"))
    manifest = work["historical_fixture_manifest"]
    root = Path(manifest["path"]).parent
    original_child = runtime.checks.child

    def historical_child(name: str, argv: list[str], logs: Path, **kwargs: Any) -> Json:
        """Other tests retain their own current bindings and real child processes."""
        if any(s.startswith("tests/python/test_compiled_sparse_8719.py::") for s in argv):
            with patch.dict(
                os.environ,
                CARNOT_8771_ARCHIVE=str(root),
                CARNOT_8771_ARCHIVE_SHA256=manifest["sha256"],
                PYTEST_PLUGINS="carnot.reporting.lifecycle_archive_authority_8771",
            ):
                return dict(original_child(name, argv, logs, **kwargs))
        return dict(original_child(name, argv, logs, **kwargs))

    with patch.object(runtime.checks, "child", historical_child):
        return list(base_validate(raw))


e.inputs, e.build, runtime.measure, runtime.validate = inputs, build, measure, validate
main = runtime.main

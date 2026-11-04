"""REQ-VERIFY-8758: freeze evidence, cross real processes and publish checked bytes."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, UTC
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import lifecycle_event_8758 as e
from carnot.reporting import v743_capstone as checks
from carnot.reporting.board_operation_boundary_8454 import resources
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
E2E = [
    "tests/python/test_resident_sparse_8731.py",
    "tests/python/test_compiled_sparse_8719.py",
    "tests/python/test_spline_lifecycle_8706.py",
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_restricted_decision_audit_8210.py",
]


def segment(raw: Path, request: Json, name: str, hard_exit: bool = False) -> tuple[Json, Json]:
    """An exit after the checked checkpoint proves resume does not rely on live objects."""
    path = raw / (name + "-request.json")
    atomic_json(path, request)
    target = raw / (name + "-result.json")
    command = child(
        name,
        [sys.executable, "-u", e.CLI, "--worker", str(path), "--worker-output", str(target)]
        + (["--hard-exit"] if hard_exit else []),
        raw / "children",
        expected=73 if hard_exit else 0,
        deadline=120,
    )
    if not command["passed"]:
        raise ValueError("qualification_child_failed")
    return e.read(e.reference(target)), command


def qualify(admitted: Json, work: Json, raw: Path) -> None:
    """Each arm sees the same frozen starting boundary and original update ordinals."""
    starts = {r["offset"]: e.read(r["reference"]) for r in work["ordinary_starting_state_rows"]}
    work.update(rows=[], restart_rows=[], threshold_control_rows=[])
    for cell in e.roster():
        e.progress("cell_before_" + cell["unit_id"], len(work["rows"]), 24 - len(work["rows"]))
        job = admitted["job"]
        synthetic = cell["synthetic_boundary"]
        low = 0 if synthetic else cell["offset"]
        high = 1 if synthetic else low + 32
        saved = deepcopy(starts[low])
        if synthetic:
            job = next(
                j
                for j in admitted["threshold_jobs"]
                if j["boundary_offset"] == cell["threshold_offset"]
            )
            f = e.numeric.Fixed(32, 24)
            state = e.numeric.encode(job["head"], f)
            state["decay_scale"] = f.unit // 256 + cell["threshold_offset"]
            saved = dict(
                state=state,
                counts=f.counts,
                operations=f.operations,
                maximum=f.maximum,
                ack=canonical_hash(dict(job_hash=canonical_hash(job), next_index=0, state=state)),
                next_index=0,
            )
        request = dict(
            job=job,
            snapshot=saved,
            arm=cell["arm"],
            start=low,
            stop=high,
            extension=admitted["extension"],
        )
        name = cell["unit_id"].replace(":", "-")
        baseline = e.worker(dict(request, arm="python"), raw / (name + "-expected.json"))
        commands = []
        if low == 2032:
            first, command = segment(raw, dict(request, stop=2048), name + "-exit", True)
            commands.append(command)
            last, command = segment(
                raw, dict(request, snapshot=first["checkpoint"], start=2048), name + "-resume"
            )
            commands.append(command)
            actual = dict(
                last,
                rows=first["rows"] + last["rows"],
                import_check=dict(
                    passed=first["import_check"]["passed"] and last["import_check"]["passed"]
                ),
            )
            restart_ok = first["checkpoint"] == starts[2048] and actual["rows"] == baseline["rows"]
            work["restart_rows"].append(
                dict(
                    arm=cell["arm"],
                    restart_after=2048,
                    hard_exit=commands[0]["actual_exit"],
                    resume_exit=commands[1]["actual_exit"],
                    acknowledged_update_replayed=False,
                    checkpoint_identical=first["checkpoint"] == starts[2048],
                    passed=restart_ok,
                )
            )
        else:
            actual, command = segment(raw, request, name)
            commands.append(command)
        matched = (
            actual == baseline
            if low != 2032
            else actual["rows"] == baseline["rows"] and actual["import_check"]["passed"]
        )
        expected_dense = 109 * e.scalar(saved["state"]["decay_scale"])[1] if synthetic else 0
        dense = sum(r["dense_writes"] for r in actual["rows"])
        passed = matched and dense == expected_dense
        mismatch = None
        if not matched:
            mismatch = dict(expected=baseline, observed=actual)
            work["failures"].append(
                dict(
                    upstream="exp8731-resident-sparse-prototype",
                    **admitted["trace_reference"],
                    check="exact_original_state_import_and_window",
                    field=cell["unit_id"],
                    operator="==",
                    expected=baseline,
                    observed=actual,
                    passed=False,
                )
            )
        row = dict(
            cell,
            passed=passed,
            status="completed" if passed else "failed",
            observed_updates=len(actual["rows"]),
            dense_writes=dense,
            expected_dense_writes=expected_dense,
            mismatch=mismatch,
            complete_state_probability_touched_coefficients_ack_identical=matched,
            import_check=actual["import_check"],
            commands=commands,
            evidence_reference=e.reference(raw / (name + "-expected.json")),
        )
        work["rows"].append(row)
        if synthetic:
            work["threshold_control_rows"].append(row)
        e.progress("cell_after_" + cell["unit_id"], len(work["rows"]), 24 - len(work["rows"]))


def measure(raw: Path, date: str) -> Json:
    """Freeze sources and actual test nodes before the first numerical operation."""
    start, wall = time.monotonic_ns(), datetime.now(UTC).isoformat()
    raw.mkdir(parents=True)
    e.progress("authenticate_before")
    admitted = e.inputs()
    pre = resources(raw)
    pre["write_probe_passed"] = (raw / "write_probe").read_bytes() == b"private disk scratch"
    collection = child(
        "premeasurement_nodes",
        [sys.executable, "-m", "pytest", *checks.COMMON, "--collect-only", "-q", *e.TESTS, *E2E],
        raw / "collection",
        deadline=90,
    )
    if not collection["passed"]:
        raise ValueError("test_collection_failed")
    nodes = [
        s
        for s in Path(collection["stdout_path"]).read_text().splitlines()
        if s.startswith("tests/") and "::" in s
    ]
    dependencies = (
        e.OWNED
        + e.TESTS
        + E2E
        + [
            "python/carnot/verify/spline_lifecycle_8706.py",
            "python/carnot/verify/spline_precision_8692.py",
            "python/carnot/verify/compiled_sparse_8719.py",
            "python/carnot/verify/resident_sparse_8731.py",
            "python/carnot/reporting/primary_publication.py",
            "python/carnot/reporting/bounded_durable_cost_8745.py",
            "scripts/experiment_template.py",
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
            "ops/exclusion_manifest.yaml",
            "ops/e2e-test-plan.md",
            "openspec/change-proposals/research-roadmap-vNEXT.md",
            "openspec/change-proposals/research-roadmap-v749-preserved-20261102.md",
        ]
    )
    code = [e.reference(e.ROOT / p) for p in dependencies]
    atomic_json(raw / "cost_protocol.json", e.METHOD)
    atomic_json(
        raw / "premeasurement_seal.json",
        dict(
            method=e.METHOD,
            code=code,
            inputs=admitted["refs"],
            nodes=nodes,
            node_hash=canonical_hash(nodes),
        ),
    )
    failures = list(admitted["failures"])
    if not (pre["disk_backed"] and pre["tools_available"] and pre["write_probe_passed"]):
        failures.append(
            dict(
                upstream="local_host",
                path=str(raw),
                sha256=None,
                check="preconditions",
                field="disk_tools_writable",
                operator="==",
                expected=True,
                observed=pre,
                passed=False,
            )
        )
    work: Json = dict(
        rows=[],
        restart_rows=[],
        threshold_control_rows=[],
        ordinary_starting_state_rows=[],
        scalar_decay_rows=[],
        natural_event_rows=[],
        failures=failures,
    )
    e.progress("sources_inputs_methods_nodes_frozen")
    if not failures:
        work.update(e.reconstruct(admitted, raw))
        qualify(admitted, work, raw)
    end = time.monotonic_ns()
    old = e.read(e.reference(e.ROOT / "results/experiment_8745_v749_bounded_durable_cost.json"))
    health = next(r for r in old["validation_receipts"] if r["name"] == "full_python_suite")
    work.update(
        raw=str(raw),
        run_date=date,
        preconditions=pre,
        source_refs=admitted["refs"],
        code_refs=code,
        cost_protocol=e.METHOD,
        historical_restart=admitted.get("historical_restart"),
        repository_health_reference=dict(historical_receipt=health, full_suite_pass_claimed=False),
        raw_refs=[e.reference(p) for p in sorted(raw.rglob("*")) if p.is_file()],
        start_ns=start,
        end_ns=end,
        started_at_utc=wall,
        ended_at_utc=datetime.now(UTC).isoformat(),
        duration_s=(end - start) / 1e9,
        phase_spans=[
            dict(
                phase="authenticate_freeze_reconstruct_qualify",
                start_ns=start,
                end_ns=end,
                duration_s=(end - start) / 1e9,
            )
        ],
    )
    atomic_json(raw / "measurement.json", work)
    atomic_json(
        raw / "measurement_seal.json", dict(measurement=e.reference(raw / "measurement.json"))
    )
    return work


def replay(path: Path) -> None:
    """Independent seals and raw boundaries prevent outer reports from blessing edits."""
    value = json.loads(path.read_bytes())
    raw = (
        Path(value["publication_output"]).parent
        / "raw"
        / e.NAME
        / "invocations"
        / value["invocation_id"]
    )
    seal = e.read(e.reference(raw / "measurement_seal.json"))
    if seal["measurement"] != value["primitive_reference"]:
        raise ValueError("independent_measurement_seal")
    work = e.read(value["primitive_reference"])
    for ref in work["source_refs"] + work["code_refs"] + work["raw_refs"]:
        e.verify(ref)
    frozen = e.read(e.reference(raw / "premeasurement_seal.json"))
    if (
        frozen["code"] != work["code_refs"]
        or frozen["inputs"] != work["source_refs"]
        or frozen["method"] != e.METHOD
        or frozen["node_hash"] != canonical_hash(frozen["nodes"])
        or work["duration_s"] != (work["end_ns"] - work["start_ns"]) / 1e9
    ):
        raise ValueError("frozen_method_or_clock_drift")
    if work["natural_event_rows"]:
        replayed = e.reconstruct(e.inputs(), raw / "cold_boundaries")
        for key in ["natural_event_rows", "scalar_decay_rows"]:
            if replayed[key] != work[key]:
                raise ValueError("cold_event_drift")
    validation = e.read(e.reference(raw / "validation_seal.json"))["validation"]
    if validation != value["validation_reference"]:
        raise ValueError("independent_validation_seal")
    receipts = e.read(validation)["receipts"]
    for receipt in receipts:
        for stream in ["stdout", "stderr"]:
            if stream + "_path" in receipt:
                e.verify(dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"]))
    if e.build(work, receipts, Path(value["publication_output"])) != value:
        raise ValueError("report_drift")
    e.progress("cold_replay_passed")


def validate(raw: Path) -> list[Json]:
    """Reuse original disjoint tests, child coverage SQLite, types and spec tracing."""
    with (
        patch.multiple(checks, e=e, E2E=E2E),
        patch.dict(os.environ, CARNOT_8719_EXTENSION=e.inputs()["extension"]["path"]),
    ):
        return list(checks.validate(raw))


def terminal(path: Path, raw: Path) -> Json:
    """Unchanged validators inspect the same final candidate as fresh-process replay."""
    with patch.object(checks, "e", e):
        return dict(checks.terminal(path, raw))


def run(output: Path, date: str, private: bool = False) -> int:
    """Private children exercise publication while production requires every scoped check."""
    if private and output.resolve().is_relative_to(e.ROOT):
        raise ValueError("private_output")
    output = output.resolve()
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    work = measure(raw, date)
    work["private_control"] = private
    if not private:
        with patch.dict(
            os.environ,
            COVERAGE_FILE=str(raw / "repository_health/.coverage"),
            PYTEST_ADDOPTS="--basetemp=" + str(raw / "repository_health/private_temp"),
        ):
            health = child(
                "full_python_suite",
                [str(e.ROOT / ".venv/bin/pytest"), "tests/python", "-q"],
                raw / "repository_health",
                deadline=180,
                scope="repository_health",
            )
        work["repository_health_reference"]["current_receipt"] = health
    atomic_json(raw / "measurement.json", work)
    atomic_json(
        raw / "measurement_seal.json", dict(measurement=e.reference(raw / "measurement.json"))
    )
    receipts = [dict(name="private_control_only", passed=True)] if private else validate(raw)
    atomic_json(raw / "validation.json", dict(receipts=receipts))
    atomic_json(raw / "validation_seal.json", dict(validation=e.reference(raw / "validation.json")))
    value = e.build(work, receipts, output)
    publication = publish_primary(
        output, value, lambda candidate: terminal(candidate, raw / "terminal")
    )
    atomic_json(raw / "terminal_validation.json", dict(publication=publication, passed=True))
    e.progress(value["honest_verdict"], value["completed_count"], value["censored_count"])
    return 0 if value["required_checks_passed"] else 1


def main(argv: list[str] | None = None) -> int:
    """Use one thin entrypoint for real qualification children and terminal replay."""
    e.progress("start_no_model_load_MODEL_SPECS_empty")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20261103")
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--hard-exit", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.worker:
            if not args.worker_output:
                raise ValueError("worker_contract")
            e.worker(e.read(e.reference(args.worker)), args.worker_output)
            if args.hard_exit:
                os._exit(73)
            return 0
        if args.cold_replay:
            replay(args.cold_replay)
            return 0
        if args.date != "20261103":
            raise ValueError("calendar_date")
        return run(args.output, args.date, args.private_control)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"[exp8758] error={error}", file=sys.stderr, flush=True)
        return 1

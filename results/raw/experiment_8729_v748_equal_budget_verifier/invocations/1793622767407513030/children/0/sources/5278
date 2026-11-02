"""REQ-VERIFY-8692: bounded CPU measurement and immutable final-byte publication."""

from __future__ import annotations

import argparse
from datetime import datetime, UTC
import json
from importlib.metadata import version
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import spline_precision_study_8692 as e
from carnot.reporting import v743_capstone as validation
from carnot.reporting.board_operation_boundary_8454 import resources
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.gatemate_continuity_8400 import budget
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child

Json = dict[str, Any]


def measure(raw: Path, date: str, private_control: bool = False) -> Json:
    """Seal source and target-blind geometry before any measured update is executed."""
    start, wall = time.monotonic_ns(), datetime.now(UTC).isoformat()
    e.progress("preconditions_before")
    with tempfile.TemporaryDirectory(prefix="spline8692-", dir="/var/tmp") as directory:
        private = Path(directory)
        pre = resources(private)
        pre["software_versions"] = dict(
            python=sys.version, numpy=version("numpy"), scipy=version("scipy")
        )
        inputs = e.authenticate()
        if private_control:
            inputs["heads"] = inputs["heads"][:1]
        failures = inputs["failures"]
        if not (
            pre["disk_backed"]
            and pre["tools_available"]
            and pre["free_disk_bytes"] >= 2**30
            and pre["memory_available_bytes"] >= 2**30
        ):
            failures.append(
                dict(
                    upstream="host",
                    check="cpu_resources",
                    path=directory,
                    sha256=None,
                    field="disk_memory_tools",
                    operator="==",
                    expected=True,
                    observed=pre,
                    passed=False,
                    operand_state="resource_unavailable",
                )
            )
        pre.update(optional_devices_required=False, language_model_calls=0)
        raw.mkdir(parents=True, exist_ok=True)
        code_refs = [e.reference(e.ROOT / p) for p in e.OWNED + e.TESTS]
        dependencies = [
            e.PROTOCOL,
            "ops/exclusion_manifest.yaml",
            "openspec/capabilities/verification/spec.md",
            "openspec/capabilities/research-reporting/spec.md",
            "python/carnot/verify/sparse_energy_7996.py",
            "python/carnot/verify/objective_checkpoint_8461.py",
            "python/carnot/inference/resident_energy.py",
            "python/carnot/reporting/primary_publication.py",
            "python/carnot/reporting/v743_capstone.py",
            "python/carnot/reporting/v709_execution.py",
            "python/carnot/verify/v726_criterion_energy_fit.py",
            "scripts/check_spec_coverage.py",
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
            "docs/research-notes/v745-spline-precision-feasibility.md",
        ]
        source_refs = inputs["refs"] + [e.reference(e.ROOT / p) for p in dependencies]
        for ref in code_refs + source_refs:
            source = Path(ref["path"])
            if source.is_file():
                target = raw / "source_closure" / source.relative_to(e.ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        atomic_json(private / "initial_heads.json", dict(heads=inputs["heads"]))
        atomic_json(raw / "initial_heads.json", dict(heads=inputs["heads"]))
        frozen = e.numeric.vectors(inputs["heads"]) if inputs["heads"] else {}
        atomic_json(raw / "frozen_vectors.json", frozen)
        source_refs.extend(
            [e.reference(raw / "initial_heads.json"), e.reference(raw / "frozen_vectors.json")]
        )
        atomic_json(
            raw / "premeasurement_seal.json",
            dict(
                source_refs=source_refs,
                code_refs=code_refs,
                rule=e.numeric.original.RULE,
                rule_hash=e.numeric.original.RULE_HASH,
                configuration=dict(
                    seed=e.numeric.SEED, formats=e.numeric.FORMATS, primary=[32, 24]
                ),
            ),
        )
        e.progress("sources_inputs_frozen_before_benchmark", 0, 3 * (len(inputs["heads"]) or 5))
        result = e.evaluate(inputs["heads"], frozen)
        e.progress("durable_storage_before", 0, len(result["serialized_final_states"]))
        for index, state in enumerate(result["serialized_final_states"]):
            start_storage = time.perf_counter_ns()
            target = raw / "states" / f"{index:03}.bin"
            target.parent.mkdir(exist_ok=True)
            with target.open("wb") as stream:
                stream.write(bytes.fromhex(state["encoded_hex"]))
                stream.flush()
                os.fsync(stream.fileno())
            end_storage = time.perf_counter_ns()
            result["cpu_operation_spans"].append(
                dict(
                    operation="host_durable_storage",
                    start_ns=start_storage,
                    end_ns=end_storage,
                    duration_ns=end_storage - start_storage,
                )
            )
            source_refs.append(e.reference(target))
        storage_spans = [
            s for s in result["cpu_operation_spans"] if s["operation"] == "host_durable_storage"
        ]
        result["operation_inventory"].append(
            dict(
                operation="host_durable_storage",
                device="cpu",
                duration_ns=sum(s["duration_ns"] for s in storage_spans),
                calls=len(storage_spans),
            )
        )
        e.progress("durable_storage_after", len(result["serialized_final_states"]), 0)
    end = time.monotonic_ns()
    if (end - start) / 1e9 >= 3600:
        raise TimeoutError("measurement_deadline")
    result.update(
        raw=str(raw),
        run_date=date,
        source_refs=source_refs,
        code_refs=code_refs,
        failures=failures,
        preconditions=pre,
        started_at_utc=wall,
        ended_at_utc=datetime.now(UTC).isoformat(),
        duration_s=(end - start) / 1e9,
        start_ns=start,
        end_ns=end,
        historical_model_provenance=inputs["historical_model_provenance"],
        phase_spans=[
            dict(
                phase="authenticate_freeze_measure",
                start_ns=start,
                end_ns=end,
                duration_s=(end - start) / 1e9,
            )
        ],
    )
    atomic_json(raw / "measurement.json", result)
    seal = dict(measurement=e.reference(raw / "measurement.json"), code_refs=code_refs)
    atomic_json(raw.parents[1] / "seals" / (raw.name + ".json"), seal)
    e.progress(
        "measurement_complete",
        len(result["rows"]),
        result["initial_head_count"] * 3 - len(result["rows"]),
    )
    return result


def validate(raw: Path) -> list[Json]:
    """Keep inherited disjoint collection, actual child exits and raw SQLite custody."""
    with tempfile.TemporaryDirectory(prefix="spline8692-cli-", dir="/var/tmp") as directory:
        with (
            patch.object(validation, "e", e),
            patch.dict(os.environ, {"CARNOT_8692_CLI_RETENTION": directory}),
        ):
            receipts = list(validation.validate(raw))
        destination = raw / "private_cli_receipts"
        shutil.copytree(directory, destination)
        nested = sorted(destination.rglob("*.receipt.json"))
        receipts.append(
            dict(
                name="retained_private_cli_children",
                passed=bool(nested),
                references=[e.reference(p) for p in sorted(destination.rglob("*")) if p.is_file()],
            )
        )
        junit = sorted((raw / "validation").glob("*.xml"))
        receipts.append(
            dict(
                name="retained_junit",
                passed=bool(junit),
                references=[e.reference(p) for p in junit],
            )
        )
        return receipts


def terminal(path: Path, raw: Path) -> Json:
    """Run unchanged scanners against the same immutable candidate as fresh replay."""
    with patch.object(validation, "e", e):
        return dict(validation.terminal(path, raw))


def replay(path: Path) -> None:
    """Independent measurement seals defeat consistent local rehashing of a candidate."""
    value = json.loads(path.read_bytes())
    anchor = Path(value["publication_output"]).parent / "raw" / e.NAME
    raw = anchor / "invocations" / value["invocation_id"]
    seal = e.read(e.reference(anchor / "seals" / (raw.name + ".json")))
    if seal["measurement"] != value["primitive_reference"]:
        raise ValueError("independent_measurement_seal")
    work = e.read(value["primitive_reference"])
    if work["raw"] != str(raw) or work["code_refs"] != seal["code_refs"]:
        raise ValueError("independent_source_seal")
    for ref in work["code_refs"] + work["source_refs"]:
        e.read(ref) if Path(ref["path"]).suffix == ".json" else verify_bytes(ref)
    if work["duration_s"] != (work["end_ns"] - work["start_ns"]) / 1e9:
        raise ValueError("clock_drift")
    heads = e.read(e.reference(raw / "initial_heads.json"))["heads"]
    frozen = e.read(e.reference(raw / "frozen_vectors.json"))
    if heads and frozen != e.numeric.vectors(heads):
        raise ValueError("geometry_drift")
    recomputed = e.evaluate(heads, frozen)
    for key in [
        "rows",
        "precision_rows",
        "trajectory_rows",
        "primary_passed",
        "saturation_counts",
        "accumulator_bounds",
        "serialized_state_bytes",
        "serialized_final_states",
    ]:
        if recomputed[key] != work[key]:
            raise ValueError("numerical_replay_drift:" + key)
    receipts = e.read(value["validation_reference"])["receipts"]
    validation_seal = e.read(e.reference(anchor / "validation_seals" / (raw.name + ".json")))
    if validation_seal["validation"] != value["validation_reference"]:
        raise ValueError("independent_validation_seal")
    for receipt in receipts:
        for ref in receipt.get("references", []) + [
            s["reference"] for s in receipt.get("shards", [])
        ]:
            verify_bytes(ref)
        for stream in ("stdout", "stderr"):
            if stream + "_path" in receipt:
                verify_bytes(
                    dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
                )
    if e.build(work, receipts, Path(value["publication_output"])) != value:
        raise ValueError("report_drift")
    e.progress("cold_replay_passed", work["initial_head_count"] * 3, 0)


def verify_bytes(ref: Json) -> None:
    """Source bytes need authentication without pretending that Python is JSON."""
    if e.sha256_file(Path(ref["path"])) != ref["sha256"]:
        raise ValueError("operand_changed_bytes")


def run(output: Path, date: str, private: bool, failed_child: bool) -> int:
    """Private controls stay outside both lexical and resolved checkout destinations."""
    if (private or failed_child) and (
        output.absolute().is_relative_to(e.ROOT) or output.resolve().is_relative_to(e.ROOT)
    ):
        raise ValueError("private output must be outside checkout")
    output = output.resolve()
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    work = measure(raw, date, private)
    receipts = (
        [
            dict(
                name="private_control_only",
                passed=True,
                scope="private_fixture_not_production_validation",
            )
        ]
        if private
        else validate(raw)
    )
    if failed_child:
        receipts.append(
            child(
                "failed_child",
                [sys.executable, "-c", "raise SystemExit(1)"],
                raw / "validation",
                deadline=10,
            )
        )
    atomic_json(raw / "validation.json", dict(receipts=receipts))
    atomic_json(
        raw.parents[1] / "validation_seals" / (raw.name + ".json"),
        dict(validation=e.reference(raw / "validation.json")),
    )
    value = e.build(work, receipts, output)
    publication = publish_primary(output, value, lambda path: terminal(path, raw / "terminal"))
    consumer = reader_receipt(
        e.TASK,
        output.parent,
        field="precision_mapping_ready_score",
        expected=value["precision_mapping_ready_score"],
    )
    atomic_json(
        raw / "terminal_validation.json",
        dict(
            publication=publication,
            consumer=consumer,
            primary_sha256=publication["primary_sha256"],
            passed=consumer["passed"],
        ),
    )
    e.progress(
        "complete_" + value["verdict_class"], value["completed_count"], value["censored_count"]
    )
    return 0 if value["required_checks_passed"] and consumer["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    """Accept the registered date and retain actual rejection exits in private children."""
    e.progress("start_no_model_load_MODEL_SPECS_empty_language_model_calls_zero")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20261101")
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--failed-child", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.cold_replay:
            replay(args.cold_replay)
            return 0
        if len(args.date) != 8:
            raise ValueError("calendar_date")
        datetime.strptime(args.date, "%Y%m%d")
        with budget(4800):
            return run(args.output.absolute(), args.date, args.private_control, args.failed_child)
    except (OSError, ValueError, KeyError, TypeError, TimeoutError) as error:
        print(f"[exp8692] error={error}", file=sys.stderr, flush=True)
        return 1

"""REQ-REPORT-8921: execute single-request development parents before publication."""

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from zipfile import ZipFile

from carnot.reporting import inline_clock_methods_8921 as m
from carnot.reporting import buffered_clock_runner_8894 as validation_runtime
from carnot.reporting import size_aware_complete_runner_8874 as parent
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
)
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.selective_methods_runner_8829 import qualified
from carnot.reporting.selective_runner_8816 import stream, unqualified
from carnot.reporting.v746_retained_evidence import load, reference
from carnot.reporting.v748_collection_contract import namespace
from scripts.experiment_template import ExperimentTemplate
from scripts.experiments import inline_clock_worker_8921 as worker

Json = dict[str, Any]


def measure(raw: Path, worlds: list[Json], evidence: Json, original: Any) -> Json:
    """Install retained bytes once; each timed parent contains one complete request."""
    workers = raw / "workers"
    workers.mkdir(parents=True)
    for ref in evidence["retained_operand_manifest"]:
        path = Path(ref["original_path"])
        if path.parent == m.ROOT / "scripts/experiments" and path.suffix == ".py":
            shutil.copy2(ref["path"], workers / path.name)
    shutil.copy2(m.ROOT / m.WORKER, workers / Path(m.WORKER).name)
    baseline = raw / "baseline.py"
    baseline.write_text(m.prior.baseline_source())
    source_work = original.load(load(evidence["original_primary"])["work_reference"])
    package = source_work["package_hashes"][0]
    wheel = next(
        r for r in evidence["retained_operand_manifest"] if r["sha256"] == package["sha256"]
    )
    with ZipFile(wheel["path"]) as archive:
        pins = {
            Path(p).name: "sha256:" + hashlib.sha256(archive.read(p)).hexdigest()
            for p in archive.namelist()
            if p.startswith("carnot/verify/") and p.endswith(".py")
        }
    rows: list[Json] = []
    refs, installation = [], []
    runner = namespace(parent, "inline8921_parent")
    runner.m = namespace(m.prior, "inline8921_configuration")
    runner.m.WORKER = str(workers / Path(m.WORKER).name)
    runner.m.progress = m.progress
    runner.m.SEED = m.SEED
    deadline = time.monotonic() + 1800
    total = len(worlds) * 4 * 3 * 2 * 6
    with TemporaryDirectory(prefix="exp8921-installed-") as directory:
        private = Path(directory)
        python = private / "venv/bin/python"
        for name, command in [
            ("venv", ["uv", "venv", str(private / "venv"), "--python", sys.executable]),
            (
                "install",
                ["uv", "pip", "install", "--python", str(python), "--no-deps", wheel["path"]],
            ),
        ]:
            m.progress("before_" + name)
            receipt = stream(name, command, raw / "installation", deadline=120)
            installation.append(receipt)
            m.progress("after_" + name)
            if not receipt["passed"]:
                raise ValueError("owned_installation")
        for world in worlds:
            for boundary in m.prior.BOUNDARIES:
                for arm in world["arm_order"]:
                    for unit in world["requests"]:
                        for block, mode in enumerate(world["mode_order"]):
                            m.progress("before_request", len(rows), total - len(rows))
                            selected = dict(
                                world,
                                requests=[unit],
                                arm_order=[arm],
                                instrumented=False if mode == "off" else mode,
                                warmup=boundary == "resident",
                            )
                            result = runner.process(
                                python,
                                baseline,
                                [selected],
                                [0],
                                "cold",
                                private,
                                raw / "parents" / str(len(rows)),
                                deadline,
                            )
                            ref = reference(
                                raw
                                / "parents"
                                / str(len(rows))
                                / (result["receipt"]["name"] + ".json")
                            )
                            row, clock = result["rows"][0], result["receipt"]
                            valid = bool(clock["passed"] and row["status"] != "failed")
                            valid &= len(clock["installed_imports"]) == 4 and all(
                                "/site-packages/carnot/verify/" in p["path"]
                                and not Path(p["path"]).is_relative_to(m.ROOT)
                                and pins.get(Path(p["path"]).name) == p["sha256"]
                                for p in clock["installed_imports"]
                            )
                            a, b = clock["started_monotonic_ns"], clock["ended_monotonic_ns"]
                            begin, end = row.get("block_start_ns", a), row.get("block_end_ns", a)
                            start, stop = row.get("start_ns", a), row.get("end_ns", a)
                            partition = (
                                m.prior.partition(
                                    row["endpoints"],
                                    start,
                                    stop,
                                    endpoint_count=row["endpoint_count"],
                                )
                                if mode != "off" and "endpoints" in row
                                else {}
                            )
                            valid &= a <= begin <= start < stop <= end <= b and (
                                mode == "off" or partition.get("valid", False)
                            )
                            semantics = [{k: row[k] for k in m.prior.SEMANTICS}]
                            rows.append(
                                dict(
                                    unit_id=f"{unit['unit_id']}/{boundary}/{arm}/{block}",
                                    request_id=unit["unit_id"],
                                    cluster_id=world["cluster_id"],
                                    size=world["size"],
                                    arm=arm,
                                    boundary=boundary,
                                    block=block,
                                    mode=mode,
                                    seed=m.SEED,
                                    metric=int(valid),
                                    denominator=1,
                                    status=row["status"],
                                    missing_reason=row["missing_reason"],
                                    elapsed_ns=b - a if boundary == "cold" else end - begin,
                                    parent_elapsed_ns=b - a,
                                    semantics=semantics,
                                    closure_valid=bool(valid),
                                    partition=partition,
                                    parent_start_ns=a,
                                    parent_end_ns=b,
                                    block_start_ns=begin,
                                    block_end_ns=end,
                                    request_start_ns=start,
                                    request_end_ns=stop,
                                    startup_ns=begin - a,
                                    exit_ns=b - end,
                                    child_remainder_ns=end - begin - (stop - start),
                                    raw_reference=ref,
                                )
                            )
                            refs.append(ref)
                            m.progress("after_request", len(rows), total - len(rows))
    return dict(
        rows=rows,
        process_references=refs,
        installation_receipts=installation,
        installed_import_pins=pins,
    )


def prepare(raw: Path, gate_path: Path = m.UPSTREAM) -> Json:
    """Authenticate external custody before spending any development compute."""
    started = time.monotonic_ns()
    raw.mkdir(parents=True)
    diagnostics, errors, refs = [], [], []
    check = m.gate(gate_path)
    ready = check["check"] == "admitted"
    diagnostics.append(check)
    result: Json = dict(rows=[], process_references=[], installation_receipts=[])
    worlds = m.freeze()
    scheduled = m.schedule(worlds)
    controls = m.prior.controls()
    allocations = [worker.allocation_control()]
    if not controls["passed"]:
        errors.append("owned_reduction_controls")
    atomic_json(
        raw / "development_roster.json", dict(worlds=worlds, modes=m.MODES, requests_per_parent=1)
    )
    future_hash = None
    if ready and not errors:
        refs.append(reference(gate_path))
        future_hash = load(refs[-1])["original_roster_hash"]
        try:
            original, _, evidence = m.custody.open_observer()
            atomic_json(raw / "custody.json", evidence)
            refs.extend([evidence["original_primary"], evidence["execution_snapshot"]])
            result = measure(raw, worlds, evidence, original)
        except m.custody.CustodyError as error:
            ready = False
            diagnostics.append(error.diagnostic)
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(str(error))
    observed = {r["unit_id"] for r in result["rows"]}
    for unit in scheduled:
        if unit["unit_id"] not in observed:
            result["rows"].append(
                dict(
                    unit,
                    status="failed" if errors else "censored",
                    missing_reason=errors[0] if errors else "external_custody_gate",
                    elapsed_ns=0,
                    parent_elapsed_ns=0,
                    semantics=[],
                    closure_valid=False,
                    partition={},
                    startup_ns=0,
                    exit_ns=0,
                    child_remainder_ns=0,
                    raw_reference=None,
                )
            )
    ended = time.monotonic_ns()
    spans = [dict(phase="development_qualification", start_s=0, end_s=(ended - started) / 1e9)]
    receipt = build_current_work_receipt(
        run_id=m.TASK + "/" + str(started),
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="no_model_load" if ready else "blocked_no_run",
        inference_substrate_class="no_model_load" if ready else "blocked_no_run",
        inference_substrate_details=dict(
            compute="single-request installed signed-record verifier", model_invoked=False
        ),
        execution_venue="host",
        started_monotonic_ns=started,
        ended_monotonic_ns=ended,
        phase_spans=spans,
    )
    return dict(
        result,
        preconditions_checked=dict(external_ready=ready, diagnostics=diagnostics),
        internal_errors=errors,
        worlds=worlds,
        future_roster_hash=future_hash,
        current_work_receipt=receipt,
        controls=controls,
        allocation_rows=allocations,
        source_artifact_hashes=refs,
        code_config_hashes=[reference(m.ROOT / p) for p in m.OWNED + m.TESTS]
        + [reference(m.ROOT / m.prior.WORKER)],
        raw_shard_hashes=[reference(p) for p in sorted(raw.rglob("*")) if p.is_file()],
    )


def artifact(work: Json, validation: Json, work_ref: Json, validation_ref: Json) -> Json:
    """A measured overhead failure is a null; broken owned checks disqualify."""
    reduced = m.reduce(work["rows"])
    receipt = work["current_work_receipt"]
    gates = dict(
        external_custody=work["preconditions_checked"]["external_ready"],
        owned_execution=not work["internal_errors"],
        controls=work["controls"]["passed"],
        owned_validation=qualified(validation),
        complete_development=reduced["complete"],
        exact_roster={r["unit_id"] for r in work["rows"]}
        == {r["unit_id"] for r in m.schedule(work["worlds"])},
        semantic_identity=reduced["semantic_identity"],
        closure=reduced["finite_nonnegative_closure"],
        resolution=all(
            r["elapsed_ns"] >= 1000 * time.get_clock_info("perf_counter").resolution * 1e9
            for r in work["rows"]
        ),
        fixed_event_storage=all(r["event_container_growth"] == 0 for r in work["allocation_rows"]),
    )
    owned = all(v for k, v in gates.items() if k != "external_custody")
    ready = int(all(gates.values()) and reduced["inline_clock_ready_score"])
    verdict = (
        "blocked"
        if not gates["external_custody"]
        else "disqualified"
        if not owned
        else "circular_positive"
        if ready
        else "null"
    )
    rows = work["rows"]
    intended = len(work["worlds"]) * 4 * 3 * 2 * 6
    value = dict(
        experiment_id=8921,
        task_id=m.TASK,
        milestone="2026.11.763",
        run_date="20261108",
        honest_verdict=f"complete_{verdict}_inline_clock_methods",
        verdict_class=verdict,
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class=receipt["inference_substrate_class"],
        execution_venue="host",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        numerical_training_calls=0,
        verifier_is_oracle=True,
        exposure_scope="constructed signed-record development worlds only",
        independent_generalization_score=0,
        duration_s=receipt["duration_s"],
        phase_spans=receipt["phase_spans"],
        preconditions_checked=work["preconditions_checked"],
        random_seed=m.SEED,
        source_artifact_hashes=work["source_artifact_hashes"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=work["raw_shard_hashes"] + validation.get("raw_references", []),
        cited_upstream_artifacts=work["source_artifact_hashes"],
        rows=rows,
        intended_count=intended,
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=sum(r["status"] == "failed" for r in rows),
        censored_count=intended - sum(r["status"] in ("completed", "failed") for r in rows),
        excluded_count=0,
        independent_count=len({r["cluster_id"] for r in rows if r["raw_reference"]}),
        gate_check_summary=work["preconditions_checked"]["diagnostics"]
        + [
            dict(
                check="observed_overhead_failure",
                upstream=m.TASK,
                artifact_field="development_overhead_rows",
                expected="simultaneous 99-percent upper <= 0.05",
                observed=row,
                path=work_ref.get("path"),
                hash=work_ref.get("sha256"),
            )
            for row in reduced["development_overhead_rows"]
            if row["mode"] == "inline" and not row["accepted"]
        ],
        acceptance_gates=gates,
        required_checks_passed=bool(owned),
        validation_receipts=validation.get("receipts", []),
        owned_statement_coverage=validation["coverage"],
        inline_clock_ready_score=ready,
        observer_schema=dict(
            phases=m.prior.PHASES,
            event_layout=["perf_counter_ns", "phase", "serialization_site"],
            partition_tolerance_ns=0,
            enabled_by_default=False,
            requests_per_parent=1,
            repetitions_per_request=1,
            temporary_integer_allocations_disclosed=True,
        ),
        observer_hashes=work["code_config_hashes"],
        allocation_rows=work["allocation_rows"],
        development_overhead_rows=reduced["development_overhead_rows"],
        closure_rows=[
            {
                k: r[k]
                for k in (
                    "unit_id",
                    "closure_valid",
                    "parent_elapsed_ns",
                    "startup_ns",
                    "exit_ns",
                    "child_remainder_ns",
                    "raw_reference",
                )
            }
            for r in rows
        ],
        future_roster_hash=work["future_roster_hash"],
        future_roster_opened=False,
        current_work_receipt=receipt,
        work_reference=work_ref,
        validation_reference=validation_ref,
        methodology="One complete request per parent. Six balanced observations compare off, buffered and inline twice. Four independent development worlds per size; four request kinds remain clustered within each world. Cold clocks include preparation, import, compilation and exit. Resident clocks preserve the original warmup boundary. All raw parent costs remain retained. Whole-world bootstrap uses 10000 draws and Bonferroni family alpha .01 across 72 comparisons. Zero-nanosecond partition tolerance is unchanged. No future worlds are evaluated. Allocation control measures net retained blocks and bytes, not total transient allocations. Python clock results remain boxed integers. Readiness is not learning or hardware benefit.",
        scientific_benefit_claim=False,
        repository_health=work.get(
            "repository_health", dict(status="not_run", scope="unrelated_repository_health")
        ),
    )
    value["field_principles"] = {
        k: "Retain actual development evidence and its scope; qualification does not establish independent generalization."
        for k in value
    }
    value["field_principles"]["allocation_rows"] = (
        "Fixed numeric event storage avoids event records; net retained snapshots do not count all transient Python allocations."
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def main(argv: list[str] | None = None) -> int:
    """Validate owned sources before publishing one terminal reader-visible artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261108"], default="20261108")
    parser.add_argument("--output", type=Path, default=m.ROOT / "results" / (m.NAME + ".json"))
    parser.add_argument("--private-output", action="store_true")
    parser.add_argument("--gate-path", type=Path, default=m.UPSTREAM)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = m.replay(args.cold_replay)
        m.progress("replay_passed" if passed else "replay_rejected")
        return int(not passed)
    output = args.output.absolute()
    if args.private_output and output.resolve().is_relative_to(m.ROOT):
        parser.error("private output must be outside checkout")
    if args.gate_path != m.UPSTREAM and not args.private_output:
        parser.error("alternate gates require private output")
    raw = output.parent / "raw" / m.NAME / "invocations" / str(time.time_ns())
    template = ExperimentTemplate(
        8921, "Inline request phase clocks", "results/" + m.NAME + ".json", seed=m.SEED
    )
    m.progress("before_development")
    with template.phase("development"):
        work = prepare(raw, args.gate_path)
    if (m.HEALTH / "receipt.json").is_file():
        shutil.copytree(m.HEALTH, raw / "repository_health")
        work["repository_health"] = dict(
            scope="unrelated_full_python_suite",
            receipt=load(reference(raw / "repository_health/receipt.json")),
            references=[reference(p) for p in sorted((raw / "repository_health").iterdir())],
        )
        work["raw_shard_hashes"] += work["repository_health"]["references"]
    m.progress("after_development", len(work["rows"]), 0)
    atomic_json(raw / "work.json", work)
    m.progress("before_owned_validation")
    validator = namespace(validation_runtime, "inline8921_validation")
    validator.m = m
    with TemporaryDirectory(prefix="exp8921-checks-") as directory:
        validation = (
            unqualified()
            if args.private_output
            else validator.validate(raw / "validation", Path(directory))
        )
    atomic_json(raw / "validation.json", validation)
    m.progress("after_owned_validation")
    value = artifact(
        work, validation, reference(raw / "work.json"), reference(raw / "validation.json")
    )
    terminal = namespace(parent, "inline8921_terminal")
    terminal.m = m
    m.progress("before_terminal_publication")
    publication = publish_primary(output, value, lambda path: terminal.terminal(path, raw))
    atomic_json(raw / "publication.json", publication)
    m.progress(
        "complete_" + value["verdict_class"], value["completed_count"], value["censored_count"]
    )
    return 0

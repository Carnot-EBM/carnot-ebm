"""REQ-REPORT-8584: complete measured costs distinguish arithmetic from process lifetime."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

import numpy as np

from carnot.reporting import resident_inference_8571 as old
from carnot.reporting import resident_measure_8572 as peer
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, atomic_json, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.reporting.v709_execution import child, progress
from carnot.verify import nine_input_8533 as numeric

Json = dict[str, Any]
ROOT = old.ROOT
NAME = "experiment_8584_v737_resident_complete_cost"
TASK = "exp8584-resident-complete-cost"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_resident_complete_cost_8584.py"
OWNED = [
    f"python/carnot/reporting/resident_complete_{s}_8584.py" for s in ["cost", "worker", "runner"]
] + [CLI]
MODEL_SPECS: list[Json] = []
GateFailure, require, reference, read, checksum = (
    old.GateFailure,
    old.require,
    old.reference,
    old.read,
    old.checksum,
)
PINS = {
    "results/experiment_8579_v737_contract_methods.json": "sha256:59aa488c9a37dd33cc5564bbe2f87b54ef6744cb63323dc6a291959985741a64",
    "results/experiment_8572_v736_resident_inference_qualification.json": "sha256:12adfee3ca0fe07bf305438484eab8362ff573f715c8379715f4c1399da0d3db",
}
ARMS = ["cold_python", "cold_native", "resident_python", "resident_native"]
REQUIRED = old.REQUIRED


def admit(root: Path) -> Json:
    """Bind original qualification and current methods without rerunning historical controllers."""
    values, refs, changed = [], [], []
    for name, digest in PINS.items():
        ref = dict(path=str(root / name), sha256=digest)
        value = read(ref)
        for key, expected in dict(required_checks_passed=True, flagged_adversarial=False).items():
            require(root / name, key, expected, value.get(key))
        terminal = Path(value["terminal_validation_sidecar_path"])
        sidecar = Path(json.loads(terminal.read_bytes())["publication"]["sidecar_path"])
        require(
            root / name,
            "bound_publication",
            True,
            read_bound_sidecar(root / name, sidecar)["report"]["passed"],
        )
        refs.extend([ref, reference(terminal), reference(sidecar)])
        values.append(value)
    methods, qualification = values
    for key in ["resident_inputs_ready_score", "methods_ready_score"]:
        require(root, key, 1, methods.get(key))
    require(
        root,
        "resident_inference_ready_score",
        1,
        qualification.get("resident_inference_ready_score"),
    )
    bundle = read(qualification["primitive_reference"])["bundle"]
    protocol = read(methods["methods_protocol_reference"])
    require(root, "registered_resident_protocol", bundle["protocol"], protocol["resident_session"])
    for ref in qualification["code_config_hashes"]:
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            changed.append(ref["path"])
    # Changed dependencies cannot inherit the old parity qualification.
    require(root, "unchanged_qualified_dependencies", [], changed)
    for ref in qualification["source_artifact_hashes"]:
        require(
            Path(ref["path"]),
            "original_operand_hash",
            ref["sha256"],
            sha256_file(Path(ref["path"])),
        )
    refs.extend(qualification["source_artifact_hashes"] + qualification["code_config_hashes"])
    refs.extend([qualification["primitive_reference"], methods["methods_protocol_reference"]])
    return dict(
        bundle,
        refs=refs,
        dependencies_changed=changed,
        preserved_quarantines=methods["preserved_quarantines"],
        methods_protocol=protocol["resident_session"],
    )


def roster() -> list[Json]:
    """Rotate all four arms while alternating the Python/native order within lifecycle pairs."""
    rows: list[Json] = []
    for length in [1, 4, 16]:
        for size in [1, 8]:
            for repeat in range(10):
                arms = ARMS[2:] + ARMS[:2] if repeat % 4 >= 2 else ARMS[:]
                if repeat % 2:
                    arms = [arms[1], arms[0], arms[3], arms[2]]
                rows.extend(dict(length=length, batch=size, repeat=repeat, arm=arm) for arm in arms)
    return rows


def session(bundle: Json, slot: Json, raw: Path, *, deadline: float | None = None) -> Json:
    """Charge config persistence, launch and exit; expose each complete request and disjoint operation."""
    began = time.monotonic_ns()
    if deadline is not None and time.monotonic() + 40 >= deadline:
        raise TimeoutError("measurement_deadline_before_launch")
    config = dict(head=bundle["head"], extension=bundle["extension"])
    with patch.object(peer, "CLI", str(ROOT / CLI)):
        process = peer.Session(config, raw, slot["arm"].split("_")[-1])
    observations: list[Json] = []
    operations: list[Json] = []
    panels = 1 if slot["arm"].startswith("cold") else slot["length"]
    try:
        for panel in range(panels):
            for index in range(0, 64, slot["batch"]):
                if deadline is not None and time.monotonic() + 40 >= deadline:
                    raise TimeoutError("measurement_deadline_reserved_shutdown")
                rows = bundle["requests"][index : index + slot["batch"]]
                response = process.request(dict(rows=rows), f"{panel}-{index}")
                if response["status"] != "ok":
                    raise ValueError("worker_response")
                observed = process.observations[-1]
                previous = operations[-1]["end_ns"] if operations else process.ready_ns
                first = observed["start_ns"]
                operations.append(
                    dict(
                        operation="transport_parse_and_parent",
                        start_ns=previous,
                        end_ns=response["spans"][0]["start_ns"],
                    )
                )
                operations.extend(response["spans"])
                operations.append(
                    dict(
                        operation="serialization_transfer",
                        start_ns=response["spans"][-1]["end_ns"],
                        end_ns=observed["end_ns"],
                    )
                )
                latency = (observed["end_ns"] - (began if not observations else first)) / 1e9
                for row, answer in zip(rows, response["responses"], strict=True):
                    observations.append(
                        dict(
                            unit_id=row["unit_id"],
                            panel=panel,
                            request_index=index,
                            response=json.loads(answer["response"]),
                            latency_s=latency,
                        )
                    )
            progress("8584_session_panel", panel + 1, panels - panel - 1)
    except (OSError, ValueError, KeyError, TypeError, TimeoutError):
        if process.process.poll() is None:
            process.process.kill()
        atomic_json(raw / "censored_session.json", process.close(shutdown=False))
        raise
    receipt = process.close()
    operations.insert(
        0, dict(operation="immutable_setup_launch_imports", start_ns=began, end_ns=process.ready_ns)
    )
    operations.append(
        dict(operation="shutdown", start_ns=operations[-1]["end_ns"], end_ns=receipt["end_ns"])
    )
    for operation in operations:
        operation["duration_s"] = (operation["end_ns"] - operation["start_ns"]) / 1e9
    return dict(
        slot,
        start_ns=began,
        end_ns=receipt["end_ns"],
        duration_s=(receipt["end_ns"] - began) / 1e9,
        startup_s=(process.ready_ns - began) / 1e9,
        shutdown_s=operations[-1]["duration_s"],
        peak_rss_kib=receipt["shutdown"]["peak_rss_kib"],
        requests=observations,
        operations=operations,
        exit_code=receipt["exit_code"],
        lifecycle=receipt,
    )


def empty_work(raw: Path) -> Json:
    """Retain an invocation even when admission prevents all measurements."""
    return dict(
        raw=str(raw),
        started_ns=time.monotonic_ns(),
        ended_ns=time.monotonic_ns(),
        bundle=None,
        rows=[],
        gate_check_summary=[],
        preconditions_checked={},
        phase_spans=[],
    )


def measure(root: Path, raw: Path) -> Json:
    """Checkpoint each paired arm and stop at the fixed budget without dropping cold controls."""
    work = empty_work(raw)
    start = time.monotonic()
    try:
        with TemporaryDirectory(prefix="exp8584-", dir="/var/tmp") as folder:
            scratch = Path(folder)
            (scratch / "probe").write_text("disk-backed writable scratch")
            filesystem = child(
                "filesystem",
                ["findmnt", "-n", "-o", "FSTYPE", "-T", folder],
                raw / "preflight",
                deadline=10,
            )
            memory = (
                int(
                    next(
                        s.split()[1]
                        for s in Path("/proc/meminfo").read_text().splitlines()
                        if s.startswith("MemAvailable:")
                    )
                )
                * 1024
            )
            require(
                scratch,
                "disk_backed_scratch",
                True,
                filesystem["passed"]
                and Path(filesystem["stdout_path"]).read_text().strip() not in {"tmpfs", "ramfs"},
            )
            require(scratch, "memory_256MiB", True, memory >= 256 * 1024**2)
            require(scratch, "disk_64MiB", True, shutil.disk_usage(scratch).free >= 64 * 1024**2)
            for tool in ["python", "pytest", "coverage", "ruff", "mypy"]:
                require(
                    root / ".venv/bin" / tool,
                    "tool_" + tool,
                    True,
                    (root / ".venv/bin" / tool).is_file(),
                )
            work["preconditions_checked"] = dict(
                memory_available_bytes=memory,
                filesystem=filesystem,
                writable_disk_scratch=folder,
                measurement_budget_s=2400,
                measurement_stop_s=3600,
                validation_reserve_s=900,
                total_cap_s=4800,
                optional_cuda_required=False,
                optional_board_required=False,
            )
        progress("8584_authenticate")
        work["bundle"] = admit(root)
        progress("8584_benchmark_start")
        for index, slot in enumerate(roster()):
            progress("8584_before_benchmark", index, 240 - index)
            row = dict(
                slot,
                unit_id=f"L{slot['length']}-B{slot['batch']}-R{slot['repeat']}-{slot['arm']}",
                primitive_reference=None,
                missing_reason=None,
                status="completed",
            )
            sessions = []
            if time.monotonic() - start >= 2400:
                row.update(status="censored", missing_reason="fixed_measurement_budget")
            else:
                try:
                    count = slot["length"] if slot["arm"].startswith("cold") else 1
                    for panel in range(count):
                        if time.monotonic() - start >= 2400:
                            raise TimeoutError("fixed_measurement_budget")
                        sessions.append(
                            session(
                                work["bundle"],
                                slot,
                                raw / "sessions" / row["unit_id"] / str(panel),
                                deadline=start + 2400,
                            )
                        )
                    target = raw / "primitives" / (row["unit_id"] + ".json")
                    atomic_json(target, dict(slot=slot, sessions=sessions))
                    row["primitive_reference"] = reference(target)
                except (OSError, ValueError, KeyError, TimeoutError) as error:
                    target = raw / "primitives" / (row["unit_id"] + ".json")
                    atomic_json(target, dict(slot=slot, sessions=sessions, failure=str(error)))
                    row.update(
                        status="censored" if isinstance(error, TimeoutError) else "failed",
                        missing_reason=str(error),
                        primitive_reference=reference(target),
                    )
            work["rows"].append(row)
            work["ended_ns"] = time.monotonic_ns()
            atomic_json(raw / "measurement.json", work)
            if index == 3:
                elapsed = time.monotonic() - start
                work["first_cell_forecast"] = dict(
                    observed_s=elapsed, projected_total_s=elapsed * 240, fixed_design_unchanged=True
                )
            progress("8584_after_benchmark", index + 1, 239 - index)
    except (GateFailure, OSError, ValueError, KeyError) as error:
        work["gate_check_summary"].append(
            error.row
            if isinstance(error, GateFailure)
            else GateFailure(root, "available_original_inputs", True, str(error)).row
        )
    work["ended_ns"] = time.monotonic_ns()
    work["phase_spans"] = [
        dict(
            phase="preflight_and_measurement", start_ns=work["started_ns"], end_ns=work["ended_ns"]
        )
    ]
    atomic_json(raw / "measurement.json", work)
    return work


def interval(python_s: list[float], native_s: list[float]) -> list[float]:
    """Resample paired repeat indices; repeated requests stay dependent within their repeat."""
    indices = np.random.default_rng(7368573).integers(0, len(python_s), size=(10000, len(python_s)))
    ratios = np.asarray(python_s)[indices].sum(axis=1) / np.asarray(native_s)[indices].sum(axis=1)
    return [float(x) for x in np.quantile(ratios, [0.025, 0.975])]


def benefit(ci: list[float] | None, p95: float | None, clean: bool) -> tuple[int, int]:
    """Readiness and timing benefit use separate gates; tenfold speed has its own threshold."""
    qualified = clean and ci is not None and p95 is not None and p95 <= 1.05
    return int(qualified and ci is not None and ci[0] > 1), int(
        qualified and ci is not None and ci[0] >= 10
    )


def reduce(work: Json) -> Json:
    """Reconstruct summaries from original response and clock rows rather than trusting counters."""
    requests: list[Json] = []
    pairs: list[Json] = []
    operations: list[Json] = []
    startup: list[Json] = []
    failures: list[Json] = []
    bundle = work["bundle"]
    require(
        ROOT,
        "canonical_roster_prefix",
        roster()[: len(work["rows"])],
        [{k: r[k] for k in ["length", "batch", "repeat", "arm"]} for r in work["rows"]],
    )
    require(ROOT, "unique_cell_ids", len(work["rows"]), len({r["unit_id"] for r in work["rows"]}))
    references = (
        {}
        if bundle is None
        else {
            r["unit_id"]: json.loads(numeric.request(bundle["head"], r)) for r in bundle["requests"]
        }
    )
    for row in work["rows"]:
        if row["status"] != "completed":
            failures.append(row)
            continue
        primitive = read(row["primitive_reference"])
        require(
            ROOT,
            "process_session_count",
            row["length"] if row["arm"].startswith("cold") else 1,
            len(primitive["sessions"]),
        )
        require(
            ROOT,
            "slot_identity",
            {k: row[k] for k in ["length", "batch", "repeat", "arm"]},
            primitive["slot"],
        )
        total, latencies, rss, mismatches = 0.0, [], 0, 0
        for index, process in enumerate(primitive["sessions"]):
            require(ROOT, "child_exit", 0, process["exit_code"])
            spans = process["operations"]
            require(
                ROOT,
                "disjoint_clocks",
                True,
                all(a["end_ns"] == b["start_ns"] for a, b in zip(spans, spans[1:]))
                and all(s["start_ns"] <= s["end_ns"] for s in spans),
            )
            require(
                ROOT,
                "complete_cost",
                process["duration_s"],
                (process["end_ns"] - process["start_ns"]) / 1e9,
            )
            require(
                ROOT,
                "clock_partition",
                True,
                abs(sum(s["duration_s"] for s in spans) - process["duration_s"]) < 1e-8,
            )
            lifecycle = process["lifecycle"]
            config = read(dict(path=lifecycle["config_path"], sha256=lifecycle["config_hash"]))
            require(
                ROOT,
                "exact_configuration",
                dict(head=bundle["head"], extension=bundle["extension"]),
                config,
            )
            require(ROOT, "lifecycle_exit", 0, lifecycle["exit_code"])
            require(ROOT, "lifecycle_end", process["end_ns"], lifecycle["end_ns"])
            native = row["arm"].endswith("native")
            require(
                ROOT,
                "loaded_extension",
                {k: bundle["extension"][k] for k in ["path", "sha256"]} if native else None,
                lifecycle["ready"]["loaded_extension"],
            )
            require(ROOT, "one_immutable_configuration_load", 1, lifecycle["ready"]["config_loads"])
            rebuilt_requests: list[Json] = []
            rebuilt_spans: list[Json] = [
                dict(
                    operation="immutable_setup_launch_imports",
                    start_ns=process["start_ns"],
                    end_ns=lifecycle["ready_ns"],
                )
            ]
            panels = 1 if row["arm"].startswith("cold") else row["length"]
            require(
                ROOT,
                "observed_batches",
                panels * 64 // row["batch"],
                len(lifecycle["observations"]),
            )
            for batch_index, observation in enumerate(lifecycle["observations"]):
                panel, offset = divmod(batch_index, 64 // row["batch"])
                request_index = offset * row["batch"]
                expected_rows = bundle["requests"][request_index : request_index + row["batch"]]
                require(
                    ROOT,
                    "uncached_original_request_rows",
                    expected_rows,
                    observation["request"]["row"]["rows"],
                )
                require(
                    ROOT,
                    "response_identity",
                    observation["request"]["request_id"],
                    observation["response"]["request_id"],
                )
                require(
                    ROOT,
                    "request_clocks",
                    True,
                    lifecycle["ready_ns"]
                    <= observation["start_ns"]
                    <= observation["response"]["spans"][0]["start_ns"]
                    <= observation["response"]["spans"][-1]["end_ns"]
                    <= observation["end_ns"]
                    <= lifecycle["end_ns"],
                )
                rebuilt_spans.append(
                    dict(
                        operation="transport_parse_and_parent",
                        start_ns=rebuilt_spans[-1]["end_ns"],
                        end_ns=observation["response"]["spans"][0]["start_ns"],
                    )
                )
                rebuilt_spans.extend(dict(s) for s in observation["response"]["spans"])
                rebuilt_spans.append(
                    dict(
                        operation="serialization_transfer",
                        start_ns=rebuilt_spans[-1]["end_ns"],
                        end_ns=observation["end_ns"],
                    )
                )
                first = process["start_ns"] if not rebuilt_requests else observation["start_ns"]
                for source, answer in zip(
                    expected_rows, observation["response"]["responses"], strict=True
                ):
                    rebuilt_requests.append(
                        dict(
                            unit_id=source["unit_id"],
                            panel=panel,
                            request_index=request_index,
                            response=json.loads(answer["response"]),
                            latency_s=(observation["end_ns"] - first) / 1e9,
                        )
                    )
            # Frame clocks supersede the preliminary per-answer first-batch latency summaries.
            require(
                ROOT,
                "primitive_response_reconstruction",
                [{k: v for k, v in r.items() if k != "latency_s"} for r in rebuilt_requests],
                [{k: v for k, v in r.items() if k != "latency_s"} for r in process["requests"]],
            )
            rebuilt_spans.append(
                dict(
                    operation="shutdown",
                    start_ns=rebuilt_spans[-1]["end_ns"],
                    end_ns=lifecycle["end_ns"],
                )
            )
            for span in rebuilt_spans:
                span["duration_s"] = (span["end_ns"] - span["start_ns"]) / 1e9
            require(ROOT, "operation_clock_reconstruction", rebuilt_spans, spans)
            require(
                ROOT,
                "stderr_bytes",
                lifecycle["stderr_sha256"],
                sha256_file(Path(lifecycle["stderr_path"])),
            )
            for observed in rebuilt_requests:
                ref = references[observed["unit_id"]]
                response = observed["response"]
                passed = (
                    abs(response["probability"] - ref["probability"])
                    <= 1e-12 * (1 + abs(ref["probability"]))
                    and response["decision"] == ref["decision"]
                )
                mismatches += int(not passed)
                request = dict(
                    observed,
                    status="completed",
                    intended_count=1,
                    completed_count=1,
                    failed_count=0,
                    censored_count=0,
                    arm=row["arm"],
                    length=row["length"],
                    batch=row["batch"],
                    repeat=row["repeat"],
                    session_index=index,
                    passed=passed,
                    missing_reason=None if passed else "decision_or_probability_mismatch",
                )
                requests.append(request)
                latencies.append(observed["latency_s"])
            total += process["duration_s"]
            rss = max(rss, process["peak_rss_kib"])
            for operation in sorted({s["operation"] for s in spans}):
                seconds = sum(s["duration_s"] for s in spans if s["operation"] == operation)
                operations.append(
                    dict(
                        operation=operation,
                        unit_id=row["unit_id"],
                        session_index=index,
                        duration_s=seconds,
                        share=seconds / process["duration_s"],
                    )
                )
            startup.append(
                dict(
                    unit_id=row["unit_id"],
                    session_index=index,
                    startup_s=process["startup_s"],
                    shutdown_s=process["shutdown_s"],
                    peak_rss_kib=process["peak_rss_kib"],
                )
            )
        require(ROOT, "full_request_count", 64 * row["length"], len(latencies))
        pairs.append(
            dict(
                row,
                intended_count=64 * row["length"],
                completed_count=len(latencies),
                failed_count=0,
                censored_count=0,
                total_session_wall_s=total,
                throughput_answers_s=len(latencies) / total,
                p95_request_latency_s=float(np.quantile(latencies, 0.95)),
                decision_mismatches=mismatches,
                peak_rss_kib=rss,
            )
        )
    contrasts = []
    for length in [1, 4, 16]:
        for size in [1, 8]:
            for label, a, b in [
                ("native", "resident_python", "resident_native"),
                ("lifecycle", "cold_python", "resident_python"),
            ]:
                left = sorted(
                    [r for r in pairs if (r["length"], r["batch"], r["arm"]) == (length, size, a)],
                    key=lambda r: r["repeat"],
                )
                right = sorted(
                    [r for r in pairs if (r["length"], r["batch"], r["arm"]) == (length, size, b)],
                    key=lambda r: r["repeat"],
                )
                full = len(left) == len(right) == 10 and [r["repeat"] for r in left] == [
                    r["repeat"] for r in right
                ] == list(range(10))
                ci = (
                    interval(
                        [r["total_session_wall_s"] for r in left],
                        [r["total_session_wall_s"] for r in right],
                    )
                    if full
                    else None
                )
                latency = (
                    float(
                        np.quantile(
                            [
                                r["latency_s"]
                                for r in requests
                                if (r["length"], r["batch"], r["arm"]) == (length, size, b)
                            ],
                            0.95,
                        )
                        / np.quantile(
                            [
                                r["latency_s"]
                                for r in requests
                                if (r["length"], r["batch"], r["arm"]) == (length, size, a)
                            ],
                            0.95,
                        )
                    )
                    if full
                    else None
                )
                contrasts.append(
                    dict(
                        contrast=label,
                        length=length,
                        batch=size,
                        baseline=a,
                        treatment=b,
                        throughput_ratio_ci95=ci,
                        p95_request_latency_ratio=latency,
                        paired_repeats=10 if full else 0,
                        full_parity=full
                        and all(r["decision_mismatches"] == 0 for r in left + right),
                    )
                )
    return dict(
        per_request_rows=requests,
        paired_session_rows=pairs,
        operation_cost_rows=operations,
        startup_shutdown_cost=startup,
        censoring_rows=failures,
        throughput_intervals=contrasts,
    )


def receipt_bytes(work: Json, receipts: list[Json]) -> bool:
    """Passing claims need their actual logs; only the two derived coverage/union gates are scalars."""
    try:
        for receipt in receipts:
            if "argv" not in receipt:
                if receipt["name"] == "exact_disjoint_node_union":
                    union = work["exact_union_receipt"]
                    require(
                        ROOT,
                        "exact_node_union",
                        True,
                        union["passed"]
                        and bool(union["collected"])
                        and len(union["collected"]) == len(set(union["collected"]))
                        and sorted(union["collected"]) == sorted(union["passed_node_ids"]),
                    )
                elif receipt["name"] == "owned_python_coverage_100":
                    totals = work["python_coverage"]["totals"]
                    require(
                        ROOT,
                        "owned_coverage_total",
                        True,
                        totals["percent_covered"] == 100
                        and totals["missing_lines"] == 0
                        and totals["covered_lines"] == totals["num_statements"] > 0,
                    )
                else:
                    return False
            else:
                for stream in ["stdout", "stderr"]:
                    require(
                        ROOT,
                        "validation_log_bytes",
                        receipt[stream + "_sha256"],
                        sha256_file(Path(receipt[stream + "_path"])),
                    )
                require(
                    ROOT,
                    "normal_validation_exit",
                    True,
                    not receipt["timed_out"]
                    and receipt["exit_code"] == receipt["expected_exit"]
                    and receipt["passed"],
                )
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def artifact(work: Json, receipts: list[Json], output: Path) -> Json:
    """Qualification remains one for a complete null; missing owned checks forbid benefit."""
    reduced = reduce(work)
    request_path = Path(work["raw"]) / "per_request_rows.json"
    if not request_path.exists():
        atomic_json(request_path, reduced["per_request_rows"])
    require(
        request_path,
        "request_reduction",
        reduced.pop("per_request_rows"),
        json.loads(request_path.read_bytes()),
    )
    primary = next(
        r
        for r in reduced["throughput_intervals"]
        if (r["contrast"], r["length"], r["batch"]) == ("native", 16, 1)
    )
    lifecycle = next(
        r
        for r in reduced["throughput_intervals"]
        if (r["contrast"], r["length"], r["batch"]) == ("lifecycle", 16, 1)
    )
    checks = {r["name"]: r["passed"] for r in receipts}
    required = REQUIRED <= checks.keys() and all(checks.values()) and receipt_bytes(work, receipts)
    clean = len(reduced["paired_session_rows"]) == 240 and all(
        r["decision_mismatches"] == 0 for r in reduced["paired_session_rows"]
    )
    ready = (
        clean
        and required
        and not work["gate_check_summary"]
        and not work.get("adversarial_findings")
    )
    native, nfr = benefit(
        primary["throughput_ratio_ci95"],
        primary["p95_request_latency_ratio"],
        ready and primary["full_parity"],
    )
    life, _ = benefit(
        lifecycle["throughput_ratio_ci95"],
        lifecycle["p95_request_latency_ratio"],
        ready and lifecycle["full_parity"],
    )
    verdict = (
        "disqualified"
        if any(not r["passed"] for r in receipts)
        or (not required and not work["gate_check_summary"])
        else "blocked"
        if work["gate_check_summary"]
        else "circular_positive"
        if ready and native
        else "null"
        if ready
        else "disqualified"
    )
    bundle = work["bundle"]
    value: Json = dict(
        experiment_id=8584,
        task_id=TASK,
        milestone="2026.10.737",
        run_date="20261016",
        honest_verdict=f"complete_{verdict}_v737_resident_complete_cost",
        verdict_class=verdict,
        gate_check_summary=work["gate_check_summary"]
        + [
            dict(
                check=r["name"],
                upstream=TASK,
                path=r.get("stdout_path"),
                sha256=r.get("stdout_sha256"),
                field="passed",
                operator="==",
                expected=True,
                observed=False,
            )
            for r in receipts
            if not r["passed"]
        ],
        inference_substrate="CPU complete resident and cold process measurements",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=ZERO_INVOCATION_COUNTS,
        current_llm_invocations=0,
        historical_model_provenance=bundle["historical_model_provenance"] if bundle else [],
        rows=reduced["paired_session_rows"] + reduced["censoring_rows"],
        intended_count=240,
        completed_count=len(reduced["paired_session_rows"]),
        failed_count=sum(r["status"] == "failed" for r in work["rows"]),
        censored_count=sum(r["status"] == "censored" for r in work["rows"])
        + (240 - len(work["rows"])),
        excluded_count=0,
        independent_count=int(bool(reduced["paired_session_rows"])),
        sample_size_budget=dict(
            hosts=1,
            paired_repeats=10,
            arms=4,
            batches=[1, 8],
            session_lengths_panels=[1, 4, 16],
            requests_per_panel=64,
            intended_arm_repeat_cells=240,
            intended_process_sessions=960,
            intended_answer_observations=107520,
            bootstrap_draws=10000,
            bootstrap_unit="paired repeat index; one host",
        ),
        verifier_is_oracle=True,
        exposure_scope="original exposed stable-order test answers; repeated on one host",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=required,
        flagged_adversarial=bool(work.get("adversarial_findings")) or verdict == "disqualified",
        acceptance_gates=dict(
            original_inputs=bundle is not None,
            complete_measurement=clean,
            required_validation=required,
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal_validation.json"),
        adversarial_findings=work.get("adversarial_findings", []),
        preconditions_checked=work["preconditions_checked"],
        duration_s=(work["ended_ns"] - work["started_ns"]) / 1e9,
        phase_spans=work["phase_spans"],
        random_seed=7368573,
        source_artifact_hashes=bundle["refs"] if bundle else [],
        code_config_hashes=[reference(ROOT / p) for p in OWNED + [TEST]],
        raw_shard_hashes=[
            r["primitive_reference"] for r in work["rows"] if r["primitive_reference"]
        ],
        cited_upstream_artifacts=bundle["refs"] if bundle else [],
        resident_cost_ready_score=int(ready),
        resident_native_benefit_score=native,
        lifecycle_benefit_score=life,
        nfr01_score=nfr,
        per_request_rows_reference=reference(request_path),
        primary_contrast=primary,
        lifecycle_contrast=lifecycle,
        historical_cost_null=bundle["historical_cost_null"] if bundle else None,
        preserved_quarantines=bundle["preserved_quarantines"] if bundle else [],
        methodology_note="Ten paired repeats on one host are timing evidence, not independent machines. Lifecycle benefit is separate from native arithmetic. No warmup or answer/feature cache.",
        work_reference=reference(Path(work["raw"]) / "measurement.json"),
        publication_output=str(output),
        python_coverage=work.get("python_coverage"),
        exact_union_receipt=work.get("exact_union_receipt"),
        first_cell_forecast=work.get("first_cell_forecast"),
        measurement_code_config_hashes=work.get("measurement_code_config_hashes", []),
        measurement_reference=work.get("measurement_reference"),
        latency_reconstruction="All answers in the first frame share launch-through-response latency; immutable frame endpoints supersede preliminary scalar summaries.",
        **reduced,
    )
    value["field_principles"] = {
        key: "Bind exact invocation evidence; preserve complete costs, missing units and separate qualification from benefit."
        for key in value
    }
    value["field_principles"].update(
        dict(
            resident_cost_ready_score="One requires every registered cell, zero parity differences and every owned acceptance check; a qualified null stays ready.",
            resident_native_benefit_score="Native effect only: resident_native versus resident_python at length16/batch1; lower95 throughput ratio>1 and p95 latency ratio<=1.05.",
            lifecycle_benefit_score="Process lifetime only: resident_python versus cold_python at length16/batch1. This cannot establish a Rust gain.",
            nfr01_score="Primary native throughput-ratio lower95>=10 with the same latency and parity gates.",
            per_request_rows_reference="All attempts retain stable answer identity, frame and repeat; first-frame latency includes startup for every answer.",
            paired_session_rows="Per arm and paired repeat: full launch-through-exit session walls, inclusive throughput and peak RSS, with exact answer counts.",
            primary_contrast="Prespecified resident native effect at sixteen panels and batch one, independent of the lifecycle comparison.",
            lifecycle_contrast="Same-work Python lifecycle comparison; cold controls launch once for each64-answer panel.",
            operation_cost_rows="Disjoint clock intervals aggregate within each process; their shares sum to complete session wall time.",
            throughput_intervals="10000 paired-repeat bootstrap draws with seed7368573; ten repeats on one host are timing evidence, not ten machines.",
            startup_shutdown_cost="Immutable config persistence, launch and imports occur once per process; every shutdown and peak child RSS is retained.",
            censoring_rows="Incomplete fixed cells retain their reason; no complete comparison can be inferred from censored cells.",
            historical_cost_null="Original Exp8559 cold-boundary null remains historical authority; resident timing cannot revise it.",
            measurement_code_config_hashes="Immutable source snapshots record the actual timing implementation separately from the final reducer and validation source.",
            gate_check_summary="Failed checks retain original producer, byte identity, field, operator and both operands; absence is distinct from zero.",
            acceptance_gates="Input authentication, complete fixed measurements and earned owned validation qualify readiness independently of benefit.",
        )
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Rebuild all derived fields in a fresh process; repaired envelope hashes grant no authority."""
    try:
        value = json.loads(path.read_bytes())
        require(path, "checksum", checksum(value), value["reproducibility_checksum"])
        work = read(value["work_reference"])
        if work["bundle"] is not None:
            require(path, "original_admitted_bundle", admit(ROOT), work["bundle"])
        timing_refs = list(value["measurement_code_config_hashes"])
        if value.get("measurement_reference"):
            timing_refs.append(value["measurement_reference"])
        for ref in value["code_config_hashes"] + value["source_artifact_hashes"] + timing_refs:
            require(
                Path(ref["path"]), "source_bytes", ref["sha256"], sha256_file(Path(ref["path"]))
            )
        rebuilt = artifact(work, value["validation_receipts"], Path(value["publication_output"]))
        return bool(value == rebuilt)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def consume(path: Path) -> Json:
    """The downstream hardware boundary requires complete qualified cost evidence."""
    value = json.loads(path.read_bytes())
    if not replay(path) or value["resident_cost_ready_score"] != 1:
        raise ValueError("resident_cost_not_ready")
    return dict(
        passed=True,
        path=str(path),
        sha256=sha256_file(path),
        primary_contrast=value["primary_contrast"],
        lifecycle_contrast=value["lifecycle_contrast"],
    )

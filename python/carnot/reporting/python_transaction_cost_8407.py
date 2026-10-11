"""REQ-REPORT-8407: source-bound local timings do not reopen closed utility results."""

from __future__ import annotations

from copy import deepcopy
import json
from importlib.metadata import version
import os
from pathlib import Path
import shutil
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import continuous_direct_learning_8406 as upstream
from carnot.reporting import historical_consumer_roots_8402 as authority
from carnot.reporting import direct_state_qualification_8404 as qualified
from carnot.reporting.v721_capstone_evidence import frozen_inputs
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    ZERO_INVOCATION_COUNTS,
)
from carnot.verify import python_transaction_cost_8407 as k

Json = dict[str, Any]
ROOT = authority.ROOT
NAME, TASK = "experiment_8407_v724_python_transaction_cost", "exp8407-python-transaction-cost"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_python_transaction_cost_8407.py"
OWNED = [
    "python/carnot/verify/python_transaction_cost_8407.py",
    "python/carnot/reporting/python_transaction_cost_8407.py",
    "python/carnot/reporting/python_transaction_runner_8407.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
reference, progress = authority.reference, k.progress


def authenticate(root: Path, raw: Path, work: Json) -> Json:
    """Reuse exact direct-producer and complete-task authentication without importing its timing."""
    with patch.object(upstream, "TASK", TASK):
        return dict(upstream.authenticate(root, raw, work))


def measure(root: Path, raw: Path, private: Path) -> Json:
    """Check sources and host resources before admitting the frozen cost panel."""
    began = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    work: Json = dict(
        root=str(root),
        raw=str(raw),
        refs=[],
        gate_check_summary=[],
        bundle={},
        results=[],
        historical_model_provenance=[],
        historical_controls={},
        current_authority={},
    )
    memory = int(Path("/proc/meminfo").read_text().split("MemAvailable:")[1].split()[0]) * 1024
    probe = private / "disk_probe"
    probe.write_bytes(b"private disk scratch")
    mounts = [line.split(" - ") for line in Path("/proc/self/mountinfo").read_text().splitlines()]
    mounted = max(
        (line for line in mounts if private.resolve().is_relative_to(Path(line[0].split()[4]))),
        key=lambda line: len(line[0].split()[4]),
    )
    filesystem_type = mounted[1].split()[0]
    resources = dict(
        private_scratch=probe.read_bytes() == b"private disk scratch",
        private_permissions=private.stat().st_mode & 0o077 == 0,
        disk_backed=filesystem_type not in ("tmpfs", "ramfs"),
        filesystem_type=filesystem_type,
        tool_versions={
            name: version(name) for name in ("numpy", "scipy", "coverage", "pytest", "ruff", "mypy")
        },
        tool_hashes=[
            reference(ROOT / ".venv/bin" / name)
            for name in ("python", "pytest", "coverage", "ruff", "mypy")
        ],
        disk_free=shutil.disk_usage(private).free > 2_000_000_000,
        memory_available=memory > 256_000_000,
        task_cap_s=4800,
        filesystem=str(private),
        available_memory_bytes=memory,
    )
    work["preconditions_checked"] = resources
    progress("preconditions_before")
    for name in ("python", "pytest", "coverage", "ruff", "mypy"):
        if not os.access(ROOT / ".venv/bin" / name, os.X_OK):
            work["gate_check_summary"].append(
                authority.failure(ROOT / ".venv/bin" / name, "executable_available", True, None)
            )
    for name in (
        "private_scratch",
        "private_permissions",
        "disk_backed",
        "disk_free",
        "memory_available",
    ):
        if not resources[name]:
            work["gate_check_summary"].append(
                authority.failure(private, name, True, resources[name])
            )
    for name, pin in upstream.PINS.items():
        if not (root / name).is_file():
            work["gate_check_summary"].append(
                authority.failure(root / name, "input_sha256", pin, None)
            )
    try:
        work["bundle"] = authenticate(root, raw, work)
    except (OSError, ValueError, KeyError, TypeError, IndexError, StopIteration) as error:
        work["gate_check_summary"].append(
            authority.failure(root / authority.DESIGN, "authenticated_operands", True, str(error))
        )
    progress("preconditions_after")
    if not work["gate_check_summary"]:
        for index, cell in enumerate(k.CELLS):
            progress("benchmark_before_cell", index, 27 - index)
            for repeat in range(6):
                destination = raw / "repeats" / f"{index:02}-{repeat}"
                progress(
                    "benchmark_before_repeat",
                    index * 5 + max(0, repeat - 1),
                    135 - index * 5 - max(0, repeat - 1),
                )
                result = k.run_repeat(work["bundle"], cell, repeat, destination)
                k.reduce_repeat(work["bundle"], result)
                path = destination / "primitive.json"
                atomic_json(path, result)
                work["results"].append(dict(reference(path), warmup=repeat == 0))
                progress("benchmark_after_repeat", index * 5 + repeat, 135 - index * 5 - repeat)
            progress("benchmark_after_cell", index + 1, 26 - index)
    work.update(
        duration_s=(time.monotonic_ns() - began) / 1e9,
        phase_spans=[
            dict(phase="preconditions_and_benchmark", start_ns=began, end_ns=time.monotonic_ns())
        ],
        work_path=str(raw / "work.json"),
    )
    atomic_json(Path(work["work_path"]), work)
    return work


def reduction(work: Json) -> Json:
    """All measured repeats and failures stay in the same complete wall denominator."""
    repeats, rows, refs = [], [], []
    for index, ref in enumerate(work["results"]):
        if index % 12 == 0:
            progress("independent_cost_reduction", index, len(work["results"]) - index)
        path = authority.checked(Path(ref["path"]), ref["sha256"])
        result = json.loads(path.read_bytes())
        if ref["warmup"] != (result["repeat"] == 0):
            raise ValueError("discarded_warmup_binding")
        summary = k.reduce_repeat(work["bundle"], result)
        refs.append(reference(path))
        if not ref["warmup"]:
            repeats.append(summary)
            rows.extend(result["rows"])
    refs.extend(work.get("coverage_shards", []))
    for ref in work.get("coverage_shards", []):
        authority.checked(Path(ref["path"]), ref["sha256"])
    cells: list[Json] = []
    for cell in k.CELLS:
        group = [r for r in repeats if r["cell"] == cell]
        if group:
            interval = k.block_interval(group)
            cells.append(
                dict(cell=cell, throughput=interval, repeat_ids=[r["repeat_id"] for r in group])
            )
        else:
            cells.append(dict(cell=cell, throughput=None, missing_reason="external_precondition"))
            for repeat in range(1, 6):
                repeats.append(
                    dict(
                        cell=cell,
                        repeat=repeat,
                        repeat_id=f"{cell}-rep{repeat}",
                        status="unstarted",
                        missing_reason="external_precondition",
                    )
                )
    if not rows:
        for cell in k.CELLS:
            for repeat in range(1, 6):
                for index, event in enumerate(authority.cost_tape(cell["predictions_per_update"])):
                    rows.append(
                        dict(
                            event,
                            request_id=f"{cell}-rep{repeat}-q{index}",
                            status="unstarted",
                            missing_reason="external_precondition",
                            cell=cell,
                        )
                    )
    completed = sum(r["status"] in ("completed", "nonapplicable") for r in rows)
    failed = sum(r["status"] == "failed" for r in rows)
    return dict(
        rows=rows,
        repeat_rows=repeats,
        workload_cells=cells,
        raw_shard_hashes=refs,
        intended_count=27360,
        completed_count=completed,
        failed_count=failed,
        censored_count=sum(r["status"] == "unstarted" for r in rows),
        excluded_count=0,
        operation_counts=dict(
            prediction_attempts=sum(
                r["kind"] == "prediction" and r["status"] != "unstarted" for r in rows
            ),
            feedback_attempts=sum(
                r["kind"] == "feedback" and r["status"] != "unstarted" for r in rows
            ),
            coefficient_updates=sum(bool(r.get("changed")) for r in rows),
        ),
        feedback_conservation=dict(
            issued=sum(r.get("feedback_attempts", 0) for r in repeats),
            released=sum(r.get("released", 0) for r in repeats),
            applied=sum(r.get("applied", 0) for r in repeats),
            missing=sum(r.get("missing", 0) for r in repeats),
            pending=0,
            lost=0,
        ),
        tail_drain_seconds=sum(r.get("tail_drain_seconds", 0.0) for r in repeats),
        serialized_bytes=sum(r.get("serialized_bytes", 0) for r in repeats),
        fsync_count=sum(r.get("fsync_count", 0) for r in repeats),
    )


def build(work: Json, receipts: list[Json]) -> Json:
    """Readiness requires complete work, five blocks per cell and every owned check."""
    reduced = reduction(work)
    passed = bool(receipts) and all(r["passed"] for r in receipts)
    with patch.object(qualified, "OWNED", OWNED):
        covered = bool(qualified.coverage_ready(work.get("owned_coverage", {})))
    complete = (
        len(work["results"]) == 162
        and reduced["completed_count"] + reduced["failed_count"] == 27360
    )
    verdict = "disqualified" if not passed else "blocked" if work["gate_check_summary"] else "null"
    if not work["gate_check_summary"] and not (complete and covered):
        verdict = "disqualified"
    measured = [r for r in reduced["repeat_rows"] if r["status"] == "completed"]
    primary = next(
        c
        for c in reduced["workload_cells"]
        if c["cell"] == dict(workload="natural", batch=1, predictions_per_update=8)
    )
    value: Json = dict(
        reduced,
        experiment_id=8407,
        task_id=TASK,
        milestone="2026.10.724",
        run_date="20261011",
        honest_verdict="complete_" + verdict + "_python_transaction_cost",
        verdict_class=verdict,
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        historical_model_provenance=work["historical_model_provenance"],
        independent_count=len(
            {
                r["source"]["source_cluster_id"]
                for r in reduced["rows"]
                if r.get("source", {}).get("source_cluster_id") and r["status"] == "completed"
            }
        ),
        sample_size_budget=dict(
            cells=27,
            measured_repeats_per_cell=5,
            warmups_per_cell=1,
            intended_transactions=27360,
            timing_repeats_add_independent_sources=0,
        ),
        verifier_is_oracle=False,
        exposure_scope="exposed_cached_development_and_constructed_numeric_controls",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        closed_exposed_utility=dict(H1=-0.00390625, H2=0, reopened=False),
        required_checks_passed=passed,
        flagged_adversarial=not passed,
        acceptance_gates=dict(owned_validation=passed, complete_work=complete, coverage=covered),
        validation_receipts=receipts,
        adversarial_findings=work.get("adversarial_findings", []),
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal_validation.json"),
        gate_check_summary=work["gate_check_summary"],
        preconditions_checked=work["preconditions_checked"],
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=k.SEED,
        source_artifact_hashes=work["refs"],
        code_config_hashes=work.get("code_config_hashes", []),
        cited_upstream_artifacts=[
            dict(
                r,
                imported_fields="task authority, qualified direct state, frozen head, source identity and labels",
            )
            for r in work["refs"]
        ],
        python_cost_ready_score=int(verdict == "null" and complete and covered),
        transactions_per_repeat=dict(
            cycles=8, prediction_attempts="8*r", designated_feedback_attempts=8
        ),
        primary_cell=primary,
        interval_method="10000 repeat-block bootstrap resamples, seed7248407, percentile95 descriptive host interval",
        independent_timing_blocks=5 if complete else 0,
        timing_unit="monotonic_ns",
        timer_resolution_ns=time.get_clock_info("monotonic").resolution * 1e9,
        throughput_estimator="sum(completed prediction decisions)/sum(complete cycle wall seconds), no failed repeat excluded",
        latency_summary=[
            dict(
                cell=r["cell"],
                repeat=r["repeat"],
                p50_seconds=r["p50_seconds"],
                p95_seconds=r["p95_seconds"],
            )
            for r in measured
        ],
        stage_costs=[dict(repeat_id=r["repeat_id"], stage_ns=r["stage_ns"]) for r in measured],
        failure_work_accounting=dict(
            failed=reduced["failed_count"],
            censored=reduced["censored_count"],
            excluded=0,
            failures_remain_in_denominator=True,
        ),
        local_host_scope=dict(
            host=os.uname().nodename,
            machine=os.uname().machine,
            service="complete local decision service",
            llm_acquisition_timed=False,
            general_performance_guarantee=False,
        ),
        independent_cold_cost_reduction_path=str(
            Path(work["raw"]) / "terminal/cold_reduction.json"
        ),
        work_reference=reference(Path(work["work_path"])),
        owned_coverage=work.get("owned_coverage", {}),
        repository_health="separate_global_health_not_launched_per_task",
        methodology_note="Whole durable local Python decisions with fixed work units, delayed designated feedback and timed tail. No LLM acquisition is timed. Numeric controls and cached sources remain exposed; qualification does not establish semantic benefit.",
    )
    value["field_principles"] = {
        f: f"Bind {f} to exact invocation sources and complete primitive work; qualification adds no semantic benefit."
        for f in value
    }
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path, reduction_path: Path | None = None) -> bool:
    """Fresh readers reconstruct fixed work and authenticate sealed input and code identities."""
    try:
        value = json.loads(path.read_bytes())
        checksum = value.pop("reproducibility_checksum")
        if value["experiment_id"] != 8407 or checksum != canonical_hash(value):
            return False
        ref = value["work_reference"]
        work = json.loads(authority.checked(Path(ref["path"]), ref["sha256"]).read_bytes())
        for ref in [*value["source_artifact_hashes"], *value["code_config_hashes"]]:
            authority.checked(Path(ref["path"]), ref["sha256"])
        if work["bundle"]:
            from tempfile import TemporaryDirectory

            with TemporaryDirectory(dir="/var/tmp", prefix="exp8407-cold-") as folder:
                fresh: Json = dict(refs=[], gate_check_summary=[])
                sealed = [dict(r, exists=True, snapshot_path=r["path"]) for r in work["refs"]]
                with frozen_inputs(sealed, Path(folder) / "inputs"):
                    authenticated = authenticate(Path(work["root"]), Path(folder) / "auth", fresh)
                for field in ("head", "slots"):
                    if authenticated[field] != work["bundle"][field]:
                        return False
                if authenticated["labels"]["sha256"] != work["bundle"]["labels"]["sha256"]:
                    return False
        reduced = reduction(work)
        if reduction_path is not None:
            atomic_json(
                reduction_path, {key: item for key, item in reduced.items() if key != "rows"}
            )
        if any(value[key] != expected for key, expected in reduced.items()):
            return False
        return bool(
            value
            == {
                key: expected
                for key, expected in build(work, value["validation_receipts"]).items()
                if key != "reproducibility_checksum"
            }
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError, StopIteration):
        return False

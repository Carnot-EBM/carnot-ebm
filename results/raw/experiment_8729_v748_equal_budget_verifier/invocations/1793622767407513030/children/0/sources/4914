"""REQ-REPORT-8343: independently measured CPU arithmetic survives external blocks.

Source authentication and optional complete transactions reuse existing readers.
The independent branch imports code, without promoting disqualified receipts.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import kv260_arithmetic_8343 as p
from carnot.reporting import kv260_local_cost_boundary_8315 as base
from carnot.reporting import kv260_workload_cost_8329 as prior
from carnot.reporting import kv260_workload_primitives_8329 as durable
from carnot.reporting import v719_contract_replay as authority_module
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.evidence_features_custody_7980 import checked
from carnot.reporting.request_trace_inventory_8200 import operand

Json = dict[str, Any]
ROOT = base.ROOT
NAME = "experiment_8343_v719_kv260_workload_cost"
CLI = "scripts/experiments/" + NAME + ".py"
TEST = "tests/python/test_kv260_workload_cost_8343.py"
OWNED = [
    "python/carnot/reporting/" + name + "_8343.py"
    for name in [
        "kv260_workload_cost",
        "kv260_workload_artifact",
        "kv260_arithmetic",
        "kv260_workload_runner",
    ]
] + [CLI]
MODEL_SPECS: list[Json] = []
SOURCE_TASKS = dict(
    constructed="exp8333-local-evidence-qualification",
    capacity="exp8337-bounded-feedback-capacity",
    natural="exp8336-continuous-local-learning",
)
SOURCES = dict(
    constructed=("experiment_8333_v719_local_evidence_qualification", "local_kernel_ready_score"),
    capacity=("experiment_8337_v719_bounded_feedback_capacity", "capacity_ready_score"),
    natural=("experiment_8336_v719_continuous_local_learning", "trajectory_ready_score"),
)
reference, pin, progress = base.reference, base.pin, p.progress


def optional(root: Path, raw: Path, work: Json) -> None:
    """Measure durable operations only from an authenticated compatible producer."""
    for branch, (name, field) in SOURCES.items():
        path = root / "results" / (name + ".json")
        progress("before_authenticate_" + branch)
        value = base.probe(path, raw, work, field)
        detail = dict(eligible=False, path=str(path), source_count=None)
        work["branches"][branch] = detail
        if value is not None:
            measuring = False
            try:
                for key, wanted in [
                    ("milestone", "2026.10.719"),
                    ("task_id", SOURCE_TASKS[branch]),
                ]:
                    work["checks"].append(operand(key, path, wanted, value.get(key)))
                    if value.get(key) != wanted:
                        raise ValueError("producer_" + key)
                plan = json.loads(
                    pin(
                        checked(value.get("workload_reference", value.get("protocol_reference"))),
                        raw,
                        work["refs"],
                    ).read_bytes()
                )
                trajectories = plan["trajectories"]
                if not trajectories or len({t["id"] for t in trajectories}) != len(trajectories):
                    raise ValueError("workload_source_denominator")
                for trajectory in trajectories:
                    durable.expected(trajectory, "indexed")
                measuring = True
                for repetition in range(-1, 5):
                    for arm in (
                        durable.ARMS if repetition % 2 == 0 else list(reversed(durable.ARMS))
                    ):
                        for trajectory in trajectories:
                            progress(f"before_benchmark_{branch}_{arm}_{repetition}")
                            row = durable.transaction(
                                trajectory, arm, raw / "scratch/checkpoint.json"
                            )
                            durable.verify_row(row)
                            if repetition >= 0:
                                work["durable_rows"].append(
                                    dict(
                                        row,
                                        branch=branch,
                                        source_id=trajectory["id"],
                                        repetition=repetition,
                                    )
                                )
                            progress(f"after_benchmark_{branch}_{arm}_{repetition}")
                detail.update(eligible=True, source_count=len(trajectories))
                work["historical_model_provenance"].append(
                    value.get("historical_model_provenance", [])
                )
            except (OSError, ValueError, KeyError, TypeError) as error:
                work["owned_failure"] |= measuring
                work["checks"].append(
                    operand("qualified_workload_primitives", path, "valid", str(error))
                )
        progress("after_authenticate_" + branch)


def measure(root: Path, raw: Path) -> Json:
    """Check scratch, tools, exact authority and numerical controls before clocks."""
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True)
    work: Json = dict(
        checks=[],
        refs=[],
        branches={},
        timing_rows=[],
        durable_rows=[],
        owned_failure=False,
        historical_model_provenance=[],
        authority={},
    )
    dependencies = [
        "python/carnot/reporting/primary_publication.py",
        "scripts/adversarial_verify.py",
        "scripts/verdict_row_consistency_lint.py",
        "ops/exclusion_manifest.yaml",
        "python/carnot/experiment_7425_v651_spline_prototype.py",
        "python/carnot/reporting/v718_replay_history.py",
        "python/carnot/reporting/kv260_workload_runner_8329.py",
        "python/carnot/reporting/kv260_workload_cost_8329.py",
        "python/carnot/reporting/kv260_workload_primitives_8329.py",
        "python/carnot/reporting/v709_execution.py",
        "python/carnot/reporting/v719_contract_replay.py",
        "python/carnot/reporting/v718_contract_replay.py",
    ]
    work["code_config_hashes"] = [reference(ROOT / name) for name in [*OWNED, TEST, *dependencies]]
    with TemporaryDirectory(prefix="carnot8343-preconditions-") as directory:
        scratch = Path(directory)
        (scratch / "probe").write_bytes(b"private")
        work["checks"].append(
            operand(
                "private_scratch", scratch, True, (scratch / "probe").read_bytes() == b"private"
            )
        )
        work["checks"].append(
            operand(
                "disk_free_at_least_1GiB", scratch, True, shutil.disk_usage(scratch).free >= 1024**3
            )
        )
        for tool in ["python", "pytest", "coverage", "ruff", "mypy"]:
            path = ROOT / ".venv/bin" / tool
            work["checks"].append(operand("tool_available", path, True, path.is_file()))
        progress("before_authority")
        try:
            work["authority"] = authority_module.authority(root, raw / "authority")
            work["checks"].append(
                operand(
                    "activated_V719_authority",
                    root / authority_module.ACTIVE,
                    True,
                    work["authority"]["activated"],
                )
            )
            work["checks"].append(
                operand(
                    "frozen_science_sha256",
                    root / authority_module.PROTOCOL,
                    authority_module.base.PIN,
                    sha256_file(root / authority_module.PROTOCOL),
                )
            )
            for name in [
                authority_module.DESIGN,
                authority_module.ACTIVE,
                authority_module.PROTOCOL,
            ]:
                pin(root / name, raw, work["refs"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            work["checks"].append(
                operand(
                    "activated_V719_authority", root / authority_module.ACTIVE, True, str(error)
                )
            )
        progress("after_authority")
        if all(row["passed"] for row in work["checks"]):
            progress("before_numeric_qualification")
            work["numeric_audit"] = p.audit()
            work["checks"].append(
                operand(
                    "numeric_audit.passed",
                    ROOT / "python/carnot/reporting/kv260_arithmetic_8343.py",
                    True,
                    work["numeric_audit"]["passed"],
                )
            )
            work["owned_failure"] = not work["numeric_audit"]["passed"]
            progress("after_numeric_qualification")
            if not work["owned_failure"]:
                work["timing_rows"] = p.measure()
                optional(root, raw, work)
    progress("before_hardware_authentication")
    historical = base.probe(
        root / "results/experiment_8315_v717_kv260_local_cost_boundary.json", raw, work, None
    )
    work["kv260_history"] = historical.get("kv260_obligation", {}) if historical else {}
    work["polarfire_graduation"] = prior.polarfire(root, raw, work)
    work["checks"].append(
        operand(
            "compatible_spline_kernel",
            root / "results/experiment_8315_v717_kv260_local_cost_boundary.json",
            True,
            False if historical else None,
        )
    )
    progress("after_hardware_authentication")
    atomic_json(raw / "arithmetic.json", work["timing_rows"])
    work["arithmetic_reference"] = reference(raw / "arithmetic.json")
    work["duration_s"] = (time.monotonic_ns() - start) / 1e9
    work["phase_spans"] = [
        dict(
            phase="authenticate_and_measure",
            started_monotonic_ns=start,
            ended_monotonic_ns=time.monotonic_ns(),
        )
    ]
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, raw: Path, receipts: list[Json]) -> Json:
    """Keep artifact reduction in a small module to make primitive replay explicit."""
    from carnot.reporting.kv260_workload_artifact_8343 import build as reduce

    return reduce(work, raw, receipts)


def replay(path: Path) -> bool:
    """Hash checks and independent numerical replay both precede reduction equality."""
    try:
        value = json.loads(path.read_bytes())
        for ref in (
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        ):
            checked(ref)
        work = json.loads(checked(value["measurement_reference"]).read_bytes())
        for key in ["execution_manifest_reference", "owned_coverage_reference"]:
            if work.get(key):
                checked(work[key])
        if json.loads(checked(work["arithmetic_reference"]).read_bytes()) != work["timing_rows"]:
            return False
        for row in work["timing_rows"]:
            p.verify(row)
        for row in work["durable_rows"]:
            durable.verify_row(row)
        for receipt in value["validation_receipts"]:
            for stream in ["stdout", "stderr"]:
                if receipt.get(stream + "_path"):
                    checked(
                        dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
                    )
        return bool(
            build(
                work,
                Path(value["measurement_reference"]["path"]).parent,
                value["validation_receipts"],
            )
            == value
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False

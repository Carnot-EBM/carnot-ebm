"""REQ-REPORT-8646: bind historical replay and fresh installed CPU inference.

The fixed small head executes real arithmetic. Packaging and exposed fixtures
cannot establish independent factual confidence or scientific benefit.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

from carnot.reporting import installed_compatibility_8634 as previous
from carnot.reporting import installed_validation_8634 as historical_validator
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child, progress

Json = dict[str, Any]
ROOT = previous.ROOT
NAME = "experiment_8646_v742_installed_compatibility"
TASK = "exp8646-installed-compatibility"
CLI = f"scripts/experiments/{NAME}.py"
MODEL_SPECS: list[Json] = []
HASH = historical_validator.ORIGINAL_PROTOCOL_HASH
PROTOCOL = "python/carnot/inference/resident_protocol.py"
HISTORY = "results/experiment_8634_v741_installed_compatibility.json"
HISTORY_PIN = "sha256:402009883ba76bdc8ae1a7863ed88e8dfb27564cfb7b628f1b004b0343ae803c"
OWNED = [
    f"python/carnot/reporting/installed_{part}_8646.py"
    for part in ["compatibility", "current", "probe", "report", "validation"]
] + [CLI]
TESTS = [
    "tests/python/test_installed_compatibility_8646.py",
    "tests/python/test_installed_validation_8646.py",
]
reference, load = previous.reference, previous.load


def union(expected: list[str], actual: list[str]) -> Json:
    """Multiplicity matters because a repeated passing test cannot replace a missing test."""
    return dict(
        passed=bool(expected)
        and len(expected) == len(set(expected))
        and sorted(expected) == sorted(actual),
        collected=expected,
        passed_node_ids=actual,
    )


def historical(raw: Path) -> Json:
    """Exercise the existing namespace overlay before creating any replacement mechanism."""
    raw.mkdir(parents=True, exist_ok=True)
    fixture = raw / "current_protocol_fixture.py"
    shutil.copyfile(ROOT / PROTOCOL, fixture)
    failure = child(
        "original_input_hash",
        [
            sys.executable,
            "-u",
            "-c",
            "from carnot.reporting.lean_resident_8608 import authenticate;"
            f"authenticate({dict(path=str(fixture), sha256=HASH)!r})",
        ],
        raw,
        expected=1,
        deadline=30,
    )
    reproduced = failure["passed"] and "input_hash:" in Path(failure["stderr_path"]).read_text()
    overlay = historical_validator.original_overlay(raw, [])
    code = (
        "import json,hashlib,sys;from pathlib import Path;"
        f"sys.path.insert(0,{str(ROOT / 'python/carnot/inference')!r});"
        "import resident_protocol as p;"
        "print(json.dumps(dict(path=p.__file__,sha256='sha256:'+hashlib.sha256(Path(p.__file__).read_bytes()).hexdigest())))"
    )
    receipt = child(
        "original_overlay_probe", [*overlay, sys.executable, "-u", "-c", code], raw, deadline=30
    )
    if not receipt["passed"]:
        raise ValueError("original_overlay_execution_failed")
    imported = json.loads(Path(receipt["stdout_path"]).read_text().splitlines()[-1])
    if imported != dict(path=str(ROOT / PROTOCOL), sha256=HASH) or not reproduced:
        raise ValueError("original_import_identity")
    history = load(dict(path=str(ROOT / HISTORY), sha256=HISTORY_PIN))
    original_nodes = history["original_node_union"]["original_nodes"]
    return dict(
        failure_reproduced=reproduced,
        overlay_passed=True,
        overlay=overlay,
        historical_source_reference=reference(raw / "original_resident_protocol.py"),
        current_source_reference=reference(ROOT / PROTOCOL),
        original_nodes=original_nodes,
        preserved_8634_verdict=history["honest_verdict"],
        resolved_import_rows=[
            dict(imported, source="historical_overlay", passed=True),
            dict(reference(fixture), source="private_current_failure_fixture", passed=reproduced),
        ],
        receipts=[failure, receipt],
    )


def measure(raw: Path, run_date: str, root: Path) -> Json:
    """External missing bytes block; failures after admission disqualify this owned attempt."""
    from carnot.reporting.installed_current_8646 import installed

    progress("8646_preconditions_no_model_load_MODEL_SPECS=[]_current_language_model_calls=0")
    raw.mkdir(parents=True, exist_ok=True)
    start = time.monotonic_ns()
    scratch = Path(tempfile.mkdtemp(prefix="carnot-8646-preconditions-", dir="/var/tmp"))
    (scratch / "writable").write_bytes(b"disk-backed scratch")
    mount = subprocess.check_output(
        ["findmnt", "-T", str(scratch), "-n", "-o", "FSTYPE"], text=True, timeout=5
    ).strip()
    preconditions = dict(
        memory_available_kib=int(
            Path("/proc/meminfo").read_text().split("MemAvailable:")[1].split()[0]
        ),
        disk_free_bytes=shutil.disk_usage(scratch).free,
        scratch=str(scratch),
        filesystem=mount,
        writable=(scratch / "writable").read_bytes() == b"disk-backed scratch",
        tools={n: shutil.which(n) for n in ["git", "bwrap", "findmnt"]},
        python_tools={
            n: (ROOT / ".venv/bin" / n).is_file()
            for n in ["python", "pytest", "coverage", "ruff", "mypy"]
        },
        optional_gpu_required=False,
        optional_board_required=False,
    )
    work: Json = dict(
        raw=str(raw),
        run_date=run_date,
        bundle={},
        gate_check_summary=[],
        installed_parity_rows=[],
        public_surface_rows=[],
        export_compatibility_rows=[],
        rejection_rows=[],
        installed_module_identity={},
        sessions=[],
        phase_spans=[],
        source_artifact_hashes=[],
        preconditions_checked=preconditions,
        isolation_rows=[],
        cleanup_rows=[],
        installed_module_hashes=[],
        historical_source_reference=None,
        resolved_import_rows=[],
        wheel_identity={},
    )
    try:
        if (
            mount in {"tmpfs", "ramfs"}
            or preconditions["memory_available_kib"] < 262144
            or preconditions["disk_free_bytes"] < 268435456
            or not all(preconditions["python_tools"].values())
        ):
            raise ValueError("memory_disk_tools")
        work["bundle"] = previous.inputs(root)
        work["source_artifact_hashes"] = work["bundle"]["refs"] + [
            dict(path=str(root / HISTORY), sha256=HISTORY_PIN)
        ]
        work.update(historical(raw / "historical"))
        progress("8646_historical_overlay_qualified")
        work.update(installed(raw, work["bundle"]))
        work["resolved_import_rows"].extend(
            dict(
                row,
                source="current_worker",
                passed=reference(Path(row["path"]))["sha256"] == row["sha256"],
            )
            for row in work.get("worker_import_rows", [])
        )
        previous.reduce(work)
        if (time.monotonic_ns() - start) / 1e9 > 3600:
            raise ValueError("measurement_deadline")
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AssertionError,
        subprocess.SubprocessError,
    ) as error:
        work["owned_execution_failed"] = bool(work["bundle"])
        path = root / previous.SOURCE
        work["gate_check_summary"].append(
            dict(
                check="original_and_current_compatibility",
                upstream="exp8633-installed-lean-prototype",
                path=str(path),
                sha256=reference(path)["sha256"] if path.exists() else None,
                field="authenticated_inputs_and_separate_execution",
                operator="==",
                expected=1,
                observed=str(error) if path.exists() else "missing",
            )
        )
    end = time.monotonic_ns()
    work.update(
        duration_s=(end - start) / 1e9,
        phase_spans=[dict(phase="measurement", start_ns=start, end_ns=end)],
    )
    work["raw_shard_hashes"] = [reference(p) for p in sorted(raw.rglob("*")) if p.is_file()]
    progress("8646_measurement_complete", len(work["installed_parity_rows"]), 0)
    return work

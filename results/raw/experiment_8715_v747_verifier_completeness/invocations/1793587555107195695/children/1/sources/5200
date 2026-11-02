"""REQ-REPORT-8571: checked research publication is separate from production serving."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import signal
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import resident_inference_8571 as e
from carnot.reporting import resident_validation_8571 as validation
from carnot.reporting import resident_worker_8571 as worker
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child, progress
from carnot.reporting.v726_feature_panel_runner import scan

Json = dict[str, Any]


def measure(root: Path, raw: Path) -> Json:
    """Record real CPU, disk, tool and deadline checks before numerical requests."""
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True)
    work: Json = dict(
        raw=str(raw),
        bundle=None,
        primitive=None,
        gate_check_summary=[],
        phase_spans=[],
        duration_s=0.0,
    )
    with TemporaryDirectory(prefix="exp8571-", dir="/var/tmp") as scratch:
        probe = Path(scratch) / "writable"
        probe.write_bytes(b"disk-backed probe")
        memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        mounts = [line.split() for line in Path("/proc/self/mountinfo").read_text().splitlines()]
        mounted = max(
            (m for m in mounts if Path(scratch).is_relative_to(m[4])), key=lambda m: len(m[4])
        )
        filesystem = mounted[mounted.index("-") + 1]
        work["preconditions_checked"] = dict(
            scratch=scratch,
            writable=probe.read_bytes() == b"disk-backed probe",
            free_disk_bytes=shutil.disk_usage(scratch).free,
            memory_available_kib=int(memory["MemAvailable"].split()[0]),
            scratch_filesystem=filesystem,
            tools={
                name: shutil.which(str(e.ROOT / ".venv/bin" / name))
                for name in ["python", "pytest", "ruff", "mypy"]
            },
            task_cap_s=4800,
            measurement_deadline_s=3600,
            validation_reserve_s=900,
            optional_cuda="not_required_for_cpu_work",
            optional_board="not_required_for_cpu_work",
        )
        try:
            checks = work["preconditions_checked"]
            for field, passed in {
                "writable_disk": checks["writable"],
                "disk_backed_scratch": checks["scratch_filesystem"] not in {"tmpfs", "ramfs"},
                "disk_capacity": checks["free_disk_bytes"] >= 64 * 1024 * 1024,
                "available_memory": checks["memory_available_kib"] >= 512 * 1024,
                "required_tools": all(checks["tools"].values()),
            }.items():
                e.require(Path(scratch), field, True, passed)
            progress("8571_before_admission")
            work["bundle"] = e.admit(root)
            admitted = time.monotonic_ns()
            work["phase_spans"].append(dict(phase="admission", start_ns=start, end_ns=admitted))
            work["primitive"] = e.prototype(work["bundle"], raw / "prototype")
            end = time.monotonic_ns()
            work["phase_spans"].append(
                dict(phase="private_prototype", start_ns=admitted, end_ns=end)
            )
            e.require(raw, "measurement_deadline", True, (end - start) / 1e9 <= 3600)
        except e.GateFailure as error:
            work["gate_check_summary"].append(error.row)
        except (OSError, ValueError, KeyError, TypeError, TimeoutError) as error:
            work["owned_measurement_failure"] = str(error)
            work["gate_check_summary"].append(
                dict(
                    check="owned_measurement",
                    upstream=e.TASK,
                    path=str(raw),
                    sha256=None,
                    field="completed_without_error",
                    operator="==",
                    expected=True,
                    observed=str(error),
                )
            )
        work["duration_s"] = (time.monotonic_ns() - start) / 1e9
    atomic_json(raw / "work.json", work)
    return work


def run(root: Path, output: Path, control: bool = False) -> int:
    """Every required failure stays visible and prevents readiness at publication."""
    start = time.monotonic_ns()
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    work = measure(root, raw)
    receipts = [] if control or not work["bundle"] else validation.validate(work, raw)
    if work.get("owned_measurement_failure"):
        receipts.append(dict(name="owned_measurement", passed=False))
    atomic_json(raw / "work.json", work)
    candidate = raw / "preliminary_candidate.json"
    atomic_json(candidate, e.artifact(work, receipts, output))
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    for name, args, expected in [
        ("cold", ["--cold-replay", str(candidate)], 0),
        ("missing", ["--cold-replay", str(raw / "absent")], 1),
        ("owned_failure", ["--deliberate-error"], 7),
    ]:
        receipts.append(child(name, argv + args, raw / "controls", expected=expected, deadline=120))
    changed = deepcopy(json.loads(candidate.read_bytes()))
    changed["completed_count"] += 1
    changed["reproducibility_checksum"] = e.checksum(changed)
    tamper = raw / "self_consistently_rehashed_tamper.json"
    atomic_json(tamper, changed)
    receipts.append(
        child(
            "rehashed_tamper",
            argv + ["--cold-replay", str(tamper)],
            raw / "controls",
            expected=1,
            deadline=120,
        )
    )
    if work["bundle"] and not control:
        receipts.extend(validation.finish(work, raw))
    atomic_json(raw / "work.json", work)
    atomic_json(candidate, e.artifact(work, receipts, output))
    checks, findings = scan(candidate, raw / "preterminal_validators")
    receipts.extend(checks)
    work["adversarial_findings"] = findings
    work["duration_s"] = (time.monotonic_ns() - start) / 1e9
    atomic_json(raw / "work.json", work)
    value = e.artifact(work, receipts, output)

    def validator(path: Path) -> Json:
        """Final scanners and cold replay read the exact bytes exposed by the publisher."""
        checked, flags = scan(path, raw / "terminal_validators")
        cold = child(
            "final_byte_cold", argv + ["--cold-replay", str(path)], raw / "controls", deadline=120
        )
        return dict(
            passed=cold["passed"]
            and (
                all(r["passed"] for r in checked)
                or value["verdict_class"] in {"blocked", "disqualified"}
            ),
            receipts=[*checked, cold],
            findings=flags,
        )

    publication = publish_primary(output, value, validator)
    selected = reader_receipt(
        e.TASK,
        output.parent,
        field="prototype_ready_score",
        expected=value["prototype_ready_score"],
    )
    atomic_json(
        Path(value["terminal_validation_sidecar_path"]),
        dict(
            publication=publication,
            selected_identity=selected,
            validation_receipts=receipts,
            adversarial_findings=findings,
        ),
    )
    progress("8571_finished", value["completed_count"], value["failed_count"])
    return 0


def main(argv: list[str] | None = None) -> int:
    """Only an explicit worker argument selects framed serving; the default performs research."""
    arguments = sys.argv[1:] if argv is None else argv
    if arguments and arguments[0] == "--sealed-tests":
        import pytest

        return int(pytest.main(arguments[2:]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261016"], default="20261016")
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--config-hash", default="")
    parser.add_argument("--arm", choices=["python", "native"], default="python")
    parser.add_argument("--timeout", type=float, default=15)
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    args = parser.parse_args(argv)
    if args.deliberate_error:
        return 7
    if args.worker:
        return worker.serve(args.worker, args.config_hash, args.arm, args.timeout)
    if args.cold_replay:
        return int(not e.replay(args.cold_replay))
    if args.private_control and args.output.absolute().is_relative_to(e.ROOT):
        parser.error("private control output must be outside checkout")
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        return run(args.root, args.output.absolute(), args.private_control)
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)

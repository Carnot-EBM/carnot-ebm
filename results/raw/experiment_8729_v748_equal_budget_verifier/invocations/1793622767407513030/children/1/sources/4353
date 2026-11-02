"""REQ-VERIFY-8454: reuse bounded validation without touching historical outputs."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import board_operation_boundary_8454 as e
from carnot.reporting import v726_feature_panel_runner as shipped
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child, execute

Json = dict[str, Any]


def manifest(private: Path) -> list[Json]:
    """The qualified plan already preserves child shards and private E2E-018/021."""
    with patch.object(shipped, "e", e):
        return list(shipped.manifest(private))


def run(root: Path, output: Path, private: Path, *, control: bool = False) -> int:
    """Exact commands and primitive bytes are retained before terminal publication."""
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, mode=0o700)
    plan = (
        [
            dict(
                name="private_child",
                argv=[sys.executable, "-u", "-c", "print('actual private child',flush=True)"],
                expected=0,
                deadline=30,
                scope="owned",
            )
        ]
        if control
        else manifest(private)
    )
    atomic_json(
        raw / "execution_manifest.json",
        dict(
            commands=plan,
            owned=e.OWNED,
            measurement_cap_s=3600,
            task_cap_s=4800,
            validation_reserve_s=900,
            child_poll_s=30,
        ),
    )
    work = e.measure(root, raw, private)
    receipts = execute(plan, raw / "validation")
    if (private / "coverage.json").is_file():
        shards = raw / "coverage_shards"
        shards.mkdir()
        for source in private.glob(".coverage*"):
            shutil.copyfile(source, shards / source.name)
        shutil.copyfile(private / "coverage.json", raw / "coverage.json")
        work["coverage_shards"] = [e.reference(p) for p in sorted(shards.iterdir())]
        work["owned_coverage"] = json.loads((raw / "coverage.json").read_bytes())["totals"]
    work["refs"].append(e.reference(raw / "execution_manifest.json"))
    atomic_json(raw / "measurement.json", work)
    candidate = raw / "preterminal.json"
    atomic_json(candidate, e.build(work, receipts, raw, output))
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    for name, args, expected in [
        ("cold", ["--cold-replay", str(candidate)], 0),
        ("missing", ["--cold-replay", str(raw / "absent.json")], 1),
        ("error", ["--deliberate-error"], 7),
    ]:
        receipts.append(
            child(name, [*argv, *args], raw / "controls", expected=expected, deadline=120)
        )
    changed = deepcopy(json.loads(candidate.read_bytes()))
    changed["board_boundary_recorded_score"] = 999
    changed["reproducibility_checksum"] = e.checksum(changed)
    tamper = raw / "rehashed_tamper.json"
    atomic_json(tamper, changed)
    receipts.append(
        child(
            "rehashed_tamper",
            [*argv, "--cold-replay", str(tamper)],
            raw / "controls",
            expected=1,
            deadline=120,
        )
    )
    atomic_json(candidate, e.build(work, receipts, raw, output))
    checks, findings = shipped.scan(candidate, raw / "preterminal_validators")
    receipts.extend(checks)
    work["adversarial_findings"] = findings
    work["ended_monotonic_ns"] = time.monotonic_ns()
    work["phase_spans"] += [dict(name=r["name"], duration_s=r["duration_s"]) for r in receipts]
    atomic_json(raw / "measurement.json", work)
    value = e.build(work, receipts, raw, output)

    def validator(path: Path) -> Json:
        """A fresh process and both unchanged scanners must accept exact candidate bytes."""
        cold = child(
            "terminal_cold",
            [*argv, "--cold-replay", str(path)],
            raw / "terminal_validators",
            deadline=120,
        )
        checked, flags = shipped.scan(path, raw / "terminal_validators")
        return dict(
            passed=cold["passed"] and all(c["passed"] for c in checked) and not flags,
            findings=flags,
            receipts=[cold, *checked],
        )

    publication = publish_primary(output, value, validator)
    consumer = reader_receipt(
        e.TASK,
        output.parent,
        field="board_boundary_recorded_score",
        expected=value["board_boundary_recorded_score"],
    )
    atomic_json(
        Path(value["terminal_validation_sidecar_path"]),
        dict(
            publication=publication,
            consumer=consumer,
            validation_receipts=receipts,
            adversarial_findings=findings,
        ),
    )
    e.progress(
        "published_" + value["verdict_class"], value["completed_count"], value["censored_count"]
    )
    return 0

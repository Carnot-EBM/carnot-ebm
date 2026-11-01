"""REQ-VERIFY-8410: freeze bounded validation before reading cached CUDA evidence."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import execute
from carnot.reporting import sentence_spline_execution_8334 as publisher
from carnot.verify import cuda_reader_8410 as q
from carnot.verify import runtime_evidence_execution_8382 as base

Json = dict[str, Any]
PY = base.PY
SCRATCH = Path.home() / ".cache/carnot-exp8410-private"


def manifest(private: Path) -> list[Json]:
    """Reuse genuine child coverage and unchanged consumers; global health is separate."""
    with patch.object(base, "q", q):
        plan = [s for s in base.manifest(private) if s["scope"] != "global"]
    code = (
        "from pathlib import Path; import shutil; p=Path("
        + repr(str(private))
        + "); d=p/'coverage_shards'; d.mkdir(exist_ok=True); rows=list(p.glob('.coverage.*')); [shutil.copyfile(s,d/s.name) for s in rows]; print('genuine child shards sealed:',len(rows),flush=True); assert rows"
    )
    plan.insert(
        2,
        dict(
            name="seal_genuine_child_coverage_shards",
            argv=[PY, "-u", "-c", code],
            deadline=30,
            expected=0,
            scope="owned",
        ),
    )
    plan[3]["argv"].append("--keep")
    return plan


def main(argv: list[str] | None = None) -> int:
    """Bound this invocation and preserve historical date contracts in their own runners."""
    args = list(sys.argv[1:] if argv is None else argv)
    q.progress("start_no_model_load_MODEL_SPECS_empty_current_LLM_loads_and_generations_zero")
    if "--deliberate-error" in args:
        return 7
    if "--cuda-probe" in args:
        q.progress("owned_driver_context_copy_before")
        result = q.primitive.main(args)
        print(
            "[exp8410] phase=owned_driver_context_copy_after completed=1 pending=0",
            file=sys.stderr,
            flush=True,
        )
        return int(result)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261011"], default="20261011")
    parser.add_argument("--root", type=Path, default=q.ROOT)
    parser.add_argument("--output", type=Path, default=q.ROOT / "results" / (q.NAME + ".json"))
    parser.add_argument("--environment-receipt", type=Path)
    parser.add_argument("--receipt-sha256")
    parser.add_argument("--private-run", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parsed = parser.parse_args(args)
    if parsed.cold_replay:
        passed = q.replay(parsed.cold_replay)
        q.progress("replay_passed" if passed else "replay_rejected")
        return int(not passed)
    output = parsed.output.absolute()
    if parsed.private_run and output.is_relative_to(q.ROOT / "results"):
        parser.error("private run requires private output")
    began, wall = time.monotonic_ns(), time.time_ns()
    raw = output.parent / "raw" / q.NAME / "invocations" / str(wall)
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        SCRATCH.mkdir(parents=True, exist_ok=True, mode=0o700)
        SCRATCH.chmod(0o700)
        with TemporaryDirectory(prefix="invocation-", dir=SCRATCH) as directory:
            private = Path(directory)
            plan = manifest(private)
            atomic_json(
                raw / "execution_manifest.json",
                dict(
                    commands=plan,
                    frozen_before_measurement=True,
                    task_cap_s=4800,
                    heartbeat_s=20,
                    terminal_commands=[
                        "cold-replay",
                        "scripts/adversarial_verify.py --json",
                        "scripts/verdict_row_consistency_lint.py --strict",
                    ],
                ),
            )
            with patch.dict(
                os.environ,
                {"TMPDIR": str(private), "PYTHONUNBUFFERED": "1", "JAX_PLATFORMS": "cpu"},
            ):
                receipts = execute(plan[:1] if parsed.private_run else plan, raw / "validation")
                work = q.measure(
                    parsed.root,
                    raw,
                    parsed.environment_receipt,
                    parsed.receipt_sha256,
                    fixture=parsed.private_run,
                )
            work["refs"].append(q.reference(raw / "execution_manifest.json"))
            coverage = private / "coverage.json"
            if coverage.is_file():
                report = json.loads(coverage.read_bytes())
                atomic_json(raw / "owned_coverage.json", report)
                work["owned_statement_coverage"] = report["totals"]
                work["refs"].append(q.reference(raw / "owned_coverage.json"))
                for shard in sorted(
                    [*private.glob(".coverage.*"), *(private / "coverage_shards").glob("*")]
                ):
                    target = raw / "coverage_shards" / shard.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(shard.read_bytes())
                    work.setdefault("coverage_shard_references", []).append(q.reference(target))
            ended = time.monotonic_ns()
            work.update(
                duration_s=(ended - began) / 1e9,
                phase_spans=[
                    dict(
                        phase="validation_and_cached_evidence",
                        started_wall_ns=wall,
                        started_monotonic_ns=began,
                        ended_monotonic_ns=ended,
                        duration_s=(ended - began) / 1e9,
                    )
                ],
            )
            atomic_json(raw / "measurement.json", work)
            value = q.build(work, raw, receipts)
            valid = private / "valid.json"
            atomic_json(valid, value)
            bad = deepcopy(value)
            bad["runtime_changed_score"] = 1 - bad["runtime_changed_score"]
            bad["reproducibility_checksum"] = q.checksum(bad)
            tamper = private / "tamper.json"
            atomic_json(tamper, bad)
            controls = [
                dict(
                    name=name,
                    argv=[PY, "-u", str(q.ROOT / q.CLI), "--cold-replay", str(path)],
                    deadline=60,
                    expected=expected,
                    scope="owned",
                )
                for name, path, expected in [
                    ("positive_replay", valid, 0),
                    ("missing_input_replay", private / "absent", 1),
                    ("self_consistently_rehashed_tamper", tamper, 1),
                ]
            ]
            controls.append(
                dict(
                    name="deliberate_error",
                    argv=[PY, "-u", str(q.ROOT / q.CLI), "--deliberate-error"],
                    deadline=30,
                    expected=7,
                    scope="owned",
                )
            )
            receipts.extend(execute(controls, raw / "cold_controls"))
            with patch.object(publisher, "e", q):
                publisher.publish(q.build(work, raw, receipts), output, raw)
        return 0
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)

"""REQ-REPORT-8436: bounded real children verify terminal bytes before publication.

Private control invocations avoid recursively launching pytest from pytest.
The requested production command always runs the full owned validation plan.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import signal
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import transaction_attribution_8436 as r
from carnot.reporting import v726_feature_panel_runner as shipped
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child, execute

Json = dict[str, Any]


def manifest(private: Path) -> list[Json]:
    """Reuse scoped genuine Coverage.py children and the named private E2E checks."""
    with patch.object(shipped, "e", r):
        return list(shipped.manifest(private))


def clock_controls(work: Json, raw: Path, argv: list[str]) -> list[Json]:
    """Actual cold readers reject damaged clocks even after all local hashes change."""
    receipts = []
    for name in ("doubled_stage", "deleted_tail", "swapped_repeat", "rehashed_timings"):
        changed = deepcopy(work)
        folder = raw / name
        folder.mkdir(parents=True)
        changed["raw"] = str(folder)
        primitive = r.read(changed["operands"][1])
        if name == "doubled_stage":
            span = primitive["transactions"][0]["stages"][0]
            span["end_ns"] += span["end_ns"] - span["start_ns"]
        elif name == "deleted_tail":
            del primitive["tail_start_ns"]
        elif name == "swapped_repeat":
            primitive["repeat_id"] = "swapped"
        else:
            primitive["transactions"][0]["stages"][0]["name"] = "rehashed_conversion"
        operand = folder / "primitive.json"
        atomic_json(operand, primitive)
        changed["operands"][1] = dict(r.reference(operand), warmup=False)
        atomic_json(
            folder / "attribution_rows.json", dict(rows=changed["rows"], warmups=changed["warmups"])
        )
        atomic_json(folder / "measurement.json", changed)
        candidate = folder / "candidate.json"
        value = r.build(changed, [], candidate)
        atomic_json(candidate, value)
        receipts.append(
            child(
                name,
                [*argv, "--cold-replay", str(candidate)],
                folder / "logs",
                expected=1,
                deadline=180,
            )
        )
    return receipts


def run(root: Path, output: Path, private: Path, control: bool) -> int:
    """Retain every primitive and validation receipt before exposing checked bytes."""
    raw = output.parent / "raw" / r.NAME / "invocations" / str(time.time_ns())
    plan = [] if control else manifest(private)
    raw.mkdir(parents=True, mode=0o700)
    atomic_json(
        raw / "validation_plan.json",
        dict(
            commands=plan,
            task_cap_s=4800,
            validation_reserve_s=900,
            full_repository_suite="not_run",
        ),
    )
    work = r.measure(root, raw, private)
    receipts = execute(plan, raw / "validation")
    if (private / "coverage.json").is_file():
        shards = raw / "coverage_shards"
        shards.mkdir()
        for source in private.glob(".coverage*"):
            shutil.copyfile(source, shards / source.name)
        shutil.copyfile(private / "coverage.json", raw / "coverage.json")
        work["coverage_shards"] = [r.reference(path) for path in sorted(shards.iterdir())]
        work["owned_coverage"] = r.read(r.reference(raw / "coverage.json"))["totals"]
    atomic_json(raw / "measurement.json", work)
    candidate = raw / "preterminal.json"
    atomic_json(candidate, r.build(work, receipts, output))
    argv = [str(r.ROOT / ".venv/bin/python"), "-u", str(r.ROOT / r.CLI)]
    if work["operands"]:
        receipts.extend(clock_controls(work, raw / "clock_controls", argv))
    for name, args, expected in [
        ("cold", ["--cold-replay", str(candidate)], 0),
        ("missing", ["--cold-replay", str(raw / "absent.json")], 1),
        ("error", ["--deliberate-error"], 7),
    ]:
        receipts.append(
            child(name, [*argv, *args], raw / "controls", expected=expected, deadline=180)
        )
    changed = deepcopy(r.read(r.reference(candidate)))
    changed["clock_attribution_ready_score"] = 999
    changed["reproducibility_checksum"] = r.checksum(changed)
    tamper = raw / "rehashed_tamper.json"
    atomic_json(tamper, changed)
    receipts.append(
        child(
            "rehashed_tamper",
            [*argv, "--cold-replay", str(tamper)],
            raw / "controls",
            expected=1,
            deadline=180,
        )
    )
    atomic_json(candidate, r.build(work, receipts, output))
    checks, flags = shipped.scan(candidate, raw / "initial_terminal")
    receipts.extend(checks)
    work.update(
        adversarial_findings=flags,
        duration_s=(time.monotonic_ns() - work["started_monotonic_ns"]) / 1e9,
    )
    work["phase_spans"].extend(
        dict(phase=row["name"], duration_s=row["duration_s"]) for row in receipts
    )
    atomic_json(raw / "measurement.json", work)
    value = r.build(work, receipts, output)

    def validator(path: Path) -> Json:
        """Findings remain binding; a failed validator cannot grant readiness."""
        checked, findings = shipped.scan(path, raw / "terminal")
        return dict(
            passed=r.replay(path) and all(row["passed"] for row in checked),
            receipts=checked,
            findings=findings,
        )

    publication = publish_primary(output, value, validator)
    reader = reader_receipt(
        r.TASK,
        output.parent,
        field="clock_attribution_ready_score",
        expected=value["clock_attribution_ready_score"],
    )
    atomic_json(
        Path(value["terminal_validation_sidecar_path"]),
        dict(publication=publication, reader=reader),
    )
    r.progress("published_" + value["verdict_class"], 1, 0)
    return int(not value["required_checks_passed"])


def main(argv: list[str] | None = None) -> int:
    """Use disk-backed private scratch and a hard task cap without padding duration."""
    r.progress("start_no_model_load_MODEL_SPECS_empty_current_LLM_invocations_zero")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261012"], default="20261012")
    parser.add_argument("--root", type=Path, default=r.ROOT)
    parser.add_argument("--output", type=Path, default=r.ROOT / "results" / (r.NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    parser.add_argument("--private-control", action="store_true")
    args = parser.parse_args(argv)
    if args.deliberate_error:
        return 7
    if args.cold_replay:
        passed = r.replay(args.cold_replay)
        r.progress("replay_passed" if passed else "replay_failed")
        return int(not passed)
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        with TemporaryDirectory(dir="/var/tmp", prefix="exp8436-private-") as folder:
            return run(args.root, args.output, Path(folder), args.private_control)
    except (OSError, ValueError, KeyError, TypeError) as error:
        r.progress("owned_failure_" + str(error))
        return 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)

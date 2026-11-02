"""REQ-REPORT-8437: bounded private children qualify exact terminal bytes."""

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

from carnot.reporting import acceleration_ceiling_8437 as r
from carnot.reporting import v726_feature_panel_runner as shipped
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child, execute

Json = dict[str, Any]


def manifest(private: Path) -> list[Json]:
    """Reuse real serial Coverage.py child shards and the task's named private E2E."""
    with patch.object(shipped, "e", r):
        return list(shipped.manifest(private))


def run(root: Path, output: Path, private: Path, control: bool) -> int:
    """Keep all failures and primitive evidence before atomic validated publication."""
    raw = output.parent / "raw" / r.NAME / "invocations" / str(time.time_ns())
    plan = [] if control else manifest(private)
    work = r.measure(root, raw, private)
    atomic_json(
        raw / "validation_plan.json",
        dict(
            commands=plan,
            task_cap_s=4800,
            validation_reserve_s=900,
            full_repository_suite="not_run",
        ),
    )
    receipts = execute(plan, raw / "validation")
    if (private / "coverage.json").is_file():
        shards = raw / "coverage_shards"
        shards.mkdir()
        for source in private.glob(".coverage*"):
            shutil.copyfile(source, shards / source.name)
        shutil.copyfile(private / "coverage.json", raw / "coverage.json")
        work["coverage_shards"] = [r.reference(p) for p in sorted(shards.iterdir())]
        work["owned_coverage"] = r.read(r.reference(raw / "coverage.json"))["totals"]
    candidate = raw / "preterminal.json"
    atomic_json(raw / "measurement.json", work)
    atomic_json(candidate, r.build(work, receipts))
    argv = [str(r.ROOT / ".venv/bin/python"), "-u", str(r.ROOT / r.CLI)]
    for name, args, expected in [
        ("cold", ["--cold-replay", str(candidate)], 0),
        ("missing", ["--cold-replay", str(raw / "absent.json")], 1),
        ("error", ["--deliberate-error"], 7),
    ]:
        receipts.append(
            child(name, [*argv, *args], raw / "controls", expected=expected, deadline=180)
        )
    changed = deepcopy(r.build(work, receipts))
    changed["acceleration_analysis_ready_score"] = 999
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
    if work["operands"]:
        altered = deepcopy(work)
        path = raw / "substituted.json"
        primitive = r.read(altered["operands"][0])
        primitive["end_ns"] += 1000
        atomic_json(path, primitive)
        altered["operands"][0] = r.reference(path)
        atomic_json(raw / "substitution_work.json", altered)
        bad = r.build(altered, receipts)
        bad["work_reference"] = r.reference(raw / "substitution_work.json")
        bad["reproducibility_checksum"] = r.checksum(bad)
        atomic_json(raw / "row_substitution.json", bad)
        receipts.append(
            child(
                "row_substitution",
                [*argv, "--cold-replay", str(raw / "row_substitution.json")],
                raw / "controls",
                expected=1,
                deadline=180,
            )
        )
    atomic_json(candidate, r.build(work, receipts))
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
    value = r.build(work, receipts)

    def validator(path: Path) -> Json:
        """Unchanged scanners and cold reconstruction bind the final candidate bytes."""
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
        field="acceleration_analysis_ready_score",
        expected=value["acceleration_analysis_ready_score"],
    )
    atomic_json(
        Path(value["terminal_validation_sidecar_path"]),
        dict(publication=publication, reader=reader),
    )
    r.progress(
        "published_" + value["verdict_class"], value["completed_count"], value["censored_count"]
    )
    return int(not value["required_checks_passed"])


def main(argv: list[str] | None = None) -> int:
    """Private scratch and the deadline keep analytical work bounded without device calls."""
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
    if args.private_control and args.output.absolute().is_relative_to(r.ROOT):
        return 2
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        with TemporaryDirectory(dir="/var/tmp", prefix="exp8437-private-") as folder:
            return run(args.root, args.output.absolute(), Path(folder), args.private_control)
    except (OSError, ValueError, KeyError, TypeError) as error:
        r.progress("owned_failure_" + str(error))
        return 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)

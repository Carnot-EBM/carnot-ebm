"""REQ-REPORT-8461: bounded private validation precedes reader-visible publication."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import objective_checkpoint_8461 as q
from carnot.reporting import v726_feature_panel_runner as consumers
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child, execute
from carnot.verify import objective_checkpoint_8461 as adapter

Json = dict[str, Any]


def run(root: Path, output: Path, private: Path, *, validate: bool = True) -> int:
    """The private candidate records owned failures before any atomic terminal write."""
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    plan = q.manifest(private)
    atomic_json(
        raw / "execution_manifest.json",
        dict(commands=plan, owned=q.OWNED, full_repository_suite="not_run"),
    )
    work = q.measure(root, raw, private / "checkpoints")
    receipts = execute(plan, raw / "validation") if validate else []
    if (private / "coverage.json").is_file():
        shards = raw / "coverage_shards"
        shards.mkdir()
        for source in [
            *private.glob(".coverage*"),
            *(private / "uncombined_shards").glob(".coverage.*"),
        ]:
            shutil.copyfile(source, shards / source.name)
        shutil.copyfile(private / "coverage.json", raw / "coverage.json")
        work["coverage_shards"] = [q.reference(p) for p in sorted(shards.iterdir())]
    atomic_json(raw / "measurement.json", work)
    candidate = raw / "preterminal.json"
    atomic_json(candidate, q.build(work, receipts, raw, output))
    argv = [sys.executable, "-u", str(q.ROOT / q.CLI)]
    for name, args, expected in [
        ("cold", ["--cold-replay", str(candidate)], 0),
        ("missing", ["--cold-replay", str(raw / "absent")], 1),
        ("error", ["--deliberate-error"], 7),
    ]:
        receipts.append(child(name, argv + args, raw / "controls", expected=expected, deadline=90))
    changed = deepcopy(json.loads(candidate.read_bytes()))
    changed["objective_kernel_ready_score"] = 999
    changed["reproducibility_checksum"] = q.checksum(changed)
    atomic_json(raw / "rehashed_tamper.json", changed)
    receipts.append(
        child(
            "rehashed_tamper",
            argv + ["--cold-replay", str(raw / "rehashed_tamper.json")],
            raw / "controls",
            expected=1,
            deadline=90,
        )
    )
    atomic_json(candidate, q.build(work, receipts, raw, output))
    checks, findings = consumers.scan(candidate, raw / "preterminal_validators")
    receipts.extend(checks)
    work["adversarial_findings"] = findings
    work["ended_monotonic_ns"] = time.monotonic_ns()
    atomic_json(raw / "measurement.json", work)
    value = q.build(work, receipts, raw, output)

    def validator(path: Path) -> Json:
        """An honest disqualification retains findings while forbidding kernel readiness."""
        terminal_checks, flags = consumers.scan(path, raw / "terminal_validators")
        honest_failure = (
            value["verdict_class"] == "disqualified"
            and value["objective_kernel_ready_score"] == 0
            and not value["required_checks_passed"]
        )
        return dict(
            passed=q.replay(path) and (all(r["passed"] for r in terminal_checks) or honest_failure),
            findings=flags,
            receipts=terminal_checks,
        )

    publication = publish_primary(output, value, validator)
    reader = reader_receipt(
        q.TASK,
        output.parent,
        field="objective_kernel_ready_score",
        expected=value["objective_kernel_ready_score"],
    )
    atomic_json(
        Path(value["terminal_validation_sidecar_path"]),
        dict(
            publication=publication,
            consumer=reader,
            primary_sha256=publication["primary_sha256"],
            validation_receipts=receipts,
            adversarial_findings=findings,
        ),
    )
    q.progress("finished", value["completed_count"], value["censored_count"])
    return 0


def main(argv: list[str] | None = None) -> int:
    """Dated CLI and private child modes load numerical artifacts without loading a generator."""
    q.progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261012"], default="20261012")
    parser.add_argument("--root", type=Path, default=q.ROOT)
    parser.add_argument("--output", type=Path, default=q.ROOT / "results" / (q.NAME + ".json"))
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--configuration", type=Path)
    parser.add_argument("--crash", default="")
    parser.add_argument("--state-replay", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    args = parser.parse_args(argv)
    if args.deliberate_error:
        return 7
    if args.cold_replay:
        passed = q.replay(args.cold_replay)
        q.progress("replay_passed" if passed else "reduction_drift")
        return int(not passed)
    if args.worker or args.state_replay:
        try:
            cfg = json.loads(args.configuration.read_bytes())
            if args.state_replay:
                return int(
                    not adapter.replay_state(json.loads(args.state_replay.read_bytes()), cfg)
                )
            result = adapter.drive(args.worker, cfg, args.crash)
            atomic_json(args.worker / "final.json", result)
            return 0
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return 1
    with TemporaryDirectory(prefix="carnot8461-", dir="/var/tmp") as directory:
        return run(args.root, args.output.absolute(), Path(directory))

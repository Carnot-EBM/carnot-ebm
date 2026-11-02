"""REQ-VERIFY-8449: bounded private workers validate before atomic publication."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import signal
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import persistence_localization_8449 as r
from carnot.reporting import v726_feature_panel_runner as shipped
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child, execute
from carnot.verify import persistence_localization_8449 as k

Json = dict[str, Any]


def manifest(private: Path) -> list[Json]:
    """The shipped scoped plan includes genuine child coverage and private E2E-018/021."""
    with patch.object(shipped, "e", r):
        return list(shipped.manifest(private))


def run(root: Path, output: Path, private: Path, control: bool) -> int:
    """Retain commands, exits and hashes before checked terminal bytes become visible."""
    raw = output.parent / "raw" / r.NAME / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, mode=0o700)
    plan = [] if control else manifest(private)
    atomic_json(
        raw / "validation_plan.json",
        dict(commands=plan, task_cap_s=4800, measurement_cap_s=1800, validation_reserve_s=900),
    )
    work = r.measure(root, raw, private)
    receipts = [*work["receipts"], *execute(plan, raw / "validation")]
    if (private / "coverage.json").is_file():
        shards = raw / "coverage_shards"
        shards.mkdir()
        for source in private.glob(".coverage*"):
            shutil.copyfile(source, shards / source.name)
        shutil.copyfile(private / "coverage.json", raw / "coverage.json")
        work.update(
            coverage_shards=[r.reference(path) for path in sorted(shards.iterdir())],
            owned_coverage=json.loads((raw / "coverage.json").read_bytes())["totals"],
        )
    atomic_json(raw / "measurement.json", work)
    candidate = raw / "preterminal.json"
    atomic_json(candidate, r.build(work, receipts, output))
    argv = [str(r.ROOT / ".venv/bin/python"), "-u", str(r.ROOT / r.CLI)]
    for name, args, expected in [
        ("cold", ["--cold-replay", str(candidate)], 0),
        ("missing", ["--cold-replay", str(raw / "absent.json")], 1),
        ("error", ["--deliberate-error"], 7),
    ]:
        receipts.append(
            child(name, [*argv, *args], raw / "controls", expected=expected, deadline=180)
        )
    changed = deepcopy(json.loads(candidate.read_bytes()))
    changed["persistence_profile_ready_score"] = 999
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
    checks, findings = shipped.scan(candidate, raw / "initial_terminal")
    receipts.extend(checks)
    work.update(
        adversarial_findings=findings,
        duration_s=(time.monotonic_ns() - work["started_monotonic_ns"]) / 1e9,
    )
    work["phase_spans"].extend(
        dict(phase=row["name"], duration_s=row["duration_s"]) for row in receipts
    )
    atomic_json(raw / "measurement.json", work)
    value = r.build(work, receipts, output)

    def validator(path: Path) -> Json:
        """Unchanged findings remain binding even when a command returns zero."""
        checked, flags = shipped.scan(path, raw / "terminal")
        return dict(
            passed=r.replay(path) and all(row["passed"] for row in checked) and not flags,
            receipts=checked,
            findings=flags,
        )

    publication = publish_primary(output, value, validator)
    reader = reader_receipt(
        r.TASK,
        output.parent,
        field="persistence_profile_ready_score",
        expected=value["persistence_profile_ready_score"],
    )
    atomic_json(
        Path(value["terminal_validation_sidecar_path"]),
        dict(publication=publication, reader=reader),
    )
    r.progress("published_" + value["verdict_class"], 1, 0)
    return int(not value["required_checks_passed"])


def main(argv: list[str] | None = None) -> int:
    """Accept only the task date and keep every private process bounded on disk."""
    r.progress("start_no_model_load_MODEL_SPECS_empty_current_LLM_invocations_zero")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261012"], default="20261012")
    parser.add_argument("--root", type=Path, default=r.ROOT)
    parser.add_argument("--output", type=Path, default=r.ROOT / "results" / (r.NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--repeat-request", type=Path)
    parser.add_argument("--history-request", type=Path)
    parser.add_argument("--history-output", type=Path)
    args = parser.parse_args(argv)
    if args.deliberate_error:
        return 7
    if args.history_request:
        if not args.history_output:
            parser.error("history output required")
        return int(r.history_worker(args.history_request, args.history_output))
    if args.repeat_request:
        request = json.loads(args.repeat_request.read_bytes())
        r.progress("benchmark_worker_before", 0, 1)
        value = k.run_repeat(
            request["bundle"],
            request["cell"],
            request["repeat"],
            Path(request["directory"]),
            request["arm"],
        )
        atomic_json(Path(request["output"]), value)
        r.progress("benchmark_worker_after", 1, 0)
        return 0
    if args.cold_replay:
        passed = r.replay(args.cold_replay)
        r.progress("replay_passed" if passed else "replay_failed", int(passed), 0)
        return int(not passed)
    if args.private_control and args.output.resolve().is_relative_to(r.ROOT):
        parser.error("private control outputs must be outside the repository")
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        with TemporaryDirectory(dir="/var/tmp", prefix="exp8449-private-") as folder:
            return run(args.root, args.output, Path(folder), args.private_control)
    except (OSError, ValueError, KeyError, TypeError) as error:
        r.progress("owned_failure_" + str(error))
        return 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)

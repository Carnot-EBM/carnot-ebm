"""REQ-REPORT-8434: bounded private checks qualify atomic terminal publication.

The runner records genuine subprocess exits and coverage shards. A private
control mode prevents a CLI unit test from recursively launching its own tests.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import shutil
import signal
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import capacity_learning_report_8434 as r
from carnot.reporting import block_checkpoint_qualification_8433 as qualification
from carnot.reporting import v726_feature_panel_runner as consumers
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child, execute
from carnot.verify import capacity_online_learning_8434 as e

Json = dict[str, Any]


def manifest(private: Path) -> list[Json]:
    """The shipped plan measures only owned unit and genuine child statements."""
    with patch.object(qualification.shipped, "m", r):
        plan = qualification.shipped.manifest(private)
    config = private / "coverage.ini"
    config.write_text(config.read_text().replace("patch=subprocess", "patch=subprocess,_exit"))
    next(row for row in plan if row["name"] == "coverage_combine")["argv"].append("--keep")
    next(row for row in plan if row["name"] == "owned_tests")["deadline"] = 1200
    for index, (label, test) in enumerate(
        [
            ("020", "test_hard_exit_learning_qualification_8206.py"),
            ("021", "test_restricted_decision_audit_8210.py"),
        ]
    ):
        plan.insert(
            2 + index,
            dict(
                name="private_E2E" + label,
                argv=[
                    str(r.ROOT / ".venv/bin/pytest"),
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    "-q",
                    "tests/python/" + test,
                    "--basetemp=" + str(private / ("e2e" + label)),
                ],
                expected=0,
                deadline=300,
                scope="owned",
            ),
        )
    return list(plan)


def qualify(plan: list[Json], logs: Path) -> list[Json]:
    """Separate private tests can share wall time; coverage reduction waits for every child."""
    names = {"owned_tests", "private_E2E018_consumers", "private_E2E020", "private_E2E021"}
    independent = [spec for spec in plan if spec["name"] in names]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(
                child,
                spec["name"],
                spec["argv"],
                logs,
                expected=spec["expected"],
                deadline=spec["deadline"],
                scope=spec["scope"],
            )
            for spec in independent
        ]
        receipts = [dict(future.result()) for future in futures]
    return [*receipts, *execute([spec for spec in plan if spec["name"] not in names], logs)]


def run(
    root: Path,
    output: Path,
    private: Path,
    *,
    control: bool = False,
    measurement: Path | None = None,
) -> int:
    """Publish qualified evidence or a complete honest failure through unchanged readers."""
    raw = (
        measurement.parent
        if measurement
        else output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    )
    work = r.read(r.reference(measurement)) if measurement else r.measure(root, raw, private)
    if measurement and (raw / "execution_manifest.json").is_file():
        archive = raw / "unfinished_validation" / str(time.time_ns())
        archive.mkdir(parents=True)
        shutil.copyfile(raw / "execution_manifest.json", archive / "execution_manifest.json")
        if (raw / "validation").exists():
            shutil.move(str(raw / "validation"), str(archive / "validation"))
    plan = [] if control else manifest(private)
    atomic_json(
        raw / "execution_manifest.json",
        dict(
            commands=plan,
            owned=r.OWNED,
            task_cap_s=4800,
            measurement_cap_s=2400,
            validation_reserve_s=900,
            full_repository_suite="not_run",
        ),
    )
    receipts = qualify(plan, raw / "validation")
    if (private / "coverage.json").is_file():
        shards = raw / "coverage_shards"
        shards.mkdir()
        for source in private.glob(".coverage*"):
            shutil.copyfile(source, shards / source.name)
        shutil.copyfile(private / "coverage.json", raw / "coverage.json")
        work["coverage_shards"] = [r.reference(p) for p in sorted(shards.iterdir())]
    atomic_json(raw / "stream_rows.json", dict(rows=r.reductions(work)["rows"]))
    atomic_json(raw / "measurement.json", work)
    candidate = raw / "preterminal.json"
    atomic_json(candidate, r.build(work, receipts, raw, output))
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
    changed["learning_stream_ready_score"] = 999
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
    atomic_json(candidate, r.build(work, receipts, raw, output))
    checks, findings = consumers.scan(candidate, raw / "preterminal_validators")
    receipts.extend(checks)
    work.update(adversarial_findings=findings, ended_monotonic_ns=time.monotonic_ns())
    atomic_json(raw / "measurement.json", work)
    value = r.build(work, receipts, raw, output)

    def validator(path: Path) -> Json:
        """Scanner failures remain evidence even when publishing an honest disqualification."""
        checks, flags = consumers.scan(path, raw / "terminal_validators")
        honest_failure = (
            value["verdict_class"] in {"blocked", "disqualified"}
            and value["learning_stream_ready_score"] == 0
        )
        return dict(
            passed=r.replay(path) and (all(c["passed"] for c in checks) or honest_failure),
            findings=flags,
            receipts=checks,
        )

    publication = publish_primary(output, value, validator)
    reader = reader_receipt(
        r.TASK,
        output.parent,
        field="learning_stream_ready_score",
        expected=value["learning_stream_ready_score"],
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
    e.progress("finished", value["completed_count"], value["censored_count"])
    return 0


def main(argv: list[str] | None = None) -> int:
    """Bind dated runs and private evaluator modes without loading a language model."""
    e.progress("start_no_model_load_LLM_invocations_0")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261012"], default="20261012")
    parser.add_argument("--root", type=Path, default=r.ROOT)
    parser.add_argument("--output", type=Path, default=r.ROOT / "results" / (r.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--resume-measurement", type=Path)
    parser.add_argument("--configuration", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--crash", action="store_true")
    parser.add_argument("--release", type=Path)
    parser.add_argument("--retention", type=Path)
    parser.add_argument("--operand-output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    args = parser.parse_args(argv)
    if args.deliberate_error:
        return 7
    if args.cold_replay:
        passed = r.replay(args.cold_replay)
        e.progress("replay_passed" if passed else "reduction_drift")
        return int(not passed)
    if args.worker or args.release or args.retention:
        try:
            cfg = json.loads(args.configuration.read_bytes())
            if args.release:
                issue = json.loads(args.release.read_bytes())
                # Prediction versions precede the transaction which seals them.
                before = dict(issue["state"], version=issue["state"]["version"] - 1)
                if issue["predictions"] != e.predictions(cfg, before, before["cursor"] // 20):
                    raise ValueError("issued_prediction_drift")
                labels = e.released(cfg, issue["state"])
                atomic_json(
                    args.operand_output, dict(labels=labels, released_wall_ns=time.time_ns())
                )
            elif args.retention:
                atomic_json(args.operand_output, e.retention(cfg, e.steps(args.retention)))
            else:
                e.worker(cfg, args.worker, args.crash)
            return 0
        except (OSError, ValueError, KeyError, TypeError, AttributeError, TimeoutError):
            return 1
    output = args.output.absolute()
    if args.private_control and output.is_relative_to(r.ROOT):
        return 2
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        with TemporaryDirectory(prefix="exp8434-private-", dir="/var/tmp") as directory:
            return run(
                args.root,
                output,
                Path(directory),
                control=args.private_control,
                measurement=args.resume_measurement,
            )
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)

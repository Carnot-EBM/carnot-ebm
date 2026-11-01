"""REQ-REPORT-8435: bounded private checks and exact terminal publication.

Control runs exercise actual children without recursively launching the tests.
Production runs require all scoped validation and genuine child coverage.
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

from carnot.reporting import capacity_benefit_report_8435 as r
from carnot.reporting import capacity_learning_runner_8434 as shipped_runner
from carnot.reporting import v717_contract_runner as shipped
from carnot.reporting import v726_feature_panel_runner as consumers
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child

Json = dict[str, Any]


def manifest(private: Path) -> list[Json]:
    """The inherited validation plan freezes genuine child coverage for owned paths only."""
    private.mkdir(parents=True, exist_ok=True)
    with patch.object(shipped, "m", r):
        plan = list(shipped.manifest(private))
    next(p for p in plan if p["name"] == "coverage_combine")["argv"].append("--keep")
    next(p for p in plan if p["name"] == "owned_tests")["deadline"] = 900
    plan.insert(
        2,
        dict(
            name="private_E2E021",
            argv=[
                str(r.ROOT / ".venv/bin/pytest"),
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                "tests/python/test_restricted_decision_audit_8210.py",
                "--basetemp=" + str(private / "e2e021"),
            ],
            expected=0,
            deadline=300,
            scope="owned",
        ),
    )
    return plan


def run(root: Path, output: Path, private: Path, *, control: bool = False) -> int:
    """Publish an honest terminal result after retaining every command and consumer finding."""
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, mode=0o700)
    work = r.measure(root, raw, private)
    plan = [] if control else manifest(private)
    atomic_json(
        raw / "execution_manifest.json",
        dict(
            commands=plan,
            owned=r.OWNED,
            task_cap_s=4800,
            validation_reserve_s=900,
            full_repository_suite="not_run",
        ),
    )
    receipts = shipped_runner.qualify(plan, raw / "validation")
    if (private / "coverage.json").is_file():
        shards = raw / "coverage_shards"
        shards.mkdir()
        for p in private.glob(".coverage*"):
            shutil.copyfile(p, shards / p.name)
        shutil.copyfile(private / "coverage.json", raw / "coverage.json")
        work["coverage_shards"] = [r.reference(p) for p in sorted(shards.iterdir())]
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
            child(name, [*argv, *args], raw / "controls", expected=expected, deadline=240)
        )
    changed = deepcopy(r.read(r.reference(candidate)))
    changed["learning_audit_ready_score"] = 999
    changed["reproducibility_checksum"] = r.checksum(changed)
    tamper = raw / "rehashed_tamper.json"
    atomic_json(tamper, changed)
    receipts.append(
        child(
            "rehashed_tamper",
            [*argv, "--cold-replay", str(tamper)],
            raw / "controls",
            expected=1,
            deadline=240,
        )
    )
    atomic_json(candidate, r.build(work, receipts, raw, output))
    checks, findings = consumers.scan(candidate, raw / "preterminal_validators")
    receipts.extend(checks)
    work.update(adversarial_findings=findings, ended_monotonic_ns=time.monotonic_ns())
    atomic_json(raw / "measurement.json", work)
    value = r.build(work, receipts, raw, output)

    def validator(path: Path) -> Json:
        """Consumer rejection can qualify only honest failure publication, never readiness."""
        checked, flags = consumers.scan(path, raw / "terminal_validators")
        honest_failure = (
            value["verdict_class"] in {"blocked", "disqualified"}
            and value["learning_audit_ready_score"] == 0
        )
        return dict(
            passed=r.replay(path) and (all(c["passed"] for c in checked) or honest_failure),
            findings=flags,
            receipts=checked,
        )

    publication = publish_primary(output, value, validator)
    consumer = reader_receipt(
        r.TASK,
        output.parent,
        field="learning_audit_ready_score",
        expected=value["learning_audit_ready_score"],
    )
    atomic_json(
        Path(value["terminal_validation_sidecar_path"]),
        dict(
            publication=publication,
            consumer=consumer,
            primary_sha256=publication["primary_sha256"],
            adversarial_findings=findings,
            validation_receipts=receipts,
        ),
    )
    r.progress("finished", value["completed_count"], value["censored_count"])
    return 0


def main(argv: list[str] | None = None) -> int:
    """Date and explicit private modes bind each invocation without loading any model."""
    r.progress("start_no_model_load_LLM_invocations_0")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261012"], default="20261012")
    parser.add_argument("--root", type=Path, default=r.ROOT)
    parser.add_argument("--output", type=Path, default=r.ROOT / "results" / (r.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--numeric", type=Path)
    parser.add_argument("--operand-output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    args = parser.parse_args(argv)
    if args.deliberate_error:
        return 7
    if args.cold_replay:
        passed = r.replay(args.cold_replay)
        r.progress("replay_passed" if passed else "reduction_drift")
        return int(not passed)
    if args.numeric:
        try:
            atomic_json(args.operand_output, r.numeric(args.numeric))
            return 0
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return 1
    output = args.output.absolute()
    if args.private_control and output.is_relative_to(r.ROOT):
        return 2
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        with TemporaryDirectory(prefix="exp8435-private-", dir="/var/tmp") as directory:
            return run(args.root, output, Path(directory), control=args.private_control)
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)

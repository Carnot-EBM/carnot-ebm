"""REQ-VERIFY-8432: bounded children qualify private terminal decision bytes.

The shipped serial validation plan supplies genuine child coverage and private
E2E checks. Failed publication findings stay visible in a disqualified result.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import v726_feature_panel_runner as shipped
from carnot.reporting import v726_criterion_decision_audit as e
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child, execute

Json = dict[str, Any]
scan = shipped.scan


def manifest(private: Path) -> list[Json]:
    """Use invocation-only coverage and both frozen private E2E suites."""
    with patch.object(shipped, "e", e):
        plan = list(shipped.manifest(private))
    next(row for row in plan if row["name"] == "owned_tests")["deadline"] = 480
    return plan


def run(root: Path, output: Path, private: Path, *, control: bool = False) -> int:
    """Publish replayable terminal accounting; failed required checks forbid readiness."""
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    work = e.measure(root, raw)
    plan = (
        [
            dict(
                name="private_child",
                argv=[sys.executable, "-u", "-c", "print('actual private control',flush=True)"],
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
            task_cap_s=4800,
            validation_reserve_s=900,
            child_poll_s=30,
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
        work["coverage_shards"] = [e.reference(p) for p in sorted(shards.iterdir())]
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
    changed["brier_gain_vs_logistic"] = 999
    changed["reproducibility_checksum"] = e.checksum(changed)
    tamper = raw / "rehashed_metric_tamper.json"
    atomic_json(tamper, changed)
    receipts.append(
        child(
            "rehashed_metric_tamper",
            [*argv, "--cold-replay", str(tamper)],
            raw / "controls",
            expected=1,
            deadline=120,
        )
    )
    atomic_json(candidate, e.build(work, receipts, raw, output))
    checks, findings = scan(candidate, raw / "preterminal_validators")
    receipts.extend(checks)
    work["adversarial_findings"] = findings
    work["ended_monotonic_ns"] = time.monotonic_ns()
    work["phase_spans"] += [dict(name=r["name"], duration_s=r["duration_s"]) for r in receipts]
    atomic_json(raw / "measurement.json", work)
    value = e.build(work, receipts, raw, output)

    def validator(path: Path) -> Json:
        """A disqualified terminal may preserve failed checks, but cannot claim readiness."""
        checks, flags = scan(path, raw / "terminal_validators")
        terminal_accounting = (
            value["verdict_class"] == "disqualified"
            and value["decision_audit_ready_score"] == 0
            and value["required_checks_passed"] is False
        )
        return dict(
            passed=e.replay(path) and (all(c["passed"] for c in checks) or terminal_accounting),
            publication_qualification_passed=all(c["passed"] for c in checks),
            findings=flags,
            receipts=checks,
        )

    publication = publish_primary(output, value, validator)
    consumer = reader_receipt(
        e.TASK,
        output.parent,
        field="decision_audit_ready_score",
        expected=value["decision_audit_ready_score"],
    )
    atomic_json(
        Path(value["terminal_validation_sidecar_path"]),
        dict(
            publication=publication,
            consumer=consumer,
            primary_sha256=publication["primary_sha256"],
            validation_receipts=receipts,
            adversarial_findings=findings,
        ),
    )
    e.progress("finished", value["completed_count"], value["censored_count"])
    return 0


def main(argv: list[str] | None = None) -> int:
    """Reuse the dated private-output guard without changing the historical CLI."""
    with patch.object(shipped, "e", e), patch.object(shipped, "run", run):
        return int(shipped.main(argv))

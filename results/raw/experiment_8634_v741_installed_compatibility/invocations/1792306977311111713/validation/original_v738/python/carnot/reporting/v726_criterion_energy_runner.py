"""REQ-VERIFY-8431: bind existing bounded controls to the frozen-head consumer.

The shipped validation plan provides real child coverage and both private E2E
checks. This adapter uses the head readiness field for tamper and consumer
checks while retaining unchanged validators and atomic publication.
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
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child, execute
from carnot.verify import v726_criterion_energy_fit as e

Json = dict[str, Any]
scan = shipped.scan


def manifest(private: Path) -> list[Json]:
    """The inherited plan covers owned files and genuine CLI children only."""
    with patch.object(shipped, "e", e):
        return list(shipped.manifest(private))


def run(root: Path, output: Path, private: Path, *, control: bool = False) -> int:
    """Freeze operands, execute controls, and expose only validated terminal bytes."""
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
            child(name, [*argv, *args], raw / "controls", expected=expected, deadline=60)
        )
    changed = deepcopy(json.loads(candidate.read_bytes()))
    changed["head_ready_score"] = 999
    changed["reproducibility_checksum"] = e.checksum(changed)
    tamper = raw / "rehashed_tamper.json"
    atomic_json(tamper, changed)
    receipts.append(
        child(
            "rehashed_tamper",
            [*argv, "--cold-replay", str(tamper)],
            raw / "controls",
            expected=1,
            deadline=60,
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
        """Qualification binds actual scanner reports and cold reduction to final bytes."""
        checks, flags = scan(path, raw / "terminal_validators")
        return dict(
            passed=e.replay(path) and all(c["passed"] for c in checks),
            findings=flags,
            receipts=checks,
        )

    publication = publish_primary(output, value, validator)
    consumer = reader_receipt(
        e.TASK,
        output.parent,
        field="head_ready_score",
        expected=value["head_ready_score"],
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
    """The dated parser keeps the existing private-output and replay failure guards."""
    with patch.object(shipped, "e", e), patch.object(shipped, "run", run):
        return int(shipped.main(argv))

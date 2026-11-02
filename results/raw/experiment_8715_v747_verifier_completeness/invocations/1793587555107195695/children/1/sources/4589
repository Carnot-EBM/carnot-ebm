"""REQ-VERIFY-8580: supervise qualification and publish only checked final bytes.

The existing process and publication helpers retain their original behavior.
Private fixtures exercise the same path without recursive validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import count_validator_runner_8567 as previous
from carnot.reporting import current_validator_8580 as e
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
CLI = e.ROOT / "scripts/experiments" / (e.NAME + ".py")


def progress(phase: str) -> None:
    """Keep phase progress visible even when no model or device is required."""
    print("[exp8580] " + phase, flush=True)


def prepare(raw: Path) -> Json:
    """Reuse original resource checks and data adapters without historical validation."""
    progress("start date=20261016 no_model_load MODEL_SPECS=[] current_llm_calls=0")
    started = time.monotonic_ns()
    work = previous.prepare(raw)
    work["started_monotonic_ns"] = started
    work.update(cli_rows=[], authority={})
    try:
        authority = e.authenticate()
        work["authority"] = authority
        work["sources"].extend(
            [
                e.reference(e.ROOT / "results" / (e.old.NAME + ".json")),
                e.reference(
                    e.ROOT / "results/experiment_8568_v736_causal_accounting_qualification.json"
                ),
                *[e.reference(e.AUTHORITY / name) for name in e.REPAIR_PINS],
            ]
        )
        deltas = authority["deltas"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        deltas = []
        work["failures"].append(
            dict(
                check="current_revision_authority",
                upstream=e.old.TASK,
                path=str(e.AUTHORITY),
                sha256=None,
                field="authenticated_repair",
                operator="==",
                expected=True,
                observed=None if isinstance(error, FileNotFoundError) else str(error),
            )
        )
    work["code"] = [
        e.reference(e.ROOT / p)
        for p in [
            *e.OWNED,
            *e.TESTS,
            *e.REPAIRED,
            "python/carnot/reporting/discrete_count_proof.py",
            "python/carnot/reporting/causal_accounting_8568.py",
            "python/carnot/reporting/primary_publication.py",
            *sorted(
                {
                    n.split("::")[0]
                    for n in work["authority"].get("regression_nodes", [])
                    + work["authority"].get("affected_nodes", [])
                }
            ),
        ]
    ]
    atomic_json(raw / "repair.diff.json", deltas)
    seal(work)
    progress("authority_and_resource_checks_complete")
    return work


def seal(work: Json) -> None:
    """Write primitives before the immutable revision descriptor references their hash."""
    raw = Path(work["raw"])
    atomic_json(raw / "work.json", work)
    atomic_json(raw / "current_validator_revision.json", e.revision(work))


def terminal(path: Path, raw: Path) -> Json:
    """Retain every finding while checking exact candidate bytes before replacement."""
    progress("terminal_validation_before")
    scan = e.old.av.verify_artifact(path)
    lint = child(
        "final_row_lint",
        [sys.executable, "scripts/verdict_row_consistency_lint.py", "--strict", str(path)],
        raw / "terminal_logs",
        deadline=60,
    )
    cold = child(
        "final_cold",
        [sys.executable, str(CLI), "--cold-replay", str(path)],
        raw / "terminal_logs",
        deadline=120,
    )
    value = json.loads(path.read_bytes())
    expression = (
        "from carnot.reporting.current_validator_8580 import accounting_revision; assert accounting_revision(current_validator_reference="
        + repr(value["current_validator_reference"])
        + ")['passed']"
    )
    accounting = child(
        "final_accounting_revision",
        [sys.executable, "-c", expression],
        raw / "terminal_logs",
        expected=0 if value["current_validator_ready_score"] else 1,
        deadline=120,
    )
    atomic_json(raw / "final_adversarial.json", scan)
    progress("terminal_validation_after")
    return dict(
        passed=cold["passed"]
        and accounting["passed"]
        and lint["passed"]
        and not any(r["severity"] == "critical" for r in scan["flags"]),
        adversarial=scan,
        row_lint=lint,
        accounting=accounting,
        cold=cold,
    )


def run(output: Path, *, fixture: bool = False) -> int:
    """Production validates once; private fixtures cannot claim current readiness."""
    start = time.monotonic()
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    work = prepare(raw)
    with TemporaryDirectory(dir="/var/tmp", prefix="carnot-8580-") as temporary:
        from carnot.reporting.current_validator_validation_8580 import replay_controls, validate

        if fixture:
            replay_controls(work, Path(temporary))
        else:
            validate(work, Path(temporary))
        work["duration_s"] = time.monotonic() - start
        work["phases"].append(
            dict(
                phase="current_validation",
                duration_s=work["duration_s"] - work["phases"][0]["duration_s"],
            )
        )
        seal(work)
        value = e.build(work, output)
        publication = publish_primary(output, value, lambda p: terminal(p, raw))
        consumer = reader_receipt(
            e.TASK,
            output.parent,
            field="current_validator_ready_score",
            expected=value["current_validator_ready_score"],
        )
        atomic_json(
            raw / "terminal_validation.json", dict(publication=publication, consumer=consumer)
        )
    progress("complete verdict=" + value["verdict_class"])
    return 0


def main(argv: list[str] | None = None) -> int:
    """Keep test destinations private and reject a different invocation date."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261016")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--fixture-output", type=Path)
    args = parser.parse_args(argv)
    if args.date != "20261016" or (
        args.fixture_output and args.fixture_output.resolve().is_relative_to(e.ROOT / "results")
    ):
        return 2
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        progress("cold_replay=" + str(passed))
        return 0 if passed else 1
    return run(
        args.fixture_output or e.ROOT / "results" / (e.NAME + ".json"),
        fixture=args.fixture_output is not None,
    )

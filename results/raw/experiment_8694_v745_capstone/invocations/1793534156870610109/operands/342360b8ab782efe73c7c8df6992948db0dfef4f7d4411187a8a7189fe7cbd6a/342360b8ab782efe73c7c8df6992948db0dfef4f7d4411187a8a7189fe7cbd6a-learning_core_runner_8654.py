"""REQ-VERIFY-8654: publish only sealed bounded core execution with a fresh replay."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import signal
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import learning_core_8654 as e
from carnot.reporting import learning_core_report_8654 as report
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from scripts.adversarial_verify import verify_artifact

Json = dict[str, Any]


def replay(path: Path) -> bool:
    """An independently retained seal prevents repaired summary hashes from gaining authority."""
    try:
        value = json.loads(path.read_bytes())
        work = e.c.read(value["primitive_reference"])
        sealed = e.c.read(work["invocation_seal_reference"])
        if sealed != {k: v for k, v in work.items() if k != "invocation_seal_reference"}:
            return False
        for receipt in work["receipts"]:
            for stream in ("stdout", "stderr"):
                if (
                    e.c.ref(Path(receipt[stream + "_path"]))["sha256"]
                    != receipt[stream + "_sha256"]
                ):
                    return False
        with TemporaryDirectory(prefix="cold8654-", dir="/var/tmp") as directory:
            e.recompute(work, Path(directory))
        return report.build(work, Path(value["publication_output"])) == value
    except (OSError, ValueError, KeyError, TypeError):
        return False


def terminal(path: Path, raw: Path) -> Json:
    """Keep actual scanner severity and strict row results beside a fresh bounded CLI replay."""
    findings = json.loads(path.read_bytes())["adversarial_findings"]
    receipts = []
    for name, args, expected in [
        (
            "adversarial",
            ["scripts/adversarial_verify.py", "--json", str(path)],
            int(bool(findings)),
        ),
        ("rows", ["scripts/verdict_row_consistency_lint.py", "--strict", str(path)], 0),
        ("cold", [e.CLI, "--cold-replay", str(path)], 0),
    ]:
        receipts.append(
            e.c.supervise(
                name,
                [sys.executable, "-u", *args],
                raw / "terminal",
                expected=expected,
                deadline=180,
            )
        )
    scan = json.loads(Path(receipts[0]["stdout_path"]).read_bytes())["reports"][0]
    return dict(
        passed=all(r["passed"] for r in receipts)
        and scan["flags"] == findings
        and not any(f["severity"] == "critical" for f in findings),
        adversarial=scan,
        receipts=receipts,
    )


def validation(work: Json) -> None:
    """Validate only owned code and the two unchanged private consumer CLI nodes."""
    raw = Path(work["raw"])
    e.progress("validation_900_seconds_reserved")
    e.instrument(work, "owned_tests", e.TESTS, [e.c.ref(e.ROOT / p) for p in e.OWNED], 360, "owned")
    checks = [
        ("ruff_check", ["-m", "ruff", "check", *e.OWNED, *e.TESTS]),
        ("ruff_format", ["-m", "ruff", "format", "--check", *e.OWNED, *e.TESTS]),
        ("mypy", ["-m", "mypy", "--strict", "--follow-imports=silent", *e.OWNED]),
        ("spec", ["scripts/check_spec_coverage.py", "--files", *e.TESTS]),
        (
            "private_e2e018",
            [
                "-m",
                "pytest",
                *e.OPTIONS,
                "tests/python/test_experiment_7891_v685_authority_lifecycle.py::test_cli_private_success_failure_and_replay",
            ],
        ),
        (
            "private_e2e021",
            [
                "-m",
                "pytest",
                *e.OPTIONS,
                "tests/python/test_restricted_decision_audit_8210.py::test_private_cli_and_rehashed_mutations",
            ],
        ),
    ]
    for index, (name, argv) in enumerate(checks):
        e.progress("validation_checks", index, len(checks) - index)
        work["receipts"].append(
            e.c.supervise(name, [sys.executable, *argv], raw / "checks", deadline=120)
        )


def freeze(work: Json) -> None:
    """Store a separate read-only primitive seal before building any reader-visible summary."""
    raw = Path(work["raw"])
    seal = raw / "invocation_seal.json"
    e.atomic_json(seal, {k: v for k, v in work.items() if k != "invocation_seal_reference"})
    seal.chmod(0o444)
    work["invocation_seal_reference"] = e.c.ref(seal)
    e.atomic_json(raw / "work.json", work)


def run(
    output: Path, run_date: str, upstream: Path = e.UPSTREAM, *, private_control: bool = False
) -> int:
    """Measure the registered core once, then validate immutable bytes before publication."""
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    work = e.prepare(raw, run_date, upstream)
    if not work["input_failures"] and not private_control:
        e.execute(work)
    if not private_control:
        validation(work)
    with TemporaryDirectory(prefix="core8654-combine-", dir="/var/tmp") as directory:
        e.recompute(work, Path(directory))
    work["duration_s"] = (time.monotonic_ns() - work["started_monotonic_ns"]) / 1e9
    work["phase_spans"] = [
        dict(
            phase="core_execution_and_validation",
            start_monotonic_ns=work["started_monotonic_ns"],
            end_monotonic_ns=work["started_monotonic_ns"] + round(work["duration_s"] * 1e9),
            duration_s=work["duration_s"],
        )
    ]
    freeze(work)
    candidate = raw / "preterminal.json"
    e.atomic_json(candidate, report.build(work, output))
    work["adversarial_findings"] = verify_artifact(candidate)["flags"]
    freeze(work)
    value = report.build(work, output)
    publication = publish_primary(output, value, lambda path: terminal(path, raw))
    selected = reader_receipt(
        e.TASK,
        output.parent,
        field="learning_core_ready_score",
        expected=value["learning_core_ready_score"],
    )
    e.atomic_json(
        Path(value["terminal_validation_sidecar_path"]),
        dict(publication=publication, selected_identity=selected),
    )
    e.progress("normal_exit_" + value["verdict_class"], 1, 0)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Accept one calendar date while keeping private controls outside shared evidence."""
    print("[exp8654] start no_model_load MODEL_SPECS=[] current_llm_calls=0", flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261019")
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--upstream", type=Path, default=e.UPSTREAM)
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    args = parser.parse_args(argv)
    try:
        datetime.strptime(args.date, "%Y%m%d")
    except ValueError:
        return 2
    if (
        args.date != "20261019"
        or (args.private_output and args.private_output.resolve().is_relative_to(e.ROOT))
        or (args.private_control and not args.private_output)
    ):
        return 2
    if args.deliberate_error:
        return 7
    if args.cold_replay:
        return int(not replay(args.cold_replay))
    timer = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        return run(
            args.private_output or e.ROOT / "results" / (e.NAME + ".json"),
            args.date,
            args.upstream,
            private_control=args.private_control,
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, *timer)

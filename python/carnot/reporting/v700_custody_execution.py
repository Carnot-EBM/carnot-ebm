"""REQ-VERIFY-8083: freeze bounded checks before custody and publish checked bytes.

Validation children write private scratch. Durable logs are copied only after
normal exit, so tests cannot rewrite the repository's historical evidence.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from carnot.reporting import v700_contract_custody as e
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v686_contract_validation import (
    CONSUMERS,
    coverage_complete,
    dependency_hashes,
    run_check,
)

Json = dict[str, Any]
OWNED = [e.MODULE, e.RUNNER, e.CLI]


def manifest(private: Path, candidate: Path) -> Json:
    """Fix command bytes before inputs open; full repository health stays diagnostic."""
    py, cov, pytest, ruff, mypy = [
        str(e.ROOT / ".venv/bin" / n) for n in ["python", "coverage", "pytest", "ruff", "mypy"]
    ]
    config = private / "coverage.ini"
    config.write_text(
        "[run]\nparallel = true\ndata_file = "
        + str(private / ".coverage")
        + "\ninclude =\n"
        + "".join("    " + str(e.ROOT / p) + "\n" for p in OWNED)
    )
    common = ["-n", "0", "-o", "addopts=", "--no-cov", "-q"]
    rc = "--rcfile=" + str(config)
    include = "--include=" + ",".join(str(e.ROOT / p) for p in OWNED)
    commands = [
        (
            "focused_E2E018",
            [
                cov,
                "run",
                rc,
                "-m",
                "pytest",
                *common,
                e.TEST,
                "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
            ],
            180,
        ),
        (
            "consumer_tests",
            [pytest, *common, *CONSUMERS, "tests/python/test_primary_publication_7928.py"],
            180,
        ),
        ("coverage_combine", [cov, "combine", rc], 30),
        ("coverage_report", [cov, "report", rc, include, "--show-missing", "--fail-under=100"], 30),
        ("coverage_json", [cov, "json", rc, include, "-o", str(private / "coverage.json")], 30),
        ("ruff_check", [ruff, "check", *OWNED, e.TEST], 30),
        ("ruff_format", [ruff, "format", "--check", *OWNED, e.TEST], 30),
        ("strict_mypy", [mypy, "--strict", "--follow-imports=silent", *OWNED], 60),
        ("scoped_spec_coverage", [py, "scripts/check_spec_coverage.py", e.TEST], 30),
    ]
    terminal = [
        ("cold_replay", [py, "-u", str(e.ROOT / e.CLI), "--cold-replay", str(candidate)], 60),
        (
            "adversarial_verify",
            [py, str(e.ROOT / "scripts/adversarial_verify.py"), "--json", str(candidate)],
            60,
        ),
        (
            "strict_row_lint",
            [
                py,
                str(e.ROOT / "scripts/verdict_row_consistency_lint.py"),
                "--strict",
                str(candidate),
            ],
            60,
        ),
    ]
    pack = lambda rows: [
        dict(name=n, argv=a, deadline_s=t, expected_exit=0, classification="required")
        for n, a, t in rows
    ]
    return dict(
        commands=pack(commands),
        terminal_commands=pack(terminal),
        coverage_config=str(config),
        repository_health=dict(
            name="repository_full_suite",
            argv=[pytest, "tests/python", "-q"],
            deadline_s=180,
            expected_exit=0,
            classification="diagnostic",
        ),
    )


def publish(value: Json, output: Path, private: Path, raw: Path) -> None:
    """Expose bytes only after normal validator exits and save their hash binding."""

    def validator(candidate: Path) -> Json:
        commands = json.loads((raw / "validation_commands.json").read_text())["terminal_commands"]
        reports = []
        for spec in commands:
            e.progress("before_" + spec["name"])
            reports.append(run_check(e.ROOT, spec, private, raw / "terminal_logs", heartbeat_s=20))
            e.progress("after_" + spec["name"])
        return dict(passed=all(r["passed"] for r in reports), checks=reports)

    publication = publish_primary(output, value, validator)
    atomic_json(
        raw / "terminal_validation.json",
        dict(
            publication=publication,
            normal_process_exit=all(
                r.get("exit_code", 0) == 0 for r in value["validation_receipts"]
            ),
            readers=reader_receipt(
                e.TASK,
                output.parent,
                field="contract_ready_score",
                expected=value["contract_ready_score"],
            ),
        ),
    )


def main(argv: list[str] | None = None) -> int:
    """Run a zero-model measurement child, private controls or immutable cold replay."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    e.progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261004"], default="20261004")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--design", type=Path, default=e.ROOT / e.DESIGN)
    parser.add_argument("--staged", type=Path, default=e.ROOT / "research-roadmap-next.yaml")
    parser.add_argument("--active", type=Path, default=e.ROOT / "research-roadmap.yaml")
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--fixture-output", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--mutate", action="store_true")
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        e.progress("replay_passed" if passed else "reduction_drift")
        return 0 if passed else 1
    if args.worker_output:
        work = e.measure(
            args.root, args.design, args.staged, args.active, args.worker_output.parent
        )
        atomic_json(args.worker_output, work)
        e.progress("measurement_normal_exit")
        return 0
    output = (args.fixture_output or args.output).absolute()
    raw = output.parent / "raw" / output.stem / "invocations" / str(__import__("time").time_ns())
    raw.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="carnot-8083-") as directory:
        private = Path(directory)
        candidate = output.parent / "raw" / output.stem / "terminal_candidate.json"
        specs = manifest(private, candidate)
        measurement = dict(
            name="measurement",
            argv=[
                str(e.ROOT / ".venv/bin/python"),
                "-u",
                str(e.ROOT / e.CLI),
                "--root",
                str(args.root),
                "--design",
                str(args.design),
                "--staged",
                str(args.staged),
                "--active",
                str(args.active),
                "--worker-output",
                str(raw / "measurement.json"),
            ],
            deadline_s=120,
            expected_exit=0,
            classification="required",
        )
        specs["measurement_command"] = measurement
        atomic_json(raw / "validation_commands.json", specs)
        if args.fixture_output:
            work = e.measure(args.root, args.design, args.staged, args.active, raw, fixture=True)
            log = raw / "fixture_control.log"
            log.write_text("Completed private custody controls; no scientific evidence.\n")
            receipts = [
                dict(
                    name="private_oracle_control",
                    passed=True,
                    scope="fixture_only",
                    log_path=str(log),
                    log_sha256=e.sha256_file(log),
                )
            ]
        else:
            e.progress("before_measurement")
            receipts = [
                run_check(e.ROOT, measurement, private, raw / "validation_logs", heartbeat_s=20)
            ]
            e.progress("after_measurement")
            work = json.loads((raw / "measurement.json").read_text())
            os.environ["CARNOT_8083_COVERAGE_CONFIG"] = specs["coverage_config"]
            for index, spec in enumerate(specs["commands"]):
                e.progress("before_" + spec["name"], index, len(specs["commands"]) - index)
                receipts.append(
                    run_check(e.ROOT, spec, private, raw / "validation_logs", heartbeat_s=20)
                )
                e.progress("after_" + spec["name"], index + 1, len(specs["commands"]) - index - 1)
            del os.environ["CARNOT_8083_COVERAGE_CONFIG"]
            proof = private / "coverage.json"
            receipts.append(
                dict(
                    name="added_statement_coverage", passed=coverage_complete(proof, includes=OWNED)
                )
            )
            if proof.is_file():
                counts = json.loads(proof.read_text())
                atomic_json(raw / "coverage.json", counts)
                work["coverage_statement_counts"] = {
                    p: counts["files"].get(p, {}).get("summary", {}) for p in OWNED
                }
            e.progress("before_repository_health")
            work["repository_health"] = [
                run_check(
                    e.ROOT, specs["repository_health"], private, raw / "health_logs", heartbeat_s=20
                )
            ]
            e.progress("after_repository_health")
        if args.mutate:
            receipts.append(
                dict(name="owned_mutation_control", passed=False, exit_code=1, expected_exit=0)
            )
        work["code_hashes"] = dependency_hashes(e.ROOT, paths=[*OWNED, e.TEST])
        code_binder = e.Binder(raw / "code")
        work["code_snapshots"] = {
            label: code_binder.bind(e.ROOT / label, digest)
            for label, digest in work["code_hashes"].items()
        }
        atomic_json(raw / "work.json", work)
        value = e.build(work, raw, receipts)
        e.progress("before_publication")
        publish(value, output, private, raw)
        e.progress("after_publication", 14, 0)
    return 0

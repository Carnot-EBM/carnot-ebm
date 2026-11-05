"""REQ-VERIFY-8825: reuse scoped checks and atomic candidate publication."""

import json
from pathlib import Path
import sys
import tempfile
from typing import Any

from coverage import Coverage

from carnot.reporting import hardware_operation_map_8825 as m
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary

Json = dict[str, Any]


def validate(raw: Path) -> Json:
    """Use real unit and subprocess traces, with an empty exclusion configuration."""
    private = Path(tempfile.mkdtemp(prefix="carnot8825-validation-"))
    config = private / "coverage.ini"
    config.write_text(
        "[run]\nparallel=true\npatch=subprocess\ndata_file="
        + str(private / ".coverage")
        + "\ninclude=\n"
        + "".join("    " + str(m.ROOT / p) + "\n" for p in m.OWNED)
        + "[report]\nexclude_lines=\n"
    )
    py = str(m.ROOT / ".venv/bin/python")
    common = ["-n", "0", "-o", "addopts=", "--no-cov", "-q"]
    plan = [
        CommandSpec(
            "owned_pytest",
            tuple(
                [
                    py,
                    "-m",
                    "coverage",
                    "run",
                    "--rcfile=" + str(config),
                    "-m",
                    "pytest",
                    *common,
                    m.TEST,
                ]
            ),
            "owned",
        ),
        CommandSpec(
            "private_e2e018_021",
            tuple(
                [
                    py,
                    "-m",
                    "pytest",
                    *common,
                    "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
                    "tests/python/test_restricted_decision_audit_8210.py",
                ]
            ),
            "owned",
        ),
    ]
    for name, args in [
        ("ruff_check", [str(m.ROOT / ".venv/bin/ruff"), "check", *m.OWNED, m.TEST]),
        ("ruff_format", [str(m.ROOT / ".venv/bin/ruff"), "format", "--check", *m.OWNED, m.TEST]),
        (
            "strict_mypy",
            [str(m.ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=silent", *m.OWNED],
        ),
        ("spec_coverage", [py, "scripts/check_spec_coverage.py", m.TEST]),
    ]:
        plan.append(CommandSpec(name, tuple(args), "owned"))
    plan.append(
        CommandSpec(
            "global_python_health",
            (str(m.ROOT / ".venv/bin/pytest"), "tests/python", "-q"),
            "repository_health",
            timeout_s=180,
        )
    )
    atomic_json(
        raw / "validation_manifest.json",
        dict(
            commands=[dict(name=s.name, argv=s.argv, scope=s.scope) for s in plan],
            owned=m.OWNED,
            tests=[m.TEST],
        ),
    )
    checks = run_commands(m.ROOT, plan, log_dir=raw / "checks", heartbeat_s=30)
    cov = Coverage(data_file=str(private / ".coverage"), config_file=str(config))
    cov.combine(data_paths=[str(private)], keep=True)
    cov.save()
    report = raw / "coverage.json"
    cov.json_report(outfile=str(report))
    data = json.loads(report.read_bytes())
    complete = len(data["files"]) == len(m.OWNED) and all(
        f["summary"]["num_statements"] > 0 and not f["missing_lines"] and not f["excluded_lines"]
        for f in data["files"].values()
    )
    coverage = dict(
        percent=data["totals"]["percent_covered"],
        complete=complete,
        zero_exclusions=all(not f["excluded_lines"] for f in data["files"].values()),
        path=str(report),
        sha256=sha256_file(report),
        files=data["files"],
    )
    result = dict(
        passed=complete and all(c["passed"] for c in checks if c["scope"] == "owned"),
        coverage=coverage,
        receipts=checks,
    )
    atomic_json(raw / "validation.json", result)
    return result


def terminal(candidate: Path, raw: Path) -> Json:
    """Run unchanged scanners and an actual cold replay before atomic publication."""
    py = str(m.ROOT / ".venv/bin/python")
    commands = [
        CommandSpec(
            "adversarial", (py, "scripts/adversarial_verify.py", str(candidate), "--json"), "owned"
        ),
        CommandSpec(
            "row_consistency",
            (py, "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "owned",
        ),
        CommandSpec(
            "cold_replay",
            (
                "env",
                "-u",
                "PYTHONPATH",
                "-C",
                "/tmp",
                py,
                "-u",
                str(m.ROOT / m.CLI),
                "--replay",
                str(candidate),
            ),
            "owned",
        ),
    ]
    checks = run_commands(m.ROOT, commands, log_dir=raw / "terminal", heartbeat_s=30)
    result = dict(passed=all(c["passed"] for c in checks), receipts=checks)
    atomic_json(raw / "terminal.json", result)
    return result


def publish(output: Path, value: Json, raw: Path, *, private: bool) -> None:
    """Fixtures use replay only; real publication must also pass both unchanged scanners."""
    publish_primary(output, value, lambda p: m.replay(p) if private else terminal(p, raw))

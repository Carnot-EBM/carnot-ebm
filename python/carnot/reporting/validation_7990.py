"""REQ-REPORT-7990: freeze scoped validation before read-only audit work.

Existing supervision keeps progress active and archives logs after children
exit. New statement coverage includes actual private script-path execution.
"""

from datetime import UTC, datetime
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import experiment_7990_v692_hardware_evidence as q
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.experiment_7913_v686_hardware_evidence import disqualify, run_child
from carnot.reporting.primary_publication import publish_primary, reader_receipt

ROOT = q.ROOT
MODULE = "python/carnot/reporting/experiment_7990_v692_hardware_evidence.py"
PLAN = "python/carnot/reporting/validation_7990.py"
SCRIPT = "scripts/experiments/experiment_7990_v692_hardware_evidence.py"
TEST = "tests/python/test_experiment_7990_v692_hardware_evidence.py"
OWNED = (MODULE, PLAN, SCRIPT)
TESTS = (
    TEST,
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_service_cost_7989.py",
)
INCLUDE = "--include=" + ",".join(str(ROOT / name) for name in OWNED)
Json = dict[str, Any]


def command(
    name: str, argv: list[str], classification: str = "required", deadline: int = 60
) -> Json:
    """Make expected exits explicit before launching any owned subprocess."""
    return dict(
        name=name,
        argv=argv,
        classification=classification,
        deadline_s=deadline,
        expected_exit=0,
        required_reason=None,
    )


def manifest(private: Path) -> list[Json]:
    """Private coverage files and fixtures cannot mutate checked-in artifacts."""
    if not private.is_relative_to(Path("/tmp")) or private.is_relative_to(ROOT):
        raise ValueError("private_tmp_required")
    python = str(ROOT / ".venv/bin/python")
    cov = [python, "-m", "coverage"]
    data = "--data-file=" + str(private / ".coverage")
    paths = [*OWNED, TEST]
    return [
        command(
            "owned_tests",
            [
                "/usr/bin/env",
                "CARNOT_7990_COVERAGE_DIR=" + str(private),
                *cov,
                "run",
                "--parallel-mode",
                data,
                INCLUDE,
                "-m",
                "pytest",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "--basetemp=" + str(private / "tests"),
                "-q",
                *TESTS,
            ],
            deadline=300,
        ),
        command("coverage_combine", [*cov, "combine", data, str(private)]),
        command("changed_coverage", [*cov, "report", data, INCLUDE, "--fail-under=100"]),
        command(
            "coverage_json", [*cov, "json", data, INCLUDE, "-o", str(private / "coverage.json")]
        ),
        command("ruff_check", [str(ROOT / ".venv/bin/ruff"), "check", *paths]),
        command("ruff_format", [str(ROOT / ".venv/bin/ruff"), "format", "--check", *paths]),
        command(
            "strict_mypy",
            [str(ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=silent", *OWNED],
        ),
        command("spec_coverage", [python, "scripts/check_spec_coverage.py", *TESTS]),
        command(
            "full_pytest",
            [str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"],
            "repository_health",
            300,
        ),
    ]


def terminal_manifest(private: Path) -> list[Json]:
    """Cold replay and terminal tools inspect the same final candidate bytes."""
    python = str(ROOT / ".venv/bin/python")
    candidate = str(private / "terminal_candidate.json")
    return [
        command(
            "adversarial",
            [python, "-u", "scripts/adversarial_verify.py", "--json", candidate],
            "terminal_validator",
        ),
        command(
            "strict_rows",
            [python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", candidate],
            "terminal_validator",
        ),
        command(
            "cold_replay",
            [
                "/usr/bin/env",
                "-u",
                "PYTHONPATH",
                "-C",
                str(private),
                python,
                "-u",
                str(ROOT / SCRIPT),
                "--cold-replay",
                candidate,
            ],
            "terminal_validator",
        ),
    ]


def qualify(root: Path, run_date: str, output: Path) -> int:
    """Required failures close readiness while unrelated full-suite health stays separate."""
    if root.resolve() != ROOT:
        raise ValueError("worktree_root_required")
    started = time.monotonic()
    utc_start = datetime.now(UTC).isoformat()
    output = output.absolute()
    raw = output.parent / "raw" / output.stem
    with TemporaryDirectory(prefix="carnot-7990-", dir="/tmp") as folder:
        private = Path(folder)
        commands, terminal = manifest(private), terminal_manifest(private)
        manifest_path = raw / "validation_command_manifest.json"
        atomic_json(
            manifest_path,
            dict(
                task_id="exp7990-hardware-evidence",
                commands=commands,
                terminal_commands=terminal,
                code_config_hashes=[
                    dict(path=name, sha256=sha256_file(ROOT / name)) for name in (*OWNED, *TESTS)
                ],
                coverage_include=INCLUDE,
                result_guard="enabled",
                historical_preflight=[
                    dict(path=str(path), sha256=sha256_file(path))
                    for path in sorted(
                        (
                            ROOT / "results/raw/experiment_7990_v692_hardware_evidence/preflight"
                        ).glob("*/custody.json")
                    )
                ],
            ),
        )
        print("[exp7990] phase=validation manifest_frozen before_audit", flush=True)
        value = q.read_evidence(root, run_date)
        receipts = [run_child(spec, private, raw, started, i) for i, spec in enumerate(commands)]
        required = [r for r in receipts if r["classification"] == "required"]
        passed = all(r["passed"] for r in required)
        value.update(
            validation_command_manifest_path=str(manifest_path),
            validation_receipts=dict(checks=receipts, required_checks_passed=passed),
            observed_child_commands=[r["command_argv"] for r in receipts if "command_argv" in r],
            terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
            primary_resolution_receipt=dict(path=str(raw / "primary_resolution.json")),
            scratch_root_receipt=dict(
                path=str(private), outside_checkout=True, archived_after_children_exit=True
            ),
            started_at=utc_start,
            finished_at=datetime.now(UTC).isoformat(),
            duration_s=time.monotonic() - started,
        )
        value["repository_health"] = dict(
            current=receipts[-1],
            historical=value["repository_health"],
            affects_required_checks=False,
        )
        if (private / "coverage.json").is_file():
            coverage = json.loads((private / "coverage.json").read_text())
            atomic_json(raw / "coverage.json", coverage)
            value["coverage_statement_counts"] = {
                name: dict(
                    covered=r["summary"]["covered_lines"], statements=r["summary"]["num_statements"]
                )
                for name, r in coverage["files"].items()
            }
        value["gate_check_summary"] += [
            dict(
                upstream_id="owned_validation",
                artifact_path=r["log_path"],
                artifact_hash=r["log_sha256"],
                artifact_field=r["name"],
                op="==",
                expected=0,
                observed=r["exit_code"],
                passed=False,
            )
            for r in required
            if not r["passed"]
        ]
        if not passed:
            disqualify(value, "owned_required_validation")
        value["phase_spans"]["owned_validation_s"] = (
            value["duration_s"] - value["phase_spans"]["evidence_read_s"]
        )
        value["field_principles"].update(
            {k: "Bind validation to actual child exits and exact final bytes." for k in value}
        )

        def validator(candidate: Path) -> Json:
            q.cold_reduce(root, json.loads(candidate.read_text()))
            reports = []
            for index, spec in enumerate(terminal):
                spec = dict(
                    spec,
                    argv=[
                        arg.replace(str(private / "terminal_candidate.json"), str(candidate))
                        for arg in spec["argv"]
                    ],
                )
                reports.append(run_child(spec, private, raw, started, index))
            report = dict(
                passed=all(r["passed"] for r in reports),
                candidate_sha256=sha256_file(candidate),
                reports=reports,
            )
            atomic_json(raw / "terminal_validation.json", report)
            return report

        publish_primary(output, value, validator)
        receipt = reader_receipt(
            value["task_id"],
            output.parent,
            field="hardware_evidence_ready_score",
            expected=value["hardware_evidence_ready_score"],
        )
        if not receipt["passed"]:
            raise ValueError("reader_identity")
        atomic_json(raw / "primary_resolution.json", receipt)
    print("[exp7990] phase=publication checked_primary_published", flush=True)
    return 0

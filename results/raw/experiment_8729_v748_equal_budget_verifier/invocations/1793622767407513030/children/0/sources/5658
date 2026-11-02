"""Freeze explicit private commands before validation consumes any results.

Spec ref: REQ-REPORT-7913-V686. Scratch ownership keeps active result guards
from redirecting an atomic rename onto a different filesystem.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
MODULE = "python/carnot/reporting/experiment_7913_v686_hardware_evidence.py"
PLAN = "python/carnot/reporting/validation_7913.py"
SCRIPT = "scripts/experiments/experiment_7913_v686_hardware_evidence.py"
TEST = "tests/python/test_experiment_7913_v686_hardware_evidence.py"
CONSUMERS = (
    "tests/python/test_experiment_7901_v685_hardware_evidence.py",
    "tests/python/test_experiment_7889_v684_hardware_evidence.py",
    "tests/python/test_experiment_7876_v683_hardware_evidence.py",
    "tests/python/test_experiment_7862_v682_hardware_evidence.py",
)
MEASURED = (MODULE, PLAN, SCRIPT)
LIBRARIES = (
    "python/carnot/experiment_artifacts.py",
    "python/carnot/paths.py",
    "python/carnot/terminal_artifacts.py",
    "python/carnot/reporting/experiment_7793_v677_hardware_evidence.py",
    "python/carnot/reporting/experiment_7806_v678_hardware_evidence.py",
    "python/carnot/reporting/experiment_7820_v679_hardware_evidence.py",
    "python/carnot/reporting/experiment_7834_v680_hardware_evidence.py",
    "python/carnot/testing/operator_curated_doc_guard.py",
    "python/carnot/testing/pytest_basetemp_isolation.py",
    "python/carnot/testing/pytest_memory_watchdog.py",
    "python/carnot/testing/worktree_import_guard.py",
    "python/carnot/verify/context_sufficiency_7854.py",
    "python/carnot/verify/intervention_protocol_7868.py",
    "python/carnot/verify/source_alignment.py",
    "python/carnot/verify/source_interventions.py",
    "pyproject.toml",
    "AGENTS.md",
    "CODEX.md",
    "ops/exclusion_manifest.yaml",
    "openspec/capabilities/research-reporting/spec.md",
    "scripts/test_suite_mutation_check.py",
    "python/carnot/reporting/current_work_receipt.py",
    "python/carnot/reporting/experiment_7303_validation_scope.py",
    "python/carnot/reporting/experiment_7847_v681_hardware_evidence.py",
    "python/carnot/reporting/experiment_7862_v682_hardware_evidence.py",
    "python/carnot/reporting/experiment_7876_v683_hardware_evidence.py",
    "python/carnot/reporting/experiment_7889_v684_hardware_evidence.py",
    "python/carnot/reporting/experiment_7901_v685_hardware_evidence.py",
    "scripts/experiments/experiment_7834_v680_hardware_evidence.py",
    "scripts/experiments/experiment_7847_v681_hardware_evidence.py",
    "scripts/experiments/experiment_7862_v682_hardware_evidence.py",
    "scripts/experiments/experiment_7876_v683_hardware_evidence.py",
    "scripts/experiments/experiment_7889_v684_hardware_evidence.py",
    "scripts/experiments/experiment_7901_v685_hardware_evidence.py",
    "scripts/experiments/experiment_7868_v683_intervention_protocol.py",
    "python/carnot/experiment_7868_v683_intervention_protocol.py",
    "scripts/adversarial_verify.py",
    "scripts/verdict_row_consistency_lint.py",
    "scripts/check_spec_coverage.py",
    "python/carnot/testing/child_results_guard.py",
    "python/carnot/testing/tracked_results_guard.py",
    "tests/python/conftest.py",
)


def command(
    name: str,
    argv: list[str],
    *,
    deadline: int = 30,
    expected: int = 0,
    reason: str | None = None,
    classification: str = "required",
) -> dict[str, Any]:
    """Expected negative exits need their rejection reason as well as their code."""
    return {
        "name": name,
        "argv": argv,
        "deadline_s": deadline,
        "classification": classification,
        "expected_exit": expected,
        "required_reason": reason,
    }


def manifest(private: Path, run_date: str) -> list[dict[str, Any]]:
    """SCENARIO-REPORT-7913-PRIVATE: no scratch path may live below results."""
    if not private.resolve().is_relative_to(Path("/tmp")):
        raise ValueError("private_tmp_required")
    py, pytest, cov, ruff, mypy = (
        str(ROOT / ".venv/bin" / name) for name in ("python", "pytest", "coverage", "ruff", "mypy")
    )
    include = ",".join(str(ROOT / name) for name in MEASURED)
    common = ["-q", "-n", "0", "-o", "addopts=", "--no-cov"]
    shards = [
        private / f"{name}.coverage"
        for name in ("unit", "success", "missing", "replay", "negative")
    ]
    commands = [
        command(
            "affected_pytest",
            [pytest, TEST, *CONSUMERS, *common, f"--basetemp={private / 'affected'}"],
            deadline=180,
        ),
        command(
            "unit_coverage",
            [
                cov,
                "run",
                f"--data-file={shards[0]}",
                f"--include={include}",
                "-m",
                "pytest",
                TEST,
                *common,
                f"--basetemp={private / 'unit-temp'}",
            ],
            deadline=180,
        ),
    ]
    for name, shard, root, output in (
        ("cli_success", shards[1], ROOT, "success.json"),
        ("cli_missing_input", shards[2], private / "absent", "missing.json"),
    ):
        commands.append(
            command(
                name,
                [
                    cov,
                    "run",
                    f"--data-file={shard}",
                    f"--include={include}",
                    str(ROOT / SCRIPT),
                    "--date",
                    run_date,
                    "--root",
                    str(root),
                    "--output",
                    str(private / output),
                    "--evidence-only",
                ],
            )
        )
    for name, shard, candidate, expected, reason in (
        ("cold_replay", shards[3], "success.json", 0, None),
        ("negative_replay", shards[4], "changed.json", 1, "claims_changed"),
    ):
        commands.append(
            command(
                name,
                [
                    cov,
                    "run",
                    f"--data-file={shard}",
                    f"--include={include}",
                    str(ROOT / SCRIPT),
                    "--date",
                    run_date,
                    "--root",
                    str(ROOT),
                    "--cold-replay",
                    str(private / candidate),
                ],
                expected=expected,
                reason=reason,
            )
        )
    combined = private / "combined.coverage"
    commands.extend(
        [
            command(
                "coverage_shards",
                [
                    py,
                    "-u",
                    "-c",
                    "from pathlib import Path; import sys; from carnot.reporting.experiment_7913_v686_hardware_evidence import check_coverage_shards; check_coverage_shards([Path(x) for x in sys.argv[1:]])",
                    *(str(p) for p in shards),
                ],
            ),
            command(
                "coverage_combine",
                [cov, "combine", "--keep", f"--data-file={combined}", *(str(p) for p in shards)],
            ),
            command(
                "changed_coverage",
                [
                    cov,
                    "report",
                    f"--data-file={combined}",
                    f"--include={include}",
                    "--show-missing",
                    "--fail-under=100",
                ],
            ),
            command("ruff_check", [ruff, "check", *MEASURED, TEST]),
            command("ruff_format", [ruff, "format", "--check", *MEASURED, TEST]),
            command("mypy", [mypy, "--strict", *MEASURED], deadline=60),
            command("scoped_spec", [py, "-u", "scripts/check_spec_coverage.py", TEST, *CONSUMERS]),
        ]
    )
    fixture = private / "e2e-016-fixture.json"
    for name, route in (("e2e_016_fixture", "--fixture-e2e"), ("e2e_016_replay", "--cold-replay")):
        commands.append(
            command(
                name,
                [
                    py,
                    "-u",
                    "scripts/experiments/experiment_7868_v683_intervention_protocol.py",
                    "--date",
                    run_date,
                    route,
                    str(fixture),
                ],
                deadline=180,
            )
        )
    commands.append(
        command(
            "full_pytest",
            [pytest, "tests/python", *common, f"--basetemp={private / 'full-temp'}"],
            deadline=120,
            classification="diagnostic",
        )
    )
    return commands


def terminal_manifest(private: Path) -> list[dict[str, Any]]:
    """Both readers inspect the exact candidate that publication will copy."""
    py = str(ROOT / ".venv/bin/python")
    candidate = str(private / "terminal-candidate.json")
    return [
        command(
            "adversarial_verify",
            [py, "-u", "scripts/adversarial_verify.py", "--json", candidate],
            classification="terminal_validator",
        ),
        command(
            "strict_rows",
            [py, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", candidate],
            classification="terminal_validator",
        ),
    ]

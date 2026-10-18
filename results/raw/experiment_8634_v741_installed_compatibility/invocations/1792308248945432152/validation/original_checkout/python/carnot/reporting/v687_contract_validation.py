"""Adapt bounded validation without copying a dispatcher (REQ-REPORT-7915-V687)."""

import gzip
import json
from pathlib import Path
from typing import Any

from carnot.reporting import v686_contract_validation as shared
from carnot.reporting import v687_contract_methods as methods

OWNED = [
    "python/carnot/reporting/v687_contract_methods.py",
    "python/carnot/reporting/v687_contract_validation.py",
    "scripts/experiments/experiment_7915_v687_contract_methods.py",
]
CHANGED = [*OWNED, *shared.CHANGED[:2], shared.CHANGED[-1]]
TESTS = ["tests/python/test_experiment_7915_v687_contract_methods.py", *shared.TESTS]


def prepare(root: Path, private: Path) -> None:
    """Copy frozen oracle authority into private space before any child starts."""
    private.mkdir(parents=True, exist_ok=True)
    for name in ("design.md", "active.yaml"):
        (private / name).write_bytes(
            gzip.decompress((root / "tests/fixtures/v687" / f"{name}.gz").read_bytes())
        )
    shared.atomic_json(private / "negative.json", {"experiment_id": 0})
    shared.atomic_json(private / "negative-rows.json", {"rows": []})


def manifest(
    root: Path, private: Path, *, repository_health_receipt: Path | None = None
) -> dict[str, Any]:
    """Freeze the reused argv with current modules and historical E2E dates."""
    value = shared.manifest(root, private)
    before = "--include=" + ",".join(str(root / name) for name in shared.CHANGED)
    after = "--include=" + ",".join(str(root / name) for name in CHANGED)
    for row in value["commands"]:
        row["argv"] = [
            arg.replace(before, after).replace(
                "experiment_7903_v686_contract_methods.py",
                "experiment_7915_v687_contract_methods.py",
            )
            for arg in row["argv"]
        ]
        if row["name"] in {"affected_pytest", "coverage_unit", "scoped_spec_coverage"}:
            row["argv"] = [arg for arg in row["argv"] if arg not in shared.TESTS] + TESTS
        if row["name"] in {"ruff_check", "ruff_format", "mypy_strict"}:
            row["argv"] = row["argv"][: (3 if row["name"] == "ruff_format" else 2)] + CHANGED
            if row["name"] != "mypy_strict":
                row["argv"].append(TESTS[0])
        if row["name"].startswith("e2e_016"):
            row["argv"][row["argv"].index("--date") + 1] = "20260929"
        if row["name"] == "repository_full_suite":
            row["argv"] = [str(root / ".venv/bin/pytest"), "tests/python", "-q"]
            row["deadline_s"] = 600
        if row["name"] == "coverage_combine":
            row["argv"].append(str(private / ".coverage.terminal"))
    terminal = {
        "name": "coverage_terminal_recheck",
        "argv": [
            str(root / ".venv/bin/coverage"),
            "run",
            f"--data-file={private / '.coverage.terminal'}",
            after,
            str(root / "scripts/experiments/experiment_7915_v687_contract_methods.py"),
            "--terminal-recheck",
            str(private / "fixture.json"),
            "--output",
            str(private / "checked.json"),
        ],
        "expected_exit": 0,
        "deadline_s": 120,
        "failure_reason": "terminal_recheck_failed",
        "classification": "required",
    }
    value["commands"].insert(
        next(i for i, row in enumerate(value["commands"]) if row["name"] == "coverage_combine"),
        terminal,
    )
    value["commands"].extend(
        [
            {
                "name": "e2e_015",
                "argv": [
                    str(root / ".venv/bin/pytest"),
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    "-q",
                    f"--basetemp={private / 'source-e2e'}",
                    "tests/python/test_source_boundary_7852.py",
                ],
                "expected_exit": 0,
                "deadline_s": 120,
                "failure_reason": "source_fixture_failed",
                "classification": "required",
            },
            {
                "name": "e2e_016_wrong_date",
                "argv": [
                    str(root / ".venv/bin/python"),
                    str(root / "scripts/experiments/experiment_7868_v683_intervention_protocol.py"),
                    "--date",
                    "20260930",
                    "--fixture-e2e",
                    str(private / "wrong-date.json"),
                ],
                "expected_exit": 1,
                "deadline_s": 60,
                "failure_reason": "run_date_mismatch",
                "classification": "required",
            },
        ]
    )
    value.update(
        affected_tests=TESTS,
        coverage_includes=CHANGED,
        dependency_hashes=shared.dependency_hashes(root, paths=CHANGED + TESTS + shared.CONSUMERS),
        historical_fixture_date="20260929",
        execution_date="20260930",
        e2e_018="Private 12/13-task lifecycle and current CLI assertions; historical publishing is inapplicable",
    )
    value["dependency_hashes"].update(
        {
            name: shared.sha256_file(root / name)
            for name in (
                "tests/fixtures/v687/design.md.gz",
                "tests/fixtures/v687/active.yaml.gz",
                "research-references.md",
                "research-studying.md",
                "research-program.md",
                "ops/exclusion_manifest.yaml",
                "openspec/capabilities/research-reporting/spec.md",
            )
        }
    )
    if repository_health_receipt:
        observed = json.loads(repository_health_receipt.read_text())
        expected = [str(root / ".venv/bin/pytest"), "tests/python", "-q"]
        if (
            observed["argv"] != expected
            or shared.sha256_file(Path(observed["log_path"])) != observed["log_sha256"]
            or shared.sha256_file(Path(observed["prior_candidate_path"]))
            != observed["prior_candidate_sha256"]
        ):
            raise ValueError("repository_health_receipt_drift")
        value["repository_health_prior_observation"] = observed
        value["repository_health_prior_receipt_sha256"] = shared.sha256_file(
            repository_health_receipt
        )
        value["commands"] = [
            row for row in value["commands"] if row["name"] != "repository_full_suite"
        ]
    return value


def coverage_complete(report: Path) -> bool:
    """Reject empty or missing per-file statement counts in the current scope."""
    return shared.coverage_complete(report, includes=CHANGED)


def execute(root: Path, args: Any, private: Path) -> dict[str, Any]:
    """Reuse supervision and atomic publication with current producer adapters."""
    health = getattr(args, "repository_health_receipt", None)

    def frozen_manifest(root: Path, private: Path) -> dict[str, Any]:
        return (
            manifest(root, private, repository_health_receipt=health)
            if health
            else manifest(root, private)
        )

    return shared.execute(
        root,
        args,
        private,
        methods_module=methods,
        prepare_fn=prepare,
        manifest_fn=frozen_manifest,
        coverage_fn=coverage_complete,
        experiment_id=7915,
        count=13,
    )

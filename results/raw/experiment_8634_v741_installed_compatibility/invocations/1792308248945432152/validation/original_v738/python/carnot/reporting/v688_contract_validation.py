"""Reuse bounded checks with only current code in the coverage scope.

REQ-REPORT-7929-V688: historical qualification is custody, not a new run.
"""

import json
from pathlib import Path
from typing import Any

from carnot.reporting import v686_contract_validation as shared
from carnot.reporting import v687_contract_validation as prior
from carnot.reporting import v688_contract_methods as methods
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import reader_receipt

OWNED = [
    "python/carnot/reporting/v688_contract_methods.py",
    "python/carnot/reporting/v688_contract_validation.py",
    "scripts/experiments/experiment_7929_v688_contract_methods.py",
]
TESTS = ["tests/python/test_experiment_7929_v688_contract_methods.py", *prior.TESTS]


def prepare(root: Path, private: Path) -> None:
    """Observe the current authority for private routes without rewriting live files."""
    private.mkdir(parents=True, exist_ok=True)
    (private / "design.md").write_bytes(
        (root / "openspec/change-proposals/research-roadmap-vNEXT.md").read_bytes()
    )
    (private / "active.yaml").write_bytes((root / "research-roadmap.yaml").read_bytes())
    atomic_json(private / "negative.json", {"experiment_id": 0})
    atomic_json(private / "negative-rows.json", {"rows": []})


def manifest(root: Path, private: Path) -> dict[str, Any]:
    """Freeze current includes while preserving historical required assertions."""
    value = prior.manifest(root, private)
    before = "--include=" + ",".join(str(root / name) for name in prior.CHANGED)
    after = "--include=" + ",".join(str(root / name) for name in OWNED)
    for row in value["commands"]:
        row["argv"] = [
            arg.replace(before, after).replace(
                "experiment_7915_v687_contract_methods.py",
                "experiment_7929_v688_contract_methods.py",
            )
            for arg in row["argv"]
        ]
        if row["name"] in {"affected_pytest", "coverage_unit", "scoped_spec_coverage"}:
            row["argv"] = [arg for arg in row["argv"] if arg not in prior.TESTS] + TESTS
        if row["name"] in {"ruff_check", "ruff_format", "mypy_strict"}:
            row["argv"] = row["argv"][: 3 if row["name"] == "ruff_format" else 2] + OWNED
            if row["name"] != "mypy_strict":
                row["argv"].append(TESTS[0])
    value.update(
        affected_tests=TESTS,
        coverage_includes=OWNED,
        dependency_hashes=shared.dependency_hashes(root, paths=OWNED + TESTS + shared.CONSUMERS),
    )
    value["dependency_hashes"].update(
        {
            name: sha256_file(root / name)
            for name in (
                "research-roadmap.yaml",
                "openspec/change-proposals/research-roadmap-vNEXT.md",
                "research-references.md",
                "research-studying.md",
                "research-program.md",
                "ops/exclusion_manifest.yaml",
                "openspec/capabilities/research-reporting/spec.md",
            )
        }
    )
    value["e2e_018"] = (
        "Private current twelve-task CLI plus preserved twelve/thirteen-task assertions"
    )
    return value


def coverage_complete(report: Path) -> bool:
    """Require nonempty statement coverage for every new module and CLI."""
    return shared.coverage_complete(report, includes=OWNED)


def execute(root: Path, args: Any, private: Path) -> dict[str, Any]:
    """Reuse child supervision and bind final reader receipts outside the primary."""
    value = shared.execute(
        root,
        args,
        private,
        methods_module=methods,
        prepare_fn=prepare,
        manifest_fn=manifest,
        coverage_fn=coverage_complete,
        experiment_id=7929,
        count=12,
    )
    if not args.fixture_e2e:
        terminal = args.raw.parent / "terminal_validation"
        digest = sha256_file(args.output)
        report = json.loads((terminal / f"terminal-{digest[7:]}.json").read_text())
        atomic_json(terminal / "terminal_validation.json", report)
        selected = reader_receipt(
            "exp7929-contract-methods",
            args.output.parent,
            field="contract_ready_score",
            expected=value["contract_ready_score"],
        )
        if selected["gate_sha256"] != digest or selected["document_sha256"] != digest:
            raise ValueError("primary_reader_drift")
        atomic_json(args.raw.parent / "primary_resolution_receipt.json", selected)
    return value

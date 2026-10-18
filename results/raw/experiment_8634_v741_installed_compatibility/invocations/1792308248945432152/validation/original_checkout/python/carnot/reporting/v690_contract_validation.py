"""REQ-REPORT-7953-V690: reuse bounded validation and unique primary publication."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import time
from typing import Any

from carnot.reporting import v686_contract_validation as shared
from carnot.reporting import v689_contract_validation as prior
from carnot.reporting import v690_contract_methods as methods
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import publish_primary, reader_receipt

OWNED = [
    "python/carnot/reporting/v690_authority.py",
    "python/carnot/reporting/v690_contract_methods.py",
    "python/carnot/reporting/v690_contract_validation.py",
    "scripts/experiments/experiment_7953_v690_contract_methods.py",
]
TESTS = [
    "tests/python/test_experiment_7953_v690_contract_methods.py",
    "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
    "tests/python/test_primary_publication_7928.py",
]


def prepare(root: Path, private: Path) -> None:
    """Copy actual independent input bytes without inventing failure history."""
    private.mkdir(parents=True, exist_ok=True)
    for target, source in (
        ("design.md", "openspec/change-proposals/research-roadmap-vNEXT.md"),
        ("active.yaml", "research-roadmap.yaml"),
    ):
        (private / target).write_bytes((root / source).read_bytes())
    (private / "stage.yaml").write_bytes((private / "active.yaml").read_bytes())
    atomic_json(private / "negative.json", {"experiment_id": 0})
    atomic_json(private / "negative-rows.json", {"rows": []})


def manifest(
    root: Path, private: Path, *, repository_health_receipt: Path | None = None
) -> dict[str, Any]:
    """Reuse command definitions with current files, independent routes and fixed dates."""
    value = prior.manifest(root, private)
    value["commands"] = [r for r in value["commands"] if r["name"] != "repository_full_suite"]
    health_path = (
        repository_health_receipt
        or root / "results/raw/experiment_7953_v690_contract_methods/repository_health/receipt.json"
    )
    value["repository_health_current_receipt"] = (
        json.loads(health_path.read_text()) if health_path.is_file() else None
    )
    before = "--include=" + ",".join(str(root / n) for n in prior.OWNED)
    after = "--include=" + ",".join(str(root / n) for n in OWNED)
    for row in value["commands"]:
        row["argv"] = [
            a.replace(before, after)
            .replace("7940_v689", "7953_v690")
            .replace("7940_fixture", "7953_fixture")
            for a in row["argv"]
        ]
        if row["name"] in {"affected_pytest", "coverage_unit", "scoped_spec_coverage"}:
            row["argv"] = [a for a in row["argv"] if a not in prior.TESTS + TESTS] + TESTS
        if row["name"] in {"ruff_check", "ruff_format", "mypy_strict"}:
            row["argv"] = row["argv"][: 3 if row["name"] == "ruff_format" else 2] + OWNED
            if row["name"] != "mypy_strict":
                row["argv"].append(TESTS[0])
    success = next(r for r in value["commands"] if r["name"] == "coverage_cli_success")
    consumed = deepcopy(success)
    consumed.update(name="e2e_018_consumed_staging", failure_reason="consumed_staging_regression")
    consumed["argv"] = [
        a.replace(".coverage.cli", ".coverage.consumed")
        .replace("success/", "consumed/")
        .replace("stage.yaml", "absent-staging.yaml")
        for a in consumed["argv"]
    ]
    value["commands"].insert(value["commands"].index(success) + 1, consumed)
    combined = next(r for r in value["commands"] if r["name"] == "coverage_combine")
    combined["argv"].append(str(private / ".coverage.consumed"))
    value.update(
        affected_tests=TESTS,
        coverage_includes=OWNED,
        dependency_hashes=shared.dependency_hashes(root, paths=OWNED + TESTS + shared.CONSUMERS),
        e2e_018="private lifecycle tests and real staged/consumed V690 CLI routes",
    )
    value["dependency_hashes"].update(
        {
            n: sha256_file(root / n)
            for n in (
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
    return value


def coverage_complete(report: Path) -> bool:
    """Reject empty coverage or missing statements in every added module and CLI."""
    return shared.coverage_complete(report, includes=OWNED)


def publish(root: Path, value: dict[str, Any], output: Path, raw: Path, private: Path) -> None:
    """Validate terminal bytes before the locked publisher makes them reader-visible."""
    terminal = raw.parent / "terminal_validation"
    checked = private / "checked" / output.name
    shared.publish(root, value, checked, private / "terminal", terminal)
    atomic_json(raw, methods.primitive_rows(value))
    digest = sha256_file(checked)
    report = json.loads((terminal / f"terminal-{digest[7:]}.json").read_text())
    report["published_path"] = str(output)
    publication = publish_primary(
        output,
        value,
        lambda path: dict(
            passed=report["passed"]
            and sha256_file(path) == digest
            and methods.cold_replay(path, raw),
            terminal_validation_path=str(terminal / "terminal_validation.json"),
        ),
    )
    atomic_json(terminal / "terminal_validation.json", {**report, **publication})
    selected = reader_receipt(
        "exp7953-contract-methods",
        output.parent,
        field="contract_ready_score",
        expected=value["contract_ready_score"],
    )
    if selected["gate_sha256"] != digest or selected["document_sha256"] != digest:
        raise ValueError("primary_reader_drift")
    atomic_json(raw.parent / "primary_resolution_receipt.json", selected)


def execute(root: Path, args: Any, private: Path) -> dict[str, Any]:
    """Freeze code-bound inputs and supervise every required child before publishing."""
    started = time.monotonic_ns()
    print("[exp7953] phase=inputs_start completed_units=0 elapsed_s=0", flush=True)
    prepare(root, private)
    durable = args.raw.parent
    assessment = methods.assess(
        args.design, args.staged, args.active, durable / "authority_snapshots"
    )
    custody = methods.source_custody(args.source, methods.SOURCE_SHA256)
    freeze = methods.method_freeze(root)
    frozen = manifest(root, private)
    frozen["input_hashes"] = dict(
        authorities=assessment["authority_snapshots"], source_custody=custody["hashes"]
    )
    manifest_path = durable / "validation_command_manifest.json"
    atomic_json(manifest_path, frozen)
    atomic_json(durable / "method_freeze.json", freeze)
    atomic_json(
        durable / "input_checkpoint.json",
        dict(
            manifest_sha256=sha256_file(manifest_path),
            method_freeze_sha256=sha256_file(durable / "method_freeze.json"),
            resume_requires_identical_dependencies=True,
        ),
    )
    print(
        f"[exp7953] phase=inputs_frozen completed_units=3 elapsed_s={(time.monotonic_ns() - started) / 1e9:.3f}",
        flush=True,
    )
    controls = methods.mutations(
        private / "design.md", private / "active.yaml", args.source, private / "mutations"
    )
    receipts = []
    if not args.fixture_e2e:
        receipts = [
            shared.run_check(root, spec, private, durable / "sealed_logs")
            for spec in frozen["commands"]
        ]
        if frozen.get("repository_health_current_receipt"):
            receipts.append(frozen["repository_health_current_receipt"])
        report = private / "coverage.json"
        complete = coverage_complete(report)
        receipts.append(
            dict(
                name="per_file_statement_coverage",
                argv=["coverage_json_reduction", str(report)],
                expected_exit=0,
                actual_exit=int(not complete),
                passed=complete,
                deadline_s=0,
            )
        )
        if report.is_file():
            shutil.copyfile(report, durable / f"coverage.json-{sha256_file(report)[7:]}")
    value = methods.candidate(
        root,
        assessment,
        custody,
        freeze,
        controls,
        receipts,
        manifest_path,
        started,
        time.monotonic_ns(),
        fixture=args.fixture_e2e,
    )
    atomic_json(args.raw, methods.primitive_rows(value))
    if args.fixture_e2e:
        atomic_json(args.output, value)
    else:
        publish(root, value, args.output, args.raw, private)
    print(
        f"[exp7953] phase=published completed_units=13 elapsed_s={(time.monotonic_ns() - started) / 1e9:.3f}",
        flush=True,
    )
    return value

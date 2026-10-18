"""REQ-REPORT-7979-V692: freeze owned checks and publish their observed result."""

import json
from pathlib import Path
import shutil
import time
from typing import Any

from carnot.reporting import v686_contract_validation as shared
from carnot.reporting import v691_contract_validation as prior
from carnot.reporting import v692_contract_methods as methods
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import publish_primary, reader_receipt

OWNED = [
    "python/carnot/reporting/v692_contract_methods.py",
    "python/carnot/reporting/v692_contract_validation.py",
    "scripts/experiments/experiment_7979_v692_contract_methods.py",
]
TESTS = ["tests/python/test_experiment_7979_v692_contract_methods.py", shared.TESTS[1]]
prepare = prior.prepare
run_check = shared.run_check


def manifest(root: Path, private: Path) -> dict[str, Any]:
    """Keep identical includes on real unit, success, blocked and negative CLI routes."""
    value = shared.manifest(root, private)
    before = "--include=" + ",".join(str(root / n) for n in shared.CHANGED)
    after = "--include=" + ",".join(str(root / n) for n in OWNED)
    value["commands"] = [r for r in value["commands"] if not r["name"].startswith("e2e_016")]
    for row in value["commands"]:
        row["argv"] = [
            a.replace(before, after).replace("7903_v686", "7979_v692") for a in row["argv"]
        ]
        if "--date" in row["argv"]:
            row["argv"][row["argv"].index("--date") + 1] = "20261001"
        if row["name"] in {"affected_pytest", "coverage_unit", "scoped_spec_coverage"}:
            row["argv"] = [a for a in row["argv"] if a not in shared.TESTS + TESTS] + TESTS
        if row["name"] in {"ruff_check", "ruff_format", "mypy_strict"}:
            row["argv"] = row["argv"][: 3 if row["name"] == "ruff_format" else 2] + OWNED
            if row["name"] != "mypy_strict":
                row["argv"].append(TESTS[0])
        if row["name"] == "coverage_expected_failure":
            row["argv"].append("--expect-rejection")
            row["expected_exit"] = 0
        if row["name"] == "repository_full_suite":
            row["argv"] = [str(root / ".venv/bin/pytest"), "tests/python", "-q"]
    success = next(r for r in value["commands"] if r["name"] == "coverage_cli_success")
    blocked = json.loads(json.dumps(success))
    blocked.update(name="coverage_cli_blocked")
    blocked["argv"] = [
        a.replace(".coverage.cli", ".coverage.blocked")
        .replace("fixture.json", "blocked.json")
        .replace("rows.json", "blocked-rows.json")
        for a in blocked["argv"]
    ]
    blocked["argv"] += ["--source", str(private / "missing-source.json")]
    value["commands"].insert(value["commands"].index(success) + 1, blocked)
    next(r for r in value["commands"] if r["name"] == "coverage_combine")["argv"].append(
        str(private / ".coverage.blocked")
    )
    value["commands"].append(
        dict(
            name="e2e_018_private",
            argv=[
                str(root / ".venv/bin/pytest"),
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                f"--basetemp={private / 'authority-e2e'}",
                TESTS[1],
            ],
            expected_exit=0,
            deadline_s=120,
            classification="required",
        )
    )
    value.update(
        coverage_includes=OWNED,
        affected_tests=TESTS,
        scratch_root=str(private),
        dependency_hashes=shared.dependency_hashes(root, paths=OWNED + TESTS + shared.CONSUMERS),
    )
    durable = root / "results/raw/experiment_7979_v692_contract_methods"
    value["prior_owned_attempts"] = [
        dict(path=str(p), sha256=sha256_file(p), role="failed_owned_attempt")
        for p in sorted((durable / "owned_attempts").glob("*/primary.json"))
    ]
    health = durable / "repository_health/receipt.json"
    if health.is_file():
        next(r for r in value["commands"] if r["name"] == "repository_full_suite")[
            "reused_receipt"
        ] = json.loads(health.read_text())
    return value


def coverage_complete(report: Path) -> bool:
    """Missing coverage or empty files cannot qualify added statements."""
    return shared.coverage_complete(report, includes=OWNED)


def publish(root: Path, value: dict[str, Any], output: Path, raw: Path, private: Path) -> None:
    """Both unchanged consumers must resolve the exact bytes checked by validators."""
    terminal = raw.parent / "terminal_validation"
    checked = private / "checked" / output.name
    shared.publish(root, value, checked, private / "terminal", terminal)
    atomic_json(raw, methods.primitive_rows(value))
    digest = sha256_file(checked)
    report = json.loads((terminal / f"terminal-{digest[7:]}.json").read_text())
    publication = publish_primary(
        output,
        value,
        lambda path: dict(
            passed=report["passed"]
            and sha256_file(path) == digest
            and methods.cold_replay(path, raw)
        ),
    )
    atomic_json(terminal / "terminal_validation.json", {**report, **publication})
    selected = reader_receipt(
        "exp7979-contract-methods",
        output.parent,
        field="contract_ready_score",
        expected=value["contract_ready_score"],
    )
    if selected["gate_sha256"] != digest or selected["document_sha256"] != digest:
        raise ValueError("primary_reader_drift")
    atomic_json(raw.parent / "primary_resolution_receipt.json", selected)


def execute(root: Path, args: Any, private: Path) -> dict[str, Any]:
    """Freeze input bytes before running children and archive only exited receipts."""
    started = time.monotonic_ns()
    print("[exp7979] phase=inputs_start completed_units=0", flush=True)
    prepare(root, private)
    durable = args.raw.parent
    assessment = methods.assess(
        args.design, args.staged, args.active, durable / "authority_snapshots"
    )
    custody = methods.source_custody(args.source, methods.SOURCE_SHA256)
    import yaml

    tasks = yaml.safe_load((private / "active.yaml").read_bytes())["tasks"]
    freeze = methods.method_freeze(root, tasks=tasks)
    frozen = manifest(root, private)
    frozen["input_hashes"] = dict(
        authorities=assessment["authority_snapshots"],
        source_custody=custody["hashes"],
        historical=methods.historical_custody(root),
    )
    manifest_path = durable / f"validation_command_manifest-{canonical_hash(frozen)[7:]}.json"
    atomic_json(manifest_path, frozen)
    atomic_json(durable / "method_freeze.json", freeze)
    print("[exp7979] phase=inputs_frozen completed_units=3", flush=True)
    controls = methods.mutations(
        private / "design.md", private / "active.yaml", args.source, private / "mutations"
    )
    receipts = [
        dict(
            name="private_fixture",
            argv=["in_process_private_fixture"],
            passed=all(r["passed"] for r in controls),
            actual_exit=0,
        )
    ]
    if not args.fixture_e2e:
        receipts = [
            {
                **spec["reused_receipt"],
                "current_execution": False,
                "reuse_reason": "repository health measured once; immutable exited receipt",
            }
            if "reused_receipt" in spec
            else run_check(root, spec, private, durable / "sealed_logs")
            for spec in frozen["commands"]
        ]
        report = private / "coverage.json"
        complete = coverage_complete(report)
        receipts.append(
            dict(
                name="per_file_statement_coverage",
                argv=["coverage_json_reduction", str(report)],
                expected_exit=0,
                actual_exit=int(not complete),
                passed=complete,
            )
        )
        for name in ("coverage.json", ".coverage.combined"):
            path = private / name
            if path.is_file():
                shutil.copyfile(path, durable / f"{name}-{sha256_file(path)[7:]}")
    print(f"[exp7979] phase=checks_complete completed_units={len(receipts)}", flush=True)
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
    )
    for report in durable.glob("coverage.json-*"):
        value["coverage_statement_counts"] = {
            name: row["summary"] for name, row in json.loads(report.read_text())["files"].items()
        }
    atomic_json(args.raw, methods.primitive_rows(value))
    if args.fixture_e2e:
        atomic_json(args.output, value)
    else:
        publish(root, value, args.output, args.raw, private)
    print(
        f"[exp7979] phase=published completed_units=13 elapsed_s={(time.monotonic_ns() - started) / 1e9:.3f}",
        flush=True,
    )
    return value

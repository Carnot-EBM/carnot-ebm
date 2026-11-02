"""REQ-REPORT-7940-V689: freeze commands and publish only validated current bytes."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import time
from typing import Any

import yaml

from carnot.reporting import v686_contract_validation as shared
from carnot.reporting import v687_contract_validation as prior
from carnot.reporting import v689_contract_methods as methods
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import publish_primary, reader_receipt

OWNED = [
    "python/carnot/reporting/v689_contract_methods.py",
    "python/carnot/reporting/v689_contract_validation.py",
    "scripts/experiments/experiment_7940_v689_contract_methods.py",
]
TESTS = ["tests/python/test_experiment_7940_v689_contract_methods.py", *prior.TESTS]


def prepare(root: Path, private: Path) -> None:
    """Declare synthetic prior history only in a private passing mutation oracle."""
    private.mkdir(parents=True, exist_ok=True)
    design = (root / "openspec/change-proposals/research-roadmap-vNEXT.md").read_text()
    active = yaml.safe_load((root / "research-roadmap.yaml").read_bytes())
    original = methods.shared.lifecycle.tasks_digest(active["tasks"])
    for task in active["tasks"]:
        if not task["prior_failures"]:
            task["prior_failures"] = [
                dict(
                    experiment_id="private-fixture",
                    verdict="complete_null_fixture",
                    addressed_by="synthetic private oracle; no live history repair",
                    retire_if_same_verdict=True,
                )
            ]
    design = design.replace(original, methods.shared.lifecycle.tasks_digest(active["tasks"]), 1)
    (private / "design.md").write_text(design)
    (private / "active.yaml").write_text(yaml.safe_dump(active, sort_keys=False))
    (private / "stage.yaml").write_bytes((private / "active.yaml").read_bytes())
    atomic_json(private / "negative.json", {"experiment_id": 0})
    atomic_json(private / "negative-rows.json", {"rows": []})


def manifest(root: Path, private: Path) -> dict[str, Any]:
    """Retain required historical checks while isolating each publication scenario."""
    value = prior.manifest(root, private)
    before = "--include=" + ",".join(str(root / name) for name in prior.CHANGED)
    after = "--include=" + ",".join(str(root / name) for name in OWNED)
    routes = {
        "fixture.json": "success/experiment_7940_fixture.json",
        "checked.json": "terminal/experiment_7940_fixture.json",
        "rows.json": "success/rows.json",
        "consumed.yaml": "stage.yaml",
    }
    for row in value["commands"]:
        row["argv"] = [
            arg.replace(before, after).replace(
                "experiment_7915_v687_contract_methods.py",
                "experiment_7940_v689_contract_methods.py",
            )
            for arg in row["argv"]
        ]
        for index, arg in enumerate(row["argv"]):
            for old, new in routes.items():
                if arg == str(private / old):
                    row["argv"][index] = str(private / new)
        if row["name"] in {"affected_pytest", "coverage_unit", "scoped_spec_coverage"}:
            row["argv"] = [arg for arg in row["argv"] if arg not in prior.TESTS] + TESTS
        if row["name"] in {"ruff_check", "ruff_format", "mypy_strict"}:
            row["argv"] = row["argv"][: 3 if row["name"] == "ruff_format" else 2] + OWNED
            if row["name"] != "mypy_strict":
                row["argv"].append(TESTS[0])
    success = next(row for row in value["commands"] if row["name"] == "coverage_cli_success")
    blocked = deepcopy(success)
    blocked.update(name="coverage_cli_blocked", failure_reason="blocked_route_failed")
    blocked["argv"] = [
        arg.replace(".coverage.cli", ".coverage.blocked")
        .replace("success/experiment_7940_fixture.json", "blocked/experiment_7940_fixture.json")
        .replace("success/rows.json", "blocked/rows.json")
        for arg in blocked["argv"]
    ]
    blocked["argv"] += ["--source", str(private / "missing-source.json")]
    value["commands"].insert(value["commands"].index(success) + 1, blocked)
    blocked_replay = deepcopy(
        next(row for row in value["commands"] if row["name"] == "coverage_cold_replay")
    )
    blocked_replay.update(name="coverage_blocked_replay", failure_reason="blocked_replay_mismatch")
    blocked_replay["argv"] = [
        arg.replace(".coverage.replay", ".coverage.blocked-replay")
        .replace("success/experiment_7940_fixture.json", "blocked/experiment_7940_fixture.json")
        .replace("success/rows.json", "blocked/rows.json")
        for arg in blocked_replay["argv"]
    ]
    value["commands"].insert(value["commands"].index(blocked) + 1, blocked_replay)
    for row in value["commands"]:
        if row["name"] in {
            "coverage_expected_failure",
            "coverage_cold_replay",
            "coverage_blocked_replay",
        }:
            row["argv"] += ["--output", str(private / row["name"] / "experiment_7940_fixture.json")]
    combined = next(row for row in value["commands"] if row["name"] == "coverage_combine")
    combined["argv"].append(str(private / ".coverage.blocked"))
    combined["argv"].append(str(private / ".coverage.blocked-replay"))
    value.update(
        affected_tests=TESTS,
        coverage_includes=OWNED,
        dependency_hashes=shared.dependency_hashes(root, paths=OWNED + TESTS + shared.CONSUMERS),
        e2e_018="Private passing thirteen-task oracle and historical twelve/thirteen-task lifecycle tests",
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
    return value


def coverage_complete(report: Path) -> bool:
    """Require nonempty complete statements in every owned module and the CLI."""
    return shared.coverage_complete(report, includes=OWNED)


def publish(root: Path, value: dict[str, Any], output: Path, raw: Path, private: Path) -> None:
    """Recheck final verdict bytes before the locked unique-primary publisher exposes them."""
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
        "exp7940-contract-methods",
        output.parent,
        field="contract_ready_score",
        expected=value["contract_ready_score"],
    )
    if selected["gate_sha256"] != digest or selected["document_sha256"] != digest:
        raise ValueError("primary_reader_drift")
    atomic_json(raw.parent / "primary_resolution_receipt.json", selected)


def execute(root: Path, args: Any, private: Path) -> dict[str, Any]:
    """Freeze current inputs before running bounded checks through the shared supervisor."""
    started = time.monotonic_ns()
    print("[exp7940] phase=inputs_start completed_units=0 elapsed_s=0", flush=True)
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
            activation_confirmed=False,
            resume_requires_identical_dependencies=True,
        ),
    )
    print(
        f"[exp7940] phase=inputs_frozen completed_units=3 elapsed_s={(time.monotonic_ns() - started) / 1e9:.3f}",
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
        for name in ("coverage.json", ".coverage.combined"):
            path = private / name
            if path.is_file():
                shutil.copyfile(path, durable / f"{name}-{sha256_file(path)[7:]}")
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
        f"[exp7940] phase=published completed_units=13 elapsed_s={(time.monotonic_ns() - started) / 1e9:.3f}",
        flush=True,
    )
    return value

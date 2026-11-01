"""Freeze bounded current checks and publish only checked terminal bytes.

REQ-REPORT-7903-V686. Scratch stays outside results. Child logs become durable
only after exit, so pytest guards continue to protect historical evidence.
"""

from __future__ import annotations

import ast
import gzip
import json
from pathlib import Path
import shutil
import time
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands

OWNED = [
    "python/carnot/reporting/v686_contract_methods.py",
    "python/carnot/reporting/v686_contract_validation.py",
    "scripts/experiments/experiment_7903_v686_contract_methods.py",
]
CHANGED = [*OWNED, "python/carnot/reporting/v685_authority_lifecycle.py"]
TESTS = [
    "tests/python/test_experiment_7903_v686_contract_methods.py",
    "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
]
CONSUMERS = [
    "tests/python/test_roadmap_schema.py",
    "tests/python/test_audit_roadmap_gates.py",
    "tests/python/test_exclusion_manifest_lint.py",
    "tests/python/test_conductor_gates.py",
]


def dependency_hashes(root: Path, *, paths: list[str] | None = None) -> dict[str, str]:
    """Follow local imports so changed helpers invalidate a frozen validation closure."""
    pending = [
        root / item for item in (paths if paths is not None else CHANGED + TESTS + CONSUMERS)
    ]
    found = {}
    while pending:
        path = pending.pop()
        label = str(path.relative_to(root))
        if label in found:
            continue
        found[label] = sha256_file(path)
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names = [node.module] + [node.module + "." + alias.name for alias in node.names]
            for name in names:
                for base in (root / "python", root):
                    parts = name.split(".")
                    for index in range(1, len(parts)):
                        initializer = base.joinpath(*parts[:index], "__init__.py")
                        if initializer.is_file():
                            pending.append(initializer)
                    module = base.joinpath(*name.split("."))
                    for candidate in (module.with_suffix(".py"), module / "__init__.py"):
                        if candidate.is_file():
                            pending.append(candidate)
    return found


def prepare(root: Path, private: Path) -> None:
    """Copy immutable oracle fixtures; the live design never borrows these bytes."""
    private.mkdir(parents=True, exist_ok=True)
    for name in ("design.md", "active.yaml"):
        (private / name).write_bytes(
            gzip.decompress((root / "tests/fixtures/v686" / f"{name}.gz").read_bytes())
        )
    atomic_json(private / "negative.json", {"experiment_id": 0})
    atomic_json(private / "negative-rows.json", {"rows": []})


def manifest(root: Path, private: Path) -> dict[str, Any]:
    """Freeze identical coverage includes, exact argv, expected exits and deadlines."""
    py, cov, pytest, ruff, mypy = (
        str(root / ".venv/bin" / name) for name in ("python", "coverage", "pytest", "ruff", "mypy")
    )
    cli = str(root / "scripts/experiments/experiment_7903_v686_contract_methods.py")
    include = "--include=" + ",".join(str(root / name) for name in CHANGED)
    common = ["-n", "0", "-o", "addopts=", "--no-cov", "-q"]
    success = [
        cli,
        "--date",
        "20260930",
        "--fixture-e2e",
        "--design",
        str(private / "design.md"),
        "--active",
        str(private / "active.yaml"),
        "--staged",
        str(private / "consumed.yaml"),
        "--output",
        str(private / "fixture.json"),
        "--raw",
        str(private / "rows.json"),
    ]
    replay = [
        cli,
        "--date",
        "20260930",
        "--cold-replay",
        str(private / "fixture.json"),
        "--raw",
        str(private / "rows.json"),
    ]
    negative = [
        cli,
        "--date",
        "20260930",
        "--cold-replay",
        str(private / "negative.json"),
        "--raw",
        str(private / "negative-rows.json"),
    ]
    e2e = str(root / "scripts/experiments/experiment_7868_v683_intervention_protocol.py")
    commands = [
        (
            "affected_pytest",
            [pytest, *common, f"--basetemp={private / 'affected'}", *TESTS],
            0,
            900,
        ),
        (
            "consumer_pytest",
            [pytest, *common, f"--basetemp={private / 'consumer'}", *CONSUMERS],
            0,
            240,
        ),
        (
            "coverage_unit",
            [
                cov,
                "run",
                f"--data-file={private / '.coverage.unit'}",
                include,
                "-m",
                "pytest",
                *common,
                f"--basetemp={private / 'coverage'}",
                *TESTS,
            ],
            0,
            900,
        ),
        (
            "coverage_cli_success",
            [cov, "run", f"--data-file={private / '.coverage.cli'}", include, *success],
            0,
            120,
        ),
        (
            "coverage_expected_failure",
            [cov, "run", f"--data-file={private / '.coverage.failure'}", include, *negative],
            1,
            60,
        ),
        (
            "coverage_cold_replay",
            [cov, "run", f"--data-file={private / '.coverage.replay'}", include, *replay],
            0,
            60,
        ),
        (
            "coverage_combine",
            [
                cov,
                "combine",
                f"--data-file={private / '.coverage.combined'}",
                *[
                    str(private / f".coverage.{part}")
                    for part in ("unit", "cli", "failure", "replay")
                ],
            ],
            0,
            60,
        ),
        (
            "coverage_report",
            [
                cov,
                "report",
                f"--data-file={private / '.coverage.combined'}",
                include,
                "--show-missing",
                "--fail-under=100",
            ],
            0,
            60,
        ),
        (
            "coverage_json",
            [
                cov,
                "json",
                f"--data-file={private / '.coverage.combined'}",
                include,
                "-o",
                str(private / "coverage.json"),
            ],
            0,
            60,
        ),
        ("ruff_check", [ruff, "check", *CHANGED, TESTS[0]], 0, 90),
        ("ruff_format", [ruff, "format", "--check", *CHANGED, TESTS[0]], 0, 90),
        ("mypy_strict", [mypy, "--strict", *CHANGED], 0, 180),
        ("scoped_spec_coverage", [py, "scripts/check_spec_coverage.py", *TESTS], 0, 180),
        (
            "e2e_016_fixture",
            [py, e2e, "--date", "20260930", "--fixture-e2e", str(private / "e2e016.json")],
            0,
            120,
        ),
        (
            "e2e_016_replay",
            [py, e2e, "--date", "20260930", "--cold-replay", str(private / "e2e016.json")],
            0,
            60,
        ),
        (
            "repository_full_suite",
            [pytest, *common, f"--basetemp={private / 'full-suite'}", "tests/python"],
            0,
            180,
        ),
    ]
    return {
        "affected_tests": TESTS,
        "consumer_tests": CONSUMERS,
        "coverage_includes": CHANGED,
        "dependency_hashes": dependency_hashes(root),
        "environment": {"PYTHONPATH": "python:.", "JAX_PLATFORMS": "cpu"},
        "commands": [
            {
                "name": name,
                "argv": argv,
                "expected_exit": expected,
                "deadline_s": deadline,
                "failure_reason": "cold_replay_mismatch"
                if expected == 1
                else "required_check_failed",
                "classification": "diagnostic" if name == "repository_full_suite" else "required",
            }
            for name, argv, expected, deadline in commands
        ],
        "e2e_018": "Private immutable lifecycle tests only; no historical publishing CLI against V686",
        "future_scope": "Task prompt snapshots freeze methods; future producers must freeze their actual code closure before running",
    }


def run_check(
    root: Path, spec: dict[str, Any], private: Path, durable: Path, *, heartbeat_s: float = 30
) -> dict[str, Any]:
    """Reuse the streaming child supervisor and seal logs after each real exit."""
    command = CommandSpec(
        spec["name"],
        tuple(spec["argv"]),
        spec.get("classification", "required"),
        spec["deadline_s"],
    )
    row = run_commands(
        root,
        [command],
        log_dir=private / "logs",
        heartbeat_s=heartbeat_s,
        extra_env={
            "PYTHONPATH": "python:.",
            "JAX_PLATFORMS": "cpu",
            "COVERAGE_FILE": str(private / ".coverage.repository"),
        },
    )[0]
    source = root / row["log_path"]
    sealed = durable / f"{spec['name']}-{row['log_sha256'][7:]}.log"
    sealed.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, sealed)
    return {
        **row,
        **spec,
        "actual_exit": row["exit_code"],
        "passed": row["exit_code"] == spec["expected_exit"] and not row["timed_out"],
        "log_path": str(sealed),
        "log_sha256": sha256_file(sealed),
    }


def coverage_complete(report: Path, *, includes: list[str] | None = None) -> bool:
    """Require nonempty and complete statement evidence for each measured file."""
    if not report.is_file():
        return False
    files = json.loads(report.read_text())["files"]
    return all(
        name in files
        and files[name]["summary"]["num_statements"] > 0
        and files[name]["summary"]["num_statements"] == files[name]["summary"]["covered_lines"]
        and not files[name]["missing_lines"]
        for name in (includes if includes is not None else CHANGED)
    )


def publish(
    root: Path, artifact: dict[str, Any], output: Path, private: Path, durable: Path
) -> None:
    """Bind validator sidecars to candidate hashes, then atomically copy checked bytes."""
    private.mkdir(parents=True, exist_ok=True)
    candidate_path = private / "terminal-candidate.json"
    for attempt in range(2):
        atomic_json(candidate_path, artifact)
        digest = sha256_file(candidate_path)
        checks = [
            {
                "name": "adversarial_verify",
                "argv": [
                    str(root / ".venv/bin/python"),
                    str(root / "scripts/adversarial_verify.py"),
                    "--json",
                    str(candidate_path),
                ],
                "expected_exit": 0,
                "deadline_s": 120,
            },
            {
                "name": "verdict_row_consistency",
                "argv": [
                    str(root / ".venv/bin/python"),
                    str(root / "scripts/verdict_row_consistency_lint.py"),
                    "--strict",
                    str(candidate_path),
                ],
                "expected_exit": 0,
                "deadline_s": 120,
            },
        ]
        reports = [run_check(root, spec, private, durable) for spec in checks]
        atomic_json(
            durable / f"terminal-{digest[7:]}.json",
            {
                "candidate_sha256": digest,
                "published_path": str(output),
                "reports": reports,
                "passed": all(row["passed"] for row in reports),
            },
        )
        if all(row["passed"] for row in reports):
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(f".{output.name}.checked")
            shutil.copyfile(candidate_path, temporary)
            if sha256_file(temporary) != digest:
                raise ValueError("published_bytes_differ")
            temporary.replace(output)
            return
        if attempt:
            raise ValueError("terminal_recheck_failed")
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = "complete_disqualified_terminal_validation"
        artifact["contract_ready_score"] = 0
        artifact["acceptance_gate_results"].update(readiness=0, validity=False)
        artifact["required_checks_passed"] = False
        artifact["flagged_adversarial"] = not reports[0]["passed"]
        artifact["gate_check_summary"].append(
            {
                "upstream_id": f"exp{artifact.get('experiment_id', 7903)}-terminal-validation",
                "path": str(candidate_path),
                "hash": digest,
                "artifact_field": "validators.passed",
                "op": "==",
                "expected": True,
                "observed": [row["passed"] for row in reports],
            }
        )


def execute(
    root: Path,
    args: Any,
    private: Path,
    *,
    methods_module: Any = None,
    prepare_fn: Any = None,
    manifest_fn: Any = None,
    coverage_fn: Any = None,
    experiment_id: int = 7903,
    count: int = 12,
) -> dict[str, Any]:
    """Freeze inputs and methods before checks, using private scratch for all children."""
    from carnot.reporting import v686_contract_methods as methods

    methods = methods_module or methods
    prepare_call = prepare_fn or prepare
    manifest_call = manifest_fn or manifest
    coverage_call = coverage_fn or coverage_complete
    started = time.monotonic_ns()
    print(f"[exp{experiment_id}] phase=start elapsed_s=0 completed_units=0", flush=True)
    prepare_call(root, private)
    durable = args.raw.parent
    assessment = methods.assess(
        args.design, args.staged, args.active, durable / "authority_snapshots"
    )
    custody = methods.source_custody(args.source, methods.SOURCE_SHA256)
    print(
        f"[exp{experiment_id}] phase=resolved_inputs elapsed_s={(time.monotonic_ns() - started) / 1e9:.3f} completed_units=3",
        flush=True,
    )
    freeze = methods.method_freeze(root)
    frozen = manifest_call(root, private)
    frozen["input_hashes"] = {
        "authorities": assessment["authority_snapshots"],
        "source_custody": custody["hashes"],
    }
    manifest_path = durable / "validation_command_manifest.json"
    atomic_json(manifest_path, frozen)
    atomic_json(durable / "method_freeze.json", freeze)
    checkpoint_key = (
        methods.shared.canonical_hash(frozen) if methods_module else methods.canonical_hash(frozen)
    )
    atomic_json(
        durable / f"checkpoint-{checkpoint_key[7:]}.json",
        {
            "status": "inputs_and_commands_frozen",
            "manifest_sha256": sha256_file(manifest_path),
            "method_freeze_sha256": sha256_file(durable / "method_freeze.json"),
        },
    )
    controls = methods.mutations(
        private / "design.md", private / "active.yaml", args.source, private / "mutations"
    )
    receipts = []
    if not args.fixture_e2e:
        receipts = [
            run_check(root, spec, private, durable / "sealed_logs") for spec in frozen["commands"]
        ]
        report = private / "coverage.json"
        receipts.append(
            {
                "name": "per_file_statement_coverage",
                "argv": ["coverage_json_reduction", str(report)],
                "expected_exit": 0,
                "actual_exit": int(not coverage_call(report)),
                "passed": coverage_call(report),
                "deadline_s": 0,
            }
        )
        for name in ("coverage.json", ".coverage.combined"):
            path = private / name
            if path.is_file():
                sealed = durable / f"{name}-{sha256_file(path)[7:]}"
                shutil.copyfile(path, sealed)
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
        publish(root, value, args.output, private, durable / "terminal_validation")
        value = json.loads(args.output.read_text())
        atomic_json(args.raw, methods.primitive_rows(value))
    print(
        f"[exp{experiment_id}] phase=published elapsed_s={(time.monotonic_ns() - started) / 1e9:.3f} completed_units={count}",
        flush=True,
    )
    return value

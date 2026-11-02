"""REQ-REPORT-7966-V691: keep mutable checks private and publish checked bytes."""

from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import time
from typing import Any

from carnot.reporting import v686_contract_validation as shared
from carnot.reporting import v690_contract_validation as prior
from carnot.reporting import v691_contract_methods as methods
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import publish_primary, reader_receipt

OWNED = [
    "python/carnot/reporting/v691_contract_methods.py",
    "python/carnot/reporting/v691_contract_validation.py",
    "scripts/experiments/experiment_7966_v691_contract_methods.py",
]
TESTS = ["tests/python/test_experiment_7966_v691_contract_methods.py", *prior.TESTS[1:]]
PRODUCERS = [
    "experiment_7953_v690_contract_methods.json",
    "experiment_7955_v690_response_targets.json",
    "experiment_7958_v690_qwen_response_risk.json",
    "experiment_7965_v690_capstone.json",
]
PRODUCER_HASHES = [
    "sha256:1bc9dfa53c25a4a29f25dfee1e6b861c0e89aaef7f36e9c9e6e4e670b3ffd042",
    "sha256:dbfac2d991450601add8506a00452c9b6d39a594ad91d3c291f742cfe378c62a",
    "sha256:3342ad4f66c5613482ac7da1da2b6f87099dfa3135647fa559b65748d7f62cad",
    "sha256:5629daeb1fe7723dc4ef0f8f9259492c12c47a71db49e3a5822d0e4b90f96c89",
]


def prepare(root: Path, private: Path) -> None:
    """Copy available independent authorities; absent staging remains absent."""
    private.mkdir(parents=True, exist_ok=True)
    for target, source in (
        ("design.md", "openspec/change-proposals/research-roadmap-vNEXT.md"),
        ("active.yaml", "research-roadmap.yaml"),
        ("stage.yaml", "research-roadmap-next.yaml"),
    ):
        path = root / source
        if path.is_file():
            (private / target).write_bytes(path.read_bytes())
    atomic_json(private / "negative.json", {"experiment_id": 0})
    atomic_json(private / "negative-rows.json", {"rows": []})


def manifest(root: Path, private: Path) -> dict[str, Any]:
    """Freeze argv and dependency closure before tests or results can influence them."""
    value = prior.manifest(root, private)
    before = "--include=" + ",".join(str(root / n) for n in prior.OWNED)
    after = "--include=" + ",".join(str(root / n) for n in OWNED)
    for row in value["commands"]:
        row["argv"] = [
            a.replace(before, after)
            .replace("7953_v690", "7966_v691")
            .replace("7953_fixture", "7966_fixture")
            for a in row["argv"]
        ]
        if (
            "experiment_7966_v691_contract_methods.py" in " ".join(row["argv"])
            and "--date" in row["argv"]
        ):
            row["argv"][row["argv"].index("--date") + 1] = "20261001"
        if row["name"] in {"affected_pytest", "coverage_unit", "scoped_spec_coverage"}:
            row["argv"] = [a for a in row["argv"] if a not in prior.TESTS + TESTS] + TESTS
        if row["name"] in {"ruff_check", "ruff_format", "mypy_strict"}:
            row["argv"] = row["argv"][: 3 if row["name"] == "ruff_format" else 2] + OWNED
            if row["name"] != "mypy_strict":
                row["argv"].append(TESTS[0])
            row["argv"] += [
                "python/carnot/reporting/v690_authority.py",
                "python/carnot/reporting/v690_contract_methods.py",
            ]
    value["commands"] += [
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
            failure_reason="private_authority_failed",
            classification="required",
        ),
        dict(
            name="repository_full_suite",
            argv=[str(root / ".venv/bin/pytest"), "tests/python", "-q"],
            expected_exit=0,
            deadline_s=180,
            failure_reason="repository_backlog",
            classification="diagnostic",
        ),
    ]
    value.update(
        affected_tests=TESTS,
        coverage_includes=OWNED,
        scratch_root=str(private),
        dependency_hashes=shared.dependency_hashes(root, paths=OWNED + TESTS + shared.CONSUMERS),
        producer_invocations=[methods.freeze_invocation(root / "results" / p) for p in PRODUCERS],
        e2e_018="private authority tests only; no historical live publishing CLI",
    )
    value["dependency_hashes"].update(
        {
            n: sha256_file(root / n)
            for n in (
                "python/carnot/reporting/v690_authority.py",
                "python/carnot/reporting/v690_contract_methods.py",
                "openspec/change-proposals/research-roadmap-vNEXT.md",
                "research-roadmap.yaml",
                "research-studying.md",
                "research-references.md",
                "ops/exclusion_manifest.yaml",
                "openspec/capabilities/research-reporting/spec.md",
            )
        }
    )
    for receipt, digest in zip(value["producer_invocations"], PRODUCER_HASHES, strict=True):
        receipt["sha256"] = digest
    health = (
        root / "results/raw/experiment_7966_v691_contract_methods/repository_health/receipt.json"
    )
    if health.is_file():
        value["repository_health_current_receipt"] = json.loads(health.read_text())
        value["commands"] = [r for r in value["commands"] if r["name"] != "repository_full_suite"]
    return value


def coverage_complete(report: Path) -> bool:
    """Require every added module and CLI statement in nonempty measured files."""
    return shared.coverage_complete(report, includes=OWNED)


def publish(root: Path, value: dict[str, Any], output: Path, raw: Path, private: Path) -> None:
    """Recheck final bytes and ask both unchanged consumers which primary they select."""
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
            and methods.cold_replay(path, raw),
            terminal_validation_path=str(terminal / "terminal_validation.json"),
        ),
    )
    atomic_json(terminal / "terminal_validation.json", {**report, **publication})
    selected = reader_receipt(
        "exp7966-contract-methods",
        output.parent,
        field="contract_ready_score",
        expected=value["contract_ready_score"],
    )
    if selected["gate_sha256"] != digest or selected["document_sha256"] != digest:
        raise ValueError("primary_reader_drift")
    atomic_json(raw.parent / "primary_resolution_receipt.json", selected)


def execute(root: Path, args: Any, private: Path) -> dict[str, Any]:
    """Freeze inputs, supervise bounded children and archive only completed evidence."""
    started, started_at = time.monotonic_ns(), datetime.now(UTC).isoformat()
    print("[exp7966] phase=inputs_start completed_units=0 elapsed_s=0", flush=True)
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
        authorities=assessment["authority_snapshots"], source_custody=custody["hashes"]
    )
    manifest_path = durable / "validation_command_manifest.json"
    atomic_json(manifest_path, frozen)
    atomic_json(durable / "method_freeze.json", freeze)
    atomic_json(durable / "producer_invocation_receipts.json", frozen["producer_invocations"])
    atomic_json(
        durable / "input_checkpoint.json",
        dict(
            manifest_sha256=sha256_file(manifest_path),
            method_freeze_sha256=sha256_file(durable / "method_freeze.json"),
            dependencies=frozen["dependency_hashes"],
            resume_requires_identical_dependencies=True,
        ),
    )
    print(
        f"[exp7966] phase=inputs_frozen completed_units=3 elapsed_s={(time.monotonic_ns() - started) / 1e9:.3f}",
        flush=True,
    )
    controls = methods.mutations(
        private / "design.md", private / "active.yaml", args.source, private / "mutations"
    )
    receipts = [
        dict(
            name="private_authority_and_mutations",
            argv=["in_process_private_fixture"],
            passed=all(r["passed"] for r in controls),
            actual_exit=0,
            expected_exit=0,
            deadline_s=0,
        )
    ]
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
    print(
        f"[exp7966] phase=checks_complete completed_units={len(receipts)} elapsed_s={(time.monotonic_ns() - started) / 1e9:.3f}",
        flush=True,
    )
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
        started_at=started_at,
    )
    value["repository_health"]["current_full_suite"] = [
        frozen.get("repository_health_current_receipt")
    ]
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
        f"[exp7966] phase=published completed_units=13 elapsed_s={(time.monotonic_ns() - started) / 1e9:.3f}",
        flush=True,
    )
    return value

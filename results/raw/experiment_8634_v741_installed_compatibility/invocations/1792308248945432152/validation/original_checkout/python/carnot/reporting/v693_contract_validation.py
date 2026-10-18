"""REQ-REPORT-7992-V693: qualify new statements and publish only checked primary bytes."""

import gzip
import json
from pathlib import Path
import shutil
import time
from typing import Any

import yaml

from carnot.reporting import v686_contract_validation as shared
from carnot.reporting import v692_contract_validation as prior
from carnot.reporting import v693_contract_methods as methods
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import publish_primary, reader_receipt

OWNED = [
    "python/carnot/reporting/v693_contract_methods.py",
    "python/carnot/reporting/v693_contract_validation.py",
    "scripts/experiments/experiment_7992_v693_contract_methods.py",
]
TESTS = ["tests/python/test_experiment_7992_v693_contract_methods.py", shared.TESTS[1]]
run_check = shared.run_check


def prepare(root: Path, private: Path) -> None:
    """Build an explicitly circular private oracle; live design bytes remain independently assessed."""
    private.mkdir(parents=True, exist_ok=True)
    (private / "active.yaml").write_bytes(
        gzip.decompress((root / "tests/fixtures/v693/active.yaml.gz").read_bytes())
    )
    atomic_json(private / "negative.json", {"experiment_id": 0})
    atomic_json(private / "negative-rows.json", {"rows": []})
    roadmap = yaml.safe_load((private / "active.yaml").read_bytes())
    tasks = roadmap["tasks"]
    table = ["| Order | ID | Title | Phase | Deliverable |", "| --- | --- | --- | --- | --- |"]
    table += [
        f"| {i} | {t['id']} | {t['title']} | {t['phase']} | {t['deliverable']} |"
        for i, t in enumerate(tasks, 1)
    ]
    design = "# Private circular V693 oracle\n\n## Exact task contract\n\n" + "\n".join(table)
    design += (
        "\n\nCanonical full-task SHA-256: `"
        + methods.authority.lifecycle.tasks_digest(tasks)
        + "`\n"
    )
    design += "\n<!-- V693_TASK_CONTRACT_START -->\n```json\n" + json.dumps(roadmap) + "\n```\n"
    (private / "design.md").write_text(design)


def manifest(root: Path, private: Path) -> dict[str, Any]:
    """Freeze identical new-code includes while preserving actual diagnostic full-suite execution."""
    value = prior.manifest(root, private)
    value["commands"] = [r for r in value["commands"] if r["name"] != "affected_pytest"]
    before = "--include=" + ",".join(str(root / n) for n in prior.OWNED)
    after = "--include=" + ",".join(str(root / n) for n in OWNED)
    for row in value["commands"]:
        row.pop("reused_receipt", None)
        row["argv"] = [
            a.replace(before, after).replace("7979_v692", "7992_v693") for a in row["argv"]
        ]
        if row["name"] in {"affected_pytest", "coverage_unit", "scoped_spec_coverage"}:
            row["argv"] = [a for a in row["argv"] if a not in prior.TESTS + TESTS] + TESTS
        if row["name"] in {"ruff_check", "ruff_format", "mypy_strict"}:
            row["argv"] = row["argv"][: 3 if row["name"] == "ruff_format" else 2] + OWNED
            if row["name"] != "mypy_strict":
                row["argv"].append(TESTS[0])
    value.update(
        coverage_includes=OWNED,
        affected_tests=TESTS,
        dependency_hashes=shared.dependency_hashes(root, paths=OWNED + TESTS + shared.CONSUMERS),
        fixture_sha256=sha256_file(root / "tests/fixtures/v693/active.yaml.gz"),
    )
    return value


def coverage_complete(report: Path) -> bool:
    """An empty denominator cannot certify the newly added executable statements."""
    return shared.coverage_complete(report, includes=OWNED)


def publish(root: Path, value: dict[str, Any], output: Path, raw: Path, private: Path) -> None:
    """Both unchanged consumers must resolve precisely the validator-checked artifact bytes."""
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
        "exp7992-contract-methods",
        output.parent,
        field="contract_ready_score",
        expected=value["contract_ready_score"],
    )
    if selected["gate_sha256"] != digest or selected["document_sha256"] != digest:
        raise ValueError("primary_reader_drift")
    atomic_json(raw.parent / "primary_resolution_receipt.json", selected)


def execute(root: Path, args: Any, private: Path) -> dict[str, Any]:
    """Freeze code and input bytes before measurements, then supervise every required child."""
    started = time.monotonic_ns()
    print("[exp7992] phase=inputs_start completed_units=0", flush=True)
    prepare(root, private)
    durable = args.raw.parent
    assessment = methods.assess(
        args.design, args.staged, args.active, durable / "authority_snapshots"
    )
    custody = methods.source_custody(args.source, methods.SOURCE_SHA256)
    observed = args.active if args.active.is_file() else args.staged
    try:
        tasks = yaml.safe_load(observed.read_bytes())["tasks"] if observed.is_file() else []
    except yaml.YAMLError:
        tasks = []
    freeze = methods.method_freeze(tasks)
    frozen = manifest(root, private)
    frozen["input_hashes"] = dict(
        authorities=assessment["authority_snapshots"], source_custody=custody["hashes"]
    )
    frozen["code_snapshot_rows"] = [
        methods.authority.lifecycle._snapshot(
            root / name,
            (root / name).read_bytes(),
            durable / "code_snapshots",
            name.replace("/", "_"),
        )
        for name in frozen["dependency_hashes"]
    ]
    frozen["historical_inputs"] = methods.historical_custody(root, durable / "producer_snapshots")
    frozen["source_snapshot_rows"] = [
        methods.authority.lifecycle._snapshot(
            Path(row["path"]),
            Path(row["path"]).read_bytes(),
            durable / "source_snapshots",
            f"source_{i}",
        )
        for i, row in enumerate(custody["hashes"])
        if Path(row["path"]).is_file()
    ]
    manifest_path = durable / f"validation_command_manifest-{canonical_hash(frozen)[7:]}.json"
    atomic_json(manifest_path, frozen)
    atomic_json(durable / "method_freeze.json", freeze)
    print("[exp7992] phase=inputs_frozen completed_units=3", flush=True)
    controls = methods.mutations(
        private / "design.md", private / "active.yaml", private / "mutations"
    )
    receipts = [
        dict(
            name="private_fixture",
            argv=["in_process_private_fixture"],
            passed=all(r["passed"] for r in controls),
            actual_exit=0,
        )
    ]
    coverage = {}
    if not args.fixture_e2e:
        receipts = []
        for spec in frozen["commands"]:
            print(f"[exp7992] subprocess_before name={spec['name']}", flush=True)
            receipt = run_check(root, spec, private, durable / "sealed_logs")
            receipts.append(receipt)
            atomic_json(durable / "validation_checkpoint.json", {"receipts": receipts})
            print(
                f"[exp7992] subprocess_after name={spec['name']} exit={receipt['actual_exit']}",
                flush=True,
            )
        complete = coverage_complete(private / "coverage.json")
        receipts.append(
            dict(
                name="per_file_statement_coverage",
                argv=["coverage_json_reduction", str(private / "coverage.json")],
                expected_exit=0,
                actual_exit=int(not complete),
                passed=complete,
            )
        )
        for name in ("coverage.json", ".coverage.combined"):
            path = private / name
            if path.is_file():
                shutil.copyfile(path, durable / f"{name}-{sha256_file(path)[7:]}")
        if (private / "coverage.json").is_file():
            coverage = {
                name: row["summary"]
                for name, row in json.loads((private / "coverage.json").read_text())[
                    "files"
                ].items()
            }
    print(f"[exp7992] phase=checks_complete completed_units={len(receipts)}", flush=True)
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
    value["coverage_statement_counts"] = coverage
    atomic_json(args.raw, methods.primitive_rows(value))
    if args.fixture_e2e:
        atomic_json(args.output, value)
    else:
        publish(root, value, args.output, args.raw, private)
    print(
        f"[exp7992] phase=published elapsed_s={(time.monotonic_ns() - started) / 1e9:.3f}",
        flush=True,
    )
    return value

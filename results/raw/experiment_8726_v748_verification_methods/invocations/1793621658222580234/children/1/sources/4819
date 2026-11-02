"""REQ-VERIFY-8707: bounded children and independently sealed publication."""

from __future__ import annotations

import argparse
from datetime import datetime, UTC
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any

from carnot.reporting import hardware_operation_boundary_8707 as e
from carnot.reporting import hardware_operation_inputs_8707 as i
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.gatemate_continuity_8400 import budget
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child

Json = dict[str, Any]


def measure(raw: Path, date: str, root: Path, private: bool) -> Json:
    """Freeze source and exact input bytes before any owned validation starts."""
    start, wall = time.monotonic_ns(), datetime.now(UTC).isoformat()
    raw.mkdir(parents=True, exist_ok=True)
    e.progress("preconditions_before")
    with tempfile.TemporaryDirectory(prefix="hardware8707-", dir="/var/tmp") as directory:
        scratch = Path(directory)
        (scratch / "probe").write_bytes(b"disk scratch")
        fs = child(
            "scratch_filesystem",
            ["findmnt", "-T", directory, "-n", "-o", "FSTYPE,SOURCE"],
            raw / "preconditions",
            deadline=5,
        )
        filesystem = Path(fs["stdout_path"]).read_text().strip()
        memory = (
            int(
                next(
                    line.split()[1]
                    for line in Path("/proc/meminfo").read_text().splitlines()
                    if line.startswith("MemAvailable:")
                )
            )
            * 1024
        )
        pre = dict(
            scratch_path=directory,
            scratch_filesystem=filesystem,
            disk_backed=bool(filesystem) and filesystem.split()[0] not in {"tmpfs", "ramfs"},
            free_disk_bytes=shutil.disk_usage(scratch).free,
            memory_available_bytes=memory,
            write_probe_passed=(scratch / "probe").read_bytes() == b"disk scratch",
            tools_available=bool(shutil.which("findmnt")),
            optional_device_tools={name: shutil.which(name) for name in ["ssh", "scp", "xmutil"]},
            optional_devices_required=False,
            language_model_calls=0,
            measurement_budget_s=3600,
            validation_reserve_s=900,
            total_budget_s=4800,
            python=sys.version,
            software_versions={
                name: version(name)
                for name in ["coverage", "pytest", "ruff", "mypy", "numpy", "scipy"]
            },
        )
    failures = []
    if not (
        fs["passed"]
        and pre["disk_backed"]
        and pre["write_probe_passed"]
        and pre["tools_available"]
        and min(pre["free_disk_bytes"], memory) >= 2**30
    ):
        failures.append(
            dict(
                upstream="host",
                check="cpu_resources",
                path="/var/tmp",
                sha256=None,
                field="disk_memory_tools",
                operator="==",
                expected=True,
                observed=pre,
                operand_state="resource_unavailable",
                passed=False,
            )
        )
    e.progress("authenticate_original_inputs", 0, 4)
    evidence = i.authenticate(root)
    code = sorted(
        set(
            e.OWNED
            + e.TESTS
            + [
                str(Path(m.__file__).resolve().relative_to(i.ROOT))
                for m in list(sys.modules.values())
                if getattr(m, "__file__", None)
                and Path(m.__file__).suffix == ".py"
                and Path(m.__file__).resolve().is_relative_to(i.ROOT)
                and ".venv" not in Path(m.__file__).parts
            ]
        )
    )
    dependencies = code + [
        "AGENTS.md",
        "CODEX.md",
        "CLAUDE.md",
        "pyproject.toml",
        i.DESIGN,
        "research-roadmap.yaml",
        "openspec/change-proposals/research-roadmap-v745-preserved-20261101.md",
        "openspec/capabilities/hardware-operation-boundary/spec.md",
        "openspec/capabilities/research-reporting/spec.md",
        "openspec/capabilities/verification/spec.md",
        "ops/exclusion_manifest.yaml",
        "ops/e2e-test-plan.md",
        "ops/north-star.md",
        "research-references.md",
        "research-hardware-wishlist.md",
        "scripts/experiment_template.py",
        "scripts/adversarial_verify.py",
        "scripts/verdict_row_consistency_lint.py",
        "scripts/check_spec_coverage.py",
        "tests/python/conftest.py",
        "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
        "tests/python/test_restricted_decision_audit_8210.py",
        "tests/python/test_primary_publication_7928.py",
        "docs/research-notes/v746-hardware-operation-boundary.md",
    ]
    source_refs = evidence["refs"].copy()
    code_refs = [i.reference(i.ROOT / p) for p in code]
    for name in sorted(set(dependencies)):
        source, target = i.ROOT / name, raw / "source_closure" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        source_refs += [i.reference(source), i.reference(target)]
    atomic_json(
        raw / "premeasurement_seal.json", dict(source_refs=source_refs, code_refs=code_refs)
    )
    end = time.monotonic_ns()
    if (end - start) / 1e9 > 3600:
        raise TimeoutError("measurement_deadline")
    work = dict(
        raw=str(raw.absolute()),
        run_date=date,
        root=str(root.absolute()),
        private_control=private,
        evidence=evidence,
        source_refs=source_refs,
        code_refs=code_refs,
        resource_failures=failures,
        preconditions=pre,
        started_at_utc=wall,
        ended_at_utc=datetime.now(UTC).isoformat(),
        start_ns=start,
        end_ns=end,
        duration_s=(end - start) / 1e9,
        phase_spans=[
            dict(
                phase="authenticate_and_freeze",
                start_ns=start,
                end_ns=end,
                duration_s=(end - start) / 1e9,
            )
        ],
    )
    atomic_json(raw / "measurement.json", work)
    atomic_json(
        raw.parents[1] / "seals" / (raw.name + ".json"),
        dict(measurement=i.reference(raw / "measurement.json"), code_refs=code_refs),
    )
    e.progress("measurement_sealed", 4, 0)
    return work


def validate(raw: Path) -> list[Json]:
    """Retain each child's original coverage before combining only owned copies."""
    consumers = [
        "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
        "tests/python/test_restricted_decision_audit_8210.py",
        "tests/python/test_primary_publication_7928.py",
    ]
    common = ["-n", "0", "-o", "addopts=", "--no-cov", "-q"]
    logs = raw / "validation"
    receipts = [
        child(
            "collect",
            [sys.executable, "-m", "pytest", *common, "--collect-only", *e.TESTS, *consumers],
            logs,
            deadline=180,
        )
    ]
    nodes = [
        line.strip()
        for line in Path(receipts[0]["stdout_path"]).read_text().splitlines()
        if line.startswith("tests/") and "::" in line
    ]
    atomic_json(logs / "collected_nodes.json", nodes)
    with tempfile.TemporaryDirectory(prefix="hardware8707-coverage-", dir="/var/tmp") as directory:
        scratch = Path(directory)
        combined = scratch / "combined"
        combined.mkdir()
        config = scratch / "coverage.ini"
        config.write_text(
            "[run]\nparallel = true\npatch = subprocess\ndata_file = "
            + str(scratch / ".coverage")
            + "\ninclude =\n"
            + "".join("    " + str(i.ROOT / p) + "\n" for p in e.OWNED)
            + "[report]\nexclude_lines =\n"
        )
        shutil.copyfile(config, logs / "coverage.ini")
        for role, selected in [
            ("owned", [n for n in nodes if n.split("::")[0] in e.TESTS]),
            ("consumers", [n for n in nodes if n.split("::")[0] in consumers]),
        ]:
            folder = scratch / role
            folder.mkdir()
            receipt = child(
                "tests_" + role,
                [
                    "env",
                    f"COVERAGE_FILE={folder / '.coverage'}",
                    f"COVERAGE_RCFILE={config}",
                    sys.executable,
                    "-m",
                    "coverage",
                    "run",
                    "--rcfile=" + str(config),
                    "-m",
                    "pytest",
                    *common,
                    "--basetemp=" + str(folder / "temp"),
                    "--junitxml=" + str(logs / (role + ".xml")),
                    *selected,
                ],
                logs,
                deadline=360,
            )
            receipt["collected_nodes"] = selected
            receipt["references"] = []
            receipt["measured_source_hashes"] = [i.reference(i.ROOT / p) for p in e.OWNED]
            for shard in sorted(folder.glob(".coverage*")):
                retained = logs / "retained" / role / shard.name
                retained.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(shard, retained)
                receipt["references"].append(i.reference(retained))
                if role == "owned":
                    shutil.copyfile(retained, combined / shard.name)
            receipt["passed"] = receipt["passed"] and bool(selected) and bool(receipt["references"])
            atomic_json(logs / (role + "_custody.json"), receipt)
            receipts.append(receipt)
        for name, arguments in [
            ("combine", ["combine", "--keep", str(combined)]),
            ("coverage_report", ["report", "--fail-under=100"]),
            ("coverage_json", ["json", "-o", str(raw / "coverage.json")]),
        ]:
            receipts.append(
                child(
                    name,
                    [
                        sys.executable,
                        "-m",
                        "coverage",
                        arguments[0],
                        "--rcfile=" + str(config),
                        *arguments[1:],
                    ],
                    logs,
                    deadline=60,
                )
            )
        retained_final = logs / "retained" / "combined.sqlite"
        if (scratch / ".coverage").is_file():
            shutil.copyfile(scratch / ".coverage", retained_final)
        receipts.append(
            dict(
                name="retained_final_sqlite",
                passed=retained_final.is_file(),
                references=[i.reference(retained_final), i.reference(logs / "coverage.ini")]
                + [i.reference(p) for p in logs.glob("*.xml")]
                + [i.reference(raw / "coverage.json")],
            )
        )
    for name, argv in [
        ("ruff", ["-m", "ruff", "check", *e.OWNED, *e.TESTS]),
        ("format", ["-m", "ruff", "format", "--check", *e.OWNED, *e.TESTS]),
        ("types", ["-m", "mypy", "--strict", "--follow-imports=silent", *e.OWNED]),
        ("spec", ["scripts/check_spec_coverage.py", "--files", *e.TESTS]),
    ]:
        receipts.append(child(name, [sys.executable, *argv], logs, deadline=120))
    return receipts


def replay(path: Path) -> None:
    """A separately located seal defeats consistently rehashed replacement primitives."""
    value = json.loads(path.read_bytes())
    anchor = Path(value["publication_output"]).parent / "raw" / e.NAME
    raw = anchor / "invocations" / value["invocation_id"]
    seal = i.load(i.reference(anchor / "seals" / (raw.name + ".json")))
    if seal["measurement"] != value["primitive_reference"]:
        raise ValueError("independent_measurement_seal")
    work = i.load(value["primitive_reference"])
    if work["raw"] != str(raw) or work["code_refs"] != seal["code_refs"]:
        raise ValueError("independent_source_seal")
    for ref in work["source_refs"] + work["code_refs"]:
        i.load_bytes(ref)
    if work["duration_s"] != (work["end_ns"] - work["start_ns"]) / 1e9:
        raise ValueError("clock_drift")
    validation_seal = i.load(i.reference(anchor / "validation_seals" / (raw.name + ".json")))
    if validation_seal["validation"] != value["validation_reference"]:
        raise ValueError("independent_validation_seal")
    receipts = i.load(value["validation_reference"])["receipts"]
    for receipt in receipts:
        for ref in receipt.get("references", []):
            i.load_bytes(ref)
        for stream in ("stdout", "stderr"):
            if stream + "_sha256" in receipt:
                i.load_bytes(
                    dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
                )
    if e.build(work, receipts, Path(value["publication_output"])) != value:
        raise ValueError("report_drift")
    e.progress("cold_replay_passed", 4, 0)


def terminal(path: Path, raw: Path) -> Json:
    """Unchanged scanners inspect final bytes, retaining their real severities."""
    receipts = [
        child(name, [sys.executable, "-u", *argv], raw, deadline=180)
        for name, argv in [
            ("cold_replay", [e.CLI, "--cold-replay", str(path)]),
            ("adversarial", ["scripts/adversarial_verify.py", "--json", str(path)]),
            ("strict_rows", ["scripts/verdict_row_consistency_lint.py", "--strict", str(path)]),
        ]
    ]
    findings = json.loads(Path(receipts[1]["stdout_path"]).read_bytes())
    return dict(
        passed=all(r["passed"] for r in receipts) and findings["flagged_count"] == 0,
        receipts=receipts,
        actual_findings=findings,
    )


def run(output: Path, date: str, root: Path, private: bool) -> int:
    """Reject private checkout writes before creating any directory or candidate."""
    if private and (
        output.absolute().is_relative_to(i.ROOT) or output.resolve().is_relative_to(i.ROOT)
    ):
        raise ValueError("private output must be outside checkout")
    if not private and (output != i.ROOT / "results" / (e.NAME + ".json") or root != i.ROOT):
        raise ValueError("production output and input root are fixed")
    output = output.resolve()
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    work = measure(raw, date, root, private)
    receipts = (
        [dict(name="private_control_only", passed=True, scope="no_production_readiness")]
        if private
        else validate(raw)
    )
    atomic_json(raw / "validation.json", dict(receipts=receipts))
    atomic_json(
        raw.parents[1] / "validation_seals" / (raw.name + ".json"),
        dict(validation=i.reference(raw / "validation.json")),
    )
    value = e.build(work, receipts, output)
    publication = publish_primary(output, value, lambda path: terminal(path, raw / "terminal"))
    consumer = reader_receipt(
        e.TASK,
        output.parent,
        field="hardware_mapping_ready_score",
        expected=value["hardware_mapping_ready_score"],
    )
    atomic_json(
        raw / "terminal_validation.json",
        dict(
            publication=publication,
            consumer=consumer,
            primary_sha256=publication["primary_sha256"],
            passed=consumer["passed"],
        ),
    )
    e.progress(
        "complete_" + value["verdict_class"], value["completed_count"], value["censored_count"]
    )
    return 0 if value["required_checks_passed"] and consumer["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    """The direct CLI also owns fresh bounded replay and calendar validation."""
    e.progress("start_no_model_load_MODEL_SPECS_empty_language_model_calls_zero")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20261101")
    parser.add_argument("--root", type=Path, default=i.ROOT)
    parser.add_argument("--output", type=Path, default=i.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.cold_replay:
            replay(args.cold_replay)
            return 0
        if len(args.date) != 8:
            raise ValueError("calendar_date")
        datetime.strptime(args.date, "%Y%m%d")
        with budget(4800):
            return run(
                args.output.absolute(), args.date, args.root.absolute(), args.private_control
            )
    except (OSError, ValueError, KeyError, TypeError, TimeoutError) as error:
        print(f"[exp8707] error={error}", file=sys.stderr, flush=True)
        return 1

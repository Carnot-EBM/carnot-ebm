"""Publish current read-only board custody with an explicit validation plan.

Spec refs: REQ-REPORT-7889 and SCENARIO-REPORT-7889-CLI.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.experiment_7889_v684_hardware_evidence import cold_reduce, read_evidence

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = "results/experiment_7889_v684_hardware_evidence.json"
MODULE = "python/carnot/reporting/experiment_7889_v684_hardware_evidence.py"
SCRIPT = "scripts/experiments/experiment_7889_v684_hardware_evidence.py"
TEST = "tests/python/test_experiment_7889_v684_hardware_evidence.py"
CONSUMERS = (
    "tests/python/test_experiment_7876_v683_hardware_evidence.py",
    "tests/python/test_experiment_7862_v682_hardware_evidence.py",
)
LIBRARIES = (
    "python/carnot/reporting/current_work_receipt.py",
    "python/carnot/reporting/experiment_7303_validation_scope.py",
    "python/carnot/reporting/experiment_7847_v681_hardware_evidence.py",
    "python/carnot/reporting/experiment_7862_v682_hardware_evidence.py",
    "python/carnot/reporting/experiment_7876_v683_hardware_evidence.py",
)


def progress(started: float, phase: str, event: str, units: int) -> None:
    """Expose elapsed work at each boundary and while a child remains active."""
    print(
        f"[exp7889] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f} completed_units={units}",
        flush=True,
    )


def manifest(private: Path) -> list[dict[str, Any]]:
    """Freeze concrete argv and deadlines before reading validation results."""
    py = str(ROOT / ".venv/bin/python")
    pytest = str(ROOT / ".venv/bin/pytest")
    cov = str(ROOT / ".venv/bin/coverage")
    ruff = str(ROOT / ".venv/bin/ruff")
    mypy = str(ROOT / ".venv/bin/mypy")
    common = ["-n", "0", "-o", "addopts=", "--no-cov"]
    include = ",".join(str(ROOT / name) for name in (MODULE, SCRIPT))
    import_code = (
        "import importlib,json,pathlib;"
        "names=['carnot.reporting.current_work_receipt',"
        "'carnot.reporting.experiment_7303_validation_scope',"
        "'carnot.reporting.experiment_7847_v681_hardware_evidence',"
        "'carnot.reporting.experiment_7862_v682_hardware_evidence',"
        "'carnot.reporting.experiment_7876_v683_hardware_evidence',"
        "'carnot.reporting.experiment_7889_v684_hardware_evidence'];"
        "d={n:str(pathlib.Path(importlib.import_module(n).__file__).resolve()) for n in names};"
        "print(json.dumps({'resolved_imports':d}));"
        f"assert all(pathlib.Path(p).is_relative_to(pathlib.Path('{ROOT / 'python'}')) for p in d.values())"
    )
    unit = private / "unit.coverage"
    success = private / "success.coverage"
    missing = private / "missing.coverage"
    replay = private / "replay.coverage"
    combined = private / "combined.coverage"
    raw = [
        ("worktree_imports", [py, "-u", "-c", import_code], 30, "required"),
        (
            "affected_pytest",
            [pytest, TEST, *CONSUMERS, "-q", *common, f"--basetemp={private / 'test-temp'}"],
            180,
            "required",
        ),
        (
            "unit_coverage",
            [
                cov,
                "run",
                "--branch",
                f"--data-file={unit}",
                f"--include={include}",
                "-m",
                "pytest",
                TEST,
                "-q",
                *common,
                f"--basetemp={private / 'coverage-temp'}",
            ],
            180,
            "required",
        ),
        (
            "cli_success",
            [
                cov,
                "run",
                "--branch",
                f"--data-file={success}",
                f"--include={include}",
                str(ROOT / SCRIPT),
                "--date",
                "20260929",
                "--root",
                str(ROOT),
                "--output",
                str(private / "success.json"),
                "--evidence-only",
            ],
            30,
            "required",
        ),
        (
            "cli_missing_input",
            [
                cov,
                "run",
                "--branch",
                f"--data-file={missing}",
                f"--include={include}",
                str(ROOT / SCRIPT),
                "--date",
                "20260929",
                "--root",
                str(private / "absent"),
                "--output",
                str(private / "missing.json"),
                "--evidence-only",
            ],
            30,
            "required",
        ),
        (
            "cold_replay",
            [
                cov,
                "run",
                "--branch",
                f"--data-file={replay}",
                f"--include={include}",
                str(ROOT / SCRIPT),
                "--root",
                str(ROOT),
                "--cold-replay",
                str(private / "success.json"),
            ],
            30,
            "required",
        ),
        (
            "coverage_combine",
            [
                cov,
                "combine",
                f"--data-file={combined}",
                str(unit),
                str(success),
                str(missing),
                str(replay),
            ],
            30,
            "required",
        ),
        (
            "changed_coverage",
            [
                cov,
                "report",
                f"--data-file={combined}",
                f"--include={include}",
                "--show-missing",
                "--fail-under=100",
            ],
            30,
            "required",
        ),
        ("ruff_check", [ruff, "check", MODULE, SCRIPT, TEST], 30, "required"),
        ("ruff_format", [ruff, "format", "--check", MODULE, SCRIPT, TEST], 30, "required"),
        ("mypy", [mypy, "--strict", MODULE, SCRIPT], 60, "required"),
        ("scoped_spec", [py, "-u", "scripts/check_spec_coverage.py", TEST], 30, "required"),
        ("full_pytest", [pytest, "tests/python", "-q"], 900, "diagnostic"),
    ]
    return [
        {"name": name, "argv": argv, "deadline_s": deadline, "classification": classification}
        for name, argv, deadline, classification in raw
    ]


def terminal_manifest(private: Path) -> list[dict[str, Any]]:
    """Name the exact final readers before any validation child starts."""
    py = str(ROOT / ".venv/bin/python")
    candidate = str(private / "terminal-candidate.json")
    return [
        {
            "name": "adversarial_verify",
            "argv": [py, "-u", "scripts/adversarial_verify.py", "--json", candidate],
            "deadline_s": 30,
            "classification": "terminal_validator",
        },
        {
            "name": "strict_rows",
            "argv": [py, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", candidate],
            "deadline_s": 30,
            "classification": "terminal_validator",
        },
    ]


def run_child(spec: dict[str, Any], private: Path, started: float, units: int) -> dict[str, Any]:
    """Wait for one child, then move its closed log to its byte hash path."""
    progress(started, str(spec["name"]), "before_subprocess", units)
    cmd = CommandSpec(spec["name"], tuple(spec["argv"]), spec["classification"], spec["deadline_s"])
    receipt = run_commands(ROOT, [cmd], log_dir=private / "open-logs", heartbeat_s=30)[0]
    source = Path(receipt["log_path"])
    sealed = (
        private
        / "logs"
        / str(spec["name"])
        / f"{receipt['log_sha256'].removeprefix('sha256:')}.log"
    )
    sealed.parent.mkdir(parents=True, exist_ok=True)
    source.replace(sealed)
    receipt.update(spec)
    receipt["log_path"] = str(sealed)
    progress(started, str(spec["name"]), "after_subprocess", units + 1)
    return receipt


def main(argv: list[str] | None = None) -> int:
    """Run current primitives and publish the same bytes that validators see."""
    started = time.monotonic()
    progress(started, "entrypoint", "start", 0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260929")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--evidence-only", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.cold_replay:
        progress(started, "cold_replay", "before_reduction", 0)
        print(
            json.dumps(cold_reduce(root, json.loads(args.cold_replay.read_text())), sort_keys=True),
            flush=True,
        )
        progress(started, "cold_replay", "after_reduction", 1)
        return 0
    progress(started, "preconditions", "before_read", 0)
    result = read_evidence(root, args.date)
    progress(started, "preconditions", "after_read", len(result["rows"]))
    output = args.output or root / OUTPUT
    if args.evidence_only:
        atomic_json(output, result)
        progress(started, "evidence_only", "written", len(result["rows"]))
        return 0
    if root != ROOT:
        raise ValueError("full validation requires the worktree root")
    closure = {
        name: sha256_file(ROOT / name) for name in (MODULE, SCRIPT, TEST, *CONSUMERS, *LIBRARIES)
    }
    source_identity = {
        name: row["sha256"] for name, row in result["source_artifact_hashes"].items()
    }
    key = canonical_hash(
        {"closure": closure, "sources": source_identity, "date": args.date}
    ).removeprefix("sha256:")
    private = Path(tempfile.gettempdir()) / "carnot-7889" / key
    private.mkdir(parents=True, exist_ok=True)
    commands = manifest(private)
    terminal_checks = terminal_manifest(private)
    frozen = {
        "task_id": result["task_id"],
        "commands": commands,
        "terminal_validators": terminal_checks,
        "source_closure": source_identity,
        "code_closure": {name: closure[name] for name in (MODULE, SCRIPT, *LIBRARIES)},
        "test_closure": {name: closure[name] for name in (TEST, *CONSUMERS)},
        "input_configuration_hash": key,
        "e2e_applicability": {
            **{
                f"E2E-{number:03d}": "inapplicable: owns another producer or requires physical execution"
                for number in range(1, 18)
            },
            "task_cli_e2e": "applicable: real private success, missing input and cold replay",
        },
    }
    manifest_path = private / "validation_command_manifest.json"
    atomic_json(manifest_path, frozen)
    result["validation_command_manifest_path"] = str(manifest_path)
    result["validation_command_manifest_sha256"] = sha256_file(manifest_path)
    result["field_principles"]["validation_command_manifest_sha256"] = (
        "Authenticate the frozen current validation plan."
    )
    checkpoint = private / "completed_units.json"
    previous = json.loads(checkpoint.read_text()).get("checks", []) if checkpoint.is_file() else []
    receipts: list[dict[str, Any]] = []
    for spec in commands:
        if len(receipts) < len(previous):
            receipt = previous[len(receipts)]
            log = Path(receipt["log_path"])
            if (
                any(
                    receipt.get(field) != spec[field]
                    for field in ("name", "argv", "classification")
                )
                or not log.is_file()
                or sha256_file(log) != receipt["log_sha256"]
            ):
                raise ValueError("checkpoint_receipt_changed")
            progress(started, spec["name"], "resumed", len(receipts) + 1)
        else:
            receipt = run_child(spec, private, started, len(receipts))
            atomic_json(checkpoint, {"checks": [*receipts, receipt]})
        receipts.append(receipt)
        result["observed_child_commands"].append(
            {
                "name": receipt["name"],
                "argv": receipt["argv"],
                "classification": receipt["classification"],
            }
        )
        if spec["name"] == "worktree_imports":
            result["resolved_imports"] = receipt.get("resolved_imports", {})
    required = [row for row in receipts if row["classification"] == "required"]
    required_passed = all(row["passed"] for row in required)
    diagnostic = next(row for row in receipts if row["name"] == "full_pytest")
    result["validation_receipts"] = {"checks": receipts, "required_checks_passed": required_passed}
    result["repository_health"] = {
        "status": "healthy" if diagnostic["passed"] else "degraded_open",
        "full_pytest": diagnostic,
        "affects_required_checks": False,
    }
    if not required_passed:
        result["honest_verdict"] = "complete_disqualified_required_checks"
        result["verdict_class"] = "disqualified"
        result["hardware_evidence_ready_score"] = 0
    result["acceptance_gate_results"]["readiness"] = int(
        required_passed and result["hardware_evidence_ready_score"] == 1
    )
    result["duration_s"] = time.monotonic() - started
    result["phase_spans"]["validation_s"] = (
        result["duration_s"] - result["phase_spans"]["evidence_read_s"]
    )
    result["reproducibility_checksum"] = canonical_hash(
        {
            "base": result["reproducibility_checksum"],
            "manifest": result["validation_command_manifest_sha256"],
            "seed": 0,
        }
    )
    candidate = private / "terminal-candidate.json"
    atomic_json(candidate, result)
    reports = [
        run_child(spec, private, started, len(receipts) + index)
        for index, spec in enumerate(terminal_checks)
    ]
    try:
        flags = json.loads(Path(reports[0]["log_path"]).read_text())["flagged_count"]
    except (ValueError, KeyError, TypeError):
        flags = 1
    if flags or not all(report["passed"] for report in reports):
        result["flagged_adversarial"] = bool(flags)
        result["honest_verdict"] = "complete_disqualified_terminal_validation"
        result["verdict_class"] = "disqualified"
        result["hardware_evidence_ready_score"] = 0
        result["acceptance_gate_results"]["readiness"] = 0
        atomic_json(candidate, result)
        reports = [
            run_child(spec, private, started, len(receipts) + index)
            for index, spec in enumerate(terminal_checks)
        ]
    sidecar = private / "terminal_validation_reports.json"
    atomic_json(sidecar, {"candidate_sha256": sha256_file(candidate), "reports": reports})
    if not all(report["passed"] for report in reports):
        raise ValueError("terminal_validation_failed")
    atomic_json(output, result)
    progress(started, "final", "written", len(receipts) + len(reports))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

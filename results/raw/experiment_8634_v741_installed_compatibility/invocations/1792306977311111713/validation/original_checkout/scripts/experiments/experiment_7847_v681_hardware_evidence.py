"""Run the direct hardware evidence reader and its frozen validation scope.

This command reads files and runs local checks. It never connects to a board.
Spec refs: REQ-REPORT-7847 and SCENARIO-REPORT-7847-CLI.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7847_v681_hardware_evidence import cold_reduce, read_evidence

ROOT = Path(__file__).resolve().parents[2]
RAW = "results/raw/experiment_7847_v681_hardware_evidence"
OUTPUT = "results/experiment_7847_v681_hardware_evidence.json"
TESTS = (
    "tests/python/test_experiment_7847_v681_hardware_evidence.py",
    "tests/python/test_experiment_7847_validation.py",
    "tests/python/test_current_work_receipt.py",
)
CODE = (
    "python/carnot/reporting/experiment_7847_v681_hardware_evidence.py",
    "scripts/experiments/experiment_7847_v681_hardware_evidence.py",
)
DEPENDENCIES = ("python/carnot/reporting/current_work_receipt.py", "scripts/adversarial_verify.py")
REQUIRED = (
    "worktree_imports",
    "affected_pytest",
    "unit_coverage",
    "cli_e2e",
    "cold_replay",
    "changed_coverage",
    "ruff_check",
    "ruff_format",
    "mypy",
    "scoped_spec",
    "adversarial_verify",
    "strict_rows",
)


def progress(started: float, phase: str, event: str, units: int) -> None:
    """Show active work so a long child cannot look like a silent stall."""
    print(
        f"[exp7847] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f} completed_units={units}",
        flush=True,
    )


def manifest(root: Path, private: Path, candidate: Path) -> list[dict[str, Any]]:
    """Freeze exact argv and deadlines before the first child starts."""
    py = str(ROOT / ".venv/bin/python")
    pytest = str(ROOT / ".venv/bin/pytest")
    cov = str(ROOT / ".venv/bin/coverage")
    unit_data = private / "unit.coverage"
    cli_data = private / "cli.coverage"
    merged = private / "combined.coverage"
    code_include = ",".join(str(ROOT / path) for path in CODE)
    imports = (
        "import importlib,json,pathlib; names=['carnot.reporting.current_work_receipt',"
        "'carnot.reporting.experiment_7847_v681_hardware_evidence']; "
        "d={n:str(pathlib.Path(importlib.import_module(n).__file__).resolve()) for n in names}; "
        "print(json.dumps({'resolved_imports':d},sort_keys=True)); "
        "assert all(pathlib.Path(p).is_relative_to(pathlib.Path('"
        + str(ROOT / "python")
        + "')) for p in d.values())"
    )
    common = ["-n", "0", "-o", "addopts=", "--no-cov"]
    commands = [
        ("worktree_imports", [py, "-u", "-c", imports], 30),
        (
            "affected_pytest",
            [pytest, *TESTS, "-q", *common, f"--basetemp={private / 'test-temp'}"],
            120,
        ),
        (
            "unit_coverage",
            [
                cov,
                "run",
                f"--data-file={unit_data}",
                f"--include={code_include}",
                "-m",
                "pytest",
                *TESTS,
                "-q",
                *common,
                f"--basetemp={private / 'coverage-temp'}",
            ],
            120,
        ),
        (
            "cli_e2e",
            [
                cov,
                "run",
                f"--data-file={cli_data}",
                f"--include={code_include}",
                CODE[1],
                "--date",
                "20260929",
                "--root",
                str(root),
                "--output",
                str(private / "cli-candidate.json"),
                "--evidence-only",
            ],
            30,
        ),
        (
            "cold_replay",
            [
                py,
                "-u",
                CODE[1],
                "--root",
                str(root),
                "--cold-replay",
                str(private / "cli-candidate.json"),
            ],
            30,
        ),
        (
            "coverage_combine",
            [cov, "combine", f"--data-file={merged}", str(unit_data), str(cli_data)],
            30,
        ),
        (
            "changed_coverage",
            [
                cov,
                "report",
                f"--data-file={merged}",
                f"--include={code_include}",
                "--show-missing",
                "--fail-under=100",
            ],
            30,
        ),
        ("ruff_check", [str(ROOT / ".venv/bin/ruff"), "check", *CODE, *TESTS], 30),
        ("ruff_format", [str(ROOT / ".venv/bin/ruff"), "format", "--check", *CODE, *TESTS], 30),
        ("mypy", [str(ROOT / ".venv/bin/mypy"), "--strict", *CODE], 60),
        ("scoped_spec", [py, "-u", "scripts/check_spec_coverage.py", *TESTS], 30),
        (
            "adversarial_verify",
            [py, "-u", "scripts/adversarial_verify.py", "--json", str(candidate)],
            30,
        ),
        (
            "strict_rows",
            [py, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)],
            30,
        ),
        (
            "repository_health_180s",
            [pytest, "tests/python", "-q", *common, f"--basetemp={private / 'health-temp'}"],
            180,
        ),
    ]
    return [
        {
            "name": name,
            "argv": argv,
            "deadline_s": timeout,
            "classification": "diagnostic" if name == "repository_health_180s" else "required",
        }
        for name, argv, timeout in commands
    ]


def run_child(
    root: Path, spec: dict[str, Any], logs: Path, started: float, units: int
) -> dict[str, Any]:
    """Supervise one owned process and seal its closed output once it exits."""
    name = spec["name"]
    progress(started, name, "before_subprocess", units)
    began = time.monotonic()
    scratch = logs / f".{name}-{uuid.uuid4().hex}.open"
    scratch.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": f"{ROOT / 'python'}:{ROOT}", "JAX_PLATFORMS": "cpu"}
    with scratch.open("wb") as stream:
        child = subprocess.Popen(
            spec["argv"],
            cwd=root,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        while child.poll() is None:
            elapsed = time.monotonic() - began
            if elapsed >= spec["deadline_s"]:
                child.terminate()
                try:
                    child.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                break
            try:
                child.wait(timeout=min(30, max(0.1, spec["deadline_s"] - elapsed)))
            except subprocess.TimeoutExpired:
                progress(started, name, "heartbeat", units)
    digest = sha256_file(scratch)
    sealed = logs / name / f"{digest.removeprefix('sha256:')}.log"
    sealed.parent.mkdir(parents=True, exist_ok=True)
    scratch.replace(sealed)
    receipt = {
        **spec,
        "exit_code": child.returncode,
        "timed_out": time.monotonic() - began >= spec["deadline_s"],
        "passed": child.returncode == 0 and time.monotonic() - began < spec["deadline_s"],
        "duration_s": time.monotonic() - began,
        "log_path": str(sealed),
        "log_sha256": digest,
    }
    if name == "worktree_imports":
        try:
            resolved = json.loads(sealed.read_text().splitlines()[-1])["resolved_imports"]
            receipt["resolved_imports"] = resolved
            receipt["passed"] = receipt["passed"] and bool(resolved)
        except (ValueError, KeyError, IndexError, TypeError):
            receipt["resolved_imports"] = {}
            receipt["passed"] = False
    progress(started, name, "after_subprocess", units + 1)
    return receipt


def main(argv: list[str] | None = None) -> int:
    """Write a terminal record after all owned checks finish."""
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
    if args.cold_replay is not None:
        progress(started, "cold_replay", "before_reduction", 0)
        candidate = json.loads(args.cold_replay.read_text())
        print(json.dumps(cold_reduce(root, candidate), sort_keys=True), flush=True)
        progress(started, "cold_replay", "after_reduction", 1)
        return 0
    progress(started, "preconditions", "before_read", 0)
    result = read_evidence(root, args.date)
    progress(started, "preconditions", "after_read", len(result["rows"]))
    output = args.output or root / OUTPUT
    if args.evidence_only:
        atomic_json(output, result)
        from scripts.adversarial_verify import verify_artifact

        report = verify_artifact(output)
        result["flagged_adversarial"] = bool(report["flag_count"])
        atomic_json(output, result)
        progress(started, "evidence_only", "written", len(result["rows"]))
        return 0
    if root != ROOT:
        raise ValueError("full validation requires the worktree root")
    raw = root / RAW
    raw.mkdir(parents=True, exist_ok=True)
    closure_hashes = {path: sha256_file(ROOT / path) for path in (*CODE, *DEPENDENCIES, *TESTS)}
    key = canonical_hash(
        {"inputs": result["source_artifact_hashes"], "closure": closure_hashes, "date": args.date}
    )
    private = raw / "checkpoints" / key.removeprefix("sha256:")
    private.mkdir(parents=True, exist_ok=True)
    candidate = private / "candidate.json"
    result["validation_receipts"] = {"checks": [], "required_checks_passed": False}
    atomic_json(candidate, result)
    commands = manifest(root, private, candidate)
    frozen = {
        "task_id": "exp7847-hardware-evidence",
        "required_names": list(REQUIRED),
        "commands": commands,
        "source_closure": {path: sha256_file(ROOT / path) for path in (*CODE, *DEPENDENCIES)},
        "test_closure": {path: sha256_file(ROOT / path) for path in TESTS},
        "config_hash": canonical_hash({"code": CODE, "tests": TESTS, "date": args.date}),
        "input_configuration_hash": key,
        "import_call_closure": {
            "imports": [
                "carnot.reporting.current_work_receipt",
                "carnot.reporting.experiment_7847_v681_hardware_evidence",
                "scripts.adversarial_verify",
            ],
            "calls": [
                "atomic_json",
                "canonical_hash",
                "sha256_file",
                "read_evidence",
                "cold_reduce",
                "verify_artifact",
            ],
        },
        "e2e_applicability": {
            **{
                f"E2E-{index:03d}": "inapplicable: no training, sampling, native binding, or ARC execution"
                for index in range(1, 15)
            },
            "task_cli_e2e": "applicable: real private-output CLI and cold replay",
        },
    }
    manifest_path = private / "validation_command_manifest.json"
    atomic_json(manifest_path, frozen)
    result["validation_command_manifest_path"] = str(manifest_path)
    result["validation_command_manifest_sha256"] = sha256_file(manifest_path)
    result["observed_child_commands"] = []
    logs = raw / "validation" / uuid.uuid4().hex
    receipts: list[dict[str, Any]] = []
    checkpoint = private / "completed_units.json"
    prior_units = json.loads(checkpoint.read_text())["checks"] if checkpoint.is_file() else []
    for spec in commands:
        if spec["name"] in {"adversarial_verify", "strict_rows"}:
            atomic_json(candidate, result)
        if len(receipts) < len(prior_units):
            receipt = prior_units[len(receipts)]
            if (
                any(receipt.get(key) != spec[key] for key in ("name", "argv", "classification"))
                or sha256_file(Path(receipt["log_path"])) != receipt["log_sha256"]
            ):
                raise ValueError("checkpoint_receipt_changed")
            progress(started, spec["name"], "resumed", len(receipts) + 1)
        else:
            receipt = run_child(root, spec, logs, started, len(receipts))
            atomic_json(checkpoint, {"checks": [*receipts, receipt]})
        receipts.append(receipt)
        result["observed_child_commands"].append(
            {
                "name": receipt["name"],
                "argv": receipt["argv"],
                "classification": receipt["classification"],
            }
        )
        if spec["name"] == "adversarial_verify":
            report_text = Path(receipt["log_path"]).read_text()
            try:
                report = json.loads(report_text)
                result["flagged_adversarial"] = bool(report["flagged_count"])
            except (ValueError, TypeError):
                result["flagged_adversarial"] = True
    passed = all(any(r["name"] == name and r["passed"] for r in receipts) for name in REQUIRED)
    result["validation_receipts"] = {"checks": receipts, "required_checks_passed": passed}
    result["repository_health"] = next(r for r in receipts if r["name"] == "repository_health_180s")
    if not passed or result["flagged_adversarial"]:
        result["honest_verdict"] = "complete_disqualified_required_checks"
        result["verdict_class"] = "disqualified"
    result["acceptance_gate_results"]["readiness"] = int(
        passed
        and not result["flagged_adversarial"]
        and result["hardware_inventory_ready_score"] == 1
        and result["service_check"]["qualified"]
    )
    result["duration_s"] = time.monotonic() - started
    result["phase_spans"]["validation_s"] = (
        result["duration_s"] - result["phase_spans"]["evidence_read_s"]
    )
    result["reproducibility_checksum"] = canonical_hash(
        {
            "base": result["reproducibility_checksum"],
            "manifest": result["validation_command_manifest_sha256"],
            "seed": result["random_seed"],
        }
    )
    atomic_json(output, result)
    progress(started, "final", "written", len(receipts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

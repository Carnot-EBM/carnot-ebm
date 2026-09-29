"""Read board history and validate the private evidence reader.

No command here connects to a board. Child output stays in /tmp so the active
results guard cannot move one side of an atomic rename to another filesystem.
Spec refs: REQ-REPORT-7862 and SCENARIO-REPORT-7862-CLI.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7862_v682_hardware_evidence import cold_reduce, read_evidence

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = "results/experiment_7862_v682_hardware_evidence.json"
CODE = (
    "python/carnot/reporting/experiment_7862_v682_hardware_evidence.py",
    "scripts/experiments/experiment_7862_v682_hardware_evidence.py",
)
TESTS = ("tests/python/test_experiment_7862_v682_hardware_evidence.py",)
LIBRARIES = (
    "python/carnot/reporting/current_work_receipt.py",
    "python/carnot/reporting/experiment_7847_v681_hardware_evidence.py",
)
REQUIRED = (
    "worktree_imports",
    "affected_pytest",
    "unit_coverage",
    "cli_success",
    "cli_missing_source",
    "cold_replay",
    "coverage_combine",
    "changed_coverage",
    "ruff_check",
    "ruff_format",
    "mypy",
    "scoped_spec",
    "adversarial_verify",
    "strict_rows",
)


def progress(started: float, phase: str, event: str, units: int) -> None:
    """Emit a flushed boundary before silence could hide a stalled child."""
    print(
        f"[exp7862] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f} completed_units={units}",
        flush=True,
    )


def manifest(private: Path, candidate: Path) -> list[dict[str, Any]]:
    """Freeze commands and deadlines before a child can change an outcome."""
    py = str(ROOT / ".venv/bin/python")
    pytest = str(ROOT / ".venv/bin/pytest")
    cov = str(ROOT / ".venv/bin/coverage")
    ruff = str(ROOT / ".venv/bin/ruff")
    mypy = str(ROOT / ".venv/bin/mypy")
    include = ",".join(str(ROOT / name) for name in CODE)
    unit = private / "unit.coverage"
    success = private / "success.coverage"
    missing = private / "missing.coverage"
    combined = private / "combined.coverage"
    common = ["-n", "0", "-o", "addopts=", "--no-cov"]
    import_code = (
        "import importlib,json,pathlib; names=['carnot.reporting.current_work_receipt',"
        "'carnot.reporting.experiment_7847_v681_hardware_evidence',"
        "'carnot.reporting.experiment_7862_v682_hardware_evidence'];"
        "d={n:str(pathlib.Path(importlib.import_module(n).__file__).resolve()) for n in names};"
        "print(json.dumps({'resolved_imports':d}));"
        f"assert all(pathlib.Path(p).is_relative_to(pathlib.Path('{ROOT / 'python'}')) for p in d.values())"
    )
    raw = [
        ("worktree_imports", [py, "-u", "-c", import_code], 30),
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
                f"--data-file={unit}",
                f"--include={include}",
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
            "cli_success",
            [
                cov,
                "run",
                f"--data-file={success}",
                f"--include={include}",
                CODE[1],
                "--date",
                "20260929",
                "--root",
                str(ROOT),
                "--output",
                str(private / "cli-success.json"),
                "--evidence-only",
            ],
            30,
        ),
        (
            "cli_missing_source",
            [
                cov,
                "run",
                f"--data-file={missing}",
                f"--include={include}",
                CODE[1],
                "--date",
                "20260929",
                "--root",
                str(private / "absent"),
                "--output",
                str(private / "cli-missing.json"),
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
                str(ROOT),
                "--cold-replay",
                str(private / "cli-success.json"),
            ],
            30,
        ),
        (
            "coverage_combine",
            [cov, "combine", f"--data-file={combined}", str(unit), str(success), str(missing)],
            30,
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
        ),
        ("ruff_check", [ruff, "check", *CODE, *TESTS], 30),
        ("ruff_format", [ruff, "format", "--check", *CODE, *TESTS], 30),
        ("mypy", [mypy, "--strict", *CODE], 60),
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
            "deadline_s": deadline,
            "classification": "diagnostic" if name == "repository_health_180s" else "required",
        }
        for name, argv, deadline in raw
    ]


def run_child(spec: dict[str, Any], private: Path, started: float, units: int) -> dict[str, Any]:
    """Supervise only this process group and seal its log after handles close."""
    name = str(spec["name"])
    progress(started, name, "before_subprocess", units)
    began = time.monotonic()
    scratch = private / "logs" / f".{name}.open"
    scratch.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": f"{ROOT / 'python'}:{ROOT}", "JAX_PLATFORMS": "cpu"}
    timed_out = False
    with scratch.open("wb") as stream:
        child = subprocess.Popen(
            spec["argv"],
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        while child.poll() is None:
            remaining = spec["deadline_s"] - (time.monotonic() - began)
            if remaining <= 0:
                timed_out = True
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                break
            try:
                child.wait(timeout=min(30, remaining))
            except subprocess.TimeoutExpired:
                progress(started, name, "heartbeat", units)
    digest = sha256_file(scratch)
    sealed = private / "logs" / name / f"{digest.removeprefix('sha256:')}.log"
    sealed.parent.mkdir(parents=True, exist_ok=True)
    scratch.replace(sealed)
    receipt = {
        **spec,
        "exit_code": child.returncode,
        "timed_out": timed_out,
        "passed": child.returncode == 0 and not timed_out,
        "duration_s": time.monotonic() - began,
        "log_path": str(sealed),
        "log_sha256": digest,
    }
    if name == "worktree_imports":
        try:
            receipt["resolved_imports"] = json.loads(sealed.read_text().splitlines()[-1])[
                "resolved_imports"
            ]
            receipt["passed"] = receipt["passed"] and bool(receipt["resolved_imports"])
        except (ValueError, KeyError, IndexError, TypeError):
            receipt["resolved_imports"] = {}
            receipt["passed"] = False
    progress(started, name, "after_subprocess", units + 1)
    return receipt


def main(argv: list[str] | None = None) -> int:
    """Publish one terminal record after all owned required checks finish."""
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
        from scripts.adversarial_verify import verify_artifact

        report = verify_artifact(output)
        result["flagged_adversarial"] = bool(report["flag_count"])
        atomic_json(output, result)
        progress(started, "evidence_only", "written", len(result["rows"]))
        return 0
    if root != ROOT:
        raise ValueError("full validation requires the worktree root")
    closure = {name: sha256_file(ROOT / name) for name in (*CODE, *LIBRARIES, *TESTS)}
    key = canonical_hash(
        {"sources": result["source_artifact_hashes"], "closure": closure, "date": args.date}
    ).removeprefix("sha256:")
    private = Path(tempfile.gettempdir()) / "carnot-7862" / key
    private.mkdir(parents=True, exist_ok=True)
    candidate = private / "pending-candidate.json"
    commands = manifest(private, candidate)
    frozen = {
        "task_id": result["task_id"],
        "required_names": list(REQUIRED),
        "commands": commands,
        "source_closure": {name: closure[name] for name in (*CODE, *LIBRARIES)},
        "test_closure": {name: closure[name] for name in TESTS},
        "input_configuration_hash": key,
        "e2e_applicability": {
            **{
                f"E2E-{i:03d}": "inapplicable: no training, sampling, binding, or ARC execution"
                for i in range(1, 16)
            },
            "task_cli_e2e": "applicable: private real CLI success, missing input, and replay",
        },
    }
    manifest_path = private / "validation_command_manifest.json"
    atomic_json(manifest_path, frozen)
    result["validation_command_manifest_path"] = str(manifest_path)
    result["validation_command_manifest_sha256"] = sha256_file(manifest_path)
    checkpoint = private / "completed_units.json"
    previous = json.loads(checkpoint.read_text())["checks"] if checkpoint.is_file() else []
    receipts: list[dict[str, Any]] = []
    for spec in commands:
        if spec["name"] in {"adversarial_verify", "strict_rows"}:
            atomic_json(candidate, result)
        if len(receipts) < len(previous):
            receipt = previous[len(receipts)]
            if (
                any(
                    receipt.get(field) != spec[field]
                    for field in ("name", "argv", "classification")
                )
                or sha256_file(Path(receipt["log_path"])) != receipt["log_sha256"]
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
        if spec["name"] == "adversarial_verify":
            try:
                report = json.loads(Path(receipt["log_path"]).read_text())
                result["flagged_adversarial"] = bool(report["flagged_count"])
            except (ValueError, KeyError, TypeError):
                result["flagged_adversarial"] = True
    required_passed = all(
        any(r["name"] == name and r["passed"] for r in receipts) for name in REQUIRED
    )
    result["validation_receipts"] = {"checks": receipts, "required_checks_passed": required_passed}
    result["repository_health"] = next(r for r in receipts if r["name"] == "repository_health_180s")
    if not required_passed or result["flagged_adversarial"]:
        result["honest_verdict"] = "complete_disqualified_required_checks"
        result["verdict_class"] = "disqualified"
    result["acceptance_gate_results"]["readiness"] = int(
        required_passed
        and not result["flagged_adversarial"]
        and result["hardware_evidence_ready_score"] == 1
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
    result["field_principles"]["validation_command_manifest_sha256"] = (
        "Freeze the exact validation plan."
    )
    atomic_json(output, result)
    progress(started, "final", "written", len(receipts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

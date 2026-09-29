"""Run the V683 read-only board audit and publish one terminal receipt.

Spec refs: REQ-REPORT-7876 and SCENARIO-REPORT-7876-CLI.
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
from carnot.reporting.experiment_7876_v683_hardware_evidence import cold_reduce, read_evidence

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = "results/experiment_7876_v683_hardware_evidence.json"
CODE = (
    "python/carnot/reporting/experiment_7876_v683_hardware_evidence.py",
    "scripts/experiments/experiment_7876_v683_hardware_evidence.py",
)
TEST = "tests/python/test_experiment_7876_v683_hardware_evidence.py"
LIBRARIES = (
    "python/carnot/reporting/current_work_receipt.py",
    "python/carnot/reporting/experiment_7847_v681_hardware_evidence.py",
    "python/carnot/reporting/experiment_7862_v682_hardware_evidence.py",
)


def _inherited_full_failure() -> dict[str, Any] | None:
    """Keep an earlier owned required failure without rerunning a long suite."""
    path = ROOT / OUTPUT
    if not path.is_file():
        return None
    try:
        prior = json.loads(path.read_text())
        receipt = next(
            x for x in prior["validation_receipts"]["checks"] if x["name"] == "full_pytest"
        )
        log = Path(receipt["log_path"])
        if (
            receipt["classification"] == "required"
            and receipt["passed"] is False
            and log.is_file()
            and sha256_file(log) == receipt["log_sha256"]
        ):
            return {**receipt, "inherited_from": {"path": OUTPUT, "sha256": sha256_file(path)}}
    except (OSError, ValueError, KeyError, StopIteration, TypeError):
        return None
    return None


def progress(started: float, phase: str, event: str, units: int) -> None:
    """Print every boundary and child heartbeat with elapsed owned time."""
    print(
        f"[exp7876] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f} completed_units={units}",
        flush=True,
    )


def manifest(private: Path, candidate: Path) -> list[dict[str, Any]]:
    """Freeze exact validation arguments, classes, and deadlines."""
    py = str(ROOT / ".venv/bin/python")
    pytest = str(ROOT / ".venv/bin/pytest")
    cov = str(ROOT / ".venv/bin/coverage")
    ruff = str(ROOT / ".venv/bin/ruff")
    mypy = str(ROOT / ".venv/bin/mypy")
    include = ",".join(str(ROOT / name) for name in CODE)
    common = ["-n", "0", "-o", "addopts=", "--no-cov"]
    import_code = (
        "import importlib,json,pathlib;"
        "names=['carnot.reporting.current_work_receipt',"
        "'carnot.reporting.experiment_7847_v681_hardware_evidence',"
        "'carnot.reporting.experiment_7862_v682_hardware_evidence',"
        "'carnot.reporting.experiment_7876_v683_hardware_evidence'];"
        "d={n:str(pathlib.Path(importlib.import_module(n).__file__).resolve()) for n in names};"
        "print(json.dumps({'resolved_imports':d}));"
        f"assert all(pathlib.Path(p).is_relative_to(pathlib.Path('{ROOT / 'python'}')) for p in d.values())"
    )
    unit = private / "unit.coverage"
    success = private / "success.coverage"
    missing = private / "missing.coverage"
    combined = private / "combined.coverage"
    raw = [
        ("worktree_imports", [py, "-u", "-c", import_code], 30),
        (
            "affected_pytest",
            [pytest, TEST, "-q", *common, f"--basetemp={private / 'test-temp'}"],
            120,
        ),
        (
            "full_pytest",
            [pytest, "tests/python", "-q", *common, f"--basetemp={private / 'full-temp'}"],
            900,
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
                TEST,
                "-q",
                *common,
                f"--basetemp={private / 'cov-temp'}",
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
                str(private / "cli-good.json"),
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
                str(private / "cli-bad.json"),
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
                str(private / "cli-good.json"),
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
        ("ruff_check", [ruff, "check", *CODE, TEST], 30),
        ("ruff_format", [ruff, "format", "--check", *CODE, TEST], 30),
        ("mypy", [mypy, "--strict", *CODE], 60),
        ("scoped_spec", [py, "-u", "scripts/check_spec_coverage.py", TEST], 30),
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
    ]
    return [
        {"name": name, "argv": argv, "deadline_s": deadline, "classification": "required"}
        for name, argv, deadline in raw
    ]


def run_child(spec: dict[str, Any], private: Path, started: float, units: int) -> dict[str, Any]:
    """Supervise one owned process and seal its log after the child exits."""
    name = str(spec["name"])
    progress(started, name, "before_subprocess", units)
    began = time.monotonic()
    scratch = private / "logs" / f".{name}.open"
    scratch.parent.mkdir(parents=True, exist_ok=True)
    timed_out = False
    env = {**os.environ, "PYTHONPATH": f"{ROOT / 'python'}:{ROOT}", "JAX_PLATFORMS": "cpu"}
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
                    child.wait(timeout=5)
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
            imports = json.loads(sealed.read_text().splitlines()[-1])["resolved_imports"]
        except (ValueError, KeyError, IndexError, TypeError):
            imports = {}
        receipt["resolved_imports"] = imports
        receipt["passed"] = receipt["passed"] and bool(imports)
    progress(started, name, "after_subprocess", units + 1)
    return receipt


def main(argv: list[str] | None = None) -> int:
    """Run the frozen checks and publish only after every child has exited."""
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
    closure = {name: sha256_file(ROOT / name) for name in (*CODE, *LIBRARIES, TEST)}
    key = canonical_hash(
        {"sources": result["source_artifact_hashes"], "closure": closure, "date": args.date}
    ).removeprefix("sha256:")
    private = Path(tempfile.gettempdir()) / "carnot-7876" / key
    private.mkdir(parents=True, exist_ok=True)
    candidate = private / "pending-candidate.json"
    commands = manifest(private, candidate)
    inherited = _inherited_full_failure()
    if inherited:
        commands = [
            {**spec, "argv": inherited["argv"], "deadline_s": inherited["deadline_s"],
             "inherited_from": inherited["inherited_from"]}
            if spec["name"] == "full_pytest" else spec
            for spec in commands
        ]
    frozen = {
        "task_id": result["task_id"],
        "commands": commands,
        "source_closure": {name: closure[name] for name in (*CODE, *LIBRARIES)},
        "test_closure": {TEST: closure[TEST]},
        "input_configuration_hash": key,
        "e2e_applicability": {
            **{
                f"E2E-{i:03d}": "inapplicable: no training, sampling, binding, or ARC execution"
                for i in range(1, 18)
            },
            "task_cli_e2e": "applicable: private CLI success, missing input, and cold replay",
        },
    }
    manifest_path = private / "validation_command_manifest.json"
    atomic_json(manifest_path, frozen)
    result["validation_command_manifest_path"] = str(manifest_path)
    result["validation_command_manifest_sha256"] = sha256_file(manifest_path)
    result["field_principles"]["validation_command_manifest_sha256"] = (
        "Freeze the exact validation plan."
    )
    checkpoint = private / "completed_units.json"
    previous = json.loads(checkpoint.read_text()).get("checks", []) if checkpoint.is_file() else []
    receipts: list[dict[str, Any]] = []
    for spec in commands:
        if spec["name"] == "adversarial_verify":
            prelim = all(receipt["passed"] for receipt in receipts)
            if not prelim:
                result["honest_verdict"] = "complete_disqualified_required_checks"
                result["verdict_class"] = "disqualified"
                result["hardware_evidence_ready_score"] = 0
            result["acceptance_gate_results"]["readiness"] = int(
                prelim and result["hardware_evidence_ready_score"] == 1
            )
            atomic_json(candidate, result)
        if len(receipts) < len(previous):
            receipt = previous[len(receipts)]
            path = Path(receipt["log_path"])
            if (
                any(
                    receipt.get(field) != spec[field]
                    for field in ("name", "argv", "classification")
                )
                or not path.is_file()
                or sha256_file(path) != receipt["log_sha256"]
            ):
                raise ValueError("checkpoint_receipt_changed")
            progress(started, str(spec["name"]), "resumed", len(receipts) + 1)
        elif spec["name"] == "full_pytest" and inherited:
            receipt = {**inherited, **spec}
            progress(started, "full_pytest", "inherited_required_failure", len(receipts) + 1)
            atomic_json(checkpoint, {"checks": [*receipts, receipt]})
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
    required_passed = all(receipt["passed"] for receipt in receipts)
    result["validation_receipts"] = {"checks": receipts, "required_checks_passed": required_passed}
    result["repository_health"] = {
        "status": "not_separately_run",
        "full_pytest": next(receipt for receipt in receipts if receipt["name"] == "full_pytest"),
    }
    if not required_passed or result["flagged_adversarial"]:
        result["honest_verdict"] = "complete_disqualified_required_checks"
        result["verdict_class"] = "disqualified"
        result["hardware_evidence_ready_score"] = 0
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
    atomic_json(output, result)
    progress(started, "final", "written", len(receipts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

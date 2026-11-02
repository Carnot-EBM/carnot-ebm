"""Run the current V683 evidence audit (REQ-REPORT-7877)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting import v683_independent_audit as audit


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/experiment_7877_v683_independent_audit.json"
MODULE = "python/carnot/reporting/v683_independent_audit.py"
SCRIPT = "scripts/experiments/experiment_7877_v683_independent_audit.py"
TEST = "tests/python/test_experiment_7877_v683_independent_audit.py"


def progress(start: float, phase: str, done: int) -> None:
    """Expose each boundary and completed unit to the supervising process."""
    print(
        f"exp7877 {phase} elapsed_s={time.monotonic() - start:.3f} completed_units={done}",
        flush=True,
    )


def command_manifest(output_root: Path) -> dict[str, Any]:
    """Freeze exact owned validation vectors and deadlines before measurement."""
    python = str(ROOT / ".venv/bin/python")
    pytest = str(ROOT / ".venv/bin/pytest")
    coverage = str(ROOT / ".venv/bin/coverage")
    ruff = str(ROOT / ".venv/bin/ruff")
    mypy = str(ROOT / ".venv/bin/mypy")
    baseline = ["-n", "0", "-o", "addopts=", "--no-cov", "-q"]
    common = ["--basetemp=" + str(output_root / "pytest")]
    cov_unit = str(output_root / "coverage.unit")
    cov_cli = str(output_root / "coverage.cli")
    combined = str(output_root / "coverage.combined")
    include = f"{MODULE},{SCRIPT}"
    entries = [
        ("affected_pytest", [pytest, TEST, *baseline, *common], "required", 180),
        (
            "full_pytest",
            [pytest, "tests/python", *baseline, "--basetemp=" + str(output_root / "full_pytest")],
            "required",
            900,
        ),
        (
            "coverage_unit",
            [
                coverage,
                "run",
                "--data-file=" + cov_unit,
                "--include=" + include,
                "-m",
                "pytest",
                TEST,
                *baseline,
                *common,
            ],
            "required",
            180,
        ),
        (
            "coverage_cli",
            [
                coverage,
                "run",
                "--data-file=" + cov_cli,
                "--include=" + include,
                SCRIPT,
                "--date",
                "20260929",
                "--science-only",
                "--output-root",
                str(output_root / "coverage_cli"),
            ],
            "required",
            180,
        ),
        (
            "coverage_combine",
            [coverage, "combine", "--data-file=" + combined, cov_unit, cov_cli],
            "required",
            60,
        ),
        (
            "coverage_report",
            [
                coverage,
                "report",
                "--data-file=" + combined,
                "--include=" + include,
                "--show-missing",
                "--fail-under=100",
            ],
            "required",
            60,
        ),
        ("ruff_check", [ruff, "check", MODULE, SCRIPT, TEST], "required", 60),
        ("ruff_format", [ruff, "format", "--check", MODULE, SCRIPT, TEST], "required", 60),
        ("strict_mypy", [mypy, "--strict", MODULE, SCRIPT], "required", 120),
        ("spec_coverage", [python, "scripts/check_spec_coverage.py", TEST], "required", 120),
        (
            "adversarial_verify",
            [
                python,
                "scripts/adversarial_verify.py",
                "--json",
                str(output_root / "candidate.json"),
            ],
            "required",
            60,
        ),
        (
            "strict_rows",
            [
                python,
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(output_root / "candidate.json"),
            ],
            "required",
            60,
        ),
    ]
    return {
        "affected_source": [MODULE, SCRIPT],
        "affected_tests": [TEST],
        "e2e_applicability": {
            "private_cli_and_cold_replay": "applicable",
            "E2E-016": "upstream Exp7868 protocol; audit reads its terminal artifact",
            "E2E-017": "upstream Exp7874 supervisor; audit reads its terminal artifact",
        },
        "commands": [
            {"name": name, "argv": argv, "classification": kind, "deadline_s": seconds}
            for name, argv, kind, seconds in entries
        ],
    }


def _seal(receipt: dict[str, Any], output_root: Path) -> dict[str, Any]:
    """Seal an exited child's closed log at an immutable content address."""
    source = ROOT / receipt["log_path"]
    digest = sha256_file(source)
    target = output_root / "sealed_logs" / f"{receipt['name']}_{digest[7:]}.log"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256_file(target) != digest:
            raise ValueError("sealed_log_collision")
    else:
        with target.open("xb") as stream:
            stream.write(source.read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
    source.unlink()
    return {**receipt, "log_path": str(target), "log_sha256": digest}


def science(output_root: Path, date: str, start: float) -> dict[str, Any]:
    """Run preconditions and write only a private hash-bound candidate."""
    progress(start, "preconditions_begin", 0)
    manifest = output_root / "validation_command_manifest.json"
    frozen = command_manifest(output_root)
    if manifest.is_file() and json.loads(manifest.read_bytes()) != frozen:
        raise ValueError("frozen_validation_manifest_changed")
    atomic_json(manifest, frozen)
    result = audit.build_candidate(ROOT, output_root, manifest, date)
    checkpoint = (
        output_root
        / "checkpoints"
        / result["reproducibility_checksum"].split(":", 1)[1]
        / "science.json"
    )
    stable = {
        key: result[key]
        for key in (
            "task_evidence_rows",
            "rows",
            "recomputed_comparison_rows",
            "gate_check_summary",
        )
    }
    if checkpoint.is_file() and json.loads(checkpoint.read_bytes()) != stable:
        raise ValueError("hash_bound_checkpoint_changed")
    atomic_json(checkpoint, stable)
    atomic_json(output_root / "candidate.json", result)
    atomic_json(
        Path(result["audit_manifest_path"]),
        {
            "task_evidence_rows": result["task_evidence_rows"],
            "recomputed_comparison_rows": result["recomputed_comparison_rows"],
            "source_artifact_hashes": result["source_artifact_hashes"],
        },
    )
    progress(start, "preconditions_end", len(result["task_evidence_rows"]))
    return result


def validate(result: dict[str, Any], output_root: Path, start: float) -> dict[str, Any]:
    """Retain real child exits, then disqualify any failed required scope."""
    manifest = json.loads(Path(result["validation_command_manifest_path"]).read_bytes())
    receipts: list[dict[str, Any]] = []
    for entry in manifest["commands"]:
        progress(start, f"before_subprocess:{entry['name']}", len(receipts))
        spec = CommandSpec(
            entry["name"], tuple(entry["argv"]), entry["classification"], entry["deadline_s"]
        )
        raw = run_commands(ROOT, [spec], log_dir=output_root / "transient_logs", heartbeat_s=30)[0]
        receipt = _seal({**raw, "classification": entry["classification"]}, output_root)
        receipts.append(receipt)
        if entry["name"] == "adversarial_verify":
            try:
                result["flagged_adversarial"] = bool(
                    json.loads(receipt["output_tail"])["flagged_count"]
                )
            except (ValueError, KeyError, TypeError):
                result["flagged_adversarial"] = True
        result["validation_receipts"] = receipts
        result["observed_child_commands"] = receipts
        atomic_json(output_root / "candidate.json", result)
        progress(start, f"after_subprocess:{entry['name']}", len(receipts))
    failed = [
        item for item in receipts if item["classification"] == "required" and not item["passed"]
    ]
    if result["flagged_adversarial"]:
        failed.extend(
            item for item in receipts if item["name"] == "adversarial_verify" and item not in failed
        )
    if failed:
        result["honest_verdict"] = "complete_disqualified_required_v683_validation"
        result["verdict_class"] = "disqualified"
        result["acceptance_gate_results"]["readiness"] = 0
        result["audit_execution_ready_score"] = 0
        result["gate_check_summary"].extend(
            {
                "upstream_id": "Exp7877",
                "path": item["log_path"],
                "hash": item["log_sha256"],
                "artifact_field": item["name"],
                "op": "==",
                "expected": 0,
                "observed": item["exit_code"],
            }
            for item in failed
        )
    else:
        result["audit_execution_ready_score"] = 1
    result["repository_health"]["status"] = "passed" if not failed else "failed_owned_validation"
    result["duration_s"] = time.monotonic() - start
    result["phase_spans"].append(
        {
            "phase": "owned_validation",
            "duration_s": result["duration_s"] - result["phase_spans"][0]["duration_s"],
            "completed_units": len(receipts),
        }
    )
    return result


def main() -> int:
    """Exercise private CLI/replay routes and publish only terminal evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260929")
    parser.add_argument("--science-only", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--check-only", type=Path)
    args = parser.parse_args()
    start = time.monotonic()
    progress(start, "start", 0)
    if args.check_only:
        errors = audit.cold_replay(args.check_only, ROOT)
        progress(start, f"cold_replay:{errors}", 1)
        return int(bool(errors))
    if args.date != "20260929":
        parser.error("run date must be 20260929")
    output_root = args.output_root or Path("/tmp/carnot-7877-v683-20260929")
    result = science(output_root, args.date, start)
    if args.science_only:
        return 0
    result = validate(result, output_root, start)
    candidate = output_root / "candidate.json"
    atomic_json(candidate, result)
    errors = audit.cold_replay(candidate, ROOT)
    if errors:
        result["honest_verdict"] = "complete_disqualified_cold_replay"
        result["verdict_class"] = "disqualified"
        result["acceptance_gate_results"]["readiness"] = 0
        result["audit_execution_ready_score"] = 0
        result["cold_replay_errors"] = errors
    result["duration_s"] = time.monotonic() - start
    for key in result:
        result["field_principles"].setdefault(
            key, "Preserve exact terminal evidence and its scope."
        )
    atomic_json(OUTPUT, result)
    progress(start, f"published:{result['honest_verdict']}", len(result["validation_receipts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

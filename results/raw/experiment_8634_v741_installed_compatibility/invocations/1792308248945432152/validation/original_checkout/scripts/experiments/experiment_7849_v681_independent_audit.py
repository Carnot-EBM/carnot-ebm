"""Execute the current V681 independent audit (REQ-REPORT-7849)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any
import uuid

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting import v681_independent_audit as audit


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/experiment_7849_v681_independent_audit.json"


def progress(start: float, phase: str, done: int) -> None:
    """Keep a flushed, measured record at each phase edge."""
    print(
        f"exp7849 {phase} elapsed_s={time.monotonic() - start:.3f} completed_units={done}",
        flush=True,
    )


def science(output_root: Path, date: str, start: float) -> dict[str, Any]:
    """Produce the private candidate without validating a validation child."""
    progress(start, "preconditions_begin", 0)
    result = audit.build_candidate(ROOT, output_root, date)
    progress(start, "preconditions_end", len(result["rows"]))
    checkpoint = (
        output_root
        / "checkpoints"
        / result["reproducibility_checksum"].split(":", 1)[1]
        / "science.json"
    )
    stable = {
        key: result[key]
        for key in ("branch_dispositions", "recomputed_metrics", "gate_check_summary")
    }
    if checkpoint.is_file():
        if json.loads(checkpoint.read_bytes()) != stable:
            raise ValueError("checkpoint_observation_changed")
    else:
        atomic_json(checkpoint, stable)
    atomic_json(output_root / "candidate.json", result)
    progress(start, "candidate_written", len(result["rows"]))
    return result


def _sealed_receipt(receipt: dict[str, Any], output_root: Path, attempt: str) -> dict[str, Any]:
    """Seal a closed child log into a unique content-addressed location."""
    original = ROOT / receipt["log_path"]
    raw = original.read_bytes()
    digest = sha256_file(original)
    sealed = output_root / "validation_logs" / attempt / f"{receipt['name']}_{digest[7:]}.log"
    sealed.parent.mkdir(parents=True, exist_ok=True)
    with sealed.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    original.unlink()
    return {**receipt, "log_path": str(sealed), "log_sha256": digest}


def validate(result: dict[str, Any], output_root: Path, start: float) -> dict[str, Any]:
    """Run only manifest commands, preserving each exit and old failures."""
    (ROOT / audit.BASE / "coverage").mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / audit.MANIFEST).read_bytes())
    required = {
        entry["name"] for entry in manifest["commands"] if entry["classification"] == "required"
    }
    if required != {
        "worktree_imports",
        "affected_pytest",
        "cli_e2e",
        "changed_coverage",
        "ruff_check",
        "ruff_format",
        "mypy",
        "scoped_spec",
        "cold_replay",
        "adversarial_verify",
        "strict_rows",
    }:
        raise ValueError("required_manifest_changed")
    attempt = uuid.uuid4().hex
    receipts = []
    for entry in manifest["commands"]:
        progress(start, f"before_subprocess:{entry['name']}", len(receipts))
        spec = CommandSpec(
            entry["name"], tuple(entry["argv"]), entry["classification"], entry["deadline_s"]
        )
        raw = run_commands(
            ROOT, [spec], log_dir=output_root / "transient_logs" / attempt, heartbeat_s=60.0
        )[0]
        receipt = _sealed_receipt(
            {**raw, "classification": entry["classification"]}, output_root, attempt
        )
        receipts.append(receipt)
        result["validation_receipts"] = receipts
        result["observed_child_commands"] = receipts
        if entry["name"] == "repository_health_180s":
            result["repository_health"].update(
                {"status": "passed" if receipt["passed"] else "failed", "receipt": receipt}
            )
        if entry["name"] == "adversarial_verify":
            try:
                report = json.loads(receipt["output_tail"])
                result["flagged_adversarial"] = bool(report["flagged_count"])
            except (ValueError, KeyError, TypeError):
                result["flagged_adversarial"] = True
        atomic_json(output_root / "candidate.json", result)
        progress(start, f"after_subprocess:{entry['name']}", len(receipts))
    failed = [r for r in receipts if r["classification"] == "required" and not r["passed"]]
    if failed:
        result["honest_verdict"] = "complete_disqualified_required_validation"
        result["verdict_class"] = "disqualified"
        result["independent_evidence_ready_score"] = 0
        result["acceptance_gate_results"]["readiness"] = 0
        result["required_validation_failures"] = [r["name"] for r in failed]
        result["gate_check_summary"].extend(
            {
                "upstream_id": "Exp7849",
                "path": r["log_path"],
                "hash": r["log_sha256"],
                "artifact_field": r["name"],
                "op": "==",
                "expected": 0,
                "observed": r["exit_code"],
            }
            for r in failed
        )
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
    """Offer a private E2E path and a single terminal parent run."""
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
    output_root = args.output_root or ROOT / audit.BASE / "current"
    result = science(output_root, args.date, start)
    if args.science_only:
        return 0
    result = validate(result, output_root, start)
    errors = audit.cold_replay(output_root / "candidate.json", ROOT)
    if errors:
        result["honest_verdict"] = "complete_disqualified_cold_replay"
        result["verdict_class"] = "disqualified"
        result["cold_replay_errors"] = errors
    atomic_json(OUTPUT, result)
    progress(start, f"complete:{result['honest_verdict']}", len(result["validation_receipts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

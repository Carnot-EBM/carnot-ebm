"""Execute the V682 cold audit (REQ-REPORT-7863)."""

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
from carnot.reporting import v682_independent_audit as audit


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/experiment_7863_v682_independent_audit.json"


def progress(start: float, phase: str, done: int) -> None:
    """Expose each phase so a quiet child cannot hide a stalled audit."""
    print(
        f"exp7863 {phase} elapsed_s={time.monotonic() - start:.3f} completed_units={done}",
        flush=True,
    )


def science(output_root: Path, date: str, start: float) -> dict[str, Any]:
    """Write a private candidate and hash-bound checkpoint before validation."""
    progress(start, "preconditions_begin", 0)
    result = audit.build_candidate(ROOT, output_root, date)
    progress(start, "preconditions_end", len(result["producer_status_rows"]))
    checkpoint = (
        output_root
        / "checkpoints"
        / result["reproducibility_checksum"].split(":", 1)[1]
        / "science.json"
    )
    stable = {
        key: result[key]
        for key in ("producer_status_rows", "independently_reduced_rows", "gate_check_summary")
    }
    if checkpoint.is_file():
        if json.loads(checkpoint.read_bytes()) != stable:
            raise ValueError("checkpoint_observation_changed")
    else:
        atomic_json(checkpoint, stable)
    atomic_json(output_root / "candidate.json", result)
    progress(start, "candidate_written", len(result["producer_status_rows"]))
    return result


def _seal(receipt: dict[str, Any], output_root: Path, attempt: str) -> dict[str, Any]:
    """Move a closed child log to an immutable digest-named private path."""
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
    """Run the frozen command list with owned deadlines and actual receipts."""
    manifest = json.loads(Path(audit.MANIFEST).read_bytes())
    attempt = uuid.uuid4().hex
    receipts = []
    for entry in manifest["commands"]:
        progress(start, f"before_subprocess:{entry['name']}", len(receipts))
        spec = CommandSpec(
            entry["name"], tuple(entry["argv"]), entry["classification"], entry["deadline_s"]
        )
        raw = run_commands(
            ROOT, [spec], log_dir=output_root / "transient_logs" / attempt, heartbeat_s=30.0
        )[0]
        receipt = _seal({**raw, "classification": entry["classification"]}, output_root, attempt)
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
    if result["flagged_adversarial"]:
        failed.extend(r for r in receipts if r["name"] == "adversarial_verify" and r not in failed)
    if failed:
        result["honest_verdict"] = "complete_disqualified_required_validation"
        result["verdict_class"] = "disqualified"
        result["acceptance_gate_results"]["readiness"] = 0
        result["required_validation_failures"] = [r["name"] for r in failed]
        result["gate_check_summary"].extend(
            {
                "upstream_id": "Exp7863",
                "path": r["log_path"],
                "hash": r["log_sha256"],
                "artifact_field": r["name"],
                "op": "==",
                "expected": 0,
                "observed": r["exit_code"],
            }
            for r in failed
        )
    else:
        result["audit_execution_ready_score"] = 1
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
    """Expose science-only and replay routes for private end-to-end tests."""
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
    output_root = args.output_root or Path("/tmp/carnot-7863")
    result = science(output_root, args.date, start)
    if args.science_only:
        return 0
    result = validate(result, output_root, start)
    errors = audit.cold_replay(output_root / "candidate.json", ROOT)
    if errors:
        result["honest_verdict"] = "complete_disqualified_cold_replay"
        result["verdict_class"] = "disqualified"
        result["audit_execution_ready_score"] = 0
        result["cold_replay_errors"] = errors
    for key in result:
        result["field_principles"].setdefault(
            key, "Keep the terminal validation result and its exact evidence readable."
        )
    atomic_json(OUTPUT, result)
    progress(start, f"complete:{result['honest_verdict']}", len(result["validation_receipts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

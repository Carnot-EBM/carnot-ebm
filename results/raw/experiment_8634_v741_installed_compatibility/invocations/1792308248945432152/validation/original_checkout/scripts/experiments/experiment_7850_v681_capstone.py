"""Run the bounded direct V681 capstone (REQ-REPORT-7850)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
from typing import Any

from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    sha256_file,
)
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting import v681_capstone as capstone

ROOT = Path(__file__).resolve().parents[2]


def progress(start: float, phase: str, completed: int) -> None:
    """Flush every phase edge with measured monotonic elapsed time."""
    print(
        f"exp7850 {phase} elapsed_s={time.monotonic() - start:.3f} completed_units={completed}",
        flush=True,
    )


def publication(start: float) -> dict[str, Any]:
    """Run the unchanged stable G1-G4 reader once with an owned deadline."""
    progress(start, "before_subprocess:publication_gate", 0)
    completed = subprocess.run(
        [str(ROOT / ".venv/bin/python"), "scripts/publication_gate.py", "--json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    progress(start, "after_subprocess:publication_gate", 1)
    return json.loads(completed.stdout)


def science(output_root: Path, date: str, start: float) -> dict[str, Any]:
    """Checkpoint exact input/configuration bytes and write a private candidate."""
    progress(start, "preconditions_begin", 0)
    capstone.load_manifest(ROOT)
    report = publication(start)
    result = capstone.build_candidate(ROOT, date, report)
    progress(start, "preconditions_end", len(result["rows"]))
    checkpoint = (
        output_root
        / "checkpoints"
        / result["reproducibility_checksum"].split(":", 1)[1]
        / "science.json"
    )
    stable = {key: result[key] for key in ("rows", "gate_check_summary", "source_artifact_hashes")}
    if checkpoint.is_file():
        if json.loads(checkpoint.read_bytes()) != stable:
            raise ValueError("checkpoint_observation_changed")
    else:
        atomic_json(checkpoint, stable)
    atomic_json(output_root / "candidate.json", result)
    progress(start, "candidate_written", len(result["rows"]))
    return result


def seal(receipt: dict[str, Any], output_root: Path, attempt: str) -> dict[str, Any]:
    """Seal a closed child log in a unique content-addressed path."""
    original = ROOT / receipt["log_path"]
    digest = sha256_file(original)
    sealed = output_root / "validation_logs" / attempt / f"{receipt['name']}_{digest[7:]}.log"
    sealed.parent.mkdir(parents=True, exist_ok=True)
    with sealed.open("xb") as stream:
        stream.write(original.read_bytes())
        stream.flush()
        os.fsync(stream.fileno())
    original.unlink()
    return {**receipt, "log_path": str(sealed), "log_sha256": digest}


def validate(result: dict[str, Any], output_root: Path, start: float) -> dict[str, Any]:
    """Execute frozen argv serially and retain every failure and health exit."""
    manifest = capstone.load_manifest(ROOT)
    attempt = uuid.uuid4().hex
    receipts: list[dict[str, Any]] = []
    for entry in manifest["commands"]:
        progress(start, f"before_subprocess:{entry['name']}", len(receipts))
        spec = CommandSpec(
            entry["name"], tuple(entry["argv"]), entry["classification"], entry["deadline_s"]
        )
        raw = run_commands(
            ROOT, [spec], log_dir=output_root / "transient_logs" / attempt, heartbeat_s=55.0
        )[0]
        receipt = seal({**raw, "classification": entry["classification"]}, output_root, attempt)
        receipts.append(receipt)
        result["validation_receipts"] = receipts
        result["observed_child_commands"] = receipts
        if entry["name"] == "adversarial_verify":
            try:
                report = json.loads(receipt["output_tail"])
                result["flagged_adversarial"] = bool(report["flagged_count"])
            except (ValueError, TypeError, KeyError):
                result["flagged_adversarial"] = True
        if entry["name"] == "repository_health_180s":
            result["repository_health"].update(
                status="passed" if receipt["passed"] else "failed", receipt=receipt
            )
        progress(start, f"after_subprocess:{entry['name']}", len(receipts))
    failed = [r for r in receipts if r["classification"] == "required" and not r["passed"]]
    if failed or result["flagged_adversarial"]:
        result["honest_verdict"] = "complete_disqualified_required_validation"
        result["verdict_class"] = "disqualified"
        result["milestone_evidence_ready_score"] = 0
        result["acceptance_gate_results"]["readiness"] = 0
        result["required_validation_failures"] = [r["name"] for r in failed]
        result["gate_check_summary"].extend(
            capstone.failure(
                "Exp7850", r["log_path"], r["log_sha256"], r["name"], "==", 0, r["exit_code"]
            )
            for r in failed
        )
    result["rows"][-1].update(
        honest_verdict=result["honest_verdict"],
        verdict_class=result["verdict_class"],
        raw_metrics={"milestone_evidence_ready_score": result["milestone_evidence_ready_score"]},
    )
    result["task_dispositions"][-1] = dict(result["rows"][-1])
    result["duration_s"] = time.monotonic() - start
    result["phase_spans"].append(
        {
            "phase": "owned_validation",
            "duration_s": result["duration_s"],
            "completed_units": len(receipts),
        }
    )
    result["current_work_receipt"] = build_current_work_receipt(
        run_id=attempt,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_details={"source_count": len(result["source_artifact_hashes"])},
        inference_substrate_class="aggregation",
        execution_venue="host",
        started_monotonic_ns=time.monotonic_ns() - int(result["duration_s"] * 1e9),
        ended_monotonic_ns=time.monotonic_ns(),
    )
    return result


def main(argv: list[str] | None = None) -> int:
    """Run private science, cold replay, or the one terminal parent process."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260929")
    parser.add_argument("--science-only", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    start = time.monotonic()
    progress(start, "start", 0)
    if args.cold_replay:
        errors = capstone.cold_replay(ROOT, args.cold_replay)
        progress(start, f"cold_replay:{errors}", 1)
        return int(bool(errors))
    if args.date != "20260929":
        parser.error("run date must be 20260929")
    output_root = args.output_root or ROOT / capstone.BASE / "current"
    result = science(output_root, args.date, start)
    if args.science_only:
        return 0
    result = validate(result, output_root, start)
    atomic_json(ROOT / capstone.OUTPUT, result)
    progress(start, f"terminal:{result['honest_verdict']}", len(result["validation_receipts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

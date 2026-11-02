"""Run the V680 capstone with frozen validation (REQ-REPORT-7836)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from carnot import experiment_7836_v680_capstone as capstone
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.experiment_7820_v679_hardware_evidence import run_child
from scripts.publication_gate import evaluate


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / capstone.RAW / "candidate.json"


def progress(started: float, phase: str, event: str, units: int) -> None:
    """Print measured elapsed time and completed work at every boundary."""
    print(
        f"[exp7836] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f} completed_units={units}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    """Prepare, dispatch and publish one terminal result with sealed receipts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260928")
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    started = time.monotonic()
    progress(started, "entrypoint", "start", 0)
    if args.cold_replay:
        errors = capstone.cold_replay(args.cold_replay, ROOT)
        print(json.dumps({"errors": errors}), flush=True)
        progress(started, "cold_replay", "complete", 1)
        return int(bool(errors))
    if args.date != "20260928":
        parser.error("run date must be 20260928")
    manifest = capstone.load_manifest(ROOT)
    progress(started, "preconditions", "before_inputs", 0)
    publication = evaluate()
    result = capstone.build_result(ROOT, publication)
    progress(started, "preconditions", "after_inputs", 14)
    target = args.prepare or CANDIDATE
    if args.prepare:
        if target.is_file():
            old = json.loads(target.read_bytes())
            if old.get("source_artifact_hashes") != result["source_artifact_hashes"]:
                raise ValueError("prepared candidate source bytes changed")
        else:
            atomic_json(target, result)
        progress(started, "prepare", "complete", 14)
        return 0
    result["phase_spans"] = [
        dict(
            phase="preconditions",
            start_s=0.0,
            end_s=time.monotonic() - started,
            duration_s=time.monotonic() - started,
            completed_units=14,
            run_date="20260928",
        )
    ]
    atomic_json(CANDIDATE, result)

    def before_child(index: int, receipts: list[dict]) -> None:
        """Save completed independent children before the next bounded call."""
        result["observed_child_commands"] = receipts
        result["validation_receipts"] = {"checks": receipts, "required_checks_passed": None}
        atomic_json(CANDIDATE, result)
        progress(started, "validation", "before_child", index)

    progress(started, "validation", "start", 0)
    receipts = capstone.dispatch(ROOT, manifest, run_child, before_child)
    progress(started, "validation", "after_children", len(receipts))
    required_failed = [r for r in receipts if r["classification"] == "required" and not r["passed"]]
    result["observed_child_commands"] = receipts
    result["validation_receipts"] = {
        "checks": receipts,
        "required_checks_passed": not required_failed,
    }
    result["repository_health"] = {
        "status": "historical_failures_open",
        "diagnostic": next((r for r in receipts if r["name"] == "repository_health"), None),
    }
    for receipt in required_failed:
        result["gate_check_summary"].append(
            capstone.failed(
                "Exp7836:validation",
                receipt["log_path"],
                receipt["log_sha256"],
                receipt["name"],
                "==",
                0,
                receipt["exit_code"],
            )
        )
    if required_failed:
        result["honest_verdict"] = "complete_disqualified_required_validation"
        result["verdict_class"] = "disqualified"
        result["capstone_complete_score"] = 0
        result["acceptance_gate_results"] = {name: 0 for name in capstone.GATE_NAMES}
    result["rows"][-1]["honest_verdict"] = result["honest_verdict"]
    result["rows"][-1]["verdict_class"] = result["verdict_class"]
    result["rows"][-1]["raw_metrics"] = {
        "capstone_complete_score": result["capstone_complete_score"]
    }
    result["task_dispositions"][-1] = dict(result["rows"][-1])
    result["duration_s"] = time.monotonic() - started
    result["phase_spans"].append(
        dict(
            phase="validation",
            start_s=result["phase_spans"][0]["end_s"],
            end_s=result["duration_s"],
            duration_s=result["duration_s"] - result["phase_spans"][0]["end_s"],
            completed_units=len(receipts),
            run_date="20260928",
        )
    )
    atomic_json(ROOT / capstone.OUTPUT, result)
    progress(started, "terminal", "complete", len(receipts))
    print(
        json.dumps(
            {"honest_verdict": result["honest_verdict"], "verdict_class": result["verdict_class"]}
        ),
        flush=True,
    )
    return int(bool(required_failed))


if __name__ == "__main__":
    raise SystemExit(main())

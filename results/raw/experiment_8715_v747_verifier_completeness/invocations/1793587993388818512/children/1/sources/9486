"""Run the V680 independent evidence audit (REQ-REPORT-7835)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from carnot import experiment_7835_v680_independent_evidence_audit as audit
from carnot.reporting.current_work_receipt import atomic_json

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/experiment_7835_v680_independent_evidence_audit.json"
CANDIDATE = ROOT / audit.BASE / "candidate.json"


def main() -> int:
    """Bind raw science and run only the prospectively frozen checks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260928")
    parser.add_argument("--check-only", type=Path)
    parser.add_argument("--dispatch")
    args = parser.parse_args()
    started = time.monotonic()
    print("exp7835 start elapsed_s=0 completed_units=0", flush=True)
    if args.check_only:
        errors = audit.cold_replay(args.check_only)
        print(
            f"exp7835 cold_replay elapsed_s={time.monotonic() - started:.3f} "
            f"completed_units=1 errors={errors}",
            flush=True,
        )
        return int(bool(errors))
    manifest = audit.load_manifest(ROOT)
    if args.dispatch:
        receipt = audit.run_declared(
            ROOT, manifest, args.dispatch, ROOT / audit.BASE / "validation_logs"
        )
        return int(receipt["exit_code"] != 0 or receipt["timed_out"])
    if args.date != "20260928":
        parser.error("run date must be 20260928")
    sources, failures = audit.inspect_sources(ROOT)
    print(
        f"exp7835 preconditions elapsed_s={time.monotonic() - started:.3f} "
        f"completed_units={len(sources)}",
        flush=True,
    )
    isolation = audit.audit_isolation(ROOT)
    result = audit.build_artifact(ROOT, args.date, sources, failures, isolation)
    result["phase_spans"] = [
        {
            "phase": "preconditions_and_raw_custody",
            "duration_s": time.monotonic() - started,
            "completed_units": len(isolation["rows"]),
        }
    ]
    CANDIDATE.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(CANDIDATE, result)
    print(
        f"exp7835 candidate elapsed_s={time.monotonic() - started:.3f} "
        f"completed_units={len(isolation['rows'])}",
        flush=True,
    )
    receipts = []
    for spec in manifest["commands"]:
        receipt = audit.run_declared(
            ROOT, manifest, spec["name"], ROOT / audit.BASE / "validation_logs"
        )
        receipts.append(receipt)
        result["observed_child_commands"] = receipts
        result["validation_receipts"] = receipts
        atomic_json(CANDIDATE, result)
        print(
            f"exp7835 validation elapsed_s={time.monotonic() - started:.3f} "
            f"completed_units={len(receipts)}/{len(manifest['commands'])}",
            flush=True,
        )
    failed = [
        row
        for row in receipts
        if row["classification"] == "required" and (row["exit_code"] != 0 or row["timed_out"])
    ]
    if failed:
        result["honest_verdict"] = "complete_disqualified_required_validation"
        result["verdict_class"] = "disqualified"
        result["independent_evidence_ready_score"] = 0
        result["acceptance_gate_results"]["readiness"] = 0
        result["gate_check_summary"].extend(
            {
                "upstream_id": "Exp7835",
                "path": row["log_path"],
                "hash": row["log_sha256"],
                "field": row["name"],
                "op": "==",
                "expected": 0,
                "observed": row["exit_code"],
            }
            for row in failed
        )
    result["duration_s"] = time.monotonic() - started
    result["phase_spans"].append(
        {
            "phase": "validation",
            "duration_s": result["duration_s"] - result["phase_spans"][0]["duration_s"],
            "completed_units": len(receipts),
        }
    )
    atomic_json(OUTPUT, result)
    print(
        f"exp7835 complete elapsed_s={result['duration_s']:.3f} "
        f"completed_units={len(receipts)} verdict={result['honest_verdict']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

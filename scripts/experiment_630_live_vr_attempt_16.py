#!/usr/bin/env python3
"""Experiment 630: Live VR Attempt #16 — Gated by Exp 629 InterWhen Diagnostic.

**Researcher summary:**
    Exp 629 computed gate_open=False (interwhen recall=0.12, below 0.20 threshold).
    This script immediately writes a blocked artifact and exits.

    RETRO-033 status: 15 consecutive VR attempts at 0% improvement.  Gate is closed.
    Do NOT run VR attempt #16 without gate_open=True from a future diagnostic.
"""

import json
import pathlib
import sys
import time

# Output path matches ExperimentTemplate convention so conductor can locate it.
OUTPUT_PATH = pathlib.Path("results/experiment_630_live_vr_attempt_16.json")


def main() -> None:
    """Write blocked artifact immediately — gate_open=False from Exp 629."""
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t0 = time.monotonic()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "experiment": 630,
        "title": "Live VR Attempt #16 (InterWhenMonitor extractor) — BLOCKED",
        "run_date": "20260421",
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_s": round(time.monotonic() - t0, 3),
        "status": "blocked",
        "schema": "carnot.live_vr_16.v1",
        "gate_open": False,
        "block_reason": (
            "Exp 629 gate_open=False: interwhen recall below 0.20 threshold. "
            "Do NOT schedule VR attempt #16 without recall >= 0.20."
        ),
        "interwhen_recall_primary": 0.12,
        "recall_threshold_required": 0.20,
        "retro_033_resolved": False,
        "signed_improvement": 0.0,
        "n_questions": 0,
        "n_violations_found": 0,
        "n_fixed": 0,
        "n_broken": 0,
        "inference_mode": "blocked_not_run",
        "extractor_used": "interwhen_symcode",
        "honest_verdict": "blocked_gate_closed_do_not_retry",
    }
    # Sort keys for deterministic diffing across runs.
    artifact["schema_fields"] = sorted(artifact.keys())

    OUTPUT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True))
    print(f"[Exp 630] BLOCKED — artifact written to {OUTPUT_PATH}", file=sys.stderr)
    print(
        "[Exp 630] gate_open=False (recall=0.12 < 0.20). "
        "Do not retry VR until recall >= 0.20.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

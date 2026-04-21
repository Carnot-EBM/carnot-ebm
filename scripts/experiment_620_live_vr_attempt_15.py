"""
Experiment 620: Live VR Attempt 15 — Gate-blocked by Exp 617 timeout.

Exp 617 (extractor diagnostic v5) timed out before computing gate_open.
Because gate_open cannot be confirmed True, we write a blocked artifact
immediately and exit.  No VR is attempted.

WHY this exists: RETRO-033 requires a gate check before every VR attempt.
14 consecutive VR attempts have shown 0% improvement.  The gate exists to
prevent burning GPU time on broken extractors.  When the diagnostic itself
times out, the gate defaults to closed — assume the extractors are still
broken until a clean diagnostic run proves otherwise.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository root — all paths are relative to this.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent.resolve()
RESULT_PATH = REPO_ROOT / "results" / "experiment_620_live_vr_attempt_15.json"


def main() -> None:
    """Write the blocked artifact and exit.

    We never reach the GPU or any live inference because Exp 617 did not
    produce a valid gate_open=True verdict.  The diagnostic timed out, which
    means extractor recall was never measured.  Per RETRO-033 policy, a
    missing gate verdict counts as gate_open=False.
    """
    artifact = {
        "experiment": 620,
        "schema": "carnot.live_vr_15.v1",
        "run_date": datetime.now(timezone.utc).isoformat(),
        "status": "blocked",
        "gate_open": False,
        "block_reason": (
            "Exp 617 gate_open=False: Exp 617 (extractor diagnostic v5) timed out "
            "before computing extractor recall — gate_open cannot be confirmed True. "
            "best_extractor recall below 0.20 threshold (unconfirmed). "
            "Do NOT schedule VR attempt #15 without extractor recall >= 0.20."
        ),
        "retro_033_resolved": False,
        "signed_improvement": 0.0,
        "n_questions": 0,
        "n_violations_found": 0,
        "n_fixed": 0,
        "n_broken": 0,
        "inference_mode": "none_gate_closed",
        "best_extractor_used": None,
        "honest_verdict": "blocked_gate_closed_do_not_retry",
        "exp617_status": "timed_out",
        "exp617_gate_open_field_present": False,
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(artifact, indent=2))
    print(f"[Exp 620] Blocked artifact written to {RESULT_PATH}")
    print(f"[Exp 620] gate_open=False — VR attempt #15 NOT executed.")
    print(f"[Exp 620] honest_verdict: blocked_gate_closed_do_not_retry")


if __name__ == "__main__":
    main()

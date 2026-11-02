#!/usr/bin/env python3
"""Experiment 3443: GateMate opportunistic detect + toolchain continuity audit.

Spec refs: REQ-HW-106, SCENARIO-HW-106.

This is a LIGHT continuity audit per north-star §3 (GateMate is
opportunistic-only). The script checks toolchain presence and board
enumeration only — no synth/pnr/flash cycle is run.

Run:
    cd /home/ianblenke/github.com/ianblenke/carnot
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_3443_gatemate_opportunistic_detect_continuity_v1.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from carnot.hardware.gatemate_detect_3443 import run_audit

OUTPUT_PATH = "results/experiment_3443_gatemate_opportunistic_detect_continuity_v1.json"


def main() -> None:
    artifact = run_audit()

    # Wrap into the standard experiment envelope so the conductor recognises it.
    envelope = {
        "experiment": 3443,
        "title": "GateMate Opportunistic Detect + Toolchain Continuity Audit v1",
        **artifact,
    }

    os.makedirs("results", exist_ok=True)
    with open(OUTPUT_PATH, "w") as fh:
        json.dump(envelope, fh, indent=2)

    print(f"Artifact written: {OUTPUT_PATH}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"gatemate_board_detected: {artifact['gatemate_board_detected']}")
    print(f"duration_s: {artifact['duration_s']:.3f}")


if __name__ == "__main__":
    main()

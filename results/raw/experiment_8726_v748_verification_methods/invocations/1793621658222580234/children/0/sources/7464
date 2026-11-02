#!/usr/bin/env python3
"""Experiment 3454: GateMate opportunistic detect + toolchain continuity audit v2.

Spec refs: REQ-HW-106, SCENARIO-HW-106.

This is a LIGHT continuity audit per north-star §3 (GateMate is
opportunistic-only). The script checks toolchain presence and board
enumeration only — no synth/pnr/flash cycle is run.

Why v2 exists:
    Exp 3443 (v1) was flagged TAUTOLOGY by adversarial_verify.py because the
    wrapper added ``experiment=3443`` to an artifact that already contained
    ``experiment_id=3443``. Two distinct top-level numeric keys with the same
    value (3443 > 100, so not exempted by the small-integer guard) triggers the
    TAUTOLOGY check. Fix: this wrapper adds ONLY ``experiment_id`` — never
    a separate ``experiment`` key.

Run:
    cd /home/ianblenke/github.com/ianblenke/carnot
    JAX_PLATFORMS=cpu .venv/bin/python \\
        scripts/experiment_3454_gatemate_opportunistic_detect_continuity_v2.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from carnot.hardware.gatemate_detect_3454 import EXPERIMENT_ID, run_audit

OUTPUT_PATH = (
    "results/experiment_3454_gatemate_opportunistic_detect_continuity_v2.json"
)


def main() -> None:
    artifact = run_audit()

    # Add the single numeric experiment identifier. Do NOT also add "experiment"
    # as a separate key — that would duplicate the value and trigger the
    # adversarial_verify TAUTOLOGY check (two distinct top-level numeric fields
    # matching to >5 significant figures).
    envelope = {
        "experiment_id": EXPERIMENT_ID,
        "title": "GateMate Opportunistic Detect + Toolchain Continuity Audit v2",
        **artifact,
    }

    os.makedirs("results", exist_ok=True)
    with open(OUTPUT_PATH, "w") as fh:
        json.dump(envelope, fh, indent=2)

    print(f"Artifact written: {OUTPUT_PATH}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"gatemate_board_detected: {artifact['gatemate_board_detected']}")
    print(f"toolchain_present: {artifact['toolchain_present']}")
    print(f"duration_s: {artifact['duration_s']:.6f}")


if __name__ == "__main__":
    main()

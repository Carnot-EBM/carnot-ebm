#!/usr/bin/env python3
"""Experiment 3478: GateMate opportunistic detect + toolchain continuity audit v4.

Spec refs: REQ-HW-106, SCENARIO-HW-106.

This is a LIGHT continuity audit per north-star §3 (GateMate is
opportunistic-only). The script checks toolchain presence and board
enumeration only — no synth/pnr/flash cycle is run.

Why v4 exists:
    Exp 3466 (v3) emitted ``complete: blocked_gatemate_toolchain_missing``
    because nextpnr-himbaechel and openFPGALoader were absent. This v4 audit
    re-checks the current state of the toolchain to produce a fresh continuity
    record for the current milestone. The logic is unchanged from v3; only the
    experiment ID and continuity note are updated.

    TAUTOLOGY prevention (inherited from v2 fix over v1/3443):
        This wrapper adds ONLY ``experiment_id`` — never a separate ``experiment``
        key. If both were present as distinct top-level numeric fields with the same
        value (3478), adversarial_verify.py would flag TAUTOLOGY (two conceptually-
        distinct metrics agreeing to >5 significant figures).

Run:
    cd /home/ianblenke/github.com/ianblenke/carnot
    JAX_PLATFORMS=cpu .venv/bin/python \\
        scripts/experiment_3478_gatemate_opportunistic_detect_continuity_v4.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from carnot.hardware.gatemate_detect_3478 import EXPERIMENT_ID, run_audit

OUTPUT_PATH = (
    "results/experiment_3478_gatemate_opportunistic_detect_continuity_v4.json"
)


def main() -> None:
    artifact = run_audit()

    # Add the single numeric experiment identifier. Do NOT also add "experiment"
    # as a separate key — that would duplicate the value and trigger the
    # adversarial_verify TAUTOLOGY check (two distinct top-level numeric fields
    # matching to >5 significant figures).
    envelope = {
        "experiment_id": EXPERIMENT_ID,
        "title": "GateMate Opportunistic Detect + Toolchain Continuity Audit v4",
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

#!/usr/bin/env python3
"""Exp 3467 PolarFire reachability audit v5 — runs audit and writes result JSON.

Spec refs: REQ-HW-070, SCENARIO-HW-070.

Usage:
    cd /path/to/carnot && JAX_PLATFORMS=cpu .venv/bin/python \
        scripts/experiment_3467_polarfire_reachability_audit_v5.py
"""

from __future__ import annotations

import json
import pathlib

from carnot.hardware.polarfire_reachability_audit_3467 import run_audit

OUTPUT_PATH = pathlib.Path("results/experiment_3467_polarfire_reachability_audit_v5.json")


def main() -> None:
    """Run the audit and persist the artifact."""
    artifact = run_audit()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(artifact, indent=2))
    print(f"Artifact written to {OUTPUT_PATH}")
    print(f"  polarfire_reachable : {artifact['polarfire_reachable']}")
    print(f"  honest_verdict      : {artifact['honest_verdict']}")
    print(f"  duration_s          : {artifact['duration_s']:.3f}")


if __name__ == "__main__":
    main()

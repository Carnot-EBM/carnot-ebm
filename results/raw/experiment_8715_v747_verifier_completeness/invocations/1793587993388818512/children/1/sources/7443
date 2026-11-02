#!/usr/bin/env python3
"""Exp 3455 PolarFire reachability audit v4 — runs audit and writes result JSON.

Spec refs: REQ-HW-070, SCENARIO-HW-070.

Usage:
    cd /path/to/carnot && JAX_PLATFORMS=cpu .venv/bin/python \
        scripts/experiment_3455_polarfire_reachability_audit_v4.py
"""

from __future__ import annotations

import json
import pathlib

from carnot.hardware.polarfire_reachability_audit_3455 import run_audit

OUTPUT_PATH = pathlib.Path("results/experiment_3455_polarfire_reachability_audit_v4.json")


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

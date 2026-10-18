#!/usr/bin/env python3
"""Experiment 1154: latent-to-validity snap sweep for Phase 4 Option A.

Run:
    JAX_PLATFORMS=cpu python scripts/experiment_1154_snap_validity_sweep.py

Outputs:
    results/experiment_1154_snap_validity_sweep.json

Spec: REQ-KONA-008, SCENARIO-KONA-007
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "python"))
sys.path.insert(0, str(_REPO_ROOT))

from carnot.phase3.snap_validity import (  # noqa: E402
    SnapSweepConfig,
    run_snap_validity_sweep,
)

RESULT_PATH = _REPO_ROOT / "results" / "experiment_1154_snap_validity_sweep.json"
DEFAULT_CONFIG = SnapSweepConfig()


def main() -> dict[str, object]:
    """Run the snap-validity sweep and write the required JSON artifact."""
    artifact = run_snap_validity_sweep(config=DEFAULT_CONFIG)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

    print("=== Exp 1154: Phase 4 Option A snap-validity sweep ===")
    print(f"latent_dim={artifact['latent_dim']}")
    print(f"n_states_sampled={artifact['n_states_sampled']}")
    print(f"n_legal_snaps={artifact['n_legal_snaps']}")
    print(f"snap_validity_rate={artifact['snap_validity_rate']:.6f}")
    print(f"phase4_option_a_viable={artifact['phase4_option_a_viable']}")
    print(f"proxy_used={artifact['proxy_used']}")
    print(f"honest_verdict={artifact['honest_verdict']}")
    print(f"written={RESULT_PATH}")
    return artifact


if __name__ == "__main__":  # pragma: no cover
    main()

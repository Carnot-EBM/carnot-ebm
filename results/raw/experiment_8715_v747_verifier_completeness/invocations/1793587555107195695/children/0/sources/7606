#!/usr/bin/env python3
"""Run Exp 3645 headroom hybrid verifier-vs-SC positive control.

Spec: REQ-AR-052, SCENARIO-AR-052-01, SCENARIO-AR-052-02.

Run:
  cd /home/ianblenke/github.com/ianblenke/carnot && \
    python scripts/experiment_3645_headroom_hybrid_verifier_vs_sc_v3.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PY_ROOT = REPO_ROOT / "python"
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from carnot.phase3.headroom_hybrid_verifier_vs_sc_v3 import run_experiment  # noqa: E402


ARTIFACT_PATH = (
    REPO_ROOT / "results" / "experiment_3645_headroom_hybrid_verifier_vs_sc_v3.json"
)


def main() -> int:
    artifact = run_experiment(repo_root=REPO_ROOT, output_path=ARTIFACT_PATH)
    print(artifact["honest_verdict"])
    print(
        "oracle_minus_sc_headroom="
        f"{artifact['oracle_minus_sc_headroom']} "
        f"sc_accuracy={artifact['sc_accuracy']} "
        f"verifier_reranked_accuracy={artifact['verifier_reranked_accuracy']} "
        f"hybrid_accuracy={artifact['hybrid_accuracy']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

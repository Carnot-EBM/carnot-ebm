#!/usr/bin/env python3
"""Run Exp 4100 conditional TRM verifier-RFT.

Spec refs: REQ-LEARN-4100, SCENARIO-LEARN-4100-SMOKE,
SCENARIO-LEARN-4100-RFT.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from carnot.agentic import arc_exp4100_trm_verifier_rft_conditional as exp4100


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=exp4100.REPO_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--trm-weights-dir", type=Path, default=exp4100.DEFAULT_TRM_WEIGHTS_DIR)
    args = parser.parse_args()

    artifact = exp4100.run_experiment(
        repo_root=args.repo_root,
        output_path=args.output,
        trm_weights_dir=args.trm_weights_dir,
    )
    print(json.dumps({field: artifact.get(field) for field in exp4100.REQUIRED_ARTIFACT_FIELDS}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

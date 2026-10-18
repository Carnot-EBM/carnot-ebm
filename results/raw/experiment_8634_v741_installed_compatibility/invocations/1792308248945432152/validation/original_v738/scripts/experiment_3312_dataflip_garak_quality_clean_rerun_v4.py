#!/usr/bin/env python3
"""Run Exp 3312 DataFlip/Garak quality-clean rerun v4.

Spec refs: REQ-REPORT-3312, SCENARIO-REPORT-3312.
"""

from __future__ import annotations

import json

from carnot.reporting.dataflip_garak_quality_clean_rerun_3312 import REPO_ROOT, run_experiment


def main() -> int:  # pragma: no cover
    artifact = run_experiment(root=REPO_ROOT)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

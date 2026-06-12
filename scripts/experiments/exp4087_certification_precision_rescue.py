#!/usr/bin/env python3
"""Run Exp 4087 GAP-5 certification precision-rescue sweep.

Spec refs: REQ-LEARN-4087, SCENARIO-LEARN-4087,
SCENARIO-LEARN-4087-FAIL.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
for path in (REPO_ROOT, PYTHON_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from carnot.agentic.arc_exp4087_certification_precision_rescue import run_experiment  # noqa: E402


def main() -> int:
    artifact = run_experiment(repo_root=REPO_ROOT)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

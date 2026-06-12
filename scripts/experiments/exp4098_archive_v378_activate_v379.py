#!/usr/bin/env python3
"""Run Exp 4098 .378 archive and .379 activation.

Spec refs: REQ-REPORT-4098, SCENARIO-REPORT-4098.

Thin entrypoint: all logic lives in
``carnot.reporting.archive_v378_activate_v379_4098`` so it is unit-testable
without a subprocess. This wrapper resolves the repo root, runs the record-only
milestone transition (which records the .378 close-state -- the LLM-LoRA
verifier-as-reward TRAINING route RETIRED, the precision rescue carried by
demo-perfect ALONE rather than the ensemble, the off-ARC demo-fit transfer, the
tenth-game accuracy advance to 10, the one flagged-and-skipped artifact, and the
per-board hardware state), prints the resulting artifact, and exits 0.
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

from carnot.reporting.archive_v378_activate_v379_4098 import run  # noqa: E402


def main() -> int:
    output_path = run(REPO_ROOT)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

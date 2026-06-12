#!/usr/bin/env python3
"""Run Exp 4106 .379 archive and .380 activation.

Spec refs: REQ-REPORT-4106, SCENARIO-REPORT-4106.

Thin entrypoint: all logic lives in
``carnot.reporting.archive_v379_activate_v380_4106`` so it is unit-testable
without a subprocess. This wrapper resolves the repo root, runs the record-only
milestone transition (which records the .379 close-state -- the Carnot verifier
ANTI-discriminates on TRM ARC grids so RFT-on-ARC-grids is bounded, the native
nano-TRM trainer mechanism is confirmed but the RFT did not run, the eleventh
ARC-AGI-3 game solved, the one flagged-and-skipped artifact, and the per-board
hardware state), prints the resulting artifact, and exits 0.
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

from carnot.reporting.archive_v379_activate_v380_4106 import run  # noqa: E402


def main() -> int:
    output_path = run(REPO_ROOT)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

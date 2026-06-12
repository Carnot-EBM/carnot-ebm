#!/usr/bin/env python3
"""Run Exp 4086 .377 archive and .378 activation.

Spec refs: REQ-REPORT-4086, SCENARIO-REPORT-4086.

Thin entrypoint: all logic lives in
``carnot.reporting.archive_v377_activate_v378_4086`` so it is unit-testable
without a subprocess. This wrapper just resolves the repo root, runs the
record-only milestone transition (which records the .377 close-state -- the
verifier-as-reward PIVOT BLOCKED at the Phase-0 verifier-precision gate
0.6818<0.85, the 4 flagged-and-skipped artifacts, the ninth-game accuracy hold,
and the per-board hardware state), prints the resulting artifact, and exits 0.
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

from carnot.reporting.archive_v377_activate_v378_4086 import run  # noqa: E402


def main() -> int:
    output_path = run(REPO_ROOT)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

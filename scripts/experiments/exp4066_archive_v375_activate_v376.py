#!/usr/bin/env python3
"""Run Exp 4066 .375 archive and .376 activation.

Spec refs: REQ-REPORT-4066, SCENARIO-REPORT-4066.

Thin entrypoint: all logic lives in
``carnot.reporting.archive_v375_activate_v376_4066`` so it is unit-testable
without a subprocess. This wrapper just resolves the repo root, runs the
record-only milestone transition (which records the .375 close-state -- the
MECHANISM failure and the two INTACT candidate checkpoints that .376 resumes),
prints the resulting artifact, and exits 0.
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

from carnot.reporting.archive_v375_activate_v376_4066 import run  # noqa: E402


def main() -> int:
    output_path = run(REPO_ROOT)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

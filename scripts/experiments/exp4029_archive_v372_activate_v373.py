#!/usr/bin/env python3
"""Run Exp 4029 .372 archive and .373 activation.

Spec refs: REQ-REPORT-4029, SCENARIO-REPORT-4029.

Thin entrypoint: all logic lives in
``carnot.reporting.archive_v372_activate_v373_4029`` so it is unit-testable
without a subprocess. This wrapper just resolves the repo root, runs the
record-only milestone transition, prints the resulting artifact, and exits 0.
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

from carnot.reporting.archive_v372_activate_v373_4029 import run  # noqa: E402


def main() -> int:
    output_path = run(REPO_ROOT)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

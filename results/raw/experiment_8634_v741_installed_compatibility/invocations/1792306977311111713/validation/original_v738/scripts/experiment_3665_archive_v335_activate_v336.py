#!/usr/bin/env python3
"""Run Exp 3665 v335 archive and v336 activation."""

from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.reporting.archive_v335_activate_v336_3665 import run  # noqa: E402


def main() -> int:
    output_path = run(REPO_ROOT)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

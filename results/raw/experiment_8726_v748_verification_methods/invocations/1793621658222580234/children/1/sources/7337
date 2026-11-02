#!/usr/bin/env python3
"""Run Exp 3302 headline SOTA repair panel v11."""

from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.verify.headline_sota_repair_panel_v11 import write_artifact  # noqa: E402


def main() -> int:
    output = write_artifact(REPO_ROOT)
    artifact = json.loads(output.read_text(encoding="utf-8"))
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

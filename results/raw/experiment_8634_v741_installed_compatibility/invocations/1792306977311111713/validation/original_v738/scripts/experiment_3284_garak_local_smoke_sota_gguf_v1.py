#!/usr/bin/env python3
"""Run Exp 3284 Garak local smoke on a mandated SOTA GGUF."""

from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.reporting.garak_local_smoke_sota_gguf_3284 import run_experiment  # noqa: E402


def main() -> int:
    artifact = run_experiment(project_root=REPO_ROOT)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

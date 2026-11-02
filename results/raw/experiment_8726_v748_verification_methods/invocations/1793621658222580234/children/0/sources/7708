#!/usr/bin/env python3
"""Run Exp 3719 fresh-corpus headline replication."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.verify.headline_replication_fresh_corpus import write_artifact  # noqa: E402


def main() -> int:
    artifact = write_artifact()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

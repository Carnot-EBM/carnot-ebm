#!/usr/bin/env python3
"""Run Exp 3112 logic-regularized verifier pilot."""

from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.eval.logic_regularized_verifier_pilot_v1 import (  # noqa: E402
    OUTPUT_REL_PATH,
    write_artifact,
)


def main() -> int:
    output_path = write_artifact(REPO_ROOT)
    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["logic_regularized_verifier_pilot_ready"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

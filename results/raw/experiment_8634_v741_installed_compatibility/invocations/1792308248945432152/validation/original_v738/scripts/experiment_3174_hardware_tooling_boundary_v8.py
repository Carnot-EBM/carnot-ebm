#!/usr/bin/env python3
"""Write the Exp 3174 hardware/tooling boundary artifact."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.reporting.hardware_tooling_boundary_3174 import build_artifact, write_artifact  # noqa: E402


def main() -> int:
    path = write_artifact(ROOT)
    artifact = build_artifact(ROOT)
    print(path.as_posix())
    return 0 if artifact["hardware_tooling_boundary_v8_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

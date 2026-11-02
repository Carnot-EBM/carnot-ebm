#!/usr/bin/env python3
"""Write the Exp 3132 hardware evidence and sampler boundary artifact."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.reporting.hardware_evidence_sampler_boundary_3132 import (  # noqa: E402
    build_artifact,
    write_artifact,
)


def main() -> int:
    path = write_artifact(ROOT)
    artifact = build_artifact(ROOT)
    print(path.as_posix())
    return 0 if artifact["hardware_evidence_sampler_boundary_v5_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

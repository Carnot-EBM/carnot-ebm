#!/usr/bin/env python3
"""Write the Exp 3202 sparse Potts/PAOA/THRML factor boundary artifact."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.reporting.sparse_potts_paoa_thrml_factor_boundary_3202 import (  # noqa: E402
    build_artifact,
    write_artifact,
)


def main() -> int:
    path = write_artifact(REPO_ROOT)
    artifact = build_artifact(REPO_ROOT)
    print(path.as_posix())
    return 0 if artifact["factor_record_count"] > 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

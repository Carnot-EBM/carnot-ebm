#!/usr/bin/env python3
"""CLI wrapper for Exp 5124 clean local SOTA runtime provenance."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.experiment_5124_clean_sota_runtime_provenance import main  # noqa: E402


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

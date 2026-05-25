#!/usr/bin/env python3
"""Run Exp 3029 repair promotion boundary audit."""

from __future__ import annotations

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carnot.reporting.repair_promotion_boundary_audit_3029 import main  # noqa: E402


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())

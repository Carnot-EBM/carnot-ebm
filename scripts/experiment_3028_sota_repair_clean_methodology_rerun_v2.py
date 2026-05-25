#!/usr/bin/env python3
"""Run Exp 3028 clean-methodology SOTA repair evidence builder."""

from __future__ import annotations

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carnot.eval.sota_repair_clean_methodology_rerun_3028 import main  # noqa: E402


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())

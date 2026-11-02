#!/usr/bin/env python3
"""CLI wrapper for Exp 5139 abstention and verification trace evaluation."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct execution guard
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "python") not in sys.path:  # pragma: no cover - direct execution guard
    sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.experiment_5139_abstention_verification_trace_v471 import main  # noqa: E402


if __name__ == "__main__":  # pragma: no cover - direct execution guard
    raise SystemExit(main())

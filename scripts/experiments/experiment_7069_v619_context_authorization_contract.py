#!/usr/bin/env python3
"""Run the V619 context-authorization contract (REQ-SELFLEARN-7069)."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.experiment_7069_v619_context_authorization_contract import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

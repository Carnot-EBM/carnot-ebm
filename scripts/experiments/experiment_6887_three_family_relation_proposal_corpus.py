#!/usr/bin/env python3
"""Run the authentic Exp6887 five-arm relation proposal corpus."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_6887_three_family_relation_proposal_corpus import main


if __name__ == "__main__":  # pragma: no cover - required command exercises this wrapper.
    raise SystemExit(main())

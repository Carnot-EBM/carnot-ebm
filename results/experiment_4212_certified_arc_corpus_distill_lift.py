"""Entrypoint for Exp 4212 certified ARC corpus distill-lift read.

Spec refs: REQ-VERIFY-4212, SCENARIO-VERIFY-4212.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_4212_certified_arc_corpus_distill_lift import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

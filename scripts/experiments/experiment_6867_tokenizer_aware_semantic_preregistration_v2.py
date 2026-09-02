#!/usr/bin/env python3
"""Run Exp6867 tokenizer-aware semantic preregistration v2.

Spec ref: REQ-INFERENCE-6867.
"""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_6867_tokenizer_aware_semantic_preregistration_v2 import main


if __name__ == "__main__":
    raise SystemExit(main())

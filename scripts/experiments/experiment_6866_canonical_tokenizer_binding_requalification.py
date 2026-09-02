#!/usr/bin/env python3
"""Run Exp6866 canonical tokenizer binding requalification.

Spec ref: REQ-INFERENCE-6866.
"""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_6866_canonical_tokenizer_binding_requalification import main


if __name__ == "__main__":
    raise SystemExit(main())

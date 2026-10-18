#!/usr/bin/env python3
"""Run the lease-aware sequential three-family GGUF runtime.

Spec ref: REQ-INFRA-6973.
"""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_6973_lease_aware_gguf_runtime import main


if __name__ == "__main__":  # pragma: no cover - required command surface.
    raise SystemExit(main())

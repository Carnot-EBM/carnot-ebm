#!/usr/bin/env python3
"""Run the Exp7085 bounded local GGUF chat transport canary."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_7085_v621_chat_transport_canary import main


if __name__ == "__main__":
    raise SystemExit(main())

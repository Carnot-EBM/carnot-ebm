#!/usr/bin/env python3
"""Run Exp 3207 llama.cpp CUDA rebuild clean subprocess gate v1."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.reporting.llama_cpp_cuda_rebuild_clean_subprocess_3207 import main  # noqa: E402


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

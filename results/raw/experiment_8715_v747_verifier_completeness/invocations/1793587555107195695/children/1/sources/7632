#!/usr/bin/env python3
"""Run Exp 3670 facts-row real-benchmark remeasurement."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def _ensure_nli_runtime() -> None:
    """Use the repo venv when bare `python` lacks cached-NLI dependencies."""

    venv_python = REPO_ROOT / ".venv/bin/python"
    if Path(sys.executable).absolute() == venv_python.absolute():
        return
    if Path(sys.prefix).resolve() == (REPO_ROOT / ".venv").resolve():
        return
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ModuleNotFoundError:
        if venv_python.exists():
            os.execv(str(venv_python), [str(venv_python), *sys.argv])


_ensure_nli_runtime()

PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.verify.facts_row_real_benchmark_3670 import (  # noqa: E402
    OUTPUT_REL_PATH,
    build_artifact,
)


def main() -> int:
    artifact = build_artifact(REPO_ROOT)
    output_path = REPO_ROOT / OUTPUT_REL_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

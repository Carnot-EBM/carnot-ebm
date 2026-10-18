#!/usr/bin/env python3
"""Pre-commit guard: refuse commits if .venv torch was installed without CUDA.

Catches the 2026-05-21 regression where `torch 2.12.0+cpu` silently replaced
the CUDA build, causing every GPU-bound experiment task to emit
`blocked_cuda_unavailable` even though both RTX 3090s were healthy via
nvidia-smi.

The check is `torch.version.cuda is not None` — that tells us the *build*
includes CUDA, regardless of whether a GPU is currently accessible (so this
works in CI environments without GPU too).

Pairs with the `[[tool.uv.index]]` + `[tool.uv.sources]` pins in
pyproject.toml that make pytorch-cu128 the structural default for any
`uv pip install` resolution.

Exit codes:
    0 — torch.version.cuda is set (e.g. "12.8") → commit allowed
    1 — torch is CPU-only or import failed → commit refused
    2 — script error itself
"""

from __future__ import annotations

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python3"


def _probe_torch() -> tuple[bool, str]:
    """Return (has_cuda_build, detail_str). Detail printed on failure."""
    if not VENV_PYTHON.exists():
        return False, f"venv python missing at {VENV_PYTHON}; cannot check torch build"
    code = (
        "import sys\n"
        "try:\n"
        "    import torch\n"
        "except Exception as e:\n"
        "    print(f'torch_import_failed:{e}')\n"
        "    sys.exit(0)\n"
        "v = torch.__version__\n"
        "cuda = torch.version.cuda\n"
        "if cuda is None:\n"
        "    print(f'cpu_only_build:{v}')\n"
        "else:\n"
        "    print(f'cuda_build:{v}:cuda{cuda}')\n"
    )
    try:
        # 90s, raised from 15s on 2026-08-22: under heavy box load (a GPU
        # A/B plus parallel agents) the torch IMPORT alone exceeded 15s and
        # this guard blocked every commit with a false "not a CUDA build".
        # The check detects build FLAVOR, not speed — a longer timeout
        # loses no detection, only load tolerance.
        result = subprocess.run(
            [str(VENV_PYTHON), "-c", code],
            capture_output=True,
            text=True,
            timeout=90,
            cwd=str(PROJECT_ROOT),
        )
    except subprocess.TimeoutExpired:
        return False, "torch probe timed out (90s)"
    except Exception as exc:
        return False, f"torch probe error: {exc}"

    out = (result.stdout or "").strip()
    if out.startswith("cuda_build:"):
        return True, out
    return False, out or (result.stderr or "").strip() or "no output from torch probe"


def main() -> int:
    has_cuda, detail = _probe_torch()
    if has_cuda:
        print(f"check_torch_cuda: OK ({detail})")
        return 0

    print(
        "check_torch_cuda: FAIL — .venv torch is not a CUDA build.\n"
        f"  detail: {detail}\n\n"
        "To fix:\n"
        "  .venv/bin/pip install --upgrade \\\n"
        "    --index-url https://download.pytorch.org/whl/cu128 \\\n"
        "    torch torchvision torchaudio\n\n"
        "Or, with uv (preferred since pyproject.toml pins the index):\n"
        "  uv pip install --reinstall --upgrade torch torchvision torchaudio\n\n"
        "Background: the 2026-05-21 incident found torch 2.12.0+cpu silently\n"
        "installed in the venv. Every GPU-bound experiment task emitted\n"
        "blocked_cuda_unavailable until torch was reinstalled with the CUDA\n"
        "wheel. This pre-commit guard exists to catch the regression.\n"
        "See ops/known-issues.md 2026-05-21 entry for the full incident."
    )
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"check_torch_cuda: script error: {exc}", file=sys.stderr)
        sys.exit(2)

"""REQ-VICTRIS-9258-V788: offline Victris Arena Gibbs tactical state-energy scorer and bounded RSI archive."""

from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ["PYTHONUNBUFFERED"] = "1"
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.victris.rsi_loop import main  # noqa: E402

MODEL_SPECS: list[dict[str, object]] = []

if __name__ == "__main__":
    raise SystemExit(main())

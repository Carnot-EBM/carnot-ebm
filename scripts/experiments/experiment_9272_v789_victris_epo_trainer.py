"""REQ-VICTRIS-9272-V789: offline Victris Arena four-module EPO composite Gibbs trainer and Gymnasium adapter."""

from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ["PYTHONUNBUFFERED"] = "1"
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.victris.epo_trainer import main  # noqa: E402

MODEL_SPECS: list[dict[str, object]] = []

if __name__ == "__main__":
    raise SystemExit(main())

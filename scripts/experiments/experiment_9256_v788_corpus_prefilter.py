#!/usr/bin/env python3
"""Exp9256: Qualify a lossless five-sentinel byte prefilter for the complete corpus walk."""

from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ["PYTHONUNBUFFERED"] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.v788_corpus_prefilter_runner import main  # noqa: E402

MODEL_SPECS: list[dict[str, object]] = []


if __name__ == "__main__":
    raise SystemExit(main())

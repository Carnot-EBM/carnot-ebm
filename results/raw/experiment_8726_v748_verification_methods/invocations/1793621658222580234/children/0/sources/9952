#!/usr/bin/env python3
"""REQ-VERIFY-8522: keep the bounded entrypoint independent of ambient imports."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.arc_effect_runner_8522 import main  # noqa: E402

MODEL_SPECS: list[dict[str, str]] = []

if __name__ == "__main__":
    raise SystemExit(main())

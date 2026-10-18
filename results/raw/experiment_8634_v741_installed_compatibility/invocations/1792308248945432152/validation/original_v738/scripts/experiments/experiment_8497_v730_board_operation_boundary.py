#!/usr/bin/env python3
"""REQ-VERIFY-8497: thin model-free runner with the shipped private CLI controls."""

from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting import board_operation_boundary_8497 as e  # noqa: E402
from carnot.reporting import board_operation_runner_8497 as r  # noqa: E402
from scripts.experiments import experiment_8483_v729_board_operation_boundary as old  # noqa: E402

MODEL_SPECS: list[dict[str, str]] = []


def main(argv: list[str] | None = None) -> int:
    """Reuse the date guard and hard deadline without loading models or issuing device commands."""
    with patch.multiple(old, e=e, r=r):
        return int(old.main(argv))


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""REQ-VERIFY-8398: reuse the bounded CLI so no alternative gameplay path appears."""

from pathlib import Path
import sys
from unittest.mock import patch

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from scripts.experiments import experiment_8384_v722_arc_supervisor_live_panel as shipped  # noqa: E402
from carnot.reporting import arc_generalization_panel_8398 as p  # noqa: E402

MODEL_SPECS: list[dict[str, str]] = []
SCRATCH = Path.home() / ".cache/carnot-exp8398-private"


def main(argv: list[str] | None = None) -> int:
    """The inherited CLI enforces private disk scratch and the full 4800-second deadline."""
    with patch.multiple(shipped, p=p, SCRATCH=SCRATCH):
        return int(shipped.main(argv))


if __name__ == "__main__":
    raise SystemExit(main())

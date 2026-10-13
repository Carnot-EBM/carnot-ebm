#!/usr/bin/env python3
"""REQ-VERIFY-8482: the dated CLI reuses bounded production collection and replay."""

from pathlib import Path
import sys
from unittest.mock import patch

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting import arc_panel_qualification_8482 as p  # noqa: E402
from carnot.reporting.arc_panel_validation_8482 import historical  # noqa: E402
from scripts.experiments import experiment_8468_v728_arc_generalization_panel as cli  # noqa: E402

MODEL_SPECS: list[dict[str, str]] = []


def main(argv: list[str] | None = None) -> int:
    """A fresh child gets private disk scratch; original CLI dates and producer files stay intact."""
    args = list(sys.argv[1:] if argv is None else argv)
    p.progress("start_no_model_load_MODEL_SPECS_empty_LLM_calls_zero")
    if args and args[0] == "--historical-shard":
        return historical(Path(args[1]))
    with patch.multiple(cli, p=p, SCRATCH=Path("/var/tmp/carnot-exp8482-private")):
        return int(cli.main(args))


if __name__ == "__main__":
    raise SystemExit(main())

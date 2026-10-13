#!/usr/bin/env python3
"""REQ-VERIFY-8496: thin dated runner keeps original history separate from current authority."""

from pathlib import Path
import sys
from unittest.mock import patch

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting import arc_panel_qualification_8496 as p  # noqa: E402
from carnot.reporting.arc_panel_validation_8482 import historical  # noqa: E402
from carnot.reporting.arc_panel_validation_8496 import rejection  # noqa: E402
from scripts.experiments import experiment_8468_v728_arc_generalization_panel as cli  # noqa: E402

MODEL_SPECS: list[dict[str, str]] = []


def main(argv: list[str] | None = None) -> int:
    """Private original replay is a control; the current invocation uses complete V730 authority."""
    args = list(sys.argv[1:] if argv is None else argv)
    p.progress("start_no_model_load_MODEL_SPECS_empty_LLM_calls_zero")
    if args and args[0] == "--historical-shard":
        return int(historical(Path(args[1])))
    if args and args[0] == "--qualify-rejection":
        rejection(Path(args[1]))
        return 0
    with patch.multiple(cli, p=p, SCRATCH=Path("/var/tmp/carnot-exp8496-private")):
        return int(cli.main(args))


if __name__ == "__main__":
    raise SystemExit(main())

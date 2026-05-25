#!/usr/bin/env python3
"""Run Exp 3071 VERGE-style MCS SMT correction-feedback pilot.

Spec: REQ-VERIFY-3071, SCENARIO-VERIFY-3071.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from carnot.eval.verge_mcs_smt_correction_pilot_v1 import (  # noqa: E402
    ARTIFACT_FILENAME,
    ExperimentConfig,
    run_experiment,
)
from scripts.experiment_template import ExperimentTemplate  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / ARTIFACT_FILENAME)
    parser.add_argument("--pilot-rows", type=Path, default=None)
    parser.add_argument("--test-run", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    template = ExperimentTemplate(
        exp_id=3071,
        title="VERGE-Style MCS SMT Correction Feedback Pilot V1",
        deliverable=str(args.output),
        requires_gpu=True,
        seed=307100,
    )
    template.setup()
    artifact = run_experiment(
        ExperimentConfig(
            repo_root=ROOT,
            output_path=args.output,
            rows_path=args.pilot_rows,
            tests_run=tuple(args.test_run),
        )
    )
    template.assert_deliverable_written()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["mcs_feedback_ready"] else 2


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())

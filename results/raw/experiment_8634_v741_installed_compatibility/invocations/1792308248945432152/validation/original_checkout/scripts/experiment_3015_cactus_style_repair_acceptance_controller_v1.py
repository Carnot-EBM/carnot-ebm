#!/usr/bin/env python3
"""Run Exp 3015 offline Cactus-style repair acceptance controller.

Spec: REQ-CODE-3015, SCENARIO-CODE-3015.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from carnot.eval.repair_acceptance_controller import (  # noqa: E402
    ARTIFACT_FILENAME,
    ExperimentConfig,
    write_artifact,
)
from scripts.experiment_template import ExperimentTemplate  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--test-run", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    template = ExperimentTemplate(
        exp_id=3015,
        title="Cactus-Style Repair Acceptance Controller",
        deliverable=f"results/{ARTIFACT_FILENAME}",
        requires_gpu=False,
        seed=3015,
    )
    template.setup()
    artifact = write_artifact(
        ExperimentConfig(
            output_path=args.output,
            tests_run=tuple(args.test_run),
        )
    )
    template.assert_deliverable_written()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["acceptance_controller_ready"] else 1


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())

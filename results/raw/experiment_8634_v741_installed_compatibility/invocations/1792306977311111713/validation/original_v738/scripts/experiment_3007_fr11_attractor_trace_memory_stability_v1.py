#!/usr/bin/env python3
"""Run Exp 3007 FR-11 trace-memory stability diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from carnot.eval.fr11_attractor_trace_memory_stability_v1 import (  # noqa: E402
    DEFAULT_OUTPUT_PATH,
    ExperimentConfig,
    write_artifact,
)
from scripts.experiment_template import ExperimentTemplate  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args(argv)

    template = ExperimentTemplate(
        exp_id=3007,
        title="FR-11 Attractor-Inspired Trace Memory Stability",
        deliverable=str(Path(args.output).relative_to(ROOT))
        if Path(args.output).is_absolute() and Path(args.output).is_relative_to(ROOT)
        else args.output,
        requires_gpu=False,
        repo_root=ROOT,
        seed=3007,
    )
    template.setup()
    artifact = write_artifact(ExperimentConfig(repo_root=ROOT, output_path=Path(args.output)))
    template.assert_deliverable_written()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["trace_memory_stability_ready"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

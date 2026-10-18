#!/usr/bin/env python3
"""Run Exp 3018 BEAVER-style validator frontier certificate.

Spec: REQ-VERIFY-3018, SCENARIO-VERIFY-3018.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from carnot.eval.beaver_style_validator_frontier_certificate_v1 import (  # noqa: E402
    ARTIFACT_FILENAME,
    ExperimentConfig,
    run_experiment,
)
from scripts.experiment_template import ExperimentTemplate  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / ARTIFACT_FILENAME)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--transcript-dir", type=Path, default=None)
    parser.add_argument("--source-artifact", type=Path, default=None)
    parser.add_argument("--source-manifest", type=Path, default=None)
    parser.add_argument("--test-run", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    template = ExperimentTemplate(
        exp_id=3018,
        title="BEAVER-Style Validator Frontier Certificate",
        deliverable=str(args.output),
        requires_gpu=False,
        seed=3018,
    )
    template.setup()
    artifact = run_experiment(
        ExperimentConfig(
            repo_root=ROOT,
            output_path=args.output,
            certificate_manifest_path=args.manifest,
            transcript_dir=args.transcript_dir,
            source_artifact_path=args.source_artifact,
            source_manifest_path=args.source_manifest,
            tests_run=tuple(args.test_run),
        )
    )
    template.assert_deliverable_written()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["frontier_certificate_ready"] else 1


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())

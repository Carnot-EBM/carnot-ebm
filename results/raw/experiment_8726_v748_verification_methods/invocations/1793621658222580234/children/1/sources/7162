#!/usr/bin/env python3
"""Run Exp 3019 FR-11 feasibility-channel de-tautology diagnostic.

Spec: REQ-VERIFY-3019, SCENARIO-VERIFY-3019.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from carnot.eval.fr11_feasibility_channel_de_tautology_diagnostic_v1 import (  # noqa: E402
    ARTIFACT_FILENAME,
    ExperimentConfig,
    run_experiment,
)
from scripts.experiment_template import ExperimentTemplate  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / ARTIFACT_FILENAME)
    parser.add_argument("--diagnostic-table", type=Path, default=None)
    parser.add_argument("--source-certificate-artifact", type=Path, default=None)
    parser.add_argument("--source-certificate-manifest", type=Path, default=None)
    parser.add_argument("--source-validator-manifest", type=Path, default=None)
    parser.add_argument("--exp3007-artifact", type=Path, default=None)
    parser.add_argument("--test-run", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    template = ExperimentTemplate(
        exp_id=3019,
        title="FR-11 Feasibility-Channel De-Tautology Diagnostic",
        deliverable=str(args.output),
        requires_gpu=False,
        seed=3019,
    )
    template.setup()
    artifact = run_experiment(
        ExperimentConfig(
            repo_root=ROOT,
            output_path=args.output,
            diagnostic_table_path=args.diagnostic_table,
            source_certificate_artifact_path=args.source_certificate_artifact,
            source_certificate_manifest_path=args.source_certificate_manifest,
            source_validator_manifest_path=args.source_validator_manifest,
            exp3007_artifact_path=args.exp3007_artifact,
            tests_run=tuple(args.test_run),
        )
    )
    template.assert_deliverable_written()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["feasibility_channel_diagnostic_ready"] else 1


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())

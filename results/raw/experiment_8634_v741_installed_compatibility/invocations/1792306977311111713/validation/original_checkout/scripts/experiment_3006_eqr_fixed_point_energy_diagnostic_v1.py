#!/usr/bin/env python3
"""Run Exp 3006 fixed-point energy diagnostic.

Spec: REQ-VERIFY-3006, SCENARIO-VERIFY-3006.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from carnot.eval.eqr_fixed_point_energy_diagnostic_v1 import (  # noqa: E402
    DEFAULT_OUTPUT_PATH,
    DEFAULT_TABLE_PATH,
    ExperimentConfig,
    run_diagnostic,
)
from carnot.eval.solver_to_validator_tree_expansion_v1 import VALIDATOR_MANIFEST_REL_PATH  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--table", default=str(DEFAULT_TABLE_PATH))
    parser.add_argument("--manifest", default=str(ROOT / VALIDATOR_MANIFEST_REL_PATH))
    args = parser.parse_args(argv)

    template = ExperimentTemplate(
        exp_id=3006,
        title="EqR Fixed-Point Energy Diagnostic",
        deliverable=args.output,
        requires_gpu=False,
        seed=3006,
    )
    template.setup()
    artifact = run_diagnostic(
        ExperimentConfig(
            repo_root=ROOT,
            output_path=Path(args.output),
            table_path=Path(args.table),
            manifest_path=Path(args.manifest),
        )
    )
    template.assert_deliverable_written()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["fixed_point_diagnostic_ready"] else 1


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())

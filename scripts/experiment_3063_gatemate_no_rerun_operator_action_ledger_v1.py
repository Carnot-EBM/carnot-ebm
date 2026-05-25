#!/usr/bin/env python3
"""Run Exp 3063 GateMate no-rerun operator-action ledger."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from carnot.reporting.gatemate_no_rerun_operator_action_ledger_3063 import (  # noqa: E402
    OUTPUT_REL_PATH,
    build_artifact,
)
from scripts.experiment_template import ExperimentTemplate  # noqa: E402


def main() -> int:
    template = ExperimentTemplate(
        exp_id=3063,
        title="GateMate no-rerun operator-action ledger",
        deliverable=OUTPUT_REL_PATH.as_posix(),
        requires_gpu=False,
        repo_root=ROOT,
        seed=3063,
    )
    template.setup()
    artifact = build_artifact(ROOT)
    output_path = ROOT / OUTPUT_REL_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    template.assert_deliverable_written()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["gatemate_no_rerun_ledger_ready"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

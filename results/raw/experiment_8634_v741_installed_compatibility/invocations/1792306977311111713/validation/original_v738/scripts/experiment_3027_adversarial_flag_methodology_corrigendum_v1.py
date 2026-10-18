#!/usr/bin/env python3
"""Run Exp 3027 adversarial-flag methodology corrigendum."""

from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.reporting.adversarial_flag_methodology_corrigendum_3027 import (  # noqa: E402
    OUTPUT_REL_PATH,
    build_artifact,
)
from scripts.experiment_template import ExperimentTemplate  # noqa: E402


def main() -> int:
    template = ExperimentTemplate(
        exp_id=3027,
        title="Adversarial Flag Methodology Corrigendum",
        deliverable=OUTPUT_REL_PATH.as_posix(),
        requires_gpu=False,
        repo_root=REPO_ROOT,
        seed=3027,
    )
    template.setup()
    artifact = build_artifact(REPO_ROOT)
    output_path = REPO_ROOT / OUTPUT_REL_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    template.assert_deliverable_written()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["methodology_corrigendum_ready"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

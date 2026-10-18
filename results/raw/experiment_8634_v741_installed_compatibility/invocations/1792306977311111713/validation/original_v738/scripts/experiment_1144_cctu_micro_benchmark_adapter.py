#!/usr/bin/env python3
"""Run Exp 1144: CCTU 25-task constrained tool-use adapter.

Spec: REQ-VERIFY-1144, SCENARIO-VERIFY-1144
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from carnot.eval.cctu_micro_benchmark_adapter import run_micro_benchmark  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

DATA_PATH = REPO_ROOT / "data" / "cctu_micro_benchmark_25.json"
OUTPUT_PATH = REPO_ROOT / "results" / "experiment_1144_cctu_micro_benchmark_adapter.json"


def run_experiment(
    *,
    data_path: Path = DATA_PATH,
    output_path: Path = OUTPUT_PATH,
    force_mock: bool = False,
) -> dict[str, object]:
    """Run the adapter and write the required Exp 1144 artifact."""

    template = ExperimentTemplate(
        1144,
        "CCTU Micro-Benchmark",
        str(output_path),
        requires_gpu=not force_mock,
        seed=1144,
    )
    template.setup()
    artifact = run_micro_benchmark(
        data_path=data_path,
        output_path=output_path,
        force_mock=force_mock,
        template=template,
    )
    template.assert_deliverable_written()
    return artifact


def main() -> int:
    """CLI entry point for conductor and manual experiment runs."""

    force_mock = os.getenv("CARNOT_CCTU_FORCE_MOCK", "0") == "1"
    artifact = run_experiment(force_mock=force_mock)
    print(
        "[exp1144] "
        f"mode={artifact['inference_mode']} "
        f"model={artifact['model_used']} "
        f"baseline={artifact['baseline_completion_rate']:.4f} "
        f"guided={artifact['carnot_guided_completion_rate']:.4f} "
        f"delta={artifact['carnot_delta_pp']:.4f} "
        f"verdict={artifact['honest_verdict']} "
        f"output={OUTPUT_PATH.relative_to(REPO_ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

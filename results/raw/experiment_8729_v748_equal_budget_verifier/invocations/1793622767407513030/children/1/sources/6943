#!/usr/bin/env python3
"""Exp 2043: AIA Knuth-Yao Hardware Simulator.

Spec: REQ-SAMPLE-2043, SCENARIO-SAMPLE-2043
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(REPO_ROOT))

from carnot.samplers.knuth_yao import SPEC_REFS, run_statistical_parity_test
from scripts.experiment_template import ExperimentTemplate

DEFAULT_OUTPUT_PATH = Path("results/experiment_2043_aia_knuth_yao.json")
DEFAULT_DISTRIBUTION = [0.125, 0.375, 0.25, 0.25]
DEFAULT_N_SAMPLES = 10_000
DEFAULT_PRECISION_BITS = 3
KNUTH_YAO_SEED = 2043
STANDARD_RNG_SEED = 2044


def run_experiment(output_path: Path | str = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    """Run the Exp 2043 statistical parity check and write the result JSON."""
    output_path = Path(output_path)
    tmpl = ExperimentTemplate(
        exp_id=2043,
        title="AIA Knuth-Yao Hardware Simulator",
        deliverable=str(output_path),
        requires_gpu=False,
        seed=KNUTH_YAO_SEED,
    )
    tmpl.setup()

    parity_metrics = run_statistical_parity_test(
        probabilities=DEFAULT_DISTRIBUTION,
        n_samples=DEFAULT_N_SAMPLES,
        precision_bits=DEFAULT_PRECISION_BITS,
        knuth_yao_seed=KNUTH_YAO_SEED,
        standard_rng_seed=STANDARD_RNG_SEED,
    )
    status = "success" if parity_metrics["parity_passed"] else "partial"
    honest_verdict = (
        "knuth_yao_statistical_parity_passed"
        if parity_metrics["parity_passed"]
        else "knuth_yao_statistical_parity_failed"
    )

    artifact = tmpl.build_result(
        data={
            "spec_refs": SPEC_REFS,
            "sampler": "KnuthYaoSampler",
            "distribution": DEFAULT_DISTRIBUTION,
            "parity_metrics": parity_metrics,
            "hardware_execution_claim": False,
            "aia_simulation_only": True,
        },
        status=status,
        honest_verdict=honest_verdict,
        metrics_used=[
            "categorical_frequency_delta",
            "total_variation_delta",
            "rng_bit_accounting",
        ],
        code_files=[
            __file__,
            "python/carnot/samplers/knuth_yao.py",
            "scripts/experiment_template.py",
        ],
    )

    tmpl._output_path.write_text(json.dumps(artifact, indent=2))
    tmpl.assert_deliverable_written()
    return json.loads(tmpl._output_path.read_text())


def main() -> None:  # pragma: no cover
    run_experiment()


if __name__ == "__main__":  # pragma: no cover
    main()

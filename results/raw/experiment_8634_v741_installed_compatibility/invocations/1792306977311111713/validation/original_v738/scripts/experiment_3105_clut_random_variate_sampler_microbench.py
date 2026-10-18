#!/usr/bin/env python3
"""Exp 3105: CPU cLUT random-variate sampler microbench.

Spec: REQ-SAMPLE-3105, SCENARIO-SAMPLE-3105
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(REPO_ROOT))

from carnot.samplers.clut_random_variate import (
    DEFAULT_N_VARIATES,
    DEFAULT_REPEATS,
    DEFAULT_SEED,
    FPGA_MAPPING_NOTES_PATH,
    IMPLEMENTATION_PATH,
    SPEC_REFS,
    run_clut_microbench,
)
from scripts.experiment_template import ExperimentTemplate

DEFAULT_OUTPUT_PATH = Path("results/experiment_3105_clut_random_variate_sampler_microbench_v1.json")


def run_experiment(
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    *,
    n_variates: int = DEFAULT_N_VARIATES,
    repeats: int = DEFAULT_REPEATS,
) -> dict[str, Any]:
    """Run the CPU-only cLUT microbench and write the terminal artifact."""
    output_path = Path(output_path)
    mapping_notes = REPO_ROOT / FPGA_MAPPING_NOTES_PATH
    if not mapping_notes.exists():
        raise FileNotFoundError(f"missing FPGA mapping notes: {FPGA_MAPPING_NOTES_PATH}")

    tmpl = ExperimentTemplate(
        exp_id=3105,
        title="CPU cLUT Random-Variate Sampler Microbench",
        deliverable=str(output_path),
        requires_gpu=False,
        seed=DEFAULT_SEED,
    )
    tmpl.setup()

    result_fields = run_clut_microbench(
        n_variates=n_variates,
        repeats=repeats,
        seed=DEFAULT_SEED,
    )
    status = "success" if result_fields["clut_microbench_ready"] else "partial"
    verdict_prefix = "complete" if result_fields["clut_microbench_ready"] else "blocked"
    honest_verdict = (
        f"{verdict_prefix}: CPU cLUT microbench ran with "
        f"distribution_error_gate_passed="
        f"{result_fields['distribution_error']['distribution_error_gate_passed']}; "
        "speedup is scoped to CPU scalar exact-logistic baseline; "
        "hardware_claim_made=false; hardware_commands_run=[]"
    )

    artifact = tmpl.build_result(
        data={
            **result_fields,
            "honest_verdict": honest_verdict,
        },
        status=status,
        metrics_used=[
            "clut_table_probability_error",
            "empirical_bernoulli_bin_error",
            "cpu_wall_clock_speedup",
        ],
        code_files=[
            __file__,
            IMPLEMENTATION_PATH,
            FPGA_MAPPING_NOTES_PATH,
            "openspec/capabilities/samplers/spec.md",
        ],
    )

    tmpl._output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True))
    tmpl.assert_deliverable_written()
    return json.loads(tmpl._output_path.read_text())


def main() -> None:  # pragma: no cover
    run_experiment()


if __name__ == "__main__":  # pragma: no cover
    main()

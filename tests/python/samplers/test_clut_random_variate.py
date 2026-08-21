"""Tests for the CPU cLUT logistic Bernoulli sampler.

Spec coverage: REQ-SAMPLE-3105, SCENARIO-SAMPLE-3105
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from carnot.experiment_artifacts import ARTIFACT_ROOT_ENV
from scripts.experiment_template import REQUIRED_RESULT_FIELDS

from carnot.samplers.clut_random_variate import (
    SPEC_REFS,
    ClutLogisticBernoulliSampler,
    distribution_error_report,
    exact_logistic_bernoulli_baseline,
    run_clut_microbench,
    stable_sigmoid,
)


def test_req_sample_3105_builds_fixed_point_table_and_samples_deterministically() -> None:
    """REQ-SAMPLE-3105: cLUT sampler stores compressed fixed-point thresholds."""
    sampler = ClutLogisticBernoulliSampler(
        logit_min=-4.0,
        logit_max=4.0,
        table_size=129,
        threshold_bits=12,
        seed=7,
    )

    assert sampler.threshold_table.dtype == np.uint16
    assert sampler.table_nbytes == 258
    assert sampler.random_modulus == 4096

    probabilities = sampler.approximate_probabilities(np.array([-4.0, 0.0, 4.0]))
    assert probabilities.shape == (3,)
    assert probabilities[0] < 0.03
    assert probabilities[1] == 0.5
    assert probabilities[2] > 0.97

    logits = np.array([-2.0, -0.25, 0.25, 2.0])
    samples_a = sampler.sample_logits(logits, seed=3105)
    samples_b = sampler.sample_logits(logits, seed=3105)
    assert samples_a.dtype == np.bool_
    assert samples_a.shape == logits.shape
    np.testing.assert_array_equal(samples_a, samples_b)


def test_req_sample_3105_exact_baseline_and_distribution_error_helpers() -> None:
    """REQ-SAMPLE-3105: baseline and distribution helpers measure sampler quality."""
    logits = np.repeat(np.array([-2.0, 0.0, 2.0], dtype=np.float64), 4096)
    expected = stable_sigmoid(logits)

    sampler = ClutLogisticBernoulliSampler(table_size=513, threshold_bits=15, seed=3105)
    clut_samples = sampler.sample_logits(logits, seed=3106)
    baseline_samples = exact_logistic_bernoulli_baseline(logits, seed=3106)

    clut_report = distribution_error_report(
        samples=clut_samples,
        expected_probabilities=expected,
        logits=logits,
        n_bins=3,
    )
    baseline_report = distribution_error_report(
        samples=baseline_samples,
        expected_probabilities=expected,
        logits=logits,
        n_bins=3,
    )

    assert clut_report["n_samples"] == 12_288
    assert len(clut_report["bins"]) == 3
    assert clut_report["max_abs_bin_error"] <= 0.04
    assert baseline_report["max_abs_bin_error"] <= 0.04
    assert sampler.table_max_abs_probability_error(logits) <= 0.004


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"logit_min": float("nan")}, "finite"),
        ({"logit_min": 1.0, "logit_max": 1.0}, "logit_min"),
        ({"table_size": 1}, "table_size"),
        ({"threshold_bits": 0}, "threshold_bits"),
        ({"threshold_bits": 17}, "threshold_bits"),
    ],
)
def test_req_sample_3105_rejects_invalid_sampler_configuration(
    kwargs: dict[str, float | int],
    message: str,
) -> None:
    """REQ-SAMPLE-3105: invalid cLUT sampler configurations fail loudly."""
    with pytest.raises(ValueError, match=message):
        ClutLogisticBernoulliSampler(**kwargs)


def test_req_sample_3105_rejects_invalid_inputs() -> None:
    """REQ-SAMPLE-3105: non-finite logits and mismatched reports fail loudly."""
    sampler = ClutLogisticBernoulliSampler()

    with pytest.raises(ValueError, match="finite"):
        sampler.approximate_probabilities(np.array([0.0, np.nan]))

    with pytest.raises(ValueError, match="same shape"):
        distribution_error_report(
            samples=np.array([True, False]),
            expected_probabilities=np.array([0.5]),
            logits=np.array([0.0]),
        )

    with pytest.raises(ValueError, match="non-empty"):
        distribution_error_report(
            samples=np.array([], dtype=bool),
            expected_probabilities=np.array([]),
            logits=np.array([]),
        )


def test_req_sample_3105_microbench_reports_cpu_timing_and_error() -> None:
    """REQ-SAMPLE-3105: microbench compares CPU cLUT timing and quality."""
    report = run_clut_microbench(n_variates=8192, repeats=2, seed=3105)

    assert report["spec_refs"] == SPEC_REFS
    assert report["clut_microbench_ready"] is True
    assert report["hardware_claim_made"] is False
    assert report["hardware_commands_run"] == []
    assert report["distribution_error"]["distribution_error_gate_passed"] is True
    assert report["distribution_error"]["clut_table_max_abs_probability_error"] <= 0.004
    assert report["speedup_vs_baseline"]["speedup_ratio"] > 0.0
    assert report["inference_substrate"]["kind"] == "cpu_numpy_microbench"


def test_scenario_sample_3105_experiment_writes_terminal_artifact(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-3105: Exp 3105 writes the claim-bounded JSON artifact."""
    output_path = tmp_path / "experiment_3105_clut_random_variate_sampler_microbench_v1.json"
    env = os.environ.copy()
    env[ARTIFACT_ROOT_ENV] = str(tmp_path)
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from scripts.experiment_3105_clut_random_variate_sampler_microbench "
                "import run_experiment; "
                f"run_experiment(output_path={str(output_path)!r}, n_variates=8192, repeats=2)"
            ),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[3],
        env=env,
    )

    assert output_path.exists()
    artifact = json.loads(output_path.read_text())
    for field in REQUIRED_RESULT_FIELDS:
        assert field in artifact

    assert artifact["experiment"] == 3105
    assert artifact["status"] == "success"
    assert artifact["spec_refs"] == SPEC_REFS
    assert artifact["clut_microbench_ready"] is True
    assert artifact["implementation_path"] == "python/carnot/samplers/clut_random_variate.py"
    assert artifact["benchmark_commands"]
    assert artifact["distribution_error"]["distribution_error_gate_passed"] is True
    assert artifact["speedup_vs_baseline"]["baseline_name"] == "scalar_exact_logistic"
    assert artifact["fpga_mapping_notes_path"] == (
        "docs/research-notes/experiment_3105_clut_sampler_fpga_mapping.md"
    )
    assert Path(artifact["fpga_mapping_notes_path"]).exists()
    assert artifact["hardware_claim_made"] is False
    assert artifact["hardware_commands_run"] == []
    assert "research-references.md" in artifact["source_artifacts"]
    assert artifact["inference_substrate"]["executes_hardware"] is False
    assert artifact["honest_verdict"].startswith("complete:")

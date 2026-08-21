"""Tests for the AIA Knuth-Yao sampler simulator.

Spec coverage: REQ-SAMPLE-2043, SCENARIO-SAMPLE-2043
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

from carnot.samplers.knuth_yao import KnuthYaoSampler, run_statistical_parity_test


def test_req_sample_2043_builds_ddg_matrix_and_tracks_bits() -> None:
    """REQ-SAMPLE-2043: KnuthYaoSampler builds a DDG matrix and reports bit use."""
    sampler = KnuthYaoSampler([0.5, 0.25, 0.125, 0.125], precision_bits=3, seed=7)

    assert sampler.ddg_matrix.tolist() == [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
    ]
    np.testing.assert_allclose(sampler.quantized_probabilities, [0.5, 0.25, 0.125, 0.125])

    samples = sampler.sample_indices(32)
    assert samples.shape == (32,)
    assert samples.dtype.kind in {"i", "u"}
    assert set(samples.tolist()) <= {0, 1, 2, 3}

    metrics = sampler.bit_metrics()
    assert metrics["samples_drawn"] == 32
    assert metrics["bits_consumed"] == sampler.bits_consumed
    assert 1.0 <= metrics["average_bits_per_sample"] <= 3.0
    assert metrics["fixed_width_bits_per_sample"] == 3
    assert metrics["rng_bit_reduction_vs_fixed_width"] >= 0.0


def test_req_sample_2043_samples_named_symbols_and_resets_metrics() -> None:
    """REQ-SAMPLE-2043: symbol sampling preserves the discrete abstraction."""
    sampler = KnuthYaoSampler(
        [0.5, 0.25, 0.25],
        symbols=("low", "mid", "high"),
        precision_bits=2,
        seed=11,
    )

    symbols = sampler.sample(12)
    assert symbols.shape == (12,)
    assert set(symbols.tolist()) <= {"low", "mid", "high"}
    assert sampler.bit_metrics()["samples_drawn"] == 12

    sampler.reset_metrics()
    assert sampler.bit_metrics()["samples_drawn"] == 0
    assert sampler.bit_metrics()["bits_consumed"] == 0

    empty = sampler.sample_indices(0)
    assert empty.shape == (0,)


@pytest.mark.parametrize(
    "probabilities, message",
    [
        ([], "one-dimensional"),
        ([[0.5, 0.5]], "one-dimensional"),
        ([0.5, float("nan")], "finite"),
        ([0.5, -0.5], "non-negative"),
        ([0.0, 0.0], "positive mass"),
    ],
)
def test_req_sample_2043_rejects_invalid_distributions(
    probabilities: object,
    message: str,
) -> None:
    """REQ-SAMPLE-2043: invalid categorical distributions fail loudly."""
    with pytest.raises(ValueError, match=message):
        KnuthYaoSampler(probabilities, precision_bits=3)


def test_req_sample_2043_rejects_invalid_configuration() -> None:
    """REQ-SAMPLE-2043: sampler configuration is validated before sampling."""
    with pytest.raises(ValueError, match="precision_bits"):
        KnuthYaoSampler([0.5, 0.5], precision_bits=0)
    with pytest.raises(ValueError, match="same length"):
        KnuthYaoSampler([0.5, 0.5], symbols=("only-one",), precision_bits=1)
    with pytest.raises(ValueError, match="non-negative"):
        KnuthYaoSampler([0.5, 0.5], precision_bits=1).sample_indices(-1)


def test_req_sample_2043_10000_sample_parity_against_standard_rng() -> None:
    """REQ-SAMPLE-2043: Knuth-Yao matches standard RNG over 10,000 samples."""
    report = run_statistical_parity_test(
        probabilities=[0.125, 0.375, 0.25, 0.25],
        n_samples=10_000,
        precision_bits=3,
        knuth_yao_seed=2043,
        standard_rng_seed=2044,
    )

    assert report["spec_refs"] == ["REQ-SAMPLE-2043", "SCENARIO-SAMPLE-2043"]
    assert report["n_samples"] == 10_000
    assert report["parity_passed"] is True
    assert report["thresholds"] == {
        "max_abs_frequency_delta": 0.03,
        "total_variation_delta": 0.04,
    }
    assert report["max_abs_frequency_delta"] <= 0.03
    assert report["total_variation_delta"] <= 0.04
    assert report["knuth_yao_counts"] != report["standard_rng_counts"]
    assert report["bit_metrics"]["average_bits_per_sample"] < 3.0


def test_scenario_sample_2043_run_experiment_writes_terminal_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-SAMPLE-2043: Exp 2043 writes the statistical parity artifact."""
    monkeypatch.setenv(ARTIFACT_ROOT_ENV, str(tmp_path))
    output_path = tmp_path / "experiment_2043_aia_knuth_yao.json"
    env = os.environ.copy()
    env[ARTIFACT_ROOT_ENV] = str(tmp_path)
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from scripts.experiment_2043_aia_knuth_yao import run_experiment; "
                f"run_experiment(output_path={str(output_path)!r})"
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
    assert artifact["experiment"] == 2043
    assert artifact["status"] == "success"
    assert artifact["spec_refs"] == ["REQ-SAMPLE-2043", "SCENARIO-SAMPLE-2043"]
    assert artifact["hardware_execution_claim"] is False
    assert artifact["parity_metrics"]["n_samples"] == 10_000
    assert artifact["parity_metrics"]["parity_passed"] is True
    assert artifact["honest_verdict"] == "knuth_yao_statistical_parity_passed"

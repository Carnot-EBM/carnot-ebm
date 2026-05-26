"""Tests for the optional CPU cLUT sampler backend adapter.

Spec coverage: REQ-SAMPLE-3118, SCENARIO-SAMPLE-3118
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from carnot.samplers.backend import CpuBackend, get_backend
from carnot.samplers.clut_backend import (
    CLUT_BACKEND_SPEC_REFS,
    ClutCpuBackend,
    build_clut_backend_integration_report,
    clut_backend_distribution_check,
)
from carnot.samplers.clut_random_variate import distribution_error_report, stable_sigmoid


def test_req_sample_3118_backend_selection_preserves_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-SAMPLE-3118: clut_cpu is explicit and the default remains cpu."""
    monkeypatch.delenv("CARNOT_BACKEND", raising=False)

    default_backend = get_backend()
    clut_backend = get_backend("clut_cpu")

    assert isinstance(default_backend, CpuBackend)
    assert default_backend.backend_name == "cpu"
    assert isinstance(clut_backend, ClutCpuBackend)
    assert clut_backend.backend_name == "clut_cpu"
    assert clut_backend.hardware_claim_made is False
    assert clut_backend.hardware_commands_run == ()
    assert clut_backend.inference_substrate["kind"] == "cpu_numpy_clut_backend"
    assert clut_backend.set_constraints(lambda state: state) is None
    assert clut_backend.dual_update_step(0.1) is None


def test_req_sample_3118_reproducible_seed_behavior() -> None:
    """REQ-SAMPLE-3118-2: identical cLUT backend seeds reproduce samples."""
    biases = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    couplings = np.zeros((3, 3), dtype=np.float64)

    backend = ClutCpuBackend(seed=3118)
    samples_a = backend.sample(
        biases,
        couplings,
        n_samples=32,
        config={"beta": 1.0, "steps_per_sample": 2, "seed": 99},
    )
    samples_b = backend.sample(
        biases,
        couplings,
        n_samples=32,
        config={"beta": 1.0, "steps_per_sample": 2, "seed": 99},
    )
    samples_c = ClutCpuBackend(seed=3118).sample(
        biases,
        couplings,
        n_samples=32,
        config={"beta": 1.0, "steps_per_sample": 2, "seed": 100},
    )

    assert samples_a.shape == (32, 3)
    assert samples_a.dtype == np.bool_
    np.testing.assert_array_equal(samples_a, samples_b)
    assert not np.array_equal(samples_a, samples_c)


def test_req_sample_3118_zero_coupling_distribution_check_passes() -> None:
    """REQ-SAMPLE-3118-4: zero-coupling cLUT backend matches sigmoid bins."""
    biases = np.array([-2.0, 0.0, 2.0], dtype=np.float64)
    couplings = np.zeros((3, 3), dtype=np.float64)
    samples = ClutCpuBackend(seed=3118).sample(
        biases,
        couplings,
        n_samples=65_536,
        config={"beta": 1.0, "steps_per_sample": 1, "seed": 3118},
    )
    logits = np.broadcast_to(biases, samples.shape)
    expected = stable_sigmoid(logits)

    report = distribution_error_report(
        samples=samples,
        expected_probabilities=expected,
        logits=logits,
        n_bins=3,
    )

    assert report["n_samples"] == 196_608
    assert report["max_abs_bin_error"] <= 0.015
    assert report["overall_abs_error"] <= 0.005


@pytest.mark.parametrize(
    "biases, couplings, match",
    [
        (np.zeros((1, 3)), np.zeros((3, 3)), "biases"),
        (np.zeros(3), np.zeros((2, 2)), "couplings"),
        (np.array([0.0, np.nan]), np.zeros((2, 2)), "finite"),
        (np.zeros(2), np.array([[0.0, np.inf], [0.0, 0.0]]), "finite"),
    ],
)
def test_req_sample_3118_rejects_invalid_backend_inputs(
    biases: np.ndarray,
    couplings: np.ndarray,
    match: str,
) -> None:
    """REQ-SAMPLE-3118-1: adapter fails loudly on invalid Ising inputs."""
    with pytest.raises(ValueError, match=match):
        ClutCpuBackend().sample(biases, couplings, n_samples=4, config={})


def test_req_sample_3118_minimize_energy_uses_same_backend_shape() -> None:
    """REQ-SAMPLE-3118-1: minimize_energy preserves the backend sample shape."""
    biases = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float64)
    couplings = np.zeros((4, 4), dtype=np.float64)

    samples = ClutCpuBackend(seed=42).minimize_energy(
        biases,
        couplings,
        n_samples=12,
        n_steps=2,
        beta=1.0,
    )

    assert samples.shape == (12, 4)
    assert samples.dtype == np.bool_
    assert samples.mean() > 0.6


def test_req_sample_3118_report_builder_records_claim_boundary() -> None:
    """REQ-SAMPLE-3118-5: report builder emits auditable CPU-only fields."""
    report = build_clut_backend_integration_report(
        tests_run=[
            {
                "command": ".venv/bin/pytest tests/python/samplers/test_clut_backend.py -q",
                "status": "passed",
            }
        ],
        n_distribution_samples=8192,
        timing_repeats=2,
    )

    assert report["spec_refs"] == CLUT_BACKEND_SPEC_REFS
    assert report["clut_backend_integration_boundary_v2_ready"] is True
    assert "python/carnot/samplers/clut_backend.py" in report["implementation_paths"]
    assert report["default_backend_preserved"] is True
    assert report["distribution_checks_passed"] is True
    assert report["cpu_timing_summary"]["measured"] is True
    assert report["hardware_claim_made"] is False
    assert report["hardware_commands_run"] == []
    assert report["inference_substrate"]["executes_hardware"] is False
    assert report["honest_verdict"].startswith("complete:")


def test_req_sample_3118_distribution_helper_reports_passed_gate() -> None:
    """REQ-SAMPLE-3118-4: helper reports the distribution gate explicitly."""
    report = clut_backend_distribution_check(n_samples=8192, seed=3118)

    assert report["distribution_checks_passed"] is True
    assert report["thresholds"]["max_abs_bin_error"] == 0.02
    assert report["distribution_error"]["max_abs_bin_error"] <= 0.02


def test_scenario_sample_3118_result_artifact_is_claim_bounded() -> None:
    """SCENARIO-SAMPLE-3118: result artifact records no hardware claim."""
    result_path = Path("results/experiment_3118_clut_sampler_backend_integration_boundary_v2.json")
    artifact = json.loads(result_path.read_text())

    for field in (
        "clut_backend_integration_boundary_v2_ready",
        "implementation_paths",
        "default_backend_preserved",
        "distribution_checks_passed",
        "cpu_timing_summary",
        "hardware_claim_made",
        "hardware_commands_run",
        "tests_run",
        "source_artifacts",
        "inference_substrate",
        "honest_verdict",
    ):
        assert field in artifact

    assert artifact["clut_backend_integration_boundary_v2_ready"] is True
    assert artifact["default_backend_preserved"] is True
    assert artifact["distribution_checks_passed"] is True
    assert artifact["hardware_claim_made"] is False
    assert artifact["hardware_commands_run"] == []
    assert artifact["inference_substrate"]["kind"] == "cpu_numpy_clut_backend"
    assert artifact["inference_substrate"]["executes_hardware"] is False
    assert artifact["honest_verdict"].startswith("complete:")
    assert os.environ.get("CARNOT_BACKEND") != "clut_cpu"

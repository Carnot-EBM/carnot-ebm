"""Tests for the vendored THRML block-Gibbs inference adapter.

Spec coverage: REQ-SAMPLE-058, SCENARIO-SAMPLE-086.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from carnot.sampling import _vendored_thrml
from carnot.sampling import gibbs


def _tiny_problem() -> tuple[np.ndarray, np.ndarray]:
    biases = np.array([0.2, -0.1, 0.15, -0.05], dtype=np.float32)
    couplings = np.array(
        [
            [0.0, 0.4, 0.0, -0.2],
            [0.4, 0.0, 0.3, 0.0],
            [0.0, 0.3, 0.0, 0.1],
            [-0.2, 0.0, 0.1, 0.0],
        ],
        dtype=np.float32,
    )
    return biases, couplings


def test_req_sample_058_vendored_thrml_metadata_and_license() -> None:
    """REQ-SAMPLE-058: vendored THRML is pinned to 0.1.3 with Apache-2.0 provenance."""

    license_path = Path(_vendored_thrml.__file__).with_name("LICENSE")

    assert _vendored_thrml.__version__ == "0.1.3"
    assert _vendored_thrml.THRML_LICENSE == "Apache-2.0"
    assert license_path.exists()
    assert "Apache License" in license_path.read_text(encoding="utf-8")


def test_req_sample_058_candidate_payload_is_initial_state_when_k_zero() -> None:
    """REQ-SAMPLE-058: payload candidate is the warm-start state, not random state."""

    biases, couplings = _tiny_problem()
    candidate = "1010"

    response = gibbs.sample_from_payload(
        {"prompt": "check this answer", "candidate": candidate},
        biases,
        couplings,
        seed=7,
        n_samples=1,
        n_warmup=0,
        steps_per_sample=1,
        beta=1.0,
    )

    assert response["initialized_from_candidate"] is True
    assert response["sampler"] == "thrml-0.1.3-block-gibbs"
    assert response["samples"] == [[1, 0, 1, 0]]


def test_req_sample_058_candidate_encodings_preserve_warm_start() -> None:
    """REQ-SAMPLE-058: candidate warm-start accepts API-friendly bool and spin encodings."""

    biases, couplings = _tiny_problem()

    bool_response = gibbs.sample_from_payload(
        {"prompt": "check this answer", "candidate": "true false true false"},
        biases,
        couplings,
        seed=7,
        n_samples=1,
        n_warmup=0,
        steps_per_sample=1,
        beta=1.0,
    )
    spin_samples = gibbs.sample(
        biases,
        couplings,
        candidate=[-1, 1, -1, 1],
        seed=7,
        n_samples=1,
        n_warmup=0,
        steps_per_sample=1,
        beta=1.0,
    )

    assert bool_response["samples"] == [[1, 0, 1, 0]]
    np.testing.assert_array_equal(spin_samples, np.array([[False, True, False, True]]))


def test_scenario_sample_086_adapter_imports_vendored_thrml() -> None:
    """SCENARIO-SAMPLE-086: Carnot adapter dispatches to the vendored THRML module."""

    assert gibbs._thrml.__name__ == "carnot.sampling._vendored_thrml"
    assert gibbs._thrml.models.__name__ == "carnot.sampling._vendored_thrml.models"
    assert "THRML" in (gibbs.sample.__doc__ or "")


def test_scenario_sample_086_constructive_parity_is_exact() -> None:
    """SCENARIO-SAMPLE-086: Carnot-vs-THRML KL is zero by same-code construction."""

    biases, couplings = _tiny_problem()
    candidate = [False, True, False, True]

    carnot_samples = gibbs.sample(
        biases,
        couplings,
        candidate=candidate,
        seed=11,
        n_samples=3,
        n_warmup=2,
        steps_per_sample=1,
        beta=0.7,
    )
    reference_samples = gibbs.reference_thrml_sample(
        biases,
        couplings,
        candidate=candidate,
        seed=11,
        n_samples=3,
        n_warmup=2,
        steps_per_sample=1,
        beta=0.7,
    )

    np.testing.assert_array_equal(carnot_samples, reference_samples)
    assert gibbs.constructive_kl_to_thrml(carnot_samples, reference_samples) == 0.0


def test_req_sample_058_zero_coupling_k1_hamming_is_binomial_center() -> None:
    """REQ-SAMPLE-058: zero-coupling K=1 THRML randomizes around n/2 Hamming distance."""

    summary = gibbs.zero_coupling_hamming_summary(n_spins=32, n_samples=96, seed=1564)

    assert summary["n_spins"] == 32
    assert summary["n_warmup"] == 1
    assert summary["mean_hamming_distance"] == pytest.approx(16.0, abs=3.0)
    assert 8 <= summary["min_hamming_distance"] <= summary["max_hamming_distance"] <= 24


def test_req_sample_058_exp1564_deliverable_schema() -> None:
    """REQ-SAMPLE-058: deliverable payload carries the required terminal fields."""

    payload = gibbs.build_exp1564_deliverable_payload(regression_tests_passed=True)

    assert payload["status"] == "complete"
    assert payload["thrml_vendoring_complete"] is True
    assert payload["kl_to_thrml_after_vendoring"] == 0.0
    assert payload["candidate_warm_start_implemented"] is True
    assert payload["regression_tests_passed"] is True
    assert payload["mirror_repo_url"]
    assert payload["honest_verdict"].startswith("complete:")

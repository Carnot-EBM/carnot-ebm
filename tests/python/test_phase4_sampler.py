"""Tests for the Exp 1156 Phase 4 regime-conditional sampler.

Spec coverage: REQ-KONA-010, SCENARIO-KONA-009
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot.phase3.continuous_ebm import ContinuousEBM
from carnot.samplers.backend import SamplerBackend
from carnot.samplers.phase4_sampler import (
    Phase4Sampler,
    continuous_ebm_energy,
    sampler_algorithm_from_exp1155,
)


def _exp1155_artifact(path: Path, **overrides: object) -> Path:
    payload: dict[str, object] = {
        "hmc_regime": "C",
        "recommended_sampler": "blocked_gibbs",
        "d4_discrete_components_bottleneck": True,
        "component_names": [
            "SOSKANEnergyV3",
            "SemEnergyProbe",
            "ASTStructureVerifier",
            "SemanticConsistencyVerifier",
            "Z3MathVerifier",
        ],
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_from_exp1155_selects_blocked_gibbs_and_protocol(tmp_path: Path) -> None:
    """REQ-KONA-010: Regime C plus D4 bottleneck selects blocked Gibbs."""
    path = _exp1155_artifact(tmp_path / "exp1155.json")

    sampler = Phase4Sampler.from_exp1155(path, seed=7)
    fallback_sampler = Phase4Sampler.from_exp1155(
        _exp1155_artifact(tmp_path / "exp1155_no_names.json", component_names=[]),
        seed=7,
    )

    assert isinstance(sampler, SamplerBackend)
    assert sampler.backend_name == "phase4_blocked_gibbs"
    assert sampler.algorithm == "blocked_gibbs"
    assert sampler.hmc_regime_used == "C"
    assert sampler.discrete_indices == (2, 4)
    assert fallback_sampler.discrete_indices == (0,)


def test_sampler_algorithm_mapping_covers_all_exp1155_recommendations() -> None:
    """REQ-KONA-010: Exp 1155 recommendations map to artifact algorithm names."""
    assert sampler_algorithm_from_exp1155({"recommended_sampler": "hmc"}) == "numpyro_nuts"
    assert (
        sampler_algorithm_from_exp1155({"recommended_sampler": "preconditioned_hmc"})
        == "preconditioned_hmc"
    )
    assert (
        sampler_algorithm_from_exp1155({"recommended_sampler": "blocked_gibbs"}) == "blocked_gibbs"
    )
    assert sampler_algorithm_from_exp1155({"recommended_sampler": "langevin"}) == "sgld"

    with pytest.raises(ValueError, match="Unknown Exp 1155 sampler recommendation"):
        sampler_algorithm_from_exp1155({"recommended_sampler": "not_a_sampler"})


def test_blocked_gibbs_samples_bounded_chain_and_flips_discrete_coordinate() -> None:
    """SCENARIO-KONA-009: blocked Gibbs updates discrete coordinates and stays bounded."""

    def energy(x: np.ndarray) -> float:
        return float(0.1 * np.dot(x, x) - 4.0 * x[0] + 4.0 * x[1])

    sampler = Phase4Sampler(
        algorithm="blocked_gibbs",
        seed=12,
        step_size=0.01,
        discrete_indices=(0, 1),
        continuous_indices=(2,),
    )

    chain = sampler.sample(energy, np.array([-1.0, 1.0, 0.2]), 40)

    assert chain.shape == (40, 3)
    assert np.all(np.isfinite(chain))
    assert np.max(np.abs(chain)) <= 1.0
    assert set(np.unique(chain[:, 0])).issubset({-1.0, 1.0})
    assert chain[-1, 0] == 1.0
    assert chain[-1, 1] == -1.0
    assert sampler.last_diagnostics["acceptance_rate"] is None
    assert sampler.last_diagnostics["discrete_update_rate"] > 0.0


def test_sgld_uses_continuous_ebm_gradient_and_records_convergence() -> None:
    """REQ-KONA-010: general Regime C fallback uses adaptive SGLD."""
    model = ContinuousEBM(
        variables=2,
        coupling=-0.5 * np.eye(2, dtype=np.float64),
        bias=np.array([0.25, -0.25], dtype=np.float64),
    )
    energy = continuous_ebm_energy(model)
    sampler = Phase4Sampler(algorithm="sgld", seed=3, step_size=0.02)

    assert energy(np.zeros(2, dtype=np.float64)) == pytest.approx(0.0)
    chain = sampler.sample(energy, np.array([0.8, -0.8], dtype=np.float64), 25)

    assert chain.shape == (25, 2)
    assert np.all(np.isfinite(chain))
    assert np.max(np.abs(chain)) <= 1.0
    assert sampler.last_diagnostics["acceptance_rate"] is None
    assert sampler.last_diagnostics["mean_step_size"] < sampler.step_size


def test_backend_style_sample_and_minimize_energy_return_boolean_samples() -> None:
    """REQ-KONA-010: Phase4Sampler keeps SamplerBackend-shaped Ising methods."""
    biases = np.array([1.5, -1.5], dtype=np.float64)
    couplings = np.zeros((2, 2), dtype=np.float64)
    sampler = Phase4Sampler(seed=5, step_size=0.01)

    fixed_temp = sampler.sample(
        biases,
        couplings,
        3,
        config={"beta": 2.0, "n_steps": 12},
    )
    minimized = sampler.minimize_energy(
        biases,
        couplings,
        n_samples=3,
        n_steps=12,
        beta=2.0,
    )

    assert fixed_temp.shape == (3, 2)
    assert fixed_temp.dtype == bool
    assert minimized.shape == (3, 2)
    assert minimized.dtype == bool


def test_invalid_and_hmc_paths_are_explicit() -> None:
    """REQ-KONA-010: unsupported or invalid sampler requests fail clearly."""
    with pytest.raises(ValueError, match="Unsupported Phase 4 sampler algorithm"):
        Phase4Sampler(algorithm="bad_sampler")

    sampler = Phase4Sampler(algorithm="numpyro_nuts")
    with pytest.raises(NotImplementedError, match="Regime C"):
        sampler.sample(lambda x: float(np.dot(x, x)), np.zeros(2), 2)

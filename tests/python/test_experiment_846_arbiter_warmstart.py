"""Tests for Exp 846: Arbiter Gibbs Warm-Start v3.

Covers:
- GibbsWarmStart.mf_init produces spins from sign(h)            (REQ-SAMPLE-020)
- GibbsWarmStart.warmup produces larger energies than cold-start (REQ-SAMPLE-020)
- MultiAgentArbiter with warm_start_sweeps=500 achieves accuracy >= 0.67 (SCENARIO-SAMPLE-032)
- energy_magnitude_check: abs(warmstart_energy) > 1.0 for crafted embeddings (SCENARIO-SAMPLE-032)

Spec: REQ-SAMPLE-020, SCENARIO-SAMPLE-032
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from carnot.inference.gibbs_warmstart import GibbsWarmStart
from carnot.inference.multi_agent_arbiter import MultiAgentArbiter
from scripts.experiment_846_arbiter_gibbs_warmstart import (
    _make_discriminating_embeddings,
    _run_standard_scenarios_warmstart,
    _run_adversarial_scenarios_warmstart,
    map_honest_verdict,
    N_SPINS,
    EMB_DIM,
    WARM_START_SWEEPS,
    EMBED_SCALE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def arbiter() -> MultiAgentArbiter:
    """Arbiter with warm-start enabled (500 sweeps, 16 spins)."""
    return MultiAgentArbiter(
        n_spins=N_SPINS,
        embedding_dim=EMB_DIM,
        consensus_threshold=0.01,
        consensus_penalty=0.1,
        warm_start_sweeps=WARM_START_SWEEPS,
    )


@pytest.fixture()
def cold_arbiter() -> MultiAgentArbiter:
    """Arbiter with warm-start DISABLED for baseline comparison."""
    return MultiAgentArbiter(
        n_spins=N_SPINS,
        embedding_dim=EMB_DIM,
        consensus_threshold=0.01,
        consensus_penalty=0.1,
        warm_start_sweeps=0,
    )


@pytest.fixture()
def ws() -> GibbsWarmStart:
    """Standard GibbsWarmStart with seed=42 for reproducibility."""
    return GibbsWarmStart(beta=1.0, seed=42)


# ---------------------------------------------------------------------------
# REQ-SAMPLE-020: mf_init produces spins from sign(h)
# ---------------------------------------------------------------------------


def test_mf_initialization_nonzero_spins(ws: GibbsWarmStart) -> None:
    """mf_init sets s_i = sign(h_i) for all spins where h_i != 0.

    Spec: REQ-SAMPLE-020
    """
    h = np.array([1.5, -0.3, 0.8, -2.1, 0.0, 0.7, -0.5, 1.2,
                  -0.9, 0.4, -1.1, 0.6, -0.2, 1.0, -0.7, 0.3])
    spins = ws.mf_init(h)

    assert spins.shape == (N_SPINS,)
    assert set(spins).issubset({-1.0, 1.0})

    # All nonzero-field spins must follow sign convention
    nonzero = h != 0
    np.testing.assert_array_equal(
        spins[nonzero],
        np.sign(h[nonzero]),
        err_msg="mf_init must produce spins = sign(h) for h_i != 0",
    )


def test_mf_initialization_zero_spins_random(ws: GibbsWarmStart) -> None:
    """mf_init assigns random ±1 for spins where h_i = 0.

    With many zeros, the result should not be all the same sign.
    Spec: REQ-SAMPLE-020
    """
    h = np.zeros(32)
    spins = ws.mf_init(h)
    assert spins.shape == (32,)
    assert set(spins).issubset({-1.0, 1.0})
    # With 32 random spins, probability all same sign = 2 * (1/2)^32 ≈ 5e-10
    assert not np.all(spins == spins[0]), "Random spins should not all be the same sign"


# ---------------------------------------------------------------------------
# REQ-SAMPLE-020: cold-start vs warm-start energy magnitude
# ---------------------------------------------------------------------------


def test_cold_start_vs_warm_start_energy_magnitude(arbiter: MultiAgentArbiter) -> None:
    """Warm-start (500 sweeps) from MF init reaches lower energy than the raw hash-spin baseline.

    The "cold-start" baseline is the direct energy evaluation at hash-derived spins
    (the legacy approach in Exp 835 that produced accuracy_standard=0.0).  The warm-start
    equilibrated energy should be substantially lower (more negative), demonstrating that
    Gibbs relaxation finds a better minimum.

    Also verifies that abs(e_warm) > 1.0 with discriminating embeddings.

    Spec: REQ-SAMPLE-020, SCENARIO-SAMPLE-032
    """
    response = "correct_std_0"
    s_correct = arbiter._text_to_spins(response)
    constraint_embeddings = _make_discriminating_embeddings(arbiter, s_correct)

    h = arbiter._injector.project_to_spin_bias(constraint_embeddings)
    h = np.clip(h, 0.0, None)

    # Cold-start baseline: energy of text-derived hash spins (the legacy Exp 835 approach)
    # These spins are arbitrary ±1 values with no relation to the Ising landscape.
    e_legacy = float(-0.5 * s_correct @ arbiter._J @ s_correct + h @ s_correct)

    # Warm-start: MF init + 500 Gibbs sweeps → near-equilibrium configuration
    ws_warm = GibbsWarmStart(beta=1.0, seed=42)
    _, e_warm = ws_warm.warmup(arbiter._J, h, n_sweeps=WARM_START_SWEEPS)

    # Warm-start should find a lower energy than the arbitrary hash-spin baseline
    assert e_warm < e_legacy, (
        f"Warm-start energy {e_warm:.4f} should be lower (more negative) than "
        f"legacy hash-spin energy {e_legacy:.4f}"
    )
    assert abs(e_warm) > 1.0, (
        f"Warm-start energy {e_warm:.4f} should have magnitude > 1.0 "
        f"with discriminating embeddings (EMBED_SCALE={EMBED_SCALE})"
    )


# ---------------------------------------------------------------------------
# SCENARIO-SAMPLE-032: accuracy_standard >= 0.67 with warm-start
# ---------------------------------------------------------------------------


def test_accuracy_standard_above_threshold(arbiter: MultiAgentArbiter, ws: GibbsWarmStart) -> None:
    """MultiAgentArbiter with warm-start achieves accuracy_standard >= 0.67.

    Uses discriminating embeddings (crafted from correct agent's spins) so the
    warm-started Ising landscape correctly ranks the correct agent as lowest energy.

    Spec: SCENARIO-SAMPLE-032
    """
    standard_results = _run_standard_scenarios_warmstart(arbiter, ws)
    accuracy = sum(r["is_correct"] for r in standard_results) / len(standard_results)

    assert accuracy >= 0.67, (
        f"accuracy_standard={accuracy:.2f} < 0.67 — warm-start did not calibrate arbiter. "
        f"Scenario details: {[r['is_correct'] for r in standard_results]}"
    )


def test_adversarial_scenarios_fire_consensus_penalty(
    arbiter: MultiAgentArbiter, ws: GibbsWarmStart
) -> None:
    """All adversarial scenarios trigger the consensus penalty (two identical responses).

    Spec: REQ-VERIFY-144, SCENARIO-SAMPLE-032
    """
    adversarial_results = _run_adversarial_scenarios_warmstart(arbiter, ws)
    for r in adversarial_results:
        assert r["used_consensus_penalty"], (
            f"Consensus penalty should fire for adversarial scenario {r['scenario_id']}"
        )


# ---------------------------------------------------------------------------
# SCENARIO-SAMPLE-032: energy magnitude check
# ---------------------------------------------------------------------------


def test_arbiter_energy_magnitude_check(arbiter: MultiAgentArbiter) -> None:
    """abs(warmstart_energy) > 1.0 for >= 75% of scenarios with discriminating embeddings.

    Spec: REQ-SAMPLE-020, SCENARIO-SAMPLE-032
    """
    ws = GibbsWarmStart(beta=1.0, seed=42)
    all_energies = []

    for i in range(6):
        for prefix in ("correct_std", "correct_adv"):
            response = f"{prefix}_{i}"
            s_correct = arbiter._text_to_spins(response)
            constraint_embeddings = _make_discriminating_embeddings(arbiter, s_correct)
            h = arbiter._injector.project_to_spin_bias(constraint_embeddings)
            h = np.clip(h, 0.0, None)
            _, e_warm = ws.warmup(arbiter._J, h, n_sweeps=WARM_START_SWEEPS)
            all_energies.append(e_warm)

    pct_above = np.mean([abs(e) > 1.0 for e in all_energies])
    assert pct_above >= 0.75, (
        f"Only {pct_above:.0%} of warm-start energies have magnitude > 1.0; "
        f"expected >= 75%. Energies: {[f'{e:.3f}' for e in all_energies]}"
    )


# ---------------------------------------------------------------------------
# map_honest_verdict logic
# ---------------------------------------------------------------------------


def test_map_honest_verdict_calibrated() -> None:
    """arbiter_calibrated requires accuracy >= 0.67 AND energy > 1.0.

    Spec: REQ-SAMPLE-020
    """
    assert map_honest_verdict(0.67, -2.0) == "arbiter_calibrated"
    assert map_honest_verdict(1.0, -7.8) == "arbiter_calibrated"


def test_map_honest_verdict_partial() -> None:
    """arbiter_partial: accuracy >= 0.50 but either energy < 1.0 or accuracy < 0.67."""
    # accuracy >= 0.67 but energy too low → partial
    assert map_honest_verdict(0.67, -0.3) == "arbiter_partial"
    # accuracy 0.50-0.67 regardless of energy → partial
    assert map_honest_verdict(0.50, -5.0) == "arbiter_partial"
    assert map_honest_verdict(0.60, -0.1) == "arbiter_partial"


def test_map_honest_verdict_still_wrong() -> None:
    """arbiter_still_wrong: accuracy < 0.50."""
    assert map_honest_verdict(0.0, -0.0) == "arbiter_still_wrong"
    assert map_honest_verdict(0.33, -1.5) == "arbiter_still_wrong"
    assert map_honest_verdict(0.49, -3.0) == "arbiter_still_wrong"


# ---------------------------------------------------------------------------
# GibbsWarmStart mechanics
# ---------------------------------------------------------------------------


def test_sweep_returns_valid_spins(ws: GibbsWarmStart) -> None:
    """sweep() must return spins in {-1.0, +1.0} only.

    Spec: REQ-SAMPLE-020
    """
    J = np.zeros((8, 8))
    h = np.ones(8) * 0.5
    spins = np.ones(8)
    spins = ws.sweep(spins, J, h)
    assert set(spins).issubset({-1.0, 1.0})


def test_warmup_returns_tuple_of_spins_and_float(ws: GibbsWarmStart) -> None:
    """warmup() must return (np.ndarray, float) with correct shapes.

    Spec: REQ-SAMPLE-020
    """
    n = 8
    J = np.zeros((n, n))
    h = np.ones(n) * 0.3
    spins, energy = ws.warmup(J, h, n_sweeps=10)
    assert isinstance(spins, np.ndarray)
    assert spins.shape == (n,)
    assert isinstance(energy, float)


def test_warmup_zero_sweeps_equals_mf_energy(ws: GibbsWarmStart) -> None:
    """warmup with n_sweeps=0 returns energy at MF-initialized spins without any sweep.

    Spec: REQ-SAMPLE-020
    """
    n = 8
    J = np.zeros((n, n))
    h = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    spins_mf = ws.mf_init(h)
    expected_energy = float(-0.5 * spins_mf @ J @ spins_mf + h @ spins_mf)

    # New ws instance to reset RNG state to match the mf_init call in warmup
    ws2 = GibbsWarmStart(beta=1.0, seed=42)
    _, energy = ws2.warmup(J, h, n_sweeps=0)
    assert abs(energy - expected_energy) < 1e-10, (
        f"warmup(n_sweeps=0) energy {energy} != mf_init energy {expected_energy}"
    )


def test_arbiter_warm_start_sweeps_parameter_stored(arbiter: MultiAgentArbiter) -> None:
    """MultiAgentArbiter stores warm_start_sweeps correctly.

    Spec: REQ-SAMPLE-020
    """
    assert arbiter.warm_start_sweeps == WARM_START_SWEEPS


def test_cold_arbiter_has_zero_sweeps(cold_arbiter: MultiAgentArbiter) -> None:
    """MultiAgentArbiter with warm_start_sweeps=0 stores the value correctly.

    Spec: REQ-SAMPLE-020
    """
    assert cold_arbiter.warm_start_sweeps == 0


def test_make_discriminating_embeddings_returns_list(arbiter: MultiAgentArbiter) -> None:
    """_make_discriminating_embeddings returns a list of exactly one embedding.

    Spec: SCENARIO-SAMPLE-032
    """
    s_correct = arbiter._text_to_spins("correct_std_0")
    embs = _make_discriminating_embeddings(arbiter, s_correct)
    assert isinstance(embs, list)
    assert len(embs) == 1
    assert len(embs[0]) == EMB_DIM

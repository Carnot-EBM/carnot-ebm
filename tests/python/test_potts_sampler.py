"""Tests for the q=3 Potts sampler and KV260 Potts RTL deliverable.

Spec: REQ-POTTS-001, REQ-POTTS-002, REQ-POTTS-003, REQ-POTTS-004,
      REQ-POTTS-005
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from python.carnot.samplers.potts_sampler import PottsSampler


def _complete_ferromagnetic_j(n_spins: int, weight: float = 1.0) -> np.ndarray:
    j_matrix = np.full((n_spins, n_spins), weight, dtype=np.float64)
    np.fill_diagonal(j_matrix, 0.0)
    return j_matrix


def test_potts_sampler_q3_energy_nonincreasing_in_expectation():
    """REQ-POTTS-002: repeated q=3 sampling lowers mean energy."""
    np.random.seed(1098)
    n_spins = 16
    sampler = PottsSampler(n_spins=n_spins, q=3, beta=3.0)
    j_matrix = _complete_ferromagnetic_j(n_spins)

    initial_energies: list[float] = []
    final_energies: list[float] = []
    for _ in range(30):
        init = np.random.randint(0, 3, size=n_spins)
        initial_energies.append(sampler.energy(j_matrix, init))
        final = sampler.sample(j_matrix, n_steps=80, init_state=init)
        final_energies.append(sampler.energy(j_matrix, final))

    assert np.mean(final_energies) <= np.mean(initial_energies)


def test_potts_sampler_q3_reaches_low_energy_configuration():
    """REQ-POTTS-002: q=3 sampling reaches a near-ground-state basin."""
    np.random.seed(7)
    n_spins = 16
    sampler = PottsSampler(n_spins=n_spins, q=3, beta=6.0)
    j_matrix = _complete_ferromagnetic_j(n_spins)
    final = sampler.sample(j_matrix, n_steps=100)

    ground_energy = -float(n_spins * (n_spins - 1))
    assert sampler.energy(j_matrix, final) <= 0.90 * ground_energy


def test_potts_sampler_q3_distribution_has_three_states():
    """REQ-POTTS-002: unconstrained q=3 runs exercise all three states."""
    np.random.seed(42)
    n_spins = 16
    sampler = PottsSampler(n_spins=n_spins, q=3, beta=2.0)
    j_matrix = np.zeros((n_spins, n_spins), dtype=np.float64)

    states_seen: set[int] = set()
    for _ in range(12):
        final = sampler.sample(j_matrix, n_steps=5)
        states_seen.update(int(v) for v in np.unique(final))

    assert states_seen == {0, 1, 2}


def test_potts_vs_ising_q2_equivalent_on_two_state_boundary():
    """REQ-POTTS-003: q=2 Potts preserves the Ising alignment preference."""
    n_spins = 8
    sampler = PottsSampler(n_spins=n_spins, q=2, beta=4.0)
    j_matrix = _complete_ferromagnetic_j(n_spins)

    aligned = np.zeros(n_spins, dtype=np.int64)
    split = np.arange(n_spins, dtype=np.int64) % 2
    potts_gap = sampler.energy(j_matrix, split) - sampler.energy(j_matrix, aligned)

    ising_aligned = 2 * aligned - 1
    ising_split = 2 * split - 1
    ising_energy_aligned = -float(np.sum(j_matrix * np.outer(ising_aligned, ising_aligned)))
    ising_energy_split = -float(np.sum(j_matrix * np.outer(ising_split, ising_split)))
    ising_gap = ising_energy_split - ising_energy_aligned

    assert potts_gap > 0.0
    assert ising_gap > 0.0
    final = sampler.sample(j_matrix, n_steps=40, init_state=split)
    assert set(np.unique(final)).issubset({0, 1})


def test_potts_verilog_file_exists_with_n_spins_parameter():
    """REQ-POTTS-004: Potts RTL exposes the required KV260 parameters."""
    rtl_path = Path("hardware/kv260/potts_sampler_v1.v")
    assert rtl_path.exists()
    rtl = rtl_path.read_text()

    assert "module potts_sampler_v1" in rtl
    assert re.search(r"parameter\s+integer\s+N_SPINS\s*=\s*64", rtl)
    assert re.search(r"parameter\s+integer\s+Q_STATES\s*=\s*3", rtl)
    assert re.search(r"parameter\s+\[7:0\]\s+BETA_FIXED\s*=\s*8'h40", rtl)
    assert "2 bits per spin" in rtl
    assert "softmax" in rtl.lower()
    assert "lfsr2" in rtl.lower()

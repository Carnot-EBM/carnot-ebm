"""Tests for the KV260 v3 sequential Ising-sampler simulator + RTL.

Spec: REQ-HARDWARE-016, SCENARIO-HARDWARE-016, REQ-SAMPLE-012
Cross-ref: experiment 1109 (phase-2a sampler correctness fix).

These tests pin the sequential single-site sampler's behaviour against
three concrete invariants and one source-level invariant on the Verilog:

1. Each step touches exactly one spin (the round-robin selector). This
   is what makes the chain ergodic on frustrated graphs; if the
   simulator silently flipped multiple spins per step we'd be back to
   the broken parallel design.
2. KL against the closed-form Boltzmann distribution falls below the
   Phase-2a acceptance gate (KL < 0.05) on the same antiferromagnetic
   ring exp1094 used.
3. Sequential KL is strictly lower than parallel KL on the same problem
   — the head-to-head proof that the update order is the load-bearing
   change.
4. The new Verilog file exists and contains the spin_select counter
   that distinguishes it from the parallel design.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Allow the test to import sampler_sim directly without pulling in the rest
# of the carnot.hardware package (which transitively requires JAX). The same
# pattern is used by other CPU-only hardware tests in this directory.
import importlib.util
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLER_SIM_PATH = REPO_ROOT / "python" / "carnot" / "hardware" / "sampler_sim.py"


def _load_sampler_sim():
    """Import sampler_sim by file path, sidestepping carnot.hardware.__init__.

    The test must run on hosts without JAX installed (CPU-only CI). The
    package's ``__init__`` eagerly imports ``carnot.hardware.fpga_backend``
    which imports JAX, which is irrelevant to this test. Loading via
    importlib.util keeps the test hermetic.
    """
    spec = importlib.util.spec_from_file_location("sampler_sim", SAMPLER_SIM_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("sampler_sim", mod)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def sampler_sim():
    """Load the sampler_sim module once per test module."""
    return _load_sampler_sim()


def test_sequential_sampler_updates_one_spin_per_step(sampler_sim):
    """A single step() call must change at most one spin.

    Why: the entire point of the sequential design is that detailed
    balance is restored by single-site updates. If the implementation
    accidentally flipped multiple spins per step (e.g., by sharing a
    random uniform across sites) we would be silently re-introducing
    the parallel-synchronous failure mode. This test fires the step()
    repeatedly and asserts the Hamming distance between consecutive
    states is at most 1.
    """
    prob = sampler_sim.antiferromagnetic_ring(n_spins=8, beta=2.0)
    sampler = sampler_sim.SynchronousIsingSamplerV3(prob, seed=123)
    prev = sampler.s.copy()
    for _ in range(200):
        sampler.step()
        diff = int(np.sum(prev != sampler.s))
        assert diff <= 1, (
            f"single-site step changed {diff} spins; expected <= 1 — "
            "the sampler is silently reverting to parallel updates."
        )
        prev = sampler.s.copy()


def test_sequential_sampler_kl_below_005_on_frustrated_ring(sampler_sim):
    """KL(sim_v3 || true_Gibbs) must fall below the Phase-2a gate.

    Why: this is the empirical claim that justifies burning the v3
    bitstream. exp1094 measured KL = 3.07 for the parallel design;
    if v3 cannot beat 0.05 in pure Python simulation (where there is
    no LFSR weirdness, no Q8.8 quantisation, no AXI race condition),
    the sequential approach is wrong even before silicon. Default
    sweep-spaced recording is used so consecutive samples are nearly
    independent and the threshold is tight on stationary distribution
    accuracy rather than on autocorrelation length.
    """
    prob = sampler_sim.antiferromagnetic_ring(n_spins=8, beta=2.0)
    sampler = sampler_sim.SynchronousIsingSamplerV3(prob, seed=42)
    samples = sampler.sample(n_steps=60000, burn_in_sweeps=500)
    kl = sampler_sim.kl_against_true_gibbs(samples, prob)
    assert kl < 0.05, (
        f"sequential v3 KL = {kl:.4f}, expected < 0.05 — "
        "the simulator does not converge to the true Boltzmann "
        "distribution; do not synthesise the bitstream."
    )


def test_sequential_vs_parallel_kl_comparison_sequential_is_lower(sampler_sim):
    """Sequential KL must be strictly lower than parallel KL on same J.

    Why: the comparison is the head-to-head evidence that the update
    order is the load-bearing change. Same problem, same beta, same
    seed, same sample budget — only the sweep order differs. We expect
    parallel KL to be at least 10x larger; we assert "strictly lower"
    rather than a specific ratio so the test is robust to changes in
    burn-in, sample count, or smoothing constants.
    """
    prob = sampler_sim.antiferromagnetic_ring(n_spins=8, beta=2.0)
    seq = sampler_sim.SynchronousIsingSamplerV3(prob, seed=7)
    seq_samples = seq.sample(n_steps=5000, burn_in_sweeps=200)
    kl_seq = sampler_sim.kl_against_true_gibbs(seq_samples, prob)

    par = sampler_sim.SynchronousIsingSamplerV1(prob, seed=7)
    par_samples = par.sample(n_steps=5000, burn_in=200)
    kl_par = sampler_sim.kl_against_true_gibbs(par_samples, prob)

    assert kl_seq < kl_par, (
        f"sequential KL ({kl_seq:.4f}) is not lower than parallel KL "
        f"({kl_par:.4f}) — the regression test for exp1094's finding "
        "has flipped, investigate the simulator before shipping."
    )


def test_verilog_v3_file_exists_and_has_spin_select_counter():
    """The new Verilog file must exist and contain a spin_select register.

    Why: a passing simulation alone does not guarantee that the matching
    RTL was actually written. We check the file presence and grep for
    the spin_select counter declaration — the structural property that
    distinguishes the sequential design from the parallel one. If the
    counter is missing, the synthesised bitstream would still update
    every spin in parallel and the simulation/RTL would diverge.
    """
    rtl_path = REPO_ROOT / "hardware" / "kv260" / "ising_sampler_v3_sequential.v"
    assert rtl_path.exists(), f"missing RTL file: {rtl_path}"
    text = rtl_path.read_text()
    assert re.search(r"\bspin_select\b", text), (
        "spin_select counter not found in ising_sampler_v3_sequential.v — "
        "RTL is not actually sequential. Did you accidentally copy v1/v2?"
    )
    assert re.search(r"reg\s*\[\s*SPIN_SEL_W\s*-\s*1\s*:\s*0\s*\]\s*spin_select", text), (
        "spin_select declaration does not match the expected form "
        "`reg [SPIN_SEL_W-1:0] spin_select` — review the RTL header."
    )

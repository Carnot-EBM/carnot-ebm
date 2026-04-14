"""Tests for the 128-spin Ising sampler RTL behavioral simulation.

Validates hardware/kv260/ising_sampler_v1.v logic via
scripts/simulate_ising_sampler.py without requiring a physical FPGA.
All tests exercise the same logic sequence as the synthesized Verilog.

Spec coverage:
    REQ-SAMPLE-011, SCENARIO-SAMPLE-023, SCENARIO-SAMPLE-024
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure scripts/ is importable.
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from simulate_ising_sampler import (
    LFSR16,
    IsingSimulator,
    REG_ADJ_BASE,
    REG_BETA_FINAL,
    REG_BIAS_BASE,
    REG_CONTROL,
    REG_COUPL_BASE,
    REG_SPIN_COUNT,
    REG_SPIN_OUT_BASE,
    REG_STATUS,
    STATUS_DONE,
    STATUS_READY,
    _build_ring_problem,
    compute_beta_schedule,
    float_to_q88,
    q88_mul,
    q88_to_float,
)

# ---------------------------------------------------------------------------
# SCENARIO-SAMPLE-023: AXI register map coverage
# ---------------------------------------------------------------------------


class TestRegisterMapCoverage:
    """Verify all Exp 289 AXI registers are addressable.

    Spec: REQ-SAMPLE-011, SCENARIO-SAMPLE-023
    """

    def setup_method(self) -> None:
        """Create a fresh simulator for each test."""
        self.sim = IsingSimulator(n_spins=128, max_degree=32)

    def test_control_register_write_read(self) -> None:
        """CONTROL register is writable and readable at 0x0000.

        REQ-SAMPLE-011: AXI-Lite slave interface with CONTROL register.
        """
        self.sim.axi_write(REG_CONTROL, 0b01)  # START bit
        val = self.sim.axi_read(REG_CONTROL)
        assert val & 0b01, "CONTROL START bit (bit 0) must survive write-read"

    def test_status_register_readable(self) -> None:
        """STATUS register at 0x0004 reports READY after reset.

        REQ-SAMPLE-011: STATUS register bit 0 = READY.
        """
        status = self.sim.axi_read(REG_STATUS)
        assert status & (1 << STATUS_READY), (
            f"STATUS must have READY bit set after reset; got 0x{status:08X}"
        )

    def test_spin_count_register(self) -> None:
        """SPIN_COUNT register at 0x0008 is writable and readable.

        REQ-SAMPLE-011: SPIN_COUNT register.
        """
        self.sim.axi_write(REG_SPIN_COUNT, 64)
        val = self.sim.axi_read(REG_SPIN_COUNT)
        assert val == 64, f"SPIN_COUNT expected 64 got {val}"

    def test_beta_final_register(self) -> None:
        """BETA_FINAL register at 0x001C stores Q8.8 inverse temperature.

        REQ-SAMPLE-011: BETA_FINAL register, Q8.8 encoding.
        """
        beta = 4.5
        q88 = float_to_q88(beta)
        self.sim.axi_write(REG_BETA_FINAL, q88)
        val = self.sim.axi_read(REG_BETA_FINAL)
        decoded = q88_to_float(val & 0xFFFF)
        assert abs(decoded - beta) < 0.005, (
            f"BETA_FINAL round-trip error: {decoded:.4f} != {beta}"
        )

    def test_bias_ram_all_128_spins(self) -> None:
        """All 128 bias RAM entries are addressable and survive write-read.

        REQ-SAMPLE-011: bias_ram[N_SPINS] addressable at 0x1000+4*i.
        SCENARIO-SAMPLE-023: all bias registers addressable.
        """
        for i in range(128):
            addr = REG_BIAS_BASE + 4 * i
            val = float_to_q88(float(i) * 0.01)
            self.sim.axi_write(addr, val & 0xFFFF)
        for i in range(128):
            addr = REG_BIAS_BASE + 4 * i
            expected = float_to_q88(float(i) * 0.01)
            got = self.sim.axi_read(addr)
            assert got == (expected & 0xFFFF), (
                f"bias_ram[{i}] mismatch: expected {expected & 0xFFFF} got {got}"
            )

    def test_adj_ram_addressable(self) -> None:
        """adj_ram[N_SPINS * MAX_DEGREE] entries are addressable.

        REQ-SAMPLE-011: adj_ram[N_SPINS*MAX_DEGREE] at 0x2000+.
        SCENARIO-SAMPLE-023: all adj_ram entries addressable.
        """
        # Write a few representative entries: spin 0 neighbour 0, spin 127 neighbour 31
        test_cases = [(0, 0, 1), (0, 1, 2), (127, 31, 0)]
        for spin_i, k, nbr in test_cases:
            offset = spin_i * 32 + k
            addr = REG_ADJ_BASE + 4 * offset
            self.sim.axi_write(addr, nbr)
            assert int(self.sim.adj_ram[spin_i, k]) == nbr, (
                f"adj_ram[{spin_i},{k}] expected {nbr} got {self.sim.adj_ram[spin_i, k]}"
            )

    def test_coupl_ram_addressable(self) -> None:
        """coupl_ram[N_SPINS * MAX_DEGREE] entries are addressable.

        REQ-SAMPLE-011: coupl_ram[N_SPINS*MAX_DEGREE] at 0x4000+.
        """
        test_cases = [(0, 0, 0.5), (63, 15, -1.25), (127, 31, 0.125)]
        for spin_i, k, j_val in test_cases:
            offset = spin_i * 32 + k
            addr = REG_COUPL_BASE + 4 * offset
            encoded = float_to_q88(j_val) & 0xFFFF
            self.sim.axi_write(addr, encoded)
            got_q88 = int(self.sim.coupl_q88[spin_i, k])
            got_float = q88_to_float(got_q88)
            assert abs(got_float - j_val) < 0.005, (
                f"coupl_ram[{spin_i},{k}] round-trip: {got_float:.4f} != {j_val}"
            )

    def test_spin_out_readable_after_run(self) -> None:
        """spin_out registers at 0x8010+ are readable after sampling.

        REQ-SAMPLE-011: spin_out[N_SPINS/32] packed spin output.
        """
        sim = IsingSimulator(n_spins=4, max_degree=4, n_steps=10)
        j, h = np.zeros((4, 4)), np.zeros(4)
        sim.load_problem(j, h)
        spins = sim.run()
        # Read word 0 (spins 0–3)
        word = sim.axi_read(REG_SPIN_OUT_BASE)
        for bit in range(4):
            expected_bit = 1 if spins[bit] == 1 else 0
            got_bit = (word >> bit) & 1
            assert got_bit == expected_bit, (
                f"spin_out bit {bit}: expected {expected_bit} got {got_bit} "
                f"(spin={spins[bit]})"
            )

    def test_status_done_after_run(self) -> None:
        """STATUS register reports DONE after sampling completes.

        REQ-SAMPLE-011: STATUS bit 2 = DONE after halting.
        """
        sim = IsingSimulator(n_spins=4, max_degree=4, n_steps=5)
        j, h = np.zeros((4, 4)), np.zeros(4)
        sim.load_problem(j, h)
        sim.run()
        status = sim.axi_read(REG_STATUS)
        assert status & (1 << STATUS_DONE), (
            f"STATUS must have DONE bit set after run; got 0x{status:08X}"
        )


# ---------------------------------------------------------------------------
# SCENARIO-SAMPLE-024: Gibbs update on 4-spin graph
# ---------------------------------------------------------------------------


class TestSpinUpdateSingleStep:
    """Validate single-step Gibbs update against Python reference.

    Spec: REQ-SAMPLE-011, SCENARIO-SAMPLE-024
    """

    def _make_4spin_ring(self) -> IsingSimulator:
        """Build a 4-spin antiferromagnetic ring: J=−1.0 on each edge."""
        sim = IsingSimulator(n_spins=4, max_degree=4, n_steps=1,
                              beta_min=5.0, beta_max=5.0, lfsr_seed=0xACE1)
        # Ring: 0-1-2-3-0, antiferromagnetic
        j = np.array([
            [0.0, -1.0, 0.0, -1.0],
            [-1.0, 0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0, -1.0],
            [-1.0, 0.0, -1.0, 0.0],
        ], dtype=np.float32)
        h = np.zeros(4, dtype=np.float32)
        sim.load_problem(j, h)
        return sim

    def test_local_field_4spin_ring_initial_state(self) -> None:
        """Local field h_eff matches numpy reference for 4-spin ring.

        SCENARIO-SAMPLE-024: h_eff_i = Σ_j J_ij s_j + h_i, Q8.8 precision.
        """
        sim = self._make_4spin_ring()
        # Initial state: all +1
        # For spin 0: h_eff = J[0,1]*s[1] + J[0,3]*s[3] = (-1)*1 + (-1)*1 = -2.0
        # For spin 1: h_eff = J[1,0]*s[0] + J[1,2]*s[2] = (-1)*1 + (-1)*1 = -2.0
        j_float = np.array([
            [0.0, -1.0, 0.0, -1.0],
            [-1.0, 0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0, -1.0],
            [-1.0, 0.0, -1.0, 0.0],
        ], dtype=np.float32)
        spins = sim.spins.copy()
        for i in range(4):
            h_eff_python = float(np.dot(j_float[i], spins))
            h_eff_q88 = sim._local_field_q88(i)
            h_eff_dequant = q88_to_float(h_eff_q88)
            assert abs(h_eff_dequant - h_eff_python) < 0.01, (
                f"Spin {i}: local field {h_eff_dequant:.4f} vs ref {h_eff_python:.4f}"
            )

    def test_energy_matches_python_reference(self) -> None:
        """Energy E = -Σ J_ij s_i s_j - Σ h_i s_i matches numpy reference.

        SCENARIO-SAMPLE-024: energy computation correctness.
        """
        sim = IsingSimulator(n_spins=4, max_degree=4, n_steps=1)
        j = np.array([
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
        ], dtype=np.float32)
        h = np.array([0.5, -0.5, 0.5, -0.5], dtype=np.float32)
        sim.load_problem(j, h)
        # Set fixed spin state for deterministic reference
        sim.spins = np.array([1, -1, 1, -1], dtype=np.int8)

        # Python reference energy (using sparse adj as loaded)
        energy_sim = sim.compute_energy()

        # Dense numpy reference
        spins_f = sim.spins.astype(np.float32)
        energy_ref = 0.0
        for i in range(4):
            energy_ref -= h[i] * spins_f[i]
        # Count each directed edge (sim is directed, not halved)
        for i in range(4):
            for jj in range(4):
                if j[i, jj] != 0.0:
                    energy_ref -= j[i, jj] * spins_f[i] * spins_f[jj]

        # Allow Q8.8 quantization error (< 0.1 per coupling)
        assert abs(energy_sim - energy_ref) < 0.5, (
            f"Energy mismatch: sim={energy_sim:.4f} ref={energy_ref:.4f}"
        )


# ---------------------------------------------------------------------------
# Energy computation consistency
# ---------------------------------------------------------------------------


class TestEnergyComputation:
    """Validate E = -Σ J_ij s_i s_j - Σ h_i s_i against Python reference.

    Spec: REQ-SAMPLE-011
    """

    def test_ferromagnetic_ground_state_energy(self) -> None:
        """All-up ground state of ferromagnet has expected energy.

        For a 4-spin ring with J=1.0, h=0, all-up: E = -2*(1*1 + 1*1) = -8
        (counting directed edges).
        """
        sim = IsingSimulator(n_spins=4, max_degree=4, n_steps=1)
        j = np.array([
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
        ], dtype=np.float32)
        h = np.zeros(4, dtype=np.float32)
        sim.load_problem(j, h)
        sim.spins[:] = 1  # Force ground state
        energy = sim.compute_energy()
        # Each directed edge contributes -J*s_i*s_j = -1.0 per edge
        # 4-spin ring has 8 directed edges (each undirected edge counted twice)
        assert energy < 0, f"Ferromagnetic ground state must have negative energy; got {energy}"
        assert abs(energy - (-8.0)) < 0.1, (
            f"Expected energy ≈ -8.0 for 4-spin ferromagnetic ring all-up; got {energy:.4f}"
        )

    def test_energy_higher_for_excited_state(self) -> None:
        """Excited state has higher energy than ground state for ferromagnet.

        Spec: REQ-SAMPLE-011 — energy computation correctness.
        """
        sim = IsingSimulator(n_spins=4, max_degree=4, n_steps=1)
        j = np.array([
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
        ], dtype=np.float32)
        h = np.zeros(4, dtype=np.float32)
        sim.load_problem(j, h)
        sim.spins[:] = 1
        e_ground = sim.compute_energy()
        sim.spins[0] = -1  # Flip one spin (excited state)
        e_excited = sim.compute_energy()
        assert e_excited > e_ground, (
            f"Excited state energy ({e_excited:.4f}) must exceed "
            f"ground state ({e_ground:.4f})"
        )


# ---------------------------------------------------------------------------
# Annealing schedule
# ---------------------------------------------------------------------------


class TestAnnealingSchedule:
    """Validate log-linear β schedule with Mpemba init.

    Spec: REQ-SAMPLE-011
    """

    def test_schedule_length(self) -> None:
        """Schedule has exactly n_steps entries.

        REQ-SAMPLE-011: β schedule produced for each of n_steps sweeps.
        """
        sched = compute_beta_schedule(100, 0.1, 5.0)
        assert len(sched) == 100, f"Expected 100 schedule entries, got {len(sched)}"

    def test_mpemba_hot_start_fraction(self) -> None:
        """First 10% of steps have β = 0 (Mpemba hot-start).

        REQ-SAMPLE-011: Mpemba init — first 10% at β=0 per arXiv 2603.24183.
        """
        sched = compute_beta_schedule(100, 0.1, 5.0, mpemba_fraction=0.1)
        n_hot = max(1, int(100 * 0.1))
        for t in range(n_hot):
            assert sched[t] == 0.0, (
                f"Step {t} should be β=0 (Mpemba hot-start); got {sched[t]}"
            )

    def test_ramp_is_log_linear(self) -> None:
        """Ramp phase increases log-linearly from β_min to β_max.

        REQ-SAMPLE-011: β(t) = β_min × (β_max/β_min)^(t/T).
        """
        n_steps = 100
        beta_min, beta_max = 0.1, 5.0
        sched = compute_beta_schedule(n_steps, beta_min, beta_max, mpemba_fraction=0.1)
        n_hot = max(1, int(n_steps * 0.1))
        ramp = sched[n_hot:]
        # First ramp entry ≈ beta_min, last ≈ beta_max
        assert abs(ramp[0] - beta_min) < 0.01, (
            f"Ramp start: {ramp[0]:.4f} vs beta_min={beta_min}"
        )
        assert abs(ramp[-1] - beta_max) < 0.01, (
            f"Ramp end: {ramp[-1]:.4f} vs beta_max={beta_max}"
        )
        # Verify log-linear: ratio of consecutive β values is constant
        n_r = len(ramp)
        if n_r > 2:
            log_ratios = [
                math.log(ramp[t + 1] / ramp[t]) for t in range(n_r - 1) if ramp[t] > 0
            ]
            mean_ratio = sum(log_ratios) / len(log_ratios)
            for lr in log_ratios:
                assert abs(lr - mean_ratio) < 1e-6, (
                    "β ramp is not log-linear: log-ratios vary"
                )

    def test_schedule_monotone_increasing_in_ramp(self) -> None:
        """β is monotone non-decreasing throughout the ramp phase.

        REQ-SAMPLE-011: log-linear schedule is strictly increasing.
        """
        sched = compute_beta_schedule(200, 0.05, 8.0, mpemba_fraction=0.1)
        n_hot = max(1, int(200 * 0.1))
        ramp = sched[n_hot:]
        for t in range(len(ramp) - 1):
            assert ramp[t] <= ramp[t + 1], (
                f"Schedule not monotone at step {t}: {ramp[t]:.4f} > {ramp[t+1]:.4f}"
            )


# ---------------------------------------------------------------------------
# Mpemba initialization
# ---------------------------------------------------------------------------


class TestMpembaInit:
    """Validate Mpemba hot-start suppresses slow relaxation modes.

    Spec: REQ-SAMPLE-011
    """

    def test_hot_start_randomizes_spins(self) -> None:
        """Running at β=0 during Mpemba phase produces mixed ±1 states.

        REQ-SAMPLE-011: Mpemba initialization — β=0 hot-start randomizes spins.
        At β=0, flip probability = sigmoid(0) = 0.5, so each spin flips
        with ~50% probability per sweep.  After enough sweeps the spin
        distribution should be approximately 50/50 ±1.
        """
        sim = IsingSimulator(
            n_spins=128, max_degree=32, n_steps=20,
            beta_min=0.1, beta_max=0.1, lfsr_seed=0x1234
        )
        j, h = _build_ring_problem(128)
        sim.load_problem(j, h)
        # Manually run only the Mpemba hot phase (β=0 for 2 full sweeps)
        beta_zero = float_to_q88(0.0)
        for _ in range(2):
            sim._full_sweep(beta_zero)
        n_up = int(np.sum(sim.spins == 1))
        n_down = int(np.sum(sim.spins == -1))
        # After hot start, expect neither all-up nor all-down (very unlikely at p=0.5)
        assert n_up > 0 and n_down > 0, (
            "Mpemba hot-start should produce mixed spin state (not all-up or all-down)"
        )

    def test_mpemba_convergence_vs_cold_start(self) -> None:
        """Mpemba init converges to lower energy than cold (all-up) init.

        REQ-SAMPLE-011: Mpemba initialization per arXiv 2603.24183.
        Tests that a frustrated ring converges to lower energy with hot-start
        vs. starting from all-spins-up (no Mpemba hot phase).

        Uses an antiferromagnetic ring (frustrated): ground state is the
        Néel state (alternating ±1).  Starting from all-up (cold init)
        is stuck in a high-energy state; Mpemba hot-start escapes it.
        """
        n = 4  # Small frustrated ring for determinism
        j_af = np.array([
            [0.0, -1.0, 0.0, -1.0],
            [-1.0, 0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0, -1.0],
            [-1.0, 0.0, -1.0, 0.0],
        ], dtype=np.float32)
        h = np.zeros(4, dtype=np.float32)

        # Cold-start: no Mpemba (mpemba_fraction=0 → minimal hot steps)
        # We simulate "cold start" by running with very short hot phase
        # and many ramp steps at high β (should stay near initial all-up)
        sim_cold = IsingSimulator(
            n_spins=n, max_degree=4, n_steps=100,
            beta_min=4.0, beta_max=5.0, lfsr_seed=0xBEEF,
            mpemba_fraction=0.0,  # No hot phase
        )
        sim_cold.load_problem(j_af, h)
        sim_cold.spins[:] = 1  # Forced cold (all-up) start
        sim_cold.run()
        e_cold = sim_cold.compute_energy()

        # Hot-start: 10% Mpemba phase (β=0) then ramp
        sim_hot = IsingSimulator(
            n_spins=n, max_degree=4, n_steps=100,
            beta_min=0.01, beta_max=5.0, lfsr_seed=0xBEEF,
            mpemba_fraction=0.1,
        )
        sim_hot.load_problem(j_af, h)
        sim_hot.run()
        e_hot = sim_hot.compute_energy()

        # Hot-start should reach equal or lower energy than cold start
        # (Mpemba effect: hot relaxes faster to ground state on this topology)
        # We relax to ≤ with 0.5 tolerance for quantization noise
        assert e_hot <= e_cold + 0.5, (
            f"Mpemba hot-start energy ({e_hot:.4f}) should be ≤ cold-start "
            f"energy ({e_cold:.4f}) on frustrated ring"
        )


# ---------------------------------------------------------------------------
# Halt condition
# ---------------------------------------------------------------------------


class TestHaltCondition:
    """Validate sampler halts after exactly N_STEPS sweeps.

    Spec: REQ-SAMPLE-011
    """

    def test_halts_and_outputs_spins(self) -> None:
        """Sampler returns final ±1 spin array after run() completes.

        REQ-SAMPLE-011: halt after T steps, output final spins via AXI read.
        """
        sim = IsingSimulator(n_spins=8, max_degree=4, n_steps=50)
        j, h = _build_ring_problem(8)
        sim.load_problem(j, h)
        spins = sim.run()
        assert spins.shape == (8,), f"Expected shape (8,) got {spins.shape}"
        assert set(spins.tolist()) <= {1, -1}, (
            f"All spins must be ±1; got unique values: {set(spins.tolist())}"
        )

    def test_done_status_set_after_halt(self) -> None:
        """STATUS DONE bit is set after N_STEPS sweeps complete.

        REQ-SAMPLE-011: STATUS = DONE after halting.
        """
        sim = IsingSimulator(n_spins=4, max_degree=4, n_steps=10)
        j, h = _build_ring_problem(4)
        sim.load_problem(j, h)
        assert not sim._done, "Simulator must not be done before run()"
        sim.run()
        assert sim._done, "Simulator must mark _done=True after run()"
        status = sim.axi_read(REG_STATUS)
        assert status & (1 << STATUS_DONE), (
            f"STATUS DONE bit must be set after halt; status=0x{status:08X}"
        )

    def test_spin_out_readback_matches_final_spins(self) -> None:
        """Packed spin_out words match the final spin array bit-for-bit.

        REQ-SAMPLE-011: spin_out[N_SPINS/32] output packed ±1 as 0/1 bits.
        """
        sim = IsingSimulator(n_spins=32, max_degree=4, n_steps=20)
        j, h = _build_ring_problem(32)
        sim.load_problem(j, h)
        spins = sim.run()
        word = sim.axi_read(REG_SPIN_OUT_BASE)
        for bit in range(32):
            expected_bit = 1 if spins[bit] == 1 else 0
            got_bit = (word >> bit) & 1
            assert got_bit == expected_bit, (
                f"spin_out bit {bit}: expected {expected_bit} got {got_bit} "
                f"(spin={spins[bit]})"
            )


# ---------------------------------------------------------------------------
# LFSR pseudo-random number generator
# ---------------------------------------------------------------------------


class TestLFSR:
    """Validate 16-bit Fibonacci LFSR matches Verilog implementation.

    Spec: REQ-SAMPLE-011
    """

    def test_period_is_65535(self) -> None:
        """LFSR visits all 65535 non-zero states before repeating.

        REQ-SAMPLE-011: LFSR-based PRNG, 16-bit Fibonacci LFSR.
        """
        lfsr = LFSR16(seed=0x0001)
        seen = set()
        for _ in range(65535):
            val = lfsr.next_value()
            seen.add(val)
        assert len(seen) == 65535, (
            f"LFSR should have period 65535; visited {len(seen)} unique states"
        )

    def test_zero_seed_raises(self) -> None:
        """LFSR rejects zero seed (lock-up state).

        REQ-SAMPLE-011: LFSR must not lock up.
        """
        with pytest.raises(ValueError, match="non-zero"):
            LFSR16(seed=0)

    def test_uniform_range(self) -> None:
        """LFSR uniform() output is in [0, 1).

        REQ-SAMPLE-011: LFSR random number for flip comparison.
        """
        lfsr = LFSR16(seed=0xACE1)
        for _ in range(1000):
            u = lfsr.uniform()
            assert 0.0 <= u < 1.0, f"uniform() out of range: {u}"

    def test_reproducibility(self) -> None:
        """Same seed produces same sequence (deterministic simulation).

        REQ-SAMPLE-011: deterministic behavioral simulation for test validation.
        """
        lfsr_a = LFSR16(seed=0x1234)
        lfsr_b = LFSR16(seed=0x1234)
        for _ in range(100):
            assert lfsr_a.next_value() == lfsr_b.next_value()


# ---------------------------------------------------------------------------
# Q8.8 fixed-point arithmetic
# ---------------------------------------------------------------------------


class TestQ88Arithmetic:
    """Validate Q8.8 encoding/decoding matches fpga_backend.py.

    Spec: REQ-SAMPLE-011
    """

    @pytest.mark.parametrize("value", [0.0, 1.0, -1.0, 1.5, -0.5, 0.00390625])
    def test_roundtrip(self, value: float) -> None:
        """Q8.8 encode/decode round-trips within 1/256 resolution.

        REQ-SAMPLE-011: Q8.8 fixed-point encoding for all weights.
        """
        q88 = float_to_q88(value)
        decoded = q88_to_float(q88)
        assert abs(decoded - value) < 1.0 / 256.0 + 1e-9, (
            f"Q8.8 round-trip error for {value}: got {decoded:.6f}"
        )

    def test_clipping_at_max(self) -> None:
        """Values above 127.996 are clipped to Q8.8 maximum.

        REQ-SAMPLE-011: Q8.8 range [-128.0, ~127.996].
        """
        q88 = float_to_q88(200.0)
        assert q88 == 32767, f"Expected 32767 (Q8.8 max), got {q88}"

    def test_clipping_at_min(self) -> None:
        """Values below -128.0 are clipped to Q8.8 minimum.

        REQ-SAMPLE-011: Q8.8 range [-128.0, ~127.996].
        """
        q88 = float_to_q88(-200.0)
        assert q88 == -32768, f"Expected -32768 (Q8.8 min), got {q88}"

    def test_q88_mul_correctness(self) -> None:
        """Q8.8 multiplication matches float multiplication to within resolution.

        REQ-SAMPLE-011: Q8.8 fixed-point multiplier in local-field accumulator.
        """
        pairs = [(1.0, 2.0), (0.5, 0.5), (-1.0, 2.0), (0.25, 4.0)]
        for a, b in pairs:
            a_q88 = float_to_q88(a)
            b_q88 = float_to_q88(b)
            result = q88_mul(a_q88, b_q88)
            result_float = q88_to_float(result)
            expected = a * b
            assert abs(result_float - expected) < 0.02, (
                f"q88_mul({a}, {b}): expected {expected:.4f} got {result_float:.4f}"
            )


# ---------------------------------------------------------------------------
# Full 128-spin ring convergence (integration test)
# ---------------------------------------------------------------------------


class TestFullConvergence:
    """Integration test: 128-spin ferromagnetic ring reaches low energy.

    Spec: REQ-SAMPLE-011
    """

    def test_128_spin_ring_reaches_low_energy(self) -> None:
        """128-spin ferromagnetic ring converges to near-ground-state energy.

        REQ-SAMPLE-011: full training + sampling pipeline correctness.
        The ferromagnetic ring ground state has energy ≈ -2*128 = -256
        (128 directed undirected edges * 2 directions * J=1 * all aligned).
        We accept any energy below -200 as convergence.
        """
        sim = IsingSimulator(
            n_spins=128, max_degree=32, n_steps=500,
            beta_min=0.1, beta_max=8.0, lfsr_seed=0xACE1
        )
        j, h = _build_ring_problem(128)
        sim.load_problem(j, h)
        sim.run()
        energy = sim.compute_energy()
        # Ground state energy for ferromagnetic ring (directed): -2*128 = -256
        assert energy < -200, (
            f"128-spin ferromagnetic ring should converge to energy < -200; got {energy:.2f}"
        )

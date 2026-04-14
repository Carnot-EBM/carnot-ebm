"""Tests for FpgaBackend: quantum-inspired sparse Ising sampler.

Spec coverage:
    REQ-SAMPLE-009, SCENARIO-SAMPLE-018, SCENARIO-SAMPLE-019
"""

from __future__ import annotations

import math

import numpy as np
from carnot.samplers.backend import SamplerBackend, get_backend
from carnot.samplers.fpga_backend import (
    FpgaBackend,
    _apply_lagrangian_penalty,
    quantize_to_q88,
    quantum_annealing_schedule,
    serialize_to_axi,
    sparsify_coupling,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ferromagnetic_problem(n: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Small ferromagnetic Ising problem: ground state is all-ones."""
    biases = np.ones(n, dtype=np.float32) * 1.5
    couplings = np.ones((n, n), dtype=np.float32) * 0.4
    np.fill_diagonal(couplings, 0.0)
    return biases, couplings


# ---------------------------------------------------------------------------
# quantize_to_q88
# ---------------------------------------------------------------------------


class TestQuantizeToQ88:
    """REQ-SAMPLE-009: Q8.8 quantization matches Exp 228 register format."""

    def test_zero_maps_to_zero(self) -> None:
        """SCENARIO-SAMPLE-019: Zero value encodes as 0."""
        assert quantize_to_q88(np.array([0.0]))[0] == 0

    def test_one_maps_to_256(self) -> None:
        """SCENARIO-SAMPLE-019: 1.0 encodes as 256 (2^8)."""
        assert quantize_to_q88(np.array([1.0]))[0] == 256

    def test_half_maps_to_128(self) -> None:
        """SCENARIO-SAMPLE-019: 0.5 encodes as 128."""
        assert quantize_to_q88(np.array([0.5]))[0] == 128

    def test_negative_value(self) -> None:
        """SCENARIO-SAMPLE-019: -1.0 encodes as -256."""
        assert quantize_to_q88(np.array([-1.0]))[0] == -256

    def test_round_trip(self) -> None:
        """SCENARIO-SAMPLE-019: Quantize then divide by 256 recovers original."""
        values = np.array([0.0, 0.5, 1.0, -0.5, -1.0, 3.14, -7.25], dtype=np.float32)
        q = quantize_to_q88(values)
        recovered = q.astype(np.float64) / 256.0
        # Maximum quantization error is half LSB = 0.5/256
        np.testing.assert_allclose(recovered, values, atol=1.0 / 256)

    def test_clipping_upper(self) -> None:
        """SCENARIO-SAMPLE-019: Values above 127.996 are clipped to 32767."""
        q = quantize_to_q88(np.array([200.0]))
        assert q[0] == 32767

    def test_clipping_lower(self) -> None:
        """SCENARIO-SAMPLE-019: Values below -128.0 are clipped to -32768."""
        q = quantize_to_q88(np.array([-200.0]))
        assert q[0] == -32768

    def test_matrix_input(self) -> None:
        """SCENARIO-SAMPLE-019: Works on 2-D matrices preserving shape."""
        mat = np.array([[1.0, -0.5], [0.25, 0.0]], dtype=np.float32)
        q = quantize_to_q88(mat)
        assert q.shape == (2, 2)
        assert q.dtype == np.int16

    def test_output_dtype_is_int16(self) -> None:
        """SCENARIO-SAMPLE-019: Return dtype is always int16."""
        q = quantize_to_q88(np.zeros(5))
        assert q.dtype == np.int16


# ---------------------------------------------------------------------------
# sparsify_coupling
# ---------------------------------------------------------------------------


class TestSparsifyCoupling:
    """REQ-SAMPLE-009: Sparsification keeps top-K couplings per spin."""

    def test_no_pruning_below_max_degree(self) -> None:
        """SCENARIO-SAMPLE-019: Dense matrix with ≤ max_degree neighbours unchanged."""
        coupling = np.array(
            [[0.0, 0.5, 0.3], [0.5, 0.0, 0.1], [0.3, 0.1, 0.0]], dtype=np.float32
        )
        result = sparsify_coupling(coupling, max_degree=10)
        np.testing.assert_array_equal(result, coupling)

    def test_diagonal_forced_zero(self) -> None:
        """SCENARIO-SAMPLE-019: Diagonal is always zeroed."""
        coupling = np.eye(5, dtype=np.float32) * 2.0
        result = sparsify_coupling(coupling, max_degree=10)
        np.testing.assert_array_equal(np.diag(result), np.zeros(5))

    def test_top_k_kept_by_magnitude(self) -> None:
        """SCENARIO-SAMPLE-019: Only top max_degree entries (by |value|) survive."""
        n = 10
        coupling = np.zeros((n, n), dtype=np.float32)
        for col in range(1, n):
            coupling[0, col] = float(col)  # magnitudes 1..9
        result = sparsify_coupling(coupling, max_degree=3)
        row = result[0]
        # Top 3 by magnitude in row 0 are columns 9, 8, 7
        assert row[9] != 0.0
        assert row[8] != 0.0
        assert row[7] != 0.0
        assert np.count_nonzero(row) == 3

    def test_max_degree_zero_zeroes_all(self) -> None:
        """SCENARIO-SAMPLE-019: max_degree=0 produces an all-zero matrix."""
        coupling = np.ones((5, 5), dtype=np.float32)
        result = sparsify_coupling(coupling, max_degree=0)
        assert np.count_nonzero(result) == 0

    def test_output_float32(self) -> None:
        """SCENARIO-SAMPLE-019: Output dtype is always float32."""
        coupling = np.ones((4, 4), dtype=np.float64)
        result = sparsify_coupling(coupling)
        assert result.dtype == np.float32

    def test_max_degree_le_32(self) -> None:
        """SCENARIO-SAMPLE-019: Default max_degree=32 matches KV260 hardware contract."""
        n = 50
        coupling = np.random.default_rng(0).random((n, n)).astype(np.float32)
        np.fill_diagonal(coupling, 0.0)
        result = sparsify_coupling(coupling, max_degree=32)
        for i in range(n):
            assert np.count_nonzero(result[i]) <= 32


# ---------------------------------------------------------------------------
# quantum_annealing_schedule
# ---------------------------------------------------------------------------


class TestQuantumAnnealingSchedule:
    """REQ-SAMPLE-009: Log-linear β-schedule from arXiv 2604.04606."""

    def test_t_zero_returns_beta_max(self) -> None:
        """SCENARIO-SAMPLE-018: n_steps=0 returns single-element list [beta_max]."""
        sched = quantum_annealing_schedule(0, beta_min=1.0, beta_max=10.0)
        assert sched == [10.0]

    def test_length_is_nsteps_plus_one(self) -> None:
        """SCENARIO-SAMPLE-018: Schedule has n_steps+1 entries."""
        sched = quantum_annealing_schedule(100, 0.1, 10.0)
        assert len(sched) == 101

    def test_starts_at_beta_min(self) -> None:
        """SCENARIO-SAMPLE-018: First element equals beta_min."""
        sched = quantum_annealing_schedule(50, beta_min=0.2, beta_max=8.0)
        assert math.isclose(sched[0], 0.2)

    def test_ends_at_beta_max(self) -> None:
        """SCENARIO-SAMPLE-018: Last element equals beta_max."""
        sched = quantum_annealing_schedule(50, beta_min=0.2, beta_max=8.0)
        assert math.isclose(sched[-1], 8.0)

    def test_monotone_increasing(self) -> None:
        """SCENARIO-SAMPLE-018: Schedule is non-decreasing throughout."""
        sched = quantum_annealing_schedule(100, beta_min=0.1, beta_max=20.0)
        for i in range(len(sched) - 1):
            assert sched[i + 1] >= sched[i]

    def test_midpoint_is_geometric_mean(self) -> None:
        """SCENARIO-SAMPLE-018: Log-linear midpoint equals sqrt(beta_min × beta_max)."""
        steps = 100
        beta_min, beta_max = 1.0, 100.0
        sched = quantum_annealing_schedule(steps, beta_min, beta_max)
        expected_mid = math.sqrt(beta_min * beta_max)  # 10.0
        assert math.isclose(sched[steps // 2], expected_mid, rel_tol=1e-6)

    def test_log_linear_property(self) -> None:
        """SCENARIO-SAMPLE-018: log(β) is linear in t (constant differences)."""
        steps = 200
        sched = quantum_annealing_schedule(steps, beta_min=0.1, beta_max=10.0)
        log_betas = [math.log(b) for b in sched]
        diffs = [log_betas[i + 1] - log_betas[i] for i in range(len(log_betas) - 1)]
        assert max(diffs) - min(diffs) < 1e-10


# ---------------------------------------------------------------------------
# serialize_to_axi
# ---------------------------------------------------------------------------


class TestSerializeToAxi:
    """REQ-SAMPLE-009: AXI serialization matches Exp 228 register-map format."""

    def test_keys_present(self) -> None:
        """SCENARIO-SAMPLE-018: Result contains all required register keys."""
        b, coupling = _ferromagnetic_problem(4)
        result = serialize_to_axi(coupling, b, beta=5.0)
        assert "SPIN_COUNT" in result
        assert "BETA_FINAL" in result
        assert "bias_words" in result
        assert "row_ptr" in result
        assert "edge_words" in result

    def test_spin_count(self) -> None:
        """SCENARIO-SAMPLE-018: SPIN_COUNT matches number of spins."""
        b, coupling = _ferromagnetic_problem(6)
        result = serialize_to_axi(coupling, b, beta=2.0)
        assert result["SPIN_COUNT"] == 6

    def test_beta_encoded_q88(self) -> None:
        """SCENARIO-SAMPLE-018: BETA_FINAL is Q8.8-encoded (beta * 256, rounded)."""
        b, coupling = _ferromagnetic_problem(4)
        beta = 2.0
        result = serialize_to_axi(coupling, b, beta=beta)
        expected = int(round(beta * 256))
        assert result["BETA_FINAL"] == expected

    def test_bias_words_count(self) -> None:
        """SCENARIO-SAMPLE-018: bias_words list has one entry per spin."""
        n = 5
        b, coupling = _ferromagnetic_problem(n)
        result = serialize_to_axi(coupling, b, beta=1.0)
        assert len(result["bias_words"]) == n

    def test_row_ptr_length(self) -> None:
        """SCENARIO-SAMPLE-018: row_ptr has n+1 entries (CSR format)."""
        n = 5
        b, coupling = _ferromagnetic_problem(n)
        result = serialize_to_axi(coupling, b, beta=1.0)
        assert len(result["row_ptr"]) == n + 1

    def test_all_values_are_ints(self) -> None:
        """SCENARIO-SAMPLE-018: All scalar values are Python ints (register-safe)."""
        b, coupling = _ferromagnetic_problem(4)
        result = serialize_to_axi(coupling, b, beta=3.0)
        assert isinstance(result["SPIN_COUNT"], int)
        assert isinstance(result["BETA_FINAL"], int)
        for w in result["bias_words"]:
            assert isinstance(w, int)


# ---------------------------------------------------------------------------
# _apply_lagrangian_penalty
# ---------------------------------------------------------------------------


class TestApplyLagrangianPenalty:
    """REQ-SAMPLE-009: LagONN penalty augments biases for frustrated spins."""

    def test_no_negative_couplings_unchanged(self) -> None:
        """SCENARIO-SAMPLE-019: All-positive coupling leaves biases unchanged."""
        coupling = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        h = np.array([0.5, 0.5], dtype=np.float32)
        _, h_out = _apply_lagrangian_penalty(coupling, h, strength=1.0)
        np.testing.assert_array_equal(h_out, h)

    def test_negative_couplings_increase_bias(self) -> None:
        """SCENARIO-SAMPLE-019: Negative coupling adds positive frustration penalty."""
        coupling = np.array([[0.0, -2.0], [-2.0, 0.0]], dtype=np.float32)
        h = np.array([0.0, 0.0], dtype=np.float32)
        _, h_out = _apply_lagrangian_penalty(coupling, h, strength=1.0)
        # frustration[i] = sum(min(coupling[i,:], 0)) = -2 for each spin
        # h_penalized[i] = h[i] - 1.0 * (-2.0) = +2.0
        np.testing.assert_allclose(h_out, [2.0, 2.0])

    def test_strength_scales_penalty(self) -> None:
        """SCENARIO-SAMPLE-019: Larger strength amplifies the penalty."""
        coupling = np.array([[0.0, -1.0], [-1.0, 0.0]], dtype=np.float32)
        h = np.zeros(2, dtype=np.float32)
        _, h_out_2 = _apply_lagrangian_penalty(coupling, h, strength=2.0)
        _, h_out_4 = _apply_lagrangian_penalty(coupling, h, strength=4.0)
        assert abs(h_out_4[0]) > abs(h_out_2[0])

    def test_coupling_unchanged(self) -> None:
        """SCENARIO-SAMPLE-019: Coupling matrix is returned unmodified."""
        coupling = np.array([[0.0, -1.0], [-1.0, 0.0]], dtype=np.float32)
        h = np.zeros(2, dtype=np.float32)
        coupling_out, _ = _apply_lagrangian_penalty(coupling, h, strength=1.0)
        np.testing.assert_array_equal(coupling_out, coupling)


# ---------------------------------------------------------------------------
# FpgaBackend
# ---------------------------------------------------------------------------


class TestFpgaBackendProtocol:
    """REQ-SAMPLE-009: FpgaBackend satisfies the SamplerBackend protocol."""

    def test_is_sampler_backend(self) -> None:
        """SCENARIO-SAMPLE-018: FpgaBackend conforms to SamplerBackend protocol."""
        assert isinstance(FpgaBackend(), SamplerBackend)

    def test_has_backend_name_property(self) -> None:
        """SCENARIO-SAMPLE-018: backend_name is accessible as a property."""
        backend = FpgaBackend()
        assert isinstance(backend.backend_name, str)

    def test_backend_name_without_env_var(self, monkeypatch) -> None:
        """SCENARIO-SAMPLE-018: Without CARNOT_KV260_BITFILE → 'fpga_cpu_fallback'."""
        monkeypatch.delenv("CARNOT_KV260_BITFILE", raising=False)
        assert FpgaBackend().backend_name == "fpga_cpu_fallback"

    def test_backend_name_with_env_var(self, monkeypatch) -> None:
        """SCENARIO-SAMPLE-018: With CARNOT_KV260_BITFILE set → 'fpga'."""
        monkeypatch.setenv("CARNOT_KV260_BITFILE", "/fake/path.bit")
        assert FpgaBackend().backend_name == "fpga"


class TestFpgaBackendMinimizeEnergy:
    """REQ-SAMPLE-009: minimize_energy returns correct shape and dtype."""

    def test_shape_and_dtype(self, monkeypatch) -> None:
        """SCENARIO-SAMPLE-018: minimize_energy → (n_samples, n_spins) bool array."""
        monkeypatch.delenv("CARNOT_KV260_BITFILE", raising=False)
        b, coupling = _ferromagnetic_problem(8)
        backend = FpgaBackend(seed=0)
        samples = backend.minimize_energy(b, coupling, n_samples=4, n_steps=50, beta=5.0)
        assert samples.shape == (4, 8)
        assert samples.dtype == bool

    def test_ferromagnet_biased_high(self, monkeypatch) -> None:
        """SCENARIO-SAMPLE-018: At low temperature, ferromagnet samples mostly-ones."""
        monkeypatch.delenv("CARNOT_KV260_BITFILE", raising=False)
        b, coupling = _ferromagnetic_problem(8)
        backend = FpgaBackend(seed=42, beta_min=1.0, beta_max=15.0)
        samples = backend.minimize_energy(b, coupling, n_samples=20, n_steps=500, beta=15.0)
        assert samples.mean() > 0.6

    def test_with_lagrangian_penalty(self, monkeypatch) -> None:
        """SCENARIO-SAMPLE-019: use_lagrangian_penalty=True completes without error."""
        monkeypatch.delenv("CARNOT_KV260_BITFILE", raising=False)
        b, coupling = _ferromagnetic_problem(6)
        backend = FpgaBackend(
            seed=0, use_lagrangian_penalty=True, lagrangian_penalty_strength=0.5
        )
        samples = backend.minimize_energy(b, coupling, n_samples=5, n_steps=100, beta=5.0)
        assert samples.shape == (5, 6)
        assert samples.dtype == bool


class TestFpgaBackendSample:
    """REQ-SAMPLE-009: sample() returns correct shape and dtype."""

    def test_shape_and_dtype(self, monkeypatch) -> None:
        """SCENARIO-SAMPLE-018: sample → (n_samples, n_spins) bool array."""
        monkeypatch.delenv("CARNOT_KV260_BITFILE", raising=False)
        b, coupling = _ferromagnetic_problem(8)
        backend = FpgaBackend(seed=0)
        samples = backend.sample(b, coupling, n_samples=5, config={"beta": 5.0})
        assert samples.shape == (5, 8)
        assert samples.dtype == bool

    def test_config_n_steps_respected(self, monkeypatch) -> None:
        """SCENARIO-SAMPLE-018: n_steps from config is used (no crash)."""
        monkeypatch.delenv("CARNOT_KV260_BITFILE", raising=False)
        b, coupling = _ferromagnetic_problem(6)
        backend = FpgaBackend(seed=1)
        samples = backend.sample(b, coupling, n_samples=3, config={"n_steps": 200})
        assert samples.shape == (3, 6)

    def test_sample_with_lagrangian_penalty(self, monkeypatch) -> None:
        """SCENARIO-SAMPLE-019: sample with LagONN penalty returns valid output."""
        monkeypatch.delenv("CARNOT_KV260_BITFILE", raising=False)
        b, coupling = _ferromagnetic_problem(6)
        backend = FpgaBackend(seed=2, use_lagrangian_penalty=True)
        samples = backend.sample(b, coupling, n_samples=4, config={})
        assert samples.shape == (4, 6)


class TestFpgaBackendDispatch:
    """REQ-SAMPLE-009: dispatch routes correctly based on CARNOT_KV260_BITFILE."""

    def test_cpu_fallback_when_no_env_var(self, monkeypatch) -> None:
        """SCENARIO-SAMPLE-018: No env var → CPU fallback, valid output."""
        monkeypatch.delenv("CARNOT_KV260_BITFILE", raising=False)
        b, coupling = _ferromagnetic_problem(6)
        backend = FpgaBackend(seed=0)
        j_sp = sparsify_coupling(coupling)
        samples = backend.dispatch(j_sp, b, n_samples=4, n_steps=50)
        assert samples.shape == (4, 6)
        assert samples.dtype == bool

    def test_fpga_path_with_env_var_missing_file(self, monkeypatch) -> None:
        """SCENARIO-SAMPLE-018: Env var set but bitfile absent → FPGAIsingSampler fallback."""
        monkeypatch.setenv("CARNOT_KV260_BITFILE", "/nonexistent/fake.bit")
        b, coupling = _ferromagnetic_problem(6)
        backend = FpgaBackend(seed=0)
        j_sp = sparsify_coupling(coupling)
        # FPGAIsingSampler with allow_cpu_fallback=True handles missing bitfile gracefully
        samples = backend.dispatch(j_sp, b, n_samples=4, n_steps=50)
        assert samples.shape == (4, 6)
        assert samples.dtype == bool


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


class TestGetBackendFpga:
    """REQ-SAMPLE-009: get_backend('fpga') returns FpgaBackend."""

    def test_get_backend_fpga_returns_fpga_backend(self, monkeypatch) -> None:
        """SCENARIO-SAMPLE-018: get_backend('fpga') returns FpgaBackend instance."""
        monkeypatch.delenv("CARNOT_KV260_BITFILE", raising=False)
        backend = get_backend("fpga")
        assert isinstance(backend, FpgaBackend)

    def test_fpga_backend_satisfies_protocol(self, monkeypatch) -> None:
        """SCENARIO-SAMPLE-018: FpgaBackend from factory satisfies SamplerBackend."""
        monkeypatch.delenv("CARNOT_KV260_BITFILE", raising=False)
        backend = get_backend("fpga")
        assert isinstance(backend, SamplerBackend)

    def test_fpga_backend_name(self, monkeypatch) -> None:
        """SCENARIO-SAMPLE-018: Backend name is 'fpga_cpu_fallback' without env var."""
        monkeypatch.delenv("CARNOT_KV260_BITFILE", raising=False)
        backend = get_backend("fpga")
        assert backend.backend_name == "fpga_cpu_fallback"

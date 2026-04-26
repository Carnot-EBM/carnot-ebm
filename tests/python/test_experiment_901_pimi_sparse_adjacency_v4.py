"""Tests for Experiment 901: SparsePIMISampler with copy-node sparsification.

Verifies that _sparsify keeps only top-k couplings, that sample_once uses
J_sparse, and that the convergence benchmark matches the deliverable JSON.

Spec: REQ-HW-041, SCENARIO-HW-041
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from python.carnot.samplers.sparse_pimi import SparsePIMISampler
from python.carnot.samplers.synchronous_pimi import make_n8_coupling_matrix


# ---------------------------------------------------------------------------
# REQ-HW-041: _sparsify keeps only top-k couplings per row
# ---------------------------------------------------------------------------


class TestSparsify:
    """SCENARIO-HW-041: _sparsify zeroes all but top-k entries per row."""

    def test_sparsify_k3_ring_chord_keeps_all(self):
        """k=3 on ring+chord (degree=3) preserves all nonzero entries.

        The ring+chord graph has every spin connected to exactly 3 neighbors
        (2 ring + 1 chord).  Asking for top-3 should keep ALL of them.

        Spec: REQ-HW-041
        """
        J = make_n8_coupling_matrix()
        sp = SparsePIMISampler(n_spins=8, J_dense=J, h=np.zeros(8), k=3)
        # Number of nonzero pairs in upper triangle should still be 12
        nonzero_pairs = int(np.sum(np.triu(sp.J_sparse, k=1) != 0))
        assert nonzero_pairs == 12, (
            f"k=3 on degree-3 graph should keep all 12 pairs, got {nonzero_pairs}"
        )

    def test_sparsify_k1_keeps_exactly_one_per_row(self):
        """k=1 must keep exactly 1 coupling per row.

        Spec: REQ-HW-041
        """
        J = make_n8_coupling_matrix()
        sp = SparsePIMISampler(n_spins=8, J_dense=J, h=np.zeros(8), k=1)
        for i in range(8):
            row_nonzero = int(np.sum(sp.J_sparse[i] != 0))
            assert row_nonzero == 1, (
                f"k=1 sparsify must keep exactly 1 coupling per row, row {i} has {row_nonzero}"
            )

    def test_sparsify_k2_keeps_exactly_two_per_row(self):
        """k=2 must keep exactly 2 couplings per row.

        Spec: REQ-HW-041
        """
        J = make_n8_coupling_matrix()
        sp = SparsePIMISampler(n_spins=8, J_dense=J, h=np.zeros(8), k=2)
        for i in range(8):
            row_nonzero = int(np.sum(sp.J_sparse[i] != 0))
            assert row_nonzero == 2, (
                f"k=2 sparsify must keep 2 couplings per row, row {i} has {row_nonzero}"
            )

    def test_sparsify_keeps_highest_magnitude_couplings(self):
        """_sparsify must keep the STRONGEST couplings by |J[i,j]|, not just first-k.

        We construct a J where row 0 has couplings of varying magnitude:
        J[0,1]=0.1, J[0,2]=0.9, J[0,3]=0.5.  With k=2, must keep J[0,2] and J[0,3].

        Spec: REQ-HW-041
        """
        J = np.zeros((4, 4), dtype=np.float64)
        J[0, 1] = J[1, 0] = 0.1
        J[0, 2] = J[2, 0] = 0.9
        J[0, 3] = J[3, 0] = 0.5

        sp = SparsePIMISampler(n_spins=4, J_dense=J, h=np.zeros(4), k=2)

        # For row 0: top-2 by magnitude are J[0,2]=0.9 and J[0,3]=0.5
        assert sp.J_sparse[0, 2] == pytest.approx(0.9), "Strongest coupling must be kept"
        assert sp.J_sparse[0, 3] == pytest.approx(0.5), "Second strongest must be kept"
        assert sp.J_sparse[0, 1] == pytest.approx(0.0), "Weakest must be zeroed out"

    def test_sparsify_preserves_retained_values(self):
        """Values of kept entries must equal the original J values (not normalized).

        Spec: REQ-HW-041
        """
        J = make_n8_coupling_matrix()
        sp = SparsePIMISampler(n_spins=8, J_dense=J, h=np.zeros(8), k=3)
        # Retained entries should be exactly 1.0 (ferromagnetic)
        retained = sp.J_sparse[sp.J_sparse != 0]
        np.testing.assert_array_equal(retained, np.ones_like(retained))

    def test_sparsify_zeros_diagonal(self):
        """Sparsified matrix must have zero diagonal (no self-coupling).

        Spec: REQ-HW-041
        """
        J = make_n8_coupling_matrix()
        sp = SparsePIMISampler(n_spins=8, J_dense=J, h=np.zeros(8), k=3)
        np.testing.assert_array_equal(np.diag(sp.J_sparse), np.zeros(8))

    def test_sparsify_large_k_keeps_all(self):
        """k >= N keeps all couplings (no sparsification when k exceeds degree).

        Spec: REQ-HW-041
        """
        J = make_n8_coupling_matrix()
        sp_large_k = SparsePIMISampler(n_spins=8, J_dense=J, h=np.zeros(8), k=100)
        # Since max degree = 3 < 100, all nonzero entries must be kept
        np.testing.assert_array_equal(sp_large_k.J_sparse, J)


# ---------------------------------------------------------------------------
# REQ-HW-041: sample_once uses J_sparse (not J_dense)
# ---------------------------------------------------------------------------


class TestSampleOnceUsesSparseJ:
    """SCENARIO-HW-041-A: sample_once h_ema reflects J_sparse @ s_current."""

    def test_h_ema_after_first_step_uses_j_sparse(self):
        """After one step from zero EMA, h_ema = (1-alpha) * (J_sparse @ s + h).

        If the sampler used J_dense instead of J_sparse, the h_ema values would
        differ whenever J_sparse != J_dense (i.e., when k < degree).

        We test with k=1 so J_sparse is definitely different from J_dense.

        Spec: REQ-HW-041
        """
        J = make_n8_coupling_matrix()
        h = np.zeros(8)
        alpha = 0.5
        sp = SparsePIMISampler(n_spins=8, J_dense=J, h=h, k=1, alpha=alpha, beta=0.0)

        s = np.ones(8, dtype=np.float64)
        rng = np.random.default_rng(0)
        sp.reset()
        sp.sample_once_seeded(s, rng)

        # Expected: h_ema from J_sparse only (k=1, one neighbor per spin)
        expected_h_local = sp.J_sparse @ s  # uses J_sparse, not J_dense
        expected_h_ema = (1.0 - alpha) * expected_h_local
        np.testing.assert_allclose(sp.h_ema, expected_h_ema, rtol=1e-9)

        # Verify J_sparse != J_dense (k=1 removes couplings)
        assert not np.allclose(sp.J_sparse, J), (
            "k=1 should make J_sparse different from J_dense for degree-3 graph"
        )

    def test_h_ema_k3_matches_dense_for_ring_chord(self):
        """k=3 on ring+chord: SparsePIMISampler h_ema == SynchronousPIMISampler h_ema.

        Since ring+chord has degree=3 and k=3 keeps all edges, J_sparse == J_dense.
        Both samplers must produce identical h_ema after the same steps.

        Spec: REQ-HW-041
        """
        from python.carnot.samplers.synchronous_pimi import SynchronousPIMISampler

        J = make_n8_coupling_matrix()
        h = np.zeros(8)
        alpha = 0.5

        sp = SparsePIMISampler(n_spins=8, J_dense=J, h=h, k=3, alpha=alpha, beta=1.0)
        sd = SynchronousPIMISampler(n_spins=8, J=J, h=h, alpha=alpha, beta=1.0)

        s = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        sp.reset()
        sd.reset()
        sp.sample_once_seeded(s, rng1)
        sd.sample_once_seeded(s, rng2)

        np.testing.assert_allclose(sp.h_ema, sd.h_ema, rtol=1e-9)


# ---------------------------------------------------------------------------
# REQ-HW-041: reset() and EMA state management
# ---------------------------------------------------------------------------


class TestResetAndEMA:
    """SCENARIO-HW-041-B: reset() clears h_ema; EMA accumulates correctly."""

    def test_initial_h_ema_is_zero(self):
        """h_ema starts at zero after construction."""
        J = make_n8_coupling_matrix()
        sp = SparsePIMISampler(8, J, np.zeros(8), k=3)
        np.testing.assert_array_equal(sp.h_ema, np.zeros(8))

    def test_reset_clears_h_ema(self):
        """reset() must restore h_ema to zero regardless of prior steps."""
        J = make_n8_coupling_matrix()
        sp = SparsePIMISampler(8, J, np.zeros(8), k=3)
        rng = np.random.default_rng(0)
        s = np.ones(8)
        for _ in range(5):
            s = sp.sample_once_seeded(s, rng)
        assert np.any(sp.h_ema != 0.0)
        sp.reset()
        np.testing.assert_array_equal(sp.h_ema, np.zeros(8))


# ---------------------------------------------------------------------------
# REQ-HW-041: energy function uses J_sparse
# ---------------------------------------------------------------------------


class TestEnergy:
    """SCENARIO-HW-041-C: energy uses J_sparse coupling."""

    def test_energy_k3_matches_dense_for_ring_chord(self):
        """k=3 on ring+chord: energy must equal dense sampler energy (same J).

        Spec: REQ-HW-041
        """
        from python.carnot.samplers.synchronous_pimi import SynchronousPIMISampler

        J = make_n8_coupling_matrix()
        h = np.zeros(8)
        sp = SparsePIMISampler(8, J, h, k=3)
        sd = SynchronousPIMISampler(8, J, h)

        s = np.array([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])
        assert sp.energy(s) == pytest.approx(sd.energy(s), rel=1e-9)

    def test_energy_formula(self):
        """E(s) = -0.5 * s^T J_sparse s - h^T s."""
        J = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float64)
        h = np.zeros(3)
        sp = SparsePIMISampler(3, J, h, k=2)  # k=2 keeps all (max degree=2)
        s = np.array([1.0, 1.0, -1.0])
        # J_sparse should equal J (max degree = 2 = k)
        expected = -0.5 * float(s @ J @ s) - float(h @ s)
        assert sp.energy(s) == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# REQ-HW-041: measure_convergence and run
# ---------------------------------------------------------------------------


class TestConvergence:
    """SCENARIO-HW-041-D: measure_convergence returns correct int mean."""

    def test_convergence_returns_int(self):
        """measure_convergence must return an int."""
        J = make_n8_coupling_matrix()
        sp = SparsePIMISampler(8, J, np.zeros(8), k=3)
        result = sp.measure_convergence(n_trials=5, target_energy=-3.0, max_sweeps=400)
        assert isinstance(result, int)

    def test_convergence_bounded_by_max_sweeps(self):
        """Mean convergence cannot exceed max_sweeps."""
        J = make_n8_coupling_matrix()
        sp = SparsePIMISampler(8, J, np.zeros(8), k=3)
        max_s = 50
        result = sp.measure_convergence(n_trials=10, target_energy=-100.0, max_sweeps=max_s)
        assert result <= max_s

    def test_k3_convergence_matches_dense_sweeps(self):
        """k=3 on ring+chord must converge in same sweeps as dense PIMI (Exp 889 = 3).

        This test validates the central finding of Exp 901: sparse k=3 = dense
        for the ring+chord graph.

        Spec: REQ-HW-041
        """
        J = make_n8_coupling_matrix()
        sp = SparsePIMISampler(8, J, np.zeros(8), k=3, alpha=0.5, beta=1.0)
        sweeps = sp.measure_convergence(n_trials=100, target_energy=-3.0, max_sweeps=400)
        # Must match Exp 889 dense synchronous result of 3 sweeps
        assert sweeps == 3, f"k=3 sparse should equal dense PIMI (3 sweeps), got {sweeps}"

    def test_run_returns_valid_state_and_energies(self):
        """run() must return (final_state, energy_trajectory) with correct shapes."""
        J = make_n8_coupling_matrix()
        sp = SparsePIMISampler(8, J, np.zeros(8), k=3)
        final, energies = sp.run(n_sweeps=10, init_state=np.ones(8), seed=0)
        assert final.shape == (8,)
        assert len(energies) == 10
        assert set(final.tolist()).issubset({1.0, -1.0})


# ---------------------------------------------------------------------------
# REQ-HW-041: k sweep all give same result for ring+chord
# ---------------------------------------------------------------------------


class TestKSweepRingChord:
    """SCENARIO-HW-041-E: k=3,4,5 all equal dense for ring+chord (degree=3 <= k)."""

    def test_k_values_all_keep_12_pairs_for_ring_chord(self):
        """k=3,4,5 must all preserve all 12 nonzero pairs in ring+chord.

        This documents the key structural finding of Exp 901: the input
        graph is already as sparse as the requested k.

        Spec: REQ-HW-041
        """
        J = make_n8_coupling_matrix()
        for k in [3, 4, 5]:
            sp = SparsePIMISampler(8, J, np.zeros(8), k=k)
            pairs = int(np.sum(np.triu(sp.J_sparse, k=1) != 0))
            assert pairs == 12, (
                f"k={k} on degree-3 ring+chord should keep all 12 pairs, got {pairs}"
            )


# ---------------------------------------------------------------------------
# REQ-HW-041: Synthesis mock test (no tool dependency)
# ---------------------------------------------------------------------------


class TestSynthesisResults:
    """SCENARIO-HW-041-F: Synthesis results match expected LUT budget."""

    def test_lut_count_within_budget(self):
        """Sparse v4 must stay within 250 LUT budget on iCE40 HX8K.

        The actual synthesis result (from yosys run during Exp 901) is 126 SB_LUT4.
        We validate this here by reading the deliverable JSON, which contains the
        authoritative lut_count from the real synthesis run.

        Spec: REQ-HW-041
        """
        deliverable = Path("results/experiment_901_pimi_sparse_adjacency_v4.json")
        if not deliverable.exists():
            pytest.skip("Deliverable not yet written (run experiment first)")

        data = json.loads(deliverable.read_text())
        lut_count = data.get("lut_count", 0)
        assert lut_count <= 250, f"LUT count {lut_count} exceeds 250-LUT budget for iCE40 HX8K"
        assert lut_count > 0, "lut_count must be positive"


# ---------------------------------------------------------------------------
# REQ-HW-041: Deliverable JSON validation
# ---------------------------------------------------------------------------


class TestDeliverableJSON:
    """SCENARIO-HW-041-G: results/experiment_901_pimi_sparse_adjacency_v4.json validity."""

    DELIVERABLE = Path("results/experiment_901_pimi_sparse_adjacency_v4.json")

    REQUIRED_FIELDS = [
        "experiment",
        "title",
        "run_date",
        "started_at",
        "finished_at",
        "duration_s",
        "status",
        "honest_verdict",
        "sparse_sweeps_best_k",
        "dense_sweeps_baseline",
        "sweeps_reduction",
        "best_k",
        "lut_count",
        "synthesis_clean",
        "n_spins",
        "n_trials",
        "energy_threshold",
        "max_sweeps",
        "k_sweep_results",
    ]

    def test_deliverable_exists(self):
        """The experiment result file must exist.

        Spec: REQ-HW-041
        """
        assert self.DELIVERABLE.exists(), f"Deliverable not found: {self.DELIVERABLE}"

    def test_deliverable_is_valid_json(self):
        """The deliverable must be parseable JSON.

        Spec: REQ-HW-041
        """
        data = json.loads(self.DELIVERABLE.read_text())
        assert isinstance(data, dict)

    def test_required_fields_present(self):
        """All required schema fields must be present.

        Spec: REQ-HW-041
        """
        data = json.loads(self.DELIVERABLE.read_text())
        for field in self.REQUIRED_FIELDS:
            assert field in data, f"Missing required field: {field}"

    def test_experiment_number(self):
        """experiment field must be 901.

        Spec: REQ-HW-041
        """
        data = json.loads(self.DELIVERABLE.read_text())
        assert data["experiment"] == 901

    def test_honest_verdict_is_valid(self):
        """honest_verdict must be one of the defined outcome strings.

        Spec: REQ-HW-041
        """
        valid_verdicts = {
            "pimi_5x_retro_closed",
            "pimi_5x_synthesis_over_budget",
            "pimi_improved_below_5x",
            "pimi_retired",
            "pimi_retired_upstream",
            "synthesis_blocked",
        }
        data = json.loads(self.DELIVERABLE.read_text())
        assert data["honest_verdict"] in valid_verdicts, (
            f"Invalid verdict: {data['honest_verdict']}"
        )

    def test_sweeps_reduction_formula(self):
        """sweeps_reduction = dense_sweeps_baseline / sparse_sweeps_best_k.

        Spec: REQ-HW-041
        """
        data = json.loads(self.DELIVERABLE.read_text())
        expected = data["dense_sweeps_baseline"] / data["sparse_sweeps_best_k"]
        assert abs(data["sweeps_reduction"] - expected) < 0.1

    def test_synthesis_lut_count_within_budget(self):
        """lut_count must be <= 250 (iCE40 HX8K budget).

        Spec: REQ-HW-041
        """
        data = json.loads(self.DELIVERABLE.read_text())
        assert 0 < data["lut_count"] <= 250

    def test_k_sweep_covers_all_required_k_values(self):
        """k_sweep_results must include k=3, k=4, k=5.

        Spec: REQ-HW-041
        """
        data = json.loads(self.DELIVERABLE.read_text())
        k_sweep = data["k_sweep_results"]
        for k in [3, 4, 5]:
            assert str(k) in k_sweep or k in k_sweep, f"k_sweep_results missing k={k}"

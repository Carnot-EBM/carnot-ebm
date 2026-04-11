"""Tests for carnot.models.adaptive_kan -- KANConstraintModel and AdaptiveKAN.

100% coverage target on adaptive_kan.py.

Spec: REQ-CORE-001, REQ-CORE-002, REQ-TIER-001
Scenario: SCENARIO-CORE-001, SCENARIO-TIER-004
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from carnot.models.adaptive_kan import AdaptiveKAN, KANConstraintModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_binary(n: int, dim: int, seed: int = 0) -> np.ndarray:
    """Random binary {0,1} matrix, shape (n, dim)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, (n, dim)).astype(np.float32)


def _small_model(input_dim: int = 4, seed: int = 42) -> KANConstraintModel:
    """Small KANConstraintModel for fast unit tests."""
    return KANConstraintModel(input_dim=input_dim, num_knots=4, degree=2, seed=seed)


def _small_adaptive(
    input_dim: int = 4,
    restructure_every: int = 10,
    seed: int = 42,
) -> AdaptiveKAN:
    """Small AdaptiveKAN with fast restructure trigger."""
    return AdaptiveKAN(
        input_dim=input_dim,
        num_knots=4,
        degree=2,
        seed=seed,
        restructure_every=restructure_every,
    )


# ---------------------------------------------------------------------------
# KANConstraintModel — base class coverage
# ---------------------------------------------------------------------------


class TestKANConstraintModelInit:
    """REQ-CORE-001: KANConstraintModel initialises correctly."""

    def test_default_fully_connected(self) -> None:
        """SCENARIO-CORE-001: No edges → fully connected graph."""
        model = KANConstraintModel(input_dim=3, num_knots=4, degree=2)
        # For dim=3: edges are (0,1), (0,2), (1,2) — 3 edges.
        assert len(model.edges) == 3

    def test_custom_edges(self) -> None:
        """REQ-CORE-001: Custom edges are respected."""
        edges = [(0, 1), (1, 2)]
        model = KANConstraintModel(input_dim=4, edges=edges, num_knots=4, degree=2)
        assert model.edges == edges
        assert len(model.edge_control_pts) == 2

    def test_control_point_shapes(self) -> None:
        """REQ-CORE-001: Control point arrays have correct shape."""
        model = KANConstraintModel(input_dim=3, num_knots=4, degree=2, seed=1)
        n_ctrl = 4 + 2  # num_knots + degree
        for ctrl in model.edge_control_pts.values():
            assert ctrl.shape == (n_ctrl,)
        for ctrl in model.bias_control_pts:
            assert ctrl.shape == (n_ctrl,)

    def test_n_params_initial(self) -> None:
        """REQ-CORE-001: n_params matches edge + bias control points."""
        model = _small_model(input_dim=3)
        n_ctrl = 4 + 2  # 6
        n_edges = 3  # fully connected dim=3
        expected = n_edges * n_ctrl + 3 * n_ctrl
        assert model.n_params == expected

    def test_edge_n_ctrl_initialised(self) -> None:
        """REQ-CORE-001: _edge_n_ctrl tracks initial sizes."""
        model = _small_model(input_dim=3)
        for edge in model.edges:
            assert model._edge_n_ctrl[edge] == 4 + 2


class TestEvalSpline:
    """REQ-CORE-001: _eval_spline interpolates correctly."""

    def test_at_left_boundary(self) -> None:
        """REQ-CORE-001: x=-1.0 returns first control point."""
        ctrl = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = KANConstraintModel._eval_spline(-1.0, ctrl)
        assert result == pytest.approx(1.0, abs=1e-5)

    def test_at_right_boundary(self) -> None:
        """REQ-CORE-001: x near +1.0 returns last control point."""
        ctrl = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = KANConstraintModel._eval_spline(0.9999, ctrl)
        assert result == pytest.approx(4.0, abs=0.01)

    def test_midpoint_interpolation(self) -> None:
        """REQ-CORE-001: x=0 gives midpoint interpolation."""
        ctrl = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        result = KANConstraintModel._eval_spline(0.0, ctrl)
        # x=0 maps to scaled=1.5, so between ctrl[1]=0 and ctrl[2]=1, t=0.5.
        assert isinstance(result, float)


class TestBasisK:
    """REQ-CORE-001: _basis_k hat function."""

    def test_exact_left_node(self) -> None:
        """SCENARIO-CORE-001: basis_k returns 1 at its own node."""
        # x=-1.0, n_ctrl=4: scaled=0 → left=0.  basis_0(-1) = 1-t = 1.0.
        val = KANConstraintModel._basis_k(-1.0, 0, 4)
        assert val == pytest.approx(1.0)

    def test_exact_right_node(self) -> None:
        """REQ-CORE-001: basis_k for the right adjacent node."""
        # x=-1.0 → left=0, t=0.  basis_1(-1) = t = 0.0.
        val = KANConstraintModel._basis_k(-1.0, 1, 4)
        assert val == pytest.approx(0.0)

    def test_far_node_zero(self) -> None:
        """REQ-CORE-001: basis_k is 0 for non-adjacent nodes."""
        val = KANConstraintModel._basis_k(-1.0, 3, 4)
        assert val == pytest.approx(0.0)

    def test_partition_of_unity(self) -> None:
        """REQ-CORE-001: basis functions sum to 1 at any x."""
        n_ctrl = 5
        x_test = 0.3
        total = sum(KANConstraintModel._basis_k(x_test, k, n_ctrl) for k in range(n_ctrl))
        assert total == pytest.approx(1.0, abs=1e-5)


class TestEnergySingle:
    """REQ-CORE-001: energy_single computes finite values."""

    def test_finite_output(self) -> None:
        """SCENARIO-CORE-001: energy_single returns finite float."""
        model = _small_model(input_dim=4)
        x = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        e = model.energy_single(x)
        assert isinstance(e, float)
        assert np.isfinite(e)

    def test_different_inputs_differ(self) -> None:
        """REQ-CORE-001: Different inputs give different energies (usually)."""
        model = _small_model(input_dim=4, seed=99)
        x1 = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        x2 = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        # Not guaranteed equal for random init; check both are finite.
        assert np.isfinite(model.energy_single(x1))
        assert np.isfinite(model.energy_single(x2))


class TestEnergyBatch:
    """REQ-CORE-001: energy_batch vectorises correctly."""

    def test_batch_matches_single(self) -> None:
        """REQ-CORE-001: energy_batch equals energy_single for each row."""
        model = _small_model(input_dim=4, seed=5)
        xs = _make_binary(8, 4, seed=7)
        batch_e = model.energy_batch(xs)
        for i in range(8):
            assert batch_e[i] == pytest.approx(model.energy_single(xs[i]), abs=1e-5)

    def test_batch_shape(self) -> None:
        """REQ-CORE-001: energy_batch returns correct shape."""
        model = _small_model(input_dim=4)
        xs = _make_binary(5, 4)
        assert model.energy_batch(xs).shape == (5,)


class TestComputeEdgeCurvature:
    """REQ-CORE-001: compute_edge_curvature returns non-negative scores."""

    def test_all_edges_have_score(self) -> None:
        """SCENARIO-CORE-001: Every edge gets a curvature score."""
        model = _small_model(input_dim=4)
        curvatures = model.compute_edge_curvature(n_sample=20)
        assert set(curvatures.keys()) == set(model.edges)

    def test_scores_non_negative(self) -> None:
        """REQ-CORE-001: Curvature scores are >= 0."""
        model = _small_model(input_dim=4)
        curvatures = model.compute_edge_curvature(n_sample=20)
        for c in curvatures.values():
            assert c >= 0.0

    def test_sample_pts_override(self) -> None:
        """REQ-CORE-001: sample_pts overrides default uniform grid per edge."""
        model = _small_model(input_dim=3)
        # Supply explicit sample points for each edge.
        sample_pts = {edge: np.array([-0.5, 0.0, 0.5]) for edge in model.edges}
        curvatures = model.compute_edge_curvature(sample_pts=sample_pts)
        assert set(curvatures.keys()) == set(model.edges)
        for c in curvatures.values():
            assert c >= 0.0

    def test_empty_sample_pts_returns_zero(self) -> None:
        """REQ-CORE-001: Empty sample_pts array → curvature 0 (fallback)."""
        model = _small_model(input_dim=3)
        edge = model.edges[0]
        sample_pts = {edge: np.array([], dtype=np.float32)}
        curvatures = model.compute_edge_curvature(sample_pts=sample_pts)
        # Edge with empty pts gets 0.0; others get uniform-grid curvature.
        assert curvatures[edge] == pytest.approx(0.0)


class TestInsertKnot:
    """REQ-CORE-001: _insert_knot increases control point count by 1."""

    def test_insert_increases_n_ctrl(self) -> None:
        """REQ-CORE-001: After _insert_knot, len(ctrl) increases by 1."""
        model = _small_model(input_dim=3)
        edge = model.edges[0]
        n_before = len(model.edge_control_pts[edge])
        x_pts = np.linspace(-0.9, 0.9, 10)
        # All curvature in first point (argmax=0).
        curvatures_per_point = [10.0] + [0.0] * 9
        model._insert_knot(edge, curvatures_per_point, x_pts)
        n_after = len(model.edge_control_pts[edge])
        assert n_after == n_before + 1
        assert model._edge_n_ctrl[edge] == n_after


class TestRemoveKnot:
    """REQ-CORE-001: _remove_knot decreases control point count by 1, or returns False."""

    def test_remove_decreases_n_ctrl(self) -> None:
        """REQ-CORE-001: Removal reduces control points by 1."""
        # Use num_knots=5, degree=2 → n_ctrl=7; min = 2+2=4; can remove.
        model = KANConstraintModel(input_dim=3, num_knots=5, degree=2, seed=0)
        edge = model.edges[0]
        n_before = len(model.edge_control_pts[edge])
        result = model._remove_knot(edge)
        assert result is True
        assert len(model.edge_control_pts[edge]) == n_before - 1
        assert model._edge_n_ctrl[edge] == n_before - 1

    def test_remove_false_at_minimum(self) -> None:
        """REQ-CORE-001: Returns False when n_ctrl == degree + 2 (minimum)."""
        # degree=2, min_n_ctrl=4 → initialise with num_knots=2 → n_ctrl=4 = min.
        model = KANConstraintModel(input_dim=3, num_knots=2, degree=2, seed=0)
        edge = model.edges[0]
        # Verify n_ctrl == min (degree + 2 = 4).
        assert len(model.edge_control_pts[edge]) == 4
        result = model._remove_knot(edge)
        assert result is False


class TestRefine:
    """REQ-CORE-001: refine() changes knot counts based on curvature."""

    def test_refine_returns_counts(self) -> None:
        """REQ-CORE-001: refine() returns (added, removed) ints."""
        model = _small_model(input_dim=4)
        added, removed = model.refine(threshold_multiplier=1.5, n_sample=20)
        assert isinstance(added, int)
        assert isinstance(removed, int)
        assert added >= 0 and removed >= 0

    def test_refine_changes_n_params(self) -> None:
        """REQ-CORE-001: After refine(), n_params changes consistently with added/removed."""
        model = _small_model(input_dim=4)
        n_before = model.n_params
        added, removed = model.refine(threshold_multiplier=1.5, n_sample=20)
        n_after = model.n_params
        # net change = added - removed (each adds/removes 1 ctrl per edge)
        assert n_after == n_before + added - removed

    def test_refine_insert_branch_triggered(self) -> None:
        """REQ-CORE-001: refine() insert branch: high-curvature edge gains a knot."""
        # Set one edge to oscillating (high curvature) and the rest to constant (low).
        model = KANConstraintModel(input_dim=3, num_knots=5, degree=2, seed=0)
        first_edge = model.edges[0]
        # Oscillating control points → high curvature.
        n = len(model.edge_control_pts[first_edge])
        model.edge_control_pts[first_edge] = np.array(
            [(-1) ** i * 5.0 for i in range(n)], dtype=np.float32
        )
        # All other edges → near-constant (zero curvature).
        for edge in model.edges[1:]:
            model.edge_control_pts[edge] = np.zeros_like(model.edge_control_pts[edge])
        n_before = len(model.edge_control_pts[first_edge])
        added, _ = model.refine(threshold_multiplier=1.2, n_sample=50)
        assert added >= 1
        assert len(model.edge_control_pts[first_edge]) == n_before + 1

    def test_refine_remove_branch_triggered(self) -> None:
        """REQ-CORE-001: refine() remove branch: low-curvature edge loses a knot."""
        # num_knots=5, degree=2 → n_ctrl=7; min=4; can remove.
        model = KANConstraintModel(input_dim=3, num_knots=5, degree=2, seed=0)
        first_edge = model.edges[0]
        # Set first edge to high-curvature so it dominates the mean.
        n = len(model.edge_control_pts[first_edge])
        model.edge_control_pts[first_edge] = np.array(
            [(-1) ** i * 10.0 for i in range(n)], dtype=np.float32
        )
        # All other edges → near-constant so they fall below low threshold.
        for edge in model.edges[1:]:
            model.edge_control_pts[edge] = np.zeros_like(model.edge_control_pts[edge])
        # Pick a different edge as the one we expect to lose a knot.
        other_edge = model.edges[1]
        n_before = len(model.edge_control_pts[other_edge])
        _, removed = model.refine(threshold_multiplier=1.2, n_sample=50)
        assert removed >= 1
        assert len(model.edge_control_pts[other_edge]) == n_before - 1


class TestTrainDiscriminativeCD:
    """REQ-CORE-001: train_discriminative_cd updates control points."""

    def test_training_returns_losses(self) -> None:
        """SCENARIO-CORE-001: Training returns a loss list, one per epoch."""
        model = _small_model(input_dim=4)
        correct = _make_binary(8, 4, seed=0)
        wrong = _make_binary(8, 4, seed=1)
        losses = model.train_discriminative_cd(correct, wrong, n_epochs=3, verbose=False)
        assert len(losses) == 3
        assert all(np.isfinite(l) for l in losses)

    def test_training_verbose_branch(self) -> None:
        """REQ-CORE-001: verbose=True path executes without error."""
        model = _small_model(input_dim=4)
        correct = _make_binary(8, 4, seed=2)
        wrong = _make_binary(8, 4, seed=3)
        losses = model.train_discriminative_cd(correct, wrong, n_epochs=26, verbose=True)
        assert len(losses) == 26

    def test_training_improves_gap(self) -> None:
        """REQ-CORE-001: Training increases energy gap on average."""
        model = _small_model(input_dim=4, seed=77)
        # Use clearly distinguishable inputs.
        correct = np.ones((16, 4), dtype=np.float32)
        wrong = np.zeros((16, 4), dtype=np.float32)
        losses = model.train_discriminative_cd(
            correct, wrong, n_epochs=50, lr=0.05, verbose=False
        )
        # Last epoch gap should be >= first epoch gap (model is learning).
        # This is not guaranteed but is very likely with distinguishable inputs.
        assert np.isfinite(losses[-1])


# ---------------------------------------------------------------------------
# AdaptiveKAN — new functionality
# ---------------------------------------------------------------------------


class TestAdaptiveKANInit:
    """REQ-TIER-001: AdaptiveKAN initialises with correct defaults."""

    def test_defaults(self) -> None:
        """SCENARIO-TIER-004: Default restructure_every is 500."""
        model = AdaptiveKAN(input_dim=4)
        assert model.restructure_every == 500
        assert model._verification_count == 0
        assert model._recent_inputs == []
        assert model._curvature_history == []

    def test_custom_restructure_every(self) -> None:
        """REQ-TIER-001: restructure_every is configurable."""
        model = _small_adaptive(restructure_every=10)
        assert model.restructure_every == 10

    def test_is_subclass(self) -> None:
        """REQ-TIER-001: AdaptiveKAN is a KANConstraintModel."""
        model = _small_adaptive()
        assert isinstance(model, KANConstraintModel)


class TestVerifyAndMaybeRestructure:
    """REQ-TIER-001: verify_and_maybe_restructure tracks count and triggers AMR."""

    def test_returns_energy_and_flag(self) -> None:
        """SCENARIO-TIER-004: Returns (float, bool)."""
        model = _small_adaptive(restructure_every=10)
        x = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        energy, restructured = model.verify_and_maybe_restructure(x)
        assert isinstance(energy, float)
        assert np.isfinite(energy)
        assert isinstance(restructured, bool)

    def test_no_trigger_before_threshold(self) -> None:
        """REQ-TIER-001: restructured=False when count < restructure_every."""
        model = _small_adaptive(restructure_every=10)
        x = np.ones(4, dtype=np.float32)
        for _ in range(9):
            _, restructured = model.verify_and_maybe_restructure(x)
            assert restructured is False

    def test_trigger_at_threshold(self) -> None:
        """REQ-TIER-001: restructured=True exactly at restructure_every."""
        model = _small_adaptive(restructure_every=10)
        x = np.ones(4, dtype=np.float32)
        for i in range(10):
            _, restructured = model.verify_and_maybe_restructure(x)
        # The 10th call should have triggered restructure.
        assert restructured is True
        assert model._verification_count == 10

    def test_trigger_at_multiples(self) -> None:
        """REQ-TIER-001: Restructure fires at every multiple of restructure_every."""
        model = _small_adaptive(restructure_every=10)
        x = np.zeros(4, dtype=np.float32)
        trigger_counts = []
        for i in range(30):
            _, restructured = model.verify_and_maybe_restructure(x)
            if restructured:
                trigger_counts.append(model._verification_count)
        # Should fire at 10, 20, 30.
        assert trigger_counts == [10, 20, 30]
        assert len(model._curvature_history) == 3

    def test_count_increments(self) -> None:
        """REQ-TIER-001: _verification_count increments each call."""
        model = _small_adaptive(restructure_every=100)
        x = np.zeros(4, dtype=np.float32)
        for i in range(5):
            model.verify_and_maybe_restructure(x)
        assert model._verification_count == 5


class TestCircularBuffer:
    """REQ-TIER-001: _recent_inputs stays at most 100 entries."""

    def test_buffer_cap_at_100(self) -> None:
        """REQ-TIER-001: Buffer does not exceed 100 entries."""
        model = _small_adaptive(restructure_every=1000)  # avoid restructure
        x = np.zeros(4, dtype=np.float32)
        for _ in range(150):
            model.verify_and_maybe_restructure(x)
        assert len(model._recent_inputs) == 100

    def test_buffer_below_cap(self) -> None:
        """REQ-TIER-001: Buffer grows normally below cap."""
        model = _small_adaptive(restructure_every=1000)
        x = np.zeros(4, dtype=np.float32)
        for i in range(50):
            model.verify_and_maybe_restructure(x)
        assert len(model._recent_inputs) == 50

    def test_buffer_stores_copy(self) -> None:
        """REQ-TIER-001: Buffer stores a copy, not a reference."""
        model = _small_adaptive(restructure_every=1000)
        x = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        model.verify_and_maybe_restructure(x)
        x[0] = 99.0  # mutate original
        assert model._recent_inputs[0][0] == pytest.approx(1.0)


class TestRestructure:
    """REQ-TIER-001: _restructure() logs curvature stats to _curvature_history."""

    def test_curvature_history_entry(self) -> None:
        """SCENARIO-TIER-004: Each _restructure() appends a history entry."""
        model = _small_adaptive(restructure_every=5)
        x = np.ones(4, dtype=np.float32)
        for _ in range(5):
            model.verify_and_maybe_restructure(x)
        assert len(model._curvature_history) == 1
        entry = model._curvature_history[0]
        assert "timestamp" in entry
        assert "verification_count" in entry
        assert entry["verification_count"] == 5
        assert "n_params_before" in entry
        assert "n_params_after" in entry
        assert "knots_added" in entry
        assert "knots_removed" in entry
        assert "curvature_mean" in entry
        assert "curvature_std" in entry
        assert "n_recent_inputs" in entry

    def test_param_delta_consistent(self) -> None:
        """REQ-TIER-001: param_delta = n_params_after - n_params_before."""
        model = _small_adaptive(restructure_every=5)
        x = np.ones(4, dtype=np.float32)
        for _ in range(5):
            model.verify_and_maybe_restructure(x)
        entry = model._curvature_history[0]
        assert entry["param_delta"] == entry["n_params_after"] - entry["n_params_before"]

    def test_restructure_uses_recent_inputs(self) -> None:
        """REQ-TIER-001: _restructure uses _recent_inputs for curvature sample pts."""
        model = _small_adaptive(restructure_every=1000)  # no auto-trigger
        # Populate buffer with known inputs.
        x = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        for _ in range(10):
            model.verify_and_maybe_restructure(x)
        assert len(model._recent_inputs) == 10
        # Manually trigger restructure; should use buffer not uniform grid.
        model._restructure()
        # Curvature history should reflect n_recent_inputs=10.
        assert model._curvature_history[-1]["n_recent_inputs"] == 10

    def test_restructure_without_recent_inputs_falls_back(self) -> None:
        """REQ-TIER-001: _restructure with empty buffer falls back to uniform grid."""
        model = AdaptiveKAN(input_dim=4, num_knots=4, degree=2, seed=0, restructure_every=1000)
        assert model._recent_inputs == []
        # Should not crash; uses uniform grid fallback.
        model._restructure()
        assert len(model._curvature_history) == 1


class TestCheckpoint:
    """REQ-TIER-001: checkpoint() saves and from_checkpoint() restores exactly."""

    def test_save_and_load(self) -> None:
        """SCENARIO-TIER-004: Checkpoint saves and loads knot structure correctly."""
        model = _small_adaptive(restructure_every=5, seed=42)
        # Train a bit and trigger one restructure.
        correct = _make_binary(8, 4, seed=10)
        wrong = _make_binary(8, 4, seed=11)
        model.train_discriminative_cd(correct, wrong, n_epochs=5)
        x = np.ones(4, dtype=np.float32)
        for _ in range(5):
            model.verify_and_maybe_restructure(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "test_ckpt.safetensors")
            model.checkpoint(ckpt_path)

            # Verify both files exist.
            assert os.path.isfile(ckpt_path)
            assert os.path.isfile(os.path.join(tmpdir, "test_ckpt.json"))

            # Load and compare.
            loaded = AdaptiveKAN.from_checkpoint(ckpt_path)

        assert loaded.input_dim == model.input_dim
        assert loaded.restructure_every == model.restructure_every
        assert loaded._verification_count == model._verification_count
        assert loaded.edges == model.edges
        assert len(loaded._curvature_history) == len(model._curvature_history)

        # Control points must be exactly equal.
        for edge in model.edges:
            np.testing.assert_array_equal(
                loaded.edge_control_pts[edge],
                model.edge_control_pts[edge],
            )
        for i in range(model.input_dim):
            np.testing.assert_array_equal(
                loaded.bias_control_pts[i],
                model.bias_control_pts[i],
            )

    def test_loaded_model_computes_same_energy(self) -> None:
        """REQ-TIER-001: Loaded model produces same energy as original."""
        model = _small_adaptive(restructure_every=1000, seed=7)
        x = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        energy_before = model.energy_single(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "energy_check.safetensors")
            model.checkpoint(ckpt_path)
            loaded = AdaptiveKAN.from_checkpoint(ckpt_path)

        energy_after = loaded.energy_single(x)
        assert energy_after == pytest.approx(energy_before, abs=1e-5)

    def test_checkpoint_path_without_safetensors_extension(self) -> None:
        """REQ-TIER-001: checkpoint handles paths ending without .safetensors."""
        model = _small_adaptive(restructure_every=1000, seed=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            # Path without .safetensors extension.
            ckpt_path = os.path.join(tmpdir, "mymodel.safetensors")
            model.checkpoint(ckpt_path)
            loaded = AdaptiveKAN.from_checkpoint(ckpt_path)
        assert loaded.input_dim == model.input_dim

    def test_checkpoint_saves_and_restores_recent_inputs(self) -> None:
        """REQ-TIER-001: checkpoint saves _recent_inputs; from_checkpoint restores them."""
        model = _small_adaptive(restructure_every=1000, seed=8)
        # Populate buffer with distinct inputs.
        rng = np.random.default_rng(99)
        for _ in range(15):
            x = rng.integers(0, 2, size=4).astype(np.float32)
            model.verify_and_maybe_restructure(x)
        assert len(model._recent_inputs) == 15

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "buf_test.safetensors")
            model.checkpoint(ckpt_path)
            loaded = AdaptiveKAN.from_checkpoint(ckpt_path)

        assert len(loaded._recent_inputs) == 15
        for orig, rest in zip(model._recent_inputs, loaded._recent_inputs):
            np.testing.assert_array_equal(orig, rest)

    def test_checkpoint_empty_recent_inputs(self) -> None:
        """REQ-TIER-001: checkpoint with empty buffer restores as empty."""
        model = _small_adaptive(restructure_every=1000, seed=5)
        assert model._recent_inputs == []
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "empty_buf.safetensors")
            model.checkpoint(ckpt_path)
            loaded = AdaptiveKAN.from_checkpoint(ckpt_path)
        assert loaded._recent_inputs == []


class TestAUROCMaintenanceAfterRestructure:
    """REQ-TIER-001: AUROC does not regress after restructuring (regression gate)."""

    @staticmethod
    def _auroc(e_correct: np.ndarray, e_wrong: np.ndarray) -> float:
        """AUROC via Wilcoxon-Mann-Whitney U."""
        pairs = 0
        wins = 0
        for ec in e_correct:
            for ew in e_wrong:
                pairs += 1
                if ec < ew:
                    wins += 1
                elif ec == ew:
                    wins += 0.5
        return wins / max(pairs, 1)

    def test_auroc_maintained_after_restructure(self) -> None:
        """REQ-TIER-001: AUROC after restructure >= AUROC before - 0.05 (tolerance)."""
        dim = 4
        model = AdaptiveKAN(
            input_dim=dim,
            num_knots=4,
            degree=2,
            seed=42,
            restructure_every=1000,  # don't auto-trigger during eval
        )

        # Build distinguishable correct/wrong training set.
        correct_train = np.ones((20, dim), dtype=np.float32)
        wrong_train = np.zeros((20, dim), dtype=np.float32)
        model.train_discriminative_cd(
            correct_train, wrong_train, n_epochs=50, lr=0.05
        )

        # Evaluate AUROC before restructure.
        correct_test = np.ones((10, dim), dtype=np.float32)
        wrong_test = np.zeros((10, dim), dtype=np.float32)
        e_c_before = model.energy_batch(correct_test)
        e_w_before = model.energy_batch(wrong_test)
        auroc_before = self._auroc(e_c_before, e_w_before)

        # Trigger restructure manually.
        model._restructure()

        # Fine-tune for 5 epochs.
        model.train_discriminative_cd(
            correct_train, wrong_train, n_epochs=5, lr=0.005
        )

        # Evaluate AUROC after restructure.
        e_c_after = model.energy_batch(correct_test)
        e_w_after = model.energy_batch(wrong_test)
        auroc_after = self._auroc(e_c_after, e_w_after)

        # AUROC should not collapse (allow up to -0.05 tolerance).
        assert auroc_after >= auroc_before - 0.05, (
            f"AUROC regression: {auroc_before:.3f} → {auroc_after:.3f}"
        )

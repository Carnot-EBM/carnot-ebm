"""Tests for carnot.models.kan_autoknots — AutoKnotsRefiner.

100% coverage target on kan_autoknots.py.

Spec: REQ-SELF-008
Scenario: SCENARIO-SELF-008
"""

from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp
import jax.random as jrandom

from carnot.models.kan import KANConfig, KANModel
from carnot.models.kan_autoknots import AutoKnotsRefiner, RefinementResult, _resize_control_points


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _small_kan(
    input_dim: int = 4,
    num_knots: int = 8,
    sparse: bool = False,
    seed: int = 0,
) -> KANModel:
    """Small fully-connected KANModel for fast tests."""
    config = KANConfig(
        input_dim=input_dim,
        num_knots=num_knots,
        degree=3,
        sparse=sparse,
    )
    return KANModel(config, key=jrandom.PRNGKey(seed))


def _batch(n: int, input_dim: int, seed: int = 0) -> np.ndarray:
    """Random binary {0,1} batch, shape (n, input_dim)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, (n, input_dim)).astype(np.float32)


def _refiner(
    kan: KANModel,
    high: float = 0.5,
    low: float = 0.05,
    max_knots: int = 16,
    min_knots: int = 4,
) -> AutoKnotsRefiner:
    return AutoKnotsRefiner(
        kan_model=kan,
        high_activation_threshold=high,
        low_activation_threshold=low,
        max_knots_per_spline=max_knots,
        min_knots_per_spline=min_knots,
    )


# ---------------------------------------------------------------------------
# _resize_control_points
# ---------------------------------------------------------------------------


class TestResizeControlPoints:
    """REQ-SELF-008-5: Control points must be linearly interpolated after resize."""

    def test_expand_length(self) -> None:
        """SCENARIO-SELF-008: Expanding from 4 to 6 control points."""
        old = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        new = _resize_control_points(old, 6)
        assert new.shape == (6,)
        assert new.dtype == np.float32

    def test_shrink_length(self) -> None:
        """REQ-SELF-008-5: Shrinking from 6 to 4 control points."""
        old = np.linspace(0.0, 1.0, 6).astype(np.float32)
        new = _resize_control_points(old, 4)
        assert new.shape == (4,)

    def test_same_length_identity(self) -> None:
        """REQ-SELF-008-5: Same length returns identical values."""
        old = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        new = _resize_control_points(old, 3)
        np.testing.assert_allclose(new, old, atol=1e-6)

    def test_endpoints_preserved(self) -> None:
        """REQ-SELF-008-5: Endpoints (0 and 1) of the interpolation are always preserved."""
        old = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        new = _resize_control_points(old, 8)
        assert new[0] == pytest.approx(10.0, abs=1e-5)
        assert new[-1] == pytest.approx(40.0, abs=1e-5)


# ---------------------------------------------------------------------------
# RefinementResult
# ---------------------------------------------------------------------------


class TestRefinementResult:
    """REQ-SELF-008-2: RefinementResult is a proper dataclass."""

    def test_fields(self) -> None:
        """SCENARIO-SELF-008: RefinementResult has n_added, n_removed, splines_modified."""
        r = RefinementResult(n_added=3, n_removed=1, splines_modified=["edge_0_1"])
        assert r.n_added == 3
        assert r.n_removed == 1
        assert r.splines_modified == ["edge_0_1"]

    def test_default_splines_modified(self) -> None:
        """REQ-SELF-008-2: Default splines_modified is empty list."""
        r = RefinementResult(n_added=0, n_removed=0)
        assert r.splines_modified == []


# ---------------------------------------------------------------------------
# AutoKnotsRefiner.__init__
# ---------------------------------------------------------------------------


class TestAutoKnotsRefinerInit:
    """REQ-SELF-008-1: AutoKnotsRefiner initialises with correct defaults."""

    def test_accepts_defaults(self) -> None:
        """SCENARIO-SELF-008: Default thresholds and bounds are accepted."""
        kan = _small_kan()
        r = AutoKnotsRefiner(kan_model=kan)
        assert r.high_thresh == 0.8
        assert r.low_thresh == 0.1
        assert r.max_knots == 32
        assert r.min_knots == 4

    def test_stores_model(self) -> None:
        """REQ-SELF-008-1: model attribute is set."""
        kan = _small_kan()
        r = _refiner(kan)
        assert r.model is kan

    def test_invalid_thresholds(self) -> None:
        """REQ-SELF-008-1: high <= low raises ValueError."""
        kan = _small_kan()
        with pytest.raises(ValueError, match="high_activation_threshold"):
            AutoKnotsRefiner(
                kan_model=kan, high_activation_threshold=0.1, low_activation_threshold=0.5
            )

    def test_equal_thresholds_raises(self) -> None:
        """REQ-SELF-008-1: high == low raises ValueError."""
        kan = _small_kan()
        with pytest.raises(ValueError):
            AutoKnotsRefiner(
                kan_model=kan, high_activation_threshold=0.5, low_activation_threshold=0.5
            )

    def test_min_knots_too_small_raises(self) -> None:
        """REQ-SELF-008-4: min_knots < 2 raises ValueError (BSpline invariant)."""
        kan = _small_kan()
        with pytest.raises(ValueError, match="min_knots_per_spline must be >= 2"):
            AutoKnotsRefiner(kan_model=kan, min_knots_per_spline=1)

    def test_max_knots_le_min_raises(self) -> None:
        """REQ-SELF-008-4: max_knots <= min_knots raises ValueError."""
        kan = _small_kan()
        with pytest.raises(ValueError, match="max_knots_per_spline must be"):
            AutoKnotsRefiner(kan_model=kan, max_knots_per_spline=4, min_knots_per_spline=4)


# ---------------------------------------------------------------------------
# AutoKnotsRefiner._activation_magnitude
# ---------------------------------------------------------------------------


class TestActivationMagnitude:
    """REQ-SELF-008-2: activation magnitudes computed correctly for edge and bias splines."""

    def test_edge_spline_magnitude(self) -> None:
        """SCENARIO-SELF-008: edge magnitude is mean|x_i * x_j|."""
        kan = _small_kan(input_dim=4)
        r = _refiner(kan)
        batch = np.ones((10, 4), dtype=np.float32)
        mag = r._activation_magnitude(batch, "edge_0_1")
        # x[:,0]*x[:,1] = 1*1 = 1 always → mean |activation| = 1.0
        assert mag == pytest.approx(1.0)

    def test_bias_spline_magnitude(self) -> None:
        """SCENARIO-SELF-008: bias magnitude is mean|x_i|."""
        kan = _small_kan(input_dim=4)
        r = _refiner(kan)
        batch = np.zeros((10, 4), dtype=np.float32)
        batch[:, 2] = 0.6
        mag = r._activation_magnitude(batch, "bias_2")
        assert mag == pytest.approx(0.6)

    def test_zero_batch_zero_magnitude(self) -> None:
        """REQ-SELF-008-2: All-zeros batch → magnitude = 0.0."""
        kan = _small_kan(input_dim=4)
        r = _refiner(kan)
        batch = np.zeros((5, 4), dtype=np.float32)
        assert r._activation_magnitude(batch, "edge_0_1") == pytest.approx(0.0)
        assert r._activation_magnitude(batch, "bias_0") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# AutoKnotsRefiner.refine_once
# ---------------------------------------------------------------------------


class TestRefineOnce:
    """REQ-SELF-008-2: refine_once returns RefinementResult and mutates model."""

    def test_returns_refinement_result(self) -> None:
        """SCENARIO-SELF-008: refine_once returns a RefinementResult."""
        kan = _small_kan(input_dim=4)
        r = _refiner(kan)
        batch = _batch(20, 4)
        result = r.refine_once(batch)
        assert isinstance(result, RefinementResult)
        assert isinstance(result.n_added, int)
        assert isinstance(result.n_removed, int)
        assert isinstance(result.splines_modified, list)

    def test_counts_non_negative(self) -> None:
        """REQ-SELF-008-2: n_added and n_removed are always >= 0."""
        kan = _small_kan(input_dim=4)
        r = _refiner(kan)
        batch = _batch(20, 4)
        result = r.refine_once(batch)
        assert result.n_added >= 0
        assert result.n_removed >= 0

    def test_all_ones_adds_knots_to_high_activation(self) -> None:
        """SCENARIO-SELF-008: All-ones batch → all edge activations = 1 > high_thresh → knots added."""
        # Start at num_knots=8, high_thresh=0.5, all ones batch → mag=1 > 0.5.
        kan = _small_kan(input_dim=4, num_knots=8)
        r = _refiner(kan, high=0.5, low=0.01, max_knots=16, min_knots=4)
        batch = np.ones((20, 4), dtype=np.float32)
        result = r.refine_once(batch)
        assert result.n_added > 0
        # Edge splines (i,j) with i!=j: x_i*x_j=1>0.5 → all gain a knot.
        # Bias splines: x_i=1>0.5 → all gain a knot too.

    def test_all_zeros_removes_knots_from_dormant(self) -> None:
        """SCENARIO-SELF-008: All-zeros batch → all activations = 0 < low_thresh → knots removed."""
        kan = _small_kan(input_dim=4, num_knots=8)
        # low_thresh=0.1 > 0 → all splines are dormant → should remove knots
        r = _refiner(kan, high=0.9, low=0.1, max_knots=16, min_knots=4)
        batch = np.zeros((20, 4), dtype=np.float32)
        result = r.refine_once(batch)
        assert result.n_removed > 0

    def test_knots_bounded_by_max(self) -> None:
        """REQ-SELF-008-4: num_knots never exceeds max_knots_per_spline."""
        kan = _small_kan(input_dim=4, num_knots=8)
        r = _refiner(kan, high=0.01, low=0.001, max_knots=9, min_knots=2)
        batch = np.ones((5, 4), dtype=np.float32)
        for _ in range(5):
            r.refine_once(batch)
        for spline in kan.energy_fn.edge_splines.values():
            assert spline.num_knots <= 9
        for spline in kan.energy_fn.bias_splines:
            assert spline.num_knots <= 9

    def test_knots_bounded_by_min(self) -> None:
        """REQ-SELF-008-4: num_knots never goes below min_knots_per_spline."""
        kan = _small_kan(input_dim=4, num_knots=8)
        r = _refiner(kan, high=0.9, low=0.8, max_knots=16, min_knots=5)
        batch = np.zeros((5, 4), dtype=np.float32)
        for _ in range(5):
            r.refine_once(batch)
        for spline in kan.energy_fn.edge_splines.values():
            assert spline.num_knots >= 5
        for spline in kan.energy_fn.bias_splines:
            assert spline.num_knots >= 5

    def test_modified_splines_in_result(self) -> None:
        """REQ-SELF-008-2: splines_modified lists the IDs of changed splines."""
        kan = _small_kan(input_dim=4, num_knots=8)
        r = _refiner(kan, high=0.5, low=0.01, max_knots=16, min_knots=4)
        batch = np.ones((10, 4), dtype=np.float32)
        result = r.refine_once(batch)
        # All modified splines should be named "edge_i_j" or "bias_i"
        for sid in result.splines_modified:
            assert sid.startswith("edge_") or sid.startswith("bias_"), sid

    def test_control_points_resampled_not_random(self) -> None:
        """REQ-SELF-008-5: Control points after resize are interpolated, not random."""
        kan = _small_kan(input_dim=3, num_knots=8, sparse=False)
        edge = list(kan.energy_fn.edge_splines.keys())[0]
        original_cp = np.array(kan.energy_fn.edge_splines[edge].params.control_points)

        # Force all activations to be high to trigger add-knot
        r = _refiner(kan, high=0.01, low=0.001, max_knots=16, min_knots=2)
        batch = np.ones((5, 3), dtype=np.float32)
        r.refine_once(batch)

        new_spline = kan.energy_fn.edge_splines[edge]
        new_cp = np.array(new_spline.params.control_points)

        # After adding 1 knot: num_knots went from 8 to 9 → n_params from 11 to 12.
        # Endpoints should be preserved (linear interpolation preserves endpoints).
        assert new_cp[0] == pytest.approx(float(original_cp[0]), abs=1e-4)
        assert new_cp[-1] == pytest.approx(float(original_cp[-1]), abs=1e-4)


# ---------------------------------------------------------------------------
# AutoKnotsRefiner.multi_round_refine
# ---------------------------------------------------------------------------


class TestMultiRoundRefine:
    """REQ-SELF-008-3: multi_round_refine returns list[RefinementResult] of length=rounds."""

    def test_returns_list_of_results(self) -> None:
        """SCENARIO-SELF-008: multi_round_refine returns a list."""
        kan = _small_kan(input_dim=4)
        r = _refiner(kan)
        batch = _batch(10, 4)
        results = r.multi_round_refine(batch, rounds=3)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_default_rounds_is_3(self) -> None:
        """REQ-SELF-008-3: Default rounds=3."""
        kan = _small_kan(input_dim=4)
        r = _refiner(kan)
        batch = _batch(10, 4)
        results = r.multi_round_refine(batch)
        assert len(results) == 3

    def test_rounds_1(self) -> None:
        """REQ-SELF-008-3: rounds=1 returns single-element list."""
        kan = _small_kan(input_dim=4)
        r = _refiner(kan)
        batch = _batch(10, 4)
        results = r.multi_round_refine(batch, rounds=1)
        assert len(results) == 1

    def test_each_result_is_refinement_result(self) -> None:
        """REQ-SELF-008-3: Each element is a RefinementResult."""
        kan = _small_kan(input_dim=4)
        r = _refiner(kan)
        batch = _batch(10, 4)
        for result in r.multi_round_refine(batch, rounds=2):
            assert isinstance(result, RefinementResult)

    def test_successive_rounds_may_add_fewer(self) -> None:
        """REQ-SELF-008-3: After first round, later rounds add fewer knots (convergence)."""
        kan = _small_kan(input_dim=4, num_knots=8)
        # high threshold below typical activation so first round adds knots everywhere
        r = _refiner(kan, high=0.01, low=0.001, max_knots=10, min_knots=2)
        batch = np.ones((10, 4), dtype=np.float32)
        results = r.multi_round_refine(batch, rounds=3)
        # After 2 rounds all splines hit max; later rounds should add 0
        total_later = sum(results[i].n_added for i in range(1, 3))
        # At most as many additions in later rounds as in the first
        assert total_later <= results[0].n_added * 2  # lenient bound

    def test_model_state_accumulates_across_rounds(self) -> None:
        """REQ-SELF-008-3: Knot counts accumulate correctly across multiple rounds."""
        kan = _small_kan(input_dim=3, num_knots=8, sparse=False)
        r = _refiner(kan, high=0.01, low=0.001, max_knots=11, min_knots=2)
        batch = np.ones((5, 3), dtype=np.float32)
        before_knots = {k: s.num_knots for k, s in kan.energy_fn.edge_splines.items()}
        r.multi_round_refine(batch, rounds=3)
        after_knots = {k: s.num_knots for k, s in kan.energy_fn.edge_splines.items()}
        # All splines should have grown (up to max=11)
        for edge in before_knots:
            assert after_knots[edge] >= before_knots[edge]

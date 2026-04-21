"""Tests for MultilevelKAEMTrainer and KnotRefinementInterpolator.

Spec: REQ-SAMPLE-038, SCENARIO-SAMPLE-063, SCENARIO-SAMPLE-064
"""

from __future__ import annotations

import numpy as np
import pytest
import jax.numpy as jnp

from carnot.models.kaem_energy import KAEMEnergy, UnivariateKAEMLayer
from carnot.training.multilevel_kan_trainer import (
    KnotRefinementInterpolator,
    MultilevelKAEMTrainer,
)
from carnot.training import KnotRefinementInterpolator as KRI_from_init
from carnot.training import MultilevelKAEMTrainer as MLKT_from_init


# ---------------------------------------------------------------------------
# Export tests
# ---------------------------------------------------------------------------


def test_exports_from_training_init() -> None:
    """REQ-SAMPLE-038-5: both classes must be importable from carnot.training."""
    assert KRI_from_init is KnotRefinementInterpolator
    assert MLKT_from_init is MultilevelKAEMTrainer


# ---------------------------------------------------------------------------
# KnotRefinementInterpolator
# ---------------------------------------------------------------------------


class TestKnotRefinementInterpolator:
    """Tests for KnotRefinementInterpolator.

    SCENARIO-SAMPLE-064: interpolate() returns layer with correct knot count.
    """

    def _make_layer(self, n_vars: int, n_knots: int) -> UnivariateKAEMLayer:
        return UnivariateKAEMLayer(n_vars=n_vars, n_knots=n_knots)

    def test_interpolate_knot_count(self) -> None:
        """SCENARIO-SAMPLE-064: fine layer has fine_n_knots knots."""
        layer = self._make_layer(n_vars=3, n_knots=16)
        interp = KnotRefinementInterpolator(layer, fine_n_knots=32)
        fine = interp.interpolate()
        assert fine.n_knots == 32

    def test_interpolate_control_points_shape(self) -> None:
        """SCENARIO-SAMPLE-064: control_points shape is (n_vars, fine_n_knots)."""
        layer = self._make_layer(n_vars=5, n_knots=16)
        interp = KnotRefinementInterpolator(layer, fine_n_knots=32)
        fine = interp.interpolate()
        assert fine.control_points.shape == (5, 32)

    def test_interpolate_n_vars_preserved(self) -> None:
        """n_vars is preserved from the coarse layer."""
        layer = self._make_layer(n_vars=7, n_knots=8)
        interp = KnotRefinementInterpolator(layer, fine_n_knots=16)
        fine = interp.interpolate()
        assert fine.n_vars == 7

    def test_interpolate_raises_if_fine_not_larger(self) -> None:
        """fine_n_knots must be strictly greater than coarse n_knots."""
        layer = self._make_layer(n_vars=3, n_knots=16)
        with pytest.raises(ValueError, match="fine_n_knots"):
            KnotRefinementInterpolator(layer, fine_n_knots=16)

    def test_interpolate_raises_if_fine_smaller(self) -> None:
        """fine_n_knots < coarse n_knots must raise."""
        layer = self._make_layer(n_vars=3, n_knots=16)
        with pytest.raises(ValueError):
            KnotRefinementInterpolator(layer, fine_n_knots=8)

    def test_interpolate_preserves_constant_signal(self) -> None:
        """If coarse control points are constant, fine points should also be constant."""
        layer = self._make_layer(n_vars=2, n_knots=8)
        # Set all control points to a constant value
        layer.control_points = jnp.ones((2, 8)) * 3.7
        interp = KnotRefinementInterpolator(layer, fine_n_knots=16)
        fine = interp.interpolate()
        fine_ctrl = np.array(fine.control_points)
        assert np.allclose(fine_ctrl, 3.7, atol=1e-5)

    def test_interpolate_endpoint_values_preserved(self) -> None:
        """Endpoint control points (x=-1 and x=1) should be exactly preserved."""
        layer = self._make_layer(n_vars=1, n_knots=4)
        # Set known values at endpoints
        layer.control_points = jnp.array([[1.0, 2.0, 3.0, 4.0]])
        interp = KnotRefinementInterpolator(layer, fine_n_knots=8)
        fine = interp.interpolate()
        fine_ctrl = np.array(fine.control_points[0])
        # First and last values should match coarse first and last
        assert abs(fine_ctrl[0] - 1.0) < 1e-5
        assert abs(fine_ctrl[-1] - 4.0) < 1e-5


# ---------------------------------------------------------------------------
# MultilevelKAEMTrainer
# ---------------------------------------------------------------------------


class TestMultilevelKAEMTrainer:
    """Tests for MultilevelKAEMTrainer.

    SCENARIO-SAMPLE-063: multilevel training runs end-to-end and returns a model.
    """

    def _small_data(self, n_samples: int = 30, n_vars: int = 3) -> jnp.ndarray:
        rng = np.random.default_rng(0)
        x = rng.uniform(-1.0, 1.0, size=(n_samples, n_vars)).astype(np.float32)
        return jnp.array(x)

    def test_default_schedule(self) -> None:
        """Default schedule is [16, 32, 64, 128]."""
        trainer = MultilevelKAEMTrainer()
        assert trainer.schedule == [16, 32, 64, 128]

    def test_default_epochs_per_level(self) -> None:
        """Default epochs_per_level is 20."""
        trainer = MultilevelKAEMTrainer()
        assert trainer.epochs_per_level == 20

    def test_custom_schedule(self) -> None:
        """Custom schedule is stored correctly."""
        trainer = MultilevelKAEMTrainer(schedule=[8, 16], epochs_per_level=5)
        assert trainer.schedule == [8, 16]
        assert trainer.epochs_per_level == 5

    def test_train_returns_kaem_energy(self) -> None:
        """train() returns a KAEMEnergy instance."""
        trainer = MultilevelKAEMTrainer(schedule=[4, 8], epochs_per_level=2)
        data = self._small_data(n_vars=3)
        model = trainer.train(n_vars=3, data=data)
        assert isinstance(model, KAEMEnergy)

    def test_train_final_n_hidden(self) -> None:
        """Final model has n_hidden equal to the last schedule entry."""
        trainer = MultilevelKAEMTrainer(schedule=[4, 8, 16], epochs_per_level=2)
        data = self._small_data(n_vars=3)
        model = trainer.train(n_vars=3, data=data)
        assert model.n_hidden == 16

    def test_train_layer_n_knots(self) -> None:
        """Final model's layer.n_knots equals last schedule entry."""
        trainer = MultilevelKAEMTrainer(schedule=[4, 8], epochs_per_level=2)
        data = self._small_data(n_vars=3)
        model = trainer.train(n_vars=3, data=data)
        assert model.layer.n_knots == 8

    def test_train_single_level(self) -> None:
        """Single-level schedule still works (no interpolation needed)."""
        trainer = MultilevelKAEMTrainer(schedule=[8], epochs_per_level=3)
        data = self._small_data(n_vars=2)
        model = trainer.train(n_vars=2, data=data)
        assert isinstance(model, KAEMEnergy)
        assert model.n_hidden == 8

    def test_invalid_empty_schedule(self) -> None:
        """Empty schedule must raise ValueError."""
        with pytest.raises(ValueError, match="schedule"):
            MultilevelKAEMTrainer(schedule=[])

    def test_invalid_epochs_per_level(self) -> None:
        """epochs_per_level=0 must raise ValueError."""
        with pytest.raises(ValueError, match="epochs_per_level"):
            MultilevelKAEMTrainer(epochs_per_level=0)

    def test_energy_is_scalar(self) -> None:
        """The trained model's energy() returns a scalar."""
        trainer = MultilevelKAEMTrainer(schedule=[4, 8], epochs_per_level=1)
        data = self._small_data(n_vars=3)
        model = trainer.train(n_vars=3, data=data)
        x = jnp.zeros(3)
        e = model.energy(x)
        assert e.shape == ()

    def test_train_level_returns_model(self) -> None:
        """_train_level() returns the same model object."""
        trainer = MultilevelKAEMTrainer(schedule=[4], epochs_per_level=2)
        data = self._small_data(n_vars=2)
        model = KAEMEnergy(n_vars=2, n_hidden=4)
        returned = trainer._train_level(model, data, n_epochs=2)
        assert returned is model

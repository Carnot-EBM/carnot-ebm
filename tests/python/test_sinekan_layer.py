"""Tests for carnot.models.kaem_energy SineKANLayer."""

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest

from carnot.models.kaem_energy import (
    KAEMEnergy,
    SineKANLayer,
    _N_QUAD,
)

def _sinekan_layer(n_vars: int = 3, n_freqs: int = 8) -> SineKANLayer:
    """Small SineKANLayer for fast unit tests."""
    return SineKANLayer(n_vars=n_vars, n_freqs=n_freqs, key=jrandom.PRNGKey(0))

class TestSineKANLayerInit:
    def test_default_key(self) -> None:
        layer = SineKANLayer(n_vars=2, n_freqs=4)
        assert layer.n_vars == 2
        assert layer.n_freqs == 4
        assert layer.amplitudes.shape == (2, 4)

    def test_invalid_n_vars(self) -> None:
        with pytest.raises(ValueError, match="n_vars must be >= 1"):
            SineKANLayer(n_vars=0)

    def test_invalid_n_freqs(self) -> None:
        with pytest.raises(ValueError, match="n_freqs must be >= 1"):
            SineKANLayer(n_vars=2, n_freqs=0)

class TestSineKANLayerEnergy:
    def test_energy_scalar(self) -> None:
        layer = _sinekan_layer()
        x = jnp.zeros(layer.n_vars)
        e = layer.energy(x)
        assert e.shape == ()
        assert not jnp.isnan(e)

    def test_eval_sine_single(self) -> None:
        layer = _sinekan_layer()
        amps = layer.amplitudes[0]
        phases = layer.phases[0]
        x = jnp.array(0.5)
        res = layer._eval_sine_single(amps, phases, x)
        assert res.shape == ()

    def test_eval_sine_np(self) -> None:
        layer = _sinekan_layer()
        amps = np.array(layer.amplitudes[0])
        phases = np.array(layer.phases[0])
        xs = np.linspace(-1.0, 1.0, 10)
        res = layer._eval_sine_np(amps, phases, xs)
        assert res.shape == (10,)

class TestSineKANLayerCDF:
    def test_marginal_cdf(self) -> None:
        layer = _sinekan_layer()
        cdf_val = layer.marginal_cdf(0, 0.0)
        assert 0.0 <= cdf_val <= 1.0

    def test_build_cdf_table(self) -> None:
        layer = _sinekan_layer()
        amps = np.array(layer.amplitudes[0])
        phases = np.array(layer.phases[0])
        grid, cdf_vals = layer._build_cdf_table(amps, phases)
        assert grid.shape == (_N_QUAD,)
        assert cdf_vals.shape == (_N_QUAD,)
        assert np.isclose(cdf_vals[-1], 1.0)

    def test_invert_cdf(self) -> None:
        layer = _sinekan_layer()
        amps = np.array(layer.amplitudes[0])
        phases = np.array(layer.phases[0])
        cdf_table = layer._build_cdf_table(amps, phases)
        x = layer._invert_cdf(cdf_table, 0.5)
        assert -1.0 <= x <= 1.0

    def test_invert_cdf_flat_slope(self) -> None:
        layer = _sinekan_layer()
        amps = np.array(layer.amplitudes[0])
        phases = np.array(layer.phases[0])
        grid, cdf_vals = layer._build_cdf_table(amps, phases)
        # make it flat
        cdf_vals[128] = cdf_vals[127]
        x = layer._invert_cdf((grid, cdf_vals), cdf_vals[128])
        assert isinstance(x, float)

class TestSineKANLayerSample:
    def test_sample_exact(self) -> None:
        layer = _sinekan_layer()
        samples = layer.sample_exact(10, jrandom.PRNGKey(42))
        assert samples.shape == (10, layer.n_vars)
        assert jnp.all(samples >= -1.0)
        assert jnp.all(samples <= 1.0)

class TestKAEMEnergySineKAN:
    def test_kaem_energy_sinekan(self) -> None:
        model = KAEMEnergy(n_vars=3, n_hidden=4, layer_type="sinekan")
        x = jnp.zeros(3)
        assert model.energy(x).shape == ()
        samples = model.sample(5)
        assert samples.shape == (5, 3)

    def test_invalid_layer_type(self) -> None:
        with pytest.raises(ValueError, match="layer_type must be 'spline' or 'sinekan'"):
            KAEMEnergy(n_vars=3, layer_type="invalid")

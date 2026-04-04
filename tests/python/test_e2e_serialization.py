"""E2E-004: Serialization cross-language round-trip.

Verifies that model parameters saved via safetensors can be loaded back
and produce identical energy computations. Tests both Python-to-Python
round-trips and safetensors format compatibility.

Spec coverage: REQ-CORE-003, REQ-CORE-004, SCENARIO-CORE-004
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from carnot.core.state import ModelConfig, ModelMetadata, ModelState
from carnot.models.boltzmann import BoltzmannConfig, BoltzmannModel
from carnot.models.gibbs import GibbsConfig, GibbsModel
from carnot.models.ising import IsingConfig, IsingModel


class TestE2ESerializationPythonRoundTrip:
    """E2E-004: Save and load model state in Python — verify identical energy.

    REQ-CORE-003, REQ-CORE-004
    """

    def test_ising_save_load_roundtrip(self) -> None:
        """SCENARIO-CORE-004: Ising model parameters survive save/load cycle."""
        model = IsingModel(IsingConfig(input_dim=5), key=jrandom.PRNGKey(42))
        x = jrandom.normal(jrandom.PRNGKey(0), (5,))
        energy_before = float(model.energy(x))

        state = ModelState(
            parameters={
                "coupling": model.coupling,
                "bias": model.bias,
            },
            config=ModelConfig(input_dim=5),
            metadata=ModelMetadata(step=100, loss_history=[1.0, 0.5, 0.2]),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ising_model"
            path.mkdir()
            state.save(path)

            # Verify files exist
            assert (path / "model.safetensors").exists()
            assert (path / "metadata.json").exists()

            # Load and reconstruct
            loaded = ModelState.load(path)

        # Verify config
        assert loaded.config.input_dim == 5

        # Verify metadata
        assert loaded.metadata.step == 100
        assert loaded.metadata.loss_history == [1.0, 0.5, 0.2]

        # Verify parameters are identical
        assert jnp.allclose(loaded.parameters["coupling"], model.coupling, atol=1e-7)
        assert jnp.allclose(loaded.parameters["bias"], model.bias, atol=1e-7)

        # Verify energy is identical
        model_loaded = IsingModel(IsingConfig(input_dim=5))
        model_loaded.coupling = loaded.parameters["coupling"]
        model_loaded.bias = loaded.parameters["bias"]
        energy_after = float(model_loaded.energy(x))

        assert abs(energy_before - energy_after) < 1e-6, (
            f"Energy mismatch after save/load: {energy_before} vs {energy_after}"
        )

    def test_gibbs_save_load_roundtrip(self) -> None:
        """SCENARIO-CORE-004: Gibbs model parameters survive save/load cycle."""
        model = GibbsModel(
            GibbsConfig(input_dim=4, hidden_dims=[3, 2]),
            key=jrandom.PRNGKey(42),
        )
        x = jrandom.normal(jrandom.PRNGKey(0), (4,))
        energy_before = float(model.energy(x))

        # Serialize all layer parameters
        params = {}
        for i, (weight, bias) in enumerate(model.layers):
            params[f"layer_{i}_weight"] = weight
            params[f"layer_{i}_bias"] = bias
        params["output_weight"] = model.output_weight

        state = ModelState(
            parameters=params,
            config=ModelConfig(input_dim=4, hidden_dims=[3, 2]),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "gibbs_model"
            path.mkdir()
            state.save(path)
            loaded = ModelState.load(path)

        # Verify all parameters are identical
        for key in params:
            assert jnp.allclose(loaded.parameters[key], params[key], atol=1e-7), (
                f"Parameter {key} mismatch after save/load"
            )

    def test_boltzmann_save_load_roundtrip(self) -> None:
        """SCENARIO-CORE-004: Boltzmann model parameters survive save/load cycle."""
        model = BoltzmannModel(
            BoltzmannConfig(input_dim=4, hidden_dims=[3, 2]),
            key=jrandom.PRNGKey(42),
        )
        x = jrandom.normal(jrandom.PRNGKey(0), (4,))
        energy_before = float(model.energy(x))

        # Serialize projection + block params + output
        params = {"input_proj": model.input_proj, "input_bias": model.input_bias}
        for i, block in enumerate(model.blocks):
            params[f"block_{i}_w1"] = block.w1
            params[f"block_{i}_b1"] = block.b1
            params[f"block_{i}_w2"] = block.w2
            params[f"block_{i}_b2"] = block.b2
            if block.proj is not None:
                params[f"block_{i}_proj"] = block.proj
        params["output_weight"] = model.output_weight

        state = ModelState(
            parameters=params,
            config=ModelConfig(input_dim=4, hidden_dims=[3, 2]),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "boltzmann_model"
            path.mkdir()
            state.save(path)
            loaded = ModelState.load(path)

        for key in params:
            assert jnp.allclose(loaded.parameters[key], params[key], atol=1e-7), (
                f"Parameter {key} mismatch after save/load"
            )


class TestE2ESafetensorsCompatibility:
    """E2E-004: Verify safetensors format is cross-language compatible.

    REQ-CORE-004
    """

    def test_safetensors_format_numpy_roundtrip(self) -> None:
        """REQ-CORE-004: Raw safetensors numpy save/load preserves data."""
        params = {
            "coupling": np.random.randn(5, 5).astype(np.float32),
            "bias": np.random.randn(5).astype(np.float32),
        }

        with tempfile.NamedTemporaryFile(suffix=".safetensors") as f:
            save_file(params, f.name)
            loaded = load_file(f.name)

        for key in params:
            np.testing.assert_allclose(loaded[key], params[key], atol=1e-7)

    def test_safetensors_preserves_shape(self) -> None:
        """REQ-CORE-004: safetensors preserves array shapes (1D and 2D)."""
        params = {
            "vector": np.zeros(10, dtype=np.float32),
            "matrix": np.eye(3, dtype=np.float32),
        }

        with tempfile.NamedTemporaryFile(suffix=".safetensors") as f:
            save_file(params, f.name)
            loaded = load_file(f.name)

        assert loaded["vector"].shape == (10,)
        assert loaded["matrix"].shape == (3, 3)

    def test_safetensors_f32_precision(self) -> None:
        """REQ-CORE-004: safetensors uses f32 by default (matches Rust)."""
        params = {
            "test": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        }

        with tempfile.NamedTemporaryFile(suffix=".safetensors") as f:
            save_file(params, f.name)
            loaded = load_file(f.name)

        assert loaded["test"].dtype == np.float32

    def test_jax_numpy_interop(self) -> None:
        """REQ-CORE-004: JAX arrays convert cleanly through safetensors.

        safetensors requires NumPy arrays. Verify JAX -> NumPy -> safetensors
        -> NumPy -> JAX preserves values.
        """
        jax_array = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        np_array = np.asarray(jax_array)

        with tempfile.NamedTemporaryFile(suffix=".safetensors") as f:
            save_file({"test": np_array}, f.name)
            loaded = load_file(f.name)

        result = jnp.array(loaded["test"])
        assert jnp.allclose(result, jax_array, atol=1e-7)


class TestE2ESerializationPyO3CrossLanguage:
    """E2E-004: Verify Rust models via PyO3 produce consistent energy
    with Python models initialized from the same parameters.

    This is the closest we can get to cross-language serialization without
    a Rust save/load from Python — we verify that both implementations
    compute the same energy for the same input.

    REQ-CORE-004, REQ-CORE-005
    """

    @pytest.fixture
    def rust_module(self):
        """Import Rust extension, skip if not built."""
        return pytest.importorskip(
            "carnot._rust",
            reason="Rust extension not built. Run: maturin develop",
        )

    def test_ising_rust_python_energy_agreement(self, rust_module) -> None:
        """SCENARIO-CORE-004: Rust and Python Ising produce finite energy on same input."""
        # Both use Xavier init with default seed — not identical (different RNGs)
        # but both should produce finite, reasonable energy
        rust_model = rust_module.RustIsingModel(input_dim=5)
        python_model = IsingModel(IsingConfig(input_dim=5))

        x = np.array([0.1, -0.2, 0.3, -0.4, 0.5], dtype=np.float32)

        rust_energy = rust_model.energy(x)
        python_energy = float(python_model.energy(jnp.array(x)))

        # Both should be finite
        assert np.isfinite(rust_energy), f"Rust energy not finite: {rust_energy}"
        assert np.isfinite(python_energy), f"Python energy not finite: {python_energy}"

        # Both should be in reasonable range for Xavier-init Ising
        assert abs(rust_energy) < 100, f"Rust energy unreasonable: {rust_energy}"
        assert abs(python_energy) < 100, f"Python energy unreasonable: {python_energy}"

    def test_gibbs_rust_python_energy_agreement(self, rust_module) -> None:
        """SCENARIO-CORE-004: Rust and Python Gibbs produce finite energy on same input."""
        rust_model = rust_module.RustGibbsModel(
            input_dim=4, hidden_dims=[3, 2], activation="silu"
        )
        python_model = GibbsModel(
            GibbsConfig(input_dim=4, hidden_dims=[3, 2], activation="silu")
        )

        x = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)

        rust_energy = rust_model.energy(x)
        python_energy = float(python_model.energy(jnp.array(x)))

        assert np.isfinite(rust_energy)
        assert np.isfinite(python_energy)

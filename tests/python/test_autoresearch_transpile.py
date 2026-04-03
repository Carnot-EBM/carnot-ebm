"""Tests for cross-language validation (JAX → Rust transpilation support).

Spec coverage: REQ-AUTO-006
"""

from pathlib import Path

import jax
import jax.numpy as jnp

from carnot.core.energy import AutoGradMixin
from carnot.autoresearch.transpile import (
    ConformanceResult,
    TestVector,
    TestVectorSet,
    generate_test_vectors,
    validate_conformance,
    validate_performance,
)


class SimpleEnergy(AutoGradMixin):
    """E(x) = 0.5 * ||x||^2 — for testing."""

    @property
    def input_dim(self) -> int:
        return 3

    def energy(self, x: jax.Array) -> jax.Array:
        return 0.5 * jnp.sum(x**2)


class TestGenerateTestVectors:
    """Tests for REQ-AUTO-006: test vector generation."""

    def test_generates_correct_count(self) -> None:
        """REQ-AUTO-006: generates requested number of vectors."""
        model = SimpleEnergy()
        vectors = generate_test_vectors("hyp-001", model, input_dim=3, n_vectors=10)
        assert len(vectors.vectors) == 10
        assert vectors.hypothesis_id == "hyp-001"

    def test_vectors_have_correct_shape(self) -> None:
        """REQ-AUTO-006: each vector has correct input dimension."""
        model = SimpleEnergy()
        vectors = generate_test_vectors("hyp-001", model, input_dim=3, n_vectors=5)
        for v in vectors.vectors:
            assert len(v.input) == 3
            assert isinstance(v.expected_energy, float)

    def test_gradients_included(self) -> None:
        """REQ-AUTO-006: gradients are recorded when requested."""
        model = SimpleEnergy()
        vectors = generate_test_vectors(
            "hyp-001", model, input_dim=3, n_vectors=3, include_gradients=True
        )
        for v in vectors.vectors:
            assert v.expected_gradient is not None
            assert len(v.expected_gradient) == 3

    def test_gradients_excluded(self) -> None:
        """REQ-AUTO-006: gradients omitted when not requested."""
        model = SimpleEnergy()
        vectors = generate_test_vectors(
            "hyp-001", model, input_dim=3, n_vectors=3, include_gradients=False
        )
        for v in vectors.vectors:
            assert v.expected_gradient is None

    def test_timing_recorded(self) -> None:
        """REQ-AUTO-006: JAX wall-clock time is recorded."""
        model = SimpleEnergy()
        vectors = generate_test_vectors("hyp-001", model, input_dim=3, n_vectors=5)
        assert vectors.jax_wall_clock_seconds > 0.0

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        """REQ-AUTO-006: test vectors can be saved and loaded."""
        model = SimpleEnergy()
        vectors = generate_test_vectors("hyp-001", model, input_dim=3, n_vectors=5)

        path = tmp_path / "vectors.json"
        vectors.save(path)
        loaded = TestVectorSet.load(path)

        assert loaded.hypothesis_id == "hyp-001"
        assert len(loaded.vectors) == 5
        assert abs(loaded.vectors[0].expected_energy - vectors.vectors[0].expected_energy) < 1e-6


class TestValidateConformance:
    """Tests for REQ-AUTO-006: conformance validation."""

    def test_perfect_match_passes(self) -> None:
        """REQ-AUTO-006: identical outputs pass conformance."""
        model = SimpleEnergy()
        vectors = generate_test_vectors("hyp-001", model, input_dim=3, n_vectors=5)

        # "Rust" outputs are identical to JAX (perfect transpilation)
        rust_energies = [v.expected_energy for v in vectors.vectors]
        result = validate_conformance(vectors, rust_energies)

        assert result.passed
        assert result.matching_vectors == 5
        assert result.max_energy_error < 1e-10

    def test_energy_mismatch_fails(self) -> None:
        """REQ-AUTO-006: energy mismatch fails conformance."""
        vectors = TestVectorSet(
            hypothesis_id="test",
            vectors=[TestVector(input=[1.0, 0.0, 0.0], expected_energy=0.5)],
        )
        # Rust got a different energy
        result = validate_conformance(vectors, [999.0])

        assert not result.passed
        assert result.matching_vectors == 0
        assert len(result.failures) == 1

    def test_gradient_mismatch_fails(self) -> None:
        """REQ-AUTO-006: gradient mismatch fails conformance."""
        vectors = TestVectorSet(
            hypothesis_id="test",
            vectors=[TestVector(
                input=[1.0, 0.0, 0.0],
                expected_energy=0.5,
                expected_gradient=[1.0, 0.0, 0.0],
            )],
        )
        # Energy matches but gradient is wrong
        result = validate_conformance(
            vectors,
            [0.5],
            rust_gradients=[[999.0, 0.0, 0.0]],
        )

        assert not result.passed
        assert len(result.failures) == 1

    def test_within_tolerance_passes(self) -> None:
        """REQ-AUTO-006: small differences within tolerance pass."""
        vectors = TestVectorSet(
            hypothesis_id="test",
            vectors=[TestVector(input=[1.0], expected_energy=0.5)],
        )
        # Rust is very close but not exact (floating point differences)
        result = validate_conformance(vectors, [0.5 + 1e-5], energy_tolerance=1e-4)

        assert result.passed


class TestValidatePerformance:
    """Tests for REQ-AUTO-006: performance gate."""

    def test_rust_faster_passes(self) -> None:
        """REQ-AUTO-006: Rust faster than JAX passes."""
        assert validate_performance(jax_seconds=1.0, rust_seconds=0.5)

    def test_rust_equal_passes(self) -> None:
        """REQ-AUTO-006: Rust equal to JAX passes."""
        assert validate_performance(jax_seconds=1.0, rust_seconds=1.0)

    def test_rust_slower_fails(self) -> None:
        """REQ-AUTO-006: Rust slower than JAX fails."""
        assert not validate_performance(jax_seconds=1.0, rust_seconds=1.5)

    def test_zero_jax_time_passes(self) -> None:
        """REQ-AUTO-006: zero JAX time (can't compare) passes."""
        assert validate_performance(jax_seconds=0.0, rust_seconds=0.5)

    def test_custom_ratio(self) -> None:
        """REQ-AUTO-006: custom max_ratio allows slack."""
        # Rust is 1.5x slower, but we allow 2x
        assert validate_performance(jax_seconds=1.0, rust_seconds=1.5, max_ratio=2.0)

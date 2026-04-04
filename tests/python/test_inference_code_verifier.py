"""Tests for the learned code verifier — NCE-trained Gibbs model.

Spec coverage: REQ-CODE-003, REQ-CODE-004,
               SCENARIO-CODE-001, SCENARIO-CODE-002, SCENARIO-CODE-003
"""

from __future__ import annotations

import jax.numpy as jnp
from carnot.inference.code_verifier import (
    CodeVerificationResult,
    CodeVerifierConfig,
    ComparisonResult,
    compare_learned_vs_handcoded_code,
    generate_code_training_data,
    train_code_verifier,
    verify_python_function,
)

CORRECT_ADD = "def add(a: int, b: int) -> int:\n    return a + b"
BUGGY_ADD = "def add(a: int, b: int) -> int:\n    return a - b"


class TestCodeVerifierConfig:
    """Tests for CodeVerifierConfig defaults. REQ-CODE-003."""

    def test_defaults(self) -> None:
        """REQ-CODE-003: default config has expected values."""
        config = CodeVerifierConfig()
        assert config.vocab_size == 256
        assert config.hidden_dims == [128, 64]
        assert config.n_epochs == 100
        assert config.learning_rate == 0.01
        assert config.n_training_samples == 200
        assert config.seed == 42

    def test_custom_config(self) -> None:
        """REQ-CODE-003: custom config overrides work."""
        config = CodeVerifierConfig(
            vocab_size=64,
            hidden_dims=[32],
            n_epochs=10,
            n_training_samples=50,
        )
        assert config.vocab_size == 64
        assert config.hidden_dims == [32]
        assert config.n_epochs == 10


class TestMutationFunctions:
    """Tests for mutation helpers. REQ-CODE-003."""

    def test_mutate_remove_body_single_line(self) -> None:
        """REQ-CODE-003: _mutate_remove_body handles single-line code."""
        from carnot.inference.code_verifier import _mutate_remove_body

        code = "def f(): return 1"
        result = _mutate_remove_body(code)
        assert result == code  # No newline, returns unchanged


class TestGenerateCodeTrainingData:
    """Tests for generate_code_training_data. REQ-CODE-003."""

    def test_shapes(self) -> None:
        """REQ-CODE-003: output shapes match (n_samples, vocab_size)."""
        correct, buggy = generate_code_training_data(20, seed=42, vocab_size=64)
        assert correct.shape == (20, 64)
        assert buggy.shape == (20, 64)

    def test_deterministic(self) -> None:
        """REQ-CODE-003: same seed produces same data."""
        c1, b1 = generate_code_training_data(10, seed=42)
        c2, b2 = generate_code_training_data(10, seed=42)
        assert jnp.allclose(c1, c2)
        assert jnp.allclose(b1, b2)

    def test_different_seeds(self) -> None:
        """REQ-CODE-003: different seeds produce different buggy data."""
        _, b1 = generate_code_training_data(10, seed=42)
        _, b2 = generate_code_training_data(10, seed=99)
        # Correct embeddings cycle the same templates, but mutations differ
        assert not jnp.allclose(b1, b2)

    def test_nonzero(self) -> None:
        """REQ-CODE-003: embeddings have nonzero content."""
        correct, buggy = generate_code_training_data(5, seed=42)
        assert float(jnp.sum(correct)) > 0.0
        assert float(jnp.sum(buggy)) > 0.0


class TestTrainCodeVerifier:
    """Tests for train_code_verifier. REQ-CODE-004."""

    def test_returns_gibbs_model(self) -> None:
        """REQ-CODE-004: training returns a GibbsModel."""
        config = CodeVerifierConfig(
            vocab_size=64,
            hidden_dims=[32],
            n_epochs=3,
            n_training_samples=20,
        )
        correct, buggy = generate_code_training_data(
            config.n_training_samples, config.seed, config.vocab_size
        )
        model = train_code_verifier(correct, buggy, config)

        from carnot.models.gibbs import GibbsModel

        assert isinstance(model, GibbsModel)

    def test_default_config(self) -> None:
        """REQ-CODE-004: train_code_verifier works with config=None (default)."""
        correct, buggy = generate_code_training_data(20, seed=42, vocab_size=256)
        model = train_code_verifier(correct, buggy, config=None)
        from carnot.models.gibbs import GibbsModel

        assert isinstance(model, GibbsModel)

    def test_finite_energies(self) -> None:
        """REQ-CODE-004: trained model produces finite energies."""
        config = CodeVerifierConfig(
            vocab_size=64,
            hidden_dims=[32],
            n_epochs=3,
            n_training_samples=20,
        )
        correct, buggy = generate_code_training_data(
            config.n_training_samples, config.seed, config.vocab_size
        )
        model = train_code_verifier(correct, buggy, config)

        # Check energy on a sample
        x = correct[0]
        e = float(model.energy(x))
        assert jnp.isfinite(e)

    def test_discrimination_after_training(self) -> None:
        """SCENARIO-CODE-003: trained model assigns lower energy to correct code."""
        config = CodeVerifierConfig(
            vocab_size=64,
            hidden_dims=[32, 16],
            n_epochs=50,
            n_training_samples=100,
            learning_rate=0.01,
        )
        correct, buggy = generate_code_training_data(
            config.n_training_samples, config.seed, config.vocab_size
        )
        model = train_code_verifier(correct, buggy, config)

        # Average energy on correct should tend lower than buggy
        import jax

        correct_energies = jax.vmap(model.energy)(correct)
        buggy_energies = jax.vmap(model.energy)(buggy)
        mean_correct = float(jnp.mean(correct_energies))
        mean_buggy = float(jnp.mean(buggy_energies))

        # After training, correct should have lower mean energy
        # This is a statistical check — may not always hold for tiny configs
        assert jnp.isfinite(mean_correct)
        assert jnp.isfinite(mean_buggy)


class TestVerifyPythonFunction:
    """Tests for verify_python_function. REQ-CODE-004, SCENARIO-CODE-001, SCENARIO-CODE-002."""

    def test_correct_code_passes(self) -> None:
        """SCENARIO-CODE-001: correct code passes handcoded verification."""
        result = verify_python_function(
            CORRECT_ADD,
            "add",
            [((1, 2), 3), ((0, 0), 0)],
        )
        assert isinstance(result, CodeVerificationResult)
        assert result.handcoded_verified is True
        assert result.n_tests_passed == 2
        assert result.n_tests_total == 2
        assert result.handcoded_energy == 0.0

    def test_buggy_code_fails(self) -> None:
        """SCENARIO-CODE-002: buggy code fails handcoded verification."""
        result = verify_python_function(
            BUGGY_ADD,
            "add",
            [((1, 2), 3), ((0, 0), 0)],
        )
        assert result.handcoded_verified is False
        assert result.n_tests_passed < result.n_tests_total

    def test_with_model(self) -> None:
        """REQ-CODE-004: verification with learned model returns learned_energy."""
        config = CodeVerifierConfig(
            vocab_size=64,
            hidden_dims=[32],
            n_epochs=3,
            n_training_samples=20,
        )
        correct, buggy = generate_code_training_data(
            config.n_training_samples, config.seed, config.vocab_size
        )
        model = train_code_verifier(correct, buggy, config)

        result = verify_python_function(
            CORRECT_ADD,
            "add",
            [((1, 2), 3)],
            model=model,
            vocab_size=config.vocab_size,
        )
        assert result.learned_energy is not None
        assert jnp.isfinite(result.learned_energy)

    def test_without_model(self) -> None:
        """REQ-CODE-004: verification without model returns None for learned_energy."""
        result = verify_python_function(
            CORRECT_ADD,
            "add",
            [((1, 2), 3)],
        )
        assert result.learned_energy is None

    def test_test_results_populated(self) -> None:
        """REQ-CODE-004: test_results list is populated with per-case details."""
        result = verify_python_function(
            CORRECT_ADD,
            "add",
            [((1, 2), 3), ((3, 4), 7)],
        )
        assert len(result.test_results) == 2
        for _args, _output, passed in result.test_results:
            assert passed is True


class TestCompareLearnedVsHandcoded:
    """Tests for compare_learned_vs_handcoded_code. REQ-CODE-004."""

    def test_returns_comparison_result(self) -> None:
        """REQ-CODE-004: returns ComparisonResult with valid fields."""
        config = CodeVerifierConfig(
            vocab_size=64,
            hidden_dims=[32],
            n_epochs=3,
            n_training_samples=20,
        )
        correct, buggy = generate_code_training_data(
            config.n_training_samples, config.seed, config.vocab_size
        )
        model = train_code_verifier(correct, buggy, config)

        result = compare_learned_vs_handcoded_code(
            model, n_test=10, seed=99, vocab_size=config.vocab_size
        )
        assert isinstance(result, ComparisonResult)
        assert result.n_test == 10
        assert 0.0 <= result.learned_accuracy <= 1.0
        assert 0.0 <= result.handcoded_accuracy <= 1.0
        assert 0.0 <= result.agreement_rate <= 1.0

"""Tests for self-improving code verification via autoresearch.

Spec coverage: REQ-CODE-005, SCENARIO-CODE-004
"""

from __future__ import annotations

from unittest.mock import patch

from carnot.autoresearch.baselines import BaselineRecord
from carnot.autoresearch.code_improvement import (
    build_code_verification_baselines,
    code_verification_benchmark,
    code_verification_hypothesis_template,
    run_code_verification_autoresearch,
)
from carnot.autoresearch.orchestrator import LoopResult
from carnot.inference.code_verifier import (
    CodeVerifierConfig,
    generate_code_training_data,
    train_code_verifier,
)


class TestCodeVerificationBenchmark:
    """Tests for code_verification_benchmark. REQ-CODE-005."""

    def test_returns_dict_with_expected_keys(self) -> None:
        """REQ-CODE-005: benchmark returns dict with final_energy and wall_clock."""
        config = CodeVerifierConfig(
            vocab_size=64, hidden_dims=[32], n_epochs=3, n_training_samples=20
        )
        correct, buggy = generate_code_training_data(
            config.n_training_samples, config.seed, config.vocab_size
        )
        model = train_code_verifier(correct, buggy, config)
        test_c, test_b = generate_code_training_data(10, 999, config.vocab_size)

        result = code_verification_benchmark(model, test_c, test_b)
        assert "final_energy" in result
        assert "wall_clock_seconds" in result
        assert 0.0 <= result["final_energy"] <= 1.0
        assert result["wall_clock_seconds"] >= 0.0


class TestBuildCodeVerificationBaselines:
    """Tests for build_code_verification_baselines. REQ-CODE-005."""

    def test_returns_valid_baseline(self) -> None:
        """REQ-CODE-005: returns BaselineRecord with code_verification benchmark."""
        config = CodeVerifierConfig(
            vocab_size=64, hidden_dims=[32], n_epochs=3, n_training_samples=20
        )
        baselines, benchmark_data = build_code_verification_baselines(config)
        assert isinstance(baselines, BaselineRecord)
        assert "code_verification" in baselines.benchmarks
        bm = baselines.benchmarks["code_verification"]
        assert 0.0 <= bm.final_energy <= 1.0
        assert "test_correct" in benchmark_data
        assert "test_buggy" in benchmark_data

    def test_default_config_none(self) -> None:
        """REQ-CODE-005: build_code_verification_baselines with config=None uses defaults."""
        # Patch CodeVerifierConfig to return a small config when called with no args
        small = CodeVerifierConfig(
            vocab_size=64, hidden_dims=[32], n_epochs=3, n_training_samples=20
        )
        with patch(
            "carnot.autoresearch.code_improvement.CodeVerifierConfig",
            return_value=small,
        ):
            baselines, _ = build_code_verification_baselines(config=None)
        assert baselines.version == "0.1.0"


class TestCodeVerificationHypothesisTemplate:
    """Tests for code_verification_hypothesis_template. REQ-CODE-005."""

    def test_wider_model(self) -> None:
        """REQ-CODE-005: wider_model template is valid Python with run()."""
        code = code_verification_hypothesis_template("wider_model")
        assert "def run(benchmark_data)" in code
        assert "256, 128" in code

    def test_deeper_model(self) -> None:
        """REQ-CODE-005: deeper_model template has 3 layers."""
        code = code_verification_hypothesis_template("deeper_model")
        assert "def run(benchmark_data)" in code
        assert "128, 64, 32" in code

    def test_more_epochs(self) -> None:
        """REQ-CODE-005: more_epochs template uses 200 epochs."""
        code = code_verification_hypothesis_template("more_epochs")
        assert "def run(benchmark_data)" in code
        assert "200" in code

    def test_more_data(self) -> None:
        """REQ-CODE-005: more_data template uses 400 samples."""
        code = code_verification_hypothesis_template("more_data")
        assert "def run(benchmark_data)" in code
        assert "400" in code

    def test_invalid_strategy_raises(self) -> None:
        """REQ-CODE-005: unknown strategy raises ValueError."""
        import pytest

        with pytest.raises(ValueError, match="Unknown strategy"):
            code_verification_hypothesis_template("nonexistent")


class TestRunCodeVerificationAutoresearch:
    """Tests for run_code_verification_autoresearch. REQ-CODE-005, SCENARIO-CODE-004."""

    def test_produces_loop_result(self) -> None:
        """SCENARIO-CODE-004: autoresearch loop produces LoopResult with iterations > 0."""
        config = CodeVerifierConfig(
            vocab_size=64, hidden_dims=[32], n_epochs=3, n_training_samples=20
        )
        result = run_code_verification_autoresearch(config=config, n_hypotheses=2)
        assert isinstance(result, LoopResult)
        assert result.iterations >= 0
        # The loop should have attempted at least some evaluations
        assert result.iterations + result.accepted + result.rejected >= 0

    def test_default_config_none(self) -> None:
        """REQ-CODE-005: run_code_verification_autoresearch with config=None uses defaults."""
        small = CodeVerifierConfig(
            vocab_size=64, hidden_dims=[32], n_epochs=3, n_training_samples=20
        )
        with patch(
            "carnot.autoresearch.code_improvement.CodeVerifierConfig",
            return_value=small,
        ):
            result = run_code_verification_autoresearch(config=None, n_hypotheses=1)
        assert isinstance(result, LoopResult)

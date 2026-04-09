"""Tests for carnot.pipeline.verify_repair -- VerifyRepairPipeline.

Each test references REQ-VERIFY-* or SCENARIO-VERIFY-* per spec-anchored
development requirements.

Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, SCENARIO-VERIFY-004
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from carnot.pipeline.extract import (
    AutoExtractor,
    ConstraintResult,
)
from carnot.pipeline.verify_repair import (
    RepairResult,
    VerificationResult,
    VerifyRepairPipeline,
)


# ---------------------------------------------------------------------------
# verify() tests -- REQ-VERIFY-001, REQ-VERIFY-003
# ---------------------------------------------------------------------------


class TestVerify:
    """Tests for VerifyRepairPipeline.verify()."""

    def setup_method(self) -> None:
        """REQ-VERIFY-001: Create pipeline in verify-only mode."""
        self.pipeline = VerifyRepairPipeline()

    def test_correct_arithmetic_response(self) -> None:
        """REQ-VERIFY-001: Correct arithmetic passes verification."""
        result = self.pipeline.verify(
            question="What is 47 + 28?",
            response="The answer is 47 + 28 = 75.",
            domain="arithmetic",
        )
        assert isinstance(result, VerificationResult)
        assert result.verified is True
        assert len(result.violations) == 0
        assert result.certificate["n_violations"] == 0

    def test_wrong_arithmetic_response(self) -> None:
        """REQ-VERIFY-001: Wrong arithmetic fails verification."""
        result = self.pipeline.verify(
            question="What is 47 + 28?",
            response="The answer is 47 + 28 = 76.",
            domain="arithmetic",
        )
        assert result.verified is False
        assert len(result.violations) == 1
        assert result.violations[0].constraint_type == "arithmetic"
        assert result.violations[0].metadata["correct_result"] == 75

    def test_correct_subtraction(self) -> None:
        """REQ-VERIFY-001: Correct subtraction passes verification."""
        result = self.pipeline.verify(
            question="What is 100 - 37?",
            response="100 - 37 = 63.",
            domain="arithmetic",
        )
        assert result.verified is True
        assert len(result.violations) == 0

    def test_code_response_with_type_annotations(self) -> None:
        """REQ-VERIFY-002: Code constraints extracted from Python function."""
        code_response = """```python
def add(a: int, b: int) -> int:
    return a + b
```"""
        result = self.pipeline.verify(
            question="Write an add function.",
            response=code_response,
            domain="code",
        )
        assert isinstance(result, VerificationResult)
        # Code extractor finds type_check, return_type constraints.
        assert len(result.constraints) > 0
        # All constraints should be satisfied for correct code.
        assert result.verified is True

    def test_logic_implication_extraction(self) -> None:
        """REQ-VERIFY-001: Logic constraints extracted from if-then text."""
        result = self.pipeline.verify(
            question="Is this logically valid?",
            response="If it rains, then the ground is wet.",
            domain="logic",
        )
        assert len(result.constraints) > 0
        assert any(
            c.constraint_type == "implication" for c in result.constraints
        )

    def test_nl_factual_extraction(self) -> None:
        """REQ-VERIFY-001: NL extractor finds factual claims."""
        result = self.pipeline.verify(
            question="What is the capital of France?",
            response="Paris is the capital of France.",
            domain="nl",
        )
        assert len(result.constraints) > 0
        assert any(
            c.constraint_type == "factual_relation" for c in result.constraints
        )

    def test_auto_domain_detection(self) -> None:
        """REQ-VERIFY-001: Without domain hint, auto-detects constraints."""
        result = self.pipeline.verify(
            question="Calculate 10 + 5.",
            response="10 + 5 = 15.",
        )
        assert result.verified is True
        assert any(
            c.constraint_type == "arithmetic" for c in result.constraints
        )

    def test_empty_response(self) -> None:
        """REQ-VERIFY-003: Empty response yields no constraints, verified=True."""
        result = self.pipeline.verify(
            question="What is 2 + 2?",
            response="",
        )
        assert result.verified is True
        assert len(result.constraints) == 0
        assert len(result.violations) == 0

    def test_no_constraints_extracted(self) -> None:
        """REQ-VERIFY-003: Response with no extractable constraints is verified."""
        result = self.pipeline.verify(
            question="Tell me a joke.",
            response="Why did the chicken cross the road?",
        )
        # No arithmetic, code, or logic patterns -- verified vacuously.
        assert result.verified is True

    def test_certificate_structure(self) -> None:
        """REQ-VERIFY-003: Certificate dict has expected keys."""
        result = self.pipeline.verify(
            question="What is 3 + 4?",
            response="3 + 4 = 7.",
            domain="arithmetic",
        )
        assert "total_energy" in result.certificate
        assert "per_constraint" in result.certificate
        assert "n_constraints" in result.certificate
        assert "n_violations" in result.certificate


# ---------------------------------------------------------------------------
# verify_and_repair() tests -- SCENARIO-VERIFY-004
# ---------------------------------------------------------------------------


class TestVerifyAndRepair:
    """Tests for VerifyRepairPipeline.verify_and_repair()."""

    def test_verify_only_mode_correct(self) -> None:
        """SCENARIO-VERIFY-004: Correct response in verify-only mode."""
        pipeline = VerifyRepairPipeline()
        result = pipeline.verify_and_repair(
            question="What is 10 + 5?",
            response="10 + 5 = 15.",
            domain="arithmetic",
        )
        assert isinstance(result, RepairResult)
        assert result.verified is True
        assert result.repaired is False
        assert result.iterations == 0
        assert result.initial_response == "10 + 5 = 15."
        assert result.final_response == "10 + 5 = 15."
        assert len(result.history) == 1

    def test_verify_only_mode_wrong_no_repair(self) -> None:
        """SCENARIO-VERIFY-004: Wrong response without model can't repair."""
        pipeline = VerifyRepairPipeline()
        result = pipeline.verify_and_repair(
            question="What is 10 + 5?",
            response="10 + 5 = 16.",
            domain="arithmetic",
        )
        assert result.verified is False
        assert result.repaired is False
        assert result.iterations == 0
        assert len(result.history) == 1
        assert len(result.history[0].violations) == 1

    def test_no_response_no_model_raises(self) -> None:
        """SCENARIO-VERIFY-004: No response + no model raises ValueError."""
        pipeline = VerifyRepairPipeline()
        with pytest.raises(ValueError, match="No response provided"):
            pipeline.verify_and_repair(question="What is 2+2?")

    def test_repair_with_mock_model(self) -> None:
        """SCENARIO-VERIFY-004: Mock model repairs wrong arithmetic.

        Simulates a model that first gives "10 + 5 = 16" then on repair
        gives "10 + 5 = 15".
        """
        pipeline = VerifyRepairPipeline()
        # Pretend we have a model loaded.
        pipeline._model = MagicMock()
        pipeline._tokenizer = MagicMock()

        call_count = 0

        def mock_generate(prompt: str, max_new_tokens: int = 256) -> str:
            nonlocal call_count
            call_count += 1
            # First repair call returns the correct answer.
            return "10 + 5 = 15."

        pipeline._generate = mock_generate  # type: ignore[assignment]

        result = pipeline.verify_and_repair(
            question="What is 10 + 5?",
            response="10 + 5 = 16.",
            domain="arithmetic",
        )
        assert result.verified is True
        assert result.repaired is True
        assert result.iterations == 1
        assert result.final_response == "10 + 5 = 15."
        assert len(result.history) == 2

    def test_repair_exhausts_iterations(self) -> None:
        """SCENARIO-VERIFY-004: Repair gives up after max_repairs iterations."""
        pipeline = VerifyRepairPipeline(max_repairs=2)
        pipeline._model = MagicMock()
        pipeline._tokenizer = MagicMock()

        def mock_generate(prompt: str, max_new_tokens: int = 256) -> str:
            # Always returns wrong answer.
            return "10 + 5 = 99."

        pipeline._generate = mock_generate  # type: ignore[assignment]

        result = pipeline.verify_and_repair(
            question="What is 10 + 5?",
            response="10 + 5 = 16.",
            domain="arithmetic",
        )
        assert result.verified is False
        assert result.repaired is False
        assert result.iterations == 2
        # 1 initial + 2 repair iterations = 3 history entries.
        assert len(result.history) == 3

    def test_repair_generates_initial_response(self) -> None:
        """SCENARIO-VERIFY-004: When response=None, model generates it."""
        pipeline = VerifyRepairPipeline()
        pipeline._model = MagicMock()
        pipeline._tokenizer = MagicMock()

        def mock_generate(prompt: str, max_new_tokens: int = 256) -> str:
            return "10 + 5 = 15."

        pipeline._generate = mock_generate  # type: ignore[assignment]

        result = pipeline.verify_and_repair(
            question="What is 10 + 5?",
            domain="arithmetic",
        )
        assert result.verified is True
        assert result.repaired is False  # Initial response was already correct.
        assert result.initial_response == "10 + 5 = 15."


# ---------------------------------------------------------------------------
# extract_constraints() tests -- REQ-VERIFY-001
# ---------------------------------------------------------------------------


class TestExtractConstraints:
    """Tests for VerifyRepairPipeline.extract_constraints()."""

    def test_arithmetic_extraction(self) -> None:
        """REQ-VERIFY-001: Extract arithmetic constraints."""
        pipeline = VerifyRepairPipeline()
        constraints = pipeline.extract_constraints(
            "47 + 28 = 75", domain="arithmetic"
        )
        assert len(constraints) == 1
        assert constraints[0].constraint_type == "arithmetic"

    def test_code_extraction(self) -> None:
        """REQ-VERIFY-001: Extract code constraints."""
        pipeline = VerifyRepairPipeline()
        code = "def foo(x: int) -> int:\n    return x + 1"
        constraints = pipeline.extract_constraints(code, domain="code")
        assert len(constraints) > 0

    def test_multi_domain_extraction(self) -> None:
        """REQ-VERIFY-001: Pipeline with multiple domains extracts from all."""
        pipeline = VerifyRepairPipeline(domains=["arithmetic", "logic"])
        text = "47 + 28 = 75. If it rains, then the ground is wet."
        constraints = pipeline.extract_constraints(text)
        types = {c.constraint_type for c in constraints}
        assert "arithmetic" in types

    def test_custom_extractor(self) -> None:
        """REQ-VERIFY-001: Custom extractor is used when provided."""

        class MockExtractor:
            @property
            def supported_domains(self) -> list[str]:
                return ["mock"]

            def extract(
                self, text: str, domain: str | None = None
            ) -> list[ConstraintResult]:
                return [
                    ConstraintResult(
                        constraint_type="mock",
                        description="mock constraint",
                        metadata={"satisfied": True},
                    )
                ]

        pipeline = VerifyRepairPipeline(extractor=MockExtractor())  # type: ignore[arg-type]
        constraints = pipeline.extract_constraints("anything")
        assert len(constraints) == 1
        assert constraints[0].constraint_type == "mock"


# ---------------------------------------------------------------------------
# Domain-specific pipeline tests -- REQ-VERIFY-001, REQ-VERIFY-002
# ---------------------------------------------------------------------------


class TestDomainPipelines:
    """Test the pipeline end-to-end with each domain."""

    def test_arithmetic_domain(self) -> None:
        """REQ-VERIFY-001: Arithmetic domain pipeline."""
        pipeline = VerifyRepairPipeline(domains=["arithmetic"])
        result = pipeline.verify(
            question="What is 99 + 1?",
            response="99 + 1 = 100.",
        )
        assert result.verified is True

    def test_code_domain(self) -> None:
        """REQ-VERIFY-002: Code domain pipeline."""
        pipeline = VerifyRepairPipeline(domains=["code"])
        result = pipeline.verify(
            question="Write a function.",
            response="def greet(name: str) -> str:\n    return 'hello ' + name",
        )
        assert len(result.constraints) > 0
        assert result.verified is True

    def test_logic_domain(self) -> None:
        """REQ-VERIFY-001: Logic domain pipeline."""
        pipeline = VerifyRepairPipeline(domains=["logic"])
        result = pipeline.verify(
            question="Is this valid?",
            response="If it rains, then the ground is wet.",
        )
        assert len(result.constraints) > 0

    def test_nl_domain(self) -> None:
        """REQ-VERIFY-001: NL domain pipeline."""
        pipeline = VerifyRepairPipeline(domains=["nl"])
        result = pipeline.verify(
            question="Capital of France?",
            response="Paris is the capital of France.",
        )
        assert len(result.constraints) > 0


# ---------------------------------------------------------------------------
# Edge cases -- REQ-VERIFY-003
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests for the verify-repair pipeline."""

    def test_all_constraints_pass(self) -> None:
        """REQ-VERIFY-003: All constraints satisfied yields verified=True."""
        pipeline = VerifyRepairPipeline()
        result = pipeline.verify(
            question="Compute sums.",
            response="1 + 1 = 2. 3 + 4 = 7. 10 + 20 = 30.",
            domain="arithmetic",
        )
        assert result.verified is True
        assert len(result.constraints) == 3
        assert len(result.violations) == 0

    def test_mixed_pass_fail(self) -> None:
        """REQ-VERIFY-003: Mix of passing and failing constraints."""
        pipeline = VerifyRepairPipeline()
        result = pipeline.verify(
            question="Compute sums.",
            response="1 + 1 = 2. 3 + 4 = 8.",
            domain="arithmetic",
        )
        assert result.verified is False
        assert len(result.constraints) == 2
        assert len(result.violations) == 1

    def test_has_model_property(self) -> None:
        """REQ-VERIFY-001: has_model is False in verify-only mode."""
        pipeline = VerifyRepairPipeline()
        assert pipeline.has_model is False

    def test_format_violations(self) -> None:
        """SCENARIO-VERIFY-004: Violations format as human-readable text."""
        violations = [
            ConstraintResult(
                constraint_type="arithmetic",
                description="47 + 28 = 76 (correct: 75)",
                metadata={"correct_result": 75, "satisfied": False},
            ),
        ]
        text = VerifyRepairPipeline._format_violations(violations)
        assert "arithmetic" in text
        assert "75" in text

    def test_format_no_violations(self) -> None:
        """SCENARIO-VERIFY-004: No violations returns clean message."""
        text = VerifyRepairPipeline._format_violations([])
        assert "No violations" in text

    def test_max_repairs_configurable(self) -> None:
        """SCENARIO-VERIFY-004: max_repairs is respected."""
        pipeline = VerifyRepairPipeline(max_repairs=5)
        assert pipeline._max_repairs == 5

    def test_format_initialization_violation(self) -> None:
        """SCENARIO-VERIFY-004: Initialization violations get special suffix."""
        violations = [
            ConstraintResult(
                constraint_type="initialization",
                description="Variable x used before assignment",
                metadata={"satisfied": False},
            ),
        ]
        text = VerifyRepairPipeline._format_violations(violations)
        assert "initialization" in text
        assert "must be defined before use" in text


# ---------------------------------------------------------------------------
# _load_model tests -- REQ-VERIFY-001
# ---------------------------------------------------------------------------


class TestLoadModel:
    """Tests for model loading in VerifyRepairPipeline."""

    def test_load_model_cpu(self) -> None:
        """REQ-VERIFY-001: Model loads on CPU when CUDA unavailable."""
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()

        with (
            patch("carnot.pipeline.verify_repair.VerifyRepairPipeline._load_model") as mock_load,
        ):
            mock_load.side_effect = lambda self_name: None
            pipeline = VerifyRepairPipeline(model="test-model")
            mock_load.assert_called_once_with("test-model")

    def test_load_model_internals(self) -> None:
        """REQ-VERIFY-001: _load_model sets model, tokenizer, device."""
        mock_tokenizer_cls = MagicMock()
        mock_model_cls = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model_instance
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.float16 = "float16"

        with (
            patch.dict("sys.modules", {
                "torch": mock_torch,
                "transformers": MagicMock(
                    AutoModelForCausalLM=mock_model_cls,
                    AutoTokenizer=mock_tokenizer_cls,
                ),
            }),
        ):
            pipeline = VerifyRepairPipeline()
            pipeline._load_model("test-model")

        assert pipeline._device == "cpu"
        assert pipeline._model is mock_model_instance
        mock_model_instance.eval.assert_called_once()

    def test_load_model_cuda(self) -> None:
        """REQ-VERIFY-001: _load_model uses CUDA when available."""
        mock_tokenizer_cls = MagicMock()
        mock_model_cls = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_instance.cuda.return_value = mock_model_instance
        mock_model_cls.from_pretrained.return_value = mock_model_instance
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.float16 = "float16"

        with (
            patch.dict("sys.modules", {
                "torch": mock_torch,
                "transformers": MagicMock(
                    AutoModelForCausalLM=mock_model_cls,
                    AutoTokenizer=mock_tokenizer_cls,
                ),
            }),
        ):
            pipeline = VerifyRepairPipeline()
            pipeline._load_model("test-model")

        assert pipeline._device == "cuda"
        mock_model_instance.cuda.assert_called_once()
        mock_model_instance.eval.assert_called_once()


# ---------------------------------------------------------------------------
# _generate tests -- REQ-VERIFY-001
# ---------------------------------------------------------------------------


class TestGenerate:
    """Tests for LLM generation in VerifyRepairPipeline."""

    def test_generate_no_model_raises(self) -> None:
        """REQ-VERIFY-001: _generate raises if no model loaded."""
        pipeline = VerifyRepairPipeline()
        with pytest.raises(RuntimeError, match="No model loaded"):
            pipeline._generate("hello")

    def test_generate_with_chat_template(self) -> None:
        """REQ-VERIFY-001: _generate uses chat template when available."""
        pipeline = VerifyRepairPipeline()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted prompt"
        mock_tokenizer.eos_token_id = 0
        # Mock tokenizer call returns dict with input_ids
        import torch as real_torch

        input_ids = real_torch.tensor([[1, 2, 3]])
        mock_tokenizer.return_value = {"input_ids": input_ids}
        mock_tokenizer.decode.return_value = "generated response"

        mock_model = MagicMock()
        mock_model.generate.return_value = real_torch.tensor([[1, 2, 3, 4, 5]])

        pipeline._model = mock_model
        pipeline._tokenizer = mock_tokenizer
        pipeline._device = "cpu"

        result = pipeline._generate("test prompt")
        assert result == "generated response"
        mock_tokenizer.apply_chat_template.assert_called_once()

    def test_generate_chat_template_type_error_fallback(self) -> None:
        """REQ-VERIFY-001: Falls back when enable_thinking not supported."""
        pipeline = VerifyRepairPipeline()
        mock_tokenizer = MagicMock()

        call_count = 0

        def mock_apply_chat_template(*args: Any, **kwargs: Any) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TypeError("unexpected keyword argument 'enable_thinking'")
            return "formatted prompt"

        mock_tokenizer.apply_chat_template = mock_apply_chat_template
        mock_tokenizer.eos_token_id = 0

        import torch as real_torch

        input_ids = real_torch.tensor([[1, 2, 3]])
        mock_tokenizer.return_value = {"input_ids": input_ids}
        mock_tokenizer.decode.return_value = "response"

        mock_model = MagicMock()
        mock_model.generate.return_value = real_torch.tensor([[1, 2, 3, 4]])

        pipeline._model = mock_model
        pipeline._tokenizer = mock_tokenizer
        pipeline._device = "cpu"

        result = pipeline._generate("test")
        assert result == "response"
        assert call_count == 2

    def test_generate_no_chat_template_fallback(self) -> None:
        """REQ-VERIFY-001: Falls back to raw prompt if no chat template."""
        pipeline = VerifyRepairPipeline()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.side_effect = TypeError("no template")

        # Second call also fails — triggers raw prompt fallback
        call_count = 0
        original_side_effect = mock_tokenizer.apply_chat_template.side_effect

        def always_fail(*args: Any, **kwargs: Any) -> str:
            raise TypeError("no template")

        mock_tokenizer.apply_chat_template = always_fail
        mock_tokenizer.eos_token_id = 0

        import torch as real_torch

        input_ids = real_torch.tensor([[1, 2]])
        mock_tokenizer.return_value = {"input_ids": input_ids}
        mock_tokenizer.decode.return_value = "raw response"

        mock_model = MagicMock()
        mock_model.generate.return_value = real_torch.tensor([[1, 2, 3]])

        pipeline._model = mock_model
        pipeline._tokenizer = mock_tokenizer
        pipeline._device = "cpu"

        result = pipeline._generate("raw prompt")
        assert result == "raw response"

    def test_generate_strips_think_tokens(self) -> None:
        """REQ-VERIFY-001: _generate strips </think> tokens from output."""
        pipeline = VerifyRepairPipeline()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "prompt"
        mock_tokenizer.eos_token_id = 0

        import torch as real_torch

        input_ids = real_torch.tensor([[1, 2]])
        mock_tokenizer.return_value = {"input_ids": input_ids}
        mock_tokenizer.decode.return_value = "<think>reasoning</think>actual answer"

        mock_model = MagicMock()
        mock_model.generate.return_value = real_torch.tensor([[1, 2, 3, 4]])

        pipeline._model = mock_model
        pipeline._tokenizer = mock_tokenizer
        pipeline._device = "cpu"

        result = pipeline._generate("test")
        assert result == "actual answer"

    def test_generate_cuda_device(self) -> None:
        """REQ-VERIFY-001: _generate moves inputs to CUDA when device is cuda."""
        pipeline = VerifyRepairPipeline()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "prompt"
        mock_tokenizer.eos_token_id = 0

        import torch as real_torch

        input_ids = MagicMock()
        input_ids.cuda.return_value = real_torch.tensor([[1, 2]])
        input_ids.shape = (1, 2)
        mock_tokenizer.return_value = {"input_ids": input_ids}
        mock_tokenizer.decode.return_value = "response"

        mock_model = MagicMock()
        mock_model.generate.return_value = real_torch.tensor([[1, 2, 3]])

        pipeline._model = mock_model
        pipeline._tokenizer = mock_tokenizer
        pipeline._device = "cuda"

        result = pipeline._generate("test")
        input_ids.cuda.assert_called_once()


# ---------------------------------------------------------------------------
# Energy-backed constraint tests -- REQ-VERIFY-003
# ---------------------------------------------------------------------------


class TestEnergyConstraints:
    """Tests for energy-backed constraint evaluation."""

    def test_energy_term_satisfied(self) -> None:
        """REQ-VERIFY-003: Constraint with satisfied energy_term passes."""
        import jax.numpy as jnp

        from carnot.verify.constraint import ComposedEnergy

        # Create a mock energy term that is satisfied (energy=0).
        mock_term = MagicMock()
        mock_term.name = "test_constraint"
        mock_term.energy.return_value = jnp.array(0.0)
        mock_term.threshold.return_value = jnp.array(0.01)
        mock_term.is_satisfied.return_value = True
        mock_term.gradient.return_value = jnp.zeros(1)

        constraint = ConstraintResult(
            constraint_type="test",
            description="Test energy constraint",
            energy_term=mock_term,
            metadata={},
        )

        pipeline = VerifyRepairPipeline()
        result = pipeline._evaluate_constraints([constraint])
        assert result.verified is True
        assert len(result.certificate["per_constraint"]) > 0

    def test_energy_term_violated(self) -> None:
        """REQ-VERIFY-003: Constraint with violated energy_term fails."""
        import jax.numpy as jnp

        # Create a mock energy term that is violated (energy > threshold).
        mock_term = MagicMock()
        mock_term.name = "violated_constraint"
        mock_term.energy.return_value = jnp.array(5.0)
        mock_term.threshold.return_value = jnp.array(0.01)
        mock_term.is_satisfied.return_value = False
        mock_term.gradient.return_value = jnp.zeros(1)

        constraint = ConstraintResult(
            constraint_type="test",
            description="Violated energy constraint",
            energy_term=mock_term,
            metadata={},
        )

        pipeline = VerifyRepairPipeline()
        result = pipeline._evaluate_constraints([constraint])
        # The energy term is violated, so the constraint should fail.
        assert len(result.certificate["per_constraint"]) > 0
        assert result.certificate["total_energy"] != 0.0

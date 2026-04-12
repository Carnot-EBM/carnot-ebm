"""Integration tests for the full verify-repair pipeline.

These tests exercise the real VerifyRepairPipeline with actual constraint
extraction, JAX-based energy computation, and no mocked dependencies.
They verify the pipeline end-to-end as a user would experience it.

Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, SCENARIO-VERIFY-004
"""

from __future__ import annotations

import pytest

from carnot.pipeline.errors import PipelineTimeoutError
from carnot.pipeline.extract import AutoExtractor, ConstraintResult
from carnot.pipeline.verify_repair import (
    RepairResult,
    VerificationResult,
    VerifyRepairPipeline,
)
from carnot.pipeline.z3_extractor import Z3ArithmeticExtractor


# ---------------------------------------------------------------------------
# Verify-only pipeline (no LLM model loaded)
# ---------------------------------------------------------------------------


class TestVerifyEndToEnd:
    """Full pipeline verification without mocks.

    Spec: REQ-VERIFY-001, REQ-VERIFY-003
    """

    def setup_method(self) -> None:
        """Create a real pipeline with default AutoExtractor."""
        self.pipeline = VerifyRepairPipeline(timeout_seconds=10.0)

    # -- Arithmetic domain --

    def test_correct_addition(self) -> None:
        """REQ-VERIFY-001: Correct addition verified end-to-end."""
        result = self.pipeline.verify(
            question="What is 123 + 456?",
            response="123 + 456 = 579.",
            domain="arithmetic",
        )
        assert isinstance(result, VerificationResult)
        assert result.verified is True
        assert len(result.violations) == 0

    def test_wrong_addition(self) -> None:
        """REQ-VERIFY-001: Wrong addition detected end-to-end."""
        result = self.pipeline.verify(
            question="What is 123 + 456?",
            response="123 + 456 = 580.",
            domain="arithmetic",
        )
        assert result.verified is False
        assert len(result.violations) >= 1
        assert result.violations[0].constraint_type == "arithmetic"
        assert result.violations[0].metadata["correct_result"] == 579

    def test_correct_multiplication(self) -> None:
        """REQ-VERIFY-001: Correct multiplication verified."""
        result = self.pipeline.verify(
            question="What is 7 * 8?",
            response="7 * 8 = 56.",
            domain="arithmetic",
        )
        assert result.verified is True

    def test_wrong_subtraction(self) -> None:
        """REQ-VERIFY-001: Wrong subtraction detected."""
        result = self.pipeline.verify(
            question="What is 100 - 37?",
            response="100 - 37 = 64.",
            domain="arithmetic",
        )
        assert result.verified is False
        assert any(v.metadata.get("correct_result") == 63 for v in result.violations)

    def test_multiple_arithmetic_claims(self) -> None:
        """REQ-VERIFY-001: Multiple arithmetic claims on separate lines."""
        result = self.pipeline.verify(
            question="Compute 2+3 and 4*5.",
            response="2 + 3 = 5.\n4 * 5 = 20.",
            domain="arithmetic",
        )
        assert result.verified is True
        assert len(result.constraints) >= 1

    def test_mixed_correct_and_wrong(self) -> None:
        """REQ-VERIFY-001: One correct + one wrong claim fails overall."""
        result = self.pipeline.verify(
            question="Compute 2+3 and 10-3.",
            response="2 + 3 = 5.\n10 - 3 = 8.",
            domain="arithmetic",
        )
        assert result.verified is False
        assert len(result.violations) >= 1

    def test_custom_z3_extractor_verifies_verbal_arithmetic(self) -> None:
        """REQ-VERIFY-001: Custom Z3 extractor works end-to-end through the pipeline."""
        pipeline = VerifyRepairPipeline(
            extractor=Z3ArithmeticExtractor(),
            timeout_seconds=10.0,
        )
        result = pipeline.verify(
            question="What is half of 48?",
            response="Half of 48 is 20.",
            domain="arithmetic",
        )
        assert result.verified is False
        assert len(result.violations) == 1
        assert result.violations[0].metadata["correct_result"] == 24

    # -- Code domain --

    def test_code_constraint_extraction(self) -> None:
        """REQ-VERIFY-002: Code constraints extracted from Python snippet."""
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
        # Code extractor should find type annotations and return type.
        assert len(result.constraints) >= 1

    # -- Auto-detect domain --

    def test_auto_domain_arithmetic(self) -> None:
        """REQ-VERIFY-001: AutoExtractor detects arithmetic without domain hint."""
        result = self.pipeline.verify(
            question="What is 10 + 20?",
            response="10 + 20 = 30.",
        )
        assert result.verified is True

    # -- No constraints extracted --

    def test_no_constraints_plain_text(self) -> None:
        """REQ-VERIFY-003: Plain text with no verifiable arithmetic claims."""
        result = self.pipeline.verify(
            question="Tell me about cats.",
            response="Cats are lovely animals.",
            domain="arithmetic",
        )
        assert isinstance(result, VerificationResult)
        # No arithmetic constraints means vacuously verified.
        assert result.verified is True
        assert len(result.constraints) == 0

    # -- Certificate structure --

    def test_certificate_structure(self) -> None:
        """REQ-VERIFY-003: Certificate has required keys."""
        result = self.pipeline.verify(
            question="What is 5 + 3?",
            response="5 + 3 = 8.",
            domain="arithmetic",
        )
        cert = result.certificate
        assert "total_energy" in cert
        assert "per_constraint" in cert
        assert "n_constraints" in cert
        assert "n_violations" in cert


# ---------------------------------------------------------------------------
# Extract constraints (standalone)
# ---------------------------------------------------------------------------


class TestExtractConstraints:
    """Test extract_constraints() method end-to-end.

    Spec: REQ-VERIFY-001
    """

    def setup_method(self) -> None:
        self.pipeline = VerifyRepairPipeline()

    def test_extract_arithmetic(self) -> None:
        """REQ-VERIFY-001: Extract arithmetic constraints."""
        constraints = self.pipeline.extract_constraints(
            "The sum is 3 + 4 = 7.", domain="arithmetic"
        )
        assert len(constraints) >= 1
        assert all(isinstance(c, ConstraintResult) for c in constraints)
        assert constraints[0].constraint_type == "arithmetic"

    def test_extract_code(self) -> None:
        """REQ-VERIFY-002: Extract code constraints."""
        code = """```python
def greet(name: str) -> str:
    return f"Hello, {name}!"
```"""
        constraints = self.pipeline.extract_constraints(code, domain="code")
        assert len(constraints) >= 1

    def test_extract_no_domain(self) -> None:
        """REQ-VERIFY-001: Extract with auto-detection."""
        constraints = self.pipeline.extract_constraints("7 + 6 = 13.")
        assert len(constraints) >= 1


# ---------------------------------------------------------------------------
# Verify-and-repair without model (verify-only fallback)
# ---------------------------------------------------------------------------


class TestVerifyAndRepairNoModel:
    """Test verify_and_repair() in verify-only mode (no LLM).

    Spec: REQ-VERIFY-003, SCENARIO-VERIFY-004
    """

    def setup_method(self) -> None:
        self.pipeline = VerifyRepairPipeline(timeout_seconds=10.0)

    def test_repair_no_model_correct_response(self) -> None:
        """SCENARIO-VERIFY-004: Correct response needs no repair."""
        result = self.pipeline.verify_and_repair(
            question="What is 2 + 2?",
            response="2 + 2 = 4.",
            domain="arithmetic",
        )
        assert isinstance(result, RepairResult)
        assert result.verified is True
        assert result.repaired is False
        assert result.iterations == 0
        assert result.initial_response == "2 + 2 = 4."
        assert result.final_response == "2 + 2 = 4."
        assert len(result.history) == 1

    def test_repair_no_model_wrong_response(self) -> None:
        """SCENARIO-VERIFY-004: Wrong response, no model = unrepaired."""
        result = self.pipeline.verify_and_repair(
            question="What is 2 + 2?",
            response="2 + 2 = 5.",
            domain="arithmetic",
        )
        assert result.verified is False
        assert result.repaired is False
        assert result.iterations == 0
        assert len(result.history) == 1

    def test_repair_no_model_no_response_raises(self) -> None:
        """SCENARIO-VERIFY-004: No response + no model = ValueError."""
        with pytest.raises(ValueError, match="No response provided"):
            self.pipeline.verify_and_repair(question="What is 1+1?")

    def test_repair_history_tracks_iterations(self) -> None:
        """SCENARIO-VERIFY-004: History has one entry for verify-only."""
        result = self.pipeline.verify_and_repair(
            question="What is 9 * 9?",
            response="9 * 9 = 81.",
            domain="arithmetic",
        )
        assert len(result.history) == 1
        assert result.history[0].verified is True


# ---------------------------------------------------------------------------
# Pipeline with domain filtering
# ---------------------------------------------------------------------------


class TestDomainFiltering:
    """Test pipeline with explicit domain configuration.

    Spec: REQ-VERIFY-001
    """

    def test_single_domain_filter(self) -> None:
        """REQ-VERIFY-001: Pipeline restricted to arithmetic domain."""
        pipeline = VerifyRepairPipeline(domains=["arithmetic"])
        result = pipeline.verify(
            question="What is 3 + 3?",
            response="3 + 3 = 6.",
        )
        assert result.verified is True
        # All constraints should be arithmetic type.
        for c in result.constraints:
            assert c.constraint_type == "arithmetic"

    def test_multi_domain_filter(self) -> None:
        """REQ-VERIFY-001: Pipeline with multiple domains merges results."""
        pipeline = VerifyRepairPipeline(domains=["arithmetic", "code"])
        result = pipeline.verify(
            question="Show 2+2 and code.",
            response="2 + 2 = 4.",
        )
        assert result.verified is True


# ---------------------------------------------------------------------------
# Timeout behavior
# ---------------------------------------------------------------------------


class TestPipelineTimeout:
    """Test that timeout configuration works.

    Spec: REQ-VERIFY-003
    """

    def test_generous_timeout_succeeds(self) -> None:
        """REQ-VERIFY-003: Normal operation within timeout budget."""
        pipeline = VerifyRepairPipeline(timeout_seconds=60.0)
        result = pipeline.verify(
            question="What is 1 + 1?",
            response="1 + 1 = 2.",
            domain="arithmetic",
        )
        assert result.verified is True

    def test_has_model_property(self) -> None:
        """REQ-VERIFY-001: has_model is False when no model loaded."""
        pipeline = VerifyRepairPipeline()
        assert pipeline.has_model is False

"""Tests for carnot.pipeline.errors and pipeline error handling / hardening.

Each test references REQ-VERIFY-* or SCENARIO-VERIFY-* per spec-anchored
development requirements. Covers: error hierarchy, empty input, very long
input, unicode, no constraints, malformed code, bad model name, concurrent
calls, and configurable timeout.

Spec: REQ-VERIFY-001, REQ-VERIFY-003, SCENARIO-VERIFY-004
"""

from __future__ import annotations

import concurrent.futures
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from carnot.pipeline.errors import (
    CarnotError,
    ExtractionError,
    ModelLoadError,
    PipelineTimeoutError,
    RepairError,
    VerificationError,
)
from carnot.pipeline.extract import ConstraintExtractor, ConstraintResult
from carnot.pipeline.verify_repair import (
    RepairResult,
    VerificationResult,
    VerifyRepairPipeline,
)


# ---------------------------------------------------------------------------
# Error hierarchy tests -- REQ-VERIFY-003
# ---------------------------------------------------------------------------


class TestErrorHierarchy:
    """Verify the exception class hierarchy and attributes."""

    def test_all_errors_inherit_from_carnot_error(self) -> None:
        """REQ-VERIFY-003: All pipeline errors are catchable via CarnotError."""
        for cls in (
            ExtractionError,
            VerificationError,
            RepairError,
            ModelLoadError,
            PipelineTimeoutError,
        ):
            assert issubclass(cls, CarnotError)
            assert issubclass(cls, Exception)

    def test_carnot_error_details(self) -> None:
        """REQ-VERIFY-003: CarnotError carries structured details dict."""
        err = CarnotError("boom", details={"key": "value"})
        assert str(err) == "boom"
        assert err.details == {"key": "value"}

    def test_carnot_error_default_details(self) -> None:
        """REQ-VERIFY-003: CarnotError defaults to empty details."""
        err = CarnotError("boom")
        assert err.details == {}

    def test_extraction_error(self) -> None:
        """REQ-VERIFY-001: ExtractionError is a CarnotError."""
        err = ExtractionError("bad input", details={"domain": "code"})
        assert isinstance(err, CarnotError)
        assert err.details["domain"] == "code"

    def test_verification_error(self) -> None:
        """REQ-VERIFY-003: VerificationError is a CarnotError."""
        err = VerificationError("JAX failed")
        assert isinstance(err, CarnotError)

    def test_repair_error(self) -> None:
        """REQ-VERIFY-003: RepairError is a CarnotError."""
        err = RepairError("generation failed")
        assert isinstance(err, CarnotError)

    def test_model_load_error(self) -> None:
        """REQ-VERIFY-001: ModelLoadError is a CarnotError."""
        err = ModelLoadError("model not found")
        assert isinstance(err, CarnotError)

    def test_pipeline_timeout_error(self) -> None:
        """REQ-VERIFY-003: PipelineTimeoutError is a CarnotError."""
        err = PipelineTimeoutError("timed out")
        assert isinstance(err, CarnotError)

    def test_pipeline_timeout_error_not_builtin(self) -> None:
        """REQ-VERIFY-003: PipelineTimeoutError does NOT shadow builtins."""
        assert not issubclass(PipelineTimeoutError, TimeoutError)


# ---------------------------------------------------------------------------
# Empty input tests -- REQ-VERIFY-003
# ---------------------------------------------------------------------------


class TestEmptyInput:
    """Edge cases with empty or missing input."""

    def test_empty_question_and_response(self) -> None:
        """REQ-VERIFY-003: Both empty strings produce no constraints."""
        pipeline = VerifyRepairPipeline(timeout_seconds=0)
        result = pipeline.verify(question="", response="")
        assert result.verified is True
        assert len(result.constraints) == 0

    def test_empty_response_all_domains(self) -> None:
        """REQ-VERIFY-003: Empty response across all domains."""
        for domain in ["arithmetic", "code", "logic", "nl"]:
            pipeline = VerifyRepairPipeline(timeout_seconds=0)
            result = pipeline.verify(
                question="test", response="", domain=domain
            )
            assert result.verified is True
            assert len(result.constraints) == 0

    def test_whitespace_only_response(self) -> None:
        """REQ-VERIFY-003: Whitespace-only response produces no constraints."""
        pipeline = VerifyRepairPipeline(timeout_seconds=0)
        result = pipeline.verify(
            question="test", response="   \n\t  \n  "
        )
        assert result.verified is True

    def test_extract_constraints_empty_text(self) -> None:
        """REQ-VERIFY-001: extract_constraints on empty string returns []."""
        pipeline = VerifyRepairPipeline(timeout_seconds=0)
        assert pipeline.extract_constraints("") == []

    def test_verify_and_repair_empty_response(self) -> None:
        """REQ-VERIFY-003: verify_and_repair with empty response string."""
        pipeline = VerifyRepairPipeline(timeout_seconds=0)
        result = pipeline.verify_and_repair(
            question="What is 2+2?", response=""
        )
        assert isinstance(result, RepairResult)
        assert result.verified is True  # no constraints = vacuously true


# ---------------------------------------------------------------------------
# Very long input tests -- REQ-VERIFY-003
# ---------------------------------------------------------------------------


class TestVeryLongInput:
    """Pipeline handles very long inputs without crashing."""

    def test_long_arithmetic_response(self) -> None:
        """REQ-VERIFY-003: Long response with many arithmetic claims."""
        # 1000 correct arithmetic claims.
        claims = " ".join(f"{i} + 1 = {i + 1}." for i in range(1000))
        pipeline = VerifyRepairPipeline(timeout_seconds=0)
        result = pipeline.verify(
            question="sums", response=claims, domain="arithmetic"
        )
        assert result.verified is True
        assert len(result.constraints) == 1000

    def test_long_nonsense_text(self) -> None:
        """REQ-VERIFY-003: Long text with no extractable constraints doesn't crash."""
        text = "xyzzy " * 50000
        pipeline = VerifyRepairPipeline(timeout_seconds=0)
        result = pipeline.verify(question="test", response=text)
        # May or may not extract constraints depending on NL patterns,
        # but must not crash or hang.
        assert isinstance(result, VerificationResult)


# ---------------------------------------------------------------------------
# Unicode input tests -- REQ-VERIFY-003
# ---------------------------------------------------------------------------


class TestUnicodeInput:
    """Pipeline handles unicode without crashing."""

    def test_unicode_response(self) -> None:
        """REQ-VERIFY-003: Unicode text does not crash extraction."""
        pipeline = VerifyRepairPipeline(timeout_seconds=0)
        result = pipeline.verify(
            question="What is pi?",
            response="The value of \u03c0 is approximately 3.14159. \u2603 \U0001f600",
        )
        assert isinstance(result, VerificationResult)

    def test_unicode_arithmetic(self) -> None:
        """REQ-VERIFY-003: Arithmetic with unicode context still works."""
        pipeline = VerifyRepairPipeline(timeout_seconds=0)
        result = pipeline.verify(
            question="\u8ba1\u7b97",
            response="The answer is 3 + 4 = 7. \u2714",
            domain="arithmetic",
        )
        assert result.verified is True

    def test_unicode_code_block(self) -> None:
        """REQ-VERIFY-003: Code with unicode strings parses correctly."""
        pipeline = VerifyRepairPipeline(timeout_seconds=0)
        code = '```python\ndef greet(name: str) -> str:\n    return "\u4f60\u597d " + name\n```'
        result = pipeline.verify(
            question="Write greet", response=code, domain="code"
        )
        assert isinstance(result, VerificationResult)
        assert len(result.constraints) > 0


# ---------------------------------------------------------------------------
# No constraints tests -- REQ-VERIFY-003
# ---------------------------------------------------------------------------


class TestNoConstraints:
    """Pipeline behavior when no constraints can be extracted."""

    def test_prose_only(self) -> None:
        """REQ-VERIFY-003: Pure prose text has no extractable constraints."""
        pipeline = VerifyRepairPipeline(timeout_seconds=0)
        result = pipeline.verify(
            question="Tell me about cats.",
            response="Cats are wonderful animals that have been "
            "domesticated for thousands of years.",
        )
        # "cats are wonderful..." may match NL "is/are" pattern,
        # but those are informational (no satisfied key), so verified=True.
        assert result.verified is True

    def test_no_constraints_certificate(self) -> None:
        """REQ-VERIFY-003: Certificate records zero constraints."""
        pipeline = VerifyRepairPipeline(timeout_seconds=0)
        result = pipeline.verify(
            question="Tell me a joke.",
            response="Knock knock.",
            domain="arithmetic",
        )
        assert result.certificate["n_constraints"] == 0
        assert result.certificate["n_violations"] == 0


# ---------------------------------------------------------------------------
# Malformed code tests -- REQ-VERIFY-001
# ---------------------------------------------------------------------------


class TestMalformedCode:
    """Pipeline handles syntactically invalid code gracefully."""

    def test_syntax_error_in_code_block(self) -> None:
        """REQ-VERIFY-001: Malformed Python code in fenced block."""
        pipeline = VerifyRepairPipeline(timeout_seconds=0)
        code = "```python\ndef broken(\n    return nope\n```"
        result = pipeline.verify(
            question="Write a function", response=code, domain="code"
        )
        # CodeExtractor catches SyntaxError and returns [].
        assert isinstance(result, VerificationResult)
        assert result.verified is True  # no constraints = vacuously true

    def test_raw_malformed_code(self) -> None:
        """REQ-VERIFY-001: Malformed code without fence markers."""
        pipeline = VerifyRepairPipeline(timeout_seconds=0)
        result = pipeline.verify(
            question="Write code",
            response="def broken(:\n  if while for",
            domain="code",
        )
        assert result.verified is True

    def test_partially_valid_code(self) -> None:
        """REQ-VERIFY-001: Mix of valid and invalid code blocks."""
        pipeline = VerifyRepairPipeline(timeout_seconds=0)
        response = (
            "```python\ndef good(x: int) -> int:\n    return x\n```\n"
            "```python\ndef bad(\n    return\n```"
        )
        result = pipeline.verify(
            question="Write functions", response=response, domain="code"
        )
        # The valid block produces constraints; the invalid one is skipped.
        assert len(result.constraints) > 0
        assert result.verified is True


# ---------------------------------------------------------------------------
# Bad model name tests -- REQ-VERIFY-001
# ---------------------------------------------------------------------------


class TestBadModelName:
    """Pipeline handles model loading failures gracefully."""

    def test_bad_model_raises_model_load_error(self) -> None:
        """REQ-VERIFY-001: Non-existent model raises ModelLoadError."""
        with patch.object(
            VerifyRepairPipeline,
            "_load_model",
            side_effect=ModelLoadError(
                "Failed to load model 'nonexistent-model-xyz-12345': not found",
                details={"model_name": "nonexistent-model-xyz-12345"},
            ),
        ):
            with pytest.raises(ModelLoadError, match="Failed to load"):
                VerifyRepairPipeline(model="nonexistent-model-xyz-12345")

    def test_model_load_error_has_details(self) -> None:
        """REQ-VERIFY-001: ModelLoadError carries model_name in details."""
        with patch.object(
            VerifyRepairPipeline,
            "_load_model",
            side_effect=ModelLoadError(
                "Failed to load model 'nonexistent-model-xyz-12345': not found",
                details={"model_name": "nonexistent-model-xyz-12345"},
            ),
        ):
            with pytest.raises(ModelLoadError) as exc_info:
                VerifyRepairPipeline(model="nonexistent-model-xyz-12345")
            assert exc_info.value.details["model_name"] == "nonexistent-model-xyz-12345"

    def test_model_load_error_is_carnot_error(self) -> None:
        """REQ-VERIFY-001: ModelLoadError catchable as CarnotError."""
        with patch.object(
            VerifyRepairPipeline,
            "_load_model",
            side_effect=ModelLoadError(
                "Failed to load model 'nonexistent-model-xyz-12345': not found",
                details={"model_name": "nonexistent-model-xyz-12345"},
            ),
        ):
            with pytest.raises(CarnotError):
                VerifyRepairPipeline(model="nonexistent-model-xyz-12345")

    def test_import_error_raises_model_load_error(self) -> None:
        """REQ-VERIFY-001: Missing torch/transformers raises ModelLoadError."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "torch":
                raise ImportError("No module named 'torch'")
            return real_import(name, *args, **kwargs)

        pipeline = VerifyRepairPipeline(timeout_seconds=0)
        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ModelLoadError, match="Required packages"):
                pipeline._load_model("some-model")

    def test_load_model_wraps_exception(self) -> None:
        """REQ-VERIFY-001: _load_model wraps load-time exceptions."""
        import sys

        pipeline = VerifyRepairPipeline(timeout_seconds=0)
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        mock_tokenizer_cls = MagicMock()
        mock_tokenizer_cls.from_pretrained.side_effect = OSError("not found")
        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer = mock_tokenizer_cls
        mock_transformers.AutoModelForCausalLM = MagicMock()

        with patch.dict(
            sys.modules,
            {"torch": mock_torch, "transformers": mock_transformers},
        ):
            with pytest.raises(ModelLoadError, match="Failed to load"):
                pipeline._load_model("bad-model")


# ---------------------------------------------------------------------------
# Configurable timeout tests -- REQ-VERIFY-003
# ---------------------------------------------------------------------------


class TestTimeout:
    """Pipeline timeout enforcement."""

    def test_default_timeout(self) -> None:
        """REQ-VERIFY-003: Default timeout is 30 seconds."""
        pipeline = VerifyRepairPipeline()
        assert pipeline._timeout_seconds == 30.0

    def test_custom_timeout(self) -> None:
        """REQ-VERIFY-003: Custom timeout is stored correctly."""
        pipeline = VerifyRepairPipeline(timeout_seconds=5.0)
        assert pipeline._timeout_seconds == 5.0

    def test_disabled_timeout_zero(self) -> None:
        """REQ-VERIFY-003: timeout_seconds=0 disables timeout."""
        pipeline = VerifyRepairPipeline(timeout_seconds=0)
        assert pipeline._timeout_seconds == 0.0

    def test_disabled_timeout_none(self) -> None:
        """REQ-VERIFY-003: timeout_seconds=None disables timeout."""
        pipeline = VerifyRepairPipeline(timeout_seconds=None)  # type: ignore[arg-type]
        assert pipeline._timeout_seconds == 0.0

    def test_timeout_raises_on_slow_verify(self) -> None:
        """REQ-VERIFY-003: Verify raises PipelineTimeoutError when deadline passes."""

        class SlowExtractor:
            """Extractor that simulates slow extraction by sleeping."""

            @property
            def supported_domains(self) -> list[str]:
                return ["slow"]

            def extract(
                self, text: str, domain: str | None = None
            ) -> list[ConstraintResult]:
                time.sleep(0.3)
                return []

        # Timeout of 0.05s with a 0.3s extractor should timeout.
        pipeline = VerifyRepairPipeline(
            extractor=SlowExtractor(),  # type: ignore[arg-type]
            timeout_seconds=0.05,
        )
        with pytest.raises(PipelineTimeoutError):
            pipeline.verify(
                question="test", response="test", domain="slow"
            )

    def test_no_timeout_when_fast(self) -> None:
        """REQ-VERIFY-003: Fast operations complete without timeout."""
        pipeline = VerifyRepairPipeline(timeout_seconds=30.0)
        result = pipeline.verify(
            question="What is 1+1?",
            response="1 + 1 = 2.",
            domain="arithmetic",
        )
        assert result.verified is True


# ---------------------------------------------------------------------------
# Extraction error handling tests -- REQ-VERIFY-001
# ---------------------------------------------------------------------------


class TestExtractionErrorHandling:
    """Pipeline degrades gracefully when extraction fails."""

    def test_broken_extractor_degrades_verify(self) -> None:
        """REQ-VERIFY-001: Broken extractor causes degraded VerificationResult."""

        class BrokenExtractor:
            @property
            def supported_domains(self) -> list[str]:
                return ["broken"]

            def extract(
                self, text: str, domain: str | None = None
            ) -> list[ConstraintResult]:
                raise RuntimeError("extractor exploded")

        pipeline = VerifyRepairPipeline(
            extractor=BrokenExtractor(),  # type: ignore[arg-type]
            timeout_seconds=0,
        )
        # verify() catches ExtractionError and returns degraded result.
        result = pipeline.verify(
            question="test", response="test", domain="broken"
        )
        assert result.verified is False
        assert "error" in result.certificate

    def test_extraction_error_direct(self) -> None:
        """REQ-VERIFY-001: extract_constraints raises ExtractionError."""

        class BrokenExtractor:
            @property
            def supported_domains(self) -> list[str]:
                return ["broken"]

            def extract(
                self, text: str, domain: str | None = None
            ) -> list[ConstraintResult]:
                raise RuntimeError("kaboom")

        pipeline = VerifyRepairPipeline(
            extractor=BrokenExtractor(),  # type: ignore[arg-type]
            timeout_seconds=0,
        )
        with pytest.raises(ExtractionError, match="kaboom"):
            pipeline.extract_constraints("test")


# ---------------------------------------------------------------------------
# Repair error handling tests -- SCENARIO-VERIFY-004
# ---------------------------------------------------------------------------


class TestRepairErrorHandling:
    """Pipeline handles repair-time failures gracefully."""

    def test_generation_failure_returns_best_so_far(self) -> None:
        """SCENARIO-VERIFY-004: Generation failure returns best response."""
        pipeline = VerifyRepairPipeline(max_repairs=3, timeout_seconds=0)
        pipeline._model = MagicMock()
        pipeline._tokenizer = MagicMock()

        call_count = 0

        def failing_generate(prompt: str, max_new_tokens: int = 256) -> str:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("GPU OOM")

        pipeline._generate = failing_generate  # type: ignore[assignment]

        result = pipeline.verify_and_repair(
            question="What is 10 + 5?",
            response="10 + 5 = 16.",
            domain="arithmetic",
        )
        # Should return gracefully with the original wrong response.
        assert isinstance(result, RepairResult)
        assert result.verified is False
        assert result.final_response == "10 + 5 = 16."


# ---------------------------------------------------------------------------
# Concurrent call tests -- REQ-VERIFY-003
# ---------------------------------------------------------------------------


class TestConcurrentCalls:
    """Pipeline is safe to call from multiple threads."""

    def test_concurrent_verify_calls(self) -> None:
        """REQ-VERIFY-003: Multiple concurrent verify calls don't crash."""
        pipeline = VerifyRepairPipeline(timeout_seconds=0)

        def do_verify(i: int) -> VerificationResult:
            return pipeline.verify(
                question=f"What is {i} + 1?",
                response=f"{i} + 1 = {i + 1}.",
                domain="arithmetic",
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(do_verify, i) for i in range(20)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 20
        assert all(r.verified is True for r in results)

    def test_concurrent_extract_calls(self) -> None:
        """REQ-VERIFY-003: Multiple concurrent extract calls don't crash."""
        pipeline = VerifyRepairPipeline(timeout_seconds=0)

        def do_extract(i: int) -> list[ConstraintResult]:
            return pipeline.extract_constraints(
                f"{i} + 1 = {i + 1}", domain="arithmetic"
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(do_extract, i) for i in range(20)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 20
        assert all(len(r) == 1 for r in results)


# ---------------------------------------------------------------------------
# Verify-and-repair with timeout -- REQ-VERIFY-003
# ---------------------------------------------------------------------------


class TestVerifyAndRepairTimeout:
    """Timeout applies to the full verify_and_repair loop."""

    def test_repair_loop_timeout(self) -> None:
        """REQ-VERIFY-003: Repair loop that exceeds timeout raises."""
        pipeline = VerifyRepairPipeline(
            max_repairs=10, timeout_seconds=0.05
        )
        pipeline._model = MagicMock()
        pipeline._tokenizer = MagicMock()

        def slow_generate(prompt: str, max_new_tokens: int = 256) -> str:
            time.sleep(0.1)  # Each call takes 100ms.
            return "10 + 5 = 99."  # Always wrong, triggers more repairs.

        pipeline._generate = slow_generate  # type: ignore[assignment]

        with pytest.raises(PipelineTimeoutError):
            pipeline.verify_and_repair(
                question="What is 10 + 5?",
                response="10 + 5 = 16.",
                domain="arithmetic",
            )


# ---------------------------------------------------------------------------
# Internal branch coverage tests -- REQ-VERIFY-003
# ---------------------------------------------------------------------------


class TestInternalBranches:
    """Cover internal error-handling branches for 100% coverage."""

    def test_extract_constraints_reraises_carnot_error(self) -> None:
        """REQ-VERIFY-003: extract_constraints re-raises CarnotError subclasses."""

        class ErrorExtractor:
            @property
            def supported_domains(self) -> list[str]:
                return ["err"]

            def extract(
                self, text: str, domain: str | None = None
            ) -> list[ConstraintResult]:
                raise ExtractionError("already wrapped")

        pipeline = VerifyRepairPipeline(
            extractor=ErrorExtractor(),  # type: ignore[arg-type]
            timeout_seconds=0,
        )
        with pytest.raises(ExtractionError, match="already wrapped"):
            pipeline.extract_constraints("test")

    def test_verify_and_repair_initial_gen_carnot_error(self) -> None:
        """REQ-VERIFY-003: CarnotError during initial generation is re-raised."""
        pipeline = VerifyRepairPipeline(timeout_seconds=0)
        pipeline._model = MagicMock()
        pipeline._tokenizer = MagicMock()

        def carnot_generate(prompt: str, max_new_tokens: int = 256) -> str:
            raise RepairError("model crashed")

        pipeline._generate = carnot_generate  # type: ignore[assignment]

        with pytest.raises(RepairError, match="model crashed"):
            pipeline.verify_and_repair(question="What is 2+2?")

    def test_verify_and_repair_initial_gen_generic_error(self) -> None:
        """REQ-VERIFY-003: Generic error during initial generation wraps as RepairError."""
        pipeline = VerifyRepairPipeline(timeout_seconds=0)
        pipeline._model = MagicMock()
        pipeline._tokenizer = MagicMock()

        def failing_generate(prompt: str, max_new_tokens: int = 256) -> str:
            raise ValueError("tokenizer failed")

        pipeline._generate = failing_generate  # type: ignore[assignment]

        with pytest.raises(RepairError, match="Initial generation failed"):
            pipeline.verify_and_repair(question="What is 2+2?")

    def test_repair_loop_reraises_carnot_error(self) -> None:
        """REQ-VERIFY-003: CarnotError during repair iteration is re-raised."""
        pipeline = VerifyRepairPipeline(max_repairs=3, timeout_seconds=0)
        pipeline._model = MagicMock()
        pipeline._tokenizer = MagicMock()

        def carnot_generate(prompt: str, max_new_tokens: int = 256) -> str:
            raise RepairError("GPU exploded")

        pipeline._generate = carnot_generate  # type: ignore[assignment]

        with pytest.raises(RepairError, match="GPU exploded"):
            pipeline.verify_and_repair(
                question="What is 10 + 5?",
                response="10 + 5 = 16.",
                domain="arithmetic",
            )

    def test_evaluate_constraints_jax_import_error(self) -> None:
        """REQ-VERIFY-003: Missing JAX raises VerificationError via verify()."""
        import sys

        pipeline = VerifyRepairPipeline(timeout_seconds=0)

        # Create a constraint with an energy_term to trigger the JAX path.
        cr = ConstraintResult(
            constraint_type="test",
            description="test constraint",
            metadata={"satisfied": True},
        )
        cr.energy_term = MagicMock()
        cr.energy_term.name = "test_term"

        import builtins

        real_import = builtins.__import__

        def block_jax(name: str, *args: Any, **kwargs: Any) -> Any:
            if "jax" in name:
                raise ImportError("No module named 'jax'")
            return real_import(name, *args, **kwargs)

        # Remove jax from sys.modules temporarily to force re-import.
        saved_modules = {}
        jax_keys = [k for k in sys.modules if k.startswith("jax")]
        for k in jax_keys:
            saved_modules[k] = sys.modules.pop(k)
        # Also remove carnot.verify.constraint to force re-import.
        constraint_keys = [k for k in sys.modules if "carnot.verify.constraint" in k]
        for k in constraint_keys:
            saved_modules[k] = sys.modules.pop(k)

        try:
            with patch("builtins.__import__", side_effect=block_jax):
                with pytest.raises(VerificationError, match="JAX not available"):
                    pipeline._evaluate_constraints([cr])
        finally:
            sys.modules.update(saved_modules)

    def test_evaluate_constraints_energy_computation_error(self) -> None:
        """REQ-VERIFY-003: Energy computation failure raises VerificationError."""
        pipeline = VerifyRepairPipeline(timeout_seconds=0)

        mock_term = MagicMock()
        mock_term.name = "broken_term"

        cr = ConstraintResult(
            constraint_type="test",
            description="broken constraint",
            metadata={},
        )
        cr.energy_term = mock_term

        # Patch ComposedEnergy to raise during verify.
        with patch(
            "carnot.verify.constraint.ComposedEnergy"
        ) as mock_composed_cls:
            mock_composed = MagicMock()
            mock_composed.verify.side_effect = RuntimeError("NaN energy")
            mock_composed_cls.return_value = mock_composed
            with pytest.raises(VerificationError, match="Energy computation failed"):
                pipeline._evaluate_constraints([cr])

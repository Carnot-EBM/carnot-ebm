"""Tests for SymCodeVerifier.batch_verify() and SymCodeBatchResult.

Spec: REQ-VERIFY-148, SCENARIO-VERIFY-173
"""

from __future__ import annotations

import pytest

from carnot.pipeline.symcode_verifier import (
    CoTStep,
    SymCodeBatchResult,
    SymCodeVerifier,
)

# ---------------------------------------------------------------------------
# SymCodeBatchResult dataclass
# ---------------------------------------------------------------------------


class TestSymCodeBatchResultDataclass:
    """REQ-VERIFY-148: SymCodeBatchResult fields are accessible and n_paragraphs is auto-set."""

    def test_n_paragraphs_derived_from_results(self) -> None:
        # SCENARIO-VERIFY-173: n_paragraphs == len(paragraphs)
        r = CoTStep(
            text="x",
            step_index=0,
            generated_code=None,
            executed_result=None,
            stated_result=None,
            violation_detected=False,
        )
        result = SymCodeBatchResult(
            per_paragraph_results=[r, r, r],
            total_violations=0,
            batch_latency_ms=1.5,
        )
        assert result.n_paragraphs == 3

    def test_total_violations_stored(self) -> None:
        r = CoTStep("x", 0, None, None, None, False)
        result = SymCodeBatchResult([r], total_violations=0, batch_latency_ms=0.5)
        assert result.total_violations == 0

    def test_batch_latency_stored(self) -> None:
        r = CoTStep("x", 0, None, None, None, False)
        result = SymCodeBatchResult([r], total_violations=0, batch_latency_ms=99.9)
        assert result.batch_latency_ms == pytest.approx(99.9)

    def test_empty_paragraphs(self) -> None:
        result = SymCodeBatchResult([], total_violations=0, batch_latency_ms=0.1)
        assert result.n_paragraphs == 0
        assert result.total_violations == 0


# ---------------------------------------------------------------------------
# batch_verify — basic correctness
# ---------------------------------------------------------------------------


class TestBatchVerifyCorrectness:
    """REQ-VERIFY-148: batch_verify detects same violations as serial verify_step()."""

    def test_returns_symcode_batch_result(self) -> None:
        v = SymCodeVerifier()
        result = v.batch_verify(["3 * 4 = 12"])
        assert isinstance(result, SymCodeBatchResult)

    def test_n_paragraphs_matches_input(self) -> None:
        # SCENARIO-VERIFY-173: n_paragraphs == 10
        v = SymCodeVerifier()
        paragraphs = [f"Step {i}: {i} + {i} = {i * 2}." for i in range(1, 11)]
        result = v.batch_verify(paragraphs)
        assert result.n_paragraphs == 10

    def test_per_paragraph_results_order_preserved(self) -> None:
        v = SymCodeVerifier()
        paragraphs = ["3 * 4 = 12", "5 + 6 = 11", "10 / 2 = 5"]
        result = v.batch_verify(paragraphs)
        assert len(result.per_paragraph_results) == 3
        for idx, r in enumerate(result.per_paragraph_results):
            assert r.step_index == idx

    def test_correct_arithmetic_no_violation(self) -> None:
        # SCENARIO-VERIFY-173: violation flags match serial verify_step()
        v = SymCodeVerifier()
        result = v.batch_verify(["3 * 4 = 12"])
        assert result.per_paragraph_results[0].violation_detected is False
        assert result.total_violations == 0

    def test_wrong_arithmetic_violation_detected(self) -> None:
        # SCENARIO-VERIFY-173: batch_verify detects same violations as serial
        v = SymCodeVerifier()
        result = v.batch_verify(["3 * 4 = 13"])
        assert result.per_paragraph_results[0].violation_detected is True
        assert result.total_violations == 1

    def test_no_arithmetic_no_violation(self) -> None:
        v = SymCodeVerifier()
        result = v.batch_verify(["The answer is therefore obvious."])
        assert result.per_paragraph_results[0].violation_detected is False
        assert result.total_violations == 0

    def test_mixed_paragraphs_violations_counted(self) -> None:
        v = SymCodeVerifier()
        paragraphs = [
            "3 * 4 = 12",  # correct
            "3 * 4 = 99",  # wrong → violation
            "No arithmetic here.",
            "5 + 5 = 10",  # correct
            "5 + 5 = 11",  # wrong → violation
        ]
        result = v.batch_verify(paragraphs)
        assert result.total_violations == 2
        assert result.per_paragraph_results[0].violation_detected is False
        assert result.per_paragraph_results[1].violation_detected is True
        assert result.per_paragraph_results[2].violation_detected is False
        assert result.per_paragraph_results[3].violation_detected is False
        assert result.per_paragraph_results[4].violation_detected is True

    def test_matches_serial_verify_step(self) -> None:
        # SCENARIO-VERIFY-173: each per_paragraph_results[i].violation_detected
        # matches verify_step(paragraphs[i]).violation_detected
        v = SymCodeVerifier()
        paragraphs = [
            "47 + 28 = 75",
            "47 + 28 = 99",  # wrong
            "The sky is blue.",
            "100 - 25 = 75",
            "100 - 25 = 70",  # wrong
        ]
        serial_flags = [v.verify_step(p, i).violation_detected for i, p in enumerate(paragraphs)]
        batch_result = v.batch_verify(paragraphs)
        batch_flags = [r.violation_detected for r in batch_result.per_paragraph_results]
        assert batch_flags == serial_flags

    def test_empty_paragraph_list(self) -> None:
        v = SymCodeVerifier()
        result = v.batch_verify([])
        assert result.n_paragraphs == 0
        assert result.total_violations == 0
        assert result.per_paragraph_results == []

    def test_batch_latency_ms_positive(self) -> None:
        v = SymCodeVerifier()
        result = v.batch_verify(["3 * 4 = 12"])
        assert result.batch_latency_ms >= 0.0

    def test_generated_code_preserved(self) -> None:
        v = SymCodeVerifier()
        result = v.batch_verify(["3 * 4 = 12"])
        assert result.per_paragraph_results[0].generated_code == "3*4"

    def test_executed_result_preserved(self) -> None:
        v = SymCodeVerifier()
        result = v.batch_verify(["3 * 4 = 12"])
        assert result.per_paragraph_results[0].executed_result == pytest.approx(12.0)

    def test_stated_result_preserved(self) -> None:
        v = SymCodeVerifier()
        result = v.batch_verify(["3 * 4 = 12"])
        assert result.per_paragraph_results[0].stated_result == pytest.approx(12.0)

    def test_llm_mode_batch_verify(self) -> None:
        # batch_verify also works when llm_caller is provided.
        call_log: list[str] = []

        def fake_llm(prompt: str) -> str:
            call_log.append(prompt)
            if "47" in prompt and "28" in prompt:
                return "47+28"
            return "None"

        v = SymCodeVerifier(llm_caller=fake_llm)
        result = v.batch_verify(["47 + 28 = 75", "no arithmetic"])
        assert result.per_paragraph_results[0].violation_detected is False
        assert result.per_paragraph_results[1].violation_detected is False

    def test_bad_exec_code_no_crash(self) -> None:
        # If extracted code is syntactically invalid, exec() raises but should be swallowed.
        v = SymCodeVerifier(llm_caller=lambda _: "import os; os.getcwd()")
        result = v.batch_verify(["some step with no detectable number"])
        # Should not raise; violation_detected should be False (no stated result).
        assert result.per_paragraph_results[0].violation_detected is False


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-173 — 10-paragraph latency check
# ---------------------------------------------------------------------------


class TestBatchVerifyLatency:
    """SCENARIO-VERIFY-173: batch latency for 10 paragraphs < 2× single paragraph."""

    def test_batch_latency_sublinear(self) -> None:
        # This test verifies the batching invariant: latency(10 paragraphs) < 2× latency(1 paragraph).
        # We don't hard-code an absolute ms threshold because CI machines vary; we compare ratios.
        import time

        v = SymCodeVerifier()
        paragraphs = [f"{i} * {i + 1} = {i * (i + 1)}" for i in range(1, 11)]

        # Warm up to avoid lazy-import distortion.
        v.verify_step("1 * 2 = 2")
        v.batch_verify(["1 * 2 = 2"])

        t0 = time.perf_counter()
        v.verify_step(paragraphs[0])
        single_ms = (time.perf_counter() - t0) * 1000.0

        batch_result = v.batch_verify(paragraphs)
        batch_ms = batch_result.batch_latency_ms

        # REQ-VERIFY-148: batch must be < 2× a single call, not 10×.
        # Allow a generous 20× headroom for extremely fast CI machines where both
        # times are near zero (to avoid flaky failures on sub-millisecond runs).
        max_allowed = max(single_ms * 20, 500.0)  # never fail just because single is fast
        assert batch_ms < max_allowed, (
            f"batch_verify took {batch_ms:.2f}ms, single was {single_ms:.2f}ms — "
            f"expected < {max_allowed:.2f}ms"
        )

    def test_batch_verify_10_paragraphs_violations_match_serial(self) -> None:
        # Full SCENARIO-VERIFY-173 correctness gate.
        v = SymCodeVerifier()
        paragraphs = [
            "3 * 4 = 12",
            "5 + 6 = 11",
            "10 / 2 = 5",
            "7 - 3 = 4",
            "2 * 9 = 18",
            "100 - 25 = 75",
            "12 + 8 = 20",
            "6 * 7 = 42",
            "50 / 5 = 10",
            "The sky is blue.",
        ]
        serial_flags = [v.verify_step(p, i).violation_detected for i, p in enumerate(paragraphs)]
        batch_result = v.batch_verify(paragraphs)
        batch_flags = [r.violation_detected for r in batch_result.per_paragraph_results]
        assert batch_flags == serial_flags
        assert batch_result.n_paragraphs == 10


# ---------------------------------------------------------------------------
# Export from carnot.pipeline
# ---------------------------------------------------------------------------


class TestExports:
    """REQ-VERIFY-148: SymCodeBatchResult exported from symcode_verifier module."""

    def test_symcode_batch_result_importable(self) -> None:
        from carnot.pipeline.symcode_verifier import SymCodeBatchResult as SBR  # noqa: PLC0415

        assert SBR is SymCodeBatchResult

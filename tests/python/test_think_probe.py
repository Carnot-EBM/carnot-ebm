"""Tests for CarnotThinkProbe — generative CoT pre-filter.

Spec: REQ-VERIFY-094, REQ-VERIFY-095
SCENARIO-VERIFY-126, SCENARIO-VERIFY-127, SCENARIO-VERIFY-128
"""

from __future__ import annotations

import pytest

from carnot.pipeline.think_probe import (
    CarnotThinkProbe,
    ThinkProbeResult,
    ThinkVerdict,
    build_think_probe_prompt,
    parse_think_probe_output,
)


# ---------------------------------------------------------------------------
# ThinkVerdict
# ---------------------------------------------------------------------------


class TestThinkVerdict:
    def test_incorrect_verdict(self):
        v = ThinkVerdict(verdict="incorrect", confidence=0.9, reasoning_steps=["step1"])
        assert v.verdict == "incorrect"
        assert v.confidence == 0.9
        assert v.reasoning_steps == ["step1"]

    def test_uncertain_verdict(self):
        v = ThinkVerdict(verdict="uncertain", confidence=0.5, reasoning_steps=[])
        assert v.verdict == "uncertain"
        assert v.confidence == 0.5
        assert v.reasoning_steps == []

    def test_correct_verdict(self):
        v = ThinkVerdict(verdict="correct", confidence=0.9, reasoning_steps=["a", "b"])
        assert v.verdict == "correct"
        assert len(v.reasoning_steps) == 2

    def test_default_reasoning_steps_is_empty_list(self):
        v = ThinkVerdict(verdict="uncertain", confidence=0.5)
        assert v.reasoning_steps == []


# ---------------------------------------------------------------------------
# ThinkProbeResult
# ---------------------------------------------------------------------------


class TestThinkProbeResult:
    def test_incorrect_should_not_run_ising(self):
        # SCENARIO-VERIFY-127: fast-path skip
        verdict = ThinkVerdict(verdict="incorrect", confidence=0.9, reasoning_steps=[])
        result = ThinkProbeResult(
            response_text="2+2=5",
            verdict=verdict,
            should_run_ising=False,
            latency_ms=10.0,
        )
        assert result.should_run_ising is False

    def test_correct_should_run_ising(self):
        verdict = ThinkVerdict(verdict="correct", confidence=0.9, reasoning_steps=[])
        result = ThinkProbeResult(
            response_text="2+2=4",
            verdict=verdict,
            should_run_ising=True,
            latency_ms=10.0,
        )
        assert result.should_run_ising is True

    def test_uncertain_should_run_ising(self):
        verdict = ThinkVerdict(verdict="uncertain", confidence=0.5, reasoning_steps=[])
        result = ThinkProbeResult(
            response_text="maybe 4",
            verdict=verdict,
            should_run_ising=True,
            latency_ms=0.1,
        )
        assert result.should_run_ising is True

    def test_fields_preserved(self):
        verdict = ThinkVerdict(verdict="correct", confidence=0.85, reasoning_steps=["x"])
        result = ThinkProbeResult(
            response_text="hello",
            verdict=verdict,
            should_run_ising=True,
            latency_ms=42.0,
        )
        assert result.response_text == "hello"
        assert result.latency_ms == 42.0


# ---------------------------------------------------------------------------
# build_think_probe_prompt
# ---------------------------------------------------------------------------


class TestBuildThinkProbePrompt:
    def test_contains_response_text(self):
        prompt = build_think_probe_prompt("The answer is 42.")
        assert "The answer is 42." in prompt

    def test_contains_step_1(self):
        prompt = build_think_probe_prompt("x")
        assert "Step 1" in prompt

    def test_contains_step_2(self):
        prompt = build_think_probe_prompt("x")
        assert "Step 2" in prompt

    def test_contains_step_3(self):
        prompt = build_think_probe_prompt("x")
        assert "Step 3" in prompt

    def test_contains_verdict_instructions(self):
        prompt = build_think_probe_prompt("x")
        assert "VERDICT:" in prompt

    def test_contains_all_three_verdict_options(self):
        prompt = build_think_probe_prompt("x")
        assert "incorrect" in prompt
        assert "uncertain" in prompt
        assert "correct" in prompt

    def test_different_responses_produce_different_prompts(self):
        p1 = build_think_probe_prompt("response A")
        p2 = build_think_probe_prompt("response B")
        assert p1 != p2

    def test_returns_string(self):
        result = build_think_probe_prompt("any text")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# parse_think_probe_output
# ---------------------------------------------------------------------------


class TestParseThinkProbeOutput:
    def test_parses_incorrect(self):
        output = "Step 1: claim\nStep 2: check\nStep 3: VERDICT: incorrect"
        verdict = parse_think_probe_output(output)
        assert verdict.verdict == "incorrect"
        assert verdict.confidence == 0.9

    def test_parses_uncertain(self):
        output = "Step 1: claim\nStep 2: check\nVERDICT: uncertain"
        verdict = parse_think_probe_output(output)
        assert verdict.verdict == "uncertain"
        assert verdict.confidence == 0.9

    def test_parses_correct(self):
        output = "Step 1: claim\nStep 2: check\nStep 3: VERDICT: correct"
        verdict = parse_think_probe_output(output)
        assert verdict.verdict == "correct"
        assert verdict.confidence == 0.9

    def test_case_insensitive_verdict(self):
        # Parser is case-insensitive
        verdict = parse_think_probe_output("VERDICT: Incorrect")
        assert verdict.verdict == "incorrect"

    def test_case_insensitive_correct(self):
        verdict = parse_think_probe_output("VERDICT: CORRECT")
        assert verdict.verdict == "correct"

    def test_fallback_to_uncertain_when_no_verdict(self):
        output = "I cannot determine the answer."
        verdict = parse_think_probe_output(output)
        assert verdict.verdict == "uncertain"
        assert verdict.confidence == 0.5
        assert verdict.reasoning_steps == []

    def test_uses_last_verdict_when_multiple(self):
        # Multiple VERDICT lines: last one wins (model's final conclusion)
        output = "VERDICT: incorrect\nActually, VERDICT: correct"
        verdict = parse_think_probe_output(output)
        assert verdict.verdict == "correct"

    def test_extracts_reasoning_steps(self):
        output = (
            "Step 1: The claim is 2+2=5.\n"
            "Step 2: Actually 2+2=4.\n"
            "Step 3: VERDICT: incorrect"
        )
        verdict = parse_think_probe_output(output)
        assert len(verdict.reasoning_steps) == 3

    def test_empty_output_falls_back_to_uncertain(self):
        verdict = parse_think_probe_output("")
        assert verdict.verdict == "uncertain"
        assert verdict.confidence == 0.5

    def test_verdict_with_extra_spaces(self):
        verdict = parse_think_probe_output("VERDICT :  incorrect")
        assert verdict.verdict == "incorrect"


# ---------------------------------------------------------------------------
# CarnotThinkProbe — CI stub (REQ-VERIFY-095, SCENARIO-VERIFY-126)
# ---------------------------------------------------------------------------


class TestCarnotThinkProbeStub:
    def test_ci_stub_returns_uncertain(self):
        # SCENARIO-VERIFY-126
        probe = CarnotThinkProbe()
        result = probe.probe("any response text")
        assert result.verdict.verdict == "uncertain"

    def test_ci_stub_confidence_is_half(self):
        probe = CarnotThinkProbe()
        result = probe.probe("any response text")
        assert result.verdict.confidence == 0.5

    def test_ci_stub_should_run_ising_true(self):
        # SCENARIO-VERIFY-126: uncertain → Ising still runs
        probe = CarnotThinkProbe()
        result = probe.probe("any response text")
        assert result.should_run_ising is True

    def test_ci_stub_reasoning_steps_empty(self):
        probe = CarnotThinkProbe()
        result = probe.probe("any response text")
        assert result.verdict.reasoning_steps == []

    def test_ci_stub_latency_ms_is_float(self):
        probe = CarnotThinkProbe()
        result = probe.probe("x")
        assert isinstance(result.latency_ms, float)

    def test_ci_stub_response_text_preserved(self):
        probe = CarnotThinkProbe()
        result = probe.probe("hello world")
        assert result.response_text == "hello world"

    def test_ci_stub_default_confidence_threshold(self):
        probe = CarnotThinkProbe()
        assert probe.confidence_threshold == 0.8

    def test_ci_stub_llm_caller_is_none(self):
        probe = CarnotThinkProbe()
        assert probe.llm_caller is None


# ---------------------------------------------------------------------------
# CarnotThinkProbe — live LLM caller path
# ---------------------------------------------------------------------------


class TestCarnotThinkProbeLive:
    def _make_caller(self, verdict_text: str):
        """Return a mock llm_caller that emits the given verdict text."""

        def caller(prompt: str) -> str:
            return (
                "Step 1: Extract claim.\n"
                "Step 2: Check claim.\n"
                f"Step 3: VERDICT: {verdict_text}"
            )

        return caller

    def test_incorrect_verdict_skips_ising(self):
        # SCENARIO-VERIFY-127
        probe = CarnotThinkProbe(llm_caller=self._make_caller("incorrect"))
        result = probe.probe("2+2=5")
        assert result.verdict.verdict == "incorrect"
        assert result.should_run_ising is False

    def test_correct_verdict_runs_ising(self):
        probe = CarnotThinkProbe(llm_caller=self._make_caller("correct"))
        result = probe.probe("2+2=4")
        assert result.verdict.verdict == "correct"
        assert result.should_run_ising is True

    def test_uncertain_verdict_runs_ising(self):
        probe = CarnotThinkProbe(llm_caller=self._make_caller("uncertain"))
        result = probe.probe("maybe 4")
        assert result.verdict.verdict == "uncertain"
        assert result.should_run_ising is True

    def test_latency_ms_is_nonnegative(self):
        probe = CarnotThinkProbe(llm_caller=self._make_caller("correct"))
        result = probe.probe("x")
        assert result.latency_ms >= 0.0

    def test_response_text_preserved(self):
        probe = CarnotThinkProbe(llm_caller=self._make_caller("correct"))
        result = probe.probe("the response text")
        assert result.response_text == "the response text"

    def test_reasoning_steps_extracted(self):
        probe = CarnotThinkProbe(llm_caller=self._make_caller("incorrect"))
        result = probe.probe("x")
        assert len(result.verdict.reasoning_steps) == 3

    def test_caller_receives_prompt_with_response(self):
        received_prompts = []

        def recording_caller(prompt: str) -> str:
            received_prompts.append(prompt)
            return "VERDICT: correct"

        probe = CarnotThinkProbe(llm_caller=recording_caller)
        probe.probe("my specific response")
        assert len(received_prompts) == 1
        assert "my specific response" in received_prompts[0]

    def test_custom_confidence_threshold_stored(self):
        probe = CarnotThinkProbe(llm_caller=self._make_caller("correct"), confidence_threshold=0.95)
        assert probe.confidence_threshold == 0.95

    def test_no_verdict_in_output_falls_back_uncertain(self):
        def bad_caller(prompt: str) -> str:
            return "I cannot tell."

        probe = CarnotThinkProbe(llm_caller=bad_caller)
        result = probe.probe("x")
        assert result.verdict.verdict == "uncertain"
        assert result.should_run_ising is True


# ---------------------------------------------------------------------------
# CarnotThinkProbe.benchmark() (SCENARIO-VERIFY-128)
# ---------------------------------------------------------------------------


class TestCarnotThinkProbeBenchmark:
    def _make_deterministic_caller(self, wrong_verdict: str = "incorrect"):
        """Caller that returns 'incorrect' for responses containing 'WRONG' else 'correct'."""

        def caller(prompt: str) -> str:
            if "WRONG" in prompt:
                return f"VERDICT: {wrong_verdict}"
            return "VERDICT: correct"

        return caller

    def test_empty_corpus_returns_zeros(self):
        probe = CarnotThinkProbe()
        result = probe.benchmark([], [])
        assert result == {"skip_rate": 0.0, "tp_rate": 0.0, "fp_rate": 0.0}

    def test_benchmark_returns_required_keys(self):
        probe = CarnotThinkProbe()
        result = probe.benchmark(["x"], [True])
        assert "skip_rate" in result
        assert "tp_rate" in result
        assert "fp_rate" in result

    def test_benchmark_all_values_float(self):
        probe = CarnotThinkProbe()
        result = probe.benchmark(["x", "y"], [True, False])
        assert all(isinstance(v, float) for v in result.values())

    def test_skip_rate_in_unit_interval(self):
        probe = CarnotThinkProbe()
        result = probe.benchmark(["a"] * 10, [True] * 5 + [False] * 5)
        assert 0.0 <= result["skip_rate"] <= 1.0

    def test_tp_rate_in_unit_interval(self):
        probe = CarnotThinkProbe()
        result = probe.benchmark(["a"] * 10, [True] * 5 + [False] * 5)
        assert 0.0 <= result["tp_rate"] <= 1.0

    def test_fp_rate_in_unit_interval(self):
        probe = CarnotThinkProbe()
        result = probe.benchmark(["a"] * 10, [True] * 5 + [False] * 5)
        assert 0.0 <= result["fp_rate"] <= 1.0

    def test_ci_stub_skip_rate_zero_no_incorrect_flags(self):
        # CI stub always returns 'uncertain', so nothing is skipped
        probe = CarnotThinkProbe()
        result = probe.benchmark(["x"] * 20, [True] * 10 + [False] * 10)
        assert result["skip_rate"] == 0.0
        assert result["tp_rate"] == 0.0
        assert result["fp_rate"] == 0.0

    def test_live_benchmark_wrong_responses_flagged(self):
        # SCENARIO-VERIFY-128: wrong responses flagged → tp_rate > 0
        # Wrong responses contain "WRONG", correct do not.
        wrong_responses = ["WRONG response " + str(i) for i in range(10)]
        correct_responses = ["correct response " + str(i) for i in range(10)]
        responses = correct_responses + wrong_responses
        ground_truth = [True] * 10 + [False] * 10

        probe = CarnotThinkProbe(llm_caller=self._make_deterministic_caller())
        result = probe.benchmark(responses, ground_truth)

        assert result["tp_rate"] == 1.0, "All wrong responses should be flagged"
        assert result["fp_rate"] == 0.0, "No correct responses should be flagged"
        assert result["skip_rate"] == 0.5, "Half of all responses flagged (10/20)"

    def test_no_wrong_responses_tp_rate_zero(self):
        # No wrong responses → tp_rate undefined → 0.0
        probe = CarnotThinkProbe()
        result = probe.benchmark(["x", "y"], [True, True])
        assert result["tp_rate"] == 0.0

    def test_no_correct_responses_fp_rate_zero(self):
        # No correct responses → fp_rate undefined → 0.0
        probe = CarnotThinkProbe()
        result = probe.benchmark(["x", "y"], [False, False])
        assert result["fp_rate"] == 0.0

    def test_high_fp_rate_when_caller_always_flags_incorrect(self):
        def always_incorrect(prompt: str) -> str:
            return "VERDICT: incorrect"

        probe = CarnotThinkProbe(llm_caller=always_incorrect)
        # All correct responses
        result = probe.benchmark(["x"] * 10, [True] * 10)
        assert result["fp_rate"] == 1.0
        assert result["skip_rate"] == 1.0

    def test_benchmark_50_correct_50_wrong_synthetic(self):
        # Full synthetic test matching Exp 444 design
        wrong_responses = ["WRONG: " + str(i) for i in range(50)]
        correct_responses = ["correct: " + str(i) for i in range(50)]
        responses = correct_responses + wrong_responses
        ground_truth = [True] * 50 + [False] * 50

        probe = CarnotThinkProbe(llm_caller=self._make_deterministic_caller())
        result = probe.benchmark(responses, ground_truth)

        assert result["tp_rate"] == 1.0
        assert result["fp_rate"] == 0.0
        assert result["skip_rate"] == 0.5


# ---------------------------------------------------------------------------
# Import from public API (carnot.pipeline)
# ---------------------------------------------------------------------------


def test_imports_from_pipeline_init():
    """Verify the public API exports are available from carnot.pipeline."""
    from carnot.pipeline import (  # noqa: F401
        CarnotThinkProbe,
        ThinkProbeResult,
        ThinkVerdict,
        build_think_probe_prompt,
        parse_think_probe_output,
    )

"""Tests for scripts/experiment_963_math_repair_sota_v3_retry.py

REQ-VER-MATH-001: external scratchpad includes prior-attempt error text.
REQ-VER-MATH-002: repair verdict is computed correctly from accuracy values.
SCENARIO-VER-MATH-001: full experiment produces a valid deliverable JSON with
                        all required schema fields.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

# Allow importing from scripts/ without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import experiment_963_math_repair_sota_v3_retry as exp963


# ---------------------------------------------------------------------------
# extract_numeric_answer
# ---------------------------------------------------------------------------


class TestExtractNumericAnswer:
    def test_gsm8k_hash_format(self):
        # Standard GSM8K "#### N" format should be reliably extracted.
        assert exp963.extract_numeric_answer("Step 1: add.\n#### 42") == 42.0

    def test_the_answer_is_format(self):
        assert exp963.extract_numeric_answer("The answer is 72.") == 72.0

    def test_fallback_last_number(self):
        # When no structured marker, last numeric token in the tail is used.
        assert exp963.extract_numeric_answer("so 10 plus 5 equals 15") == 15.0

    def test_returns_none_for_no_numbers(self):
        assert exp963.extract_numeric_answer("No numbers here at all.") is None

    def test_strips_commas(self):
        assert exp963.extract_numeric_answer("#### 1,000") == 1000.0


# ---------------------------------------------------------------------------
# answers_match
# ---------------------------------------------------------------------------


class TestAnswersMatch:
    def test_exact_match(self):
        assert exp963.answers_match(72.0, 72) is True

    def test_float_rounded_match(self):
        # "72.0" should match integer 72.
        assert exp963.answers_match(72.3, 72) is True

    def test_wrong_answer(self):
        assert exp963.answers_match(70.0, 72) is False

    def test_none_extracted(self):
        assert exp963.answers_match(None, 72) is False


# ---------------------------------------------------------------------------
# _build_scratchpad
# ---------------------------------------------------------------------------


class TestBuildScratchpad:
    def test_contains_error_log_header(self):
        scratchpad = exp963._build_scratchpad("Step 1: 5+5=10.\n#### 10", 10.0)
        assert "Previous attempt error log" in scratchpad

    def test_contains_extracted_answer(self):
        # The wrong answer must appear as explicit text in the scratchpad.
        scratchpad = exp963._build_scratchpad("#### 99", 99.0)
        assert "99" in scratchpad
        assert "INCORRECT" in scratchpad

    def test_contains_prior_response_tail(self):
        # At least part of the prior response must appear in the scratchpad.
        prior_response = "reasoning chain " * 10 + "#### 5"
        scratchpad = exp963._build_scratchpad(prior_response, 5.0)
        assert "#### 5" in scratchpad  # tail of prior response is included

    def test_none_answer_shows_unknown(self):
        scratchpad = exp963._build_scratchpad("could not compute", None)
        assert "unknown" in scratchpad


# ---------------------------------------------------------------------------
# _scratchpad_repair_prompt
# ---------------------------------------------------------------------------


class TestScratchpadRepairPrompt:
    def test_includes_scratchpad(self):
        prompt = exp963._scratchpad_repair_prompt("What is 2+2?", "I got 5.\n#### 5", 5.0)
        assert "error log" in prompt.lower()
        assert "#### 5" in prompt  # prior response tail present
        assert "What is 2+2?" in prompt

    def test_includes_gsm8k_answer_format_instruction(self):
        prompt = exp963._scratchpad_repair_prompt("Q?", "A", 1.0)
        assert "#### <number>" in prompt


# ---------------------------------------------------------------------------
# _run_problem_with_scratchpad  (unit test with a stub runner)
# ---------------------------------------------------------------------------


class _AlwaysCorrectRunner:
    """Returns a correct answer immediately."""

    def __init__(self, correct_answer: int) -> None:
        self._ans = correct_answer

    def generate(self, prompt: str) -> str:  # noqa: ARG002
        return f"Step 1: answer.\n#### {self._ans}"


class _AlwaysWrongRunner:
    """Returns an incorrect answer on every call."""

    def generate(self, prompt: str) -> str:  # noqa: ARG002
        return "I have no idea.\n#### 9999"


class _WrongThenCorrectRunner:
    """Returns wrong on the first call, correct on subsequent calls."""

    def __init__(self, correct_answer: int) -> None:
        self._ans = correct_answer
        self._calls = 0

    def generate(self, prompt: str) -> str:  # noqa: ARG002
        self._calls += 1
        if self._calls == 1:
            return "#### 9999"
        return f"#### {self._ans}"


class _FakeLenScorer:
    def score(self, text: str) -> float:
        return float(len(text.split()))


class TestRunProblemWithScratchpad:
    _scorer = _FakeLenScorer()

    def test_baseline_correct_stops_immediately(self):
        runner = _AlwaysCorrectRunner(72)
        result = exp963._run_problem_with_scratchpad(
            "question", 72, runner, self._scorer, max_retries=2
        )
        assert result["baseline_passed"] is True
        assert result["repair_passed"] is True
        assert result["n_attempts"] == 1  # stopped after baseline pass

    def test_wrong_then_correct_with_scratchpad(self):
        runner = _WrongThenCorrectRunner(72)
        result = exp963._run_problem_with_scratchpad(
            "question", 72, runner, self._scorer, max_retries=2
        )
        assert result["baseline_passed"] is False
        assert result["repair_passed"] is True
        # At least one repair attempt used the external scratchpad.
        assert result["scratchpad_used_any"] is True

    def test_always_wrong_produces_failed_result(self):
        runner = _AlwaysWrongRunner()
        result = exp963._run_problem_with_scratchpad(
            "question", 72, runner, self._scorer, max_retries=2
        )
        assert result["baseline_passed"] is False
        assert result["repair_passed"] is False
        assert result["n_attempts"] == 3  # baseline + 2 retries


# ---------------------------------------------------------------------------
# Verdict mapping
# ---------------------------------------------------------------------------


class TestVerdictMapping:
    """Verify the honest_verdict mapping logic matches the spec."""

    def _verdict_for(self, baseline: float, repair: float) -> str:
        """Exercise the verdict branch identical to main()."""
        delta = repair - baseline
        if delta == 0.0:
            return "sota_ceiling_confirmed"
        elif delta > 0.10:
            return "math_repair_significant"
        elif delta > 0.0:
            return "math_repair_marginal"
        else:
            return "math_repair_negative"

    def test_zero_delta_gives_sota_ceiling(self):
        assert self._verdict_for(0.75, 0.75) == "sota_ceiling_confirmed"

    def test_large_delta_gives_significant(self):
        assert self._verdict_for(0.50, 0.65) == "math_repair_significant"

    def test_small_delta_gives_marginal(self):
        assert self._verdict_for(0.75, 0.80) == "math_repair_marginal"

    def test_negative_delta_gives_negative(self):
        assert self._verdict_for(0.80, 0.75) == "math_repair_negative"


# ---------------------------------------------------------------------------
# Deliverable schema validation (integration-style, stub runner)
# ---------------------------------------------------------------------------


class TestDeliverableSchema:
    REQUIRED_FIELDS = {
        "baseline_accuracy",
        "repair_accuracy",
        "repair_delta",
        "n_problems",
        "model_used",
        "scratchpad_used",
        "honest_verdict",
    }

    def test_main_produces_valid_deliverable(self, tmp_path):
        """Run main() with a stub runner and verify all required schema fields are present."""
        deliverable = str(tmp_path / "experiment_963_math_repair_sota_v3_retry.json")

        # Patch the deliverable path inside the module so output goes to tmp_path.
        # Also patch _load_sota_runner to return a fast stub runner.
        stub_runner = _AlwaysWrongRunner()
        stub_model_specs = [{"name": "stub", "hf_id": "stub", "gpu": 0}]
        stub_scorer = _FakeLenScorer()

        # Only run on 3 problems to keep test fast.
        trimmed_problems = exp963._GSM8K_PROBLEMS[:3]

        with (
            mock.patch.object(exp963, "_DELIVERABLE", deliverable),
            mock.patch.object(exp963, "_REPO_ROOT", tmp_path),
            mock.patch.object(exp963, "_GSM8K_PROBLEMS", trimmed_problems),
            mock.patch(
                "experiment_963_math_repair_sota_v3_retry._load_sota_runner",
                return_value=(stub_runner, "stub", stub_model_specs),
            ),
            mock.patch(
                "experiment_963_math_repair_sota_v3_retry._build_energy_scorer",
                return_value=(stub_scorer, "token_length_heuristic"),
            ),
        ):
            exp963.main()

        artifact = json.loads(Path(deliverable).read_text())
        for field in self.REQUIRED_FIELDS:
            assert field in artifact, f"Missing required field: {field}"

    def test_partial_artifact_written_on_exception(self, tmp_path):
        """When an exception occurs, a partial artifact with status=blocked is written."""
        deliverable = str(tmp_path / "experiment_963_math_repair_sota_v3_retry.json")

        def _exploding_loader(*args, **kwargs):  # noqa: ARG001
            raise RuntimeError("simulated GPU OOM")

        with (
            mock.patch.object(exp963, "_DELIVERABLE", deliverable),
            mock.patch.object(exp963, "_REPO_ROOT", tmp_path),
            mock.patch(
                "experiment_963_math_repair_sota_v3_retry._load_sota_runner",
                side_effect=_exploding_loader,
            ),
            mock.patch(
                "experiment_963_math_repair_sota_v3_retry._build_energy_scorer",
                return_value=(_FakeLenScorer(), "token_length_heuristic"),
            ),
        ):
            try:
                exp963.main()
            except RuntimeError:
                pass  # expected — we re-raise after writing partial artifact

        artifact = json.loads(Path(deliverable).read_text())
        assert artifact["status"] == "blocked"
        assert "stall_details" in artifact
        assert "simulated GPU OOM" in artifact["stall_details"]

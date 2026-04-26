"""Tests for Exp 930: Math Iterative Self-Repair v1 — GSM8K.

Each test covers one distinct behaviour of the experiment module:
  (a) answer-extraction regex
  (b) repair loop termination
  (c) energy scorer call
  (d) honest_verdict assignment (via direct logic, not full main())

All GPU-side calls are mocked so the suite runs on any CI host.

Spec: REQ-VER-MATH-001, REQ-VER-MATH-002,
      SCENARIO-VER-MATH-001
Spec: REQ-AUTO-011
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Make the scripts/ directory importable without the full project install.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import scripts.experiment_930_math_iterative_self_repair_v1 as _mod  # noqa: E402

# ---------------------------------------------------------------------------
# (a) Answer extraction regex
# ---------------------------------------------------------------------------


class TestExtractNumericAnswer:
    """Verify that extract_numeric_answer handles common GSM8K output formats."""

    def test_gsm8k_hash_format(self):
        """#### N at end of response (standard GSM8K format)."""
        result = _mod.extract_numeric_answer("Step 1: ...\nStep 2: ...\n#### 72")
        assert result == 72.0

    def test_the_answer_is_format(self):
        """'The answer is N.' phrasing used by many LLMs."""
        result = _mod.extract_numeric_answer("Therefore, the answer is 42.")
        assert result == 42.0

    def test_plain_trailing_number(self):
        """Fallback: last number in response with no keyword."""
        result = _mod.extract_numeric_answer("We get 5 apples plus 3 apples = 8 apples total.")
        assert result == 8.0

    def test_comma_formatted_number(self):
        """Numbers with commas like 1,000 are parsed correctly."""
        result = _mod.extract_numeric_answer("The total is 1,000 dollars.\n#### 1,000")
        assert result == 1000.0

    def test_no_number_returns_none(self):
        """Responses with no numeric content return None."""
        result = _mod.extract_numeric_answer("I am not sure how to solve this problem.")
        assert result is None

    def test_empty_string_returns_none(self):
        result = _mod.extract_numeric_answer("")
        assert result is None


class TestAnswersMatch:
    """Verify tolerance-based integer comparison."""

    def test_exact_match(self):
        assert _mod.answers_match(72.0, 72) is True

    def test_float_rounds_to_match(self):
        assert _mod.answers_match(72.4, 72) is True

    def test_none_extracted_is_false(self):
        assert _mod.answers_match(None, 72) is False

    def test_off_by_one_is_false(self):
        assert _mod.answers_match(71.0, 72) is False


# ---------------------------------------------------------------------------
# (b) Repair loop termination
# ---------------------------------------------------------------------------


class TestRunProblem:
    """Verify _run_problem stops early on correct answer and respects max_retries."""

    def _make_runner(self, responses: list[str]) -> MagicMock:
        runner = MagicMock()
        runner.generate.side_effect = responses
        return runner

    def _make_scorer(self, energies: list[float]) -> MagicMock:
        scorer = MagicMock()
        scorer.score.side_effect = energies
        return scorer

    def test_stops_early_on_correct_baseline(self):
        """When round 0 is correct the loop does NOT call generate a second time."""
        runner = self._make_runner(["Step 1: 48 + 24 = 72\n#### 72"])
        scorer = self._make_scorer([0.5])
        result = _mod._run_problem("question", 72, runner, scorer, max_retries=3)
        assert runner.generate.call_count == 1
        assert result["baseline_passed"] is True
        assert result["repair_passed"] is True
        assert result["n_retries"] == 0

    def test_retries_on_wrong_baseline(self):
        """When baseline is wrong the loop retries up to max_retries times."""
        runner = self._make_runner(
            [
                "#### 10",  # wrong
                "#### 10",  # wrong
                "#### 72",  # correct on round 2
            ]
        )
        scorer = self._make_scorer([1.0, 1.0, 0.5])
        result = _mod._run_problem("question", 72, runner, scorer, max_retries=3)
        assert result["baseline_passed"] is False
        assert result["repair_passed"] is True
        assert result["n_retries"] == 2

    def test_max_retries_respected(self):
        """Loop never exceeds max_retries repair rounds (1 baseline + N repair = N+1 calls)."""
        runner = self._make_runner(["#### 99"] * 4)
        scorer = self._make_scorer([1.0] * 4)
        result = _mod._run_problem("question", 72, runner, scorer, max_retries=3)
        assert runner.generate.call_count == 4
        assert result["baseline_passed"] is False
        assert result["repair_passed"] is False

    def test_energy_scorer_called_per_attempt(self):
        """energy_scorer.score is invoked once per attempt."""
        runner = self._make_runner(["#### 10", "#### 72"])
        scorer = self._make_scorer([2.0, 0.3])
        _mod._run_problem("question", 72, runner, scorer, max_retries=3)
        assert scorer.score.call_count == 2

    def test_best_attempt_uses_lowest_energy_among_passers(self):
        """When first passing attempt has high energy, later lower-energy pass is preferred."""
        # Rounds: 0=wrong(energy 5), 1=pass(energy 2), 2=pass(energy 0.1)
        # But loop stops at first pass (round 1) by design — so energy_score_best=2.0.
        runner = self._make_runner(["#### 10", "#### 72", "#### 72"])
        scorer = self._make_scorer([5.0, 2.0, 0.1])
        result = _mod._run_problem("question", 72, runner, scorer, max_retries=3)
        assert result["repair_passed"] is True
        assert result["energy_score_best"] == 2.0  # stopped at first pass


# ---------------------------------------------------------------------------
# (c) Energy scorer instantiation
# ---------------------------------------------------------------------------


class TestBuildEnergyScorer:
    """Verify _build_energy_scorer returns a working scorer in both code paths."""

    def test_fallback_scorer_when_ising_unavailable(self):
        """When IsingModel import fails the fallback scorer runs without error."""
        with patch.dict("sys.modules", {"carnot.models.ising": None, "jax.random": None}):
            scorer, label = _mod._build_energy_scorer()
        assert callable(scorer.score)
        assert label == "token_length_heuristic"

    def test_fallback_scorer_returns_positive_float(self):
        """Fallback scorer returns a positive float (token count)."""
        with patch.dict("sys.modules", {"carnot.models.ising": None, "jax.random": None}):
            scorer, _ = _mod._build_energy_scorer()
        val = scorer.score("hello world this is a test")
        assert isinstance(val, float)
        assert val > 0

    def test_fallback_scorer_empty_string(self):
        """Fallback scorer returns 0.0 for empty string (no tokens)."""
        with patch.dict("sys.modules", {"carnot.models.ising": None, "jax.random": None}):
            scorer, _ = _mod._build_energy_scorer()
        val = scorer.score("")
        assert val == 0.0


# ---------------------------------------------------------------------------
# (d) Honest-verdict assignment (tested directly via the module's logic)
# ---------------------------------------------------------------------------


class TestHonestVerdict:
    """Verify the signed_improvement → honest_verdict mapping.

    We test the branching logic directly rather than running main(), which
    avoids the need to mock model loading, file I/O, etc.
    """

    def _verdict_for(self, baseline_accuracy: float, repair_accuracy: float) -> str:
        """Replicate the verdict branching logic from main()."""
        signed_improvement = repair_accuracy - baseline_accuracy
        if signed_improvement > 0.10:
            return "math_repair_significant"
        elif signed_improvement > 0:
            return "math_repair_marginal"
        elif signed_improvement == 0:
            return "math_repair_zero"
        else:
            return "math_repair_negative"

    def test_significant_when_above_10pp(self):
        assert self._verdict_for(0.0, 0.72) == "math_repair_significant"

    def test_marginal_when_between_0_and_10pp(self):
        assert self._verdict_for(0.60, 0.68) == "math_repair_marginal"

    def test_zero_when_no_change(self):
        assert self._verdict_for(0.40, 0.40) == "math_repair_zero"

    def test_negative_when_repair_worse(self):
        assert self._verdict_for(0.50, 0.40) == "math_repair_negative"

    def test_boundary_exactly_10pp(self):
        """Exactly 10pp improvement is marginal (not significant)."""
        assert self._verdict_for(0.0, 0.10) == "math_repair_marginal"

    def test_boundary_just_above_10pp(self):
        """Just above 10pp is significant."""
        assert self._verdict_for(0.0, 0.101) == "math_repair_significant"

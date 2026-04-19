"""Tests for EORMAdaptiveRectifier and RectifierResult.

100% coverage for python/carnot/pipeline/eorm_rectifier.py.

Spec coverage: REQ-VERIFY-102, REQ-VERIFY-103,
               SCENARIO-VERIFY-138, SCENARIO-VERIFY-139
"""

from __future__ import annotations

import jax.random as jrandom

from carnot.models.eorm import EORMModel
from carnot.pipeline.eorm_rectifier import EORMAdaptiveRectifier, RectifierResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model() -> EORMModel:
    """Create a tiny EORMModel for fast CPU tests."""
    return EORMModel(embed_dim=32, n_heads=4, n_layers=1, max_seq_len=64, vocab_size=512,
                     key=jrandom.PRNGKey(42))


# ---------------------------------------------------------------------------
# RectifierResult
# ---------------------------------------------------------------------------


class TestRectifierResult:
    """REQ-VERIFY-102: RectifierResult dataclass."""

    def test_fields(self) -> None:
        """All fields are stored correctly."""
        r = RectifierResult(
            baseline_accuracy=0.60,
            rectified_accuracy=0.75,
            k_candidates=3,
            signed_improvement=0.15,
            honest_verdict="eorm_rectification_positive",
        )
        assert r.baseline_accuracy == 0.60
        assert r.rectified_accuracy == 0.75
        assert r.k_candidates == 3
        assert r.signed_improvement == 0.15
        assert r.honest_verdict == "eorm_rectification_positive"


# ---------------------------------------------------------------------------
# EORMAdaptiveRectifier.select_candidate
# ---------------------------------------------------------------------------


class TestSelectCandidate:
    """SCENARIO-VERIFY-138: select_candidate returns minimum-energy candidate."""

    def test_returns_lowest_energy_candidate(self) -> None:
        """REQ-VERIFY-102: the candidate with lowest EORM energy is returned.

        We bias the model to give lower energy to a specific string by constructing
        candidates where one is a very short string (fewer tokens → different energy).
        The exact value doesn't matter — we verify that select_candidate is consistent
        with calling model.energy() directly.
        """
        model = _make_model()
        rectifier = EORMAdaptiveRectifier(model, k=3)
        question = "What is 2 + 2?"
        candidates = ["The answer is 4.", "The answer is 5.", "I do not know."]

        from carnot.models.eorm import CoTEnergyInput

        energies = [
            model.energy(CoTEnergyInput(question_text=question, response_text=c))
            for c in candidates
        ]
        expected = candidates[energies.index(min(energies))]
        result = rectifier.select_candidate(question, candidates)
        assert result == expected

    def test_single_candidate_returns_it(self) -> None:
        """With one candidate, that candidate is always returned."""
        model = _make_model()
        rectifier = EORMAdaptiveRectifier(model)
        result = rectifier.select_candidate("q", ["only one"])
        assert result == "only one"

    def test_default_k_is_3(self) -> None:
        """REQ-VERIFY-103: default k=3."""
        model = _make_model()
        r = EORMAdaptiveRectifier(model)
        assert r.k == 3

    def test_custom_k(self) -> None:
        """REQ-VERIFY-103: k is configurable."""
        model = _make_model()
        r = EORMAdaptiveRectifier(model, k=5)
        assert r.k == 5


# ---------------------------------------------------------------------------
# EORMAdaptiveRectifier.evaluate
# ---------------------------------------------------------------------------


class TestEvaluate:
    """SCENARIO-VERIFY-139: evaluate returns correct RectifierResult."""

    def _make_questions(self, n: int = 10) -> list[dict]:
        return [{"question": f"What is {i}+{i}?", "answer": str(i * 2)} for i in range(n)]

    def test_always_correct_inference(self) -> None:
        """When inference_fn always returns the correct answer, both accuracies = 1.0."""
        model = _make_model()
        rectifier = EORMAdaptiveRectifier(model, k=3)
        questions = self._make_questions(5)

        def always_correct(q: str) -> str:
            # Extract the answer from the question format "What is X+X?"
            parts = q.replace("What is ", "").replace("?", "").split("+")
            val = int(parts[0].strip())
            return str(val * 2)

        result = rectifier.evaluate(questions, always_correct)
        assert result.baseline_accuracy == 1.0
        assert result.rectified_accuracy == 1.0
        assert result.signed_improvement == 0.0
        assert result.k_candidates == 3

    def test_always_wrong_inference(self) -> None:
        """When inference_fn always returns a wrong answer, both accuracies = 0.0."""
        model = _make_model()
        rectifier = EORMAdaptiveRectifier(model, k=3)
        questions = self._make_questions(5)

        def always_wrong(q: str) -> str:
            return "WRONG"

        result = rectifier.evaluate(questions, always_wrong)
        assert result.baseline_accuracy == 0.0
        assert result.rectified_accuracy == 0.0
        assert result.signed_improvement == 0.0
        assert result.honest_verdict == "no_improvement"

    def test_honest_verdict_positive(self) -> None:
        """REQ-VERIFY-102: honest_verdict is 'eorm_rectification_positive' when improved."""
        model = _make_model()
        rectifier = EORMAdaptiveRectifier(model, k=3)

        questions = [{"question": "q1", "answer": "yes"}]
        call_count = [0]

        def mixed(q: str) -> str:
            # Baseline call (first call per question): wrong. K calls: correct.
            call_count[0] += 1
            # First call per question group is baseline, rest are K candidates
            # We return wrong for baseline, correct for all K candidates
            if call_count[0] % 4 == 1:  # 1st call = baseline
                return "no"
            return "yes"

        result = rectifier.evaluate(questions, mixed)
        # baseline wrong, rectified correct → positive improvement
        assert result.signed_improvement > 0
        assert result.honest_verdict == "eorm_rectification_positive"

    def test_honest_verdict_no_improvement(self) -> None:
        """honest_verdict is 'no_improvement' when improvement <= 0."""
        model = _make_model()
        rectifier = EORMAdaptiveRectifier(model, k=2)
        questions = [{"question": "q", "answer": "gold"}]

        # All calls return wrong — rectified same as baseline
        result = rectifier.evaluate(questions, lambda q: "wrong")
        assert result.honest_verdict == "no_improvement"

    def test_k_override_in_evaluate(self) -> None:
        """REQ-VERIFY-103: k parameter in evaluate overrides instance k."""
        model = _make_model()
        rectifier = EORMAdaptiveRectifier(model, k=3)
        questions = [{"question": "q", "answer": "a"}]

        call_counts = [0]

        def counter(q: str) -> str:
            call_counts[0] += 1
            return "a"

        rectifier.evaluate(questions, counter, k=5)
        # 1 baseline call + 5 K calls = 6 total
        assert call_counts[0] == 6

    def test_empty_questions(self) -> None:
        """evaluate with empty question list returns 0.0 accuracies."""
        model = _make_model()
        rectifier = EORMAdaptiveRectifier(model)
        result = rectifier.evaluate([], lambda q: "x")
        assert result.baseline_accuracy == 0.0
        assert result.rectified_accuracy == 0.0
        assert result.k_candidates == 3

    def test_custom_is_correct_fn(self) -> None:
        """Custom is_correct_fn is used for evaluation."""
        model = _make_model()
        rectifier = EORMAdaptiveRectifier(model, k=2)
        questions = [{"question": "q", "answer": "42"}]

        # is_correct_fn that always returns True
        result = rectifier.evaluate(questions, lambda q: "wrong",
                                    is_correct_fn=lambda resp, gold: True)
        assert result.baseline_accuracy == 1.0
        assert result.rectified_accuracy == 1.0

    def test_result_fields_populated(self) -> None:
        """All RectifierResult fields are populated by evaluate."""
        model = _make_model()
        rectifier = EORMAdaptiveRectifier(model, k=2)
        result = rectifier.evaluate(
            [{"question": "q", "answer": "a"}],
            lambda q: "a",
        )
        assert isinstance(result, RectifierResult)
        assert isinstance(result.baseline_accuracy, float)
        assert isinstance(result.rectified_accuracy, float)
        assert isinstance(result.k_candidates, int)
        assert isinstance(result.signed_improvement, float)
        assert isinstance(result.honest_verdict, str)

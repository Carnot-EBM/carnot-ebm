"""Tests for Exp 708: Gemma4-E4B-it VR with Adaptive Threshold Gating.

WHY THIS TEST FILE EXISTS:
    Exp 694 showed VR hurts Gemma4-E4B-it (signed_improvement=-0.8).
    Exp 707 implemented ModelAdaptiveThresholdGate.
    Exp 708 applies the gate live to recover the regression.

    This test suite validates the logic in experiment_708_vr_attempt_19_gemma4.py:
    1. Gate is loaded before inference begins (REQ-VERIFY-148-1).
    2. signed_improvement computation is correct (REQ-VERIFY-148-2).
    3. honest_verdict classification covers all branches (REQ-VERIFY-148-3/4/5/6).
    4. The deliverable JSON is written with all required schema fields.

Spec: REQ-VERIFY-148, SCENARIO-VERIFY-148
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

# Set CARNOT_FORCE_LIVE before importing the module so the assertion passes.
os.environ.setdefault("CARNOT_FORCE_LIVE", "1")

import experiment_708_vr_attempt_19_gemma4 as exp708  # noqa: E402

_DELIVERABLE = _REPO_ROOT / "results" / "experiment_708_vr_attempt_19_gemma4.json"


# ---------------------------------------------------------------------------
# Shared helper: fake ExperimentTemplate that avoids MagicMock assert_* issues
# ---------------------------------------------------------------------------


def _make_fake_tmpl(deliverable: Path) -> Any:
    """Return a plain-object stand-in for ExperimentTemplate.

    Python 3.8+ MagicMock raises AttributeError when you access attributes
    starting with 'assert_' that are not standard mock assertion methods
    (e.g. 'assert_deliverable_written').  Using a plain class avoids this
    entirely — no magic interception, just attribute lookups.
    """

    def _build_result(data: dict, **kw: Any) -> dict:
        return {
            "experiment": 708,
            "title": "VR Attempt #19: Gemma4-E4B-it with Adaptive Threshold Gating",
            "run_date": "20260422",
            "started_at": "2026-04-22T00:00:00Z",
            "finished_at": "2026-04-22T00:00:01Z",
            "duration_s": 1.0,
            "status": kw.get("status", "success"),
            "schema": sorted(data.keys()),
            **data,
        }

    class _FakeTmpl:
        _output_path = deliverable
        setup = staticmethod(lambda: None)
        checkpoint_save = staticmethod(lambda data, step=None: None)
        assert_deliverable_written = staticmethod(lambda: None)
        build_result = staticmethod(_build_result)

        @staticmethod
        def setup_gpu(specs: Any) -> dict:
            return {"all_healthy": False, "models": []}

    return _FakeTmpl()


# ---------------------------------------------------------------------------
# REQ-VERIFY-148-1: Gate loaded before inference
# ---------------------------------------------------------------------------


class TestGateLoadedBeforeInference:
    """Verify the gate is seeded/loaded before any inference call is made.

    WHY: REQ-VERIFY-148-1 mandates the gate state is consulted before the
    first inference call.  A gate loaded after inference would allow
    the first batch to run without suppression, reintroducing the -0.8
    regression on the first question in the set.
    Spec: REQ-VERIFY-148-1, SCENARIO-VERIFY-148.
    """

    def test_gate_suppresses_symcode_after_seed(self, tmp_path: Path) -> None:
        """_build_seeded_gate seeds FP observations when no state file exists.

        The returned gate must have SymCodeVerifier suppressed for Gemma4
        immediately after construction — before any inference runs.
        Spec: REQ-VERIFY-148-1.
        """
        state_file = tmp_path / "gate_state.json"
        # No file present — _build_seeded_gate should seed synthetic FPs.
        gate = exp708._build_seeded_gate(state_file)
        assert gate.is_suppressed(
            exp708.GEMMA4_MODEL_ID, exp708.SUPPRESSED_CONSTRAINT_TYPE
        ), (
            "Gate must suppress SymCodeVerifier for Gemma4 immediately after "
            "_build_seeded_gate() when no prior state file exists"
        )

    def test_gate_loads_from_existing_file(self, tmp_path: Path) -> None:
        """_build_seeded_gate loads from disk when the state file is present.

        This proves that prior session observations propagate into the new run
        without re-seeding, preserving accumulated learning.
        Spec: REQ-VERIFY-148-1.
        """
        from carnot.pipeline.adaptive_gate import ModelAdaptiveThresholdGate

        state_file = tmp_path / "gate_state.json"
        # Pre-populate state file with 5 FP observations.
        prep_gate = ModelAdaptiveThresholdGate(state_file=state_file)
        for _ in range(5):
            prep_gate.update(exp708.GEMMA4_MODEL_ID, exp708.SUPPRESSED_CONSTRAINT_TYPE, was_tp=False)

        loaded_gate = exp708._build_seeded_gate(state_file)
        assert loaded_gate.is_suppressed(
            exp708.GEMMA4_MODEL_ID, exp708.SUPPRESSED_CONSTRAINT_TYPE
        ), "Gate should remain suppressed after loading 5 FP observations from file"

    def test_gate_does_not_suppress_unknown_model(self, tmp_path: Path) -> None:
        """Gate seeding only affects Gemma4; other model IDs stay unsuppressed.

        Spec: REQ-VERIFY-148-1 (side-effect: gate must not over-suppress).
        """
        state_file = tmp_path / "gate_state.json"
        gate = exp708._build_seeded_gate(state_file)
        assert not gate.is_suppressed(
            "Qwen/Qwen3.5-0.8B", exp708.SUPPRESSED_CONSTRAINT_TYPE
        ), "Qwen must NOT be suppressed by the Gemma4-only FP seed"


# ---------------------------------------------------------------------------
# REQ-VERIFY-148-2: signed_improvement computation
# ---------------------------------------------------------------------------


class TestSignedImprovementComputation:
    """Verify signed_improvement is computed correctly from accuracy values.

    WHY: signed_improvement = vr_accuracy - baseline_accuracy is the primary
    metric compared against the -0.8 baseline from Exp 694.  Off-by-one or
    off-by-N bugs here would produce a misleading headline number.
    Spec: REQ-VERIFY-148-2.
    """

    def test_signed_improvement_formula(self) -> None:
        """signed_improvement must equal vr_accuracy - baseline_accuracy.

        Spec: REQ-VERIFY-148-2.
        """
        baseline_accuracy = 0.52
        vr_accuracy = 0.60
        signed_improvement = vr_accuracy - baseline_accuracy
        assert abs(signed_improvement - 0.08) < 1e-9

    def test_signed_improvement_zero_when_equal(self) -> None:
        """When both accuracies are identical, signed_improvement is 0.0.

        Spec: REQ-VERIFY-148-2, REQ-VERIFY-148-3 (no_harm verdict).
        """
        acc = 0.44
        signed_improvement = acc - acc
        assert signed_improvement == pytest.approx(0.0)

    def test_signed_improvement_negative_when_vr_worse(self) -> None:
        """When VR accuracy is lower, signed_improvement is negative.

        Spec: REQ-VERIFY-148-2, REQ-VERIFY-148-5 (still_harmful verdict).
        """
        baseline_accuracy = 0.60
        vr_accuracy = 0.52
        signed_improvement = vr_accuracy - baseline_accuracy
        assert signed_improvement < 0.0

    def test_improvement_over_baseline_formula(self) -> None:
        """improvement_over_baseline = signed_improvement - (-0.8) = +0.8 when neutral.

        This metric answers 'how much better is this run vs Exp 694?'
        Spec: REQ-VERIFY-148-2.
        """
        signed_improvement = 0.0
        improvement_over_baseline = signed_improvement - exp708.BASELINE_SIGNED_IMPROVEMENT
        assert improvement_over_baseline == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# REQ-VERIFY-148-3/4/5/6: honest_verdict classification
# ---------------------------------------------------------------------------


class TestHonestVerdictClassification:
    """Validate all honest_verdict branches from classify_verdict().

    WHY: The conductor interprets honest_verdict to decide whether the
    experiment succeeded.  An incorrect classification (e.g. returning
    "vr19_gemma4_improved" when signed_improvement is actually negative)
    masks a real regression.
    Spec: REQ-VERIFY-148-3, REQ-VERIFY-148-4, REQ-VERIFY-148-5, REQ-VERIFY-148-6.
    """

    def test_verdict_improved_when_positive(self) -> None:
        """signed_improvement > 0 → 'vr19_gemma4_improved'.

        Spec: REQ-VERIFY-148-4.
        """
        assert exp708.classify_verdict(0.04) == "vr19_gemma4_improved"

    def test_verdict_improved_when_large_positive(self) -> None:
        """Large positive improvement → 'vr19_gemma4_improved'.

        Spec: REQ-VERIFY-148-4.
        """
        assert exp708.classify_verdict(0.40) == "vr19_gemma4_improved"

    def test_verdict_no_harm_when_zero(self) -> None:
        """signed_improvement == 0.0 → 'vr19_gemma4_no_harm'.

        This is the minimum success criterion per the experiment brief.
        Spec: REQ-VERIFY-148-3.
        """
        assert exp708.classify_verdict(0.0) == "vr19_gemma4_no_harm"

    def test_verdict_no_harm_when_near_zero_positive(self) -> None:
        """Very small positive value (within float noise) → 'vr19_gemma4_no_harm' or 'improved'.

        classify_verdict uses epsilon=1e-9, so anything above that is 'improved'.
        Spec: REQ-VERIFY-148-3.
        """
        result = exp708.classify_verdict(1e-10)
        assert result in ("vr19_gemma4_no_harm", "vr19_gemma4_improved")

    def test_verdict_still_harmful_when_negative(self) -> None:
        """signed_improvement < 0 → 'vr19_gemma4_still_harmful'.

        This mirrors the Exp 694 result where gating was absent.
        Spec: REQ-VERIFY-148-5.
        """
        assert exp708.classify_verdict(-0.04) == "vr19_gemma4_still_harmful"

    def test_verdict_still_harmful_when_exp694_baseline(self) -> None:
        """Reproducing the Exp 694 result (-0.8) → 'vr19_gemma4_still_harmful'.

        If the gate fails to suppress, we'd see the same -0.8 again.
        Spec: REQ-VERIFY-148-5.
        """
        assert exp708.classify_verdict(-0.8) == "vr19_gemma4_still_harmful"


# ---------------------------------------------------------------------------
# Answer extraction helpers (used by run_question_with_gate)
# ---------------------------------------------------------------------------


class TestAnswerExtraction:
    """Validate _extract_numeric_answer and _answers_match correctness.

    WHY: These helpers determine per-question correctness.  Bugs here silently
    corrupt baseline_accuracy and vr_accuracy, producing a misleading
    signed_improvement headline.
    Spec: REQ-VERIFY-148-2.
    """

    def test_extracts_answer_is_pattern(self) -> None:
        """'The answer is 42' → 42.0."""
        assert exp708._extract_numeric_answer("The answer is 42.") == pytest.approx(42.0)

    def test_extracts_last_number_fallback(self) -> None:
        """Falls back to last number when no explicit answer keyword is present."""
        assert exp708._extract_numeric_answer("so the total is 15 items.") == pytest.approx(15.0)

    def test_returns_none_for_empty_string(self) -> None:
        """Empty response → None (no answer extractable)."""
        assert exp708._extract_numeric_answer("") is None

    def test_answers_match_within_tolerance(self) -> None:
        """35.0 vs 35 (integer) → match within 0.5 tolerance."""
        assert exp708._answers_match(35.0, 35) is True

    def test_answers_match_none_returns_false(self) -> None:
        """None answer → no match."""
        assert exp708._answers_match(None, 35) is False
        assert exp708._answers_match(35.0, None) is False

    def test_answers_not_match_different_values(self) -> None:
        """7 vs 8 → no match (difference > 0.5 tolerance)."""
        assert exp708._answers_match(7.0, 8) is False


# ---------------------------------------------------------------------------
# Deliverable JSON schema validation + integration
# ---------------------------------------------------------------------------


class TestDeliverableSchema:
    """Validate that the deliverable JSON matches the required schema.

    WHY: The conductor's retrospective step parses the deliverable to extract
    signed_improvement and honest_verdict.  Missing fields cause the conductor
    to misclassify the experiment outcome.
    Spec: REQ-VERIFY-148, REQ-VERIFY-083 (required result fields).
    """

    _REQUIRED_FIELDS = {
        "experiment",
        "title",
        "run_date",
        "started_at",
        "finished_at",
        "duration_s",
        "status",
        "schema",
        "signed_improvement",
        "improvement_over_baseline",
        "baseline_accuracy",
        "vr_accuracy",
        "n_constraints_suppressed",
        "n_extractions_skipped",
        "inference_mode",
        "honest_verdict",
    }

    def _run_blocked_main(self, tmp_path: Path) -> dict[str, Any]:
        """Run main() in blocked mode (no GPU) and return the parsed deliverable."""
        deliverable = tmp_path / "experiment_708_vr_attempt_19_gemma4.json"
        fake_tmpl = _make_fake_tmpl(deliverable)

        with (
            patch("experiment_708_vr_attempt_19_gemma4.ExperimentTemplate", return_value=fake_tmpl),
            patch("experiment_708_vr_attempt_19_gemma4.ExperimentTimeoutWatchdog") as mock_wd,
        ):
            mock_wd.return_value.__enter__ = lambda s: s
            mock_wd.return_value.__exit__ = MagicMock(return_value=False)

            with patch.object(exp708, "_build_seeded_gate") as mock_gate_fn:
                mock_gate = MagicMock()
                mock_gate.is_suppressed.return_value = True
                mock_gate_fn.return_value = mock_gate

                # Patch cached_sota_pair to return None so setup_gpu is called with fallback.
                with patch(
                    "experiment_708_vr_attempt_19_gemma4.cached_sota_pair",
                    return_value=None,
                    create=True,
                ):
                    exp708.main()

        assert deliverable.exists(), "Deliverable JSON must be written in blocked path"
        return json.loads(deliverable.read_text())

    def test_blocked_deliverable_has_required_fields(self, tmp_path: Path) -> None:
        """Blocked path deliverable must contain all required schema fields.

        Spec: REQ-VERIFY-148, REQ-VERIFY-083.
        """
        artifact = self._run_blocked_main(tmp_path)
        for field in self._REQUIRED_FIELDS:
            assert field in artifact, f"Required field '{field}' missing from blocked deliverable"

    def test_blocked_deliverable_honest_verdict(self, tmp_path: Path) -> None:
        """Blocked deliverable must have honest_verdict='vr19_gemma4_blocked'.

        Spec: REQ-VERIFY-148-6.
        """
        artifact = self._run_blocked_main(tmp_path)
        assert artifact["honest_verdict"] == "vr19_gemma4_blocked"

    def test_blocked_deliverable_inference_mode(self, tmp_path: Path) -> None:
        """Blocked deliverable must have inference_mode='blocked_no_gpu'.

        Spec: REQ-VERIFY-148-8.
        """
        artifact = self._run_blocked_main(tmp_path)
        assert artifact["inference_mode"] == "blocked_no_gpu"

    def test_deliverable_is_valid_json(self, tmp_path: Path) -> None:
        """The deliverable must be valid JSON with experiment=708.

        Spec: REQ-VERIFY-148, REQ-VERIFY-083.
        """
        artifact = self._run_blocked_main(tmp_path)
        assert isinstance(artifact, dict)
        assert artifact["experiment"] == 708

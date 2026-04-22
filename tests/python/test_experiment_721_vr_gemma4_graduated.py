"""Tests for Exp 721: Gemma4-A4B VR Graduated Threshold Calibration.

WHY THIS TEST FILE EXISTS:
    Gemma4-A4B VR always produces signed_improvement <= 0 at tight thresholds.
    arXiv 2601.01490 explains this as constraint-induced distortion.  Exp 721
    tests 5 threshold conditions (0.10, 0.20, 0.30, 0.40, abstain) to find the
    distortion-free operating point.

    This suite validates the logic in experiment_721_vr_gemma4_graduated.py:
    1. results_per_condition has exactly 5 entries (REQ-VER-031-1).
    2. Abstain mode only fires constraint when EORM confidence > 0.90 (REQ-VER-031-4).
    3. classify_verdict covers all three honest_verdict branches (REQ-VER-031-5).
    4. Blocked artifact has all required schema fields (REQ-VERIFY-083).
    5. The deliverable JSON (blocked or live) is valid (REQ-VER-031-6).

Spec: REQ-VER-031, SCENARIO-VER-038
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import experiment_721_vr_gemma4_graduated as exp721  # noqa: E402

_DELIVERABLE = _REPO_ROOT / "results" / "experiment_721_vr_gemma4_graduated.json"


# ---------------------------------------------------------------------------
# Shared helper: fake ExperimentTemplate
# ---------------------------------------------------------------------------


def _make_fake_tmpl(deliverable: Path, all_healthy: bool = False) -> Any:
    """Stand-in for ExperimentTemplate that avoids MagicMock 'assert_' issues.

    WHY: Python MagicMock raises AttributeError on attributes starting with
    'assert_' unless they are standard assertion helpers.  A plain class avoids
    this entirely while still letting us track what was called.

    Args:
        deliverable: Path where the artifact JSON will be written.
        all_healthy: Whether setup_gpu() should report GPU as healthy.
    """

    def _build_result(data: dict, **kw: Any) -> dict:
        return {
            "experiment": 721,
            "title": "Gemma4-A4B VR Graduated Threshold Calibration (arXiv 2601.01490)",
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

        @staticmethod
        def setup() -> None:
            pass

        @staticmethod
        def assert_deliverable_written() -> None:
            pass

        @staticmethod
        def checkpoint_save(data: Any, step: int = 0) -> None:
            pass

        @staticmethod
        def build_result(data: dict, **kw: Any) -> dict:
            return _build_result(data, **kw)

        @staticmethod
        def setup_gpu(specs: Any) -> dict:
            return {"all_healthy": all_healthy, "models": []}

    return _FakeTmpl()


def _run_blocked_main(tmp_path: Path) -> dict[str, Any]:
    """Run main() with no GPU available; return parsed deliverable dict."""
    deliverable = tmp_path / "experiment_721_vr_gemma4_graduated.json"
    fake_tmpl = _make_fake_tmpl(deliverable, all_healthy=False)

    with (
        patch("experiment_721_vr_gemma4_graduated.ExperimentTemplate", return_value=fake_tmpl),
        patch("experiment_721_vr_gemma4_graduated.ExperimentTimeoutWatchdog") as mock_wd,
        patch(
            "experiment_721_vr_gemma4_graduated.cached_sota_pair",
            return_value=None,
            create=True,
        ),
    ):
        mock_wd.return_value.__enter__ = lambda s: s
        mock_wd.return_value.__exit__ = MagicMock(return_value=False)
        exp721.main()

    assert deliverable.exists(), "Deliverable must be written in blocked path"
    return json.loads(deliverable.read_text())


# ---------------------------------------------------------------------------
# REQ-VER-031-1: results_per_condition has exactly 5 entries
# ---------------------------------------------------------------------------


class TestResultsPerConditionCount:
    """Verify results_per_condition always has exactly 5 entries.

    WHY: The conductor reads results_per_condition to select the optimal
    threshold.  Missing entries indicate a condition was skipped, which
    silently corrupts the threshold selection.
    Spec: REQ-VER-031-1, SCENARIO-VER-038.
    """

    def test_classify_verdict_receives_five_conditions(self) -> None:
        """classify_verdict processes exactly 5 condition results.

        We pass 5 synthetic condition results to validate that the function
        handles all 5 without dropping any.
        Spec: REQ-VER-031-1.
        """
        conditions = [
            {"threshold": 0.10, "signed_improvement": -0.10},
            {"threshold": 0.20, "signed_improvement": -0.05},
            {"threshold": 0.30, "signed_improvement": -0.02},
            {"threshold": 0.40, "signed_improvement": -0.01},
            {"threshold": "abstain", "signed_improvement": -0.01},
        ]
        verdict, optimal = exp721.classify_verdict(conditions)
        # With all negative, we expect the distortion-confirmed verdict.
        assert verdict == "gemma4_distortion_confirmed_all_negative"
        assert optimal is None

    def test_conditions_list_has_five_entries(self) -> None:
        """Module-level _CONDITIONS constant has exactly 5 entries.

        This is the canonical list of conditions that experiment main() iterates.
        Spec: REQ-VER-031-1.
        """
        assert len(exp721._CONDITIONS) == 5, (
            f"Expected 5 conditions, got {len(exp721._CONDITIONS)} — "
            "REQ-VER-031-1 requires exactly 5 threshold conditions"
        )

    def test_conditions_include_all_required_thresholds(self) -> None:
        """_CONDITIONS must contain 0.10, 0.20, 0.30, 0.40, and 'abstain'.

        Spec: REQ-VER-031-1.
        """
        cond_set = set(exp721._CONDITIONS)
        for expected in [0.10, 0.20, 0.30, 0.40, "abstain"]:
            assert expected in cond_set, (
                f"Threshold condition {expected!r} is missing from _CONDITIONS — "
                "REQ-VER-031-1 requires all five conditions"
            )

    def test_blocked_artifact_has_empty_results_per_condition(self, tmp_path: Path) -> None:
        """Blocked artifact must contain results_per_condition (empty list).

        Even when GPU is unavailable, the field must be present so downstream
        tooling can parse the artifact schema without branching.
        Spec: REQ-VER-031-6.
        """
        artifact = _run_blocked_main(tmp_path)
        assert "results_per_condition" in artifact, (
            "results_per_condition field missing from blocked artifact"
        )
        assert isinstance(artifact["results_per_condition"], list)


# ---------------------------------------------------------------------------
# REQ-VER-031-4: Abstain mode only fires when EORM confidence > 0.90
# ---------------------------------------------------------------------------


class TestAbstainMode:
    """Verify the abstain gating logic fires only above the confidence threshold.

    WHY: The abstain mode is the key differentiator from numeric thresholds.
    If it fires too often (low gate), it behaves like threshold=0.10 (tight,
    distorted).  If it fires too rarely (very high gate), it never corrects
    anything.  The REQ-VER-031-4 contract sets the gate at 0.90.
    Spec: REQ-VER-031-4, SCENARIO-VER-038.
    """

    def test_abstain_mode_fires_when_confidence_above_threshold(self) -> None:
        """Constraint is applied when EORM confidence > 0.90 in abstain mode.

        We use a mock pipeline to verify the VR path is taken.
        Spec: REQ-VER-031-4.
        """
        mock_pipeline = MagicMock()
        mock_pipeline._generate.return_value = "42"  # short response → high confidence
        mock_vr_result = MagicMock()
        mock_vr_result.final_response = "42"
        mock_pipeline.verify_and_repair.return_value = mock_vr_result

        # "42" is 2 chars → _eorm_confidence returns 0.95 > 0.90 → should apply
        result = exp721._run_one_question_with_threshold(
            mock_pipeline, "What is 6*7?", 42, "abstain"
        )

        assert result["constraint_applied"] is True, (
            "Abstain mode must apply constraint when EORM confidence > 0.90 — "
            "REQ-VER-031-4"
        )
        mock_pipeline.verify_and_repair.assert_called_once()

    def test_abstain_mode_does_not_fire_when_confidence_below_threshold(self) -> None:
        """Constraint is NOT applied when EORM confidence <= 0.90 in abstain mode.

        A long, detailed response gets low EORM confidence (model is probably
        correct) — no repair should be triggered.
        Spec: REQ-VER-031-4.
        """
        mock_pipeline = MagicMock()
        # Long response → _eorm_confidence returns 0.35 < 0.90 → should NOT apply
        long_response = "Step 1: multiply 6 by 7. Step 2: 6 * 7 = 42. The answer is 42."
        mock_pipeline._generate.return_value = long_response

        result = exp721._run_one_question_with_threshold(
            mock_pipeline, "What is 6*7?", 42, "abstain"
        )

        assert result["constraint_applied"] is False, (
            "Abstain mode must NOT apply constraint when EORM confidence <= 0.90 — "
            "REQ-VER-031-4"
        )
        mock_pipeline.verify_and_repair.assert_not_called()

    def test_eorm_confidence_short_response_is_high(self) -> None:
        """Very short responses get EORM confidence > 0.90.

        WHY: Short responses are likely incomplete ('I don't know' or bare
        numbers) — the EORM should be highly confident repair is needed.
        Spec: REQ-VER-031-4.
        """
        conf = exp721._eorm_confidence("42", "What is 6*7?")
        assert conf > _ABSTAIN_CONFIDENCE_THRESHOLD_FOR_TEST, (
            f"Short response should get EORM confidence > 0.90, got {conf}"
        )

    def test_eorm_confidence_long_response_is_low(self) -> None:
        """Long CoT-style responses get EORM confidence < 0.90.

        WHY: A detailed reasoning chain is less likely to need repair — the EORM
        should leave it alone in abstain mode.
        Spec: REQ-VER-031-4.
        """
        long_resp = "Step 1: multiply 6 by 7. Step 2: 6 * 7 = 42. The answer is 42."
        conf = exp721._eorm_confidence(long_resp, "What is 6*7?")
        assert conf < _ABSTAIN_CONFIDENCE_THRESHOLD_FOR_TEST, (
            f"Long response should get EORM confidence < 0.90, got {conf}"
        )

    def test_abstain_mode_passes_confidence_threshold(self) -> None:
        """_ABSTAIN_CONFIDENCE_THRESHOLD is exactly 0.90 per REQ-VER-031-4.

        Spec: REQ-VER-031-4.
        """
        assert exp721._ABSTAIN_CONFIDENCE_THRESHOLD == pytest.approx(0.90), (
            f"Expected 0.90, got {exp721._ABSTAIN_CONFIDENCE_THRESHOLD} — "
            "REQ-VER-031-4 specifies the gate at EORM confidence > 0.90"
        )


_ABSTAIN_CONFIDENCE_THRESHOLD_FOR_TEST = exp721._ABSTAIN_CONFIDENCE_THRESHOLD


# ---------------------------------------------------------------------------
# REQ-VER-031-5: classify_verdict covers all three honest_verdict branches
# ---------------------------------------------------------------------------


class TestClassifyVerdict:
    """Verify all three honest_verdict branches from classify_verdict().

    WHY: The conductor interprets honest_verdict to dispatch the next task:
    - "gemma4_optimal_threshold_found" → schedule threshold-tuning work
    - "gemma4_abstain_wins" → deploy with abstain mode (low cost)
    - "gemma4_distortion_confirmed_all_negative" → close Gemma4 VR investigation
    Spec: REQ-VER-031-5, SCENARIO-VER-038.
    """

    def _make_conditions(self, improvements: dict) -> list[dict]:
        """Build a conditions list from a threshold → signed_improvement mapping."""
        all_thresholds = [0.10, 0.20, 0.30, 0.40, "abstain"]
        return [
            {"threshold": t, "signed_improvement": improvements.get(t, -0.01)}
            for t in all_thresholds
        ]

    def test_optimal_found_when_numeric_threshold_positive(self) -> None:
        """When a numeric threshold produces signed_improvement > 0, verdict is 'found'.

        Spec: REQ-VER-031-5.
        """
        conditions = self._make_conditions({0.40: 0.05})
        verdict, optimal = exp721.classify_verdict(conditions)
        assert verdict == "gemma4_optimal_threshold_found"
        assert optimal == pytest.approx(0.40)

    def test_optimal_threshold_is_best_positive(self) -> None:
        """When multiple numeric thresholds are positive, optimal is the highest improvement.

        Spec: REQ-VER-031-5.
        """
        conditions = self._make_conditions({0.30: 0.03, 0.40: 0.07})
        verdict, optimal = exp721.classify_verdict(conditions)
        assert verdict == "gemma4_optimal_threshold_found"
        assert optimal == pytest.approx(0.40)  # 0.07 > 0.03

    def test_abstain_wins_when_only_abstain_positive(self) -> None:
        """When only the abstain condition is positive, verdict is 'abstain_wins'.

        Spec: REQ-VER-031-5.
        """
        conditions = self._make_conditions({"abstain": 0.04})
        verdict, optimal = exp721.classify_verdict(conditions)
        assert verdict == "gemma4_abstain_wins"
        assert optimal == "abstain"

    def test_distortion_confirmed_when_all_negative(self) -> None:
        """When all 5 conditions produce signed_improvement <= 0, verdict is 'all_negative'.

        This is the worst-case outcome — confirms Gemma4 is in the distortion
        regime across the entire threshold range tested.
        Spec: REQ-VER-031-5.
        """
        conditions = self._make_conditions({})  # all default to -0.01
        verdict, optimal = exp721.classify_verdict(conditions)
        assert verdict == "gemma4_distortion_confirmed_all_negative"
        assert optimal is None

    def test_distortion_confirmed_when_all_zero(self) -> None:
        """signed_improvement == 0 still counts as non-positive (distortion confirmed).

        Zero means no improvement — not a positive outcome.
        Spec: REQ-VER-031-5.
        """
        conditions = [
            {"threshold": t, "signed_improvement": 0.0}
            for t in [0.10, 0.20, 0.30, 0.40, "abstain"]
        ]
        verdict, optimal = exp721.classify_verdict(conditions)
        assert verdict == "gemma4_distortion_confirmed_all_negative"
        assert optimal is None

    def test_numeric_beats_abstain_for_optimal(self) -> None:
        """When both a numeric threshold AND abstain are positive, numeric takes precedence.

        The 'abstain_wins' verdict only fires when NO numeric threshold is positive.
        Spec: REQ-VER-031-5.
        """
        conditions = self._make_conditions({0.30: 0.02, "abstain": 0.08})
        verdict, optimal = exp721.classify_verdict(conditions)
        # numeric threshold is positive → 'found' (not 'abstain_wins')
        assert verdict == "gemma4_optimal_threshold_found"
        assert optimal == pytest.approx(0.30)


# ---------------------------------------------------------------------------
# Blocked artifact schema validation (REQ-VER-031-6, REQ-VERIFY-083)
# ---------------------------------------------------------------------------


class TestBlockedArtifactSchema:
    """Validate blocked artifact contains all required schema fields.

    WHY: The conductor reads blocked and live deliverables in the same code path.
    Missing fields cause KeyError in the retrospective agent.
    Spec: REQ-VER-031-6, REQ-VERIFY-083.
    """

    _REQUIRED_BASE_FIELDS = {
        "experiment",
        "title",
        "run_date",
        "started_at",
        "finished_at",
        "duration_s",
        "status",
        "schema",
    }
    _REQUIRED_EXP_FIELDS = {
        "results_per_condition",
        "honest_verdict",
        "optimal_threshold",
        "n_conditions_tested",
        "models_used",
        "batch_log",
    }

    def test_blocked_has_all_base_fields(self, tmp_path: Path) -> None:
        """Blocked artifact has all standard ExperimentTemplate fields.

        Spec: REQ-VERIFY-083.
        """
        artifact = _run_blocked_main(tmp_path)
        for field in self._REQUIRED_BASE_FIELDS:
            assert field in artifact, f"Base field '{field}' missing from blocked artifact"

    def test_blocked_has_all_exp_fields(self, tmp_path: Path) -> None:
        """Blocked artifact has all experiment-specific fields.

        Spec: REQ-VER-031-6, SCENARIO-VER-038.
        """
        artifact = _run_blocked_main(tmp_path)
        for field in self._REQUIRED_EXP_FIELDS:
            assert field in artifact, f"Experiment field '{field}' missing from blocked artifact"

    def test_blocked_artifact_experiment_id_is_721(self, tmp_path: Path) -> None:
        """Blocked artifact must have experiment id 721.

        Spec: REQ-VER-031.
        """
        artifact = _run_blocked_main(tmp_path)
        assert artifact["experiment"] == 721

    def test_blocked_artifact_is_valid_json_dict(self, tmp_path: Path) -> None:
        """Deliverable must be valid JSON parseable as a dict.

        Spec: REQ-VERIFY-083.
        """
        artifact = _run_blocked_main(tmp_path)
        assert isinstance(artifact, dict)

    def test_blocked_artifact_status_is_blocked(self, tmp_path: Path) -> None:
        """Blocked artifact status must be 'blocked'.

        Spec: REQ-VER-031.
        """
        artifact = _run_blocked_main(tmp_path)
        assert artifact["status"] == "blocked"

    def test_blocked_artifact_honest_verdict_is_blocked(self, tmp_path: Path) -> None:
        """Blocked artifact honest_verdict must indicate GPU was unavailable.

        Spec: REQ-VER-031-5.
        """
        artifact = _run_blocked_main(tmp_path)
        assert artifact["honest_verdict"] == "gemma4_blocked_no_gpu"

    def test_on_disk_deliverable_has_required_fields(self) -> None:
        """The committed deliverable on disk must have all required fields.

        Reads the actual results/experiment_721_vr_gemma4_graduated.json.
        Passes once the deliverable is written.
        Spec: REQ-VER-031-6, SCENARIO-VER-038.
        """
        if not _DELIVERABLE.exists():
            pytest.skip("Deliverable not yet written — run experiment_721 first")

        artifact = json.loads(_DELIVERABLE.read_text())
        for field in self._REQUIRED_BASE_FIELDS | self._REQUIRED_EXP_FIELDS:
            assert field in artifact, f"Field '{field}' missing from on-disk deliverable"

    def test_on_disk_deliverable_results_per_condition_count(self) -> None:
        """On-disk deliverable results_per_condition has exactly 5 entries if live.

        Spec: REQ-VER-031-1, SCENARIO-VER-038.
        """
        if not _DELIVERABLE.exists():
            pytest.skip("Deliverable not yet written — run experiment_721 first")

        artifact = json.loads(_DELIVERABLE.read_text())
        if artifact.get("status") == "blocked":
            pytest.skip("Deliverable is blocked — results_per_condition will be empty")

        rpc = artifact.get("results_per_condition", [])
        assert len(rpc) == 5, (
            f"results_per_condition must have exactly 5 entries, got {len(rpc)} — "
            "REQ-VER-031-1"
        )

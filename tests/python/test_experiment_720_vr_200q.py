"""Tests for Exp 720: VR 200q Scale Evaluation — RETRO-033 Resolution.

WHY THIS TEST FILE EXISTS:
    19 consecutive VR attempts at 100q showed signed_improvement <= 0.
    RETRO-033 has been open for 8 milestones.  Exp 720 runs at 200q to either
    confirm VR works at scale or close RETRO-033 as "not_viable_at_scale".

    This suite validates the logic in experiment_720_vr_200q_qwen.py:
    1. signed_improvement checkpoint fields are present in the artifact.
    2. BatchedInferenceRunner produces a non-empty batch_log (REQ-VER-030-7).
    3. classify_verdict covers all branches (REQ-VER-030-3/4/5).
    4. compute_signed_improvement_at computes correctly.
    5. The deliverable JSON (blocked or live) has all required schema fields.

Spec: REQ-VER-030, SCENARIO-VER-037
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

import experiment_720_vr_200q_qwen as exp720  # noqa: E402
from scripts.experiment_template import BatchedInferenceRunner  # noqa: E402

_DELIVERABLE = _REPO_ROOT / "results" / "experiment_720_vr_200q_qwen.json"


# ---------------------------------------------------------------------------
# Shared helper: fake ExperimentTemplate (avoids MagicMock assert_ issues)
# ---------------------------------------------------------------------------


def _make_fake_tmpl(deliverable: Path, all_healthy: bool = False) -> Any:
    """Return a plain-object stand-in for ExperimentTemplate.

    Python 3.8+ MagicMock raises AttributeError when accessing attributes
    starting with 'assert_' that are not standard mock assertion methods.
    Using a plain class avoids this entirely.

    Args:
        deliverable: Path where the artifact JSON will be written.
        all_healthy: Whether setup_gpu() should report GPU healthy.
    """

    def _build_result(data: dict, **kw: Any) -> dict:
        return {
            "experiment": 720,
            "title": "VR 200q Scale Evaluation: RETRO-033 Resolution",
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
        assert_deliverable_written = staticmethod(lambda: None)
        checkpoint_save = staticmethod(lambda data, step=None: None)
        build_result = staticmethod(_build_result)

        @staticmethod
        def setup_gpu(specs: Any) -> dict:
            return {"all_healthy": all_healthy, "models": []}

    return _FakeTmpl()


def _run_blocked_main(tmp_path: Path) -> dict[str, Any]:
    """Run main() with no GPU available; return parsed deliverable dict."""
    deliverable = tmp_path / "experiment_720_vr_200q_qwen.json"
    fake_tmpl = _make_fake_tmpl(deliverable, all_healthy=False)

    with (
        patch("experiment_720_vr_200q_qwen.ExperimentTemplate", return_value=fake_tmpl),
        patch("experiment_720_vr_200q_qwen.ExperimentTimeoutWatchdog") as mock_wd,
        patch("experiment_720_vr_200q_qwen.cached_sota_pair", return_value=None, create=True),
    ):
        mock_wd.return_value.__enter__ = lambda s: s
        mock_wd.return_value.__exit__ = MagicMock(return_value=False)
        exp720.main()

    assert deliverable.exists(), "Deliverable must be written in blocked path"
    return json.loads(deliverable.read_text())


# ---------------------------------------------------------------------------
# REQ-VER-030-2: signed_improvement checkpoint fields present in artifact
# ---------------------------------------------------------------------------


class TestSignedImprovementFieldsPresent:
    """Verify the four checkpoint fields are present in the artifact.

    WHY: The conductor and retrospective agent both read these fields to
    determine whether the 200q signal is stable across question counts.
    Missing fields would cause a KeyError in the retrospective parser.
    Spec: REQ-VER-030-2, SCENARIO-VER-037.
    """

    _CHECKPOINT_FIELDS = {
        "signed_improvement_50q",
        "signed_improvement_100q",
        "signed_improvement_150q",
        "signed_improvement_200q",
    }

    def test_blocked_artifact_has_all_checkpoint_fields(self, tmp_path: Path) -> None:
        """Blocked artifact must contain all four signed_improvement checkpoint fields.

        Even when GPU is unavailable, the fields must be present (as None)
        so downstream tooling can parse the artifact schema without branching.
        Spec: REQ-VER-030-2.
        """
        artifact = _run_blocked_main(tmp_path)
        for field in self._CHECKPOINT_FIELDS:
            assert field in artifact, (
                f"Field '{field}' missing from blocked artifact — "
                f"REQ-VER-030-2 requires all checkpoint fields present"
            )

    def test_blocked_artifact_checkpoint_fields_are_none(self, tmp_path: Path) -> None:
        """In blocked mode, checkpoint fields are None (no inference was run).

        Spec: REQ-VER-030-2.
        """
        artifact = _run_blocked_main(tmp_path)
        for field in self._CHECKPOINT_FIELDS:
            assert artifact[field] is None, (
                f"Field '{field}' must be None in blocked artifact — "
                f"no inference was run"
            )

    def test_blocked_artifact_has_honest_verdict(self, tmp_path: Path) -> None:
        """Blocked artifact must have honest_verdict field.

        Spec: REQ-VER-030-3/4/5 (verdict must always be present).
        """
        artifact = _run_blocked_main(tmp_path)
        assert "honest_verdict" in artifact
        assert artifact["honest_verdict"] == "vr_blocked_no_gpu"

    def test_compute_signed_improvement_at_50q(self) -> None:
        """compute_signed_improvement_at slices correctly at 50 questions.

        Spec: REQ-VER-030-2.
        """
        baseline = [True] * 30 + [False] * 20  # 30/50 correct = 0.60
        vr = [True] * 35 + [False] * 15  # 35/50 correct = 0.70
        si = exp720.compute_signed_improvement_at(50, baseline, vr)
        assert abs(si - 0.10) < 1e-9, f"Expected 0.10, got {si}"

    def test_compute_signed_improvement_at_100q_uses_first_100(self) -> None:
        """compute_signed_improvement_at ignores questions beyond n_questions.

        Spec: REQ-VER-030-2.
        """
        # 200 items: first 100 baseline=50/100 correct, vr=60/100 correct.
        # Last 100: baseline=100/100, vr=0/100 — must NOT affect 100q metric.
        baseline = [True] * 50 + [False] * 50 + [True] * 100
        vr = [True] * 60 + [False] * 40 + [False] * 100
        si = exp720.compute_signed_improvement_at(100, baseline, vr)
        assert abs(si - 0.10) < 1e-9, f"Expected 0.10, got {si}"

    def test_compute_signed_improvement_at_zero_when_empty(self) -> None:
        """Returns 0.0 when sliced list is empty (no questions run yet).

        Spec: REQ-VER-030-2.
        """
        si = exp720.compute_signed_improvement_at(50, [], [])
        assert si == 0.0


# ---------------------------------------------------------------------------
# REQ-VER-030-7: BatchedInferenceRunner produces non-empty batch_log
# ---------------------------------------------------------------------------


class TestBatchedInferenceRunnerUsed:
    """Verify BatchedInferenceRunner is used and produces batch_log entries.

    WHY: The conductor checks batch_log to confirm batching was used (not
    sequential per-question inference).  An empty batch_log would indicate
    the runner was bypassed, undermining the throughput gain from Exp 721
    retrospective.  Spec: REQ-VER-030-7.
    """

    def test_batch_runner_produces_nonempty_batch_log(self) -> None:
        """BatchedInferenceRunner.batch_log is non-empty after run_batch().

        This is the canonical proof that BatchedInferenceRunner was invoked in
        any successful inference run.  We test the runner directly (not the
        deliverable) because the blocked path skips inference.
        Spec: REQ-VER-030-7.
        """
        responses = iter(["resp_a", "resp_b", "resp_c"])

        def _fake_runner(item: str) -> str:
            return next(responses)

        bir = BatchedInferenceRunner(_fake_runner, batch_size=2)
        results = bir.run_batch(["q1", "q2", "q3"])

        assert len(bir.batch_log) > 0, (
            "batch_log must be non-empty after run_batch — "
            "this proves BatchedInferenceRunner was used (REQ-VER-030-7)"
        )
        assert len(results) == 3, "run_batch must return result for every question"

    def test_batch_log_has_required_keys(self) -> None:
        """Each batch_log entry must have batch_id, batch_size, batch_time_s.

        Spec: REQ-VER-030-7.
        """

        def _fake_runner(item: str) -> str:
            return "42"

        bir = BatchedInferenceRunner(_fake_runner, batch_size=4)
        bir.run_batch(["q1", "q2", "q3", "q4"])

        for entry in bir.batch_log:
            assert "batch_id" in entry, f"batch_log entry missing 'batch_id': {entry}"
            assert "batch_size" in entry, f"batch_log entry missing 'batch_size': {entry}"
            assert "batch_time_s" in entry, f"batch_log entry missing 'batch_time_s': {entry}"

    def test_batch_log_batch_size_matches_configured(self) -> None:
        """batch_log entries report the actual batch size used.

        Spec: REQ-VER-030-7.
        """

        def _fake_runner(item: str) -> str:
            return "ok"

        bir = BatchedInferenceRunner(_fake_runner, batch_size=8)
        bir.run_batch([f"q{i}" for i in range(8)])

        assert bir.batch_log[0]["batch_size"] == 8


# ---------------------------------------------------------------------------
# REQ-VER-030-3/4/5: classify_verdict covers all branches
# ---------------------------------------------------------------------------


class TestClassifyVerdict:
    """Verify all honest_verdict branches from classify_verdict().

    WHY: The conductor interprets honest_verdict to decide whether to close
    RETRO-033.  A wrong classification (e.g. "vr_finally_positive" when the
    signed_improvement is actually negative) would reopen work that is genuinely
    blocked.  All three branches must be exercised.
    Spec: REQ-VER-030-3, REQ-VER-030-4, REQ-VER-030-5.
    """

    def test_finally_positive_when_above_threshold(self) -> None:
        """signed_improvement_200q > 0.01 → 'vr_finally_positive'.

        Spec: REQ-VER-030-3.
        """
        assert exp720.classify_verdict(0.02) == "vr_finally_positive"

    def test_finally_positive_when_large(self) -> None:
        """Large positive → 'vr_finally_positive'.

        Spec: REQ-VER-030-3.
        """
        assert exp720.classify_verdict(0.50) == "vr_finally_positive"

    def test_marginal_when_just_above_zero(self) -> None:
        """0 < signed_improvement_200q <= 0.01 → 'vr_marginal'.

        Spec: REQ-VER-030-4.
        """
        assert exp720.classify_verdict(0.005) == "vr_marginal"

    def test_marginal_at_boundary(self) -> None:
        """Exactly 0.01 → 'vr_marginal' (boundary inclusive on marginal side).

        Spec: REQ-VER-030-4.
        """
        assert exp720.classify_verdict(0.01) == "vr_marginal"

    def test_not_viable_when_zero(self) -> None:
        """signed_improvement_200q == 0.0 → 'vr_not_viable_at_scale'.

        This is the 19-consecutive-failure outcome replicated at 200q.
        Spec: REQ-VER-030-5.
        """
        assert exp720.classify_verdict(0.0) == "vr_not_viable_at_scale"

    def test_not_viable_when_negative(self) -> None:
        """signed_improvement_200q < 0 → 'vr_not_viable_at_scale'.

        Spec: REQ-VER-030-5.
        """
        assert exp720.classify_verdict(-0.05) == "vr_not_viable_at_scale"

    def test_not_viable_when_strongly_negative(self) -> None:
        """Strong negative → 'vr_not_viable_at_scale'.

        Mirrors Exp 694 result (-0.8) extended to 200q scale.
        Spec: REQ-VER-030-5.
        """
        assert exp720.classify_verdict(-0.8) == "vr_not_viable_at_scale"


# ---------------------------------------------------------------------------
# Answer extraction helpers
# ---------------------------------------------------------------------------


class TestAnswerExtraction:
    """Validate _extract_numeric_answer and _answers_match correctness.

    WHY: These helpers determine per-question correctness at each checkpoint.
    Bugs here silently corrupt all four signed_improvement values.
    Spec: REQ-VER-030-2.
    """

    def test_extracts_answer_is_pattern(self) -> None:
        """'The answer is 42' → 42.0. Spec: REQ-VER-030-2."""
        assert exp720._extract_numeric_answer("The answer is 42.") == pytest.approx(42.0)

    def test_extracts_last_number_fallback(self) -> None:
        """Falls back to last number when no explicit answer keyword present."""
        assert exp720._extract_numeric_answer("total is 15 items.") == pytest.approx(15.0)

    def test_returns_none_for_empty(self) -> None:
        """Empty string → None."""
        assert exp720._extract_numeric_answer("") is None

    def test_returns_none_for_no_digits(self) -> None:
        """String with no digits → None."""
        assert exp720._extract_numeric_answer("no numbers here") is None

    def test_answers_match_within_tolerance(self) -> None:
        """35.0 vs 35 → match (REQ-VER-030-2)."""
        assert exp720._answers_match(35.0, 35) is True

    def test_answers_not_match_outside_tolerance(self) -> None:
        """7.0 vs 8 → no match (difference = 1.0 > tol=0.5)."""
        assert exp720._answers_match(7.0, 8) is False

    def test_answers_match_none_is_false(self) -> None:
        """None answers → False."""
        assert exp720._answers_match(None, 35) is False
        assert exp720._answers_match(35.0, None) is False


# ---------------------------------------------------------------------------
# Deliverable schema validation
# ---------------------------------------------------------------------------


class TestDeliverableSchema:
    """Validate that deliverable JSON contains all required schema fields.

    WHY: The conductor reads blocked and live deliverables in the same code path.
    Missing fields cause KeyError in the retrospective agent.
    Spec: REQ-VER-030, REQ-VERIFY-083 (required result fields).
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
        "signed_improvement_50q",
        "signed_improvement_100q",
        "signed_improvement_150q",
        "signed_improvement_200q",
        "honest_verdict",
        "inference_mode",
        "batch_log",
        "retro_033_resolution",
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

        Spec: REQ-VER-030-2, SCENARIO-VER-037.
        """
        artifact = _run_blocked_main(tmp_path)
        for field in self._REQUIRED_EXP_FIELDS:
            assert field in artifact, f"Experiment field '{field}' missing from blocked artifact"

    def test_blocked_artifact_experiment_id(self, tmp_path: Path) -> None:
        """Blocked artifact experiment id must be 720.

        Spec: REQ-VER-030.
        """
        artifact = _run_blocked_main(tmp_path)
        assert artifact["experiment"] == 720

    def test_blocked_artifact_is_valid_json(self, tmp_path: Path) -> None:
        """Deliverable must be valid JSON parseable as a dict.

        Spec: REQ-VERIFY-083.
        """
        artifact = _run_blocked_main(tmp_path)
        assert isinstance(artifact, dict)

    def test_blocked_artifact_status_is_blocked(self, tmp_path: Path) -> None:
        """Blocked artifact status must be 'blocked'.

        Spec: REQ-VER-030.
        """
        artifact = _run_blocked_main(tmp_path)
        assert artifact["status"] == "blocked"

    def test_on_disk_deliverable_has_required_fields(self) -> None:
        """The committed deliverable on disk must have all required fields.

        This test reads the actual results/experiment_720_vr_200q_qwen.json
        file and validates its schema.  Passes once the deliverable is written.
        Spec: REQ-VER-030, SCENARIO-VER-037.
        """
        if not _DELIVERABLE.exists():
            pytest.skip("Deliverable not yet written — run experiment_720 first")

        artifact = json.loads(_DELIVERABLE.read_text())
        for field in self._REQUIRED_BASE_FIELDS | self._REQUIRED_EXP_FIELDS:
            assert field in artifact, f"Field '{field}' missing from on-disk deliverable"

    def test_on_disk_deliverable_batch_log_nonempty(self) -> None:
        """The on-disk deliverable must have non-empty batch_log if inference ran.

        Batch_log being non-empty proves BatchedInferenceRunner was actually
        used during the live inference run (REQ-VER-030-7).
        Spec: REQ-VER-030-7, SCENARIO-VER-037.
        """
        if not _DELIVERABLE.exists():
            pytest.skip("Deliverable not yet written — run experiment_720 first")

        artifact = json.loads(_DELIVERABLE.read_text())
        if artifact.get("inference_mode") == "blocked_no_gpu":
            pytest.skip("Deliverable is blocked — batch_log will be empty")

        batch_log = artifact.get("batch_log", [])
        assert len(batch_log) > 0, (
            "batch_log must be non-empty in live inference deliverable — "
            "this proves BatchedInferenceRunner was used (REQ-VER-030-7)"
        )

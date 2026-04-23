"""Tests for Exp 742: RETRO-033 Confirmation Trial (seed 999).

WHY THIS TEST FILE EXISTS:
    Exp 720 produced a single positive signed_improvement=0.00510 at 200q scale
    with seed=218.  One positive result at 0.51pp is within statistical noise —
    REQ-VERIFY-150 requires >= 2 independent trials before RETRO-033 can be
    definitively closed.  Exp 742 runs with seed=999 to either confirm or reopen.

    This suite validates the logic in experiment_742_retro033_confirmation.py:
    1. classify_verdict maps correctly to all three threshold branches.
    2. compute_signed_improvement is computed correctly from accuracy pair.
    3. shuffle_questions produces a different order with seed=999 vs seed=218.
    4. Blocked artifact contains all required schema fields.
    5. On-disk deliverable (if present) has correct schema.

Spec: REQ-VERIFY-150, SCENARIO-VERIFY-200
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

import experiment_742_retro033_confirmation as exp742  # noqa: E402

_DELIVERABLE = _REPO_ROOT / "results" / "experiment_742_retro033_confirmation.json"


# ---------------------------------------------------------------------------
# Shared helper: fake ExperimentTemplate (avoids MagicMock assert_ issues)
# ---------------------------------------------------------------------------


def _make_fake_tmpl(deliverable: Path, all_healthy: bool = False) -> Any:
    """Return a plain-object stand-in for ExperimentTemplate.

    MagicMock raises AttributeError on attributes starting with 'assert_'.
    Using a plain class avoids this.

    Args:
        deliverable: Path where the artifact JSON will be written.
        all_healthy: Whether setup_gpu() should report GPU healthy.
    """

    def _build_result(data: dict, **kw: Any) -> dict:
        return {
            "experiment": 742,
            "title": "RETRO-033 Confirmation Trial: seed 999",
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
    deliverable = tmp_path / "experiment_742_retro033_confirmation.json"
    fake_tmpl = _make_fake_tmpl(deliverable, all_healthy=False)

    with (
        patch("experiment_742_retro033_confirmation.ExperimentTemplate", return_value=fake_tmpl),
        patch("experiment_742_retro033_confirmation.ExperimentTimeoutWatchdog") as mock_wd,
        patch("experiment_742_retro033_confirmation.cached_sota_pair", return_value=None, create=True),
    ):
        mock_wd.return_value.__enter__ = lambda s: s
        mock_wd.return_value.__exit__ = MagicMock(return_value=False)
        exp742.main()

    assert deliverable.exists(), "Deliverable must be written in blocked path"
    return json.loads(deliverable.read_text())


# ---------------------------------------------------------------------------
# REQ-VERIFY-150-3: classify_verdict covers all three threshold branches
# ---------------------------------------------------------------------------


class TestClassifyVerdict:
    """Verify all honest_verdict branches from classify_verdict().

    WHY: The conductor reads honest_verdict to update RETRO-033 status.
    An incorrect classification (e.g., marking a negative result as "confirmed_closed")
    would falsely close a genuine open issue and suppress necessary follow-up work.
    Spec: REQ-VERIFY-150-3, SCENARIO-VERIFY-200.
    """

    def test_confirmed_closed_when_above_noise_floor(self) -> None:
        """signed_improvement > 0.003 → 'retro033_confirmed_closed'.

        0.003 is the conservative noise floor for 200q.
        Spec: REQ-VERIFY-150-3.
        """
        assert exp742.classify_verdict(0.005) == "retro033_confirmed_closed"

    def test_confirmed_closed_when_large_positive(self) -> None:
        """Large positive signed_improvement → 'retro033_confirmed_closed'.

        Spec: REQ-VERIFY-150-3.
        """
        assert exp742.classify_verdict(0.10) == "retro033_confirmed_closed"

    def test_confirmed_closed_boundary(self) -> None:
        """signed_improvement just above 0.003 → 'retro033_confirmed_closed'.

        The boundary itself (0.003) belongs to marginal_inconclusive;
        anything strictly above is confirmed_closed.
        Spec: REQ-VERIFY-150-3.
        """
        assert exp742.classify_verdict(0.0031) == "retro033_confirmed_closed"

    def test_marginal_inconclusive_at_boundary(self) -> None:
        """Exactly 0.003 → 'retro033_marginal_inconclusive' (boundary inclusive).

        Spec: REQ-VERIFY-150-3.
        """
        assert exp742.classify_verdict(0.003) == "retro033_marginal_inconclusive"

    def test_marginal_inconclusive_just_above_zero(self) -> None:
        """0 < signed_improvement <= 0.003 → 'retro033_marginal_inconclusive'.

        Positive but inside the 200q noise floor — needs more data.
        Spec: REQ-VERIFY-150-3.
        """
        assert exp742.classify_verdict(0.001) == "retro033_marginal_inconclusive"

    def test_reopened_when_zero(self) -> None:
        """signed_improvement == 0 → 'retro033_reopened'.

        VR produced no improvement; Exp 720 was a statistical fluke.
        Spec: REQ-VERIFY-150-3.
        """
        assert exp742.classify_verdict(0.0) == "retro033_reopened"

    def test_reopened_when_negative(self) -> None:
        """signed_improvement < 0 → 'retro033_reopened'.

        VR actively hurt accuracy; clearly not viable.
        Spec: REQ-VERIFY-150-3.
        """
        assert exp742.classify_verdict(-0.05) == "retro033_reopened"

    def test_reopened_when_strongly_negative(self) -> None:
        """Large negative → 'retro033_reopened'.

        Matches the worst-case pattern seen in earlier 100q experiments.
        Spec: REQ-VERIFY-150-3.
        """
        assert exp742.classify_verdict(-0.8) == "retro033_reopened"


# ---------------------------------------------------------------------------
# REQ-VERIFY-150: compute_signed_improvement correctness
# ---------------------------------------------------------------------------


class TestComputeSignedImprovement:
    """Verify compute_signed_improvement produces the correct arithmetic.

    WHY: This is the primary outcome metric.  A bug here silently corrupts the
    verdict and RETRO-033 status.  We test with known values so failures are
    immediately diagnosable.
    Spec: REQ-VERIFY-150.
    """

    def test_positive_when_vr_helps(self) -> None:
        """When VR gets 7/10 vs baseline 5/10, signed_improvement is +0.20.

        Spec: REQ-VERIFY-150.
        """
        baseline = [True] * 5 + [False] * 5
        vr = [True] * 7 + [False] * 3
        si = exp742.compute_signed_improvement(baseline, vr)
        assert abs(si - 0.20) < 1e-9, f"Expected 0.20, got {si}"

    def test_negative_when_vr_hurts(self) -> None:
        """When VR gets 3/10 vs baseline 8/10, signed_improvement is -0.50.

        Spec: REQ-VERIFY-150.
        """
        baseline = [True] * 8 + [False] * 2
        vr = [True] * 3 + [False] * 7
        si = exp742.compute_signed_improvement(baseline, vr)
        assert abs(si - (-0.50)) < 1e-9, f"Expected -0.50, got {si}"

    def test_zero_when_identical(self) -> None:
        """When baseline and VR have same correct count, signed_improvement is 0.

        Spec: REQ-VERIFY-150.
        """
        baseline = [True] * 5 + [False] * 5
        vr = [True] * 5 + [False] * 5
        si = exp742.compute_signed_improvement(baseline, vr)
        assert si == 0.0

    def test_empty_lists_return_zero(self) -> None:
        """Empty lists (no questions run) → 0.0, not a division error.

        Spec: REQ-VERIFY-150.
        """
        si = exp742.compute_signed_improvement([], [])
        assert si == 0.0

    def test_large_scale_200q(self) -> None:
        """At 200q scale, one question difference = ±0.005 signed_improvement.

        This verifies the noise floor calculation used for RETRO-033 closure.
        Spec: REQ-VERIFY-150.
        """
        baseline = [True] * 100 + [False] * 100  # 100/200 = 0.500
        vr = [True] * 101 + [False] * 99          # 101/200 = 0.505
        si = exp742.compute_signed_improvement(baseline, vr)
        assert abs(si - 0.005) < 1e-9, f"Expected 0.005, got {si}"


# ---------------------------------------------------------------------------
# REQ-VERIFY-150-1/2: seed=999 produces different order from seed=218
# ---------------------------------------------------------------------------


class TestShuffleQuestions:
    """Verify shuffle_questions uses seed correctly and differs between seeds.

    WHY: Statistical independence between Exp 720 (seed=218) and Exp 742
    (seed=999) is the whole point of the confirmation trial.  If both seeds
    produce the same order, the two trials are not independent.
    Spec: REQ-VERIFY-150-1, REQ-VERIFY-150-2.
    """

    def test_seed_999_produces_different_order_from_seed_218(self) -> None:
        """Shuffling with seed=999 vs seed=218 must produce different question orders.

        This is the core independence requirement: the two trials must be
        statistically independent draws from the question pool.
        Spec: REQ-VERIFY-150-2.
        """
        questions = exp742._QUESTIONS
        order_218 = exp742.shuffle_questions(questions, seed=218)
        order_999 = exp742.shuffle_questions(questions, seed=999)
        first_218 = [q["question"] for q in order_218[:5]]
        first_999 = [q["question"] for q in order_999[:5]]
        assert first_218 != first_999, (
            "seed=218 and seed=999 must produce different question orderings — "
            "identical orderings would mean the trials are NOT independent (REQ-VERIFY-150-2)"
        )

    def test_seed_999_is_reproducible(self) -> None:
        """Same seed=999 always produces the same order (reproducibility guarantee).

        Spec: REQ-VERIFY-150-1.
        """
        questions = exp742._QUESTIONS
        order_a = exp742.shuffle_questions(questions, seed=999)
        order_b = exp742.shuffle_questions(questions, seed=999)
        assert [q["question"] for q in order_a] == [q["question"] for q in order_b], (
            "shuffle_questions must be deterministic for the same seed — "
            "non-determinism would make the trial non-reproducible (REQ-VERIFY-150-1)"
        )

    def test_shuffle_preserves_all_questions(self) -> None:
        """After shuffling, all questions are still present (none dropped or duplicated).

        Spec: REQ-VERIFY-150-1.
        """
        questions = exp742._QUESTIONS
        shuffled = exp742.shuffle_questions(questions, seed=999)
        assert len(shuffled) == len(questions), (
            "shuffle must not drop or duplicate questions"
        )
        original_qs = {q["question"] for q in questions}
        shuffled_qs = {q["question"] for q in shuffled}
        assert original_qs == shuffled_qs, "shuffle must preserve all question texts"

    def test_original_list_not_mutated(self) -> None:
        """shuffle_questions must not modify the input list in place.

        Mutating the input would corrupt _QUESTIONS for subsequent runs.
        Spec: REQ-VERIFY-150-1.
        """
        questions = exp742._QUESTIONS
        first_before = questions[0]["question"]
        exp742.shuffle_questions(questions, seed=999)
        assert questions[0]["question"] == first_before, (
            "shuffle_questions must not mutate the input list (REQ-VERIFY-150-1)"
        )


# ---------------------------------------------------------------------------
# Blocked artifact schema (REQ-VERIFY-150-4)
# ---------------------------------------------------------------------------


class TestBlockedArtifactSchema:
    """Verify the blocked artifact has all required fields including seed_218_signed_improvement.

    WHY: REQ-VERIFY-150-4 requires the artifact to record seed_218_signed_improvement
    for direct comparison.  Missing this field breaks the conductor's comparison logic.
    Spec: REQ-VERIFY-150-4.
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
        "seed",
        "n_questions",
        "n_questions_attempted",
        "baseline_accuracy",
        "verify_repair_accuracy",
        "signed_improvement",
        "seed_218_signed_improvement",
        "honest_verdict",
        "retro033_status",
        "inference_mode",
        "models_used",
        "batch_log",
    }

    def test_blocked_has_all_required_fields(self, tmp_path: Path) -> None:
        """Blocked artifact must contain every required schema field.

        Spec: REQ-VERIFY-150-4.
        """
        artifact = _run_blocked_main(tmp_path)
        for field in self._REQUIRED_FIELDS:
            assert field in artifact, (
                f"Field '{field}' missing from blocked artifact — "
                f"REQ-VERIFY-150-4 requires seed_218_signed_improvement for comparison"
            )

    def test_blocked_has_correct_seed_218_value(self, tmp_path: Path) -> None:
        """seed_218_signed_improvement must equal Exp 720's result (0.00510).

        Spec: REQ-VERIFY-150-4.
        """
        artifact = _run_blocked_main(tmp_path)
        si_218 = artifact.get("seed_218_signed_improvement")
        assert si_218 is not None, "seed_218_signed_improvement must be present"
        assert abs(si_218 - 0.00510) < 1e-4, (
            f"seed_218_signed_improvement must equal Exp 720 result 0.00510, got {si_218}"
        )

    def test_blocked_seed_is_999(self, tmp_path: Path) -> None:
        """Blocked artifact must record seed=999 (REQ-VERIFY-150-1).

        Spec: REQ-VERIFY-150-1.
        """
        artifact = _run_blocked_main(tmp_path)
        assert artifact.get("seed") == 999, (
            f"seed must be 999 in the artifact, got {artifact.get('seed')}"
        )

    def test_blocked_experiment_id_is_742(self, tmp_path: Path) -> None:
        """Blocked artifact must have experiment=742.

        Spec: REQ-VERIFY-150.
        """
        artifact = _run_blocked_main(tmp_path)
        assert artifact["experiment"] == 742

    def test_blocked_status_is_blocked(self, tmp_path: Path) -> None:
        """Blocked artifact status must be 'blocked'.

        Spec: REQ-VERIFY-150.
        """
        artifact = _run_blocked_main(tmp_path)
        assert artifact["status"] == "blocked"

    def test_blocked_honest_verdict_is_blocked_no_gpu(self, tmp_path: Path) -> None:
        """Blocked honest_verdict must indicate GPU unavailability.

        Spec: REQ-VERIFY-150-3.
        """
        artifact = _run_blocked_main(tmp_path)
        assert artifact.get("honest_verdict") == "blocked_no_gpu"


# ---------------------------------------------------------------------------
# On-disk deliverable validation
# ---------------------------------------------------------------------------


class TestOnDiskDeliverable:
    """Validate the on-disk deliverable (if present) has correct schema.

    WHY: The conductor reads this file after the experiment runs.  Missing fields
    cause KeyError in the retrospective agent.  Running this test suite after
    main() completes validates the actual experiment output.
    Spec: REQ-VERIFY-150.
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
        "seed",
        "n_questions",
        "baseline_accuracy",
        "verify_repair_accuracy",
        "signed_improvement",
        "seed_218_signed_improvement",
        "honest_verdict",
        "retro033_status",
        "inference_mode",
        "models_used",
        "batch_log",
    }

    def test_on_disk_deliverable_has_all_required_fields(self) -> None:
        """On-disk deliverable must have all required fields.

        Spec: REQ-VERIFY-150-4, SCENARIO-VERIFY-200.
        """
        if not _DELIVERABLE.exists():
            pytest.skip("Deliverable not yet written — run experiment_742 first")

        artifact = json.loads(_DELIVERABLE.read_text())
        for field in self._REQUIRED_FIELDS:
            assert field in artifact, (
                f"Field '{field}' missing from on-disk deliverable"
            )

    def test_on_disk_deliverable_seed_is_999(self) -> None:
        """On-disk deliverable must record seed=999.

        Spec: REQ-VERIFY-150-1.
        """
        if not _DELIVERABLE.exists():
            pytest.skip("Deliverable not yet written")

        artifact = json.loads(_DELIVERABLE.read_text())
        assert artifact.get("seed") == 999, (
            f"seed must be 999 in the deliverable, got {artifact.get('seed')}"
        )

    def test_on_disk_deliverable_honest_verdict_valid(self) -> None:
        """honest_verdict must be one of the three valid RETRO-033 verdicts.

        Spec: REQ-VERIFY-150-3.
        """
        if not _DELIVERABLE.exists():
            pytest.skip("Deliverable not yet written")

        artifact = json.loads(_DELIVERABLE.read_text())
        if artifact.get("inference_mode") == "blocked_no_gpu":
            pytest.skip("Blocked run — honest_verdict is 'blocked_no_gpu', not a RETRO-033 verdict")

        valid_verdicts = {"retro033_confirmed_closed", "retro033_marginal_inconclusive", "retro033_reopened"}
        assert artifact.get("honest_verdict") in valid_verdicts, (
            f"honest_verdict '{artifact.get('honest_verdict')}' is not a valid RETRO-033 verdict — "
            f"must be one of {valid_verdicts} (REQ-VERIFY-150-3)"
        )

    def test_on_disk_deliverable_seed_218_value_preserved(self) -> None:
        """seed_218_signed_improvement must match Exp 720 result (0.00510 ± 0.0001).

        Spec: REQ-VERIFY-150-4.
        """
        if not _DELIVERABLE.exists():
            pytest.skip("Deliverable not yet written")

        artifact = json.loads(_DELIVERABLE.read_text())
        si_218 = artifact.get("seed_218_signed_improvement")
        assert si_218 is not None, "seed_218_signed_improvement must be present"
        assert abs(si_218 - 0.00510) < 1e-4, (
            f"seed_218_signed_improvement must match Exp 720 (0.00510), got {si_218}"
        )

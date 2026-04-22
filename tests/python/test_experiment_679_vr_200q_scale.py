"""Tests for scripts/experiment_679_vr_200q_scale.py — VR 200q scale with Wilson CI.

Covers:
- Wilson CI computation (manual formula correctness)
- honest_verdict logic for all four outcome cases
- blocked artifact when CARNOT_FORCE_LIVE is not set
- LongRunBenchmarkExecutor checkpoint path naming (SCENARIO-VERIFY-207)

Spec: REQ-VERIFY-155, REQ-VERIFY-156,
      SCENARIO-VERIFY-205, SCENARIO-VERIFY-206, SCENARIO-VERIFY-207
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.experiment_679_vr_200q_scale as mod
from scripts.experiment_679_vr_200q_scale import (
    DELIVERABLE,
    EXP_ID,
    N_QUESTIONS,
    SCHEMA,
    compute_honest_verdict_679,
    compute_wilson_ci,
)


# ---------------------------------------------------------------------------
# compute_wilson_ci — SCENARIO-VERIFY-205
# ---------------------------------------------------------------------------


def test_wilson_ci_known_proportion() -> None:
    """Wilson CI for p=0.80, n=100 has lower in [0.71, 0.73] and upper in [0.86, 0.88].

    Reference: standard Wilson 95% CI for 80/100 is approximately (0.711, 0.867).

    Spec: REQ-VERIFY-155, SCENARIO-VERIFY-205
    """
    lower, upper = compute_wilson_ci(80, 100)
    assert 0.71 <= lower <= 0.73, f"Expected lower in [0.71, 0.73], got {lower:.4f}"
    assert 0.86 <= upper <= 0.88, f"Expected upper in [0.86, 0.88], got {upper:.4f}"


def test_wilson_ci_returns_tuple_of_two_floats() -> None:
    """compute_wilson_ci returns a (lower, upper) tuple of floats in [0, 1].

    Spec: REQ-VERIFY-155
    """
    lower, upper = compute_wilson_ci(50, 100)
    assert isinstance(lower, float)
    assert isinstance(upper, float)
    assert 0.0 <= lower <= 1.0
    assert 0.0 <= upper <= 1.0
    assert lower <= upper


def test_wilson_ci_zero_questions_safe() -> None:
    """compute_wilson_ci handles n_total=0 without raising — returns (0.0, 1.0).

    WHY test zero: LongRunBenchmarkExecutor may produce n_answered=0 if all batches
    time out.  The CI must still be defined rather than crashing with ZeroDivisionError.

    Spec: REQ-VERIFY-155
    """
    lower, upper = compute_wilson_ci(0, 0)
    assert lower == 0.0
    assert upper == 1.0


def test_wilson_ci_all_correct() -> None:
    """When all questions correct (p=1.0), upper bound must be 1.0 and lower > 0.9.

    Spec: REQ-VERIFY-155, SCENARIO-VERIFY-205
    """
    lower, upper = compute_wilson_ci(100, 100)
    assert upper == 1.0
    assert lower > 0.9


def test_wilson_ci_all_wrong() -> None:
    """When no questions correct (p=0.0), lower bound must be 0.0 and upper < 0.1.

    Spec: REQ-VERIFY-155, SCENARIO-VERIFY-205
    """
    lower, upper = compute_wilson_ci(0, 100)
    assert lower == 0.0
    assert upper < 0.1


def test_wilson_ci_wider_for_small_n() -> None:
    """Wilson CI must be wider for n=10 than n=100 at the same proportion p=0.5.

    WHY test width vs n: the interval shrinks as n grows — this is the key property
    that motivates scaling from 25 questions (Exp 668) to 200 questions (Exp 679).

    Spec: REQ-VERIFY-155
    """
    lo_small, hi_small = compute_wilson_ci(5, 10)
    lo_large, hi_large = compute_wilson_ci(50, 100)
    width_small = hi_small - lo_small
    width_large = hi_large - lo_large
    assert width_small > width_large, (
        f"Expected wider CI for n=10 ({width_small:.3f}) than n=100 ({width_large:.3f})"
    )


def test_wilson_ci_manual_formula_matches_known_result() -> None:
    """Cross-check the Wilson formula against a known reference value.

    Reference: for n=25, k=9 (Exp 668 baseline), the Wilson 95% CI lower bound
    should be approximately 0.21 (standard stats table).

    Spec: REQ-VERIFY-155, SCENARIO-VERIFY-205
    """
    lower, upper = compute_wilson_ci(9, 25)
    # Known approximate values from standard Wilson CI tables
    assert 0.18 <= lower <= 0.26, f"Expected lower ~0.21, got {lower:.4f}"
    assert 0.55 <= upper <= 0.65, f"Expected upper ~0.60, got {upper:.4f}"


# ---------------------------------------------------------------------------
# compute_honest_verdict_679 — SCENARIO-VERIFY-206
# ---------------------------------------------------------------------------


def test_verdict_positive_strong() -> None:
    """signed_improvement > 0.05 AND ci_lower > 0 → 'vr_200q_positive'.

    Spec: REQ-VERIFY-155, SCENARIO-VERIFY-206
    """
    assert compute_honest_verdict_679(0.10, 0.02, "live_gpu") == "vr_200q_positive"


def test_verdict_positive_boundary_exactly_0_05() -> None:
    """signed_improvement exactly 0.05 is NOT positive (threshold is strict > 0.05).

    Spec: REQ-VERIFY-155
    """
    result = compute_honest_verdict_679(0.05, 0.01, "live_gpu")
    assert result == "vr_200q_marginal"


def test_verdict_marginal_small_improvement() -> None:
    """0 < signed_improvement <= 0.05 → 'vr_200q_marginal'.

    Spec: REQ-VERIFY-155, SCENARIO-VERIFY-206
    """
    assert compute_honest_verdict_679(0.02, -0.01, "live_gpu") == "vr_200q_marginal"


def test_verdict_marginal_ci_crosses_zero() -> None:
    """Even with ci_lower < 0, a small positive improvement is marginal (not blocked).

    WHY this matters: wilson_ci_lower < 0 means the CI crosses zero — we have improvement
    but it's not statistically significant.  We report 'marginal', not 'no_improvement'.

    Spec: REQ-VERIFY-155
    """
    assert compute_honest_verdict_679(0.04, -0.005, "live_gpu") == "vr_200q_marginal"


def test_verdict_no_improvement_zero() -> None:
    """signed_improvement = 0.0 → 'vr_200q_no_improvement'.

    Spec: REQ-VERIFY-155, SCENARIO-VERIFY-206
    """
    assert compute_honest_verdict_679(0.0, 0.0, "live_gpu") == "vr_200q_no_improvement"


def test_verdict_no_improvement_negative() -> None:
    """Negative signed_improvement → 'vr_200q_no_improvement'.

    Spec: REQ-VERIFY-155, SCENARIO-VERIFY-206
    """
    assert compute_honest_verdict_679(-0.05, -0.08, "live_gpu") == "vr_200q_no_improvement"


def test_verdict_blocked_inference_mode() -> None:
    """inference_mode='blocked' always yields 'vr_200q_blocked'.

    Spec: REQ-VERIFY-156, SCENARIO-VERIFY-206
    """
    assert compute_honest_verdict_679(0.5, 0.3, "blocked") == "vr_200q_blocked"
    assert compute_honest_verdict_679(0.0, 0.0, "blocked") == "vr_200q_blocked"
    assert compute_honest_verdict_679(-0.1, -0.2, "blocked") == "vr_200q_blocked"


# ---------------------------------------------------------------------------
# Blocked artifact when no live GPU — REQ-VERIFY-156
# ---------------------------------------------------------------------------


def test_blocked_artifact_when_no_carnot_force_live(tmp_path: Path) -> None:
    """When CARNOT_FORCE_LIVE is not set, _run_inner writes a blocked artifact and exits 0.

    Spec: REQ-VERIFY-156
    """
    deliverable = tmp_path / "experiment_679_vr_200q_scale.json"

    written: list[dict] = []

    class _FakeWriter:
        def write(self, data: dict) -> None:
            written.append(data)

    class _FakeTemplate:
        def setup(self) -> None:
            pass

        def assert_deliverable_written(self) -> None:
            pass

        def build_result(self, *a, **kw):
            return {}

        def setup_gpu(self, *a, **kw):
            return {"all_healthy": True, "models": []}

    _fake_tmpl = _FakeTemplate()

    with (
        patch("scripts.experiment_template.ExperimentTemplate", return_value=_fake_tmpl),
        patch("carnot.pipeline.atomic_writer.AtomicResultWriter", return_value=_FakeWriter()),
        patch.dict(os.environ, {}, clear=False),
    ):
        # Ensure CARNOT_FORCE_LIVE is absent
        env = dict(os.environ)
        env.pop("CARNOT_FORCE_LIVE", None)
        with patch.dict(os.environ, env, clear=True):
            mock_watchdog = MagicMock()
            with pytest.raises(SystemExit) as exc_info:
                mod._run_inner(mock_watchdog)

    assert exc_info.value.code == 0
    assert len(written) == 1
    result = written[0]
    assert result["honest_verdict"] == "vr_200q_blocked"
    assert result["inference_mode"] == "blocked"
    assert result["experiment"] == EXP_ID
    assert result["retro_033_validated"] is False


def test_blocked_artifact_has_all_required_fields(tmp_path: Path) -> None:
    """Blocked artifact must include every required schema field.

    Spec: REQ-VERIFY-156
    """
    written: list[dict] = []

    class _FakeWriter:
        def write(self, data: dict) -> None:
            written.append(data)

    class _FakeTemplate:
        def setup(self) -> None:
            pass

        def assert_deliverable_written(self) -> None:
            pass

        def build_result(self, *a, **kw):
            return {}

        def setup_gpu(self, *a, **kw):
            return {"all_healthy": True, "models": []}

    with (
        patch("scripts.experiment_template.ExperimentTemplate", return_value=_FakeTemplate()),
        patch("carnot.pipeline.atomic_writer.AtomicResultWriter", return_value=_FakeWriter()),
    ):
        env = dict(os.environ)
        env.pop("CARNOT_FORCE_LIVE", None)
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(SystemExit):
                mod._run_inner(MagicMock())

    required_fields = {
        "experiment", "schema", "run_date", "status", "honest_verdict",
        "inference_mode", "baseline_accuracy", "post_accuracy",
        "signed_improvement", "wilson_ci_lower", "wilson_ci_upper",
        "n_questions", "forcing_recall", "retro_033_validated",
    }
    missing = required_fields - set(written[0].keys())
    assert not missing, f"Missing schema fields in blocked artifact: {missing}"


# ---------------------------------------------------------------------------
# LongRunBenchmarkExecutor checkpoint path — SCENARIO-VERIFY-207
# ---------------------------------------------------------------------------


def test_checkpoint_path_uses_exp679_prefix(tmp_path: Path) -> None:
    """LongRunBenchmarkExecutor.save_batch with prefix='exp679' names the file correctly.

    Expected filename: {checkpoint_dir}/exp679_batch_0000.json

    Spec: REQ-VERIFY-155, SCENARIO-VERIFY-207
    """
    from carnot.pipeline.long_run_executor import BenchmarkBatch, LongRunBenchmarkExecutor

    executor = LongRunBenchmarkExecutor(batch_size=25, checkpoint_dir=str(tmp_path))
    batch = BenchmarkBatch(
        batch_id=0,
        start_idx=0,
        end_idx=25,
        questions=["q1"] * 25,
        results=[{"baseline_correct": True, "post_correct": True, "compute_lines_found": 1}] * 25,
        status="complete",
    )
    saved_path = executor.save_batch(batch, prefix="exp679")

    expected_name = "exp679_batch_0000.json"
    assert Path(saved_path).name == expected_name, (
        f"Expected checkpoint filename '{expected_name}', got '{Path(saved_path).name}'"
    )
    assert Path(saved_path).exists()


def test_checkpoint_contents_are_valid_json(tmp_path: Path) -> None:
    """Checkpoint file written by save_batch is parseable JSON.

    Spec: REQ-VERIFY-155, SCENARIO-VERIFY-207
    """
    from carnot.pipeline.long_run_executor import BenchmarkBatch, LongRunBenchmarkExecutor

    executor = LongRunBenchmarkExecutor(batch_size=25, checkpoint_dir=str(tmp_path))
    batch = BenchmarkBatch(
        batch_id=2,
        start_idx=50,
        end_idx=75,
        questions=["q"] * 25,
        results=[{"baseline_correct": False, "post_correct": True, "compute_lines_found": 2}] * 25,
        status="complete",
    )
    path = executor.save_batch(batch, prefix="exp679")
    data = json.loads(Path(path).read_text())
    assert data["batch_id"] == 2
    assert data["status"] == "complete"
    assert len(data["results"]) == 25


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


def test_exp_id() -> None:
    """EXP_ID must be 679.  Spec: REQ-VERIFY-155"""
    assert EXP_ID == 679


def test_n_questions() -> None:
    """N_QUESTIONS must be 200.  Spec: REQ-VERIFY-155"""
    assert N_QUESTIONS == 200


def test_schema() -> None:
    """SCHEMA must encode the experiment version.  Spec: REQ-VERIFY-155"""
    assert SCHEMA == "carnot.vr_200q_scale.v1"


# ---------------------------------------------------------------------------
# Deliverable JSON on disk — validates the actual run artifact
# ---------------------------------------------------------------------------


def test_deliverable_json_exists_and_valid() -> None:
    """Deliverable JSON on disk must exist and contain all required schema fields.

    Spec: REQ-VERIFY-155, REQ-VERIFY-156
    """
    result_path = _REPO_ROOT / DELIVERABLE
    if not result_path.exists():
        pytest.skip("Deliverable not yet written — run the experiment first")

    data = json.loads(result_path.read_text())

    required = {
        "experiment", "schema", "run_date", "status", "honest_verdict",
        "inference_mode", "baseline_accuracy", "post_accuracy",
        "signed_improvement", "wilson_ci_lower", "wilson_ci_upper",
        "n_questions", "forcing_recall", "retro_033_validated",
    }
    missing = required - set(data.keys())
    assert not missing, f"Missing fields in deliverable: {missing}"

    assert data["experiment"] == EXP_ID
    valid_verdicts = {
        "vr_200q_positive",
        "vr_200q_marginal",
        "vr_200q_no_improvement",
        "vr_200q_blocked",
    }
    assert data["honest_verdict"] in valid_verdicts, (
        f"Unknown honest_verdict: {data['honest_verdict']}"
    )
    assert isinstance(data["retro_033_validated"], bool)
    assert 0.0 <= data["wilson_ci_lower"] <= 1.0
    assert 0.0 <= data["wilson_ci_upper"] <= 1.0
    assert data["wilson_ci_lower"] <= data["wilson_ci_upper"]

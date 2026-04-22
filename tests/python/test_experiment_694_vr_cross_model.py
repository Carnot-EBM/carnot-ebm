"""Tests for experiment 694 VR cross-model — Gemma-4-E4B-it grammar-constrained COMPUTE: forcing.

Covers:
- GrammarConstrainedDecoder.grammar_recall computation (SCENARIO-VERIFY-215)
- cross_model_delta logic (SCENARIO-VERIFY-214)
- select_hard_questions filter / proxy set (SCENARIO-VERIFY-216)
- compute_honest_verdict_694 for all verdict paths
- Blocked artifact when CARNOT_FORCE_LIVE is not set
- Deliverable JSON on disk

Spec: REQ-VERIFY-162, REQ-VERIFY-163, REQ-VERIFY-164,
      SCENARIO-VERIFY-214, SCENARIO-VERIFY-215, SCENARIO-VERIFY-216
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.grammar_constrained_decoder import GrammarConstrainedDecoder
from scripts.experiment_694_vr_cross_model import (
    DELIVERABLE,
    EXP_ID,
    HARD_BASELINE_THRESHOLD,
    HARD_PROXY_END,
    HARD_PROXY_START,
    N_HARD_QUESTIONS,
    SCHEMA,
    compute_cross_model_delta,
    compute_honest_verdict_694,
    select_hard_questions,
)
import scripts.experiment_694_vr_cross_model as mod


# ---------------------------------------------------------------------------
# GrammarConstrainedDecoder.grammar_recall — SCENARIO-VERIFY-215
# ---------------------------------------------------------------------------


def test_grammar_recall_two_of_three() -> None:
    """grammar_recall returns 2/3 when exactly 2 of 3 outputs contain COMPUTE:.

    Spec: REQ-VERIFY-164-3, SCENARIO-VERIFY-215
    """
    decoder = GrammarConstrainedDecoder(None, None, ["COMPUTE:"])
    outputs = [
        "step1 COMPUTE: 2+2=4 done",
        "no compute here",
        "COMPUTE: 5*3=15",
    ]
    recall = decoder.grammar_recall(outputs)
    assert abs(recall - 2 / 3) < 1e-9, f"Expected 0.6667, got {recall}"


def test_grammar_recall_all_pass() -> None:
    """grammar_recall returns 1.0 when all outputs contain COMPUTE:.

    Spec: REQ-VERIFY-164-3
    """
    decoder = GrammarConstrainedDecoder(None, None, ["COMPUTE:"])
    outputs = ["COMPUTE: 1+1=2", "step COMPUTE: 3*4=12 end", "COMPUTE: 5-2=3"]
    assert decoder.grammar_recall(outputs) == 1.0


def test_grammar_recall_none_pass() -> None:
    """grammar_recall returns 0.0 when no outputs contain COMPUTE:.

    Spec: REQ-VERIFY-164-3
    """
    decoder = GrammarConstrainedDecoder(None, None, ["COMPUTE:"])
    outputs = ["the answer is 7", "no structured format here"]
    assert decoder.grammar_recall(outputs) == 0.0


def test_grammar_recall_empty_list() -> None:
    """grammar_recall returns 0.0 for an empty output list (avoid ZeroDivisionError).

    Spec: REQ-VERIFY-164-3
    """
    decoder = GrammarConstrainedDecoder(None, None, ["COMPUTE:"])
    assert decoder.grammar_recall([]) == 0.0


def test_grammar_recall_case_sensitive() -> None:
    """grammar_recall is case-sensitive: 'compute:' does not match 'COMPUTE:'.

    WHY test case sensitivity: SymCodeVerifier also uses case-sensitive matching.
    Mixed-case variants indicate the model invented its own format.

    Spec: REQ-VERIFY-164
    """
    decoder = GrammarConstrainedDecoder(None, None, ["COMPUTE:"])
    outputs = ["compute: 2+2=4", "Compute: 3+3=6"]
    assert decoder.grammar_recall(outputs) == 0.0


def test_grammar_recall_ci_mode_decode_contains_required_token() -> None:
    """CI-mode decode() returns a string containing the required token.

    WHY test CI mode: ensures the synthetic fallback path exercises grammar_recall > 0
    even without a live model.

    Spec: REQ-VERIFY-164-2
    """
    decoder = GrammarConstrainedDecoder(None, None, ["COMPUTE:"])
    output = decoder.decode("What is 2+2?")
    assert "COMPUTE:" in output, f"Expected COMPUTE: in CI output, got: {output!r}"
    assert decoder.grammar_recall([output]) == 1.0


# ---------------------------------------------------------------------------
# compute_cross_model_delta — SCENARIO-VERIFY-214
# ---------------------------------------------------------------------------


def test_cross_model_delta_negative() -> None:
    """Gemma improves less than Qwen → cross_model_delta is negative.

    Spec: REQ-VERIFY-162-2, SCENARIO-VERIFY-214
    """
    delta = compute_cross_model_delta(0.08, 0.10)
    assert abs(delta - (-0.02)) < 1e-9, f"Expected -0.02, got {delta}"


def test_cross_model_delta_positive() -> None:
    """Gemma improves more than Qwen → cross_model_delta is positive.

    Spec: REQ-VERIFY-162-2, SCENARIO-VERIFY-214
    """
    delta = compute_cross_model_delta(0.15, 0.10)
    assert abs(delta - 0.05) < 1e-9, f"Expected 0.05, got {delta}"


def test_cross_model_delta_zero() -> None:
    """Equal improvements → cross_model_delta is 0.0.

    Spec: REQ-VERIFY-162-2
    """
    delta = compute_cross_model_delta(0.10, 0.10)
    assert delta == 0.0


# ---------------------------------------------------------------------------
# select_hard_questions — SCENARIO-VERIFY-216
# ---------------------------------------------------------------------------


def test_select_hard_questions_proxy_set() -> None:
    """select_hard_questions returns indices 600-649 when dataset is large enough.

    Spec: REQ-VERIFY-163-2, SCENARIO-VERIFY-216
    """
    all_q = [f"q{i}" for i in range(700)]
    result = select_hard_questions(all_q, n=50)
    assert len(result) == 50
    assert result[0] == "q600"
    assert result[49] == "q649"


def test_select_hard_questions_returns_exactly_n() -> None:
    """select_hard_questions always returns exactly n questions.

    Spec: REQ-VERIFY-163-3
    """
    all_q = [f"q{i}" for i in range(700)]
    result = select_hard_questions(all_q, n=50)
    assert len(result) == N_HARD_QUESTIONS


def test_select_hard_questions_no_overlap_with_exp679() -> None:
    """Proxy set indices 600-649 do not overlap with Exp 679 indices 0-199.

    Spec: REQ-VERIFY-163-2
    """
    assert HARD_PROXY_START >= 200, (
        f"Proxy start {HARD_PROXY_START} overlaps with Exp 679 range 0-199"
    )
    assert HARD_PROXY_END <= 650, (
        f"Proxy end {HARD_PROXY_END} extends beyond expected 650"
    )


def test_select_hard_questions_fallback_small_dataset() -> None:
    """select_hard_questions falls back gracefully when dataset is smaller than proxy range.

    WHY test small dataset: HuggingFace download may fail or return fewer questions
    than HARD_PROXY_END=650 in some environments.

    Spec: REQ-VERIFY-163-2
    """
    all_q = [f"q{i}" for i in range(100)]  # only 100 questions
    result = select_hard_questions(all_q, n=50)
    assert len(result) == 50


def test_hard_baseline_threshold_value() -> None:
    """HARD_BASELINE_THRESHOLD must be 0.40 per REQ-VERIFY-163-1.

    Spec: REQ-VERIFY-163-1
    """
    assert HARD_BASELINE_THRESHOLD == 0.4


# ---------------------------------------------------------------------------
# compute_honest_verdict_694 — all verdict paths
# ---------------------------------------------------------------------------


def test_verdict_confirmed_high_recall() -> None:
    """gemma_improvement > 0 AND grammar_recall > 0.9 → 'vr_cross_model_confirmed'.

    Spec: REQ-VERIFY-162-3
    """
    v = compute_honest_verdict_694(1.0, 0.10, 0.95, "live_gpu")
    assert v == "vr_cross_model_confirmed"


def test_verdict_partial_low_recall() -> None:
    """gemma_improvement > 0 AND grammar_recall <= 0.9 → 'vr_cross_model_partial'.

    Spec: REQ-VERIFY-162-4
    """
    v = compute_honest_verdict_694(1.0, 0.10, 0.80, "live_gpu")
    assert v == "vr_cross_model_partial"


def test_verdict_no_improvement() -> None:
    """gemma_improvement <= 0 → 'vr_cross_model_no_improvement'.

    Spec: REQ-VERIFY-162-5
    """
    v = compute_honest_verdict_694(1.0, 0.0, 0.95, "live_gpu")
    assert v == "vr_cross_model_no_improvement"


def test_verdict_no_improvement_negative() -> None:
    """Negative gemma_improvement → 'vr_cross_model_no_improvement'.

    Spec: REQ-VERIFY-162-5
    """
    v = compute_honest_verdict_694(1.0, -0.05, 0.95, "live_gpu")
    assert v == "vr_cross_model_no_improvement"


def test_verdict_analysis_only() -> None:
    """inference_mode='analysis_only' → 'cross_model_analysis_only'.

    Spec: REQ-VERIFY-162
    """
    v = compute_honest_verdict_694(0.0, 0.0, 0.0, "analysis_only")
    assert v == "cross_model_analysis_only"


def test_verdict_blocked_no_gpu() -> None:
    """inference_mode='blocked' → 'cross_model_blocked_no_gpu'.

    Spec: REQ-VERIFY-162
    """
    v = compute_honest_verdict_694(1.0, 0.0, 0.0, "blocked")
    assert v == "cross_model_blocked_no_gpu"


def test_verdict_boundary_grammar_recall_exactly_09() -> None:
    """grammar_recall exactly 0.9 is NOT > 0.9 → 'vr_cross_model_partial', not confirmed.

    WHY boundary test: the confirmed threshold is strict (> 0.9), not >=.

    Spec: REQ-VERIFY-162-3, REQ-VERIFY-162-4
    """
    v = compute_honest_verdict_694(1.0, 0.10, 0.9, "live_gpu")
    assert v == "vr_cross_model_partial"


# ---------------------------------------------------------------------------
# Blocked artifact when no live GPU — REQ-VERIFY-162
# ---------------------------------------------------------------------------


def test_blocked_artifact_when_no_carnot_force_live() -> None:
    """When CARNOT_FORCE_LIVE is not set and Exp 679 gate passes, writes blocked artifact.

    Spec: REQ-VERIFY-162
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

    fake_exp679 = {
        "experiment": 679,
        "signed_improvement": 1.0,
    }

    _fake_tmpl = _FakeTemplate()

    with (
        patch("scripts.experiment_template.ExperimentTemplate", return_value=_fake_tmpl),
        patch("carnot.pipeline.atomic_writer.AtomicResultWriter", return_value=_FakeWriter()),
        patch.object(
            Path,
            "read_text",
            return_value=json.dumps(fake_exp679),
        ),
        patch.object(Path, "exists", return_value=True),
    ):
        env = dict(os.environ)
        env.pop("CARNOT_FORCE_LIVE", None)
        with patch.dict(os.environ, env, clear=True):
            mock_watchdog = MagicMock()
            with pytest.raises(SystemExit) as exc_info:
                mod._run_inner(mock_watchdog)

    assert exc_info.value.code == 0
    assert len(written) == 1
    result = written[0]
    assert result["honest_verdict"] == "cross_model_blocked_no_gpu"
    assert result["inference_mode"] == "blocked"
    assert result["experiment"] == EXP_ID


def test_analysis_only_when_exp679_no_improvement() -> None:
    """When Exp 679 signed_improvement <= 0, _run_inner writes analysis_only artifact.

    Spec: REQ-VERIFY-162
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

    fake_exp679 = {"experiment": 679, "signed_improvement": -0.05}

    with (
        patch("scripts.experiment_template.ExperimentTemplate", return_value=_FakeTemplate()),
        patch("carnot.pipeline.atomic_writer.AtomicResultWriter", return_value=_FakeWriter()),
        patch.object(Path, "read_text", return_value=json.dumps(fake_exp679)),
        patch.object(Path, "exists", return_value=True),
    ):
        with pytest.raises(SystemExit) as exc_info:
            mod._run_inner(MagicMock())

    assert exc_info.value.code == 0
    assert len(written) == 1
    result = written[0]
    assert result["honest_verdict"] == "cross_model_analysis_only"
    assert result["inference_mode"] == "analysis_only"


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


def test_exp_id() -> None:
    """EXP_ID must be 694.  Spec: REQ-VERIFY-162"""
    assert EXP_ID == 694


def test_schema() -> None:
    """SCHEMA must encode the experiment version.  Spec: REQ-VERIFY-162"""
    assert SCHEMA == "carnot.vr_cross_model.v1"


def test_n_hard_questions() -> None:
    """N_HARD_QUESTIONS must be 50 per REQ-VERIFY-163-3.  Spec: REQ-VERIFY-163"""
    assert N_HARD_QUESTIONS == 50


# ---------------------------------------------------------------------------
# Deliverable JSON on disk — validates the actual run artifact
# ---------------------------------------------------------------------------


def test_deliverable_json_exists_and_valid() -> None:
    """Deliverable JSON on disk must exist and contain all required schema fields.

    Spec: REQ-VERIFY-162
    """
    result_path = _REPO_ROOT / DELIVERABLE
    if not result_path.exists():
        pytest.skip("Deliverable not yet written — run the experiment first")

    data = json.loads(result_path.read_text())

    required = {
        "experiment", "schema", "run_date", "status", "honest_verdict",
        "inference_mode", "qwen_signed_improvement", "gemma_baseline_acc",
        "gemma_post_acc", "gemma_signed_improvement", "cross_model_delta",
        "grammar_recall", "n_hard_questions", "hard_baseline_threshold",
    }
    missing = required - set(data.keys())
    assert not missing, f"Missing fields in deliverable: {missing}"

    assert data["experiment"] == EXP_ID
    valid_verdicts = {
        "vr_cross_model_confirmed",
        "vr_cross_model_partial",
        "vr_cross_model_no_improvement",
        "cross_model_analysis_only",
        "cross_model_blocked_no_gpu",
    }
    assert data["honest_verdict"] in valid_verdicts, (
        f"Unknown honest_verdict: {data['honest_verdict']}"
    )
    assert data["n_hard_questions"] == N_HARD_QUESTIONS
    assert data["hard_baseline_threshold"] == HARD_BASELINE_THRESHOLD

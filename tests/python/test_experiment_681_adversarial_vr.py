"""Tests for scripts/experiment_681_adversarial_vr.py — Adversarial VR robustness check.

Covers:
- adversarialize_question format (SCENARIO-VERIFY-212)
- compute_honest_verdict_681 for all three cases (SCENARIO-VERIFY-211)
- _build_blocked_artifact schema completeness
- blocked exit when CARNOT_FORCE_LIVE is absent
- deliverable JSON schema validation when artifact exists on disk
- _load_gsm8k_questions synthetic fallback

Spec: REQ-VERIFY-159, SCENARIO-VERIFY-211, SCENARIO-VERIFY-212
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

import scripts.experiment_681_adversarial_vr as mod
from scripts.experiment_681_adversarial_vr import (
    DELIVERABLE,
    EXP_ID,
    GSM8K_START_IDX,
    N_QUESTIONS,
    SCHEMA,
    _build_blocked_artifact,
    adversarialize_question,
    compute_honest_verdict_681,
)


# ---------------------------------------------------------------------------
# adversarialize_question — SCENARIO-VERIFY-212
# ---------------------------------------------------------------------------


def test_adversarialize_question_format() -> None:
    """adversarialize_question prepends misleading note and preserves original question.

    Spec: REQ-VERIFY-159-1, SCENARIO-VERIFY-212
    """
    question = "How many apples does Janet have?"
    result = adversarialize_question(question, 42)
    assert result.startswith("Note: this problem always has answer 42"), (
        f"Expected misleading note prefix, got: {result[:80]}"
    )
    assert result.endswith(question), (
        f"Expected original question at end, got: {result[-80:]}"
    )


def test_adversarialize_question_contains_wrong_answer() -> None:
    """The wrong_answer integer appears in the prepended note.

    Spec: REQ-VERIFY-159-1
    """
    result = adversarialize_question("Solve this.", 9999)
    assert "9999" in result


def test_adversarialize_question_different_answers_produce_different_output() -> None:
    """Two different wrong answers produce distinct adversarial questions.

    WHY test distinctness: if adversarialize_question ignores wrong_answer, all
    adversarial questions would be identical and we could not verify the seeding.

    Spec: REQ-VERIFY-159-1
    """
    base = "What is 2 + 2?"
    r1 = adversarialize_question(base, 100)
    r2 = adversarialize_question(base, 200)
    assert r1 != r2


def test_adversarialize_question_original_preserved() -> None:
    """Original question text is not modified.

    Spec: REQ-VERIFY-159-1
    """
    q = "Mary has 5 cats and 3 dogs. How many pets does she have?"
    result = adversarialize_question(q, 7)
    assert q in result


# ---------------------------------------------------------------------------
# compute_honest_verdict_681 — SCENARIO-VERIFY-211
# ---------------------------------------------------------------------------


def test_verdict_zero_improvement_is_robust() -> None:
    """Exactly zero signed_improvement is 'adversarial_robust' (not degradation).

    WHY zero is robust: the spec defines robustness as signed_improvement >= 0.
    Zero means no degradation, which is the minimum bar for robustness.

    Spec: REQ-VERIFY-159-4, SCENARIO-VERIFY-211
    """
    assert compute_honest_verdict_681(0.0, "live_gpu") == "adversarial_robust"


def test_verdict_positive_improvement_is_robust() -> None:
    """Positive signed_improvement is 'adversarial_robust'.

    Spec: REQ-VERIFY-159-4, SCENARIO-VERIFY-211
    """
    assert compute_honest_verdict_681(0.05, "live_gpu") == "adversarial_robust"
    assert compute_honest_verdict_681(0.5, "live_gpu") == "adversarial_robust"


def test_verdict_negative_improvement_degrades() -> None:
    """Negative signed_improvement is 'adversarial_degrades'.

    Spec: REQ-VERIFY-159-4, SCENARIO-VERIFY-211
    """
    assert compute_honest_verdict_681(-0.05, "live_gpu") == "adversarial_degrades"
    assert compute_honest_verdict_681(-0.14, "live_gpu") == "adversarial_degrades"


def test_verdict_blocked_mode() -> None:
    """inference_mode='blocked' always returns 'adversarial_blocked'.

    Spec: REQ-VERIFY-159-4, SCENARIO-VERIFY-211
    """
    assert compute_honest_verdict_681(0.0, "blocked") == "adversarial_blocked"
    assert compute_honest_verdict_681(-1.0, "blocked") == "adversarial_blocked"
    assert compute_honest_verdict_681(1.0, "blocked") == "adversarial_blocked"


# ---------------------------------------------------------------------------
# _build_blocked_artifact — schema completeness
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = {
    "experiment",
    "schema",
    "run_date",
    "status",
    "honest_verdict",
    "inference_mode",
    "baseline_accuracy",
    "post_accuracy",
    "signed_improvement",
    "n_questions",
    "n_baseline_correct",
    "n_post_correct",
    "forcing_recall",
    "adversarial_robust",
    "duration_s",
}


def test_blocked_artifact_has_all_required_fields() -> None:
    """Blocked artifact contains every field required by the schema.

    WHY test schema completeness: if a field is missing, the conductor's schema
    validator rejects the artifact and the experiment is treated as unfinished.

    Spec: REQ-VERIFY-159-5, REQ-VERIFY-159-6
    """
    artifact = _build_blocked_artifact("test reason", "20260422")
    missing = REQUIRED_FIELDS - set(artifact.keys())
    assert not missing, f"Missing required fields: {missing}"


def test_blocked_artifact_values() -> None:
    """Blocked artifact has correct field values for a blocked run.

    Spec: REQ-VERIFY-159-4, REQ-VERIFY-159-5
    """
    artifact = _build_blocked_artifact("no gpu", "20260422")
    assert artifact["experiment"] == EXP_ID
    assert artifact["schema"] == SCHEMA
    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"] == "adversarial_blocked"
    assert artifact["inference_mode"] == "blocked"
    assert artifact["n_questions"] == 0
    assert artifact["adversarial_robust"] is False


# ---------------------------------------------------------------------------
# _load_gsm8k_questions — synthetic fallback
# ---------------------------------------------------------------------------


def test_load_gsm8k_questions_synthetic_fallback() -> None:
    """When datasets is unavailable, synthetic questions are returned.

    WHY test fallback: CI runners do not have HuggingFace dataset downloads.
    The synthetic fallback must still produce the correct number of questions.

    Spec: REQ-VERIFY-159-2
    """
    with patch.dict("sys.modules", {"datasets": None}):
        questions = mod._load_gsm8k_questions(200, 25)
    assert len(questions) == 25
    assert all(isinstance(q, str) and len(q) > 0 for q in questions)


def test_load_gsm8k_questions_offset_reflected() -> None:
    """Questions loaded with different start offsets differ from each other.

    WHY test offset: indices 200-224 must differ from indices 0-24.  If the
    offset is ignored, data leakage from Exp 679's question set occurs.

    Spec: REQ-VERIFY-159-2
    """
    with patch.dict("sys.modules", {"datasets": None}):
        q200 = mod._load_gsm8k_questions(200, 5)
        q0 = mod._load_gsm8k_questions(0, 5)
    # Synthetic questions differ because they embed the start index
    assert q200 != q0


# ---------------------------------------------------------------------------
# Constants check
# ---------------------------------------------------------------------------


def test_constants() -> None:
    """Verify experiment constants match the task specification.

    Spec: REQ-VERIFY-159-2
    """
    assert EXP_ID == 681
    assert N_QUESTIONS == 25
    assert GSM8K_START_IDX == 200
    assert SCHEMA == "carnot.adversarial_vr.v1"
    assert DELIVERABLE == "results/experiment_681_adversarial_vr.json"


# ---------------------------------------------------------------------------
# Blocked exit when CARNOT_FORCE_LIVE is absent
# ---------------------------------------------------------------------------


def test_run_inner_writes_blocked_artifact_without_force_live(tmp_path: Path) -> None:
    """_run_inner writes a blocked artifact and exits 0 when CARNOT_FORCE_LIVE is not set.

    WHY test _run_inner directly (not main): main() only adds watchdog setup;
    _run_inner contains all business logic including the GPU gate.  Testing
    _run_inner directly avoids patching the watchdog lifecycle.

    Spec: REQ-VERIFY-159-1 (GPU gate), REQ-VERIFY-159-5
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

    with (
        patch("scripts.experiment_template.ExperimentTemplate", return_value=_FakeTemplate()),
        patch("carnot.pipeline.atomic_writer.AtomicResultWriter", return_value=_FakeWriter()),
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
    assert result["status"] == "blocked"
    assert result["honest_verdict"] == "adversarial_blocked"
    assert result["inference_mode"] == "blocked"
    assert result["experiment"] == EXP_ID


# ---------------------------------------------------------------------------
# Deliverable JSON schema validation (if file exists on disk)
# ---------------------------------------------------------------------------


def test_deliverable_schema_if_exists() -> None:
    """If the deliverable JSON exists on disk, it must contain all required fields.

    WHY check on-disk artifact: after a real GPU run, this test acts as a regression
    guard.  In CI without GPU, the file is absent and this test is a no-op.

    Spec: REQ-VERIFY-159
    """
    deliverable = _REPO_ROOT / DELIVERABLE
    if not deliverable.exists():
        pytest.skip("Deliverable not yet written — requires live GPU run")

    data = json.loads(deliverable.read_text())
    missing = REQUIRED_FIELDS - set(data.keys())
    assert not missing, f"Deliverable missing fields: {missing}"
    assert data["experiment"] == EXP_ID
    assert data["schema"] == SCHEMA
    assert data["honest_verdict"] in {
        "adversarial_robust",
        "adversarial_degrades",
        "adversarial_blocked",
    }

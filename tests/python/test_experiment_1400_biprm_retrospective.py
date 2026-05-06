"""Tests for Exp 1400 BiPRM R2L retrospective FoVer pivot probe.

Spec: REQ-VERIFY-1400, SCENARIO-VERIFY-1400
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.eval import biprm_retrospective_verification_probe as exp


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_write_in_progress_artifact_for_req1400(tmp_path: Path) -> None:
    """REQ-VERIFY-1400: the probe writes an in-progress artifact first."""

    output_path = tmp_path / "experiment_1400_biprm_retrospective_verification_probe.json"

    artifact = exp.write_in_progress_artifact(output_path, run_date="20260506")
    persisted = json.loads(output_path.read_text(encoding="utf-8"))

    assert artifact == persisted
    assert persisted["status"] == "in_progress"
    assert persisted["corpus_cases_used"] == 0
    assert persisted["r2l_update_rule"] == "r_t^R2L = f_theta(s_t | q, s_>t)"


def test_load_fover_verified_pairs_uses_positive_and_negative_rows(tmp_path: Path) -> None:
    """REQ-VERIFY-1400: FoVer rows become same-question verified pairs."""

    corpus = tmp_path / "fover.jsonl"
    _write_jsonl(
        corpus,
        [
            {"question_id": "q1", "step_text": "2 + 2 = 4.", "label": "correct"},
            {"question_id": "q1", "step_text": "2 + 2 = 5.", "label": "incorrect"},
            {"question_id": "q2", "step_text": "3 + 3 = 6.", "label": "correct"},
        ],
    )

    pairs = exp.load_fover_verified_pairs(corpus, limit=10)

    assert len(pairs) == 1
    assert pairs[0].case_id == "q1"
    assert pairs[0].positive_text == "2 + 2 = 4."
    assert pairs[0].negative_text == "2 + 2 = 5."
    assert pairs[0].gold_pivot_indices == (0,)


def test_r2l_scores_missing_premise_pivot_from_future_context() -> None:
    """SCENARIO-VERIFY-1400: R2L uses later evidence to localize a pivot."""

    positive = "Remaining students are 12. 12% of 12 students is 1.44. Answer is 1.44."
    negative = (
        "Remaining students are 12.\n"
        "Therefore, 12% of 20 students is 2.4.\n"
        "The final answer is 2.4."
    )
    pair = exp.FoVerRetrospectivePair.from_texts(
        case_id="missing-premise",
        positive_text=positive,
        negative_text=negative,
    )

    scored = exp.score_pair(pair)

    assert pair.gold_pivot_indices == (1,)
    assert pair.gold_pivot_category == "missing_premise"
    assert scored.biprm_pivot_index == 1
    assert scored.biprm_r2l_scores[1] > scored.biprm_r2l_scores[2]


def test_forward_leave_one_out_can_blame_final_answer_baseline() -> None:
    """REQ-VERIFY-1400: forward-only scoring remains the leave-one-out baseline."""

    positive = "Remaining students are 12. 12% of 12 students is 1.44. Answer is 1.44."
    negative = (
        "Remaining students are 12.\n"
        "Therefore, 12% of 20 students is 2.4.\n"
        "The final answer is 2.4."
    )
    pair = exp.FoVerRetrospectivePair.from_texts(
        case_id="baseline-final",
        positive_text=positive,
        negative_text=negative,
    )

    scored = exp.score_pair(pair)

    assert scored.forward_pivot_index == 2
    assert scored.forward_pivot_index not in pair.gold_pivot_indices


def test_build_artifact_required_fields_and_viability_gate() -> None:
    """REQ-VERIFY-1400: artifact fields and delta gate are deterministic."""

    pairs = [
        exp.FoVerRetrospectivePair.from_texts(
            case_id="a",
            positive_text="2 + 2 = 4. Answer is 4.",
            negative_text="2 + 2 = 5. Answer is 5.",
        ),
        exp.FoVerRetrospectivePair.from_texts(
            case_id="b",
            positive_text="Remaining is 12. 12% of 12 = 1.44. Answer is 1.44.",
            negative_text="Remaining is 12.\nTherefore, 12% of 20 = 2.4.\nAnswer is 2.4.",
        ),
    ]
    scores = [exp.score_pair(pair) for pair in pairs]

    artifact = exp.build_artifact(
        pairs,
        scores,
        corpus_path=Path("fixture.jsonl"),
        started_at="2026-05-06T00:00:00Z",
        duration_s=0.1,
        run_date="20260506",
    )

    assert exp.REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["corpus_cases_used"] == 2
    assert artifact["pivot_precision_delta"] == pytest.approx(
        artifact["biprm_r2l_pivot_precision"] - artifact["forward_only_pivot_precision"]
    )
    assert artifact["retrospective_verification_viable"] is (artifact["pivot_precision_delta"] > 0)
    assert set(artifact["pivotal_step_categories"]) == {
        "arithmetic_error",
        "logical_fallacy",
        "missing_premise",
        "hallucination",
    }


def test_run_experiment_writes_complete_req1400_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1400: runner writes the complete required artifact."""

    corpus = tmp_path / "fover.jsonl"
    output_path = tmp_path / "experiment_1400_biprm_retrospective_verification_probe.json"
    _write_jsonl(
        corpus,
        [
            {
                "question_id": "q1",
                "step_text": "Compute 2 + 2 = 4. The final answer is 4.",
                "label": "correct",
            },
            {
                "question_id": "q1",
                "step_text": "Compute 2 + 2 = 5. The final answer is 5.",
                "label": "incorrect",
            },
            {
                "question_id": "q2",
                "step_text": "Remaining is 12. 12% of 12 = 1.44. Answer is 1.44.",
                "label": "correct",
            },
            {
                "question_id": "q2",
                "step_text": "Remaining is 12.\nTherefore, 12% of 20 = 2.4.\nAnswer is 2.4.",
                "label": "incorrect",
            },
        ],
    )

    artifact = exp.run_experiment(
        corpus_path=corpus,
        output_path=output_path,
        limit=10,
        run_date="20260506",
    )
    persisted = json.loads(output_path.read_text(encoding="utf-8"))

    assert artifact == persisted
    assert persisted["status"] == "complete"
    assert persisted["corpus_cases_used"] == 2
    assert isinstance(persisted["honest_verdict"], str)

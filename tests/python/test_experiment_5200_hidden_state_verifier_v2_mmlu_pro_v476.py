"""Tests for Exp 5200 hidden-state verifier v2 on MMLU-Pro.

Spec refs: REQ-REPORT-5200, SCENARIO-REPORT-5200,
SCENARIO-REPORT-5200-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot import experiment_5200_hidden_state_verifier_v2_mmlu_pro_v476 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _write_headroom(root: Path) -> None:
    path = root / mod.HEADROOM_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "oracle_at_k": 0.35,
                "sc_vote": 0.075,
                "headroom": 0.275,
                "headroom_ci95": [0.15, 0.425],
                "adversarial_verify_flags": 0,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_pool(root: Path, *, n_questions: int = 8, n_candidates: int = 4) -> None:
    path = root / mod.CANDIDATE_POOL_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for qi in range(n_questions):
        gold = "D"
        for ci in range(n_candidates):
            correct = ci == n_candidates - 1
            rows.append(
                {
                    "question_index": qi,
                    "question_id": f"mmlu-{qi:03d}",
                    "category": "fixture",
                    "k": ci,
                    "gold": gold,
                    "parsed_letter": gold if correct else "A",
                    "correct": correct,
                    "full_text": (
                        f"Step {ci}. {'correct-signal' if correct else 'wrong-cluster'} "
                        f"reasoning for question {qi}. Final answer: {gold if correct else 'A'}."
                    ),
                    "token_logprobs": [-4.0] if correct else [-0.01],
                    "top_logprobs": [
                        {"A": -1.3863, "B": -1.3863, "C": -1.3863, "D": -1.3863}
                        if correct
                        else {"A": -0.01, "B": -6.0, "C": -6.0, "D": -6.0}
                    ],
                }
            )
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _repo_with_pool(tmp_path: Path) -> Path:
    _write_headroom(tmp_path)
    _write_pool(tmp_path)
    (tmp_path / "ops").mkdir(parents=True, exist_ok=True)
    (tmp_path / mod.VERIFIER_GAPS_RELATIVE_PATH).write_text("# Verifier gaps\n", encoding="utf-8")
    return tmp_path


def _fake_vectors(texts: list[str]) -> np.ndarray:
    vectors: list[list[float]] = []
    for text in texts:
        if "correct-signal" in text:
            vectors.append([4.0, 0.0, 1.0, 0.0])
        elif "wrong-cluster" in text:
            vectors.append([0.0, 3.0, 0.0, 1.0])
        else:
            vectors.append([0.0, 0.0, 0.0, 0.0])
    return np.asarray(vectors, dtype=float)


def test_req_report_5200_spec_declares_v2_contract() -> None:
    """REQ-REPORT-5200: OpenSpec declares the five-arm MMLU-Pro contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-5200") :]

    for marker in (
        "REQ-REPORT-5200",
        "SCENARIO-REPORT-5200",
        "SCENARIO-REPORT-5200-BLOCKED-PRECONDITION",
        mod.RESULT_RELATIVE_PATH,
        mod.CANDIDATE_POOL_RELATIVE_PATH,
    ):
        assert marker in section
    for field in mod.REQUIRED_PRINCIPLED_FIELDS:
        assert f"`{field}`" in section


def test_scenario_report_5200_builds_five_way_artifact(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5200: the probe is evaluated against all four controls."""

    root = _repo_with_pool(tmp_path)
    artifact = mod.run(
        root=root,
        result_path=root / mod.RESULT_RELATIVE_PATH,
        vector_provider=_fake_vectors,
        hidden_state_status=mod.HiddenStateAccessStatus(
            feasible=True,
            reason="fixture final-token vectors available",
            metadata={
                "model_id": mod.HIDDEN_MODEL_ID,
                "gguf_path": "fixture.gguf",
                "hidden_state_extraction_path": "fixture",
                "vector_shape": [4],
            },
        ),
        expected_pool_rows=32,
        n_folds=4,
        n_bootstrap=200,
        duration_s=2.5,
        tests_run=["unit fixture"],
    )

    assert artifact["probe_accuracy"]["value"] == pytest.approx(1.0)
    assert artifact["probe_accuracy"]["value"] > artifact["tuned_sc_accuracy"]["value"]
    assert artifact["probe_accuracy"]["value"] > artifact["self_certainty_accuracy"]["value"]
    assert artifact["probe_accuracy"]["value"] > artifact["clue_accuracy"]["value"]
    assert artifact["probe_accuracy"]["value"] > artifact["radial_consensus_score_accuracy"]["value"]
    assert artifact["probe_vs_sc_delta_ci95"]["value"][0] > 0.0
    assert 0.0 <= artifact["probe_vs_sc_mcnemar_p"]["value"] <= 1.0
    assert artifact["probe_vs_rcs_delta_ci95"]["value"][0] > 0.0
    assert artifact["n_questions"]["value"] == 8
    assert artifact["layer_sweep_attempted"]["value"] is False
    assert artifact["headroom_present"]["value"] is True
    assert artifact["verifier_is_oracle"]["value"] is False
    assert artifact["inference_substrate"]["value"] == "live_llm_embedding_extraction"
    assert "beats_tuned_sc" in artifact["honest_verdict"]["value"]
    assert "beats_all_zero_training_controls" in artifact["honest_verdict"]["value"]
    assert artifact["reproducibility_checksum"]["value"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []
    assert "experiment_5200" in (root / mod.VERIFIER_GAPS_RELATIVE_PATH).read_text(encoding="utf-8")
    assert json.loads((root / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact


def test_req_report_5200_question_folds_do_not_leak_candidates(tmp_path: Path) -> None:
    """REQ-REPORT-5200: train/eval splits are grouped by question_id."""

    root = _repo_with_pool(tmp_path)
    questions = mod.load_mmlu_questions(root, expected_rows=32)
    folds = mod.question_folds([q.question_id for q in questions], n_folds=4, seed=7)

    seen: set[str] = set()
    for fold in folds:
        assert seen.isdisjoint(fold)
        seen.update(fold)
        train_rows, eval_rows = mod.rows_for_split(questions, fold)
        train_q = {questions[row.question_pos].question_id for row in train_rows}
        eval_q = {questions[row.question_pos].question_id for row in eval_rows}
        assert train_q.isdisjoint(eval_q)
    assert seen == {q.question_id for q in questions}


def test_scenario_report_5200_blocked_candidate_pool_is_terminal(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5200-BLOCKED-PRECONDITION: missing pool blocks honestly."""

    _write_headroom(tmp_path)
    artifact = mod.run(
        root=tmp_path,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        expected_pool_rows=32,
        duration_s=1.0,
        tests_run=["blocked fixture"],
    )

    assert artifact["honest_verdict"]["value"].startswith("blocked_candidate_pool")
    assert artifact["n_questions"]["value"] == 0
    assert artifact["headroom_present"]["value"] is True
    assert artifact["verifier_is_oracle"]["value"] is False
    assert artifact["inference_substrate"]["value"] == "live_llm_embedding_extraction"
    assert mod.artifact_schema_errors(artifact) == []


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda artifact: {key: value for key, value in artifact.items() if key != "probe_accuracy"},
            "missing required fields",
        ),
        (
            lambda artifact: artifact
            | {"verifier_is_oracle": {"value": True, "principle": mod.FIELD_PRINCIPLES["verifier_is_oracle"]}},
            "verifier_is_oracle",
        ),
        (
            lambda artifact: artifact | {"headroom_present": True},
            "principle-wrapped",
        ),
        (
            lambda artifact: artifact
            | {"probe_vs_sc_delta_ci95": {"value": [0.1], "principle": mod.FIELD_PRINCIPLES["probe_vs_sc_delta_ci95"]}},
            "CI95",
        ),
        (
            lambda artifact: artifact
            | {"honest_verdict": {"value": "done", "principle": mod.FIELD_PRINCIPLES["honest_verdict"]}},
            "honest_verdict",
        ),
    ],
)
def test_req_report_5200_schema_rejects_bad_artifacts(
    tmp_path: Path,
    mutate: Any,
    message: str,
) -> None:
    """REQ-REPORT-5200: malformed required fields fail closed."""

    artifact = mod.run(
        root=_repo_with_pool(tmp_path),
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        vector_provider=_fake_vectors,
        hidden_state_status=mod.HiddenStateAccessStatus(
            feasible=True,
            reason="fixture",
            metadata={"hidden_state_extraction_path": "fixture", "vector_shape": [4]},
        ),
        expected_pool_rows=32,
        n_folds=4,
        n_bootstrap=80,
        duration_s=2.5,
    )

    errors = mod.artifact_schema_errors(mutate(artifact))

    assert any(message in error for error in errors)

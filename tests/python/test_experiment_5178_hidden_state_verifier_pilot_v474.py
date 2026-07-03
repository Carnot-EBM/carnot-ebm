"""Tests for Exp 5178 hidden-state verifier pilot.

Spec refs: REQ-REPORT-5178, SCENARIO-REPORT-5178,
SCENARIO-REPORT-5178-BLOCKED-HIDDEN-ACCESS.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot import experiment_5178_hidden_state_verifier_pilot_v474 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _candidate(answer: str, correct: bool, signal: str) -> dict[str, Any]:
    return {
        "answer": answer,
        "correct": int(correct),
        "reasoning": f"{signal} reasoning paragraph. ANSWER: {answer}",
    }


def _write_trace(root: Path, q: int, gold: str, candidates: list[dict[str, Any]]) -> None:
    trace_dir = root / mod.MUSR_TRACES_RELATIVE_PATH
    trace_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "q": q,
        "question": f"Who is responsible in synthetic case {q}?",
        "narrative": f"Synthetic MuSR narrative {q}",
        "choices": ["A", "B", "C"],
        "gold": gold,
        "n_candidates": len(candidates),
        "candidates": candidates,
    }
    (trace_dir / f"q{q:04d}.json").write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )


def _write_phase_d_context(root: Path) -> None:
    path = root / mod.PHASE_D_MUSR_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "honest_verdict": "complete_musr_energy_verifier_no_win",
                "self_consistency_accuracy": 0.56,
                "distributional_energy_accuracy": 0.515,
                "verifier_is_oracle": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (root / "research-references.md").write_text(
        "TrajSelector VerifySteer Discriminative Verification Explanatory Verifier\n",
        encoding="utf-8",
    )


def _repo_with_traces(tmp_path: Path) -> Path:
    _write_phase_d_context(tmp_path)
    rows = [
        ("A", [("B", False, "distractor"), ("A", True, "reliable"), ("A", True, "reliable")]),
        ("A", [("B", False, "distractor"), ("B", False, "distractor"), ("A", True, "reliable")]),
        ("B", [("A", False, "distractor"), ("B", True, "reliable"), ("B", True, "reliable")]),
        ("B", [("A", False, "distractor"), ("A", False, "distractor"), ("B", True, "reliable")]),
        ("C", [("A", False, "distractor"), ("A", False, "distractor"), ("A", False, "distractor")]),
        ("C", [("B", False, "distractor"), ("C", True, "reliable"), ("C", True, "reliable")]),
    ]
    for q, (gold, candidates) in enumerate(rows):
        _write_trace(
            tmp_path,
            q,
            gold,
            [_candidate(answer, correct, signal) for answer, correct, signal in candidates],
        )
    return tmp_path


def _fake_vectors(texts: list[str]) -> np.ndarray:
    vectors: list[list[float]] = []
    for text in texts:
        if "reliable" in text:
            vectors.append([2.0, 0.1, 1.0])
        elif "distractor" in text:
            vectors.append([0.1, 2.0, -1.0])
        else:
            vectors.append([0.5, 0.5, 0.0])
    return np.asarray(vectors, dtype=float)


def test_req_report_5178_spec_declares_hidden_state_contract() -> None:
    """REQ-REPORT-5178: OpenSpec declares the hidden-state pilot fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-5178") :]

    for marker in (
        "REQ-REPORT-5178",
        "SCENARIO-REPORT-5178",
        "SCENARIO-REPORT-5178-BLOCKED-HIDDEN-ACCESS",
        mod.RESULT_RELATIVE_PATH,
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_report_5178_builds_hidden_vector_pilot_artifact(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5178: hidden-vector scoring is compared to tuned SC."""

    root = _repo_with_traces(tmp_path)
    artifact = mod.run(
        root=root,
        result_path=root / mod.RESULT_RELATIVE_PATH,
        vector_provider=_fake_vectors,
        max_questions=6,
        n_folds=3,
        n_bootstrap=200,
        duration_s=12.5,
        hidden_state_status=mod.HiddenStateAccessStatus(
            feasible=True,
            reason="fixture final-token hidden vectors available",
            metadata={
                "model_id": mod.HIDDEN_MODEL_ID,
                "hidden_state_extraction_path": "fixture",
                "vector_shape": [3],
            },
        ),
        tests_run=["unit fixture"],
    )

    assert artifact["hidden_state_access_feasible"]["value"] is True
    assert artifact["design_path_taken"]["value"].startswith("trajselector_trained_probe")
    assert artifact["corpus_used"]["value"] == "MuSR/murder_mysteries"
    assert artifact["tuned_sc_baseline_accuracy"]["value"] < artifact["oracle_at_k_accuracy"]
    assert artifact["hidden_state_verifier_accuracy"]["value"] >= artifact["tuned_sc_baseline_accuracy"]["value"]
    assert len(artifact["accuracy_delta_ci95"]["value"]) == 2
    assert 0.0 <= artifact["mcnemar_p_value"]["value"] <= 1.0
    assert artifact["identically_wrong_detection_result"]["value"]["n_cases"] >= 1
    assert artifact["identically_wrong_detection_result"]["value"]["sc_detection_rate"] == 0.0
    assert artifact["compute_cost_vs_sc"]["value"]["candidate_vectors_scored"] == artifact["pilot_n_candidates"]
    assert artifact["compute_cost_vs_llm_judge"]["value"]["generative_decode_tokens_required"] == 0
    assert artifact["verifier_is_oracle"]["value"] is False
    assert artifact["headroom_present"]["value"] is True
    assert artifact["random_seed"]["value"] == mod.RANDOM_SEED
    assert artifact["inference_substrate"]["value"] == "live_llm_inference"
    assert artifact["honest_verdict"]["value"].startswith(("complete_", "success_"))
    assert artifact["reproducibility_checksum"]["value"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []
    assert json.loads((root / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact


def test_scenario_report_5178_blocked_hidden_access_is_terminal(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5178-BLOCKED-HIDDEN-ACCESS: no vectors blocks honestly."""

    _write_phase_d_context(tmp_path)
    artifact = mod.build_blocked_artifact(
        reason="blocked_hidden_state_access_infeasible: llama.cpp exposed no vectors",
        duration_s=1.0,
        tests_run=["blocked fixture"],
    )

    assert artifact["hidden_state_access_feasible"]["value"] is False
    assert artifact["design_path_taken"]["value"].startswith("blocked_hidden_state_access_infeasible")
    assert artifact["hidden_state_verifier_accuracy"]["value"] == 0.0
    assert artifact["tuned_sc_baseline_accuracy"]["value"] == 0.0
    assert artifact["accuracy_delta_ci95"]["value"] == [0.0, 0.0]
    assert artifact["verifier_is_oracle"]["value"] is False
    assert artifact["honest_verdict"]["value"].startswith("blocked_hidden_state_access_infeasible")
    assert mod.artifact_schema_errors(artifact) == []


def test_req_report_5178_statistics_are_paired_and_tuned(tmp_path: Path) -> None:
    """REQ-REPORT-5178: tuned SC, McNemar, and bootstrap stay paired."""

    questions = mod.load_musr_questions(_repo_with_traces(tmp_path))
    folds = mod.question_folds(len(questions), n_folds=3, seed=123)
    tuned = mod.cross_validated_tuned_sc(questions, folds)

    assert set(tuned.tuned_k_by_fold) == {0, 1, 2}
    assert set(tuned.sc_correct_by_question) == set(range(len(questions)))
    assert max(tuned.k_candidates) == 3

    ci = mod.paired_bootstrap_ci([1, 0, 1, 0], [0, 0, 1, 1], n_bootstrap=200, seed=3)
    p = mod.mcnemar_exact_p([1, 0, 1, 0], [0, 0, 1, 1])

    assert len(ci) == 2
    assert ci[0] <= ci[1]
    assert p == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda artifact: {key: value for key, value in artifact.items() if key != "random_seed"},
            "missing required fields",
        ),
        (
            lambda artifact: artifact
            | {"verifier_is_oracle": {"value": True, "principle": mod.FIELD_PRINCIPLES["verifier_is_oracle"]}},
            "verifier_is_oracle",
        ),
        (
            lambda artifact: artifact | {"hidden_state_access_feasible": True},
            "principle-wrapped",
        ),
        (
            lambda artifact: artifact
            | {"accuracy_delta_ci95": {"value": [0.1], "principle": mod.FIELD_PRINCIPLES["accuracy_delta_ci95"]}},
            "CI95",
        ),
        (
            lambda artifact: artifact
            | {"honest_verdict": {"value": "done", "principle": mod.FIELD_PRINCIPLES["honest_verdict"]}},
            "honest_verdict",
        ),
    ],
)
def test_req_report_5178_schema_rejects_bad_artifacts(
    tmp_path: Path,
    mutate: Any,
    message: str,
) -> None:
    """REQ-REPORT-5178: malformed required fields fail closed."""

    artifact = mod.run(
        root=_repo_with_traces(tmp_path),
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        vector_provider=_fake_vectors,
        max_questions=6,
        n_folds=3,
        n_bootstrap=80,
        duration_s=12.5,
        hidden_state_status=mod.HiddenStateAccessStatus(
            feasible=True,
            reason="fixture",
            metadata={"hidden_state_extraction_path": "fixture"},
        ),
    )

    errors = mod.artifact_schema_errors(mutate(artifact))

    assert any(message in error for error in errors)

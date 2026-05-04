"""Tests for Exp 1288 InterWhen DVI verifier-feedback replay.

Spec: REQ-LEARN-1288, SCENARIO-LEARN-1288.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.training import interwhen_dvi_verifier_feedback_replay as exp


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fover_payload() -> dict:
    return {
        "metadata": {"schema": "test.fover.v5"},
        "pairs": [
            {
                "question_index": 1,
                "question": "What is 3 plus 4?",
                "response": "3 + 4 = 8. Therefore final answer: 8.",
                "is_correct": False,
            },
            {
                "question_index": 2,
                "question": "What is 5 plus 6?",
                "response": "5 + 6 = 11. Therefore final answer: 11.",
                "is_correct": True,
            },
            {
                "question_index": 3,
                "question": "A recipe uses half of a 10 cup bag. How many cups?",
                "response": "The amount is 6 cups.",
                "is_correct": False,
            },
            {
                "question_index": 4,
                "question": "Mia needs half of an 8 mile route. How many miles?",
                "response": "The route is 5 miles.",
                "is_correct": False,
            },
            {
                "question_index": 5,
                "question": "What is 9 plus 1?",
                "response": "9 + 1 = 10. Therefore final answer: 10.",
                "is_correct": True,
            },
        ],
    }


def test_in_progress_artifact_written_first_for_req1288(tmp_path: Path) -> None:
    """REQ-LEARN-1288-1: the skeleton artifact is parseable before replay finishes."""

    output_path = tmp_path / "experiment_1288_interwhen_dvi_verifier_feedback_replay.json"

    artifact = exp.write_in_progress_artifact(output_path, run_date="20260504")

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert artifact["experiment"] == "1288_interwhen_dvi_verifier_feedback_replay"
    assert artifact["schema"] == "interwhen_dvi_verifier_feedback_replay_v1"
    assert artifact["status"] == "in_progress"
    assert artifact["honest_verdict"] == "in_progress"
    assert artifact["headline_result_allowed"] is False


def test_fover_fallback_online_replay_updates_before_next_item_for_scenario1288(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-1288-2/3 and SCENARIO-LEARN-1288: fallback replay learns online."""

    exp1274_path = tmp_path / "experiment_1274_online_self_learning_certificate_memory_v3.json"
    fover_path = tmp_path / "fover_corpus_v5.json"
    blocked_sota_path = tmp_path / "experiment_1286_grad_beaver_nsvif_semantic_routing.json"
    _write_json(
        exp1274_path,
        {
            "status": "complete",
            "source": "fover_fallback",
            "self_learning_delta_overall": 0.1,
            "memory_entries": 2,
            "skill_graph_candidates": [],
        },
    )
    _write_json(fover_path, _fover_payload())
    _write_json(blocked_sota_path, {"status": "blocked", "routing_records": []})

    examples, source = exp.load_feedback_examples(
        sota_paths=(blocked_sota_path,),
        exp1274_path=exp1274_path,
        fover_path=fover_path,
    )
    result = exp.compare_replay_modes(examples, build_fraction=0.4)

    assert source == "fover_fallback"
    assert result["n_memory_build_examples"] == 2
    assert result["n_replay_eval_examples"] == 3
    assert result["dvi_acceptance_delta"] == 0.0
    assert result["online_acceptance_delta"] > result["dvi_acceptance_delta"]
    assert result["violation_delta"] < 0.0
    assert result["self_learning_delta_overall"] > 0.0
    assert result["self_verify_signal_used"] is True
    assert result["memory_update_written"] is True
    assert result["claim_level_memory_entries"] >= 3
    first_slice, second_slice = result["replay_slices"][:2]
    assert first_slice["before_policy_state"]["memory_entries"] == 2
    assert first_slice["after_policy_state"]["memory_entries"] == 3
    assert second_slice["online_decision"] == "repair"


def test_run_experiment_writes_complete_artifact_for_req1288(tmp_path: Path) -> None:
    """REQ-LEARN-1288-4/5: final artifact records honest non-headline replay metrics."""

    exp1274_path = tmp_path / "experiment_1274_online_self_learning_certificate_memory_v3.json"
    fover_path = tmp_path / "fover_corpus_v5.json"
    output_path = tmp_path / "experiment_1288_interwhen_dvi_verifier_feedback_replay.json"
    _write_json(exp1274_path, {"status": "complete", "source": "fover_fallback"})
    _write_json(fover_path, _fover_payload())

    artifact = exp.run_experiment(
        sota_paths=(),
        exp1274_path=exp1274_path,
        fover_path=fover_path,
        output_path=output_path,
        run_date="20260504",
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
        build_fraction=0.4,
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    exp.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["source"] == "fover_fallback"
    assert artifact["headline_result_allowed"] is False
    assert artifact["honest_verdict"] in {
        "online_verifier_feedback_improved_non_headline",
        "online_verifier_feedback_neutral_non_headline",
        "online_verifier_feedback_regressed_non_headline",
    }
    assert artifact["verification_gain"] == artifact["self_learning_delta_overall"]
    assert isinstance(artifact["reasoning_trace_length_delta"], float)


def test_req1288_sota_path_and_validation_edges(tmp_path: Path) -> None:
    """REQ-LEARN-1288: SOTA records, verdicts, empty replay, and validation are strict."""

    sota_path = tmp_path / "experiment_1285_triggered_certificate_extraction_v2.json"
    fover_path = tmp_path / "fover.json"
    _write_json(fover_path, _fover_payload())
    _write_json(
        sota_path,
        {
            "status": "complete",
            "verification_certificates": [
                {
                    "id": "cert_ok",
                    "question": "What is 2 plus 2?",
                    "response": "2 + 2 = 4.",
                    "verifier_result": "passed",
                    "constraint_pattern": "arithmetic:addition",
                },
                {
                    "id": "cert_bad",
                    "question": "What is half of 10?",
                    "response": "The result is 7.",
                    "verifier_result": "failed",
                    "constraint_pattern": "arithmetic:ratio",
                    "repair_hint": "recompute_arithmetic_result",
                },
            ],
        },
    )

    examples, source = exp.load_feedback_examples(
        sota_paths=(sota_path,),
        exp1274_path=None,
        fover_path=fover_path,
    )

    assert source == "sota_certificates"
    assert [example.example_id for example in examples] == ["cert_ok", "cert_bad"]
    assert examples[0].target_decision == "accept"
    assert examples[1].target_decision == "repair"
    assert exp.compare_replay_modes(examples[:1])["n_replay_eval_examples"] == 0
    assert exp.derive_honest_verdict(0.1, headline_allowed=True) == (
        "online_verifier_feedback_improved_headline_candidate"
    )
    assert exp.derive_honest_verdict(-0.1, headline_allowed=False) == (
        "online_verifier_feedback_regressed_non_headline"
    )
    assert exp.derive_honest_verdict(0.0, headline_allowed=False) == (
        "online_verifier_feedback_neutral_non_headline"
    )
    assert exp._read_json_if_exists(None) is None
    assert exp._sota_rows({"status": "complete"}) == []
    inferred = exp._examples_from_sota(
        {
            "status": "complete",
            "verification_certificates": [
                {
                    "question": "What is half of 12?",
                    "answer": "The result is 6.",
                    "verifier_result": "passed",
                }
            ],
        }
    )
    assert inferred[0].constraint_pattern == "arithmetic:ratio"
    with pytest.raises(ValueError, match="build_fraction"):
        exp.compare_replay_modes(examples, build_fraction=1.0)

    good = exp.run_experiment(
        sota_paths=(sota_path,),
        exp1274_path=None,
        fover_path=fover_path,
        output_path=tmp_path / "artifact.json",
        run_date="20260504",
    )
    for key, message in [
        ("experiment", "missing required fields"),
        ("status", "status must be complete"),
        ("source", "unsupported source"),
        ("verification_gain", "verification_gain must equal"),
        ("self_learning_delta_overall", "verification_gain"),
        ("memory_update_written", "memory_update_written must be boolean"),
        ("headline_result_allowed", "headline_result_allowed must be boolean"),
        ("replay_slices", "replay_slices must be a list"),
        ("claim_level_memory_entries", "claim_level_memory_entries must be non-negative"),
    ]:
        bad = dict(good)
        if key == "status":
            bad[key] = "blocked"
        elif key == "source":
            bad[key] = "unknown"
        elif key == "verification_gain":
            bad[key] = 99.0
        elif key == "self_learning_delta_overall":
            bad[key] = 99.0
        elif key in {"memory_update_written", "headline_result_allowed"}:
            bad[key] = "yes"
        elif key == "replay_slices":
            bad[key] = {}
        elif key == "claim_level_memory_entries":
            bad[key] = -1
        else:
            del bad[key]
        with pytest.raises(AssertionError, match=message):
            exp.validate_artifact(bad)

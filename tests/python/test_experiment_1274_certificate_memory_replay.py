"""Tests for Exp 1274 certificate-memory replay evaluation.

Spec: REQ-LEARN-1274, SCENARIO-LEARN-1275.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pytest

from carnot.training import certificate_memory_replay as exp


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fover_payload() -> dict:
    return {
        "metadata": {"schema": "test.fover"},
        "pairs": [
            {
                "question_index": 1,
                "question": "What is 3 plus 4?",
                "response": "3 + 4 = 8. Therefore final answer: 8.",
                "is_correct": False,
                "model": "wrong-model",
            },
            {
                "question_index": 2,
                "question": "What is 5 plus 6?",
                "response": "5 + 6 = 11. Therefore final answer: 11.",
                "is_correct": True,
                "model": "right-model",
            },
            {
                "question_index": 3,
                "question": "What is 7 plus 8?",
                "response": "7 + 8 = 16. Therefore final answer: 16.",
                "is_correct": False,
                "model": "wrong-model",
            },
            {
                "question_index": 4,
                "question": "What is 9 plus 1?",
                "response": "9 + 1 = 10. Therefore final answer: 10.",
                "is_correct": True,
                "model": "right-model",
            },
        ],
    }


def test_in_progress_artifact_written_first_for_req1274(tmp_path: Path) -> None:
    """REQ-LEARN-1274-1: skeleton artifact is parseable before replay finishes."""

    output_path = tmp_path / "experiment_1274_online_self_learning_certificate_memory_v3.json"

    artifact = exp.write_in_progress_artifact(output_path, run_date="20260504")

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert artifact["experiment"] == "1274_online_self_learning_certificate_memory_v3"
    assert artifact["schema"] == "certificate_memory_replay_v3"
    assert artifact["status"] == "in_progress"
    assert artifact["honest_verdict"] == "in_progress"


def test_fover_fallback_examples_for_scenario1275(tmp_path: Path) -> None:
    """REQ-LEARN-1274-2/SCENARIO-LEARN-1275: blocked Exp 1271 falls back to FoVer labels."""

    exp1271_path = tmp_path / "experiment_1271_triggered_certificate_extraction_sota_gguf.json"
    fover_path = tmp_path / "fover_corpus_v5.json"
    _write_json(exp1271_path, {"status": "blocked", "verification_certificates": []})
    _write_json(fover_path, _fover_payload())

    examples, source = exp.load_certificate_examples(
        exp1271_path=exp1271_path,
        fover_path=fover_path,
    )

    assert source == "fover_fallback"
    assert len(examples) == 4
    assert examples[0].verifier_result == "failed"
    assert examples[1].verifier_result == "passed"
    assert examples[0].target_decision == "repair"
    assert examples[1].target_decision == "accept"


def test_memory_table_keyed_replay_improves_score_for_req1274() -> None:
    """REQ-LEARN-1274-3/4: memory lookup improves replay decisions on matching keys."""

    examples = [
        exp.CertificateMemoryExample(
            example_id="build_fail",
            source="unit",
            question="What is 3 plus 4?",
            response="3 + 4 = 8. Therefore final answer: 8.",
            is_correct=False,
            constraint_pattern="arithmetic:addition",
            verifier_result="failed",
            repair_hint="recompute_arithmetic_result",
            target_decision="repair",
        ),
        exp.CertificateMemoryExample(
            example_id="build_pass",
            source="unit",
            question="What is 5 plus 6?",
            response="5 + 6 = 11. Therefore final answer: 11.",
            is_correct=True,
            constraint_pattern="arithmetic:addition",
            verifier_result="passed",
            repair_hint="accept_verified_answer",
            target_decision="accept",
        ),
        exp.CertificateMemoryExample(
            example_id="eval_fail",
            source="unit",
            question="What is 7 plus 8?",
            response="7 + 8 = 16. Therefore final answer: 16.",
            is_correct=False,
            constraint_pattern="arithmetic:addition",
            verifier_result="failed",
            repair_hint="recompute_arithmetic_result",
            target_decision="repair",
        ),
        exp.CertificateMemoryExample(
            example_id="eval_pass",
            source="unit",
            question="What is 9 plus 1?",
            response="9 + 1 = 10. Therefore final answer: 10.",
            is_correct=True,
            constraint_pattern="arithmetic:addition",
            verifier_result="passed",
            repair_hint="accept_verified_answer",
            target_decision="accept",
        ),
    ]

    result = exp.run_replay_evaluation(examples, build_fraction=0.5)

    assert result["memory_entries"] == 2
    assert result["before_score"] == 0.5
    assert result["after_score"] == 1.0
    assert result["self_learning_delta_overall"] == 0.5
    assert result["skill_graph_candidate_count"] == 2
    for candidate in result["skill_graph_candidates"]:
        assert {
            "contract",
            "evidence_count",
            "replay_success_rate",
            "demotion_condition",
        } <= candidate.keys()


def test_run_experiment_writes_complete_artifact_for_req1274(tmp_path: Path) -> None:
    """REQ-LEARN-1274-4/5: final artifact contains measured scores and validates."""

    exp1271_path = tmp_path / "exp1271.json"
    fover_path = tmp_path / "fover.json"
    output_path = tmp_path / "experiment_1274_online_self_learning_certificate_memory_v3.json"
    _write_json(exp1271_path, {"status": "blocked"})
    _write_json(fover_path, _fover_payload())

    artifact = exp.run_experiment(
        exp1271_path=exp1271_path,
        fover_path=fover_path,
        output_path=output_path,
        run_date="20260504",
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    exp.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["source"] == "fover_fallback"
    assert artifact["memory_entries"] >= 0
    assert artifact["self_learning_delta_overall"] == (
        artifact["after_score"] - artifact["before_score"]
    )
    assert artifact["honest_verdict"] in {
        "certificate_memory_replay_improved",
        "certificate_memory_replay_neutral",
        "certificate_memory_replay_regressed",
    }


def test_req1274_exp1271_and_validation_edges(tmp_path: Path) -> None:
    """REQ-LEARN-1274: Exp 1271 parsing, empty replay, and schema validation are strict."""

    exp1271_path = tmp_path / "exp1271.json"
    fover_path = tmp_path / "fover.json"
    _write_json(fover_path, _fover_payload())
    _write_json(
        exp1271_path,
        {
            "status": "complete",
            "verification_certificates": [
                {
                    "id": "cert_ok",
                    "prompt": "A proof with no arithmetic markers.",
                    "completion": "verified proof",
                    "verified": True,
                    "label": "unknown",
                },
                {
                    "example_id": "cert_bad",
                    "question": "Check this code contract.",
                    "answer": "violates the invariant",
                    "z3_verdict": "unsat",
                    "constraint_type": "code:invariant",
                },
            ],
        },
    )

    examples, source = exp.load_certificate_examples(
        exp1271_path=exp1271_path,
        fover_path=fover_path,
    )

    assert source == "exp1271"
    assert [example.example_id for example in examples] == ["cert_ok", "cert_bad"]
    assert examples[0].constraint_pattern == "arithmetic:general"
    assert examples[0].verifier_result == "passed"
    assert examples[1].repair_hint == "repair_constraint_violation"
    assert exp._read_json_if_exists(None) is None
    assert exp._certificate_rows(None) == []
    assert exp._certificate_rows({"status": "complete"}) == []
    assert exp._certificate_rows([{"label": "correct"}, "skip"]) == [{"label": "correct"}]
    assert exp._examples_from_fover([{"question": "q", "response": "r", "is_correct": True}, "skip"])[
        0
    ].source == "fover_fallback"
    assert exp.infer_constraint_pattern("half of 10", "") == "arithmetic:ratio"
    assert exp.infer_constraint_pattern("budget left over", "") == "arithmetic:balance"
    assert exp._normalise_verifier_result("mystery", default_correct=False) == "failed"
    assert exp._majority_decision(Counter({"repair": 2, "accept": 1})) == "repair"
    assert exp.baseline_decision(
        exp.CertificateMemoryExample(
            example_id="constant",
            source="unit",
            question="q",
            response="The answer is 42.",
            is_correct=False,
            constraint_pattern="arithmetic:general",
            verifier_result="failed",
            repair_hint="recompute_arithmetic_result",
            target_decision="repair",
        )
    ) == "repair"
    assert exp.memory_augmented_decision(examples[0], {}) == "accept"
    assert exp._score_decisions([], []) == 0.0
    assert exp.run_replay_evaluation(examples[:1])["n_replay_eval_examples"] == 0
    with pytest.raises(ValueError, match="build_fraction"):
        exp.split_examples(examples, build_fraction=1.0)

    memory = exp.build_memory_table(examples[:1])
    assert exp.build_skill_graph_candidates(memory, {}) == []
    assert exp.derive_honest_verdict(-0.1) == "certificate_memory_replay_regressed"
    assert exp.derive_honest_verdict(0.0) == "certificate_memory_replay_neutral"

    good = exp.run_experiment(
        exp1271_path=exp1271_path,
        fover_path=fover_path,
        output_path=tmp_path / "artifact.json",
        run_date="20260504",
    )
    for key, message in [
        ("experiment", "missing required fields"),
        ("status", "status must be complete"),
        ("source", "source must be exp1271"),
        ("self_learning_delta_overall", "after_score - before_score"),
        ("memory_entries", "non-negative"),
        ("skill_graph_candidate_count", "must match candidates"),
        ("skill_graph_candidates", "missing required fields"),
    ]:
        bad = dict(good)
        if key == "status":
            bad[key] = "blocked"
        elif key == "source":
            bad[key] = "unknown"
        elif key == "self_learning_delta_overall":
            bad[key] = 99.0
        elif key == "memory_entries":
            bad[key] = -1
        elif key == "skill_graph_candidate_count":
            bad[key] = int(bad[key]) + 1
        elif key == "skill_graph_candidates":
            bad[key] = [{"contract": "incomplete"}]
            bad["skill_graph_candidate_count"] = 1
        else:
            del bad[key]
        with pytest.raises(AssertionError, match=message):
            exp.validate_artifact(bad)

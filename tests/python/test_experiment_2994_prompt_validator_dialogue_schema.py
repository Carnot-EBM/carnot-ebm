"""Tests for Exp 2994 prompt-to-validator dialogue schema.

Spec refs: REQ-VERIFY-2994, SCENARIO-VERIFY-2994.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from carnot.eval import prompt_validator_dialogue_schema_v1 as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"
PROTOCOL_PATH = (
    REPO_ROOT / "openspec" / "change-proposals" / "prompt-validator-dialogue-schema-v1.md"
)
REQUIRED_FIELDS = {
    "prompt_validator_protocol_ready",
    "protocol_doc_path",
    "deterministic_harness_path",
    "n_validator_tree_fixtures",
    "exact_verifier_authority_preserved",
    "static_transition_representation_designed",
    "no_speed_claim_made",
    "validation_commands",
    "honest_verdict",
}


def test_req_verify_2994_spec_and_protocol_doc_anchor_stages() -> None:
    """REQ-VERIFY-2994: protocol is reviewable and OpenSpec anchored."""
    spec = SPEC_PATH.read_text(encoding="utf-8")
    protocol = PROTOCOL_PATH.read_text(encoding="utf-8")

    assert "REQ-VERIFY-2994" in spec
    assert "SCENARIO-VERIFY-2994" in spec
    assert exp.OUTPUT_FILENAME in spec
    assert "prompt constraints" in protocol.lower()
    assert "validator tree" in protocol.lower()
    assert "exact-check nodes" in protocol.lower()
    assert "feedback object" in protocol.lower()
    assert "rejection reasons" in protocol.lower()
    assert "no speed" in protocol.lower()
    assert exp.PROTOCOL_DOC_PATH.endswith("prompt-validator-dialogue-schema-v1.md")


def test_scenario_verify_2994_compiles_three_prompt_fixtures_to_exact_trees() -> None:
    """SCENARIO-VERIFY-2994: fixed prompts compile and good/bad candidates separate."""
    fixtures = exp.build_validator_tree_fixtures()

    assert len(fixtures) == 3
    assert {fixture["fixture_id"] for fixture in fixtures} == {
        "json-final-answer-confidence",
        "python-normalize-slug-ast",
        "z3-linear-integer-assignment",
    }

    for fixture in fixtures:
        compiled = exp.compile_prompt_to_validator_tree(
            fixture["prompt"],
            constraint_id=fixture["fixture_id"],
        )

        assert compiled["compiled"] is True
        assert compiled["rejection_reasons"] == []
        assert compiled["validator_tree"]["root"]["op"] == "all"
        assert {
            node["authority"] for node in compiled["validator_tree"]["nodes"]
        } <= exp.EXACT_AUTHORITIES

        good = exp.evaluate_validator_tree(compiled["validator_tree"], fixture["known_good"])
        bad = exp.evaluate_validator_tree(compiled["validator_tree"], fixture["known_bad"])

        assert good["accepted"] is True, fixture["fixture_id"]
        assert good["llm_judge_used"] is False
        assert good["failing_node_ids"] == []
        assert bad["accepted"] is False, fixture["fixture_id"]
        assert bad["llm_judge_used"] is False
        assert bad["failing_node_ids"]
        assert bad["rejection_reasons"]


def test_req_verify_2994_feedback_reasons_fail_closed() -> None:
    """REQ-VERIFY-2994: unsupported prompts and malformed candidates reject by reason."""
    unsupported = exp.compile_prompt_to_validator_tree(
        "Write a pleasant haiku and decide whether it feels correct.",
        constraint_id="unsupported",
    )
    assert unsupported == {
        "constraint_id": "unsupported",
        "compiled": False,
        "validator_tree": None,
        "rejection_reasons": ["unsupported_prompt_pattern"],
    }

    fixtures = {
        fixture["fixture_id"]: exp.compile_prompt_to_validator_tree(
            fixture["prompt"],
            constraint_id=fixture["fixture_id"],
        )["validator_tree"]
        for fixture in exp.build_validator_tree_fixtures()
    }

    json_feedback = exp.evaluate_validator_tree(
        fixtures["json-final-answer-confidence"],
        '{"final_answer": "SAFE"}',
    )
    json_missing_answer_feedback = exp.evaluate_validator_tree(
        fixtures["json-final-answer-confidence"],
        '{"confidence": 0.8}',
    )
    json_wrong_answer_feedback = exp.evaluate_validator_tree(
        fixtures["json-final-answer-confidence"],
        '{"final_answer": "UNSAFE", "confidence": 0.8}',
    )
    json_parse_feedback = exp.evaluate_validator_tree(
        fixtures["json-final-answer-confidence"],
        "{",
    )
    json_non_object_feedback = exp.evaluate_validator_tree(
        fixtures["json-final-answer-confidence"],
        "[]",
    )
    python_signature_feedback = exp.evaluate_validator_tree(
        fixtures["python-normalize-slug-ast"],
        "def normalize_slug(left, right):\n    return left\n",
    )
    python_missing_function_feedback = exp.evaluate_validator_tree(
        fixtures["python-normalize-slug-ast"],
        "def other(text):\n    return text\n",
    )
    python_import_feedback = exp.evaluate_validator_tree(
        fixtures["python-normalize-slug-ast"],
        "import os\ndef normalize_slug(text):\n    return text\n",
    )
    python_syntax_feedback = exp.evaluate_validator_tree(
        fixtures["python-normalize-slug-ast"],
        "def normalize_slug(:\n",
    )
    z3_unavailable_feedback = exp.evaluate_validator_tree(
        fixtures["z3-linear-integer-assignment"],
        '{"x": 6, "y": 4}',
        z3_module=None,
    )
    z3_json_feedback = exp.evaluate_validator_tree(
        fixtures["z3-linear-integer-assignment"],
        "{",
    )
    z3_missing_assignment_feedback = exp.evaluate_validator_tree(
        fixtures["z3-linear-integer-assignment"],
        '{"x": 6}',
    )

    assert "missing_required_field" in json_feedback["rejection_reasons"]
    assert "missing_required_field" in json_missing_answer_feedback["rejection_reasons"]
    assert "field_value_mismatch" in json_wrong_answer_feedback["rejection_reasons"]
    assert "json_parse_error" in json_parse_feedback["rejection_reasons"]
    assert "json_parse_error" in json_non_object_feedback["rejection_reasons"]
    assert "function_signature_mismatch" in python_signature_feedback["rejection_reasons"]
    assert "function_signature_mismatch" in python_missing_function_feedback["rejection_reasons"]
    assert "import_statement_disallowed" in python_import_feedback["rejection_reasons"]
    assert "python_syntax_error" in python_syntax_feedback["rejection_reasons"]
    assert "z3_unavailable" in z3_unavailable_feedback["rejection_reasons"]
    assert "json_parse_error" in z3_json_feedback["rejection_reasons"]
    assert "missing_required_field" in z3_missing_assignment_feedback["rejection_reasons"]

    assert (
        exp._exact_authority_preserved(
            [
                {
                    "known_good_feedback": {"llm_judge_used": True},
                    "known_bad_feedback": {"llm_judge_used": False},
                    "validator_tree": {"nodes": []},
                }
            ]
        )
        is False
    )
    assert (
        exp._exact_authority_preserved(
            [
                {
                    "known_good_feedback": {"llm_judge_used": False},
                    "known_bad_feedback": {"llm_judge_used": False},
                    "validator_tree": {"nodes": [{"authority": "llm_judge"}]},
                }
            ]
        )
        is False
    )


def test_scenario_verify_2994_runner_writes_ready_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-2994: harness writes required artifact without live inference."""
    output_path = tmp_path / exp.OUTPUT_FILENAME

    artifact = exp.write_artifact(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            output_path=output_path,
            validation_commands=["focused pytest"],
            started_at=10.0,
            clock=lambda: 10.25,
        )
    )
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved == artifact
    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["prompt_validator_protocol_ready"] is True
    assert artifact["protocol_doc_path"] == exp.PROTOCOL_DOC_PATH
    assert artifact["deterministic_harness_path"] == exp.DETERMINISTIC_HARNESS_PATH
    assert artifact["n_validator_tree_fixtures"] == 3
    assert artifact["exact_verifier_authority_preserved"] is True
    assert artifact["static_transition_representation_designed"] is True
    assert artifact["no_speed_claim_made"] is True
    assert artifact["validation_commands"] == ["focused pytest"]
    assert artifact["duration_s"] == pytest.approx(0.25)
    assert artifact["llm_inference_run"] is False
    assert artifact["unsupported_prompt_rejection"]["rejection_reasons"] == [
        "unsupported_prompt_pattern"
    ]
    assert artifact["static_transition_representation"]["speed_claim"] is None
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["source_artifacts"]["exp2979"]["present"] is False
    exp.validate_artifact(artifact)


def test_req_verify_2994_validation_rejects_unready_or_claimed_speed(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-2994: artifact validation enforces authority and no-speed gates."""
    artifact = exp.build_artifact(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            validation_commands=["focused pytest"],
            started_at=1.0,
            clock=lambda: 1.5,
        )
    )
    exp.validate_artifact(artifact)

    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact({"honest_verdict": "complete: x"})
    with pytest.raises(ValueError, match="prompt_validator_protocol_ready"):
        exp.validate_artifact(artifact | {"prompt_validator_protocol_ready": False})
    with pytest.raises(ValueError, match="exact_verifier_authority_preserved"):
        exp.validate_artifact(artifact | {"exact_verifier_authority_preserved": False})
    with pytest.raises(ValueError, match="static_transition_representation_designed"):
        exp.validate_artifact(artifact | {"static_transition_representation_designed": False})
    with pytest.raises(ValueError, match="no_speed_claim_made"):
        exp.validate_artifact(artifact | {"no_speed_claim_made": False})
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(artifact | {"honest_verdict": "flagged: fixture"})


def test_req_verify_2994_cli_entrypoint_writes_selected_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-2994: deterministic harness can be run as a local command."""
    output_path = tmp_path / "cli-artifact.json"

    rc = exp.main(
        [
            "--output",
            str(output_path),
            "--validation-command",
            "cli pytest",
        ]
    )
    artifact = json.loads(output_path.read_text(encoding="utf-8"))

    assert rc == 0
    assert artifact["prompt_validator_protocol_ready"] is True
    assert artifact["validation_commands"] == ["cli pytest"]

    argv_output_path = tmp_path / "sys-argv-artifact.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prompt_validator_dialogue_schema_v1",
            "--output",
            str(argv_output_path),
            "--validation-command",
            "sys argv pytest",
        ],
    )

    assert exp.main() == 0
    sys_argv_artifact = json.loads(argv_output_path.read_text(encoding="utf-8"))
    assert sys_argv_artifact["validation_commands"] == ["sys argv pytest"]

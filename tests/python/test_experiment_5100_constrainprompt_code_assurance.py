"""Tests for Exp 5100 ConstrainPrompt code-assurance exact checks.

Spec refs: REQ-VERIFY-5100, SCENARIO-VERIFY-5100.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5100_constrainprompt_code_assurance as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
ARTIFACT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_verify_5100_spec_declares_exact_code_assurance_contract() -> None:
    """REQ-VERIFY-5100: OpenSpec anchors paths, fields, verdicts, and models."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-5100",
        "SCENARIO-VERIFY-5100",
        "python/carnot/experiment_5100_constrainprompt_code_assurance.py",
        "results/experiment_5100_constrainprompt_code_assurance_v468.json",
        "python_json_logical_tree",
        "success_constrainprompt_code_assurance_exact_checks_passed",
        "complete_constrainprompt_assurance_partial_constraints_only",
    ):
        assert marker in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for model_id in mod.MANDATED_MODEL_IDS:
        assert model_id in spec


def test_req_verify_5100_compiles_five_to_ten_prompt_constraints() -> None:
    """REQ-VERIFY-5100: fixed prompt constraints become executable tree nodes."""

    constraints = mod.build_prompt_constraints()
    compiled = mod.compile_constraints(constraints)

    assert 5 <= len(constraints) <= 10
    assert compiled.rejected_constraints == []
    assert len(compiled.executable_constraints) == len(constraints)
    assert compiled.evaluation_tree["backend"] == mod.EXACT_CHECKER_BACKEND
    assert compiled.evaluation_tree["root"]["op"] == "all"
    assert {
        node["authority"] for node in compiled.evaluation_tree["nodes"]
    } == {"python_exact_check"}
    assert {
        constraint.constraint_id for constraint in constraints
    } == {
        "required_fields_and_schema",
        "verdict_enum",
        "confidence_enum_and_mapping",
        "verdict_evidence_consistency",
        "evidence_refs_for_decisive_verdicts",
        "claim_id_format",
        "checker_backend_exact",
        "duration_bounds",
        "rationale_bounded",
    }

    unsupported = mod.PromptConstraint(
        constraint_id="unsupported_sentiment_judge",
        prompt="The verdict must feel persuasive to the reader.",
        check_name="semantic_sentiment_judge",
    )
    partial = mod.compile_constraints((*constraints, unsupported))
    assert partial.rejected_constraints == [
        {
            "constraint_id": "unsupported_sentiment_judge",
            "reason": "no_allowlisted_executable_check",
            "prompt": unsupported.prompt,
        }
    ]


def test_scenario_verify_5100_positive_negative_and_adversarial_fixtures_split() -> None:
    """SCENARIO-VERIFY-5100: exact checks accept positives and reject controls."""

    compiled = mod.compile_constraints(mod.build_prompt_constraints())
    fixture_sets = mod.build_fixture_sets()

    assert {row["fixture_id"] for row in fixture_sets["positive"]} == {
        "accept_solver_verified",
        "reject_schema_missing",
    }
    assert len(fixture_sets["negative"]) >= len(compiled.executable_constraints)
    assert len(fixture_sets["adversarial"]) >= 3

    for fixture in fixture_sets["positive"]:
        feedback = mod.evaluate_candidate(fixture["candidate"], compiled)
        assert feedback["accepted"] is True, fixture["fixture_id"]
        assert feedback["failing_constraints"] == []
        assert feedback["llm_judge_used"] is False

    for group in ("negative", "adversarial"):
        for fixture in fixture_sets[group]:
            feedback = mod.evaluate_candidate(fixture["candidate"], compiled)
            assert feedback["accepted"] is False, fixture["fixture_id"]
            assert feedback["failing_constraints"], fixture["fixture_id"]
            assert feedback["llm_judge_used"] is False

    malformed = mod.evaluate_candidate("{", compiled)
    assert malformed["accepted"] is False
    assert malformed["failing_constraints"] == ["json_parse_error"]
    assert malformed["constraint_results"][0]["reason"] == "json_parse_error"


def test_req_verify_5100_edge_rejection_reasons_are_named() -> None:
    """REQ-VERIFY-5100: checker edge cases reject with exact, auditable reasons."""

    compiled = mod.compile_constraints(mod.build_prompt_constraints())
    base = mod.build_fixture_sets()["positive"][0]["candidate"]

    cases = {
        "confidence_not_allowed": {"confidence": "certain"},
        "abstain_confidence_not_low": {
            "verdict": "abstain",
            "confidence": "medium",
            "evidence_label": "unsupported_prompt",
            "evidence_refs": [],
        },
        "evidence_refs_not_list": {"evidence_refs": "evidence://solver/z3/x"},
        "evidence_ref_invalid": {"evidence_refs": ["http://example.invalid/evidence"]},
        "duration_not_number": {"duration_s": True},
        "rationale_not_string": {"rationale": None},
    }

    for reason, updates in cases.items():
        candidate = dict(base)
        candidate.update(updates)
        feedback = mod.evaluate_candidate(candidate, compiled)
        assert feedback["accepted"] is False
        assert reason in feedback["rejection_reasons"]


def test_req_verify_5100_preconditions_and_model_specs_are_honest() -> None:
    """REQ-VERIFY-5100: preconditions record schema, prompt constraints, backend, and LLM use."""

    preconditions = mod.build_preconditions()
    model_specs = mod.build_model_specs(llm_invoked=False)
    invoked_specs = mod.build_model_specs(llm_invoked=True)

    assert preconditions["schema_path"] == mod.SCHEMA_PATH
    assert preconditions["schema_name"] == mod.SCHEMA_NAME
    assert preconditions["prompt_constraints"]
    assert preconditions["parser_backend"] == mod.PARSER_BACKEND
    assert preconditions["checker_backend"] == mod.EXACT_CHECKER_BACKEND
    assert preconditions["llm_invoked"] is False
    assert [row["hf_id"] for row in model_specs] == list(mod.MANDATED_MODEL_IDS)
    assert {row["invocation_status"] for row in model_specs} == {
        "not_invoked_reference_only"
    }
    assert {row["invocation_status"] for row in invoked_specs} == {"required_if_invoked"}


def test_req_verify_5100_artifact_fields_principles_and_success_verdict(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-5100: artifact emits the required principle-annotated schema."""

    artifact = mod.write_artifact(
        root=tmp_path,
        output_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        validation_commands=["focused pytest"],
    )
    loaded = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert loaded == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(artifact["field_principles"])
    assert artifact["honest_verdict"].startswith(
        "success_constrainprompt_code_assurance_exact_checks_passed"
    )
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["preconditions_checked"]["schema_path"] == mod.SCHEMA_PATH
    assert artifact["schema_name"] == mod.SCHEMA_NAME
    assert artifact["constraints_total"] == 9
    assert artifact["executable_constraints_total"] == 9
    assert artifact["positive_tests_passed"] is True
    assert artifact["negative_tests_passed"] is True
    assert artifact["adversarial_tests_passed"] is True
    assert artifact["rejected_constraints"] == []
    assert artifact["llm_invoked"] is False
    assert artifact["exact_checker_backend"] == mod.EXACT_CHECKER_BACKEND
    assert artifact["flagged_adversarial"] is False
    assert {row["hf_id"] for row in artifact["model_specs"]} == set(mod.MANDATED_MODEL_IDS)
    assert {row["group"] for row in artifact["fixture_results"]} == {
        "positive",
        "negative",
        "adversarial",
    }
    assert len(artifact["reproducibility_checksum"]) == 64
    mod.validate_artifact(artifact)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("honest_verdict", "optimistic_pass", "honest_verdict"),
        ("duration_s", -0.01, "duration_s"),
        ("inference_substrate", "live_llm_inference", "live_llm_inference"),
        ("schema_name", "other_schema", "schema_name"),
        ("constraints_total", 4, "constraints_total"),
        ("executable_constraints_total", 8, "executable_constraints_total"),
        ("positive_tests_passed", False, "positive_tests_passed"),
        ("negative_tests_passed", False, "negative_tests_passed"),
        ("adversarial_tests_passed", False, "adversarial_tests_passed"),
        ("llm_invoked", True, "llm_invoked"),
        ("exact_checker_backend", "llm_judge", "exact_checker_backend"),
        ("flagged_adversarial", True, "flagged_adversarial"),
    ],
)
def test_req_verify_5100_validate_artifact_rejects_schema_violations(
    field: str,
    value: Any,
    message: str,
) -> None:
    """REQ-VERIFY-5100: malformed terminal artifacts fail closed."""

    artifact = mod.run()
    artifact[field] = value

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(artifact)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda artifact: artifact.pop("model_specs"), "missing required"),
        (
            lambda artifact: artifact.update(
                {"field_principles": {"honest_verdict": {"principle": "x"}}}
            ),
            "field_principles",
        ),
        (
            lambda artifact: artifact.update({"preconditions_checked": {"llm_invoked": False}}),
            "preconditions_checked",
        ),
        (
            lambda artifact: artifact.update(
                {"model_specs": artifact["model_specs"][:-1]}
            ),
            "model_specs",
        ),
        (lambda artifact: artifact.update({"fixture_summary": {}}), "fixture_summary"),
        (
            lambda artifact: artifact.update(
                {
                    "rejected_constraints": [
                        {
                            "constraint_id": "not_executable",
                            "reason": "manual_review",
                            "prompt": "unknown",
                        }
                    ]
                }
            ),
            "rejected_constraints",
        ),
    ],
)
def test_req_verify_5100_validate_artifact_rejects_consistency_violations(
    mutator: Any,
    message: str,
) -> None:
    """REQ-VERIFY-5100: internally inconsistent artifacts fail closed."""

    artifact = mod.run()
    mutator(artifact)

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(artifact)


def test_req_verify_5100_main_writes_default_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-5100: CLI entrypoint writes the configured result path."""

    monkeypatch.setenv("CARNOT_EXP5100_ROOT", str(tmp_path))

    assert mod.main() == 0
    payload = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert payload["honest_verdict"].startswith(
        "success_constrainprompt_code_assurance_exact_checks_passed"
    )
    mod.validate_artifact(payload)


def test_deliverable_file_validates_for_req_verify_5100() -> None:
    """SCENARIO-VERIFY-5100: committed deliverable JSON satisfies the schema."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["result_path"] == mod.RESULT_RELATIVE_PATH
    assert artifact["honest_verdict"].startswith(
        "success_constrainprompt_code_assurance_exact_checks_passed"
    )
    assert artifact["llm_invoked"] is False

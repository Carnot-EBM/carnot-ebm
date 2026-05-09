"""Tests for Exp 1588 bounded instruction-to-constraint DSL.

Spec: REQ-VERIFY-1588, SCENARIO-VERIFY-1588.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.verifiers import dsl


def test_req_verify_1588_parses_supported_nl_constraints_into_schema() -> None:
    """REQ-VERIFY-1588: bounded NL patterns become schema-valid DSL constraints."""

    pack = dsl.parse_instruction_constraints(
        'Respond in JSON with keys answer and confidence. Include "approved". '
        'Do not mention "secret". Use at most 12 words.'
    )
    payload = pack.to_dict()

    assert payload["schema_version"] == dsl.DSL_SCHEMA_VERSION
    assert dsl.validate_constraint_pack(payload) == []
    assert [constraint["op"] for constraint in payload["constraints"]] == [
        "json_object",
        "json_has_keys",
        "contains",
        "not_contains",
        "max_words",
    ]
    assert payload["constraints"][1]["value"] == ["answer", "confidence"]
    assert payload["constraints"][2]["value"] == "approved"
    assert payload["constraints"][3]["value"] == "secret"
    assert payload["constraints"][4]["value"] == 12


def test_req_verify_1588_compiles_python_validator_with_specific_failures() -> None:
    """REQ-VERIFY-1588: compiled Python validators accept good rows and explain failures."""

    validator = dsl.compile_instruction_validator(
        'Respond in JSON with keys answer and confidence. Include "approved". '
        'Do not mention "secret". Use at most 12 words.'
    )
    good = '{"answer": "approved", "confidence": "high"}'
    bad = '{"answer": "secret", "extra": true}'

    good_result = validator.validate(good)
    bad_result = validator.validate(bad)

    assert good_result.accepted is True
    assert good_result.failure_ids == []
    assert bad_result.accepted is False
    assert {"c003-contains", "c004-not_contains"} <= set(bad_result.failure_ids)
    assert "c002-json_has_keys" in bad_result.failure_ids
    assert validator(good) is True
    assert validator(bad) is False


def test_req_verify_1588_schema_and_compiler_fail_closed_for_unsafe_or_unbounded_input() -> None:
    """REQ-VERIFY-1588: unsupported operators, unsafe text, and bounds fail closed."""

    payload = {
        "schema_version": dsl.DSL_SCHEMA_VERSION,
        "instruction": "bad",
        "constraints": [{"id": "c001", "op": "python_eval", "field": "text", "value": "x"}],
    }

    assert dsl.validate_constraint_pack(payload) == ["constraint[0].op unsupported:python_eval"]
    with pytest.raises(dsl.ConstraintDslError, match="unsupported"):
        dsl.constraint_pack_from_dict(payload)
    with pytest.raises(dsl.ConstraintDslError, match="unsafe"):
        dsl.parse_instruction_constraints('Include "__import__".')
    with pytest.raises(dsl.ConstraintDslError, match="instruction too long"):
        dsl.parse_instruction_constraints("word " * 600)
    with pytest.raises(dsl.ConstraintDslError, match="too many constraints"):
        dsl.parse_instruction_constraints(
            " ".join(f'Include "token{i}".' for i in range(dsl.MAX_CONSTRAINTS + 1))
        )
    with pytest.raises(dsl.ConstraintDslError, match="no supported constraints"):
        dsl.compile_instruction_validator("Answer carefully.")


def test_req_verify_1588_schema_edges_and_typed_round_trip() -> None:
    """REQ-VERIFY-1588: raw DSL schema errors and typed-pack compilation are explicit."""

    bad_header = {"schema_version": "bad", "instruction": 123, "constraints": "bad"}
    too_many = {
        "schema_version": dsl.DSL_SCHEMA_VERSION,
        "instruction": "many",
        "constraints": [{"id": f"c{i}", "op": "json_object"} for i in range(9)],
    }
    bad_values = [
        {"id": "c1", "op": "contains", "value": 7},
        {"id": "c2", "op": "max_words", "value": -1},
        {"id": "c3", "op": "json_has_keys", "value": [""]},
        {"id": "c4", "op": "enum", "value": ["yes"]},
    ]

    assert dsl.validate_constraint_pack(bad_header) == [
        "schema_version unsupported",
        "instruction must be string",
        "constraints must be list",
    ]
    assert dsl.validate_constraint_pack(too_many)[0] == "constraints too many:9>8"
    assert dsl.validate_constraint_pack(
        {
            "schema_version": dsl.DSL_SCHEMA_VERSION,
            "instruction": "bad",
            "constraints": [None],
        }
    ) == ["constraint[0] must be object"]
    assert dsl.validate_constraint_pack(
        {
            "schema_version": dsl.DSL_SCHEMA_VERSION,
            "instruction": "bad",
            "constraints": bad_values,
        }
    ) == [
        "constraint[0].value must be string",
        "constraint[1].value must be nonnegative integer",
        "constraint[2].value must be string list",
        "constraint[3].value must be two-or-more string list",
    ]

    pack = dsl.parse_instruction_constraints('Include "alpha".')
    round_tripped = dsl.constraint_pack_from_dict(pack.to_dict())
    assert round_tripped == pack
    assert dsl.compile_constraint_pack(pack.to_dict()).validate("alpha").accepted is True
    with pytest.raises(dsl.ConstraintDslError, match="value must be string"):
        dsl.compile_constraint_pack(
            dsl.ConstraintPack(
                instruction="bad typed",
                constraints=(dsl.ConstraintSpec(id="c001-contains", op="contains", value=7),),
            )
        )
    with pytest.raises(dsl.ConstraintDslError, match="instruction must be string"):
        dsl.parse_instruction_constraints(123)  # type: ignore[arg-type]


def test_req_verify_1588_runtime_edges_cover_json_enum_and_count_helpers() -> None:
    """REQ-VERIFY-1588: runtime validator branches stay deterministic on edge shapes."""

    json_validator = dsl.compile_instruction_validator("Respond in JSON.")
    assert json_validator.validate("[]").failure_ids == ["c001-json_object"]
    assert "json_decode_error" in str(json_validator.validate("{").issues[0].observed)

    enum_validator = dsl.compile_instruction_validator('Answer must be one of "yes", "no".')
    assert enum_validator.validate('"yes"').accepted is True
    assert enum_validator.validate('{"answer": "no"}').accepted is True
    assert enum_validator.validate("5").accepted is False
    assert dsl.parse_instruction_constraints('Answer must be one of "yes".').constraints == ()

    assert dsl._word_count("two-word answer") == 2
    assert dsl._bullet_count("1. first\n2) second\nplain") == 2
    unsupported = dsl._evaluate_constraint(
        dsl.ConstraintSpec(id="c999", op="unsupported", value=True),
        "text",
        None,
        None,
    )
    assert unsupported is not None
    assert unsupported.observed == "unsupported"


def test_req_verify_1588_exports_pysat_compatible_hard_conjunction() -> None:
    """REQ-VERIFY-1588: compiled packs expose deterministic PySAT-compatible CNF."""

    pack = dsl.parse_instruction_constraints(
        'Answer must be one of "yes", "no". Use exactly 2 bullet points.'
    )
    validator = dsl.compile_constraint_pack(pack)
    cnf = validator.pysat_problem.to_dict()

    assert validator.pysat_problem.backend == "pysat-compatible-cnf"
    assert cnf["variables"] == {"c001-enum": 1, "c002-exact_bullets": 2}
    assert cnf["clauses"] == [[1], [2]]
    assert cnf["description"] == "hard conjunction of instruction constraints"
    assert validator.validate("- yes\n- no").accepted is False
    assert validator.validate("- yes\n- yes").accepted is False
    assert validator.validate("yes").accepted is False

    bullet_only = dsl.compile_instruction_validator("Use exactly 2 bullet points.")
    assert bullet_only.validate("- first\n- second").accepted is True
    assert bullet_only.validate("- only one").failure_ids == ["c001-exact_bullets"]


def test_scenario_verify_1588_writes_complete_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1588: terminal artifact records metrics and schema fields."""

    output_path = tmp_path / "experiment_1588_nsvif_dsl.json"
    artifact = dsl.write_experiment_artifact(
        output_path=output_path,
        tests_run=[".venv/bin/pytest tests/python/test_experiment_1588_nsvif_dsl.py -q"],
    )
    persisted = json.loads(output_path.read_text(encoding="utf-8"))

    assert artifact == persisted
    assert set(dsl.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["experiment_id"] == "experiment_1588_nsvif_dsl"
    assert artifact["dsl_schema_version"] == dsl.DSL_SCHEMA_VERSION
    assert artifact["instructions_tested"] == 4
    assert artifact["constraints_extracted"] >= 8
    assert artifact["validators_compiled"] == 4
    assert artifact["pysat_cnf_compiled"] == 4
    assert artifact["known_good_pass_rate"] == pytest.approx(1.0)
    assert artifact["known_bad_reject_rate"] == pytest.approx(1.0)
    assert artifact["false_accept_rate"] == pytest.approx(0.0)
    assert artifact["arbitrary_code_execution_path_introduced"] is False
    assert artifact["honest_verdict"].startswith("complete:")

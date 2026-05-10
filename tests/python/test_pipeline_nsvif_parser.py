"""Tests for the product NSVIF instruction-to-constraint parser.

Spec: REQ-VERIFY-1666, SCENARIO-VERIFY-1666.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.pipeline import nsvif_parser as mod


def test_req_verify_1666_parses_prompt_to_dsl_and_carnot_constraints() -> None:
    """REQ-VERIFY-1666: NL prompts become bounded DSL and Carnot constraints."""

    parsed = mod.parse_nsvif_prompt(
        'Respond in JSON with keys answer and confidence. Include "approved". '
        'Do not mention "secret". Use at most 12 words.'
    )

    assert parsed["parser_success"] is True
    assert parsed["dsl_pack"]["schema_version"] == mod.dsl.DSL_SCHEMA_VERSION
    assert [item["op"] for item in parsed["dsl_pack"]["constraints"]] == [
        "json_object",
        "json_has_keys",
        "contains",
        "not_contains",
        "max_words",
    ]
    assert [item["metadata"]["nsvif_operator"] for item in parsed["carnot_constraints"]] == [
        "json_object",
        "json_has_keys",
        "contains",
        "not_contains",
        "max_words",
    ]
    assert parsed["carnot_constraints"][0]["constraint_type"] == "instruction_constraint"
    assert parsed["carnot_constraints"][0]["metadata"]["spec_traces"] == mod.SPEC_TRACES


def test_req_verify_1666_compiles_python_pysat_and_z3_validators() -> None:
    """REQ-VERIFY-1666: parsed constraints compile to all local validator backends."""

    case = mod.default_model_prompt_cases()[0]
    result = mod.evaluate_prompt_case(case)

    assert result["parser_success"] is True
    assert result["validators_compiled"] is True
    assert result["compiled_backends"] == ["python", "pysat_cnf", "z3"]
    assert result["python_known_good"]["accepted"] is True
    assert result["python_known_bad"]["accepted"] is False
    assert result["z3_known_good"]["accepted"] is True
    assert result["z3_known_good"]["sat_status"] == "sat"
    assert result["z3_known_bad"]["accepted"] is False
    assert result["z3_known_bad"]["sat_status"] == "unsat"
    assert result["pysat_problem"]["clauses"] == [[1], [2], [3], [4], [5]]
    assert result["z3_problem"]["assertions"] == [
        "c001_json_object",
        "c002_json_has_keys",
        "c003_contains",
        "c004_not_contains",
        "c005_max_words",
    ]


def test_req_verify_1666_fails_closed_for_unsafe_or_unsupported_prompts() -> None:
    """REQ-VERIFY-1666: unsafe and unsupported prompts never compile validators."""

    unsafe = mod.evaluate_prompt_case(
        {
            "case_id": "unsafe",
            "model_hf_id": mod.MODEL_SPECS[0],
            "prompt": 'Include "__import__".',
            "known_good": "__import__",
            "known_bad": "safe",
        }
    )
    unsupported = mod.evaluate_prompt_case(
        {
            "case_id": "unsupported",
            "model_hf_id": mod.MODEL_SPECS[1],
            "prompt": "Answer carefully.",
            "known_good": "careful",
            "known_bad": "bad",
        }
    )

    assert unsafe["parser_success"] is False
    assert unsafe["validators_compiled"] is False
    assert unsafe["error"].startswith("unsafe token:")
    assert unsupported["parser_success"] is False
    assert unsupported["validators_compiled"] is False
    assert unsupported["error"] == "no supported constraints"

    artifact = mod.build_artifact(cases=[unsupported], tests_run=[])
    mod.validate_artifact(artifact)
    assert artifact["status"] == "partial"
    assert artifact["compilation_rate"] == pytest.approx(0.0)


def test_req_verify_1666_backend_edges_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-1666: backend fallback and compile errors remain rejecting."""

    pack = mod.dsl.parse_instruction_constraints("Use exactly 1 bullet point.")
    bundle = mod.compile_nsvif_validators(pack)
    monkeypatch.setattr(mod, "z3_backend_available", lambda: False)

    fallback = bundle.z3_validator.validate("plain")

    assert fallback.accepted is False
    assert fallback.sat_status == "unsat"
    assert fallback.z3_backend_available is False

    def raise_compile_error(_pack: object) -> object:
        raise mod.dsl.ConstraintDslError("backend unavailable")

    monkeypatch.setattr(mod, "compile_nsvif_validators", raise_compile_error)
    failed = mod.evaluate_prompt_case(mod.default_model_prompt_cases()[0])

    assert failed["parser_success"] is False
    assert failed["validators_compiled"] is False
    assert failed["error"] == "backend unavailable"


def test_scenario_verify_1666_builds_complete_zero_false_accept_artifact() -> None:
    """SCENARIO-VERIFY-1666: bounded model rows compile with zero false accepts."""

    artifact = mod.build_artifact(
        cases=mod.default_model_prompt_cases(),
        tests_run=[".venv/bin/pytest tests/python/test_pipeline_nsvif_parser.py -q"],
    )

    mod.validate_artifact(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["spec_traces"] == mod.SPEC_TRACES
    assert artifact["model_specs"] == list(mod.MODEL_SPECS)
    assert artifact["cases_attempted"] == 2
    assert artifact["validators_compiled"] == 2
    assert artifact["python_validators_compiled"] == 2
    assert artifact["pysat_validators_compiled"] == 2
    assert artifact["z3_validators_compiled"] == 2
    assert artifact["false_accepts"] == 0
    assert artifact["compilation_rate"] == pytest.approx(1.0)
    assert artifact["known_good_pass_rate"] == pytest.approx(1.0)
    assert artifact["known_bad_reject_rate"] == pytest.approx(1.0)
    assert artifact["false_accept_rate"] == pytest.approx(0.0)
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_verify_1666_run_experiment_writes_required_json(tmp_path: Path) -> None:
    """REQ-VERIFY-1666: run_experiment writes experiment_1666_nsvif.json."""

    output_path = tmp_path / "results" / "experiment_1666_nsvif.json"

    artifact = mod.run_experiment(
        output_path=output_path,
        cases=mod.default_model_prompt_cases(),
        tests_run=["focused"],
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    assert artifact["artifact_path"] == str(output_path)
    assert artifact["tests_run"] == ["focused"]

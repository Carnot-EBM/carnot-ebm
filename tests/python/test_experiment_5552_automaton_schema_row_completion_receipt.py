"""Tests for Exp5552 automaton/schema row-completion receipt.

Spec refs: REQ-VERIFY-5552, SCENARIO-VERIFY-5552.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5512_structured_output_positive_control as positive
from carnot import experiment_5539_gram2token_grammar_table_preflight as gate5539
from carnot import experiment_5540_sota_hard_soft_live_panel_v3 as panel5540
from carnot import experiment_5552_automaton_schema_row_completion_receipt as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5552_automaton_schema_row_completion_receipt.py")


def _clean_grammar_gate() -> dict:
    return {
        "grammar_table_preflight_ready": True,
        "backend_available": True,
        "llm_invoked": False,
        "no_model_specs_required": True,
        "decoding_speedup_claim": False,
        "selected_backend": "llama_cpp_gbnf",
        "schema_hash": positive.sha256_json(positive.candidate_schema()),
        "valid_fixture_acceptance_rate": 1.0,
        "invalid_fixture_rejection_rate": 1.0,
        "grammar_backend_candidates": [
            {
                "name": "llama_cpp_gbnf",
                "available": True,
                "grammar_compiled": True,
                "constrained_generation": True,
                "table_exposed": False,
            }
        ],
        "unsupported_schema_features": ["llama_cpp_token_transition_table_not_exposed"],
        "inference_substrate": gate5539.INFERENCE_SUBSTRATE,
        "research_conductor_modified": False,
        "honest_verdict": "complete: unit grammar gate clean",
    }


def _panel_stub(model_ids: list[str]) -> dict:
    return {
        "models_attempted": model_ids,
        "panel_rows": [],
        "rows_requested": len(model_ids) * len(positive.build_fixture_candidate_payloads()),
        "inference_substrate": panel5540.INFERENCE_SUBSTRATE,
        "research_conductor_modified": False,
        "honest_verdict": "complete: unit panel stub",
    }


def _records(model_id: str, payloads: list[dict]) -> list[dict]:
    return [
        {
            "model_hf_id": model_id,
            "parsed_payload": payload,
            "production_mode": "grammar_masking",
        }
        for payload in payloads
    ]


def test_req_verify_5552_spec_declares_automaton_receipt_contract() -> None:
    """REQ-VERIFY-5552: OpenSpec anchors the no-LLM row-completion receipt."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5552") : spec.index("### REQ-VERIFY-5541")]

    assert "SCENARIO-VERIFY-5552" in section
    assert str(mod.RESULT_RELATIVE_PATH) in section
    assert positive.CANDIDATE_SCHEMA_VERSION in section
    assert mod.INFERENCE_SUBSTRATE in section
    assert "`llm_invoked` SHALL be `false`" in section
    assert "SHALL NOT invoke an LLM" in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_verify_5552_complete_rows_open_receipt(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5552: complete schema-valid rows reach a terminal state."""

    fixture = positive.load_fixture_artifact()["fixture"]
    payloads = positive.build_fixture_candidate_payloads(fixture)
    model_id = "unit/model"
    result_path = tmp_path / mod.RESULT_RELATIVE_PATH

    artifact = mod.run(
        result_path=result_path,
        upstream_grammar_artifact=_clean_grammar_gate(),
        upstream_panel_artifact=_panel_stub([model_id]),
        proposal_records=_records(model_id, payloads),
        tests_run=[{"command": str(TEST_PATH), "outcome": "passed"}],
    )

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact["required_row_count"] == 3
    assert artifact["reachable_state_count"] == 8
    assert artifact["terminal_state_count"] == 1
    assert artifact["valid_fixture_acceptance_rate"] == pytest.approx(1.0)
    assert artifact["invalid_fixture_rejection_rate"] == pytest.approx(1.0)
    assert artifact["row_completion_support_rate"] == pytest.approx(1.0)
    assert artifact["missing_row_risk"] == "low_missing_row_risk"
    assert artifact["local_mask_bias_diagnostic"]["reachable_but_proposal_unsupported_rows"] == []
    assert artifact["automaton_row_completion_ready"] is True
    assert artifact["llm_invoked"] is False
    assert artifact["no_model_specs_required"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert "model_specs" not in artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])

    mod.validate_artifact(artifact)


def test_scenario_verify_5552_missing_rows_are_bias_diagnostic() -> None:
    """SCENARIO-VERIFY-5552: missing rows block readiness but stay reachable."""

    fixture = positive.load_fixture_artifact()["fixture"]
    payloads = positive.build_fixture_candidate_payloads(fixture)
    model_id = "unit/model"
    artifact = mod.build_artifact(
        upstream_grammar_artifact=_clean_grammar_gate(),
        upstream_panel_artifact=_panel_stub([model_id]),
        proposal_records=_records(model_id, payloads[:2]),
    )

    unsupported = artifact["local_mask_bias_diagnostic"]["reachable_but_proposal_unsupported_rows"]

    assert artifact["row_completion_support_rate"] == pytest.approx(2 / 3)
    assert artifact["missing_row_risk"] == "high_missing_row_risk"
    assert artifact["automaton_row_completion_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked:")
    assert len(unsupported) == 1
    assert unsupported[0]["instance_id"] == payloads[2]["instance_id"]
    assert unsupported[0]["schema_reachable"] is True
    assert "proposal_path_missing_required_rows" in artifact["readiness_blockers"]


def test_req_verify_5552_malformed_and_invalid_enum_rows_dead_end() -> None:
    """REQ-VERIFY-5552: malformed rows and invalid enum values are rejected."""

    fixture = positive.load_fixture_artifact()["fixture"]
    payloads = positive.build_fixture_candidate_payloads(fixture)
    model_id = "unit/model"
    required_rows = mod.required_rows_from_panel(_panel_stub([model_id]), fixture=fixture)

    malformed_records = mod.proposal_records_from_text(model_id, '{"candidate_rows":')
    malformed = mod.evaluate_completion(required_rows, malformed_records, fixture=fixture)
    assert malformed["accepted_row_count"] == 0
    assert {row["reason"] for row in malformed["observed_dead_end_transitions"]} == {
        "malformed_row",
        "missing_required_row",
    }

    invalid_enum_payloads = deepcopy(payloads)
    invalid_enum_payloads[0]["conclusion"]["status"] = "maybe"
    invalid_enum = mod.evaluate_completion(
        required_rows,
        _records(model_id, invalid_enum_payloads),
        fixture=fixture,
    )
    reasons = {row["reason"] for row in invalid_enum["observed_dead_end_transitions"]}
    assert "invalid_enum_value" in reasons
    assert "missing_required_row" in reasons
    assert invalid_enum["row_completion_terminal"] is False

    invalid_fixtures = mod.build_invalid_fixture_payloads(payloads)
    invalid_rows = mod.evaluate_fixture_payloads(invalid_fixtures, fixture=fixture)
    assert mod.rejection_rate(invalid_rows) == pytest.approx(1.0)
    assert "invalid_enum_value" in {row["rejection_reason"] for row in invalid_rows}


def test_req_verify_5552_upstream_panel_rows_reproduce_exp5540_gap() -> None:
    """REQ-VERIFY-5552: Exp5540 panel rows show reachable but unsupported rows."""

    artifact = mod.build_artifact(
        upstream_grammar_artifact=_clean_grammar_gate(),
        upstream_panel_artifact={
            "models_attempted": [
                positive.MANDATED_HEADLINE_MODEL_IDS[0],
                positive.MANDATED_HEADLINE_MODEL_IDS[2],
            ],
            "panel_rows": [
                {
                    "model_hf_id": positive.MANDATED_HEADLINE_MODEL_IDS[0],
                    "instance_id": "claim_support_preference",
                    "schema_valid": True,
                    "parseable": True,
                    "exact_validator_correct": True,
                    "production_mode": "grammar_masking",
                },
                {
                    "model_hf_id": positive.MANDATED_HEADLINE_MODEL_IDS[2],
                    "instance_id": "claim_support_preference",
                    "schema_valid": True,
                    "parseable": True,
                    "exact_validator_correct": True,
                    "production_mode": "grammar_masking",
                },
            ],
            "inference_substrate": panel5540.INFERENCE_SUBSTRATE,
        },
    )

    diagnostic = artifact["local_mask_bias_diagnostic"]

    assert artifact["required_row_count"] == 6
    assert artifact["row_completion_support_rate"] == pytest.approx(2 / 6)
    assert diagnostic["syntactically_reachable_row_count"] == 6
    assert diagnostic["proposal_supported_row_count"] == 2
    assert diagnostic["unsupported_required_row_count"] == 4
    assert len(diagnostic["reachable_but_proposal_unsupported_rows"]) == 4
    assert "generic_json_gbnf_does_not_force_candidate_row_ids" in diagnostic["mask_bias_flags"]


def test_req_verify_5552_validation_fails_closed_on_overclaim() -> None:
    """REQ-VERIFY-5552: validation rejects LLM use, model specs, and false gates."""

    fixture = positive.load_fixture_artifact()["fixture"]
    payloads = positive.build_fixture_candidate_payloads(fixture)
    model_id = "unit/model"
    artifact = mod.build_artifact(
        upstream_grammar_artifact=_clean_grammar_gate(),
        upstream_panel_artifact=_panel_stub([model_id]),
        proposal_records=_records(model_id, payloads[:1]),
    )

    bad_llm = deepcopy(artifact)
    bad_llm["llm_invoked"] = True
    bad_llm["reproducibility_checksum"] = mod.payload_checksum(bad_llm)
    with pytest.raises(ValueError, match="llm_invoked"):
        mod.validate_artifact(bad_llm)

    bad_model_specs = deepcopy(artifact)
    bad_model_specs["model_specs"] = []
    bad_model_specs["reproducibility_checksum"] = mod.payload_checksum(bad_model_specs)
    with pytest.raises(ValueError, match="model_specs"):
        mod.validate_artifact(bad_model_specs)

    bad_ready = deepcopy(artifact)
    bad_ready["automaton_row_completion_ready"] = True
    bad_ready["reproducibility_checksum"] = mod.payload_checksum(bad_ready)
    with pytest.raises(ValueError, match="automaton_row_completion_ready"):
        mod.validate_artifact(bad_ready)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "bad"
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(bad_checksum)


def test_req_verify_5552_defensive_edges_stay_deterministic(tmp_path: Path) -> None:
    """REQ-VERIFY-5552: helper edge cases produce explicit deterministic labels."""

    fixture = positive.load_fixture_artifact()["fixture"]
    payloads = positive.build_fixture_candidate_payloads(fixture)
    model_id = "unit/model"
    required_rows = mod.required_rows_from_panel(_panel_stub([model_id]), fixture=fixture)
    duplicate_records = _records(model_id, [payloads[0], payloads[0]])
    duplicate = mod.evaluate_completion(required_rows, duplicate_records, fixture=fixture)

    assert "duplicate_required_row" in {
        row["reason"] for row in duplicate["observed_dead_end_transitions"]
    }
    assert mod.missing_row_risk(0.0) == "critical_missing_row_risk"
    assert mod.row_key_from_record("", "") == ""

    malformed = mod.evaluate_completion(required_rows, [{"model_hf_id": model_id}], fixture=fixture)
    assert "malformed_row" in {row["reason"] for row in malformed["observed_dead_end_transitions"]}

    unknown_payload = mod.evaluate_completion(
        required_rows,
        [{"model_hf_id": "unknown/model", "parsed_payload": payloads[0]}],
        fixture=fixture,
    )
    assert "unknown_required_row" in {
        row["reason"] for row in unknown_payload["observed_dead_end_transitions"]
    }

    unknown_classified = mod.evaluate_completion(
        required_rows,
        [
            {
                "model_hf_id": "unknown/model",
                "classified_row": {
                    "model_hf_id": "unknown/model",
                    "instance_id": payloads[0]["instance_id"],
                    "schema_valid": True,
                    "parseable": True,
                    "exact_validator_correct": True,
                },
            }
        ],
        fixture=fixture,
    )
    assert "unknown_required_row" in {
        row["reason"] for row in unknown_classified["observed_dead_end_transitions"]
    }

    bad_classified = mod.evaluate_completion(
        required_rows,
        [
            {
                "model_hf_id": model_id,
                "classified_row": {
                    "model_hf_id": model_id,
                    "instance_id": payloads[0]["instance_id"],
                    "schema_valid": False,
                    "parseable": True,
                    "exact_validator_correct": False,
                },
            }
        ],
        fixture=fixture,
    )
    assert "schema_invalid_row" in {
        row["reason"] for row in bad_classified["observed_dead_end_transitions"]
    }

    blockers = mod._readiness_blockers(
        grammar_artifact={"load_error": "missing"},
        required_rows=[{"schema_reachable": False}],
        valid_fixture_acceptance_rate=0.0,
        invalid_fixture_rejection_rate=0.0,
        completion={"row_completion_terminal": False},
        automaton={"terminal_state_count": 0},
    )
    assert blockers == [
        "invalid_fixture_not_fully_rejected",
        "no_terminal_completion_state",
        "proposal_path_missing_required_rows",
        "unreachable_required_rows",
        "upstream_grammar_preflight_not_clean",
        "valid_fixture_not_fully_accepted",
    ]
    assert "no_required_rows" in mod._readiness_blockers(
        grammar_artifact=_clean_grammar_gate(),
        required_rows=[],
        valid_fixture_acceptance_rate=1.0,
        invalid_fixture_rejection_rate=1.0,
        completion={"row_completion_terminal": False},
        automaton={"terminal_state_count": 0},
    )

    assert mod._required_model_ids({"per_model_reports": [{"model_hf_id": "a"}]}) == ["a"]
    assert mod._required_model_ids({"missing_instance_ids": [{"model_hf_id": "b"}]}) == ["b"]
    assert mod._required_model_ids({}) == ["deterministic_fixture_table"]
    assert mod._selected_backend_table_exposed({"selected_backend": "none"}) is False

    assert mod._rejection_reason({"schema_errors": [], "parse_status": ""}) == (
        "exact_validator_rejected_row"
    )
    assert mod._classified_rejection_reason(
        {"schema_errors": ["$.x expected one of ['a']"]}
    ) == "invalid_enum_value"
    assert mod._classified_rejection_reason({"schema_valid": False}) == "schema_invalid_row"
    assert mod._classified_rejection_reason(
        {"schema_valid": True, "exact_validator_correct": False}
    ) == "exact_validator_rejected_row"
    assert mod._classified_rejection_reason(
        {"schema_valid": True, "exact_validator_correct": True}
    ) == "malformed_row"

    assert mod._json_clone(object()) is not None
    valid_json = tmp_path / "valid.json"
    valid_json.write_text(json.dumps({"ok": True}), encoding="utf-8")
    assert mod._load_json(valid_json) == {"ok": True}
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert mod._load_json(list_json) == {"load_error": "json_not_object"}
    assert "load_error" in mod._load_json(tmp_path / "missing.json")

"""Tests for Exp5539 Gram2Token-style grammar-table preflight.

Spec refs: REQ-VERIFY-5539, SCENARIO-VERIFY-5539.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5512_structured_output_positive_control as positive
from carnot import experiment_5539_gram2token_grammar_table_preflight as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5539_gram2token_grammar_table_preflight.py")


def test_req_verify_5539_spec_declares_preflight_contract() -> None:
    """REQ-VERIFY-5539: OpenSpec anchors backend, table, and no-LLM fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5539") : spec.index("### REQ-VERIFY-5527")]

    assert "SCENARIO-VERIFY-5539" in section
    assert str(mod.RESULT_RELATIVE_PATH) in section
    assert positive.CANDIDATE_SCHEMA_VERSION in section
    assert mod.INFERENCE_SUBSTRATE in section
    assert "`llm_invoked` SHALL be `false`" in section
    assert "`decoding_speedup_claim` SHALL be" in section
    assert "SHALL NOT invoke an LLM or load" in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_verify_5539_selects_reachable_llama_cpp_compile_path(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5539: llama.cpp compile reachability produces table hashes."""

    compiled: list[str] = []

    def module_available(name: str) -> bool:
        return name == "llama_cpp"

    def llama_compile(grammar: str) -> object:
        compiled.append(grammar)
        return {"compiled": True}

    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        module_available=module_available,
        llama_grammar_compiler=llama_compile,
        tests_run=[{"command": str(TEST_PATH), "outcome": "passed"}],
    )
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert written == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["selected_backend"] == "llama_cpp_gbnf"
    assert artifact["backend_available"] is True
    assert artifact["grammar_table_preflight_ready"] is True
    assert artifact["schema_hash"] == positive.sha256_json(positive.candidate_schema())
    assert artifact["valid_fixture_acceptance_rate"] == pytest.approx(1.0)
    assert artifact["invalid_fixture_rejection_rate"] == pytest.approx(1.0)
    assert artifact["llm_invoked"] is False
    assert artifact["no_model_specs_required"] is True
    assert artifact["decoding_speedup_claim"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert "model_specs" not in artifact
    assert compiled and "root ::=" in compiled[0]

    hashes_by_name = {row["name"]: row for row in artifact["table_hashes"]}
    assert "hard_soft_schema_transition_table" in hashes_by_name
    assert "llama_cpp_json_gbnf" in hashes_by_name
    assert hashes_by_name["llama_cpp_json_gbnf"]["backend"] == "llama_cpp_gbnf"
    assert hashes_by_name["hard_soft_schema_transition_table"]["row_count"] > 0
    assert hashes_by_name["hard_soft_schema_transition_table"]["hash"] == mod.sha256_json(
        mod.build_schema_transition_table(positive.candidate_schema())
    )
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert "llama_cpp_token_transition_table_not_exposed" in artifact["unsupported_schema_features"]

    mod.validate_artifact(artifact)


def test_req_verify_5539_fixture_tables_reject_invalid_rows() -> None:
    """REQ-VERIFY-5539: valid and invalid fixture evidence is deterministic."""

    fixture = positive.load_fixture_artifact()["fixture"]
    valid_payloads = positive.build_fixture_candidate_payloads(fixture)
    invalid_payloads = mod.build_invalid_fixture_payloads(valid_payloads)
    valid_rows = mod.evaluate_payloads(valid_payloads, fixture=fixture)
    invalid_rows = mod.evaluate_payloads(invalid_payloads, fixture=fixture)

    assert len(valid_rows) == 3
    assert len(invalid_rows) == 3
    assert all(row["accepted"] is True for row in valid_rows)
    assert all(row["accepted"] is False for row in invalid_rows)
    assert {row["acceptance_status"] for row in invalid_rows} == {
        "schema_invalid",
        "validator_target_mismatch",
        "invalid_assignment_domain",
    }
    assert mod.acceptance_rate(valid_rows) == pytest.approx(1.0)
    assert mod.rejection_rate(invalid_rows) == pytest.approx(1.0)


def test_req_verify_5539_missing_backend_blocks_without_laundering_fallback() -> None:
    """REQ-VERIFY-5539: no constrained backend keeps the preflight gate closed."""

    artifact = mod.build_artifact(
        module_available=lambda _name: False,
        tests_run=[{"command": str(TEST_PATH), "outcome": "passed"}],
    )

    assert artifact["selected_backend"] == "none"
    assert artifact["backend_available"] is False
    assert artifact["grammar_table_preflight_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["valid_fixture_acceptance_rate"] == pytest.approx(1.0)
    assert artifact["invalid_fixture_rejection_rate"] == pytest.approx(1.0)
    by_name = {row["name"]: row for row in artifact["grammar_backend_candidates"]}
    assert by_name["llama_cpp_gbnf"]["failure_reason"] == "llama_cpp_not_installed"
    assert by_name["llguidance_json_schema"]["failure_reason"] == "llguidance_not_installed"
    assert by_name["xgrammar_json_schema"]["failure_reason"] == "xgrammar_not_installed"
    assert by_name["repository_json_schema_validator"]["available"] is True
    assert by_name["repository_json_schema_validator"]["constrained_generation"] is False


def test_req_verify_5539_validation_fails_closed() -> None:
    """REQ-VERIFY-5539: validation rejects generation, speedup, and checksum drift."""

    artifact = mod.build_artifact(
        module_available=lambda name: name == "llama_cpp",
        llama_grammar_compiler=lambda _grammar: object(),
    )

    bad_llm = deepcopy(artifact)
    bad_llm["llm_invoked"] = True
    bad_llm["reproducibility_checksum"] = mod.payload_checksum(bad_llm)
    with pytest.raises(ValueError, match="llm_invoked"):
        mod.validate_artifact(bad_llm)

    bad_speed = deepcopy(artifact)
    bad_speed["decoding_speedup_claim"] = True
    bad_speed["reproducibility_checksum"] = mod.payload_checksum(bad_speed)
    with pytest.raises(ValueError, match="decoding_speedup_claim"):
        mod.validate_artifact(bad_speed)

    bad_gate = deepcopy(artifact)
    bad_gate["backend_available"] = False
    bad_gate["reproducibility_checksum"] = mod.payload_checksum(bad_gate)
    with pytest.raises(ValueError, match="backend_available"):
        mod.validate_artifact(bad_gate)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "bad"
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(bad_checksum)


def test_req_verify_5539_optional_backend_and_defensive_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-5539: optional compiler edges stay honest and covered."""

    assert mod.build_invalid_fixture_payloads([{"too": "short"}]) == []

    candidates = mod.grammar_backend_candidates(
        module_available=lambda name: name in {"llama_cpp", "llguidance", "xgrammar"},
        llama_grammar_compiler=lambda _grammar: (_ for _ in ()).throw(RuntimeError("bad gbnf")),
        llguidance_grammar_compiler=lambda _schema: "LLG_GRAMMAR",
    )
    by_name = {row["name"]: row for row in candidates}

    assert by_name["llama_cpp_gbnf"]["failure_reason"] == "llama_cpp_grammar_compile_failed:RuntimeError"
    assert by_name["llguidance_json_schema"]["available"] is True
    assert by_name["llguidance_json_schema"]["grammar_hash"] == mod.sha256_text("LLG_GRAMMAR")
    assert by_name["xgrammar_json_schema"]["failure_reason"] == "xgrammar_compile_api_not_wired_in_this_preflight"
    assert mod.select_backend(candidates) == ("llguidance_json_schema", True)

    hashes = mod._table_hashes(
        candidates=candidates,
        selected_backend="llguidance_json_schema",
        schema_table=[{"path": "$", "type": "object"}],
        valid_rows=[{"accepted": True}],
        invalid_rows=[{"accepted": False}],
    )
    assert "llguidance_schema_grammar" in {row["name"] for row in hashes}

    failed_llguidance = mod.grammar_backend_candidates(
        module_available=lambda name: name == "llguidance",
        llguidance_grammar_compiler=lambda _schema: None,
    )
    assert (
        {row["name"]: row for row in failed_llguidance}["llguidance_json_schema"]["failure_reason"]
        == "llguidance_grammar_compile_failed:ValueError"
    )

    class FakeMatcher:
        @staticmethod
        def grammar_from_json_schema(schema: dict, overrides: dict | None = None) -> str:
            assert schema["type"] == "object"
            assert overrides == {"whitespace_flexible": False}
            return "compiled_from_import"

    class FakeModule:
        LLMatcher = FakeMatcher

    monkeypatch.setattr(mod.importlib, "import_module", lambda _name: FakeModule)
    assert mod._compile_llguidance_schema_grammar({"type": "object"}) == "compiled_from_import"

    class MissingMatcher:
        LLMatcher = object()

    monkeypatch.setattr(mod.importlib, "import_module", lambda _name: MissingMatcher)
    with pytest.raises(AttributeError, match="LLMatcher"):
        mod._compile_llguidance_schema_grammar({"type": "object"})

    assert mod._json_clone(object()) is not None
    assert mod._module_available("json") is True

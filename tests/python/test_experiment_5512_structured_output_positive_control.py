"""Tests for Exp5512 structured-output positive control.

Spec refs: REQ-VERIFY-5512, SCENARIO-VERIFY-5512.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5512_structured_output_positive_control as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5512_structured_output_positive_control.py")


def test_req_verify_5512_spec_declares_structured_positive_control() -> None:
    """REQ-VERIFY-5512: OpenSpec anchors schema, parser, runtime, and artifact fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[
        spec.index("### REQ-VERIFY-5512") : spec.index("### REQ-VERIFY-5501")
    ]

    assert "SCENARIO-VERIFY-5512" in section
    assert str(mod.RESULT_RELATIVE_PATH) in section
    assert "premises" in section
    assert "rules or constraints" in section
    assert "abstention reason" in section
    assert "validator target fields" in section
    for hf_id in mod.MANDATED_HEADLINE_MODEL_IDS:
        assert hf_id in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_verify_5512_fixture_rows_validate_schema_before_llm() -> None:
    """SCENARIO-VERIFY-5512: deterministic Exp5499 rows are schema-valid first."""

    payloads = mod.build_fixture_candidate_payloads()
    report = mod.evaluate_candidate_payloads(payloads)

    assert len(payloads) == 3
    assert all(not mod.schema_errors(row) for row in payloads)
    assert report["schema_validity_rate"] == pytest.approx(1.0)
    assert report["parseable_candidate_rows"] == 3
    assert report["missing_candidate_rows"] == 0
    assert report["exact_validator_handoff_ready"] is True
    assert report["structured_output_positive_control_ready"] is True
    assert {row["parse_status"] for row in report["candidate_rows"]} == {
        "schema_valid_assignment",
        "schema_valid_abstention",
    }

    by_instance = {row["instance_id"]: row for row in report["candidate_rows"]}
    assert by_instance["claim_support_preference"]["exact_validator_verdict"] == "exact_match"
    assert by_instance["claim_safety_conflict"]["exact_validator_verdict"] == "exact_match"
    assert (
        by_instance["claim_infeasible_negative_control"]["exact_validator_verdict"]
        == "correct_abstention"
    )
    assert by_instance["claim_infeasible_negative_control"]["abstention_reason"]


def test_req_verify_5512_parser_classifies_failures_without_dropping_rows() -> None:
    """REQ-VERIFY-5512: malformed rows become explicit parse classifications."""

    payloads = mod.build_fixture_candidate_payloads()
    no_json = mod.classify_candidate_text("no candidate here")
    assert no_json["parse_status"] == "no_json_object"
    assert no_json["schema_valid"] is False
    assert no_json["parseable"] is False

    missing_premises = deepcopy(payloads[0])
    missing_premises.pop("premises")
    schema_bad = mod.classify_candidate_payload(missing_premises)
    assert schema_bad["parse_status"] == "schema_invalid"
    assert "$.premises is required" in schema_bad["schema_errors"]

    unknown = deepcopy(payloads[0])
    unknown["instance_id"] = "not_in_fixture"
    assert mod.classify_candidate_payload(unknown)["parse_status"] == "unknown_instance"

    bad_target = deepcopy(payloads[0])
    bad_target["validator_target"]["instance_id"] = "claim_safety_conflict"
    assert mod.classify_candidate_payload(bad_target)["parse_status"] == "validator_target_mismatch"

    bad_keys = deepcopy(payloads[0])
    bad_keys["conclusion"]["assignment"] = {"support": "entailed"}
    assert mod.classify_candidate_payload(bad_keys)["parse_status"] == "invalid_assignment_keys"

    bad_domain = deepcopy(payloads[0])
    bad_domain["conclusion"]["assignment"]["source_quality"] = "tertiary"
    assert mod.classify_candidate_payload(bad_domain)["parse_status"] == "invalid_assignment_domain"

    bad_abstain = deepcopy(payloads[2])
    bad_abstain["abstention_reason"] = ""
    assert mod.classify_candidate_payload(bad_abstain)["parse_status"] == "abstention_reason_missing"


def test_req_verify_5512_exact_validator_handoff_rejects_bad_but_parseable_rows() -> None:
    """REQ-VERIFY-5512: parseable assignments still defer to exact validators."""

    hard_violation = deepcopy(mod.build_fixture_candidate_payloads()[0])
    hard_violation["candidate_id"] = "support_hard_violation_positive_control_probe"
    hard_violation["conclusion"]["assignment"]["support"] = "unsupported"

    report = mod.evaluate_candidate_payloads([hard_violation])
    row = report["candidate_rows"][0]

    assert row["parse_status"] == "schema_valid_assignment"
    assert row["parseable"] is True
    assert row["hard_constraints_pass"] is False
    assert row["exact_validator_verdict"] == "hard_constraint_violation"
    assert row["exact_validator_correct"] is False
    assert report["exact_validator_handoff_ready"] is True
    assert report["structured_output_positive_control_ready"] is False


def test_req_verify_5512_runtime_probe_reports_parser_only_fallback() -> None:
    """REQ-VERIFY-5512: absent grammar runtimes keep parser-only fallback explicit."""

    status = mod.probe_structured_runtime(
        module_available=lambda _name: False,
        llama_cpp_cuda_probe=lambda: False,
        llama_grammar_compiler=lambda _grammar: (_ for _ in ()).throw(RuntimeError("absent")),
    )

    assert status["grammar_runtime_available"] is False
    assert status["parser_only_fallback_used"] is True
    assert status["llama_cpp_cuda_available"] is False
    assert "llguidance_not_installed" in status["runtime_blockers"]
    assert "xgrammar_not_installed" in status["runtime_blockers"]
    assert "llama_cpp_not_installed" in status["runtime_blockers"]


def test_req_verify_5512_runtime_probe_accepts_llama_cpp_grammar() -> None:
    """REQ-VERIFY-5512: llama.cpp GBNF support is a constrained runtime path."""

    status = mod.probe_structured_runtime(
        module_available=lambda name: name == "llama_cpp",
        llama_cpp_cuda_probe=lambda: True,
        llama_grammar_compiler=lambda grammar: {"compiled": grammar},
    )

    assert status["grammar_runtime_available"] is True
    assert status["parser_only_fallback_used"] is False
    assert status["llama_cpp_cuda_available"] is True
    assert status["llama_cpp_grammar_available"] is True


def test_req_verify_5512_artifact_writes_required_fields_with_parser_fallback(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-5512: artifact preserves parser-only fallback and gate state."""

    runtime_status = {
        "grammar_runtime_available": False,
        "parser_only_fallback_used": True,
        "llama_cpp_cuda_available": False,
        "llama_cpp_grammar_available": False,
        "llguidance_available": False,
        "xgrammar_available": False,
        "runtime_blockers": ["llguidance_not_installed", "xgrammar_not_installed"],
    }
    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        runtime_status=runtime_status,
        cache_resolver=lambda _hf_id, _preferred_quant="Q4_K_M": None,
        smoke_runner=lambda _spec, _prompt, _grammar: pytest.fail("smoke must not run"),
        tests_run=[{"command": str(TEST_PATH), "outcome": "passed"}],
    )
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert written == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["schema_path"] == mod.SCHEMA_PATH
    assert artifact["parser_path"] == mod.PARSER_PATH
    assert artifact["tests_added_or_reused"] == [
        str(TEST_PATH),
        "tests/python/test_experiment_5499_preference_maxsat_minimal_fixture_v499.py",
        "tests/python/test_experiment_5500_sota_concept_claim_panel_v499.py",
    ]
    assert artifact["smoke_models_used"] == []
    assert artifact["grammar_runtime_available"] is False
    assert artifact["parser_only_fallback_used"] is True
    assert artifact["schema_validity_rate"] == pytest.approx(1.0)
    assert artifact["parseable_candidate_rows"] == 3
    assert artifact["missing_candidate_rows"] == 0
    assert artifact["exact_validator_handoff_ready"] is True
    assert artifact["structured_output_positive_control_ready"] is True
    assert artifact["sota_panel_gate_open"] is False
    assert artifact["llama_cpp_cuda_available"] is False
    assert artifact["inference_substrate"] == "structured_output_fixture_or_live_llm_smoke"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    mod.validate_artifact(artifact)


def test_req_verify_5512_artifact_records_schema_path_live_smoke_when_injected(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5512: cached-model smoke rows reuse the same parser path."""

    payload = mod.build_fixture_candidate_payloads()[0]
    model_paths = {hf_id: str(tmp_path / f"{hf_id.rsplit('/', 1)[-1]}-Q4_K_M.gguf") for hf_id in mod.MANDATED_HEADLINE_MODEL_IDS}
    for path in model_paths.values():
        Path(path).write_text("fake gguf", encoding="utf-8")

    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        runtime_status={
            "grammar_runtime_available": True,
            "parser_only_fallback_used": False,
            "llama_cpp_cuda_available": True,
            "llama_cpp_grammar_available": True,
            "llguidance_available": False,
            "xgrammar_available": False,
            "runtime_blockers": [],
        },
        cache_resolver=lambda hf_id, _preferred_quant="Q4_K_M": model_paths.get(hf_id),
        smoke_runner=lambda _spec, _prompt, _grammar: json.dumps(payload),
        max_smoke_models=1,
    )

    assert artifact["smoke_models_used"] == [mod.MANDATED_HEADLINE_MODEL_IDS[0]]
    assert artifact["live_smoke_rows"][0]["parse_status"] == "schema_valid_assignment"
    assert artifact["live_smoke_rows"][0]["exact_validator_verdict"] == "exact_match"
    assert artifact["sota_panel_gate_open"] is True
    mod.validate_artifact(artifact)


def test_req_verify_5512_artifact_validation_fails_closed() -> None:
    """REQ-VERIFY-5512: artifact validator rejects schema drift and false gates."""

    artifact = mod.build_artifact(
        runtime_status={
            "grammar_runtime_available": False,
            "parser_only_fallback_used": True,
            "llama_cpp_cuda_available": False,
            "llama_cpp_grammar_available": False,
            "llguidance_available": False,
            "xgrammar_available": False,
            "runtime_blockers": ["llguidance_not_installed"],
        },
    )
    mod.validate_artifact(artifact)

    missing = deepcopy(artifact)
    missing.pop("schema_path")
    with pytest.raises(ValueError, match="schema_path"):
        mod.validate_artifact(missing)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)

    bad_gate = deepcopy(artifact)
    bad_gate["parser_only_fallback_used"] = True
    bad_gate["sota_panel_gate_open"] = True
    bad_gate["reproducibility_checksum"] = mod.payload_checksum(bad_gate)
    with pytest.raises(ValueError, match="sota_panel_gate_open"):
        mod.validate_artifact(bad_gate)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "bad"
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(bad_checksum)


def test_req_verify_5512_edge_branches_remain_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-5512: fallback, blocker, and runtime-error branches stay covered."""

    fallback_fixture = mod.load_fixture_artifact(path=tmp_path / "missing_5499.json")
    assert fallback_fixture["fixture"]["schema"].endswith("typed_claim_state.v1")

    soft_suboptimal = deepcopy(mod.build_fixture_candidate_payloads()[1])
    soft_suboptimal["conclusion"]["assignment"]["action"] = "reject"
    soft_row = mod.classify_candidate_payload(soft_suboptimal)
    assert soft_row["exact_validator_verdict"] == "soft_suboptimal"

    invalid = deepcopy(mod.build_fixture_candidate_payloads()[0])
    invalid.pop("premises")
    invalid_report = mod.evaluate_candidate_payloads([invalid])
    assert invalid_report["parse_failure_counts"] == {"schema_invalid": 1}

    original_llguidance_probe = mod._llguidance_schema_available
    monkeypatch.setattr(mod, "_llguidance_schema_available", lambda: False)
    blocked_runtime = mod.probe_structured_runtime(
        module_available=lambda name: name in {"llguidance", "llama_cpp"},
        llama_cpp_cuda_probe=lambda: (_ for _ in ()).throw(RuntimeError("cuda probe failed")),
        llama_grammar_compiler=lambda _grammar: (_ for _ in ()).throw(RuntimeError("bad grammar")),
    )
    assert "llguidance_schema_compiler_unavailable" in blocked_runtime["runtime_blockers"]
    assert "llama_cpp_grammar_unavailable:RuntimeError" in blocked_runtime["runtime_blockers"]
    assert "llama_cpp_cuda_probe_failed:RuntimeError" in blocked_runtime["runtime_blockers"]
    assert "llama_cpp_cuda_unavailable" in blocked_runtime["runtime_blockers"]
    monkeypatch.setattr(mod, "_llguidance_schema_available", original_llguidance_probe)

    no_cache = mod.run(
        result_path=tmp_path / "no_cache.json",
        runtime_status={
            "grammar_runtime_available": True,
            "parser_only_fallback_used": False,
            "llama_cpp_cuda_available": True,
            "llama_cpp_grammar_available": True,
            "llguidance_available": False,
            "xgrammar_available": False,
            "runtime_blockers": [],
        },
        cache_resolver=lambda _hf_id, _preferred_quant="Q4_K_M": None,
        smoke_runner=lambda _spec, _prompt, _grammar: pytest.fail("no cached model"),
    )
    assert no_cache["live_smoke_rows"] == []
    assert no_cache["sota_panel_gate_open"] is False

    cached_path = tmp_path / "Qwen3.6-35B-A3B-Q4_K_M.gguf"
    cached_path.write_text("fake gguf", encoding="utf-8")
    runtime_error = mod.run(
        result_path=tmp_path / "runtime_error.json",
        runtime_status={
            "grammar_runtime_available": True,
            "parser_only_fallback_used": False,
            "llama_cpp_cuda_available": True,
            "llama_cpp_grammar_available": False,
            "llguidance_available": False,
            "xgrammar_available": False,
            "runtime_blockers": [],
        },
        cache_resolver=lambda hf_id, _preferred_quant="Q4_K_M": (
            str(cached_path) if hf_id == mod.MANDATED_HEADLINE_MODEL_IDS[0] else None
        ),
        pair_resolver=lambda: None,
        smoke_runner=lambda _spec, _prompt, _grammar: (_ for _ in ()).throw(
            RuntimeError("load failed")
        ),
    )
    assert runtime_error["live_smoke_rows"][0]["parse_status"] == "runtime_error"
    assert runtime_error["smoke_models_used"] == []
    assert mod.honest_verdict(
        positive_ready=False,
        parser_only_fallback_used=False,
        smoke_models_used=[],
    ).startswith("blocked:")
    assert mod.honest_verdict(
        positive_ready=True,
        parser_only_fallback_used=False,
        smoke_models_used=[],
    ).endswith("sota_gate_closed")
    assert (
        "unparseable"
        in mod.honest_verdict(
            positive_ready=True,
            parser_only_fallback_used=False,
            smoke_models_used=[mod.MANDATED_HEADLINE_MODEL_IDS[0]],
            live_smoke_parseable=False,
        )
    )

    false_open = deepcopy(runtime_error)
    false_open["sota_panel_gate_open"] = True
    false_open["reproducibility_checksum"] = mod.payload_checksum(false_open)
    with pytest.raises(ValueError, match="sota_panel_gate_open"):
        mod.validate_artifact(false_open)

    assert mod._module_available("json") is True

    class FakeMatcher:
        @staticmethod
        def grammar_from_json_schema(_schema: object) -> str:
            return "root ::= \"{}\""

    class FakeLlGuidance:
        LLMatcher = FakeMatcher

    monkeypatch.setattr(mod.importlib, "import_module", lambda _name: FakeLlGuidance)
    assert mod._llguidance_schema_available() is True

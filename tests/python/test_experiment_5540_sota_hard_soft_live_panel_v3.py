"""Tests for Exp5540 SOTA hard/soft live panel v3.

Spec refs: REQ-VERIFY-5540, SCENARIO-VERIFY-5540.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5512_structured_output_positive_control as positive
from carnot import experiment_5538_sota_panel_duration_substrate_corrigendum as gate5538
from carnot import experiment_5539_gram2token_grammar_table_preflight as gate5539
from carnot import experiment_5540_sota_hard_soft_live_panel_v3 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5540_sota_hard_soft_live_panel_v3.py")


def _unit_model_specs() -> list[dict]:
    return [
        {
            "name": row["name"],
            "hf_id": row["hf_id"],
            "role": row["role"],
            "preferred_quant": row["preferred_quant"],
            "headline_eligible": True,
            "model_path": f"/tmp/{row['name']}.gguf",
            "local_model_present": True,
            "model_filename": f"{row['name']}.gguf",
            "model_size_bytes": 123,
        }
        for row in positive.MODEL_SPECS
    ]


def _clean_duration_gate() -> dict:
    return {
        "sota_panel_duration_corrigendum_ready": True,
        "adversarial_clean": True,
        "inference_substrate": gate5538.INFERENCE_SUBSTRATE,
        "research_conductor_modified": False,
        "honest_verdict": "complete: unit duration gate clean",
    }


def _clean_grammar_gate() -> dict:
    return {
        "grammar_table_preflight_ready": True,
        "backend_available": True,
        "llm_invoked": False,
        "decoding_speedup_claim": False,
        "inference_substrate": gate5539.INFERENCE_SUBSTRATE,
        "research_conductor_modified": False,
        "selected_backend": "llama_cpp_gbnf",
        "honest_verdict": "complete: unit grammar gate clean",
    }


def _valid_live_receipt(raw_output: str, *, mode: str = "grammar_masking") -> dict:
    return {
        "live_model_invoked": True,
        "models_attempted": [positive.MANDATED_HEADLINE_MODEL_IDS[0]],
        "raw_output": raw_output,
        "measured_duration_s": mod.DURATION_FLOOR_S + 2.0,
        "backend": "llama_cpp_python_cuda_gguf",
        "helper_path": "carnot.inference.sota_models.cached_sota_pair",
        "binding": "llama_cpp.Llama.create_chat_completion",
        "command": None,
        "random_seed": mod.RANDOM_SEED,
        "prompt_hash": "a" * 64,
        "prompt_tokens": 321,
        "completion_tokens": 123,
        "max_tokens": 1024,
        "n_ctx": 4096,
        "n_batch": 128,
        "n_gpu_layers": -1,
        "production_mode": mode,
        "grammar_masking_used": mode == "grammar_masking",
        "gpu_offload_evidence": {
            "gpu_offload_verified": True,
            "offload_evidence": True,
            "gpu_memory_delta_mb": 2048.0,
            "n_gpu_layers": -1,
        },
        "runtime_error": None,
    }


def test_req_verify_5540_spec_declares_live_panel_contract() -> None:
    """REQ-VERIFY-5540: OpenSpec anchors the v3 live panel contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5540") : spec.index("### REQ-VERIFY-5501")]

    assert "SCENARIO-VERIFY-5540" in section
    assert str(mod.RESULT_RELATIVE_PATH) in section
    assert mod.INFERENCE_SUBSTRATE in section
    assert "AutoTokenizer.from_pretrained" in section
    for hf_id in positive.MANDATED_HEADLINE_MODEL_IDS:
        assert hf_id in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_verify_5540_unclean_gate_emits_gated_null() -> None:
    """SCENARIO-VERIFY-5540: unclean prerequisites stop before live rows."""

    dirty_duration_gate = {**_clean_duration_gate(), "adversarial_clean": False}
    artifact = mod.build_artifact(
        duration_gate_artifact=dirty_duration_gate,
        grammar_gate_artifact=_clean_grammar_gate(),
        live_receipts=[
            _valid_live_receipt(json.dumps({"candidate_rows": [{"candidate_id": "ignored"}]}))
        ],
        model_specs=_unit_model_specs(),
        tests_run=[{"command": str(TEST_PATH), "outcome": "passed"}],
    )

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["gates_clean"] is False
    assert artifact["models_attempted"] == []
    assert artifact["rows_requested"] == 3
    assert artifact["rows_emitted"] == 0
    assert artifact["missing_candidate_rows"] == 3
    assert artifact["schema_validity_rate"] == pytest.approx(0.0)
    assert artifact["exact_validator_accuracy"] == pytest.approx(0.0)
    assert artifact["abstention_rate"] == pytest.approx(0.0)
    assert artifact["sota_hard_soft_claim_allowed"] is False
    assert artifact["adversarial_clean"] is True
    assert artifact["honest_verdict"].startswith("blocked:")
    assert "duration_substrate_gate_not_clean" in artifact["readiness_blockers"]
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])

    mod.validate_artifact(artifact)


def test_scenario_verify_5540_live_exact_rows_open_claim() -> None:
    """SCENARIO-VERIFY-5540: live rows are schema and exact-validator scored."""

    fixture = positive.load_fixture_artifact()["fixture"]
    payloads = positive.build_fixture_candidate_payloads(fixture)
    receipt = _valid_live_receipt(json.dumps({"candidate_rows": payloads}))
    artifact = mod.build_artifact(
        duration_gate_artifact=_clean_duration_gate(),
        grammar_gate_artifact=_clean_grammar_gate(),
        live_receipts=[receipt],
        model_specs=_unit_model_specs(),
    )

    assert artifact["gates_clean"] is True
    assert artifact["models_attempted"] == [positive.MANDATED_HEADLINE_MODEL_IDS[0]]
    assert artifact["rows_requested"] == 3
    assert artifact["rows_emitted"] == 3
    assert artifact["missing_candidate_rows"] == 0
    assert artifact["schema_validity_rate"] == pytest.approx(1.0)
    assert artifact["exact_validator_accuracy"] == pytest.approx(1.0)
    assert artifact["preference_optimality_rate"] == pytest.approx(1.0)
    assert artifact["abstention_rate"] == pytest.approx(1 / 3)
    assert artifact["confident_wrong_rate"] == pytest.approx(0.0)
    assert artifact["exact_correct_rows"] == 3
    assert artifact["row_production_mode_counts"]["grammar_masking"] == 3
    assert artifact["row_production_mode_counts"]["repair"] == 0
    assert artifact["prompt_hashes"] == ["a" * 64]
    assert artifact["output_hashes"] == [mod.sha256_text(receipt["raw_output"])]
    assert artifact["gpu_offload_evidence"]["gpu_offload_verified"] is True
    assert artifact["measured_duration_s"] > mod.DURATION_FLOOR_S
    assert artifact["sota_hard_soft_claim_allowed"] is True
    assert artifact["honest_verdict"].startswith("complete:")

    mod.validate_artifact(artifact)


def test_req_verify_5540_wrong_or_repaired_rows_do_not_open_claim() -> None:
    """REQ-VERIFY-5540: exact failures and repair provenance fail closed."""

    fixture = positive.load_fixture_artifact()["fixture"]
    payloads = positive.build_fixture_candidate_payloads(fixture)
    wrong_payloads = deepcopy(payloads)
    wrong_payloads[0]["conclusion"]["assignment"]["support"] = "unsupported"
    repaired_receipt = _valid_live_receipt(
        json.dumps({"candidate_rows": wrong_payloads}),
        mode="repair",
    )
    artifact = mod.build_artifact(
        duration_gate_artifact=_clean_duration_gate(),
        grammar_gate_artifact=_clean_grammar_gate(),
        live_receipts=[repaired_receipt],
        model_specs=_unit_model_specs(),
    )

    assert artifact["rows_requested"] == 3
    assert artifact["rows_emitted"] == 3
    assert artifact["schema_validity_rate"] == pytest.approx(1.0)
    assert artifact["exact_validator_accuracy"] == pytest.approx(2 / 3)
    assert artifact["preference_optimality_rate"] == pytest.approx(1 / 2)
    assert artifact["confident_wrong_rate"] == pytest.approx(1 / 3)
    assert artifact["row_production_mode_counts"]["repair"] == 3
    assert artifact["sota_hard_soft_claim_allowed"] is False
    assert "exact_validator_mismatch" in artifact["readiness_blockers"]
    assert "repair_rows_not_headline_eligible" in artifact["readiness_blockers"]


def test_req_verify_5540_runtime_and_helper_branches_fail_closed() -> None:
    """REQ-VERIFY-5540: helper fallbacks and runtime failures stay explicit."""

    assert positive.CANDIDATE_SCHEMA_VERSION in mod.build_live_prompt()
    assert "load_error" in mod._load_gate_artifact(Path("/tmp/missing_exp5540_gate.json"))
    assert mod._model_specs(None)
    assert mod._model_specs_match_mandated("not rows") is False
    assert mod._models_attempted(positive.MANDATED_HEADLINE_MODEL_IDS[0]) == [
        positive.MANDATED_HEADLINE_MODEL_IDS[0]
    ]
    assert mod._models_attempted(object()) == []
    assert mod._safe_float(object(), 1.25) == pytest.approx(1.25)

    grammar_blocked = mod.gate_status(
        _clean_duration_gate(),
        {**_clean_grammar_gate(), "backend_available": False},
    )
    assert grammar_blocked["gates_clean"] is False
    assert "grammar_table_preflight_not_clean" in grammar_blocked["gate_blockers"]

    parsed_model_only = mod.parse_live_receipt(
        {
            "live_model_invoked": True,
            "model_hf_id": positive.MANDATED_HEADLINE_MODEL_IDS[1],
            "raw_output": "{}",
            "duration_s": "not-a-float",
            "grammar_masking_used": True,
            "gpu_offload_evidence": {"load_offload_evidence": True},
        },
        default_prompt_hash="b" * 64,
    )
    assert parsed_model_only["models_attempted"] == [positive.MANDATED_HEADLINE_MODEL_IDS[1]]
    assert parsed_model_only["production_mode"] == "grammar_masking"
    assert parsed_model_only["measured_duration_s"] == pytest.approx(0.0)
    assert parsed_model_only["gpu_offload_evidence"]["gpu_offload_verified"] is True

    parsed_extraction = mod.parse_live_receipt(
        {"live_model_invoked": True, "model_hf_id": "legacy/tiny", "raw_output": "{}"},
        default_prompt_hash="c" * 64,
    )
    assert parsed_extraction["models_attempted"] == []
    assert parsed_extraction["production_mode"] == "post_hoc_extraction"
    assert mod._candidate_records_from_receipt({"live_model_invoked": False}) == ([], [])

    blockers = mod._readiness_blockers(
        gates_clean=True,
        gate_blockers=[],
        metrics={
            "rows_requested": 0,
            "rows_emitted": 0,
            "missing_candidate_rows": 1,
            "schema_validity_rate": 0.0,
            "exact_validator_accuracy": 0.0,
            "preference_optimality_rate": 0.0,
            "confident_wrong_rate": 0.0,
            "parse_failures": [{"parse_status": "unit"}],
            "row_production_mode_counts": {"repair": 0},
        },
        models_attempted=[],
        receipts=[{"live_model_invoked": False}],
        gpu_offload_evidence={"gpu_offload_verified": False},
        measured_duration_s=0.0,
    )
    assert "live_model_runtime_failed" in blockers
    assert "no_rows_requested" in blockers
    assert "parse_failures" in blockers

    cold_specs = deepcopy(_unit_model_specs())
    for spec in cold_specs:
        spec["local_model_present"] = False
        spec["model_path"] = None
    cold_artifact = mod.build_artifact(
        duration_gate_artifact=_clean_duration_gate(),
        grammar_gate_artifact=_clean_grammar_gate(),
        model_specs=cold_specs,
        pair_resolver=lambda: None,
        live_receipts=None,
    )
    assert cold_artifact["gates_clean"] is True
    assert cold_artifact["prompt_hashes"]
    assert cold_artifact["models_attempted"] == []
    assert "no_live_model_invoked" in cold_artifact["readiness_blockers"]


def test_req_verify_5540_extra_rows_and_runtime_errors_are_visible() -> None:
    """REQ-VERIFY-5540: duplicate/extra rows and runtime errors are preserved."""

    fixture = positive.load_fixture_artifact()["fixture"]
    payloads = positive.build_fixture_candidate_payloads(fixture)
    receipt = _valid_live_receipt(json.dumps({"candidate_rows": [*payloads, payloads[0]]}))
    receipt["runtime_error"] = "unit_runtime_warning"

    artifact = mod.build_artifact(
        duration_gate_artifact=_clean_duration_gate(),
        grammar_gate_artifact=_clean_grammar_gate(),
        live_receipts=[receipt],
        model_specs=_unit_model_specs(),
    )

    assert artifact["rows_emitted"] == 4
    assert len(artifact["extra_emitted_rows"]) == 1
    assert artifact["parse_failures"] == [
        {"parse_status": "runtime_error", "detail": "unit_runtime_warning"}
    ]
    assert artifact["sota_hard_soft_claim_allowed"] is False
    assert "parse_failures" in artifact["readiness_blockers"]


def test_req_verify_5540_validation_fails_closed_on_overclaim() -> None:
    """REQ-VERIFY-5540: validation rejects false claim gates and checksums."""

    artifact = mod.build_artifact(
        duration_gate_artifact={**_clean_duration_gate(), "adversarial_clean": False},
        grammar_gate_artifact=_clean_grammar_gate(),
        model_specs=_unit_model_specs(),
        live_receipts=[],
    )

    bad_claim = deepcopy(artifact)
    bad_claim["sota_hard_soft_claim_allowed"] = True
    bad_claim["reproducibility_checksum"] = mod.payload_checksum(bad_claim)
    with pytest.raises(ValueError, match="sota_hard_soft_claim_allowed"):
        mod.validate_artifact(bad_claim)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    bad_substrate["reproducibility_checksum"] = mod.payload_checksum(bad_substrate)
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "bad"
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(bad_checksum)


def test_scenario_verify_5540_run_writes_requested_result_path(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5540: run writes the v3 deliverable JSON artifact."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.run(
        result_path=result_path,
        duration_gate_artifact={**_clean_duration_gate(), "adversarial_clean": False},
        grammar_gate_artifact=_clean_grammar_gate(),
        model_specs=_unit_model_specs(),
        live_receipts=[],
        write=True,
    )

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact["result_path"] == str(mod.RESULT_RELATIVE_PATH)

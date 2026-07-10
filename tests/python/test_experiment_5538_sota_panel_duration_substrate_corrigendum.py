"""Tests for Exp5538 SOTA panel duration/substrate corrigendum.

Spec refs: REQ-VERIFY-5538, SCENARIO-VERIFY-5538.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5512_structured_output_positive_control as positive
from carnot import experiment_5538_sota_panel_duration_substrate_corrigendum as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5538_sota_panel_duration_substrate_corrigendum.py")


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
        }
        for row in positive.MODEL_SPECS
    ]


def _valid_live_receipt(raw_output: str) -> dict:
    return {
        "live_model_invoked": True,
        "models_attempted": [positive.MANDATED_HEADLINE_MODEL_IDS[0]],
        "raw_output": raw_output,
        "measured_duration_s": mod.DURATION_FLOOR_S + 5.0,
        "backend": "llama_cpp_python_cuda_gguf",
        "helper_path": "carnot.inference.sota_models.cached_sota_pair",
        "binding": "llama_cpp.Llama.create_chat_completion",
        "command": None,
        "random_seed": mod.RANDOM_SEED,
        "prompt_hash": "a" * 64,
        "gpu_offload_evidence": {
            "gpu_offload_verified": True,
            "offload_evidence": True,
            "gpu_memory_delta_mb": 2048.0,
            "n_gpu_layers": -1,
        },
        "runtime_error": None,
    }


def test_req_verify_5538_spec_declares_corrigendum_contract() -> None:
    """REQ-VERIFY-5538: OpenSpec anchors the corrigendum artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5538") : spec.index("### REQ-VERIFY-5501")]

    assert "SCENARIO-VERIFY-5538" in section
    assert str(mod.RESULT_RELATIVE_PATH) in section
    assert str(mod.UPSTREAM_PANEL_RELATIVE_PATH) in section
    assert mod.INFERENCE_SUBSTRATE in section
    assert "AutoTokenizer.from_pretrained" in section
    for hf_id in positive.MANDATED_HEADLINE_MODEL_IDS:
        assert hf_id in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_verify_5538_unavailable_live_runtime_downgrades_without_fabrication() -> None:
    """SCENARIO-VERIFY-5538: missing live execution produces no quality claim."""

    receipt = mod.claim_downgrade_receipt("unit_test_runtime_unavailable")
    artifact = mod.build_artifact(
        live_receipt=receipt,
        model_specs=_unit_model_specs(),
        tests_run=[{"command": str(TEST_PATH), "outcome": "passed"}],
    )

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["upstream_panel_path"] == str(mod.UPSTREAM_PANEL_RELATIVE_PATH)
    assert artifact["live_model_invoked"] is False
    assert artifact["models_attempted"] == []
    assert artifact["rows_requested"] == 3
    assert artifact["rows_emitted"] == 0
    assert artifact["missing_candidate_rows"] == 3
    assert artifact["schema_validity_rate"] == pytest.approx(0.0)
    assert artifact["exact_validator_accuracy"] == pytest.approx(0.0)
    assert artifact["preference_optimality_rate"] == pytest.approx(0.0)
    assert artifact["abstention_rate"] == pytest.approx(0.0)
    assert artifact["confident_wrong_rate"] == pytest.approx(0.0)
    assert artifact["no_quality_claim_if_not_live"] is True
    assert artifact["quality_claim_allowed"] is False
    assert artifact["sota_panel_duration_corrigendum_ready"] is True
    assert artifact["adversarial_clean"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert "no_live_model_invoked" in artifact["readiness_blockers"]
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])

    mod.validate_artifact(artifact)


def test_scenario_verify_5538_live_schema_valid_rows_can_authenticate_claim() -> None:
    """SCENARIO-VERIFY-5538: live rows need duration, offload, schema, and exact scores."""

    fixture = positive.load_fixture_artifact()["fixture"]
    payloads = positive.build_fixture_candidate_payloads(fixture)
    receipt = _valid_live_receipt(json.dumps({"candidate_rows": payloads}))

    artifact = mod.build_artifact(live_receipt=receipt, model_specs=_unit_model_specs())

    assert artifact["live_model_invoked"] is True
    assert artifact["models_attempted"] == [positive.MANDATED_HEADLINE_MODEL_IDS[0]]
    assert artifact["rows_requested"] == 3
    assert artifact["rows_emitted"] == 3
    assert artifact["missing_candidate_rows"] == 0
    assert artifact["schema_validity_rate"] == pytest.approx(1.0)
    assert artifact["exact_validator_accuracy"] == pytest.approx(1.0)
    assert artifact["preference_optimality_rate"] == pytest.approx(1.0)
    assert artifact["abstention_rate"] == pytest.approx(1 / 3)
    assert artifact["confident_wrong_rate"] == pytest.approx(0.0)
    assert artifact["duration_floor_s"] == pytest.approx(mod.DURATION_FLOOR_S)
    assert artifact["measured_duration_s"] > artifact["duration_floor_s"]
    assert artifact["gpu_offload_evidence"]["gpu_offload_verified"] is True
    assert artifact["no_quality_claim_if_not_live"] is False
    assert artifact["quality_claim_allowed"] is True
    assert artifact["sota_panel_duration_corrigendum_ready"] is True
    assert artifact["honest_verdict"].startswith("complete:")

    mod.validate_artifact(artifact)


def test_req_verify_5538_live_but_too_short_or_schema_invalid_stays_downgraded() -> None:
    """REQ-VERIFY-5538: too-short or schema-invalid live rows do not open a claim."""

    receipt = _valid_live_receipt(json.dumps({"candidate_rows": [{"candidate_id": "bad"}]}))
    receipt["measured_duration_s"] = 1.25
    artifact = mod.build_artifact(live_receipt=receipt, model_specs=_unit_model_specs())

    assert artifact["live_model_invoked"] is True
    assert artifact["rows_requested"] == 3
    assert artifact["rows_emitted"] == 1
    assert artifact["missing_candidate_rows"] == 3
    assert artifact["schema_validity_rate"] == pytest.approx(0.0)
    assert artifact["exact_validator_accuracy"] == pytest.approx(0.0)
    assert artifact["abstention_rate"] == pytest.approx(0.0)
    assert artifact["quality_claim_allowed"] is False
    assert artifact["no_quality_claim_if_not_live"] is True
    assert artifact["sota_panel_duration_corrigendum_ready"] is True
    assert "duration_below_live_claim_floor" in artifact["readiness_blockers"]
    assert "schema_invalid_or_missing_rows" in artifact["readiness_blockers"]


def test_req_verify_5538_duration_substrate_receipt_parser_normalizes_aliases() -> None:
    """REQ-VERIFY-5538: duration/substrate receipts are parsed before claim gating."""

    parsed = mod.parse_duration_substrate_receipt(
        {
            "duration_s": "61.5",
            "backend": "llama_cpp_python_cuda_gguf",
            "prompt_sha256": "b" * 64,
            "gpu_offload_evidence": {
                "offload_evidence": True,
                "gpu_memory_delta_mb": "512.5",
            },
        }
    )

    assert parsed["measured_duration_s"] == pytest.approx(61.5)
    assert parsed["backend"] == "llama_cpp_python_cuda_gguf"
    assert parsed["prompt_hash"] == "b" * 64
    assert parsed["gpu_offload_evidence"]["gpu_offload_verified"] is True
    assert parsed["gpu_offload_evidence"]["gpu_memory_delta_mb"] == pytest.approx(512.5)

    absent = mod.parse_duration_substrate_receipt({})
    assert absent["measured_duration_s"] == pytest.approx(0.0)
    assert absent["gpu_offload_evidence"]["gpu_offload_verified"] is False

    assert "load_error" in mod.load_upstream_panel(Path("/tmp/definitely_missing_5538.json"))
    assert mod._model_specs(None)
    assert mod._model_specs_match_mandated("not rows") is False
    assert mod._models_attempted("not rows") == []

    blockers = mod._readiness_blockers(
        receipt={
            "live_model_invoked": True,
            "models_attempted": ["legacy/tiny"],
            "measured_duration_s": mod.DURATION_FLOOR_S + 1.0,
            "gpu_offload_evidence": {"gpu_offload_verified": True},
        },
        report={
            "missing_candidate_rows": 0,
            "schema_validity_rate": 1.0,
            "exact_validator_accuracy": 1.0,
            "preference_optimality_rate": 1.0,
            "confident_wrong_rate": 1.0,
        },
    )
    assert "no_mandated_sota_model_attempted" in blockers
    assert "confident_wrong_rows" in blockers


def test_req_verify_5538_validation_fails_closed_on_overclaim() -> None:
    """REQ-VERIFY-5538: validation rejects false live-quality gates and checksums."""

    artifact = mod.build_artifact(
        live_receipt=mod.claim_downgrade_receipt("unit_test_runtime_unavailable"),
        model_specs=_unit_model_specs(),
    )

    bad_claim = deepcopy(artifact)
    bad_claim["quality_claim_allowed"] = True
    bad_claim["reproducibility_checksum"] = mod.payload_checksum(bad_claim)
    with pytest.raises(ValueError, match="quality_claim_allowed"):
        mod.validate_artifact(bad_claim)

    bad_upstream = deepcopy(artifact)
    bad_upstream["upstream_panel_path"] = "results/other.json"
    bad_upstream["reproducibility_checksum"] = mod.payload_checksum(bad_upstream)
    with pytest.raises(ValueError, match="upstream_panel_path"):
        mod.validate_artifact(bad_upstream)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "bad"
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(bad_checksum)


def test_scenario_verify_5538_run_writes_requested_result_path(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5538: run writes the corrigendum JSON artifact."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    receipt = mod.claim_downgrade_receipt("unit_test_runtime_unavailable")
    artifact = mod.run(
        result_path=result_path,
        live_receipt=receipt,
        model_specs=_unit_model_specs(),
        write=True,
    )

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact["result_path"] == str(mod.RESULT_RELATIVE_PATH)

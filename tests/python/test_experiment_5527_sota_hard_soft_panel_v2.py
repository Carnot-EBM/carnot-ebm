"""Tests for Exp5527 exact-validated SOTA hard/soft panel v2.

Spec refs: REQ-VERIFY-5527, SCENARIO-VERIFY-5527.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5512_structured_output_positive_control as positive
from carnot import experiment_5527_sota_hard_soft_panel_v2 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5527_sota_hard_soft_panel_v2.py")


def _rows_from_payloads(payloads: list[dict]) -> list[dict]:
    return [{"parsed_payload": payload} for payload in payloads]


def test_req_verify_5527_spec_declares_panel_v2_contract() -> None:
    """REQ-VERIFY-5527: OpenSpec anchors the v2 exact-validation contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5527") : spec.index("### REQ-VERIFY-5501")]

    assert "SCENARIO-VERIFY-5527" in section
    assert str(mod.RESULT_RELATIVE_PATH) in section
    assert str(mod.UPSTREAM_REPAIR_RELATIVE_PATH) in section
    assert mod.INFERENCE_SUBSTRATE in section
    assert "AutoTokenizer.from_pretrained" in section
    for hf_id in positive.MANDATED_HEADLINE_MODEL_IDS:
        assert hf_id in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_verify_5527_default_panel_consumes_ready_repair_gate(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5527: ready repaired rows become exact-validated panel rows."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.run(
        result_path=result_path,
        tests_run=[{"command": str(TEST_PATH), "outcome": "passed"}],
    )
    written = json.loads(result_path.read_text(encoding="utf-8"))

    assert written == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["upstream_repair_loop_ready"] is True
    assert artifact["models_attempted"] == [positive.MANDATED_HEADLINE_MODEL_IDS[0]]
    assert artifact["rows_requested"] == 3
    assert artifact["rows_emitted"] == 3
    assert artifact["schema_validity_rate"] == pytest.approx(1.0)
    assert artifact["missing_candidate_rows"] == 0
    assert artifact["exact_validator_accuracy"] == pytest.approx(1.0)
    assert artifact["preference_optimality_rate"] == pytest.approx(1.0)
    assert artifact["abstention_rate"] == pytest.approx(1 / 3)
    assert artifact["confident_wrong_rate"] == pytest.approx(0.0)
    assert artifact["gpu_offload_evidence"]["gpu_offload_verified"] is True
    assert artifact["sota_structured_panel_ready"] is True
    assert artifact["sota_hard_soft_claim_allowed"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["no_autotokenizer_on_gguf"] is True

    mod.validate_artifact(artifact)


def test_req_verify_5527_recomputes_exact_validator_metrics() -> None:
    """REQ-VERIFY-5527: prior row-level verdict fields are not trusted."""

    fixture = positive.load_fixture_artifact()["fixture"]
    payloads = positive.build_fixture_candidate_payloads(fixture)
    bad_payloads = deepcopy(payloads)
    bad_payloads[0]["conclusion"]["assignment"]["support"] = "unsupported"
    rows = _rows_from_payloads(bad_payloads)
    rows[0].update(
        {
            "exact_validator_correct": True,
            "exact_validator_verdict": "exact_match",
            "soft_optimal": True,
            "reference_agreement": True,
        }
    )

    report = mod.evaluate_candidate_rows(
        rows,
        fixture=fixture,
        requested_instance_ids=[str(row["instance_id"]) for row in payloads],
    )

    assert report["rows_requested"] == 3
    assert report["rows_emitted"] == 3
    assert report["schema_validity_rate"] == pytest.approx(1.0)
    assert report["exact_validator_accuracy"] == pytest.approx(2 / 3)
    assert report["preference_optimality_rate"] == pytest.approx(1 / 2)
    assert report["abstention_rate"] == pytest.approx(1 / 3)
    assert report["confident_wrong_rate"] == pytest.approx(1 / 3)
    assert report["panel_rows"][0]["exact_validator_verdict"] == "hard_constraint_violation"
    assert report["panel_rows"][0]["exact_validator_correct"] is False


def test_scenario_verify_5527_missing_rows_are_not_abstentions() -> None:
    """SCENARIO-VERIFY-5527: absent expected rows stay missing, not abstained."""

    fixture = positive.load_fixture_artifact()["fixture"]
    payloads = positive.build_fixture_candidate_payloads(fixture)
    report = mod.evaluate_candidate_rows(
        _rows_from_payloads(payloads[:2]),
        fixture=fixture,
        requested_instance_ids=[str(row["instance_id"]) for row in payloads],
    )

    assert report["rows_requested"] == 3
    assert report["rows_emitted"] == 2
    assert report["schema_validity_rate"] == pytest.approx(2 / 3)
    assert report["missing_candidate_rows"] == 1
    assert report["exact_validator_accuracy"] == pytest.approx(1.0)
    assert report["preference_optimality_rate"] == pytest.approx(1.0)
    assert report["abstention_rate"] == pytest.approx(0.0)
    assert report["confident_wrong_rate"] == pytest.approx(0.0)


def test_req_verify_5527_validation_fails_closed() -> None:
    """REQ-VERIFY-5527: validation rejects false claim gates and bad checksums."""

    artifact = mod.build_artifact()

    missing = deepcopy(artifact)
    missing.pop("model_specs")
    with pytest.raises(ValueError, match="model_specs"):
        mod.validate_artifact(missing)

    bad_claim = deepcopy(artifact)
    bad_claim["exact_validator_accuracy"] = 0.5
    bad_claim["reproducibility_checksum"] = mod.payload_checksum(bad_claim)
    with pytest.raises(ValueError, match="exact_validator_accuracy"):
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


def test_req_verify_5527_blocked_and_fallback_paths_are_explicit(tmp_path: Path) -> None:
    """REQ-VERIFY-5527: closed gates and fallback evidence remain blocked."""

    missing_path_artifact = mod.build_artifact(repair_path=tmp_path / "missing.json")
    assert missing_path_artifact["upstream_repair_loadable"] is False
    assert "upstream_repair_artifact_not_loadable" in missing_path_artifact["readiness_blockers"]
    assert missing_path_artifact["sota_hard_soft_claim_allowed"] is False
    assert missing_path_artifact["honest_verdict"].startswith("blocked:")

    closed_gate = {
        "repair_loop_ready": False,
        "model_specs": [],
        "models_attempted": "legacy/model",
        "upstream_taxonomy_path": str(tmp_path / "missing_taxonomy.json"),
    }
    blocked = mod.build_artifact(upstream_repair_artifact=closed_gate, row_limit=0)
    assert blocked["rows_requested"] == 0
    assert blocked["model_specs"] == list(positive.MODEL_SPECS)
    assert blocked["gpu_offload_evidence"]["gpu_offload_verified"] is False
    assert "no_rows_requested" in blocked["readiness_blockers"]
    assert "no_mandated_sota_model_attempted" in blocked["readiness_blockers"]
    assert "gpu_offload_evidence_absent_or_false" in blocked["readiness_blockers"]
    assert mod._model_ids_match_mandated("not rows") is False

    direct_evidence = mod._gpu_offload_evidence(
        {"gpu_offload_evidence": {"gpu_offload_verified": True}}
    )
    assert direct_evidence["gpu_offload_verified"] is True
    assert direct_evidence["evidence_source"] == str(mod.UPSTREAM_REPAIR_RELATIVE_PATH)
    assert mod._bounded_candidate_rows(["not-a-row"], []) == []
    assert mod._record_instance_id({}) is None
    metadata_row: dict[str, object] = {}
    mod._attach_payload_metadata(metadata_row, {})
    assert metadata_row == {"conclusion_status": "", "candidate_confidence": 0.0}

    fixture = positive.load_fixture_artifact()["fixture"]
    payloads = positive.build_fixture_candidate_payloads(fixture)
    malformed_confidence = deepcopy(payloads[0])
    malformed_confidence["conclusion"]["confidence"] = "not-a-number"
    direct_report = mod.evaluate_candidate_rows(
        [malformed_confidence, malformed_confidence, payloads[1]],
        fixture=fixture,
    )
    assert direct_report["rows_emitted"] == 3
    assert direct_report["extra_emitted_rows"][0]["instance_id"] == payloads[0]["instance_id"]
    assert direct_report["panel_rows"][0]["candidate_confidence"] == pytest.approx(0.0)

    all_blockers = mod._readiness_blockers(
        upstream_loadable=False,
        upstream_ready=False,
        models_attempted=[],
        gpu_offload_evidence={"gpu_offload_verified": False},
        report={
            "rows_requested": 0,
            "missing_candidate_rows": 1,
            "schema_validity_rate": 0.0,
            "exact_validator_accuracy": 0.0,
            "preference_optimality_rate": 0.0,
            "confident_wrong_rate": 1.0,
        },
    )
    assert all_blockers == [
        "confident_wrong_rows",
        "exact_validator_mismatch",
        "gpu_offload_evidence_absent_or_false",
        "missing_candidate_rows",
        "no_mandated_sota_model_attempted",
        "no_rows_requested",
        "preference_suboptimal_or_unscored",
        "schema_invalid_or_missing_rows",
        "upstream_repair_artifact_not_loadable",
        "upstream_repair_loop_not_ready",
    ]

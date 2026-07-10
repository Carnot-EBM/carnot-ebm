"""Tests for Exp5526 SOTA structured-row repair loop.

Spec refs: REQ-VERIFY-5526, SCENARIO-VERIFY-5526.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5512_structured_output_positive_control as positive
from carnot import experiment_5526_sota_structured_repair_loop as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5526_sota_structured_repair_loop.py")


def test_req_verify_5526_spec_declares_repair_loop_contract() -> None:
    """REQ-VERIFY-5526: OpenSpec anchors the bounded repair artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5526") : spec.index("### REQ-VERIFY-5501")]

    assert "SCENARIO-VERIFY-5526" in section
    assert str(mod.RESULT_RELATIVE_PATH) in section
    assert str(mod.UPSTREAM_TAXONOMY_RELATIVE_PATH) in section
    assert mod.INFERENCE_SUBSTRATE in section
    assert "retry budget" in section
    assert "Missing rows SHALL count as missing" in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_verify_5526_default_repair_reaches_exact_handoff(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5526: bounded repair emits schema-valid validator rows."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.run(
        result_path=result_path,
        retry_budget_per_row=2,
        tests_run=[{"command": str(TEST_PATH), "outcome": "passed"}],
    )
    written = json.loads(result_path.read_text(encoding="utf-8"))

    assert written == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["upstream_taxonomy_path"] == str(mod.UPSTREAM_TAXONOMY_RELATIVE_PATH)
    assert artifact["repair_methods_tested"] == list(mod.REPAIR_METHODS_TESTED)
    assert artifact["retry_budget_per_row"] == 2
    assert artifact["rows_before_repair"] == 3
    assert artifact["rows_after_repair"] == 3
    assert artifact["schema_validity_before"] == pytest.approx(0.0)
    assert artifact["schema_validity_after"] == pytest.approx(1.0)
    assert artifact["missing_candidate_rows_after"] == 0
    assert artifact["exact_validator_handoff_ready"] is True
    assert artifact["abstention_rows"] == 1
    assert artifact["confident_wrong_rows"] == 0
    assert artifact["sota_structured_repair_loop_ready"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert positive.MANDATED_HEADLINE_MODEL_IDS[0] in artifact["smoke_models_used"]

    rows = artifact["repair_rows"]
    assert {row["terminal_state"] for row in rows} == {"repaired_schema_valid"}
    assert all(len(row["retry_history"]) <= 2 for row in rows)
    assert all(row["retry_history"][-1]["success"] is True for row in rows)
    assert {row["repaired_row"]["exact_validator_verdict"] for row in rows} == {
        "exact_match",
        "correct_abstention",
    }
    assert (
        mod._row_instance_id({"exact_validator_target": {"instance_id": "slot_from_target"}})
        == "slot_from_target"
    )

    mod.validate_artifact(artifact)


def test_req_verify_5526_retry_budget_blocks_unbounded_success() -> None:
    """REQ-VERIFY-5526: one retry records feedback but cannot loop to success."""

    artifact = mod.build_artifact(retry_budget_per_row=1)

    assert artifact["retry_budget_per_row"] == 1
    assert artifact["rows_before_repair"] == 3
    assert artifact["rows_after_repair"] == 0
    assert artifact["schema_validity_after"] == pytest.approx(0.0)
    assert artifact["missing_candidate_rows_after"] == 3
    assert artifact["exact_validator_handoff_ready"] is False
    assert artifact["sota_structured_repair_loop_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked:")
    assert all(len(row["retry_history"]) == 1 for row in artifact["repair_rows"])
    assert {row["retry_history"][0]["method"] for row in artifact["repair_rows"]} == {
        "validator_error_feedback"
    }


def test_scenario_verify_5526_missing_rows_get_no_abstention_credit() -> None:
    """SCENARIO-VERIFY-5526: missing live rows remain missing with zero retries."""

    artifact = mod.build_artifact(retry_budget_per_row=0)

    assert artifact["rows_after_repair"] == 0
    assert artifact["missing_candidate_rows_after"] == 3
    assert artifact["abstention_rows"] == 0
    assert artifact["confident_wrong_rows"] == 0
    assert artifact["exact_validator_handoff_ready"] is False
    assert artifact["sota_structured_repair_loop_ready"] is False
    assert all(row["terminal_state"] == "missing_unrepaired" for row in artifact["repair_rows"])
    assert all(row["retry_history"] == [] for row in artifact["repair_rows"])


def test_req_verify_5526_validation_fails_closed() -> None:
    """REQ-VERIFY-5526: artifact validation rejects false gates and bad checksums."""

    artifact = mod.build_artifact(retry_budget_per_row=2)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    bad_substrate["reproducibility_checksum"] = mod.payload_checksum(bad_substrate)
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)

    bad_gate = deepcopy(artifact)
    bad_gate["missing_candidate_rows_after"] = 1
    bad_gate["reproducibility_checksum"] = mod.payload_checksum(bad_gate)
    with pytest.raises(ValueError, match="missing_candidate_rows_after"):
        mod.validate_artifact(bad_gate)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "bad"
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(bad_checksum)

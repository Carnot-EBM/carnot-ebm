"""Tests for Exp 5362 .488 capstone synthesis.

Spec refs: REQ-CAPSTONE-5362, SCENARIO-CAPSTONE-5362,
SCENARIO-CAPSTONE-5362-BLOCKED-MISSING-INPUT,
SCENARIO-CAPSTONE-5362-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from shutil import copyfile

import pytest

from carnot import experiment_5362_capstone_v488 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/capstone/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _copy_expected_artifacts(root: Path) -> None:
    for source in mod.EXPECTED_ARTIFACTS:
        destination = root / source.relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        copyfile(REPO / source.relative_path, destination)


def test_req_capstone_5362_spec_declares_no_laundering_contract() -> None:
    """REQ-CAPSTONE-5362: OpenSpec anchors the .488 capstone contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-5362") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-CAPSTONE-5362",
        "SCENARIO-CAPSTONE-5362",
        "SCENARIO-CAPSTONE-5362-BLOCKED-MISSING-INPUT",
        "SCENARIO-CAPSTONE-5362-FIELD-PRINCIPLES",
        str(mod.RESULT_RELATIVE_PATH),
        str(mod.EXP5352_REQUESTED_ALIAS),
        str(mod.EXP5352.relative_path),
        mod.INFERENCE_SUBSTRATE,
        "source delta",
        "structured protocol",
        "constraint-tax panel",
        "token-probability feature audit",
        "carry diagnostic",
        "dependency provenance",
        "memory-tool drift",
        "self-learning scale-up",
        "solver projection",
        "p-bit schedules",
        "ARC level-up",
        "hardware continuity",
        "`hardware_speedup_claim=false`",
        "`active_roadmap_modified=false`",
        "`conductor_modified=false`",
    ):
        assert marker in section

    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_capstone_5362_builds_gate_table_without_laundering() -> None:
    """SCENARIO-CAPSTONE-5362: checked-in .488 artifacts produce honest gates."""

    artifact = mod.build_result_artifact(
        root=REPO,
        tests_run=[{"command": "unit capstone gate table", "outcome": "passed"}],
    )

    mod.validate_artifact(artifact)
    assert artifact["experiment_id"]["value"] == mod.EXPERIMENT_ID
    assert artifact["milestone"]["value"] == mod.MILESTONE
    assert artifact["status"]["value"] == "complete"
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert artifact["inference_substrate"]["value"] == mod.INFERENCE_SUBSTRATE
    assert len(artifact["artifacts_read"]["value"]) == len(mod.EXPECTED_ARTIFACTS)
    assert any(
        row["path"] == str(mod.EXP5352.relative_path) for row in artifact["artifacts_read"]["value"]
    )

    assert artifact["structured_protocol_clean"] is False
    assert artifact["constraint_tax_panel_ready"] is False
    assert artifact["tokenprob_feature_rows_ready"] is True
    assert artifact["carry_token_energy_signal_ready"] is False
    assert artifact["dependency_provenance_ready"] is True
    assert artifact["memory_tool_drift_ready"] is True
    assert artifact["self_learning_scaleup_ready"] is True
    assert artifact["solver_projection_ready"] is True
    assert artifact["pbit_schedule_signal_ready"] is True
    assert artifact["arc_new_level_banked"] is False
    assert artifact["hardware_speedup_claim"] is False
    assert artifact["active_roadmap_modified"] is False
    assert artifact["conductor_modified"] is False

    gates = {row["gate"]: row for row in artifact["gate_table"]["value"]}
    assert list(gates) == list(mod.GATE_ORDER)
    assert gates["source_delta"]["ready"] is True
    assert gates["source_delta"]["classification"] == "source_delta_complete_plan_unchanged"
    assert gates["structured_protocol"]["ready"] is False
    assert (
        gates["structured_protocol"]["classification"] == "blocked_structured_protocol_clean_false"
    )
    assert gates["constraint_tax_panel"]["ready"] is False
    assert gates["constraint_tax_panel"]["classification"] == "conductor_pre_gate_skipped"
    assert gates["constraint_tax_panel"]["evidence"]["blocked_at_layer"] == "conductor_pre_gate"
    assert gates["tokenprob_feature_audit"]["ready"] is True
    assert (
        gates["tokenprob_feature_audit"]["classification"]
        == "feature_rows_present_but_flagged_methodology"
    )
    assert gates["tokenprob_feature_audit"]["claim_boundary"] == (
        "feature rows only; flagged methodology means no token-energy or quality claim"
    )
    assert gates["carry_diagnostic"]["ready"] is False
    assert (
        gates["carry_diagnostic"]["classification"] == "blocked_and_flagged_carry_signal_not_ready"
    )
    assert gates["dependency_provenance"]["ready"] is True
    assert gates["memory_tool_drift"]["ready"] is True
    assert gates["self_learning_scaleup"]["ready"] is True
    assert gates["solver_projection"]["ready"] is True
    assert gates["pbit_schedules"]["ready"] is True
    assert gates["pbit_schedules"]["evidence"]["hardware_speedup_claim"] is False
    assert gates["arc_level_up"]["ready"] is False
    assert gates["arc_level_up"]["classification"] == "honest_null_no_new_level_banked"
    assert gates["hardware_continuity"]["ready"] is True
    assert gates["hardware_continuity"]["classification"] == (
        "partial_continuity_polarfire_workload_kv260_gatemate_blocked_no_speedup"
    )
    assert gates["hardware_continuity"]["evidence"]["speedup_claim"] is False
    assert gates["hardware_continuity"]["evidence"]["polarfire_workload_validated"] is True
    assert gates["hardware_continuity"]["evidence"]["kv260_ssh_reachable"] is False

    issues = artifact["missing_blocked_flagged_or_skipped_artifacts"]["value"]
    classifications = {(row["experiment_number"], row["classification"]) for row in issues}
    assert (5349, "blocked") in classifications
    assert (5351, "blocked") in classifications
    assert (5352, "requested_alias_missing") in classifications
    assert (5352, "conductor_gate_skip") in classifications
    assert (5353, "flagged") in classifications
    assert (5354, "blocked_and_flagged") in classifications
    assert (5360, "honest_null") in classifications
    assert (5361, "hardware_subgate_blocked") in classifications

    recommendation = artifact["next_milestone_recommendation"]["value"]
    assert (
        recommendation["recommendation"]
        == "structured_sota_protocol_repair_then_constraint_tax_panel"
    )
    assert "constraint_tax_metrics" in recommendation["do_not_claim"]
    assert "token_probability_energy_signal" in recommendation["do_not_claim"]
    assert "hardware_speedup" in recommendation["do_not_claim"]
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_scenario_capstone_5362_run_writes_stable_artifact(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5362: run writes deterministic capstone JSON."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    tests_run = [{"command": "unit run", "outcome": "passed"}]
    artifact = mod.run(root=REPO, result_path=result_path, tests_run=tests_run)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact == mod.build_result_artifact(root=REPO, tests_run=tests_run)
    mod.validate_artifact(artifact)


def test_scenario_capstone_5362_missing_artifact_blocks_without_claims(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-5362-BLOCKED-MISSING-INPUT: missing inputs fail closed."""

    _copy_expected_artifacts(tmp_path)
    (tmp_path / mod.EXP5358.relative_path).unlink()
    (tmp_path / mod.EXP5361.relative_path).unlink()
    (tmp_path / mod.EXP5355.relative_path).write_text("{", encoding="utf-8")
    (tmp_path / mod.EXP5359.relative_path).write_text("[]", encoding="utf-8")

    artifact = mod.build_result_artifact(root=tmp_path)

    mod.validate_artifact(artifact)
    assert artifact["status"]["value"] == "blocked_missing_required"
    assert artifact["honest_verdict"]["value"].startswith("blocked_missing_required")
    assert artifact["dependency_provenance_ready"] is False
    assert artifact["self_learning_scaleup_ready"] is False
    assert artifact["solver_projection_ready"] is False
    assert artifact["pbit_schedule_signal_ready"] is False
    assert artifact["hardware_speedup_claim"] is False
    assert artifact["active_roadmap_modified"] is False
    assert artifact["conductor_modified"] is False
    rows = artifact["missing_blocked_flagged_or_skipped_artifacts"]["value"]
    assert any(
        row["experiment_number"] == 5358 and row["classification"] == "missing" for row in rows
    )
    assert any(
        row["experiment_number"] == 5355
        and row["classification"] == "malformed"
        and row["reason"].startswith("malformed_json:")
        for row in rows
    )
    assert any(
        row["experiment_number"] == 5359
        and row["classification"] == "malformed"
        and row["reason"] == "not_json_object"
        for row in rows
    )
    assert any(
        row["experiment_number"] == 5361 and row["classification"] == "missing" for row in rows
    )
    assert any(
        row["experiment_number"] == 5352 and row["classification"] == "requested_alias_missing"
        for row in rows
    )


def test_scenario_capstone_5362_defensive_nonblocking_branches(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5362: optional alias and clean subgates do not add issues."""

    _copy_expected_artifacts(tmp_path)
    alias_path = tmp_path / mod.EXP5352_REQUESTED_ALIAS
    alias_path.write_text("{}", encoding="utf-8")

    arc_path = tmp_path / mod.EXP5360.relative_path
    arc_payload = json.loads(arc_path.read_text(encoding="utf-8"))
    arc_payload["new_level_banked"] = True
    arc_payload["offline_reproduced"] = True
    arc_path.write_text(json.dumps(arc_payload), encoding="utf-8")

    hardware_path = tmp_path / mod.EXP5361.relative_path
    hardware_payload = json.loads(hardware_path.read_text(encoding="utf-8"))
    hardware_payload["kv260_status"]["value"]["ssh_reachable"] = True
    hardware_payload["kv260_status"]["value"]["status"] = "reachable_ssh_status_only"
    hardware_payload["gatemate_status"]["value"]["status"] = "visible_ready_no_workload"
    hardware_payload["blocked_reason"] = {"KV260": None, "GateMate": None, "PolarFire": None}
    hardware_path.write_text(json.dumps(hardware_payload), encoding="utf-8")

    artifact = mod.build_result_artifact(root=tmp_path)

    mod.validate_artifact(artifact)
    rows = artifact["missing_blocked_flagged_or_skipped_artifacts"]["value"]
    classifications = {(row["experiment_number"], row["classification"]) for row in rows}
    assert (5352, "requested_alias_missing") not in classifications
    assert (5360, "honest_null") not in classifications
    assert (5361, "hardware_subgate_blocked") not in classifications
    assert artifact["arc_new_level_banked"] is True
    gates = {row["gate"]: row for row in artifact["gate_table"]["value"]}
    assert gates["hardware_continuity"]["classification"] == (
        "hardware_continuity_workload_no_speedup"
    )


def test_req_capstone_5362_validation_rejects_schema_drift() -> None:
    """REQ-CAPSTONE-5362: validator rejects false claims and schema drift."""

    artifact = mod.build_result_artifact(root=REPO)

    missing_field = deepcopy(artifact)
    missing_field.pop("artifacts_read")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing_field)

    bad_wrapped = deepcopy(artifact)
    bad_wrapped["gate_table"] = bad_wrapped["gate_table"]["value"]
    with pytest.raises(ValueError, match="gate_table"):
        mod.validate_artifact(bad_wrapped)

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"]["value"] = "done"
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_verdict)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"]["value"] = "live_llm_inference"
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)

    bad_speedup = deepcopy(artifact)
    bad_speedup["hardware_speedup_claim"] = True
    with pytest.raises(ValueError, match="hardware_speedup_claim"):
        mod.validate_artifact(bad_speedup)

    bad_bool = deepcopy(artifact)
    bad_bool["structured_protocol_clean"] = {"value": False}
    with pytest.raises(ValueError, match="structured_protocol_clean"):
        mod.validate_artifact(bad_bool)

    bad_gate_order = deepcopy(artifact)
    bad_gate_order["gate_table"]["value"] = list(reversed(bad_gate_order["gate_table"]["value"]))
    with pytest.raises(ValueError, match="gate_table"):
        mod.validate_artifact(bad_gate_order)

    bad_roadmap = deepcopy(artifact)
    bad_roadmap["active_roadmap_modified"] = True
    with pytest.raises(ValueError, match="active_roadmap_modified"):
        mod.validate_artifact(bad_roadmap)

    bad_conductor = deepcopy(artifact)
    bad_conductor["conductor_modified"] = True
    with pytest.raises(ValueError, match="conductor_modified"):
        mod.validate_artifact(bad_conductor)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:not-real"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)


def test_req_capstone_5362_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-CAPSTONE-5362: checked-in deliverable is a stable capstone."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = mod.build_result_artifact(root=REPO, tests_run=result["tests_run"]["value"])

    assert result == replay
    mod.validate_artifact(result)
    assert result["honest_verdict"]["value"].startswith(("complete:", "blocked_"))
    assert result["hardware_speedup_claim"] is False

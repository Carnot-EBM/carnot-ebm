"""Tests for Exp 5388 .490 capstone synthesis.

Spec refs: REQ-CAPSTONE-5388, SCENARIO-CAPSTONE-5388,
SCENARIO-CAPSTONE-5388-MISSING-OR-GATED-INPUT,
SCENARIO-CAPSTONE-5388-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from shutil import copyfile

import pytest

from carnot import experiment_5388_capstone_v490 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/capstone/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _copy_available_inputs(root: Path) -> None:
    for relative in mod.EXPECTED_ARTIFACT_PATHS:
        source = REPO / relative
        if not source.exists():
            continue
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        copyfile(source, destination)


def test_req_capstone_5388_spec_declares_v490_closeout_contract() -> None:
    """REQ-CAPSTONE-5388: OpenSpec anchors the .490 capstone contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-5388") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-CAPSTONE-5388",
        "SCENARIO-CAPSTONE-5388",
        "SCENARIO-CAPSTONE-5388-MISSING-OR-GATED-INPUT",
        "SCENARIO-CAPSTONE-5388-FIELD-PRINCIPLES",
        str(mod.RESULT_RELATIVE_PATH),
        "Exp5376 through Exp5387",
        "gate-skipped or gate-blocked artifact",
        "honest ARC no-bank",
        "closed token/internal-feature backend gate",
        "`hardware_speedup_claim` = \"must be false unless Exp5386 has repeatable board",
        "`active_roadmap_modified` = \"must be false",
        "`conductor_modified` = \"must be",
    ):
        assert marker in section

    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_capstone_5388_builds_default_artifact_without_laundering() -> None:
    """SCENARIO-CAPSTONE-5388: checked-in .490 artifacts produce honest gates."""

    artifact = mod.build_artifact(
        root=REPO,
        tests_run=[{"command": "unit capstone synthesis", "outcome": "passed"}],
    )

    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["milestone"] == mod.MILESTONE
    assert artifact["expected_artifacts"] == list(mod.EXPECTED_ARTIFACT_PATHS)
    assert artifact["artifacts_found"] == list(mod.EXPECTED_ARTIFACT_PATHS)
    assert artifact["artifacts_missing"] == []
    assert artifact["skipped_by_gate"] == {}

    assert artifact["structured_methodology_receipt_ready"] is True
    assert artifact["structured_protocol_clean"] is True
    assert artifact["constraint_tax_panel_ready"] is True
    assert artifact["budget_memory_corrigendum_clean"] is True
    assert artifact["continuous_self_learning_real_workflow_ready"] is True
    assert artifact["continuous_self_learning_requirement_satisfied"] is True
    assert artifact["overwrite_guidance_scale_ready"] is True
    assert artifact["pbit_boundary_overwrite_ready"] is True
    assert artifact["arc_new_level_banked"] is False
    assert artifact["hardware_hash_chained_receipt_ready"] is True
    assert artifact["hardware_speedup_claim"] is False
    assert artifact["future_token_signal_allowed"] is False
    assert artifact["active_roadmap_modified"] is False
    assert artifact["conductor_modified"] is False

    phases = {row["lane"]: row for row in artifact["phase_summaries"]}
    assert phases["structured_sota"]["outcome"] == "clean_ready"
    assert phases["constraint_tax"]["outcome"] == "ready"
    assert phases["continuous_self_learning"]["outcome"] == "requirement_satisfied"
    assert phases["solver_guidance"]["claim_boundary"] == "ready_but_flagged_adversarial"
    assert phases["pbit_boundary_overwrite"]["claim_boundary"] == "cpu_only_no_hardware_speedup"
    assert phases["arc_geometric_salience"]["outcome"] == "honest_null_no_level_banked"
    assert phases["hardware"]["claim_boundary"] == "hash_chained_receipts_no_repeatable_speedup"
    assert phases["token_backend"]["outcome"] == "closed_no_backend_signal"

    blocked = {row["lane"]: row for row in artifact["retired_or_blocked_lanes"]}
    assert blocked["overwrite_guidance_scale"]["state"] == "blocked_flagged_adversarial"
    assert blocked["arc_geometric_salience_live_path"]["state"] == "blocked_no_bank"
    assert blocked["token_internal_feature_signal"]["state"] == "retired_until_backend_features"
    assert blocked["hardware_speedup_claim"]["state"] == "blocked_on_repeatable_board_timing"
    assert blocked["kv260_workload"]["state"] == "blocked_unreachable"
    assert blocked["gatemate_workload"]["state"] == "blocked_physical_or_jtag"

    actions = [row["action"] for row in artifact["next_milestone_recommendations"]]
    assert actions == [
        "carry_clean_structured_constraint_tax_into_v491",
        "reconcile_or_rerun_flagged_overwrite_guidance",
        "convert_arc_geometric_salience_from_no_bank_to_levelup",
        "keep_token_backend_closed_until_real_features",
        "get_repeatable_board_timing_before_speedup_claims",
    ]
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_scenario_capstone_5388_run_writes_stable_json(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5388: run writes deterministic capstone JSON."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    tests_run = [{"command": "unit run", "outcome": "passed"}]

    artifact = mod.run(root=REPO, result_path=result_path, tests_run=tests_run)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact == mod.build_artifact(root=REPO, tests_run=tests_run)
    mod.validate_artifact(artifact)


def test_scenario_capstone_5388_missing_and_gated_inputs_fail_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-5388-MISSING-OR-GATED-INPUT: gaps never imply success."""

    _copy_available_inputs(tmp_path)
    (tmp_path / mod.EXP5382).unlink()
    (tmp_path / mod.EXP5377).write_text("[]", encoding="utf-8")
    gated_path = tmp_path / mod.EXP5380
    gated_path.write_text(
        json.dumps(
            {
                "status": "blocked",
                "constraint_tax_panel_ready": True,
                "blocked_at_layer": "conductor_pre_gate",
                "honest_verdict": "blocked_gate_check_failed",
                "upstream_gate": {
                    "all_passed": False,
                    "failed_gates": ["structured_protocol_clean"],
                    "source_artifact": mod.EXP5379,
                    "source_structured_protocol_clean": False,
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / mod.EXP5387).write_text("{", encoding="utf-8")

    artifact = mod.build_artifact(root=tmp_path)

    mod.validate_artifact(artifact)
    assert artifact["status"] == "honest_partial"
    assert mod.EXP5377 in artifact["artifacts_missing"]
    assert mod.EXP5382 in artifact["artifacts_missing"]
    assert mod.EXP5387 in artifact["artifacts_missing"]
    assert mod.EXP5380 in artifact["artifacts_found"]
    assert artifact["constraint_tax_panel_ready"] is False
    assert artifact["continuous_self_learning_real_workflow_ready"] is False
    assert artifact["continuous_self_learning_requirement_satisfied"] is False
    assert artifact["future_token_signal_allowed"] is False
    assert artifact["hardware_speedup_claim"] is False

    skipped = artifact["skipped_by_gate"]["exp5380-constraint-tax-tool-action-panel-v3-v490"]
    assert skipped["source_artifact"] == mod.EXP5380
    assert skipped["blocked_at_layer"] == "conductor_pre_gate"
    assert skipped["gate_conditions"]["failed_gates"] == ["structured_protocol_clean"]
    assert skipped["gate_conditions"]["source_structured_protocol_clean"] is False

    unreadable = {row["path"]: row for row in artifact["artifact_read_errors"]}
    assert unreadable[mod.EXP5377]["classification"] == "not_json_object"
    assert unreadable[mod.EXP5387]["classification"].startswith("malformed_json:")


def test_req_capstone_5388_validation_rejects_schema_drift() -> None:
    """REQ-CAPSTONE-5388: validator rejects false claims and schema drift."""

    artifact = mod.build_artifact(root=REPO)

    missing_field = deepcopy(artifact)
    missing_field.pop("expected_artifacts")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing_field)

    bad_milestone = deepcopy(artifact)
    bad_milestone["milestone"] = "2026.07.489"
    with pytest.raises(ValueError, match="milestone"):
        mod.validate_artifact(bad_milestone)

    bad_status = deepcopy(artifact)
    bad_status["status"] = "done"
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(bad_status)

    bad_missing = deepcopy(artifact)
    bad_missing["artifacts_missing"] = [mod.EXP5382]
    bad_missing["status"] = "complete"
    with pytest.raises(ValueError, match="honest_partial"):
        mod.validate_artifact(bad_missing)

    bad_expected = deepcopy(artifact)
    bad_expected["expected_artifacts"] = []
    with pytest.raises(ValueError, match="expected_artifacts"):
        mod.validate_artifact(bad_expected)

    bad_bool = deepcopy(artifact)
    bad_bool["structured_protocol_clean"] = {"value": True}
    with pytest.raises(ValueError, match="structured_protocol_clean"):
        mod.validate_artifact(bad_bool)

    bad_csl = deepcopy(artifact)
    bad_csl["continuous_self_learning_real_workflow_ready"] = False
    with pytest.raises(ValueError, match="continuous self-learning"):
        mod.validate_artifact(bad_csl)

    bad_speedup = deepcopy(artifact)
    bad_speedup["hardware_speedup_claim"] = True
    with pytest.raises(ValueError, match="hardware_speedup_claim"):
        mod.validate_artifact(bad_speedup)

    bad_roadmap = deepcopy(artifact)
    bad_roadmap["active_roadmap_modified"] = True
    with pytest.raises(ValueError, match="active_roadmap_modified"):
        mod.validate_artifact(bad_roadmap)

    bad_conductor = deepcopy(artifact)
    bad_conductor["conductor_modified"] = True
    with pytest.raises(ValueError, match="conductor_modified"):
        mod.validate_artifact(bad_conductor)

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"] = "done"
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_verdict)

    bad_principles = deepcopy(artifact)
    bad_principles["field_principles"] = {}
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(bad_principles)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:not-real"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)


def test_capstone_5388_helper_branches_are_fail_closed() -> None:
    """SCENARIO-CAPSTONE-5388-MISSING-OR-GATED-INPUT: helpers default closed."""

    assert mod._status(None) == "missing"
    assert mod._verdict(None) == "missing"
    assert mod._is_gate_blocked(None) is True
    assert mod._source_bool({"status": "complete"}, "missing_field") is False
    assert mod._source_number(None, "methodology_duration_s", default=7.0) == 7.0
    assert mod._source_number({"methodology_duration_s": "bad"}, "methodology_duration_s") == 0.0

    skipped = mod._skipped_by_gate(
        {
            mod.EXP5382: {
                "status": "blocked",
                "blocked_at_layer": "conductor_pre_gate",
                "honest_verdict": "blocked_gate_check_failed",
            }
        }
    )
    gate = skipped["exp5382-real-workflow-continuous-self-learning-v490"]
    assert gate["gate_conditions"] == {"required": mod.GATED_TASKS[mod.EXP5382]["requires"]}


def test_req_capstone_5388_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-CAPSTONE-5388: checked-in deliverable is a stable capstone."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = mod.build_artifact(root=REPO, tests_run=result["tests_run"])

    assert result == replay
    mod.validate_artifact(result)
    assert result["honest_verdict"].startswith(("complete:", "honest_partial:"))

"""Tests for Exp 5348 .487 capstone synthesis.

Spec refs: REQ-CAPSTONE-5348, SCENARIO-CAPSTONE-5348,
SCENARIO-CAPSTONE-5348-BLOCKED-MISSING-INPUT,
SCENARIO-CAPSTONE-5348-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from shutil import copyfile

import pytest

from carnot import experiment_5348_capstone_v487 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/capstone/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _copy_expected_artifacts(root: Path) -> None:
    for source in mod.EXPECTED_ARTIFACTS:
        destination = root / source.relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        copyfile(REPO / source.relative_path, destination)


def test_req_capstone_5348_spec_declares_no_laundering_contract() -> None:
    """REQ-CAPSTONE-5348: OpenSpec anchors the .487 capstone contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-CAPSTONE-5348") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-CAPSTONE-5348",
        "SCENARIO-CAPSTONE-5348",
        "SCENARIO-CAPSTONE-5348-BLOCKED-MISSING-INPUT",
        "SCENARIO-CAPSTONE-5348-FIELD-PRINCIPLES",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "runtime",
        "structured-output protocol",
        "SOTA bounded quality",
        "self-learning scale-up",
        "token-probability energy",
        "KAN bridge",
        "`hardware_speedup_claim=false`",
        "`active_roadmap_modified=false`",
        "`conductor_modified=false`",
    ):
        assert marker in section

    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_capstone_5348_builds_flag_aware_gate_table() -> None:
    """SCENARIO-CAPSTONE-5348: checked-in .487 artifacts produce honest gates."""

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
        row["path"] == "results/experiment_5340_utility_weighted_context_memory_q_values_v487.json"
        for row in artifact["artifacts_read"]["value"]
    )

    assert artifact["runtime_clean"] is True
    assert artifact["structured_output_protocol_ready"] is False
    assert artifact["bounded_sota_quality_usable"] is False
    assert artifact["utility_memory_ready"] is True
    assert artifact["bounded_compressor_ready"] is True
    assert artifact["self_learning_scaleup_ready"] is False
    assert artifact["qstr_fixture_ready"] is True
    assert artifact["solver_guidance_ready"] is True
    assert artifact["internal_energy_corrigendum_clean"] is False
    assert artifact["kan_constraint_bridge_ready"] is True
    assert artifact["hardware_speedup_claim"] is False
    assert artifact["active_roadmap_modified"] is False
    assert artifact["conductor_modified"] is False

    gates = {row["gate"]: row for row in artifact["gate_table"]["value"]}
    assert gates["runtime"]["ready"] is True
    assert gates["runtime"]["claim_boundary"] == "runtime receipt only; no SOTA quality claim"
    assert gates["structured_output_protocol"]["ready"] is False
    assert gates["structured_output_protocol"]["classification"] == "flagged_parse_only_protocol_candidate"
    assert gates["structured_output_protocol"]["evidence"]["reported_ready"] is True
    assert gates["sota_bounded_quality"]["ready"] is False
    assert gates["sota_bounded_quality"]["classification"] == "blocked_quality_panel_not_usable"
    assert gates["utility_memory"]["ready"] is True
    assert gates["bounded_compressor"]["ready"] is True
    assert gates["self_learning_scaleup"]["ready"] is False
    assert gates["self_learning_scaleup"]["classification"] == "flagged_scaleup_not_claimable"
    assert gates["qstr_fixture"]["ready"] is True
    assert gates["solver_guidance"]["ready"] is True
    assert gates["token_probability_energy"]["ready"] is False
    assert gates["token_probability_energy"]["classification"] == "blocked_and_flagged_energy_corrigendum"
    assert gates["kan_constraint_bridge"]["ready"] is True
    assert gates["kan_constraint_bridge"]["claim_boundary"] == "bounded explicit cuts only; no broad KAN certificate claim"
    assert gates["hardware"]["ready"] is True
    assert gates["hardware"]["classification"] == "continuity_workload_receipt_no_speedup"
    assert gates["hardware"]["evidence"]["speedup_claim"] is False

    flagged_or_blocked = artifact["missing_blocked_flagged_or_skipped_artifacts"]["value"]
    classifications = {
        (row["experiment_number"], row["classification"]) for row in flagged_or_blocked
    }
    assert (5335, "blocked") in classifications
    assert (5338, "flagged") in classifications
    assert (5339, "blocked") in classifications
    assert (5342, "flagged") in classifications
    assert (5345, "blocked_and_flagged") in classifications

    recommendation = artifact["next_milestone_recommendation"]["value"]
    assert recommendation["recommendation"] == "token-energy cleanup"
    assert "headline_sota_quality" in recommendation["do_not_claim"]
    assert "hardware_speedup" in recommendation["do_not_claim"]
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_scenario_capstone_5348_run_writes_stable_artifact(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-5348: run writes deterministic capstone JSON."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    tests_run = [{"command": "unit run", "outcome": "passed"}]
    artifact = mod.run(root=REPO, result_path=result_path, tests_run=tests_run)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact == mod.build_result_artifact(root=REPO, tests_run=tests_run)
    mod.validate_artifact(artifact)


def test_scenario_capstone_5348_missing_artifact_blocks_without_claims(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-5348-BLOCKED-MISSING-INPUT: missing inputs fail closed."""

    _copy_expected_artifacts(tmp_path)
    (tmp_path / mod.EXP5345.relative_path).unlink()
    (tmp_path / mod.EXP5338.relative_path).write_text("{", encoding="utf-8")
    (tmp_path / mod.EXP5340_Q_VALUES.relative_path).write_text("[]", encoding="utf-8")

    artifact = mod.build_result_artifact(root=tmp_path)

    mod.validate_artifact(artifact)
    assert artifact["status"]["value"] == "blocked_missing_required"
    assert artifact["honest_verdict"]["value"].startswith("blocked_missing_required")
    assert artifact["structured_output_protocol_ready"] is False
    assert artifact["internal_energy_corrigendum_clean"] is False
    assert artifact["hardware_speedup_claim"] is False
    assert artifact["active_roadmap_modified"] is False
    assert artifact["conductor_modified"] is False
    blocked_rows = artifact["missing_blocked_flagged_or_skipped_artifacts"]["value"]
    assert any(
        row["experiment_number"] == 5345 and row["classification"] == "missing"
        for row in blocked_rows
    )
    assert any(
        row["experiment_number"] == 5338
        and row["classification"] == "malformed"
        and row["reason"].startswith("malformed_json:")
        for row in blocked_rows
    )
    assert any(
        row["experiment_number"] == 5340
        and row["classification"] == "malformed"
        and row["reason"] == "not_json_object"
        and row["path"].endswith("q_values_v487.json")
        for row in blocked_rows
    )


def test_req_capstone_5348_validation_rejects_schema_drift() -> None:
    """REQ-CAPSTONE-5348: validator rejects false claims and schema drift."""

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
    bad_bool["runtime_clean"] = {"value": True}
    with pytest.raises(ValueError, match="runtime_clean"):
        mod.validate_artifact(bad_bool)

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


def test_req_capstone_5348_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-CAPSTONE-5348: checked-in deliverable is a stable capstone."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = mod.build_result_artifact(root=REPO, tests_run=result["tests_run"]["value"])

    assert result == replay
    mod.validate_artifact(result)
    assert result["honest_verdict"]["value"].startswith(("complete:", "blocked_"))
    assert result["hardware_speedup_claim"] is False

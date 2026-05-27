"""Tests for Exp 3213 repair-gate decision v6.

Spec refs: REQ-VERIFY-3213, SCENARIO-VERIFY-3213.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import repair_gate_decision_v6 as mod


REQUIRED_FIELDS = {
    "schema_version",
    "experiment_id",
    "milestone",
    "required_artifacts",
    "missing_artifacts",
    "receipt_gate_passed",
    "clean_verifier_gate_passed",
    "structured_proposal_gate_passed",
    "auxiliary_fixture_artifacts",
    "repair_gate_state",
    "repair_ladder_allowed",
    "blockers",
    "conductor_file_modified",
    "active_roadmap_modified",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text_sources(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "AGENTS.md").write_text("Read CODEX.md before changes.\n", encoding="utf-8")
    (root / "CODEX.md").write_text("Spec First\nWrite Tests First\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("Verifier authenticity discipline.\n", encoding="utf-8")
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "ops/conductor-log.md").write_text("exp3208 blocked\n", encoding="utf-8")
    spec = root / mod.SPEC_REL_PATH
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(
        "REQ-VERIFY-3213\nSCENARIO-VERIFY-3213\n"
        "results/experiment_3213_repair_gate_decision_v6.json\n",
        encoding="utf-8",
    )


def _write_standard_sources(
    root: Path,
    *,
    receipt: dict[str, Any] | None = None,
    clean: dict[str, Any] | None = None,
    proposal: dict[str, Any] | None = None,
    aux3210: dict[str, Any] | None = None,
    aux3211: dict[str, Any] | None = None,
    include_receipt: bool = True,
    include_clean: bool = True,
    include_proposal: bool = True,
    include_auxiliary: bool = True,
) -> None:
    _write_text_sources(root)
    if include_receipt:
        payload = {
            "experiment_id": "exp3208",
            "schema_version": "carnot.full_local_sota_receipt.v5",
            "status": "complete",
            "clean_rerun_allowed": True,
            "honest_verdict": "complete: receipt fixture",
        }
        if receipt:
            payload.update(receipt)
        _write_json(root, mod.EXP3208_REL_PATH, payload)
    if include_clean:
        payload = {
            "experiment_id": "exp3209",
            "schema_version": "carnot.clean_live_sota_verifier_rerun.v12",
            "status": "complete",
            "clean_verifier_state": "clean",
            "flagged_adversarial": False,
            "unhandled_adversarial_methodology_flags": [],
            "honest_verdict": "complete: clean verifier fixture",
        }
        if clean:
            payload.update(clean)
        _write_json(root, mod.EXP3209_REL_PATH, payload)
    if include_proposal:
        payload = {
            "experiment_id": "exp3212",
            "schema_version": "carnot.structured_repair_proposal_preflight.v1",
            "ready_for_repair_gate": True,
            "repair_correctness_claimed": False,
            "honest_verdict": "complete: proposal preflight fixture",
        }
        if proposal:
            payload.update(proposal)
        _write_json(root, mod.EXP3212_REL_PATH, payload)
    if include_auxiliary:
        payload3210 = {
            "experiment_id": "exp3210",
            "schema_version": "carnot.context_cot_clbench_parametric_shortcut_fixtures.v1",
            "ready_for_clean_verifier": True,
            "fixture_count": 30,
            "honest_verdict": "complete: context fixtures",
        }
        if aux3210:
            payload3210.update(aux3210)
        _write_json(root, mod.EXP3210_REL_PATH, payload3210)
        payload3211 = {
            "experiment_id": "exp3211",
            "schema_version": "carnot.constraintbench_feasibility_objective_pilot.v1",
            "ready_for_clean_verifier": True,
            "fixture_count": 15,
            "honest_verdict": "complete: constraint fixtures",
        }
        if aux3211:
            payload3211.update(aux3211)
        _write_json(root, mod.EXP3211_REL_PATH, payload3211)
    _write_json(
        root,
        mod.EXP3198_REL_PATH,
        {
            "experiment_id": "exp3198",
            "schema_version": "carnot.repair_gate_decision.v5",
            "repair_gate_state": "blocked_clean_verifier_gate_skipped",
            "honest_verdict": "complete: prior v5 fixture",
        },
    )


def test_req_verify_3213_spec_anchor_exists() -> None:
    """REQ-VERIFY-3213: OpenSpec declares the v6 repair gate artifact."""

    spec = (mod.REPO_ROOT / mod.SPEC_REL_PATH).read_text(encoding="utf-8")

    assert "REQ-VERIFY-3213" in spec
    assert "SCENARIO-VERIFY-3213" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "repair_gate_state=unblocked" in spec


def test_scenario_verify_3213_unblocks_only_when_mandatory_gates_pass(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-3213: all three mandatory gates are required to unblock."""

    _write_standard_sources(tmp_path)

    output = mod.write_artifact(tmp_path, output_path=Path("results/out.json"), tests_run=["focused"])
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / "results/out.json"
    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3213"
    assert artifact["milestone"] == "2026.05.297"
    assert artifact["missing_artifacts"] == []
    assert artifact["receipt_gate_passed"] is True
    assert artifact["clean_verifier_gate_passed"] is True
    assert artifact["structured_proposal_gate_passed"] is True
    assert artifact["repair_gate_state"] == "unblocked"
    assert artifact["repair_ladder_allowed"] is True
    assert artifact["blockers"] == []
    assert artifact["conductor_file_modified"] is False
    assert artifact["active_roadmap_modified"] is False
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["tests_run"] == ["focused"]
    assert {row["role"] for row in artifact["auxiliary_fixture_artifacts"]} == {
        "exp3210_context_cot_clbench_parametric_shortcut_fixtures_v1",
        "exp3211_constraintbench_feasibility_objective_pilot_v1",
    }

    mod.validate_artifact(artifact)


def test_req_verify_3213_blocked_shape_records_missing_and_failed_upstreams(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3213: current-style gate-skips stay blocked with explicit evidence."""

    _write_standard_sources(
        tmp_path,
        receipt={
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "1 of 1 gate(s) failed; exp3207.cuda_receipt_ready",
            "clean_rerun_allowed": False,
        },
        include_clean=False,
        include_proposal=False,
    )

    artifact = mod.build_artifact(tmp_path)

    assert artifact["receipt_gate_passed"] is False
    assert artifact["clean_verifier_gate_passed"] is False
    assert artifact["structured_proposal_gate_passed"] is False
    assert artifact["repair_gate_state"] == "blocked"
    assert artifact["repair_ladder_allowed"] is False
    assert mod.EXP3209_REL_PATH.as_posix() in artifact["missing_artifacts"]
    assert mod.EXP3212_REL_PATH.as_posix() in artifact["missing_artifacts"]

    codes = {row["code"] for row in artifact["blockers"]}
    assert "exp3208_clean_rerun_not_allowed" in codes
    assert "missing_mandatory_artifact" in codes
    assert "exp3209_clean_verifier_state_not_clean" in codes
    assert "exp3212_ready_for_repair_gate_not_true" in codes
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_verify_3213_auxiliary_missing_is_nonblocking_but_invalid_blocks(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3213: auxiliary fixtures block only when present and invalid."""

    _write_standard_sources(tmp_path, include_auxiliary=False)

    missing_aux = mod.build_artifact(tmp_path)

    assert missing_aux["repair_gate_state"] == "unblocked"
    assert missing_aux["repair_ladder_allowed"] is True
    assert mod.EXP3210_REL_PATH.as_posix() in missing_aux["missing_artifacts"]
    assert mod.EXP3211_REL_PATH.as_posix() in missing_aux["missing_artifacts"]
    assert {row["code"] for row in missing_aux["blockers"]} == set()

    invalid_root = tmp_path / "invalid"
    _write_standard_sources(invalid_root, aux3211={"invalidity": "schema_mismatch"})

    invalid_aux = mod.build_artifact(invalid_root)

    assert invalid_aux["repair_gate_state"] == "blocked"
    assert invalid_aux["repair_ladder_allowed"] is False
    assert any(row["code"] == "auxiliary_fixture_invalid" for row in invalid_aux["blockers"])


def test_scenario_verify_3213_diagnostic_only_when_no_mandatory_artifact_readable(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-3213: no readable mandatory gates yields diagnostic-only."""

    _write_text_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path)

    assert artifact["repair_gate_state"] == "diagnostic_only"
    assert artifact["repair_ladder_allowed"] is False
    assert artifact["receipt_gate_passed"] is False
    assert artifact["clean_verifier_gate_passed"] is False
    assert artifact["structured_proposal_gate_passed"] is False
    assert {row["code"] for row in artifact["blockers"]} >= {
        "missing_mandatory_artifact",
        "no_mandatory_gate_artifacts_readable",
    }


def test_req_verify_3213_unhandled_flags_and_correctness_claims_block(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3213: adversarial flags and repair correctness claims fail closed."""

    _write_standard_sources(
        tmp_path,
        clean={
            "unhandled_adversarial_methodology_flags": ["duration_padding_suspicion"],
        },
        proposal={"repair_correctness_claimed": True},
    )

    artifact = mod.build_artifact(tmp_path)

    assert artifact["clean_verifier_gate_passed"] is False
    assert artifact["structured_proposal_gate_passed"] is False
    assert artifact["repair_gate_state"] == "blocked"
    codes = {row["code"] for row in artifact["blockers"]}
    assert "exp3209_unhandled_adversarial_methodology_flag" in codes
    assert "exp3212_repair_correctness_claimed" in codes


def test_req_verify_3213_validation_rejects_contradictory_artifacts() -> None:
    """REQ-VERIFY-3213: validation rejects impossible gate outputs."""

    artifact = {
        "schema_version": mod.SCHEMA_VERSION,
        "experiment_id": "exp3213",
        "milestone": "2026.05.297",
        "required_artifacts": [],
        "missing_artifacts": [],
        "receipt_gate_passed": True,
        "clean_verifier_gate_passed": True,
        "structured_proposal_gate_passed": True,
        "auxiliary_fixture_artifacts": [],
        "repair_gate_state": "unblocked",
        "repair_ladder_allowed": True,
        "blockers": [],
        "conductor_file_modified": False,
        "active_roadmap_modified": False,
        "honest_verdict": "complete: valid",
    }

    mod.validate_artifact(artifact)
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({k: v for k, v in artifact.items() if k != "honest_verdict"})
    with pytest.raises(ValueError, match="repair_gate_state"):
        mod.validate_artifact(artifact | {"repair_gate_state": "maybe"})
    with pytest.raises(ValueError, match="gate booleans"):
        mod.validate_artifact(artifact | {"receipt_gate_passed": "yes"})
    with pytest.raises(ValueError, match="blockers"):
        mod.validate_artifact(artifact | {"blockers": "none"})
    with pytest.raises(ValueError, match="unblocked"):
        mod.validate_artifact(artifact | {"repair_ladder_allowed": False})
    with pytest.raises(ValueError, match="blockers"):
        mod.validate_artifact(artifact | {"blockers": [{"code": "unexpected"}]})
    with pytest.raises(ValueError, match="mandatory gates"):
        mod.validate_artifact(artifact | {"structured_proposal_gate_passed": False})
    with pytest.raises(ValueError, match="conductor_file_modified"):
        mod.validate_artifact(artifact | {"conductor_file_modified": True})
    with pytest.raises(ValueError, match="active_roadmap_modified"):
        mod.validate_artifact(artifact | {"active_roadmap_modified": True})
    with pytest.raises(ValueError, match="complete:"):
        mod.validate_artifact(artifact | {"honest_verdict": "blocked"})

    blocked = artifact | {
        "receipt_gate_passed": False,
        "repair_gate_state": "blocked",
        "repair_ladder_allowed": False,
        "blockers": [{"code": "blocked", "source_artifact": "x", "detail": "blocked"}],
        "honest_verdict": "complete: blocked",
    }
    mod.validate_artifact(blocked)
    with pytest.raises(ValueError, match="blocked"):
        mod.validate_artifact(blocked | {"repair_ladder_allowed": True})
    with pytest.raises(ValueError, match="blockers"):
        mod.validate_artifact(blocked | {"blockers": []})


def test_req_verify_3213_helpers_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-3213: malformed JSON and flag helpers remain conservative."""

    malformed = tmp_path / "bad.json"
    malformed.write_text("{bad json", encoding="utf-8")
    array_json = tmp_path / "array.json"
    array_json.write_text("[]", encoding="utf-8")

    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.read_json_object(malformed) == {}
    assert mod.read_json_object(array_json) == {}
    assert mod.clean_verifier_has_unhandled_flag({"flagged_adversarial": True}) is True
    assert mod.clean_verifier_has_unhandled_flag(
        {"unhandled_adversarial_methodology_flag": True}
    ) is True
    assert mod.clean_verifier_has_unhandled_flag(
        {"adversarial_methodology_flag_unhandled": True}
    ) is True
    assert mod.clean_verifier_has_unhandled_flag(
        {"adversarial_methodology_status": "flagged"}
    ) is True
    assert mod.clean_verifier_has_unhandled_flag(
        {"adversarial_methodology_flags": [{"handled": False, "code": "flag"}]}
    ) is True
    assert mod.clean_verifier_has_unhandled_flag(
        {"adversarial_methodology_flags": [{"handled": True, "code": "flag"}]}
    ) is False
    assert mod.auxiliary_invalidity({"artifact_invalid": True}) == "artifact_invalid=true"
    assert mod.auxiliary_invalidity({"flagged_invalid": True}) == "flagged_invalid=true"
    assert mod.auxiliary_invalidity({"valid": False}) == "valid=false"
    assert mod.auxiliary_invalidity({"invalidity": "schema"}) == "schema"
    assert mod.auxiliary_invalidity({"invalidity": None}) is None

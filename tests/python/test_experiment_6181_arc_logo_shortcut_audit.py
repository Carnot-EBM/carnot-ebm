"""Tests for Exp6181 ARC logo shortcut audit.

Spec refs: REQ-ARC-WMTE-6181,
SCENARIO-ARC-WMTE-6181-SINGLE-SLOT-FIXED-POLICY-PRECONDITIONS,
SCENARIO-ARC-WMTE-6181-LABEL-CONTROLS-AND-SHORTCUT-AUDIT,
SCENARIO-ARC-WMTE-6181-NO-SOLVE-PATH-AND-REGISTRY-DELTA.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

import scripts.adversarial_verify as adversarial_verify
from carnot import experiment_6181_arc_logo_shortcut_audit as mod


pytestmark = pytest.mark.memory_watchdog_skip

REPO = Path(__file__).resolve().parents[2]
ARC_SPEC = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def test_req_6181_spec_declares_single_slot_shortcut_audit_contract() -> None:
    """REQ-ARC-WMTE-6181: OpenSpec names the fixed-policy audit contract."""

    text = ARC_SPEC.read_text(encoding="utf-8")
    section = text[text.index("## REQ-ARC-WMTE-6181") :]
    section = section[: section.index("### REQ-ARC-WMTE-6180")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-ARC-WMTE-6181",
        "SCENARIO-ARC-WMTE-6181-SINGLE-SLOT-FIXED-POLICY-PRECONDITIONS",
        "SCENARIO-ARC-WMTE-6181-LABEL-CONTROLS-AND-SHORTCUT-AUDIT",
        "SCENARIO-ARC-WMTE-6181-NO-SOLVE-PATH-AND-REGISTRY-DELTA",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "leave-one-game-out",
        "known-label",
        "held-out-label",
        "shuffled-label",
        "alias",
        "unknown-label",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_6181_preconditions_snapshot_fixed_policy_and_task_labels(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-6181-SINGLE-SLOT-FIXED-POLICY-PRECONDITIONS."""

    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        test_exit_codes=_passing_exit_codes(),
        duration_s=1.0,
        write=False,
    )

    pre = artifact["preconditions_checked"]
    assert pre["run_date"] == "20260807"
    assert pre["registry_snapshot"]["path"] == "ops/arc_solve_registry.yaml"
    assert pre["root_clutter"]["ok"] is True
    assert pre["protected_git_status_short"] == []
    assert pre["task_labels"]["games"] == list(mod.DEFAULT_GAMES)
    assert pre["live_runtime_paths"]["live_entrypoint"] == (
        "python/carnot/agentic/arc_competition_agent.py"
    )
    assert artifact["single_arc_slot_receipt"]["slot_count_claimed"] == 1
    assert artifact["single_arc_slot_receipt"]["only_arc_slot_for_v535"] is True
    assert artifact["fixed_exp6167_policy_freeze"]["held_control_refit_count"] == 0
    assert artifact["fixed_exp6167_policy_freeze"]["policy_freeze_hash"].startswith("sha256:")
    assert artifact["fixed_exp6167_policy_freeze"]["exp6167_result_sha256"].startswith("sha256:")
    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert mod.validate_artifact(artifact) is True


def test_scenario_6181_label_controls_are_invariant_and_no_solve(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-6181-LABEL-CONTROLS-AND-SHORTCUT-AUDIT."""

    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        test_exit_codes=_passing_exit_codes(),
        duration_s=1.0,
        write=True,
    )

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete_no_shortcut_detected"
    assert artifact["honest_verdict"].startswith("complete_no_shortcut_detected:")
    assert artifact["live_attempt_label_dataset"]["row_count"] == 144
    assert artifact["live_attempt_label_dataset"]["source"] == (
        "exp6167_live_agent_self_discovery_aggregate"
    )
    assert artifact["leave_one_game_out_controls"]["fold_count"] == 6
    assert artifact["leave_one_game_out_controls"]["all_games_held_once"] is True
    assert artifact["leave_one_game_out_controls"]["all_folds_invariant"] is True
    assert artifact["leave_one_game_out_controls"]["policy_refit_count_total"] == 0
    assert set(artifact["label_control_results"]) == {
        "known_label",
        "held_out_label",
        "shuffled_label",
        "alias",
        "unknown_label",
    }
    baseline = artifact["shortcut_audit_summary"]["baseline_decision_signature_sha256"]
    for control in artifact["label_control_results"].values():
        assert control["passed"] is True
        assert control["decision_signature_sha256"] == baseline
        assert control["changed_decision_count"] == 0
    assert artifact["shortcut_audit_summary"]["shortcut_detected"] is False
    assert artifact["shortcut_audit_summary"]["all_controls_passed"] is True
    assert artifact["solve_claimed"] is False
    assert artifact["level_credit_delta"] == 0
    assert artifact["registry_delta"] == 0
    assert artifact["registry_levels_unchanged"] is True
    assert mod.validate_artifact(artifact) is True
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text()) == artifact


def test_scenario_6181_live_rows_and_control_helpers_preserve_exp6167_denominators() -> None:
    """SCENARIO-ARC-WMTE-6181-LABEL-CONTROLS-AND-SHORTCUT-AUDIT."""

    exp6167 = mod.load_exp6167_artifact(REPO)
    rows = mod.abstract_live_attempt_rows(exp6167)
    manifest = mod.fixed_task_aware_manifest(exp6167)
    baseline = mod.score_control(rows, manifest=manifest, label_transform=mod.known_label)
    shuffled = mod.score_control(rows, manifest=manifest, label_transform=mod.shuffled_label)
    alias = mod.score_control(rows, manifest=manifest, label_transform=mod.alias_label)
    unknown = mod.score_control(rows, manifest=manifest, label_transform=mod.unknown_label)

    assert len(rows) == exp6167["game_seed_action_budget_and_arm_counts"]["live_row_count"]
    assert {row["game"] for row in rows} == set(mod.DEFAULT_GAMES)
    assert {row["source"] for row in rows} == {
        "exp6167_live_agent_self_discovery_aggregate"
    }
    assert all(row["live_agent_self_discovery_attempt"] is True for row in rows)
    assert baseline["decision_count"] == 288
    assert baseline["decision_signature_sha256"] == shuffled["decision_signature_sha256"]
    assert baseline["decision_signature_sha256"] == alias["decision_signature_sha256"]
    assert baseline["decision_signature_sha256"] == unknown["decision_signature_sha256"]


def test_scenario_6181_validation_fails_closed_for_paths_controls_and_credit(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-6181-NO-SOLVE-PATH-AND-REGISTRY-DELTA."""

    artifact = mod.run(
        result_path=tmp_path / "fixture.json",
        test_exit_codes=_passing_exit_codes(),
        duration_s=1.0,
        write=False,
    )

    for field, value in (
        ("solve_claimed", True),
        ("level_credit_delta", 1),
        ("registry_delta", 1),
        ("registry_levels_unchanged", False),
    ):
        bad = deepcopy(artifact)
        bad[field] = value
        bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
        with pytest.raises(ValueError, match=field):
            mod.validate_artifact(bad)

    bad_path = deepcopy(artifact)
    bad_path["no_source_bfs_solver_kit_path_receipt"]["solver_kit_reproduce_called"] = True
    bad_path["reproducibility_checksum"] = mod.reproducibility_checksum(bad_path)
    with pytest.raises(ValueError, match="solver_kit_reproduce_called"):
        mod.validate_artifact(bad_path)

    for path_field in (
        "source_read_used",
        "offline_ground_truth_bfs_run",
        "solver_kit_imported_by_exp6181_module",
        "adapter_route_used",
    ):
        bad = deepcopy(artifact)
        bad["no_source_bfs_solver_kit_path_receipt"][path_field] = True
        bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
        with pytest.raises(ValueError, match=path_field):
            mod.validate_artifact(bad)

    bad_path_summary = deepcopy(artifact)
    bad_path_summary["no_source_bfs_solver_kit_path_receipt"][
        "proves_no_source_bfs_solver_path"
    ] = False
    bad_path_summary["reproducibility_checksum"] = mod.reproducibility_checksum(
        bad_path_summary
    )
    with pytest.raises(ValueError, match="no_source_bfs_solver_kit_path_receipt"):
        mod.validate_artifact(bad_path_summary)

    bad_control = deepcopy(artifact)
    bad_control["label_control_results"]["shuffled_label"]["passed"] = False
    bad_control["shortcut_audit_summary"] = mod.shortcut_audit_summary(bad_control)
    bad_control["status"] = mod.status(bad_control)
    bad_control["honest_verdict"] = mod.honest_verdict(bad_control)
    bad_control["reproducibility_checksum"] = mod.reproducibility_checksum(bad_control)
    with pytest.raises(ValueError, match="label_control_results"):
        mod.validate_artifact(bad_control)

    bad_slot = deepcopy(artifact)
    bad_slot["single_arc_slot_receipt"]["slot_count_claimed"] = 2
    bad_slot["shortcut_audit_summary"] = mod.shortcut_audit_summary(bad_slot)
    bad_slot["status"] = mod.status(bad_slot)
    bad_slot["honest_verdict"] = mod.honest_verdict(bad_slot)
    bad_slot["reproducibility_checksum"] = mod.reproducibility_checksum(bad_slot)
    with pytest.raises(ValueError, match="single_arc_slot_receipt"):
        mod.validate_artifact(bad_slot)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = mod.sha256_json({"wrong": True})
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)

    bad_reasons = deepcopy(artifact)
    bad_reasons["preconditions_checked"]["root_clutter"]["ok"] = False
    bad_reasons["single_arc_slot_receipt"]["slot_count_claimed"] = 0
    bad_reasons["fixed_exp6167_policy_freeze"]["held_control_refit_count"] = 1
    bad_reasons["adapter_disabled_live_path_receipt"]["adapter_disabled"] = False
    bad_reasons["live_attempt_label_dataset"]["all_rows_live_agent_self_discovery"] = False
    bad_reasons["leave_one_game_out_controls"]["all_folds_invariant"] = False
    bad_reasons["label_control_results"]["unknown_label"]["passed"] = False
    bad_reasons["no_source_bfs_solver_kit_path_receipt"][
        "proves_no_source_bfs_solver_path"
    ] = False
    bad_reasons["solve_claimed"] = True
    bad_reasons["level_credit_delta"] = 1
    bad_reasons["registry_delta"] = 1
    bad_reasons["registry_levels_unchanged"] = False
    bad_reasons["protected_files_unchanged"]["unchanged"] = False
    bad_reasons["inference_substrate"] = "wrong"
    reasons = set(mod._blocked_reasons(bad_reasons))
    assert {
        "root_clutter",
        "single_arc_slot_receipt",
        "fixed_exp6167_policy_freeze",
        "adapter_disabled_live_path_receipt",
        "live_attempt_label_dataset",
        "leave_one_game_out_controls",
        "label_control_results",
        "no_source_bfs_solver_kit_path_receipt",
        "solve_claimed",
        "level_credit_delta",
        "registry_delta",
        "registry_levels_unchanged",
        "protected_files_unchanged",
        "inference_substrate",
    } <= reasons


def test_req_6181_adversarial_verify_accepts_no_llm_no_solve_substrate(
    tmp_path: Path,
) -> None:
    """REQ-ARC-WMTE-6181: adversarial verification accepts the no-LLM audit."""

    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        test_exit_codes=_passing_exit_codes(),
        duration_s=1.0,
        write=True,
    )
    report = adversarial_verify.verify_artifact(tmp_path / mod.RESULT_RELATIVE_PATH.name)
    kinds = {flag["kind"] for flag in report["flags"]}

    assert adversarial_verify._classify_inference_substrate(artifact)["kind"] == "no_llm"
    assert "DURATION_TOO_SHORT" not in kinds
    assert "IMPLAUSIBLE_PERFECT" not in kinds
    assert "METHODOLOGY_MISSING" not in kinds

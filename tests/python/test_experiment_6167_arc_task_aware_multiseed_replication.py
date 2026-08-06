"""Tests for Exp6167 ARC task-aware multi-seed replication.

Spec refs: REQ-ARC-WMTE-6167,
SCENARIO-ARC-WMTE-6167-LIVE-ENTRYPOINT-FIXED-POLICY-AND-PROVENANCE,
SCENARIO-ARC-WMTE-6167-MULTIGAME-MULTISEED-METRICS-AND-CONTROLS,
SCENARIO-ARC-WMTE-6167-NO-SOLVE-REGISTRY-IMMUTABILITY-AND-SCHEMA.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

import scripts.adversarial_verify as adversarial_verify
from carnot import experiment_6167_arc_task_aware_multiseed_replication as mod


pytestmark = pytest.mark.memory_watchdog_skip

REPO = Path(__file__).resolve().parents[2]
ARC_SPEC = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _synthetic_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    changes_by_game = {
        "lp85": [0, 3, 0, 2],
        "su15": [0, 1, 0, 4],
        "tu93": [0, 2, 0, 5],
        "r11l": [0, 1, 0, 3],
        "ls20": [0, 4, 0, 1],
        "sp80": [0, 2, 0, 6],
    }
    for game, changes in changes_by_game.items():
        for seed in (6167, 6168, 6169):
            for action_index, changed_cells in enumerate(changes):
                safety_event = "invalid_action" if game == "sp80" and action_index == 0 else "none"
                rows.append(
                    {
                        "row_id": f"{game}|{seed}|{action_index}",
                        "game": game,
                        "seed": seed,
                        "action_index": action_index,
                        "action_id": 6,
                        "action_data": None,
                        "changed_cell_count": changed_cells,
                        "frame_changed": changed_cells > 0,
                        "valid_action": safety_event != "invalid_action",
                        "safety_event": safety_event,
                        "latency_ms": 0.1 + action_index,
                        "level_before": 0,
                        "level_after": 0,
                        "level_delta": 0,
                        "reward_delta": 0.0,
                        "live_entrypoint": "make_carnot_agent/E3AgentPolicy.choose_action",
                        "e3_policy_seen": True,
                        "source": "live_agent_runtime_action",
                    }
                )
    return rows


def test_req_6167_spec_declares_fixed_multiseed_replication_contract() -> None:
    """REQ-ARC-WMTE-6167: OpenSpec names the live no-solve artifact contract."""

    text = ARC_SPEC.read_text(encoding="utf-8")
    section = text[text.index("## REQ-ARC-WMTE-6167") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-ARC-WMTE-6167",
        "SCENARIO-ARC-WMTE-6167-LIVE-ENTRYPOINT-FIXED-POLICY-AND-PROVENANCE",
        "SCENARIO-ARC-WMTE-6167-MULTIGAME-MULTISEED-METRICS-AND-CONTROLS",
        "SCENARIO-ARC-WMTE-6167-NO-SOLVE-REGISTRY-IMMUTABILITY-AND-SCHEMA",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_6167_synthetic_multiseed_metrics_controls_and_schema(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6167-MULTIGAME-MULTISEED-METRICS-AND-CONTROLS."""

    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        live_rows=_synthetic_rows(),
        games=mod.DEFAULT_GAMES,
        held_games=mod.DEFAULT_GAMES,
        seeds=mod.DEFAULT_SEEDS,
        action_budget=4,
        test_exit_codes=_passing_exit_codes(),
        duration_s=1.0,
        write=True,
    )

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete_positive"
    assert artifact["honest_verdict"].startswith("complete_positive:")
    assert artifact["arc_task_aware_multiseed_replication_ready_score"] == 1.0
    assert artifact["game_seed_action_budget_and_arm_counts"]["game_count"] == 6
    assert artifact["game_seed_action_budget_and_arm_counts"]["seed_count"] == 3
    assert artifact["per_arm_triggered_decision_counts"] == {"global": 72, "task_aware": 72}
    assert artifact["grouped_paired_intervals"]["interval"]["lower_ci"] > 0
    assert artifact["known_negative_tail_receipt"]["prior_exp6154_tu93_delta"] < 0
    assert set(
        artifact["per_game_seed_transition_change_recall_safety_action_and_latency_metrics"]
    ) == set(mod.DEFAULT_GAMES)
    sp80_seed = artifact[
        "per_game_seed_transition_change_recall_safety_action_and_latency_metrics"
    ]["sp80"]["6167"]["global"]
    assert sp80_seed["invalid_action_count"] == 1
    assert sp80_seed["action_recall"] == sp80_seed["transition_recall"]
    assert (
        artifact[
            "shuffle_alias_identity_noop_invented_trigger_denominator_light_and_label_controls"
        ]["all_controls_passed"]
        is True
    )
    assert artifact["solve_claimed"] is False
    assert artifact["offline_ground_truth_bfs"] is False
    assert artifact["used_game_source"] is False
    assert artifact["level_credit_delta"] == 0
    assert artifact["registry_levels_unchanged"] is True
    assert artifact["llm_invocation_count"] == 0
    assert mod.validate_artifact(artifact) is True
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text()) == artifact


def test_scenario_6167_live_entrypoint_smoke_and_provenance(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6167-LIVE-ENTRYPOINT-FIXED-POLICY-AND-PROVENANCE."""

    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        games=mod.DEFAULT_GAMES,
        held_games=mod.DEFAULT_GAMES,
        seeds=mod.DEFAULT_SEEDS,
        action_budget=1,
        test_exit_codes=_passing_exit_codes(),
        duration_s=1.0,
        write=False,
    )

    assert artifact["game_seed_action_budget_and_arm_counts"]["episode_count"] == 18
    assert artifact["own_attempt_transition_provenance"]["scored_row_count"] == 18
    assert artifact["own_attempt_transition_provenance"]["all_rows_live_agent_owned"] is True
    assert (
        artifact["live_entrypoint_and_import_reachability"]["make_carnot_agent_constructed"] is True
    )
    assert artifact["live_entrypoint_and_import_reachability"]["e3_policy_seen"] is True
    assert (
        artifact["live_entrypoint_and_import_reachability"][
            "calibration_module_in_live_import_closure"
        ]
        is True
    )
    assert (
        artifact["adapter_per_game_lookup_solver_gotcha_and_hand_calibration_disable_receipts"][
            "hand_calibration_disabled"
        ]
        is True
    )
    assert artifact["llm_invocation_count"] == 0
    assert mod.validate_artifact(artifact) is True


def test_req_6167_validation_fails_closed_for_no_solve_and_registry_gates(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-6167-NO-SOLVE-REGISTRY-IMMUTABILITY-AND-SCHEMA."""

    artifact = mod.run(
        result_path=tmp_path / "fixture.json",
        live_rows=_synthetic_rows(),
        games=mod.DEFAULT_GAMES,
        held_games=mod.DEFAULT_GAMES,
        seeds=mod.DEFAULT_SEEDS,
        action_budget=4,
        test_exit_codes=_passing_exit_codes(),
        duration_s=1.0,
        write=False,
    )

    bad_no_trigger = deepcopy(artifact)
    bad_no_trigger["per_arm_triggered_decision_counts"]["task_aware"] = 0
    bad_no_trigger["arc_task_aware_multiseed_replication_ready_score"] = mod.ready_score(
        bad_no_trigger
    )
    bad_no_trigger["status"] = mod.status(bad_no_trigger)
    bad_no_trigger["honest_verdict"] = mod.honest_verdict(bad_no_trigger)
    bad_no_trigger["reproducibility_checksum"] = mod.reproducibility_checksum(bad_no_trigger)
    assert bad_no_trigger["arc_task_aware_multiseed_replication_ready_score"] == 0.0
    with pytest.raises(ValueError, match="triggered"):
        mod.validate_artifact(bad_no_trigger)

    for field, value in (
        ("solve_claimed", True),
        ("offline_ground_truth_bfs", True),
        ("used_game_source", True),
        ("registry_levels_unchanged", False),
    ):
        bad = deepcopy(artifact)
        bad[field] = value
        bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
        with pytest.raises(ValueError, match=field):
            mod.validate_artifact(bad)

    bad_level = deepcopy(artifact)
    bad_level["level_credit_delta"] = 1
    bad_level["reproducibility_checksum"] = mod.reproducibility_checksum(bad_level)
    with pytest.raises(ValueError, match="level_credit_delta"):
        mod.validate_artifact(bad_level)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = mod.sha256_json({"wrong": True})
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)

    bad_reasons = deepcopy(artifact)
    bad_reasons["preconditions_checked"]["root_clutter"]["ok"] = False
    bad_reasons["registry_precheck_and_no_duplicate_receipt"]["ok"] = False
    bad_reasons["upstream_policy_code_result_and_registry_hashes"][
        "upstream_exp6154_policy_and_result_hashed"
    ] = False
    bad_reasons["live_entrypoint_and_import_reachability"][
        "calibration_module_in_live_import_closure"
    ] = False
    bad_reasons["own_attempt_transition_provenance"]["all_rows_live_agent_owned"] = False
    bad_reasons["per_arm_triggered_decision_counts"]["global"] = 0
    bad_reasons["game_seed_action_budget_and_arm_counts"]["game_count"] = 5
    bad_reasons["game_seed_action_budget_and_arm_counts"]["seed_count"] = 2
    bad_reasons["grouped_paired_intervals"]["interval"]["lower_ci"] = 0.0
    bad_reasons["grouped_paired_intervals"]["support"]["no_safety_regression"] = False
    bad_reasons["false_confident_admission_and_abstention_matrices"][
        "task_aware_reduces_or_preserves_false_confident"
    ] = False
    bad_reasons[
        "shuffle_alias_identity_noop_invented_trigger_denominator_light_and_label_controls"
    ]["all_controls_passed"] = False
    bad_reasons["known_negative_tail_receipt"]["known_negative_tail_named_before_claim"] = False
    bad_reasons["solve_claimed"] = True
    bad_reasons["offline_ground_truth_bfs"] = True
    bad_reasons["used_game_source"] = True
    bad_reasons["offline_reproduced"] = True
    bad_reasons["level_credit_delta"] = 1
    bad_reasons["registry_levels_unchanged"] = False
    bad_reasons["llm_invocation_count"] = 1
    bad_reasons["protected_files_unchanged"]["unchanged"] = False
    bad_reasons["inference_substrate"] = "wrong"
    bad_reasons["verifier_is_oracle"] = True
    reasons = set(mod._blocked_reasons(bad_reasons))
    assert {
        "root_clutter",
        "registry_precheck",
        "upstream_hashes",
        "live_import_reachability",
        "own_attempt_transition_provenance",
        "triggered_decision_counts",
        "game_count",
        "seed_count",
        "nonpositive_grouped_lower_ci",
        "safety_regression",
        "false_confident_regression",
        "control_failure",
        "known_negative_tail_missing",
        "solve_claimed",
        "offline_ground_truth_bfs",
        "used_game_source",
        "offline_reproduced",
        "level_credit_delta",
        "registry_levels_unchanged",
        "llm_invocation_count",
        "protected_files_unchanged",
        "inference_substrate",
        "verifier_is_oracle",
    } <= reasons

    bad_ready_score = deepcopy(artifact)
    bad_ready_score["arc_task_aware_multiseed_replication_ready_score"] = 0.0
    bad_ready_score["reproducibility_checksum"] = mod.reproducibility_checksum(bad_ready_score)
    with pytest.raises(ValueError, match="arc_task_aware_multiseed_replication_ready_score"):
        mod.validate_artifact(bad_ready_score)


def test_req_6167_adversarial_verify_accepts_exact_no_llm_substrate(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-6167: adversarial verification accepts the exact substrate."""

    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        live_rows=_synthetic_rows(),
        games=mod.DEFAULT_GAMES,
        held_games=mod.DEFAULT_GAMES,
        seeds=mod.DEFAULT_SEEDS,
        action_budget=4,
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

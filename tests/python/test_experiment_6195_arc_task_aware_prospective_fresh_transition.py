"""Tests for Exp6195 prospective fresh ARC transition replay.

Spec refs: REQ-ARC-WMTE-6195,
SCENARIO-ARC-WMTE-6195-FRESH-DISJOINT-SEAL-BEFORE-REPLAY,
SCENARIO-ARC-WMTE-6195-FROZEN-IDENTICAL-REPLAY-AND-CONTROLS,
SCENARIO-ARC-WMTE-6195-NO-SOLVE-REGISTRY-AND-PROTECTED-FILES.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

import scripts.adversarial_verify as adversarial_verify
from carnot import experiment_6195_arc_task_aware_prospective_fresh_transition as mod


pytestmark = pytest.mark.memory_watchdog_skip

REPO = Path(__file__).resolve().parents[2]
ARC_SPEC = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _synthetic_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    changed_by_game = {
        "lp85": [0, 2, 0, 5],
        "su15": [0, 1, 0, 3],
        "tu93": [0, 4, 0, 2],
        "r11l": [0, 1, 0, 6],
        "ls20": [0, 3, 0, 1],
        "sp80": [0, 2, 0, 4],
    }
    for game, changes in changed_by_game.items():
        for seed in mod.DEFAULT_SEEDS:
            for action_index, changed_cells in enumerate(changes):
                rows.append(
                    {
                        "row_id": f"{game}|{seed}|{action_index}",
                        "game": game,
                        "seed": int(seed),
                        "action_index": int(action_index),
                        "action_id": 6,
                        "action_data": None,
                        "valid_action": True,
                        "level_before": 0,
                        "level_after": 0,
                        "level_delta": 0,
                        "reward_delta": 0.0,
                        "state_before": "",
                        "state_after": "",
                        "frame_changed": changed_cells > 0,
                        "changed_cell_count": changed_cells,
                        "safety_event": "none",
                        "latency_ms": 0.25 + action_index,
                        "action_budget": 4,
                        "live_entrypoint": "make_carnot_agent/E3AgentPolicy.choose_action",
                        "e3_policy_seen": True,
                        "provenance_rows_seen": 1,
                        "source": "live_agent_runtime_action",
                    }
                )
    return rows


def test_req_6195_spec_declares_fresh_replay_contract() -> None:
    """REQ-ARC-WMTE-6195: OpenSpec names the prospective replay contract."""

    text = ARC_SPEC.read_text(encoding="utf-8")
    section = text[text.index("## REQ-ARC-WMTE-6195") :]
    section = section[: section.index("### REQ-ARC-WMTE-6180")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-ARC-WMTE-6195",
        "SCENARIO-ARC-WMTE-6195-FRESH-DISJOINT-SEAL-BEFORE-REPLAY",
        "SCENARIO-ARC-WMTE-6195-FROZEN-IDENTICAL-REPLAY-AND-CONTROLS",
        "SCENARIO-ARC-WMTE-6195-NO-SOLVE-REGISTRY-AND-PROTECTED-FILES",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "seal timestamp SHALL precede any policy replay timestamp",
        "live_action_influence_count=0",
        "arc_solve_registry_delta=[]",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_6195_fresh_disjoint_sealed_before_replay(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6195-FRESH-DISJOINT-SEAL-BEFORE-REPLAY."""

    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        live_rows=_synthetic_rows(),
        test_exit_codes=_passing_exit_codes(),
        duration_s=1.0,
        write=True,
    )

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete_positive"
    assert artifact["honest_verdict"].startswith("complete_positive:")
    assert artifact["preconditions_checked"]["exp6184_preflight"]["ready"] is True
    assert artifact["registry_precheck_and_hash"]["no_duplicate_solve"] is True
    matrix = artifact["submitted_kernel_hash_and_escape_hatch_matrix"]
    assert matrix["submitted_kernel"] == "make_carnot_agent/E3AgentPolicy.choose_action"
    assert matrix["all_escape_hatches_disabled"] is True
    disjoint = artifact["prior_transition_hashes_and_disjointness_receipt"]
    assert disjoint["fresh_transition_count"] == len(_synthetic_rows())
    assert disjoint["overlap_count"] == 0
    assert disjoint["disjoint"] is True
    seal = artifact["seal_before_policy_replay_timestamp"]
    assert seal["seal_before_replay"] is True
    assert seal["policy_loaded_before_seal_count"] == 0
    assert artifact["fresh_live_agent_owned_transition_path_hash_count_and_provenance"][
        "all_rows_live_agent_owned"
    ] is True
    assert mod.validate_artifact(artifact) is True
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text()) == artifact


def test_scenario_6195_frozen_identical_replay_metrics_and_controls(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6195-FROZEN-IDENTICAL-REPLAY-AND-CONTROLS."""

    artifact = mod.run(
        result_path=tmp_path / "fixture.json",
        live_rows=_synthetic_rows(),
        test_exit_codes=_passing_exit_codes(),
        duration_s=1.0,
        write=False,
    )

    frozen = artifact["frozen_exp6167_policy_code_config_and_hash"]
    assert frozen["held_control_refit_count"] == 0
    assert frozen["threshold_changed_count"] == 0
    assert frozen["policy_freeze_hash"].startswith("sha256:")
    replay = artifact["identical_transition_replay_receipt"]
    assert replay["identical_transition_ids"] is True
    assert replay["global_replay_count"] == replay["task_aware_replay_count"]
    assert replay["policy_requested_new_observation_count"] == 0
    assert replay["policy_chose_live_action_count"] == 0
    metrics = artifact["global_and_task_aware_proposal_quality_metrics"]
    assert metrics["task_aware"]["false_confident_admissions"] < (
        metrics["global"]["false_confident_admissions"]
    )
    assert metrics["task_aware"]["proposal_quality"] > metrics["global"]["proposal_quality"]
    assert artifact["paired_delta_intervals_and_seed"]["mean_task_aware_minus_global"] > 0
    assert artifact["paired_delta_intervals_and_seed"]["seed"] == mod.RANDOM_SEED
    controls = artifact["task_logo_and_shuffle_controls"]
    assert controls["all_controls_passed"] is True
    assert controls["task_logo"]["changed_decision_count"] == 0
    assert controls["negative_control_shuffles"]["row_order"]["changed_decision_count"] == 0
    assert artifact["calibration_support_and_per_game_metrics"]["support"]["fresh_transition_count"] == (
        len(_synthetic_rows())
    )
    assert mod.validate_artifact(artifact) is True


def test_req_6195_submitted_kernel_acquisition_path_is_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-6195: live acquisition uses the submitted kernel collector."""

    calls: dict[str, Any] = {}

    def fake_collect_live_rows(*, games: Any, seeds: Any, action_budget: int) -> Any:
        calls["games"] = tuple(games)
        calls["seeds"] = tuple(seeds)
        calls["action_budget"] = action_budget
        return _synthetic_rows(), mod.synthetic_disable_receipt(), 0

    monkeypatch.setattr(mod.exp6167, "collect_live_rows", fake_collect_live_rows)
    artifact = mod.run(
        result_path=tmp_path / "fixture.json",
        test_exit_codes=_passing_exit_codes(),
        duration_s=1.0,
        write=False,
    )

    assert calls == {
        "games": mod.DEFAULT_GAMES,
        "seeds": mod.DEFAULT_SEEDS,
        "action_budget": mod.DEFAULT_ACTION_BUDGET,
    }
    assert artifact["fresh_live_agent_owned_transition_path_hash_count_and_provenance"][
        "collection_mode"
    ] == "submitted_live_kernel"
    assert mod.validate_artifact(artifact) is True


def test_scenario_6195_no_solve_registry_and_validation_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6195-NO-SOLVE-REGISTRY-AND-PROTECTED-FILES."""

    artifact = mod.run(
        result_path=tmp_path / "fixture.json",
        live_rows=_synthetic_rows(),
        test_exit_codes=_passing_exit_codes(),
        duration_s=1.0,
        write=False,
    )

    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["solve_claimed"] is False
    assert artifact["level_credit_claimed"] is False
    assert artifact["live_action_influence_count"] == 0
    assert artifact["arc_solve_registry_delta"] == []
    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert all(
        value == 0
        for value in artifact[
            "forbidden_source_bfs_adapter_prior_game_hidden_state_access_counts"
        ].values()
        if isinstance(value, int)
    )

    for field, value in (
        ("solve_claimed", True),
        ("level_credit_claimed", True),
        ("live_action_influence_count", 1),
        ("arc_solve_registry_delta", [{"game": "lp85"}]),
    ):
        bad = deepcopy(artifact)
        bad[field] = value
        bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
        with pytest.raises(ValueError, match=field):
            mod.validate_artifact(bad)

    bad_forbidden = deepcopy(artifact)
    bad_forbidden["forbidden_source_bfs_adapter_prior_game_hidden_state_access_counts"][
        "game_source_read_count"
    ] = 1
    bad_forbidden["reproducibility_checksum"] = mod.reproducibility_checksum(bad_forbidden)
    with pytest.raises(ValueError, match="forbidden"):
        mod.validate_artifact(bad_forbidden)

    bad_overlap = deepcopy(artifact)
    bad_overlap["prior_transition_hashes_and_disjointness_receipt"]["overlap_count"] = 1
    bad_overlap["prior_transition_hashes_and_disjointness_receipt"]["disjoint"] = False
    bad_overlap["status"] = mod.status(bad_overlap)
    bad_overlap["honest_verdict"] = mod.honest_verdict(bad_overlap)
    bad_overlap["reproducibility_checksum"] = mod.reproducibility_checksum(bad_overlap)
    with pytest.raises(ValueError, match="disjoint"):
        mod.validate_artifact(bad_overlap)

    bad_seal = deepcopy(artifact)
    bad_seal["seal_before_policy_replay_timestamp"]["seal_before_replay"] = False
    bad_seal["status"] = mod.status(bad_seal)
    bad_seal["honest_verdict"] = mod.honest_verdict(bad_seal)
    bad_seal["reproducibility_checksum"] = mod.reproducibility_checksum(bad_seal)
    with pytest.raises(ValueError, match="seal"):
        mod.validate_artifact(bad_seal)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = mod.sha256_json({"wrong": True})
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)


def test_req_6195_defensive_receipts_blockers_and_schema_guards(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-6195: defensive receipts and schema guards fail closed."""

    assert mod._transition_ids_from_exp6154(tmp_path) == []
    assert mod._transition_ids_from_exp6167(tmp_path) == []
    assert mod._transition_ids_from_exp6181(tmp_path) == []
    assert mod._exp6184_preflight_receipt(tmp_path)["ready"] is False
    assert (
        mod.paired_delta_intervals_and_seed(
            [{"row_id": "missing-task", "arm": "global", "admitted": True}]
        )["paired_transition_count"]
        == 0
    )

    artifact = mod.run(
        result_path=tmp_path / "fixture.json",
        live_rows=_synthetic_rows(),
        test_exit_codes=_passing_exit_codes(),
        duration_s=1.0,
        write=False,
    )

    bad_reasons = deepcopy(artifact)
    bad_reasons["preconditions_checked"]["exp6184_preflight"]["ready"] = False
    bad_reasons["preconditions_checked"]["root_clutter"]["ok"] = False
    bad_reasons["registry_precheck_and_hash"]["ok"] = False
    bad_reasons["submitted_kernel_hash_and_escape_hatch_matrix"][
        "all_escape_hatches_disabled"
    ] = False
    bad_reasons["prior_transition_hashes_and_disjointness_receipt"]["disjoint"] = False
    bad_reasons["fresh_live_agent_owned_transition_path_hash_count_and_provenance"][
        "all_rows_live_agent_owned"
    ] = False
    bad_reasons["seal_before_policy_replay_timestamp"]["seal_before_replay"] = False
    bad_reasons["frozen_exp6167_policy_code_config_and_hash"]["held_control_refit_count"] = 1
    bad_reasons["identical_transition_replay_receipt"]["identical_transition_ids"] = False
    bad_reasons["task_logo_and_shuffle_controls"]["all_controls_passed"] = False
    bad_reasons["live_action_influence_count"] = 1
    bad_reasons["forbidden_source_bfs_adapter_prior_game_hidden_state_access_counts"][
        "hidden_state_access_count"
    ] = 1
    bad_reasons["solve_provenance"] = "outer_loop_re"
    bad_reasons["solve_claimed"] = True
    bad_reasons["level_credit_claimed"] = True
    bad_reasons["arc_solve_registry_delta"] = [{"game": "lp85"}]
    bad_reasons["protected_files_unchanged"]["unchanged"] = False
    bad_reasons["inference_substrate"] = "wrong"
    reasons = set(mod._blocked_reasons(bad_reasons))
    assert {
        "exp6184_preflight",
        "root_clutter",
        "registry_precheck_and_hash",
        "submitted_kernel_hash_and_escape_hatch_matrix",
        "disjoint",
        "fresh_live_agent_owned_transition_path_hash_count_and_provenance",
        "seal",
        "frozen_exp6167_policy_code_config_and_hash",
        "identical_transition_replay_receipt",
        "task_logo_and_shuffle_controls",
        "live_action_influence_count",
        "forbidden",
        "solve_provenance",
        "solve_claimed",
        "level_credit_claimed",
        "arc_solve_registry_delta",
        "protected_files_unchanged",
        "inference_substrate",
    } <= reasons

    guard_cases = [
        ("field_provenance", lambda row: row["field_provenance"].pop("status")),
        (
            "seal",
            lambda row: row["seal_before_policy_replay_timestamp"].update(
                policy_loaded_before_seal_count=1
            ),
        ),
        (
            "submitted_kernel_hash_and_escape_hatch_matrix",
            lambda row: row["submitted_kernel_hash_and_escape_hatch_matrix"].update(
                all_escape_hatches_disabled=False
            ),
        ),
        (
            "fresh_live_agent_owned_transition_path_hash_count_and_provenance",
            lambda row: row[
                "fresh_live_agent_owned_transition_path_hash_count_and_provenance"
            ].update(all_rows_live_agent_owned=False),
        ),
        (
            "frozen_exp6167_policy_code_config_and_hash",
            lambda row: row["frozen_exp6167_policy_code_config_and_hash"].update(
                held_control_refit_count=1
            ),
        ),
        (
            "frozen_exp6167_policy_code_config_and_hash",
            lambda row: row["frozen_exp6167_policy_code_config_and_hash"].update(
                threshold_changed_count=1
            ),
        ),
        (
            "identical_transition_replay_receipt",
            lambda row: row["identical_transition_replay_receipt"].update(
                identical_transition_ids=False
            ),
        ),
        (
            "task_logo_and_shuffle_controls",
            lambda row: row["task_logo_and_shuffle_controls"].update(
                all_controls_passed=False
            ),
        ),
        (
            "protected_files_unchanged",
            lambda row: row["protected_files_unchanged"].update(unchanged=False),
        ),
        ("status", lambda row: row.update(status="wrong")),
        ("honest_verdict", lambda row: row.update(honest_verdict="wrong")),
    ]
    for message, mutate in guard_cases:
        bad = deepcopy(artifact)
        mutate(bad)
        bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(bad)


def test_req_6195_adversarial_verify_accepts_replay_substrate(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-6195: adversarial verification accepts replay substrate."""

    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        live_rows=_synthetic_rows(),
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

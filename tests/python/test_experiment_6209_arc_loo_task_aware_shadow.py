"""Tests for Exp6209 ARC leave-one-game-out shadow policy replay.

Spec refs: REQ-ARC-WMTE-6209,
SCENARIO-ARC-WMTE-6209-REGISTRY-MATRIX-AND-LIVE-COLLECTION,
SCENARIO-ARC-WMTE-6209-SHADOW-IDENTICAL-POLICIES-AND-CONTROLS,
SCENARIO-ARC-WMTE-6209-NO-SOLVE-REGISTRY-AND-FORBIDDEN-ACCESS.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

import scripts.adversarial_verify as adversarial_verify
from carnot import experiment_6209_arc_loo_task_aware_shadow as mod


pytestmark = pytest.mark.memory_watchdog_skip

REPO = Path(__file__).resolve().parents[2]
ARC_SPEC = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _synthetic_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    changes_by_game = {
        "lp85": [0, 2, 0, 5],
        "su15": [0, 1, 0, 3],
        "tu93": [0, 4, 0, 2],
        "r11l": [0, 1, 0, 6],
        "ls20": [0, 3, 0, 1],
        "sp80": [0, 2, 0, 4],
    }
    for game, changes in changes_by_game.items():
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
                        "action_budget": mod.DEFAULT_ACTION_BUDGET,
                        "live_entrypoint": "make_carnot_agent/E3AgentPolicy.choose_action",
                        "e3_policy_seen": True,
                        "provenance_rows_seen": 1,
                        "source": "live_agent_runtime_action",
                    }
                )
    return rows


def test_req_6209_spec_declares_loo_shadow_contract() -> None:
    """REQ-ARC-WMTE-6209: OpenSpec names the LOO shadow contract."""

    text = ARC_SPEC.read_text(encoding="utf-8")
    section = text[text.index("## REQ-ARC-WMTE-6209") :]
    section = section[: section.index("### REQ-ARC-WMTE-6180")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-ARC-WMTE-6209",
        "SCENARIO-ARC-WMTE-6209-REGISTRY-MATRIX-AND-LIVE-COLLECTION",
        "SCENARIO-ARC-WMTE-6209-SHADOW-IDENTICAL-POLICIES-AND-CONTROLS",
        "SCENARIO-ARC-WMTE-6209-NO-SOLVE-REGISTRY-AND-FORBIDDEN-ACCESS",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "solve_provenance SHALL be absent",
        "registry_update_count=0",
        "live_action_influence_count=0",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_6209_registry_matrix_and_live_collection(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6209-REGISTRY-MATRIX-AND-LIVE-COLLECTION."""

    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        live_rows=_synthetic_rows(),
        test_exit_codes=_passing_exit_codes(),
        duration_s=1.0,
        write=True,
    )

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert "solve_provenance" not in artifact
    assert artifact["status"] == "complete_positive"
    assert artifact["honest_verdict"].startswith("complete_positive:")
    registry = artifact["registry_precheck_and_hash_before_after"]
    assert registry["all_chosen_games_already_cleared"] is True
    assert registry["registry_hash_unchanged"] is True
    assert artifact["duplicate_solve_target_count"] == 0
    matrix = artifact["preregistered_loo_game_seed_matrix"]
    assert matrix["selection_frozen_before_acquisition"] is True
    assert matrix["minimum_fresh_transition_count"] == len(_synthetic_rows())
    assert matrix["matrix_hash"].startswith("sha256:")
    entrypoint = artifact["canonical_live_agent_entrypoint_receipts"]
    assert entrypoint["make_carnot_agent_constructed"] is True
    assert entrypoint["e3_policy_seen"] is True
    assert entrypoint["all_rows_from_canonical_entrypoint"] is True
    for receipt in artifact["adapter_disabled_receipts_by_held_out_game"].values():
        assert receipt["held_out_game_adapter_disabled"] is True
        assert receipt["all_escape_hatches_disabled"] is True
        assert receipt["source_bfs_prior_game_hidden_state_counts"] == {
            "adapter_route_count": 0,
            "game_source_read_count": 0,
            "hidden_state_access_count": 0,
            "offline_ground_truth_bfs_count": 0,
            "prior_game_memory_access_count": 0,
        }
    assert artifact["fresh_transition_paths_hashes_and_counts"]["transition_count"] == len(
        _synthetic_rows()
    )
    assert artifact["train_eval_overlap_counts"]["total_overlap_count"] == 0
    assert mod.validate_artifact(artifact) is True
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text()) == artifact


def test_scenario_6209_shadow_identical_policies_metrics_and_controls(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-6209-SHADOW-IDENTICAL-POLICIES-AND-CONTROLS."""

    artifact = mod.run(
        result_path=tmp_path / "fixture.json",
        live_rows=_synthetic_rows(),
        test_exit_codes=_passing_exit_codes(),
        duration_s=1.0,
        write=False,
    )

    frozen = artifact["frozen_policy_paths_and_hashes"]
    assert frozen["held_control_refit_count"] == 0
    assert frozen["threshold_changed_count"] == 0
    assert frozen["policy_code_config_hash"].startswith("sha256:")
    shadow = artifact["task_aware_and_global_shadow_decisions"]
    assert shadow["identical_transition_ids"] is True
    assert shadow["global_decision_count"] == shadow["task_aware_decision_count"]
    assert shadow["policy_requested_new_observation_count"] == 0
    assert shadow["policy_chose_live_action_count"] == 0
    metrics = artifact["loo_accuracy_quality_and_safety_by_game"]
    assert set(metrics["by_game"]) == set(mod.DEFAULT_GAMES)
    assert metrics["summary"]["losing_game_count"] == 0
    assert metrics["summary"]["winning_game_count"] > 0
    for game, row in metrics["by_game"].items():
        assert row["held_out_game"] == game
        assert row["global"]["decision_count"] == row["task_aware"]["decision_count"]
        assert row["task_aware_minus_global"] >= 0
    intervals = artifact["paired_clustered_intervals"]
    assert intervals["mean_task_aware_minus_global"] > 0
    assert intervals["by_game"]["lp85"]["task_aware_minus_global"] > 0
    controls = artifact["treatment_activation_and_aa_controls"]
    assert controls["treatment_activation"]["task_aware_changed_decision_count"] > 0
    assert controls["aa_controls"]["global_vs_global"]["changed_decision_count"] == 0
    assert controls["aa_controls"]["task_aware_vs_task_aware"]["changed_decision_count"] == 0
    assert controls["label_alias_controls"]["all_controls_passed"] is True
    assert controls["all_controls_passed"] is True
    assert mod.validate_artifact(artifact) is True


def test_req_6209_submitted_kernel_acquisition_path_is_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-6209: acquisition uses the submitted kernel collector."""

    calls: dict[str, Any] = {}

    def fake_collect_live_rows(*, games: Any, seeds: Any, action_budget: int) -> Any:
        calls["games"] = tuple(games)
        calls["seeds"] = tuple(seeds)
        calls["action_budget"] = int(action_budget)
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
    assert artifact["fresh_transition_paths_hashes_and_counts"]["collection_mode"] == (
        "submitted_live_kernel"
    )
    assert artifact["source_bfs_adapter_prior_game_hidden_state_access_counts"] == {
        "adapter_route_count": 0,
        "game_source_read_count": 0,
        "hidden_state_access_count": 0,
        "llm_invocation_count": 0,
        "offline_ground_truth_bfs_count": 0,
        "prior_game_memory_access_count": 0,
        "solver_kit_reproduce_count": 0,
    }
    assert mod.validate_artifact(artifact) is True


def test_scenario_6209_no_solve_registry_and_validation_fail_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-6209-NO-SOLVE-REGISTRY-AND-FORBIDDEN-ACCESS."""

    artifact = mod.run(
        result_path=tmp_path / "fixture.json",
        live_rows=_synthetic_rows(),
        test_exit_codes=_passing_exit_codes(),
        duration_s=1.0,
        write=False,
    )

    assert artifact["solve_claimed"] is False
    assert artifact["level_credit_delta"] == 0
    assert artifact["registry_update_count"] == 0
    assert artifact["live_action_influence_count"] == 0
    assert artifact["verifier_is_oracle"] is False
    assert all(
        value == 0
        for value in artifact[
            "source_bfs_adapter_prior_game_hidden_state_access_counts"
        ].values()
    )

    for field, value in (
        ("solve_claimed", True),
        ("level_credit_delta", 1),
        ("registry_update_count", 1),
        ("live_action_influence_count", 1),
        ("verifier_is_oracle", True),
    ):
        bad = deepcopy(artifact)
        bad[field] = value
        bad["status"] = mod.status(bad)
        bad["honest_verdict"] = mod.honest_verdict(bad)
        bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
        with pytest.raises(ValueError, match=field):
            mod.validate_artifact(bad)

    bad_solve_provenance = deepcopy(artifact)
    bad_solve_provenance["solve_provenance"] = "live_agent_self_discovery"
    bad_solve_provenance["reproducibility_checksum"] = mod.reproducibility_checksum(
        bad_solve_provenance
    )
    with pytest.raises(ValueError, match="solve_provenance"):
        mod.validate_artifact(bad_solve_provenance)

    bad_forbidden = deepcopy(artifact)
    bad_forbidden["source_bfs_adapter_prior_game_hidden_state_access_counts"][
        "hidden_state_access_count"
    ] = 1
    bad_forbidden["status"] = mod.status(bad_forbidden)
    bad_forbidden["honest_verdict"] = mod.honest_verdict(bad_forbidden)
    bad_forbidden["reproducibility_checksum"] = mod.reproducibility_checksum(bad_forbidden)
    with pytest.raises(ValueError, match="forbidden"):
        mod.validate_artifact(bad_forbidden)

    bad_registry = deepcopy(artifact)
    bad_registry["registry_precheck_and_hash_before_after"]["registry_hash_unchanged"] = False
    bad_registry["status"] = mod.status(bad_registry)
    bad_registry["honest_verdict"] = mod.honest_verdict(bad_registry)
    bad_registry["reproducibility_checksum"] = mod.reproducibility_checksum(bad_registry)
    with pytest.raises(ValueError, match="registry_precheck"):
        mod.validate_artifact(bad_registry)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = mod.sha256_json({"wrong": True})
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)


def test_req_6209_blocked_reason_guards_and_complete_null_are_covered(
    tmp_path: Path,
) -> None:
    """REQ-ARC-WMTE-6209: blocked guards name every invalid shadow path."""

    assert mod.sha256_file(tmp_path / "missing.json") is None
    artifact = mod.run(
        result_path=tmp_path / "fixture.json",
        live_rows=_synthetic_rows(),
        test_exit_codes=_passing_exit_codes(),
        duration_s=1.0,
        write=False,
    )

    null_artifact = deepcopy(artifact)
    null_artifact["paired_clustered_intervals"]["mean_task_aware_minus_global"] = 0.0
    null_artifact["status"] = mod.status(null_artifact)
    null_artifact["honest_verdict"] = mod.honest_verdict(null_artifact)
    null_artifact["reproducibility_checksum"] = mod.reproducibility_checksum(null_artifact)
    assert null_artifact["status"] == "complete_null"
    assert mod.validate_artifact(null_artifact) is True

    guard_cases = [
        (
            "registry_precheck",
            lambda row: row["registry_precheck_and_hash_before_after"].update(ok=False),
        ),
        ("duplicate_solve_target_count", lambda row: row.update(duplicate_solve_target_count=1)),
        (
            "preregistered_loo_game_seed_matrix",
            lambda row: row["preregistered_loo_game_seed_matrix"].update(
                selection_frozen_before_acquisition=False
            ),
        ),
        (
            "canonical_live_agent_entrypoint_receipts",
            lambda row: row["canonical_live_agent_entrypoint_receipts"].update(
                e3_policy_seen=False
            ),
        ),
        (
            "adapter_disabled_receipts_by_held_out_game",
            lambda row: next(iter(row["adapter_disabled_receipts_by_held_out_game"].values())).update(
                all_escape_hatches_disabled=False
            ),
        ),
        (
            "fresh_transition_paths_hashes_and_counts",
            lambda row: row["fresh_transition_paths_hashes_and_counts"].update(
                all_rows_live_agent_owned=False
            ),
        ),
        (
            "train_eval_overlap_counts",
            lambda row: row["train_eval_overlap_counts"].update(total_overlap_count=1),
        ),
        (
            "task_aware_and_global_shadow_decisions",
            lambda row: row["task_aware_and_global_shadow_decisions"].update(
                identical_transition_ids=False
            ),
        ),
        (
            "treatment_activation_and_aa_controls",
            lambda row: row["treatment_activation_and_aa_controls"].update(
                aa_controls_passed=False
            ),
        ),
        ("solve_provenance", lambda row: row.update(solve_provenance="outer_loop_re")),
        ("inference_substrate", lambda row: row.update(inference_substrate="wrong")),
    ]
    for expected, mutate in guard_cases:
        bad = deepcopy(artifact)
        mutate(bad)
        assert expected in mod._blocked_reasons(bad)

    blocked = deepcopy(artifact)
    blocked["preregistered_loo_game_seed_matrix"]["selection_frozen_before_acquisition"] = False
    blocked["status"] = mod.status(blocked)
    blocked["honest_verdict"] = mod.honest_verdict(blocked)
    blocked["reproducibility_checksum"] = mod.reproducibility_checksum(blocked)
    with pytest.raises(ValueError, match="preregistered_loo_game_seed_matrix"):
        mod.validate_artifact(blocked)


def test_req_6209_adversarial_verify_accepts_shadow_no_llm_substrate(
    tmp_path: Path,
) -> None:
    """REQ-ARC-WMTE-6209: adversarial verification accepts the shadow substrate."""

    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        live_rows=_synthetic_rows(),
        test_exit_codes=_passing_exit_codes(),
        duration_s=1.0,
        write=True,
    )
    report = adversarial_verify.verify_artifact(tmp_path / mod.RESULT_RELATIVE_PATH.name)
    kinds = {flag["kind"] for flag in report["flags"]}
    critical_kinds = {flag["kind"] for flag in report["flags"] if flag["severity"] == "critical"}

    assert adversarial_verify._classify_inference_substrate(artifact)["kind"] == "no_llm"
    assert critical_kinds == set()
    assert "DURATION_TOO_SHORT" not in kinds
    assert "METHODOLOGY_MISSING" not in kinds

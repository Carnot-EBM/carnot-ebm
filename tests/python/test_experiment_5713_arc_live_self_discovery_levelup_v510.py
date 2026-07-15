"""Tests for Exp5713 registry-prechecked ARC live self-discovery attempt.

Spec refs: REQ-ARC-WMTE-5713,
SCENARIO-ARC-WMTE-5713-PRECHECK-AND-ADVISORY,
SCENARIO-ARC-WMTE-5713-TRAJECTORY-COMPLETE-NULL,
SCENARIO-ARC-WMTE-5713-REPRODUCTION-GATE.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5713_arc_live_self_discovery_levelup_v510 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _registry(total: int = 181) -> dict[str, Any]:
    return {
        "reproducible_total_levels": total,
        "games": [
            {"game": "lf52", "reproducibility": "reproduced", "levels_reproduced": 6},
            {"game": "bp35", "reproducibility": "reproduced", "levels_reproduced": 8},
            {"game": "sk48", "reproducibility": "reproduced", "levels_reproduced": 7},
            {"game": "aa00", "reproducibility": "reproduced", "levels_reproduced": 1},
            {"game": "zz99", "reproducibility": "reproduced", "levels_reproduced": 1},
        ],
    }


def _public_envs() -> dict[str, dict[str, Any]]:
    return {
        "lf52": {"game_id": "lf52-public", "baseline_actions": [1] * 10},
        "bp35": {"game_id": "bp35-public", "baseline_actions": [1] * 9},
        "sk48": {"game_id": "sk48-public", "baseline_actions": [1] * 8},
        "aa00": {"game_id": "aa00-public", "baseline_actions": [1] * 11},
        "zz99": {"game_id": "zz99-public", "baseline_actions": [1] * 12},
    }


def _precheck() -> dict[str, Any]:
    return mod.registry_precheck(
        registry=_registry(),
        public_envs=_public_envs(),
        arc_loop_depths={"aa00": 11},
        registry_hash_before="sha256:test-registry",
    )


def _target() -> dict[str, Any]:
    return mod.select_target_from_precheck(_precheck())


def _null_attempt() -> dict[str, Any]:
    rows = [
        {
            "step": 0,
            "kind": "RESET",
            "action": None,
            "data": None,
            "level_before": None,
            "level_after": 0,
        },
        {
            "step": 1,
            "kind": "ACTION",
            "action": 6,
            "data": {"x": 10, "y": 10},
            "label": '{"action":6,"data":{"x":10,"y":10}}',
            "level_before": 0,
            "level_after": 0,
        },
        {
            "step": 2,
            "kind": "ACTION",
            "action": 1,
            "data": None,
            "label": '{"action":1,"data":null}',
            "level_before": 0,
            "level_after": 0,
        },
    ]
    checksum = mod.trajectory_hash_from_rows(rows)
    return {
        "live_attempt_executed": True,
        "action_budget": mod.ACTION_BUDGET,
        "action_rows": rows,
        "observations": [
            {"step": 0, "event": "reset", "level_after": 0},
            {"step": 1, "event": "action", "level_before": 0, "level_after": 0},
            {"step": 2, "event": "action", "level_before": 0, "level_after": 0},
        ],
        "level_counter_changes": [],
        "candidate_energy_receipts": [],
        "route_activations": [],
        "environment_actions": rows,
        "runtime_reverse_engineering": {
            "source": "runtime_observations_actions_state_transitions_only",
            "observations_recorded": 3,
            "level_changes_recorded": 0,
            "source_files_read": False,
            "per_game_adapter_used": False,
            "offline_ground_truth_bfs_used": False,
        },
        "max_level_reached": 0,
        "post_levels_reproduced": 1,
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "reproduction_gate": {
            "attempted": False,
            "reproduced": False,
            "reason": "target_level_not_reached_live",
        },
        "reproduction_seed_count": 0,
        "solution_labels": [],
        "action_trace_sha256": checksum,
        "trace_replay_checksum": checksum,
        "llm_invoked": False,
        "model_specs": [],
        "source_files_read": False,
        "per_game_adapter_used": False,
        "offline_bfs_used": False,
        "hand_solution_used": False,
        "terminal_reason": "action_budget_exhausted",
    }


def _success_attempt() -> dict[str, Any]:
    rows = [
        {
            "step": 0,
            "kind": "RESET",
            "action": None,
            "data": None,
            "level_before": None,
            "level_after": 0,
        },
        {
            "step": 1,
            "kind": "ACTION",
            "action": 1,
            "data": None,
            "label": '{"action":1,"data":null}',
            "level_before": 1,
            "level_after": 2,
        },
    ]
    checksum = mod.trajectory_hash_from_rows(rows)
    return {
        **_null_attempt(),
        "action_rows": rows,
        "observations": [
            {"step": 0, "event": "reset", "level_after": 1},
            {"step": 1, "event": "action", "level_before": 1, "level_after": 2},
        ],
        "level_counter_changes": [{"step": 1, "level_before": 1, "level_after": 2}],
        "environment_actions": rows,
        "max_level_reached": 2,
        "post_levels_reproduced": 2,
        "offline_reproduced": True,
        "reproduced_levels": 1,
        "reproduction_gate": {
            "attempted": True,
            "reproduced": True,
            "claimed_level": 2,
            "post_levels_reproduced": 2,
            "mode": "generic_live_path_clean_state_reproduction",
        },
        "reproduction_seed_count": 1,
        "solution_labels": ['{"action":1,"data":null}'],
        "action_trace_sha256": checksum,
        "trace_replay_checksum": checksum,
        "terminal_reason": "target_level_reached_live",
    }


def test_req_arc_wmte_5713_spec_declares_artifact_contract() -> None:
    """REQ-ARC-WMTE-5713: OpenSpec anchors the V510 receipt and field principles."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-WMTE-5713") :]
    section = section[: section.index("### REQ-ARC-WMTE-4738")]

    for marker in (
        mod.RESULT_RELATIVE_PATH,
        "SCENARIO-ARC-WMTE-5713-PRECHECK-AND-ADVISORY",
        "SCENARIO-ARC-WMTE-5713-TRAJECTORY-COMPLETE-NULL",
        "SCENARIO-ARC-WMTE-5713-REPRODUCTION-GATE",
        "arc_live_agent_own_attempts_no_llm",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle["principle"] in section


def test_scenario_arc_wmte_5713_precheck_excludes_and_ranks_headroom() -> None:
    """SCENARIO-ARC-WMTE-5713-PRECHECK-AND-ADVISORY: target selection is frozen."""

    precheck = _precheck()
    target = mod.select_target_from_precheck(precheck)
    by_game = {row["game"]: row for row in precheck["candidate_rows"]}

    assert precheck["ok"] is True
    assert precheck["registry_hash_before"] == "sha256:test-registry"
    assert by_game["lf52"]["closed_level_reasons"]["7"] == ["explicit_recent_failed_target"]
    assert by_game["lf52"]["closed_level_reasons"]["8"] == ["explicit_recent_failed_target"]
    assert by_game["lf52"]["target_level"] == 9
    assert by_game["bp35"]["exclude_reasons"] == ["all_authenticated_headroom_targets_closed"]
    assert by_game["sk48"]["exclude_reasons"] == ["all_authenticated_headroom_targets_closed"]
    assert by_game["aa00"]["exclude_reasons"] == ["all_authenticated_headroom_targets_closed"]
    assert by_game["aa00"]["closed_level_reasons"]["2"] == [
        "current_live_mechanism_already_reaches"
    ]
    assert target["selected_game"] == "zz99"
    assert target["selected_level"] == "L2"
    assert target["target_frozen_before_interaction"] is True
    assert target["target_selection_hash"].startswith("sha256:")


def test_req_arc_wmte_5713_io_helpers_and_blocked_selection(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-5713: helper branches keep blocked target selection auditable."""

    json_path = tmp_path / "payload.json"
    yaml_path = tmp_path / "payload.yaml"
    bytes_path = tmp_path / "bytes.txt"

    assert mod.read_json(tmp_path / "missing.json") == {}
    assert mod.read_yaml(tmp_path / "missing.yaml") == {}
    json_path.write_text('{"ok": true}', encoding="utf-8")
    yaml_path.write_text("ok: true\n", encoding="utf-8")
    bytes_path.write_text("abc", encoding="utf-8")

    assert mod.read_json(json_path) == {"ok": True}
    assert mod.read_yaml(yaml_path) == {"ok": True}
    assert mod.file_sha256(bytes_path).startswith("sha256:")
    assert mod._float("not-a-float", default=1.25) == 1.25
    assert mod._level_number("L12") == 12

    out_path = tmp_path / "nested" / "artifact.json"
    mod.write_json(out_path, {"b": 2})
    assert json.loads(out_path.read_text(encoding="utf-8")) == {"b": 2}

    precheck = mod.registry_precheck(
        registry={
            "reproducible_total_levels": 2,
            "games": [{"game": "zz99", "levels_reproduced": 2}],
        },
        public_envs={"zz99": {"baseline_actions": [1, 1]}},
    )
    row = precheck["candidate_rows"][0]
    blocked = mod.select_target_from_precheck(precheck)

    assert row["exclude_reasons"] == ["no_authenticated_headroom"]
    assert blocked["blocked"] is True
    assert blocked["selected_game"] is None
    assert blocked["target_selection_hash"].startswith("sha256:")


def test_scenario_arc_wmte_5713_exp5712_advisory_does_not_replace_baseline() -> None:
    """SCENARIO-ARC-WMTE-5713-PRECHECK-AND-ADVISORY: Exp5712 promotion is advisory."""

    null_advisory = mod.mechanism_selection_from_exp5712(
        {"relational_live_ab_ready_score": 0.0, "unsafe_route_accept_count": 0}
    )
    ready_but_not_target_local = mod.mechanism_selection_from_exp5712(
        {
            "relational_live_ab_ready_score": 1.0,
            "unsafe_route_accept_count": 0,
            "level_regression_count": 0,
        },
        target_hypothesis_induced_from_this_run=False,
    )
    promoted = mod.mechanism_selection_from_exp5712(
        {
            "relational_live_ab_ready_score": 1.0,
            "unsafe_route_accept_count": 0,
            "level_regression_count": 0,
        },
        target_hypothesis_induced_from_this_run=True,
    )

    assert null_advisory["mechanism_selection"]["baseline_unchanged"] is True
    assert null_advisory["mechanism_selection"]["enabled_exp5712"] is False
    assert ready_but_not_target_local["mechanism_selection"]["baseline_unchanged"] is True
    assert promoted["mechanism_selection"]["enabled_exp5712"] is True


def test_req_arc_wmte_5713_receipt_helpers_capture_flags_and_trajectory() -> None:
    """REQ-ARC-WMTE-5713: receipts expose provenance flags and trace material."""

    advisory = mod.mechanism_selection_from_exp5712({"relational_live_ab_ready_score": 0.0})
    attempt = {
        **_null_attempt(),
        "rewards": [0, 1],
        "source_files_read": True,
        "per_game_adapter_used": True,
        "offline_bfs_used": True,
        "hand_solution_used": True,
    }
    flags = mod._critical_flags(attempt, llm_used=True)
    artifact = mod.build_artifact(
        registry_precheck=_precheck(),
        target_selection_receipt=_target(),
        mechanism_selection=advisory["mechanism_selection"],
        exp5712_advisory_receipt=advisory["exp5712_advisory_receipt"],
        attempt=_null_attempt(),
        trajectory_path="results/helper-5713-trace.json",
        wall_time_seconds=0.05,
    )
    trajectory = mod.build_trajectory(
        target_selection_receipt=_target(),
        mechanism_selection=advisory["mechanism_selection"],
        attempt=attempt,
        artifact=artifact,
    )

    assert mod._baseline_filter_configuration()["baseline_unchanged"] is True
    assert flags == [
        "llm_used",
        "game_source_read",
        "game_adapter_used",
        "outer_loop_bfs_used",
        "hand_solution_used",
    ]
    assert trajectory["visible_observations"] == attempt["observations"]
    assert trajectory["actions"] == attempt["action_rows"]
    assert trajectory["rewards"] == [0, 1]
    assert trajectory["environment_actions"] == attempt["environment_actions"]


def test_scenario_arc_wmte_5713_trajectory_complete_null() -> None:
    """SCENARIO-ARC-WMTE-5713-TRAJECTORY-COMPLETE-NULL: nulls keep complete receipts."""

    advisory = mod.mechanism_selection_from_exp5712({"relational_live_ab_ready_score": 0.0})
    artifact = mod.build_artifact(
        registry_precheck=_precheck(),
        target_selection_receipt=_target(),
        mechanism_selection=advisory["mechanism_selection"],
        exp5712_advisory_receipt=advisory["exp5712_advisory_receipt"],
        attempt=_null_attempt(),
        trajectory_path="results/null-5713-trace.json",
        wall_time_seconds=0.05,
    )

    mod.validate_artifact(artifact)
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["agent_entrypoint"] == mod.AGENT_ENTRYPOINT
    assert artifact["model_specs"] == []
    assert artifact["llm_used"] is False
    assert artifact["environment_action_budget"] == mod.ACTION_BUDGET
    assert artifact["environment_actions_used"] == 2
    assert artifact["agent_visible_observation_count"] == 3
    assert artifact["action_count"] == 2
    assert artifact["level_transition_events"] == []
    assert artifact["new_level_candidate"]["candidate"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["offline_reproduced"] is False
    assert artifact["independent_reproduction_pass"] is False
    assert artifact["reproduction_receipts"] == []
    assert artifact["registry_count_before"] == 181
    assert artifact["registry_count_after"] == 181
    assert artifact["registry_delta"] == 0
    assert artifact["registry_updated"] is False
    assert artifact["critical_flags"] == []
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_arc_wmte_5713_validate_artifact_reports_defensive_errors() -> None:
    """REQ-ARC-WMTE-5713: validation names malformed banking and provenance states."""

    advisory = mod.mechanism_selection_from_exp5712({"relational_live_ab_ready_score": 0.0})
    artifact = mod.build_artifact(
        registry_precheck=_precheck(),
        target_selection_receipt=_target(),
        mechanism_selection=advisory["mechanism_selection"],
        exp5712_advisory_receipt=advisory["exp5712_advisory_receipt"],
        attempt=_null_attempt(),
        trajectory_path="results/bad-5713-trace.json",
        wall_time_seconds=0.05,
    )

    bad = dict(artifact)
    bad.pop("field_principles")
    bad.update(
        {
            "solve_provenance": "manual",
            "inference_substrate": "live_llm_inference",
            "agent_entrypoint": "manual",
            "target_frozen_before_interaction": False,
            "llm_used": True,
            "model_specs": [{"name": "legacy"}],
            "game_source_read_count": 1,
            "game_adapter_count": 1,
            "outer_loop_bfs_used": True,
            "hand_solution_used": True,
            "registry_delta": -1,
            "registry_count_after": 999,
            "offline_reproduced": True,
            "independent_reproduction_pass": False,
            "reproduced_levels": 0,
            "reproduction_receipts": [],
            "environment_actions_used": 2,
            "action_count": 1,
            "agent_visible_observation_count": 0,
            "trajectory_hash": "not-a-sha",
            "target_selection_hash": "sha256:bad",
            "honest_verdict": "not-terminal",
            "reproducibility_checksum": "bad",
        }
    )

    with pytest.raises(ValueError) as first_error:
        mod.validate_artifact(bad)
    message = str(first_error.value)
    for expected in (
        "missing required fields",
        "field_principles mismatch",
        "solve_provenance must be live_agent_self_discovery",
        "inference_substrate mismatch",
        "agent_entrypoint mismatch",
        "target must be frozen before interaction",
        "no-LLM artifact requires llm_used=false and model_specs=[]",
        "game_source_read_count must be zero",
        "game_adapter_count must be zero",
        "outer_loop_bfs_used must be false",
        "hand_solution_used must be false",
        "registry_delta must be non-negative",
        "registry_count_after must equal registry_count_before plus delta",
        "offline_reproduced requires independent_reproduction_pass",
        "offline_reproduced requires reproduced_levels >= 1",
        "offline_reproduced requires reproduction receipts",
        "environment_actions_used and action_count must match",
        "observation count must cover action count",
        "trajectory_hash must be sha256",
        "target_selection_hash mismatch",
        "honest_verdict must start with complete: or blocked:",
        "reproducibility_checksum mismatch",
    ):
        assert expected in message

    bad_null = dict(artifact)
    bad_null.update(
        {
            "independent_reproduction_pass": True,
            "reproduced_levels": 1,
            "registry_delta": 1,
            "registry_count_after": artifact["registry_count_before"] + 1,
            "registry_updated": True,
        }
    )
    bad_null["reproducibility_checksum"] = mod.compute_artifact_checksum(bad_null)

    with pytest.raises(ValueError) as second_error:
        mod.validate_artifact(bad_null)
    null_message = str(second_error.value)
    assert "null artifact requires independent_reproduction_pass=false" in null_message
    assert "null artifact requires reproduced_levels=0" in null_message
    assert "null artifact cannot update registry" in null_message


def test_scenario_arc_wmte_5713_reproduction_gate_banks_only_reproduced_target() -> None:
    """SCENARIO-ARC-WMTE-5713-REPRODUCTION-GATE: only generic reproduction banks."""

    precheck = mod.registry_precheck(
        registry={
            "reproducible_total_levels": 1,
            "games": [{"game": "zz99", "levels_reproduced": 1}],
        },
        public_envs={"zz99": {"baseline_actions": [1] * 3}},
        registry_hash_before="sha256:success-registry",
    )
    target = mod.select_target_from_precheck(precheck)
    advisory = mod.mechanism_selection_from_exp5712({"relational_live_ab_ready_score": 0.0})
    artifact = mod.build_artifact(
        registry_precheck=precheck,
        target_selection_receipt=target,
        mechanism_selection=advisory["mechanism_selection"],
        exp5712_advisory_receipt=advisory["exp5712_advisory_receipt"],
        attempt=_success_attempt(),
        trajectory_path="results/success-5713-trace.json",
        wall_time_seconds=0.05,
    )

    mod.validate_artifact(artifact)
    assert artifact["new_level_candidate"]["candidate"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["offline_reproduced"] is True
    assert artifact["independent_reproduction_pass"] is True
    assert artifact["reproduction_seed_count"] == 1
    assert artifact["registry_count_after"] == 2
    assert artifact["registry_delta"] == 1
    assert artifact["registry_updated"] is True
    assert (
        artifact["reproduction_receipts"][0]["mode"] == "generic_live_path_clean_state_reproduction"
    )


def test_req_arc_wmte_5713_repository_artifact_is_schema_valid() -> None:
    """REQ-ARC-WMTE-5713: checked-in artifact is the stable live-attempt receipt."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
    mod.validate_artifact(artifact)
    assert artifact["schema"] == mod.SCHEMA
    assert artifact["selected_game"]
    assert str(artifact["selected_level"]).startswith("L")
    assert artifact["target_frozen_before_interaction"] is True
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["inference_substrate"] == "arc_live_agent_own_attempts_no_llm"
    assert artifact["model_specs"] == []
    assert artifact["llm_used"] is False
    assert artifact["registry_delta"] == 0
    assert artifact["registry_updated"] is False
    assert artifact["honest_verdict"].startswith(("complete:", "blocked:"))

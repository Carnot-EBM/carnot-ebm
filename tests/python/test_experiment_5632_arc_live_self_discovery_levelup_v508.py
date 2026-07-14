"""Tests for Exp5632 ARC live self-discovery level-up attempt.

Spec refs: REQ-ARC-FCP-5632,
SCENARIO-ARC-FCP-5632-PRECHECK-EXCLUDES-RECENT-AND-REGISTRY-DUPLICATES,
SCENARIO-ARC-FCP-5632-EPISTEMIC-POLICY-ADVISORY-NOT-GATING,
SCENARIO-ARC-FCP-5632-REPRODUCTION-GATE-BANKS-AT-MOST-ONE-LEVEL.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5632_arc_live_self_discovery_levelup_v508 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _registry(*, total: int = 177) -> dict[str, Any]:
    return {
        "reproducible_total_levels": total,
        "games": [
            {"game": "lf52", "reproducibility": "reproduced", "levels_reproduced": 6},
            {"game": "sk48", "reproducibility": "reproduced", "levels_reproduced": 7},
            {"game": "bp35", "reproducibility": "reproduced", "levels_reproduced": 8},
            {"game": "done", "reproducibility": "reproduced", "levels_reproduced": 2},
        ],
    }


def _public_envs() -> dict[str, dict[str, Any]]:
    return {
        "lf52": {"game_id": "lf52-public", "baseline_actions": [1] * 10, "tags": ["click"]},
        "sk48": {"game_id": "sk48-public", "baseline_actions": [1] * 8, "tags": ["keyboard_click"]},
        "bp35": {"game_id": "bp35-public", "baseline_actions": [1] * 9, "tags": ["keyboard_click"]},
        "done": {"game_id": "done-public", "baseline_actions": [1] * 2, "tags": ["click"]},
    }


def _precheck() -> dict[str, Any]:
    return mod.registry_precheck(
        registry=_registry(),
        public_envs=_public_envs(),
        arc_loop_depths={"lf52": 2, "sk48": 2, "bp35": 2, "done": 2},
    )


def _target() -> dict[str, Any]:
    return mod.select_target_from_precheck(_precheck())


def _policy_source() -> dict[str, Any]:
    return mod.policy_source_from_exp5631(
        {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "unsafe_model_accept_count": 0,
        }
    )


def _null_attempt() -> dict[str, Any]:
    rows = [
        {"step": 0, "kind": "RESET", "action": None, "data": None, "level_before": None, "level_after": 0},
        {"step": 1, "kind": "ACTION", "action": 6, "data": {"x": 10, "y": 10}, "level_before": 0, "level_after": 0},
        {"step": 2, "kind": "ACTION", "action": 1, "data": None, "level_before": 0, "level_after": 0},
    ]
    checksum = mod.action_trace_sha256(rows)
    return {
        "live_attempt_executed": True,
        "action_budget": 48,
        "action_rows": rows,
        "observations": [{"step": 0, "event": "reset", "level": 0}],
        "level_counter_changes": [],
        "runtime_reverse_engineering": {
            "source": "runtime_observations_actions_state_transitions_only",
            "observations_recorded": 2,
            "source_files_read": False,
            "per_game_adapter_used": False,
            "offline_ground_truth_bfs_used": False,
        },
        "max_level_reached": 0,
        "post_levels_reproduced": 6,
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "reproduction_gate": {
            "game": "lf52",
            "claimed_level": 0,
            "reached_level": 6,
            "reproduced": False,
            "mode": "standard_reproduction_gate_not_run_no_new_target_candidate",
        },
        "solution_labels": [],
        "action_trace_sha256": checksum,
        "trace_replay_checksum": checksum,
        "llm_invoked": False,
        "model_specs": [],
        "source_files_read": False,
        "per_game_adapter_used": False,
        "offline_bfs_used": False,
        "outer_loop_re_used": False,
        "terminal_reason": "bounded_budget_no_target_level_reproduction",
    }


def _success_attempt() -> dict[str, Any]:
    rows = [
        {"step": 0, "kind": "RESET", "action": None, "data": None, "level_before": None, "level_after": 0},
        {"step": 1, "kind": "ACTION", "action": 6, "data": {"x": 20, "y": 20}, "level_before": 6, "level_after": 8},
    ]
    checksum = mod.action_trace_sha256(rows)
    return {
        **_null_attempt(),
        "action_rows": rows,
        "level_counter_changes": [{"step": 1, "level_before": 6, "level_after": 8}],
        "max_level_reached": 8,
        "post_levels_reproduced": 8,
        "offline_reproduced": True,
        "reproduced_levels": 2,
        "reproduction_gate": {
            "game": "lf52",
            "claimed_level": 8,
            "reached_level": 8,
            "reproduced": True,
            "mode": "generic_live_path_clean_state_reproduction",
        },
        "solution_labels": ['{"action":6,"data":{"x":20,"y":20}}'],
        "action_trace_sha256": checksum,
        "trace_replay_checksum": checksum,
        "terminal_reason": "target_level_reached_live",
    }


def test_req_arc_fcp_5632_spec_declares_required_artifact_fields() -> None:
    """REQ-ARC-FCP-5632: OpenSpec anchors the V508 live level-up receipt."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-FCP-5632") :]

    for marker in (
        mod.RESULT_RELATIVE_PATH,
        "SCENARIO-ARC-FCP-5632-PRECHECK-EXCLUDES-RECENT-AND-REGISTRY-DUPLICATES",
        "SCENARIO-ARC-FCP-5632-EPISTEMIC-POLICY-ADVISORY-NOT-GATING",
        "SCENARIO-ARC-FCP-5632-REPRODUCTION-GATE-BANKS-AT-MOST-ONE-LEVEL",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle["principle"] in section


def test_scenario_5632_precheck_excludes_recent_targets_and_picks_lf52() -> None:
    """SCENARIO-ARC-FCP-5632-PRECHECK-EXCLUDES-RECENT-AND-REGISTRY-DUPLICATES."""

    precheck = _precheck()
    target = mod.select_target_from_precheck(precheck)
    sk48 = next(row for row in precheck["candidate_rows"] if row["game"] == "sk48")
    bp35 = next(row for row in precheck["candidate_rows"] if row["game"] == "bp35")
    done = next(row for row in precheck["candidate_rows"] if row["game"] == "done")

    assert precheck["ok"] is True
    assert precheck["registry_count_before"] == 177
    assert precheck["public_games_checked"] == 4
    assert sk48["exclude_reasons"] == ["explicit_recent_unbanked_attempt"]
    assert bp35["exclude_reasons"] == ["explicit_recent_unbanked_attempt"]
    assert "no_authenticated_headroom" in done["exclude_reasons"]
    assert precheck["excluded_targets"]["explicit_recent_attempts"] == mod.EXPLICIT_EXCLUDED_TARGETS
    assert precheck["excluded_targets"]["registry_duplicate_levels"][0]["closed_levels"].startswith("L1-L")
    assert target["selected_game"] == "lf52"
    assert target["selected_level"] == "L7"
    assert target["prior_levels_reproduced"] == 6
    assert target["target_level"] == 7
    assert target["authenticated_headroom"] == 10
    assert target["target_selection_hash"].startswith("sha256:")
    assert target["selection_reason"] == "v508_rotated_unreproduced_authenticated_headroom"


def test_scenario_5632_exp5631_policy_is_advisory_not_gating() -> None:
    """SCENARIO-ARC-FCP-5632-EPISTEMIC-POLICY-ADVISORY-NOT-GATING."""

    blocked = _policy_source()
    promoted = mod.policy_source_from_exp5631(
        {
            "status": "complete",
            "live_epistemic_policy_ready": True,
            "unsafe_model_accept_count": 0,
            "known_level_regression_count": 0,
            "promoted_policy": {"name": "epistemic_probe_policy_v1"},
        }
    )
    unsafe = mod.policy_source_from_exp5631(
        {
            "status": "complete",
            "live_epistemic_policy_ready": True,
            "unsafe_model_accept_count": 1,
            "known_level_regression_count": 0,
            "promoted_policy": {"name": "epistemic_probe_policy_v1"},
        }
    )

    assert blocked["attempt_gated_by_exp5631"] is False
    assert blocked["policy_name"] == "unchanged_no_new_llm_e3_baseline"
    assert blocked["baseline_unchanged"] is True
    assert promoted["policy_name"] == "promoted_exp5631_epistemic_policy"
    assert promoted["baseline_unchanged"] is False
    assert unsafe["policy_name"] == "unchanged_no_new_llm_e3_baseline"


def test_scenario_5632_reproduction_gate_banks_at_most_one_level(tmp_path: Path) -> None:
    """SCENARIO-ARC-FCP-5632-REPRODUCTION-GATE-BANKS-AT-MOST-ONE-LEVEL."""

    null_artifact = mod.build_artifact(
        registry_precheck_receipt=_precheck(),
        target_selection_receipt=_target(),
        policy_source=_policy_source(),
        attempt=_null_attempt(),
        live_trace_path="results/null-5632.json",
        duration_s=0.02,
        tests_run=["unit"],
    )
    success = mod.build_artifact(
        registry_precheck_receipt=_precheck(),
        target_selection_receipt=_target(),
        policy_source=_policy_source(),
        attempt=_success_attempt(),
        live_trace_path="results/success-5632.json",
        duration_s=0.02,
        tests_run=["unit"],
    )

    mod.validate_artifact(null_artifact)
    mod.validate_artifact(success)
    assert null_artifact["model_specs"] == []
    assert null_artifact["random_seeds"] == [5632]
    assert null_artifact["offline_reproduced"] is False
    assert null_artifact["reproduced_levels"] == 0
    assert null_artifact["registry_count_before"] == 177
    assert null_artifact["registry_count_after"] == 177
    assert null_artifact["registry_delta"] == 0
    assert null_artifact["honest_verdict"].startswith("complete:")
    assert success["offline_reproduced"] is True
    assert success["new_reproducible_levels"] == [{"game": "lf52", "level": 7}]
    assert success["reproduced_levels"] == 1
    assert success["level_reached"] == 8
    assert success["registry_count_after"] == 178
    assert success["registry_delta"] == 1

    trace = mod.build_live_trace(
        target_selection_receipt=_target(),
        policy_source=_policy_source(),
        attempt=_null_attempt(),
        artifact=null_artifact,
    )
    trace_path = tmp_path / "trace.json"
    mod.write_json(trace_path, trace)
    loaded = json.loads(trace_path.read_text(encoding="utf-8"))
    assert loaded["action_trace_sha256"] == null_artifact["action_trace_sha256"]
    assert loaded["executed_actions"] == _null_attempt()["action_rows"]


def test_req_arc_fcp_5632_validation_rejects_missing_fields_and_drift() -> None:
    """REQ-ARC-FCP-5632: malformed artifacts fail closed before being written."""

    artifact = mod.build_artifact(
        registry_precheck_receipt=_precheck(),
        target_selection_receipt=_target(),
        policy_source=_policy_source(),
        attempt=_null_attempt(),
        live_trace_path="results/null-5632.json",
        duration_s=0.02,
        tests_run=["unit"],
    )
    missing = dict(artifact)
    missing.pop("policy_source")
    drifted = dict(artifact)
    drifted["registry_count_after"] = 999
    bad_model_specs = dict(artifact)
    bad_model_specs["model_specs"] = ["legacy-small-model"]
    bad_source = dict(artifact)
    bad_source["source_read"] = True

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing)
    with pytest.raises(ValueError, match="reproducibility_checksum mismatch"):
        mod.validate_artifact(drifted)
    with pytest.raises(ValueError, match="no-LLM attempts require model_specs=\\[\\]"):
        mod.validate_artifact(bad_model_specs)
    with pytest.raises(ValueError, match="source_read must be false"):
        mod.validate_artifact(bad_source)


def test_req_arc_fcp_5632_repository_artifact_has_required_schema() -> None:
    """REQ-ARC-FCP-5632: checked-in Exp5632 artifact is the stable live-attempt receipt."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["experiment"] == mod.EXPERIMENT
    assert artifact["solve_provenance"] == mod.SOLVE_PROVENANCE
    assert artifact["source_read"] is False
    assert artifact["game_adapter_used"] is False
    assert artifact["outer_loop_re_used"] is False
    assert artifact["model_specs"] == []
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["selected_game"]
    assert artifact["selected_level"]
    assert artifact["registry_delta"] in (0, 1)
    for field in mod.REQUIRED_FIELDS:
        assert field in artifact

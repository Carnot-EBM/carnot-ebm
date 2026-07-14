"""Tests for Exp5610 ARC live self-discovery level-up attempt.

Spec refs: REQ-ARC-FCP-5610,
SCENARIO-ARC-FCP-5610-PRECHECK-ROTATES-NON-DUPLICATE-HEADROOM,
SCENARIO-ARC-FCP-5610-FILTER-ADVISORY-NOT-GATING,
SCENARIO-ARC-FCP-5610-REPRODUCTION-GATE-BANKS-ONLY-NEW-LEVELS.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5610_arc_live_self_discovery_levelup_v506 as mod


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


def _previous_artifact() -> dict[str, Any]:
    return {
        "experiment": "experiment_5585_arc_levelup_attempt_v505",
        "game_targeted": "lf52",
        "target_level": 7,
        "target_selection": {"game": "lf52", "target_level": 7},
    }


def _precheck() -> dict[str, Any]:
    return mod.registry_precheck(
        registry=_registry(),
        public_envs=_public_envs(),
        arc_loop_depths={"lf52": 2, "sk48": 2, "bp35": 2, "done": 2},
        previous_artifact=_previous_artifact(),
        current_artifact={},
    )


def _filter_config() -> dict[str, Any]:
    return mod.filter_configuration_from_exp5609(
        {
            "filter_promotion_decisions": {
                "inert_click": {
                    "decision": "retire_reachable_downstream_noop",
                    "safety_regression": False,
                },
                "object_history": {
                    "decision": "retire_reachable_downstream_noop",
                    "safety_regression": False,
                },
            }
        }
    )


def _target() -> dict[str, Any]:
    return mod.select_target_from_precheck(_precheck())


def _null_attempt() -> dict[str, Any]:
    rows = [
        {"step": 1, "action": 6, "data": {"x": 10, "y": 10}, "level_before": 0, "level_after": 0},
        {"step": 2, "action": 1, "data": None, "level_before": 0, "level_after": 0},
    ]
    checksum = mod.action_trace_sha256(rows)
    return {
        "live_attempt_executed": True,
        "attempts": 2,
        "action_rows": rows,
        "observations": [{"step": 0, "event": "reset", "level": 0}],
        "level_counter_changes": [],
        "runtime_reverse_engineering": [
            {"step": 1, "signal": "observed_transition", "source": "live_runtime"}
        ],
        "max_level_reached": 0,
        "post_levels_reproduced": 7,
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "reproduction_gate": {
            "game": "sk48",
            "claimed_level": 0,
            "reached_level": 7,
            "reproduced": False,
            "mode": "standard_reproduction_gate_not_run_no_new_target_candidate",
        },
        "solution_labels": [],
        "action_trace_sha256": checksum,
        "trace_replay_checksum": checksum,
        "llm_invoked": False,
        "source_files_read": False,
        "per_game_adapter_used": False,
        "failure_mode": "bounded_budget_no_target_level_reproduction",
    }


def _success_attempt() -> dict[str, Any]:
    rows = [
        {"step": 1, "action": 6, "data": {"x": 20, "y": 20}, "level_before": 7, "level_after": 8}
    ]
    checksum = mod.action_trace_sha256(rows)
    return {
        **_null_attempt(),
        "attempts": 1,
        "action_rows": rows,
        "level_counter_changes": [{"step": 1, "level_before": 7, "level_after": 8}],
        "max_level_reached": 8,
        "post_levels_reproduced": 8,
        "offline_reproduced": True,
        "reproduced_levels": 1,
        "reproduction_gate": {
            "game": "sk48",
            "claimed_level": 8,
            "reached_level": 8,
            "reproduced": True,
            "mode": "offline_reproduction_gate_no_quota",
        },
        "solution_labels": ['{"action":6,"data":{"x":20,"y":20}}'],
        "action_trace_sha256": checksum,
        "trace_replay_checksum": checksum,
        "failure_mode": "",
    }


def test_req_arc_fcp_5610_spec_declares_required_artifact_fields() -> None:
    """REQ-ARC-FCP-5610: OpenSpec anchors the V506 live level-up receipt."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-FCP-5610") :]

    for marker in (
        mod.RESULT_RELATIVE_PATH,
        "SCENARIO-ARC-FCP-5610-PRECHECK-ROTATES-NON-DUPLICATE-HEADROOM",
        "SCENARIO-ARC-FCP-5610-FILTER-ADVISORY-NOT-GATING",
        "SCENARIO-ARC-FCP-5610-REPRODUCTION-GATE-BANKS-ONLY-NEW-LEVELS",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle["principle"] in section


def test_scenario_5610_precheck_rotates_to_non_duplicate_headroom_target() -> None:
    """SCENARIO-ARC-FCP-5610-PRECHECK-ROTATES-NON-DUPLICATE-HEADROOM: lf52 L7 is excluded."""

    precheck = _precheck()
    target = mod.select_target_from_precheck(precheck)
    lf52 = next(row for row in precheck["candidate_rows"] if row["game"] == "lf52")

    assert precheck["ok"] is True
    assert precheck["levels_before"] == 177
    assert precheck["public_games_checked"] == 4
    assert lf52["excluded"] is True
    assert "previous_milestone_target" in lf52["exclude_reasons"]
    assert target["selected_game"] == "sk48"
    assert target["selected_level"] == "L8"
    assert target["prior_levels_reproduced"] == 7
    assert target["target_level"] == 8
    assert target["authenticated_headroom"] == 8
    assert target["selection_reason"] == "rotated_non_duplicate_authenticated_headroom"


def test_scenario_5610_filter_configuration_is_advisory_not_gating() -> None:
    """SCENARIO-ARC-FCP-5610-FILTER-ADVISORY-NOT-GATING: failed Exp5609 filters do not skip."""

    retired = _filter_config()
    promoted = mod.filter_configuration_from_exp5609(
        {
            "filter_promotion_decisions": {
                "inert_click": {
                    "decision": "promote_candidate_pending_operator_review",
                    "safety_regression": False,
                },
                "object_history": {
                    "decision": "promote_candidate_pending_operator_review",
                    "safety_regression": True,
                },
            }
        }
    )

    assert retired["attempt_gated_by_exp5609"] is False
    assert retired["enabled_filters"] == []
    assert retired["baseline_unchanged"] is True
    assert promoted["attempt_gated_by_exp5609"] is False
    assert promoted["enabled_filters"] == ["inert_click"]
    assert promoted["baseline_unchanged"] is False


def test_scenario_5610_reproduction_gate_banks_only_new_levels(tmp_path: Path) -> None:
    """SCENARIO-ARC-FCP-5610-REPRODUCTION-GATE-BANKS-ONLY-NEW-LEVELS: nulls do not bank."""

    null_artifact = mod.build_artifact(
        registry_precheck=_precheck(),
        target_selection_receipt=_target(),
        filter_configuration=_filter_config(),
        attempt=_null_attempt(),
        attempt_trace_path="results/null-5610.json",
        duration_s=0.02,
        tests_run=["unit"],
    )
    success = mod.build_artifact(
        registry_precheck=_precheck(),
        target_selection_receipt=_target(),
        filter_configuration=_filter_config(),
        attempt=_success_attempt(),
        attempt_trace_path="results/success-5610.json",
        duration_s=0.02,
        tests_run=["unit"],
    )

    mod.validate_artifact(null_artifact)
    mod.validate_artifact(success)
    assert null_artifact["live_attempt_executed"] is True
    assert null_artifact["offline_reproduced"] is False
    assert null_artifact["new_reproducible_levels"] == []
    assert null_artifact["levels_before"] == 177
    assert null_artifact["levels_after"] == 177
    assert null_artifact["registry_updated"] is False
    assert null_artifact["honest_verdict"].startswith("complete:")
    assert success["offline_reproduced"] is True
    assert success["new_reproducible_levels"] == [{"game": "sk48", "level": 8}]
    assert success["levels_after"] == 178
    assert success["registry_updated"] is True

    trace = mod.build_attempt_trace(
        target_selection_receipt=_target(),
        attempt=_null_attempt(),
        artifact=null_artifact,
    )
    trace_path = tmp_path / "trace.json"
    mod.write_json(trace_path, trace)
    loaded = json.loads(trace_path.read_text(encoding="utf-8"))
    assert loaded["action_trace_sha256"] == null_artifact["action_trace_sha256"]
    assert loaded["executed_actions"] == _null_attempt()["action_rows"]


def test_req_arc_fcp_5610_validation_rejects_missing_fields_and_checksum_drift() -> None:
    """REQ-ARC-FCP-5610: malformed artifacts fail closed before being written."""

    artifact = mod.build_artifact(
        registry_precheck=_precheck(),
        target_selection_receipt=_target(),
        filter_configuration=_filter_config(),
        attempt=_null_attempt(),
        attempt_trace_path="results/null-5610.json",
        duration_s=0.02,
        tests_run=["unit"],
    )
    missing = dict(artifact)
    missing.pop("registry_precheck")
    drifted = dict(artifact)
    drifted["levels_after"] = 999

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing)
    with pytest.raises(ValueError, match="reproducibility_checksum mismatch"):
        mod.validate_artifact(drifted)


def test_req_arc_fcp_5610_repository_artifact_has_required_schema() -> None:
    """REQ-ARC-FCP-5610: checked-in Exp5610 artifact is the stable live-attempt receipt."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["experiment"] == mod.EXPERIMENT
    assert artifact["live_attempt_executed"] is True
    assert artifact["solve_provenance"] == mod.SOLVE_PROVENANCE
    assert artifact["source_files_read"] is False
    assert artifact["per_game_adapter_used"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["target_selection_receipt"]["selected_game"]
    for field in mod.REQUIRED_FIELDS:
        assert field in artifact

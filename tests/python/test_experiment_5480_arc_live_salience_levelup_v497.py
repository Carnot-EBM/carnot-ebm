"""Tests for Exp5480 rotated ARC live salience level-up attempt.

Spec refs: REQ-ARC-FCP-5480,
SCENARIO-ARC-FCP-5480.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5480_arc_live_salience_levelup_v497 as exp5480


pytestmark = pytest.mark.memory_watchdog_skip

REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"


def _registry(sb26_levels: int = 2) -> dict[str, Any]:
    return {
        "reproducible_total_levels": 69,
        "games": [
            {
                "game": "sb26",
                "reproducibility": "reproduced",
                "levels_reproduced": sb26_levels,
                "dead_ends": ["Exp4937 sb26 no-bank no_grounded_l3_delta"],
            },
            {"game": "g50t", "reproducibility": "reproduced", "levels_reproduced": 2},
        ],
    }


def _exp5479(selected_game: str = "sb26", selected_level: int = 3) -> dict[str, Any]:
    return {
        "arc_target_rotation_ready": True,
        "selected_game": selected_game,
        "selected_target_level": selected_level,
        "salience_feature_summary": {
            "target_region_candidates": [
                {
                    "action": 6,
                    "data": {"x": 14, "y": 14},
                    "tier": 0,
                    "source": "color_blob_prior",
                    "color": 9,
                    "score": 4000.2,
                    "button_like": True,
                }
            ],
            "changed_cells": {"count": 4},
            "connected_components": {"count": 4},
            "color_blobs": {"count": 4},
        },
    }


def _target() -> dict[str, Any]:
    return exp5480.select_exp5479_target(_exp5479(), _registry())


def _tmp_ready_root(tmp_path: Path, *, registry: dict[str, Any] | None = None) -> Path:
    root = tmp_path
    (root / "openspec" / "capabilities" / "arc-human-replay-frame-change").mkdir(parents=True)
    (root / "ops").mkdir()
    (root / "results").mkdir()
    (root / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (root / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    (root / exp5480.SPEC_RELATIVE_PATH).write_text(
        "REQ-ARC-FCP-5480\nSCENARIO-ARC-FCP-5480\n",
        encoding="utf-8",
    )
    (root / exp5480.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(registry or _registry()),
        encoding="utf-8",
    )
    (root / exp5480.EXP5479_RELATIVE_PATH).write_text(
        json.dumps(_exp5479()),
        encoding="utf-8",
    )
    return root


def test_req_arc_fcp_5480_spec_declares_required_artifact_fields() -> None:
    """REQ-ARC-FCP-5480: OpenSpec anchors the Exp5480 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-FCP-5480" in spec
    assert "SCENARIO-ARC-FCP-5480" in spec
    assert exp5480.RESULT_RELATIVE_PATH in spec
    for field, principle in exp5480.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in spec
        assert principle["principle"] in spec


def test_scenario_arc_fcp_5480_loads_rotated_target_and_blocks_duplicates() -> None:
    """SCENARIO-ARC-FCP-5480: Exp5479 target must exist and be unreproduced."""

    selected = exp5480.select_exp5479_target(_exp5479(), _registry())
    already_reproduced = exp5480.select_exp5479_target(_exp5479(), _registry(sb26_levels=3))
    missing = exp5480.select_exp5479_target(
        {"arc_target_rotation_ready": True},
        _registry(),
    )
    not_ready = exp5480.select_exp5479_target(
        {**_exp5479(), "arc_target_rotation_ready": False},
        _registry(),
    )

    assert selected["blocked"] is False
    assert selected["game"] == "sb26"
    assert selected["target_level"] == 3
    assert selected["reproduced_levels_before"] == 2
    assert selected["registry_total_before"] == 69
    assert already_reproduced["blocked"] is True
    assert already_reproduced["blocker"] == "target_already_reproduced"
    assert missing["blocked"] is True
    assert missing["blocker"] == "missing_exp5479_target"
    assert not_ready["blocked"] is True
    assert not_ready["blocker"] == "exp5479_target_rotation_not_ready"


def test_scenario_arc_fcp_5480_artifact_gates_null_and_success() -> None:
    """SCENARIO-ARC-FCP-5480: only reproduced target-level deltas bank levels."""

    null_artifact = exp5480.build_artifact(
        target=_target(),
        exp5479_artifact=_exp5479(),
        attempt={
            "action_count": 8,
            "explored_state_count": 5,
            "offline_reproduced": False,
            "reproduced_levels_after": 2,
        },
        registry_updated=False,
        preconditions_checked={"unit": True},
        tests_run=["unit"],
        duration_s=0.1,
    )

    exp5480.validate_artifact(null_artifact)
    assert null_artifact["game"] == "sb26"
    assert null_artifact["target_level"] == 3
    assert null_artifact["solve_provenance"] == "live_agent_self_discovery"
    assert null_artifact["hidden_source_reading"] is False
    assert null_artifact["offline_bfs_used"] is False
    assert null_artifact["hand_adapter_used"] is False
    assert null_artifact["outer_loop_re_used"] is False
    assert null_artifact["action_count"] == 8
    assert null_artifact["explored_state_count"] == 5
    assert null_artifact["failed_hypotheses"]
    assert null_artifact["offline_reproduced"] is False
    assert null_artifact["reproduced_levels"] == 0
    assert null_artifact["new_level_banked"] is False
    assert null_artifact["reproduced_levels_before"] == 2
    assert null_artifact["reproduced_levels_after"] == 2
    assert null_artifact["registry_updated"] is False
    assert null_artifact["first_win_trace_path"] == ""
    assert null_artifact["inference_substrate"] == "arc_live_agent_self_discovery"
    assert null_artifact["random_seed"] == 5480
    assert null_artifact["honest_verdict"].startswith("honest_null:")

    success = exp5480.build_artifact(
        target=_target(),
        exp5479_artifact=_exp5479(),
        attempt={
            "action_count": 11,
            "explored_state_count": 9,
            "offline_reproduced": True,
            "reproduced_levels_after": 3,
            "first_win_trace_path": "results/experiment_5480_first_win_trace_sb26_L3.json",
        },
        registry_updated=True,
        preconditions_checked={"unit": True},
        tests_run=["unit"],
        duration_s=0.1,
    )

    exp5480.validate_artifact(success)
    assert success["offline_reproduced"] is True
    assert success["reproduced_levels"] == 1
    assert success["new_level_banked"] is True
    assert success["reproduced_levels_after"] == 3
    assert success["registry_updated"] is True
    assert success["failed_hypotheses"] == []
    assert success["honest_verdict"].startswith("complete:")


def test_scenario_arc_fcp_5480_schema_rejects_off_path_or_nonincreasing_credit() -> None:
    """SCENARIO-ARC-FCP-5480: prohibited paths and duplicate depth fail closed."""

    artifact = exp5480.build_artifact(
        target=_target(),
        exp5479_artifact=_exp5479(),
        attempt={
            "action_count": 1,
            "explored_state_count": 1,
            "offline_reproduced": False,
            "reproduced_levels_after": 2,
        },
        registry_updated=False,
        preconditions_checked={"unit": True},
        tests_run=["unit"],
        duration_s=0.1,
    )
    invalid = {
        **artifact,
        "game": "",
        "target_level": "3",
        "solve_provenance": "development_proxy",
        "hidden_source_reading": True,
        "offline_bfs_used": True,
        "hand_adapter_used": True,
        "outer_loop_re_used": True,
        "action_count": "1",
        "explored_state_count": "1",
        "failed_hypotheses": "none",
        "offline_reproduced": True,
        "reproduced_levels": 0,
        "new_level_banked": True,
        "reproduced_levels_before": "2",
        "reproduced_levels_after": 2,
        "registry_updated": False,
        "first_win_trace_path": 0,
        "inference_substrate": "development_proxy",
        "random_seed": "5480",
        "honest_verdict": "solved",
    }

    errors = exp5480.artifact_schema_errors(invalid)

    assert "game must be a non-empty string" in errors
    assert "target_level must be bare int" in errors
    assert "solve_provenance must be live_agent_self_discovery" in errors
    assert "hidden_source_reading must be false" in errors
    assert "offline_bfs_used must be false" in errors
    assert "hand_adapter_used must be false" in errors
    assert "outer_loop_re_used must be false" in errors
    assert "action_count must be bare int" in errors
    assert "explored_state_count must be bare int" in errors
    assert "failed_hypotheses must be a list" in errors
    assert "offline_reproduced requires reproduced_levels >= 1" in errors
    assert "offline_reproduced requires reproduced_levels_after > reproduced_levels_before" in errors
    assert "new_level_banked requires registry_updated true" in errors
    assert "first_win_trace_path must be a string" in errors
    assert "inference_substrate must be arc_live_agent_self_discovery" in errors
    assert "random_seed must be bare int" in errors
    assert "honest_verdict must start with complete:, honest_null:, or blocked:" in errors
    with pytest.raises(ValueError):
        exp5480.validate_artifact(invalid)

    extra_errors = exp5480.artifact_schema_errors(
        {
            **artifact,
            "action_count": -1,
            "hidden_source_reading": "false",
            "offline_reproduced": "false",
            "new_level_banked": True,
            "registry_updated": True,
        }
    )
    registry_errors = exp5480.artifact_schema_errors({**artifact, "registry_updated": True})

    assert "action_count must be non-negative" in extra_errors
    assert "hidden_source_reading must be bare bool" in extra_errors
    assert "offline_reproduced must be bare bool" in extra_errors
    assert "new_level_banked requires offline_reproduced true" in extra_errors
    assert "registry_updated requires new_level_banked true" in registry_errors


def test_scenario_arc_fcp_5480_run_experiment_writes_honest_null_without_registry_update(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-FCP-5480: no reproduced target level leaves registry unchanged."""

    root = _tmp_ready_root(tmp_path)

    def attempt_runner(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["target"]["game"] == "sb26"
        assert kwargs["target"]["target_level"] == 3
        return {
            "action_count": 12,
            "explored_state_count": 7,
            "offline_reproduced": False,
            "reproduced_levels_after": 2,
            "failed_hypotheses": [{"hypothesis": "unit_no_bank"}],
        }

    artifact = exp5480.run_experiment(
        root=root,
        attempt_runner=attempt_runner,
        offline_arcade_check=lambda: True,
        tests_run=["unit 5480 null"],
    )
    written = json.loads((root / exp5480.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    registry = yaml.safe_load((root / exp5480.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert written == artifact
    assert artifact["honest_verdict"].startswith("honest_null:")
    assert artifact["action_count"] == 12
    assert artifact["reproduced_levels"] == 0
    assert artifact["new_level_banked"] is False
    assert artifact["registry_updated"] is False
    assert registry["games"][0]["levels_reproduced"] == 2
    assert registry["reproducible_total_levels"] == 69
    assert artifact["tests_run"] == ["unit 5480 null"]


def test_scenario_arc_fcp_5480_run_experiment_updates_registry_only_on_banked_success(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-FCP-5480: reproduced target-level success updates the registry."""

    root = _tmp_ready_root(tmp_path)

    def attempt_runner(**_kwargs: Any) -> dict[str, Any]:
        return {
            "action_count": 13,
            "explored_state_count": 8,
            "offline_reproduced": True,
            "reproduced_levels_after": 3,
            "first_win_trace_path": "results/experiment_5480_first_win_trace_sb26_L3.json",
        }

    artifact = exp5480.run_experiment(
        root=root,
        attempt_runner=attempt_runner,
        offline_arcade_check=lambda: True,
        tests_run=["unit 5480 success"],
    )
    registry = yaml.safe_load((root / exp5480.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["new_level_banked"] is True
    assert artifact["registry_updated"] is True
    assert registry["games"][0]["levels_reproduced"] == 3
    assert registry["reproducible_total_levels"] == 70
    assert registry["games"][0]["latest_exp5480_levelup_attempt"]["artifact"] == (
        exp5480.RESULT_RELATIVE_PATH
    )


def test_scenario_arc_fcp_5480_blocked_preconditions_and_helpers(tmp_path: Path) -> None:
    """SCENARIO-ARC-FCP-5480: blocked paths still emit the required schema fields."""

    blocked_artifact = exp5480.build_artifact(
        target=exp5480.select_exp5479_target(_exp5479(), _registry(sb26_levels=3)),
        exp5479_artifact=_exp5479(),
        attempt={"blocked": True, "failure_mode": "target_already_reproduced"},
        registry_updated=False,
        preconditions_checked={"unit": True},
        tests_run=["unit"],
        duration_s=0.1,
    )
    root = _tmp_ready_root(tmp_path / "blocked")
    arcade_blocked = exp5480.run_experiment(
        root=root,
        offline_arcade_check=lambda: False,
        tests_run=["arcade blocked"],
    )
    missing = exp5480.run_experiment(
        root=tmp_path / "missing",
        offline_arcade_check=lambda: True,
        tests_run=["missing"],
    )
    recent = exp5480._build_failed_hypotheses(  # noqa: SLF001
        _exp5479(),
        {"failure_mode": "unit_failure", "reproduced_levels_after": 2},
        _target(),
    )
    no_candidates = exp5480._build_failed_hypotheses(  # noqa: SLF001
        {"salience_feature_summary": {}},
        {},
        _target(),
    )
    null_artifact = exp5480.build_artifact(
        target=_target(),
        exp5479_artifact=_exp5479(),
        attempt={"offline_reproduced": False, "reproduced_levels_after": 2},
        registry_updated=False,
        preconditions_checked={"unit": True},
        tests_run=["unit"],
        duration_s=0.1,
    )
    appended_registry_root = tmp_path / "append_registry"
    new_game_artifact = exp5480.build_artifact(
        target={
            "blocked": False,
            "game": "zz99",
            "target_level": 1,
            "reproduced_levels_before": 0,
            "registry_total_before": 0,
        },
        exp5479_artifact={"salience_feature_summary": {}},
        attempt={
            "action_count": 2,
            "explored_state_count": 2,
            "offline_reproduced": True,
            "reproduced_levels_after": 1,
            "first_win_trace_path": "results/experiment_5480_first_win_trace_zz99_L1.json",
        },
        registry_updated=True,
        preconditions_checked={"unit": True},
        tests_run=["unit"],
        duration_s=0.1,
    )
    registry_not_updated = exp5480.update_registry_if_banked(
        root=tmp_path / "no_update",
        artifact=null_artifact,
        registry=_registry(),
    )
    registry_appended = exp5480.update_registry_if_banked(
        root=appended_registry_root,
        artifact=new_game_artifact,
        registry={"reproducible_total_levels": 0, "games": []},
    )
    appended_registry = yaml.safe_load(
        (appended_registry_root / exp5480.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8")
    )

    exp5480.validate_artifact(blocked_artifact)
    assert blocked_artifact["status"] == "blocked"
    assert blocked_artifact["game"] == "sb26"
    assert blocked_artifact["action_count"] == 0
    assert blocked_artifact["honest_verdict"] == "blocked: target_already_reproduced"
    assert arcade_blocked["status"] == "blocked"
    assert arcade_blocked["failure_mode"] == "missing_harness_access"
    assert arcade_blocked["preconditions_checked"]["offline_arcade_available"] is False
    assert missing["status"] == "blocked"
    assert missing["failure_mode"] == "missing_exp5479_target"
    assert recent[0]["hypothesis"] == "tier_0_color_9_click_candidate"
    assert recent[-1]["failure_mode"] == "unit_failure"
    assert no_candidates[0]["hypothesis"] == "exp5479_salience_candidates_absent"
    assert exp5480._trace_path_for("sb26", 3) == (  # noqa: SLF001
        "results/experiment_5480_first_win_trace_sb26_L3.json"
    )
    assert registry_not_updated is False
    assert registry_appended is True
    assert appended_registry["games"][0]["game"] == "zz99"

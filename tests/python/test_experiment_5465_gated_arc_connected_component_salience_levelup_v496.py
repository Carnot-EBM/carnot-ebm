"""Tests for Exp5465 gated ARC connected-component salience level-up attempt.

Spec refs: REQ-ARC-FCP-5465,
SCENARIO-ARC-FCP-5465.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5465_gated_arc_connected_component_salience_levelup_v496 as exp5465


pytestmark = pytest.mark.memory_watchdog_skip

REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"


def _registry(bp35_levels: int = 2, sb26_levels: int = 2) -> dict[str, Any]:
    return {
        "reproducible_total_levels": 69,
        "games": [
            {"game": "bp35", "reproducibility": "reproduced", "levels_reproduced": bp35_levels},
            {"game": "sb26", "reproducibility": "reproduced", "levels_reproduced": sb26_levels},
        ],
    }


def _precheck(ready: bool = True) -> dict[str, Any]:
    return {
        "arc_metric_integrity_ready": ready,
        "registry_precheck_performed": True,
        "target_shortlist": [
            {
                "game": "bp35",
                "target": "bp35:L3",
                "target_level": 3,
                "current_reproduced_levels": 2,
            },
            {
                "game": "sb26",
                "target": "sb26:L3",
                "target_level": 3,
                "current_reproduced_levels": 2,
            },
        ],
    }


def _selection() -> dict[str, Any]:
    return exp5465.select_target_from_precheck(_precheck(), _registry())


def test_req_arc_fcp_5465_spec_declares_required_fields() -> None:
    """REQ-ARC-FCP-5465: OpenSpec anchors the Exp5465 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-FCP-5465" in spec
    assert "SCENARIO-ARC-FCP-5465" in spec
    assert exp5465.RESULT_RELATIVE_PATH in spec
    for field, principle in exp5465.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in spec
        assert principle["principle"] in spec


def test_scenario_arc_fcp_5465_selects_target_from_ready_shortlist() -> None:
    """SCENARIO-ARC-FCP-5465: registry rerun selects an Exp5464 shortlist target."""

    selected = exp5465.select_target_from_precheck(_precheck(), _registry())
    rotated = exp5465.select_target_from_precheck(_precheck(), _registry(bp35_levels=3))
    blocked = exp5465.select_target_from_precheck(_precheck(ready=False), _registry())

    assert selected["registry_precheck_performed"] is True
    assert selected["target_game"] == "bp35"
    assert selected["target_level_before"] == 2
    assert selected["target_level_attempted"] == 3
    assert selected["target_from_exp5464_shortlist"] is True
    assert rotated["target_game"] == "sb26"
    assert rotated["target_level_before"] == 2
    assert blocked["blocked"] is True
    assert blocked["blocker"] == "exp5464_precheck_not_ready"


def test_scenario_arc_fcp_5465_live_feature_receipts_cover_required_features() -> None:
    """SCENARIO-ARC-FCP-5465: live-path receipts exercise all required features."""

    receipts = exp5465.build_live_feature_receipts()

    assert receipts["live_agent_policy_reachable"] is True
    assert receipts["connected_component_rows"]
    assert receipts["color_blob_rows"]
    assert receipts["changed_pixel_rows"]
    assert receipts["salience_tier_rows"]
    assert receipts["action_effect_observations"]
    assert receipts["perception_features_used"] == [
        "connected_component",
        "color_blob",
        "changed_pixel",
        "salience_tier",
        "action_effect",
    ]


def test_scenario_arc_fcp_5465_artifact_honest_null_and_success_gates() -> None:
    """SCENARIO-ARC-FCP-5465: only reproduced deeper live attempts bank a level."""

    receipts = exp5465.build_live_feature_receipts()
    null_artifact = exp5465.build_artifact(
        selection=_selection(),
        feature_receipts=receipts,
        attempt={
            "offline_reproduced": False,
            "reproduced_levels": 2,
            "max_level_reached": 2,
            "failure_mode": "bounded_budget_no_levelup",
        },
        preconditions_checked={"unit": True},
        tests_run=["unit"],
        duration_s=0.1,
    )

    exp5465.validate_artifact(null_artifact)
    assert null_artifact["solve_provenance"] == "live_agent_self_discovery"
    assert null_artifact["registry_precheck_performed"] is True
    assert null_artifact["target_game"] == "bp35"
    assert null_artifact["target_level_before"] == 2
    assert null_artifact["target_level_attempted"] == 3
    assert null_artifact["live_attempt_count"] == 1
    assert null_artifact["source_reading_used"] is False
    assert null_artifact["offline_bfs_used"] is False
    assert null_artifact["hand_adapter_credited"] is False
    assert null_artifact["offline_reproduced"] is False
    assert null_artifact["reproduced_levels"] == 2
    assert null_artifact["new_level_banked"] is False
    assert null_artifact["arc_registry_update_required"] is False
    assert null_artifact["inference_substrate"] == "arc_live_agent_self_discovery"
    assert null_artifact["honest_verdict"].startswith("honest_null:")

    success = exp5465.build_artifact(
        selection=_selection(),
        feature_receipts=receipts,
        attempt={
            "offline_reproduced": True,
            "reproduced_levels": 3,
            "max_level_reached": 3,
            "solution_labels": ['{"action":6,"data":{"x":14,"y":14}}'],
        },
        preconditions_checked={"unit": True},
        tests_run=["unit"],
        duration_s=0.1,
    )

    exp5465.validate_artifact(success)
    assert success["offline_reproduced"] is True
    assert success["reproduced_levels"] == 3
    assert success["new_level_banked"] is True
    assert success["arc_registry_update_required"] is True
    assert success["honest_verdict"].startswith("complete:")


def test_scenario_arc_fcp_5465_schema_rejects_off_path_or_nonincreasing_credit() -> None:
    """SCENARIO-ARC-FCP-5465: prohibited inputs and duplicate depth fail closed."""

    artifact = exp5465.build_artifact(
        selection=_selection(),
        feature_receipts=exp5465.build_live_feature_receipts(),
        attempt={"offline_reproduced": False, "reproduced_levels": 2},
        preconditions_checked={"unit": True},
        tests_run=["unit"],
        duration_s=0.1,
    )
    invalid = {
        **artifact,
        "solve_provenance": "outer_loop_re",
        "registry_precheck_performed": "yes",
        "target_game": "",
        "target_level_before": "2",
        "target_level_attempted": 2,
        "live_attempt_count": "1",
        "perception_features_used": ["connected_component"],
        "source_reading_used": True,
        "offline_bfs_used": True,
        "hand_adapter_credited": True,
        "offline_reproduced": True,
        "reproduced_levels": 2,
        "new_level_banked": True,
        "arc_registry_update_required": False,
        "inference_substrate": "development_proxy",
        "honest_verdict": "solved",
    }

    errors = exp5465.artifact_schema_errors(invalid)

    assert "solve_provenance must be live_agent_self_discovery" in errors
    assert "registry_precheck_performed must be bare bool" in errors
    assert "target_game must be a non-empty string" in errors
    assert "target_level_before must be bare int" in errors
    assert "target_level_attempted must be target_level_before + 1" in errors
    assert "live_attempt_count must be bare int" in errors
    assert "perception_features_used missing color_blob" in errors
    assert "source_reading_used must be false" in errors
    assert "offline_bfs_used must be false" in errors
    assert "hand_adapter_credited must be false" in errors
    assert "offline_reproduced requires reproduced_levels > target_level_before" in errors
    assert "new_level_banked requires arc_registry_update_required true" in errors
    assert "inference_substrate must be arc_live_agent_self_discovery" in errors
    assert "honest_verdict must start with complete:, honest_null:, or blocked:" in errors
    with pytest.raises(ValueError):
        exp5465.validate_artifact(invalid)

    missing_features = {**artifact, "perception_features_used": "connected_component"}
    bool_type_errors = {
        **artifact,
        "registry_precheck_performed": False,
        "source_reading_used": "false",
        "offline_reproduced": "false",
        "new_level_banked": True,
        "arc_registry_update_required": True,
    }
    registry_update_without_bank = {
        **artifact,
        "arc_registry_update_required": True,
    }
    more_errors = exp5465.artifact_schema_errors(missing_features)
    bool_errors = exp5465.artifact_schema_errors(bool_type_errors)
    registry_errors = exp5465.artifact_schema_errors(registry_update_without_bank)

    assert "perception_features_used must be a list" in more_errors
    assert "registry_precheck_performed must be true" in bool_errors
    assert "source_reading_used must be bare bool" in bool_errors
    assert "offline_reproduced must be bare bool" in bool_errors
    assert "new_level_banked requires offline_reproduced true" in bool_errors
    assert "arc_registry_update_required requires new_level_banked true" not in bool_errors
    assert "arc_registry_update_required requires new_level_banked true" in registry_errors


def test_scenario_arc_fcp_5465_helper_and_blocked_branches(tmp_path: Path) -> None:
    """SCENARIO-ARC-FCP-5465: helper fallbacks remain explicit and reproducible."""

    assert exp5465._as_int("bad", default=7) == 7  # noqa: SLF001
    assert exp5465.load_json(tmp_path / "missing.json") == {}
    assert exp5465.load_registry(tmp_path) == {"reproducible_total_levels": 0, "games": []}
    assert exp5465._action_label(6, {"x": 1, "y": 2}) == (  # noqa: SLF001
        '{"action":6,"data":{"x":1,"y":2}}'
    )
    assert exp5465._NoOpProposer().induce() == (  # noqa: SLF001
        False,
        "disabled_exp5465_no_live_llm",
    )
    assert exp5465._NoOpProposer().world_model_candidates("bp35") == []  # noqa: SLF001

    no_survivor = exp5465.select_target_from_precheck(
        {
            "arc_metric_integrity_ready": True,
            "target_shortlist": [
                "malformed",
                {"game": "missing", "target_level": 3},
                {"game": "bp35", "target_level": 2},
            ],
        },
        _registry(),
    )
    blocked_artifact = exp5465.build_artifact(
        selection=no_survivor,
        feature_receipts=exp5465.build_live_feature_receipts(),
        attempt={"blocked": True, "reproduced_levels": 0},
        preconditions_checked={"unit": True},
        tests_run=["unit"],
        duration_s=0.1,
    )

    assert no_survivor["blocked"] is True
    assert no_survivor["blocker"] == "no_exp5464_shortlist_target_survived_registry_rerun"
    assert blocked_artifact["status"] == "blocked"
    assert blocked_artifact["failure_mode"] == "no_exp5464_shortlist_target_survived_registry_rerun"
    assert blocked_artifact["live_attempt_count"] == 0

    ready_root = tmp_path / "ready"
    (ready_root / "openspec" / "capabilities" / "arc-human-replay-frame-change").mkdir(
        parents=True
    )
    (ready_root / "ops").mkdir()
    (ready_root / "results").mkdir()
    (ready_root / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (ready_root / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    (ready_root / exp5465.SPEC_RELATIVE_PATH).write_text(
        "REQ-ARC-FCP-5465\n",
        encoding="utf-8",
    )
    (ready_root / exp5465.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(_registry()),
        encoding="utf-8",
    )
    (ready_root / exp5465.PRECHECK_RELATIVE_PATH).write_text(
        json.dumps(_precheck()),
        encoding="utf-8",
    )
    arcade_blocked = exp5465.run_experiment(
        root=ready_root,
        offline_arcade_check=lambda: False,
        tests_run=["arcade blocked unit"],
    )
    missing_preconditions = exp5465.run_experiment(
        root=tmp_path / "missing_preconditions",
        offline_arcade_check=lambda: True,
        tests_run=["missing preconditions unit"],
    )

    assert arcade_blocked["status"] == "blocked"
    assert arcade_blocked["failure_mode"] == "missing_harness_access"
    assert arcade_blocked["preconditions_checked"]["offline_arcade_available"] is False
    assert missing_preconditions["status"] == "blocked"
    assert missing_preconditions["failure_mode"] == "exp5464_precheck_not_ready"


def test_scenario_arc_fcp_5465_run_experiment_writes_json(tmp_path: Path) -> None:
    """SCENARIO-ARC-FCP-5465: runner writes the required deliverable JSON."""

    root = tmp_path
    (root / "openspec" / "capabilities" / "arc-human-replay-frame-change").mkdir(parents=True)
    (root / "ops").mkdir()
    (root / "results").mkdir()
    (root / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (root / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    (root / exp5465.SPEC_RELATIVE_PATH).write_text(
        "REQ-ARC-FCP-5465\nSCENARIO-ARC-FCP-5465\n",
        encoding="utf-8",
    )
    (root / exp5465.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(_registry()),
        encoding="utf-8",
    )
    (root / exp5465.PRECHECK_RELATIVE_PATH).write_text(
        json.dumps(_precheck()),
        encoding="utf-8",
    )

    def attempt_runner(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["selection"]["target_game"] == "bp35"
        return {
            "offline_reproduced": False,
            "reproduced_levels": 2,
            "max_level_reached": 2,
            "failure_mode": "bounded_budget_no_levelup",
        }

    artifact = exp5465.run_experiment(
        root=root,
        attempt_runner=attempt_runner,
        offline_arcade_check=lambda: True,
        tests_run=["unit 5465"],
    )
    written = json.loads((root / exp5465.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert written == artifact
    assert artifact["target_game"] == "bp35"
    assert artifact["offline_reproduced"] is False
    assert artifact["new_level_banked"] is False
    assert artifact["arc_registry_update_required"] is False
    assert artifact["tests_run"] == ["unit 5465"]

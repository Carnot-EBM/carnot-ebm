"""Tests for Exp5479 ARC target-rotation live-path precheck.

Spec refs: REQ-ARC-FCP-5479,
SCENARIO-ARC-FCP-5479.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5479_arc_target_rotation_precheck_v497 as exp5479


pytestmark = pytest.mark.memory_watchdog_skip

REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"


def _registry() -> dict[str, Any]:
    return {
        "reproducible_total_levels": 69,
        "games": [
            {
                "game": "bp35",
                "reproducibility": "reproduced",
                "levels_reproduced": 2,
                "dead_ends": ["Exp5479 fixture: recent bp35 L3 no-bank"],
            },
            {
                "game": "sb26",
                "reproducibility": "reproduced",
                "levels_reproduced": 2,
                "dead_ends": ["Exp4937 sb26 no-bank no_grounded_l3_delta"],
            },
            {"game": "g50t", "reproducibility": "reproduced", "levels_reproduced": 2},
            {"game": "dc22", "reproducibility": "reproduced", "levels_reproduced": 2},
            {"game": "sp80", "reproducibility": "reproduced", "levels_reproduced": 2},
            {"game": "ka59", "reproducibility": "reproduced", "levels_reproduced": 1},
            {"game": "cn04", "reproducibility": "reproduced", "levels_reproduced": 3},
        ],
    }


def _precheck() -> dict[str, Any]:
    return {
        "arc_metric_integrity_ready": True,
        "target_shortlist": [
            {"game": "bp35", "target": "bp35:L3", "target_level": 3},
            {"game": "sb26", "target": "sb26:L3", "target_level": 3},
            {"game": "g50t", "target": "g50t:L3", "target_level": 3},
            {"game": "dc22", "target": "dc22:L3", "target_level": 3},
            {"game": "sp80", "target": "sp80:L3", "target_level": 3},
        ],
    }


def _known_issues() -> str:
    return "sb26 = search/goal-bound hard target; bp35 L3 no-bank; cn04 L4 stale lane"


def _selection() -> dict[str, Any]:
    return exp5479.select_rotated_target(_registry(), _precheck(), _known_issues())


def test_req_arc_fcp_5479_spec_declares_required_artifact_fields() -> None:
    """REQ-ARC-FCP-5479: OpenSpec anchors the no-solve artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-FCP-5479" in spec
    assert "SCENARIO-ARC-FCP-5479" in spec
    assert exp5479.RESULT_RELATIVE_PATH in spec
    for field, principle in exp5479.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in spec
        assert principle["principle"] in spec


def test_scenario_arc_fcp_5479_selects_rotated_nonduplicate_target() -> None:
    """SCENARIO-ARC-FCP-5479: target rotation avoids duplicate and no-bank lanes."""

    selection = exp5479.select_rotated_target(_registry(), _precheck(), _known_issues())
    blocked = exp5479.select_rotated_target(
        _registry(),
        {
            "arc_metric_integrity_ready": True,
            "target_shortlist": [
                {"game": "bp35", "target_level": 3},
                {"game": "sb26", "target_level": 2},
            ],
        },
        _known_issues(),
    )

    assert selection["selected_game"] == "sb26"
    assert selection["selected_target_level"] == 3
    assert selection["registry_reproducible_total_levels_before"] == 69
    assert selection["duplicate_target_rejected"] is True
    assert selection["recent_no_bank_targets_avoided"] == ["bp35:L3", "ka59:L2", "cn04:L4"]
    assert selection["target_audit"]["bp35:L3"]["decision"] == "rejected_recent_no_bank"
    assert selection["target_audit"]["sb26:L3"]["decision"] == "selected"
    assert selection["target_audit"]["sb26:L2"]["decision"] == "rejected_duplicate"
    assert selection["known_issues_checked"] is True
    assert blocked["blocked"] is True
    assert blocked["selected_game"] == ""
    assert blocked["blocker"] == "no_eligible_rotated_target"


def test_scenario_arc_fcp_5479_salience_dry_check_reports_required_features() -> None:
    """SCENARIO-ARC-FCP-5479: bounded salience dry check emits required receipts."""

    summary = exp5479.build_salience_feature_summary(
        selected_game="sb26",
        registry_row=_registry()["games"][1],
        known_issues_text=_known_issues(),
    )

    assert summary["connected_components"]["count"] > 0
    assert summary["color_blobs"]["count"] > 0
    assert summary["changed_cells"]["count"] == 4
    assert summary["target_region_candidates"]
    assert summary["target_region_candidates"][0]["tier"] == 0
    assert summary["known_blockers"]
    assert summary["known_issues_mentions"] >= 1
    assert summary["features_present"] == [
        "connected_component",
        "color_blob",
        "changed_pixel",
        "target_region_candidate",
        "known_blocker",
    ]


def test_scenario_arc_fcp_5479_artifact_schema_no_solve_claim() -> None:
    """SCENARIO-ARC-FCP-5479: artifact validates no-solve precheck fields."""

    artifact = exp5479.build_artifact(
        selection=_selection(),
        salience_feature_summary=exp5479.build_salience_feature_summary(
            selected_game="sb26",
            registry_row=_registry()["games"][1],
            known_issues_text=_known_issues(),
        ),
        preconditions_checked={"unit": True},
        tests_run=["unit 5479"],
        duration_s=0.1,
    )

    exp5479.validate_artifact(artifact)
    assert artifact["selected_game"] == "sb26"
    assert artifact["selected_target_level"] == 3
    assert artifact["registry_reproducible_total_levels_before"] == 69
    assert artifact["duplicate_target_rejected"] is True
    assert artifact["recent_no_bank_targets_avoided"] == ["bp35:L3", "ka59:L2", "cn04:L4"]
    assert artifact["live_path_reachable"] is True
    assert artifact["hidden_source_reading"] is False
    assert artifact["offline_bfs_used"] is False
    assert artifact["hand_adapter_used"] is False
    assert artifact["arc_target_rotation_ready"] is True
    assert artifact["solve_claimed"] is False
    assert artifact["inference_substrate"] == "arc_live_path_precheck_no_solve"
    assert artifact["random_seed"] == 5479
    assert artifact["honest_verdict"].startswith("complete:")
    assert "solved" not in artifact["honest_verdict"].lower()
    assert artifact["tests_run"] == ["unit 5479"]


def test_scenario_arc_fcp_5479_schema_rejects_bad_claims() -> None:
    """SCENARIO-ARC-FCP-5479: schema rejects source/BFS/adapter and solve claims."""

    artifact = exp5479.build_artifact(
        selection=_selection(),
        salience_feature_summary=exp5479.build_salience_feature_summary(
            selected_game="sb26",
            registry_row=_registry()["games"][1],
            known_issues_text=_known_issues(),
        ),
        preconditions_checked={"unit": True},
        tests_run=["unit"],
        duration_s=0.1,
    )
    invalid = {
        **artifact,
        "selected_game": "",
        "selected_target_level": "3",
        "registry_reproducible_total_levels_before": "69",
        "duplicate_target_rejected": "true",
        "recent_no_bank_targets_avoided": ["bp35:L3"],
        "live_path_reachable": "true",
        "hidden_source_reading": True,
        "offline_bfs_used": True,
        "hand_adapter_used": True,
        "salience_feature_summary": [],
        "arc_target_rotation_ready": True,
        "solve_claimed": True,
        "inference_substrate": "arc_live_agent_self_discovery",
        "random_seed": "5479",
        "honest_verdict": "complete: solved sb26 L3",
    }

    errors = exp5479.artifact_schema_errors(invalid)

    assert "selected_game must be a non-empty string" in errors
    assert "selected_target_level must be bare int" in errors
    assert "registry_reproducible_total_levels_before must be bare int" in errors
    assert "duplicate_target_rejected must be bare bool" in errors
    assert "recent_no_bank_targets_avoided missing ka59:L2" in errors
    assert "live_path_reachable must be bare bool" in errors
    assert "hidden_source_reading must be false" in errors
    assert "offline_bfs_used must be false" in errors
    assert "hand_adapter_used must be false" in errors
    assert "salience_feature_summary must be a dict" in errors
    assert "arc_target_rotation_ready requires solve_claimed false" in errors
    assert "solve_claimed must be false" in errors
    assert "inference_substrate must be arc_live_path_precheck_no_solve" in errors
    assert "random_seed must be bare int" in errors
    assert "honest_verdict must not claim a solve" in errors
    with pytest.raises(ValueError):
        exp5479.validate_artifact(invalid)


def test_scenario_arc_fcp_5479_helper_edge_branches() -> None:
    """SCENARIO-ARC-FCP-5479: rotation and schema helper branches are explicit."""

    recent = exp5479._recent_targets_from_no_bank(  # noqa: SLF001
        {"target_game": "lf52", "target_level_attempted": 3}
    )
    selection = exp5479.select_rotated_target(
        {
            "reproducible_total_levels": 69,
            "games": [
                {"game": "lf52", "reproducibility": "reproduced", "levels_reproduced": 2},
                {"game": "sb26", "reproducibility": "reproduced", "levels_reproduced": 2},
            ],
        },
        {
            "arc_metric_integrity_ready": True,
            "target_shortlist": [
                {"game": "lf52", "target": "lf52:L3", "target_level": 3},
                {"game": "sb26", "target": "sb26:L3", "target_level": 3},
            ],
        },
        "sb26 known issue",
        recent_no_bank_targets=("bp35:L3", "ka59:L2", "cn04:L4"),
    )
    blockers = exp5479._known_blockers(  # noqa: SLF001
        selected_game="lf52",
        registry_row={"dead_ends": "string blocker"},
        known_issues_text="lf52 known issue",
    )
    invalid_errors = exp5479.artifact_schema_errors(
        {
            "selected_game": "sb26",
            "selected_target_level": 3,
            "registry_reproducible_total_levels_before": 69,
            "duplicate_target_rejected": True,
            "recent_no_bank_targets_avoided": "bp35:L3",
            "live_path_reachable": True,
            "hidden_source_reading": False,
            "offline_bfs_used": False,
            "hand_adapter_used": False,
            "salience_feature_summary": {
                "connected_components": {},
                "color_blobs": {},
                "changed_cells": {},
                "target_region_candidates": [],
                "known_blockers": [],
            },
            "arc_target_rotation_ready": False,
            "solve_claimed": False,
            "inference_substrate": exp5479.INFERENCE_SUBSTRATE,
            "random_seed": exp5479.RANDOM_SEED,
            "honest_verdict": "ready without terminal prefix",
        }
    )

    assert recent[0] == "lf52:L3"
    assert selection["target_audit"]["lf52:L3"]["decision"] == "skipped_not_rotated_priority"
    assert selection["selected_game"] == "sb26"
    assert blockers[0] == "string blocker"
    assert "recent_no_bank_targets_avoided must be a list" in invalid_errors
    assert "honest_verdict must start with complete:, honest_null:, or blocked:" in invalid_errors


def test_scenario_arc_fcp_5479_run_experiment_writes_json(tmp_path: Path) -> None:
    """SCENARIO-ARC-FCP-5479: runner writes the required deliverable JSON."""

    root = tmp_path
    (root / "openspec" / "capabilities" / "arc-human-replay-frame-change").mkdir(parents=True)
    (root / "ops").mkdir()
    (root / "results").mkdir()
    (root / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (root / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    (root / exp5479.SPEC_RELATIVE_PATH).write_text(
        "REQ-ARC-FCP-5479\nSCENARIO-ARC-FCP-5479\n",
        encoding="utf-8",
    )
    (root / exp5479.REGISTRY_RELATIVE_PATH).write_text(
        yaml.safe_dump(_registry()),
        encoding="utf-8",
    )
    (root / exp5479.PRECHECK_RELATIVE_PATH).write_text(
        json.dumps(_precheck()),
        encoding="utf-8",
    )
    (root / exp5479.NO_BANK_RELATIVE_PATH).write_text(
        json.dumps({"target_game": "bp35", "target_level_attempted": 3}),
        encoding="utf-8",
    )
    (root / exp5479.KNOWN_ISSUES_RELATIVE_PATH).write_text(
        _known_issues(),
        encoding="utf-8",
    )

    artifact = exp5479.run_experiment(root=root, tests_run=["unit 5479 run"])
    written = json.loads((root / exp5479.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert written == artifact
    assert artifact["selected_game"] == "sb26"
    assert artifact["selected_target_level"] == 3
    assert artifact["arc_target_rotation_ready"] is True
    assert artifact["solve_claimed"] is False
    assert artifact["tests_run"] == ["unit 5479 run"]

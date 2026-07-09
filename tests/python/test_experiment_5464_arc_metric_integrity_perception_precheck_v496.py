"""Tests for Exp5464 ARC metric-integrity and perception precheck.

Spec refs: REQ-ARC-FCP-5464,
SCENARIO-ARC-FCP-5464.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5464_arc_metric_integrity_perception_precheck_v496 as exp5464


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"


def _registry() -> dict[str, Any]:
    return {
        "reproducible_total_levels": 69,
        "games": [
            {"game": "re86", "reproducibility": "reproduced", "levels_reproduced": 2},
            {"game": "lf52", "reproducibility": "reproduced", "levels_reproduced": 2},
            {"game": "cn04", "reproducibility": "reproduced", "levels_reproduced": 3},
            {"game": "ka59", "reproducibility": "reproduced", "levels_reproduced": 1},
            {"game": "bp35", "reproducibility": "reproduced", "levels_reproduced": 2},
            {"game": "sb26", "reproducibility": "reproduced", "levels_reproduced": 2},
            {"game": "g50t", "reproducibility": "reproduced", "levels_reproduced": 2},
        ],
    }


def _clean_loop_artifacts() -> dict[str, dict[str, Any]]:
    return {
        "bp35": {
            "game": "bp35",
            "offline_reproduced": True,
            "reproduced_levels": 2,
            "solution": [
                {"action": 4},
                {"action": 6, "data": {"x": 20, "y": 12}},
            ],
        },
        "sb26": {
            "game": "sb26",
            "offline_reproduced": True,
            "reproduced_levels": 2,
            "solution": [
                {"action": 6, "data": {"x": 36, "y": 59}},
                {"action": 5},
            ],
        },
    }


def test_req_arc_fcp_5464_spec_declares_required_fields() -> None:
    """REQ-ARC-FCP-5464: OpenSpec anchors the precheck artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-FCP-5464" in spec
    assert "SCENARIO-ARC-FCP-5464" in spec
    assert exp5464.RESULT_RELATIVE_PATH in spec
    for field, principle in exp5464.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in spec
        assert principle["principle"] in spec


def test_scenario_arc_fcp_5464_duplicate_and_off_path_claims_rejected() -> None:
    """SCENARIO-ARC-FCP-5464: duplicate/off-path solve credit fails closed."""

    duplicate = exp5464.audit_solve_claim(
        {
            "game": "re86",
            "target_level": 2,
            "offline_reproduced": True,
            "solve_provenance": "live_agent_self_discovery",
            "runtime_trace": [{"action": 1}],
        },
        _registry(),
    )
    off_path = exp5464.audit_solve_claim(
        {
            "game": "bp35",
            "target_level": 3,
            "offline_reproduced": True,
            "solve_provenance": "outer_loop_re",
            "used_env_source": True,
            "offline_ground_truth_bfs": True,
            "replay_only_artifact": True,
            "runtime_trace": [],
        },
        _registry(),
    )
    adapter_and_unreproduced = exp5464.audit_solve_claim(
        {
            "game": "sb26",
            "target_level": 3,
            "offline_reproduced": False,
            "solve_provenance": "live_agent_self_discovery",
            "per_game_adapter_used": True,
            "runtime_trace": [{"action": 6, "data": {"x": 1, "y": 2}}],
        },
        _registry(),
    )
    valid_probe = exp5464.audit_solve_claim(
        {
            "game": "bp35",
            "target_level": 3,
            "offline_reproduced": True,
            "solve_provenance": "live_agent_self_discovery",
            "runtime_trace": [{"action": 6, "data": {"x": 20, "y": 12}}],
        },
        _registry(),
    )

    assert duplicate["accepted"] is False
    assert duplicate["duplicate_rejected"] is True
    assert "duplicate_depth" in duplicate["rejection_reasons"]
    assert off_path["accepted"] is False
    assert off_path["off_path_rejected"] is True
    assert "source_or_ground_truth_path" in off_path["rejection_reasons"]
    assert "replay_only_without_runtime_trace" in off_path["rejection_reasons"]
    assert "per_game_adapter_path" in adapter_and_unreproduced["rejection_reasons"]
    assert "not_offline_reproduced" in adapter_and_unreproduced["rejection_reasons"]
    assert valid_probe["accepted"] is True
    assert exp5464._as_int("bad", default=7) == 7  # noqa: SLF001


def test_scenario_arc_fcp_5464_null_coordinate_exploit_audit() -> None:
    """SCENARIO-ARC-FCP-5464: null-coordinate one-step solves are contamination."""

    contaminated = exp5464.audit_null_coordinate_exploit(
        {
            "bad": {
                "game": "bad",
                "offline_reproduced": True,
                "reproduced_levels": 1,
                "solution": [{"action": 6, "data": None}],
            }
        }
    )
    label_based = exp5464.audit_null_coordinate_exploit(
        {
            "direct": {
                "game": "direct",
                "offline_reproduced": True,
                "reproduced_levels": 1,
                "solution": [{"x": 5, "y": 6}],
            },
            "keyboard": {
                "game": "keyboard",
                "offline_reproduced": True,
                "reproduced_levels": 1,
                "solution": [{"action": 1}],
            },
            "labels": {
                "game": "labels",
                "offline_reproduced": True,
                "reproduced_levels": 1,
                "solution_labels": ['{"action": 6, "data": {"x": 1, "y": 2}}', "RESET", {"action": 6}],
            },
            "skip_offline": {
                "game": "skip_offline",
                "offline_reproduced": False,
                "reproduced_levels": 1,
                "solution": [{"action": 6, "data": None}],
            },
            "skip_zero": {
                "game": "skip_zero",
                "offline_reproduced": True,
                "reproduced_levels": 0,
                "solution": [{"action": 6, "data": None}],
            },
        }
    )
    clean = exp5464.audit_null_coordinate_exploit(_clean_loop_artifacts())

    assert contaminated["null_coordinate_exploit_valid"] is True
    assert contaminated["contaminated_artifacts"][0]["game"] == "bad"
    assert label_based["null_coordinate_exploit_valid"] is False
    assert exp5464._step_action(object()) is None  # noqa: SLF001
    assert exp5464._step_data(object()) is None  # noqa: SLF001
    assert exp5464._step_data({"data": {"x": 1, "y": 2}}) == {"x": 1, "y": 2}  # noqa: SLF001
    assert exp5464._solution_steps({}) == []  # noqa: SLF001
    assert clean["null_coordinate_exploit_valid"] is False
    assert clean["contaminated_artifacts"] == []


def test_scenario_arc_fcp_5464_perception_receipts_cover_live_features(tmp_path: Path) -> None:
    """SCENARIO-ARC-FCP-5464: receipts cover live-path perception preconditions."""

    receipts = exp5464.build_perception_feature_receipts()
    path = exp5464.write_perception_receipts(tmp_path, receipts)
    written = json.loads(path.read_text(encoding="utf-8"))

    assert written == receipts
    assert receipts["live_agent_policy_reachable"] is True
    assert receipts["connected_component_rows"]
    assert receipts["color_blob_rows"]
    assert receipts["changed_pixel_rows"]
    assert receipts["salience_tier_rows"]
    assert receipts["action_effect_observations"]
    assert receipts["live_salience_diagnostics"]["connected_component_salience_enabled"] is True


def test_scenario_arc_fcp_5464_shortlist_avoids_recent_no_bank_lanes() -> None:
    """SCENARIO-ARC-FCP-5464: Exp5465 shortlist rotates off recent no-bank targets."""

    shortlist, avoided = exp5464.build_target_shortlist(
        _registry(),
        recent_no_bank_targets=("re86:L3", "lf52:L3", "cn04:L4", "ka59:L2"),
        priority=("re86", "lf52", "cn04", "ka59", "bp35", "sb26", "g50t"),
        limit=3,
    )
    zero_skipped, _zero_avoided = exp5464.build_target_shortlist(
        {
            "games": [
                {"game": "zero", "reproducibility": "reproduced", "levels_reproduced": 0},
                {"game": "bp35", "reproducibility": "reproduced", "levels_reproduced": 2},
            ]
        },
        recent_no_bank_targets=(),
        priority=("missing", "zero", "bp35"),
        limit=1,
    )

    assert [row["target"] for row in shortlist] == ["bp35:L3", "sb26:L3", "g50t:L3"]
    assert [row["target"] for row in zero_skipped] == ["bp35:L3"]
    assert all(row["current_reproduced_levels"] < row["target_level"] for row in shortlist)
    assert {row["target"] for row in avoided if row["decision"] == "avoided"} == {
        "re86:L3",
        "lf52:L3",
        "cn04:L4",
        "ka59:L2",
    }
    assert all(row["justification"] for row in avoided)


def test_req_arc_fcp_5464_artifact_schema_and_runner_write_json(tmp_path: Path) -> None:
    """REQ-ARC-FCP-5464: runner writes stable no-solve JSON artifacts."""

    (tmp_path / "AGENTS.md").write_text("repo instructions\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("codex instructions\n", encoding="utf-8")
    spec_path = tmp_path / exp5464.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True)
    spec_path.write_text("REQ-ARC-FCP-5464\nSCENARIO-ARC-FCP-5464\n", encoding="utf-8")
    registry_path = tmp_path / exp5464.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(yaml.safe_dump(_registry()), encoding="utf-8")
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    for game, row in _clean_loop_artifacts().items():
        (results_dir / f"arc_loop_solve_{game}.json").write_text(
            json.dumps({"game": game, **row}),
            encoding="utf-8",
        )

    artifact = exp5464.run_experiment(root=tmp_path, tests_run=["unit 5464"])
    written = json.loads((tmp_path / exp5464.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    receipts_path = tmp_path / artifact["perception_feature_receipts_path"]

    assert written == artifact
    assert receipts_path.exists()
    assert artifact["registry_precheck_performed"] is True
    assert artifact["reproduced_total_levels_before"] == 69
    assert artifact["duplicate_solve_rejected"] is True
    assert artifact["off_path_solve_rejected"] is True
    assert artifact["null_coordinate_exploit_valid"] is False
    assert artifact["arc_metric_integrity_ready"] is True
    assert artifact["inference_substrate"] == exp5464.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    exp5464.validate_artifact(artifact, root=tmp_path)

    corrupt = {
        **artifact,
        "registry_precheck_performed": "yes",
        "duplicate_solve_rejected": False,
        "off_path_solve_rejected": False,
        "null_coordinate_exploit_valid": True,
        "reproduced_total_levels_before": "69",
        "perception_feature_receipts_path": "",
        "target_shortlist": [],
        "recent_no_bank_targets_avoided_or_justified": [],
        "arc_metric_integrity_ready": True,
        "inference_substrate": "solve_claim",
        "honest_verdict": "solved",
    }
    errors = exp5464.artifact_schema_errors(corrupt, root=tmp_path)

    assert "registry_precheck_performed must be bare bool" in errors
    assert "reproduced_total_levels_before must be bare int" in errors
    assert "duplicate_solve_rejected must be true" in errors
    assert "off_path_solve_rejected must be true" in errors
    assert "arc_metric_integrity_ready requires null_coordinate_exploit_valid false" in errors
    assert "perception_feature_receipts_path must be non-empty string" in errors
    assert "target_shortlist must be a non-empty list" in errors
    assert "recent_no_bank_targets_avoided_or_justified must be a non-empty list" in errors
    assert f"inference_substrate must be {exp5464.INFERENCE_SUBSTRATE}" in errors
    assert "honest_verdict must start with complete:, honest_null:, or blocked:" in errors
    with pytest.raises(ValueError):
        exp5464.validate_artifact(corrupt, root=tmp_path)

    missing_receipts = {**artifact, "perception_feature_receipts_path": "results/missing.json"}
    assert "perception_feature_receipts_path must exist" in exp5464.artifact_schema_errors(
        missing_receipts,
        root=tmp_path,
    )

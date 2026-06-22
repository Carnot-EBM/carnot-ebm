"""Tests for Exp 4586 live-submittable co-headline.

Spec refs: REQ-CAPSTONE-4586, SCENARIO-CAPSTONE-4586,
SCENARIO-CAPSTONE-4586-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot import experiment_4574_action_efficiency_coheadline as exp4574
from carnot import experiment_4586_live_submittable_coheadline as mod
from carnot import live_submittable_metrics as live_metrics


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _registry(total: int = 5) -> JsonDict:
    return {
        "schema_version": 1,
        "reproducible_total_levels": total,
        "games": [
            {
                "game": "with_traj",
                "reproducibility": "reproduced",
                "levels_reproduced": 2,
            },
            {
                "game": "no_traj",
                "reproducibility": "reproduced",
                "levels_reproduced": 2,
            },
            {
                "game": "adaptive",
                "reproducibility": "reproduced",
                "levels_reproduced": 1,
            },
        ],
    }


def _package() -> JsonDict:
    return {
        "experiment": "fixture_package",
        "package_manifest": [
            {
                "game": "with_traj",
                "levels": 2,
                "offline_reproduced_level": 2,
                "trajectory_path": "results/arc3_live_banked_trajectories/with_traj.json",
                "action_count": 4,
                "env_matched": True,
            },
            {
                "game": "no_traj",
                "levels": 2,
                "offline_reproduced_level": 2,
                "trajectory_path": "",
                "action_count": 0,
                "env_matched": True,
            },
            {
                "game": "adaptive",
                "levels": 1,
                "offline_reproduced_level": 1,
                "trajectory_path": "",
                "action_count": 0,
                "adaptive_solver": "fixture_env_adaptive_solver",
                "env_matched": True,
            },
        ],
    }


def _b1_artifact() -> JsonDict:
    return {
        "reproducible_total_levels": 5,
        "generic_transfer_rate_over_variants": 0.125,
        "generic_transfer_ci": [0.0, 0.25],
        "action_efficiency_score": 0.75,
        "action_efficiency_ci": [0.5, 1.0],
        "median_actions_to_first_levelup": 6.0,
        "human_baseline_actions": 9.0,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_req_capstone_4586_spec_declares_live_submittable_coheadline() -> None:
    """REQ-CAPSTONE-4586: OpenSpec declares the live-submittable co-headline."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-CAPSTONE-4586",
        "SCENARIO-CAPSTONE-4586",
        "SCENARIO-CAPSTONE-4586-FIELD-PRINCIPLES",
        mod.RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_capstone_4586_helper_subset_excludes_no_trajectory_and_reports_gap() -> None:
    """REQ-CAPSTONE-4586: live-submittable is gated and remains a reproducible subset."""

    metrics = live_metrics.compute_live_submittable_metrics(
        registry=_registry(),
        package=_package(),
        package_path="results/fixture_package.json",
    )

    by_game = {row["game"]: row for row in metrics["per_game_live_submittable"]}
    assert metrics["live_submittable_level_count"] <= metrics["reproducible_total_levels"]
    assert metrics["reproducible_total_levels"] == 5
    assert metrics["live_submittable_level_count"] == 3
    assert metrics["reproducible_vs_submittable_gap"] == 2

    assert by_game["with_traj"]["included"] is True
    assert by_game["with_traj"]["submittable_level"] == 2
    assert by_game["with_traj"]["has_replayable_trajectory"] is True
    assert by_game["with_traj"]["env_matchable"] is True

    assert by_game["no_traj"]["included"] is False
    assert by_game["no_traj"]["submittable_level"] == 0
    assert by_game["no_traj"]["exclusion_reason"] == "missing_trajectory_or_adaptive_resolver"

    assert by_game["adaptive"]["included"] is True
    assert by_game["adaptive"]["submittable_level"] == 1
    assert by_game["adaptive"]["has_env_adaptive_resolver"] is True


def test_scenario_capstone_4586_capstone_function_reports_four_coheadlines() -> None:
    """SCENARIO-CAPSTONE-4586: the shared capstone function reports all co-headlines."""

    coheadline = exp4574.build_live_submittable_coheadline_metrics(
        registry=_registry(),
        package=_package(),
        package_path="results/fixture_package.json",
        b1_artifact=_b1_artifact(),
    )

    assert coheadline["reported_side_by_side"] == [
        "reproducible_total_levels",
        "live_submittable_level_count",
        "reproducible_vs_submittable_gap",
        "generic_transfer_rate_over_variants",
        "generic_transfer_ci",
        "action_efficiency_score",
        "action_efficiency_ci",
    ]
    assert coheadline["reproducible_total_levels"] == 5
    assert coheadline["live_submittable_level_count"] == 3
    assert coheadline["reproducible_vs_submittable_gap"] == 2
    assert coheadline["generic_transfer_rate_over_variants"] == pytest.approx(0.125)
    assert coheadline["action_efficiency_score"] == pytest.approx(0.75)


def test_scenario_capstone_4586_artifact_schema_and_required_fields(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4586-FIELD-PRINCIPLES: artifact exposes the honest score."""

    artifact = mod.build_artifact(
        root=tmp_path,
        registry=_registry(),
        package=_package(),
        package_path="results/fixture_package.json",
        b1_artifact=_b1_artifact(),
        preconditions_checked={"ok": True, "fixture": True},
    )

    assert artifact["honest_verdict"] == "shipped: live_submittable_coheadline_wired"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["live_submittable_level_count"] == 3
    assert artifact["reproducible_total_levels"] == 5
    assert artifact["reproducible_vs_submittable_gap"] == 2
    assert artifact["generic_transfer_rate_over_variants"] == pytest.approx(0.125)
    assert artifact["action_efficiency_score"] == pytest.approx(0.75)
    assert "live-submittable = the honest leaderboard score" in artifact["honest_metric_framing"]
    assert artifact["metric_wired_into_capstone"]["live_submittable_subset_of_reproducible"] is True
    assert artifact["tests_added_pass"]["passed"] is True
    assert artifact["leaderboard_submission"] is False
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4586_run_writes_artifact_from_files(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4586: run writes a stable artifact from registry and package files."""

    (tmp_path / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    spec_path = tmp_path / "openspec" / "capabilities" / "capstone" / "spec.md"
    spec_path.parent.mkdir(parents=True)
    spec_path.write_text(SPEC_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    registry_path = tmp_path / live_metrics.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(
        "schema_version: 1\n"
        "reproducible_total_levels: 5\n"
        "games:\n"
        "- game: with_traj\n"
        "  reproducibility: reproduced\n"
        "  levels_reproduced: 2\n"
        "- game: no_traj\n"
        "  reproducibility: reproduced\n"
        "  levels_reproduced: 2\n"
        "- game: adaptive\n"
        "  reproducibility: reproduced\n"
        "  levels_reproduced: 1\n",
        encoding="utf-8",
    )
    _write_json(tmp_path / live_metrics.DEFAULT_PACKAGE_RELATIVE_PATH, _package())
    _write_json(tmp_path / mod.B1_COHEADLINE_RELATIVE_PATH, _b1_artifact())
    _write_json(tmp_path / mod.CAPSTONE_422_RELATIVE_PATH, {"experiment": 4578})
    _write_json(tmp_path / mod.LIVE_SCORECARD_RELATIVE_PATH, {"live_total_levels": 33})

    artifact = mod.run(tmp_path, write=True)

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]
    assert written["live_submittable_level_count"] == 3
    assert written["reproducible_vs_submittable_gap"] == 2

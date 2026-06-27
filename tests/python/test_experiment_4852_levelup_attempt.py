"""Tests for Exp 4852 ARC rotated level-up attempt ledger.

Spec refs: REQ-ARC-WMTE-4852,
SCENARIO-ARC-WMTE-4852-ROTATED-TARGET,
SCENARIO-ARC-WMTE-4852-REPRODUCTION-GATE,
SCENARIO-ARC-WMTE-4852-STABLE-ARTIFACT.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from carnot import experiment_4852_levelup_attempt as exp4852


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _registry_text() -> str:
    return """schema_version: 1
games:
- game: ka59
  reproducibility: reproduced
  levels_reproduced: 1
- game: g50t
  reproducibility: reproduced
  levels_reproduced: 1
  dead_ends:
  - g50t: clone_replay_L2_route_reached_distance_12_no_bank
- game: s5i5
  reproducibility: reproduced
  levels_reproduced: 1
  mechanic_class: config_toggle_marker_coverage
- game: wa30
  reproducibility: reproduced
  levels_reproduced: 1
  dead_ends:
  - wa30: hidden-state-bound registry row
- game: r11l
  reproducibility: reproduced
  levels_reproduced: 1
  dead_ends:
  - r11l: prefix_rooted_graph_search_stalled_at_L1
reproducible_total_levels: 65
"""


def _recommendation(game: str = "s5i5") -> dict[str, object]:
    return {
        "target_game": game,
        "recommended": [{"game": "ft09", "similarity": 6.0}],
        "selected_generic_operators": [{"operator": "config_rule_verifier"}],
        "cautions": ["avoid repeating no-grounded-delta walls"],
    }


def _preconditions(game: str = "s5i5") -> dict[str, object]:
    return {
        "AGENTS.md": True,
        "CODEX.md": True,
        "offline_arcade": {"ok": True, "check": "arc_solver_kit.offline_arcade()"},
        "registry_loadable": {"ok": True, "path": "ops/arc_solve_registry.yaml"},
        "target_offline_env": {"game": game, "ok": True},
        "induction_needed": False,
        "qwen_igpu": {"needed": False, "ok": None},
    }


def _needs_re_loop_result(game: str = "s5i5") -> dict[str, object]:
    return {
        "game": game,
        "status": "needs_per_game_RE",
        "mode": "standing_arc_loop_routing_only",
        "transfer_recommendation": [{"game": "ft09"}],
        "selected_generic_operators": [{"operator": "config_rule_verifier"}],
        "guidance": "Reverse-engineer this game's win/action/state DELTA.",
    }


def _success_loop_result(game: str = "s5i5", reached_level: int = 2) -> dict[str, object]:
    return {
        "game": game,
        "reached_level": reached_level,
        "offline_reproduced": True,
        "reproduced_levels": reached_level,
        "solve_provenance": "live_agent_self_discovery",
        "mode": "standing_arc_loop_graph_explore_no_quota",
        "reproduction_gate": {
            "game": game,
            "claimed_level": reached_level,
            "reached_level": reached_level,
            "reproduced": True,
        },
        "solution_labels": ["seed", "tail"],
    }


def test_req_arc_wmte_4852_spec_declares_rotated_contract() -> None:
    """REQ-ARC-WMTE-4852: OpenSpec anchors fields, scenarios, and result path."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-ARC-WMTE-4852",
        "SCENARIO-ARC-WMTE-4852-ROTATED-TARGET",
        "SCENARIO-ARC-WMTE-4852-REPRODUCTION-GATE",
        "SCENARIO-ARC-WMTE-4852-STABLE-ARTIFACT",
        exp4852.RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in exp4852.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4852_selects_s5i5_and_excludes_ka59() -> None:
    """SCENARIO-ARC-WMTE-4852-ROTATED-TARGET: rotate off ka59 and pick s5i5."""

    selection = exp4852.select_rotation_target(
        yaml.safe_load(_registry_text()),
        approach_recommendation=_recommendation("s5i5"),
    )

    assert selection["game"] == "s5i5"
    assert selection["prior_level"] == 1
    assert selection["target_level"] == 2
    assert selection["reason"] == "grounded_marker_coverage_delta_adapter_needed"
    assert selection["approach_recommendation"] == _recommendation("s5i5")
    assert all(row["game"] != "ka59" for row in selection["candidate_audit"])
    assert [row["game"] for row in selection["candidate_audit"]] == [
        "g50t",
        "s5i5",
        "wa30",
        "r11l",
    ]
    assert selection["candidate_audit"][0]["status"] == "skip_prior_no_bank_wall"
    assert selection["candidate_audit"][1]["status"] == "selected"


def test_scenario_arc_wmte_4852_summarizes_needs_re_without_bank() -> None:
    """SCENARIO-ARC-WMTE-4852-REPRODUCTION-GATE: needs_RE is a no-bank residual."""

    attempt = exp4852.summarize_loop_attempt(
        selection={
            "game": "s5i5",
            "prior_level": 1,
            "target_level": 2,
            "reason": "grounded_marker_coverage_delta_adapter_needed",
        },
        loop_result=_needs_re_loop_result(),
        loop_result_path="results/arc_loop_solve_s5i5.json",
    )

    assert attempt["game"] == "s5i5"
    assert attempt["reached_level"] == 0
    assert attempt["offline_reproduced_new_depth"] is False
    assert attempt["new_levels_banked"] == 0
    assert attempt["residual_cause"] == "needs_per_game_RE"
    assert "needs_per_game_RE" in attempt["dead_end"]


def test_req_arc_wmte_4852_builds_no_bank_artifact_without_fabrication() -> None:
    """REQ-ARC-WMTE-4852: no-bank artifact preserves the registry total."""

    registry = yaml.safe_load(_registry_text())
    selection = exp4852.select_rotation_target(
        registry,
        approach_recommendation=_recommendation("s5i5"),
    )
    attempts = [
        exp4852.summarize_loop_attempt(
            selection=selection,
            loop_result=_needs_re_loop_result(),
            loop_result_path="results/arc_loop_solve_s5i5.json",
        )
    ]

    artifact = exp4852.build_artifact(
        registry=registry,
        selection=selection,
        attempts=attempts,
        preconditions_checked=_preconditions("s5i5"),
    )

    assert artifact["honest_verdict"] == "complete_s5i5_no_new_level_residual_needs_per_game_RE"
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["target_game"] == "s5i5"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["new_levels_banked"] == 0
    assert artifact["inference_substrate"] == "adapter_free_graph_explore_no_induction"
    assert artifact["retire_if_same_verdict"] is True
    assert artifact["registry_update"]["reproducible_total_levels_after"] == 65
    assert artifact["schema_errors"] == []
    assert exp4852.artifact_schema_errors(artifact) == []


def test_req_arc_wmte_4852_success_requires_new_reproduced_depth(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4852: success requires a gate above prior registry depth."""

    registry = yaml.safe_load(_registry_text())
    selection = exp4852.select_rotation_target(
        registry,
        approach_recommendation=_recommendation("s5i5"),
    )
    attempts = [
        exp4852.summarize_loop_attempt(
            selection=selection,
            loop_result=_success_loop_result(),
            loop_result_path="results/arc_loop_solve_s5i5.json",
        )
    ]

    artifact = exp4852.build_artifact(
        registry=registry,
        selection=selection,
        attempts=attempts,
        preconditions_checked=_preconditions("s5i5"),
    )
    output = exp4852.write_artifact(artifact, tmp_path / "experiment_4852_levelup_attempt.json")
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert saved["honest_verdict"] == "success_s5i5_levelup_banked"
    assert saved["offline_reproduced"] is True
    assert saved["reproduced_levels"] == 2
    assert saved["new_levels_banked"] == 1
    assert saved["registry_update"]["updated"] is True
    assert saved["registry_update"]["reproducible_total_levels_after"] == 66
    assert saved["schema_errors"] == []


def test_req_arc_wmte_4852_blocks_missing_target_env() -> None:
    """REQ-ARC-WMTE-4852: missing target environments produce blocked artifacts."""

    registry = yaml.safe_load(_registry_text())
    selection = exp4852.select_rotation_target(registry)
    preconditions = _preconditions("s5i5")
    preconditions["target_offline_env"] = {"game": "s5i5", "ok": False}

    artifact = exp4852.build_artifact(
        registry=registry,
        selection=selection,
        attempts=[],
        preconditions_checked=preconditions,
    )

    assert artifact["honest_verdict"] == "blocked_s5i5_offline_env_missing"
    assert artifact["offline_reproduced"] is False
    assert artifact["new_levels_banked"] == 0
    assert artifact["registry_update"]["updated"] is False
    assert artifact["schema_errors"] == []


def test_req_arc_wmte_4852_covers_defensive_no_bank_branches() -> None:
    """REQ-ARC-WMTE-4852: defensive residuals stay honest and non-banking."""

    registry = yaml.safe_load(
        """schema_version: 1
games:
- game: g50t
  levels_reproduced: 1
- game: s5i5
  levels_reproduced: 2
- game: wa30
  levels_reproduced: 2
- game: r11l
  levels_reproduced: 2
reproducible_total_levels: 7
"""
    )
    selection = exp4852.select_rotation_target(registry)

    assert selection["game"] == "none"
    assert selection["candidate_audit"][0]["status"] == "candidate_unselected"
    assert selection["candidate_audit"][1]["status"] == "skip_not_l1_only"

    existing_depth = exp4852.summarize_loop_attempt(
        selection={"game": "s5i5", "prior_level": 1, "target_level": 2},
        loop_result={
            "offline_reproduced": True,
            "reproduction_gate": {"reached_level": "not-an-int", "reproduced": True},
        },
        loop_result_path="results/arc_loop_solve_s5i5.json",
    )
    failed_gate = exp4852.summarize_loop_attempt(
        selection={"game": "s5i5", "prior_level": 1, "target_level": 2},
        loop_result={"offline_reproduced": False, "reached_level": 2},
        loop_result_path="results/arc_loop_solve_s5i5.json",
    )
    artifact_existing = exp4852.build_artifact(
        registry=registry,
        selection={"game": "s5i5", "prior_level": 1, "target_level": 2},
        attempts=[existing_depth, failed_gate],
        preconditions_checked=_preconditions("s5i5"),
    )
    artifact_no_attempts = exp4852.build_artifact(
        registry=registry,
        selection={"game": "s5i5", "prior_level": 1, "target_level": 2},
        attempts=[],
        preconditions_checked=_preconditions("s5i5"),
    )

    assert existing_depth["residual_cause"] == "reproduced_existing_or_lower_level"
    assert failed_gate["residual_cause"] == "offline_reproduction_failed"
    assert artifact_existing["honest_verdict"].endswith("_residual_existing_depth")
    assert artifact_no_attempts["honest_verdict"].endswith("_residual_no_attempts")
    assert artifact_existing["schema_errors"] == []
    assert artifact_no_attempts["schema_errors"] == []

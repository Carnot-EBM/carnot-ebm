"""Tests for Exp 4842 ARC rotated level-up attempt ledger.

Spec refs: REQ-ARC-WMTE-4842,
SCENARIO-ARC-WMTE-4842-ROTATION-TARGET,
SCENARIO-ARC-WMTE-4842-REPRODUCTION-GATE,
SCENARIO-ARC-WMTE-4842-STABLE-ARTIFACT.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from carnot import experiment_4842_levelup_attempt as exp4842


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _registry_text() -> str:
    return """schema_version: 1
games:
- game: bp35
  reproducibility: reproduced
  levels_reproduced: 2
- game: sb26
  reproducibility: reproduced
  levels_reproduced: 2
- game: lf52
  reproducibility: reproduced
  levels_reproduced: 2
- game: ka59
  reproducibility: reproduced
  levels_reproduced: 1
- game: cd82
  reproducibility: reproduced
  levels_reproduced: 2
reproducible_total_levels: 65
"""


def _loop_result(game: str, reached_level: int, reproduced: bool = True) -> dict[str, object]:
    return {
        "game": game,
        "reached_level": reached_level,
        "offline_reproduced": reproduced,
        "reproduced_levels": reached_level if reproduced else 0,
        "solve_provenance": "development_proxy",
        "mode": "standing_arc_loop_offline_no_quota",
        "learned_verifier_checkpoint": f"models/arc_verifier_{game}.json",
        "reproduction_gate": {
            "game": game,
            "reached_level": reached_level,
            "claimed_level": reached_level,
            "reproduced": reproduced,
            "mode": "offline_reproduction_gate_no_quota",
        },
        "solution_labels": ["seed", "tail"],
    }


def _preconditions(game: str = "ka59") -> dict[str, object]:
    return {
        "AGENTS.md": True,
        "CODEX.md": True,
        "offline_arcade": {"ok": True, "check": "arc_solver_kit.offline_arcade()"},
        "registry_loadable": {"ok": True, "path": "ops/arc_solve_registry.yaml"},
        "target_offline_env": {"game": game, "ok": True},
        "induction_needed": False,
        "qwen_igpu": {"needed": False, "ok": None},
    }


def _recommendation(game: str = "ka59") -> dict[str, object]:
    return {
        "game": game,
        "recommended": "reuse_standing_loop_delta",
        "selected_generic_operators": [{"operator": "graph_astar_action_cost"}],
        "guidance": ["derive only the per-game delta"],
    }


def test_req_arc_wmte_4842_spec_declares_contract() -> None:
    """REQ-ARC-WMTE-4842: OpenSpec declares the 4842 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-WMTE-4842") : spec.index("### REQ-ARC-WMTE-4832")]

    for ref in exp4842.SPEC_REFS:
        assert ref in section
    assert exp4842.RESULT_RELATIVE_PATH in section
    for field, principle in exp4842.FIELD_PRINCIPLES.items():
        assert field in section
        assert principle in section


def test_scenario_arc_wmte_4842_selects_shallowest_after_public_contacts() -> None:
    """SCENARIO-ARC-WMTE-4842-ROTATION-TARGET: sb26/lf52/bp35 done means deepen ka59."""

    registry = yaml.safe_load(_registry_text())

    selection = exp4842.select_rotation_target(
        registry,
        adaptered_games={"bp35", "sb26", "lf52", "ka59", "cd82"},
        approach_recommendation=_recommendation("ka59"),
    )

    assert selection["game"] == "ka59"
    assert selection["prior_level"] == 1
    assert selection["target_level"] == 2
    assert selection["reason"] == "shallowest_already_solved_deepen"
    assert selection["approach_recommendation"] == _recommendation("ka59")
    assert selection["public_rotation"] == [
        {"game": "sb26", "known": True, "prior_level": 2, "status": "already_reproduced"},
        {"game": "lf52", "known": True, "prior_level": 2, "status": "already_reproduced"},
        {"game": "bp35", "known": True, "prior_level": 2, "status": "already_reproduced"},
    ]
    assert selection["rotate_if_no_bank"][0] == {
        "game": "bp35",
        "prior_level": 2,
        "target_level": 3,
        "reason": "shallowest_already_solved_deepen",
    }


def test_scenario_arc_wmte_4842_public_first_contact_order() -> None:
    """SCENARIO-ARC-WMTE-4842-ROTATION-TARGET: unreproduced lf52 follows sb26."""

    registry = yaml.safe_load(_registry_text())
    for row in registry["games"]:
        if row["game"] == "lf52":
            row["levels_reproduced"] = 0

    selection = exp4842.select_rotation_target(
        registry,
        adaptered_games={"bp35", "sb26", "lf52", "ka59"},
    )

    assert selection["game"] == "lf52"
    assert selection["prior_level"] == 0
    assert selection["target_level"] == 1
    assert selection["reason"] == "preferred_public_first_contact"


def test_scenario_arc_wmte_4842_no_adaptered_target_is_explicit() -> None:
    """SCENARIO-ARC-WMTE-4842-ROTATION-TARGET: no solved adapter emits no target."""

    selection = exp4842.select_rotation_target({"games": []}, adaptered_games=set())

    assert selection == {
        "game": "none",
        "prior_level": 0,
        "target_level": 0,
        "reason": "no_reproduced_standing_loop_target",
        "public_rotation": [
            {"game": "sb26", "known": False, "prior_level": 0, "status": "unreproduced"},
            {"game": "lf52", "known": False, "prior_level": 0, "status": "unreproduced"},
            {"game": "bp35", "known": False, "prior_level": 0, "status": "unreproduced"},
        ],
        "rotate_if_no_bank": [],
        "shallowest_solved_candidates": [],
        "approach_recommendation": {},
    }


def test_scenario_arc_wmte_4842_same_depth_attempt_does_not_bank() -> None:
    """SCENARIO-ARC-WMTE-4842-REPRODUCTION-GATE: same-depth gates retire with no bank."""

    attempt = exp4842.summarize_loop_attempt(
        selection={"game": "ka59", "prior_level": 1, "target_level": 2, "reason": "unit"},
        loop_result=_loop_result("ka59", 1),
        loop_result_path="results/arc_loop_solve_ka59.json",
    )

    assert attempt["offline_reproduced_existing_depth"] is True
    assert attempt["offline_reproduced_new_depth"] is False
    assert attempt["new_levels_banked"] == 0
    assert attempt["residual_cause"] == "reproduced_existing_or_lower_level"
    assert "same-depth" in attempt["dead_end"]


def test_req_arc_wmte_4842_builds_no_bank_artifact_without_fabrication() -> None:
    """REQ-ARC-WMTE-4842: no-bank artifact preserves the registry total."""

    registry = yaml.safe_load(_registry_text())
    selection = exp4842.select_rotation_target(
        registry,
        adaptered_games={"bp35", "sb26", "lf52", "ka59", "cd82"},
        approach_recommendation=_recommendation("ka59"),
    )
    attempts = [
        exp4842.summarize_loop_attempt(
            selection=selection,
            loop_result=_loop_result("ka59", 1),
            loop_result_path="results/arc_loop_solve_ka59.json",
        ),
        exp4842.summarize_loop_attempt(
            selection=selection["rotate_if_no_bank"][0],
            loop_result=_loop_result("bp35", 2),
            loop_result_path="results/arc_loop_solve_bp35.json",
        ),
    ]

    artifact = exp4842.build_artifact(
        registry=registry,
        selection=selection,
        attempts=attempts,
        preconditions_checked=_preconditions("ka59"),
    )

    assert artifact["honest_verdict"] == "complete_ka59_no_new_level_residual_existing_depth"
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["new_levels_banked"] == 0
    assert artifact["retire_if_same_verdict"] is True
    assert artifact["registry_update"]["reproducible_total_levels_after"] == 65
    assert artifact["field_principles"]["honest_verdict"] == (
        "terminal prefix; banked is success_, no-bank is "
        "complete_<game>_no_new_level_residual_<cause>."
    )
    assert artifact["schema_errors"] == []
    assert exp4842.artifact_schema_errors(artifact) == []


def test_req_arc_wmte_4842_success_requires_new_reproduced_depth(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4842: success requires a gate above prior registry depth."""

    registry = yaml.safe_load(_registry_text())
    selection = exp4842.select_rotation_target(
        registry,
        adaptered_games={"bp35", "sb26", "lf52", "ka59", "cd82"},
        approach_recommendation=_recommendation("ka59"),
    )
    attempts = [
        exp4842.summarize_loop_attempt(
            selection=selection,
            loop_result=_loop_result("ka59", 2),
            loop_result_path="results/arc_loop_solve_ka59.json",
        )
    ]

    artifact = exp4842.build_artifact(
        registry=registry,
        selection=selection,
        attempts=attempts,
        preconditions_checked=_preconditions("ka59"),
    )
    output = exp4842.write_artifact(artifact, tmp_path / "experiment_4842_levelup_attempt.json")
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert saved["honest_verdict"] == "success_ka59_L2_offline_reproduced"
    assert saved["offline_reproduced"] is True
    assert saved["reproduced_levels"] == 2
    assert saved["new_levels_banked"] == 1
    assert saved["target_game"] == "ka59"
    assert saved["registry_update"]["updated"] is True
    assert saved["registry_update"]["reproducible_total_levels_after"] == 66
    assert saved["schema_errors"] == []


def test_req_arc_wmte_4842_blocks_missing_target_env() -> None:
    """REQ-ARC-WMTE-4842: missing target environments produce blocked artifacts."""

    registry = yaml.safe_load(_registry_text())
    selection = exp4842.select_rotation_target(registry, adaptered_games={"ka59"})
    preconditions = _preconditions("ka59")
    preconditions["target_offline_env"] = {"game": "ka59", "ok": False}

    artifact = exp4842.build_artifact(
        registry=registry,
        selection=selection,
        attempts=[],
        preconditions_checked=preconditions,
    )

    assert artifact["honest_verdict"] == "blocked_ka59_offline_env_missing"
    assert artifact["offline_reproduced"] is False
    assert artifact["new_levels_banked"] == 0
    assert artifact["registry_update"]["updated"] is False

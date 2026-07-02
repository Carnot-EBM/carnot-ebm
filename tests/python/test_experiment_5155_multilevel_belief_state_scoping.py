"""Tests for Exp 5155 multi-level ARC belief-state scoping.

Spec refs: REQ-ARC-WMTE-5155,
SCENARIO-ARC-WMTE-5155-RESET-CHARACTERIZATION,
SCENARIO-ARC-WMTE-5155-FALSIFIABLE-PROPOSALS.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from carnot import experiment_5155_multilevel_belief_state_scoping_v472 as exp5155


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _registry() -> dict[str, object]:
    return yaml.safe_load(
        """schema_version: 1
updated: '2026-06-30'
games:
- game: lp85
  reproducibility: reproduced
  levels_reproduced: 5
  mechanic_class: graph_rotation_alignment
  win_condition: align each moveable piece with its goal sprite by button rotations.
  action_model: keyboard rotations and moves.
  dead_ends:
  - 'Exp5040 lp85 no-bank no_grounded_l6_delta'
- game: ka59
  reproducibility: reproduced
  levels_reproduced: 1
  mechanic_class: push_block_step_counter
  win_condition: selected block must push through wall under StepCounter.
  action_model: ACTION6 click block selection plus keyboard pushes.
- game: wa30
  reproducibility: reproduced
  levels_reproduced: 1
  win_condition: first-solve L1; ACTION + kind sequence replays offline.
- game: untouched
  reproducibility: none
  levels_reproduced: 0
reproducible_total_levels: 69
"""
    )


def _checks() -> dict[str, object]:
    return {
        "AGENTS.md": True,
        "CODEX.md": True,
        "CLAUDE.md_live_framing_read": True,
        "arc_live_ttt.py_read": True,
        "arc_agi3_world_model.py_read": True,
        "arc_competition_agent.py_read": True,
        "arc_solve_registry.yaml_read": True,
        "spec_has_req_5155": True,
    }


def test_req_arc_wmte_5155_spec_declares_scoping_contract() -> None:
    """REQ-ARC-WMTE-5155: OpenSpec anchors the Exp5155 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in exp5155.SPEC_REFS + (exp5155.RESULT_RELATIVE_PATH,):
        assert marker in spec
    for field, principle in exp5155.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_5155_reset_characterization_is_code_verified() -> None:
    """SCENARIO-ARC-WMTE-5155-RESET-CHARACTERIZATION: reset fact has code evidence."""

    state = exp5155.current_state_characterization()

    assert state["belief_state_resets_at_level_boundary"] is True
    assert state["reset_scope"] == "active_world_model_induction_slice"
    assert state["preserved_state"] == "explorer_graph_and_navigation_edges"
    assert any("_episode_transition_start" in row for row in state["code_evidence"])
    assert any("gated_engine_from_transitions" in row for row in state["code_evidence"])
    assert "StepwiseExplorer" in state["nuance"]


def test_scenario_arc_wmte_5155_characterizes_each_solved_registry_game() -> None:
    """SCENARIO-ARC-WMTE-5155-RESET-CHARACTERIZATION: all solved games get loss notes."""

    registry = _registry()
    registry["games"].extend(
        [
            "not-a-row",
            {"game": "bad-level", "levels_reproduced": "not-an-int"},
        ]
    )
    rows = exp5155.characterize_games(registry)
    by_game = {row["game"]: row for row in rows}

    assert set(by_game) == {"ka59", "lp85", "wa30"}
    assert by_game["lp85"]["status"] == "deepened_but_stuck"
    assert by_game["ka59"]["status"] == "shallow_solved_l1"
    assert by_game["wa30"]["status"] == "shallow_solved_l1"
    assert by_game["ka59"]["information_thrown_away_by_active_reset"]["hidden_or_register_state"]
    assert by_game["lp85"]["information_thrown_away_by_active_reset"]["goal_structure"]
    assert by_game["wa30"]["information_thrown_away_by_active_reset"]["action_effects"]
    assert all(row["belief_state_reset_applies"] is True for row in rows)


def test_scenario_arc_wmte_5155_falsifiable_proposals_are_ranked() -> None:
    """SCENARIO-ARC-WMTE-5155-FALSIFIABLE-PROPOSALS: 2-3 small gates with controls."""

    proposals = exp5155.proposed_experiments()
    ranking = exp5155.rank_proposals(proposals)

    assert 2 <= len(proposals) <= 3
    assert [row["name"] for row in ranking] == [
        "transition_slice_warm_start_replay_ablation",
        "cross_level_goal_energy_ranker_replay",
        "hidden_register_hazard_belief_carryover_probe",
    ]
    for proposal in proposals:
        assert set(proposal) == {
            "name",
            "hypothesis",
            "falsifiable_gate",
            "estimated_effort",
            "control",
            "signal",
            "effort_rank",
            "signal_rank",
            "distinct_from_exp5154",
        }
        assert "cold" in proposal["control"]
        assert proposal["distinct_from_exp5154"] is True


def test_req_arc_wmte_5155_artifact_schema_and_checksum(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-5155: artifact has required value/principle fields and checksum."""

    artifact = exp5155.build_artifact(_registry(), preconditions_checked=_checks())

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["belief_state_resets_at_level_boundary"]["value"] is True
    assert (
        artifact["belief_state_resets_at_level_boundary"]["principle"]
        == exp5155.FIELD_PRINCIPLES["belief_state_resets_at_level_boundary"]["principle"]
    )
    assert len(artifact["proposed_experiments"]["value"]) == 3
    assert artifact["reproducible_total_levels"] == 69
    assert artifact["registry_solved_games"] == 3
    assert artifact["registry_never_contacted_games"] == ["untouched"]
    assert artifact["reproducibility_checksum"] == exp5155.reproducibility_checksum(artifact)
    exp5155.validate_artifact(artifact)

    output = tmp_path / exp5155.RESULT_RELATIVE_PATH
    exp5155.write_artifact(artifact, output)
    reloaded = json.loads(output.read_text(encoding="utf-8"))
    assert reloaded == artifact


def test_req_arc_wmte_5155_validation_fails_closed() -> None:
    """REQ-ARC-WMTE-5155: malformed scoping artifacts are rejected."""

    artifact = exp5155.build_artifact(_registry(), preconditions_checked=_checks())

    missing = dict(artifact)
    missing.pop("schema")
    missing["reproducibility_checksum"] = exp5155.reproducibility_checksum(missing)
    with pytest.raises(ValueError, match="artifact missing fields"):
        exp5155.validate_artifact(missing)

    bad_verdict = dict(artifact, honest_verdict="done")
    bad_verdict["reproducibility_checksum"] = exp5155.reproducibility_checksum(bad_verdict)
    with pytest.raises(ValueError, match="honest_verdict"):
        exp5155.validate_artifact(bad_verdict)

    bad_reset = dict(artifact)
    bad_reset["belief_state_resets_at_level_boundary"] = {"value": "yes", "principle": "x"}
    bad_reset["reproducibility_checksum"] = exp5155.reproducibility_checksum(bad_reset)
    with pytest.raises(ValueError, match="belief_state_resets_at_level_boundary.value"):
        exp5155.validate_artifact(bad_reset)

    bad_count = dict(artifact)
    bad_count["proposed_experiments"] = {
        **artifact["proposed_experiments"],
        "value": artifact["proposed_experiments"]["value"] * 2,
    }
    bad_count["reproducibility_checksum"] = exp5155.reproducibility_checksum(bad_count)
    with pytest.raises(ValueError, match="proposed_experiments"):
        exp5155.validate_artifact(bad_count)

    bad_proposals_shape = dict(artifact, proposed_experiments=[])
    bad_proposals_shape["reproducibility_checksum"] = exp5155.reproducibility_checksum(
        bad_proposals_shape
    )
    with pytest.raises(ValueError, match="proposed_experiments must be"):
        exp5155.validate_artifact(bad_proposals_shape)

    bad_proposal_entry = dict(artifact)
    bad_proposal_entry["proposed_experiments"] = {
        **artifact["proposed_experiments"],
        "value": [{"name": "missing_gate"}, *artifact["proposed_experiments"]["value"][1:]],
    }
    bad_proposal_entry["reproducibility_checksum"] = exp5155.reproducibility_checksum(
        bad_proposal_entry
    )
    with pytest.raises(ValueError, match="entries missing required fields"):
        exp5155.validate_artifact(bad_proposal_entry)

    bad_checksum = dict(artifact, reproducibility_checksum="sha256:bad")
    with pytest.raises(ValueError, match="checksum"):
        exp5155.validate_artifact(bad_checksum)

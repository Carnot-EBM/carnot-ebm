"""Tests for Exp 5158 cross-level goal-energy ranker replay.

Spec refs: REQ-ARC-WMTE-5158,
SCENARIO-ARC-WMTE-5158-DYNAMITE-WARM-START,
SCENARIO-ARC-WMTE-5158-TARGET-PREFIX-RANK,
SCENARIO-ARC-WMTE-5158-STABLE-ARTIFACT.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5158_deepen_goal_energy_ranker_replay_v473 as exp5158
from carnot.agentic.arc_goal_energy_live import GoalSatisfactionEnergy


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
PREDICATE_CODE = 'def is_goal(state):\n    return state["unsatisfied_targets"] == 0\n'


def _candidate(label: str, latent: tuple[float, float], unsatisfied: int = 1) -> dict[str, object]:
    return {
        "label": label,
        "action": 6,
        "data": {"x": int(latent[0] * 10), "y": int(latent[1] * 10)},
        "state": {
            "total_targets": 4,
            "satisfied_targets": 4 - unsatisfied,
            "unsatisfied_targets": unsatisfied,
            "latent_features": list(latent),
        },
    }


def _case(
    game: str,
    *,
    target_prefix: str = "target",
    control_level: int = 2,
    warm_level: int = 2,
) -> exp5158.RankingCase:
    return exp5158.RankingCase(
        game=game,
        level_from=1,
        level_to=2,
        win_near_win_states=(
            {
                "total_targets": 4,
                "satisfied_targets": 4,
                "unsatisfied_targets": 0,
                "latent_features": [0.0, 0.0],
            },
            {
                "total_targets": 4,
                "satisfied_targets": 3,
                "unsatisfied_targets": 1,
                "latent_features": [0.1, 0.0],
            },
        ),
        frontier_candidates=(
            _candidate("decoy_a", (0.9, 0.9)),
            _candidate("target", (0.05, 0.0)),
            _candidate("decoy_b", (1.0, 1.0)),
        ),
        target_prefix_label=target_prefix,
        cold_level_reached=control_level,
        warmstart_level_reached=warm_level,
        source_artifact=f"results/{game}.json",
    )


def _checks() -> dict[str, object]:
    return {
        "AGENTS.md": True,
        "CODEX.md": True,
        "CLAUDE.md": True,
        "experiment_5155_read": True,
        "registry_entries_checked": ["lp85", "sc25", "tr87"],
        "artifact_trace_extraction": "passed",
    }


def test_req_arc_wmte_5158_spec_declares_ranker_contract() -> None:
    """REQ-ARC-WMTE-5158: OpenSpec anchors the Exp5158 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in exp5158.SPEC_REFS + (exp5158.RESULT_RELATIVE_PATH,):
        assert marker in spec
    for field, principle in exp5158.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_5158_dynamite_warm_start_reranks_target_prefix() -> None:
    """SCENARIO-ARC-WMTE-5158-DYNAMITE-WARM-START: level-N energy state carries over."""

    goal_energy = GoalSatisfactionEnergy.from_predicate_code(PREDICATE_CODE)
    row = exp5158.evaluate_ranking_case(_case("toy"), goal_energy=goal_energy)

    assert row["cold_target_rank"] == 2
    assert row["warmstart_target_rank"] == 1
    assert row["reciprocal_rank_cold"] == 0.5
    assert row["reciprocal_rank_warmstart"] == 1.0
    assert row["warmstart_model"]["mechanism"] == "DynaMITE-style terminal carryover"
    assert row["fit_uses_target_prefix"] is False


def test_scenario_arc_wmte_5158_target_prefix_reciprocal_rank_handles_misses() -> None:
    """SCENARIO-ARC-WMTE-5158-TARGET-PREFIX-RANK: missing target prefix scores zero."""

    ranked = [{"label": "a"}, {"label": "b"}]

    assert exp5158.target_prefix_rank(ranked, "b") == 2
    assert exp5158.target_prefix_reciprocal_rank(ranked, "b") == 0.5
    assert exp5158.target_prefix_rank(ranked, "missing") is None
    assert exp5158.target_prefix_reciprocal_rank(ranked, "missing") == 0.0


def test_req_arc_wmte_5158_artifact_gate_counts_games_not_transitions(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-5158-STABLE-ARTIFACT: exp5155's >=2/3 game gate is verbatim."""

    goal_energy = GoalSatisfactionEnergy.from_predicate_code(PREDICATE_CODE)
    rows = [
        exp5158.evaluate_ranking_case(_case("lp85"), goal_energy=goal_energy),
        exp5158.evaluate_ranking_case(_case("sc25"), goal_energy=goal_energy),
        exp5158.evaluate_ranking_case(_case("tr87", target_prefix="decoy_a"), goal_energy=goal_energy),
    ]
    artifact = exp5158.build_artifact(rows, preconditions_checked=_checks())

    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["games_improved_count"] == 2
    assert artifact["gate_passed"] is True
    assert artifact["games_tested"] == [
        {"game": "lp85", "n_level_transitions_tested": 1},
        {"game": "sc25", "n_level_transitions_tested": 1},
        {"game": "tr87", "n_level_transitions_tested": 1},
    ]
    assert artifact["reciprocal_rank_cold"] == {"lp85": 0.5, "sc25": 0.5, "tr87": 1.0}
    assert artifact["reciprocal_rank_warmstart"] == {"lp85": 1.0, "sc25": 1.0, "tr87": 0.5}
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["verifier_is_oracle"] is False
    assert "Exp4020 graded goal-satisfaction energy" in artifact["energy_signal_source"]
    assert "DynaMITE-style terminal carryover" in artifact["energy_signal_source"]
    assert artifact["reproducibility_checksum"] == exp5158.reproducibility_checksum(artifact)
    exp5158.validate_artifact(artifact)

    output = tmp_path / exp5158.RESULT_RELATIVE_PATH
    exp5158.write_artifact(artifact, output)
    assert json.loads(output.read_text(encoding="utf-8")) == artifact


def test_req_arc_wmte_5158_gate_fails_on_level_regression() -> None:
    """REQ-ARC-WMTE-5158: a ranking lift cannot pass with a level-reached regression."""

    goal_energy = GoalSatisfactionEnergy.from_predicate_code(PREDICATE_CODE)
    rows = [
        exp5158.evaluate_ranking_case(_case("lp85"), goal_energy=goal_energy),
        exp5158.evaluate_ranking_case(_case("sc25"), goal_energy=goal_energy),
        exp5158.evaluate_ranking_case(_case("tr87", warm_level=1), goal_energy=goal_energy),
    ]
    artifact = exp5158.build_artifact(rows, preconditions_checked=_checks())

    assert artifact["games_improved_count"] == 3
    assert artifact["no_level_regression"] is False
    assert artifact["gate_passed"] is False
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_arc_wmte_5158_validation_fails_closed() -> None:
    """REQ-ARC-WMTE-5158: malformed artifacts do not validate."""

    goal_energy = GoalSatisfactionEnergy.from_predicate_code(PREDICATE_CODE)
    rows = [exp5158.evaluate_ranking_case(_case(game), goal_energy=goal_energy) for game in exp5158.REQUIRED_GAMES]
    artifact = exp5158.build_artifact(rows, preconditions_checked=_checks())

    missing = dict(artifact)
    missing.pop("gate_passed")
    missing["reproducibility_checksum"] = exp5158.reproducibility_checksum(missing)
    with pytest.raises(ValueError, match="artifact missing fields"):
        exp5158.validate_artifact(missing)

    bad_games = dict(artifact, games_tested=[{"game": "lp85", "n_level_transitions_tested": 1}])
    bad_games["reproducibility_checksum"] = exp5158.reproducibility_checksum(bad_games)
    with pytest.raises(ValueError, match="lp85, sc25, and tr87"):
        exp5158.validate_artifact(bad_games)

    bad_provenance = dict(artifact, solve_provenance="live_agent_self_discovery")
    bad_provenance["reproducibility_checksum"] = exp5158.reproducibility_checksum(bad_provenance)
    with pytest.raises(ValueError, match="solve_provenance"):
        exp5158.validate_artifact(bad_provenance)

    bad_checksum = dict(artifact, reproducibility_checksum="sha256:bad")
    with pytest.raises(ValueError, match="checksum"):
        exp5158.validate_artifact(bad_checksum)

    bad_verdict = dict(artifact, honest_verdict="done")
    bad_verdict["reproducibility_checksum"] = exp5158.reproducibility_checksum(bad_verdict)
    with pytest.raises(ValueError, match="honest_verdict"):
        exp5158.validate_artifact(bad_verdict)

    bad_oracle = dict(artifact, verifier_is_oracle=True)
    bad_oracle["reproducibility_checksum"] = exp5158.reproducibility_checksum(bad_oracle)
    with pytest.raises(ValueError, match="verifier_is_oracle"):
        exp5158.validate_artifact(bad_oracle)

    bad_gate = dict(artifact, gate_passed="yes")
    bad_gate["reproducibility_checksum"] = exp5158.reproducibility_checksum(bad_gate)
    with pytest.raises(ValueError, match="gate_passed"):
        exp5158.validate_artifact(bad_gate)

    bad_principles = dict(artifact, field_principles=[])
    bad_principles["reproducibility_checksum"] = exp5158.reproducibility_checksum(bad_principles)
    with pytest.raises(ValueError, match="field_principles"):
        exp5158.validate_artifact(bad_principles)

    missing_principle = dict(artifact, field_principles=dict(artifact["field_principles"]))
    missing_principle["field_principles"].pop("gate_passed")
    missing_principle["reproducibility_checksum"] = exp5158.reproducibility_checksum(missing_principle)
    with pytest.raises(ValueError, match="missing principle"):
        exp5158.validate_artifact(missing_principle)


def test_req_arc_wmte_5158_defensive_helper_branches() -> None:
    """REQ-ARC-WMTE-5158: helper edge cases stay deterministic and non-crashing."""

    goal_energy = GoalSatisfactionEnergy.from_predicate_code(PREDICATE_CODE)
    empty_model = exp5158.fit_terminal_carryover_energy([{}])

    assert empty_model({}) == 0.0
    assert empty_model.diagnostics()["evidence_count"] == 0
    assert exp5158._as_float(True, default=7.0) == 7.0
    assert exp5158._as_float("bad", default=3.0) == 3.0
    assert exp5158._parse_json_label("oops") == (0, None)
    assert exp5158._parse_json_label("2") == (2, None)
    assert exp5158._parse_json_label('{"x": 4, "y": 5}') == (6, {"x": 4, "y": 5})
    assert exp5158._parse_json_label('{"action": 3}') == (3, None)

    numeric_model = exp5158.fit_terminal_carryover_energy(
        [
            {
                "total_targets": 4,
                "satisfied_targets": 3,
                "unsatisfied_targets": 1,
                "hand_verifier_energy": 2,
                "action_id": 6,
            }
        ]
    )
    assert numeric_model.feature_count == 5

    visible_ranked = exp5158.rank_candidates(
        [
            {
                "action_label": "visible",
                "visible_goal_state": {
                    "total_targets": 2,
                    "satisfied_targets": 1,
                    "unsatisfied_targets": 1,
                },
            },
            {"label": "fallback", "total_targets": 2, "satisfied_targets": 2, "unsatisfied_targets": 0},
        ],
        goal_energy=goal_energy,
    )
    assert [row["candidate_index"] for row in visible_ranked] == [1, 0]
    assert exp5158.rank_candidates([{"label": "bad"}], goal_energy=lambda _state: (_ for _ in ()).throw(RuntimeError("boom")))[0]["cold_goal_energy"] == 1.0

    blocked = exp5158.build_blocked_artifact(preconditions_checked=_checks())
    assert blocked["honest_verdict"].startswith("complete:")
    assert blocked["gate_passed"] is False
    exp5158.validate_artifact(blocked)

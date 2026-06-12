"""Tests for Exp 4071 verifier action-pruner efficiency ablation.

Spec refs: REQ-PHASE4-043, SCENARIO-PHASE4-043.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.agentic import arc_exp4071_verifier_action_pruner_efficiency as exp4071


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"


def _actions() -> list[dict[str, object]]:
    return [
        {"action": 6, "x": 10, "y": 20, "role": "select_item", "color": 3},
        {"action": 6, "x": 30, "y": 40, "role": "place_slot", "color": 3},
        {"action": 5, "role": "validate"},
    ]


def _trace(*, game_id: str = "aa00-test", decoys_per_step: int = 2) -> exp4071.SolvedGameTrace:
    return exp4071.SolvedGameTrace.from_actions(
        game_id=game_id,
        source_artifact=f"results/{game_id}.json",
        actions=_actions(),
        random_seed=4071,
        decoys_per_step=decoys_per_step,
    )


def test_req_phase4_043_spec_declares_exp4071_contract() -> None:
    """REQ-PHASE4-043: OpenSpec declares the ablation and required artifact fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-043" in spec
    assert "SCENARIO-PHASE4-043" in spec
    assert "experiment_4071_verifier_action_pruner_efficiency.json" in spec
    assert "offline_arc_agi3_verifier_action_pruning_ablation" in spec
    for field in exp4071.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_phase4_043_loads_real_env_confirmed_solved_traces(tmp_path: Path) -> None:
    """REQ-PHASE4-043: only real-env-confirmed solved traces become replay inputs."""

    solved_path = tmp_path / "solved.json"
    solved_path.write_text(
        json.dumps(
            {
                "target_game": "aa00-test",
                "game_solved": True,
                "real_env_confirmed": True,
                "first_solve_at_action": 3,
                "action_plan": _actions(),
            }
        ),
        encoding="utf-8",
    )
    trace = exp4071.load_solved_game_trace(solved_path, random_seed=4071)

    assert trace.game_id == "aa00-test"
    assert trace.recorded_action_count == 3
    assert trace.real_env_confirmed is True
    assert len(trace.rounds) == 3
    assert all(any(candidate.accepted for candidate in round_.candidates) for round_ in trace.rounds)

    unsolved_path = tmp_path / "unsolved.json"
    unsolved_path.write_text(
        json.dumps({"target_game": "bb00-test", "game_solved": False, "real_env_confirmed": True}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must be real-env-confirmed solved"):
        exp4071.load_solved_game_trace(unsolved_path, random_seed=4071)


def test_scenario_phase4_043_gap4_pruner_rejects_decoys_before_execution() -> None:
    """SCENARIO-PHASE4-043: GAP-4 action pruning preserves the winning action."""

    trace = _trace(decoys_per_step=2)

    baseline = exp4071.run_trace_baseline(trace)
    pruned = exp4071.run_trace_pruned(trace)

    assert baseline.solved is True
    assert pruned.solved is True
    assert baseline.actions_to_solve == 9
    assert pruned.actions_to_solve == 3
    assert pruned.pruned_count == 6
    assert pruned.winning_action_pruned is False
    assert all(decision["selected_accepted"] for decision in pruned.verifier_decisions)


def test_scenario_phase4_043_measurement_reports_equal_solverate_and_action_cut() -> None:
    """SCENARIO-PHASE4-043: the ablation reports action reduction at equal solve rate."""

    traces = tuple(_trace(game_id=f"game{i}-test", decoys_per_step=2) for i in range(3))
    measurement = exp4071.run_action_pruner_ablation(traces)
    artifact = exp4071.build_result_artifact(
        measurement,
        preflight=exp4071.ArcEnvPreflight(True, 25, "test ARC catalog reachable"),
        duration_s=0.5,
    )

    assert artifact["games_evaluated"] == 3
    assert artifact["actions_baseline_mean"] == 9.0
    assert artifact["actions_pruned_mean"] == 3.0
    assert artifact["action_reduction_pct"] == pytest.approx(66.6667, rel=1e-4)
    assert artifact["solverate_baseline"] == 1.0
    assert artifact["solverate_pruned"] == 1.0
    assert artifact["solverate_parity_held"] is True
    assert artifact["inference_substrate"] == exp4071.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("success: verifier_pruner_cuts_actions_66.7pct_equal_solverate")
    assert exp4071.artifact_schema_errors(artifact) == []
    assert artifact["field_principles"]["action_reduction_pct"].startswith("north-star efficiency")
    assert artifact["field_principles"]["solverate_parity_held"].startswith("positive control")


def test_scenario_phase4_043_no_gain_and_regression_verdicts_are_honest() -> None:
    """SCENARIO-PHASE4-043: null and regressed solve-rate outcomes are not promoted."""

    no_gain = exp4071.ActionPrunerMeasurement(
        baseline_runs=(exp4071.ArmRun("a", "a.json", True, 3, 0.03, 0, False, ()),),
        pruned_runs=(exp4071.ArmRun("a", "a.json", True, 3, 0.04, 0, False, ()),),
        random_seed=4071,
    )
    no_gain_artifact = exp4071.build_result_artifact(
        no_gain,
        preflight=exp4071.ArcEnvPreflight(True, 25, "ok"),
        duration_s=0.1,
        enforce_game_count=False,
    )
    assert no_gain_artifact["honest_verdict"] == "complete: verifier_pruner_no_efficiency_gain"
    assert no_gain_artifact["wallclock_reduction_pct"] < 0.0

    regression = exp4071.ActionPrunerMeasurement(
        baseline_runs=(exp4071.ArmRun("a", "a.json", True, 3, 0.03, 0, False, ()),),
        pruned_runs=(exp4071.ArmRun("a", "a.json", False, 2, 0.02, 1, True, ()),),
        random_seed=4071,
    )
    regression_artifact = exp4071.build_result_artifact(
        regression,
        preflight=exp4071.ArcEnvPreflight(True, 25, "ok"),
        duration_s=0.1,
        enforce_game_count=False,
    )
    assert regression_artifact["honest_verdict"] == "complete: verifier_pruner_regressed_solverate"
    assert regression_artifact["solverate_parity_held"] is False


def test_req_phase4_043_preflight_and_blocked_artifact_fail_closed() -> None:
    """REQ-PHASE4-043: unreachable ARC preflight produces the mandated blocked artifact."""

    class GoodArcade:
        def get_environments(self) -> list[str]:
            return ["env1", "env2"]

    good = exp4071.probe_arc_env_reachable(arcade_factory=lambda **_: GoodArcade())
    assert good.reachable is True
    assert good.environment_count == 2

    bad = exp4071.probe_arc_env_reachable(
        arcade_factory=lambda **_: (_ for _ in ()).throw(RuntimeError("network down"))
    )
    blocked = exp4071.build_blocked_artifact(preflight=bad, duration_s=0.25)

    assert bad.reachable is False
    assert blocked["honest_verdict"] == "blocked_arc_env_unreachable"
    assert blocked["games_evaluated"] == 0
    assert blocked["solverate_parity_held"] is False
    assert exp4071.artifact_schema_errors(blocked) == []


def test_req_phase4_043_result_writer_and_default_trace_selection(tmp_path: Path) -> None:
    """REQ-PHASE4-043: script support writes stable JSON from 3-5 selected traces."""

    traces = exp4071.load_default_solved_traces(repo_root=REPO, random_seed=4071, limit=3)
    assert 3 <= len(traces) <= 5
    assert all(trace.real_env_confirmed for trace in traces)

    artifact = exp4071.build_result_artifact(
        exp4071.run_action_pruner_ablation(traces),
        preflight=exp4071.ArcEnvPreflight(True, 25, "test ARC catalog reachable"),
        duration_s=0.01,
    )
    output_path = exp4071.write_result_artifact(artifact, tmp_path / "artifact.json")

    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["games_evaluated"] == len(traces)
    assert loaded["source_traces"] == [trace.source_artifact for trace in traces]


def test_req_phase4_043_defensive_branches_and_run_experiment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-PHASE4-043: defensive branches fail closed instead of over-claiming."""

    with pytest.raises(ValueError, match="at least one action"):
        exp4071.SolvedGameTrace.from_actions(game_id="empty", source_artifact="empty.json", actions=[])
    with pytest.raises(ValueError, match="decoys_per_step"):
        exp4071.build_replay_round({"action": 1}, step_index=1, game_id="aa", decoys_per_step=-1)

    assert exp4071.encode_action_grid(
        {"action": True, "x": None, "grid": [1, 2], "role": "text"},
        step_index=1,
        game_id="aa",
    )[0][0] == 1
    assert exp4071.encode_action_grid({"action": "click"}, step_index=1, game_id="aa")[0][0] > 0
    assert exp4071.build_replay_round({}, step_index=1, game_id="aa").candidates[-1].accepted is True

    solve_trace_path = tmp_path / "solve_trace.json"
    solve_trace_path.write_text(
        json.dumps(
            {
                "target_game": "cc00-test",
                "game_solved": True,
                "real_env_confirmed": True,
                "first_solve_at_action": 2,
                "solve_trace": {"actions": _actions()},
            }
        ),
        encoding="utf-8",
    )
    assert exp4071.load_solved_game_trace(solve_trace_path).recorded_action_count == 2

    missing_actions_path = tmp_path / "missing_actions.json"
    missing_actions_path.write_text(
        json.dumps({"target_game": "dd00-test", "game_solved": True, "real_env_confirmed": True}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing game id or replay actions"):
        exp4071.load_solved_game_trace(missing_actions_path)

    with pytest.raises(ValueError, match="between 3 and 5"):
        exp4071.load_default_solved_traces(repo_root=tmp_path, limit=2)
    with pytest.raises(ValueError, match="fewer than 3"):
        exp4071.load_default_solved_traces(repo_root=tmp_path, limit=3)

    trace = _trace(decoys_per_step=1)
    assert exp4071.run_trace_baseline(trace, max_actions=0).solved is False

    rejected_round = exp4071.ActionReplayRound(
        step_index=1,
        expected_action={"action": 1},
        expected_grid=((1,),),
        candidates=(
            exp4071.ReplayCandidate("bad", {"action": 2}, ((2,),), False),
        ),
    )
    rejected_trace = exp4071.SolvedGameTrace("reject", "reject.json", (rejected_round,), 1)
    assert exp4071.run_trace_baseline(rejected_trace).solved is False

    all_rejected = exp4071.run_trace_pruned(trace, reject_threshold=-1.0)
    assert all_rejected.solved is False
    assert all_rejected.winning_action_pruned is True
    assert all_rejected.verifier_decisions[0]["selected_candidate_id"] == ""

    decoy_selected = exp4071.run_trace_pruned(trace, reject_threshold=999.0)
    assert decoy_selected.solved is False
    assert decoy_selected.winning_action_pruned is True

    with pytest.raises(ValueError, match="at least one solved-game trace"):
        exp4071.run_action_pruner_ablation(())
    assert exp4071._mean([]) == 0.0
    assert exp4071._solve_rate(()) == 0.0
    assert exp4071._reduction_pct(0.0, 1.0) == 0.0

    with pytest.raises(ValueError, match="games_evaluated"):
        exp4071.build_result_artifact(
            exp4071.ActionPrunerMeasurement(
                baseline_runs=(exp4071.ArmRun("a", "a.json", True, 1, 0.01, 0, False, ()),),
                pruned_runs=(exp4071.ArmRun("a", "a.json", True, 1, 0.01, 0, False, ()),),
                random_seed=4071,
            ),
            preflight=exp4071.ArcEnvPreflight(True, 1, "ok"),
            duration_s=0.1,
        )

    malformed = {
        "honest_verdict": 1,
        "games_evaluated": "3",
        "actions_baseline_mean": "bad",
        "actions_pruned_mean": 1.0,
        "action_reduction_pct": 0.0,
        "solverate_baseline": 1.0,
        "solverate_pruned": 1.0,
        "solverate_parity_held": "yes",
        "wallclock_reduction_pct": 0.0,
        "inference_substrate": "wrong",
    }
    errors = exp4071.artifact_schema_errors(malformed)
    assert "honest_verdict must be a string" in errors
    assert "inference_substrate must declare the offline action-pruning ablation" in errors
    assert "solverate_parity_held must be a bare bool" in errors
    assert "games_evaluated must be a bare int" in errors
    assert "actions_baseline_mean must be numeric" in errors
    assert "missing required field wallclock_reduction_pct" in exp4071.artifact_schema_errors(
        {key: value for key, value in malformed.items() if key != "wallclock_reduction_pct"}
    )

    prefixed = dict(malformed)
    prefixed.update(
        {
            "honest_verdict": "maybe",
            "games_evaluated": 1,
            "actions_baseline_mean": 1.0,
            "solverate_parity_held": True,
            "inference_substrate": exp4071.INFERENCE_SUBSTRATE,
        }
    )
    assert "honest_verdict must be terminal-prefixed" in exp4071.artifact_schema_errors(prefixed)

    bad_success = dict(prefixed)
    bad_success.update(
        {
            "honest_verdict": "success: bad",
            "games_evaluated": 3,
            "solverate_parity_held": False,
            "action_reduction_pct": 0.0,
        }
    )
    success_errors = exp4071.artifact_schema_errors(bad_success)
    assert "success requires solve-rate parity" in success_errors
    assert "success requires positive action reduction" in success_errors

    assert exp4071.probe_arc_env_reachable(arcade_factory=lambda **_: type("Empty", (), {"get_environments": lambda self: []})()).reachable is False

    original_schema_errors = exp4071.artifact_schema_errors
    monkeypatch.setattr(exp4071, "artifact_schema_errors", lambda *_args, **_kwargs: ["forced"])
    with pytest.raises(ValueError, match="forced"):
        exp4071.build_blocked_artifact(
            preflight=exp4071.ArcEnvPreflight(False, 0, "blocked"),
            duration_s=0.0,
        )
    monkeypatch.setattr(exp4071, "artifact_schema_errors", original_schema_errors)

    output_path = tmp_path / "result.json"
    monkeypatch.setattr(exp4071, "probe_arc_env_reachable", lambda **_: exp4071.ArcEnvPreflight(True, 25, "ok"))
    monkeypatch.setattr(exp4071, "load_default_solved_traces", lambda **_: tuple(_trace(game_id=f"r{i}") for i in range(3)))
    artifact = exp4071.run_experiment(repo_root=tmp_path, output_path=output_path)
    assert artifact["honest_verdict"].startswith("success:")
    assert output_path.exists()

    blocked_path = tmp_path / "blocked.json"
    monkeypatch.setattr(exp4071, "probe_arc_env_reachable", lambda **_: exp4071.ArcEnvPreflight(False, 0, "blocked"))
    blocked = exp4071.run_experiment(repo_root=tmp_path, output_path=blocked_path)
    assert blocked["honest_verdict"] == "blocked_arc_env_unreachable"
    assert blocked_path.exists()

"""Tests for Exp 4330 ARC adapter-free shallow-tail discovery sweep.

Spec refs: REQ-PHASE4-077, SCENARIO-PHASE4-077.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import carnot.experiment_4330_arc_adapter_free_discovery_sweep_shallow_tail as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"


def _preconditions(*, ok: bool = True) -> dict[str, object]:
    return {
        "sweep_driver_import": ok,
        "arc_graph_explore_import": ok,
        "arc_solver_kit_import": ok,
        "offline_env_reachable": ok,
        "environment_files_present": {game: ok for game in exp.SWEEP_GAMES},
        "candidate_games": list(exp.SHALLOW_TAIL_GAMES),
        "excluded_games": list(exp.EXCLUDED_GAMES),
        "discovery_budget": exp.DEFAULT_MAX_DISCOVERY_EXPANSIONS,
        "advance_budget_inflated": False,
        "leaderboard_submission": False,
    }


def _result(game: str, *, explored: int = 5, advanced: bool = False) -> exp.GameSweepResult:
    return exp.GameSweepResult(
        game=game,
        solver="graph_explore_solve_v2",
        status="advanced" if advanced else "no_advance",
        reached_level=1 if advanced else 0,
        advanced=advanced,
        exploration_actions_used=explored,
        dead_end_class="none" if advanced else "adapter_free_no_level_delta_12000_budget",
        trajectory=[{"action": 5, "data": None}] if advanced else [],
        reproduction_gate={
            "game": game,
            "reached_level": 1 if advanced else 0,
            "claimed_level": 1,
            "reproduced": advanced,
        },
        reproduced_levels=1 if advanced else 0,
        trajectory_path=f"results/arc_explore_trajectory_{game}.json" if advanced else "",
    )


def _per_game(**overrides: exp.GameSweepResult) -> dict[str, exp.GameSweepResult]:
    rows = {game: _result(game) for game in exp.SWEEP_GAMES}
    rows.update(overrides)
    return rows


def _tn36_finding(*, advanced: bool = False) -> dict[str, object]:
    finding = exp.tn36_schema_finding_from_source(
        'result = self.camera.display_to_grid(self.action.data["x"], self.action.data["y"])'
    )
    return {
        **finding,
        "wrapped_payload_explorer_advanced": advanced,
        "exploration_actions_used": 9,
    }


def test_req_phase4_077_spec_declares_exp4330_contract() -> None:
    """REQ-PHASE4-077: OpenSpec declares the shallow-tail sweep artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-077" in spec
    assert "SCENARIO-PHASE4-077" in spec
    assert "experiment_4330_arc_adapter_free_discovery_sweep_shallow_tail.json" in spec
    assert "ARC_MAX_EXPANSIONS=12000" in spec
    assert "blocked_arc_env_unreachable" in spec
    for game in exp.SHALLOW_TAIL_GAMES:
        assert game in spec
    for game in exp.EXCLUDED_GAMES:
        assert game in spec
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for principle in exp.REQUIRED_FIELD_PRINCIPLES.values():
        assert principle in spec


def test_req_phase4_077_candidate_set_excludes_deep_tail_and_provisional() -> None:
    """REQ-PHASE4-077: the sweep game set is shallow-tail plus tn36 schema-RE only."""

    assert exp.SHALLOW_TAIL_GAMES == ("bp35", "dc22", "g50t", "lf52", "re86", "s5i5", "sb26", "vc33")
    assert exp.SWEEP_GAMES == exp.SHALLOW_TAIL_GAMES + ("tn36",)
    assert exp.EXCLUDED_GAMES == ("ar25", "ka59", "tr87", "ft09", "sc25")
    assert set(exp.SWEEP_GAMES).isdisjoint(exp.EXCLUDED_GAMES)
    assert exp.DEFAULT_MAX_DISCOVERY_EXPANSIONS == 12000


def test_scenario_phase4_077_tn36_schema_is_top_level_xy_and_normalized() -> None:
    """SCENARIO-PHASE4-077: tn36 ACTION6 click schema is reverse-engineered."""

    source = 'return self.camera.display_to_grid(self.action.data["x"], self.action.data["y"])'
    finding = exp.tn36_schema_finding_from_source(source)

    assert finding["payload_schema"] == {"x": "int display_x", "y": "int display_y"}
    assert finding["schema"] == 'ACTION6 data must be top-level {"x": int, "y": int}'
    assert finding["source_evidence"] == 'self.action.data["x"], self.action.data["y"]'
    assert finding["rejects_nested_payloads"] is True
    assert exp.normalise_tn36_click_payload({"x": "4", "y": 7}) == {"x": 4, "y": 7}
    assert exp.normalise_tn36_click_payload({"position": {"x": 4, "y": 7}}) == {"x": 4, "y": 7}
    assert exp.normalise_tn36_click_payload({"click": {"x": 4, "y": 7}}) == {"x": 4, "y": 7}
    assert exp.normalise_tn36_click_payload({"data": {"x": 4, "y": 7}}) == {"x": 4, "y": 7}
    assert exp.normalise_tn36_click_payload(None) is None
    assert exp.normalise_tn36_click_payload({"row": 1}) == {"row": 1}

    unknown = exp.tn36_schema_finding_from_source("pass")
    assert unknown["schema"] == "unknown"
    assert unknown["rejects_nested_payloads"] is False


def test_scenario_phase4_077_no_advance_artifact_is_complete_with_real_exploration() -> None:
    """SCENARIO-PHASE4-077: no-advance is valid only with actions for every game."""

    artifact = exp.build_artifact(
        per_game_results=_per_game(),
        tn36_schema_finding=_tn36_finding(),
        preconditions_checked=_preconditions(),
        random_seed=4330,
        duration_s=1.25,
    )

    assert artifact["honest_verdict"] == "complete: adapter_free_shallow_tail_no_advance_real_exploration_total13"
    assert artifact["reproducible_total_levels"] == exp.PRIOR_REPRODUCIBLE_TOTAL_LEVELS
    assert artifact["games_advanced"] == []
    assert artifact["offline_reproduced"] is False
    assert artifact["verifier_is_oracle"] is True
    assert all(
        row["exploration_actions_used"] > 0
        for row in artifact["per_game_exploration_actions"].values()
    )
    assert exp.artifact_schema_errors(artifact) == []


def test_req_phase4_077_success_artifact_counts_only_reproduced_advances() -> None:
    """REQ-PHASE4-077: reproduced L1 rows increase the cumulative total."""

    artifact = exp.build_artifact(
        per_game_results=_per_game(bp35=_result("bp35", advanced=True)),
        tn36_schema_finding=_tn36_finding(),
        preconditions_checked=_preconditions(),
        random_seed=4330,
        duration_s=2.0,
    )

    assert artifact["honest_verdict"] == "success: adapter_free_shallow_tail_1_games_advanced_total14"
    assert artifact["reproducible_total_levels"] == exp.PRIOR_REPRODUCIBLE_TOTAL_LEVELS + 1
    assert artifact["games_advanced"] == ["bp35"]
    assert artifact["offline_reproduced"] is True
    assert artifact["per_game_exploration_actions"]["bp35"]["advanced"] is True
    assert artifact["per_game_exploration_actions"]["bp35"]["reproduction_gate"]["reproduced"] is True
    assert exp.artifact_schema_errors(artifact) == []


def test_req_phase4_077_blocked_artifact_stops_without_fabricating_progress() -> None:
    """REQ-PHASE4-077: missing offline env writes the honest blocked verdict."""

    artifact = exp.blocked_artifact(
        preconditions_checked=_preconditions(ok=False),
        random_seed=4330,
        duration_s=0.0,
    )

    assert artifact["honest_verdict"] == "blocked_arc_env_unreachable"
    assert artifact["reproducible_total_levels"] == exp.PRIOR_REPRODUCIBLE_TOTAL_LEVELS
    assert artifact["games_advanced"] == []
    assert artifact["per_game_exploration_actions"] == {}
    assert artifact["offline_reproduced"] is False
    assert artifact["submitted_to_leaderboard"] is False
    assert exp.artifact_schema_errors(artifact) == []


def test_scenario_phase4_077_schema_rejects_zero_action_and_malformed_artifacts() -> None:
    """SCENARIO-PHASE4-077: zero-action rows cannot pass as decision-grade."""

    artifact = exp.build_artifact(
        per_game_results=_per_game(vc33=_result("vc33", explored=0)),
        tn36_schema_finding=_tn36_finding(),
        preconditions_checked=_preconditions(),
        random_seed=4330,
        duration_s=0.0,
    )

    assert any("vc33 exploration_actions_used must be >0" in err for err in exp.artifact_schema_errors(artifact))

    malformed = {
        **artifact,
        "honest_verdict": "invalid",
        "reproducible_total_levels": 12,
        "games_advanced": "bp35",
        "offline_reproduced": "false",
        "verifier_is_oracle": False,
        "random_seed": "4330",
        "reproducibility_checksum": "bad",
        "field_principles": {"honest_verdict": "wrong"},
    }
    errors = exp.artifact_schema_errors(malformed)
    assert any("honest_verdict must be terminal-prefixed" in err for err in errors)
    assert any("reproducible_total_levels must be >= 13" in err for err in errors)
    assert any("games_advanced must be a list" in err for err in errors)
    assert any("offline_reproduced must be a bare bool" in err for err in errors)
    assert any("verifier_is_oracle must be true" in err for err in errors)
    assert any("random_seed must be a bare int" in err for err in errors)
    assert any("reproducibility_checksum must be 64-char sha256 hex" in err for err in errors)
    assert any("principle mismatch for honest_verdict" in err for err in errors)
    assert any("missing honest_verdict" in err for err in exp.artifact_schema_errors({}))


def test_req_phase4_077_checksum_is_stable_and_sensitive() -> None:
    """REQ-PHASE4-077: checksum binds trajectories, reproduce gates, and seed."""

    rows = {game: result.to_json() for game, result in _per_game().items()}
    base = exp.compute_reproducibility_checksum(
        per_game_rows=rows,
        tn36_schema_finding=_tn36_finding(),
        random_seed=4330,
    )
    same = exp.compute_reproducibility_checksum(
        per_game_rows=rows,
        tn36_schema_finding=_tn36_finding(),
        random_seed=4330,
    )
    changed = exp.compute_reproducibility_checksum(
        per_game_rows={**rows, "bp35": {**rows["bp35"], "exploration_actions_used": 6}},
        tn36_schema_finding=_tn36_finding(),
        random_seed=4330,
    )

    assert base == same
    assert base != changed
    assert len(base) == 64


def test_scenario_phase4_077_counting_and_tn36_env_wrappers_normalize_steps() -> None:
    """SCENARIO-PHASE4-077: runtime wrappers count real actions and normalize tn36 clicks."""

    class FakeEnv:
        def __init__(self) -> None:
            self.calls: list[tuple[object, object]] = []

        def reset(self) -> dict[str, int]:
            return {"levels_completed": 0}

        def step(self, action: object, data: object = None, reasoning: object = None) -> dict[str, int]:
            self.calls.append((action, data))
            return {"levels_completed": 0}

    fake = FakeEnv()
    counted = exp.CountingEnv(fake)
    counted.step("ACTION6", data={"x": 1, "y": 2})
    assert counted.exploration_actions_used == 1
    assert fake.calls[-1] == ("ACTION6", {"x": 1, "y": 2})
    assert counted.reset() == {"levels_completed": 0}

    tn36 = exp.Tn36ClickSchemaEnv(FakeEnv())
    tn36.step("ACTION6", data={"position": {"x": 3, "y": 4}})
    assert tn36.exploration_actions_used == 1
    assert tn36._env.calls[-1] == ("ACTION6", {"x": 3, "y": 4})


def test_scenario_phase4_077_run_writes_artifact_with_injected_sweep(tmp_path: Path) -> None:
    """SCENARIO-PHASE4-077: run() writes the requested JSON when preconditions pass."""

    artifact = exp.run(
        repo=tmp_path,
        write=True,
        sweep_fn=lambda **_: _per_game(bp35=_result("bp35", advanced=True)),
        precondition_fn=lambda _repo: _preconditions(),
        tn36_source_fn=lambda _repo: 'self.action.data["x"], self.action.data["y"]',
    )

    output = tmp_path / exp.RESULT_RELATIVE_PATH
    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8"))["honest_verdict"] == artifact["honest_verdict"]
    assert artifact["games_advanced"] == ["bp35"]
    assert exp.artifact_schema_errors(artifact) == []


def test_req_phase4_077_run_blocks_when_preconditions_fail(tmp_path: Path) -> None:
    """REQ-PHASE4-077: run() does not call the sweep when envs are blocked."""

    def fail_sweep(**_: object) -> dict[str, exp.GameSweepResult]:
        raise AssertionError("sweep must not run")

    artifact = exp.run(
        repo=tmp_path,
        write=False,
        sweep_fn=fail_sweep,
        precondition_fn=lambda _repo: _preconditions(ok=False),
        tn36_source_fn=lambda _repo: "",
    )

    assert artifact["honest_verdict"] == "blocked_arc_env_unreachable"

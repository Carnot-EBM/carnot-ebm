"""Tests for Exp 4925 rotated ARC level-up attempt.

Spec refs: REQ-REPORT-4925, SCENARIO-REPORT-4925,
SCENARIO-REPORT-4925-REPRODUCTION-GATE,
SCENARIO-REPORT-4925-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from carnot import experiment_4925_levelup_attempt as exp4925
from carnot.agentic import arc_game_adapters as adapters


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _registry_text() -> str:
    return """schema_version: 1
updated: '2026-06-28'
reproducible_total_levels: 69
reproducible_total_games: 25
games:
- game: cn04
  reproducibility: reproduced
  levels_reproduced: 3
- game: m0r0
  reproducibility: reproduced
  levels_reproduced: 2
- game: dc22
  reproducibility: reproduced
  levels_reproduced: 2
- game: g50t
  reproducibility: reproduced
  levels_reproduced: 2
- game: s5i5
  reproducibility: reproduced
  levels_reproduced: 2
- game: bp35
  reproducibility: reproduced
  levels_reproduced: 2
- game: ka59
  reproducibility: reproduced
  levels_reproduced: 1
  dead_ends:
  - hidden-state-bound registry row
- game: wa30
  reproducibility: reproduced
  levels_reproduced: 1
  dead_ends:
  - hidden-state-bound registry row
- game: sp80
  reproducibility: reproduced
  levels_reproduced: 2
  solver: GameAdapter _sp80.
  reproduce: Exp4535 prior L2 bank.
  dead_ends:
  - Exp4535 arc_loop_solve --game sp80 --target-level 3 replays to L2 only; the current adapter has no grounded L3 splitter-placement delta.
- game: cd82
  reproducibility: reproduced
  levels_reproduced: 2
  dead_ends:
  - Exp4525 arc_loop_solve --game cd82 --target-level 3 replays to L2 only; the current adapter has no grounded L3 delta.
- game: su15
  reproducibility: reproduced
  levels_reproduced: 2
  dead_ends:
  - Exp4546 unbounded fruit-relative best-first search from L2 stalled before a gate.
"""


def _recommendations() -> dict[str, object]:
    return {
        "sp80": {
            "target_game": "sp80",
            "confident_transfer": True,
            "recommended": [{"game": "lf52", "similarity": 7.0}],
            "selected_generic_operators": [{"operator": "graph_astar_action_cost"}],
        },
        "cd82": {"target_game": "cd82", "recommended": [{"game": "sc25"}]},
        "su15": {"target_game": "su15", "recommended": [{"game": "lp85"}]},
    }


def _loop_result(
    *,
    game: str = "sp80",
    reached_level: int = 2,
    reproduced: bool = True,
) -> dict[str, object]:
    return {
        "game": game,
        "reached_level": reached_level,
        "offline_reproduced": reproduced,
        "reproduced_levels": reached_level,
        "states_expanded": 33,
        "solve_provenance": "development_proxy",
        "reproduction_gate": {
            "game": game,
            "claimed_level": reached_level,
            "reached_level": reached_level,
            "reproduced": reproduced,
            "mode": "offline_reproduction_gate_no_quota",
        },
        "solution_labels": [json.dumps({"action": 4}), json.dumps({"action": 5})],
        "solution": [{"action": 4}, {"action": 5}],
        "mode": "standing_arc_loop_offline_no_quota",
    }


def _preconditions(target_game: str = "sp80") -> dict[str, object]:
    return {
        "AGENTS.md": True,
        "CODEX.md": True,
        "offline_arcade_exits_0": True,
        "target_env_present": True,
        "generator_required": False,
        "rotated_off": ["cn04", "m0r0", "dc22", "g50t", "s5i5"],
        "a2_target_avoided": "su15",
        "a3_self_play_target_avoided": "bp35",
        "hidden_state_targets_avoided": ["ka59", "wa30"],
        "standing_loop_command": f".venv/bin/python scripts/arc_loop_solve.py --game {target_game}",
    }


def test_req_report_4925_spec_declares_sp80_contract() -> None:
    """REQ-REPORT-4925: OpenSpec anchors fields, command, and result path."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4925" in spec
    assert "SCENARIO-REPORT-4925" in spec
    assert "SCENARIO-REPORT-4925-REPRODUCTION-GATE" in spec
    assert "SCENARIO-REPORT-4925-BLOCKED-PRECONDITION" in spec
    assert exp4925.RESULT_RELATIVE_PATH in spec
    assert ".venv/bin/python scripts/arc_loop_solve.py --game <rotated-target>" in spec
    assert "solve_provenance=live_agent_self_discovery" in spec
    assert "live_path_reachable=true" in spec
    assert "new_levels_banked>=1" in spec
    for field in exp4925.REQUIRED_FIELDS:
        assert field in spec


def test_scenario_report_4925_selects_sp80_and_records_dead_end_context() -> None:
    """SCENARIO-REPORT-4925: target rotation chooses sp80 and records consulted dead ends."""

    selection = exp4925.select_rotation_target(
        yaml.safe_load(_registry_text()),
        recommendations=_recommendations(),
    )

    assert selection["game"] == "sp80"
    assert selection["prior_level"] == 2
    assert selection["target_level"] == 3
    assert selection["reason"] == "preferred_sp80_grounded_probe"
    assert selection["has_recorded_next_level_dead_end"] is True
    assert selection["approach_recommendation"] == _recommendations()["sp80"]
    assert selection["excluded_recent_targets"] == ["cn04", "m0r0", "dc22", "g50t", "s5i5"]
    assert selection["a2_target_avoided"] == "su15"
    assert selection["a3_self_play_target_avoided"] == "bp35"
    assert selection["hidden_state_targets_avoided"] == ["ka59", "wa30"]


def test_scenario_report_4925_selection_audits_skip_paths(monkeypatch) -> None:
    """REQ-REPORT-4925: selector records why unsuitable rotated candidates are skipped."""

    registry = yaml.safe_load(_registry_text())
    monkeypatch.setattr(exp4925, "CANDIDATE_GAMES", ("cn04", "su15", "bp35", "ka59", "cd82"))
    selection = exp4925.select_rotation_target(registry)
    status_by_game = {row["game"]: row["status"] for row in selection["candidate_audit"]}

    assert status_by_game == {
        "cn04": "skip_recent_target",
        "su15": "skip_a2_target",
        "bp35": "skip_a3_self_play_target",
        "ka59": "skip_hidden_state_bound",
        "cd82": "selected_recorded_dead_end",
    }

    cd82 = next(row for row in registry["games"] if row["game"] == "cd82")
    cd82["levels_reproduced"] = 1
    monkeypatch.setattr(exp4925, "CANDIDATE_GAMES", ("cd82",))
    no_selection = exp4925.select_rotation_target(registry)

    assert no_selection["game"] == "none"
    assert no_selection["reason"] == "no_rotated_l2_candidate"
    assert no_selection["candidate_audit"][0]["status"] == "skip_not_l2"


def test_req_report_4925_sp80_adapter_is_registered_for_live_path() -> None:
    """SCENARIO-REPORT-4925: sp80 runs through a live GameAdapter, not a parallel solver."""

    adapter = adapters.get_adapter("sp80")

    assert adapter is not None
    assert adapter.game == "sp80"
    assert adapter.branch_mode == "fresh_env"
    assert adapter.depth_caps[3] == 80


def test_req_report_4925_registry_and_residual_helpers_are_defensive(tmp_path: Path) -> None:
    """REQ-REPORT-4925: defensive helper branches fail closed without banking."""

    (tmp_path / "ops").mkdir()
    (tmp_path / exp4925.REGISTRY_RELATIVE_PATH).write_text(_registry_text(), encoding="utf-8")

    assert exp4925.registry_level("sp80", root=tmp_path) == 2
    assert exp4925.registry_total_levels(root=tmp_path) == 69
    assert exp4925.registry_level("missing", root=tmp_path) == 0
    assert exp4925._dead_ends({"dead_ends": "scalar dead end"}) == ["scalar dead end"]
    assert exp4925._live_path_reachable({"status": "needs_per_game_RE"}) is False

    (tmp_path / exp4925.REGISTRY_RELATIVE_PATH).write_text(
        _registry_text().replace("reproducible_total_levels: 69", "reproducible_total_levels: nope"),
        encoding="utf-8",
    )
    assert exp4925.registry_total_levels(root=tmp_path) == 0

    assert (
        exp4925._residual_cause({"status": "needs_per_game_RE", "offline_reproduced": False}, 2)
        == "needs_per_game_re"
    )
    assert exp4925._residual_cause({"offline_reproduced": False}, 2) == "offline_reproduction_failed"
    assert (
        exp4925._residual_cause(
            {
                "offline_reproduced": True,
                "reproduction_gate": {"reproduced": True, "reached_level": 3},
            },
            2,
        )
        == "live_path_unreachable"
    )
    assert exp4925._artifact_residual_reason({"registry_update": {"reason": "fallback"}}) == "fallback"


def test_req_report_4925_success_artifact_counts_only_new_reproduced_depth() -> None:
    """SCENARIO-REPORT-4925-REPRODUCTION-GATE: success requires new offline depth."""

    artifact = exp4925.build_artifact(
        loop_result=_loop_result(reached_level=3),
        prior_level=2,
        prior_total_levels=69,
        preconditions_checked=_preconditions(),
        approach_recommendation=_recommendations()["sp80"],
        registry_update={"updated": True, "banked_levels": 1},
        candidate_selection={"game": "sp80"},
    )

    assert artifact["honest_verdict"] == "success_sp80_levelup_banked"
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["target_game"] == "sp80"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 3
    assert artifact["new_levels_banked"] == 1
    assert artifact["live_path_reachable"] is True
    assert artifact["verifier_is_oracle"] is True
    assert artifact["schema_errors"] == []
    assert exp4925.artifact_schema_errors(artifact) == []

    duplicate = exp4925.build_artifact(
        loop_result=_loop_result(reached_level=2),
        prior_level=2,
        prior_total_levels=69,
        preconditions_checked=_preconditions(),
        approach_recommendation=_recommendations()["sp80"],
        registry_update={"updated": True, "banked_levels": 0, "reason": "duplicate_depth"},
        candidate_selection={"game": "sp80"},
    )
    assert duplicate["honest_verdict"] == "complete_sp80_no_new_level_residual_duplicate_depth"
    assert duplicate["offline_reproduced"] is False
    assert duplicate["reproduced_levels"] == 0
    assert duplicate["new_levels_banked"] == 0
    assert duplicate["live_path_reachable"] is True
    assert duplicate["retire_if_same_verdict"] is True


def test_req_report_4925_blocked_and_schema_errors_are_explicit() -> None:
    """SCENARIO-REPORT-4925-BLOCKED-PRECONDITION: blocked runs never fabricate progress."""

    artifact = exp4925.blocked_artifact(
        target_game="sp80",
        reason="offline_env_missing",
        preconditions_checked={"offline_arcade_exits_0": True, "target_env_present": False},
    )

    assert artifact["honest_verdict"] == "blocked_sp80_offline_env_missing"
    assert artifact["offline_reproduced"] is False
    assert artifact["live_path_reachable"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["new_levels_banked"] == 0
    assert artifact["schema_errors"] == []

    malformed = {"target_game": "cn04", "reproducibility_checksum": "not-a-checksum"}
    errors = exp4925.artifact_schema_errors(malformed)
    assert "missing required field: honest_verdict" in errors
    assert "target_game violates rotation exclusions" in errors
    assert "honest_verdict must use a terminal prefix" in errors

    good = exp4925.build_artifact(
        loop_result=_loop_result(reached_level=3),
        prior_level=2,
        prior_total_levels=69,
        preconditions_checked=_preconditions(),
        approach_recommendation=_recommendations()["sp80"],
        registry_update={"updated": True, "banked_levels": 1},
        candidate_selection={"game": "sp80"},
    )
    corrupted = dict(good)
    corrupted["reproducibility_checksum"] = "0" * 64
    assert "checksum mismatch" in exp4925.artifact_schema_errors(corrupted)

    impossible_success = dict(good)
    impossible_success.update(offline_reproduced=False, reproduced_levels=2, new_levels_banked=0)
    success_errors = exp4925.artifact_schema_errors(impossible_success)
    assert "success requires offline_reproduced true" in success_errors
    assert "success requires new_levels_banked >= 1" in success_errors
    assert "success requires reproduced_levels > prior_reproduced_level" in success_errors

    impossible_live_path = dict(good)
    impossible_live_path.update(live_path_reachable=False)
    assert "success/complete requires live_path_reachable true" in exp4925.artifact_schema_errors(
        impossible_live_path
    )

    impossible_oracle = dict(good)
    impossible_oracle.update(
        honest_verdict="complete_sp80_no_new_level_residual_duplicate_depth",
        offline_reproduced=True,
        new_levels_banked=0,
    )
    assert (
        "offline_reproduced true requires new_levels_banked >= 1"
        in exp4925.artifact_schema_errors(impossible_oracle)
    )


def test_scenario_report_4925_registry_update_records_bank_or_dead_end() -> None:
    """SCENARIO-REPORT-4925-REPRODUCTION-GATE: registry updates only the selected game row."""

    artifact = exp4925.build_artifact(
        loop_result=_loop_result(reached_level=3),
        prior_level=2,
        prior_total_levels=69,
        preconditions_checked=_preconditions(),
        approach_recommendation=_recommendations()["sp80"],
        registry_update={"updated": True, "banked_levels": 1},
        candidate_selection={"game": "sp80"},
    )
    updated_text, update = exp4925.apply_registry_result(_registry_text(), artifact=artifact)
    registry = yaml.safe_load(updated_text)
    sp80 = next(row for row in registry["games"] if row["game"] == "sp80")
    cd82 = next(row for row in registry["games"] if row["game"] == "cd82")

    assert update["updated"] is True
    assert update["prior_game_levels"] == 2
    assert update["new_game_levels"] == 3
    assert update["banked_levels"] == 1
    assert registry["reproducible_total_levels"] == 70
    assert sp80["levels_reproduced"] == 3
    assert sp80["latest_exp4925_levelup_attempt"]["new_levels_banked"] == 1
    assert "Exp4925" in sp80["reproduce"]
    assert cd82["levels_reproduced"] == 2

    no_bank = exp4925.build_artifact(
        loop_result=_loop_result(reached_level=2),
        prior_level=2,
        prior_total_levels=69,
        preconditions_checked=_preconditions(),
        approach_recommendation=_recommendations()["sp80"],
        registry_update={"updated": True, "banked_levels": 0, "reason": "duplicate_depth"},
        candidate_selection={"game": "sp80"},
    )
    dead_end_text, dead_end_update = exp4925.apply_registry_result(_registry_text(), artifact=no_bank)
    dead_end_registry = yaml.safe_load(dead_end_text)
    dead_end_sp80 = next(row for row in dead_end_registry["games"] if row["game"] == "sp80")

    assert dead_end_update["updated"] is True
    assert dead_end_update["banked_levels"] == 0
    assert dead_end_registry["reproducible_total_levels"] == 69
    assert any("Exp4925 sp80 no-bank duplicate_depth" in item for item in dead_end_sp80["dead_ends"])


def test_scenario_report_4925_run_experiment_writes_artifact_and_registry(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4925: runner writes stable JSON and registry evidence."""

    (tmp_path / "ops").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / exp4925.REGISTRY_RELATIVE_PATH).write_text(_registry_text(), encoding="utf-8")

    no_bank = exp4925.run_experiment(
        root=tmp_path,
        target_game="sp80",
        loop_result=_loop_result(reached_level=2),
        approach_recommendation=_recommendations()["sp80"],
        preconditions_checked=_preconditions(),
    )

    written = json.loads((tmp_path / exp4925.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    registry = yaml.safe_load((tmp_path / exp4925.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))
    sp80 = next(row for row in registry["games"] if row["game"] == "sp80")

    assert no_bank == written
    assert written["schema_errors"] == []
    assert written["honest_verdict"] == "complete_sp80_no_new_level_residual_duplicate_depth"
    assert written["standing_loop_result_path"] == "results/arc_loop_solve_sp80.json"
    assert sp80["latest_exp4925_levelup_attempt"]["reproducibility_checksum"] == no_bank[
        "reproducibility_checksum"
    ]

    blocked = exp4925.run_experiment(
        root=tmp_path,
        target_game="sp80",
        preconditions_checked={"offline_arcade_exits_0": False, "target_env_present": True},
    )
    assert blocked["honest_verdict"] == "blocked_sp80_offline_arcade_missing"
    assert blocked["new_levels_banked"] == 0

    rotated_root = tmp_path / "explicit_after_rotation"
    (rotated_root / "ops").mkdir(parents=True)
    (rotated_root / "results").mkdir()
    rotated_registry = _registry_text().replace(
        "  levels_reproduced: 2\n  solver: GameAdapter _sp80.",
        "  levels_reproduced: 1\n  solver: GameAdapter _sp80.",
        1,
    )
    (rotated_root / exp4925.REGISTRY_RELATIVE_PATH).write_text(rotated_registry, encoding="utf-8")
    explicit = exp4925.run_experiment(
        root=rotated_root,
        target_game="sp80",
        loop_result=_loop_result(reached_level=1),
        approach_recommendation=_recommendations()["sp80"],
        preconditions_checked=_preconditions(),
    )

    assert explicit["candidate_selection"]["game"] == "cd82"
    assert explicit["candidate_selection"]["actual_target_game"] == "sp80"

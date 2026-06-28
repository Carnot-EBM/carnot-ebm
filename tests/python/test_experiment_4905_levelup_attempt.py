"""Tests for Exp 4905 rotated ARC level-up attempt.

Spec refs: REQ-REPORT-4905, SCENARIO-REPORT-4905,
SCENARIO-REPORT-4905-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from carnot import experiment_4905_levelup_attempt as exp4905


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _registry_text() -> str:
    return """schema_version: 1
updated: '2026-06-28'
reproducible_total_levels: 68
reproducible_total_games: 25
games:
- game: dc22
  reproducibility: reproduced
  levels_reproduced: 2
- game: g50t
  reproducibility: reproduced
  levels_reproduced: 2
- game: s5i5
  reproducibility: reproduced
  levels_reproduced: 2
- game: r11l
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
- game: m0r0
  reproducibility: reproduced
  levels_reproduced: 2
  solver: GameAdapter _m0r0.
- game: sp80
  reproducibility: reproduced
  levels_reproduced: 2
  dead_ends:
  - Exp4535 arc_loop_solve --game sp80 --target-level 3 replays to L2 only; the current adapter has no grounded L3 splitter-placement delta.
- game: su15
  reproducibility: reproduced
  levels_reproduced: 2
  dead_ends:
  - Exp4546 unbounded fruit-relative best-first search from L2 stalled before a gate.
- game: cn04
  reproducibility: reproduced
  levels_reproduced: 2
  dead_ends:
  - Pre-adapter arc_loop_solve --game cn04 returned needs_per_game_RE.
"""


def _recommendations() -> dict[str, object]:
    return {
        "m0r0": {
            "target_game": "m0r0",
            "confident_transfer": True,
            "recommended": [{"game": "sp80", "similarity": 6.0}],
            "selected_generic_operators": [{"operator": "graph_astar_action_cost"}],
        },
        "cn04": {"target_game": "cn04", "recommended": [{"game": "sp80"}]},
    }


def _loop_result(
    *,
    game: str = "m0r0",
    reached_level: int = 2,
    reproduced: bool = True,
) -> dict[str, object]:
    return {
        "game": game,
        "reached_level": reached_level,
        "offline_reproduced": reproduced,
        "reproduced_levels": reached_level,
        "states_expanded": 1802,
        "solve_provenance": "development_proxy",
        "reproduction_gate": {
            "game": game,
            "claimed_level": reached_level,
            "reached_level": reached_level,
            "reproduced": reproduced,
            "mode": "offline_reproduction_gate_no_quota",
        },
        "solution_labels": [json.dumps({"action": 4}), json.dumps({"action": 1})],
        "solution": [{"action": 4}, {"action": 1}],
        "mode": "standing_arc_loop_offline_no_quota",
    }


def _preconditions(target_game: str = "m0r0") -> dict[str, object]:
    return {
        "AGENTS.md": True,
        "CODEX.md": True,
        "offline_arcade_exits_0": True,
        "target_env_present": True,
        "generator_required": False,
        "rotated_off": ["dc22", "g50t", "s5i5", "r11l"],
        "hidden_state_targets_avoided": ["ka59", "wa30"],
        "standing_loop_command": f".venv/bin/python scripts/arc_loop_solve.py --game {target_game}",
    }


def test_req_report_4905_spec_declares_rotated_contract() -> None:
    """REQ-REPORT-4905: OpenSpec anchors fields, command, and result path."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4905" in spec
    assert "SCENARIO-REPORT-4905" in spec
    assert "SCENARIO-REPORT-4905-BLOCKED-PRECONDITION" in spec
    assert exp4905.RESULT_RELATIVE_PATH in spec
    assert ".venv/bin/python scripts/arc_loop_solve.py --game <rotated-target>" in spec
    assert "solve_provenance=live_agent_self_discovery" in spec
    assert "new_levels_banked>=1" in spec
    for field in exp4905.REQUIRED_FIELDS:
        assert field in spec


def test_scenario_report_4905_selects_m0r0_and_skips_recent_or_dead_targets() -> None:
    """SCENARIO-REPORT-4905: target rotation avoids recent targets and dead-end rows."""

    selection = exp4905.select_rotation_target(
        yaml.safe_load(_registry_text()),
        recommendations=_recommendations(),
    )

    assert selection["game"] == "m0r0"
    assert selection["prior_level"] == 2
    assert selection["target_level"] == 3
    assert selection["reason"] == "grounded_rotated_l2_to_l3_candidate"
    assert selection["excluded_recent_targets"] == ["dc22", "g50t", "s5i5", "r11l"]
    assert selection["hidden_state_targets_avoided"] == ["ka59", "wa30"]
    assert selection["approach_recommendation"] == _recommendations()["m0r0"]
    status_by_game = {row["game"]: row["status"] for row in selection["candidate_audit"]}
    assert status_by_game["m0r0"] == "selected"
    assert status_by_game["sp80"] == "skip_recorded_dead_end"
    assert status_by_game["su15"] == "skip_recorded_dead_end"


def test_scenario_report_4905_selection_audits_defensive_skip_paths(monkeypatch) -> None:
    """REQ-REPORT-4905: selector records unsuitable rotated-candidate reasons."""

    registry = yaml.safe_load(_registry_text())
    monkeypatch.setattr(exp4905, "CANDIDATE_GAMES", ("dc22", "ka59", "m0r0"))
    selection = exp4905.select_rotation_target(registry)
    status_by_game = {row["game"]: row["status"] for row in selection["candidate_audit"]}

    assert status_by_game == {
        "dc22": "skip_recent_target",
        "ka59": "skip_hidden_state_bound",
        "m0r0": "selected",
    }

    m0r0 = next(row for row in registry["games"] if row["game"] == "m0r0")
    m0r0["levels_reproduced"] = 1
    m0r0["dead_ends"] = "scalar dead-end record"
    monkeypatch.setattr(exp4905, "CANDIDATE_GAMES", ("m0r0",))
    no_selection = exp4905.select_rotation_target(registry)

    assert no_selection["game"] == "none"
    assert no_selection["reason"] == "no_grounded_rotated_l2_to_l3_candidate"
    assert no_selection["candidate_audit"][0]["status"] == "skip_not_l2"
    assert no_selection["candidate_audit"][0]["dead_ends_consulted"] == ["scalar dead-end record"]


def test_req_report_4905_registry_helpers_are_defensive(tmp_path: Path) -> None:
    """REQ-REPORT-4905: registry readers default safely when rows or totals drift."""

    (tmp_path / "ops").mkdir()
    (tmp_path / exp4905.REGISTRY_RELATIVE_PATH).write_text(_registry_text(), encoding="utf-8")

    assert exp4905.registry_level("m0r0", root=tmp_path) == 2
    assert exp4905.registry_total_levels(root=tmp_path) == 68
    assert exp4905.registry_level("missing", root=tmp_path) == 0

    (tmp_path / exp4905.REGISTRY_RELATIVE_PATH).write_text(
        _registry_text().replace("reproducible_total_levels: 68", "reproducible_total_levels: nope"),
        encoding="utf-8",
    )
    assert exp4905.registry_total_levels(root=tmp_path) == 0


def test_req_report_4905_success_artifact_counts_new_depth_not_duplicate() -> None:
    """REQ-REPORT-4905: success requires a new offline-reproduced depth."""

    artifact = exp4905.build_artifact(
        loop_result=_loop_result(reached_level=3),
        prior_level=2,
        prior_total_levels=68,
        preconditions_checked=_preconditions(),
        approach_recommendation=_recommendations()["m0r0"],
        registry_update={"updated": True, "banked_levels": 1},
        candidate_selection={"game": "m0r0"},
    )

    assert artifact["honest_verdict"] == "success_m0r0_levelup_banked"
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["target_game"] == "m0r0"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 3
    assert artifact["new_levels_banked"] == 1
    assert artifact["verifier_is_oracle"] is True
    assert artifact["schema_errors"] == []
    assert exp4905.artifact_schema_errors(artifact) == []

    duplicate = exp4905.build_artifact(
        loop_result=_loop_result(reached_level=2),
        prior_level=2,
        prior_total_levels=68,
        preconditions_checked=_preconditions(),
        approach_recommendation=_recommendations()["m0r0"],
        registry_update={"updated": True, "banked_levels": 0, "reason": "duplicate_depth"},
        candidate_selection={"game": "m0r0"},
    )
    assert duplicate["honest_verdict"] == "complete_m0r0_no_new_level_residual_duplicate_depth"
    assert duplicate["offline_reproduced"] is False
    assert duplicate["new_levels_banked"] == 0
    assert duplicate["retire_if_same_verdict"] is True


def test_req_report_4905_residual_and_blocked_branches_are_explicit() -> None:
    """SCENARIO-REPORT-4905-BLOCKED-PRECONDITION: no-bank causes never fabricate progress."""

    assert exp4905._residual_cause(_loop_result(reached_level=2), 2) == "duplicate_depth"
    assert (
        exp4905._residual_cause({"status": "needs_per_game_RE", "offline_reproduced": False}, 2)
        == "needs_per_game_re"
    )
    assert exp4905._residual_cause({"offline_reproduced": False}, 2) == "offline_reproduction_failed"

    artifact = exp4905.blocked_artifact(
        target_game="m0r0",
        reason="offline_env_missing",
        preconditions_checked={"offline_arcade_exits_0": True, "target_env_present": False},
    )

    assert artifact["honest_verdict"] == "blocked_m0r0_offline_env_missing"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["new_levels_banked"] == 0
    assert artifact["schema_errors"] == []


def test_req_report_4905_schema_errors_are_explicit() -> None:
    """REQ-REPORT-4905: validation distinguishes malformed and impossible artifacts."""

    good = exp4905.build_artifact(
        loop_result=_loop_result(reached_level=3),
        prior_level=2,
        prior_total_levels=68,
        preconditions_checked=_preconditions(),
        approach_recommendation=_recommendations()["m0r0"],
        registry_update={"updated": True, "banked_levels": 1},
        candidate_selection={"game": "m0r0"},
    )
    malformed = {"target_game": "dc22", "reproducibility_checksum": "not-a-checksum"}
    errors = exp4905.artifact_schema_errors(malformed)
    assert "missing required field: honest_verdict" in errors
    assert "experiment mismatch" in errors
    assert "target_game violates rotation exclusions" in errors
    assert "honest_verdict must use a terminal prefix" in errors

    corrupted = dict(good)
    corrupted["reproducibility_checksum"] = "0" * 64
    assert "checksum mismatch" in exp4905.artifact_schema_errors(corrupted)

    impossible_success = dict(good)
    impossible_success.update(offline_reproduced=False, reproduced_levels=2, new_levels_banked=0)
    success_errors = exp4905.artifact_schema_errors(impossible_success)
    assert "success requires offline_reproduced true" in success_errors
    assert "success requires new_levels_banked >= 1" in success_errors
    assert "success requires reproduced_levels > prior_reproduced_level" in success_errors

    impossible_oracle = dict(good)
    impossible_oracle.update(
        honest_verdict="complete_m0r0_no_new_level_residual_duplicate_depth",
        offline_reproduced=True,
        new_levels_banked=0,
    )
    assert (
        "offline_reproduced true requires new_levels_banked >= 1"
        in exp4905.artifact_schema_errors(impossible_oracle)
    )


def test_scenario_report_4905_registry_update_records_bank_or_dead_end() -> None:
    """SCENARIO-REPORT-4905: registry updates only the selected game row."""

    artifact = exp4905.build_artifact(
        loop_result=_loop_result(reached_level=3),
        prior_level=2,
        prior_total_levels=68,
        preconditions_checked=_preconditions(),
        approach_recommendation=_recommendations()["m0r0"],
        registry_update={"updated": True, "banked_levels": 1},
        candidate_selection={"game": "m0r0"},
    )
    updated_text, update = exp4905.apply_registry_result(_registry_text(), artifact=artifact)
    registry = yaml.safe_load(updated_text)
    m0r0 = next(row for row in registry["games"] if row["game"] == "m0r0")
    g50t = next(row for row in registry["games"] if row["game"] == "g50t")

    assert update["updated"] is True
    assert update["prior_game_levels"] == 2
    assert update["new_game_levels"] == 3
    assert update["banked_levels"] == 1
    assert registry["reproducible_total_levels"] == 69
    assert m0r0["levels_reproduced"] == 3
    assert m0r0["latest_exp4905_levelup_attempt"]["new_levels_banked"] == 1
    assert "Exp4905" in m0r0["reproduce"]
    assert g50t["levels_reproduced"] == 2

    no_bank = exp4905.build_artifact(
        loop_result=_loop_result(reached_level=2),
        prior_level=2,
        prior_total_levels=68,
        preconditions_checked=_preconditions(),
        approach_recommendation=_recommendations()["m0r0"],
        registry_update={"updated": True, "banked_levels": 0, "reason": "duplicate_depth"},
        candidate_selection={"game": "m0r0"},
    )
    dead_end_text, dead_end_update = exp4905.apply_registry_result(_registry_text(), artifact=no_bank)
    dead_end_registry = yaml.safe_load(dead_end_text)
    dead_end_m0r0 = next(row for row in dead_end_registry["games"] if row["game"] == "m0r0")

    assert dead_end_update["updated"] is True
    assert dead_end_update["banked_levels"] == 0
    assert dead_end_registry["reproducible_total_levels"] == 68
    assert any("Exp4905 m0r0 no-bank duplicate_depth" in item for item in dead_end_m0r0["dead_ends"])
    assert exp4905._artifact_residual_reason({"registry_update": {"reason": "fallback"}}) == "fallback"


def test_scenario_report_4905_run_experiment_writes_artifact_and_registry(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4905: runner writes stable JSON and registry evidence."""

    (tmp_path / "ops").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / exp4905.REGISTRY_RELATIVE_PATH).write_text(_registry_text(), encoding="utf-8")

    no_bank = exp4905.run_experiment(
        root=tmp_path,
        target_game="m0r0",
        loop_result=_loop_result(reached_level=2),
        approach_recommendation=_recommendations()["m0r0"],
        preconditions_checked=_preconditions(),
    )

    written = json.loads((tmp_path / exp4905.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    registry = yaml.safe_load((tmp_path / exp4905.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))
    m0r0 = next(row for row in registry["games"] if row["game"] == "m0r0")

    assert no_bank == written
    assert written["schema_errors"] == []
    assert written["honest_verdict"] == "complete_m0r0_no_new_level_residual_duplicate_depth"
    assert written["standing_loop_result_path"] == "results/arc_loop_solve_m0r0.json"
    assert m0r0["latest_exp4905_levelup_attempt"]["reproducibility_checksum"] == no_bank[
        "reproducibility_checksum"
    ]

    blocked = exp4905.run_experiment(
        root=tmp_path,
        target_game="m0r0",
        preconditions_checked={"offline_arcade_exits_0": False, "target_env_present": True},
    )
    assert blocked["honest_verdict"] == "blocked_m0r0_offline_arcade_missing"
    assert blocked["new_levels_banked"] == 0

    rotated_root = tmp_path / "explicit_after_rotation"
    (rotated_root / "ops").mkdir(parents=True)
    (rotated_root / "results").mkdir()
    rotated_registry = _registry_text().replace(
        "solver: GameAdapter _m0r0.",
        "solver: GameAdapter _m0r0.\n  dead_ends:\n  - Exp4905 m0r0 no-bank duplicate_depth.",
    )
    (rotated_root / exp4905.REGISTRY_RELATIVE_PATH).write_text(rotated_registry, encoding="utf-8")
    explicit = exp4905.run_experiment(
        root=rotated_root,
        target_game="m0r0",
        loop_result=_loop_result(reached_level=2),
        approach_recommendation=_recommendations()["m0r0"],
        preconditions_checked=_preconditions(),
    )

    assert explicit["candidate_selection"]["game"] != "m0r0"
    assert explicit["candidate_selection"]["actual_target_game"] == "m0r0"

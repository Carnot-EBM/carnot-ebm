"""Tests for Exp 4894 rotated ARC level-up attempt.

Spec refs: REQ-REPORT-4894, SCENARIO-REPORT-4894,
SCENARIO-REPORT-4894-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from carnot import experiment_4894_levelup_attempt as exp4894


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _registry_text() -> str:
    return """schema_version: 1
updated: '2026-06-27'
reproducible_total_levels: 67
reproducible_total_games: 25
games:
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
  levels_reproduced: 2
  dead_ends:
  - hidden-state-bound registry row
- game: wa30
  reproducibility: reproduced
  levels_reproduced: 1
  dead_ends:
  - hidden-state-bound registry row
- game: dc22
  reproducibility: reproduced
  levels_reproduced: 2
  mechanic_class: config_toggle_navigation
  win_condition: old L1+L2 predicate.
  action_model: old action model.
  solver: old solver.
  reproduce: old reproduce.
  dead_ends:
  - bp35, re86, and sb26 remain preferred hard targets but have no grounded next-level adapter.
- game: sp80
  reproducibility: reproduced
  levels_reproduced: 2
  dead_ends:
  - current adapter has no grounded L3 splitter-placement delta.
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
        "dc22": {
            "target_game": "dc22",
            "recommended": [{"game": "sc25", "similarity": 6.0}],
            "selected_generic_operators": [{"operator": "graph_astar_action_cost"}],
        },
        "sp80": {"target_game": "sp80", "recommended": [{"game": "cn04"}]},
    }


def _loop_result(game: str = "dc22", reached_level: int = 3, reproduced: bool = True) -> dict[str, object]:
    return {
        "game": game,
        "reached_level": reached_level,
        "offline_reproduced": reproduced,
        "reproduced_levels": reached_level,
        "states_expanded": 321,
        "solve_provenance": "development_proxy",
        "reproduction_gate": {
            "game": game,
            "claimed_level": reached_level,
            "reached_level": reached_level,
            "reproduced": reproduced,
            "mode": "offline_reproduction_gate_no_quota",
        },
        "solution_labels": [json.dumps({"action": 4}), json.dumps({"action": 6, "x": 9, "y": 10})],
        "solution": [{"action": 4}, {"action": 6, "x": 9, "y": 10}],
        "mode": "standing_arc_loop_offline_no_quota",
    }


def _preconditions() -> dict[str, object]:
    return {
        "AGENTS.md": True,
        "CODEX.md": True,
        "offline_arcade_exits_0": True,
        "target_env_present": True,
        "generator_required": False,
        "rotated_off": ["g50t", "s5i5", "r11l"],
        "hidden_state_targets_avoided": ["ka59", "wa30"],
        "standing_loop_command": ".venv/bin/python scripts/arc_loop_solve.py --game dc22",
    }


def test_req_report_4894_spec_declares_rotated_contract() -> None:
    """REQ-REPORT-4894: OpenSpec anchors fields, command, and result path."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4894" in spec
    assert "SCENARIO-REPORT-4894" in spec
    assert "SCENARIO-REPORT-4894-BLOCKED-PRECONDITION" in spec
    assert exp4894.RESULT_RELATIVE_PATH in spec
    assert ".venv/bin/python scripts/arc_loop_solve.py --game <rotated-target>" in spec
    assert "solve_provenance=live_agent_self_discovery" in spec
    assert "new_levels_banked>=1" in spec
    for field in exp4894.REQUIRED_FIELDS:
        assert field in spec


def test_scenario_report_4894_selects_dc22_and_skips_recent_or_dead_targets() -> None:
    """SCENARIO-REPORT-4894: target rotation avoids recent and recorded dead-end rows."""

    selection = exp4894.select_rotation_target(
        yaml.safe_load(_registry_text()),
        recommendations=_recommendations(),
    )

    assert selection["game"] == "dc22"
    assert selection["prior_level"] == 2
    assert selection["target_level"] == 3
    assert selection["reason"] == "grounded_rotated_l2_to_l3_candidate"
    assert selection["excluded_recent_targets"] == ["g50t", "s5i5", "r11l"]
    assert selection["hidden_state_targets_avoided"] == ["ka59", "wa30"]
    assert selection["approach_recommendation"] == _recommendations()["dc22"]
    status_by_game = {row["game"]: row["status"] for row in selection["candidate_audit"]}
    assert status_by_game["dc22"] == "selected"
    assert status_by_game["sp80"] == "skip_recorded_dead_end"
    assert "ka59" not in status_by_game


def test_scenario_report_4894_selection_audits_defensive_skip_paths(monkeypatch) -> None:
    """REQ-REPORT-4894: selector records why unsuitable rotated candidates are skipped."""

    registry = yaml.safe_load(_registry_text())
    monkeypatch.setattr(exp4894, "CANDIDATE_GAMES", ("g50t", "ka59", "dc22"))
    selection = exp4894.select_rotation_target(registry)
    status_by_game = {row["game"]: row["status"] for row in selection["candidate_audit"]}

    assert status_by_game == {
        "g50t": "skip_recent_target",
        "ka59": "skip_hidden_state_bound",
        "dc22": "selected",
    }

    dc22 = next(row for row in registry["games"] if row["game"] == "dc22")
    dc22["levels_reproduced"] = 1
    dc22["dead_ends"] = "scalar dead-end record"
    monkeypatch.setattr(exp4894, "CANDIDATE_GAMES", ("dc22",))
    no_selection = exp4894.select_rotation_target(registry)

    assert no_selection["game"] == "none"
    assert no_selection["reason"] == "no_grounded_rotated_l2_to_l3_candidate"
    assert no_selection["candidate_audit"][0]["status"] == "skip_not_l2"
    assert no_selection["candidate_audit"][0]["dead_ends_consulted"] == ["scalar dead-end record"]


def test_req_report_4894_registry_helpers_are_defensive(tmp_path: Path) -> None:
    """REQ-REPORT-4894: registry readers default safely when rows or totals drift."""

    (tmp_path / "ops").mkdir()
    (tmp_path / exp4894.REGISTRY_RELATIVE_PATH).write_text(_registry_text(), encoding="utf-8")

    assert exp4894.registry_level("dc22", root=tmp_path) == 2
    assert exp4894.registry_total_levels(root=tmp_path) == 67
    assert exp4894.registry_level("missing", root=tmp_path) == 0

    (tmp_path / exp4894.REGISTRY_RELATIVE_PATH).write_text(
        _registry_text().replace("reproducible_total_levels: 67", "reproducible_total_levels: nope"),
        encoding="utf-8",
    )
    assert exp4894.registry_total_levels(root=tmp_path) == 0


def test_req_report_4894_success_artifact_counts_new_depth_not_duplicate() -> None:
    """REQ-REPORT-4894: success requires new offline-reproduced depth."""

    artifact = exp4894.build_artifact(
        loop_result=_loop_result(),
        prior_level=2,
        prior_total_levels=67,
        preconditions_checked=_preconditions(),
        approach_recommendation=_recommendations()["dc22"],
        registry_update={"updated": True, "banked_levels": 1},
    )

    assert artifact["honest_verdict"] == "success_dc22_levelup_banked"
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["target_game"] == "dc22"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 3
    assert artifact["new_levels_banked"] == 1
    assert artifact["verifier_is_oracle"] is True
    assert artifact["schema_errors"] == []
    assert exp4894.artifact_schema_errors(artifact) == []

    duplicate = exp4894.build_artifact(
        loop_result=_loop_result(reached_level=2),
        prior_level=2,
        prior_total_levels=67,
        preconditions_checked=_preconditions(),
        approach_recommendation=_recommendations()["dc22"],
        registry_update={"updated": False, "banked_levels": 0, "reason": "duplicate_depth"},
    )
    assert duplicate["honest_verdict"] == "complete_dc22_no_new_level_residual_duplicate_depth"
    assert duplicate["offline_reproduced"] is False
    assert duplicate["new_levels_banked"] == 0


def test_req_report_4894_residual_causes_are_classified() -> None:
    """REQ-REPORT-4894: no-bank verdicts expose the concrete residual cause."""

    assert exp4894._residual_cause(_loop_result(reached_level=2), 2) == "duplicate_depth"
    assert (
        exp4894._residual_cause({"status": "needs_per_game_RE", "offline_reproduced": False}, 2)
        == "needs_per_game_re"
    )
    assert exp4894._residual_cause({"offline_reproduced": False}, 2) == "offline_reproduction_failed"


def test_req_report_4894_blocked_artifact_never_fabricates_bank() -> None:
    """SCENARIO-REPORT-4894-BLOCKED-PRECONDITION: missing resources block cleanly."""

    artifact = exp4894.blocked_artifact(
        target_game="dc22",
        reason="offline_env_missing",
        preconditions_checked={"offline_arcade_exits_0": True, "target_env_present": False},
    )

    assert artifact["honest_verdict"] == "blocked_dc22_offline_env_missing"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["new_levels_banked"] == 0
    assert artifact["schema_errors"] == []


def test_req_report_4894_schema_errors_are_explicit() -> None:
    """REQ-REPORT-4894: validation distinguishes malformed and impossible artifacts."""

    good = exp4894.build_artifact(
        loop_result=_loop_result(),
        prior_level=2,
        prior_total_levels=67,
        preconditions_checked=_preconditions(),
        approach_recommendation=_recommendations()["dc22"],
        registry_update={"updated": True, "banked_levels": 1},
    )
    malformed = {"target_game": "g50t", "reproducibility_checksum": "not-a-checksum"}
    errors = exp4894.artifact_schema_errors(malformed)
    assert "missing required field: honest_verdict" in errors
    assert "experiment mismatch" in errors
    assert "target_game violates rotation exclusions" in errors
    assert "honest_verdict must use a terminal prefix" in errors

    corrupted = dict(good)
    corrupted["reproducibility_checksum"] = "0" * 64
    assert "checksum mismatch" in exp4894.artifact_schema_errors(corrupted)

    impossible_success = dict(good)
    impossible_success.update(offline_reproduced=False, reproduced_levels=2, new_levels_banked=0)
    success_errors = exp4894.artifact_schema_errors(impossible_success)
    assert "success requires offline_reproduced true" in success_errors
    assert "success requires new_levels_banked >= 1" in success_errors
    assert "success requires reproduced_levels > prior_reproduced_level" in success_errors

    impossible_oracle = dict(good)
    impossible_oracle.update(
        honest_verdict="complete_dc22_no_new_level_residual_duplicate_depth",
        offline_reproduced=True,
        new_levels_banked=0,
    )
    assert (
        "offline_reproduced true requires new_levels_banked >= 1"
        in exp4894.artifact_schema_errors(impossible_oracle)
    )


def test_scenario_report_4894_registry_update_records_bank_or_dead_end() -> None:
    """SCENARIO-REPORT-4894: registry updates only the selected game row."""

    artifact = exp4894.build_artifact(
        loop_result=_loop_result(),
        prior_level=2,
        prior_total_levels=67,
        preconditions_checked=_preconditions(),
        approach_recommendation=_recommendations()["dc22"],
        registry_update={"updated": True, "banked_levels": 1},
    )
    updated_text, update = exp4894.apply_registry_result(_registry_text(), artifact=artifact)
    registry = yaml.safe_load(updated_text)
    dc22 = next(row for row in registry["games"] if row["game"] == "dc22")
    g50t = next(row for row in registry["games"] if row["game"] == "g50t")

    assert update["updated"] is True
    assert update["prior_game_levels"] == 2
    assert update["new_game_levels"] == 3
    assert update["banked_levels"] == 1
    assert registry["reproducible_total_levels"] == 68
    assert dc22["levels_reproduced"] == 3
    assert dc22["latest_exp4894_levelup_attempt"]["new_levels_banked"] == 1
    assert "Exp4894" in dc22["reproduce"]
    assert g50t["levels_reproduced"] == 2

    no_bank = exp4894.build_artifact(
        loop_result=_loop_result(reached_level=2),
        prior_level=2,
        prior_total_levels=67,
        preconditions_checked=_preconditions(),
        approach_recommendation=_recommendations()["dc22"],
        registry_update={"updated": False, "banked_levels": 0, "reason": "duplicate_depth"},
    )
    dead_end_text, dead_end_update = exp4894.apply_registry_result(_registry_text(), artifact=no_bank)
    dead_end_registry = yaml.safe_load(dead_end_text)
    dead_end_dc22 = next(row for row in dead_end_registry["games"] if row["game"] == "dc22")

    assert dead_end_update["updated"] is True
    assert dead_end_update["banked_levels"] == 0
    assert dead_end_registry["reproducible_total_levels"] == 67
    assert any("Exp4894 dc22 no-bank duplicate_depth" in item for item in dead_end_dc22["dead_ends"])
    assert exp4894._artifact_residual_reason({"registry_update": {"reason": "fallback"}}) == "fallback"


def test_scenario_report_4894_run_experiment_writes_artifact_and_registry(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4894: runner writes stable JSON and registry evidence."""

    (tmp_path / "ops").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / exp4894.REGISTRY_RELATIVE_PATH).write_text(_registry_text(), encoding="utf-8")

    artifact = exp4894.run_experiment(
        root=tmp_path,
        target_game="dc22",
        loop_result=_loop_result(),
        approach_recommendation=_recommendations()["dc22"],
        preconditions_checked=_preconditions(),
    )

    written = json.loads((tmp_path / exp4894.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    registry = yaml.safe_load((tmp_path / exp4894.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))
    dc22 = next(row for row in registry["games"] if row["game"] == "dc22")

    assert artifact == written
    assert written["schema_errors"] == []
    assert written["honest_verdict"] == "success_dc22_levelup_banked"
    assert written["standing_loop_result_path"] == "results/arc_loop_solve_dc22.json"
    assert dc22["levels_reproduced"] == 3

    no_bank_root = tmp_path / "no_bank"
    (no_bank_root / "ops").mkdir(parents=True)
    (no_bank_root / "results").mkdir()
    (no_bank_root / exp4894.REGISTRY_RELATIVE_PATH).write_text(_registry_text(), encoding="utf-8")
    no_bank = exp4894.run_experiment(
        root=no_bank_root,
        target_game="dc22",
        loop_result=_loop_result(reached_level=2),
        approach_recommendation=_recommendations()["dc22"],
        preconditions_checked=_preconditions(),
    )
    no_bank_registry = yaml.safe_load(
        (no_bank_root / exp4894.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8")
    )
    no_bank_dc22 = next(row for row in no_bank_registry["games"] if row["game"] == "dc22")

    assert no_bank["schema_errors"] == []
    assert no_bank["honest_verdict"] == "complete_dc22_no_new_level_residual_duplicate_depth"
    assert no_bank_dc22["latest_exp4894_levelup_attempt"]["reproducibility_checksum"] == no_bank[
        "reproducibility_checksum"
    ]

    blocked = exp4894.run_experiment(
        root=tmp_path,
        target_game="dc22",
        preconditions_checked={"offline_arcade_exits_0": False, "target_env_present": True},
    )
    assert blocked["honest_verdict"] == "blocked_dc22_offline_arcade_missing"
    assert blocked["new_levels_banked"] == 0

"""Tests for Exp 4915 rotated ARC level-up attempt.

Spec refs: REQ-REPORT-4915, SCENARIO-REPORT-4915,
SCENARIO-REPORT-4915-REPRODUCTION-GATE,
SCENARIO-REPORT-4915-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from carnot import experiment_4915_levelup_attempt as exp4915
from carnot.agentic import arc_game_adapters as adapters


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _registry_text() -> str:
    return """schema_version: 1
updated: '2026-06-28'
reproducible_total_levels: 68
reproducible_total_games: 25
games:
- game: m0r0
  reproducibility: reproduced
  levels_reproduced: 2
  dead_ends:
  - Exp4905 m0r0 no-bank duplicate_depth.
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
- game: vc33
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
  solver: GameAdapter _cn04.
  dead_ends:
  - Pre-adapter arc_loop_solve --game cn04 returned needs_per_game_RE.
"""


def _recommendations() -> dict[str, object]:
    return {
        "sp80": {"target_game": "sp80", "recommended": [{"game": "cn04"}]},
        "su15": {"target_game": "su15", "recommended": [{"game": "sp80"}]},
        "cn04": {
            "target_game": "cn04",
            "confident_transfer": True,
            "recommended": [{"game": "sp80", "similarity": 6.0}],
            "selected_generic_operators": [{"operator": "graph_astar_action_cost"}],
        },
    }


def _loop_result(
    *,
    game: str = "cn04",
    reached_level: int = 2,
    reproduced: bool = True,
) -> dict[str, object]:
    return {
        "game": game,
        "reached_level": reached_level,
        "offline_reproduced": reproduced,
        "reproduced_levels": reached_level,
        "states_expanded": 43,
        "solve_provenance": "development_proxy",
        "reproduction_gate": {
            "game": game,
            "claimed_level": reached_level,
            "reached_level": reached_level,
            "reproduced": reproduced,
            "mode": "offline_reproduction_gate_no_quota",
        },
        "solution_labels": [json.dumps({"action": 6, "data": {"x": 10, "y": 12}})],
        "solution": [{"action": 6, "data": {"x": 10, "y": 12}}],
        "mode": "standing_arc_loop_offline_no_quota",
    }


def _preconditions(target_game: str = "cn04") -> dict[str, object]:
    return {
        "AGENTS.md": True,
        "CODEX.md": True,
        "offline_arcade_exits_0": True,
        "target_env_present": True,
        "generator_required": False,
        "rotated_off": ["m0r0", "dc22", "g50t", "s5i5", "r11l"],
        "a3_self_play_target_avoided": "vc33",
        "hidden_state_targets_avoided": ["ka59", "wa30"],
        "standing_loop_command": f".venv/bin/python scripts/arc_loop_solve.py --game {target_game}",
    }


def test_req_report_4915_spec_declares_rotated_contract() -> None:
    """REQ-REPORT-4915: OpenSpec anchors fields, command, and result path."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4915" in spec
    assert "SCENARIO-REPORT-4915" in spec
    assert "SCENARIO-REPORT-4915-REPRODUCTION-GATE" in spec
    assert "SCENARIO-REPORT-4915-BLOCKED-PRECONDITION" in spec
    assert exp4915.RESULT_RELATIVE_PATH in spec
    assert ".venv/bin/python scripts/arc_loop_solve.py --game <rotated-target>" in spec
    assert "solve_provenance=live_agent_self_discovery" in spec
    assert "new_levels_banked>=1" in spec
    for field in exp4915.REQUIRED_FIELDS:
        assert field in spec


def test_scenario_report_4915_selects_cn04_and_skips_recent_a3_or_dead_targets() -> None:
    """SCENARIO-REPORT-4915: target rotation avoids recent, A3, hidden, and dead-end rows."""

    selection = exp4915.select_rotation_target(
        yaml.safe_load(_registry_text()),
        recommendations=_recommendations(),
    )

    assert selection["game"] == "cn04"
    assert selection["prior_level"] == 2
    assert selection["target_level"] == 3
    assert selection["reason"] == "grounded_rotated_l2_to_l3_candidate"
    assert selection["excluded_recent_targets"] == ["m0r0", "dc22", "g50t", "s5i5", "r11l"]
    assert selection["a3_self_play_target_avoided"] == "vc33"
    assert selection["hidden_state_targets_avoided"] == ["ka59", "wa30"]
    assert selection["approach_recommendation"] == _recommendations()["cn04"]
    status_by_game = {row["game"]: row["status"] for row in selection["candidate_audit"]}
    assert status_by_game["sp80"] == "skip_recorded_dead_end"
    assert status_by_game["su15"] == "skip_recorded_dead_end"
    assert status_by_game["cn04"] == "selected"


def test_scenario_report_4915_selection_audits_defensive_skip_paths(monkeypatch) -> None:
    """REQ-REPORT-4915: selector records why unsuitable rotated candidates are skipped."""

    registry = yaml.safe_load(_registry_text())
    monkeypatch.setattr(exp4915, "CANDIDATE_GAMES", ("m0r0", "vc33", "ka59", "cn04"))
    selection = exp4915.select_rotation_target(registry)
    status_by_game = {row["game"]: row["status"] for row in selection["candidate_audit"]}

    assert status_by_game == {
        "m0r0": "skip_recent_target",
        "vc33": "skip_a3_self_play_target",
        "ka59": "skip_hidden_state_bound",
        "cn04": "selected",
    }

    cn04 = next(row for row in registry["games"] if row["game"] == "cn04")
    cn04["levels_reproduced"] = 1
    cn04["dead_ends"] = "scalar dead-end record"
    monkeypatch.setattr(exp4915, "CANDIDATE_GAMES", ("cn04",))
    no_selection = exp4915.select_rotation_target(registry)

    assert no_selection["game"] == "none"
    assert no_selection["reason"] == "no_grounded_rotated_l2_to_l3_candidate"
    assert no_selection["candidate_audit"][0]["status"] == "skip_not_l2"
    assert no_selection["candidate_audit"][0]["dead_ends_consulted"] == ["scalar dead-end record"]


def test_req_report_4915_registry_helpers_are_defensive(tmp_path: Path) -> None:
    """REQ-REPORT-4915: registry readers default safely when rows or totals drift."""

    (tmp_path / "ops").mkdir()
    (tmp_path / exp4915.REGISTRY_RELATIVE_PATH).write_text(_registry_text(), encoding="utf-8")

    assert exp4915.registry_level("cn04", root=tmp_path) == 2
    assert exp4915.registry_total_levels(root=tmp_path) == 68
    assert exp4915.registry_level("missing", root=tmp_path) == 0

    (tmp_path / exp4915.REGISTRY_RELATIVE_PATH).write_text(
        _registry_text().replace("reproducible_total_levels: 68", "reproducible_total_levels: nope"),
        encoding="utf-8",
    )
    assert exp4915.registry_total_levels(root=tmp_path) == 0


def test_req_report_4915_success_artifact_counts_new_depth_not_duplicate() -> None:
    """SCENARIO-REPORT-4915-REPRODUCTION-GATE: success requires new offline-reproduced depth."""

    artifact = exp4915.build_artifact(
        loop_result=_loop_result(reached_level=3),
        prior_level=2,
        prior_total_levels=68,
        preconditions_checked=_preconditions(),
        approach_recommendation=_recommendations()["cn04"],
        registry_update={"updated": True, "banked_levels": 1},
        candidate_selection={"game": "cn04"},
    )

    assert artifact["honest_verdict"] == "success_cn04_levelup_banked"
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["target_game"] == "cn04"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 3
    assert artifact["new_levels_banked"] == 1
    assert artifact["verifier_is_oracle"] is True
    assert artifact["schema_errors"] == []
    assert exp4915.artifact_schema_errors(artifact) == []

    duplicate = exp4915.build_artifact(
        loop_result=_loop_result(reached_level=2),
        prior_level=2,
        prior_total_levels=68,
        preconditions_checked=_preconditions(),
        approach_recommendation=_recommendations()["cn04"],
        registry_update={"updated": True, "banked_levels": 0, "reason": "duplicate_depth"},
        candidate_selection={"game": "cn04"},
    )
    assert duplicate["honest_verdict"] == "complete_cn04_no_new_level_residual_duplicate_depth"
    assert duplicate["offline_reproduced"] is False
    assert duplicate["new_levels_banked"] == 0
    assert duplicate["retire_if_same_verdict"] is True


def test_scenario_report_4915_cn04_adapter_exposes_l3_delta() -> None:
    """SCENARIO-REPORT-4915: the CN04 adapter exposes the new L3 marker-pair tail."""

    adapter = adapters.get_adapter("cn04")
    assert adapter is not None

    frame = SimpleNamespace(levels_completed=2)
    first_l3 = adapter.action_labels(None, frame, ())
    next_l3 = adapter.action_labels(None, frame, ("done",))
    exhausted_l3 = adapter.action_labels(
        None,
        frame,
        tuple("x" for _ in adapters.CN04_L3_TAIL_LABELS),
    )

    assert len(adapters.CN04_L3_TAIL_LABELS) == 20
    assert adapters.CN04_L3_TAIL_LABELS[:3] == (
        adapters._json_action_label(4),
        adapters._json_action_label(4),
        adapters._json_action_label(4),
    )
    assert adapters.CN04_L3_TAIL_LABELS[-6:] == tuple(adapters._json_action_label(2) for _ in range(6))
    assert first_l3 == [adapters.CN04_L3_TAIL_LABELS[0]]
    assert next_l3 == [adapters.CN04_L3_TAIL_LABELS[1]]
    assert exhausted_l3 == []
    assert adapter.depth_caps[3] == len(adapters.CN04_L3_TAIL_LABELS)
    assert adapter.level_tails[3] == adapters.CN04_L3_TAIL_LABELS


def test_req_report_4915_residual_blocked_and_schema_errors_are_explicit() -> None:
    """SCENARIO-REPORT-4915-BLOCKED-PRECONDITION: no-bank causes never fabricate progress."""

    assert exp4915._residual_cause(_loop_result(reached_level=2), 2) == "duplicate_depth"
    assert (
        exp4915._residual_cause({"status": "needs_per_game_RE", "offline_reproduced": False}, 2)
        == "needs_per_game_re"
    )
    assert exp4915._residual_cause({"offline_reproduced": False}, 2) == "offline_reproduction_failed"

    artifact = exp4915.blocked_artifact(
        target_game="cn04",
        reason="offline_env_missing",
        preconditions_checked={"offline_arcade_exits_0": True, "target_env_present": False},
    )

    assert artifact["honest_verdict"] == "blocked_cn04_offline_env_missing"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["new_levels_banked"] == 0
    assert artifact["schema_errors"] == []

    malformed = {"target_game": "m0r0", "reproducibility_checksum": "not-a-checksum"}
    errors = exp4915.artifact_schema_errors(malformed)
    assert "missing required field: honest_verdict" in errors
    assert "experiment mismatch" in errors
    assert "target_game violates rotation exclusions" in errors
    assert "honest_verdict must use a terminal prefix" in errors

    good = exp4915.build_artifact(
        loop_result=_loop_result(reached_level=3),
        prior_level=2,
        prior_total_levels=68,
        preconditions_checked=_preconditions(),
        approach_recommendation=_recommendations()["cn04"],
        registry_update={"updated": True, "banked_levels": 1},
        candidate_selection={"game": "cn04"},
    )
    corrupted = dict(good)
    corrupted["reproducibility_checksum"] = "0" * 64
    assert "checksum mismatch" in exp4915.artifact_schema_errors(corrupted)

    impossible_success = dict(good)
    impossible_success.update(offline_reproduced=False, reproduced_levels=2, new_levels_banked=0)
    success_errors = exp4915.artifact_schema_errors(impossible_success)
    assert "success requires offline_reproduced true" in success_errors
    assert "success requires new_levels_banked >= 1" in success_errors
    assert "success requires reproduced_levels > prior_reproduced_level" in success_errors

    impossible_oracle = dict(good)
    impossible_oracle.update(
        honest_verdict="complete_cn04_no_new_level_residual_duplicate_depth",
        offline_reproduced=True,
        new_levels_banked=0,
    )
    assert (
        "offline_reproduced true requires new_levels_banked >= 1"
        in exp4915.artifact_schema_errors(impossible_oracle)
    )


def test_scenario_report_4915_registry_update_records_bank_or_dead_end() -> None:
    """SCENARIO-REPORT-4915-REPRODUCTION-GATE: registry updates only the selected game row."""

    artifact = exp4915.build_artifact(
        loop_result=_loop_result(reached_level=3),
        prior_level=2,
        prior_total_levels=68,
        preconditions_checked=_preconditions(),
        approach_recommendation=_recommendations()["cn04"],
        registry_update={"updated": True, "banked_levels": 1},
        candidate_selection={"game": "cn04"},
    )
    updated_text, update = exp4915.apply_registry_result(_registry_text(), artifact=artifact)
    registry = yaml.safe_load(updated_text)
    cn04 = next(row for row in registry["games"] if row["game"] == "cn04")
    sp80 = next(row for row in registry["games"] if row["game"] == "sp80")

    assert update["updated"] is True
    assert update["prior_game_levels"] == 2
    assert update["new_game_levels"] == 3
    assert update["banked_levels"] == 1
    assert registry["reproducible_total_levels"] == 69
    assert cn04["levels_reproduced"] == 3
    assert cn04["latest_exp4915_levelup_attempt"]["new_levels_banked"] == 1
    assert "Exp4915" in cn04["reproduce"]
    assert sp80["levels_reproduced"] == 2

    no_bank = exp4915.build_artifact(
        loop_result=_loop_result(reached_level=2),
        prior_level=2,
        prior_total_levels=68,
        preconditions_checked=_preconditions(),
        approach_recommendation=_recommendations()["cn04"],
        registry_update={"updated": True, "banked_levels": 0, "reason": "duplicate_depth"},
        candidate_selection={"game": "cn04"},
    )
    dead_end_text, dead_end_update = exp4915.apply_registry_result(_registry_text(), artifact=no_bank)
    dead_end_registry = yaml.safe_load(dead_end_text)
    dead_end_cn04 = next(row for row in dead_end_registry["games"] if row["game"] == "cn04")

    assert dead_end_update["updated"] is True
    assert dead_end_update["banked_levels"] == 0
    assert dead_end_registry["reproducible_total_levels"] == 68
    assert any("Exp4915 cn04 no-bank duplicate_depth" in item for item in dead_end_cn04["dead_ends"])
    assert exp4915._artifact_residual_reason({"registry_update": {"reason": "fallback"}}) == "fallback"


def test_scenario_report_4915_run_experiment_writes_artifact_and_registry(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4915: runner writes stable JSON and registry evidence."""

    (tmp_path / "ops").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / exp4915.REGISTRY_RELATIVE_PATH).write_text(_registry_text(), encoding="utf-8")

    no_bank = exp4915.run_experiment(
        root=tmp_path,
        target_game="cn04",
        loop_result=_loop_result(reached_level=2),
        approach_recommendation=_recommendations()["cn04"],
        preconditions_checked=_preconditions(),
    )

    written = json.loads((tmp_path / exp4915.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    registry = yaml.safe_load((tmp_path / exp4915.REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))
    cn04 = next(row for row in registry["games"] if row["game"] == "cn04")

    assert no_bank == written
    assert written["schema_errors"] == []
    assert written["honest_verdict"] == "complete_cn04_no_new_level_residual_duplicate_depth"
    assert written["standing_loop_result_path"] == "results/arc_loop_solve_cn04.json"
    assert cn04["latest_exp4915_levelup_attempt"]["reproducibility_checksum"] == no_bank[
        "reproducibility_checksum"
    ]

    blocked = exp4915.run_experiment(
        root=tmp_path,
        target_game="cn04",
        preconditions_checked={"offline_arcade_exits_0": False, "target_env_present": True},
    )
    assert blocked["honest_verdict"] == "blocked_cn04_offline_arcade_missing"
    assert blocked["new_levels_banked"] == 0

    rotated_root = tmp_path / "explicit_after_rotation"
    (rotated_root / "ops").mkdir(parents=True)
    (rotated_root / "results").mkdir()
    rotated_registry = _registry_text().replace(
        "  - Pre-adapter arc_loop_solve --game cn04 returned needs_per_game_RE.",
        (
            "  - Pre-adapter arc_loop_solve --game cn04 returned needs_per_game_RE.\n"
            "  - Exp4915 cn04 no-bank duplicate_depth."
        ),
    )
    (rotated_root / exp4915.REGISTRY_RELATIVE_PATH).write_text(rotated_registry, encoding="utf-8")
    explicit = exp4915.run_experiment(
        root=rotated_root,
        target_game="cn04",
        loop_result=_loop_result(reached_level=2),
        approach_recommendation=_recommendations()["cn04"],
        preconditions_checked=_preconditions(),
    )

    assert explicit["candidate_selection"]["game"] != "cn04"
    assert explicit["candidate_selection"]["actual_target_game"] == "cn04"

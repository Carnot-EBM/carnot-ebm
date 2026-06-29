"""Tests for Exp 4969 fresh deep ARC level-up attempt.

Spec refs: REQ-ARC-WMTE-4969,
SCENARIO-ARC-WMTE-4969-FRESH-DEEP-TARGET,
SCENARIO-ARC-WMTE-4969-REPRODUCTION-GATE,
SCENARIO-ARC-WMTE-4969-STABLE-ARTIFACT.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_4969_levelup_attempt as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _game(
    game: str,
    level: int,
    *,
    reproducibility: str = "reproduced",
    solver: str = "GameAdapter",
    mechanic_class: str = "graph_explore",
    dead_ends: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "game": game,
        "reproducibility": reproducibility,
        "levels_reproduced": level,
        "mechanic_class": mechanic_class,
        "solver": solver,
        "dead_ends": list(dead_ends or []),
    }


def _registry() -> dict[str, Any]:
    return {
        "games": [
            _game("tr87", 6, dead_ends=["Exp4958 no_grounded_l7_delta"]),
            _game("s5i5", 2, dead_ends=["Exp4959 no_grounded_l3_delta"]),
            _game("tu93", 5, solver="GameAdapter _tu93 branch_mode=fresh_env"),
            _game("tn36", 7, mechanic_class="program_editor"),
            _game("cn04", 3, dead_ends=["pre-adapter L2 delta was missing"]),
            _game("ka59", 1, dead_ends=["hidden-state-bound"]),
        ],
        "general_gotchas": [{"id": "non_idempotent_reset"}],
    }


def _selection() -> dict[str, Any]:
    return mod.select_target(
        _registry(),
        recommend_fn=lambda game: {"recommended": [{"game": "ls20"}], "game": game},
    )


def _preconditions() -> dict[str, Any]:
    return {
        "offline_arcade": {"ok": True, "backend": "arc_solver_kit.offline_arcade"},
        "target_env": {"ok": True, "game": "tu93"},
        "generator": {"required": False, "checked": False},
    }


def test_req_arc_wmte_4969_spec_declares_artifact_contract() -> None:
    """REQ-ARC-WMTE-4969: OpenSpec anchors fields, scenarios, and result path."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-ARC-WMTE-4969",
        "SCENARIO-ARC-WMTE-4969-FRESH-DEEP-TARGET",
        "SCENARIO-ARC-WMTE-4969-REPRODUCTION-GATE",
        "SCENARIO-ARC-WMTE-4969-STABLE-ARTIFACT",
        mod.RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4969_selects_tu93_l6_not_duplicate_l3() -> None:
    """SCENARIO-ARC-WMTE-4969-FRESH-DEEP-TARGET: fresh deep target is prior+1."""

    calls: list[str] = []
    selection = mod.select_target(
        _registry(),
        recommend_fn=lambda game: calls.append(game) or {"route": "goal_distance", "game": game},
    )

    assert selection["target_game"] == "tu93"
    assert selection["prior_reproduced_levels"] == 5
    assert selection["target_level"] == 6
    assert selection["grounded_next_level_delta"] is True
    assert calls == ["tu93"]
    assert "tr87" in selection["excluded_lanes"]
    assert "ka59" in selection["excluded_lanes"]


def test_scenario_arc_wmte_4969_success_artifact_banks_strictly_deeper_level() -> None:
    """SCENARIO-ARC-WMTE-4969-REPRODUCTION-GATE: only prior+1 reproduction banks."""

    selection = _selection()
    loop_result = {
        "game": "tu93",
        "offline_reproduced": True,
        "reproduced_levels": 6,
        "reached_level": 6,
        "solution_labels": ['{"action": 1}', '{"action": 2}'],
        "reproduction_gate": {"reproduced": True, "claimed_level": 6},
        "solve_provenance": "development_proxy",
    }

    artifact = mod.build_artifact(
        selection=selection,
        preconditions_checked=_preconditions(),
        loop_result=loop_result,
        loop_artifact="results/arc_loop_solve_tu93.json",
        loop_command=[".venv/bin/python", "scripts/arc_loop_solve.py", "--game", "tu93", "--target-level", "6"],
    )

    assert artifact["honest_verdict"] == "success_tu93_levelup_banked"
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 6
    assert artifact["new_levels_banked"] == 1
    assert artifact["registry_update"]["new_total_levels"] == 6
    assert artifact["retire_if_same_verdict"] is False
    assert artifact["schema_errors"] == []
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_arc_wmte_4969_same_depth_is_no_bank_dead_end() -> None:
    """SCENARIO-ARC-WMTE-4969-REPRODUCTION-GATE: duplicate depth never banks."""

    selection = _selection()
    loop_result = {
        "game": "tu93",
        "offline_reproduced": True,
        "reproduced_levels": 5,
        "reached_level": 5,
        "solution_labels": ['{"action": 1}'],
        "reproduction_gate": {"reproduced": True, "claimed_level": 5},
    }

    artifact = mod.build_artifact(
        selection=selection,
        preconditions_checked=_preconditions(),
        loop_result=loop_result,
        loop_artifact="results/arc_loop_solve_tu93.json",
        loop_command=[".venv/bin/python", "scripts/arc_loop_solve.py", "--game", "tu93", "--target-level", "6"],
    )

    assert artifact["honest_verdict"] == "complete_tu93_no_new_level_residual_duplicate_depth"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 5
    assert artifact["new_levels_banked"] == 0
    assert artifact["retire_if_same_verdict"] is True
    assert artifact["dead_ends"][-1]["residual_cause"] == "duplicate_depth"
    assert artifact["registry_update"]["action"] == "record_dead_end_only"
    assert artifact["schema_errors"] == []


def test_scenario_arc_wmte_4969_blocked_artifact_does_not_fabricate_levels() -> None:
    """SCENARIO-ARC-WMTE-4969-STABLE-ARTIFACT: blocked env reports zero new bank."""

    selection = _selection()
    artifact = mod.build_blocked_artifact(
        selection=selection,
        preconditions_checked={
            "offline_arcade": {"ok": False, "error": "arcade missing"},
            "target_env": {"ok": False, "game": "tu93"},
            "generator": {"required": False, "checked": False},
        },
        residual_cause="offline_env_missing",
    )

    assert artifact["honest_verdict"] == "blocked_tu93_offline_env_missing"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 5
    assert artifact["new_levels_banked"] == 0
    assert artifact["live_path_reachable"] is False
    assert artifact["schema_errors"] == []


def test_req_arc_wmte_4969_checksum_is_stable_and_content_sensitive() -> None:
    """REQ-ARC-WMTE-4969: checksum hashes game, plan, and claimed level."""

    checksum_a = mod.reproducibility_checksum("tu93", ['{"action": 1}'], 6)
    checksum_b = mod.reproducibility_checksum("tu93", ['{"action": 1}'], 6)
    checksum_c = mod.reproducibility_checksum("tu93", ['{"action": 2}'], 6)

    assert checksum_a == checksum_b
    assert checksum_a.startswith("sha256:")
    assert checksum_a != checksum_c


def test_req_arc_wmte_4969_loader_and_command_helpers_are_deterministic(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4969: file and command helpers preserve the artifact contract."""

    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "arc_solve_registry.yaml").write_text(
        "games:\n- game: tu93\n  levels_reproduced: 5\n",
        encoding="utf-8",
    )
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps({"ok": True}), encoding="utf-8")

    selection = _selection()

    assert mod._load_json(payload_path) == {"ok": True}
    assert mod._load_registry(tmp_path)["games"][0]["game"] == "tu93"
    assert mod._dead_ends({"dead_ends": "one"}) == ["one"]
    assert mod.loop_result_relative_path("tu93") == "results/arc_loop_solve_tu93.json"
    assert mod.loop_command(selection)[-2:] == ["--target-level", "6"]


def test_scenario_arc_wmte_4969_no_grounded_delta_records_dead_end() -> None:
    """SCENARIO-ARC-WMTE-4969-FRESH-DEEP-TARGET: no grounded delta exits honestly."""

    registry = {
        "games": [
            _game("tu93", 5, dead_ends=["Exp4969 no_grounded_l6_delta"]),
        ]
    }
    selection = mod.select_target(
        registry,
        recommend_fn=lambda game: {"unexpected": game},
    )
    artifact = mod.build_no_grounded_delta_artifact(
        selection=selection,
        preconditions_checked=_preconditions(),
    )

    assert selection["target_game"] == "none"
    assert selection["grounded_next_level_delta"] is False
    assert selection["candidate_audit"][0]["status"] == "skip_no_grounded_delta"
    assert selection["candidate_audit"][1]["status"] == "missing_registry_row"
    assert artifact["honest_verdict"] == "complete_none_no_new_level_residual_no_grounded_next_level_delta"
    assert artifact["registry_update"]["action"] == "record_dead_end_only"
    assert artifact["schema_errors"] == []


def test_scenario_arc_wmte_4969_residual_cause_and_solution_fallbacks() -> None:
    """SCENARIO-ARC-WMTE-4969-REPRODUCTION-GATE: residual causes stay explicit."""

    selection = _selection()
    command = mod.loop_command(selection)
    needs_re = mod.build_artifact(
        selection=selection,
        preconditions_checked=_preconditions(),
        loop_result={
            "game": "tu93",
            "status": "needs_per_game_RE",
            "offline_reproduced": False,
            "reproduced_levels": 5,
            "solution": [{"action": 1}],
        },
        loop_artifact="results/arc_loop_solve_tu93.json",
        loop_command=command,
    )
    failed_gate = mod.build_artifact(
        selection=selection,
        preconditions_checked=_preconditions(),
        loop_result={"game": "tu93", "offline_reproduced": False, "reproduced_levels": 0},
        loop_artifact="results/arc_loop_solve_tu93.json",
        loop_command=command,
    )
    timeout = mod.build_artifact(
        selection=selection,
        preconditions_checked=_preconditions(),
        loop_result={
            "game": "tu93",
            "offline_reproduced": False,
            "reproduced_levels": 5,
            "status": "standing_loop_timeout",
            "standing_loop_subprocess": {"returncode": "timeout", "timeout_s": 120},
        },
        loop_artifact=None,
        loop_command=command,
    )

    assert needs_re["honest_verdict"] == "complete_tu93_no_new_level_residual_needs_per_game_re"
    assert needs_re["solution_labels"] == [{"action": 1}]
    assert failed_gate["honest_verdict"] == (
        "complete_tu93_no_new_level_residual_offline_reproduction_failed"
    )
    assert timeout["honest_verdict"] == "complete_tu93_no_new_level_residual_standing_loop_timeout"
    assert timeout["loop_artifact"] is None
    assert timeout["standing_loop_subprocess"]["returncode"] == "timeout"
    assert failed_gate["solution_labels"] == []
    assert needs_re["schema_errors"] == []
    assert failed_gate["schema_errors"] == []


def test_req_arc_wmte_4969_schema_errors_fail_closed() -> None:
    """REQ-ARC-WMTE-4969: malformed artifacts produce named schema errors."""

    good = mod.build_artifact(
        selection=_selection(),
        preconditions_checked=_preconditions(),
        loop_result={
            "game": "tu93",
            "offline_reproduced": True,
            "reproduced_levels": 6,
            "reached_level": 6,
            "solution_labels": ['{"action": 1}'],
            "reproduction_gate": {"reproduced": True, "claimed_level": 6},
        },
        loop_artifact="results/arc_loop_solve_tu93.json",
        loop_command=mod.loop_command(_selection()),
    )
    bad = {
        **good,
        "experiment": "wrong",
        "schema": "wrong",
        "spec_refs": [],
        "field_principles": {},
        "honest_verdict": "wrong",
        "solve_provenance": "outer_loop_re",
        "target_game": None,
        "offline_reproduced": "yes",
        "reproduced_levels": "6",
        "new_levels_banked": "1",
        "live_path_reachable": "true",
        "verifier_is_oracle": False,
        "inference_substrate": "live_llm_inference",
        "preconditions_checked": None,
        "random_seed": 0,
        "reproducibility_checksum": "bad",
        "retire_if_same_verdict": False,
    }
    success_bad = {
        **good,
        "offline_reproduced": False,
        "reproduced_levels": 5,
        "new_levels_banked": 0,
    }
    offline_bad = {
        **good,
        "honest_verdict": "complete_tu93_no_new_level_residual_duplicate_depth",
        "offline_reproduced": True,
        "new_levels_banked": 0,
        "retire_if_same_verdict": False,
    }
    missing = dict(good)
    del missing["schema_errors"]

    errors = set(mod.artifact_schema_errors(bad))
    assert {
        "experiment_mismatch",
        "schema_mismatch",
        "spec_refs_mismatch",
        "field_principles_mismatch",
        "honest_verdict_terminal_prefix",
        "solve_provenance_mismatch",
        "target_game_type",
        "offline_reproduced_type",
        "reproduced_levels_type",
        "new_levels_banked_type",
        "live_path_reachable_type",
        "verifier_is_oracle_not_true",
        "inference_substrate_mismatch",
        "preconditions_checked_type",
        "random_seed_mismatch",
        "reproducibility_checksum_format",
        "retire_flag_false_on_non_success",
    } <= errors
    assert "success_requires_offline_reproduced" in mod.artifact_schema_errors(success_bad)
    assert "success_requires_new_level" in mod.artifact_schema_errors(success_bad)
    assert "success_requires_strictly_deeper_level" in mod.artifact_schema_errors(success_bad)
    assert "offline_reproduced_without_bank" in mod.artifact_schema_errors(offline_bad)
    assert "missing:schema_errors" in mod.artifact_schema_errors(missing)


def test_req_arc_wmte_4969_precondition_probe_paths(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4969: preconditions distinguish missing env and arcade errors."""

    selection = _selection()

    missing_env = mod.check_preconditions(selection, root=tmp_path)
    assert missing_env["target_env"]["ok"] is False
    assert missing_env["target_env"]["error"] == "environment_files entry missing"

    env_dir = tmp_path / "environment_files" / "tu93"
    env_dir.mkdir(parents=True)

    class FakeEnv:
        def reset(self) -> None:
            return None

    class FakeArcade:
        def make(self, game: str, scorecard_id: str) -> FakeEnv:
            assert game == "tu93"
            assert scorecard_id == "score"
            return FakeEnv()

        def open_scorecard(self) -> str:
            return "score"

    ok = mod.check_preconditions(selection, root=tmp_path, offline_arcade_factory=FakeArcade)
    failed = mod.check_preconditions(
        selection,
        root=tmp_path,
        offline_arcade_factory=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    assert ok["offline_arcade"]["ok"] is True
    assert ok["target_env"]["ok"] is True
    assert failed["offline_arcade"]["ok"] is False
    assert failed["offline_arcade"]["error"] == "boom"
    assert failed["target_env"]["error"] == "boom"

    real = mod.check_preconditions(selection, root=REPO)
    assert real["offline_arcade"]["ok"] is True
    assert real["target_env"]["ok"] is True


def test_req_arc_wmte_4969_run_experiment_writes_success(tmp_path: Path, monkeypatch: Any) -> None:
    """REQ-ARC-WMTE-4969: run_experiment writes the stable result artifact."""

    monkeypatch.setattr(mod, "check_preconditions", lambda selection, root: _preconditions())
    artifact = mod.run_experiment(
        root=tmp_path,
        registry=_registry(),
        recommend_fn=lambda game: {"game": game},
        loop_result={
            "game": "tu93",
            "offline_reproduced": True,
            "reproduced_levels": 6,
            "reached_level": 6,
            "solution_labels": ['{"action": 1}'],
            "reproduction_gate": {"reproduced": True, "claimed_level": 6},
        },
    )

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    assert artifact["honest_verdict"] == "success_tu93_levelup_banked"
    assert result_path.exists()
    assert json.loads(result_path.read_text(encoding="utf-8"))["schema_errors"] == []


def test_req_arc_wmte_4969_run_experiment_handles_blocked_and_no_delta(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """REQ-ARC-WMTE-4969: run_experiment exits cleanly for blocked/no-delta paths."""

    monkeypatch.setattr(
        mod,
        "check_preconditions",
        lambda selection, root: {
            "offline_arcade": {"ok": False},
            "target_env": {"ok": False, "game": selection["target_game"]},
            "generator": {"required": False, "checked": False},
        },
    )
    blocked = mod.run_experiment(
        root=tmp_path,
        registry=_registry(),
        recommend_fn=lambda game: {"game": game},
        loop_result={},
    )

    monkeypatch.setattr(
        mod,
        "check_preconditions",
        lambda selection, root: {
            "offline_arcade": {"ok": True},
            "target_env": {"ok": False, "game": selection["target_game"]},
            "generator": {"required": False, "checked": False},
        },
    )
    target_blocked = mod.run_experiment(
        root=tmp_path,
        registry=_registry(),
        recommend_fn=lambda game: {"game": game},
        loop_result={},
    )

    monkeypatch.setattr(mod, "check_preconditions", lambda selection, root: _preconditions())
    no_delta = mod.run_experiment(
        root=tmp_path,
        registry={"games": [_game("tu93", 5, dead_ends=["no_grounded_l6_delta"])]},
        recommend_fn=lambda game: {"game": game},
        loop_result={},
    )

    assert blocked["honest_verdict"] == "blocked_tu93_offline_env_missing"
    assert target_blocked["honest_verdict"] == "blocked_tu93_offline_env_missing"
    assert no_delta["honest_verdict"] == "complete_none_no_new_level_residual_no_grounded_next_level_delta"

"""Tests for Exp 4415 Agent2World adaptive ARC E3 repair.

Spec refs: REQ-PHASE4-4415, SCENARIO-PHASE4-4415.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

import carnot.experiment_4415_agent2world_adaptive_e3_repair as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"


def _write_fake_repo(tmp_path: Path) -> None:
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "arc_solve_registry.yaml").write_text(
        yaml.safe_dump(
            {
                "games": [
                    {"game": "ar25", "levels_reproduced": 1},
                    {"game": "tn36", "levels_reproduced": 7},
                    {"game": "lp85", "levels_reproduced": 5},
                ],
                "reproducible_total_levels": 34,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "ops" / "verifier_gaps.md").write_text("# gaps\n", encoding="utf-8")
    for game in exp.TARGET_ORDER:
        env_dir = tmp_path / "environment_files" / game
        env_dir.mkdir(parents=True)
        (env_dir / "fixture").write_text("present\n", encoding="utf-8")


def _adaptive_checks(game: str) -> list[dict[str, object]]:
    return [
        {
            "game": game,
            "name": f"{game}_adaptive_repair_pass",
            "round": 1,
            "source_failing_transition": f"{game}:rollout:mismatch0",
            "derived_from_rollout_trace": True,
            "fresh_agent_state": True,
            "expected": "repair",
            "observed": "repair",
            "passed": True,
            "residual_behavior_after_test": f"{game}_residual_after_round1",
            "test_path": exp.UNIT_TEST_PATH,
            "world_model_path": exp.WORLD_MODEL_PATHS[game],
        },
        {
            "game": game,
            "name": f"{game}_adaptive_residual_gap",
            "round": 2,
            "source_failing_transition": f"{game}:rollout:mismatch1",
            "derived_from_rollout_trace": True,
            "fresh_agent_state": True,
            "expected": "fixed",
            "observed": "still_wrong",
            "passed": False,
            "residual_behavior_after_test": f"{game}_residual_after_round2",
            "test_path": exp.UNIT_TEST_PATH,
            "world_model_path": exp.WORLD_MODEL_PATHS[game],
        },
    ]


def _passing_check_runner(repo: Path, game: str) -> list[dict[str, object]]:
    return _adaptive_checks(game)


def test_req_phase4_4415_spec_declares_adaptive_contract() -> None:
    """REQ-PHASE4-4415: OpenSpec declares the adaptive repair artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-4415" in spec
    assert "SCENARIO-PHASE4-4415" in spec
    assert "results/experiment_4415_agent2world_adaptive_e3_repair.json" in spec
    assert "adaptive behavior-aware tests" in spec
    assert "held-out mechanic checks" in spec
    for principle in exp.REQUIRED_FIELD_PRINCIPLES.values():
        assert principle in spec


def test_req_phase4_4415_adaptive_checks_are_trace_derived() -> None:
    """REQ-PHASE4-4415: adaptive checks carry rollout-trace provenance."""

    checks = exp.run_adaptive_checks(REPO)
    by_game = {game: [row for row in checks if row["game"] == game] for game in exp.TARGET_ORDER}

    assert all(by_game.values())
    assert any(row["passed"] is True for row in checks)
    assert any(row["passed"] is False for row in checks)
    assert all(row["derived_from_rollout_trace"] is True for row in checks)
    assert all(str(row["source_failing_transition"]).startswith(tuple(exp.TARGET_ORDER)) for row in checks)
    assert all(row["fresh_agent_state"] is True for row in checks)
    assert all("named_register" not in row for row in checks)
    assert exp.held_out_check_for_game(REPO, "lp85") is True


def test_req_phase4_4415_fallback_fixture_branches(tmp_path: Path, monkeypatch) -> None:
    """REQ-PHASE4-4415: fallback adaptive probes remain trace-derived."""

    wm_dir = tmp_path / "results" / "arc_e3" / "ar25"
    wm_dir.mkdir(parents=True)
    wm_path = wm_dir / "world_model.py"
    wm_path.write_text(
        "def transition_fixture():\n"
        "    return {'expected': 'ok', 'observed': 'ok', 'passed': True}\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(exp.WORLD_MODEL_PATHS, "ar25", "results/arc_e3/ar25/world_model.py")

    checks = exp.adaptive_checks_for_game(tmp_path, "ar25")

    assert checks[0]["source_failing_transition"] == "ar25:rollout:prior_fixture_residual"
    assert checks[0]["passed"] is True
    assert exp.held_out_check_for_game(tmp_path, "ar25") is True

    wm_path.write_text("VALUE = 1\n", encoding="utf-8")
    missing = exp.adaptive_checks_for_game(tmp_path, "ar25")

    assert missing[0]["source_failing_transition"] == "ar25:rollout:fixture_missing"
    assert missing[0]["passed"] is False
    assert exp.held_out_check_for_game(tmp_path, "ar25") is False

    monkeypatch.setattr(exp.importlib.util, "spec_from_file_location", lambda *_args: None)
    try:
        exp._load_module_from_path(tmp_path, "missing.py", "missing_4415")
    except ImportError as exc:
        assert "cannot load" in str(exc)
    else:  # pragma: no cover - defensive assertion branch.
        raise AssertionError("expected ImportError")


def test_scenario_phase4_4415_partial_run_preserves_all_target_rows(tmp_path: Path, monkeypatch) -> None:
    """SCENARIO-PHASE4-4415: no reproduced level remains an honest adaptive partial."""

    _write_fake_repo(tmp_path)
    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "RESULT_PATH", tmp_path / exp.RESULT_RELATIVE_PATH)
    monkeypatch.setattr(exp, "adaptive_checks_for_game", _passing_check_runner)
    monkeypatch.setattr(exp, "held_out_check_for_game", lambda _repo, _game: True)
    monkeypatch.setattr(exp, "_imports_ok", lambda: {"harness_import": True, "solver_kit_import": True})
    monkeypatch.setattr(exp, "_research_conductor_modified", lambda _repo: False)

    artifact = exp.run_experiment(repo=tmp_path, write_artifact=True)

    assert artifact["honest_verdict"] == "complete_e3_adaptive_partial"
    assert artifact["new_levels_reproduced"] == 0
    assert artifact["reproducible_total_levels"] == 34
    assert [row["game"] for row in artifact["per_target_scorecard"]] == list(exp.TARGET_ORDER)
    assert all(row["adaptive_tests_passed"] == 1 for row in artifact["per_target_scorecard"])
    assert all(row["adaptive_tests_total"] == 2 for row in artifact["per_target_scorecard"])
    assert all(row["held_out_mechanic_test_pass"] is True for row in artifact["per_target_scorecard"])
    assert all(row["offline_reproduced"] is False for row in artifact["per_target_scorecard"])
    assert artifact["verifier_is_oracle"] is True
    assert Path(artifact["artifact_path"]).exists()
    assert not exp.artifact_schema_errors(artifact)


def test_req_phase4_4415_counts_only_reproduced_new_level(tmp_path: Path, monkeypatch) -> None:
    """REQ-PHASE4-4415: adaptive-test success does not count without reproduce()."""

    _write_fake_repo(tmp_path)
    monkeypatch.setattr(exp, "adaptive_checks_for_game", _passing_check_runner)
    monkeypatch.setattr(exp, "held_out_check_for_game", lambda _repo, _game: True)
    monkeypatch.setattr(exp, "_imports_ok", lambda: {"harness_import": True, "solver_kit_import": True})
    monkeypatch.setattr(exp, "_research_conductor_modified", lambda _repo: False)

    def reproduction_runner(game: str, target_level: int) -> dict[str, object]:
        return {
            "game": game,
            "claimed_level": target_level,
            "reached_level": target_level if game == "tn36" else target_level - 1,
            "reproduced": game == "tn36",
        }

    artifact = exp.run_experiment(repo=tmp_path, reproduction_runner=reproduction_runner, write_artifact=False)
    rows = {row["game"]: row for row in artifact["per_target_scorecard"]}

    assert artifact["honest_verdict"] == "success_e3_tn36_reproduced"
    assert artifact["new_levels_reproduced"] == 1
    assert artifact["reproducible_total_levels"] == 35
    assert rows["tn36"]["new_reproduced_level"] == 8
    assert rows["ar25"]["new_reproduced_level"] == 1
    assert rows["lp85"]["new_reproduced_level"] == 5


def test_req_phase4_4415_missing_env_blocks_one_target_and_continues(tmp_path: Path, monkeypatch) -> None:
    """REQ-PHASE4-4415: missing offline env records a blocked row only for that target."""

    _write_fake_repo(tmp_path)
    for child in (tmp_path / "environment_files" / "lp85").iterdir():
        child.unlink()
    monkeypatch.setattr(exp, "adaptive_checks_for_game", _passing_check_runner)
    monkeypatch.setattr(exp, "held_out_check_for_game", lambda _repo, _game: True)
    monkeypatch.setattr(exp, "_imports_ok", lambda: {"harness_import": True, "solver_kit_import": True})
    monkeypatch.setattr(exp, "_research_conductor_modified", lambda _repo: False)

    artifact = exp.run_experiment(repo=tmp_path, write_artifact=False)
    rows = {row["game"]: row for row in artifact["per_target_scorecard"]}

    assert rows["lp85"]["checkpoint_status"] == "blocked_offline_env_missing_lp85"
    assert rows["lp85"]["adaptive_tests_total"] == 0
    assert rows["ar25"]["checkpoint_status"] == "honest_partial_adaptive_residual_behavior_remaining"


def test_req_phase4_4415_writes_adaptive_test_artifacts(tmp_path: Path, monkeypatch) -> None:
    """REQ-PHASE4-4415: adaptive-test files preserve trace and leakage-control metadata."""

    _write_fake_repo(tmp_path)
    monkeypatch.setattr(exp, "adaptive_checks_for_game", _passing_check_runner)
    monkeypatch.setattr(exp, "held_out_check_for_game", lambda _repo, _game: True)
    monkeypatch.setattr(exp, "_imports_ok", lambda: {"harness_import": True, "solver_kit_import": True})
    monkeypatch.setattr(exp, "_research_conductor_modified", lambda _repo: False)

    artifact = exp.run_experiment(repo=tmp_path, write_artifact=True)
    ar25_path = tmp_path / "results" / "arc_e3" / "ar25" / "adaptive_tests_4415.json"
    payload = json.loads(ar25_path.read_text(encoding="utf-8"))

    assert str(ar25_path.relative_to(tmp_path)) in artifact["world_model_paths"]
    assert payload["game"] == "ar25"
    assert payload["held_out_mechanic_test_pass"] is True
    assert payload["fresh_agent_state"] is True
    assert payload["solve_claim_separate_from_test_repair"] is True
    assert payload["tests"][0]["source_failing_transition"] == "ar25:rollout:mismatch0"


def test_req_phase4_4415_gap_checkpoint_is_idempotent(tmp_path: Path) -> None:
    """REQ-PHASE4-4415: residual behavior checkpoints append once and skip reproduced rows."""

    assert exp.read_registry_total(tmp_path) == exp.PRIOR_REPRODUCIBLE_TOTAL_LEVELS
    rows = [
        {
            "game": "ar25",
            "target_level": 2,
            "adaptive_tests_passed": 1,
            "adaptive_tests_total": 2,
            "residual_failing_behavior": "ar25_gap",
            "offline_reproduced": False,
        },
        {
            "game": "tn36",
            "target_level": 8,
            "adaptive_tests_passed": 2,
            "adaptive_tests_total": 2,
            "residual_failing_behavior": "none",
            "offline_reproduced": True,
        },
    ]

    exp.write_verifier_gap_checkpoint(tmp_path, rows)
    path = tmp_path / "ops" / "verifier_gaps.md"
    path.parent.mkdir()
    path.write_text("# gaps\n", encoding="utf-8")
    exp.write_verifier_gap_checkpoint(tmp_path, rows)
    exp.write_verifier_gap_checkpoint(tmp_path, rows)
    text = path.read_text(encoding="utf-8")

    assert text.count("exp4415-gap-ar25-l2:start") == 1
    assert "exp4415-gap-tn36-l8" not in text
    assert "ar25_gap" in text


def test_req_phase4_4415_schema_errors_are_specific() -> None:
    """REQ-PHASE4-4415: malformed artifacts report actionable schema errors."""

    bad = {
        "honest_verdict": "complete_e3_adaptive_partial",
        "per_target_scorecard": [{"game": "ar25", "offline_reproduced": "false"}],
        "reproducible_total_levels": "34",
        "new_levels_reproduced": {"value": 0},
        "world_model_paths": ["ok.py", 12],
        "verifier_is_oracle": False,
        "preconditions_checked": {},
        "random_seed": "4415",
        "reproducibility_checksum": "short",
        "field_principles": {"honest_verdict": "wrong"},
    }

    errors = exp.artifact_schema_errors(bad)

    assert "per_target_scorecard[0] missing prior_best_level" in errors
    assert "per_target_scorecard[0] missing adaptive_tests_passed" in errors
    assert "per_target_scorecard[0] missing held_out_mechanic_test_pass" in errors
    assert "per_target_scorecard[0].offline_reproduced must be bare bool" in errors
    assert "per_target_scorecard[0].held_out_mechanic_test_pass must be bare bool" in errors
    assert "reproducible_total_levels must be bare int" in errors
    assert "new_levels_reproduced must be bare int" in errors
    assert "world_model_paths must be list[str]" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "random_seed must be bare int" in errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in errors
    assert "principle mismatch for honest_verdict" in errors

    missing = exp.artifact_schema_errors({"per_target_scorecard": "bad"})
    assert "missing honest_verdict" in missing
    assert "per_target_scorecard must be list" in missing
    assert "field_principles missing" in missing

    row_errors = exp.artifact_schema_errors(
        {
            "honest_verdict": "complete_e3_adaptive_partial",
            "per_target_scorecard": ["bad-row"],
            "reproducible_total_levels": 34,
            "new_levels_reproduced": 0,
            "world_model_paths": [],
            "verifier_is_oracle": True,
            "preconditions_checked": {},
            "random_seed": 4415,
            "reproducibility_checksum": "0" * 64,
            "field_principles": exp.REQUIRED_FIELD_PRINCIPLES,
        }
    )
    assert "per_target_scorecard[0] must be dict" in row_errors


def test_req_phase4_4415_schema_failure_blocks_artifact(tmp_path: Path, monkeypatch) -> None:
    """REQ-PHASE4-4415: schema failures stop before writing a complete artifact."""

    _write_fake_repo(tmp_path)
    monkeypatch.setattr(exp, "adaptive_checks_for_game", _passing_check_runner)
    monkeypatch.setattr(exp, "held_out_check_for_game", lambda _repo, _game: True)
    monkeypatch.setattr(exp, "_imports_ok", lambda: {"harness_import": True, "solver_kit_import": True})
    monkeypatch.setattr(exp, "_research_conductor_modified", lambda _repo: False)
    monkeypatch.setattr(exp, "artifact_schema_errors", lambda _artifact: ["forced_error"])

    try:
        exp.run_experiment(repo=tmp_path, write_artifact=False)
    except ValueError as exc:
        assert "forced_error" in str(exc)
    else:  # pragma: no cover - defensive assertion branch.
        raise AssertionError("expected ValueError")


def test_req_phase4_4415_default_residual_for_all_passing_checks() -> None:
    """REQ-PHASE4-4415: all-passing adaptive tests still leave solve residual separate."""

    assert exp._last_residual_behavior([{"passed": True}], "ar25") == exp.RESIDUAL_BEHAVIORS["ar25"]

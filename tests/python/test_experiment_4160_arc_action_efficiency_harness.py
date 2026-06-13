"""Tests for Exp 4160 ARC-AGI-3 offline action-efficiency harness.

Spec refs: REQ-PHASE4-052, SCENARIO-PHASE4-052.
"""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest

import carnot.experiment_4160_arc_action_efficiency_harness as exp
from carnot.experiment_4160_arc_action_efficiency_harness import (
    INFERENCE_SUBSTRATE,
    PRIOR_TOTAL_GAMES_SOLVED,
    REQUIRED_ARTIFACT_FIELDS,
    REQUIRED_FIELD_PRINCIPLES,
    REQUIREMENTS,
    EfficiencyMeasurement,
    IncrementalAttempt,
    OfflineBaseline,
    VerifierPrunerRun,
    artifact_schema_errors,
    blocked_artifact,
    build_artifact,
    load_access_probe_baselines,
    load_fixture_baselines,
    load_verified_pruner_run,
    record_random_greedy_baseline,
)


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"
BP35_ARTIFACT = REPO / "results" / "experiment_4129_fourteenth_game_explore_first.json"


def _baseline() -> OfflineBaseline:
    return OfflineBaseline(
        game_id="bp35-0a0ad940",
        level_index=1,
        actions_to_solve_or_timeout=21,
        policy="random_greedy",
        source="results/arc_agi3_access_probe.json",
    )


def _verifier() -> VerifierPrunerRun:
    return VerifierPrunerRun(
        game_id="bp35-0a0ad940",
        level_index=1,
        actions_to_solve=16,
        observed_transition_count=5,
        heldout_transition_count=11,
        validated=True,
        pruned_action_count=5,
        source_artifact="results/experiment_4129_fourteenth_game_explore_first.json",
    )


def _attempt(*, solved: bool = False) -> IncrementalAttempt:
    return IncrementalAttempt(
        target_game="r11l-495a7899",
        target_level=5,
        new_level_solved=solved,
        new_levels_solved=1 if solved else 0,
        actions_executed=8 if solved else 0,
        honest_verdict="success: incremental_progress_r11l-495a7899_advanced_to_L5_total14"
        if solved
        else "complete: incremental_progress_no_solve_r11l-495a7899_L5_no_verifier_validated_level_up_candidate",
        reason="" if solved else "no_verifier_validated_level_up_candidate",
    )


def _measurement(*, ratio_override: float | None = None) -> EfficiencyMeasurement:
    baseline = _baseline()
    verifier = _verifier()
    ratio = (
        float(ratio_override)
        if ratio_override is not None
        else baseline.actions_to_solve_or_timeout / verifier.actions_to_solve
    )
    return EfficiencyMeasurement(
        baseline=baseline,
        verifier=verifier,
        action_efficiency_ratio=ratio,
    )


def test_req_phase4_052_spec_declares_exp4160_contract() -> None:
    """REQ-PHASE4-052: OpenSpec declares the offline action-efficiency artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-052" in spec
    assert "SCENARIO-PHASE4-052" in spec
    assert "experiment_4160_arc_action_efficiency_harness.json" in spec
    assert "blocked_arc_offline_fixtures_missing" in spec
    assert "baseline_actions / verifier_actions" in spec
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for field in REQUIRED_FIELD_PRINCIPLES:
        assert field in spec


def test_req_phase4_052_loads_offline_baseline_sources(tmp_path: Path) -> None:
    """REQ-PHASE4-052: random/greedy baseline actions come from offline fixtures."""

    probe = {
        "games": [
            {"game_id": "bp35-0a0ad940", "baseline_actions": [21, 48]},
            {"game_id": "empty-00000000", "baseline_actions": []},
            {"game_id": "", "baseline_actions": [99]},
            ["not", "a", "dict"],
        ]
    }
    probe_path = tmp_path / "probe.json"
    probe_path.write_text(json.dumps(probe), encoding="utf-8")
    assert load_access_probe_baselines(probe_path) == {"bp35-0a0ad940": 21}

    bad = tmp_path / "environment_files" / "bad" / "0000"
    bad.mkdir(parents=True)
    bad.joinpath("metadata.json").write_text("{bad json", encoding="utf-8")
    missing_py = tmp_path / "environment_files" / "aa00" / "1111"
    missing_py.mkdir(parents=True)
    missing_py.joinpath("metadata.json").write_text(
        json.dumps({"game_id": "aa00-1111", "baseline_actions": [9]}),
        encoding="utf-8",
    )
    no_hyphen = tmp_path / "environment_files" / "cc03" / "cccc3333"
    no_hyphen.mkdir(parents=True)
    no_hyphen.joinpath("metadata.json").write_text(
        json.dumps({"game_id": "cc03", "baseline_actions": [8]}),
        encoding="utf-8",
    )
    good = tmp_path / "environment_files" / "bp35" / "0a0ad940"
    good.mkdir(parents=True)
    good.joinpath("metadata.json").write_text(
        json.dumps({"game_id": "bp35-0a0ad940", "baseline_actions": [22]}),
        encoding="utf-8",
    )
    good.joinpath("bp35.py").write_text("# offline fixture marker\n", encoding="utf-8")

    fixtures = load_fixture_baselines(tmp_path / "environment_files")
    assert fixtures == {"bp35-0a0ad940": 22}

    baseline = record_random_greedy_baseline(
        "bp35-0a0ad940",
        access_baselines={"bp35-0a0ad940": 21},
        fixture_baselines=fixtures,
        access_source="results/arc_agi3_access_probe.json",
        fixture_source="environment_files",
    )
    assert baseline == _baseline()

    fixture_only = record_random_greedy_baseline(
        "bp35-0a0ad940",
        access_baselines={},
        fixture_baselines=fixtures,
        access_source="missing",
        fixture_source="environment_files",
    )
    assert fixture_only.actions_to_solve_or_timeout == 22
    assert fixture_only.source == "environment_files"

    with pytest.raises(ValueError, match="no baseline actions"):
        record_random_greedy_baseline("zz99-abcd1234", access_baselines={}, fixture_baselines={})


def test_scenario_phase4_052_loads_bp35_verified_pruner_run() -> None:
    """SCENARIO-PHASE4-052: observed and held-out evidence defines verifier actions."""

    run = load_verified_pruner_run(BP35_ARTIFACT)

    assert run.game_id == "bp35-0a0ad940"
    assert run.level_index == 1
    assert run.actions_to_solve == 16
    assert run.observed_transition_count == 5
    assert run.heldout_transition_count == 11
    assert run.validated is True
    assert run.pruned_action_count == 0
    assert exp._relative_source(Path("/tmp/not-under-repo.json")) == "/tmp/not-under-repo.json"


def test_scenario_phase4_052_rejects_bad_pruner_and_loads_incremental_attempt(tmp_path: Path) -> None:
    """SCENARIO-PHASE4-052: malformed verifier evidence cannot become an action run."""

    unsolved = tmp_path / "unsolved.json"
    unsolved.write_text(
        json.dumps({"target_game": "bp35-0a0ad940", "game_solved": False, "real_env_confirmed": True}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="real-env-confirmed solved offline trace"):
        load_verified_pruner_run(unsolved)

    unvalidated = tmp_path / "unvalidated.json"
    unvalidated.write_text(
        json.dumps(
            {
                "target_game": "bp35-0a0ad940",
                "game_solved": True,
                "real_env_confirmed": True,
                "first_solve_at_action": 16,
                "verification_decisions": [{"retained": False, "level_increment": False}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing validated verifier-pruner solve evidence"):
        load_verified_pruner_run(unvalidated)

    attempt_path = tmp_path / "attempt.json"
    attempt_path.write_text(
        json.dumps(
            {
                "honest_verdict": "success: incremental_progress_r11l-495a7899_advanced_to_L5_total14",
                "target_game": "r11l-495a7899",
                "target_level": 5,
                "new_levels_solved_this_task": 1,
                "executed_real_env_actions": 8,
            }
        ),
        encoding="utf-8",
    )
    attempt = exp.load_incremental_attempt(attempt_path)
    assert attempt.new_level_solved is True
    assert attempt.new_levels_solved == 1
    assert attempt.reason == ""


def test_scenario_phase4_052_artifact_reports_ratio_and_honest_no_solve() -> None:
    """SCENARIO-PHASE4-052: complete artifact reports efficiency without solve inflation."""

    artifact = build_artifact(
        _measurement(),
        _attempt(solved=False),
        random_seed=4160,
        duration_s=0.25,
    )

    assert artifact["honest_verdict"] == "complete: verifier_pruner_1.31x_action_efficient_no_new_level_solved_offline"
    assert artifact["action_efficiency_ratio"] == pytest.approx(1.3125)
    assert artifact["baseline_actions"] == 21
    assert artifact["verifier_actions"] == 16
    assert artifact["total_games_solved"] == PRIOR_TOTAL_GAMES_SOLVED
    assert artifact["real_env_confirmed"] is False
    assert artifact["inference_substrate"] == INFERENCE_SUBSTRATE
    assert artifact["requirements"] == REQUIREMENTS
    assert artifact["acceptance_gate_passed"] is True
    assert artifact["submitted_to_leaderboard"] is False
    assert artifact["field_principles"] == REQUIRED_FIELD_PRINCIPLES
    assert artifact_schema_errors(artifact) == []
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact

    success_attempt = build_artifact(
        _measurement(),
        _attempt(solved=True),
        random_seed=4160,
        duration_s=0.25,
    )
    assert success_attempt["honest_verdict"].startswith("success: verifier_pruner_1.31x_action_efficient_new_level_solved_offline")
    assert success_attempt["total_games_solved"] == 14
    assert success_attempt["real_env_confirmed"] is False
    assert artifact_schema_errors(success_attempt) == []

    no_gain = build_artifact(
        _measurement(ratio_override=0.75),
        _attempt(solved=False),
        random_seed=4160,
        duration_s=0.25,
    )
    assert no_gain["honest_verdict"] == "complete: verifier_pruner_0.75x_no_efficiency_gain_no_new_level_solved_offline"
    assert artifact_schema_errors(no_gain) == []


def test_req_phase4_052_blocked_and_schema_guards_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-PHASE4-052: missing fixtures or malformed artifacts cannot pass the gate."""

    blocked = blocked_artifact(random_seed=4160, duration_s=0.0, reason="missing_fixture")
    assert blocked["honest_verdict"] == "blocked_arc_offline_fixtures_missing"
    assert blocked["action_efficiency_ratio"] == 0.0
    assert blocked["baseline_actions"] == 0
    assert blocked["verifier_actions"] == 0
    assert blocked["total_games_solved"] == PRIOR_TOTAL_GAMES_SOLVED
    assert blocked["real_env_confirmed"] is False
    assert blocked["acceptance_gate_passed"] is False
    assert artifact_schema_errors(blocked) == []

    malformed = {
        "honest_verdict": 4160,
        "action_efficiency_ratio": "bad",
        "verifier_actions": True,
        "baseline_actions": "21",
        "total_games_solved": 12,
        "real_env_confirmed": 0,
        "inference_substrate": "wrong",
        "field_principles": {"honest_verdict": "x"},
        "requirements": [],
        "baseline_runs": "bad",
        "verifier_runs": "bad",
        "next_incremental_attempt": "bad",
    }
    errors = artifact_schema_errors(malformed)
    assert "honest_verdict must be a string" in errors
    assert "action_efficiency_ratio must be numeric" in errors
    assert "verifier_actions must be a bare int" in errors
    assert "baseline_actions must be a bare int" in errors
    assert "real_env_confirmed must be a bare bool" in errors
    assert "inference_substrate must equal offline_arc_explore_induce_verify" in errors
    assert "requirements must include REQ-PHASE4-052 and SCENARIO-PHASE4-052" in errors
    assert "baseline_runs must be a list" in errors
    assert "verifier_runs must be a list" in errors
    assert "next_incremental_attempt must be a dict" in errors
    assert any("field_principles missing action_efficiency_ratio" in err for err in errors)
    assert any("total_games_solved must be >= 13" in err for err in errors)
    assert "honest_verdict must be terminal-prefixed" in artifact_schema_errors(
        {
            **blocked,
            "honest_verdict": "maybe",
        }
    )
    assert "field_principles must be a dict" in artifact_schema_errors(
        {
            **blocked,
            "field_principles": [],
        }
    )

    missing = artifact_schema_errors({})
    assert "missing required field honest_verdict" in missing
    assert "honest_verdict must be a string" in missing

    bad_success = build_artifact(_measurement(), _attempt(solved=True), random_seed=4160, duration_s=0.0)
    bad_success["total_games_solved"] = 13
    bad_success["next_incremental_attempt"]["new_level_solved"] = False
    assert any("total_games_solved must increment" in err for err in artifact_schema_errors(bad_success))
    assert any(
        "next_incremental_attempt must record a solved level" in err
        for err in artifact_schema_errors(bad_success)
    )

    bad_complete = build_artifact(_measurement(), _attempt(solved=False), random_seed=4160, duration_s=0.0)
    bad_complete["total_games_solved"] = 14
    bad_complete["real_env_confirmed"] = True
    complete_errors = artifact_schema_errors(bad_complete)
    assert any("total_games_solved must remain" in err for err in complete_errors)
    assert any("real_env_confirmed must be false" in err for err in complete_errors)

    monkeypatch.setattr(exp, "artifact_schema_errors", lambda artifact: ["forced schema error"])
    with pytest.raises(ValueError, match="forced schema error"):
        build_artifact(_measurement(), _attempt(), random_seed=4160, duration_s=0.0)
    with pytest.raises(ValueError, match="forced schema error"):
        blocked_artifact(random_seed=4160, duration_s=0.0, reason="forced")


def test_scenario_phase4_052_runner_writes_blocked_and_complete_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-PHASE4-052: run writes stable offline terminal artifacts."""

    monkeypatch.setattr(exp, "REPO", tmp_path)
    missing = exp.run(write=True)
    assert missing["honest_verdict"] == "blocked_arc_offline_fixtures_missing"
    assert (tmp_path / "results" / "experiment_4160_arc_action_efficiency_harness.json").exists()

    results = tmp_path / "results"
    results.mkdir(exist_ok=True)
    (results / "arc3_win_condition_survey.json").write_text(json.dumps({"per_game_surveys": []}), encoding="utf-8")
    (results / "arc_agi3_access_probe.json").write_text(
        json.dumps({"games": [{"game_id": "bp35-0a0ad940", "baseline_actions": [21]}]}),
        encoding="utf-8",
    )
    no_fixtures = exp.run(write=True)
    assert no_fixtures["honest_verdict"] == "blocked_arc_offline_fixtures_missing"

    (results / "arc_agi3_access_probe.json").write_text("{bad json", encoding="utf-8")
    malformed = exp.run(write=True)
    assert malformed["honest_verdict"] == "blocked_arc_offline_fixtures_missing"
    (results / "arc_agi3_access_probe.json").write_text(
        json.dumps({"games": [{"game_id": "bp35-0a0ad940", "baseline_actions": [21]}]}),
        encoding="utf-8",
    )

    fixture_dir = tmp_path / "environment_files" / "bp35" / "0a0ad940"
    fixture_dir.mkdir(parents=True)
    fixture_dir.joinpath("metadata.json").write_text(
        json.dumps({"game_id": "bp35-0a0ad940", "baseline_actions": [21]}),
        encoding="utf-8",
    )
    fixture_dir.joinpath("bp35.py").write_text("# marker\n", encoding="utf-8")
    no_pruner = exp.run(write=True)
    assert no_pruner["honest_verdict"] == "blocked_arc_offline_fixtures_missing"

    (results / "experiment_4129_fourteenth_game_explore_first.json").write_text(
        BP35_ARTIFACT.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    complete = exp.run(write=True)
    assert complete["honest_verdict"] == "complete: verifier_pruner_1.31x_action_efficient_no_new_level_solved_offline"
    assert complete["action_efficiency_ratio"] == pytest.approx(1.3125)
    written = json.loads((results / "experiment_4160_arc_action_efficiency_harness.json").read_text(encoding="utf-8"))
    assert written == complete

    monkeypatch.setattr(exp, "load_verified_pruner_run", lambda path: (_ for _ in ()).throw(ValueError("bad pruner")))
    blocked = exp.run(write=False)
    assert blocked["honest_verdict"] == "blocked_arc_offline_fixtures_missing"


def test_scenario_phase4_052_cli_prints_terminal_verdict(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-PHASE4-052: CLI wrapper prints the terminal verdict."""

    monkeypatch.setattr(sys, "argv", ["experiment_4160_arc_action_efficiency_harness.py", "--no-write"])
    monkeypatch.setattr(
        exp,
        "run",
        lambda *, write=True: {
            "honest_verdict": "complete: verifier_pruner_1.31x_action_efficient_no_new_level_solved_offline"
        },
    )

    exp.main()

    assert capsys.readouterr().out.strip() == (
        "complete: verifier_pruner_1.31x_action_efficient_no_new_level_solved_offline"
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [str(REPO / "python" / "carnot" / "experiment_4160_arc_action_efficiency_harness.py"), "--no-write"],
    )
    runpy.run_path(str(REPO / "python" / "carnot" / "experiment_4160_arc_action_efficiency_harness.py"), run_name="__main__")
    assert capsys.readouterr().out.strip().startswith("complete:")

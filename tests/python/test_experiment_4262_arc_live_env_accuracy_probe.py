"""Tests for Exp 4262 ARC-AGI-3 scored-only live accuracy probe.

Spec refs: REQ-PHASE4-068, SCENARIO-PHASE4-068.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import carnot.experiment_4262_arc_live_env_accuracy_probe as exp
from carnot.agentic.arc_agi3_live_adapter import MetricMapping


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"
GAME_ID = "lp85-305b61c3"


def _preconditions() -> dict[str, object]:
    return {
        "sdk_importable": True,
        "sdk_version": "0.9.8",
        "network_reachable": True,
        "base_url": "https://three.arcprize.org",
        "error": "",
    }


def _environment() -> dict[str, object]:
    return {
        "game_id": GAME_ID,
        "title": "LP85",
        "tags": ["click"],
        "baseline_actions": [17, 38],
    }


def _score(levels_completed: int, actions: int, score: float) -> dict[str, object]:
    return MetricMapping.Score(
        score=score,
        levels_completed=levels_completed,
        actions=actions,
        level_actions=[actions],
        level_baseline_actions=[17],
        completed=False,
        resets=1,
        guid="fixture-guid",
        message="",
    ).to_json()


def _source_artifact(*, levels_completed: int = 1, actions: int = 5, score: float = 125.0) -> dict[str, object]:
    completes = levels_completed >= 1
    return {
        "experiment": "experiment_4250_arc_live_env_solver_accuracy",
        "honest_verdict": (
            f"success: solver_completes_level_live_{GAME_ID}"
            if completes
            else f"complete: solver_completes_0_levels_live_{GAME_ID}_efficiency_only"
        ),
        "live_env_reachable": True,
        "solver_completes_level": completes,
        "solver_beats_floor": {
            "accuracy": {
                "beats": completes,
                "solver_score": score,
                "floor_score": 0.0,
                "solver_levels_completed": levels_completed,
                "floor_levels_completed": 0,
            },
            "efficiency": {
                "beats": True,
                "solver_actions": actions,
                "floor_actions": 6,
                "solver_actions_vs_baseline_actions": actions / 17,
                "floor_actions_vs_baseline_actions": 6 / 17,
            },
            "overall": True,
        },
        "live_env_metrics": {
            "environment": _environment(),
            "score": score,
            "levels_completed": levels_completed,
            "actions_taken": actions,
            "baseline_actions": 17,
            "actions_vs_baseline_actions": actions / 17,
            "environment_score": _score(levels_completed, actions, score),
            "scorecard_id": "fixture-open-scorecard",
            "score_source": "sdk_get_scorecard_open_scorecard_polled",
            "action_budget": 17,
            "observed_frame_levels_completed": levels_completed,
        },
        "random_greedy_floor": {
            "environment": _environment(),
            "actions_taken": 6,
            "baseline_actions": 17,
            "actions_vs_baseline_actions": 6 / 17,
            "score": 0.0,
            "levels_completed": 0,
            "source_path": "results/experiment_4225_arc_live_env_solver_accuracy.json",
        },
        "solver_trace": [
            {
                "action_index": index + 1,
                "action": {"action_id": 6, "data": {"x": 4, "y": 32}},
                "levels_completed_after": levels_completed if index + 1 == actions else 0,
            }
            for index in range(actions)
        ],
        "solver_policy": "margin_triggered_override_explore_induce_verify_replay",
        "solver_source_artifact": "results/experiment_4190_arc_incremental_progress.json",
        "environment_count": 25,
        "margin_triggered_override": {
            "policy": "margin-triggered override",
            "commit_induced_rule": True,
            "fallback_policy": "execute_verified_policy",
            "learned_margin": 0.2,
            "verifier_margin": 0.2,
            "margin_threshold": 0.1,
            "trigger_source": "2606.04323",
            "references": ["2606.04323", "2509.06870"],
        },
        "no_leaderboard_submission": True,
        "leaderboard_submission_attempted": False,
        "scorecard_closed": False,
        "preconditions_checked": _preconditions(),
        "real_metric_mapping": MetricMapping().to_json(),
        "offline_validation": {"passed": True},
        "online_mode": "official_sdk_online_anonymous_key_open_scorecard_not_closed",
        "inference_substrate": "official_arc_agi3_online_anonymous_key_margin_trigger_solver_accuracy_completion_probe",
        "requirements": ["REQ-PHASE4-066", "SCENARIO-PHASE4-066"],
        "random_seed": 4250,
        "duration_s": 0.1,
        "acceptance_gate_passed": True,
    }


def _blocked_source_artifact() -> dict[str, object]:
    return {
        "experiment": "experiment_4250_arc_live_env_solver_accuracy",
        "honest_verdict": "blocked_arc_live_unreachable",
        "live_env_reachable": False,
        "solver_completes_level": False,
        "solver_beats_floor": {},
        "live_env_metrics": {},
        "random_greedy_floor": {},
        "margin_triggered_override": {},
        "no_leaderboard_submission": True,
        "leaderboard_submission_attempted": False,
        "scorecard_closed": False,
        "preconditions_checked": {**_preconditions(), "sdk_importable": False, "sdk_version": "missing"},
        "offline_validation": {"passed": False, "skipped": True},
        "requirements": ["REQ-PHASE4-066", "SCENARIO-PHASE4-066"],
    }


def _leak_audit() -> dict[str, object]:
    return {
        "experiment": "experiment_4256_arc_oracle_distinct_leak_audit",
        "win_survives_provenance_blind": True,
        "headline_outcome": "arc_provenance_blind_win_survives",
        "reproducibility_checksum": "audit-checksum",
    }


def test_req_phase4_068_spec_declares_exp4262_contract() -> None:
    """REQ-PHASE4-068: OpenSpec declares the Exp 4262 live accuracy contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-068" in spec
    assert "SCENARIO-PHASE4-068" in spec
    assert "experiment_4262_arc_live_env_accuracy_probe.json" in spec
    assert "blocked_arc_live_env_unreachable" in spec
    assert "blocked_operator_only_submission" in spec
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for principle in exp.REQUIRED_FIELD_PRINCIPLES.values():
        assert principle in spec


def test_scenario_phase4_068_reachable_artifact_reports_bare_live_metrics() -> None:
    """SCENARIO-PHASE4-068: reachable artifacts expose bare completion and efficiency fields."""

    artifact = exp.retarget_artifact(_source_artifact(), provenance_audit=_leak_audit())
    repeated = exp.retarget_artifact(_source_artifact(), provenance_audit=_leak_audit())
    changed = exp.retarget_artifact(_source_artifact(actions=4), provenance_audit=_leak_audit())

    assert artifact["experiment"] == "experiment_4262_arc_live_env_accuracy_probe"
    assert artifact["requirements"] == ["REQ-PHASE4-068", "SCENARIO-PHASE4-068"]
    assert artifact["honest_verdict"] == f"success: live_env_accuracy_probe_completed_level_{GAME_ID}"
    assert type(artifact["levels_completed"]) is int
    assert artifact["levels_completed"] == 1
    assert type(artifact["actions_vs_baseline_ratio"]) is float
    assert artifact["actions_vs_baseline_ratio"] == pytest.approx(5 / 17)
    assert artifact["leaderboard_submitted"] is False
    assert artifact["preconditions_checked"]["no_submit_verified"] is True
    assert artifact["preconditions_checked"]["live_scorecard_returned"] is True
    assert artifact["game_probed"] == GAME_ID
    assert artifact["actions_taken"] == 5
    assert artifact["baseline_actions"] == 17
    assert artifact["model_specs"]["routing"]["provenance_blind_enabled"] is True
    assert artifact["model_specs"]["routing"]["margin_triggered"]["commit_induced_rule"] is True
    assert artifact["reproducibility_checksum"] == repeated["reproducibility_checksum"]
    assert artifact["reproducibility_checksum"] != changed["reproducibility_checksum"]
    assert exp.artifact_schema_errors(artifact) == []

    zero = exp.retarget_artifact(_source_artifact(levels_completed=0, score=0.0), provenance_audit=_leak_audit())
    assert zero["honest_verdict"] == f"complete: live_env_accuracy_probe_0_levels_efficiency_only_{GAME_ID}"
    assert zero["levels_completed"] == 0
    assert zero["acceptance_gate_passed"] is True
    assert exp.artifact_schema_errors(zero) == []


def test_scenario_phase4_068_blocked_artifacts_do_not_fabricate_metrics() -> None:
    """SCENARIO-PHASE4-068: blocked live env and operator-only paths stop honestly."""

    blocked = exp.retarget_artifact(_blocked_source_artifact(), provenance_audit={})
    malformed = exp.retarget_artifact({**_source_artifact(), "live_env_metrics": "bad"}, provenance_audit={})

    assert blocked["honest_verdict"] == "blocked_arc_live_env_unreachable"
    assert blocked["levels_completed"] == 0
    assert blocked["actions_vs_baseline_ratio"] == 0.0
    assert blocked["leaderboard_submitted"] is False
    assert blocked["preconditions_checked"]["live_scorecard_returned"] is False
    assert blocked["environment_score"] == {}
    assert exp.artifact_schema_errors(blocked) == []
    assert malformed["honest_verdict"] == "blocked_arc_live_env_unreachable"
    assert malformed["preconditions_checked"]["live_scorecard_returned"] is False

    operator_block = exp.blocked_operator_only_submission_artifact(
        no_submit_check={"no_submit_verified": False, "forbidden_markers": {"probe.py": ["close_scorecard("]}},
        duration_s=0.2,
    )
    assert operator_block["honest_verdict"] == "blocked_operator_only_submission"
    assert operator_block["leaderboard_submitted"] is False
    assert operator_block["preconditions_checked"]["no_submit_verified"] is False
    assert exp.artifact_schema_errors(operator_block) == []


def test_scenario_phase4_068_schema_and_no_submit_guard_edges(tmp_path: Path) -> None:
    """SCENARIO-PHASE4-068: schema rejects non-bare fields and submission-capable code paths."""

    clean = tmp_path / "clean.py"
    dirty = tmp_path / "dirty.py"
    audit = tmp_path / "audit.json"
    clean.write_text("def read_open_scorecard():\n    return None\n", encoding="utf-8")
    dirty.write_text("def bad(scorecard):\n    close_scorecard(scorecard)\n", encoding="utf-8")
    audit.write_text(json.dumps({"win_survives_provenance_blind": True}), encoding="utf-8")

    assert exp.verify_no_submit_path([clean])["no_submit_verified"] is True
    dirty_check = exp.verify_no_submit_path([clean, dirty])
    assert dirty_check["no_submit_verified"] is False
    assert str(dirty) in dirty_check["forbidden_markers"]
    assert exp.load_provenance_audit(tmp_path / "missing.json") == {}
    assert exp.load_provenance_audit(audit) == {"win_survives_provenance_blind": True}

    no_efficiency = _source_artifact(levels_completed=0, score=0.0)
    no_efficiency["solver_beats_floor"]["efficiency"]["beats"] = False
    assert exp.retarget_artifact(no_efficiency, provenance_audit={})[
        "honest_verdict"
    ] == f"complete: live_env_accuracy_probe_0_levels_{GAME_ID}"

    bad = {
        **exp.retarget_artifact(_source_artifact(), provenance_audit={}),
        "honest_verdict": "maybe",
        "levels_completed": True,
        "actions_vs_baseline_ratio": "0.2",
        "leaderboard_submitted": True,
        "leaderboard_submission_attempted": True,
        "scorecard_closed": True,
        "preconditions_checked": {"sdk_importable": "yes"},
        "random_seed": "4262",
        "reproducibility_checksum": "",
        "model_specs": [],
        "field_principles": [],
        "requirements": [],
    }
    errors = exp.artifact_schema_errors(bad)
    assert "honest_verdict must be terminal-prefixed" in errors
    assert "levels_completed must be a bare int" in errors
    assert "actions_vs_baseline_ratio must be a bare float" in errors
    assert "leaderboard_submitted must be false" in errors
    assert "leaderboard_submission_attempted must be false" in errors
    assert "scorecard_closed must be false" in errors
    assert "preconditions_checked.sdk_importable must be a bare bool" in errors
    assert "random_seed must be a bare int" in errors
    assert "reproducibility_checksum must be a non-empty string" in errors
    assert "model_specs must be a dict" in errors
    assert "requirements must include REQ-PHASE4-068 and SCENARIO-PHASE4-068" in errors
    assert "field_principles must be a dict" in errors

    missing = exp.artifact_schema_errors({})
    assert "missing required field honest_verdict" in missing
    assert "honest_verdict must be a string" in missing
    assert "preconditions_checked must be a dict" in missing

    good = exp.retarget_artifact(_source_artifact(), provenance_audit={})
    missing_principle = {**good, "field_principles": {}}
    assert "field_principles missing honest_verdict" in exp.artifact_schema_errors(missing_principle)
    metric_mismatch = {**good, "levels_completed": 0}
    assert "levels_completed must equal environment_score.levels_completed" in exp.artifact_schema_errors(metric_mismatch)
    ratio_mismatch = {**good, "actions_vs_baseline_ratio": 0.1}
    assert "actions_vs_baseline_ratio must equal actions_taken / baseline_actions" in exp.artifact_schema_errors(
        ratio_mismatch
    )


def test_scenario_phase4_068_retarget_defensive_schema_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-PHASE4-068: retargeting fails closed if its own schema guard fails."""

    monkeypatch.setattr(exp, "artifact_schema_errors", lambda artifact: ["forced schema error"])

    with pytest.raises(ValueError, match="forced schema error"):
        exp.retarget_artifact(_source_artifact(), provenance_audit={})


def test_scenario_phase4_068_run_checks_no_submit_before_live_delegate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-PHASE4-068: run verifies scored-only mode before delegating to Exp 4250."""

    events: list[str] = []
    calls: list[dict[str, object]] = []

    def fake_no_submit_check() -> dict[str, object]:
        events.append("no-submit")
        return {"no_submit_verified": True, "forbidden_markers": {}, "checked_files": []}

    def fake_exp4250_run(**kwargs: object) -> dict[str, object]:
        assert events == ["no-submit"]
        calls.append(kwargs)
        return _source_artifact()

    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "verify_no_submit_path", fake_no_submit_check)
    monkeypatch.setattr(exp.exp4250, "run", fake_exp4250_run)
    monkeypatch.setattr(exp, "load_provenance_audit", lambda: _leak_audit())

    artifact = exp.run(write=True, action_budget=17, base_url="https://fixture.example")

    assert calls == [
        {
            "write": False,
            "action_budget": 17,
            "base_url": "https://fixture.example",
            "margin_config": exp.DEFAULT_MARGIN_CONFIG,
        }
    ]
    written = json.loads((tmp_path / "results" / exp.RESULT_NAME).read_text(encoding="utf-8"))
    assert written == artifact

    monkeypatch.setattr(
        exp,
        "verify_no_submit_path",
        lambda: {"no_submit_verified": False, "forbidden_markers": {"x": ["submit_scorecard("]}},
    )
    calls.clear()
    blocked = exp.run(write=True)
    assert blocked["honest_verdict"] == "blocked_operator_only_submission"
    assert calls == []
    written_block = json.loads((tmp_path / "results" / exp.RESULT_NAME).read_text(encoding="utf-8"))
    assert written_block == blocked

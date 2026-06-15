"""Tests for Exp 4250 ARC-AGI-3 live completion-targeting accuracy rerun.

Spec refs: REQ-PHASE4-066, SCENARIO-PHASE4-066.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import carnot.experiment_4250_arc_live_env_solver_accuracy as exp
from carnot.agentic.arc_agi3_live_adapter import MetricMapping
from carnot.experiment_4237_arc_live_env_solver_accuracy import (
    LP85_GAME_ID,
    MarginTriggeredOverrideConfig,
)


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"


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
        "game_id": LP85_GAME_ID,
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


def _upstream_artifact(*, levels_completed: int = 1, score: float = 125.0) -> dict[str, object]:
    actions = 5
    solver_completes = levels_completed >= 1
    return {
        "experiment": "experiment_4237_arc_live_env_solver_accuracy",
        "title": "arc3_live_env_solver_accuracy_margin_trigger",
        "honest_verdict": (
            f"success: solver_completes_level_live_{LP85_GAME_ID}"
            if solver_completes
            else f"complete: solver_completes_0_levels_live_{LP85_GAME_ID}_efficiency_only"
        ),
        "live_env_reachable": True,
        "solver_completes_level": solver_completes,
        "solver_beats_floor": {
            "accuracy": {
                "beats": solver_completes,
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
        "solver_trace": [],
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
        "inference_substrate": "official_arc_agi3_online_anonymous_key_margin_trigger_solver_accuracy_probe",
        "field_principles": {},
        "requirements": ["REQ-PHASE4-064", "SCENARIO-PHASE4-064"],
        "random_seed": 4237,
        "duration_s": 0.1,
        "acceptance_gate_passed": True,
    }


def _blocked_upstream_artifact() -> dict[str, object]:
    artifact = _upstream_artifact(levels_completed=0, score=0.0)
    artifact.update(
        {
            "honest_verdict": "blocked_arc_live_unreachable",
            "live_env_reachable": False,
            "solver_beats_floor": {},
            "live_env_metrics": {},
            "random_greedy_floor": {},
            "margin_triggered_override": {},
            "offline_validation": {"passed": False, "skipped": True},
            "preconditions_checked": {
                **_preconditions(),
                "sdk_importable": False,
                "sdk_version": "missing",
            },
        }
    )
    return artifact


def test_req_phase4_066_spec_declares_exp4250_contract() -> None:
    """REQ-PHASE4-066: OpenSpec declares the Exp 4250 live rerun contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-066" in spec
    assert "SCENARIO-PHASE4-066" in spec
    assert "experiment_4250_arc_live_env_solver_accuracy.json" in spec
    assert "margin-triggered override keyed to 2606.04323" in spec
    assert "open-scorecard `EnvironmentScore` without closing or submitting" in spec
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for principle in exp.REQUIRED_FIELD_PRINCIPLES.values():
        assert principle in spec


def test_scenario_phase4_066_retargeted_artifact_preserves_official_metrics() -> None:
    """SCENARIO-PHASE4-066: 4250 reports bare bools from EnvironmentScore evidence."""

    artifact = exp.retarget_artifact(_upstream_artifact())

    assert artifact["experiment"] == "experiment_4250_arc_live_env_solver_accuracy"
    assert artifact["requirements"] == ["REQ-PHASE4-066", "SCENARIO-PHASE4-066"]
    assert artifact["random_seed"] == 4250
    assert artifact["solver_completes_level"] is True
    assert artifact["live_env_metrics"]["levels_completed"] == 1
    assert artifact["live_env_metrics"]["action_budget"] == 17
    assert artifact["scorecard_closed"] is False
    assert artifact["no_leaderboard_submission"] is True
    assert artifact["field_principles"]["solver_completes_level"].startswith("BARE bool")
    assert artifact["source_experiment_artifact"] == "results/experiment_4237_arc_live_env_solver_accuracy.json"
    assert exp.artifact_schema_errors(artifact) == []

    zero = exp.retarget_artifact(_upstream_artifact(levels_completed=0, score=0.0))
    assert zero["solver_completes_level"] is False
    assert zero["honest_verdict"].startswith("complete:")
    assert zero["solver_beats_floor"]["efficiency"]["beats"] is True
    assert exp.artifact_schema_errors(zero) == []


def test_scenario_phase4_066_blocked_and_schema_edges() -> None:
    """SCENARIO-PHASE4-066: blocked verdicts do not fabricate live metrics."""

    blocked = exp.retarget_artifact(_blocked_upstream_artifact())

    assert blocked["honest_verdict"] == "blocked_arc_live_unreachable"
    assert blocked["solver_completes_level"] is False
    assert blocked["solver_beats_floor"] == {}
    assert blocked["live_env_metrics"] == {}
    assert blocked["scorecard_closed"] is False
    assert exp.artifact_schema_errors(blocked) == []

    bad = {
        **exp.retarget_artifact(_upstream_artifact()),
        "honest_verdict": "maybe",
        "solver_completes_level": "true",
        "no_leaderboard_submission": False,
        "leaderboard_submission_attempted": True,
        "scorecard_closed": True,
        "requirements": [],
        "preconditions_checked": {"sdk_importable": "yes", "network_reachable": "yes"},
        "live_env_metrics": {"score": "bad", "levels_completed": "1", "actions_taken": "5", "baseline_actions": "17", "action_budget": 1},
        "solver_beats_floor": [],
        "margin_triggered_override": {"commit_induced_rule": "yes"},
        "offline_validation": {"passed": False},
        "real_metric_mapping": {},
        "field_principles": [],
    }
    errors = exp.artifact_schema_errors(bad)
    assert "honest_verdict must be terminal-prefixed" in errors
    assert "solver_completes_level must be a bare bool" in errors
    assert "no_leaderboard_submission must be true" in errors
    assert "leaderboard_submission_attempted must be false" in errors
    assert "scorecard_closed must be false" in errors
    assert "requirements must include REQ-PHASE4-066 and SCENARIO-PHASE4-066" in errors
    assert "preconditions_checked missing sdk_version" in errors
    assert "preconditions_checked.sdk_importable must be a bare bool" in errors
    assert "solver_beats_floor must be a dict" in errors
    assert "live_env_metrics.score must be numeric" in errors
    assert "live_env_metrics.levels_completed must be a bare int" in errors
    assert "margin_triggered_override.commit_induced_rule must be a bare bool" in errors
    assert "reachable artifacts require passed offline_validation" in errors
    assert "real_metric_mapping must equal the ARC live EnvironmentScore mapping" in errors
    assert "field_principles must be a dict" in errors

    with pytest.raises(ValueError, match="no_leaderboard_submission must be true"):
        exp.retarget_artifact({**_upstream_artifact(), "no_leaderboard_submission": False})


def test_scenario_phase4_066_run_writes_transformed_exp4237_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-PHASE4-066: CLI run delegates to 4237 but writes the 4250 artifact."""

    calls: list[dict[str, object]] = []

    def fake_exp4237_run(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return _upstream_artifact()

    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp.exp4237, "run", fake_exp4237_run)

    margin_config = MarginTriggeredOverrideConfig(learned_margin=0.3, verifier_margin=0.4)
    artifact = exp.run(
        write=True,
        action_budget=17,
        base_url="https://fixture.example",
        margin_config=margin_config,
    )

    assert calls == [
        {
            "write": False,
            "action_budget": 17,
            "base_url": "https://fixture.example",
            "margin_config": margin_config,
        }
    ]
    assert artifact["experiment"] == "experiment_4250_arc_live_env_solver_accuracy"
    written = json.loads((tmp_path / "results" / exp.RESULT_NAME).read_text(encoding="utf-8"))
    assert written == artifact

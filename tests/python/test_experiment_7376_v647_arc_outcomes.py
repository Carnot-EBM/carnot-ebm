"""Tests for REQ-ARC-WMTE-7376 and its four named scenarios."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7376_v647_arc_outcomes as exp


pytestmark = pytest.mark.usefixtures("tmp_path")


def _registry() -> dict:
    return {
        "games": [
            {
                "game": game,
                "levels_reproduced": index + 1,
                "reproducibility": "reproduced",
                "mechanic_class": f"family-{index}",
            }
            for index, game in enumerate((*exp.TARGET_GAMES, "other"))
        ]
    }


def _episode(
    episode_id: str = "bp35:seed-11",
    *,
    game: str = "bp35",
    seed: int = 11,
    redirects: list[dict] | None = None,
    disposition: str = "complete",
    censored: bool = False,
) -> dict:
    if redirects is None:
        redirects = [
            {
                "arm": "drop_goal_bias",
                "action_index": 120,
                "level": 0,
                "stretch_level": 0,
                "diagnosis": "observed stalled goal bias",
                "resolved_by_levelup": False,
                "actions_to_levelup": None,
                "co_credited_count": None,
            }
        ]
    return {
        "episode_id": episode_id,
        "game": game,
        "seed": seed,
        "arm": "curated_supervisor",
        "disposition": disposition,
        "censored": censored,
        "action_count": 128,
        "action_limit": 128,
        "generation_calls_attempted": 1,
        "generation_calls_completed": 1,
        "generated_tokens": 80,
        "levels": 0,
        "error": None,
        "adapter_disabled": True,
        "banked_solution_disabled": True,
        "hand_solver_disabled": True,
        "saved_engine_disabled": True,
        "replay_route_disabled": True,
        "fresh_store": True,
        "supervisor_mode": "applied",
        "trajectory_supervisor": {
            "enabled": True,
            "window": 120,
            "arms_enabled": list(exp.CURATED_ARMS),
            "redirects": redirects,
            "unredirected_windows": [],
            "stagnations_unredirected": 0,
        },
        "raw_request_manifest": [
            {
                "request_sha256": "sha256:" + "1" * 64,
                "response_sha256": "sha256:" + "2" * 64,
                "transport_completed": True,
                "completion_tokens": 80,
            }
        ],
        "policy_consumption_rows": [],
        "factory_receipt": {
            "factory": "make_carnot_agent",
            "policy_class": "E3AgentPolicy",
            "adapter_disabled": True,
        },
    }


def _passing_receipts() -> list[dict]:
    return [
        {
            "name": name,
            "exit_code": 0,
            "duration_s": 0.01,
            "passed": True,
            "timed_out": False,
            "command_argv": [name],
            "scope": "test",
            "log_sha256": "sha256:" + "3" * 64,
        }
        for name in (*exp.REQUIRED_VALIDATION_NAMES, *exp.REQUIRED_E2E_NAMES, *exp.TERMINAL_NAMES)
    ]


def test_scenario_arc_wmte_7376_frozen_live_panel_and_schedule() -> None:
    """SCENARIO-ARC-WMTE-7376-FROZEN-LIVE-PANEL."""

    panel = exp.freeze_panel(_registry(), adaptered_games=set(exp.TARGET_GAMES))
    schedule = exp.build_schedule(panel["games"])

    assert panel["passed"] is True
    assert panel["games"] == list(exp.TARGET_GAMES)
    assert panel["outcomes_seen_before_freeze"] is False
    assert len(schedule) == 6
    assert {row["seed"] for row in schedule} == set(exp.EPISODE_SEEDS)
    assert all(row["action_limit"] == 128 for row in schedule)
    assert all(row["completion_limit"] == 2 for row in schedule)
    assert all(row["max_new_tokens_per_call"] == 256 for row in schedule)
    assert all(row["adapter_disabled"] and row["saved_engine_disabled"] for row in schedule)


def test_frozen_panel_rejects_missing_registry_or_adapter() -> None:
    """REQ-ARC-WMTE-7376 rejects an incomplete registry precheck."""

    registry = _registry()
    registry["games"] = registry["games"][:-2]
    panel = exp.freeze_panel(registry, adaptered_games={exp.TARGET_GAMES[0]})

    assert panel["passed"] is False
    assert len(panel["failures"]) == 2


def test_live_environment_uses_unchanged_curated_applied_policy(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7376 uses the existing applied arm policy without a selector."""

    env = exp.session_environment(
        {
            "CARNOT_ARC_SUPERVISOR_TOOL_ARM": "1",
            "CARNOT_ARC_SUPERVISOR_ORDER": "invented",
        },
        arm="curated_supervisor",
        episode_dir=tmp_path / "bp35__seed-7376001",
        gpu_index=1,
        port=8123,
    )

    assert env["CARNOT_FORCE_LIVE"] == "1"
    assert env["CARNOT_ARC_TRAJECTORY_SUPERVISOR"] == "1"
    assert env["CARNOT_ARC_INDUCE_MAX_TOKENS"] == "256"
    assert env["CARNOT_ARC_RANDOM_SEED"] == "7376001"
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert "CARNOT_ARC_SUPERVISOR_TOOL_ARM" not in env
    assert "CARNOT_ARC_SUPERVISOR_ORDER" not in env

    with pytest.raises(ValueError, match="unknown live arm"):
        exp.session_environment({}, arm="tuned", episode_dir=tmp_path, gpu_index=0, port=1)


def test_scenario_arc_wmte_7376_redirect_outcomes_and_empty_ledger() -> None:
    """SCENARIO-ARC-WMTE-7376-REDIRECT-OUTCOMES."""

    resolved = _episode(
        redirects=[
            {
                "arm": "drop_goal_bias",
                "action_index": 120,
                "level": 0,
                "stretch_level": 0,
                "diagnosis": "stalled",
                "resolved_by_levelup": True,
                "actions_to_levelup": 4,
                "co_credited_count": 1,
            }
        ]
    )
    empty = _episode(episode_id="cn04:seed-12", game="cn04", seed=12, redirects=[])
    rows = exp.extract_new_outcomes([resolved, empty])

    assert len(rows) == 1
    assert rows[0]["trigger_action"] == 120
    assert rows[0]["later_action_range"] == [121, 124]
    assert rows[0]["resolved_by_levelup"] is True
    assert rows[0]["outcome_observed"] is True
    assert rows[0]["causal_interpretation"] == "descriptive_association_only"
    assert empty["trajectory_supervisor"]["redirects"] == []


def test_censored_episode_does_not_invent_redirect_outcome() -> None:
    """REQ-ARC-WMTE-7376 censors an unresolved redirect when its episode is censored."""

    rows = exp.extract_new_outcomes([_episode(disposition="censored_timeout", censored=True)])

    assert rows[0]["resolved_by_levelup"] is None
    assert rows[0]["outcome_observed"] is False
    assert rows[0]["censoring_reason"] == "episode_censored_before_terminal_outcome"


def test_scenario_arc_wmte_7376_support_dedupes_and_computes_loo() -> None:
    """SCENARIO-ARC-WMTE-7376-SUPPORT-REDUCTION."""

    rows = []
    for game in ("g1", "g2", "g3", "g4"):
        for arm in exp.CURATED_ARMS:
            for index in range(10):
                rows.append(
                    {
                        "runtime_event_id": f"{game}:{arm}:{index}",
                        "game": game,
                        "selected_arm": arm,
                        "mode": "applied",
                        "outcome_observed": True,
                        "resolved_by_levelup": index % 2 == 0,
                        "censored": False,
                    }
                )
    rows.append(deepcopy(rows[0]))
    rows.append(
        {
            **deepcopy(rows[1]),
            "runtime_event_id": "shadow-row",
            "mode": "shadow",
        }
    )

    reduced = exp.reduce_support(rows)

    assert reduced["duplicate_event_count"] == 1
    assert reduced["ineligible_event_count"] == 1
    assert reduced["supervisor_support_ready_score"] == 1
    assert all(row["passed"] for row in reduced["leave_one_game_out_rows"])
    assert all(value["supported_decision_count"] == 40 for value in reduced["per_arm"].values())


def test_support_below_threshold_stays_null() -> None:
    """REQ-ARC-WMTE-7376 preserves the ten-decisions and three-games floor."""

    rows = exp.extract_new_outcomes([_episode()])
    reduced = exp.reduce_support(rows)

    assert reduced["supervisor_support_ready_score"] == 0
    assert reduced["per_arm"]["drop_goal_bias"]["supported_decision_count"] == 1
    assert reduced["per_arm"]["allow_reinduction"]["supported_decision_count"] == 0


def test_historical_support_rows_keep_applied_observations_only() -> None:
    """REQ-ARC-WMTE-7376 joins labeled history without counting it as current inference."""

    ledger = {
        "entries": {
            "receipt-a": {
                "game": "g1",
                "seed": 1,
                "mode": "applied",
                "redirects": [
                    {
                        "arm": "drop_goal_bias",
                        "action_index": 120,
                        "resolved_by_levelup": False,
                        "actions_to_levelup": None,
                    }
                ],
            }
        },
        "controls": {
            "shadow-a": {
                "game": "g2",
                "seed": 2,
                "mode": "shadow",
                "redirects": [],
            }
        },
    }

    rows = exp.historical_support_rows(ledger)

    assert len(rows) == 1
    assert rows[0]["runtime_event_id"] == "historical:receipt-a:0"
    assert rows[0]["historical_only"] is True
    assert rows[0]["outcome_observed"] is True


def test_episode_completion_accepts_empty_outcome_ledger() -> None:
    """REQ-ARC-WMTE-7376 makes a complete six-episode null valid."""

    schedule = exp.build_schedule(exp.TARGET_GAMES)
    episodes = [
        _episode(
            row["episode_id"],
            game=row["game"],
            seed=row["seed"],
            redirects=[],
        )
        for row in schedule
    ]
    reduced = exp.reduce_episode_accounting(schedule, episodes)

    assert reduced["arc_outcome_capture_complete_score"] == 1
    assert reduced["completed_units"] == 6
    assert reduced["redirect_event_count"] == 0

    episodes[0]["adapter_disabled"] = False
    assert (
        exp.reduce_episode_accounting(schedule, episodes)["arc_outcome_capture_complete_score"] == 0
    )


def test_validation_plan_is_exp7358_scoped_and_has_private_parents(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7376 builds the affected command plan through Exp7358."""

    commands = exp.build_validation_plan(exp.REPO_ROOT, tmp_path / "private")

    assert [command.name for command in commands] == list(exp.REQUIRED_VALIDATION_NAMES)
    assert exp.validate_validation_plan(exp.REPO_ROOT, commands) == []
    assert not any("tests/python" == argument for command in commands for argument in command.argv)
    assert (tmp_path / "private" / "basetemp").is_dir()
    assert (tmp_path / "private" / "coverage").is_dir()


def test_scenario_arc_wmte_7376_terminal_accounting_complete_null() -> None:
    """SCENARIO-ARC-WMTE-7376-TERMINAL-ACCOUNTING."""

    schedule = exp.build_schedule(exp.TARGET_GAMES)
    episodes = [
        _episode(row["episode_id"], game=row["game"], seed=row["seed"], redirects=[])
        for row in schedule
    ]
    artifact = exp.build_artifact(
        preconditions=[exp.gate("inputs", "repo", "available", True, True)],
        source_hashes={"source": "sha256:" + "4" * 64},
        selection={"passed": True, "games": list(exp.TARGET_GAMES)},
        schedule=schedule,
        episodes=episodes,
        historical_rows=[],
        runtime_receipt={"task_linked_cuda_execution": True},
        model_specs=[{"hf_id": exp.MODEL_ID, "quantization": "Q4_K_M"}],
        invocation_counts={
            **exp.ZERO_INVOCATION_COUNTS,
            "model_loads_attempted": 1,
            "model_loads_completed": 1,
            "generation_calls_attempted": 6,
            "generation_calls_completed": 6,
        },
        validation_receipts=_passing_receipts(),
        repository_health={"status": "healthy", "affects_required_checks": False},
        phase_spans=[{"phase": "generate", "duration_s": 10.0}],
        started_at_utc="2026-09-17T10:00:00+00:00",
        completed_at_utc="2026-09-17T10:01:00+00:00",
        duration_s=60.0,
    )

    assert artifact["status"] == "complete_null_insufficient_supervisor_support"
    assert artifact["verdict_class"] == "null"
    assert artifact["arc_outcome_capture_complete_score"] == 1
    assert artifact["supervisor_support_ready_score"] == 0
    assert artifact["promotion_score"] == 0
    assert artifact["inference_substrate_class"] == "model_bounded_generation"
    assert exp.validate_artifact(artifact) == []
    assert set(artifact["field_principles"]) == set(artifact)


def test_terminal_validation_failure_disqualifies_science() -> None:
    """REQ-ARC-WMTE-7376 keeps readiness zero after a required failure."""

    schedule = exp.build_schedule(exp.TARGET_GAMES)
    episodes = [
        _episode(row["episode_id"], game=row["game"], seed=row["seed"], redirects=[])
        for row in schedule
    ]
    receipts = _passing_receipts()
    receipts[0]["passed"] = False
    receipts[0]["exit_code"] = 1
    artifact = exp.build_artifact(
        preconditions=[exp.gate("inputs", "repo", "available", True, True)],
        source_hashes={},
        selection={"passed": True, "games": list(exp.TARGET_GAMES)},
        schedule=schedule,
        episodes=episodes,
        historical_rows=[],
        runtime_receipt={"task_linked_cuda_execution": True},
        model_specs=[{"hf_id": exp.MODEL_ID, "quantization": "Q4_K_M"}],
        invocation_counts={
            **exp.ZERO_INVOCATION_COUNTS,
            "model_loads_attempted": 1,
            "model_loads_completed": 1,
            "generation_calls_attempted": 6,
            "generation_calls_completed": 6,
        },
        validation_receipts=receipts,
        repository_health={"status": "healthy"},
        phase_spans=[],
        started_at_utc="2026-09-17T10:00:00+00:00",
        completed_at_utc="2026-09-17T10:01:00+00:00",
        duration_s=60.0,
    )

    assert artifact["verdict_class"] == "disqualified"
    assert artifact["flagged_adversarial"] is False
    assert artifact["arc_outcome_capture_complete_score"] == 0
    assert artifact["promotion_score"] == 0


def test_blocked_artifact_names_exact_failed_gate() -> None:
    """REQ-ARC-WMTE-7376 emits terminal blocked_* data for unavailable inputs."""

    failure = exp.gate("model", "cache", "path", "readable", "missing")
    artifact = exp.build_blocked_artifact(
        preconditions=[failure],
        source_hashes={},
        started_at_utc="2026-09-17T10:00:00+00:00",
        completed_at_utc="2026-09-17T10:00:01+00:00",
        duration_s=1.0,
    )

    assert artifact["status"].startswith("blocked_")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"]["first_failure"]["artifact_field"] == "path"
    assert artifact["model_invoked"] is False
    assert exp.validate_artifact(artifact) == []


def test_atomic_json_and_independent_reduce(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7376 independently reloads raw rows before publication."""

    path = tmp_path / "raw.json"
    schedule = exp.build_schedule(exp.TARGET_GAMES)
    episodes = [
        _episode(row["episode_id"], game=row["game"], seed=row["seed"], redirects=[])
        for row in schedule
    ]
    exp.atomic_json(path, {"schedule": schedule, "episodes": episodes, "historical_rows": []})
    reduced = exp.independent_reduce(path)

    assert json.loads(path.read_text())["schedule"] == schedule
    assert reduced["episode_accounting"]["arc_outcome_capture_complete_score"] == 1
    assert reduced["support"]["supervisor_support_ready_score"] == 0


def test_strict_reducer_and_fail_closed_boundaries(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-ARC-WMTE-7376 independently rejects malformed evidence and reports progress."""

    schedule = exp.build_schedule(exp.TARGET_GAMES)
    episodes = []
    for planned in schedule:
        episode = _episode(
            planned["episode_id"], game=planned["game"], seed=planned["seed"], redirects=[]
        )
        episode.update(
            {
                "completion_limit": exp.MODEL_CALL_LIMIT,
                "generated_token_limit": exp.GENERATED_TOKEN_LIMIT,
                "max_new_tokens_per_call": exp.MAX_NEW_TOKENS,
                "starting_policy": {"sha256": "sha256:start"},
                "ending_policy": {"sha256": "sha256:end"},
                "source_engine_provenance": {"sha256": "sha256:engine"},
            }
        )
        episode["trajectory_supervisor"]["mode"] = "applied"
        episodes.append(episode)

    reduced = exp.reduce_raw_panel({"schedule": schedule, "episodes": episodes}, historical_rows=[])
    assert reduced["arc_outcome_capture_complete_score"] == 1

    malformed = exp.reduce_raw_panel(
        {"schedule": schedule, "episodes": episodes[:-1]}, historical_rows=[]
    )
    assert malformed["arc_outcome_capture_complete_score"] == 0
    inauthentic = deepcopy(episodes)
    inauthentic[0]["saved_engine_disabled"] = False
    assert (
        exp.reduce_raw_panel({"schedule": schedule, "episodes": inauthentic}, historical_rows=[])[
            "arc_outcome_capture_complete_score"
        ]
        == 0
    )
    assert (
        exp.reduce_episode_accounting(schedule, episodes[:-1])["arc_outcome_capture_complete_score"]
        == 0
    )

    assert exp.redirect_rows_for_episode({"trajectory_supervisor": None}) == []
    shadow = deepcopy(episodes[0])
    shadow["trajectory_supervisor"]["mode"] = "shadow"
    assert exp.redirect_rows_for_episode(shadow) == []
    invalid_arm = deepcopy(episodes[0])
    invalid_arm["trajectory_supervisor"]["redirects"] = [
        {"arm": "invented", "action_index": 1, "resolved_by_levelup": True}
    ]
    assert exp.redirect_rows_for_episode(invalid_arm) == []
    assert exp.historical_support_rows({}) == []
    assert (
        exp.historical_support_rows(
            {
                "entries": {
                    "shadow": {"mode": "shadow"},
                    "bad": {"mode": "applied", "redirects": [None, {"arm": "invented"}]},
                }
            }
        )
        == []
    )
    assert exp.join_event_rows([{"runtime_event_id": None}], [])["rows"] == []

    source = tmp_path / "source.txt"
    source.write_text("evidence", encoding="utf-8")
    assert exp.sha256_file(source).startswith("sha256:")
    assert exp.load_json(tmp_path / "missing.json") is None
    malformed_json = tmp_path / "malformed.json"
    malformed_json.write_text("{", encoding="utf-8")
    assert exp.load_json(malformed_json) is None
    exp.progress(0.0, "test", "boundary", unit=1)
    assert "phase=test event=boundary" in capsys.readouterr().out


def test_artifact_validator_error_paths_and_argument_parser() -> None:
    """REQ-ARC-WMTE-7376 keeps terminal validation and CLI classification fail closed."""

    assert exp.validate_artifact(None) == ["artifact_not_object"]
    blocked = exp.build_blocked_artifact(
        preconditions=[exp.gate("model", "cache", "path", "readable", "missing")],
        source_hashes={},
        started_at_utc="2026-09-17T10:00:00+00:00",
        completed_at_utc="2026-09-17T10:00:01+00:00",
        duration_s=1.0,
    )
    broken_blocked = deepcopy(blocked)
    broken_blocked.update(
        {
            "model_invoked": True,
            "inference_substrate_class": "wrong",
            "gate_check_summary": {"first_failure": None},
            "supervisor_support_ready_score": 1,
            "scientific_value_score": 1,
        }
    )
    assert {
        "blocked_invocations",
        "blocked_substrate",
        "blocked_gate_check_summary",
        "unsafe_readiness",
    }.issubset(exp.validate_artifact(broken_blocked))

    schedule = exp.build_schedule(exp.TARGET_GAMES)
    episodes = [
        _episode(row["episode_id"], game=row["game"], seed=row["seed"], redirects=[])
        for row in schedule
    ]
    complete = exp.build_artifact(
        preconditions=[exp.gate("inputs", "repo", "available", True, True)],
        source_hashes={},
        selection={"passed": True, "games": list(exp.TARGET_GAMES)},
        schedule=schedule,
        episodes=episodes,
        historical_rows=[],
        runtime_receipt={"task_linked_cuda_execution": True},
        model_specs=[{"hf_id": exp.MODEL_ID, "quantization": "Q4_K_M"}],
        invocation_counts={
            **exp.ZERO_INVOCATION_COUNTS,
            "model_loads_attempted": 1,
            "model_loads_completed": 1,
            "generation_calls_attempted": 6,
            "generation_calls_completed": 6,
        },
        validation_receipts=_passing_receipts(),
        repository_health={"status": "healthy"},
        phase_spans=[],
        started_at_utc="2026-09-17T10:00:00+00:00",
        completed_at_utc="2026-09-17T10:01:00+00:00",
        duration_s=60.0,
    )
    broken = deepcopy(complete)
    broken.update(
        {
            "schema": "wrong",
            "run_date": "wrong",
            "verdict_class": "wrong",
            "honest_verdict": "wrong",
            "MODEL_SPECS": [],
            "model_invoked": False,
            "inference_substrate_class": "wrong",
            "duration_s": 0,
            "validation_receipts": [],
            "promotion_score": 1,
            "production_defaults_changed": True,
            "field_principles": {},
            "reproducibility_checksum": "wrong",
        }
    )
    errors = exp.validate_artifact(broken)
    assert {
        "identity",
        "run_date",
        "verdict_class",
        "honest_verdict",
        "MODEL_SPECS",
        "model_invoked",
        "inference_substrate_class",
        "duration_s",
        "validation_receipts",
        "promotion_score",
        "production_defaults_changed",
        "field_principles",
        "reproducibility_checksum",
    }.issubset(errors)
    args = exp.parse_args(["--date", exp.RUN_DATE, "--role", "live-session"])
    assert args.role == "live-session"

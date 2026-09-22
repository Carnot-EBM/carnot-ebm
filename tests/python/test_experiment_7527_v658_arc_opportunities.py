"""Tests for the bounded live ARC opportunity measurement.

Spec: REQ-ARC-7527 and SCENARIO-ARC-7527-*.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from carnot import experiment_7527_v658_arc_opportunities as exp


def _upstream() -> dict:
    """Return the minimum authenticated Exp7526 shape used by pure tests."""

    return {
        "schema": "carnot.exp7526.v658.arc_eligibility.v1",
        "experiment_id": "exp7526-arc-eligibility",
        "milestone": "2026.09.658",
        "run_date": "20260922",
        "eligibility_receipt_ready_score": 1,
        "flagged_adversarial": False,
        "verdict_class": "null",
        "panel_manifest": {
            "games": ["g50t", "ka59", "vc33", "ft09", "su15", "bp35"],
            "episode_seeds": [658027, 658028],
            "supervisor_window": 120,
            "action_cap": 840,
            "episode_cap_s": 180,
            "collection_cap_s": 3000,
            "schedule": [
                {
                    "episode_id": f"{game}:seed-{seed}",
                    "game": game,
                    "seed": seed,
                    "order": order,
                }
                for order, (game, seed) in enumerate(
                    (game, seed)
                    for game in ["g50t", "ka59", "vc33", "ft09", "su15", "bp35"]
                    for seed in [658027, 658028]
                )
            ],
        },
        "registry_precheck": {
            "policy_received_registry_data": False,
            "read_before_outcomes": True,
            "rows": [{"game": game, "registered": True} for game in exp.PANEL_GAMES],
        },
        "validation_receipts": [{"name": "required", "required": True, "exit_code": 0}],
    }


def _episode(game: str, seed: int, *, explicit: int = 2, eligible: int = 1) -> dict:
    """Build one complete row with model, timing, and eligibility evidence."""

    episode_id = f"{game}:seed-{seed}"
    return {
        "episode_id": episode_id,
        "game": game,
        "seed": seed,
        "disposition": "complete",
        "action_count": 5,
        "start_level": 0,
        "peak_level": 0,
        "terminal_level": 0,
        "elapsed_s": 61.0,
        "action_rows": [
            {
                "action_index": 1,
                "action": "RESET",
                "state_sha256": "sha256:frame",
                "level": 0,
            }
        ],
        "request_budget_receipt": {
            "attempted": 1,
            "completed": 1,
            "failed": 0,
            "cancelled": 0,
        },
        "server_request_rows": [
            {
                "request_sha256": "sha256:request",
                "response_sha256": "sha256:response",
                "prompt_tokens": 10,
                "completion_tokens": 4,
            }
        ],
        "normalized_cost": {
            "episode_wall_s": 61.0,
            "phases": [{"decision_point": "environment", "wall_s": 1.0, "tokens": 0}],
        },
        "window_exposure_count": explicit,
        "explicit_eligibility_count": explicit,
        "eligible_count": eligible,
        "selected_count": eligible,
        "applied_count": 0,
        "supervisor_rows": [
            {
                "action_id": 120,
                "level_id": 0,
                "arm": "force_exploration_diversity",
                "enabled": True,
                "eligible": True,
                "selected": True,
                "applied": False,
                "mode": "shadow",
                "outcome_horizon": "censored_at_episode_end",
            }
        ],
        "error": None,
    }


def _complete_panel() -> list[dict]:
    return [_episode(game, seed) for game in exp.PANEL_GAMES for seed in exp.EPISODE_SEEDS]


def test_schedule_is_exactly_authenticated_upstream_panel() -> None:
    """SCENARIO-ARC-7527-PRECONDITIONS: freeze the producer-owned panel."""

    schedule = exp.build_schedule(_upstream())

    assert len(schedule) == 12
    assert {row["game"] for row in schedule} == set(exp.PANEL_GAMES)
    assert {row["seed"] for row in schedule} == set(exp.EPISODE_SEEDS)
    assert all(row["action_limit"] == 840 for row in schedule)
    assert all(row["episode_limit_s"] == 180 for row in schedule)
    assert all(row["supervisor_mode"] == "shadow" for row in schedule)
    assert all(set(exp.WITHHELD_INPUTS) <= set(row["withheld_inputs"]) for row in schedule)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("eligibility_receipt_ready_score", 0),
        ("flagged_adversarial", True),
        ("experiment_id", "wrong"),
    ],
)
def test_upstream_authentication_fails_closed(field: str, value: object) -> None:
    """SCENARIO-ARC-7527-PRECONDITIONS: altered producer fields cannot pass."""

    upstream = _upstream()
    upstream[field] = value

    checks = exp.authenticate_upstream_fields(upstream)

    assert any(row["passed"] is False and row["field"] == field for row in checks)


def test_opportunity_support_is_independent_of_eligible_count() -> None:
    """SCENARIO-ARC-7527-SUPPORT: an authenticated zero remains a valid null."""

    rows = _complete_panel()
    for row in rows:
        row["eligible_count"] = 0
        row["selected_count"] = 0
        row["supervisor_rows"] = []

    reduced = exp.reduce_panel(rows)

    assert reduced["complete_episode_count"] == 12
    assert reduced["complete_game_count"] == 6
    assert reduced["eligibility_authentication_rate"] == 1.0
    assert reduced["eligible_opportunity_count"] == 0
    assert reduced["opportunity_support_score"] == 1
    assert reduced["benefit_supported"] is False


def test_support_requires_ten_episodes_six_games_and_ninety_percent() -> None:
    """REQ-ARC-7527: every support threshold is independently load-bearing."""

    rows = _complete_panel()
    assert exp.reduce_panel(rows)["opportunity_support_score"] == 1
    assert exp.reduce_panel(rows[:9])["opportunity_support_score"] == 0
    assert (
        exp.reduce_panel([row for row in rows if row["game"] != "bp35"])[
            "opportunity_support_score"
        ]
        == 0
    )
    degraded = deepcopy(rows)
    degraded[0]["explicit_eligibility_count"] = 0
    degraded[0]["window_exposure_count"] = 3
    degraded[1]["explicit_eligibility_count"] = 0
    degraded[1]["window_exposure_count"] = 3
    assert exp.reduce_panel(degraded)["eligibility_authentication_rate"] < 0.9
    assert exp.reduce_panel(degraded)["opportunity_support_score"] == 0


def test_reducer_keeps_shadow_selection_separate_from_application() -> None:
    """SCENARIO-ARC-7527-OPPORTUNITY: selection is not treatment."""

    reduced = exp.reduce_panel(_complete_panel())

    assert reduced["eligible_opportunity_count"] == 12
    assert reduced["selected_opportunity_count"] == 12
    assert reduced["applied_opportunity_count"] == 0
    assert reduced["effect_estimate"] is None
    assert reduced["arm_refinement"] is None
    assert reduced["new_level_credit"] == 0
    assert reduced["future_causal_trial_ready"] is False


def test_future_trial_readiness_uses_actual_choices_games_and_arms() -> None:
    """REQ-ARC-7527: a future trial threshold does not claim present efficacy."""

    rows = _complete_panel()
    for index, row in enumerate(rows):
        row["supervisor_rows"][0]["arm"] = (
            "force_exploration_diversity" if index % 2 else "allow_reinduction"
        )
        row["supervisor_rows"][0]["mode"] = "applied"
        row["supervisor_rows"][0]["applied"] = True
        row["applied_count"] = 1

    reduced = exp.reduce_panel(rows)

    assert reduced["future_causal_trial_ready"] is True
    assert reduced["benefit_supported"] is False
    assert reduced["effect_estimate"] is None


def test_full_generation_class_depends_on_current_generation_events() -> None:
    """SCENARIO-ARC-7527-LIVE: declare the work that actually ran."""

    none = exp.reduce_invocations([], child_terminal=True)
    loaded = exp.reduce_invocations(
        [{"kind": "model_load", "event": "attempted"}], child_terminal=True
    )
    generated = exp.reduce_invocations(
        [
            {"kind": "model_load", "event": "attempted"},
            {"kind": "generation", "event": "attempted"},
        ],
        child_terminal=True,
    )

    assert none["inference_substrate_class"] == "blocked_no_run"
    assert none["MODEL_SPECS"] == []
    assert loaded["inference_substrate_class"] == "model_load_no_generation"
    assert generated["inference_substrate_class"] == "model_full_generation"
    assert generated["inference_substrate"] == "live_llm_inference"
    assert generated["MODEL_SPECS"] == [exp.MODEL_ID]


def test_blocked_artifact_names_exact_failed_path_and_values(tmp_path: Path) -> None:
    """SCENARIO-ARC-7527-PRECONDITIONS: external absence is blocked, not partial."""

    artifact = exp.build_blocked_artifact(
        tmp_path,
        failed_check={
            "check": "exp7526_readiness",
            "path": exp.UPSTREAM_PATH.as_posix(),
            "field": "eligibility_receipt_ready_score",
            "expected": 1,
            "observed": None,
        },
        schedule=[],
        preconditions=[],
        sources={},
        duration_s=0.25,
    )

    assert artifact["honest_verdict"].startswith("complete_blocked_")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["arc_measurement_complete_score"] == 0
    assert artifact["opportunity_support_score"] == 0
    assert artifact["gate_check_summary"]["first_failure"] == {
        "check": "exp7526_readiness",
        "path": exp.UPSTREAM_PATH.as_posix(),
        "field": "eligibility_receipt_ready_score",
        "expected": 1,
        "observed": None,
    }


def test_fixture_artifact_validates_and_independently_reduces(tmp_path: Path) -> None:
    """SCENARIO-ARC-7527-TERMINAL: readers reproduce the candidate from rows."""

    artifact = exp.build_artifact_for_test(tmp_path, _upstream(), _complete_panel())
    replay = exp.independent_reduce(artifact)

    assert exp.validate_artifact(artifact, require_terminal=True) == []
    assert replay["matches_declared"] is True
    assert artifact["arc_measurement_complete_score"] == 1
    assert artifact["opportunity_support_score"] == 1
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_null_")
    assert artifact["inference_substrate_class"] == "model_full_generation"
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["new_level_credit"] == 0


def test_checksum_and_reduction_reject_row_tampering(tmp_path: Path) -> None:
    """SCENARIO-ARC-7527-TERMINAL: changed evidence cannot replay as identical."""

    artifact = exp.build_artifact_for_test(tmp_path, _upstream(), _complete_panel())
    artifact["rows"][0]["eligible_count"] = 99

    assert "independent_reduction_mismatch" in exp.validate_artifact(
        artifact, require_terminal=True
    )
    assert exp.independent_reduce(artifact)["matches_declared"] is False


def test_validation_plan_is_scoped_and_has_terminal_readers(tmp_path: Path) -> None:
    """SCENARIO-ARC-7527-TERMINAL: freeze affected checks before publication."""

    commands = exp.build_validation_plan(exp.REPO_ROOT, tmp_path / "private")
    errors = exp.validate_validation_plan(exp.REPO_ROOT, commands)
    names = {command.name for command in commands}

    assert errors == []
    assert {"focused_pytest", "changed_module_coverage", "changed_module_mypy"} <= names
    assert {"e2e_009", "e2e_010", "e2e_011", "private_arc_smoke"} <= names


def test_replay_cli_never_writes_the_terminal_result(tmp_path: Path) -> None:
    """SCENARIO-ARC-7527-TERMINAL: cold replay is read-only."""

    artifact = exp.build_artifact_for_test(tmp_path, _upstream(), _complete_panel())
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(artifact), encoding="utf-8")

    assert exp.main(["--replay", str(candidate), "--reduce-only"]) == 0
    assert not (tmp_path / exp.RESULT_PATH).exists()


def test_small_io_progress_and_precondition_helpers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-ARC-7527: hashes, clocks, progress, and malformed bytes fail explicitly."""

    payload = tmp_path / "payload.json"
    payload.write_text('{"ok": true}', encoding="utf-8")
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    sequence = tmp_path / "sequence.json"
    sequence.write_text("[]", encoding="utf-8")

    assert exp.utc_now().endswith("Z")
    exp.progress(time.monotonic(), "test", "boundary", unit=1)
    assert "phase=test event=boundary" in capsys.readouterr().out
    assert exp.sha256_file(payload) == "sha256:" + hashlib.sha256(payload.read_bytes()).hexdigest()
    assert exp.load_object(payload) == {"ok": True}
    assert exp.load_object(invalid) == {}
    assert exp.load_object(sequence) == {}
    assert exp.load_object(tmp_path / "missing.json") == {}
    assert exp._source_row(tmp_path, Path("payload.json"))["exists"] is True
    assert exp._source_row(tmp_path, Path("missing.json"))["exists"] is False


def test_repository_preconditions_and_registry_are_authenticated() -> None:
    """SCENARIO-ARC-7527-PRECONDITIONS: inspect the real named inputs."""

    checks, sources, upstream = exp.collect_preconditions(exp.REPO_ROOT)
    registry = exp.registry_precheck(exp.REPO_ROOT, exp.PANEL_GAMES)

    assert checks
    assert all(row["passed"] is True for row in checks)
    assert sources[exp.UPSTREAM_PATH.as_posix()]["exists"] is True
    assert upstream["eligibility_receipt_ready_score"] == 1
    assert registry["policy_received_registry_data"] is False
    assert len(registry["rows"]) == 6


def test_invalid_panel_and_gate_operator_raise() -> None:
    """REQ-ARC-7527: malformed protocol and comparisons fail closed."""

    upstream = _upstream()
    upstream["experiment_id"] = "wrong"
    with pytest.raises(ValueError, match="unauthenticated_exp7526"):
        exp.build_schedule(upstream)
    with pytest.raises(ValueError, match="unsupported_gate_operator"):
        exp._gate("x", "validity", 1, 1, "!=", "test")


def test_natural_interface_strata_cover_coordinate_discrete_and_empty() -> None:
    """REQ-ARC-7527: descriptive strata follow observed action labels."""

    assert exp._interface_stratum({"action_rows": [{"action": "ACTION6"}]}) == (
        "coordinate_action_exposed"
    )
    assert exp._interface_stratum({"action_rows": [{"action": "ACTION1"}]}) == (
        "discrete_action_only_observed"
    )
    assert exp._interface_stratum({"action_rows": [{"action": "RESET"}]}) == (
        "no_post_reset_action_observed"
    )


def test_terminal_verdict_branches_remain_null_or_disqualified(tmp_path: Path) -> None:
    """SCENARIO-ARC-7527-SUPPORT: support and validity select separate outcomes."""

    zero = _complete_panel()
    for row in zero:
        row["eligible_count"] = 0
        row["selected_count"] = 0
        row["supervisor_rows"] = []
    zero_artifact = exp.build_artifact_for_test(tmp_path, _upstream(), zero)

    insufficient = _complete_panel()
    for row in insufficient[-3:]:
        row["disposition"] = "censored_timeout"
    insufficient_artifact = exp.build_artifact_for_test(tmp_path, _upstream(), insufficient)
    invalid_artifact = exp.build_artifact_for_test(
        tmp_path, _upstream(), _complete_panel(), validation_passed=False
    )

    assert zero_artifact["honest_verdict"] == "complete_null_zero_eligible_supervisor_opportunities"
    assert insufficient_artifact["honest_verdict"] == (
        "complete_null_insufficient_authenticated_opportunity_support"
    )
    assert invalid_artifact["verdict_class"] == "disqualified"


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (lambda value: value.pop("schema"), "missing_field:schema"),
        (lambda value: value.__setitem__("schema", "wrong"), "schema_mismatch"),
        (lambda value: value.__setitem__("experiment_id", "wrong"), "experiment_id_mismatch"),
        (lambda value: value.__setitem__("milestone", "wrong"), "milestone_mismatch"),
        (
            lambda value: value.__setitem__("arc_measurement_complete_score", True),
            "measurement_score_not_bare_numeric",
        ),
        (
            lambda value: value.__setitem__("opportunity_support_score", None),
            "opportunity_score_not_bare_numeric",
        ),
        (lambda value: value.__setitem__("rows", {}), "rows_not_list"),
        (
            lambda value: value.__setitem__("supervisor_opportunity_rows", []),
            "supervisor_rows_mismatch",
        ),
        (
            lambda value: value.__setitem__("model_invoked", False),
            "full_generation_declaration_mismatch",
        ),
        (lambda value: value.__setitem__("MODEL_SPECS", []), "full_generation_model_specs_missing"),
        (
            lambda value: value.__setitem__("duration_s", 1.0),
            "full_generation_duration_below_floor",
        ),
        (lambda value: value.__setitem__("field_principles", {}), "field_principles_incomplete"),
        (lambda value: value.__setitem__("honest_verdict", "null"), "terminal_prefix_missing"),
        (lambda value: value.__setitem__("verdict_class", "unknown"), "verdict_class_invalid"),
    ],
)
def test_cold_validator_rejects_each_contract_violation(
    tmp_path: Path, mutation: object, expected_error: str
) -> None:
    """SCENARIO-ARC-7527-TERMINAL: every required reader condition bites."""

    artifact = exp.build_artifact_for_test(tmp_path, _upstream(), _complete_panel())
    mutation(artifact)  # type: ignore[operator]

    assert expected_error in exp.validate_artifact(artifact, require_terminal=True)


def test_independent_reader_rejects_supervisor_row_rewrite(tmp_path: Path) -> None:
    """SCENARIO-ARC-7527-TERMINAL: comparative rows are independently derived."""

    artifact = exp.build_artifact_for_test(tmp_path, _upstream(), _complete_panel())
    artifact["supervisor_opportunity_rows"] = []

    assert "supervisor_rows_mismatch" in exp.independent_reduce(artifact)["errors"]


def test_terminal_specs_receipt_reduction_and_phase_rows(tmp_path: Path) -> None:
    """REQ-ARC-7527: exact readers, receipt status, and monotonic spans are explicit."""

    specs = exp.terminal_command_specs(exp.REPO_ROOT, tmp_path / "candidate.json")
    names = [row.name for row in specs]
    receipts = [
        {"name": name, "passed": True, "exit_code": 0, "timed_out": False} for name in names
    ]
    span = exp._phase("fixture", time.monotonic(), time.monotonic(), 1, "checkpoint")

    assert names == [
        "declared_entrypoint_cold_replay",
        "independent_cold_reducer",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    ]
    assert exp._receipts_pass(receipts, names) is True
    receipts[0]["timed_out"] = True
    assert exp._receipts_pass(receipts, names) is False
    assert span["phase"] == "fixture"
    assert span["completed_units"] == 1


def test_runtime_row_helpers_preserve_eligibility_and_progress(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-ARC-7527-LIVE: join raw eligibility and wrap generation unchanged."""

    eligibility = tmp_path / "raw" / "eligibility.jsonl"
    eligibility.parent.mkdir(parents=True)
    eligibility.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "action_id": 119,
                    "level_id": 0,
                    "observation_status": "complete",
                    "arm_rows": [{"arm": "too_early", "eligible": True}],
                },
                {
                    "action_id": 120,
                    "level_id": 0,
                    "observation_status": "complete",
                    "predicate_inputs": {"induced": True},
                    "application_disposition": "shadow_recommendation",
                    "old_state_hash": None,
                    "new_state_hash": None,
                    "arm_rows": [
                        {
                            "arm": "allow_reinduction",
                            "enabled": True,
                            "eligible": True,
                            "selected": True,
                            "applied": False,
                            "mode": "shadow",
                        }
                    ],
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    row = exp._attach_eligibility(
        {"disposition": "censored_timeout", "episode_id": "fixture"}, eligibility, tmp_path
    )
    unstarted = exp._unstarted_row({"episode_id": "x", "game": "g50t", "seed": 1}, "unstarted")

    class Proposer:
        def generate(self, value: str) -> str:
            return value.upper()

    proposer = Proposer()
    checkpoint = tmp_path / "checkpoint.json"
    exp._install_generation_progress(proposer, time.monotonic(), checkpoint)

    assert proposer.generate("ok") == "OK"
    assert "phase=generation event=before" in capsys.readouterr().out
    assert row["window_exposure_count"] == 1
    assert row["eligible_count"] == 1
    assert row["selected_count"] == 1
    assert row["applied_count"] == 0
    assert row["censored_outcome_horizon"] == "censored_timeout"
    assert unstarted["window_exposure_count"] == 0


def test_shared_runner_failure_retag_and_atomic_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-7527: shared helpers retain identity and publish one complete object."""

    exp._configure_shared_runner()
    artifact = exp.build_blocked_artifact(
        tmp_path,
        failed_check={
            "check": "fixture",
            "path": "fixture",
            "field": "ready",
            "expected": 1,
            "observed": 0,
        },
        schedule=[],
        preconditions=[],
        sources={},
        duration_s=0.1,
    )
    retagged = exp._retag_failure(
        artifact,
        verdict="complete_disqualified_fixture",
        verdict_class="disqualified",
    )
    monkeypatch.setattr(exp, "RESULT_PATH", Path("result.json"))
    exp._publish(tmp_path, retagged, time.monotonic())

    assert exp.exp7471.WRAPPER_PATH == exp.WRAPPER_PATH
    assert exp.exp7491.ACTION_LIMIT == exp.ACTION_LIMIT
    assert json.loads((tmp_path / "result.json").read_text())["verdict_class"] == "disqualified"


def test_git_revision_returns_none_when_process_launch_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-7527: unavailable source revision stays explicit."""

    def raising(*_args: object, **_kwargs: object) -> object:
        raise OSError("git unavailable")

    monkeypatch.setattr(exp.subprocess, "run", raising)
    assert exp._git_revision(tmp_path) is None


def test_cli_validation_replay_and_argument_failure(tmp_path: Path) -> None:
    """SCENARIO-ARC-7527-TERMINAL: both cold-reader roles are exercised."""

    artifact = exp.build_artifact_for_test(tmp_path, _upstream(), _complete_panel())
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(artifact), encoding="utf-8")

    assert exp.main(["--replay", str(candidate)]) == 0
    with pytest.raises(SystemExit):
        exp.parse_args([])


def test_main_dispatches_owned_child_and_experiment(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-ARC-7527: the thin CLI dispatches without changing returned status."""

    monkeypatch.setattr(exp, "_configure_shared_runner", lambda: None)
    monkeypatch.setattr(exp, "run_live_session", lambda _args: 0)
    assert exp.main(["--date", exp.RUN_DATE, "--role", "live-session"]) == 0

    monkeypatch.setattr(
        exp,
        "run_experiment",
        lambda _root, _date: {"honest_verdict": "complete_null_fixture"},
    )
    assert exp.main(["--date", exp.RUN_DATE]) == 0
    monkeypatch.setattr(
        exp,
        "run_experiment",
        lambda _root, _date: {"honest_verdict": "partial_fixture"},
    )
    assert exp.main(["--date", exp.RUN_DATE]) == 1

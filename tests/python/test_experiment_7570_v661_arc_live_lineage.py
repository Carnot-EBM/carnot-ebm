"""Tests for REQ-ARC-WMTE-7570 live ARC plan-lineage measurement."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time

import pytest

from carnot import experiment_7570_v661_arc_live_lineage as exp
from carnot import experiment_7562_v661_arc_plan_lineage as lineage


def _upstream() -> dict:
    return {
        "experiment_id": "exp7562-arc-plan-lineage",
        "run_date": "20260923",
        "plan_lineage_ready_score": 1,
        "verdict_class": "circular_positive",
        "flagged_adversarial": False,
        "frozen_arc_roster": lineage.freeze_arc_roster(),
    }


def _episode(schedule: dict, disposition: str = "complete") -> dict:
    return {
        **deepcopy(schedule),
        "disposition": disposition,
        "action_count": 40,
        "elapsed_s": 12.5,
        "start_level": 0,
        "peak_level": 1,
        "terminal_level": 1,
        "backend_usage_rows": [{"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50}],
        "action_rows": [
            {"action_index": 1, "level": 0, "state_sha256": "a"},
            {"action_index": 2, "level": 1, "state_sha256": "b"},
        ],
        "solve_provenance": "live_agent_self_discovery",
        "trace_reproduction": {"attempted": True, "passed": True},
        "new_level_credit": 0,
        "recorder_error_count": 0,
        "error": None,
    }


def _telemetry(episode_id: str) -> list[dict]:
    attempt = f"{episode_id}:attempt:1"
    model = f"{episode_id}:model:1"
    plan = f"{episode_id}:plan:1"
    action = f"{episode_id}:action:1"
    return [
        {
            "episode_id": episode_id,
            "seam": "induction_timing",
            "gate_decision": "induce_now",
            "attempt_id": attempt,
        },
        {
            "episode_id": episode_id,
            "seam": "world_model_hypothesis_gate",
            "attempt_id": attempt,
            "model_version": model,
            "outcome": "accept",
        },
        {
            "episode_id": episode_id,
            "seam": "planner_invocation",
            "induction_attempt_id": attempt,
            "model_version": model,
            "plan_id": plan,
        },
        {
            "episode_id": episode_id,
            "seam": "policy_action",
            "induction_attempt_id": attempt,
            "model_version": model,
            "plan_id": plan,
            "action_id": action,
            "plan_linked": True,
        },
        {
            "episode_id": episode_id,
            "seam": "level_transition",
            "action_id": action,
            "plan_linked": True,
            "level_delta": 1,
        },
        {
            "episode_id": episode_id,
            "record_type": "plan_lineage_terminal",
            "induction_attempt_id": attempt,
            "model_version": model,
            "plan_id": plan,
            "executed_action_ids": [action],
            "level_progress_action_id": action,
            "terminal_stage": "executed_with_level_progress",
            "closure_reason": "level_progress",
        },
        {
            "episode_id": episode_id,
            "seam": "supervisor_arm_selection",
            "chosen_arm": "force_exploration_diversity",
            "trajectory_snapshot": {"level": 0},
        },
    ]


def test_upstream_precondition_failure_is_exact_and_blocked(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7570-UPSTREAM-BLOCK names the failed field."""

    checks, upstream = exp.collect_preconditions(tmp_path)
    assert upstream == {}
    failed = exp.first_failed_precondition(checks)
    assert failed is not None
    assert failed["upstream"] == exp.UPSTREAM_PATH.as_posix()
    assert failed["artifact_field"] == "path"
    assert failed["expected"] == "readable_file"
    assert failed["observed"] is None
    blocked = exp.build_blocked_artifact("20260923", checks, duration_s=0.01)
    assert blocked["honest_verdict"].startswith("complete_blocked_")
    assert blocked["verdict_class"] == "blocked"
    assert blocked["planned_inference_substrate_class"] == "model_full_generation"
    assert blocked["inference_substrate_class"] == "blocked_no_run"
    assert blocked["MODEL_SPECS"] == [exp.MODEL_ID]
    assert blocked["model_specs"] == []
    assert blocked["sample_size_budget"]["unstarted"] == 12
    assert exp.validate_artifact(blocked, require_terminal=False) == []


def test_roster_is_copied_exactly_from_exp7562() -> None:
    """SCENARIO-ARC-WMTE-7570-FROZEN-ROSTER preserves all sealed fields."""

    upstream = _upstream()
    schedule = exp.schedule_from_upstream(upstream)
    assert len(schedule) == 12
    for source, row in zip(upstream["frozen_arc_roster"], schedule, strict=True):
        for key in (
            "episode_id",
            "game",
            "seed",
            "policy_action_limit",
            "induction_limit",
            "request_token_ceiling",
            "sampler",
        ):
            assert row[key] == source[key]
        assert row["disposition"] == "unstarted"
        assert row["adapter_disabled"] is True
        assert row["registry_trajectories_disabled"] is True

    wrong = deepcopy(upstream)
    wrong["frozen_arc_roster"][0]["request_token_ceiling"] = 8192
    with pytest.raises(ValueError, match="frozen_roster_mismatch"):
        exp.schedule_from_upstream(wrong)


def test_budget_stop_uses_completed_episode_p95() -> None:
    """SCENARIO-ARC-WMTE-7570-BUDGET-STOP reserves the measured p95."""

    assert exp.episode_p95([]) == exp.DEFAULT_EPISODE_RESERVE_S
    assert exp.episode_p95([10.0, 20.0, 30.0]) == pytest.approx(29.0)
    assert exp.can_start_episode(29.0, [10.0, 20.0, 30.0]) is True
    assert exp.can_start_episode(28.999, [10.0, 20.0, 30.0]) is False
    assert exp.remaining_episode_timeout_s(400.0) == 300.0
    assert exp.remaining_episode_timeout_s(100.0) == 99.0


def test_live_reducer_joins_lineage_and_supervisor_outcomes() -> None:
    """SCENARIO-ARC-WMTE-7570-LINEAGE-REDUCTION keeps joined outcomes distinct."""

    schedule = exp.schedule_from_upstream(_upstream())
    episodes = [_episode(schedule[0])]
    telemetry_rows = _telemetry(schedule[0]["episode_id"])
    reduced = exp.reduce_live_measurement(schedule, episodes, telemetry_rows)
    assert reduced["lineage"]["valid"] is True
    assert reduced["support_counts"] == {
        "opportunities": 1,
        "attempts": 1,
        "valid_lifecycle_dispositions": 1,
        "useful": 1,
        "unusable": 0,
        "unknown": 0,
        "censored": 0,
        "useful_games": 1,
        "unusable_games": 0,
    }
    assert reduced["plan_lineage_measured_score"] == 1
    assert reduced["lineage_disposition_fraction"] == 1.0
    assert reduced["sample_size_budget"] == {
        "planned": 12,
        "attempted": 1,
        "completed": 1,
        "excluded": 0,
        "failed": 0,
        "censored": 0,
        "unstarted": 11,
    }
    game = reduced["per_game_results"][0]
    assert game["attempt_stage_counts"] == {"executed_with_level_progress": 1}
    assert game["tokens"] == {"prompt": 20, "completion": 30, "total": 50}
    assert game["plan_derived_level_progress"] == 1
    assert game["incidental_frame_change"] == 0
    assert game["unidentifiable_attribution"] == 0
    assert reduced["trajectory_supervisor"]["arms"] == [
        {"arm": "force_exploration_diversity", "fired": 1, "helped": 1}
    ]
    assert reduced["trajectory_supervisor"]["stagnations_unredirected"] == 0
    assert reduced["trajectory_supervisor"]["refinement_supported"] is True

    unknown = exp.reduce_live_measurement(schedule, episodes, telemetry_rows[:-2])
    assert unknown["plan_lineage_measured_score"] == 0
    assert unknown["support_counts"]["unknown"] == 1


def test_zero_attempts_close_as_valid_null_without_refinement() -> None:
    """SCENARIO-ARC-WMTE-7570-TERMINAL-NULL keeps a zero-attempt null valid."""

    schedule = exp.schedule_from_upstream(_upstream())
    reduced = exp.reduce_live_measurement(schedule, [_episode(schedule[0])], [])
    assert reduced["lineage"]["valid"] is True
    assert reduced["support_counts"]["attempts"] == 0
    assert reduced["plan_lineage_measured_score"] == 0
    assert reduced["lineage_disposition_fraction"] is None
    assert reduced["trajectory_supervisor"]["refinement_supported"] is False
    assert reduced["numeric_gate_quality_claim"] is False
    assert reduced["gate_ready_to_ship"] is False


def test_candidate_schema_and_cold_reduction_are_mutation_sensitive(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7570 binds model custody, rows, gates, and raw evidence."""

    checks = [exp.precondition_row("upstream", "upstream.json", "ready", 1, 1)]
    schedule = exp.schedule_from_upstream(_upstream())
    episodes = [_episode(schedule[0])]
    telemetry_rows = _telemetry(schedule[0]["episode_id"])
    reduced = exp.reduce_live_measurement(schedule, episodes, telemetry_rows)
    counts = {
        **exp.ZERO_INVOCATION_COUNTS,
        "model_loads_attempted": 1,
        "model_loads_completed": 1,
        "generation_calls_attempted": 1,
        "generation_calls_completed": 1,
    }
    artifact = exp.build_artifact(
        "20260923",
        checks=checks,
        schedule=schedule,
        episodes=episodes,
        telemetry_rows=telemetry_rows,
        reduced=reduced,
        invocation_counts=counts,
        model_specs=[{"repository": exp.MODEL_ID, "sha256": "sha256:model"}],
        runtime_receipt={"offload_real": True, "gpu_uuid": "GPU-test"},
        validation_receipts=[],
        duration_s=61.0,
        phase_spans=[],
        source_hashes={exp.UPSTREAM_PATH.as_posix(): "sha256:upstream"},
        terminal=False,
    )
    assert exp.validate_artifact(artifact, require_terminal=False) == []
    assert artifact["verdict_class"] == "partial"
    assert artifact["arc_measurement_complete_score"] == 0
    assert artifact["plan_lineage_measured_score"] == 1
    assert artifact["gate_ready_to_ship"] is False
    assert artifact["numeric_gate_quality_claim"] is False

    path = tmp_path / "candidate.json"
    exp.atomic_json(path, artifact)
    replay = exp.cold_replay(path)
    assert replay["plan_lineage_measured_score"] == 1
    assert replay["planned"] == 12

    corrupt = deepcopy(artifact)
    corrupt["raw_telemetry_rows"][-2]["model_version"] = "corrupt"
    corrupt["reproducibility_checksum"] = exp.reproducibility_checksum(corrupt)
    assert "lineage_reduction_mismatch" in exp.validate_artifact(corrupt, require_terminal=False)

    bad = deepcopy(artifact)
    bad["execution_venue"] = "host_cpu"
    bad["reproducibility_checksum"] = exp.reproducibility_checksum(bad)
    assert "execution_venue_invalid" in exp.validate_artifact(bad, require_terminal=False)


def test_helpers_progress_io_and_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """REQ-ARC-WMTE-7570 keeps public utility behavior explicit and bounded."""

    started = time.monotonic()
    exp.progress(started, "unit", "boundary", completed_units=1)
    assert "phase=unit event=boundary" in capsys.readouterr().out
    path = tmp_path / "nested" / "artifact.json"
    exp.atomic_json(path, {"value": 1})
    assert exp.load_json(path) == {"value": 1}
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="json_object_required"):
        exp.load_json(path)
    args = exp.parse_args(["--date", "20260923", "--validate", "/tmp/candidate.json"])
    assert args.validate == Path("/tmp/candidate.json")
    assert exp.first_failed_precondition([{"passed": True}]) is None
    private = exp.prepare_validation_directories(tmp_path / "private")
    assert private == tmp_path / "private"
    assert (private / "pytest").is_dir()


def test_precondition_and_small_helper_branches(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7570 rejects malformed custody and covers deterministic helpers."""

    malformed = tmp_path / exp.UPSTREAM_PATH
    malformed.parent.mkdir(parents=True)
    malformed.write_text("[]", encoding="utf-8")
    checks, upstream = exp.collect_preconditions(tmp_path)
    assert upstream == {}
    assert checks[0]["observed"] == "malformed_json"

    actual_checks, actual_upstream = exp.collect_preconditions(exp.REPO_ROOT)
    assert actual_upstream["experiment_id"] == "exp7562-arc-plan-lineage"
    assert all(row["passed"] is True for row in actual_checks)
    with pytest.raises(ValueError, match="frozen_roster_mismatch:count"):
        exp.schedule_from_upstream({"frozen_arc_roster": []})
    assert exp.episode_p95([3.0]) == 3.0
    assert exp.utc_now().endswith("Z")

    passing_receipts = [
        {"name": name, "passed": True, "exit_code": 0} for name in exp.REQUIRED_RECEIPT_NAMES
    ]
    assert exp._receipts_pass(passing_receipts) is True
    assert exp._receipts_pass(passing_receipts[:-1]) is False

    tick = time.monotonic()
    assert exp._phase("unit", tick, tick)["phase"] == "unit"
    model = exp._model_spec({"model_path": "/tmp/model.gguf", "model_hash": "sha256:x"})
    assert model["repository"] == exp.MODEL_ID
    assert model["max_new_tokens"] == 4096
    assert exp.UPSTREAM_PATH.as_posix() in exp._source_hashes(exp.REPO_ROOT)
    assert exp._invocation_counts([]) == exp.ZERO_INVOCATION_COUNTS


def test_validation_rejects_terminal_and_blocked_mutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-7570 fails closed on schema, accounting, and receipt mutations."""

    checks = [exp.precondition_row("upstream", "upstream.json", "ready", 1, 1)]
    schedule = exp.schedule_from_upstream(_upstream())
    episodes = [_episode(schedule[0])]
    telemetry_rows = _telemetry(schedule[0]["episode_id"])
    reduced = exp.reduce_live_measurement(schedule, episodes, telemetry_rows)
    counts = {
        **exp.ZERO_INVOCATION_COUNTS,
        "model_loads_attempted": 1,
        "model_loads_completed": 1,
        "generation_calls_attempted": 1,
        "generation_calls_completed": 1,
    }
    receipts = [
        {"name": name, "passed": True, "exit_code": 0} for name in exp.REQUIRED_RECEIPT_NAMES
    ]
    final = exp.build_artifact(
        "20260923",
        checks=checks,
        schedule=schedule,
        episodes=episodes,
        telemetry_rows=telemetry_rows,
        reduced=reduced,
        invocation_counts=counts,
        model_specs=[{"repository": exp.MODEL_ID}],
        runtime_receipt={"offload_real": True},
        validation_receipts=receipts,
        duration_s=61.0,
        phase_spans=[],
        source_hashes={},
        terminal=True,
    )
    assert exp.validate_artifact(final, require_terminal=True) == []

    mutations = (
        ("experiment_id", "wrong", "experiment_id_mismatch"),
        ("milestone", "wrong", "experiment_identity_mismatch"),
        ("verdict_class", "wrong", "verdict_class_invalid"),
        ("arc_measurement_complete_score", 2, "arc_measurement_complete_score_invalid"),
        ("plan_lineage_measured_score", 2, "plan_lineage_measured_score_invalid"),
        ("gate_ready_to_ship", True, "gate_ready_to_ship_must_be_false"),
        ("model_invoked", False, "model_invoked_mismatch"),
    )
    for field, value, expected in mutations:
        broken = deepcopy(final)
        broken[field] = value
        broken["reproducibility_checksum"] = exp.reproducibility_checksum(broken)
        assert expected in exp.validate_artifact(broken, require_terminal=True)

    bad_hash = deepcopy(final)
    bad_hash["reproducibility_checksum"] = "sha256:wrong"
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(
        bad_hash, require_terminal=True
    )
    no_principles = deepcopy(final)
    no_principles["field_principles"] = {}
    no_principles["reproducibility_checksum"] = exp.reproducibility_checksum(no_principles)
    assert "field_principles_incomplete" in exp.validate_artifact(
        no_principles, require_terminal=True
    )
    wrong_budget = deepcopy(final)
    wrong_budget["sample_size_budget"]["unstarted"] -= 1
    wrong_budget["reproducibility_checksum"] = exp.reproducibility_checksum(wrong_budget)
    assert "sample_size_budget_mismatch" in exp.validate_artifact(
        wrong_budget, require_terminal=True
    )
    wrong_support = deepcopy(final)
    wrong_support["support_counts"]["attempts"] = 99
    wrong_support["reproducibility_checksum"] = exp.reproducibility_checksum(wrong_support)
    assert "lineage_reduction_mismatch" in exp.validate_artifact(
        wrong_support, require_terminal=True
    )

    failed_check = exp.precondition_row("upstream", "upstream.json", "path", "readable_file", None)
    blocked = exp.build_blocked_artifact("20260923", [failed_check], duration_s=0.1)
    blocked_mutations = (
        ("model_invoked", True, "blocked_model_invoked"),
        (
            "invocation_counts",
            {**exp.ZERO_INVOCATION_COUNTS, "model_loads_attempted": 1},
            "blocked_invocation_counts_nonzero",
        ),
        ("inference_substrate_class", "model_full_generation", "blocked_substrate_invalid"),
        ("honest_verdict", "blocked", "blocked_terminal_prefix_missing"),
    )
    for field, value, expected in blocked_mutations:
        broken = deepcopy(blocked)
        broken[field] = value
        broken["reproducibility_checksum"] = exp.reproducibility_checksum(broken)
        assert expected in exp.validate_artifact(broken, require_terminal=False)

    missing_receipt = deepcopy(final)
    missing_receipt["validation_receipts"] = []
    missing_receipt["reproducibility_checksum"] = exp.reproducibility_checksum(missing_receipt)
    assert "required_validation_failed" in exp.validate_artifact(
        missing_receipt, require_terminal=True
    )
    candidate_wrong = deepcopy(final)
    candidate_wrong["reproducibility_checksum"] = exp.reproducibility_checksum(candidate_wrong)
    assert "candidate_verdict_class_mismatch" in exp.validate_artifact(
        candidate_wrong, require_terminal=False
    )
    cold_bad = deepcopy(final)
    cold_bad["execution_venue"] = "host_cpu"
    cold_bad["reproducibility_checksum"] = exp.reproducibility_checksum(cold_bad)
    invalid_path = tmp_path / "invalid.json"
    exp.atomic_json(invalid_path, cold_bad)
    with pytest.raises(ValueError, match="cold_replay_invalid"):
        exp.cold_replay(invalid_path)

    monkeypatch.setattr(exp, "TEST_PATH", Path("tests/python/not-present-exp7570.py"))
    no_test_hash = exp.build_artifact(
        "20260923",
        checks=checks,
        schedule=schedule,
        episodes=episodes,
        telemetry_rows=telemetry_rows,
        reduced=reduced,
        invocation_counts=counts,
        model_specs=[{"repository": exp.MODEL_ID}],
        runtime_receipt={"offload_real": True},
        validation_receipts=[],
        duration_s=61.0,
        phase_spans=[],
        source_hashes={},
        terminal=False,
    )
    assert "tests/python/not-present-exp7570.py" not in no_test_hash["code_hashes"]

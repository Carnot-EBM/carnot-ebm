"""Tests for the live typed-witness case study.

Spec: REQ-ARC-WMTE-7263 and SCENARIO-ARC-WMTE-7263-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

import carnot.experiment_7263_v639_arc_live as live


def _episode(
    game: str,
    arm: str,
    *,
    heldout: float,
    identity: float,
    levels: int = 0,
    consumed: bool = False,
) -> dict[str, object]:
    return {
        "episode_id": f"{game}:{arm}",
        "game": game,
        "arm": arm,
        "seed": live.RANDOM_SEED,
        "disposition": "complete",
        "censored": False,
        "model_invoked": True,
        "model_loaded": True,
        "generation_calls_attempted": 2,
        "generation_calls_completed": 2,
        "usable_answers": 1,
        "generated_tokens": 3200,
        "action_count": 192,
        "action_limit": live.ACTION_LIMIT,
        "identity_baseline_accuracy": identity,
        "heldout_accuracy": heldout,
        "non_identity_predictions": 1,
        "candidate_valid": True,
        "trust_accepted": True,
        "installed_plans": 1,
        "model_planned_actions": 2,
        "levels": levels,
        "action_cost": 192,
        "compute_cost": {"wall_s": 61.0, "prompt_tokens": 900, "generated_tokens": 3200},
        "policy_consumption_rows": (
            [
                {
                    "episode_id": f"{game}:{arm}",
                    "engine_sha256": "sha256:engine",
                    "plan_sha256": "sha256:plan",
                    "action_index": 91,
                }
            ]
            if consumed
            else []
        ),
        "adapter_disabled": True,
        "raw_request_manifest": [],
        "error": None,
    }


def _positive_rows() -> list[dict[str, object]]:
    return [
        _episode("aa11", "current_feedback", heldout=0.45, identity=0.30, levels=1),
        _episode(
            "aa11", "typed_witness_feedback", heldout=0.75, identity=0.30, levels=1, consumed=True
        ),
        _episode("bb22", "typed_witness_feedback", heldout=0.80, identity=0.35, levels=2),
        _episode("bb22", "current_feedback", heldout=0.50, identity=0.35, levels=2),
    ]


def test_registry_precheck_freezes_two_least_exposed_games_and_counterbalances() -> None:
    """SCENARIO-ARC-WMTE-7263-COUNTERBALANCED-LIVE-LOOP: selection is metadata-only."""
    registry = {
        "games": [
            {"game": "cc33", "levels_reproduced": 2},
            {"game": "aa11", "levels_reproduced": 0},
            {"game": "bb22", "levels_reproduced": 1},
        ],
        "history": ["cc33", "cc33", "bb22"],
    }
    receipt = live.select_frozen_roster(registry, adaptered_games={"aa11", "bb22", "cc33"})
    assert receipt["selection_steps"] == ["registry_precheck", "roster_freeze"]
    assert receipt["games"] == ["aa11", "bb22"]
    assert receipt["selection_used_outcome_labels"] is False
    assert all(row["adapter_disabled"] for row in receipt["game_rows"])
    assert receipt["game_rows"][0]["adapter_available_but_withheld"] is True

    schedule = live.counterbalanced_schedule(receipt["games"], live.RANDOM_SEED)
    assert [(row["game"], row["arm"]) for row in schedule] == [
        ("aa11", "current_feedback"),
        ("aa11", "typed_witness_feedback"),
        ("bb22", "typed_witness_feedback"),
        ("bb22", "current_feedback"),
    ]
    assert len({row["episode_id"] for row in schedule}) == 4


def test_episode_environment_enforces_live_selfparse_and_equal_caps(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7263: both arms differ only at the shipped witness switch."""
    inherited = {"CARNOT_ARC_TRANSITION_WITNESS": "stale", "KEEP": "yes"}
    current = live.episode_environment(
        inherited,
        arm="current_feedback",
        episode_dir=tmp_path / "current",
        gpu_index=1,
        port=8123,
    )
    treatment = live.episode_environment(
        inherited,
        arm="typed_witness_feedback",
        episode_dir=tmp_path / "treatment",
        gpu_index=1,
        port=8123,
    )
    assert "CARNOT_ARC_TRANSITION_WITNESS" not in current
    assert treatment["CARNOT_ARC_TRANSITION_WITNESS"] == "1"
    for env in (current, treatment):
        assert env["CARNOT_FORCE_LIVE"] == "1"
        assert env["CARNOT_ARC_INDUCE_TOOL_LOOP"] == "selfparse"
        assert env["CARNOT_ARC_CEGIS_ACCEPT_SPLIT"] == "1"
        assert env["CARNOT_ARC_CEGIS_TOOL_LOOP"] == "1"
        assert env["CARNOT_ARC_MAX_REFINEMENT_ROUNDS"] == "2"
        assert env["CARNOT_ARC_INDUCE_TOOL_TURNS"] == "1"
        assert env["CARNOT_ARC_INDUCE_MAX_TOKENS"] == str(live.TOKENS_PER_CALL)
        assert env["CARNOT_ARC_E3_DIR"].startswith(str(tmp_path))
        assert env["KEEP"] == "yes"


def test_identity_baseline_uses_reserved_pre_refinement_rows() -> None:
    """REQ-ARC-WMTE-7263: identity is scored only on the reserved acceptance block."""
    changed = [[0, 1], [0, 0]]
    unchanged = [[0, 0], [0, 0]]
    payload = {
        "rows": [
            {
                "index": index,
                "action": 1,
                "grid": unchanged,
                "next_grid": changed if index in {1, 5} else unchanged,
                "level_before": 0,
                "level_after": 0,
            }
            for index in range(6)
        ],
        "prompt_row_ids": ["not-used"],
    }
    scored = live.identity_baseline_from_transition_source(payload)
    assert scored["reserved_row_indices"] == [5]
    assert scored["identity_correct"] == 0
    assert scored["identity_total"] == 1
    assert scored["identity_baseline_accuracy"] == 0.0
    assert scored["reserved_before_refinement"] is True

    assert (
        live.identity_baseline_from_transition_source({"rows": []})["identity_baseline_accuracy"]
        is None
    )


def test_policy_consumption_joins_engine_and_plan_to_executed_actions() -> None:
    """SCENARIO-ARC-WMTE-7263-POLICY-CONSUMPTION: provenance must join all three facts."""
    attempts = [
        {
            "planned": True,
            "refinement_rounds": [
                {"engine_source_sha256": {"sha256_full": "abc"}, "retained_as_best_engine": True}
            ],
        }
    ]
    actions = [
        {
            "i": 17,
            "top_branch": "execute.plan_step",
            "plan_epoch": 3,
            "plan_installed_by_attempt": 0,
            "action": 4,
            "data": None,
        },
        {
            "i": 18,
            "top_branch": "execute.plan_step",
            "plan_epoch": 3,
            "plan_installed_by_attempt": 0,
            "action": 6,
            "data": {"x": 2, "y": 3},
        },
        {"i": 19, "top_branch": "explore.explorer", "action": 1, "data": None},
    ]
    rows = live.build_policy_consumption_rows("aa11:typed_witness_feedback", attempts, actions)
    assert len(rows) == 2
    assert {row["engine_sha256"] for row in rows} == {"sha256:abc"}
    assert len({row["plan_sha256"] for row in rows}) == 1
    assert all(row["policy_action_executed"] for row in rows)
    assert live.build_policy_consumption_rows("empty", [], actions) == []


def test_reducer_separates_capture_from_case_study_value_and_rejects_regression() -> None:
    """SCENARIO-ARC-WMTE-7263-TERMINAL-REDUCTION: all frozen gates are recomputable."""
    reduced = live.reduce_episode_rows(_positive_rows())
    assert reduced["arc_capture_complete_score"] == 1
    assert reduced["arc_method_value_score"] == 1
    assert reduced["typed_witness_prediction_lift_vs_current"] == pytest.approx(0.30)
    assert reduced["typed_witness_prediction_lift_vs_identity"] == pytest.approx(0.45)
    assert reduced["treatment_policy_consumed_plans"] == 1
    assert reduced["matched_game_level_regression"] is False
    assert reduced["population_confidence_interval"] is None
    assert reduced["claim_scope"] == "two_game_case_study_only"
    assert reduced["invocation_counts"] == {
        "model_loads_attempted": 1,
        "model_loads_completed": 1,
        "generation_calls_attempted": 8,
        "generation_calls_completed": 8,
        "usable_answers": 4,
    }

    no_consumption = deepcopy(_positive_rows())
    no_consumption[1]["policy_consumption_rows"] = []
    assert live.reduce_episode_rows(no_consumption)["arc_method_value_score"] == 0

    regression = deepcopy(_positive_rows())
    regression[2]["levels"] = 1
    assert live.reduce_episode_rows(regression)["arc_method_value_score"] == 0

    censored = deepcopy(_positive_rows())
    censored[0].update(
        {"disposition": "censored_timeout", "censored": True, "generation_calls_completed": 0}
    )
    assert live.reduce_episode_rows(censored)["arc_capture_complete_score"] == 1


def test_episode_validator_refuses_budget_or_adapter_drift() -> None:
    """REQ-ARC-WMTE-7263: action, call, token, and adapter limits fail closed."""
    row = _positive_rows()[0]
    assert live.validate_episode_row(row) == []
    for field, value, expected in (
        ("action_count", 193, "action_limit_exceeded"),
        ("generation_calls_attempted", 3, "generation_call_limit_exceeded"),
        ("generated_tokens", 4097, "generated_token_limit_exceeded"),
        ("adapter_disabled", False, "adapter_not_disabled"),
    ):
        broken = deepcopy(row)
        broken[field] = value
        assert expected in live.validate_episode_row(broken)


def test_terminal_artifact_has_required_fields_and_consumption_null_is_explicit() -> None:
    """REQ-ARC-WMTE-7263: terminal evidence keeps ordinary values and field principles."""
    artifact = live.build_terminal_artifact(
        started_at_utc="2026-09-13T12:00:00+00:00",
        ended_at_utc="2026-09-13T12:02:00+00:00",
        duration_s=120.0,
        preconditions=[live.gate_check("upstream", "exp7262", "ready", 1, 1)],
        source_hashes={"results/upstream.json": "sha256:one"},
        selection_receipt={"games": ["aa11", "bb22"]},
        episode_rows=_positive_rows(),
        model_spec={"hf_id": live.MODEL_ID, "quantization": live.QUANTIZATION},
        runtime_receipt={"server_pid": 123, "gpu_uuid": "GPU-one"},
        validation_receipts=[{"name": "focused", "passed": True, "exit_code": 0}],
        phase_spans=[{"phase": "generation", "duration_s": 100.0}],
    )
    assert live.validate_artifact(artifact) == []
    assert artifact["status"] == "complete"
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["arc_method_value_score"] == 1
    assert artifact["verifier_is_oracle"] is True
    assert set(artifact["field_principles"]) == set(artifact)
    assert artifact["MODEL_SPECS"] == [{"hf_id": live.MODEL_ID, "quantization": live.QUANTIZATION}]
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["inference_substrate_class"] == "model_full_generation"
    assert artifact["execution_venue"] == "host"

    no_use = deepcopy(_positive_rows())
    no_use[1]["policy_consumption_rows"] = []
    null_artifact = live.build_terminal_artifact(
        started_at_utc="2026-09-13T12:00:00+00:00",
        ended_at_utc="2026-09-13T12:02:00+00:00",
        duration_s=120.0,
        preconditions=[],
        source_hashes={},
        selection_receipt={"games": ["aa11", "bb22"]},
        episode_rows=no_use,
        model_spec={"hf_id": live.MODEL_ID},
        runtime_receipt={},
        validation_receipts=[],
        phase_spans=[],
    )
    assert null_artifact["honest_verdict"] == "complete_null_no_policy_consumed_plan"
    assert null_artifact["verdict_class"] == "null"

    mutated = deepcopy(artifact)
    mutated["rows"][0]["heldout_accuracy"] = 0.0
    assert "reproducibility_checksum_mismatch" in live.validate_artifact(mutated)


def test_external_failure_builds_terminal_block_with_zero_invocations() -> None:
    """SCENARIO-ARC-WMTE-7263-EXTERNAL-BLOCK: absence is blocked, never partial."""
    failure = live.gate_check("upstream", "exp7262", "arc_witness_ready_score", 1, 0)
    artifact = live.build_blocked_artifact(
        started_at_utc="2026-09-13T12:00:00+00:00",
        ended_at_utc="2026-09-13T12:00:01+00:00",
        duration_s=1.0,
        preconditions=[failure],
        source_hashes={},
    )
    assert live.validate_artifact(artifact) == []
    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"] == [failure]
    assert artifact["inference_substrate"] == "blocked_no_run"
    assert artifact["invocation_counts"]["generation_calls_attempted"] == 0
    assert "partial" not in artifact["honest_verdict"]


def test_upstream_authentication_checks_clean_terminal_bytes(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7263: the exact terminal receipt and newline are authenticated."""
    path = tmp_path / "upstream.json"
    path.write_text(
        json.dumps(
            {
                "experiment_id": "exp7262-arc-witness-receipt",
                "status": "complete",
                "arc_witness_ready_score": 1,
                "verdict_class": "circular_positive",
            }
        )
        + "\n"
    )
    checks, payload = live.authenticate_upstream(path, quarantined=False)
    assert payload["arc_witness_ready_score"] == 1
    assert all(row["passed"] for row in checks)
    assert next(row for row in checks if row["field"] == "clean_terminal_bytes")["observed"] is True

    path.write_bytes(path.read_bytes() + b"\x00")
    checks, _payload = live.authenticate_upstream(path, quarantined=True)
    assert {row["field"] for row in checks if not row["passed"]} == {
        "quarantine_state",
        "clean_terminal_bytes",
    }


def test_independent_raw_reducer_and_scoped_validation_plan(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7263-TERMINAL-REDUCTION: validation is scoped and independent."""
    raw = tmp_path / "rows.json"
    raw.write_text(json.dumps({"rows": _positive_rows()}))
    assert live.independent_reduce(raw)["arc_method_value_score"] == 1

    commands = live.build_validation_commands(
        terminal_candidate=tmp_path / "terminal.json", raw_rows=raw
    )
    names = [row["name"] for row in commands]
    assert names == [
        "focused_exp7263",
        "affected_exp7262",
        "affected_transition_witness",
        "affected_induction_state",
        "scoped_coverage_run",
        "scoped_coverage_report",
        "ruff_check",
        "ruff_format",
        "changed_module_mypy",
        "scoped_spec_coverage",
        "independent_raw_row_reducer",
        "terminal_candidate_adversarial_verify",
        "terminal_candidate_row_consistency",
    ]
    joined = [" ".join(row["command"]) for row in commands]
    assert all("tests/python -q" not in command for command in joined)
    assert all("-n 0" in command for command in joined[:4])
    assert str(tmp_path / "terminal.json") in joined[-1]


def test_gate_check_and_atomic_write_are_stable(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7263: ordinary gates and terminal bytes have stable forms."""
    passed = live.gate_check("kind", "source", "field", True, True)
    assert passed["passed"] is True
    target = tmp_path / "nested" / "artifact.json"
    live.atomic_write(target, {"b": 2, "a": 1})
    assert target.read_bytes() == b'{\n  "a": 1,\n  "b": 2\n}\n'


def test_fail_closed_helper_edges_are_explicit(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7263: malformed inputs and provenance joins fail closed."""
    missing_checks, missing_payload = live.authenticate_upstream(
        tmp_path / "missing.json", quarantined=False
    )
    assert missing_payload == {}
    assert next(row for row in missing_checks if row["field"] == "exists")["passed"] is False

    invalid = tmp_path / "invalid.json"
    invalid.write_bytes(b"\xff\n")
    _checks, invalid_payload = live.authenticate_upstream(invalid, quarantined=False)
    assert invalid_payload == {}

    selection = live.select_frozen_roster(
        {"games": [{"game": ""}, "invalid", {"game": "aa11"}]}, adaptered_games=set()
    )
    assert selection["games"] == ["aa11"]

    actions = [
        {
            "i": 1,
            "top_branch": "induce.plan_from_current",
            "induction_attempt_index": 0,
            "plan_epoch": 1,
            "action": 1,
        }
    ]
    assert live.build_policy_consumption_rows("unplanned", [{"planned": False}], actions) == []
    assert live.build_policy_consumption_rows("no-digest", [{"planned": True}], actions) == []
    invalid_action = deepcopy(_positive_rows()[0])
    invalid_action.update(
        {
            "generation_calls_attempted": 1,
            "generation_calls_completed": 2,
            "arm": "unknown",
            "disposition": "running",
        }
    )
    assert set(live.validate_episode_row(invalid_action)) >= {
        "generation_completion_count_invalid",
        "arm_invalid",
        "disposition_not_terminal",
    }


def test_terminal_builder_and_validator_cover_every_fail_closed_disposition(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7263-TERMINAL-REDUCTION: every invalid terminal shape is named."""
    common = {
        "started_at_utc": "2026-09-13T12:00:00+00:00",
        "ended_at_utc": "2026-09-13T12:02:00+00:00",
        "duration_s": 120.0,
        "preconditions": [],
        "source_hashes": {},
        "selection_receipt": {"games": ["aa11", "bb22"]},
        "model_spec": {"hf_id": live.MODEL_ID, "quantization": live.QUANTIZATION},
        "runtime_receipt": {},
        "phase_spans": [],
    }
    validation_failed = live.build_terminal_artifact(
        **common,
        episode_rows=_positive_rows(),
        validation_receipts=[{"passed": False}],
    )
    assert validation_failed["honest_verdict"] == "complete_null_validation_failed"

    gates_failed_rows = deepcopy(_positive_rows())
    gates_failed_rows[1]["heldout_accuracy"] = 0.40
    gates_failed_rows[2]["heldout_accuracy"] = 0.40
    gates_failed = live.build_terminal_artifact(
        **common, episode_rows=gates_failed_rows, validation_receipts=[]
    )
    assert gates_failed["honest_verdict"] == "complete_null_typed_witness_case_study_gates_not_met"

    load_only_rows = deepcopy(_positive_rows())
    for row in load_only_rows:
        row.update(
            {
                "model_invoked": False,
                "generation_calls_attempted": 0,
                "generation_calls_completed": 0,
                "usable_answers": 0,
                "generated_tokens": 0,
            }
        )
    load_only = live.build_terminal_artifact(
        **common, episode_rows=load_only_rows, validation_receipts=[]
    )
    assert load_only["inference_substrate"] == "model_load_no_generation"
    for row in load_only_rows:
        row["model_loaded"] = False
    no_load = live.build_terminal_artifact(
        **common, episode_rows=load_only_rows, validation_receipts=[]
    )
    assert no_load["inference_substrate"] == "blocked_no_run"

    valid = live.build_terminal_artifact(
        **common, episode_rows=_positive_rows(), validation_receipts=[]
    )
    valid_path = tmp_path / "valid.json"
    valid_path.write_text(json.dumps(valid))
    assert live.validate_artifact(valid_path) == []
    assert live.validate_artifact(tmp_path / "absent.json")[0].startswith("artifact_unreadable")

    malformed = deepcopy(valid)
    malformed.update(
        {
            "schema": "wrong",
            "milestone": "wrong",
            "status": "running",
            "verdict_class": "wrong",
            "field_principles": {},
        }
    )
    malformed_errors = live.validate_artifact(malformed)
    assert {
        "schema_or_experiment_identity_mismatch",
        "milestone_or_run_date_mismatch",
        "status_not_terminal",
        "verdict_class_invalid",
        "field_principles_must_cover_every_top_level_field",
    } <= set(malformed_errors)

    oracle_positive = deepcopy(valid)
    oracle_positive["verdict_class"] = "positive"
    assert "oracle_forbids_positive" in live.validate_artifact(oracle_positive)

    blocked = live.build_blocked_artifact(
        started_at_utc=common["started_at_utc"],
        ended_at_utc=common["ended_at_utc"],
        duration_s=1.0,
        preconditions=[live.gate_check("x", "y", "z", True, False)],
        source_hashes={},
    )
    broken_blocked = deepcopy(blocked)
    broken_blocked.update(
        {
            "honest_verdict": "wrong",
            "rows": [{"unexpected": True}],
            "model_invoked": True,
            "inference_substrate": "live_llm_inference",
            "gate_check_summary": [],
        }
    )
    assert {
        "blocked_verdict_prefix_invalid",
        "blocked_artifact_contains_measurement",
        "blocked_substrate_invalid",
        "blocked_gate_summary_missing",
    } <= set(live.validate_artifact(broken_blocked))

    broken_complete = deepcopy(valid)
    broken_complete["rows"][1]["policy_consumption_rows"] = []
    broken_complete.update(
        {
            "honest_verdict": "wrong",
            "arc_capture_complete_score": 0,
            "arc_method_value_score": 1,
            "invocation_counts": {},
            "MODEL_SPECS": [{"hf_id": "wrong", "quantization": "wrong"}],
            "model_invoked": False,
            "inference_substrate": "wrong",
            "duration_s": 1.0,
            "population_confidence_interval": [0.0, 1.0],
        }
    )
    broken_errors = set(live.validate_artifact(broken_complete))
    assert {
        "complete_verdict_prefix_invalid",
        "arc_capture_complete_score_inconsistent",
        "arc_method_value_score_inconsistent",
        "invocation_counts_inconsistent",
        "model_specs_inconsistent",
        "model_invoked_inconsistent",
        "live_substrate_inconsistent",
        "model_full_generation_duration_floor_failed",
        "no_consumption_verdict_inconsistent",
        "two_game_population_ci_forbidden",
    } <= broken_errors

    bad_no_load = deepcopy(no_load)
    bad_no_load["MODEL_SPECS"] = [{"hf_id": live.MODEL_ID}]
    assert "model_specs_inconsistent" in live.validate_artifact(bad_no_load)

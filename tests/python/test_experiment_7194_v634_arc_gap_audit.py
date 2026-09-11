"""Requirement tests for the Exp7194 ARC gap audit.

Spec refs: REQ-ARC-WMTE-7194 and SCENARIO-ARC-WMTE-7194-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from carnot.experiment_7194_v634_arc_gap_audit import (
    FIELD_PRINCIPLES,
    MODEL_SPECS,
    artifact_checksum,
    build_artifact,
    generalization_recommendation,
    recompute_banked_credit_rows,
    recompute_gap_rows,
    recompute_session_cost_rows,
    upstream_measurement_gate,
    validate_artifact,
)


def test_execution_venue_stays_in_the_closed_set() -> None:
    """REQ-ARC-WMTE-7194 / REQ-SUBSTRATE-VENUE-1: execution_venue must be the bare enum
    value ('host'), never a hostname suffix -- the 2026-09-10 EXECUTION_VENUE_INVALID
    incident (execution_venue='host:icbfl1') quarantined an otherwise honest null result.
    The hostname belongs in the separate, unchecked execution_host field instead."""

    artifact = build_artifact(
        run_date="20260910",
        duration_s=1.25,
        preconditions_checked=[],
        source_artifact_hashes={},
        gap_rows=[],
        banked_credit_rows=[],
        session_cost_rows=[],
        refinement_tool_runs=[],
        historical_context={},
        blocked=True,
    )
    assert artifact["execution_venue"] == "host"
    assert isinstance(artifact["execution_host"], str) and artifact["execution_host"]


def _write_completion(root: Path, name: str, text: str) -> dict[str, object]:
    from carnot.experiment_7194_v634_arc_gap_audit import sha256_bytes

    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return {
        "completion_id": name,
        "content_path": str(path),
        "content_sha256": sha256_bytes(text.encode()),
        "stage": "environment",
        "completion_tokens": 7,
        "error": None,
    }


def _run_row() -> dict[str, object]:
    return {
        "game": "r11l",
        "actions": 18,
        "charged_actions": 20,
        "levels": 2,
        "wall_s": 12.5,
        "generator_wall_s": 10.0,
        "level_up_charged": [8, 12],
        "per_level": [
            {"level": 0, "agent_actions": 7, "completed": True},
            {"level": 1, "agent_actions": 3, "completed": True},
            {"level": 2, "agent_actions": 8, "completed": False},
        ],
        "level_reset_attribution": {
            "run_total_resets": 2,
            "run_total_gateway_charged": 20,
            "segments": [
                {"level_completed": 1, "offline_actions": 7, "resets": 1},
                {"level_completed": 2, "offline_actions": 3, "resets": 1},
            ],
        },
        "policy_diagnostics": {
            "induction_attempts": [
                {
                    "reason": "stall",
                    "started_at": "2026-09-10T00:00:00Z",
                    "wall_s": 6.0,
                    "skipped": "world_model_accuracy_below_threshold",
                    "tool_gap": {
                        "terminated_by": "turn_cap",
                        "tool_calls_total": 1,
                        "tool_gap_events": [],
                        "tool_gap_events_dropped": 0,
                    },
                },
                {
                    "reason": "level_up_reinduction",
                    "started_at": "2026-09-10T00:00:07Z",
                    "wall_s": 4.0,
                    "skipped": "degenerate_goal_predicate",
                },
            ]
        },
        "trajectory_supervisor": {
            "mode": "shadow",
            "enabled": False,
            "would_have_redirects": [
                {
                    "action_index": 2,
                    "level": 0,
                    "arm": "drop_goal_bias",
                    "levelup_followed_without_redirect": True,
                    "actions_to_levelup_without_redirect": 6,
                    "co_credited_count": 2,
                },
                {
                    "action_index": 4,
                    "level": 0,
                    "arm": "force_exploration_diversity",
                    "levelup_followed_without_redirect": True,
                    "actions_to_levelup_without_redirect": 4,
                    "co_credited_count": 2,
                },
            ],
        },
    }


def test_quarantine_is_rejected_before_gate_value_consumption() -> None:
    """SCENARIO-ARC-WMTE-7194-QUARANTINE."""

    check = upstream_measurement_gate(
        {"arc_tool_measurement_complete_score": 1, "flagged_adversarial": True},
        "results/upstream.json",
    )

    assert check["passed"] is False
    assert check["observed_value"] == {
        "value": "not_consumed",
        "quarantined": True,
        "consumed": False,
    }


def test_gap_replay_keeps_zero_gap_rows_and_parser_failures(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7194-ZERO-GAP."""

    valid = _write_completion(
        tmp_path,
        "valid.txt",
        "<tool_call>\n<function=query_region>\n<parameter=transition_index>0</parameter>\n"
        "<parameter=r0>0</parameter><parameter=c0>0</parameter>"
        "<parameter=r1>1</parameter><parameter=c1>1</parameter>\n</tool_call>",
    )
    invalid = _write_completion(
        tmp_path,
        "invalid.txt",
        "<tool_call>\n<function=query_region>\n<parameter=transition_index>0</parameter>",
    )
    valid["induction_attempt_index"] = 0
    invalid["induction_attempt_index"] = 1

    rows = recompute_gap_rows(
        root=tmp_path,
        run_row=_run_row(),
        completions=[valid, invalid],
        tool_gap_manifest={"rows": []},
        upstream_tool_rows=[],
        seed=7_193_001,
        source_hashes={"run": "sha256:run", "gaps": "sha256:gaps"},
    )

    assert len(rows) == 2
    assert rows[0]["parsed_tool_calls"] == 1
    assert rows[0]["tool_calls_by_name"] == {"query_region": 1}
    assert rows[0]["parser_failures"] == 0
    assert rows[0]["gap_event_count"] == 0
    assert rows[0]["gap_capture_state"] == "capture_complete"
    assert rows[1]["parsed_tool_calls"] == 0
    assert rows[1]["parser_failures"] == 1
    assert rows[1]["gap_event_count"] == 0
    assert rows[1]["gap_capture_state"] == "capture_not_recorded"
    assert rows[0]["induction_id"] != rows[1]["induction_id"]


def test_candidate_requires_call_failure_and_exact_counterexample(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7194 rejects a gap inferred only from prose."""

    call = _write_completion(
        tmp_path,
        "missing.txt",
        "<tool_call>\n<function=missing_tool>\n<parameter=item>3</parameter>\n</tool_call>",
    )
    call["induction_attempt_index"] = 0
    run_row = _run_row()
    attempt = run_row["policy_diagnostics"]["induction_attempts"][0]  # type: ignore[index]
    attempt["tool_gap"]["tool_gap_events"] = [  # type: ignore[index]
        {
            "kind": "unknown_tool",
            "requested_tool": "missing_tool",
            "error": "unknown tool: missing_tool",
            "runtime_counterexample": {"arguments": {"item": 3}, "result": "dispatch_rejected"},
        }
    ]
    rows = recompute_gap_rows(
        root=tmp_path,
        run_row=run_row,
        completions=[call],
        tool_gap_manifest={"rows": []},
        upstream_tool_rows=[],
        seed=7_193_001,
        source_hashes={},
    )

    recommendation = generalization_recommendation(rows)

    assert recommendation["kind"] == "reusable_tool_candidate"
    assert recommendation["candidate_name"] == "missing_tool"
    without_counterexample = deepcopy(rows)
    without_counterexample[0]["gap_events"][0].pop("runtime_counterexample")
    assert generalization_recommendation(without_counterexample)["kind"] == "honest_no_gap"


def test_banked_progress_excludes_shadow_and_shared_help_credit() -> None:
    """SCENARIO-ARC-WMTE-7194-BANKED-CREDIT."""

    rows = recompute_banked_credit_rows(_run_row(), seed=7_193_001)
    banked = [row for row in rows if row["row_kind"] == "banked_level_transition"]
    shadow = [row for row in rows if row["row_kind"] == "shadow_supervisor_context"]

    assert [row["to_level"] for row in banked] == [1, 2]
    assert [row["charged_action_index"] for row in banked] == [8, 12]
    assert all(row["banked_progress"] is True for row in banked)
    assert len(shadow) == 2
    assert all(row["promoted_banked_credit"] == 0 for row in shadow)
    assert all(row["shared_supervisor_credit"] is True for row in shadow)


def test_session_costs_use_completion_and_run_receipts(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7194 recomputes the measured session cost."""

    first = _write_completion(tmp_path, "a.txt", "plain")
    second = _write_completion(tmp_path, "b.txt", "plain")
    first["induction_attempt_index"] = 0
    second["induction_attempt_index"] = 1
    gaps = [
        {"attempt_index": 0, "induction_id": "i0", "parsed_tool_calls": 2, "parser_failures": 0},
        {"attempt_index": 1, "induction_id": "i1", "parsed_tool_calls": 0, "parser_failures": 1},
    ]

    costs = recompute_session_cost_rows(_run_row(), [first, second], gaps)

    assert costs[-1] == {
        "row_kind": "session_total",
        "game": "r11l",
        "actions": 18,
        "resets": 2,
        "charged_actions": 20,
        "banked_levels": 2,
        "completion_count": 2,
        "completion_tokens": 14,
        "parsed_tool_calls": 2,
        "parser_failures": 1,
        "generator_wall_s": 10.0,
        "session_wall_s": 12.5,
        "paired_efficacy_estimate": None,
    }


def test_complete_and_blocked_artifacts_validate() -> None:
    """SCENARIO-ARC-WMTE-7194-BLOCKED and the complete terminal contract."""

    base = {
        "run_date": "20260910",
        "duration_s": 1.25,
        "preconditions_checked": [],
        "source_artifact_hashes": {"upstream": "sha256:abc"},
        "gap_rows": [],
        "banked_credit_rows": [],
        "session_cost_rows": [],
        "refinement_tool_runs": [],
        "historical_context": {"counts_toward_new_volume": False},
    }
    complete = build_artifact(**base, blocked=False)
    assert complete["MODEL_SPECS"] == MODEL_SPECS == []
    assert complete["model_invoked"] is False
    assert complete["arc_gap_audit_complete_score"] == 1
    assert complete["verdict_class"] == "null"
    assert set(complete["field_principles"]) == set(complete)
    assert complete["field_principles"]["field_principles"] == FIELD_PRINCIPLES["field_principles"]
    assert validate_artifact(complete) == []

    failed = {
        "check": "raw_manifest",
        "upstream": "results/raw/missing.json",
        "field": "file",
        "expected_value": "present",
        "observed_value": "missing",
        "passed": False,
    }
    blocked = build_artifact(**{**base, "preconditions_checked": [failed]}, blocked=True)
    assert blocked["arc_gap_audit_complete_score"] == 0
    assert blocked["verdict_class"] == "blocked"
    assert blocked["honest_verdict"].startswith("blocked_")
    assert blocked["inference_substrate_class"] == "blocked_no_run"
    assert blocked["gate_check_summary"]["upstream"] == "results/raw/missing.json"
    assert validate_artifact(blocked) == []

    changed = deepcopy(complete)
    changed["model_invoked"] = True
    changed["reproducibility_checksum"] = artifact_checksum(changed)
    assert "model_invoked_must_be_false" in validate_artifact(changed)


def test_fixture_helpers_emit_valid_json(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7194 rows remain JSON serializable for the terminal artifact."""

    completion = _write_completion(tmp_path, "call.txt", "plain")
    completion["induction_attempt_index"] = 0
    rows = recompute_gap_rows(
        root=tmp_path,
        run_row=_run_row(),
        completions=[completion],
        tool_gap_manifest={"rows": []},
        upstream_tool_rows=[],
        seed=7_193_001,
        source_hashes={},
    )
    assert json.loads(json.dumps(rows)) == rows

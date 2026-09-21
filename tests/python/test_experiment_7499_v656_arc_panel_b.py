"""Tests for the frozen V656 ARC Panel B observation.

Spec refs: REQ-ARC-WMTE-7499 and SCENARIO-ARC-WMTE-7499-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7499_v656_arc_panel_b as panel


REPO = Path(__file__).resolve().parents[2]


def _episode(sealed: dict, *, disposition: str = "complete") -> dict:
    """Build one small row without pretending that it is live evidence."""

    return {
        **deepcopy(sealed),
        "disposition": disposition,
        "action_count": 2,
        "start_level": 0,
        "peak_level": 0,
        "terminal_level": 0,
        "action_rows": [
            {
                "action_index": 1,
                "level": 0,
                "later_progress": False,
                "interval_start_monotonic_ns": 1_000,
                "interval_end_monotonic_ns": 1_400,
            },
            {
                "action_index": 2,
                "level": 0,
                "later_progress": False,
                "interval_start_monotonic_ns": 1_500,
                "interval_end_monotonic_ns": 2_000,
            },
        ],
        "request_budget_receipt": {
            "attempted": 1,
            "completed": 1,
            "failed": 0,
            "cancelled": 0,
            "in_flight": 0,
        },
        "trace_reproduction": {"attempted": False, "passed": False},
        "solve_provenance": "no_level_reached",
        "elapsed_s": 0.000001,
        "new_level_credit": 0,
        "error": None,
    }


def _events(episode_id: str) -> list[dict]:
    """Create duplicate and nested spans plus one supervisor selection."""

    common = {
        "run_id": panel.EXPERIMENT_ID,
        "process_id": 42,
        "episode_id": episode_id,
        "clock_identity": "time.monotonic_ns",
        "decision_id": "choice",
        "parent_decision_id": None,
        "seam": "supervisor_arm_selection",
        "work_class": "replaceable_decision",
    }
    return [
        {
            **common,
            "event": "stage_start",
            "event_monotonic_ns": 1_050,
            "interval_start_monotonic_ns": 1_050,
        },
        {
            **common,
            "event": "eligible_candidate_set",
            "event_monotonic_ns": 1_075,
            "candidate_ids": ["no_redirect", "allow_reinduction"],
        },
        {
            **common,
            "event": "selection",
            "event_monotonic_ns": 1_100,
            "selected_candidate_ids": ["no_redirect"],
            "supervisor_fired": False,
            "applied_redirection": False,
        },
        {
            **common,
            "event": "stage_end",
            "event_monotonic_ns": 1_200,
            "interval_end_monotonic_ns": 1_200,
            "disposition": "completed",
        },
        {
            **common,
            "event": "stage_start",
            "event_monotonic_ns": 1_050,
            "interval_start_monotonic_ns": 1_050,
        },
        {
            **common,
            "event": "stage_end",
            "event_monotonic_ns": 1_200,
            "interval_end_monotonic_ns": 1_200,
            "disposition": "completed",
        },
        {
            **common,
            "decision_id": "generation",
            "seam": "downstream_generation",
            "work_class": "text_generation",
            "event": "stage_start",
            "event_monotonic_ns": 1_100,
            "interval_start_monotonic_ns": 1_100,
        },
        {
            **common,
            "decision_id": "generation",
            "seam": "downstream_generation",
            "work_class": "text_generation",
            "event": "stage_end",
            "event_monotonic_ns": 1_300,
            "interval_end_monotonic_ns": 1_300,
            "input_tokens": 12,
            "output_tokens": 3,
            "disposition": "completed",
        },
    ]


def test_req_arc_wmte_7499_spec_and_exact_schedule() -> None:
    """REQ-ARC-WMTE-7499 seals only the original eighteen Panel B units."""

    text = (REPO / panel.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("REQ-ARC-WMTE-7499") : text.index("REQ-ARC-WMTE-7471")]
    for anchor in (
        "SCENARIO-ARC-WMTE-7499-PRECONDITIONS",
        "SCENARIO-ARC-WMTE-7499-SCHEDULE",
        "SCENARIO-ARC-WMTE-7499-INTERVALS",
        "SCENARIO-ARC-WMTE-7499-SUPERVISOR",
        "SCENARIO-ARC-WMTE-7499-COMPARABILITY",
        "SCENARIO-ARC-WMTE-7499-TERMINAL",
    ):
        assert anchor in section
    assert panel.MODEL_SPECS == ["unsloth/Qwen3.8-27B-GGUF"]
    assert panel.INFERENCE_SUBSTRATE_CLASS == "model_bounded_generation"
    schedule = panel.build_panel_schedule()
    assert [row["game"] for row in schedule[::3]] == [
        "tu93",
        "g50t",
        "tn36",
        "vc33",
        "re86",
        "dc22",
    ]
    assert len(schedule) == 18
    assert {row["seed"] for row in schedule} == {65501, 65502, 65503}
    assert all(row["panel"] == "B" and row["adapter_disabled"] for row in schedule)
    assert all(row["action_limit"] == 180 for row in schedule)
    assert all(row["request_limit"] == 2 for row in schedule)
    assert all(row["max_new_tokens_per_call"] == 256 for row in schedule)
    assert all(row["episode_limit_s"] == 240 for row in schedule)
    assert all(row["panel_live_limit_s"] == 3600 for row in schedule)


def test_scenario_arc_wmte_7499_intervals_and_supervisor_rows() -> None:
    """SCENARIO-7499-INTERVALS/SUPERVISOR keep exclusive cost and zero rows."""

    schedule = panel.build_panel_schedule()
    events = _events(schedule[0]["episode_id"])
    reduced = panel.reduce_panel(schedule, [_episode(schedule[0])], events)
    first = reduced["rows"][0]
    assert first["exclusive_cost"]["bounds_valid"] is True
    assert first["exclusive_cost"]["duplicate_event_count"] == 2
    assert first["exclusive_cost"]["replaceable_upper_ns"] <= 1_000
    assert first["request_tokens"] == {"input": 12, "output": 3}
    assert len(reduced["supervisor_opportunity_rows"]) == 18
    opportunity = reduced["supervisor_opportunity_rows"][0]
    assert opportunity["opportunity_count"] == 1
    assert opportunity["eligible_arms"] == ["allow_reinduction", "no_redirect"]
    assert opportunity["selected_arms"] == ["no_redirect"]
    assert opportunity["abstention_count"] == 1
    assert opportunity["no_selector_change"] is True
    assert reduced["supervisor_opportunity_rows"][1]["opportunity_count"] == 0


def test_scenario_arc_wmte_7499_comparability_stratifies_code_drift() -> None:
    """SCENARIO-7499-COMPARABILITY forbids pooling after any identity drift."""

    panel_a = panel.load_object(REPO / panel.PANEL_A_PATH)
    model_specs = panel_a["model_specs"]
    current = panel.current_comparability_inputs(REPO, model_specs, panel_a["execution_venue_details"])
    receipt = panel.compare_panel_a(panel_a, current)
    assert receipt["comparable"] is False
    assert receipt["pooling_allowed"] is False
    assert "panel_runner_code" in receipt["mismatched_identities"]
    assert receipt["pooled_episode_count"] is None
    matched = panel.compare_panel_a(panel_a, receipt["panel_a_identities"])
    assert matched["comparable"] is True
    assert matched["pooled_episode_count"] == 36


def test_req_arc_wmte_7499_fixture_reduces_required_fields() -> None:
    """REQ-ARC-WMTE-7499 independently rebuilds every terminal headline row."""

    artifact = panel.build_artifact_for_test()
    assert panel.validate_artifact(artifact, require_terminal=False) == []
    assert panel.independent_reduce(artifact)["matches_declared"] is True
    assert artifact["arc_panel_b_complete_score"] == 1
    assert artifact["arc_support_score"] == {"complete_episodes": 18, "complete_games": 6}
    assert artifact["panel_comparability"]["pooling_allowed"] is False
    assert artifact["pooled_estimate"] is None
    assert len(artifact["per_game_results"]) == 6
    assert len(artifact["supervisor_opportunity_rows"]) == 18
    assert artifact["verdict_class"] == "null"
    assert artifact["scientific_benefit_score"] == 0
    assert len(json.dumps(artifact)) < 20 * 1024 * 1024
    assert set(artifact).issubset(artifact["field_principles"])
    assert all(row["principle"] for row in artifact["acceptance_gate_results"])

    changed = deepcopy(artifact)
    changed["supervisor_opportunity_rows"][0]["abstention_count"] += 1
    assert "independent_reduction_mismatch" in panel.validate_artifact(
        changed, require_terminal=False
    )


def test_scenario_arc_wmte_7499_preconditions_preserve_absence_and_registry() -> None:
    """SCENARIO-7499-PRECONDITIONS keeps Exp7486 absence external and typed."""

    checks, hashes, protocol, panel_a, registry = panel.collect_preconditions(
        REPO, force_live="1"
    )
    assert all(row["passed"] is True for row in checks)
    assert protocol["arc_interval_protocol_ready_score"] == 1
    assert panel_a["arc_panel_a_complete_score"] == 1
    assert hashes[panel.PROTOCOL_PATH.as_posix()]["original_flags"]["verdict_class"] == "null"
    assert registry["policy_received_registry_data"] is False
    assert registry["new_credit_allowed"] is False
    assert [row["game"] for row in registry["rows"]] == list(panel.PANEL_GAMES)
    assert all(isinstance(row["levels_reproduced"], int) for row in registry["rows"])
    absence = next(row for row in checks if row["check"] == "exp7486_external_absence")
    assert absence["passed"] is True
    assert absence["observed"] == "absent"


def test_scenario_arc_wmte_7499_terminal_scope_and_replay(tmp_path: Path) -> None:
    """SCENARIO-7499-TERMINAL freezes exact checks and fresh-process readers."""

    plan = panel.build_validation_plan(REPO, tmp_path / "private")
    assert panel.validate_validation_plan(REPO, plan) == []
    assert [row.name for row in plan] == [
        *panel.validation_scope.REQUIRED_CHECK_NAMES,
        "e2e_009",
        "e2e_010",
        "e2e_011",
        "private_arc_smoke",
    ]
    terminal = panel.terminal_command_specs(REPO, tmp_path / "candidate.json")
    assert [row.name for row in terminal] == list(panel.REQUIRED_TERMINAL_NAMES)

    artifact = panel.build_artifact_for_test()
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert panel.main(["--replay", str(path), "--reduce-only"]) == 0
    for field in panel.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact

    broken = deepcopy(artifact)
    broken["verdict_class"] = "positive"
    assert "oracle_positive_forbidden" in panel.validate_artifact(broken, require_terminal=False)
    with pytest.raises(SystemExit):
        panel.parse_args([])

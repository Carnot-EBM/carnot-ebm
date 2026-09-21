"""Tests for the V656 cross-game ARC opportunity audit."""

from __future__ import annotations

from copy import deepcopy

import pytest

from carnot import experiment_7500_v656_arc_opportunity_audit as audit


def _event(
    *,
    decision: str,
    event: str,
    start: int | None = None,
    end: int | None = None,
    seam: str = "supervisor_arm_selection",
    **extra: object,
) -> dict[str, object]:
    row: dict[str, object] = {
        "run_id": "run-a",
        "process_id": 17,
        "episode_id": "panel-a:g0:seed-1",
        "decision_id": decision,
        "seam": seam,
        "event": event,
        "clock_identity": "time.monotonic_ns",
        "work_class": "replaceable_decision",
        **extra,
    }
    if start is not None:
        row["interval_start_monotonic_ns"] = start
        row["event_monotonic_ns"] = start
    if end is not None:
        row["interval_end_monotonic_ns"] = end
        row["event_monotonic_ns"] = end
        row["disposition"] = "completed"
    return row


def _episode_events() -> list[dict[str, object]]:
    return [
        _event(decision="outer", event="stage_start", start=0, seam="candidate_action_selection"),
        _event(decision="unknown", event="stage_start", start=100),
        _event(
            decision="unknown",
            event="eligible_candidate_set",
            candidates=[{"stable_candidate_id": "force_exploration_diversity", "eligibility": None}],
        ),
        _event(
            decision="unknown",
            event="selection",
            selected_candidate_ids=["force_exploration_diversity"],
            supervisor_fired=True,
            applied_redirection=False,
        ),
        _event(decision="unknown", event="stage_end", end=200),
        _event(decision="eligible", event="stage_start", start=300),
        _event(
            decision="eligible",
            event="eligible_candidate_set",
            candidates=[{"stable_candidate_id": "drop_goal_bias", "eligibility": True}],
        ),
        _event(
            decision="eligible",
            event="selection",
            selected_candidate_ids=["drop_goal_bias"],
            supervisor_fired=True,
            applied_redirection=True,
            runtime_observation_reproducible=True,
        ),
        _event(decision="eligible", event="stage_end", end=400),
        _event(decision="outer", event="stage_end", end=500, seam="candidate_action_selection"),
    ]


# REQ-ARC-WMTE-7500; SCENARIO-ARC-WMTE-7500-EXCLUSIVE-UNION
# SCENARIO-ARC-WMTE-7500-SUPERVISOR-ELIGIBILITY
def test_episode_reduction_unions_spans_and_requires_explicit_eligibility() -> None:
    events = _episode_events()
    events.append(deepcopy(events[5]))
    row = audit.reduce_episode_row(
        {
            "episode_id": "panel-a:g0:seed-1",
            "game": "g0",
            "seed": 1,
            "panel": "A",
            "disposition": "complete",
            "episode_start_ns": 0,
            "episode_end_ns": 1_000,
            "actions_to_progress_censored": True,
            "solve_provenance": "no_level_reached",
        },
        events,
    )

    assert row["accounting"]["stage_union_ns"] == 500
    assert row["accounting"]["duplicate_event_count"] == 1
    assert row["accounting"]["bounds_valid"] is True
    assert row["supervisor"]["selected_arm_counts"] == {
        "drop_goal_bias": 1,
        "force_exploration_diversity": 1,
    }
    assert row["supervisor"]["unknown_eligibility_selection_count"] == 1
    assert row["supervisor"]["triggered_opportunity_count"] == 1
    assert row["eligible_service_lower_ns"] == 100
    assert row["eligible_service_upper_ns"] == 100

    no_runtime = deepcopy(events)
    no_runtime[7]["runtime_observation_reproducible"] = False
    reduced = audit.reduce_episode_row(row, no_runtime)
    assert reduced["supervisor"]["triggered_opportunity_count"] == 0
    assert reduced["eligible_service_upper_ns"] == 0


# REQ-ARC-WMTE-7500; SCENARIO-ARC-WMTE-7500-EXCLUSIVE-UNION
def test_incomplete_and_mismatched_intervals_do_not_inflate_eligible_time() -> None:
    events = _episode_events()
    events[8]["clock_identity"] = "other-clock"
    events.extend(
        [
            _event(decision="unfinished", event="stage_start", start=600),
            _event(
                decision="unfinished",
                event="eligible_candidate_set",
                candidates=[{"stable_candidate_id": "allow_reinduction", "eligibility": True}],
            ),
            _event(
                decision="unfinished",
                event="selection",
                selected_candidate_ids=["allow_reinduction"],
                runtime_observation_reproducible=True,
            ),
        ]
    )
    row = audit.reduce_episode_row(
        {
            "episode_id": "panel-a:g0:seed-1",
            "game": "g0",
            "seed": 1,
            "panel": "A",
            "disposition": "complete",
            "episode_start_ns": 0,
            "episode_end_ns": 1_000,
        },
        events,
    )

    assert row["accounting"]["mismatched_clock_count"] == 1
    assert row["accounting"]["incomplete_interval_count"] == 1
    assert row["eligible_service_lower_ns"] == 0
    assert row["eligible_service_upper_ns"] == 0


def _panel(name: str, games: range, *, identity: str = "same") -> dict[str, object]:
    rows = [
        {
            "episode_id": f"panel-{name.lower()}:g{game}:seed-{seed}",
            "game": f"g{game}",
            "seed": seed,
            "disposition": "complete",
            "accounting": {"bounds_valid": True},
            "eligible_service_upper_ns": 1,
            "supervisor": {"triggered_opportunity_count": 1},
        }
        for game in games
        for seed in (1, 2, 3)
    ]
    return {
        "panel": name,
        "disposition": "available_valid",
        "identities": {
            "observer": identity,
            "policy": identity,
            "model": identity,
            "protocol": identity,
        },
        "rows": rows,
    }


# REQ-ARC-WMTE-7500; SCENARIO-ARC-WMTE-7500-POOLING
@pytest.mark.parametrize(
    ("panels", "expected", "reason"),
    [
        ([_panel("A", range(5)), _panel("B", range(5, 10))], True, "supported"),
        ([_panel("A", range(5)), _panel("B", range(5, 10), identity="drift")], False, "identity_drift"),
        ([_panel("A", range(5)), _panel("B", range(5, 9))], False, "support_below_floor"),
        ([_panel("A", range(5)), {"panel": "B", "disposition": "blocked_absent", "rows": []}], False, "panel_unavailable"),
    ],
)
def test_pooling_requires_identity_and_episode_game_support(
    panels: list[dict[str, object]], expected: bool, reason: str
) -> None:
    result = audit.pooling_assessment(panels)
    assert result["pooled_support"] is expected
    assert result["reason"] == reason
    assert result["episode_floor"] == 30
    assert result["game_floor"] == 10


# REQ-ARC-WMTE-7500; SCENARIO-ARC-WMTE-7500-MISSING-PANEL
# SCENARIO-ARC-WMTE-7500-ZERO-OPPORTUNITY
def test_missing_panel_b_stays_blocked_without_fabricated_rows() -> None:
    artifact = audit.build_artifact_for_test()

    assert artifact["panel_dispositions"][1] == {
        "panel": "B",
        "path": "results/experiment_7499_v656_arc_panel_b.json",
        "disposition": "blocked_absent",
        "valid_episode_count": 0,
        "principle": "An absent producer cannot become eighteen zero-valued observations.",
    }
    assert len(artifact["rows"]) == 18
    assert artifact["sample_size_budget"]["external_missing_planned_units"] == 18
    assert artifact["sample_size_budget"]["unstarted_independent_units"] == 0
    assert artifact["pooling_assessment"]["pooled_support"] is False
    assert artifact["supervisor_disposition"]["triggered_opportunity_count"] == 0
    assert artifact["intervention_ledger"] == []
    assert artifact["efficacy_estimate"] is None
    assert artifact["amdahl_upper_bound"]["speedup_upper_bound"] == 1.0
    assert artifact["arc_opportunity_audit_complete_score"] == 1
    assert artifact["opportunity_present_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("complete_")
    assert audit.validate_artifact(artifact, verify_files=False, require_terminal=True) == []


# REQ-ARC-WMTE-7500; SCENARIO-ARC-WMTE-7500-TERMINAL
def test_artifact_validation_rejects_forged_opportunity_and_contract_drift() -> None:
    artifact = audit.build_artifact_for_test()

    cases = []
    substrate = deepcopy(artifact)
    substrate["inference_substrate"] = "live_llm_inference"
    cases.append((substrate, "current_provenance_invalid"))
    model = deepcopy(artifact)
    model["model_invoked"] = True
    cases.append((model, "current_provenance_invalid"))
    principle = deepcopy(artifact)
    principle["field_principles"].pop("schema")
    cases.append((principle, "field_principles_incomplete"))
    gate = deepcopy(artifact)
    gate["acceptance_gate_results"][0].pop("principle")
    cases.append((gate, "gate_contract_invalid"))
    forged = deepcopy(artifact)
    forged["opportunity_present_score"] = 1
    cases.append((forged, "opportunity_score_mismatch"))
    ledger = deepcopy(artifact)
    ledger["intervention_ledger"] = [{"arm": "invented"}]
    cases.append((ledger, "empty_opportunity_ledger_invalid"))
    oracle = deepcopy(artifact)
    oracle["verifier_is_oracle"] = True
    oracle["verdict_class"] = "positive"
    cases.append((oracle, "oracle_positive_forbidden"))
    checksum = deepcopy(artifact)
    checksum["status"] = "mutated"
    cases.append((checksum, "reproducibility_checksum_mismatch"))

    for candidate, error in cases:
        assert error in audit.validate_artifact(
            candidate, verify_files=False, require_terminal=True
        )


# REQ-ARC-WMTE-7500; SCENARIO-ARC-WMTE-7500-MISSING-PANEL
def test_roadmap_resolution_uses_exact_exp7499_deliverable() -> None:
    roadmap = {
        "milestone": "2026.09.656",
        "tasks": [
            {
                "id": "exp7499-arc-panel-b",
                "deliverable": "results/experiment_7499_v656_arc_panel_b.json",
            }
        ],
    }
    assert audit.resolve_panel_b_deliverable(roadmap) == audit.PANEL_B_PATH

    wrong = deepcopy(roadmap)
    wrong["tasks"][0]["deliverable"] = "results/guessed.json"
    with pytest.raises(ValueError, match="panel_b_deliverable_invalid"):
        audit.resolve_panel_b_deliverable(wrong)


# REQ-ARC-WMTE-7500; SCENARIO-ARC-WMTE-7500-TERMINAL
def test_terminal_classification_separates_accounting_from_opportunity() -> None:
    assert audit.classify_terminal(
        accounting_valid=True, panel_b_disposition="blocked_absent", opportunity_present=False
    ) == {
        "arc_opportunity_audit_complete_score": 1,
        "opportunity_present_score": 0,
        "verdict_class": "blocked",
        "honest_verdict": "complete_blocked_panel_b_absent_no_pooled_opportunity",
    }
    assert audit.classify_terminal(
        accounting_valid=True, panel_b_disposition="available_valid", opportunity_present=False
    )["verdict_class"] == "null"
    assert audit.classify_terminal(
        accounting_valid=False, panel_b_disposition="available_invalid", opportunity_present=False
    )["verdict_class"] == "disqualified"

"""Test the frozen Panel B live observation and its independent evidence checks.

Spec refs: REQ-ARC-WMTE-7499 and SCENARIO-ARC-WMTE-7499-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7499_v656_arc_panel_b as panel


REPO = Path(__file__).resolve().parents[2]


def _episode(sealed: dict, *, disposition: str = "complete", progressed: bool = False) -> dict:
    """Build one measured-looking row without replacing the production reducer."""

    return {
        **deepcopy(sealed),
        "disposition": disposition,
        "action_count": 2,
        "start_level": 0,
        "peak_level": 1 if progressed else 0,
        "terminal_level": 1 if progressed else 0,
        "action_rows": [
            {
                "action_index": 1,
                "level": 0,
                "later_progress": progressed,
                "interval_start_monotonic_ns": 1_000,
                "interval_end_monotonic_ns": 1_400,
            },
            {
                "action_index": 2,
                "level": 1 if progressed else 0,
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
        "trace_reproduction": {"attempted": progressed, "passed": progressed},
        "solve_provenance": "live_agent_self_discovery" if progressed else "no_level_reached",
        "elapsed_s": 0.000001,
        "new_level_credit": 0,
        "error": None,
    }


def _events(episode_id: str) -> list[dict]:
    """Expose one eligible selection, one abstention, and duplicate nested work."""

    common = {
        "run_id": panel.EXPERIMENT_ID,
        "process_id": 42,
        "episode_id": episode_id,
        "clock_identity": "time.monotonic_ns",
        "parent_decision_id": None,
        "seam": "supervisor_arm_selection",
        "work_class": "replaceable_decision",
    }
    return [
        {
            **common,
            "decision_id": "choice-1",
            "event": "stage_start",
            "event_monotonic_ns": 1_050,
            "interval_start_monotonic_ns": 1_050,
        },
        {
            **common,
            "decision_id": "choice-1",
            "event": "selection",
            "event_monotonic_ns": 1_100,
            "eligible_arms": ["force_exploration_diversity"],
            "chosen_arm": "force_exploration_diversity",
            "selected_candidate_ids": ["force_exploration_diversity"],
            "supervisor_fired": True,
            "applied_redirection": False,
        },
        {
            **common,
            "decision_id": "choice-1",
            "event": "stage_end",
            "event_monotonic_ns": 1_200,
            "interval_end_monotonic_ns": 1_200,
            "disposition": "completed",
        },
        {
            **common,
            "decision_id": "choice-1",
            "event": "stage_start",
            "event_monotonic_ns": 1_050,
            "interval_start_monotonic_ns": 1_050,
        },
        {
            **common,
            "decision_id": "choice-2",
            "event": "selection",
            "event_monotonic_ns": 1_300,
            "eligible_arms": [],
            "chosen_arm": "no_redirect",
            "selected_candidate_ids": ["no_redirect"],
            "supervisor_fired": False,
            "applied_redirection": False,
        },
    ]


def test_req_arc_wmte_7499_spec_and_frozen_panel_b_schedule() -> None:
    """REQ-ARC-WMTE-7499 seals the exact missing half before outcomes exist."""

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
    assert all(row["panel"] == "B" for row in schedule)
    assert all(row["action_limit"] == 180 for row in schedule)
    assert all(row["request_limit"] == 2 for row in schedule)
    assert all(row["max_new_tokens_per_call"] == 256 for row in schedule)
    assert all(row["episode_limit_s"] == 240 for row in schedule)
    assert all(row["panel_live_limit_s"] == 3600 for row in schedule)


def test_scenario_arc_wmte_7499_intervals_and_supervisor_rows() -> None:
    """SCENARIO-ARC-WMTE-7499-INTERVALS/SUPERVISOR retain exclusive choices."""

    schedule = panel.build_panel_schedule()
    episode = _episode(schedule[0])
    events = _events(schedule[0]["episode_id"])
    reduced = panel.reduce_panel(schedule, [episode], events)
    first = reduced["rows"][0]
    assert first["exclusive_cost"]["complete_union_ns"] == 150
    assert first["exclusive_cost"]["bounds_valid"] is True
    assert first["exclusive_cost"]["duplicate_event_count"] == 1
    assert reduced["rows"][1]["disposition"] == "unstarted"
    assert reduced["sample_size_budget"]["planned_independent_units"] == 18

    opportunity_rows = reduced["supervisor_opportunity_rows"]
    assert opportunity_rows[0] == {
        "episode_id": schedule[0]["episode_id"],
        "game": "tu93",
        "seed": 65501,
        "decision_id": "choice-1",
        "eligible_arms": ["force_exploration_diversity"],
        "selected_arm": "force_exploration_diversity",
        "abstained": False,
        "applied": False,
    }
    assert opportunity_rows[1]["selected_arm"] == "no_redirect"
    assert opportunity_rows[1]["abstained"] is True
    zero_row = next(row for row in opportunity_rows if row["episode_id"] == schedule[1]["episode_id"])
    assert zero_row["decision_id"] is None
    assert zero_row["eligible_arms"] == []
    assert zero_row["selected_arm"] is None
    assert zero_row["abstained"] is True


def test_scenario_arc_wmte_7499_preconditions_preserve_absence_and_registry() -> None:
    """SCENARIO-ARC-WMTE-7499-PRECONDITIONS authenticates inputs without laundering absence."""

    checks, hashes, upstream, registry, absent = panel.collect_preconditions(
        REPO, force_live="1"
    )
    assert all(row["passed"] is True for row in checks)
    assert upstream["arc_interval_protocol_ready_score"] == 1
    assert upstream["timing_correction"]["future_protocol_field_sufficiency"] is True
    assert absent == {
        "path": "results/experiment_7486_v655_arc_cost_panel_b.json",
        "exists": False,
        "classification": "external_absence_not_scientific_null",
    }
    assert len(registry["rows"]) == 6
    assert all(isinstance(row["levels_reproduced"], int) for row in registry["rows"])
    assert registry["policy_received_registry_data"] is False
    assert hashes[panel.UPSTREAM_PATH.as_posix()]["original_flags"]["flagged_adversarial"] is False


def test_scenario_arc_wmte_7499_comparability_is_hash_driven() -> None:
    """SCENARIO-ARC-WMTE-7499-COMPARABILITY forbids pooling after identity drift."""

    panel_a = panel.load_object(REPO / panel.PANEL_A_PATH)
    current = panel.current_comparability_identity(REPO, panel_a["model_specs"])
    same = panel.compare_panel_identity(panel_a, current)
    assert same["comparable"] is True
    assert same["pooling_allowed"] is True
    assert all(row["principle"] for row in same["checks"])

    changed = deepcopy(current)
    changed["code_hashes"]["python/carnot/agentic/arc_competition_agent.py"] = "sha256:changed"
    drift = panel.compare_panel_identity(panel_a, changed)
    assert drift["comparable"] is False
    assert drift["pooling_allowed"] is False
    assert drift["stratify_by_version"] is True
    assert "code:python/carnot/agentic/arc_competition_agent.py" in drift["failed_checks"]


def test_req_arc_wmte_7499_fixture_artifact_reduces_and_rejects_mutation() -> None:
    """REQ-ARC-WMTE-7499 recomputes every headline field from unit rows."""

    artifact = panel.build_artifact_for_test()
    assert panel.validate_artifact(artifact, require_terminal=False) == []
    assert panel.independent_reduce(artifact)["matches_declared"] is True
    assert artifact["arc_panel_b_complete_score"] == 1
    assert artifact["arc_support_score"] == {
        "completed_episode_units": 18,
        "planned_episode_units": 18,
        "completed_game_clusters": 6,
        "planned_game_clusters": 6,
    }
    assert artifact["panel_comparability"]["comparable"] is True
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["hidden_game_efficacy_claim"] is False
    assert set(artifact).issubset(artifact["field_principles"])
    assert all(row["principle"] for row in artifact["acceptance_gate_results"])

    changed = deepcopy(artifact)
    changed["supervisor_opportunity_rows"][0]["abstained"] = False
    assert "independent_reduction_mismatch" in panel.validate_artifact(
        changed, require_terminal=False
    )


def test_scenario_arc_wmte_7499_terminal_scope_and_cold_replay(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7499-TERMINAL freezes scoped checks and exact readers."""

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
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(artifact), encoding="utf-8")
    assert panel.main(["--replay", str(candidate), "--reduce-only"]) == 0
    for field in panel.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact

    wrong = deepcopy(artifact)
    wrong["verdict_class"] = "positive"
    assert "oracle_positive_forbidden" in panel.validate_artifact(wrong, require_terminal=False)
    with pytest.raises(SystemExit):
        panel.parse_args([])

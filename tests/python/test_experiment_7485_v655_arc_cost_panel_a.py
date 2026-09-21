"""Tests for the V655 live ARC cost Panel A capture.

Spec refs: REQ-ARC-WMTE-7485 and SCENARIO-ARC-WMTE-7485-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7485_v655_arc_cost_panel_a as panel


REPO = Path(__file__).resolve().parents[2]


def _episode(sealed: dict, *, progressed: bool = False, disposition: str = "complete") -> dict:
    start = 1_000
    end = 2_000
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
                "interval_start_monotonic_ns": start,
                "interval_end_monotonic_ns": 1_400,
            },
            {
                "action_index": 2,
                "level": 1 if progressed else 0,
                "later_progress": False,
                "interval_start_monotonic_ns": 1_500,
                "interval_end_monotonic_ns": end,
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
    common = {
        "run_id": panel.EXPERIMENT_ID,
        "process_id": 42,
        "episode_id": episode_id,
        "clock_identity": "time.monotonic_ns",
    }

    def span(decision: str, seam: str, work_class: str, start: int, end: int) -> list[dict]:
        return [
            {
                **common,
                "decision_id": decision,
                "parent_decision_id": None,
                "seam": seam,
                "work_class": work_class,
                "event": "stage_start",
                "event_monotonic_ns": start,
                "interval_start_monotonic_ns": start,
            },
            {
                **common,
                "decision_id": decision,
                "parent_decision_id": None,
                "seam": seam,
                "work_class": work_class,
                "event": "stage_end",
                "event_monotonic_ns": end,
                "interval_end_monotonic_ns": end,
                "disposition": "completed",
            },
        ]

    return [
        *span("choice", "supervisor_arm_selection", "replaceable_decision", 1_050, 1_200),
        {
            **common,
            "decision_id": "choice",
            "parent_decision_id": None,
            "seam": "supervisor_arm_selection",
            "work_class": "replaceable_decision",
            "event": "selection",
            "event_monotonic_ns": 1_150,
            "selected_candidate_ids": ["no_redirect"],
            "supervisor_fired": False,
            "applied_redirection": False,
        },
        *span("generation", "downstream_generation", "text_generation", 1_100, 1_300),
        {
            **common,
            "decision_id": "generation",
            "parent_decision_id": None,
            "seam": "downstream_generation",
            "work_class": "text_generation",
            "event": "request_usage",
            "event_monotonic_ns": 1_300,
            "input_tokens": 12,
            "output_tokens": 3,
            "disposition": "completed",
        },
    ]


def test_req_arc_wmte_7485_spec_and_exact_schedule() -> None:
    """REQ-ARC-WMTE-7485 seals exactly the qualified eighteen-unit panel."""

    text = (REPO / panel.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("REQ-ARC-WMTE-7485") : text.index("REQ-ARC-WMTE-7471")]
    for anchor in (
        "SCENARIO-ARC-WMTE-7485-SCHEDULE",
        "SCENARIO-ARC-WMTE-7485-INTERVALS",
        "SCENARIO-ARC-WMTE-7485-FAILURES",
        "SCENARIO-ARC-WMTE-7485-PROGRESS",
        "SCENARIO-ARC-WMTE-7485-TERMINAL",
    ):
        assert anchor in section
    assert panel.MODEL_SPECS == ["unsloth/Qwen3.8-27B-GGUF"]
    assert panel.INFERENCE_SUBSTRATE_CLASS == "model_bounded_generation"
    schedule = panel.build_panel_schedule()
    assert [row["game"] for row in schedule[::3]] == [
        "sk48",
        "tr87",
        "s5i5",
        "lp85",
        "lf52",
        "cn04",
    ]
    assert len(schedule) == 18
    assert {row["seed"] for row in schedule} == {65501, 65502, 65503}
    assert all(row["action_limit"] == 180 for row in schedule)
    assert all(row["request_limit"] == 2 for row in schedule)
    assert all(row["max_new_tokens_per_call"] == 256 for row in schedule)
    assert all(row["episode_limit_s"] == 240 for row in schedule)
    assert all(row["panel_live_limit_s"] == 3600 for row in schedule)


def test_scenario_arc_wmte_7485_intervals_exclude_generation() -> None:
    """SCENARIO-ARC-WMTE-7485-INTERVALS keeps generation nonreplaceable."""

    schedule = panel.build_panel_schedule()
    episodes = [_episode(schedule[0])]
    reduced = panel.reduce_panel(schedule, episodes, _events(schedule[0]["episode_id"]))
    first = reduced["rows"][0]
    assert first["exclusive_cost"]["replaceable_lower_ns"] == 50
    assert first["exclusive_cost"]["replaceable_upper_ns"] >= 50
    generation = next(
        row for row in first["exclusive_cost"]["stage_rows"] if row["decision_id"] == "generation"
    )
    assert generation["work_class"] == "text_generation"
    assert generation["replaceable"] is False
    assert first["request_tokens"] == {"input": 12, "output": 3}
    assert first["no_op_decisions"] == 1
    assert first["delegate_decisions"] == 0


def test_scenario_arc_wmte_7485_failures_and_progress_remain_explicit() -> None:
    """SCENARIO-ARC-WMTE-7485-FAILURES/PROGRESS retain every disposition."""

    schedule = panel.build_panel_schedule()
    episodes = [
        _episode(schedule[0], progressed=True),
        _episode(schedule[1], disposition="censored_timeout"),
    ]
    reduced = panel.reduce_panel(schedule, episodes, _events(schedule[0]["episode_id"]))
    assert len(reduced["rows"]) == 18
    assert reduced["rows"][0]["offline_reproduced"] is True
    assert reduced["rows"][0]["solve_provenance"] == "live_agent_self_discovery"
    assert reduced["rows"][0]["reproduced_levels"] == 1
    assert reduced["rows"][0]["new_level_credit"] == 0
    assert reduced["rows"][1]["disposition"] == "censored_timeout"
    assert reduced["rows"][2]["disposition"] == "unstarted"
    assert reduced["sample_size_budget"] == {
        "planned_independent_units": 18,
        "attempted_independent_units": 2,
        "complete_independent_units": 1,
        "failed_independent_units": 0,
        "censored_independent_units": 1,
        "excluded_independent_units": 0,
        "unstarted_independent_units": 16,
        "independent_game_clusters": 6,
    }
    assert len(reduced["per_game_results"]) == 6
    assert reduced["all_dispositions_present"] is True


def test_req_arc_wmte_7485_fixture_reduces_and_rejects_mutation() -> None:
    """REQ-ARC-WMTE-7485 independently recomputes rows and required fields."""

    artifact = panel.build_artifact_for_test()
    assert panel.validate_artifact(artifact, require_terminal=False) == []
    assert panel.independent_reduce(artifact)["matches_declared"] is True
    assert artifact["arc_panel_a_complete_score"] == 1
    assert artifact["generalization_scope"] == "public_adapter_withheld_proxy"
    assert artifact["verdict_class"] == "null"
    assert artifact["scientific_benefit_score"] == 0
    for gate in artifact["acceptance_gate_results"]:
        assert isinstance(gate["principle"], str) and gate["principle"]

    changed = deepcopy(artifact)
    changed["rows"][0]["request_tokens"]["input"] += 1
    assert "independent_reduction_mismatch" in panel.validate_artifact(
        changed, require_terminal=False
    )


def test_scenario_arc_wmte_7485_terminal_scope_and_replay(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7485-TERMINAL freezes scoped checks and cold replay."""

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


def test_req_arc_wmte_7485_preconditions_and_reader_guards(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7485 authenticates inputs and fails closed on bad bytes."""

    assert panel.utc_now().endswith("Z")
    checks, hashes, upstream, registry = panel.collect_preconditions(REPO, force_live="1")
    assert all(row["passed"] is True for row in checks)
    assert hashes[panel.UPSTREAM_PATH.as_posix()]["original_flags"] == {
        "status": "complete_null_interval_protocol_ready_sample_limited",
        "honest_verdict": "complete_null_interval_protocol_ready_sample_limited",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "arc_interval_protocol_ready_score": 1,
    }
    assert upstream["arc_interval_protocol_ready_score"] == 1
    assert len(registry["rows"]) == 6
    assert all(row["registered"] is True for row in registry["rows"])
    assert registry["policy_received_registry_data"] is False

    assert panel.load_object(tmp_path / "missing.json") == {}
    malformed = tmp_path / "malformed.json"
    malformed.write_text("not-json", encoding="utf-8")
    assert panel.load_object(malformed) == {}
    array = tmp_path / "array.json"
    array.write_text("[]", encoding="utf-8")
    assert panel.load_object(array) == {}

    shard = tmp_path / "rows.jsonl"
    shard.write_text('{"episode_id":"e"}\n', encoding="utf-8")
    assert panel._events_from_shards(
        [{"path": shard.name}, {"path": str(shard)}, {}, {"inline_rows": [None]}],
        root=tmp_path,
    ) == [{"episode_id": "e"}, {"episode_id": "e"}]


def test_req_arc_wmte_7485_rejection_and_scope_drift_paths(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7485 rejects false identity, accounting and scope claims."""

    assert panel._actions_to_progress({"actions_to_progress": 3}) == 3
    artifact = panel.build_artifact_for_test()
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert panel.validate_artifact(path, require_terminal=True) == []

    cases = []
    wrong_identity = deepcopy(artifact)
    wrong_identity["run_date"] = "bad"
    cases.append((wrong_identity, "identity_mismatch:run_date"))
    unbalanced = deepcopy(artifact)
    unbalanced["invocation_counts"]["generation_calls_attempted"] += 1
    cases.append((unbalanced, "unbalanced_invocations:generation_calls"))
    invalid_class = deepcopy(artifact)
    invalid_class["verdict_class"] = "invented"
    cases.append((invalid_class, "invalid_verdict_class"))
    nonzero_credit = deepcopy(artifact)
    nonzero_credit["new_level_credit"] = 1
    cases.append((nonzero_credit, "registered_public_credit_nonzero"))
    no_receipts = deepcopy(artifact)
    no_receipts["validation_receipts"] = []
    cases.append((no_receipts, "required_validation_missing_or_failed"))
    for changed, expected in cases:
        assert expected in panel.validate_artifact(changed, require_terminal=True)

    plan = panel.build_validation_plan(REPO, tmp_path / "scope")
    duplicate = [*plan, plan[0]]
    assert "command_count:worktree_imports:2" in panel.validate_validation_plan(REPO, duplicate)
    broad = panel.validation_scope.CommandSpec(
        "extra",
        (".venv/bin/pytest", "tests/python"),
        "invalid",
    )
    assert "broad_test_target:extra" in panel.validate_validation_plan(REPO, [*plan, broad])

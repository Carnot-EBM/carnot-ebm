"""REQ-ARC-6845 tool-gap causal support audit tests."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.agentic import arc_solve_artifact_discipline as discipline
from carnot import experiment_6845_tool_gap_causal_support_audit as exp
from scripts import arc_artifact_lint as arc_lint


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _digest(prefix: str, index: int) -> str:
    return f"sha256:{prefix}{index:058d}"


def _inventory_row(
    *,
    game: str = "sp80",
    run: str = "sp80-selfparse",
    source_path: str | None = None,
    tool_loop: str = "selfparse",
    complete: bool = True,
) -> dict[str, Any]:
    source = source_path or f"results/arc_leaderboard_eval_runs/{run}.json"
    return {
        "artifact_family": "leaderboard_trajectory",
        "source_path": source,
        "source_artifact_sha256": _digest("source", len(game)),
        "run_id": run,
        "game": game,
        "model_id": "unsloth/Qwen3.8-27B-GGUF",
        "policy": "e3",
        "budget": 2500,
        "tool_loop_state": tool_loop,
        "supervisor_state": "unobserved",
        "receipt_completeness": "complete",
        "trajectory_receipt_complete": True,
        "tool_gap_receipt_complete": complete,
        "tool_gap_receipt_count": 1 if complete else 0,
        "tool_gap_calls_total": 3 if complete else 0,
        "tool_gap_event_count": 1 if complete else 0,
        "producer_configuration": {
            "levels_recorded": 1,
            "actions_recorded": 50,
            "charged_actions": 50,
        },
        "stratum_identity": (
            "leaderboard_trajectory|"
            f"{run}|{game}|unsloth/Qwen3.8-27B-GGUF|e3|2500|"
            f"{tool_loop}|unobserved|complete"
        ),
        "row_sha256": _digest("row", len(game)),
    }


def _inventory(rows: list[dict[str, Any]] | None = None, *, complete: int = 1) -> dict[str, Any]:
    selected = rows if rows is not None else [_inventory_row()]
    eligible = [row for row in selected if row.get("tool_gap_receipt_complete") is True]
    return {
        "schema": "carnot.experiment_6843.live_arc_evidence_stratum_freeze.v1",
        "status": "complete_live_arc_inventory",
        "arc_inventory_complete_score": complete,
        "tool_gap_eligible_cells": {
            "count": len(eligible),
            "row_ids": [row["stratum_identity"] for row in eligible],
        },
        "rows": selected,
        "unmatched_cell_reasons": [],
        "honest_verdict": "complete_live_arc_inventory",
        "verdict_class": "null",
    }


def _event(
    *,
    index: int = 1,
    tool: str = "get_full_grid",
    before: int = 0,
    after: int = 1,
    used_response: bool = True,
    changed_after_response: bool = True,
    valid_action: bool = True,
    headroom: bool = True,
) -> dict[str, Any]:
    available = [6]
    next_action = {"kind": 6, "data": {"x": index, "y": index + 1}}
    if not valid_action:
        next_action = {"kind": 9, "data": {"x": index, "y": index + 1}}
    return {
        "kind": "unknown_tool",
        "missing_fact": "full grid bytes for transition 3",
        "requested_tool": tool,
        "argument_keys": ["t", "which"],
        "turn": index,
        "request_sequence": 10 + index,
        "receipt_sequence": 20 + index,
        "visibility_sequence": 30 + index,
        "next_action_sequence": 40 + index,
        "outcome_sequence": 50 + index,
        "request_id": _digest("request", index),
        "tool_receipt_id": _digest("receipt", index),
        "response_id": _digest("response", index),
        "agent_visible_response_id": _digest("visible", index),
        "next_action_id": _digest("nextaction", index),
        "raw_transcript_sha256": _digest("transcript", index),
        "actual_call": {
            "name": tool,
            "arguments": {"t": 3, "which": "before"},
        },
        "exact_response": {
            "ok": False,
            "error": f"unknown tool: {tool}",
            "source": "dispatch_tool",
        },
        "agent_visible_text": (
            "<tool_response>\n"
            + json.dumps({"ok": False, "error": f"unknown tool: {tool}"})
            + "\n</tool_response>"
        ),
        "next_action": next_action,
        "next_action_receipt": {
            "next_action_id": _digest("nextaction", index),
            "used_tool_response": used_response,
            "changed_after_response": changed_after_response,
            "available_actions": available,
        },
        "later_exact_outcome": {
            "outcome_id": _digest("outcome", index),
            "outcome_status": "returned",
            "live_return": True,
            "fully_joined": True,
            "levels_completed_before": before,
            "levels_completed_after": after,
            "reward": {"present": False, "value": None},
            "termination": {"state": "NOT_FINISHED"},
            "headroom": {
                "baseline_score": 0.0,
                "best_available_score": 1.0 if headroom else 0.0,
            },
        },
    }


def _leaderboard(
    *,
    game: str = "sp80",
    events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "experiment": "arc_leaderboard_eval",
        "games_mode": "claimed",
        "policy": "e3",
        "budget": 2500,
        "random_seed": 20260719,
        "honest_verdict": "complete_leaderboard_eval_1_levels_1_gaps",
        "per_game": [
            {
                "game": game,
                "levels": 1,
                "reached": 1,
                "actions": 50,
                "charged_actions": 50,
                "policy_diagnostics": {
                    "proposer": {"repo_substr": "Qwen3.8-27B"},
                    "induction_attempts": [
                        {
                            "model_specs": "Qwen3.8-27B-Q4_K_M GGUF",
                            "tool_gap": {
                                "tool_gap_events": events if events is not None else [_event()],
                                "tool_gap_events_dropped": 0,
                                "candidate_tools_enabled": [],
                                "candidate_tools_rejected": [],
                                "terminated_by": "early_stop_non_improving",
                                "tool_calls_total": 3,
                            },
                        }
                    ],
                },
            }
        ],
    }


def _source_paths(
    tmp_path: Path,
    *,
    inventory: dict[str, Any] | None = None,
    leaderboards: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Path]:
    boards = leaderboards or {"sp80-selfparse": _leaderboard()}
    paths = {
        "agents_md": _write_text(tmp_path / "AGENTS.md", "# agents\n"),
        "claude_md": _write_text(tmp_path / "CLAUDE.md", "# claude\n"),
        "codex_md": _write_text(tmp_path / "CODEX.md", "# codex\n"),
        "north_star": _write_text(tmp_path / "ops/north-star.md", "# north star\n"),
        "ops_status": _write_text(tmp_path / "ops/status.md", "# status\n"),
        "spec": _write_text(
            tmp_path / exp.SPEC_PATH,
            (REPO_ROOT / exp.SPEC_PATH).read_text(encoding="utf-8"),
        ),
        "experiment_6473": _write_json(
            tmp_path / "results/experiment_6473_tool_loop_compaction_pilot_ab.json",
            {"honest_verdict": "complete: pilot transport only"},
        ),
        "experiment_6777": _write_json(
            tmp_path / "results/experiment_6777_arc_tool_gap_transport.json",
            {
                "status": "blocked",
                "honest_verdict": "blocked_gate_check_failed",
                "gate_check_summary": "upstream transport gate blocked",
            },
        ),
        "experiment_6820_missing": tmp_path
        / "results/experiment_6820_arc_tool_gap_obligation_transport_v2.json",
        "experiment_6843": _write_json(
            tmp_path / "results/experiment_6843_live_arc_evidence_stratum_freeze.json",
            inventory or _inventory(),
        ),
    }
    for run, payload in boards.items():
        paths[f"leaderboard:{run}.json"] = _write_json(
            tmp_path / f"results/arc_leaderboard_eval_runs/{run}.json",
            payload,
        )
    return paths


def test_req_6845_spec_precedes_implementation() -> None:
    """REQ-ARC-6845 declares gates, scenarios, and artifact fields."""

    text = (REPO_ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("## REQ-ARC-6845:") :]
    for marker in (
        "SCENARIO-ARC-6845-REQUEST-RECEIPT-JOIN",
        "SCENARIO-ARC-6845-VISIBILITY-NEXT-ACTION",
        "SCENARIO-ARC-6845-GATES-FAIL-CLOSED",
        "SCENARIO-ARC-6845-STRATA-NOT-POOLED",
        "SCENARIO-ARC-6845-HASHES-NO-SOLVE",
    ):
        assert marker in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert field in exp.FIELD_PRINCIPLES


def test_scenario_6845_request_receipt_visibility_and_utility_success(tmp_path: Path) -> None:
    """SCENARIO-ARC-6845-REQUEST-RECEIPT-JOIN links the whole chain."""

    artifact = exp.build_artifact(
        run_date="20260901",
        duration_s=0.25,
        source_paths=_source_paths(tmp_path),
    )

    assert artifact["status"] == "complete_tool_gap_causal_support_audit"
    assert artifact["tool_gap_audit_complete_score"] == 1
    assert artifact["tool_gap_effect_eligible_score"] == 1
    assert artifact["solve_claim"] is False
    assert artifact["verifier_is_oracle"] is False
    row = artifact["per_game_results"][0]
    assert row["missing_fact"] == "full grid bytes for transition 3"
    assert row["requested_tool"] == "get_full_grid"
    assert row["actual_call"]["name"] == "get_full_grid"
    assert row["request_receipt_joined"] is True
    assert row["agent_visible"] is True
    assert row["next_action_receipted"] is True
    assert row["exact_later_outcome_joined"] is True
    assert row["later_exact_outcome"]["direction"] == "progress"
    assert artifact["request_receipt_joins"]["joined_count"] == 1
    assert artifact["agent_visibility_results"][0]["agent_visible_rate"] == 1
    assert artifact["next_action_results"][0]["action_change_rate"] == 1
    assert artifact["transport_results"][0]["transport_success_rate"] == 1
    assert artifact["utility_results"][0]["use_rate"] == 1
    assert artifact["utility_results"][0]["transition_progress_rate"] == 1
    assert artifact["utility_results"][0]["invalid_action_rate"] == 0
    assert artifact["headroom_results"][0]["nonzero_headroom"] is True
    assert exp.validate_artifact(artifact) == []


def test_scenario_6845_transport_success_does_not_imply_utility(tmp_path: Path) -> None:
    """SCENARIO-ARC-6845-VISIBILITY-NEXT-ACTION separates transport and use."""

    event = _event(after=0, used_response=False, changed_after_response=False)
    artifact = exp.build_artifact(
        run_date="20260901",
        duration_s=0.25,
        source_paths=_source_paths(
            tmp_path,
            leaderboards={"sp80-selfparse": _leaderboard(events=[event])},
        ),
    )

    assert artifact["status"] == "complete_tool_gap_causal_support_audit"
    utility = artifact["utility_results"][0]
    assert artifact["transport_results"][0]["transport_success_rate"] == 1
    assert utility["use_rate"] == 0
    assert utility["action_change_rate"] == 0
    assert utility["transition_progress_rate"] == 0
    assert utility["effect_eligible"] is True


@pytest.mark.parametrize(
    ("mutator", "failed_check"),
    [
        (
            lambda event: event.update({"raw_transcript_sha256": None}),
            "immutable_raw_transcript_hashes",
        ),
        (
            lambda event: event.update({"response_id": None}),
            "request_response_identities",
        ),
        (
            lambda event: event.update({"agent_visible_text": ""}),
            "agent_visible_responses",
        ),
        (
            lambda event: event.pop("next_action_receipt"),
            "next_action_receipts",
        ),
        (
            lambda event: event["later_exact_outcome"].update({"outcome_id": None}),
            "exact_later_outcomes",
        ),
        (
            lambda event: event.update({"outcome_sequence": event["next_action_sequence"]}),
            "temporal_order",
        ),
        (
            lambda event: event["later_exact_outcome"]["headroom"].update(
                {"best_available_score": 0.0}
            ),
            "headroom_nonzero",
        ),
    ],
)
def test_scenario_6845_gates_fail_closed(
    tmp_path: Path,
    mutator: Any,
    failed_check: str,
) -> None:
    """SCENARIO-ARC-6845-GATES-FAIL-CLOSED records the first failed check."""

    event = _event()
    mutator(event)
    artifact = exp.build_artifact(
        run_date="20260901",
        duration_s=0.25,
        source_paths=_source_paths(
            tmp_path,
            leaderboards={"sp80-selfparse": _leaderboard(events=[event])},
        ),
    )

    assert artifact["status"] == "complete_blocked_tool_gap_causal_support_audit"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["tool_gap_effect_eligible_score"] == 0
    assert artifact["gate_check_summary"]["failed_check"] == failed_check
    assert artifact["gate_check_summary"]["observed"] is not None
    assert exp.validate_artifact(artifact) == []


def test_scenario_6845_duplicate_identities_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-ARC-6845-GATES-FAIL-CLOSED rejects duplicate transport IDs."""

    event = _event(index=1)
    duplicate = deepcopy(event)
    duplicate["turn"] = 2
    duplicate["request_sequence"] = 12
    duplicate["receipt_sequence"] = 22
    duplicate["visibility_sequence"] = 32
    duplicate["next_action_sequence"] = 42
    duplicate["outcome_sequence"] = 52
    artifact = exp.build_artifact(
        run_date="20260901",
        duration_s=0.25,
        source_paths=_source_paths(
            tmp_path,
            leaderboards={"sp80-selfparse": _leaderboard(events=[event, duplicate])},
        ),
    )

    assert artifact["gate_check_summary"]["failed_check"] == "duplicate_transport_identities"
    assert artifact["request_receipt_joins"]["duplicate_request_ids"] == [event["request_id"]]
    assert artifact["tool_gap_effect_eligible_score"] == 0


def test_scenario_6845_terminal_tool_gap_cells_required(tmp_path: Path) -> None:
    """SCENARIO-ARC-6845-GATES-FAIL-CLOSED rejects no terminal tool-gap cell."""

    artifact = exp.build_artifact(
        run_date="20260901",
        duration_s=0.25,
        source_paths=_source_paths(tmp_path, inventory=_inventory(rows=[]), leaderboards={}),
    )

    assert artifact["status"] == "complete_blocked_tool_gap_causal_support_audit"
    assert artifact["gate_check_summary"]["failed_check"] == "terminal_tool_gap_cells"
    assert artifact["per_game_results"] == []
    assert artifact["unmatched_cell_results"]
    assert exp.validate_artifact(artifact) == []


def test_scenario_6845_configuration_strata_never_pool_loop_states(tmp_path: Path) -> None:
    """SCENARIO-ARC-6845-STRATA-NOT-POOLED keeps loop states separate."""

    on_row = _inventory_row(game="sp80", run="sp80-on", tool_loop="selfparse")
    off_row = _inventory_row(game="sp80", run="sp80-off", tool_loop="off_or_unobserved")
    artifact = exp.build_artifact(
        run_date="20260901",
        duration_s=0.25,
        source_paths=_source_paths(
            tmp_path,
            inventory=_inventory(rows=[on_row, off_row]),
            leaderboards={
                "sp80-on": _leaderboard(game="sp80", events=[_event(index=1)]),
                "sp80-off": _leaderboard(game="sp80", events=[_event(index=2)]),
            },
        ),
    )

    strata = artifact["configuration_strata"]["strata"]
    assert artifact["configuration_strata"]["stratum_count"] == 2
    assert {row["tool_loop_state"] for row in strata} == {"selfparse", "off_or_unobserved"}
    assert all("tool-loop state" in artifact["configuration_strata"]["pooling_rule"] for _ in [0])
    assert len({row["matched_stratum"] for row in artifact["per_game_results"]}) == 2


def test_scenario_6845_hashes_validator_lint_and_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-ARC-6845-HASHES-NO-SOLVE keeps stable hashes and writes JSON."""

    paths = _source_paths(tmp_path)
    artifact = exp.build_artifact(run_date="20260901", duration_s=0.25, source_paths=paths)

    assert artifact["reproducibility_checksum"].startswith("sha256:")
    assert artifact["per_game_results"][0]["row_sha256"].startswith("sha256:")
    assert artifact["per_game_results"][0]["raw_transcript_sha256"].startswith("sha256:")
    changed = deepcopy(artifact)
    changed["duration_s"] = 999.0
    assert exp.reproducibility_checksum(changed) == artifact["reproducibility_checksum"]
    assert discipline.duration_floor_s(exp.INFERENCE_SUBSTRATE) == 0.0001
    assert (
        arc_lint.lint_artifact(
            tmp_path / "results/experiment_6845_tool_gap_causal_support_audit.json",
            artifact,
        )
        == []
    )

    broken = deepcopy(artifact)
    broken["solve_claim"] = True
    broken["reproducibility_checksum"] = exp.reproducibility_checksum(broken)
    assert "solve_claim must be false" in exp.validate_artifact(broken)

    broken = deepcopy(artifact)
    broken["field_principles"].pop("per_game_results")
    broken["reproducibility_checksum"] = exp.reproducibility_checksum(broken)
    assert "field principles do not cover every top-level field" in exp.validate_artifact(broken)

    monkeypatch.setattr(exp, "collect_default_source_paths", lambda _root: paths)
    output = tmp_path / "out.json"
    assert exp.main(["--date", "20260901", "--output", str(output)]) == 0
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["honest_verdict"].startswith("complete_")
    assert written["run_date"] == "20260901"

    monkeypatch.setattr(exp, "validate_artifact", lambda _artifact: ["forced failure"])
    assert exp.main(["--date", "20260901", "--output", str(tmp_path / "bad.json")]) == 1


def test_scenario_6845_defensive_validation_branches(tmp_path: Path) -> None:
    """SCENARIO-ARC-6845-GATES-FAIL-CLOSED covers malformed inputs."""

    assert exp._load_json(b"{") == {}
    assert exp._load_json(b"[]") == {}
    assert exp._source_record("missing", tmp_path / "missing.json", tmp_path)["exists"] is False
    assert exp._relative(Path("/not/under/root.json"), tmp_path).endswith("root.json")
    assert exp._common_root(None) == exp.REPO_ROOT
    assert exp._common_root({"missing": tmp_path / "missing.json"}) == exp.REPO_ROOT
    assert exp.collect_default_source_paths(tmp_path)["experiment_6843"].name.endswith(".json")
    _write_json(
        tmp_path / "results/experiment_6843_live_arc_evidence_stratum_freeze.json",
        _inventory(rows=[_inventory_row(source_path="results/arc_leaderboard_eval_runs/x.json")]),
    )
    assert "leaderboard:x.json" in exp.collect_default_source_paths(tmp_path)
    _write_json(
        tmp_path / "results/experiment_6843_live_arc_evidence_stratum_freeze.json",
        {"rows": []},
    )
    _write_json(tmp_path / "results/arc_leaderboard_eval_runs/fallback.json", {})
    assert "leaderboard:fallback.json" in exp.collect_default_source_paths(tmp_path)
    assert exp._mean([]) is None
    assert exp._rate_interval(0, 0)["lower"] is None
    assert exp._inventory_cells({"experiment_6843": {"_payload": {"rows": [None]}}}) == []
    assert exp._game_rows({"per_game": {}}, "sp80") == []
    assert exp._outcome_score({"reward": {"present": True, "value": 2}}) == 2
    assert (
        exp._outcome_score(
            {
                "levels_completed_before": 0,
                "levels_completed_after": 0,
                "termination": {"state": "GAME_OVER"},
            }
        )
        == -1
    )
    assert exp._direction({"error": "boom", "outcome_status": "returned"}) == "regression"
    assert exp._is_sha256(_digest("x", 1))
    assert exp._is_sha256(None) is False
    assert (
        exp._event_requested_tool({"kind": "bad_arguments", "tool": "diff_grids"}) == "diff_grids"
    )
    assert exp._event_missing_fact({"kind": "bad_arguments", "tool": "diff_grids"}).startswith(
        "signature mismatch"
    )
    assert (
        exp._event_missing_fact({"requested_tool": "scan_board"}) == "tool unavailable: scan_board"
    )
    assert exp._headroom_nonzero({}) is False
    assert exp._headroom_nonzero({"headroom": {"nonzero_headroom": True}}) is True
    assert exp._action_validity(
        {"next_action_receipt": {"proposal_validity": {"valid": False, "reason": "fixture"}}}
    ) == {"valid": False, "reason": "fixture"}
    assert exp._action_validity({"next_action": None})["reason"] == "missing_next_action"
    assert (
        exp._action_validity(
            {
                "next_action": {"kind": 6, "data": {"x": "bad", "y": 1}},
                "next_action_receipt": {"available_actions": [6]},
            }
        )["reason"]
        == "action6_requires_integer_xy"
    )
    assert exp._action_validity(
        {
            "next_action": {"kind": "RESET"},
            "next_action_receipt": {"available_actions": []},
        }
    )["valid"]
    assert exp._event_sequence_ok({"request_sequence": 1}) is False

    event = _event(valid_action=False)
    artifact = exp.build_artifact(
        run_date="20260901",
        duration_s=0.25,
        source_paths=_source_paths(
            tmp_path,
            leaderboards={"sp80-selfparse": _leaderboard(events=[event])},
        ),
    )
    assert artifact["utility_results"][0]["invalid_action_rate"] == 1

    empty_event_artifact = exp.build_artifact(
        run_date="20260901",
        duration_s=0.25,
        source_paths=_source_paths(
            tmp_path,
            leaderboards={"sp80-selfparse": _leaderboard(events=[])},
        ),
    )
    assert empty_event_artifact["gate_check_summary"]["failed_check"] == "tool_gap_obligations"
    assert any(
        row["reason"] == "no_tool_gap_obligation_events"
        for row in empty_event_artifact["unmatched_cell_results"]
    )
    off_cell_artifact = exp.build_artifact(
        run_date="20260901",
        duration_s=0.25,
        source_paths=_source_paths(
            tmp_path,
            inventory=_inventory(rows=[_inventory_row(complete=False)]),
            leaderboards={"sp80-selfparse": _leaderboard(events=[_event()])},
        ),
    )
    assert any(
        row["reason"] == "tool_gap_receipt_missing_or_loop_off"
        for row in off_cell_artifact["unmatched_cell_results"]
    )
    bad_stats_artifact = exp.build_artifact(
        run_date="20260901",
        duration_s=0.25,
        source_paths=_source_paths(
            tmp_path,
            leaderboards={
                "sp80-selfparse": _leaderboard(events={"not": "a list"}),  # type: ignore[arg-type]
            },
        ),
    )
    assert bad_stats_artifact["gate_check_summary"]["failed_check"] == "tool_gap_obligations"
    expected_paths = _source_paths(tmp_path)
    expected_paths["ops_status"].write_text(
        "Batches 3-5 (`sp80,su15`, `tu93,cn04`, `m0r0,sk48`) run with selfparse.\n",
        encoding="utf-8",
    )
    expected_artifact = exp.build_artifact(
        run_date="20260901",
        duration_s=0.25,
        source_paths=expected_paths,
    )
    missing_games = {
        row["game"]
        for row in expected_artifact["unmatched_cell_results"]
        if row["reason"] == "expected_configuration_missing_terminal_receipt"
    }
    assert {"cn04", "tu93"} <= missing_games
    assert expected_artifact["configuration_strata"]["expected_missing_configurations"]
    assert (
        "_raw_text"
        not in exp.source_artifact_hashes(
            {"ops_status": exp._source_record("ops_status", expected_paths["ops_status"], tmp_path)}
        )["ops_status"]
    )

    def checked(mutator: Any, expected: str) -> None:
        broken = deepcopy(artifact)
        mutator(broken)
        if "reproducibility_checksum" in broken:
            broken["reproducibility_checksum"] = exp.reproducibility_checksum(broken)
        assert expected in exp.validate_artifact(broken)

    checked(lambda data: data.pop("schema"), "required artifact fields are missing")
    checked(lambda data: data.update({"schema": "bad"}), "schema mismatch")
    checked(
        lambda data: data.update({"inference_substrate": "bad"}), "inference substrate mismatch"
    )
    checked(
        lambda data: data.update({"verifier_is_oracle": True}), "verifier_is_oracle must be false"
    )
    checked(
        lambda data: data.update({"verdict_class": "surprise"}),
        "verdict class is outside the closed set",
    )
    checked(
        lambda data: data.update({"honest_verdict": "blocked"}),
        "honest verdict lacks complete_ terminal prefix",
    )
    checked(
        lambda data: data.update({"tool_gap_audit_complete_score": 0}),
        "tool-gap audit complete score mismatch",
    )
    checked(
        lambda data: data.update({"tool_gap_effect_eligible_score": 2}),
        "effect eligible score must be binary",
    )
    checked(
        lambda data: data["per_game_results"][0].update({"row_sha256": "bad"}),
        "per_game row missing hash",
    )
    broken_checksum = deepcopy(artifact)
    broken_checksum["status"] = "changed"
    assert "reproducibility checksum mismatch" in exp.validate_artifact(broken_checksum)

    blocked = deepcopy(artifact)
    blocked.update(
        {
            "status": "complete_blocked_tool_gap_causal_support_audit",
            "verdict_class": "null",
            "tool_gap_effect_eligible_score": 1,
            "gate_check_summary": {"passed": False},
        }
    )
    blocked["reproducibility_checksum"] = exp.reproducibility_checksum(blocked)
    errors = exp.validate_artifact(blocked)
    assert "blocked verdict_class mismatch" in errors
    assert "blocked artifact marked effect eligible" in errors
    assert "blocked artifact lacks failed check" in errors

    complete = deepcopy(artifact)
    complete.update(
        {
            "status": "complete_tool_gap_causal_support_audit",
            "verdict_class": "blocked",
            "tool_gap_effect_eligible_score": 0,
            "gate_check_summary": {"passed": False},
        }
    )
    complete["reproducibility_checksum"] = exp.reproducibility_checksum(complete)
    errors = exp.validate_artifact(complete)
    assert "complete audit verdict_class mismatch" in errors
    assert "complete artifact lacks effect eligibility" in errors
    assert "complete artifact gate summary mismatch" in errors

    no_rows_complete = deepcopy(artifact)
    no_rows_complete["per_game_results"] = []
    no_rows_complete["reproducibility_checksum"] = exp.reproducibility_checksum(no_rows_complete)
    assert "complete audit lacks per_game rows" in exp.validate_artifact(no_rows_complete)

    checked(lambda data: data.update({"status": "unknown"}), "status mismatch")

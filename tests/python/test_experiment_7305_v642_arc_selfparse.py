"""Tests for the durable direct-selfparse live session.

Spec refs: REQ-ARC-WMTE-7305 and SCENARIO-ARC-WMTE-7305-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy

import pytest

from carnot import experiment_7305_v642_arc_selfparse as exp


REPO = Path(__file__).resolve().parents[2]


def _boundary() -> dict[str, object]:
    return {
        "activity_known": True,
        "disqualified": False,
        "errors": [],
        "model_invoked": True,
        "inference_substrate": "model_full_generation",
        "invocation_counts": {
            "model_loads_attempted": 1,
            "model_loads_completed": 1,
            "model_loads_failed": 0,
            "model_loads_cancelled": 0,
            "model_loads_in_flight": 0,
            "generation_calls_attempted": 2,
            "generation_calls_completed": 2,
            "generation_calls_failed": 0,
            "generation_calls_cancelled": 0,
            "generation_calls_in_flight": 0,
            "usable_answers": 2,
        },
        "call_rows": [],
    }


def _episode(tmp_path: Path) -> dict[str, object]:
    request0 = tmp_path / "request0.json"
    request1 = tmp_path / "request1.json"
    request0.write_text(json.dumps({"messages": [{"content": "observe"}]}), encoding="utf-8")
    bounded = '<tool_response>\n{"ok": true}\n</tool_response>'
    request1.write_text(json.dumps({"messages": [{"content": bounded}]}), encoding="utf-8")
    return {
        "episode_id": "r11l:direct_selfparse",
        "game": "r11l",
        "seed": exp.EVALUATION_SEED,
        "arm": "direct_selfparse",
        "disposition": "complete",
        "censored": False,
        "model_loaded": True,
        "model_invoked": True,
        "generation_calls_attempted": 2,
        "generation_calls_completed": 2,
        "generated_tokens": 1800,
        "action_count": 81,
        "action_limit": exp.ACTION_LIMIT,
        "levels": 0,
        "adapter_disabled": True,
        "banked_solution_disabled": True,
        "induction_rows": [{"planned": True, "attempt_index": 0}],
        "policy_consumption_rows": [
            {
                "attempt_index": 0,
                "action_index": 80,
                "policy_action_executed": True,
                "engine_sha256": "sha256:" + "e" * 64,
                "plan_sha256": "sha256:" + "p" * 64,
            }
        ],
        "raw_request_manifest": [
            {"call_index": 0, "request_path": str(request0)},
            {"call_index": 1, "request_path": str(request1)},
        ],
        "action_rows": [{"i": 80, "action": 1, "top_branch": "execute.plan_step"}],
        "error": None,
    }


def _tool_events() -> list[dict[str, object]]:
    return [
        {
            "episode_id": "r11l:direct_selfparse",
            "induction_index": 0,
            "turn": 0,
            "parsed_tool": "list_transitions",
            "dispatch_result": {"ok": True},
            "bounded_response": '<tool_response>\n{"ok": true}\n</tool_response>',
        }
    ]


def test_req_7305_spec_and_frozen_live_contract() -> None:
    """REQ-ARC-WMTE-7305 fixes the model, route, target, and inherited ceilings."""

    spec = (REPO / exp.SPEC_PATH).read_text(encoding="utf-8")
    assert "## REQ-ARC-WMTE-7305:" in spec
    assert "SCENARIO-ARC-WMTE-7305-TOOL-POLICY-ACTION" in spec
    assert exp.MODEL_SPECS == [{"hf_id": "unsloth/Qwen3.8-27B-GGUF", "quantization": "Q4_K_M"}]
    assert (exp.ACTION_LIMIT, exp.COMPLETION_LIMIT, exp.GENERATED_TOKEN_LIMIT) == (192, 2, 4096)
    assert (exp.SESSION_LIMIT_S, exp.MODEL_LOAD_LIMIT_S) == (3000, 600)
    assert exp.TARGET_GAME == "r11l"


@pytest.mark.parametrize(
    ("change", "observed"),
    [
        (None, "missing_artifact"),
        ({"flagged_adversarial": True}, "quarantined"),
        ({"verdict_class": "disqualified"}, "disqualified"),
        ({"status": "running"}, "running"),
        ({"arc_receipt_ready_score": 0}, 0),
    ],
)
def test_scenario_7305_dependency_rejects_before_numeric_consumption(
    change: dict[str, object] | None, observed: object
) -> None:
    """SCENARIO-ARC-WMTE-7305-DEPENDENCY-BLOCK rejects unsafe producer state."""

    baseline: dict[str, object] = {
        "status": "complete",
        "verdict_class": "circular_positive",
        "arc_receipt_ready_score": 1,
    }
    artifact = None if change is None else {**baseline, **change}
    row = exp.check_dependency(artifact)

    assert row["passed"] is False
    assert row["upstream"] == exp.EXP7304_PATH.as_posix()
    assert row["observed"] == observed
    assert row["expected"] in {1, "complete", "unquarantined", "not_disqualified"}


def test_scenario_7305_caller_hash_mismatch_is_exact() -> None:
    """SCENARIO-ARC-WMTE-7305-DEPENDENCY-BLOCK names the changed caller bytes."""

    handoff = {"caller_code_hashes": {"python/example.py": "sha256:" + "a" * 64}}
    rows = exp.authenticate_caller_hashes(
        Path("/repo"), handoff, hasher=lambda path: "sha256:" + "b" * 64
    )

    assert rows == [
        exp.gate_check(
            "exp7304_caller_hash",
            "python/example.py",
            "sha256",
            "sha256:" + "a" * 64,
            "sha256:" + "b" * 64,
        )
    ]


def test_scenario_7305_dependency_checksum_and_missing_handoff_fail_closed() -> None:
    """SCENARIO-ARC-WMTE-7305-DEPENDENCY-BLOCK authenticates the terminal bytes."""

    dependency = {
        "status": "complete",
        "verdict_class": "circular_positive",
        "arc_receipt_ready_score": 1,
        "reproducibility_checksum": "",
    }
    dependency["reproducibility_checksum"] = exp.artifact_checksum(dependency)
    assert exp.check_dependency(dependency)["passed"] is True
    dependency.pop("reproducibility_checksum")
    assert exp.check_dependency(dependency)["observed"] == "missing"
    assert exp.authenticate_caller_hashes(Path("/repo"), {})[0]["observed"] == "missing"


def test_scenario_7305_target_freezes_before_outcomes_and_withholds_adapter() -> None:
    """SCENARIO-ARC-WMTE-7305-FROZEN-LIVE-SESSION uses registry metadata only."""

    registry = {
        "games": [{"game": "r11l", "reproducibility": "reproduced", "levels_reproduced": 6}]
    }
    receipt = exp.freeze_target(registry, adaptered_games={"r11l"})

    assert receipt["passed"] is True
    assert receipt["target"] == "r11l"
    assert receipt["adapter_available_but_withheld"] is True
    assert receipt["adapter_disabled"] is True
    assert receipt["banked_solution_disabled"] is True
    assert receipt["outcomes_seen_before_freeze"] is False
    assert receipt["registered_levels_are_transfer_only"] is True


def test_scenario_7305_environment_enforces_direct_selfparse_and_receipt(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7305-FROZEN-LIVE-SESSION fixes all runtime caps."""

    ledger = tmp_path / "boundary.jsonl"
    env = exp.session_environment(
        {"CARNOT_ARC_SUPERVISOR_TOOL_ARM": "1"},
        episode_dir=tmp_path / "episode",
        gpu_index=1,
        port=9123,
        boundary_path=ledger,
    )

    assert env["CARNOT_FORCE_LIVE"] == "1"
    assert env["CARNOT_ARC_INDUCE_TOOL_LOOP"] == "selfparse"
    assert env["CARNOT_ARC_INDUCE_TOOL_TURNS"] == "2"
    assert env["CARNOT_ARC_INDUCE_MAX_TOKENS"] == "2048"
    assert env["CARNOT_ARC_INDUCE_TIMEOUT"] == "600"
    assert env["CARNOT_ARC_BOUNDARY_LEDGER_PATH"] == str(ledger)
    assert "CARNOT_ARC_SUPERVISOR_TOOL_ARM" not in env
    assert "CARNOT_ARC_TRANSITION_WITNESS" not in env


def test_scenario_7305_boundary_counts_retain_in_flight_and_cancelled_fields(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-7305-DURABLE-CENSORING uses the shipped durable reducer."""

    from carnot.agentic.arc_inference_boundary import InvocationBoundaryLedger

    ledger = InvocationBoundaryLedger(tmp_path / "boundary.jsonl")
    identity = {
        "model_repository": exp.MODEL_ID,
        "model_filename": "Qwen3.8-27B-Q4_K_M.gguf",
        "model_revision": "a" * 40,
        "model_path": "/cache/Qwen3.8-27B-Q4_K_M.gguf",
    }
    load = ledger.begin("model_load", identity, call_id="load")
    load.complete()
    ledger.begin("generation", identity, child_pid=44, call_id="generation")

    reduced = exp.reduce_boundary_ledger(ledger.path)

    assert reduced["model_invoked"] is True
    assert reduced["invocation_counts"]["model_loads_completed"] == 1
    assert reduced["invocation_counts"]["generation_calls_in_flight"] == 1
    assert reduced["invocation_counts"]["model_loads_cancelled"] == 0
    assert reduced["invocation_counts"]["generation_calls_cancelled"] == 0


def test_scenario_7305_tool_result_to_policy_action_chain_is_exact(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7305-TOOL-POLICY-ACTION requires every E2E link."""

    episode = _episode(tmp_path)
    chain = exp.reduce_tool_use_chain(episode, _tool_events())

    assert chain["arc_tool_use_score"] == 1
    assert chain["successful_tool_results"] == 1
    assert chain["results_in_later_request"] == 1
    assert chain["policy_consumed_results"] == 1
    assert chain["subsequent_environment_actions"] == 1
    assert chain["rows"][0]["passed"] is True


@pytest.mark.parametrize(
    "mutation",
    ["failed_tool", "not_in_later_request", "unplanned", "no_policy_action"],
)
def test_scenario_7305_missing_chain_link_is_null(tmp_path: Path, mutation: str) -> None:
    """SCENARIO-ARC-WMTE-7305-TOOL-POLICY-ACTION fails closed per missing link."""

    episode = _episode(tmp_path)
    events = _tool_events()
    if mutation == "failed_tool":
        events[0]["dispatch_result"] = {"ok": False}
    elif mutation == "not_in_later_request":
        Path(episode["raw_request_manifest"][1]["request_path"]).write_text(
            json.dumps({"messages": [{"content": "different"}]}), encoding="utf-8"
        )
    elif mutation == "unplanned":
        episode["induction_rows"] = [{"planned": False, "attempt_index": 0}]
    else:
        episode["policy_consumption_rows"] = []

    chain = exp.reduce_tool_use_chain(episode, events)

    assert chain["arc_tool_use_score"] == 0
    assert chain["rows"][0]["passed"] is False


def test_scenario_7305_malformed_request_and_non_text_content_are_not_delivery(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-7305-TOOL-POLICY-ACTION rejects unreadable delivery evidence."""

    episode = _episode(tmp_path)
    request = Path(episode["raw_request_manifest"][1]["request_path"])
    request.write_text("not json", encoding="utf-8")
    assert exp.reduce_tool_use_chain(episode, _tool_events())["results_in_later_request"] == 0
    request.write_text(json.dumps({"messages": [{"content": 7}]}), encoding="utf-8")
    assert exp.reduce_tool_use_chain(episode, _tool_events())["results_in_later_request"] == 0


def test_scenario_7305_historical_ledger_rejects_quarantine_and_deduplicates(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-7305-CUMULATIVE-SEPARATION keeps history out of calls."""

    row = {"induction_id": "sha256:" + "a" * 64, "engaged": True}
    clean = {
        "status": "complete",
        "arc_session_complete_score": 1,
        "flagged_adversarial": False,
        "cumulative_induction_rows": [row, deepcopy(row)],
    }
    rejected = {
        "status": "complete",
        "arc_session_complete_score": 1,
        "flagged_adversarial": True,
        "cumulative_induction_rows": [{"induction_id": "sha256:" + "b" * 64, "engaged": True}],
    }
    clean_path = tmp_path / "clean.json"
    rejected_path = tmp_path / "rejected.json"
    clean_path.write_text(json.dumps(clean), encoding="utf-8")
    rejected_path.write_text(json.dumps(rejected), encoding="utf-8")

    ledger = exp.build_cumulative_ledger(
        tmp_path / "ledger.json",
        [clean_path, rejected_path],
        quarantine_check=lambda p: bool(p.get("flagged_adversarial")),
    )

    assert ledger["authenticated_unique_inductions"] == 1
    assert ledger["evidence_goal"] == 10
    assert ledger["remaining_gap"] == 9
    assert ledger["source_receipts"][0]["consumed"] is True
    assert ledger["source_receipts"][1]["observed"] == "quarantined"
    assert ledger["source_receipts"][1]["consumed"] is False


def test_scenario_7305_historical_ledger_rejects_other_unsafe_shapes(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7305-CUMULATIVE-SEPARATION rejects malformed history."""

    missing = tmp_path / "missing.json"
    non_object = tmp_path / "non_object.json"
    incomplete = tmp_path / "incomplete.json"
    invalid_checksum = tmp_path / "invalid_checksum.json"
    missing_rows = tmp_path / "missing_rows.json"
    mixed_rows = tmp_path / "mixed_rows.json"
    non_object.write_text("[]", encoding="utf-8")
    incomplete.write_text(json.dumps({"status": "running"}), encoding="utf-8")
    invalid_checksum.write_text(
        json.dumps(
            {
                "status": "complete",
                "arc_session_complete_score": 1,
                "reproducibility_checksum": "sha256:" + "0" * 64,
                "cumulative_induction_rows": [],
            }
        ),
        encoding="utf-8",
    )
    missing_rows.write_text(
        json.dumps({"status": "complete", "arc_session_complete_score": 1}),
        encoding="utf-8",
    )
    mixed_rows.write_text(
        json.dumps(
            {
                "status": "complete",
                "arc_session_complete_score": 1,
                "cumulative_induction_rows": ["bad", {"engaged": False}],
            }
        ),
        encoding="utf-8",
    )

    ledger = exp.build_cumulative_ledger(
        tmp_path / "ledger.json",
        [missing, non_object, incomplete, invalid_checksum, missing_rows, mixed_rows],
    )

    assert [row["observed"] for row in ledger["source_receipts"]] == [
        "missing_or_unreadable",
        "missing_or_unreadable",
        "nonterminal_or_incomplete",
        "checksum_invalid",
        "induction_rows_missing",
        "available",
    ]
    assert ledger["authenticated_unique_inductions"] == 0


def test_scenario_7305_reduction_separates_capture_and_mechanism(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7305 allows complete capture with a zero-use null."""

    episode = _episode(tmp_path)
    null_events = deepcopy(_tool_events())
    null_events[0]["dispatch_result"] = {"ok": False}
    reduction = exp.reduce_current_session(episode, _boundary(), null_events)

    assert reduction["arc_capture_complete_score"] == 1
    assert reduction["arc_tool_use_score"] == 0
    assert reduction["verdict_class"] == "null"
    assert reduction["invocation_counts"] == _boundary()["invocation_counts"]


def test_scenario_7305_terminal_artifact_validates_and_block_is_terminal(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7305-TERMINAL-PUBLICATION separates complete and blocked."""

    episode = _episode(tmp_path)
    selection = exp.freeze_target(
        {"games": [{"game": "r11l", "reproducibility": "reproduced", "levels_reproduced": 6}]},
        adaptered_games={"r11l"},
    )
    artifact = exp.build_terminal_artifact(
        started_at_utc="2026-09-14T00:00:00+00:00",
        ended_at_utc="2026-09-14T00:02:00+00:00",
        duration_s=120.0,
        preconditions=[exp.gate_check("fixture", "test", "ready", True, True)],
        source_hashes={},
        selection=selection,
        episode=episode,
        boundary=_boundary(),
        runner={"task_linked_cuda_execution": True},
        tool_events=_tool_events(),
        cumulative_ledger={
            "path": "raw/historical.json",
            "sha256": "sha256:" + "h" * 64,
            "authenticated_unique_inductions": 10,
            "remaining_gap": 0,
        },
        validation_receipts=[{"name": "fixture", "passed": True, "exit_code": 0}],
        phase_spans=[{"phase": "live", "duration_s": 100.0}],
    )
    assert artifact["arc_capture_complete_score"] == 1
    assert artifact["arc_tool_use_score"] == 1
    assert artifact["verdict_class"] == "circular_positive"
    assert exp.validate_artifact(artifact) == []

    failed = exp.gate_check(
        "dependency_gate", exp.EXP7304_PATH.as_posix(), "status", "complete", "missing"
    )
    blocked = exp.build_blocked_artifact(
        started_at_utc="2026-09-14T00:00:00+00:00",
        ended_at_utc="2026-09-14T00:00:01+00:00",
        duration_s=1.0,
        preconditions=[failed],
        source_hashes={},
    )
    assert blocked["status"] == "blocked"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["honest_verdict"].startswith("blocked_")
    assert blocked["gate_check_summary"]["first_failure"] == failed
    assert exp.validate_artifact(blocked) == []


def test_scenario_7305_terminal_validator_rejects_inconsistent_claims(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7305-TERMINAL-PUBLICATION rejects contradictory claims."""

    episode = _episode(tmp_path)
    selection = exp.freeze_target(
        {"games": [{"game": "r11l", "reproducibility": "reproduced"}]},
        adaptered_games=set(),
    )
    boundary = _boundary()
    boundary["call_rows"] = [
        {
            "model_identity": {
                "model_repository": exp.MODEL_ID,
                "model_filename": "Qwen3.8-27B-Q4_K_M.gguf",
                "model_revision": "a" * 40,
                "model_path": "/cache/Qwen3.8-27B-Q4_K_M.gguf",
            }
        }
    ]
    complete = exp.build_terminal_artifact(
        started_at_utc="2026-09-14T00:00:00+00:00",
        ended_at_utc="2026-09-14T00:02:00+00:00",
        duration_s=120.0,
        preconditions=[exp.gate_check("fixture", "test", "ready", True, True)],
        source_hashes={},
        selection=selection,
        episode=episode,
        boundary=boundary,
        runner={"task_linked_cuda_execution": True},
        tool_events=_tool_events(),
        cumulative_ledger={},
        validation_receipts=[],
        phase_spans=[],
    )
    assert complete["MODEL_SPECS"][0]["model_revision"] == "a" * 40
    failed_validation = exp.build_terminal_artifact(
        started_at_utc="2026-09-14T00:00:00+00:00",
        ended_at_utc="2026-09-14T00:02:00+00:00",
        duration_s=120.0,
        preconditions=[exp.gate_check("fixture", "test", "ready", True, True)],
        source_hashes={},
        selection=selection,
        episode=episode,
        boundary=boundary,
        runner={"task_linked_cuda_execution": True},
        tool_events=_tool_events(),
        cumulative_ledger={},
        validation_receipts=[{"passed": False}],
        phase_spans=[],
    )
    assert failed_validation["verdict_class"] == "disqualified"
    assert failed_validation["acceptance_gate_results"][-1]["passed"] is False

    mutations = [
        ("schema", "bad", "identity_mismatch"),
        ("run_date", "bad", "run_date_mismatch"),
        ("status", "running", "status_not_terminal"),
        ("verdict_class", "unknown", "verdict_class_invalid"),
        ("honest_verdict", "bad", "honest_verdict_prefix_invalid"),
        ("field_principles", {}, "field_principles_incomplete"),
        ("official_score", 1, "official_score_must_be_null"),
        ("registry_modified", True, "forbidden_state_change"),
        ("per_game_results", [], "episode_accounting_invalid"),
        ("arc_capture_complete_score", 0, "capture_score_inconsistent"),
        ("arc_tool_use_score", 0, "tool_score_inconsistent"),
        ("duration_s", 1, "substrate_duration_floor_failed"),
    ]
    for field, value, expected_error in mutations:
        changed = deepcopy(complete)
        changed[field] = value
        assert expected_error in exp.validate_artifact(changed)

    failed = exp.gate_check("dependency", "upstream", "ready", True, False)
    blocked = exp.build_blocked_artifact(
        started_at_utc="2026-09-14T00:00:00+00:00",
        ended_at_utc="2026-09-14T00:00:01+00:00",
        duration_s=1,
        preconditions=[failed],
        source_hashes={},
    )
    invoked = deepcopy(blocked)
    invoked["model_invoked"] = True
    assert "blocked_invocation_evidence_invalid" in exp.validate_artifact(invoked)
    no_failure = deepcopy(blocked)
    no_failure["gate_check_summary"]["first_failure"] = None
    assert "blocked_failure_summary_missing" in exp.validate_artifact(no_failure)


def test_req_7305_validation_plan_and_thin_entrypoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-7305 scopes checks, E2E, strict lint, and the wrapper."""

    commands = exp.build_validation_commands(
        terminal_candidate=tmp_path / "candidate.json",
        raw_row=tmp_path / "row.json",
        private_root=tmp_path / "private",
    )
    by_name = {row["name"]: row for row in commands}
    assert {
        "focused_exp7305",
        "affected_boundary_receipts",
        "affected_live_policy",
        "e2e_009_cross_call_persistence",
        "e2e_010_tool_transport",
        "e2e_009_llm_off_environment_smoke",
        "changed_module_coverage_report",
        "ruff_check",
        "ruff_format",
        "changed_module_mypy",
        "scoped_spec_coverage",
        "independent_raw_reducer",
        "terminal_candidate_adversarial_verify",
        "terminal_candidate_row_consistency_strict",
    } <= set(by_name)
    assert "--strict" in by_name["terminal_candidate_row_consistency_strict"]["argv"]
    assert all("scripts/research_conductor.py" not in row["argv"] for row in commands)

    monkeypatch.setattr(exp, "main", lambda argv=None: 0)
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(REPO / exp.WRAPPER_PATH), run_name="__main__")
    assert stopped.value.code == 0

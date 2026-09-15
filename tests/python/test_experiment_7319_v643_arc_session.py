"""Tests for one authority-backed live selfparse session.

Spec refs: REQ-ARC-WMTE-7319 and SCENARIO-ARC-WMTE-7319-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy

import pytest

from carnot import experiment_7319_v643_arc_session as exp


REPO = Path(__file__).resolve().parents[2]


def _hash(index: int) -> str:
    return f"sha256:{index:064x}"


def _authority_artifact() -> dict[str, object]:
    artifact: dict[str, object] = {
        "schema": "carnot.experiment_7318.v643.arc_authority.v1",
        "experiment_id": "exp7318-arc-authority",
        "milestone": "2026.09.643",
        "status": "complete",
        "verdict_class": "null",
        "arc_authority_ready_score": 1,
        "live_entrypoint_receipt": {"file_hashes": {}},
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = exp.authority_artifact_checksum(artifact)
    return artifact


def _boundary() -> dict[str, object]:
    return {
        "activity_known": True,
        "disqualified": False,
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
            "usable_answers": 1,
        },
        "call_rows": [
            {
                "model_identity": {
                    "model_repository": exp.MODEL_ID,
                    "model_filename": "Qwen3.8-27B-Q4_K_M.gguf",
                    "model_revision": "a" * 40,
                    "model_path": "/cache/Qwen3.8-27B-Q4_K_M.gguf",
                }
            }
        ],
    }


def _episode(tmp_path: Path) -> dict[str, object]:
    bounded = '<tool_response>\n{"ok": true, "rows": [1]}\n</tool_response>'
    first = tmp_path / "00_request.json"
    second = tmp_path / "01_request.json"
    first.write_text(json.dumps({"messages": [{"content": "observe"}]}), encoding="utf-8")
    second.write_text(json.dumps({"messages": [{"content": bounded}]}), encoding="utf-8")
    return {
        "episode_id": "re86:direct_selfparse",
        "game": "re86",
        "seed": exp.EVALUATION_SEED,
        "arm": "direct_selfparse",
        "disposition": "complete",
        "censored": False,
        "model_loaded": True,
        "model_invoked": True,
        "generation_calls_attempted": 2,
        "generation_calls_completed": 2,
        "generated_tokens": 3072,
        "action_count": 81,
        "action_limit": exp.ACTION_LIMIT,
        "levels": 0,
        "adapter_disabled": True,
        "banked_solution_disabled": True,
        "fresh_store": True,
        "heldout_accuracy": 0.75,
        "identity_baseline_accuracy": 0.5,
        "changed_cell_prediction_metrics": {
            "learned_accuracy": 0.75,
            "identity_accuracy": 0.5,
            "same_changed_cells": True,
        },
        "induction_rows": [
            {
                "planned": True,
                "attempt_index": 0,
                "reason": "stall",
                "started_at": "2026-09-15T12:00:00+00:00",
                "tool_gap": {
                    "selfparse": True,
                    "tool_calls_total": 1,
                    "terminated_by": "final_answer",
                },
            }
        ],
        "policy_consumption_rows": [
            {
                "attempt_index": 0,
                "action_index": 80,
                "policy_action_executed": True,
                "engine_sha256": _hash(90),
                "plan_sha256": _hash(91),
            }
        ],
        "raw_request_manifest": [
            {"call_index": 0, "request_path": str(first)},
            {"call_index": 1, "request_path": str(second)},
        ],
        "action_rows": [{"i": 80, "action": 1, "top_branch": "execute.plan_step"}],
        "error": None,
    }


def _tool_events() -> list[dict[str, object]]:
    return [
        {
            "episode_id": "re86:direct_selfparse",
            "induction_index": 0,
            "turn": 0,
            "parsed_tool": "list_transitions",
            "parsed_arguments": {"limit": 4},
            "dispatch_result": {"ok": True, "rows": [1]},
            "bounded_response": '<tool_response>\n{"ok": true, "rows": [1]}\n</tool_response>',
            "exception": None,
        }
    ]


def test_req_7319_spec_and_fixed_live_limits() -> None:
    """REQ-ARC-WMTE-7319 fixes the model, date, and inherited ceilings."""

    spec = (REPO / exp.SPEC_PATH).read_text(encoding="utf-8")
    assert "## REQ-ARC-WMTE-7319:" in spec
    assert "SCENARIO-ARC-WMTE-7319-CAUSAL-TOOL-USE" in spec
    assert exp.RUN_DATE == "20260915"
    assert exp.MODEL_SPECS == [{"hf_id": exp.MODEL_ID, "quantization": "Q4_K_M"}]
    assert (exp.ACTION_LIMIT, exp.COMPLETION_LIMIT, exp.GENERATED_TOKEN_LIMIT) == (192, 2, 4096)
    assert (exp.SESSION_LIMIT_S, exp.MODEL_LOAD_LIMIT_S) == (3000, 600)


@pytest.mark.parametrize(
    ("change", "field", "expected", "observed"),
    [
        (None, "artifact", "available", "missing"),
        ({"quarantined": True}, "quarantine_state", False, True),
        ({"verdict_class": "disqualified"}, "verdict_class", "not_disqualified", "disqualified"),
        ({"verdict_class": "blocked"}, "verdict_class", "not_blocked", "blocked"),
        ({"verdict_class": "partial"}, "verdict_class", "not_partial", "partial"),
        ({"status": "running"}, "status", "complete", "running"),
        ({"arc_authority_ready_score": 0}, "arc_authority_ready_score", 1, 0),
    ],
)
def test_scenario_7319_dependency_fails_before_unsafe_score_use(
    change: dict[str, object] | None,
    field: str,
    expected: object,
    observed: object,
) -> None:
    """SCENARIO-ARC-WMTE-7319-DEPENDENCY-BLOCK keeps the exact first failure."""

    baseline = _authority_artifact()
    artifact = None if change is None else {**baseline, **change}
    row = exp.check_dependency(artifact)

    assert row["passed"] is False
    assert row["upstream"] == exp.EXP7318_PATH.as_posix()
    assert row["artifact_field"] == field
    assert row["expected"] == expected
    assert row["observed"] == observed
    assert exp.gate_summary([row])["first_failure"] == {
        "upstream": exp.EXP7318_PATH.as_posix(),
        "check": "exp7318_dependency",
        "artifact_field": field,
        "expected": expected,
        "observed": observed,
    }


def test_scenario_7319_dependency_checksum_and_caller_hashes(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7319-DEPENDENCY-BLOCK authenticates terminal and caller bytes."""

    caller = tmp_path / "caller.py"
    caller.write_text("value = 1\n", encoding="utf-8")
    artifact = _authority_artifact()
    artifact["live_entrypoint_receipt"] = {
        "file_hashes": {
            "scored_caller": {
                "path": "caller.py",
                "sha256": exp.sha256_file(caller),
            }
        }
    }
    artifact["reproducibility_checksum"] = exp.authority_artifact_checksum(artifact)

    assert exp.check_dependency(artifact)["passed"] is True
    assert exp.authenticate_caller_hashes(tmp_path, artifact)[0]["passed"] is True
    caller.write_text("value = 2\n", encoding="utf-8")
    mismatch = exp.authenticate_caller_hashes(tmp_path, artifact)[0]
    assert mismatch["passed"] is False
    assert mismatch["upstream"] == "caller.py"
    assert mismatch["artifact_field"] == "sha256"

    missing = deepcopy(artifact)
    missing["live_entrypoint_receipt"] = {}
    assert exp.authenticate_caller_hashes(tmp_path, missing)[0]["observed"] == "missing"
    malformed = deepcopy(artifact)
    malformed["live_entrypoint_receipt"] = {"file_hashes": {"scored_caller": 7}}
    assert exp.authenticate_caller_hashes(tmp_path, malformed)[0]["artifact_field"] == (
        "file_hash_receipt"
    )
    absent = deepcopy(artifact)
    absent["live_entrypoint_receipt"] = {
        "file_hashes": {"scored_caller": {"path": "", "sha256": _hash(99)}}
    }
    assert exp.authenticate_caller_hashes(tmp_path, absent)[0]["observed"] == "missing"
    unrelated = deepcopy(artifact)
    unrelated["live_entrypoint_receipt"] = {
        "file_hashes": {"environment_construction": {"path": "caller.py"}}
    }
    assert exp.authenticate_caller_hashes(tmp_path, unrelated)[0]["observed"] == "missing"


def test_scenario_7319_rotation_selects_least_recent_and_withholds_adapter() -> None:
    """SCENARIO-ARC-WMTE-7319-FROZEN-WITHHELD-TARGET uses metadata before outcomes."""

    registry = {
        "games": [
            {"game": "r11l", "reproducibility": "reproduced", "levels_reproduced": 6},
            {"game": "re86", "reproducibility": "reproduced", "levels_reproduced": 8},
        ]
    }
    receipt = exp.freeze_target(
        registry,
        adaptered_games={"r11l", "re86"},
        previous_target="r11l",
    )

    assert receipt["passed"] is True
    assert receipt["target"] == "re86"
    assert receipt["rotation"] == ["r11l", "re86"]
    assert receipt["least_recently_measured_basis"] == {"r11l": "exp7305", "re86": None}
    assert receipt["adapter_available_but_withheld"] is True
    assert receipt["adapter_disabled"] is True
    assert receipt["banked_solution_disabled"] is True
    assert receipt["outcomes_seen_before_freeze"] is False
    assert receipt["game_source_read"] is False
    assert receipt["offline_ground_truth_search_used"] is False


def test_scenario_7319_environment_preserves_selfparse_and_bounds(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7319 keeps the authenticated live runner settings."""

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
    assert env["CARNOT_ARC_CEGIS_TOOL_LOOP"] == "1"
    assert env["CARNOT_ARC_INDUCE_TOOL_TURNS"] == "2"
    assert env["CARNOT_ARC_INDUCE_MAX_TOKENS"] == "2048"
    assert env["CARNOT_ARC_INDUCE_N_CTX"] == "49152"
    assert env["CARNOT_ARC_INDUCE_TIMEOUT"] == "600"
    assert env["CARNOT_ARC_BOUNDARY_LEDGER_PATH"] == str(ledger)
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert "CARNOT_ARC_SUPERVISOR_TOOL_ARM" not in env


def test_scenario_7319_tool_result_requires_the_complete_causal_chain(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7319-CAUSAL-TOOL-USE binds result through action."""

    chain = exp.reduce_tool_use_chain(_episode(tmp_path), _tool_events())

    assert chain["arc_tool_use_score"] == 1
    row = chain["rows"][0]
    assert row["tool_name"] == "list_transitions"
    assert row["tool_arguments"] == {"limit": 4}
    assert row["runtime_result"] == {"ok": True, "rows": [1]}
    assert row["later_request_path"].endswith("01_request.json")
    assert row["policy_consumption"]["attempt_index"] == 0
    assert row["environment_action"]["action"] == 1
    assert row["absent_links"] == []
    assert row["actual_exception"] is None
    assert row["passed"] is True

    episode = _episode(tmp_path)
    episode["action_rows"] = []
    missing = exp.reduce_tool_use_chain(episode, _tool_events())
    assert missing["arc_tool_use_score"] == 0
    assert missing["rows"][0]["absent_links"] == ["environment_action"]

    failed_events = _tool_events()
    failed_events[0]["dispatch_result"] = {"ok": False, "error": "rejected"}
    failed_events[0]["exception"] = "ValueError: rejected"
    failed = exp.reduce_tool_use_chain(_episode(tmp_path), failed_events)
    assert failed["rows"][0]["absent_links"] == [
        "runtime_result",
        "later_request",
        "installed_plan",
        "policy_consumption",
        "environment_action",
    ]
    assert failed["rows"][0]["actual_exception"] == "ValueError: rejected"

    unmatched = _tool_events()
    unmatched[0]["bounded_response"] = "<tool_response>absent</tool_response>"
    no_later_request = exp.reduce_tool_use_chain(_episode(tmp_path), unmatched)
    assert "later_request" in no_later_request["rows"][0]["absent_links"]


def test_scenario_7319_current_induction_projection_counts_invalid_and_raw_rows() -> None:
    """SCENARIO-ARC-WMTE-7319-TERMINAL-ACCOUNTING hashes only authentic attempts."""

    rows, censored = exp._current_induction_rows(
        {
            "episode_id": "re86:direct_selfparse",
            "induction_rows": [
                "invalid",
                {
                    "attempt_index": 1,
                    "reason": "stall",
                    "started_at": "2026-09-15T12:00:00+00:00",
                    "tool_gap": {
                        "selfparse": True,
                        "tool_calls_total": 1,
                        "terminated_by": "final_answer",
                    },
                },
            ],
        }
    )
    assert censored == 1
    assert len(rows) == 1
    assert exp.HASH_RE.fullmatch(rows[0]["induction_id"])
    assert rows[0]["content_hash_basis"] == "sealed_current_attempt_identity"


def test_scenario_7319_historical_ledger_keeps_ten_and_separates_counts(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-7319-TERMINAL-ACCOUNTING deduplicates content hashes."""

    rows = [
        {
            "induction_id": _hash(index),
            "engaged": True,
            "source_authenticated": True,
        }
        for index in range(10)
    ]
    sidecar = tmp_path / "historical.json"
    exp.atomic_write(
        sidecar,
        {
            "schema": "carnot.experiment_7305.cumulative_induction_ledger.v1",
            "authenticated_unique_inductions": 10,
            "source_receipts": [{"accepted_inductions": 21}],
            "rows": rows,
        },
    )
    upstream = tmp_path / "exp7305.json"
    prior: dict[str, object] = {
        "status": "complete",
        "verdict_class": "null",
        "cumulative_induction_ledger": {
            "authenticated_unique_inductions": 10,
            "sha256": exp.sha256_file(sidecar),
        },
        "reproducibility_checksum": "",
    }
    prior["reproducibility_checksum"] = exp.prior_artifact_checksum(prior)
    exp.atomic_write(upstream, prior)
    episode = _episode(tmp_path)
    episode["induction_rows"] = [
        {**rows[0], "attempt_index": 0},
        {**rows[0], "attempt_index": 1},
        {
            "induction_id": _hash(20),
            "engaged": True,
            "source_authenticated": True,
            "attempt_index": 2,
        },
        {"attempt_index": 3, "engaged": False},
    ]
    output = tmp_path / "merged.json"

    ledger = exp.build_cumulative_ledger(output, upstream, sidecar, episode)

    assert ledger["historical_authenticated_count"] == 10
    assert ledger["historical_duplicate_count"] == 11
    assert ledger["new_authentic_count"] == 1
    assert ledger["current_duplicate_count"] == 2
    assert ledger["censored_attempt_count"] == 1
    assert ledger["cumulative_total"] == 11
    assert ledger["historical_ledger_authenticated"] is True
    assert ledger["sha256"] == exp.sha256_file(output)

    rejected = exp.build_cumulative_ledger(
        tmp_path / "rejected.json",
        tmp_path / "missing-upstream.json",
        tmp_path / "missing-ledger.json",
        {"induction_rows": []},
    )
    assert rejected["historical_ledger_authenticated"] is False
    assert rejected["historical_authenticated_count"] == 0


def test_scenario_7319_session_reducer_separates_completion_from_tool_score(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-7319-TERMINAL-ACCOUNTING allows an honest null."""

    episode = _episode(tmp_path)
    complete = exp.reduce_current_session(episode, _boundary(), _tool_events())
    assert complete["arc_session_complete_score"] == 1
    assert complete["arc_tool_use_score"] == 1
    assert complete["verdict_class"] == "circular_positive"

    episode["action_rows"] = []
    null = exp.reduce_current_session(episode, _boundary(), _tool_events())
    assert null["arc_session_complete_score"] == 1
    assert null["arc_tool_use_score"] == 0
    assert null["verdict_class"] == "null"


def test_req_7319_terminal_and_blocked_artifacts_validate(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7319 preserves every required field and blocks safely."""

    episode = _episode(tmp_path)
    ledger = {
        "historical_authenticated_count": 10,
        "historical_duplicate_count": 11,
        "new_authentic_count": 1,
        "current_duplicate_count": 0,
        "censored_attempt_count": 0,
        "cumulative_total": 11,
        "historical_ledger_authenticated": True,
    }
    artifact = exp.build_terminal_artifact(
        started_at_utc="2026-09-15T12:00:00+00:00",
        ended_at_utc="2026-09-15T12:02:00+00:00",
        duration_s=120.0,
        preconditions=[
            exp.gate_check("dependency", "authority.json", "status", "complete", "complete")
        ],
        source_hashes={"authority.json": {"sha256": _hash(30)}},
        selection={"target": "re86", "passed": True},
        episode=episode,
        boundary=_boundary(),
        runner={
            "task_linked_cuda_execution": True,
            "runner": "LocalGGUFProposer_native_llama.cpp",
            "lease_owner": {"lease_id": "lease:owned"},
        },
        tool_events=_tool_events(),
        cumulative_ledger=ledger,
        validation_receipts=[
            {"name": name, "passed": True, "exit_code": 0} for name in exp.REQUIRED_CHECK_NAMES
        ],
        repository_health={"status": "degraded_open", "affects_required_checks": False},
        phase_spans=[{"phase": "live_window", "duration_s": 100.0, "units": 1}],
    )

    assert exp.validate_artifact(artifact) == []
    assert artifact["arc_session_complete_score"] == 1
    assert artifact["arc_tool_use_score"] == 1
    assert artifact["inference_substrate"] == "model_full_generation"
    assert artifact["inference_substrate_class"] == "model_full_generation"
    assert artifact["inference_mode"] == "live_gpu"
    assert artifact["execution_venue"] == "host"
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["official_score"] is None
    assert artifact["registry_modified"] is False
    assert set(artifact["field_principles"]) == set(artifact)

    fallback_episode = deepcopy(episode)
    fallback_episode.pop("changed_cell_prediction_metrics")
    fallback = exp.build_terminal_artifact(
        started_at_utc="2026-09-15T12:00:00+00:00",
        ended_at_utc="2026-09-15T12:02:00+00:00",
        duration_s=120.0,
        preconditions=[],
        source_hashes={},
        selection={"target": "re86", "passed": True},
        episode=fallback_episode,
        boundary=_boundary(),
        runner={"task_linked_cuda_execution": True, "model_count": 1},
        tool_events=_tool_events(),
        cumulative_ledger=ledger,
        validation_receipts=[
            {"name": name, "passed": True, "exit_code": 0} for name in exp.REQUIRED_CHECK_NAMES
        ],
        repository_health={"status": "healthy", "affects_required_checks": False},
        phase_spans=[],
    )
    assert fallback["rows"][0]["metrics"]["changed_cell_predictions"] == {
        "learned_accuracy": 0.75,
        "identity_accuracy": 0.5,
        "same_changed_cells": True,
    }

    failed = exp.gate_check(
        "exp7318_dependency",
        exp.EXP7318_PATH.as_posix(),
        "artifact",
        "available",
        "missing",
    )
    blocked = exp.build_blocked_artifact(
        started_at_utc="2026-09-15T12:00:00+00:00",
        ended_at_utc="2026-09-15T12:00:01+00:00",
        duration_s=1.0,
        preconditions=[failed],
        source_hashes={},
        model_specs=[],
    )
    assert exp.validate_artifact(blocked) == []
    assert blocked["status"] == "blocked"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["arc_session_complete_score"] == 0
    assert blocked["arc_tool_use_score"] == 0
    assert "artifact:expected='available':observed='missing'" in blocked["honest_verdict"]

    terminal_mutations = {
        "identity_mismatch": {"schema": "wrong"},
        "run_date_mismatch": {"run_date": "20260914"},
        "status_not_terminal": {"status": "running"},
        "verdict_class_invalid": {"verdict_class": "unknown"},
        "honest_verdict_prefix_invalid": {"honest_verdict": "wrong"},
        "reproducibility_checksum_mismatch": {"reproducibility_checksum": "wrong"},
        "field_principles_incomplete": {"field_principles": {}},
        "official_score_must_be_null": {"official_score": 1},
        "forbidden_claim_or_state_change": {"new_solve_claimed": True},
        "execution_venue_invalid": {"execution_venue": "kv260"},
        "episode_accounting_invalid": {
            "per_game_results": [{**episode, "action_count": exp.ACTION_LIMIT + 1}]
        },
        "disqualified_scores_nonzero": {"verdict_class": "disqualified"},
        "non_disqualified_validation_failure": {"validation_receipts": []},
        "historical_induction_ledger_invalid": {
            "cumulative_induction_ledger": {
                **ledger,
                "historical_ledger_authenticated": False,
            }
        },
        "current_model_spec_invalid": {"MODEL_SPECS": []},
        "substrate_duration_floor_failed": {"duration_s": 1},
        "live_gpu_receipt_missing": {"inference_mode": "not_verified"},
    }
    for expected_error, changes in terminal_mutations.items():
        mutated = {**deepcopy(artifact), **changes}
        assert expected_error in exp.validate_artifact(mutated)

    blocked_mutations = {
        "blocked_model_invoked": {"model_invoked": True},
        "blocked_invocation_counts_invalid": {
            "invocation_counts": {**exp.ZERO_INVOCATION_COUNTS, "model_loads_attempted": 1}
        },
        "blocked_scores_nonzero": {"arc_session_complete_score": 1},
        "blocked_failure_summary_missing": {
            "gate_check_summary": {"all_passed": False, "first_failure": None}
        },
    }
    for expected_error, changes in blocked_mutations.items():
        mutated = {**deepcopy(blocked), **changes}
        assert expected_error in exp.validate_artifact(mutated)


def test_req_7319_cold_replay_checks_invocations_and_actions(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7319 cold-replays current invocation and action receipts."""

    payload = {
        "episode": _episode(tmp_path),
        "boundary": _boundary(),
        "tool_events": _tool_events(),
    }
    path = tmp_path / "raw.json"
    exp.atomic_write(path, payload)
    replay = exp.independent_reduce(path)
    assert replay["arc_session_complete_score"] == 1
    assert replay["arc_tool_use_score"] == 1
    assert replay["action_receipts_replayed"] == 1
    assert replay["generation_receipts_replayed"] == 2
    assert replay["cold_replay_passed"] is True


def test_req_7319_validation_scope_is_explicit_and_wrapper_is_thin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-7319 uses the shipped scoped runner and a thin wrapper."""

    captured: dict[str, object] = {}

    def fake_runner(
        root: Path, tests: list[str], modules: list[str], **kwargs: object
    ) -> dict[str, object]:
        captured.update({"root": root, "tests": tests, "modules": modules, **kwargs})
        return {
            "validation_receipts": [],
            "required_checks_passed": True,
            "repository_health": {"status": "healthy", "affects_required_checks": False},
        }

    monkeypatch.setattr(exp.validation_scope, "run_scoped_validation", fake_runner)
    receipt = exp.run_scoped_validation(tmp_path)
    assert receipt["required_checks_passed"] is True
    assert captured["tests"] == [exp.TEST_PATH.as_posix()]
    assert captured["modules"] == [exp.MODULE_PATH.as_posix()]
    assert "full_python_suite" not in json.dumps(
        {"tests": captured["tests"], "modules": captured["modules"]}
    )

    wrapper = REPO / exp.WRAPPER_PATH
    assert len(wrapper.read_text(encoding="utf-8").splitlines()) <= 20
    monkeypatch.setattr(exp, "main", lambda: 0)
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(wrapper), run_name="__main__")
    assert stopped.value.code == 0

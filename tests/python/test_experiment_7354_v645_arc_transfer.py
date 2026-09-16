"""Test the bounded live result-resume transfer pilot.

Spec refs: REQ-ARC-WMTE-7354 and SCENARIO-ARC-WMTE-7354-*.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import runpy

import pytest

from carnot import experiment_7354_v645_arc_transfer as exp


REPO = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.memory_watchdog_skip


def _request(call: int, content: str = "") -> dict[str, object]:
    body = json.dumps({"messages": [{"role": "user", "content": content}]})
    return {
        "call_index": call,
        "transport_completed": True,
        "completion_tokens": 20,
        "prompt_tokens": 100,
        "request_body": body,
        "request_sha256": "sha256:" + hashlib.sha256(body.encode()).hexdigest(),
        "response_sha256": f"sha256:{call + 11:064x}",
        "error": None,
    }


def _episode(game: str, arm: str, *, levels: int = 1, actions: int = 5) -> dict[str, object]:
    result = '<tool_response>{"ok": true}</tool_response>'
    treatment = arm == "result_resume"
    attempt_id = f"attempt:{game}"
    receipt: dict[str, object]
    requests = [_request(0)]
    induction_rows: list[dict[str, object]] = []
    consumption: list[dict[str, object]] = []
    if treatment:
        requests.append(_request(1, result))
        receipt = {
            "enabled": True,
            "episode_id": game,
            "attempt_id": attempt_id,
            "completion_limit": 2,
            "result_rows": [
                {
                    "episode_id": game,
                    "attempt_id": attempt_id,
                    "source_request_id": "request:0",
                    "next_request_id": "request:1",
                    "result_id": f"result:{game}",
                    "bounded_response": result,
                    "dispatch_results": [{"ok": True}],
                    "tool_names": ["diff_grids"],
                    "delivery_count": 1,
                    "receipt_captured": True,
                }
            ],
            "request_rows": [
                {
                    "episode_id": game,
                    "attempt_id": attempt_id,
                    "request_id": "request:1",
                    "result_id": f"result:{game}",
                    "response_received": True,
                    "receipt_captured": True,
                }
            ],
            "rejections": [],
        }
        induction_rows = [
            {
                "planned": True,
                "engine_source_sha256": f"sha256:{31:064x}",
                "refinement_rounds": [{"accepted_by_heldout_verifier": True}],
            }
        ]
        consumption = [
            {
                "attempt_index": 0,
                "policy_action_executed": True,
                "action_index": 2,
                "engine_sha256": f"sha256:{31:064x}",
                "plan_sha256": f"sha256:{32:064x}",
                "action": 1,
                "data": None,
            }
        ]
    else:
        receipt = {"enabled": False}
    return {
        "episode_id": f"{game}:{arm}",
        "game": game,
        "arm": arm,
        "seed": exp.EVALUATION_SEED,
        "disposition": "complete",
        "censored": False,
        "fresh_store": True,
        "adapter_disabled": True,
        "banked_solution_disabled": True,
        "off_path_engines_disabled": True,
        "action_limit": exp.ACTION_LIMIT,
        "completion_limit": exp.COMPLETION_LIMIT,
        "generated_token_limit": exp.GENERATED_TOKEN_LIMIT,
        "action_count": actions,
        "generation_calls_attempted": len(requests),
        "generation_calls_completed": len(requests),
        "generated_tokens": 20 * len(requests),
        "levels": levels,
        "factory_receipt": {
            "factory": "make_carnot_agent",
            "policy_class": "E3AgentPolicy",
            "adapter_disabled": True,
            "denied_inputs": ["banked_solutions", "game_adapter", "game_source"],
        },
        "raw_request_manifest": requests,
        "induction_rows": induction_rows,
        "policy_consumption_rows": consumption,
        "action_rows": [{"i": index, "action": 1, "data": None} for index in range(actions)],
        "error": None,
        "evidence_receipt": receipt,
    }


def _raw_panel() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for game in exp.TARGET_ROTATION:
        rows.extend((_episode(game, "result_resume"), _episode(game, "result_withheld")))
    return {"schedule": exp.matched_schedule(exp.TARGET_ROTATION), "episodes": rows}


def _receipt(name: str, passed: bool = True) -> dict[str, object]:
    return {
        "name": name,
        "command": f"fixture {name}",
        "command_argv": ["fixture", name],
        "scope": "fixture",
        "exit_code": 0 if passed else 1,
        "duration_s": 0.01,
        "log_path": f"/tmp/{name}.log",
        "log_sha256": f"sha256:{len(name):064x}",
        "passed": passed,
        "timed_out": False,
    }


def test_req_7354_dependency_fail_closed_and_qualified() -> None:
    """SCENARIO-ARC-WMTE-7354-UPSTREAM-BLOCK checks fields before use."""

    spec = (REPO / exp.SPEC_PATH).read_text(encoding="utf-8")
    assert "## REQ-ARC-WMTE-7354:" in spec
    upstream = json.loads((REPO / exp.EXP7345_PATH).read_text(encoding="utf-8"))
    qualified = exp.check_dependency(upstream)
    assert qualified["passed"] is True
    assert qualified["first_failure"] is None

    cases = (
        (None, "artifact", "available", "missing"),
        ({**upstream, "flagged_adversarial": True}, "flagged_adversarial", False, True),
        ({**upstream, "verdict_class": "partial"}, "verdict_class", "terminal_safe", "partial"),
        ({**upstream, "status": "running"}, "status", "complete", "running"),
        ({**upstream, "seal_for_exp7354": False}, "seal_for_exp7354", True, False),
        ({**upstream, "arc_resume_ready_score": 0}, "arc_resume_ready_score", 1, 0),
        (
            {**upstream, "gate_check_summary": {"all_passed": False}},
            "gate_check_summary.all_passed",
            True,
            False,
        ),
    )
    for candidate, field, expected, observed in cases:
        result = exp.check_dependency(candidate)
        assert result["passed"] is False
        assert result["first_failure"]["artifact_field"] == field
        assert result["first_failure"]["expected"] == expected
        assert result["first_failure"]["observed"] == observed


def test_scenario_7354_frozen_matched_panel_and_environment() -> None:
    """SCENARIO-ARC-WMTE-7354-FROZEN-MATCHED-PANEL seals equal live limits."""

    registry = {"games": [{"game": game, "levels_reproduced": 2} for game in exp.TARGET_ROTATION]}
    frozen = exp.freeze_panel(registry, adaptered_games=set(exp.TARGET_ROTATION))
    assert frozen["passed"] is True
    assert frozen["games"] == list(exp.TARGET_ROTATION)
    assert frozen["outcomes_seen_before_freeze"] is False
    schedule = exp.matched_schedule(frozen["games"])
    assert [row["arm"] for row in schedule] == [
        "result_resume",
        "result_withheld",
        "result_withheld",
        "result_resume",
    ]
    assert len({row["episode_id"] for row in schedule}) == 4
    assert all(row["seed"] == exp.EVALUATION_SEED for row in schedule)
    assert all(row["action_limit"] == 192 for row in schedule)
    assert all(row["completion_limit"] == 2 for row in schedule)
    assert all(row["generated_token_limit"] == 4096 for row in schedule)
    assert exp.freeze_panel({"games": []}, adaptered_games=set())["passed"] is False

    treatment = exp.session_environment(
        {}, arm="result_resume", episode_dir=Path("/tmp/treatment"), gpu_index=1, port=9001
    )
    control = exp.session_environment(
        {}, arm="result_withheld", episode_dir=Path("/tmp/control"), gpu_index=1, port=9001
    )
    assert treatment["CARNOT_FORCE_LIVE"] == "1"
    assert treatment["CARNOT_ARC_SELFPARSE_RESULT_RESUME"] == "1"
    assert treatment["CARNOT_ARC_INDUCE_TOOL_TURNS"] == "2"
    assert "CARNOT_ARC_SELFPARSE_RESULT_RESUME" not in control
    assert control["CARNOT_ARC_INDUCE_TOOL_TURNS"] == "1"
    assert treatment["CARNOT_ARC_RANDOM_SEED"] == control["CARNOT_ARC_RANDOM_SEED"]

    # The live runtime temporarily rebinds the reused module's environment function.
    # This call proves the wrapper still reaches the saved production implementation.
    with exp._configured_runtime():
        rebound = exp.session_environment(
            {}, arm="result_resume", episode_dir=Path("/tmp/rebound"), gpu_index=1, port=9001
        )
        bootstrap = exp.session_environment(
            {}, arm="current_feedback", episode_dir=Path("/tmp/bootstrap"), gpu_index=1, port=9001
        )
    assert rebound["CARNOT_ARC_SELFPARSE_RESULT_RESUME"] == "1"
    assert "CARNOT_ARC_SELFPARSE_RESULT_RESUME" not in bootstrap


def test_scenario_7354_independent_reduction_and_mutations(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7354-INDEPENDENT-REDUCTION catches withheld and stale traces."""

    raw = _raw_panel()
    reduced = exp.reduce_raw_panel(raw)
    assert reduced["arc_capture_complete_score"] == 1
    assert reduced["arc_feedback_value_score"] == 1
    assert reduced["attempted_units"] == reduced["completed_units"] == 4
    assert reduced["censored_units"] == 0
    assert reduced["causal_treatment_games"] == list(exp.TARGET_ROTATION)
    assert len(reduced["per_game_results"]) == 4
    assert len(reduced["causal_chain_rows"]) == 2
    assert all(row["chain_complete"] for row in reduced["causal_chain_rows"])
    assert all(row["raw_action_count"] == 5 for row in reduced["per_game_results"])
    assert all(row["raw_generated_tokens"] in {20, 40} for row in reduced["per_game_results"])

    raw_path = tmp_path / "raw.json"
    exp.atomic_write(raw_path, raw)
    assert exp.independent_reduce(raw_path) == reduced
    raw_path.write_text("{", encoding="utf-8")
    assert exp.independent_reduce(raw_path)["arc_capture_complete_score"] == 0

    attacks = exp.run_trace_mutations(raw)
    assert attacks == {
        "result_withheld_detected": True,
        "stale_result_detected": True,
        "baseline_feedback_value_score": 1,
        "result_withheld_feedback_value_score": 0,
        "stale_result_feedback_value_score": 0,
    }

    no_result = deepcopy(raw)
    for episode in no_result["episodes"]:
        if episode["arm"] != "result_resume":
            continue
        receipt = episode.pop("evidence_receipt")
        receipt["result_rows"] = []
        receipt["request_rows"] = []
        episode["evidence_receipts"] = [receipt, deepcopy(receipt)]
    assert exp.run_trace_mutations(no_result) == {
        "result_withheld_detected": True,
        "stale_result_detected": True,
        "baseline_feedback_value_score": 0,
        "result_withheld_feedback_value_score": 0,
        "stale_result_feedback_value_score": 0,
    }

    null = deepcopy(raw)
    treatment = next(row for row in null["episodes"] if row["arm"] == "result_resume")
    treatment["policy_consumption_rows"] = []
    assert exp.reduce_raw_panel(null)["arc_feedback_value_score"] == 0

    regression = deepcopy(raw)
    treatment = next(
        row
        for row in regression["episodes"]
        if row["game"] == exp.TARGET_ROTATION[0] and row["arm"] == "result_resume"
    )
    treatment["levels"] = 0
    assert exp.reduce_raw_panel(regression)["arc_feedback_value_score"] == 0


def test_scenario_7354_reducer_separates_reset_and_environment_actions() -> None:
    """SCENARIO-ARC-WMTE-7354-INDEPENDENT-REDUCTION replays action semantics."""

    episode = _episode("r11l", "result_withheld", actions=5)
    episode["action_rows"][0]["action"] = "RESET"
    episode["action_rows"][1]["action"] = "RESET"
    episode["action_count"] = 3

    summary = exp._summary_row(episode)

    assert summary["authentic_terminal"] is True
    assert summary["raw_action_count"] == 5
    assert summary["raw_environment_action_count"] == 3


def test_req_7354_terminal_and_blocked_artifacts() -> None:
    """SCENARIO-ARC-WMTE-7354-TERMINAL separates accounting from causal value."""

    raw = _raw_panel()
    reduced = exp.reduce_raw_panel(raw)
    names = (
        *exp.REQUIRED_VALIDATION_NAMES,
        *exp.REQUIRED_E2E_NAMES,
        *exp.REQUIRED_TERMINAL_NAMES,
    )
    receipts = [_receipt(name) for name in names]
    artifact = exp.build_terminal_artifact(
        preconditions=exp.check_dependency(
            json.loads((REPO / exp.EXP7345_PATH).read_text(encoding="utf-8"))
        )["checks"],
        source_hashes={"fixture": {"sha256": f"sha256:{1:064x}"}},
        selection={"passed": True, "games": list(exp.TARGET_ROTATION)},
        raw_panel=raw,
        reduction=reduced,
        mutations=exp.run_trace_mutations(raw),
        runtime_identity={"task_linked_cuda_execution": True},
        model_specs=[
            {
                **exp.MODEL_SPECS[0],
                "model_path": "/tmp/model-Q4_K_M.gguf",
                "sha256": f"sha256:{2:064x}",
                "bytes": 123,
            }
        ],
        invocation_counts={
            "model_loads_attempted": 1,
            "model_loads_completed": 1,
            "model_loads_failed": 0,
            "model_loads_cancelled": 0,
            "model_loads_in_flight": 0,
            "generation_calls_attempted": 6,
            "generation_calls_completed": 6,
            "generation_calls_failed": 0,
            "generation_calls_cancelled": 0,
            "generation_calls_in_flight": 0,
            "usable_answers": 6,
        },
        validation_receipts=receipts,
        repository_health={"historical_failures": []},
        started_at_utc="2026-09-16T12:00:00+00:00",
        ended_at_utc="2026-09-16T12:01:00+00:00",
        duration_s=60.0,
        phase_spans=[{"phase": "generation", "duration_s": 40.0, "completed_units": 4}],
    )
    assert exp.validate_artifact(artifact) == []
    assert artifact["status"] == "complete"
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["inference_substrate_class"] == "model_bounded_generation"
    assert artifact["inference_mode"] == "live_gpu"
    assert artifact["arc_capture_complete_score"] == 1
    assert artifact["arc_feedback_value_score"] == 1
    assert artifact["promotion_value"] == 0
    assert artifact["verifier_is_oracle"] is True
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["field_principles"]
    assert set(artifact) <= set(artifact["field_principles"])

    failed = deepcopy(receipts)
    failed[0]["passed"] = False
    failed[0]["exit_code"] = 1
    disqualified = exp.build_terminal_artifact(
        preconditions=artifact["preconditions_checked"],
        source_hashes=artifact["source_artifact_hashes"],
        selection=artifact["selection_receipt"],
        raw_panel=raw,
        reduction=reduced,
        mutations=artifact["trace_mutation_results"],
        runtime_identity=artifact["runtime_identity_receipt"],
        model_specs=artifact["resolved_model_specs"],
        invocation_counts=artifact["invocation_counts"],
        validation_receipts=failed,
        repository_health={},
        started_at_utc=artifact["started_at_utc"],
        ended_at_utc=artifact["ended_at_utc"],
        duration_s=60.0,
        phase_spans=[],
    )
    assert disqualified["verdict_class"] == "disqualified"
    assert disqualified["arc_capture_complete_score"] == 0
    assert disqualified["arc_feedback_value_score"] == 0
    assert exp.validate_artifact(disqualified) == []

    blocked = exp.build_blocked_artifact(
        checks=[
            exp.gate_check(
                "exp7345_dependency",
                exp.EXP7345_PATH.as_posix(),
                "artifact",
                "available",
                "missing",
            )
        ],
        started_at_utc="2026-09-16T12:00:00+00:00",
        ended_at_utc="2026-09-16T12:00:01+00:00",
        duration_s=1.0,
    )
    assert blocked["status"].startswith("blocked_")
    assert blocked["verdict_class"] == "blocked"
    assert blocked["MODEL_SPECS"] == exp.MODEL_SPECS
    assert blocked["model_invoked"] is False
    assert blocked["inference_substrate"] == "blocked_no_run"
    assert blocked["arc_capture_complete_score"] == 0
    assert blocked["arc_feedback_value_score"] == 0
    assert blocked["gate_check_summary"]["first_failure"]["artifact_field"] == "artifact"
    assert exp.validate_artifact(blocked) == []

    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = f"sha256:{0:064x}"
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(changed)
    assert exp.validate_artifact([]) == ["artifact_not_object"]


def test_req_7354_progress_validation_and_thin_entrypoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-ARC-WMTE-7354 uses scoped validation and one thin executable wrapper."""

    captured: dict[str, object] = {}

    def scoped(*args: object, **kwargs: object) -> dict[str, object]:
        captured["args"] = args
        captured["kwargs"] = kwargs
        assert Path(str(kwargs["basetemp"])).is_dir()
        assert Path(str(kwargs["coverage_file"])).parent.is_dir()
        return {"validation_receipts": [_receipt(name) for name in exp.REQUIRED_VALIDATION_NAMES]}

    monkeypatch.setattr(exp.validation_scope, "run_scoped_validation", scoped)
    receipts = exp.run_scoped_validation(tmp_path, tmp_path / "private")
    assert [row["name"] for row in receipts] == list(exp.REQUIRED_VALIDATION_NAMES)
    assert captured["kwargs"]["test_paths"] == [str(exp.TEST_PATH)]
    assert captured["kwargs"]["changed_modules"] == [str(exp.MODULE_PATH)]

    specs = exp.e2e_command_specs(tmp_path, tmp_path / "private")
    assert [row.name for row in specs] == list(exp.REQUIRED_E2E_NAMES)
    assert all("--no-cov" in row.argv for row in specs[:2])
    terminal = exp.terminal_command_specs(tmp_path, tmp_path / "candidate.json")
    assert [row.name for row in terminal] == list(exp.REQUIRED_TERMINAL_NAMES)

    exp.progress(0.0, "test", "boundary", completed_units=1)
    assert "phase=test event=boundary" in capsys.readouterr().out

    wrapper = REPO / exp.WRAPPER_PATH
    monkeypatch.setattr(exp, "main", lambda argv=None: 0)
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(wrapper), run_name="__main__")
    assert exc.value.code == 0


def test_req_7354_completed_live_checkpoint_is_authenticated(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7354 reuses only the exact completed four-episode live window."""

    raw = _raw_panel()
    schedule = raw["schedule"]
    episodes = [
        next(row for row in raw["episodes"] if row["episode_id"] == sealed["episode_id"])
        for sealed in schedule
    ]
    session = {
        "model_loaded": True,
        "timed_out": False,
        "error": None,
        "model_spec": exp.MODEL_SPECS[0],
        "runtime_receipt": {
            "task_linked_cuda_execution": True,
            "identity_authentication_valid": True,
        },
        "episodes": episodes,
    }
    checkpoint = tmp_path / "live_session.json"
    exp.atomic_write(checkpoint, session)
    assert exp.load_completed_live_checkpoint(checkpoint, schedule) == session

    for field, value in (
        ("model_loaded", False),
        ("timed_out", True),
        ("model_spec", {"hf_id": "wrong"}),
        ("runtime_receipt", {"task_linked_cuda_execution": False}),
        ("episodes", episodes[:-1]),
    ):
        changed = deepcopy(session)
        changed[field] = value
        exp.atomic_write(checkpoint, changed)
        assert exp.load_completed_live_checkpoint(checkpoint, schedule) is None

    changed = deepcopy(session)
    changed["episodes"][0]["seed"] += 1
    exp.atomic_write(checkpoint, changed)
    assert exp.load_completed_live_checkpoint(checkpoint, schedule) is None
    checkpoint.write_text("{", encoding="utf-8")
    assert exp.load_completed_live_checkpoint(checkpoint, schedule) is None


def test_req_7354_fail_closed_helpers_and_validator(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7354 rejects malformed trace bytes and contradictory terminal fields."""

    with pytest.raises(ValueError, match="unknown matched arm"):
        exp.session_environment(
            {}, arm="unknown", episode_dir=tmp_path / "unknown", gpu_index=0, port=9002
        )

    request_path = tmp_path / "request.json"
    request_path.write_text('{"messages": []}', encoding="utf-8")
    assert exp._request_bytes({"request_path": str(request_path)}) == b'{"messages": []}'
    assert exp._request_bytes({}) is None
    assert exp._request_bytes({"request_path": str(tmp_path / "missing")}) is None
    assert exp._request_occurrences(None, "x") == 0
    assert exp._request_occurrences(b"{", "x") == 0
    assert exp._evidence_receipts({"evidence_receipts": [{"enabled": False}, "bad"]}) == [
        {"enabled": False}
    ]

    malformed = _episode("r11l", "result_resume")
    malformed["evidence_receipt"]["result_rows"] = ["bad"]
    assert exp._causal_chain(malformed)["chain_complete"] is False
    malformed = _episode("r11l", "result_resume")
    malformed["evidence_receipt"]["result_rows"][0]["next_request_id"] = "bad"
    assert exp._causal_chain(malformed)["links"]["next_request"] is False

    bad_hash = _raw_panel()
    bad_hash["episodes"][0]["raw_request_manifest"][0]["request_sha256"] = "sha256:bad"
    assert exp.reduce_raw_panel(bad_hash)["arc_capture_complete_score"] == 0

    raw = _raw_panel()
    reduced = exp.reduce_raw_panel(raw)
    receipts = [
        _receipt(name)
        for name in (
            *exp.REQUIRED_VALIDATION_NAMES,
            *exp.REQUIRED_E2E_NAMES,
            *exp.REQUIRED_TERMINAL_NAMES,
        )
    ]
    artifact = exp.build_terminal_artifact(
        preconditions=[],
        source_hashes={},
        selection={},
        raw_panel=raw,
        reduction=reduced,
        mutations=exp.run_trace_mutations(raw),
        runtime_identity={"task_linked_cuda_execution": True},
        model_specs=exp.MODEL_SPECS,
        invocation_counts={
            "model_loads_attempted": 1,
            "generation_calls_attempted": 1,
        },
        validation_receipts=receipts,
        repository_health={},
        started_at_utc="2026-09-16T12:00:00+00:00",
        ended_at_utc="2026-09-16T12:01:00+00:00",
        duration_s=60.0,
        phase_spans=[],
    )
    changes = (
        ("schema", "bad", "identity_mismatch"),
        ("run_date", "bad", "run_date_mismatch"),
        ("execution_venue", "board", "execution_venue_mismatch"),
        ("verdict_class", "bad", "verdict_class_invalid"),
        ("honest_verdict", "bad", "honest_verdict_prefix_invalid"),
        ("model_invoked", False, "model_invoked_mismatch"),
        ("inference_substrate_class", "bad", "inference_substrate_class_mismatch"),
        ("duration_s", 1.0, "duration_floor_failed"),
        ("runtime_identity_receipt", {}, "live_gpu_receipt_missing"),
        ("field_principles", {}, "field_principles_incomplete"),
        ("production_default_changed", True, "forbidden_state_change"),
    )
    for field, value, expected in changes:
        changed = deepcopy(artifact)
        changed[field] = value
        assert expected in exp.validate_artifact(changed)

    load_only = deepcopy(artifact)
    load_only["invocation_counts"]["generation_calls_attempted"] = 0
    load_only["inference_substrate_class"] = "model_load_no_generation"
    load_only["duration_s"] = 3.0
    load_only["reproducibility_checksum"] = exp.artifact_checksum(load_only)
    assert exp.validate_artifact(load_only) == []

    blocked = exp.build_blocked_artifact(
        checks=[exp.gate_check("missing", "upstream", "artifact", True, False)],
        started_at_utc="2026-09-16T12:00:00+00:00",
        ended_at_utc="2026-09-16T12:00:01+00:00",
        duration_s=1.0,
    )
    blocked_changes = (
        ("model_invoked", True, "blocked_invocation_mismatch"),
        ("inference_substrate_class", "no_model_load", "blocked_substrate_mismatch"),
        ("gate_check_summary", {}, "blocked_gate_summary_missing"),
        ("arc_capture_complete_score", 1, "unsafe_scores_on_failed_artifact"),
    )
    for field, value, expected in blocked_changes:
        changed = deepcopy(blocked)
        changed[field] = value
        assert expected in exp.validate_artifact(changed)

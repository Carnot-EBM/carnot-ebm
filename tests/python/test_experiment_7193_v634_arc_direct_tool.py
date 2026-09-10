"""Tests for the Exp7193 direct selfparse ARC measurement.

Spec refs: REQ-ARC-WMTE-7193 and SCENARIO-ARC-WMTE-7193-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys

import pytest

from carnot import experiment_7193_v634_arc_direct_tool as exp


def _pass_check() -> dict[str, object]:
    return exp.gate_check("fixture", "fixture", "ready", True, True)


def _model() -> dict[str, object]:
    return {
        "name": "Qwen3.8-27B",
        "hf_id": exp.MODEL_ID,
        "quantization": exp.QUANTIZATION,
        "gpu": 1,
        "model_path": "/cache/snapshots/revision/Qwen3.8-27B-Q4_K_M.gguf",
        "revision": "revision",
        "content_hash": "sha256:" + "a" * 64,
    }


def _completion(index: int, stage: str = "environment") -> dict[str, object]:
    return {
        "completion_id": f"completion-{index}",
        "index": index,
        "stage": stage,
        "content_path": f"results/raw/experiment_7193/completions/{index}.txt",
        "content_sha256": "sha256:" + str(index) * 64,
        "content_chars": 20,
        "prompt_tokens": 120,
        "completion_tokens": 8,
        "stop_type": "eos",
        "error": None,
    }


def _run_row(*, calls: int = 2, terminated_by: str = "zero_mismatches") -> dict[str, object]:
    return {
        "game": "r11l",
        "levels": 1,
        "reached": 1,
        "actions": 137,
        "wall_s": 82.0,
        "actions_to_first_levelup": 130,
        "llm_reached": True,
        "completions_consumed": {"completions": 2, "chars_raw": 40},
        "policy_diagnostics": {
            "induction_attempts": [
                {
                    "reason": "stall",
                    "started_at": "2026-09-10T12:00:00+00:00",
                    "wall_s": 70.0,
                    "planned": False,
                    "skipped": "world_model_accuracy_below_threshold",
                    "tool_gap": {
                        "selfparse": True,
                        "terminated_by": terminated_by,
                        "tool_calls_total": calls,
                        "tool_calls_by_name": {"diff_grids": calls},
                        "tool_gap_events": [],
                        "tool_gap_events_dropped": 0,
                    },
                }
            ]
        },
    }


def _session(
    *,
    row: dict[str, object] | None = None,
    timed_out: bool = False,
    canary_ok: bool = True,
) -> dict[str, object]:
    return {
        "status": "timed_out" if timed_out else "complete",
        "terminal_receipt": True,
        "timed_out": timed_out,
        "canary": {
            "ok": canary_ok,
            "completion_id": "completion-0" if canary_ok else None,
            "elapsed_s": 11.0,
        },
        "run_row": row,
        "completions": [_completion(0, "canary"), _completion(1)] if canary_ok else [],
        "model_identity": _model(),
        "runner_receipt": {
            "model_count": 1,
            "replica_count": 1,
            "runner": "LocalGGUFProposer_llama.cpp",
            "dual_gpu_runner_used": False,
            "task_owned": True,
            "actual_context_slots": 1,
            "observed_n_ctx": 49152,
            "completion_budget": 4096,
        },
        "gpu_receipts": {
            "provenance_ok": True,
            "task_linked_cuda_execution": True,
            "samples": [{"owned_by_task": True, "used_memory_mb": 19000}],
        },
        "phase_spans": [{"phase": "environment_session", "duration_s": 82.0}],
        "error": "session_deadline_3600s" if timed_out else None,
    }


def test_req_7193_frozen_contract_and_required_fields() -> None:
    """REQ-ARC-WMTE-7193 freezes the model, game, seed, caps, and schema."""

    assert exp.MODEL_SPECS == [{"hf_id": exp.MODEL_ID, "quantization": "Q4_K_M"}]
    assert exp.GAME == "r11l"
    assert exp.RANDOM_SEED == 7_193_001
    assert exp.ACTION_BUDGET == 4000
    assert exp.SESSION_TIMEOUT_S == 3600
    assert exp.INDUCTION_TIMEOUT_S == 2400
    assert exp.N_CTX == 49152
    assert exp.COMPLETION_BUDGET == 4096
    assert exp.REQUIRED_ARTIFACT_FIELDS == frozenset(exp.FIELD_PRINCIPLES)
    assert "arc_eval_runner.py" not in {path.name for path in exp.REQUIRED_SOURCE_PATHS}


def test_scenario_preflight_quarantine_precedes_matching_field() -> None:
    """SCENARIO-ARC-WMTE-7193-PREFLIGHT rejects a quarantined matching value."""

    payload = {
        "honest_verdict": exp.EXPECTED_PRIOR_VERDICT,
        "flagged_adversarial": True,
    }
    check = exp.upstream_field_gate(
        payload,
        upstream=exp.PRIOR_ARTIFACT_PATH.as_posix(),
        field="honest_verdict",
        expected=exp.EXPECTED_PRIOR_VERDICT,
    )
    assert check["passed"] is False
    assert check["observed_value"] == {
        "value": exp.EXPECTED_PRIOR_VERDICT,
        "quarantined": True,
        "consumed": False,
    }
    summary = exp.gate_summary([check])
    assert summary["failed_check"] == "upstream_field_not_quarantined"
    assert summary["upstream"] == exp.PRIOR_ARTIFACT_PATH.as_posix()
    assert summary["field"] == "honest_verdict"


def test_scenario_preflight_known_failure_is_history_not_science() -> None:
    """REQ-ARC-WMTE-7193 records Exp7186 without promoting its failed value."""

    check = exp.upstream_field_gate(
        {"honest_verdict": exp.EXPECTED_PRIOR_VERDICT},
        upstream=exp.PRIOR_ARTIFACT_PATH.as_posix(),
        field="honest_verdict",
        expected=exp.EXPECTED_PRIOR_VERDICT,
    )
    assert check["passed"] is True
    assert check["observed_value"]["consumed"] is False
    assert check["evidence_role"] == "addressed_prior_failure_only"


def test_scenario_isolation_denies_policy_solution_channels() -> None:
    """SCENARIO-ARC-WMTE-7193-ISOLATION allows only public frames into policy."""

    receipt = exp.adapter_isolation_receipt()
    assert receipt["passed"] is True
    assert receipt["policy_class"] == "E3AgentPolicy"
    assert receipt["policy_constructor_solutions_argument"] == "not_present"
    assert receipt["policy_inputs"] == ["public_frames", "available_actions", "own_transitions"]
    assert set(receipt["policy_denied"]) == {
        "per_game_adapter",
        "game_source",
        "registry_contents",
        "solved_trajectories",
        "banked_solutions",
    }
    assert receipt["environment_executable_source_allowed"] is True


def test_scenario_runner_environment_is_single_stream_direct_selfparse(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7193-RUNNER fixes the direct live environment."""

    env = exp.session_environment(
        {}, model_path="/cache/model.gguf", gpu_index=1, port=8943, raw_dir=tmp_path
    )
    assert env["CARNOT_FORCE_LIVE"] == "1"
    assert env["CARNOT_ARC_INDUCE_TOOL_LOOP"] == "selfparse"
    assert "CARNOT_ARC_SUPERVISOR_TOOL_ARM" not in env
    assert env["CARNOT_ARC_INDUCE_N_CTX"] == "49152"
    assert env["CARNOT_ARC_INDUCE_MAX_TOKENS"] == "4096"
    assert env["CARNOT_ARC_INDUCE_TIMEOUT"] == "2400"
    assert env["CARNOT_ARC_LLAMA_SERVER_PARALLEL"] == "1"
    assert env["CARNOT_ARC_GENERATOR_REQUIRE_CUDA"] == "1"
    assert "CARNOT_ARC_FFN_CPU_LAYERS" not in env
    assert env["CARNOT_ARC_GGUF_PATH"] == "/cache/model.gguf"
    assert env["CARNOT_ARC_RANDOM_SEED"] == str(exp.RANDOM_SEED)
    assert env["CARNOT_ARC_GENERATOR_SEED"] == str(exp.RANDOM_SEED)
    assert env["CARNOT_ARC_E3_DIR"].startswith(str(tmp_path))


def test_shipped_path_trace_requires_run_game_and_gap_attachment(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7193 traces the real eval and attempt attachment before model work."""

    eval_path = tmp_path / "arc_leaderboard_eval.py"
    policy_path = tmp_path / "arc_competition_agent.py"
    eval_path.write_text("def run_game(game, policy):\n    return {}\n", encoding="utf-8")
    policy_path.write_text(
        'attempt["tool_gap"] = {"tool_gap_events": stats.get("tool_gap_events")}\n',
        encoding="utf-8",
    )
    receipt = exp.trace_shipped_path(eval_path, policy_path)
    assert receipt["passed"] is True
    policy_path.write_text("def unrelated():\n    pass\n", encoding="utf-8")
    failed = exp.trace_shipped_path(eval_path, policy_path)
    assert failed["passed"] is False
    assert failed["tool_gap_attachment"] is False


def test_tool_induction_projection_preserves_calls_gaps_and_ids() -> None:
    """SCENARIO-ARC-WMTE-7193-ENGAGEMENT keeps returned selfparse evidence."""

    completions = [_completion(0, "canary"), _completion(1), _completion(2)]
    rows = exp.project_tool_inductions(_run_row(calls=2), completions)
    assert len(rows) == 1
    assert rows[0]["induction_id"].startswith("sha256:")
    assert rows[0]["selfparse"] is True
    assert rows[0]["tool_calls_total"] == 2
    assert rows[0]["terminal_result_returned"] is True
    assert rows[0]["engaged"] is True
    assert rows[0]["tool_gap_events"] == []
    assert rows[0]["completion_ids"] == ["completion-1", "completion-2"]


@pytest.mark.parametrize("calls,terminated", [(0, "zero_mismatches"), (2, "")])
def test_tool_induction_projection_does_not_invent_engagement(calls: int, terminated: str) -> None:
    """REQ-ARC-WMTE-7193 requires calls and a terminal result for engagement."""

    rows = exp.project_tool_inductions(_run_row(calls=calls, terminated_by=terminated), [])
    assert rows[0]["engaged"] is False


def test_historical_summary_keeps_old_inductions_out_of_new_volume() -> None:
    """REQ-ARC-WMTE-7193 separates cited history from measured volume."""

    payload = {"per_game": [_run_row(calls=6), _run_row(calls=0)]}
    summary = exp.historical_induction_summary(payload, "sha256:" + "b" * 64)
    assert summary["historical_real_tool_loop_inductions"] == 1
    assert summary["historical_tool_calls"] == 6
    assert summary["counts_toward_new_volume"] is False


def test_scenario_terminal_null_after_zero_call_session() -> None:
    """SCENARIO-ARC-WMTE-7193-TERMINAL-NULL makes zero calls terminal evidence."""

    artifact = exp.build_terminal_artifact(
        run_date=exp.RUN_DATE,
        duration_s=75.0,
        checks=[_pass_check()],
        source_hashes={"source": "sha256:" + "c" * 64},
        session=_session(row=_run_row(calls=0)),
        historical={"historical_real_tool_loop_inductions": 2},
        isolation=exp.adapter_isolation_receipt(),
    )
    assert artifact["status"] == "complete"
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"] == "complete_null_direct_tool_session_no_returned_engagement"
    assert artifact["arc_tool_measurement_complete_score"] == 1
    assert artifact["arc_tool_engagement_score"] == 0
    assert artifact["arc_volume_sufficient_score"] == 0
    assert artifact["sample_size_budget"]["planned_sessions"] == 1
    assert artifact["sample_size_budget"]["completed_sessions"] == 1
    assert artifact["paired_efficacy_reported"] is False
    assert artifact["official_leaderboard_score_reported"] is False
    assert exp.validate_artifact(artifact) == []


def test_scenario_terminal_timeout_is_complete_not_partial() -> None:
    """SCENARIO-ARC-WMTE-7193-TERMINAL-NULL classifies the bounded deadline."""

    artifact = exp.build_terminal_artifact(
        run_date=exp.RUN_DATE,
        duration_s=3601.0,
        checks=[_pass_check()],
        source_hashes={},
        session=_session(row=None, timed_out=True),
        isolation=exp.adapter_isolation_receipt(),
    )
    assert artifact["status"] == "complete"
    assert artifact["verdict_class"] == "null"
    assert artifact["arc_tool_measurement_complete_score"] == 1
    assert artifact["rows"][0]["error"] == "session_deadline_3600s"
    assert artifact["rows"][0]["abstention"] is True
    assert artifact["inference_substrate_class"] == "model_bounded_generation"


def test_scenario_engagement_sets_score_without_efficacy_claim() -> None:
    """SCENARIO-ARC-WMTE-7193-ENGAGEMENT scores execution, not efficacy."""

    artifact = exp.build_terminal_artifact(
        run_date=exp.RUN_DATE,
        duration_s=95.0,
        checks=[_pass_check()],
        source_hashes={},
        session=_session(row=_run_row(calls=2)),
        historical={"historical_real_tool_loop_inductions": 2},
        isolation=exp.adapter_isolation_receipt(),
    )
    assert artifact["verdict_class"] == "positive"
    assert (
        artifact["honest_verdict"] == "complete_positive_direct_tool_engagement_no_efficacy_claim"
    )
    assert artifact["arc_tool_engagement_score"] == 1
    assert artifact["arc_volume_sufficient_score"] == 0
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["inference_substrate_class"] == "model_full_generation"
    assert artifact["inference_mode"] == "live_gpu"
    assert artifact["per_game_results"][0]["banked_levels"] == 1
    assert artifact["new_solve_claimed"] is False
    assert exp.validate_artifact(artifact) == []


def test_blocked_artifact_names_exact_first_failure() -> None:
    """SCENARIO-ARC-WMTE-7193-PREFLIGHT emits an actionable terminal block."""

    failed = exp.gate_check("cached_model", "host_cache", "model_path", "present", None)
    artifact = exp.build_terminal_artifact(
        run_date=exp.RUN_DATE,
        duration_s=0.2,
        checks=[failed],
        source_hashes={},
        isolation=exp.adapter_isolation_receipt(),
    )
    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_cached_model"
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["gate_check_summary"] == {
        "passed": False,
        "failed_check": "cached_model",
        "upstream": "host_cache",
        "field": "model_path",
        "expected_value": "present",
        "observed_value": None,
        "checks": [failed],
    }
    assert exp.validate_artifact(artifact) == []


def test_work_classification_tracks_actual_model_work() -> None:
    """REQ-ARC-WMTE-7193 applies floors to the computation that ran."""

    assert exp.classify_inference_work(None) == (
        "preflight_only_no_model_load",
        "blocked_no_run",
        "not_run",
    )
    load_only = _session(canary_ok=False)
    load_only["model_loaded"] = True
    assert exp.classify_inference_work(load_only)[1] == "model_load_no_generation"
    assert exp.classify_inference_work(_session(row=None))[1] == "model_bounded_generation"
    assert exp.classify_inference_work(_session(row=_run_row(calls=2)))[1] == (
        "model_full_generation"
    )


def test_validator_rejects_schema_checksum_floor_and_claim_mutations() -> None:
    """REQ-ARC-WMTE-7193 closes every terminal claim against its evidence."""

    base = exp.build_terminal_artifact(
        run_date=exp.RUN_DATE,
        duration_s=95.0,
        checks=[_pass_check()],
        source_hashes={},
        session=_session(row=_run_row(calls=2)),
        isolation=exp.adapter_isolation_receipt(),
    )
    assert "missing_fields:" in exp.validate_artifact({})[0]
    changed = deepcopy(base)
    changed.update(
        field_principles={},
        run_date="20260909",
        MODEL_SPECS=[],
        execution_venue="unknown",
        verdict_class="unknown",
        status="partial",
        solve_provenance="registry_replay",
        new_solve_claimed=True,
        official_leaderboard_score_reported=True,
        arc_tool_measurement_complete_score=0,
        arc_tool_engagement_score=0,
        arc_volume_sufficient_score=1,
        reproducibility_checksum="bad",
    )
    errors = set(exp.validate_artifact(changed))
    assert {
        "field_principles_mismatch",
        "run_date_mismatch",
        "model_specs_declaration_mismatch",
        "execution_venue_invalid",
        "verdict_class_invalid",
        "status_invalid",
        "solve_provenance_invalid",
        "new_solve_claim_forbidden",
        "official_score_claim_forbidden",
        "measurement_complete_score_inconsistent",
        "engagement_score_inconsistent",
        "volume_score_inconsistent",
        "reproducibility_checksum_invalid",
    } <= errors

    short = deepcopy(base)
    short["duration_s"] = 20.0
    short["reproducibility_checksum"] = exp.artifact_checksum(short)
    assert "duration_floor_not_met:model_full_generation" in exp.validate_artifact(short)

    checksum = deepcopy(base)
    checksum["duration_s"] = 96.0
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(checksum)


def test_atomic_write_and_path_validator_round_trip(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7193 validates the exact bytes selected for publication."""

    artifact = exp.build_terminal_artifact(
        run_date=exp.RUN_DATE,
        duration_s=0.1,
        checks=[exp.gate_check("missing", "fixture", "path", True, False)],
        source_hashes={},
        isolation=exp.adapter_isolation_receipt(),
    )
    path = tmp_path / "nested" / "artifact.json"
    exp.atomic_write(path, artifact)
    assert json.loads(path.read_text(encoding="utf-8")) == artifact
    assert exp.validate_artifact(path) == []


def test_parse_gpu_inventory_and_choose_conflict_free_card() -> None:
    """SCENARIO-ARC-WMTE-7193-RUNNER refuses cards with unowned compute."""

    inventory = exp.parse_gpu_inventory(
        "0, NVIDIA GeForce RTX 3090, GPU-a, 24100, 24576\n"
        "1, NVIDIA GeForce RTX 3090, GPU-b, 23000, 24576\n",
        "88, GPU-a, 100\n",
    )
    assert inventory[0]["compute_apps"] == [{"pid": 88, "used_memory_mb": 100}]
    assert inventory[1]["compute_apps"] == []
    assert exp.choose_gpu(inventory)["index"] == 1
    assert exp.choose_gpu(inventory, minimum_free_mb=24000) is None


def test_collect_static_preconditions_blocks_missing_bytes_and_skips_runner(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-7193-PREFLIGHT blocks before the live callback."""

    root = tmp_path / "repo"
    root.mkdir()
    result = tmp_path / "results" / "terminal.json"
    checkpoint = tmp_path / "results" / "checkpoints" / "running.json"
    raw_dir = tmp_path / "results" / "raw"
    called: list[bool] = []
    artifact = exp.run_experiment(
        root=root,
        run_date=exp.RUN_DATE,
        result_path=result,
        checkpoint_path=checkpoint,
        raw_dir=raw_dir,
        live_runner=lambda **_kwargs: called.append(True),
    )
    assert called == []
    assert artifact["status"] == "blocked"
    assert artifact["gate_check_summary"]["failed_check"] == "driving_capability_spec"
    assert result.is_file()
    assert checkpoint.is_file()


def test_parse_args_defaults_and_validate_failure(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7193 exposes the fixed execution date and cold validator."""

    args = exp.parse_args([])
    assert args.date == exp.RUN_DATE
    assert args.role == "driver"
    missing = tmp_path / "missing.json"
    assert exp.main(["--validate", str(missing)]) == 1


def test_file_hash_snapshot_and_unreadable_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-7193 binds readable bytes and records read failures explicitly."""

    source = tmp_path / "source.txt"
    source.write_text("bound bytes", encoding="utf-8")
    assert exp.sha256_file(source) == exp.sha256_bytes(b"bound bytes")
    monkeypatch.setattr(exp, "REQUIRED_SOURCE_PATHS", (Path("source.txt"),))
    sizes, hashes = exp._snapshot_sources(tmp_path)
    assert sizes == {"source.txt": 11}
    assert hashes == {"source.txt": exp.sha256_bytes(b"bound bytes")}
    monkeypatch.setattr(exp, "sha256_file", lambda _path: (_ for _ in ()).throw(OSError("gone")))
    sizes, hashes = exp._snapshot_sources(tmp_path)
    assert sizes == {"source.txt": 0}
    assert hashes == {"source.txt": "missing"}


def test_quarantine_fallback_handles_wrapped_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-ARC-WMTE-7193 fails closed when the shared quarantine helper cannot load."""

    monkeypatch.setitem(sys.modules, "conductor_gates", None)
    assert exp._quarantined({"flagged_adversarial": {"value": True}}) is True
    assert exp._quarantined({"flagged_adversarial": False}) is False


def test_quarantine_helper_adds_scripts_import_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-ARC-WMTE-7193 reaches the repository quarantine authority by its shipped path."""

    monkeypatch.delitem(sys.modules, "conductor_gates", raising=False)
    monkeypatch.setattr(sys, "path", [item for item in sys.path if item != str(exp.SCRIPTS_ROOT)])
    assert exp._quarantined({"flagged_adversarial": False}) is False


def test_gpu_parser_ignores_malformed_inventory_and_app_rows() -> None:
    """SCENARIO-ARC-WMTE-7193-RUNNER does not turn malformed GPU rows into capacity."""

    inventory = exp.parse_gpu_inventory(
        "short,row\n"
        "x, NVIDIA, GPU-bad, free, total\n"
        "2, NVIDIA GeForce RTX 3090, GPU-good, 24100, 24576\n",
        "missing,row\n9, GPU-unknown, 20\nbad, GPU-good, nope\n",
    )
    assert [row["index"] for row in inventory] == [2]
    assert inventory[0]["compute_apps"] == []


def test_projection_and_out_of_line_row_fail_closed(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7193 skips malformed attempts and unreadable raw rows."""

    malformed = {
        "policy_diagnostics": {
            "induction_attempts": [
                None,
                {"tool_gap": "wrong"},
                _run_row()["policy_diagnostics"]["induction_attempts"][0],
            ]
        }
    }
    assert len(exp.project_tool_inductions(malformed, [])) == 1
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    assert exp._session_run_row({"run_row_path": str(bad)}) is None
    good = tmp_path / "good.json"
    good.write_text(json.dumps(_run_row()), encoding="utf-8")
    assert exp._session_run_row({"run_row_path": str(good)})["game"] == "r11l"
    assert exp.classify_inference_work({"canary": {"ok": False}})[1] == "blocked_no_run"


def test_validator_catches_blocked_and_positive_internal_contradictions() -> None:
    """REQ-ARC-WMTE-7193 rejects terminal labels that contradict their evidence."""

    blocked = exp.build_terminal_artifact(
        run_date=exp.RUN_DATE,
        duration_s=0.1,
        checks=[exp.gate_check("missing", "fixture", "ready", True, False)],
        source_hashes={},
        isolation=exp.adapter_isolation_receipt(),
    )
    blocked["status"] = "complete"
    blocked["rows"] = [{"unit_id": "invented"}]
    blocked["inference_substrate_class"] = "model_full_generation"
    blocked["reproducibility_checksum"] = exp.artifact_checksum(blocked)
    errors = exp.validate_artifact(blocked)
    assert "blocked_terminal_inconsistent" in errors
    assert "blocked_substrate_inconsistent" in errors
    assert "duration_floor_not_met:model_full_generation" in errors

    positive = exp.build_terminal_artifact(
        run_date=exp.RUN_DATE,
        duration_s=75.0,
        checks=[_pass_check()],
        source_hashes={},
        session=_session(row=_run_row(calls=0)),
        isolation=exp.adapter_isolation_receipt(),
    )
    positive["verdict_class"] = "positive"
    positive["reproducibility_checksum"] = exp.artifact_checksum(positive)
    assert "positive_without_engagement" in exp.validate_artifact(positive)
    positive["gpu_receipts"] = {}
    positive["reproducibility_checksum"] = exp.artifact_checksum(positive)
    assert "live_gpu_without_task_linked_cuda_receipt" in exp.validate_artifact(positive)


def test_task_contract_history_json_and_relative_helpers(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7193 parses only the task contract and quarantines cited history."""

    contract = tmp_path / "contract.yaml"
    contract.write_text(
        f"- id: {exp.TASK_ID}\n"
        f"  milestone: {exp.MILESTONE}\n"
        "  deliverable: result.json\n"
        "  gated_on: null\n"
        "  prior_failures: []\n",
        encoding="utf-8",
    )
    assert exp._task_contract(contract)["milestone"] == exp.MILESTONE
    contract.write_text("[", encoding="utf-8")
    assert exp._task_contract(contract) == {}

    root = tmp_path / "repo"
    history = root / exp.HISTORICAL_RECEIPT_PATH
    history.parent.mkdir(parents=True)
    history.write_text(json.dumps({"flagged_adversarial": True}), encoding="utf-8")
    check, summary = exp._load_historical(root)
    assert check["passed"] is False
    assert summary == {}
    history.write_text("[]", encoding="utf-8")
    check, summary = exp._load_historical(root)
    assert check["passed"] is True
    assert summary["historical_real_tool_loop_inductions"] == 0

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    assert exp._read_json(invalid) == {}
    assert exp._read_json(tmp_path / "absent.json") == {}
    invalid.write_text("[]", encoding="utf-8")
    assert exp._read_json(invalid) == {}
    assert exp._relative(tmp_path).startswith("/")


def test_static_precondition_non_object_prior_is_not_promoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ARC-WMTE-7193-PREFLIGHT treats a non-object prior as absent evidence."""

    monkeypatch.setattr(exp, "REQUIRED_SOURCE_PATHS", ())
    prior = tmp_path / exp.PRIOR_ARTIFACT_PATH
    prior.parent.mkdir(parents=True)
    prior.write_text("[]", encoding="utf-8")
    result = tmp_path / "results" / "result.json"
    checkpoint = tmp_path / "results" / "checkpoints" / "run.json"
    raw = tmp_path / "results" / "raw"
    result.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    checks, _hashes = exp.collect_static_preconditions(
        root=tmp_path, result_path=result, checkpoint_path=checkpoint, raw_dir=raw
    )
    prior_check = next(row for row in checks if row["check"] == "upstream_field_not_quarantined")
    assert prior_check["observed_value"]["value"] is None


def test_run_experiment_live_callback_and_driver_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-7193 executes one injected session and publishes through the driver."""

    monkeypatch.setattr(
        exp, "collect_static_preconditions", lambda **_kwargs: ([_pass_check()], {})
    )
    monkeypatch.setattr(exp, "_load_historical", lambda _root: (_pass_check(), {}))
    ticks = iter((0.0, 95.0))
    monkeypatch.setattr(exp.time, "monotonic", lambda: next(ticks))
    artifact = exp.run_experiment(
        root=tmp_path,
        run_date=exp.RUN_DATE,
        result_path=tmp_path / "result.json",
        checkpoint_path=tmp_path / "checkpoints" / "run.json",
        raw_dir=tmp_path / "raw",
        live_runner=lambda **_kwargs: _session(row=_run_row(calls=2)),
    )
    assert artifact["status"] == "complete"

    monkeypatch.setattr(exp, "run_experiment", lambda **_kwargs: artifact)
    assert (
        exp.main(
            [
                "--result-path",
                str(tmp_path / "main-result.json"),
                "--checkpoint-path",
                str(tmp_path / "main-checkpoint.json"),
                "--raw-dir",
                str(tmp_path / "main-raw"),
            ]
        )
        == 0
    )


def test_run_experiment_refuses_invalid_terminal_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-7193 never writes a terminal artifact that fails cold validation."""

    monkeypatch.setattr(
        exp, "collect_static_preconditions", lambda **_kwargs: ([_pass_check()], {})
    )
    monkeypatch.setattr(exp, "_load_historical", lambda _root: (_pass_check(), {}))
    monkeypatch.setattr(exp, "validate_artifact", lambda _value: ["fixture_error"])
    with pytest.raises(ValueError, match="terminal_artifact_invalid:fixture_error"):
        exp.run_experiment(
            root=tmp_path,
            run_date=exp.RUN_DATE,
            result_path=tmp_path / "result.json",
            checkpoint_path=tmp_path / "checkpoints" / "run.json",
            raw_dir=tmp_path / "raw",
            live_runner=lambda **_kwargs: _session(row=_run_row()),
        )


def test_identity_failure_gate_names_the_contradicted_obligation() -> None:
    """SCENARIO-ARC-WMTE-7193-RUNNER preserves a typed identity external block."""

    session = {
        "model_identity_validation": {
            "valid": False,
            "errors": ["every identity obligation must be supported"],
        },
        "model_identity": {
            "requested_model_path": "/cache/model.gguf",
            "identity_obligation_rows": [
                {
                    "obligation": "unique_file_identity",
                    "status": "contradicted",
                    "evidence_source": "stable file descriptor device, inode, and link count",
                    "observed_value": {"requested_nlink": 2, "observed_nlink": 2},
                    "terminal": True,
                }
            ],
        },
    }
    check = exp.model_identity_gate(session)
    assert check == {
        "check": "typed_model_identity",
        "upstream": "/cache/model.gguf",
        "field": "unique_file_identity",
        "expected_value": {"status": "supported"},
        "observed_value": {
            "obligation": "unique_file_identity",
            "status": "contradicted",
            "evidence_source": "stable file descriptor device, inode, and link count",
            "observed_value": {"requested_nlink": 2, "observed_nlink": 2},
            "terminal": True,
        },
        "passed": False,
        "validation_errors": ["every identity obligation must be supported"],
    }
    assert exp.model_identity_gate({"model_identity_validation": {"valid": True}}) is None


def test_gpu_process_ownership_distinguishes_parent_from_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-ARC-WMTE-7193-RUNNER unloads only the owned generator group."""

    assert exp.gpu_process_ownership(101, process_group=303, parent_pid=101) == {
        "owned_by_task": True,
        "owned_generator_process": False,
    }
    monkeypatch.setattr(exp.os, "getpgid", lambda _pid: 303)
    assert exp.gpu_process_ownership(202, process_group=303, parent_pid=101) == {
        "owned_by_task": True,
        "owned_generator_process": True,
    }

    def missing(_pid: int) -> int:
        raise OSError("gone")

    monkeypatch.setattr(exp.os, "getpgid", missing)
    assert exp.gpu_process_ownership(404, process_group=303, parent_pid=101) == {
        "owned_by_task": False,
        "owned_generator_process": False,
    }


def test_scenario_unique_model_bytes_stages_a_multilink_cache_blob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ARC-WMTE-7193-UNIQUE-MODEL-BYTES keeps shared links unchanged."""

    revision = "a" * 40
    source_blob = tmp_path / "shared" / "blobs" / ("b" * 64)
    source_blob.parent.mkdir(parents=True)
    source_blob.write_bytes(b"required qwen bytes")
    second_link = tmp_path / "external-job" / "model.gguf"
    second_link.parent.mkdir()
    os.link(source_blob, second_link)
    source_alias = tmp_path / "shared" / "snapshots" / revision / "Qwen3.8-27B-Q4_K_M.gguf"
    source_alias.parent.mkdir(parents=True)
    source_alias.symlink_to(Path("../../blobs") / source_blob.name)
    expected_hash = exp.sha256_file(source_alias)

    def copy_on_write_fixture(source: Path, destination: Path) -> str:
        shutil.copyfile(source, destination)
        return "fixture_copy_on_write_clone"

    monkeypatch.setattr(exp, "_clone_copy_on_write", copy_on_write_fixture)
    selected = {
        "hf_id": exp.MODEL_ID,
        "model_path": str(source_alias),
        "revision": revision,
        "content_hash": expected_hash,
    }
    staged, receipt = exp.stage_unique_model_snapshot(selected, tmp_path / "raw")

    staged_path = Path(staged["model_path"])
    assert source_blob.stat().st_nlink == 2
    assert second_link.read_bytes() == b"required qwen bytes"
    assert staged_path.is_symlink()
    assert staged_path.resolve().stat().st_nlink == 1
    assert exp.sha256_file(staged_path) == expected_hash
    assert staged["source_cache_model_path"] == str(source_alias.absolute())
    assert receipt == {
        "required": True,
        "passed": True,
        "method": "fixture_copy_on_write_clone",
        "source_cache_model_path": str(source_alias.absolute()),
        "execution_model_path": str(staged_path),
        "source_nlink": 2,
        "execution_nlink": 1,
        "source_size": len(b"required qwen bytes"),
        "execution_size": len(b"required qwen bytes"),
        "source_content_hash": expected_hash,
        "execution_content_hash": expected_hash,
    }
    reused, reused_receipt = exp.stage_unique_model_snapshot(selected, tmp_path / "raw")
    assert reused["model_path"] == str(staged_path)
    assert reused_receipt["method"] == "existing_verified_task_owned_clone"


def test_scenario_unique_model_bytes_keeps_an_already_unique_snapshot(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7193 does not copy a cache blob that already has unique identity."""

    source = tmp_path / "snapshots" / ("c" * 40) / "Qwen3.8-27B-Q4_K_M.gguf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"unique qwen bytes")
    digest = exp.sha256_file(source)
    selected = {
        "hf_id": exp.MODEL_ID,
        "model_path": str(source),
        "revision": "c" * 40,
        "content_hash": digest,
    }
    staged, receipt = exp.stage_unique_model_snapshot(selected, tmp_path / "raw")

    assert staged["model_path"] == str(source.absolute())
    assert staged["source_cache_model_path"] == str(source.absolute())
    assert receipt["required"] is False
    assert receipt["passed"] is True
    assert receipt["method"] == "existing_unique_cache_blob"


def test_scenario_unique_model_bytes_rejects_a_wrong_declared_hash(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7193-UNIQUE-MODEL-BYTES blocks a byte mismatch."""

    source = tmp_path / "model.gguf"
    source.write_bytes(b"wrong bytes")
    with pytest.raises(ValueError, match="source_model_hash_mismatch"):
        exp.stage_unique_model_snapshot(
            {
                "hf_id": exp.MODEL_ID,
                "model_path": str(source),
                "revision": "d" * 40,
                "content_hash": "sha256:" + "0" * 64,
            },
            tmp_path / "raw",
        )


def test_scenario_unique_model_bytes_rejects_missing_source_and_revision(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7193 fails closed when a private snapshot cannot be identified."""

    with pytest.raises(ValueError, match="source_model_missing"):
        exp.stage_unique_model_snapshot(
            {"model_path": str(tmp_path / "missing.gguf")}, tmp_path / "raw"
        )
    source = tmp_path / "source.gguf"
    source.write_bytes(b"shared bytes")
    os.link(source, tmp_path / "second-link.gguf")
    with pytest.raises(ValueError, match="source_model_revision_missing"):
        exp.stage_unique_model_snapshot(
            {
                "model_path": str(source),
                "content_hash": exp.sha256_file(source),
                "revision": "",
            },
            tmp_path / "raw",
        )


def test_copy_on_write_clone_closes_files_and_removes_a_failed_clone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ARC-WMTE-7193-UNIQUE-MODEL-BYTES leaves no failed clone."""

    source = tmp_path / "source.gguf"
    source.write_bytes(b"clone bytes")
    destination = tmp_path / "destination.gguf"

    def emulate_clone(destination_fd: int, _request: int, source_fd: int) -> None:
        os.write(destination_fd, os.read(source_fd, len(b"clone bytes")))

    monkeypatch.setattr(exp.fcntl, "ioctl", emulate_clone)
    assert exp._clone_copy_on_write(source, destination) == "linux_ficlone_copy_on_write"
    assert destination.read_bytes() == b"clone bytes"

    failed = tmp_path / "failed.gguf"
    monkeypatch.setattr(
        exp.fcntl,
        "ioctl",
        lambda _destination, _request, _source: (_ for _ in ()).throw(OSError("no clone")),
    )
    with pytest.raises(OSError, match="no clone"):
        exp._clone_copy_on_write(source, failed)
    assert not failed.exists()


def test_scenario_unique_model_bytes_rejects_a_corrupt_clone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ARC-WMTE-7193-UNIQUE-MODEL-BYTES verifies cloned content."""

    source = tmp_path / "source.gguf"
    source.write_bytes(b"right")
    os.link(source, tmp_path / "second-link.gguf")

    def corrupt_clone(_source: Path, destination: Path) -> str:
        destination.write_bytes(b"wrong")
        return "fixture_corrupt_clone"

    monkeypatch.setattr(exp, "_clone_copy_on_write", corrupt_clone)
    with pytest.raises(ValueError, match="task_owned_model_clone_verification_failed"):
        exp.stage_unique_model_snapshot(
            {
                "model_path": str(source),
                "content_hash": exp.sha256_file(source),
                "revision": "e" * 40,
            },
            tmp_path / "raw",
        )

"""Tests for the independently bounded V635 ARC selfparse volume session.

Spec refs: REQ-ARC-WMTE-7206 and SCENARIO-ARC-WMTE-7206-*.
"""

from __future__ import annotations

import builtins
from copy import deepcopy
import itertools
import json
from pathlib import Path
import runpy
import sys
import types

import pytest

from carnot import experiment_7206_v635_arc_volume_a as exp


def _hash(char: str = "a") -> str:
    return "sha256:" + char * 64


def _pass_check() -> dict[str, object]:
    return exp.gate_check("fixture", "fixture", "ready", True, True)


def _completion(path: Path, index: int, text: str) -> dict[str, object]:
    path.write_text(text, encoding="utf-8")
    return {
        "completion_id": f"completion-{index}",
        "index": index,
        "stage": "environment",
        "induction_attempt_index": 0,
        "content_path": str(path),
        "content_sha256": exp.sha256_bytes(text.encode()),
        "completion_tokens": 8,
        "timings": {"prompt_ms": 20.0, "predicted_ms": 30.0},
        "error": None,
    }


def _tool(name: str, **arguments: object) -> str:
    params = "".join(
        f"<parameter={key}>\n{json.dumps(value)}\n</parameter>\n"
        for key, value in arguments.items()
    )
    return f"<tool_call>\n<function={name}>\n{params}</function>\n</tool_call>"


def _run_row(*, calls: int = 2, names: dict[str, int] | None = None) -> dict[str, object]:
    return {
        "game": "r11l",
        "levels": 2,
        "reached": 2,
        "deepest_level_reached": 2,
        "actions": 211,
        "wall_s": 91.0,
        "per_level": [{"level": 0, "completed": True}],
        "frame_sequence": [
            {"frame_index": 0, "levels_completed": 0},
            {"frame_index": 40, "levels_completed": 1},
            {"frame_index": 80, "levels_completed": 2},
        ],
        "policy_diagnostics": {
            "induction_attempts": [
                {
                    "reason": "stall",
                    "started_at": "2026-09-11T12:00:00+00:00",
                    "wall_s": 80.0,
                    "skipped": "world_model_accuracy_below_threshold",
                    "engine_functionally_identity": False,
                    "engine_identity_measurable": True,
                    "goal_predicate_satisfiable": False,
                    "tool_gap": {
                        "selfparse": True,
                        "terminated_by": "early_stop_non_improving",
                        "tool_calls_total": calls,
                        "tool_calls_by_name": names or {},
                        "tool_gap_events": [],
                        "tool_gap_events_dropped": 0,
                    },
                }
            ]
        },
    }


def _session(tmp_path: Path, *, calls: int = 2, row: dict[str, object] | None = None) -> dict:
    completions = [
        _completion(tmp_path / "c0.txt", 0, _tool("query_region", t=0)),
        _completion(tmp_path / "c1.txt", 1, _tool("diff_grids", t=0)),
    ]
    return {
        "status": "complete",
        "terminal_receipt": True,
        "timed_out": False,
        "model_loaded": True,
        "canary": {"ok": True},
        "run_row": row or _run_row(calls=calls),
        "completions": completions,
        "phase_spans": [{"phase": "model_load", "duration_s": 4.0}],
        "gpu_receipts": {
            "provenance_ok": True,
            "task_linked_cuda_execution": True,
            "samples": [{"free_memory_mb": 4010, "compute_apps": [{"owned_by_task": True}]}],
        },
        "model_identity": {
            "hf_id": exp.MODEL_ID,
            "revision": "revision",
            "model_file_hash": _hash(),
            "actual_runtime": "llama.cpp",
        },
        "model_identity_validation": {"valid": True, "errors": []},
        "runner_receipt": {
            "model_count": 1,
            "runner": "LocalGGUFProposer_llama.cpp",
            "dual_gpu_runner_used": False,
            "observed_n_ctx": 49152,
            "completion_budget": 4096,
        },
        "error": None,
    }


def test_req_7206_frozen_contract_and_base_configuration() -> None:
    """REQ-ARC-WMTE-7206 freezes one seed, model, context, and bounded session."""

    assert exp.RUN_DATE == "20260911"
    assert exp.GAME == "r11l"
    assert exp.RANDOM_SEED == 7_206_001
    assert exp.ACTION_BUDGET == 4000
    assert exp.SESSION_TIMEOUT_S == 3600
    assert exp.INDUCTION_TIMEOUT_S == 2400
    assert exp.N_CTX == 49152
    assert exp.COMPLETION_BUDGET == 4096
    assert exp.MODEL_SPECS == [{"hf_id": exp.MODEL_ID, "quantization": "Q4_K_M"}]
    configured = exp.configure_reused_driver()
    assert configured.RANDOM_SEED == exp.RANDOM_SEED
    assert configured.WRAPPER_PATH == exp.WRAPPER_PATH
    env = configured.session_environment(
        {}, model_path="/cache/model.gguf", gpu_index=1, port=9123, raw_dir=Path("/tmp/raw")
    )
    assert env["CARNOT_FORCE_LIVE"] == "1"
    assert env["CARNOT_ARC_INDUCE_TOOL_LOOP"] == "selfparse"
    assert env["CARNOT_ARC_INDUCE_N_CTX"] == "49152"
    assert env["CARNOT_ARC_INDUCE_MAX_TOKENS"] == "4096"
    assert "CARNOT_ARC_SUPERVISOR_TOOL_ARM" not in env


def test_scenario_preflight_unwraps_only_explicit_principle_value_wrapper() -> None:
    """SCENARIO-ARC-WMTE-7206-PREFLIGHT never unwraps an arbitrary dictionary."""

    wrapped = {"principle": "reason", "value": {"passed": True}}
    arbitrary = {"value": True, "evidence": "keep me"}
    assert exp.unwrap_evidence_value(wrapped) == {"passed": True}
    assert exp.unwrap_evidence_value(arbitrary) is arbitrary
    assert exp.unwrap_evidence_value({"value": True}) == {"value": True}


def test_scenario_preflight_quarantine_precedes_matching_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-ARC-WMTE-7206-PREFLIGHT rejects before consuming a matching value."""

    monkeypatch.setattr(exp, "is_quarantined", lambda payload: True)
    check = exp.authenticated_field_gate(
        {"arc_tool_measurement_complete_score": 1, "flagged_adversarial": True},
        upstream="results/upstream.json",
        field="arc_tool_measurement_complete_score",
        expected=1,
    )
    assert check["passed"] is False
    assert check["observed_value"] == {
        "value": "not_consumed",
        "quarantined": True,
        "consumed": False,
    }
    assert exp.gate_summary([check])["field"] == "arc_tool_measurement_complete_score"


def test_scenario_tool_event_reduction_repairs_exp7193_name_loss(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7206-TOOL-EVENT-REDUCTION uses one event suffix for both counts."""

    names = [
        "list_transitions",
        "diff_grids",
        "diff_grids",
        "run_engine_on_transitions",
        "run_engine_on_transitions",
        "run_engine_on_transitions",
    ]
    completions = [
        _completion(tmp_path / f"completion_{index}.txt", index, _tool(name, t=index))
        for index, name in enumerate(names)
    ]
    receipt = exp.terminal_tool_event_receipt(
        root=tmp_path, completions=completions, attempt_index=0, recorded_total=6
    )
    assert receipt["aggregation_consistent"] is True
    assert receipt["tool_calls_total"] == 6
    assert receipt["tool_calls_by_name"] == {
        "diff_grids": 2,
        "list_transitions": 1,
        "run_engine_on_transitions": 3,
    }
    assert [row["tool_name"] for row in receipt["tool_call_events"]] == names


def test_scenario_tool_event_mismatch_does_not_invent_names(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7206-TOOL-EVENT-REDUCTION fails closed on missing events."""

    completions = [_completion(tmp_path / "only.txt", 0, _tool("diff_grids", t=0))]
    receipt = exp.terminal_tool_event_receipt(
        root=tmp_path, completions=completions, attempt_index=0, recorded_total=6
    )
    assert receipt["aggregation_consistent"] is False
    assert receipt["tool_calls_total"] is None
    assert receipt["tool_calls_by_name"] == {}
    assert receipt["error"] == "recorded_total_exceeds_parsed_event_stream"


def test_tool_event_hash_mismatch_and_invalid_total_fail_closed(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7206 binds strict-parser events to completion bytes."""

    row = _completion(tmp_path / "bad.txt", 0, _tool("diff_grids", t=0))
    row["content_sha256"] = _hash("f")
    with pytest.raises(ValueError, match="completion_hash_mismatch"):
        exp.terminal_tool_event_receipt(
            root=tmp_path, completions=[row], attempt_index=0, recorded_total=1
        )
    invalid = exp.terminal_tool_event_receipt(
        root=tmp_path, completions=[], attempt_index=0, recorded_total=-1
    )
    assert invalid["error"] == "recorded_total_invalid"


def test_project_tool_inductions_preserves_terminal_and_model_validity(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7206 keeps names, gaps, terminal outcome, and validity errors."""

    session = _session(tmp_path)
    rows = exp.project_tool_inductions(tmp_path, session["run_row"], session["completions"])
    assert len(rows) == 1
    assert rows[0]["tool_calls_total"] == 2
    assert rows[0]["tool_calls_by_name"] == {"diff_grids": 1, "query_region": 1}
    assert rows[0]["terminal_outcome"] == "early_stop_non_improving"
    assert rows[0]["model_validity_errors"] == ["world_model_accuracy_below_threshold"]
    assert rows[0]["world_model_nondegeneracy"]["engine_functionally_identity"] is False
    assert rows[0]["raw_completion_hashes"] == [
        session["completions"][0]["content_sha256"],
        session["completions"][1]["content_sha256"],
    ]


def test_scenario_cumulative_unique_deduplicates_and_counts_sessions() -> None:
    """SCENARIO-ARC-WMTE-7206-CUMULATIVE-UNIQUE counts IDs once across sources."""

    old = {
        "induction_id": _hash("1"),
        "source_authenticated": True,
        "source_session_id": "historical",
        "seed": 11,
        "tool_calls_total": 2,
        "tool_gap_events": [],
        "terminal_result_returned": True,
        "engaged": True,
    }
    duplicate = {**old, "source_session_id": "exp7193"}
    fresh = {**old, "induction_id": _hash("2"), "source_session_id": "exp7206", "seed": 22}
    merged, summary = exp.merge_cumulative_rows(
        [("history", [old]), ("exp7193", [duplicate]), ("exp7206", [fresh])]
    )
    assert [row["induction_id"] for row in merged] == [_hash("1"), _hash("2")]
    assert merged[0]["duplicate_sources"] == ["exp7193"]
    assert summary == {
        "cumulative_unique_inductions": 2,
        "distinct_session_count": 2,
        "distinct_seed_count": 2,
        "observed_tool_calls": 4,
        "observed_gap_events": 0,
    }


def test_historical_projection_uses_source_seed_and_only_authentic_loops() -> None:
    """REQ-ARC-WMTE-7206 keeps the two historical inductions independent of Exp7193."""

    payload = {
        "random_seed": 1594772,
        "per_game": [
            {
                "game": "r11l",
                "policy_diagnostics": {
                    "induction_attempts": [
                        _run_row(calls=6)["policy_diagnostics"]["induction_attempts"][0],
                        {
                            **_run_row(calls=0)["policy_diagnostics"]["induction_attempts"][0],
                            "started_at": "later",
                        },
                    ]
                },
            }
        ],
    }
    rows = exp.historical_induction_rows(payload, source_hash=_hash("b"))
    assert len(rows) == 1
    assert rows[0]["seed"] == 1594772
    assert rows[0]["tool_calls_by_name"] == {}
    assert rows[0]["per_name_evidence_state"] == "not_recorded_do_not_invent"


def test_adapter_isolation_and_registry_receipts_use_observed_source(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7206 verifies the policy boundary and prior reproduced state."""

    eval_path = tmp_path / "eval.py"
    policy_path = tmp_path / "policy.py"
    registry = tmp_path / "registry.yaml"
    eval_path.write_text(
        "def run_game(game, policy):\n    return {}\n"
        "def _build_policy(kind, game):\n    return E3AgentPolicy(game)\n",
        encoding="utf-8",
    )
    policy_path.write_text(
        "class E3AgentPolicy:\n    def __init__(self, game_id, proposer=None): pass\n"
        "def make_carnot_agent(base_cls): return E3AgentPolicy\n",
        encoding="utf-8",
    )
    registry.write_text(
        "games:\n- game: r11l\n  reproducibility: reproduced\n  levels_reproduced: 6\n",
        encoding="utf-8",
    )
    isolation = exp.adapter_isolation_receipt(eval_path, policy_path)
    assert isolation["passed"] is True
    assert isolation["observed_policy_constructor_has_solutions"] is False
    assert isolation["policy_denied"] == [
        "per_game_adapter",
        "game_source",
        "registry_contents",
        "solved_trajectories",
        "historical_world_models",
        "banked_solutions",
    ]
    reproduced = exp.registry_reproduction_receipt(registry)
    assert reproduced["passed"] is True
    assert reproduced["levels_reproduced"] == 6


def test_build_complete_artifact_reports_cumulative_limits(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7206-NONCLAIM reports no observed demand at an exact denominator."""

    session = _session(tmp_path)
    current = exp.project_tool_inductions(tmp_path, session["run_row"], session["completions"])
    old = [{**current[0], "induction_id": _hash("9"), "source_session_id": "historical"}]
    cumulative, summary = exp.merge_cumulative_rows([("history", old), (exp.TASK_ID, current)])
    artifact = exp.build_terminal_artifact(
        duration_s=95.0,
        checks=[_pass_check()],
        source_hashes={"source": _hash("c")},
        session=session,
        current_rows=current,
        cumulative_rows=cumulative,
        cumulative_summary=summary,
        historical_count=1,
        optional_source_receipts=[{"source": "exp7207", "state": "absent_nonblocking"}],
        isolation={"passed": True},
        registry={"passed": True, "levels_reproduced": 6},
        parsing_duration_s=0.2,
    )
    assert artifact["status"] == "complete"
    assert artifact["verdict_class"] == "null"
    assert artifact["arc_session_complete_score"] == 1
    assert artifact["sample_size_budget"]["attempted_sessions"] == 1
    assert artifact["sample_size_budget"]["completed_sessions"] == 1
    assert artifact["sample_size_budget"]["censored_sessions"] == 0
    assert artifact["sample_size_budget"]["historical_tool_loop_inductions"] == 1
    assert artifact["sample_size_budget"]["new_tool_loop_inductions"] == 1
    assert artifact["sample_size_budget"]["cumulative_unique_inductions"] == 2
    assert artifact["sample_size_budget"]["evidence_target"] == 10
    assert artifact["demand_observation"]["finding"] == "no_observed_missing_tool_demand"
    assert artifact["demand_observation"]["observed_induction_denominator"] == 2
    assert artifact["demand_observation"]["statistical_absence_claimed"] is False
    assert artifact["per_game_results"][0]["registry_banked_levels_before_run"] == 6
    assert artifact["per_game_results"][0]["session_completed_levels"] == 2
    assert len(artifact["per_game_results"][0]["transient_level_transitions"]) == 2
    assert artifact["model_invoked"] is True
    assert artifact["new_solve_claimed"] is False
    assert artifact["paired_efficacy_reported"] is False
    assert artifact["submitted"] is False
    assert exp.validate_artifact(artifact) == []


def test_build_blocked_and_canary_only_artifacts_classify_actual_work(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7206-TERMINAL-NULL distinguishes block from bounded generation."""

    failed = exp.gate_check("cached_model", "host_cache", "model_path", "present", None)
    blocked = exp.build_terminal_artifact(
        duration_s=0.4,
        checks=[failed],
        source_hashes={},
        session=None,
        current_rows=[],
        cumulative_rows=[],
        cumulative_summary={},
        historical_count=0,
        optional_source_receipts=[],
        isolation={},
        registry={},
        parsing_duration_s=0.0,
    )
    assert blocked["status"] == "blocked"
    assert blocked["honest_verdict"] == "blocked_cached_model"
    assert blocked["inference_substrate_class"] == "blocked_no_run"
    assert blocked["arc_session_complete_score"] == 0
    assert blocked["gate_check_summary"]["observed_value"] is None
    assert exp.validate_artifact(blocked) == []

    session = _session(tmp_path)
    session["run_row"] = None
    session["completions"] = [session["completions"][0]]
    canary = exp.build_terminal_artifact(
        duration_s=12.0,
        checks=[_pass_check()],
        source_hashes={},
        session=session,
        current_rows=[],
        cumulative_rows=[],
        cumulative_summary={},
        historical_count=0,
        optional_source_receipts=[],
        isolation={"passed": True},
        registry={"passed": True, "levels_reproduced": 6},
        parsing_duration_s=0.0,
    )
    assert canary["status"] == "complete"
    assert canary["verdict_class"] == "null"
    assert canary["inference_substrate_class"] == "model_bounded_generation"
    assert canary["arc_session_complete_score"] == 1
    assert exp.validate_artifact(canary) == []


def test_validator_rejects_claim_checksum_floor_and_inconsistent_counts(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7206 validates evidence-linked closed fields."""

    session = _session(tmp_path)
    rows = exp.project_tool_inductions(tmp_path, session["run_row"], session["completions"])
    merged, summary = exp.merge_cumulative_rows([(exp.TASK_ID, rows)])
    artifact = exp.build_terminal_artifact(
        duration_s=95.0,
        checks=[_pass_check()],
        source_hashes={},
        session=session,
        current_rows=rows,
        cumulative_rows=merged,
        cumulative_summary=summary,
        historical_count=0,
        optional_source_receipts=[],
        isolation={"passed": True},
        registry={"passed": True, "levels_reproduced": 6},
        parsing_duration_s=0.1,
    )
    bad = deepcopy(artifact)
    bad["new_solve_claimed"] = True
    bad["sample_size_budget"]["cumulative_unique_inductions"] = 99
    bad["duration_s"] = 1.0
    assert set(exp.validate_artifact(bad)) >= {
        "new_solve_claim_forbidden",
        "cumulative_count_inconsistent",
        "duration_floor_not_met:model_full_generation",
        "reproducibility_checksum_mismatch",
    }


def test_atomic_write_parse_args_and_validation_round_trip(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7206 writes only a stable terminal JSON artifact."""

    path = tmp_path / "artifact.json"
    payload = {"hello": "world"}
    exp.atomic_write(path, payload)
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    args = exp.parse_args(["--date", exp.RUN_DATE, "--validate", str(path)])
    assert args.date == exp.RUN_DATE
    assert args.validate == path
    assert exp.validate_artifact(path) != []


def test_source_snapshot_records_missing_without_raising(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7206-PREFLIGHT records source-byte failures as gates."""

    present = tmp_path / "present.txt"
    present.write_text("bytes", encoding="utf-8")
    sizes, hashes = exp.snapshot_sources(tmp_path, [Path("present.txt"), Path("missing.txt")])
    assert sizes == {"present.txt": 5, "missing.txt": 0}
    assert hashes["present.txt"].startswith("sha256:")
    assert hashes["missing.txt"] == "missing"


def test_static_preflight_and_real_historical_join_use_repository_evidence(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7206-PREFLIGHT authenticates the actual frozen inputs."""

    result = tmp_path / "results" / "terminal.json"
    checkpoint = tmp_path / "results" / "checkpoints" / "running.json"
    raw = tmp_path / "results" / "raw"
    result.parent.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    raw.mkdir(parents=True)
    checks, hashes, context = exp.collect_static_preconditions(
        root=exp.REPO_ROOT,
        result_path=result,
        checkpoint_path=checkpoint,
        raw_dir=raw,
    )
    assert all(row["passed"] is True for row in checks)
    assert all(exp.HASH_RE.fullmatch(value) for value in hashes.values())
    groups, historical_count = exp._historical_groups(exp.REPO_ROOT, context)
    assert [len(rows) for _, rows in groups] == [2, 1]
    assert historical_count == 3
    assert groups[1][1][0]["tool_calls_total"] == 6
    assert groups[1][1][0]["tool_calls_by_name"] == {
        "diff_grids": 2,
        "list_transitions": 1,
        "run_engine_on_transitions": 3,
    }


def test_preflight_helpers_cover_success_rejection_and_read_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-7206 fails closed for malformed or unauthenticated evidence."""

    assert (
        exp.authenticated_field_gate(
            {"complete": {"principle": "observed", "value": 1}},
            upstream="fixture.json",
            field="complete",
            expected=1,
        )["passed"]
        is True
    )

    real_import = builtins.__import__

    def unavailable_conductor(name: str, *args: object, **kwargs: object) -> object:
        if name == "conductor_gates":
            raise ImportError("fixture")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable_conductor)
    monkeypatch.delitem(sys.modules, "conductor_gates", raising=False)
    assert exp.is_quarantined({"flagged_adversarial": True}) is True
    monkeypatch.undo()

    broken_json = tmp_path / "broken.json"
    broken_json.write_text("{", encoding="utf-8")
    assert exp._load_json(broken_json)[1].startswith("JSONDecodeError:")
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert exp._load_json(list_json) == ({}, "not_json_object")
    assert exp._load_json(tmp_path / "absent.json")[1].startswith("FileNotFoundError:")

    roadmap = tmp_path / "roadmap.yaml"
    roadmap.write_text(
        "- id: exp7206-arc-volume-a\n  milestone: 2026.09.635\n",
        encoding="utf-8",
    )
    assert exp._task_contract(roadmap)["id"] == exp.TASK_ID
    roadmap.write_text("tasks: []\n", encoding="utf-8")
    assert exp._task_contract(roadmap)["id"] is None
    assert exp._task_contract(tmp_path / "missing.yaml") == {}

    monkeypatch.setattr(
        sys, "path", [entry for entry in sys.path if entry != str(exp.SCRIPTS_ROOT)]
    )
    fake_gates = types.ModuleType("conductor_gates")
    fake_gates._is_quarantined = lambda _payload: False  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "conductor_gates", fake_gates)
    assert exp.is_quarantined({}) is False
    assert str(exp.SCRIPTS_ROOT) in sys.path


def test_receipt_projection_skips_malformed_rows_and_rejects_bad_merge_ids(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7206-CUMULATIVE-UNIQUE excludes non-evidence rows."""

    attempt = _run_row()["policy_diagnostics"]["induction_attempts"][0]
    attempt["refinement_rounds"] = [None, {"skipped": "refine_invalid"}]
    attempt["counterexamples"] = [None, {"kind": "heldout_mismatch"}]
    run_row = _run_row()
    run_row["policy_diagnostics"]["induction_attempts"] = [None, {}, attempt]
    completions = [
        _completion(tmp_path / "p0.txt", 0, _tool("query_region", t=0)),
        _completion(tmp_path / "p1.txt", 1, _tool("diff_grids", t=0)),
    ]
    for row in completions:
        row["induction_attempt_index"] = 2
    rows = exp.project_tool_inductions(tmp_path, run_row, completions)
    assert len(rows) == 1
    assert rows[0]["model_validity_errors"] == [
        "world_model_accuracy_below_threshold",
        "refine_invalid",
        "heldout_mismatch",
    ]

    historical = {
        "random_seed": 9,
        "per_game": [
            None,
            {"game": "other"},
            {
                "game": exp.GAME,
                "policy_diagnostics": {
                    "induction_attempts": [None, {}, {"tool_gap": {"tool_calls_total": 0}}]
                },
            },
        ],
    }
    assert exp.historical_induction_rows(historical, source_hash=_hash()) == []
    merged, summary = exp.merge_cumulative_rows(
        [
            (
                "bad",
                [
                    {"source_authenticated": False, "engaged": True, "induction_id": _hash()},
                    {"source_authenticated": True, "engaged": True, "induction_id": "bad"},
                ],
            )
        ]
    )
    assert merged == []
    assert summary["cumulative_unique_inductions"] == 0


def test_source_registry_isolation_snapshot_and_optional_sibling_error_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ARC-WMTE-7206-PREFLIGHT makes every failed observation explicit."""

    assert (
        exp.adapter_isolation_receipt(tmp_path / "missing.py", tmp_path / "missing2.py")["passed"]
        is False
    )
    invalid_registry = tmp_path / "registry.yaml"
    invalid_registry.write_text("games: [", encoding="utf-8")
    assert exp.registry_reproduction_receipt(invalid_registry)["passed"] is False

    boom = tmp_path / "boom"
    boom.write_text("x", encoding="utf-8")
    original_stat = Path.stat

    def failing_stat(path: Path, *args: object, **kwargs: object) -> object:
        if path == boom:
            raise OSError("fixture")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", failing_stat)
    sizes, hashes = exp.snapshot_sources(tmp_path, [Path("boom")])
    assert sizes == {"boom": 0}
    assert hashes == {"boom": "missing"}
    monkeypatch.undo()

    sibling = tmp_path / exp.SIBLING_PATH
    sibling.parent.mkdir(parents=True)
    sibling.write_text(
        json.dumps({"arc_session_complete_score": 1, "tool_induction_rows": [{"x": 1}]}),
        encoding="utf-8",
    )
    source_hashes: dict[str, object] = {}
    receipt, rows = exp._optional_sibling(tmp_path, source_hashes)
    assert receipt["state"] == "authenticated"
    assert rows == [{"x": 1}]
    sibling.write_text("{", encoding="utf-8")
    rejected, rows = exp._optional_sibling(tmp_path, source_hashes)
    assert rejected["state"] == "rejected_nonblocking"
    assert rows == []

    transitions = exp._transient_level_transitions(
        {"frame_sequence": [None, {}, {"levels_completed": 0}, {"levels_completed": 1}]}
    )
    assert transitions == [
        {"from_level": 0, "to_level": 1, "frame_index": None, "action_count": None}
    ]


def test_static_preflight_records_required_upstream_parse_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ARC-WMTE-7206-PREFLIGHT does not promote known unreadable upstreams."""

    result = tmp_path / "result.json"
    checkpoint = tmp_path / "checkpoints" / "running.json"
    raw = tmp_path / "raw"
    result.parent.mkdir(exist_ok=True)
    checkpoint.parent.mkdir()
    raw.mkdir()
    real_load = exp._load_json

    def fail_two(path: Path) -> tuple[dict[str, object], str | None]:
        if path in {
            exp.REPO_ROOT / exp.PRIOR_ARTIFACT_PATH,
            exp.REPO_ROOT / exp.EXP7193_PATH,
        }:
            return {}, "fixture_unreadable"
        return real_load(path)

    monkeypatch.setattr(exp, "_load_json", fail_two)
    checks, _, _ = exp.collect_static_preconditions(
        root=exp.REPO_ROOT,
        result_path=result,
        checkpoint_path=checkpoint,
        raw_dir=raw,
    )
    failures = [row for row in checks if row["passed"] is not True]
    assert any(row["upstream"] == exp.PRIOR_ARTIFACT_PATH.as_posix() for row in failures)
    assert any(row["upstream"] == exp.EXP7193_PATH.as_posix() for row in failures)


def test_validator_covers_closed_schema_failure_modes(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7206 rejects inconsistent terminal fields and evidence claims."""

    session = _session(tmp_path)
    rows = exp.project_tool_inductions(tmp_path, session["run_row"], session["completions"])
    cumulative, summary = exp.merge_cumulative_rows([(exp.TASK_ID, rows)])
    artifact = exp.build_terminal_artifact(
        duration_s=95.0,
        checks=[_pass_check()],
        source_hashes={},
        session=session,
        current_rows=rows,
        cumulative_rows=cumulative,
        cumulative_summary=summary,
        historical_count=0,
        optional_source_receipts=[],
        isolation={"passed": True},
        registry={"passed": True, "levels_reproduced": 6},
        parsing_duration_s=0.1,
    )
    bad = deepcopy(artifact)
    bad["field_principles"]["status"] = "wrong"
    bad["status"] = "blocked"
    bad["arc_session_complete_score"] = 0
    bad["honest_verdict"] = "wrong"
    bad["rows"].append({"unit_id": "incomplete"})
    bad["cumulative_induction_rows"].append(deepcopy(bad["cumulative_induction_rows"][0]))
    bad["inference_mode"] = "live_gpu"
    bad["gpu_receipts"] = {}
    bad["reproducibility_checksum"] = exp.artifact_checksum(bad)
    errors = exp.validate_artifact(bad)
    assert {
        "field_principles_mismatch",
        "complete_terminal_inconsistent",
        "complete_verdict_prefix_missing",
        "comparison_row_contract_invalid",
        "cumulative_induction_ids_not_unique",
        "cumulative_count_inconsistent",
        "demand_denominator_inconsistent",
        "live_gpu_without_task_linked_cuda_receipt",
    } <= set(errors)

    blocked = exp.build_terminal_artifact(
        duration_s=0.1,
        checks=[exp.gate_check("x", "y", "z", 1, 0)],
        source_hashes={},
        session=None,
        current_rows=[],
        cumulative_rows=[],
        cumulative_summary={},
        historical_count=0,
        optional_source_receipts=[],
        isolation={},
        registry={},
        parsing_duration_s=0.0,
    )
    blocked["status"] = "complete"
    blocked["arc_session_complete_score"] = 1
    blocked["honest_verdict"] = "wrong"
    blocked["reproducibility_checksum"] = exp.artifact_checksum(blocked)
    assert {"blocked_terminal_inconsistent", "blocked_verdict_prefix_missing"} <= set(
        exp.validate_artifact(blocked)
    )
    assert exp.validate_artifact(tmp_path / "unreadable.json") == ["artifact_unreadable"]


def test_run_experiment_injected_session_and_wrong_date_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ARC-WMTE-7206-TERMINAL-NULL writes complete or diagnosed blocked output."""

    source_context = {
        "optional_source_receipts": [],
        "sibling_rows": [],
        "isolation": {"passed": True},
        "registry": {"passed": True, "levels_reproduced": 6},
    }
    monkeypatch.setattr(
        exp,
        "collect_static_preconditions",
        lambda **_kwargs: ([_pass_check()], {"fixture": _hash()}, source_context),
    )
    monkeypatch.setattr(exp, "_historical_groups", lambda *_args: ([], 0))
    ticks = itertools.count(0.0, 20.0)
    monkeypatch.setattr(exp.time, "monotonic", lambda: next(ticks))
    session = _session(tmp_path)
    result = tmp_path / "complete.json"
    artifact = exp.run_experiment(
        root=tmp_path,
        run_date=exp.RUN_DATE,
        result_path=result,
        checkpoint_path=tmp_path / "checkpoints" / "running.json",
        raw_dir=tmp_path / "raw",
        live_runner=lambda **_kwargs: (session, [_pass_check()]),
    )
    assert artifact["status"] == "complete"
    assert exp.validate_artifact(result) == []

    ticks = itertools.count(0.0, 20.0)
    monkeypatch.setattr(exp.time, "monotonic", lambda: next(ticks))
    blocked_path = tmp_path / "blocked.json"
    blocked = exp.run_experiment(
        root=tmp_path,
        run_date="19990101",
        result_path=blocked_path,
        checkpoint_path=tmp_path / "checkpoints2" / "running.json",
        raw_dir=tmp_path / "raw2",
        live_runner=lambda **_kwargs: session,
    )
    assert blocked["status"] == "blocked"
    assert blocked["gate_check_summary"]["failed_check"] == "execution_date"

    ticks = itertools.count(0.0, 20.0)
    monkeypatch.setattr(exp.time, "monotonic", lambda: next(ticks))
    direct = exp.run_experiment(
        root=tmp_path,
        run_date=exp.RUN_DATE,
        result_path=tmp_path / "direct.json",
        checkpoint_path=tmp_path / "checkpoints3" / "running.json",
        raw_dir=tmp_path / "raw3",
        live_runner=lambda **_kwargs: session,
    )
    assert direct["status"] == "complete"

    ticks = itertools.count(0.0, 20.0)
    monkeypatch.setattr(exp.time, "monotonic", lambda: next(ticks))
    absent = exp.run_experiment(
        root=tmp_path,
        run_date=exp.RUN_DATE,
        result_path=tmp_path / "absent.json",
        checkpoint_path=tmp_path / "checkpoints4" / "running.json",
        raw_dir=tmp_path / "raw4",
        live_runner=lambda **_kwargs: None,
    )
    assert absent["status"] == "blocked"
    assert absent["gate_check_summary"]["failed_check"] == "live_session_receipt"

    ticks = itertools.count(0.0, 20.0)
    monkeypatch.setattr(exp.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(exp, "validate_artifact", lambda _artifact: ["fixture_invalid"])
    with pytest.raises(ValueError, match="terminal_artifact_invalid:fixture_invalid"):
        exp.run_experiment(
            root=tmp_path,
            run_date=exp.RUN_DATE,
            result_path=tmp_path / "invalid.json",
            checkpoint_path=tmp_path / "checkpoints5" / "running.json",
            raw_dir=tmp_path / "raw5",
            live_runner=lambda **_kwargs: session,
        )


def test_run_experiment_reused_runtime_branches_and_cli_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-7206 keeps the real reused runtime and all CLI modes reachable."""

    context = {
        "optional_source_receipts": [],
        "sibling_rows": [],
        "isolation": {"passed": True},
        "registry": {"passed": True, "levels_reproduced": 6},
    }
    monkeypatch.setattr(
        exp,
        "collect_static_preconditions",
        lambda **_kwargs: ([_pass_check()], {}, context),
    )
    monkeypatch.setattr(exp, "_historical_groups", lambda *_args: ([], 0))
    monkeypatch.setattr(
        exp.reused,
        "_live_resource_preconditions",
        lambda _root: ([_pass_check()], {"path": "model"}, {"index": 0}, "llama-server"),
    )
    session = _session(tmp_path)
    monkeypatch.setattr(exp.reused, "_run_live_session", lambda **_kwargs: (session, []))
    ticks = itertools.count(0.0, 20.0)
    monkeypatch.setattr(exp.time, "monotonic", lambda: next(ticks))
    artifact = exp.run_experiment(
        root=tmp_path,
        run_date=exp.RUN_DATE,
        result_path=tmp_path / "live.json",
        checkpoint_path=tmp_path / "cp" / "running.json",
        raw_dir=tmp_path / "raw-live",
    )
    assert artifact["status"] == "complete"

    monkeypatch.setattr(exp.reused, "run_session_child", lambda _args: 7)
    assert exp.run_session_child(exp.parse_args(["--role", "session"])) == 7
    assert exp.main(["--role", "session"]) == 7

    terminal = tmp_path / "terminal.json"
    exp.atomic_write(terminal, artifact)
    assert exp.main(["--validate", str(terminal)]) == 0
    assert exp.main(["--validate", str(tmp_path / "missing.json")]) == 1

    monkeypatch.setenv("CARNOT_REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(
        exp,
        "run_experiment",
        lambda **_kwargs: {"status": "complete", "honest_verdict": "complete_fixture"},
    )
    assert (
        exp.main(
            [
                "--result-path",
                "relative.json",
                "--checkpoint-path",
                str(tmp_path / "absolute-checkpoint.json"),
                "--raw-dir",
                str(tmp_path / "absolute-raw"),
            ]
        )
        == 0
    )


def test_module_dunder_main_validation_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-7206 keeps the module itself executable for cold validation."""

    monkeypatch.setattr(
        sys,
        "argv",
        [str(exp.REPO_ROOT / exp.MODULE_PATH), "--validate", str(tmp_path / "absent.json")],
    )
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(exp.REPO_ROOT / exp.MODULE_PATH), run_name="__main__")
    assert raised.value.code == 1

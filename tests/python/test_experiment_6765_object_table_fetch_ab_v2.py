"""Tests for the exclusive-load object-table A/B rerun.

Spec refs: REQ-ARC-WMTE-6765 and SCENARIO-ARC-WMTE-6765-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from carnot import experiment_6753_object_table_fetch_on_demand_ab as exp6753
from carnot import experiment_6764_arc_exclusive_load_preflight as exp6764
from carnot import experiment_6765_object_table_fetch_ab_v2 as exp


def _model(tmp_path: Path, index: int) -> dict:
    spec = exp6764.MODEL_SPECS[index]
    path = tmp_path / spec["filename"]
    path.write_bytes(f"model-{index}".encode())
    return {
        **spec,
        "resolved": True,
        "model_path": str(path),
        "model_size_bytes": path.stat().st_size,
        "model_sha256": spec["expected_sha256"],
        "tokenizer": {"source": "llama.cpp_embedded_gguf", "loadable": True},
    }


def _lifecycle(row_id: str) -> dict:
    return {
        "session_id": row_id,
        "device": {
            "index": 1,
            "uuid": "GPU-eligible",
            "name": "NVIDIA GeForce RTX 3090",
        },
        "lease_owner": {"task_id": f"exp6765-{row_id}", "device_uuid": "GPU-eligible"},
        "phase_history": [
            {"phase": phase, "monotonic_ns": index + 1}
            for index, phase in enumerate(exp.COMPLETE_PHASE_SEQUENCE)
        ],
        "lease_release": {"released": True, "phase": "terminal_complete"},
        "owned_cleanup": {
            "pid": 123,
            "absent_after_exit": True,
            "unrelated_processes_signaled": [],
        },
        "vram_recovery": {
            "before_used_mb": 4,
            "after_used_mb": 4,
            "absolute_delta_mb": 0,
            "tolerance_mb": 512,
            "owned_pid_present": False,
            "passed": True,
        },
        "peak_owned_vram_mb": 18_000,
        "runtime_context": exp.CONTEXT_REQUESTED,
        "gpu_layers": {"requested": 999, "offloaded": 66, "total": 66},
        "unrelated_processes_signaled": [],
        "errors": [],
    }


def _row(tmp_path: Path, planned: dict, *, treatment_delta: float = 0.02) -> dict:
    science = planned["row_kind"] == "science"
    arm = planned["arm"]
    baseline = arm == exp.BASELINE_ARM
    model = _model(tmp_path, 0 if science else 1)
    prompt = f"prompt:{planned['game']}:{planned['seed']}:{arm}"
    events = []
    if not baseline:
        events = [
            {
                "turn": 0,
                "raw_emission": "<tool_call>find_objects</tool_call>",
                "parsed_tool": "find_objects",
                "parsed_arguments": {
                    "t": 0,
                    "which": "before",
                    "predicate_code": "def accept(obj):\n    return True",
                    "max_objects": 8,
                },
                "dispatch_result": {"ok": True, "objects": [], "response_bytes": 64},
                "bounded_response": "<tool_response>{}</tool_response>",
            },
            {
                "turn": 1,
                "raw_emission": "<tool_call>run_engine_on_transitions</tool_call>",
                "parsed_tool": "run_engine_on_transitions",
                "parsed_arguments": {"code": "def engine(grid, action, data): return grid"},
                "dispatch_result": {"ok": True, "change_fidelity": 0.5},
                "bounded_response": "<tool_response>{}</tool_response>",
            },
        ]
    accounting = exp6753.fetch_accounting(events)
    change_fidelity = None if not science else 0.5 + (treatment_delta if not baseline else 0)
    row = {
        **planned,
        "model_role": model["role"],
        "model_path": model["model_path"],
        "model_sha256": model["model_sha256"],
        "production_route": exp.PRODUCTION_ROUTE,
        "context_requested": exp.CONTEXT_REQUESTED,
        "context_observed_by_model": exp.CONTEXT_REQUESTED,
        "gpu_receipt": {
            "assigned_device": {
                "physical_index": 1,
                "uuid": "GPU-eligible",
                "name": "NVIDIA GeForce RTX 3090",
            },
            "gpu_layers": {"requested": 999, "offloaded": 66, "total": 66},
            "peak_vram_mb": 18_000,
        },
        "live_model_invoked": True,
        "prompt": prompt,
        "raw_prompt_sha256": exp.sha256_text(prompt),
        "prompt_tokens": 1_000 if baseline else 600,
        "inline_object_table": "OBJECT TABLE" if baseline else "",
        "prompt_isolation_receipt": {"only_object_table_removed": True},
        "public_observations": [{"grid": [[0]], "next_grid": [[1]], "action": 1}],
        "tool_events": events,
        "tool_requests": [
            {
                "turn": event["turn"],
                "tool": event["parsed_tool"],
                "arguments": deepcopy(event["parsed_arguments"]),
            }
            for event in events
        ],
        "bounded_responses": [event["bounded_response"] for event in events],
        **accounting,
        "tool_loop_stats": {
            "tool_calls_total": len(events),
            "tool_calls_by_name": {event["parsed_tool"]: 1 for event in events},
            "terminated_by": "zero_mismatches",
        },
        "transition_result": (
            None
            if not science
            else {
                "n_heldout": 2,
                "true_changed_cells": 10,
                "correct_changed_cells": 8,
                "spurious_changed_cells": 1,
            }
        ),
        "transition_receipt": {
            "shown_count": 1,
            "held_count": 2 if science else 0,
            "held_public_observations": [],
        },
        "change_fidelity": change_fidelity,
        "transition_utility": None if not science else (0.7 + (0.1 if not baseline else 0)),
        "verifier_metrics": None if not science else {"accuracy": 0.5},
        "engine_sha256": None if not science else "sha256:engine",
        "duration_s": 2.0,
        "actions": [],
        "stop_reason": "zero_mismatches",
        "failure_class": None,
        "session_receipt": _lifecycle(planned["row_id"]),
        "source_receipt": {
            "game_source_read": False,
            "offline_bfs_used": False,
            "per_game_query_injected": False,
        },
        "solve_claim": False,
    }
    row["row_sha256"] = exp.row_checksum(row)
    return row


def _models(tmp_path: Path) -> list[dict]:
    return [_model(tmp_path, 0), _model(tmp_path, 1)]


def _passing_preflight(tmp_path: Path) -> dict:
    models = _models(tmp_path)
    return {
        "all_passed": True,
        "checks": [
            {
                "check": "arc_exclusive_load_ready",
                "expected": True,
                "observed": True,
                "passed": True,
            }
        ],
        "models": models,
        "device_selection_receipt": {
            "selected_device": {
                "index": 1,
                "uuid": "GPU-eligible",
                "name": "NVIDIA GeForce RTX 3090",
                "memory_used_mb": 4,
                "memory_free_mb": 24_120,
            }
        },
        "source_receipts": {
            "source_access": {
                "game_source_read": False,
                "offline_bfs_used": False,
                "solve_trace_used": False,
            }
        },
    }


def _all_rows(tmp_path: Path) -> list[dict]:
    return [_row(tmp_path, planned) for planned in exp.row_plan()]


def test_scenario_arc_wmte_6765_frozen_pairing_matches_exp6753() -> None:
    """SCENARIO-ARC-WMTE-6765-FROZEN-PAIRING keeps every frozen choice."""
    manifest = exp.frozen_manifest()
    prior = exp6753.frozen_design()
    for field in (
        "game_ids",
        "seeds",
        "arms",
        "arm_order_rule",
        "context_requested",
        "science_budgets",
        "sidecar_budgets",
        "generator_configuration",
        "noninferiority_margin",
    ):
        assert manifest[field] == prior[field]
    rows = exp.row_plan()
    assert len(rows) == 122
    assert [row["row_id"] for row in rows[:120]] == [
        row["row_id"] for row in exp6753.science_plan()
    ]
    assert {row["row_kind"] for row in rows[-2:]} == {"transport_sidecar"}
    assert all(row["quality_pool"] == "excluded_canary" for row in rows[-2:])


def test_scenario_arc_wmte_6765_production_fetch_environment_isolated(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-6765-PRODUCTION-FETCH changes only table presence."""
    model = _model(tmp_path, 0)
    selected = {"index": 1, "uuid": "GPU-eligible"}
    baseline_plan, treatment_plan = exp.row_plan()[:2]
    baseline = exp.worker_environment({}, model, selected, baseline_plan, port=4567)
    treatment = exp.worker_environment({}, model, selected, treatment_plan, port=4567)
    differences = {key for key in baseline | treatment if baseline.get(key) != treatment.get(key)}
    assert differences == {"CARNOT_ARC_OBJECT_PERCEPTION"}
    assert baseline["CARNOT_ARC_OBJECT_PERCEPTION"] == "1"
    assert treatment["CARNOT_ARC_OBJECT_PERCEPTION"] == "0"
    assert baseline["CARNOT_ARC_INDUCE_TOOL_LOOP"] == "selfparse"
    assert baseline["CARNOT_ARC_INDUCE_N_CTX"] == "32768"
    assert baseline["CARNOT_ARC_GENERATOR_CUDA_GPU"] == "1"
    assert baseline["CARNOT_ARC_EXCLUSIVE_PORT"] == "4567"


def test_req_arc_wmte_6765_preflight_extends_exclusive_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-6765 checks Exp6764, the frozen design, route, and no-solve boundary."""
    prior_6764 = {
        "arc_exclusive_load_ready": True,
        "reproducibility_checksum": "sha256:ready",
    }
    prior_6753 = {"frozen_design": exp6753.frozen_design()}
    registry = tmp_path / "registry.yaml"
    registry.write_text("reproducible_total_games: 25\n")
    architecture = tmp_path / "architecture.md"
    architecture.write_text("**Last Reconciled:** 2026-08-26\n")
    ready_path = tmp_path / "6764.json"
    ready_path.write_text(json.dumps(prior_6764))
    prior_path = tmp_path / "6753.json"
    prior_path.write_text(json.dumps(prior_6753))
    monkeypatch.setattr(exp, "EXP6764_PATH", ready_path)
    monkeypatch.setattr(exp, "EXP6753_PATH", prior_path)
    monkeypatch.setattr(exp, "REGISTRY_PATH", registry)
    monkeypatch.setattr(exp, "ARCHITECTURE_PATH", architecture)
    monkeypatch.setattr(exp6764, "validate_artifact", lambda artifact: [])
    base = _passing_preflight(tmp_path)
    result = exp.collect_preconditions(
        date="20260830", base_preflight_fn=lambda root: deepcopy(base), root=tmp_path
    )
    assert result["all_passed"] is True
    by_name = {row["check"]: row for row in result["checks"]}
    assert by_name["arc_exclusive_load_ready"]["passed"] is True
    assert by_name["frozen_exp6753_manifest"]["passed"] is True
    assert by_name["production_e3_selfparse_route"]["passed"] is True
    assert by_name["registry_no_new_or_duplicate_solve_target"]["passed"] is True
    assert by_name["architecture_map_fresh"]["observed"] == "2026-08-26"


def test_req_arc_wmte_6765_accepts_only_the_exp6764_model_alias_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-WMTE-6765 preserves a valid pre-alias Exp6764 readiness receipt."""
    ready = {
        "models_used": [{"model_id": "immutable"}],
        "field_principles": {},
        "reproducibility_checksum": "sha256:original",
    }

    def validate(value):
        if "model_specs" not in value:
            return ["missing_field:model_specs", "model_specs"]
        return []

    monkeypatch.setattr(exp6764, "validate_artifact", validate)
    monkeypatch.setattr(exp6764, "artifact_checksum", lambda value: "sha256:normalized")
    errors, normalization = exp._validate_exp6764_receipt(ready)
    assert errors == []
    assert normalization == "models_used_to_model_specs"
    assert ready["reproducibility_checksum"] == "sha256:original"


def test_scenario_arc_wmte_6765_blocked_keeps_denominator(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-6765-BLOCKED writes 122 rows and starts no worker."""
    models = _models(tmp_path)
    preflight = {
        "all_passed": False,
        "checks": [
            {
                "check": "arc_exclusive_load_ready",
                "expected": True,
                "observed": False,
                "passed": False,
            }
        ],
        "models": models,
        "source_receipts": {
            "source_access": {
                "game_source_read": False,
                "offline_bfs_used": False,
                "solve_trace_used": False,
            }
        },
    }
    calls: list[str] = []
    artifact = exp.run(
        result_path=tmp_path / "blocked.json",
        date="20260830",
        preflight_fn=lambda **kwargs: preflight,
        worker_runner=lambda *args, **kwargs: calls.append("called"),
        clock=iter((1, 2)).__next__,
    )
    assert calls == []
    assert len(artifact["rows"]) == 122
    assert artifact["honest_verdict"] == "complete_blocked_object_table_ab_v2"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["object_table_ab_completed"] is False
    assert artifact["adoption_gate_passed"] is False
    assert artifact["solve_claim"] is False
    assert artifact["gate_check_summary"][0]["observed"] is False
    assert all(
        row["failure_class"].endswith("arc_exclusive_load_ready") for row in artifact["rows"]
    )
    assert exp.validate_artifact(artifact) == []


def test_scenario_arc_wmte_6765_reducers_exclude_canary_and_derive_rows(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-6765-REDUCTION-AND-CLAIM-BOUNDARY derives the fixed metrics."""
    rows = _all_rows(tmp_path)
    reduction = exp.reduce_rows(rows)
    assert reduction["object_table_ab_completed"] is True
    assert reduction["prompt_tokens_by_arm"][exp.BASELINE_ARM]["mean"] == 1_000
    assert reduction["prompt_tokens_by_arm"][exp.TREATMENT_ARM]["mean"] == 600
    assert reduction["tool_calls_by_arm"][exp.TREATMENT_ARM]["by_name"] == {
        "find_objects": 60,
        "run_engine_on_transitions": 60,
    }
    assert reduction["useful_fetch_rate"] == 1.0
    assert reduction["mean_prompt_token_savings"] == 400.0
    assert reduction["change_fidelity_by_arm"][exp.BASELINE_ARM] == pytest.approx(0.5)
    assert reduction["change_fidelity_by_arm"][exp.TREATMENT_ARM] == pytest.approx(0.52)
    assert reduction["change_fidelity_delta"] == pytest.approx(0.02)
    assert reduction["change_fidelity_interval"][0] == pytest.approx(0.02)
    assert reduction["transition_utility_delta"] == pytest.approx(0.1)
    assert reduction["within_arm_variance"][exp.BASELINE_ARM] == pytest.approx(0.0)
    assert reduction["adoption_gate_passed"] is True
    assert reduction["solve_claim"] is False


def test_req_arc_wmte_6765_row_validation_requires_attributable_lifecycle(
    tmp_path: Path,
) -> None:
    """REQ-ARC-WMTE-6765 rejects source use, missing prompts, leases, and teardown."""
    row = _row(tmp_path, exp.row_plan()[0])
    assert exp.row_evidence_errors(row) == []
    mutations = {
        "prompt": "",
        "public_observations": [],
        "solve_claim": True,
        "failure_class": "failed",
    }
    for field, value in mutations.items():
        changed = deepcopy(row)
        changed[field] = value
        changed["row_sha256"] = exp.row_checksum(changed)
        assert exp.row_evidence_errors(changed)
    changed = deepcopy(row)
    changed["session_receipt"]["lease_release"]["released"] = False
    changed["row_sha256"] = exp.row_checksum(changed)
    assert "lease_release" in exp.row_evidence_errors(changed)
    changed = deepcopy(row)
    changed["session_receipt"]["vram_recovery"]["passed"] = False
    changed["row_sha256"] = exp.row_checksum(changed)
    assert "vram_recovery" in exp.row_evidence_errors(changed)
    changed = deepcopy(row)
    changed["source_receipt"]["game_source_read"] = True
    changed["row_sha256"] = exp.row_checksum(changed)
    assert "source_boundary" in exp.row_evidence_errors(changed)


def test_scenario_arc_wmte_6765_session_order_stops_after_owned_failure(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-6765-LEASED-SESSION runs one row per fresh worker."""
    preflight = _passing_preflight(tmp_path)
    calls: list[str] = []

    def worker(model, selected, planned, runtime_dir, **kwargs):
        calls.append(planned["row_id"])
        row = _row(tmp_path, planned)
        if len(calls) == 2:
            row["failure_class"] = "session_lifecycle_failed:lease_busy"
            row["row_sha256"] = exp.row_checksum(row)
        return row

    artifact = exp.run(
        result_path=tmp_path / "partial.json",
        date="20260830",
        preflight_fn=lambda **kwargs: preflight,
        worker_runner=worker,
        clock=iter((1, 2)).__next__,
    )
    assert calls == [row["row_id"] for row in exp.row_plan()[:2]]
    assert len(artifact["rows"]) == 122
    assert artifact["verdict_class"] == "partial"
    assert artifact["object_table_ab_completed"] is False
    assert artifact["rows"][2]["failure_class"].startswith("not_run_after_session_failure")


def test_scenario_arc_wmte_6765_interruption_resume_reuses_only_complete_row(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-6765-INTERRUPTION-RESUME keeps valid owned evidence."""
    preflight = _passing_preflight(tmp_path)
    selected = preflight["device_selection_receipt"]["selected_device"]
    planned = exp.row_plan()[0]
    model = preflight["models"][0]
    runtime_dir = tmp_path / ".experiment_6765_object_table_fetch_ab_v2"
    row_path = runtime_dir / exp._session_slug(planned["row_id"]) / "row.json"
    row_path.parent.mkdir(parents=True)
    assert exp._load_completed_checkpoint(runtime_dir, planned, model, selected) is None
    checkpoint = _row(tmp_path, planned)
    checkpoint["prompt_isolation_receipt"] = {}
    checkpoint["row_sha256"] = exp.row_checksum(checkpoint)
    row_path.write_text(json.dumps(checkpoint))

    assert exp._load_completed_checkpoint(runtime_dir, planned, model, selected) == checkpoint
    mutations = (
        ("row_sha256", "sha256:damaged"),
        ("game", "changed"),
        ("model_sha256", "sha256:substituted"),
        ("prompt", ""),
    )
    for field, value in mutations:
        changed = deepcopy(checkpoint)
        changed[field] = value
        if field != "row_sha256":
            changed["row_sha256"] = exp.row_checksum(changed)
        row_path.write_text(json.dumps(changed))
        assert exp._load_completed_checkpoint(runtime_dir, planned, model, selected) is None
    changed = deepcopy(checkpoint)
    changed["gpu_receipt"]["assigned_device"]["uuid"] = "GPU-other"
    changed["row_sha256"] = exp.row_checksum(changed)
    row_path.write_text(json.dumps(changed))
    assert exp._load_completed_checkpoint(runtime_dir, planned, model, selected) is None
    row_path.write_text(json.dumps(checkpoint))

    calls: list[str] = []

    def worker(model, selected, planned, runtime_dir, **kwargs):
        calls.append(planned["row_id"])
        raise RuntimeError("stop after proving resume")

    artifact = exp.run(
        result_path=tmp_path / "partial.json",
        date="20260830",
        preflight_fn=lambda **kwargs: preflight,
        worker_runner=worker,
        clock=iter((1, 2)).__next__,
    )
    assert calls == [exp.row_plan()[1]["row_id"]]
    assert artifact["rows"][0]["row_id"] == planned["row_id"]
    assert artifact["rows"][0]["session_receipt"]["lease_release"]["released"] is True
    assert artifact["object_table_ab_completed"] is False


def test_scenario_arc_wmte_6765_prompt_pair_receipt_is_exact(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6765-PRODUCTION-FETCH detects any prompt drift."""
    plans = exp.row_plan()[:2]
    rows = [_row(tmp_path, plan) for plan in plans]
    table = "OBJECT TABLE"
    schemas = "SCHEMAS"
    rows[0]["prompt"] = f"HEAD{table}TAIL{schemas}"
    rows[0]["inline_object_table"] = table
    rows[1]["prompt"] = f"HEADTAIL{schemas}"
    rows[1]["inline_object_table"] = ""
    exp.attach_prompt_pair_receipts(rows, tool_schemas=schemas)
    assert all(row["prompt_isolation_receipt"]["only_object_table_removed"] for row in rows)
    rows[1]["prompt"] += "attack"
    exp.attach_prompt_pair_receipts(rows, tool_schemas=schemas)
    assert all(row["failure_class"] == "prompt_arm_isolation_failed" for row in rows)


def test_req_arc_wmte_6765_build_and_validate_complete_artifact(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-6765 validates required fields, reductions, and no-solve receipts."""
    preflight = _passing_preflight(tmp_path)
    rows = _all_rows(tmp_path)
    artifact = exp.build_artifact(
        date="20260830",
        rows=rows,
        preflight=preflight,
        started_ns=1,
        finished_ns=2,
    )
    assert artifact["object_table_ab_completed"] is True
    assert artifact["adoption_gate_passed"] is True
    assert artifact["honest_verdict"] == "complete_object_table_ab_v2_adopt_fetch_on_demand"
    assert artifact["verdict_class"] == "positive"
    assert len(artifact["gpu_receipts"]) == 122
    assert len(artifact["lease_receipts"]) == 122
    assert set(artifact) <= set(artifact["field_principles"])
    assert exp.validate_artifact(artifact) == []
    changed = deepcopy(artifact)
    changed["solve_claim"] = True
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert "solve_claim" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["mean_prompt_token_savings"] = -99
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert "reduction:mean_prompt_token_savings" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["extra"] = True
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert "field_principles" in exp.validate_artifact(changed)


def test_req_arc_wmte_6765_live_session_receipts_and_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-6765 owns the lease and teardown around one production row."""
    model = _model(tmp_path, 0)
    selected = {
        "index": 1,
        "uuid": "GPU-eligible",
        "name": "NVIDIA GeForce RTX 3090",
        "memory_used_mb": 4,
        "memory_free_mb": 24_120,
    }
    planned = exp.row_plan()[0]
    phases: list[str] = []

    class Lease:
        document = {"phase": "preflight"}
        journal_path = tmp_path / "journal.json"

        def owner_receipt(self):
            return {"task_id": planned["row_id"], "device_uuid": "GPU-eligible"}

        def transition(self, phase, **kwargs):
            phases.append(phase)
            self.document["phase"] = phase

        def release(self):
            return {"released": True, "phase": self.document["phase"]}

        def close(self):
            return None

    process = SimpleNamespace(pid=123, poll=lambda: None)

    class Proposer:
        port = 4567
        n_gpu_layers = 999
        n_ctx = exp.CONTEXT_REQUESTED

        def __init__(self, **kwargs):
            self._proc = process

        def _ensure_server(self):
            return True

        def observed_n_ctx(self):
            return exp.CONTEXT_REQUESTED

        def observed_model_path(self):
            return model["model_path"]

    from carnot.agentic import arc_executable_world_model as world

    monkeypatch.setattr(world, "LocalGGUFProposer", Proposer)
    monkeypatch.setattr(exp, "acquire_selected_lease", lambda **kwargs: Lease())
    monkeypatch.setattr(
        exp,
        "_gpu_snapshot",
        lambda *args: {
            **selected,
            "owned_pid_present": bool(args and args[-1] == 123),
            "owned_pid_vram_mb": 18_000,
        },
    )
    monkeypatch.setattr(exp, "_eligible_device_still_selected", lambda selected: True)
    monkeypatch.setattr(
        exp,
        "_process_identity",
        lambda process: {
            "pid": 123,
            "pid_start_ticks": 1,
            "parent_pid": 1,
            "executable": "llama-server",
            "exit_code": None,
            "absent_after_exit": False,
        },
    )
    monkeypatch.setattr(
        exp,
        "_read_gpu_layers",
        lambda proposer: {
            "requested": 999,
            "offloaded": 66,
            "total": 66,
        },
    )
    monkeypatch.setattr(
        exp,
        "_build_window",
        lambda game: {
            "shown": [SimpleNamespace(grid=[[0]], next_grid=[[1]], action=1, data={})],
            "held": [],
            "cell": 1,
        },
    )
    base_row = _row(tmp_path, planned)
    monkeypatch.setattr(exp6753, "_run_live_row", lambda *args: (deepcopy(base_row), "PROMPT"))
    monkeypatch.setattr(
        exp,
        "terminate_owned_process",
        lambda process: {
            "pid": 123,
            "absent_after_exit": True,
            "unrelated_processes_signaled": [],
        },
    )
    monkeypatch.setattr(
        exp,
        "_wait_for_vram_recovery",
        lambda *args: (
            {"passed": True, "owned_pid_present": False, "before_used_mb": 4, "after_used_mb": 4},
            {"memory_used_mb": 4},
        ),
    )
    monkeypatch.setattr(
        exp,
        "read_journal",
        lambda path: {
            "phase_history": [
                {"phase": phase, "monotonic_ns": index + 1}
                for index, phase in enumerate(exp.COMPLETE_PHASE_SEQUENCE)
            ]
        },
    )
    monkeypatch.setattr(exp, "_object_table_for_window", lambda window: "TABLE")
    monkeypatch.setenv("CARNOT_ARC_E3_DIR", str(tmp_path / "e3"))
    row = exp.run_live_row_session(
        model,
        selected,
        planned,
        port=4567,
        lease_runtime_dir=tmp_path / "leases",
    )
    assert phases == [
        "admitted",
        "loading",
        "resident",
        "inferencing",
        "unloading",
        "validating",
        "terminal_complete",
    ]
    assert row["session_receipt"]["lease_release"]["released"] is True
    assert row["session_receipt"]["vram_recovery"]["passed"] is True
    assert row["session_receipt"]["owned_cleanup"]["absent_after_exit"] is True
    assert row["prompt"] == "PROMPT"
    assert row["source_receipt"]["game_source_read"] is False


def test_req_arc_wmte_6765_worker_subprocess_and_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-ARC-WMTE-6765 keeps one-row worker and repository CLI paths explicit."""
    model = _model(tmp_path, 0)
    selected = _passing_preflight(tmp_path)["device_selection_receipt"]["selected_device"]
    planned = exp.row_plan()[0]

    def popen(command, **kwargs):
        output = Path(command[command.index("--worker-output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(_row(tmp_path, planned)))

        class Process:
            pid = 321
            returncode = 0

            def communicate(self, timeout=None):
                return "", ""

            def poll(self):
                return 0

        return Process()

    monkeypatch.setattr(exp.subprocess, "Popen", popen)
    monkeypatch.setattr(exp, "proc_start_ticks", lambda pid: 1)
    row = exp.run_row_worker_subprocess(
        model,
        selected,
        planned,
        tmp_path / "runtime",
        port=4567,
        lease_runtime_dir=tmp_path / "leases",
    )
    assert row["row_id"] == planned["row_id"]

    job = tmp_path / "job.json"
    output = tmp_path / "worker.json"
    job.write_text(json.dumps({"model": model, "selected_device": selected, "planned": planned}))
    monkeypatch.setattr(
        exp, "run_live_row_session", lambda *args, **kwargs: _row(tmp_path, planned)
    )
    assert exp._worker_entry(job, output, 4567, tmp_path / "leases") == 0
    assert json.loads(output.read_text())["row_id"] == planned["row_id"]

    monkeypatch.setattr(
        exp,
        "run",
        lambda **kwargs: {
            "object_table_ab_completed": False,
            "adoption_gate_passed": False,
            "honest_verdict": "complete_blocked_object_table_ab_v2",
        },
    )
    assert exp.main(["--date", "20260830"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["verdict"] == "complete_blocked_object_table_ab_v2"


def test_req_arc_wmte_6765_defensive_branches(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-6765 fails closed on duplicates and malformed terminal evidence."""
    row = _row(tmp_path, exp.row_plan()[0])
    duplicate = exp.reduce_rows([row, deepcopy(row)])
    assert duplicate["object_table_ab_completed"] is False
    assert duplicate["row_completion_receipt"]["duplicate_row_ids"] == [row["row_id"]]
    assert exp._load_json(tmp_path / "missing.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("[]")
    assert exp._load_json(bad) == {}
    assert exp._first_failed_check({"checks": []}) == "unknown_precondition"
    canary = _row(tmp_path, exp.row_plan()[-1])
    canary["quality_pool"] = "science"
    assert "canary_quality_pool" in exp.row_evidence_errors(canary)


def test_req_arc_wmte_6765_preflight_and_public_serialization_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-6765 records unavailable route/docs and JSON-safe public data."""
    missing = tmp_path / "missing"
    monkeypatch.setattr(exp, "REPO_ROOT", missing)
    monkeypatch.setattr(exp, "EXP6764_PATH", missing / "6764.json")
    monkeypatch.setattr(exp, "EXP6753_PATH", missing / "6753.json")
    monkeypatch.setattr(exp, "REGISTRY_PATH", missing / "registry.yaml")
    monkeypatch.setattr(exp, "ARCHITECTURE_PATH", missing / "architecture.md")
    monkeypatch.setattr(exp6753, "sha256_file", lambda path: "sha256:missing")
    result = exp.collect_preconditions(
        date="invalid", base_preflight_fn=lambda root: {"checks": []}, root=missing
    )
    checks = {item["check"]: item for item in result["checks"]}
    assert result["all_passed"] is False
    assert checks["production_e3_selfparse_route"]["observed"] == "unavailable"
    assert checks["registry_no_new_or_duplicate_solve_target"]["passed"] is False
    assert checks["architecture_map_fresh"]["age_days"] is None

    class IntegerLike:
        def __int__(self):
            return 7

    class StringOnly:
        def __int__(self):
            raise TypeError

        def __str__(self):
            return "string-only"

    transition = SimpleNamespace(
        grid=np.asarray([[1]], dtype=np.int64),
        next_grid=np.asarray([[2]], dtype=np.int64),
        action=np.int64(3),
        data={1: (IntegerLike(), StringOnly())},
        level_before=None,
        level_after=True,
    )
    serialized = exp._serialize_transition(transition)
    assert serialized["grid"] == [[1]]
    assert serialized["action"] == 3
    assert serialized["data"] == {"1": [7, "string-only"]}

    monkeypatch.setattr(exp6753, "_build_windows", lambda games: {games[0]: {"shown": []}})
    assert exp._build_window("ls20") == {"shown": []}
    from carnot.agentic import arc_executable_world_model as world

    monkeypatch.setattr(world, "objects_block", lambda shown: "ROWS")
    assert exp._object_table_for_window({"shown": []}).endswith("ROWS")

    log = tmp_path / "server.log"
    log.write_text("offloaded 4/4 layers to GPU\n")
    proposer = SimpleNamespace(_stderr_log_path=log, n_gpu_layers=999)
    assert exp._read_gpu_layers(proposer)["requested"] == 999
    proposer._stderr_log_path = tmp_path / "gone.log"
    assert exp._read_gpu_layers(proposer)["offloaded"] == 0

    monkeypatch.setattr(exp6764, "nvidia_smi_inventory", lambda: {"devices": [{"uuid": "x"}]})
    monkeypatch.setattr(
        exp6764,
        "rank_eligible_devices",
        lambda devices: {"selected_device": {"uuid": "GPU-eligible"}},
    )
    assert exp._eligible_device_still_selected({"uuid": "GPU-eligible"}) is True


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("device", "selected_device_no_longer_first_eligible"),
        ("port", "selected_port_no_longer_free"),
        ("load", "llama_server_load_failed"),
        ("changed_port", "llama_server_changed_frozen_port"),
        ("process", "llama_server_process_missing"),
        ("residency", "owner_bound_cuda_residency_missing"),
        ("unloading", "unload failed"),
        ("validating", "validate failed"),
        ("journal", "journal failed"),
        ("release", "teardown_or_release"),
    ],
)
def test_req_arc_wmte_6765_session_failure_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected: str,
) -> None:
    """REQ-ARC-WMTE-6765 converts every owned-session failure into row evidence."""
    planned = exp.row_plan()[0]
    model = _model(tmp_path, 0)
    selected = {
        "index": 1,
        "uuid": "GPU-eligible",
        "name": "NVIDIA GeForce RTX 3090",
        "memory_used_mb": 4,
        "memory_free_mb": 24_120,
    }

    class Lease:
        document = {"phase": "preflight"}
        journal_path = tmp_path / "journal.json"

        def owner_receipt(self):
            return {"task_id": planned["row_id"], "device_uuid": "GPU-eligible"}

        def transition(self, phase, **kwargs):
            if mode == "unloading" and phase == "unloading":
                raise exp.lease_api.LeaseError("unload failed")
            if mode == "validating" and phase == "validating":
                raise exp.lease_api.LeaseError("validate failed")
            self.document["phase"] = phase

        def release(self):
            return {"released": mode != "release", "phase": self.document["phase"]}

        def close(self):
            self.document["closed"] = True

    process = SimpleNamespace(pid=123, poll=lambda: None)

    class Proposer:
        n_gpu_layers = 999
        n_ctx = exp.CONTEXT_REQUESTED

        def __init__(self, **kwargs):
            self.port = kwargs["port"] + (1 if mode == "changed_port" else 0)
            self._proc = None if mode == "process" else process

        def _ensure_server(self):
            return mode != "load"

        def observed_n_ctx(self):
            return exp.CONTEXT_REQUESTED

        def observed_model_path(self):
            return model["model_path"]

    from carnot.agentic import arc_executable_world_model as world

    monkeypatch.setattr(world, "LocalGGUFProposer", Proposer)
    monkeypatch.setattr(exp, "_eligible_device_still_selected", lambda selected: mode != "device")
    monkeypatch.setattr(exp6764, "port_is_free", lambda port: mode != "port")
    monkeypatch.setattr(exp, "acquire_selected_lease", lambda **kwargs: Lease())
    monkeypatch.setattr(
        exp,
        "_gpu_snapshot",
        lambda *args: {
            **selected,
            "owned_pid_present": mode != "residency",
            "owned_pid_vram_mb": 0 if mode == "residency" else 18_000,
        },
    )
    monkeypatch.setattr(
        exp,
        "_process_identity",
        lambda process: {"pid": 123, "exit_code": None, "absent_after_exit": False},
    )
    monkeypatch.setattr(
        exp,
        "_read_gpu_layers",
        lambda proposer: {"requested": 999, "offloaded": 66, "total": 66},
    )
    monkeypatch.setattr(
        exp,
        "_build_window",
        lambda game: {
            "shown": [SimpleNamespace(grid=[[0]], next_grid=[[1]], action=1, data={})],
            "held": [],
            "cell": 1,
        },
    )
    monkeypatch.setattr(
        exp6753,
        "_run_live_row",
        lambda *args: (deepcopy(_row(tmp_path, planned)), "PROMPT"),
    )
    monkeypatch.setattr(
        exp,
        "terminate_owned_process",
        lambda process: {"pid": 123, "exit_code": 0, "absent_after_exit": True},
    )
    monkeypatch.setattr(
        exp,
        "_wait_for_vram_recovery",
        lambda *args: ({"passed": True, "owned_pid_present": False}, {"memory_used_mb": 4}),
    )

    def journal(path):
        if mode == "journal":
            raise exp.lease_api.LeaseError("journal failed")
        return {"phase_history": []}

    monkeypatch.setattr(exp, "read_journal", journal)
    monkeypatch.setenv("CARNOT_ARC_E3_DIR", str(tmp_path / "e3"))
    row = exp.run_live_row_session(
        model, selected, planned, port=4567, lease_runtime_dir=tmp_path / "leases"
    )
    assert row["failure_class"] is not None
    evidence = json.dumps(row)
    assert expected in evidence


def test_req_arc_wmte_6765_worker_timeout_and_missing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-6765 bounds a worker timeout and retains a failed row."""
    model = _model(tmp_path, 0)
    selected = _passing_preflight(tmp_path)["device_selection_receipt"]["selected_device"]
    planned = exp.row_plan()[0]

    class Process:
        pid = 321
        returncode = 9
        calls = 0

        def communicate(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise exp.subprocess.TimeoutExpired("worker", timeout)
            return "late-out", "late-error"

        def poll(self):
            return 9

    monkeypatch.setattr(exp.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(exp, "proc_start_ticks", lambda pid: 1)
    monkeypatch.setattr(
        exp6764, "_terminate_worker_group", lambda process, ticks: {"terminated": True}
    )
    row = exp.run_row_worker_subprocess(
        model,
        selected,
        planned,
        tmp_path / "runtime",
        port=4567,
        lease_runtime_dir=tmp_path / "leases",
    )
    assert row["failure_class"].startswith("session_lifecycle_failed:worker_output_missing")
    assert row["session_receipt"]["worker_process"]["timeout_cleanup"] == {"terminated": True}


def test_req_arc_wmte_6765_validator_and_parent_error_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-6765 rejects tampering and parent orchestration omissions."""
    preflight = _passing_preflight(tmp_path)
    artifact = exp.build_artifact(
        date="20260830",
        rows=_all_rows(tmp_path),
        preflight=preflight,
        started_ns=1,
        finished_ns=2,
    )
    changed = deepcopy(artifact)
    changed.pop("schema")
    changed["inference_substrate"] = "CPU"
    changed["frozen_manifest"] = {}
    changed["verifier_is_oracle"] = True
    changed["verdict_class"] = "invented"
    changed["source_receipts"]["source_access"]["game_source_read"] = True
    changed["rows"] = changed["rows"][:-1]
    changed["gpu_receipts"] = []
    changed["lease_receipts"] = []
    changed["model_specs"] = []
    changed["honest_verdict"] = "invented"
    changed["live_model_invoked"] = False
    changed["noninferiority_margin"] = -1
    changed["reproducibility_checksum"] = "bad"
    errors = exp.validate_artifact(changed)
    for expected in (
        "missing_field:schema",
        "schema",
        "inference_substrate",
        "frozen_manifest",
        "verifier_is_oracle",
        "verdict_class",
        "source_receipts",
        "row_denominator_or_order",
        "gpu_receipts",
        "lease_receipts",
        "model_specs",
        "honest_verdict",
        "live_model_invoked",
        "noninferiority_margin",
        "reproducibility_checksum",
    ):
        assert expected in errors

    blocked_preflight = deepcopy(preflight)
    blocked_preflight["all_passed"] = False
    blocked = exp.build_artifact(
        date="20260830",
        rows=[
            exp._blocked_row(planned, {}, "preflight_blocked:test") for planned in exp.row_plan()
        ],
        preflight=blocked_preflight,
        started_ns=1,
        finished_ns=2,
    )
    blocked["verdict_class"] = "null"
    blocked["rows"][0]["failure_class"] = "wrong"
    assert "blocked_verdict_class" in exp.validate_artifact(blocked)
    assert "blocked_rows" in exp.validate_artifact(blocked)

    missing_device = deepcopy(preflight)
    missing_device["device_selection_receipt"] = {}
    with pytest.raises(ValueError, match="selected_device"):
        exp.run(
            result_path=tmp_path / "missing-device.json",
            preflight_fn=lambda **kwargs: missing_device,
            clock=iter((1, 2)).__next__,
        )

    def failed_worker(*args, **kwargs):
        raise RuntimeError("owned worker failed")

    partial = exp.run(
        result_path=tmp_path / "worker-failed.json",
        preflight_fn=lambda **kwargs: preflight,
        worker_runner=failed_worker,
        clock=iter((1, 2)).__next__,
    )
    assert "owned worker failed" in partial["rows"][0]["failure_class"]

    monkeypatch.setattr(exp, "validate_artifact", lambda artifact: ["attack"])
    with pytest.raises(ValueError, match="attack"):
        exp.run(
            result_path=tmp_path / "invalid.json",
            preflight_fn=lambda **kwargs: blocked_preflight,
            clock=iter((1, 2)).__next__,
        )


def test_req_arc_wmte_6765_main_worker_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-6765 requires a complete one-row worker CLI contract."""
    with pytest.raises(SystemExit) as exc:
        exp.main(["--worker-job", str(tmp_path / "job.json")])
    assert exc.value.code == 2
    monkeypatch.setattr(exp, "_worker_entry", lambda *args: 7)
    assert (
        exp.main(
            [
                "--worker-job",
                str(tmp_path / "job.json"),
                "--worker-output",
                str(tmp_path / "row.json"),
                "--port",
                "4567",
                "--lease-runtime-dir",
                str(tmp_path / "leases"),
            ]
        )
        == 7
    )

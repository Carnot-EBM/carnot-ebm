"""Run the V637 local scored-stack ARC selfparse dry run.

The child uses the tracked submission factory and the shipped evaluator lifecycle.
The parent owns resource checks, sequential GPU leases, deadlines, and the terminal
artifact. No game adapter, banked solution, or private state enters the policy.

Spec refs: REQ-ARC-WMTE-7234 and SCENARIO-ARC-WMTE-7234-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import io
import json
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import sys
import time
from typing import Any
import urllib.request

import yaml

from carnot import experiment_7206_v635_arc_volume_a as base


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"

TASK_ID = "exp7234-arc-scored-dryrun"
EXPERIMENT_ID = 7234
MILESTONE = "2026.09.637"
RUN_DATE = "20260912"
RANDOM_SEED = 7_234_001
ACTION_BUDGET = 256
STARTUP_TIMEOUT_S = 240
BACKEND_TIMEOUT_S = 1500
TOTAL_RUNTIME_CAP_S = 3300
INDUCTION_TIMEOUT_S = 1440
N_CTX = 49152
COMPLETION_BUDGET = 4096

PRIMARY_MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
PRIMARY_QUANTIZATION = "Q4_K_M"
COMPARATOR_MODEL_ID = "RedHatAI/Qwen3.8-27B-INT4"
GAME_ROTATION = ("r11l", "lp85", "ls20", "wa30", "cd82", "sp80", "su15", "tu93")
PRIOR_TASK_TARGETS = frozenset({"r11l"})
GAME = "lp85"

SCHEMA = "carnot.experiment_7234.v637_arc_scored_dryrun.v1"
SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
ROADMAP_PATH = Path("results/raw/experiment_7233/selected-roadmap.yaml")
RESULT_PATH = Path("results/experiment_7234_v637_arc_scored_dryrun.json")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7234_v637_arc_scored_dryrun/running.json")
RAW_DIR = Path("results/raw/experiment_7234")
MODULE_PATH = Path("python/carnot/experiment_7234_v637_arc_scored_dryrun.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7234_v637_arc_scored_dryrun.py")
TEST_PATH = Path("tests/python/test_experiment_7234_v637_arc_scored_dryrun.py")

EXPECTED_TASK_CONTRACT = {
    "id": TASK_ID,
    "milestone": MILESTONE,
    "deliverable": RESULT_PATH.as_posix(),
    "gated_on": None,
    "prior_failures": [
        {
            "experiment_id": "exp7221-arc-session",
            "verdict": "complete_null_no_observed_missing_tool_demand_cumulative_n_10",
            "addressed_by": (
                "The ten-induction target is complete. This attempt exercises the current scored "
                "construction and vLLM backend after the documented isolated transport success; "
                "it is not another volume collection."
            ),
            "retire_if_same_verdict": True,
        }
    ],
    "operator_override": (
        "2026-09-11 ops/known-issues.md scored-stack dry-run directive; the later same-day "
        "selfparse correction closes isolated parsing only. Run locally without submitting."
    ),
}

REQUIRED_SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/kaggle/submission_kernel/main.py"),
    Path("scripts/kaggle/submission_kernel/kernel-metadata.json"),
    Path("scripts/arc_leaderboard_eval.py"),
    Path("python/carnot/agentic/arc_competition_agent.py"),
    Path("python/carnot/agentic/arc_executable_world_model.py"),
    Path("python/carnot/agentic/arc_induction_tool_loop.py"),
    Path("python/carnot/agentic/arc_eval_provenance.py"),
    Path("python/carnot/experiment_7221_v636_arc_session.py"),
    Path("results/experiment_7221_v636_arc_session.json"),
    Path("python/carnot/inference/sota_models.py"),
    Path("python/carnot/inference/llama_server_supervisor.py"),
    Path("python/carnot/gpu_lease_phase_journal.py"),
    Path("ops/known-issues.md"),
    Path("ops/arc_solve_registry.yaml"),
    Path("tests/python/test_arc_vllm_backend.py"),
    SPEC_PATH,
    ROADMAP_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)

FIELD_PRINCIPLES: JsonDict = {
    "schema": "Version the artifact and bind experiment_id and milestone to this task.",
    "experiment_id": "Bind this receipt to Experiment 7234.",
    "milestone": "Bind this receipt to milestone 2026.09.637.",
    "status": "Terminal complete or blocked only; unfinished work uses a separate checkpoint path.",
    "run_date": "Use 20260912; retain actual UTC start and end timestamps.",
    "started_at_utc": "Actual task start in UTC.",
    "ended_at_utc": "Actual terminal construction time in UTC.",
    "field_principles": "Keep ordinary values at top level; put their explanations in this map.",
    "preconditions_checked": "Observed paths, resources, model identity and upstream checks before expensive work.",
    "inference_substrate": "Use the recognized literal for the operation actually executed.",
    "inference_substrate_class": "Actual class determines the duration floor; never pad duration or relabel to pass.",
    "execution_venue": "Top-level orchestration is host; board rows name kv260, gatemate or polarfire.",
    "execution_host": "Actual hostname, distinct from execution_venue.",
    "duration_s": "Measured monotonic elapsed work; record phase spans separately.",
    "MODEL_SPECS": "Models actually invoked; [] for tasks with no LLM.",
    "model_invoked": "Current task execution only; historical sources and injected fixtures are separate.",
    "source_artifact_hashes": "Hash source code, public inputs, private evaluator inputs and raw output files.",
    "rows": "Every comparison retains one row per independent unit and arm, with errors and abstentions.",
    "sample_size_budget": "Predeclared independent units, attempted/completed/censored units, and stopping rule.",
    "random_seed": "Freeze seeds and schedules before observing evaluation labels.",
    "reproducibility_checksum": "Hash the exact settings, inputs and raw rows supporting the result.",
    "gate_check_summary": "Every blocked_* verdict names check, upstream, artifact_field, expected and observed value.",
    "verifier_is_oracle": "True when the verification authority also defines correctness; separate code is insufficient independence.",
    "verdict_class": "Exactly positive | circular_positive | null | blocked | disqualified | partial. External incompleteness is blocked.",
    "honest_verdict": "Completed findings start complete_ or complete:. External absence starts blocked_. Failed acceptance forbids positive.",
    "acceptance_gate_results": "Preserve each frozen criterion, actual value and pass/fail independently of task completion.",
    "inference_mode": "live_gpu only after actual CUDA work.",
    "runner_receipt": "Task-owned invocation identity and actual transport counts, not copied upstream receipts.",
    "raw_request_manifest": "Exact model, prompt, parameters and response hashes for every call.",
    "phase_spans": "Measured model load, generation, scoring and validation times.",
    "scored_dryrun_complete_score": "Every planned local backend has a complete outcome or an explicit external block.",
    "local_scored_path_ready_score": "Actual dispatch, generated engine and policy consumption, not import or HTTP success.",
    "per_game_results": "Backend, seed, adapter disabled, actions, valid models, progress and cost for each session.",
    "submission_configuration_diff": "Every difference from the tracked scored kernel, including unavailable NVFP4, Blackwell and gateway.",
    "solve_provenance": "live_agent_self_discovery for any credited solve; development_proxy for any off-path probe.",
    "offline_reproduced": "True only after action replay through the shipped reproduction gate.",
    "reproduced_levels": "New reproduced levels only; zero is a legitimate outcome.",
    "selection_receipt": "Freeze the metadata-only game selection and exclusions before model work.",
    "new_solve_claimed": "A solve requires registry precheck and live replay; this dry run defaults false.",
    "official_leaderboard_score_reported": "Local public-gateway sessions do not expose a hidden competition score.",
    "cross_backend_efficacy_claimed": "Two clustered sessions cannot establish backend efficacy.",
    "validation_receipts": "Retain exact validation commands and outcomes beside the terminal evidence.",
}

atomic_write = base.atomic_write
artifact_checksum = base.artifact_checksum
gate_check = base.gate_check
gate_summary = base.gate_summary
sha256_file = base.sha256_file
unwrap_evidence_value = base.unwrap_evidence_value
is_quarantined = base.is_quarantined


def _progress(phase: int, event: str, **fields: Any) -> None:  # pragma: no cover - live output.
    """Emit one unbuffered truthful progress row."""

    print(json.dumps({"phase": phase, "event": event, **fields}, sort_keys=True), flush=True)


def _iso_now() -> str:  # pragma: no cover - live timestamp.
    return datetime.now(UTC).isoformat(timespec="seconds")


def select_game_from_rotation(
    rotation: Sequence[str], *, credited_task_targets: set[str] | frozenset[str]
) -> tuple[str, JsonDict]:
    """Select the first metadata rotation entry not credited by the prior task."""

    selected = next(game for game in rotation if game not in credited_task_targets)
    return selected, {
        "selected_game": selected,
        "rotation": list(rotation),
        "excluded_credited_task_targets": sorted(credited_task_targets),
        "selection_basis": "first_metadata_rotation_entry_not_used_by_prior_task",
        "selection_used_outcome_labels": False,
    }


def task_contract(path: Path) -> JsonDict:
    """Read only the frozen roadmap row for this task."""

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    tasks = payload if isinstance(payload, list) else payload.get("tasks", [])
    task = next(
        (dict(row) for row in tasks if isinstance(row, Mapping) and row.get("id") == TASK_ID), {}
    )
    return {key: task.get(key) for key in EXPECTED_TASK_CONTRACT}


def build_disposable_submitted_policy(  # pragma: no cover - separate heavy test.
    game: str, proposer: Any
) -> tuple[Any, JsonDict]:
    """Construct the real submitted policy without importing any solution source."""

    from carnot.agentic.arc_competition_agent import SUBMITTED_AGENT_CONFIG, make_carnot_agent

    class LocalAgentBase:
        def __init__(self, game_id: str) -> None:
            self.game_id = game_id

        def cleanup(self, scorecard: Any = None) -> None:
            del scorecard

    agent_type = make_carnot_agent(LocalAgentBase, cascade=True, proposer=proposer)
    holder = agent_type(game_id=game)
    config_bytes = json.dumps(SUBMITTED_AGENT_CONFIG, sort_keys=True, default=str).encode()
    return holder._policy, {
        "factory": "make_carnot_agent",
        "policy_class": "E3AgentPolicy",
        "cascade": True,
        "adapter_disabled": True,
        "denied_inputs": [
            "banked_solutions",
            "game_adapter",
            "game_source",
            "ground_truth_state",
            "registry_contents",
            "solved_trajectories",
        ],
        "policy_inputs": ["public_frames", "available_actions", "own_transitions"],
        "submitted_config_sha256": "sha256:" + hashlib.sha256(config_bytes).hexdigest(),
        "submitted_config": deepcopy(SUBMITTED_AGENT_CONFIG),
    }


def session_environment(
    base_env: Mapping[str, str],
    *,
    backend: str,
    model_path: Path,
    gpu_index: int,
    port: int,
    raw_dir: Path,
) -> dict[str, str]:
    """Create the isolated child environment before ARC imports."""

    env = dict(base_env)
    for key in (
        "CARNOT_ARC_SUPERVISOR_TOOL_ARM",
        "CARNOT_ARC_GGUF_PATH",
        "CARNOT_ARC_VLLM_MODEL_DIR",
    ):
        env.pop(key, None)
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(REPO_ROOT / "python"),
            "CARNOT_FORCE_LIVE": "1",
            "CARNOT_ARC_LLM_BACKEND": backend,
            "CARNOT_ARC_INDUCE_TOOL_LOOP": "selfparse",
            "CARNOT_ARC_INDUCE_N_CTX": str(N_CTX),
            "CARNOT_ARC_INDUCE_MAX_TOKENS": str(COMPLETION_BUDGET),
            "CARNOT_ARC_INDUCE_TIMEOUT": str(INDUCTION_TIMEOUT_S),
            "CARNOT_ARC_LLAMA_SERVER_PARALLEL": "1",
            "CARNOT_ARC_GENERATOR_REQUIRE_CUDA": "1",
            "CARNOT_ARC_GENERATOR_CUDA_GPU": str(gpu_index),
            "CARNOT_ARC_RANDOM_SEED": str(RANDOM_SEED),
            "CARNOT_ARC_GENERATOR_SEED": str(RANDOM_SEED),
            "CARNOT_ARC_PROPOSER_PORT": str(port),
            "CARNOT_ARC_TOOL_GAP_RECEIPTS": "1",
            "CARNOT_ARC_TOOL_GAP_RECEIPT_PATH": str(raw_dir / "tool_gap_receipts.json"),
            "CARNOT_ARC_E3_DIR": str(raw_dir / "e3"),
            "CARNOT_ARC_SERVER_LOG_DIR": str(raw_dir / "server_logs"),
            "CARNOT_ARC_ACTION_PROVENANCE": "1",
            "CARNOT_ARC_ACTION_PROVENANCE_DIR": str(raw_dir / "action_provenance"),
            "CUDA_VISIBLE_DEVICES": str(gpu_index),
        }
    )
    if backend == "vllm":
        env["CARNOT_ARC_VLLM_MODEL_DIR"] = str(model_path)
    else:
        env["CARNOT_ARC_GGUF_PATH"] = str(model_path)
        env["CARNOT_ARC_MTP"] = "0"
    return env


def _tool_calls(attempt: Mapping[str, Any]) -> int:
    value = attempt.get("tool_calls_total")
    gap = attempt.get("tool_gap")
    if value is None and isinstance(gap, Mapping):
        value = gap.get("tool_calls_total")
    return int(value or 0)


def _tool_names(attempt: Mapping[str, Any]) -> Mapping[str, Any]:
    value = attempt.get("tool_calls_by_name")
    gap = attempt.get("tool_gap")
    if not isinstance(value, Mapping) and isinstance(gap, Mapping):
        value = gap.get("tool_calls_by_name")
    return value if isinstance(value, Mapping) else {}


def reduce_backend_session(session: Mapping[str, Any]) -> JsonDict:
    """Reduce immutable child rows without converting failures into missing work."""

    attempts = [dict(row) for row in session.get("induction_rows", []) if isinstance(row, Mapping)]
    actions = [dict(row) for row in session.get("action_rows", []) if isinstance(row, Mapping)]
    run_row = dict(session.get("run_row", {}) or {})
    parsed_dispatches = sum(
        len(row.get("parsed_tool_call_names", []))
        for row in session.get("completions", [])
        if isinstance(row, Mapping)
    )
    dispatches = parsed_dispatches or sum(_tool_calls(row) for row in attempts)
    writes = sum(bool(row.get("engine_written")) for row in attempts)
    consumed = sum(
        row.get("top_branch") in {"execute.plan_step", "induce.plan_from_current"}
        for row in actions
    )
    non_identity = sum(
        row.get("engine_functionally_identity") is False and row.get("engine_written") is True
        for row in attempts
    )
    trust = sum(
        row.get("engine_written") is True
        and not row.get("skipped")
        and (row.get("planned") is True or float(row.get("verify_accuracy") or 0.0) >= 0.5)
        for row in attempts
    )
    planned = sum(bool(row.get("planned")) for row in attempts)
    completions = [dict(row) for row in session.get("completions", []) if isinstance(row, Mapping)]
    transport = any(row.get("response_sha256") or row.get("response_bytes") for row in completions)
    return {
        "backend": str(session.get("backend") or "unknown"),
        "disposition": str(session.get("disposition") or "complete"),
        "model_invoked": bool(session.get("model_invoked")),
        "model_spec": deepcopy(session.get("model_spec")),
        "transport_completed": bool(transport),
        "semantic_usable": bool(dispatches and writes and consumed),
        "tool_dispatches": dispatches,
        "engine_writes": writes,
        "policy_engine_consumptions": consumed,
        "valid_engine_count": writes,
        "non_identity_prediction_count": non_identity,
        "held_future_prediction_accuracy": [
            row.get("heldout_accuracy")
            for row in attempts
            if row.get("heldout_accuracy") is not None
        ],
        "trust_acceptance_count": trust,
        "model_planned_actions": planned,
        "actions": int(run_row.get("actions") or len(actions)),
        "levels": int(run_row.get("levels") or 0),
        "progress": run_row.get("reached", run_row.get("levels", 0)),
        "total_cost": run_row.get("wall_s", session.get("duration_s")),
        "action_rows": actions,
        "induction_rows": attempts,
        "raw_request_manifest": completions,
        "phase_spans": deepcopy(session.get("phase_spans", [])),
        "runner_receipt": deepcopy(session.get("runner_receipt", {})),
        "gpu_receipts": deepcopy(session.get("gpu_receipts", {})),
        "factory_receipt": deepcopy(session.get("factory_receipt", {})),
        "error": session.get("error"),
    }


def _comparison_rows(backends: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for backend in backends:
        for metric in (
            "actions",
            "valid_engine_count",
            "non_identity_prediction_count",
            "trust_acceptance_count",
            "model_planned_actions",
            "levels",
        ):
            rows.append(
                {
                    "unit_id": f"{GAME}:{backend.get('backend')}",
                    "arm": backend.get("backend"),
                    "seed": RANDOM_SEED,
                    "metric": metric,
                    "value": backend.get(metric),
                    "error": backend.get("error"),
                    "abstention": backend.get("disposition") == "blocked_external_absence",
                }
            )
    return rows


def build_terminal_artifact(
    *,
    run_date: str,
    duration_s: float,
    started_at_utc: str,
    ended_at_utc: str,
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    selection_receipt: Mapping[str, Any],
    backend_rows: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build one terminal artifact from preflight and backend dispositions."""

    backends = [deepcopy(dict(row)) for row in backend_rows]
    summary = gate_summary(checks)
    names = {row.get("backend") for row in backends}
    dispositions_complete = names == {"llamacpp", "vllm"} and all(
        row.get("disposition") in {"complete", "blocked_external_absence"} for row in backends
    )
    blocked = not summary["passed"] or (not backends and not dispositions_complete)
    invoked = any(row.get("model_invoked") is True for row in backends)
    ready = any(
        int(row.get("tool_dispatches") or 0) > 0
        and int(row.get("engine_writes") or 0) > 0
        and int(row.get("policy_engine_consumptions") or 0) > 0
        for row in backends
    )
    status = "blocked" if blocked else "complete"
    verdict = "blocked" if blocked else "positive" if ready else "null"
    honest = (
        f"blocked_{summary.get('failed_check') or 'required_preconditions'}"
        if blocked
        else "complete_positive_local_scored_path_reached"
        if ready
        else "complete_null_no_policy_consumed_world_model"
    )
    specs = [deepcopy(row["model_spec"]) for row in backends if row.get("model_invoked")]
    raw_manifest = [
        deepcopy(request) for row in backends for request in row.get("raw_request_manifest", [])
    ]
    spans = [deepcopy(span) for row in backends for span in row.get("phase_spans", [])]
    attempted = sum(row.get("model_invoked") is True for row in backends)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": status,
        "run_date": run_date,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "field_principles": {},
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "inference_substrate": "live_llm_inference" if invoked else "blocked_no_run",
        "inference_substrate_class": "model_full_generation" if invoked else "blocked_no_run",
        "execution_venue": "host",
        "execution_host": socket.gethostname(),
        "duration_s": round(float(duration_s), 6),
        "MODEL_SPECS": specs,
        "model_invoked": invoked,
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": _comparison_rows(backends),
        "sample_size_budget": {
            "planned_independent_units": 2,
            "attempted_units": attempted,
            "completed_dispositions": len(backends),
            "censored_units": sum(
                row.get("disposition") == "blocked_external_absence" for row in backends
            ),
            "action_limit_per_backend": ACTION_BUDGET,
            "backend_timeout_s": BACKEND_TIMEOUT_S,
            "total_runtime_cap_s": TOTAL_RUNTIME_CAP_S,
            "stopping_rule": "one frozen seed per backend; absence is recorded; no outcome tuning",
        },
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": summary,
        "verifier_is_oracle": True,
        "verdict_class": verdict,
        "honest_verdict": honest,
        "acceptance_gate_results": [
            {
                "criterion": "both_backend_dispositions_recorded",
                "actual_value": dispositions_complete,
                "passed": dispositions_complete,
            },
            {
                "criterion": "actual_dispatch_engine_write_policy_consumption",
                "actual_value": ready,
                "passed": ready,
            },
            {
                "criterion": "equal_action_and_token_caps",
                "actual_value": {"actions": ACTION_BUDGET, "tokens": COMPLETION_BUDGET},
                "passed": True,
            },
        ],
        "inference_mode": "live_gpu" if invoked else "not_invoked",
        "runner_receipt": {
            "backends": [deepcopy(row.get("runner_receipt", {})) for row in backends],
            "completed_transports": sum(row.get("transport_completed") is True for row in backends),
            "semantically_usable_transports": sum(
                row.get("semantic_usable") is True for row in backends
            ),
            "raw_generation_endpoints": sorted(
                {
                    str(request.get("endpoint"))
                    for request in raw_manifest
                    if request.get("endpoint") in {"/completion", "/v1/completions"}
                }
            ),
            "selfparse_chat_endpoints": sorted(
                {
                    str(request.get("endpoint"))
                    for request in raw_manifest
                    if request.get("endpoint") == "/v1/chat/completions"
                }
            ),
        },
        "raw_request_manifest": raw_manifest,
        "phase_spans": spans,
        "scored_dryrun_complete_score": int(dispositions_complete and summary["passed"]),
        "local_scored_path_ready_score": int(ready),
        "per_game_results": backends,
        "submission_configuration_diff": {
            "gateway": "public_local_arcade_gateway_not_competition_gateway",
            "gpu": "local_RTX_3090_Ampere_not_Kaggle_Blackwell_96GB",
            "primary_quantization": "Q4_K_M_GGUF_not_tracked_NVFP4_safetensors",
            "primary_runtime": "llama.cpp_not_tracked_vLLM",
            "context": f"local_single_stream_{N_CTX}_not_tracked_concurrency_derived_pool",
            "induction_token_cap": f"local_{COMPLETION_BUDGET}_not_tracked_kernel_131072",
            "kv_precision": "llama_q8_0_and_vllm_auto_not_tracked_fp8",
            "batching": "sequential_single_sequence_not_continuous_batch_8",
            "competition_equivalent_claim": False,
        },
        "solve_provenance": "live_agent_self_discovery",
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "selection_receipt": deepcopy(dict(selection_receipt)),
        "new_solve_claimed": False,
        "official_leaderboard_score_reported": False,
        "cross_backend_efficacy_claimed": False,
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
    }
    artifact["field_principles"] = {key: FIELD_PRINCIPLES[key] for key in artifact}
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(value: Mapping[str, Any] | Path) -> list[str]:
    """Cold-check the terminal artifact without trusting wrapped dictionaries."""

    if isinstance(value, Path):
        try:
            artifact = json.loads(value.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return [f"artifact_unreadable:{type(exc).__name__}"]
    else:
        artifact = dict(value)
    errors: list[str] = []
    if set(artifact.get("field_principles", {})) != set(artifact):
        errors.append("field_principles_must_cover_every_top_level_field")
    if artifact.get("schema") != SCHEMA or artifact.get("experiment_id") != EXPERIMENT_ID:
        errors.append("schema_or_experiment_identity_mismatch")
    if artifact.get("run_date") != RUN_DATE:
        errors.append("run_date_mismatch")
    if artifact.get("status") not in {"complete", "blocked"}:
        errors.append("status_not_terminal")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if artifact.get("status") == "complete" and not str(artifact.get("honest_verdict")).startswith(
        ("complete_", "complete:")
    ):
        errors.append("complete_honest_verdict_prefix_invalid")
    if artifact.get("status") == "blocked" and not str(artifact.get("honest_verdict")).startswith(
        "blocked_"
    ):
        errors.append("blocked_honest_verdict_prefix_invalid")
    backends = artifact.get("per_game_results", [])
    computed_ready = any(
        int(row.get("tool_dispatches") or 0) > 0
        and int(row.get("engine_writes") or 0) > 0
        and int(row.get("policy_engine_consumptions") or 0) > 0
        for row in backends
        if isinstance(row, Mapping)
    )
    if artifact.get("local_scored_path_ready_score") != int(computed_ready):
        errors.append("local_scored_path_ready_score_inconsistent")
    invoked_specs = [
        row.get("model_spec")
        for row in backends
        if isinstance(row, Mapping) and row.get("model_invoked") is True
    ]
    if artifact.get("MODEL_SPECS") != invoked_specs:
        errors.append("model_specs_do_not_match_invocations")
    if artifact.get("model_invoked") != bool(invoked_specs):
        errors.append("model_invoked_inconsistent")
    if artifact.get("model_invoked") and artifact.get("duration_s", 0) < 60:
        errors.append("model_full_generation_duration_floor_failed")
    checksum = artifact.get("reproducibility_checksum")
    if checksum != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def _free_port() -> int:  # pragma: no cover - live resource allocation.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _process_start_tick(pid: int) -> int | None:  # pragma: no cover - live receipt.
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().split()[21])
    except (OSError, ValueError, IndexError):
        return None


def _gpu_inventory() -> list[JsonDict]:  # pragma: no cover - live CUDA host.
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    apps = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    rows: list[JsonDict] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 5:
            rows.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "uuid": parts[2],
                    "free_memory_mb": int(parts[3]),
                    "total_memory_mb": int(parts[4]),
                    "compute_apps": [],
                }
            )
    by_uuid = {str(row["uuid"]): row for row in rows}
    for line in apps.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3 and parts[1] in by_uuid:
            by_uuid[parts[1]]["compute_apps"].append(
                {"pid": int(parts[0]), "used_memory_mb": int(parts[2])}
            )
    return rows


def _model_digest(path: Path) -> str:  # pragma: no cover - large live model bytes.
    """Hash one GGUF or the exact files in one safetensors snapshot."""

    if path.is_file():
        return sha256_file(path)
    rows = []
    for child in sorted(path.iterdir()):
        if child.is_file() and (
            child.suffix in {".json", ".safetensors"} or child.name.startswith("tokenizer")
        ):
            rows.append([child.name, sha256_file(child)])
    payload = json.dumps(rows, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


class _CapturedResponse:  # pragma: no cover - live HTTP seam.
    def __init__(self, response: Any, data: bytes) -> None:
        self._buffer = io.BytesIO(data)
        self.status = getattr(response, "status", None)
        self.headers = getattr(response, "headers", {})

    def read(self, amount: int = -1) -> bytes:
        return self._buffer.read(amount)

    def __enter__(self) -> "_CapturedResponse":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def getcode(self) -> Any:
        return self.status

    def info(self) -> Any:
        return self.headers


class RequestCapture:  # pragma: no cover - live HTTP seam.
    """Persist exact generation request and response bytes."""

    def __init__(self, root: Path, backend: str) -> None:
        self.root = root
        self.backend = backend
        self.rows: list[JsonDict] = []
        self.original = urllib.request.urlopen

    def install(self) -> None:
        urllib.request.urlopen = self.urlopen

    def restore(self) -> None:
        urllib.request.urlopen = self.original

    def urlopen(self, request: Any, *args: Any, **kwargs: Any) -> Any:
        url = request.full_url if isinstance(request, urllib.request.Request) else str(request)
        endpoint = next(
            (
                value
                for value in ("/v1/chat/completions", "/v1/completions", "/completion")
                if url.endswith(value)
            ),
            None,
        )
        if endpoint is None or not isinstance(request, urllib.request.Request):
            return self.original(request, *args, **kwargs)
        body = bytes(request.data or b"")
        index = len(self.rows)
        request_path = self.root / "requests" / f"{index:03d}_request.json"
        response_path = self.root / "requests" / f"{index:03d}_response.json"
        request_path.parent.mkdir(parents=True, exist_ok=True)
        request_path.write_bytes(body)
        started = time.monotonic()
        payload = json.loads(body) if body else {}
        row = {
            "backend": self.backend,
            "sequence": index,
            "endpoint": endpoint,
            "url": url,
            "request_path": str(request_path),
            "request_sha256": "sha256:" + hashlib.sha256(body).hexdigest(),
            "request_bytes": len(body),
            "model": payload.get("model"),
            "seed": payload.get("seed"),
            "parameters": {
                key: payload.get(key)
                for key in (
                    "max_tokens",
                    "n_predict",
                    "temperature",
                    "top_p",
                    "top_k",
                    "thinking_budget_tokens",
                )
                if key in payload
            },
            "response_path": str(response_path),
            "response_sha256": None,
            "response_bytes": 0,
            "prompt_tokens": None,
            "completion_tokens": None,
            "finish_reason": None,
            "wall_s": None,
            "transport_completed": False,
            "transport_error": None,
        }
        self.rows.append(row)
        try:
            response = self.original(request, *args, **kwargs)
            response_bytes = response.read()
            response.close()
        except Exception as exc:
            row["wall_s"] = round(time.monotonic() - started, 6)
            row["transport_error"] = f"{type(exc).__name__}: {exc}"[:300]
            raise
        response_path.write_bytes(response_bytes)
        output = json.loads(response_bytes) if response_bytes else {}
        choice = (output.get("choices") or [{}])[0] if isinstance(output, Mapping) else {}
        usage = output.get("usage", {}) if isinstance(output, Mapping) else {}
        timings = output.get("timings", {}) if isinstance(output, Mapping) else {}
        row.update(
            {
                "response_sha256": "sha256:" + hashlib.sha256(response_bytes).hexdigest(),
                "response_bytes": len(response_bytes),
                "prompt_tokens": usage.get("prompt_tokens", timings.get("prompt_n")),
                "completion_tokens": usage.get("completion_tokens", timings.get("predicted_n")),
                "finish_reason": choice.get("finish_reason", output.get("stop_type")),
                "wall_s": round(time.monotonic() - started, 6),
                "transport_completed": True,
            }
        )
        return _CapturedResponse(response, response_bytes)


def _launch_vllm(
    model_path: Path, port: int, python: Path
) -> subprocess.Popen[bytes]:  # pragma: no cover
    command = [
        str(python),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model_path),
        "--served-model-name",
        "qwen38-int4",
        "--max-model-len",
        str(N_CTX),
        "--gpu-memory-utilization",
        "0.95",
        "--max-num-seqs",
        "1",
        "--enforce-eager",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    return subprocess.Popen(command, start_new_session=True)


def _wait_health(port: int, deadline_s: float) -> bool:  # pragma: no cover
    started = time.monotonic()
    next_beat = started
    while time.monotonic() - started < deadline_s:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                return True
        except Exception:
            now = time.monotonic()
            if now >= next_beat:
                _progress(4, "model_load_heartbeat", elapsed_s=round(now - started, 1))
                next_beat = now + 30
            time.sleep(2)
    return False


def _attempt_rows(policy: Any, raw_dir: Path) -> list[JsonDict]:  # pragma: no cover
    rows: list[JsonDict] = []
    world_model = raw_dir / "e3" / GAME / "world_model.py"
    attempts = list(getattr(policy, "induction_attempts", []) or [])
    for index, value in enumerate(attempts):
        row = deepcopy(dict(value))
        gap = row.get("tool_gap") if isinstance(row.get("tool_gap"), Mapping) else {}
        row["attempt_index"] = index
        row["tool_calls_total"] = _tool_calls(row)
        row["tool_calls_by_name"] = deepcopy(dict(_tool_names(row)))
        row["engine_written"] = bool(
            row.get("engine_retention", {}).get("store_path")
            or (world_model.is_file() and index == len(attempts) - 1)
        )
        identity = row.get("engine_functionally_identity")
        refinements = row.get("refinement_rounds", [])
        refinement = refinements[-1] if refinements else {}
        if identity is None:
            identity = refinement.get("engine_functionally_identity")
        row["engine_functionally_identity"] = identity
        row["heldout_accuracy"] = row.get("heldout_accuracy")
        if row["heldout_accuracy"] is None:
            row["heldout_accuracy"] = refinement.get(
                "heldout_accuracy", gap.get("best_holdout_accuracy")
            )
        rows.append(row)
    return rows


def _enrich_request_manifest(session: JsonDict) -> None:  # pragma: no cover - raw live bytes.
    """Re-run the production XML parser over persisted response bytes."""

    from carnot.agentic.arc_induction_tools import parse_xml_tool_calls

    for row in session.get("completions", []):
        path = Path(str(row.get("response_path") or ""))
        if not path.is_file():
            row["parsed_tool_call_names"] = []
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            message = (raw.get("choices") or [{}])[0].get("message") or {}
            calls, blocks, unparsed = parse_xml_tool_calls(str(message.get("content") or ""))
        except (OSError, json.JSONDecodeError, AttributeError):
            calls, blocks, unparsed = [], 0, 0
        row["parsed_tool_call_names"] = [call["function"]["name"] for call in calls]
        row["xml_blocks_seen"] = blocks
        row["xml_blocks_unparsed"] = unparsed


def run_backend_child(args: argparse.Namespace) -> int:  # pragma: no cover - live CUDA.
    """Run one isolated backend through the production factory and evaluator."""

    random.seed(RANDOM_SEED)
    import numpy as np

    np.random.seed(RANDOM_SEED)
    if str(SCRIPTS_ROOT) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_ROOT))
    from arc_leaderboard_eval import ProgressWriter, run_game
    from carnot.agentic.arc_eval_provenance import huggingface_snapshot_revision
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint_path)
    session_path = Path(args.session_output)
    started = time.monotonic()
    spans: list[JsonDict] = []
    capture = RequestCapture(raw_dir, args.backend)
    proposer: Any = None
    server: subprocess.Popen[Any] | None = None
    session: JsonDict = {
        "backend": args.backend,
        "model_invoked": False,
        "completions": capture.rows,
        "action_rows": [],
        "induction_rows": [],
        "run_row": {},
        "phase_spans": spans,
        "error": None,
    }
    try:
        capture.install()
        load_start = time.monotonic()
        _progress(4, "model_load_start", backend=args.backend, model_path=args.model_path)
        if args.backend == "vllm":
            server = _launch_vllm(
                Path(args.model_path),
                int(args.port),
                Path(os.environ["CARNOT_EXP7234_VLLM_PYTHON"]),
            )
            if not _wait_health(int(args.port), STARTUP_TIMEOUT_S):
                raise RuntimeError("vllm_startup_timeout")
            proposer = LocalGGUFProposer(
                repo_substr="Qwen3.8-27B",
                model_path=None,
                port=int(args.port),
                mtp=False,
                use_chat_template=False,
                max_tokens=COMPLETION_BUDGET,
                timeout=INDUCTION_TIMEOUT_S,
                tries=1,
            )
            runner = "LocalGGUFProposer_vllm_selfparse"
            quantization = "INT4"
            model_id = COMPARATOR_MODEL_ID
            revision = huggingface_snapshot_revision(str(Path(args.model_path)), model_id)
        else:
            proposer = LocalGGUFProposer(
                repo_substr="Qwen3.8-27B",
                model_path=str(Path(args.model_path).absolute()),
                port=int(args.port),
                mtp=False,
                kv_quant="q8_0",
                use_chat_template=True,
                n_gpu_layers=999,
                max_tokens=COMPLETION_BUDGET,
                timeout=INDUCTION_TIMEOUT_S,
                tries=1,
            )
            if not proposer._ensure_server():
                raise RuntimeError("llama_server_startup_failed")
            server = proposer._proc
            runner = "LocalGGUFProposer_llama.cpp"
            quantization = PRIMARY_QUANTIZATION
            model_id = PRIMARY_MODEL_ID
            revision = huggingface_snapshot_revision(str(Path(args.model_path)), model_id)
        load_end = time.monotonic()
        spans.append(
            {"backend": args.backend, "phase": "model_load", "duration_s": load_end - load_start}
        )
        _progress(4, "model_load_end", backend=args.backend, elapsed_s=load_end - load_start)
        session["model_loaded"] = True
        model_hash = os.environ.get("CARNOT_EXP7234_MODEL_HASH") or _model_digest(
            Path(args.model_path)
        )
        session["model_spec"] = {
            "hf_id": model_id,
            "quantization": quantization,
            "model_path": str(Path(args.model_path).absolute()),
            "revision": revision,
            "model_file_hash": model_hash,
        }
        server_pid = getattr(server, "pid", None)
        session["runner_receipt"] = {
            "runner": runner,
            "server_pid": server_pid,
            "server_pid_start_tick": _process_start_tick(server_pid) if server_pid else None,
            "native_binary": (
                proposer.last_launch_argv[0]
                if args.backend == "llamacpp" and proposer.last_launch_argv
                else sys.executable
            ),
            "server_command": (
                list(proposer.last_launch_argv)
                if args.backend == "llamacpp"
                else list(getattr(server, "args", []))
            ),
            "raw_generation_endpoint": "/completion"
            if args.backend == "llamacpp"
            else "/v1/completions",
            "selfparse_chat_endpoint": "/v1/chat/completions",
            "completion_budget": COMPLETION_BUDGET,
            "context": N_CTX,
            "seed": RANDOM_SEED,
        }
        atomic_write(
            checkpoint,
            {
                "stage": "model_loaded",
                "model_loaded": True,
                "backend": args.backend,
                "server_pid": server_pid,
                "elapsed_s": time.monotonic() - started,
            },
        )
        policy, factory = build_disposable_submitted_policy(GAME, proposer)
        session["factory_receipt"] = factory
        progress = ProgressWriter(
            checkpoint.parent / "arc_run_game.json",
            game=GAME,
            game_index=0,
            games_planned=1,
            policy=policy,
        )
        run_start = time.monotonic()
        _progress(5, "benchmark_start", backend=args.backend, game=GAME, actions=ACTION_BUDGET)
        _progress(5, "generation_start", backend=args.backend, operation="scored_policy_episode")
        run_row = run_game(GAME, policy, budget=ACTION_BUDGET, progress=progress)
        run_end = time.monotonic()
        _progress(5, "generation_end", backend=args.backend, elapsed_s=run_end - run_start)
        _progress(5, "benchmark_end", backend=args.backend, actions=run_row.get("actions"))
        spans.append(
            {
                "backend": args.backend,
                "phase": "generation_episode",
                "duration_s": run_end - run_start,
            }
        )
        recorder = policy.action_provenance()
        action_rows = deepcopy(recorder.rows if recorder is not None else [])
        if recorder is not None:
            recorder.flush()
        induction_rows = _attempt_rows(policy, raw_dir)
        session.update(
            {"run_row": run_row, "action_rows": action_rows, "induction_rows": induction_rows}
        )
        session["model_invoked"] = bool(capture.rows)
        atomic_write(raw_dir / "run_game_row.json", run_row)
        atomic_write(raw_dir / "action_rows.json", {"rows": action_rows})
        atomic_write(raw_dir / "induction_rows.json", {"rows": induction_rows})
    except Exception as exc:
        session["error"] = f"{type(exc).__name__}: {exc}"[:500]
        _progress(5, "backend_error", backend=args.backend, error=session["error"])
    finally:
        capture.restore()
        session["model_invoked"] = bool(capture.rows)
        _progress(6, "model_unload_start", backend=args.backend, pid=getattr(server, "pid", None))
        if args.backend == "llamacpp" and proposer is not None:
            proposer.stop()
        elif server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
        _progress(6, "model_unload_end", backend=args.backend)
        session["duration_s"] = time.monotonic() - started
        session["completions"] = capture.rows
        _enrich_request_manifest(session)
        atomic_write(session_path, session)
        atomic_write(
            checkpoint,
            {
                "stage": "terminal",
                "model_loaded": bool(session.get("model_loaded")),
                "terminal": True,
                "backend": args.backend,
                "elapsed_s": session["duration_s"],
            },
        )
    return 0


def _model_paths() -> tuple[Path | None, Path | None, Path | None]:  # pragma: no cover
    from carnot.inference.sota_models import cached_current_model

    current = cached_current_model(preferred_quant=PRIMARY_QUANTIZATION)
    primary = Path(str(current["model_path"])) if current is not None else None
    snapshots = Path.home() / ".cache/huggingface/hub/models--RedHatAI--Qwen3.8-27B-INT4/snapshots"
    comparator = next(
        (path for path in snapshots.glob("*") if (path / "config.json").is_file()), None
    )
    runtime = REPO_ROOT / ".venv-vllm-trial/bin/python"
    return primary, comparator, runtime if runtime.is_file() else None


def collect_preconditions(
    root: Path,
) -> tuple[list[JsonDict], JsonDict, JsonDict]:  # pragma: no cover
    """Authenticate required bytes and resolve both live backends."""

    checks: list[JsonDict] = []
    hashes: JsonDict = {}
    for relative in REQUIRED_SOURCE_PATHS:
        path = root / relative
        regular = path.is_file()
        checks.append(gate_check("required_source", str(path), relative.as_posix(), True, regular))
        if regular:
            hashes[relative.as_posix()] = sha256_file(path)
    observed_contract = task_contract(root / ROADMAP_PATH)
    checks.append(
        gate_check(
            "exact_task_contract",
            ROADMAP_PATH.as_posix(),
            TASK_ID,
            EXPECTED_TASK_CONTRACT,
            observed_contract,
        )
    )
    prior_path = root / "results/experiment_7221_v636_arc_session.json"
    try:
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        prior = {}
    prior_quarantined = is_quarantined(prior)
    prior_verdict = (
        unwrap_evidence_value(prior.get("honest_verdict"))
        if prior and not prior_quarantined
        else "not_consumed"
    )
    checks.append(
        gate_check(
            "prior_failure_authenticated_not_quarantined",
            "results/experiment_7221_v636_arc_session.json",
            "honest_verdict",
            {
                "quarantined": False,
                "value": "complete_null_no_observed_missing_tool_demand_cumulative_n_10",
            },
            {"quarantined": prior_quarantined, "value": prior_verdict},
        )
    )
    primary, comparator, runtime = _model_paths()
    checks.append(gate_check("current_gguf", PRIMARY_MODEL_ID, "Q4_K_M", True, bool(primary)))
    server_candidates = (
        os.environ.get("CARNOT_LLAMA_SERVER"),
        str(Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"),
    )
    server = next(
        (candidate for candidate in server_candidates if candidate and Path(candidate).is_file()),
        None,
    )
    checks.append(gate_check("llama_server", "native_runtime", "binary", True, bool(server)))
    gpus = _gpu_inventory()
    idle_gpus = [
        row for row in gpus if not row["compute_apps"] and int(row["free_memory_mb"]) >= 21_000
    ]
    checks.append(
        gate_check("cuda_headroom", "nvidia-smi", "idle_24GB_cards", True, len(idle_gpus) >= 2)
    )
    if primary:
        _progress(0, "benchmark_start", operation="primary_model_hash", path=str(primary))
        primary_hash = _model_digest(primary)
        hashes[str(primary)] = primary_hash
        _progress(0, "benchmark_end", operation="primary_model_hash", digest=primary_hash)
    else:
        primary_hash = None
    if comparator and runtime:
        _progress(0, "benchmark_start", operation="comparator_model_hash", path=str(comparator))
        comparator_hash = _model_digest(comparator)
        hashes[str(comparator)] = comparator_hash
        _progress(0, "benchmark_end", operation="comparator_model_hash", digest=comparator_hash)
    else:
        comparator_hash = None
    return (
        checks,
        hashes,
        {
            "primary": str(primary) if primary else None,
            "comparator": str(comparator) if comparator else None,
            "comparator_runtime": str(runtime) if runtime else None,
            "llama_server": server,
            "gpus": idle_gpus,
            "primary_hash": primary_hash,
            "comparator_hash": comparator_hash,
            "comparator_available": bool(comparator and runtime),
        },
    )


def _owned_gpu_samples(index: int, group: int) -> JsonDict:  # pragma: no cover
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    owned: list[JsonDict] = []
    for line in query.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        pid = int(parts[0])
        try:
            owned_group = os.getpgid(pid) == group
        except ProcessLookupError:
            owned_group = False
        if owned_group:
            owned.append({"pid": pid, "gpu_uuid": parts[1], "used_memory_mb": int(parts[2])})
    return {"gpu_index": index, "owned_compute_apps": owned, "sampled_at_utc": _iso_now()}


def _run_backend(
    root: Path,
    *,
    backend: str,
    model_path: Path,
    python: Path,
    gpu: Mapping[str, Any],
    raw_dir: Path,
    checkpoint_dir: Path,
    llama_server: str | None,
    model_hash: str,
    runtime_cap_s: float,
) -> JsonDict:  # pragma: no cover
    from carnot.gpu_lease_phase_journal import GpuLease

    backend_raw = raw_dir / backend
    backend_raw.mkdir(parents=True, exist_ok=True)
    session_path = backend_raw / "session.json"
    progress_path = checkpoint_dir / backend / "progress.json"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    lease = GpuLease.acquire(
        runtime_dir=checkpoint_dir / "gpu_leases",
        task_id=f"{TASK_ID}:{backend}",
        device_uuid=str(gpu["uuid"]),
        expected_model=str(model_path),
        vram_before_mb=int(gpu["total_memory_mb"]) - int(gpu["free_memory_mb"]),
        ttl_s=90,
    )
    lease.transition("admitted")
    lease.transition("loading")
    port = _free_port()
    command = [
        str(python),
        "-u",
        str(root / WRAPPER_PATH),
        "--role",
        "backend-session",
        "--backend",
        backend,
        "--date",
        RUN_DATE,
        "--model-path",
        str(model_path),
        "--gpu-index",
        str(gpu["index"]),
        "--port",
        str(port),
        "--raw-dir",
        str(backend_raw),
        "--checkpoint-path",
        str(progress_path),
        "--session-output",
        str(session_path),
    ]
    env = session_environment(
        os.environ,
        backend=backend,
        model_path=model_path,
        gpu_index=int(gpu["index"]),
        port=port,
        raw_dir=backend_raw,
    )
    if llama_server:
        env["CARNOT_LLAMA_SERVER"] = llama_server
    env["CARNOT_EXP7234_MODEL_HASH"] = model_hash
    _progress(4, "subprocess_start", backend=backend, command=command)
    process = subprocess.Popen(command, cwd=root, env=env, start_new_session=True)
    started = time.monotonic()
    next_beat = started
    samples: list[JsonDict] = []
    resident = False
    startup_timed_out = False
    while process.poll() is None and time.monotonic() - started < runtime_cap_s:
        now = time.monotonic()
        progress = {}
        if progress_path.is_file():
            try:
                progress = json.loads(progress_path.read_text())
            except (OSError, json.JSONDecodeError):
                progress = {}
        if progress.get("model_loaded") and not resident:
            lease.transition("resident", vram_mb=0)
            lease.transition("inferencing")
            resident = True
        if not resident and now - started > STARTUP_TIMEOUT_S:
            startup_timed_out = True
            break
        if now >= next_beat:
            lease.heartbeat()
            sample = _owned_gpu_samples(int(gpu["index"]), process.pid)
            samples.append(sample)
            _progress(
                4,
                "subprocess_heartbeat",
                backend=backend,
                elapsed_s=round(now - started, 1),
                progress=progress,
                owned_gpu_processes=len(sample["owned_compute_apps"]),
            )
            next_beat = now + 45
        time.sleep(2)
    timed_out = process.poll() is None
    signals: list[str] = []
    if timed_out:
        os.killpg(process.pid, signal.SIGTERM)
        signals.append("SIGTERM:owned_process_group")
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            signals.append("SIGKILL:owned_process_group")
            process.wait(timeout=10)
    _progress(
        4,
        "subprocess_end",
        backend=backend,
        returncode=process.returncode,
        timed_out=timed_out,
    )
    try:
        session = json.loads(session_path.read_text())
    except (OSError, json.JSONDecodeError):
        session = {
            "backend": backend,
            "model_invoked": resident,
            "model_spec": {
                "hf_id": PRIMARY_MODEL_ID if backend == "llamacpp" else COMPARATOR_MODEL_ID,
                "quantization": PRIMARY_QUANTIZATION if backend == "llamacpp" else "INT4",
                "model_path": str(model_path),
            },
            "completions": [],
            "action_rows": [],
            "induction_rows": [],
            "run_row": {},
            "runner_receipt": {},
            "phase_spans": [],
            "error": "startup_timeout" if startup_timed_out else "backend_timeout_or_child_failure",
        }
    if resident:
        lease.transition("unloading")
        lease.transition(
            "validating", vram_mb=0, exit_code=int(process.returncode or 0), unload_observed=True
        )
        lease.transition("terminal_complete")
    else:
        lease.transition("terminal_blocked")
    release = lease.release()
    owned = [app for sample in samples for app in sample["owned_compute_apps"]]
    session["gpu_receipts"] = {
        "gpu_uuid": gpu["uuid"],
        "gpu_index": gpu["index"],
        "task_linked_cuda_execution": bool(owned),
        "provenance_ok": bool(owned),
        "samples": samples,
        "lease_owner": lease.owner_receipt(),
        "lease_release": release,
        "signals_sent": signals,
    }
    session["model_invoked"] = bool(session.get("model_invoked") and owned)
    session["timed_out"] = timed_out
    atomic_write(session_path, session)
    return reduce_backend_session(session)


def _external_block_row(backend: str, error: str) -> JsonDict:  # pragma: no cover
    return {
        "backend": backend,
        "disposition": "blocked_external_absence",
        "model_invoked": False,
        "model_spec": None,
        "transport_completed": False,
        "semantic_usable": False,
        "tool_dispatches": 0,
        "engine_writes": 0,
        "policy_engine_consumptions": 0,
        "valid_engine_count": 0,
        "non_identity_prediction_count": 0,
        "held_future_prediction_accuracy": [],
        "trust_acceptance_count": 0,
        "model_planned_actions": 0,
        "actions": 0,
        "levels": 0,
        "progress": 0,
        "total_cost": 0,
        "action_rows": [],
        "induction_rows": [],
        "raw_request_manifest": [],
        "phase_spans": [],
        "runner_receipt": {},
        "gpu_receipts": {},
        "factory_receipt": {},
        "error": error,
    }


def run_experiment(args: argparse.Namespace) -> JsonDict:  # pragma: no cover - live task.
    """Run preflight, sequential backends, and atomic terminal publication."""

    started = time.monotonic()
    started_utc = _iso_now()
    root = REPO_ROOT
    result = args.result_path if args.result_path.is_absolute() else root / args.result_path
    checkpoint = (
        args.checkpoint_path if args.checkpoint_path.is_absolute() else root / args.checkpoint_path
    )
    raw_dir = args.raw_dir if args.raw_dir.is_absolute() else root / args.raw_dir
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    _progress(0, "phase_start", name="preconditions")
    checks, hashes, resources = collect_preconditions(root)
    selected, selection = select_game_from_rotation(
        GAME_ROTATION, credited_task_targets=PRIOR_TASK_TARGETS
    )
    if selected != GAME:
        checks.append(gate_check("frozen_game", TASK_ID, "game", GAME, selected))
    atomic_write(
        checkpoint,
        {"status": "running", "stage": "preconditions", "checks": checks, "terminal": False},
    )
    _progress(0, "phase_end", name="preconditions", passed=all(row["passed"] for row in checks))
    backends: list[JsonDict] = []
    if all(row["passed"] for row in checks):
        _progress(1, "phase_start", name="llamacpp_backend")
        backends.append(
            _run_backend(
                root,
                backend="llamacpp",
                model_path=Path(resources["primary"]),
                python=Path(sys.executable),
                gpu=resources["gpus"][0],
                raw_dir=raw_dir,
                checkpoint_dir=checkpoint.parent,
                llama_server=resources["llama_server"],
                model_hash=resources["primary_hash"],
                runtime_cap_s=min(BACKEND_TIMEOUT_S, TOTAL_RUNTIME_CAP_S),
            )
        )
        _progress(1, "phase_end", name="llamacpp_backend")
        remaining = TOTAL_RUNTIME_CAP_S - (time.monotonic() - started)
        _progress(2, "phase_start", name="vllm_backend", remaining_s=round(remaining, 1))
        if not resources["comparator_available"]:
            backends.append(_external_block_row("vllm", "comparator_runtime_absent"))
        elif remaining < STARTUP_TIMEOUT_S:
            backends.append(_external_block_row("vllm", "total_runtime_cap_prevents_start"))
        else:
            backends.append(
                _run_backend(
                    root,
                    backend="vllm",
                    model_path=Path(resources["comparator"]),
                    python=Path(resources["comparator_runtime"]),
                    gpu=resources["gpus"][1],
                    raw_dir=raw_dir,
                    checkpoint_dir=checkpoint.parent,
                    llama_server=None,
                    model_hash=resources["comparator_hash"],
                    runtime_cap_s=min(BACKEND_TIMEOUT_S, remaining),
                )
            )
        _progress(2, "phase_end", name="vllm_backend")
    for path in raw_dir.rglob("*"):
        if path.is_file():
            hashes[str(path.relative_to(root))] = sha256_file(path)
    _progress(7, "phase_start", name="terminal_validation_and_atomic_write")
    artifact = build_terminal_artifact(
        run_date=args.date,
        duration_s=time.monotonic() - started,
        started_at_utc=started_utc,
        ended_at_utc=_iso_now(),
        checks=checks,
        source_hashes=hashes,
        selection_receipt=selection,
        backend_rows=backends,
        validation_receipts=[],
    )
    _progress(7, "validation_start", operation="cold_artifact_validation")
    errors = validate_artifact(artifact)
    _progress(7, "validation_end", operation="cold_artifact_validation", errors=errors)
    if errors:
        raise ValueError("terminal_artifact_invalid:" + ",".join(errors))
    _progress(7, "artifact_write_start", path=str(result))
    atomic_write(result, artifact)
    _progress(7, "artifact_write_end", path=str(result))
    _progress(7, "phase_end", name="terminal_validation_and_atomic_write")
    return artifact


def parse_args(  # pragma: no cover - thin CLI parsing.
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse driver, isolated backend, and validation modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("driver", "backend-session"), default="driver")
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--backend", choices=("llamacpp", "vllm"))
    parser.add_argument("--model-path")
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--checkpoint-path", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--session-output", type=Path)
    parser.add_argument("--result-path", type=Path, default=RESULT_PATH)
    parser.add_argument("--validate", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI dispatch.
    """Run the dry run, one isolated backend, or a cold validation."""

    _progress(0, "phase_start", name="entrypoint")
    args = parse_args(argv)
    if args.validate is not None:
        return int(bool(validate_artifact(args.validate)))
    if args.role == "backend-session":  # pragma: no cover - live CUDA.
        return run_backend_child(args)
    artifact = run_experiment(args)  # pragma: no cover - live CUDA.
    print(
        json.dumps(
            {
                "artifact": str(args.result_path),
                "status": artifact["status"],
                "verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

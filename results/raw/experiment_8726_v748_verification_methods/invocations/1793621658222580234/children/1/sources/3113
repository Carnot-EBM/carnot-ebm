"""Observe one live E3 goal assertion against fresh ARC SDK progress.

REQ-REPORT-7667. Public-game observations are one case study, not hidden
leaderboard evidence or a counterfactual solve-rate experiment.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

from carnot.agentic.arc_goal_confirmation import shadow_goal_only_decision
from carnot.experiment_7653_v667_arc_live_generalization import (
    canonical_hash,
    collect_preconditions,
    gate_check,
)
from carnot.reporting.current_work_receipt import atomic_json, sha256_file

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
RESULT = Path("results/experiment_7667_v668_arc_live_goal_observation.json")
RAW = Path("results/raw/experiment_7667_v668_arc_live_goal_observation")
MODULE = Path("python/carnot/experiment_7667_v668_arc_live_goal_observation.py")
WRAPPER = Path("scripts/experiments/experiment_7667_v668_arc_live_goal_observation.py")
TEST = Path("tests/python/test_experiment_7667_v668_arc_live_goal_observation.py")
GUARD = Path("python/carnot/agentic/arc_goal_confirmation.py")
SALT = "v668-live-goal-observation-20260925"
SEED = 7_667_668


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Keep each phase and long wait visible to the operator."""
    print(
        f"[exp7667] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        f" {details}",
        flush=True,
    )


def select_game(roster: Sequence[str], *, salt: str = SALT) -> str:
    """Select one distinct SDK identity without consulting registry outcomes."""
    identities = set(roster)
    if not identities:
        raise ValueError("sdk_roster_empty")
    return min(identities, key=lambda game: (canonical_hash([salt, game]), game))


def raw_for_run(base: Path, run_id: str) -> Path:
    """Keep each attempt's request and action IDs in its own durable directory."""
    if not run_id or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for char in run_id):
        raise ValueError("invalid_run_id")
    return base / run_id


def isolated_session_environment(
    base: Mapping[str, str], *, gpu_index: int, port: int, raw_dir: Path
) -> dict[str, str]:
    """Keep the inherited boundary ledger with this attempt's other raw evidence."""
    from carnot import experiment_7471_v654_arc_seam_observation as live
    from carnot.agentic.arc_inference_boundary import BOUNDARY_LEDGER_ENV

    environment = live.session_environment(base, gpu_index=gpu_index, port=port, raw_dir=raw_dir)
    environment[BOUNDARY_LEDGER_ENV] = str(raw_dir / "current_invocation_events.jsonl")
    return environment


def bounded_induction_call(call: Callable[[], Any], limit_s: float) -> Any:
    """Enforce one wall limit while preserving the enclosing episode alarm."""
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_remaining, previous_interval = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()

    def expired(_signum: int, _frame: Any) -> None:
        raise TimeoutError("induction_wall_limit")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(
        signal.ITIMER_REAL,
        min(limit_s, previous_remaining) if previous_remaining > 0 else limit_s,
    )
    try:
        return call()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_remaining > 0:
            remaining = previous_remaining - (time.monotonic() - started)
            if remaining > 0:
                signal.setitimer(signal.ITIMER_REAL, remaining, previous_interval)


def make_schedule(game: str) -> Json:
    """Freeze the single independent game and physical episode limits."""
    return {
        "episode_id": f"{game}:goal_confirm:{SEED}",
        "game": game,
        "arm": "goal_confirmation_on_with_old_shadow",
        "seed": SEED,
        "max_actions": 256,
        "max_inductions": 1,
        "max_induction_s": 2400,
        "max_engine_calls": 20000,
        "max_seconds": 3000,
        "adapter_withheld": True,
    }


def reduce_rows(rows: Sequence[Mapping[str, Any]], schedule: Mapping[str, Any]) -> Json:
    """Rebuild guard opportunities from the one raw physical episode."""
    if len(rows) != 1 or rows[0].get("episode_id") != schedule["episode_id"]:
        raise ValueError("one_episode_accounting")
    row = rows[0]
    inductions = row.get("induction_events") or []
    goals = row.get("goal_observations") or []
    comparisons = [
        {
            **shadow_goal_only_decision(bool(goal.get("predicted_goal")), str(goal.get("status"))),
            "status": goal.get("status"),
            "frames_seen": goal.get("frames_seen"),
        }
        for goal in goals
    ]
    counts = {
        "attempted_inductions": len(inductions),
        "accepted_inductions": sum(event.get("decision") == "accepted" for event in inductions),
        "rejected_inductions": sum(event.get("decision") == "rejected" for event in inductions),
        "executed_plans": int(bool(row.get("selected_plan") and row.get("actions"))),
        "predicted_goals": sum(bool(goal.get("predicted_goal")) for goal in goals),
        "settled_observations": sum(
            goal.get("status") in {"confirmed", "contradiction"} for goal in goals
        ),
        "confirmed_contradictions": sum(goal.get("status") == "contradiction" for goal in goals),
    }
    return {
        "opportunity_counts": counts,
        "decision_comparisons": comparisons,
        "guard_opportunity": "observed"
        if counts["accepted_inductions"] and counts["predicted_goals"]
        else "insufficient",
        "counterfactual_solve_rate_claim": False,
        "sample_size_budget": {
            "intended_independent_groups": 1,
            "observed_independent_groups": 1,
            "eligible_independent_groups": int(row.get("exclusion") is None),
            "excluded_independent_groups": int(row.get("exclusion") is not None),
            "censored_independent_groups": int(bool(row.get("censored"))),
            "prior_exposure": "public_registry_game",
            "claim_limit": "single_public_case_study",
        },
    }


def cold_reduce(path: Path) -> Json:
    """Read a persisted candidate and reject any forged producer reduction."""
    value = json.loads(path.read_text(encoding="utf-8"))
    reduction = reduce_rows(value["rows"], value["schedule"])
    if value.get("reduction") != reduction:
        raise ValueError("reduction_mismatch")
    return reduction


def _read_jsonl(path: Path) -> list[Json]:
    """Read only complete checkpoint lines after the owned child stops."""
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _child_measure(root: Path, schedule: Mapping[str, Any], raw: Path) -> int:  # pragma: no cover
    """Own the lease, server, and one real SDK episode in a killable child."""
    from carnot import experiment_7471_v654_arc_seam_observation as live
    from carnot.agentic.arc_executable_world_model import (
        LocalGGUFProposer,
        _induce_max_tokens_default,
    )
    from carnot.agentic.arc_goal_confirmation import GoalConfirmation
    from carnot.experiment_7581_v662_arc_bounded_canary import (
        _observed_offload_layers,
        _owned_vram_mb,
    )
    from carnot.experiment_7630_v666_cuda_ownership import (
        ProcessRegistry,
        _current_inventory,
        recheck_before_launch,
        select_owned_capacity,
    )
    from carnot.gpu_lease_phase_journal import GpuLease, LeaseBusy
    from carnot.inference.sota_models import cached_current_model
    from carnot.experiment_7431_v651_arc_live_sentinel import _free_port

    started = time.monotonic()
    raw.mkdir(parents=True, exist_ok=True)
    owner = ProcessRegistry.current()
    owner.task_id = "experiment_7667_v668_arc_live_goal_observation"
    gpu, _ = select_owned_capacity(_current_inventory(), owner)
    if gpu is None:
        atomic_json(raw / "child_result.json", {"error": "exclusive_cuda_capacity"})
        return 2
    model = cached_current_model()
    if model is None:
        atomic_json(raw / "child_result.json", {"error": "cached_mandated_model"})
        return 2
    model_path = Path(str(model["model_path"])).resolve()
    server = (Path.home() / ".cache/llama.cpp-master/build/bin/llama-server").resolve()
    progress(started, "lease", "before", uuid=gpu["uuid"])
    lease = None
    wait_started = time.monotonic()
    while lease is None:
        try:
            lease = GpuLease.acquire(
                runtime_dir=Path("/tmp/carnot-gpu-leases"),
                task_id=owner.task_id,
                device_uuid=str(gpu["uuid"]),
                expected_model=str(model_path),
                vram_before_mb=int(gpu.get("memory_used_mb") or 0),
                ttl_s=120,
            )
        except LeaseBusy:
            if time.monotonic() - wait_started >= 180:
                atomic_json(raw / "child_result.json", {"error": "lease_wait_180s"})
                return 2
            progress(started, "lease", "wait")
            time.sleep(15)
    progress(started, "lease", "after", lease_id=lease.lease_id)
    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.wait(45):
            lease.heartbeat()
            progress(started, "owned_child", "heartbeat")

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    proposer = None
    capture = None
    original_env = dict(os.environ)
    runtime: Json = {"load_attempted": False, "model_loaded": False, "gpu_uuid": gpu["uuid"]}
    try:
        check = recheck_before_launch(
            str(gpu["uuid"]), [_current_inventory(), _current_inventory()], owner
        )
        runtime["prelaunch_recheck"] = check
        if not check["passed"]:
            raise RuntimeError("foreign_or_capacity_recheck_failed")
        port = _free_port()
        os.environ.update(
            isolated_session_environment(
                os.environ, gpu_index=int(gpu["index"]), port=port, raw_dir=raw
            )
        )
        for name in ("CARNOT_ARC_INDUCE_MAX_TOKENS", "CARNOT_ARC_INDUCE_N_CTX"):
            os.environ.pop(name, None)
        os.environ.update(
            {
                "CARNOT_ARC_GOAL_CONFIRMATION": "1",
                "CARNOT_ARC_INDUCE_TIMEOUT": "2400",
                "CARNOT_ARC_MAX_REFINEMENT_ROUNDS": "1",
                "CARNOT_ARC_INDUCE_TOOL_TURNS": "1",
                "CARNOT_ARC_GENERATOR_REQUIRE_CUDA": "1",
                "CARNOT_ARC_GGUF_PATH": str(model_path),
                "CARNOT_LLAMA_SERVER": str(server),
                "CARNOT_ARC_SERVER_LOG_DIR": str(raw / "server_logs"),
            }
        )
        budget = _induce_max_tokens_default()
        runtime["actual_token_budget"] = budget
        lease.transition("admitted")
        lease.transition("loading")
        progress(started, "model_load", "before", model=MODEL_ID, token_budget=budget)
        runtime["load_attempted"] = True
        atomic_json(raw / "runtime_checkpoint.json", runtime)
        proposer = LocalGGUFProposer(
            repo_substr="Qwen3.8-27B",
            model_path=str(model_path),
            port=port,
            mtp=False,
            kv_quant="q8_0",
            use_chat_template=True,
            n_gpu_layers=999,
            timeout=2400,
            tries=1,
        )
        proposer.model_repository = MODEL_ID
        healthy = proposer._ensure_server()
        server_pid = getattr(getattr(proposer, "_proc", None), "pid", None)
        runtime.update(
            {
                "model_loaded": bool(healthy),
                "server_pid": server_pid,
                "server_command": list(getattr(proposer, "last_launch_argv", ()) or ()),
                "offload_layers": _observed_offload_layers(
                    Path(proposer._stderr_log_path) if proposer._stderr_log_path else None
                ),
                "owned_server_vram_mb": _owned_vram_mb(server_pid),
                "model_sha256": sha256_file(model_path),
                "runtime_sha256": sha256_file(server),
            }
        )
        progress(started, "model_load", "after", healthy=healthy, server_pid=server_pid)
        atomic_json(raw / "runtime_checkpoint.json", runtime)
        if not healthy or not runtime["owned_server_vram_mb"]:
            raise RuntimeError("current_server_offload_unverified")
        lease.transition("resident", vram_mb=int(runtime["owned_server_vram_mb"]))
        lease.transition("inferencing")
        capture = live.live_support.DurableRequestCapture(
            raw, raw / "runtime_events.jsonl", max_new_tokens=budget
        )
        capture.install()
        row = _one_episode(schedule, proposer, capture, raw, started)
        atomic_json(raw / "child_result.json", {"row": row, "runtime": runtime})
        return 0
    except BaseException as exc:
        runtime["error"] = f"{type(exc).__name__}: {exc}"[:500]
        progress(started, "child", "error", error=runtime["error"])
        atomic_json(raw / "child_result.json", {"error": runtime["error"], "runtime": runtime})
        return 2
    finally:
        if capture is not None:
            capture.restore()
        progress(started, "model_unload", "before")
        if proposer is not None:
            proposer.stop()
        progress(started, "model_unload", "after")
        os.environ.clear()
        os.environ.update(original_env)
        stop.set()
        thread.join(timeout=1)
        if runtime["model_loaded"]:
            lease.transition("unloading")
            lease.transition("validating", vram_mb=0, exit_code=0, unload_observed=True)
            lease.transition("terminal_complete")
        else:
            lease.transition("terminal_blocked")
        runtime["lease_release"] = lease.release()
        result_path = raw / "child_result.json"
        if result_path.is_file():
            final_result = json.loads(result_path.read_text())
            final_result["runtime"] = runtime
            atomic_json(result_path, final_result)


def _one_episode(
    schedule: Mapping[str, Any], proposer: Any, capture: Any, raw: Path, started: float
) -> Json:  # pragma: no cover - live SDK boundary.
    """Observe the shipped E3 policy while preserving its decisions."""
    from functools import wraps
    from carnot import experiment_7471_v654_arc_seam_observation as live
    from carnot.agentic import arc_executable_world_model as world
    from carnot.agentic.arc_competition_agent import E3AgentPolicy
    from carnot.agentic.arc_goal_confirmation import GoalConfirmation

    goal_path = raw / "goal_events.jsonl"
    original_arm = GoalConfirmation.arm
    original_observe = GoalConfirmation.observe
    original_induce = E3AgentPolicy._induce_and_plan
    original_plan = world.plan_in_model
    previous = (live.ACTION_LIMIT, live.EPISODE_LIMIT_S, live.REQUEST_LIMIT)
    live.ACTION_LIMIT, live.EPISODE_LIMIT_S, live.REQUEST_LIMIT = 256, 3000.0, 4
    induction_count = 0
    engine_calls = 0
    plans: list[Json] = []

    @wraps(original_arm)
    def arm(self: Any, frame: Any, **kwargs: Any) -> None:
        original_arm(self, frame, **kwargs)
        live._append_jsonl(
            goal_path,
            {
                "event": "arm",
                "predicted_goal": kwargs["predicted_goal"],
                "frames_seen": kwargs["frames_seen"],
                "level_before": kwargs["level"],
                "frame_hash": canonical_hash(str(frame)),
            },
        )

    @wraps(original_observe)
    def observe(self: Any, frame: Any, **kwargs: Any) -> Json:
        receipt = original_observe(self, frame, **kwargs)
        if receipt.get("reason") != "no_executed_endpoint":
            live._append_jsonl(
                goal_path,
                {
                    "event": "observe",
                    **receipt,
                    "frame_hash": canonical_hash(str(frame)),
                    "monotonic_ns": time.monotonic_ns(),
                },
            )
        return receipt

    @wraps(original_induce)
    def induce(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal induction_count
        induction_count += 1
        if induction_count > 1:
            return None
        progress(started, "induction", "before", attempt=induction_count)
        try:
            return bounded_induction_call(lambda: original_induce(self, *args, **kwargs), 2400.0)
        finally:
            progress(started, "induction", "after", attempt=induction_count)

    @wraps(original_plan)
    def plan(engine: Any, goal: Any, grid: Any, **kwargs: Any) -> Any:
        nonlocal engine_calls

        def counted(*args: Any, **inner: Any) -> Any:
            nonlocal engine_calls
            if engine_calls >= 20000:
                raise RuntimeError("planner_engine_call_budget_exhausted")
            engine_calls += 1
            return engine(*args, **inner)

        progress(started, "planner", "before", engine_calls=engine_calls)
        try:
            result = original_plan(counted, goal, grid, **kwargs)
        except RuntimeError as exc:
            if str(exc) != "planner_engine_call_budget_exhausted":
                raise
            result = None
        plans.append({"plan": deepcopy(result), "engine_calls": engine_calls})
        progress(started, "planner", "after", engine_calls=engine_calls)
        return result

    GoalConfirmation.arm = arm
    GoalConfirmation.observe = observe
    E3AgentPolicy._induce_and_plan = induce
    world.plan_in_model = plan
    try:
        progress(started, "episode", "before_benchmark", episode_id=schedule["episode_id"])
        source = live._run_policy_episode(
            schedule, proposer, capture, raw / "runtime_events.jsonl", raw / "actions.jsonl"
        )
        progress(started, "episode", "after_benchmark", actions=source["action_count"])
    finally:
        GoalConfirmation.arm = original_arm
        GoalConfirmation.observe = original_observe
        E3AgentPolicy._induce_and_plan = original_induce
        world.plan_in_model = original_plan
        live.ACTION_LIMIT, live.EPISODE_LIMIT_S, live.REQUEST_LIMIT = previous
    seams = live.read_jsonl(Path(source["seam_event_path"]))
    hypothesis = [event for event in seams if event.get("seam") == "hypothesis_gate"]
    goal_events = _read_jsonl(goal_path)
    goals = [event for event in goal_events if event["event"] == "observe"]
    requests = source["server_request_rows"]
    tokens = 0
    for request in requests:
        path = Path(str(request.get("response_path") or ""))
        if path.is_file():
            response = json.loads(path.read_text())
            usage = response.get("usage") or {}
            tokens += int(usage.get("completion_tokens") or response.get("tokens_predicted") or 0)
    accepted = any(event.get("gate_decision") == "accept" for event in hypothesis)
    attempted = any(event.get("event") == "stage_start" for event in hypothesis)
    induction_timeout = bool(source["error"] and "induction_wall_limit" in source["error"])
    return {
        **dict(schedule),
        "start_level": source["start_level"],
        "peak_level": source["peak_level"],
        "terminal_level": source["terminal_level"],
        "actions_used": source["action_count"],
        "actions": source["action_rows"],
        "request_rows": requests,
        "current_output_tokens": tokens,
        "induction_events": [
            {"decision": "accepted" if accepted else "rejected", "seam_events": hypothesis}
        ]
        if attempted
        else [],
        "selected_plan": next((item["plan"] for item in plans if item["plan"]), None),
        "planner_rows": plans,
        "engine_calls": engine_calls,
        "goal_arms": [event for event in goal_events if event["event"] == "arm"],
        "goal_observations": goals,
        "supervisor_events": [
            event for event in seams if event.get("seam") == "supervisor_arm_selection"
        ],
        "recovery_events": [event for event in seams if event.get("event") == "redirect"],
        "censored": source["disposition"].startswith("censored") or induction_timeout,
        "censor_reason": "censored_induction_wall_limit"
        if induction_timeout
        else source["disposition"]
        if source["disposition"].startswith("censored")
        else None,
        "exclusion": None,
        "raw_provenance": "live_agent_self_discovery",
        "action_log": str(raw / "actions.jsonl"),
        "goal_log": str(goal_path),
        "seam_log": source["seam_event_path"],
        "official_reproduction": source["trace_reproduction"],
        "policy_entry": source["policy_entry"],
        "error": source["error"],
    }


def _gate(passed: bool, principle: str, **operands: Any) -> Json:
    """Keep each gate's measured operands beside its rule."""
    return {"passed": bool(passed), "principle": principle, "measured_operands": operands}


def _artifact(
    checks: Sequence[Mapping[str, Any]],
    hashes: Json,
    schedule: Json,
    rows: list[Json],
    runtime: Json,
    duration_s: float,
) -> Json:
    """Distinguish blocked resources, measured nulls, and invalid receipts."""
    failed = [dict(check) for check in checks if not check["passed"]]
    invoked = bool(runtime.get("load_attempted"))
    requests = [request for row in rows for request in row.get("request_rows", [])]
    generations = sum(bool(request.get("request_dispatched")) for request in requests)
    tokens = sum(int(row.get("current_output_tokens") or 0) for row in rows)
    reduction = reduce_rows(rows, schedule) if len(rows) == 1 else None
    expected_censors = {"censored_timeout", "censored_induction_wall_limit"}
    episode_error = any(
        row.get("error") and row.get("censor_reason") not in expected_censors for row in rows
    )
    valid_live = bool(
        reduction
        and not runtime.get("error")
        and not episode_error
        and generations
        and duration_s >= 60
    )
    if failed:
        verdict, verdict_class = f"complete_blocked_{failed[0]['check']}", "blocked"
    elif runtime.get("error") or episode_error or not rows or not generations:
        verdict, verdict_class = "complete_disqualified_live_execution", "disqualified"
    else:
        verdict, verdict_class = "complete_null_public_goal_opportunity", "null"
    counts = (
        reduction["opportunity_counts"]
        if reduction
        else {
            "attempted_inductions": 0,
            "accepted_inductions": 0,
            "rejected_inductions": 0,
            "executed_plans": 0,
            "predicted_goals": 0,
            "settled_observations": 0,
            "confirmed_contradictions": 0,
        }
    )
    gates = {
        "validity": _gate(
            valid_live,
            "One real SDK episode and current generation with valid receipts.",
            episodes=len(rows),
            generations=generations,
            runtime_error=runtime.get("error"),
        ),
        "readiness": _gate(False, "All frozen scoped and terminal checks must pass.", checks=0),
        "coverage": _gate(
            bool(rows), "The one selected game must have a durable row.", rows=len(rows)
        ),
        "probability_benefit": _gate(
            False, "One public game cannot estimate hidden-game benefit.", hidden_games=0
        ),
        "utility": _gate(
            False,
            "A decision shadow does not establish counterfactual solve utility.",
            observations=counts["settled_observations"],
        ),
        "retention": _gate(
            False, "One episode cannot test retained learning.", independent_followups=0
        ),
        "freshness": _gate(
            bool(generations),
            "Only current dispatched generation establishes freshness.",
            generations=generations,
        ),
    }
    result: Json = {
        "schema": "carnot.exp7667.v668.arc_live_goal_observation.v1",
        "experiment_id": 7667,
        "milestone": "2026.09.668",
        "run_date": "20260925",
        "status": "complete",
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": False,
        "gate_check_summary": {"failed_checks": failed},
        "acceptance_gate_results": gates,
        "preconditions_checked": list(checks),
        "MODEL_SPECS": [MODEL_ID] if invoked else [],
        "model_specs": [{"hf_id": MODEL_ID}] if invoked else [],
        "planned_MODEL_SPECS": [MODEL_ID],
        "model_invoked": invoked,
        "invocation_counts": {
            "loads_attempted": int(invoked),
            "loads_completed": int(bool(runtime.get("model_loaded"))),
            "forwards": generations,
            "generations": generations,
            "tokens": tokens,
            "cancelled_generations": int(bool(runtime.get("cancelled_generation"))),
        },
        "inference_substrate": "current_owned_CUDA_local_GGUF_E3"
        if invoked
        else "host_preflight_no_model_call",
        "inference_substrate_class": "model_full_generation"
        if generations and duration_s >= 60
        else "model_bounded_generation"
        if generations and duration_s >= 10
        else "model_load_no_generation"
        if invoked
        else "no_model_load",
        "planned_inference_substrate_class": "model_full_generation",
        "execution_venue": "host",
        "execution_venue_details": {
            "host": socket.gethostname(),
            "owned_pid": os.getpid(),
            "gpu_uuid": runtime.get("gpu_uuid"),
            "server_pid": runtime.get("server_pid"),
        },
        "duration_s": duration_s,
        "phase_spans": [],
        "random_seed": {"game_order_salt": SALT, "episode_seed": SEED},
        "source_artifact_hashes": hashes,
        "schedule": schedule,
        "rows": rows,
        "reduction": reduction,
        "sample_size_budget": reduction["sample_size_budget"]
        if reduction
        else {
            "intended_independent_groups": 1,
            "observed_independent_groups": 0,
            "eligible_independent_groups": 0,
            "excluded_independent_groups": 0,
            "censored_independent_groups": 0,
            "prior_exposure": "public_registry_game",
            "claim_limit": "single_public_case_study",
        },
        "opportunity_counts": counts,
        "counterfactual_solve_rate_claim": False,
        "per_game_results": [
            {"game": schedule["game"], "row": row, "registry_credit_granted": False} for row in rows
        ],
        "solve_provenance": "live_agent_self_discovery" if rows else "no_live_attempt",
        "live_goal_observation_complete_score": int(valid_live),
        "verifier_is_oracle": False,
        "validation_receipts": [],
        "gpu_lease_receipt": runtime,
        "registry_credit_granted": False,
        "production_defaults_changed": False,
        "external_publication_authorized": False,
    }
    result["field_principles"] = {
        key: "Current raw operands and scoped validation govern this field." for key in result
    }
    result["reproducibility_checksum"] = canonical_hash(
        [
            hashes,
            schedule,
            sha256_file(ROOT / MODULE),
            sha256_file(ROOT / GUARD),
        ]
    )
    return result


def _preconditions(root: Path, started: float) -> tuple[list[Json], Json, Json]:  # pragma: no cover
    """Reuse V667's authenticated resource checks and add this run's inputs."""
    checks, hashes, context = collect_preconditions(root, started)
    for path in (MODULE, WRAPPER, TEST, GUARD):
        full = root / path
        checks.append(
            gate_check(
                f"named_input:{path}",
                upstream="declared_task_input",
                path=str(full),
                field="readable_nonempty_file",
                operator="eq",
                expected=True,
                observed=full.is_file() and full.stat().st_size > 0,
            )
        )
        if full.is_file():
            hashes["producer_files"][str(path)] = {"path": str(full), "sha256": sha256_file(full)}
    spec = root / "openspec/capabilities/research-reporting/spec.md"
    checks.append(
        gate_check(
            "driving_requirement_7667",
            upstream="research_reporting_spec",
            path=str(spec),
            field="REQ-REPORT-7667",
            operator="eq",
            expected=True,
            observed="REQ-REPORT-7667" in spec.read_text(),
        )
    )
    hashes["planned_outputs"] = [str(root / RESULT)]
    return checks, hashes, context


def _validation_commands(root: Path, private: Path) -> list[Any]:  # pragma: no cover
    """Use one frozen, serial file scope with private pytest and coverage data."""
    from carnot.reporting.experiment_7303_validation_scope import build_scoped_commands, CommandSpec

    tests = [TEST.as_posix(), "tests/python/test_experiment_7666_v668_arc_goal_confirmation.py"]
    parent = private / "pytest"
    parent.mkdir(parents=True, exist_ok=True)
    commands = build_scoped_commands(
        root,
        tests,
        [MODULE.as_posix(), GUARD.as_posix()],
        static_paths=[WRAPPER.as_posix()],
        basetemp=parent,
        coverage_file=private / ".coverage.exp7667",
    )
    commands.append(
        CommandSpec(
            "e2e_013",
            (
                str(root / ".venv/bin/pytest"),
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                f"--basetemp={parent / 'e2e013'}",
                "tests/python/test_arc_decision_telemetry.py",
                "tests/python/test_experiment_7491_e6_timed_live_profile.py",
                "tests/python/test_experiment_7531_b2_induction_gate_measurement.py",
                "tests/python/test_semif_arc_readout_eval.py",
                "-q",
            ),
            "ops/e2e-test-plan.md:E2E-013",
            900,
        )
    )
    return commands


def _terminal_commands(root: Path, candidate: Path) -> list[Any]:  # pragma: no cover
    """Run fresh reducers and both guards against one immutable candidate path."""
    from carnot.reporting.experiment_7303_validation_scope import CommandSpec

    python = str(root / ".venv/bin/python")
    wrapper = str(root / WRAPPER)
    return [
        CommandSpec(
            "cold_reduction",
            (python, "-u", wrapper, "--cold-reduce", str(candidate)),
            "exact_candidate",
            180,
        ),
        CommandSpec(
            "independent_reduction",
            (python, "-u", wrapper, "--independent-reduce", str(candidate)),
            "exact_candidate",
            180,
        ),
        CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "exact_candidate",
            180,
        ),
        CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "exact_candidate",
            180,
        ),
    ]


def _reader(path: Path) -> int:
    """Cold-read raw rows or verify a complete external-input block."""
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("verdict_class") == "blocked" and not artifact.get("rows"):
        failed = artifact.get("gate_check_summary", {}).get("failed_checks") or []
        required = {"check", "upstream", "path", "field", "operator", "expected", "observed"}
        if not failed or any(not required.issubset(check) for check in failed):
            raise ValueError("blocked_gate_operands_missing")
        print(json.dumps({"blocked_checks": len(failed)}), flush=True)
        return 0
    print(json.dumps(cold_reduce(path), sort_keys=True), flush=True)
    return 0


def run_experiment(root: Path, run_date: str, output: Path) -> int:  # pragma: no cover
    """Preflight, run one owned child, validate, and publish exact read bytes."""
    from carnot.reporting.experiment_7303_validation_scope import run_commands

    started = time.monotonic()
    progress(started, "preflight", "before", root=root)
    if root != ROOT or run_date != "20260925":
        raise ValueError("absolute_root_and_run_date_required")
    private = Path(tempfile.mkdtemp(prefix="exp7667-"))
    manifest = {
        "requirement": "REQ-REPORT-7667",
        "tests": [
            TEST.as_posix(),
            "tests/python/test_experiment_7666_v668_arc_goal_confirmation.py",
        ],
        "changed_modules": [MODULE.as_posix(), GUARD.as_posix()],
        "static_paths": [WRAPPER.as_posix()],
        "e2e": "E2E-013",
    }
    commands = _validation_commands(root, private)
    checks, hashes, context = _preconditions(root, started)
    progress(started, "preflight", "after", failed=sum(not row["passed"] for row in checks))
    spans: list[Json] = [
        {
            "phase": "preflight",
            "start_s": 0.0,
            "end_s": time.monotonic() - started,
            "completed_units": len(checks),
            "heartbeat_times": [],
        }
    ]
    selected = select_game(context["roster"]) if context.get("roster") else "unavailable"
    schedule = make_schedule(selected)
    rows: list[Json] = []
    runtime: Json = {}
    if all(row["passed"] for row in checks):
        registry = {
            str(item.get("game")): item
            for item in context["registry"].get("games", [])
            if isinstance(item, Mapping)
        }
        precheck = {
            "game": selected,
            "registered": selected in registry,
            "historical_levels_reproduced": registry.get(selected, {}).get("levels_reproduced"),
            "consulted_before_launch": True,
            "outcome_used_for_selection": False,
        }
        raw = root / raw_for_run(RAW, f"{run_date}-{os.getpid()}-{time.monotonic_ns()}")
        raw.mkdir(parents=True, exist_ok=True)
        atomic_json(
            raw / "schedule.json",
            {"schedule": schedule, "registry_precheck": precheck, "salt": SALT},
        )
        live_start = time.monotonic() - started
        progress(started, "live", "before_subprocess", game=selected)
        child = [
            str(root / ".venv/bin/python"),
            "-u",
            str(root / WRAPPER),
            "--live-child",
            str(raw / "schedule.json"),
        ]
        receipt = run_commands(
            root,
            [
                __import__(
                    "carnot.reporting.experiment_7303_validation_scope", fromlist=["CommandSpec"]
                ).CommandSpec(
                    "owned_live_child", tuple(child), "one_adapter_withheld_episode", 3480
                )
            ],
            log_dir=private / "live_logs",
            heartbeat_s=45,
        )[0]
        child_result = (
            json.loads((raw / "child_result.json").read_text())
            if (raw / "child_result.json").is_file()
            else {}
        )
        checkpoint = (
            json.loads((raw / "runtime_checkpoint.json").read_text())
            if (raw / "runtime_checkpoint.json").is_file()
            else {}
        )
        runtime = {**checkpoint, **dict(child_result.get("runtime") or {})}
        runtime["child_receipt"] = receipt
        if runtime.get("model_sha256"):
            hashes["producer_files"]["model_weights"] = {
                "path": str(context["model_path"]),
                "sha256": runtime["model_sha256"],
            }
        if runtime.get("runtime_sha256"):
            hashes["producer_files"]["native_runtime"] = {
                "path": str(context["server"]),
                "sha256": runtime["runtime_sha256"],
            }
        if child_result.get("row"):
            rows = [child_result["row"]]
        else:
            actions = [
                event
                for event in _read_jsonl(raw / "actions.jsonl")
                if event.get("event") == "action_end"
            ]
            runtime["error"] = child_result.get("error") or "owned_child_incomplete"
            if actions:
                rows = [
                    {
                        **schedule,
                        "actions": actions,
                        "actions_used": len(actions),
                        "request_rows": [],
                        "goal_observations": [],
                        "induction_events": [],
                        "censored": True,
                        "censor_reason": runtime["error"],
                        "exclusion": None,
                        "raw_provenance": "live_agent_self_discovery",
                    }
                ]
        progress(started, "live", "after_subprocess", exit=receipt["exit_code"], rows=len(rows))
        spans.append(
            {
                "phase": "live",
                "start_s": live_start,
                "end_s": time.monotonic() - started,
                "completed_units": len(rows),
                "heartbeat_times": [],
                "checkpoint": str(raw / "actions.jsonl"),
            }
        )
    artifact = _artifact(checks, hashes, schedule, rows, runtime, time.monotonic() - started)
    artifact["affected_validation_manifest"] = manifest
    artifact["registry_precheck"] = (
        precheck if not any(not row["passed"] for row in checks) else None
    )
    validation_start = time.monotonic() - started
    progress(started, "validation", "before", commands=len(commands))
    receipts = run_commands(root, commands, log_dir=private / "validation_logs", heartbeat_s=45)
    progress(started, "validation", "after", passed=sum(row["passed"] for row in receipts))
    artifact["validation_receipts"] = receipts
    valid = all(row["passed"] for row in receipts)
    artifact["acceptance_gate_results"]["readiness"] = _gate(
        valid and artifact["acceptance_gate_results"]["validity"]["passed"],
        "Frozen scoped checks and valid current SDK evidence govern readiness.",
        required_exits=[row["exit_code"] for row in receipts],
    )
    if not valid:
        artifact["honest_verdict"] = "complete_disqualified_required_validation"
        artifact["verdict_class"] = "disqualified"
        artifact["live_goal_observation_complete_score"] = 0
    spans.append(
        {
            "phase": "validation",
            "start_s": validation_start,
            "end_s": time.monotonic() - started,
            "completed_units": len(receipts),
            "heartbeat_times": [],
        }
    )
    artifact["phase_spans"] = spans
    artifact["duration_s"] = time.monotonic() - started
    candidate = private / "terminal_candidate.json"
    atomic_json(candidate, artifact)
    progress(started, "terminal_readers", "before")
    terminal = run_commands(
        root, _terminal_commands(root, candidate), log_dir=private / "terminal_logs", heartbeat_s=45
    )
    if any(not row["passed"] for row in terminal):
        artifact["honest_verdict"] = "complete_disqualified_terminal_reader"
        artifact["verdict_class"] = "disqualified"
        artifact["live_goal_observation_complete_score"] = 0
        artifact["acceptance_gate_results"]["readiness"]["passed"] = False
    artifact["flagged_adversarial"] = any(
        row["name"] == "adversarial_verify" and not row["passed"] for row in terminal
    )
    artifact["terminal_reader_receipts_path"] = str(root / RAW / "terminal_reader_receipts.json")
    artifact["duration_s"] = time.monotonic() - started
    artifact["field_principles"] = {
        key: "Current raw operands and scoped validation govern this field." for key in artifact
    }
    atomic_json(candidate, artifact)
    exact = run_commands(
        root,
        _terminal_commands(root, candidate),
        log_dir=private / "terminal_exact",
        heartbeat_s=45,
    )
    sidecar = root / RAW / "terminal_reader_receipts.json"
    atomic_json(sidecar, {"candidate_sha256": sha256_file(candidate), "receipts": exact})
    progress(started, "terminal_readers", "after", passed=sum(row["passed"] for row in exact))
    if any(not row["passed"] for row in exact):
        return 2
    atomic_json(output, artifact)
    progress(started, "publication", "after", path=output, sha256=sha256_file(output))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Serve the declared run and read-only cold reducer modes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260925")
    parser.add_argument("--output", type=Path, default=RESULT)
    parser.add_argument("--cold-reduce", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    parser.add_argument("--live-child", type=Path)
    args = parser.parse_args(argv)
    if args.cold_reduce or args.independent_reduce:
        return _reader(args.cold_reduce or args.independent_reduce)
    if args.live_child:
        schedule = json.loads(args.live_child.read_text())["schedule"]
        return _child_measure(ROOT, schedule, args.live_child.resolve().parent)
    return run_experiment(ROOT.resolve(), args.date, (ROOT / args.output).resolve())

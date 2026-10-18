"""Bounded public ARC first contact with current Qwen evidence (REQ-REPORT-7709)."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import socket
import tempfile
import threading
import time
from typing import Any

from carnot.agentic.arc_generalization_runtime import WITHHELD
from carnot.reporting.current_work_receipt import atomic_json, sha256_file

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
RESULT = Path("results/experiment_7709_v671_arc_first_contact.json")
RAW = Path("results/raw/experiment_7709_v671_arc_first_contact")
UPSTREAM = Path("results/experiment_7708_v671_arc_generalization_runner.json")
SCHEDULE = Path("results/raw/experiment_7708_v671_arc_generalization_runner/schedule.json")
TEST = "tests/python/test_experiment_7709_v671_arc_first_contact.py"
MODULE = "python/carnot/experiment_7709_v671_arc_first_contact.py"
WRAPPER = "scripts/experiments/experiment_7709_v671_arc_first_contact.py"
REQUIRED_FIELDS = (
    "honest_verdict",
    "verdict_class",
    "flagged_adversarial",
    "gate_check_summary",
    "acceptance_gate_results",
    "rows",
    "sample_size_budget",
    "inference_substrate",
    "inference_substrate_class",
    "MODEL_SPECS",
    "model_invoked",
    "execution_venue",
    "phase_spans",
    "random_seed",
    "reproducibility_checksum",
    "source_artifact_hashes",
    "preconditions_checked",
    "validation_receipts",
    "verifier_is_oracle",
    "arc_measurement_complete_score",
    "solve_provenance",
    "per_game_results",
    "current_model_receipts",
    "new_solve_credit",
)
GATE_NAMES = (
    "validity",
    "readiness",
    "coverage",
    "freshness",
    "probability",
    "utility",
    "retention",
    "efficiency",
)


def progress(started: float, phase: str, event: str, **detail: Any) -> None:
    """Emit a flushed owner heartbeat with elapsed time and completed work."""
    values = " ".join(f"{key}={value}" for key, value in detail.items())
    print(
        f"[exp7709] {phase} {event} elapsed_s={time.monotonic() - started:.3f} {values}", flush=True
    )


def gate_check(
    name: str, upstream: str, path: str, field: str, expected: Any, observed: Any
) -> Json:
    """Name both operands of an immutable prerequisite or resource check."""
    return {
        "check": name,
        "upstream": upstream,
        "path": path,
        "field": field,
        "operator": "==",
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
    }


def schedule_check(schedule: Mapping[str, Any]) -> Json:
    """Require Exp7708's two sealed games and every adapter denial."""
    rows = schedule.get("rows", [])
    observed = [str(row.get("game")) for row in rows] if isinstance(rows, list) else []
    valid = (
        schedule.get("selection_salt") == "v671-arc-20260926"
        and observed == ["wa30", "lf52"]
        and all(
            row.get("adapter_withheld") is True
            and set(row.get("withheld_inputs", [])) == set(WITHHELD)
            for row in rows
        )
    )
    return gate_check("exp7708_schedule", "experiment_7708", str(SCHEDULE), "rows", True, valid)


def goal_support(
    attempts: Sequence[Mapping[str, Any]], actions: Sequence[Mapping[str, Any]]
) -> list[Json]:
    """Join a reachable predicate firing to later SDK progress per attempt."""
    support = []
    for attempt in attempts:
        attempt_id = str(attempt["attempt_id"])
        positions = [
            index for index, row in enumerate(actions) if attempt_id in row.get("goal_firings", [])
        ]
        fired = bool(positions)
        confirmed = any(
            int(row.get("level_after") or 0) > int(actions[index].get("level_before") or 0)
            for index in positions
            for row in actions[index:]
        )
        support.append(
            {
                "attempt_id": attempt_id,
                "goal_predicate": attempt.get("goal_predicate"),
                "engine_accepted": attempt.get("accepted") is True,
                "reachable_firing": fired,
                "sdk_confirmed_level_progress": confirmed,
                "recall": "unknown"
                if not fired
                else "observed_support"
                if confirmed
                else "observed_no_support",
            }
        )
    return support


def reduce_rows(rows: Sequence[Mapping[str, Any]]) -> Json:
    """Recompute descriptive support from two independent public-game units."""
    if [row.get("game") for row in rows] != ["wa30", "lf52"]:
        raise ValueError("frozen_game_identity")
    joined_actions = 0
    joined_requests = 0
    per_game = []
    for row in rows:
        observations = {str(item["observation_id"]): item for item in row.get("observations", [])}
        for action in row.get("actions", []):
            if str(action.get("observation_id")) not in observations:
                raise ValueError("observation_join")
            joined_actions += 1
        for request in row.get("requests", []):
            if not request.get("request_path") or (
                request.get("completed") and not request.get("response_path")
            ):
                raise ValueError("request_response_join")
            joined_requests += 1
        support = goal_support(row.get("induction_attempts", []), row.get("actions", []))
        per_game.append(
            {
                "game": row["game"],
                "actions": len(row.get("actions", [])),
                "sdk_peak_level": row.get("peak_level"),
                "censoring": row.get("censoring"),
                "model_calls": len(row.get("requests", [])),
                "output_tokens": sum(
                    int(item.get("output_tokens") or 0) for item in row.get("requests", [])
                ),
                "goal_support": support,
            }
        )
    return {
        "observed_games": 2,
        "joined_actions": joined_actions,
        "joined_requests": joined_requests,
        "per_game_results": per_game,
        "goal_recall": "unknown"
        if not any(goal["reachable_firing"] for game in per_game for goal in game["goal_support"])
        else "descriptive_only",
    }


def cold_reduce(path: Path) -> Json:
    """Read raw rows only; no game source or expert model enters this reduction."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return reduce_rows(payload["rows"])


def _gate(passed: bool, principle: str, **operands: Any) -> Json:
    return {"passed": passed, "principle": principle, "measured_operands": operands}


def build_artifact(
    *,
    checks: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    reduction: Mapping[str, Any] | None,
    hashes: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    duration_s: float,
    run_date: str,
    runtime: Mapping[str, Any] | None = None,
    spans: Sequence[Mapping[str, Any]] = (),
    flagged_adversarial: bool = False,
) -> Json:
    """Reduce measured inputs without converting absent science into a partial run."""
    actual = dict(runtime or {})
    failed = [dict(check) for check in checks if check.get("passed") is not True]
    complete = len(rows) == 2 and reduction is not None
    valid_receipts = bool(receipts) and all(item.get("exit_code") == 0 for item in receipts)
    if failed:
        verdict_class, reason = "blocked", str(failed[0]["check"])
    elif flagged_adversarial or (receipts and not valid_receipts):
        verdict_class, reason = "disqualified", "required_validation"
    elif complete:
        verdict_class, reason = "null", "two_public_game_descriptive_support"
    else:
        verdict_class, reason = "partial", "owned_episode_unfinished"
    loads = int(actual.get("loads_attempted") or 0)
    requests = [request for row in rows for request in row.get("requests", [])]
    generations = sum(request.get("completed") is True for request in requests)
    tokens = sum(int(request.get("output_tokens") or 0) for request in requests)
    current_class = (
        "model_full_generation"
        if generations
        else "model_load_no_generation"
        if loads
        else "no_model_load"
    )
    ready = complete and valid_receipts and verdict_class == "null"
    gates = {
        "validity": _gate(
            complete and not failed,
            "Both current episode dispositions and authentic operands are required.",
            episodes=len(rows),
            failed_checks=len(failed),
        ),
        "readiness": _gate(
            ready,
            "Readiness prevents invalid evidence propagation.",
            valid_receipts=valid_receipts,
            complete=complete,
        ),
        "coverage": _gate(
            valid_receipts,
            "Changed code and required E2E checks must pass.",
            passed_receipts=sum(item.get("exit_code") == 0 for item in receipts),
            required_receipts=len(receipts),
        ),
        "freshness": _gate(
            generations > 0,
            "Current generation needs an owned completed request.",
            generations=generations,
        ),
        "probability": _gate(
            False,
            "Two exposed public games cannot estimate hidden solve probability.",
            public_games=len(rows),
            hidden_games=0,
        ),
        "utility": _gate(
            False,
            "Quality thresholds prevent effects being inferred from plumbing.",
            paired_causal_arms=0,
        ),
        "retention": _gate(
            False, "Retention bounds prevent improvement by forgetting.", retention_groups=0
        ),
        "efficiency": _gate(
            False, "Efficiency requires a causal paired comparison.", paired_causal_arms=0
        ),
    }
    budget = {
        "intended_games": 2,
        "intended_episodes": 2,
        "observed_games": len({row.get("game") for row in rows}),
        "observed_episodes": len(rows),
        "eligible_episodes": sum(not row.get("exclusions") for row in rows),
        "excluded_episodes": sum(bool(row.get("exclusions")) for row in rows),
        "censored_episodes": sum(bool(row.get("censoring")) for row in rows),
        "effective_blocks": len({row.get("game") for row in rows}),
        "prior_exposure": "25 public survey games; selected games historically cleared",
        "max_seconds_per_episode": 1200,
        "total_live_seconds": 2400,
        "max_actions_per_episode": 128,
        "max_engine_calls_per_episode": 20000,
        "max_qwen_calls_per_episode": 2,
        "max_output_tokens_per_call": 4096,
    }
    artifact: Json = {
        "schema": "carnot.exp7709.v671.arc_first_contact.v1",
        "experiment_id": 7709,
        "milestone": "2026.09.671",
        "run_date": run_date,
        "status": "complete" if verdict_class != "partial" else "partial",
        "honest_verdict": f"complete_{verdict_class}_{reason}"
        if verdict_class != "partial"
        else f"partial_{reason}",
        "verdict_class": verdict_class,
        "flagged_adversarial": flagged_adversarial,
        "gate_check_summary": {"failed_checks": failed},
        "acceptance_gate_results": gates,
        "rows": [dict(row) for row in rows],
        "sample_size_budget": budget,
        "inference_substrate": "current_owned_cuda_qwen_e3"
        if loads
        else "host_preflight_no_model_call",
        "inference_substrate_class": current_class,
        "planned_inference_substrate_class": "model_full_generation",
        "MODEL_SPECS": [MODEL_ID] if loads else [],
        "planned_MODEL_SPECS": [MODEL_ID],
        "model_specs": [{"name": MODEL_ID, "scope": "current"}]
        if loads
        else [{"name": "none", "scope": "current", "reason": "no_model_load"}],
        "model_invoked": bool(loads or requests),
        "invocation_counts": {
            "loads": loads,
            "forwards": len(requests),
            "generations": generations,
            "tokens": tokens,
            "failures": sum(request.get("failed") is True for request in requests),
            "cancellations": sum(request.get("cancelled") is True for request in requests),
        },
        "execution_venue": "host",
        "execution_venue_details": {
            "host": socket.gethostname(),
            "owned_pid": os.getpid(),
            "gpu_uuid": actual.get("gpu_uuid"),
            "server_pid": actual.get("server_pid"),
        },
        "phase_spans": [dict(span) for span in spans],
        "duration_s": duration_s,
        "random_seed": {"selection_salt": "v671-arc-20260926", "episode_seed": 7709},
        "source_artifact_hashes": dict(hashes),
        "preconditions_checked": [dict(item) for item in checks],
        "validation_receipts": [dict(item) for item in receipts],
        "verifier_is_oracle": False,
        "arc_measurement_complete_score": int(complete and not failed),
        "solve_provenance": "live_agent_self_discovery" if rows else "no_live_attempt",
        "per_game_results": list(reduction.get("per_game_results", [])) if reduction else [],
        "current_model_receipts": actual,
        "new_solve_credit": False,
        "reduction": dict(reduction) if reduction else None,
        "effective_agent_backend": {
            "requested": "codex",
            "effective": "codex"
            if os.environ.get("CODEX_FORCE_EXPERIMENTS") == "1"
            else "route_requested",
            "CODEX_FORCE_EXPERIMENTS": os.environ.get("CODEX_FORCE_EXPERIMENTS"),
            "session_id": os.environ.get("CODEX_SESSION_ID"),
        },
        "current_agent_invocation_receipt": {
            "session_id": os.environ.get("CODEX_SESSION_ID"),
            "backend": "codex",
            "successful_current_process": bool(os.environ.get("CODEX_SESSION_ID")),
        },
        "prior_failures": [
            {
                "experiment_id": "exp7694",
                "custody": "not_emitted_usage_limit_three_attempts",
                "same_verdict_retirement": "not_applicable_no_producer",
            }
        ],
    }
    artifact["field_principles"] = {
        name: "This record makes the stated claim independently checkable and limits its scope."
        for name in REQUIRED_FIELDS
    }
    artifact["field_principles"].update({name: gates[name]["principle"] for name in GATE_NAMES})
    artifact["reproducibility_checksum"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                [hashes, artifact["random_seed"], MODULE, reduction], sort_keys=True, default=str
            ).encode()
        ).hexdigest()
    )
    return artifact


def collect_preconditions(root: Path, started: float) -> tuple[list[Json], Json, Json]:
    """Authenticate immutable inputs and inspect exclusive current resources."""
    from carnot.experiment_7630_v666_cuda_ownership import (
        ProcessRegistry,
        _current_inventory,
        select_owned_capacity,
    )
    from carnot.inference.sota_models import cached_current_model

    inputs = (
        "AGENTS.md",
        "CODEX.md",
        "CLAUDE.md",
        "research-program.md",
        "ops/exclusion_manifest.yaml",
        "ops/e2e-test-plan.md",
        "scripts/experiment_template.py",
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/experiment_7303_validation_scope.py",
        "openspec/capabilities/research-reporting/spec.md",
        "openspec/capabilities/arc-world-model-trust-energy/spec.md",
        "python/carnot/agentic/arc_competition_agent.py",
        "python/carnot/agentic/arc_executable_world_model.py",
        "python/carnot/agentic/arc_goal_confirmation.py",
        "python/carnot/experiment_7630_v666_cuda_ownership.py",
        "python/carnot/inference/sota_models.py",
        "ops/arc_solve_registry.yaml",
        str(UPSTREAM),
        str(SCHEDULE),
        MODULE,
        WRAPPER,
        TEST,
    )
    checks = []
    hashes: Json = {"producer_files": {}, "pre_gate_receipts": {}, "missing_custody": []}
    for relative in inputs:
        path = root / relative
        present = path.is_file() and path.stat().st_size > 0
        checks.append(
            gate_check("input_bytes", relative, str(path), "readable_nonempty", True, present)
        )
        if present:
            target = (
                "pre_gate_receipts"
                if relative in (str(UPSTREAM), str(SCHEDULE))
                else "producer_files"
            )
            hashes[target][relative] = sha256_file(path)
        else:
            hashes["missing_custody"].append(relative)
    upstream = json.loads((root / UPSTREAM).read_text()) if (root / UPSTREAM).is_file() else {}
    for field, expected in (
        ("arc_runner_ready_score", 1),
        ("flagged_adversarial", False),
        ("verdict_class", "circular_positive"),
        ("arc_schedule_path", str(SCHEDULE)),
    ):
        checks.append(
            gate_check(
                f"exp7708_{field}",
                "experiment_7708",
                str(root / UPSTREAM),
                field,
                expected,
                upstream.get(field),
            )
        )
    schedule = json.loads((root / SCHEDULE).read_text()) if (root / SCHEDULE).is_file() else {}
    checks.append(schedule_check(schedule))
    checks.append(
        gate_check(
            "scored_entrypoint",
            "arc_competition_agent",
            str(root / "python/carnot/agentic/arc_competition_agent.py"),
            "E3AgentPolicy_and_make_carnot_agent",
            True,
            "E3AgentPolicy" in (root / "python/carnot/agentic/arc_competition_agent.py").read_text()
            and "make_carnot_agent"
            in (root / "python/carnot/agentic/arc_competition_agent.py").read_text(),
        )
    )
    checks.append(
        gate_check(
            "current_backend_invocation",
            "current_codex_invocation",
            "environment:CODEX_SESSION_ID",
            "successful_current_session_visible",
            True,
            bool(os.environ.get("CODEX_SESSION_ID")),
        )
    )
    model = cached_current_model()
    model_path = Path(str(model["model_path"])).resolve() if model else Path("/missing/model.gguf")
    checks.append(
        gate_check(
            "mandated_qwen_cache",
            MODEL_ID,
            str(model_path),
            "nonempty_Q4_K_M_GGUF",
            True,
            model is not None
            and model_path.is_file()
            and model_path.stat().st_size > 15_000_000_000,
        )
    )
    server = (Path.home() / ".cache/llama.cpp-master/build/bin/llama-server").resolve()
    checks.append(
        gate_check(
            "cuda_server_binary",
            "local_llama_cpp",
            str(server),
            "executable_with_cuda_backend",
            True,
            server.is_file()
            and os.access(server, os.X_OK)
            and (server.parent / "libggml-cuda.so").is_file(),
        )
    )
    progress(started, "cuda_preflight", "before")
    owner = ProcessRegistry.current()
    owner.task_id = "experiment_7709_v671_arc_first_contact"
    inventory = _current_inventory()
    selected, ownership_rows = select_owned_capacity(inventory, owner)
    progress(
        started, "cuda_preflight", "after", selected=selected.get("uuid") if selected else None
    )
    checks.append(
        gate_check(
            "exclusive_cuda_capacity",
            "current_gpu_inventory",
            "nvidia-smi",
            "exclusive_device_available",
            True,
            selected is not None,
        )
    )
    return (
        checks,
        hashes,
        {
            "schedule": schedule,
            "model_path": model_path,
            "server": server,
            "selected_gpu": selected,
            "owner": owner,
            "ownership_rows": ownership_rows,
        },
    )


def run_episode(
    unit: Mapping[str, Any], proposer: Any, capture: Any, raw: Path, started: float
) -> Json:
    """Drive one scored E3 policy with only its visible SDK observations."""
    import signal
    import functools
    from arcengine import GameAction
    from carnot import experiment_7471_v654_arc_seam_observation as live
    from carnot.agentic import arc_executable_world_model as world
    from carnot.agentic.arc_agi3_world_model import grid_of
    from carnot.agentic.arc_competition_agent import make_carnot_agent
    from carnot.agentic.arc_request_budget import attach_request_budget
    from carnot.agentic.arc_solver_kit import offline_arcade

    episode_id = str(unit["episode_id"])
    game = str(unit["game"])
    episode_raw = raw / "episodes" / game
    episode_raw.mkdir(parents=True, exist_ok=True)
    event_path = raw / "runtime_events.jsonl"
    action_path = raw / "actions.jsonl"
    old_dir = world.E3_DIR
    world.E3_DIR = episode_raw / "fresh_e3"
    capture.begin_episode(episode_id)
    budget = live.live_support.DurableEpisodeRequestBudget(
        episode_id, limit=2, deadline_s=1200, event_path=event_path
    )
    attach_request_budget(proposer, budget)

    class LocalAgentBase:
        def __init__(self, game_id: str) -> None:
            self.game_id = game_id

    originals = live._disable_cross_game_loaders()
    try:
        agent_type = make_carnot_agent(LocalAgentBase, cascade=True, proposer=proposer)
        agent = agent_type(game_id=game)
    finally:
        live._restore_cross_game_loaders(originals)
    policy = agent._policy
    observer = live.E3SeamObserver(episode_id, episode_raw / "seam_events.jsonl")
    observer.install(policy)
    arcade = offline_arcade()
    env = arcade.make(game, scorecard_id=arcade.open_scorecard())
    frames: list[Any] = []
    latest = None
    actions: list[Json] = []
    observations: list[Json] = []
    entered = time.monotonic()
    disposition = "complete"
    error = None
    goal_probe_errors: list[str] = []
    engine_calls = 0
    original_plan = world.plan_in_model

    @functools.wraps(original_plan)
    def counted_plan(engine: Any, goal: Any, grid: Any, **kwargs: Any) -> Any:
        nonlocal engine_calls

        def counted_engine(*args: Any, **inner_kwargs: Any) -> Any:
            nonlocal engine_calls
            if engine_calls >= 20000:
                raise RuntimeError("planner_engine_call_budget_exhausted")
            engine_calls += 1
            return engine(*args, **inner_kwargs)

        progress(started, "planner", "before", game=game, completed_calls=engine_calls)
        result = original_plan(
            counted_engine,
            goal,
            grid,
            **{
                **kwargs,
                "max_nodes": min(int(kwargs.get("max_nodes", 20000)), max(0, 20000 - engine_calls)),
            },
        )
        progress(started, "planner", "after", game=game, completed_calls=engine_calls)
        return result

    world.plan_in_model = counted_plan

    def timeout(_signum: int, _frame: Any) -> None:
        raise live.EpisodeTimeout("1200s episode ceiling")

    previous_alarm = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 1200)
    try:
        for index in range(128):
            if agent.is_done(frames, latest):
                break
            progress(started, "action", "before", game=game, completed=index)
            action = agent.choose_action(frames, latest)
            name = str(getattr(action, "name", action))
            payload = getattr(action, "action_data", None)
            data = payload.model_dump() if hasattr(payload, "model_dump") else None
            if isinstance(data, dict):
                data = {key: value for key, value in data.items() if key != "game_id"}
            level_before = observations[-1]["level"] if observations else 0
            latest = (
                env.reset()
                if name == "RESET"
                else env.step(action if isinstance(action, GameAction) else action, data=data)
            )
            if latest is None:
                raise RuntimeError("sdk_null_observation")
            level_after = live.live_support._level(latest)
            observation_id = f"{episode_id}:o{index + 1}"
            observation = {
                "observation_id": observation_id,
                "level": level_after,
                "state_sha256": live.canonical_hash(str(latest)),
            }
            observations.append(observation)
            action_row = {
                "action_index": index + 1,
                "action": name,
                "data": data,
                "observation_id": observation_id,
                "level_before": level_before,
                "level_after": level_after,
                "goal_firings": [],
            }
            predicate = getattr(policy, "_goal_confirmation_predicate", None)
            if callable(predicate) and getattr(policy, "induction_attempts", None):
                try:
                    logical = world.to_logical(grid_of(latest), world.detect_cell(grid_of(latest)))
                    if predicate(logical):
                        action_row["goal_firings"].append(
                            f"{episode_id}:induction:{len(policy.induction_attempts)}"
                        )
                except Exception as exc:
                    goal_probe_errors.append(f"{type(exc).__name__}: {exc}"[:200])
            actions.append(action_row)
            live._append_jsonl(
                action_path, {"episode_id": episode_id, **action_row, "observation": observation}
            )
            atomic_json(
                episode_raw / "checkpoint.json", {"actions": actions, "observations": observations}
            )
            frames.append(latest)
            progress(started, "action", "after", game=game, completed=index + 1, level=level_after)
        else:
            disposition = "censored_action_limit"
    except live.EpisodeTimeout as exc:
        disposition, error = "censored_timeout", str(exc)
    except BaseException as exc:
        disposition, error = "censored_execution_error", f"{type(exc).__name__}: {exc}"[:500]
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm)
        budget.cancel("episode_terminal")
        world.E3_DIR = old_dir
        world.plan_in_model = original_plan
    requests = live.live_support._transport_rows(
        live.live_support._read_jsonl(event_path), episode_id
    )
    request_rows = []
    for request in requests:
        response_path = Path(str(request.get("response_path") or ""))
        response = json.loads(response_path.read_text()) if response_path.is_file() else {}
        usage = response.get("usage") or {}
        request_rows.append(
            {
                **request,
                "completed": bool(request.get("response_observed")),
                "failed": request.get("event") == "server_error",
                "output_tokens": int(
                    usage.get("completion_tokens") or response.get("tokens_predicted") or 0
                ),
            }
        )
    attempts = []
    for index, attempt in enumerate(getattr(policy, "induction_attempts", [])):
        attempts.append(
            {
                "attempt_id": f"{episode_id}:induction:{index + 1}",
                "goal_predicate": attempt.get("goal_expression"),
                "goal_candidate_names": attempt.get("goal_candidate_names", []),
                "accepted": bool(attempt.get("planned")),
                "reason": attempt.get("reason"),
                "skipped": attempt.get("skipped"),
                "goal_predicate_satisfiable_in_model": attempt.get("goal_predicate_satisfiable"),
                "refinement_rounds": attempt.get("refinement_rounds", []),
            }
        )
    return {
        "episode_id": episode_id,
        "game": game,
        "arm": "adapter_withheld",
        "actions": actions,
        "observations": observations,
        "requests": request_rows,
        "induction_attempts": attempts,
        "peak_level": max((item["level"] for item in observations), default=None),
        "censoring": disposition if disposition.startswith("censored") else None,
        "disposition": disposition,
        "error": error,
        "exclusions": [],
        "policy_entry": {
            "factory": "make_carnot_agent",
            "policy_class": type(policy).__name__,
            "withheld_inputs": list(WITHHELD),
        },
        "engine_calls": engine_calls,
        "goal_probe_errors": goal_probe_errors,
        "elapsed_s": time.monotonic() - entered,
        "solve_provenance": "live_agent_self_discovery",
        "new_solve_credit": False,
    }


def run_live(root: Path, context: Mapping[str, Any], started: float) -> tuple[list[Json], Json]:
    """Hold one owned CUDA server for the sealed two-episode budget."""
    from carnot import experiment_7471_v654_arc_seam_observation as live
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer
    from carnot.experiment_7431_v651_arc_live_sentinel import _free_port
    from carnot.experiment_7630_v666_cuda_ownership import _current_inventory, recheck_before_launch
    from carnot.experiment_7581_v662_arc_bounded_canary import _owned_vram_mb
    from carnot.gpu_lease_phase_journal import GpuLease

    raw = root / RAW
    raw.mkdir(parents=True, exist_ok=True)
    gpu = dict(context["selected_gpu"])
    model_path = Path(context["model_path"])
    server = Path(context["server"])
    progress(started, "model_hash", "before", bytes=model_path.stat().st_size)
    model_hash = sha256_file(model_path)
    progress(started, "model_hash", "after", sha256=model_hash)
    lease = GpuLease.acquire(
        runtime_dir=Path("/tmp/carnot-gpu-leases"),
        task_id="experiment_7709_v671_arc_first_contact",
        device_uuid=str(gpu["uuid"]),
        expected_model=str(model_path),
        vram_before_mb=int(gpu.get("memory_used_mb") or 0),
        ttl_s=120,
    )
    progress(started, "gpu_lease", "after", lease_id=lease.lease_id)
    stop = threading.Event()
    action_log = raw / "actions.jsonl"

    def heartbeat() -> None:
        while not stop.wait(45):
            lease.heartbeat()
            completed = (
                sum(1 for line in action_log.read_text().splitlines() if line.strip())
                if action_log.is_file()
                else 0
            )
            progress(started, "owned_wait", "heartbeat", completed_actions=completed)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    runtime: Json = {
        "model_path": str(model_path),
        "gguf_sha256": model_hash,
        "server_path": str(server),
        "server_sha256": sha256_file(server),
        "gpu_uuid": gpu["uuid"],
        "owner_pid": os.getpid(),
        "lease_id": lease.lease_id,
        "loads_attempted": 0,
        "loads_completed": 0,
        "error": None,
    }
    rows: list[Json] = []
    proposer = None
    capture = None
    old_env = dict(os.environ)
    try:
        recheck = recheck_before_launch(
            str(gpu["uuid"]), [_current_inventory(), _current_inventory()], context["owner"]
        )
        runtime["prelaunch_recheck"] = recheck
        if recheck.get("passed") is not True:
            raise RuntimeError("foreign_or_capacity_recheck_failed")
        port = _free_port()
        env = live.live_support.session_environment(
            os.environ, gpu_index=int(gpu["index"]), port=port, raw_dir=raw
        )
        os.environ.update(env)
        os.environ.update(
            {
                "CARNOT_ARC_GGUF_PATH": str(model_path),
                "CARNOT_LLAMA_SERVER": str(server),
                "CARNOT_ARC_INDUCE_MAX_TOKENS": "4096",
                "CARNOT_ARC_INDUCE_TIMEOUT": "1200",
                "CARNOT_ARC_GENERATOR_REQUIRE_CUDA": "1",
                "CARNOT_ARC_MTP": "0",
                "CARNOT_ARC_PLAN_HUD_DEDUP": "0",
                "CARNOT_ARC_PROBE_PROTOCOL": "0",
                "CARNOT_ARC_SERVER_LOG_DIR": str(raw / "server_logs"),
            }
        )
        runtime["effective_settings"] = {
            key: os.environ.get(key)
            for key in (
                "CARNOT_ARC_INDUCE_MAX_TOKENS",
                "CARNOT_ARC_INDUCE_TIMEOUT",
                "CARNOT_ARC_GENERATOR_REQUIRE_CUDA",
                "CARNOT_ARC_MTP",
                "CARNOT_ARC_PLAN_HUD_DEDUP",
                "CARNOT_ARC_PROBE_PROTOCOL",
            )
        }
        lease.transition("admitted")
        lease.transition("loading")
        progress(started, "model_load", "before", model=MODEL_ID)
        runtime["loads_attempted"] = 1
        proposer = LocalGGUFProposer(
            repo_substr="Qwen3.8-27B",
            model_path=str(model_path),
            port=port,
            mtp=False,
            kv_quant="q8_0",
            use_chat_template=True,
            n_gpu_layers=999,
            n_ctx=32768,
            max_tokens=4096,
            timeout=1200,
            tries=1,
        )
        proposer.model_repository = MODEL_ID
        proposer.requested_model_path = str(model_path)
        healthy = proposer._ensure_server()
        runtime["server_pid"] = getattr(getattr(proposer, "_proc", None), "pid", None)
        runtime["server_command"] = list(getattr(proposer, "last_launch_argv", ()) or ())
        runtime["owned_server_vram_mb"] = _owned_vram_mb(runtime["server_pid"])
        runtime["loads_completed"] = int(bool(healthy and runtime["owned_server_vram_mb"]))
        progress(
            started,
            "model_load",
            "after",
            healthy=healthy,
            server_pid=runtime["server_pid"],
            owned_vram_mb=runtime["owned_server_vram_mb"],
        )
        if not runtime["loads_completed"]:
            raise RuntimeError("current_server_offload_unverified")
        lease.transition("resident", vram_mb=int(runtime["owned_server_vram_mb"]))
        lease.transition("inferencing")

        class ObservedCapture(live.live_support.DurableRequestCapture):
            def _open(self, request: Any, *args: Any, **kwargs: Any) -> Any:
                is_generation = str(getattr(request, "full_url", "")).endswith(
                    ("/completion", "/v1/chat/completions", "/v1/completions")
                )
                if is_generation:
                    progress(
                        started,
                        "generation",
                        "before",
                        episode=self.episode_id,
                        attempted=self.indices[self.episode_id],
                    )
                try:
                    return super()._open(request, *args, **kwargs)
                finally:
                    if is_generation:
                        progress(
                            started,
                            "generation",
                            "after",
                            episode=self.episode_id,
                            attempted=self.indices[self.episode_id],
                        )

        capture = ObservedCapture(raw, raw / "runtime_events.jsonl", max_new_tokens=4096)
        capture.install()
        for game in ("wa30", "lf52"):
            unit = {
                "game": game,
                "episode_id": f"{game}:live",
                "arm": "adapter_withheld",
                "seed": 7709,
            }
            progress(started, "episode", "before", game=game, completed_units=len(rows))
            row = run_episode(unit, proposer, capture, raw, started)
            rows.append(row)
            atomic_json(raw / "episode_rows.json", {"rows": rows})
            progress(
                started,
                "episode",
                "after",
                game=game,
                completed_units=len(rows),
                disposition=row["disposition"],
            )
        runtime["request_rows"] = [request for row in rows for request in row["requests"]]
    except BaseException as exc:
        runtime["error"] = f"{type(exc).__name__}: {exc}"[:500]
        progress(started, "live", "error", error=runtime["error"])
    finally:
        if capture is not None:
            capture.restore()
        progress(started, "model_unload", "before")
        if proposer is not None:
            proposer.stop()
        progress(started, "model_unload", "after")
        os.environ.clear()
        os.environ.update(old_env)
        stop.set()
        thread.join(timeout=1)
        if runtime["loads_completed"]:
            lease.transition("unloading")
            lease.transition("validating", vram_mb=0, exit_code=0, unload_observed=True)
            lease.transition("terminal_complete")
        else:
            lease.transition("terminal_blocked")
        runtime["lease_release"] = lease.release()
    return rows, runtime


def _validation(root: Path, private: Path, started: float) -> list[Json]:
    """Run the frozen affected checks and the applicable numbered ARC E2E checks."""
    from carnot.reporting.experiment_7303_validation_scope import (
        CommandSpec,
        build_scoped_commands,
        run_commands,
    )

    basetemp = private / "pytest"
    basetemp.mkdir(parents=True, exist_ok=True)
    commands = build_scoped_commands(
        root,
        [TEST],
        [MODULE],
        static_paths=[WRAPPER],
        basetemp=basetemp,
        coverage_file=private / ".coverage",
    )
    python = str(root / ".venv/bin/python")
    pytest = str(root / ".venv/bin/pytest")
    checks = (
        (
            "e2e_009",
            (
                pytest,
                "tests/python/test_arc_induction_state_persistence.py",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                f"--basetemp={basetemp / 'e2e009'}",
            ),
        ),
        (
            "e2e_011",
            (
                pytest,
                "tests/python/test_arc_decision_telemetry.py",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                f"--basetemp={basetemp / 'e2e011'}",
            ),
        ),
        (
            "e2e_013",
            (
                pytest,
                "tests/python/test_arc_decision_telemetry.py",
                "tests/python/test_experiment_7491_e6_timed_live_profile.py",
                "tests/python/test_experiment_7531_b2_induction_gate_measurement.py",
                "tests/python/test_semif_arc_readout_eval.py",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                f"--basetemp={basetemp / 'e2e013'}",
            ),
        ),
        (
            "e2e_009_smoke",
            (
                python,
                "-u",
                "scripts/arc_loop_solve.py",
                "--mechanism",
                "e3",
                "--game",
                "r11l",
                "--max-actions",
                "12",
                "--output",
                str(private / "e2e009_smoke.json"),
            ),
        ),
    )
    for name, argv in checks:
        commands.append(CommandSpec(name, argv, "applicable_e2e", 900))
    progress(started, "validation", "before", commands=len(commands))
    receipts = run_commands(
        root,
        commands,
        log_dir=private / "logs",
        extra_env={"JAX_PLATFORMS": "cpu", "CARNOT_ARC_DISABLE_INDUCTION": "1"},
        heartbeat_s=45,
    )
    progress(
        started,
        "validation",
        "after",
        passed=sum(row["passed"] for row in receipts),
        total=len(receipts),
    )
    return receipts


def _terminal(root: Path, candidate: Path, private: Path, started: float) -> list[Json]:
    """Verify exact candidate bytes with a fresh reducer and two terminal readers."""
    from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands

    python = str(root / ".venv/bin/python")
    commands = [
        CommandSpec(
            "cold_reduction",
            (python, "-u", str(root / WRAPPER), "--cold-read", str(candidate)),
            "terminal_candidate",
            120,
        ),
        CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "terminal_candidate",
            300,
        ),
        CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "terminal_candidate",
            300,
        ),
    ]
    progress(started, "terminal_readers", "before", candidate=candidate)
    receipts = run_commands(root, commands, log_dir=private / "terminal_logs", heartbeat_s=45)
    progress(started, "terminal_readers", "after", exits=[item["exit_code"] for item in receipts])
    return receipts


def run_experiment(root: Path, run_date: str, output: Path) -> Json:
    """Preflight, measure, validate, cold-read and publish one terminal artifact."""
    root = root.resolve()
    started = time.monotonic()
    progress(started, "preflight", "before", root=root)
    if root != ROOT or run_date != "20260926":
        raise ValueError("absolute_root_and_run_date_required")
    private = Path(tempfile.mkdtemp(prefix="exp7709-"))
    spans: list[Json] = []
    phase_start = time.monotonic()
    checks, hashes, context = collect_preconditions(root, started)
    spans.append(
        {
            "phase": "preflight",
            "start_monotonic_s": phase_start,
            "end_monotonic_s": time.monotonic(),
            "completed_units": len(checks),
        }
    )
    progress(started, "preflight", "after", failed=sum(not row["passed"] for row in checks))
    rows: list[Json] = []
    runtime: Json = {}
    if all(row["passed"] for row in checks):
        phase_start = time.monotonic()
        progress(started, "live", "before", planned_episodes=2)
        try:
            rows, runtime = run_live(root, context, started)
            if runtime.get("gguf_sha256"):
                hashes["producer_files"]["model_weights"] = {
                    "path": runtime["model_path"],
                    "sha256": runtime["gguf_sha256"],
                }
            if runtime.get("server_sha256"):
                hashes["producer_files"]["native_runtime"] = {
                    "path": runtime["server_path"],
                    "sha256": runtime["server_sha256"],
                }
        except BaseException as exc:
            checks.append(
                gate_check(
                    "owned_cuda_lease",
                    "gpu_lease_phase_journal",
                    "/tmp/carnot-gpu-leases",
                    "current_lease_acquired",
                    True,
                    False,
                )
            )
            runtime = {"error": f"{type(exc).__name__}: {exc}"[:500]}
        spans.append(
            {
                "phase": "live",
                "start_monotonic_s": phase_start,
                "end_monotonic_s": time.monotonic(),
                "completed_units": len(rows),
                "checkpoint": str(RAW / "episode_rows.json"),
            }
        )
        progress(started, "live", "after", completed_units=len(rows), error=runtime.get("error"))
    raw = root / RAW
    raw.mkdir(parents=True, exist_ok=True)
    if not (raw / "episode_rows.json").is_file():
        atomic_json(raw / "episode_rows.json", {"rows": rows})
    reduction = None
    if len(rows) == 2:
        try:
            reduction = cold_reduce(raw / "episode_rows.json")
        except ValueError as exc:
            checks.append(
                gate_check(
                    "action_observation_invocation_join",
                    "exp7709_raw_rows",
                    str(raw / "episode_rows.json"),
                    "cold_reduction_valid",
                    True,
                    str(exc),
                )
            )
    phase_start = time.monotonic()
    receipts = _validation(root, private, started)
    spans.append(
        {
            "phase": "validation",
            "start_monotonic_s": phase_start,
            "end_monotonic_s": time.monotonic(),
            "completed_units": len(receipts),
        }
    )
    candidate = build_artifact(
        checks=checks,
        rows=rows,
        reduction=reduction,
        hashes=hashes,
        receipts=receipts,
        duration_s=time.monotonic() - started,
        run_date=run_date,
        runtime=runtime,
        spans=spans,
    )
    candidate["frozen_validation_scope"] = {
        "tests": [TEST],
        "changed_modules": [MODULE],
        "static_paths": [WRAPPER],
        "e2e": ["E2E-009", "E2E-011", "E2E-013"],
    }
    candidate_path = raw / "terminal_candidate.json"
    atomic_json(candidate_path, candidate)
    terminal = _terminal(root, candidate_path, private, started)
    if any(item["exit_code"] != 0 for item in terminal):
        candidate = build_artifact(
            checks=checks,
            rows=rows,
            reduction=reduction,
            hashes=hashes,
            receipts=[*receipts, *terminal],
            duration_s=time.monotonic() - started,
            run_date=run_date,
            runtime=runtime,
            spans=spans,
            flagged_adversarial=any(
                item["name"] == "adversarial_verify" and item["exit_code"] != 0 for item in terminal
            ),
        )
        candidate["frozen_validation_scope"] = {
            "tests": [TEST],
            "changed_modules": [MODULE],
            "static_paths": [WRAPPER],
            "e2e": ["E2E-009", "E2E-011", "E2E-013"],
        }
        atomic_json(candidate_path, candidate)
        terminal = _terminal(root, candidate_path, private, started)
    atomic_json(
        raw / "validation_receipts.json",
        {
            "candidate_sha256": sha256_file(candidate_path),
            "validation_receipts": receipts,
            "terminal_reader_receipts": terminal,
        },
    )
    candidate["terminal_reader_receipts_path"] = str(RAW / "validation_receipts.json")
    atomic_json(output, candidate)
    progress(started, "publication", "after", verdict=candidate["honest_verdict"], output=output)
    return candidate


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260926")
    parser.add_argument("--output", default=str(RESULT))
    parser.add_argument("--cold-read", type=Path)
    args = parser.parse_args(argv)
    if args.cold_read:
        candidate = json.loads(args.cold_read.read_text())
        raw_rows = ROOT / RAW / "episode_rows.json"
        raw_value = json.loads(raw_rows.read_text())
        if raw_value.get("rows") != candidate.get("rows"):
            raise ValueError("cold_raw_rows_mismatch")
        observed = cold_reduce(raw_rows) if len(raw_value["rows"]) == 2 else None
        if candidate.get("reduction") != observed:
            raise ValueError("cold_reduction_mismatch")
        print(json.dumps(observed, sort_keys=True), flush=True)
        return 0
    run_experiment(ROOT, args.date, ROOT / args.output)
    return 0

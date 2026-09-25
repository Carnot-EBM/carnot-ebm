"""Run a bounded, adapter-withheld ARC paired case series.

The experiment uses the shipped E3 policy and a current local generator. Its
three public games are diagnostic cases, not hidden-game solve credit.
Spec: REQ-REPORT-7653 and SCENARIO-REPORT-7653-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, sha256_file

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
EXPERIMENT = "experiment_7653_v667_arc_live_generalization"
RESULT = Path("results/experiment_7653_v667_arc_live_generalization.json")
RAW = Path("results/raw/experiment_7653_v667_arc_live_generalization")
SPEC = Path("openspec/capabilities/research-reporting/spec.md")
MODULE = Path("python/carnot/experiment_7653_v667_arc_live_generalization.py")
WRAPPER = Path("scripts/experiments/experiment_7653_v667_arc_live_generalization.py")
TEST = Path("tests/python/test_experiment_7653_v667_arc_live_generalization.py")
SEED = 7_653_667
SALT = "v667-arc-live-generalization-20260925"
ARMS = ("baseline", "hud_dedup")


def canonical_hash(value: Any) -> str:
    """Bind identities and reduction inputs to stable JSON bytes."""

    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


def select_games(roster: Iterable[str], *, salt: str = SALT) -> list[str]:
    """Select three distinct SDK game IDs before any outcome is inspected."""

    games = sorted(set(roster), key=lambda game: (canonical_hash([salt, game]), game))
    if len(games) < 3:
        raise ValueError("three_sdk_games_required")
    return games[:3]


def build_schedule(games: Sequence[str], *, seed: int = SEED) -> list[Json]:
    """Seal one matched seed per game and the only allowed arm difference."""

    if len(games) != 3 or len(set(games)) != 3:
        raise ValueError("three_distinct_games_required")
    return [
        {
            "episode_id": f"{game}:{arm}:{seed}",
            "game": game,
            "arm": arm,
            "seed": seed,
            "max_seconds": 400,
            "max_actions": 128,
            "max_inductions": 1,
            "max_output_tokens": 4096,
            "max_engine_calls": 20000,
            "adapter_withheld": True,
        }
        for game in games
        for arm in ARMS
    ]


def gate_check(
    check: str,
    *,
    upstream: str,
    path: str,
    field: str,
    operator: str,
    expected: Any,
    observed: Any,
) -> Json:
    """Record each input or resource comparison with exact operands."""

    passed = {"eq": lambda: observed == expected, "ge": lambda: observed >= expected}[operator]()
    return {
        "check": check,
        "upstream": upstream,
        "path": path,
        "field": field,
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": passed,
    }


def _gate(passed: bool, principle: str, **operands: Any) -> Json:
    return {"passed": passed, "principle": principle, "measured_operands": operands}


def blocked_artifact(
    checks: Sequence[Mapping[str, Any]], *, source_hashes: Mapping[str, Any], duration_s: float
) -> Json:
    """Make a complete external block with no fictitious current invocation."""

    failed = [dict(row) for row in checks if row.get("passed") is not True]
    reason = str(failed[0]["check"]).replace(":", "_") if failed else "unknown_input"
    artifact: Json = {
        "schema": "carnot.exp7653.v667.arc_live_generalization.v1",
        "experiment_id": 7653,
        "milestone": "2026.09.667",
        "run_date": "20260925",
        "status": "complete",
        "honest_verdict": f"complete_blocked_{reason}",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "gate_check_summary": {"failed_checks": failed},
        "preconditions_checked": [dict(row) for row in checks],
        "MODEL_SPECS": [],
        "planned_MODEL_SPECS": [MODEL_ID],
        "model_invoked": False,
        "invocation_counts": {"loads": 0, "forwards": 0, "generations": 0, "tokens": 0},
        "inference_substrate": "host_preflight_no_model_call",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "model_full_generation",
        "execution_venue": "host",
        "execution_venue_details": {"host": socket.gethostname(), "owned_pid": os.getpid()},
        "duration_s": duration_s,
        "phase_spans": [],
        "random_seed": {"game_order_salt": SALT, "matched_episode_seed": SEED},
        "source_artifact_hashes": dict(source_hashes),
        "rows": [],
        "per_game_results": [],
        "sample_size_budget": {
            "intended_independent_games": 3,
            "observed_independent_games": 0,
            "intended_episodes": 6,
            "observed_episodes": 0,
        },
        "live_measurement_complete_score": 0,
        "solve_provenance": "no_live_attempt",
        "current_induction_counts": {"attempted": 0, "complete": 0, "accepted": 0, "executed": 0},
        "historical_induction_counts": {},
        "gpu_lease_receipt": None,
        "verifier_is_oracle": False,
        "validation_receipts": [],
    }
    artifact["acceptance_gate_results"] = {
        "validity": _gate(
            False, "Required external operands must be present and authentic.", failed=failed
        ),
        "readiness": _gate(False, "A blocked run has no measured readiness.", checks=len(checks)),
        "probability_benefit": _gate(
            False, "Public case series cannot estimate hidden-game probability.", games=0
        ),
        "utility": _gate(False, "Benefit and cost require live paired outcomes.", episodes=0),
        "retention": _gate(False, "No retained learning is measured by an absent run.", groups=0),
        "freshness": _gate(False, "No current generation occurred.", generations=0),
    }
    artifact["field_principles"] = field_principles(artifact)
    artifact["reproducibility_checksum"] = canonical_hash([source_hashes, SALT, SEED, failed])
    return artifact


def reduce_rows(rows: Sequence[Mapping[str, Any]], schedule: Sequence[Mapping[str, Any]]) -> Json:
    """Rebuild paired outcomes from six raw units, never from stored scores."""

    expected = [str(unit["episode_id"]) for unit in schedule]
    actual = [str(row["episode_id"]) for row in rows]
    if len(expected) != 6 or actual != expected:
        raise ValueError("six_episode_accounting")
    by_game: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        by_game.setdefault(str(row["game"]), {})[str(row["arm"])] = row
    if len(by_game) != 3 or any(set(pair) != set(ARMS) for pair in by_game.values()):
        raise ValueError("three_paired_games_required")
    deltas = {
        game: int(pair["hud_dedup"].get("peak_level") or 0)
        - int(pair["baseline"].get("peak_level") or 0)
        for game, pair in by_game.items()
    }
    counts = {
        "attempted": sum(bool(row.get("induction_attempted")) for row in rows),
        "complete": sum(bool(row.get("induction_completed")) for row in rows),
        "accepted": sum(bool(row.get("engine_accepted")) for row in rows),
        "executed": sum(bool(row.get("plan_executed")) for row in rows),
    }
    return {
        "intended_independent_games": 3,
        "observed_independent_games": 3,
        "intended_episodes": 6,
        "observed_episodes": 6,
        "eligible_episodes": sum(row.get("exclusion") is None for row in rows),
        "excluded_episodes": sum(row.get("exclusion") is not None for row in rows),
        "censored_episodes": sum(bool(row.get("censored")) for row in rows),
        "paired_level_deltas": deltas,
        "current_induction_counts": counts,
        "planner_lever_reachable": counts["accepted"] > 0 and counts["executed"] > 0,
        "action_limit_per_episode": 128,
        "induction_limit_per_episode": 1,
        "output_token_limit_per_induction": 4096,
        "planner_engine_call_limit_per_episode": 20000,
        "episode_wall_limit_s": 400,
        "episode_budget_s": 2400,
        "validation_budget_s": 1200,
    }


def cold_reduce(path: Path) -> Json:
    """Read persisted rows in a fresh process and reject a forged reduction."""

    artifact = json.loads(path.read_text(encoding="utf-8"))
    reduction = reduce_rows(artifact["rows"], artifact["schedule"])
    if artifact.get("reduction") != reduction:
        raise ValueError("reduction_mismatch")
    return reduction


def field_principles(artifact: Mapping[str, Any]) -> Json:
    """Keep the governing claim beside every terminal field."""

    specific = {
        "honest_verdict": "Completion is separate from scientific benefit.",
        "verdict_class": "A closed verdict class prevents readiness from claiming benefit.",
        "flagged_adversarial": "Flagged evidence cannot open a downstream gate.",
        "gate_check_summary": "A blocked claim must expose exact failed operands.",
        "acceptance_gate_results": "Validity, readiness, probability, utility, retention and freshness are distinct.",
        "rows": "Independent units and arms retain absolute observed quantities and censoring.",
        "sample_size_budget": "Repeated views and seeds do not increase the three-game sample.",
        "preconditions_checked": "Only actual inputs and resources can be preconditions.",
        "inference_substrate": "Current inference cannot be inherited from historical runs.",
        "inference_substrate_class": "Full generation requires real discovery, not a token canary.",
        "MODEL_SPECS": "Only identities invoked during current work belong here.",
        "model_invoked": "A current load attempt is distinct from a planned load.",
        "execution_venue": "Host is a closed execution venue; device details are separate.",
        "phase_spans": "Monotonic disjoint stages expose completed work.",
        "random_seed": "The game salt and matched episode seed are frozen before outcomes.",
        "reproducibility_checksum": "A stable digest binds inputs, schedule and reduction code.",
        "source_artifact_hashes": "Producers and pre-gate receipts are separate from planned outputs.",
        "validation_receipts": "Actual child exits and log hashes govern validity.",
        "verifier_is_oracle": "Exact structural truth does not show oracle-distinct learned gain.",
        "live_measurement_complete_score": "One needs all six rows, including censored or no-plan cases.",
        "per_game_results": "Each game keeps its official levels, actions and compute.",
        "solve_provenance": "Only own-attempt runtime induction has live discovery provenance.",
        "current_induction_counts": "Attempted, completed, accepted and executed are distinct.",
        "gpu_lease_receipt": "A current lease binds physical UUID, owner and offload.",
    }
    return {
        key: specific.get(key, "Current measured operands govern this field.") for key in artifact
    }


def _hash_input(root: Path, relative: Path) -> Json:
    path = root / relative
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _input_check(root: Path, relative: Path) -> Json:
    path = root / relative
    return gate_check(
        f"named_input:{relative.as_posix()}",
        upstream="declared_task_input",
        path=str(path),
        field="readable_nonempty_file",
        operator="eq",
        expected=True,
        observed=path.is_file() and path.stat().st_size > 0,
    )


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Flush every boundary and long-running wait to the operator."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7653] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f} {suffix}",
        flush=True,
    )


def collect_preconditions(
    root: Path, started: float
) -> tuple[list[Json], Json, Json]:  # pragma: no cover
    """Authenticate external inputs and inspect current resources before model load."""

    import yaml

    from carnot.experiment_7630_v666_cuda_ownership import (
        ProcessRegistry,
        _current_inventory,
        select_owned_capacity,
    )
    from carnot.inference.sota_models import cached_current_model

    named = (
        Path("AGENTS.md"),
        Path("CODEX.md"),
        Path("CLAUDE.md"),
        Path("research-program.md"),
        Path("ops/exclusion_manifest.yaml"),
        Path("ops/e2e-test-plan.md"),
        Path("scripts/experiment_template.py"),
        Path("python/carnot/reporting/current_work_receipt.py"),
        Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
        Path("python/carnot/agentic/arc_competition_agent.py"),
        Path("python/carnot/agentic/arc_executable_world_model.py"),
        Path("python/carnot/experiment_7589_v663_arc_output_boundary.py"),
        Path("python/carnot/experiment_7630_v666_cuda_ownership.py"),
        Path("ops/arc_solve_registry.yaml"),
        Path("results/experiment_7645_v667_arc_validation_requalification.json"),
        SPEC,
        MODULE,
        WRAPPER,
        TEST,
    )
    checks = [_input_check(root, path) for path in named]
    hashes: Json = {
        "producer_files": {
            str(path): _hash_input(root, path) for path in named if (root / path).is_file()
        },
        "pre_gate_receipts": {},
        "missing_inputs": [row["path"] for row in checks if not row["passed"]],
        "planned_outputs": [str(root / RESULT)],
    }
    spec_text = (root / SPEC).read_text(encoding="utf-8") if (root / SPEC).is_file() else ""
    checks.append(
        gate_check(
            "driving_requirement",
            upstream="research_reporting_spec",
            path=str(root / SPEC),
            field="REQ-*",
            operator="eq",
            expected="REQ-REPORT-7653",
            observed="REQ-REPORT-7653" if "REQ-REPORT-7653" in spec_text else None,
        )
    )
    guard_path = root / "results/experiment_7645_v667_arc_validation_requalification.json"
    guard = json.loads(guard_path.read_text(encoding="utf-8")) if guard_path.is_file() else {}
    for field, expected in (
        ("verdict_class", "null"),
        ("flagged_adversarial", False),
        ("planner_goal_guard_ready_score", 1),
    ):
        checks.append(
            gate_check(
                f"exp7645:{field}",
                upstream="experiment_7645",
                path=str(guard_path),
                field=field,
                operator="eq",
                expected=expected,
                observed=guard.get(field),
            )
        )
    if guard_path.is_file():
        hashes["pre_gate_receipts"]["experiment_7645"] = _hash_input(
            root, guard_path.relative_to(root)
        )
    registry_path = root / "ops/arc_solve_registry.yaml"
    registry = (
        yaml.safe_load(registry_path.read_text(encoding="utf-8")) if registry_path.is_file() else {}
    )
    checks.append(
        gate_check(
            "registry_historical_only",
            upstream="arc_solve_registry",
            path=str(registry_path),
            field="games_list_present",
            operator="eq",
            expected=True,
            observed=isinstance(registry.get("games"), list),
        )
    )
    progress(started, "sdk_roster", "before")
    from carnot.agentic.arc_solver_kit import offline_arcade

    arcade = offline_arcade()
    roster = sorted({str(item.game_id).split("-")[0] for item in arcade.available_environments})
    progress(started, "sdk_roster", "after", count=len(roster))
    checks.append(
        gate_check(
            "sdk_roster_three_games",
            upstream="arc_agi_sdk",
            path=str(root / "environment_files"),
            field="distinct_supported_games",
            operator="ge",
            expected=3,
            observed=len(roster),
        )
    )
    model = cached_current_model()
    model_path = Path(str(model["model_path"])).resolve() if model else Path("/missing/model.gguf")
    checks.append(
        gate_check(
            "cached_mandated_model",
            upstream="cached_current_model",
            path=str(model_path),
            field="nonempty_Q4_K_M_GGUF",
            operator="eq",
            expected=True,
            observed=model is not None
            and model_path.is_file()
            and model_path.stat().st_size > 15_000_000_000,
        )
    )
    server = (Path.home() / ".cache/llama.cpp-master/build/bin/llama-server").resolve()
    checks.append(
        gate_check(
            "native_runtime",
            upstream="installed_cuda_llama_cpp",
            path=str(server),
            field="cuda_binary_and_backend_present",
            operator="eq",
            expected=True,
            observed=(
                server.is_file()
                and os.access(server, os.X_OK)
                and (server.parent / "libggml-cuda.so").is_file()
            ),
        )
    )
    registry_owner = ProcessRegistry.current()
    registry_owner.task_id = EXPERIMENT
    inventory = _current_inventory()
    selected, ownership_rows = select_owned_capacity(inventory, registry_owner)
    checks.append(
        gate_check(
            "exclusive_cuda_capacity",
            upstream="current_gpu_inventory",
            path="nvidia-smi",
            field="exclusive_device_available",
            operator="eq",
            expected=True,
            observed=selected is not None,
        )
    )
    return (
        checks,
        hashes,
        {
            "roster": roster,
            "registry": registry,
            "model_path": model_path,
            "server": server,
            "inventory": inventory,
            "selected_gpu": selected,
            "ownership_rows": ownership_rows,
            "process_registry": registry_owner,
        },
    )


def run_episode(
    unit: Mapping[str, Any], proposer: Any, capture: Any, raw: Path, started: float
) -> Json:  # pragma: no cover - real ARC SDK and generator boundary.
    """Drive the scored E3 choice path and retain every action checkpoint."""

    import functools

    from carnot import experiment_7471_v654_arc_seam_observation as live
    from carnot.agentic import arc_executable_world_model as world

    previous = (live.ACTION_LIMIT, live.EPISODE_LIMIT_S, live.REQUEST_LIMIT)
    old_flag = os.environ.get("CARNOT_ARC_PLAN_HUD_DEDUP")
    original_plan = world.plan_in_model
    live.ACTION_LIMIT = 128
    live.EPISODE_LIMIT_S = 400.0
    live.REQUEST_LIMIT = 1
    os.environ["CARNOT_ARC_PLAN_HUD_DEDUP"] = "1" if unit["arm"] == "hud_dedup" else "0"
    plan_rows: list[Json] = []
    engine_calls = 0
    limit_hit = False

    @functools.wraps(original_plan)
    def observed_plan(engine: Any, goal: Any, grid: Any, **kwargs: Any) -> Any:
        nonlocal engine_calls, limit_hit
        diagnostics = kwargs.setdefault("diagnostics", {})

        def counted_engine(*args: Any, **inner_kwargs: Any) -> Any:
            nonlocal engine_calls, limit_hit
            if engine_calls >= 20000:
                limit_hit = True
                raise RuntimeError("planner_engine_call_budget_exhausted")
            engine_calls += 1
            return engine(*args, **inner_kwargs)

        progress(started, "planner", "before", episode_id=unit["episode_id"])
        try:
            plan = original_plan(
                counted_engine,
                goal,
                grid,
                **{
                    **kwargs,
                    "max_nodes": min(int(kwargs.get("max_nodes", 20000)), 20000 - engine_calls),
                },
            )
        except RuntimeError as exc:
            if str(exc) != "planner_engine_call_budget_exhausted":
                raise
            plan = None
        plan_rows.append(
            {
                "plan": deepcopy(plan),
                "diagnostics": deepcopy(diagnostics),
                "engine_calls_cumulative": engine_calls,
            }
        )
        progress(
            started,
            "planner",
            "after",
            episode_id=unit["episode_id"],
            engine_calls=engine_calls,
            plan_length=len(plan) if plan else 0,
        )
        return plan

    world.plan_in_model = observed_plan
    try:
        progress(started, "episode", "before_benchmark", episode_id=unit["episode_id"])
        prior_requests = capture.indices[str(unit["episode_id"])]
        source = live._run_policy_episode(
            unit, proposer, capture, raw / "runtime_events.jsonl", raw / "actions.jsonl"
        )
        progress(
            started,
            "episode",
            "after_benchmark",
            episode_id=unit["episode_id"],
            actions=source["action_count"],
            disposition=source["disposition"],
        )
    finally:
        world.plan_in_model = original_plan
        live.ACTION_LIMIT, live.EPISODE_LIMIT_S, live.REQUEST_LIMIT = previous
        if old_flag is None:
            os.environ.pop("CARNOT_ARC_PLAN_HUD_DEDUP", None)
        else:
            os.environ["CARNOT_ARC_PLAN_HUD_DEDUP"] = old_flag
    seam_path = Path(source["seam_event_path"])
    seams = live.read_jsonl(seam_path)
    hypothesis = [row for row in seams if row.get("seam") == "hypothesis_gate"]
    start_events = [row for row in hypothesis if row.get("event") == "stage_start"]
    end_events = [row for row in hypothesis if row.get("event") == "stage_end"]
    accepted = any(row.get("gate_decision") == "accept" for row in hypothesis)
    plan = next((row["plan"] for row in plan_rows if row["plan"]), None)
    action_rows = source["action_rows"]
    post_induction = [
        row
        for row in action_rows
        if end_events and row["action_index"] > end_events[0]["action_count"]
    ]
    executed = bool(accepted and plan and post_induction)
    supervisor_firings = sum(
        row.get("supervisor_fired") is True
        for row in seams
        if row.get("seam") == "supervisor_arm_selection" and row.get("event") == "selection"
    )
    response_rows = source["server_request_rows"]
    tokens = 0
    for request in response_rows:
        response_path = Path(str(request.get("response_path") or ""))
        if response_path.is_file():
            response = json.loads(response_path.read_text(encoding="utf-8"))
            usage = response.get("usage") or {}
            tokens += int(usage.get("completion_tokens") or response.get("tokens_predicted") or 0)
    return {
        **dict(unit),
        "start_level": source["start_level"],
        "peak_level": source["peak_level"],
        "terminal_level": source["terminal_level"],
        "actions_used": source["action_count"],
        "actions": action_rows,
        "engine_calls": engine_calls,
        "planner_rows": plan_rows,
        "selected_plan": plan,
        "executed_actions_after_induction": post_induction,
        "induction_attempted": bool(
            start_events or capture.indices[str(unit["episode_id"])] > prior_requests
        ),
        "induction_completed": bool(end_events),
        "engine_accepted": accepted,
        "plan_executed": executed,
        "current_output_tokens": tokens,
        "request_rows": response_rows,
        "supervisor_firings": supervisor_firings,
        "censored": source["disposition"].startswith("censored") or limit_hit,
        "censor_reason": "planner_engine_calls"
        if limit_hit
        else source["disposition"]
        if source["disposition"].startswith("censored")
        else None,
        "exclusion": None,
        "raw_provenance": "live_agent_self_discovery",
        "action_log": str(raw / "actions.jsonl"),
        "seam_log": str(seam_path),
        "official_reproduction": source["trace_reproduction"],
        "policy_entry": source["policy_entry"],
        "error": source["error"],
    }


def run_live(
    root: Path, context: Mapping[str, Any], schedule: Sequence[Mapping[str, Any]], started: float
) -> tuple[list[Json], Json, Json]:  # pragma: no cover - current CUDA and real SDK boundary.
    """Own one CUDA server for all six bounded episodes and release it once."""

    import threading

    from carnot import experiment_7471_v654_arc_seam_observation as live
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer
    from carnot.experiment_7630_v666_cuda_ownership import (
        _current_inventory,
        recheck_before_launch,
    )
    from carnot.experiment_7581_v662_arc_bounded_canary import (
        _observed_offload_layers,
        _owned_vram_mb,
    )
    from carnot.gpu_lease_phase_journal import GpuLease, LeaseBusy, RecoveryError

    raw = root / RAW
    raw.mkdir(parents=True, exist_ok=True)
    gpu = dict(context["selected_gpu"])
    model_path = Path(context["model_path"])
    server = Path(context["server"])
    progress(started, "model_hash", "before", bytes=model_path.stat().st_size)
    model_hash = sha256_file(model_path)
    progress(started, "model_hash", "after", sha256=model_hash)
    runtime_hash = sha256_file(server)
    begun = time.monotonic()
    while True:
        progress(started, "gpu_lease", "before", uuid=gpu["uuid"])
        try:
            lease = GpuLease.acquire(
                runtime_dir=Path("/tmp/carnot-gpu-leases"),
                task_id=EXPERIMENT,
                device_uuid=str(gpu["uuid"]),
                expected_model=str(model_path),
                vram_before_mb=int(gpu.get("memory_used_mb") or 0),
                ttl_s=120,
            )
            break
        except (LeaseBusy, RecoveryError):
            if time.monotonic() - begun >= 180:
                raise LeaseBusy("exclusive_cuda_lease_wait_timeout") from None
            progress(started, "gpu_lease", "wait", elapsed_s=time.monotonic() - begun)
            time.sleep(min(15.0, 180.0 - (time.monotonic() - begun)))
    progress(started, "gpu_lease", "after", lease_id=lease.lease_id)
    stop_heartbeats = threading.Event()

    def heartbeat() -> None:
        while not stop_heartbeats.wait(45):
            try:
                lease.heartbeat()
                progress(started, "owned_wait", "heartbeat")
            except Exception as exc:
                progress(started, "owned_wait", "heartbeat_error", error=str(exc))

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    proposer: Any = None
    capture: Any = None
    rows: list[Json] = []
    runtime: Json = {
        "model_path": str(model_path),
        "model_sha256": model_hash,
        "runtime_path": str(server),
        "runtime_sha256": runtime_hash,
        "gpu_uuid": gpu["uuid"],
        "owner_pid": os.getpid(),
        "lease_owner": lease.owner_receipt(),
        "model_loaded": False,
        "load_attempted": False,
        "error": None,
    }
    old_env = dict(os.environ)
    try:
        check = recheck_before_launch(
            str(gpu["uuid"]),
            [_current_inventory(), _current_inventory()],
            context["process_registry"],
        )
        runtime["prelaunch_recheck"] = check
        if check["passed"] is not True:
            raise RuntimeError("foreign_or_capacity_recheck_failed")
        from carnot.experiment_7431_v651_arc_live_sentinel import _free_port

        port = _free_port()
        environment = live.session_environment(
            os.environ, gpu_index=int(gpu["index"]), port=port, raw_dir=raw
        )
        os.environ.update(environment)
        os.environ.update(
            {
                "CARNOT_ARC_INDUCE_MAX_TOKENS": "4096",
                "CARNOT_ARC_INDUCE_N_CTX": "32768",
                "CARNOT_ARC_INDUCE_TIMEOUT": "400",
                "CARNOT_ARC_GENERATOR_REQUIRE_CUDA": "1",
                "CARNOT_ARC_MAX_REFINEMENT_ROUNDS": "1",
                "CARNOT_ARC_INDUCE_TOOL_TURNS": "1",
                "CARNOT_ARC_GGUF_PATH": str(model_path),
                "CARNOT_LLAMA_SERVER": str(server),
                "CARNOT_ARC_SERVER_LOG_DIR": str(raw / "server_logs"),
            }
        )
        lease.transition("admitted")
        lease.transition("loading")
        progress(started, "model_load", "before", model=MODEL_ID)
        runtime["load_attempted"] = True
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
            timeout=400,
            tries=1,
        )
        proposer.model_repository = MODEL_ID
        proposer.requested_model_path = str(model_path)
        healthy = proposer._ensure_server()
        server_pid = getattr(getattr(proposer, "_proc", None), "pid", None)
        runtime["model_loaded"] = bool(healthy)
        runtime["server_pid"] = server_pid
        runtime["server_pid_start_ticks"] = (
            __import__(
                "carnot.gpu_lease_phase_journal", fromlist=["proc_start_ticks"]
            ).proc_start_ticks(server_pid)
            if server_pid
            else None
        )
        runtime["server_command"] = list(getattr(proposer, "last_launch_argv", ()) or ())
        log_path = Path(proposer._stderr_log_path) if proposer._stderr_log_path else None
        runtime["offload_layers"] = _observed_offload_layers(log_path)
        runtime["owned_server_vram_mb"] = _owned_vram_mb(server_pid)
        progress(
            started,
            "model_load",
            "after",
            healthy=healthy,
            server_pid=server_pid,
            owned_vram_mb=runtime["owned_server_vram_mb"],
        )
        if not healthy or not runtime["owned_server_vram_mb"]:
            raise RuntimeError("current_server_offload_unverified")
        lease.transition("resident", vram_mb=int(runtime["owned_server_vram_mb"]))
        lease.transition("inferencing")
        capture = live.live_support.DurableRequestCapture(
            raw, raw / "runtime_events.jsonl", max_new_tokens=4096
        )
        capture.install()
        for unit in schedule:
            row = run_episode(unit, proposer, capture, raw, started)
            rows.append(row)
            atomic_json(raw / "episode_rows.json", {"rows": rows})
            progress(started, "checkpoint", "after", completed_units=len(rows))
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
        stop_heartbeats.set()
        thread.join(timeout=1)
        if runtime["model_loaded"] and runtime.get("owned_server_vram_mb"):
            lease.transition("unloading")
            lease.transition("validating", vram_mb=0, exit_code=0, unload_observed=True)
            lease.transition("terminal_complete")
        else:
            lease.transition("terminal_blocked")
        runtime["lease_release"] = lease.release()
    return rows, runtime, {"model_sha256": model_hash, "runtime_sha256": runtime_hash}


def build_live_artifact(
    checks: Sequence[Mapping[str, Any]],
    hashes: Json,
    schedule: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    runtime: Mapping[str, Any],
    duration_s: float,
) -> Json:
    """Keep live evidence and invalid or unfinished work in distinct verdicts."""

    complete = len(rows) == 6
    reduction = reduce_rows(rows, schedule) if complete else None
    requests = [request for row in rows for request in row.get("request_rows", [])]
    generations = sum(bool(request.get("request_dispatched")) for request in requests)
    tokens = sum(int(row.get("current_output_tokens") or 0) for row in rows)
    invoked = bool(runtime.get("load_attempted"))
    full = generations > 0
    substrate_class = (
        "model_full_generation"
        if full
        else "model_load_no_generation"
        if invoked
        else "no_model_load"
    )
    if runtime.get("error") or not complete:
        verdict = (
            "complete_disqualified_live_execution_incomplete"
            if invoked
            else "complete_blocked_exclusive_cuda_lease"
        )
        verdict_class = "disqualified" if invoked else "blocked"
    elif reduction and not reduction["planner_lever_reachable"]:
        verdict, verdict_class = "complete_null_planner_reachability", "null"
    else:
        verdict, verdict_class = "complete_null_exploratory_public_case_series", "null"
    failed_checks = [dict(check) for check in checks if not check["passed"]]
    if verdict_class == "blocked":
        failed_checks.append(
            gate_check(
                "current_prelaunch_recheck",
                upstream="experiment_7630_cuda_ownership",
                path="/tmp/carnot-gpu-leases",
                field="model_launch_admitted",
                operator="eq",
                expected=True,
                observed=False,
            )
        )
    gates = {
        "validity": _gate(
            complete and bool(runtime.get("owned_server_vram_mb")),
            "Six current SDK episodes and measured offload establish validity.",
            episodes=len(rows),
            owned_vram_mb=runtime.get("owned_server_vram_mb"),
        ),
        "readiness": _gate(
            False, "Required validation and terminal readers govern readiness.", receipts=0
        ),
        "probability_benefit": _gate(
            False,
            "Three public games cannot estimate hidden-game benefit.",
            public_games=3,
            hidden_games=0,
        ),
        "utility": _gate(
            False,
            "Observed action and compute cost needs independent outcome benefit.",
            paired_level_deltas=reduction["paired_level_deltas"] if reduction else {},
        ),
        "retention": _gate(
            False, "A one-seed case series does not measure retained learning.", groups=0
        ),
        "freshness": _gate(
            full, "Current generation is counted from dispatched requests.", generations=generations
        ),
    }
    per_game = [
        {
            "game": game,
            "arms": [dict(row) for row in rows if row["game"] == game],
            "registry_credit_granted": False,
        }
        for game in dict.fromkeys(str(unit["game"]) for unit in schedule)
    ]
    artifact: Json = {
        "schema": "carnot.exp7653.v667.arc_live_generalization.v1",
        "experiment_id": 7653,
        "milestone": "2026.09.667",
        "run_date": "20260925",
        "status": "complete",
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": False,
        "gate_check_summary": {
            "failed_checks": failed_checks,
            "reachability": reduction["planner_lever_reachable"] if reduction else False,
            "runtime_error": runtime.get("error"),
        },
        "acceptance_gate_results": gates,
        "preconditions_checked": [dict(check) for check in checks],
        "MODEL_SPECS": [MODEL_ID] if invoked else [],
        "planned_MODEL_SPECS": [MODEL_ID],
        "model_invoked": invoked,
        "invocation_counts": {
            "loads": int(invoked),
            "forwards": generations,
            "generations": generations,
            "tokens": tokens,
        },
        "inference_substrate": "current_owned_CUDA_local_GGUF_E3"
        if invoked
        else "host_preflight_no_model_call",
        "inference_substrate_class": substrate_class,
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
        "random_seed": {"game_order_salt": SALT, "matched_episode_seed": SEED},
        "source_artifact_hashes": hashes,
        "schedule": [dict(unit) for unit in schedule],
        "rows": [dict(row) for row in rows],
        "reduction": reduction,
        "sample_size_budget": reduction
        or {
            "intended_independent_games": 3,
            "intended_episodes": 6,
            "observed_episodes": len(rows),
        },
        "live_measurement_complete_score": int(complete),
        "per_game_results": per_game,
        "solve_provenance": "live_agent_self_discovery" if rows else "no_live_attempt",
        "current_induction_counts": reduction["current_induction_counts"]
        if reduction
        else {"attempted": 0, "complete": 0, "accepted": 0, "executed": 0},
        "historical_induction_counts": {},
        "gpu_lease_receipt": dict(runtime),
        "verifier_is_oracle": False,
        "validation_receipts": [],
        "production_defaults_changed": False,
        "registry_credit_granted": False,
        "external_publication_authorized": False,
    }
    artifact["field_principles"] = field_principles(artifact)
    artifact["reproducibility_checksum"] = canonical_hash(
        [
            hashes,
            schedule,
            artifact["reduction"],
            sha256_file(ROOT / MODULE),
        ]
    )
    return artifact


def validation_commands(root: Path, private: Path) -> list[Any]:
    """Freeze affected files and private serial validation locations."""

    from carnot.reporting import experiment_7303_validation_scope as validation

    parent = private / "pytest"
    parent.mkdir(parents=True, exist_ok=True)
    commands = validation.build_scoped_commands(
        root,
        [TEST.as_posix()],
        [MODULE.as_posix()],
        static_paths=[WRAPPER.as_posix()],
        basetemp=parent,
        coverage_file=private / ".coverage.exp7653",
    )
    commands.append(
        validation.CommandSpec(
            "full_python_suite",
            (
                str(root / ".venv/bin/pytest"),
                "tests/python",
                "-q",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                f"--basetemp={parent / 'full'}",
            ),
            "repository_python_tests",
            1200,
        )
    )
    return commands


def terminal_commands(root: Path, candidate: Path) -> list[Any]:
    """Run fresh readers and strict guards against one candidate path."""

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
            (python, "-u", "scripts/adversarial_verify.py", "--json", str(candidate)),
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


def _reader_main(path: Path) -> int:
    """Cold-read one terminal candidate without loading a model."""

    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("verdict_class") == "blocked" and not artifact.get("rows"):
        failed = (artifact.get("gate_check_summary") or {}).get("failed_checks") or []
        if not failed or any(
            not all(
                key in row
                for key in (
                    "check",
                    "upstream",
                    "path",
                    "field",
                    "operator",
                    "expected",
                    "observed",
                )
            )
            for row in failed
        ):
            raise ValueError("blocked_gate_operands_missing")
        print(
            json.dumps(
                {"blocked_checks": len(failed), "current_calls": artifact["invocation_counts"]}
            ),
            flush=True,
        )
        return 0
    result = cold_reduce(path)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


def run_experiment(root: Path, run_date: str, output: Path) -> int:  # pragma: no cover
    """Preflight, measure, validate, cold-read, and atomically publish."""

    from carnot.reporting import experiment_7303_validation_scope as validation

    started = time.monotonic()
    progress(started, "preflight", "before", root=root)
    if root != ROOT or run_date != "20260925":
        raise ValueError("absolute_root_and_run_date_required")
    private = Path(tempfile.mkdtemp(prefix="exp7653-"))
    checks, hashes, context = collect_preconditions(root, started)
    progress(started, "preflight", "after", failed=sum(not row["passed"] for row in checks))
    spans: list[Json] = [
        {
            "phase": "preflight",
            "start_s": 0.0,
            "end_s": time.monotonic() - started,
            "completed_units": len(checks),
        }
    ]
    failed = [row for row in checks if not row["passed"]]
    if failed:
        artifact = blocked_artifact(
            checks, source_hashes=hashes, duration_s=time.monotonic() - started
        )
    else:
        selection = select_games(context["roster"])
        schedule = build_schedule(selection)
        registry_games = {
            str(row.get("game")): row
            for row in context["registry"].get("games", [])
            if isinstance(row, Mapping)
        }
        historical = [
            {
                "game": game,
                "historical_only": True,
                "registered": game in registry_games,
                "levels_reproduced": registry_games.get(game, {}).get("levels_reproduced"),
            }
            for game in selection
        ]
        raw = root / RAW
        raw.mkdir(parents=True, exist_ok=True)
        atomic_json(
            raw / "schedule.json",
            {
                "selection_basis": "salted_sdk_game_id_hash",
                "salt": SALT,
                "schedule": schedule,
                "registry_precheck": historical,
            },
        )
        progress(started, "live", "before", games=selection, episodes=6)
        live_start = time.monotonic() - started
        try:
            rows, runtime, model_hashes = run_live(root, context, schedule, started)
        except Exception as exc:
            checks.append(
                gate_check(
                    "exclusive_cuda_lease",
                    upstream="gpu_lease_phase_journal",
                    path=str(Path("/tmp/carnot-gpu-leases")),
                    field="current_lease_acquired",
                    operator="eq",
                    expected=True,
                    observed=False,
                )
            )
            artifact = blocked_artifact(
                checks, source_hashes=hashes, duration_s=time.monotonic() - started
            )
            artifact["gpu_lease_error"] = f"{type(exc).__name__}: {exc}"[:500]
        else:
            hashes["producer_files"]["model_weights"] = {
                "path": str(context["model_path"]),
                "sha256": model_hashes["model_sha256"],
            }
            hashes["producer_files"]["native_runtime"] = {
                "path": str(context["server"]),
                "sha256": model_hashes["runtime_sha256"],
            }
            artifact = build_live_artifact(
                checks, hashes, schedule, rows, runtime, time.monotonic() - started
            )
            artifact["registry_precheck"] = historical
        spans.append(
            {
                "phase": "live",
                "start_s": live_start,
                "end_s": time.monotonic() - started,
                "completed_units": len(artifact["rows"]),
            }
        )
        progress(
            started,
            "live",
            "after",
            verdict=artifact["honest_verdict"],
            completed_units=len(artifact["rows"]),
        )
    artifact["affected_validation_manifest"] = {
        "requirement": "REQ-REPORT-7653",
        "tests": [TEST.as_posix()],
        "changed_modules": [MODULE.as_posix()],
        "static_paths": [WRAPPER.as_posix()],
    }
    progress(started, "validation", "before")
    validation_start = time.monotonic() - started
    commands = validation_commands(root, private)
    receipts = validation.run_commands(
        root, commands, log_dir=private / "validation_logs", heartbeat_s=45
    )
    artifact["validation_receipts"] = receipts
    valid = all(row["passed"] for row in receipts)
    artifact["acceptance_gate_results"]["readiness"] = _gate(
        valid,
        "Every scoped and repository Python check must pass.",
        required_exits=[row["exit_code"] for row in receipts],
    )
    if not valid and artifact["verdict_class"] != "blocked":
        artifact["honest_verdict"] = "complete_disqualified_required_validation"
        artifact["verdict_class"] = "disqualified"
    spans.append(
        {
            "phase": "validation",
            "start_s": validation_start,
            "end_s": time.monotonic() - started,
            "completed_units": len(receipts),
        }
    )
    artifact["phase_spans"] = spans
    artifact["duration_s"] = time.monotonic() - started
    artifact["field_principles"] = field_principles(artifact)
    candidate = private / "terminal_candidate.json"
    atomic_json(candidate, artifact)
    progress(started, "terminal_readers", "before")
    first = validation.run_commands(
        root, terminal_commands(root, candidate), log_dir=private / "terminal_first", heartbeat_s=45
    )
    artifact["validation_receipts"].extend(first)
    artifact["flagged_adversarial"] = any(
        row["name"] == "adversarial_verify" and not row["passed"] for row in first
    )
    if any(not row["passed"] for row in first) and artifact["verdict_class"] != "blocked":
        artifact["honest_verdict"] = "complete_disqualified_terminal_reader"
        artifact["verdict_class"] = "disqualified"
    artifact["duration_s"] = time.monotonic() - started
    artifact["field_principles"] = field_principles(artifact)
    atomic_json(candidate, artifact)
    exact = validation.run_commands(
        root, terminal_commands(root, candidate), log_dir=private / "terminal_exact", heartbeat_s=45
    )
    exact_path = root / RAW / "terminal_reader_receipts.json"
    exact_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(exact_path, {"candidate_sha256": sha256_file(candidate), "receipts": exact})
    if any(not row["passed"] for row in exact):
        progress(started, "terminal_readers", "failed", receipts=str(exact_path))
        return 2
    progress(started, "terminal_readers", "after", receipts=str(exact_path))
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, artifact)
    progress(started, "publication", "after", output=output, sha256=sha256_file(output))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Provide the declared entrypoint and two cold reader modes."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260925")
    parser.add_argument("--output", type=Path, default=RESULT)
    parser.add_argument("--cold-reduce", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    args = parser.parse_args(argv)
    if args.cold_reduce or args.independent_reduce:
        return _reader_main(args.cold_reduce or args.independent_reduce)
    root = ROOT.resolve()
    return run_experiment(root, args.date, (root / args.output).resolve())

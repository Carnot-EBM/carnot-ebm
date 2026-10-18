"""Select one novel ARC target and report literal live-probe custody.

REQ-REPORT-7681. The installed catalogue is checked before a model is loaded.
If every public game is already cleared, the run terminates with an exact
external-input block. No previously reached level becomes a new target.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import json
import os
from pathlib import Path
import socket
import time
from typing import Any

import yaml

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
SALT = "v669-live-probe-20260926"
SEED = 7_681_669
RESULT = Path("results/experiment_7681_v669_arc_live_probes.json")
RAW = Path("results/raw/experiment_7681_v669_arc_live_probes")
INPUTS = (
    "AGENTS.md",
    "CODEX.md",
    "CLAUDE.md",
    "research-program.md",
    "ops/exclusion_manifest.yaml",
    "ops/e2e-test-plan.md",
    "ops/arc_solve_registry.yaml",
    "scripts/experiment_template.py",
    "python/carnot/reporting/current_work_receipt.py",
    "python/carnot/reporting/experiment_7303_validation_scope.py",
    "openspec/capabilities/research-reporting/spec.md",
    "openspec/capabilities/arc-agi/spec.md",
    "python/carnot/agentic/arc_competition_agent.py",
    "python/carnot/agentic/arc_goal_confirmation.py",
    "python/carnot/agentic/arc_executable_world_model.py",
    "python/carnot/agentic/arc_probe_protocol.py",
    "python/carnot/experiment_7630_v666_cuda_ownership.py",
    "python/carnot/inference/sota_models.py",
    "python/carnot/experiment_7681_v669_arc_live_probes.py",
    "scripts/experiments/experiment_7681_v669_arc_live_probes.py",
    "tests/python/test_experiment_7681_v669_arc_live_probes.py",
)


def progress(started: float, phase: str, event: str, **detail: Any) -> None:
    """Flush a phase boundary and its elapsed time immediately."""
    print(
        f"[exp7681] {phase} {event} elapsed_s={time.monotonic() - started:.3f} {detail}", flush=True
    )


def block_check(
    check: str, upstream: str, path: str, field: str, expected: Any, observed: Any
) -> Json:
    """Retain every operand needed to explain an unavailable external input."""
    return {
        "check": check,
        "upstream": upstream,
        "path": path,
        "field": field,
        "operator": "==",
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
    }


def select_target(
    roster: Iterable[str], registry: Iterable[Mapping[str, Any]], *, salt: str = SALT
) -> Json:
    """Choose an uncleared, adapter-free SDK game by a frozen stable hash."""
    known = {str(row.get("game")): row for row in registry}
    eligible: list[Json] = []
    excluded_reproduced: dict[str, int] = {}
    excluded_adapter: list[str] = []
    excluded_cleared: list[str] = []
    for game in sorted(set(roster)):
        row = known.get(game)
        if row is None:
            continue
        levels = int(row.get("levels_reproduced") or 0)
        excluded_reproduced[game] = levels
        if row.get("adapter") or row.get("requires_adapter"):
            excluded_adapter.append(game)
            continue
        if row.get("full_game_clear") is True:
            excluded_cleared.append(game)
            continue
        eligible.append({"game": game, "levels_reproduced": levels, "target_level": levels + 1})
    if not eligible:
        raise ValueError("no_eligible_game")
    selected = min(eligible, key=lambda item: (canonical_hash([salt, item["game"]]), item["game"]))
    return {
        **selected,
        "selection_rule": "min_sha256_of_salt_and_game_id",
        "selection_salt": salt,
        "excluded_reproduced_levels": excluded_reproduced,
        "excluded_adapter_games": excluded_adapter,
        "excluded_full_clear_games": excluded_cleared,
        "eligible_game_count": len(eligible),
    }


def reduce_episode(
    events: Sequence[Mapping[str, Any]], *, baseline_level: int, registry_level: int
) -> Json:
    """Reduce only factual decisions paired with subsequent SDK observations."""
    pending: int | None = None
    actions = 0
    admitted = 0
    peak = baseline_level
    for row in events:
        index = int(row["action_index"])
        if row["event"] == "decision":
            if pending is not None:
                raise ValueError("unpaired_decision")
            pending = index
            admitted += int(bool(row.get("admitted")))
        elif row["event"] == "observation":
            if pending != index:
                raise ValueError("observation_without_decision")
            peak = max(peak, int(row["level"]))
            actions += 1
            pending = None
    if pending is not None:
        raise ValueError("unpaired_decision")
    return {
        "actions": actions,
        "probes_admitted": admitted,
        "sdk_confirmed_level_ups": max(0, peak - baseline_level),
        "credited_solve": False,  # requires independent reproduction and novelty receipt
        "registry_novel_level_observed": peak > registry_level,
    }


def collect_preconditions(root: Path, started: float) -> tuple[list[Json], Json, Json]:
    """Authenticate named bytes and resource availability before SDK play."""
    from carnot.agentic.arc_solver_kit import offline_arcade
    from carnot.experiment_7630_v666_cuda_ownership import (
        ProcessRegistry,
        _current_inventory,
        select_owned_capacity,
    )
    from carnot.inference.sota_models import cached_current_model

    checks: list[Json] = []
    hashes: Json = {"producers": {}, "pre_gate_receipts": {}, "missing_evidence": []}
    for label in INPUTS:
        path = root / label
        exists = path.is_file() and path.stat().st_size > 0
        checks.append(
            block_check("input_exists", label, str(path), "readable_nonempty", True, exists)
        )
        if exists:
            hashes["producers"][label] = sha256_file(path)
        else:
            hashes["missing_evidence"].append(label)
    predecessor = root / "results/experiment_7667_v668_arc_live_goal_observation.json"
    if predecessor.is_file():
        hashes["pre_gate_receipts"][str(predecessor.relative_to(root))] = sha256_file(predecessor)
    else:
        hashes["missing_evidence"].append(str(predecessor.relative_to(root)))

    registry_path = root / "ops/arc_solve_registry.yaml"
    registry = yaml.safe_load(registry_path.read_text()) if registry_path.is_file() else {}
    games = registry.get("games", []) if isinstance(registry, dict) else []
    progress(started, "sdk_catalogue", "before")
    try:
        catalogue = offline_arcade().available_environments
        roster = sorted({str(item.game_id).split("-")[0] for item in catalogue})
        sdk_error = None
    except Exception as exc:
        roster = []
        sdk_error = f"{type(exc).__name__}: {exc}"[:300]
    progress(started, "sdk_catalogue", "after", games=len(roster), error=sdk_error)
    checks.append(
        block_check(
            "sdk_catalogue",
            "arc_agi_offline_sdk",
            str(root / "environment_files"),
            "catalogue_available",
            True,
            bool(roster),
        )
    )
    eligible = [
        item
        for item in roster
        if any(
            row.get("game") == item
            and row.get("full_game_clear") is not True
            and not row.get("adapter")
            and not row.get("requires_adapter")
            for row in games
        )
    ]
    checks.append(
        block_check(
            "eligible_novel_target",
            "arc_solve_registry",
            str(registry_path),
            "uncleared_adapter_free_sdk_games_at_least_one",
            True,
            bool(eligible),
        )
    )
    model = cached_current_model()
    model_path = Path(str(model["model_path"])).resolve() if model else None
    model_present = bool(model_path and model_path.is_file() and model_path.stat().st_size > 0)
    checks.append(
        block_check(
            "mandated_model",
            "cached_current_model",
            str(model_path),
            "cached_gguf_present",
            True,
            model_present,
        )
    )
    model_hash = sha256_file(model_path) if model_present and model_path else None
    server = (Path.home() / ".cache/llama.cpp-master/build/bin/llama-server").resolve()
    server_present = server.is_file() and os.access(server, os.X_OK)
    checks.append(
        block_check(
            "llama_server",
            "local_cuda_runtime",
            str(server),
            "executable_present",
            True,
            server_present,
        )
    )
    owner = ProcessRegistry.current()
    owner.task_id = "experiment_7681_v669_arc_live_probes"
    inventory = _current_inventory()
    selected, process_rows = select_owned_capacity(inventory, owner)
    checks.append(
        block_check(
            "owned_cuda_capacity",
            "nvidia_smi",
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
            "registry_games": games,
            "sdk_roster": roster,
            "sdk_error": sdk_error,
            "eligible_game_ids": eligible,
            "model_path": str(model_path) if model_path else None,
            "model_sha256": model_hash,
            "server_path": str(server),
            "gpu_uuid": selected.get("uuid") if selected else None,
            "gpu_process_rows": process_rows,
            "gpu_inventory_count": len(inventory),
        },
    )


def _gate(passed: bool, principle: str, **operands: Any) -> Json:
    """Keep measured operands beside each gate decision."""
    return {"passed": passed, "principle": principle, "measured_operands": operands}


def blocked_artifact(
    checks: Sequence[Mapping[str, Any]],
    hashes: Json,
    context: Json,
    *,
    run_date: str,
    started: float,
    phase_spans: Sequence[Json],
    validation_receipts: Sequence[Json] = (),
) -> Json:
    """Emit a complete external block without pretending to run an episode."""
    failed = [dict(row) for row in checks if not row["passed"]]
    reason = failed[0]["check"] if failed else "unknown_input"
    elapsed = time.monotonic() - started
    registry = context["registry_games"]
    roster = context["sdk_roster"]
    excluded = {
        str(row["game"]): int(row.get("levels_reproduced") or 0)
        for row in registry
        if str(row.get("game")) in roster
    }
    result: Json = {
        "schema": "carnot.exp7681.v669.arc_live_probes.v1",
        "experiment_id": 7681,
        "milestone": "2026.09.669",
        "run_date": run_date,
        "status": "complete",
        "honest_verdict": f"complete_blocked_{reason}",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "gate_check_summary": {"failed_checks": failed},
        "preconditions_checked": list(checks),
        "MODEL_SPECS": [],
        "model_specs": [{"no_model_invoked": True}],
        "planned_MODEL_SPECS": [MODEL_ID],
        "model_invoked": False,
        "invocation_counts": {
            "loads_attempted": 0,
            "loads_completed": 0,
            "forwards": 0,
            "generations_attempted": 0,
            "generations_completed": 0,
            "generations_cancelled": 0,
            "tokens": 0,
        },
        "inference_substrate": "host_preflight_no_model_call",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "model_full_generation",
        "execution_venue": "host",
        "execution_venue_details": {
            "host": socket.gethostname(),
            "owned_pid": os.getpid(),
            "gpu_uuid": context["gpu_uuid"],
        },
        "duration_s": elapsed,
        "phase_spans": list(phase_spans),
        "random_seed": {"selection_salt": SALT, "episode_seed": SEED},
        "source_artifact_hashes": hashes,
        "rows": [],
        "per_game_results": [],
        "sample_size_budget": {
            "intended_independent_groups": 1,
            "observed_independent_groups": 0,
            "eligible_independent_groups": 0,
            "excluded_independent_groups": 1,
            "censored_independent_groups": 0,
            "prior_exposure": "public_registry_games_all_full_clear",
            "effective_blocks": 0,
            "limits": {"actions": 256, "engine_calls": 20000, "hypotheses": 16, "episode_s": 3000},
        },
        "registry_precheck": {
            "sdk_game_ids": roster,
            "excluded_reproduced_levels": excluded,
            "excluded_full_clear_games": sorted(
                str(row["game"])
                for row in registry
                if str(row.get("game")) in roster and row.get("full_game_clear") is True
            ),
            "eligible_game_ids": context["eligible_game_ids"],
            "selected_target": None,
            "sdk_outcome_custody": context["sdk_error"] or "catalogue_only",
        },
        "current_model_receipts": {
            "planned_identity": MODEL_ID,
            "cached_gguf_path": context["model_path"],
            "gguf_sha256": context["model_sha256"],
            "owned_gpu_uuid_available": context["gpu_uuid"],
            "load_attempted": False,
            "generation_tokens": 0,
            "episode_budget_s": 3000,
        },
        "probe_opportunities": {
            "eligible": 0,
            "proposed": 0,
            "admitted": 0,
            "executed": 0,
            "informative": 0,
            "rejected": 0,
            "accepted_engines": 0,
        },
        "arc_live_measurement_complete_score": 0,
        "solve_provenance": "no_live_attempt",
        "verifier_is_oracle": False,
        "validation_receipts": list(validation_receipts),
        "production_defaults_changed": False,
        "counterfactual_solve_rate_claim": False,
        "registry_credit_granted": False,
    }
    gates = {
        "validity": _gate(
            False,
            "A novel eligible SDK target is required.",
            eligible_game_count=len(context["eligible_game_ids"]),
            failed_checks=failed,
        ),
        "readiness": _gate(
            False, "Blocked evidence cannot open a readiness gate.", live_episodes=0
        ),
        "coverage": _gate(False, "One factual episode is required.", observed_groups=0),
        "freshness": _gate(False, "No current generation occurred.", generations=0),
        "probability": _gate(
            False, "One public episode cannot estimate solve probability.", independent_games=0
        ),
        "decision_utility": _gate(
            False, "Unexecuted shadows cannot prove utility.", factual_actions=0
        ),
        "retention": _gate(False, "No later episode tested retained learning.", followups=0),
        "efficiency": _gate(False, "No action or compute outcome was measured.", actions=0),
    }
    result["acceptance_gate_results"] = gates
    result["field_principles"] = {
        field: "Literal current-run custody and raw operands govern this field." for field in result
    }
    result["reproducibility_checksum"] = canonical_hash(
        {
            "sources": hashes,
            "model_hash": context["model_sha256"],
            "roster": roster,
            "excluded": excluded,
            "seed": SEED,
            "reducer": hashes["producers"].get(
                "python/carnot/experiment_7681_v669_arc_live_probes.py"
            ),
        }
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Run bounded preflight and publish literal custody for this SDK install."""
    started = time.monotonic()
    progress(started, "startup", "begin", root=str(ROOT.resolve()))
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=RESULT)
    parser.add_argument("--cold-reduce", type=Path)
    args = parser.parse_args(argv)
    if args.cold_reduce is not None:
        candidate = json.loads(args.cold_reduce.read_text())
        if candidate.get("rows"):
            row = candidate["rows"][0]
            events_path = ROOT / row["event_log"]
            events = [json.loads(line) for line in events_path.read_text().splitlines()]
            reduced = reduce_episode(
                events,
                baseline_level=int(row["start_level"]),
                registry_level=int(row["registry_level"]),
            )
            if reduced != candidate.get("independent_reduction"):
                raise ValueError("cold_reduction_mismatch")
            print(json.dumps(reduced, sort_keys=True), flush=True)
        else:
            if candidate.get("sample_size_budget", {}).get("observed_independent_groups") != 0:
                raise ValueError("blocked_group_count_mismatch")
            if candidate.get("registry_precheck", {}).get("eligible_game_ids"):
                raise ValueError("blocked_eligible_game_mismatch")
            print("cold_reduction=blocked_zero_groups", flush=True)
        return 0
    root = ROOT.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    raw = root / RAW
    raw.mkdir(parents=True, exist_ok=True)
    progress(started, "preconditions", "before")
    pre_start = time.monotonic()
    checks, hashes, context = collect_preconditions(root, started)
    phase_spans = [
        {
            "phase": "preconditions",
            "start_s": pre_start - started,
            "end_s": time.monotonic() - started,
            "completed_units": len(checks),
            "checkpoint": str(raw / "preconditions.json"),
        }
    ]
    atomic_json(
        raw / "preconditions.json", {"checks": checks, "hashes": hashes, "context": context}
    )
    progress(
        started,
        "preconditions",
        "after",
        checks=len(checks),
        failed=sum(not row["passed"] for row in checks),
    )
    if not context["eligible_game_ids"]:
        artifact = blocked_artifact(
            checks, hashes, context, run_date=args.date, started=started, phase_spans=phase_spans
        )
        atomic_json(output, artifact)
        progress(
            started,
            "publication",
            "complete_blocked",
            path=str(output),
            reason=artifact["honest_verdict"],
        )
        return 0
    # A future SDK catalogue with an uncleared game must use an owned live
    # episode runner. Never silently reclassify an eligible game as absent.
    raise RuntimeError("eligible_live_episode_requires_owned_runner")

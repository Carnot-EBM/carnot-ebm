"""Rerun the frozen object-table A/B under exclusive CUDA ownership.

The module reuses Exp6753's prompt, production route, row pairing, and quality
definitions. It adds the Exp6764 admission receipt and a fresh leased process
for every row. This keeps model and teardown evidence attributable to one arm.
It never reads game source, runs offline BFS, or claims a solve.

Spec refs: REQ-ARC-WMTE-6765 and SCENARIO-ARC-WMTE-6765-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import date as calendar_date
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np

from carnot import experiment_6753_object_table_fetch_on_demand_ab as exp6753
from carnot import experiment_6764_arc_exclusive_load_preflight as exp6764
from carnot import gpu_lease_phase_journal as lease_api


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = REPO_ROOT / "results/experiment_6765_object_table_fetch_ab_v2.json"
EXP6753_PATH = REPO_ROOT / "results/experiment_6753_object_table_fetch_on_demand_ab.json"
EXP6764_PATH = REPO_ROOT / "results/experiment_6764_arc_exclusive_load_preflight.json"
REGISTRY_PATH = REPO_ROOT / "ops/arc_solve_registry.yaml"
ARCHITECTURE_PATH = REPO_ROOT / "_bmad/architecture.md"
LEASE_RUNTIME_DIR = Path(os.environ.get("CARNOT_GPU_LEASE_RUNTIME_DIR", "/tmp/carnot-gpu-leases"))
SCHEMA = "carnot.experiment_6765.object_table_fetch_ab_v2.v1"
RUN_DATE = "20260830"
CONTEXT_REQUESTED = exp6753.CONTEXT_REQUESTED
NONINFERIORITY_MARGIN = exp6753.NONINFERIORITY_MARGIN
BOOTSTRAP_SEED = exp6753.BOOTSTRAP_SEED
BOOTSTRAP_RESAMPLES = exp6753.BOOTSTRAP_RESAMPLES
BASELINE_ARM = exp6753.BASELINE_ARM
TREATMENT_ARM = exp6753.TREATMENT_ARM
ARMS = exp6753.ARMS
PRODUCTION_ROUTE = exp6753.PRODUCTION_ROUTE
INFERENCE_SUBSTRATE = "production E3AgentPolicy on task-owned llama.cpp CUDA Qwen3.8-27B-GGUF"
COMPLETE_PHASE_SEQUENCE = lease_api.COMPLETE_PHASE_SEQUENCE
VERDICT_CLASSES = exp6753.VERDICT_CLASSES

acquire_selected_lease = exp6764.acquire_selected_lease
terminate_owned_process = exp6764.terminate_owned_process
_gpu_snapshot = exp6764._gpu_snapshot
_process_identity = exp6764._process_identity
_wait_for_vram_recovery = exp6764._wait_for_vram_recovery
proc_start_ticks = lease_api.proc_start_ticks
read_journal = lease_api.read_journal
sha256_text = exp6753.sha256_text

FIELD_PRINCIPLES: JsonDict = {
    "schema": "A versioned shape lets readers reject incompatible evidence.",
    "experiment": "The experiment number binds this result to REQ-ARC-WMTE-6765.",
    "title": "The title states the frozen paired question and its prerequisite.",
    "run_date": "The planning date fixes the requested evidence period.",
    "status": "The status separates complete, blocked, and partial evidence.",
    "field_principles": "Each top-level field states why it exists.",
    "inference_substrate": "The substrate excludes CPU, remote, and helper-only substitutes.",
    "duration_s": "Monotonic wall time makes live work visible.",
    "random_seed": "Frozen science and bootstrap seeds make the comparison repeatable.",
    "reproducibility_checksum": "One digest detects any design, row, or receipt drift.",
    "models_used": "Exact model paths and hashes prevent substitution.",
    "model_specs": "This alias lets the artifact verifier inspect exact model evidence.",
    "live_model_invoked": "The flag distinguishes live evidence from a blocked denominator.",
    "frozen_manifest": "The manifest proves the Exp6753 design stayed fixed.",
    "rows": "One row per game, seed, arm, and canary keeps failures visible.",
    "gpu_receipts": "Per-row device and VRAM facts prove CUDA residency.",
    "lease_receipts": "Per-row ownership, release, and recovery prove exclusion.",
    "prompt_tokens_by_arm": "Row-derived token totals show realized prompt cost.",
    "tool_calls_by_arm": "Row-derived calls show how each prompt used production tools.",
    "useful_fetch_rate": "A fetch counts only when later successful code used its response.",
    "transition_utility_delta": "Net changed-cell utility is the second quality axis.",
    "mean_prompt_token_savings": "Baseline minus treatment tokens measures the intended saving.",
    "change_fidelity_by_arm": "Arm means expose absolute quality before a paired delta.",
    "change_fidelity_delta": "Treatment minus baseline change fidelity is the main effect.",
    "change_fidelity_interval": "The frozen game-clustered interval supports non-inferiority.",
    "within_arm_variance": "Row-derived variance exposes unstable arms.",
    "noninferiority_margin": "The frozen Exp6753 noise floor limits an allowed quality loss.",
    "harmful_regressions": "Games below the frozen margin remain visible.",
    "paired_analysis": "Per-seed and per-game rows make every reducer reproducible.",
    "adoption_gate_conditions": "Named booleans separate adoption from completion.",
    "adoption_gate_passed": "Adoption needs savings and non-inferior change fidelity.",
    "object_table_ab_completed": "Completion needs all attributable science and canary rows.",
    "solve_claim": "False prevents induction evidence from becoming a solve claim.",
    "source_receipts": "Input hashes and source-boundary facts prove the allowed evidence path.",
    "preconditions_checked": "The full gate record explains admission or a block.",
    "gate_check_summary": "Named observed values make blocked and terminal decisions auditable.",
    "verifier_is_oracle": "False states that this quality measurement is not an oracle verifier.",
    "verdict_class": "A closed class makes the scientific result machine-readable.",
    "honest_verdict": "The terminal prefix states the owned outcome without a solve claim.",
}

REQUIRED_FIELDS = tuple(FIELD_PRINCIPLES)


def row_checksum(row: Mapping[str, Any]) -> str:
    """Hash one row without its self-referential checksum field."""

    return exp6753.sha256_json({key: value for key, value in row.items() if key != "row_sha256"})


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash one artifact without its self-referential checksum field."""

    return exp6753.sha256_json(
        {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    )


def frozen_manifest() -> JsonDict:
    """Return the Exp6753 design with its source identity attached."""

    design = deepcopy(exp6753.frozen_design())
    design["source_experiment"] = 6753
    design["source_design_sha256"] = exp6753.sha256_json(exp6753.frozen_design())
    design["rerun_requirement"] = "REQ-ARC-WMTE-6765"
    return design


def row_plan() -> list[JsonDict]:
    """Return 120 science rows and two separately typed canary rows."""

    science = [
        {**deepcopy(row), "quality_pool": "qwen3.8_science"} for row in exp6753.science_plan()
    ]
    canaries = [
        {**deepcopy(row), "quality_pool": "excluded_canary", "canary_unit": True}
        for row in exp6753.sidecar_plan()
    ]
    return science + canaries


def worker_environment(
    base: Mapping[str, str],
    model: Mapping[str, Any],
    selected_device: Mapping[str, Any],
    planned: Mapping[str, Any],
    *,
    port: int,
) -> dict[str, str]:
    """Build the frozen row environment before proposer construction."""

    bound_model = {**dict(model), "device_index": int(selected_device["index"])}
    env = exp6753.worker_environment(base, bound_model, planned)
    env["CARNOT_ARC_GENERATOR_CUDA_GPU"] = str(selected_device["index"])
    env["CARNOT_ARC_EXCLUSIVE_PORT"] = str(int(port))
    return env


def _load_json(path: Path) -> JsonDict:
    """Read a JSON object, or return an empty object for a missing or bad file."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _architecture_date(text: str) -> str | None:
    match = re.search(r"\*\*Last Reconciled:\*\*\s*(\d{4}-\d{2}-\d{2})", text)
    return match.group(1) if match else None


def _validate_exp6764_receipt(ready: Mapping[str, Any]) -> tuple[list[str], str | None]:
    """Validate the ready receipt while accepting its pre-alias schema exactly once."""

    errors = exp6764.validate_artifact(ready)
    if errors != ["missing_field:model_specs", "model_specs"]:
        return errors, None
    normalized = deepcopy(dict(ready))
    normalized["model_specs"] = deepcopy(normalized.get("models_used"))
    principles = deepcopy(dict(normalized.get("field_principles", {})))
    principles["model_specs"] = exp6764.FIELD_PRINCIPLES["model_specs"]
    normalized["field_principles"] = principles
    normalized["reproducibility_checksum"] = exp6764.artifact_checksum(normalized)
    return exp6764.validate_artifact(normalized), "models_used_to_model_specs"


def collect_preconditions(
    *,
    date: str = RUN_DATE,
    base_preflight_fn: Callable[[Path], JsonDict] = exp6764.collect_preconditions,
    root: Path = REPO_ROOT,
) -> JsonDict:
    """Extend Exp6764 host admission with the frozen science and route gates."""

    base = deepcopy(base_preflight_fn(root))
    ready = _load_json(EXP6764_PATH)
    prior = _load_json(EXP6753_PATH)
    try:
        registry_text = REGISTRY_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        registry_text = f"read_error:{type(exc).__name__}:{exc}"
    try:
        architecture_text = ARCHITECTURE_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        architecture_text = f"read_error:{type(exc).__name__}:{exc}"

    ready_errors, ready_normalization = (
        _validate_exp6764_receipt(ready) if ready else (["missing"], None)
    )
    checks: list[JsonDict] = list(base.get("checks", []))
    checks.append(
        {
            "check": "arc_exclusive_load_ready",
            "expected": True,
            "observed": ready.get("arc_exclusive_load_ready"),
            "validator_errors": ready_errors,
            "schema_normalization": ready_normalization,
            "passed": ready.get("arc_exclusive_load_ready") is True and not ready_errors,
        }
    )
    observed_design = prior.get("frozen_design")
    checks.append(
        {
            "check": "frozen_exp6753_manifest",
            "expected": exp6753.frozen_design(),
            "observed": observed_design,
            "passed": observed_design == exp6753.frozen_design(),
        }
    )
    route_files = {
        "agent": REPO_ROOT / "python/carnot/agentic/arc_competition_agent.py",
        "loop": REPO_ROOT / "python/carnot/agentic/arc_induction_tool_loop.py",
        "tools": REPO_ROOT / "python/carnot/agentic/arc_induction_tools.py",
    }
    try:
        route_source = {
            name: path.read_text(encoding="utf-8") for name, path in route_files.items()
        }
        route_ready = (
            "class E3AgentPolicy" in route_source["agent"]
            and "def make_carnot_agent" in route_source["agent"]
            and "induce_with_tool_loop(" in route_source["agent"]
            and "def induce_with_tool_loop" in route_source["loop"]
            and "def parse_xml_tool_calls" in route_source["tools"]
            and "def dispatch_tool" in route_source["tools"]
            and '"find_objects": session.find_objects' in route_source["tools"]
        )
    except OSError:
        route_source = {}
        route_ready = False
    checks.append(
        {
            "check": "production_e3_selfparse_route",
            "expected": PRODUCTION_ROUTE,
            "observed": PRODUCTION_ROUTE if route_ready else "unavailable",
            "source_sha256": {
                name: exp6753.sha256_text(source) for name, source in route_source.items()
            },
            "passed": route_ready,
        }
    )
    registry_ok = (
        not registry_text.startswith("read_error:")
        and "6765" not in registry_text
        and len(exp6753.GAME_IDS) == len(set(exp6753.GAME_IDS))
        and frozen_manifest().get("solve_target") is None
    )
    checks.append(
        {
            "check": "registry_no_new_or_duplicate_solve_target",
            "expected": {"experiment_target": None, "experiment_absent": True},
            "observed": {
                "experiment_target": frozen_manifest().get("solve_target"),
                "experiment_absent": "6765" not in registry_text,
                "registry_sha256": exp6753.sha256_text(registry_text),
            },
            "passed": registry_ok,
        }
    )
    reconciled = _architecture_date(architecture_text)
    try:
        planned = calendar_date.fromisoformat(f"{date[:4]}-{date[4:6]}-{date[6:]}")
        architecture_age = (
            (planned - calendar_date.fromisoformat(reconciled)).days if reconciled else None
        )
    except ValueError:
        architecture_age = None
    architecture_fresh = architecture_age is not None and 0 <= architecture_age <= 30
    checks.append(
        {
            "check": "architecture_map_fresh",
            "expected": f"last reconciled no more than 30 days before {date}",
            "observed": reconciled,
            "age_days": architecture_age,
            "passed": architecture_fresh,
        }
    )
    source_receipts = deepcopy(base.get("source_receipts", {}))
    source_receipts.update(
        {
            "exp6764": {"path": str(EXP6764_PATH), "sha256": exp6753.sha256_file(EXP6764_PATH)},
            "exp6753": {"path": str(EXP6753_PATH), "sha256": exp6753.sha256_file(EXP6753_PATH)},
            "solve_registry": {
                "path": str(REGISTRY_PATH),
                "sha256": exp6753.sha256_text(registry_text),
            },
            "architecture_map": {
                "path": str(ARCHITECTURE_PATH),
                "sha256": exp6753.sha256_text(architecture_text),
            },
            "source_access": {
                "game_source_read": False,
                "offline_bfs_used": False,
                "solve_trace_used": False,
                "per_game_query_injected": False,
            },
        }
    )
    return {
        **base,
        "all_passed": all(row.get("passed") is True for row in checks),
        "checks": checks,
        "source_receipts": source_receipts,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _serialize_transition(transition: Any) -> JsonDict:
    """Keep only public transition fields in a JSON-safe form."""

    return {
        "grid": _jsonable(getattr(transition, "grid", [])),
        "next_grid": _jsonable(getattr(transition, "next_grid", [])),
        "action": _jsonable(getattr(transition, "action", None)),
        "data": _jsonable(getattr(transition, "data", None)),
        "level_before": _jsonable(getattr(transition, "level_before", None)),
        "level_after": _jsonable(getattr(transition, "level_after", None)),
    }


def _build_window(game: str) -> JsonDict:
    return exp6753._build_windows((game,))[game]


def _object_table_for_window(window: Mapping[str, Any]) -> str:
    from carnot.agentic.arc_executable_world_model import objects_block

    return (
        "OBJECT STRUCTURE (same frames, connected-component view -- use object "
        "shape ids to track objects across the deltas above):\n"
        + objects_block(list(window["shown"]))
    )


def _read_gpu_layers(proposer: Any) -> JsonDict:
    log_path = getattr(proposer, "_stderr_log_path", None)
    try:
        text = Path(log_path).read_text(errors="replace") if log_path else ""
    except OSError:
        text = ""
    from carnot.experiment_6752_arc_code_carrying_tool_preflight import _gpu_layers_from_log

    return _gpu_layers_from_log(text, int(proposer.n_gpu_layers))


def _eligible_device_still_selected(selected_device: Mapping[str, Any]) -> bool:
    selection = exp6764.rank_eligible_devices(exp6764.nvidia_smi_inventory()["devices"])
    current = selection.get("selected_device")
    return isinstance(current, Mapping) and current.get("uuid") == selected_device.get("uuid")


def _blocked_row(planned: Mapping[str, Any], model: Mapping[str, Any], failure: str) -> JsonDict:
    row = exp6753._failed_row(planned, model, failure)
    row.update(
        {
            "prompt": "",
            "inline_object_table": "",
            "prompt_isolation_receipt": {},
            "public_observations": [],
            "tool_requests": [],
            "bounded_responses": [],
            "tool_loop_stats": {},
            "transition_receipt": {},
            "verifier_metrics": None,
            "engine_sha256": None,
            "session_receipt": {},
            "source_receipt": {
                "game_source_read": False,
                "offline_bfs_used": False,
                "per_game_query_injected": False,
            },
        }
    )
    row["row_sha256"] = row_checksum(row)
    return row


def _enrich_live_row(
    row: Mapping[str, Any],
    *,
    prompt: str,
    window: Mapping[str, Any],
    session_receipt: Mapping[str, Any],
) -> JsonDict:
    events = list(row.get("tool_events", []))
    value = deepcopy(dict(row))
    value.update(
        {
            "prompt": prompt,
            "raw_prompt_sha256": exp6753.sha256_text(prompt) if prompt else None,
            "inline_object_table": (
                _object_table_for_window(window)
                if row.get("arm") == BASELINE_ARM and window.get("shown")
                else ""
            ),
            "prompt_isolation_receipt": {},
            "public_observations": [_serialize_transition(item) for item in window["shown"]],
            "tool_requests": [
                {
                    "turn": event.get("turn"),
                    "tool": event.get("parsed_tool"),
                    "arguments": deepcopy(event.get("parsed_arguments")),
                    "raw_emission": event.get("raw_emission"),
                }
                for event in events
            ],
            "bounded_responses": [
                event.get("bounded_response") for event in events if event.get("bounded_response")
            ],
            "transition_receipt": {
                "shown_count": len(window["shown"]),
                "held_count": len(window["held"]),
                "held_public_observations": [
                    _serialize_transition(item) for item in window["held"]
                ],
            },
            "session_receipt": deepcopy(dict(session_receipt)),
            "source_receipt": {
                "game_source_read": False,
                "offline_bfs_used": False,
                "per_game_query_injected": False,
            },
        }
    )
    value["row_sha256"] = row_checksum(value)
    return value


def run_live_row_session(
    model: Mapping[str, Any],
    selected_device: Mapping[str, Any],
    planned: Mapping[str, Any],
    *,
    port: int,
    lease_runtime_dir: Path = LEASE_RUNTIME_DIR,
) -> JsonDict:
    """Run one production row inside one lease and return all owned resources."""

    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    env = worker_environment(os.environ, model, selected_device, planned, port=port)
    os.environ.update(env)
    started_ns = time.monotonic_ns()
    device_uuid = str(selected_device["uuid"])
    before = _gpu_snapshot(device_uuid)
    lease: Any = None
    proposer: Any = None
    row: JsonDict | None = None
    prompt = ""
    window: JsonDict | None = None
    owner: JsonDict = {}
    release: JsonDict = {}
    journal: JsonDict = {}
    process_receipt: JsonDict = {
        "pid": 0,
        "absent_after_exit": True,
        "unrelated_processes_signaled": [],
    }
    cleanup: JsonDict = {
        "pid": 0,
        "absent_after_exit": True,
        "unrelated_processes_signaled": [],
    }
    recovery: JsonDict = {
        "passed": False,
        "owned_pid_present": False,
        "before_used_mb": int(before.get("memory_used_mb", 0) or 0),
        "after_used_mb": int(before.get("memory_used_mb", 0) or 0),
    }
    context: int | None = None
    observed_model_path: str | None = None
    layers: JsonDict = {"requested": 999, "offloaded": 0, "total": None}
    peak_vram = 0
    errors: list[str] = []
    try:
        if not _eligible_device_still_selected(selected_device):
            raise RuntimeError("selected_device_no_longer_first_eligible")
        if not exp6764.port_is_free(port):
            raise RuntimeError("selected_port_no_longer_free")
        lease = acquire_selected_lease(
            runtime_dir=lease_runtime_dir,
            task_id=f"exp6765-{planned['row_id']}",
            selected_device=before,
            expected_model=str(model["model_path"]),
        )
        owner = lease.owner_receipt()
        lease.transition("admitted")
        lease.transition("loading")
        budgets = exp6753._budgets_for(planned)
        proposer = LocalGGUFProposer(
            repo_substr=str(model["repo_substr"]),
            model_path=str(model["model_path"]),
            n_ctx=CONTEXT_REQUESTED,
            max_tokens=int(budgets["completion_tokens_per_turn"]),
            timeout=int(budgets["timeout_s"]),
            port=int(port),
            mtp=False,
            n_gpu_layers=999,
            use_chat_template=True,
            extra_server_args=("-v",),
        )
        if not proposer._ensure_server():
            raise RuntimeError("llama_server_load_failed")
        if int(proposer.port) != int(port):
            raise RuntimeError("llama_server_changed_frozen_port")
        process = proposer._proc
        if process is None:
            raise RuntimeError("llama_server_process_missing")
        process_receipt = _process_identity(process)
        resident = _gpu_snapshot(device_uuid, int(process.pid))
        peak_vram = int(resident.get("owned_pid_vram_mb", 0) or 0)
        context = proposer.observed_n_ctx()
        observed_model_path = proposer.observed_model_path()
        layers = _read_gpu_layers(proposer)
        if (
            resident.get("owned_pid_present") is not True
            or peak_vram <= 0
            or int(layers.get("offloaded", 0) or 0) <= 0
        ):
            raise RuntimeError("owner_bound_cuda_residency_missing")
        lease.transition("resident", vram_mb=int(resident.get("memory_used_mb", 0) or 0))
        lease.transition("inferencing")
        window = _build_window(str(planned["game"]))
        bound_model = {**dict(model), "device_index": int(selected_device["index"])}
        output_root = Path(os.environ["CARNOT_ARC_E3_DIR"])
        row, prompt = exp6753._run_live_row(planned, bound_model, proposer, window, output_root)
        after_inference = _gpu_snapshot(device_uuid, int(process.pid))
        peak_vram = max(peak_vram, int(after_inference.get("owned_pid_vram_mb", 0) or 0))
    except Exception as exc:  # noqa: BLE001 - a live row keeps its owned terminal failure
        errors.append(f"{type(exc).__name__}: {exc}"[:500])
    finally:
        if lease is not None and lease.document.get("phase") in {"resident", "inferencing"}:
            try:
                lease.transition("unloading")
            except lease_api.LeaseError as exc:
                errors.append(f"{type(exc).__name__}: {exc}"[:500])
        if proposer is not None and proposer._proc is not None:
            process = proposer._proc
            cleanup = terminate_owned_process(process)
            proposer._proc = None
            process_receipt.update(
                {
                    "exit_code": cleanup.get("exit_code"),
                    "absent_after_exit": cleanup.get("absent_after_exit"),
                }
            )
        owned_pid = int(process_receipt.get("pid", 0) or 0)
        recovery, after = _wait_for_vram_recovery(
            device_uuid,
            owned_pid,
            int(before.get("memory_used_mb", 0) or 0),
        )
        if lease is not None:
            try:
                phase = lease.document.get("phase")
                if phase == "unloading":
                    lease.transition(
                        "validating",
                        vram_mb=int(after.get("memory_used_mb", 0) or 0),
                        exit_code=int(process_receipt.get("exit_code", 0) or 0),
                        unload_observed=process_receipt.get("absent_after_exit") is True,
                    )
                    complete = bool(
                        not errors
                        and row is not None
                        and row.get("failure_class") is None
                        and context == CONTEXT_REQUESTED
                        and observed_model_path == str(model["model_path"])
                        and int(layers.get("offloaded", 0) or 0) > 0
                        and cleanup.get("absent_after_exit") is True
                        and recovery.get("passed") is True
                    )
                    lease.transition("terminal_complete" if complete else "terminal_blocked")
                elif phase in {"preflight", "admitted", "loading"}:
                    lease.transition("terminal_blocked")
                if lease.document.get("phase") in lease_api.TERMINAL_PHASES:
                    release = lease.release()
                else:
                    lease.close()
            except lease_api.LeaseError as exc:
                errors.append(f"{type(exc).__name__}: {exc}"[:500])
                lease.close()
            try:
                journal = read_journal(lease.journal_path)
            except lease_api.LeaseError as exc:
                errors.append(f"{type(exc).__name__}: {exc}"[:500])

    session_receipt = {
        "session_id": str(planned["row_id"]),
        "device": deepcopy(dict(selected_device)),
        "lease_owner": owner,
        "phase_history": deepcopy(journal.get("phase_history", [])),
        "lease_release": release,
        "model_process": process_receipt,
        "owned_cleanup": cleanup,
        "vram_recovery": recovery,
        "peak_owned_vram_mb": int(peak_vram),
        "runtime_context": context,
        "observed_model_path": observed_model_path,
        "gpu_layers": layers,
        "duration_s": round((time.monotonic_ns() - started_ns) / 1_000_000_000, 6),
        "unrelated_processes_signaled": [],
        "errors": errors,
    }
    if row is None:
        failure = "session_lifecycle_failed:" + (errors[0] if errors else "no_row")
        row = _blocked_row(planned, model, failure)
    if window is None:
        window = {"shown": [], "held": [], "cell": 0}
    value = _enrich_live_row(row, prompt=prompt, window=window, session_receipt=session_receipt)
    if (
        session_receipt["lease_release"].get("released") is not True
        or session_receipt["owned_cleanup"].get("absent_after_exit") is not True
        or session_receipt["vram_recovery"].get("passed") is not True
        or errors
    ) and value.get("failure_class") is None:
        value["failure_class"] = "session_lifecycle_failed:" + (
            errors[0] if errors else "teardown_or_release"
        )
    value["gpu_receipt"] = {
        **deepcopy(value.get("gpu_receipt") or {}),
        "assigned_device": {
            "physical_index": selected_device.get("index"),
            "uuid": selected_device.get("uuid"),
            "name": selected_device.get("name"),
        },
        "gpu_layers": deepcopy(layers),
        "peak_vram_mb": int(peak_vram),
    }
    value["context_observed_by_model"] = context
    value["row_sha256"] = row_checksum(value)
    return value


def _session_slug(row_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", row_id).strip("-").lower()


def _load_completed_checkpoint(
    runtime_dir: Path,
    planned: Mapping[str, Any],
    model: Mapping[str, Any],
    selected_device: Mapping[str, Any],
) -> JsonDict | None:
    """Reuse a row only when its owned evidence still matches this frozen run.

    The worker writes the row atomically after teardown. The parent adds the
    pair-level prompt receipt later, so this check validates every other field.
    A partial or changed file returns ``None`` and the row runs again.
    """

    path = runtime_dir / _session_slug(str(planned["row_id"])) / "row.json"
    row = _load_json(path)
    if not row or row.get("row_sha256") != row_checksum(row):
        return None
    if any(row.get(key) != value for key, value in planned.items()):
        return None
    expected_model = {
        "model_role": model.get("role"),
        "model_path": model.get("model_path"),
        "model_sha256": model.get("model_sha256"),
    }
    if any(row.get(key) != value for key, value in expected_model.items()):
        return None
    gpu = row.get("gpu_receipt") if isinstance(row.get("gpu_receipt"), Mapping) else {}
    assigned = gpu.get("assigned_device") if isinstance(gpu.get("assigned_device"), Mapping) else {}
    session = row.get("session_receipt") if isinstance(row.get("session_receipt"), Mapping) else {}
    owner = session.get("lease_owner") if isinstance(session.get("lease_owner"), Mapping) else {}
    if (
        assigned.get("physical_index") != selected_device.get("index")
        or assigned.get("uuid") != selected_device.get("uuid")
        or owner.get("device_uuid") != selected_device.get("uuid")
        or owner.get("expected_model") not in (None, model.get("model_path"))
        or session.get("observed_model_path") not in (None, model.get("model_path"))
        or session.get("errors") not in (None, [])
    ):
        return None
    attributable = deepcopy(row)
    attributable["prompt_isolation_receipt"] = {"only_object_table_removed": True}
    attributable["row_sha256"] = row_checksum(attributable)
    if row_evidence_errors(attributable):
        return None
    return row


def run_row_worker_subprocess(
    model: Mapping[str, Any],
    selected_device: Mapping[str, Any],
    planned: Mapping[str, Any],
    runtime_dir: Path,
    *,
    port: int,
    lease_runtime_dir: Path = LEASE_RUNTIME_DIR,
) -> JsonDict:
    """Launch one fresh process for one planned row and retain its receipt."""

    session_dir = runtime_dir / _session_slug(str(planned["row_id"]))
    session_dir.mkdir(parents=True, exist_ok=True)
    job_path = session_dir / "job.json"
    output_path = session_dir / "row.json"
    exp6764.write_json_atomic(
        job_path,
        {
            "model": dict(model),
            "selected_device": dict(selected_device),
            "planned": dict(planned),
        },
    )
    command = [
        sys.executable,
        "-m",
        "carnot.experiment_6765_object_table_fetch_ab_v2",
        "--worker-job",
        str(job_path),
        "--worker-output",
        str(output_path),
        "--port",
        str(int(port)),
        "--lease-runtime-dir",
        str(lease_runtime_dir),
    ]
    env = worker_environment(os.environ, model, selected_device, planned, port=port)
    env["CARNOT_ARC_E3_DIR"] = str(session_dir / "e3")
    timeout_s = int(exp6753._budgets_for(planned)["timeout_s"]) + 900
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    start_ticks = proc_start_ticks(process.pid)
    timeout_cleanup: JsonDict = {}
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timeout_cleanup = exp6764._terminate_worker_group(process, start_ticks)
        stdout, stderr = process.communicate()
    row = _load_json(output_path)
    if not row:
        failure = (
            f"session_lifecycle_failed:worker_output_missing:exit={process.returncode}:"
            f"stderr_sha256={exp6753.sha256_text(stderr)}"
        )
        row = _blocked_row(planned, model, failure)
    session = deepcopy(row.get("session_receipt") or {})
    session["worker_process"] = {
        "pid": int(process.pid),
        "pid_start_ticks": start_ticks,
        "exit_code": process.returncode,
        "absent_after_exit": process.poll() is not None,
        "stdout_sha256": exp6753.sha256_text(stdout),
        "stderr_sha256": exp6753.sha256_text(stderr),
        "timeout_cleanup": timeout_cleanup,
        "unrelated_processes_signaled": [],
    }
    row["session_receipt"] = session
    row["row_sha256"] = row_checksum(row)
    return row


def attach_prompt_pair_receipts(rows: Sequence[JsonDict], *, tool_schemas: str) -> None:
    """Attach exact prompt-isolation evidence to every observed arm pair."""

    indexed = {
        (str(row.get("game")), int(row.get("seed", -1)), str(row.get("arm"))): row for row in rows
    }
    pair_keys = sorted({(key[0], key[1]) for key in indexed})
    for game, seed in pair_keys:
        baseline = indexed.get((game, seed, BASELINE_ARM))
        treatment = indexed.get((game, seed, TREATMENT_ARM))
        if (
            not baseline
            or not treatment
            or not baseline.get("prompt")
            or not treatment.get("prompt")
        ):
            continue
        try:
            receipt = exp6753.prompt_isolation_receipt(
                str(baseline["prompt"]),
                str(treatment["prompt"]),
                str(baseline.get("inline_object_table") or ""),
                tool_schemas,
            )
        except ValueError as exc:
            receipt = {"only_object_table_removed": False, "error": str(exc)}
        for row in (baseline, treatment):
            row["prompt_isolation_receipt"] = deepcopy(receipt)
            if receipt.get("only_object_table_removed") is not True:
                row["failure_class"] = "prompt_arm_isolation_failed"
            row["row_sha256"] = row_checksum(row)


def row_evidence_errors(row: Mapping[str, Any]) -> list[str]:
    """Return every defect that prevents a row from being attributable."""

    errors = list(exp6753.row_evidence_errors(row))
    prompt = str(row.get("prompt") or "")
    if not prompt or row.get("raw_prompt_sha256") != exp6753.sha256_text(prompt):
        errors.append("prompt")
    if not isinstance(row.get("public_observations"), list) or not row.get("public_observations"):
        errors.append("public_observations")
    if (row.get("prompt_isolation_receipt") or {}).get("only_object_table_removed") is not True:
        errors.append("prompt_isolation_receipt")
    session = row.get("session_receipt") if isinstance(row.get("session_receipt"), Mapping) else {}
    phases = [item.get("phase") for item in session.get("phase_history", [])]
    if phases != list(COMPLETE_PHASE_SEQUENCE):
        errors.append("lease_phase_history")
    if (session.get("lease_release") or {}).get("released") is not True:
        errors.append("lease_release")
    if (session.get("owned_cleanup") or {}).get("absent_after_exit") is not True:
        errors.append("owned_cleanup")
    if (session.get("vram_recovery") or {}).get("passed") is not True:
        errors.append("vram_recovery")
    if session.get("unrelated_processes_signaled") != []:
        errors.append("unrelated_processes_signaled")
    source = row.get("source_receipt") if isinstance(row.get("source_receipt"), Mapping) else {}
    if any(
        source.get(field) is not False
        for field in ("game_source_read", "offline_bfs_used", "per_game_query_injected")
    ):
        errors.append("source_boundary")
    if row.get("row_kind") == "transport_sidecar" and row.get("quality_pool") != "excluded_canary":
        errors.append("canary_quality_pool")
    if row.get("row_sha256") != row_checksum(row):
        errors.append("row_sha256")
    return list(dict.fromkeys(errors))


def _empty_reduction(receipt: Mapping[str, Any]) -> JsonDict:
    return {
        "prompt_tokens_by_arm": {arm: {"rows": 0, "total": None, "mean": None} for arm in ARMS},
        "tool_calls_by_arm": {arm: {"total": None, "by_name": {}} for arm in ARMS},
        "useful_fetch_rate": None,
        "transition_utility_delta": None,
        "mean_prompt_token_savings": None,
        "change_fidelity_by_arm": {arm: None for arm in ARMS},
        "change_fidelity_delta": None,
        "change_fidelity_interval": None,
        "within_arm_variance": {arm: None for arm in ARMS},
        "harmful_regressions": [],
        "paired_analysis": {},
        "adoption_gate_conditions": {
            "all_planned_rows_valid": False,
            "positive_prompt_token_savings": False,
            "change_fidelity_noninferior": False,
            "solve_claim_false": True,
        },
        "adoption_gate_passed": False,
        "object_table_ab_completed": False,
        "row_completion_receipt": deepcopy(dict(receipt)),
        "solve_claim": False,
    }


def reduce_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Derive completion, science metrics, and adoption from retained rows."""

    expected = row_plan()
    expected_ids = [str(row["row_id"]) for row in expected]
    by_id: dict[str, Mapping[str, Any]] = {}
    duplicates: list[str] = []
    for row in rows:
        row_id = str(row.get("row_id"))
        if row_id in by_id:
            duplicates.append(row_id)
        by_id[row_id] = row
    missing = [row_id for row_id in expected_ids if row_id not in by_id]
    unexpected = [row_id for row_id in by_id if row_id not in set(expected_ids)]
    invalid = {
        row_id: row_evidence_errors(by_id[row_id])
        for row_id in expected_ids
        if row_id in by_id and row_evidence_errors(by_id[row_id])
    }
    receipt = {
        "planned": len(expected_ids),
        "observed": len(rows),
        "missing_row_ids": missing,
        "unexpected_row_ids": unexpected,
        "duplicate_row_ids": duplicates,
        "invalid_rows": invalid,
    }
    complete = not missing and not unexpected and not duplicates and not invalid
    if not complete:
        return _empty_reduction(receipt)

    science = [row for row in rows if row.get("row_kind") == "science"]
    paired = exp6753.paired_statistics(science)
    token_by_arm: JsonDict = {}
    calls_by_arm: JsonDict = {}
    fidelity_by_arm: JsonDict = {}
    variance_by_arm: JsonDict = {}
    for arm in ARMS:
        arm_rows = [row for row in science if row.get("arm") == arm]
        token_values = [int(row["prompt_tokens"]) for row in arm_rows]
        names = Counter(
            str(event.get("parsed_tool"))
            for row in arm_rows
            for event in row.get("tool_events", [])
            if event.get("parsed_tool")
        )
        fidelity = [float(row["change_fidelity"]) for row in arm_rows]
        token_by_arm[arm] = {
            "rows": len(arm_rows),
            "total": sum(token_values),
            "mean": float(np.mean(token_values)),
        }
        calls_by_arm[arm] = {"total": sum(names.values()), "by_name": dict(names)}
        fidelity_by_arm[arm] = float(np.mean(fidelity))
        variance_by_arm[arm] = float(np.var(fidelity))
    conditions = {
        "all_planned_rows_valid": True,
        "positive_prompt_token_savings": paired["mean_prompt_token_savings"] > 0,
        "change_fidelity_noninferior": paired["noninferiority_passed"] is True,
        "solve_claim_false": all(row.get("solve_claim") is False for row in rows),
    }
    return {
        "prompt_tokens_by_arm": token_by_arm,
        "tool_calls_by_arm": calls_by_arm,
        "useful_fetch_rate": paired["useful_fetch_rate"],
        "transition_utility_delta": paired["transition_utility_delta"],
        "mean_prompt_token_savings": paired["mean_prompt_token_savings"],
        "change_fidelity_by_arm": fidelity_by_arm,
        "change_fidelity_delta": paired["change_fidelity_delta"],
        "change_fidelity_interval": paired["change_fidelity_ci95"],
        "within_arm_variance": variance_by_arm,
        "harmful_regressions": paired["harmful_regressions"],
        "paired_analysis": paired,
        "adoption_gate_conditions": conditions,
        "adoption_gate_passed": all(conditions.values()),
        "object_table_ab_completed": True,
        "row_completion_receipt": receipt,
        "solve_claim": False,
    }


def _first_failed_check(preflight: Mapping[str, Any]) -> str:
    failed = next(
        (row for row in preflight.get("checks", []) if row.get("passed") is not True),
        None,
    )
    return str(failed.get("check")) if isinstance(failed, Mapping) else "unknown_precondition"


def _gpu_receipts(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "row_id": row.get("row_id"),
            "model_id": row.get("model_id"),
            "gpu_receipt": deepcopy(row.get("gpu_receipt")),
            "peak_owned_vram_mb": (row.get("session_receipt") or {}).get("peak_owned_vram_mb"),
        }
        for row in rows
        if row.get("gpu_receipt") is not None
    ]


def _lease_receipts(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "row_id": row.get("row_id"),
            "lease_owner": deepcopy((row.get("session_receipt") or {}).get("lease_owner")),
            "phase_history": deepcopy((row.get("session_receipt") or {}).get("phase_history")),
            "lease_release": deepcopy((row.get("session_receipt") or {}).get("lease_release")),
            "owned_cleanup": deepcopy((row.get("session_receipt") or {}).get("owned_cleanup")),
            "vram_recovery": deepcopy((row.get("session_receipt") or {}).get("vram_recovery")),
        }
        for row in rows
        if row.get("session_receipt")
    ]


def build_artifact(
    *,
    date: str,
    rows: Sequence[Mapping[str, Any]],
    preflight: Mapping[str, Any],
    started_ns: int,
    finished_ns: int,
) -> JsonDict:
    """Build one terminal artifact from the full denominator and row reducers."""

    retained = [deepcopy(dict(row)) for row in rows]
    reduction = reduce_rows(retained)
    blocked = preflight.get("all_passed") is not True
    if blocked:
        verdict_class = "blocked"
        honest_verdict = "complete_blocked_object_table_ab_v2"
        status = "blocked"
        gate_summary = deepcopy(list(preflight.get("checks", [])))
    elif reduction["object_table_ab_completed"]:
        verdict_class = "positive" if reduction["adoption_gate_passed"] else "null"
        honest_verdict = (
            "complete_object_table_ab_v2_adopt_fetch_on_demand"
            if reduction["adoption_gate_passed"]
            else "complete_object_table_ab_v2_do_not_adopt"
        )
        status = "complete"
        gate_summary = [
            {"check": key, "expected": True, "observed": value, "passed": value is True}
            for key, value in reduction["adoption_gate_conditions"].items()
        ]
    else:
        verdict_class = "partial"
        honest_verdict = "complete_partial_object_table_ab_v2"
        status = "partial"
        gate_summary = [
            {
                "check": "row_completeness",
                "expected": len(row_plan()),
                "observed": reduction["row_completion_receipt"],
                "passed": False,
            }
        ]
    source_receipts = deepcopy(dict(preflight.get("source_receipts", {})))
    source_receipts.setdefault(
        "source_access",
        {
            "game_source_read": False,
            "offline_bfs_used": False,
            "solve_trace_used": False,
            "per_game_query_injected": False,
        },
    )
    models = [deepcopy(dict(row)) for row in preflight.get("models", [])]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": 6765,
        "title": "Exclusive-load inline object table versus production fetch on demand",
        "run_date": str(date),
        "status": status,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(max(0, finished_ns - started_ns) / 1_000_000_000, 6),
        "random_seed": {
            "science_game_arm_seeds": list(exp6753.SEEDS),
            "bootstrap": BOOTSTRAP_SEED,
        },
        "reproducibility_checksum": "",
        "models_used": models,
        "model_specs": deepcopy(models),
        "live_model_invoked": any(row.get("live_model_invoked") is True for row in retained),
        "frozen_manifest": frozen_manifest(),
        "rows": retained,
        "gpu_receipts": _gpu_receipts(retained),
        "lease_receipts": _lease_receipts(retained),
        "prompt_tokens_by_arm": reduction["prompt_tokens_by_arm"],
        "tool_calls_by_arm": reduction["tool_calls_by_arm"],
        "useful_fetch_rate": reduction["useful_fetch_rate"],
        "transition_utility_delta": reduction["transition_utility_delta"],
        "mean_prompt_token_savings": reduction["mean_prompt_token_savings"],
        "change_fidelity_by_arm": reduction["change_fidelity_by_arm"],
        "change_fidelity_delta": reduction["change_fidelity_delta"],
        "change_fidelity_interval": reduction["change_fidelity_interval"],
        "within_arm_variance": reduction["within_arm_variance"],
        "noninferiority_margin": NONINFERIORITY_MARGIN,
        "harmful_regressions": reduction["harmful_regressions"],
        "paired_analysis": reduction["paired_analysis"],
        "adoption_gate_conditions": reduction["adoption_gate_conditions"],
        "adoption_gate_passed": reduction["adoption_gate_passed"],
        "object_table_ab_completed": reduction["object_table_ab_completed"],
        "solve_claim": False,
        "source_receipts": source_receipts,
        "preconditions_checked": deepcopy(dict(preflight)),
        "gate_check_summary": gate_summary,
        "verifier_is_oracle": False,
        "verdict_class": verdict_class,
        "honest_verdict": honest_verdict,
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute the terminal schema and row-derived decisions."""

    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in artifact:
            errors.append(f"missing_field:{field}")
    principles = artifact.get("field_principles")
    principles = principles if isinstance(principles, Mapping) else {}
    if not set(artifact).issubset(principles):
        errors.append("field_principles")
    if artifact.get("schema") != SCHEMA:
        errors.append("schema")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact.get("frozen_manifest") != frozen_manifest():
        errors.append("frozen_manifest")
    if artifact.get("solve_claim") is not False:
        errors.append("solve_claim")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        errors.append("verdict_class")
    source = artifact.get("source_receipts", {}).get("source_access", {})
    if any(source.get(field) is not False for field in ("game_source_read", "offline_bfs_used")):
        errors.append("source_receipts")
    rows = artifact.get("rows") if isinstance(artifact.get("rows"), list) else []
    if [row.get("row_id") for row in rows] != [row["row_id"] for row in row_plan()]:
        errors.append("row_denominator_or_order")
    reduction = reduce_rows(rows)
    for field in (
        "prompt_tokens_by_arm",
        "tool_calls_by_arm",
        "useful_fetch_rate",
        "transition_utility_delta",
        "mean_prompt_token_savings",
        "change_fidelity_by_arm",
        "change_fidelity_delta",
        "change_fidelity_interval",
        "within_arm_variance",
        "harmful_regressions",
        "paired_analysis",
        "adoption_gate_conditions",
        "adoption_gate_passed",
        "object_table_ab_completed",
    ):
        if artifact.get(field) != reduction.get(field):
            errors.append(f"reduction:{field}")
    if artifact.get("gpu_receipts") != _gpu_receipts(rows):
        errors.append("gpu_receipts")
    if artifact.get("lease_receipts") != _lease_receipts(rows):
        errors.append("lease_receipts")
    if artifact.get("model_specs") != artifact.get("models_used"):
        errors.append("model_specs")
    preflight_passed = artifact.get("preconditions_checked", {}).get("all_passed") is True
    blocked = not preflight_passed
    expected_verdict = (
        "complete_blocked_object_table_ab_v2"
        if blocked
        else (
            "complete_object_table_ab_v2_adopt_fetch_on_demand"
            if reduction["object_table_ab_completed"] and reduction["adoption_gate_passed"]
            else (
                "complete_object_table_ab_v2_do_not_adopt"
                if reduction["object_table_ab_completed"]
                else "complete_partial_object_table_ab_v2"
            )
        )
    )
    if artifact.get("honest_verdict") != expected_verdict:
        errors.append("honest_verdict")
    if blocked:
        if artifact.get("verdict_class") != "blocked":
            errors.append("blocked_verdict_class")
        if any(
            not str(row.get("failure_class") or "").startswith("preflight_blocked:") for row in rows
        ):
            errors.append("blocked_rows")
    expected_live = any(row.get("live_model_invoked") is True for row in rows)
    if artifact.get("live_model_invoked") is not expected_live:
        errors.append("live_model_invoked")
    if artifact.get("noninferiority_margin") != NONINFERIORITY_MARGIN:
        errors.append("noninferiority_margin")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum")
    return list(dict.fromkeys(errors))


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    exp6764.write_json_atomic(path, value)


def run(
    *,
    result_path: Path = RESULT_PATH,
    date: str = RUN_DATE,
    preflight_fn: Callable[..., JsonDict] = collect_preconditions,
    worker_runner: Callable[..., JsonDict] = run_row_worker_subprocess,
    clock: Callable[[], int] = time.monotonic_ns,
) -> JsonDict:
    """Run admission, then one fresh leased worker for each frozen row."""

    started_ns = clock()
    preflight = preflight_fn(date=date, root=REPO_ROOT)
    planned_rows = row_plan()
    models = list(preflight.get("models", []))
    model_by_id = {row.get("model_id"): row for row in models}
    if preflight.get("all_passed") is not True:
        failure = f"preflight_blocked:{_first_failed_check(preflight)}"
        rows = [
            _blocked_row(planned, model_by_id.get(planned["model_id"], {}), failure)
            for planned in planned_rows
        ]
    else:
        selected = preflight.get("device_selection_receipt", {}).get("selected_device")
        if not isinstance(selected, Mapping):
            raise ValueError("passing preflight omitted selected_device")
        runtime_dir = result_path.parent / ".experiment_6765_object_table_fetch_ab_v2"
        rows: list[JsonDict] = []
        stopped = False
        failure = ""
        for planned in planned_rows:
            model = model_by_id.get(planned["model_id"])
            if stopped or not isinstance(model, Mapping):
                reason = failure or "not_run_after_session_failure:model_missing"
                rows.append(_blocked_row(planned, model or {}, reason))
                continue
            checkpoint = _load_completed_checkpoint(runtime_dir, planned, model, selected)
            if checkpoint is not None:
                rows.append(checkpoint)
                continue
            port = exp6764.choose_free_ports(1)[0]
            try:
                row = worker_runner(
                    model,
                    selected,
                    planned,
                    runtime_dir,
                    port=port,
                    lease_runtime_dir=LEASE_RUNTIME_DIR,
                )
            except Exception as exc:  # noqa: BLE001 - retain the row and stop the run
                row = _blocked_row(
                    planned,
                    model,
                    f"session_lifecycle_failed:{type(exc).__name__}:{exc}"[:500],
                )
            rows.append(row)
            if row.get("failure_class") is not None:
                stopped = True
                failure = f"not_run_after_session_failure:{row.get('row_id')}"
        from carnot.agentic.arc_induction_tools import render_tool_schemas_for_prompt

        attach_prompt_pair_receipts(rows, tool_schemas=render_tool_schemas_for_prompt())
    artifact = build_artifact(
        date=date,
        rows=rows,
        preflight=preflight,
        started_ns=started_ns,
        finished_ns=clock(),
    )
    validation_errors = validate_artifact(artifact)
    if validation_errors:
        raise ValueError("invalid Exp6765 artifact:" + ",".join(validation_errors))
    _atomic_write(result_path, artifact)
    return artifact


def _worker_entry(
    job_path: Path,
    output_path: Path,
    port: int,
    lease_runtime_dir: Path,
) -> int:
    job = _load_json(job_path)
    row = run_live_row_session(
        job["model"],
        job["selected_device"],
        job["planned"],
        port=port,
        lease_runtime_dir=lease_runtime_dir,
    )
    _atomic_write(output_path, row)
    return 0 if row.get("failure_class") is None else 2


def main(argv: Sequence[str] | None = None) -> int:
    """Run the parent experiment or one explicitly owned row worker."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--worker-job", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--port", type=int)
    parser.add_argument("--lease-runtime-dir", type=Path, default=LEASE_RUNTIME_DIR)
    args = parser.parse_args(argv)
    if args.worker_job is not None:
        if args.worker_output is None or args.port is None:
            parser.error("--worker-job requires --worker-output and --port")
        return _worker_entry(
            args.worker_job,
            args.worker_output,
            int(args.port),
            args.lease_runtime_dir,
        )
    artifact = run(date=args.date)
    print(
        json.dumps(
            {
                "artifact": str(RESULT_PATH),
                "completed": artifact["object_table_ab_completed"],
                "adopted": artifact["adoption_gate_passed"],
                "verdict": artifact["honest_verdict"],
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - the repository wrapper is the public entry point
    raise SystemExit(main())

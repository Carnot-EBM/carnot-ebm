"""Prove the production ARC selfparse route with a live code-carrying call.

The experiment uses a small synthetic transition. It measures transport and
CUDA admission only. It does not inspect a game, evaluate model quality, or
claim a level solve.

Spec refs: REQ-ARC-WMTE-6752 and SCENARIO-ARC-WMTE-6752-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

import numpy as np

from carnot.agentic.arc_induction_tools import MAX_FIND_OBJECT_RESPONSE_BYTES


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = REPO_ROOT / "results/experiment_6752_arc_code_carrying_tool_preflight.json"
REGISTRY_PATH = REPO_ROOT / "ops/arc_solve_registry.yaml"
SCHEMA = "carnot.experiment_6752.arc_code_carrying_tool_preflight.v1"
RUN_DATE = "20260829"
RANDOM_SEED = 6_752
CONTEXT_REQUESTED = 32_768
REQUESTED_MAX_OBJECTS = 8
WORKER_TIMEOUT_S = 1_200.0
VRAM_GUARD_MB = 1_500
INFERENCE_SUBSTRATE = "local llama.cpp CUDA GGUF production tool route"
PRODUCTION_ROUTE = "induce_with_tool_loop/selfparse/dispatch_tool"
CLAIM_BOUNDARY = (
    "This preflight proves 32K CUDA admission and code-carrying tool transport only. "
    "It measures no game quality and claims no level solve."
)

MODEL_SPECS: tuple[JsonDict, ...] = (
    {
        "model_id": "unsloth/Qwen3.8-27B-GGUF",
        "role": "immutable_scored_arc_generator",
        "repo_substr": "Qwen3.8-27B",
        "filename": "Qwen3.8-27B-Q4_K_M.gguf",
        "device_index": 0,
    },
    {
        "model_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "role": "flagship_moe_transport_canary",
        "repo_substr": "Qwen3.6-35B-A3B",
        "filename": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        "device_index": 0,
    },
)

FIND_OBJECTS_PREDICATE_CODE = (
    "def accept(obj):\n    return obj['color'] in (2, 4) and obj['pixel_count'] >= 2"
)

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "inference_substrate",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "models_used",
    "live_model_invoked",
    "context_requested",
    "context_observed_by_model",
    "gpu_admission_by_model",
    "rows",
    "multi_parameter_parse_successes",
    "multi_parameter_dispatch_successes",
    "bounded_response_successes",
    "arc_context_tool_preflight_ready",
    "solve_claim",
    "live_path_reached",
    "gate_check_summary",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES: JsonDict = {
    "schema": "A versioned contract lets downstream readers reject incompatible shapes.",
    "experiment": "The numeric ARC root binds this artifact to Exp6752.",
    "title": "The title states the narrow transport-only scope in human-readable form.",
    "run_date": "The fixed planning date anchors the evidence to its requested run.",
    "status": "The terminal status distinguishes a ready, partial, or blocked run.",
    "field_principles": "Each field and gate states why an auditor needs it.",
    "inference_substrate": "The declaration excludes CPU, remote, and helper-only substitutes.",
    "duration_s": "A monotonic interval makes real model work visible.",
    "random_seed": "The fixed seed makes both bounded requests repeatable.",
    "reproducibility_checksum": "The fixture, config, transcript, and row hashes detect drift.",
    "models_used": "Exact IDs, roles, paths, and hashes prevent model substitution.",
    "live_model_invoked": "Readiness needs a real decode from both models.",
    "context_requested": "The fixed 32768 request is the task-owned admission target.",
    "context_observed_by_model": "Server values distinguish runtime context from declared intent.",
    "gpu_admission_by_model": "PID, device, layers, and VRAM bind each decode to CUDA.",
    "rows": "One row per model preserves every call-shape receipt.",
    "multi_parameter_parse_successes": "The count proves typed multi-parameter XML parsed.",
    "multi_parameter_dispatch_successes": "The count proves parsed calls reached shared dispatch.",
    "bounded_response_successes": "The count proves tool output returned within its byte cap.",
    "arc_context_tool_preflight_ready": "The downstream gate needs both full transport receipts.",
    "solve_claim": "False prevents a transport result from becoming a solve claim.",
    "live_path_reached": "The flag distinguishes the production loop from direct helper calls.",
    "gate_check_summary": "A blocked result names each failed check and observed value.",
    "verdict_class": "A closed class makes the terminal state machine-readable.",
    "honest_verdict": "The terminal prefix states completion or an owned block plainly.",
    "preconditions_checked": "Complete host receipts explain why live inference did or did not start.",
    "fixture_manifest": "The bounded synthetic input proves no game source or solve adapter was used.",
    "fixture_checksum": "The fixture digest detects any input mutation.",
    "claim_boundary": "The prose boundary prevents transport evidence from implying game quality.",
    "any_live_decode_before_terminal_state": "Partial runs retain whether any real decode occurred.",
    "gate:exact_models": "Both fixed model IDs must be present; substitutes do not count.",
    "gate:cuda_offload": "CUDA support must be observed before any model starts.",
    "gate:cached_paths": "Exact local GGUF paths prevent downloads and model drift.",
    "gate:free_vram": "Each sequential model needs enough free memory before launch.",
    "gate:registry_no_solve_target": "A transport preflight must not duplicate or add a solve.",
    "gate:owned_32k": "Each server must report at least the requested context.",
    "gate:multi_parameter_parse": "Each model must emit the typed code-carrying call.",
    "gate:dispatch": "Each parsed call must execute through the shared dispatcher.",
    "gate:bounded_response": "Each tool result must stay inside the production bound.",
    "gate:no_solve": "Every row and the artifact must keep solve_claim false.",
}

VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
REQUIRED_GATE_PRINCIPLES = {key for key in FIELD_PRINCIPLES if key.startswith("gate:")}

GPU_LAYER_PATTERNS = (
    re.compile(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers\s+to\s+GPU", re.I),
    re.compile(r"offloaded\s+(\d+)\s+repeating layers", re.I),
)


def canonical_json(value: Any) -> str:
    """Return stable compact JSON for evidence hashes."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_text(value: str) -> str:
    """Hash UTF-8 text and name the algorithm in the receipt."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash one JSON-compatible value after canonical serialization."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash a cached GGUF in chunks so memory use stays small."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def row_checksum(row: Mapping[str, Any]) -> str:
    """Hash a row without its self-referential field."""

    return sha256_json({key: value for key, value in row.items() if key != "row_sha256"})


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind the terminal artifact without its self-referential checksum."""

    return sha256_json(
        {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    )


def fixture_manifest() -> JsonDict:
    """Return the fixed synthetic transition without any game identity."""

    before = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 2, 0, 0, 3, 0, 0],
        [0, 2, 0, 0, 0, 3, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 4, 4, 4, 0, 5, 5, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 6, 0, 7, 7, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
    after = deepcopy(before)
    after[1][1:3] = [4, 4]
    after[2][1] = 4
    return {
        "source_class": "synthetic_bounded_transition_fixture",
        "transition": {
            "grid": before,
            "action": 1,
            "data": None,
            "next_grid": after,
            "reward": 0,
            "done": 0,
        },
    }


def fixture_checksum() -> str:
    """Hash the fixed fixture that both model workers receive."""

    return sha256_json(fixture_manifest())


def fixture_transitions() -> list[Any]:
    """Build production Transition values from the JSON-safe fixture."""

    from carnot.agentic.arc_executable_world_model import Transition

    source = fixture_manifest()["transition"]
    return [
        Transition(
            np.asarray(source["grid"], dtype=np.int16),
            int(source["action"]),
            source["data"],
            np.asarray(source["next_grid"], dtype=np.int16),
            int(source["reward"]),
            int(source["done"]),
        )
    ]


def expected_xml_call() -> str:
    """Return the exact multi-parameter call requested from each model."""

    return (
        "<tool_call>\n"
        "<function=find_objects>\n"
        "<parameter=t>\n0\n</parameter>\n"
        "<parameter=which>\nbefore\n</parameter>\n"
        f"<parameter=predicate_code>\n{FIND_OBJECTS_PREDICATE_CODE}\n</parameter>\n"
        f"<parameter=max_objects>\n{REQUESTED_MAX_OBJECTS}\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )


def build_probe_instruction() -> str:
    """Tell the live model to emit one exact call and no substitute shape."""

    return (
        "TRANSPORT PREFLIGHT. This is not a game-solving request. Your FIRST AND ONLY tool call "
        "must be the exact XML below. Copy every parameter and the Python code exactly. Do not "
        "call list_transitions. Do not write an engine. Stop after the closing tool_call tag.\n\n"
        + expected_xml_call()
    )


def model_receipt(
    spec: Mapping[str, Any],
    path: str | Path,
    *,
    file_hasher: Callable[[str | Path], str] = sha256_file,
) -> JsonDict:
    """Bind a fixed model ID to one exact cached GGUF and content hash."""

    resolved = Path(path).resolve()
    present = resolved.is_file() and resolved.name == spec["filename"]
    size = resolved.stat().st_size if present else 0
    return {
        **dict(spec),
        "resolved": present,
        "model_path": str(resolved) if present else str(path),
        "model_size_bytes": size,
        "model_sha256": file_hasher(resolved) if present else None,
        "required_vram_mb": math.ceil(size / (1024 * 1024)) + VRAM_GUARD_MB if present else 0,
    }


def resolve_model_specs() -> list[JsonDict]:
    """Resolve only the two mandated GGUFs through the production cache resolver."""

    from carnot.agentic.arc_executable_world_model import _resolve_gguf

    rows = []
    for spec in MODEL_SPECS:
        path = _resolve_gguf(str(spec["repo_substr"]))
        rows.append(model_receipt(spec, path or ""))
    return rows


def _run_text_command(command: Sequence[str], timeout_s: float = 10.0) -> JsonDict:
    """Run one host probe and retain its return state."""

    try:
        result = subprocess.run(
            list(command), capture_output=True, text=True, timeout=timeout_s, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "command": list(command),
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "ok": False,
        }
    return {
        "command": list(command),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "ok": result.returncode == 0,
    }


def nvidia_smi_inventory() -> JsonDict:
    """Read physical CUDA identities and current free memory."""

    receipt = _run_text_command(
        (
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        )
    )
    devices = []
    for line in receipt["stdout"].splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        try:
            devices.append(
                {
                    "index": int(parts[0]),
                    "uuid": parts[1],
                    "name": parts[2],
                    "memory_total_mb": int(parts[3]),
                    "memory_used_mb": int(parts[4]),
                    "memory_free_mb": int(parts[5]),
                }
            )
        except ValueError:
            continue
    return {**receipt, "devices": devices}


def live_preflight(models: list[JsonDict]) -> JsonDict:
    """Check CUDA, exact caches, free VRAM, server, and the no-solve registry gate."""

    try:
        from llama_cpp import llama_cpp

        offload: Any = bool(llama_cpp.llama_supports_gpu_offload())
    except Exception as exc:  # noqa: BLE001 - the observed import failure belongs in the artifact
        offload = f"{type(exc).__name__}: {exc}"
    inventory = nvidia_smi_inventory()
    by_index = {row["index"]: row for row in inventory["devices"]}
    registry_text = REGISTRY_PATH.read_text() if REGISTRY_PATH.is_file() else ""
    registry_ok = bool(registry_text) and "6752" not in registry_text
    server_path = Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"
    checks: list[JsonDict] = [
        {
            "check": "llama_cpp_cuda_offload",
            "expected": True,
            "observed": offload,
            "passed": offload is True,
        },
        {
            "check": "llama_server_cuda_binary",
            "expected": True,
            "observed": server_path.is_file(),
            "path": str(server_path),
            "passed": server_path.is_file(),
        },
        {
            "check": "arc_registry_no_solve_target",
            "expected": True,
            "observed": registry_ok,
            "registry_sha256": sha256_text(registry_text) if registry_text else None,
            "passed": registry_ok,
        },
    ]
    for model in models:
        checks.append(
            {
                "check": f"cache:{model['model_id']}",
                "expected": {"filename": model["filename"]},
                "observed": {
                    "resolved": model.get("resolved"),
                    "path": model.get("model_path"),
                },
                "passed": model.get("resolved") is True,
            }
        )
        device = by_index.get(int(model["device_index"]))
        observed_free = device.get("memory_free_mb") if device else None
        required = int(model.get("required_vram_mb") or 0)
        checks.append(
            {
                "check": f"free_vram:{model['model_id']}",
                "expected": {"at_least_mb": required},
                "observed": observed_free,
                "device_index": model["device_index"],
                "passed": observed_free is not None and required > 0 and observed_free >= required,
            }
        )
    return {
        "all_passed": all(check["passed"] is True for check in checks),
        "checks": checks,
        "gpu_inventory": inventory,
    }


def worker_environment(
    base: Mapping[str, str], model: Mapping[str, Any], *, device_index: int
) -> dict[str, str]:
    """Build the explicit owned environment before the worker constructs a proposer."""

    env = dict(base)
    env.update(
        {
            "CARNOT_ARC_INDUCE_N_CTX": str(CONTEXT_REQUESTED),
            "CARNOT_ARC_INDUCE_TOOL_LOOP": "selfparse",
            "CARNOT_ARC_INDUCE_TOOL_TURNS": "1",
            "CARNOT_ARC_INDUCE_TOOL_THINK_BUDGET": "512",
            "CARNOT_ARC_INDUCE_MAX_TOKENS": "1024",
            "CARNOT_ARC_INDUCE_TIMEOUT": str(int(WORKER_TIMEOUT_S)),
            "CARNOT_ARC_GENERATOR_CUDA_GPU": str(device_index),
            "CARNOT_ARC_GENERATOR_REQUIRE_CUDA": "1",
            "CARNOT_ARC_GENERATOR_SEED": str(RANDOM_SEED),
            "CARNOT_ARC_MTP": "0",
            "CARNOT_ARC_KV_QUANT": "q8_0",
            "CARNOT_ARC_GGUF_PATH": str(model["model_path"]),
            "PYTHONPATH": str(REPO_ROOT / "python")
            + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
        }
    )
    return env


def _pid_vram_mb(pid: int) -> int:
    """Return current nvidia-smi memory for one owned server PID."""

    receipt = _run_text_command(
        (
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        )
    )
    values = []
    for line in receipt["stdout"].splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            if int(parts[0]) == int(pid):
                values.append(int(parts[1]))
        except ValueError:
            continue
    return max(values, default=0)


def _gpu_layers_from_log(text: str, requested: int) -> JsonDict:
    """Parse the runtime's own CUDA layer receipt from server stderr."""

    for pattern in GPU_LAYER_PATTERNS:
        matches = list(pattern.finditer(text))
        if not matches:
            continue
        match = matches[-1]
        offloaded = int(match.group(1))
        total = int(match.group(2)) if match.lastindex and match.lastindex >= 2 else offloaded
        return {"requested": requested, "offloaded": offloaded, "total": total}
    return {"requested": requested, "offloaded": 0, "total": None}


def _assigned_device(device_index: int) -> JsonDict:
    """Return the physical device identity assigned to this worker."""

    inventory = nvidia_smi_inventory()
    device = next((row for row in inventory["devices"] if row["index"] == device_index), {})
    return {
        "physical_index": device_index,
        "uuid": device.get("uuid"),
        "name": device.get("name"),
    }


def run_live_worker(model: Mapping[str, Any], *, device_index: int) -> JsonDict:
    """Run one model through the production loop inside its owned worker process."""

    from carnot.agentic import arc_induction_tool_loop as loop
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer, _free_port

    start_ns = time.monotonic_ns()
    proposer = LocalGGUFProposer(
        repo_substr=str(model["repo_substr"]),
        model_path=str(model["model_path"]),
        n_ctx=CONTEXT_REQUESTED,
        max_tokens=1024,
        timeout=int(WORKER_TIMEOUT_S),
        port=_free_port(),
        mtp=False,
        n_gpu_layers=999,
        use_chat_template=True,
    )
    events: list[JsonDict] = []
    peak_vram = 0
    stop_monitor = threading.Event()
    server_pid = 0
    loop_started = False
    loop_ok = False
    loop_note = ""
    ensure_error = None
    try:
        if not proposer._ensure_server():
            ensure_error = "server_unavailable"
        else:
            server_pid = int(getattr(proposer._proc, "pid", 0) or 0)

            def monitor() -> None:
                nonlocal peak_vram
                while not stop_monitor.wait(0.1):
                    peak_vram = max(peak_vram, _pid_vram_mb(server_pid))

            monitor_thread = threading.Thread(target=monitor, daemon=True)
            monitor_thread.start()
            loop_started = True
            loop_ok, loop_note = loop.induce_with_tool_loop(
                proposer,
                "transport_fixture",
                fixture_transitions(),
                1,
                extra_user_instruction=build_probe_instruction(),
                tool_event_sink=events,
            )
            peak_vram = max(peak_vram, _pid_vram_mb(server_pid))
            stop_monitor.set()
            monitor_thread.join(timeout=1)
    except Exception as exc:  # noqa: BLE001 - every live failure becomes a retained row
        ensure_error = f"{type(exc).__name__}: {exc}"[:500]
    finally:
        stop_monitor.set()

    latency_s = round((time.monotonic_ns() - start_ns) / 1_000_000_000, 6)
    observed_context = proposer.observed_n_ctx() if proposer._healthy() else None
    log_path = getattr(proposer, "_stderr_log_path", None)
    try:
        log_text = Path(log_path).read_text(errors="replace") if log_path else ""
    except OSError:
        log_text = ""
    gpu_layers = _gpu_layers_from_log(log_text, proposer.n_gpu_layers)
    event = next((row for row in events if row.get("parsed_tool") == "find_objects"), None)
    raw_emission = str((event or {}).get("raw_emission") or "")
    bounded_response = str((event or {}).get("bounded_response") or "")
    dispatch_result = deepcopy((event or {}).get("dispatch_result"))
    parsed_arguments = deepcopy((event or {}).get("parsed_arguments"))
    stats = getattr(proposer, "last_tool_loop_stats", {})
    live_decode = bool(stats.get("turns", 0) > 0 and raw_emission)
    if ensure_error:
        failure_class = "cuda_admission_or_worker_failure"
    elif event is None:
        failure_class = "no_live_find_objects_call"
    elif not isinstance(dispatch_result, dict) or dispatch_result.get("ok") is not True:
        failure_class = "dispatch_failure"
    elif len(bounded_response.encode("utf-8")) > MAX_FIND_OBJECT_RESPONSE_BYTES + 64:
        failure_class = "response_bound_failure"
    else:
        failure_class = None
    row: JsonDict = {
        **dict(model),
        "call_shape": "find_objects_t_which_predicate_code_max_objects",
        "owned_pid": os.getpid(),
        "server_pid": server_pid,
        "assigned_device": _assigned_device(device_index),
        "context_requested": CONTEXT_REQUESTED,
        "context_observed_by_model": observed_context,
        "gpu_layers": gpu_layers,
        "peak_vram_mb": peak_vram,
        "live_model_invoked": live_decode,
        "live_path_reached": loop_started,
        "production_route": PRODUCTION_ROUTE,
        "raw_emission_sha256": sha256_text(raw_emission),
        "parsed_tool": (event or {}).get("parsed_tool"),
        "parsed_arguments": parsed_arguments,
        "dispatch_result": dispatch_result,
        "bounded_response_bytes": len(bounded_response.encode("utf-8")),
        "bounded_response_sha256": sha256_text(bounded_response),
        "latency_s": latency_s,
        "failure_class": failure_class,
        "transcript_sha256": sha256_json([raw_emission, bounded_response]),
        "loop_returned_success": loop_ok,
        "loop_note_sha256": sha256_text(loop_note),
        "process_exit_code": 0,
        "solve_claim": False,
    }
    row["row_sha256"] = row_checksum(row)
    proposer.stop()
    return row


def _failed_worker_row(model: Mapping[str, Any], failure: str, exit_code: int | None) -> JsonDict:
    """Preserve one model denominator when its owned subprocess cannot return a row."""

    row: JsonDict = {
        **dict(model),
        "call_shape": "find_objects_t_which_predicate_code_max_objects",
        "owned_pid": None,
        "server_pid": None,
        "assigned_device": None,
        "context_requested": CONTEXT_REQUESTED,
        "context_observed_by_model": None,
        "gpu_layers": {"requested": 999, "offloaded": 0, "total": None},
        "peak_vram_mb": 0,
        "live_model_invoked": False,
        "live_path_reached": False,
        "production_route": PRODUCTION_ROUTE,
        "raw_emission_sha256": sha256_text(""),
        "parsed_tool": None,
        "parsed_arguments": None,
        "dispatch_result": None,
        "bounded_response_bytes": 0,
        "bounded_response_sha256": sha256_text(""),
        "latency_s": 0.0,
        "failure_class": failure,
        "transcript_sha256": sha256_json([]),
        "process_exit_code": exit_code,
        "solve_claim": False,
    }
    row["row_sha256"] = row_checksum(row)
    return row


def run_model_subprocess(
    model: Mapping[str, Any],
    *,
    device_index: int | None = None,
    timeout_s: float = WORKER_TIMEOUT_S,
) -> JsonDict:
    """Run one model in a fresh interpreter with an explicit 32K environment."""

    device = int(model["device_index"] if device_index is None else device_index)
    with tempfile.TemporaryDirectory(prefix="carnot_exp6752_") as tmp_dir:
        output_path = Path(tmp_dir) / "worker.json"
        command = [
            sys.executable,
            "-m",
            "carnot.experiment_6752_arc_code_carrying_tool_preflight",
            "--worker",
            "--worker-output",
            str(output_path),
            "--model-json",
            canonical_json(dict(model)),
            "--device-index",
            str(device),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=worker_environment(os.environ, model, device_index=device),
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return _failed_worker_row(model, "worker_timeout", None)
        except OSError as exc:
            return _failed_worker_row(model, f"worker_launch_{type(exc).__name__}", None)
        if not output_path.is_file():
            detail = (completed.stderr or completed.stdout or "missing worker output")[-300:]
            return _failed_worker_row(
                model, f"worker_no_output:{sha256_text(detail)}", completed.returncode
            )
        try:
            row = json.loads(output_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            return _failed_worker_row(
                model, f"worker_output_{type(exc).__name__}", completed.returncode
            )
        row["process_exit_code"] = completed.returncode
        row["row_sha256"] = row_checksum(row)
        return row


def model_row_errors(row: Mapping[str, Any]) -> list[str]:
    """Return every reason one model row cannot satisfy readiness."""

    errors = []
    if row.get("row_sha256") != row_checksum(row):
        errors.append("row_sha256")
    if row.get("model_id") not in {spec["model_id"] for spec in MODEL_SPECS}:
        errors.append("model_id")
    if row.get("context_requested") != CONTEXT_REQUESTED:
        errors.append("context_requested")
    observed = row.get("context_observed_by_model")
    if not isinstance(observed, int) or observed < CONTEXT_REQUESTED:
        errors.append("context_observed_by_model")
    layers = row.get("gpu_layers") if isinstance(row.get("gpu_layers"), dict) else {}
    if not isinstance(layers.get("offloaded"), int) or layers.get("offloaded", 0) <= 0:
        errors.append("gpu_layers")
    if not isinstance(row.get("peak_vram_mb"), int) or row.get("peak_vram_mb", 0) <= 0:
        errors.append("peak_vram_mb")
    if row.get("live_model_invoked") is not True:
        errors.append("live_model_invoked")
    if row.get("live_path_reached") is not True or row.get("production_route") != PRODUCTION_ROUTE:
        errors.append("production_route")
    if row.get("parsed_tool") != "find_objects":
        errors.append("parsed_tool")
    args = row.get("parsed_arguments") if isinstance(row.get("parsed_arguments"), dict) else {}
    if not (
        type(args.get("t")) is int
        and isinstance(args.get("which"), str)
        and isinstance(args.get("predicate_code"), str)
        and type(args.get("max_objects")) is int
        and args.get("predicate_code") == FIND_OBJECTS_PREDICATE_CODE
    ):
        errors.append("parsed_arguments")
    dispatch = row.get("dispatch_result") if isinstance(row.get("dispatch_result"), dict) else {}
    if dispatch.get("ok") is not True:
        errors.append("dispatch_result")
    response_bytes = row.get("bounded_response_bytes")
    if (
        not isinstance(response_bytes, int)
        or response_bytes <= 0
        or response_bytes > MAX_FIND_OBJECT_RESPONSE_BYTES + 64
    ):
        errors.append("bounded_response")
    if row.get("failure_class") is not None:
        errors.append("failure_class")
    if row.get("process_exit_code") != 0:
        errors.append("process_exit_code")
    if row.get("solve_claim") is not False:
        errors.append("solve_claim")
    return errors


def row_evidence_errors(row: Mapping[str, Any]) -> list[str]:
    """Check row integrity without treating an honest failed gate as malformed evidence."""

    errors = []
    if row.get("row_sha256") != row_checksum(row):
        errors.append("row_sha256")
    if row.get("model_id") not in {spec["model_id"] for spec in MODEL_SPECS}:
        errors.append("model_id")
    if row.get("context_requested") != CONTEXT_REQUESTED:
        errors.append("context_requested")
    if row.get("call_shape") != "find_objects_t_which_predicate_code_max_objects":
        errors.append("call_shape")
    for field in (
        "raw_emission_sha256",
        "bounded_response_sha256",
        "transcript_sha256",
    ):
        value = row.get(field)
        if not isinstance(value, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
            errors.append(field)
    if not isinstance(row.get("latency_s"), (int, float)) or row.get("latency_s", -1) < 0:
        errors.append("latency_s")
    if row.get("solve_claim") is not False:
        errors.append("solve_claim")
    return errors


def reduce_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Require one valid row from each fixed model and no extra denominator."""

    return bool(
        len(rows) == len(MODEL_SPECS)
        and [row.get("model_id") for row in rows] == [spec["model_id"] for spec in MODEL_SPECS]
        and all(not model_row_errors(row) for row in rows)
    )


def _derived_gate_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Build the terminal transport gates from retained model rows."""

    ready = reduce_ready(rows)
    return [
        {
            "check": "owned_32k_cuda_by_model",
            "expected": len(MODEL_SPECS),
            "observed": sum(
                1
                for row in rows
                if isinstance(row.get("context_observed_by_model"), int)
                and row["context_observed_by_model"] >= CONTEXT_REQUESTED
                and (row.get("gpu_layers") or {}).get("offloaded", 0) > 0
            ),
            "passed": ready,
        },
        {
            "check": "parse_dispatch_bounded_response_by_model",
            "expected": len(MODEL_SPECS),
            "observed": sum(1 for row in rows if not model_row_errors(row)),
            "passed": ready,
        },
        {
            "check": "solve_claim",
            "expected": False,
            "observed": any(row.get("solve_claim") is not False for row in rows),
            "passed": all(row.get("solve_claim") is False for row in rows),
        },
    ]


def build_artifact(
    *,
    rows: Sequence[Mapping[str, Any]],
    preflight: Mapping[str, Any],
    started_ns: int,
    finished_ns: int,
    models: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Reduce preflight checks and model rows into one stable terminal artifact."""

    retained = [deepcopy(dict(row)) for row in rows]
    model_rows = [deepcopy(dict(row)) for row in (models or rows)]
    ready = bool(preflight.get("all_passed") is True and reduce_ready(retained))
    parse_successes = sum(
        1
        for row in retained
        if row.get("parsed_tool") == "find_objects"
        and isinstance(row.get("parsed_arguments"), dict)
        and len(row["parsed_arguments"]) >= 3
    )
    dispatch_successes = sum(
        1
        for row in retained
        if isinstance(row.get("dispatch_result"), dict) and row["dispatch_result"].get("ok") is True
    )
    bounded_successes = sum(
        1
        for row in retained
        if isinstance(row.get("bounded_response_bytes"), int)
        and 0 < row["bounded_response_bytes"] <= MAX_FIND_OBJECT_RESPONSE_BYTES + 64
    )
    live_any = any(row.get("live_model_invoked") is True for row in retained)
    if ready:
        verdict_class = "positive"
        honest_verdict = "complete_arc_context_tool_preflight_ready"
    elif preflight.get("all_passed") is not True:
        verdict_class = "blocked"
        failed = next(
            (check for check in preflight.get("checks", []) if check.get("passed") is not True),
            {"check": "unknown_preflight", "observed": None},
        )
        honest_verdict = f"complete_blocked_arc_transport:{failed.get('check')}"
    else:
        verdict_class = "partial"
        honest_verdict = "complete_partial_arc_transport_live_rows_incomplete"
    gate_summary = (
        deepcopy(list(preflight.get("checks", [])))
        if preflight.get("all_passed") is not True
        else _derived_gate_rows(retained)
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": 6752,
        "title": "Task-owned 32K code-carrying ARC tool preflight",
        "run_date": RUN_DATE,
        "status": "complete" if ready else verdict_class,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(max(0, finished_ns - started_ns) / 1_000_000_000, 6),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "models_used": model_rows,
        "live_model_invoked": bool(retained)
        and all(row.get("live_model_invoked") is True for row in retained),
        "context_requested": CONTEXT_REQUESTED,
        "context_observed_by_model": {
            str(row.get("model_id")): row.get("context_observed_by_model") for row in retained
        },
        "gpu_admission_by_model": {
            str(row.get("model_id")): {
                "assigned_device": deepcopy(row.get("assigned_device")),
                "gpu_layers": deepcopy(row.get("gpu_layers")),
                "peak_vram_mb": row.get("peak_vram_mb"),
                "owned_pid": row.get("owned_pid"),
                "server_pid": row.get("server_pid"),
            }
            for row in retained
        },
        "rows": retained,
        "multi_parameter_parse_successes": parse_successes,
        "multi_parameter_dispatch_successes": dispatch_successes,
        "bounded_response_successes": bounded_successes,
        "arc_context_tool_preflight_ready": ready,
        "solve_claim": False,
        "live_path_reached": bool(retained)
        and all(row.get("live_path_reached") is True for row in retained),
        "gate_check_summary": gate_summary,
        "verdict_class": verdict_class,
        "honest_verdict": honest_verdict,
        "preconditions_checked": deepcopy(dict(preflight)),
        "fixture_manifest": fixture_manifest(),
        "fixture_checksum": fixture_checksum(),
        "claim_boundary": CLAIM_BOUNDARY,
        "any_live_decode_before_terminal_state": live_any,
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute every gate that can be checked from the terminal artifact."""

    errors = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing_field:{field}")
    principles = (
        artifact.get("field_principles")
        if isinstance(artifact.get("field_principles"), dict)
        else {}
    )
    if not set(artifact).issubset(principles):
        errors.append("field_principles_incomplete")
    if not REQUIRED_GATE_PRINCIPLES.issubset(principles):
        errors.append("gate_principles_incomplete")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if (
        not isinstance(artifact.get("duration_s"), (int, float))
        or artifact.get("duration_s", -1) < 0
    ):
        errors.append("duration_s")
    if artifact.get("context_requested") != CONTEXT_REQUESTED:
        errors.append("context_requested")
    if artifact.get("solve_claim") is not False:
        errors.append("solve_claim_must_be_false")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        errors.append("verdict_class")
    rows = artifact.get("rows") if isinstance(artifact.get("rows"), list) else []
    if [row.get("model_id") for row in rows] != [spec["model_id"] for spec in MODEL_SPECS]:
        errors.append("row_denominator")
    models_used = (
        artifact.get("models_used") if isinstance(artifact.get("models_used"), list) else []
    )
    if [row.get("model_id") for row in models_used] != [spec["model_id"] for spec in MODEL_SPECS]:
        errors.append("models_used")
    for row in rows:
        if row_evidence_errors(row):
            errors.append(f"row_invalid:{row.get('model_id')}")
    expected_contexts = {
        str(row.get("model_id")): row.get("context_observed_by_model") for row in rows
    }
    if artifact.get("context_observed_by_model") != expected_contexts:
        errors.append("context_observed_by_model")
    expected_gpu = {
        str(row.get("model_id")): {
            "assigned_device": deepcopy(row.get("assigned_device")),
            "gpu_layers": deepcopy(row.get("gpu_layers")),
            "peak_vram_mb": row.get("peak_vram_mb"),
            "owned_pid": row.get("owned_pid"),
            "server_pid": row.get("server_pid"),
        }
        for row in rows
    }
    if artifact.get("gpu_admission_by_model") != expected_gpu:
        errors.append("gpu_admission_by_model")
    expected_live = bool(rows) and all(row.get("live_model_invoked") is True for row in rows)
    if artifact.get("live_model_invoked") is not expected_live:
        errors.append("live_model_invoked")
    expected_path = bool(rows) and all(row.get("live_path_reached") is True for row in rows)
    if artifact.get("live_path_reached") is not expected_path:
        errors.append("live_path_reached")
    expected_ready = bool(
        artifact.get("preconditions_checked", {}).get("all_passed") is True and reduce_ready(rows)
    )
    if artifact.get("arc_context_tool_preflight_ready") is not expected_ready:
        errors.append("ready_reduction")
    expected_gate_summary = (
        _derived_gate_rows(rows)
        if artifact.get("preconditions_checked", {}).get("all_passed") is True
        else artifact.get("preconditions_checked", {}).get("checks", [])
    )
    if artifact.get("gate_check_summary") != expected_gate_summary:
        errors.append("gate_check_summary")
    if expected_ready:
        expected_verdict = "complete_arc_context_tool_preflight_ready"
    elif artifact.get("preconditions_checked", {}).get("all_passed") is not True:
        failed = next(
            (
                check
                for check in artifact.get("preconditions_checked", {}).get("checks", [])
                if check.get("passed") is not True
            ),
            {"check": "unknown_preflight"},
        )
        expected_verdict = f"complete_blocked_arc_transport:{failed.get('check')}"
    else:
        expected_verdict = "complete_partial_arc_transport_live_rows_incomplete"
    if artifact.get("honest_verdict") != expected_verdict:
        errors.append("honest_verdict")
    if artifact.get("multi_parameter_parse_successes") != sum(
        1
        for row in rows
        if row.get("parsed_tool") == "find_objects"
        and isinstance(row.get("parsed_arguments"), dict)
        and len(row["parsed_arguments"]) >= 3
    ):
        errors.append("parse_count")
    if artifact.get("multi_parameter_dispatch_successes") != sum(
        1
        for row in rows
        if isinstance(row.get("dispatch_result"), dict) and row["dispatch_result"].get("ok") is True
    ):
        errors.append("dispatch_count")
    if artifact.get("bounded_response_successes") != sum(
        1
        for row in rows
        if isinstance(row.get("bounded_response_bytes"), int)
        and 0 < row["bounded_response_bytes"] <= MAX_FIND_OBJECT_RESPONSE_BYTES + 64
    ):
        errors.append("bounded_count")
    if artifact.get("fixture_checksum") != fixture_checksum():
        errors.append("fixture_checksum")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def _atomic_write(path: Path, artifact: Mapping[str, Any]) -> None:
    """Replace the deliverable only after its complete JSON is on disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(artifact, handle, indent=2, sort_keys=False)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(path)


def run(
    *,
    result_path: Path = RESULT_PATH,
    resolver: Callable[[], list[JsonDict]] = resolve_model_specs,
    preflight_fn: Callable[[list[JsonDict]], JsonDict] = live_preflight,
    worker_runner: Callable[[Mapping[str, Any]], JsonDict] = run_model_subprocess,
    clock: Callable[[], int] = time.monotonic_ns,
) -> JsonDict:
    """Run preconditions, then both fresh model subprocesses sequentially."""

    started_ns = clock()
    models = resolver()
    preflight = preflight_fn(models)
    rows: list[JsonDict] = []
    if preflight.get("all_passed") is True:
        for model in models:
            rows.append(worker_runner(model))
    else:
        failed = next(
            (check for check in preflight.get("checks", []) if check.get("passed") is not True),
            {"check": "unknown_preflight"},
        )
        rows = [
            _failed_worker_row(model, f"preflight_blocked:{failed.get('check')}", None)
            for model in models
        ]
    finished_ns = clock()
    artifact = build_artifact(
        rows=rows,
        preflight=preflight,
        started_ns=started_ns,
        finished_ns=finished_ns,
        models=models,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError(f"invalid Exp6752 artifact: {errors}")
    _atomic_write(result_path, artifact)
    return artifact


def _worker_entry(model_json: str, output: Path, device_index: int) -> int:
    """Execute one owned worker and write its single row."""

    model = json.loads(model_json)
    row = run_live_worker(model, device_index=device_index)
    _atomic_write(output, row)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the parent experiment or one explicitly requested model worker."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--model-json")
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args(argv)
    if args.worker:
        if args.worker_output is None or not args.model_json:
            parser.error("--worker requires --worker-output and --model-json")
        return _worker_entry(args.model_json, args.worker_output, args.device_index)
    artifact = run()
    print(
        json.dumps(
            {
                "artifact": str(RESULT_PATH),
                "ready": artifact["arc_context_tool_preflight_ready"],
                "verdict": artifact["honest_verdict"],
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the script wrapper
    raise SystemExit(main())

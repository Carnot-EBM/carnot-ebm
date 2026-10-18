"""Write the fail-fast receipt for the queued Qwen3 XML parser canary.

The serving packages are external prerequisites. This module records their
absence before it touches a tokenizer, GPU lease, model, or server. It keeps
the four planned calls visible so a package block cannot look like a completed
parser trial.

Spec refs: REQ-ARC-WMTE-7220 and SCENARIO-ARC-WMTE-7220-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.request

from carnot.experiment_artifacts import atomic_write_json
from carnot.inference.llama_server_supervisor import (
    canonical_json,
    sha256_bytes,
    sha256_file,
    sha256_text,
    utc_now,
    write_bytes_atomic,
)
from carnot.inference.sota_models import cached_current_model


JsonDict = dict[str, Any]
Importer = Callable[[str], object]
VersionReader = Callable[[str], str]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260911"
RANDOM_SEED = 7_220_202_609_11
RESULT_PATH = Path("results/experiment_7220_v636_xml_canary.json")
RAW_DIR = Path("results/raw/experiment_7220")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7220_v636_xml_canary.json")
SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
MODULE_PATH = Path("python/carnot/experiment_7220_v636_xml_canary.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7220_v636_xml_canary.py")
TEST_PATH = Path("tests/python/test_experiment_7220_v636_xml_canary.py")

MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
QUANTIZATION = "Q4_K_M"
GPU_INDEX = 1
EXPECTED_TOOL_NAMES = (
    "query_region",
    "diff_grids",
    "run_engine_on_transitions",
    "list_transitions",
)
MANDATED_MODEL_SPEC: JsonDict = {
    "hf_id": MODEL_ID,
    "quantization": QUANTIZATION,
}

#: vLLM serving constants for the live canary path. A fixed high port keeps this
#: canary from colliding with the conductor's own generator on GPU 0.
VLLM_PORT = 8712
VLLM_STARTUP_TIMEOUT_S = 420.0
VLLM_HEALTH_POLL_INTERVAL_S = 3.0
VLLM_REQUEST_TIMEOUT_S = 180.0
VLLM_LEASE_TTL_S = 60.0
TASK_ID = "exp7220-xml-canary"

#: One user prompt per expected tool, worded to make that specific tool the
#: obviously useful next step -- this is a MECHANISM canary (does tool_calls
#: populate at all), not a puzzle-solving quality measurement.
TOOL_PROMPTS: dict[str, str] = {
    "query_region": (
        "You are inspecting an ARC grid. You need to see the pixel values in the "
        "top-left 5x5 region before deciding what to do next. Use the appropriate tool."
    ),
    "diff_grids": (
        "You have a before-grid and an after-grid from one action. You need to know "
        "exactly which cells changed. Use the appropriate tool."
    ),
    "run_engine_on_transitions": (
        "You have written candidate Python code for a transition-prediction engine "
        "and need to check its accuracy against observed transitions before trusting "
        "it. Use the appropriate tool."
    ),
    "list_transitions": (
        "You need to see the full list of recorded state transitions collected so "
        "far in this session before choosing your next action. Use the appropriate "
        "tool."
    ),
}


def _tool_schema_for(name: str) -> JsonDict:
    """The one named tool's OpenAI-shaped schema, reused from the live induction loop.

    Imported lazily so this module's own package-preflight path never needs the
    agentic package's heavier import graph before packages are known importable.
    """

    from carnot.agentic.arc_induction_tools import TOOL_SCHEMAS

    for schema in TOOL_SCHEMAS:
        if schema["function"]["name"] == name:
            return deepcopy(schema)
    raise KeyError(f"no TOOL_SCHEMAS entry named {name!r}")


SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("ops/known-issues.md"),
    Path("docs/research-notes/avo-adaptation-for-local-generator-2026-08-21.md"),
    Path("python/carnot/agentic/arc_induction_tool_loop.py"),
    Path("python/carnot/inference/sota_models.py"),
    Path("python/carnot/inference/llama_server_supervisor.py"),
    Path("python/carnot/gpu_lease_phase_journal.py"),
    SPEC_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)

PACKAGE_IMPORTS = (
    ("vllm", "vllm", "vllm_import"),
    ("vllm-gguf-plugin", "vllm_gguf_plugin", "vllm_gguf_plugin_import"),
)

REQUIRED_FIELD_PRINCIPLES: dict[str, str] = {
    "field_principles": "Annotate actual values in this map; do not wrap arbitrary dictionaries as principle/value records.",
    "status": "Write a terminal artifact only when done or externally blocked; running checkpoints use a different path.",
    "run_date": "Use 20260911 and record actual UTC timestamps, never copy an upstream run date.",
    "preconditions_checked": "Actual code, resource, identity and gate observations before expensive work.",
    "inference_substrate": "Use the recognized literal for the work actually executed; custom free text caused the Exp7208 quarantine.",
    "inference_substrate_class": "Match actual generation, load-only, CPU or aggregation work and its duration floor.",
    "execution_venue": "Exactly host, kv260, gatemate or polarfire; the top-level orchestration here is host.",
    "execution_host": "Actual hostname separate from venue.",
    "duration_s": "Measured monotonic work time; no padding or reclassification to evade a floor.",
    "source_artifact_hashes": "Bind code, source documents, manifests and raw evidence to claims.",
    "rows": "Per unit/arm/seed metric, error and abstention for every comparison; retain full denominators.",
    "sample_size_budget": "Planned, attempted, completed, censored and independent units; no silent removal.",
    "random_seed": "Freeze random choices before reading held-out outcomes.",
    "reproducibility_checksum": "Hash exact source, inputs, settings and raw unit rows.",
    "gate_check_summary": "Every blocked_* verdict names failed check, upstream, field, expected and observed value.",
    "verifier_is_oracle": "True when correctness authority is reused as the verifier; independent code alone is not distinct authority.",
    "verdict_class": "Closed enum positive | circular_positive | null | blocked | disqualified | partial. partial means unfinished own work only.",
    "honest_verdict": "Use complete_ or complete: for completed findings; blocked_* for external absence. A failed acceptance gate forbids positive.",
    "MODEL_SPECS": "Only models actually invoked; [] for CPU/aggregation, mandated Qwen3.8 for every model task.",
    "model_invoked": "True only for actual model execution; upstream model outputs are cached evidence.",
    "xml_canary_complete_score": "The bounded scheduled observations exist.",
    "xml_transport_ready_score": "All four actual tool calls conform; not a model-quality claim.",
    "parser_rows": "One row per prompt and actual parser control with byte hashes.",
    "failure_stage": "Disambiguate package, tokenizer, load, generation and parsing.",
    "quant_path": "Exact GGUF plugin and quantization path.",
    "token_budget_receipt": "Actual tokens, limits and stop reasons.",
    "model_identity_receipt": "GGUF revision/hash, exact loader and tokenizer, and actual CUDA execution.",
    "gpu_receipts": "Task-owned lease and correlated CUDA observations.",
    "phase_spans": "Monotonic load/generate/score/cleanup spans.",
    "runner_receipt": "One model, runner choice, PID identity and cleanup.",
}


class LiveExecutionRequired(RuntimeError):
    """Stop the thin preflight when the missing-package terminal path does not apply."""


def progress(phase: int, state: str, detail: str) -> None:
    """Flush each observed boundary so preflight work never appears silent."""

    print(f"[exp7220] phase {phase} {state}: {detail}", flush=True)


def package_preflight(
    importer: Importer = importlib.import_module,
    version_reader: VersionReader = importlib.metadata.version,
) -> list[JsonDict]:
    """Import both required packages and preserve each exact outcome."""

    rows: list[JsonDict] = []
    for distribution, module, check in PACKAGE_IMPORTS:
        try:
            importer(module)
        except Exception as exc:
            rows.append(
                {
                    "check": check,
                    "package": distribution,
                    "module": module,
                    "importable": False,
                    "version": None,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue
        try:
            version = version_reader(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = "distribution_metadata_missing"
        rows.append(
            {
                "check": check,
                "package": distribution,
                "module": module,
                "importable": True,
                "version": version,
                "error_type": None,
                "error": None,
            }
        )
    return rows


def first_package_block(rows: Sequence[Mapping[str, Any]]) -> JsonDict | None:
    """Return the first package failure in the required gate-summary shape."""

    failed = next((row for row in rows if row.get("importable") is not True), None)
    if failed is None:
        return None
    return {
        "failed_check": str(failed.get("check")),
        "upstream": ".venv",
        "field": f"{str(failed.get('package')).replace('-', '_')}_importable",
        "expected_value": True,
        "observed_value": False,
        "passed": False,
    }


def _integer(text: str) -> int:
    """Read the leading integer from one nvidia-smi CSV field."""

    return int(text.strip().split()[0])


def parse_gpu_receipt(query_bytes: bytes, app_bytes: bytes) -> JsonDict:
    """Turn exact nvidia-smi bytes into the GPU 1 ownership preflight."""

    query_rows = [line.split(",") for line in query_bytes.decode("utf-8", "replace").splitlines()]
    selected = next(
        (
            row
            for row in query_rows
            if len(row) == 6 and row[0].strip().isdigit() and int(row[0]) == GPU_INDEX
        ),
        None,
    )
    receipt: JsonDict = {
        "selected_gpu_index": GPU_INDEX,
        "query_sha256": sha256_bytes(query_bytes),
        "compute_apps_sha256": sha256_bytes(app_bytes),
        "compute_processes": [],
        "lease_acquired": False,
        "lease_reason": "package_preflight_stopped_before_lease",
    }
    if selected is None:
        receipt.update(
            selected_gpu_uuid=None,
            selected_gpu_name=None,
            memory_total_mb=None,
            memory_used_mb=None,
            utilization_percent=None,
            task_ownable=False,
            parse_error="gpu_1_row_missing_or_malformed",
        )
        return receipt

    uuid = selected[1].strip()
    processes: list[JsonDict] = []
    for line in app_bytes.decode("utf-8", "replace").splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 4 and fields[0] == uuid:
            processes.append(
                {
                    "gpu_uuid": fields[0],
                    "pid": _integer(fields[1]),
                    "process_name": fields[2],
                    "used_memory_mb": _integer(fields[3]),
                }
            )
    utilization = _integer(selected[5])
    receipt.update(
        selected_gpu_uuid=uuid,
        selected_gpu_name=selected[2].strip(),
        memory_total_mb=_integer(selected[3]),
        memory_used_mb=_integer(selected[4]),
        utilization_percent=utilization,
        compute_processes=processes,
        task_ownable=not processes and utilization == 0,
        parse_error=None,
    )
    return receipt


def read_gpu_bytes() -> tuple[bytes, bytes]:
    """Collect bounded, unbuffered nvidia-smi evidence without changing GPU state."""

    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    apps = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    return query.stdout, apps.stdout


def resolve_quant_path() -> str | None:
    """Reuse the single-model registry and never download or substitute weights."""

    cached = cached_current_model(gpu_index=GPU_INDEX, preferred_quant=QUANTIZATION)
    return str(cached["model_path"]) if cached is not None else None


def quant_identity(path: str | Path) -> JsonDict:
    """Bind the cached snapshot name, target blob, size, and exact bytes."""

    cached = Path(path)
    resolved = cached.resolve(strict=True)
    revision = cached.parent.name if cached.parent.parent.name == "snapshots" else None
    return {
        "hf_id": MODEL_ID,
        "quantization": QUANTIZATION,
        "cached_path": str(cached),
        "resolved_path": str(resolved),
        "filename": cached.name,
        "revision": revision,
        "blob_identity": resolved.name,
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def vllm_binary_path() -> Path:
    """The vllm CLI, resolved next to the interpreter running THIS process.

    Never resolved via PATH: the point of the isolated trial venv is that its
    torch/CUDA stack never leaks into or depends on the project's own .venv, so
    the binary must come from the same venv as sys.executable, not whatever a
    shell's PATH happens to find first.
    """

    return Path(sys.executable).parent / "vllm"


#: vLLM's own GGUF doc: prefer the base repo's tokenizer over GGUF-embedded
#: conversion (slow, unstable for large-vocab models), and provide the base
#: repo's config when the bare GGUF blob carries no model_type. Base repo,
#: not the -GGUF repo: https://docs.vllm.ai/en/stable/features/quantization/gguf/.
VLLM_GGUF_BASE_REPO = "unsloth/Qwen3.8-27B"


def launch_vllm_server(model_path: str, port: int, gpu_index: int) -> subprocess.Popen:
    """Start vLLM's OpenAI-compatible server with the qwen3_xml tool parser.

    CUDA_VISIBLE_DEVICES pins this process to one physical GPU regardless of
    what index vLLM would otherwise pick, matching the task-owned-GPU
    discipline every other live ARC/inference task in this project follows.
    """

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    cmd = [
        str(vllm_binary_path()),
        "serve",
        model_path,
        "--tokenizer",
        VLLM_GGUF_BASE_REPO,
        "--hf-config-path",
        VLLM_GGUF_BASE_REPO,
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_xml",
        "--port",
        str(port),
        "--gpu-memory-utilization",
        "0.85",
        "--max-model-len",
        "8192",
        "--dtype",
        "auto",
    ]
    return subprocess.Popen(  # noqa: S603 - fixed argv, no shell, trusted local binary
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )


def wait_for_server_health(
    proc: subprocess.Popen,
    port: int,
    timeout_s: float,
    poll_interval_s: float,
    on_wait: Callable[[], None] | None = None,
) -> tuple[bool, str, float]:
    """Poll /health until it answers, the process exits, or the timeout expires.

    Returns (healthy, reason, elapsed_s). `on_wait` fires once per poll so a
    caller can renew a GPU lease during a multi-minute model load without this
    function knowing anything about leases.
    """

    started = time.monotonic()
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() - started < timeout_s:
        exit_code = proc.poll()
        if exit_code is not None:
            return False, f"server_exited_early:{exit_code}", time.monotonic() - started
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:  # noqa: S310
                if resp.status == 200:
                    return True, "healthy", time.monotonic() - started
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        if on_wait is not None:
            on_wait()
        time.sleep(poll_interval_s)
    return False, "startup_timeout", time.monotonic() - started


def send_tool_call_probe(
    port: int,
    model_path: str,
    tool_name: str,
    prompt: str,
    timeout_s: float,
) -> JsonDict:
    """One real HTTP round-trip: does the server's tool_calls field populate?

    Sends only the ONE tool this prompt targets, so a populated tool_calls
    entry unambiguously names which mechanism worked -- never inferred from
    which tool a multi-tool response happened to pick.
    """

    schema = _tool_schema_for(tool_name)
    payload = {
        "model": model_path,
        "messages": [{"role": "user", "content": prompt}],
        "tools": [schema],
        "tool_choice": "auto",
        "max_tokens": 512,
        "temperature": 0.0,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
            raw = resp.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {
            "tool_name": tool_name,
            "ok": False,
            "error": repr(exc),
            "tool_calls": None,
            "elapsed_s": round(time.monotonic() - started, 3),
            "response_sha256": None,
        }
    elapsed = time.monotonic() - started
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {
            "tool_name": tool_name,
            "ok": False,
            "error": f"non_json_response:{exc}",
            "tool_calls": None,
            "elapsed_s": round(elapsed, 3),
            "response_sha256": sha256_bytes(raw),
        }
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    tool_calls = message.get("tool_calls")
    return {
        "tool_name": tool_name,
        "ok": True,
        "error": None,
        "tool_calls": tool_calls,
        "populated": bool(tool_calls),
        "finish_reason": choice.get("finish_reason"),
        "content_preview": str(message.get("content"))[:200],
        "elapsed_s": round(elapsed, 3),
        "response_sha256": sha256_bytes(raw),
    }


def read_gpu_used_mb(gpu_index: int) -> int:
    """Real nvidia-smi read of one GPU's used VRAM, for lease phase evidence."""

    query, _ = read_gpu_bytes()
    for line in query.decode("utf-8", errors="replace").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5 and _integer(parts[0]) == gpu_index:
            return _integer(parts[4])
    raise LookupError(f"gpu_index_not_found:{gpu_index}")


def run_live_xml_canary(
    *,
    model_path: str,
    device_uuid: str,
    content_hash: str,
    checkpoint_path: Path,
    gpu_index: int = GPU_INDEX,
    port: int = VLLM_PORT,
) -> JsonDict:
    """Run the real live path: lease, serve, probe four tools, release.

    Every phase transition is real GpuLease state, not a label -- resident and
    validating both carry measured VRAM, matching what every other live GPU
    task in this project already requires of itself.
    """

    from carnot.gpu_lease_phase_journal import GpuLease, LeaseError

    runtime_dir = checkpoint_path.parent / "gpu_lease"
    vram_before = read_gpu_used_mb(gpu_index)
    try:
        lease = GpuLease.acquire(
            runtime_dir=runtime_dir,
            task_id=TASK_ID,
            device_uuid=device_uuid,
            expected_model=content_hash,
            vram_before_mb=vram_before,
            ttl_s=VLLM_LEASE_TTL_S,
        )
    except LeaseError as exc:
        return {
            "lease_acquired": False,
            "lease_error": repr(exc),
            "server_started": False,
            "server_pid": None,
            "health": None,
            "parser_rows": [],
            "vram_resident_mb": None,
            "vram_after_mb": None,
            "exit_code": None,
            "unload_observed": None,
            "phase_reached": None,
        }

    lease.transition("admitted")
    lease.transition("loading")
    proc = launch_vllm_server(model_path, port, gpu_index)
    healthy, health_reason, load_elapsed_s = wait_for_server_health(
        proc,
        port,
        VLLM_STARTUP_TIMEOUT_S,
        VLLM_HEALTH_POLL_INTERVAL_S,
        on_wait=lambda: lease.heartbeat(),
    )
    if not healthy:
        server_output = ""
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        if proc.stdout is not None:
            try:
                server_output = proc.stdout.read()
            except (OSError, ValueError):
                server_output = ""
        vram_after = read_gpu_used_mb(gpu_index)
        lease.transition("terminal_blocked")
        lease.release()
        return {
            "lease_acquired": True,
            "lease_error": None,
            "server_started": True,
            "server_pid": proc.pid,
            "health": {
                "healthy": False,
                "reason": health_reason,
                "elapsed_s": round(load_elapsed_s, 3),
                "server_output_tail": server_output[-16000:],
            },
            "parser_rows": [],
            "vram_resident_mb": None,
            "vram_after_mb": vram_after,
            "exit_code": proc.returncode,
            "unload_observed": vram_after < vram_before + 512,
            "phase_reached": "terminal_blocked",
        }

    vram_resident = read_gpu_used_mb(gpu_index)
    lease.transition("resident", vram_mb=vram_resident)
    lease.transition("inferencing")

    parser_rows: list[JsonDict] = []
    for tool_name in EXPECTED_TOOL_NAMES:
        lease.heartbeat()
        parser_rows.append(
            send_tool_call_probe(
                port, model_path, tool_name, TOOL_PROMPTS[tool_name], VLLM_REQUEST_TIMEOUT_S
            )
        )

    lease.transition("unloading")
    proc.terminate()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
    vram_after = read_gpu_used_mb(gpu_index)
    unload_observed = vram_after < vram_resident - 512
    lease.transition(
        "validating",
        vram_mb=vram_after,
        exit_code=proc.returncode if proc.returncode is not None else -1,
        unload_observed=unload_observed,
    )
    lease.transition("terminal_complete")
    lease.release()

    return {
        "lease_acquired": True,
        "lease_error": None,
        "server_started": True,
        "server_pid": proc.pid,
        "health": {"healthy": True, "reason": health_reason, "elapsed_s": round(load_elapsed_s, 3)},
        "parser_rows": parser_rows,
        "vram_resident_mb": vram_resident,
        "vram_after_mb": vram_after,
        "exit_code": proc.returncode,
        "unload_observed": unload_observed,
        "phase_reached": "terminal_complete",
    }


def _planned_rows() -> list[JsonDict]:
    """Keep every frozen prompt denominator visible before any request exists."""

    return [
        {
            "unit_id": f"xml_canary_{index}",
            "arm": "qwen3_xml",
            "seed": RANDOM_SEED,
            "expected_tool_name": tool_name,
            "metric": None,
            "error": "blocked_vllm_not_installed",
            "abstention": True,
            "attempted": False,
            "completed": False,
            "censored": True,
        }
        for index, tool_name in enumerate(EXPECTED_TOOL_NAMES, start=1)
    ]


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash exact settings, sources, and rows without hashing the field itself."""

    payload = deepcopy(dict(artifact))
    payload.pop("reproducibility_checksum", None)
    return sha256_text(canonical_json(payload))


def base_artifact(run_date: str, host: str, started_at: str) -> JsonDict:
    """Create the complete running shape for the checkpoint path only."""

    artifact: JsonDict = {
        "schema": "carnot.exp7220.v636_xml_canary.v1",
        "experiment_id": "exp7220-xml-canary",
        "field_principles": deepcopy(REQUIRED_FIELD_PRINCIPLES),
        "status": "running",
        "run_date": run_date,
        "started_at_utc": started_at,
        "completed_at_utc": None,
        "preconditions_checked": [],
        "inference_substrate": "blocked_no_run",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": "host",
        "execution_host": host,
        "duration_s": 0.0,
        "source_artifact_hashes": {},
        "rows": _planned_rows(),
        "sample_size_budget": {
            "planned": 4,
            "attempted": 0,
            "completed": 0,
            "censored": 4,
            "independent_units": 4,
            "exclusions": [],
        },
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": {
            "failed_check": "preconditions_not_complete",
            "upstream": None,
            "field": None,
            "expected_value": True,
            "observed_value": False,
            "passed": False,
        },
        "verifier_is_oracle": False,
        "verdict_class": "partial",
        "honest_verdict": "partial_running_xml_canary",
        "MODEL_SPECS": [deepcopy(MANDATED_MODEL_SPEC)],
        "model_count": 1,
        "model_invoked": False,
        "xml_canary_complete_score": 0,
        "xml_transport_ready_score": 0,
        "parser_rows": [],
        "failure_stage": None,
        "quant_path": {},
        "token_budget_receipt": {
            "prompt_count_limit": 4,
            "completion_tokens_per_prompt_limit": 256,
            "planned_completion_token_limit": 1024,
            "actual_prompt_tokens": 0,
            "actual_completion_tokens": 0,
            "stop_reasons": [],
        },
        "model_identity_receipt": {},
        "gpu_receipts": [],
        "phase_spans": [],
        "runner_receipt": {
            "runner": "single_model_vllm_openai_server",
            "planned_model_count": 1,
            "server_pid": None,
            "server_started": False,
            "cleanup": "not_started",
        },
        "raw_evidence": [],
        "package_receipts": [],
        "upstream_artifacts_consumed": [],
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def finish_package_block(
    artifact: Mapping[str, Any],
    *,
    packages: Sequence[Mapping[str, Any]],
    gpu: Mapping[str, Any],
    quant: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    raw_evidence: Sequence[Mapping[str, Any]],
    completed_at: str,
    duration_s: float,
) -> JsonDict:
    """Publish the terminal package block without claiming parser observations."""

    summary = first_package_block(packages)
    if summary is None:
        raise LiveExecutionRequired("required packages are importable; run the live canary path")
    result = deepcopy(dict(artifact))
    failed_check = str(summary["failed_check"])
    verdict = (
        "blocked_vllm_not_installed"
        if failed_check == "vllm_import"
        else "blocked_vllm_gguf_plugin_not_installed"
    )
    rows = _planned_rows()
    for row in rows:
        row["error"] = verdict
    result.update(
        status="blocked",
        completed_at_utc=completed_at,
        preconditions_checked=[
            {
                "check": "CARNOT_FORCE_LIVE",
                "upstream": "process_environment",
                "field": "CARNOT_FORCE_LIVE",
                "expected_value": "1",
                "observed_value": "1",
                "passed": True,
            },
            {
                "check": "gpu_1_idle_and_task_ownable",
                "upstream": "nvidia-smi",
                "field": "task_ownable",
                "expected_value": True,
                "observed_value": gpu.get("task_ownable"),
                "passed": gpu.get("task_ownable") is True,
            },
            {
                "check": "exact_qwen38_q4_cached",
                "upstream": MODEL_ID,
                "field": "cached_path_revision_sha256",
                "expected_value": f"{MODEL_ID}:{QUANTIZATION}",
                "observed_value": {
                    "cached_path": quant.get("cached_path"),
                    "revision": quant.get("revision"),
                    "sha256": quant.get("sha256"),
                },
                "passed": bool(quant.get("cached_path") and quant.get("sha256")),
            },
            *[
                {
                    "check": row.get("check"),
                    "upstream": ".venv",
                    "field": f"{str(row.get('package')).replace('-', '_')}_importable",
                    "expected_value": True,
                    "observed_value": row.get("importable"),
                    "passed": row.get("importable") is True,
                    "version": row.get("version"),
                    "module": row.get("module"),
                    "error_type": row.get("error_type"),
                    "error": row.get("error"),
                }
                for row in packages
            ],
            {
                "check": "tokenizer_support",
                "upstream": "vllm-gguf-plugin",
                "field": "tokenizer_path",
                "expected_value": "installed-version-supported cached matching tokenizer",
                "observed_value": "not_checked_after_package_block",
                "passed": False,
                "blocking_external": False,
            },
        ],
        inference_substrate="blocked_no_run",
        inference_substrate_class="blocked_no_run",
        duration_s=round(max(0.0, float(duration_s)), 6),
        source_artifact_hashes=dict(source_hashes),
        rows=rows,
        gate_check_summary=summary,
        verdict_class="blocked",
        honest_verdict=verdict,
        model_invoked=False,
        xml_canary_complete_score=0,
        xml_transport_ready_score=0,
        parser_rows=[],
        failure_stage="package",
        quant_path={
            "loader": "vllm-gguf-plugin",
            "quantization": QUANTIZATION,
            "cached_path": quant.get("cached_path"),
            "resolved_path": quant.get("resolved_path"),
        },
        model_identity_receipt={
            **dict(quant),
            "loader": "vllm-gguf-plugin",
            "tokenizer_path": None,
            "tokenizer_status": "not_checked_after_package_block",
            "cuda_execution": False,
        },
        gpu_receipts=[dict(gpu)],
        phase_spans=[
            {
                "phase": "preconditions",
                "started_offset_s": 0.0,
                "ended_offset_s": round(max(0.0, float(duration_s)), 6),
                "elapsed_s": round(max(0.0, float(duration_s)), 6),
                "executed": True,
            },
            {
                "phase": "load",
                "started_offset_s": None,
                "ended_offset_s": None,
                "elapsed_s": 0.0,
                "executed": False,
            },
            {
                "phase": "generate",
                "started_offset_s": None,
                "ended_offset_s": None,
                "elapsed_s": 0.0,
                "executed": False,
            },
            {
                "phase": "score",
                "started_offset_s": None,
                "ended_offset_s": None,
                "elapsed_s": 0.0,
                "executed": False,
            },
            {
                "phase": "cleanup",
                "started_offset_s": round(max(0.0, float(duration_s)), 6),
                "ended_offset_s": round(max(0.0, float(duration_s)), 6),
                "elapsed_s": 0.0,
                "executed": True,
                "observed_state": "no_task_owned_server_or_lease_created",
            },
        ],
        runner_receipt={
            "runner": "single_model_vllm_openai_server",
            "planned_model_count": 1,
            "server_pid": None,
            "server_started": False,
            "lease_acquired": False,
            "cleanup": "not_required_no_owned_resources",
        },
        raw_evidence=[dict(row) for row in raw_evidence],
        package_receipts=[dict(row) for row in packages],
    )
    result["reproducibility_checksum"] = artifact_checksum(result)
    return result


def finish_live_block(
    artifact: Mapping[str, Any],
    *,
    packages: Sequence[Mapping[str, Any]],
    gpu: Mapping[str, Any],
    quant: Mapping[str, Any],
    live: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    raw_evidence: Sequence[Mapping[str, Any]],
    completed_at: str,
    duration_s: float,
) -> JsonDict:
    """Publish the terminal artifact for a real, completed live canary attempt.

    `live` is exactly `run_live_xml_canary`'s return -- this function turns that
    real evidence into the artifact schema; it never independently claims a
    tool call happened.
    """

    result = deepcopy(dict(artifact))
    parser_rows = list(live.get("parser_rows") or [])
    populated = [row for row in parser_rows if row.get("populated") is True]
    attempted = [row for row in parser_rows if row.get("ok") is True]
    reached_terminal = live.get("phase_reached") == "terminal_complete"

    if not live.get("lease_acquired"):
        honest_verdict = "blocked_gpu_lease_unavailable"
        substrate = "blocked_no_run"
        substrate_class = "blocked_no_run"
        complete_score = 0
        transport_score = 0
    elif not reached_terminal:
        honest_verdict = "blocked_vllm_server_startup_failed"
        substrate = "blocked_no_run"
        substrate_class = "blocked_no_run"
        complete_score = 0
        transport_score = 0
    elif len(attempted) == len(EXPECTED_TOOL_NAMES) and len(populated) == len(EXPECTED_TOOL_NAMES):
        honest_verdict = "complete_positive_xml_transport_confirmed"
        substrate = "live_llm_inference"
        substrate_class = "model_full_generation"
        complete_score = 1
        transport_score = 1
    elif attempted:
        honest_verdict = "complete_partial_xml_transport_some_calls_did_not_populate"
        substrate = "live_llm_inference"
        substrate_class = "model_full_generation"
        complete_score = 1
        transport_score = 0
    else:
        honest_verdict = "complete_negative_xml_transport_not_confirmed"
        substrate = "live_llm_inference"
        substrate_class = "model_full_generation"
        complete_score = 1
        transport_score = 0

    rows = [
        {
            "unit_id": f"xml_canary_{index}",
            "arm": "qwen3_xml",
            "seed": RANDOM_SEED,
            "expected_tool_name": tool_name,
            "metric": 1.0 if row and row.get("populated") else 0.0,
            "error": row.get("error") if row else "not_attempted",
            "abstention": row is None,
            "attempted": bool(row and row.get("ok")),
            "completed": bool(row and row.get("populated")),
            "censored": row is None,
        }
        for index, tool_name in enumerate(EXPECTED_TOOL_NAMES, start=1)
        for row in [next((r for r in parser_rows if r.get("tool_name") == tool_name), None)]
    ]

    result.update(
        status="complete"
        if reached_terminal or honest_verdict.startswith("blocked_")
        else "blocked",
        completed_at_utc=completed_at,
        preconditions_checked=[
            {
                "check": "gpu_1_idle_and_task_ownable",
                "upstream": "nvidia-smi",
                "field": "task_ownable",
                "expected_value": True,
                "observed_value": gpu.get("task_ownable"),
                "passed": gpu.get("task_ownable") is True,
            },
            *[
                {
                    "check": row.get("check"),
                    "upstream": ".venv-vllm-trial",
                    "field": f"{str(row.get('package')).replace('-', '_')}_importable",
                    "expected_value": True,
                    "observed_value": row.get("importable"),
                    "passed": row.get("importable") is True,
                }
                for row in packages
            ],
            {
                "check": "gpu_lease_acquired",
                "upstream": str(
                    (Path("results/checkpoints") / "experiment_7220_v636_xml_canary" / "gpu_lease")
                ),
                "field": "lease_acquired",
                "expected_value": True,
                "observed_value": live.get("lease_acquired"),
                "passed": live.get("lease_acquired") is True,
            },
            {
                "check": "vllm_server_healthy",
                "upstream": f"http://127.0.0.1:{VLLM_PORT}/health",
                "field": "healthy",
                "expected_value": True,
                "observed_value": (live.get("health") or {}).get("healthy"),
                "passed": bool((live.get("health") or {}).get("healthy")),
            },
        ],
        inference_substrate=substrate,
        inference_substrate_class=substrate_class,
        duration_s=round(max(0.0, float(duration_s)), 6),
        source_artifact_hashes=dict(source_hashes),
        rows=rows,
        sample_size_budget={
            "planned": len(EXPECTED_TOOL_NAMES),
            "attempted": len(attempted),
            "completed": len(populated),
            "censored": len(EXPECTED_TOOL_NAMES) - len(attempted),
            "independent_units": len(EXPECTED_TOOL_NAMES),
            "exclusions": [],
        },
        gate_check_summary={
            "failed_check": None,
            "upstream": None,
            "field": None,
            "expected_value": "all_tool_calls_populate",
            "observed_value": f"{len(populated)}_of_{len(EXPECTED_TOOL_NAMES)}_populated",
            "passed": len(populated) == len(EXPECTED_TOOL_NAMES),
        },
        verdict_class="positive" if transport_score == 1 else "null",
        honest_verdict=honest_verdict,
        model_invoked=bool(live.get("server_started")),
        xml_canary_complete_score=complete_score,
        xml_transport_ready_score=transport_score,
        parser_rows=parser_rows,
        failure_stage=(
            "none"
            if transport_score == 1
            else (
                "lease"
                if not live.get("lease_acquired")
                else ("server" if not reached_terminal else "parsing")
            )
        ),
        quant_path={
            "loader": "vllm-gguf-plugin",
            "quantization": QUANTIZATION,
            "cached_path": quant.get("cached_path"),
            "resolved_path": quant.get("resolved_path"),
        },
        model_identity_receipt={
            **dict(quant),
            "loader": "vllm-gguf-plugin",
            "tokenizer_path": None,
            "tokenizer_status": "embedded_gguf_tokenizer",
            "cuda_execution": bool(live.get("server_started")),
        },
        gpu_receipts=[dict(gpu)],
        phase_spans=[
            {
                "phase": "preconditions",
                "started_offset_s": 0.0,
                "ended_offset_s": round(max(0.0, float(duration_s)), 6),
                "elapsed_s": round(max(0.0, float(duration_s)), 6),
                "executed": True,
            },
            {
                "phase": "load",
                "started_offset_s": None,
                "ended_offset_s": None,
                "elapsed_s": (live.get("health") or {}).get("elapsed_s", 0.0),
                "executed": bool(live.get("server_started")),
            },
            {
                "phase": "generate",
                "started_offset_s": None,
                "ended_offset_s": None,
                "elapsed_s": round(sum(r.get("elapsed_s") or 0.0 for r in parser_rows), 3),
                "executed": bool(parser_rows),
            },
            {
                "phase": "score",
                "started_offset_s": None,
                "ended_offset_s": None,
                "elapsed_s": 0.0,
                "executed": bool(parser_rows),
            },
            {
                "phase": "cleanup",
                "started_offset_s": None,
                "ended_offset_s": None,
                "elapsed_s": 0.0,
                "executed": True,
                "observed_state": live.get("phase_reached"),
            },
        ],
        runner_receipt={
            "runner": "single_model_vllm_openai_server",
            "planned_model_count": 1,
            "server_pid": live.get("server_pid"),
            "server_started": bool(live.get("server_started")),
            "lease_acquired": bool(live.get("lease_acquired")),
            "exit_code": live.get("exit_code"),
            "unload_observed": live.get("unload_observed"),
            "vram_resident_mb": live.get("vram_resident_mb"),
            "vram_after_mb": live.get("vram_after_mb"),
            "cleanup": "terminated_and_released" if reached_terminal else "aborted",
            "server_output_tail": (live.get("health") or {}).get("server_output_tail"),
        },
        raw_evidence=[dict(row) for row in raw_evidence],
        package_receipts=[dict(row) for row in packages],
    )
    result["reproducibility_checksum"] = artifact_checksum(result)
    return result


def validate_artifact(value: object) -> list[str]:
    """Recompute the terminal block, denominator, identity, and checksum rules."""

    if isinstance(value, (str, Path)):
        value = json.loads(Path(value).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        return ["artifact_mapping_required"]
    required = set(REQUIRED_FIELD_PRINCIPLES) | {
        "schema",
        "experiment_id",
        "started_at_utc",
        "completed_at_utc",
        "model_count",
        "raw_evidence",
        "package_receipts",
        "upstream_artifacts_consumed",
    }
    missing = sorted(required - set(value))
    if missing:
        return [f"missing_required_field:{missing[0]}"]
    errors: list[str] = []
    if value.get("run_date") != RUN_DATE:
        errors.append("run_date")
    if value.get("field_principles") != REQUIRED_FIELD_PRINCIPLES:
        errors.append("field_principles")
    if value.get("MODEL_SPECS") != [MANDATED_MODEL_SPEC] or value.get("model_count") != 1:
        errors.append("model_spec_mismatch")
    blocked = value.get("honest_verdict") in {
        "blocked_vllm_not_installed",
        "blocked_vllm_gguf_plugin_not_installed",
    }
    if value.get("status") == "blocked" and blocked:
        if value.get("model_invoked") is not False:
            errors.append("blocked_model_invoked")
        budget = value.get("sample_size_budget")
        expected_budget = {
            "planned": 4,
            "attempted": 0,
            "completed": 0,
            "censored": 4,
            "independent_units": 4,
            "exclusions": [],
        }
        if budget != expected_budget or len(value.get("rows", [])) != 4:
            errors.append("blocked_denominators")
        if value.get("parser_rows") != []:
            errors.append("blocked_parser_rows")
        if (
            value.get("xml_canary_complete_score") != 0
            or value.get("xml_transport_ready_score") != 0
        ):
            errors.append("blocked_scores")
        if (
            value.get("inference_substrate") != "blocked_no_run"
            or value.get("inference_substrate_class") != "blocked_no_run"
        ):
            errors.append("blocked_substrate")
        summary = value.get("gate_check_summary")
        if not isinstance(summary, Mapping) or any(
            key not in summary
            for key in (
                "failed_check",
                "upstream",
                "field",
                "expected_value",
                "observed_value",
            )
        ):
            errors.append("gate_check_summary")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum")
    return errors


def _source_hashes(root: Path, quant: Mapping[str, Any], raw: Sequence[Path]) -> JsonDict:
    """Bind every cited input plus exact preflight bytes and quant content."""

    hashes: JsonDict = {
        str(path): sha256_file(root / path) if (root / path).is_file() else None
        for path in SOURCE_PATHS
    }
    hashes.update({str(path): sha256_file(path) for path in raw})
    hashes[str(quant["cached_path"])] = str(quant["sha256"])
    return hashes


def _write_raw(path: Path, payload: bytes) -> JsonDict:
    """Atomically preserve exact preflight bytes and return their identity."""

    write_bytes_atomic(path, payload)
    return {"path": str(path), "size_bytes": len(payload), "sha256": sha256_bytes(payload)}


def run_experiment(
    *,
    root: Path = REPO_ROOT,
    output_path: Path = RESULT_PATH,
    raw_dir: Path = RAW_DIR,
    checkpoint_path: Path = CHECKPOINT_PATH,
    run_date: str = RUN_DATE,
    importer: Importer = importlib.import_module,
    version_reader: VersionReader = importlib.metadata.version,
    gpu_reader: Callable[[], tuple[bytes, bytes]] = read_gpu_bytes,
    quant_resolver: Callable[[], str | None] = resolve_quant_path,
    live_canary_runner: Callable[..., JsonDict] = run_live_xml_canary,
    clock: Callable[[], float] = time.monotonic,
    utc_reader: Callable[[], str] = utc_now,
) -> JsonDict:
    """Run preflight and stop at the required missing-package terminal gate."""

    if run_date != RUN_DATE:
        raise ValueError(f"run_date must be {RUN_DATE}")
    started = clock()
    started_at = utc_reader()
    os.environ["CARNOT_FORCE_LIVE"] = "1"
    output_path = output_path if output_path.is_absolute() else root / output_path
    raw_dir = raw_dir if raw_dir.is_absolute() else root / raw_dir
    checkpoint_path = checkpoint_path if checkpoint_path.is_absolute() else root / checkpoint_path
    for directory in (output_path.parent, raw_dir, checkpoint_path.parent):
        directory.mkdir(parents=True, exist_ok=True)

    progress(0, "start", "preconditions before any model, tokenizer, server, or lease")
    missing_sources = [str(path) for path in SOURCE_PATHS if not (root / path).is_file()]
    if missing_sources:
        raise FileNotFoundError(f"required source paths missing: {missing_sources}")

    progress(0, "before", "GPU 1 occupancy query")
    gpu_query, gpu_apps = gpu_reader()
    gpu = parse_gpu_receipt(gpu_query, gpu_apps)
    progress(0, "after", f"GPU 1 task_ownable={gpu['task_ownable']}")

    progress(0, "before", "exact cached Qwen3.8 Q4_K_M identity and hash")
    quant_path = quant_resolver()
    if quant_path is None:
        raise FileNotFoundError(f"cached {MODEL_ID}:{QUANTIZATION} is missing")
    quant = quant_identity(quant_path)
    progress(0, "after", f"quant revision={quant['revision']} sha256={quant['sha256']}")

    progress(0, "before", "vLLM and vLLM GGUF plugin imports")
    packages = package_preflight(importer, version_reader)
    progress(0, "after", f"package imports={[row['importable'] for row in packages]}")

    raw_rows = [
        _write_raw(raw_dir / "gpu_query.csv", gpu_query),
        _write_raw(raw_dir / "gpu_compute_apps.csv", gpu_apps),
    ]
    package_bytes = (json.dumps(packages, indent=2, sort_keys=True) + "\n").encode()
    quant_bytes = (json.dumps(quant, indent=2, sort_keys=True) + "\n").encode()
    raw_rows.extend(
        [
            _write_raw(raw_dir / "package_imports.json", package_bytes),
            _write_raw(raw_dir / "quant_identity.json", quant_bytes),
        ]
    )

    running = base_artifact(run_date, platform.node() or "unknown", started_at)
    atomic_write_json(checkpoint_path, running, allow_override=False, sort_keys=True)

    if first_package_block(packages) is not None:
        progress(7, "start", "no task-owned server or lease exists; cleanup is not required")
        completed_at = utc_reader()
        duration_s = max(0.0, clock() - started)
        source_hashes = _source_hashes(root, quant, [Path(row["path"]) for row in raw_rows])
        result = finish_package_block(
            running,
            packages=packages,
            gpu=gpu,
            quant=quant,
            source_hashes=source_hashes,
            raw_evidence=raw_rows,
            completed_at=completed_at,
            duration_s=duration_s,
        )
        errors = validate_artifact(result)
        if errors:
            raise ValueError(f"terminal artifact invalid: {errors}")
        progress(7, "complete", "cleanup observed no task-owned process or lease")
        progress(9, "before", f"atomic terminal write {output_path}")
        atomic_write_json(output_path, result, allow_override=False, sort_keys=True)
        progress(9, "after", f"terminal verdict={result['honest_verdict']}")
        return result

    progress(1, "before", "acquire GPU lease, launch vLLM, probe four tools, release")
    device_uuid = str(gpu.get("selected_gpu_uuid") or "")
    live = live_canary_runner(
        model_path=quant_path,
        device_uuid=device_uuid,
        content_hash=str(quant.get("sha256")),
        checkpoint_path=checkpoint_path,
    )
    progress(
        1,
        "after",
        f"phase_reached={live.get('phase_reached')} "
        f"populated={sum(1 for r in (live.get('parser_rows') or []) if r.get('populated'))}"
        f"/{len(EXPECTED_TOOL_NAMES)}",
    )

    completed_at = utc_reader()
    duration_s = max(0.0, clock() - started)
    source_hashes = _source_hashes(root, quant, [Path(row["path"]) for row in raw_rows])
    result = finish_live_block(
        running,
        packages=packages,
        gpu=gpu,
        quant=quant,
        live=live,
        source_hashes=source_hashes,
        raw_evidence=raw_rows,
        completed_at=completed_at,
        duration_s=duration_s,
    )
    errors = validate_artifact(result)
    if errors:
        raise ValueError(f"terminal artifact invalid: {errors}")
    progress(9, "before", f"atomic terminal write {output_path}")
    atomic_write_json(output_path, result, allow_override=False, sort_keys=True)
    progress(9, "after", f"terminal verdict={result['honest_verdict']}")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the frozen date and write the terminal result artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    args = parser.parse_args(argv)
    result = run_experiment(run_date=args.date)
    print(json.dumps(result["gate_check_summary"], sort_keys=True), flush=True)
    return 0

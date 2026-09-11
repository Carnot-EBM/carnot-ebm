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
import time
from typing import Any

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
    tokenizer_probe: Callable[[], object] | None = None,
    lease_factory: Callable[[], object] | None = None,
    server_factory: Callable[[], object] | None = None,
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
    if first_package_block(packages) is None:
        dependencies = (
            ("tokenizer", tokenizer_probe),
            ("lease", lease_factory),
            ("server", server_factory),
        )
        names = [name for name, value in dependencies if value is not None]
        raise LiveExecutionRequired(
            "package preflight passed; the live execution path is required before calling "
            + ", ".join(names or ["runtime dependencies"])
        )

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


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the frozen date and write the terminal result artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    args = parser.parse_args(argv)
    result = run_experiment(run_date=args.date)
    print(json.dumps(result["gate_check_summary"], sort_keys=True), flush=True)
    return 0

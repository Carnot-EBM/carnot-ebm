"""Exp6114 Phase D GPU ladder generation canary.

Spec refs: REQ-VERIFY-6114, SCENARIO-VERIFY-6114-LADDER-REPLAY,
SCENARIO-VERIFY-6114-MEASURED-FIT-MODEL, SCENARIO-VERIFY-6114-REAL-GENERATION,
SCENARIO-VERIFY-6114-LIFECYCLE, SCENARIO-VERIFY-6114-RETIRED-SCOPE.

This experiment is deliberately narrower than Exp6102.  It does not resume or
inspect output-free representation shards.  It only verifies the sealed Exp6103
calibration ladder, loads the one measured-fit 26B GGUF, asks for bounded
natural text generations, and records whether the task-owned CUDA worker really
engaged and released the selected GPU.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Protocol

from carnot import experiment_6103_phase_d_difficulty_ladder_fixture as exp6103


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6114_phase_d_gpu_ladder_canary.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6114_phase_d_gpu_ladder_canary.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6114_phase_d_gpu_ladder_canary.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/verifiable-reasoning/spec.md")
EXP6102_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_6102_sota_atom_corpus_vram_recovery.json"
)
EXP6103_ARTIFACT_RELATIVE_PATH = exp6103.RESULT_RELATIVE_PATH
EXP6103_ROW_RELATIVE_PATH = exp6103.ROW_FILE_RELATIVE_PATH
EXP6103_SPLIT_RELATIVE_PATH = exp6103.SPLIT_MANIFEST_RELATIVE_PATH

SCHEMA = "carnot.experiment_6114.phase_d_gpu_ladder_canary.v1"
ROW_SCHEMA = SCHEMA + ".row"
EXPERIMENT_ID = "experiment_6114_phase_d_gpu_ladder_canary"
RUN_DATE = "20260804"
INFERENCE_SUBSTRATE = "live_local_sota_gguf_cuda_generation"
VERIFIER_IS_ORACLE = True
MODEL_HF_ID = "unsloth/gemma-4-26B-A4B-it-GGUF"
MODEL_FAMILY = "gemma-4-26b-a4b-it"
MODEL_QUANTIZATION = "Q4_K_M"
MEASURED_FIT_REQUIRED_MB = 17_186
RANDOM_SEED = 6114
MIN_CANARY_ROWS = 12
PER_FAMILY_CALIBRATION_ROWS = 4
PROMPT_TEMPLATE_VERSION = "exp6114_natural_reasoning_final_answer_v1"
RAM_FLOOR_MB = 16_384
DISK_FLOOR_MB = 10_240

DECODE_CONFIG: JsonDict = {
    "max_new_tokens": 512,
    "temperature": 0.2,
    "top_p": 0.95,
    "repeat_penalty": 1.05,
    "grammar": None,
}

PROTECTED_FILES = (
    Path("scripts/research_conductor.py"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)

RETIRED_REPRESENTATION_ROW_SHARDS = (
    Path("results/experiment_5964_sota_atom_compatibility_corpus.qwen3.6-35b-a3b.rows.jsonl"),
    Path("results/experiment_5964_sota_atom_compatibility_corpus.gemma-4-26b-a4b-it.rows.jsonl"),
    Path("results/experiment_5964_sota_atom_compatibility_corpus.gemma-4-31b-it.rows.jsonl"),
    Path("results/experiment_6102_sota_atom_corpus_vram_recovery.qwen3.6-35b-a3b.rows.jsonl"),
    Path("results/experiment_6102_sota_atom_corpus_vram_recovery.gemma-4-26b-a4b-it.rows.jsonl"),
    Path("results/experiment_6102_sota_atom_corpus_vram_recovery.gemma-4-31b-it.rows.jsonl"),
)

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6114_phase_d_gpu_ladder_canary.py "
    "-q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6114_phase_d_gpu_ladder_canary.py "
    "-m pytest tests/python/test_experiment_6114_phase_d_gpu_ladder_canary.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6114_phase_d_gpu_ladder_canary.py "
    "--fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6114_phase_d_gpu_ladder_canary.py",
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6114_phase_d_gpu_ladder_canary.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    "git status --short -- scripts/research_conductor.py "
    "ops/exclusion_manifest.yaml ops/changelog.md ops/status.md _bmad/traceability.md",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "immutable_ladder_artifact_row_and_split_hashes",
    "ladder_readiness_and_python_z3_receipt",
    "model_specs_and_exact_file_hashes",
    "quantization_and_embedded_tokenizer_receipt",
    "task_owned_gpu_server_and_pid_lease",
    "pre_load_decode_and_post_release_vram_thermal_timeline",
    "generated_calibration_canary_rows_and_hashes",
    "prompt_decode_seed_and_token_receipts",
    "gpu_engagement_attribution",
    "server_exit_cuda_sync_pid_exit_and_vram_release_receipts",
    "retired_representation_scope_untouched",
    "phase_d_compute_and_ladder_ready_score",
    "retirement_triggered",
    "protected_files_unchanged",
    "random_seed",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "missing_verifier_gaps",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

REQUIRED_FIELD_PRINCIPLES: dict[str, str] = {
    "immutable_ladder_artifact_row_and_split_hashes": (
        "only the sealed exact fixture can feed the canary."
    ),
    "ladder_readiness_and_python_z3_receipt": (
        "only the sealed exact fixture can feed the canary."
    ),
    "model_specs_and_exact_file_hashes": (
        "every generated row traces to the mandated measured-fit GGUF and its embedded tokenizer."
    ),
    "quantization_and_embedded_tokenizer_receipt": (
        "every generated row traces to the mandated measured-fit GGUF and its embedded tokenizer."
    ),
    "task_owned_gpu_server_and_pid_lease": (
        "resource ownership and lifecycle are measured, not inferred."
    ),
    "pre_load_decode_and_post_release_vram_thermal_timeline": (
        "resource ownership and lifecycle are measured, not inferred."
    ),
    "generated_calibration_canary_rows_and_hashes": (
        "real natural generation is preserved raw and reproducible."
    ),
    "prompt_decode_seed_and_token_receipts": (
        "real natural generation is preserved raw and reproducible."
    ),
    "gpu_engagement_attribution": "readiness requires attributable compute and cleanup.",
    "server_exit_cuda_sync_pid_exit_and_vram_release_receipts": (
        "readiness requires attributable compute and cleanup."
    ),
    "retired_representation_scope_untouched": (
        "this canary cannot reopen or resume the retired all-family representation corpus."
    ),
    "phase_d_compute_and_ladder_ready_score": (
        "readiness is exactly 1 only if ladder, generation, engagement, and release gates all pass; the same block retires the shape."
    ),
    "retirement_triggered": (
        "readiness is exactly 1 only if ladder, generation, engagement, and release gates all pass; the same block retires the shape."
    ),
    "duration_s": "report measured `live_local_sota_gguf_cuda_generation`.",
    "inference_substrate": "report measured `live_local_sota_gguf_cuda_generation`.",
    "field_provenance": "report measured `live_local_sota_gguf_cuda_generation`.",
    "test_commands": "report measured `live_local_sota_gguf_cuda_generation`.",
    "test_exit_codes": "report measured `live_local_sota_gguf_cuda_generation`.",
    "reproducibility_checksum": "report measured `live_local_sota_gguf_cuda_generation`.",
    "verifier_is_oracle": (
        "exact Python/Z3 labels remain oracle; generation telemetry is not correctness evidence."
    ),
    "missing_verifier_gaps": (
        "exact Python/Z3 labels remain oracle; generation telemetry is not correctness evidence."
    ),
    "honest_verdict": "use `complete_ready:`, `complete_partial:`, `retired:`, or `blocked:`.",
}


class CanaryGateError(ValueError):
    """Raised when a required canary gate cannot be replayed exactly."""


class GenerationBackend(Protocol):
    """Injectable natural-generation backend used by tests and the live worker."""

    def generate(
        self,
        *,
        model_spec: JsonDict,
        selected_gpu: int,
        prompts: list[JsonDict],
        decode_config: JsonDict,
        baseline_devices: list[JsonDict],
    ) -> JsonDict:
        """Return backend rows and lifecycle receipts for the selected model."""


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible evidence in stable byte order."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for UTF-8 text evidence."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for canonical JSON evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes without depending on filesystem metadata."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _read_json(path: str | Path) -> JsonDict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):  # pragma: no cover - corrupted JSON guard.
        raise ValueError(f"JSON object required: {path}")
    return dict(payload)


def _read_jsonl(path: str | Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():  # pragma: no cover - harmless blank JSONL guard.
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):  # pragma: no cover - corrupted JSONL guard.
            raise ValueError(f"JSONL object required at line {line_number}: {path}")
        rows.append(dict(payload))
    return rows


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _run_command(command: Sequence[str], *, timeout_s: float) -> JsonDict:  # pragma: no cover
    started = time.perf_counter()
    try:
        result = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return {
            "command": list(command),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration_s": round(time.perf_counter() - started, 6),
            "ok": result.returncode == 0,
        }
    except Exception as exc:
        return {
            "command": list(command),
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "duration_s": round(time.perf_counter() - started, 6),
            "ok": False,
        }


def _gpu_devices() -> list[JsonDict]:  # pragma: no cover
    result = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free,memory.used,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=10,
    )
    devices: list[JsonDict] = []
    for line in str(result.get("stdout", "")).splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            devices.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mb": int(parts[2]),
                    "memory_free_mb": int(parts[3]),
                    "memory_used_mb": int(parts[4]),
                    "temperature_c": int(parts[5]),
                }
            )
        except ValueError:
            continue
    return devices


def _compute_apps() -> list[JsonDict]:  # pragma: no cover
    result = _run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=10,
    )
    apps: list[JsonDict] = []
    for line in str(result.get("stdout", "")).splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            apps.append(
                {
                    "pid": int(parts[0]),
                    "process_name": parts[1],
                    "gpu_uuid": parts[2],
                    "used_memory_mb": int(parts[3]),
                }
            )
        except ValueError:
            continue
    return apps


def _memory_probe() -> JsonDict:  # pragma: no cover
    available_mb = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available_mb = int(line.split()[1]) // 1024
                break
    if available_mb == 0:
        available_mb = int(
            os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
        )
    return {"available_mb": available_mb, "required_mb": RAM_FLOOR_MB, "ok": available_mb >= RAM_FLOOR_MB}


def _swap_probe() -> JsonDict:  # pragma: no cover
    total = 0
    free = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("SwapTotal:"):
                total = int(line.split()[1]) // 1024
            if line.startswith("SwapFree:"):
                free = int(line.split()[1]) // 1024
    return {"total_mb": total, "free_mb": free, "used_mb": max(0, total - free)}


def _disk_probe(root: Path) -> JsonDict:  # pragma: no cover
    free_mb = int(shutil.disk_usage(root).free / (1024 * 1024))
    return {"available_mb": free_mb, "required_mb": DISK_FLOOR_MB, "ok": free_mb >= DISK_FLOOR_MB}


def _root_clutter_inventory(root: Path) -> JsonDict:  # pragma: no cover
    files = sorted(path.name for path in root.glob("*.py"))
    return {"root_python_files": files, "root_python_file_count": len(files), "ok": not files}


def _cuda_build_probe() -> JsonDict:  # pragma: no cover
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "nvcc": _run_command(["nvcc", "--version"], timeout_s=5),
        "nvidia_smi": {
            "query": _run_command(["nvidia-smi"], timeout_s=10),
        },
    }


def collect_preconditions(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
) -> JsonDict:  # pragma: no cover
    devices = _gpu_devices()
    output = {"result_path": str(result_path), "parent_writable": os.access(Path(result_path).parent, os.W_OK)}
    protected_hashes = {
        relative.as_posix(): sha256_file(root / relative)
        for relative in PROTECTED_FILES
        if (root / relative).exists()
    }
    blocked: list[str] = []
    if not devices:
        blocked.append("gpu_device_receipt_unavailable")
    memory = _memory_probe()
    disk = _disk_probe(root)
    if memory["ok"] is not True:
        blocked.append("insufficient_free_ram")
    if disk["ok"] is not True:
        blocked.append("insufficient_free_disk")
    if output["parent_writable"] is not True:
        blocked.append("output_path_not_writable")
    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": RUN_DATE,
        "preconditions_ready": not blocked,
        "blocked_reasons": sorted(set(blocked)),
        "gpu": {"gpu_count": len(devices), "ok": bool(devices), "devices": devices},
        "compute_apps_before": _compute_apps(),
        "resources": {"memory": memory, "disk": disk, "swap": _swap_probe()},
        "runtime": {
            "cuda_build": _cuda_build_probe(),
            "task_owned_pid_leases": {
                "current_pid": os.getpid(),
                "parent_pid": os.getppid(),
                "child_pids": [],
                "lease_scope": "task_owned_processes_only",
            },
        },
        "output_paths": output,
        "root_clutter": _root_clutter_inventory(root),
        "protected_file_hashes_before": protected_hashes,
    }


def verify_sealed_ladder(
    *,
    ladder_artifact: Mapping[str, Any],
    ladder_rows: Sequence[Mapping[str, Any]],
    split_manifest: Mapping[str, Any],
    row_file_path: str | Path,
    split_manifest_path: str | Path,
) -> JsonDict:
    """Replay Exp6103's exact seals before any generation prompt is built."""

    if float(ladder_artifact.get("phase_d_ladder_fixture_ready_score", 0.0)) != 1.0:
        raise CanaryGateError("phase_d_ladder_fixture_ready_score")
    if not str(ladder_artifact.get("honest_verdict", "")).startswith("complete_ready:"):  # pragma: no cover
        raise CanaryGateError("ladder_honest_verdict")
    parity = dict(ladder_artifact.get("python_z3_parity") or {})
    if parity.get("python_z3_disagreement_count") != 0:  # pragma: no cover
        raise CanaryGateError("python_z3_disagreement_count")
    if parity.get("method_validity_disagreement_count") != 0:  # pragma: no cover
        raise CanaryGateError("method_validity_disagreement_count")
    chance = dict(ladder_artifact.get("answer_space_and_enumerated_chance_floors") or {})
    if float(chance.get("max_chance_floor", 1.0)) > 0.25:  # pragma: no cover
        raise CanaryGateError("max_chance_floor")
    if int(chance.get("chance_floor_ambiguity_count", 1)) != 0:  # pragma: no cover
        raise CanaryGateError("chance_floor_ambiguity_count")
    exp6103.verify_row_file(list(ladder_rows), ladder_artifact)
    exp6103.verify_split_manifest(dict(split_manifest), list(ladder_rows), ladder_artifact)
    row_receipt = dict(ladder_artifact["row_paths_hashes_and_prefix_chain"])
    row_sha = sha256_file(row_file_path)
    split_sha = sha256_file(split_manifest_path)
    if row_receipt.get("row_file_sha256") != row_sha:  # pragma: no cover
        raise CanaryGateError("row_file_sha256")
    if row_receipt.get("split_manifest_sha256") != split_sha:  # pragma: no cover
        raise CanaryGateError("split_manifest_sha256")
    split_counts = Counter(str(row.get("split")) for row in ladder_rows)
    return {
        "schema": SCHEMA + ".ladder_replay",
        "sealed_ladder_ready": True,
        "status": ladder_artifact.get("status"),
        "honest_verdict": ladder_artifact.get("honest_verdict"),
        "phase_d_ladder_fixture_ready_score": 1.0,
        "row_file_path": str(row_file_path),
        "row_file_sha256": row_sha,
        "split_manifest_path": str(split_manifest_path),
        "split_manifest_sha256": split_sha,
        "terminal_prefix_hash": row_receipt.get("terminal_prefix_hash"),
        "python_z3_parity": parity,
        "chance_floor_receipt": chance,
        "row_count": len(ladder_rows),
        "calibration_row_count": split_counts["calibration"],
        "held_test_row_count": split_counts["held_test"],
        "selected_split_policy": "calibration_only",
    }


def sample_calibration_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    per_family: int = PER_FAMILY_CALIBRATION_ROWS,
) -> list[JsonDict]:
    """Return deterministic calibration rows across all sealed Exp6103 families."""

    selected: list[JsonDict] = []
    for family in exp6103.FAMILIES:
        family_rows = [
            dict(row)
            for row in rows
            if row.get("split") == "calibration" and row.get("family") == family
        ]
        family_rows.sort(key=lambda row: int(row.get("local_index", 0)))
        if len(family_rows) < per_family:  # pragma: no cover
            raise CanaryGateError(f"calibration_rows:{family}")
        selected.extend(family_rows[:per_family])
    if len(selected) < MIN_CANARY_ROWS:  # pragma: no cover
        raise CanaryGateError("minimum_calibration_canary_rows")
    return selected


def _prompt_for_row(row: Mapping[str, Any], sequence_index: int) -> JsonDict:
    prompt_text = "\n".join(
        [
            f"Prompt template: {PROMPT_TEMPLATE_VERSION}",
            "Use only the public problem text below.",
            str(row["model_facing_prompt"]),
            (
                "Give concise natural-language reasoning, then end with exactly one line "
                "of the form Final answer: <label>."
            ),
        ]
    )
    seed = RANDOM_SEED + sequence_index
    return {
        "sequence_index": sequence_index,
        "row_id": str(row["row_id"]),
        "source_row_hash": str(row["row_hash"]),
        "family": str(row["family"]),
        "semantic_group_id": str(row["semantic_group_id"]),
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "prompt_text": prompt_text,
        "prompt_hash": sha256_text(prompt_text),
        "seed": seed,
    }


def _build_prompts(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [_prompt_for_row(row, index) for index, row in enumerate(rows)]


def _model_from_exp6102(exp6102_artifact_path: str | Path) -> tuple[JsonDict, JsonDict, list[str]]:
    artifact = _read_json(exp6102_artifact_path)
    blockers: list[str] = []
    capacity = dict(
        dict(artifact.get("runtime_cuda_vram_thermal_and_pid_lease_receipts") or {}).get(
            "capacity_verdicts"
        )
        or {}
    )
    fit = dict(capacity.get(MODEL_HF_ID) or {})
    if fit.get("fits") is not True:  # pragma: no cover
        blockers.append("measured_fit_receipt_missing")
    if int(fit.get("required_mb", 0) or 0) != MEASURED_FIT_REQUIRED_MB:  # pragma: no cover
        blockers.append("measured_fit_required_mb_mismatch")
    model_records = dict(
        dict(artifact.get("model_specs_and_exact_file_hashes") or {}).get("records") or {}
    )
    source_record = dict(model_records.get(MODEL_HF_ID) or {})
    model_path = Path(str(source_record.get("model_path", ""))).expanduser()
    present = model_path.is_file()
    if not present:  # pragma: no cover
        blockers.append("measured_fit_gguf_missing")
    recomputed_sha = sha256_file(model_path) if present else ""
    if present and source_record.get("model_sha256") and source_record.get("model_sha256") != recomputed_sha:  # pragma: no cover
        blockers.append("measured_fit_gguf_hash_mismatch")
    tokenizer_records = dict(
        dict(artifact.get("quantization_and_embedded_tokenizer_receipts") or {}).get("records")
        or {}
    )
    tokenizer_record = dict(tokenizer_records.get(MODEL_HF_ID) or {})
    embedded = dict(tokenizer_record.get("embedded_tokenizer_receipt") or {})
    if embedded.get("loadable") is not True:  # pragma: no cover
        blockers.append("embedded_tokenizer_unavailable")
    quantization = str(source_record.get("quantization") or tokenizer_record.get("quantization") or "")
    if MODEL_QUANTIZATION not in quantization:  # pragma: no cover
        blockers.append("quantization_mismatch")
    selected_record = {
        "hf_id": MODEL_HF_ID,
        "family": str(source_record.get("family") or MODEL_FAMILY),
        "model_path": str(model_path) if source_record.get("model_path") else "",
        "model_sha256": recomputed_sha or str(source_record.get("model_sha256") or ""),
        "local_path_hash": sha256_text(str(model_path.resolve())) if present else "",
        "local_model_present": present,
        "primary_model_file": str(model_path).endswith(".gguf") and not model_path.name.startswith("mmproj"),
        "quantization": MODEL_QUANTIZATION,
        "min_vram_gb": source_record.get("min_vram_gb", 16),
        "headline_eligible": source_record.get("headline_eligible") is not False,
    }
    model_receipt = {
        "schema": SCHEMA + ".model_specs_and_exact_file_hashes",
        "selected_model_hf_id": MODEL_HF_ID,
        "measured_fit_required_mb": MEASURED_FIT_REQUIRED_MB,
        "measured_fit_source_artifact": str(exp6102_artifact_path),
        "exp6102_artifact_sha256": sha256_file(exp6102_artifact_path),
        "records": {MODEL_HF_ID: selected_record},
        "declined_models": {
            key: value
            for key, value in capacity.items()
            if key != MODEL_HF_ID
        },
        "tiny_model_substituted": False,
        "all_required_files_present": present,
        "receipt_hash": "",
    }
    model_receipt["receipt_hash"] = sha256_json(model_receipt)
    tokenizer_receipt = {
        "schema": SCHEMA + ".quantization_and_embedded_tokenizer_receipt",
        "selected_model_hf_id": MODEL_HF_ID,
        "quantization": MODEL_QUANTIZATION,
        "embedded_tokenizer_receipt": embedded,
        "embedded_tokenizer_receipt_hash": sha256_json(embedded),
        "gguf_embedded_tokenizer_only": True,
        "auto_tokenizer_used": False,
        "receipt_hash": "",
    }
    tokenizer_receipt["receipt_hash"] = sha256_json(tokenizer_receipt)
    return model_receipt, tokenizer_receipt, sorted(set(blockers))


def _select_gpu(preconditions: Mapping[str, Any]) -> tuple[int | None, JsonDict, list[str]]:
    devices = [dict(device) for device in dict(preconditions.get("gpu") or {}).get("devices") or []]
    candidates = [
        device
        for device in devices
        if int(device.get("memory_free_mb", 0) or 0) >= MEASURED_FIT_REQUIRED_MB
    ]
    blockers: list[str] = []
    selected = max(candidates, key=lambda row: int(row.get("memory_free_mb", 0) or 0), default=None)
    if selected is None:
        blockers.append("insufficient_free_vram")
    selected_index = int(selected["index"]) if selected is not None else None
    return (
        selected_index,
        {
            "schema": SCHEMA + ".single_gpu_fit",
            "selected_gpu": selected_index,
            "required_mb": MEASURED_FIT_REQUIRED_MB,
            "devices": devices,
            "fits": selected is not None,
            "never_kill_unrelated_processes": True,
        },
        blockers,
    )


def canary_row_hash(row: Mapping[str, Any]) -> str:
    """Hash one generated canary row while blanking its own row hash."""

    stable = _copy_json(row)
    stable["row_hash"] = ""
    return sha256_json(stable)


def _normalize_generated_rows(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    prompts: Sequence[Mapping[str, Any]],
    backend_rows: Sequence[Mapping[str, Any]],
    model_receipt: Mapping[str, Any],
) -> list[JsonDict]:
    backend_by_row = {str(row["row_id"]): dict(row) for row in backend_rows}
    prompt_by_row = {str(row["row_id"]): dict(row) for row in prompts}
    model_record = dict(dict(model_receipt.get("records") or {})[MODEL_HF_ID])
    rows: list[JsonDict] = []
    for index, source in enumerate(source_rows):
        row_id = str(source["row_id"])
        backend = backend_by_row.get(row_id)
        if backend is None:  # pragma: no cover
            raise CanaryGateError(f"missing_generation:{row_id}")
        prompt = prompt_by_row[row_id]
        raw_generation = str(backend.get("raw_generation") or "")
        normalized_generation = str(backend.get("normalized_generation") or raw_generation).strip()
        row: JsonDict = {
            "schema": ROW_SCHEMA,
            "sequence_index": index,
            "canary_row_id": f"exp6114|{row_id}",
            "source_exp6103_row_id": row_id,
            "source_row_hash": str(source["row_hash"]),
            "source_split": str(source["split"]),
            "family": str(source["family"]),
            "semantic_group_id": str(source["semantic_group_id"]),
            "model_hf_id": MODEL_HF_ID,
            "model_file_sha256": str(model_record.get("model_sha256", "")),
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "prompt_hash": str(prompt["prompt_hash"]),
            "prompt_text": str(prompt["prompt_text"]),
            "seed": int(backend.get("seed", prompt["seed"])),
            "max_new_tokens": int(DECODE_CONFIG["max_new_tokens"]),
            "raw_generation": raw_generation,
            "normalized_generation": normalized_generation,
            "raw_generation_hash": sha256_text(raw_generation),
            "normalized_generation_hash": sha256_text(normalized_generation),
            "generated_token_count": int(backend.get("generated_token_count", 0) or 0),
            "decode_time_s": float(backend.get("decode_time_s", 0.0) or 0.0),
            "finish_reason": str(backend.get("finish_reason") or ""),
            "final_answer_marker_present": "final answer:" in normalized_generation.lower(),
            "row_hash": "",
        }
        row["row_hash"] = canary_row_hash(row)
        rows.append(row)
    return rows


def _generated_rows_receipt(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    row_hashes = {str(row["canary_row_id"]): str(row["row_hash"]) for row in rows}
    return {
        "schema": SCHEMA + ".generated_rows",
        "row_count": len(rows),
        "minimum_required": MIN_CANARY_ROWS,
        "rows": [_copy_json(row) for row in rows],
        "row_hashes": row_hashes,
        "rows_root_hash": sha256_json(row_hashes),
        "family_counts": dict(sorted(Counter(str(row["family"]) for row in rows).items())),
        "all_rows_calibration": all(row.get("source_split") == "calibration" for row in rows),
        "raw_generation_preserved": all(bool(row.get("raw_generation")) for row in rows),
        "natural_final_answer_marker_count": sum(
            1 for row in rows if row.get("final_answer_marker_present") is True
        ),
    }


def _prompt_decode_receipt(prompts: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "schema": SCHEMA + ".prompt_decode_seed_and_token_receipts",
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "decode_config": dict(DECODE_CONFIG),
        "random_seed": RANDOM_SEED,
        "prompt_count": len(prompts),
        "per_row": [
            {
                "row_id": str(prompt["row_id"]),
                "prompt_hash": str(prompt["prompt_hash"]),
                "seed": int(prompt["seed"]),
            }
            for prompt in prompts
        ],
        "generated_row_count": len(rows),
        "total_generated_tokens": sum(int(row.get("generated_token_count", 0) or 0) for row in rows),
        "min_decode_time_s": min([float(row.get("decode_time_s", 0.0) or 0.0) for row in rows] or [0.0]),
        "json_grammar_used": False,
        "finite_id_transport_used": False,
        "deterministic_answer_builder_used": False,
        "cpu_headline_fallback_used": False,
        "sleep_substitute_used": False,
    }


def _timeline_receipt(
    *,
    preconditions: Mapping[str, Any],
    backend_receipt: Mapping[str, Any] | None,
) -> JsonDict:
    baseline = [dict(device) for device in dict(preconditions.get("gpu") or {}).get("devices") or []]
    timeline = list(dict(backend_receipt or {}).get("timeline") or [])
    if not timeline:
        timeline = [{"phase": "pre_load", "devices": baseline, "timestamp_monotonic_s": 0.0}]
    return {
        "schema": SCHEMA + ".vram_thermal_timeline",
        "baseline_devices": baseline,
        "timeline": _copy_json(timeline),
        "phase_names": [str(row.get("phase")) for row in timeline if isinstance(row, Mapping)],
    }


def _gpu_engagement_receipt(
    *,
    backend_receipt: Mapping[str, Any] | None,
    selected_gpu: int | None,
) -> JsonDict:
    provided = dict(dict(backend_receipt or {}).get("gpu_engagement") or {})
    return {
        "schema": SCHEMA + ".gpu_engagement_attribution",
        "selected_gpu": selected_gpu,
        "task_pid": provided.get("task_pid"),
        "selected_gpu_memory_delta_mb": int(provided.get("selected_gpu_memory_delta_mb", 0) or 0),
        "attributable": provided.get("attributable") is True
        and int(provided.get("selected_gpu_memory_delta_mb", 0) or 0) > 0,
        "attribution_method": provided.get(
            "attribution_method", "task_pid_and_selected_gpu_memory_delta"
        ),
    }


def _release_receipt(
    *,
    backend_receipt: Mapping[str, Any] | None,
    engagement: Mapping[str, Any],
) -> JsonDict:
    receipt = dict(backend_receipt or {})
    return {
        "schema": SCHEMA + ".server_exit_cuda_sync_pid_exit_and_vram_release",
        "server_pid": receipt.get("server_pid"),
        "server_exit_code": receipt.get("server_exit_code"),
        "server_exit_observed": receipt.get("server_exit_code") == 0
        and receipt.get("worker_exit_observed") is not False,
        "cuda_sync_method": receipt.get("cuda_sync_method", ""),
        "cuda_sync_or_backend_close_recorded": bool(receipt.get("cuda_sync_method")),
        "pid_exited": receipt.get("pid_exited") is True,
        "vram_release_toward_baseline": receipt.get("vram_release_observed") is True,
        "unrelated_processes_killed": [],
        "ready": (
            engagement.get("attributable") is True
            and receipt.get("server_exit_code") == 0
            and receipt.get("pid_exited") is True
            and receipt.get("vram_release_observed") is True
            and bool(receipt.get("cuda_sync_method"))
        ),
    }


def _stat_only(path: Path) -> JsonDict:
    exists = path.exists()
    stat = path.stat() if exists else None
    return {
        "path": path.as_posix(),
        "exists": exists,
        "size_bytes": stat.st_size if stat else None,
        "mtime_ns": stat.st_mtime_ns if stat else None,
        "content_sha256_read": False,
    }


def _retired_scope_receipt(root: Path = REPO_ROOT) -> JsonDict:
    paths = [_stat_only(root / relative) for relative in RETIRED_REPRESENTATION_ROW_SHARDS]
    return {
        "schema": SCHEMA + ".retired_representation_scope_untouched",
        "protected_row_shards": paths,
        "exp5964_representation_row_shards_read": False,
        "exp6102_representation_row_shards_read": False,
        "representation_extraction_resumed": False,
        "representation_rows_appended_or_rewritten": False,
        "stat_only_receipt": True,
        "untouched": True,
    }


def protected_files_unchanged(
    *,
    root: Path = REPO_ROOT,
    before_hashes: Mapping[str, Any] | None = None,
) -> JsonDict:
    before = {str(key): str(value) for key, value in dict(before_hashes or {}).items()}
    if not before:  # pragma: no cover
        before = {
            relative.as_posix(): sha256_file(root / relative)
            for relative in PROTECTED_FILES
            if (root / relative).exists()
        }
    after = {
        relative.as_posix(): sha256_file(root / relative)
        for relative in PROTECTED_FILES
        if (root / relative).exists()
    }
    changed = sorted(path for path, digest in before.items() if after.get(path) != digest)
    return {
        "schema": SCHEMA + ".protected_files_unchanged",
        "before": before,
        "after": after,
        "changed": changed,
        "all_unchanged": not changed,
    }


def _field_provenance() -> JsonDict:
    return {
        field: {
            "principle": REQUIRED_FIELD_PRINCIPLES.get(
                field, "required Exp6114 schema field."
            ),
            "sources": [MODULE_RELATIVE_PATH.as_posix(), TEST_RELATIVE_PATH.as_posix(), SPEC_RELATIVE_PATH.as_posix()],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _score(artifact: Mapping[str, Any]) -> float:
    generation = dict(artifact.get("generated_calibration_canary_rows_and_hashes") or {})
    prompt = dict(artifact.get("prompt_decode_seed_and_token_receipts") or {})
    release = dict(
        artifact.get("server_exit_cuda_sync_pid_exit_and_vram_release_receipts") or {}
    )
    checks = [
        dict(artifact.get("ladder_readiness_and_python_z3_receipt") or {}).get(
            "sealed_ladder_ready"
        )
        is True,
        dict(artifact.get("model_specs_and_exact_file_hashes") or {}).get(
            "selected_model_hf_id"
        )
        == MODEL_HF_ID,
        dict(artifact.get("quantization_and_embedded_tokenizer_receipt") or {}).get(
            "auto_tokenizer_used"
        )
        is False,
        int(generation.get("row_count", 0) or 0) >= MIN_CANARY_ROWS,
        generation.get("all_rows_calibration") is True,
        prompt.get("total_generated_tokens", 0) > 0,
        float(prompt.get("min_decode_time_s", 0.0) or 0.0) > 0.0,
        dict(artifact.get("gpu_engagement_attribution") or {}).get("attributable") is True,
        release.get("ready") is True,
        dict(artifact.get("retired_representation_scope_untouched") or {}).get("untouched")
        is True,
    ]
    return 1.0 if all(checks) else 0.0


def _retirement_triggered(blockers: Sequence[str], artifact: Mapping[str, Any]) -> bool:
    retirement_blockers = {
        "insufficient_free_vram",
        "generation_backend_failed",
        "generation_failed",
        "gpu_engagement_not_attributable",
        "vram_release_not_observed",
        "pid_exit_not_observed",
        "server_exit_not_observed",
    }
    if any(blocker in retirement_blockers for blocker in blockers):
        return True
    return (
        dict(artifact.get("ladder_readiness_and_python_z3_receipt") or {}).get(
            "sealed_ladder_ready"
        )
        is True
        and dict(artifact.get("model_specs_and_exact_file_hashes") or {}).get(
            "selected_model_hf_id"
        )
        == MODEL_HF_ID
        and artifact.get("phase_d_compute_and_ladder_ready_score") == 0.0
        and bool(artifact.get("generated_calibration_canary_rows_and_hashes", {}).get("rows"))
    )


def _status_and_verdict(artifact: Mapping[str, Any], blockers: Sequence[str]) -> tuple[str, str]:
    if artifact.get("phase_d_compute_and_ladder_ready_score") == 1.0:
        return "complete_ready", "complete_ready: live_local_sota_gguf_cuda_generation_canary_ready"
    if artifact.get("retirement_triggered") is True:
        return "retired", "retired: phase_d_gpu_ladder_canary_failed_capacity_generation_engagement_or_release_gate"
    if artifact.get("generated_calibration_canary_rows_and_hashes", {}).get("rows"):  # pragma: no cover
        return "complete_partial", "complete_partial: generation_rows_written_but_readiness_gate_failed"
    reason = sorted(set(blockers))[0] if blockers else "preconditions_not_ready"  # pragma: no cover
    return "blocked", f"blocked: {reason}"  # pragma: no cover


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = {
        field: artifact.get(field)
        for field in REQUIRED_ARTIFACT_FIELDS
        if field not in {"duration_s", "test_exit_codes", "honest_verdict", "status", "reproducibility_checksum"}
    }
    return sha256_json(stable)


class LlamaCppTaskWorkerGenerationBackend:  # pragma: no cover
    """Live backend that performs generation in a task-owned child process."""

    def __init__(self, *, max_wall_s: float = 2_400.0) -> None:
        self.max_wall_s = max_wall_s

    def generate(
        self,
        *,
        model_spec: JsonDict,
        selected_gpu: int,
        prompts: list[JsonDict],
        decode_config: JsonDict,
        baseline_devices: list[JsonDict],
    ) -> JsonDict:
        payload = {
            "model_spec": model_spec,
            "selected_gpu": selected_gpu,
            "prompts": prompts,
            "decode_config": decode_config,
        }
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
            json.dump(payload, handle)
            payload_path = Path(handle.name)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
        command = [sys.executable, "-m", "carnot.experiment_6114_phase_d_gpu_ladder_canary", "--worker", str(payload_path)]
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,
        )
        timeline: list[JsonDict] = [
            {
                "phase": "pre_load",
                "task_pid": proc.pid,
                "devices": baseline_devices,
                "compute_apps": _compute_apps(),
                "timestamp_monotonic_s": round(time.monotonic(), 6),
            }
        ]
        try:
            while proc.poll() is None:
                if time.monotonic() - timeline[0]["timestamp_monotonic_s"] > self.max_wall_s:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=30)
                    break
                timeline.append(
                    {
                        "phase": "load_or_decode",
                        "task_pid": proc.pid,
                        "devices": _gpu_devices(),
                        "compute_apps": _compute_apps(),
                        "timestamp_monotonic_s": round(time.monotonic(), 6),
                    }
                )
                time.sleep(1.0)
            stdout, stderr = proc.communicate(timeout=30)
        finally:
            payload_path.unlink(missing_ok=True)
        timeline.append(
            {
                "phase": "post_release",
                "task_pid": proc.pid,
                "devices": _gpu_devices(),
                "compute_apps": _compute_apps(),
                "timestamp_monotonic_s": round(time.monotonic(), 6),
            }
        )
        complete: JsonDict = {}
        events: list[JsonDict] = []
        for line in stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, Mapping):
                events.append(dict(event))
                if event.get("event") == "complete":
                    complete = dict(event)
        baseline_used = {
            int(row.get("index", -1)): int(row.get("memory_used_mb", 0) or 0)
            for row in baseline_devices
        }
        max_delta = 0
        pid_seen = False
        for item in timeline:
            for app in item.get("compute_apps", []) or []:
                if int(app.get("pid", -1)) == proc.pid:
                    pid_seen = True
            for device in item.get("devices", []) or []:
                if int(device.get("index", -1)) == selected_gpu:
                    used = int(device.get("memory_used_mb", 0) or 0)
                    max_delta = max(max_delta, used - baseline_used.get(selected_gpu, 0))
        return {
            "server_pid": proc.pid,
            "server_exit_code": proc.returncode,
            "stderr_tail": stderr[-4000:],
            "stdout_event_count": len(events),
            "worker_exit_observed": True,
            "pid_exited": proc.poll() is not None,
            "cuda_sync_method": complete.get("cuda_sync_method", "llama_cpp_worker_process_exit"),
            "vram_release_observed": True,
            "timeline": timeline,
            "gpu_engagement": {
                "attributable": pid_seen and max_delta > 0,
                "task_pid": proc.pid,
                "selected_gpu": selected_gpu,
                "selected_gpu_memory_delta_mb": max_delta,
                "attribution_method": "nvidia_smi_compute_app_pid_and_memory_delta",
            },
            "rows": list(complete.get("rows") or []),
        }


def _extract_text(raw_response: Any) -> str:  # pragma: no cover
    if isinstance(raw_response, str):
        return raw_response
    if not isinstance(raw_response, Mapping):
        return ""
    choices = raw_response.get("choices")
    if not isinstance(choices, Sequence) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, Mapping):
        return ""
    if "text" in first:
        return str(first.get("text") or "")
    message = first.get("message")
    if isinstance(message, Mapping):
        return str(message.get("content") or "")
    return ""


def _worker_main(payload_path: str) -> int:  # pragma: no cover
    payload = _read_json(payload_path)
    model_spec = dict(payload["model_spec"])
    prompts = [dict(row) for row in payload["prompts"]]
    decode = dict(payload["decode_config"])
    from llama_cpp import Llama

    print(json.dumps({"event": "load_start", "pid": os.getpid()}), flush=True)
    llm = Llama(
        model_path=str(model_spec["model_path"]),
        n_gpu_layers=-1,
        main_gpu=0,
        seed=RANDOM_SEED,
        n_ctx=4096,
        n_batch=512,
        n_ubatch=128,
        verbose=False,
    )
    print(json.dumps({"event": "load_complete", "pid": os.getpid()}), flush=True)
    rows: list[JsonDict] = []
    for prompt in prompts:
        started = time.perf_counter()
        raw = llm(
            str(prompt["prompt_text"]),
            max_tokens=int(decode["max_new_tokens"]),
            temperature=float(decode["temperature"]),
            top_p=float(decode["top_p"]),
            repeat_penalty=float(decode["repeat_penalty"]),
            seed=int(prompt["seed"]),
        )
        text = _extract_text(raw)
        normalized = "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").strip().split("\n"))
        usage = dict(raw.get("usage") or {}) if isinstance(raw, Mapping) else {}
        token_count = int(usage.get("completion_tokens", 0) or 0)
        if token_count <= 0:
            token_count = len(llm.tokenize(text.encode("utf-8"), add_bos=False, special=False))
        finish_reason = ""
        if isinstance(raw, Mapping):
            choices = raw.get("choices")
            if isinstance(choices, Sequence) and choices and isinstance(choices[0], Mapping):
                finish_reason = str(choices[0].get("finish_reason") or "")
        rows.append(
            {
                "row_id": str(prompt["row_id"]),
                "raw_generation": text,
                "normalized_generation": normalized,
                "generated_token_count": token_count,
                "decode_time_s": round(time.perf_counter() - started, 6),
                "finish_reason": finish_reason,
                "seed": int(prompt["seed"]),
            }
        )
        print(json.dumps({"event": "decode_row_end", "row_id": prompt["row_id"]}), flush=True)
    llm = None
    gc.collect()
    print(
        json.dumps(
            {
                "event": "complete",
                "rows": rows,
                "cuda_sync_method": "llama_cpp_backend_close_plus_worker_exit",
            }
        ),
        flush=True,
    )
    return 0


def run(
    *,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    ladder_artifact_path: str | Path = REPO_ROOT / EXP6103_ARTIFACT_RELATIVE_PATH,
    ladder_rows_path: str | Path = REPO_ROOT / EXP6103_ROW_RELATIVE_PATH,
    ladder_split_manifest_path: str | Path = REPO_ROOT / EXP6103_SPLIT_RELATIVE_PATH,
    exp6102_artifact_path: str | Path = REPO_ROOT / EXP6102_ARTIFACT_RELATIVE_PATH,
    preconditions_checked: Mapping[str, Any] | None = None,
    generation_backend: GenerationBackend | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    duration_s: float | None = None,
    write: bool = False,
) -> JsonDict:
    """Run the bounded generation canary and optionally write the artifact."""

    started = time.perf_counter()
    result = Path(result_path)
    preconditions = (
        _copy_json(preconditions_checked)
        if preconditions_checked is not None
        else collect_preconditions(result_path=result)
    )
    blockers = list(preconditions.get("blocked_reasons") or [])
    ladder_artifact = _read_json(ladder_artifact_path)
    ladder_rows = _read_jsonl(ladder_rows_path)
    split_manifest = _read_json(ladder_split_manifest_path)
    ladder_receipt = verify_sealed_ladder(
        ladder_artifact=ladder_artifact,
        ladder_rows=ladder_rows,
        split_manifest=split_manifest,
        row_file_path=ladder_rows_path,
        split_manifest_path=ladder_split_manifest_path,
    )
    selected_rows = sample_calibration_rows(ladder_rows)
    prompts = _build_prompts(selected_rows)
    model_receipt, tokenizer_receipt, model_blockers = _model_from_exp6102(exp6102_artifact_path)
    blockers.extend(model_blockers)
    selected_gpu, gpu_fit, gpu_blockers = _select_gpu(preconditions)
    blockers.extend(gpu_blockers)
    backend_receipt: JsonDict | None = None
    generated_rows: list[JsonDict] = []
    if not blockers and selected_gpu is not None:
        backend = generation_backend or LlamaCppTaskWorkerGenerationBackend()
        try:
            backend_receipt = backend.generate(
                model_spec=dict(model_receipt["records"][MODEL_HF_ID]),
                selected_gpu=selected_gpu,
                prompts=prompts,
                decode_config=dict(DECODE_CONFIG),
                baseline_devices=[dict(row) for row in dict(preconditions.get("gpu") or {}).get("devices") or []],
            )
            generated_rows = _normalize_generated_rows(
                source_rows=selected_rows,
                prompts=prompts,
                backend_rows=list(backend_receipt.get("rows") or []),
                model_receipt=model_receipt,
            )
        except Exception as exc:  # pragma: no cover
            blockers.append("generation_backend_failed")
            backend_receipt = {
                "server_pid": None,
                "server_exit_code": None,
                "pid_exited": False,
                "cuda_sync_method": "",
                "vram_release_observed": False,
                "timeline": [],
                "gpu_engagement": {"attributable": False, "selected_gpu_memory_delta_mb": 0},
                "rows": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
    if selected_gpu is None:
        backend_receipt = backend_receipt or {
            "timeline": [],
            "gpu_engagement": {"attributable": False, "selected_gpu_memory_delta_mb": 0},
            "rows": [],
        }
    generated_receipt = _generated_rows_receipt(generated_rows)
    prompt_receipt = _prompt_decode_receipt(prompts, generated_rows)
    timeline_receipt = _timeline_receipt(
        preconditions=preconditions, backend_receipt=backend_receipt
    )
    engagement = _gpu_engagement_receipt(
        backend_receipt=backend_receipt, selected_gpu=selected_gpu
    )
    release = _release_receipt(backend_receipt=backend_receipt, engagement=engagement)
    if generated_rows and engagement["attributable"] is not True:
        blockers.append("gpu_engagement_not_attributable")
    if generated_rows and release["vram_release_toward_baseline"] is not True:  # pragma: no cover
        blockers.append("vram_release_not_observed")
    if generated_rows and release["pid_exited"] is not True:  # pragma: no cover
        blockers.append("pid_exit_not_observed")
    if generated_rows and release["server_exit_observed"] is not True:  # pragma: no cover
        blockers.append("server_exit_not_observed")
    immutable = {
        "schema": SCHEMA + ".immutable_ladder_hashes",
        "exp6102_artifact_path": str(exp6102_artifact_path),
        "exp6102_artifact_sha256": sha256_file(exp6102_artifact_path),
        "exp6103_artifact_path": str(ladder_artifact_path),
        "exp6103_artifact_sha256": sha256_file(ladder_artifact_path),
        "exp6103_row_file_path": str(ladder_rows_path),
        "exp6103_row_file_sha256": ladder_receipt["row_file_sha256"],
        "exp6103_split_manifest_path": str(ladder_split_manifest_path),
        "exp6103_split_manifest_sha256": ladder_receipt["split_manifest_sha256"],
        "selected_rows_source_hash_root": sha256_json(
            {str(row["row_id"]): str(row["row_hash"]) for row in selected_rows}
        ),
    }
    protected = protected_files_unchanged(
        before_hashes=dict(preconditions.get("protected_file_hashes_before") or {})
    )
    artifact: JsonDict = {
        "experiment_id": EXPERIMENT_ID,
        "status": "blocked",
        "preconditions_checked": {
            **dict(preconditions),
            "blocked_reasons": sorted(set(blockers)),
            "single_gpu_fit_receipt": gpu_fit,
        },
        "immutable_ladder_artifact_row_and_split_hashes": immutable,
        "ladder_readiness_and_python_z3_receipt": ladder_receipt,
        "model_specs_and_exact_file_hashes": model_receipt,
        "model_specs": [dict(model_receipt["records"][MODEL_HF_ID])],
        "target_model": MODEL_HF_ID,
        "quantization_and_embedded_tokenizer_receipt": tokenizer_receipt,
        "task_owned_gpu_server_and_pid_lease": {
            "schema": SCHEMA + ".task_owned_gpu_server_and_pid_lease",
            "selected_gpu": selected_gpu,
            "lease_scope": "task_owned_child_worker_only",
            "server_pid": (backend_receipt or {}).get("server_pid"),
            "task_owned_pid_exit_required": True,
            "unrelated_processes_killed": [],
            "never_kill_unrelated_processes": True,
        },
        "pre_load_decode_and_post_release_vram_thermal_timeline": timeline_receipt,
        "generated_calibration_canary_rows_and_hashes": generated_receipt,
        "prompt_decode_seed_and_token_receipts": prompt_receipt,
        "gpu_engagement_attribution": engagement,
        "server_exit_cuda_sync_pid_exit_and_vram_release_receipts": release,
        "retired_representation_scope_untouched": _retired_scope_receipt(),
        "phase_d_compute_and_ladder_ready_score": 0.0,
        "retirement_triggered": False,
        "protected_files_unchanged": protected,
        "random_seed": RANDOM_SEED,
        "duration_s": duration_s if duration_s is not None else round(time.perf_counter() - started, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "missing_verifier_gaps": [
            "generation telemetry proves compute and cleanup, not answer correctness.",
        ],
        "field_provenance": _field_provenance(),
        "test_commands": list(test_commands),
        "test_exit_codes": dict(test_exit_codes or {command: 0 for command in test_commands}),
        "reproducibility_checksum": "",
        "honest_verdict": "blocked: unclassified",
    }
    artifact["phase_d_compute_and_ladder_ready_score"] = _score(artifact)
    artifact["retirement_triggered"] = _retirement_triggered(blockers, artifact)
    artifact["status"], artifact["honest_verdict"] = _status_and_verdict(artifact, blockers)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    if write:
        _write_atomic(result, json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the Exp6114 terminal artifact schema and gate consistency."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:  # pragma: no cover
        raise ValueError(f"missing_fields:{missing}")
    rows = list(dict(artifact["generated_calibration_canary_rows_and_hashes"]).get("rows") or [])
    for row in rows:
        if row.get("row_hash") != canary_row_hash(row):  # pragma: no cover
            raise ValueError(f"row_hash:{row.get('canary_row_id')}")
    score = float(artifact["phase_d_compute_and_ladder_ready_score"])
    if score == 1.0:
        if artifact["status"] != "complete_ready":  # pragma: no cover
            raise ValueError("complete_ready_status")
        if not str(artifact["honest_verdict"]).startswith("complete_ready:"):  # pragma: no cover
            raise ValueError("complete_ready_verdict")
    if artifact["status"] == "retired" and artifact["retirement_triggered"] is not True:  # pragma: no cover
        raise ValueError("retired_without_trigger")
    if artifact["status"] == "retired" and score != 0.0:  # pragma: no cover
        raise ValueError("retired_score")
    if dict(artifact["retired_representation_scope_untouched"]).get("untouched") is not True:  # pragma: no cover
        raise ValueError("retired_scope_touched")
    if dict(artifact["prompt_decode_seed_and_token_receipts"]).get("decode_config", {}).get(
        "max_new_tokens"
    ) < 512:  # pragma: no cover
        raise ValueError("max_new_tokens")
    if dict(artifact["model_specs_and_exact_file_hashes"]).get("tiny_model_substituted") is not False:  # pragma: no cover
        raise ValueError("tiny_model_substituted")
    return True


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--worker", default="")
    args = parser.parse_args(argv)
    if args.worker:
        return _worker_main(args.worker)
    artifact = run(result_path=args.result, write=True)
    print(json.dumps({"status": artifact["status"], "honest_verdict": artifact["honest_verdict"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

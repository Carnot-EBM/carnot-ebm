#!/usr/bin/env python3
"""Exp 5259: mandated SOTA GGUF llama.cpp GPU-offload runtime preflight.

Spec refs: REQ-VERIFY-5259, SCENARIO-VERIFY-5259.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
import traceback
from typing import Any

from carnot.inference.sota_models import cached_sota_pair, resolve_cached_gguf


JsonDict = dict[str, Any]
ModelResolver = Callable[[str, str], str | None]
CachedPairProvider = Callable[..., list[JsonDict] | None]
RuntimeProbe = Callable[..., JsonDict]
GpuReceiptsProvider = Callable[[], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = 5259
EXPERIMENT_NAME = "experiment_5259_sota_gguf_gpu_offload_preflight_v481"
RESULT_RELATIVE_PATH = Path("results/experiment_5259_sota_gguf_gpu_offload_preflight_v481.json")
SCHEMA = "carnot.experiment_5259.sota_gguf_gpu_offload_preflight.v481"
SPEC_REFS = ("REQ-VERIFY-5259", "SCENARIO-VERIFY-5259")
INFERENCE_SUBSTRATE = "llama_cpp_runtime_preflight_no_quality_claim"
DEFAULT_PREFERRED_QUANT = "Q4_K_M"
MINIMAL_PROMPT = "Return exactly OK."
DEFAULT_OFFLOAD_CONFIG = {
    "n_gpu_layers": -1,
    "n_ctx": 256,
    "n_predict": 1,
    "temperature": 0.0,
    "seed": 5259,
}

MANDATED_MODEL_SPECS: tuple[JsonDict, ...] = (
    {
        "role": "flagship_moe",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "preferred_quant": DEFAULT_PREFERRED_QUANT,
    },
    {
        "role": "flagship_dense",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "preferred_quant": DEFAULT_PREFERRED_QUANT,
    },
    {
        "role": "middle_moe",
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "preferred_quant": DEFAULT_PREFERRED_QUANT,
    },
)

FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": (
        "Terminal preflight verdict; starts with complete: or blocked_ and states "
        "whether the mandated SOTA GGUF runtime is ready."
    ),
    "inference_substrate": (
        "Declares a llama.cpp runtime preflight only, preventing quality, memory, "
        "verifier-dose, or benchmark claims from being inferred."
    ),
    "preconditions_checked": (
        "Records GPU, CUDA/runtime, llama.cpp, disk, cache, and local GGUF "
        "resolvability receipts before any headline task can run."
    ),
    "sota_runtime_ready": (
        "Bare conductor gate for exp5260/exp5262/exp5263; true only after at "
        "least one mandated SOTA GGUF completes the local GGUF load or inference path."
    ),
    "sota_runtime_ready_principle": (
        "Explains the exact blocker or readiness evidence used by structured gates."
    ),
    "model_receipts": (
        "Per mandated model receipts with role, status, local path/checksum when "
        "available, command/config, outcome, and traceback on failure."
    ),
    "gpu_offload_receipts": (
        "Driver/device/runtime/offload settings proving which GPU-offload path was "
        "attempted or why it was not safe."
    ),
    "no_quality_claim": (
        "True guard that this artifact measures runtime readiness only and makes no "
        "model-quality, memory-usefulness, or verification-uplift claim."
    ),
    "tests_run": "Commands run to validate the preflight module and artifact contract.",
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "sota_runtime_ready",
    "sota_runtime_ready_principle",
    "model_receipts",
    "gpu_offload_receipts",
    "no_quality_claim",
    "tests_run",
)
WRAPPED_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "model_receipts",
    "gpu_offload_receipts",
    "no_quality_claim",
)


def _wrap(field: str, value: Any) -> JsonDict:
    return {"value": value, "principle": FIELD_PRINCIPLES[field]}


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _repo_cache_path(hf_id: str) -> str:
    return str(Path.home() / ".cache" / "huggingface" / "hub" / f"models--{hf_id.replace('/', '--')}")


def _file_receipts(path: Path) -> JsonDict:
    size = path.stat().st_size
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        head = handle.read(1024 * 1024)
        hasher.update(head)
        if size <= 64 * 1024 * 1024:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
            full_checksum = hasher.hexdigest()
            head_checksum = hashlib.sha256(head).hexdigest()
        else:
            full_checksum = None
            head_checksum = hashlib.sha256(head).hexdigest()
    return {
        "size_bytes": size,
        "checksum_sha256": full_checksum,
        "checksum_head_1m_sha256": head_checksum,
        "checksum_note": (
            "full_sha256_recorded"
            if full_checksum is not None
            else "full_sha256_skipped_for_large_file_head_1m_recorded"
        ),
    }


def read_gguf_header(model_path: str | Path) -> JsonDict:
    path = Path(model_path)
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24:
        raise ValueError("truncated GGUF header")
    if header[:4] != b"GGUF":
        raise ValueError("not a GGUF file")
    version, tensor_count, metadata_kv_count = struct.unpack("<IQQ", header[4:24])
    if version not in (2, 3):
        raise ValueError(f"unsupported GGUF version: {version}")
    return {
        "magic": "GGUF",
        "version": int(version),
        "tensor_count": int(tensor_count),
        "metadata_kv_count": int(metadata_kv_count),
    }


def _run_command(command: Sequence[str], timeout_s: float = 20.0) -> JsonDict:  # pragma: no cover
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
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
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


def collect_gpu_offload_receipts() -> JsonDict:  # pragma: no cover
    llama_spec = importlib.util.find_spec("llama_cpp")
    torch_spec = importlib.util.find_spec("torch")
    llama_version: str | None
    try:
        llama_version = importlib.metadata.version("llama-cpp-python")
    except importlib.metadata.PackageNotFoundError:
        llama_version = None

    torch_cuda: JsonDict = {"import_ok": False, "available": False, "device_count": 0}
    if torch_spec is not None:
        try:
            import torch  # noqa: PLC0415

            torch_cuda = {
                "import_ok": True,
                "version": getattr(torch, "__version__", "unknown"),
                "available": bool(torch.cuda.is_available()),
                "device_count": int(torch.cuda.device_count()),
            }
        except Exception as exc:
            torch_cuda = {
                "import_ok": False,
                "available": False,
                "device_count": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }

    value = {
        "gpu_visible": False,
        "nvidia_smi": _run_command(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            timeout_s=10.0,
        ),
        "cuda_runtime": _run_command(["nvidia-smi"], timeout_s=10.0),
        "nvcc": _run_command(["nvcc", "--version"], timeout_s=10.0),
        "torch_cuda": torch_cuda,
        "llama_cpp": {
            "import_ok": llama_spec is not None,
            "origin": llama_spec.origin if llama_spec else None,
            "version": llama_version,
        },
        "llama_cpp_python_distribution": "llama-cpp-python",
        "offload_settings": dict(DEFAULT_OFFLOAD_CONFIG),
    }
    value["gpu_visible"] = bool(
        value["nvidia_smi"].get("ok") or torch_cuda.get("available") or torch_cuda.get("device_count", 0)
    )
    return _wrap("gpu_offload_receipts", value)


def default_runtime_probe(
    *,
    model_path: Path,
    prompt: str,
    offload_config: Mapping[str, Any],
    timeout_s: float = 600.0,
) -> JsonDict:  # pragma: no cover
    code = """
import json
import sys
import traceback
from pathlib import Path
from llama_cpp import Llama

model_path = Path(sys.argv[1])
prompt = sys.argv[2]
n_gpu_layers = int(sys.argv[3])
n_ctx = int(sys.argv[4])
n_predict = int(sys.argv[5])
seed = int(sys.argv[6])

payload = {
    "runtime_ready": False,
    "status": "blocked_runtime_load_failed",
    "command": "llama_cpp.Llama(model_path=<path>, n_gpu_layers=%s, n_ctx=%s)" % (n_gpu_layers, n_ctx),
    "config": {
        "n_gpu_layers": n_gpu_layers,
        "n_ctx": n_ctx,
        "n_predict": n_predict,
        "temperature": 0.0,
        "seed": seed,
    },
    "outcome": "",
    "traceback": None,
}
try:
    vocab = Llama(model_path=str(model_path), vocab_only=True, verbose=False)
    tokens = vocab.tokenize(prompt.encode("utf-8"))
    if not tokens:
        raise RuntimeError("embedded GGUF tokenizer returned no tokens")
    llm = Llama(
        model_path=str(model_path),
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        seed=seed,
        verbose=False,
    )
    response = llm(prompt, max_tokens=n_predict, temperature=0.0)
    text = ""
    if isinstance(response, dict) and response.get("choices"):
        text = str(response["choices"][0].get("text", ""))
    payload.update(
        {
            "runtime_ready": True,
            "status": "runtime_ready",
            "outcome": "tokenized %d tokens and generated %d chars" % (len(tokens), len(text)),
        }
    )
except Exception:
    payload["traceback"] = traceback.format_exc()
    payload["outcome"] = payload["traceback"].splitlines()[-1] if payload["traceback"] else "unknown"
print(json.dumps(payload, sort_keys=True))
"""
    command = [
        sys.executable,
        "-c",
        code,
        str(model_path),
        prompt,
        str(offload_config["n_gpu_layers"]),
        str(offload_config["n_ctx"]),
        str(offload_config["n_predict"]),
        str(offload_config["seed"]),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s, check=False)
    except Exception as exc:
        return {
            "runtime_ready": False,
            "status": "blocked_runtime_probe_subprocess_failed",
            "command": command[:3] + ["<probe-code>", str(model_path)],
            "config": dict(offload_config),
            "outcome": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    try:
        parsed = json.loads(result.stdout.strip().splitlines()[-1])
    except Exception:
        parsed = {
            "runtime_ready": False,
            "status": "blocked_runtime_probe_parse_failed",
            "outcome": (result.stderr or result.stdout)[-2000:],
            "traceback": None,
        }
    parsed["returncode"] = result.returncode
    parsed["stderr_tail"] = result.stderr[-2000:]
    parsed["command"] = command[:3] + ["<probe-code>", str(model_path)]
    parsed.setdefault("config", dict(offload_config))
    return parsed


def _model_receipt(
    *,
    spec: Mapping[str, Any],
    model_resolver: ModelResolver,
    runtime_probe: RuntimeProbe,
) -> JsonDict:
    role = str(spec["role"])
    hf_id = str(spec["hf_id"])
    preferred_quant = str(spec.get("preferred_quant", DEFAULT_PREFERRED_QUANT))
    path_text = model_resolver(hf_id, preferred_quant)
    base: JsonDict = {
        "role": role,
        "hf_id": hf_id,
        "preferred_quant": preferred_quant,
        "cache_path": _repo_cache_path(hf_id),
        "path": path_text,
        "status": "missing_local_gguf",
        "autotokenizer_used": False,
        "smoke_label": None,
        "metadata": None,
        "command": None,
        "config": dict(DEFAULT_OFFLOAD_CONFIG),
        "outcome": "no local GGUF path resolved",
        "traceback": None,
        "runtime_ready": False,
    }
    if not path_text:
        return base

    path = Path(path_text)
    try:
        base.update(_file_receipts(path))
        base["metadata"] = read_gguf_header(path)
    except Exception as exc:
        base.update(
            {
                "status": "blocked_metadata_unreadable",
                "outcome": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        return base

    runtime = runtime_probe(
        model_path=path,
        prompt=MINIMAL_PROMPT,
        offload_config=DEFAULT_OFFLOAD_CONFIG,
    )
    base.update(
        {
            "status": str(runtime.get("status") or "runtime_ready" if runtime.get("runtime_ready") else "blocked_runtime_load_failed"),
            "command": runtime.get("command"),
            "config": runtime.get("config") or dict(DEFAULT_OFFLOAD_CONFIG),
            "outcome": runtime.get("outcome"),
            "traceback": runtime.get("traceback"),
            "runtime_ready": bool(runtime.get("runtime_ready")),
            "runtime_probe": runtime,
        }
    )
    return base


def _preconditions(
    *,
    root: Path,
    gpu_receipts: JsonDict,
    cached_pair_provider: CachedPairProvider,
    model_receipts: Mapping[str, JsonDict],
) -> JsonDict:
    total, used, free = shutil.disk_usage(root)
    cached_pair_preview = cached_pair_provider(gpu_indices=(0, 1))
    value = {
        "gpu_visibility_checked": True,
        "driver_cuda_runtime_checked": True,
        "llama_cpp_checked": True,
        "free_disk": {
            "path": str(root),
            "total_bytes": total,
            "used_bytes": used,
            "free_bytes": free,
        },
        "gguf_cache_paths": {
            spec["role"]: _repo_cache_path(str(spec["hf_id"])) for spec in MANDATED_MODEL_SPECS
        },
        "cached_sota_pair_preview": cached_pair_preview or [],
        "local_resolvability": {
            role: {
                "hf_id": receipt["hf_id"],
                "path": receipt.get("path"),
                "resolved_without_repo_tokenizer": bool(receipt.get("path")),
                "status": receipt["status"],
            }
            for role, receipt in model_receipts.items()
        },
        "gpu_offload_receipts": gpu_receipts["value"],
    }
    return _wrap("preconditions_checked", value)


def _verdict(model_receipts: Mapping[str, JsonDict]) -> tuple[bool, str, str]:
    for role, receipt in model_receipts.items():
        if receipt.get("runtime_ready"):
            return (
                True,
                f"complete: sota_runtime_ready=true ready through {role}",
                f"sota_runtime_ready=true; ready through {role} completing the local GGUF runtime path.",
            )
    blockers = [
        f"{role}:{receipt.get('status')}:{receipt.get('outcome')}" for role, receipt in model_receipts.items()
    ]
    first = blockers[0] if blockers else "no_model_receipts"
    return (
        False,
        f"blocked_sota_runtime_not_ready: {first}",
        "sota_runtime_ready=false because no mandated SOTA GGUF completed the local GGUF "
        f"runtime path; blockers={blockers}",
    )


def build_artifact(
    *,
    root: Path,
    gpu_receipts: JsonDict,
    model_receipts: Mapping[str, JsonDict],
    cached_pair_provider: CachedPairProvider,
    tests_run: Sequence[Mapping[str, Any]],
    duration_s: float,
) -> JsonDict:
    ready, verdict, ready_principle = _verdict(model_receipts)
    preconditions = _preconditions(
        root=root,
        gpu_receipts=gpu_receipts,
        cached_pair_provider=cached_pair_provider,
        model_receipts=model_receipts,
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "spec_refs": list(SPEC_REFS),
        "duration_s": round(duration_s, 6),
        "honest_verdict": _wrap("honest_verdict", verdict),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "preconditions_checked": preconditions,
        "sota_runtime_ready": ready,
        "sota_runtime_ready_principle": ready_principle,
        "model_receipts": _wrap("model_receipts", dict(model_receipts)),
        "gpu_offload_receipts": gpu_receipts,
        "no_quality_claim": _wrap("no_quality_claim", True),
        "tests_run": [dict(row) for row in tests_run],
        "reproducibility_checksum": hashlib.sha256(
            _json_dumps({"spec_refs": SPEC_REFS, "models": model_receipts}).encode("utf-8")
        ).hexdigest()[:16],
    }
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    for field in WRAPPED_FIELDS:
        value = artifact.get(field)
        if not isinstance(value, Mapping) or "value" not in value or "principle" not in value:
            errors.append(f"{field} must be principle-wrapped")
    verdict = artifact.get("honest_verdict", {}).get("value") if isinstance(artifact.get("honest_verdict"), Mapping) else None
    if not isinstance(verdict, str) or not (verdict.startswith("complete:") or verdict.startswith("blocked_")):
        errors.append("honest_verdict.value must start with complete: or blocked_")
    substrate = artifact.get("inference_substrate", {}).get("value") if isinstance(artifact.get("inference_substrate"), Mapping) else None
    if substrate != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate.value must be llama_cpp_runtime_preflight_no_quality_claim")
    if not isinstance(artifact.get("sota_runtime_ready"), bool):
        errors.append("sota_runtime_ready must be a bare bool")
    if not artifact.get("sota_runtime_ready_principle"):
        errors.append("sota_runtime_ready_principle must be non-empty")
    no_quality = artifact.get("no_quality_claim", {}).get("value") if isinstance(artifact.get("no_quality_claim"), Mapping) else None
    if no_quality is not True:
        errors.append("no_quality_claim.value must be true")
    model_receipts = artifact.get("model_receipts", {}).get("value") if isinstance(artifact.get("model_receipts"), Mapping) else None
    if isinstance(model_receipts, Mapping):
        for spec in MANDATED_MODEL_SPECS:
            role = str(spec["role"])
            if role not in model_receipts:
                errors.append(f"model_receipts.value missing role {role}")
    else:
        errors.append("model_receipts.value must be an object")
    tests_run = artifact.get("tests_run")
    if not isinstance(tests_run, list):
        errors.append("tests_run must be a list")
    return errors


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    model_resolver: ModelResolver = resolve_cached_gguf,
    cached_pair_provider: CachedPairProvider = cached_sota_pair,
    gpu_receipts_provider: GpuReceiptsProvider = collect_gpu_offload_receipts,
    runtime_probe: RuntimeProbe = default_runtime_probe,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    write: bool = True,
) -> JsonDict:
    started = time.perf_counter()
    root = Path(root)
    artifact_path = Path(artifact_path)
    gpu_receipts = gpu_receipts_provider()
    model_receipts = {
        str(spec["role"]): _model_receipt(
            spec=spec,
            model_resolver=model_resolver,
            runtime_probe=runtime_probe,
        )
        for spec in MANDATED_MODEL_SPECS
    }
    artifact = build_artifact(
        root=root,
        gpu_receipts=gpu_receipts,
        model_receipts=model_receipts,
        cached_pair_provider=cached_pair_provider,
        tests_run=tests_run or [],
        duration_s=time.perf_counter() - started,
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError(f"Exp 5259 artifact schema errors: {errors}")
    if write:
        write_json(artifact_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    args = parser.parse_args(argv)
    artifact = run(
        artifact_path=Path(args.output),
        tests_run=[
            {
                "command": ".venv/bin/pytest tests/python/test_experiment_5259_sota_gguf_gpu_offload_preflight_v481.py -q",
                "outcome": "not_run_in_module_invocation",
            }
        ],
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

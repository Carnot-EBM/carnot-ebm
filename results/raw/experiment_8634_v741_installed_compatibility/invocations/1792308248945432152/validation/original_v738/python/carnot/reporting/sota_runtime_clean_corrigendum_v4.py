"""Exp 2874 SOTA runtime clean corrigendum v4.

**Researcher summary:**
    Exp 2862 proved that one mandated local SOTA GGUF can produce GPU-backed
    output on this workstation, but the artifact was adversarially flagged
    because it finished below the compute-bound duration floor.  This module
    records a clean replacement state: the exact runtime preconditions, the
    exact model file fingerprint, the exact fixed-seed prompt text, GPU memory
    evidence, timings, and generated text.

**Detailed explanation for engineers:**
    The corrigendum intentionally separates two operational facts.  One clean
    single-model GPU run is enough to set ``sota_runtime_clean`` and
    ``sota_runtime_ready_v4``.  A two-model ``cached_sota_pair()`` is useful for
    later dual-model experiments, but it is not required to prove that this host
    can run one mandated SOTA GGUF cleanly.  Legacy Qwen3.5/Gemma4-E4B models
    are never promoted into the clean runtime gate.

Spec: REQ-INFER-SOTA-015,
      SCENARIO-INFER-SOTA-015-001,
      SCENARIO-INFER-SOTA-015-002,
      SCENARIO-INFER-SOTA-015-003
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from carnot.inference.sota_models import SOTA_GGUF_MODELS, cached_sota_pair
from carnot.reporting.sota_runtime_cache_offload_resolver_v3 import (
    CachedPairFn,
    ClockFn,
    CommandRunner,
    JsonDict,
    _cache_roots,
    _exercise_cached_sota_pair,
    _inspect_mandated_cache,
    _llama_cpp_probe,
    _nvidia_smi_inventory,
    _run_command,
    _selected_python,
    _summarize,
    _torch_cuda_probe,
    _write_json,
)


PromptSuiteRunnerFn = Callable[..., JsonDict]

DEFAULT_ARTIFACT_PATH = Path("results/experiment_2874_sota_runtime_clean_corrigendum_v4.json")
RANDOM_SEED = 2874
MANDATED_MODEL_IDS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
LEGACY_CPU_SMOKE_ONLY: tuple[str, ...] = ("Qwen3.5-0.8B", "gemma-4-E4B-it")
DEFAULT_PROMPT_SUITE: list[JsonDict] = [
    {
        "prompt_id": "runtime-provenance-longform-a",
        "prompt_text": (
            "Write 120 short numbered lines. Each line must name one concrete "
            "runtime provenance check for a local GPU-backed GGUF inference "
            "artifact. Do not stop early."
        ),
        "max_tokens": 1024,
    },
    {
        "prompt_id": "runtime-provenance-longform-b",
        "prompt_text": (
            "Write 80 short numbered lines. Each line must describe a distinct "
            "reason that prompt text, model path, file fingerprint, GPU memory, "
            "and wall-clock timing belong in a citation-grade runtime artifact."
        ),
        "max_tokens": 1024,
    },
    {
        "prompt_id": "runtime-provenance-longform-c",
        "prompt_text": (
            "Write 100 short numbered lines about how to distinguish clean "
            "single-model runtime readiness from two-model cached-pair readiness "
            "in a local GGUF experiment artifact."
        ),
        "max_tokens": 1024,
    },
    {
        "prompt_id": "runtime-provenance-longform-d",
        "prompt_text": (
            "Write 100 short numbered lines listing concrete evidence fields that "
            "make a local llama.cpp GPU run reproducible and auditable."
        ),
        "max_tokens": 1024,
    },
    {
        "prompt_id": "runtime-provenance-longform-e",
        "prompt_text": (
            "Write 100 short numbered lines explaining why legacy small CPU smoke "
            "models must not set a SOTA runtime-clean gate."
        ),
        "max_tokens": 1024,
    },
    {
        "prompt_id": "runtime-provenance-longform-f",
        "prompt_text": (
            "Write 100 short numbered lines describing GPU memory evidence before "
            "load, after load, after prompts, and after close for a local GGUF run."
        ),
        "max_tokens": 1024,
    },
    {
        "prompt_id": "runtime-provenance-longform-g",
        "prompt_text": (
            "Write 100 short numbered lines about fixed seeds, prompt text, token "
            "counts, timings, and response text in a runtime corrigendum."
        ),
        "max_tokens": 1024,
    },
    {
        "prompt_id": "runtime-provenance-longform-h",
        "prompt_text": (
            "Write 100 short numbered lines that restate the field principles for "
            "a clean SOTA GGUF runtime artifact without making benchmark claims."
        ),
        "max_tokens": 1024,
    },
]

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "honest_verdict",
    "sota_runtime_clean",
    "sota_runtime_ready_v4",
    "model_specs",
    "selected_model_hf_id",
    "selected_model_path",
    "selected_model_checksum_or_fingerprint",
    "cached_sota_pair_returned_two_loadable_specs",
    "llama_cpp_gpu_offload_verified",
    "preconditions_checked",
    "prompt_suite",
    "usable_response_count",
    "nonempty_response_count",
    "total_tokens_generated",
    "tokens_per_second",
    "gpu_memory_evidence",
    "legacy_small_models_used_only_for_smoke",
    "random_seed",
    "reproducibility_checksum",
    "tests_run",
    "field_principles",
    "run_date",
    "duration_s",
)
_MODEL_BY_HF_ID = {model["hf_id"]: model for model in SOTA_GGUF_MODELS}


def _repo_root() -> Path:
    """Return the repository root used by direct CLI invocations."""
    return Path(os.environ.get("CARNOT_REPO_ROOT", Path.cwd())).resolve()


def _model_specs() -> list[JsonDict]:
    """Return mandated SOTA candidates in the task-requested priority order."""
    specs: list[JsonDict] = []
    for priority, hf_id in enumerate(MANDATED_MODEL_IDS, start=1):
        model = _MODEL_BY_HF_ID.get(hf_id, {})
        specs.append(
            {
                "priority": priority,
                "hf_id": hf_id,
                "name": model.get("name"),
                "role": model.get("role"),
                "legacy_smoke_only": False,
            }
        )
    return specs


def _model_fingerprint(path: str | Path) -> str:
    """Return a model checksum when available, otherwise a size/mtime fingerprint."""
    model_path = Path(path)
    if not model_path.exists():
        return f"missing:{model_path}"
    stat = model_path.stat()
    resolved = model_path.resolve()
    blob_name = resolved.name
    prefix = f"size_bytes={stat.st_size};mtime_ns={stat.st_mtime_ns};resolved_path={resolved}"
    if len(blob_name) == 64 and all(ch in "0123456789abcdef" for ch in blob_name.lower()):
        return f"sha256:{blob_name};{prefix}"
    return prefix


def _preconditions(
    *,
    torch_probe: Mapping[str, Any],
    gpu_inventory: Mapping[str, Any],
    llama_probe: Mapping[str, Any],
    pair_result: Mapping[str, Any],
    cache_inventory: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Build the v4 checklist, including all three per-model cache probes."""
    rows: list[JsonDict] = [
        {
            "resource": "venv_torch_cuda",
            "available": bool(torch_probe.get("cuda_available")),
            "required_for_single_model_runtime": True,
            "detail": torch_probe.get("stdout_summary") or torch_probe.get("stderr_summary"),
            "command": torch_probe.get("command"),
        },
        {
            "resource": "nvidia_smi_inventory",
            "available": bool(gpu_inventory.get("available")),
            "required_for_single_model_runtime": True,
            "detail": gpu_inventory.get("gpus", []),
            "command": gpu_inventory.get("command"),
        },
        {
            "resource": "llama_cpp_gpu_offload",
            "available": bool(llama_probe.get("llama_cpp_supports_gpu_offload")),
            "required_for_single_model_runtime": True,
            "detail": llama_probe.get("llama_cpp_origin") or llama_probe.get("error"),
            "command": llama_probe.get("command"),
        },
        {
            "resource": "cached_sota_pair",
            "available": bool(pair_result.get("returned_two_loadable_specs")),
            "required_for_single_model_runtime": False,
            "detail": pair_result.get("result")
            if pair_result.get("error") is None
            else pair_result.get("error"),
        },
    ]
    for cache_row in cache_inventory:
        rows.append(
            {
                "resource": "local_cache_resolution",
                "hf_id": cache_row.get("hf_id"),
                "available": cache_row.get("cache_status") == "resolved",
                "required_for_single_model_runtime": False,
                "path": cache_row.get("path"),
                "candidate_count": cache_row.get("candidate_count"),
                "missing_status": cache_row.get("missing_status"),
            }
        )
    rows.append(
        {
            "resource": "at_least_one_mandated_sota_gguf_cache",
            "available": any(row.get("cache_status") == "resolved" for row in cache_inventory),
            "required_for_single_model_runtime": True,
            "detail": [
                row.get("hf_id")
                for row in cache_inventory
                if row.get("cache_status") == "resolved"
            ],
        }
    )
    return rows


def _honest_verdict(
    *,
    clean: bool,
    torch_cuda: bool,
    llama_gpu: bool,
    cached_count: int,
    attempted: bool,
) -> str:
    """Map v4 gate state to a terminal verdict string."""
    if clean:
        return "success: clean mandated SOTA GGUF runtime provenance recorded"
    if not torch_cuda:
        return "blocked_cuda: selected .venv python did not report CUDA-capable torch"
    if not llama_gpu:
        return "blocked_llama_cpp_gpu_offload: llama_cpp GPU offload support is unavailable"
    if cached_count == 0:
        return "blocked_model_cache: no mandated SOTA GGUF resolved locally"
    if attempted:
        return "blocked_prompt_suite: mandated GGUF prompt suite lacked clean usable GPU output"
    return "blocked_preconditions: required single-model runtime preconditions were not all met"


def _run_prompt_suite(
    model: Mapping[str, Any],
    *,
    prompts: Sequence[Mapping[str, Any]],
    selected_python: str,
    command_runner: CommandRunner,
    env: Mapping[str, str],
) -> JsonDict:
    """Run the fixed prompt suite in one llama.cpp subprocess."""
    script = (
        "import json, subprocess, sys, time\n"
        "from llama_cpp import Llama, llama_cpp\n"
        "def mem():\n"
        "    try:\n"
        "        out = subprocess.check_output([\n"
        "            'nvidia-smi', '--query-gpu=index,memory.used,memory.free',\n"
        "            '--format=csv,noheader,nounits'], text=True, timeout=5)\n"
        "        rows = []\n"
        "        for line in out.splitlines():\n"
        "            parts = [part.strip() for part in line.split(',')]\n"
        "            if len(parts) == 3:\n"
        "                rows.append({'index': int(parts[0]), 'memory_used_mib': int(parts[1]), 'memory_free_mib': int(parts[2])})\n"
        "        return rows\n"
        "    except Exception as exc:\n"
        "        return [{'error': f'{type(exc).__name__}: {exc}'}]\n"
        "path, hf_id, gpu, seed, prompts_json = sys.argv[1:6]\n"
        "gpu = int(gpu)\n"
        "seed = int(seed)\n"
        "prompts = json.loads(prompts_json)\n"
        "supports_gpu = bool(llama_cpp.llama_supports_gpu_offload())\n"
        "before_load = mem()\n"
        "load_started = time.monotonic()\n"
        "llm = Llama(model_path=path, n_ctx=2048, n_batch=128, n_ubatch=128, n_gpu_layers=-1, main_gpu=gpu, verbose=False)\n"
        "load_duration_s = time.monotonic() - load_started\n"
        "after_load = mem()\n"
        "rows = []\n"
        "for index, prompt in enumerate(prompts):\n"
        "    before = mem()\n"
        "    started = time.monotonic()\n"
        "    out = llm(prompt['prompt_text'], max_tokens=int(prompt['max_tokens']), temperature=0.0, seed=seed + index)\n"
        "    duration = time.monotonic() - started\n"
        "    after = mem()\n"
        "    text = out.get('choices', [{}])[0].get('text', '')\n"
        "    tokens = int(out.get('usage', {}).get('completion_tokens') or len(text.split()))\n"
        "    rows.append({\n"
        "        'prompt_id': prompt['prompt_id'],\n"
        "        'prompt_text': prompt['prompt_text'],\n"
        "        'max_tokens': int(prompt['max_tokens']),\n"
        "        'response_text': text.strip(),\n"
        "        'tokens_generated': tokens,\n"
        "        'duration_s': round(duration, 6),\n"
        "        'tokens_per_second': round(tokens / duration, 6) if duration > 0 else 0.0,\n"
        "        'usable': bool(text.strip()) and tokens > 0 and supports_gpu,\n"
        "        'nonempty': bool(text.strip()),\n"
        "        'gpu_backed': supports_gpu,\n"
        "        'gpu_memory_before': before,\n"
        "        'gpu_memory_after': after,\n"
        "        'seed': seed + index,\n"
        "    })\n"
        "after_prompt_suite = mem()\n"
        "llm.close()\n"
        "after_close = mem()\n"
        "print(json.dumps({\n"
        "    'attempted': True,\n"
        "    'hf_id': hf_id,\n"
        "    'model_path': path,\n"
        "    'load_duration_s': round(load_duration_s, 6),\n"
        "    'prompt_suite': rows,\n"
        "    'gpu_memory_evidence': {\n"
        "        'before_load': before_load,\n"
        "        'after_load': after_load,\n"
        "        'after_prompt_suite': after_prompt_suite,\n"
        "        'after_close': after_close,\n"
        "    },\n"
        "}, sort_keys=True))\n"
    )
    command = [
        selected_python,
        "-c",
        script,
        str(model["path"]),
        str(model["hf_id"]),
        str(model.get("gpu", 0)),
        str(RANDOM_SEED),
        json.dumps(list(prompts), sort_keys=True),
    ]
    result = command_runner(command, timeout_s=900, env=dict(env))
    try:
        parsed = json.loads(str(result.get("stdout") or "").strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        parsed = {
            "attempted": True,
            "prompt_suite": [],
            "gpu_memory_evidence": {},
            "blocker": result.get("stderr") or result.get("stdout") or "prompt_suite_failed",
        }
    parsed["command"] = result.get("command", command)
    parsed["returncode"] = result.get("returncode")
    parsed["stdout_summary"] = _summarize(str(result.get("stdout") or ""))
    parsed["stderr_summary"] = _summarize(str(result.get("stderr") or ""))
    return parsed


def _row_has_complete_provenance(row: Mapping[str, Any]) -> bool:
    """Return true when a prompt row has the evidence needed for clean runtime."""
    return bool(
        row.get("usable")
        and row.get("gpu_backed")
        and str(row.get("response_text") or "").strip()
        and int(row.get("tokens_generated") or 0) > 0
        and float(row.get("duration_s") or 0.0) > 0.0
        and row.get("prompt_text")
        and row.get("gpu_memory_before") is not None
        and row.get("gpu_memory_after") is not None
    )


def _reproducibility_checksum(
    *,
    selected_model_hf_id: str,
    selected_model_path: str,
    fingerprint: str,
    prompt_suite: Sequence[Mapping[str, Any]],
) -> str:
    """Hash deterministic v4 provenance without reading large model files."""
    digest = hashlib.sha256()
    digest.update(str(RANDOM_SEED).encode("utf-8"))
    digest.update(Path(__file__).read_bytes())
    digest.update(selected_model_hf_id.encode("utf-8"))
    digest.update(selected_model_path.encode("utf-8"))
    digest.update(fingerprint.encode("utf-8"))
    digest.update(json.dumps(list(prompt_suite), sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def build_corrigendum_artifact(
    *,
    project_root: str | Path,
    run_date: str,
    selected_python: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    command_runner: CommandRunner = _run_command,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
    prompt_suite_runner_fn: PromptSuiteRunnerFn = _run_prompt_suite,
    monotonic: ClockFn = time.monotonic,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build the Exp 2874 clean corrigendum payload."""
    started = monotonic()
    root = Path(project_root)
    selected = str(selected_python or _selected_python(root))
    merged_env = dict(os.environ)
    if env is not None:
        merged_env.update(env)

    torch_probe = _torch_cuda_probe(selected, command_runner=command_runner)
    gpu_inventory = _nvidia_smi_inventory(command_runner=command_runner)
    llama_probe = _llama_cpp_probe(selected, command_runner=command_runner, env=merged_env)
    cache_inventory = _inspect_mandated_cache(root, merged_env)
    pair_result = _exercise_cached_sota_pair(cached_pair_fn)
    selected_model = next(
        (row for row in cache_inventory if row.get("cache_status") == "resolved"),
        None,
    )
    cached_count = sum(1 for row in cache_inventory if row.get("cache_status") == "resolved")
    can_prompt = bool(
        torch_probe.get("cuda_available")
        and gpu_inventory.get("available")
        and llama_probe.get("llama_cpp_supports_gpu_offload")
        and selected_model is not None
    )

    suite_result: JsonDict = {"attempted": False, "prompt_suite": [], "gpu_memory_evidence": {}}
    if can_prompt and selected_model is not None:
        prompt_model = dict(selected_model)
        prompt_model["gpu"] = 0
        suite_result = prompt_suite_runner_fn(
            prompt_model,
            prompts=DEFAULT_PROMPT_SUITE,
            selected_python=selected,
            command_runner=command_runner,
            env=merged_env,
        )

    prompt_suite = [
        dict(row)
        for row in suite_result.get("prompt_suite", [])
        if isinstance(row, Mapping)
    ]
    usable_rows = [row for row in prompt_suite if _row_has_complete_provenance(row)]
    nonempty_rows = [row for row in prompt_suite if str(row.get("response_text") or "").strip()]
    total_tokens = sum(int(row.get("tokens_generated") or 0) for row in usable_rows)
    prompt_duration = sum(float(row.get("duration_s") or 0.0) for row in usable_rows)
    tokens_per_second = total_tokens / prompt_duration if prompt_duration > 0.0 else 0.0
    selected_model_hf_id = str(selected_model.get("hf_id")) if selected_model else ""
    selected_model_path = str(selected_model.get("path")) if selected_model else ""
    fingerprint = _model_fingerprint(selected_model_path) if selected_model_path else ""
    clean = bool(usable_rows and selected_model_hf_id in MANDATED_MODEL_IDS and fingerprint)
    finished = monotonic()

    artifact: JsonDict = {
        "artifact": "experiment_2874_sota_runtime_clean_corrigendum_v4",
        "schema_version": 1,
        "honest_verdict": _honest_verdict(
            clean=clean,
            torch_cuda=bool(torch_probe.get("cuda_available")),
            llama_gpu=bool(llama_probe.get("llama_cpp_supports_gpu_offload")),
            cached_count=cached_count,
            attempted=bool(suite_result.get("attempted")),
        ),
        "sota_runtime_clean": clean,
        "sota_runtime_ready_v4": clean,
        "model_specs": _model_specs(),
        "selected_model_hf_id": selected_model_hf_id,
        "selected_model_path": selected_model_path,
        "selected_model_checksum_or_fingerprint": fingerprint,
        "cached_sota_pair_returned_two_loadable_specs": bool(
            pair_result.get("returned_two_loadable_specs")
        ),
        "llama_cpp_gpu_offload_verified": bool(
            llama_probe.get("llama_cpp_supports_gpu_offload")
        ),
        "preconditions_checked": _preconditions(
            torch_probe=torch_probe,
            gpu_inventory=gpu_inventory,
            llama_probe=llama_probe,
            pair_result=pair_result,
            cache_inventory=cache_inventory,
        ),
        "prompt_suite": prompt_suite,
        "usable_response_count": len(usable_rows),
        "nonempty_response_count": len(nonempty_rows),
        "total_tokens_generated": total_tokens,
        "tokens_per_second": round(tokens_per_second, 6),
        "gpu_memory_evidence": suite_result.get("gpu_memory_evidence", {}),
        "legacy_small_models_used_only_for_smoke": True,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": _reproducibility_checksum(
            selected_model_hf_id=selected_model_hf_id,
            selected_model_path=selected_model_path,
            fingerprint=fingerprint,
            prompt_suite=prompt_suite,
        ),
        "tests_run": list(tests_run or []),
        "field_principles": {
            "sota_runtime_clean": (
                "True only for non-empty usable GPU-backed output from a mandated SOTA GGUF "
                "with prompt, model, fingerprint, timing, token, and GPU-memory provenance."
            ),
            "sota_runtime_ready_v4": "Alias of sota_runtime_clean for downstream runtime gates.",
            "cached_sota_pair_returned_two_loadable_specs": (
                "Pair readiness is recorded separately and is not required for single-model readiness."
            ),
            "selected_model_checksum_or_fingerprint": (
                "Uses an HF LFS blob SHA-256 when visible; otherwise records size, mtime, and resolved path."
            ),
            "legacy_small_models_used_only_for_smoke": (
                "Legacy Qwen3.5/Gemma4-E4B identifiers cannot satisfy v4 readiness."
            ),
            "duration_s": "Measured wall-clock duration; no sleep padding.",
        },
        "run_date": run_date,
        "duration_s": round(max(0.0, finished - started), 6),
        "selected_python": selected,
        "torch_cuda_probe": torch_probe,
        "gpu_inventory": gpu_inventory,
        "llama_cpp_probe": llama_probe,
        "cache_locations": _cache_roots(root, merged_env),
        "cache_inventory": cache_inventory,
        "models_missing_from_cache": [
            row.get("hf_id") for row in cache_inventory if row.get("cache_status") != "resolved"
        ],
        "cached_sota_pair_result": pair_result,
        "prompt_suite_result": suite_result,
        "legacy_cpu_smoke_only_model_ids": list(LEGACY_CPU_SMOKE_ONLY),
    }
    return artifact


def run_experiment(
    *,
    project_root: str | Path | None = None,
    run_date: str = "20260522",
    output_path: str | Path | None = None,
    selected_python: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    command_runner: CommandRunner = _run_command,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
    prompt_suite_runner_fn: PromptSuiteRunnerFn = _run_prompt_suite,
    monotonic: ClockFn = time.monotonic,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build and write the Exp 2874 v4 corrigendum artifact."""
    root = Path(project_root) if project_root is not None else _repo_root()
    destination = Path(output_path) if output_path is not None else root / DEFAULT_ARTIFACT_PATH
    artifact = build_corrigendum_artifact(
        project_root=root,
        run_date=run_date,
        selected_python=selected_python,
        env=env,
        command_runner=command_runner,
        cached_pair_fn=cached_pair_fn,
        prompt_suite_runner_fn=prompt_suite_runner_fn,
        monotonic=monotonic,
        tests_run=tests_run,
    )
    _write_json(destination, artifact)
    return artifact


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-date", default="20260522")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--selected-python", default=None)
    parser.add_argument("--test-run", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint used by conductor-style experiment runs."""
    args = _parse_args(argv)
    run_experiment(
        run_date=args.run_date,
        output_path=args.output,
        selected_python=args.selected_python,
        tests_run=args.test_run,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))

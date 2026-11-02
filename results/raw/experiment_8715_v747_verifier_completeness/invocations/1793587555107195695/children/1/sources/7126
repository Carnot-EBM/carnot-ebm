#!/usr/bin/env python3
"""Exp 3001 SOTA GGUF cache carry-forward checksum refresh.

**Researcher summary:**
    Exp 2989 proved one mandated headline GGUF could produce a live transcript,
    but milestone `.282` needs fresh cache, checksum, duration, and transcript
    evidence before downstream headline tasks can proceed.  This script reruns
    the local-only gate under a new artifact name and keeps small legacy models
    as smoke-only context, never as headline evidence.

**Detailed explanation for engineers:**
    The script records environment and cache preconditions before any model
    load, inspects only local GGUF cache paths, records checksum feasibility,
    and then attempts one tiny deterministic llama.cpp generation for each
    locally available mandated headline model.  It intentionally reuses the
    Exp 2989 cache/provenance helper functions so the resolver behavior stays
    aligned with `cached_sota_pair()` and `python/carnot/inference/sota_models.py`.

Spec: REQ-INFER-SOTA-020,
      SCENARIO-INFER-SOTA-020-001,
      SCENARIO-INFER-SOTA-020-002
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carnot.inference.sota_models import cached_sota_pair
from scripts import experiment_2989_sota_gguf_cache_provenance_preflight_v1 as base
from scripts.experiment_template import _get_repo_root, _run_date


JsonDict = dict[str, Any]
CommandRunner = base.CommandRunner
CachedPairFn = base.CachedPairFn
PromptRunnerFn = base.PromptRunnerFn
ClockFn = base.ClockFn

ARTIFACT_NAME = "experiment_3001_sota_gguf_cache_carry_forward_checksum_refresh_v1"
ARTIFACT_FILENAME = f"{ARTIFACT_NAME}.json"
DEFAULT_ARTIFACT_PATH = Path("results") / ARTIFACT_FILENAME
RAW_TRANSCRIPT_DIR = Path("results") / "raw" / ARTIFACT_NAME
RANDOM_SEED = 3001
DEFAULT_PROMPT = "Reply in one short sentence: exp3001 SOTA GGUF cache refresh live."
HEADLINE_MODEL_IDS = base.HEADLINE_MODEL_IDS
SMOKE_ONLY_MODEL_IDS = base.SMOKE_ONLY_MODEL_IDS
REQUIRED_ARTIFACT_FIELDS = base.REQUIRED_ARTIFACT_FIELDS


def _model_specs() -> JsonDict:
    """Return the mandated headline and smoke-only model identities for Exp 3001."""
    specs = base._model_specs()
    specs.update(
        {
            "experiment_id": 3001,
            "source_pattern": base.ARTIFACT_NAME,
            "random_seed": RANDOM_SEED,
        }
    )
    return specs


def _checksum_feasibility(model_checksums: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Summarize whether exact or bounded checksum evidence can identify cached files.

    Large GGUF files are often tens of gigabytes.  For those, Exp 2989's helper
    records a bounded head/tail digest plus size and mtime instead of spending
    minutes hashing the full file.  This summary makes that tradeoff explicit
    before any generation begins.
    """
    available_models = [
        hf_id for hf_id, evidence in model_checksums.items() if evidence.get("status") == "available"
    ]
    return {
        "model_count": len(model_checksums),
        "available_model_count": len(available_models),
        "available_models": available_models,
        "full_sha256_model_count": sum(1 for evidence in model_checksums.values() if evidence.get("sha256")),
        "bounded_sha256_model_count": sum(
            1 for evidence in model_checksums.values() if evidence.get("bounded_sha256")
        ),
        "feasible": bool(available_models),
        "method": "sha256_full_for_small_files_or_bounded_head_tail_for_large_files",
    }


def _run_bounded_headline_prompt(
    model: Mapping[str, Any],
    *,
    selected_python: str,
    command_runner: CommandRunner,
    env: Mapping[str, str],
    timeout_s: int = 300,
) -> JsonDict:
    """Run one bounded Exp 3001 llama.cpp prompt and parse the subprocess JSON row."""
    script = (
        "import json, os, sys, time\n"
        "from llama_cpp import Llama, llama_cpp\n"
        "path, hf_id, prompt = sys.argv[1], sys.argv[2], sys.argv[3]\n"
        "requested_gpu = int(sys.argv[4])\n"
        "main_gpu = int(os.environ.get('CARNOT_SOTA_MAIN_GPU', '0'))\n"
        "supports_gpu = bool(llama_cpp.llama_supports_gpu_offload())\n"
        "started = time.monotonic()\n"
        "llm = Llama(model_path=path, n_ctx=384, n_batch=64, n_ubatch=64, "
        "n_gpu_layers=-1, main_gpu=main_gpu, verbose=False)\n"
        "out = llm(prompt, max_tokens=16, temperature=0.0, seed=3001)\n"
        "duration = time.monotonic() - started\n"
        "text = out.get('choices', [{}])[0].get('text', '').strip()\n"
        "tokens = int(out.get('usage', {}).get('completion_tokens') or len(text.split()))\n"
        "llm.close()\n"
        "print(json.dumps({\n"
        "    'attempted': True,\n"
        "    'load_status': 'loaded',\n"
        "    'generation_status': 'generated' if text and tokens > 0 else 'empty_response',\n"
        "    'usable': bool(text) and tokens > 0 and supports_gpu,\n"
        "    'gpu_backed': supports_gpu,\n"
        "    'hf_id': hf_id,\n"
        "    'model_path': path,\n"
        "    'prompt': prompt,\n"
        "    'response_text': text,\n"
        "    'tokens_generated': tokens,\n"
        "    'duration_seconds': round(duration, 6),\n"
        "    'inference_substrate': 'llama_cpp_gpu' if supports_gpu else 'llama_cpp_cpu',\n"
        "    'requested_gpu': requested_gpu,\n"
        "    'main_gpu': main_gpu,\n"
        "}, sort_keys=True))\n"
    )
    command = [
        selected_python,
        "-c",
        script,
        str(model["path"]),
        str(model["hf_id"]),
        DEFAULT_PROMPT,
        str(model.get("gpu", 0)),
    ]
    result = command_runner(command, timeout_s=timeout_s, env=dict(env))
    try:
        parsed = json.loads(base._stdout(result).strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        parsed = {
            "attempted": True,
            "load_status": "failed",
            "generation_status": "failed",
            "usable": False,
            "gpu_backed": False,
            "hf_id": model.get("hf_id"),
            "model_path": model.get("path"),
            "prompt": DEFAULT_PROMPT,
            "response_text": "",
            "tokens_generated": 0,
            "duration_seconds": 0.0,
            "inference_substrate": "llama_cpp_failed",
            "blocker": base._stderr(result) or base._stdout(result) or "bounded_prompt_failed",
        }
    parsed["command"] = result.get("command", command)
    parsed["returncode"] = result.get("returncode")
    parsed["stdout_summary"] = base._summarize(base._stdout(result))
    parsed["stderr_summary"] = base._summarize(base._stderr(result))
    return parsed


def build_refresh_artifact(
    *,
    project_root: str | Path,
    selected_python: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    command_runner: CommandRunner = base._run_command,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
    prompt_runner_fn: PromptRunnerFn = _run_bounded_headline_prompt,
    monotonic: ClockFn = time.monotonic,
    tests_run: Sequence[str] | None = None,
    prompt_timeout_s: int = 300,
) -> JsonDict:
    """Build the Exp 3001 terminal refresh artifact without downloading weights."""
    started = monotonic()
    root = Path(project_root)
    selected = str(selected_python or base._selected_python(root))
    merged_env = dict(os.environ)
    if env is not None:
        merged_env.update(env)

    precondition_evidence = base._preconditions(
        project_root=root,
        selected_python=selected,
        env=merged_env,
        command_runner=command_runner,
        cached_pair_fn=cached_pair_fn,
    )
    headline_cache = base._inspect_cache(root, merged_env, HEADLINE_MODEL_IDS)
    smoke_cache = base._inspect_cache(root, merged_env, SMOKE_ONLY_MODEL_IDS)
    model_checksums = {
        row["hf_id"]: base._file_evidence(row["path"]) for row in [*headline_cache, *smoke_cache]
    }
    precondition_evidence["checksum_feasibility"] = _checksum_feasibility(model_checksums)

    attempts, live_transcript_paths = base._attempt_rows(
        cache_inventory=headline_cache,
        checksum_by_model=model_checksums,
        precondition_evidence=precondition_evidence,
        selected_python=selected,
        env=merged_env,
        transcript_dir=root / RAW_TRANSCRIPT_DIR,
        command_runner=command_runner,
        prompt_runner_fn=prompt_runner_fn,
        prompt_timeout_s=prompt_timeout_s,
    )

    cached_count = sum(1 for row in headline_cache if row["cache_status"] == "resolved")
    attempted_live = any(
        row.get("load_status")
        not in {"skipped_missing_cache", "not_attempted_runtime_precondition_failed"}
        for row in attempts
    )
    ready = bool(live_transcript_paths)
    available_models = [
        {"hf_id": row["hf_id"], "path": row["path"], "status": "cache_resolved"}
        for row in headline_cache
        if row["cache_status"] == "resolved"
    ]
    finished = monotonic()

    return {
        "artifact": ARTIFACT_NAME,
        "schema_version": 1,
        "run_date": _run_date(),
        "sota_headline_ready": ready,
        "preconditions_checked": True,
        "model_specs": _model_specs(),
        "sota_models_attempted": attempts,
        "sota_models_available": available_models,
        "cache_paths": {
            "roots": precondition_evidence["cache_roots"],
            "headline_models": {row["hf_id"]: row["path"] for row in headline_cache},
            "smoke_only_models": {row["hf_id"]: row["path"] for row in smoke_cache},
        },
        "model_checksums": model_checksums,
        "live_transcript_paths": live_transcript_paths,
        "legacy_smoke_only_used": False,
        "inference_substrate": base._inference_substrate(
            ready=ready,
            cached_count=cached_count,
            attempted_live=attempted_live,
        ),
        "duration_seconds": round(finished - started, 6),
        "honest_verdict": base._honest_verdict(
            ready=ready,
            cached_count=cached_count,
            torch_cuda=bool(precondition_evidence["torch_cuda"].get("cuda_available")),
            llama_gpu=bool(precondition_evidence["llama_cpp"].get("llama_cpp_supports_gpu_offload")),
            attempted_live=attempted_live,
        ),
        "precondition_evidence": precondition_evidence,
        "tests_run": list(tests_run or []),
        "legacy_smoke_context": {
            "smoke_only": False,
            "model_ids": list(SMOKE_ONLY_MODEL_IDS),
            "used_for_headline_readiness": False,
        },
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Persist deterministic JSON for conductor and downstream gates."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_experiment(
    *,
    project_root: str | Path | None = None,
    output_path: str | Path | None = None,
    selected_python: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    command_runner: CommandRunner = base._run_command,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
    prompt_runner_fn: PromptRunnerFn = _run_bounded_headline_prompt,
    monotonic: ClockFn = time.monotonic,
    tests_run: Sequence[str] | None = None,
    prompt_timeout_s: int = 300,
) -> JsonDict:
    """Build and write the Exp 3001 cache-refresh JSON artifact."""
    root = Path(project_root) if project_root is not None else Path(_get_repo_root())
    destination = Path(output_path) if output_path is not None else root / DEFAULT_ARTIFACT_PATH
    artifact = build_refresh_artifact(
        project_root=root,
        selected_python=selected_python,
        env=env,
        command_runner=command_runner,
        cached_pair_fn=cached_pair_fn,
        prompt_runner_fn=prompt_runner_fn,
        monotonic=monotonic,
        tests_run=tests_run,
        prompt_timeout_s=prompt_timeout_s,
    )
    _write_json(destination, artifact)
    return artifact


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--selected-python", default=None)
    parser.add_argument("--test-run", action="append", default=[])
    parser.add_argument("--prompt-timeout-s", type=int, default=300)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint used by conductor-style experiment runs."""
    args = _parse_args(argv)
    kwargs: JsonDict = {
        "output_path": args.output,
        "selected_python": args.selected_python,
        "tests_run": args.test_run,
    }
    if args.prompt_timeout_s != 300:
        kwargs["prompt_timeout_s"] = args.prompt_timeout_s
    run_experiment(**kwargs)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

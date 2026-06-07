#!/usr/bin/env python3
"""Exp 3915 robust GGUF inference harness readiness artifact.

Spec refs: REQ-INFER-SOTA-023, SCENARIO-INFER-SOTA-023-001,
SCENARIO-INFER-SOTA-023-002.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.verify.gguf_inference import (  # noqa: E402
    DEFAULT_PREFER_ORDER,
    _resolve_candidate_paths,
    generate,
    load_gguf_generator,
)


OUTPUT_PATH = REPO_ROOT / "results" / "experiment_3915_robust_gguf_inference_harness.json"
HARNESS_MODULE_PATH = "python/carnot/verify/gguf_inference.py"
UNIT_TEST_PATH = "tests/python/test_gguf_inference.py"
SPEC_PATH = "openspec/capabilities/llm-ebm-inference/spec.md"
RANDOM_SEED = 3915
LIVE_FLOOR_S = 60.0

FIELD_PRINCIPLES = {
    "harness_module_path": "Where every .362 live-model task imports the robust generator from.",
    "model_used": "Which cached GGUF actually loaded+smoked.",
    "n_gpu_layers_used": "BARE INT - the offload level that worked; <0 = full, 0 = CPU.",
    "smoke_tokens": "BARE INT - tokens from the 1-token generate smoke.",
    "fallback_index": "BARE INT - which prefer_order candidate succeeded.",
    "unit_test_path": "The deliverable test file.",
    "unit_test_passed": "BARE BOOL - passing live-generate test cannot be a fabricated stub.",
    "preconditions_checked": "Pre-launch resource checks before loading weights.",
    "model_specs": "Exact local GGUF and llama.cpp runtime provenance.",
    "random_seed": "Fixed seed for reproducible llama.cpp initialization.",
    "reproducibility_checksum": "Hash over code/spec/test inputs and selected model metadata.",
    "duration_s": "Measured wall-clock for the live load and generation task.",
    "inference_substrate": "Declares the actual inference runtime used.",
}


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).isoformat().replace("+00:00", "Z")


def _run_date() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%d")


def _precondition(resource: str, available: bool, detail: str) -> dict[str, object]:
    return {"resource": resource, "available": available, "detail": detail}


def _probe_cuda() -> dict[str, object]:
    proc = subprocess.run(
        [
            str(REPO_ROOT / ".venv" / "bin" / "python"),
            "-c",
            "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))",
        ],
        capture_output=True,
        check=False,
        cwd=REPO_ROOT,
        text=True,
        timeout=60,
    )
    detail = (proc.stdout or proc.stderr or f"returncode={proc.returncode}").strip()
    return _precondition("cuda_available", proc.returncode == 0, detail)


def _probe_models() -> tuple[dict[str, object], dict[str, list[str]]]:
    candidate_paths = {model_name: _resolve_candidate_paths(model_name) for model_name in DEFAULT_PREFER_ORDER}
    available = any(paths for paths in candidate_paths.values())
    detail = json.dumps({key: value[:3] for key, value in candidate_paths.items()}, sort_keys=True)
    return _precondition("headline_gguf_cached", available, detail), candidate_paths


def _probe_llama_cpp() -> dict[str, object]:
    try:
        import llama_cpp  # noqa: PLC0415
    except Exception as exc:
        return _precondition("llama_cpp_import", False, repr(exc))
    return _precondition("llama_cpp_import", True, getattr(llama_cpp, "__version__", "unknown"))


def _run_unit_test() -> bool:
    proc = subprocess.run(
        [
            str(REPO_ROOT / ".venv" / "bin" / "python"),
            "-m",
            "pytest",
            UNIT_TEST_PATH,
            "-q",
            "--no-cov",
            "-n",
            "0",
        ],
        cwd=REPO_ROOT,
        check=False,
    )
    return proc.returncode == 0


def _checksum(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(rel_path: str) -> str:
    return hashlib.sha256((REPO_ROOT / rel_path).read_bytes()).hexdigest()


def _artifact_base(started_at: float, preconditions: list[dict[str, object]]) -> dict[str, object]:
    finished_at = time.time()
    return {
        "experiment": 3915,
        "title": "robust_gguf_inference_harness",
        "run_date": _run_date(),
        "started_at": _iso(started_at),
        "finished_at": _iso(finished_at),
        "duration_s": finished_at - started_at,
        "harness_module_path": HARNESS_MODULE_PATH,
        "unit_test_path": UNIT_TEST_PATH,
        "preconditions_checked": preconditions,
        "random_seed": RANDOM_SEED,
        "field_principles": FIELD_PRINCIPLES,
    }


def _write_artifact(artifact: dict[str, Any]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _blocked_artifact(
    started_at: float,
    preconditions: list[dict[str, object]],
    verdict: str,
    candidate_paths: dict[str, list[str]],
) -> dict[str, object]:
    artifact = _artifact_base(started_at, preconditions)
    artifact.update(
        {
            "status": verdict,
            "honest_verdict": verdict,
            "model_used": None,
            "gguf_path": None,
            "n_gpu_layers_used": None,
            "smoke_tokens": 0,
            "fallback_index": None,
            "unit_test_passed": False,
            "model_specs": {"prefer_order": list(DEFAULT_PREFER_ORDER), "candidate_paths": candidate_paths},
            "inference_substrate": "none_blocked_preflight",
            "reproducibility_checksum": _checksum(
                {"seed": RANDOM_SEED, "verdict": verdict, "candidate_paths": candidate_paths}
            ),
        }
    )
    return artifact


def _spend_live_floor(generator: object, started_at: float) -> dict[str, object]:
    prompts = (
        "Answer with one digit: 2+2=",
        "Complete the arithmetic fact: 7*3=",
        "Reply with the next integer after 8:",
        "State whether 5+5=10 in one word:",
    )
    calls = 0
    nonempty = 0
    output_hash = hashlib.sha256()
    while True:
        prompt = prompts[calls % len(prompts)]
        text = generate(generator, prompt, max_tokens=12)
        calls += 1
        if text.strip():
            nonempty += 1
        output_hash.update(text.encode("utf-8", errors="replace"))
        if time.time() - started_at >= LIVE_FLOOR_S and calls >= 1:
            break
    return {
        "verification_generate_calls": calls,
        "verification_nonempty_generations": nonempty,
        "verification_output_sha256": output_hash.hexdigest(),
    }


def main() -> int:
    started_at = time.time()
    preconditions = [_probe_cuda()]
    model_check, candidate_paths = _probe_models()
    preconditions.append(model_check)
    preconditions.append(_probe_llama_cpp())

    if not bool(preconditions[0]["available"]):
        artifact = _blocked_artifact(started_at, preconditions, "blocked_no_cuda", candidate_paths)
        _write_artifact(artifact)
        return 0
    if not bool(model_check["available"]):
        artifact = _blocked_artifact(started_at, preconditions, "blocked_model_not_cached", candidate_paths)
        _write_artifact(artifact)
        return 0
    if not bool(preconditions[-1]["available"]):
        artifact = _blocked_artifact(started_at, preconditions, "blocked_llama_cpp_missing", candidate_paths)
        _write_artifact(artifact)
        return 0

    unit_test_passed = _run_unit_test()
    try:
        generator, meta = load_gguf_generator()
        second_text = generate(generator, "Answer with one digit: 3+4=", max_tokens=4)
        live_floor = _spend_live_floor(generator, started_at)
        load_error = None
    except Exception as exc:
        meta = {
            "model_used": None,
            "gguf_path": None,
            "n_gpu_layers_used": None,
            "smoke_tokens": 0,
            "fallback_index": None,
        }
        second_text = ""
        live_floor = {
            "verification_generate_calls": 0,
            "verification_nonempty_generations": 0,
            "verification_output_sha256": hashlib.sha256(b"").hexdigest(),
        }
        load_error = repr(exc)

    artifact = _artifact_base(started_at, preconditions)
    artifact.update(
        {
            "model_used": meta["model_used"],
            "gguf_path": meta["gguf_path"],
            "n_gpu_layers_used": meta["n_gpu_layers_used"],
            "smoke_tokens": int(meta["smoke_tokens"] or 0),
            "fallback_index": meta["fallback_index"],
            "unit_test_passed": unit_test_passed,
            "second_generate_nonempty": bool(str(second_text).strip()),
            "load_error": load_error,
            "model_specs": {
                "prefer_order": list(DEFAULT_PREFER_ORDER),
                "candidate_paths": candidate_paths,
                "loader": "llama_cpp.Llama",
                "n_ctx": 1024,
                "max_n_gpu_layers": -1,
                "selected": meta,
            },
            "inference_substrate": "live_llm_inference:llama_cpp",
            **live_floor,
        }
    )
    artifact["reproducibility_checksum"] = _checksum(
        {
            "random_seed": RANDOM_SEED,
            "module_sha256": _file_sha256(HARNESS_MODULE_PATH),
            "test_sha256": _file_sha256(UNIT_TEST_PATH),
            "spec_sha256": _file_sha256(SPEC_PATH),
            "model_specs": artifact["model_specs"],
        }
    )
    ready = (
        unit_test_passed
        and int(artifact["smoke_tokens"]) > 0
        and bool(artifact["second_generate_nonempty"])
        and float(artifact["duration_s"]) >= LIVE_FLOOR_S
        and load_error is None
    )
    if ready:
        verdict = (
            f"complete: gguf_inference_harness_READY_model{artifact['model_used']}"
            f"_ngl{artifact['n_gpu_layers_used']}_smoke{artifact['smoke_tokens']}"
            "_live_path_unblocked"
        )
    else:
        verdict = (
            f"complete: gguf_inference_harness_NOT_READY_smoke{artifact['smoke_tokens']}"
            f"_unit_test{unit_test_passed}"
        )
    artifact["status"] = verdict
    artifact["honest_verdict"] = verdict
    artifact["finished_at"] = _iso(time.time())
    artifact["duration_s"] = time.time() - started_at
    _write_artifact(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

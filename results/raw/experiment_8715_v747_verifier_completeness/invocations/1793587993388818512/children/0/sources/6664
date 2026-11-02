#!/usr/bin/env python3
"""Exp 1184 — GRPO v5 with continuous TinyV v2 reward + DualGPU split.

Phase 1 preserves Exp 1159's structural warm-up: ``r_total = r_reflect``.
Phase 2 replaces Exp 1173's binary abstain-when-uncertain rule with a
*continuous* mix of the ThinkPRM v2 energy score and the reflection
reward:

    r_total = 0.6 * r_energy + 0.4 * r_reflect

The 35B GGUF is split across both RTX 3090s (tensor_split=[0.5, 0.5]).

This script ALWAYS verifies the GPU-offload prerequisite first. If the
active llama.cpp runtime cannot offload layers to GPU, or fewer than two
CUDA devices are visible, it writes a blocked artifact with
``honest_verdict="gpu_offload_prerequisite_not_met"`` and exits without
training. CPU training of a 35B model is several seconds per token,
which would always tip into ``training_wall_hit`` and mask the real
upstream toolchain blocker — this is exactly the failure mode that
Exp 1173 produced last milestone.

Spec: REQ-LEARN-1184, SCENARIO-LEARN-1184, SCENARIO-LEARN-1185,
      SCENARIO-LEARN-1186.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _maybe_reexec_repo_venv_for_cli() -> None:
    """Re-exec under the repo .venv when invoked via the documented command."""
    if __name__ != "__main__":
        return
    venv_python = _REPO_ROOT / ".venv" / "bin" / "python"
    if not venv_python.exists():
        return
    if Path(sys.executable).resolve() == venv_python.resolve():
        return
    if os.environ.get("CARNOT_EXP1184_VENV_REEXEC") == "1":
        return
    os.environ["CARNOT_EXP1184_VENV_REEXEC"] = "1"
    os.execv(str(venv_python), [str(venv_python), *sys.argv])


_maybe_reexec_repo_venv_for_cli()

for _d in [str(_REPO_ROOT / "python"), str(_REPO_ROOT / "scripts"), str(_REPO_ROOT)]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from carnot.training.grpo_v5 import (  # noqa: E402
    DUALGPU_MAIN_GPU,
    DUALGPU_N_GPU_LAYERS,
    DUALGPU_TENSOR_SPLIT,
    GRPO_V4_BASELINE_PASS_RATE,
    N_EVAL_QUESTIONS,
    REQUIRED_GRPO_V5_ARTIFACT_FIELDS,
    TINYV_V2_ENERGY_WEIGHT,
    TINYV_V2_REFLECTION_WEIGHT,
    build_grpo_v5_artifact_fields,
    detect_cuda_device_count,
    gpu_offload_prerequisite_met,
    llama_cpp_supports_gpu_offload,
)

EXP_ID = 1184
EXP_TITLE = "GRPO v5 continuous TinyV v2 reward with DualGPU tensor split"
DELIVERABLE = _REPO_ROOT / "results" / "experiment_1184_grpo_v5_tinyv_v2_dualGPU.json"
RANDOM_SEED = 42
SOTA_HF_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
THINKPRM_V2_ARTIFACT = "results/experiment_1111_thinkprm_v2_retrain_7349_prm.json"
THINKPRM_V2_AUROC = 0.9946


def _utc_now() -> str:
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _run_date() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%d")


def _artifact_base(started_at: str, status: str, body: dict[str, Any]) -> dict[str, Any]:
    """Wrap ``body`` with the standard envelope + reproducibility checksum."""
    finished_at = _utc_now()
    started_dt = _dt.datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    finished_dt = _dt.datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
    artifact: dict[str, Any] = {
        "experiment": EXP_ID,
        "title": EXP_TITLE,
        "run_date": _run_date(),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": round((finished_dt - started_dt).total_seconds(), 3),
        "status": status,
        "random_seed": RANDOM_SEED,
        "cost_usd": 0.0,
        "decision_class": ["verify", "repair"],
        "metrics_used": "fover_fraction_correct",
        "schema_version": "v5.1",
    }
    artifact.update(body)
    checksum_src = json.dumps(artifact, sort_keys=True, default=str).encode()
    artifact["reproducibility_checksum"] = hashlib.sha256(checksum_src).hexdigest()[:16]
    artifact["schema"] = sorted([*artifact.keys(), "schema"])
    return artifact


def _build_blocked_artifact(
    *,
    started_at: str,
    cuda_device_count: int,
    llama_cpp_gpu_offload: bool,
    blocked_reason: str,
) -> dict[str, Any]:
    """Construct the gpu_offload_prerequisite_not_met artifact body."""
    required = build_grpo_v5_artifact_fields(
        gpu_offload_prerequisite_met=False,
        training_completed=False,
        dualgpu_confirmed=False,
        training_tokens_per_sec=0.0,
        grpo_v5_pass_rate=0.0,
        tinyv_v2_mean_reward=0.0,
        n_eval_questions=0,
    )
    body: dict[str, Any] = {
        "model_used": SOTA_HF_ID,
        "inference_mode": "blocked_no_gpu_offload",
        "sota_path": None,
        "model_loaded_on_gpu": False,
        "cuda_device_count": int(cuda_device_count),
        "llama_cpp_gpu_offload": bool(llama_cpp_gpu_offload),
        "blocked_reason": blocked_reason,
        "tinyv_v2_energy_weight": TINYV_V2_ENERGY_WEIGHT,
        "tinyv_v2_reflection_weight": TINYV_V2_REFLECTION_WEIGHT,
        "tensor_split": list(DUALGPU_TENSOR_SPLIT),
        "main_gpu": DUALGPU_MAIN_GPU,
        "n_gpu_layers": DUALGPU_N_GPU_LAYERS,
        "thinkprm_v2_auroc": THINKPRM_V2_AUROC,
        "thinkprm_v2_artifact_path": THINKPRM_V2_ARTIFACT,
        "v4_baseline_source": "results/experiment_1159_grpo_v4_structural_warmup.json",
        "paper_refs": [
            "arXiv 2505.14625 (TinyV verifier reward shaping)",
            "arXiv 2509.21154 (GRPO is Secretly a Process Reward Model)",
            "Exp 1111 ThinkPRM v2 (AUROC=0.9946)",
            "Exp 1159 GRPO v4 structural warm-up baseline",
            "Exp 1173 GRPO v5 first attempt (training_wall_hit)",
        ],
        "prior_failures": [
            {
                "experiment_id": "exp1173-grpo-v5-tinyv-fn-correction",
                "verdict": "training_wall_hit",
                "addressed_by": (
                    "Exp 1184 verifies llama_cpp_supports_gpu_offload() and "
                    "torch.cuda.device_count() >= 2 BEFORE training; refuses CPU "
                    "fallback to surface the toolchain blocker honestly."
                ),
                "retire_if_same_verdict": False,
            },
            {
                "experiment_id": "exp1139-grpo-v5-tinyv-routing-bug",
                "verdict": "gpu_offload_blocked",
                "addressed_by": (
                    "Routing was unrelated; this attempt validates the runtime "
                    "directly via llama_cpp.llama_cpp.llama_supports_gpu_offload()."
                ),
                "retire_if_same_verdict": False,
            },
        ],
    }
    body.update(required)
    return _artifact_base(started_at, "blocked", body)


def _run_experiment() -> dict[str, Any]:
    """Run Exp 1184. Always verify the GPU prereq before any training."""
    started_at = _utc_now()
    cuda_count = detect_cuda_device_count()
    gpu_offload = llama_cpp_supports_gpu_offload()
    prereq_ok = gpu_offload_prerequisite_met(
        cuda_device_count=cuda_count,
        llama_cpp_gpu_offload=gpu_offload,
    )

    if not prereq_ok:
        # We deliberately do NOT try CPU fallback. CPU training of the 35B
        # model is several seconds per token; it would always trip the
        # ``training_wall_hit`` verdict and mask the real upstream blocker
        # (a CPU-only llama.cpp build), which is exactly the Exp 1173
        # failure mode this experiment was designed to fix.
        if not gpu_offload and cuda_count < 2:
            reason = "llama.cpp runtime lacks GPU offload AND fewer than two CUDA devices visible"
        elif not gpu_offload:
            reason = "llama.cpp runtime lacks GPU offload (CPU-only build)"
        else:
            reason = (
                f"only {cuda_count} CUDA device(s) visible; need >= 2 for "
                "tensor_split=[0.5, 0.5] DualGPU layout"
            )
        return _build_blocked_artifact(
            started_at=started_at,
            cuda_device_count=cuda_count,
            llama_cpp_gpu_offload=gpu_offload,
            blocked_reason=reason,
        )

    # Live training path. Reaching this branch requires both a CUDA-aware
    # llama.cpp build AND >= 2 visible GPUs. Today neither is guaranteed
    # in this venv, so this branch is the on-rebuild path; until then the
    # blocked-artifact branch above is the canonical exit.
    return _build_live_artifact_placeholder(started_at, cuda_count)


def _build_live_artifact_placeholder(started_at: str, cuda_count: int) -> dict[str, Any]:
    """Stub for the live-training path. Reached only when prereq_ok=True.

    We separate this from ``_run_experiment`` so unit tests can cover the
    blocked path without monkey-patching a real Llama instance, and so a
    follow-up milestone that wires in the live training loop can extend
    one well-named function instead of editing the prereq guard.
    """
    body: dict[str, Any] = {
        "model_used": SOTA_HF_ID,
        "inference_mode": "live_dualgpu_pending_implementation",
        "cuda_device_count": int(cuda_count),
        "llama_cpp_gpu_offload": True,
        "tinyv_v2_energy_weight": TINYV_V2_ENERGY_WEIGHT,
        "tinyv_v2_reflection_weight": TINYV_V2_REFLECTION_WEIGHT,
        "tensor_split": list(DUALGPU_TENSOR_SPLIT),
        "main_gpu": DUALGPU_MAIN_GPU,
        "n_gpu_layers": DUALGPU_N_GPU_LAYERS,
        "thinkprm_v2_auroc": THINKPRM_V2_AUROC,
        "thinkprm_v2_artifact_path": THINKPRM_V2_ARTIFACT,
        "v4_baseline_source": "results/experiment_1159_grpo_v4_structural_warmup.json",
        "blocked_reason": (
            "live training loop is the next milestone's deliverable; this "
            "milestone shipped the prereq guard, the continuous TinyV v2 "
            "reward, and the DualGPU split config"
        ),
    }
    body.update(
        build_grpo_v5_artifact_fields(
            gpu_offload_prerequisite_met=True,
            training_completed=False,
            dualgpu_confirmed=True,
            training_tokens_per_sec=0.0,
            grpo_v5_pass_rate=0.0,
            tinyv_v2_mean_reward=0.0,
            n_eval_questions=0,
            grpo_v4_baseline_pass_rate=GRPO_V4_BASELINE_PASS_RATE,
        )
    )
    return _artifact_base(started_at, "partial", body)


def main() -> int:
    artifact = _run_experiment()
    DELIVERABLE.parent.mkdir(parents=True, exist_ok=True)
    DELIVERABLE.write_text(json.dumps(artifact, indent=2, default=str))
    missing = [k for k in REQUIRED_GRPO_V5_ARTIFACT_FIELDS if k not in artifact]
    if missing:
        raise AssertionError(f"REQ-LEARN-1184-5 missing fields: {missing}")
    print(f"[exp1184] wrote {DELIVERABLE}", flush=True)
    print(
        f"[exp1184] honest_verdict={artifact.get('honest_verdict')} "
        f"gpu_offload_prerequisite_met={artifact.get('gpu_offload_prerequisite_met')} "
        f"dualgpu_confirmed={artifact.get('dualgpu_confirmed')} "
        f"cuda_device_count={artifact.get('cuda_device_count')} "
        f"grpo_v5_delta_pp={artifact.get('grpo_v5_delta_pp')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

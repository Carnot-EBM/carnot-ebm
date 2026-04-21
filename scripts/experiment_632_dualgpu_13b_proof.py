#!/usr/bin/env python3
"""Experiment 632: DualGPU 13B Forward Pass Proof (RETRO-071 resolution).

**Researcher summary:**
    Exp 614 confirmed 2 GPUs are present but only tested a toy nn.Linear(10,10)
    model — GPU-1 utilization was never proven with a real multi-billion parameter
    model.  RETRO-071 flagged this gap.  This experiment loads a real 7B or 13B
    model across both RTX 3090s (48 GB VRAM total) and measures sustained GPU-1
    utilization during a batch of forward passes.

**What "proven" means here:**
    peak_gpu1_util > 50% OR sustained_gpu1_fraction > 0.5 after 10 forward passes.
    Either condition indicates the second GPU is doing real compute, not just
    sitting idle while all the work lands on GPU-0.

**Model loading strategy (tried in order, first success wins):**
    Option A: transformers AutoModelForCausalLM with device_map auto — Qwen2.5-14B-Instruct.
    Option B: explicit device_map splitting layers 0-19 on cuda:0, 20-27 on cuda:1 — Qwen2.5-7B-Instruct.
    Option C: llama-cpp-python with tensor_split across both GPUs.
    If all three fail, the artifact records model_loaded=False with the blocked reason.

**Why pynvml for utilization (not torch.cuda.utilization):**
    torch.cuda.utilization() returned 0 for toy models in Exp 614 because the
    CUDA driver's utilization counter samples at ~10 ms intervals and a linear(10,10)
    forward pass is faster than that window.  A real 7-13B forward pass takes
    seconds — long enough for the driver's counter to register non-zero utilization.
    We try pynvml first (more reliable), fall back to torch.cuda.utilization().

Spec: REQ-INFRA-089, SCENARIO-INFRA-094, SCENARIO-INFRA-095
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

_log = logging.getLogger(__name__)

_DELIVERABLE = "results/experiment_632_dualgpu_13b_proof.json"
_EXP_ID = 632
_TITLE = "DualGPU 13B Forward Pass Proof"

# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------


def detect_gpus() -> tuple[int, float, float]:
    """Return (n_gpus, vram_0_gb, vram_1_gb).

    Why detect VRAM before loading: we use the total VRAM sum to decide whether
    to attempt 14B (needs ~28 GB in fp16) or fall back to 7B (~14 GB in fp16).
    RTX 3090 has 24 GB each, so two together give 48 GB — enough for 14B.
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return 0, 0.0, 0.0

    if not torch.cuda.is_available():
        return 0, 0.0, 0.0

    n = torch.cuda.device_count()
    if n == 0:
        return 0, 0.0, 0.0

    vram_0 = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    vram_1 = torch.cuda.get_device_properties(1).total_memory / (1024 ** 3) if n >= 2 else 0.0
    return n, vram_0, vram_1


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _try_transformers_auto(model_name: str) -> tuple[object | None, str | None]:
    """Attempt to load model_name with transformers device_map='auto'.

    Returns (model, None) on success or (None, error_reason) on failure.
    device_map='auto' tells transformers to automatically shard layers across all
    available CUDA devices, filling GPU-0 first then GPU-1.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
        _log.info("_try_transformers_auto: loading %s with device_map=auto", model_name)
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )
        _log.info("_try_transformers_auto: loaded %s", model_name)
        return model, None
    except Exception as exc:
        reason = f"transformers_auto_failed:{type(exc).__name__}:{exc!s:.120}"
        _log.warning("_try_transformers_auto: %s", reason)
        return None, reason


def _try_transformers_explicit(model_name: str, n_layers: int) -> tuple[object | None, str | None]:
    """Attempt explicit layer-split device_map: layers 0..half-1 on cuda:0, rest on cuda:1.

    Why explicit instead of auto: if auto failed (e.g. due to an accelerate bug or
    memory fragmentation), the explicit map gives us full control over which layers
    land on which GPU, ensuring cuda:1 gets real work.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
        split = n_layers // 2
        # Build the explicit map: embed on gpu0, first half layers on gpu0,
        # second half on gpu1, norm+lm_head on gpu1.
        device_map: dict[str, str] = {"model.embed_tokens": "cuda:0", "model.norm": "cuda:1", "lm_head": "cuda:1"}
        for i in range(n_layers):
            device_map[f"model.layers.{i}"] = "cuda:0" if i < split else "cuda:1"
        _log.info("_try_transformers_explicit: loading %s, split at layer %d", model_name, split)
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )
        _log.info("_try_transformers_explicit: loaded %s", model_name)
        return model, None
    except Exception as exc:
        reason = f"transformers_explicit_failed:{type(exc).__name__}:{exc!s:.120}"
        _log.warning("_try_transformers_explicit: %s", reason)
        return None, reason


def _try_llama_cpp(model_path_or_repo: str) -> tuple[object | None, str | None]:
    """Attempt to load via llama-cpp-python with tensor_split across two GPUs.

    tensor_split=[0.5, 0.5] tells llama.cpp to assign half the layers to each GPU.
    This is the Option C fallback when transformers is unavailable or fails.
    """
    try:
        from llama_cpp import Llama  # noqa: PLC0415
        _log.info("_try_llama_cpp: loading %s with tensor_split=[0.5,0.5]", model_path_or_repo)
        model = Llama(
            model_path=model_path_or_repo,
            n_gpu_layers=-1,
            tensor_split=[0.5, 0.5],
            verbose=False,
        )
        _log.info("_try_llama_cpp: loaded %s", model_path_or_repo)
        return model, None
    except Exception as exc:
        reason = f"llama_cpp_failed:{type(exc).__name__}:{exc!s:.120}"
        _log.warning("_try_llama_cpp: %s", reason)
        return None, reason


def load_model(vram_total_gb: float) -> tuple[object | None, str, int | None, list[str]]:
    """Load the largest feasible model across both GPUs.

    Returns (model, model_name, model_size_B, blocked_reasons).
    blocked_reasons accumulates each failed attempt so the artifact is honest
    about what was tried.

    The model_size_B field is None when loading via llama-cpp (size not easily
    inferrable without loading the whole file) or when loading fails entirely.
    """
    blocked: list[str] = []

    # Option A: 14B if we have >= 30 GB total (conservative margin for activations)
    if vram_total_gb >= 30.0:
        model, reason = _try_transformers_auto("Qwen/Qwen2.5-14B-Instruct")
        if model is not None:
            return model, "Qwen/Qwen2.5-14B-Instruct", 14, blocked
        blocked.append(reason or "unknown")

    # Option B: 7B explicit split (always attempted as a fallback from 14B or directly)
    model, reason = _try_transformers_auto("Qwen/Qwen2.5-7B-Instruct")
    if model is not None:
        return model, "Qwen/Qwen2.5-7B-Instruct", 7, blocked
    blocked.append(reason or "unknown")

    # Option B-explicit: 7B with hard-coded 28-layer split (Qwen2.5-7B has 28 transformer layers)
    model, reason = _try_transformers_explicit("Qwen/Qwen2.5-7B-Instruct", n_layers=28)
    if model is not None:
        return model, "Qwen/Qwen2.5-7B-Instruct", 7, blocked
    blocked.append(reason or "unknown")

    # Option C: llama-cpp fallback (needs a local GGUF path — skip if not present)
    gguf_candidates = list(_REPO_ROOT.glob("models/**/*.gguf")) + list(Path("/tmp").glob("*.gguf"))
    if gguf_candidates:
        gguf_path = str(gguf_candidates[0])
        model, reason = _try_llama_cpp(gguf_path)
        if model is not None:
            return model, gguf_path, None, blocked
        blocked.append(reason or "unknown")
    else:
        blocked.append("llama_cpp_no_gguf_found")

    return None, "", None, blocked


# ---------------------------------------------------------------------------
# Utilization measurement
# ---------------------------------------------------------------------------


def _sample_util_pynvml(gpu_idx: int) -> float:
    """Sample GPU utilization (0-100) for gpu_idx via pynvml.

    pynvml is the NVML Python binding — it reads utilization from the NVIDIA
    driver's hardware counter, which updates every ~10 ms.  More reliable than
    torch.cuda.utilization() for sustained workloads because it doesn't go through
    the CUDA runtime's cached view.
    """
    try:
        import pynvml  # noqa: PLC0415
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
        rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return float(rates.gpu)
    except Exception:
        return -1.0  # sentinel: pynvml unavailable or query failed


def _sample_util_torch(gpu_idx: int) -> float:
    """Sample GPU utilization (0-100) for gpu_idx via torch.cuda.utilization().

    Fallback when pynvml is not installed.  torch.cuda.utilization() queries the
    same NVML counter under the hood but goes through the CUDA runtime — in practice
    it reads 0 for very short forward passes (< 10 ms).  For real 7-13B models
    the forward pass takes multiple seconds, so this counter should register non-zero.
    """
    try:
        import torch  # noqa: PLC0415
        return float(torch.cuda.utilization(gpu_idx))
    except Exception:
        return -1.0


def sample_utilization(gpu_idx: int) -> float:
    """Sample GPU utilization, preferring pynvml over torch.cuda.utilization.

    Returns a value in [0, 100] or -1.0 if both methods fail.
    """
    val = _sample_util_pynvml(gpu_idx)
    if val >= 0.0:
        return val
    return _sample_util_torch(gpu_idx)


# ---------------------------------------------------------------------------
# Forward pass batch
# ---------------------------------------------------------------------------


def run_forward_passes(model: object, model_name: str, n_passes: int = 10) -> tuple[list[float], list[float]]:
    """Run n_passes forward passes and record GPU-0 and GPU-1 utilization after each.

    We use generate() rather than a raw forward() because:
    1. generate() with max_new_tokens=50 produces a sustained multi-second workload
       (unlike a single forward which may complete in < 10 ms on a 7B model).
    2. The NVML utilization counter needs ~10 ms of sustained compute to register
       non-zero — a single short forward pass may miss the counter's sample window.

    Returns (util_0_list, util_1_list) — one reading per pass.
    """
    util_0_list: list[float] = []
    util_1_list: list[float] = []

    try:
        from transformers import AutoTokenizer  # noqa: PLC0415
        import torch  # noqa: PLC0415
        tok = AutoTokenizer.from_pretrained(model_name)
        prompts = ["What is the capital of France?"] * 4  # batch_size=4
        inputs = tok(prompts, return_tensors="pt", padding=True)
        # Move inputs to the first GPU (model's embed_tokens layer)
        input_ids = inputs["input_ids"].to("cuda:0")
        attention_mask = inputs["attention_mask"].to("cuda:0")
    except Exception as exc:
        _log.warning("run_forward_passes: tokenizer setup failed — %s", exc)
        # Return empty lists; caller treats this as zero utilization.
        return [], []

    for i in range(n_passes):
        try:
            with __import__("torch").no_grad():
                _ = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=50,
                    do_sample=False,
                )
            # Sample immediately after the forward pass so the NVML counter sees
            # the tail of the compute burst.
            u0 = sample_utilization(0)
            u1 = sample_utilization(1)
            util_0_list.append(u0)
            util_1_list.append(u1)
            _log.info("pass %d/%d: gpu0_util=%.1f gpu1_util=%.1f", i + 1, n_passes, u0, u1)
        except Exception as exc:
            _log.warning("run_forward_passes: pass %d failed — %s", i + 1, exc)
            util_0_list.append(-1.0)
            util_1_list.append(-1.0)

    return util_0_list, util_1_list


def _llama_cpp_forward_passes(model: object, n_passes: int = 10) -> tuple[list[float], list[float]]:
    """Run n_passes generations via llama-cpp model and record utilization.

    Separate from run_forward_passes because llama-cpp's API differs from
    transformers — it takes a string prompt and returns a dict, no tokenizer needed.
    """
    util_0_list: list[float] = []
    util_1_list: list[float] = []
    prompt = "What is the capital of France?"
    for i in range(n_passes):
        try:
            _ = model(prompt, max_tokens=50)  # type: ignore[operator]
            u0 = sample_utilization(0)
            u1 = sample_utilization(1)
            util_0_list.append(u0)
            util_1_list.append(u1)
            _log.info("llama_cpp pass %d/%d: gpu0=%.1f gpu1=%.1f", i + 1, n_passes, u0, u1)
        except Exception as exc:
            _log.warning("llama_cpp pass %d failed — %s", i + 1, exc)
            util_0_list.append(-1.0)
            util_1_list.append(-1.0)
    return util_0_list, util_1_list


# ---------------------------------------------------------------------------
# Main experiment logic
# ---------------------------------------------------------------------------


def run_experiment() -> dict:  # type: ignore[type-arg]
    """Run the full DualGPU 13B proof experiment and return the result dict.

    Separated from main() so tests can call it without touching the filesystem
    or sys.exit().  Returns the result dict with all required schema fields.
    The caller is responsible for writing the dict to disk.
    """
    n_gpus, vram_0_gb, vram_1_gb = detect_gpus()

    if n_gpus < 2:
        return {
            "n_gpus": n_gpus,
            "vram_0_gb": round(vram_0_gb, 2),
            "vram_1_gb": round(vram_1_gb, 2),
            "dualgpu_available": False,
            "blocked_reason": "only_one_gpu" if n_gpus == 1 else "no_cuda_gpus",
            "model_loaded": False,
            "model_name": None,
            "model_size_B": None,
            "peak_gpu0_util": 0.0,
            "peak_gpu1_util": 0.0,
            "sustained_gpu1_fraction": 0.0,
            "dualgpu_proven": False,
            "retro_071_resolved": False,
            "honest_verdict": "dualgpu_model_load_failed",
        }

    vram_total_gb = vram_0_gb + vram_1_gb
    model, model_name, model_size_b, blocked_reasons = load_model(vram_total_gb)

    if model is None:
        return {
            "n_gpus": n_gpus,
            "vram_0_gb": round(vram_0_gb, 2),
            "vram_1_gb": round(vram_1_gb, 2),
            "dualgpu_available": True,
            "model_loaded": False,
            "model_name": None,
            "model_size_B": None,
            "blocked_reasons": blocked_reasons,
            "peak_gpu0_util": 0.0,
            "peak_gpu1_util": 0.0,
            "sustained_gpu1_fraction": 0.0,
            "dualgpu_proven": False,
            "retro_071_resolved": False,
            "honest_verdict": "dualgpu_model_load_failed",
        }

    # Sample baseline utilization before any forward passes — establishes the idle floor.
    baseline_util_0 = [sample_utilization(0) for _ in range(3)]
    baseline_util_1 = [sample_utilization(1) for _ in range(3)]
    _log.info("baseline: gpu0=%s gpu1=%s", baseline_util_0, baseline_util_1)

    # Choose the correct forward pass runner based on model type.
    is_llama_cpp = not hasattr(model, "generate")
    if is_llama_cpp:
        util_0_list, util_1_list = _llama_cpp_forward_passes(model, n_passes=10)
    else:
        util_0_list, util_1_list = run_forward_passes(model, model_name, n_passes=10)

    # Free VRAM immediately — no reason to hold the model after measurement.
    try:
        del model
        import torch  # noqa: PLC0415
        torch.cuda.empty_cache()
    except Exception as exc:
        _log.warning("cleanup: %s", exc)

    # Compute summary statistics, ignoring -1.0 sentinel values (failed queries).
    valid_u0 = [u for u in util_0_list if u >= 0.0]
    valid_u1 = [u for u in util_1_list if u >= 0.0]
    peak_gpu0_util = max(valid_u0) if valid_u0 else 0.0
    peak_gpu1_util = max(valid_u1) if valid_u1 else 0.0
    # sustained_gpu1_fraction: fraction of passes where GPU-1 was above 10% busy.
    # We use 10% as the threshold (not 50%) because sustained compute on a real model
    # should easily exceed 10%, whereas idle noise stays at 0%.
    sustained_gpu1_fraction = (
        sum(1 for u in valid_u1 if u > 10) / len(valid_u1) if valid_u1 else 0.0
    )

    dualgpu_proven = peak_gpu1_util > 50 or sustained_gpu1_fraction > 0.5
    retro_071_resolved = model_name != "" and dualgpu_proven

    if not model_name:
        honest_verdict = "dualgpu_model_load_failed"
    elif dualgpu_proven:
        honest_verdict = "dualgpu_proven"
    else:
        honest_verdict = "dualgpu_loaded_low_util"

    return {
        "n_gpus": n_gpus,
        "vram_0_gb": round(vram_0_gb, 2),
        "vram_1_gb": round(vram_1_gb, 2),
        "dualgpu_available": True,
        "model_loaded": True,
        "model_name": model_name,
        "model_size_B": model_size_b,
        "baseline_util_0": baseline_util_0,
        "baseline_util_1": baseline_util_1,
        "util_0_per_pass": util_0_list,
        "util_1_per_pass": util_1_list,
        "peak_gpu0_util": round(peak_gpu0_util, 2),
        "peak_gpu1_util": round(peak_gpu1_util, 2),
        "sustained_gpu1_fraction": round(sustained_gpu1_fraction, 4),
        "dualgpu_proven": dualgpu_proven,
        "retro_071_resolved": retro_071_resolved,
        "honest_verdict": honest_verdict,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Experiment 632 main — DualGPU 13B Forward Pass Proof.

    Execution order:
    1. apply_env_autofix() — ensures CARNOT_FORCE_LIVE=1 is set if GPU is present.
    2. ExperimentTimeoutWatchdog — hard 45-minute wall-clock cap so the conductor
       is never blocked by a hung model load or stalled forward pass.
    3. ExperimentTemplate.setup() — create output dirs, load checkpoint if present.
    4. check_exclusion_manifest() — exit early if already excluded.
    5. GPU detection — exit with dualgpu_available=False if < 2 GPUs.
    6. Model loading + forward passes + utilization measurement.
    7. build_result() + write JSON + assert_deliverable_written().
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    # Step 1: self-configure the GPU environment.
    apply_env_autofix()

    result_path = str(_REPO_ROOT / _DELIVERABLE)

    with ExperimentTimeoutWatchdog(_EXP_ID, timeout_minutes=45, result_path=result_path):
        tmpl = ExperimentTemplate(
            _EXP_ID,
            _TITLE,
            _DELIVERABLE,
            requires_gpu=True,
        )
        tmpl.setup()
        tmpl.check_exclusion_manifest()

        exp_data = run_experiment()

        status = "success" if exp_data.get("model_loaded", False) else "blocked"
        artifact = tmpl.build_result(exp_data, status=status, schema="carnot.dualgpu_13b_proof.v1")

        tmpl._output_path.parent.mkdir(parents=True, exist_ok=True)
        tmpl._output_path.write_text(json.dumps(artifact, indent=2))

        tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()

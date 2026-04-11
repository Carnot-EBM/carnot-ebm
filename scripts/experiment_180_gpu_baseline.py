#!/usr/bin/env python3
"""Experiment 180: GPU Baseline — Dual RTX 3090 inference speed and VRAM validation.

**Researcher summary:**
    Establishes baseline GPU inference speed with two NVIDIA RTX 3090s (48 GB
    total VRAM). Loads Qwen3.5-0.8B on GPU 0 and Gemma4-E4B-it on GPU 1,
    benchmarks 50 prompts per model, and compares GPU vs CPU latency.

**Detailed explanation for engineers:**
    Context: Two RTX 3090s (24 GB VRAM each) are now available via CUDA
    (PyTorch 2.11.0+cu126). Previous experiments either ran CPU-only (too slow
    for large benchmark sets) or fell back to simulation when live models were
    unavailable. This experiment:

    1. GPU 0 — Qwen/Qwen3.5-0.8B in float16:
       - Measures torch.cuda.memory_allocated() before/after load.
       - Benchmarks 50 short prompts (arithmetic questions from a fixed seed).
       - Records per-generation latency; computes p50, p95, tokens/sec.

    2. GPU 1 — google/gemma-4-E4B-it in float16:
       - Same procedure as GPU 0.

    3. Simultaneous dual-GPU test:
       - Both models loaded at the same time (already done by steps 1-2).
       - Runs 10 prompts through each model concurrently using Python threads.
       - Verifies neither GPU OOMs with both models resident.

    4. CPU baseline for each model:
       - Reloads each model in float32 on CPU (only 5 prompts — slow).
       - Records latency so GPU speedup ratio can be computed.

    CARNOT_FORCE_CPU is explicitly set to "0" in os.environ before any import
    of carnot.inference so the model_loader respects the ``device`` argument
    instead of forcing CPU.

    Results are saved to results/experiment_180_results.json.

Usage:
    .venv/bin/python scripts/experiment_180_gpu_baseline.py

Spec: REQ-VERIFY-001, REQ-VERIFY-002
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
import threading
from pathlib import Path
from statistics import median, quantiles
from typing import Any

# ---------------------------------------------------------------------------
# Force GPU mode — must be set before carnot imports.
# model_loader defaults CARNOT_FORCE_CPU=1 to avoid ROCm hangs.
# RTX 3090s are NVIDIA CUDA — safe to use GPU.
# ---------------------------------------------------------------------------
os.environ["CARNOT_FORCE_CPU"] = "0"
os.environ["CARNOT_FORCE_LIVE"] = "1"

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

import torch  # noqa: E402 — after path setup

from carnot.inference.model_loader import load_model, generate  # noqa: E402

# ---------------------------------------------------------------------------
# Benchmark prompts — 50 simple arithmetic / reasoning questions.
# Fixed seed so results are reproducible across runs.
# ---------------------------------------------------------------------------

_BENCH_PROMPTS_50 = [
    f"What is {a} + {b}?"
    for a, b in [
        (17, 38), (55, 44), (123, 456), (789, 321), (1001, 999),
        (42, 58), (100, 200), (37, 63), (250, 750), (111, 222),
        (333, 667), (400, 600), (512, 488), (1024, 976), (2048, 952),
        (3, 7), (15, 85), (64, 36), (128, 872), (256, 744),
        (500, 500), (999, 1), (1000, 1000), (12, 88), (24, 76),
        (48, 52), (96, 4), (192, 808), (384, 616), (768, 232),
        (7, 3), (14, 86), (21, 79), (28, 72), (35, 65),
        (49, 51), (56, 44), (63, 37), (70, 30), (77, 23),
        (84, 16), (91, 9), (98, 2), (105, 895), (112, 888),
        (119, 881), (126, 874), (133, 867), (140, 860), (147, 853),
    ]
]

_BENCH_PROMPTS_5_CPU = _BENCH_PROMPTS_50[:5]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vram_used_mb(device_index: int) -> float:
    """Return currently allocated VRAM in MiB for a CUDA device."""
    return torch.cuda.memory_allocated(device_index) / 1024 ** 2


def _vram_reserved_mb(device_index: int) -> float:
    """Return reserved (cached) VRAM in MiB for a CUDA device."""
    return torch.cuda.memory_reserved(device_index) / 1024 ** 2


def _benchmark_model(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    label: str,
    device_index: int | None = None,
) -> dict[str, Any]:
    """Run prompts through model, record per-generation latency and token counts.

    **Detailed explanation for engineers:**
        For each prompt:
        - Tokenize to count input tokens.
        - Time model.generate() wall-clock (torch.cuda.synchronize() before/after
          if on CUDA to ensure accurate GPU timing — otherwise GPU kernels can
          overlap with Python timing).
        - Decode output, count generated tokens.
        - Compute tokens/sec = generated_tokens / elapsed_seconds.

        Returns a dict with raw per-sample data and aggregate statistics:
        - latencies_ms: list of wall-clock ms per generation
        - tokens_per_sec: list of tokens/sec per generation
        - p50_latency_ms, p95_latency_ms: percentile latencies
        - mean_tokens_per_sec: average throughput
        - total_prompts: number of prompts run
        - errors: list of error messages if any generation failed

    Args:
        model: Loaded HuggingFace model.
        tokenizer: Matching tokenizer.
        prompts: List of prompt strings to benchmark.
        label: Human-readable label for logging.
        device_index: CUDA device index (None for CPU). Used for synchronization.

    Returns:
        Dict with latency, throughput, and error statistics.
    """
    on_cuda = device_index is not None
    latencies_ms: list[float] = []
    tokens_per_sec: list[float] = []
    errors: list[str] = []

    for i, prompt in enumerate(prompts):
        try:
            input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
            input_len = input_ids.shape[1]

            if on_cuda:
                torch.cuda.synchronize(device_index)
            t0 = time.perf_counter()

            output_text = generate(model, tokenizer, prompt, max_new_tokens=64)

            if on_cuda:
                torch.cuda.synchronize(device_index)
            elapsed = time.perf_counter() - t0

            # Count generated tokens from decoded output.
            out_ids = tokenizer(output_text, return_tensors="pt")["input_ids"]
            gen_tokens = max(out_ids.shape[1], 1)

            lat_ms = elapsed * 1000.0
            tps = gen_tokens / elapsed
            latencies_ms.append(lat_ms)
            tokens_per_sec.append(tps)

            if (i + 1) % 10 == 0:
                print(
                    f"  [{label}] {i+1}/{len(prompts)} | "
                    f"lat={lat_ms:.0f}ms | {tps:.1f} tok/s"
                )
        except Exception as exc:
            errors.append(f"prompt {i}: {exc}")
            print(f"  [{label}] ERROR prompt {i}: {exc}")

    if not latencies_ms:
        return {
            "total_prompts": len(prompts),
            "successful": 0,
            "errors": errors,
            "latencies_ms": [],
            "tokens_per_sec": [],
            "p50_latency_ms": None,
            "p95_latency_ms": None,
            "mean_tokens_per_sec": None,
        }

    p50 = median(latencies_ms)
    # quantiles() needs at least 2 data points for p95; fall back to max.
    p95 = (
        quantiles(latencies_ms, n=20)[18]  # index 18 = 95th percentile (19/20)
        if len(latencies_ms) >= 20
        else max(latencies_ms)
    )
    mean_tps = sum(tokens_per_sec) / len(tokens_per_sec)

    return {
        "total_prompts": len(prompts),
        "successful": len(latencies_ms),
        "errors": errors,
        "latencies_ms": [round(x, 2) for x in latencies_ms],
        "tokens_per_sec": [round(x, 2) for x in tokens_per_sec],
        "p50_latency_ms": round(p50, 2),
        "p95_latency_ms": round(p95, 2),
        "mean_tokens_per_sec": round(mean_tps, 2),
    }


def _load_and_benchmark_gpu(
    model_name: str,
    hf_id: str,
    device_index: int,
) -> dict[str, Any]:
    """Load a model on a specific CUDA device and benchmark it.

    **Detailed explanation for engineers:**
        Sequence:
        1. Record VRAM before load (to capture baseline VRAM already used by
           CUDA context, other models, etc.).
        2. Time AutoModelForCausalLM.from_pretrained() + .cuda(device_index).
        3. Record VRAM after load — delta is the model's footprint.
        4. Run 50-prompt benchmark.
        5. Return load metadata + benchmark results.

    Args:
        model_name: Short label (e.g., "Qwen3.5-0.8B").
        hf_id: HuggingFace model ID (e.g., "Qwen/Qwen3.5-0.8B").
        device_index: CUDA device index (0 or 1).

    Returns:
        Dict with keys: model_name, hf_id, device, load_time_s, vram_before_mb,
        vram_after_mb, vram_delta_mb, benchmark_50q.
    """
    print(f"\n{'='*60}")
    print(f"[GPU {device_index}] Loading {model_name} ({hf_id})...")

    torch.cuda.synchronize(device_index)
    vram_before_mb = _vram_used_mb(device_index)

    t_load_start = time.perf_counter()
    model, tokenizer = load_model(
        hf_id,
        device=f"cuda:{device_index}",
        dtype=torch.float16,
        max_retries=2,
    )
    torch.cuda.synchronize(device_index)
    load_time_s = time.perf_counter() - t_load_start

    vram_after_mb = _vram_used_mb(device_index)
    vram_reserved_mb = _vram_reserved_mb(device_index)
    vram_delta_mb = vram_after_mb - vram_before_mb

    print(
        f"[GPU {device_index}] Loaded in {load_time_s:.1f}s | "
        f"VRAM: {vram_delta_mb:.0f} MB used (+{vram_delta_mb:.0f} MB delta)"
    )

    print(f"[GPU {device_index}] Benchmarking 50 prompts...")
    bench = _benchmark_model(
        model, tokenizer, _BENCH_PROMPTS_50,
        label=f"GPU{device_index}/{model_name}",
        device_index=device_index,
    )

    return {
        "model_name": model_name,
        "hf_id": hf_id,
        "device": f"cuda:{device_index}",
        "load_time_s": round(load_time_s, 3),
        "vram_before_mb": round(vram_before_mb, 1),
        "vram_after_mb": round(vram_after_mb, 1),
        "vram_reserved_mb": round(vram_reserved_mb, 1),
        "vram_delta_mb": round(vram_delta_mb, 1),
        "benchmark_50q": bench,
        # Keep references for later dual-GPU and CPU tests.
        "_model": model,
        "_tokenizer": tokenizer,
    }


def _load_and_benchmark_cpu(
    model_name: str,
    hf_id: str,
) -> dict[str, Any]:
    """Load model on CPU (float32) and benchmark 5 prompts for speedup baseline.

    **Detailed explanation for engineers:**
        CPU inference in float32 is slow. We run only 5 prompts to keep total
        experiment time reasonable. The GPU/CPU speedup ratio is computed by
        the caller as gpu_mean_tps / cpu_mean_tps.

    Args:
        model_name: Short label.
        hf_id: HuggingFace model ID.

    Returns:
        Dict with load_time_s, benchmark_5q results.
    """
    print(f"\n[CPU] Loading {model_name} for CPU baseline (5 prompts)...")
    t_load_start = time.perf_counter()
    # Temporarily force CPU load regardless of CARNOT_FORCE_CPU setting.
    model_cpu, tok_cpu = load_model(
        hf_id,
        device="cpu",
        dtype=torch.float32,
        max_retries=1,
    )
    load_time_s = time.perf_counter() - t_load_start
    print(f"[CPU] Loaded in {load_time_s:.1f}s")

    bench = _benchmark_model(
        model_cpu, tok_cpu, _BENCH_PROMPTS_5_CPU,
        label=f"CPU/{model_name}",
        device_index=None,
    )

    # Free CPU model immediately — it's large.
    del model_cpu, tok_cpu
    gc.collect()

    return {
        "load_time_s": round(load_time_s, 3),
        "benchmark_5q": bench,
    }


def _dual_gpu_concurrent_test(
    model0: Any, tok0: Any,
    model1: Any, tok1: Any,
) -> dict[str, Any]:
    """Run 10 prompts through each model concurrently using Python threads.

    **Detailed explanation for engineers:**
        Uses threading.Thread (not multiprocessing) because both models are
        already in the same process. GIL is released during torch.no_grad()
        .generate() calls so CUDA kernels on GPU 0 and GPU 1 run in parallel.

        Measures total wall-clock time for both batches to complete. If both
        GPUs are truly independent, total_time ≈ max(time0, time1) rather than
        sum(time0, time1).

    Returns:
        Dict with per-model concurrent latencies and total elapsed time.
    """
    print("\n[DUAL-GPU] Running concurrent inference test (10 prompts each)...")
    prompts_10 = _BENCH_PROMPTS_50[:10]
    results: dict[str, Any] = {}
    errors_0: list[str] = []
    errors_1: list[str] = []

    def run_gpu0() -> None:
        lats = []
        for i, p in enumerate(prompts_10):
            try:
                torch.cuda.synchronize(0)
                t0 = time.perf_counter()
                generate(model0, tok0, p, max_new_tokens=64)
                torch.cuda.synchronize(0)
                lats.append((time.perf_counter() - t0) * 1000.0)
            except Exception as exc:
                errors_0.append(f"prompt {i}: {exc}")
        results["gpu0_latencies_ms"] = [round(x, 2) for x in lats]

    def run_gpu1() -> None:
        lats = []
        for i, p in enumerate(prompts_10):
            try:
                torch.cuda.synchronize(1)
                t0 = time.perf_counter()
                generate(model1, tok1, p, max_new_tokens=64)
                torch.cuda.synchronize(1)
                lats.append((time.perf_counter() - t0) * 1000.0)
            except Exception as exc:
                errors_1.append(f"prompt {i}: {exc}")
        results["gpu1_latencies_ms"] = [round(x, 2) for x in lats]

    t_wall_start = time.perf_counter()
    t0 = threading.Thread(target=run_gpu0, daemon=True)
    t1 = threading.Thread(target=run_gpu1, daemon=True)
    t0.start(); t1.start()
    t0.join(); t1.join()
    total_wall_s = time.perf_counter() - t_wall_start

    results["total_wall_time_s"] = round(total_wall_s, 3)
    results["gpu0_errors"] = errors_0
    results["gpu1_errors"] = errors_1
    results["concurrent_prompts_per_gpu"] = 10

    # Sanity: if serial, total ≈ sum of all latencies; if parallel, ≈ half.
    sum_gpu0 = sum(results.get("gpu0_latencies_ms", []))
    sum_gpu1 = sum(results.get("gpu1_latencies_ms", []))
    serial_expected_s = (sum_gpu0 + sum_gpu1) / 1000.0
    results["serial_expected_s"] = round(serial_expected_s, 3)
    results["parallelism_efficiency"] = round(
        serial_expected_s / (total_wall_s * 2 + 1e-9), 4
    ) if total_wall_s > 0 else None

    print(
        f"[DUAL-GPU] Done — wall={total_wall_s:.1f}s, "
        f"serial_expected={serial_expected_s:.1f}s"
    )
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full GPU baseline experiment and save results."""
    t_exp_start = time.time()
    print("=" * 60)
    print("Experiment 180: Dual RTX 3090 GPU Baseline")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_vram_mb = props.total_memory / 1024 ** 2
        print(f"  GPU {i}: {props.name} — {total_vram_mb:.0f} MB VRAM total")
    print("=" * 60)

    results: dict[str, Any] = {
        "experiment": 180,
        "title": "Dual RTX 3090 GPU Baseline — load time, VRAM, throughput, latency",
        "pytorch_version": torch.__version__,
        "cuda_device_count": torch.cuda.device_count(),
        "devices": [],
        "gpu0": {},
        "gpu1": {},
        "dual_gpu_concurrent": {},
        "cpu_baseline": {},
        "speedup_ratios": {},
        "errors": [],
    }

    # Collect device info.
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        results["devices"].append({
            "index": i,
            "name": props.name,
            "total_vram_mb": round(props.total_memory / 1024 ** 2, 1),
            "compute_capability": f"{props.major}.{props.minor}",
        })

    # -----------------------------------------------------------------------
    # Step 1: Load Qwen3.5-0.8B on GPU 0
    # -----------------------------------------------------------------------
    try:
        gpu0_result = _load_and_benchmark_gpu(
            model_name="Qwen3.5-0.8B",
            hf_id="Qwen/Qwen3.5-0.8B",
            device_index=0,
        )
        model0 = gpu0_result.pop("_model")
        tok0 = gpu0_result.pop("_tokenizer")
        results["gpu0"] = gpu0_result
        print(
            f"[GPU 0] p50={gpu0_result['benchmark_50q']['p50_latency_ms']}ms "
            f"p95={gpu0_result['benchmark_50q']['p95_latency_ms']}ms "
            f"mean_tps={gpu0_result['benchmark_50q']['mean_tokens_per_sec']}"
        )
    except Exception as exc:
        results["errors"].append(f"gpu0_load: {exc}")
        print(f"ERROR loading GPU 0 model: {exc}")
        model0, tok0 = None, None

    # -----------------------------------------------------------------------
    # Step 2: Load Gemma4-E4B-it on GPU 1
    # -----------------------------------------------------------------------
    try:
        gpu1_result = _load_and_benchmark_gpu(
            model_name="Gemma4-E4B-it",
            hf_id="google/gemma-4-E4B-it",
            device_index=1,
        )
        model1 = gpu1_result.pop("_model")
        tok1 = gpu1_result.pop("_tokenizer")
        results["gpu1"] = gpu1_result
        print(
            f"[GPU 1] p50={gpu1_result['benchmark_50q']['p50_latency_ms']}ms "
            f"p95={gpu1_result['benchmark_50q']['p95_latency_ms']}ms "
            f"mean_tps={gpu1_result['benchmark_50q']['mean_tokens_per_sec']}"
        )
    except Exception as exc:
        results["errors"].append(f"gpu1_load: {exc}")
        print(f"ERROR loading GPU 1 model: {exc}")
        model1, tok1 = None, None

    # -----------------------------------------------------------------------
    # Step 3: Dual-GPU concurrent test (both models already loaded)
    # -----------------------------------------------------------------------
    if model0 is not None and model1 is not None:
        try:
            dual = _dual_gpu_concurrent_test(model0, tok0, model1, tok1)
            results["dual_gpu_concurrent"] = dual
        except Exception as exc:
            results["errors"].append(f"dual_gpu_concurrent: {exc}")
            print(f"ERROR in dual-GPU test: {exc}")
    else:
        results["dual_gpu_concurrent"]["skipped"] = "one or both GPU models failed to load"

    # -----------------------------------------------------------------------
    # Step 4: CPU baseline — Qwen3.5-0.8B (5 prompts)
    # -----------------------------------------------------------------------
    print("\nRunning CPU baseline for Qwen3.5-0.8B (5 prompts)...")
    try:
        cpu_qwen = _load_and_benchmark_cpu("Qwen3.5-0.8B", "Qwen/Qwen3.5-0.8B")
        results["cpu_baseline"]["Qwen3.5-0.8B"] = cpu_qwen

        # Speedup ratio: GPU mean_tps / CPU mean_tps
        gpu_tps = results["gpu0"].get("benchmark_50q", {}).get("mean_tokens_per_sec")
        cpu_tps = cpu_qwen["benchmark_5q"].get("mean_tokens_per_sec")
        if gpu_tps and cpu_tps and cpu_tps > 0:
            results["speedup_ratios"]["Qwen3.5-0.8B_gpu_vs_cpu"] = round(
                gpu_tps / cpu_tps, 2
            )
            print(
                f"[Speedup] Qwen3.5-0.8B GPU/CPU = "
                f"{results['speedup_ratios']['Qwen3.5-0.8B_gpu_vs_cpu']}x"
            )
    except Exception as exc:
        results["errors"].append(f"cpu_baseline_qwen: {exc}")
        print(f"ERROR in CPU baseline (Qwen): {exc}")

    # -----------------------------------------------------------------------
    # Step 5: CPU baseline — Gemma4-E4B-it (5 prompts) — skipped if VRAM tight
    # NOTE: Gemma-4-E4B-it in float32 on CPU may need >16 GB RAM.
    # We attempt it but catch OOM gracefully.
    # -----------------------------------------------------------------------
    print("\nRunning CPU baseline for Gemma4-E4B-it (5 prompts)...")
    try:
        cpu_gemma = _load_and_benchmark_cpu("Gemma4-E4B-it", "google/gemma-4-E4B-it")
        results["cpu_baseline"]["Gemma4-E4B-it"] = cpu_gemma

        gpu_tps = results["gpu1"].get("benchmark_50q", {}).get("mean_tokens_per_sec")
        cpu_tps = cpu_gemma["benchmark_5q"].get("mean_tokens_per_sec")
        if gpu_tps and cpu_tps and cpu_tps > 0:
            results["speedup_ratios"]["Gemma4-E4B-it_gpu_vs_cpu"] = round(
                gpu_tps / cpu_tps, 2
            )
            print(
                f"[Speedup] Gemma4-E4B-it GPU/CPU = "
                f"{results['speedup_ratios']['Gemma4-E4B-it_gpu_vs_cpu']}x"
            )
    except Exception as exc:
        results["errors"].append(f"cpu_baseline_gemma: {exc}")
        print(f"ERROR in CPU baseline (Gemma) — may be OOM: {exc}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    results["total_experiment_time_s"] = round(time.time() - t_exp_start, 1)

    out_path = RESULTS_DIR / "experiment_180_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print compact summary.
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for gpu_key, gpu_label in [("gpu0", "GPU 0 (Qwen3.5-0.8B)"), ("gpu1", "GPU 1 (Gemma4-E4B-it)")]:
        g = results.get(gpu_key, {})
        b = g.get("benchmark_50q", {})
        print(
            f"{gpu_label}: load={g.get('load_time_s', 'N/A')}s | "
            f"VRAM={g.get('vram_delta_mb', 'N/A')} MB | "
            f"p50={b.get('p50_latency_ms', 'N/A')}ms | "
            f"p95={b.get('p95_latency_ms', 'N/A')}ms | "
            f"tps={b.get('mean_tokens_per_sec', 'N/A')}"
        )
    for name, ratio in results["speedup_ratios"].items():
        print(f"Speedup {name}: {ratio}x GPU vs CPU")
    print(f"Total time: {results['total_experiment_time_s']}s")
    if results["errors"]:
        print(f"Errors: {results['errors']}")


if __name__ == "__main__":
    main()

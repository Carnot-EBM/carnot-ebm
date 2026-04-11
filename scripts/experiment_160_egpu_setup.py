#!/usr/bin/env python3
"""Experiment 160: eGPU Hardware Validation — RX 7900 XTX via Thunderbolt.

**Researcher summary:**
    Validates the Radeon RX 7900 XTX (gfx1100, 24GB VRAM) connected via
    Thunderbolt eGPU chassis. If the eGPU is present, benchmarks Qwen3.5-0.8B
    inference speed on GPU vs CPU and tests JAX GPU capability.

    If the eGPU is NOT detected (Thunderbolt not connected or driver not ready),
    the script documents the blocker clearly and runs a CPU baseline, so the
    result file always exists and the conductor can proceed.

**Detailed explanation for engineers:**
    Detection chain:
    1. Run `rocminfo` and look for "gfx" strings to enumerate detected GPUs.
       - gfx1100 = RX 7900 XTX (well-supported by ROCm 6.x and JAX)
       - gfx1150 = iGPU 890M (ROCm crashes JAX — broken for our purposes)
    2. Check PyTorch ROCm: torch.cuda.is_available() + device name.
    3. Check JAX devices: jax.devices() — if it returns a GPU device, JAX GPU works.
    4. If eGPU detected:
       - Benchmark Qwen3.5-0.8B: 10 warmup + 20 timed responses on GPU and CPU.
       - JAX matmul benchmark: 1000×1000 on GPU vs CPU (jnp.dot).
    5. If not detected:
       - Classify blocker: Thunderbolt not connected vs driver issue.
       - Run CPU inference baseline only.

    ROCm note: on AMD hardware torch.cuda is actually the ROCm HIP runtime.
    `torch.cuda.is_available()` returns True when ROCm is active and a GPU
    is detected by the HIP runtime. `torch.cuda.get_device_name(0)` returns
    the actual GPU name (e.g., "AMD Radeon RX 7900 XTX").

    JAX note: with ROCm/HIP plugin installed, `jax.devices()` will contain
    CudaDevice objects when a GPU is detected. The default JAX pip install
    uses the XLA CPU backend. JAX ROCm requires the `jax[rocm]` extra or
    the separate `jaxlib-rocm` wheel.

    This experiment intentionally does NOT set JAX_PLATFORMS=cpu so that JAX
    auto-detects available hardware. If gfx1100 + JAX GPU works, subsequent
    experiments can drop the JAX_PLATFORMS=cpu restriction.

Usage:
    .venv/bin/python scripts/experiment_160_egpu_setup.py

    (NO JAX_PLATFORMS=cpu prefix — this experiment tests GPU availability)

Spec: REQ-VERIFY-001, SCENARIO-VERIFY-003
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup — make carnot library importable.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("exp160")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3.5-0.8B"
WARMUP_RUNS = 10       # discarded
TIMED_RUNS = 20        # for p50/p99 calculation
MATMUL_SIZE = 1000     # side length of the square matrix for JAX benchmark
MATMUL_RUNS = 20       # repeated matmul runs to get stable timing

# A simple prompt that produces short, deterministic output — fast enough to
# benchmark without waiting minutes per run.
BENCHMARK_PROMPT = "What is 7 times 8? Answer with just the number."


# ---------------------------------------------------------------------------
# Step 1: System detection helpers
# ---------------------------------------------------------------------------

def detect_rocm_devices() -> list[str]:
    """Run rocminfo and return a list of gfx architecture strings found.

    **Detailed explanation for engineers:**
        rocminfo prints device information for every GPU visible to the ROCm
        stack. We grep for lines containing "gfx" to find the architecture
        identifiers (e.g., "gfx1100" for RX 7900 XTX). Returns an empty list
        if rocminfo is not installed or no GPUs are found.
    """
    try:
        result = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        lines = result.stdout.lower().splitlines() + result.stderr.lower().splitlines()
        gfx_strings: list[str] = []
        for line in lines:
            if "gfx" in line:
                # Extract the gfxNNNN token from the line.
                for token in line.split():
                    if token.startswith("gfx") or "gfx" in token:
                        # Clean up surrounding punctuation.
                        clean = token.strip("(),;:")
                        if clean not in gfx_strings:
                            gfx_strings.append(clean)
        return gfx_strings
    except FileNotFoundError:
        log.warning("rocminfo not found — ROCm stack not installed or not on PATH.")
        return []
    except subprocess.TimeoutExpired:
        log.warning("rocminfo timed out — ROCm stack may be unresponsive.")
        return []
    except Exception as exc:
        log.warning("rocminfo failed: %s", exc)
        return []


def detect_rocm_version() -> str:
    """Return ROCm version string, or 'not_installed' if unavailable.

    **Detailed explanation for engineers:**
        We try two sources in order:
        1. /opt/rocm/.info/version — written by the ROCm installer.
        2. rocminfo output line containing "Runtime Version".
        Returns the first successful match.
    """
    # Try the version file first (most reliable).
    for version_path in ["/opt/rocm/.info/version", "/opt/rocm/share/info/version"]:
        try:
            content = Path(version_path).read_text().strip()
            if content:
                return content
        except (FileNotFoundError, PermissionError):
            pass

    # Fall back to rocminfo output.
    try:
        result = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        for line in result.stdout.splitlines():
            if "runtime version" in line.lower():
                parts = line.split(":")
                if len(parts) >= 2:
                    return parts[-1].strip()
    except Exception:
        pass

    return "not_installed"


def detect_pytorch_gpu() -> dict[str, Any]:
    """Check PyTorch CUDA/ROCm availability and device info.

    **Detailed explanation for engineers:**
        On AMD hardware with ROCm, PyTorch uses the HIP backend which exposes
        itself through the torch.cuda namespace. torch.cuda.is_available() is
        the canonical check. We also collect device_name and device_count.

    Returns:
        Dict with keys: available (bool), device_name (str), device_count (int).
    """
    try:
        import torch  # type: ignore[import]
        available = torch.cuda.is_available()
        if available:
            device_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
        else:
            device_name = "none"
            device_count = 0
        return {
            "available": available,
            "device_name": device_name,
            "device_count": device_count,
            "torch_version": torch.__version__,
        }
    except ImportError:
        return {
            "available": False,
            "device_name": "torch_not_installed",
            "device_count": 0,
            "torch_version": "not_installed",
        }


def detect_jax_devices() -> dict[str, Any]:
    """Return JAX device info: backend name and list of device descriptions.

    **Detailed explanation for engineers:**
        jax.devices() returns a list of device objects. The repr() of each
        device shows the backend and index, e.g. "CudaDevice(id=0)" for GPU
        or "CpuDevice(id=0)" for CPU. We collect reprs for all devices.
        The jax.default_backend() string is "gpu", "cpu", or "tpu".

    Returns:
        Dict with keys: backend (str), devices (list[str]), gpu_available (bool).
    """
    try:
        import jax  # type: ignore[import]
        devices = jax.devices()
        device_reprs = [repr(d) for d in devices]
        backend = jax.default_backend()
        gpu_available = backend in ("gpu", "cuda", "rocm")
        return {
            "backend": backend,
            "devices": device_reprs,
            "gpu_available": gpu_available,
        }
    except ImportError:
        return {
            "backend": "jax_not_installed",
            "devices": [],
            "gpu_available": False,
        }
    except Exception as exc:
        return {
            "backend": f"error: {exc}",
            "devices": [],
            "gpu_available": False,
        }


def classify_blocker(rocm_devices: list[str], pytorch_info: dict[str, Any]) -> str:
    """Classify why the eGPU is not usable, for clear experiment documentation.

    **Detailed explanation for engineers:**
        We distinguish four cases:
        - "thunderbolt_not_connected": rocminfo returns no GPUs at all. The
          chassis isn't connected or isn't powered on. Physical action needed.
        - "igpu_only_gfx1150": rocminfo sees only the iGPU (gfx1150), not the
          eGPU (gfx1100). Thunderbolt may be connected but GPU not powered.
        - "driver_issue": rocminfo sees something but PyTorch says unavailable.
          ROCm driver stack problem.
        - "none": eGPU is detected and working.
    """
    if not pytorch_info["available"]:
        has_gfx1100 = any("gfx1100" in d for d in rocm_devices)
        has_any_gpu = bool(rocm_devices)
        if not has_any_gpu:
            return "thunderbolt_not_connected"
        elif not has_gfx1100:
            return "igpu_only_gfx1150"
        else:
            return "driver_issue"
    return "none"


# ---------------------------------------------------------------------------
# Step 2: JAX matmul benchmark
# ---------------------------------------------------------------------------

def benchmark_jax_matmul(device: str = "cpu") -> dict[str, Any]:
    """Run repeated 1000×1000 matmuls on the given JAX device, return timing.

    **Detailed explanation for engineers:**
        jax.device_put() moves a buffer to the requested device. We use
        jnp.dot() which maps to XLA's DotGeneral operation — well-optimised
        on both CPU and GPU backends. jax.block_until_ready() waits for
        async computation to complete before stopping the timer.

        We run MATMUL_RUNS times and report min/median/max in milliseconds.

    Args:
        device: "cpu" or "gpu" — selects the JAX device.

    Returns:
        Dict with keys: device, min_ms, median_ms, max_ms, error (if failed).
    """
    try:
        import jax  # type: ignore[import]
        import jax.numpy as jnp  # type: ignore[import]
        import numpy as np  # type: ignore[import]

        # Pick the target device.
        devices = jax.devices(device)
        target_device = devices[0]

        rng = np.random.default_rng(42)
        a = jnp.array(rng.standard_normal((MATMUL_SIZE, MATMUL_SIZE)), dtype=jnp.float32)
        b = jnp.array(rng.standard_normal((MATMUL_SIZE, MATMUL_SIZE)), dtype=jnp.float32)
        a = jax.device_put(a, target_device)
        b = jax.device_put(b, target_device)

        # Warmup — JIT compile and fill caches.
        for _ in range(3):
            c = jnp.dot(a, b)
            jax.block_until_ready(c)

        times_ms: list[float] = []
        for _ in range(MATMUL_RUNS):
            t0 = time.perf_counter()
            c = jnp.dot(a, b)
            jax.block_until_ready(c)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

        return {
            "device": device,
            "min_ms": round(min(times_ms), 3),
            "median_ms": round(statistics.median(times_ms), 3),
            "max_ms": round(max(times_ms), 3),
            "error": None,
        }
    except Exception as exc:
        return {
            "device": device,
            "min_ms": None,
            "median_ms": None,
            "max_ms": None,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Step 3: LLM inference benchmark
# ---------------------------------------------------------------------------

def _time_inference_runs(
    model: Any,
    tokenizer: Any,
    device_label: str,
    n_warmup: int,
    n_timed: int,
) -> dict[str, Any]:
    """Run warmup + timed inference passes; return timing stats in ms.

    **Detailed explanation for engineers:**
        We use the generate() function from carnot.inference.model_loader which
        handles Qwen3 chat-template quirks. Each call produces a short response
        to BENCHMARK_PROMPT. We collect wall-clock times with time.perf_counter()
        (nanosecond resolution) and report p50 and p99 percentiles.

        Note: the first run on GPU has additional JIT/kernel launch overhead;
        warmup runs absorb this so timed runs are steady-state.
    """
    from carnot.inference.model_loader import generate  # type: ignore[import]

    log.info("  [%s] warming up (%d runs)...", device_label, n_warmup)
    for _ in range(n_warmup):
        generate(model, tokenizer, BENCHMARK_PROMPT, max_new_tokens=16)

    log.info("  [%s] timing (%d runs)...", device_label, n_timed)
    times_ms: list[float] = []
    for i in range(n_timed):
        t0 = time.perf_counter()
        generate(model, tokenizer, BENCHMARK_PROMPT, max_new_tokens=16)
        t1 = time.perf_counter()
        elapsed = (t1 - t0) * 1000.0
        times_ms.append(elapsed)
        log.info("    run %d/%d: %.1f ms", i + 1, n_timed, elapsed)

    sorted_times = sorted(times_ms)
    p50 = statistics.median(sorted_times)
    # p99: index at 99th percentile
    p99_idx = max(0, int(0.99 * len(sorted_times)) - 1)
    p99 = sorted_times[p99_idx]

    return {
        "device": device_label,
        "p50_ms": round(p50, 1),
        "p99_ms": round(p99, 1),
        "min_ms": round(min(times_ms), 1),
        "max_ms": round(max(times_ms), 1),
        "n_runs": n_timed,
    }


def benchmark_llm_inference(egpu_available: bool) -> dict[str, Any]:
    """Benchmark Qwen3.5-0.8B inference on CPU (and GPU if available).

    **Detailed explanation for engineers:**
        For CPU benchmark: we clear CARNOT_FORCE_CPU to let load_model use
        whatever device we request; we explicitly pass device="cpu".

        For GPU benchmark: we set CARNOT_FORCE_CPU=0 and pass device="cuda"
        so that load_model places the model on the GPU. We then reload the
        model on GPU (separate object to avoid contamination).

        The model is loaded once per device to avoid cross-device caching
        artifacts. On GPU we pass dtype=None (auto) which resolves to float16.

    Returns:
        Dict with cpu_timing, gpu_timing (or None), speedup_ratio, error.
    """
    # We need to temporarily manipulate env vars for device selection.
    original_force_cpu = os.environ.get("CARNOT_FORCE_CPU", "1")
    original_skip_llm = os.environ.get("CARNOT_SKIP_LLM", "")

    try:
        from carnot.inference.model_loader import load_model  # type: ignore[import]

        # --- CPU benchmark ---
        log.info("[LLM] Loading %s on CPU...", MODEL_NAME)
        os.environ["CARNOT_FORCE_CPU"] = "1"
        os.environ.pop("CARNOT_SKIP_LLM", None)
        model_cpu, tok_cpu = load_model(MODEL_NAME, device="cpu")

        if model_cpu is None:
            return {
                "cpu_timing": None,
                "gpu_timing": None,
                "speedup_ratio": None,
                "error": "Model failed to load on CPU — check HuggingFace cache.",
                "model_name": MODEL_NAME,
            }

        log.info("[LLM] Running CPU inference benchmark...")
        cpu_timing = _time_inference_runs(
            model_cpu, tok_cpu, "cpu", WARMUP_RUNS, TIMED_RUNS
        )

        # Free CPU model before GPU load to avoid OOM.
        del model_cpu, tok_cpu
        try:
            import gc
            import torch  # type: ignore[import]
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

        # --- GPU benchmark (only if eGPU detected) ---
        gpu_timing = None
        speedup_ratio = None

        if egpu_available:
            log.info("[LLM] Loading %s on GPU...", MODEL_NAME)
            os.environ["CARNOT_FORCE_CPU"] = "0"
            model_gpu, tok_gpu = load_model(MODEL_NAME, device="cuda")

            if model_gpu is not None:
                log.info("[LLM] Running GPU inference benchmark...")
                gpu_timing = _time_inference_runs(
                    model_gpu, tok_gpu, "gpu", WARMUP_RUNS, TIMED_RUNS
                )
                if cpu_timing["p50_ms"] and gpu_timing["p50_ms"]:
                    speedup_ratio = round(cpu_timing["p50_ms"] / gpu_timing["p50_ms"], 2)
                del model_gpu, tok_gpu
            else:
                gpu_timing = {"error": "load_model returned None on GPU"}

        return {
            "cpu_timing": cpu_timing,
            "gpu_timing": gpu_timing,
            "speedup_ratio": speedup_ratio,
            "error": None,
            "model_name": MODEL_NAME,
        }

    except Exception as exc:
        return {
            "cpu_timing": None,
            "gpu_timing": None,
            "speedup_ratio": None,
            "error": str(exc),
            "model_name": MODEL_NAME,
        }
    finally:
        # Restore original env vars regardless of outcome.
        os.environ["CARNOT_FORCE_CPU"] = original_force_cpu
        if original_skip_llm:
            os.environ["CARNOT_SKIP_LLM"] = original_skip_llm
        else:
            os.environ.pop("CARNOT_SKIP_LLM", None)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    """Run experiment 160 end-to-end and write results JSON."""
    log.info("=" * 60)
    log.info("Experiment 160: eGPU Hardware Validation")
    log.info("=" * 60)

    # --- Step 1: System detection ---
    log.info("[Step 1] Detecting hardware...")

    rocm_devices = detect_rocm_devices()
    log.info("  rocminfo gfx strings: %s", rocm_devices or ["(none)"])

    rocm_version = detect_rocm_version()
    log.info("  ROCm version: %s", rocm_version)

    pytorch_info = detect_pytorch_gpu()
    log.info(
        "  PyTorch: version=%s, cuda_available=%s, device=%s",
        pytorch_info["torch_version"],
        pytorch_info["available"],
        pytorch_info["device_name"],
    )

    jax_info = detect_jax_devices()
    log.info(
        "  JAX: backend=%s, gpu_available=%s, devices=%s",
        jax_info["backend"],
        jax_info["gpu_available"],
        jax_info["devices"],
    )

    # Check whether we have the RX 7900 XTX (gfx1100) specifically.
    has_gfx1100 = any("gfx1100" in d for d in rocm_devices)
    egpu_detected = pytorch_info["available"] and has_gfx1100
    blocker = classify_blocker(rocm_devices, pytorch_info)

    log.info(
        "  eGPU detected (gfx1100 + PyTorch): %s  |  blocker: %s",
        egpu_detected,
        blocker,
    )

    # --- Step 2: JAX matmul benchmark ---
    log.info("[Step 2] JAX matmul benchmark (CPU)...")
    jax_cpu_matmul = benchmark_jax_matmul("cpu")
    log.info(
        "  CPU matmul %dx%d: median=%.3f ms, min=%.3f ms, max=%.3f ms",
        MATMUL_SIZE, MATMUL_SIZE,
        jax_cpu_matmul.get("median_ms") or 0,
        jax_cpu_matmul.get("min_ms") or 0,
        jax_cpu_matmul.get("max_ms") or 0,
    )

    jax_gpu_matmul = None
    if jax_info["gpu_available"]:
        log.info("[Step 2] JAX matmul benchmark (GPU)...")
        jax_gpu_matmul = benchmark_jax_matmul("gpu")
        log.info(
            "  GPU matmul %dx%d: median=%.3f ms, min=%.3f ms, max=%.3f ms",
            MATMUL_SIZE, MATMUL_SIZE,
            jax_gpu_matmul.get("median_ms") or 0,
            jax_gpu_matmul.get("min_ms") or 0,
            jax_gpu_matmul.get("max_ms") or 0,
        )
    else:
        log.info("[Step 2] JAX GPU not available — skipping GPU matmul benchmark.")

    # --- Step 3: LLM inference benchmark ---
    log.info("[Step 3] LLM inference benchmark (model: %s)...", MODEL_NAME)
    if os.environ.get("CARNOT_SKIP_LLM"):
        log.info("  CARNOT_SKIP_LLM set — skipping LLM benchmark.")
        llm_results: dict[str, Any] = {
            "cpu_timing": None,
            "gpu_timing": None,
            "speedup_ratio": None,
            "error": "CARNOT_SKIP_LLM was set",
            "model_name": MODEL_NAME,
        }
    else:
        llm_results = benchmark_llm_inference(egpu_available=egpu_detected)

    # --- Step 4: Build result document ---
    if not egpu_detected:
        log.warning("eGPU NOT detected. Blocker: %s", blocker)
        blocker_message = {
            "thunderbolt_not_connected": (
                "RX 7900 XTX not visible to ROCm. The Thunderbolt chassis is "
                "likely not physically connected or not powered on. "
                "Action: connect the Thunderbolt eGPU chassis with the RX 7900 XTX, "
                "then re-run this experiment. "
                "Fallback: subsequent live-inference experiments should use "
                "CARNOT_SKIP_LLM=1 (simulation mode) and note this hardware blocker."
            ),
            "igpu_only_gfx1150": (
                "Only the iGPU (gfx1150 / Radeon 890M) is visible to ROCm — "
                "the eGPU (gfx1100 / RX 7900 XTX) is not detected. "
                "The Thunderbolt chassis may be connected but the GPU not powered. "
                "Action: ensure the chassis has power and the RX 7900 XTX is seated. "
                "Fallback: use CARNOT_SKIP_LLM=1 for experiments until resolved."
            ),
            "driver_issue": (
                "ROCm sees a GPU but PyTorch reports CUDA unavailable. "
                "Likely a ROCm driver version mismatch or missing HIP libraries. "
                "Action: verify ROCm installation with `rocminfo` and reinstall if needed."
            ),
        }.get(blocker, "Unknown blocker — investigate manually.")
    else:
        blocker_message = None
        log.info(
            "eGPU DETECTED: %s (gfx1100). JAX GPU: %s. "
            "If JAX GPU works, JAX_PLATFORMS=cpu is no longer required for Exp 161+.",
            pytorch_info["device_name"],
            jax_info["gpu_available"],
        )

    # Summarise LLM timings at top level for easy scanning.
    cpu_p50 = None
    cpu_p99 = None
    gpu_p50 = None
    gpu_p99 = None
    if llm_results.get("cpu_timing"):
        cpu_p50 = llm_results["cpu_timing"].get("p50_ms")
        cpu_p99 = llm_results["cpu_timing"].get("p99_ms")
    if llm_results.get("gpu_timing") and isinstance(llm_results["gpu_timing"], dict):
        gpu_p50 = llm_results["gpu_timing"].get("p50_ms")
        gpu_p99 = llm_results["gpu_timing"].get("p99_ms")

    results: dict[str, Any] = {
        "experiment": 160,
        "title": "eGPU Hardware Validation — RX 7900 XTX via Thunderbolt",

        # Detection summary (top-level for quick scanning)
        "egpu_detected": egpu_detected,
        "rocm_version": rocm_version,
        "rocm_devices": rocm_devices,
        "jax_backend": jax_info["backend"],
        "jax_gpu_available": jax_info["gpu_available"],
        "pytorch_cuda_available": pytorch_info["available"],
        "pytorch_device_name": pytorch_info["device_name"],

        # LLM inference timing (top-level for easy comparison with other exps)
        "cpu_inference_p50_ms": cpu_p50,
        "cpu_inference_p99_ms": cpu_p99,
        "gpu_inference_p50_ms": gpu_p50,
        "gpu_inference_p99_ms": gpu_p99,
        "speedup_ratio": llm_results.get("speedup_ratio"),

        # Blocker (null when eGPU works)
        "blocker": blocker if blocker != "none" else None,
        "blocker_message": blocker_message,
        "fallback_plan": (
            "Use CARNOT_SKIP_LLM=1 (simulation mode) for subsequent "
            "live-inference experiments until the eGPU is connected."
            if blocker != "none"
            else None
        ),

        # Detailed sub-results
        "jax_cpu_matmul": jax_cpu_matmul,
        "jax_gpu_matmul": jax_gpu_matmul,
        "llm_benchmark": llm_results,
        "pytorch_info": pytorch_info,
        "jax_info": jax_info,

        # Recommendations
        "recommendations": _build_recommendations(
            egpu_detected=egpu_detected,
            jax_gpu_available=jax_info["gpu_available"],
            blocker=blocker,
            speedup_ratio=llm_results.get("speedup_ratio"),
        ),
    }

    out_path = RESULTS_DIR / "experiment_160_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    log.info("Results written to %s", out_path)

    # --- Summary to stdout ---
    print("\n" + "=" * 60)
    print("Experiment 160 Summary")
    print("=" * 60)
    print(f"eGPU detected:        {egpu_detected}")
    print(f"ROCm version:         {rocm_version}")
    print(f"ROCm devices:         {rocm_devices or '(none)'}")
    print(f"JAX backend:          {jax_info['backend']}")
    print(f"JAX GPU available:    {jax_info['gpu_available']}")
    print(f"PyTorch GPU:          {pytorch_info['device_name']}")
    if cpu_p50 is not None:
        print(f"CPU inference p50:    {cpu_p50} ms")
        print(f"CPU inference p99:    {cpu_p99} ms")
    if gpu_p50 is not None:
        print(f"GPU inference p50:    {gpu_p50} ms")
        print(f"GPU inference p99:    {gpu_p99} ms")
    if llm_results.get("speedup_ratio"):
        print(f"GPU speedup ratio:    {llm_results['speedup_ratio']}×")
    if blocker_message:
        print(f"\nBLOCKER: {blocker}")
        print(f"  {blocker_message[:160]}...")
    for rec in results["recommendations"]:
        print(f"\nRecommendation: {rec}")
    print("=" * 60)


def _build_recommendations(
    egpu_detected: bool,
    jax_gpu_available: bool,
    blocker: str,
    speedup_ratio: float | None,
) -> list[str]:
    """Build a list of actionable recommendations based on detection results.

    **Detailed explanation for engineers:**
        These recommendations are surfaced in the JSON and printed at the end
        of the run so the researcher knows immediately what to do next.
    """
    recs: list[str] = []

    if not egpu_detected:
        if blocker == "thunderbolt_not_connected":
            recs.append(
                "ACTION REQUIRED: Connect the Thunderbolt eGPU chassis with the "
                "RX 7900 XTX, ensure it is powered, then re-run Exp 160."
            )
        elif blocker == "igpu_only_gfx1150":
            recs.append(
                "ACTION REQUIRED: eGPU chassis appears connected but RX 7900 XTX "
                "is not powering on. Check chassis power and GPU seating."
            )
        elif blocker == "driver_issue":
            recs.append(
                "ACTION REQUIRED: ROCm driver issue. Run `sudo rocminfo` and verify "
                "the ROCm stack is installed correctly."
            )
        recs.append(
            "FALLBACK: Run subsequent experiments with CARNOT_SKIP_LLM=1 "
            "(simulation mode) until eGPU is available."
        )
    else:
        if jax_gpu_available:
            recs.append(
                "JAX GPU is working on gfx1100. Remove JAX_PLATFORMS=cpu from "
                "Exp 161+ invocations. Update research-program.md constraints."
            )
        else:
            recs.append(
                "JAX GPU is NOT working even though PyTorch ROCm works. "
                "Install jaxlib with ROCm support: pip install jaxlib-rocm "
                "(or the appropriate wheel from the JAX releases page)."
            )
        if speedup_ratio is not None:
            if speedup_ratio >= 5.0:
                recs.append(
                    f"GPU speedup is {speedup_ratio}× — strong enough to use for "
                    "all live benchmark experiments (Goals #1, #5, #6 unblocked)."
                )
            elif speedup_ratio >= 2.0:
                recs.append(
                    f"GPU speedup is {speedup_ratio}× — modest but real. "
                    "GPU mode is worthwhile for batch benchmarks."
                )
            else:
                recs.append(
                    f"GPU speedup is {speedup_ratio}× — marginal. "
                    "Investigate whether model fits fully in VRAM or is PCIe-bandwidth bound."
                )

    return recs


if __name__ == "__main__":
    main()

"""Experiment 146 — AMD XDNA NPU latency benchmark for JEPAViolationPredictor.

**Researcher summary:**
    Attempts to run the JEPA Tier-3 predictor (Exp 144) on the AMD Ryzen AI
    NPU (XDNA architecture), measures p50/p95/p99 inference latency, and
    compares to CPU-only ONNX inference. Documents environment state and any
    blockers to full NPU execution.

**Detailed explanation for engineers:**
    The AMD Ryzen AI NPU (Neural Processing Unit) is a dedicated inference
    accelerator built into certain Ryzen AI series CPUs. It uses AMD's XDNA
    architecture and is exposed to the OS through the ``amdxdna`` kernel module
    and the ``/dev/accel*`` device nodes.

    To run ML models on this NPU there are two software paths:
    1. **ONNX Runtime with AMDXDNAExecutionProvider** — available in the
       Ryzen AI software stack from AMD (onnxruntime-vitisai or AMD's custom
       build). NOT included in the standard PyPI ``onnxruntime`` package.
    2. **Torch-MLIR / XCL / Vitis AI flow** — compiles models through AMD's
       Vitis AI toolchain targeting the AIE (AI Engine) tiles.

    This experiment:
    a. Detects whether the NPU device node and kernel module are present.
    b. Exports the JEPAViolationPredictor MLP (256→64→32→3) to ONNX format
       using PyTorch (via a lightweight re-implementation) or the onnx library
       directly from the numpy weight arrays.
    c. Tries to create an onnxruntime session with AMDXDNAExecutionProvider.
    d. If unavailable, falls back to CPUExecutionProvider and documents exactly
       what is missing to enable NPU acceleration.
    e. Runs 1000 warmup + 1000 timed inference calls and reports p50/p95/p99.
    f. Saves results to results/experiment_146_npu_results.json.

Spec: REQ-JEPA-001 (Tier 3 predictor), research-program.md §"Next Milestone Focus" #5
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# 0. Session timer
# ---------------------------------------------------------------------------

_t_start = time.time()


def _elapsed() -> float:
    """Return wall-clock seconds since script start."""
    return time.time() - _t_start


# ---------------------------------------------------------------------------
# 1. Environment detection helpers
# ---------------------------------------------------------------------------


def _detect_npu_device() -> dict[str, Any]:
    """Probe the host for NPU hardware and driver availability.

    **Detailed explanation for engineers:**
        Checks three independent signals:
        1. ``lspci | grep -i xdna`` — PCI device listing. An AMD XDNA NPU
           shows up as a PCI function (typically "Signal processing controller").
        2. ``/dev/accel*`` — the Linux DRM accelerator device nodes. The
           ``amdxdna`` driver registers the NPU as an accelerator device (not
           a render node like GPUs) under /dev/accel/.
        3. ``lsmod | grep amdxdna`` — verifies the kernel module is loaded.

    Returns:
        Dict with keys: ``lspci_xdna``, ``dev_accel``, ``amdxdna_loaded``,
        ``kernel_version``.
    """
    info: dict[str, Any] = {}

    # PCI enumeration
    try:
        lspci_out = subprocess.check_output(
            ["lspci"], text=True, stderr=subprocess.DEVNULL
        )
        xdna_lines = [l for l in lspci_out.splitlines() if "xdna" in l.lower()]
        info["lspci_xdna"] = xdna_lines
    except Exception as e:
        info["lspci_xdna"] = f"error: {e}"

    # Accelerator device nodes
    accel_nodes = list(Path("/dev").glob("accel*"))
    info["dev_accel"] = [str(p) for p in accel_nodes]

    # Kernel module
    try:
        lsmod_out = subprocess.check_output(
            ["lsmod"], text=True, stderr=subprocess.DEVNULL
        )
        xdna_mod = [l for l in lsmod_out.splitlines() if "amdxdna" in l]
        info["amdxdna_loaded"] = bool(xdna_mod)
        info["amdxdna_module_lines"] = xdna_mod
    except Exception as e:
        info["amdxdna_loaded"] = False
        info["amdxdna_module_lines"] = [f"error: {e}"]

    # Kernel version
    try:
        info["kernel_version"] = subprocess.check_output(
            ["uname", "-r"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        info["kernel_version"] = "unknown"

    return info


def _detect_onnxruntime_providers() -> dict[str, Any]:
    """Query onnxruntime for available execution providers.

    **Detailed explanation for engineers:**
        The standard PyPI ``onnxruntime`` package ships with:
        - CPUExecutionProvider (always present)
        - AzureExecutionProvider (cloud inference, always present)
        - CUDAExecutionProvider (if CUDA build)
        - TensorrtExecutionProvider (if TensorRT build)

        The ``AMDXDNAExecutionProvider`` is NOT in the standard package.
        It is only available through:
        - AMD Ryzen AI software stack (``onnxruntime-vitisai`` conda package
          from AMD's channel: https://ryzenai.docs.amd.com)
        - Custom AMD build of onnxruntime built with
          ``--use_xnnpack`` or ``--use_vitisai`` flags

    Returns:
        Dict with keys: ``providers`` (list), ``npu_provider_available`` (bool),
        ``onnxruntime_version`` (str).
    """
    try:
        import onnxruntime as ort  # type: ignore[import]

        providers = ort.get_available_providers()
        return {
            "providers": providers,
            "npu_provider_available": "AMDXDNAExecutionProvider" in providers,
            "onnxruntime_version": ort.__version__,
        }
    except ImportError:
        return {
            "providers": [],
            "npu_provider_available": False,
            "onnxruntime_version": "not_installed",
        }


# ---------------------------------------------------------------------------
# 2. ONNX model export
# ---------------------------------------------------------------------------

NPU_RESULTS_DIR = Path(__file__).parent.parent / "results"
WEIGHTS_PATH = NPU_RESULTS_DIR / "jepa_predictor.safetensors"
ONNX_MODEL_PATH = NPU_RESULTS_DIR / "jepa_predictor_146.onnx"

EMBED_DIM = 256
HIDDEN1 = 64
HIDDEN2 = 32
N_DOMAINS = 3


def _export_to_onnx(weights_path: Path, onnx_path: Path) -> str:
    """Export the JEPAViolationPredictor MLP to ONNX format.

    **Detailed explanation for engineers:**
        ONNX (Open Neural Network Exchange) is an open standard format for
        ML models. We build the graph manually using the ``onnx`` library's
        helper functions — no PyTorch or TensorFlow needed — because the
        JEPAViolationPredictor is a simple 3-layer MLP that maps directly to
        standard ONNX operators:

        Graph operators used:
        - ``Gemm`` (General Matrix Multiply): implements y = x @ W^T + b
          (equivalent to a linear layer)
        - ``Relu``: element-wise max(0, x)
        - ``Sigmoid``: element-wise 1/(1+exp(-x))

        The 6 weight tensors (w1,b1,w2,b2,w3,b3) are embedded as initializers
        (constant tensors) in the ONNX graph, so the model is self-contained.

        Input shape: [batch_size, 256] (dynamic batch)
        Output shape: [batch_size, 3] (sigmoid probabilities per domain)

        The exported model is opset 17 (stable, widely supported).

    Args:
        weights_path: Path to jepa_predictor.safetensors.
        onnx_path: Where to write the .onnx file.

    Returns:
        Absolute path to the written ONNX model.

    Raises:
        FileNotFoundError: If weights file doesn't exist.
    """
    import onnx  # type: ignore[import]
    import onnx.helper as oh  # type: ignore[import]
    import onnx.numpy_helper as onh  # type: ignore[import]
    from safetensors.numpy import load_file

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    # Load trained weights
    raw = load_file(str(weights_path))
    w1 = raw["w1"].astype(np.float32)   # (256, 64)
    b1 = raw["b1"].astype(np.float32)   # (64,)
    w2 = raw["w2"].astype(np.float32)   # (64, 32)
    b2 = raw["b2"].astype(np.float32)   # (32,)
    w3 = raw["w3"].astype(np.float32)   # (32, 3)
    b3 = raw["b3"].astype(np.float32)   # (3,)

    # Build ONNX graph nodes
    # Gemm: Y = alpha * A * B + beta * C
    # With transB=1: Y = X @ W^T + b (standard linear layer convention)
    node_gemm1 = oh.make_node(
        "Gemm", inputs=["input", "w1", "b1"], outputs=["gemm1"],
        transB=1, alpha=1.0, beta=1.0,
        name="Gemm1"
    )
    node_relu1 = oh.make_node("Relu", inputs=["gemm1"], outputs=["relu1"], name="Relu1")
    node_gemm2 = oh.make_node(
        "Gemm", inputs=["relu1", "w2", "b2"], outputs=["gemm2"],
        transB=1, alpha=1.0, beta=1.0,
        name="Gemm2"
    )
    node_relu2 = oh.make_node("Relu", inputs=["gemm2"], outputs=["relu2"], name="Relu2")
    node_gemm3 = oh.make_node(
        "Gemm", inputs=["relu2", "w3", "b3"], outputs=["logits"],
        transB=1, alpha=1.0, beta=1.0,
        name="Gemm3"
    )
    node_sigmoid = oh.make_node("Sigmoid", inputs=["logits"], outputs=["probs"], name="Sigmoid1")

    # Initializers (weight tensors embedded in the model)
    init_w1 = onh.from_array(w1.T, name="w1")   # Gemm expects (out, in) for transB=1
    init_b1 = onh.from_array(b1, name="b1")
    init_w2 = onh.from_array(w2.T, name="w2")
    init_b2 = onh.from_array(b2, name="b2")
    init_w3 = onh.from_array(w3.T, name="w3")
    init_b3 = onh.from_array(b3, name="b3")

    # Graph I/O
    # Batch dimension is dynamic (None → marked as "batch_size" string)
    input_tensor = oh.make_tensor_value_info("input", onnx.TensorProto.FLOAT, ["batch_size", EMBED_DIM])
    output_tensor = oh.make_tensor_value_info("probs", onnx.TensorProto.FLOAT, ["batch_size", N_DOMAINS])

    graph = oh.make_graph(
        nodes=[node_gemm1, node_relu1, node_gemm2, node_relu2, node_gemm3, node_sigmoid],
        name="JEPAViolationPredictor",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[init_w1, init_b1, init_w2, init_b2, init_w3, init_b3],
    )

    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])
    model.doc_string = (
        "JEPAViolationPredictor: 3-layer MLP for Tier-3 JEPA constraint "
        "violation prediction. Input: (batch, 256) float32 embeddings. "
        "Output: (batch, 3) float32 sigmoid probabilities for [arithmetic, code, logic] domains."
    )
    model.model_version = 146

    # Validate the graph
    onnx.checker.check_model(model)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(onnx_path))
    print(f"[Exp 146] ONNX model saved → {onnx_path}")
    return str(onnx_path.resolve())


# ---------------------------------------------------------------------------
# 3. Latency benchmark
# ---------------------------------------------------------------------------


def _run_benchmark(
    session,  # onnxruntime.InferenceSession
    n_warmup: int = 100,
    n_trials: int = 1000,
) -> dict[str, float]:
    """Run repeated single-sample inference and report latency percentiles.

    **Detailed explanation for engineers:**
        Latency measurement methodology:
        - Single sample (batch_size=1) inference to reflect real-time
          per-token prediction use case (not bulk throughput).
        - n_warmup calls first: these let the ONNX runtime compile/cache
          kernels, prime caches, and stabilize JIT state. These times are
          discarded.
        - n_trials timed calls using time.perf_counter() (nanosecond
          resolution on Linux). Each call generates a fresh random input
          to avoid caching artifacts.
        - Percentiles computed with numpy: p50 (median), p95, p99.

        **Why single-sample?** The JEPA predictor is called during generation,
        once per partial response. Batch latency is less relevant than
        single-request p99 (tail latency matters for interactive systems).

    Args:
        session: onnxruntime InferenceSession (CPU or NPU backend).
        n_warmup: Number of calls to discard before timing. Default 100.
        n_trials: Number of timed calls. Default 1000.

    Returns:
        Dict with keys ``p50_ms``, ``p95_ms``, ``p99_ms``, ``mean_ms``,
        ``min_ms``, ``max_ms``.
    """
    input_name = session.get_inputs()[0].name
    rng = np.random.default_rng(42)

    # Warmup
    for _ in range(n_warmup):
        x = rng.standard_normal((1, EMBED_DIM)).astype(np.float32)
        session.run(None, {input_name: x})

    # Timed trials
    latencies_ms: list[float] = []
    for _ in range(n_trials):
        x = rng.standard_normal((1, EMBED_DIM)).astype(np.float32)
        t0 = time.perf_counter()
        session.run(None, {input_name: x})
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    arr = np.array(latencies_ms)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(np.mean(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
    }


# ---------------------------------------------------------------------------
# 4. Main experiment
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full NPU detection + benchmark + results-saving pipeline."""
    import onnxruntime as ort  # type: ignore[import]

    print("=" * 70)
    print("Experiment 146: AMD XDNA NPU latency benchmark")
    print("=" * 70)

    # ---- Step 1: Environment detection ----
    print("\n[Step 1] Detecting NPU hardware and software environment...")
    npu_hw = _detect_npu_device()
    onnx_info = _detect_onnxruntime_providers()

    hw_present = bool(npu_hw.get("dev_accel")) or npu_hw.get("amdxdna_loaded", False)
    npu_sw_available = onnx_info["npu_provider_available"]
    npu_available = hw_present and npu_sw_available

    print(f"  Kernel version : {npu_hw['kernel_version']}")
    print(f"  /dev/accel*    : {npu_hw['dev_accel']}")
    print(f"  amdxdna module : {npu_hw['amdxdna_loaded']} — {npu_hw['amdxdna_module_lines']}")
    print(f"  ONNX providers : {onnx_info['providers']}")
    print(f"  NPU HW present : {hw_present}")
    print(f"  NPU SW ready   : {npu_sw_available}")
    print(f"  NPU available  : {npu_available}")

    if hw_present and not npu_sw_available:
        print(
            "\n  [!] NPU HARDWARE IS PRESENT but AMDXDNAExecutionProvider is MISSING.\n"
            "      The standard PyPI onnxruntime does not include this provider.\n"
            "      To enable NPU acceleration, install AMD's Ryzen AI software stack:\n"
            "\n"
            "        # Option A: conda (recommended by AMD)\n"
            "        conda install -c amd onnxruntime-vitisai\n"
            "\n"
            "        # Option B: pip from AMD's index (if available)\n"
            "        pip install --index-url https://download.amd.com/ryzenai/wheels \\\n"
            "            onnxruntime-vitisai\n"
            "\n"
            "      After installation, AMDXDNAExecutionProvider should appear in\n"
            "      onnxruntime.get_available_providers() and this experiment can\n"
            "      be re-run to get real NPU latency measurements.\n"
            "\n"
            "      References:\n"
            "        https://ryzenai.docs.amd.com/en/latest/inst.html\n"
            "        https://github.com/amd/RyzenAI-SW"
        )

    # ---- Step 2: Export ONNX model ----
    print("\n[Step 2] Exporting JEPAViolationPredictor to ONNX...")
    onnx_model_path = _export_to_onnx(WEIGHTS_PATH, ONNX_MODEL_PATH)

    # ---- Step 3: CPU benchmark (always runs) ----
    print("\n[Step 3] CPU benchmark (CPUExecutionProvider)...")
    cpu_session = ort.InferenceSession(
        onnx_model_path,
        providers=["CPUExecutionProvider"],
    )
    cpu_stats = _run_benchmark(cpu_session, n_warmup=100, n_trials=1000)
    print(f"  CPU p50={cpu_stats['p50_ms']:.3f}ms  "
          f"p95={cpu_stats['p95_ms']:.3f}ms  "
          f"p99={cpu_stats['p99_ms']:.3f}ms")

    # ---- Step 4: NPU benchmark (conditional) ----
    npu_stats: dict[str, Any] = {}
    speedup: float | None = None

    if npu_available:
        print("\n[Step 4] NPU benchmark (AMDXDNAExecutionProvider)...")
        npu_session = ort.InferenceSession(
            onnx_model_path,
            providers=["AMDXDNAExecutionProvider", "CPUExecutionProvider"],
        )
        npu_stats = _run_benchmark(npu_session, n_warmup=100, n_trials=1000)
        speedup = cpu_stats["p50_ms"] / npu_stats["p50_ms"] if npu_stats["p50_ms"] > 0 else None
        print(f"  NPU p50={npu_stats['p50_ms']:.3f}ms  "
              f"p95={npu_stats['p95_ms']:.3f}ms  "
              f"p99={npu_stats['p99_ms']:.3f}ms")
        print(f"  Speedup (p50): {speedup:.2f}x" if speedup else "  Speedup: N/A")
        npu_p99_target_met = npu_stats["p99_ms"] < 1.0
        print(f"  Target (<1ms p99): {'PASS' if npu_p99_target_met else 'FAIL'}")
    else:
        print("\n[Step 4] Skipping NPU benchmark — AMDXDNAExecutionProvider unavailable.")
        npu_p99_target_met = None

    # ---- Step 5: Assemble results ----
    blocker_note = ""
    if hw_present and not npu_sw_available:
        blocker_note = (
            "NPU hardware present (amdxdna module loaded, /dev/accel0 exists) "
            "but AMDXDNAExecutionProvider is absent from onnxruntime. "
            "Requires AMD Ryzen AI software stack: "
            "conda install -c amd onnxruntime-vitisai"
        )

    results: dict[str, Any] = {
        "experiment": 146,
        "title": "JEPA Predictor NPU Latency Benchmark",
        "spec_refs": ["REQ-JEPA-001"],
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(_elapsed(), 2),
        # Hardware detection
        "npu_hw_present": hw_present,
        "npu_sw_available": npu_sw_available,
        "npu_available": npu_available,
        "device_info": {
            "kernel_version": npu_hw["kernel_version"],
            "dev_accel": npu_hw["dev_accel"],
            "amdxdna_loaded": npu_hw["amdxdna_loaded"],
            "amdxdna_module_lines": npu_hw["amdxdna_module_lines"],
            "lspci_xdna": npu_hw["lspci_xdna"],
            "onnxruntime_version": onnx_info["onnxruntime_version"],
            "providers": onnx_info["providers"],
        },
        "blocker": blocker_note,
        # Model export
        "onnx_model_path": onnx_model_path,
        # CPU benchmark
        "cpu_benchmark": cpu_stats,
        "cpu_p50_ms": cpu_stats["p50_ms"],
        "cpu_p99_ms": cpu_stats["p99_ms"],
        # NPU benchmark (null if not run)
        "npu_benchmark": npu_stats if npu_available else None,
        "npu_p50_ms": npu_stats.get("p50_ms") if npu_available else None,
        "npu_p99_ms": npu_stats.get("p99_ms") if npu_available else None,
        "speedup": speedup,
        "npu_p99_target_met": npu_p99_target_met,
        "target_npu_p99_ms": 1.0,
        "conclusion": (
            "NPU hardware detected and kernel module active, but ONNX "
            "AMDXDNAExecutionProvider not available in standard onnxruntime. "
            "CPU ONNX baseline established. NPU path blocked until AMD Ryzen "
            "AI software stack installed."
            if (hw_present and not npu_sw_available)
            else (
                "NPU acceleration active — see npu_benchmark for results."
                if npu_available
                else "Neither NPU hardware nor software available. CPU baseline only."
            )
        ),
    }

    # ---- Step 6: Save results ----
    out_path = NPU_RESULTS_DIR / "experiment_146_npu_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Exp 146] Results saved → {out_path}")

    print("\n=== SUMMARY ===")
    print(f"  NPU hardware present : {hw_present}")
    print(f"  NPU SW available     : {npu_sw_available}")
    print(f"  ONNX model exported  : {onnx_model_path}")
    print(f"  CPU p50              : {cpu_stats['p50_ms']:.3f} ms")
    print(f"  CPU p99              : {cpu_stats['p99_ms']:.3f} ms")
    if npu_available:
        print(f"  NPU p50              : {npu_stats['p50_ms']:.3f} ms")
        print(f"  NPU p99              : {npu_stats['p99_ms']:.3f} ms")
        print(f"  Speedup              : {speedup:.2f}x")
    else:
        print("  NPU p50              : N/A (provider unavailable)")
        if hw_present:
            print("  Blocker              : AMDXDNAExecutionProvider missing from onnxruntime")
            print("  Fix                  : conda install -c amd onnxruntime-vitisai")


if __name__ == "__main__":
    main()

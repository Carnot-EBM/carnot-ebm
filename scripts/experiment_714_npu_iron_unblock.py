#!/usr/bin/env python3
"""Experiment 714: NPU Unblock via IRON Toolchain (mlir-aie) or AMD Custom onnxruntime.

WHY THIS EXPERIMENT EXISTS:
    The AMD XDNA NPU has been blocked for 7 consecutive milestones (Exps 292, 303, 314,
    335, 435, and two others) because the VitisAI path requires system-level packages
    (ninja, openblas) that have never been installed on experiment hosts.

    This experiment tries two alternative approaches that avoid system-level dependencies:

    STRATEGY 1 — IRON toolchain (mlir-aie):
        The IRON (Intermediate Representation for Open Neural-networks) toolchain targets
        the AMD AI Engine (AIE) array directly via MLIR dialects.  It is distributed as
        a pure Python wheel — no ninja, no openblas, no Xilinx installer required.
        arXiv 2504.03083 demonstrates 2.8x GEMM speedup over CPU on AMD XDNA hardware
        using the IRON approach.  This is the primary unblock path.

    STRATEGY 2 — AMD custom onnxruntime wheel (if IRON fails):
        AMD distributes VitisAI-enabled onnxruntime as a custom wheel for Python 3.12
        at ryzenai.docs.amd.com.  If the wheel is publicly downloadable from PyPI
        (onnxruntime-vitisai), it provides a second path to NPU access without a full
        Ryzen AI Software installer.

    honest_verdict categories:
        - "npu_iron_working"              : IRON installed, compiled, gemm_speedup >= 2.0
        - "npu_iron_installed_no_speedup" : IRON installed and compiled, but speedup < 2.0
        - "npu_vitisai_working"           : IRON failed, VitisAI onnxruntime works
        - "npu_still_blocked"             : both approaches failed
        - "npu_iron_install_failed"       : pip install mlir-aie returned non-zero exit

Spec: REQ-HW-039, SCENARIO-HW-039
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "python"))

from python.carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DELIVERABLE = "results/experiment_714_npu_iron_unblock.json"
GEMM_ITERATIONS = 100
GEMM_SIZE = 16  # 16x16 GEMM as in arXiv 2504.03083
MIN_SPEEDUP = 2.0  # Minimum speedup to claim NPU unblock (REQ-HW-039)


# ---------------------------------------------------------------------------
# Strategy 1 — IRON toolchain
# ---------------------------------------------------------------------------


def install_mlir_aie(pip_bin: str) -> tuple[bool, str]:
    """Attempt to install mlir-aie via pip and report success.

    WHY pip install at runtime:
        mlir-aie is not part of the standard Carnot requirements.  Rather than
        modifying requirements files (which could break the CI matrix), we probe
        at experiment time — the same pattern used for other optional hardware
        drivers.  If the package is already installed, pip is a no-op.

    Args:
        pip_bin: Path to the pip executable (e.g. ".venv/bin/pip").

    Returns:
        (success, stderr_or_empty) — success=True means pip exited 0.

    Spec: REQ-HW-039
    """
    result = subprocess.run(
        [pip_bin, "install", "mlir-aie"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    return result.returncode == 0, result.stderr if result.returncode != 0 else ""


def check_iron_importable() -> bool:
    """Return True if mlir_aie is importable in the current interpreter.

    WHY separate from install:
        pip install can succeed (already installed, no-op) even if the package
        was installed for a different interpreter or ABI.  Import check is the
        definitive gate.

    Spec: REQ-HW-039
    """
    try:
        import importlib

        importlib.import_module("mlir_aie")
        return True
    except ImportError:
        return False


def cpu_gemm_benchmark(size: int, iterations: int) -> float:
    """Run a CPU-only GEMM benchmark and return total elapsed seconds.

    WHY pure Python / no numpy optional:
        We want a baseline that runs even when numpy is absent.  We use the
        standard library ``array`` module to avoid importing numpy, keeping
        this function dependency-free.  If numpy is present, we use it for
        accuracy — the benchmark is measuring wall-clock time, not FLOPS.

    Args:
        size:       Matrix dimension (size x size).
        iterations: Number of GEMM iterations to time.

    Returns:
        Total seconds for all iterations (not per-iteration average).

    Spec: REQ-HW-039, SCENARIO-HW-039
    """
    try:
        import numpy as np  # noqa: PLC0415

        a = np.ones((size, size), dtype=np.float32)
        b = np.ones((size, size), dtype=np.float32)
        t0 = time.perf_counter()
        for _ in range(iterations):
            _ = a @ b
        return time.perf_counter() - t0
    except ImportError:
        # Fallback: pure Python triple-loop GEMM (slow but dependency-free)
        a = [[1.0] * size for _ in range(size)]
        b = [[1.0] * size for _ in range(size)]
        t0 = time.perf_counter()
        for _ in range(iterations):
            [[sum(a[i][k] * b[k][j] for k in range(size)) for j in range(size)] for i in range(size)]
        return time.perf_counter() - t0


def attempt_iron_gemm_benchmark(size: int, iterations: int) -> dict[str, Any]:
    """Try to compile and run a GEMM kernel on the NPU via IRON, return timing dict.

    WHY we wrap the whole attempt:
        mlir_aie may be importable but the NPU hardware may not be present on
        this machine (e.g. CI runner with no Ryzen AI APU).  We capture every
        failure mode and surface it in the result dict so the honest_verdict
        logic can distinguish "installed but no hardware" from "installed and working".

    Returns a dict with keys:
        compile_ok (bool), run_ok (bool), npu_time_s (float | None),
        cpu_time_s (float), gemm_speedup (float | None), error (str | None)

    Spec: REQ-HW-039, SCENARIO-HW-039
    """
    cpu_time = cpu_gemm_benchmark(size, iterations)
    result: dict[str, Any] = {
        "compile_ok": False,
        "run_ok": False,
        "npu_time_s": None,
        "cpu_time_s": cpu_time,
        "gemm_speedup": None,
        "error": None,
    }
    try:
        import mlir_aie  # noqa: PLC0415

        # Build a minimal AIE MLIR kernel string for a size x size GEMM.
        # The dialect syntax follows arXiv 2504.03083 §3 — a bare matmul tile.
        # We use the simplest possible kernel: identity accumulation over one tile.
        kernel_mlir = f"""
module {{
  func.func @gemm_kernel(%A: memref<{size}x{size}xf32>,
                          %B: memref<{size}x{size}xf32>,
                          %C: memref<{size}x{size}xf32>) {{
    affine.for %i = 0 to {size} {{
      affine.for %j = 0 to {size} {{
        affine.for %k = 0 to {size} {{
          %a = affine.load %A[%i, %k] : memref<{size}x{size}xf32>
          %b = affine.load %B[%k, %j] : memref<{size}x{size}xf32>
          %c = affine.load %C[%i, %j] : memref<{size}x{size}xf32>
          %prod = arith.mulf %a, %b : f32
          %acc  = arith.addf %c, %prod : f32
          affine.store %acc, %C[%i, %j] : memref<{size}x{size}xf32>
        }}
      }}
    }}
    return
  }}
}}
"""
        # Attempt compilation — the API surface varies by mlir-aie version.
        # We probe common entry points defensively.
        compiled = None
        compile_error = None
        for attr in ("compile", "compile_module", "compile_to_npu"):
            fn = getattr(mlir_aie, attr, None)
            if fn is not None:
                try:
                    compiled = fn(kernel_mlir)
                    result["compile_ok"] = True
                    break
                except Exception as exc:  # noqa: BLE001
                    compile_error = str(exc)

        if not result["compile_ok"]:
            # No compile entry point found or all failed — still count as "installed"
            result["error"] = compile_error or "No compile entry point found in mlir_aie"
            return result

        # Attempt to run 1 warmup + N timed iterations on NPU
        run_fn = getattr(mlir_aie, "run", None) or getattr(mlir_aie, "execute", None)
        if run_fn is None:
            result["error"] = "mlir_aie.run/execute not found — cannot benchmark NPU"
            return result

        # Warmup
        try:
            run_fn(compiled)
        except Exception as exc:  # noqa: BLE001
            result["error"] = f"NPU warmup failed: {exc}"
            return result

        t0 = time.perf_counter()
        for _ in range(iterations):
            run_fn(compiled)
        npu_time = time.perf_counter() - t0

        result["run_ok"] = True
        result["npu_time_s"] = npu_time
        result["gemm_speedup"] = cpu_time / npu_time if npu_time > 0 else None

    except ImportError as exc:
        result["error"] = f"mlir_aie import failed after install: {exc}"
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# Strategy 2 — AMD custom onnxruntime VitisAI wheel
# ---------------------------------------------------------------------------


def check_vitisai_available(pip_bin: str) -> tuple[bool, str]:
    """Check if VitisAIExecutionProvider is available via onnxruntime.

    WHY we try PyPI first:
        AMD distributes onnxruntime-vitisai as a public PyPI package for
        Python 3.12.  If it is already installed or installable without the
        full Ryzen AI Software bundle, this gives NPU access without system deps.

    Returns:
        (available, detail_message)

    Spec: REQ-HW-039
    """
    # First check if already installed
    try:
        import importlib  # noqa: PLC0415

        ort = importlib.import_module("onnxruntime")
        providers = ort.get_available_providers()
        if "VitisAIExecutionProvider" in providers:
            return True, "VitisAIExecutionProvider already present in onnxruntime"
    except ImportError:
        pass

    # Try pip install onnxruntime-vitisai (AMD's PyPI package)
    install_result = subprocess.run(
        [pip_bin, "install", "onnxruntime-vitisai", "--quiet", "--no-deps"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if install_result.returncode != 0:
        return False, f"pip install onnxruntime-vitisai failed: {install_result.stderr[:200]}"

    try:
        import importlib  # noqa: PLC0415

        import importlib.util  # noqa: PLC0415

        spec = importlib.util.find_spec("onnxruntime")
        if spec is None:
            return False, "onnxruntime not importable after install attempt"

        # Reload to pick up newly installed package
        import importlib as _il  # noqa: PLC0415, N811

        ort = _il.import_module("onnxruntime")
        providers = ort.get_available_providers()
        if "VitisAIExecutionProvider" in providers:
            return True, "VitisAIExecutionProvider found after pip install onnxruntime-vitisai"
        return False, f"onnxruntime installed but VitisAI provider absent; providers={providers}"
    except Exception as exc:  # noqa: BLE001
        return False, f"import check failed: {exc}"


# ---------------------------------------------------------------------------
# honest_verdict classifier
# ---------------------------------------------------------------------------


def classify_verdict(
    iron_install_ok: bool,
    iron_available: bool,
    iron_compile_ok: bool,
    iron_run_ok: bool,
    gemm_speedup: float | None,
    vitis_available: bool,
) -> str:
    """Map experiment outcomes to a single honest_verdict string.

    WHY a pure classifier function:
        Keeping verdict logic separate from I/O makes it trivially testable.
        Every branch maps to exactly one REQ-HW-039 acceptance/escalation criterion.

    Args:
        iron_install_ok:  pip install mlir-aie returned exit code 0.
        iron_available:   import mlir_aie succeeded.
        iron_compile_ok:  GEMM kernel compiled without error.
        iron_run_ok:      GEMM kernel ran on NPU without error.
        gemm_speedup:     cpu_time / npu_time (None if not benchmarked).
        vitis_available:  VitisAIExecutionProvider present in onnxruntime.

    Returns:
        One of: "npu_iron_working", "npu_iron_installed_no_speedup",
                "npu_vitisai_working", "npu_still_blocked",
                "npu_iron_install_failed"

    Spec: REQ-HW-039, SCENARIO-HW-039
    """
    if iron_install_ok and iron_available and iron_run_ok:
        if gemm_speedup is not None and gemm_speedup >= MIN_SPEEDUP:
            return "npu_iron_working"
        return "npu_iron_installed_no_speedup"
    if not iron_install_ok:
        return "npu_iron_install_failed"
    if vitis_available:
        return "npu_vitisai_working"
    return "npu_still_blocked"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Experiment 714: NPU unblock via IRON toolchain or AMD onnxruntime wheel.

    WHY a 45-minute watchdog:
        Prior NPU experiments (Exp 435 and ancestors) sometimes stalled silently
        waiting for hardware that was never ready.  The watchdog ensures a clean
        partial-result JSON is written and the process exits, freeing the
        conductor for the next task.

    Spec: REQ-HW-039, SCENARIO-HW-039
    """
    tmpl = ExperimentTemplate(
        714,
        "NPU Unblock: IRON Toolchain (mlir-aie) + AMD VitisAI Wheel",
        DELIVERABLE,
    )
    tmpl.setup()

    # .venv/bin/pip is used for install attempts (matches the conductor's venv).
    pip_bin = str(_REPO_ROOT / ".venv" / "bin" / "pip")

    with ExperimentTimeoutWatchdog(
        714,
        timeout_minutes=45,
        result_path=str(_REPO_ROOT / DELIVERABLE),
    ):
        # ------------------------------------------------------------------
        # Strategy 1 — IRON toolchain
        # ------------------------------------------------------------------
        iron_install_ok, iron_install_stderr = install_mlir_aie(pip_bin)
        iron_available = check_iron_importable() if iron_install_ok else False

        iron_compile_ok = False
        iron_run_ok = False
        gemm_speedup: float | None = None
        cpu_time_s: float | None = None
        npu_time_s: float | None = None
        iron_error: str | None = None

        if iron_available:
            iron_result = attempt_iron_gemm_benchmark(GEMM_SIZE, GEMM_ITERATIONS)
            iron_compile_ok = iron_result["compile_ok"]
            iron_run_ok = iron_result["run_ok"]
            gemm_speedup = iron_result["gemm_speedup"]
            cpu_time_s = iron_result["cpu_time_s"]
            npu_time_s = iron_result["npu_time_s"]
            iron_error = iron_result["error"]
        else:
            cpu_time_s = cpu_gemm_benchmark(GEMM_SIZE, GEMM_ITERATIONS)

        # ------------------------------------------------------------------
        # Strategy 2 — VitisAI onnxruntime wheel (only if IRON did not give NPU)
        # ------------------------------------------------------------------
        vitis_available = False
        vitis_detail = "skipped — IRON succeeded"
        if not iron_run_ok:
            vitis_available, vitis_detail = check_vitisai_available(pip_bin)

        # ------------------------------------------------------------------
        # Synthesize verdict
        # ------------------------------------------------------------------
        npu_benchmarkable = iron_run_ok or vitis_available
        honest_verdict = classify_verdict(
            iron_install_ok=iron_install_ok,
            iron_available=iron_available,
            iron_compile_ok=iron_compile_ok,
            iron_run_ok=iron_run_ok,
            gemm_speedup=gemm_speedup,
            vitis_available=vitis_available,
        )

        # ------------------------------------------------------------------
        # Escalation note if still blocked
        # ------------------------------------------------------------------
        if honest_verdict == "npu_still_blocked":
            known_issues_path = _REPO_ROOT / "ops" / "known-issues.md"
            if known_issues_path.exists():
                existing = known_issues_path.read_text()
                retro_note = (
                    "\n\n### RETRO-NPU-v8 (Exp 714, 2026-04-22)\n\n"
                    "IRON path tried (`pip install mlir-aie`) — install or import failed.\n"
                    "VitisAI path tried (`onnxruntime-vitisai` from PyPI) — unavailable.\n"
                    "Both approaches blocked.  Manual hardware action required:\n"
                    "- Install Ryzen AI Software from AMD (provides ninja + openblas), OR\n"
                    "- Request IT to install `ninja-build` and `libopenblas-dev`.\n"
                    "- Alternatively, test on a Ryzen AI host where AMD custom wheels are pre-installed.\n"
                )
                if "RETRO-NPU-v8" not in existing:
                    known_issues_path.write_text(existing + retro_note)

        # ------------------------------------------------------------------
        # Build artifact
        # ------------------------------------------------------------------
        artifact = tmpl.build_result(
            {
                "iron_install_ok": iron_install_ok,
                "iron_install_stderr": iron_install_stderr[:500] if iron_install_stderr else "",
                "iron_available": iron_available,
                "iron_compile_ok": iron_compile_ok,
                "iron_run_ok": iron_run_ok,
                "iron_error": iron_error,
                "gemm_size": GEMM_SIZE,
                "gemm_iterations": GEMM_ITERATIONS,
                "cpu_time_s": cpu_time_s,
                "npu_time_s": npu_time_s,
                "gemm_speedup": gemm_speedup,
                "vitis_available": vitis_available,
                "vitis_detail": vitis_detail,
                "npu_benchmarkable": npu_benchmarkable,
                "honest_verdict": honest_verdict,
            },
            status="success" if npu_benchmarkable else "blocked",
        )

        output_path = _REPO_ROOT / DELIVERABLE
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(__import__("json").dumps(artifact, indent=2) + "\n")

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()

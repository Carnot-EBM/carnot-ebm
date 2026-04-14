#!/usr/bin/env python3
"""Experiment 303: AMD XDNA NPU unblock — install prereqs, source build, benchmark.

This experiment picks up where Exp 292 was blocked: missing `ninja` and `openblas`
prevented the onnxruntime 1.20.1 source build with the VitisAI Execution Provider.

Execution flow:
  1. [PREREQ CHECK] Detect ninja, openblas, cmake, RyzenAI-SW presence.
     If any required item is missing, emit honest_verdict="blocked_prereq" and stop.
  2. [SOURCE BUILD] Clone onnxruntime 1.20.1 to /tmp/ort_build_303 if not present.
     Run cmake + make with -DONNXRUNTIME_USE_VITISAI=ON. Hard timeout: 45 min.
     If build fails or times out, emit honest_verdict="blocked_build" and stop.
  3. [INSTALL WHEEL] Install the freshly built .whl into .venv-npu.
  4. [INFERENCE TEST] Load jepa_predictor_291.onnx (fallback: 146.onnx) via
     VitisAIExecutionProvider. Run 100 timed calls on NPU and 100 on CPU.
     Record npu_latency_us, cpu_latency_us, speedup_factor, provider_used.
     If VitisAI EP is not available after the source build, emit "blocked_abi".
  5. Emit results/experiment_303_npu_results.json with honest_verdict.
  6. Update research-hardware-wishlist.md AMD XDNA section with findings.

CPU ORT baseline from Exp 257: 5.847 µs/call

Writes:
    results/experiment_303_npu_results.json

Spec: REQ-PRED-003
SCENARIO-EXP303-A (prereq check — ninja and openblas detection with install_command)
SCENARIO-EXP303-B (source build path — attempt with 45-min timeout, log tail on failure)
SCENARIO-EXP303-C (inference benchmark — npu_latency_us vs cpu_latency_us when working)
SCENARIO-EXP303-D (honest labeling — null inference_result on all blocked paths)

Run:
    cd /home/ianblenke/github.com/ianblenke/carnot
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_303_npu_unblock.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT: int = 303
RUN_DATE: str = "20260414"

# CPU ORT baseline from Exp 257 (onnx_cpu record, latency_us = 5.847)
CPU_ORT_BASELINE_US: float = 5.847

# Source build timeout: 45 minutes
BUILD_TIMEOUT_SECONDS: int = 45 * 60

# Benchmark calls for inference test
TIMED_CALLS: int = 100
WARMUP_CALLS: int = 20

# ORT version that matches the pre-built AMD .so files
ORT_TARGET_VERSION: str = "1.20.1"

# Repo layout
_REPO_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_DIR = _REPO_ROOT / "results"
_OUTPUT_FILE = _RESULTS_DIR / "experiment_303_npu_results.json"
_VENV_NPU = _REPO_ROOT / ".venv-npu"
_WISHLIST_MD = _REPO_ROOT / "research-hardware-wishlist.md"

# RyzenAI-SW paths
_RYZEN_AI_SW = Path.home() / "github.com" / "amd" / "RyzenAI-SW"
_VITISAI_SO_DIR = (
    _RYZEN_AI_SW / "Ryzen-AI-CVML-Library" / "linux" / "onnx" / "ryzen14"
)
_VAIP_CONFIG = _VITISAI_SO_DIR / "vaip_config_npu_2_3.json"

# ONNX model selection
_ONNX_291 = _RESULTS_DIR / "jepa_predictor_291.onnx"
_ONNX_146 = _RESULTS_DIR / "jepa_predictor_146.onnx"

# ORT source build directory
_BUILD_DIR = Path("/tmp/ort_build_303")

# ORT git tag matching 1.20.1
ORT_GIT_TAG = "v1.20.1"
ORT_GIT_URL = "https://github.com/microsoft/onnxruntime.git"


# ---------------------------------------------------------------------------
# Prereq detection helpers (reused from Exp 292 logic, expanded)
# ---------------------------------------------------------------------------


def _cmake_version() -> tuple[int, int, int] | None:
    """Return cmake version as (major, minor, patch) or None if not found.

    cmake 3.26+ is required for the ORT source build.
    """
    try:
        r = subprocess.run(
            ["cmake", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        match = re.search(r"cmake version (\d+)\.(\d+)\.(\d+)", r.stdout)
        if match:
            return int(match.group(1)), int(match.group(2)), int(match.group(3))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _ninja_available() -> bool:
    """Return True if `ninja` or `ninja-build` is on PATH."""
    return shutil.which("ninja") is not None or shutil.which("ninja-build") is not None


def _openblas_available() -> bool:
    """Return True if openblas is detectable via pkg-config, ldconfig, or filesystem.

    Checks in order:
      1. pkg-config --modversion openblas
      2. ldconfig -p | grep libopenblas
      3. Known .so paths in common system directories
    """
    # pkg-config
    try:
        r = subprocess.run(
            ["pkg-config", "--modversion", "openblas"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode == 0:
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # ldconfig -p
    try:
        r = subprocess.run(
            ["ldconfig", "-p"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if "libopenblas" in r.stdout.lower():
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Filesystem check
    for candidate in [
        "/usr/lib/libopenblas.so",
        "/usr/lib/x86_64-linux-gnu/libopenblas.so",
        "/usr/local/lib/libopenblas.so",
        "/usr/lib/libopenblas.so.0",
        "/usr/lib/x86_64-linux-gnu/libopenblas.so.0",
    ]:
        if Path(candidate).exists():
            return True
    return False


def _collect_prereq_check() -> dict[str, Any]:
    """Collect all prereq detection results into a dict.

    Returns a dict with:
      ninja_installed (bool)          — ninja or ninja-build on PATH
      ninja_install_command (str)     — present only when ninja_installed=False
      openblas_installed (bool)       — openblas detectable
      openblas_install_command (str)  — present only when openblas_installed=False
      cmake_version (str|None)        — e.g. "4.3.1"
      cmake_sufficient (bool)         — cmake >= 3.26
      ryzen_ai_sw_present (bool)      — RyzenAI-SW checkout present
      vitisai_so_present (bool)       — libonnxruntime_providers_vitisai.so present
      vaip_config_present (bool)      — vaip_config_npu_2_3.json present
      venv_npu_present (bool)         — .venv-npu Python venv present
    """
    ninja_ok = _ninja_available()
    openblas_ok = _openblas_available()
    cmake_ver = _cmake_version()
    cmake_str = ".".join(str(x) for x in cmake_ver) if cmake_ver else None
    cmake_ok = cmake_ver is not None and cmake_ver >= (3, 26, 0)
    vitisai_so = _VITISAI_SO_DIR / "libonnxruntime_providers_vitisai.so"

    result: dict[str, Any] = {
        "ninja_installed": ninja_ok,
        "openblas_installed": openblas_ok,
        "cmake_version": cmake_str,
        "cmake_sufficient": cmake_ok,
        "ryzen_ai_sw_present": _RYZEN_AI_SW.is_dir(),
        "vitisai_so_present": vitisai_so.exists(),
        "vaip_config_present": _VAIP_CONFIG.exists(),
        "venv_npu_present": _VENV_NPU.is_dir(),
    }

    # Explain how to fix each missing item (SCENARIO-EXP303-A)
    if not ninja_ok:
        result["ninja_install_command"] = (
            "sudo pacman -S ninja  (Arch)  OR  sudo apt install ninja-build  (Debian/Ubuntu)"
        )
    if not openblas_ok:
        result["openblas_install_command"] = (
            "sudo pacman -S openblas  (Arch)  OR  sudo apt install libopenblas-dev  (Debian/Ubuntu)"
        )
    if not cmake_ok:
        if cmake_ver is None:
            result["cmake_install_command"] = (
                "sudo pacman -S cmake  OR  sudo apt install cmake"
            )
        else:
            result["cmake_note"] = (
                f"cmake {cmake_str} is too old — need >= 3.26. "
                "Upgrade via package manager."
            )

    return result


def _select_onnx_model() -> Path | None:
    """Return the best ONNX model path: prefer jepa_predictor_291, fall back to 146."""
    if _ONNX_291.exists():
        return _ONNX_291
    if _ONNX_146.exists():
        return _ONNX_146
    return None


# ---------------------------------------------------------------------------
# Source build path (SCENARIO-EXP303-B)
# ---------------------------------------------------------------------------


def _attempt_source_build() -> dict[str, Any]:
    """Attempt to build onnxruntime 1.20.1 from source with VitisAI EP enabled.

    Steps:
      1. Clone onnxruntime at tag v1.20.1 to _BUILD_DIR (skipped if already present).
      2. Run cmake -DONNXRUNTIME_USE_VITISAI=ON with XRT and VitisAI paths.
      3. Run cmake --build with a 45-minute hard timeout.
      4. If successful, locate and return path to the built .whl file.

    Returns a build_outcome dict with:
      success (bool)
      duration_seconds (float)
      whl_path (str|None)       — path to .whl if success=True
      error_summary (str)       — present when success=False
      build_log_tail (list[str]) — last 50 build output lines when success=False
      timeout_exceeded (bool)   — True if the 45-min limit was hit
    """
    venv_python = _VENV_NPU / "bin" / "python"
    vitisai_so = _VITISAI_SO_DIR / "libonnxruntime_providers_vitisai.so"

    build_log_lines: list[str] = []
    start_time = time.monotonic()

    # Step 1: Clone if needed
    if not _BUILD_DIR.exists():
        print(f"  Cloning onnxruntime {ORT_GIT_TAG} to {_BUILD_DIR} ...")
        clone_result = subprocess.run(
            [
                "git", "clone",
                "--depth", "1",
                "--branch", ORT_GIT_TAG,
                ORT_GIT_URL,
                str(_BUILD_DIR),
            ],
            capture_output=True,
            text=True,
            timeout=600,  # 10 min for clone
        )
        if clone_result.returncode != 0:
            elapsed = time.monotonic() - start_time
            err = (clone_result.stderr or clone_result.stdout)[:500]
            return {
                "success": False,
                "duration_seconds": round(elapsed, 1),
                "error_summary": f"git clone failed: {err}",
                "build_log_tail": [line for line in err.splitlines()[-50:] if line],
                "timeout_exceeded": False,
                "whl_path": None,
            }
        build_log_lines.append(f"git clone succeeded for {ORT_GIT_TAG}")
    else:
        build_log_lines.append(f"Using existing clone at {_BUILD_DIR}")

    # Step 2: Create build subdirectory
    cmake_build_dir = _BUILD_DIR / "build_vitisai"
    cmake_build_dir.mkdir(exist_ok=True)

    # Step 3: cmake configure
    xrt_include = str(Path("/opt/xilinx/xrt/include"))
    xrt_lib = str(Path("/opt/xilinx/xrt/lib"))
    vitisai_ep_lib = str(_VITISAI_SO_DIR)

    cmake_cmd = [
        "cmake",
        str(_BUILD_DIR),
        "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DONNXRUNTIME_USE_VITISAI=ON",
        f"-DONNXRUNTIME_VITISAI_EP_LIBRARY_PATH={vitisai_ep_lib}",
        f"-DXRT_ROOT=/opt/xilinx/xrt",
        f"-DXRT_INCLUDE_DIR={xrt_include}",
        f"-DXRT_LIB_DIR={xrt_lib}",
        "-Donnxruntime_USE_OPENBLAS=ON",
        "-Donnxruntime_BUILD_WHEEL=ON",
        f"-DPYTHON_EXECUTABLE={venv_python}",
    ]

    print("  Running cmake configure ...")
    configure_result = subprocess.run(
        cmake_cmd,
        capture_output=True,
        text=True,
        cwd=str(cmake_build_dir),
        timeout=300,  # 5 min for configure
    )
    build_log_lines.extend(configure_result.stdout.splitlines()[-50:])
    build_log_lines.extend(configure_result.stderr.splitlines()[-20:])

    if configure_result.returncode != 0:
        elapsed = time.monotonic() - start_time
        err = (configure_result.stderr or configure_result.stdout)[:500]
        tail = [l for l in build_log_lines[-50:] if l]
        return {
            "success": False,
            "duration_seconds": round(elapsed, 1),
            "error_summary": f"cmake configure failed (rc={configure_result.returncode}): "
                             f"{err[:300]}",
            "build_log_tail": tail if tail else [err[:300]],
            "timeout_exceeded": False,
            "whl_path": None,
        }

    # Step 4: cmake --build with timeout
    remaining_seconds = BUILD_TIMEOUT_SECONDS - (time.monotonic() - start_time)
    if remaining_seconds <= 0:
        elapsed = time.monotonic() - start_time
        return {
            "success": False,
            "duration_seconds": round(elapsed, 1),
            "error_summary": "cmake configure exceeded 45-minute budget",
            "build_log_tail": [l for l in build_log_lines[-50:] if l],
            "timeout_exceeded": True,
            "whl_path": None,
        }

    print(f"  Running cmake --build (timeout={int(remaining_seconds)}s) ...")
    try:
        build_result = subprocess.run(
            ["cmake", "--build", ".", "--parallel"],
            capture_output=True,
            text=True,
            cwd=str(cmake_build_dir),
            timeout=int(remaining_seconds),
        )
        build_log_lines.extend(build_result.stdout.splitlines()[-50:])
        build_log_lines.extend(build_result.stderr.splitlines()[-20:])

        elapsed = time.monotonic() - start_time

        if build_result.returncode != 0:
            err = (build_result.stderr or build_result.stdout)[:500]
            tail = [l for l in build_log_lines[-50:] if l]
            return {
                "success": False,
                "duration_seconds": round(elapsed, 1),
                "error_summary": f"cmake --build failed (rc={build_result.returncode}): "
                                 f"{err[:300]}",
                "build_log_tail": tail if tail else [err[:300]],
                "timeout_exceeded": False,
                "whl_path": None,
            }

    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start_time
        tail = [l for l in build_log_lines[-50:] if l]
        return {
            "success": False,
            "duration_seconds": round(elapsed, 1),
            "error_summary": f"cmake --build timed out after {BUILD_TIMEOUT_SECONDS}s",
            "build_log_tail": tail if tail else [f"timeout after {BUILD_TIMEOUT_SECONDS}s"],
            "timeout_exceeded": True,
            "whl_path": None,
        }

    # Step 5: Find the built wheel
    elapsed = time.monotonic() - start_time
    whl_candidates = list(cmake_build_dir.rglob("dist/*.whl")) + list(
        cmake_build_dir.rglob("*.whl")
    )
    if whl_candidates:
        whl_path = str(whl_candidates[0])
        print(f"  Build succeeded. Wheel: {whl_path}")
        return {
            "success": True,
            "duration_seconds": round(elapsed, 1),
            "whl_path": whl_path,
            "timeout_exceeded": False,
        }

    # No wheel found even with rc=0 — treat as failure
    return {
        "success": False,
        "duration_seconds": round(elapsed, 1),
        "error_summary": "cmake --build succeeded but no .whl found in build directory",
        "build_log_tail": [l for l in build_log_lines[-50:] if l],
        "timeout_exceeded": False,
        "whl_path": None,
    }


def _install_wheel_into_venv(whl_path: str) -> tuple[bool, str]:
    """Install the given wheel into .venv-npu.

    Returns (success, message).
    """
    venv_pip = _VENV_NPU / "bin" / "pip"
    if not venv_pip.exists():
        return False, f"pip not found at {venv_pip}"

    print(f"  Installing {whl_path} into .venv-npu ...")
    try:
        r = subprocess.run(
            [str(venv_pip), "install", whl_path, "--force-reinstall", "--quiet"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if r.returncode == 0:
            return True, "wheel installed successfully"
        return False, f"pip install failed: {r.stderr[:300]}"
    except subprocess.TimeoutExpired:
        return False, "pip install timed out"


# ---------------------------------------------------------------------------
# Inference test (SCENARIO-EXP303-C)
# ---------------------------------------------------------------------------


def _run_inference_benchmark(onnx_model: Path) -> dict[str, Any] | str:
    """Run NPU + CPU inference benchmark using .venv-npu.

    Runs as a subprocess inside .venv-npu to pick up the newly installed ORT wheel
    with VitisAI EP compiled in.

    Returns a dict with inference_result fields, or a string error message.
    The string error message is used to distinguish blocked_abi from other failures.
    """
    venv_python = _VENV_NPU / "bin" / "python"
    if not venv_python.exists():
        return "venv_python not found"

    # Build the inline benchmark script
    vaip_config_str = str(_VAIP_CONFIG) if _VAIP_CONFIG.exists() else ""
    vitisai_so_dir_str = str(_VITISAI_SO_DIR)

    benchmark_script = f"""
import json
import sys
import time
import os

# Add VitisAI .so to LD_LIBRARY_PATH for loading
os.environ.setdefault("LD_LIBRARY_PATH", "")
existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
vitisai_dir = {vitisai_so_dir_str!r}
if vitisai_dir not in existing_ld:
    os.environ["LD_LIBRARY_PATH"] = vitisai_dir + ":" + existing_ld

import onnxruntime as ort
import numpy as np

onnx_model = {str(onnx_model)!r}
timed_calls = {TIMED_CALLS}
warmup_calls = {WARMUP_CALLS}
vaip_config = {vaip_config_str!r}

# Check available providers
available = ort.get_available_providers()

# NPU session
npu_latency_us = None
provider_used = None
if "VitisAIExecutionProvider" in available:
    provider_options = {{}}
    if vaip_config:
        provider_options = {{"config_file": vaip_config}}
    try:
        sess_npu = ort.InferenceSession(
            onnx_model,
            providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
            provider_options=[provider_options, {{}}],
        )
        input_name = sess_npu.get_inputs()[0].name
        input_shape = sess_npu.get_inputs()[0].shape
        # Replace symbolic dimensions with concrete values
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in input_shape]
        dummy = np.random.randn(*shape).astype(np.float32)

        # Warmup
        for _ in range(warmup_calls):
            sess_npu.run(None, {{input_name: dummy}})

        # Timed
        t0 = time.perf_counter()
        for _ in range(timed_calls):
            sess_npu.run(None, {{input_name: dummy}})
        elapsed = time.perf_counter() - t0
        npu_latency_us = (elapsed / timed_calls) * 1e6
        active_providers = sess_npu.get_providers()
        provider_used = next(
            (p for p in active_providers if "VitisAI" in p), active_providers[0]
        )
    except Exception as e:
        npu_latency_us = None
        provider_used = f"error: {{e}}"
else:
    provider_used = f"VitisAI not available. Available: {{available}}"

# CPU baseline session (same run, so hardware is consistent)
sess_cpu = ort.InferenceSession(
    onnx_model,
    providers=["CPUExecutionProvider"],
)
input_name = sess_cpu.get_inputs()[0].name
input_shape = sess_cpu.get_inputs()[0].shape
shape = [d if isinstance(d, int) and d > 0 else 1 for d in input_shape]
dummy = np.random.randn(*shape).astype(np.float32)

for _ in range(warmup_calls):
    sess_cpu.run(None, {{input_name: dummy}})

t0 = time.perf_counter()
for _ in range(timed_calls):
    sess_cpu.run(None, {{input_name: dummy}})
elapsed_cpu = time.perf_counter() - t0
cpu_latency_us = (elapsed_cpu / timed_calls) * 1e6

result = {{
    "available_providers": available,
    "npu_latency_us": npu_latency_us,
    "cpu_latency_us": cpu_latency_us,
    "timed_calls": timed_calls,
    "provider_used": provider_used,
}}
print(json.dumps(result))
"""

    # Set LD_LIBRARY_PATH for subprocess
    env = os.environ.copy()
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = (
        str(_VITISAI_SO_DIR) + (":" + existing_ld if existing_ld else "")
    )

    try:
        result = subprocess.run(
            [str(venv_python), "-c", benchmark_script],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return "inference benchmark timed out after 300s"

    if result.returncode != 0:
        err = (result.stderr or result.stdout)[:400]
        return f"inference subprocess failed (rc={result.returncode}): {err}"

    # Parse JSON output from benchmark script
    output_lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
    if not output_lines:
        return f"no JSON output from benchmark. stderr: {result.stderr[:200]}"

    try:
        data = json.loads(output_lines[-1])
    except json.JSONDecodeError as e:
        return f"could not parse benchmark JSON: {e}. output: {result.stdout[:200]}"

    available = data.get("available_providers", [])
    npu_latency_us = data.get("npu_latency_us")
    cpu_latency_us = data.get("cpu_latency_us")
    provider_used = data.get("provider_used", "")
    timed_calls = data.get("timed_calls", TIMED_CALLS)

    # Check if VitisAI was actually used
    if "VitisAI" not in str(provider_used) or npu_latency_us is None:
        # VitisAI EP not available — ABI block
        return f"blocked_abi: VitisAI EP not in available providers after build. " \
               f"Available: {available}. provider_used: {provider_used!r}"

    speedup = round(cpu_latency_us / npu_latency_us, 3) if npu_latency_us else None

    return {
        "npu_latency_us": round(npu_latency_us, 3),
        "cpu_latency_us": round(cpu_latency_us, 3),
        "speedup_factor": speedup,
        "provider_used": provider_used,
        "timed_calls": timed_calls,
        "available_providers": available,
    }


# ---------------------------------------------------------------------------
# Hardware wishlist updater
# ---------------------------------------------------------------------------


def _update_hardware_wishlist(honest_verdict: str, details: dict[str, Any]) -> None:
    """Append Exp 303 findings to the AMD XDNA section of research-hardware-wishlist.md.

    Does NOT remove existing content (per documentation style rules).
    Appends a dated findings block after the existing Exp 292 findings.
    """
    if not _WISHLIST_MD.exists():
        return

    content = _WISHLIST_MD.read_text()

    # Build findings block
    lines: list[str] = [
        f"  - **Exp 303 findings ({RUN_DATE}):**",
    ]

    pc = details.get("prereq_check", {})
    ninja_ok = pc.get("ninja_installed", False)
    openblas_ok = pc.get("openblas_installed", False)
    cmake_ok = pc.get("cmake_sufficient", False)

    if honest_verdict == "blocked_prereq":
        missing_items: list[str] = []
        if not ninja_ok:
            missing_items.append(
                f"`ninja`: not found. Install: `{pc.get('ninja_install_command', 'see pacman/apt')}`"
            )
        if not openblas_ok:
            missing_items.append(
                f"`openblas`: not found. Install: `{pc.get('openblas_install_command', 'see pacman/apt')}`"
            )
        if not cmake_ok:
            missing_items.append(f"`cmake`: {pc.get('cmake_note', 'insufficient version')}")
        lines.append(
            "    - Still blocked by missing prerequisites:"
        )
        for item in missing_items:
            lines.append(f"      - {item}")
        lines.append(
            "    - **Status:** BLOCKED — install prerequisites, then re-run Exp 303."
        )
    elif honest_verdict == "blocked_build":
        bo = details.get("build_outcome", {})
        err = bo.get("error_summary", "unknown build error")
        timeout = bo.get("timeout_exceeded", False)
        label = "TIMEOUT" if timeout else "BUILD FAILED"
        lines.append(f"    - Prerequisites satisfied. ORT source build: **{label}**.")
        lines.append(f"    - Error: `{err[:120]}`")
        lines.append(
            "    - **Next:** inspect build_log_tail in experiment_303_npu_results.json for fix."
        )
    elif honest_verdict == "blocked_abi":
        lines.append(
            "    - Source build succeeded but VitisAI EP not available after install."
        )
        lines.append(
            "    - **ABI mismatch** — ORT wheel built without VitisAI EP being linked."
        )
        lines.append(
            "    - **Next:** verify cmake -DONNXRUNTIME_USE_VITISAI=ON was respected."
        )
    elif honest_verdict == "npu_working":
        ir = details.get("inference_result", {})
        npu_us = ir.get("npu_latency_us", "?")
        cpu_us = ir.get("cpu_latency_us", "?")
        speedup = ir.get("speedup_factor", "?")
        lines.append(f"    - **NPU WORKING!** npu={npu_us}µs, cpu={cpu_us}µs, speedup={speedup}x")
        lines.append(
            "    - **Status:** UNBLOCKED — VitisAI EP operational via source build."
        )

    findings_block = "\n".join(lines)

    # Insert after the Exp 292 findings block if present; else before the last AMD XDNA item
    marker = "  - **Exp 292 findings (20260414):**"
    exp292_end_marker = "  - **What we have:**"

    if marker in content and exp292_end_marker in content:
        # Insert between Exp 292 findings and "What we have" section
        insert_pos = content.index(exp292_end_marker)
        content = content[:insert_pos] + findings_block + "\n" + content[insert_pos:]
    elif "AMD XDNA NPU" in content:
        # Fallback: append at end of AMD XDNA bullet block
        # Find "Priority 3" section end and insert before Priority 4
        priority4_marker = "## Priority 4"
        if priority4_marker in content:
            insert_pos = content.index(priority4_marker)
            content = content[:insert_pos] + findings_block + "\n\n" + content[insert_pos:]
        else:
            content += "\n" + findings_block + "\n"
    else:
        content += "\n" + findings_block + "\n"

    _WISHLIST_MD.write_text(content)
    print(f"  Updated {_WISHLIST_MD.name} with Exp 303 findings.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Exp 303: AMD XDNA NPU unblock — prereq check, source build, benchmark."""
    print(f"\n=== Experiment {EXPERIMENT}: AMD XDNA NPU Unblock ===")
    print(f"    Run date: {RUN_DATE}\n")

    _RESULTS_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: PREREQ CHECK
    # ------------------------------------------------------------------
    print("[1/4] Checking prerequisites ...")
    prereq_check = _collect_prereq_check()

    ninja_ok = prereq_check["ninja_installed"]
    openblas_ok = prereq_check["openblas_installed"]
    cmake_ok = prereq_check["cmake_sufficient"]
    ryzen_ok = prereq_check["ryzen_ai_sw_present"]
    vitisai_ok = prereq_check["vitisai_so_present"]

    print(f"      ninja: {'OK' if ninja_ok else 'MISSING'}")
    print(f"      openblas: {'OK' if openblas_ok else 'MISSING'}")
    print(f"      cmake: {prereq_check['cmake_version']} ({'OK' if cmake_ok else 'TOO OLD/MISSING'})")
    print(f"      RyzenAI-SW: {'present' if ryzen_ok else 'MISSING'}")
    print(f"      libonnxruntime_providers_vitisai.so: {'present' if vitisai_ok else 'MISSING'}")

    if not ninja_ok or not openblas_ok or not cmake_ok:
        print("\n  Prereqs missing — emitting blocked_prereq artifact.")
        onnx_model = _select_onnx_model()
        result = {
            "experiment": EXPERIMENT,
            "description": (
                "AMD XDNA NPU unblock — blocked: source build prerequisites missing"
            ),
            "run_date": RUN_DATE,
            "execution_path": "blocked_prereq",
            "prereq_check": prereq_check,
            "build_outcome": None,
            "inference_result": None,
            "honest_verdict": "blocked_prereq",
            "onnx_model_considered": str(onnx_model) if onnx_model else None,
            "next_steps": _build_next_steps(prereq_check),
        }
        _OUTPUT_FILE.write_text(json.dumps(result, indent=2))
        _update_hardware_wishlist("blocked_prereq", result)
        print(f"\n  Output: {_OUTPUT_FILE}")
        return

    # ------------------------------------------------------------------
    # Step 2: SOURCE BUILD
    # ------------------------------------------------------------------
    print("\n[2/4] Attempting ORT 1.20.1 source build with VitisAI EP ...")
    build_outcome = _attempt_source_build()
    print(f"      success: {build_outcome['success']}, "
          f"duration: {build_outcome['duration_seconds']}s")

    if not build_outcome["success"]:
        onnx_model = _select_onnx_model()
        result = {
            "experiment": EXPERIMENT,
            "description": (
                "AMD XDNA NPU unblock — blocked: ORT source build failed"
            ),
            "run_date": RUN_DATE,
            "execution_path": "blocked_build",
            "prereq_check": prereq_check,
            "build_outcome": build_outcome,
            "inference_result": None,
            "honest_verdict": "blocked_build",
            "onnx_model_considered": str(onnx_model) if onnx_model else None,
        }
        _OUTPUT_FILE.write_text(json.dumps(result, indent=2))
        _update_hardware_wishlist("blocked_build", result)
        print(f"\n  Output: {_OUTPUT_FILE}")
        return

    # ------------------------------------------------------------------
    # Step 3: INSTALL WHEEL
    # ------------------------------------------------------------------
    whl_path = build_outcome.get("whl_path")
    if whl_path:
        print(f"\n[3/4] Installing wheel {whl_path} into .venv-npu ...")
        install_ok, install_msg = _install_wheel_into_venv(whl_path)
        print(f"      {install_msg}")
        if not install_ok:
            build_outcome["wheel_install_error"] = install_msg
            build_outcome["success"] = False
            onnx_model = _select_onnx_model()
            result = {
                "experiment": EXPERIMENT,
                "description": "AMD XDNA NPU unblock — blocked: wheel install failed",
                "run_date": RUN_DATE,
                "execution_path": "blocked_build",
                "prereq_check": prereq_check,
                "build_outcome": build_outcome,
                "inference_result": None,
                "honest_verdict": "blocked_build",
                "onnx_model_considered": str(onnx_model) if onnx_model else None,
            }
            _OUTPUT_FILE.write_text(json.dumps(result, indent=2))
            _update_hardware_wishlist("blocked_build", result)
            print(f"\n  Output: {_OUTPUT_FILE}")
            return
    else:
        print("\n[3/4] No wheel to install — build reported success with no .whl path.")
        # Rare edge case: treat as blocked_build
        onnx_model = _select_onnx_model()
        build_outcome["error_summary"] = "build reported success but whl_path is None"
        build_outcome["success"] = False
        result = {
            "experiment": EXPERIMENT,
            "description": "AMD XDNA NPU unblock — blocked: no wheel produced",
            "run_date": RUN_DATE,
            "execution_path": "blocked_build",
            "prereq_check": prereq_check,
            "build_outcome": build_outcome,
            "inference_result": None,
            "honest_verdict": "blocked_build",
            "onnx_model_considered": str(onnx_model) if onnx_model else None,
        }
        _OUTPUT_FILE.write_text(json.dumps(result, indent=2))
        _update_hardware_wishlist("blocked_build", result)
        print(f"\n  Output: {_OUTPUT_FILE}")
        return

    # ------------------------------------------------------------------
    # Step 4: INFERENCE BENCHMARK
    # ------------------------------------------------------------------
    print("\n[4/4] Running inference benchmark on NPU and CPU ...")
    onnx_model = _select_onnx_model()
    if onnx_model is None:
        result = {
            "experiment": EXPERIMENT,
            "description": "AMD XDNA NPU unblock — blocked: no ONNX model found",
            "run_date": RUN_DATE,
            "execution_path": "blocked_build",
            "prereq_check": prereq_check,
            "build_outcome": build_outcome,
            "inference_result": None,
            "honest_verdict": "blocked_build",
            "onnx_model_considered": None,
        }
        _OUTPUT_FILE.write_text(json.dumps(result, indent=2))
        print(f"\n  Output: {_OUTPUT_FILE}")
        return

    print(f"      Using ONNX model: {onnx_model.name}")
    inference_result = _run_inference_benchmark(onnx_model)

    # inference_result is either a dict (success) or a string (error/blocked)
    if isinstance(inference_result, str):
        # Check whether it's a blocked_abi case
        if "blocked_abi" in inference_result:
            honest_verdict = "blocked_abi"
            description = "AMD XDNA NPU unblock — blocked: ABI/EP mismatch after source build"
        else:
            honest_verdict = "blocked_build"
            description = "AMD XDNA NPU unblock — blocked: inference failed after build"

        result = {
            "experiment": EXPERIMENT,
            "description": description,
            "run_date": RUN_DATE,
            "execution_path": honest_verdict,
            "prereq_check": prereq_check,
            "build_outcome": build_outcome,
            "inference_result": None,
            "honest_verdict": honest_verdict,
            "onnx_model_used": onnx_model.name,
            "inference_error": inference_result,
        }
        _OUTPUT_FILE.write_text(json.dumps(result, indent=2))
        _update_hardware_wishlist(honest_verdict, result)
        print(f"\n  Output: {_OUTPUT_FILE}")
        return

    # Success
    print(
        f"      NPU latency: {inference_result['npu_latency_us']:.1f}µs | "
        f"CPU latency: {inference_result['cpu_latency_us']:.1f}µs | "
        f"Speedup: {inference_result['speedup_factor']:.2f}x"
    )

    result = {
        "experiment": EXPERIMENT,
        "description": "AMD XDNA NPU unblock — NPU working via ORT source build",
        "run_date": RUN_DATE,
        "execution_path": "npu_working",
        "cpu_ort_baseline_us": CPU_ORT_BASELINE_US,
        "prereq_check": prereq_check,
        "build_outcome": build_outcome,
        "inference_result": inference_result,
        "honest_verdict": "npu_working",
        "onnx_model_used": onnx_model.name,
    }
    _OUTPUT_FILE.write_text(json.dumps(result, indent=2))
    _update_hardware_wishlist("npu_working", result)
    print(f"\n  Output: {_OUTPUT_FILE}")
    print("  *** NPU UNBLOCKED ***")


def _build_next_steps(prereq_check: dict[str, Any]) -> list[str]:
    """Build a human-readable list of next steps given the prereq_check result."""
    steps: list[str] = []
    if not prereq_check.get("ninja_installed"):
        cmd = prereq_check.get("ninja_install_command", "install ninja")
        steps.append(f"Install ninja: {cmd}")
    if not prereq_check.get("openblas_installed"):
        cmd = prereq_check.get("openblas_install_command", "install openblas")
        steps.append(f"Install openblas: {cmd}")
    if not prereq_check.get("cmake_sufficient"):
        note = prereq_check.get("cmake_note") or prereq_check.get("cmake_install_command", "upgrade cmake to >= 3.26")
        steps.append(f"Fix cmake: {note}")
    if steps:
        steps.append(
            "Then re-run: JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_303_npu_unblock.py"
        )
    return steps


if __name__ == "__main__":
    main()

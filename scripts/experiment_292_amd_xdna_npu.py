#!/usr/bin/env python3
"""Experiment 292: AMD XDNA NPU VitisAI EP benchmark.

Attempts to benchmark the Tier 3 JEPA predictor on the AMD XDNA NPU
(Ryzen AI 9 HX 370) using the VitisAI Execution Provider for onnxruntime.

Two paths are tried in order, emitting honest blocker artifacts on failure:

  Path A — Pre-built .so approach (fast):
    Use the pre-built libonnxruntime.so.1.20.1 + libonnxruntime_providers_vitisai.so
    from ~/github.com/amd/RyzenAI-SW/Ryzen-AI-CVML-Library/linux/onnx/ryzen14/.
    Install onnxruntime==1.20.1 in .venv-npu (if not already 1.20.1), then run
    inference with LD_LIBRARY_PATH pointing at the RyzenAI-SW .so files.

  Path B — Source build (fallback, 45-minute hard timeout):
    cmake -DONNXRUNTIME_USE_VITISAI=ON -DONNXRUNTIME_VITISAI_EP_LIBRARY_PATH=...
    Build onnxruntime 1.20.1 from source in a temp dir, install the resulting
    wheel into .venv-npu, then benchmark.

Prerequisite check (fails fast with blocked artifact if missing):
  - cmake ≥ 3.26 (for source build path)
  - ninja (for source build path)
  - openblas (for source build path)
  - ~/github.com/amd/RyzenAI-SW/ directory
  - ~/github.com/amd/RyzenAI-SW/.../libonnxruntime_providers_vitisai.so

If all prerequisites for Path B are missing, Path A is still attempted.
If Path A also fails, a blocked artifact is emitted with specific missing items.

Baseline: CPU ORT 5.847 µs/call (Exp 257, onnx_cpu record)
ONNX model: results/jepa_predictor_291.onnx (fallback: jepa_predictor_146.onnx)

Writes:
    results/experiment_292_results.json

Spec: REQ-PRED-003
SCENARIO-EXP292-A (prerequisite check and blocked artifact)
SCENARIO-EXP292-B (build timeout handling — emit blocker with build log tail)
SCENARIO-EXP292-C (benchmark schema — latency, speedup, baseline comparison)
SCENARIO-EXP292-D (honest labeling — no fabricated numbers for non-hardware paths)

Run:
    cd /home/ianblenke/github.com/ianblenke/carnot
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_292_amd_xdna_npu.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT: int = 292
RUN_DATE: str = "20260414"

# CPU ORT baseline from Exp 257 (onnx_cpu record, latency_us = 5.847)
CPU_ORT_BASELINE_US: float = 5.847

# Build timeout: 45 minutes in seconds
BUILD_TIMEOUT_SECONDS: int = 45 * 60

# Benchmark calls
WARMUP_CALLS: int = 500
TIMED_CALLS: int = 5_000

# onnxruntime version that matches the pre-built AMD .so files
ORT_TARGET_VERSION: str = "1.20.1"

# Repo layout
_REPO_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_DIR = _REPO_ROOT / "results"
_OUTPUT_FILE = _RESULTS_DIR / "experiment_292_results.json"
_VENV_NPU = _REPO_ROOT / ".venv-npu"

# RyzenAI-SW pre-built library paths
_RYZEN_AI_SW = Path.home() / "github.com" / "amd" / "RyzenAI-SW"
_VITISAI_SO_DIR = (
    _RYZEN_AI_SW / "Ryzen-AI-CVML-Library" / "linux" / "onnx" / "ryzen14"
)

# ONNX model: prefer Exp 291 result, fall back to Exp 146
_ONNX_291 = _RESULTS_DIR / "jepa_predictor_291.onnx"
_ONNX_146 = _RESULTS_DIR / "jepa_predictor_146.onnx"

# XRT install path
_XRT_PREFIX = Path("/opt/xilinx/xrt")

# VitisAI NPU config bundled with RyzenAI-SW
_VAIP_CONFIG = _VITISAI_SO_DIR / "vaip_config_npu_2_3.json"


# ---------------------------------------------------------------------------
# Prerequisite detection helpers
# ---------------------------------------------------------------------------


def _cmake_version() -> tuple[int, int, int] | None:
    """Return cmake version as (major, minor, patch) or None if not found."""
    try:
        result = subprocess.run(
            ["cmake", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        match = re.search(r"cmake version (\d+)\.(\d+)\.(\d+)", result.stdout)
        if match:
            return int(match.group(1)), int(match.group(2)), int(match.group(3))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _ninja_available() -> bool:
    """Return True if `ninja` (or `ninja-build`) is on PATH."""
    return shutil.which("ninja") is not None or shutil.which("ninja-build") is not None


def _openblas_available() -> bool:
    """Return True if openblas is detectable via pkg-config or ldconfig."""
    # Try pkg-config first
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

    # Fallback: ldconfig -p
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

    # Last resort: look for header or .so in common paths
    for candidate in [
        "/usr/lib/libopenblas.so",
        "/usr/lib/x86_64-linux-gnu/libopenblas.so",
        "/usr/local/lib/libopenblas.so",
    ]:
        if Path(candidate).exists():
            return True
    return False


def _xrt_version() -> str | None:
    """Return the XRT version string from /opt/xilinx/xrt/version.json, or None."""
    version_file = _XRT_PREFIX / "version.json"
    if version_file.exists():
        try:
            data = json.loads(version_file.read_text())
            # version.json typically has {"BUILD_VERSION": "2.20.0", ...}
            return data.get("BUILD_VERSION") or data.get("VERSION")
        except (json.JSONDecodeError, KeyError):
            pass
    # Try xrt-smi
    try:
        r = subprocess.run(
            ["xrt-smi", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        match = re.search(r"(\d+\.\d+\.\d+)", r.stdout)
        if match:
            return match.group(1)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _amdxdna_driver_loaded() -> bool:
    """Return True if the amdxdna kernel module is loaded."""
    try:
        r = subprocess.run(
            ["lsmod"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return "amdxdna" in r.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _collect_npu_hardware_info() -> dict[str, Any]:
    """Collect all relevant NPU hardware and software information."""
    vitisai_so = _VITISAI_SO_DIR / "libonnxruntime_providers_vitisai.so"
    return {
        "amdxdna_driver_loaded": _amdxdna_driver_loaded(),
        "xrt_version": _xrt_version(),
        "ryzen_ai_sw_present": _RYZEN_AI_SW.is_dir(),
        "vitisai_so_dir": str(_VITISAI_SO_DIR),
        "vitisai_so_present": vitisai_so.exists(),
        "vaip_config_present": _VAIP_CONFIG.exists(),
        "venv_npu_present": _VENV_NPU.is_dir(),
    }


def _check_source_build_prereqs() -> list[str]:
    """Return a list of missing prerequisites for the source build path.

    Returns an empty list if all prerequisites are present.
    Each entry is a human-readable description of the missing item.
    """
    missing: list[str] = []

    cmake_ver = _cmake_version()
    if cmake_ver is None:
        missing.append("cmake: not found (need cmake ≥ 3.26)")
    elif cmake_ver < (3, 26, 0):
        v_str = ".".join(str(x) for x in cmake_ver)
        missing.append(f"cmake {v_str}: too old (need ≥ 3.26)")

    if not _ninja_available():
        missing.append(
            "ninja: not found (install via: sudo pacman -S ninja OR sudo apt install ninja-build)"
        )

    if not _openblas_available():
        missing.append(
            "openblas: not found (install via: sudo pacman -S openblas OR sudo apt install libopenblas-dev)"
        )

    if not _RYZEN_AI_SW.is_dir():
        missing.append(
            f"RyzenAI-SW directory missing: {_RYZEN_AI_SW} "
            "(git clone https://github.com/amd/RyzenAI-SW)"
        )

    vitisai_so = _VITISAI_SO_DIR / "libonnxruntime_providers_vitisai.so"
    if not vitisai_so.exists():
        missing.append(
            f"libonnxruntime_providers_vitisai.so missing at {_VITISAI_SO_DIR}"
        )

    return missing


# ---------------------------------------------------------------------------
# ONNX model selection
# ---------------------------------------------------------------------------


def _select_onnx_model() -> Path | None:
    """Return the ONNX model path to use, preferring Exp 291 over Exp 146."""
    if _ONNX_291.exists():
        return _ONNX_291
    if _ONNX_146.exists():
        return _ONNX_146
    return None


# ---------------------------------------------------------------------------
# Path A: pre-built .so approach
# ---------------------------------------------------------------------------


def _ensure_ort_1201_in_venv() -> tuple[bool, str]:
    """Ensure onnxruntime==1.20.1 is installed in .venv-npu.

    The pre-built AMD .so files in RyzenAI-SW were built against ORT 1.20.1.
    Loading them with ORT 1.24.x causes a segfault due to ABI incompatibility.

    Returns (success, message).
    """
    venv_pip = _VENV_NPU / "bin" / "pip"
    if not venv_pip.exists():
        return False, f"pip not found in {_VENV_NPU}/bin/"

    # Check current version
    try:
        r = subprocess.run(
            [str(_VENV_NPU / "bin" / "python"), "-c",
             "import onnxruntime; print(onnxruntime.__version__)"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        current_ver = r.stdout.strip()
        if current_ver == ORT_TARGET_VERSION:
            return True, f"onnxruntime {ORT_TARGET_VERSION} already installed"
    except subprocess.TimeoutExpired:
        return False, "timeout checking onnxruntime version"

    # Downgrade/install 1.20.1
    print(f"  Installing onnxruntime=={ORT_TARGET_VERSION} in .venv-npu "
          f"(current: {current_ver})...")
    try:
        r = subprocess.run(
            [str(venv_pip), "install", f"onnxruntime=={ORT_TARGET_VERSION}",
             "--force-reinstall", "--quiet"],
            capture_output=True,
            text=True,
            timeout=300,  # 5-minute timeout for pip install
        )
        if r.returncode == 0:
            return True, f"onnxruntime=={ORT_TARGET_VERSION} installed successfully"
        return False, f"pip install failed: {r.stderr[:300]}"
    except subprocess.TimeoutExpired:
        return False, "pip install timed out after 300s"


def _try_prebuilt_so_path(
    onnx_model: Path,
) -> dict[str, Any] | None:
    """Attempt NPU inference using the pre-built RyzenAI-SW .so files.

    Strategy:
      1. Ensure .venv-npu has onnxruntime==1.20.1 (matching AMD .so ABI)
      2. Set LD_LIBRARY_PATH to include the VitisAI .so directory
      3. Run a subprocess that creates a VitisAI InferenceSession and benchmarks

    Returns a result dict on success, or None if this path is not available.
    """
    vitisai_so = _VITISAI_SO_DIR / "libonnxruntime_providers_vitisai.so"
    if not vitisai_so.exists():
        return None

    if not _VENV_NPU.is_dir():
        return None

    venv_python = _VENV_NPU / "bin" / "python"
    if not venv_python.exists():
        return None

    # Ensure ORT 1.20.1 — AMD .so files are ABI-incompatible with ORT 1.24.x
    ok, msg = _ensure_ort_1201_in_venv()
    if not ok:
        return {"path_a_error": f"Could not install onnxruntime=={ORT_TARGET_VERSION}: {msg}"}
    print(f"  {msg}")

    # Inline Python script that:
    # 1. Creates an InferenceSession with VitisAIExecutionProvider (via LD_LIBRARY_PATH)
    # 2. Runs WARMUP_CALLS + TIMED_CALLS inference calls
    # 3. Returns a JSON result dict on stdout
    # NOTE: Do NOT use ctypes.CDLL here — LD_LIBRARY_PATH must be set at process start
    #       for the dynamic linker to pick up the AMD shared libraries.
    inline_script = f"""
import sys
import os
import json
import time
import numpy as np

try:
    import onnxruntime as ort
except ImportError as e:
    print(json.dumps({{"error": f"onnxruntime import failed: {{e}}"}}))
    sys.exit(1)

# Check available providers — VitisAI should appear because LD_LIBRARY_PATH
# includes the AMD .so directory at process start.
available = ort.get_available_providers()
has_vitisai = any("VitisAI" in p for p in available)

if not has_vitisai:
    print(json.dumps({{
        "error": (
            f"VitisAIExecutionProvider not in available providers even with "
            f"LD_LIBRARY_PATH set. ort version: {{ort.__version__}}. "
            f"Available: {{available}}"
        ),
        "available_providers": available,
        "ort_version": ort.__version__,
    }}))
    sys.exit(1)

# Create session options with VitisAI config
onnx_model_path = {str(onnx_model)!r}
vaip_config = {str(_VAIP_CONFIG)!r}

sess_options = ort.SessionOptions()

try:
    provider_options = [{{
        "config_file": vaip_config,
    }}]
    sess = ort.InferenceSession(
        onnx_model_path,
        sess_options=sess_options,
        providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
        provider_options=provider_options,
    )
except Exception as e:
    print(json.dumps({{"error": f"Session creation failed: {{e}}", "available_providers": available}}))
    sys.exit(1)

# Determine input shape from model
inp = sess.get_inputs()[0]
inp_shape = inp.shape
inp_name = inp.name

# Use feature_dim = 9 (Exp 257 model has 9-dimensional input)
feature_dim = inp_shape[-1] if inp_shape and inp_shape[-1] else 9
batch = inp_shape[0] if inp_shape and inp_shape[0] else 1
if batch is None or batch == "batch_size":
    batch = 1

dummy = np.random.randn(int(batch), int(feature_dim)).astype(np.float32)

# Warm up
warmup_calls = {WARMUP_CALLS}
for _ in range(warmup_calls):
    sess.run(None, {{inp_name: dummy}})

# Timed calls
timed_calls = {TIMED_CALLS}
t0 = time.perf_counter()
for _ in range(timed_calls):
    sess.run(None, {{inp_name: dummy}})
elapsed = time.perf_counter() - t0

latency_us = (elapsed / timed_calls) * 1e6
throughput = timed_calls / elapsed

# Which providers were actually used?
providers_used = sess.get_providers()

print(json.dumps({{
    "ok": True,
    "latency_us": latency_us,
    "throughput_calls_per_sec": throughput,
    "timed_calls": timed_calls,
    "providers_used": providers_used,
    "available_providers": available,
    "ort_version": ort.__version__,
}}))
"""

    # Set LD_LIBRARY_PATH to include RyzenAI-SW .so directory
    env = os.environ.copy()
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    new_ld = str(_VITISAI_SO_DIR)
    if existing_ld:
        new_ld = f"{new_ld}:{existing_ld}"
    env["LD_LIBRARY_PATH"] = new_ld

    # XRT runtime environment variables
    if _XRT_PREFIX.is_dir():
        env.setdefault("XILINX_XRT", str(_XRT_PREFIX))
        xrt_lib = _XRT_PREFIX / "lib"
        if xrt_lib.is_dir():
            env["LD_LIBRARY_PATH"] = f"{xrt_lib}:{env['LD_LIBRARY_PATH']}"

    # VitisAI EP needs this config to locate the xclbin overlays
    if _VAIP_CONFIG.exists():
        env.setdefault("VITISAI_EP_JSON_CONFIG", str(_VAIP_CONFIG))

    try:
        result = subprocess.run(
            [str(venv_python), "-c", inline_script],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        stdout = result.stdout.strip()
        if not stdout:
            return {"path_a_error": f"No stdout from pre-built path subprocess. stderr: {result.stderr[:500]}"}
        data = json.loads(stdout)
        if data.get("ok"):
            return data
        return {"path_a_error": data.get("error", "Unknown error from pre-built path")}
    except subprocess.TimeoutExpired:
        return {"path_a_error": "Pre-built path subprocess timed out after 120s"}
    except (json.JSONDecodeError, Exception) as e:
        return {"path_a_error": f"Pre-built path failed: {e}"}


# ---------------------------------------------------------------------------
# Path B: source build
# ---------------------------------------------------------------------------


def _attempt_source_build(
    onnx_model: Path,
    build_dir: Path,
) -> dict[str, Any]:
    """Attempt to build onnxruntime 1.20.1 from source with VitisAI EP enabled.

    Enforces a 45-minute wall-clock timeout.  Returns a result dict with either:
      - {"ok": True, ...benchmark fields...}  on success
      - {"timeout_exceeded": True, "build_step": ..., "build_log_tail": [...], "next_action": ...}
      - {"build_failed": True, "build_step": ..., "build_log_tail": [...], "next_action": ...}
    """
    vitisai_so_dir = str(_VITISAI_SO_DIR)
    vaip_config = str(_VAIP_CONFIG)
    venv_python = str(_VENV_NPU / "bin" / "python")
    ort_source_dir = build_dir / "onnxruntime"
    ort_build_dir = build_dir / "ort_build"
    log_file = build_dir / "build.log"

    log_file.parent.mkdir(parents=True, exist_ok=True)

    deadline = time.monotonic() + BUILD_TIMEOUT_SECONDS
    current_step = "init"
    build_log_lines: list[str] = []

    def _run_step(
        cmd: list[str],
        step_name: str,
        cwd: Path | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> tuple[bool, str]:
        """Run a build step, appending output to build_log_lines.

        Returns (success, last_output_snippet).
        """
        nonlocal current_step, build_log_lines
        current_step = step_name
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False, f"TIMEOUT before starting {step_name}"

        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)

        try:
            with open(log_file, "a") as lf:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=cwd,
                    env=env,
                )
                timer = threading.Timer(remaining, proc.kill)
                timer.start()
                try:
                    stdout, _ = proc.communicate()
                finally:
                    timer.cancel()

                lf.write(f"\n=== {step_name} ===\n")
                lf.write(stdout)
                build_log_lines.extend(stdout.splitlines())

            if proc.returncode != 0:
                if proc.returncode == -9:  # SIGKILL (timeout)
                    return False, f"KILLED (timeout) during {step_name}"
                return False, f"Exit code {proc.returncode} during {step_name}"
            return True, f"OK: {step_name}"
        except Exception as e:
            return False, f"Exception during {step_name}: {e}"

    # Step 1: Clone onnxruntime 1.20.1
    ok, msg = _run_step(
        [
            "git",
            "clone",
            "--depth=1",
            "--branch",
            f"v{ORT_TARGET_VERSION}",
            "https://github.com/microsoft/onnxruntime.git",
            str(ort_source_dir),
        ],
        step_name="git_clone_onnxruntime",
    )
    if not ok:
        tail = build_log_lines[-50:] if len(build_log_lines) > 50 else build_log_lines
        return {
            "timeout_exceeded": "TIMEOUT" in msg or "KILLED" in msg,
            "build_failed": True,
            "build_step": current_step,
            "build_log_tail": tail,
            "next_action": (
                "git clone failed. Check internet connectivity and retry: "
                "git clone --depth=1 --branch v1.20.1 https://github.com/microsoft/onnxruntime.git"
            ),
        }

    # Step 2: cmake configure
    ninja_cmd = shutil.which("ninja") or shutil.which("ninja-build") or "ninja"
    cmake_args = [
        "cmake",
        str(ort_source_dir / "cmake"),
        f"-B{ort_build_dir}",
        f"-GNinja",
        f"-DCMAKE_BUILD_TYPE=Release",
        f"-Donnxruntime_USE_VITISAI=ON",
        f"-Donnxruntime_VITISAI_EP_LIBRARY_PATH={vitisai_so_dir}",
        f"-Donnxruntime_BUILD_UNIT_TESTS=OFF",
        f"-DPYTHON_EXECUTABLE={venv_python}",
        f"-Donnxruntime_ENABLE_PYTHON=ON",
        f"-Donnxruntime_BUILD_SHARED_LIB=ON",
    ]
    ok, msg = _run_step(cmake_args, step_name="cmake_configure", cwd=build_dir)
    if not ok:
        tail = build_log_lines[-50:] if len(build_log_lines) > 50 else build_log_lines
        return {
            "timeout_exceeded": "TIMEOUT" in msg or "KILLED" in msg,
            "build_failed": True,
            "build_step": current_step,
            "build_log_tail": tail,
            "next_action": (
                f"cmake configure failed. Verify cmake ≥ 3.26, ninja, and openblas are installed. "
                f"Then rerun: {' '.join(cmake_args[:6])} ..."
            ),
        }

    # Step 3: ninja build (this is the long step — most of the 45 minutes)
    ok, msg = _run_step(
        [ninja_cmd, "-C", str(ort_build_dir), "onnxruntime_pybind11_state"],
        step_name="ninja_build",
        cwd=ort_build_dir,
    )
    if not ok:
        tail = build_log_lines[-50:] if len(build_log_lines) > 50 else build_log_lines
        is_timeout = "TIMEOUT" in msg or "KILLED" in msg
        return {
            "timeout_exceeded": is_timeout,
            "build_failed": True,
            "build_step": current_step,
            "build_log_tail": tail,
            "next_action": (
                "ninja build timed out (>45 min). "
                "Consider using the pre-built AMD wheel from ryzenai.docs.amd.com. "
                f"To retry: ninja -C {ort_build_dir} onnxruntime_pybind11_state"
            )
            if is_timeout
            else (
                f"ninja build failed ({msg}). Review {log_file} for compiler errors. "
                f"To retry: ninja -C {ort_build_dir} onnxruntime_pybind11_state"
            ),
        }

    # Step 4: install Python wheel
    ok, msg = _run_step(
        [
            venv_python,
            str(ort_source_dir / "setup.py"),
            "install",
            f"--build-directory={ort_build_dir}",
        ],
        step_name="python_wheel_install",
        cwd=ort_source_dir,
    )
    if not ok:
        tail = build_log_lines[-50:] if len(build_log_lines) > 50 else build_log_lines
        return {
            "timeout_exceeded": "TIMEOUT" in msg or "KILLED" in msg,
            "build_failed": True,
            "build_step": current_step,
            "build_log_tail": tail,
            "next_action": (
                f"Python wheel install failed. Review {log_file}. "
                f"Try manually: {venv_python} {ort_source_dir}/setup.py install"
            ),
        }

    # Step 5: benchmark using the newly built onnxruntime
    bench_result = _try_prebuilt_so_path(onnx_model)
    if bench_result and bench_result.get("ok"):
        return {"ok": True, **bench_result}
    return {
        "build_failed": True,
        "build_step": "benchmark",
        "build_log_tail": [],
        "next_action": (
            "Build succeeded but benchmark failed. "
            f"Error: {bench_result}. "
            f"Try manually: LD_LIBRARY_PATH={vitisai_so_dir} {venv_python} -c "
            "'import onnxruntime; print(onnxruntime.get_available_providers())'"
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Exp 292: AMD XDNA NPU VitisAI EP benchmark."""
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[Exp {EXPERIMENT}] Starting AMD XDNA NPU VitisAI EP benchmark")
    print(f"  Baseline: CPU ORT {CPU_ORT_BASELINE_US} µs/call (Exp 257)")

    # --- Collect hardware info ---
    npu_hw = _collect_npu_hardware_info()
    print(f"  NPU hardware: amdxdna_loaded={npu_hw['amdxdna_driver_loaded']}, "
          f"xrt={npu_hw['xrt_version']}, vitisai_so={npu_hw['vitisai_so_present']}")

    # --- Select ONNX model ---
    onnx_model = _select_onnx_model()
    if onnx_model is None:
        result_doc = {
            "experiment": EXPERIMENT,
            "description": (
                "AMD XDNA NPU VitisAI EP benchmark — blocked: no ONNX model found"
            ),
            "run_date": RUN_DATE,
            "execution_path": "blocked",
            "cpu_ort_baseline_us": CPU_ORT_BASELINE_US,
            "onnx_model_used": "none",
            "npu_hardware_info": npu_hw,
            "result": {
                "missing_prereqs": [
                    f"ONNX model not found: tried {_ONNX_291} and {_ONNX_146}"
                ],
                "next_action": (
                    "Run Exp 291 or Exp 146 first to generate the ONNX model. "
                    "Command: JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_291_jepa_predictor_retrain.py"
                ),
                "npu_latency_us": None,
                "speedup_vs_cpu_ort": None,
            },
            "honest_verdict": {
                "explanation": "ONNX model missing — cannot benchmark NPU inference.",
                "recommended_next_steps": [
                    "Run Exp 291 to generate jepa_predictor_291.onnx"
                ],
            },
        }
        _OUTPUT_FILE.write_text(json.dumps(result_doc, indent=2))
        print(f"[Exp {EXPERIMENT}] BLOCKED — ONNX model missing. Written: {_OUTPUT_FILE}")
        return

    print(f"  ONNX model: {onnx_model.name}")

    # --- Path A: pre-built .so approach ---
    print("[Exp 292] Trying Path A: pre-built .so approach...")
    path_a_result = _try_prebuilt_so_path(onnx_model)

    if path_a_result and path_a_result.get("ok"):
        # SUCCESS via pre-built .so
        latency_us: float = path_a_result["latency_us"]
        speedup = CPU_ORT_BASELINE_US / latency_us
        providers_used = path_a_result.get("providers_used", [])

        # Confirm VitisAI EP was actually used (not just CPU fallback)
        actually_on_npu = any("VitisAI" in p for p in providers_used)
        if not actually_on_npu:
            # ORT fell back to CPU — not a real NPU result
            print(f"  Path A: ORT fell back to CPU (providers: {providers_used}). "
                  "VitisAI EP not used — treating as blocked.")
        else:
            print(f"  Path A SUCCESS: {latency_us:.3f} µs/call, {speedup:.2f}× vs CPU ORT")
            result_doc = {
                "experiment": EXPERIMENT,
                "description": (
                    "AMD XDNA NPU VitisAI EP benchmark via pre-built RyzenAI-SW .so files"
                ),
                "run_date": RUN_DATE,
                "execution_path": "hardware",
                "cpu_ort_baseline_us": CPU_ORT_BASELINE_US,
                "onnx_model_used": onnx_model.name,
                "npu_hardware_info": npu_hw,
                "result": {
                    "npu_latency_us": round(latency_us, 3),
                    "npu_throughput_calls_per_sec": round(
                        path_a_result["throughput_calls_per_sec"], 1
                    ),
                    "speedup_vs_cpu_ort": round(speedup, 4),
                    "timed_calls": path_a_result["timed_calls"],
                    "providers_used": providers_used,
                    "ort_version": path_a_result.get("ort_version"),
                    "approach": "prebuilt_so_ld_library_path",
                },
                "honest_verdict": {
                    "npu_ep_loaded": True,
                    "explanation": (
                        f"VitisAI EP loaded via pre-built RyzenAI-SW .so files. "
                        f"NPU latency: {latency_us:.3f} µs/call, "
                        f"{speedup:.2f}× vs CPU ORT baseline ({CPU_ORT_BASELINE_US} µs)."
                    ),
                    "recommended_next_steps": [
                        "Package the LD_LIBRARY_PATH approach into .venv-npu activation script",
                        "Benchmark at larger batch sizes to find NPU break-even point",
                    ],
                },
            }
            _OUTPUT_FILE.write_text(json.dumps(result_doc, indent=2))
            print(f"[Exp {EXPERIMENT}] DONE (hardware). Written: {_OUTPUT_FILE}")
            return

    # Path A failed or fell back to CPU. Log the error.
    path_a_error = (path_a_result or {}).get("path_a_error", str(path_a_result))
    print(f"  Path A failed: {path_a_error}")

    # --- Check source build prerequisites ---
    missing_prereqs = _check_source_build_prereqs()

    if missing_prereqs:
        print(f"  Source build blocked — missing: {missing_prereqs}")

        # Compose next_action based on what's missing
        if any("ninja" in p for p in missing_prereqs):
            next_action = (
                "Install ninja first: sudo pacman -S ninja  (Arch) or  "
                "sudo apt install ninja-build  (Debian/Ubuntu), then re-run this script."
            )
        elif any("openblas" in p for p in missing_prereqs):
            next_action = (
                "Install openblas: sudo pacman -S openblas  (Arch) or  "
                "sudo apt install libopenblas-dev  (Debian/Ubuntu), then re-run."
            )
        elif any("cmake" in p for p in missing_prereqs):
            next_action = "Install cmake ≥ 3.26 and re-run this script."
        else:
            next_action = (
                f"Resolve missing prerequisites and re-run. Missing: {missing_prereqs}"
            )

        result_doc = {
            "experiment": EXPERIMENT,
            "description": (
                "AMD XDNA NPU VitisAI EP — blocked: source build prerequisites missing"
            ),
            "run_date": RUN_DATE,
            "execution_path": "blocked",
            "cpu_ort_baseline_us": CPU_ORT_BASELINE_US,
            "onnx_model_used": onnx_model.name,
            "npu_hardware_info": npu_hw,
            "result": {
                "missing_prereqs": missing_prereqs,
                "next_action": next_action,
                "path_a_error": path_a_error,
                "npu_latency_us": None,
                "speedup_vs_cpu_ort": None,
            },
            "honest_verdict": {
                "npu_ep_loaded": False,
                "explanation": (
                    f"Pre-built .so path failed ({path_a_error}). "
                    "Source build path blocked by missing prerequisites: "
                    + "; ".join(missing_prereqs)
                ),
                "recommended_next_steps": [
                    next_action,
                    "Alternatively: download AMD custom onnxruntime wheel from "
                    "ryzenai.docs.amd.com/en/latest/inst.html (requires AMD account + EULA; "
                    "Python 3.9-3.12 only)",
                ],
            },
        }
        _OUTPUT_FILE.write_text(json.dumps(result_doc, indent=2))
        print(f"[Exp {EXPERIMENT}] BLOCKED. Written: {_OUTPUT_FILE}")
        return

    # --- Path B: source build ---
    print("[Exp 292] All prerequisites present — attempting source build (45 min timeout)...")
    with tempfile.TemporaryDirectory(prefix="carnot_ort292_") as tmp_dir:
        build_result = _attempt_source_build(onnx_model, Path(tmp_dir))

    if build_result.get("ok"):
        latency_us = build_result["latency_us"]
        speedup = CPU_ORT_BASELINE_US / latency_us
        providers_used = build_result.get("providers_used", [])
        print(f"  Source build SUCCESS: {latency_us:.3f} µs/call, {speedup:.2f}× vs CPU ORT")
        result_doc = {
            "experiment": EXPERIMENT,
            "description": "AMD XDNA NPU VitisAI EP benchmark via onnxruntime source build",
            "run_date": RUN_DATE,
            "execution_path": "hardware",
            "cpu_ort_baseline_us": CPU_ORT_BASELINE_US,
            "onnx_model_used": onnx_model.name,
            "npu_hardware_info": npu_hw,
            "result": {
                "npu_latency_us": round(latency_us, 3),
                "npu_throughput_calls_per_sec": round(
                    build_result["throughput_calls_per_sec"], 1
                ),
                "speedup_vs_cpu_ort": round(speedup, 4),
                "timed_calls": build_result["timed_calls"],
                "providers_used": providers_used,
                "ort_version": build_result.get("ort_version"),
                "approach": "source_build",
            },
            "honest_verdict": {
                "npu_ep_loaded": True,
                "explanation": (
                    f"onnxruntime 1.20.1 built from source with VitisAI EP. "
                    f"NPU latency: {latency_us:.3f} µs/call, "
                    f"{speedup:.2f}× vs CPU ORT baseline ({CPU_ORT_BASELINE_US} µs)."
                ),
                "recommended_next_steps": [
                    "Install the built wheel permanently in .venv-npu",
                    "Benchmark at larger batch sizes to find NPU break-even point",
                ],
            },
        }
    else:
        # Build failed or timed out
        is_timeout = build_result.get("timeout_exceeded", False)
        step = build_result.get("build_step", "unknown")
        tail = build_result.get("build_log_tail", [])
        next_action = build_result.get("next_action", "Review build log and retry.")

        status_label = "TIMEOUT" if is_timeout else "BUILD FAILED"
        print(f"  Source build {status_label} at step: {step}")
        if tail:
            print(f"  Last build line: {tail[-1]}")

        result_doc = {
            "experiment": EXPERIMENT,
            "description": (
                f"AMD XDNA NPU VitisAI EP — build {'timed out' if is_timeout else 'failed'} "
                f"at step: {step}"
            ),
            "run_date": RUN_DATE,
            "execution_path": "build_failed",
            "cpu_ort_baseline_us": CPU_ORT_BASELINE_US,
            "onnx_model_used": onnx_model.name,
            "npu_hardware_info": npu_hw,
            "result": {
                "build_step": step,
                "build_log_tail": tail[-50:] if len(tail) > 50 else tail,
                "timeout_exceeded": is_timeout,
                "next_action": next_action,
                "npu_latency_us": None,
                "speedup_vs_cpu_ort": None,
            },
            "honest_verdict": {
                "npu_ep_loaded": False,
                "explanation": (
                    f"onnxruntime source build {'timed out (>45 min)' if is_timeout else 'failed'} "
                    f"at step '{step}'. No NPU numbers generated."
                ),
                "recommended_next_steps": [
                    next_action,
                    "Alternative: download AMD custom onnxruntime wheel from "
                    "ryzenai.docs.amd.com/en/latest/inst.html",
                ],
            },
        }

    _OUTPUT_FILE.write_text(json.dumps(result_doc, indent=2))
    print(f"[Exp {EXPERIMENT}] Written: {_OUTPUT_FILE}")


if __name__ == "__main__":
    main()

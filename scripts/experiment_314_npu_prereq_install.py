#!/usr/bin/env python3
"""Experiment 314: AMD XDNA NPU prereq retry — check if ninja/openblas now installed.

This experiment re-runs the Exp 303 NPU unblock workflow AFTER checking whether
the prerequisites that blocked Exp 303 are now installed on the system.

Context:
  Exps 292 and 303 were both blocked by missing `ninja` and `openblas` packages.
  Install commands for those blockers are well-known:
    - ninja:    sudo pacman -S ninja   OR  sudo apt install ninja-build
    - openblas: sudo pacman -S openblas OR  sudo apt install libopenblas-dev

Execution flow:
  1. [PREREQ CHECK] Rerun Exp 303's prereq detection (same functions, same logic).
     Compute prereq_changes: compare current state to Exp 303's blocked state.
     If any required prereq still missing: emit honest_verdict="blocked_prereq" and stop.
  2. [SOURCE BUILD] All prereqs present? Attempt ORT 1.20.1 source build with 45-min timeout.
     Reuses the same cmake + ninja + VitisAI EP flags from Exp 303.
     If build times out: honest_verdict="timeout".
     If build fails (non-timeout): honest_verdict="blocked_build".
  3. [INSTALL WHEEL] Install the freshly built .whl into .venv-npu.
  4. [INFERENCE TEST] Load jepa_predictor_291.onnx (fallback: 146.onnx) via
     VitisAIExecutionProvider. Run WARMUP_CALLS=20 + TIMED_CALLS=100.
     Record npu_latency_us, cpu_latency_us, speedup_factor, provider_used.
     If VitisAI EP is not available after the source build: "blocked_abi".
  5. Emit results/experiment_314_npu_prereq_install.json with honest_verdict.
  6. Update research-hardware-wishlist.md AMD XDNA section (additive only).

Key differences from Exp 303:
  - experiment=314, output file is experiment_314_npu_prereq_install.json
  - New field: prereq_changes — dict showing which packages changed since Exp 303
    (values: "now_available" or "still_missing")
  - honest_verdict includes "timeout" as a distinct value (separate from "blocked_build")
    so the researcher knows whether to increase the build timeout vs fix compile errors.
  - BUILD_DIR is /tmp/ort_build_314 (fresh build, avoids stale 303 artifacts)

Writes:
    results/experiment_314_npu_prereq_install.json

Spec: REQ-PRED-003
SCENARIO-EXP303-A (prereq check — ninja and openblas detection with install_command)
SCENARIO-EXP303-B (source build path — attempt with 45-min timeout, log tail on failure)
SCENARIO-EXP303-C (inference benchmark — npu_latency_us vs cpu_latency_us when working)
SCENARIO-EXP303-D (honest labeling — null inference_result on all blocked paths)

Run:
    cd /home/ianblenke/github.com/ianblenke/carnot
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_314_npu_prereq_install.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup — add scripts/ dir so we can import experiment_303 helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# Import prereq detection and benchmark helpers from Exp 303.
# These functions are purely functional (no global side-effects on import):
#   _collect_prereq_check, _select_onnx_model, _install_wheel_into_venv,
#   _run_inference_benchmark
# We also need the path constants for .venv-npu, VitisAI .so dir, vaip config.
import experiment_303_npu_unblock as exp303  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT: int = 314
RUN_DATE: str = "20260414"

# Source build timeout: 45 minutes (same as Exp 303)
BUILD_TIMEOUT_SECONDS: int = 45 * 60

# Benchmark parameters (same as Exp 303)
TIMED_CALLS: int = 100
WARMUP_CALLS: int = 20

# ORT version
ORT_TARGET_VERSION: str = "1.20.1"
ORT_GIT_TAG: str = "v1.20.1"
ORT_GIT_URL: str = "https://github.com/microsoft/onnxruntime.git"

# Repo layout
_RESULTS_DIR = _REPO_ROOT / "results"
_OUTPUT_FILE = _RESULTS_DIR / "experiment_314_npu_prereq_install.json"
_WISHLIST_MD = _REPO_ROOT / "research-hardware-wishlist.md"

# Build dir — fresh directory to avoid stale Exp 303 artifacts.
# If the Exp 303 build dir already has a successful build we could reuse it,
# but starting fresh gives us a clean signal about whether the env is now correct.
_BUILD_DIR = Path("/tmp/ort_build_314")

# Exp 303 prior state — what was blocked there.
# We hardcode this because the Exp 303 JSON is the ground truth for the prior state.
_EXP303_PREREQ_STATE = {
    "ninja_installed": False,
    "openblas_installed": False,
}


# ---------------------------------------------------------------------------
# prereq_changes computation
# ---------------------------------------------------------------------------


def _compute_prereq_changes(
    current_check: dict[str, Any],
    prior_check: dict[str, bool] | None = None,
) -> dict[str, str]:
    """Compare current prereq state to Exp 303's blocked state.

    WHY: Exp 303 was blocked because ninja and openblas were both missing.
    This function tells the researcher, at a glance, which of those packages
    have been installed since Exp 303 ran.  Without this, the researcher would
    need to diff two JSON files manually.

    Args:
        current_check: The output of _collect_prereq_check() for this run.
        prior_check: The Exp 303 prereq state (defaults to _EXP303_PREREQ_STATE).

    Returns:
        A dict with "ninja" and "openblas" keys, each either "now_available"
        or "still_missing".
    """
    if prior_check is None:
        prior_check = _EXP303_PREREQ_STATE

    # We only report changes for packages that were blocked in the prior run.
    # ninja: was False in Exp 303
    ninja_change = (
        "now_available" if current_check["ninja_installed"] else "still_missing"
    )
    # openblas: was False in Exp 303
    openblas_change = (
        "now_available" if current_check["openblas_installed"] else "still_missing"
    )

    return {"ninja": ninja_change, "openblas": openblas_change}


# ---------------------------------------------------------------------------
# Source build (same logic as Exp 303, using _BUILD_DIR = /tmp/ort_build_314)
# ---------------------------------------------------------------------------


def _attempt_source_build_314() -> dict[str, Any]:
    """Attempt to build onnxruntime 1.20.1 from source with VitisAI EP enabled.

    This is the same logic as exp303._attempt_source_build() but uses
    _BUILD_DIR = /tmp/ort_build_314 so this run gets a clean slate.

    Steps:
      1. Clone onnxruntime at tag v1.20.1 to /tmp/ort_build_314 (skipped if present).
      2. Run cmake -DONNXRUNTIME_USE_VITISAI=ON with XRT and VitisAI paths.
      3. Run cmake --build with a 45-minute hard timeout.
      4. If successful, locate and return path to the built .whl file.

    Returns a build_outcome dict with:
      success (bool)
      duration_seconds (float)
      whl_path (str|None)         — path to .whl if success=True
      error_summary (str)         — present when success=False
      build_log_tail (list[str])  — last 50 build output lines when success=False
      timeout_exceeded (bool)     — True if the 45-min limit was hit
    """
    venv_python = exp303._VENV_NPU / "bin" / "python"
    vitisai_so_dir = exp303._VITISAI_SO_DIR

    build_log_lines: list[str] = []
    start_time = time.monotonic()

    # Step 1: Clone if needed
    if not _BUILD_DIR.exists():
        print(f"  Cloning onnxruntime {ORT_GIT_TAG} to {_BUILD_DIR} ...")
        clone_result = subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                ORT_GIT_TAG,
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
    vitisai_ep_lib = str(vitisai_so_dir)

    cmake_cmd = [
        "cmake",
        str(_BUILD_DIR),
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DONNXRUNTIME_USE_VITISAI=ON",
        f"-DONNXRUNTIME_VITISAI_EP_LIBRARY_PATH={vitisai_ep_lib}",
        "-DXRT_ROOT=/opt/xilinx/xrt",
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
        tail = [line for line in build_log_lines[-50:] if line]
        return {
            "success": False,
            "duration_seconds": round(elapsed, 1),
            "error_summary": (
                f"cmake configure failed (rc={configure_result.returncode}): "
                f"{err[:300]}"
            ),
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
            "build_log_tail": [line for line in build_log_lines[-50:] if line],
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
            tail = [line for line in build_log_lines[-50:] if line]
            return {
                "success": False,
                "duration_seconds": round(elapsed, 1),
                "error_summary": (
                    f"cmake --build failed (rc={build_result.returncode}): "
                    f"{err[:300]}"
                ),
                "build_log_tail": tail if tail else [err[:300]],
                "timeout_exceeded": False,
                "whl_path": None,
            }

    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start_time
        tail = [line for line in build_log_lines[-50:] if line]
        return {
            "success": False,
            "duration_seconds": round(elapsed, 1),
            "error_summary": f"cmake --build timed out after {BUILD_TIMEOUT_SECONDS}s",
            "build_log_tail": (
                tail if tail else [f"timeout after {BUILD_TIMEOUT_SECONDS}s"]
            ),
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

    # No wheel found even with rc=0 — treat as failure.
    return {
        "success": False,
        "duration_seconds": round(elapsed, 1),
        "error_summary": (
            "cmake --build succeeded but no .whl found in build directory"
        ),
        "build_log_tail": [line for line in build_log_lines[-50:] if line],
        "timeout_exceeded": False,
        "whl_path": None,
    }


# ---------------------------------------------------------------------------
# Next steps builder
# ---------------------------------------------------------------------------


def _build_next_steps(
    prereq_check: dict[str, Any],
    prereq_changes: dict[str, str],
    honest_verdict: str,
) -> list[str]:
    """Build a human-readable list of next steps given the current state.

    WHY: The artifact is read by humans scanning for what to do next.
    Including next steps directly in the JSON saves a trip to the CLAUDE.md.
    """
    steps: list[str] = []

    if honest_verdict == "blocked_prereq":
        if not prereq_check["ninja_installed"]:
            cmd = prereq_check.get(
                "ninja_install_command",
                "sudo pacman -S ninja  OR  sudo apt install ninja-build",
            )
            steps.append(f"Install ninja: {cmd}")
        if not prereq_check["openblas_installed"]:
            cmd = prereq_check.get(
                "openblas_install_command",
                "sudo pacman -S openblas  OR  sudo apt install libopenblas-dev",
            )
            steps.append(f"Install openblas: {cmd}")
        steps.append(
            "Then re-run: "
            "JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_314_npu_prereq_install.py"
        )
    elif honest_verdict == "timeout":
        steps.append(
            "cmake --build timed out after 45 minutes — "
            "try: export MAKEFLAGS=-j$(nproc) before re-running"
        )
        steps.append(
            "Re-run: "
            "JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_314_npu_prereq_install.py"
        )
    elif honest_verdict == "blocked_build":
        steps.append(
            "cmake --build failed — check build_outcome.build_log_tail for compile errors"
        )
        steps.append(
            "Re-run after fixing build errors: "
            "JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_314_npu_prereq_install.py"
        )
    elif honest_verdict == "npu_working":
        steps.append("NPU is working! Proceed to Exp 315 for full benchmark comparison.")
        steps.append(
            "Add REQ-PRED-004 and SCENARIO-EXP314-A to "
            "openspec/capabilities/training-inference/spec.md"
        )

    return steps


# ---------------------------------------------------------------------------
# Hardware wishlist updater
# ---------------------------------------------------------------------------


def _update_hardware_wishlist(
    honest_verdict: str,
    prereq_changes: dict[str, str],
    details: dict[str, Any],
) -> None:
    """Append Exp 314 findings to the AMD XDNA section of research-hardware-wishlist.md.

    WHY: The wishlist is the canonical place for hardware status. Every experiment
    that touches NPU must update it so the researcher can read one file to know
    the current state of NPU bring-up.  We only ADD content — never remove existing.

    Args:
        honest_verdict: The Exp 314 outcome (blocked_prereq/blocked_build/timeout/npu_working).
        prereq_changes: The delta from Exp 303 (ninja/openblas now_available/still_missing).
        details: The full artifact dict for accessing prereq_check and inference_result.
    """
    if not _WISHLIST_MD.exists():
        return

    content = _WISHLIST_MD.read_text()

    # Build findings block
    lines: list[str] = [
        f"\n### AMD XDNA NPU Status (Exp 314 — {RUN_DATE})\n",
    ]

    pc = details.get("prereq_check", {})

    if honest_verdict == "blocked_prereq":
        ninja_status = prereq_changes.get("ninja", "still_missing")
        openblas_status = prereq_changes.get("openblas", "still_missing")
        lines.append(
            f"- **Exp 314 result:** `honest_verdict=blocked_prereq`"
        )
        lines.append(f"  - ninja: {ninja_status}")
        lines.append(f"  - openblas: {openblas_status}")
        lines.append(
            "  - Both were also missing in Exp 303 — packages have NOT been installed yet."
        )
        if not pc.get("ninja_installed"):
            cmd = pc.get(
                "ninja_install_command",
                "sudo pacman -S ninja  OR  sudo apt install ninja-build",
            )
            lines.append(f"  - Install ninja: `{cmd}`")
        if not pc.get("openblas_installed"):
            cmd = pc.get(
                "openblas_install_command",
                "sudo pacman -S openblas  OR  sudo apt install libopenblas-dev",
            )
            lines.append(f"  - Install openblas: `{cmd}`")

    elif honest_verdict == "timeout":
        lines.append("- **Exp 314 result:** `honest_verdict=timeout`")
        lines.append(
            "  - Prereqs are now installed (ninja + openblas both available)"
        )
        lines.append(
            "  - ORT source build started but hit the 45-minute timeout"
        )
        lines.append(
            "  - Try: `export MAKEFLAGS=-j$(nproc)` before re-running"
        )

    elif honest_verdict == "blocked_build":
        lines.append("- **Exp 314 result:** `honest_verdict=blocked_build`")
        lines.append("  - Prereqs available but cmake --build failed (non-timeout)")
        lines.append(
            "  - Check `results/experiment_314_npu_prereq_install.json` "
            "build_outcome.build_log_tail for compile errors"
        )

    elif honest_verdict == "npu_working":
        ir = details.get("inference_result", {}) or {}
        lines.append("- **Exp 314 result:** `honest_verdict=npu_working` — NPU WORKING!")
        lines.append(f"  - NPU latency: {ir.get('npu_latency_us', '?'):.3f} µs/call")
        lines.append(f"  - CPU latency: {ir.get('cpu_latency_us', '?'):.3f} µs/call")
        lines.append(f"  - Speedup: {ir.get('speedup_factor', '?'):.2f}x")
        lines.append(f"  - Provider: {ir.get('provider_used', '?')}")

    findings_block = "\n".join(lines)

    # Append to end of file (additive — never remove existing content)
    if findings_block.strip() not in content:
        _WISHLIST_MD.write_text(content.rstrip() + "\n" + findings_block + "\n")
        print(f"  Updated {_WISHLIST_MD.name} with Exp 314 findings.")
    else:
        print(f"  {_WISHLIST_MD.name} already contains Exp 314 findings — skipping update.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Exp 314: AMD XDNA NPU prereq retry — check prereqs, build, benchmark."""
    print(f"[Exp {EXPERIMENT}] AMD XDNA NPU prereq retry — starting {RUN_DATE}")
    print("=" * 70)

    # -------------------------------------------------------------------
    # Step 1: Prereq check (SCENARIO-EXP303-A)
    # -------------------------------------------------------------------
    print("\n[Step 1] Checking prerequisites (reusing Exp 303 detection logic) ...")
    prereq_check = exp303._collect_prereq_check()

    ninja_ok = prereq_check["ninja_installed"]
    openblas_ok = prereq_check["openblas_installed"]
    cmake_ok = prereq_check["cmake_sufficient"]

    print(f"  ninja_installed:    {ninja_ok}")
    print(f"  openblas_installed: {openblas_ok}")
    print(f"  cmake_sufficient:   {cmake_ok} (version: {prereq_check.get('cmake_version')})")
    print(f"  ryzen_ai_sw_present: {prereq_check.get('ryzen_ai_sw_present')}")
    print(f"  vitisai_so_present:  {prereq_check.get('vitisai_so_present')}")

    # Compute delta vs Exp 303
    prereq_changes = _compute_prereq_changes(prereq_check)
    print(f"\n  prereq_changes vs Exp 303: {prereq_changes}")

    # ONNX model selection (needed for onnx_model_considered field)
    onnx_model = exp303._select_onnx_model()
    onnx_model_str = str(onnx_model) if onnx_model else None

    # If prereqs still missing: emit blocked_prereq immediately (SCENARIO-EXP303-A)
    if not ninja_ok or not openblas_ok:
        print("\n  => Prereqs still missing — emitting blocked_prereq artifact")
        next_steps = _build_next_steps(prereq_check, prereq_changes, "blocked_prereq")
        artifact: dict[str, Any] = {
            "experiment": EXPERIMENT,
            "description": (
                "AMD XDNA NPU prereq retry — blocked: "
                "source build prerequisites still missing"
            ),
            "run_date": RUN_DATE,
            "execution_path": "blocked_prereq",
            "prereq_check": prereq_check,
            "prereq_changes": prereq_changes,
            "build_outcome": None,
            "inference_result": None,
            "honest_verdict": "blocked_prereq",
            "onnx_model_considered": onnx_model_str,
            "next_steps": next_steps,
        }
        _RESULTS_DIR.mkdir(exist_ok=True)
        _OUTPUT_FILE.write_text(json.dumps(artifact, indent=2))
        print(f"\n  => Artifact written to: {_OUTPUT_FILE}")
        _update_hardware_wishlist("blocked_prereq", prereq_changes, artifact)
        return

    # -------------------------------------------------------------------
    # Step 2: Source build (SCENARIO-EXP303-B)
    # -------------------------------------------------------------------
    print("\n[Step 2] All prereqs present — attempting ORT source build (45-min timeout) ...")
    build_outcome = _attempt_source_build_314()

    if not build_outcome["success"]:
        if build_outcome.get("timeout_exceeded"):
            honest_verdict = "timeout"
            description = (
                "AMD XDNA NPU prereq retry — prereqs available but source build timed out"
            )
        else:
            honest_verdict = "blocked_build"
            description = (
                "AMD XDNA NPU prereq retry — prereqs available but source build failed"
            )
        print(f"\n  => Build failed. honest_verdict={honest_verdict}")
        next_steps = _build_next_steps(prereq_check, prereq_changes, honest_verdict)
        artifact = {
            "experiment": EXPERIMENT,
            "description": description,
            "run_date": RUN_DATE,
            "execution_path": honest_verdict,
            "prereq_check": prereq_check,
            "prereq_changes": prereq_changes,
            "build_outcome": build_outcome,
            "inference_result": None,
            "honest_verdict": honest_verdict,
            "onnx_model_considered": onnx_model_str,
            "next_steps": next_steps,
        }
        _RESULTS_DIR.mkdir(exist_ok=True)
        _OUTPUT_FILE.write_text(json.dumps(artifact, indent=2))
        print(f"\n  => Artifact written to: {_OUTPUT_FILE}")
        _update_hardware_wishlist(honest_verdict, prereq_changes, artifact)
        return

    # -------------------------------------------------------------------
    # Step 3: Install wheel into .venv-npu
    # -------------------------------------------------------------------
    print(f"\n[Step 3] Installing wheel: {build_outcome['whl_path']} ...")
    install_ok, install_msg = exp303._install_wheel_into_venv(build_outcome["whl_path"])
    print(f"  => {install_msg}")

    if not install_ok:
        artifact = {
            "experiment": EXPERIMENT,
            "description": "AMD XDNA NPU prereq retry — build succeeded but wheel install failed",
            "run_date": RUN_DATE,
            "execution_path": "blocked_build",
            "prereq_check": prereq_check,
            "prereq_changes": prereq_changes,
            "build_outcome": build_outcome,
            "inference_result": None,
            "honest_verdict": "blocked_build",
            "onnx_model_considered": onnx_model_str,
            "next_steps": [f"Wheel install failed: {install_msg}"],
        }
        _RESULTS_DIR.mkdir(exist_ok=True)
        _OUTPUT_FILE.write_text(json.dumps(artifact, indent=2))
        print(f"\n  => Artifact written to: {_OUTPUT_FILE}")
        _update_hardware_wishlist("blocked_build", prereq_changes, artifact)
        return

    # -------------------------------------------------------------------
    # Step 4: Inference benchmark (SCENARIO-EXP303-C)
    # -------------------------------------------------------------------
    if onnx_model is None:
        print("\n[Step 4] No ONNX model found — skipping inference benchmark.")
        artifact = {
            "experiment": EXPERIMENT,
            "description": "AMD XDNA NPU prereq retry — build succeeded but no ONNX model",
            "run_date": RUN_DATE,
            "execution_path": "blocked_build",
            "prereq_check": prereq_check,
            "prereq_changes": prereq_changes,
            "build_outcome": build_outcome,
            "inference_result": None,
            "honest_verdict": "blocked_build",
            "onnx_model_considered": None,
            "next_steps": [
                "Run Exp 291 to generate jepa_predictor_291.onnx first",
                "Then re-run experiment_314_npu_prereq_install.py",
            ],
        }
        _RESULTS_DIR.mkdir(exist_ok=True)
        _OUTPUT_FILE.write_text(json.dumps(artifact, indent=2))
        print(f"\n  => Artifact written to: {_OUTPUT_FILE}")
        _update_hardware_wishlist("blocked_build", prereq_changes, artifact)
        return

    print(f"\n[Step 4] Running inference benchmark against {onnx_model.name} ...")
    benchmark_result = exp303._run_inference_benchmark(onnx_model)

    if isinstance(benchmark_result, str):
        # Benchmark returned an error string — either blocked_abi or other failure
        if "blocked_abi" in benchmark_result:
            honest_verdict = "blocked_build"  # VitisAI EP absent post-build = ABI issue
            description = "AMD XDNA NPU prereq retry — build succeeded but VitisAI EP not available (ABI mismatch)"
        else:
            honest_verdict = "blocked_build"
            description = f"AMD XDNA NPU prereq retry — inference benchmark failed: {benchmark_result[:100]}"

        print(f"\n  => Benchmark failed. honest_verdict={honest_verdict}")
        print(f"     Error: {benchmark_result[:200]}")
        next_steps = _build_next_steps(prereq_check, prereq_changes, honest_verdict)
        next_steps.insert(0, f"Benchmark error: {benchmark_result[:200]}")
        artifact = {
            "experiment": EXPERIMENT,
            "description": description,
            "run_date": RUN_DATE,
            "execution_path": honest_verdict,
            "prereq_check": prereq_check,
            "prereq_changes": prereq_changes,
            "build_outcome": build_outcome,
            "inference_result": None,
            "honest_verdict": honest_verdict,
            "onnx_model_considered": onnx_model_str,
            "next_steps": next_steps,
        }
        _RESULTS_DIR.mkdir(exist_ok=True)
        _OUTPUT_FILE.write_text(json.dumps(artifact, indent=2))
        print(f"\n  => Artifact written to: {_OUTPUT_FILE}")
        _update_hardware_wishlist(honest_verdict, prereq_changes, artifact)
        return

    # NPU is working! (SCENARIO-EXP303-C)
    inference_result = benchmark_result
    print(
        f"\n  => NPU WORKING! "
        f"npu={inference_result['npu_latency_us']:.3f}µs  "
        f"cpu={inference_result['cpu_latency_us']:.3f}µs  "
        f"speedup={inference_result['speedup_factor']:.2f}x"
    )

    artifact = {
        "experiment": EXPERIMENT,
        "description": "AMD XDNA NPU prereq retry — NPU WORKING via VitisAI EP",
        "run_date": RUN_DATE,
        "execution_path": "npu_working",
        "prereq_check": prereq_check,
        "prereq_changes": prereq_changes,
        "build_outcome": build_outcome,
        "inference_result": inference_result,
        "honest_verdict": "npu_working",
        "onnx_model_considered": onnx_model_str,
        "next_steps": _build_next_steps(prereq_check, prereq_changes, "npu_working"),
    }
    _RESULTS_DIR.mkdir(exist_ok=True)
    _OUTPUT_FILE.write_text(json.dumps(artifact, indent=2))
    print(f"\n  => Artifact written to: {_OUTPUT_FILE}")
    _update_hardware_wishlist("npu_working", prereq_changes, artifact)
    print("\n  If honest_verdict=npu_working: add REQ-PRED-004 and SCENARIO-EXP314-A to spec.")


if __name__ == "__main__":
    main()

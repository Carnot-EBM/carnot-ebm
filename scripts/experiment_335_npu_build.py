#!/usr/bin/env python3
"""Experiment 335: AMD XDNA NPU build — third prereq retry with ORT source build.

This is the third attempt to bring up the AMD XDNA NPU via a VitisAI-enabled
onnxruntime source build.  Exps 292, 303, and 314 were all blocked at the same
point: missing `ninja` and `openblas` system packages.

WHY this experiment exists (not just another retry):
  - Exp 314 (20260414) confirmed the packages were STILL missing.
  - The researcher has been asked to install them manually between milestones.
  - Exp 335 detects whether that install happened and either:
      (a) Emits a "blocked_prereq" artifact with the exact install commands if
          still missing (SCENARIO-EXP303-E).
      (b) Proceeds to the ORT source build and inference test if prereqs are met
          (SCENARIO-EXP303-F).
  - New field: `prereq_changes_vs_exp314` — tells the researcher at a glance
    which packages changed from Exp 314's blocked state without diffing files.
  - New distinction in verdict vocabulary:
      "blocked_prereq"  — prereqs not met, build not attempted
      "build_failed"    — prereqs met, build tried but cmake failed
      "timeout"         — prereqs met, build tried but hit the time limit
      "inference_success" — full NPU inference succeeded end-to-end

Key differences from Exp 314:
  - experiment=335, output = results/experiment_335_npu_build.json
  - Four individual check functions (check_ninja_available, check_openblas_available,
    check_xrt_available, check_amdxdna_module_loaded) are importable for unit tests.
  - prereq_status() aggregates all four into one dict; main() calls only that.
  - Build directory is /tmp/ort_build_335 (fresh, avoids stale Exp 314 artifacts).
  - Top-level artifact key changed: build_outcome → build_attempt_result,
    inference_result → npu_inference_result (clearer names for downstream parsers).

Writes:
    results/experiment_335_npu_build.json

Spec:
  REQ-PRED-003
  SCENARIO-EXP303-A (prereq check — ninja, openblas, XRT, amdxdna detection)
  SCENARIO-EXP303-B (source build — 10-min timeout, log tail on failure)
  SCENARIO-EXP303-C (inference benchmark — npu_latency_us vs cpu_latency_us)
  SCENARIO-EXP303-D (honest labeling — null npu_inference_result on blocked paths)
  SCENARIO-EXP303-E (still blocked — same state as Exp 314, no build attempted)
  SCENARIO-EXP303-F (build attempted — prereqs now met, build_attempt_result present)

Run:
    cd /home/ianblenke/github.com/ianblenke/carnot
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_335_npu_build.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT: int = 335
RUN_DATE: str = "20260415"

# ORT source build timeout (10 minutes — shorter than prior experiments as a
# fast-fail signal; if prereqs are now present the configure step should be fast).
BUILD_TIMEOUT_SECONDS: int = 10 * 60

ORT_GIT_TAG: str = "v1.20.1"
ORT_GIT_URL: str = "https://github.com/microsoft/onnxruntime.git"

# Filesystem paths
_RESULTS_DIR = _REPO_ROOT / "results"
_OUTPUT_FILE = _RESULTS_DIR / "experiment_335_npu_build.json"
_WISHLIST_MD = _REPO_ROOT / "research-hardware-wishlist.md"
_BUILD_DIR = Path("/tmp/ort_build_335")

# XRT installation directory — checked by check_xrt_available()
_XRT_DIR = Path("/opt/xilinx/xrt")

# Exp 314 prior state (hardcoded from results/experiment_314_npu_prereq_install.json).
# WHY: We compare against a known baseline rather than re-reading the JSON at runtime
# so the comparison works even if the file is missing (e.g., in CI).
_EXP314_PRIOR: dict[str, bool] = {
    "ninja_installed": False,
    "openblas_installed": False,
}

# ONNX model produced by Exp 291 — used for inference test if build succeeds.
_ONNX_MODEL = _RESULTS_DIR / "jepa_predictor_291.onnx"


# ---------------------------------------------------------------------------
# Individual prereq check functions (importable for unit tests)
# ---------------------------------------------------------------------------


def check_ninja_available() -> bool:
    """Return True if `ninja` build tool is installed and executable.

    WHY: ninja is required by the cmake -G Ninja build generator used in the
    ORT source build.  Its absence caused every previous NPU build attempt to
    fail at the cmake configure step before any compilation occurred.

    Detection: run `ninja --version`; any non-zero exit or missing binary → False.
    """
    try:
        result = subprocess.run(
            ["ninja", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_openblas_available() -> bool:
    """Return True if openblas shared library is available on the system.

    WHY: onnxruntime's cmake build with -Donnxruntime_USE_OPENBLAS=ON links
    against libopenblas at compile time.  If the library headers and shared
    object are absent, cmake configure fails with a linker search error.

    Detection strategy (in order):
    1. `pkg-config --exists openblas` — authoritative on systems that install
       openblas with pkg-config support (most Arch and Debian packages do).
    2. `ldconfig -p | grep libopenblas` — fallback for systems where pkg-config
       is not configured but the .so is in the linker cache.
    Both probes returning nothing → False.
    """
    try:
        result = subprocess.run(
            ["pkg-config", "--exists", "openblas"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: ldconfig
    try:
        result = subprocess.run(
            ["ldconfig", "-p"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0 and "libopenblas" in result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_xrt_available() -> bool:
    """Return True if the Xilinx Runtime (XRT) installation directory exists.

    WHY: XRT is the low-level runtime that communicates with the XDNA NPU
    firmware.  cmake requires -DXRT_ROOT=/opt/xilinx/xrt to locate XRT headers
    and libraries.  If the directory does not exist, the cmake configure step
    fails immediately with a missing include path error.

    Detection: filesystem existence check — faster and more reliable than a
    subprocess probe for a directory that either exists or doesn't.
    """
    return _XRT_DIR.is_dir()


def check_amdxdna_module_loaded() -> bool:
    """Return True if the amdxdna kernel module is currently loaded.

    WHY: Even when XRT is installed, the NPU requires the amdxdna kernel
    driver to be active.  If the module is not loaded, VitisAI EP will fail
    at inference time (not at build time), producing a confusing "device not
    found" error.  Detecting this early lets us report a more specific status.

    Detection: parse `lsmod` output for 'amdxdna'.  lsmod is world-readable
    and does not require root access.
    """
    try:
        result = subprocess.run(
            ["lsmod"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0 and "amdxdna" in result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ---------------------------------------------------------------------------
# Aggregate prereq check
# ---------------------------------------------------------------------------


def prereq_status() -> dict[str, Any]:
    """Run all four prereq checks and return an aggregate status dict.

    WHY: Centralising the checks into one function makes main() logic simple
    (one call, one dict) and makes the return value easily serializable into
    the artifact JSON.

    Returns a dict with keys:
      ninja_available (bool): output of check_ninja_available()
      openblas_available (bool): output of check_openblas_available()
      xrt_available (bool): output of check_xrt_available()
      amdxdna_module_loaded (bool): output of check_amdxdna_module_loaded()
      all_met (bool): True only when all four checks are True

    Spec: SCENARIO-EXP303-A
    """
    ninja = check_ninja_available()
    openblas = check_openblas_available()
    xrt = check_xrt_available()
    amdxdna = check_amdxdna_module_loaded()
    return {
        "ninja_available": ninja,
        "openblas_available": openblas,
        "xrt_available": xrt,
        "amdxdna_module_loaded": amdxdna,
        "all_met": ninja and openblas and xrt,
        # NOTE: amdxdna is informational only — we do not block the build on it.
        # The ORT build works without the module; the module is only needed at
        # inference time.  We still report it so the researcher knows the full state.
    }


# ---------------------------------------------------------------------------
# Prereq delta vs Exp 314
# ---------------------------------------------------------------------------


def prereq_changes_vs_exp314(status: dict[str, Any]) -> dict[str, str]:
    """Compare current prereq state to Exp 314's blocked state.

    WHY: Exp 314 (20260414) had both ninja_installed=False and
    openblas_installed=False.  This function produces an at-a-glance summary
    of what has changed — or not changed — since then, so the researcher does
    not need to diff two JSON files.

    Args:
        status: The dict returned by prereq_status() for this run.

    Returns:
        A dict with exactly two keys ("ninja", "openblas"), each with value
        "now_available" (package is now present) or "still_missing" (not yet).
        This controlled vocabulary simplifies downstream parsing.

    Spec: SCENARIO-EXP303-E, SCENARIO-EXP303-F
    """
    ninja_change = "now_available" if status["ninja_available"] else "still_missing"
    openblas_change = "now_available" if status["openblas_available"] else "still_missing"
    return {"ninja": ninja_change, "openblas": openblas_change}


# ---------------------------------------------------------------------------
# ORT source build
# ---------------------------------------------------------------------------


def attempt_ort_source_build(
    build_dir: Path,
    timeout_s: int = BUILD_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Attempt to build onnxruntime 1.20.1 from source with VitisAI EP.

    WHY: Pre-built onnxruntime wheels do not include the VitisAI execution
    provider because it depends on XRT and VitisAI libraries that are not
    present on generic Linux systems.  A source build with the right cmake
    flags produces a wheel that links against the local XRT installation.

    Steps:
      1. Clone onnxruntime at tag v1.20.1 to `build_dir` (skip if already present).
      2. cmake configure with -G Ninja -DONNXRUNTIME_USE_VITISAI=ON and XRT paths.
      3. cmake --build with the given timeout.
      4. Locate the produced .whl file in the build directory.

    Returns a dict with:
      success (bool)
      duration_seconds (float): wall time of the entire build sequence
      whl_path (str | None): path to the built .whl if success=True
      error_summary (str): present when success=False
      build_log_tail (list[str]): last ≤50 lines of build output when success=False
      timeout_exceeded (bool): True only when the subprocess TimeoutExpired

    Spec: SCENARIO-EXP303-B, SCENARIO-EXP303-F
    """
    build_log_lines: list[str] = []
    start_time = time.monotonic()

    def _elapsed() -> float:
        return round(time.monotonic() - start_time, 1)

    def _tail(extra: list[str] | None = None) -> list[str]:
        lines = build_log_lines + (extra or [])
        return [ln for ln in lines[-50:] if ln]

    # Step 1: Clone if the target directory does not exist yet.
    if not build_dir.exists():
        print(f"  Cloning onnxruntime {ORT_GIT_TAG} to {build_dir} ...")
        clone_result = subprocess.run(
            [
                "git", "clone",
                "--depth", "1",
                "--branch", ORT_GIT_TAG,
                ORT_GIT_URL,
                str(build_dir),
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if clone_result.returncode != 0:
            err = (clone_result.stderr or clone_result.stdout)[:500]
            return {
                "success": False,
                "duration_seconds": _elapsed(),
                "whl_path": None,
                "error_summary": f"git clone failed: {err}",
                "build_log_tail": _tail(err.splitlines()[-50:]),
                "timeout_exceeded": False,
            }
        build_log_lines.append(f"git clone succeeded for {ORT_GIT_TAG}")
    else:
        build_log_lines.append(f"Using existing clone at {build_dir}")

    # Step 2: cmake configure
    cmake_build_dir = build_dir / "build_vitisai"
    cmake_build_dir.mkdir(exist_ok=True)

    cmake_cmd = [
        "cmake", str(build_dir),
        "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DONNXRUNTIME_USE_VITISAI=ON",
        f"-DONNXRUNTIME_VITISAI_EP_LIBRARY_PATH={_XRT_DIR.parent / 'vart' / 'vitisai'}",
        "-DXRT_ROOT=/opt/xilinx/xrt",
        f"-DXRT_INCLUDE_DIR={_XRT_DIR / 'include'}",
        f"-DXRT_LIB_DIR={_XRT_DIR / 'lib'}",
        "-Donnxruntime_USE_OPENBLAS=ON",
        "-Donnxruntime_BUILD_WHEEL=ON",
    ]

    print("  Running cmake configure ...")
    try:
        cfg_result = subprocess.run(
            cmake_cmd,
            capture_output=True,
            text=True,
            cwd=str(cmake_build_dir),
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "duration_seconds": _elapsed(),
            "whl_path": None,
            "error_summary": "cmake configure timed out after 300s",
            "build_log_tail": _tail(),
            "timeout_exceeded": True,
        }

    build_log_lines.extend(cfg_result.stdout.splitlines()[-50:])
    build_log_lines.extend(cfg_result.stderr.splitlines()[-20:])

    if cfg_result.returncode != 0:
        err = (cfg_result.stderr or cfg_result.stdout)[:500]
        return {
            "success": False,
            "duration_seconds": _elapsed(),
            "whl_path": None,
            "error_summary": f"cmake configure failed (rc={cfg_result.returncode}): {err[:300]}",
            "build_log_tail": _tail(),
            "timeout_exceeded": False,
        }

    # Step 3: cmake --build with the overall timeout
    remaining = timeout_s - (time.monotonic() - start_time)
    if remaining <= 0:
        return {
            "success": False,
            "duration_seconds": _elapsed(),
            "whl_path": None,
            "error_summary": "cmake configure consumed entire timeout budget",
            "build_log_tail": _tail(),
            "timeout_exceeded": True,
        }

    print(f"  Running cmake --build (timeout={int(remaining)}s) ...")
    try:
        build_result = subprocess.run(
            ["cmake", "--build", ".", "--parallel"],
            capture_output=True,
            text=True,
            cwd=str(cmake_build_dir),
            timeout=int(remaining),
        )
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "duration_seconds": _elapsed(),
            "whl_path": None,
            "error_summary": f"cmake --build timed out after {timeout_s}s",
            "build_log_tail": _tail(),
            "timeout_exceeded": True,
        }

    build_log_lines.extend(build_result.stdout.splitlines()[-50:])
    build_log_lines.extend(build_result.stderr.splitlines()[-20:])

    if build_result.returncode != 0:
        err = (build_result.stderr or build_result.stdout)[:500]
        return {
            "success": False,
            "duration_seconds": _elapsed(),
            "whl_path": None,
            "error_summary": f"cmake --build failed (rc={build_result.returncode}): {err[:300]}",
            "build_log_tail": _tail(),
            "timeout_exceeded": False,
        }

    # Step 4: Locate the produced wheel
    whl_candidates = list(cmake_build_dir.rglob("dist/*.whl")) + list(
        cmake_build_dir.rglob("*.whl")
    )
    if whl_candidates:
        whl_path = str(whl_candidates[0])
        print(f"  Build succeeded. Wheel: {whl_path}")
        return {
            "success": True,
            "duration_seconds": _elapsed(),
            "whl_path": whl_path,
            "timeout_exceeded": False,
        }

    # Build claimed success but no wheel found — treat as build_failed.
    return {
        "success": False,
        "duration_seconds": _elapsed(),
        "whl_path": None,
        "error_summary": "cmake --build succeeded but no .whl found in build directory",
        "build_log_tail": _tail(),
        "timeout_exceeded": False,
    }


# ---------------------------------------------------------------------------
# Next-steps builder
# ---------------------------------------------------------------------------


def _build_next_steps(
    ps: dict[str, Any],
    changes: dict[str, str],
    verdict: str,
) -> list[str]:
    """Build a human-readable list of next steps given the current state.

    WHY: The artifact is the primary communication channel between the
    autonomous agent and the human researcher.  Embedding next steps in the
    JSON means the researcher can act on the artifact without reading CLAUDE.md.

    Args:
        ps: Output of prereq_status().
        changes: Output of prereq_changes_vs_exp314().
        verdict: The honest_verdict being emitted.
    """
    steps: list[str] = []

    if verdict == "blocked_prereq":
        if not ps["ninja_available"]:
            steps.append(
                "Install ninja: sudo pacman -S ninja  (Arch)  "
                "OR  sudo apt install ninja-build  (Debian/Ubuntu)"
            )
        if not ps["openblas_available"]:
            steps.append(
                "Install openblas: sudo pacman -S openblas  (Arch)  "
                "OR  sudo apt install libopenblas-dev  (Debian/Ubuntu)"
            )
        steps.append(
            "Then re-run: JAX_PLATFORMS=cpu .venv/bin/python "
            "scripts/experiment_335_npu_build.py"
        )
    elif verdict == "timeout":
        steps.append(
            "cmake --build timed out — try: "
            "export MAKEFLAGS=-j$(nproc) then re-run experiment_335_npu_build.py"
        )
    elif verdict == "build_failed":
        steps.append(
            "cmake --build failed — check build_attempt_result.build_log_tail "
            "for compile errors, then re-run"
        )
    elif verdict == "inference_success":
        steps.append("NPU is working! Run Exp 336 for full benchmark comparison.")

    return steps


# ---------------------------------------------------------------------------
# Hardware wishlist updater
# ---------------------------------------------------------------------------


def _update_hardware_wishlist(
    verdict: str,
    changes: dict[str, str],
    details: dict[str, Any],
) -> None:
    """Append Exp 335 findings to the AMD XDNA section of research-hardware-wishlist.md.

    WHY: The wishlist is the single source of truth for hardware bring-up status.
    Every NPU experiment must update it so the researcher can read one file to
    see the full history.  This function only ADDS content — never removes.

    Spec: Documentation Update Rules (MANDATORY)
    """
    if not _WISHLIST_MD.exists():
        return

    content = _WISHLIST_MD.read_text()

    lines: list[str] = [f"\n### AMD XDNA NPU Status (Exp 335 — {RUN_DATE})\n"]
    ps = details.get("prereq_status", {})

    if verdict == "blocked_prereq":
        lines.append(f"- **Exp 335 result:** `honest_verdict=blocked_prereq`")
        ninja_s = changes.get("ninja", "still_missing")
        openblas_s = changes.get("openblas", "still_missing")
        lines.append(f"  - ninja: {ninja_s} (was missing in Exp 314)")
        lines.append(f"  - openblas: {openblas_s} (was missing in Exp 314)")
        if not ps.get("ninja_available"):
            lines.append(
                "  - Install ninja: "
                "`sudo pacman -S ninja  OR  sudo apt install ninja-build`"
            )
        if not ps.get("openblas_available"):
            lines.append(
                "  - Install openblas: "
                "`sudo pacman -S openblas  OR  sudo apt install libopenblas-dev`"
            )
        lines.append(
            "  - Blocked for 4 milestones (Exps 292, 303, 314, 335). "
            "Human install required before next attempt."
        )

    elif verdict == "timeout":
        lines.append("- **Exp 335 result:** `honest_verdict=timeout`")
        lines.append("  - Prereqs now installed (ninja + openblas available)")
        lines.append("  - ORT source build hit the 10-minute timeout")
        lines.append("  - Try: `export MAKEFLAGS=-j$(nproc)` before re-running")

    elif verdict == "build_failed":
        bar = details.get("build_attempt_result") or {}
        lines.append("- **Exp 335 result:** `honest_verdict=build_failed`")
        lines.append("  - Prereqs available; cmake --build failed (non-timeout)")
        lines.append(
            "  - Check `results/experiment_335_npu_build.json` "
            "build_attempt_result.build_log_tail for compile errors"
        )
        err = bar.get("error_summary", "")
        if err:
            lines.append(f"  - Error summary: {err[:200]}")

    elif verdict == "inference_success":
        nir = details.get("npu_inference_result") or {}
        lines.append("- **Exp 335 result:** `honest_verdict=inference_success` — NPU WORKING!")
        lines.append(f"  - NPU latency: {nir.get('npu_latency_us', '?'):.3f} µs/call")
        lines.append(f"  - CPU latency: {nir.get('cpu_latency_us', '?'):.3f} µs/call")
        lines.append(f"  - Speedup: {nir.get('speedup_factor', '?'):.2f}x")
        lines.append(f"  - Provider: {nir.get('provider_used', '?')}")

    findings_block = "\n".join(lines)
    if findings_block.strip() not in content:
        _WISHLIST_MD.write_text(content.rstrip() + "\n" + findings_block + "\n")
        print(f"  Updated {_WISHLIST_MD.name} with Exp 335 findings.")
    else:
        print(f"  {_WISHLIST_MD.name} already contains Exp 335 findings — skipping.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Exp 335: AMD XDNA NPU build — prereq check, optional ORT build, inference."""
    print(f"[Exp {EXPERIMENT}] AMD XDNA NPU build — starting {RUN_DATE}")
    print("=" * 70)

    # -------------------------------------------------------------------
    # Step 1: Prereq check (SCENARIO-EXP303-A)
    # -------------------------------------------------------------------
    print("\n[Step 1] Checking prerequisites ...")
    ps = prereq_status()

    print(f"  ninja_available:       {ps['ninja_available']}")
    print(f"  openblas_available:    {ps['openblas_available']}")
    print(f"  xrt_available:         {ps['xrt_available']}")
    print(f"  amdxdna_module_loaded: {ps['amdxdna_module_loaded']}")
    print(f"  all_met:               {ps['all_met']}")

    changes = prereq_changes_vs_exp314(ps)
    print(f"\n  prereq_changes_vs_exp314: {changes}")

    # -------------------------------------------------------------------
    # Decision: if ninja or openblas still missing → blocked_prereq
    # (SCENARIO-EXP303-E)
    # -------------------------------------------------------------------
    if not ps["ninja_available"] or not ps["openblas_available"]:
        print("\n  => Prereqs still missing — emitting blocked_prereq artifact")
        verdict = "blocked_prereq"
        steps = _build_next_steps(ps, changes, verdict)
        artifact: dict[str, Any] = {
            "experiment": EXPERIMENT,
            "description": (
                "AMD XDNA NPU build — blocked: "
                "source build prerequisites still missing (same as Exp 314)"
            ),
            "run_date": RUN_DATE,
            "honest_verdict": verdict,
            "prereq_status": ps,
            "prereq_changes_vs_exp314": changes,
            "build_attempt_result": None,
            "npu_inference_result": None,
            "next_steps": steps,
        }
        _RESULTS_DIR.mkdir(exist_ok=True)
        _OUTPUT_FILE.write_text(json.dumps(artifact, indent=2))
        print(f"\n  => Artifact written to: {_OUTPUT_FILE}")
        _update_hardware_wishlist(verdict, changes, artifact)
        print(
            "\n  BLOCKED for the 4th consecutive milestone (Exps 292, 303, 314, 335)."
            "\n  Human action required: install ninja and openblas packages."
        )
        return

    # -------------------------------------------------------------------
    # Step 2: Source build (SCENARIO-EXP303-F)
    # -------------------------------------------------------------------
    print(
        f"\n[Step 2] All build prereqs met — attempting ORT source build "
        f"(timeout={BUILD_TIMEOUT_SECONDS}s) ..."
    )
    build_result = attempt_ort_source_build(_BUILD_DIR, timeout_s=BUILD_TIMEOUT_SECONDS)

    if not build_result["success"]:
        verdict = "timeout" if build_result.get("timeout_exceeded") else "build_failed"
        print(f"\n  => Build failed. honest_verdict={verdict}")
        steps = _build_next_steps(ps, changes, verdict)
        artifact = {
            "experiment": EXPERIMENT,
            "description": f"AMD XDNA NPU build — prereqs met but build {verdict}",
            "run_date": RUN_DATE,
            "honest_verdict": verdict,
            "prereq_status": ps,
            "prereq_changes_vs_exp314": changes,
            "build_attempt_result": build_result,
            "npu_inference_result": None,
            "next_steps": steps,
        }
        _RESULTS_DIR.mkdir(exist_ok=True)
        _OUTPUT_FILE.write_text(json.dumps(artifact, indent=2))
        print(f"\n  => Artifact written to: {_OUTPUT_FILE}")
        _update_hardware_wishlist(verdict, changes, artifact)
        return

    # -------------------------------------------------------------------
    # Step 3: Install wheel and run inference
    # -------------------------------------------------------------------
    print(f"\n[Step 3] Build succeeded. Attempting inference test ...")

    if not _ONNX_MODEL.exists():
        print(f"  ONNX model not found: {_ONNX_MODEL}")
        print("  => Emitting build_failed (no model to test with)")
        verdict = "build_failed"
        artifact = {
            "experiment": EXPERIMENT,
            "description": "AMD XDNA NPU build — build succeeded but no ONNX model",
            "run_date": RUN_DATE,
            "honest_verdict": verdict,
            "prereq_status": ps,
            "prereq_changes_vs_exp314": changes,
            "build_attempt_result": build_result,
            "npu_inference_result": None,
            "next_steps": [
                "Run Exp 291 to generate jepa_predictor_291.onnx",
                "Then re-run experiment_335_npu_build.py",
            ],
        }
        _RESULTS_DIR.mkdir(exist_ok=True)
        _OUTPUT_FILE.write_text(json.dumps(artifact, indent=2))
        print(f"\n  => Artifact written to: {_OUTPUT_FILE}")
        _update_hardware_wishlist(verdict, changes, artifact)
        return

    # Attempt inference via VitisAI EP
    npu_inference = _run_npu_inference(build_result["whl_path"])

    if npu_inference is None:
        # VitisAI EP not available after build — ABI or driver issue
        verdict = "build_failed"
        artifact = {
            "experiment": EXPERIMENT,
            "description": (
                "AMD XDNA NPU build — build succeeded but VitisAI EP unavailable at inference"
            ),
            "run_date": RUN_DATE,
            "honest_verdict": verdict,
            "prereq_status": ps,
            "prereq_changes_vs_exp314": changes,
            "build_attempt_result": build_result,
            "npu_inference_result": None,
            "next_steps": [
                "VitisAI EP not available after build — check ABI or XRT version mismatch",
                "Check: ldd <whl_path>/onnxruntime/capi/onnxruntime_providers_vitisai.so",
            ],
        }
    else:
        verdict = "inference_success"
        print(
            f"\n  => NPU WORKING! "
            f"npu={npu_inference['npu_latency_us']:.3f}µs  "
            f"cpu={npu_inference['cpu_latency_us']:.3f}µs  "
            f"speedup={npu_inference['speedup_factor']:.2f}x"
        )
        artifact = {
            "experiment": EXPERIMENT,
            "description": "AMD XDNA NPU build — NPU WORKING via VitisAI EP",
            "run_date": RUN_DATE,
            "honest_verdict": verdict,
            "prereq_status": ps,
            "prereq_changes_vs_exp314": changes,
            "build_attempt_result": build_result,
            "npu_inference_result": npu_inference,
            "next_steps": _build_next_steps(ps, changes, verdict),
        }

    _RESULTS_DIR.mkdir(exist_ok=True)
    _OUTPUT_FILE.write_text(json.dumps(artifact, indent=2))
    print(f"\n  => Artifact written to: {_OUTPUT_FILE}")
    _update_hardware_wishlist(verdict, changes, artifact)


def _run_npu_inference(whl_path: str | None) -> dict[str, Any] | None:
    """Install the built ORT wheel and run inference on jepa_predictor_291.onnx.

    WHY: We cannot import onnxruntime in the current process because it may not
    be installed (or may be a different version).  Instead, we install the newly
    built wheel into a subprocess and probe VitisAI EP availability there.

    Returns a dict with npu_latency_us, cpu_latency_us, speedup_factor,
    provider_used, and timed_calls if inference succeeded; None otherwise.

    Spec: SCENARIO-EXP303-C
    """
    # Install wheel into .venv-npu if it exists, otherwise into current venv
    venv_npu = _REPO_ROOT / ".venv-npu"
    pip = str(venv_npu / "bin" / "pip") if venv_npu.exists() else "pip"

    if whl_path:
        install_result = subprocess.run(
            [pip, "install", "--force-reinstall", whl_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if install_result.returncode != 0:
            print(f"  Wheel install failed: {install_result.stderr[:300]}")
            return None

    # Probe VitisAI EP via a short Python subprocess
    python = str(venv_npu / "bin" / "python") if venv_npu.exists() else sys.executable
    probe_script = f"""
import time, onnxruntime as ort, numpy as np, json, sys
providers = ort.get_available_providers()
if "VitisAIExecutionProvider" not in providers:
    print(json.dumps({{"error": "VitisAI EP not available", "providers": providers}}))
    sys.exit(1)

sess = ort.InferenceSession(
    "{_ONNX_MODEL}",
    providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
)
inp = sess.get_inputs()[0]
dummy = np.zeros(inp.shape, dtype=np.float32)

# Warmup
for _ in range(20):
    sess.run(None, {{inp.name: dummy}})

# Timed (NPU)
t0 = time.perf_counter()
for _ in range(100):
    sess.run(None, {{inp.name: dummy}})
npu_us = (time.perf_counter() - t0) / 100 * 1e6

# CPU baseline
cpu_sess = ort.InferenceSession("{_ONNX_MODEL}", providers=["CPUExecutionProvider"])
t1 = time.perf_counter()
for _ in range(100):
    cpu_sess.run(None, {{inp.name: dummy}})
cpu_us = (time.perf_counter() - t1) / 100 * 1e6

result = {{
    "npu_latency_us": round(npu_us, 3),
    "cpu_latency_us": round(cpu_us, 3),
    "speedup_factor": round(cpu_us / npu_us, 4),
    "provider_used": "VitisAIExecutionProvider",
    "timed_calls": 100,
}}
print(json.dumps(result))
"""
    try:
        probe_result = subprocess.run(
            [python, "-c", probe_script],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        print("  Inference probe timed out after 120s")
        return None

    if probe_result.returncode != 0:
        print(f"  Inference probe failed: {probe_result.stderr[:300]}")
        return None

    try:
        data = json.loads(probe_result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        print(f"  Could not parse probe output: {probe_result.stdout[:200]}")
        return None

    if "error" in data:
        print(f"  Probe error: {data['error']}")
        return None

    return data


if __name__ == "__main__":
    main()

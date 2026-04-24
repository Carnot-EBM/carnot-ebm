#!/usr/bin/env python3
"""Experiment 790 — NPU Unblock v9: Option A (GitHub Releases wheel) then Option B fallback.

**Researcher summary (RETRO-NPU-v9):**
    Eight consecutive milestones (Exps 292, 303, 314, 335, 435, 714) were blocked by
    the same root cause: mlir-aie is not published on PyPI, and VitisAI requires a
    compiled-in onnxruntime that is also not on PyPI.  RETRO-NPU-v8 (Exp 714) identified
    two new paths:

    Option A — Download the mlir-aie wheel directly from AMD's GitHub Releases page
    (not PyPI).  AMD publishes wheel files at:
        https://github.com/Xilinx/mlir-aie/releases/latest
    This requires no AMD account and no compiled onnxruntime — it is a pre-built
    Python wheel that can be pip-installed in < 5 minutes on a fast connection.

    Option B — Use the Ryzen AI Software installer from ryzenai.docs.amd.com.  This
    is a fallback only, because it requires an AMD account login to download and
    the automated path cannot authenticate.

    Per REQ-INFRA-057, we MUST NOT attempt more than 2 strategies per run.  Option A
    is tried first; Option B only if A fails.

**Binary verdict expected:** NPU GEMM benchmark runs (npu_gemm_runs=True) or it doesn't.

**CPU baseline (from Exp 714):** ~117 µs for 16×16 GEMM.

Spec: REQ-INFRA-057, SCENARIO-INFRA-066
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root or scripts/ dir
# ---------------------------------------------------------------------------
_REPO_ROOT = os.environ.get(
    "CARNOT_REPO_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.experiment_template import ExperimentTemplate  # noqa: E402
from python.carnot.pipeline.experiment_watchdog import (  # noqa: E402
    ExperimentTimeoutWatchdog,
)

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXP_ID = 790
TITLE = "NPU Unblock v9 — Option A (GitHub wheel) then Option B fallback"
DELIVERABLE = "results/experiment_790_npu_unblock_v9.json"
TIMEOUT_MINUTES = 45

# CPU baseline from Exp 714 (16×16 GEMM, NumPy, µs)
CPU_GEMM_US_BASELINE = 117.0

# mlir-aie GitHub API endpoint
MLIR_AIE_RELEASES_API = (
    "https://api.github.com/repos/Xilinx/mlir-aie/releases/latest"
)


# ---------------------------------------------------------------------------
# Helper: find the Python interpreter inside .venv
# ---------------------------------------------------------------------------
def _venv_python() -> str:
    """Return the path to the Python interpreter in the project .venv.

    Why .venv and not sys.executable?  The experiment runs inside the project
    virtual-environment so that pip install puts packages where our import
    statements can find them.  sys.executable might point to the system Python
    when the conductor wraps us in a subprocess.
    """
    candidates = [
        os.path.join(_REPO_ROOT, ".venv", "bin", "python"),
        os.path.join(_REPO_ROOT, ".venv", "bin", "python3"),
        sys.executable,
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return sys.executable


# ---------------------------------------------------------------------------
# Step 4: Option A — GitHub Releases wheel
# ---------------------------------------------------------------------------
def attempt_option_a() -> dict:
    """Try to install mlir-aie from the AMD GitHub Releases page.

    Why GitHub Releases and not PyPI?
        mlir-aie has never been published to PyPI (as of 2026-04-24).  AMD
        publishes pre-built wheels on GitHub Releases, which are freely
        downloadable without authentication.

    Returns a dict with:
        success (bool)   — True if pip install succeeded AND import worked.
        wheel_url (str)  — The wheel URL attempted (empty string if not found).
        pip_stderr (str) — Last pip error lines (empty on success).
        import_ok (bool) — True if `import mlir_aie` succeeded after install.
    """
    result: dict = {
        "success": False,
        "wheel_url": "",
        "pip_stderr": "",
        "import_ok": False,
    }

    # --- Query GitHub API for the latest release ---
    try:
        api_proc = subprocess.run(
            ["curl", "-s", "--max-time", "30", MLIR_AIE_RELEASES_API],
            capture_output=True,
            text=True,
            timeout=35,
        )
        release_data = json.loads(api_proc.stdout)
    except Exception as exc:
        result["pip_stderr"] = f"GitHub API query failed: {exc}"
        _log.warning("Option A: GitHub API query failed: %s", exc)
        return result

    # --- Find a wheel asset matching the current Python version ---
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    assets = release_data.get("assets", [])
    wheel_url = ""
    for asset in assets:
        name = asset.get("name", "")
        url = asset.get("browser_download_url", "")
        # Accept any .whl that matches the major.minor Python or is abi3
        if name.endswith(".whl") and ("mlir_aie" in name or "mlir-aie" in name):
            if py_tag in name or "abi3" in name or "py3" in name:
                wheel_url = url
                break
    # Fall back to first .whl if no version-tagged wheel found
    if not wheel_url:
        for asset in assets:
            name = asset.get("name", "")
            url = asset.get("browser_download_url", "")
            if name.endswith(".whl") and ("mlir_aie" in name or "mlir-aie" in name):
                wheel_url = url
                break

    if not wheel_url:
        result["pip_stderr"] = (
            f"No mlir_aie wheel found in release assets. "
            f"Release tag: {release_data.get('tag_name', 'unknown')}. "
            f"Assets: {[a.get('name') for a in assets[:10]]}"
        )
        _log.warning("Option A: %s", result["pip_stderr"])
        return result

    result["wheel_url"] = wheel_url
    _log.info("Option A: attempting pip install %s", wheel_url)

    # --- pip install the wheel ---
    pip_proc = subprocess.run(
        [_venv_python(), "-m", "pip", "install", wheel_url],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if pip_proc.returncode != 0:
        result["pip_stderr"] = pip_proc.stderr[-2000:]
        _log.warning("Option A: pip install failed (rc=%d)", pip_proc.returncode)
        return result

    # --- Verify import ---
    import_proc = subprocess.run(
        [_venv_python(), "-c", "import mlir_aie; print('import OK')"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    result["import_ok"] = import_proc.returncode == 0 and "import OK" in import_proc.stdout
    result["success"] = result["import_ok"]
    if not result["success"]:
        result["pip_stderr"] = import_proc.stderr[-1000:]
    _log.info("Option A: success=%s import_ok=%s", result["success"], result["import_ok"])
    return result


# ---------------------------------------------------------------------------
# Step 5: Option B — Ryzen AI SDK installer
# ---------------------------------------------------------------------------
def attempt_option_b() -> dict:
    """Check whether the Ryzen AI SDK installer is available locally.

    Why can't this be fully automated?
        The Ryzen AI SDK installer is hosted at ryzenai.docs.amd.com and requires
        an AMD account login.  There is no unauthenticated download URL.  So Option
        B succeeds only if a human has already downloaded the installer into
        ~/Downloads/ before the experiment runs.

    Returns a dict with:
        attempted (bool)  — Always True when this function is called.
        blocker (str|None)— Description of why it failed, or None on success.
        success (bool)    — True if installer was found and executed successfully.
    """
    result: dict = {"attempted": True, "blocker": None, "success": False}

    # Check for installer in common download locations
    search_dirs = [
        os.path.expanduser("~/Downloads"),
        os.path.expanduser("~/"),
        "/tmp",
    ]
    installer_path: str | None = None
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if "RyzenAI" in fname and (fname.endswith(".run") or fname.endswith(".sh")):
                installer_path = os.path.join(d, fname)
                break
        if installer_path:
            break

    if installer_path is None:
        result["blocker"] = (
            "Ryzen AI SDK installer not found in ~/Downloads or /tmp.  "
            "Automated download requires AMD account authentication.  "
            "A human operator must download the installer from "
            "ryzenai.docs.amd.com before this option can proceed."
        )
        _log.warning("Option B: %s", result["blocker"])
        return result

    # Installer found — attempt execution (non-interactive, extract-only if possible)
    _log.info("Option B: found installer at %s", installer_path)
    try:
        install_proc = subprocess.run(
            ["bash", installer_path, "--noexec", "--target", "/tmp/ryzenai_extract"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if install_proc.returncode != 0:
            result["blocker"] = (
                f"Installer extraction failed (rc={install_proc.returncode}): "
                + install_proc.stderr[-500:]
            )
        else:
            result["success"] = True
    except Exception as exc:
        result["blocker"] = f"Installer execution error: {exc}"

    _log.info("Option B: success=%s blocker=%s", result["success"], result["blocker"])
    return result


# ---------------------------------------------------------------------------
# Step 6: VitisAI pre-conditions check
# ---------------------------------------------------------------------------
def check_vitisai_preconditions() -> dict:
    """Check whether the system has the tools needed for a VitisAI source build.

    Why ninja and openblas?
        VitisAI's CMakeLists.txt requires ninja as the build generator and links
        against openblas for BLAS routines.  Without both, cmake configure fails
        before any C++ compilation begins.  Checking them upfront avoids a 20-minute
        wasted cmake run that errors immediately.

    Returns a dict with:
        ninja_found (bool)    — True if `which ninja` finds an executable.
        openblas_found (bool) — True if ldconfig reports a libopenblas.
    """
    ninja_proc = subprocess.run(
        ["which", "ninja"], capture_output=True, text=True, timeout=10
    )
    ninja_found = ninja_proc.returncode == 0

    openblas_proc = subprocess.run(
        ["ldconfig", "-p"], capture_output=True, text=True, timeout=10
    )
    openblas_found = "openblas" in openblas_proc.stdout.lower()

    _log.info("VitisAI pre-conditions: ninja=%s openblas=%s", ninja_found, openblas_found)
    return {"ninja_found": ninja_found, "openblas_found": openblas_found}


# ---------------------------------------------------------------------------
# Step 7: NPU GEMM benchmark
# ---------------------------------------------------------------------------
def run_npu_gemm_benchmark() -> dict:
    """Run a 16×16 GEMM on the NPU and compare to the CPU baseline.

    Why 16×16?
        The NPU GEMM must fit within the XDNA DPU tile local memory (256 KiB).
        A 16×16 float32 matrix pair occupies 2×16×16×4 = 2 048 bytes — comfortably
        within limits and matches the Exp 714 CPU baseline for a fair comparison.

    CPU baseline: ~117 µs (NumPy on Ryzen AI 9 HX 370, Exp 714).

    Returns a dict with:
        npu_gemm_runs (bool)         — True if the GEMM ran on the NPU without error.
        npu_gemm_us (float)          — Wall-clock µs for one 16×16 GEMM on the NPU.
        cpu_gemm_us (float)          — CPU baseline µs (from Exp 714 constant).
        npu_speedup_vs_cpu (float)   — cpu_gemm_us / npu_gemm_us (>1 means NPU wins).
        error (str)                  — Error description if npu_gemm_runs=False.
    """
    result: dict = {
        "npu_gemm_runs": False,
        "npu_gemm_us": 0.0,
        "cpu_gemm_us": CPU_GEMM_US_BASELINE,
        "npu_speedup_vs_cpu": 0.0,
        "error": "",
    }

    # Try to import mlir_aie and run a GEMM via its IRON API
    benchmark_code = """
import time, sys
try:
    import mlir_aie  # noqa: F401
    # IRON does not expose a pure-Python GEMM API without kernel compilation.
    # If mlir_aie imported, we at minimum confirm the driver path is open.
    # A true GEMM requires AIE kernel synthesis which needs the AIE toolchain.
    # Record that mlir_aie imported but full GEMM compilation is out of scope
    # for this automated run.
    print("mlir_aie_imported=True")
    print("npu_gemm_runs=False")
    print("error=mlir_aie imported but AIE kernel synthesis requires Vitis toolchain")
except ImportError as e:
    print(f"mlir_aie_imported=False")
    print("npu_gemm_runs=False")
    print(f"error={e}")
"""
    proc = subprocess.run(
        [_venv_python(), "-c", benchmark_code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = proc.stdout
    for line in output.splitlines():
        if line.startswith("npu_gemm_runs="):
            result["npu_gemm_runs"] = line.split("=", 1)[1].strip() == "True"
        elif line.startswith("error="):
            result["error"] = line.split("=", 1)[1].strip()

    if result["npu_gemm_runs"] and result["npu_gemm_us"] > 0:
        result["npu_speedup_vs_cpu"] = (
            CPU_GEMM_US_BASELINE / result["npu_gemm_us"]
        )
    _log.info(
        "NPU GEMM benchmark: runs=%s error=%s",
        result["npu_gemm_runs"],
        result["error"],
    )
    return result


# ---------------------------------------------------------------------------
# honest_verdict logic
# ---------------------------------------------------------------------------
def compute_honest_verdict(
    option_a_success: bool,
    option_b_attempted: bool,
    option_b_blocker: str | None,
    ninja_found: bool,
    npu_gemm_runs: bool,
    mlir_aie_import_ok: bool,
) -> str:
    """Derive a single machine-readable verdict string from the run outcomes.

    Why a single verdict string?
        The conductor's retrospective and planning steps need to grep the result
        artifact for a verdict without parsing boolean combinations.  The string
        is deterministic — given the same inputs it always returns the same value —
        so unit tests can verify it exactly.

    Verdict priority (highest to lowest):
        1. npu_gemm_running — NPU GEMM ran end-to-end (breakthrough).
        2. option_a_installed_no_benchmark — mlir-aie installed, GEMM blocked by
           missing Vitis toolchain (partial progress).
        3. all_options_exhausted_ninja_missing — both options failed and ninja is
           absent (next action: apt install ninja-build).
        4. all_options_exhausted_no_auth — both options failed due to authentication
           requirements (next action: human downloads installer).
        5. new_blocker_discovered — a failure mode not covered by previous retros.
    """
    if npu_gemm_runs:
        return "npu_gemm_running"
    if mlir_aie_import_ok:
        return "option_a_installed_no_benchmark"
    if not option_a_success and option_b_attempted:
        if not ninja_found:
            return "all_options_exhausted_ninja_missing"
        if option_b_blocker and "auth" in option_b_blocker.lower():
            return "all_options_exhausted_no_auth"
    return "new_blocker_discovered"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Orchestrate the NPU unblock v9 experiment (REQ-INFRA-057)."""
    tmpl = ExperimentTemplate(EXP_ID, TITLE, DELIVERABLE)
    tmpl.setup()

    deliverable_path = os.path.join(_REPO_ROOT, DELIVERABLE)

    with ExperimentTimeoutWatchdog(EXP_ID, TIMEOUT_MINUTES, deliverable_path):
        # --- Record v8 baseline verdict ---
        v8_result_path = os.path.join(_REPO_ROOT, "results/experiment_714_npu_iron_unblock.json")
        v8_verdict = "unknown"
        try:
            with open(v8_result_path) as f:
                v8_data = json.load(f)
            v8_verdict = v8_data.get("honest_verdict", v8_data.get("status", "unknown"))
        except Exception:
            pass

        # --- Step 4: Option A ---
        option_a = attempt_option_a()
        option_a_success: bool = option_a["success"]
        option_a_wheel_url: str = option_a["wheel_url"]
        option_a_pip_stderr: str = option_a["pip_stderr"]
        mlir_aie_import_ok: bool = option_a["import_ok"]

        # --- Step 5: Option B (only if Option A failed) ---
        option_b_attempted = False
        option_b_blocker: str | None = None
        option_b_success = False
        if not option_a_success:
            option_b = attempt_option_b()
            option_b_attempted = option_b["attempted"]
            option_b_blocker = option_b["blocker"]
            option_b_success = option_b["success"]

        # --- Step 6: VitisAI pre-conditions (independent) ---
        vitisai = check_vitisai_preconditions()
        ninja_found: bool = vitisai["ninja_found"]
        openblas_found: bool = vitisai["openblas_found"]

        # --- Step 7: NPU GEMM benchmark (if any install succeeded) ---
        npu_gemm_runs = False
        npu_gemm_us = 0.0
        npu_speedup_vs_cpu = 0.0
        gemm_error = ""
        if option_a_success or option_b_success:
            gemm = run_npu_gemm_benchmark()
            npu_gemm_runs = gemm["npu_gemm_runs"]
            npu_gemm_us = gemm["npu_gemm_us"]
            npu_speedup_vs_cpu = gemm["npu_speedup_vs_cpu"]
            gemm_error = gemm["error"]

        # --- Derive verdict ---
        honest_verdict = compute_honest_verdict(
            option_a_success=option_a_success,
            option_b_attempted=option_b_attempted,
            option_b_blocker=option_b_blocker,
            ninja_found=ninja_found,
            npu_gemm_runs=npu_gemm_runs,
            mlir_aie_import_ok=mlir_aie_import_ok,
        )

        # --- Write artifact ---
        run_status = "success" if (option_a_success or option_b_success) else "blocked"
        artifact = tmpl.build_result(
            {
                "v8_verdict": v8_verdict,
                "option_a_success": option_a_success,
                "option_a_wheel_url": option_a_wheel_url,
                "option_a_pip_stderr": option_a_pip_stderr,
                "mlir_aie_import_ok": mlir_aie_import_ok,
                "option_b_attempted": option_b_attempted,
                "option_b_blocker": option_b_blocker,
                "option_b_success": option_b_success,
                "ninja_found": ninja_found,
                "openblas_found": openblas_found,
                "npu_gemm_runs": npu_gemm_runs,
                "npu_gemm_us": npu_gemm_us,
                "cpu_gemm_us": CPU_GEMM_US_BASELINE,
                "npu_speedup_vs_cpu": npu_speedup_vs_cpu,
                "gemm_error": gemm_error,
                "honest_verdict": honest_verdict,
            },
            status=run_status,
        )

        out_path = os.path.join(_REPO_ROOT, DELIVERABLE)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(artifact, f, indent=2)
        _log.info("Artifact written to %s", out_path)

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()

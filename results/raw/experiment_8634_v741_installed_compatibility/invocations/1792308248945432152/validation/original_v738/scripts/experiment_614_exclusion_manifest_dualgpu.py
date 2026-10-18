#!/usr/bin/env python3
"""Experiment 614 — ExclusionManifest Precheck Timing + DualGPU Parallel Forward-Pass Validation.

**Researcher summary:**
    RETRO-067 was marked resolved in Exp 601 (precheck_created_sentinel_proven=True),
    but the .46 retro still shows conductor_consulted=False for the milestone as a whole.
    The root cause: nobody verified that the sentinel file's mtime was actually within
    60 seconds of an experiment start — the precheck could have run hours earlier and
    the sentinel would still pass a simple exists() check.

    Additionally, DualGPU parallel forward-pass has been UNCONFIRMED for five
    consecutive milestones.  Both RTX 3090s are listed by nvidia-smi and torch
    detects two CUDA devices, but no experiment ever logged gpu1_utilization > 0
    during an actual forward pass.

    This experiment validates both in a single run:
    1. TIMING: Runs conductor_manifest_precheck.py 614 (not excluded), then immediately
       reads the sentinel mtime and confirms it is < 60 seconds old.
    2. DUALGPU: If torch.cuda.device_count() >= 2, loads nn.Linear(10, 10) onto each
       GPU, runs concurrent forward passes via threading.Thread, and reads utilization.

**Exit paths (every path writes the deliverable):**
    1. apply_env_autofix()
    2. ExperimentTimeoutWatchdog(614, timeout_minutes=20)
    3. ExperimentTemplate.setup()
    4. Precheck timing test (always runs)
    5. DualGPU utilization test (graceful fallback if no CUDA)
    6. tmpl.build_result(...)
    7. tmpl.assert_deliverable_written()  -- FINAL LINE

Spec: REQ-INFRA-087, REQ-INFRA-088, SCENARIO-INFRA-095, SCENARIO-INFRA-096
"""

from __future__ import annotations

# apply_env_autofix MUST run before any JAX or CUDA import to set CARNOT_FORCE_LIVE
# and JAX_PLATFORMS correctly.  Delay everything else until after this call.
from carnot.pipeline.env_autofix import apply_env_autofix

_env_result = apply_env_autofix()

import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_template import ExperimentTemplate  # noqa: E402

_RESULT_PATH = "results/experiment_614_exclusion_manifest_dualgpu.json"
_PRECHECK_PATH = _REPO_ROOT / "scripts" / "conductor_manifest_precheck.py"
_SENTINEL_PATH = _REPO_ROOT / "scripts" / "conductor_consulted_at.txt"


# ---------------------------------------------------------------------------
# Precheck timing test
# ---------------------------------------------------------------------------


def run_precheck_and_time(exp_id: int) -> tuple[bool, float]:
    """Run conductor_manifest_precheck.py for exp_id, then measure sentinel age.

    The test calls the precheck CLI (not the importable function) because the
    CLI is what the human conductor actually uses — this is the integration path
    that writes the sentinel file.  After the subprocess returns, we read the
    sentinel's mtime and compare it to the current wall-clock time.

    Parameters
    ----------
    exp_id : int
        The experiment ID to pass to the precheck.  Must NOT be in the exclusion
        manifest or the precheck will exit 1 and NOT write a sentinel.

    Returns
    -------
    precheck_ok : bool
        True if the precheck exited 0 (experiment is not excluded).
    sentinel_age_seconds : float
        How many seconds old the sentinel file is after the precheck ran.
        Returns float('inf') if the sentinel does not exist.
    """
    proc = subprocess.run(
        [sys.executable, str(_PRECHECK_PATH), str(exp_id)],
        capture_output=True,
        text=True,
    )
    precheck_ok = proc.returncode == 0

    if not _SENTINEL_PATH.exists():
        return precheck_ok, float("inf")

    sentinel_age = time.time() - os.path.getmtime(str(_SENTINEL_PATH))
    return precheck_ok, sentinel_age


# ---------------------------------------------------------------------------
# DualGPU parallel forward-pass test
# ---------------------------------------------------------------------------


def _forward_pass_on_device(device_str: str, results_dict: dict, key: str) -> None:
    """Run a single linear forward pass on device_str and record any exception.

    This function is designed to run in a threading.Thread.  It stores the
    result (or error string) in results_dict[key] so the caller can inspect
    outcomes after join().

    Why threading instead of multiprocessing: CUDA contexts are per-process,
    so two processes would each need their own context initialisation (~5 s).
    Threads within one process share the existing CUDA context and can each
    hold their own device tensor, which is faster for this lightweight test.

    Parameters
    ----------
    device_str : str
        PyTorch device string, e.g. 'cuda:0' or 'cuda:1'.
    results_dict : dict
        Shared dict for storing the outcome.
    key : str
        Key under which to store 'ok' or the error message string.
    """
    try:
        import torch  # noqa: PLC0415
        import torch.nn as nn  # noqa: PLC0415

        device = torch.device(device_str)
        # Tiny model — just enough to put a kernel on the GPU without consuming
        # meaningful VRAM or taking more than a few milliseconds.
        model = nn.Linear(10, 10).to(device)
        # Run 200 forward passes in a tight loop so the GPU utilization counter
        # (sampled by torch.cuda.utilization at 1 Hz resolution) has a chance to
        # register above 0.
        x = torch.randn(64, 10, device=device)
        for _ in range(200):
            _ = model(x)
        results_dict[key] = "ok"
    except Exception as exc:  # noqa: BLE001
        results_dict[key] = str(exc)


def run_dualgpu_test() -> tuple[int, bool, str | None]:
    """Attempt a concurrent forward pass on two CUDA GPUs.

    Steps:
      1. Import torch (returns n_gpus=0 on ImportError).
      2. If n_gpus >= 2, launch two threads — one per GPU — each running
         _forward_pass_on_device().
      3. After both threads complete, query torch.cuda.utilization(1) to
         confirm GPU1 registered compute activity.

    Why we only gate on gpu1: GPU0 (the primary) is almost always confirmed
    through normal experiment runs.  GPU1 has been the unconfirmed device for
    five milestones.

    Returns
    -------
    n_gpus_detected : int
        Number of CUDA GPUs torch found (0 if torch unavailable).
    gpu1_utilization_confirmed : bool
        True if GPU1 utilization was > 0 after the concurrent passes.
    dualgpu_blocked_reason : str | None
        Human-readable reason if the test could not run, or None on success.
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return 0, False, "cuda_unavailable"

    if not torch.cuda.is_available():
        return 0, False, "cuda_unavailable"

    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        return n_gpus, False, "only_one_gpu"

    # Launch both forward passes concurrently.
    thread_results: dict[str, str] = {}
    t0 = threading.Thread(
        target=_forward_pass_on_device,
        args=("cuda:0", thread_results, "gpu0"),
        daemon=True,
    )
    t1 = threading.Thread(
        target=_forward_pass_on_device,
        args=("cuda:1", thread_results, "gpu1"),
        daemon=True,
    )
    t0.start()
    t1.start()
    t0.join(timeout=60)
    t1.join(timeout=60)

    gpu1_result = thread_results.get("gpu1", "timeout")
    if gpu1_result != "ok":
        # Forward pass itself failed — still report what we can.
        return n_gpus, False, f"gpu1_forward_failed: {gpu1_result}"

    # Read utilization immediately after threads finish.  torch.cuda.utilization
    # returns an integer 0-100 representing the recent GPU utilization percentage.
    # There is a small race: the sampling window may not include our burst if the
    # GPU kernel completed too quickly.  The 200-iteration loop above is designed
    # to hold the GPU busy long enough to be sampled.
    try:
        gpu1_util = torch.cuda.utilization(1)
    except Exception as exc:  # noqa: BLE001
        return n_gpus, False, f"utilization_query_failed: {exc}"

    gpu1_utilization_confirmed = gpu1_util > 0
    return n_gpus, gpu1_utilization_confirmed, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_experiment() -> dict:
    """Execute precheck timing and DualGPU validation checks.

    Both checks are independent — a failure in one does not abort the other,
    so the artifact always captures the full state of both validations.

    Returns
    -------
    dict
        Payload to pass into ExperimentTemplate.build_result().
    """
    # ------------------------------------------------------------------
    # Check 1: Precheck timing
    # ------------------------------------------------------------------
    precheck_ok, sentinel_age_seconds = run_precheck_and_time(614)

    sentinel_within_60s = sentinel_age_seconds < 60.0
    retro_067_timing_confirmed = sentinel_within_60s

    # ------------------------------------------------------------------
    # Check 2: DualGPU parallel forward-pass
    # ------------------------------------------------------------------
    n_gpus_detected, gpu1_utilization_confirmed, dualgpu_blocked_reason = run_dualgpu_test()

    # ------------------------------------------------------------------
    # Honest verdict
    # ------------------------------------------------------------------
    if sentinel_within_60s and gpu1_utilization_confirmed:
        honest_verdict = "precheck_timed_dualgpu_confirmed"
    elif sentinel_within_60s:
        honest_verdict = "precheck_timed_dualgpu_blocked"
    else:
        honest_verdict = "precheck_not_timed_dualgpu_checked"

    return {
        "precheck_ok": precheck_ok,
        "sentinel_age_seconds": sentinel_age_seconds,
        "sentinel_within_60s": sentinel_within_60s,
        "retro_067_timing_confirmed": retro_067_timing_confirmed,
        "n_gpus_detected": n_gpus_detected,
        "gpu1_utilization_confirmed": gpu1_utilization_confirmed,
        "dualgpu_blocked_reason": dualgpu_blocked_reason,
        "honest_verdict": honest_verdict,
    }


def main() -> None:
    """Entry point."""
    result_path = str(_REPO_ROOT / _RESULT_PATH)

    tmpl = ExperimentTemplate(
        614,
        "ExclusionManifest DualGPU Validation",
        _RESULT_PATH,
        requires_gpu=False,
    )
    tmpl.setup()

    with ExperimentTimeoutWatchdog(614, timeout_minutes=20, result_path=result_path):
        payload = run_experiment()

    artifact = tmpl.build_result(
        {
            **payload,
            "schema_name": "carnot.exclusion_dualgpu_validation.v1",
        },
        status="success",
    )

    out_path = _REPO_ROOT / _RESULT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(artifact, indent=2))
    os.replace(str(tmp), str(out_path))

    print(f"\nResult: {out_path}")
    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()

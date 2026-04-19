"""dual_gpu_controlled_test — controlled GPU utilization measurement for RETRO-052.

**Why this module exists (RETRO-052):**
    Exp 505 (DualGPU sweep) found n_scripts_patched=0, yet GPU 1 remained at 0%
    compute utilization.  The sweep only patches device_map='auto' assignments; if
    no scripts needed patching, the root cause lies elsewhere.  This module provides
    a controlled test that bypasses all harness-level assignment logic and directly
    loads one model per GPU, measures utilization during inference, and reports
    whether GPU 1 actually runs forward-pass compute.

    The key instrument is nvmlDeviceGetUtilizationRates() — this is the same API
    that nvidia-smi uses for its "Volatile GPU-Util%" column.  It reports compute
    engine utilization averaged over the driver's 1/6-second sampling window.  A
    value > 10% during active inference reliably indicates real GPU compute.

**CI stub path:**
    When pynvml is not installed, sample_gpu_utilization() returns {device_id: 0.0}
    for each requested device.  This lets the test suite verify the data contract
    and DualGPUTestResult logic without GPU hardware.

Spec: REQ-INFRA-070, SCENARIO-INFRA-079, SCENARIO-INFRA-080
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

_log = logging.getLogger(__name__)

__all__ = ["DualGPUTestResult", "sample_gpu_utilization", "run_dual_inference"]


@dataclass
class DualGPUTestResult:
    """Result from a controlled dual-GPU utilization test.

    Fields
    ------
    gpu0_compute_pct : float
        Mean compute utilization on GPU 0 during inference (0-100).
    gpu1_compute_pct : float
        Mean compute utilization on GPU 1 during inference (0-100).
    n_samples_run : int
        Number of utilization samples collected per GPU.
    inference_mode : str
        One of 'live_gpu' (real hardware ran), 'gpu_required' (no GPU present).
    honest_verdict : str
        'gpu1_active' if gpu1_compute_pct > 10 and inference_mode='live_gpu'.
        'gpu1_idle'   if gpu1_compute_pct <= 10 and inference_mode='live_gpu'.
        'gpu_required' if inference_mode='gpu_required'.

    Why the 10% threshold:
        nvml reports utilization averaged over a ~167 ms window.  During a short
        inference burst (< 1s per token on a 3090), the mean across 20 samples
        spaced 0.5 s apart is typically 5-25%.  10% is a conservative floor that
        distinguishes real compute from noise (driver polling overhead ≈ 0-2%).
    """

    gpu0_compute_pct: float
    gpu1_compute_pct: float
    n_samples_run: int
    inference_mode: str
    honest_verdict: str


def sample_gpu_utilization(
    device_ids: List[int],
    n_samples: int = 20,
    interval_s: float = 0.5,
) -> Dict[int, float]:
    """Poll nvmlDeviceGetUtilizationRates() and return mean compute utilization per GPU.

    This function is the measurement instrument for RETRO-052.  It samples compute
    utilization (not memory utilization — compute is what matters for forward-pass
    activity) repeatedly and returns the mean, which smooths out the driver's own
    sampling window artifacts.

    CI stub: when pynvml is not installed, returns {device_id: 0.0} for all device_ids.
    The test suite uses this path to verify that DualGPUTestResult correctly classifies
    a 0% result as 'gpu1_idle' rather than 'gpu1_active'.

    Parameters
    ----------
    device_ids : list of int
        GPU indices to sample (e.g. [0, 1]).
    n_samples : int
        How many times to read the utilization counter per device.  More samples
        produce a more reliable mean at the cost of wall-clock time.
    interval_s : float
        Seconds to sleep between samples.  0.5 s gives 10-second total coverage
        for n_samples=20, which is enough to catch a 10-pass inference burst.

    Returns
    -------
    dict[int, float]
        Mean compute utilization percentage per device_id, averaged over n_samples.

    Spec: REQ-INFRA-070, SCENARIO-INFRA-079
    """
    try:
        import pynvml  # noqa: PLC0415 — optional dep

        pynvml.nvmlInit()
        handles = {dev: pynvml.nvmlDeviceGetHandleByIndex(dev) for dev in device_ids}
    except ImportError:
        # CI stub: no GPU hardware — return zero utilization for all devices.
        _log.debug(
            "sample_gpu_utilization: pynvml not installed — returning CI stub 0.0 for %s",
            device_ids,
        )
        return {dev: 0.0 for dev in device_ids}
    except Exception as exc:
        _log.warning("sample_gpu_utilization: pynvml init error: %s — returning 0.0", exc)
        return {dev: 0.0 for dev in device_ids}

    # Accumulate utilization readings per device.
    readings: Dict[int, List[float]] = {dev: [] for dev in device_ids}

    for _ in range(n_samples):
        for dev, handle in handles.items():
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                # util.gpu is the compute engine utilization (0-100).
                readings[dev].append(float(util.gpu))
            except Exception as exc:
                _log.warning(
                    "sample_gpu_utilization: device %d query error: %s — skipping sample",
                    dev,
                    exc,
                )
        if interval_s > 0:
            time.sleep(interval_s)

    # Compute means; devices with no successful readings fall back to 0.0.
    return {
        dev: (sum(vals) / len(vals) if vals else 0.0)
        for dev, vals in readings.items()
    }


def run_dual_inference(
    model_a: Callable[[str], str],
    model_b: Callable[[str], str],
    prompts: List[str],
) -> Tuple[List[str], List[str]]:
    """Run two models simultaneously in separate threads and collect their outputs.

    This is the load generator for the RETRO-052 controlled test.  Both models run
    their full set of prompts concurrently so that GPU 0 and GPU 1 are both active
    at the same time.  The utilization sampler (sample_gpu_utilization) runs in a
    third thread while these two are executing.

    The two lists of responses are returned in the same order as `prompts`.  Errors
    from individual prompts are caught and replaced with an empty string so a single
    failed inference does not abort the entire measurement run.

    Parameters
    ----------
    model_a : callable
        Inference function for the model on GPU 0.  Signature: (prompt: str) -> str.
    model_b : callable
        Inference function for the model on GPU 1.  Signature: (prompt: str) -> str.
    prompts : list of str
        Prompts to run through each model.

    Returns
    -------
    (responses_a, responses_b) : tuple of list[str]
        Parallel response lists, one per model, in prompt order.

    Spec: REQ-INFRA-070
    """

    def _run_model(fn: Callable[[str], str]) -> List[str]:
        results: List[str] = []
        for prompt in prompts:
            try:
                results.append(fn(prompt))
            except Exception as exc:
                _log.warning("run_dual_inference: inference error: %s", exc)
                results.append("")
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fut_a = pool.submit(_run_model, model_a)
        fut_b = pool.submit(_run_model, model_b)
        responses_a = fut_a.result()
        responses_b = fut_b.result()

    return responses_a, responses_b

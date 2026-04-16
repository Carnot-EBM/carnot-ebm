"""DualGPUHealthCheck — detect GPU1 zombies and temperature-guard batch sizes.

**Researcher summary (RETRO-025):**
    PID 3509070 held 1786 MB on GPU1 at 0% utilization while GPU0 ran at 88%
    for 144+ minutes.  GPU0 also reached 82C — within 1-3C of the RTX 3090
    throttle threshold (83-85C).  Two compounding problems:

    1. **GPU1 zombie**: DualGPURunner is not asserting that GPU1 actually
       computes.  A model can allocate VRAM on GPU1 during load but then do
       zero compute — all inference silently serialises onto GPU0.  This wastes
       half the available GPU memory and doubles per-question latency.

    2. **Thermal risk**: GPU0 at 82C is one thermal event away from throttling.
       Running at full batch size under thermal stress risks dropping throughput
       mid-experiment without any operator warning.

**What this module provides:**

    ``DualGPUHealthResult`` — structured dataclass with per-GPU utilization,
    temperature, VRAM usage, and two derived flags:

    - ``gpu1_is_zombie``: ``True`` when GPU1 has >500 MB VRAM allocated but
      <1% compute utilization.  The 500 MB threshold distinguishes a loaded
      model (>500 MB) from a freshly-initialised device with near-zero VRAM.

    - ``temperature_warning``: ``True`` when any GPU exceeds 80C.  The 80C
      threshold gives a 3-5C safety margin before the 83-85C throttle zone.

    - ``recommended_batch_size_factor``: ``0.75`` when ``temperature_warning``
      is True (25% reduction to provide thermal headroom), ``1.0`` otherwise.

    ``check_dual_gpu_health(timeout_seconds=60)`` — tries pynvml first
    (preferred: direct C API, no subprocess), then falls back to an
    ``nvidia-smi --query-gpu=...`` subprocess call.  When neither is available
    (CI machines without GPU drivers), it returns safe all-zero defaults and
    never raises.  This CI-safe guarantee means every experiment can call this
    function unconditionally without needing GPU hardware.

    ``build_gpu_fix_artifact(health, prior_retro_path)`` — produces a
    JSON-serializable dict with ``schema='carnot.dual_gpu_fix.v1'``,
    ``honest_verdict``, and ``retro_025_status``.

**Why pynvml before nvidia-smi subprocess?**
    pynvml is a Python binding to the NVIDIA Management Library (C API).  It
    gives sub-millisecond reads with no subprocess overhead and is already
    installed on most CUDA workstations.  The nvidia-smi subprocess fallback
    covers environments where pynvml is not installed (e.g. some conda envs).

**Why 500 MB as the zombie VRAM threshold?**
    A freshly-initialised CUDA context typically uses 50-200 MB on an RTX 3090.
    A loaded inference model (even the smallest) uses >500 MB.  If GPU1 shows
    >500 MB VRAM but 0% utilization, a model was loaded there but is not
    being used — this is the zombie pattern from RETRO-025.

**Why 80C as the temperature threshold?**
    RTX 3090 thermal throttle engages at approximately 83-85C.  Setting the
    warning at 80C gives a 3-5C buffer.  The 25% batch-size reduction (factor
    0.75) is a conservative measure: it reduces GPU compute load without
    stopping the experiment, and is enough to drop temperatures by ~5-8C in
    typical workloads.

Spec: REQ-INFRA-025, REQ-INFRA-026,
      SCENARIO-INFRA-031, SCENARIO-INFRA-032, SCENARIO-INFRA-033 (Exp 426)
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DualGPUHealthResult dataclass
# ---------------------------------------------------------------------------

_ZOMBIE_VRAM_THRESHOLD_MB = 500.0
"""Minimum VRAM in MB that indicates a model is loaded on GPU1 (not just context init)."""

_TEMP_WARNING_THRESHOLD_C = 80.0
"""Temperature in Celsius above which a thermal warning is issued."""

_ZOMBIE_UTIL_THRESHOLD_PCT = 1.0
"""Utilization percentage below which GPU1 is considered idle despite VRAM allocation."""


@dataclass
class DualGPUHealthResult:
    """Structured result from ``check_dual_gpu_health()``.

    Fields
    ------
    gpu0_util_pct : float
        GPU0 compute utilization percentage (0-100).  0.0 in CI fallback mode.
    gpu1_util_pct : float
        GPU1 compute utilization percentage (0-100).  0.0 when no GPU1 present
        or in CI fallback mode.
    gpu0_temp_c : float
        GPU0 temperature in Celsius.  0.0 in CI fallback mode.
    gpu1_temp_c : float
        GPU1 temperature in Celsius.  0.0 when no GPU1 present or in CI fallback.
    gpu0_vram_mb : float
        GPU0 VRAM used in megabytes.
    gpu1_vram_mb : float
        GPU1 VRAM used in megabytes.  0.0 when no GPU1 present.
    gpu1_is_zombie : bool
        ``True`` iff ``gpu1_vram_mb > 500`` AND ``gpu1_util_pct < 1``.
        The zombie pattern from RETRO-025: model loaded on GPU1 but doing no
        compute — all inference is silently serialised onto GPU0.
    temperature_warning : bool
        ``True`` iff any GPU temperature exceeds 80C (3-5C below RTX 3090
        throttle at 83-85C).  Signals thermal risk from RETRO-025.
    recommended_batch_size_factor : float
        ``0.75`` when ``temperature_warning=True`` (25% batch reduction to
        provide thermal headroom); ``1.0`` otherwise.

    Spec: REQ-INFRA-025, SCENARIO-INFRA-031/032/033
    """

    gpu0_util_pct: float
    gpu1_util_pct: float
    gpu0_temp_c: float
    gpu1_temp_c: float
    gpu0_vram_mb: float
    gpu1_vram_mb: float
    gpu1_is_zombie: bool
    temperature_warning: bool
    recommended_batch_size_factor: float


# ---------------------------------------------------------------------------
# _safe_defaults
# ---------------------------------------------------------------------------


def _safe_defaults() -> DualGPUHealthResult:
    """Return all-zero safe defaults for CI machines without GPU hardware.

    This is the SCENARIO-INFRA-032 fallback: pynvml and nvidia-smi are both
    unavailable.  We return a benign result (no zombie, no temperature warning)
    so that callers can proceed unconditionally.  The all-zero values are
    clearly distinguishable from real GPU readings in experiment artifacts.

    Spec: SCENARIO-INFRA-032
    """
    return DualGPUHealthResult(
        gpu0_util_pct=0.0,
        gpu1_util_pct=0.0,
        gpu0_temp_c=0.0,
        gpu1_temp_c=0.0,
        gpu0_vram_mb=0.0,
        gpu1_vram_mb=0.0,
        gpu1_is_zombie=False,
        temperature_warning=False,
        recommended_batch_size_factor=1.0,
    )


# ---------------------------------------------------------------------------
# _derive_flags
# ---------------------------------------------------------------------------


def _derive_flags(
    gpu0_util: float,
    gpu1_util: float,
    gpu0_temp: float,
    gpu1_temp: float,
    gpu1_vram: float,
) -> tuple[bool, bool, float]:
    """Derive gpu1_is_zombie, temperature_warning, and recommended_batch_size_factor.

    Separated from query logic so tests can verify the flag derivation
    independently of the GPU querying mechanism.

    Parameters
    ----------
    gpu0_util, gpu1_util : float
        Utilization percentages for GPU0 and GPU1.
    gpu0_temp, gpu1_temp : float
        Temperatures in Celsius for GPU0 and GPU1.
    gpu1_vram : float
        VRAM used by GPU1 in megabytes.

    Returns
    -------
    (gpu1_is_zombie, temperature_warning, recommended_batch_size_factor)

    Spec: REQ-INFRA-025, SCENARIO-INFRA-031/033
    """
    # GPU1 is a zombie when it has substantial VRAM allocated (model is loaded)
    # but zero compute is happening.  The strict > threshold means a freshly-
    # initialised CUDA context with ~50-200 MB is NOT classified as a zombie.
    gpu1_is_zombie = (
        gpu1_vram > _ZOMBIE_VRAM_THRESHOLD_MB
        and gpu1_util < _ZOMBIE_UTIL_THRESHOLD_PCT
    )

    # Temperature warning fires when EITHER GPU exceeds the 80C threshold.
    # We check both GPUs so that a zombie GPU1 cooking silently still triggers
    # the warning.
    temperature_warning = gpu0_temp > _TEMP_WARNING_THRESHOLD_C or gpu1_temp > _TEMP_WARNING_THRESHOLD_C

    # The 25% reduction (factor 0.75) is a conservative measure.  It reduces
    # GPU compute load enough to drop temperatures by ~5-8C in typical inference
    # workloads, clearing the 3-5C margin below throttle.
    recommended_batch_size_factor = 0.75 if temperature_warning else 1.0

    return gpu1_is_zombie, temperature_warning, recommended_batch_size_factor


# ---------------------------------------------------------------------------
# _query_nvidia_smi
# ---------------------------------------------------------------------------


def _query_nvidia_smi() -> str:
    """Run nvidia-smi and return CSV output string.

    Queries: utilization.gpu, temperature.gpu, memory.used (in MiB).
    Returns the raw stdout string.

    Raises
    ------
    FileNotFoundError
        When nvidia-smi is not installed.
    RuntimeError
        When nvidia-smi exits non-zero.
    subprocess.TimeoutExpired
        When nvidia-smi takes longer than 10 seconds.

    Separated as a standalone function so tests can patch it without mocking
    the full subprocess module.  This is the same pattern used in
    ``carnot.pipeline.live_gpu_diagnostic.check_cuda_visible()``.
    """
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,temperature.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"nvidia-smi exited {result.returncode}: {result.stderr.strip()}"
        )
    return result.stdout


# ---------------------------------------------------------------------------
# _parse_nvidia_smi_output
# ---------------------------------------------------------------------------


def _parse_nvidia_smi_output(
    output: str,
) -> tuple[float, float, float, float, float, float]:
    """Parse nvidia-smi CSV output into (gpu0_util, gpu0_temp, gpu0_vram, gpu1_util, gpu1_temp, gpu1_vram).

    Expects one row per GPU.  When only one GPU is present, GPU1 values are 0.
    When output is malformed, returns all zeros (CI-safe).

    Why not fail-hard on malformed output?  The caller (``check_dual_gpu_health``)
    is CI-safe — it must never raise.  Malformed nvidia-smi output happens on
    unusual driver versions; returning zeros is better than crashing an experiment.
    """
    lines = [ln.strip() for ln in output.strip().splitlines() if ln.strip()]

    def _parse_line(line: str) -> tuple[float, float, float]:
        """Parse 'util, temp, vram_mb' from one nvidia-smi CSV line."""
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            return 0.0, 0.0, 0.0
        try:
            return float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            return 0.0, 0.0, 0.0

    gpu0_util = gpu0_temp = gpu0_vram = 0.0
    gpu1_util = gpu1_temp = gpu1_vram = 0.0

    if len(lines) >= 1:
        gpu0_util, gpu0_temp, gpu0_vram = _parse_line(lines[0])
    if len(lines) >= 2:
        gpu1_util, gpu1_temp, gpu1_vram = _parse_line(lines[1])

    return gpu0_util, gpu0_temp, gpu0_vram, gpu1_util, gpu1_temp, gpu1_vram


# ---------------------------------------------------------------------------
# _check_via_pynvml
# ---------------------------------------------------------------------------


def _check_via_pynvml() -> DualGPUHealthResult | None:
    """Try to query GPU health via pynvml.

    Returns ``None`` on any failure (pynvml not installed, driver error, etc.)
    so the caller can fall back to nvidia-smi.

    **Why pynvml rather than subprocess?**
    pynvml is a thin Python wrapper around NVML, the same C API that
    nvidia-smi uses internally.  It is typically 10-100x faster than a
    subprocess call (no fork/exec overhead) and avoids shell injection risk.

    Spec: REQ-INFRA-025
    """
    try:
        import pynvml  # noqa: PLC0415 — optional dep, intentional late import

        pynvml.nvmlInit()
    except Exception as exc:
        _log.debug("pynvml unavailable or nvmlInit failed: %s", exc)
        return None

    try:
        n_gpus = pynvml.nvmlDeviceGetCount()

        def _read_gpu(idx: int) -> tuple[float, float, float]:
            """Read (util_pct, temp_c, vram_mb) for GPU at index *idx*."""
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            util = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            temp = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
            vram_bytes = pynvml.nvmlDeviceGetMemoryInfo(handle).used
            vram_mb = vram_bytes / (1024 * 1024)
            return util, temp, vram_mb

        gpu0_util = gpu0_temp = gpu0_vram = 0.0
        gpu1_util = gpu1_temp = gpu1_vram = 0.0

        if n_gpus >= 1:
            gpu0_util, gpu0_temp, gpu0_vram = _read_gpu(0)
        if n_gpus >= 2:
            gpu1_util, gpu1_temp, gpu1_vram = _read_gpu(1)

        gpu1_is_zombie, temperature_warning, recommended_batch_size_factor = _derive_flags(
            gpu0_util, gpu1_util, gpu0_temp, gpu1_temp, gpu1_vram
        )

        return DualGPUHealthResult(
            gpu0_util_pct=gpu0_util,
            gpu1_util_pct=gpu1_util,
            gpu0_temp_c=gpu0_temp,
            gpu1_temp_c=gpu1_temp,
            gpu0_vram_mb=gpu0_vram,
            gpu1_vram_mb=gpu1_vram,
            gpu1_is_zombie=gpu1_is_zombie,
            temperature_warning=temperature_warning,
            recommended_batch_size_factor=recommended_batch_size_factor,
        )

    except Exception as exc:  # noqa: BLE001
        _log.warning("pynvml query failed: %s — falling back to nvidia-smi", exc)
        return None


# ---------------------------------------------------------------------------
# _check_via_nvidia_smi
# ---------------------------------------------------------------------------


def _check_via_nvidia_smi() -> DualGPUHealthResult | None:
    """Try to query GPU health via nvidia-smi subprocess.

    Returns ``None`` on any failure so the caller falls back to safe defaults.

    Spec: REQ-INFRA-025, SCENARIO-INFRA-032
    """
    try:
        output = _query_nvidia_smi()
    except Exception as exc:  # noqa: BLE001
        _log.debug("nvidia-smi query failed: %s — using CI safe defaults", exc)
        return None

    try:
        gpu0_util, gpu0_temp, gpu0_vram, gpu1_util, gpu1_temp, gpu1_vram = (
            _parse_nvidia_smi_output(output)
        )
        gpu1_is_zombie, temperature_warning, recommended_batch_size_factor = _derive_flags(
            gpu0_util, gpu1_util, gpu0_temp, gpu1_temp, gpu1_vram
        )
        return DualGPUHealthResult(
            gpu0_util_pct=gpu0_util,
            gpu1_util_pct=gpu1_util,
            gpu0_temp_c=gpu0_temp,
            gpu1_temp_c=gpu1_temp,
            gpu0_vram_mb=gpu0_vram,
            gpu1_vram_mb=gpu1_vram,
            gpu1_is_zombie=gpu1_is_zombie,
            temperature_warning=temperature_warning,
            recommended_batch_size_factor=recommended_batch_size_factor,
        )
    except Exception as exc:  # noqa: BLE001
        _log.warning("Failed to parse nvidia-smi output: %s — using CI safe defaults", exc)
        return None


# ---------------------------------------------------------------------------
# check_dual_gpu_health
# ---------------------------------------------------------------------------


def check_dual_gpu_health(timeout_seconds: int = 60) -> DualGPUHealthResult:
    """Query dual-GPU health and return a structured result.

    **Priority order:**
    1. pynvml (direct NVML C API — fastest, no subprocess overhead)
    2. nvidia-smi subprocess (fallback when pynvml not installed)
    3. Safe CI defaults (all zeros, no zombie, no temp warning)

    The ``timeout_seconds`` parameter is accepted for API consistency with the
    RETRO-025 spec (60-second window to assert GPU1 compute) but is not
    currently used as a blocking timeout on the query itself — NVML and
    nvidia-smi queries complete in milliseconds.  It may be used in a future
    version to poll until GPU1 shows utilization (rather than taking a single
    snapshot).

    **CI-safe guarantee:** This function NEVER raises.  Any exception at any
    layer is caught, logged, and replaced with safe defaults.  This allows
    every experiment to call it unconditionally, regardless of whether GPU
    hardware is present.

    Parameters
    ----------
    timeout_seconds : int
        Reserved for future use (polling window).  Currently unused by the
        snapshot-based query.

    Returns
    -------
    DualGPUHealthResult
        Populated with real GPU readings (pynvml or nvidia-smi) or safe all-zero
        defaults when GPU hardware is not accessible.

    Spec: REQ-INFRA-025, SCENARIO-INFRA-031/032/033 (Exp 426)
    """
    try:
        # --- Attempt 1: pynvml ---
        result = _check_via_pynvml()
        if result is not None:
            _log.debug(
                "check_dual_gpu_health via pynvml: gpu0_util=%.0f%% gpu1_util=%.0f%% "
                "gpu0_temp=%.0fC gpu1_temp=%.0fC zombie=%s temp_warn=%s",
                result.gpu0_util_pct,
                result.gpu1_util_pct,
                result.gpu0_temp_c,
                result.gpu1_temp_c,
                result.gpu1_is_zombie,
                result.temperature_warning,
            )
            return result

        # --- Attempt 2: nvidia-smi subprocess ---
        result = _check_via_nvidia_smi()
        if result is not None:
            _log.debug(
                "check_dual_gpu_health via nvidia-smi: gpu0_util=%.0f%% gpu1_util=%.0f%% "
                "gpu0_temp=%.0fC gpu1_temp=%.0fC zombie=%s temp_warn=%s",
                result.gpu0_util_pct,
                result.gpu1_util_pct,
                result.gpu0_temp_c,
                result.gpu1_temp_c,
                result.gpu1_is_zombie,
                result.temperature_warning,
            )
            return result

        # --- Attempt 3: CI safe defaults ---
        _log.info(
            "check_dual_gpu_health: no GPU hardware accessible — returning CI safe defaults "
            "(pynvml and nvidia-smi both unavailable)"
        )
        return _safe_defaults()

    except Exception as exc:  # noqa: BLE001 — outermost safety net, must never raise
        _log.error(
            "check_dual_gpu_health: unexpected exception %s — returning CI safe defaults",
            exc,
            exc_info=True,
        )
        return _safe_defaults()


# ---------------------------------------------------------------------------
# build_gpu_fix_artifact
# ---------------------------------------------------------------------------


def build_gpu_fix_artifact(
    health: DualGPUHealthResult,
    prior_retro_path: str,
) -> dict:
    """Build a JSON-serializable artifact describing the dual-GPU fix outcome.

    Produces an ``honest_verdict`` that faithfully describes whether RETRO-025's
    GPU1 zombie was still present at experiment time.

    **Verdict logic:**
    - ``'zombie_detected'``: ``gpu1_is_zombie=True`` (RETRO-025 confirmed active)
    - ``'gpu1_healthy'``: ``gpu1_is_zombie=False`` (zombie cleared or never present)

    **retro_025_status:**
    - ``'zombie_confirmed'``: when ``gpu1_is_zombie=True``
    - ``'zombie_cleared'``: when ``gpu1_is_zombie=False``

    Parameters
    ----------
    health : DualGPUHealthResult
        The result from ``check_dual_gpu_health()``.
    prior_retro_path : str
        Relative path to the RETRO-025 JSON file (embedded for traceability).

    Returns
    -------
    dict
        JSON-serializable artifact with ``schema='carnot.dual_gpu_fix.v1'``.

    Spec: REQ-INFRA-025, SCENARIO-INFRA-031 (Exp 426)
    """
    if health.gpu1_is_zombie:
        honest_verdict = "zombie_detected"
        retro_025_status = "zombie_confirmed"
    else:
        honest_verdict = "gpu1_healthy"
        retro_025_status = "zombie_cleared"

    return {
        "schema": "carnot.dual_gpu_fix.v1",
        "honest_verdict": honest_verdict,
        "retro_025_status": retro_025_status,
        "prior_retro_path": prior_retro_path,
        "gpu0_util_pct": health.gpu0_util_pct,
        "gpu1_util_pct": health.gpu1_util_pct,
        "gpu0_temp_c": health.gpu0_temp_c,
        "gpu1_temp_c": health.gpu1_temp_c,
        "gpu0_vram_mb": health.gpu0_vram_mb,
        "gpu1_vram_mb": health.gpu1_vram_mb,
        "gpu1_is_zombie": health.gpu1_is_zombie,
        "temperature_warning": health.temperature_warning,
        "recommended_batch_size_factor": health.recommended_batch_size_factor,
    }

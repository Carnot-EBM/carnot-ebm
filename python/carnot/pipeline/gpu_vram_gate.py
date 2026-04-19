"""GPUVRAMGate — per-experiment VRAM guard that kills zombie processes before model load.

**Why this module exists (RETRO-037, RETRO-042, milestone .35):**
    The session-start zombie kill (Exp 463 / ConductorSessionHealthCheck) fires once at
    conductor startup but CANNOT prevent mid-session zombie accumulation.  When a GPU-
    required experiment crashes without releasing its VRAM allocation, subsequent experiments
    in the SAME milestone session see a fully saturated GPU and defer with
    honest_verdict='deferred_to_gpu'.  In milestone .35, four of twelve experiments
    (Exps ?, ?, ?, ?) all hit this same root cause: zombie processes held 23.8 GB of GPU 0
    VRAM at 0% utilisation, preventing any new model load.

    A session-start check cannot prevent mid-session accumulation because zombie processes
    are created by the experiments themselves as they run and crash.  The fix must run
    BEFORE EVERY GPU-REQUIRED EXPERIMENT, not just once at session start.

**What this module provides:**
    VRAMStatus — lightweight snapshot of a single GPU's VRAM and utilisation.
    GPUVRAMInsufficientError — raised when VRAM cannot be freed within the timeout.
    GPUVRAMGate — context manager that inspects VRAM, kills zombies if needed, and
                  waits up to ``wait_seconds`` for VRAM to free before proceeding.

**Why 8 GB minimum (REQ-INFRA-039):**
    Qwen3.5-0.8B requires ~2-3 GB for weights plus inference overhead; Gemma4-E4B-it
    requires ~4-6 GB.  Adding batched inference overhead and activation buffers, 8 GB is
    the minimum that guarantees a clean model load without OOM.  Tighter thresholds cause
    intermittent OOM errors mid-experiment that are harder to debug than a clean gate fail.

**Why 60-second wait (REQ-INFRA-040):**
    After sending SIGKILL to a zombie process, the GPU driver (nvidia-smi / NVML) may
    take 10-30 seconds to reclaim and report freed VRAM.  The 60-second window gives the
    driver two full reclaim cycles before declaring failure.  Waiting longer than 60 s
    would exceed the experiment's expected startup budget.

**CPU-only / no-GPU behaviour:**
    When no GPU hardware is present (CI machines, pure CPU experiments), GPUVRAMGate is a
    complete no-op.  It checks the GPU count via pynvml; if pynvml is absent or reports
    zero devices, the gate raises no error and the experiment proceeds normally.

**Integration with ExperimentTemplate (REQ-INFRA-041):**
    ExperimentTemplate.setup_gpu() wraps every model load with this gate when
    requires_gpu=True.  This ensures the check fires inside the experiment process,
    not just at conductor session startup.

Spec: REQ-INFRA-039, REQ-INFRA-040, REQ-INFRA-041,
      SCENARIO-INFRA-047, SCENARIO-INFRA-048, SCENARIO-INFRA-049
"""

from __future__ import annotations

import logging
import os
import signal
import time
from dataclasses import dataclass
from typing import List

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# VRAMStatus
# ---------------------------------------------------------------------------


@dataclass
class VRAMStatus:
    """Lightweight VRAM snapshot for a single GPU.

    Fields
    ------
    gpu_index : int
        Zero-based GPU index as reported by NVML.
    total_mb : int
        Total VRAM on this GPU in megabytes.
    used_mb : int
        VRAM currently in use in megabytes.
    free_mb : int
        VRAM currently available in megabytes.
    utilization_pct : int
        GPU compute utilisation percentage (0-100).  0 means the GPU is idle
        even if VRAM is consumed — the zombie signature.

    Spec: REQ-INFRA-039, SCENARIO-INFRA-047
    """

    gpu_index: int
    total_mb: int
    used_mb: int
    free_mb: int
    utilization_pct: int = 0

    @property
    def free_gb(self) -> float:
        """Free VRAM in gigabytes (convenience property for threshold comparisons)."""
        return self.free_mb / 1024.0

    @property
    def is_zombie_saturated(self) -> bool:
        """True when this GPU is likely zombie-saturated.

        Zombie saturation = memory held by dead processes that show 0% utilisation.
        We define saturation as >90% VRAM used at 0% compute utilisation.

        Why 90%: leaving 10% headroom avoids false positives during normal
        model warm-up where VRAM briefly spikes before the driver settles.
        Why 0% utilisation: if any compute is running the process is still alive
        and we must not kill it — only idle-but-holding processes are zombies.
        """
        if self.total_mb == 0:
            return False
        return (self.used_mb > 0.90 * self.total_mb) and (self.utilization_pct == 0)


# ---------------------------------------------------------------------------
# GPUVRAMInsufficientError
# ---------------------------------------------------------------------------


class GPUVRAMInsufficientError(RuntimeError):
    """Raised when GPU VRAM cannot be freed to meet the minimum threshold.

    Attributes
    ----------
    gpu_index : int
        The GPU index that failed the VRAM check.
    free_gb : float
        The free VRAM (in GB) at the time of failure, after zombie kill and wait.
    min_free_gb : float
        The minimum threshold that was required.

    Spec: REQ-INFRA-040, SCENARIO-INFRA-049
    """

    def __init__(self, gpu_index: int, free_gb: float, min_free_gb: float) -> None:
        self.gpu_index = gpu_index
        self.free_gb = free_gb
        self.min_free_gb = min_free_gb
        super().__init__(
            f"GPU {gpu_index}: free VRAM {free_gb:.2f} GB < required {min_free_gb:.2f} GB "
            f"after zombie kill and {0}-second wait. "
            f"honest_verdict='gpu_vram_insufficient'"
        )


# ---------------------------------------------------------------------------
# GPUVRAMGate
# ---------------------------------------------------------------------------


class GPUVRAMGate:
    """Context manager that verifies sufficient VRAM is free before a GPU experiment.

    This gate runs BEFORE EVERY GPU-required experiment, not just at session start.
    That distinction is the fix for RETRO-037 and RETRO-042: zombie accumulation is
    a mid-session phenomenon that session-start checks cannot prevent.

    Usage
    -----
    ::

        with GPUVRAMGate(min_free_gb=8.0, auto_kill=True):
            load_model(...)   # guaranteed >= 8 GB free on each GPU

    On CPU-only machines or when no GPU is detected, the gate is a no-op.

    Parameters
    ----------
    min_free_gb : float
        Minimum free VRAM required on each GPU before the experiment begins.
        Default 8.0 GB covers Qwen3.5-0.8B and Gemma4-E4B-it with overhead.
    wait_seconds : int
        Seconds to wait (polling every 5 s) for VRAM to free after zombie kill.
        Default 60 s gives the GPU driver two full reclaim cycles.
    auto_kill : bool
        If True, automatically send SIGKILL to zombie processes when VRAM is low.
        Set False in tests or when manual intervention is preferred.

    Spec: REQ-INFRA-039, REQ-INFRA-040, REQ-INFRA-041,
          SCENARIO-INFRA-047, SCENARIO-INFRA-048, SCENARIO-INFRA-049
    """

    def __init__(
        self,
        min_free_gb: float = 8.0,
        wait_seconds: int = 60,
        auto_kill: bool = True,
    ) -> None:
        self.min_free_gb = min_free_gb
        self.wait_seconds = wait_seconds
        self.auto_kill = auto_kill

    # ------------------------------------------------------------------
    # check_vram
    # ------------------------------------------------------------------

    def check_vram(self, gpu_index: int) -> VRAMStatus:
        """Read current VRAM stats for ``gpu_index`` via pynvml.

        Returns a VRAMStatus with all fields populated from NVML queries.
        If pynvml is unavailable, returns a synthetic 'healthy' status so the
        gate is a no-op on machines without NVML (CI, CPU-only environments).

        Parameters
        ----------
        gpu_index : int
            Zero-based GPU device index.

        Returns
        -------
        VRAMStatus
            Current VRAM snapshot for this GPU.
        """
        try:
            import pynvml  # noqa: PLC0415

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return VRAMStatus(
                gpu_index=gpu_index,
                total_mb=mem.total // (1024 * 1024),
                used_mb=mem.used // (1024 * 1024),
                free_mb=mem.free // (1024 * 1024),
                utilization_pct=util.gpu,
            )
        except Exception as exc:  # pynvml absent, index invalid, etc.
            _log.debug("check_vram(%d) unavailable: %s — returning synthetic healthy", gpu_index, exc)
            # Return a synthetic 'unlimited free' status so the gate is a no-op.
            return VRAMStatus(
                gpu_index=gpu_index,
                total_mb=0,
                used_mb=0,
                free_mb=0,
                utilization_pct=0,
            )

    # ------------------------------------------------------------------
    # kill_zombies
    # ------------------------------------------------------------------

    def kill_zombies(self, gpu_index: int) -> int:
        """Kill GPU processes that hold VRAM at 0% utilisation (zombies).

        Enumerates all NVML-tracked processes on ``gpu_index``, then for each
        PID that appears dead (resident VRAM > 0, process not in psutil's process
        list OR process has 0 CPU time and is old enough), sends SIGKILL.

        Why SIGKILL and not SIGTERM: zombie processes by definition do not respond
        to SIGTERM (they are either truly dead or permanently stalled).  SIGKILL
        bypasses the signal handler and forces the OS to reclaim the process.

        Parameters
        ----------
        gpu_index : int
            Zero-based GPU device index.

        Returns
        -------
        int
            Number of processes killed.  0 when pynvml is unavailable or no
            zombies were found.
        """
        killed = 0
        try:
            import pynvml  # noqa: PLC0415

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        except Exception as exc:
            _log.debug("kill_zombies(%d): pynvml unavailable (%s)", gpu_index, exc)
            return 0

        for proc in procs:
            pid = proc.pid
            vram_mb = getattr(proc, "usedGpuMemory", 0) // (1024 * 1024) if hasattr(proc, "usedGpuMemory") else 0
            if vram_mb < 100:
                # Too small to matter — skip (avoids killing system processes with tiny VRAM)
                continue
            try:
                # Check if the process is still alive and responsive via psutil
                import psutil  # noqa: PLC0415

                try:
                    p = psutil.Process(pid)
                    cpu_times = p.cpu_times()
                    age_s = time.time() - p.create_time()
                    # A process is a zombie candidate if it has been around for >60 s
                    # with no CPU activity at all (both user and system times are 0).
                    if age_s > 60 and (cpu_times.user + cpu_times.system) < 0.1:
                        _log.warning(
                            "kill_zombies: killing GPU %d PID %d (VRAM %d MB, age %.0fs, CPU %.2fs)",
                            gpu_index, pid, vram_mb, age_s, cpu_times.user + cpu_times.system,
                        )
                        os.kill(pid, signal.SIGKILL)
                        killed += 1
                except psutil.NoSuchProcess:
                    # Process already dead but NVML still sees it — try to clean up
                    try:
                        os.kill(pid, signal.SIGKILL)
                        killed += 1
                    except ProcessLookupError:
                        pass  # already gone
            except ImportError:
                # psutil not available — use a simpler heuristic: kill if the PID
                # does not have a corresponding /proc entry (Linux only)
                try:
                    if not os.path.exists(f"/proc/{pid}"):
                        os.kill(pid, signal.SIGKILL)
                        killed += 1
                except (ProcessLookupError, PermissionError):
                    pass

        if killed:
            _log.warning(
                "kill_zombies: killed %d zombie process(es) on GPU %d; "
                "waiting for VRAM reclaim…",
                killed, gpu_index,
            )
        return killed

    # ------------------------------------------------------------------
    # wait_for_vram
    # ------------------------------------------------------------------

    def wait_for_vram(self, gpu_index: int) -> bool:
        """Poll until free VRAM meets the threshold or the wait window expires.

        Polls every 5 seconds.  Returns True as soon as the threshold is met,
        False if the window expires without sufficient VRAM becoming available.

        Why 5-second poll interval: the GPU driver typically reports VRAM changes
        within 1-3 seconds of process death, so 5 s is a safe and low-overhead
        cadence.

        Parameters
        ----------
        gpu_index : int
            Zero-based GPU device index.

        Returns
        -------
        bool
            True if free_gb >= min_free_gb within wait_seconds.
            False if the wait window expired.
        """
        deadline = time.monotonic() + self.wait_seconds
        poll_interval = 5.0

        while True:
            status = self.check_vram(gpu_index)
            # Special case: pynvml unavailable → total_mb==0 means no GPU, gate is no-op
            if status.total_mb == 0:
                return True
            if status.free_gb >= self.min_free_gb:
                _log.info(
                    "wait_for_vram: GPU %d has %.2f GB free (threshold %.2f GB) — proceeding",
                    gpu_index, status.free_gb, self.min_free_gb,
                )
                return True

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _log.warning(
                    "wait_for_vram: GPU %d still only %.2f GB free after %ds wait "
                    "(threshold %.2f GB) — deferring experiment",
                    gpu_index, status.free_gb, self.wait_seconds, self.min_free_gb,
                )
                return False

            sleep_time = min(poll_interval, remaining)
            _log.debug(
                "wait_for_vram: GPU %d free %.2f GB < %.2f GB, retrying in %.0fs",
                gpu_index, status.free_gb, self.min_free_gb, sleep_time,
            )
            time.sleep(sleep_time)

    # ------------------------------------------------------------------
    # _n_gpus
    # ------------------------------------------------------------------

    def _n_gpus(self) -> int:
        """Return the number of NVML-detected GPUs, or 0 if pynvml is absent."""
        try:
            import pynvml  # noqa: PLC0415

            pynvml.nvmlInit()
            return pynvml.nvmlDeviceGetCount()
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "GPUVRAMGate":
        """Run the VRAM gate for every detected GPU.

        Algorithm for each GPU index:
        1. check_vram() — read current state.
        2. If total_mb == 0 (no GPU / pynvml unavailable) → no-op, skip.
        3. If free_gb >= min_free_gb → proceed immediately.
        4. If auto_kill=True → call kill_zombies() to free stuck VRAM.
        5. Call wait_for_vram() — poll until threshold met or timeout.
        6. If wait_for_vram returns False → raise GPUVRAMInsufficientError.

        Spec: REQ-INFRA-039, REQ-INFRA-040, SCENARIO-INFRA-047/048/049
        """
        n_gpus = self._n_gpus()
        if n_gpus == 0:
            _log.debug("GPUVRAMGate: no GPUs detected — gate is a no-op")
            return self

        for gpu_idx in range(n_gpus):
            status = self.check_vram(gpu_idx)
            if status.total_mb == 0:
                continue  # pynvml reported this GPU as unavailable

            if status.free_gb >= self.min_free_gb:
                _log.info(
                    "GPUVRAMGate: GPU %d OK — %.2f GB free (>= %.2f GB threshold)",
                    gpu_idx, status.free_gb, self.min_free_gb,
                )
                continue

            _log.warning(
                "GPUVRAMGate: GPU %d low VRAM — %.2f GB free < %.2f GB threshold; "
                "auto_kill=%s",
                gpu_idx, status.free_gb, self.min_free_gb, self.auto_kill,
            )

            if self.auto_kill:
                self.kill_zombies(gpu_idx)

            if not self.wait_for_vram(gpu_idx):
                final = self.check_vram(gpu_idx)
                raise GPUVRAMInsufficientError(
                    gpu_index=gpu_idx,
                    free_gb=final.free_gb,
                    min_free_gb=self.min_free_gb,
                )

        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """No cleanup needed — VRAM is managed by the model load / unload lifecycle."""
        pass

"""ConductorSessionHealthCheck — GPU and environment health verification at conductor session start.

**Why this module exists (RETRO-034, milestone .34):**
    During milestone .34, three zombie processes held 23,795 MB of VRAM on GPU 0 for the
    ENTIRE milestone (roughly 11.5 hours).  GPU 0 showed 97% VRAM saturation but 0%
    utilization — the classic zombie signature: memory locked by a dead-but-not-reaped
    process.  Meanwhile, GPU 0 peaked at 82°C during runaway experiments, shortening
    hardware life on an RTX 3090 (safe max ~83°C, recommended sustained max ~80°C).

    A conductor-level session health check at startup would have:
    1. Detected the zombie processes and killed them before Experiment 1 ran.
    2. Caught the CARNOT_FORCE_LIVE propagation failure (RETRO-022) for all milestones
       at once, instead of each experiment applying the per-experiment workaround.
    3. Blocked the conductor if thermals were already at risk (proactive, not reactive).

**What this module provides:**
    GPUHealth — per-GPU health snapshot (VRAM, utilization, temperature).
    ZombieProcess — a GPU process that is consuming VRAM but appears dead/stalled.
    SessionHealthResult — the final verdict of a full health check run.
    ConductorSessionHealthCheck — orchestrates the health check and auto-remediation.

**Zombie detection algorithm:**
    pynvml enumerates all processes holding VRAM on each GPU.  For each such PID,
    psutil is queried for the process's CPU times and create_time.  A process is
    classified as a zombie candidate when:
      - It holds > 500 MB VRAM (enough to be significant), AND
      - Its wall-clock age exceeds 300 seconds (5 minutes), AND
      - Its GPU utilization contribution is 0% (no active compute).
    In practice this catches experiment processes that crashed without releasing VRAM.

**Thermal gate (WHY 80°C):**
    The RTX 3090 throttles at 83°C and may sustain damage above 85°C under prolonged
    load.  NVIDIA's own recommended sustained operating temperature is 80°C.  If any
    GPU is at or above 80°C when the conductor starts, we assume the previous session
    left hardware in thermal stress and pause until the operator reviews the situation.

**pynvml / psutil availability:**
    Both are optional.  If pynvml is unavailable (CI, CPU-only machines), GPU health
    defaults to 'unknown' — the health check still runs but cannot inspect GPU state.
    If psutil is unavailable, zombie wall-time cannot be measured (fallback: age=0).

Spec: REQ-INFRA-036, REQ-INFRA-037, REQ-INFRA-038,
      SCENARIO-INFRA-044, SCENARIO-INFRA-045, SCENARIO-INFRA-046
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from carnot.pipeline.env_autofix import apply_env_autofix

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPUHealth
# ---------------------------------------------------------------------------


@dataclass
class GPUHealth:
    """Per-GPU health snapshot.

    Fields
    ------
    gpu_index : int
        Zero-based GPU index (as reported by pynvml / nvidia-smi).
    vram_used_mb : int
        VRAM currently in use on this GPU in megabytes.
    utilization_pct : int
        GPU compute utilization percentage (0-100).  Obtained from
        pynvml.nvmlDeviceGetUtilizationRates().gpu.
    temp_c : int
        GPU die temperature in Celsius.  Obtained from
        pynvml.nvmlDeviceGetTemperature(NVML_TEMPERATURE_GPU).

    Computed properties
    -------------------
    is_zombie_saturated : bool
        True when the GPU holds > 1000 MB VRAM but shows 0% utilization.
        This is the RETRO-034 zombie signature: 23,795 MB held, 0% compute.
        Threshold of 1000 MB chosen to exclude small driver/context allocations
        (which are normal even on idle GPUs).
    is_overheating : bool
        True when temperature is at or above 80°C — the thermal gate threshold.
        At 80°C+ we pause the conductor; at 83°C+ the GPU self-throttles.
    is_idle : bool
        True when VRAM usage is below 200 MB.  Below this threshold the GPU
        has only the driver's small context allocation and holds no model weights.

    Spec: REQ-INFRA-036, SCENARIO-INFRA-044
    """

    gpu_index: int
    vram_used_mb: int
    utilization_pct: int
    temp_c: int

    @property
    def is_zombie_saturated(self) -> bool:
        """True when VRAM > 1000 MB AND utilization == 0% (zombie signature)."""
        return self.vram_used_mb > 1000 and self.utilization_pct == 0

    @property
    def is_overheating(self) -> bool:
        """True when temperature >= 80°C (thermal gate threshold)."""
        return self.temp_c >= 80

    @property
    def is_idle(self) -> bool:
        """True when VRAM < 200 MB (driver context only, no model weights)."""
        return self.vram_used_mb < 200

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict for artifact embedding."""
        return {
            "gpu_index": self.gpu_index,
            "vram_used_mb": self.vram_used_mb,
            "utilization_pct": self.utilization_pct,
            "temp_c": self.temp_c,
            "is_zombie_saturated": self.is_zombie_saturated,
            "is_overheating": self.is_overheating,
            "is_idle": self.is_idle,
        }


# ---------------------------------------------------------------------------
# ZombieProcess
# ---------------------------------------------------------------------------


@dataclass
class ZombieProcess:
    """A GPU-attached process that appears to be dead or permanently stalled.

    Zombie processes are the RETRO-034 root cause: a crashed experiment's Python
    process exits (or hangs in a wait loop) but the CUDA context is not released
    until the OS reclaims the PID.  pynvml can still see the PID holding VRAM
    even after the process is unkillable, which is why proactive detection and
    SIGKILL are needed.

    Fields
    ------
    pid : int
        OS process ID.
    gpu_index : int
        Which GPU the process is attached to.
    vram_mb : int
        VRAM held by this process in megabytes.
    wall_time_s : float
        Wall-clock seconds since the process was created (from psutil.create_time).
        If psutil is unavailable, this is 0.0 and should_kill will be False.

    Computed properties
    -------------------
    should_kill : bool
        True when the process has been running for > 300 seconds AND holds > 500 MB
        VRAM.  The 300-second threshold filters out legitimate short-lived processes
        (model loaders, preflight scripts).  The 500 MB threshold filters out tiny
        background processes that happen to have a CUDA context open.

    Spec: REQ-INFRA-037, SCENARIO-INFRA-045
    """

    pid: int
    gpu_index: int
    vram_mb: int
    wall_time_s: float

    @property
    def should_kill(self) -> bool:
        """True when process is old enough and large enough to be a zombie candidate."""
        return self.wall_time_s > 300 and self.vram_mb > 500

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict."""
        return {
            "pid": self.pid,
            "gpu_index": self.gpu_index,
            "vram_mb": self.vram_mb,
            "wall_time_s": round(self.wall_time_s, 1),
            "should_kill": self.should_kill,
        }


# ---------------------------------------------------------------------------
# SessionHealthResult
# ---------------------------------------------------------------------------


@dataclass
class SessionHealthResult:
    """Final verdict of a ConductorSessionHealthCheck.run() call.

    Fields
    ------
    env_ok : bool
        True when CARNOT_FORCE_LIVE is '1' after apply_env_autofix().
    gpu_ok : bool
        True when all detected GPUs have VRAM < 200 MB (idle).
        Also True when no GPUs are present (CI / CPU-only machines).
    zombies_killed : int
        Number of zombie processes actually killed (0 in auto_remediate=False mode).
    thermal_ok : bool
        True when no GPU temperature is >= 80°C.
    honest_verdict : str
        One of:
          'session_healthy'          — all checks passed, nothing remediated
          'session_remediated'       — checks passed after auto-remediation (env fix or zombie kills)
          'session_thermal_blocked'  — thermal gate triggered; conductor must pause

    Spec: REQ-INFRA-036, REQ-INFRA-037, REQ-INFRA-038
    """

    env_ok: bool
    gpu_ok: bool
    zombies_killed: int
    thermal_ok: bool
    honest_verdict: str

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict."""
        return {
            "env_ok": self.env_ok,
            "gpu_ok": self.gpu_ok,
            "zombies_killed": self.zombies_killed,
            "thermal_ok": self.thermal_ok,
            "honest_verdict": self.honest_verdict,
        }


# ---------------------------------------------------------------------------
# ConductorSessionHealthCheck
# ---------------------------------------------------------------------------


class ConductorSessionHealthCheck:
    """Verify and optionally remediate GPU + environment health at conductor session start.

    Run this ONCE at the very beginning of a conductor session, before any experiment
    is spawned.  It checks four things:

    1. Environment: CARNOT_FORCE_LIVE is set (or can be set via apply_env_autofix).
    2. GPU VRAM: both GPUs are under 200 MB (idle — no zombie processes holding memory).
    3. Thermal: no GPU is at or above 80°C (thermal gate to prevent hardware damage).
    4. Zombie processes: any GPU-attached processes older than 5 minutes holding > 500 MB
       are identified and (if auto_remediate=True) killed with SIGKILL.

    Parameters
    ----------
    auto_remediate : bool
        When True (default, production use): kills zombie processes and patches the
        environment.  When False (CI, non-destructive mode): inspects but does not
        kill anything — safe to run in automated test pipelines without side effects.

    Usage::

        result = ConductorSessionHealthCheck(auto_remediate=True).run()
        if not result.thermal_ok:
            sys.exit(1)  # conductor must pause until thermals recover

    Spec: REQ-INFRA-036, REQ-INFRA-037, REQ-INFRA-038,
          SCENARIO-INFRA-044, SCENARIO-INFRA-045, SCENARIO-INFRA-046
    """

    def __init__(self, auto_remediate: bool = True) -> None:
        self.auto_remediate = auto_remediate

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> SessionHealthResult:
        """Run the full health check and return a SessionHealthResult.

        Algorithm:
        1. apply_env_autofix() — injects CARNOT_FORCE_LIVE=1 if absent and GPU is present.
        2. _check_env() — confirm CARNOT_FORCE_LIVE is '1'.
        3. _check_gpu_health() — snapshot VRAM, utilization, temperature for each GPU.
        4. Identify zombie processes on zombie-saturated GPUs.
        5. If auto_remediate: kill zombie processes (_kill_zombies).
        6. Re-check GPU health after kills (allow 2 seconds for CUDA context release).
        7. Compute honest_verdict and return SessionHealthResult.

        Returns
        -------
        SessionHealthResult
            Fully populated result.  Never raises — errors in pynvml/psutil are
            caught and logged; the check degrades gracefully to 'unknown' state.

        Spec: REQ-INFRA-036, SCENARIO-INFRA-044
        """
        # Step 1: env fix
        apply_env_autofix()

        # Step 2: env check
        env_ok = self._check_env()

        # Step 3: GPU snapshot
        gpu_healths = self._check_gpu_health()

        # Step 4: find zombie processes on saturated GPUs
        zombie_gpu_indices = [g.gpu_index for g in gpu_healths if g.is_zombie_saturated]
        zombies = self._find_zombie_processes(zombie_gpu_indices)

        # Step 5: kill zombies if auto_remediate
        zombies_killed = 0
        if self.auto_remediate and zombies:
            zombies_killed = self._kill_zombies(zombies)
            # Give the OS time to release CUDA contexts before re-checking
            if zombies_killed > 0:
                time.sleep(2)
            gpu_healths = self._check_gpu_health()

        # Step 6: compute aggregate verdicts
        gpu_ok = all(g.is_idle for g in gpu_healths) if gpu_healths else True
        thermal_ok = all(not g.is_overheating for g in gpu_healths) if gpu_healths else True

        # Step 7: honest_verdict (REQ-INFRA-038: thermal gate takes priority)
        if not thermal_ok:
            honest_verdict = "session_thermal_blocked"
        elif not env_ok or zombies_killed > 0:
            honest_verdict = "session_remediated"
        else:
            honest_verdict = "session_healthy"

        result = SessionHealthResult(
            env_ok=env_ok,
            gpu_ok=gpu_ok,
            zombies_killed=zombies_killed,
            thermal_ok=thermal_ok,
            honest_verdict=honest_verdict,
        )

        _log.info(
            "ConductorSessionHealthCheck: verdict=%s env_ok=%s gpu_ok=%s "
            "zombies_killed=%d thermal_ok=%s",
            honest_verdict,
            env_ok,
            gpu_ok,
            zombies_killed,
            thermal_ok,
        )
        return result

    def _check_env(self) -> bool:
        """Return True if CARNOT_FORCE_LIVE is '1' in the current environment.

        This is called AFTER apply_env_autofix(), so if the GPU is present,
        the var should already be set.  A False return here means either:
        - No GPU was detected (expected on CI).
        - apply_env_autofix() failed silently (unexpected; log as warning).

        Spec: REQ-INFRA-036, SCENARIO-INFRA-044
        """
        return os.environ.get("CARNOT_FORCE_LIVE") == "1"

    def _check_gpu_health(self) -> list[GPUHealth]:
        """Query pynvml for per-GPU VRAM, utilization, and temperature.

        Returns an empty list if pynvml is not installed or no GPUs are present.
        Errors are caught and logged — the caller treats an empty list as 'no GPUs,
        skip GPU checks'.

        Spec: REQ-INFRA-036
        """
        healths: list[GPUHealth] = []
        try:
            import pynvml  # noqa: PLC0415 — optional dependency

            pynvml.nvmlInit()
            n = pynvml.nvmlDeviceGetCount()
            for i in range(n):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
                vram_used_mb = int(mem.used / (1024 * 1024))
                healths.append(
                    GPUHealth(
                        gpu_index=i,
                        vram_used_mb=vram_used_mb,
                        utilization_pct=int(util.gpu),
                        temp_c=int(temp),
                    )
                )
            pynvml.nvmlShutdown()
        except Exception as exc:  # pynvml not installed, no NVIDIA driver, etc.
            _log.debug("ConductorSessionHealthCheck: pynvml unavailable (%s)", exc)
        return healths

    def _find_zombie_processes(self, gpu_indices: list[int]) -> list[ZombieProcess]:
        """Find GPU-attached processes that are candidates for zombie killing.

        For each GPU index in gpu_indices, queries pynvml for the list of
        processes holding VRAM.  For each such PID, queries psutil for the
        process creation time to compute wall_time_s.  Returns all ZombieProcess
        instances where should_kill is True.

        Parameters
        ----------
        gpu_indices : list[int]
            Indices of GPUs that are zombie-saturated (VRAM > 1000 MB, util == 0).

        Returns
        -------
        list[ZombieProcess]
            Zombie candidates with should_kill == True.

        Spec: REQ-INFRA-037
        """
        if not gpu_indices:
            return []

        candidates: list[ZombieProcess] = []
        now = time.time()

        try:
            import pynvml  # noqa: PLC0415

            pynvml.nvmlInit()
            for idx in gpu_indices:
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                try:
                    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                except Exception:
                    procs = []
                for p in procs:
                    vram_mb = int(p.usedGpuMemory / (1024 * 1024))
                    wall_time_s = self._get_process_age_s(p.pid, now)
                    z = ZombieProcess(
                        pid=p.pid,
                        gpu_index=idx,
                        vram_mb=vram_mb,
                        wall_time_s=wall_time_s,
                    )
                    if z.should_kill:
                        candidates.append(z)
            pynvml.nvmlShutdown()
        except Exception as exc:
            _log.debug("ConductorSessionHealthCheck: zombie scan failed (%s)", exc)

        return candidates

    def _get_process_age_s(self, pid: int, now: float) -> float:
        """Return wall-clock age of a process in seconds using psutil.

        Returns 0.0 if psutil is not available or the process has already exited.
        A return of 0.0 means should_kill will be False — safe default.
        """
        try:
            import psutil  # noqa: PLC0415 — optional dependency

            proc = psutil.Process(pid)
            return now - proc.create_time()
        except Exception:
            return 0.0

    def _kill_zombies(self, zombies: list[ZombieProcess]) -> int:
        """Send SIGKILL to each zombie process and return the count killed.

        SIGKILL (not SIGTERM) is used because a zombie/stalled CUDA process
        often will not respond to SIGTERM — it is already stuck in a kernel
        wait that does not process signals.  SIGKILL forces the OS to reclaim
        all resources including the CUDA context.

        Parameters
        ----------
        zombies : list[ZombieProcess]
            Zombie candidates (all have should_kill == True).

        Returns
        -------
        int
            Number of processes successfully killed.

        Spec: REQ-INFRA-037, SCENARIO-INFRA-045
        """
        killed = 0
        for z in zombies:
            try:
                import signal as _signal  # noqa: PLC0415

                os.kill(z.pid, _signal.SIGKILL)
                _log.warning(
                    "ConductorSessionHealthCheck: SIGKILL sent to PID %d "
                    "(GPU %d, %d MB, age %.0f s) — RETRO-034 zombie remediation",
                    z.pid,
                    z.gpu_index,
                    z.vram_mb,
                    z.wall_time_s,
                )
                killed += 1
            except ProcessLookupError:
                # Process already dead — still count as 'handled'
                killed += 1
            except PermissionError as exc:
                _log.error(
                    "ConductorSessionHealthCheck: cannot kill PID %d (%s) — "
                    "run conductor as root or use nvidia-smi to kill manually",
                    z.pid,
                    exc,
                )
        return killed

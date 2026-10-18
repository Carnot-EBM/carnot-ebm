"""vram_loop_eviction.py — VRAM eviction with nvidia-smi verification retry loop.

**Why this module exists (RETRO-028 Fix v5, Exp 810):**
    Fix v4 (Exp 795) applied kill_gpu_zombies() and a single pkill pass, but did
    NOT use a verification loop.  If other processes restart or hold VRAM between
    the kill and the threshold check, the model load proceeds into CUDA OOM anyway.

    Fix v5 mandates:
    1. kill_gpu_zombies(gpu_index) — primary SIGKILL pass.
    2. Retry loop (up to max_retries=3, 10s sleep per iteration):
       a. Query nvidia-smi for compute PIDs using > 100 MB; SIGKILL them.
       b. Sleep retry_sleep_s to let the GPU driver drain VRAM.
       c. Read VRAM via nvidia-smi; if < threshold_mb (500 MB), return cleared.
    3. If VRAM is still >= threshold_mb after max_retries, abort with
       abort_reason="max_retries_exceeded" — do NOT attempt model load.

    The key invariant: the model load MUST NOT proceed until this function
    returns vram_cleared=True.  Any caller that ignores vram_cleared=False
    will get CUDA OOM.

**What this module provides:**
    ``VRAMLoopEvictionResult`` — structured dataclass for the retry-loop outcome.
    ``evict_vram_with_loop()`` — run the retry-loop protocol, return result.

Spec: REQ-LOADER-014, SCENARIO-LOADER-014, SCENARIO-LOADER-015
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field

from carnot.pipeline.gpu_zombie_killer import _pid_is_protected_server, kill_gpu_zombies

_log = logging.getLogger(__name__)

# Processes using more than this many MiB of VRAM are killed in each retry.
_PKILL_VRAM_THRESHOLD_MB: float = 100.0


@dataclass
class VRAMLoopEvictionResult:
    """Structured outcome from ``evict_vram_with_loop()``.

    Records every retry iteration so the caller can write a full audit trail
    to the experiment artifact — essential for diagnosing stuck-VRAM failures.

    Fields
    ------
    gpu_index : int
        CUDA device index that was evicted (0-based).
    n_retries_attempted : int
        Number of retry iterations that actually ran (0 … max_retries).
    vram_mb_per_retry : list[float]
        VRAM used (MiB) as read at the END of each retry iteration.
        Length equals n_retries_attempted unless vram_cleared=True on a middle
        retry (in which case the loop exits early and this list is shorter than
        max_retries).
    final_vram_mb : float
        VRAM used (MiB) after the last completed retry, or 0.0 if nvidia-smi
        was unavailable.  Equals the last entry of vram_mb_per_retry when the
        list is non-empty.
    vram_cleared : bool
        True iff final_vram_mb < threshold_mb.  The model load gate.
    abort_reason : str | None
        Set to ``"max_retries_exceeded"`` when all retries are exhausted and
        VRAM is still above threshold.  None when vram_cleared=True.
    honest_verdict : str
        One of:
        - ``"vram_cleared"``             — threshold met; model load is safe.
        - ``"max_retries_exceeded"``     — all retries exhausted; abort.
        - ``"nvidia_smi_unavailable"``   — nvidia-smi binary not found.

    Spec: REQ-LOADER-014
    """

    gpu_index: int
    n_retries_attempted: int = 0
    vram_mb_per_retry: list[float] = field(default_factory=list)
    final_vram_mb: float = 0.0
    vram_cleared: bool = False
    abort_reason: str | None = None
    honest_verdict: str = "max_retries_exceeded"


# ---------------------------------------------------------------------------
# Internal helpers (private)
# ---------------------------------------------------------------------------


def _query_nvidia_smi(args: list[str]) -> str | None:
    """Run nvidia-smi with *args*; return stdout string or None if unavailable.

    FileNotFoundError means the binary is not installed — callers treat None
    as "GPU state unknown" and return a safe-failure result.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi"] + args,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout
    except FileNotFoundError:
        return None
    except Exception as exc:
        _log.debug("nvidia-smi query failed: %s", exc)
        return None


def _get_vram_used_mb(gpu_index: int) -> float:
    """Return GPU memory used in MiB for *gpu_index*, or 0.0 if unavailable."""
    out = _query_nvidia_smi(
        ["--query-gpu=memory.used", "--format=csv,noheader,nounits", f"-i {gpu_index}"]
    )
    if out is None:
        return 0.0
    line = out.strip().splitlines()[0].strip() if out.strip() else ""
    try:
        return float(line)
    except (ValueError, IndexError):
        return 0.0


def _get_compute_apps_with_memory(gpu_index: int) -> list[tuple[int, float]]:
    """Return (pid, used_memory_mb) for each compute process on *gpu_index*.

    Uses ``--query-compute-apps=pid,used_memory`` which gives per-process VRAM
    usage.  This lets us selectively kill only large-VRAM processes (> 100 MB)
    rather than blindly killing everything — important if the calling process
    itself has a small CUDA context we must not kill.

    Returns an empty list when nvidia-smi is unavailable or no compute apps
    are found.
    """
    out = _query_nvidia_smi(
        [
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader",
            f"-i {gpu_index}",
        ]
    )
    if out is None:
        return []

    results: list[tuple[int, float]] = []
    for line in out.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0].strip())
            # Memory column may include units like "MiB" — strip non-numeric suffix.
            mem_str = parts[1].strip().split()[0]
            mem_mb = float(mem_str)
            results.append((pid, mem_mb))
        except (ValueError, IndexError):
            continue
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evict_vram_with_loop(
    gpu_index: int = 1,
    max_retries: int = 3,
    retry_sleep_s: float = 10.0,
    threshold_mb: float = 500.0,
) -> VRAMLoopEvictionResult:
    """Evict GPU VRAM using a kill-and-verify retry loop.

    This is the RETRO-028 Fix v5 protocol.  It replaces the single-pass
    eviction used in Fix v4 (Exp 795) with a verified retry loop so that
    processes that restart or linger after the first SIGKILL are caught
    on subsequent iterations.

    Algorithm
    ---------
    1. kill_gpu_zombies(gpu_index) — primary SIGKILL pass via the existing
       zombie killer (belt-and-suspenders with the loop below).
    2. For i in range(max_retries):
       a. Query nvidia-smi for compute PIDs with used_memory > 100 MB.
       b. SIGKILL each (excluding this process).
       c. Sleep retry_sleep_s to allow the GPU driver to drain VRAM.
       d. Read VRAM used via nvidia-smi.
       e. If VRAM < threshold_mb: return VRAMLoopEvictionResult(vram_cleared=True).
       f. Append vram_used to vram_mb_per_retry and continue.
    3. If VRAM still >= threshold_mb after max_retries:
       return VRAMLoopEvictionResult(vram_cleared=False,
                                     abort_reason="max_retries_exceeded").

    Parameters
    ----------
    gpu_index : int
        CUDA device index to evict (0-based, default=1 per REQ-LOADER-013).
    max_retries : int
        Maximum number of kill-and-verify iterations before giving up.
    retry_sleep_s : float
        Seconds to sleep between the SIGKILL and the VRAM re-read.  10 s is
        chosen because the CUDA driver may take several seconds to drain freed
        allocations, especially on large (14+ GiB) models.
    threshold_mb : float
        VRAM (MiB) that must be free before model load is allowed.  500 MB
        gives room for the CUDA context itself without false-blocking.

    Returns
    -------
    VRAMLoopEvictionResult
        Fully populated.  Never raises.

    Spec: REQ-LOADER-014, SCENARIO-LOADER-014, SCENARIO-LOADER-015
    """
    result = VRAMLoopEvictionResult(gpu_index=gpu_index)

    # Step 1: Check nvidia-smi availability with a cheap probe.
    probe = _query_nvidia_smi(
        ["--query-gpu=memory.used", "--format=csv,noheader,nounits", f"-i {gpu_index}"]
    )
    if probe is None:
        result.honest_verdict = "nvidia_smi_unavailable"
        _log.debug("evict_vram_with_loop: nvidia-smi unavailable — aborting eviction")
        return result

    # Primary SIGKILL pass — catches the bulk of zombie processes before the loop.
    kill_gpu_zombies(gpu_index=gpu_index)

    my_pid = os.getpid()

    # Step 2: Retry loop with nvidia-smi verification.
    for i in range(max_retries):
        result.n_retries_attempted += 1

        # Kill any compute process using > _PKILL_VRAM_THRESHOLD_MB MiB.
        apps = _get_compute_apps_with_memory(gpu_index)
        for pid, mem_mb in apps:
            if pid == my_pid:
                continue
            if mem_mb <= _PKILL_VRAM_THRESHOLD_MB:
                continue
            # REQ-INFRA-079: the primary pass's server exemption must not be
            # defeated by this retry loop. Servers are never eviction targets.
            if _pid_is_protected_server(pid):
                _log.info(
                    "evict_vram_with_loop retry %d: SKIP protected server PID %d on gpu=%d "
                    "(REQ-INFRA-079)",
                    i + 1,
                    pid,
                    gpu_index,
                )
                continue
            try:
                os.kill(pid, signal.SIGKILL)
                _log.warning(
                    "evict_vram_with_loop retry %d: SIGKILL PID %d (%.0f MiB) on gpu=%d",
                    i + 1,
                    pid,
                    mem_mb,
                    gpu_index,
                )
            except OSError as exc:
                _log.warning(
                    "evict_vram_with_loop retry %d: could not kill PID %d — %s",
                    i + 1,
                    pid,
                    exc,
                )

        # Wait for GPU driver to drain freed allocations.
        time.sleep(retry_sleep_s)

        # Read VRAM after the sleep.
        vram_used = _get_vram_used_mb(gpu_index)
        result.vram_mb_per_retry.append(vram_used)
        result.final_vram_mb = vram_used

        _log.info(
            "evict_vram_with_loop retry %d/%d: vram_used=%.0f MB threshold=%.0f MB",
            i + 1,
            max_retries,
            vram_used,
            threshold_mb,
        )

        if vram_used < threshold_mb:
            result.vram_cleared = True
            result.honest_verdict = "vram_cleared"
            _log.info(
                "evict_vram_with_loop: VRAM cleared on retry %d (%.0f MB < %.0f MB threshold)",
                i + 1,
                vram_used,
                threshold_mb,
            )
            return result

    # All retries exhausted — VRAM still too high.
    result.vram_cleared = False
    result.abort_reason = "max_retries_exceeded"
    result.honest_verdict = "max_retries_exceeded"
    _log.error(
        "evict_vram_with_loop: VRAM still %.0f MB after %d retries (threshold=%.0f MB) — ABORT",
        result.final_vram_mb,
        max_retries,
        threshold_mb,
    )
    return result

"""GPU zombie killer — mandatory pre-model-load cleanup for ExperimentTemplate.setup_gpu().

**Researcher summary (RETRO-028, RETRO-SOTA-GGUF-TIMEOUT):**
    RETRO-028: Gemma4 allocation of 14.89 GiB failed because 15 GiB was already
    occupied by zombie processes from a previous experiment.  RETRO-SOTA-GGUF-TIMEOUT
    (Exp 769): timed out for the same reason — GPUs were OOM before model load began.

    The existing ExperimentTemplate.kill_gpu_zombies() classmethod (called at session
    start via setup()) uses a VRAM+utilization threshold heuristic.  That heuristic
    misses zombies that look "active" by utilization but are actually stalled.

    This module takes a simpler and more aggressive approach:
    1. Ask nvidia-smi for ALL PIDs currently holding compute memory on a specific GPU.
    2. Kill every one of those PIDs except the current process and any caller-supplied
       exclusions, using SIGKILL (not SIGTERM) for guaranteed termination.
    3. Wait 2 seconds for the GPU driver to reclaim the released VRAM.
    4. Report the before/after VRAM delta with an honest_verdict string.

    This is intentionally aggressive: if a process is holding GPU memory before a model
    load, it is a zombie for our purposes — we need that VRAM.

**What this module provides:**
    ``GPUZombieResult`` — structured dataclass describing the kill operation outcome.
    ``get_gpu_memory_pids(gpu_index)`` — list PIDs holding compute memory on a GPU.
    ``kill_gpu_zombies(gpu_index, exclude_pids)`` — kill zombies, return result.

**Honest verdict semantics:**
    - ``"zombies_killed_vram_freed"``   — PIDs killed AND vram_freed_mb > 100
    - ``"zombies_killed_vram_unclear"`` — PIDs killed but vram_freed_mb <= 100
    - ``"no_zombies_found"``            — nvidia-smi reported no compute processes
    - ``"nvidia_smi_unavailable"``      — nvidia-smi binary not present on this host

Spec: REQ-INFRA-055, REQ-INFRA-056, SCENARIO-INFRA-064, SCENARIO-INFRA-065
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field

_log = logging.getLogger(__name__)

# How long to wait after SIGKILL for the GPU driver to reclaim VRAM.
# The CUDA driver typically drains in 1-3 seconds; 2 s is the validated sweet spot.
_POST_KILL_WAIT_S: float = 2.0


# ---------------------------------------------------------------------------
# GPUZombieResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class GPUZombieResult:
    """Structured outcome from ``kill_gpu_zombies()``.

    All fields are populated regardless of whether any PIDs were killed so that
    downstream code can always log or serialize the result without None-checks.

    Fields
    ------
    gpu_index : int
        Which GPU was inspected (0-based device index).
    pids_found : list[int]
        All PIDs reported by nvidia-smi as holding compute memory on this GPU.
        Includes the calling process and any other excluded PIDs — i.e. this is
        the raw set BEFORE exclusion filtering.
    pids_killed : list[int]
        PIDs that were sent SIGKILL.  Never contains os.getpid() or any PID in
        the caller-supplied exclude list.
    vram_before_mb : float
        GPU memory used (in MiB) before any kills were issued.
    vram_after_mb : float
        GPU memory used (in MiB) 2 seconds after the last kill.
    vram_freed_mb : float
        vram_before_mb - vram_after_mb.  Negative values mean VRAM usage
        increased (possible if another process loaded a model during the wait).
    kill_attempted : bool
        True iff at least one PID was outside the exclude set (we tried to kill).
    honest_verdict : str
        One of the four verdict strings defined in the module docstring.

    Spec: REQ-INFRA-055, REQ-INFRA-056
    """

    gpu_index: int
    pids_found: list[int] = field(default_factory=list)
    pids_killed: list[int] = field(default_factory=list)
    vram_before_mb: float = 0.0
    vram_after_mb: float = 0.0
    vram_freed_mb: float = 0.0
    kill_attempted: bool = False
    honest_verdict: str = "no_zombies_found"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _query_nvidia_smi(args: list[str]) -> str | None:
    """Run nvidia-smi with *args* and return stdout, or None if unavailable.

    Why a helper: every nvidia-smi call site needs the same FileNotFoundError
    guard (binary may not be present on CPU-only hosts).  Centralising it means
    we only need one try/except instead of four.
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


def _parse_float_from_smi_output(output: str) -> float:
    """Extract the first numeric value from a single-line nvidia-smi --query-gpu output."""
    line = output.strip().splitlines()[0].strip() if output.strip() else ""
    try:
        return float(line)
    except (ValueError, IndexError):
        return 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_gpu_memory_pids(gpu_index: int = 0) -> list[int]:
    """Return a list of PIDs holding compute memory on *gpu_index*.

    Uses ``nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits``
    which enumerates every process that has allocated CUDA compute memory on the
    given device.  Returns an empty list when nvidia-smi is unavailable or when
    no processes are found (clean GPU state).

    Why this query and not ``--query-gpu``: ``--query-compute-apps`` gives per-PID
    granularity so we can identify which processes to kill.  ``--query-gpu`` gives
    only device-level totals, which is insufficient for targeted killing.

    Parameters
    ----------
    gpu_index : int
        The CUDA device index (0 = first GPU).

    Returns
    -------
    list[int]
        PIDs of all processes with compute memory allocated on *gpu_index*.
        Empty when nvidia-smi is unavailable or the GPU is clean.

    Spec: REQ-INFRA-055, REQ-INFRA-056
    """
    output = _query_nvidia_smi(
        [
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
            f"-i {gpu_index}",
        ]
    )
    if output is None:
        return []

    pids: list[int] = []
    for line in output.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pids.append(int(line))
        except ValueError:
            continue
    return pids


def _get_vram_used_mb(gpu_index: int) -> float:
    """Return GPU memory used in MiB for *gpu_index*, or 0.0 if unavailable."""
    output = _query_nvidia_smi(
        [
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            f"-i {gpu_index}",
        ]
    )
    if output is None:
        return 0.0
    return _parse_float_from_smi_output(output)


def kill_gpu_zombies(
    gpu_index: int = 0,
    exclude_pids: list[int] | None = None,
) -> GPUZombieResult:
    """Kill all processes holding GPU memory on *gpu_index*, except excluded PIDs.

    This is the mandatory pre-model-load cleanup mandated by RETRO-028 and
    RETRO-SOTA-GGUF-TIMEOUT.  It must be called inside ExperimentTemplate.setup_gpu()
    BEFORE any model loading attempt when CARNOT_FORCE_LIVE=1.

    Algorithm
    ---------
    1. If nvidia-smi is unavailable, return immediately with
       ``honest_verdict="nvidia_smi_unavailable"``.
    2. Read ``vram_before_mb`` from nvidia-smi.
    3. Get PIDs from ``get_gpu_memory_pids(gpu_index)``.
    4. Build kill list = pids_found - exclude_pids (exclude_pids always contains
       os.getpid() so we never kill ourselves).
    5. Send SIGKILL to each PID in the kill list.  Record kills in pids_killed.
    6. If any PIDs were killed, wait ``_POST_KILL_WAIT_S`` for the GPU driver to
       drain the freed VRAM.
    7. Read ``vram_after_mb``.  Compute ``vram_freed_mb = vram_before - vram_after``.
    8. Set honest_verdict based on pids_killed and vram_freed_mb.

    Parameters
    ----------
    gpu_index : int
        The CUDA device index to inspect and clean (0-based).
    exclude_pids : list[int] | None
        PIDs that must NOT be killed.  Defaults to ``[os.getpid()]`` — the calling
        process is always excluded.  Callers may pass additional PIDs (e.g. the
        conductor process) to protect.

    Returns
    -------
    GPUZombieResult
        Fully populated result.  Never raises.

    Spec: REQ-INFRA-055, REQ-INFRA-056, SCENARIO-INFRA-064, SCENARIO-INFRA-065
    """
    # Always protect the calling process.
    if exclude_pids is None:
        effective_excludes: set[int] = {os.getpid()}
    else:
        effective_excludes = set(exclude_pids) | {os.getpid()}

    result = GPUZombieResult(gpu_index=gpu_index)

    # Step 1: check nvidia-smi availability via a cheap query
    vram_before_raw = _query_nvidia_smi(
        ["--query-gpu=memory.used", "--format=csv,noheader,nounits", f"-i {gpu_index}"]
    )
    if vram_before_raw is None:
        result.honest_verdict = "nvidia_smi_unavailable"
        _log.debug("kill_gpu_zombies: nvidia-smi not available — skipping zombie kill")
        return result

    result.vram_before_mb = _parse_float_from_smi_output(vram_before_raw)

    # Step 2: enumerate PIDs
    result.pids_found = get_gpu_memory_pids(gpu_index)

    if not result.pids_found:
        result.honest_verdict = "no_zombies_found"
        result.vram_after_mb = result.vram_before_mb
        return result

    # Step 3: build kill list (exclude calling process and any caller exclusions)
    kill_targets = [pid for pid in result.pids_found if pid not in effective_excludes]
    result.kill_attempted = bool(kill_targets)

    if not kill_targets:
        # All pids_found are excluded (e.g. they are all the calling process)
        result.honest_verdict = "no_zombies_found"
        result.vram_after_mb = result.vram_before_mb
        return result

    # Step 4: SIGKILL each target
    for pid in kill_targets:
        try:
            os.kill(pid, signal.SIGKILL)
            result.pids_killed.append(pid)
            _log.warning(
                "kill_gpu_zombies: sent SIGKILL to PID %d (gpu=%d, vram_before_mb=%.0f)",
                pid,
                gpu_index,
                result.vram_before_mb,
            )
        except OSError as exc:
            _log.warning("kill_gpu_zombies: could not kill PID %d — %s", pid, exc)

    # Step 5: wait for the GPU driver to drain freed VRAM
    if result.pids_killed:
        time.sleep(_POST_KILL_WAIT_S)

    # Step 6: read vram_after and compute freed
    result.vram_after_mb = _get_vram_used_mb(gpu_index)
    result.vram_freed_mb = result.vram_before_mb - result.vram_after_mb

    # Step 7: set honest_verdict
    if result.pids_killed and result.vram_freed_mb > 100:
        result.honest_verdict = "zombies_killed_vram_freed"
    elif result.pids_killed:
        result.honest_verdict = "zombies_killed_vram_unclear"
    else:
        # Kill targets existed but all os.kill() calls failed
        result.honest_verdict = "no_zombies_found"

    _log.info(
        "kill_gpu_zombies: gpu=%d pids_found=%d pids_killed=%d "
        "vram_before=%.0f vram_after=%.0f freed=%.0f verdict=%s",
        gpu_index,
        len(result.pids_found),
        len(result.pids_killed),
        result.vram_before_mb,
        result.vram_after_mb,
        result.vram_freed_mb,
        result.honest_verdict,
    )
    return result

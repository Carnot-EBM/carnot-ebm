"""gemma_isolation.py — four-step VRAM isolation protocol before Gemma4-E4B-it load.

**Why this module exists (RETRO-028 Fix v4):**
    RETRO-028 failed to close across three attempts (Exp .58, .59, .60).  Each attempt
    crashed with CUDA OOM because ~15 GiB of VRAM was already occupied by zombie
    processes from earlier experiments before Gemma4 tried to allocate its 14.89 GiB
    footprint.

    Exp 780 deployed ``kill_gpu_zombies()``, but that alone was insufficient on GPU 0
    (which also ran at higher temperature).  The RETRO-028 Fix v4 protocol adds three
    additional steps:

    1. ``kill_gpu_zombies(gpu_index)`` — SIGKILL all processes holding GPU memory.
    2. Explicit pkill sweep — additional SIGKILL to any residual PIDs still holding
       >100 MB VRAM (belt-and-suspenders: some processes ignore the first SIGKILL).
    3. Verify <500 MB used on the target GPU via nvidia-smi before proceeding.
    4. Target GPU 1 instead of GPU 0 — GPU 1 typically runs 8-10°C cooler, reducing
       thermal throttle risk during the long 14.89 GiB model allocation.

**What this module provides:**
    ``VRAMEvictionResult`` — structured dataclass recording every step of the eviction.
    ``evict_gpu_vram(gpu_index)`` — run all four eviction steps, return result.
    ``load_gemma4_on_gpu1(model_id)`` — evict then load Gemma4 on GPU 1.

Spec: REQ-LOADER-012, REQ-LOADER-013, SCENARIO-LOADER-012, SCENARIO-LOADER-013
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field

from carnot.pipeline.gemma_loader import GemmaTransformersLoader
from carnot.pipeline.gpu_zombie_killer import (
    GPUZombieResult,
    _pid_is_protected_server,
    kill_gpu_zombies,
)

_log = logging.getLogger(__name__)

# Threshold below which a process is NOT considered a VRAM hog worth killing.
# Anything over 100 MB is evicted in the pkill sweep.
_PKILL_VRAM_THRESHOLD_MB: float = 100.0

# Target VRAM used (in MiB) that must be achieved before the model load is allowed.
# 500 MB gives room for the CUDA context itself without triggering false-block.
_VRAM_CLEAR_THRESHOLD_MB: float = 500.0

# Seconds to wait after SIGKILL before re-reading VRAM.  Mirrors gpu_zombie_killer.py.
_POST_KILL_WAIT_S: float = 3.0

# Default Gemma4 model ID — HuggingFace hub path.
_DEFAULT_MODEL_ID: str = "google/gemma-4-E4B-it"


# ---------------------------------------------------------------------------
# VRAMEvictionResult
# ---------------------------------------------------------------------------


@dataclass
class VRAMEvictionResult:
    """Structured outcome from ``evict_gpu_vram()``.

    Every field is populated regardless of how far eviction progressed, so
    downstream code can log or serialize the result without any None-checks.

    Fields
    ------
    gpu_index : int
        Which GPU was evicted (0-based CUDA device index).
    vram_before_mb : float
        GPU memory used (MiB) BEFORE any killing — snapshot taken before
        ``kill_gpu_zombies()`` runs.
    pids_killed : list[int]
        All PIDs killed across BOTH the kill_gpu_zombies() call AND the pkill
        sweep.  Combined list so the caller gets a single count.
    pkill_attempts : int
        Number of additional SIGKILL signals sent during the pkill sweep
        (after kill_gpu_zombies() has already run once).
    vram_after_mb : float
        GPU memory used (MiB) AFTER all killing and the 3-second drain wait.
    vram_clear : bool
        True if vram_after_mb < 500 MB — the gate that allows model load.
    honest_verdict : str
        One of:
        - ``"vram_cleared"``          — eviction succeeded, <500 MB free
        - ``"vram_not_cleared"``      — eviction ran but >=500 MB still used
        - ``"nvidia_smi_unavailable"``— nvidia-smi binary not found

    Spec: REQ-LOADER-012, SCENARIO-LOADER-012
    """

    gpu_index: int
    vram_before_mb: float = 0.0
    pids_killed: list[int] = field(default_factory=list)
    pkill_attempts: int = 0
    vram_after_mb: float = 0.0
    vram_clear: bool = False
    honest_verdict: str = "vram_not_cleared"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _query_nvidia_smi(args: list[str]) -> str | None:
    """Run nvidia-smi with *args*; return stdout string or None if unavailable.

    Mirrors the helper in gpu_zombie_killer.py — duplicated here so this module
    has no circular dependency on the private internals of that module.
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


def _get_compute_pids(gpu_index: int) -> list[int]:
    """Return PIDs with compute memory on *gpu_index* via nvidia-smi.

    Returns an empty list when nvidia-smi is unavailable or the GPU is clean.
    This is the same query as gpu_zombie_killer.get_gpu_memory_pids() — re-used
    here to re-check after the kill_gpu_zombies() call without importing the
    private implementation.
    """
    out = _query_nvidia_smi(
        [
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
            f"-i {gpu_index}",
        ]
    )
    if out is None:
        return []
    pids: list[int] = []
    for line in out.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pids.append(int(line))
        except ValueError:
            continue
    return pids


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evict_gpu_vram(gpu_index: int = 1) -> VRAMEvictionResult:
    """Run the four-step VRAM eviction protocol and return a structured result.

    The four steps are:
    1. Read ``vram_before_mb`` from nvidia-smi (snapshot before any killing).
    2. Call ``kill_gpu_zombies(gpu_index)`` — SIGKILL all processes holding
       compute memory on the target GPU.
    3. Pkill sweep — re-query nvidia-smi for any residual PIDs.  SIGKILL any
       PID still present (excluding os.getpid()).  Increment pkill_attempts.
    4. Wait ``_POST_KILL_WAIT_S`` seconds, then read ``vram_after_mb``.
       Set ``vram_clear = (vram_after_mb < _VRAM_CLEAR_THRESHOLD_MB)``.

    Why the pkill sweep (step 3): some processes survive a single SIGKILL for
    up to 2-3 seconds (they're in an uninterruptible kernel wait while the CUDA
    driver cleans up).  A second sweep catches these stragglers.

    Parameters
    ----------
    gpu_index : int
        CUDA device index to evict (default=1, the cooler GPU per REQ-LOADER-013).

    Returns
    -------
    VRAMEvictionResult
        Fully populated.  Never raises.

    Spec: REQ-LOADER-012, SCENARIO-LOADER-012
    """
    result = VRAMEvictionResult(gpu_index=gpu_index)

    # Step 1: check nvidia-smi availability and read initial VRAM.
    vram_raw = _query_nvidia_smi(
        ["--query-gpu=memory.used", "--format=csv,noheader,nounits", f"-i {gpu_index}"]
    )
    if vram_raw is None:
        result.honest_verdict = "nvidia_smi_unavailable"
        _log.debug("evict_gpu_vram: nvidia-smi unavailable — skipping eviction")
        return result

    result.vram_before_mb = _get_vram_used_mb(gpu_index)

    # Step 2: kill_gpu_zombies() — primary kill pass.
    zombie_result: GPUZombieResult = kill_gpu_zombies(gpu_index=gpu_index)
    result.pids_killed.extend(zombie_result.pids_killed)
    _log.info(
        "evict_gpu_vram: kill_gpu_zombies gpu=%d verdict=%s pids_killed=%d",
        gpu_index,
        zombie_result.honest_verdict,
        len(zombie_result.pids_killed),
    )

    # Step 3: pkill sweep — re-query and SIGKILL any residual PIDs.
    my_pid = os.getpid()
    residual_pids = _get_compute_pids(gpu_index)
    for pid in residual_pids:
        if pid == my_pid:
            continue
        # REQ-INFRA-079: step 2's server exemption must not be defeated ten
        # lines below it. An inference server is never a residual to sweep.
        if _pid_is_protected_server(pid):
            _log.info(
                "evict_gpu_vram: SKIP protected server PID %d on gpu=%d (REQ-INFRA-079)",
                pid,
                gpu_index,
            )
            continue
        if pid in result.pids_killed:
            # Already killed in step 2 — send a second SIGKILL anyway in case
            # the process is still draining (belt-and-suspenders).
            pass
        try:
            os.kill(pid, signal.SIGKILL)
            result.pkill_attempts += 1
            if pid not in result.pids_killed:
                result.pids_killed.append(pid)
            _log.warning("evict_gpu_vram: pkill sweep SIGKILL PID %d on gpu=%d", pid, gpu_index)
        except OSError as exc:
            _log.warning("evict_gpu_vram: could not pkill PID %d — %s", pid, exc)

    # Step 4: wait for GPU driver to drain, then re-read VRAM.
    if result.pids_killed:
        time.sleep(_POST_KILL_WAIT_S)

    result.vram_after_mb = _get_vram_used_mb(gpu_index)
    result.vram_clear = result.vram_after_mb < _VRAM_CLEAR_THRESHOLD_MB

    if result.vram_clear:
        result.honest_verdict = "vram_cleared"
    else:
        result.honest_verdict = "vram_not_cleared"

    _log.info(
        "evict_gpu_vram: gpu=%d vram_before=%.0f vram_after=%.0f "
        "pids_killed=%d pkill_attempts=%d vram_clear=%s verdict=%s",
        gpu_index,
        result.vram_before_mb,
        result.vram_after_mb,
        len(result.pids_killed),
        result.pkill_attempts,
        result.vram_clear,
        result.honest_verdict,
    )
    return result


def load_gemma4_on_gpu1(
    model_id: str = _DEFAULT_MODEL_ID,
) -> dict:
    """Evict VRAM on GPU 1, then load Gemma4-E4B-it on cuda:1.

    Returns a dict with:
    - ``"loaded"`` (bool) — True if model loaded successfully.
    - ``"device"`` (str | None) — ``"cuda:1"`` if loaded, else None.
    - ``"vram_before_mb"`` (float) — VRAM used before eviction.
    - ``"vram_after_mb"`` (float) — VRAM used after eviction (before load).
    - ``"vram_clear"`` (bool) — True if eviction succeeded (<500 MB).
    - ``"pids_killed"`` (list[int]) — all PIDs killed during eviction.
    - ``"pkill_attempts"`` (int) — pkill sweep SIGKILL count.
    - ``"reason"`` (str | None) — failure reason when loaded=False.

    Why GPU 1: per REQ-LOADER-013, GPU 1 typically runs 8-10°C cooler than
    GPU 0 on this machine.  Allocating the large 14.89 GiB Gemma4 footprint
    on the cooler device reduces thermal throttle risk during the long load.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID to load.  Must be a Gemma model (validated by
        GemmaTransformersLoader).

    Returns
    -------
    dict
        Always returns a dict — never raises.

    Spec: REQ-LOADER-013, SCENARIO-LOADER-013
    """
    eviction = evict_gpu_vram(gpu_index=1)
    base = {
        "vram_before_mb": eviction.vram_before_mb,
        "vram_after_mb": eviction.vram_after_mb,
        "vram_clear": eviction.vram_clear,
        "pids_killed": eviction.pids_killed,
        "pkill_attempts": eviction.pkill_attempts,
    }

    if not eviction.vram_clear:
        _log.error(
            "load_gemma4_on_gpu1: eviction failed (vram_after=%.0f MB >= 500 MB threshold) — aborting",
            eviction.vram_after_mb,
        )
        return {**base, "loaded": False, "device": None, "reason": "vram_not_cleared"}

    # Load GemmaTransformersLoader with explicit device_map pointing to cuda:1.
    # device_map={"": "cuda:1"} tells transformers to place all model layers on
    # device 1 — equivalent to .to("cuda:1") but compatible with multi-layer dispatch.
    try:
        loader = GemmaTransformersLoader(model_id=model_id, device={"": "cuda:1"})
        loader.load()
        _log.info("load_gemma4_on_gpu1: model loaded on cuda:1 (model_id=%s)", model_id)
        return {
            **base,
            "loaded": True,
            "device": "cuda:1",
            "reason": None,
            "loader": loader,
        }
    except Exception as exc:
        _log.error("load_gemma4_on_gpu1: GemmaTransformersLoader.load() failed — %s", exc)
        return {**base, "loaded": False, "device": None, "reason": str(exc)}

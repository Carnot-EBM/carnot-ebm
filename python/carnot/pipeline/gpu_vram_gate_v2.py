"""GPUVRAMGateV2 — kill-first VRAM guard that fixes the race condition from RETRO-044.

**Why GPUVRAMGateV2 exists (RETRO-044, four consecutive milestones lost):**
    GPUVRAMGate (Exp 474) used a check-first, kill-if-needed order:
      1. check_vram() — read current VRAM
      2. If insufficient: kill_zombies() — SIGKILL zombie processes
      3. wait_for_vram() — poll until threshold met

    The fatal flaw: the GPU driver holds the zombie process's VRAM context for 5-15
    seconds AFTER the SIGKILL while draining queued GPU operations from the dead
    process.  During this drain window, VRAM is still reported as consumed.  The
    wait_for_vram() poll fires during this drain window, sees VRAM still above
    threshold, hits the wait_seconds deadline, and defers the experiment with
    gpu_vram_insufficient — even though the VRAM WOULD have been free in 5-15 more
    seconds.

    Exps 476, 478, and 479 all hit gpu_vram_insufficient for exactly this reason,
    confirmed by RETRO-036 and RETRO-044.

**The fix (REQ-INFRA-049): kill FIRST, sleep for the drain window, THEN check.**
    New order:
      1. kill_zombies() — SIGKILL all zombie processes unconditionally
      2. sleep(zombie_drain_sleep_seconds) — let the driver drain the killed contexts
      3. check_vram() — NOW the freed VRAM is visible
      4. If still insufficient: wait_seconds loop → check again

    This eliminates the race condition entirely.  By the time check_vram() fires,
    the drain window has already elapsed and the VRAM is actually reported as free.

**Why default zombie_drain_sleep_seconds=15:**
    Benchmark observations from retro .35/.36: the RTX 3090 driver drains CUDA
    contexts within 10-15 seconds after SIGKILL.  15 seconds provides a 0-5 second
    safety margin beyond the observed maximum.  Shorter sleep (e.g. 5s) would still
    fire during the drain window on loaded systems; longer (e.g. 30s) wastes startup
    time unnecessarily.

**Why kill_first=True is the default:**
    There is no reason to check VRAM before killing zombies.  If zombies exist, they
    will fail the check.  If no zombies exist, kill_zombies() returns 0 and is a
    sub-second no-op.  Always killing first is strictly better: it eliminates the
    race with zero downside.  kill_first=False is preserved for backward compatibility
    only — it reproduces the original (broken) behavior for tests that verify the
    old ordering.

**Reuse policy:**
    VRAMStatus and GPUVRAMInsufficientError are imported from gpu_vram_gate.py — they
    are NOT redefined here.  check_vram() and kill_zombies() are also inherited from
    GPUVRAMGate to avoid code duplication.

Spec: REQ-INFRA-049, REQ-INFRA-050, REQ-INFRA-051,
      SCENARIO-INFRA-057, SCENARIO-INFRA-058, SCENARIO-INFRA-059
"""

from __future__ import annotations

import logging
import time

from carnot.pipeline.gpu_vram_gate import (
    GPUVRAMGate,
    GPUVRAMInsufficientError,
    VRAMStatus,
)

_log = logging.getLogger(__name__)

__all__ = ["GPUVRAMGateV2"]


class GPUVRAMGateV2(GPUVRAMGate):
    """Kill-first VRAM guard — fixes the RETRO-044 race condition.

    Inherits check_vram(), kill_zombies(), _n_gpus(), and wait_for_vram() from
    GPUVRAMGate.  Overrides __enter__() with the corrected kill-first order.

    **Why kill-first matters:**
        The GPU driver retains a zombie process's VRAM allocation for 5-15 seconds
        after SIGKILL while it drains pending GPU operations.  GPUVRAMGate (V1)
        checked VRAM first and then killed, which meant the subsequent poll saw
        VRAM still held during this drain window and incorrectly deferred the
        experiment.  Four consecutive milestones (RETRO-044) were lost to this
        single ordering mistake.

    Parameters
    ----------
    min_free_gb : float
        Minimum free VRAM required on each GPU.  Default 8.0 GB.
    wait_seconds : int
        Seconds to poll for VRAM recovery after the drain sleep.  Default 60.
    zombie_drain_sleep_seconds : int
        Seconds to sleep after kill_zombies() to allow the GPU driver to flush
        pending operations from killed contexts.  Default 15 s based on RTX 3090
        driver observations (retro .35/.36: drain completes within 10-15 s).
    kill_first : bool
        If True (default): kill_zombies() → sleep(drain) → check_vram().
        If False: old check-first behavior (V1 compatible; for backward-compat tests
        only — do NOT use in new experiments).

    Spec: REQ-INFRA-049, REQ-INFRA-050, REQ-INFRA-051,
          SCENARIO-INFRA-057, SCENARIO-INFRA-058, SCENARIO-INFRA-059
    """

    def __init__(
        self,
        min_free_gb: float = 8.0,
        wait_seconds: int = 60,
        zombie_drain_sleep_seconds: int = 15,
        kill_first: bool = True,
    ) -> None:
        # Pass auto_kill=True so the inherited wait_for_vram() / kill_zombies() work
        # correctly when called from the fallback wait loop.
        super().__init__(min_free_gb=min_free_gb, wait_seconds=wait_seconds, auto_kill=True)
        self.zombie_drain_sleep_seconds = zombie_drain_sleep_seconds
        self.kill_first = kill_first

    def ensure_vram_available(self, gpu_index: int) -> bool:
        """Ensure sufficient VRAM is available on ``gpu_index``.

        **Kill-first path (kill_first=True — recommended):**
          1. kill_zombies(gpu_index) — remove zombie processes unconditionally
          2. sleep(zombie_drain_sleep_seconds) — let driver flush killed contexts
          3. check_vram(gpu_index) — read VRAM AFTER drain window has elapsed
          4. If still insufficient: poll via wait_for_vram() up to wait_seconds

        **Check-first path (kill_first=False — backward compat only):**
          1. check_vram(gpu_index) — read current VRAM first
          2. If insufficient: kill_zombies(gpu_index) — then kill
          3. wait_for_vram(gpu_index) — poll until threshold met or timeout

        Why step 2 in the kill-first path is critical:
            Without the sleep, check_vram fires during the GPU driver's drain window
            (5-15 s) and sees VRAM still allocated to the now-dead process.  This was
            the exact mechanism behind RETRO-044 / Exps 476, 478, 479 deferrals.

        Parameters
        ----------
        gpu_index : int
            Zero-based GPU device index.

        Returns
        -------
        bool
            True when free_gb >= min_free_gb (or no GPU present).
            False when VRAM remains insufficient after the full wait.
        """
        if self.kill_first:
            # --- Kill-first path (REQ-INFRA-049) ---
            n_killed = self.kill_zombies(gpu_index)
            if n_killed > 0 or True:
                # Always sleep: even if no zombies were found by our heuristic,
                # the driver may still be draining contexts from processes that
                # terminated between our pynvml query and now.  The sleep is cheap
                # (15 s) relative to the cost of an incorrect deferral.
                #
                # Why "or True": if n_killed == 0 we still sleep because pynvml's
                # zombie detection heuristic is not perfect — some zombie contexts
                # are not listed as ComputeRunningProcesses but still consume VRAM.
                _log.debug(
                    "GPUVRAMGateV2: sleeping %ds for GPU %d driver drain (killed=%d)",
                    self.zombie_drain_sleep_seconds, gpu_index, n_killed,
                )
                time.sleep(self.zombie_drain_sleep_seconds)

            status = self.check_vram(gpu_index)
            # pynvml unavailable → total_mb==0 → no GPU, gate is no-op
            if status.total_mb == 0:
                return True
            if status.free_gb >= self.min_free_gb:
                _log.info(
                    "GPUVRAMGateV2: GPU %d OK after drain — %.2f GB free (>= %.2f GB)",
                    gpu_index, status.free_gb, self.min_free_gb,
                )
                return True

            # Still insufficient after drain sleep — fall through to timed wait
            _log.warning(
                "GPUVRAMGateV2: GPU %d still %.2f GB free after drain sleep; "
                "entering %ds wait loop",
                gpu_index, status.free_gb, self.wait_seconds,
            )
            return self.wait_for_vram(gpu_index)

        else:
            # --- Check-first path (backward compat / kill_first=False) ---
            # This reproduces the V1 behavior.  Do NOT use in new experiments —
            # this path hits the RETRO-044 race condition during the drain window.
            status = self.check_vram(gpu_index)
            if status.total_mb == 0:
                return True
            if status.free_gb >= self.min_free_gb:
                return True
            self.kill_zombies(gpu_index)
            return self.wait_for_vram(gpu_index)

    def __enter__(self) -> "GPUVRAMGateV2":
        """Run the kill-first VRAM gate for every detected GPU.

        For each GPU: calls ensure_vram_available().  If it returns False,
        raises GPUVRAMInsufficientError.

        On CPU-only machines (n_gpus == 0) the gate is a complete no-op.

        Spec: REQ-INFRA-049, REQ-INFRA-050, SCENARIO-INFRA-057/058/059
        """
        n_gpus = self._n_gpus()
        if n_gpus == 0:
            _log.debug("GPUVRAMGateV2: no GPUs detected — gate is a no-op")
            return self

        for gpu_idx in range(n_gpus):
            if not self.ensure_vram_available(gpu_idx):
                final = self.check_vram(gpu_idx)
                raise GPUVRAMInsufficientError(
                    gpu_index=gpu_idx,
                    free_gb=final.free_gb,
                    min_free_gb=self.min_free_gb,
                )

        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """No cleanup needed — VRAM is managed by the model load/unload lifecycle."""
        pass

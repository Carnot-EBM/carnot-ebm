"""JITVRAMCheck — just-in-time VRAM availability gate for model loading.

**Why JIT instead of planning-time VRAM checks (RETRO-051):**
    Planning-time VRAM forecasts (VRAMBudgetLedger, Exp 500) compute the VRAM budget
    once at script startup, before any model is loaded.  Between startup and the actual
    model.load() call, the VRAM state can change: the conductor may have loaded a new
    model, a background process may have allocated GPU memory, or a prior model in the
    same script may still be holding its allocation.  The planning-time snapshot is
    therefore STALE by the time the load actually happens.

    Exps 502, 503, and 504 all hit runtime CUDA OOM even though the planning-time
    forecast said the model would fit.  The root cause was exactly this staleness.

    JITVRAMCheck fixes this by querying pynvml IMMEDIATELY BEFORE each model.load()
    call — in the same call frame as the load.  This converts silent CUDA OOM into a
    fast-fail diagnostic with a single retry (wait 30 s → check again → abort if still
    insufficient).

**CI stub path:**
    When pynvml is not installed (CI, CPU-only machines), get_available_gb() returns
    24.0 GB — the full RTX 3090 capacity — so CI never blocks on a missing GPU.
    The stub is transparently identified by the JITVRAMResult.available_gb value in
    that environment.

Spec: REQ-INFRA-064, REQ-INFRA-065, REQ-INFRA-066,
      SCENARIO-INFRA-073, SCENARIO-INFRA-074, SCENARIO-INFRA-075
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

_log = logging.getLogger(__name__)

__all__ = ["JITVRAMCheck", "JITVRAMResult"]

# Sentinel value returned by the CI stub when pynvml is not installed.
# 24.0 GB = full RTX 3090 capacity — CI never has a GPU, so we assume
# maximum headroom to avoid false negatives in automated test pipelines.
_CI_STUB_AVAILABLE_GB = 24.0


@dataclass
class JITVRAMResult:
    """Result from a single JIT VRAM gate check.

    Fields
    ------
    device_id : int
        Zero-based GPU device index that was queried.
    model_id : str
        Identifier of the model that was about to be loaded (for logging).
    required_gb : float
        VRAM headroom required before the load is permitted.
    available_gb : float
        Actual free VRAM at the time of the LAST pynvml query (attempt 1 or 2).
    is_cleared : bool
        True if available_gb >= required_gb at the time of the last check.
        The model load is safe to proceed iff is_cleared is True.
    attempts : int
        Number of pynvml queries performed: 1 if the first check passed,
        2 if a retry was needed (regardless of whether the retry succeeded).
    wait_applied : bool
        True iff a sleep(retry_wait_s) was inserted between attempt 1 and 2.
    """

    device_id: int
    model_id: str
    required_gb: float
    available_gb: float
    is_cleared: bool
    attempts: int
    wait_applied: bool = field(default=False)


class JITVRAMCheck:
    """Just-in-time VRAM gate — query GPU free memory immediately before model.load().

    This class is the RETRO-051 fix.  Planning-time VRAM forecasts are stale by
    the time the actual model.load() fires.  JITVRAMCheck queries the GPU driver
    (via pynvml) in the same call frame as the load, giving a real-time reading.

    If the first check fails (available_gb < required_gb), the gate waits
    retry_wait_s seconds (default 30) and checks once more.  One retry is
    sufficient: if another load is in progress, 30 s is typically enough for it
    to finish and release VRAM.  If VRAM is still insufficient after the retry,
    the caller gets is_cleared=False and can abort rather than crash with OOM.

    Parameters
    ----------
    device_id : int
        Zero-based GPU device index to query.  Default 0.

    Spec: REQ-INFRA-064, REQ-INFRA-065, REQ-INFRA-066
    """

    def __init__(self, device_id: int = 0) -> None:
        self.device_id = device_id

    def get_available_gb(self) -> float:
        """Query free VRAM on self.device_id right now via pynvml.

        This is a real-time query — it hits the GPU driver every time it is
        called, never returning a cached value.  That is the whole point of JIT.

        CI stub: when pynvml is not installed, returns _CI_STUB_AVAILABLE_GB
        (24.0 GB) so CI never blocks on a missing GPU.

        Returns
        -------
        float
            Free VRAM in gigabytes at the instant of the call.

        Spec: REQ-INFRA-064, REQ-INFRA-066
        """
        try:
            import pynvml  # noqa: PLC0415 — optional dep

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return info.free / (1024 ** 3)
        except ImportError:
            # CI stub path: no GPU hardware in CI — assume full VRAM available.
            _log.debug(
                "JITVRAMCheck: pynvml not installed — returning CI stub %.1f GB",
                _CI_STUB_AVAILABLE_GB,
            )
            return _CI_STUB_AVAILABLE_GB
        except Exception as exc:
            # Any pynvml error (e.g. driver not loaded) → assume full VRAM to
            # avoid blocking loads on machines where the GPU is simply unreachable.
            _log.warning("JITVRAMCheck.get_available_gb: pynvml error: %s — returning stub", exc)
            return _CI_STUB_AVAILABLE_GB

    def gate_model_load(
        self,
        model_id: str,
        required_gb: float,
        retry_wait_s: float = 30.0,
    ) -> JITVRAMResult:
        """Check VRAM availability immediately before a model.load() call.

        Algorithm:
          1. Query free VRAM right now (attempt 1).
          2. If available_gb >= required_gb → return is_cleared=True, attempts=1.
          3. Otherwise: sleep retry_wait_s seconds (wait_applied=True), query again
             (attempt 2).
          4. Return is_cleared=(available_gb >= required_gb), attempts=2.

        The caller is responsible for aborting the load if is_cleared is False.
        A single retry is the right number: if VRAM has not freed up after 30 s,
        it is unlikely to free up on its own without manual intervention.

        Parameters
        ----------
        model_id : str
            Human-readable model identifier for log messages.
        required_gb : float
            Minimum free VRAM in GB required before the load proceeds.
        retry_wait_s : float
            Seconds to sleep between attempt 1 and attempt 2.  Default 30.

        Returns
        -------
        JITVRAMResult
            Full result including is_cleared, available_gb, attempts, wait_applied.

        Spec: REQ-INFRA-064, REQ-INFRA-065
        """
        # Attempt 1 — immediate check
        available = self.get_available_gb()
        _log.info(
            "JITVRAMCheck[attempt=1]: model=%r device=%d available=%.2f GB required=%.2f GB",
            model_id, self.device_id, available, required_gb,
        )

        if available >= required_gb:
            return JITVRAMResult(
                device_id=self.device_id,
                model_id=model_id,
                required_gb=required_gb,
                available_gb=available,
                is_cleared=True,
                attempts=1,
                wait_applied=False,
            )

        # Attempt 1 failed — wait and retry once
        _log.warning(
            "JITVRAMCheck[attempt=1]: INSUFFICIENT — %.2f GB free, need %.2f GB; "
            "waiting %.0f s before retry",
            available, required_gb, retry_wait_s,
        )
        time.sleep(retry_wait_s)

        # Attempt 2 — post-wait check
        available = self.get_available_gb()
        _log.info(
            "JITVRAMCheck[attempt=2]: model=%r device=%d available=%.2f GB required=%.2f GB",
            model_id, self.device_id, available, required_gb,
        )
        is_cleared = available >= required_gb
        if not is_cleared:
            _log.warning(
                "JITVRAMCheck[attempt=2]: STILL INSUFFICIENT — %.2f GB free, need %.2f GB; "
                "model load will be aborted",
                available, required_gb,
            )

        return JITVRAMResult(
            device_id=self.device_id,
            model_id=model_id,
            required_gb=required_gb,
            available_gb=available,
            is_cleared=is_cleared,
            attempts=2,
            wait_applied=True,
        )

    def sequential_load_gate(
        self,
        model_specs: List[Dict[str, Any]],
    ) -> List[JITVRAMResult]:
        """Gate a list of model loads sequentially.

        For each entry in model_specs, calls gate_model_load(model_id, required_gb).
        Each check happens immediately before that model's conceptual load, so later
        checks reflect VRAM state AFTER earlier models have been loaded.

        Parameters
        ----------
        model_specs : list of dict
            Each dict must have keys:
              - 'model_id' (str): human-readable model name
              - 'required_gb' (float): VRAM headroom needed before the load

        Returns
        -------
        list of JITVRAMResult
            One result per model spec, in input order.

        Spec: REQ-INFRA-064
        """
        results: List[JITVRAMResult] = []
        for spec in model_specs:
            result = self.gate_model_load(
                model_id=spec["model_id"],
                required_gb=spec["required_gb"],
            )
            results.append(result)
        return results

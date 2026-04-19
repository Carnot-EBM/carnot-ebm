"""GPUThermalGate — block experiment start when the GPU is thermally throttling.

**Why this module exists (RETRO-046, three consecutive milestones open):**
    GPU thermal throttling silently reduces clock speed when the die temperature
    exceeds the throttle onset temperature.  On an RTX 3090 this starts at 83-85°C
    and can drop the effective clock from ~1900 MHz to ~900-1200 MHz — a 30-50%
    throughput reduction.  Critically, the conductor cannot observe this: the GPU
    is running, the model is loaded, inference returns results — but the benchmark
    times are 20-40% slower than they would be at normal operating temperature.

    Without the thermal gate, experiments that run while the GPU is hot produce
    artificially slow benchmark numbers.  The gate enforces a simple invariant:
    every benchmark run starts with the GPU at or below the cool threshold.

**Why the hot_threshold_c=85.0 default:**
    RTX 3090 thermal limit is 93°C; NVIDIA's internal throttle curve begins at
    83-85°C.  Using 85°C as the trigger gives 3°C headroom before the actual
    throttle cutoff, catching the onset before clock reduction becomes significant.

**Why the cool_threshold_c=80.0 default:**
    We wait until 80°C — not 85°C — before allowing the experiment to proceed.
    5°C below the trigger gives time for the temperature to stabilise; if we
    resumed at 84.9°C the GPU could re-hit threshold within the first few
    inference batches.

**Why exponential backoff in wait_for_cool():**
    GPU cooling is not linear.  After a hot workload, the first 5°C drop typically
    happens in 15-30 seconds as the heat-pipe picks up the spike; the next 5°C
    takes longer as the heatsink reaches a new equilibrium.  Exponential backoff
    (starting at backoff_base_seconds, doubling each time) avoids hammering pynvml
    while the GPU is still hot, yet catches the inflection point where cooling
    accelerates.  A 15-second base covers the initial spike; subsequent doublings
    (30s, 60s, 120s) span the longer stabilisation phase.

Spec: REQ-INFRA-054, REQ-INFRA-055, REQ-INFRA-056,
      SCENARIO-INFRA-062, SCENARIO-INFRA-063, SCENARIO-INFRA-064
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

_log = logging.getLogger(__name__)

__all__ = [
    "GPUThermalGate",
    "GPUThermalThrottleError",
    "ThermalStatus",
]


@dataclass
class ThermalStatus:
    """Snapshot of one GPU's thermal state at a single point in time.

    **Why a dataclass (not a plain dict):**
        Structured access makes test assertions readable and prevents the
        'dict typo' failure mode where a misspelled key returns None silently.

    Attributes
    ----------
    gpu_index : int
        Zero-based device index queried via pynvml.
    temperature_c : float | None
        Current die temperature in Celsius.  ``None`` when pynvml is unavailable
        or the machine has no NVIDIA GPU — in that case the gate is a no-op.
    is_safe : bool
        ``True`` when the temperature is below the gate's hot threshold (or when
        ``temperature_c`` is ``None`` — no GPU means no thermal risk).

    Computed property
    -----------------
    is_throttling : bool
        ``True`` when ``temperature_c`` is not ``None`` and exceeds 85°C.  This
        is a read-only property so it stays in sync with ``temperature_c`` even
        if the dataclass is mutated in tests.

    Spec: REQ-INFRA-054
    """

    gpu_index: int
    temperature_c: float | None
    is_safe: bool

    @property
    def is_throttling(self) -> bool:
        """Return True when the GPU is at or above the thermal throttle onset.

        Why 85°C: RTX 3090 throttle onset is 83-85°C.  Using temperature_c > 85
        (strictly greater than) means we treat exactly 85°C as throttling, which
        is conservative but correct — 85°C is already in the throttle zone on
        most RTX 3090 units.
        """
        if self.temperature_c is None:
            return False
        return self.temperature_c > 85.0


class GPUThermalThrottleError(Exception):
    """Raised when the GPU temperature does not drop below cool_threshold_c
    within max_wait_seconds.

    **Why raise instead of warn:**
        A warning would allow the experiment to proceed with a thermally
        throttled GPU, producing benchmark results that are 20-40% slower than
        they should be.  Raising forces the conductor to defer the experiment
        (honest_verdict='gpu_thermal_throttle') and retry later when the GPU
        has cooled.

    Spec: REQ-INFRA-055, SCENARIO-INFRA-064
    """

    def __init__(self, gpu_index: int, temperature_c: float | None, max_wait_seconds: int) -> None:
        self.gpu_index = gpu_index
        self.temperature_c = temperature_c
        self.max_wait_seconds = max_wait_seconds
        super().__init__(
            f"GPU {gpu_index} temperature ({temperature_c}°C) did not drop below cool "
            f"threshold within {max_wait_seconds}s — deferring experiment "
            f"(honest_verdict='gpu_thermal_throttle')"
        )


class GPUThermalGate:
    """Block experiment start when the GPU is thermally throttling.

    On CPU-only machines (pynvml unavailable or no NVIDIA GPU) this class is a
    complete no-op — every method returns safe defaults.  This ensures all
    CPU-only experiments (CI, local dev) run without modification.

    **Usage as a context manager (preferred):**

        with GPUThermalGate():
            # GPU is guaranteed to be below cool_threshold_c here
            load_model_and_run_experiment()
        # GPUThermalThrottleError raised if GPU stayed hot for > max_wait_seconds

    **Usage via wait_for_cool() directly (for setup_gpu integration):**

        gate = GPUThermalGate()
        status = gate.check_temperature(gpu_index=0)
        if not gate.wait_for_cool(gpu_index=0):
            raise GPUThermalThrottleError(0, status.temperature_c, gate.max_wait_seconds)

    Parameters
    ----------
    hot_threshold_c : float
        Temperature above which the gate triggers.  Default 85°C (RTX 3090
        throttle onset; see module docstring for rationale).
    cool_threshold_c : float
        Temperature the GPU must reach before the experiment is allowed to
        proceed.  Default 80°C (5°C below trigger for thermal stability margin).
    max_wait_seconds : int
        Maximum total wall-clock seconds to wait for the GPU to cool.  Default
        300 s (5 minutes) — if the GPU cannot cool in 5 minutes, either the
        workload is still running or the cooling system is inadequate.
    backoff_base_seconds : float
        Initial sleep duration for the exponential backoff loop.  Default 15 s.
        Subsequent waits are 30 s, 60 s, 120 s, up to the remaining budget.

    Spec: REQ-INFRA-054, REQ-INFRA-055, SCENARIO-INFRA-062, SCENARIO-INFRA-063,
          SCENARIO-INFRA-064
    """

    def __init__(
        self,
        hot_threshold_c: float = 85.0,
        cool_threshold_c: float = 80.0,
        max_wait_seconds: int = 300,
        backoff_base_seconds: float = 15.0,
    ) -> None:
        self.hot_threshold_c = hot_threshold_c
        self.cool_threshold_c = cool_threshold_c
        self.max_wait_seconds = max_wait_seconds
        self.backoff_base_seconds = backoff_base_seconds

    # ------------------------------------------------------------------
    # check_temperature()
    # ------------------------------------------------------------------

    def check_temperature(self, gpu_index: int) -> ThermalStatus:
        """Return the current thermal status for *gpu_index*.

        Queries pynvml for the GPU die temperature.  When pynvml is unavailable
        (CPU-only machine, no NVIDIA driver, or pynvml not installed) this method
        returns ``ThermalStatus(gpu_index, temperature_c=None, is_safe=True)`` so
        the gate is a transparent no-op on non-NVIDIA hardware.

        **Why pynvml (not subprocess nvidia-smi):**
            pynvml binds directly to the NVML C library — one DLL/SO call,
            no process fork, no shell parsing.  nvidia-smi spawns a new process
            for each call, which adds 100-300 ms latency and is fragile on
            systems where the PATH does not include the CUDA bin directory.

        Parameters
        ----------
        gpu_index : int
            Zero-based NVIDIA device index.

        Returns
        -------
        ThermalStatus
            temperature_c=None when pynvml is unavailable or no GPU present.

        Spec: REQ-INFRA-054
        """
        try:
            import pynvml  # noqa: PLC0415

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            temp = float(pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            ))
            is_safe = temp <= self.hot_threshold_c
            _log.debug(
                "GPUThermalGate: GPU %d temperature = %.1f°C (safe=%s, threshold=%.1f°C)",
                gpu_index, temp, is_safe, self.hot_threshold_c,
            )
            return ThermalStatus(gpu_index=gpu_index, temperature_c=temp, is_safe=is_safe)
        except Exception as exc:
            # pynvml unavailable, NVML init failed, or invalid GPU index —
            # treat as no GPU present; gate is a no-op.
            _log.debug(
                "GPUThermalGate: pynvml unavailable or GPU %d not found (%s) — no-op",
                gpu_index, exc,
            )
            return ThermalStatus(gpu_index=gpu_index, temperature_c=None, is_safe=True)

    # ------------------------------------------------------------------
    # wait_for_cool()
    # ------------------------------------------------------------------

    def wait_for_cool(self, gpu_index: int) -> bool:
        """Block with exponential backoff until GPU *gpu_index* is below cool_threshold_c.

        **Fast path:** if the current temperature is already below
        ``cool_threshold_c``, returns ``True`` immediately with no sleep.

        **No GPU / pynvml unavailable:** returns ``True`` immediately — the gate
        is a no-op on CPU-only machines.

        **Exponential backoff rationale:**
            GPU cooling is not linear — the first 5°C drop happens quickly as
            heat-pipe pickups the spike, subsequent degrees take longer as the
            heatsink reaches equilibrium.  Backoff starts at ``backoff_base_seconds``
            (15 s) and doubles: 15 → 30 → 60 → 120 s.  This avoids unnecessary
            polling at the long-tail part of the cooling curve while still
            catching the early inflection point.

        Parameters
        ----------
        gpu_index : int
            Zero-based NVIDIA device index.

        Returns
        -------
        bool
            ``True`` if the GPU cooled within ``max_wait_seconds``.
            ``False`` if ``max_wait_seconds`` elapsed before the temperature
            dropped to ``cool_threshold_c``.

        Spec: REQ-INFRA-055, SCENARIO-INFRA-062, SCENARIO-INFRA-063
        """
        # Fast path: check once before entering the wait loop.
        status = self.check_temperature(gpu_index)
        if status.temperature_c is None:
            # No GPU / pynvml unavailable — gate is a no-op.
            return True
        if status.temperature_c <= self.cool_threshold_c:
            _log.debug(
                "GPUThermalGate: GPU %d already cool (%.1f°C <= %.1f°C)",
                gpu_index, status.temperature_c, self.cool_threshold_c,
            )
            return True

        _log.warning(
            "GPUThermalGate: GPU %d is hot (%.1f°C > %.1f°C threshold); "
            "waiting up to %ds for it to cool to %.1f°C",
            gpu_index, status.temperature_c, self.hot_threshold_c,
            self.max_wait_seconds, self.cool_threshold_c,
        )

        elapsed = 0.0
        backoff = self.backoff_base_seconds

        while elapsed < self.max_wait_seconds:
            sleep_time = min(backoff, self.max_wait_seconds - elapsed)
            _log.debug(
                "GPUThermalGate: sleeping %.0fs (elapsed=%.0fs / %ds)",
                sleep_time, elapsed, self.max_wait_seconds,
            )
            time.sleep(sleep_time)
            elapsed += sleep_time

            status = self.check_temperature(gpu_index)
            if status.temperature_c is None:
                return True  # GPU disappeared — treat as no-op
            if status.temperature_c <= self.cool_threshold_c:
                _log.info(
                    "GPUThermalGate: GPU %d cooled to %.1f°C after %.0fs",
                    gpu_index, status.temperature_c, elapsed,
                )
                return True

            _log.debug(
                "GPUThermalGate: GPU %d still hot (%.1f°C), continuing wait",
                gpu_index, status.temperature_c,
            )
            backoff *= 2.0  # exponential backoff

        # Timed out.
        final = self.check_temperature(gpu_index)
        _log.error(
            "GPUThermalGate: GPU %d failed to cool within %ds "
            "(final temperature: %s°C)",
            gpu_index, self.max_wait_seconds,
            f"{final.temperature_c:.1f}" if final.temperature_c is not None else "N/A",
        )
        return False

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "GPUThermalGate":
        """Run the thermal gate for GPU 0 (single-GPU default).

        For multi-GPU experiments, call wait_for_cool(gpu_index) directly.

        Spec: REQ-INFRA-056, SCENARIO-INFRA-062, SCENARIO-INFRA-064
        """
        status = self.check_temperature(0)
        if not self.wait_for_cool(0):
            raise GPUThermalThrottleError(
                gpu_index=0,
                temperature_c=status.temperature_c,
                max_wait_seconds=self.max_wait_seconds,
            )
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """No cleanup needed — thermal state is managed by the GPU driver."""
        pass

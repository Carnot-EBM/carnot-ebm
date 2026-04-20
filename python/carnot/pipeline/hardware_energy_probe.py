"""HardwareEnergyProbe — read CPU hardware energy via RAPL or fallback to mock.

**Researcher summary:**
    arXiv 2603.20224 shows that hardware-measured energy-per-token (joules/token via
    RAPL or nvml) correlates with reasoning difficulty: hard steps cost 3-8x more
    hardware energy than easy steps.  Carnot's EORM energy scores are learned proxies
    for reasoning quality.  If hardware energy and EORM energy co-move (r > 0.5), then
    a power meter becomes a FREE, label-free calibration signal — no human annotation
    required, just a Linux CPU.

**What RAPL is (for engineers):**
    RAPL = Running Average Power Limit.  Intel (and AMD Zen) CPUs expose cumulative
    energy counters via the Linux powercap driver.  Reading
    /sys/class/powercap/intel-rapl:0/energy_uj gives the total joules (in
    microjoules) consumed by the CPU package since boot.  Taking two readings around
    a computation yields the energy delta for that computation.

**Fallback chain:**
    1. Direct sysfs read — works on bare-metal Linux with powercap enabled.
    2. pyRAPL library — same underlying counter, nicer API, optional dependency.
    3. Mock (0.0) — used in CI / containers where powercap is inaccessible.
       source='mock' in the reading flags this case to callers.

**Correlation formula:**
    Pearson r is the standard linear correlation coefficient.  r > 0.5 and
    p < 0.05 indicates a statistically meaningful positive relationship between
    hardware energy and EORM energy over the measured steps — enough to use
    hardware readings as a calibration signal.

Spec: REQ-LEARN-064,
      SCENARIO-LEARN-098, SCENARIO-LEARN-099, SCENARIO-LEARN-100
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar

from scipy.stats import pearsonr  # type: ignore[import]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RAPL_ENERGY_PATH = Path("/sys/class/powercap/intel-rapl:0/energy_uj")
_MICROJOULES_PER_JOULE = 1_000_000.0

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class HardwareEnergyReading:
    """A single hardware energy sample.

    Attributes:
        timestamp_ns: Monotonic clock reading at sample time, in nanoseconds.
            Use this to order readings; do not convert to wall-clock time.
        joules: Energy value in joules at sample time.  Subtract two consecutive
            readings to get the delta for the work done between them.
        source: Where the energy value came from.
            'rapl'  — direct sysfs read (/sys/class/powercap/intel-rapl:0/energy_uj)
            'pyrapl' — via the optional pyRAPL Python library
            'mock'  — placeholder used when hardware counters are unavailable
                      (CI, containers without powercap).  joules will be 0.0.
    """

    timestamp_ns: int
    joules: float
    source: str  # 'rapl' | 'pyrapl' | 'mock'


@dataclass
class EORMHardwareCorrelation:
    """Pearson correlation result between hardware energy and EORM scores.

    Attributes:
        n_steps: Number of CoT steps measured.
        pearson_r: Pearson correlation coefficient in [-1, 1].
            Positive means hardware energy rises when EORM energy rises (hard steps
            cost more in both physical and learned energy).
        p_value: Two-tailed p-value for the null hypothesis r=0.
            Values below 0.05 indicate the correlation is statistically significant.
        hardware_energies: Per-step hardware energy deltas in joules, length n_steps.
        eorm_energies: Per-step EORM energy scores (lower = better), length n_steps.
        calibration_viable: True when r > 0.5 AND p_value < 0.05, meaning hardware
            energy is a usable label-free calibration signal for EORM training.
    """

    n_steps: int
    pearson_r: float
    p_value: float
    hardware_energies: list[float] = field(default_factory=list)
    eorm_energies: list[float] = field(default_factory=list)
    calibration_viable: bool = False


# ---------------------------------------------------------------------------
# HardwareEnergyProbe
# ---------------------------------------------------------------------------

class HardwareEnergyProbe:
    """Read CPU hardware energy counters with graceful fallback.

    **For engineers:**
        Instantiate once at experiment start.  Call read() to get a snapshot.
        Call measure_segment(fn) to time a callable and measure its energy cost.
        The probe auto-detects which energy source is available and sticks to it
        throughout the experiment.

    **Thread safety:** Not thread-safe — use one instance per thread if needed.
    """

    def __init__(self) -> None:
        # Detect source at construction time so every read() call is consistent.
        self._source: str = self._detect_source()

    # ------------------------------------------------------------------
    # Source detection
    # ------------------------------------------------------------------

    def _detect_source(self) -> str:
        """Probe which energy source is usable; return the source label."""
        if self._try_rapl() is not None:
            return "rapl"
        if self._try_pyrapl() is not None:
            return "pyrapl"
        return "mock"

    # ------------------------------------------------------------------
    # Backend readers
    # ------------------------------------------------------------------

    def _try_rapl(self) -> float | None:
        """Read energy_uj from the Linux RAPL sysfs interface.

        The file is a plain integer in microjoules, updated by the kernel every
        ~1ms.  Returns joules (divided by 1e6) or None if the file is absent or
        unreadable (CI, non-Linux, or powercap module not loaded).
        """
        try:
            raw = _RAPL_ENERGY_PATH.read_text().strip()
            return int(raw) / _MICROJOULES_PER_JOULE
        except (FileNotFoundError, PermissionError, ValueError, OSError):
            return None

    def _try_pyrapl(self) -> float | None:
        """Read package energy via the optional pyRAPL library.

        pyRAPL wraps the same RAPL sysfs counters with a friendlier API.
        Returns joules or None if the library is not installed or fails.

        Why attempt this after direct sysfs?  Some systems restrict direct sysfs
        reads but allow pyRAPL via a capability-enabled wrapper.
        """
        try:
            import pyRAPL  # type: ignore[import]  # optional dependency

            pyRAPL.setup()
            meter = pyRAPL.Measurement("probe")
            meter.begin()
            meter.end()
            # energy is in microjoules per CPU package
            pkg = meter.result.pkg
            if pkg is not None and len(pkg) > 0:
                return pkg[0] / _MICROJOULES_PER_JOULE
        except Exception:  # noqa: BLE001
            pass
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def source(self) -> str:
        """The active energy source: 'rapl', 'pyrapl', or 'mock'."""
        return self._source

    def read(self) -> HardwareEnergyReading:
        """Take a single hardware energy snapshot.

        Returns:
            HardwareEnergyReading with the current joules and source label.
            When source='mock', joules is always 0.0 — subtract two mock readings
            and the delta is 0.0, which will produce r=NaN or 0 in the correlation.

        Spec: SCENARIO-LEARN-098
        """
        ts = time.monotonic_ns()
        if self._source == "rapl":
            joules = self._try_rapl() or 0.0
        elif self._source == "pyrapl":
            joules = self._try_pyrapl() or 0.0
        else:
            joules = 0.0
        return HardwareEnergyReading(timestamp_ns=ts, joules=joules, source=self._source)

    def measure_segment(self, fn: Callable[[], T]) -> tuple[T, float]:
        """Run fn() and measure the hardware energy consumed.

        Takes a reading before and after calling fn(), then returns fn()'s return
        value plus the energy delta in joules.

        Args:
            fn: A zero-argument callable to measure.

        Returns:
            (result, delta_joules) where result = fn() and delta_joules is the
            hardware energy consumed.  delta_joules == 0.0 when source='mock'.

        Spec: SCENARIO-LEARN-099
        """
        before = self.read()
        result = fn()
        after = self.read()
        delta = after.joules - before.joules
        # RAPL counters wrap at ~65 536 J on most chips; protect against wrap-around.
        if delta < 0:
            delta = 0.0
        return result, delta


# ---------------------------------------------------------------------------
# Correlation computation
# ---------------------------------------------------------------------------

def compute_eorm_hardware_correlation(
    probe: HardwareEnergyProbe,
    eorm_model: Any,
    cot_steps: list[str],
) -> EORMHardwareCorrelation:
    """Measure Pearson correlation between hardware energy and EORM scores per step.

    **For engineers:**
        For each CoT step text:
          1. Wrap the EORM scoring call in measure_segment() to capture hardware energy.
          2. Record the EORM energy scalar returned by eorm_model.energy().
        After all steps, compute Pearson r and p-value.  If r > 0.5 and p < 0.05,
        hardware energy is a viable free calibration signal for EORM training.

    The eorm_model is expected to expose an .energy(CoTEnergyInput) -> float method
    (the EORMModel class from carnot.models.eorm).  We score each step as both the
    question and response to avoid needing a separate question corpus — for calibration
    purposes we only need relative ordering, not absolute accuracy.

    Args:
        probe: HardwareEnergyProbe instance (may be mock source on CI).
        eorm_model: An EORMModel instance with an .energy(CoTEnergyInput) method.
        cot_steps: List of step text strings to measure.

    Returns:
        EORMHardwareCorrelation with per-step measurements and the Pearson result.

    Spec: REQ-LEARN-064, SCENARIO-LEARN-100
    """
    from carnot.models.eorm import CoTEnergyInput  # local import avoids circular deps

    hw_energies: list[float] = []
    eorm_energies: list[float] = []

    for step_text in cot_steps:
        cot_input = CoTEnergyInput(question_text=step_text, response_text=step_text)

        def _score(ci: "CoTEnergyInput" = cot_input) -> float:
            return eorm_model.energy(ci)

        eorm_score, hw_delta = probe.measure_segment(_score)
        hw_energies.append(hw_delta)
        eorm_energies.append(eorm_score)

    n = len(cot_steps)
    # scipy.stats.pearsonr needs at least 2 points and non-constant arrays
    try:
        r_val, p_val = pearsonr(hw_energies, eorm_energies)
        pearson_r = float(r_val)
        p_value = float(p_val)
    except Exception:  # noqa: BLE001
        # Happens with mock (all hw_energies == 0.0, constant array)
        pearson_r = 0.0
        p_value = 1.0

    # Handle NaN from constant input (mock source always returns 0.0 delta)
    import math
    if math.isnan(pearson_r):
        pearson_r = 0.0
    if math.isnan(p_value):
        p_value = 1.0

    calibration_viable = pearson_r > 0.5 and p_value < 0.05

    return EORMHardwareCorrelation(
        n_steps=n,
        pearson_r=pearson_r,
        p_value=p_value,
        hardware_energies=hw_energies,
        eorm_energies=eorm_energies,
        calibration_viable=calibration_viable,
    )

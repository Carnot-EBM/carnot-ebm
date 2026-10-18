"""THRML/Extropic sampler backend stub.

**Researcher summary:**
    `ThrmlSamplerBackend` reserves Carnot's SamplerBackend integration point for
    Extropic Z1/XTR-0 access. Until an Extropic TSU device and SDK are available,
    it runs the existing CPU Gibbs backend when `CARNOT_TSU_DEVICE` is unset and
    raises an explicit `NotImplementedError` when hardware is requested.

**Detailed explanation for engineers:**
    This class deliberately does not fake Z1/XTR-0 hardware behavior. The CPU
    fallback keeps Carnot experiments runnable through the same interface, while
    the hardware branch fails loudly so benchmark artifacts cannot confuse a
    host-side simulation with live thermodynamic hardware.

Spec: REQ-SAMPLE-040, SCENARIO-SAMPLE-066, SCENARIO-SAMPLE-067
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from carnot.samplers.backend import CpuBackend, SamplerBackend


@dataclass
class ThrmlSamplerBackend(SamplerBackend):
    """SamplerBackend adapter for THRML CPU fallback and future Extropic TSU access.

    Attributes:
        seed: Random seed forwarded to the CPU Gibbs fallback. The future hardware
            implementation may use this only for host-side initialization because
            the TSU stochastic process is hardware-native.

    Spec: REQ-SAMPLE-040
    """

    seed: int = 42
    _cpu_backend: CpuBackend = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._cpu_backend = CpuBackend(seed=self.seed)

    @property
    def requested_device(self) -> str | None:
        """Return the requested TSU device label, if any."""
        return os.environ.get("CARNOT_TSU_DEVICE")

    @property
    def using_hardware(self) -> bool:
        """Whether calls should route to the future Extropic hardware path."""
        return self.requested_device is not None

    @property
    def backend_name(self) -> str:
        """Report the active execution path honestly."""
        if self.requested_device:
            return f"thrml_hardware:{self.requested_device}"
        return "thrml_cpu_fallback"

    def _raise_hardware_not_implemented(self, method: str) -> None:
        device = self.requested_device or "<unset>"
        raise NotImplementedError(
            "Extropic TSU hardware path requested via "
            f"CARNOT_TSU_DEVICE={device!r}, but Carnot does not yet have the "
            f"Extropic SDK/device driver integration required for {method}()."
        )

    def minimize_energy(
        self,
        biases: np.ndarray,
        couplings: np.ndarray,
        n_samples: int,
        n_steps: int,
        beta: float,
    ) -> np.ndarray:
        """Run CPU fallback annealing unless a real TSU device was requested.

        Spec: REQ-SAMPLE-040, SCENARIO-SAMPLE-066, SCENARIO-SAMPLE-067
        """
        if self.using_hardware:
            self._raise_hardware_not_implemented("minimize_energy")
        return self._cpu_backend.minimize_energy(biases, couplings, n_samples, n_steps, beta)

    def sample(
        self,
        biases: np.ndarray,
        couplings: np.ndarray,
        n_samples: int,
        config: dict[str, Any],
    ) -> np.ndarray:
        """Run CPU fallback fixed-temperature sampling unless TSU hardware is requested.

        Spec: REQ-SAMPLE-040, SCENARIO-SAMPLE-066, SCENARIO-SAMPLE-067
        """
        if self.using_hardware:
            self._raise_hardware_not_implemented("sample")
        return self._cpu_backend.sample(biases, couplings, n_samples, config)

    def sample_multi_period(
        self,
        biases: np.ndarray,
        couplings: np.ndarray,
        turnover_penalty: float,
        n_samples: int,
        config: dict[str, Any],
    ) -> np.ndarray:
        """Run multi-period fixed-temperature sampling with turnover constraints.

        Spec: REQ-SAMPLE-1834
        """
        if self.using_hardware:
            self._raise_hardware_not_implemented("sample_multi_period")
        
        n_periods, n_spins = biases.shape
        flat_biases = np.zeros(n_periods * n_spins, dtype=biases.dtype)
        flat_couplings = np.zeros((n_periods * n_spins, n_periods * n_spins), dtype=couplings.dtype)
        
        for t in range(n_periods):
            start = t * n_spins
            end = start + n_spins
            flat_biases[start:end] = biases[t]
            flat_couplings[start:end, start:end] = couplings[t]
            
            if t < n_periods - 1:
                for i in range(n_spins):
                    idx_t = start + i
                    idx_next = start + n_spins + i
                    flat_biases[idx_t] -= turnover_penalty
                    flat_biases[idx_next] -= turnover_penalty
                    flat_couplings[idx_t, idx_next] += turnover_penalty
                    flat_couplings[idx_next, idx_t] += turnover_penalty

        flat_samples = self._cpu_backend.sample(flat_biases, flat_couplings, n_samples, config)
        return flat_samples.reshape(n_samples, n_periods, n_spins)

    def minimize_energy_multi_period(
        self,
        biases: np.ndarray,
        couplings: np.ndarray,
        turnover_penalty: float,
        n_samples: int,
        n_steps: int,
        beta: float,
    ) -> np.ndarray:
        """Run multi-period annealing with turnover constraints.

        Spec: REQ-SAMPLE-1834
        """
        if self.using_hardware:
            self._raise_hardware_not_implemented("minimize_energy_multi_period")
            
        n_periods, n_spins = biases.shape
        flat_biases = np.zeros(n_periods * n_spins, dtype=biases.dtype)
        flat_couplings = np.zeros((n_periods * n_spins, n_periods * n_spins), dtype=couplings.dtype)
        
        for t in range(n_periods):
            start = t * n_spins
            end = start + n_spins
            flat_biases[start:end] = biases[t]
            flat_couplings[start:end, start:end] = couplings[t]
            
            if t < n_periods - 1:
                for i in range(n_spins):
                    idx_t = start + i
                    idx_next = start + n_spins + i
                    flat_biases[idx_t] -= turnover_penalty
                    flat_biases[idx_next] -= turnover_penalty
                    flat_couplings[idx_t, idx_next] += turnover_penalty
                    flat_couplings[idx_next, idx_t] += turnover_penalty

        flat_samples = self._cpu_backend.minimize_energy(flat_biases, flat_couplings, n_samples, n_steps, beta)
        return flat_samples.reshape(n_samples, n_periods, n_spins)

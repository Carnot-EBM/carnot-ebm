"""Regime-conditional Phase 4 sampler for Exp 1156.

**Researcher summary:**
    Exp 1155 classified Carnot's current k=5 verifier bridge as Regime C with a
    discrete-component bottleneck. This module therefore implements the deployed
    Regime C path: blocked Gibbs updates for symbolic/discrete verifier
    coordinates and Langevin updates for continuous latent coordinates.

**Detailed explanation for engineers:**
    Direct HMC assumes a smooth differentiable energy landscape. Exp 1155 found
    that AST and Z3-style verifier components break that assumption, so Phase 4
    uses a mixed sampler instead of forcing NUTS onto a discontinuous target. The
    discrete block is sampled exactly as a two-state Gibbs conditional, while the
    continuous block uses overdamped Langevin dynamics inside the bounded latent
    cube.

Spec: REQ-KONA-010, SCENARIO-KONA-009
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from carnot.phase3.continuous_ebm import ContinuousEBM

try:
    from carnot.samplers.backend import SamplerBackend
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by system python without JAX
    if exc.name != "jax":
        raise

    class SamplerBackend(Protocol):  # type: ignore[no-redef]
        """Minimal fallback so Exp1156 can run in a NumPy-only Python."""

        @property
        def backend_name(self) -> str: ...

        def minimize_energy(
            self,
            biases: np.ndarray,
            couplings: np.ndarray,
            n_samples: int,
            n_steps: int,
            beta: float,
        ) -> np.ndarray: ...

        def sample(
            self, energy_fn: Any, init_state: Any, n_steps: int, **kwargs: Any
        ) -> np.ndarray: ...


Phase4Algorithm = Literal["numpyro_nuts", "preconditioned_hmc", "blocked_gibbs", "sgld"]

_SUPPORTED_ALGORITHMS: frozenset[str] = frozenset(
    {"numpyro_nuts", "preconditioned_hmc", "blocked_gibbs", "sgld"}
)
_RECOMMENDATION_TO_ALGORITHM: dict[str, Phase4Algorithm] = {
    "hmc": "numpyro_nuts",
    "preconditioned_hmc": "preconditioned_hmc",
    "blocked_gibbs": "blocked_gibbs",
    "langevin": "sgld",
}


def sampler_algorithm_from_exp1155(artifact: dict[str, Any]) -> Phase4Algorithm:
    """Map the Exp 1155 sampler recommendation to the Exp 1156 artifact enum.

    **Why this mapping exists:**
        Exp 1155 reports recommendations in diagnostic language (`hmc`,
        `langevin`). Exp 1156 reports the concrete algorithm deployed by the
        sampler module (`numpyro_nuts`, `sgld`). Keeping the mapping in one
        function avoids quiet schema drift between the two experiment artifacts.
    """
    recommendation = str(artifact.get("recommended_sampler", ""))
    try:
        return _RECOMMENDATION_TO_ALGORITHM[recommendation]
    except KeyError as exc:
        raise ValueError(f"Unknown Exp 1155 sampler recommendation {recommendation!r}") from exc


@dataclass(frozen=True)
class ContinuousEBMEnergy:
    """Callable energy adapter for `ContinuousEBM`.

    **Why this wrapper is needed:**
        `ContinuousEBM` is a parameter container rather than a full
        `EnergyFunction`. The sampler benefits from an analytic gradient, so
        this adapter provides both `energy(x)` and `grad_energy(x)` without
        modifying the older Phase 3 dataclass.
    """

    model: ContinuousEBM

    def energy(self, x: np.ndarray) -> float:
        """Return E(x) = -0.5 * x^T J x - h^T x for one latent vector."""
        x_arr = np.asarray(x, dtype=np.float64)
        return float(-0.5 * x_arr @ self.model.coupling @ x_arr - self.model.bias @ x_arr)

    def grad_energy(self, x: np.ndarray) -> np.ndarray:
        """Return the analytic gradient dE/dx = -Jx - h."""
        x_arr = np.asarray(x, dtype=np.float64)
        return np.asarray(-self.model.coupling @ x_arr - self.model.bias, dtype=np.float64)

    def __call__(self, x: np.ndarray) -> float:
        """Allow the adapter to be passed wherever a plain callable is expected."""
        return self.energy(x)


def continuous_ebm_energy(model: ContinuousEBM) -> ContinuousEBMEnergy:
    """Build a sampler-compatible energy function from a `ContinuousEBM`."""
    return ContinuousEBMEnergy(model)


@dataclass
class Phase4Sampler(SamplerBackend):
    """SamplerBackend-compatible Phase 4 sampler.

    **Researcher summary:**
        Selects the regime-appropriate sampler from Exp 1155. For the current
        artifact this means `blocked_gibbs`: discrete AST/Z3 coordinates use
        Gibbs conditionals, and all other latent coordinates use Langevin
        dynamics.

    **Detailed explanation for engineers:**
        This class intentionally keeps the public `sample(energy_fn, init_state,
        n_steps, **kwargs)` shape requested by Exp 1156 while also providing the
        `minimize_energy` method required by Carnot's `SamplerBackend` protocol.
        A compatibility path accepts the older Ising backend call shape when a
        `config` dictionary is supplied.

    Spec: REQ-KONA-010
    """

    algorithm: str = "blocked_gibbs"
    seed: int = 1156
    step_size: float = 0.02
    temperature: float = 1.0
    discrete_indices: tuple[int, ...] = ()
    continuous_indices: tuple[int, ...] = ()
    hmc_regime_used: str = "C"
    finite_difference_eps: float = 1e-5
    max_grad_norm: float = 25.0
    _rng: np.random.Generator = field(init=False, repr=False)
    last_diagnostics: dict[str, float | None] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.algorithm not in _SUPPORTED_ALGORITHMS:
            raise ValueError(f"Unsupported Phase 4 sampler algorithm {self.algorithm!r}")
        self._rng = np.random.default_rng(self.seed)

    @classmethod
    def from_exp1155(cls, path: str | Path, **kwargs: Any) -> "Phase4Sampler":
        """Construct the sampler recommended by an Exp 1155 artifact.

        Spec: REQ-KONA-010
        """
        artifact = json.loads(Path(path).read_text(encoding="utf-8"))
        algorithm = sampler_algorithm_from_exp1155(artifact)
        discrete_indices = _discrete_indices_from_artifact(artifact)
        if algorithm == "blocked_gibbs" and not discrete_indices:
            discrete_indices = (0,)
        return cls(
            algorithm=algorithm,
            hmc_regime_used=str(artifact.get("hmc_regime", "")),
            discrete_indices=discrete_indices,
            **kwargs,
        )

    @property
    def backend_name(self) -> str:
        """Human-readable backend name for logs and protocol checks."""
        return f"phase4_{self.algorithm}"

    def minimize_energy(
        self,
        biases: np.ndarray,
        couplings: np.ndarray,
        n_samples: int,
        n_steps: int,
        beta: float,
    ) -> np.ndarray:
        """SamplerBackend high-level Ising minimization compatibility method."""
        return self._sample_ising_backend(
            np.asarray(biases, dtype=np.float64),
            np.asarray(couplings, dtype=np.float64),
            int(n_samples),
            {"beta": float(beta), "n_steps": int(n_steps)},
        )

    def sample(  # type: ignore[override]
        # Deliberately dual-mode: dispatches to the SamplerBackend
        # (biases, couplings, n_samples, config) protocol call when args/
        # "config" are present, else to latent-chain sampling below. A
        # duck-typed polymorphic signature the SamplerBackend ABC can't
        # express without @overload; not a real interface violation.
        self,
        energy_fn: Any,
        init_state: Any = None,
        n_steps: int | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        """Sample a latent chain or, when `config` is supplied, Ising states.

        Spec: REQ-KONA-010
        """
        if args or "config" in kwargs:
            config = args[0] if args else kwargs["config"]
            return self._sample_ising_backend(
                np.asarray(energy_fn, dtype=np.float64),
                np.asarray(init_state, dtype=np.float64),
                int(n_steps),
                config,
            )

        state = _as_1d_state(init_state)
        n_chain_steps = int(n_steps)
        if self.algorithm == "blocked_gibbs":
            return self._sample_blocked_gibbs(energy_fn, state, n_chain_steps)
        if self.algorithm == "sgld":
            return self._sample_sgld(energy_fn, state, n_chain_steps)
        raise NotImplementedError(
            "Exp 1155 selected Regime C for the checked-in artifact; "
            "use blocked_gibbs or sgld for this Phase 4 deployment."
        )

    def _sample_ising_backend(
        self,
        biases: np.ndarray,
        couplings: np.ndarray,
        n_samples: int,
        config: dict[str, Any],
    ) -> np.ndarray:
        beta = float(config.get("beta", 1.0))
        n_steps = int(config.get("n_steps", config.get("n_warmup", 100)))
        temperature = 1.0 / max(beta, 1e-12)
        n_spins = int(biases.shape[0])
        samples = np.empty((n_samples, n_spins), dtype=bool)

        def ising_energy(spins: np.ndarray) -> float:
            return float(-biases @ spins - 0.5 * spins @ couplings @ spins)

        for row in range(n_samples):
            init = self._rng.choice(np.array([-1.0, 1.0]), size=n_spins)
            chain = self._sample_blocked_gibbs(
                ising_energy,
                init,
                n_steps,
                discrete_indices=tuple(range(n_spins)),
                continuous_indices=(),
                temperature=temperature,
            )
            samples[row] = chain[-1] > 0.0
        return samples

    def _sample_blocked_gibbs(
        self,
        energy_fn: Any,
        init_state: np.ndarray,
        n_steps: int,
        *,
        discrete_indices: tuple[int, ...] | None = None,
        continuous_indices: tuple[int, ...] | None = None,
        temperature: float | None = None,
    ) -> np.ndarray:
        x = np.clip(np.asarray(init_state, dtype=np.float64).copy(), -1.0, 1.0)
        discrete, continuous = self._resolve_blocks(
            x.size,
            discrete_indices=discrete_indices,
            continuous_indices=continuous_indices,
        )
        temp = float(self.temperature if temperature is None else temperature)
        chain = np.empty((n_steps, x.size), dtype=np.float64)
        discrete_updates = 0
        discrete_proposals = 0
        start_energy = _energy_value(energy_fn, x)

        if discrete:
            x[list(discrete)] = np.where(x[list(discrete)] >= 0.0, 1.0, -1.0)

        for step in range(n_steps):
            for idx in discrete:
                old_value = x[idx]
                x[idx] = 1.0
                energy_pos = _energy_value(energy_fn, x)
                x[idx] = -1.0
                energy_neg = _energy_value(energy_fn, x)
                p_pos = _binary_boltzmann_probability(energy_pos, energy_neg, temp)
                x[idx] = 1.0 if self._rng.random() < p_pos else -1.0
                discrete_updates += int(x[idx] != old_value)
                discrete_proposals += 1

            if continuous:
                grad = _clip_gradient(
                    _energy_gradient(energy_fn, x, self.finite_difference_eps),
                    self.max_grad_norm,
                )
                noise_scale = np.sqrt(2.0 * self.step_size * temp)
                noise = self._rng.normal(0.0, noise_scale, size=len(continuous))
                cont_idx = np.asarray(continuous, dtype=np.int64)
                x[cont_idx] = np.clip(
                    x[cont_idx] - self.step_size * grad[cont_idx] + noise,
                    -1.0,
                    1.0,
                )

            chain[step] = x

        end_energy = _energy_value(energy_fn, x)
        self.last_diagnostics = {
            "acceptance_rate": None,
            "discrete_update_rate": discrete_updates / max(discrete_proposals, 1),
            "convergence_metric": start_energy - end_energy,
        }
        return chain

    def _sample_sgld(self, energy_fn: Any, init_state: np.ndarray, n_steps: int) -> np.ndarray:
        x = np.clip(np.asarray(init_state, dtype=np.float64).copy(), -1.0, 1.0)
        chain = np.empty((n_steps, x.size), dtype=np.float64)
        step_sizes = np.empty(n_steps, dtype=np.float64)
        start_energy = _energy_value(energy_fn, x)

        for step in range(n_steps):
            step_t = self.step_size / np.sqrt(step + 1.0)
            grad = _clip_gradient(
                _energy_gradient(energy_fn, x, self.finite_difference_eps),
                self.max_grad_norm,
            )
            noise = self._rng.normal(0.0, np.sqrt(2.0 * step_t * self.temperature), size=x.size)
            x = np.clip(x - step_t * grad + noise, -1.0, 1.0)
            chain[step] = x
            step_sizes[step] = step_t

        end_energy = _energy_value(energy_fn, x)
        self.last_diagnostics = {
            "acceptance_rate": None,
            "mean_step_size": float(np.mean(step_sizes)),
            "convergence_metric": start_energy - end_energy,
        }
        return chain

    def _resolve_blocks(
        self,
        dim: int,
        *,
        discrete_indices: tuple[int, ...] | None,
        continuous_indices: tuple[int, ...] | None,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        discrete = _normalise_indices(
            self.discrete_indices if discrete_indices is None else discrete_indices,
            dim,
        )
        if continuous_indices is None:
            default_continuous = self.continuous_indices or tuple(
                idx for idx in range(dim) if idx not in discrete
            )
            continuous = _normalise_indices(default_continuous, dim)
        else:
            continuous = _normalise_indices(continuous_indices, dim)
        return discrete, continuous


def _discrete_indices_from_artifact(artifact: dict[str, Any]) -> tuple[int, ...]:
    names = list(artifact.get("component_names", []))
    return tuple(idx for idx, name in enumerate(names) if "AST" in str(name) or "Z3" in str(name))


def _normalise_indices(indices: tuple[int, ...], dim: int) -> tuple[int, ...]:
    seen: list[int] = []
    for raw_idx in indices:
        idx = int(raw_idx)
        if idx < 0 or idx >= dim:
            raise ValueError(f"Latent index {idx} is outside dimension {dim}")  # pragma: no cover
        if idx not in seen:
            seen.append(idx)
    return tuple(seen)


def _as_1d_state(init_state: Any) -> np.ndarray:
    state = np.asarray(init_state, dtype=np.float64)
    if state.ndim != 1:
        raise ValueError("init_state must be a one-dimensional latent vector")  # pragma: no cover
    return state


def _energy_value(energy_fn: Any, x: np.ndarray) -> float:
    if hasattr(energy_fn, "energy"):
        return float(energy_fn.energy(x))
    return float(energy_fn(x))


def _energy_gradient(energy_fn: Any, x: np.ndarray, eps: float) -> np.ndarray:
    if hasattr(energy_fn, "grad_energy"):
        return np.asarray(energy_fn.grad_energy(x), dtype=np.float64)

    grad = np.empty_like(x, dtype=np.float64)
    for idx in range(x.size):
        delta = np.zeros_like(x, dtype=np.float64)
        delta[idx] = eps
        grad[idx] = (_energy_value(energy_fn, x + delta) - _energy_value(energy_fn, x - delta)) / (
            2.0 * eps
        )
    return grad


def _clip_gradient(grad: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(grad))
    scale = min(1.0, max_norm / (norm + 1e-12))
    return np.asarray(grad * scale, dtype=np.float64)


def _binary_boltzmann_probability(
    energy_pos: float, energy_neg: float, temperature: float
) -> float:
    delta = np.clip((energy_pos - energy_neg) / max(temperature, 1e-12), -60.0, 60.0)
    return float(1.0 / (1.0 + np.exp(delta)))


BlockedGibbsSampler = Phase4Sampler

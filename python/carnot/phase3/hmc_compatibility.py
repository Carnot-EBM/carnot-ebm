"""HMC compatibility diagnostics for Phase 4 latent-space sampling.

**Researcher summary:**
    Deep Think Q7 requires a cheap gate before Carnot spends implementation time
    on Hamiltonian Monte Carlo. This module runs D1-D4 on a latent-space energy
    bridge for the current verifier ensemble and reports whether vanilla HMC is
    viable, needs preconditioning, or should be replaced by a fallback sampler.

**Detailed explanation for engineers:**
    The production k=5 verifier ensemble contains text and symbolic components
    such as AST and Z3 arithmetic checks. Those components are not directly
    differentiable with JAX, so the diagnostic uses central finite differences
    for every component gradient and records ``gradient_method="numerical_fd"``.
    This is intentionally a diagnostic bridge, not a sampler implementation.

Spec: REQ-KONA-009, SCENARIO-KONA-008
"""

from __future__ import annotations

import datetime as _datetime
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

EnergyFn = Callable[[np.ndarray], float]

GRADIENT_METHOD = "numerical_fd"
EXPERIMENT_ID = 1155
SCHEMA = "carnot.hmc_compatibility_diagnostics.v1"
_VARIANCE_FLOOR = 1e-12
_MAX_RATIO = 1e12


@dataclass(frozen=True)
class HMCCompatibilityConfig:
    """Configuration for the D1-D4 HMC compatibility diagnostics.

    Attributes:
        n_diagnostic_points: Number of random latent starts used for D1-D3/D4.
            The Exp 1155 task uses 100 points.
        seed: NumPy random seed for reproducible latent and momentum draws.
        d1_leapfrog_steps: Leapfrog steps for the reversibility diagnostic.
        d2_leapfrog_steps: Leapfrog steps for Hamiltonian conservation and D4.
        step_size: Leapfrog step size epsilon.
        fd_eps: Central finite-difference epsilon for gradient approximation.
    """

    n_diagnostic_points: int = 100
    seed: int = 1155
    d1_leapfrog_steps: int = 10
    d2_leapfrog_steps: int = 20
    step_size: float = 0.01
    fd_eps: float = 1e-5


@dataclass(frozen=True)
class LatentEnergyComponent:
    """One latent-space verifier component used by the diagnostics."""

    name: str
    energy_fn: EnergyFn
    continuous: bool = False

    def energy(self, x: np.ndarray) -> float:
        """Return the component energy at one latent vector."""
        return _finite_float(self.energy_fn(_as_vector(x)))


def _as_vector(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"expected a 1D latent vector, got shape {arr.shape}")  # pragma: no cover
    return arr


def _finite_float(value: float) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"energy must be finite, got {out}")  # pragma: no cover
    return out


def classify_d1_signal(error_mean: float) -> str:
    """Classify D1 symplectic reversibility error."""
    if error_mean < 0.01:
        return "A"
    if error_mean < 0.1:
        return "B"
    return "C"


def classify_d2_signal(variance: float) -> str:
    """Classify D2 Hamiltonian conservation variance."""
    if variance < 0.1:
        return "A"
    if variance < 1.0:
        return "B"
    return "C"


def classify_d3_signal(disparity_ratio: float) -> str:
    """Classify D3 cross-component gradient norm disparity."""
    if disparity_ratio < 10.0:
        return "A"
    if disparity_ratio < 100.0:
        return "B"
    return "C"


def classify_hmc_regime(d1_signal: str, d2_signal: str, d3_signal: str) -> str:
    """Return the worst D1-D3 signal as the final HMC regime."""
    order = {"A": 0, "B": 1, "C": 2}
    signals = (d1_signal, d2_signal, d3_signal)
    return max(signals, key=lambda signal: order[signal])


def finite_difference_gradient(energy_fn: EnergyFn, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Approximate ``grad energy_fn(x)`` with central finite differences."""
    x0 = _as_vector(x)
    grad = np.zeros_like(x0, dtype=np.float64)
    for idx in range(x0.size):
        step = np.zeros_like(x0, dtype=np.float64)
        step[idx] = eps
        grad[idx] = (energy_fn(x0 + step) - energy_fn(x0 - step)) / (2.0 * eps)
    return grad


def _total_energy(components: list[LatentEnergyComponent], x: np.ndarray) -> float:
    return float(sum(component.energy(x) for component in components))


def _gradient_for_components(
    components: list[LatentEnergyComponent],
    x: np.ndarray,
    fd_eps: float,
) -> np.ndarray:
    return finite_difference_gradient(lambda z: _total_energy(components, z), x, eps=fd_eps)


def _hamiltonian(components: list[LatentEnergyComponent], q: np.ndarray, p: np.ndarray) -> float:
    return float(_total_energy(components, q) + 0.5 * np.dot(p, p))


def leapfrog(
    components: list[LatentEnergyComponent],
    q: np.ndarray,
    p: np.ndarray,
    *,
    step_size: float,
    n_steps: int,
    fd_eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run leapfrog integration with finite-difference verifier gradients."""
    q_next = _as_vector(q).copy()
    p_next = _as_vector(p).copy()

    p_next -= 0.5 * step_size * _gradient_for_components(components, q_next, fd_eps)
    for idx in range(n_steps):
        q_next += step_size * p_next
        grad = _gradient_for_components(components, q_next, fd_eps)
        factor = 0.5 if idx == n_steps - 1 else 1.0
        p_next -= factor * step_size * grad
    return q_next, p_next


def symplectic_reversibility_error_mean(
    components: list[LatentEnergyComponent],
    q_points: np.ndarray,
    p_points: np.ndarray,
    config: HMCCompatibilityConfig,
) -> float:
    """Compute D1 mean reconstruction error after forward/reversed leapfrog."""
    errors = []
    for q0, p0 in zip(q_points, p_points):
        q_forward, p_forward = leapfrog(
            components,
            q0,
            p0,
            step_size=config.step_size,
            n_steps=config.d1_leapfrog_steps,
            fd_eps=config.fd_eps,
        )
        q_reconstructed, _ = leapfrog(
            components,
            q_forward,
            -p_forward,
            step_size=config.step_size,
            n_steps=config.d1_leapfrog_steps,
            fd_eps=config.fd_eps,
        )
        errors.append(float(np.linalg.norm(q0 - q_reconstructed)))
    return float(np.mean(errors))


def hamiltonian_delta_variance(
    components: list[LatentEnergyComponent],
    q_points: np.ndarray,
    p_points: np.ndarray,
    config: HMCCompatibilityConfig,
) -> float:
    """Compute D2/D4 variance of absolute Hamiltonian drift."""
    deltas = []
    for q0, p0 in zip(q_points, p_points):
        h_start = _hamiltonian(components, q0, p0)
        q_final, p_final = leapfrog(
            components,
            q0,
            p0,
            step_size=config.step_size,
            n_steps=config.d2_leapfrog_steps,
            fd_eps=config.fd_eps,
        )
        h_final = _hamiltonian(components, q_final, p_final)
        deltas.append(abs(h_final - h_start))
    return float(np.var(np.asarray(deltas, dtype=np.float64)))


def gradient_disparity_ratio(
    components: list[LatentEnergyComponent],
    q_points: np.ndarray,
    fd_eps: float,
) -> tuple[float, dict[str, float]]:
    """Compute D3 max/min variance of per-component gradient norms."""
    variances: dict[str, float] = {}
    for component in components:
        norms = [
            float(np.linalg.norm(finite_difference_gradient(component.energy, q, eps=fd_eps)))
            for q in q_points
        ]
        variances[component.name] = float(np.var(np.asarray(norms, dtype=np.float64)))

    values = np.asarray(list(variances.values()), dtype=np.float64)
    max_var = float(np.max(values))
    min_var = float(np.min(values))
    if max_var <= _VARIANCE_FLOOR:
        ratio = 1.0
    elif min_var <= _VARIANCE_FLOOR:
        ratio = min(_MAX_RATIO, max_var / _VARIANCE_FLOOR)
    else:
        ratio = min(_MAX_RATIO, max_var / min_var)
    return float(ratio), variances


def _d4_bottleneck(subspace_variance: float, full_variance: float) -> bool:
    return bool(subspace_variance < 0.1 and full_variance >= 0.1)


def _recommended_sampler(regime: str, d4_bottleneck: bool) -> str:
    if regime == "A":
        return "hmc"
    if regime == "B":
        return "preconditioned_hmc"
    return "blocked_gibbs" if d4_bottleneck else "langevin"


def _honest_verdict(regime: str) -> str:
    return {
        "A": "regime_A_hmc_viable",
        "B": "regime_B_preconditioning_needed",
        "C": "regime_C_hmc_inappropriate",
    }[regime]


def build_hmc_compatibility_artifact(
    *,
    latent_dim: int,
    d1_error_mean: float,
    d2_variance: float,
    d3_disparity_ratio: float,
    d4_subspace_variance: float,
    d4_full_variance: float,
    gradient_method: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the JSON-serialisable Exp 1155 diagnostic artifact."""
    if latent_dim <= 0:
        raise ValueError("latent_dim must be positive")  # pragma: no cover
    d1_signal = classify_d1_signal(d1_error_mean)
    d2_signal = classify_d2_signal(d2_variance)
    d3_signal = classify_d3_signal(d3_disparity_ratio)
    regime = classify_hmc_regime(d1_signal, d2_signal, d3_signal)
    d4_bottleneck = _d4_bottleneck(d4_subspace_variance, d4_full_variance)
    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_ID,
        "run_date": _datetime.date.today().isoformat(),
        "latent_dim": int(latent_dim),
        "d1_symplectic_reversibility_error_mean": float(d1_error_mean),
        "d1_regime_signal": d1_signal,
        "d2_hamiltonian_variance": float(d2_variance),
        "d2_regime_signal": d2_signal,
        "d3_gradient_disparity_ratio": float(d3_disparity_ratio),
        "d3_regime_signal": d3_signal,
        "d4_subspace_delta_h_variance": float(d4_subspace_variance),
        "d4_full_delta_h_variance": float(d4_full_variance),
        "d4_discrete_components_bottleneck": d4_bottleneck,
        "gradient_method": gradient_method,
        "hmc_regime_classified": True,
        "hmc_regime": regime,
        "recommended_sampler": _recommended_sampler(regime, d4_bottleneck),
        "honest_verdict": _honest_verdict(regime),
    }
    if extra:
        artifact.update(extra)
    return artifact


def load_latent_dim_from_exp1154(path: str | Path) -> int:
    """Read latent_dim from the Exp 1154 snap-validity artifact."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    latent_dim = int(payload.get("latent_dim", 0))
    if latent_dim <= 0:
        raise ValueError(f"latent_dim missing or invalid in {path}")  # pragma: no cover
    return latent_dim


def _sample_phase_points(
    latent_dim: int,
    n_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    q_points = rng.uniform(-1.0, 1.0, size=(n_points, latent_dim)).astype(np.float64)
    p_points = rng.normal(0.0, 1.0, size=(n_points, latent_dim)).astype(np.float64)
    return q_points, p_points


def _smooth_semenergy_proxy(x: np.ndarray) -> float:
    x_vec = _as_vector(x)
    centered = x_vec - float(np.mean(x_vec))
    return float(np.mean(np.logaddexp(0.0, centered)))


def _smooth_thinkprm_proxy(x: np.ndarray) -> float:
    x_vec = _as_vector(x)
    weights = np.linspace(-0.75, 0.75, x_vec.size, dtype=np.float64)
    margin = float(np.dot(weights, x_vec) / max(x_vec.size, 1))
    return float(np.logaddexp(0.0, margin))


def build_default_latent_components(latent_dim: int) -> list[LatentEnergyComponent]:
    """Build the default k=5 latent verifier energy bridge for Exp 1155."""
    SOSKANEnergyV3 = _load_soskan_energy_v3()

    sos = SOSKANEnergyV3(
        n_splines=8,
        rank=4,
        n_features=latent_dim,
        hidden_dim=16,
        seed=1121,
    )

    def sos_energy(x: np.ndarray) -> float:
        x_vec = np.clip(_as_vector(x), -1.0, 1.0)
        return float(sos.energy(x_vec) / max(latent_dim, 1))

    def ast_structure_energy(x: np.ndarray) -> float:
        x_vec = _as_vector(x)
        signs = np.where(x_vec >= 0.0, 1.0, -1.0)
        flips = np.mean(np.abs(np.diff(signs))) / 2.0 if x_vec.size > 1 else 0.0
        return float(min(1.0, 0.25 * flips))

    def semantic_consistency_energy(x: np.ndarray) -> float:
        x_vec = _as_vector(x)
        adjacent = float(np.mean(x_vec[:-1] * x_vec[1:])) if x_vec.size > 1 else 0.0
        return float(0.5 * (1.0 + np.tanh(adjacent)))

    def z3_math_energy(x: np.ndarray) -> float:
        x_vec = _as_vector(x)
        rounded_balance = float(np.sum(np.round(3.0 * x_vec)))
        return 0.0 if abs(rounded_balance) <= max(1.0, 0.25 * x_vec.size) else 1.0

    return [
        LatentEnergyComponent("SOSKANEnergyV3", sos_energy, continuous=True),
        LatentEnergyComponent("SemEnergyProbe", _smooth_semenergy_proxy, continuous=True),
        LatentEnergyComponent("ASTStructureVerifier", ast_structure_energy, continuous=False),
        LatentEnergyComponent(
            "SemanticConsistencyVerifier",
            semantic_consistency_energy,
            continuous=False,
        ),
        LatentEnergyComponent("Z3MathVerifier", z3_math_energy, continuous=False),
    ]


def _load_soskan_energy_v3() -> Any:
    """Load SOSKANEnergyV3 without importing the JAX-heavy models package."""
    module_name = "_carnot_exp1155_sos_kan"
    if module_name in sys.modules:
        return sys.modules[module_name].SOSKANEnergyV3

    module_path = Path(__file__).resolve().parents[1] / "models" / "sos_kan.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {module_path}")  # pragma: no cover
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.SOSKANEnergyV3


def build_default_continuous_subspace_components(latent_dim: int) -> list[LatentEnergyComponent]:
    """Build the D4 SemEnergy + ThinkPRM continuous-subspace bridge."""
    return [
        LatentEnergyComponent("SemEnergyProbe", _smooth_semenergy_proxy, continuous=True),
        LatentEnergyComponent("ThinkPRMProbe", _smooth_thinkprm_proxy, continuous=True),
    ]


def run_hmc_compatibility_diagnostics(
    *,
    latent_dim: int,
    components: list[LatentEnergyComponent] | None = None,
    continuous_components: list[LatentEnergyComponent] | None = None,
    config: HMCCompatibilityConfig | None = None,
) -> dict[str, Any]:
    """Run D1-D4 and return the Exp 1155 HMC compatibility artifact."""
    if latent_dim <= 0:
        raise ValueError("latent_dim must be positive")  # pragma: no cover
    cfg = config or HMCCompatibilityConfig()
    full_components = components or build_default_latent_components(latent_dim)
    subspace_components = continuous_components or build_default_continuous_subspace_components(
        latent_dim
    )

    q_points, p_points = _sample_phase_points(latent_dim, cfg.n_diagnostic_points, cfg.seed)
    d1_error = symplectic_reversibility_error_mean(full_components, q_points, p_points, cfg)
    d2_variance = hamiltonian_delta_variance(full_components, q_points, p_points, cfg)
    d3_ratio, component_variances = gradient_disparity_ratio(full_components, q_points, cfg.fd_eps)
    d4_subspace_variance = hamiltonian_delta_variance(
        subspace_components,
        q_points,
        p_points,
        cfg,
    )

    return build_hmc_compatibility_artifact(
        latent_dim=latent_dim,
        d1_error_mean=d1_error,
        d2_variance=d2_variance,
        d3_disparity_ratio=d3_ratio,
        d4_subspace_variance=d4_subspace_variance,
        d4_full_variance=d2_variance,
        gradient_method=GRADIENT_METHOD,
        extra={
            "seed": cfg.seed,
            "n_diagnostic_points": cfg.n_diagnostic_points,
            "step_size": cfg.step_size,
            "d1_leapfrog_steps": cfg.d1_leapfrog_steps,
            "d2_leapfrog_steps": cfg.d2_leapfrog_steps,
            "finite_difference_eps": cfg.fd_eps,
            "component_names": [component.name for component in full_components],
            "continuous_subspace_component_names": [
                component.name for component in subspace_components
            ],
            "d3_component_gradient_norm_variances": component_variances,
            "latent_energy_bridge": (
                "finite-difference latent proxy for text/symbolic k=5 verifiers; "
                "SOSKANEnergyV3 is evaluated directly, text-only verifier families "
                "use deterministic latent proxy energies"
            ),
        },
    )

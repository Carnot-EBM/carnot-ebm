"""Analytical benchmark energy functions with known solutions.

Provides classical optimization test functions implementing the EnergyFunction
protocol, with known global minima for quantitative evaluation of samplers
and optimizers. These are the Python/JAX equivalents of the Rust benchmarks
in crates/carnot-core/src/benchmarks/.

Each benchmark tests a different challenge:
- DoubleWell: multimodal sampling (can the sampler find both valleys?)
- Rosenbrock: narrow valley navigation (can it follow curved geometry?)
- Ackley: many local minima (can it escape traps?)
- Rastrigin: regular local minima grid (systematic trap avoidance)
- GaussianMixture: known distribution (does it recover correct probabilities?)

Spec: REQ-AUTO-001
"""

from carnot.benchmarks.functions import (
    Ackley,
    BenchmarkInfo,
    DoubleWell,
    GaussianMixture,
    Rastrigin,
    Rosenbrock,
)

__all__ = [
    "Ackley",
    "BenchmarkInfo",
    "DoubleWell",
    "GaussianMixture",
    "Rastrigin",
    "Rosenbrock",
]

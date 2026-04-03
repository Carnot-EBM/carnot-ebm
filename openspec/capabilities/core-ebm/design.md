# Core EBM — Design Document

**Capability:** core-ebm
**Version:** 0.1.0

## Rust Design

### EnergyFunction Trait

```rust
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Core trait for energy-based models.
/// All model tiers implement this trait.
pub trait EnergyFunction: Send + Sync {
    /// Compute scalar energy for a single input.
    fn energy(&self, x: &ArrayView1<f32>) -> f32;

    /// Compute energy for a batch of inputs.
    /// Default implementation maps over batch dimension.
    fn energy_batch(&self, xs: &ArrayView2<f32>) -> Array1<f32> {
        Array1::from_iter(xs.rows().into_iter().map(|row| self.energy(&row)))
    }

    /// Compute gradient of energy w.r.t. input.
    fn grad_energy(&self, x: &ArrayView1<f32>) -> Array1<f32>;

    /// Number of input dimensions this model expects.
    fn input_dim(&self) -> usize;
}
```

### ModelState

```rust
use std::collections::HashMap;

pub struct ModelState {
    pub parameters: HashMap<String, Array1<f32>>,
    pub config: ModelConfig,
    pub metadata: ModelMetadata,
}

pub struct ModelConfig {
    pub input_dim: usize,
    pub hidden_dims: Vec<usize>,
    pub precision: Precision,
}

pub enum Precision {
    F32,
    F64,
}

pub struct ModelMetadata {
    pub step: u64,
    pub loss_history: Vec<f32>,
}
```

## Python/JAX Design

### EnergyFunction Protocol

```python
from typing import Protocol, runtime_checkable
import jax
import jax.numpy as jnp

@runtime_checkable
class EnergyFunction(Protocol):
    def energy(self, x: jax.Array) -> jax.Array:
        """Compute scalar energy for input x."""
        ...

    def energy_batch(self, xs: jax.Array) -> jax.Array:
        """Compute energy for batch of inputs."""
        ...

    def grad_energy(self, x: jax.Array) -> jax.Array:
        """Gradient of energy w.r.t. x. Default: jax.grad(self.energy)."""
        ...

    @property
    def input_dim(self) -> int:
        ...
```

### Auto-Gradient Mixin

```python
class AutoGradMixin:
    """Mixin that auto-derives grad_energy from energy using jax.grad."""
    def grad_energy(self, x: jax.Array) -> jax.Array:
        return jax.grad(self.energy)(x)
```

## Serialization

Both Rust and Python use the `safetensors` format:
- Rust: `safetensors` crate
- Python: `safetensors` Python package

This ensures cross-language model portability.

## PyO3 Binding Strategy

The `carnot-python` crate wraps each Rust model in a `#[pyclass]` struct that:
1. Accepts numpy arrays via `PyReadonlyArray`
2. Converts to ndarray views (zero-copy for contiguous C-order arrays)
3. Calls the Rust `EnergyFunction` trait method
4. Returns results as numpy arrays

```rust
#[pyclass]
struct PyEnergyModel {
    inner: Box<dyn EnergyFunction>,
}

#[pymethods]
impl PyEnergyModel {
    fn energy(&self, x: PyReadonlyArray1<f32>) -> f32 {
        self.inner.energy(&x.as_array())
    }
}
```

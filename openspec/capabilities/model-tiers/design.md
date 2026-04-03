# Model Tiers — Design Document

**Capability:** model-tiers
**Version:** 0.1.0

## Ising Tier (Small)

The Ising model implements a classical pairwise interaction energy:

```
E(x) = -0.5 * x^T J x - b^T x
```

Where:
- `J` is a symmetric coupling matrix (input_dim x input_dim)
- `b` is a bias vector (input_dim)
- Optionally, a single hidden layer: `E(x) = -log sum_h exp(-x^T W h - b^T x - c^T h)`

### Rust Implementation

```rust
pub struct IsingModel {
    coupling: Array2<f32>,  // J matrix
    bias: Array1<f32>,      // b vector
    config: IsingConfig,
}

impl EnergyFunction for IsingModel {
    fn energy(&self, x: &ArrayView1<f32>) -> f32 {
        -0.5 * x.dot(&self.coupling.dot(x)) - self.bias.dot(x)
    }

    fn grad_energy(&self, x: &ArrayView1<f32>) -> Array1<f32> {
        -(&self.coupling.dot(x)) - &self.bias
    }

    fn input_dim(&self) -> usize { self.config.input_dim }
}
```

## Gibbs Tier (Medium)

Multi-layer energy network with configurable architecture:

```
E(x) = f_L(...f_2(f_1(x)))  (scalar output)
```

Where each `f_i` is a dense layer with activation. The final layer outputs a scalar.

### Architecture
- Input layer: `input_dim -> hidden_dims[0]`
- Hidden layers: `hidden_dims[i] -> hidden_dims[i+1]` with SiLU activation
- Output layer: `hidden_dims[-1] -> 1` (scalar energy)
- Optional dropout between layers

## Boltzmann Tier (Large)

Deep residual energy network with optional attention:

### Architecture
- Input projection: `input_dim -> hidden_dims[0]`
- Residual blocks: each with LayerNorm -> Linear -> SiLU -> Linear + skip
- Optional multi-head self-attention between residual blocks
- Output projection: `hidden_dims[-1] -> 1`

### Design Rationale
The residual connections prevent gradient degradation in deep networks, which is critical for stable energy landscape learning. Attention allows the model to capture long-range interactions in structured data.

## Common Patterns

### Parameter Initialization
All tiers use a shared `Initializer` enum:
```rust
pub enum Initializer {
    XavierUniform,
    HeNormal,
    Custom(Box<dyn Fn(usize, usize) -> Array2<f32>>),
}
```

### Configuration Validation
All tier configs implement a `validate() -> Result<(), ConfigError>` method called at construction time, ensuring invalid configurations fail fast with descriptive errors.

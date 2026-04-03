# Training & Inference — Design Document

**Capability:** training-inference
**Version:** 0.1.0

## Sampler Design

### Rust Sampler Trait

```rust
pub trait Sampler: Send + Sync {
    /// Run sampler for n_steps, return final sample.
    fn sample(
        &self,
        energy_fn: &dyn EnergyFunction,
        init: &ArrayView1<f32>,
        n_steps: usize,
    ) -> Array1<f32>;

    /// Run sampler, return full chain for diagnostics.
    fn sample_chain(
        &self,
        energy_fn: &dyn EnergyFunction,
        init: &ArrayView1<f32>,
        n_steps: usize,
    ) -> Vec<Array1<f32>>;
}
```

### Langevin Dynamics

```rust
pub struct LangevinSampler {
    pub step_size: f32,
    pub noise_scale: f32,  // typically sqrt(2 * step_size)
}

impl Sampler for LangevinSampler {
    fn sample(&self, energy_fn: &dyn EnergyFunction, init: &ArrayView1<f32>, n_steps: usize) -> Array1<f32> {
        let mut x = init.to_owned();
        for _ in 0..n_steps {
            let grad = energy_fn.grad_energy(&x.view());
            let noise = /* sample from N(0, I) */;
            x = &x - self.step_size * 0.5 * &grad + self.noise_scale * &noise;
        }
        x
    }
}
```

### HMC

Leapfrog integration with Metropolis correction:
1. Sample momentum `p ~ N(0, M)`
2. Half-step momentum: `p -= (step_size/2) * grad_energy(x)`
3. Full-step position: `x += step_size * M^{-1} p`  (repeat L times)
4. Half-step momentum: `p -= (step_size/2) * grad_energy(x)`
5. Accept/reject based on Hamiltonian difference

## Training Algorithm Design

### Trainer Trait (Rust)

```rust
pub trait Trainer {
    type Model: EnergyFunction;

    fn train_step(
        &mut self,
        model: &mut Self::Model,
        batch: &ArrayView2<f32>,
    ) -> TrainStepResult;
}

pub struct TrainStepResult {
    pub loss: f32,
    pub grad_norm: f32,
}
```

### CD-k Algorithm

```
for each batch:
    x_pos = batch                        # positive samples
    x_neg = x_pos.clone()
    for _ in range(k):
        x_neg = gibbs_step(model, x_neg)  # negative sampling
    loss = mean(energy(x_pos)) - mean(energy(x_neg))
    update params to minimize loss
```

### Score Matching

Denoising score matching loss:
```
L = E_x E_noise [ || grad_energy(x + noise) - (-noise/sigma^2) ||^2 ]
```

### Python/JAX Training

JAX training uses functional transforms:
```python
@jax.jit
def train_step(params, batch, opt_state):
    def loss_fn(params):
        return cd_loss(model, params, batch, k=1)
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss
```

## Optimizer Integration

- Rust: Custom SGD/Adam implementations (minimal dependencies)
- Python: `optax` library for JAX-compatible optimizers

## Checkpointing

Both languages use `safetensors` for parameter serialization. Training state (optimizer moments, step count, RNG state) is stored as a separate JSON sidecar file alongside the safetensors checkpoint.

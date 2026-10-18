# Verifiable Reasoning — Design Document

**Capability:** verifiable-reasoning
**Version:** 0.1.0

## Core Concept: Constraints as Energy Terms

Every constraint in the system is an energy function that maps a configuration to a non-negative scalar. The key insight: if you can write a function that returns 0 when a constraint is satisfied and > 0 when violated, you have both a loss function for optimization AND a verification predicate.

```
constraint_satisfied(x) ⟺ E_constraint(x) = 0
constraint_violated(x)  ⟺ E_constraint(x) > 0
```

This duality — the same function trains AND verifies — is what makes EBMs fundamentally different from LLMs. There is no separate "checker" that might disagree with the "generator."

## Rust Design

### ConstraintTerm Trait

```rust
/// A single verifiable constraint, expressed as an energy term.
pub trait ConstraintTerm: Send + Sync {
    /// Human-readable name for this constraint.
    fn name(&self) -> &str;

    /// Compute constraint energy. Returns 0.0 if satisfied.
    fn energy(&self, x: &ArrayView1<Float>) -> Float;

    /// Gradient of constraint energy w.r.t. x.
    fn grad_energy(&self, x: &ArrayView1<Float>) -> Array1<Float>;

    /// Threshold below which the constraint is considered satisfied.
    fn satisfaction_threshold(&self) -> Float {
        1e-6
    }

    /// Is this constraint satisfied for configuration x?
    fn is_satisfied(&self, x: &ArrayView1<Float>) -> bool {
        self.energy(x) <= self.satisfaction_threshold()
    }
}
```

### ComposedEnergyFunction

```rust
/// An energy function composed of weighted constraint terms.
pub struct ComposedEnergy {
    terms: Vec<(Box<dyn ConstraintTerm>, Float)>,  // (constraint, weight)
}

impl ComposedEnergy {
    pub fn add_constraint(&mut self, term: Box<dyn ConstraintTerm>, weight: Float) {
        self.terms.push((term, weight));
    }

    /// Per-constraint energy decomposition.
    pub fn decompose(&self, x: &ArrayView1<Float>) -> Vec<ConstraintReport> {
        self.terms.iter().map(|(term, weight)| {
            let raw_energy = term.energy(x);
            ConstraintReport {
                name: term.name().to_string(),
                energy: raw_energy,
                weighted_energy: weight * raw_energy,
                satisfied: term.is_satisfied(x),
            }
        }).collect()
    }
}

impl EnergyFunction for ComposedEnergy {
    fn energy(&self, x: &ArrayView1<Float>) -> Float {
        self.terms.iter()
            .map(|(term, weight)| weight * term.energy(x))
            .sum()
    }

    fn grad_energy(&self, x: &ArrayView1<Float>) -> Array1<Float> {
        self.terms.iter()
            .map(|(term, weight)| *weight * term.grad_energy(x))
            .fold(Array1::zeros(x.len()), |acc, g| acc + g)
    }
}
```

### VerificationResult

```rust
pub struct VerificationResult {
    pub total_energy: Float,
    pub constraints: Vec<ConstraintReport>,
    pub verdict: Verdict,
}

pub struct ConstraintReport {
    pub name: String,
    pub energy: Float,
    pub weighted_energy: Float,
    pub satisfied: bool,
}

pub enum Verdict {
    Verified,
    Violated { failing: Vec<String> },
}
```

### Gradient-Based Repair

Repair operates by computing gradients only from the violated constraint terms:

```rust
pub fn repair(
    composed: &ComposedEnergy,
    x: &Array1<Float>,
    step_size: Float,
    max_steps: usize,
) -> (Array1<Float>, Vec<VerificationResult>) {
    let mut x = x.clone();
    let mut history = Vec::new();

    for _ in 0..max_steps {
        let report = composed.verify(&x.view());
        history.push(report.clone());
        if matches!(report.verdict, Verdict::Verified) { break; }

        // Gradient only from violated constraints
        let grad = composed.grad_violated_only(&x.view());
        x = &x - step_size * &grad;
    }
    (x, history)
}
```

The key property: repair only descends on violated constraint energy, leaving satisfied constraint regions undisturbed. This prevents the "fix one thing, break another" failure mode.

## Python/JAX Design

### Constraint Protocol

```python
@runtime_checkable
class ConstraintTerm(Protocol):
    @property
    def name(self) -> str: ...
    def energy(self, x: jax.Array) -> jax.Array: ...
    def grad_energy(self, x: jax.Array) -> jax.Array: ...
    def satisfaction_threshold(self) -> float: ...
```

### JAX-Specific Advantages

JAX's functional transforms make verification particularly elegant:

```python
# Automatic per-constraint decomposition via vmap
def decompose(constraints, weights, x):
    energies = jnp.array([c.energy(x) for c in constraints])
    weighted = weights * energies
    satisfied = energies < jnp.array([c.satisfaction_threshold() for c in constraints])
    return energies, weighted, satisfied

# JIT-compiled repair loop
@jax.jit
def repair_step(constraints, weights, x, step_size):
    # Gradient only from violated terms
    def violated_energy(x):
        energies = jnp.array([c.energy(x) for c in constraints])
        thresholds = jnp.array([c.satisfaction_threshold() for c in constraints])
        mask = energies > thresholds  # only violated
        return jnp.sum(weights * energies * mask)
    grad = jax.grad(violated_energy)(x)
    return x - step_size * grad
```

## Energy Landscape Certification

### Local Minimum Verification

```python
def verify_local_minimum(energy_fn, x, eps=1e-3):
    """Check if x is a local minimum via Hessian eigenvalue analysis."""
    hessian = jax.hessian(energy_fn.energy)(x)
    eigenvalues = jnp.linalg.eigvalsh(hessian)
    is_minimum = jnp.all(eigenvalues > 0)
    condition_number = jnp.max(eigenvalues) / jnp.max(jnp.min(eigenvalues), eps)
    return is_minimum, eigenvalues, condition_number
```

### Basin of Attraction

Estimate by measuring how far you can perturb the solution before it leaves the basin:

```python
def estimate_basin_radius(energy_fn, x, n_perturbations=100, key=None):
    """Estimate basin of attraction radius via random perturbations."""
    base_energy = energy_fn.energy(x)
    radii = jnp.logspace(-4, 0, 20)
    for r in radii:
        perturbations = x + r * jrandom.normal(key, (n_perturbations, x.shape[0]))
        energies = jax.vmap(energy_fn.energy)(perturbations)
        # If most perturbations lead to higher energy, we're still in the basin
        fraction_higher = jnp.mean(energies > base_energy)
        if fraction_higher < 0.9:
            return float(r)  # basin boundary
    return float(radii[-1])
```

## Sudoku Example: Constraint Encoding

To demonstrate the full verification pipeline:

```rust
struct RowConstraint { row: usize }

impl ConstraintTerm for RowConstraint {
    fn name(&self) -> &str { "row_uniqueness" }

    fn energy(&self, grid: &ArrayView1<Float>) -> Float {
        // grid is flattened 9x9 = 81 elements
        let row = &grid.slice(s![self.row*9..(self.row+1)*9]);
        // Penalty: sum of (count - 1)^2 for each value that appears more than once
        let mut penalty = 0.0;
        for val in 1..=9 {
            let count = row.iter().filter(|&&x| (x - val as Float).abs() < 0.5).count();
            if count > 1 { penalty += (count - 1) as Float; }
        }
        penalty
    }
}
```

This encoding means:
- A valid Sudoku row has energy 0 (each digit appears exactly once)
- An invalid row has energy proportional to the number of duplicates
- The gradient points toward fixing the duplicates
- Verification is just checking energy = 0

use carnot_core::{EnergyFunction, Float};
use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand_distr::StandardNormal;

/// CASAL sampler variant for strictly constrained generative modeling.
///
/// This implements a Split Augmented Langevin Sampling approach where an unconstrained
/// Langevin step is followed by a projection step to enforce hard constraints.
pub fn casal_sample(
    energy_fn: &dyn EnergyFunction,
    constraint_value_and_grad: &dyn Fn(&Array1<Float>) -> (Float, Array1<Float>),
    init_state: &Array1<Float>,
    steps: usize,
    step_size: Float,
    proj_steps: usize,
    proj_lr: Float,
) -> Array1<Float> {
    let noise_scale = (2.0 * step_size).sqrt();
    let mut state = init_state.clone();

    for _ in 0..steps {
        // 1. Unconstrained Langevin step
        let grad_energy = energy_fn.grad_energy(&state.view());
        let noise: Array1<Float> = Array1::random(state.len(), StandardNormal);
        
        let mut proposed_state = &state - (step_size * &grad_energy) + (noise_scale * &noise);
        
        // 2. Split augmentation (projection) step
        for _ in 0..proj_steps {
            let (violation, grad_c) = constraint_value_and_grad(&proposed_state);
            if violation > 0.0 {
                proposed_state = &proposed_state - (proj_lr * &grad_c);
            }
        }
        
        // 3. Strict gate
        let (violation, _) = constraint_value_and_grad(&proposed_state);
        if violation <= 1e-5 {
            state = proposed_state;
        }
        // else state remains the same
    }
    
    state
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, ArrayView1};

    struct DummyEnergy;
    impl EnergyFunction for DummyEnergy {
        fn energy(&self, x: &ArrayView1<Float>) -> Float {
            0.5 * x.dot(x)
        }
        fn grad_energy(&self, x: &ArrayView1<Float>) -> Array1<Float> {
            x.to_owned()
        }
        fn input_dim(&self) -> usize {
            2
        }
    }

    #[test]
    fn test_casal_sample() {
        let energy_fn = DummyEnergy;
        let constraint_fn = |x: &Array1<Float>| -> (Float, Array1<Float>) {
            // constraint: x[0] + x[1] = 0
            let val = x[0] + x[1];
            let violation = val * val; // Smooth constraint
            let grad = array![2.0 * val, 2.0 * val];
            (violation, grad)
        };
        let init_state = array![1.0, 1.0];
        let state = casal_sample(
            &energy_fn,
            &constraint_fn,
            &init_state,
            100,
            0.01,
            50,
            0.1,
        );
        assert!((state[0] + state[1]).abs() <= 1e-2); // The gate is violation <= 1e-5, meaning val^2 <= 1e-5, so val <= 3.16e-3.
    }
}


//! E2E tests: sample from benchmark energy functions and verify sample quality.
//!
//! These are the first real end-to-end tests: create a known energy landscape,
//! run MCMC sampling, and verify the samples have the correct statistical properties.
//!
//! Spec: REQ-AUTO-001, SCENARIO-SAMPLE-001, SCENARIO-SAMPLE-002

use carnot_core::benchmarks::{DoubleWell, GaussianMixture, Rosenbrock};
use carnot_core::{EnergyFunction, Float};
use carnot_samplers::{HmcSampler, LangevinSampler, Sampler};
use ndarray::Array1;

/// Helper: compute sample mean from the latter half of a chain (burn-in).
fn chain_mean(chain: &[Array1<Float>]) -> Array1<Float> {
    let burn_in = chain.len() / 2;
    let samples = &chain[burn_in..];
    let n = samples.len() as Float;
    let mut mean = Array1::zeros(samples[0].len());
    for s in samples {
        mean = &mean + s;
    }
    mean / n
}

#[test]
fn test_langevin_on_double_well() {
    // REQ-AUTO-001, SCENARIO-SAMPLE-001: Langevin samples from DoubleWell
    let dw = DoubleWell::new(2);
    let sampler = LangevinSampler::new(0.005);
    let init = Array1::from_vec(vec![0.8, 0.0]);

    let chain = sampler.sample_chain(&dw, &init, 20000);
    let mean = chain_mean(&chain);

    // DoubleWell has minima at x[0]=±1. Samples should cluster near one.
    // x[1] mean should be near 0.
    assert!(
        mean[1].abs() < 1.0,
        "x[1] mean should be near 0, got {}", mean[1]
    );

    // Check samples are finite
    let burn_in = chain.len() / 2;
    assert!(
        chain[burn_in..].iter().all(|s| s.iter().all(|v| v.is_finite())),
        "All samples should be finite"
    );
}

#[test]
fn test_langevin_on_rosenbrock() {
    // REQ-AUTO-001, SCENARIO-SAMPLE-001: Langevin explores Rosenbrock valley
    let rb = Rosenbrock::new(2);
    let sampler = LangevinSampler::new(0.0001); // small step for narrow valley
    let init = Array1::from_vec(vec![0.5, 0.5]);

    let final_sample = sampler.sample(&rb, &init, 5000);
    let final_energy = rb.energy(&final_sample.view());

    // Should have found a low-energy region
    assert!(
        final_energy < 10.0,
        "Should be in low-energy region, got energy={}",
        final_energy
    );
    assert!(final_sample.iter().all(|v| v.is_finite()));
}

#[test]
fn test_hmc_on_double_well() {
    // REQ-AUTO-001, SCENARIO-SAMPLE-002: HMC samples from DoubleWell
    let dw = DoubleWell::new(2);
    let sampler = HmcSampler::new(0.05, 10);
    let init = Array1::from_vec(vec![0.8, 0.0]);

    let chain = sampler.sample_chain(&dw, &init, 2000);

    // Check samples are finite and in reasonable range
    let burn_in = chain.len() / 2;
    for s in &chain[burn_in..] {
        assert!(
            s.iter().all(|v| v.is_finite() && v.abs() < 100.0),
            "Sample should be finite and bounded: {:?}", s
        );
    }
}

#[test]
fn test_langevin_on_gaussian_mixture() {
    // REQ-AUTO-001, SCENARIO-SAMPLE-001: sample from known Gaussian mixture
    let gmm = GaussianMixture::two_modes(4.0); // modes at -2 and +2
    let sampler = LangevinSampler::new(0.01);
    let init = Array1::from_vec(vec![-2.0]); // start at mode 1

    let chain = sampler.sample_chain(&gmm, &init, 10000);
    let burn_in = chain.len() / 2;
    let samples = &chain[burn_in..];

    // Samples should be concentrated near the modes (±2)
    // Most samples should have |x| > 0.5 (not stuck at origin)
    let near_modes: usize = samples
        .iter()
        .filter(|s| s[0].abs() > 0.5)
        .count();
    let fraction = near_modes as Float / samples.len() as Float;
    assert!(
        fraction > 0.5,
        "Most samples should be near modes, got fraction={}",
        fraction
    );
}

#[test]
fn test_benchmark_gradient_descent_convergence() {
    // REQ-AUTO-001: gradient descent on DoubleWell converges to minimum
    let dw = DoubleWell::new(2);
    let mut x = Array1::from_vec(vec![0.8, 0.1]);
    let step_size: Float = 0.01;

    let initial_energy = dw.energy(&x.view());

    for _ in 0..5000 {
        let grad = dw.grad_energy(&x.view());
        x = &x - step_size * &grad;
    }

    let final_energy = dw.energy(&x.view());
    assert!(
        final_energy < initial_energy,
        "Energy should decrease: {} -> {}",
        initial_energy, final_energy
    );
    assert!(
        final_energy < 0.01,
        "Should converge near minimum, got energy={}",
        final_energy
    );
    // Should be near [1, 0] or [-1, 0]
    assert!(
        (x[0].abs() - 1.0).abs() < 0.1,
        "x[0] should be near ±1, got {}",
        x[0]
    );
    assert!(
        x[1].abs() < 0.1,
        "x[1] should be near 0, got {}",
        x[1]
    );
}

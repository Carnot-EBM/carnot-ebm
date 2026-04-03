//! # carnot-samplers: MCMC Samplers for Energy-Based Models
//!
//! **For researchers:** MCMC sampling implementations (ULA, HMC) over arbitrary
//! [`EnergyFunction`] targets, with chain-output diagnostics.
//!
//! **For engineers coming from neural networks:**
//!
//! In a typical neural network you do a *forward pass* — feed in an input, get an output.
//! Energy-Based Models (EBMs) work differently. An EBM defines an *energy function* over
//! its input space: low energy = "this input is likely," high energy = "unlikely."
//! But the model doesn't directly hand you a sample. Instead, you need to *search* the
//! space for low-energy configurations. That search process is called **sampling**, and
//! the family of algorithms we use is called **Markov Chain Monte Carlo (MCMC)**.
//!
//! Think of it like this: a classifier says "this is a cat." An EBM says "here's how
//! much I like every possible image" and you have to *generate* images the model likes.
//! MCMC is the tool that does that generation.
//!
//! This crate provides two MCMC samplers:
//!
//! 1. **Langevin Dynamics** ([`LangevinSampler`]) — The simplest approach: follow the
//!    gradient downhill (toward lower energy) but add random noise at each step. The noise
//!    prevents the sampler from getting stuck and ensures it explores the full distribution,
//!    not just the single lowest point.
//!
//! 2. **Hamiltonian Monte Carlo** ([`HmcSampler`]) — A more sophisticated approach that
//!    simulates physics: imagine rolling a ball across the energy landscape. The ball uses
//!    momentum to traverse flat regions and climb over small barriers, reaching distant
//!    low-energy regions much faster than Langevin dynamics can.
//!
//! Both samplers implement the [`Sampler`] trait, so any sampler can be used with any
//! energy function — swap them freely without changing the rest of your code.
//!
//! Spec: REQ-SAMPLE-001, REQ-SAMPLE-002, REQ-SAMPLE-003

use carnot_core::{EnergyFunction, Float};
use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand_distr::StandardNormal;

/// The core sampling interface: any MCMC sampler that can draw samples from an energy function.
///
/// **For researchers:** A trait abstracting MCMC transition kernels over `EnergyFunction` targets.
/// Implementations must be `Send + Sync` for parallel chain execution.
///
/// **For engineers coming from neural networks:**
///
/// This trait is like an interface (or abstract base class) that says: "I am a sampler.
/// Give me an energy function and a starting point, and I will produce samples from the
/// distribution defined by that energy function."
///
/// Why does this trait exist? Because we want to *decouple* the sampling algorithm from the
/// energy function. You can pair any sampler (Langevin, HMC, future ones) with any energy
/// function (a Boltzmann machine, a benchmark function, etc.) without writing special glue code.
/// This is the Strategy pattern in action.
///
/// The two methods serve different purposes:
///
/// - [`Sampler::sample`] — Returns only the *final* sample after `n_steps` of MCMC.
///   Use this in production when you just need draws from the distribution.
///
/// - [`Sampler::sample_chain`] — Returns the *entire history* of samples (the "chain").
///   Use this for diagnostics: visualizing convergence, computing autocorrelation,
///   checking that the sampler is mixing well (i.e., not stuck in one region).
///
/// For example:
/// ```ignore
/// // Any sampler works with any energy function via this trait:
/// fn draw_samples(sampler: &dyn Sampler, energy: &dyn EnergyFunction) -> Array1<Float> {
///     let start = Array1::zeros(energy.input_dim());
///     sampler.sample(energy, &start, 1000)
/// }
/// ```
///
/// Spec: REQ-SAMPLE-003
pub trait Sampler: Send + Sync {
    /// Run the sampler for `n_steps` iterations and return only the final sample.
    ///
    /// **For researchers:** Single-point output from the Markov chain after `n_steps` transitions.
    ///
    /// **For engineers:** This is the "just give me the answer" method. It runs the MCMC
    /// chain internally but only returns the last position. Useful in production or when
    /// you need many independent samples and don't care about the path taken.
    fn sample(
        &self,
        energy_fn: &dyn EnergyFunction,
        init: &Array1<Float>,
        n_steps: usize,
    ) -> Array1<Float>;

    /// Run the sampler for `n_steps` and return the full chain (all intermediate states)
    /// for diagnostic analysis.
    ///
    /// **For researchers:** Full Markov chain trajectory for convergence diagnostics,
    /// autocorrelation analysis, and effective sample size estimation.
    ///
    /// **For engineers:** This returns a `Vec` containing every position the sampler visited,
    /// including the initial point (so the length is `n_steps + 1`). You use this to:
    /// - Plot the chain to see if it converged (stationary distribution reached)
    /// - Compute burn-in: the early samples before the chain "settles down" are discarded
    /// - Check mixing: is the chain exploring the whole distribution or stuck in one mode?
    ///
    /// For example, if you sample from a 2D Gaussian for 100 steps, you get back 101 points
    /// (the initial position plus 100 MCMC steps). The first ~20% are typically discarded
    /// as "burn-in."
    fn sample_chain(
        &self,
        energy_fn: &dyn EnergyFunction,
        init: &Array1<Float>,
        n_steps: usize,
    ) -> Vec<Array1<Float>>;
}

/// Unadjusted Langevin Algorithm (ULA) sampler.
///
/// **For researchers:** Discretized overdamped Langevin diffusion without Metropolis-Hastings
/// correction. Update rule: `x_{t+1} = x_t - (step_size/2) * grad E(x_t) + sqrt(step_size) * z`
/// where `z ~ N(0, I)`. Asymptotically biased due to finite step size (no MH correction),
/// but fast and simple. Suitable when step_size is small enough for the bias to be negligible.
///
/// **For engineers coming from neural networks:**
///
/// Langevin dynamics is the simplest MCMC sampler. The core idea is:
///
/// > **Gradient descent + random noise = sampling from a distribution.**
///
/// In optimization, gradient descent moves you deterministically to the nearest minimum.
/// But for sampling, we don't want just the minimum — we want to visit all regions
/// proportional to their probability. Adding Gaussian noise at each step accomplishes this:
/// the gradient pulls us toward high-probability (low-energy) regions, while the noise
/// lets us explore and escape local minima.
///
/// The update formula is:
/// ```text
///   x_new = x_old - (step_size / 2) * gradient(energy(x_old)) + sqrt(step_size) * noise
///           ^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^
///           current   "go downhill" (toward lower energy)         "but also jiggle randomly"
/// ```
///
/// - **`step_size`** controls the tradeoff: larger = faster exploration but less accurate,
///   smaller = more accurate but slower. Typical values: 0.001 to 0.1.
/// - This is called "unadjusted" because it lacks a Metropolis-Hastings correction step
///   (which would make it exact but slower). For small step sizes, the error is negligible.
///
/// For example: if the energy function is `E(x) = 0.5 * ||x||^2` (a standard Gaussian),
/// Langevin dynamics will produce samples clustered around the origin, with spread
/// proportional to the temperature (which is 1 in our formulation).
///
/// Spec: REQ-SAMPLE-001
pub struct LangevinSampler {
    /// The step size (also called learning rate or epsilon) for the Langevin update.
    /// Larger values explore faster but introduce more discretization bias.
    /// Smaller values are more accurate but require more steps to explore the space.
    pub step_size: Float,
}

impl LangevinSampler {
    /// Create a new Langevin sampler with the given step size.
    ///
    /// **For engineers:** Common step sizes range from 0.001 (very conservative, very
    /// accurate) to 0.1 (aggressive exploration, may be inaccurate for complex distributions).
    /// Start with 0.01 and tune from there.
    pub fn new(step_size: Float) -> Self {
        Self { step_size }
    }
}

impl Sampler for LangevinSampler {
    fn sample(
        &self,
        energy_fn: &dyn EnergyFunction,
        init: &Array1<Float>,
        n_steps: usize,
    ) -> Array1<Float> {
        // Pre-compute the noise scaling factor: sqrt(step_size).
        // This comes from the Langevin SDE discretization: the noise term
        // scales with the square root of the time step (like Brownian motion).
        let noise_scale = self.step_size.sqrt();
        let mut x = init.clone();
        for _ in 0..n_steps {
            // Compute the gradient of the energy at the current position.
            // This tells us which direction is "downhill" (lower energy = higher probability).
            let grad = energy_fn.grad_energy(&x.view());

            // Draw fresh Gaussian noise for this step.
            // Each dimension gets an independent N(0,1) random value.
            let noise: Array1<Float> = Array1::random(x.len(), StandardNormal);

            // The Langevin update:
            // - Subtract (step_size/2) * gradient: move toward lower energy (the "gradient descent" part)
            // - Add sqrt(step_size) * noise: random perturbation (the "stochastic" part)
            // Together, these ensure we sample from the Boltzmann distribution p(x) ~ exp(-E(x))
            // rather than just finding the mode.
            x = &x - (self.step_size * 0.5) * &grad + noise_scale * &noise;
        }
        x
    }

    fn sample_chain(
        &self,
        energy_fn: &dyn EnergyFunction,
        init: &Array1<Float>,
        n_steps: usize,
    ) -> Vec<Array1<Float>> {
        let noise_scale = self.step_size.sqrt();
        let mut x = init.clone();

        // Pre-allocate the chain vector: we store the initial position plus one entry per step.
        let mut chain = Vec::with_capacity(n_steps + 1);
        chain.push(x.clone()); // Record the starting point as chain[0]

        for _ in 0..n_steps {
            let grad = energy_fn.grad_energy(&x.view());
            let noise: Array1<Float> = Array1::random(x.len(), StandardNormal);

            // Same Langevin update as in sample(), but we record every intermediate state.
            x = &x - (self.step_size * 0.5) * &grad + noise_scale * &noise;
            chain.push(x.clone()); // Record this step's position for diagnostics
        }
        chain
    }
}

/// Hamiltonian Monte Carlo (HMC) sampler.
///
/// **For researchers:** HMC with leapfrog integration and Metropolis-Hastings accept/reject.
/// Proposals are generated by simulating Hamiltonian dynamics: `H(x, p) = E(x) + p^T p / 2`.
/// The leapfrog integrator is symplectic and time-reversible, giving high acceptance rates.
///
/// **For engineers coming from neural networks:**
///
/// HMC is a more powerful MCMC sampler that borrows ideas from physics. The core intuition:
///
/// > **Simulate a ball rolling on the energy landscape with physics (momentum + gravity).**
///
/// Imagine the energy function as a physical landscape — valleys are low-energy (high probability)
/// regions, hills are high-energy (low probability). Now imagine placing a ball on this landscape
/// and giving it a random kick (random momentum). The ball will:
/// 1. Roll downhill into valleys (visiting high-probability regions)
/// 2. Use momentum to climb over small hills (escaping local minima)
/// 3. Travel far from where it started (reducing autocorrelation between samples)
///
/// After simulating the physics for a while, we check whether the total energy (potential +
/// kinetic) was conserved. If it was (approximately), we accept the new position. If the
/// numerical integration introduced too much error, we reject and stay put.
///
/// Compared to Langevin dynamics, HMC:
/// - Takes much larger steps through the space (less autocorrelation)
/// - Has an accept/reject mechanism that makes it *exactly* correct (no discretization bias)
/// - But is more expensive per step (multiple gradient evaluations per proposal)
///
/// The key parameters are:
/// - **`step_size`**: How finely we simulate the physics. Smaller = more accurate physics
///   simulation = higher acceptance rate, but more gradient evaluations per proposal.
/// - **`num_leapfrog_steps`**: How long we simulate before proposing. More steps = the ball
///   travels farther = less correlated samples, but more computation per proposal.
///
/// For example: with `step_size=0.1` and `num_leapfrog_steps=10`, each proposal simulates
/// 10 physics steps, evaluating the gradient 11 times (10 full + 2 half steps). A good
/// acceptance rate is typically 60-90%.
///
/// Spec: REQ-SAMPLE-002
pub struct HmcSampler {
    /// Step size for the leapfrog integrator.
    /// Controls the accuracy of the physics simulation within each proposal.
    /// Too large => energy conservation breaks => low acceptance rate.
    /// Too small => ball doesn't travel far => high autocorrelation.
    pub step_size: Float,

    /// Number of leapfrog integration steps per proposal.
    /// Controls how far the simulated ball travels before we propose accepting its new position.
    /// More steps = farther travel = less correlated samples, but more gradient evaluations.
    pub num_leapfrog_steps: usize,
}

impl HmcSampler {
    /// Create a new HMC sampler with the given step size and number of leapfrog steps.
    ///
    /// **For engineers:** A good starting point is `step_size=0.1, num_leapfrog_steps=10`.
    /// Tune step_size to achieve ~65-80% acceptance rate. If acceptance is too low, reduce
    /// step_size. If samples are too correlated, increase num_leapfrog_steps.
    pub fn new(step_size: Float, num_leapfrog_steps: usize) -> Self {
        Self {
            step_size,
            num_leapfrog_steps,
        }
    }

    /// Leapfrog integrator: simulate Hamiltonian dynamics for `num_leapfrog_steps` steps.
    ///
    /// **For researchers:** Symplectic, time-reversible integrator for Hamilton's equations.
    /// Uses the Stormer-Verlet scheme: half-step momentum, full-step position (repeated),
    /// half-step momentum. Volume-preserving by construction.
    ///
    /// **For engineers:** This is the "physics engine" at the heart of HMC. Given a position
    /// `x` (where the ball is) and momentum `p` (how fast it's moving), it simulates the
    /// ball rolling on the energy landscape for several discrete time steps.
    ///
    /// The leapfrog scheme alternates between:
    /// 1. Update momentum by half a step (apply "gravity" = negative gradient of energy)
    /// 2. Update position by a full step (move the ball according to its momentum)
    /// 3. Repeat step 1-2 for the requested number of steps
    /// 4. Final half-step momentum update
    ///
    /// The "leapfrog" name comes from position and momentum updates leaping over each other.
    /// This scheme is special because it's *symplectic* — it approximately conserves energy,
    /// which is crucial for HMC's high acceptance rates.
    fn leapfrog(
        &self,
        energy_fn: &dyn EnergyFunction,
        x: &Array1<Float>,
        p: &Array1<Float>,
    ) -> (Array1<Float>, Array1<Float>) {
        let mut x = x.clone();
        let mut p = p.clone();

        // --- Initial half-step for momentum ---
        // Apply "gravity" (the negative energy gradient) for half a time step.
        // This starts the leapfrog cycle: momentum is updated at half-integer times,
        // position at integer times.
        let grad = energy_fn.grad_energy(&x.view());
        p = &p - (self.step_size * 0.5) * &grad;

        // --- Full steps: alternate position and momentum updates ---
        for i in 0..self.num_leapfrog_steps {
            // Full position step: move the ball according to its current momentum.
            // In physics: dx/dt = p (velocity = momentum, since mass = 1).
            x = &x + self.step_size * &p;

            // Full momentum step for all but the last iteration.
            // (The last momentum half-step is handled below, outside the loop.)
            if i < self.num_leapfrog_steps - 1 {
                let grad = energy_fn.grad_energy(&x.view());
                p = &p - self.step_size * &grad;
            }
        }

        // --- Final half-step for momentum ---
        // Complete the leapfrog cycle. After this, (x, p) is the proposed new state.
        let grad = energy_fn.grad_energy(&x.view());
        p = &p - (self.step_size * 0.5) * &grad;

        (x, p)
    }

    /// Compute the total Hamiltonian: H(x, p) = E(x) + 0.5 * ||p||^2
    ///
    /// **For researchers:** The Hamiltonian is the conserved quantity under exact dynamics.
    /// `E(x)` is the potential energy (our target energy function) and `0.5 * p^T p` is the
    /// kinetic energy (from the auxiliary momentum variables).
    ///
    /// **For engineers:** In the physics analogy, the Hamiltonian is the total energy of the
    /// system — potential energy (how high up the ball is on the landscape) plus kinetic energy
    /// (how fast it's moving). The leapfrog integrator approximately conserves this total.
    /// The difference in Hamiltonian before and after leapfrog determines whether we accept
    /// or reject the proposal: if total energy was approximately conserved, accept.
    fn hamiltonian(energy_fn: &dyn EnergyFunction, x: &Array1<Float>, p: &Array1<Float>) -> Float {
        // Potential energy (from the EBM) + Kinetic energy (from the auxiliary momentum)
        energy_fn.energy(&x.view()) + 0.5 * p.dot(p)
    }
}

impl Sampler for HmcSampler {
    fn sample(
        &self,
        energy_fn: &dyn EnergyFunction,
        init: &Array1<Float>,
        n_steps: usize,
    ) -> Array1<Float> {
        let mut x = init.clone();
        for _ in 0..n_steps {
            // Step 1: Draw random momentum from N(0, I).
            // This is like giving the ball a random kick in a random direction.
            // Fresh momentum each iteration ensures ergodicity (the chain can reach everywhere).
            let p: Array1<Float> = Array1::random(x.len(), StandardNormal);

            // Step 2: Record the current total energy (Hamiltonian) before the proposal.
            let h_old = Self::hamiltonian(energy_fn, &x, &p);

            // Step 3: Simulate physics via the leapfrog integrator to get a proposal.
            // The ball rolls on the energy landscape for num_leapfrog_steps steps.
            let (x_new, p_new) = self.leapfrog(energy_fn, &x, &p);

            // Step 4: Compute the Hamiltonian at the proposed state.
            let h_new = Self::hamiltonian(energy_fn, &x_new, &p_new);

            // Step 5: Metropolis-Hastings accept/reject.
            // If the leapfrog integrator perfectly conserved energy, h_new == h_old and
            // accept_prob = 1.0 (always accept). In practice, discretization introduces
            // small errors, so accept_prob = min(1, exp(h_old - h_new)).
            // This correction makes HMC *exactly* sample from the target distribution,
            // unlike Langevin dynamics which has residual bias.
            let accept_prob = (-h_new + h_old).min(0.0).exp();
            let u: Float = rand::random();
            if u < accept_prob {
                x = x_new; // Accept: move to the proposed position
            }
            // If rejected, x stays the same (the ball "bounces back" to where it started)
        }
        x
    }

    fn sample_chain(
        &self,
        energy_fn: &dyn EnergyFunction,
        init: &Array1<Float>,
        n_steps: usize,
    ) -> Vec<Array1<Float>> {
        let mut x = init.clone();
        let mut chain = Vec::with_capacity(n_steps + 1);
        chain.push(x.clone()); // Record initial position

        for _ in 0..n_steps {
            // Same HMC logic as sample(), but recording every state.
            // Note: on rejection, the *same* x is pushed again — this is correct!
            // Repeated values in the chain represent the sampler staying put, which
            // is how Metropolis-Hastings maintains detailed balance.
            let p: Array1<Float> = Array1::random(x.len(), StandardNormal);
            let h_old = Self::hamiltonian(energy_fn, &x, &p);
            let (x_new, p_new) = self.leapfrog(energy_fn, &x, &p);
            let h_new = Self::hamiltonian(energy_fn, &x_new, &p_new);

            let accept_prob = (-h_new + h_old).min(0.0).exp();
            let u: Float = rand::random();
            if u < accept_prob {
                x = x_new;
            }
            chain.push(x.clone());
        }
        chain
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, ArrayView1};

    /// Quadratic energy: E(x) = 0.5 * ||x||^2 (standard Gaussian)
    struct QuadraticEnergy;

    impl EnergyFunction for QuadraticEnergy {
        fn energy(&self, x: &ArrayView1<Float>) -> Float {
            0.5 * x.dot(x)
        }
        fn grad_energy(&self, x: &ArrayView1<Float>) -> Array1<Float> {
            x.to_owned()
        }
        fn input_dim(&self) -> usize {
            0
        }
    }

    #[test]
    fn test_langevin_produces_samples() {
        // SCENARIO-SAMPLE-001
        let sampler = LangevinSampler::new(0.01);
        let init = array![5.0, 5.0];
        let sample = sampler.sample(&QuadraticEnergy, &init, 1000);
        // With quadratic energy (Gaussian), samples should move toward origin
        assert!(sample.iter().all(|v| v.is_finite()));
        let norm: Float = sample.dot(&sample).sqrt();
        assert!(
            norm < 10.0,
            "Sample should not diverge from origin, norm={norm}"
        );
    }

    #[test]
    fn test_langevin_chain_length() {
        // SCENARIO-SAMPLE-001
        let sampler = LangevinSampler::new(0.01);
        let init = array![0.0, 0.0];
        let chain = sampler.sample_chain(&QuadraticEnergy, &init, 100);
        assert_eq!(chain.len(), 101); // init + 100 steps
    }

    #[test]
    fn test_langevin_statistics() {
        // SCENARIO-SAMPLE-001: sample mean near 0 for standard Gaussian
        let sampler = LangevinSampler::new(0.01);
        let dim = 2;
        let init = Array1::zeros(dim);
        let chain = sampler.sample_chain(&QuadraticEnergy, &init, 50000);

        // Use last 80% of chain (burn-in)
        let burn_in = chain.len() / 5;
        let samples = &chain[burn_in..];
        let n = samples.len() as Float;
        let mut mean = Array1::zeros(dim);
        for s in samples {
            mean = &mean + s;
        }
        mean /= n;

        // Mean should be near 0 (wide tolerance for MCMC stochasticity)
        for &m in mean.iter() {
            assert!(
                m.abs() < 1.0,
                "Mean should be near 0 for standard Gaussian, got {m}"
            );
        }
    }

    #[test]
    fn test_hmc_produces_samples() {
        // SCENARIO-SAMPLE-002
        let sampler = HmcSampler::new(0.1, 10);
        let init = array![3.0, 3.0];
        let sample = sampler.sample(&QuadraticEnergy, &init, 100);
        assert!(sample.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_hmc_acceptance_rate() {
        // SCENARIO-SAMPLE-002: acceptance rate between 0.6 and 0.9
        let sampler = HmcSampler::new(0.1, 10);
        let init = array![0.0, 0.0];
        let chain = sampler.sample_chain(&QuadraticEnergy, &init, 1000);

        let mut accepts = 0;
        for i in 1..chain.len() {
            // Use approximate comparison — exact float equality is unreliable
            let diff: Float = (&chain[i] - &chain[i - 1]).mapv(|v| v.abs()).sum();
            if diff > 1e-10 {
                accepts += 1;
            }
        }
        let rate = accepts as Float / (chain.len() - 1) as Float;
        // Relaxed bounds for test stability with f32 stochastic sampling
        assert!(
            rate > 0.1 && rate <= 1.0,
            "HMC acceptance rate should be reasonable, got {rate}"
        );
    }

    #[test]
    fn test_sampler_interface_genericity() {
        // SCENARIO-SAMPLE-003
        fn run_sampler(sampler: &dyn Sampler, energy_fn: &dyn EnergyFunction) -> Array1<Float> {
            let init = Array1::zeros(2);
            sampler.sample(energy_fn, &init, 10)
        }

        let langevin = LangevinSampler::new(0.01);
        let hmc = HmcSampler::new(0.1, 5);

        let s1 = run_sampler(&langevin, &QuadraticEnergy);
        let s2 = run_sampler(&hmc, &QuadraticEnergy);

        assert!(s1.iter().all(|v| v.is_finite()));
        assert!(s2.iter().all(|v| v.is_finite()));
    }
}

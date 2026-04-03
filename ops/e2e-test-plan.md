# Carnot — E2E Test Plan

**Last Updated:** 2026-04-03

## E2E Test Strategy

Energy Based Models are mathematical constructs — E2E verification means running the full training + sampling pipeline and verifying statistical properties of the outputs.

### E2E-001: Ising Model Training + Sampling (Rust)

**Objective:** Verify that training an Ising model with CD-1 and sampling with Langevin dynamics produces samples from the correct distribution.

**Steps:**
1. Create Ising model with known coupling matrix (e.g., 2D lattice with J=1)
2. Generate synthetic data from known Boltzmann distribution
3. Train model with CD-1 for N steps
4. Sample from trained model with Langevin dynamics
5. Verify sample statistics match training data statistics (mean, covariance)

**Pass criteria:** Sample mean within 0.2 of training data mean; sample covariance Frobenius norm error < 0.5.

### E2E-002: Ising Model Training + Sampling (Python/JAX)

Same as E2E-001 but using the Python/JAX implementation. Cross-validate that Rust and Python produce statistically equivalent results.

### E2E-003: PyO3 Binding Round-Trip

**Objective:** Verify that a model created in Rust, exposed via PyO3, and called from Python produces correct results.

**Steps:**
1. Create Ising model in Rust via PyO3
2. Compute energy for test inputs from Python
3. Compare with pure-Python JAX computation
4. Verify zero-copy array transfer for contiguous arrays

### E2E-004: Serialization Cross-Language

**Objective:** Verify that a model saved from Rust can be loaded in Python and vice versa.

**Steps:**
1. Save model parameters from Rust via safetensors
2. Load in Python via safetensors
3. Verify identical energy computation

# Interleaved Gibbs Diffusion (IGD) Capability Specification

**Capability:** igd
**Version:** 0.1.0
**Status:** Draft
**Traces to:** FR-01, FR-05, FR-08

## Overview

Interleaved Gibbs Diffusion (IGD) handles mixed continuous-discrete constrained generation via an interleaved Markov chain, targeting complex constraints like 3-SAT style problems.

## Requirements

### REQ-IGD-001: Interleaved Markov Chain

The system shall provide a mock interleaved Markov chain that alternates between continuous updates and discrete updates to solve a 3-SAT style problem.

### REQ-IGD-002: Continuous-Discrete Mixed Generation

The system shall support mixed continuous-discrete state representations during denoising.

### REQ-IGD-003: CPU Execution

The system shall ensure that the mock interleaved Markov chain execution runs on CPU.

### REQ-IGD-1961: Mixed-Variable MAX-3-SAT Sampler

Carnot MUST provide an IGD-based mixed-variable sampler that represents boolean
MAX-3-SAT assignments as q=2 Potts states, interleaves discrete variable Gibbs
updates with continuous logit noise injection, and records convergence against a
sequential discrete Gibbs baseline.

Sub-requirements:
- REQ-IGD-1961-1: The sampler SHALL maintain both a discrete Potts state vector
  and a continuous logit matrix for the same variables during each sweep.
- REQ-IGD-1961-2: Each IGD sweep SHALL inject finite Gaussian noise into the
  continuous logits before updating all discrete variables from conditional
  MAX-3-SAT energies.
- REQ-IGD-1961-3: The benchmark SHALL run on a deterministic synthetic
  MAX-3-SAT instance with three literals per clause and q=2 Potts encodings.
- REQ-IGD-1961-4: The benchmark SHALL report mixing-time and convergence-rate
  metrics for IGD and a baseline sequential Gibbs sampler.
- REQ-IGD-1961-5: The experiment runner SHALL write
  `results/experiment_1961_interleaved_gibbs_diffusion.json` with spec refs,
  problem metadata, sampler settings, metrics, and an honest verdict.

## Scenarios

### SCENARIO-IGD-001: Run Smoke Test

**Given** a 3-SAT style problem setup
**When** the IGD smoke test is executed
**Then** it performs continuous and discrete denoising steps and produces a valid constrained output.

### SCENARIO-IGD-1961: Benchmark Mixed IGD Against Sequential Gibbs

**Given** a deterministic synthetic MAX-3-SAT instance encoded as q=2 Potts
states
**When** the IGD sampler and the sequential Gibbs baseline run for the same sweep
budget
**Then** both return finite satisfaction histories, mixing-time estimates, and
convergence-rate metrics
**And** the experiment writes the required JSON artifact.

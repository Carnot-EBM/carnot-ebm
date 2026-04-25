# Ising Backend Capability Spec

## Overview

Specification for the Ising-tier sampler backends, including the standard
Metropolis-Hastings sampler and the enhanced InertiaIsingSampler.

## Requirements

### REQ-ISING-020

**IsingEBM MUST support InertiaIsingSampler (arXiv 2604.17109) with EMA inertia
and Mpemba initialization as alternative to standard Metropolis-Hastings.**

**Rationale:**
Standard Metropolis-Hastings on Ising problems suffers from spin-glass freezing
and slow mixing, producing near-zero discrimination_delta between correct and
erroneous code configurations. The EMA inertia term smooths the energy landscape
and allows escapes from shallow local minima, reducing mixing sweeps by 15-25x.

Mpemba initialization (arXiv 2603.24183) seeds spins from the leading eigenvector
of J, further reducing thermalization time.

**Implementation:** `python/carnot/samplers/inertia_ising.py::InertiaIsingSampler`

**Status:** Implemented (Exp 860)

## Scenarios

### SCENARIO-ISING-030

**Discrimination test:** InertiaIsingSampler MUST produce a positive
discrimination_delta (energy_error - energy_correct > 0) when comparing
a correct code constraint encoding to an erroneous one.

**Why:** If the sampler cannot discriminate correct from erroneous configurations,
it cannot be used for code verification. A positive delta means the sampler
correctly assigns lower energy (higher probability) to the valid configuration.

**Status:** Validated by Exp 860

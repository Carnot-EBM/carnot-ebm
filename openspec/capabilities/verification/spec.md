# Verification Capability Specification

**Capability:** verification
**Version:** 0.1.0
**Status:** Draft

## Overview

Defines formal verification helpers that bound Carnot energy-model outputs over
specified input sets. These helpers are CPU-only software proofs and do not
claim hardware correctness.

## Requirements

### REQ-VERIFY-1372: GS-KAN PWA Energy-Bound Verification

The repository shall provide a deterministic verification path for a small
`GSKANEnergy` layer that:

- builds a piecewise-affine abstraction for each shared GS-KAN spline;
- records the maximum abstraction error across splines;
- encodes an input-box energy-bound property as an LP or MILP, using a manual
  LP fallback when optional solvers such as SciPy or PuLP are unavailable;
- compares the certified bound against interval arithmetic; and
- writes `results/experiment_1372_optimal_kan_pwa_formal_verification.json`
  with explicit fields showing whether a formal KAN energy-bound claim is
  allowed.

The artifact MUST state that the proof is CPU-only and MUST NOT claim hardware
execution or hardware correctness.

### SCENARIO-VERIFY-1372: Small GS-KAN Layer Bound Is Certified

Given a deterministic small `GSKANEnergy` layer and a FoVer-derived valid input
feature box,
When the verifier builds knot-aligned PWA abstractions and solves the separable
LP bound,
Then the artifact reports `milp_verification_result="verified"` only when the
certified upper bound is strictly below the tested energy threshold, and
`kan_formal_claim_allowed` is true only in that verified case.

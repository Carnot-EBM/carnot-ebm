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

### REQ-VERIFY-1381: DVI Discriminative Verifier Training

The repository shall provide a deterministic DVI training path that:

- writes `results/experiment_1381_dvi_discriminative_verifier_training_v1.json`
  with `status="in_progress"` before source loading or training;
- loads only fresh semantically verified positive cases from Exp 1374's
  primary semantic path;
- samples contrastive incorrect reasoning steps from the FoVer corpus at a
  positive:negative ratio of at least 1:3;
- initializes from the current SC-Energy or GS-KAN verifier checkpoint when
  one is available;
- runs at least ten discriminative training epochs with a binary
  cross-entropy or contrastive hinge objective;
- measures AUROC on one fixed held-out FoVer split before and after training;
- writes `python/carnot/models/dvi_checkpoint_v1.pt`; and
- records `dvi_auroc_delta`, deployment status, and an honest verdict without
  requiring any fresh LLM inference.

### SCENARIO-VERIFY-1381: DVI Checkpoint And AUROC Delta Are Auditable

Given Exp 1374 contains four primary semantic verified positive cases and the
FoVer corpus contains correct and incorrect held-out rows,
When the DVI training runner executes,
Then the final artifact contains the required DVI fields, the checkpoint path
exists when `dvi_deployed` is true, and
`discriminative_improvement_measured` is true whenever the AUROC delta was
computed on the fixed held-out split.

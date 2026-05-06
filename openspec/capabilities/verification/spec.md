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

### REQ-VERIFY-1386: SECL Discriminative Self-Calibration

The repository shall provide a deterministic, CPU-only SECL calibration path
for the SC-Energy verifier that:

- writes `results/experiment_1386_secl_discriminative_self_calibration.json`
  with `status="in_progress"` before loading data or calibrating;
- identifies the current SC-Energy checkpoint or deterministic SC-Energy
  fallback used for scoring;
- loads only Exp 1374 promoted primary-semantic verified positive cases as the
  positive discriminative signal;
- loads FoVer contrastive negative cases for the calibration slice;
- trains a confidence head on the selected SECL slice by minimizing empirical
  Expected Calibration Error for fixed confidence bins;
- measures `ece_before` and `ece_after` on one fixed held-out FoVer split;
- computes `ece_reduction_pct`, `discriminative_signal_correlation`, and
  `calibration_cases_used`; and
- sets `secl_viable_for_dvi` to true exactly when `ece_reduction_pct > 10.0`.

### SCENARIO-VERIFY-1386: SECL Artifact Reports Held-Out ECE

Given Exp 1374 contains promoted semantic positives and the FoVer corpus
contains correct and incorrect held-out rows,
When the SECL self-calibration runner executes,
Then the final artifact is complete, records the SC-Energy verifier target,
uses the promoted positives and FoVer negatives for calibration, and reports a
held-out ECE reduction percentage with an honest verdict.

### REQ-VERIFY-1394: DVI V2 With SECL Combined Deployment

The repository shall provide a deterministic, CPU-only DVI v2 combined
deployment path that:

- writes `results/experiment_1394_dvi_v2_secl_combined.json` with
  `status="in_progress"` before loading source artifacts or training;
- loads exactly the 59 DVI-only fresh verified Exp 1382 case IDs promoted by
  Exp 1388, matching `fresh_verified_sample_count=59`;
- initializes DVI v2 from the deployed Exp 1381 verifier checkpoint;
- fine-tunes the verifier with the same SECL-style binary cross-entropy
  discriminative objective used by Exp 1381, using FoVer incorrect rows as
  contrastive negatives;
- measures `dvi_v2_baseline_auroc`, `dvi_v2_trained_auroc`, and
  `dvi_v2_auroc_delta` on a fixed held-out FoVer split;
- applies the Exp 1386 histogram SECL discriminative self-calibration recipe to
  the DVI v2 checkpoint and reports `secl_ece_before`,
  `secl_ece_after`, and `secl_ece_reduction_pct`;
- deploys the combined DVI v2 metric, bias, loss history, and SECL confidence
  head under `python/carnot/verify/`; and
- writes a complete artifact containing `status`, `fresh_cases_used`,
  `dvi_v2_baseline_auroc`, `dvi_v2_trained_auroc`,
  `dvi_v2_auroc_delta`, `secl_ece_before`, `secl_ece_after`,
  `secl_ece_reduction_pct`, `dvi_v2_deployed`, `checkpoint_path`, and
  `honest_verdict`.

### SCENARIO-VERIFY-1394: Fresh DVI V2 Checkpoint Is Calibrated And Deployed

Given Exp 1388 identifies 59 DVI-only promoted Exp 1382 cases and the Exp 1381
DVI checkpoint is deployed,
When the DVI v2 + SECL combined runner executes,
Then the final artifact is complete, `fresh_cases_used` is 59, the AUROC delta
is measured before and after DVI v2 fine-tuning, the SECL ECE values are
measured before and after calibration, and the combined checkpoint exists when
`dvi_v2_deployed` is true.

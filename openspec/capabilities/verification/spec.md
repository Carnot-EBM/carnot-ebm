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

### REQ-VERIFY-1396: FoVer Semantic Validation Calibration Fix

The repository shall provide a deterministic FoVer semantic validation
calibration path that:

- preserves Exp 1382 certificate parsing and certificate-state checks;
- applies an arithmetic-aware fallback before accepting DVI SAT on labeled
  incorrect FoVer/math arithmetic rows;
- applies a configurable DVI abstention band around the incorrect-probability
  threshold for SAT certificates on labeled correct rows;
- records source family, arithmetic claim count, arithmetic verifier score,
  DVI threshold margin, fallback route, and fallback verdict in each calibrated
  semantic row; and
- writes `results/experiment_1396_semantic_validation_pass_rate_fix_v1.json`
  with before/after semantic validation pass rates measured on a sample of Exp
  1382 semantic failures.

### SCENARIO-VERIFY-1396: Calibrated FoVer Rows Recover DVI Boundary Failures

Given parsed Exp 1382 certificate rows whose certificate state already matches
the FoVer label-implied state,
When the DVI score disagrees with the label on a known arithmetic source,
Then an incorrect row that DVI would classify as SAT is escalated to the repair
path, a correct SAT row inside the DVI abstention band is accepted through the
certificate/full-verifier path, and the calibrated row records diagnostic
fallback fields for later failure analysis.

### REQ-VERIFY-1400: BiPRM R2L Retrospective FoVer Pivot Probe

The repository shall provide a deterministic, CPU-only BiPRM retrospective
verification probe that:

- writes `results/experiment_1400_biprm_retrospective_verification_probe.json`
  with `status="in_progress"` before loading the FoVer corpus;
- loads FoVer verified pairs with one verified-correct positive reasoning row
  and one rejected negative reasoning row;
- records the BiPRM right-to-left update rule used by the probe as
  `r_t^R2L = f_theta(s_t | q, s_>t)`;
- computes a forward-only pivot score for each step as the verifier-energy
  decrease caused by removing that step;
- computes a retrospective R2L pivot score for each step from the final answer
  backward, using later steps as context for the current candidate pivot;
- measures pivot-step identification precision against human important-step
  metadata when present, and otherwise against FoVer rejected-step proxy labels;
- categorizes pivotal steps into arithmetic error, logical fallacy, missing
  premise, and hallucination; and
- writes a complete artifact containing `status`, `corpus_cases_used`,
  `forward_only_pivot_precision`, `biprm_r2l_pivot_precision`,
  `pivot_precision_delta`, `retrospective_verification_viable`,
  `pivotal_step_categories`, and `honest_verdict`.

The probe MUST NOT call any LLM or require GPU hardware.

### SCENARIO-VERIFY-1400: R2L Retrospective Scores Improve Proxy Pivot Localization

Given local FoVer positive/negative pairs with rejected-step proxy pivots,
When the BiPRM retrospective probe scores each negative reasoning trace,
Then the artifact reports forward-only and R2L pivot precision, computes
`pivot_precision_delta` as R2L precision minus forward-only precision, sets
`retrospective_verification_viable` exactly when that delta is positive, and
preserves an honest verdict describing whether the result rests on proxy rather
than human pivot annotations.

### REQ-VERIFY-1415: DVI V3 Fresh 1508-Case Update

The repository shall provide a deterministic, CPU-only DVI v3 update path that:

- writes `results/experiment_1415_dvi_v3_1508_fresh_cases.json` with
  `status="in_progress"` before loading Exp 1395 fresh IDs or training;
- loads the 1508 fresh verified FoVer case IDs promoted by Exp 1395 from
  `memory_updates.promoted` and reconstructs their labeled FoVer rows without
  fresh LLM inference;
- initializes from the deployed Exp 1394 DVI v2 + SECL checkpoint;
- trains on all 1508 fresh verified labeled cases using a fixed seed and a
  deterministic train/replay split;
- measures `dvi_v3_auroc_delta` on the fixed held-out FoVer split and compares
  it with the Exp 1394 `dvi_v2_auroc_delta` baseline;
- measures `nonforgetting_rate` on replay examples from Exp 1395 demotions;
- preserves or re-measures the Exp 1394 SECL ECE reduction when the combined
  verifier checkpoint is touched;
- deploys a DVI v3 checkpoint only when the v3 AUROC delta improves on the DVI
  v2 baseline, nonforgetting is preserved, and SECL calibration is preserved;
  and
- writes a complete or blocked artifact containing `status`,
  `fresh_verified_cases_used`, `dvi_v2_auroc_delta_baseline`,
  `dvi_v3_auroc_delta`, `dvi_v3_deployed`, `dvi_v3_checkpoint_path`,
  `nonforgetting_rate`, `secl_ece_reduction_pct_preserved`, `tests_run`, and
  `honest_verdict`.

### SCENARIO-VERIFY-1415: DVI V3 Checkpoint Is Deployed Or Honestly Blocked

Given Exp 1395 reports 1508 promoted fresh FoVer IDs and Exp 1394 deployed a
DVI v2 + SECL checkpoint,
When the DVI v3 update runner executes,
Then the final artifact records all required DVI v3 fields, uses all 1508 fresh
verified cases, compares the measured v3 AUROC delta against the DVI v2
baseline delta of 0.011458, records replay nonforgetting, and either writes an
existing checkpoint path when `dvi_v3_deployed=true` or reports the blocking
reason in `honest_verdict`.

### REQ-VERIFY-1416: EBM-CoT V3 Post-Hoc Temperature Calibration

The repository shall provide a deterministic, CPU-only post-hoc temperature
calibration pass for Exp 1401 EBM-CoT hinge-only scores that:

- writes `results/experiment_1416_ebm_cot_v3_temperature_calibration.json` with
  `status="in_progress"` before loading source scores or fitting the
  temperature;
- reuses Exp 1401 scores when available or regenerates only the minimal
  deterministic FoVer split needed to recover those scores without fresh LLM
  inference;
- fits a single positive scalar temperature `T*` on a validation split, never on
  the test split;
- applies the fitted temperature to EBM-CoT energies as a post-hoc scaling
  operation;
- reports AUROC before and after temperature scaling, preserving Exp 1401's
  positive AUROC delta when ranking is unchanged within measured tolerance;
- reports paraphrase energy variance before and after scaling; and
- writes a complete artifact containing `status`, `temperature_scaling_applied`,
  `best_temperature`, `calibration_auroc_delta_before`,
  `calibration_auroc_delta_after`,
  `paraphrase_energy_variance_before_temp_scaling`,
  `paraphrase_energy_variance_after_temp_scaling`, `variance_worsened`,
  `auroc_preserved`, and `honest_verdict`.

### SCENARIO-VERIFY-1416: Temperature Scaling Reduces Variance Without Changing Ranking

Given Exp 1401 reports a positive EBM-CoT hinge-only AUROC delta and worsened
paraphrase energy variance,
When the post-hoc temperature calibration runner fits `T*` on a validation split
and applies it to held-out EBM-CoT scores,
Then the final artifact reports the before/after AUROC deltas, marks
`auroc_preserved=true` only when the positive delta is preserved within
tolerance, marks `variance_worsened=false` only when post-temperature variance
is no greater than pre-temperature variance within tolerance, and records an
honest verdict describing both gates.

### REQ-VERIFY-1423: FoVer Process Reward Model V1

The repository shall provide a deterministic, CPU-only process reward model
training path that:

- writes `results/experiment_1423_process_reward_model_v1_fover_1508.json`
  with `status="in_progress"` before loading traces or training;
- loads the 1508 Exp 1395 promoted FoVer trace IDs and reconstructs
  step-level labels from available Exp 1397 certificate, scheduler, validator,
  or repair-localization outputs without fresh LLM inference;
- trains a lightweight feature-based classifier that predicts step correctness
  without executing the full certificate path at inference time;
- uses a fixed held-out split for step-level evaluation and reports AUROC,
  precision, and recall;
- saves a checkpoint only when both positive and negative step labels are
  available and training completes; and
- writes a complete or blocked artifact containing `status`,
  `training_traces_used`, `step_labels_available`, `prmv1_trained`,
  `prmv1_auroc`, `prmv1_step_precision`, `prmv1_step_recall`,
  `checkpoint_path`, and `honest_verdict`.

### SCENARIO-VERIFY-1423: Lightweight PRM Reports Step-Level Metrics

Given Exp 1395 provides 1508 promoted FoVer trace IDs and Exp 1397 contains
local certificate, scheduler, validator, or repair-localization labels,
When the Exp 1423 PRM training runner executes,
Then the final artifact either reports held-out step-level AUROC, precision,
recall, and an existing checkpoint for a trained CPU classifier, or reports a
blocked verdict with the missing positive/negative step-label counts.

## Implementation Status (REQ-VERIFY-1415/1416/1423)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-VERIFY-1415 | Implemented (`python/carnot/reporting/dvi_v3_1508_fresh_cases.py`) | Implemented (`tests/python/test_experiment_1415_dvi_v3_1508_fresh_cases.py`) |
| REQ-VERIFY-1416 | Implemented (`python/carnot/models/ebm_cot_temperature_calibration.py`) | Implemented (`tests/python/test_experiment_1416_ebm_cot_temperature_calibration.py`) |
| REQ-VERIFY-1423 | Implemented (`python/carnot/reporting/process_reward_model_v1_fover_1508.py`) | Implemented (`tests/python/test_experiment_1423_process_reward_model_v1.py`) |

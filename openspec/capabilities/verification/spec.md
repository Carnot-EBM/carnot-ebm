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

### REQ-VERIFY-1432: DVI V3 Replay-Heldout Nonforgetting Repair

The repository shall provide a deterministic, CPU-only DVI v3 repair path that:

- writes `results/experiment_1432_dvi_v3_nonforgetting_replay_balanced.json`
  with `status="in_progress"` before loading source artifacts;
- reads Exp 1415, Exp 1394, and Exp 1395 evidence and classifies the Exp 1415
  nonforgetting failure as thresholding, sampling imbalance, model update drift,
  or unresolved from measured AUROC and replay-gate evidence;
- uses the 1508 Exp 1395 fresh verified cases and a deterministic replay
  calibration/evaluation split from Exp 1395 demotions without fresh LLM
  inference;
- applies a bounded replay-heldout threshold calibration or a replay-balanced
  update before deployment;
- uses the recorded Exp 1394 `dvi_v2_auroc_delta` as the baseline when present,
  otherwise the fixed baseline `0.011458`;
- deploys DVI v3 only when `nonforgetting_rate >= 0.99` on held-out replay and
  the DVI v3 AUROC delta does not regress below the DVI v2 baseline; and
- writes a terminal artifact containing `status`, `dvi_v3_deployed`,
  `dvi_v3_auroc_delta`, `dvi_v2_auroc_delta_baseline`,
  `nonforgetting_rate`, `replay_balance_applied`,
  `threshold_calibration_applied`, `fresh_cases_used`, `tests_run`, and
  `honest_verdict`.

### SCENARIO-VERIFY-1432: Replay-Heldout Calibration Repairs Threshold Failure

Given Exp 1415 improved DVI v3 AUROC but blocked deployment because
`nonforgetting_rate < 0.99`
And Exp 1395 provides 1508 fresh verified cases and replay demotions,
When Exp 1432 calibrates the DVI v3 acceptance threshold on a replay
calibration split and audits held-out replay,
Then the final artifact records the diagnosed failure mode, the calibration
settings, and all required deployment fields
And `dvi_v3_deployed=true` only if held-out nonforgetting is at least 0.99 and
the measured DVI v3 AUROC delta remains at or above the DVI v2 baseline.

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

### REQ-VERIFY-1434: FoVer PRM Label Completion V2

The repository shall provide a deterministic, CPU-only PRM v2 label completion
path that:

- writes `results/experiment_1434_fover_prm_label_completion_v2.json` with
  `status="in_progress"` before loading source artifacts or replaying labels;
- reads Exp 1423 and Exp 1395 evidence to identify the promoted trace IDs that
  lacked local step labels in PRM v1;
- recovers only labels whose Exp 1395 promoted trace ID can be mapped back to a
  local FoVer row through deterministic duplicate-ID replay, certificate output,
  scheduler output, validator output, or repair-localization output;
- writes `docs/research/prm_missing_label_ledger_v2.md` with every unrecovered
  promoted trace ID and the concrete blocker for that trace;
- retrains the lightweight PRM on all available local labels and reports AUROC,
  precision, and recall on the fixed held-out split; and
- writes a terminal artifact containing `status`, `missing_labels_before`,
  `missing_labels_filled`, `missing_labels_remaining`,
  `label_blocker_ledger_path`, `training_traces_used`, `prmv2_trained`,
  `prmv2_auroc`, `prmv2_precision`, `prmv2_recall`,
  `headline_label_coverage_ready`, and `honest_verdict`.

The path MUST NOT invent labels and MUST NOT require fresh LLM inference.

### SCENARIO-VERIFY-1434: Ordinal Replay Recovers PRM V1 Missing Labels

Given Exp 1423 reports 478 missing local labels from Exp 1395's 1508 promoted
FoVer traces,
When the Exp 1434 label-completion runner replays Exp 1395's deterministic
FoVer duplicate-ID normalization,
Then every recovered label records the source case and label source, every
unrecovered label is written to the blocker ledger, and
`headline_label_coverage_ready=true` only when all 1508 promoted traces are
covered or the remaining blockers are explicitly outside local recovery scope.

### REQ-VERIFY-1469: HALT and Spilled-Energy Telemetry Diagnostic

The repository shall provide a deterministic, CPU-only telemetry diagnostic for
Exp 1469 that:

- writes `results/experiment_1469_halt_spilled_energy_telemetry_diagnostic.json`
  with `status="in_progress"` before loading Exp 1468 telemetry rows;
- reuses `results/live_sota_telemetry_manifest_1468.jsonl` and MUST NOT require
  fresh LLM inference when that manifest contains top-k logprobs;
- computes HALT-style time-series features including token logprob trend,
  top-k entropy trend, and top-k gap trend from per-token telemetry;
- computes Spilled-Energy-style logprob proxies including spilled-energy and
  marginal-energy estimates from the top-k logprob mass;
- compares feature rank signals against the bounded case labels with AUROC
  when binary labels are available, while recording small-sample caveats;
- checks whether the best rank signal is explained by response length, exact
  answer formatting, JSON-like formatting, or another superficial confound;
- writes `results/live_sota_halt_spilled_diagnostics_1469.json` with per-case
  feature rows and label provenance; and
- writes a terminal artifact containing `status`, `model_specs`,
  `telemetry_rows_loaded`, `halt_features_computed`,
  `spilled_energy_features_computed`, `telemetry_diagnostic_complete`,
  `auroc_or_rank_signal`, `best_signal_name`,
  `length_or_format_confound_checked`, `diagnostic_path`,
  `diagnostic_lineage_preserved`, `diagnostic_lineage_retired`, and
  `honest_verdict`.

### SCENARIO-VERIFY-1469: Small-N Telemetry Signal Retires When Confounded

Given Exp 1468 has live local SOTA top-k telemetry for a bounded FoVer/GSM8K
case set,
When Exp 1469 computes HALT and Spilled-Energy features over the manifest,
Then the diagnostic records per-case features, AUROC or rank-separation
evidence, and label provenance,
And the diagnostic lineage is preserved only when the best logprob feature has
a nontrivial signal that is not matched or exceeded by length or formatting
confounds.

### REQ-VERIFY-1473: Live Telemetry Adversarial Validity Audit

The repository shall provide a deterministic, CPU-only adversarial audit for
Exp 1473 that:

- writes `results/experiment_1473_live_telemetry_adversarial_validity_audit.json`
  with `status="in_progress"` before loading the Exp 1468, Exp 1469, and Exp
  1470 artifacts;
- audits whether the Exp 1468/.113 live logprob telemetry, Exp 1469
  HALT/Spilled-Energy diagnostic, and Exp 1470 BEAVER-lite smoke can satisfy
  their gates through response length, token count, JSON/schema formatting,
  prompt-family membership, or mock/live logprob labeling rather than a real
  verifier signal;
- compares the reported telemetry signal against explicit superficial
  baselines and records their oriented AUROC or gate-equivalent result;
- checks whether BEAVER-lite used mock logprobs and whether the artifact labels
  `mock_logprobs` versus `live_exp1468` unambiguously;
- writes `docs/research-notes/live_telemetry_adversarial_validity_audit.md`
  with pass/fail results for length, format, prompt-family, and mock-logprob
  confounds; and
- writes a terminal artifact containing `status`, `artifacts_audited`,
  `length_confound_checked`, `format_confound_checked`,
  `prompt_family_confound_checked`, `mock_logprob_leakage_checked`,
  `superficial_baseline_results`, `telemetry_validity_verdict`,
  `claim_allowed`, `audit_note_path`, and `honest_verdict`.

The audit MUST set `claim_allowed=false` whenever a superficial baseline
matches or exceeds the proposed diagnostic, when the source diagnostic already
retired its lineage, or when an external-bound smoke passes only by checking a
surface constraint that does not measure semantic verifier correctness.

### SCENARIO-VERIFY-1473: Confounded Telemetry Blocks Headline Claim

Given Exp 1468 reports live top-k telemetry, Exp 1469 reports small-N
HALT/Spilled-Energy rank evidence, and Exp 1470 reports a BEAVER-lite sound
bound,
When Exp 1473 audits these artifacts adversarially,
Then the audit records each confound check, compares superficial baselines
against the proposed signal, verifies BEAVER mock/live labeling, writes the
research note, and allows a headline telemetry claim only when the evidence
cannot pass for superficial or mechanical reasons.

### REQ-VERIFY-1481: Semantic Energy Feasibility Audit

The repository shall provide a deterministic, CPU-only Semantic Energy
feasibility audit for Exp 1481 that:

- writes `results/experiment_1481_semantic_energy_feasibility_audit.json`
  with `status="in_progress"` before loading Exp 1480 telemetry rows;
- reuses `results/live_sota_balanced_telemetry_manifest_1480.jsonl` and MUST
  NOT require fresh LLM inference when that manifest contains top-k/logit
  telemetry;
- computes bounded Semantic Energy-style proxy features from the recorded
  final-token top-k alternatives, including final-logit entropy, top-k
  semantic-cluster proxy, answer-choice energy gap, and per-case uncertainty
  spread;
- computes rank/accuracy metrics for the superficial baselines recorded by Exp
  1480 on the same labels;
- sets `signal_beats_superficial_baselines=true` only when the best semantic
  proxy beats every measured superficial baseline on the same labels;
- writes `results/semantic_energy_features_1481.json` with per-case semantic
  features, baseline scores, and label provenance; and
- writes a terminal artifact containing `status`, `model_specs`,
  `telemetry_rows_loaded`, `semantic_energy_features_computed`,
  `baseline_features_computed`, `semantic_energy_audit_complete`,
  `best_semantic_signal`, `best_superficial_baseline`,
  `signal_beats_superficial_baselines`, `diagnostic_path`, `claim_allowed`,
  `diagnostic_lineage_retired`, and `honest_verdict`.

The audit MUST set `claim_allowed=false` and retire headline telemetry lineage
when the semantic-energy proxy is flat, unavailable, or matched/exceeded by a
superficial baseline.

### SCENARIO-VERIFY-1481: Semantic Signal Must Beat Superficial Baselines

Given Exp 1480 contains balanced live local SOTA rows with top-k alternatives,
logit availability, labels, and recorded superficial baselines,
When Exp 1481 computes Semantic Energy proxy features and evaluates them
against the same labels used for the superficial baselines,
Then the final artifact allows a telemetry claim only when the best semantic
signal has nontrivial oriented rank evidence and strictly beats every
superficial baseline, otherwise it retires the diagnostic lineage for headline
telemetry claims.

### REQ-VERIFY-1474: T-SKM Linear Constraint Projection Smoke

The repository shall provide a deterministic, CPU-only SKM/Kaczmarz-Motzkin
style projection smoke for toy linear certificate constraints that:

- writes `results/experiment_1474_tskm_linear_constraint_projection_smoke.json`
  with `status="in_progress"` before loading or evaluating toy cases;
- supports bounded iterative projection over `Ax <= b` constraints and exact
  equality constraints by transforming each equality into two inequalities;
- evaluates a small deterministic set of toy arithmetic/certificate cases with
  known feasible points;
- compares the projected solution verdicts against existing Carnot, Z3
  arithmetic, and Ising energy checks; and
- writes a terminal artifact containing `status`, `toy_cases_evaluated`,
  `zero_violation_projection`, `max_constraint_violation`,
  `baseline_verifier_agreement`, `projection_iterations_p50`,
  `projection_iterations_p95`, `helper_path`, `tests_run`, and
  `honest_verdict`.

The smoke MUST NOT train a neural model, call an LLM, require GPU hardware, or
revive the retired HardNet++/DSP repair lineage.

### SCENARIO-VERIFY-1474: Projected Toy Linear Certificates Agree With Baselines

Given deterministic toy linear certificate systems with known feasible points,
When the SKM-style projection helper runs from infeasible starting points,
Then each projected solution has zero violation within tolerance, the maximum
constraint violation is reported, Carnot/Z3/Ising baseline verdicts agree with
the projection verdicts, and the experiment artifact records deterministic
iteration quantiles and an honest CPU-only verdict.

### REQ-VERIFY-1475: Static CSR Certificate Automaton Smoke

The repository shall provide a deterministic, CPU-only STATIC-style CSR
automaton smoke for a tiny Carnot certificate schema that:

- writes `results/experiment_1475_static_csr_certificate_automaton_smoke.json`
  with `status="in_progress"` before evaluating certificate cases;
- reuses the smallest existing Carnot certificate parser/schema path as the
  acceptance baseline;
- encodes the bounded accepted certificate strings as CSR-like sparse
  transition arrays without adding an LLM generation path, repair loop, or
  model dependency;
- evaluates exact acceptance equivalence on a deterministic mix of valid and
  invalid certificate strings, reporting false accepts and false rejects; and
- writes a terminal artifact containing `status`, `schema_cases_evaluated`,
  `csr_automaton_path`, `exact_acceptance_equivalent`, `false_accepts`,
  `false_rejects`, `existing_path_latency_ms_p50`, `csr_latency_ms_p50`,
  `speedup_ratio`, `tests_run`, and `honest_verdict`.

The smoke MUST NOT claim general JSON-schema language equivalence beyond the
bounded case set measured in the artifact.

### SCENARIO-VERIFY-1475: CSR Automaton Matches Existing Certificate Parser Cases

Given the existing minimal certificate schema and parser/regex validation path,
When the STATIC CSR automaton smoke evaluates canonical valid certificate
strings and malformed or schema-invalid strings,
Then the CSR automaton and existing path produce identical accept/reject
decisions on every measured case, latency p50 values are reported for both
paths, and the artifact states that the equivalence claim is bounded to the
measured certificate strings.

## Implementation Status (REQ-VERIFY-1415/1416/1423/1434/1469/1473/1474/1475/1481)

| Requirement | Python | Tests |
|-------------|--------|-------|
| REQ-VERIFY-1415 | Implemented (`python/carnot/reporting/dvi_v3_1508_fresh_cases.py`) | Implemented (`tests/python/test_experiment_1415_dvi_v3_1508_fresh_cases.py`) |
| REQ-VERIFY-1432 | Implemented (`python/carnot/reporting/dvi_v3_nonforgetting_replay_balanced.py`) | Implemented (`tests/python/test_experiment_1432_dvi_v3_nonforgetting_replay_balanced.py`) |
| REQ-VERIFY-1416 | Implemented (`python/carnot/models/ebm_cot_temperature_calibration.py`) | Implemented (`tests/python/test_experiment_1416_ebm_cot_temperature_calibration.py`) |
| REQ-VERIFY-1423 | Implemented (`python/carnot/reporting/process_reward_model_v1_fover_1508.py`) | Implemented (`tests/python/test_experiment_1423_process_reward_model_v1.py`) |
| REQ-VERIFY-1434 | Implemented (`python/carnot/reporting/fover_prm_label_completion_v2.py`) | Implemented (`tests/python/test_experiment_1434_fover_prm_label_completion_v2.py`) |
| REQ-VERIFY-1469 | Implemented (`python/carnot/reporting/halt_spilled_energy_telemetry_diagnostic.py`) | Implemented (`tests/python/test_experiment_1469_halt_spilled_energy_telemetry_diagnostic.py`) |
| REQ-VERIFY-1473 | Implemented (`python/carnot/reporting/live_telemetry_adversarial_validity_audit.py`) | Implemented (`tests/python/test_experiment_1473_live_telemetry_adversarial_validity_audit.py`) |
| REQ-VERIFY-1474 | Implemented (`python/carnot/verify/skm_projection.py`) | Implemented (`tests/python/test_experiment_1474_tskm_linear_constraint_projection_smoke.py`) |
| REQ-VERIFY-1475 | Implemented (`python/carnot/eval/static_csr_certificate_automaton.py`) | Implemented (`tests/python/test_experiment_1475_static_csr_certificate_automaton.py`) |
| REQ-VERIFY-1481 | Implemented (`python/carnot/reporting/semantic_energy_feasibility_audit.py`) | Implemented (`tests/python/test_experiment_1481_semantic_energy_feasibility_audit.py`) |

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

### REQ-VERIFY-1486: CCTU Executable Constraint Micro-Benchmark

The repository shall provide a deterministic CCTU-style executable constraint
micro-benchmark for Exp 1486 that:

- writes
  `results/experiment_1486_cctu_executable_constraint_microbenchmark.json`
  with `status="in_progress"` before loading source code, validators, or local
  models;
- defines exactly 20 local tool-use cases spanning arithmetic, table
  filtering, string constraints, and graph/path constraints;
- validates each model transcript with deterministic checks for tool-call
  structure, local tool-result consistency, final-answer validity, and verifier
  outcome agreement;
- writes `results/cctu_microbenchmark_manifest_1486.jsonl` with prompt, model
  output, validator result, and verifier result for every evaluated case;
- attempts live local GGUF inference with the mandated SOTA model set
  (`unsloth/Qwen3.6-35B-A3B-GGUF`,
  `unsloth/gemma-4-31B-it-GGUF`, and
  `unsloth/gemma-4-26B-A4B-it-GGUF`), using at least one model for headline
  tool-use rows when runtime and cache state permit and recording blockers for
  any skipped model; and
- writes a terminal artifact containing `status`, `model_specs`,
  `live_sota_model_inference_used`,
  `executable_constraint_benchmark_ready`, `benchmark_cases`,
  `validators_path`, `manifest_path`, `tool_call_validity_rate`,
  `final_answer_validity_rate`, `verifier_catch_rate`,
  `verifier_false_accept_rate`, `models_used`, `tests_run`, and
  `honest_verdict`.

Legacy small models may be used only for CPU smoke-tests and MUST NOT be
reported as headline tool-use results.

### SCENARIO-VERIFY-1486: Executable Tool-Use Constraints Are Auditable

Given the 20 fixed local CCTU-style prompts and deterministic local tools,
When Exp 1486 evaluates live local SOTA GGUF transcripts,
Then each manifest row records the raw model output, the executable validator
decisions, and the verifier accept/reject outcome, while the final artifact
reports aggregate tool-call validity, final-answer validity, verifier catch
rate, false-accept rate, model provenance, and an honest terminal verdict.

### REQ-VERIFY-1487: V_1 Pairwise Self-Verification vs Energy

The repository shall provide a bounded V_1-style pairwise self-verification
evaluation for Exp 1487 that:

- writes
  `results/experiment_1487_v1_pairwise_self_verification_vs_energy.json`
  with `status="in_progress"` before loading Exp 1486 rows or calling local
  models;
- loads only Exp 1486 deterministic executable-constraint rows and constructs
  candidate answer pairs with one executable-validator-valid answer and one
  executable-validator-invalid answer where possible;
- asks at least one mandated local SOTA GGUF verifier
  (`unsloth/Qwen3.6-35B-A3B-GGUF`,
  `unsloth/gemma-4-31B-it-GGUF`, or
  `unsloth/gemma-4-26B-A4B-it-GGUF`) to select the better answer pairwise,
  while recording blockers for unavailable mandated models;
- scores Carnot energy and BEAVER-style ranking on the same candidate pairs
  using deterministic executable constraint signals when available;
- compares pairwise accuracy against random, response-length, format-validity,
  and energy-ranking baselines; and
- writes `results/v1_pairwise_verification_1487.json` with per-pair decisions,
  scores, baseline decisions, and provenance before completing the terminal
  artifact.

The artifact MUST include `status`, `model_specs`,
`live_sota_model_inference_used`, `pairwise_verification_complete`,
`benchmark_cases_loaded`, `candidate_pairs_evaluated`, `pairwise_accuracy`,
`energy_ranking_accuracy`, `random_baseline_accuracy`,
`superficial_baseline_accuracy`, `pairwise_delta_over_energy`,
`improvement_allowed`, `diagnostic_path`, `tests_run`, and `honest_verdict`.

`improvement_allowed` MUST be true only when pairwise accuracy strictly exceeds
energy-ranking accuracy and the improvement is not matched or exceeded by the
best superficial baseline on the same pairs. Legacy small models may be used
only for CPU smoke-tests and MUST NOT be reported as headline pairwise verifier
results.

### SCENARIO-VERIFY-1487: Pairwise Selection Must Beat Superficial Baselines

Given Exp 1486 has a complete executable CCTU manifest with live local SOTA
rows,
When Exp 1487 constructs valid/invalid answer pairs, obtains pairwise choices
from at least one mandated local SOTA GGUF verifier, and scores deterministic
energy and superficial baselines on the same pair set,
Then the diagnostic records every per-pair choice and score, the terminal
artifact reports all required fields, and `improvement_allowed=true` only when
the pairwise verifier beats energy ranking and is not explained by response
length or format-validity baselines.

### REQ-VERIFY-1494: Bounded ConstrainPrompt Validator Compiler Audit

The repository shall provide a bounded ConstrainPrompt-style prompt-to-validator
compiler audit for Exp 1494 that:

- writes
  `results/experiment_1494_constrainprompt_validator_compiler_audit.json` with
  `status="in_progress"` before loading local models or compiling validators;
- builds exactly 30 fixed CCTU-style prompts from Exp 1486 cases plus new
  arithmetic, JSON-schema, simple-code, and graph/path cases;
- asks at least one mandated local SOTA GGUF model
  (`unsloth/Qwen3.6-35B-A3B-GGUF`,
  `unsloth/gemma-4-31B-it-GGUF`, or
  `unsloth/gemma-4-26B-A4B-it-GGUF`) to propose constraint fields or validator
  skeletons, while recording a terminal blocker when the mandated model path
  cannot load;
- compiles validators only through a deterministic safe DSL and fixed Python
  validator functions over restricted inputs, with no arbitrary `eval`,
  `exec`, or model-generated code execution path;
- tests every compiled validator against at least one known-good and one
  known-bad output and records compile failures, manual-review markers, false
  accepts, and false rejects;
- writes `results/constrainprompt_validator_manifest_1494.jsonl` with one row
  per prompt; and
- writes a terminal artifact containing `status`, `model_specs`,
  `live_sota_model_inference_used`, `validator_compiler_ready`,
  `prompts_attempted`, `validator_skeletons_generated`,
  `validators_compiled`, `validator_compile_rate`, `known_good_pass_rate`,
  `known_bad_reject_rate`, `verifier_false_accept_rate`,
  `manual_review_required_count`, `validator_manifest_path`, `models_used`,
  `gpu_probe`, `blockers`, and `honest_verdict`.

`validator_compiler_ready` MUST be true only when compile metrics and
false-accept metrics are present, at least one mandated live local SOTA GGUF
contributed skeleton rows, and the compiler did not introduce an arbitrary-code
execution path. Legacy small models may be used only for CPU smoke-tests and
MUST NOT count as headline validator compiler evidence.

### SCENARIO-VERIFY-1494: Prompt-To-Validator Audit Reports Safe DSL Metrics

Given the fixed 30-prompt Exp 1494 CCTU-style prompt set and a mandated local
SOTA GGUF skeleton proposer,
When the bounded ConstrainPrompt compiler audit runs on the run date `20260507`,
Then it writes the in-progress artifact first
And it writes one manifest row per prompt with prompt text, model skeleton
provenance, compiled DSL, known-good result, known-bad result, manual-review
status, and false-accept status
And it reports compile rate, known-good pass rate, known-bad reject rate, and
false-accept rate in the terminal artifact
And it sets `validator_compiler_ready=true` only when the safe DSL metrics are
present and no arbitrary-code execution path was introduced.

### REQ-VERIFY-1495: CPU-Only Interwhen Monitor Prototype Replay

The repository shall provide a CPU-only interwhen-style monitor prototype for
Exp 1495 that replays existing CCTU trigger-certificate and validator-compiler
artifacts without generating new LLM rows.

- REQ-VERIFY-1495-1: The workflow SHALL write
  `results/experiment_1495_interwhen_monitor_prototype.json` with
  `status="in_progress"` before loading gated upstream artifacts or emitting
  monitor events.
- REQ-VERIFY-1495-2: The workflow SHALL require both
  `results/experiment_1493_trigger_token_certificate_export_v1.json` and
  `results/experiment_1494_constrainprompt_validator_compiler_audit.json` to
  be complete and ready before setting `gated_inputs_present=true`; otherwise
  it SHALL write a terminal gated artifact with concrete blockers.
- REQ-VERIFY-1495-3: Certificate rows and validator rows SHALL be converted
  into replayable trace states with deterministic token offsets or synthetic
  polling intervals and no fresh model generation.
- REQ-VERIFY-1495-4: Each poll SHALL run deterministic certificate,
  validator, and verifier checks, emit exactly one JSONL monitor event, and
  mark whether the monitor interrupts only because an error was detected.
- REQ-VERIFY-1495-5: The workflow SHALL write
  `results/interwhen_monitor_events_1495.jsonl` with one event per poll and
  SHALL report detection count, interruptions, false interruptions, false
  accepts, and verifier false-accept rate.
- REQ-VERIFY-1495-6: The terminal artifact SHALL include `status`,
  `monitor_intervention_ready`, `gated_inputs_present`, `traces_replayed`,
  `polling_interval_tokens`, `monitor_events_emitted`, `errors_detected`,
  `interruptions_triggered`, `false_interruptions`,
  `verifier_false_accept_rate`, `monitor_event_manifest_path`, `blockers`, and
  `honest_verdict`.

`monitor_intervention_ready` MUST be true only when the event manifest exists,
at least one real recorded error is detected, and the verifier false-accept
rate is zero.

### SCENARIO-VERIFY-1495: Replayed Monitor Events Gate Intervention Readiness

Given complete Exp 1493 trigger-certificate evidence and complete Exp 1494
safe-DSL validator evidence,
When the Exp 1495 CPU-only interwhen replay runs on the run date `20260507`,
Then it emits one monitor event per synthetic poll over the replayed trace
states
And each event records the case ID, poll offset, deterministic check outcomes,
error detection status, interrupt decision, and false-interruption status
And it sets `monitor_intervention_ready=true` only when at least one recorded
error is detected, no false interruptions are triggered, and verifier
false-accept rate remains zero.

### REQ-VERIFY-1496: HoVer Safe-Prefix Continuation Audit

The repository shall provide a bounded HoVer-style safe-prefix continuation
audit for Exp 1496 that:

- writes
  `results/experiment_1496_hover_safe_prefix_continuation_audit.json` with
  `status="in_progress"` before loading monitor events, validators, or local
  models;
- requires the Exp 1495 monitor event manifest to exist before attempting
  continuation, and writes a terminal blocked artifact with concrete blockers
  when the gated monitor evidence is unavailable;
- defines a deterministic last-safe-prefix selection rule from monitor events:
  for each selected CCTU trigger-certificate row, choose the earliest
  interrupting monitor event for that case and lane, then keep only the
  free-form reasoning plus trigger-token boundary before the unsafe
  certificate suffix;
- asks at least one mandated local SOTA GGUF model
  (`unsloth/Qwen3.6-35B-A3B-GGUF`,
  `unsloth/gemma-4-31B-it-GGUF`, or
  `unsloth/gemma-4-26B-A4B-it-GGUF`) to continue from those selected prefixes,
  while recording a terminal blocker when no mandated model can load;
- evaluates matched no-continuation, safe-prefix continuation, and full
  regeneration rows for the same selected cases with deterministic CCTU and
  compiled safe-DSL validators; and
- writes `results/safe_prefix_continuations_1496.jsonl` with one row per
  case/baseline and a terminal artifact containing `status`, `model_specs`,
  `live_sota_model_inference_used`, `safe_prefix_continuation_ready`,
  `cases_attempted`, `continuations_completed`,
  `baseline_validator_pass_rate`, `safe_prefix_validator_pass_rate`,
  `full_regeneration_validator_pass_rate`, `verifier_false_accept_rate`,
  `last_safe_prefix_selection_rule`, `continuation_manifest_path`,
  `models_used`, `gpu_probe`, `blockers`, and `honest_verdict`.

`safe_prefix_continuation_ready` MUST be true only when the pass-rate and
false-accept metrics are present and at least one mandated live local SOTA GGUF
contributed safe-prefix continuation rows. Legacy small models may be used only
for CPU smoke-tests and MUST NOT count as headline continuation evidence.

### SCENARIO-VERIFY-1496: Safe-Prefix Continuation Reports Matched Validator Rates

Given Exp 1495 has emitted monitor events over CCTU trigger-certificate rows,
When the Exp 1496 audit selects the last safe prefix for interrupted cases and
continues from that prefix on the run date `20260507`,
Then it writes one manifest row for each no-continuation, safe-prefix, and
full-regeneration evaluation
And every row records the selected prefix, model provenance, deterministic
CCTU validation result, compiled-validator result, final validator decision,
and false-accept status
And the terminal artifact reports matched pass rates and verifier false-accept
rate before setting `safe_prefix_continuation_ready=true`.

### REQ-VERIFY-1499: Verifier Ensemble DRY And Conditional Orthogonality Audit

The repository shall provide a deterministic Exp 1499 audit over the active
post-.114 verifier surfaces: BEAVER-lite bounds, CCTU executable validators,
energy/localization, query-time memory-policy checks, and structured verdict
records. The audit MUST NOT count retired Semantic Energy or V_1 headline
signals as active verifiers.

The audit shall:

- write
  `results/experiment_1499_verifier_ensemble_dry_orthogonality_v2.json` with
  `status="in_progress"` before loading source artifacts;
- inventory active verifier surfaces from code and checked-in result artifacts;
- build a bounded case table with pass/fail/abstain labels from available
  verifier outputs;
- compute pairwise agreement, pairwise conditional acceptance rates,
  redundant verifier pairs, and a conservative `k_effective_estimate`;
- identify duplicated verifier wrappers or conditional duplicates that should
  be merged, retired, or demoted;
- prefer deterministic validators and energy ranking ahead of generative
  pairwise self-verification in recommendations; and
- write a terminal artifact containing `status`,
  `orthogonality_matrix_written`, `verifiers_audited`, `cases_evaluated`,
  `conditional_acceptance_matrix`, `redundant_verifier_pairs`,
  `k_effective_estimate`, `deterministic_first_recommendations`,
  `retire_or_keep_decisions`, `blockers`, and `honest_verdict`.

### SCENARIO-VERIFY-1499: Conditional Matrix Flags Duplicate Validators

Given complete Exp 1482 BEAVER-lite, Exp 1486 CCTU, Exp 1490 localization,
Exp 1484 query-time memory, and structured-verdict artifacts,
When the Exp 1499 audit builds the bounded case table,
Then it writes a terminal artifact with a conditional acceptance matrix for
the active verifier labels,
And it reports redundant verifier pairs when two labels accept exactly the
same observed cases or one label's accepts are conditionally contained in the
other at the configured redundancy threshold,
And it excludes retired Semantic Energy and V_1 headline signals from the
active verifier inventory.

### REQ-VERIFY-1500: Latent-Vs-Deterministic Discipline Gate

The repository shall provide a deterministic Exp 1500 discipline gate that
classifies post-.114 verifier and telemetry signals by the strongest claim
surface they may support: headline evidence, auxiliary ranking evidence,
triage evidence, or retired/no-claim evidence.

The gate shall:

- write
  `results/experiment_1500_latent_deterministic_discipline_gate.json` with
  `status="in_progress"` before loading gated upstream artifacts;
- require Exp 1499 to have written an orthogonality matrix before setting
  `discipline_gate_ready=true`, otherwise write a terminal gated artifact with
  concrete blockers;
- read the Exp 1481 Semantic Energy retirement, Exp 1487 V_1 pairwise
  self-verification comparison, and Exp 1499 deterministic-first
  recommendations before assigning signal roles;
- publish `ops/latent_deterministic_discipline_gate_1500.md` with policy
  tables for headline, auxiliary ranking, triage, and retired/no-claim
  signals;
- allow latent, energy-like, probabilistic, or LLM-derived signals to influence
  decisions only after deterministic validator comparison, superficial-baseline
  comparison, held-out calibration, and false-accept accounting are present;
- require deterministic validators to dominate whenever a deterministic
  validator is applicable to the same claim; and
- write a terminal artifact containing `status`, `discipline_gate_ready`,
  `gated_inputs_present`, `signal_classes_audited`,
  `headline_allowed_signals`, `auxiliary_allowed_signals`, `retired_signals`,
  `deterministic_first_rules`, `superficial_baseline_required_rules`,
  `ops_note_path`, `blockers`, and `honest_verdict`.

`honest_verdict` MUST begin with one of `complete:`, `complete_`, `success:`,
`success_`, `passed:`, `passed_`, `shipped:`, or `shipped_`.

### SCENARIO-VERIFY-1500: Discipline Gate Demotes Confounded Latent Signals

Given Exp 1499 contains a written orthogonality matrix, Exp 1481 retires
Semantic Energy because it does not beat superficial baselines, and Exp 1487
shows V_1 pairwise self-verification does not beat deterministic energy
ranking,
When the Exp 1500 discipline gate builds the policy artifact,
Then deterministic executable validators and conservative deterministic bounds
are the only headline-allowed signals, energy/localization and calibrated
probabilistic signals are limited to auxiliary or triage roles, Semantic
Energy headline telemetry and V_1 pairwise self-verification are retired from
claims, and the terminal artifact records the required acceptance rules and an
allowed-prefix honest verdict.

### REQ-VERIFY-1501: GNNVerifier Plan-Graph Energy Adapter

The repository shall provide a deterministic CPU-only Exp 1501 adapter that
converts bounded CCTU tool-use cases into directed plan graphs and scores
injected dependency faults with graph-risk energy. The adapter MUST NOT claim
trained GNN performance unless a trained model and honest train/eval split are
implemented.

The adapter shall:

- write
  `results/experiment_1501_gnnverifier_plan_graph_energy_adapter.json` with
  `status="in_progress"` before loading CCTU rows or scoring faults;
- select a bounded deterministic subset of CCTU tool-use cases and represent
  each as directed nodes and edges with node types, edge types, tool
  dependencies, and expected outputs;
- inject deterministic dependency faults including missing edges, wrong tool
  input type, missing intermediate, wrong ordering, and dangling output;
- score each faulty graph with deterministic graph-risk energy and localize the
  highest-risk node and edge when the fault exposes a node or edge target;
- compare node and edge top-1 localization against random and length/degree
  baselines on the same fault rows;
- write `results/plan_graph_energy_manifest_1501.jsonl` with one row per
  trace/fault; and
- write a terminal artifact containing `status`, `plan_graph_energy_ready`,
  `traces_converted`, `injected_graph_faults`,
  `node_localization_top1_rate`, `edge_localization_top1_rate`,
  `random_baseline_top1_rate`, `length_baseline_top1_rate`,
  `graph_energy_beats_baselines`, `adapter_manifest_path`, `blockers`, and
  `honest_verdict`.

`plan_graph_energy_ready` MUST be true only when graph conversion rows,
fault-injection rows, and baseline comparison metrics are written.
`honest_verdict` MUST begin with one of `complete:`, `complete_`, `success:`,
`success_`, `passed:`, `passed_`, `shipped:`, or `shipped_`.

### SCENARIO-VERIFY-1501: Plan-Graph Energy Localizes Injected Dependency Faults

Given the fixed Exp 1486 CCTU tool-use cases and deterministic local expected
tool outputs,
When Exp 1501 converts a bounded subset into plan graphs on the run date
`20260507`,
Then each manifest row records the trace ID, graph node and edge attributes,
injected fault type, expected risky node or edge, graph-risk ranking, random
baseline credit, length or degree baseline credit, and localization outcome
And the terminal artifact sets `plan_graph_energy_ready=true` only when at
least one trace and fault row are written, baseline rates are present, and the
deterministic graph energy beats both baselines without claiming trained GNN
performance.

### REQ-VERIFY-1507: Safe-DSL Verifier Induction Pack

The repository shall provide a bounded AutoPyVerifier-inspired verifier
induction pack for Exp 1507 that keeps generated verifier proposals outside
the Python execution trust boundary.

The pack shall:

- write
  `results/experiment_1507_autopyverifier_safe_dsl_induction_pack.json` with
  `status="in_progress"` before loading manifests, models, or candidates;
- require both `results/cctu_trigger_certificates_1493.jsonl` and
  `results/constrainprompt_validator_manifest_1494.jsonl`, and write a
  terminal artifact with concrete missing-path blockers if either is absent;
- load labeled certificate rows and compiled-validator audit rows from those
  manifests, preserving deterministic pass/fail labels for false-accept
  accounting;
- ask at least one mandated local SOTA GGUF model
  (`unsloth/Qwen3.6-35B-A3B-GGUF`,
  `unsloth/gemma-4-31B-it-GGUF`, or
  `unsloth/gemma-4-26B-A4B-it-GGUF`) for safe-DSL verifier skeletons only,
  while rejecting arbitrary Python, filesystem access, imports, eval/exec,
  network calls, and non-deterministic logic;
- compile candidate skeletons only through a minimal safe-DSL compiler or the
  existing safe validator compiler, recording compile failures explicitly;
- search for a compact verifier set that maximizes labeled-row coverage while
  preserving zero false accepts on available labels;
- write `results/safe_dsl_verifier_induction_1507.jsonl` with one row per
  candidate and a final selected-set summary; and
- write a terminal artifact containing `status`, `model_specs`,
  `live_sota_model_inference_used`, `verifier_induction_ready`,
  `labeled_rows_loaded`, `candidate_verifiers_proposed`,
  `candidate_verifiers_compiled`, `verifier_compile_rate`,
  `verifier_set_size`, `verifier_coverage_rate`,
  `verifier_false_accept_rate`, `baseline_validator_coverage_rate`,
  `induction_manifest_path`, `models_used`, `gpu_probe`, `blockers`, and
  `honest_verdict`.

`verifier_induction_ready` MUST be true only when at least one mandated live
local SOTA GGUF model proposed candidates, at least one candidate compiled, and
`verifier_false_accept_rate` is reported. Legacy small models may be used only
for CPU smoke-tests and MUST NOT count as headline verifier-induction evidence.
`honest_verdict` MUST begin with one of `complete:`, `complete_`, `success:`,
`success_`, `passed:`, `passed_`, `shipped:`, or `shipped_`.

### SCENARIO-VERIFY-1507: Safe-DSL Induction Preserves Zero False Accepts

Given the Exp 1493 CCTU certificate manifest and the Exp 1494 safe validator
compiler manifest exist on the run date `20260507`,
When Exp 1507 asks a mandated local SOTA GGUF model for safe-DSL verifier
skeletons, compiles them, and searches over the labeled rows,
Then arbitrary generated Python and unsafe capabilities are rejected before
scoring
And each candidate manifest row records provenance, compile status, compile
failure reason, coverage, false accepts, and accepted labeled row IDs
And the terminal selected-set summary reports the compact zero-false-accept
set before the artifact sets `verifier_induction_ready=true`.

### REQ-VERIFY-1508: Trigger+Grammar Certificate Decoder Audit

The repository shall provide a bounded trigger-token plus grammar/GBNF
certificate decoder audit for Exp 1508 that compares runtime-constrained
certificate tails against Exp 1493 schema-only certificate parsing.

The audit shall:

- write
  `results/experiment_1508_trigger_grammar_certificate_decoder_audit.json`
  with `status="in_progress"` before loading gates, manifests, models, or
  grammar backends;
- require
  `results/experiment_1507_autopyverifier_safe_dsl_induction_pack.json` to
  report `verifier_induction_ready=true`, and write a terminal gated artifact
  when that prerequisite is absent or false;
- load Exp 1493 CCTU certificate rows and the Exp 1507 selected verifier set
  where available, preserving schema-only parse, validation, and false-accept
  accounting for comparison;
- resolve mandated local SOTA GGUF model specs through `cached_sota_pair()` or
  the established SOTA GGUF cache resolver, without using legacy small models
  for headline decoder evidence;
- run at least one mandated local SOTA GGUF model for trigger+grammar decoder
  rows when a local grammar backend can enforce a bounded certificate grammar;
- record a concrete grammar backend name and blocker when runtime grammar
  enforcement is unavailable, falling back only to parse-only diagnostic rows;
- write `results/trigger_grammar_certificates_1508.jsonl` with one row per
  case, model, and decoder mode;
- report trigger-token presence, grammar parse, schema-only parse, grammar
  validation, schema-only validation, and verifier false-accept rates; and
- write a terminal artifact containing `status`, `model_specs`,
  `live_sota_model_inference_used`, `certificate_decoder_ready`,
  `gated_inputs_present`, `cases_attempted`, `grammar_backend`,
  `trigger_token_presence_rate`, `grammar_parse_rate`,
  `schema_only_parse_rate`, `grammar_validation_rate`,
  `schema_only_validation_rate`, `verifier_false_accept_rate`,
  `decoder_manifest_path`, `models_used`, `gpu_probe`, `blockers`, and
  `honest_verdict`.

`certificate_decoder_ready` MUST be true only when gated inputs are present,
at least one mandated live local SOTA GGUF grammar row exists, and both parse
and validation metrics are reported. Legacy small models may be used only for
CPU smoke-tests and MUST NOT count as headline decoder evidence.
`honest_verdict` MUST begin with one of `complete:`, `complete_`, `success:`,
`success_`, `passed:`, `passed_`, `shipped:`, or `shipped_`.

### SCENARIO-VERIFY-1508: Trigger+Grammar Rows Compare Against Schema-Only Rows

Given Exp 1507 verifier induction is ready and Exp 1493 CCTU certificate rows
exist on the run date `20260507`,
When Exp 1508 runs a mandated local SOTA GGUF model through a trigger-token
reasoning phase followed by a grammar-bounded certificate phase,
Then each manifest row records case ID, decoder mode, model provenance,
grammar backend, trigger-token presence, parser result, deterministic
validation result, and false-accept status
And the terminal artifact compares trigger+grammar rates against Exp 1493
schema-only rates before setting `certificate_decoder_ready=true`.

### REQ-VERIFY-1509: Executable Monitor Runtime Adapter

The repository shall provide a reusable CPU-only executable monitor runtime
adapter for Exp 1509 that normalizes already-recorded monitor, safe-prefix,
safe-DSL verifier, and trigger+grammar certificate artifacts into one
replayable manifest without generating new LLM rows.

The adapter shall:

- write
  `results/experiment_1509_executable_monitor_runtime_adapter.json` with
  `status="in_progress"` before loading upstream gates or manifest rows;
- require Exp 1507 verifier induction and Exp 1508 trigger+grammar certificate
  decoder artifacts to be complete and ready before setting
  `gated_inputs_present=true`, otherwise writing a terminal gated artifact
  with concrete blockers;
- inventory the Exp 1495 monitor event manifest, Exp 1496 safe-prefix
  continuation manifest, Exp 1507 safe-DSL verifier induction manifest, and
  Exp 1508 trigger+grammar certificate manifest;
- define a small normalized event schema with event schema version, replay
  order, source experiment, source path, source line, source row ID, case ID,
  event kind, validation status, verifier false-accept status, and provenance
  fields;
- validate every normalized event against that schema before writing the
  replay manifest;
- link safe-prefix continuation rows to monitor events only when recorded
  event IDs, token offsets, or unambiguous case IDs match, and record unmatched
  rows without inventing links;
- write `results/executable_monitor_events_1509.jsonl` with one normalized
  event per source manifest row; and
- write a terminal artifact containing `status`, `monitor_runtime_ready`,
  `gated_inputs_present`, `events_loaded`, `events_normalized`,
  `event_schema_version`, `verifier_false_accept_rate`,
  `safe_prefix_events_linked`, `monitor_event_manifest_path`,
  `adapter_tests_run`, `blockers`, and `honest_verdict`.

`monitor_runtime_ready` MUST be true only when the replay manifest exists, all
loaded events normalize successfully, the Exp 1507 and Exp 1508 gates are
ready, and `verifier_false_accept_rate` remains reported. `honest_verdict`
MUST begin with one of `complete:`, `complete_`, `success:`, `success_`,
`passed:`, `passed_`, `shipped:`, or `shipped_`.

### SCENARIO-VERIFY-1509: Runtime Adapter Replays Normalized Monitor Events

Given complete Exp 1507 verifier induction evidence, complete Exp 1508
trigger+grammar certificate evidence, and available .115/.116 monitor,
safe-prefix, verifier, and certificate manifests on the run date `20260507`,
When the Exp 1509 adapter loads and normalizes those recorded rows,
Then it writes exactly one normalized runtime event for each source manifest
row
And every event carries deterministic replay order, source provenance,
validation status, false-accept status, and a stable schema version
And safe-prefix rows carry a monitor link only when a recorded event ID, token
offset, or unambiguous case ID match exists
And the terminal artifact sets `monitor_runtime_ready=true` only when the
manifest exists, all events validate, and verifier false-accept rate is
reported.

### REQ-VERIFY-1510: Plan-Graph Structural Contract Gate

The repository shall provide a deterministic CPU-only Exp 1510 structural
contract gate that checks CCTU plan graphs before execution. The gate moves
Exp 1501's post-hoc dependency localization signal into a pre-execution
contract check and MUST NOT invoke external tools, LLMs, or learned graph
models while classifying violations.

The gate shall:

- write
  `results/experiment_1510_plan_graph_structural_contract_gate.json` with
  `status="in_progress"` before loading Exp 1501 or Exp 1509 artifacts;
- load Exp 1501 plan-graph cases and normalized Exp 1509 runtime events when
  available, preserving concrete source blockers without fabricating rows;
- define a small structural contract schema covering required prerequisites,
  acquisition paths from tool calls to final answers, tool ordering, required
  object acquisition, and incompatible API operations;
- evaluate known-good graphs and deterministic injected-violation graphs with
  exact structural checks rather than probabilistic scoring;
- classify violations by contract family and compare detection against
  deterministic random and length baselines when injected violations exist;
- write `results/plan_graph_structural_contracts_1510.jsonl` with one row per
  graph and contract result; and
- write a terminal artifact containing `status`,
  `structural_contract_gate_ready`, `plan_graphs_checked`,
  `contracts_defined`, `violations_injected`, `violations_detected`,
  `false_accept_rate`, `false_reject_rate`,
  `random_baseline_detection_rate`, `length_baseline_detection_rate`,
  `contract_manifest_path`, `blockers`, and `honest_verdict`.

`structural_contract_gate_ready` MUST be true only when the contract manifest
exists, at least one graph is checked, at least one contract is defined, and
`false_accept_rate` is reported. `honest_verdict` MUST begin with one of
`complete:`, `complete_`, `success:`, `success_`, `passed:`, `passed_`,
`shipped:`, or `shipped_`.

### SCENARIO-VERIFY-1510: Contract Gate Rejects Structural Plan Violations

Given Exp 1501 plan graphs and any available Exp 1509 normalized runtime
events on the run date `20260507`,
When Exp 1510 defines structural contracts and evaluates known-good plus
injected-violation plan graphs,
Then known-good graphs pass without false rejects,
And injected missing-prerequisite, broken-acquisition-path, wrong-ordering,
missing-object-acquisition, and incompatible-operation graphs are rejected
with deterministic violation classifications,
And each manifest row records graph provenance, contract family, expected
violation status, detected violation status, classifier outcome, random
baseline outcome, length baseline outcome, and contract evidence before the
terminal artifact sets `structural_contract_gate_ready=true`.

### REQ-VERIFY-1520: Runtime-Contract E2E Harness

The repository shall provide a deterministic CPU-only runtime-contract E2E
harness for Exp 1520 that loads the Exp 1507 safe-DSL verifier induction pack,
Exp 1508 trigger+grammar certificate decoder audit, Exp 1509 executable
monitor runtime adapter, and Exp 1510 plan-graph structural contract gate into
one acceptance manifest without generating new LLM rows.

The harness shall:

- write `results/experiment_1520_runtime_contract_e2e_harness.json` with
  `status="in_progress"` before loading source artifacts or manifest rows;
- load the Exp 1507, Exp 1508, Exp 1509, Exp 1510, and Exp 1511 terminal JSON
  artifacts and the Exp 1507, Exp 1508, Exp 1509, and Exp 1510 source JSONL
  manifests, writing a terminal blocker with concrete missing paths when any
  required source cannot be resolved;
- define a normalized contract-case schema containing prompt or case ID,
  proposed output, certificate parse result, safe-DSL verifier result, monitor
  event result, structural-contract result, expected label when explicitly
  available, and final deterministic accept/reject;
- write `results/runtime_contract_e2e_manifest_1520.jsonl` with one normalized
  contract-case row per linked source row and a final summary row;
- compute false accepts and false rejects only for normalized rows with an
  explicit expected label, never by inferring success from LLM prose;
- report linked-row counts separately for safe-DSL, grammar certificate,
  monitor event, and structural-contract families; and
- write a terminal artifact containing `status`, `runtime_contract_e2e_ready`,
  `source_artifacts_loaded`, `contract_cases_total`,
  `safe_dsl_cases_linked`, `grammar_certificate_cases_linked`,
  `monitor_events_linked`, `structural_contract_cases_linked`,
  `false_accept_count`, `false_accept_rate`, `false_reject_count`,
  `runtime_contract_manifest_path`, `focused_tests_passed`, `blockers`, and
  `honest_verdict`.

`runtime_contract_e2e_ready` MUST be true only when all mandatory source
artifacts load, at least one row from each of the four .116 contract families
is linked, `false_accept_rate` is reported as `0.0`, and the focused harness
tests have passed. `honest_verdict` MUST begin with one of `complete:`,
`complete_`, `success:`, `success_`, `passed:`, `passed_`, `shipped:`, or
`shipped_`.

### SCENARIO-VERIFY-1520: Runtime Contract Ledger Combines .116 Families

Given complete Exp 1507 safe-DSL, Exp 1508 certificate, Exp 1509 monitor, and
Exp 1510 structural-contract artifacts on the run date `20260508`,
When Exp 1520 resolves source manifests, normalizes rows, and writes the E2E
manifest,
Then every contract-case row records source provenance plus certificate,
safe-DSL, monitor, and structural-contract result fields
And the false-accept ledger counts only rows whose source artifact provides an
explicit expected label
And the terminal artifact sets `runtime_contract_e2e_ready=true` only when each
.116 contract family contributes at least one linked row and the reported
false-accept rate is exactly zero.

### REQ-VERIFY-1521: Live SOTA Contract-Guided Repair

The repository shall provide an Exp 1521 live local-SOTA repair adapter that
uses the Exp 1520 runtime-contract E2E manifest as deterministic authority and
compares baseline generation, grammar-only repair, and draft-conditioned
contract-guided repair without using legacy small models for headline rows.

The adapter shall:

- write `results/experiment_1521_live_sota_contract_guided_repair_v1.json`
  with `status="in_progress"` before model loading or row generation;
- load `results/runtime_contract_e2e_manifest_1520.jsonl` and select a bounded
  subset of deterministic, explicitly labeled contract-failing or
  contract-marginal rows;
- resolve mandated local SOTA GGUFs using `cached_sota_pair()` or the same local
  cache pattern, and write a terminal blocker if no mandated SOTA GGUF completes
  live inference;
- generate one baseline, grammar-only, and draft-conditioned output for each
  selected case and model, recording a raw-output hash or path per row;
- validate every generated output by converting it into an Exp 1520
  contract-case row and computing deterministic false-accept outcomes with the
  runtime-contract ledger helpers, never by trusting LLM prose;
- write `results/live_contract_guided_repair_1521.jsonl` with one row per case,
  model, and mode containing model provenance, mode, output hash or raw path,
  deterministic validator outcome, and repair outcome;
- report `baseline_accept_rate`, `grammar_only_accept_rate`,
  `draft_conditioned_accept_rate`, `repair_accept_rate_delta`,
  `false_accept_count`, and `false_accept_rate`; and
- write a terminal artifact containing `status`, `model_specs`,
  `live_sota_model_inference_used`, `contract_guided_repair_ready`,
  `e2e_cases_loaded`, `repair_cases_attempted`, `models_used`, `gpu_probe`,
  `repair_manifest_path`, `blockers`, and `honest_verdict`.

`contract_guided_repair_ready` MUST be true only when at least one mandated SOTA
GGUF produced live inference rows, repair metrics are reported, and
`false_accept_rate` is exactly `0.0`. `honest_verdict` MUST begin with one of
`complete:`, `complete_`, `success:`, `success_`, `passed:`, `passed_`,
`shipped:`, or `shipped_`.

### SCENARIO-VERIFY-1521: Draft-Conditioned Repair Stays Contract-Grounded

Given a complete Exp 1520 runtime-contract E2E manifest with explicitly labeled
reject or marginal rows on the run date `20260508`,
When Exp 1521 runs mandated local SOTA GGUF generation in baseline,
grammar-only, and draft-conditioned modes,
Then the JSONL repair manifest records a deterministic validator outcome for
each model, case, and mode
And the aggregate false-accept ledger is computed from Exp 1520 contract-case
rows rather than from generated prose
And the terminal artifact sets `contract_guided_repair_ready=true` only when
live SOTA inference completed and the reported false-accept rate remains zero.

### REQ-VERIFY-1535: XGrammar/ABS Contract Decoder Adapter

The repository shall provide an Exp 1535 contract decoder adapter that compares
the current grammar-only/post-decode runtime-contract path against an
XGrammar-2-compatible adapter interface with ABS-style DFA token masking for
bounded regular contract constraints.

The adapter shall:

- write `results/experiment_1535_xgrammar_abs_contract_decoder_adapter.json`
  with `status="in_progress"` before loading source manifests, probing
  optional grammar packages, or resolving models;
- load the Exp 1520 runtime-contract E2E manifest and select a bounded case set
  containing certificate, safe-DSL validator, monitor-event, and structural
  contract families when those families are available;
- probe whether a local XGrammar-compatible package is importable, and when it
  is absent, provide the same adapter interface through a deterministic
  ABS-style DFA mask simulation for regular contract fields;
- resolve mandated local SOTA GGUF model specs through the established
  `cached_sota_pair()` cache pattern where runtime inference is attempted, and
  mark any legacy small model rows as smoke-test-only and excluded from
  headline metrics;
- compare baseline grammar-only/post-decode outputs with automata-guided
  outputs on parse rate, contract accept rate, latency delta, and deterministic
  false-accept rate;
- hand every parsed output to the existing Exp 1520 deterministic validators
  before treating a row as accepted; and
- write a terminal artifact containing `status`, `milestone`,
  `contract_decoder_adapter_ready`, `model_specs`,
  `live_sota_model_inference_used`, `cases_attempted`,
  `baseline_parse_rate`, `automata_parse_rate`,
  `baseline_contract_accept_rate`, `automata_contract_accept_rate`,
  `latency_delta_seconds`, `false_accept_rate`, `xgrammar_available`,
  `abs_dfa_masks_used`, `adapter_path`, `focused_tests_passed`, and
  `honest_verdict`.

`contract_decoder_adapter_ready` MUST be true only when at least one case from
each available runtime-contract family is evaluated, automata metrics are
reported, deterministic validator handoff runs, and `false_accept_rate` is
exactly `0.0`. `honest_verdict` MUST begin with one of `complete:`,
`complete_`, `success:`, `success_`, `passed:`, `passed_`, `shipped:`, or
`shipped_`.

### SCENARIO-VERIFY-1535: Automata Decoder Masks Invalid Contract Prefixes

Given the Exp 1520 runtime-contract E2E manifest contains certificate,
safe-DSL, monitor-event, and structural contract rows on the run date
`20260508`,
When Exp 1535 compiles bounded DFA masks for the selected contract cases and
compares grammar-only/post-decode decoding against automata-guided decoding,
Then malformed or case-mismatched baseline outputs may fail parsing before
validation
And automata-guided outputs preserve only valid bounded JSON contract fields
before deterministic validator handoff
And the terminal artifact reports parse-rate, contract-accept-rate, latency,
and false-accept metrics without counting legacy small-model smoke rows as
headline evidence.

### REQ-VERIFY-1580: DCCD/JSONSchemaBench SOTA Structured-Output Smoke

The repository shall provide an Exp 1580 bounded structured-output smoke test
for Carnot verifier-output schemas using a JSONSchemaBench-style schema slice
and draft-conditioned constrained decoding.

The smoke test shall:

- write
  `results/experiment_1580_dccd_jsonschemabench_sota_structured_output_smoke.json`
  with `status="in_progress"` before resolving models or running decoders;
- define `MODEL_SPECS` by calling `cached_sota_pair(gpu_indices=(0, 1))`
  where local SOTA GGUF inference is attempted, and record exact resolved
  model paths and hub IDs;
- include at least one mandated SOTA GGUF from
  `unsloth/Qwen3.6-35B-A3B-GGUF`,
  `unsloth/gemma-4-31B-it-GGUF`, or
  `unsloth/gemma-4-26B-A4B-it-GGUF` in `models_used` when live rows complete;
- evaluate a bounded case slice that includes both JSONSchemaBench-style
  object constraints and Carnot verifier-output schemas;
- compare unconstrained draft output, standard constrained decoding, and DCCD
  output where local runtime support is available;
- report strict schema validity, semantic verifier correctness, latency,
  projection-tax proxy delta, false accepts, and whether a legacy tiny-model
  fallback path was used; and
- write a terminal artifact containing `status`, `MODEL_SPECS`, `models_used`,
  `used_mandated_sota_gguf`, `legacy_tiny_model_fallback_used`, `n_schemas`,
  `strict_schema_validity_rate`, `semantic_correctness_rate`,
  `false_accept_count`, `projection_tax_proxy_delta`,
  `dccd_jsonschema_smoke_complete`, and `honest_verdict`.

Legacy tiny-model fallback rows MUST be excluded from headline metrics and MUST
NOT set `used_mandated_sota_gguf=true`. `dccd_jsonschema_smoke_complete` MUST
be true only when at least one mandated SOTA GGUF row completes and the DCCD
mode reports strict schema validity and semantic correctness for every selected
schema without false accepts.

### SCENARIO-VERIFY-1580: DCCD Output Remains Schema-Strict And Semantically Checked

Given cached SOTA GGUFs are available through
`cached_sota_pair(gpu_indices=(0, 1))` and the selected Carnot verifier-output
schemas include deterministic semantic targets,
When Exp 1580 evaluates unconstrained draft, constrained decoding, and DCCD
outputs on the bounded JSONSchemaBench-style slice,
Then the terminal artifact records exact models used, per-mode schema and
semantic metrics, latency and projection-tax proxy deltas, and zero DCCD false
accepts before setting `dccd_jsonschema_smoke_complete=true`.

### REQ-VERIFY-1537: BEAVER-Lite Prefix-Bound Contract Audit

The repository shall provide an Exp 1537 BEAVER-lite prefix-bound audit that
uses the Exp 1535 contract decoder adapter rows and Exp 1520 runtime-contract
rows to produce deterministic routing-risk bounds for selected contract and
automata-decoder prefixes.  The audit shall never treat BEAVER bounds,
logprobs, or top-k mass as acceptance authority; deterministic runtime-contract
validators remain the final source of false-accept truth.

The audit shall:

- write `results/experiment_1537_beaver_prefix_bound_contracts_v3.json` with
  `status="in_progress"` before source manifest loading or prefix evaluation;
- select bounded runtime-contract and automata-decoder cases from Exp 1535,
  preserving contract case IDs, source families, decoder modes, and
  deterministic accept/reject labels;
- build a bounded prefix trie/frontier over canonical contract JSON targets and
  report monotone structural upper bounds for unexplored or invalid prefixes;
- use token logprob and top-k telemetry when a selected decoder row exposes it,
  otherwise set `token_logprob_available=false`, `topk_available=false`, and
  record the structural-bound simulation path explicitly;
- rank high-risk instances by BEAVER-lite structural/logprob bound and compare
  them with deterministic validator outcomes without promoting any bound to an
  accept/reject decision;
- compute `false_accept_rate` from Exp 1520-compatible validator rows rather
  than from prefix bounds; and
- write a terminal artifact containing `status`, `milestone`,
  `beaver_bound_ready`, `model_specs`, `live_sota_model_inference_used`,
  `bounded_prefixes`, `token_logprob_available`, `topk_available`,
  `bound_violations`, `high_risk_instances`,
  `deterministic_validator_final_authority`, `false_accept_rate`,
  `bound_audit_path`, `focused_tests_passed`, and `honest_verdict`.

`beaver_bound_ready` MUST be true only when at least one selected prefix is
bounded, every reported upper bound is in `[0, 1]`, deterministic validator
authority is explicitly true, false-accept metrics come from validator rows,
and focused tests have passed. `honest_verdict` MUST begin with one of
`complete:`, `complete_`, `success:`, `success_`, `passed:`, `passed_`,
`shipped:`, or `shipped_`.

### SCENARIO-VERIFY-1537: Prefix Bounds Rank Risk Without Acceptance Authority

Given complete Exp 1535 decoder rows and the Exp 1520 runtime-contract manifest
on the run date `20260508`,
When Exp 1537 audits selected contract JSON prefixes with live telemetry when
available or structural simulation otherwise,
Then prefix-bound series remain structurally monotone for canonical automata
targets
And high-risk rankings record deterministic validator outcomes alongside the
bound
And the terminal artifact reports `deterministic_validator_final_authority=true`
with `false_accept_rate` computed from validator rows, not BEAVER bounds.

### REQ-VERIFY-1538: Residual-Drift Commitment Ledger

The repository shall provide an Exp 1538 deterministic commitment ledger that
replays bounded multi-turn SATQuest, product-line, and runtime-contract cases
and classifies residual failures as true contradictions, satisfiable drift, or
other blockers.

The ledger shall:

- write `results/experiment_1538_residual_drift_commitment_ledger.json` with
  `status="in_progress"` before loading source manifests;
- load bounded cases from `results/satquest_cnf_verifier_1536.jsonl`,
  `results/product_line_rescue_1523.jsonl`, and
  `results/cdg_root_cause_repair_1522.jsonl` when available, preserving
  concrete missing-source blockers without fabricating rows;
- record per-turn commitments introduced by the prompt, intermediate answer or
  staged feedback, oracle/validator replay, and final decision;
- validate final answers with the deterministic SAT, product-line, and
  runtime-contract validators rather than LLM self-evaluation;
- classify a failure as `satisfiable_drift` only when the prior commitments
  have a deterministic satisfying completion but the final answer or plan
  forgets at least one prior commitment;
- classify a failure as `true_contradiction` only when the solver or contract
  oracle proves the combined commitments are unsatisfiable for the attempted
  final decision;
- write `results/residual_drift_commitment_ledger_1538.jsonl` with one row per
  replayed case plus a final summary row; and
- write a terminal artifact containing `status`, `milestone`,
  `residual_drift_ledger_ready`, `model_specs`,
  `live_sota_model_inference_used`, `multi_turn_cases`,
  `contradiction_cases`, `satisfiable_drift_cases`, `drift_rate`,
  `repaired_drift_cases`, `solver_oracle_used`, `false_accept_rate`,
  `ledger_path`, `focused_tests_passed`, and `honest_verdict`.

`residual_drift_ledger_ready` MUST be true only when at least one case from
each available SATQuest, product-line, and runtime-contract source is replayed,
deterministic oracle replay has run, the ledger file exists, focused tests have
passed, and `false_accept_rate` is exactly `0.0`. `honest_verdict` MUST begin
with one of `complete:`, `complete_`, `success:`, `success_`, `passed:`,
`passed_`, `shipped:`, or `shipped_`.

### SCENARIO-VERIFY-1538: Ledger Distinguishes Drift From Contradiction

Given complete Exp 1536 SATQuest rows, Exp 1523 product-line rescue rows, and
Exp 1522 runtime-contract CDG rows on the run date `20260508`,
When Exp 1538 builds the residual-drift commitment ledger,
Then every ledger case records ordered commitments and deterministic validation
evidence before classification
And SAT/product-line/runtime failures with a known satisfying completion are
classified as satisfiable drift rather than contradiction
And impossible SAT final decisions are classified as true contradictions
And the terminal artifact reports bounded metrics without claiming results
beyond the replayed source manifests.

### REQ-VERIFY-1522: Constraint Dependency Graph Root-Cause Repair Ordering

The repository shall provide an Exp 1522 deterministic CPU-only Constraint
Dependency Graph (CDG) analyzer that uses the Exp 1520 runtime-contract E2E
manifest as the local trust boundary and optionally loads Exp 1521 repair rows
when present.

The analyzer shall:

- write `results/experiment_1522_constraint_dependency_graph_root_cause_repair.json`
  with `status="in_progress"` before loading source rows;
- define auditable CDG nodes for parse, certificate, safe-DSL verifier, monitor
  event, structural dependency, solver oracle, and final accept contract
  categories;
- estimate directed dependencies deterministically from lifecycle ordering and
  observed co-failure evidence in the loaded runtime-contract rows;
- select failing rows from Exp 1520, compare flat validator-order localization
  against CDG-prioritized upstream localization, and report one manifest row per
  attempted failing case;
- validate every candidate repair decision through deterministic Exp 1520
  contract ledger semantics and never through LLM self-evaluation;
- write `results/cdg_root_cause_repair_1522.jsonl` with one row per case plus
  a final graph summary row containing node, edge, efficiency, and false-accept
  metrics;
- report `flat_order_fix_efficiency`, `cdg_fix_efficiency`, and
  `cdg_efficiency_delta`; and
- write a terminal artifact containing `status`, `model_specs`,
  `live_sota_model_inference_used`, `cdg_root_cause_repair_ready`,
  `e2e_cases_loaded`, `cdg_nodes`, `cdg_edges`,
  `root_cause_cases_attempted`, `flat_order_fix_efficiency`,
  `cdg_fix_efficiency`, `cdg_efficiency_delta`, `false_accept_count`,
  `false_accept_rate`, `cdg_manifest_path`, `models_used`, `blockers`, and
  `honest_verdict`.

`cdg_root_cause_repair_ready` MUST be true when CDG metrics are computed and
the deterministic false-accept rate is exactly `0.0`, even if the CDG
efficiency delta is negative. If no LLM repair proposal is invoked, the
artifact MUST set `live_sota_model_inference_used=false` and `models_used=[]`
while still recording the mandated local SOTA GGUF `model_specs`.
`honest_verdict` MUST begin with one of `complete:`, `complete_`, `success:`,
`success_`, `passed:`, `passed_`, `shipped:`, or `shipped_`.

### SCENARIO-VERIFY-1522: CDG Prioritizes Upstream Contract Failures

Given a complete Exp 1520 runtime-contract E2E manifest on the run date
`20260508` and optional Exp 1521 repair rows,
When Exp 1522 builds the runtime-contract CDG and analyzes failing contract
rows,
Then graph edges are deterministic and derived from lifecycle ordering plus
observed co-failures
And each attempted case records flat and CDG localization order, root-cause
category, repair-ready status, deterministic validation outcome, and false
accept status
And the terminal artifact sets `cdg_root_cause_repair_ready=true` exactly when
case metrics are computed and the reported false-accept rate remains zero.

### REQ-VERIFY-1525: MARCH Claim-Isolation Verifier Ablation

The repository shall provide an Exp 1525 MARCH-style ablation that compares
full-context verifier feedback against claim-isolated verifier feedback on
promoted Exp 1524 FR-11/runtime-contract cases while keeping deterministic
runtime-contract validators as the final authority.

The ablation shall:

- write
  `results/experiment_1525_march_claim_isolation_verifier_ablation.json` with
  `status="in_progress"` before loading source rows or running model checks;
- load promoted-policy evaluation rows from
  `results/fr11_live_policy_promotion_1524.jsonl` and deterministic contract
  rows from `results/runtime_contract_e2e_manifest_1520.jsonl`;
- resolve mandated local SOTA GGUFs using `cached_sota_pair()` or the same
  local cache pattern, and write a terminal blocker if no mandated SOTA GGUF can
  run rather than using legacy tiny models for headline claim-isolation rows;
- extract an atomic-claim schema from each answer or contract row and record
  stable claim IDs, source case IDs, claim text, source mode, source family, and
  deterministic accept/reject labels;
- run both full-context and claim-isolated checker modes, where the
  claim-isolated checker does not receive the original answer text;
- validate every checker accept/reject through the Exp 1520 deterministic
  runtime-contract ledger and treat checker disagreement as auxiliary evidence
  only;
- write `results/march_claim_isolation_1525.jsonl` with one row per claim/case
  plus a summary row;
- report full-context and claim-isolated accept rates, verifier-call budgets,
  `budget_delta`, false-accept count, and false-accept rate; and
- write a terminal artifact containing `status`, `model_specs`,
  `live_sota_model_inference_used`, `claim_isolation_ablation_ready`,
  `cases_loaded`, `claims_extracted`, `full_context_accept_rate`,
  `claim_isolated_accept_rate`, `claim_isolation_delta`,
  `verifier_calls_full_context`, `verifier_calls_claim_isolated`,
  `budget_delta`, `false_accept_count`, `false_accept_rate`,
  `claim_isolation_manifest_path`, `models_used`, `blockers`, and
  `honest_verdict`.

`claim_isolation_ablation_ready` MUST be true only when both checker modes run
on at least one mandated SOTA GGUF and `false_accept_rate` is reported.
`honest_verdict` MUST begin with one of `complete:`, `complete_`, `success:`,
`success_`, `passed:`, `passed_`, `shipped:`, or `shipped_`.

### SCENARIO-VERIFY-1525: Claim-Isolated Feedback Stays Validator-Grounded

Given promoted Exp 1524 FR-11 policy rows and deterministic Exp 1520
runtime-contract rows on the run date `20260508`,
When Exp 1525 extracts atomic claims and evaluates each case in full-context
and claim-isolated checker modes,
Then every claim row records whether the original answer was hidden from the
claim-isolated checker
And deterministic runtime-contract labels remain the final accept/reject
authority for false accepts
And the terminal artifact sets `claim_isolation_ablation_ready=true` only when
at least one mandated SOTA GGUF produced both full-context and claim-isolated
rows and a false-accept rate is reported.

### REQ-VERIFY-1541: Claim-Isolation Uncertainty Router

The repository shall provide an Exp 1541 claim-isolation uncertainty router
that combines Verify-When-Uncertain-style validator uncertainty with
BEAVER-lite prefix-risk signals so claim-isolated verification is applied only
to cases where it can change risk or cost.

The router shall:

- write `results/experiment_1541_claim_isolation_uncertainty_router_v2.json`
  with `status="in_progress"` before source manifest loading;
- load the Exp 1525 claim-isolation artifact/manifest and Exp 1537
  BEAVER-lite high-risk instance rankings;
- build a bounded case set from `results/runtime_contract_e2e_manifest_1520.jsonl`,
  `results/satquest_cnf_verifier_1536.jsonl`, and
  `results/product_line_rescue_1523.jsonl`, preserving source family, case ID,
  deterministic validator label, and extractable atomic claims;
- route only cases with uncertainty, prefix risk, or validator disagreement to
  claim-isolated verification while leaving low-risk full-context decisions
  unexpanded;
- compute full-context acceptance, routed claim-isolated acceptance,
  disagreements, verifier-call budget delta, and false-accept rate from
  deterministic SAT/product-line/runtime-contract validators rather than LLM
  self-evaluation;
- write `results/claim_isolation_uncertainty_router_1541.jsonl` with one row
  per routed or bypassed case plus a summary row; and
- write a terminal artifact containing `status`, `milestone`,
  `uncertainty_router_ready`, `model_specs`,
  `live_sota_model_inference_used`, `cases_loaded`, `claims_extracted`,
  `routed_cases`, `full_context_accept_rate`, `claim_isolated_accept_rate`,
  `disagreements`, `budget_delta`, `false_accept_rate`,
  `routing_policy_path`, `focused_tests_passed`, and `honest_verdict`.

`uncertainty_router_ready` MUST be true only when at least one case from each
available source family is loaded, at least one but not all cases are routed,
focused tests have passed, and `false_accept_rate` is exactly `0.0`.  The
artifact MUST claim budget improvement only when `budget_delta <= 0` and
`false_accept_rate == 0.0`.  `honest_verdict` MUST begin with one of
`complete:`, `complete_`, `success:`, `success_`, `passed:`, `passed_`,
`shipped:`, or `shipped_`.

### SCENARIO-VERIFY-1541: Uncertainty Routing Limits Claim Isolation

Given complete Exp 1525 claim-isolation rows, Exp 1537 BEAVER-lite risk
rankings, Exp 1536 SATQuest rows, Exp 1523 product-line rows, and Exp 1520
runtime-contract rows on the run date `20260508`,
When Exp 1541 extracts deterministic claims and evaluates the uncertainty
routing policy,
Then uncertain, prefix-risky, or validator-disagreement cases are routed to
claim-isolated verification
And low-risk cases remain full-context-only
And the terminal artifact reports whether routing improves verifier-call cost
or error detection without deterministic false accepts.

### REQ-VERIFY-1542: ARM/EBT Soft-Value Diagnostic

The repository shall provide an Exp 1542 ARM/EBT diagnostic that compares
autoregressive soft-value proxies with explicit Carnot energy and deterministic
validator labels without using soft values as verifier authority.

The diagnostic shall:

- write `results/experiment_1542_arm_ebm_soft_value_diagnostic.json` with
  `status="in_progress"` before source loading or metric computation;
- load deterministic-labeled cases from SATQuest CNF rows, runtime-contract
  rows, and BEAVER-lite prefix-risk outputs;
- use mandated local SOTA GGUF provenance when available and record whether
  logprob/top-k/value proxies are available; legacy small GGUFs shall not count
  as headline evidence;
- compute explicit Carnot-energy-to-label correlation whenever finite energy
  scores and both deterministic labels are present;
- compute soft-value/logprob-to-label correlation only when real soft-value
  telemetry is present, otherwise report `logprob_available=false` and an
  honest blocker while still running the Carnot-energy-only diagnostic;
- compute a bounded routing AUC when the case set supports positive and
  negative deterministic labels;
- keep deterministic SAT/runtime-contract validators as the final accept/reject
  authority and record `no_model_weight_mutation=true`; and
- write a terminal artifact containing `status`, `milestone`,
  `arm_ebm_diagnostic_ready`, `model_specs`,
  `live_sota_model_inference_used`, `diagnostic_cases`,
  `logprob_available`, `carnot_energy_available`,
  `energy_label_correlation`, `soft_value_label_correlation`, `routing_auc`,
  `deterministic_validators_final_authority`, `no_model_weight_mutation`,
  `diagnostic_report_path`, `focused_tests_passed`, and `honest_verdict`.

`arm_ebm_diagnostic_ready` MUST be true only when at least one diagnostic case
is loaded, explicit Carnot energy is available, deterministic validators remain
final authority, no model weights are mutated, and focused tests have passed.
`honest_verdict` MUST begin with one of `complete:`, `complete_`, `success:`,
`success_`, `passed:`, `passed_`, `shipped:`, or `shipped_`.

### SCENARIO-VERIFY-1542: Soft Values Remain Diagnostic Only

Given complete Exp 1536 SATQuest rows, Exp 1520 runtime-contract rows, and Exp
1537 BEAVER-lite prefix-risk outputs on the run date `20260508`,
When Exp 1542 builds the ARM/EBT soft-value diagnostic,
Then the report compares explicit Carnot energy, available soft-value/logprob
proxies, prefix-risk scores, and deterministic accept/reject labels
And missing logprob telemetry blocks only the soft-value correlation field
And deterministic SAT/runtime-contract labels remain the final accept/reject
authority in every diagnostic row.

### REQ-VERIFY-1556: ARM/EBT Logprob Telemetry Repair Diagnostic

The repository shall provide an Exp 1556 ARM/EBT telemetry-repair diagnostic
that reuses the mandated local SOTA GGUF runtime path to capture token logprob
and top-k telemetry for deterministic-labeled verifier cases when the runtime
exposes that telemetry, and otherwise records exact blockers without promoting
soft signals to acceptance authority.

The diagnostic shall:

- write `results/experiment_1556_arm_ebm_logprob_telemetry_repair.json` with
  `status="in_progress"` before source loading, runtime probing, or metric
  computation;
- select a bounded case set from SATQuest, runtime-contract, or product-line
  rows where deterministic accept/reject labels and explicit energy scores are
  available;
- invoke the existing local SOTA GGUF telemetry path for at least one mandated
  model when locally available, requesting token logprobs and top-k alternatives
  without using legacy small GGUFs as headline telemetry evidence;
- parse token logprobs and top-k alternatives into per-case diagnostic rows,
  recording `logprob_available=false` or `topk_available=false` with precise
  blockers when the runtime returns text without those telemetry fields;
- compute energy-to-label correlation and routing AUC only as diagnostic
  measurements over deterministic labels;
- keep deterministic SAT/runtime-contract/product-line validators as the final
  accept/reject authority, so logprob, top-k, soft-value, model confidence, and
  model-declared acceptance can never override a validator rejection; and
- write a terminal artifact containing `status`, `milestone`,
  `arm_ebm_logprob_telemetry_ready`, `model_specs`,
  `live_sota_model_inference_used`, `logprob_available`, `topk_available`,
  `telemetry_adapter_path`, `diagnostic_cases`,
  `energy_label_correlation`, `routing_auc`,
  `deterministic_validators_final_authority`, `telemetry_blockers`,
  `focused_tests_passed`, and `honest_verdict`.

`arm_ebm_logprob_telemetry_ready` MUST be true only when at least one
diagnostic case has live mandated SOTA text, token logprob telemetry, top-k
alternatives, deterministic validator labels, explicit energy scores,
deterministic validators remain final authority, and focused tests have passed.
`honest_verdict` MUST begin with one of `complete:`, `complete_`, `success:`,
`success_`, `passed:`, `passed_`, `shipped:`, or `shipped_`.

### SCENARIO-VERIFY-1556: Logprobs Stay Below Validator Authority

Given deterministic-labeled SATQuest, runtime-contract, or product-line cases
on the run date `20260508`,
When Exp 1556 captures local SOTA token logprob and top-k telemetry for those
cases,
Then the diagnostic report records available token logprobs, top-k
alternatives, energy scores, deterministic labels, energy-label correlation,
and routing AUC
And missing runtime telemetry is recorded as explicit blockers rather than
fabricated from legacy models
And every final accept/reject decision remains the deterministic validator
decision even when soft telemetry would prefer the opposite answer.

### REQ-VERIFY-1551: Automata/SAT Unified Contract Gate

The repository shall provide an Exp 1551 unified contract gate that sequences
generation-time constraints and deterministic validators for bounded SATQuest,
product-line, and runtime-contract cases.

The gate shall:

- write `results/experiment_1551_automata_sat_unified_contract_gate.json` with
  `status="in_progress"` before loading predecessor artifacts or evaluating
  cases;
- load Exp 1535 automata/ABS, Exp 1549 SATQuest oracle repair, and Exp 1540
  product-line artifacts, and refuse SATQuest acceptance authority unless Exp
  1549 reports zero repaired solver-oracle false accepts;
- expose one shared gate abstraction that routes a generated output through
  syntax or automata masks, semantic repair, the relevant SAT or product-line
  solver oracle, and runtime contracts in that order;
- keep deterministic SAT, product-line, and runtime-contract validators as the
  final accept/reject authority, so soft signals, model confidence, or prefix
  masks can never override a validator mismatch;
- run at least one mandated headline SOTA GGUF model when locally available,
  otherwise record concrete availability blockers and exclude legacy small
  GGUF smoke tests from headline metrics;
- evaluate a bounded mixed set containing SATQuest, product-line, and
  runtime-contract cases, reporting syntax acceptance, semantic repair success,
  oracle agreement, false accepts, and latency delta; and
- write a terminal artifact containing `status`, `milestone`,
  `unified_contract_gate_ready`, `model_specs`,
  `live_sota_model_inference_used`, `cases_attempted`,
  `automata_masks_used`, `semantic_repair_layer_used`, `sat_oracle_used`,
  `product_line_oracle_used`, `runtime_contracts_used`,
  `syntax_accept_rate`, `semantic_repair_success_rate`,
  `oracle_agreement_rate`, `false_accept_rate`, `latency_delta_seconds`,
  `gate_module_path`, `focused_tests_passed`, and `honest_verdict`.

`unified_contract_gate_ready` MUST be true only when each available case family
is evaluated, automata masks and semantic repair are exercised, deterministic
validator final authority is preserved, focused tests have passed, and
`false_accept_rate` is exactly `0.0`. `honest_verdict` MUST begin with one of
`complete:`, `complete_`, `success:`, `success_`, `passed:`, `passed_`,
`shipped:`, or `shipped_`.

### SCENARIO-VERIFY-1551: Gate Rejects Solver Mismatches After Repair

Given complete Exp 1535, Exp 1549, and Exp 1540 artifacts on the run date
`20260508`,
When Exp 1551 evaluates bounded SATQuest, product-line, and runtime-contract
outputs through the unified gate,
Then syntax or automata masks run before semantic repair
And semantic repair runs before SAT/product-line/runtime validators
And any solver or runtime-contract mismatch is rejected even when a soft signal
or model-declared accept says the output should pass
And the terminal artifact reports zero deterministic false accepts for the
bounded mixed case set.

### REQ-VERIFY-1552: Residual-Drift Local Repair Policy

The repository shall provide an Exp 1552 residual-drift repair policy that
loads the Exp 1538 commitment ledger, separates true contradictions from
satisfiable drift cases, localizes the violated commitment or validator span,
and proposes minimal repairs instead of regenerating whole answers.

The policy shall:

- write `results/experiment_1552_residual_drift_repair_policy_v1.json` with
  `status="in_progress"` before loading the residual-drift ledger;
- load `results/residual_drift_commitment_ledger_1538.jsonl` and preserve
  concrete missing-ledger blockers without fabricating drift rows;
- attempt repair only for `satisfiable_drift` rows and leave
  `true_contradiction` rows untouched;
- localize each attempted repair to the forgotten SAT answer/assignment,
  product-line feature selection, or runtime-contract root-cause span;
- use at least one mandated local SOTA GGUF model for repair-proposal text when
  available, otherwise record concrete availability blockers and exclude
  legacy small GGUFs from headline results;
- replay deterministic SAT, product-line, and runtime-contract validators after
  every proposed repair;
- reject repairs that hide contradictions, create deterministic false accepts,
  or fail the relevant validator; and
- write a terminal artifact containing `status`, `milestone`,
  `residual_drift_repair_ready`, `model_specs`,
  `live_sota_model_inference_used`, `drift_cases_before`,
  `repair_attempts`, `localized_repairs_attempted`,
  `repaired_drift_cases`, `drift_reduction_delta`,
  `contradiction_cases_untouched`, `false_accept_rate`,
  `replay_pass_rate`, `repair_policy_path`, `focused_tests_passed`, and
  `honest_verdict`.

`residual_drift_repair_ready` MUST be true only when at least one satisfiable
drift case is attempted, all true contradictions are untouched, every accepted
repair passes deterministic replay, focused tests have passed, and
`false_accept_rate` is exactly `0.0`. `honest_verdict` MUST begin with one of
`complete:`, `complete_`, `success:`, `success_`, `passed:`, `passed_`,
`shipped:`, or `shipped_`.

### SCENARIO-VERIFY-1552: Local Repairs Replay Before Acceptance

Given a complete Exp 1538 residual-drift commitment ledger on the run date
`20260508`,
When Exp 1552 evaluates the repair policy,
Then true contradiction rows are counted as untouched and receive no repair
proposal
And satisfiable drift rows receive localized edit plans for only the violated
commitment or validator span
And candidate repairs are accepted only after deterministic SAT/product-line/
runtime-contract replay passes
And candidate repairs that would create false accepts are rejected before they
can reduce the reported drift count.

### REQ-VERIFY-1553: Claim-Isolation Router Scale Behind Unified Gate

The repository shall provide an Exp 1553 claim-isolation router scale run that
loads the Exp 1541 uncertainty-router artifact and the Exp 1551 unified
contract-gate artifact before evaluating a larger mixed case set.

The scale run shall:

- write `results/experiment_1553_claim_isolation_router_scale_v3.json` with
  `status="in_progress"` before loading predecessor artifacts or source rows;
- require `unified_contract_gate_ready=true` from Exp 1551 before claiming a
  ready terminal artifact;
- reuse the Exp 1541 routing policy and evaluate at least 75 mixed cases when
  the checked-in manifests contain enough rows, including runtime-contract,
  SATQuest, product-line, and residual-drift examples;
- compare routed claim-isolation verifier calls against a full-context
  verification baseline under matched unified-gate acceptance criteria;
- keep deterministic SAT, product-line, runtime-contract, and residual-drift
  replay validators as final authority so hidden deterministic failures cannot
  be accepted by either routed or bypassed paths;
- compute `budget_delta`, `budget_reduced`, `false_accept_rate`, and
  `missed_failure_count` from deterministic labels rather than model
  self-evaluation;
- record mandated SOTA GGUF provenance from predecessor artifacts and concrete
  availability blockers whenever a mandated model is unavailable; and
- write a terminal artifact containing `status`, `milestone`,
  `claim_isolation_router_scale_ready`, `model_specs`,
  `live_sota_model_inference_used`, `cases_total`, `routed_cases`,
  `full_context_cases`, `claims_extracted`, `budget_delta`, `budget_reduced`,
  `false_accept_rate`, `missed_failure_count`, `router_policy_path`,
  `focused_tests_passed`, and `honest_verdict`.

`claim_isolation_router_scale_ready` MUST be true only when Exp 1551 is ready,
at least 75 cases are evaluated, all four required source kinds are present,
at least one but not all cases are routed, focused tests have passed,
`false_accept_rate` is exactly `0.0`, and `budget_reduced=true`.
`budget_reduced` MUST be true only when the routed verifier-call budget is
lower than the full-context baseline under the same unified-gate acceptance
criteria. `honest_verdict` MUST begin with one of `complete:`, `complete_`,
`success:`, `success_`, `passed:`, `passed_`, `shipped:`, or `shipped_`.

### SCENARIO-VERIFY-1553: Scaled Routing Preserves Gate Safety

Given complete Exp 1541 and Exp 1551 artifacts on the run date `20260508`,
When Exp 1553 evaluates the uncertainty router on a 75+ mixed case set behind
the unified contract gate,
Then threshold-risky, prefix-risky, validator-disagreement, and residual-drift
cases are routed to claim-isolated verification
And low-risk cases are excluded from the routed verifier-call budget
And every final accept is still gated by deterministic SAT/product-line/
runtime-contract or residual-drift replay authority
And the terminal artifact reports whether the routed budget is lower than the
full-context baseline with zero deterministic false accepts.

### REQ-VERIFY-1557: Weaver Verification-Compute Router

The repository shall provide an Exp 1557 verification-compute router that loads
the Exp 1550 SATQuest SOTA re-evaluation artifact and the Exp 1551 unified
contract-gate artifact before selecting a bounded mixed candidate set.

The router shall:

- write `results/experiment_1557_weaver_verification_compute_router.json` with
  `status="in_progress"` before loading predecessor artifacts or source rows;
- use weak verifier signals, including automata/format validity, BEAVER-style
  prefix risk, claim-router uncertainty, energy/logprob diagnostics, and model
  self-declared accept flags, only to choose how much verification compute to
  spend;
- never allow a weak or soft signal to become acceptance authority;
- run at least one deterministic validator before any candidate can be accepted
  and run all available deterministic validators for high-risk candidates;
- compare routed verification cost against an always-run-all-deterministic
  baseline on the same bounded candidate set;
- compute `false_accept_rate` and `missed_failure_count` from deterministic
  validator outcomes rather than model self-evaluation;
- mark `verification_compute_router_ready=false` whenever routing would hide a
  deterministic failure; and
- write a terminal artifact containing `status`, `milestone`,
  `verification_compute_router_ready`, `candidate_selection_cases`,
  `weak_verifiers_used`, `deterministic_validators_used`,
  `soft_signals_used_for_routing_only`, `verification_cost_baseline`,
  `verification_cost_router`, `verification_cost_delta`, `false_accept_rate`,
  `missed_failure_count`, `router_policy_path`, `focused_tests_passed`, and
  `honest_verdict`.

`verification_compute_router_ready` MUST be true only when Exp 1550 and Exp
1551 are loaded, at least one candidate is evaluated, focused tests have
passed, routed cost is lower than the all-deterministic baseline, every final
accept has deterministic validator support, `false_accept_rate` is exactly
`0.0`, and `missed_failure_count` is `0`. `honest_verdict` MUST begin with one
of `complete:`, `complete_`, `success:`, `success_`, `passed:`, `passed_`,
`shipped:`, or `shipped_`.

### SCENARIO-VERIFY-1557: Routing Saves Cost Without Acceptance Authority

Given complete Exp 1550 and Exp 1551 artifacts on the run date `20260508`,
When Exp 1557 evaluates weak verifier signals over a bounded SATQuest,
runtime-contract, product-line, and residual-drift candidate set,
Then low-risk candidates use the cheap path plus one deterministic source
validator
And high-risk candidates fall back to all available deterministic validators
And a soft accept or high confidence signal never overrides a deterministic
rejection
And the terminal artifact reports whether routed verification cost decreases
with zero deterministic false accepts and zero missed deterministic failures.

### REQ-VERIFY-1554: Product-Line Staged Scale V4 Behind Unified Gate

The repository shall provide an Exp 1554 product-line staged scale run that
loads the Exp 1540 product-line scale artifact and the Exp 1551 unified
contract gate before reporting larger solver-grounded product-line metrics.

The scale run shall:

- write `results/experiment_1554_product_line_staged_scale_v4.json` with
  `status="in_progress"` before loading predecessor artifacts or evaluating
  product-line cases;
- require `unified_contract_gate_ready=true` from Exp 1551 before claiming a
  ready terminal artifact;
- build or select a staged product-line pack up to 120 cases when the
  checked-in generators and runtime permit, covering syntax-only,
  feasibility, objective-quality, and natural-language product-line variants;
- route rows through deterministic product-line parsing and solver-oracle
  evaluation, preserving the unified contract gate as an upstream prerequisite
  and the product-line solver oracle as final accept/reject authority;
- compute parse rate, feasibility rate, oracle agreement rate, mean objective
  gap, entity hallucination rate, and false accept rate from structured row
  fields rather than model self-evaluation;
- record mandated SOTA GGUF model availability honestly, excluding legacy
  small GGUF smoke tests from headline-result model lists; and
- write a terminal artifact containing `status`, `milestone`,
  `product_line_scale_v4_ready`, `branch_retired`, `model_specs`,
  `live_sota_model_inference_used`, `cases_total`, `stages_tested`,
  `parse_rate`, `feasibility_rate`, `objective_gap_mean`,
  `oracle_agreement_rate`, `entity_hallucination_rate`, `false_accept_rate`,
  `automata_constraints_used`, `product_line_manifest_path`,
  `focused_tests_passed`, and `honest_verdict`.

`product_line_scale_v4_ready` MUST be true only when Exp 1551 is ready, at
least one product-line case is evaluated, focused tests have passed,
deterministic checks are made from structured solver-backed fields,
`false_accept_rate` is exactly `0.0`, and `branch_retired=false`.
`branch_retired` MUST be true when false accepts recur or deterministic
feasibility/oracle checks cannot be made. `honest_verdict` MUST begin with one
of `complete:`, `complete_`, `success:`, `success_`, `passed:`, `passed_`,
`shipped:`, or `shipped_`.

### SCENARIO-VERIFY-1554: Product-Line Scaling Reports Solver-Grounded Fields

Given complete Exp 1540 and Exp 1551 artifacts on the run date `20260508`,
When Exp 1554 evaluates a larger staged product-line pack behind the unified
contract gate,
Then syntax-only, feasibility, objective-quality, and natural-language product
line variants are represented in the manifest
And parse, feasibility, objective-gap, oracle-agreement, hallucination, and
false-accept metrics are aggregated from deterministic row fields
And any false accept or missing deterministic oracle field retires the branch
before a ready artifact can be reported.

### REQ-VERIFY-1562: BRAIN Linear-AR k-Sweep Verification

The repository shall provide a runnable Exp 1562 BRAIN-correlation verification
workflow that extends the DT-BRAIN-CORRELATIONS brute-force enumeration from
`k=4` to `k in {4, 8, 12, 15}` at `n=16`.

The workflow shall:

- enumerate all `2^16 = 65,536` binary states exactly for each `k`;
- preserve the original deterministic `k=4` seed semantics so the baseline
  factorized and Linear-AR KL values remain comparable to the 2026-05-08
  partial validation run;
- optimize and report reverse KL `KL(q || pi_beta)` for a factorized Bernoulli
  model with `n=16` parameters and a Linear-AR model with
  `n + n(n - 1) / 2 = 136` parameters;
- run the optional one-hidden-layer MADE arm with 32 hidden units only when the
  Linear-AR `k=15` KL is above `0.1`;
- write `results/experiment_1562_brain_linear_ar_k_sweep_extended.json` with
  `status`, `brain_linear_ar_rescue_validated`,
  `kl_by_k_by_parameterization`, `factorized_vs_ar_ratio_at_k15`,
  `made_required_at_k15`, `phase_3_recommendation`, and `honest_verdict`; and
- recommend `linear_ar_sufficient` only when the `k=15` ratio is at least
  `10.0` and the best `k=15` KL is at most `0.1`, `made_required` only when
  MADE is needed and reaches the `0.1` KL gate, and `brain_dropped` when the
  `k=15` factorized-vs-AR ratio is below `5.0`.

`status` MUST be `complete` for terminal artifacts. `honest_verdict` MUST begin
with `complete:` and MUST distinguish positive validation from falsification.

### SCENARIO-VERIFY-1562: k-Sweep Writes Honest BRAIN Recommendation

Given the deterministic DT-BRAIN-CORRELATIONS constraint generator with
`n=16`, `m=10`, `beta=2.0`, and `seed=42`,
When Exp 1562 runs the exact `{4, 8, 12, 15}` k-sweep,
Then the artifact reports factorized and Linear-AR reverse KL for every `k`
And preserves the `k=4` baseline comparison fields
And records `made_optional` as null unless the Linear-AR `k=15` KL exceeds
`0.1`
And maps a `k=15` ratio below `5.0` to `phase_3_recommendation="brain_dropped"`
instead of claiming the Linear-AR rescue was validated.

### REQ-VERIFY-1571: Step-Wise Baseline AR-REINFORCE

The repository shall provide a runnable Exp 1571 AR-REINFORCE variance
microbenchmark for a Linear-AR Bernoulli model on an `n=32`, `k=15`
AND-composition stress case with `3%` Gaussian energy noise.

The workflow shall:

- implement a per-token step-wise baseline for Linear-AR REINFORCE whose
  baseline for token `t` depends only on the prefix `x_<t` and model
  parameters, not on `x_t` or later sampled tokens;
- compare scalar-batch-mean REINFORCE against the step-wise baseline using
  trace variance over the Linear-AR coupling-parameter score-function
  gradient;
- report `gradient_variance_reduction_factor >= 10.0` for the step-wise
  baseline versus the scalar baseline on the `n=32`, `k=15` noisy-energy
  benchmark;
- adapt BRAIN Theorem 2's noise-resilience claim to AR stochastic
  approximation by comparing the step-wise estimator's noisy and noiseless
  gradient signal-to-noise convergence-rate proxy, and require the noisy
  `3%` run to retain at least `97%` of the noiseless proxy; and
- write `results/experiment_1571_step_wise_baseline_AR_REINFORCE.json` with
  `status`, `step_wise_baseline_implemented`,
  `gradient_variance_reduction_factor`,
  `convergence_rate_matches_theorem_2`, and `honest_verdict`.

`status` MUST be `complete` for terminal artifacts. `honest_verdict` MUST begin
with `complete:` and MUST state whether the variance and noise-resilience gates
passed.

### SCENARIO-VERIFY-1571: Step-Wise Baseline Reduces AR Coupling Variance

Given the planted `n=32`, `k=15`, ten-constraint AND-composition stress case
with `3%` Gaussian reward noise,
When Exp 1571 evaluates Linear-AR REINFORCE gradient samples with a scalar
batch-mean baseline and with the prefix-only step-wise baseline,
Then the artifact reports `step_wise_baseline_implemented=true`
And the reported AR-coupling gradient variance reduction factor is at least
`10.0`
And `convergence_rate_matches_theorem_2=true` only when the noisy step-wise
signal-to-noise convergence-rate proxy is at least `97%` of the noiseless
step-wise proxy.

### REQ-VERIFY-1578: BRAIN REINFORCE Training-Dynamics Audit at k=15

The repository shall provide a runnable Exp 1578 BRAIN REINFORCE
training-dynamics audit for the same `n=16`, `k=15`, `beta=2.0`, ten random
AND-composition regime used to interpret Exp 1562.

The workflow shall:

- initialise both the factorized Bernoulli and Linear-AR Bernoulli
  parameterizations uniformly with zero logits and zero lower-triangular AR
  weights;
- train both parameterizations with a scalar-baseline REINFORCE estimator for
  reverse KL `KL(q || pi_beta)`, using batch size `512` and at most `50,000`
  iterations;
- compute exact finite-state `KL(q || pi_beta)` against the enumerated
  `2^16`-state target distribution at iteration `0` and every `1000`
  iterations;
- track gradient L2 norm, marginal escape from `0.5`, first-1000-iteration
  gradient-active fraction, convergence iteration, and wall-clock time for
  both parameterizations;
- write `results/experiment_1578_brain_reinforce_training_dynamics_at_k15.json`
  with `status`, `factorized_gradient_active_fraction_first_1000`,
  `linear_ar_gradient_active_fraction_first_1000`, `factorized_final_kl`,
  `linear_ar_final_kl`, `factorized_converged`, `linear_ar_converged`,
  `brain_training_dynamics_verdict_ready`, `paper_v6_brain_recommendation`,
  and `honest_verdict`; and
- choose exactly one training-dynamics verdict from `factorized gradient starvation real`,
  `starvation overstated`, or `both parameterizations inadequate`, then propagate that recommendation into
  `docs/research-notes/brain-reinforce-training-dynamics-k15.md`.

`status` MUST be `complete` for terminal artifacts. `honest_verdict` MUST begin
with `complete:` and MUST state the selected training-dynamics verdict.

### SCENARIO-VERIFY-1578: k15 REINFORCE Audit Writes a Paper-v6 Recommendation

Given the deterministic `n=16`, `k=15`, ten-constraint BRAIN target with
`beta=2.0` and uniform initial q-parameters,
When Exp 1578 trains factorized Bernoulli and Linear-AR q models with
REINFORCE,
Then the artifact reports per-parameterization first-1000 gradient-active
fractions and terminal exact KL values
And `brain_training_dynamics_verdict_ready=true` only when both required
parameterizations have complete traces and one allowed verdict is selected
And `paper_v6_brain_recommendation` records whether paper v6 should cite
factorized starvation, treat starvation as overstated, or drop both
parameterizations as inadequate.

### REQ-VERIFY-1588: Bounded Instruction-To-Constraint DSL Pack

The repository shall provide a bounded instruction-to-constraint pack for Exp
1588 that converts natural-language output instructions into a small Carnot DSL
and compiles that DSL into deterministic local validators.

The pack shall:

- define the DSL schema in `python/carnot/verifiers/dsl.py`;
- parse only a fixed, auditable set of instruction patterns, including required
  text, forbidden text, JSON-object requirements, JSON key requirements, word
  count bounds, answer enumerations, and exact bullet counts;
- cap instruction length and constraint count, and fail closed rather than
  executing generated Python or accepting unsupported operators;
- compile every supported DSL constraint into a Python validator that reports
  per-constraint failures;
- emit a PySAT-compatible CNF representation of the hard conjunction whenever
  constraints compile, without requiring the optional `python-sat` package in
  the Python test environment;
- write `results/experiment_1588_nsvif_dsl.json` with `status`,
  `experiment_id`, `dsl_schema_version`, `instructions_tested`,
  `constraints_extracted`, `validators_compiled`, `pysat_cnf_compiled`,
  `python_validator_pass_rate`, `known_good_pass_rate`,
  `known_bad_reject_rate`, `false_accept_rate`,
  `arbitrary_code_execution_path_introduced`, `tests_run`, and
  `honest_verdict`; and
- set terminal `status="complete"` only when all fixture validators compile,
  known-good examples pass, known-bad examples reject, and no arbitrary-code
  execution path is introduced.

### SCENARIO-VERIFY-1588: Natural-Language Instructions Compile To Local Validators

Given bounded natural-language instructions with supported semantic constraints,
When the Exp 1588 DSL parser and compiler run,
Then the parsed pack validates against the local schema,
And the compiled Python validator accepts the known-good output while reporting
specific constraint failures for known-bad output,
And a PySAT-compatible CNF hard-conjunction view is available for the same
constraint IDs,
And `results/experiment_1588_nsvif_dsl.json` records complete validator metrics
with `false_accept_rate=0.0`.

### REQ-VERIFY-1591: Reusable DCCD Structured Verdict Adapter

The repository shall provide a reusable structured-verdict adapter for Exp 1591
that upgrades the Exp 1580 DCCD smoke path into a deterministic API for Carnot
verifier-output JSON schemas.

The adapter shall:

- define `DCCDStructuredVerdictAdapter` under `python/carnot/verifiers/`;
- accept a bounded JSON schema, an optional semantic-path expectation mapping,
  and raw unconstrained draft text;
- compile an llguidance JSON-schema grammar when the optional `llguidance`
  Python bindings are importable, while preserving a deterministic post-decode
  fallback when they are not installed;
- perform DCCD-style structural projection from draft payload to target payload
  without introducing arbitrary code execution;
- return a structured `VerdictRecord` whose `extras` include schema errors,
  semantic errors, false-accept status, backend diagnostics, and the parsed
  payload;
- expose prompt and grammar metadata that downstream local GGUF/llama.cpp
  call sites can use for constrained regeneration; and
- write `results/experiment_1591_dccd_adapter.json` with `status`,
  `experiment_id`, `adapter_module`, `llguidance_backend_available`,
  `fallback_backend_available`, `strict_schema_validity_rate`,
  `semantic_correctness_rate`, `false_accept_count`,
  `arbitrary_code_execution_path_introduced`, `tests_run`, and
  `honest_verdict`.

`status` MUST be `complete` only when the adapter accepts known-good structured
rows, rejects schema-valid semantic false accepts, records backend diagnostics,
and the deterministic fallback remains available without `llguidance`.

### SCENARIO-VERIFY-1591: DCCD Adapter Emits Structured Verdict Records

Given a Carnot verifier-output schema with deterministic semantic-path
expectations and a target payload,
When `DCCDStructuredVerdictAdapter` evaluates unconstrained draft text, a
DCCD-projected payload, and a schema-valid semantic false accept,
Then the draft row records schema/semantic failures without becoming a pass,
the DCCD-projected row returns a `VerdictRecord` with `verdict="pass"` and
zero false accepts,
the semantic false accept returns a `VerdictRecord` with `verdict="fail"` and
`extras["false_accept"]=true`,
and `results/experiment_1591_dccd_adapter.json` records complete metrics and
backend diagnostics for the reusable adapter.

## Implementation Status (REQ-VERIFY-1415/1416/1423/1434/1469/1473/1474/1475/1481/1486/1487/1495/1496/1499/1500/1501/1507/1508/1509/1510/1520/1521/1522/1525/1537/1538/1541/1542/1551/1552/1553/1554/1557/1562/1571/1578/1580/1588/1591)

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
| REQ-VERIFY-1486 | Implemented (`python/carnot/eval/cctu_executable_constraint_microbenchmark.py`) | Implemented (`tests/python/test_experiment_1486_cctu_executable_constraint_microbenchmark.py`) |
| REQ-VERIFY-1487 | Planned (`python/carnot/eval/v1_pairwise_self_verification_vs_energy.py`) | Planned (`tests/python/test_experiment_1487_v1_pairwise_self_verification_vs_energy.py`) |
| REQ-VERIFY-1494 | Planned (`python/carnot/eval/constrainprompt_validator_compiler_audit.py`) | Planned (`tests/python/test_experiment_1494_constrainprompt_validator_compiler_audit.py`) |
| REQ-VERIFY-1495 | Implemented (`python/carnot/eval/interwhen_monitor_prototype.py`) | Implemented (`tests/python/test_experiment_1495_interwhen_monitor_prototype.py`) |
| REQ-VERIFY-1496 | Planned (`python/carnot/eval/hover_safe_prefix_continuation_audit.py`) | Planned (`tests/python/test_experiment_1496_hover_safe_prefix_continuation_audit.py`) |
| REQ-VERIFY-1499 | Implemented (`python/carnot/eval/verifier_ensemble_dry_orthogonality_v2.py`) | Implemented (`tests/python/test_experiment_1499_verifier_ensemble_dry_orthogonality_v2.py`) |
| REQ-VERIFY-1500 | Implemented (`python/carnot/verify/latent_deterministic_gate.py`) | Implemented (`tests/python/test_experiment_1500_latent_deterministic_discipline_gate.py`) |
| REQ-VERIFY-1501 | Implemented (`python/carnot/verify/plan_graph_energy_adapter.py`) | Implemented (`tests/python/test_experiment_1501_gnnverifier_plan_graph_energy_adapter.py`) |
| REQ-VERIFY-1507 | Planned (`python/carnot/verify/safe_dsl_verifier_induction.py`) | Planned (`tests/python/test_experiment_1507_autopyverifier_safe_dsl_induction_pack.py`) |
| REQ-VERIFY-1508 | Planned (`python/carnot/verify/trigger_grammar_certificate_decoder.py`) | Planned (`tests/python/test_experiment_1508_trigger_grammar_certificate_decoder_audit.py`) |
| REQ-VERIFY-1509 | Implemented (`python/carnot/verify/executable_monitor_runtime_adapter.py`) | Implemented (`tests/python/test_experiment_1509_executable_monitor_runtime_adapter.py`) |
| REQ-VERIFY-1510 | Implemented (`python/carnot/verify/plan_graph_structural_contract_gate.py`) | Implemented (`tests/python/test_experiment_1510_plan_graph_structural_contract_gate.py`) |
| REQ-VERIFY-1520 | Implemented (`python/carnot/verify/runtime_contract_e2e_harness.py`) | Implemented (`tests/python/test_experiment_1520_runtime_contract_e2e_harness.py`) |
| REQ-VERIFY-1521 | Implemented (`python/carnot/verify/live_sota_contract_guided_repair.py`) | Implemented (`tests/python/test_experiment_1521_live_sota_contract_guided_repair.py`) |
| REQ-VERIFY-1522 | Implemented (`python/carnot/verify/constraint_dependency_graph_repair.py`) | Implemented (`tests/python/test_experiment_1522_constraint_dependency_graph_repair.py`) |
| REQ-VERIFY-1525 | Implemented (`python/carnot/verify/march_claim_isolation_ablation.py`) | Implemented (`tests/python/test_experiment_1525_march_claim_isolation_ablation.py`) |
| REQ-VERIFY-1537 | Implemented (`python/carnot/verify/beaver_prefix_bound_contracts.py`) | Implemented (`tests/python/test_experiment_1537_beaver_prefix_bound_contracts.py`) |
| REQ-VERIFY-1538 | Planned (`python/carnot/verify/residual_drift_commitment_ledger.py`) | Planned (`tests/python/test_experiment_1538_residual_drift_commitment_ledger.py`) |
| REQ-VERIFY-1541 | Implemented (`python/carnot/verify/claim_isolation_uncertainty_router.py`) | Implemented (`tests/python/test_experiment_1541_claim_isolation_uncertainty_router.py`) |
| REQ-VERIFY-1542 | Implemented (`python/carnot/verify/arm_ebm_soft_value_diagnostic.py`) | Implemented (`tests/python/test_experiment_1542_arm_ebm_soft_value_diagnostic.py`) |
| REQ-VERIFY-1556 | Implemented (`python/carnot/verify/arm_ebm_logprob_telemetry_repair.py`) | Implemented (`tests/python/test_experiment_1556_arm_ebm_logprob_telemetry_repair.py`) |
| REQ-VERIFY-1551 | Implemented (`python/carnot/verify/unified_contract_gate.py`) | Implemented (`tests/python/test_experiment_1551_automata_sat_unified_contract_gate.py`) |
| REQ-VERIFY-1552 | Implemented (`python/carnot/verify/residual_drift_repair_policy.py`) | Implemented (`tests/python/test_experiment_1552_residual_drift_repair_policy.py`) |
| REQ-VERIFY-1553 | Implemented (`python/carnot/verify/claim_isolation_router_scale.py`) | Implemented (`tests/python/test_experiment_1553_claim_isolation_router_scale.py`) |
| REQ-VERIFY-1554 | Planned (`python/carnot/verify/product_line_staged_scale_v4.py`) | Planned (`tests/python/test_experiment_1554_product_line_staged_scale_v4.py`) |
| REQ-VERIFY-1557 | Implemented (`python/carnot/verify/verification_compute_router.py`) | Implemented (`tests/python/test_experiment_1557_weaver_verification_compute_router.py`) |
| REQ-VERIFY-1562 | Implemented (`python/scripts/dt_brain_correlations_verification.py`) | Implemented (`tests/python/test_experiment_1562_brain_linear_ar_k_sweep.py`) |
| REQ-VERIFY-1571 | Implemented (`python/carnot/training/ar_reinforce_stepwise_baseline.py`) | Implemented (`tests/python/test_experiment_1571_step_wise_baseline_ar_reinforce.py`) |
| REQ-VERIFY-1578 | Implemented (`python/carnot/training/brain_reinforce_training_dynamics.py`) | Implemented (`tests/python/test_experiment_1578_brain_reinforce_training_dynamics.py`) |
| REQ-VERIFY-1580 | Implemented (`python/carnot/reporting/dccd_jsonschemabench_sota_structured_output_smoke.py`) | Implemented (`tests/python/test_experiment_1580_dccd_jsonschemabench_sota_structured_output_smoke.py`) |
| REQ-VERIFY-1588 | Implemented (`python/carnot/verifiers/dsl.py`) | Implemented (`tests/python/test_experiment_1588_nsvif_dsl.py`) |
| REQ-VERIFY-1591 | Implemented (`python/carnot/verifiers/dccd_adapter.py`) | Implemented (`tests/python/test_experiment_1591_dccd_adapter.py`) |

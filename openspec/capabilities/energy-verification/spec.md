# Energy Verification Capability

## Requirements

### REQ-ENERGY-6746: Oracle-Distinct Diagnostic Energy

Exp6746 SHALL freeze its feature schema before it joins any outcome label.
The schema SHALL contain only pre-oracle syntax, formula topology, encoder
structure, local inconsistency, and model-independent format features. The
dual-encoding, encoder-A-only, encoder-B-only, and undifferentiated-scalar
arms SHALL use equal training capacity, seeds, and budgets.

The experiment SHALL deny current-row exact outcomes, diagnoses, labels,
answer keys, solver work or counters, certificate-validity fields, and
deterministic proxies derived from them. A denylist, family-split, taint, or
proxy failure SHALL set `oracle_leakage_detected` to true and disqualify
positive credit.

Evaluation SHALL use immutable family-held-out splits. No family can occur
in both training and evaluation for one fold. Base and relabel rows SHALL be
paired by model and source pair. Reported metrics SHALL be recomputed from
retained unit-arm rows. These metrics SHALL include AUROC, AUPRC,
calibration, localization accuracy, bootstrap intervals, and paired relabel
deltas. `heldout_reasoning_error_auroc` SHALL come from raw held rows.

Before training, Exp6746 SHALL require the Exp6745 artifact,
`dual_encoding_corpus_ready=true`, the frozen source family assignments, at
least two diagnosis classes in every held family, and the registered minimum
row and class counts for bootstrap resampling. A failed check SHALL stop all
training. It SHALL emit a complete artifact whose `honest_verdict` starts
with `complete_blocked_diagnostic_energy` and whose `gate_check_summary`
names the failed check, expected value, and observed value.

The terminal artifact SHALL be
`results/experiment_6746_oracle_distinct_diagnostic_energy.json`. It SHALL
include `field_principles`, `inference_substrate`, `duration_s`,
`random_seed`, `reproducibility_checksum`, `verifier_is_oracle`,
`feature_schema`, `oracle_feature_denylist`, `rows`,
`heldout_metrics_by_family`, `paired_relabel_metrics`,
`heldout_reasoning_error_auroc`, `oracle_leakage_detected`,
`diagnostic_energy_ready`, `gate_check_summary`, `verdict_class`, and
`honest_verdict`. Field principles SHALL cover every field and every gate.
Readiness means that all four arms and audits completed. It does not assert a
positive scientific result.

### SCENARIO-ENERGY-6746-DENYLIST: Prohibited Features Fail Closed

**Given** a frozen allowed schema or a schema containing an oracle field

**When** Exp6746 audits every feature path and taint source

**Then** the allowed schema passes and any prohibited path records a leakage
failure before model training.

**Spec traces:** REQ-ENERGY-6746

### SCENARIO-ENERGY-6746-SPLITS: Held Families Stay Disjoint

**Given** rows from the three frozen source families

**When** Exp6746 builds each held-family fold

**Then** the held family is absent from training and each row appears in the
declared side exactly once.

**Spec traces:** REQ-ENERGY-6746

### SCENARIO-ENERGY-6746-RELABEL: Relabel Mates Stay Paired

**Given** base and relabel rows for each model and source pair

**When** Exp6746 constructs paired relabel units

**Then** each unit contains exactly one base and one relabel row from the same
family, and incomplete or duplicate pairs fail closed.

**Spec traces:** REQ-ENERGY-6746

### SCENARIO-ENERGY-6746-METRICS: Metrics Derive From Unit Rows

**Given** retained row-level energy, prediction, target, and localization
values

**When** Exp6746 recomputes the report

**Then** AUROC, AUPRC, calibration, and localization accuracy equal the
values derived from those rows, not stored summary values.

**Spec traces:** REQ-ENERGY-6746

### SCENARIO-ENERGY-6746-PRECONDITION: Single-Class Families Block Training

**Given** a ready Exp6745 corpus with fewer than two diagnosis classes in any
held family

**When** Exp6746 evaluates its registered gates

**Then** no arm trains, readiness is false, and the complete blocked artifact
records the observed per-family class counts.

**Spec traces:** REQ-ENERGY-6746

### REQ-ENERGY-6958: Input-Convex Factor Energy Canary

Carnot SHALL provide a bounded CPU canary at
`python/carnot/experiment_6958_convex_factor_energy_canary.py`. The command
`.venv/bin/python scripts/experiments/experiment_6958_convex_factor_energy_canary.py --date 20260904`
SHALL write `results/experiment_6958_convex_factor_energy_canary.json`. This
canary adapts the input-convex factor-sum shape described by
arXiv:2605.23395 to the Exp6955 mapping schema. It SHALL NOT claim to
reproduce that paper or to establish a hardware result.

Before candidate construction, the canary SHALL require the Exp6955
`reformulation_fixture_ready_score` to equal one, complete mapping,
feasibility-witness, and objective-order-witness rows, template-frozen splits,
working PyTorch CPU tensors and gradients, and a writable checkpoint
directory. Any failed precondition SHALL prevent fitting and SHALL still write
a schema-complete artifact with both scores zero,
`verdict_class="blocked"`,
`honest_verdict="blocked_convex_factor_energy_canary"`, and a
`gate_check_summary` naming each failed check, expected value, and observed
value.

For every exact-equivalent fixture row, the canary SHALL pair the unchanged
mapping with one deterministic corruption frozen before fitting. The
corruption roster SHALL exercise variable coverage, domain correspondence,
affine consistency, objective direction, and order witnesses. Feature tensors
SHALL encode those five pre-oracle factor channels with an explicit padding
mask. They SHALL exclude final exact labels, claimed relation, exact solver
status, authority decisions, quarantine state, and any derived proxy for those
fields. Generator templates and normalized fixture identities SHALL not cross
train, calibration, held-out, or prospective-family splits.

The experiment SHALL compare five arms: an input-convex factor sum, a
parameter-matched unconstrained MLP factor sum, a linear factor score, exact
hand penalties, and an input-convex factor sum fitted to deterministically
shuffled labels. Learned arms SHALL use the same examples, epoch count,
optimizer-call count, starts, and seeds. The two nonlinear learned arms SHALL
have equal trainable parameter counts. Exact hand penalties SHALL remain a
transparent non-learned upper control and SHALL not be presented as the
learned-energy headline. Every start and failure SHALL remain visible.

All input-convex paths SHALL project the weights between convex hidden units,
the nonnegative input paths, and the output paths onto the nonnegative
orthant. Tests and artifact rows SHALL check projection, empirical Jensen
inequalities, directional finite differences, monotonicity of the input
gradient, factor-permutation invariance, padding-mask invariance, and
deterministic replay. Projected input optimization SHALL start from multiple
fixed points and report convergence or failure without dropping a start.

Evaluation SHALL report paired valid-before-corrupt exact ordering with
explicit half credit for energy ties, calibration, CPU latency, and transfer
from smaller fitted factor counts to larger held-out factor counts. Paired
metrics and deterministic bootstrap CI95 intervals SHALL compare the convex
learned arm with the unconstrained MLP and linear learned controls on the same
held-out pair/seed rows. Missing comparisons and one-class cells SHALL remain
null rather than becoming zero.

`convex_factor_run_complete_score` SHALL equal one only when every arm, seed,
split, optimizer start, and size band has a terminal row; every nonlinear
budget match passes; all checkpoints replay their energies and ordering
metrics in a fresh process; and every required row surface is populated.
`convex_factor_positive_score` SHALL equal one only when completion is one,
the convex learned arm passes every Jensen, finite-difference, gradient,
projection, permutation, and mask check, and its held-out exact-ordering delta
has a paired CI95 lower bound strictly above zero against both the
unconstrained MLP and linear learned controls. Exact hand penalties SHALL not
enter this positive gate. A complete positive run SHALL use
`verdict_class="positive"`; a complete score-zero run SHALL use
`verdict_class="null"`; and incomplete non-blocked work SHALL use
`verdict_class="partial"`. Complete verdicts SHALL begin with `complete_`.

The artifact SHALL include `field_principles`, `preconditions_checked`,
`inference_substrate`, `duration_s`, `source_artifact_hashes`, `rows`,
`feature_rows`, `corruption_rows`, `split_rows`, `arm_rows`, `seed_rows`,
`training_rows`, `convex_weight_rows`, `jensen_rows`,
`finite_difference_rows`, `projection_rows`, `optimizer_rows`,
`ordering_rows`, `calibration_rows`, `size_transfer_rows`, `latency_rows`,
`baseline_rows`, `shuffled_label_rows`, `paired_metric_rows`,
`confidence_interval_rows`, `label_isolation_rows`,
`fresh_process_replay_rows`, `checkpoint_paths`, `random_seed`,
`reproducibility_checksum`, `convex_factor_run_complete_score`,
`convex_factor_positive_score`, `gate_check_summary`, `verifier_is_oracle`,
`verdict_class`, and `honest_verdict`. `field_principles` SHALL give one
scientific reason for every required field and both scores.
`inference_substrate` SHALL equal
`small_cpu_input_convex_factor_energy_training`, and `verifier_is_oracle`
SHALL be false because exact fixture evidence evaluates the method rather than
being the method under test.

#### SCENARIO-ENERGY-6958-PRECONDITIONS: Missing Fixture Evidence Blocks Fitting

**Given** a fixture with a false readiness score, incomplete mappings or
witnesses, leaking splits, unavailable CPU autograd, or an unwritable target

**When** the canary performs preflight

**Then** no arm fits and the blocked artifact records expected and observed
values for every failed check.

#### SCENARIO-ENERGY-6958-FACTORS: Factors Are Isolated, Permutable, And Masked

**Given** paired valid and frozen-corrupt mappings

**When** their structured factors are encoded, permuted, or padded

**Then** all five channels can expose their named corruption, forbidden label
and solver fields are absent, and factor-sum energy is invariant to permutation
and masked padding.

#### SCENARIO-ENERGY-6958-CONVEXITY: Projection Preserves Input Convexity

**Given** an input-convex factor network with deliberately negative constrained
weights

**When** nonnegative projection and empirical property checks run

**Then** all constrained weights are nonnegative and Jensen, finite-difference,
and monotone-gradient tolerances pass for the projected network.

#### SCENARIO-ENERGY-6958-BUDGET: Learned Arms Use Matched Training Budgets

**Given** the frozen training candidates and registered seeds

**When** all learned arms fit

**Then** examples, epochs, optimizer calls, starts, and seeds match, the two
nonlinear arms have equal parameter counts, and shuffled labels are fixed
before their optimizer starts.

#### SCENARIO-ENERGY-6958-ORDERING: Ordering And Ties Stay Paired

**Given** one valid and one corrupt candidate for each held-out mapping

**When** energy ordering and paired intervals are reduced

**Then** wins, losses, and half-credit ties stay on the same pair and seed,
and neither a zero delta nor a CI95 lower bound equal to zero passes the
positive gate.

#### SCENARIO-ENERGY-6958-OPTIMIZATION: Projected Starts Terminate Deterministically

**Given** multiple fixed initial factor tensors inside the registered box

**When** projected first-order minimization runs against the convex energy

**Then** every start terminates, stays inside the box, reports its convergence
status, and repeated seeds reproduce the same terminal values.

#### SCENARIO-ENERGY-6958-TRANSFER: Larger Factor Counts Remain Held Out

**Given** training candidates in the smaller registered factor-count band

**When** the same factor-sum parameters score larger held-out mappings

**Then** size-band ordering and latency are reported without refitting or
padding leakage.

#### SCENARIO-ENERGY-6958-REPLAY: Fresh Processes Reload Every Checkpoint

**Given** saved learned-arm checkpoints and a serialized replay manifest

**When** a fresh Python process reloads them

**Then** checkpoint hashes, candidate energies, and exact-ordering metrics
match the parent run before completion can equal one.

## Implementation Status (REQ-ENERGY-6958)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-ENERGY-6958 and SCENARIO-ENERGY-6958-* | Implemented (`python/carnot/experiment_6958_convex_factor_energy_canary.py`; `scripts/experiments/experiment_6958_convex_factor_energy_canary.py`) | Implemented (`tests/python/test_experiment_6958_convex_factor_energy_canary.py`; projection, Jensen, finite differences, gradient monotonicity, factor permutation, padding masks, tie handling, split isolation, deterministic fitting, blocked preflight, checkpoint binding, fresh-process replay, and 100% new-module statement coverage) |

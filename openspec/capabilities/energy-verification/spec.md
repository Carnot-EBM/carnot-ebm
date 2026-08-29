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

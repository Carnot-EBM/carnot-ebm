# Autoresearch Capability Specification

**Capability:** autoresearch
**Version:** 0.1.0
**Status:** Draft
**Traces to:** FR-11

## Overview

Defines Carnot's autonomous self-improvement loop: a pipeline where the system proposes modifications to its own architecture, training algorithms, or hyperparameters, evaluates them against objective energy-based benchmarks, and incorporates proven improvements — all without human supervision.

The fundamental insight: because the energy function provides mathematical ground truth (energy decreased = real improvement, energy didn't = rejected), the system has an objective evaluator that cannot be gamed. This is the property that makes autonomous self-improvement safe and convergent, unlike LLM-based self-evaluation which inherits all the hallucination problems of the base model.

## Requirements

### REQ-AUTO-001: Benchmark Suite

The system shall provide a standard benchmark suite of energy landscape problems with known optimal solutions, including:
- **Analytical benchmarks**: DoubleWell, Rosenbrock, Ackley, Rastrigin (known global minima)
- **Structured benchmarks**: Sudoku (constraint satisfaction), graph coloring, scheduling
- **Statistical benchmarks**: Gaussian mixture models (known distribution parameters)

Each benchmark shall define:
- Energy function
- Known optimal energy value (or bound)
- Evaluation metrics (convergence speed, final energy, sample quality, wall-clock time)
- Pass/fail thresholds

**Cross-language note:** The Ackley benchmark in Python/JAX adds a small epsilon (1e-10) inside the sqrt to prevent NaN gradients from jax.grad at the origin (d/dx sqrt(0) is undefined). The Rust implementation uses numerical gradients instead. Energy values may differ by up to ~1e-5 near the origin. This is an intentional implementation divergence documented here rather than in the code alone.

### REQ-AUTO-002: Baseline Registry

The system shall maintain a registry of baseline performance metrics for the current production models and algorithms, stored as versioned JSON files. Each entry records:
- Benchmark name
- Algorithm/model configuration
- Metrics (final energy, convergence steps, wall-clock time, memory usage)
- Git commit hash of the implementation
- Timestamp

### REQ-AUTO-003: Hypothesis Generation Interface

The system shall define a structured format for improvement hypotheses:
- **Target**: what is being modified (architecture, sampler, training algorithm, hyperparameter)
- **Rationale**: why this might improve performance (optional, for human-readable logging)
- **Specification**: a complete, executable description of the change (Python/JAX code or configuration diff)
- **Expected impact**: which benchmark metrics should improve and by how much
- **Risk assessment**: which metrics might degrade

### REQ-AUTO-004: Sandbox Execution

The system shall execute hypothesis code in an isolated environment with:
- Read-only access to immutable validation datasets
- Hard timeout (configurable, default 30 minutes)
- Memory limit (configurable, default 16GB)
- No network access
- No write access to production code or data
- Captured stdout/stderr and metrics output

### REQ-AUTO-005: Evaluation Protocol

The system shall evaluate sandbox results against baselines using:
- **Primary gate**: benchmark energy must be <= baseline energy (improvement or at least no regression)
- **Secondary gate**: wall-clock time must be <= 2x baseline time (not catastrophically slower)
- **Tertiary gate**: memory usage must be <= 2x baseline usage
- All three gates must pass for the hypothesis to advance

### REQ-AUTO-006: Cross-Language Validation

For hypotheses that pass evaluation in Python/JAX:
- The system shall support transpilation to Rust (initially agent-assisted, eventually automated)
- The Rust implementation must produce energy values within floating-point tolerance of the JAX implementation on the same inputs
- The Rust implementation must meet or exceed the JAX implementation's wall-clock performance

### REQ-AUTO-007: Rollback Mechanism

The system shall support automatic rollback:
- If production energy metrics degrade by more than a configurable threshold (default 5%) over a monitoring window (default 1 hour), automatically revert to the previous version
- All reverted changes are logged with the regression metrics
- Reverted hypotheses are added to a "rejected" registry to prevent re-proposal

### REQ-AUTO-008: Experiment Logging

The system shall maintain a structured experiment log recording:
- Hypothesis (full specification)
- Sandbox results (metrics, stdout, stderr)
- Evaluation verdict (pass/fail per gate)
- If transpiled: Rust validation results
- If deployed: production monitoring results
- If rolled back: regression metrics and reason

### REQ-AUTO-009: Safety Invariants

The following invariants shall hold at all times during autoresearch:
- Validation data is never modified by any component
- Production model can always be restored to the last known-good state within 60 seconds
- No hypothesis can modify the evaluation protocol or benchmark definitions
- No hypothesis can modify its own evaluation criteria
- The system shall halt and alert if it detects more than N consecutive failures (configurable, default 10)

### REQ-AUTO-010: Improvement Composition

When multiple hypotheses independently pass evaluation, the system shall:
- Test them in combination (improvements may conflict)
- Apply the combination only if joint evaluation passes all gates
- If conflicting, rank by primary gate improvement magnitude and apply the best

### REQ-AUTO-011: Trajectory Analysis

The system shall provide parallel analyst sub-agents that extract structured lessons from experiment outcomes:
- **Error analysts** receive a failed experiment's full trajectory (hypothesis code, sandbox metrics, evaluation verdict, error messages) and diagnose the root cause via LLM reasoning (e.g., "gradient explosion due to Rosenbrock curvature exceeding step size")
- **Success analysts** receive an accepted experiment's trajectory and extract the generalizable optimization pattern (e.g., "annealing schedules improve convergence on multi-basin landscapes")
- Analysts run in parallel via thread pool and produce structured `Lesson` objects with: title, description, concrete examples, confidence score, applicable benchmarks, model tier, and lesson type
- Analyst dispatch is configurable (can be disabled to save LLM cost)

### REQ-AUTO-012: Skill Directory

The system shall maintain a persistent, evolving optimization playbook (skill directory) that accumulates lessons across iterations:
- **SKILL.md**: Natural-language optimization guide, periodically rewritten by LLM from accumulated lessons
- **scripts/**: Proven sampler configurations and code snippets extracted from successful hypotheses
- **references/**: Benchmark-specific edge cases and niche patterns (low-frequency lessons)
- **lessons.json**: Structured lesson store with confidence scores and metadata
- The skill directory shall be serialized as `to_prompt_context()` and injected into the hypothesis generator's prompt, replacing the shallow `recent_failures` list with structured knowledge
- Maximum lesson count shall be configurable (default 200) to prevent unbounded growth

### REQ-AUTO-013: Hierarchical Lesson Consolidation

The system shall consolidate raw lessons into a conflict-free set via hierarchical tree-reduction:
- Group lessons into batches of configurable size (default 32)
- For each batch, use LLM to: deduplicate equivalent lessons (merging confidence scores), resolve contradictory lessons (keeping the better-supported one), and extract cross-cutting meta-patterns
- Repeat reduction until a single batch remains (L = ceil(log_batch(N)) levels)
- Filter lessons below a configurable minimum confidence threshold (default 0.3)
- Consolidation runs periodically (configurable interval, default every 5 iterations)

### REQ-AUTO-014: Cross-Tier Skill Transfer

The system shall support transfer of optimization knowledge across model tiers:
- Lessons learned on fast-to-evaluate tiers (Ising) shall be available when generating hypotheses for slower tiers (Gibbs, Boltzmann)
- Each lesson is tagged with its originating model tier and applicable benchmarks
- The `to_prompt_context()` method accepts a target model tier and includes relevant lessons from other tiers
- Tier-specific edge cases are stored in the references subdirectory, not propagated as general lessons

### REQ-AUTO-018: Calibrated-Decision Benchmark

Per CLAUDE.md "Energy-Based Calibrated-Decision Training Floor" (2026-09-18), the
system shall provide an autoresearch benchmark that trains a small energy-based
selector to convert an existing verifier's raw signal into a calibrated decision,
rather than only tuning a linear combination of that signal (the limitation of
REQ-AUTO-025's `verifier_auroc` benchmark):

- A hypothesis trains a `carnot.models.gibbs.GibbsModel` (fixed architecture:
  `input_dim=2, hidden_dims=[4]`) via `carnot.training.nce.nce_loss`, over the raw
  PCIB entity-uptake / falsifiability-score features of `data/fover_corpus_v4.json`
  (the same corpus REQ-AUTO-025 uses, but the two features unweighted, not the
  probe's weighted combination), treating "correct" rows as NCE's low-energy data
  and "incorrect" rows as its high-energy noise.
- `GibbsModel`/`GibbsConfig`/`nce_loss` are handed to the hypothesis directly via
  `benchmark_data` (never importable — the sandbox blocks all `carnot` imports),
  the same trust pattern REQ-AUTO-025 established for `PCIBProbe`.
- The harness recomputes TWO independently-measured numbers from a hypothesis's
  claimed `final_state` (never a self-reported value), in a fresh subprocess per
  REQ-AUTO-025's CRITICAL-2 hardening: `final_energy = 1.0 - auroc` (the sole
  gating metric, matching every other benchmark's convention) and `brier` (mean
  squared error between `sigmoid(energy)` and the true label — a calibration
  measure, reported but not yet gating).
- A weight set that scores every held-out row identically shall be rejected as
  degenerate from the accept path, except when honestly measuring the untrained
  default's seed value (which is exactly the degenerate/chance case by
  construction — a freshly constructed `GibbsModel`'s output layer is
  zero-initialized).

#### Experiment-local proper-score protocol

An experiment-local protocol SHALL leave the standing conductor benchmark
unchanged. It SHALL freeze the two raw PCIB features and the 2-4-1 Gibbs head.
The head has 17 parameters. Its typed output SHALL contain `decision`,
`p_incorrect`, `confidence_correct`, `model_version`, and `reason`.
`confidence_correct` SHALL always equal `1 - p_incorrect`, including escalation.

Before fitting, the protocol SHALL form connected groups from shared
`question_id` values and exact normalized step-text duplicates. It SHALL
quarantine rows with missing identifiers. The fixed salt
`v648-calibration-7382` SHALL assign complete groups to training (40 percent),
probability calibration (20 percent), policy calibration (20 percent), and
final test (20 percent). Each partition SHALL contain at least ten incorrect
examples. The manifest SHALL lock source hashes and exact group membership.

The training partition SHALL also define a group-disjoint controlled archive
replay. An independent fixed hash SHALL order its groups. The first half SHALL
initialize weights, and the remaining groups SHALL form later blocks. The
online reducer SHALL use 10,000 paired moving-block draws, block length 32,
sensitivity length 64, and seed 7386307. This replay is not real-world temporal
evidence. The final split is experiment-held-out, not virgin external evidence.

The protocol SHALL pre-register seeds 7382001 through 7382005, at most 500
optimizer steps, and one optimizer configuration. It SHALL compare training
prevalence, L2 logistic calibration, raw balanced-NCE Gibbs, prior-corrected
NCE Gibbs, and natural-prevalence Bernoulli Gibbs arms. NCE prior correction
SHALL add `log(pi / (1 - pi))` to energy before the sigmoid. Affine probability
calibration SHALL use only its own partition. Unknown labels and one-class
fitting inputs SHALL fail closed.

The policy SHALL freeze threshold pairs `(0.005, 0.50)`, `(0.01, 0.75)`,
`(0.02, 0.90)`, `(0.03, 0.95)`, and `(0.05, 0.99)`. Policy selection SHALL use
only policy-calibration groups and one label-blind representative per group.
One-sided exact binomial bounds SHALL use simultaneous correction over all
arms, seeds, threshold pairs, and action classes. Incorrect-accept risk SHALL
not exceed 0.05. Correct-reject risk SHALL not exceed 0.10. An empty selected
action SHALL have no certificate and SHALL be disabled.

Primary comparisons SHALL use complete-population per-group Brier and log loss,
plus typed-policy risk, coverage, and utility. A paired group bootstrap SHALL
use 10,000 draws and seed 7382307. A value claim requires a Brier 95-percent
upper delta below zero against both prevalence and logistic controls, non-worse
log loss, coverage of at least 0.25, and no certified-risk coverage loss against
logistic. Archive-only evidence SHALL remain a bounded claim. Insufficient
evidence SHALL be a completed null.

#### SCENARIO-AUTO-7382-01: Connected groups cannot cross roles

**Given** rows that share either a question identifier or normalized step text,
**when** the protocol builds and partitions connected groups,
**then** every connected row has one group and one partition, while rows with a
missing identifier appear in explicit quarantine records.

#### SCENARIO-AUTO-7382-02: Probability conversion is stable and prior-aware

**Given** finite and extreme energies and a training prevalence in `(0, 1)`,
**when** the scorer converts energy to incorrect-label probability,
**then** it stays finite, preserves the energy sign convention, and applies the
training-only log-odds correction. One-class prevalence fails closed.

#### SCENARIO-AUTO-7382-03: Typed decisions preserve confidence meaning

**Given** valid probabilities and frozen accept and reject thresholds,
**when** the policy emits accept, reject, or escalate,
**then** confidence always means correctness probability. Invalid probability
input escalates with an explicit reason and no invented numeric confidence.

#### SCENARIO-AUTO-7382-04: Empty action sets cannot certify safety

**Given** zero selected accept or reject groups,
**when** the exact risk certificate is reduced,
**then** the action is disabled and no certificate is emitted. Non-empty sets
use the declared simultaneous correction and preserve their exact counts.

#### SCENARIO-AUTO-7382-05: Test labels stay behind the evaluator boundary

**Given** separate training, probability-calibration, policy-selection, and
final-test readers,
**when** ordinary protocol code requests rows,
**then** only the trusted evaluator can read final-test labels. Fixture checks
may prove plumbing, but they cannot establish learned calibration value.

### REQ-AUTO-7385: Execute the sealed calibrated-decision training protocol

Exp7385 SHALL authenticate the exact Exp7382 artifact before dependent work.
The check SHALL include its byte hash, terminal status, verdict class,
adversarial flag, readiness score, required-gate summary, partition hash,
partition membership, source corpus hash, and trusted-label sidecar hash.
Missing, blocked, partial, disqualified, flagged, quarantined, or changed input
SHALL produce a terminal blocked artifact. The block SHALL name the exact path,
field, expected value, and observed value. Historical diagnostic inputs SHALL
not become current readiness evidence.

The experiment SHALL use the frozen Exp7382 rows, five seeds, five arms,
2-4-1 architecture, Adam settings, and 500-step limit. It SHALL fit only on
training rows. The five arms SHALL remain training prevalence, L2 logistic,
raw balanced-NCE Gibbs, prior-corrected NCE Gibbs, and natural-prevalence
Bernoulli Gibbs. The last arm is the pre-registered primary value arm. Final
outcomes SHALL not select an arm or architecture. Each arm and seed SHALL keep
its real loss curve, initial and final weight hashes, update count, numeric
weights, optimizer state, and checkpoint hash.

Affine probability calibration SHALL use only probability-calibration groups.
Policy selection SHALL use only policy-calibration groups. It SHALL evaluate
all frozen threshold pairs. It SHALL select the largest certified coverage,
then the largest utility, then the earliest registered pair. Certification
SHALL use the pre-registered 250-test correction before final labels are read.
An empty or uncertified action SHALL escalate.

After all weights, transforms, and policies are sealed, a fresh trusted
subprocess SHALL score final-test labels exactly once. Its request SHALL contain
plain numeric state only. It SHALL reject missing hashes, extra keys, wrong
shapes, non-finite values, and executable payloads. A direct Gibbs energy and
an independent NumPy recomputation SHALL agree on development inputs before
final scoring.

The artifact SHALL emit one final row for every arm, seed, and final group.
Each row SHALL contain the group ID, label, raw energy, calibrated probability,
typed decision, Brier contribution, log-loss contribution, correctness, action
risk, and measured scoring cost. It SHALL retain prevalence and logistic rows
when a learned arm wins. It SHALL report Brier, log loss, AUROC, PR-AUC,
prevalence, typed-policy risk, coverage, utility, policy certificates, and
10,000-draw paired group intervals with seed 7382307.

`calibration_value_score` SHALL equal one only when the primary
natural-prevalence Bernoulli Gibbs arm has a Brier interval strictly below both
controls, non-worse mean log loss against both controls, certified and observed
action risk within the registered budgets, at least 0.25 coverage, and no
paired coverage loss against logistic. A valid efficacy miss SHALL be a
completed null. `decision_capture_complete_score` SHALL depend on sealed row
and validation completeness, not benefit. `promotion_score` SHALL remain zero.
The evidence SHALL remain a reused single-archive result and SHALL not close a
general verifier-moat gap.

The current run SHALL declare `MODEL_SPECS=[]`, `model_invoked=false`, zero
current LLM invocation counts, `inference_substrate_class=no_model_load`, and
`execution_venue=host`. It SHALL record CPU/JAX work and Gibbs updates under
`small_ebm_training`. It SHALL save numeric checkpoints below
`results/checkpoints/experiment_7385_v648_decision_training/`.

The workflow SHALL use the Exp7358 command-plan boundary and the Exp7303
runner. It SHALL run focused tests, 100 percent changed-module coverage, scoped
Ruff check and format, changed-module mypy, scoped specification coverage, a
cold artifact reducer, adversarial verification, strict verdict-row
consistency, the declared entrypoint, and an independent cold replay. No
numbered E2E check applies because this is an isolated experiment.

#### SCENARIO-AUTO-7385-01: Ineligible protocol input blocks before fitting

**Given** an Exp7382 artifact with a changed hash, ineligible terminal class,
failed required gate, or adversarial flag,
**when** Exp7385 checks its prerequisites,
**then** it emits an exact `blocked_*` gate summary
**and** performs no fit, calibration, policy selection, or final scoring.

#### SCENARIO-AUTO-7385-02: Training and calibration roles stay sealed

**Given** the authenticated Exp7382 partition membership,
**when** all arm and seed states are fitted and calibrated,
**then** weights use training rows only and affine transforms use probability
calibration rows only
**and** every completed fit has a numeric checkpoint and real update record.

#### SCENARIO-AUTO-7385-03: Policy choices precede final-test access

**Given** fitted and probability-calibrated states,
**when** threshold pairs are certified and selected,
**then** only policy-calibration representatives determine the selection
**and** empty or uncertified actions map to escalation.

#### SCENARIO-AUTO-7385-04: Trusted scoring rejects unsafe state

**Given** a sealed plain-numeric scoring request,
**when** it has an extra key, wrong shape, non-finite value, executable value,
or missing protocol hash,
**then** the trusted scorer rejects it before it reads final labels
**and** valid state produces exactly one row per arm, seed, and final group.

#### SCENARIO-AUTO-7385-05: Completion and benefit remain independent

**Given** all pre-registered rows and required checks are complete,
**when** the primary Brier, log-loss, risk, and coverage conjunction misses,
**then** decision capture is complete and calibration value remains zero
**and** the terminal verdict is null rather than partial or positive.

### REQ-AUTO-7386: Replay delayed-feedback online decisions without future-label leakage

Exp7386 SHALL authenticate the exact Exp7382 protocol artifact before dependent
work. The gate SHALL check the artifact hash, terminal status, allowed verdict
class, adversarial flag, ready score, required-gate summary, partition hash,
source corpus hash, and the frozen online membership. A missing, quarantined,
blocked, partial, disqualified, flagged, or changed input SHALL produce a
terminal `blocked_*` artifact with the exact expected and observed gate values.

The experiment SHALL initialize a natural-prevalence Bernoulli Gibbs head and
an L2 logistic comparator on only the frozen initialization half. It SHALL use
seeds 7382001 through 7382005. It SHALL replay only Exp7382 later groups in the
fixed hash order and in a constructed four-block feature-quantile order. The
second order is a covariate-shift challenge and SHALL not relabel examples.

Each order SHALL run feedback delays of zero and eight group decisions. Every
fourth group SHALL omit feedback by a label-blind position rule. Each adaptive
arm SHALL predict and record its probability, typed decision, state version,
pending feedback count, and state hash before feedback is visible. Admitted
feedback SHALL cause at most one SGD step per group. Replay buffers SHALL hold
at most 128 groups. A frozen Gibbs arm, bounded online Gibbs arm, online
logistic arm, recent-frequency arm, and no-feedback Gibbs arm SHALL remain
separate. The experiment SHALL also run a feedback-permutation control with the
same delays and update count.

After half of each stream, every adaptive arm SHALL cold-restart from durable
numeric state. Development controls SHALL cover corrupt-state rollback and a
revoked-label update. Erasure controls SHALL restore the snapshot from before
each later block and compare predictions on the same later groups. No future
label SHALL affect an earlier prediction.

The artifact SHALL retain every group, arm, seed, order, and delay outcome.
Each row SHALL include probability, loss, action, update count, model lineage,
state hash, pending feedback, service time, and feedback admission. It SHALL
report 10,000 paired moving-block resamples with seed 7386307 and block length
32. It SHALL repeat the frozen sensitivity check at block length 64. Identical
block indices SHALL compare arms, with seeds averaged inside each block.

Conditions SHALL remain separate. Online value requires the later-block Brier
95-percent upper delta below zero against frozen Gibbs and online logistic, no
increase in incorrect accepts at matched coverage, and strictly smaller benefit
after true-feedback erasure. A permutation benefit or zero decision change
invalidates a learning claim. These intervals describe only the fixed archive
replay. They are not IID population certificates.

`online_capture_complete_score` SHALL equal one only when all registered
streams, controls, causal ordering checks, restart checks, and required
validation complete. `online_learning_value_score` SHALL equal one only when
every value gate passes for every order and delay condition. A valid efficacy
miss SHALL be a completed null. `promotion_score` SHALL remain zero.

The run SHALL declare `MODEL_SPECS=[]`, `model_invoked=false`, zero current LLM
invocation counts, `inference_substrate_class=no_model_load`, and
`execution_venue=host`. It SHALL record CPU/JAX fitting under
`small_ebm_training`, set `continuous_self_learning_task=true`, keep the
controller experiment-local and default-off, and make no unmeasured speedup
claim. The validation workflow SHALL use the Exp7358 command plan and Exp7303
runner. It SHALL execute the declared entrypoint, an independent cold replay,
adversarial verification, and strict verdict-row consistency. No numbered E2E
check applies to this isolated experiment.

#### SCENARIO-AUTO-7386-01: An ineligible protocol blocks before fitting

**Given** an Exp7382 artifact with a changed hash, unavailable terminal class,
failed required gate, or adversarial flag,
**when** Exp7386 checks all upstream conditions,
**then** it emits a terminal blocked artifact with an exact gate summary
**and** performs no initialization, replay, update, or resampling.

#### SCENARIO-AUTO-7386-02: Prediction precedes delayed feedback

**Given** a frozen stream and a delay of zero or eight decisions,
**when** an adaptive arm handles one group,
**then** its durable prediction references the state that existed before any
feedback released at that decision
**and** omitted or revoked feedback never changes the adaptive state.

#### SCENARIO-AUTO-7386-03: Restart and rollback preserve causal state

**Given** a durable midpoint snapshot and a pre-update snapshot,
**when** the controller cold-restarts or encounters corrupt state,
**then** valid state resumes with identical predictions and lineage
**and** corrupt state rolls back without admitting an update.

#### SCENARIO-AUTO-7386-04: Erasure and permutation expose false learning

**Given** the same later groups and admitted-update budget,
**when** true feedback is erased or permuted,
**then** paired rows compare the resulting probabilities and actions
**and** retained benefit under erasure or permutation prevents a value claim.

#### SCENARIO-AUTO-7386-05: Completion is independent of online value

**Given** all streams, controls, rows, resamples, and required checks complete,
**when** any later-block benefit, action-safety, erasure, permutation, or
decision-change gate fails,
**then** online capture remains complete and online learning value stays zero
**and** the terminal verdict is complete null rather than partial or positive.

### REQ-AUTO-7410: Seal an attributed source-context calibration corpus

Exp7410 SHALL fetch only the README, development Parquet, and test Parquet
from `s-nlp/EnokiQA` revision
`06638fd6fa5c599f3249e27d1cb489b9bd584411`. The fetch SHALL use a private
external cache. It SHALL stop after 600 seconds or 300 MiB. Each asset SHALL
retain its revision URL, byte size, SHA-256 hash, license, and attribution.
The experiment SHALL not install Enoki or load a neural model.

The corpus SHALL contain at most the 3,990 published rows. Before reading
labels, it SHALL connect rows through normalized article titles, full context
hashes, normalized questions, or duplicate normalized answer text. Official
test groups SHALL remain a sealed final partition. Development groups that
overlap a test group SHALL be excluded. No test row may move to a development
role.

For each answer, a frozen label-blind hash over its answer identity and
sentence index SHALL select at most one usable annotated sentence. The
predictor view SHALL contain only question, context, answer, and selected
sentence text. It SHALL not contain triples, hypotheses, entailment scores,
neutral scores, contradiction scores, hallucination probability, spans,
model identity, source identifiers, or labels. The evaluator view SHALL apply
the published `hall_prob > 0.5` rule only after selection. Empty, unmatched,
or malformed annotations SHALL remain in disposition rows and SHALL be
unscored.

Whole eligible development groups SHALL be assigned by salt
`carnot-v650-source-1` to train, probability calibration, policy calibration,
and online stream with proportions 40, 15, 15, and 30 percent. Membership and
label masks SHALL be sealed before feature fitting. Natural class prevalence
SHALL not trigger a resplit. Inferential support requires at least 20 groups
in each calibration role and 80 online groups. Lower support SHALL remain a
valid corpus contract and SHALL force later inferential claims to a
support-limited null.

The experiment SHALL write a compact manifest and capped text shards below
`results/raw/experiment_7410_v650_source_corpus/`. Every shard SHALL declare
machine-label authority and attribution. It SHALL not store the full release,
teacher fields, or model weights in git. `source_corpus_ready_score` SHALL be
one only when source authentication, grouping, sealed views, deterministic
reload, scoped validation, cold replay, adversarial verification, and strict
row consistency pass. Readiness SHALL remain independent of class balance or
model benefit.

The terminal artifact SHALL use date `20260919`, `MODEL_SPECS=[]`,
`model_invoked=false`, zero current LLM counts,
`inference_substrate=no_model_load`,
`inference_substrate_class=no_model_load`, and `execution_venue=host`.
`promotion_score` SHALL remain zero. The label authority SHALL be
`machine_annotation`. The artifact SHALL preserve all required ordinary
schema fields, exact source hashes, split membership, source disposition rows,
validation receipts, and a checksum that binds code, protocol, source bytes,
and emitted raw rows.

The required ordinary fields SHALL include `run_date`,
`preconditions_checked`, `MODEL_SPECS`, `model_invoked`,
`invocation_counts`, `inference_substrate`, `inference_substrate_details`,
`inference_substrate_class`, `execution_venue`,
`duration_s`, `phase_spans`, `random_seed`, `reproducibility_checksum`,
`source_artifact_hashes`, `rows`, `sample_size_budget`,
`acceptance_gate_results`, `gate_check_summary`, `verifier_is_oracle`,
`honest_verdict`, `verdict_class`, `flagged_adversarial`,
`validation_receipts`, `field_principles`, `promotion_score`,
`source_corpus_ready_score`,
`corpus_manifest_path`, `split_manifest`, `label_authority`, and
`source_disposition_rows`.

#### SCENARIO-AUTO-7410-01: Pinned source assets fail closed

**Given** the exact EnokiQA revision and the three allowed asset paths,
**when** a revision is missing, an asset exceeds a bound, decoding is
unavailable, or network access fails,
**then** Exp7410 emits a terminal blocked result with the exact failed check
**and** it does not publish a ready corpus manifest.

#### SCENARIO-AUTO-7410-02: Connected groups cannot cross the final test boundary

**Given** rows linked by title, context, question, or duplicate answer text,
**when** groups and roles are assigned before labels are read,
**then** every connected row has one group disposition
**and** any development group connected to official test is excluded.

#### SCENARIO-AUTO-7410-03: Predictor readers cannot access teacher fields

**Given** a selected sentence and its separate evaluator annotation,
**when** an ordinary downstream reader loads predictor records,
**then** only question, context, answer, and sentence text are visible
**and** labels, scores, spans, triples, model identity, and source identifiers
remain denied.

#### SCENARIO-AUTO-7410-04: Unsupported annotations stay unscored

**Given** an empty, unmatched, or malformed annotation,
**when** the frozen sentence selector and evaluator run,
**then** the input remains in the source disposition ledger
**and** it does not become either a correct or incorrect scored row.

#### SCENARIO-AUTO-7410-05: Reload preserves the sealed corpus

**Given** authenticated source bytes and completed corpus shards,
**when** a fresh process reloads and independently reduces the manifest,
**then** memberships, masks, counts, hashes, and predictor boundaries match
**and** any changed row, duplicate leak, or test overlap fails readiness.

### REQ-AUTO-7412: Seal source-aware feature and decision protocol

Exp7412 SHALL authenticate the exact Exp7410 artifact and corpus manifest before
dependent work. It SHALL require `source_corpus_ready_score=1`, an allowed
`verdict_class`, `flagged_adversarial=false`, matching manifest hashes, and a
predictor-only reload. A failed gate SHALL emit a terminal blocked artifact that
names the upstream, path, check, field, expected value, and observed value.

The feature reader SHALL use only question, context, answer, selected sentence,
row key, group ID, and partition. It SHALL never request official final-test
labels. It SHALL preserve the response-only PCIB entity-uptake and
falsifiability features as a two-input ablation. It SHALL define these six
source-aware inputs in this order:

1. numeric novelty with actual context;
2. the existing PCIB falsifiability score;
3. normalized number-token overlap;
4. normalized content-token overlap;
5. maximum answer-sentence overlap with one source sentence; and
6. a missing-or-empty-source indicator.

Text SHALL be normalized with Unicode NFKC and lowercase token matching.
Question, answer, and selected sentence inputs SHALL be capped at 1,024 tokens.
Context SHALL be capped at 4,096 tokens. Number tokens SHALL use finite decimal
conversion and exact canonical values. Content tokens SHALL contain Unicode
word tokens with at least two characters and SHALL exclude a frozen stop-word
set. Each overlap SHALL divide the number of unique answer-side matches by the
number of unique answer-side tokens. An empty denominator SHALL return zero.
Numeric novelty SHALL be the unmatched unique answer-number fraction. It SHALL
return zero when the answer has no numbers. Sentence overlap SHALL return the
largest answer-side content-token overlap across bounded source sentences.
Missing or whitespace-only context SHALL set the final feature to one and all
source overlaps to zero. Every feature SHALL be finite and bounded to `[0, 1]`.
These values are interpretable proxies. They are not an EnokiQA or PCIB paper
replication. A lexical match or a valid span SHALL not claim entailment.

The experiment SHALL provide a reusable experiment-local Gibbs head with six
inputs, one four-unit hidden layer, and scalar energy. It SHALL use
`p_incorrect=sigmoid(energy)`, natural-prevalence Bernoulli loss, learning rate
`0.01`, L2 coefficient `0.001`, 500 optimizer steps, and seeds 65001 through
65005. Checkpoints SHALL contain finite plain numeric arrays with exact shapes.
Reload SHALL reject extra keys, non-finite values, wrong shapes, booleans, and
executable values. The typed policy SHALL emit accept when probability is at or
below its accept threshold, reject when it is at or above its reject threshold,
and escalate otherwise. Invalid probabilities SHALL escalate without invented
confidence.

The frozen arms SHALL be training prevalence, L2 logistic on the six inputs,
response-only 2-4-1 Gibbs, and source-aware 6-4-1 Gibbs. Accept thresholds SHALL
be `0.01`, `0.025`, and `0.05`. Reject thresholds SHALL be `0.90`, `0.95`, and
`0.99`. Later fitting SHALL use training rows only. Probability calibration
SHALL use probability-calibration groups only. Policy selection SHALL use
policy-calibration groups only. Each risk calculation SHALL use one
label-blind representative per independent group. One-sided 95-percent exact
binomial bounds SHALL use Bonferroni correction across four arms, five seeds,
nine threshold pairs, and two actions. An action with no selected group SHALL
have no certificate and SHALL be disabled. Incorrect accepts SHALL have a
machine-annotation risk budget of 0.05. Correct rejects SHALL have a
machine-annotation risk budget of 0.10.

Exp7412 SHALL seal eight nontraining semantic minimal pairs for negation,
subject/object reversal, comparator reversal, time qualifier, unit mismatch,
count mismatch, omitted condition, and coreference ambiguity. Each pair SHALL
declare a small source relation, exact answer spans, and deterministic expected
scope in an evaluator-only view. It SHALL also seal eight equivalent
paraphrase or format controls. Prediction readers SHALL not expose expected
verdicts or expected scope. The fixtures SHALL remain future inputs for Exp7416
and Exp7417. They SHALL not provide external accuracy evidence.

This experiment SHALL train only on test fixtures. It SHALL not fit real corpus
labels or open official test labels. It SHALL emit unstarted rows for every
frozen real arm, seed, and threshold condition. The protocol manifest SHALL
freeze architecture, seeds, roles, losses, costs, thresholds, intervals,
machine-annotation budgets, and task-specific acceptance rules before later
real fitting. `source_feature_protocol_ready_score` SHALL equal one only when
the fixture training, checkpoint reload, predictor reader, challenge masking,
and required validation checks pass. Readiness SHALL remain independent of
scientific benefit. `promotion_score` SHALL remain zero.

The current run SHALL declare `MODEL_SPECS=[]`, `model_invoked=false`, zero
current LLM invocation counts, `inference_substrate=no_model_load`,
`inference_substrate_class=no_model_load`, and `execution_venue=host`. Fixture
Gibbs work SHALL appear only in `small_ebm_training`. The workflow SHALL use the
Exp7358 scoped command plan and the Exp7303 runner. It SHALL run the declared
entrypoint, fresh-process cold replay, independent row recomputation,
adversarial verification, and strict verdict-row consistency. No numbered E2E
scenario applies because shared training, sampling, serialization, and PyO3
behavior remain unchanged.

The required ordinary fields SHALL include `run_date`,
`preconditions_checked`, `MODEL_SPECS`, `model_invoked`,
`invocation_counts`, `inference_substrate`, `inference_substrate_details`,
`inference_substrate_class`, `execution_venue`, `duration_s`, `phase_spans`,
`random_seed`, `reproducibility_checksum`, `source_artifact_hashes`, `rows`,
`sample_size_budget`, `acceptance_gate_results`, `gate_check_summary`,
`verifier_is_oracle`, `honest_verdict`, `verdict_class`,
`flagged_adversarial`, `validation_receipts`, `field_principles`,
`promotion_score`, `source_feature_protocol_ready_score`,
`feature_definitions`, `protocol_manifest_path`, `challenge_manifest_path`,
and `annotation_risk_scope`.

#### SCENARIO-AUTO-7412-01: Source features stay bounded and predictor-only

**Given** numeric text, empty context, long text, and Unicode-equivalent text,
**when** the six fixed source features are computed,
**then** every value is finite and in `[0, 1]`
**and** the computation uses no label or evaluator field.

#### SCENARIO-AUTO-7412-02: Checkpoints reject unsafe numeric state

**Given** a valid 6-4-1 checkpoint and mutations with non-finite values, wrong
shapes, extra keys, booleans, or executable values,
**when** the experiment reloads the checkpoint,
**then** valid numeric state reproduces identical probabilities
**and** every unsafe mutation fails before scoring.

#### SCENARIO-AUTO-7412-03: Fixture fitting survives constant and imbalanced data

**Given** constant feature columns and a highly imbalanced two-class fixture,
**when** the frozen Bernoulli Gibbs and logistic harnesses fit,
**then** each completed update and loss remains finite
**and** one-class fitting input fails closed.

#### SCENARIO-AUTO-7412-04: Policy certificates use independent groups

**Given** multiple rows from one group and all nine frozen threshold pairs,
**when** risk policies are selected,
**then** one label-blind representative per group enters each exact bound
**and** an empty or uncertified action is disabled.

#### SCENARIO-AUTO-7412-05: Challenge semantics stay outside prediction views

**Given** eight semantic pairs and eight equivalent controls,
**when** the challenge manifest is sealed and reloaded,
**then** exact spans, relation scope, and fixture identities reproduce
**and** predictor records contain no expected verdict or expected scope.

#### SCENARIO-AUTO-7412-06: Official labels remain unopened

**Given** the authenticated Exp7410 predictor reader and a fixture-only trainer,
**when** Exp7412 builds its protocol artifact,
**then** official final-test labels are never requested
**and** every real fitting row remains explicitly unstarted for Exp7413.

#### SCENARIO-AUTO-7412-07: Completion and efficacy remain separate

**Given** complete feature, fixture, reader, manifest, and validation checks,
**when** no real source-aware efficacy measurement has run,
**then** protocol readiness equals one while the verdict is complete null
**and** no lexical feature, span, or constructed fixture is reported as
entailment or external accuracy evidence.

### REQ-AUTO-7413: Measure source-grounded calibration against matched controls

Exp7413 SHALL authenticate the exact Exp7410 corpus manifest and Exp7412
artifact, protocol manifest, and feature rows before dependent work. It SHALL
require `source_feature_protocol_ready_score=1`, an allowed `verdict_class`,
and `flagged_adversarial=false`. A failed prerequisite SHALL produce a terminal
blocked artifact naming the exact upstream, path, check, field, expected value,
and observed value.

The experiment SHALL train exactly four registered arms for seeds 65001 through
65005: training prevalence, L2 logistic on the six fixed source features,
response-only 2-4-1 Gibbs, and source-aware 6-4-1 Gibbs. Weight fitting SHALL
use scored training groups only. A two-parameter affine probability transform
SHALL use scored probability-calibration groups only. Policy thresholds SHALL
be selected on scored policy-calibration groups only. Feature vectors SHALL not
contain a source ID, article ID, teacher score, gold triple, label, or evaluator
field. Numeric checkpoints SHALL record architecture, optimizer work, byte
size, score direction, partition identity, and input hashes. Small-head fitting
SHALL be reported as `small_ebm_training`, not as a current LLM invocation.

Every eligible official-test row SHALL be scored once by every arm and seed.
Official rows without a machine label SHALL remain explicit unscored rows for
every arm and seed. The paired metric ledger SHALL retain row key, connected
group, arm, seed, probability, machine label, typed action, Brier and log-loss
contributions, and measured service cost. It SHALL report Brier score, log
loss, AUROC, incorrect-class PR-AUC, correct-class PR-AUC, macro-F1, typed-action
risk and coverage, and decision changes. Seeds SHALL be averaged within each
row before resampling article-connected groups. The paired bootstrap SHALL use
10,000 draws and seed 6501307. The upper Brier interval SHALL be simultaneous
across the three source-Gibbs contrasts against prevalence, logistic, and
response-only Gibbs by taking the per-draw maximum centered contrast.

Policy certification SHALL use one label-blind row-key representative per
connected policy group and the nine thresholds frozen by Exp7412. An empty or
uncertified action SHALL be disabled. Scientific value SHALL require all three
source-Gibbs Brier-delta upper simultaneous 95-percent bounds below zero,
source-Gibbs mean log loss no worse than each control, certified official-test
coverage of at least 0.25, and no source-Gibbs coverage loss versus logistic
under their selected policies and the same annotation-risk budgets. Too little
support SHALL disable the action and produce a valid null. Ranking improvement
without certified decision coverage SHALL remain diagnostic only.

The experiment SHALL also run two frozen source ablations. Source removal SHALL
replace the four overlap or novelty source values with zero, set the
missing-source indicator to one, and retain the row's response-only
falsifiability value.
Cross-group source permutation SHALL replace each row's context with the next
lexicographically ordered group's representative context inside the same
partition, wrapping once, before recomputing all six source features. The
source-aware Gibbs arm SHALL be retrained on training data for both conditions
with the same five seeds and unchanged 500-step fitting and calibration budgets.
Neither ablation SHALL tune on official-test labels. Ablation rows SHALL report
proper scores and whether an observed full-condition benefit depends on source
information; a failed ablation SHALL stay recorded without adding a feature.

`calibration_capture_complete_score` SHALL equal one when preconditions,
registered fitting, complete paired scoring including unscored preservation,
ablations, affected validation, cold replay, independent metric recomputation,
adversarial verification, and strict row consistency all pass. A complete null
MAY receive this score. `calibration_value_score` SHALL equal one only when the
registered scientific conjunction passes. `promotion_score` SHALL remain zero.
`label_authority` SHALL be `machine_annotation`, and `verifier_is_oracle` SHALL
be false because agreement with held-out teacher labels is not general semantic
correctness.

The current run SHALL declare `MODEL_SPECS=[]`, `model_invoked=false`, zero
current LLM invocation counts, `inference_substrate=no_model_load`,
`inference_substrate_class=no_model_load`, and `execution_venue=host`. The
workflow SHALL freeze the Exp7358 affected manifest before using the Exp7303
runner. It SHALL run worktree imports, affected pytest without distributed or
coverage add-ons, isolated changed-module coverage at 100 percent, scoped Ruff
check and format, changed-module mypy, exact-test spec coverage, the declared
entrypoint, fresh-process cold replay, independent reduction,
`scripts/adversarial_verify.py`, and strict verdict-row consistency. No numbered
E2E scenario applies because shared training, sampling, serialization, and PyO3
behavior do not change.

The required ordinary fields SHALL include `schema`, `experiment_id`,
`milestone`, `status`, `run_date`, `preconditions_checked`, `MODEL_SPECS`,
`model_invoked`, `invocation_counts`, `inference_substrate`,
`inference_substrate_details`, `inference_substrate_class`, `execution_venue`,
`duration_s`, `phase_spans`, `random_seed`, `reproducibility_checksum`,
`source_artifact_hashes`, `rows`, `sample_size_budget`,
`acceptance_gate_results`, `gate_check_summary`, `verifier_is_oracle`,
`honest_verdict`, `verdict_class`, `flagged_adversarial`,
`validation_receipts`, `field_principles`, `promotion_score`,
`calibration_capture_complete_score`, `calibration_value_score`,
`checkpoint_manifest`, `paired_metric_rows`, `source_ablation_rows`, and
`label_authority`.

#### SCENARIO-AUTO-7413-01: Partitions remain disjoint and features stay label-blind

**Given** the authenticated Exp7410 labels and Exp7412 feature rows,
**when** the four arms fit, calibrate, and select policies,
**then** each operation uses only its registered partition
**and** no prohibited identity or evaluator value enters a feature vector.

#### SCENARIO-AUTO-7413-02: Paired scoring preserves every official unit

**Given** scored and unscored official-test rows,
**when** all arm-seed units are evaluated,
**then** every official row appears once for every arm and seed
**and** unscored rows remain present without invented labels or metrics.

#### SCENARIO-AUTO-7413-03: Metrics and simultaneous intervals are reproducible

**Given** the paired scored ledger,
**when** metrics and 10,000 connected-group bootstrap draws are reduced,
**then** both class PR-AUCs, proper scores, macro-F1, actions, and decision
changes reproduce
**and** the three primary Brier contrasts use one simultaneous upper bound.

#### SCENARIO-AUTO-7413-04: Unsupported actions fail to safe escalation

**Given** the frozen thresholds and annotation-risk budgets,
**when** policy-calibration support cannot certify accept or reject,
**then** that action is disabled and official decisions escalate
**and** ranking-only evidence cannot pass the scientific value gate.

#### SCENARIO-AUTO-7413-05: Source ablations retrain without test tuning

**Given** the full source condition, fixed source removal, and cross-group
source permutation,
**when** source-aware Gibbs is retrained under matched seeds and budgets,
**then** all three conditions retain their own checkpoints and scored rows
**and** no official-test label changes a fit, affine transform, or threshold.

#### SCENARIO-AUTO-7413-06: Completion stays independent of scientific benefit

**Given** complete fitting, scoring, ablation, and validation evidence,
**when** one or more registered benefit gates fail,
**then** calibration capture can complete with a valid null
**and** calibration value remains zero with promotion disabled.

#### SCENARIO-AUTO-7413-07: Cold readers reject material evidence drift

**Given** a terminal candidate and its hash-bound inputs and checkpoints,
**when** a fresh process reloads and independently reduces it,
**then** changed rows, metrics, gates, source hashes, counters, or checksums fail
**and** the final JSON is published atomically only after all readers pass.

### REQ-AUTO-7414: Measure fixed selected-feedback source adaptation

Exp7414 SHALL authenticate the exact Exp7412 artifact, protocol manifest, and
source-feature rows before dependent work. It SHALL require
`source_feature_protocol_ready_score=1`, an allowed `verdict_class`, and
`flagged_adversarial=false`. A failed prerequisite SHALL produce a terminal
blocked artifact that names the upstream, path, check, field, expected value,
and observed value.

The experiment SHALL use only Exp7410 development train, probability-calibration,
policy-calibration, and online-stream groups through the predictor and evaluator
readers sealed by Exp7410 and the feature protocol sealed by Exp7412. It SHALL
not read official final-test labels. Initial source-aware 6-4-1 Gibbs weights and
the initial L2 logistic control SHALL fit on scored train rows only. Initial
affine probability calibration SHALL use scored probability-calibration rows
only. The frozen baseline policy SHALL use scored policy-calibration groups only.
No checkpoint that saw an online-stream label may initialize the replay.

The five compared arms SHALL be frozen calibrated Gibbs, projected adaptive
affine Gibbs, online L2 logistic, recent-frequency Beta(1,1) with window 128,
and a no-feedback copy of the adaptive arm. The adaptive Gibbs arm SHALL keep
all Gibbs weights fixed. For each newly revealed label, it SHALL update only
`a,b` in `p=sigmoid(a*E+b)` once, use learning rate `0.01`, clip the joint
gradient norm to `1`, and project `a` to `[0.25,4]` and `b` to `[-8,8]`.
Every prediction and its durable state hash SHALL be recorded before feedback.
Duplicate or premature reveals SHALL not update state.

The replay SHALL seal one label-blind representative per independent online
group in hash order and one reversed-block sensitivity order. These are
constructed archive replays, not real chronology. It SHALL evaluate delays 1
and 32. The primary feedback regime SHALL use a label-blind deterministic
75-percent availability mask shared by every learner. The selected-feedback
regime SHALL reveal only events escalated by the frozen baseline policy, with
the same frozen mask shared by every learner. Learner-specific escalation masks
MAY be reported as diagnostics, but SHALL not drive a paired causal claim.

Inferential claims SHALL require at least 80 independent online groups and at
least ten observations of each binary label. Lower support SHALL produce
`complete_null_insufficient_online_support` with exact counts. Paired moving-
block intervals SHALL use 10,000 draws, primary block length 16, sensitivity
length 32, and seed 6501407. Seeds SHALL be averaged within each event before
resampling. Registered value SHALL require a simultaneous Brier-delta upper
bound below zero against frozen Gibbs, online logistic, and recent frequency on
the primary delay-1 hash-order stream. It SHALL also require no worse log loss,
no lower typed-action coverage, and no higher observed action risk against each
control. Each feedback regime SHALL be reported separately. The experiment
SHALL assert neither IID sampling nor a conformal guarantee.

Independent fixtures SHALL test label replacement, label erasure, duplicate
reveal, premature reveal, crash/restart, no-feedback equality, and erased-update
replay. A revoked event SHALL deterministically reconstruct state from the
trusted journal before the next prediction. Reconstruction and persistence
costs SHALL be included in measured service costs. Prediction, reveal, and
commit rows SHALL retain observation identity, availability index, fixed and
diagnostic masks, prior and new state hashes, and label authority.

`online_capture_complete_score` SHALL equal one when the registered replay,
controls, persistence, affected validation, cold replay, independent reduction,
adversarial verification, and strict row consistency pass. A support-limited
null MAY receive this score. `online_value_score` SHALL equal one only when the
registered equal-information causal conjunction passes. `promotion_score`
SHALL remain zero. `label_authority` SHALL be `machine_annotation`, and
`verifier_is_oracle` SHALL be false. The exact-learning branch in Exp7418 SHALL
remain independent of these machine labels.

The current run SHALL declare `MODEL_SPECS=[]`, `model_invoked=false`, zero
current LLM invocation counts, `inference_substrate=no_model_load`,
`inference_substrate_class=no_model_load`, and `execution_venue=host`. Small
numeric-head work SHALL appear only in `small_ebm_training`. The workflow SHALL
freeze the Exp7358 affected manifest before using the Exp7303 runner. It SHALL
run scoped imports, tests, changed-module coverage at 100 percent, Ruff, mypy,
exact-test spec coverage, the declared entrypoint, fresh-process cold replay,
independent row reduction, adversarial verification, and strict verdict-row
consistency. No numbered E2E scenario applies because shared training,
sampling, serialization, and PyO3 behavior do not change.

The required ordinary fields SHALL include `schema`, `experiment_id`,
`milestone`, `status`, `run_date`, `preconditions_checked`, `MODEL_SPECS`,
`model_invoked`, `invocation_counts`, `inference_substrate`,
`inference_substrate_details`, `inference_substrate_class`, `execution_venue`,
`duration_s`, `phase_spans`, `random_seed`, `reproducibility_checksum`,
`source_artifact_hashes`, `rows`, `sample_size_budget`,
`acceptance_gate_results`, `gate_check_summary`, `verifier_is_oracle`,
`honest_verdict`, `verdict_class`, `flagged_adversarial`,
`validation_receipts`, `field_principles`, `promotion_score`,
`continuous_self_learning_task`, `online_capture_complete_score`,
`online_value_score`, `feedback_event_rows`, `revocation_rows`,
`condition_reports`, `paired_moving_block_intervals`, `hardware_path`, and
`label_authority`.

#### SCENARIO-AUTO-7414-01: Inputs stop at the sealed online boundary

**Given** authenticated Exp7410 development roles and the Exp7412 protocol,
**when** initial heads, calibration, policy, and online replay data load,
**then** each operation reads only its registered role
**and** no official final-test label or online label enters initial state.

#### SCENARIO-AUTO-7414-02: Prediction precedes one bounded update

**Given** a predicted event with a future availability index,
**when** its machine label becomes visible,
**then** the prediction state is durable before the reveal and one projected
affine update is committed
**and** early or repeated reveals do not change state.

#### SCENARIO-AUTO-7414-03: Fixed feedback masks preserve equal information

**Given** primary and frozen-baseline-selected feedback regimes,
**when** all five arms replay the two delays and two constructed orders,
**then** each compared learner receives the same registered event mask
**and** learner-specific selection remains diagnostic only.

#### SCENARIO-AUTO-7414-04: Support and value claims fail closed

**Given** the completed primary delay-1 stream and paired moving-block draws,
**when** independent groups or either label count is below its registered floor,
**then** online capture can complete but online value remains zero
**and** the verdict is `complete_null_insufficient_online_support`.

#### SCENARIO-AUTO-7414-05: Revocation reconstructs trusted state

**Given** committed feedback and its trusted prediction journal,
**when** an authority replaces or erases an earlier label,
**then** the adapter replays active commits in order before later prediction
**and** restart, replacement, and erasure reproduce deterministic state hashes.

#### SCENARIO-AUTO-7414-06: No-feedback stays equal to frozen Gibbs

**Given** equal initial Gibbs weights and affine state,
**when** other learners receive delayed feedback,
**then** the no-feedback copy never admits an update
**and** its predictions and state hashes remain equal to frozen Gibbs.

#### SCENARIO-AUTO-7414-07: Cold readers reject material replay drift

**Given** a terminal candidate with raw event, journal, interval, and control
rows,
**when** a fresh process independently reduces the artifact,
**then** changed masks, metrics, state lineage, counters, hashes, or gates fail
**and** terminal JSON is published atomically only after all readers pass.

#### SCENARIO-AUTO-7414-08: Service costs include durable online work

**Given** prediction, feedback, restart, and revocation operations,
**when** the replay reports its CPU service path,
**then** it measures prediction, persistence, update, and reconstruction costs
**and** it reports p50 and p95 latency without claiming a hardware speedup.

### REQ-AUTO-7423: Seal a human-annotated source-support protocol

Exp7423 SHALL resolve the public ParticleMedia/RAGTruth repository to commit
`c103204b9ce28d6bbad859304bf30de72b8ed8fe`. It SHALL authenticate only
`dataset/source_info.jsonl` and `dataset/response.jsonl` in an external cache.
Each asset SHALL retain its immutable URL, byte size, SHA-256 hash, license,
and author attribution. Network and cache failures SHALL produce a terminal
blocked artifact. Synthetic rows, Enoki labels, and historical graph-grounding
outputs SHALL not replace unavailable RAGTruth data.

The experiment SHALL join the two files by `source_id`. It SHALL preserve QA,
Summary, and Data2txt records. An evaluator-only record SHALL retain quality,
span offsets, `label_type`, and `implicit_true`. The primary label SHALL be one
when the response contains no annotated source-unsupported span. A span with
`implicit_true=true` SHALL remain unsupported by the supplied source. It SHALL
not be called false in the world. A frozen sensitivity view SHALL exclude such
spans without tuning on that view. The documented exclusion rule SHALL reject
non-good quality, truncated, and refusal records. It SHALL count every reason.

Before labels are read, the experiment SHALL connect identical source IDs and
duplicate normalized source bytes. All sibling responses SHALL remain in one
group. Official test groups SHALL remain final-test only. A development group
that overlaps official test SHALL be excluded. Remaining official training
groups SHALL use salt `carnot-v651-human-1` and fixed hash ranges of 40, 15,
15, and 30 percent for fit, probability calibration, policy calibration, and
prospective stream. Labels SHALL not affect assignment, caps, or representative
selection.

The label-blind caps SHALL be 2,000 fit groups, 500 groups in each calibration
partition, 1,000 stream groups, and 1,000 final-test groups. One hash-ranked
response per group SHALL supply certificates and the stream. All siblings
SHALL remain available for secondary grouped metrics. Inferential online value
SHALL require 400 stream groups and 40 examples of each class. Each calibration
partition SHALL require 100 groups and 20 examples of each class. Lower support
SHALL yield a complete null with protocol readiness one when every contract
check passes.

Predictor records SHALL contain only a row key, group ID, partition, task type,
label-blind source serialization, and bounded response text. Source text SHALL
be capped at 4,096 lexical tokens. Response text SHALL be capped at 1,024
lexical tokens. Predictor readers SHALL deny annotations, labels, model identity,
quality, original metadata, source IDs, and split outcome statistics. The six
features SHALL reuse the exact Exp7412 definitions and order. They are lexical
and PCIB proxies. They do not prove entailment. Leakage mutations, sibling
group checks, and a fresh reload SHALL pass before any later fitting.

The experiment SHALL write hash-bound predictor and evaluator shards plus a
frozen manifest under
`results/raw/experiment_7423_v651_annotated_protocol/`. It SHALL seal seeds
65101 through 65105, a 500-step small-energy-head budget, disjoint fitting and
calibration roles, the Exp7412 threshold grid, and the fixed online schedules.
No current LLM SHALL run. Current declarations SHALL be `MODEL_SPECS=[]`,
`model_invoked=false`, zero invocation counts,
`inference_substrate=no_model_load`,
`inference_substrate_class=no_model_load`, and `execution_venue=host`.
Small numeric-head work SHALL use `small_ebm_training` receipts. Archived or
scripted model events SHALL remain hash-bound sidecars and never current calls.

`annotated_protocol_ready_score` SHALL equal one when source authentication,
human-label semantics, grouping, masking, sealed partitions, cold reload,
affected validation, entrypoint replay, independent reduction, adversarial
verification, and strict row consistency pass. Readiness SHALL not depend on
inferential support. `promotion_score` SHALL remain zero. `label_authority`
SHALL be `human_annotation_source_support`, explicitly fallible and not formal
truth. `verifier_is_oracle` SHALL be false.

The workflow SHALL freeze an Exp7358 affected-file manifest. It SHALL use the
Exp7303 runner with command-local coverage data. It SHALL run worktree imports,
affected tests, 100-percent changed-module coverage, Ruff, mypy, and exact-test
spec coverage. The declared entrypoint and fresh-process cold replay form the
capability E2E. No numbered E2E scenario applies because shared training,
sampling, serialization, PyO3, and ARC code do not change.

The terminal artifact SHALL contain `schema`, `experiment_id`, `milestone`,
`status`, `run_date`, `preconditions_checked`, `MODEL_SPECS`, `model_invoked`,
`invocation_counts`, `inference_substrate`, `inference_substrate_details`,
`inference_substrate_class`, `execution_venue`, `duration_s`, `phase_spans`,
`random_seed`, `reproducibility_checksum`, `source_artifact_hashes`, `rows`,
`sample_size_budget`, `acceptance_gate_results`, `gate_check_summary`,
`verifier_is_oracle`, `honest_verdict`, `verdict_class`,
`flagged_adversarial`, `validation_receipts`, `field_principles`,
`promotion_score`, `corpus_manifest`, `support_counts`, `label_authority`,
`feature_contract`, and `annotated_protocol_ready_score`. It SHALL retain the
exact candidate validation receipts. A complete finding SHALL use a
`complete_` or `complete:` verdict. An unavailable source SHALL use
`blocked_*`. `verdict_class` SHALL be one of `positive`, `circular_positive`,
`null`, `blocked`, `disqualified`, or `partial`. The reproducibility checksum
SHALL bind code, protocol, inputs, raw rows, and validation scope.

#### SCENARIO-AUTO-7423-01: Human source-support labels stay evaluator-only

**Given** joined RAGTruth source and response records,
**when** predictor and evaluator shards are sealed and reloaded,
**then** predictor rows expose no label, span, quality, model, metadata, or split
statistics
**and** evaluator rows retain exact spans and primary plus sensitivity labels.

#### SCENARIO-AUTO-7423-02: Source duplicates cannot cross partitions

**Given** sibling responses, repeated source IDs, and normalized duplicate
source bytes,
**when** official boundaries, hashed partitions, and label-blind caps apply,
**then** each connected group has one disposition
**and** cross-split components leave development instead of contaminating it.

#### SCENARIO-AUTO-7423-03: Fixed features have bounded source inputs

**Given** each supported task type and over-limit source or response text,
**when** its predictor serialization and six features are computed,
**then** source and response truncation are recorded
**and** the feature names, order, bounds, and values match Exp7412.

#### SCENARIO-AUTO-7423-04: Support changes value, not protocol readiness

**Given** a valid sealed corpus with too few stream or calibration labels,
**when** support gates are reduced,
**then** the terminal verdict is a complete null and benefit is forbidden
**and** annotated protocol readiness remains one.

#### SCENARIO-AUTO-7423-05: Cold readers reject evidence drift

**Given** a terminal candidate and its raw predictor, evaluator, and manifest
shards,
**when** a fresh process reloads and independently reduces them,
**then** any hash, membership, mask, label, feature, receipt, or checksum drift
fails
**and** terminal JSON publishes atomically only after all required checks pass.

### REQ-AUTO-7426: Measure static human source-support decisions

Exp7426 SHALL authenticate the exact Exp7423 annotated protocol and Exp7425
spline prototype before dependent work. It SHALL require both readiness scores
to equal one. It SHALL require each upstream verdict class to be `positive`,
`circular_positive`, or `null`. Both upstream artifacts SHALL be unflagged.
A failed prerequisite SHALL produce a terminal blocked artifact. The artifact
SHALL name the upstream, path, check, field, expected value, and observed value.

The primary label SHALL be the Exp7423 human source-support label. Label one
means that no human-annotated source-unsupported span is present. The authority
SHALL be `human_annotation_source_support`. It is a fallible support judgment.
It is not general semantic truth and is not a replacement-grade verifier.

The experiment SHALL fit five arms with seeds 65101 through 65105 and at most
500 updates. The arms SHALL be training prevalence, L2 logistic regression on
the six raw features, the existing 6-4-1 Gibbs head, the sparse 49-parameter
cubic-spline head, and dense logistic regression on the identical spline basis.
All learned heads SHALL use natural-prevalence Bernoulli loss. Spline knots
SHALL use training features only. Sparse and dense spline probabilities and
parameters SHALL match within `1e-10`. Spline parity is a validity gate. It is
not a scientific contrast.

Each arm SHALL average its five seed probabilities before affine logit
calibration. Affine calibration SHALL use only the sealed probability
calibration partition. Policy selection SHALL use only the separate policy
partition. It SHALL use one label-blind representative per source group. The
nine registered threshold pairs SHALL combine accept probabilities `0.95`,
`0.975`, and `0.99` with reject probabilities `0.01`, `0.05`, and `0.10`.
Accept harm is label zero and has a 0.05 budget. Reject harm is label one and
has a 0.10 budget. Certificates SHALL use one-sided Clopper-Pearson bounds with
Bonferroni correction over five arms, nine pairs, and two actions. No selected
group means no certificate. An uncertified action SHALL escalate.

The experiment SHALL freeze all fitted state, affine transforms, and policies
before it opens official final-test labels. It SHALL evaluate final test once.
It SHALL retain every arm, seed, condition, row key, source group, task domain,
raw probability, seed-averaged calibrated probability, label, action, Brier
contribution, log-loss contribution, cost, and authority. Hash-bound shards MAY
hold the detailed ledger when the terminal artifact would exceed 20 MiB.
Malformed, unsupported, and excluded units SHALL remain visible with null
predictions and their exact disposition. Domain support and empirical risk
SHALL be reported separately.

The registered conditions SHALL be full source, source masked, and label-blind
source swapped. Masking and swapping SHALL happen before evaluator labels join.
All arms in one condition SHALL receive identical six-feature information.
Lexical agreement SHALL not be called entailment. Source ablations are
diagnostics and SHALL not supply another chance to pass the benefit gate.

The predictive-value gate SHALL compare the sparse spline arm with raw-feature
logistic and the 6-4-1 Gibbs head. Seeds SHALL be averaged before connected
source groups enter a 10,000-draw paired bootstrap. The bootstrap SHALL use
simultaneous upper Brier bounds corrected across the two primary contrasts.
Both upper bounds SHALL be below zero. Spline log loss SHALL be no worse than
both controls. Certified coverage SHALL be at least 0.25 and no lower than both
controls. Empirical accept and reject risk SHALL not increase. All clauses
SHALL pass for `decision_value_score=1`.

The experiment SHALL record numeric initial checkpoints, final checkpoints,
loss traces, update counts, parameter counts, partition hashes, probability
maps, affine state, and policy thresholds. These experiment-local weights
SHALL not change a generator or production default. `promotion_score` SHALL
remain zero. Valid complete paired measurements SHALL set
`decision_capture_complete_score=1`, including a complete null. Basis parity,
partition leakage, missing required evidence, or affected validation failure
SHALL disqualify the result.

The current run SHALL declare `MODEL_SPECS=[]`, `model_invoked=false`, zero
current LLM counts, `inference_substrate=no_model_load`,
`inference_substrate_class=no_model_load`, and `execution_venue=host`. Small
head fitting SHALL appear only in `small_ebm_training`. Historical or scripted
model events SHALL remain typed hash-bound sidecars and SHALL not enter current
invocation counts.

The workflow SHALL freeze an Exp7358 affected-file manifest. It SHALL use the
Exp7303 runner with command-local coverage data. It SHALL run worktree imports,
affected tests, 100-percent changed-module coverage, scoped Ruff, changed-module
mypy, exact-test spec coverage, the declared entrypoint, fresh-process replay,
independent reduction, adversarial verification, and strict row consistency.
No numbered E2E scenario applies because this experiment changes no shared
training, sampling, serialization, PyO3, or ARC behavior.

The terminal artifact SHALL contain the ordinary required experiment fields.
It SHALL also contain `decision_capture_complete_score`,
`decision_value_score`, `checkpoint_manifest`, `policy_certificates`,
`grouped_intervals`, and `small_ebm_training`. Its checksum SHALL bind code,
protocol, input bytes, detailed row shards, checkpoints, and validation scope.

#### SCENARIO-AUTO-7426-01: Prerequisites fail closed

**Given** the two exact upstream artifacts and their sealed manifests,
**when** any readiness, verdict, adversarial flag, byte hash, or identity differs,
**then** dependent fitting does not start
**and** the terminal block names the exact failed upstream field.

#### SCENARIO-AUTO-7426-02: Label authority changes decision direction

**Given** a probability of human source support,
**when** a registered policy emits a typed action,
**then** high probability can accept and low probability can reject
**and** unsupported or uncertified actions escalate.

#### SCENARIO-AUTO-7426-03: Seeds precede calibration and policy fitting

**Given** five fitted seed states for one arm,
**when** probability calibration and policy selection run,
**then** raw seed probabilities average before the affine logit transform
**and** only the two sealed calibration roles determine later state.

#### SCENARIO-AUTO-7426-04: Sparse and dense spline heads stay equivalent

**Given** identical training-only knots and initial numeric state,
**when** both spline representations receive the same Bernoulli updates,
**then** final parameters and every probability agree within `1e-10`
**and** a mismatch disqualifies rather than becoming a claimed benefit.

#### SCENARIO-AUTO-7426-05: Policy certificates use independent groups

**Given** repeated responses from one source group and all nine threshold pairs,
**when** policy risk is certified,
**then** one label-blind representative enters each exact bound
**and** empty or uncertified actions become escalation.

#### SCENARIO-AUTO-7426-06: Official scoring preserves every disposition

**Given** eligible final-test rows and excluded corpus dispositions,
**when** frozen heads score the official test once,
**then** every eligible arm-seed row retains its probability and proper scores
**and** each excluded unit remains visible with no invented probability.

#### SCENARIO-AUTO-7426-07: Source controls stay label blind

**Given** predictor-only source and response text,
**when** source masking or within-role source swapping runs,
**then** features change before labels join and all arms see the same inputs
**and** the diagnostic controls cannot change the primary pass rule.

#### SCENARIO-AUTO-7426-08: Grouped value reduction is reproducible

**Given** the full-source paired ledger,
**when** 10,000 source-group bootstrap draws are reduced,
**then** the two primary Brier contrasts use simultaneous corrected bounds
**and** log loss, coverage, and empirical risk clauses reproduce exactly.

#### SCENARIO-AUTO-7426-09: Completion remains independent of benefit

**Given** valid complete paired measurement with inadequate decision support,
**when** one or more predictive or policy gates fail,
**then** the result is a complete null with capture score one
**and** value and promotion remain zero.

#### SCENARIO-AUTO-7426-10: Cold readers bind detailed evidence

**Given** a terminal candidate and its row and checkpoint shards,
**when** a fresh process replays and independently reduces the result,
**then** changed probabilities, metrics, hashes, counters, or gates fail
**and** terminal JSON publishes atomically only after all required readers pass.

### REQ-AUTO-7436: Select and certify typed actions with isolated roles

Exp7436 SHALL authenticate the exact Exp7423 corpus, Exp7426 static-decision
artifact, and Exp7428 decision audit before dependent work. It SHALL preserve
each producer's original adversarial flag and byte identity. Missing or changed
external evidence SHALL produce `blocked_*` with an exact gate summary. The run
SHALL not invent an unavailable branch.

The protocol SHALL use only fit, probability-calibration, and
policy-calibration rows. It SHALL not read final-test or prospective-stream
labels, quantiles, or metrics for method design. It SHALL hash original role
membership and source-group identities. Prior exposure of the reused RAGTruth
evaluation corpus SHALL remain disclosed.

The three deployed heads SHALL be Gibbs 6-4-1, sparse spline 49, and raw L2
logistic from the Exp7426 full-source checkpoints. Each head SHALL average the
same five fixed training seeds before scalar calibration or policy fitting.
Dense spline logistic SHALL remain an equivalence control and SHALL not become
a fourth deployed policy.

Source groups in probability calibration SHALL split deterministically into
disjoint calibration and tuning halves without labels. Each half SHALL use one
label-blind representative per source group. Scalar calibration SHALL use only
the calibration half. Policy tuning SHALL use only the tuning half. Candidate
thresholds SHALL use tuning-score deciles plus an explicit no-action sentinel
for each action. Exactly one accept/reject pair SHALL be selected per deployed
head by utility, then a fixed deterministic tie rule. Utility SHALL be +1 for a
correct action, -20 for a harmful accept, -10 for a harmful reject, and 0 for
escalation.

Policy calibration SHALL remain certification-only. Familywise delta 0.05
SHALL divide equally across nine planned checks: accept-harm upper bound,
reject-harm upper bound, and non-escalation coverage lower bound for each of
three policies. Exact one-sided Clopper-Pearson bounds SHALL be used. Accept
risk SHALL be at most 0.05. Reject risk SHALL be at most 0.10. Certified
coverage SHALL be at least 0.25. An action disabled during tuning SHALL have an
undefined risk and a not-applicable check; its alpha remains unspent. Every
enabled action check and the coverage check SHALL pass together. Otherwise the
whole policy SHALL become all-escalate without threshold reselection.

Every arm and action SHALL report score quantiles, selected source-group count,
harmful count, exact upper bound, and a diagnosis that distinguishes empty
selection, non-empty insufficient support, and excessive observed risk. The
zero-error support requirement SHALL equal
`ceil(log(delta / 9) / log(1 - risk_budget))`. Empty selection SHALL not be
attributed to multiplicity. Coverage SHALL use an exact lower bound.

Before the measured trial, the experiment SHALL seal
`results/raw/experiment_7436_v652_selection_protocol/protocol.json`. The seal
SHALL include role disjointness, group sampling assumptions, seeds, candidate
rules, thresholds, utility, familywise allocation, and 10,000 source-group
bootstrap draws. The joint-certificate paper motivates role separation only;
the artifact SHALL not claim it improves the binary exact bounds. Mutation
checks SHALL cover duplicated groups, tune/certification overlap, empty
selection, and hidden labels.

The current run SHALL declare `MODEL_SPECS=[]`, `model_invoked=false`, zero
current LLM counts, `inference_substrate_class=aggregation`, and
`execution_venue=host`. Numeric fitting receipts SHALL remain separate under
`small_ebm_training`. Archived or scripted model events SHALL remain typed,
hash-bound sidecars.

The artifact SHALL set `deployment_certificate_valid=false` and
`certificate_scope=exploratory_reused_corpus`. A nominal feasibility pass MAY
justify a separately sealed future corpus. It SHALL not authorize rollout or
describe any tested policy as safe for new users. `promotion_score` SHALL remain
zero. `selection_protocol_ready_score=1` SHALL require sealed roles,
finite-sample checks, label-isolation controls, and all required validation. It
SHALL remain independent of whether any policy is nominally feasible.

The workflow SHALL freeze an Exp7358 affected-file manifest. It SHALL use the
Exp7303 runner with private temporary parents. It SHALL run worktree imports,
affected tests without coverage, separate 100-percent changed-module coverage,
scoped Ruff, changed-module mypy, exact-test spec coverage, the declared
entrypoint, fresh-process replay, independent reduction, adversarial
verification, and strict row consistency. These entrypoint checks form the
capability E2E. No numbered E2E scenario applies because shared training,
sampling, bindings, and ARC code do not change.

The terminal artifact SHALL include all ordinary experiment fields plus
`selection_protocol_ready_score`, `selection_diagnosis_rows`,
`protocol_manifest`, `evaluation_reuse_disclosure`,
`deployment_certificate_valid`, and `certificate_scope`. Detailed policy rows
MAY use hash-bound shards. The checksum SHALL bind code, protocol, input bytes,
row shards, and the exact validation scope.

#### SCENARIO-AUTO-7436-01: Role separation rejects leakage

**Given** source groups assigned to fit, scalar calibration, policy tuning, and
certification,
**when** membership is validated,
**then** duplicate groups or tuning/certification overlap fail closed
**and** predictor-side hidden labels are rejected before scoring.

#### SCENARIO-AUTO-7436-02: Seed averaging precedes policy fitting

**Given** five frozen seed states for each deployed head,
**when** calibration and tuning scores are computed,
**then** the five probabilities average before scalar calibration
**and** dense spline matches sparse spline only as an equivalence control.

#### SCENARIO-AUTO-7436-03: Tuning selects exactly one fixed policy

**Given** label-blind representatives and tuning-score deciles,
**when** candidate pairs are evaluated,
**then** one pair per head wins by the frozen asymmetric utility and tie rule
**and** no-action sentinels can disable either action without inventing risk.

#### SCENARIO-AUTO-7436-04: Certification diagnoses support honestly

**Given** one frozen policy and certification-only source groups,
**when** its exact risk and coverage checks run,
**then** empty, insufficient-support, and excessive-risk outcomes stay distinct
**and** a failed conjunction converts the whole policy to all-escalate.

#### SCENARIO-AUTO-7436-05: Reused evidence cannot become deployment proof

**Given** nominal exact bounds on the previously exposed RAGTruth corpus,
**when** the terminal result is classified,
**then** deployment certification remains false with exploratory reused scope
**and** readiness can pass without promotion or a new-user safety claim.

#### SCENARIO-AUTO-7436-06: Cold readers bind the sealed protocol

**Given** the terminal candidate, protocol seal, detailed rows, and validation
receipts,
**when** fresh processes independently replay and reduce the evidence,
**then** changed roles, thresholds, counts, bounds, hashes, gates, or checksum
fail
**and** the terminal JSON publishes only after every required reader passes.

### REQ-AUTO-7439: Measure tuned certified decisions on the preserved corpus

Exp7439 SHALL authenticate the exact Exp7436 selection-protocol artifact and
sealed protocol before dependent work. It SHALL require
`selection_protocol_ready_score=1`, an allowed null or positive verdict, and
`flagged_adversarial=false`. Missing, changed, or ineligible evidence SHALL
produce a terminal `blocked_*` artifact with an exact gate summary.

The experiment SHALL use only the six shipped predictor features. Predictor
inputs SHALL exclude labels, annotation spans, model identity, and group IDs.
It SHALL fit Gibbs 6-4-1, sparse spline 49, and raw L2 logistic heads on fit
groups with seeds 65201 through 65205. Dense spline and no-training prevalence
SHALL remain diagnostics and SHALL not add certification observations. Each
deployed head SHALL average seed probabilities before scalar calibration.
Scalar calibration and selection of one accept/reject pair SHALL use the two
separate group halves sealed by Exp7436.

Certification labels SHALL be read only after all policies are frozen. The
nine registered checks SHALL use exact one-sided bounds with delta 0.05/9:
accept harm at most 0.05, reject harm at most 0.10, and non-escalation coverage
at least 0.25 for each deployed head. A disabled action SHALL report null risk,
not applicable, and zero alpha spent. Every applicable action check and the
coverage check SHALL pass together. A failed certificate SHALL disable the
whole policy without threshold reselection.

Final-test labels SHALL be read once after certification. The result SHALL
report Brier score, log loss, accept and reject harm with denominators,
coverage, and registered utility. Undefined action risks SHALL remain null.
The old fixed-threshold protocol SHALL use the same head predictions and source
groups as the tuned policy. Ten thousand paired source-group bootstrap draws
SHALL give simultaneous intervals for tuned spline coverage minus its old
policy and tuned spline coverage minus tuned raw logistic coverage.

`decision_value_score=1` SHALL require a valid spline certificate, certificate
coverage lower bound at least 0.25, both paired coverage lower bounds above
zero, utility no worse than either control, and Brier and log-loss
non-inferiority within 0.001. A policy-only coverage gain without a gain over
tuned logistic SHALL be a limited policy-wrapper finding with null energy
advantage. Gibbs results SHALL remain visible for every terminal outcome.
`decision_capture_complete_score=1` SHALL allow complete valid nulls.

The current run SHALL declare `MODEL_SPECS=[]`, `model_invoked=false`, zero
current LLM invocation counts, `inference_substrate_class=no_model_load`, and
`execution_venue=host`. It SHALL keep head fitting, scalar calibration, and
policy fitting times separate in `small_ebm_training`. The artifact SHALL set
`deployment_certificate_valid=false`,
`certificate_scope=exploratory_reused_corpus`, and `promotion_score=0` because
historical label exposure prevents a fresh deployment guarantee.

The terminal result SHALL bind probability-row shards, fitted checkpoints,
source bytes, the validation manifest, and its checksum. It SHALL run the
frozen affected-file checks, declared entrypoint, cold replay, independent raw
reduction, adversarial verification, and strict row consistency before atomic
publication. These entrypoint checks form the capability E2E. No numbered E2E
scenario applies because shared training, sampling, bindings, and ARC code do
not change.

#### SCENARIO-AUTO-7439-01: Predictor projection excludes forbidden fields

**Given** preserved predictor and evaluator views,
**when** compact heads receive a training or scoring matrix,
**then** only the six shipped numeric features enter the matrix
**and** labels, spans, model identity, and group identity remain excluded.

#### SCENARIO-AUTO-7439-02: Frozen roles control calibration and certification

**Given** the exact Exp7436 role seal,
**when** probabilities are calibrated, tuned, and certified,
**then** seed probabilities average before calibration
**and** certification labels cannot alter the frozen threshold pair.

#### SCENARIO-AUTO-7439-03: Empty actions and failed certificates stay honest

**Given** a disabled or empty action and a frozen typed policy,
**when** exact risk and coverage checks run,
**then** undefined risks remain null and disabled checks spend no alpha
**and** any failed conjunction disables the whole policy without reselection.

#### SCENARIO-AUTO-7439-04: Policy coverage differs from calibration quality

**Given** final-test probability rows for tuned and old policies,
**when** metrics and paired group intervals are reduced,
**then** coverage contrasts use actions while Brier and log loss use probabilities
**and** controls do not contribute rows to certification counts.

#### SCENARIO-AUTO-7439-05: Benefit requires every registered gate

**Given** a complete valid measurement,
**when** decision value is reduced,
**then** one failed certificate, coverage, contrast, utility, or score gate keeps
`decision_value_score=0` while `decision_capture_complete_score` can remain one.

#### SCENARIO-AUTO-7439-06: Reused labels cannot authorize deployment

**Given** nominally valid bounds on the historically exposed corpus,
**when** the artifact is classified and replayed,
**then** deployment certification and promotion remain false
**and** fresh-process readers reproduce the same measurements from bound rows.

### REQ-AUTO-7427: Measure randomized delayed-feedback source adaptation

Exp7427 SHALL authenticate the exact Exp7426 static-decision artifact before
dependent work. It SHALL require `decision_capture_complete_score=1`, an
allowed `verdict_class`, and `flagged_adversarial=false`. A failed prerequisite
SHALL produce a terminal blocked artifact. The artifact SHALL name the upstream,
path, check, field, expected value, and observed value.

The experiment SHALL initialize only from the Exp7426 full-source numeric
checkpoints, affine calibrators, and frozen policies. It SHALL use one
label-blind representative from each independent Exp7423 prospective-stream
source group. Hash order and a domain-blocked shift order SHALL use only source
and task metadata. No stream label may change an initial checkpoint, order,
block, reveal schedule, or current prediction. The replay is prospective use of
archived human annotations. It is not new live user feedback.

Within each block of 32 requests, frozen initial Gibbs risk SHALL define three
shared reveal schedules. `top_risk_eight` SHALL reveal the eight highest risks.
`uniform_eight` SHALL sample eight groups without replacement.
`hybrid_four_plus_four` SHALL reveal the highest-risk four plus four uniformly
sampled groups from the remaining 28. An incomplete block SHALL reveal
`floor(block_size/4)` groups. Its hybrid schedule SHALL split this total
deterministically between top-risk and uniform selections. The schedule SHALL
be drawn without labels and shared across model arms. Full-block marginal
propensities SHALL be one for deterministic selections, `8/32` for uniform
selection, and `4/28` for the hybrid random remainder. Zero-probability
omissions under top-risk selection SHALL remain explicit. They SHALL not enter
inverse-probability-weighted claims.

The compared arms SHALL be frozen spline, online sparse spline, online Gibbs,
online raw logistic, and a no-feedback spline copy. All online learners SHALL
use the same frozen gradient-norm clipping rule and the same learning rate.
Each revealed label SHALL update only its bounded numeric coefficients. It
SHALL be used once after delay zero or eight groups. Every action and prediction
SHALL be committed before the reveal is enqueued. Every prediction, reveal,
propensity, arrival, coefficient change, failed request, and censored request
SHALL remain in the journal. A numeric checkpoint SHALL bind parent, event, and
new-state hashes after each admitted update.

Independent development fixtures SHALL test revoked feedback, removal and
trusted-journal replay, restart equality, duplicate rejection, early-access
rejection, and rollback after corrupted state. These fixtures SHALL not enter
the scientific replay. Fixed budgets, identical feedback delays, and equal
persistence work SHALL apply to all compared arms.

Inferential claims SHALL require at least 400 independent online groups and at
least 40 examples of each primary label. Lower support SHALL produce
`complete_null_insufficient_online_support` after a complete valid capture.
Proper-score reduction SHALL use inverse-probability weighting only where the
recorded marginal reveal probability is positive. Top-risk-only estimates
SHALL be marked biased. The primary comparison SHALL be hybrid-audit online
spline against frozen spline and online raw logistic. Seeds SHALL average
within each source group before 10,000 moving-block resamples. Primary block
length SHALL be 32 and frozen sensitivity length SHALL be 64. One simultaneous
correction SHALL cover two contrasts, two delays, and two orders. Each schedule,
delay, and order SHALL retain its own result.

`online_value_score` SHALL equal one only when every registered primary
condition has an upper Brier delta below zero against both controls, no worse
log loss, no higher empirical action risk, no coverage loss, and reveal cost at
most 25 percent of independent groups. The no-feedback arm SHALL show no
learning benefit. A development-only shuffled-label control SHALL not improve.
Failure of any benefit clause SHALL yield a valid terminal null when validity
and completion remain intact. Delay and shift evidence SHALL be described as
empirical replay. It SHALL not inherit an IID, conformal, or FDR guarantee.

The experiment SHALL measure prediction, update, journal, and full-service p50
and p95 latency. It SHALL record peak memory and touched coefficients. It SHALL
report whether CPU update cost is below one microsecond and whether memory
lookup cost is below one millisecond. A kernel-only measurement SHALL not be
presented as end-to-end cost. Complete journals and terminal rows SHALL be
stored in hash-bound shards below 20 MiB each.

`online_capture_complete_score` SHALL equal one when temporal rows, controls,
lineage, affected validation, declared entrypoint, cold replay, independent
reduction, adversarial verification, and strict row consistency all pass. A
support-limited or no-gain result MAY receive this score. `promotion_score`
SHALL remain zero. `continuous_self_learning_task` SHALL be true. The authority
SHALL be `human_annotation_source_support`. It is fallible annotation evidence,
not exact proof. `verifier_is_oracle` SHALL be false.

The current run SHALL declare `MODEL_SPECS=[]`, `model_invoked=false`, zero
current LLM counts, `inference_substrate=no_model_load`,
`inference_substrate_class=no_model_load`, and `execution_venue=host`. Numeric
coefficient updates SHALL appear only in `small_ebm_training`. Archived or
scripted model events SHALL remain typed hash-bound sidecars. They SHALL not
enter current invocation counts.

The workflow SHALL freeze an Exp7358 affected-file manifest. It SHALL use the
Exp7303 runner with command-local coverage data. It SHALL run worktree imports,
affected tests, 100-percent changed-module coverage, scoped Ruff, changed-module
mypy, exact-test spec coverage, the declared entrypoint, fresh-process replay,
independent reduction, adversarial verification, and strict row consistency.
No numbered E2E scenario applies because this experiment changes no shared
training, sampling, serialization, PyO3, or ARC behavior.

The terminal artifact SHALL contain the ordinary required experiment fields.
It SHALL also contain `online_capture_complete_score`, `online_value_score`,
`continuous_self_learning_task`, `feedback_event_rows`, `checkpoint_lineage`,
`condition_reports`, `paired_moving_block_intervals`, `hardware_path`, and
`small_ebm_training`. The checksum SHALL bind code, protocol, input bytes, raw
rows, checkpoints, and validation scope.

#### SCENARIO-AUTO-7427-01: Static state and label-blind orders are sealed

**Given** authenticated Exp7426 full-source checkpoints and the prospective view,
**when** initial states and both stream orders load,
**then** no prospective label enters initial state or ordering
**and** every order contains one identical set of independent source groups.

#### SCENARIO-AUTO-7427-02: Shared reveal schedules preserve their propensities

**Given** a 32-request block and frozen initial Gibbs risks,
**when** the three reveal schedules are drawn,
**then** all arms receive identical reveal decisions with exact marginal probabilities
**and** changing labels cannot change a selected group or its propensity.

#### SCENARIO-AUTO-7427-03: Prediction and action precede one delayed update

**Given** a selected event and delay zero or eight,
**when** its human annotation becomes available,
**then** the durable prediction and action precede the reveal and update
**and** the label changes numeric state at most once with a clipped gradient.

#### SCENARIO-AUTO-7427-04: Journal controls restore trusted numeric state

**Given** committed development-fixture feedback and persisted checkpoints,
**when** feedback is revoked, state is restarted, or corrupted state is loaded,
**then** trusted active events replay to the same hash
**and** corrupted descendants roll back to the last authenticated parent.

#### SCENARIO-AUTO-7427-05: Weighted scores fail closed on support and propensity

**Given** retained labels and registered reveal probabilities,
**when** proper scores and support are reduced,
**then** inverse-probability weights use positive propensities only
**and** low support or zero-probability omissions cannot produce a value claim.

#### SCENARIO-AUTO-7427-06: Moving-block benefit keeps every condition

**Given** both orders, both delays, three schedules, and five seeds,
**when** group-averaged paired deltas enter 10,000 moving-block draws,
**then** one simultaneous correction covers both controls, delays, and orders
**and** no failed condition is pooled away.

#### SCENARIO-AUTO-7427-07: No-feedback and shuffled labels reject false learning

**Given** the frozen no-feedback arm and the development-only label shuffle,
**when** the same replay budget completes,
**then** no-feedback remains prediction-identical to frozen spline
**and** a learning claim fails unless the registered benefit disappears there.

#### SCENARIO-AUTO-7427-08: Cold readers bind complete replay evidence

**Given** a terminal candidate with journal, terminal rows, lineage, and costs,
**when** fresh processes replay and independently reduce it,
**then** changed schedules, probabilities, arrivals, hashes, metrics, gates, or receipts fail
**and** terminal JSON publishes atomically only after all required readers pass.

### REQ-AUTO-7438: Prototype a causal four-expert probability mixture

Exp7438 SHALL implement four named experts: frozen spline, adaptive spline,
frozen Gibbs, and adaptive Gibbs. It SHALL call the shipped spline and Gibbs
numeric APIs. It SHALL not load a language model or change generator, ARC, or
production defaults. The four weights SHALL start uniformly. Each expert
probability and the mixture probability SHALL be clipped to
`[1e-6, 1-1e-6]`. The reported energy SHALL be the probability re-expression
`-log(q/(1-q))`, where `q` is the weighted probability sum. It SHALL not be
described as a new generative EBM.

After one label becomes visible, the controller SHALL use only the four expert
probabilities stored with that request. It SHALL apply Bernoulli log loss with
`eta=1`, normalize the log weights stably, and then apply a uniform fixed share
of `0.01`. Only the adaptive spline and adaptive Gibbs experts SHALL receive a
numeric coefficient update. The frozen experts SHALL remain available without
state changes. An unrevealed label, a hindsight prediction, or a future
checkpoint SHALL not enter either update.

The prototype SHALL cover feedback delays zero and eight. Event IDs SHALL
commit at most once. Pending feedback storage SHALL have a fixed capacity and
report its measured peak bytes. A late duplicate SHALL not change weights or
expert state. A restart SHALL preserve predictions, weights, pending feedback,
and numeric expert state. Revocation SHALL restore the last safe checkpoint and
replay later active events. Original prediction hashes SHALL not change.
Checkpoint publication SHALL be atomic, and an interrupted replacement SHALL
leave the prior checkpoint readable.

Analytic controls SHALL cover no-feedback equality to the frozen prior
mixture, constant experts, one persistently harmful expert, a regime change,
extreme finite probabilities, restart, revocation, duplicate feedback, and an
interrupted write. A full-information, no-share fixture SHALL verify the exact
log-loss mixture identity. The delayed fixed-share prototype SHALL claim no
inherited no-share regret, calibration, IID, conformal, or delayed-feedback
guarantee.

The frozen protocol SHALL be written to
`results/raw/experiment_7438_v652_mixture_prototype/protocol.json`. It SHALL
require probability and energy parity within `1e-10`, zero future-label reads,
zero duplicate commits, and exact restart equality. The terminal artifact
SHALL keep per-event prediction, reveal, update, revocation, and restart rows.
It SHALL report actual serialized bytes and update work. Synthetic success
SHALL use `verdict_class=circular_positive` and SHALL set
`verifier_is_oracle=true`. `mixture_prototype_ready_score=1` SHALL certify only
implementation readiness. `promotion_score` SHALL remain zero.

The current run SHALL declare `MODEL_SPECS=[]`, `model_invoked=false`, zero
current LLM invocation counts, `inference_substrate=no_model_load`,
`inference_substrate_class=no_model_load`, and `execution_venue=host`. Compact
numeric updates SHALL appear only in `small_ebm_training`. Archived and
scripted model events SHALL remain typed, hash-bound sidecars.

The workflow SHALL freeze an Exp7358 affected-file manifest. It SHALL use the
Exp7303 runner with a private existing base-temp parent and command-local
coverage data. It SHALL run worktree imports, affected tests, 100-percent
changed-module coverage, scoped Ruff, changed-module mypy, exact-test spec
coverage, the declared entrypoint, fresh-process replay, independent
reduction, adversarial verification, and strict row consistency. The declared
entrypoint and fresh-process replay form the capability E2E. No numbered E2E
scenario applies because shared training, sampling, serialization, PyO3, and
ARC behavior do not change.

The terminal artifact SHALL contain the ordinary required experiment fields.
It SHALL also contain `mixture_prototype_ready_score`, `mixture_definition`,
`learning_control_rows`, `hardware_path`, `small_ebm_training`, and the frozen
validation manifest. The checksum SHALL bind code, protocol, input bytes, raw
control rows, and exact validation scope.

#### SCENARIO-AUTO-7438-01: A prediction is a four-way probability mixture

**Given** equal initial weights and paired frozen and adaptive numeric states,
**when** one request is predicted before feedback,
**then** its probability equals the frozen prior mixture within `1e-10`
**and** its energy is the finite log-odds re-expression of that probability.

#### SCENARIO-AUTO-7438-02: Revealed loss updates the stored prediction once

**Given** one durable request and its four stored expert probabilities,
**when** its label reaches the declared availability index,
**then** the stable eta-one fixed-share update uses those stored probabilities
**and** only the two adaptive numeric experts change state.

#### SCENARIO-AUTO-7438-03: Delayed feedback stays causal and bounded

**Given** delays zero and eight with a fixed pending capacity,
**when** requests, releases, and updates replay,
**then** every prediction precedes its label and no future label is read
**and** actual pending bytes and peak update work remain in the evidence.

#### SCENARIO-AUTO-7438-04: Duplicate, restart, and interruption controls fail closed

**Given** committed and pending events plus one readable checkpoint,
**when** late feedback repeats, the process restarts, or replacement stops early,
**then** duplicate commits remain zero and restarted state is equal
**and** the interrupted write leaves the prior checkpoint readable.

#### SCENARIO-AUTO-7438-05: Revocation replays from safe state

**Given** an active committed label and later committed labels,
**when** the earlier label is revoked,
**then** the controller restores the last safe checkpoint and replays later events
**and** every original prediction hash remains unchanged.

#### SCENARIO-AUTO-7438-06: Analytic expert controls expose adaptation behavior

**Given** constant, harmful-expert, regime-change, and extreme fixtures,
**when** the probability reducer applies the frozen rule,
**then** constant predictions stay constant and harmful weight decreases
**and** fixed share permits recovery after the registered regime change.

#### SCENARIO-AUTO-7438-07: No-share log-loss identity is exact

**Given** a full-information analytic probability table and uniform prior,
**when** the reducer uses eta one with zero share,
**then** cumulative mixture loss equals the negative log marginal likelihood
within `1e-10`
**and** this identity is not transferred to the delayed fixed-share deployment.

#### SCENARIO-AUTO-7438-08: Cold readers bind readiness evidence

**Given** the protocol, raw control rows, current-work receipt, and exact checks,
**when** fresh processes replay and independently reduce the candidate,
**then** drift in identity, chronology, hashes, counters, parity, or gates fails
**and** terminal JSON publishes atomically only after every required check passes.

### REQ-LEARN-010: Constraint Addition from CaseMemory Patterns

When CaseMemory has accumulated error patterns for a violation family with support ≥ 3, the
system shall generate and add new IsingConstraint types to the active extractor set:
- `extract_patterns(case_memory, min_support=3)` groups CaseMemory entries by violation_family,
  computes observed_precision = improved_repairs / total_flagged per family, returns ConstraintPattern
  objects for families meeting the support threshold.
- `generate_arithmetic_constraint(pattern)` maps families to targeted constraint types:
  "carry_error" → carry-propagation check; "sign_error" → sign-consistency check;
  "magnitude_error" → order-of-magnitude check; unknown families → generic learned check.
- `add_to_extractor(extractor, constraint)` appends the generated constraint to the extractor's
  `_dynamic_constraints` list without removing or modifying any existing constraints.
- `constraint_already_exists(extractor, constraint_id)` prevents duplicate insertion.

### REQ-LEARN-011: Soundness Bound for Constraint Addition

Constraint addition shall be gated by a soundness bound derived from arXiv 2603.03538
(CoT Verifier Online Learnability):
- `soundness_filter(patterns, min_precision=0.85)` returns only patterns where
  observed_precision ≥ min_precision (85% of flagged cases were confirmed real errors).
- Patterns below min_precision are explicitly logged as "rejected_soundness" in the
  `ConstraintGenerator.generation_log`, not silently dropped.
- The 0.85 threshold ensures that generated constraints have high precision — they rarely
  flag correct answers — preserving the soundness of downstream verification.

### REQ-LEARN-020: JEPA Training Domain Coverage Pre-flight

JEPA training MUST assert that every target evaluation domain has at least 15 labeled
training pairs before the first gradient step.  If any domain is below the threshold,
training must raise AssertionError with a diagnostic message naming the domain and
its actual pair count.

Rationale: DreamPRM per-domain loss reweighting (arXiv 2505.20241) multiplies the
loss for each sample by a domain scalar.  When a domain has zero training samples,
the weight scalar is mathematically irrelevant — no gradient signal exists.  Exp 834
demonstrated this failure mode: auc_svamp=0.0 despite SVAMP weight=1.5 because there
were only 10+10 step texts (not triplets) with insufficient TF-IDF vocabulary coverage.

Implementation: `assert_domain_coverage(n_gsm8k, n_humaneval, n_arc, n_svamp, min_pairs=15)`
raises AssertionError for any domain below the threshold.

## Scenarios

### SCENARIO-AUTO-001: Successful Self-Improvement Cycle

**Given** a baseline Ising model trained with CD-1, achieving energy -5.2 on the DoubleWell benchmark
**When** the autoresearch loop proposes switching to Denoising Score Matching
**And** the sandbox evaluates DSM on DoubleWell, achieving energy -5.8
**And** wall-clock time is 0.7x baseline
**Then** the hypothesis passes all three gates
**And** it is queued for Rust transpilation
**And** after Rust validation matches JAX output, it is merged as the new baseline
**And** the baseline registry is updated with the new metrics

### SCENARIO-AUTO-002: Rejected Hypothesis

**Given** a baseline Langevin sampler with step_size=0.01
**When** the autoresearch loop proposes step_size=0.5
**And** the sandbox evaluates this, producing divergent (NaN) energies
**Then** the hypothesis fails the primary gate
**And** it is logged as rejected with reason "divergent energy"
**And** the production model is unaffected

### SCENARIO-AUTO-003: Automatic Rollback

**Given** a hypothesis that passed sandbox evaluation
**And** was transpiled to Rust and deployed
**When** production energy metrics degrade by 8% over the monitoring window
**Then** the system automatically reverts to the previous version
**And** logs the regression with metrics
**And** adds the hypothesis to the rejected registry

### SCENARIO-AUTO-004: Safety Invariant Enforcement

**Given** a hypothesis that attempts to write to the validation dataset
**When** it is executed in the sandbox
**Then** the write is blocked by the sandbox filesystem policy
**And** the hypothesis is immediately terminated and flagged as unsafe

### SCENARIO-AUTO-005: Consecutive Failure Halt

**Given** a configuration with max_consecutive_failures=10
**When** 10 consecutive hypotheses fail evaluation
**Then** the autoresearch loop halts
**And** an alert is generated with the failure log
**And** the system waits for human review before resuming

### SCENARIO-AUTO-006: Improvement Composition

**Given** Hypothesis A improves sampler convergence by 15%
**And** Hypothesis B improves training loss by 10%
**When** both pass individual evaluation
**And** combined evaluation also passes all gates
**Then** both improvements are merged as a single update
**And** the baseline registry reflects the combined metrics

### SCENARIO-AUTO-007: Benchmark Regression Prevention

**Given** a hypothesis that improves DoubleWell energy by 5%
**But** degrades Rosenbrock energy by 3%
**When** evaluated across the full benchmark suite
**Then** the hypothesis is flagged for review (mixed results)
**And** is not auto-merged (requires human decision on the tradeoff)

### SCENARIO-AUTO-008: Error Analyst Diagnoses Gradient Explosion

**Given** a hypothesis that attempted Langevin sampling on Rosenbrock with step_size=0.1
**And** the sandbox produced NaN energies after 47 steps
**When** the error analyst receives the full experiment trajectory
**Then** it produces a Lesson with title "Gradient explosion on steep landscapes"
**And** description identifies the `100*(x[i+1]-x[i]^2)^2` term curvature as the root cause
**And** applicable_benchmarks includes "rosenbrock"
**And** confidence is >= 0.7 (clear diagnosis from the execution trace)

### SCENARIO-AUTO-009: Success Analyst Extracts Annealing Pattern

**Given** a hypothesis that used step_size annealing from 0.1 to 0.001 over 5000 steps
**And** it was accepted with 30% energy improvement on DoubleWell
**When** the success analyst receives the full experiment trajectory
**Then** it produces a Lesson with title "Step-size annealing for multi-basin landscapes"
**And** the description generalizes beyond the specific parameters to the annealing principle
**And** lesson_type is "success_pattern"

### SCENARIO-AUTO-010: Lessons Consolidated Across 10 Iterations

**Given** 15 raw lessons accumulated over 10 iterations (8 error, 7 success)
**And** 3 pairs of near-duplicate lessons (e.g., both say "small step sizes prevent divergence")
**When** hierarchical consolidation runs
**Then** duplicates are merged (confidence increased)
**And** the consolidated set has fewer lessons than the input
**And** contradictory lessons (e.g., "use large step size" vs "use small step size") are resolved
**And** lessons below min_confidence are filtered out

### SCENARIO-AUTO-011: Ising Skill Transfers to Gibbs Model

**Given** a skill directory with 5 lessons learned from Ising model experiments
**And** one lesson is "HMC outperforms Langevin on narrow-valley landscapes" (model_tier="ising")
**When** generating hypotheses for the Gibbs model tier
**Then** `to_prompt_context(model_tier="gibbs")` includes the Ising HMC lesson
**And** the generator's prompt contains this cross-tier knowledge
**And** the generated hypothesis tries HMC on the Gibbs tier

### SCENARIO-AUTO-018-A: Real Weights Recompute Real Calibrated-Decision Metrics

**Given** a hypothesis trains a `GibbsModel` and reports `final_state` (its trained
weights, as plain nested lists/floats)
**When** the harness recomputes `final_energy` and `brier` against the fixed
held-out split
**Then** both numbers are real, independently measured, and bounded in `[0.0, 1.0]`
**And** the hypothesis's own training-set metric is never trusted or seen by the
evaluator

### SCENARIO-AUTO-018-B: A Degenerate Weight Set Is Rejected From the Accept Path

**Given** a hypothesis's weights score every held-out row identically (e.g. an
all-zero output layer)
**When** the harness recomputes its metrics
**Then** the result is `None`, not a fabricated chance-level "improvement"
**And** the SAME zero-initialized state, measured with `reject_degenerate=False`
for the seed-baseline path only, honestly reports `final_energy=0.5, brier=0.25`

### SCENARIO-LEARN-015: extract_patterns Groups CaseMemory by Violation Family

**Given** a CaseMemory with 5 entries for violation_family "carry_error" (3 "improved", 2 "unchanged_failure")
**When** `extract_patterns(case_memory, min_support=3)` is called
**Then** one ConstraintPattern is returned for family "carry_error"
**And** observed_precision = 3/5 = 0.60
**And** support_count = 5
**And** families below min_support=3 are absent from the result

### SCENARIO-LEARN-016: soundness_filter Rejects Low-Precision Patterns

**Given** two ConstraintPatterns: one with observed_precision=0.90 and one with 0.60
**When** `soundness_filter(patterns, min_precision=0.85)` is called
**Then** only the pattern with precision 0.90 is returned
**And** the rejected pattern is NOT silently dropped — it must be tracked by the caller

### SCENARIO-LEARN-017: generate_arithmetic_constraint Maps Families to Constraint Types

**Given** ConstraintPatterns for families "carry_error", "sign_error", "magnitude_error"
**When** `generate_arithmetic_constraint(pattern)` is called for each
**Then** "carry_error" yields a LearnedConstraint with constraint_id "learned:carry_error"
**And** "sign_error" yields constraint_id "learned:sign_error"
**And** "magnitude_error" yields constraint_id "learned:magnitude_error"
**And** each LearnedConstraint has a human-readable description of what it checks

### SCENARIO-LEARN-020: Training with Zero Domain Pairs Fires Assertion with Diagnostic

**Given** a JEPA v24b training run where the SVAMP corpus is empty (0 pairs)
**When** `build_corpus_v24b(svamp_triplets=[])` is called
**Then** `assert_domain_coverage(n_svamp=0)` is invoked internally
**And** AssertionError is raised before any gradient step
**And** the message contains "SVAMP coverage insufficient" and the actual count "0 pairs"
**And** no model weights are written (fail-fast, no partial checkpoint)

### SCENARIO-LEARN-018: ConstraintGenerator Orchestrates and Logs All Outcomes

**Given** a CaseMemory with high-precision carry_error (precision=0.92), low-precision sign_error (0.60),
and an already-existing magnitude_error constraint in the extractor
**When** `ConstraintGenerator().generate_from_memory(case_memory, extractor)` is called
**Then** generation_log["carry_check:carry_error"] == "added"
**And** generation_log["sign_consistency:sign_error"] == "rejected_soundness"
**And** generation_log["magnitude_check:magnitude_error"] == "already_exists"
**And** only carry_error is added to extractor._dynamic_constraints

### REQ-LEARN-030: FOVER-Style Z3 Step Annotation

The system shall provide a `FOVERAnnotator` that implements FoVer-style annotation
(arXiv 2505.15960) of chain-of-thought reasoning steps with Z3-verified correctness labels:

- Parse a CoT response into discrete steps using `parse_cot_into_steps(response) -> list[FOVERCoTStep]`.
  Split on numbered steps ("1.", "2.") or "Step N:" patterns.
- For each step, extract any claimed arithmetic equation using the same `_INLINE_EQ` pattern as
  CRANEExtractionGate.
- Annotate each step via `annotate_step_with_z3(step) -> FOVERCoTStep`:
  - If no equation: `z3_label='not_verifiable'`, `z3_confidence=0.0`
  - If equation present: run `_exec_z3_snippet` on an inline Z3 assertion.
  - `z3_label='correct'` when Z3 returns 'sat'; `z3_label='incorrect'` when Z3 returns 'unsat'.
  - `z3_confidence=1.0` for a complete equation (all three operands present); `0.5` otherwise.
- `FOVERAnnotator(z3_timeout_seconds=5)` wraps the above with corpus-level batching and
  training-pair output.

Rationale: Z3 annotations are deterministic, scalable, and formally correct — unlike
human labels which require expensive annotation at scale. This is the training signal
that was missing for FR-11 (EORM/JEPA retrains on synthetic-only data).

Spec refs: arXiv 2505.15960 (FoVer), arXiv 2601.17223 (VPRM)

### REQ-LEARN-032: EORM Retrain on FOVER-Labeled Real Pairs

EORM retrained on FOVER-labeled (correct_step, incorrect_step) contrastive pairs from
real LLM inference shall achieve AUC-ROC > 0.5 on the held-out 20% test split.

- `load_fover_pairs(path) -> list[dict]`: load FOVER labeled steps from JSON; filter by
  `label in ('correct', 'incorrect')` and `confidence >= 0.3`.
- `fover_pairs_to_contrastive(pairs) -> list[tuple[jnp.ndarray, jnp.ndarray]]`: convert
  co-occurring (correct, incorrect) pairs on the same question into (positive, negative)
  contrastive tensors for EORM training.
- If `n_real_pairs < 10`, fall back to synthetic data; set `honest_verdict='synthetic_only'`.
- `honest_verdict='real_data_improvement'` requires both `after_auc > before_auc` AND
  `n_real_pairs >= 10`.
- Retro-024 is closed when `honest_verdict='real_data_improvement'`.

Spec refs: arXiv 2505.14999 (EORM), RETRO-024

### REQ-LEARN-033: JEPA Predictor Retrain on FOVER Pairs

The JEPA predictor shall be retrained on the same FOVER-labeled pairs, treating each
labeled step as a (partial_step_prefix, violation_occurred) pair:

- `label='incorrect'` maps to `has_violation=True`
- `label='correct'` maps to `has_violation=False`
- Use existing `JEPARetrainer.train_epoch()` and `evaluate_auc_roc()`.
- Save the retrained model to `jepa_431_real.safetensors`.

### REQ-LEARN-031: FOVER Training Pair Export

The annotated (step, label) pairs shall be written to `results/fover_labeled_steps.json`
for use as training targets for EORM (Exp 431). The export format:

- Schema: `carnot.fover_labels.v1`
- Each pair: `{question_id, step_text, label, confidence}`
- Filter: only include steps where `z3_label in ('correct', 'incorrect')` AND
  `z3_confidence >= 0.3`. Steps labeled `not_verifiable` are excluded from training
  pairs because they provide no learning signal.
- `FOVERAnnotator.to_training_pairs(annotated) -> list[dict]` performs this filtering.

## Scenarios

### SCENARIO-LEARN-054: Numbered Step Parsing

**Given** a CoT response with text "1. First step. 2. Second step. 3. Third step."
**When** `parse_cot_into_steps(response)` is called
**Then** exactly 3 `FOVERCoTStep` objects are returned
**And** each step has the correct `step_idx` (0, 1, 2)

### SCENARIO-LEARN-055: Z3 Correct Equation Label

**Given** a `FOVERCoTStep` with `claimed_equation="2 + 3 = 5"`
**When** `annotate_step_with_z3(step)` is called
**Then** the returned step has `z3_label='correct'`
**And** `z3_confidence >= 0.3`

### SCENARIO-LEARN-056: Z3 Incorrect Equation Label

**Given** a `FOVERCoTStep` with `claimed_equation="2 + 3 = 6"`
**When** `annotate_step_with_z3(step)` is called
**Then** the returned step has `z3_label='incorrect'`

### SCENARIO-LEARN-057: FOVER Pair Loading with Confidence Filter

**Given** a FOVER-labeled JSON file with 100 pairs (80 with confidence >= 0.3, 20 with confidence < 0.3)
**When** `load_fover_pairs(path)` is called
**Then** exactly 80 pairs are returned (confidence filter applied)
**And** all returned pairs have `label in ('correct', 'incorrect')`

### SCENARIO-LEARN-058: Contrastive Tensor Conversion

**Given** a list of FOVER pairs including co-occurring correct and incorrect steps for the same question
**When** `fover_pairs_to_contrastive(pairs)` is called
**Then** a list of (positive, negative) tensor tuples is returned
**And** each tuple contains two `jnp.ndarray` vectors (one per step text)
**And** pairs from different questions are NOT cross-matched

### SCENARIO-LEARN-059: Retrain Verdict Computation

**Given** `before_auc=0.5`, `after_auc=0.62`, `n_real_pairs=25`
**When** `compute_retrain_verdict(0.5, 0.62, 25)` is called
**Then** the result is `'real_data_improvement'`

**Given** `n_real_pairs=5` (below threshold)
**When** `compute_retrain_verdict(0.5, 0.8, 5)` is called
**Then** the result is `'synthetic_only'`

**Given** `before_auc=0.62`, `after_auc=0.55`, `n_real_pairs=25`
**When** `compute_retrain_verdict(0.62, 0.55, 25)` is called
**Then** the result is `'real_data_no_improvement'`

### REQ-LEARN-034: JitRL Memory Validated on Real Data Produces Measurable FP Reduction

The system shall validate JitRL constraint memory on real GSM8K model output data
(from Exp 427) and produce a measurable false-positive reduction percentage.

The validation shall:
- Load real violation records from an Exp 427 result file (or fall back to synthetic
  if unavailable).
- Split records into a warm-up set (first 50) and a validation set (last 50).
- Feed warm-up records into ``JitRLConstraintMemory.record()`` to build per-domain
  threshold history.
- Compare fp_rate before JitRL (no threshold adaptation) vs after (with adapted
  thresholds applied to the validation set).
- Compute ``fp_reduction_pct = (before_fp - after_fp) / before_fp * 100`` (0 if
  before_fp == 0).
- Produce an honest verdict:
  - ``'live_fp_reduction'``   — source='live' AND fp_reduction_pct > 0
  - ``'live_no_reduction'``   — source='live' AND fp_reduction_pct <= 0
  - ``'synthetic_fallback'``  — source='synthetic' (Exp 427 unavailable)

Spec: Exp 432

### SCENARIO-LEARN-060: JitRL Record Raises Threshold for FP Domain

**Given** a ``JitRLConstraintMemory`` with base_threshold=0.5, lr=0.02
**When** ``record('rate_problems', violation_energy=0.6, was_fp=True)`` is called
**Then** ``threshold('rate_problems')`` returns 0.52
**And** the history list has length 1

### SCENARIO-LEARN-061: JitRL Validation Artifact Schema

**Given** ``before_fp=0.3``, ``after_fp=0.2``, ``n_questions=50``, ``source='live'``
**When** ``build_jitrl_validation_artifact(0.3, 0.2, 50, 'live')`` is called
**Then** the result has ``schema='carnot.jitrl_validation.v1'``
**And** ``fp_reduction_pct`` is approximately 33.33
**And** ``honest_verdict`` is ``'live_fp_reduction'``

**Given** ``before_fp=0.0``, ``after_fp=0.0``, ``n_questions=50``, ``source='synthetic'``
**When** ``build_jitrl_validation_artifact(0.0, 0.0, 50, 'synthetic')`` is called
**Then** ``fp_reduction_pct`` is 0
**And** ``honest_verdict`` is ``'synthetic_fallback'``

### REQ-LEARN-035: FOVER Annotator Runs on Live GPU CoT Data

The system shall run FOVERAnnotator on live GPU CoT responses (from Exp 439) and produce
labeled training pairs sufficient for downstream EORM/JEPA training:

- Load ``results/experiment_439_live_cot.json``; if present and the companion
  ``results/experiment_439_live_precision_micro.json`` confirms ``inference_mode='live_gpu'``,
  treat all cot_responses as real data (``source='live'``).
- If the live CoT file is absent or the companion does not confirm live_gpu, fall back to
  generating 100 synthetic GSM8K-style CoT responses (``source='synthetic'``).
- Run ``FOVERAnnotator.annotate_corpus(responses)`` then ``to_training_pairs(annotated, responses)``.
- Write labeled pairs to ``results/fover_labeled_steps_live.json``
  (separate file from Exp 430's ``fover_labeled_steps.json``).
- Honest verdict:

  - ``'real_data_labeled'``      — source='live' AND n_labeled >= 20
  - ``'real_data_insufficient'`` — source='live' AND n_labeled < 20
  - ``'synthetic_fallback'``     — source='synthetic'

Spec: Exp 442

### SCENARIO-LEARN-062: Live CoT Annotation Produces Labeled Pairs

**Given** ``results/experiment_439_live_cot.json`` contains 300 cot_responses
**And** the companion confirms ``inference_mode='live_gpu'``
**When** ``build_live_fover_artifact(result)`` is called with source='live' and n_labeled>=20
**Then** ``honest_verdict`` is ``'real_data_labeled'``
**And** ``schema`` is ``'carnot.fover_live.v1'``
**And** ``labeling_rate`` equals ``n_labeled / n_steps_found``

### SCENARIO-LEARN-063: Synthetic Fallback When Live Data Absent

**Given** ``results/experiment_439_live_cot.json`` is absent or does not confirm live_gpu
**When** ``build_live_fover_artifact(result)`` is called with source='synthetic'
**Then** ``honest_verdict`` is ``'synthetic_fallback'``
**And** ``schema`` is ``'carnot.fover_live.v1'``

### REQ-LEARN-036: EORM Retrained on Real FOVER Pairs Achieves AUC > 0.50

The system shall retrain EORM on ≥20 real FOVER-labeled (step_text, label) pairs sourced from
live GPU CoT annotation (Exp 442) and evaluate on a held-out 20% test split:

- Load ``results/fover_labeled_steps_live.json`` (Exp 442 output).
- If n_labeled < 20: use synthetic fallback (``honest_verdict='real_data_insufficient'``).
- Evaluate EORM ``before_auc`` on held-out 20% split.
- Retrain EORM for 150 epochs on contrastive triples derived from real pairs.
- Also retrain JEPA predictor on (step_prefix, violation_flag) pairs.
- Evaluate ``after_auc`` on same held-out split.
- Use ``compute_retrain_verdict_v2(before_auc, after_auc, n_real_pairs, source)`` to produce verdict:

  - ``'real_data_improvement'``   — source='live' AND after_auc > before_auc AND n_real_pairs >= 20
  - ``'real_data_no_improvement'``— source='live' AND after_auc <= before_auc AND n_real_pairs >= 20
  - ``'real_data_insufficient'``  — source='live' AND n_real_pairs < 20
  - ``'synthetic_only'``          — source='synthetic'

- Save models: ``results/eorm_443_live.safetensors``, ``results/jepa_443_live.safetensors``.
- Set ``retro_024_closed=True`` iff ``honest_verdict='real_data_improvement'``.
- Emit artifact ``schema='carnot.eorm_jepa_retrain.v3'``.

Spec: Exp 443

### SCENARIO-LEARN-064: EORM Retrain on Real Pairs Closes RETRO-024

**Given** ``results/fover_labeled_steps_live.json`` contains ≥20 labeled pairs (n_correct > 0, n_incorrect > 0)
**When** ``compute_retrain_verdict_v2(before_auc, after_auc, n_real_pairs=57, source='live')`` is called
**And** after_auc > before_auc
**Then** verdict is ``'real_data_improvement'``
**And** ``retro_024_closed=True`` in the Exp 443 artifact

### SCENARIO-LEARN-065: Synthetic Fallback When Pairs Insufficient

**Given** n_real_pairs < 20 for Exp 443
**When** ``compute_retrain_verdict_v2(before_auc, after_auc, n_real_pairs=5, source='live')`` is called
**Then** verdict is ``'real_data_insufficient'``
**And** ``retro_024_closed=False`` in the artifact

### REQ-LEARN-037: CoTPairQualityFilter Rejects Low-Quality Training Pairs

The system shall provide a ``CoTPairQualityFilter`` that rejects CoT training pairs
where ``arithmetic_coverage < 0.3`` OR ``label_confidence < 0.7``.

This filter addresses RETRO-040: Exp 472 AUC regressed from 0.667 to 0.400 because
54 real CoT pairs included partially-verifiable steps labeled with low confidence.
Quality gating before training prevents noisy supervision from degrading the JEPA
energy landscape.

Spec: Exp 477

### REQ-LEARN-038: JEPAQualityAugmentor Generates EBM-Guided Synthetic Pairs

The system shall provide a ``JEPAQualityAugmentor`` that generates synthetic CoT
training pairs by sampling from the Ising model's violation distribution (high-energy
spin configurations = incorrect, low-energy = correct).

This is NOT random synthetic generation. The EBM's own energy landscape provides a
principled source of violations: the Ising coupling matrix encodes learned constraint
interactions, so sampling near energy maxima produces examples representative of the
actual failure modes the JEPA must predict.

Spec: Exp 477, FR-11 JEPA, RETRO-040

### REQ-LEARN-039: JEPA Retrained on Quality-Gated Corpus Targets AUC > 0.700

The system shall retrain the JEPA model on a quality-gated corpus (filtered real pairs
+ EBM-guided synthetic pairs) and target AUC > 0.700 on a held-out 20% test set.

Regression baseline is AUC = 0.400 (Exp 472). Recovery threshold is AUC > 0.571
(Exp 443 level). RETRO-040 is closed when AUC > 0.600.

Spec: Exp 477, FR-11 JEPA, RETRO-040

### SCENARIO-LEARN-066: CoTPairQualityFilter Rejects Low-Coverage Pairs

**Given** a CoT pair with arithmetic_coverage=0.2 and label_confidence=0.8
**When** ``CoTPairQualityFilter(min_coverage=0.3, min_confidence=0.7).filter([pair])``
**Then** the returned list is empty (pair rejected for coverage < 0.3)

### SCENARIO-LEARN-067: JEPAQualityAugmentor Violation Pairs Are Labeled Incorrect

**Given** a small Ising model (input_dim=8)
**When** ``JEPAQualityAugmentor(ising_model, n_samples=10).generate_violation_pairs()``
**Then** all returned pairs have ``correct=False``
**And** the energy of their spin configurations is above the model's mean energy

### SCENARIO-LEARN-068: JEPARetrainV2Result Regression Recovery Detection

**Given** ``before_auc=0.400``, ``after_auc=0.620``
**When** ``JEPARetrainV2Result(n_pairs_raw=57, n_pairs_filtered=30, n_synthetic=170, before_auc=0.400, after_auc=0.620)``
**Then** ``regression_recovered=True`` (after_auc > 0.571)
**And** ``retro_040_closed=True`` (after_auc > 0.600)
**And** ``target_met=False`` (after_auc <= 0.700)

### REQ-LEARN-040: JEPACurriculumTrainer Stage 1 Trains on High-Confidence Pairs

``JEPACurriculumTrainer.train(pairs)`` stage 1 shall train the EORM model exclusively on
pairs where ``label_confidence >= high_conf_threshold`` (default 0.85), establishing a
stable baseline before full-distribution exposure.  This prevents the majority-class
collapse that caused the 0.281 AUC regression in Exp 477.

Spec: Exp 492, RETRO-040, arXiv 2509.14252

### REQ-LEARN-041: JEPACurriculumTrainer Stage 2 Fine-Tunes on All Pairs Unfiltered

``JEPACurriculumTrainer.train(pairs)`` stage 2 shall fine-tune the model (warm-started from
stage 1) on ALL available pairs without any confidence gate.  The information loss from
quality-gate filtering is the root cause of the regression — stage 2 recovers that lost
information after stage 1 provides a stable anchor.

Spec: Exp 492, RETRO-040

### REQ-LEARN-042: JEPACurriculumTrainer Stage 3 Augments to n_total >= 200 and Validates AUC

``JEPACurriculumTrainer.train(pairs)`` stage 3 shall augment the training corpus with
EBM-guided synthetic pairs (via ``JEPAQualityAugmentor``) until the total corpus size
reaches at least 200 pairs, then train for ``n_stage3_epochs`` epochs.
``JEPACurriculumTrainer.get_final_auc(held_out_pairs)`` shall return AUC on any held-out set.
``JEPARetrainV3Result.target_met`` shall be True iff ``after_auc > 0.600``.

Spec: Exp 492, RETRO-040

### SCENARIO-LEARN-069: Stage 1 Excludes Low-Confidence Pairs

**Given** a corpus with pairs of varying ``label_confidence`` (some above 0.85, some below)
**When** ``JEPACurriculumTrainer(high_conf_threshold=0.85).train(pairs)`` stage 1
**Then** stage 1 ``n_pairs`` counts only pairs with ``label_confidence >= 0.85``
**And** no pair with ``label_confidence < 0.85`` enters stage 1 training

### SCENARIO-LEARN-070: JEPARetrainV3Result.regression_recovered When after_auc > 0.400

**Given** ``before_auc=0.281``, ``after_auc=0.450``
**When** ``JEPARetrainV3Result(n_pairs_raw=57, curriculum_stages=[...], before_auc=0.281, after_auc=0.450)``
**Then** ``regression_recovered=True`` (after_auc > 0.400, reversing the quality-gate collapse)
**And** ``target_met=False`` (after_auc <= 0.600, not yet at closure bar)

### SCENARIO-RETRO-032: Milestone 2026.04.32 Retrospective Complete

**Given** result JSONs for Exps 425-435 (partial: some results are scaffolding_only or absent)
**When** ``MilestoneRetro2026_04_32`` is computed from available result files
**Then** the artifact has ``schema='carnot.operational_retro.v6'``
**And** ``milestone='2026.04.32'``
**And** ``conductor_timeout_implemented`` reflects whether experiment_watchdog.py is present
**And** ``gpu1_zombie_fixed`` reflects Exp 426 retro_025_status
**And** ``live_numbers_confirmed`` is True only if at least one live benchmark produced a signed improvement
**And** ``fr11_relay_confirmed`` reflects Exp 431 retro_024_closed
**And** all boolean fields are derived exclusively from result JSON files, never asserted without provenance

### SCENARIO-RETRO-033: Milestone 2026.04.33 Retrospective Complete

**Given** result JSONs for Exps 437-448 (partial: Exp 446 absent, Exp 444 timed out)
**When** ``MilestoneRetro2026_04_33`` is computed from available result files
**Then** the artifact has ``schema='carnot.operational_retro.v7'``
**And** ``milestone='2026.04.33'``
**And** ``retro_026_resolved`` reflects Exp 437 retro_026_resolved flag
**And** ``retro_025_resolved`` reflects whether Exp 438 fix_applied is True
**And** ``live_precision_result`` is the honest_verdict from Exp 439 (first live GPU precision run)
**And** ``live_humaneval_result`` is the honest_verdict from Exp 440 (first live GPU HumanEval run)
**And** ``live_adversarial_result`` is the honest_verdict from Exp 441 (first live GPU adversarial run)
**And** ``fr11_relay_confirmed`` reflects Exp 443 retro_024_closed
**And** ``think_probe_viable`` is False when Exp 444 timed_out=True
**And** ``kaem_faster`` is True only when Exp 447 mean_speedup > 5
**And** ``cross_session_improvement`` is True only when Exp 448 honest_verdict indicates improvement
**And** all boolean fields are derived exclusively from result JSON files, never asserted without provenance
**And** the headline confirms live GPU benchmark numbers were obtained for the first time after 7 consecutive scaffolding-only milestones

### REQ-SELFLEARN-019: PPSConstraintLearner Maintains Partition Isolation on Naturally-Interleaved Real Violations

The system shall maintain ``partition_isolation_score > 0.7`` on naturally-interleaved
real violation sequences from live CoT data.  "Naturally interleaved" means steps are
processed in their original occurrence order — NOT sorted by domain.  The threshold is
0.7 (vs 0.8 on synthetic data) to account for real-data noise in domain label assignment.

RETRO-043: Exp 470 used independent domain batches (easy mode).  This requirement
closes RETRO-043 by validating under the harder interleaved condition.

Spec: Exp 485, FR-11, RETRO-043, arXiv 2512.15658

### REQ-SELFLEARN-020: Interleaved Validation Uses FOVERAnnotator-Labeled Steps in Natural Order

The system shall use FOVERAnnotator-labeled steps in the order they occurred in the live
CoT chain (not sorted by domain) when validating PPSConstraintLearner.  Training batches
drawn from the natural-order sequence will contain mixed domains, stressing the partition
walls.

Spec: Exp 485, FR-11, RETRO-043

### SCENARIO-SELFLEARN-019: InterleavedViolationSequence Alternating Steps Have Rate 1.0

**Given** a sequence of steps alternating between arithmetic and code domains
**When** ``InterleavedViolationSequence(steps)``
**Then** ``interleaving_rate == 1.0``
**And** every adjacent pair in ``domain_sequence`` has a different domain label

### SCENARIO-SELFLEARN-020: PPSEBMRealValidationResult Isolation Maintained at 0.75

**Given** ``isolation_score_after=0.75``, ``synthetic_isolation_score=1.0``
**When** ``PPSEBMRealValidationResult(n_steps=57, interleaving_rate=0.4, isolation_score_before=1.0, isolation_score_after=0.75, fp_rate_real=0.05)``
**Then** ``isolation_maintained=True`` (0.75 > 0.7)
**And** ``better_than_synthetic=False`` (0.75 < 1.0)

### REQ-DIAG-001: JEPA Corpus Diagnostic Analysis

``JEPACurriculumDiagnostic.analyze_corpus(quality_filter)`` shall compute the following
metrics for the quality-gated corpus:
- ``label_imbalance_ratio``: ratio of correct to incorrect steps in the filtered corpus
- ``filter_rate``: fraction of raw pairs that passed the quality filter
- ``n_pairs_remaining`` (``n_pairs_filtered``): count of pairs surviving the filter

These metrics together diagnose the RETRO-040 regression: excessive filtering combined
with label imbalance causes JEPA majority-class collapse and AUC below 0.5.

Spec: Exp 491, RETRO-040

### REQ-DIAG-002: JEPA Curriculum Regime Simulation

``JEPACurriculumDiagnostic.simulate_regime(regime, n_epochs=100)`` shall train a JEPA
(EORM) model on pairs selected/ordered by the given regime for n_epochs passes, then
return AUC on a held-out 20% set.

Supported regimes:
- ``'all_pairs'``: train on all pairs — baseline
- ``'quality_gated'``: train only on pairs with label_confidence >= 0.7 — reproduces Exp 477
- ``'curriculum_high_to_low'``: sort by decreasing label_confidence — curriculum learning
- ``'random_50pct'``: random 50% sample — isolates size effect from quality effect

Return value is a float in [0, 1]. AUC < 0.5 indicates the model learned an inverted signal.

Spec: Exp 491, RETRO-040

### SCENARIO-DIAG-001: CorpusAnalysis is_imbalanced for Heavily Correct Corpus

**Given** ``n_correct=80``, ``n_incorrect=5``, ``label_imbalance_ratio=16.0``
**When** ``CorpusAnalysis.is_imbalanced``
**Then** ``is_imbalanced=True`` (16.0 > 3.0)

### SCENARIO-DIAG-002: CorpusAnalysis diagnosis='imbalance' When Filtered Corpus is Imbalanced

**Given** ``is_imbalanced=True`` and ``n_pairs_filtered >= 5``
**When** ``CorpusAnalysis.diagnosis``
**Then** ``diagnosis='imbalance'``

This is the Exp 477 root cause: the quality gate removed 73% of pairs, leaving a corpus
with many more correct than incorrect steps. The JEPA model collapsed to majority-class
prediction (predict all correct), yielding AUC = 0.281 (actively wrong, not just random).

### SCENARIO-DIAG-003: simulate_regime Returns Float in [0, 1]

**Given** a ``JEPACurriculumDiagnostic`` with labeled pairs
**When** ``simulate_regime(regime)`` for any valid regime string
**Then** return value is a ``float`` in ``[0, 1]``

### REQ-SELFLEARN-021: SuRePriorityReplay Ranks Violations by EBM Energy Surprise

SuRePriorityReplay (arXiv 2511.22367) shall rank constraint violations in the replay
buffer by EBM energy surprise — energy relative to the per-domain running mean — and
select the top-k highest-surprise violations for replay at each update step.

Uniform random replay wastes replay budget on easy examples the model already handles
well. Surprise-driven selection (SuRe) prioritizes examples near the knowledge
boundary — where catastrophic forgetting is most likely. arXiv 2511.22367 shows +5%
accuracy improvement in continual learning from NLL-based surprise prioritization.

In the EBM domain, high Ising energy = multiple competing low-energy configurations =
the constraint is ambiguous for the current model = high cognitive surprise. This is a
direct translation of NLL-based surprise to the EBM domain.

Spec: Exp 497, FR-11, arXiv 2511.22367

### REQ-SELFLEARN-022: SuRePriorityReplay Maintains Partition Isolation > 0.7

SuRePriorityReplay combined with PPSConstraintLearner shall maintain
partition_isolation_score > 0.7 on interleaved real violation sequences from
fover_labeled_steps_live.json (57 steps). The isolation score with SuRe replay
shall be compared against uniform random replay as a baseline.

Spec: Exp 497, FR-11, arXiv 2511.22367

### SCENARIO-SELFLEARN-021: SuRePriorityReplay Returns Highest-Surprise Items First

**Given** a SuRePriorityReplay with violations at energies [0.1, 0.9, 0.5, 0.8]
**When** domain_mean_energy = 0.4 and get_replay_batch(n=2) is called
**Then** returns the two violations with energies 0.9 and 0.8 (highest surprise)
**And** surprise_scores are sorted descending

### SCENARIO-SELFLEARN-022: SuReReplayResult.sure_better True When SuRe Improves Isolation

**Given** ``isolation_score_uniform=0.72``, ``isolation_score_sure=0.85``
**When** ``SuReReplayResult(n_violations_processed=57, n_replay_items=17, isolation_score_uniform=0.72, isolation_score_sure=0.85)``
**Then** ``sure_better=True`` (0.85 > 0.72)
**And** ``isolation_improvement=0.13`` (0.85 - 0.72)

### REQ-SELFLEARN-013: Tier 1 Relay Uses Real FP Patterns from Live Inference

The Tier 1 self-learning relay (FR-11) shall use real false-positive and
false-negative patterns extracted from live inference diagnostics (e.g.,
Exp 554 per_question_flags) as the seed for ConstraintAdditionFromMemory —
not synthetic data.  The relay artifact must carry ``inference_mode='real_data'``
and ``fr11_real_data=True``.  If the diagnostic source is unavailable, the
experiment must gate and report ``status='blocked'`` rather than substituting
synthetic data.

Spec: Exp 561, FR-11

### SCENARIO-SELFLEARN-013: Exp 554 FN Patterns Produce ViolationPattern for 'low_tp_extraction'

**Given** Exp 554 vericot_result has 17 FN cells and 0 FP cells
**When** ``load_exp554_fp_patterns(path)`` is called
**Then** returns a list containing one ViolationPattern with type='low_tp_extraction'
**And** ViolationPattern.count == 34 (17 FN from VeriCoT + 17 FN from VPRM)

### SCENARIO-SELFLEARN-014: Session 2 fp_rate_delta Reported Honestly

**Given** Session 1 fp_rate and Session 2 fp_rate are both computed on the same
25-response corpus with and without constraint_memory
**When** fp_rate_delta = session2_fp_rate - session1_fp_rate
**Then** honest_verdict is 'real_data_improvement' iff fp_rate_delta < -0.05
**And** honest_verdict is 'real_data_no_improvement' otherwise

### SCENARIO-SELFLEARN-015: Missing Exp 554 Diagnostic Gates the Experiment

**Given** Exp 554 result file does not exist
**When** load_exp554_fp_patterns() is called
**Then** returns an empty list
**And** the experiment sets status='blocked' and honest_verdict='blocked_no_real_data'

### REQ-LEARN-043: EnergyMagnitudeReplay Ranks Violations by |energy - session_mean|

EnergyMagnitudeReplay shall rank constraint violations by absolute energy deviation
|energy(x) - domain_mean_energy| for replay priority, ensuring domain boundary violations
(the examples the EBM is most wrong about) are always replayed first.

RETRO-050 root cause: SuRe used LLM NLL as surprise proxy (Exp 497, isolation=-0.1172).
LLM surface-form surprise and EBM energy magnitude are anticorrelated: common sentences
can violate hard constraints at high energy, and rare sentences can have low constraint
energy. Replacing LLM-surprise with energy-magnitude priority directly targets domain
boundary cases.

The EBM's energy function is the ground truth. High |energy - mean| = the constraint
model is maximally wrong about this boundary = highest replay value.

Spec: Exp 509, RETRO-050

### REQ-LEARN-044: EnergyMagnitudeBuffer Maintains Per-Domain Energy Buffer Sorted by Deviation

EnergyMagnitudeBuffer shall maintain a per-domain sorted buffer of at most max_size
violations, ordered by |energy - running_mean| descending. When the buffer is full, the
lowest-deviation item is evicted (not the oldest), ensuring the buffer always contains
the hardest domain boundary examples regardless of recency.

The running mean shall be maintained via Welford's online algorithm (O(1) per update,
unbiased over all observed energies, not just buffered ones).

Spec: Exp 509, RETRO-050

### REQ-LEARN-045: EnergyMagnitudeReplay.isolation_score Measures Domain Boundary Interference

EnergyMagnitudeReplay.isolation_score(domain_a, domain_b, n_steps) shall return a float
in [-1.0, 1.0] measuring how much replaying domain_a would interfere with domain_b's
learned constraint boundaries. Score 1.0 = perfect isolation. Score -1.0 = complete
interference.

The score is computed as 1.0 - 2.0 * change_rate, where change_rate is the fraction of
domain_a replay steps that share constraint key structure with domain_b's top-k violations.

Spec: Exp 509, RETRO-050

### SCENARIO-LEARN-071: EnergyMagnitudeBuffer top_k Returns Highest-Deviation Violations

**Given** violations added with energies [1.0, 5.0, 2.0, 4.0] to a buffer (mean ~3.0)
**When** top_k(2) is called
**Then** returns the 2 violations with highest |energy - mean| (energies 5.0 and 1.0)
**And** sorted highest deviation first

### SCENARIO-LEARN-072: EnergyMagnitudeReplay isolation_score in [-1, 1]

**Given** EnergyMagnitudeReplay with domains ['arithmetic', 'code'] and 20 violations added
**When** isolation_score('arithmetic', 'code', n_steps=20) is called
**Then** result is in [-1.0, 1.0]
**And** perfectly disjoint domains (no shared keys) return score=1.0

### SCENARIO-LEARN-073: EnergyMagnitudeReplay Beats SuRe Baseline (RETRO-050 Closure)

**Given** 200 simulated constraint violations across arithmetic/code/logical domains
**When** EnergyMagnitudeReplay.isolation_score('arithmetic', 'code', n_steps=50)
**Then** isolation_improvement = isolation_score - (-0.1172) > 0
**And** retro_050_closed=True
**And** honest_verdict='energy_magnitude_wins'

---

### REQ-LEARN-039: JEPA v4 Retrains on Live CoT Pairs from Exps 502-503 with Curriculum Ordering

JEPA v4 retraining (Exp 510) must load live CoT pairs from
`results/exp502_cot_pairs.json` and `results/exp503_cot_pairs.json` (when available)
and apply curriculum ordering: sort pairs by `label_confidence` descending so
high-confidence pairs are trained first.  If live files are absent, fall back to
100 synthetic pairs (ci_mode) and set `inference_mode='synthetic'`.

### REQ-LEARN-040: JEPA v4 Training Loss Includes Quasimetric Regularization Term

JEPA v4 training loss must include a quasimetric regularization term:

    L_quasimetric = lambda * max(0, d(conclusion, premise) - d(premise, conclusion))

where `d` is Euclidean distance, `lambda=0.1` (arXiv 2602.12245).  This penalizes
symmetric embedding distances to encode the directed structure of reasoning chains
(premise → conclusion is a one-way relationship).

### REQ-LEARN-041: JEPA v4 Saves Retrained Checkpoint to results/jepa_predictor_510_live.safetensors

Upon successful training, JEPA v4 must save the retrained model parameters as a
safetensors file at `results/jepa_predictor_510_live.safetensors`.  The artifact
must record `checkpoint_saved=True`.

### SCENARIO-LEARN-067: JEPA v4 Curriculum Ordering — High-Confidence Pairs First

**Given** a mixed set of CoT pairs with varying `label_confidence` values
**When** JEPA v4 training begins
**Then** pairs are sorted by `label_confidence` descending before training
**And** the first 200 training epochs use only the top-50% highest-confidence pairs
**And** the next 100 epochs use all pairs

### SCENARIO-LEARN-068: QuasimetricRegularizer Returns Zero Loss for Euclidean Symmetric Distances

**Given** a QuasimetricRegularizer with lambda=0.1
**When** loss(premise_emb, conclusion_emb) is called with standard numpy arrays
**Then** the loss is >= 0.0 (hinge lower bound)
**And** for symmetric Euclidean distance the loss is 0.0
**And** penalizes_symmetry is True

### SCENARIO-LEARN-069: JEPALiveRetrainResult Reports FR-11 Relay Status

**Given** a JEPALiveRetrainResult with post_auc=0.850 and inference_mode='live'
**When** target_met and auc_improvement are accessed
**Then** target_met is True (post_auc >= 0.800)
**And** auc_improvement = post_auc - pre_auc
**And** to_dict() contains all required schema fields

### REQ-RETRO-038: Milestone 2026.04.38 Operational Retrospective

The system shall produce a milestone retrospective artifact (schema v13) covering all 13
experiments in milestone 2026.04.38 (Exps 500-512) that:
- Reports retro closure rates for carry-forward items (RETRO-033, -038, -039, -048, -049, -050)
- Reports GPU utilization state at milestone close via pynvml
- Reports the five credibility milestone results: RETRO-048 resolved, RETRO-033 closed,
  RETRO-038 closed, RETRO-039 confirmed, GPU 1 utilization improved
- Adds new RETRO items for issues discovered in .38 experiments
- Sets credibility_milestone_reached = retro_033_closed OR retro_038_closed
- Writes result atomically via AtomicResultWriter to results/experiment_512_retro_2026_04_38.json

### SCENARIO-RETRO-038: Milestone .38 Retro Produces Schema v13 Artifact

**Given** Exp 500-511 result JSONs are present in the results/ directory
**When** scripts/experiment_512_retro_2026_04_38.py is executed
**Then** results/experiment_512_retro_2026_04_38.json is written with schema='carnot.operational_retro.v13'
**And** all five credibility fields are present: retro_048_resolved, retro_033_closed,
  retro_038_closed, retro_039_confirmed, gpu1_utilization_improved
**And** credibility_milestone_reached equals retro_033_closed OR retro_038_closed
**And** new_retro_items is a list with at least the carry-forward open items
**And** honest_verdict is 'milestone_complete'

### REQ-LEARN-046: LeWorldModel Two-Term Objective for JEPA Training

LeWorldModelJEPATrainer must use a two-term loss objective:

    L_total = L_prediction + λ * KL(q(z)||N(0,I))

where L_prediction is MSE between predicted and actual embeddings, and the KL term
is the analytical Gaussian KL divergence:

    KL = 0.5 * sum(exp(log_var) + mean^2 - 1 - log_var)

Default λ=0.01.  The KL term prevents embedding collapse (where the predictor maps
all inputs to the same point) by forcing the latent distribution to stay near N(0,I),
maintaining diversity in the embedding space.  This is the key stability trick from
arXiv 2603.19312 (LeWorldModel, 15M param model trained stably on single GPU).

### REQ-LEARN-047: LeWorldModel Training Stability — AUC Variance < 0.05

Three independent training runs with LeWorldModelLoss on the same 100-pair synthetic
corpus must produce AUC variance < 0.05 across runs.  This validates that the Gaussian
KL regularization has resolved the BCE collapse instability observed in Exps 472/510.

### SCENARIO-LEARN-074: gaussian_kl_regularization Returns Zero at Origin

**Given** gaussian_kl_regularization is called with z_mean=0 and z_log_var=0
**When** the KL divergence is computed
**Then** the return value is 0.0 (within float32 tolerance)
**And** this represents N(0,1) which is exactly N(0,I), so KL divergence is zero

### SCENARIO-LEARN-075: LeWorldModelLoss Total Is Non-Negative

**Given** a LeWorldModelLoss with any lambda_reg >= 0
**When** total_loss(predicted, actual, z_mean, z_log_var) is called with any inputs
**Then** the total loss is >= 0.0
**And** this holds because MSE >= 0 and KL divergence >= 0 by definition

### REQ-LEARN-048: JEPA Live Retrain v6 — LeWorldModel Two-Term Objective on Real Data

Exp 522 shall train the JEPA predictor using the LeWorldModel two-term objective
(REQ-LEARN-046) on real CoT pairs sourced from Exps 514-515 when available, falling
back to fover_labeled_steps_live.json (57 real labeled pairs from Exp 442) and finally
to 100 synthetic pairs.

The retrain shall:
- Record data_source as 'live_exp514_515', 'live_fover_442', or 'synthetic'
- Split pairs 80/20 train/test deterministically
- Train using LeWorldModelJEPATrainer with max_epochs=50
- Evaluate AUC on held-out test pairs
- Save checkpoint to results/jepa_predictor_522_live.safetensors
- Set fr11_live_relay=True when data_source != 'synthetic' and final_auc >= 0.800

FR-11 relay is achieved only when real data is used AND AUC >= 0.800.  Synthetic
fallback runs are labelled fr11_synthetic_only and do not count as a relay.

### SCENARIO-LEARN-076: load_cot_pairs_from_experiments Falls Back Gracefully

**Given** exp514_cot_pairs.json and exp515_cot_pairs.json do not exist on disk
**When** load_cot_pairs_from_experiments([514, 515], fallback_path) is called
**Then** the function loads from fallback_path (fover_labeled_steps_live.json)
**And** returns a non-empty list of ViolationPair objects
**And** each ViolationPair has non-empty partial_response and a valid has_violation bool

### SCENARIO-LEARN-077: compute_held_out_split Produces Correct Fractions

**Given** a list of N ViolationPair objects
**When** compute_held_out_split(pairs, test_fraction=0.2) is called
**Then** the returned (train, test) split sums to N
**And** len(test) == round(N * 0.2) (or floor, ensuring at least 1 test pair)
**And** the split is deterministic — calling twice with the same input yields the same split

### REQ-LEARN-049: JEPA Live Retrain v7 — LeWorldModel on Exps 527/528 Real Data

Exp 535 shall train the JEPA predictor using the LeWorldModel two-term objective on
real CoT pairs sourced from Exps 527/528 (live 100q/200q benchmarks) when available,
falling back to fover_labeled_steps_live.json (Exp 442) and exp514_cot_pairs.json,
and finally to 100 synthetic pairs.

The retrain shall:
- Implement load_v7_cot_corpus(preferred_paths, fallback_paths) returning (pairs, data_source)
  where data_source is 'live_exp527_528', 'live_fover_442', or 'synthetic'
- Implement summarize_corpus(pairs) returning n_pairs, n_correct, n_incorrect, source_breakdown
- Record data_source and corpus size in the artifact
- Split pairs 80/20 train/test deterministically
- Train using LeWorldModelJEPATrainer with max_epochs=50
- Evaluate AUC on held-out test pairs
- Save checkpoint to results/jepa_predictor_535_live.safetensors

### REQ-LEARN-050: JEPA Retrain v7 AUC Target >= 0.800 on Held-Out Real Pairs

When Exp 535 uses real CoT pairs (data_source != 'synthetic'), the final AUC on the
held-out test set shall be recorded honestly.  FR-11 relay is achieved when:
  fr11_live_relay = (data_source != 'synthetic') AND (final_auc >= 0.800)

Synthetic fallback runs are labelled fr11_synthetic_fallback and do not count as relay.
Partial relay (real data but AUC < 0.800) is labelled fr11_partial.

### SCENARIO-LEARN-078: load_v7_cot_corpus Prefers Exps 527/528 Over Fallback

**Given** exp527_cot_pairs.json or exp528_cot_pairs.json exist in results/
**When** load_v7_cot_corpus(preferred_paths=[527_path, 528_path], fallback_paths=[fover_path]) is called
**Then** data_source is 'live_exp527_528'
**And** the returned pairs are sourced from the preferred files
**And** summarize_corpus(pairs) returns n_pairs > 0

### SCENARIO-LEARN-079: load_v7_cot_corpus Falls Back When Preferred Files Absent

**Given** exp527_cot_pairs.json and exp528_cot_pairs.json do not exist on disk
**When** load_v7_cot_corpus(preferred_paths=[527_path, 528_path], fallback_paths=[fover_path]) is called
**Then** data_source is 'live_fover_442' (or 'synthetic' if fover also absent)
**And** the function never raises an exception
**And** the returned (pairs, data_source) tuple is always a valid result

### REQ-LEARN-051: GRPO Contrastive Pairing for EORM Retrain from Live Binary Verdicts

Exp 540 shall build contrastive (correct, incorrect) response pairs directly from live
benchmark binary verdicts (Exp 538 format), applying the GRPO insight (arXiv 2503.06639)
that verifiable binary rewards naturally induce contrastive pairs without additional labeling.

For each question in the benchmark:
- If baseline_correct=False and pipeline_correct=True: pair (correct=pipeline_response, incorrect=baseline_response)
- If baseline_correct=True and pipeline_correct=False: pair (correct=baseline_response, incorrect=pipeline_response)

When fewer than 5 pairs are extracted from the benchmark, fall back to FOVER
fover_labeled_steps_live.json (Exp 442 labels, 57 annotated steps).

The module shall expose:
- GRPOContrastivePair(question_id, correct_response, incorrect_response) dataclass
- build_grpo_pairs_from_benchmark(benchmark_result_path) → List[GRPOContrastivePair]
- train_eorm_grpo(eorm_model, pairs, margin, epochs, lr) → (training_loss, before_auc, after_auc)
- GRPOEORMRetrainResult(n_pairs, before_auc, after_auc, auc_improvement, honest_verdict)

### REQ-LEARN-052: GRPO EORM Contrastive Loss — Maximize Energy Gap

The contrastive loss used in Exp 540 EORM retrain shall be:
    L = mean(max(0, margin - (E(incorrect) - E(correct))))

This is equivalent to the EORM hinge loss rewritten to explicitly maximize the gap
(E(incorrect) - E(correct)).  The margin (default 1.0) ensures the model does not
stop learning once it achieves a minimal separation.

The loss is zero when E(incorrect) > E(correct) + margin for every pair in the batch,
indicating the model already discriminates correctly with sufficient confidence.

### SCENARIO-LEARN-080: build_grpo_pairs_from_benchmark Returns Empty When No Paired Fields

**Given** a benchmark JSON that has no per-question baseline_correct/pipeline_correct fields
**When** build_grpo_pairs_from_benchmark(benchmark_result_path) is called
**Then** it returns an empty list (no pairs, not an error)
**And** no exception is raised

### SCENARIO-LEARN-081: train_eorm_grpo Reduces Loss Over Epochs on Synthetic Pairs

**Given** 5 synthetic GRPOContrastivePair objects with distinct correct/incorrect responses
**When** train_eorm_grpo(model, pairs, margin=1.0, epochs=10, lr=1e-3) is called
**Then** the returned training_loss is a non-negative float
**And** before_auc and after_auc are in [0.0, 1.0]
**And** the model parameters are updated (not identical to initial)

### SCENARIO-LEARN-082: GRPOEORMRetrainResult honest_verdict Logic

**Given** a GRPOEORMRetrainResult with auc_improvement=0.10
**Then** honest_verdict is 'grpo_improved'
**Given** a GRPOEORMRetrainResult with auc_improvement=0.02
**Then** honest_verdict is 'no_improvement'
**Given** a GRPOEORMRetrainResult constructed from synthetic fallback pairs
**Then** honest_verdict is 'synthetic_fallback'

### REQ-LEARN-053: VerifyRepairPipeline Accepts Optional constraint_memory Parameter

``VerifyRepairPipeline.__init__()`` shall accept an optional
``constraint_memory: ConstraintAdditionFromMemory | None = None`` parameter.
When ``constraint_memory`` is provided, the pipeline integrates Tier 2
self-learning via the ConstraintAdditionFromMemory mechanism validated in Exp 456.
When ``None`` (the default), all existing behavior is unchanged — no new code
paths are exercised.

### REQ-LEARN-054: VerifyRepairPipeline Records Violations into constraint_memory and Applies Learned Constraints

After each ``verify()`` call that produces violations, the pipeline shall call
``self._constraint_memory.observe(violation_type, step_text)`` for each violation,
where ``violation_type`` is the leading token of the constraint_type field (split
on ``':'``).  Before running constraint evaluation, if ``constraint_memory`` is not
None, the pipeline shall call ``constraint_memory.check_and_add(self)`` and prepend
any newly added constraint names as synthetic ConstraintResult objects to the active
constraint set.  After check_and_add() is called, if the pattern count for any
violation type exceeds the threshold, new constraints are active for subsequent calls.

### SCENARIO-LEARN-083: Pipeline with constraint_memory=None Uses Existing Behavior

**Given** a VerifyRepairPipeline constructed without constraint_memory
**When** verify() is called with a response that has violations
**Then** the result is identical to the pre-wire-in behavior
**And** no ConstraintAdditionFromMemory method is called

### SCENARIO-LEARN-084: Pipeline Records Violations into constraint_memory

**Given** a VerifyRepairPipeline constructed with constraint_memory set
**When** verify() is called and violations are found
**Then** constraint_memory.observe() is called once per violation
**And** the violation_type passed to observe() is the constraint_type prefix

### SCENARIO-LEARN-085: Pipeline Applies Learned Constraints on Subsequent Calls

**Given** a VerifyRepairPipeline with constraint_memory whose pattern count
for 'carry' has been pre-seeded above the threshold
**When** verify() is called
**Then** check_and_add() promotes the pattern to an active constraint
**And** the pipeline applies the new constraint during evaluation

### REQ-LEARN-055: FOVER Corpus Expansion via Multi-Source Merge

**Why this requirement exists:**
JEPA v7 was retrained on 57 real FOVER pairs (fover_442 source, n_train_pairs=46,
n_test_pairs=11). Exp 538 produced 25 additional CoT responses (50 total when two
models each answer 25 questions). Merging these sources with deduplication by
step_text hash produces a 100+ pair corpus for JEPA v8 retrain (Exp 543).

**Given** a FOVER corpus from fover_labeled_steps_live.json (57 pairs)
**And** new CoT responses from exp538_cot_pairs.json (25 responses)
**When** Exp 542 runs FOVERAnnotator on the new responses and merges with prior pairs
**Then** the merged corpus is deduplicated by SHA-256 hash of step_text
**And** the merged corpus is written to results/fover_labeled_steps_expanded.json
**And** honest_verdict is 'corpus_expanded' when n_total_pairs >= 100

### SCENARIO-LEARN-086: Deduplication Removes Pairs with Identical step_text

**Given** two training pairs with identical step_text content
**When** merge_fover_corpora() combines them
**Then** only one copy is retained in the merged output

### SCENARIO-LEARN-087: Multi-Source Load Tolerates Missing exp538 File

**Given** exp538_cot_pairs.json is absent from the results directory
**When** Exp 542 runs
**Then** n_new_pairs = 0 and the merged corpus equals the prior 57-pair corpus
**And** honest_verdict reflects partial_expansion

### REQ-LEARN-056: JEPA v8 Retrain on Expanded FOVER Corpus

**Why this requirement exists:**
JEPA v7 (Exp 535) achieved AUC=0.967 on 57 real FOVER pairs. Exp 542 expanded the corpus
via multi-source merge. Exp 543 retrains JEPA v8 on this expanded corpus to validate that
the larger dataset maintains or improves AUC, satisfying FR-11 with honest live data.

**Given** fover_labeled_steps_expanded.json exists (from Exp 542)
**When** Exp 543 runs JEPA v8 retrain
**Then** the model trains on the expanded corpus with 80/20 train/test split
**And** the artifact contains n_train_pairs, n_test_pairs, final_auc, auc_improvement
**And** honest_verdict is set based on AUC threshold and n_train_pairs

### REQ-LEARN-057: LeWorldModel Two-Term Objective for JEPA v8

**Why this requirement exists:**
The LeWorldModel objective (L_total = L_prediction + lambda * L_KL) provides stable Gaussian
regularization that prevents embedding collapse during training. JEPA v8 uses lambda=0.1
(vs 0.01 in v6/v7) to apply stronger regularization on the expanded corpus.

**Given** a JEPA predictor and training pairs
**When** LeWorldModelJEPATrainer trains for 100 epochs with lambda_reg=0.1
**Then** L_KL = KL(q(z) || N(0,I)) = 0.5 * sum(mu^2 + sigma^2 - log(sigma^2) - 1)
**And** L_total = L_prediction + 0.1 * L_KL per batch

### SCENARIO-LEARN-088: JEPA v8 AUC on Expanded Corpus Meets Threshold

**Given** JEPA v8 trained on the expanded FOVER corpus
**When** final_auc is evaluated on the held-out test split
**Then** honest_verdict is 'jepa_v8_improved' when auc>=0.900 AND n_train>=80
**Or** honest_verdict is 'auc_stable' when auc>=0.800
**Or** honest_verdict is 'synthetic_fallback' otherwise

### SCENARIO-LEARN-089: JEPA v8 Falls Back to fover_labeled_steps_live.json

**Given** fover_labeled_steps_expanded.json is absent from results/
**When** Exp 543 runs
**Then** the experiment loads fover_labeled_steps_live.json as fallback
**And** the artifact correctly reflects the actual data source used

---

### REQ-LEARN-058: Offline Distillation of Violation Patterns into Named Constraint Templates

The system SHALL provide a ConstraintTemplateStore that accumulates violation
observations (violation_type, context) and distils mature patterns into named
ConstraintTemplate objects.  A pattern is mature when its observation count
reaches min_observations (default 3).  The store MUST be serialisable to JSON
and rehydratable across sessions.

Implemented by: python/carnot/pipeline/constraint_template_store.py

### REQ-LEARN-059: Template Retrieval at Inference Time via String Similarity

The ConstraintTemplateStore SHALL provide a retrieve(query_context, top_k)
method that returns up to top_k distilled templates ranked by keyword overlap
score — the count of template context_keywords that appear as substrings in
the lowercased query_context.  This allows the pipeline to apply domain-
specific constraints only when the current query context matches a template.

Implemented by: python/carnot/pipeline/constraint_template_store.py

### SCENARIO-LEARN-090: add_violation Accumulates Observations and Distill Promotes Mature Patterns

**Given** a ConstraintTemplateStore with min_observations=3
**When** add_violation("carry", ctx) is called 3 or more times
**Then** distill(min_observations=3) returns a ConstraintTemplate with violation_type="carry"
**And** the template's n_violations_observed equals the total observation count
**And** context_keywords contains tokens extracted from the context strings

### SCENARIO-LEARN-091: Distill Threshold Filters Immature Patterns

**Given** a ConstraintTemplateStore where "sign" has been observed 2 times and "carry" 4 times
**When** distill(min_observations=3) is called
**Then** the result contains the "carry" template
**And** the result does NOT contain the "sign" template (below threshold)

### SCENARIO-LEARN-092: Retrieve Returns Top-K Templates by Keyword Overlap

**Given** a distilled store with carry_guard (keywords: arithmetic, carry) and semantic_guard
**When** retrieve("arithmetic carry overflow", top_k=3) is called
**Then** carry_guard is ranked first (highest keyword overlap)
**And** the result contains at most top_k templates

---

### REQ-LEARN-060: EORM Retrain on Real GRPO Contrastive Pairs from FOVER Corpus v2

**Statement:**
    When fover_corpus_v2.json contains at least 100 entries with is_correct labels,
    the experiment pipeline SHALL retrain EORM using GRPO-style contrastive pairs
    (arXiv 2503.06639) formed from same-question (correct, incorrect) response pairs,
    report AUC on a held-out 20% split, and save the retrained model as
    results/eorm_model_556_real.safetensors.

**Acceptance criteria:**
    - load_fover_corpus_v2() loads entries from fover_corpus_v2.json and groups by question
    - Contrastive triples built from questions that have both correct and incorrect responses
    - n_contrastive_triples >= 1 when corpus has at least one question with both correct
      and incorrect responses
    - AUC reported before and after retraining on 20% held-out split
    - honest_verdict='real_data_improvement' iff after_auc >= 0.700
    - Blocked artifact emitted if n_pairs < 100

Spec: REQ-LEARN-060

---

### SCENARIO-LEARN-093: load_fover_corpus_v2 Returns GRPOContrastivePairs from Both-Verdict Questions

**Given** fover_corpus_v2.json with entries including at least one question with
         both a correct and an incorrect response
**When** load_fover_corpus_v2(path) is called
**Then** the returned list contains one GRPOContrastivePair per such question
**And** correct_response and incorrect_response fields are non-empty strings

---

### SCENARIO-LEARN-094: load_fover_corpus_v2 Returns Empty List for Missing File

**Given** a file path that does not exist
**When** load_fover_corpus_v2(path) is called
**Then** an empty list is returned without raising an exception

---

### REQ-LEARN-061: JEPAPUREMinForm — PURE Min-Form Contrastive Margin Loss for JEPA Training

**Motivation (RETRO-060):** Binary BCE loss lets the JEPA model hedge toward P=0.5 everywhere,
producing near-zero gradient and AUC=0.4286 — below random — for the second consecutive retrain
(Exp 557).  The root cause is that BCE does not enforce a contrastive margin between correct and
incorrect chain scores.

**Fix (arXiv 2504.15275, PURE PRM):** Score a chain by the MINIMUM step score across all its
steps.  A chain with one bad step gets a strong low signal; a chain with all good steps gets a
strong high signal.  The contrastive loss is:

    loss = mean(max(0, margin - (min_score_incorrect - min_score_correct)))

over all (correct, incorrect) pairs from the same question.  This enforces a hard margin gap of
at least `margin` between the minimum step scores of incorrect and correct chains — exactly the
signal that NUP Probe v4 used to achieve AUC=1.0.

**Requirements:**
- JEPAChainScore dataclass: chain_id (str), step_scores (list[float]), min_score (float), is_correct (bool).
- PUREMinFormLoss class with compute_chain_scores, compute_loss, zero_if_empty methods.
- pairs_to_pure_chains helper: groups FOVERCorpusEntry list by question_id, embeds steps, returns (correct_chains, incorrect_chains).

Spec: REQ-LEARN-061

---

### REQ-LEARN-062: PUREMinFormLoss Margin Parameter Is Configurable

The contrastive margin used by PUREMinFormLoss must be exposed as a constructor parameter
`margin: float = 1.0`.  Callers may override it without subclassing.

Spec: REQ-LEARN-062

---

### SCENARIO-LEARN-095: PUREMinFormLoss Returns Positive Loss When Incorrect Score > Correct Score But Gap < Margin

**Given** a correct chain with min_score=0.3 and an incorrect chain with min_score=0.7 and margin=1.0
**When** compute_loss is called
**Then** loss = max(0, 1.0 - (0.7 - 0.3)) = 0.6 > 0

---

### SCENARIO-LEARN-096: PUREMinFormLoss Returns Zero When Gap Exceeds Margin

**Given** a correct chain with min_score=0.0 and an incorrect chain with min_score=1.5 and margin=1.0
**When** compute_loss is called
**Then** loss = max(0, 1.0 - (1.5 - 0.0)) = 0.0

---

### SCENARIO-LEARN-097: PUREMinFormLoss.zero_if_empty Returns 0.0 for Empty Pair List

**Given** an empty list of pairs
**When** zero_if_empty is called
**Then** 0.0 is returned without error

---

### REQ-LEARN-063: JEPA v10 Full Retrain with PUREMinFormLoss on 132-Pair FOVER Corpus

**What:** Full retrain of the JEPA chain scorer (v10) on the complete 132-pair FOVER corpus v2
using PUREMinFormLoss (REQ-LEARN-061) as the training objective. This is the mandatory FR-11
JEPA retrain that follows every milestone and the direct response to RETRO-060.

**Why (layman):** Exps 543 and 557 both produced AUC < 0.5 (worse than random) because they
used binary cross-entropy loss which lets the model hedge toward 0.5 everywhere. Exp 566 showed
that PUREMinFormLoss can push AUC above random. This requirement mandates a full 200-epoch retrain
with the new objective and saves the best checkpoint for downstream use.

**Acceptance criteria:**
- Model trained for 200 epochs on 80% split of 132-pair corpus.
- Val AUC evaluated every 20 epochs; best checkpoint saved.
- Final artifact reports v9_auc=0.4286, v10_auc, auc_improvement, retro_060_resolved flag.
- Model saved to results/jepa_predictor_v10.safetensors.
- fr11_retrain_complete=True always set in artifact.

Spec: REQ-LEARN-063

---

### SCENARIO-LEARN-081: JEPA v10 Retrain Produces AUC Above Random Baseline

**Given** the 132-pair FOVER corpus v2 loaded and split 80/20
**When** JEPA v10 is trained for 200 epochs with PUREMinFormLoss(margin=1.0)
**Then** the best validation AUC > 0.5 (above random) and retro_060_resolved=True

---

### SCENARIO-LEARN-082: JEPA v10 Retrain Saves Model Checkpoint

**Given** training completes successfully
**When** the best val AUC epoch is identified
**Then** model weights are saved to results/jepa_predictor_v10.safetensors via AtomicResultWriter

---

### SCENARIO-LEARN-083: JEPA v10 Artifact Contains All Required Fields

**Given** training completes (success or partial)
**When** the result artifact is built
**Then** it contains schema, n_train, n_val, loss_function, v9_auc, v10_auc, auc_improvement,
best_epoch, model_path, retro_060_resolved, fr11_retrain_complete, honest_verdict

---

### REQ-LEARN-064: Hardware Energy as EORM Calibration Signal via RAPL Power Trace Correlation

**What:** Measure Pearson correlation between hardware energy expenditure (RAPL joules/step
on CPU) and EORM model energy scores during inference over a set of CoT steps.

**Why (layman):** Hard reasoning steps require more compute cycles — the CPU draws more
power to compute large activations and long attention. Carnot's EORM is a learned proxy
for reasoning quality. If RAPL hardware energy and EORM energy co-move (r > 0.5), then
a simple power meter becomes a free, label-free calibration signal: we can align EORM
scores to physical energy without any human annotation.

**Acceptance criteria:**
- HardwareEnergyProbe reads from /sys/class/powercap/intel-rapl:0/energy_uj when available.
- Falls back to pyRAPL, then to a mock(0.0) source with source label.
- compute_eorm_hardware_correlation() runs EORM scoring + hardware measurement for each step.
- Pearson r and p-value computed via scipy.stats.pearsonr.
- calibration_viable = r > 0.5 AND p_value < 0.05.
- Artifact reports rapl_available, pearson_r, p_value, calibration_viable, honest_verdict.

Spec: REQ-LEARN-064,
      SCENARIO-LEARN-098, SCENARIO-LEARN-099, SCENARIO-LEARN-100

---

### SCENARIO-LEARN-098: HardwareEnergyProbe Falls Back to Mock When RAPL Unavailable

**Given** the system does not have /sys/class/powercap/intel-rapl:0/energy_uj
**When** HardwareEnergyProbe.read() is called
**Then** it returns a HardwareEnergyReading with source='mock' and joules=0.0

---

### SCENARIO-LEARN-099: measure_segment Returns Result and Delta Joules

**Given** a callable function fn and a HardwareEnergyProbe instance
**When** probe.measure_segment(fn) is called
**Then** it returns a tuple (result, delta_joules) where result is fn()'s return value

---

### SCENARIO-LEARN-100: Correlation Artifact Contains All Required Fields

**Given** 30 CoT steps (15 correct, 15 incorrect) from FOVER corpus v2
**When** compute_eorm_hardware_correlation() runs with mock probe
**Then** EORMHardwareCorrelation has n_steps=30, pearson_r, p_value, hardware_energies,
eorm_energies, calibration_viable; and artifact has honest_verdict set correctly

---

### REQ-LEARN-065: JEPACPMIPairBuilder — Hard-Negative Contrastive Pair Construction from FOVER Corpus

**Why (layman):** The JEPA predictor AUC was stuck at 0.4444 across three retrains because
all three used scalar loss on step-level labels — the model could hedge by making all scores
near 0.5.  The CPMI fix (arXiv 2604.10660) constructs explicit contrastive pairs: one
(correct_chain, incorrect_chain) from the SAME question.  This forces the model to learn a
RELATIVE ordering between two whole chains, which is exactly what AUC measures.

**Acceptance criteria:**
- JEPACPMIPairBuilder.build_pairs() groups FOVERCorpusEntry objects by question_id.
- For each group with >= 1 correct AND >= 1 incorrect entry: yields one JEPACPMIPair.
- Hardest incorrect = entry with most cot_steps (hard-negative mining).
- JEPACPMIPair has: question_id, correct_embeddings, incorrect_embeddings,
  hard_negative_step_idx, pair_quality.
- pair_quality = len(correct_steps) / max(len(incorrect_steps), 1).
- build_synthetic_pairs(n) returns exactly n pairs as fallback for small corpora.

Spec: REQ-LEARN-065,
      SCENARIO-LEARN-101, SCENARIO-LEARN-103

---

### REQ-LEARN-066: CPMIContrastiveLoss — Hinge Margin Loss for JEPA CPMI Pairs

**Why (layman):** The contrastive hinge margin loss directly optimises the ranking
objective that AUC measures.  For each pair it computes:
    L = max(0, margin - (E_incorrect - E_correct))
This is zero when the incorrect chain scores at least `margin` above the correct chain,
and positive otherwise.  Unlike BCE or PURE, the gradient of this loss is always zero
when the constraint is satisfied — no gradient flows when the model is already correct.

**Acceptance criteria:**
- CPMIContrastiveLoss.__init__(margin, chain_energy_mode) with mode in ('mean','max','min').
- chain_energy(model, embeddings) aggregates per-step scores by the selected mode.
- compute_loss(model, pairs) returns mean(max(0, margin - gap)) over all pairs.
- Returns 0.0 for empty pair list (zero_if_empty guard).
- chain_energy returns 0.0 for empty embeddings.

Spec: REQ-LEARN-066,
      SCENARIO-LEARN-101, SCENARIO-LEARN-102, SCENARIO-LEARN-103

---

### SCENARIO-LEARN-101: build_pairs Yields One Pair Per Question With Both Verdicts

**Given** a FOVER corpus with questions that each have at least one correct and one incorrect entry
**When** JEPACPMIPairBuilder.build_pairs() is called
**Then** it yields exactly one JEPACPMIPair per qualifying question; questions with only
correct or only incorrect entries are skipped

---

### SCENARIO-LEARN-102: CPMIContrastiveLoss Returns Zero When Gap Exceeds Margin

**Given** a pair where E_incorrect - E_correct = 5.0 and margin = 1.0
**When** CPMIContrastiveLoss.compute_loss() is called
**Then** it returns 0.0 (constraint already satisfied; no gradient needed)

---

### SCENARIO-LEARN-103: build_synthetic_pairs Pads Small Corpus to Requested Count

**Given** a corpus that yields fewer than min_pairs real pairs
**When** JEPACPMIPairBuilder.build_synthetic_pairs(n) is called
**Then** it returns exactly n JEPACPMIPair objects with pair_quality=1.0 and
correct arithmetic in the correct chain, one off-by-one error in the incorrect chain

---

### REQ-LEARN-067: JEPA v11 Full Retrain with CPMIContrastiveLoss on FOVER Corpus

**Summary:** Mandatory FR-11 retrain using CPMI contrastive objective to resolve RETRO-063.

**Motivation:** JEPA v8/v9/v10 all trained on scalar loss with step-level labels, producing
AUC <= 0.4444 (below random). RETRO-063 identified the root cause: step-level labels are
noisy, and per-chain objectives allow hedging to P=0.5. This retrain uses CPMIContrastiveLoss
(pairwise ordering constraint) which forces the model to directly rank correct chains above
incorrect chains for the same question — the same mechanism that yielded AUC=1.0 in NUP Probe v4.

**Requirements:**
- SHALL load FOVER corpus v3 if available, else v2 (132 pairs minimum)
- SHALL build contrastive pairs with JEPACPMIPairBuilder; augment with synthetic pairs if < 5 real
- SHALL split train/val 80/20 by question_id to prevent cross-question leakage
- SHALL train 300 epochs with optax.adamw(lr=1e-3, weight_decay=1e-4)
- SHALL use CPMIContrastiveLoss(margin=1.0, chain_energy_mode='mean')
- SHALL save best checkpoint to results/jepa_predictor_v11.safetensors
- SHALL report retro_063_resolved = True if v11_auc > 0.5
- SHALL emit honest_verdict: 'jepa_v11_above_random' | 'jepa_v11_still_inverted' | 'jepa_v11_at_random'

Spec: REQ-LEARN-067,
      SCENARIO-LEARN-104, SCENARIO-LEARN-105, SCENARIO-LEARN-106

---

### SCENARIO-LEARN-104: JEPA v11 Retrain Produces Valid Artifact With All Required Fields

**Given** a FOVER corpus with at least 5 questions having both correct and incorrect entries
**When** experiment_580_jepa_v11_retrain runs to completion
**Then** the artifact contains all required schema fields including v10_auc, v11_auc,
auc_improvement, best_epoch, n_real_pairs, n_synthetic_pairs, retro_063_resolved,
fr11_retrain_complete=True, and honest_verdict

---

### SCENARIO-LEARN-105: JEPA v11 Question-ID Split Prevents Cross-Question Leakage

**Given** a FOVER corpus where multiple entries share the same question
**When** the 80/20 train/val split is applied by question_id
**Then** no question_id appears in both the train and val pair sets

---

### SCENARIO-LEARN-106: JEPA v11 Synthetic Pair Augmentation Kicks In For Small Corpus

**Given** a corpus that yields fewer than 5 real CPMI pairs
**When** experiment_580_jepa_v11_retrain runs
**Then** build_synthetic_pairs(20) is called and n_synthetic_pairs >= 20 in the artifact

---

### REQ-LEARN-068: JEPA v12 — Live Corpus CPMI Retrain with >= 20 Real Pairs (Exp 593)

JEPA v12 must be trained on the full live corpus from Exp 578 (GSM8K 0-49, both
Qwen3.5-0.8B and Gemma4-E4B-it, inference_mode='live_gpu') using the CPMIContrastiveLoss
objective.  The training corpus must contain at least 20 real live pairs (pairs derived
from actual GPU inference, not synthetic fallback).  Validates whether v11 AUC=1.0 on
9 pairs was genuine learning or overfitting.

Spec: REQ-LEARN-068

### REQ-LEARN-069: PROGRSCentering — Outcome-Conditioned Group Centering for Contrastive Stability (arXiv 2604.02341)

PROGRSCentering must implement PROGRS outcome-conditioned group centering to prevent
reward hacking on easy questions.  For each question_id group, the mean energy gap is
subtracted from each pair's energy gap before computing the contrastive margin loss.
This ensures every question group contributes an equally-weighted learning signal
regardless of absolute gap magnitude.  For single-pair groups, a graceful fallback
to raw gap must be used to preserve gradient flow.

Spec: REQ-LEARN-069

### SCENARIO-LEARN-107: PROGRSCentering.center_pairs Correctly Normalises Within-Group Gaps

**Given** a list of JEPACPMIPair objects spanning multiple question_id groups
**When** PROGRSCentering.center_pairs() is called
**Then** the centered gap for each pair equals raw_gap minus the group mean gap,
         the output list preserves input order, and empty input returns []

Spec: SCENARIO-LEARN-107

### SCENARIO-LEARN-108: PROGRSCentering.compute_centered_loss Applies Hinge on Centered Gaps

**Given** a list of pairs where some groups have multiple pairs (different raw gaps)
**When** PROGRSCentering.compute_centered_loss(model, pairs, margin) is called
**Then** the loss equals mean(max(0, margin - centered_gap_p)) over all pairs,
         where centered gaps account for within-group mean subtraction

Spec: SCENARIO-LEARN-108

### SCENARIO-LEARN-109: Single-Pair Groups Yield Centered Gap = 0 in PROGRSCentering

**Given** a corpus where every question_id has exactly one pair (output of JEPACPMIPairBuilder)
**When** PROGRSCentering.center_pairs() is called
**Then** centered_gap = 0 for every pair (raw_gap - group_mean = raw_gap - raw_gap = 0),
         and compute_centered_loss returns margin for each pair

Spec: SCENARIO-LEARN-109

---

### REQ-LEARN-070: FR-11 Real Violations Relay from Live Verify-Repair (Exp 597)

The system shall attempt to source real constraint violations from live verify-repair runs
(Exp 594 CoACEv3, Exp 595 DSVD) and feed them into ConstraintAdditionFromMemory.
When live violations are unavailable (both upstream experiments blocked), synthetic
violations shall be used as a clearly-labeled fallback.  The artifact must always record:
- n_live_violations (int): count of violations from live GPU inference.
- violations_source: one of 'exp594', 'exp595', 'synthetic_fallback'.
- fr11_real_violations_confirmed (bool): True iff n_live_violations >= 5.
- honest_verdict: 'real_violations_improved', 'real_violations_no_improvement',
  or 'synthetic_fallback'.

**Rationale:** FR-11 requires self-learning to operate on real data.  When live runs are
blocked, synthetic data preserves pipeline exercise without misleading headline claims.

Spec: REQ-LEARN-070,
      SCENARIO-LEARN-110, SCENARIO-LEARN-111, SCENARIO-LEARN-112

---

### SCENARIO-LEARN-110: Violations Source Recorded in Artifact

**Given** Exp 594 and Exp 595 are both blocked (n_violations=0)
**When** Exp 597 runs
**Then** violations_source='synthetic_fallback', fr11_real_violations_confirmed=False,
         honest_verdict='synthetic_fallback', and all required schema fields are present.

Spec: SCENARIO-LEARN-110

---

### REQ-LEARN-071: MISE Backward Inference for EORM Reward Calibration (arXiv 2604.11611)

The system shall implement MISECalibrator with backward_inference_score() computing
cosine similarity between embed(response) and embed(question) as a proxy for how well
the response answers the question.  The calibrate() method computes:
- mean_alignment_correct: mean score for verdict_correct=True triples.
- mean_alignment_incorrect: mean score for verdict_correct=False triples.
- calibration_gap: mean_alignment_correct - mean_alignment_incorrect.
A positive calibration_gap indicates the alignment signal reliably separates correct
from incorrect responses and can be used as a dense EORM reward signal.

Spec: REQ-LEARN-071,
      SCENARIO-LEARN-111, SCENARIO-LEARN-112

---

### SCENARIO-LEARN-111: backward_inference_score Returns Float in [-1, 1]

**Given** any two non-empty strings
**When** MISECalibrator.backward_inference_score(response, question) is called
**Then** a float in [-1.0, 1.0] is returned; zero-norm embeddings return 0.0.

Spec: SCENARIO-LEARN-111

---

### SCENARIO-LEARN-112: calibrate() Returns Three-Key Dict with calibration_gap

**Given** a list of MISETriple objects (may be empty)
**When** MISECalibrator.calibrate(triples) is called
**Then** a dict with keys 'mean_alignment_correct', 'mean_alignment_incorrect',
         'calibration_gap' is returned; empty input yields all-zero values;
         calibration_gap = mean_alignment_correct - mean_alignment_incorrect.

Spec: SCENARIO-LEARN-112

### REQ-LEARN-072: HISR — Hindsight Importance Score Reweighting for Constraint Violations (arXiv 2603.18683) (Exp 598)

Applies credit-assignment from HISR paper to ConstraintAdditionFromMemory:
violations that temporally preceded a final incorrect answer receive higher
hindsight scores, focusing constraint-addition signal on causally relevant events.

- REQ-LEARN-072-1: ``HISRViolationWeight`` SHALL store violation_type, question_id,
  final_incorrect (bool), and hindsight_score (float in [0, 1]).
- REQ-LEARN-072-2: ``HISRWeighter.compute_hindsight_score(violations, final_correct)``
  SHALL assign score = 0.0 to all violations when final_correct=True.
- REQ-LEARN-072-3: When final_correct=False, the last violation SHALL receive score 1.0
  and earlier violations SHALL receive score = 1/(1 + distance_from_last).
- REQ-LEARN-072-4: ``HISRWeighter.weighted_violations(weights, threshold=0.5)``
  SHALL return ViolationPattern objects only for weights with hindsight_score >= threshold.

Spec: REQ-LEARN-072, SCENARIO-LEARN-113, SCENARIO-LEARN-114

### SCENARIO-LEARN-113: Violations in Correct Chains Score Zero (HISR False-Positive Suppression)

**Given** a list of ViolationPattern objects from a chain where final_correct=True
**When** HISRWeighter.compute_hindsight_score(violations, final_correct=True) is called
**Then** every HISRViolationWeight has hindsight_score == 0.0 and final_incorrect == False.

Spec: SCENARIO-LEARN-113

### SCENARIO-LEARN-114: Violations in Incorrect Chains Score Higher Near the Final Error

**Given** a list of 5+ ViolationPattern objects from a chain where final_correct=False
**When** HISRWeighter.compute_hindsight_score(violations, final_correct=False) is called
**Then** scores are strictly increasing (earliest < latest), and the last score is 1.0.

Spec: SCENARIO-LEARN-114

---

### REQ-LEARN-073: JEPA v12 OOD Generalization Validation on Held-Out GSM8K Questions (Exp 607)

The system shall validate JEPA v12's generalization by evaluating it on strictly held-out
GSM8K questions never seen during training (question indices >= 50, sourced from
live_pairs_602.json GSM8K 250-349).  The artifact must record:
- n_ood_pairs (int): number of OOD entries used for evaluation.
- v12_val_auc (float): 1.0 from Exp 593 (in-distribution).
- v12_ood_auc (float): AUC on the held-out OOD set.
- v12_generalized (bool): True iff v12_ood_auc >= 0.65.
- v12_overfit (bool): True iff v12_ood_auc < 0.55.
- fr11_generalization_confirmed (bool): True iff v12 generalized OR v13 retrain
  achieves val_auc >= 0.65.
- honest_verdict: one of 'v12_generalized', 'v12_overfit_v13_saved', 'jepa_fails_ood'.

**Rationale:** AUC=1.0 on 20 in-distribution pairs is insufficient evidence of generalization.
FR-11 claims the JEPA relay is validated only when OOD performance is confirmed.

### REQ-LEARN-074: JEPA v13 Corpus-v4 Retrain When v12 Is Overfit (Exp 607 conditional)

If JEPA v12 is overfit (ood_auc < 0.55), the system shall retrain JEPA v13 on the full
fover_corpus_v4.json corpus using the same CPMI+PROGRS architecture (100 epochs, 80/20
split).  v13 is saved to results/jepa_predictor_v13.safetensors iff val_auc >= 0.65.
The artifact must record v13_retrained (bool) and v13_val_auc (float | None).

Spec: REQ-LEARN-073, REQ-LEARN-074,
      SCENARIO-LEARN-115, SCENARIO-LEARN-116, SCENARIO-LEARN-117

### SCENARIO-LEARN-115: v12 Generalizes When OOD AUC Exceeds Threshold

**Given** a loaded jepa_predictor_v12.safetensors and OOD entries from live_pairs_602.json
**When** scores are computed for each entry and roc_auc_score is called
**Then** if ood_auc >= 0.65: v12_generalized=True, fr11_generalization_confirmed=True,
         honest_verdict='v12_generalized'.

Spec: SCENARIO-LEARN-115

### SCENARIO-LEARN-116: v12 Overfit Triggers v13 Retrain

**Given** v12_ood_auc < 0.55 (overfit detected)
**When** retrain_jepa_v13() is called with fover_corpus_v4.json entries
**Then** v13_retrained=True, and if v13_val_auc >= 0.65: fr11_generalization_confirmed=True,
         honest_verdict='v12_overfit_v13_saved'; else honest_verdict='jepa_fails_ood'.

Spec: SCENARIO-LEARN-116

### SCENARIO-LEARN-117: Blocked When v12 Model File Is Absent

**Given** results/jepa_predictor_v12.safetensors does not exist
**When** Exp 607 is run
**Then** the artifact records model_available=False, status='blocked_no_v12_model',
         and the script exits cleanly without crashing.

Spec: SCENARIO-LEARN-117

---

### REQ-LEARN-076: FLIP Backward Inference for Repair Quality Scoring (FR-11, arXiv 2602.13551)

**Requirement:**
    The system MUST implement FLIPRewardCalibrator, which uses backward inference
    (cosine similarity between embed(repaired) and embed(question)) to score
    whether a repair improved constraint alignment relative to the original response.
    A positive alignment delta means the repair moved the response closer to the
    question's constraint space.  A zero or negative delta signals a potentially
    constraint-inconsistent repair.

**Rationale:**
    Binary verify-repair verdicts are weak FR-11 signals.  FLIP provides a dense
    signal by measuring whether each repair improved the model's constraint alignment
    — without requiring human labels.

Spec: REQ-LEARN-076, SCENARIO-LEARN-118, SCENARIO-LEARN-119, SCENARIO-LEARN-120

### REQ-LEARN-077: FR-11 Real Violations Relay v5 from Exp 609 or Synthetic Fallback

**Requirement:**
    Experiment 611 MUST load live violations from experiment_609 (n_violations_found > 0)
    as the primary source.  If Exp 609 has zero violations, fall back to Exp 597.
    If both are empty, use 10 synthetic violations from fover_corpus_v4.json, labeled
    explicitly as synthetic_fallback.  The violations_source field in the artifact MUST
    record which path was taken.

Spec: REQ-LEARN-077, SCENARIO-LEARN-118

---

### SCENARIO-LEARN-118: FLIP Calibration Runs on Live or Synthetic Violations

**Given** experiment 611 loads violations from Exp 609, Exp 597, or synthetic fallback,

**When** FLIPRewardCalibrator.batch_calibrate() is called with the resulting triples,

**Then** the artifact records flip_mean_score (float), flip_n_improved (int),
         flip_repair_quality ('good'|'neutral'|'bad'), and violations_source.

Spec: SCENARIO-LEARN-118

### SCENARIO-LEARN-119: backward_inference_score Returns Float in [0, 1] for Hash Embed

**Given** FLIPRewardCalibrator is constructed with the hash-projection embed_fn,

**When** backward_inference_score(response, question) is called with non-empty strings,

**Then** the return value is a float in [-1.0, 1.0] (cosine similarity range) and
         does not raise an exception.

Spec: SCENARIO-LEARN-119

### SCENARIO-LEARN-120: batch_calibrate Reports 'good' When Majority Repairs Improve Alignment

**Given** a list of FLIPRepairTriples where more than half have repaired responses
         that score higher alignment than the originals,

**When** batch_calibrate() is called,

**Then** repair_quality = 'good', n_improved > 0, and mean_flip_score > 0.

Spec: SCENARIO-LEARN-120

---

### REQ-LEARN-078: MetaJuLS Extractor Policy Adapts from Live Batch Feedback

**Requirement:** After LLMAsExtractorV1 processes a batch of live outputs, a
MetaJuLS-style meta-RL update must adjust the extractor's policy parameters
(temperature, claim_confidence_threshold, strategy_weights) based on observed
precision measured against true violation labels.  The adapter must maintain
an experience list so that policy drift over multiple batches is auditable.

Reference: arXiv 2601.00095 (MetaJuLS — meta-RL for universal constraint
propagation).  Self-Learning Tier 2 (Constraint Memory) applied to the
extractor itself.

Spec: REQ-LEARN-078, SCENARIO-LEARN-121, SCENARIO-LEARN-122

---

### REQ-LEARN-079: MetaJuLS Precision Trend is Non-Negative After Adaptation

**Requirement:** After at least two batches of meta-RL adaptation, the
precision_trend() method must return a value >= 0 when the adapter has
successfully learned from feedback — indicating that precision is non-decreasing
over the most recent experience window.

Spec: REQ-LEARN-079, SCENARIO-LEARN-123

---

### SCENARIO-LEARN-121: update_from_batch Adjusts Temperature Down When Precision is Low

**Given** a MetaJuLSAdapter with default policy (temperature=0.1),

**When** update_from_batch() is called with a batch where precision < 0.5
(more false positives than true positives),

**Then** policy.temperature decreases (multiplied by 0.9) and
policy.claim_confidence_threshold increases (multiplied by 1.1).

Spec: SCENARIO-LEARN-121

---

### SCENARIO-LEARN-122: update_from_batch Relaxes Thresholds When Precision is High

**Given** a MetaJuLSAdapter with default policy (temperature=0.1,
claim_confidence_threshold=0.5),

**When** update_from_batch() is called with a batch where precision > 0.8
(strong true-positive signal),

**Then** policy.temperature increases (multiplied by 1.05) and
policy.claim_confidence_threshold decreases (multiplied by 0.95).

Spec: SCENARIO-LEARN-122

---

### SCENARIO-LEARN-123: precision_trend Returns Non-Negative After Improving Batches

**Given** a MetaJuLSAdapter that has processed two batches where the second
batch has equal or higher precision than the first,

**When** precision_trend() is called,

**Then** the return value is >= 0.0, confirming non-decreasing precision.

Spec: SCENARIO-LEARN-123

### REQ-LEARN-080: Tier 1 FR-11 Relay — ConstraintAdditionFromMemory Updates on Real Violations When Available

When Exp 620 (or successor live VR attempt) reports signed_improvement > 0 AND
n_violations_found > 0, the FR-11 relay experiment shall feed those REAL violations
into ConstraintAdditionFromMemory.add_from_violations() and measure the FP-rate delta
across two sessions (before and after the update).

When Exp 620 is blocked (signed_improvement <= 0 or n_violations_found == 0), the relay
shall generate 25 explicit synthetic arithmetic violations to maintain relay continuity
and document fr11_real_violations_confirmed = False.

In both modes the artifact must record: mode, n_violations_used,
fr11_real_violations_confirmed, fp_rate_before, fp_rate_after, fp_rate_delta.

### SCENARIO-LEARN-124: Real-Mode Relay Completes When Exp 620 Has Positive signed_improvement

**Given** Exp 620 reports signed_improvement > 0 and n_violations_found > 0
**When** the Tier 1 relay runs
**Then** violations are loaded from Exp 620 and fed to ConstraintAdditionFromMemory
**And** fp_rate_delta = fp_rate_after - fp_rate_before is recorded
**And** fr11_real_violations_confirmed = True
**And** honest_verdict = 'real_violations_relay_complete'

### SCENARIO-LEARN-125: Synthetic-Fallback Relay Fires When Exp 620 Is Blocked

**Given** Exp 620 reports signed_improvement <= 0 or is absent
**When** the Tier 1 relay runs
**Then** 25 synthetic arithmetic violations are generated
**And** ConstraintAdditionFromMemory processes the synthetic violations
**And** fr11_real_violations_confirmed = False
**And** honest_verdict = 'synthetic_fallback_relay_complete'

### REQ-LEARN-081: FR-11 Relay v2 — ConstraintAdditionFromMemory Updates from Real Violations When Available, Interwhen-Detected Violations When VR Blocked

When Exp 630 (live VR attempt 16) reports signed_improvement > 0 AND n_violations_found > 0,
the FR-11 relay experiment shall feed those REAL violations into ConstraintAdditionFromMemory
and record fr11_mode='real_violations'.

When Exp 630 is blocked but Exp 629 (interwhen diagnostic) reports gate_open=True AND
interwhen_tp > 0, the relay shall use interwhen-detected violations as a semi-real proxy
and record fr11_mode='semi_real_interwhen'.

Otherwise the relay generates 25 synthetic arithmetic violations
(fr11_mode='synthetic_fallback').

All modes record: schema='carnot.tier1_fr11_relay.v2', mode, n_violations_used,
fr11_real_violations_confirmed, fp_rate_before, fp_rate_after, fp_rate_delta,
honest_verdict.

Spec: REQ-LEARN-081

### SCENARIO-LEARN-126: Real-Mode Relay v2 Fires When Exp 630 Has Positive signed_improvement

**Given** Exp 630 reports signed_improvement > 0 and n_violations_found > 0
**When** the Tier 1 FR-11 relay v2 runs
**Then** violations from Exp 630 are fed to ConstraintAdditionFromMemory
**And** fr11_real_violations_confirmed = True
**And** honest_verdict = 'real_violations_relay'

### SCENARIO-LEARN-127: Synthetic-Fallback Relay v2 Fires When Both Exp 630 and Exp 629 Are Blocked

**Given** Exp 630 is blocked (signed_improvement <= 0 or n_violations_found == 0)
**And** Exp 629 shows gate_open = False or interwhen_tp = None
**When** the Tier 1 relay v2 runs
**Then** 25 synthetic arithmetic violations are generated
**And** fr11_real_violations_confirmed = False
**And** honest_verdict = 'synthetic_fallback_relay'

---

### REQ-LEARN-082: FR-11 Relay v3 — ConstraintAdditionFromMemory Updates from Real Violations When Available, Ensemble-Detected Violations When VR Blocked

When Exp 644 (live VR attempt 17) reports signed_improvement > 0,
the FR-11 relay experiment shall feed those REAL violations into ConstraintAdditionFromMemory
and record fr11_mode='real_violations'.

When Exp 644 is blocked but Exp 643 (Ensemble Recall Gate v2) reports gate_open=True AND
ensemble_tp > 0, the relay shall use ensemble-detected violations as a semi-real proxy
and record fr11_mode='semi_real_ensemble'.

Otherwise the relay generates 25 synthetic arithmetic violations
(fr11_mode='synthetic_fallback').

All modes record: schema='carnot.tier1_fr11_relay.v3', mode, n_violations_used,
fr11_real_violations_confirmed, fp_rate_before, fp_rate_after, fp_rate_delta,
honest_verdict.

Spec: REQ-LEARN-082

### SCENARIO-LEARN-128: Semi-Real Ensemble Relay v3 Fires When Exp 644 Is Blocked but Exp 643 Gate Is Open

**Given** Exp 644 reports signed_improvement <= 0
**And** Exp 643 shows gate_open = True and ensemble_tp > 0
**When** the Tier 1 FR-11 relay v3 runs
**Then** ensemble-detected violations from Exp 643 are fed to ConstraintAdditionFromMemory
**And** fr11_real_violations_confirmed = False
**And** honest_verdict = 'semi_real_ensemble_relay'

### SCENARIO-LEARN-129: Synthetic-Fallback Relay v3 Fires When Both Exp 644 and Exp 643 Are Blocked

**Given** Exp 644 is blocked (signed_improvement <= 0)
**And** Exp 643 shows gate_open = False or ensemble_tp = 0
**When** the Tier 1 relay v3 runs
**Then** 25 synthetic arithmetic violations are generated
**And** fr11_real_violations_confirmed = False
**And** honest_verdict = 'synthetic_fallback_relay'

---

### REQ-LEARN-083: JEPA v15 Trained on Real Violation Data Achieves OOD AUC >= 0.80

**Requirement:** When retrained on real violation pairs sourced from Exp 659
(FR-11 Tier 2 relay) and FOVER live pairs, the JEPAViolationPredictor v15 must
achieve OOD AUC >= 0.80 on a 20% held-out subset of those real pairs.

**Rationale:** The CPMI + PURE min-form objective provides contrastive gradient signal
(correct chain energy < incorrect chain energy by a margin) that BCE alone cannot.
Training on real violation data rather than synthetic pairs closes the loop from FR-11
relay → model improvement.

**Acceptance criteria:**
- ood_auc >= 0.80 in the experiment artifact
- Training data includes at least one non-synthetic pair from Exp 659 or FOVER live

Spec: REQ-LEARN-083, SCENARIO-LEARN-130

### SCENARIO-LEARN-130: JEPA v15 OOD AUC Meets Target on Real Violation Pairs

**Given** Exp 659 violation relay and FOVER live pairs are available
**When** JEPAViolationPredictor v15 is trained with CPMI+PURE for 100 epochs
**Then** ood_auc >= 0.80 on the 20% held-out test split
**And** the artifact records honest_verdict = 'jepa_v15_target_met' or 'jepa_v15_auc_met'

---

### REQ-LEARN-084: JEPA v15 ECE < 0.10 After Platt Calibration

**Requirement:** After training v15 on real violation pairs, a Platt temperature T
is fitted on the validation set. The Expected Calibration Error (ECE) of the calibrated
model must be < 0.10.

**Rationale:** Raw energy scores are uncalibrated probabilities.  Dividing by T before
sigmoid converts the energy output into a well-calibrated probability estimate that
downstream confidence thresholds can trust.

**Acceptance criteria:**
- ece_post_calibration < 0.10 in the experiment artifact
- platt_temperature_T > 0.0 (a positive scalar)

Spec: REQ-LEARN-084, SCENARIO-LEARN-131

### SCENARIO-LEARN-131: JEPA v15 Platt Calibration Yields ECE < 0.10

**Given** v15 is trained and produces raw energy scores on the validation set
**When** Platt temperature T is fitted via Brent's method on the validation set
**Then** ece_post_calibration < 0.10
**And** platt_temperature_T is a finite positive scalar

### SCENARIO-LEARN-132: JEPA v15 honest_verdict Is a Valid Enum Member

**Given** any execution path of experiment_671_jepa_v15.py
**When** the experiment artifact is written
**Then** honest_verdict is one of:
    'jepa_v15_target_met' | 'jepa_v15_auc_met' | 'jepa_v15_partial'
    | 'jepa_v15_no_improvement' | 'ci_mode_synthetic'

### REQ-LEARN-085: MetaJuLS Forcing Adapter Updates Forcing Strategy from Live Feedback

**Given** a MetaJuLSForcingAdapter initialized with base FORCER_SYSTEM_ADDENDUM
**When** ForcingFeedback observations are submitted via update()
**Then** the adapter accumulates per-domain recall history
**And** domains with mean recall < 0.30 receive a CRITICAL: emphasis string

### REQ-LEARN-086: Adapted Forcing Addendum Is Extended for Low-Recall Domains

**Given** a MetaJuLSForcingAdapter that has received low-recall feedback for a domain
**When** get_adapted_addendum() is called for that domain
**Then** the returned addendum is longer than the base FORCER_SYSTEM_ADDENDUM
**And** the base addendum is preserved as a prefix in the extended addendum

### SCENARIO-LEARN-133: Low Recall Domain Triggers CRITICAL Emphasis

**Given** a MetaJuLSForcingAdapter with base addendum
**When** ForcingFeedback with recall < 0.30 is submitted for domain 'percentage'
**Then** adapter.domain_emphasis['percentage'] contains "CRITICAL"
**And** the emphasis string references the 'percentage' domain by name

### SCENARIO-LEARN-134: High Recall Domain Keeps Base Addendum Unchanged

**Given** a MetaJuLSForcingAdapter
**When** ForcingFeedback with recall >= 0.30 is submitted for domain 'arithmetic'
**Then** no emphasis is installed for 'arithmetic'
**And** get_adapted_addendum() returns the base addendum unchanged

### SCENARIO-LEARN-135: Save/Load Round-Trip Preserves Full Adapter State

**Given** a MetaJuLSForcingAdapter with accumulated recall history and installed emphasis
**When** save_state() is called and load_state() restores from that file
**Then** domain_recalls matches the original
**And** domain_emphasis matches the original
**And** get_adapted_addendum() produces identical output before and after round-trip

### REQ-LEARN-087: JEPA v15 OOD AUC Measured on Truly Unseen GSM8K 500-699

**Given** a trained JEPA v15 model (jepa_predictor_v15_real.safetensors)
**When** evaluated on 200 GSM8K questions with indices 500-699 that were never present
    in any form during training (the FOVER corpus covers ~indices 150-499)
**Then** true_ood_auc is computed via ROC-AUC on the held-out questions
**And** if true_ood_auc == 1.0, honest_verdict = 'jepa_v15_overfit' (confirms data leakage suspicion)
**And** if true_ood_auc >= 0.80 and ece < 0.10, honest_verdict = 'jepa_v15_ood_target_met'
**And** if 0.60 <= true_ood_auc < 0.80, honest_verdict = 'jepa_v15_ood_partial'
**And** if true_ood_auc < 0.50, honest_verdict = 'jepa_v15_ood_below_random'

Spec: REQ-LEARN-087, SCENARIO-LEARN-136

### REQ-LEARN-088: JEPA v15 ECE < 0.10 on True OOD Set via Platt Calibration

**Given** JEPA v15 raw energy scores on GSM8K 500-699
**When** Platt temperature T is fitted on those same 200 OOD samples (post-hoc calibration)
**Then** ECE (10-bin histogram) of calibrated probabilities is reported in the artifact
**And** platt_temperature is a finite positive scalar in [0.01, 10.0]

Spec: REQ-LEARN-088, SCENARIO-LEARN-137

### SCENARIO-LEARN-136: JEPA v15 OOD Verdict Matches True AUC Threshold

**Given** Exp 682 runs on GSM8K indices 500-699 (never seen during training)
**When** true_ood_auc is computed and compared against the thresholds in REQ-LEARN-087
**Then** honest_verdict is a member of VALID_VERDICTS
**And** the artifact includes true_ood_auc, training_indices_avoided, and n_ood_questions=200

### SCENARIO-LEARN-137: ECE Is Computed After Platt Temperature Scaling on OOD Set

**Given** JEPA v15 raw scores on GSM8K 500-699
**When** Platt temperature T is fitted via Brent's method (or grid fallback) on the OOD set
**Then** calibrated_probs = sigmoid(score / T) are computed for all 200 questions
**And** ece = weighted mean |confidence - accuracy| across 10 equal-width probability bins
**And** ece is a finite non-negative float in [0, 1]

### REQ-LEARN-089: JEPA v16 Design Must Address Identified Root Cause from v15 Audit

**Given** the JEPA v15 OOD AUC=0.4751 (below random chance) confirmed by Exp 682
**When** root cause probes (H1/H2/H3) identify the primary failure mode
**Then** the v16 architecture specification must directly address the identified root cause
**And** v16_architecture_spec must be a non-empty string describing the concrete change
**And** v16_training_data_target must be a positive integer specifying the minimum training
    pairs required

Spec: REQ-LEARN-089, SCENARIO-LEARN-138, SCENARIO-LEARN-139, SCENARIO-LEARN-140

### REQ-LEARN-090: JEPA v16 OOD AUC Target >= 0.75 on GSM8K 500-699

**Given** a JEPA v16 model trained per the architecture specification in REQ-LEARN-089
**When** evaluated on GSM8K 500-699 (truly OOD from the training distribution)
**Then** true_ood_auc >= 0.75, meaning the model is meaningfully better than random chance
**And** the result must be validated with the same evaluation protocol as Exp 682
    (RandomProjectionEmbedding seed=671, ground_truth_label = idx%3==0 fallback)

Spec: REQ-LEARN-090, SCENARIO-LEARN-138

### SCENARIO-LEARN-138: H1 Probe Detects Distribution Shift via L2 and Variance Ratio

**Given** train latents (from fover_labeled_formal_v1.json, GSM8K ~0-499 range)
    and OOD latents (from GSM8K 500-699 questions)
**When** probe_h1_distribution_shift is computed on the extracted h2 (32-dim) representations
**Then** H1_confirmed = True if distribution_shift_l2 > 0.5 OR variance_ratio > 2.0
**And** the result is reproducible given the same JEPA weights and RandomProjectionEmbedding seed

### SCENARIO-LEARN-139: H3 Probe Detects Latent Collapse via Effective Rank

**Given** train latents extracted from JEPA v15 for the FOVER training pairs
**When** SVD is computed on the (N_train, 32) latent matrix
**Then** effective_rank = count of singular values > 0.01
**And** H3_confirmed = True if effective_rank < 5 OR (top singular value)^2 / total variance > 0.9
**And** a confirmed H3 indicates that 57 training pairs were insufficient to fill the 32-dim latent space

### SCENARIO-LEARN-140: Root Cause Assignment and v16 Spec Generation

**Given** the probe results H1_confirmed, H2_confirmed, H3_confirmed from Exp 693
**When** determine_root_cause is called with those flags
**Then** root_cause is selected by priority: H1 > H2 > H3 > unknown_requires_ablation
**And** build_v16_spec returns a non-empty architecture prescription for the identified root cause
**And** honest_verdict = 'root_cause_identified_v16_specced' if root_cause != 'unknown_requires_ablation'
**And** honest_verdict = 'root_cause_ambiguous_ablation_needed' otherwise

### REQ-LEARN-091: PSV Real Self-Play with K=2 Parallel Chains (Exp 697)

**Given** two parallel PSV chains running simultaneously on disjoint question subsets
**When** each chain completes n_iterations=10 with live Qwen3.5-0.8B inference
**Then** constraint updates from both chains are merged into shared JitRLConstraintMemory
**And** fp_rate_trend_slope is non-increasing (slope <= 0) to confirm self-improvement
**And** inference_mode must be 'live_gpu' for the verdict to count as 'psv_real_fp_improving'

Spec: REQ-LEARN-091, SCENARIO-LEARN-141, SCENARIO-LEARN-142, SCENARIO-LEARN-143

### REQ-LEARN-092: Parallel Chain Merge — Conflict-Free Constraint Update Aggregation

**Given** two PSV chains that processed different questions and recorded different violation pairs
**When** their constraint updates are merged via the shared JitRLConstraintMemory
**Then** all violation pairs from both chains appear in the merged memory without conflicts
**And** merged_constraint_updates >= sum(updates from each chain individually)
**And** no constraint domain thresholds are lost or overwritten by the merge

Spec: REQ-LEARN-092, SCENARIO-LEARN-142

### SCENARIO-LEARN-141: PSV Parallel Chains Reduce FP Rate Over 10 Iterations

**Given** PSVParallelChains(n_chains=2, n_iterations=10, n_questions_per_iter=10)
**When** run_parallel is called with 200 GSM8K questions and live Qwen3.5-0.8B inference
**Then** fp_rate_per_iteration is a list of 10 floats
**And** fp_rate_trend_slope < 0 indicates the parallel self-play is improving the FP rate
**And** honest_verdict = 'psv_real_fp_improving' when slope < 0 and inference_mode='live_gpu'

### SCENARIO-LEARN-142: Parallel Chain Merge Produces Valid merged_constraint_updates Count

**Given** two chains each running 10 iterations with 10 questions per iteration
**When** all violation pairs from both chains are aggregated into the shared constraint memory
**Then** merged_constraint_updates is an int >= 0
**And** parallel_speedup_factor > 0.0 (positive speedup measured from wall-clock time)
**And** the merged memory contains pairs from both chain 0 and chain 1

### SCENARIO-LEARN-143: PSV Real Blocked When GPU Gate Missing

**Given** CARNOT_FORCE_LIVE is not set or falsy and env_autofix cannot inject it
**When** the experiment checks the GPU gate
**Then** honest_verdict = 'psv_real_blocked_no_gpu'
**And** the artifact is written with status='blocked' and the experiment exits 0

### REQ-LEARN-093: JEPA v21 Multi-Source Training Data — Domain Coverage

JEPA v21 training data MUST span at least 2 distinct benchmark domains
(GSM8K, MATH-500, HumanEval) to support OOD generalization. A single-source
corpus (all from one domain) is insufficient evidence that the JEPA encoder has
learned domain-invariant correctness signals.

**Rationale:** Exps v13-v20 used only Qwen3.5-0.8B on GSM8K q1-300. All 57 pairs
shared the same domain and question range, causing the JEPA OOD AUC to stay
below 0.75 for 8 consecutive retrains. Diversity in benchmark domain — not model
— is the key fix.

**Acceptance criteria:**
- Data collection produces labeled pairs from >= 2 of: GSM8K, MATH-500, HumanEval.
- The merged corpus file contains a `source_domain` field per pair identifying its origin.
- Total n_labeled >= 80 across all sources combined.

Spec: SCENARIO-LEARN-144

### REQ-LEARN-094: Multi-Source FOVER Corpus — Single Merged File with source_domain

The multi-source FOVER annotation pipeline MUST write all labeled pairs from all
benchmark domains into a single merged corpus file (`fover_labeled_steps_v21_multi.json`)
with a `source_domain` field per pair. This merged file is distinct from the Exp 442
corpus (`fover_labeled_steps_live.json`) and must NOT overwrite it.

**Why a merged file:** Downstream JEPA v21 training needs a single unified corpus
that the trainer can load without domain-specific logic. The `source_domain` field
lets the trainer optionally stratify the corpus by domain for OOD evaluation.

Spec: SCENARIO-LEARN-144

### SCENARIO-LEARN-144: Multi-Domain FOVER Data Collection Completes with Adequate Corpus

**Given** CARNOT_FORCE_LIVE=1 and GPU is available
**And** Qwen3.5-0.8B is loaded on GPU 0
**When** the experiment collects CoT responses from GSM8K q301-360, MATH-500 q0-29, and HumanEval p0-29
**And** FOVERAnnotator.annotate_corpus() labels each step
**Then** n_labeled_total >= 80
**And** >= 2 source domains have at least 1 labeled pair
**And** results/fover_labeled_steps_v21_multi.json is written with source_domain per pair
**And** honest_verdict = 'multi_source_corpus_adequate'
**And** results/fover_labeled_steps_live.json is NOT modified

---

## Implementation Status

| Requirement | Rust | Python | Tests |
|------------|------|--------|-------|
| REQ-AUTO-001 | Implemented | Partial | 14 Rust |
| REQ-AUTO-002 | Partial | Implemented | 3 Python |
| REQ-AUTO-003 | Not Started | Implemented | Integration |
| REQ-AUTO-004 | Not Started | Implemented | 13 + 21 Python |
| REQ-AUTO-005 | Not Started | Implemented | 7 Python |
| REQ-AUTO-006 | Not Started | Implemented (`transpile.py`) | 15 Python (fixed 2026-09-11 -- this row said "Not Started" across the board while `generate_test_vectors`/`validate_conformance`/`validate_performance` were implemented and exported from `carnot.autoresearch.__init__`; the Rust-transpilation-automation half of the REQ is genuinely not started, only the Python validation side is) |
| REQ-AUTO-007 | N/A | Implemented (`rollback.py`) | 9 Python (fixed 2026-09-11, same drift as REQ-AUTO-006's row) |
| REQ-AUTO-008 | Not Started | Implemented | 5 Python |
| REQ-AUTO-009 | Not Started | Implemented | 4 Python |
| REQ-AUTO-010 | Not Started | Not Started | Not Started |
| REQ-AUTO-011 | N/A | Implemented | 10+ Python |
| REQ-AUTO-012 | N/A | Implemented | 11+ Python |
| REQ-AUTO-013 | N/A | Implemented | 9+ Python |
| REQ-AUTO-014 | N/A | Implemented | Integration |
| REQ-AUTO-019 | N/A | Implemented (`scripts/autoresearch_conductor_round.py`) | 11 Python |
| REQ-AUTO-020 | N/A | Implemented (`scripts/autoresearch_conductor_round.py`) | 11 Python (shared test file with REQ-AUTO-019) |
| REQ-AUTO-021 | N/A | Implemented (`python/carnot/autoresearch/toy_benchmarks.py`, `scripts/autoresearch_conductor_round.py`) | 16 Python (`test_autoresearch_toy_benchmarks.py`) + shared conductor-round test file |
| REQ-AUTO-022 | N/A | Implemented (`scripts/autoresearch_conductor_round.py`) | 2 Python (shared conductor-round test file) |
| REQ-AUTO-023 | N/A | Implemented (`python/carnot/autoresearch/orchestrator.py`, `scripts/autoresearch_conductor_round.py`) | 6 Python (`test_autoresearch_generator.py`, `test_autoresearch_skills_loop.py`) |
| REQ-AUTO-024 | N/A | Implemented (`scripts/autoresearch_conductor_round.py`) | 5 Python (shared conductor-round test file) |
| REQ-AUTO-025 | N/A | Implemented, hardened same-day per 2026-09-16 adversarial review (`python/carnot/autoresearch/verifier_auroc_benchmark.py`, `scripts/autoresearch_conductor_round.py`, `scripts/_autoresearch_energy_recompute_worker.py`) | 31 Python (`test_autoresearch_verifier_auroc_benchmark.py`) + 13 shared conductor-round test file + 1 shared toy-benchmarks test file |
| REQ-AUTO-018 | N/A | Implemented (`python/carnot/autoresearch/calibrated_decision_benchmark.py`, `scripts/autoresearch_conductor_round.py`, `scripts/_autoresearch_energy_recompute_worker.py`) | 31 Python (`test_calibrated_decision_benchmark.py`) + shared conductor-round test file |
| REQ-LEARN-010 | N/A | Implemented | 22 Python |
| REQ-LEARN-011 | N/A | Implemented | 22 Python |
| REQ-LEARN-030 | N/A | Implemented | 10+ Python |
| REQ-LEARN-031 | N/A | Implemented | 10+ Python |
| REQ-LEARN-032 | N/A | Implemented | 10+ Python |
| REQ-LEARN-033 | N/A | Implemented | 10+ Python |
| REQ-LEARN-034 | N/A | Implemented | 10+ Python |
| REQ-LEARN-035 | N/A | Implemented | 10+ Python |
| REQ-LEARN-036 | N/A | Implemented | 10+ Python |
| REQ-LEARN-037 | N/A | Implemented | Python |
| REQ-LEARN-038 | N/A | Implemented | Python |
| REQ-LEARN-039 | N/A | Implemented | Python |
| REQ-LEARN-040 | N/A | Implemented | Python |
| REQ-LEARN-041 | N/A | Implemented | Python |
| REQ-LEARN-042 | N/A | Implemented | Python |
| REQ-LEARN-039 (v4) | N/A | Implemented | 11 Python |
| REQ-LEARN-040 (v4) | N/A | Implemented | 11 Python |
| REQ-LEARN-041 (v4) | N/A | Implemented | 11 Python |
| REQ-LEARN-046 | N/A | Implemented | Python |
| REQ-LEARN-047 | N/A | Implemented | Python |
| REQ-LEARN-048 | N/A | Implemented | Python |
| REQ-LEARN-049 | N/A | Implemented | Python |
| REQ-LEARN-050 | N/A | Implemented | Python |
| REQ-LEARN-051 | N/A | Implemented | Python |
| REQ-LEARN-052 | N/A | Implemented | Python |
| REQ-LEARN-053 | N/A | Implemented | Python |
| REQ-LEARN-054 | N/A | Implemented | Python |
| REQ-LEARN-055 | N/A | Implemented | Python |
| REQ-LEARN-056 | N/A | Implemented | Python |
| REQ-LEARN-057 | N/A | Implemented | Python |
| REQ-LEARN-058 | N/A | Implemented | 26 Python |
| REQ-LEARN-059 | N/A | Implemented | 26 Python |
| REQ-LEARN-060 | N/A | Implemented | Python |
| REQ-LEARN-061 | N/A | Implemented | Python |
| REQ-LEARN-062 | N/A | Implemented | Python |
| REQ-LEARN-063 | N/A | Implemented | Python |
| REQ-LEARN-064 | N/A | Implemented | Python |
| REQ-LEARN-065 | N/A | Implemented | 28 Python |
| REQ-LEARN-066 | N/A | Implemented | 28 Python |
| REQ-LEARN-067 | N/A | Implemented | Python |
| REQ-LEARN-068 | N/A | Implemented | Python |
| REQ-LEARN-069 | N/A | Implemented | 12 Python |
| REQ-LEARN-070 | N/A | Implemented | Python (Exp 597) |
| REQ-LEARN-071 | N/A | Implemented | Python (Exp 597) |
| REQ-LEARN-072 | N/A | Implemented | Python (test_hisr_weights.py, 100%) |
| REQ-LEARN-071 | N/A | Implemented | Python (test_mise_calibrator) |
| REQ-LEARN-073 | N/A | Implemented | Python (test_jepa_ood_validation.py) |
| REQ-LEARN-074 | N/A | Implemented | Python (test_jepa_ood_validation.py) |
| REQ-LEARN-076 | N/A | Implemented | Python (test_flip_calibrator.py) |
| REQ-LEARN-077 | N/A | Implemented | Python (Exp 611) |
| REQ-LEARN-078 | N/A | Implemented | Python (test_metajuls_adapter.py) |
| REQ-LEARN-079 | N/A | Implemented | Python (test_metajuls_adapter.py) |
| REQ-LEARN-080 | N/A | Implemented | Python (test_experiment_625_fr11_relay.py) |
| REQ-LEARN-081 | N/A | Implemented | Python (test_experiment_638_fr11_relay.py) |
| REQ-LEARN-082 | N/A | Implemented | Python (test_experiment_645_fr11_relay.py) |
| REQ-LEARN-083 | N/A | Implemented | Python (test_experiment_671_jepa_v15.py) |
| REQ-LEARN-084 | N/A | Implemented | Python (test_experiment_671_jepa_v15.py) |
| REQ-LEARN-085 | N/A | Implemented | Python (test_metajuls_forcing_adapter.py) |
| REQ-LEARN-086 | N/A | Implemented | Python (test_metajuls_forcing_adapter.py) |
| REQ-LEARN-087 | N/A | Implemented | Python (test_experiment_682_jepa_v15_ood_audit.py) |
| REQ-LEARN-088 | N/A | Implemented | Python (test_experiment_682_jepa_v15_ood_audit.py) |
| REQ-LEARN-089 | N/A | Implemented | Python (test_experiment_693_jepa_root_cause.py) |
| REQ-LEARN-090 | N/A | Implemented | Python (test_experiment_693_jepa_root_cause.py) |
| REQ-LEARN-091 | N/A | Implemented | Python (test_psv_parallel_chains.py) |
| REQ-LEARN-092 | N/A | Implemented | Python (test_psv_parallel_chains.py) |
| REQ-LEARN-093 | N/A | Implemented | Python (test_experiment_797_jepa_v21_data_collection.py) |
| REQ-LEARN-094 | N/A | Implemented | Python (test_experiment_797_jepa_v21_data_collection.py) |
| REQ-LEARN-095 | N/A | Implemented | Python (test_experiment_799_jepa_v21_retrain.py) |
| REQ-LEARN-096 | N/A | Implemented | Python (test_experiment_799_jepa_v21_retrain.py) |
| REQ-LEARN-097 | N/A | Implemented | Python (test_experiment_799_jepa_v21_retrain.py) |
| REQ-INFRA-1337 | N/A | Implemented | Python (test_environment_gate.py) |

## REQ-LEARN-052

CPMIContrastivePairBuilder MUST produce (prefix, positive_step, hard_negative_step) triples
where hard_negative is sampled from the model distribution at temperature=0.9 with CPMI score
in [0.15, 0.60]. In CI mode, CPMI score is approximated via cosine-similarity proxy.

## REQ-LEARN-053

The augmented corpus MUST have augmentation_ratio >= 2.0 (triples / input pairs), ensuring
the contrastive training set is at least twice the size of the original labeled set.

## REQ-LEARN-095: JEPA v21 Multi-Source Training with CPMI Augmentation

JEPA v21 MUST be trained on a multi-source corpus spanning >= 2 domains (GSM8K, MATH-500,
HumanEval) with CPMI-augmented contrastive triples. augmentation_ratio >= 2.0 (triples /
input labeled pairs). Sources are merged from fover_labeled_steps_v21_multi.json (primary)
and experiment_798_cpmi_pairs_triples.json (CPMI triples).

Spec: REQ-LEARN-095, SCENARIO-LEARN-096, SCENARIO-LEARN-097

## REQ-LEARN-096: JEPA v21 PROGRS Outcome-Conditioned Centering

JEPA v21 training MUST apply PROGRS outcome-conditioned centering: weight each step-pair
loss by source_domain_accuracy for the pair's source domain. Domain accuracy values reflect
live benchmark performance (gsm8k: 0.14, math500: 0.12, humaneval: 0.20). This corrects
for domain difficulty — harder domains get lower weights so the model does not overfit to
high-error domains.

Spec: REQ-LEARN-096, SCENARIO-LEARN-096

## REQ-LEARN-097: JEPA v21 OOD Deployment Gate

JEPA v21 deployment gate: ood_auc >= 0.75 required to wire into ThreeTierPipeline as
Tier 3.5. If gate passes, model saved to results/jepa_predictor_v21.safetensors and
tier35_deployed=True. If gate fails, per-domain AUC failure analysis MUST be produced
with recommendations for v22.

Spec: REQ-LEARN-097, SCENARIO-LEARN-096, SCENARIO-LEARN-097

## SCENARIO-LEARN-096: JEPA v21 Multi-Source Training Succeeds, OOD Gate Passes

Given multi-source corpus (>= 2 domains, n_labeled_total >= 80) and CPMI triples from
Exp 798, JEPA v21 trains with PROGRS outcome-conditioned weights; ood_auc >= 0.75;
model saved; Tier 3.5 wired into ThreeTierPipeline; honest_verdict = "jepa_v21_tier35_deployed".

## SCENARIO-LEARN-097: JEPA v21 OOD Below Gate, Failure Analysis Produced

Given training completes but ood_auc < 0.75; per-domain AUC breakdown identifies which
domain contributes most to OOD variance; recommendations for v22 produced;
tier35_deployed = False; honest_verdict = "jepa_v21_below_gate".

## SCENARIO-LEARN-095

Given 80 input pairs from the multi-source FOVER corpus, CPMIContrastivePairBuilder produces
>= 160 contrastive triples with CPMI scores in the target range [0.15, 0.60];
augmentation_ratio >= 2.0.

## REQ-LEARN-098: FR-11 Tier 1 Relay MUST Use EmbeddingConstraintStore

The FR-11 Tier 1 relay experiment (Exp 802) MUST use EmbeddingConstraintStore as the primary
constraint learning mechanism, replacing the scalar keyword-count encoding (CaseMemoryTemplateWiring)
that produced zero delta in Exps 761 and 788.

The relay MUST run for 10 sessions of 50 questions each.  After each session, violation events
from that session MUST be used to update the EmbeddingConstraintStore (online learning loop).
Precision MUST be non-decreasing across all 10 sessions.  Delta (precision[session5] - precision[session1])
MUST be positive by session 5 for the relay to claim FR-11 Tier 1 is satisfied.

Spec: REQ-LEARN-098, SCENARIO-LEARN-145

## SCENARIO-LEARN-145: FR-11 Tier 1 Relay — Precision Non-Decreasing, Delta Positive by Session 5

Given a 10-session Tier 1 relay using EmbeddingConstraintStore (bootstrapped from 5 canonical
CaseMemory patterns) and 50 synthetic GSM8K-style questions per session:
  - precision_per_session is a list of 10 float values in [0.0, 1.0]
  - is_monotonically_non_decreasing = True (no session drops below the previous)
  - delta_positive_by_s5 = True (precision[4] > precision[0])
  - honest_verdict = "tier1_relay_works"
  - tier1_relay_works = True

When delta_positive_by_s5 = True but monotonic = False:
  - honest_verdict = "tier1_partial_improvement"

When precision[4] == precision[0] (EmbeddingConstraintStore did not move precision):
  - honest_verdict = "tier1_plateau_persists"
  - This reproduces the Exps 761/788 failure mode and triggers a research escalation.

## REQ-LEARN-099: JEPA v22 Training Data MUST Merge Multi-Source FoVer Corpus with CPMI Triples

JEPA v22 training scripts MUST load and merge both:
  - fover_labeled_steps_v21_multi.json (multi-source FoVer corpus, 300 pairs)
  - experiment_798_cpmi_pairs_triples.json (CPMI contrastive triples)

augmentation_ratio = total_training_items / n_fover_pairs MUST be >= 1.5 before
training begins.  augmentation_ratio = 1.0 (CPMI not merged) is the Exp 799 failure
mode that produced ood_auc=0.2444, the all-time project low.

CPMI triple expansion: each triple (prefix, positive_step, negative_step) produces two
training pairs.  Negative pairs are weighted at CPMI_NEGATIVE_WEIGHT=0.7× to reduce
overfitting to synthetic hard negatives.

Spec: REQ-LEARN-099, SCENARIO-LEARN-146

## REQ-LEARN-100: JEPA v22 Training Scripts MUST Call check_cpmi_wiring() as First Assertion

Before any model initialisation or data loading, JEPA v22 training scripts MUST call:
  check_cpmi_wiring(triples_path, min_augmentation_ratio=1.5)
from python/carnot/pipeline/jepa_wiring_guard.py.

If the guard raises AssertionError or FileNotFoundError, the experiment MUST:
  1. Write a blocked artifact with honest_verdict="blocked_wiring_miss"
  2. Set status="blocked"
  3. Exit immediately without loading data or training

This prevents the silent data-loader misconfiguration that caused Exp 799 to train
for 5+ minutes producing ood_auc=0.2444 with no CPMI augmentation.

Spec: REQ-LEARN-100, SCENARIO-LEARN-147

## SCENARIO-LEARN-146: CPMI Triples Loaded; Training Proceeds; OOD AUC Evaluated

Given CPMI triples loaded (n_cpmi_triples=300) and FoVer corpus (n_fover_pairs=300):
  - augmentation_ratio = total_training_items / n_fover_pairs MUST be >= 1.5
  - Training proceeds for 80 epochs with PROGRS outcome-conditioned weights
  - OOD AUC evaluated on fover_labeled_steps_live.json (Exp 442 held-out set)
  - tier35_deployed=True if ood_auc >= 0.75
  - honest_verdict = "jepa_v22_tier35_deployed" if ood_auc >= 0.75
  - honest_verdict = "jepa_v22_improvement_vs_v21" if 0.5 <= ood_auc < 0.75
  - honest_verdict = "jepa_v22_below_random" if ood_auc < 0.5

## SCENARIO-LEARN-147: Training Blocked When Wiring Guard Detects augmentation_ratio < 1.5

Given training starts and check_cpmi_wiring() detects augmentation_ratio=1.0
(or CPMI file missing):
  - AssertionError is caught before any model initialisation
  - Experiment writes artifact with status="blocked"
  - honest_verdict = "blocked_wiring_miss"
  - tier35_deployed = False
  - Training does NOT proceed
  - This prevents silent repetition of the Exp 799 failure (ood_auc=0.2444, aug_ratio=1.0)

## REQ-LEARN-101: Held-Out Evaluation on Distinct Domain When JEPA v22 ood_auc >= 0.75

If Exp 808 reports ood_auc >= 0.75, Exp 809 MUST evaluate JEPA v22 on a held-out benchmark
(ARC, SVAMP, or StrategyQA) that was NOT present in the training corpus.  The held-out set
MUST contain at least 20 reasoning problems.  The resulting held_out_auc MUST be logged in
the deliverable.

Spec: REQ-LEARN-101, SCENARIO-LEARN-148

## REQ-LEARN-102: RA-PRM Retrieval-Augmented Soft Supervision When JEPA v22 ood_auc < 0.75

If Exp 808 reports ood_auc < 0.75, Exp 809 MUST apply RA-PRM (Retrieval-Augmented PRM):
  1. Populate EmbeddingConstraintStore with all FoVer-labeled steps from
     fover_labeled_steps_v21_multi.json.
  2. For each training example, retrieve K=3 similar steps via retrieve().
  3. Compute augmented soft labels: ground_truth_label × 1.0 + retrieved_labels × 0.4.
  4. Retrain JEPA v22-rapbm for 80 epochs on augmented pairs.
  5. Evaluate in-distribution AUC and OOD AUC on fover_labeled_steps_live.json.
  6. Save model to results/jepa_predictor_v22_rapbm.safetensors if ood_auc improves.

Spec: REQ-LEARN-102, SCENARIO-LEARN-149

## SCENARIO-LEARN-148: Path A — JEPA v22 OOD Confirmed on Held-Out Benchmark

Given Exp 808 ood_auc=0.80 (>= 0.75):
  - Exp 809 loads 20 ARC reasoning problems (hardcoded, CPU)
  - Generates synthetic CoT via template, runs FoVer annotation
  - Evaluates JEPA v22 on FoVer-labeled steps; held_out_auc=0.72
  - honest_verdict = "v22_ood_confirmed" (held_out_auc >= 0.65)
  - tier35 generalization validated on unseen domain

## SCENARIO-LEARN-149: Path B — RA-PRM Applied When JEPA v22 OOD Below Threshold

Given Exp 808 ood_auc=0.45 (< 0.75):
  - Exp 809 populates EmbeddingConstraintStore with fover_labeled_steps_v21_multi.json
  - Retrieves 3 similar steps per training example via EmbeddingConstraintStore.retrieve()
  - Computes soft labels: ground_truth × 1.0 + retrieved × 0.4
  - Retrains JEPA v22-rapbm for 80 epochs on augmented pairs
  - Evaluates and logs improved ood_auc
  - honest_verdict = "rapbm_ood_improved" if ood_auc > Exp 808 ood_auc
  - honest_verdict = "rapbm_no_gain" otherwise

### REQ-LEARN-813-001: Constraint Addition Delta MUST Be Evaluated on Live GPU Inference

Exp 813 evaluates constraint addition delta using inference_mode=live_gpu.
Synthetic_cpu delta results (as seen in Exps 801/802) are labeled synthetic_only
and do NOT count toward RETRO-CONSTRAINT-ZERO-DELTA closure.

Before running live sessions, Exp 813 loads Exp 812's result and gates on
honest_verdict == "injection_works".  If the gate fails, the experiment writes a
blocked artifact with honest_verdict="injection_not_wired" and exits immediately.

Spec: REQ-LEARN-813-001

### REQ-LEARN-813-002: Closing RETRO-CONSTRAINT-ZERO-DELTA Requires delta_overall > 0 on Live GPU

Closing RETRO-CONSTRAINT-ZERO-DELTA requires ALL of the following:
  1. inference_mode = "live_gpu" (confirmed by LiveGPUGate.require_live_or_blocked)
  2. IsingConstraintInjector wired to EmbeddingConstraintStore (Exp 812 gate passes)
  3. delta_overall > 0 across 3 sessions × 10 questions = 30 questions total
  4. honest_verdict = "constraint_addition_works_live"

delta_overall is defined as the arithmetic mean of per-session delta values:
  delta_session_i = inject_correct_i - baseline_correct_i

Spec: REQ-LEARN-813-002

### SCENARIO-LEARN-813-001: 30q x 3 Sessions Live GPU — Constraint Injection Active

Given live GPU available AND Exp 812 honest_verdict == "injection_works":
  - Exp 813 runs 3 sessions of 10 GSM8K questions each
  - Each session compares baseline VerifyRepairPipeline vs embedding-injected pipeline
  - Example target values: delta_s1=0.05, delta_s2=0.08, delta_s3=0.12
  - delta_overall = mean([0.05, 0.08, 0.12]) = 0.0833
  - retro_constraint_zero_delta_closed = True
  - honest_verdict = "constraint_addition_works_live"
  - store.update_from_session_violations() called after each session

### REQ-LEARN-814-001: FR-11 Tier 1 Live Relay MUST Use IsingConstraintInjector + EmbeddingConstraintStore with Capacity-Constrained Update

Exp 814 evaluates whether precision increases monotonically across 5 relay sessions
using live GPU inference.  Each session uses the full pipeline:
    EmbeddingConstraintStore (SPO embeddings) + IsingConstraintInjector (coupling bias).

After each session, compute retrieval variance per constraint type; update only
top-K (K=3) highest-variance constraints; freeze the rest.  This prevents plateau
by avoiding over-fitting to well-learned constraints (arXiv 2507.21479).

Gate: Exp 813 delta_overall must be > 0.  If null or <= 0, write blocked artifact.

Spec: REQ-LEARN-814-001

### SCENARIO-LEARN-814-001: 5-Session Live Relay — Capacity-Constrained Update per Session

Given live GPU available AND Exp 813 delta_overall > 0:
  - Exp 814 runs 5 sessions of 10 GSM8K questions each (50 total)
  - Each session uses VerifyRepairPipeline with EmbeddingConstraintStore + IsingConstraintInjector
  - After each session, selective_update() updates only top-3 highest-variance constraint types
  - precision is non-decreasing sessions 1-5 (monotonic non-decrease)
  - delta_s1_to_s5 = precision[4] - precision[0] > 0
  - honest_verdict = "tier1_relay_works_live" if monotonic AND delta_positive_by_s3
  - honest_verdict = "tier1_partial_improvement_live" if delta_positive_by_s3 but not monotonic
  - honest_verdict = "tier1_plateau_persists_live" if delta_s1_to_s5 <= 0
  - honest_verdict = "blocked_no_delta" if Exp 813 gate blocks
  - tier1_relay_works_live=True when honest_verdict == "tier1_relay_works_live"

### REQ-LEARN-821-001: Exp 819 Gate — External Field Fix Required Before Live Measurement

Exp 821 MUST gate on Exp 819 honest_verdict == "injection_field_fixed" before running
any live session.  If the gate fails, the experiment writes a blocked artifact with
honest_verdict="blocked_gate" and exits immediately.

Rationale: compute_energy_with_external_field() was validated in Exp 819 to discriminate
violations from correct responses (discrimination_rate=1.0).  Without that validation,
the precision delta measured in Exp 821 could be entirely due to the coupling matrix
rather than the constraint injection signal.

Spec: REQ-LEARN-821-001

### REQ-LEARN-821-002: Constraint Addition MUST Produce Measurable Precision Delta Over 3 Sessions

After applying the external field fix (Exp 819), constraint addition to EmbeddingConstraintStore
MUST produce a measurable increase in verification precision (delta_overall > 0) over at least
3 live sessions on >= 30 GSM8K questions.

Definition:
  - precision_s = fraction of violation/correct pairs where E_viol > E_corr using external field
  - delta_s1_to_s3 = precision[2] - precision[0]
  - delta_overall = max(precision[0..2]) - precision[0]
  - retro_constraint_zero_delta_closed = (delta_overall > 0)

This is the core FR-11 Tier 1 hypothesis: more constraints → higher energy for violations.
The external field fix (Exp 819) enables real measurement for the first time.

Spec: REQ-LEARN-821-002

### SCENARIO-LEARN-821-001: 30q x 3 Sessions — External Field Constraint Accumulation

Given Exp 819 honest_verdict == "injection_field_fixed":
  - Exp 821 creates an empty EmbeddingConstraintStore
  - Session 1: store is empty; external field h≈0; precision is coupling-only baseline
  - After session 1: SPO constraints added for each failed question (n_added_1 > 0)
  - Session 2: store has n_added_1 constraints; h>0 for related questions; precision >= session 1
  - After session 2: SPO constraints added for remaining failures (n_added_2 >= 0)
  - Session 3: store has n_added_1 + n_added_2 constraints; precision >= session 2
  - delta_overall = max(precision_1..3) - precision_1 >= 0
  - honest_verdict = "constraint_addition_works_live" if delta_overall > 0
  - honest_verdict = "constraint_addition_no_delta_live" if delta_overall <= 0
  - honest_verdict = "blocked_gate" if Exp 819 gate fails

### REQ-LEARN-824-001: JEPA Training Corpus MUST Use LIMO-Style Curation

After 11 consecutive JEPA OOD AUC failures (Exps v13-v22), random or all-pairs training
is prohibited.  The training corpus MUST be curated using the LIMO principle
(arXiv 2402.09353): select top-50 pairs by z3_confidence × cpmi_score from the full
FoVer + CPMI corpus.  Low-quality pairs with z3_confidence < 0.9 or cpmi_score < 0.0
are excluded to reduce noise.

Rationale: LIMO showed 817 curated examples beat 100k random examples for LLM reasoning.
The same principle applies to EBM training data: quality beats quantity.

Spec: REQ-LEARN-824-001

### REQ-LEARN-824-002: JEPA Training Corpus MUST Include Domain Diversity

Single-domain training (GSM8K only) causes systematic OOD failures because the model
overfits to GSM8K surface patterns that do not generalise.  The corpus MUST include
>= 10 HumanEval code reasoning pairs and >= 10 SVAMP arithmetic pairs.

Target composition: top-50 GSM8K + 10 HumanEval + 10 SVAMP = 70 pairs minimum.

Spec: REQ-LEARN-824-002

### REQ-LEARN-824-003: JEPA v23+ MUST Use Contrastive Triplet Loss

Binary BCE loss is deprecated after 11 consecutive retrain failures (Exps v13-v22).
JEPA v23 MUST use contrastive triplet loss with margin=0.5:
  L = max(0, d(anchor, positive) - d(anchor, negative) + margin)
where d is cosine distance in embedding space.

Hard negatives come from CPMI triples where cpmi_score is high (the hardest cases
for the model to distinguish).

Spec: REQ-LEARN-824-003

### SCENARIO-LEARN-824-001: LIMO Curation + Triplet Loss + Domain Diversity

Given FoVer + CPMI corpus available:
  - LIMOCurator selects top-50 GSM8K pairs + 10 HumanEval + 10 SVAMP = 70 total
  - JEPAv23 trains for 100 epochs with triplet margin loss (margin=0.5)
  - OOD evaluated on fover_labeled_steps_live.json (57 held-out steps)
  - Target: ood_auc >= 0.65 (jepa_v23_viable)
  - Baseline: ood_auc = 0.50 (v22 ceiling)
  - honest_verdict = "jepa_v23_viable" if ood_auc >= 0.65
  - honest_verdict = "jepa_v23_improvement" if 0.50 <= ood_auc < 0.65
  - honest_verdict = "jepa_v23_below_random" if ood_auc < 0.50

### REQ-LEARN-051: JEPA v23 MUST Be Evaluated on 3 Domains Before Tier 3.5 Deployment

After confirming jepa_v23_viable gate (Exp 824 honest_verdict in ["jepa_v23_viable",
"jepa_v23_improvement"]), JEPA v23 MUST be evaluated on 3 domains:
  - GSM8K (in-distribution): 20 CoT step sequences, auc_gsm8k computed.
  - HumanEval code steps (OOD): 10 function implementations as step-level traces, auc_humaneval.
  - ARC-Challenge planning (OOD): 10 multi-step reasoning problems, auc_arc.

overall_ood_auc = mean(auc_humaneval, auc_arc).  GSM8K is treated as in-distribution
because the JEPA v23 training corpus contained GSM8K pairs.

If overall_ood_auc >= 0.65, JEPA v23 MUST be deployed as ThreeTierPipeline Tier 3.5
by setting the `tier_35` attribute on the pipeline instance, satisfying FR-11 Tier 3
requirement (Exp 825).

Spec: REQ-LEARN-051

### REQ-LEARN-052: VerificationCertificate MUST Be Emitted Per Step When Tier 3.5 Is Active

When JEPA v23 Tier 3.5 is active, each step's verification result MUST include a
VerificationCertificate namedtuple with fields:
  (step_id, jepa_energy_delta, constraint_type, z3_verdict, confidence_score)

Format based on arXiv 2601.17223 (Beyond Outcome Verification / Verifiable PRM design).
At least 20 VerificationCertificates MUST be emitted and stored in the experiment artifact.

  - step_id: str identifier for the evaluated step (domain + index)
  - jepa_energy_delta: float cosine distance from JEPAv23Predictor.predict_energy()
  - constraint_type: str category ("arithmetic", "code_logic", "planning")
  - z3_verdict: str formal verdict ("sat", "unsat", "unknown")
  - confidence_score: float in [0, 1] derived from energy_delta

Spec: REQ-LEARN-052

### SCENARIO-LEARN-061: 3-Domain Eval with Tier 3.5 Deployment and VerificationCertificates

Given JEPA v23 model loaded from Exp 824 checkpoint:
  - Evaluate on GSM8K (20 steps), HumanEval (10 steps), ARC-Challenge (10 steps)
  - overall_ood_auc = mean(auc_humaneval, auc_arc) is computed
  - If overall_ood_auc >= 0.65: Tier 3.5 deployed (ThreeTierPipeline.tier_35 set)
  - 20 VerificationCertificates emitted (one per randomly selected evaluated step)
  - honest_verdict = "jepa_v23_tier35_deployed" when tier35_deployed=True
  - honest_verdict = "jepa_v23_improvement_not_deployed" when 0.50 <= overall_ood_auc < 0.65
  - honest_verdict = "blocked_gate" when Exp 824 honest_verdict is jepa_v23_below_random

Spec: SCENARIO-LEARN-061

### REQ-LEARN-048: VerifyRepairPipeline MUST Call EmbeddingConstraintStore.store() on Violations

When `enable_constraint_accumulation=True` (or when `embedding_constraint_store` is provided
to `verify()`), `VerifyRepairPipeline.verify()` MUST call `EmbeddingConstraintStore.store()`
with a `ConstraintSPOTuple` encoding each detected violation.

Rationale: Exp 821 showed delta_overall=0.0 across 3 sessions of 30 GSM8K questions despite
the external field injection fix from Exp 819.  Root-cause diagnosis (Exp 833) confirmed that
`verify()` calls `retrieve()` but NEVER calls `store()` — so the constraint store is always
empty and retrieval always returns zero constraints.  The write path is structurally absent.

Spec: REQ-LEARN-048

### REQ-LEARN-049: VerifyRepairPipeline MUST Route Energy Computation Through compute_energy_with_external_field

When an `IsingConstraintInjector` is provided to `verify()`, energy computation MUST be
routed through `IsingConstraintInjector.compute_energy_with_external_field()` rather than
the legacy `compute_energy_with_injection()` path.

Rationale: Exp 833 confirmed that even when `ising_constraint_injector` is provided,
`compute_energy_with_external_field` is never called.  Only `project_to_spin_bias()` is
called for logging — the bias is computed but never applied to the actual energy.  The
legacy diagonal injection path (which adds a constant shift indistinguishable across
all spin configs) was already diagnosed as non-discriminating in RETRO-ISING-INJECTION-
NO-DISCRIMINATION (Exp 812).

Spec: REQ-LEARN-049

### SCENARIO-LEARN-060: Pipeline Write Path — 2 Violations Trigger 2 Store Writes

Given a `VerifyRepairPipeline` with `embedding_constraint_store` containing 0 entries:
  - Run `verify()` on a response with 2 known violations
  - After the call, `EmbeddingConstraintStore._store` MUST have length == 2
  - `n_store_write_calls` counter MUST equal 2
  - Each stored entry MUST have a non-None `embedding` field

Spec: SCENARIO-LEARN-060

### REQ-LEARN-834-001: JEPA v24 Training MUST Use Domain-Balanced Corpus with ARC Coverage

JEPA training MUST include at least 10 ARC-Challenge reasoning pairs when ARC was absent
from prior training data.  A corpus assertion `assert n_arc_pairs >= 10` MUST be present
in the corpus builder and MUST raise before any training begins if the constraint is violated.

Domain weights for DG-PRM inference MUST set ARC weight >= 3.0 (three times the GSM8K
baseline) when ARC is not represented in prior training data, to compensate for the
model's historical under-scoring of ARC steps.

Rationale: Exp 832 confirmed JEPA v23 AUC=0.04 on ARC was caused directly by zero ARC
training examples in the LIMO corpus (Exp 824).  Without an assertion, the next corpus
builder could repeat the omission silently.

Spec: REQ-LEARN-834-001

### SCENARIO-LEARN-834-001: JEPA v24 Trained with DG-PRM; ARC AUC Improves from 0.04

Given:
  - Balanced corpus: 20 GSM8K + 20 HumanEval + 20 ARC + 10 SVAMP = 70 pairs
  - DG-PRM domain head with ARC weight = 3.0 at inference
  - DreamPRM per-domain loss weight: arc=5.0, humaneval=1.5, gsm8k=1.0, svamp=1.5
  - ΔEnergy triplet loss weighted by energy gap, clamped to [0.5, 3.0]
  - 200 training epochs, Adam lr=1e-3
Then:
  - auc_arc > 0.04 (improvement over JEPA v23 baseline)
  - honest_verdict in {"jepa_v24_domain_balanced", "jepa_v24_improvement",
                       "jepa_v24_arc_improved", "jepa_v24_still_unbalanced"}
  - Results written to results/experiment_834_jepa_v24_dg_prm.json

Target: min_domain_auc > 0.55 → honest_verdict = "jepa_v24_domain_balanced"

Spec: SCENARIO-LEARN-834-001

### SCENARIO-LEARN-836-001: Constraint Accumulation Fix v3 — Write Path Validated

Given a `VerifyRepairPipeline` constructed with `enable_constraint_accumulation=True`:
  - Session 1: empty `EmbeddingConstraintStore`, 30 GSM8K questions verified
    - `n_constraints_written_s1 >= 1` (at least one violation written to store)
  - Session 2: store has entries from session 1, same 30 questions re-verified
    - `n_constraints_written_s2 >= 0` (additional violations written)
    - Retrieved constraints from session 1 are injected into verification
  - Session 3: store has entries from sessions 1+2, same 30 questions re-verified
  - `delta_overall = max(precision_s1, precision_s2, precision_s3) - precision_s1`
  - `honest_verdict` in {"constraint_accumulation_fixed", "write_path_fixed_no_delta",
    "still_delta_zero", "blocked_no_diagnosis"}

Root cause confirmed by Exp 833: `write_path_missing` (H1).
Fix: `VerifyRepairPipeline.verify()` now calls `embedding_constraint_store.store(spo)`
for each violation when `enable_constraint_accumulation=True`.

Spec: REQ-LEARN-048, REQ-LEARN-049, SCENARIO-LEARN-060, SCENARIO-LEARN-836-001


---

## REQ-INFRA-072: ExclusionManifestEnforcer

**REQ-INFRA-072**: ExclusionManifestEnforcer MUST prevent retired experiments from
re-entering the conductor queue by writing gate entries to MILESTONE_PREREQS.md.
The enforcer reads ops/exclusion_manifest.yaml (YAML authority) and writes a
dated "## Exclusion Manifest Gate" section.  The conductor reads MILESTONE_PREREQS.md
at pre-flight; any experiment ID listed in the gate section MUST NOT be launched.

This is the side-channel enforcement path that satisfies RETRO-MANIFEST-FULL-SCOPE
without modifying scripts/research_conductor.py (CLAUDE.md constraint).

Implementation: python/carnot/pipeline/manifest_enforcer.py — ExclusionManifestEnforcer

### SCENARIO-INFRA-081: write_prereqs_section appends Exclusion Manifest Gate

Given ops/exclusion_manifest.yaml with retired experiment IDs (e.g. 527, 491, 603):
  - ExclusionManifestEnforcer().load_manifest("ops/exclusion_manifest.yaml") succeeds
  - write_prereqs_section("MILESTONE_PREREQS.md") appends a section containing
    "## Exclusion Manifest Gate" with a table of all retired IDs
  - is_retired(527) returns True
  - is_retired(999) returns False
  - check_queue([527, 999]) returns [527]
  - manifest_enforcer_deployed: true appears in the appended section

Spec: REQ-INFRA-072, SCENARIO-INFRA-081 (Exp 868)


---

## REQ-INFRA-074: GGUFCacheResolver CLI Download Fallback

**REQ-INFRA-074**: GGUFCacheResolver MUST provide a ``cli_download()`` method that
invokes the ``hf`` or ``huggingface-cli`` binary via subprocess as an alternative to
the Python API.  This path is required because hf_hub_download() raised
RepositoryNotFoundError on 11 consecutive attempts (Exps 857–869) despite the repo
existing.  The CLI method MUST:

1. Detect ``hf`` or ``huggingface-cli`` on PATH; return ``{"success": False, "error":
   "hf CLI not found: ..."}`` if absent rather than raising.
2. Run ``[hf_cmd, "download", hf_repo, filename, "--local-dir", dest_dir]`` with
   a configurable ``timeout_s`` (default 300 s).
3. Verify the downloaded file exists on disk and return
   ``{"success": True, "path": str, "size_mb": float}`` on success.
4. Return ``{"success": False, "error": str}`` on non-zero returncode or timeout
   (never raise from this method — callers need to distinguish outcomes).

A companion method ``resolve_with_cli_fallback()`` MUST try the local HF cache
first and only invoke ``cli_download()`` on a cache miss.

Implementation: ``python/carnot/resolvers/gguf_cache.py`` — GGUFCacheResolver

### SCENARIO-INFRA-083: CLI Download on Valid Repo Returns success=True

Given a ``GGUFCacheResolver`` instance and a valid HuggingFace repo with a known
GGUF file:
  - ``cli_download(hf_repo, filename, dest_dir, timeout_s=300)`` returns a dict
    where ``success == True``, ``"path"`` key exists, and ``"size_mb" > 0``
  - The returned ``path`` points to a real file on disk
  - ``resolve_with_cli_fallback()`` on the same inputs returns a ``pathlib.Path``
    that ``exists()``
  - When the ``hf`` binary is absent from PATH, ``cli_download()`` returns
    ``{"success": False, "error": ...}`` rather than raising

Spec: REQ-INFRA-074, SCENARIO-INFRA-083 (Exp 890)


---

## REQ-INFRA-075: Roadmap Gate and Prior-Failures Pre-Conductor Audit

**REQ-INFRA-075**: A standalone roadmap audit script MUST validate a
new research roadmap before conductor dispatch without modifying
`scripts/research_conductor.py`. The audit MUST report structured counts
and line-item failure details for:

1. `GATE_UPSTREAM_EXISTS`: every `gated_on[].upstream` task id must exist
   in the same roadmap YAML.
2. `PRIOR_FAILURES_COVERAGE`: every roadmap task must be compared against
   `research-complete.yaml`; when its title shares at least two
   substantive scope keywords with any prior task, the roadmap task must
   contain a non-empty `prior_failures` field.
3. `MODEL_AGENT_COHERENCE`: `agent_type: codex` tasks must use
   `model: gpt-5.5`, and `agent_type: gemini` must be rejected.
4. `GATE_FIELD_CROSS_REF`: each `gated_on[].artifact_field` must appear in
   the upstream task prompt's `REQUIRED ARTIFACT FIELDS:` section.

The audit result MUST include the required Exp 1140 artifact fields:
`n_tasks_audited`, gate upstream counts, prior-failure counts,
model-agent coherence failure count, gate-field cross-reference failure
count, pass/fail status, failure details, `audit_script_written`, and
`honest_verdict`.

### SCENARIO-INFRA-084: Roadmap Audit Flags Missing Upstream and Artifact Field

Given a roadmap task gated on an unknown upstream id, the audit reports a
`GATE_UPSTREAM_EXISTS` failure. Given a gate that references an upstream
task but names an artifact field absent from that upstream prompt's
`REQUIRED ARTIFACT FIELDS:` section, the audit reports a
`GATE_FIELD_CROSS_REF` failure.

### SCENARIO-INFRA-085: Roadmap Audit Flags Missing Prior Failures

Given a new roadmap task whose title shares at least two substantive
scope keywords with a task in `research-complete.yaml`, the audit reports
a `PRIOR_FAILURES_COVERAGE` failure unless the new task has a non-empty
`prior_failures` field.

### SCENARIO-INFRA-086: Roadmap Audit Flags Unsupported Agent Routing

Given a roadmap task with `agent_type: codex` and a model other than
`gpt-5.5`, or any roadmap task with `agent_type: gemini`, the audit
reports a `MODEL_AGENT_COHERENCE` failure.

### SCENARIO-INFRA-087: Pre-Activation Audit Artifact Records ArXiv Prior Coverage

Given a planned milestone roadmap, the Exp 1152 pre-activation runner MUST
execute the roadmap audit before conductor activation and write a JSON
artifact containing the audit counts, failure details with fix guidance,
`roadmap_gate_audit_passed`, and an
`arxiv_task_prior_failures_complete` boolean. That boolean MUST be true
only when task `exp1153-arxiv-final-submission-v4` declares prior
failures for `exp1139-arxiv-final-submission-v3`,
`exp1127-arxiv-pdf-compilation-final-submission`, and
`exp1116-arxiv-pdf-compilation-submission`.

Spec: REQ-INFRA-075, SCENARIO-INFRA-084, SCENARIO-INFRA-085, SCENARIO-INFRA-086, SCENARIO-INFRA-087 (Exp 1140, Exp 1152)

---

## REQ-INFRA-076: Pytest RSS Memory Watchdog

**REQ-INFRA-076**: The Python pytest suite MUST install a memory watchdog from
`tests/python/conftest.py` that records process `ru_maxrss` before and after each
test. The watchdog MUST:

1. Record a per-test RSS baseline during `pytest_runtest_setup(item)`.
2. Record RSS again during `pytest_runtest_teardown(item, nextitem)` and compute
   a per-test delta in megabytes.
3. Fail any test whose RSS delta exceeds 500 MB with a message beginning
   `Memory leak: +`.
4. Track cumulative positive RSS deltas across the pytest session.
5. At session finish, when cumulative RSS growth exceeds 8192 MB, emit a warning
   that includes the top five per-test RSS deltas and write a
   `results/pytest_memory_{timestamp}.log` file.
6. Permit an explicit `memory_watchdog_skip` marker for known high-RSS
   compiler/cache smoke tests where `ru_maxrss` high-water growth is expected;
   marked tests still record their RSS sample for session diagnostics.

### SCENARIO-INFRA-088: Runaway Test Is Failed Before Session OOM

Given a pytest test that retains more than 500 MB of additional resident memory
after its setup baseline is recorded, the teardown watchdog fails that test with
`Memory leak: +{delta}MB` and keeps the remaining pytest session bounded by
recording the offender for the session summary.

### SCENARIO-INFRA-089: Session Growth Emits Operator Artifact

Given a pytest session whose cumulative positive RSS deltas exceed 8192 MB, the
watchdog emits a warning naming the top five RSS-delta tests and writes a
timestamped `results/pytest_memory_*.log` artifact for conductor diagnostics.

## REQ-INFRA-077: Pytest Address-Space Memory Cap

**REQ-INFRA-077**: The Python pytest suite MUST set a process address-space
limit from `tests/python/conftest.py` during `pytest_configure` before any test
item runs. The soft `RLIMIT_AS` limit MUST be capped at 32 GB while preserving
the kernel hard limit, and unsupported-kernel failures MUST warn and continue.

### SCENARIO-INFRA-090: Address-Space Cap Is Active During Tests

Given a Python pytest run using the repository `tests/python/conftest.py`, test
items observe a finite `RLIMIT_AS` soft limit no larger than 32 GB.

### SCENARIO-INFRA-091: Address-Space Cap Preserves Existing Test Imports

Given the address-space cap is active for pytest, normal imports of the packaged
`carnot` module continue to work alongside the existing RSS watchdog plugin.

## REQ-INFRA-078: Roadmap Prior-Failures Autofill

**REQ-INFRA-078**: A standalone `scripts/conductor_priors_autofill.py` command
MUST populate missing roadmap `prior_failures` blocks from the local
`FailureLedger` without modifying `scripts/research_conductor.py`. The command
MUST:

1. Load `research-roadmap-next.yaml` by default, falling back to the active
   `research-roadmap.yaml` when the next roadmap is not present.
2. Scan every task and skip tasks that already have a non-empty
   `prior_failures` list.
3. Query `FailureLedger.matching_priors()` for each unpopulated task using the
   task id and title.
4. Classify matching priors whose verdict contains reconciler partial/failed
   tokens as `true_failure`, and classify all other priors as
   `successful_upstream`.
5. Insert only the missing `prior_failures` block while preserving existing
   roadmap content.
6. Support `--dry-run`, which reports counts without writing file changes.

### SCENARIO-INFRA-092: Autofill Skips Populated Tasks

Given a roadmap task with a non-empty `prior_failures` list, the autofill
command counts it as already populated and does not alter that task's block.

### SCENARIO-INFRA-093: Autofill Classifies Successful Upstreams

Given a matching ledger prior whose verdict does not contain any reconciler
partial or failed token, the generated prior-failure stub classifies it as
`successful_upstream` and uses an automatic explanatory `addressed_by` value.

### SCENARIO-INFRA-094: Autofill Dry Run Does Not Write

Given `--dry-run`, the autofill command reports tasks scanned, generated stubs,
and already-populated tasks while leaving the roadmap file byte-for-byte
unchanged.

## REQ-INFRA-1296: Prior-Failures Activation Audit Artifact

**REQ-INFRA-1296**: The Exp 1296 activation audit runner MUST write
`results/experiment_1296_prior_failures_activation_audit.json` without modifying
`scripts/research_conductor.py` or `research-roadmap.yaml`. The runner MUST:

1. Write an in-progress artifact before running the terminal audit.
2. Request `research-roadmap-next.yaml` and fall back to the active
   `research-roadmap.yaml` when the next-roadmap handoff file is absent.
3. Import the prior-failures validator and roadmap gate auditor, capturing
   schema errors, prior-failure findings, gate upstream failures, and gate
   artifact-field cross-reference failures.
4. Read Exp 1283 and Exp 1288 artifacts and expose
   `exp1283_grammar_backend_available` and
   `exp1288_memory_update_written` only when the source artifact says the
   corresponding boolean is true.
5. Emit `activation_blockers` entries with exact task ids, fields, and raw
   details for any audit failure.
6. Finish with `status="complete"` and `honest_verdict` equal to
   `activation_audit_passed` only when both imported audits pass.

### SCENARIO-INFRA-1296: Activation Audit Passes On Clean Active Roadmap Fallback

Given `research-roadmap-next.yaml` is absent and the active `.101` roadmap
passes prior-failure and gate audits, the Exp 1296 artifact records the active
roadmap as the audited path, sets both audit booleans true, preserves the Exp
1283/1288 proxy booleans, and reports `activation_audit_passed`.

### SCENARIO-INFRA-1296-BLOCKED: Activation Audit Reports Exact Blockers

Given a planned roadmap with missing prior-failure metadata or a gate that
references an artifact field absent from its upstream prompt, the Exp 1296
artifact sets the relevant audit booleans false, reports the failure counts, and
adds `activation_blockers` entries naming the affected task id and field.

## REQ-INFRA-1337: Environment Gate Disk, Pre-Test, Stale Skeleton, and Roadmap Audit Artifact

**REQ-INFRA-1337**: The Exp 1337 environment gate runner MUST write
`results/experiment_1337_environment_gate_disk_pretest_stale_skeleton_audit.json`
without deleting local files, modifying `.103` result artifacts, changing
`research-roadmap.yaml`, or editing `scripts/research_conductor.py`. The runner
MUST:

1. Write an in-progress artifact before collecting measurements.
2. Record filesystem free gigabytes and inode-free percentage for the project
   root, and set `disk_quota_ok` from a deterministic minimum-free-space gate.
3. Extract the .103 disk-quota and repeated pre-test signatures from the
   conductor log and operational retrospective instead of inventing a new
   failure description.
4. Classify .103 result artifacts whose JSON status is `in_progress` or whose
   contents are bootstrap-only skeletons as `stale_artifact_paths`, with
   `stale_skeleton_count` matching that list length.
5. Run the focused pre-test only when its command target exists; otherwise write
   `focused_pretest_status="not_available"` with the missing path.
6. Run the prior-failures validator and roadmap-gate auditor when available and
   summarize actionable failures without modifying the roadmap.
7. Set `environment_ready` true only when the disk gate passes, no repeated
   focused pre-test signature is active, and stale artifacts have been
   explicitly classified rather than silently reused.
8. Finish with all required Exp 1337 fields, `status="complete"`, and an
   `honest_verdict` that states whether .104 scientific work may proceed.

### SCENARIO-INFRA-1337: Environment Gate Blocks Reuse of Stale .103 Skeletons

Given .103 artifacts include an `in_progress` or bootstrap-only JSON artifact and
the focused pre-test is unavailable, the Exp 1337 gate artifact lists the stale
path, records `focused_pretest_status="not_available"`, sets
`environment_ready=false` when the disk gate or pre-test signature remains
unsafe, and recommends pruning blocked .104 work until the environment gate is
fixed.

### REQ-RETRO-1784: Milestone 1784 Operational Retrospective

**Requirement:** The Exp 1784 retrospective runner MUST aggregate `honest_verdict` from
experiments 1771 through 1783 into `results/experiment_1784_retro.json`.

Spec: REQ-RETRO-1784

### SCENARIO-RETRO-1784: Exp 1784 Aggregates Results

**Given** results for experiments 1771 to 1783 exist,

**When** the retrospective runner executes,

**Then** it writes `experiment_1784_retro.json` with an aggregated view of the milestone.

Spec: SCENARIO-RETRO-1784

### REQ-RETRO-1811: Milestone 1811 Operational Retrospective

**Requirement:** The Exp 1811 retrospective runner MUST aggregate `honest_verdict` and key metrics from
experiments 1799 through 1810 into `results/experiment_1811_retro.json`.

Spec: REQ-RETRO-1811

### SCENARIO-RETRO-1811: Exp 1811 Aggregates Phase-16 Results

**Given** results for experiments 1799 to 1810 exist,

**When** the retrospective runner executes,

**Then** it writes `experiment_1811_retro.json` with an aggregated view of the Phase-16 milestone.

Spec: SCENARIO-RETRO-1811

### REQ-AUTO-1904: Activation Contract for Milestone 2026.05.149
The system shall generate an activation contract for Milestone 2026.05.149 that explicitly asserts baseline readiness states as false.
The required fields are: status, honest_verdict, milestone_148_archived, live_sota_blocked_missing_models, telemetry_missing_terminal_artifact, next_gate_contract_ready, and tests_run.

### SCENARIO-AUTO-1904: Activation Contract Validation
**Given** the artifact generator for experiment 1904
**When** the activation contract is produced
**Then** all required baseline readiness fields are present and initialized appropriately.

### REQ-RETRO-176: Milestone 2026.05.176 Operational Retrospective
The system MUST generate an operational retrospective for milestone 2026.05.176 following the `carnot.operational_retro.v64` schema.

Spec: REQ-RETRO-176

### SCENARIO-RETRO-176: Generation of 2026.05.176 Retrospective
**Given** the completion of milestone 2026.05.176,
**When** the retro generation task runs,
**Then** it MUST output `results/operational_retro_2026_05_176.json` containing the appropriate performance metrics, preconditions_checked, and an honest_verdict.

Spec: SCENARIO-RETRO-176

### REQ-RETRO-181: Milestone 2026.05.181 Operational Retrospective
The system MUST generate an operational retrospective for milestone 2026.05.181 following the `carnot.operational_retro.v64` schema.

Spec: REQ-RETRO-181

### SCENARIO-RETRO-181: Generation of 2026.05.181 Retrospective
**Given** the completion of milestone 2026.05.181,
**When** the retro generation task runs,
**Then** it MUST output `results/operational_retro_2026_05_181.json` containing the appropriate performance metrics, preconditions_checked, and an honest_verdict.

Spec: SCENARIO-RETRO-181

### REQ-RETRO-187: Milestone 2026.05.187 Retrospective
The system SHALL generate a retrospective JSON artifact for milestone 2026.05.187.

### SCENARIO-RETRO-187: Validate 187 Retro
GIVEN the 187 retrospective is generated
WHEN the artifact is parsed
THEN it contains the required honest_verdict and schema fields.

### REQ-RETRO-198: Milestone 2026.05.198 Operational Retrospective
The system MUST generate an operational retrospective for milestone 2026.05.198 following the `carnot.operational_retro.v64` schema.

Spec: REQ-RETRO-198

### SCENARIO-RETRO-198: Generation of 2026.05.198 Retrospective

**When** the retro generation task runs,
**Then** it MUST output `results/operational_retro_2026_05_198.json` containing the appropriate performance metrics, preconditions_checked, and an honest_verdict.

Spec: SCENARIO-RETRO-198

### REQ-RETRO-199: Milestone 2026.05.199 Operational Retrospective
The system MUST generate an operational retrospective for milestone 2026.05.199 following the `carnot.operational_retro.v64` schema.

Spec: REQ-RETRO-199

### SCENARIO-RETRO-199: Generation of 2026.05.199 Retrospective

**When** the retro generation task runs,
**Then** it MUST output `results/operational_retro_2026_05_199.json` containing the appropriate performance metrics, preconditions_checked, and an honest_verdict.

Spec: SCENARIO-RETRO-199

### REQ-RETRO-200: Milestone 2026.05.200 Operational Retrospective
The system MUST generate an operational retrospective for milestone 2026.05.200 following the `carnot.operational_retro.v64` schema.

Spec: REQ-RETRO-200

### SCENARIO-RETRO-200: Generation of 2026.05.200 Retrospective

**When** the retro generation task runs,
**Then** it MUST output `results/operational_retro_2026_05_200.json` containing the appropriate performance metrics, preconditions_checked, meta_reflection, and an honest_verdict.

Spec: SCENARIO-RETRO-200

### REQ-AUTO-SWEEP-2013: Routine Citation Sweep
The system shall execute a routine citation sweep to discover relevant papers and deduplicate against the known research queue.

### REQ-AUTO-3391: Milestone 313 Planning
The system shall generate a planning artifact for milestone 313 containing tasks proposed based on milestone 312.

### SCENARIO-AUTO-3391: Generation of Milestone 313 Plan
**Given** milestone 312 results,
**When** the milestone 313 planning task runs,
**Then** it outputs `results/experiment_3391_plan_milestone_313.json` with required schema fields.


### REQ-LEARN-3401: FR-11 Continuous Learning End-to-End Stress Test

**Requirement:**
    The system MUST validate FR-11 end-to-end with the new Hopfield and CAS methods by running a continuous learning stress test simulating 1000 user interactions.
    The script MUST:
      1. Setup a simulated interaction loop.
      2. Apply CAS updates and Energy-based replay.
      3. Evaluate the final constraint fidelity.
    The script MUST output a JSON deliverable to `results/experiment_3401_fr11_stress.json`.

**Rationale:**
    Continuous learning requires stress testing under sustained updates to ensure constraint fidelity is maintained when subjected to CAS updates and Energy-based replay over 1000 interactions.

Spec: REQ-LEARN-3401, SCENARIO-LEARN-3401

### SCENARIO-LEARN-3401: FR-11 Stress Test Completion

**Given** a simulated interaction loop of 1000 interactions
**When** CAS updates and Energy-based replay are applied
**Then** the script evaluates final constraint fidelity
**And** produces `results/experiment_3401_fr11_stress.json` containing the results.

### REQ-AR-050: P0.1 Difficulty-Matched Corpus Builder v3 — Adaptive Level Selection + Process Traces

**Requirement:**
The system MUST build a MATH-500 difficulty-matched corpus for the P0.1 energy-selection experiment (exp3496) with:
1. An adaptive warm-up that probes SC per candidate MATH level (3, 4, 3+4 mix) and selects the level(s) whose SC lands in the headroom band [0.40, 0.70].
2. Resume semantics: re-invocations skip already-completed problem IDs and add only new rows.
3. Per-problem output: 1 greedy (temp=0) + k=6 sampled (temp=0.8) solutions, extracted `\boxed{}` answer, correctness label, and parsed reasoning steps.
4. Checkpointing: one JSONL row appended per completed problem immediately (fsync'd) so kills lose at most one row.
5. Terminal verdicts starting with `complete:`, encoding the corpus-size band (headline-eligible >=80, scorable-partial 40-79, partial <40, or blocked if no in-band split found).

**Rationale:**
P0.1 is the existential question: "does energy-based selection beat majority-vote at equal compute?" It is only testable when the corpus's self-consistency lands in [0.40, 0.70] — the headroom window where both floor and ceiling are non-degenerate. MATH-500 levels 3-4 is the identified candidate; the adaptive warm-up ensures the chosen level actually lands in band rather than assuming it.

**Scenarios:**
- SCENARIO-AR-050-01: Final-answer extraction and normalization — `\boxed{}` content extracted from the last occurrence; LaTeX wrappers stripped for comparison.
- SCENARIO-AR-050-02: Self-consistency band classification — majority-vote SC computed over sampled answers; in_headroom_band returns True iff SC in [0.40, 0.70].
- SCENARIO-AR-050-03: Per-step reasoning traces captured — each generation row carries a parsed list of reasoning paragraphs.
- SCENARIO-AR-050-04: Resume skips completed problems — completed_problem_ids reads the JSONL and returns the set of problem_id values already written.

### REQ-AUTO-015: P0.1 Route 2 Headroom Corpus (Greedy-Wrong Construction)

The system shall construct a selectable-headroom corpus for Route 2 evaluation by intentionally selecting for problems where the greedy (temp=0) answer is incorrect, but at least one of $k \ge 16$ sampled candidates is correct. This guarantees headroom by construction, meaning the oracle accuracy strictly exceeds both greedy and self-consistency (majority-vote) accuracy.

**Rationale:**
Previous attempts to build a headroom corpus via difficulty filtering (MATH L4-5) yielded near-optimal self-consistency (SC) accuracy, where the correct answer was usually the most likely outcome, leaving no headroom for energy-based selection to improve upon. By filtering for "greedy-wrong and recoverable" problems, the correct answer is guaranteed to be a minority among the samples, creating a strict gap where oracle > SC.

**Scenarios:**
- SCENARIO-AR-051-01: Headroom condition correctly identified — `has_selectable_headroom(record)` returns True iff the greedy answer is wrong AND at least one sampled answer is correct.
- SCENARIO-AR-051-02: Oracle strictly exceeds SC — `compute_corpus_stats` returns `oracle_exceeds_sc = True` when the oracle accuracy is strictly greater than SC accuracy.
- SCENARIO-AR-051-03: Artifact reports required headroom bounds — the deliverable JSON captures greedy, SC, and oracle accuracies, and asserts selectable_headroom > 0.

### REQ-AR-051: P0.1 Route-2 NL-Math Final Headroom or Retire

**Requirement:**
The autoresearch harness MUST pull harder competition-grade problems (AIME / MATH level-5) to construct a greedy-wrong headroom corpus where oracle > SC by construction. If such a corpus is built (n >= 40), it MUST score the multi-verifier Weaver/BoN-MAV combination (along with energy reranker and STRONG SC) and compute significance. If no headroom corpus can be built even from harder problems, it MUST emit a terminal negative 'permanently retired' verdict.

#### SCENARIO-AR-051-01: Terminal Verdict Emission
**Given** the Route-2 NL-math final experiment is executed
**When** a headroom corpus is built and scored, or cannot be built
**Then** it MUST output a valid JSON artifact with 'route2_nlmath_terminal' verdict and 'multi_verifier_accuracy'.

### REQ-AR-052: P0.1 Headroom Hybrid Verifier-vs-SC Positive Control

**Requirement:**
The autoresearch harness MUST provide a cached-candidate verifier-vs-self-consistency
experiment (`scripts/experiment_3645_headroom_hybrid_verifier_vs_sc_v3.py`) that:
1. Loads a multi-candidate corpus with per-candidate correctness labels without
   invoking a live LLM.
2. Selects a self-consistency-contested stratum whose oracle best-of-N accuracy
   strictly exceeds the self-consistency majority-vote accuracy.
3. Reports `oracle_minus_sc_headroom`, `sc_accuracy`,
   `verifier_reranked_accuracy`, `verifier_over_sc_lift`, `hybrid_accuracy`,
   `hybrid_beats_both`, `verifier_beats_sc_where_headroom_exists`,
   `n_examples`, `random_seed`, `reproducibility_checksum`, and `duration_s`.
4. Emits one of the terminal verdicts:
   `complete: verifier_beats_sc_on_headroom_corpus_hybrid_wins_under_budget`,
   `complete: verifier_does_not_beat_sc_even_with_headroom_selection_value_weak`,
   `complete: no_headroom_corpus_found_verifier_study_uninformative`, or
   `complete: blocked_no_multicandidate_corpus`.

**Rationale:**
Exp 3507 was an honest negative because oracle and self-consistency were equal,
leaving no selectable headroom. This experiment is the complementary positive
control: it first measures oracle > SC on cached candidates, then measures whether
the FoVer verifier ensemble and a verifier+SC hybrid add selection value at the
same candidate budget.

#### SCENARIO-AR-052-01: Headroom Gate Before Verifier Verdict
**Given** cached best-of-N rows with per-candidate correctness labels
**When** the experiment selects a contested stratum
**Then** `oracle_minus_sc_headroom` MUST be present and strictly positive before
any verifier-vs-SC positive verdict is emitted.

#### SCENARIO-AR-052-02: Hybrid and Verifier Lift Fields
**Given** a headroom-bearing stratum
**When** the verifier ensemble and verifier+SC hybrid are scored
**Then** the artifact MUST include paired lift estimates with confidence
intervals and the `hybrid_beats_both` and
`verifier_beats_sc_where_headroom_exists` booleans.

### REQ-AR-053: P0.1 Matched-Compute FLOP Accounting Harness

**Requirement:**
The autoresearch harness MUST provide a matched-compute evaluation instrument
(`scripts/experiment_3727_matched_compute_eval_harness.py`) that:
1. Computes transparent inference FLOP estimates for EBT energy descent and AR
   best-of-M generation using the documented model `parameter_count * sequence_tokens * forward_passes`.
2. Counts an EBT prediction as one initial sequence pass plus `K`
   energy-descent sequence passes, so energy descent cannot hide extra forward
   passes behind a matched-parameter comparison.
3. Tunes the AR best-of-M count to match the EBT total FLOP budget within an
   explicit relative tolerance before comparing held-out accuracy.
4. Reports the EBT and AR held-out accuracies, matched budgets, chosen AR
   best-of-M count, tolerance, random seed, checksum, and terminal artifact
   fields without invoking live model inference.

**Rationale:**
The P0.1 comparison is only decisive when "matched compute" means equal
inference FLOPs, not merely equal parameter count. Energy descent performs
multiple forward passes per prediction; without explicit accounting, an EBT win
can be a disguised best-of-N compute advantage.

**Scenarios:**
- SCENARIO-AR-053-01: Hand-computed FLOPs — toy EBT and AR configurations with
  equal `parameter_count * sequence_tokens * forward_passes` produce identical
  FLOP totals.
- SCENARIO-AR-053-02: Budget matcher — the AR best-of-M tuner selects an
  integer `M` whose total FLOPs are within the configured relative tolerance of
  the EBT budget.
- SCENARIO-AR-053-03: Synthetic matched-compute verdict — deterministic EBT and
  AR fixture generators with known labels return the expected equal-FLOP
  accuracy comparison.

### REQ-LEARN-3697: FR-11 Continuous Self-Learning v12 Drift Reset and Cross-Session Persistence

The FR-11 continuous self-learning experiment v12 MUST run on cached
per-verifier scores and labels without live LLM, GGUF, CUDA, or compute-bound
markers. It MUST simulate at least 200 online updates across a stream with both
recoverable drift and transient/non-recoverable drift. The deploy arm MUST use
drift detection, window-gated reset to the last-known-good dependency structure,
cross-session persistence of that learned structure, and the conservative
default collapse guard. The control arm MUST represent v11 continuous
re-estimation without the reset policy. The artifact MUST report the required
drift, reset, persistence, collapse, quality, non-tautology, checksum, duration,
and terminal-verdict fields.

Spec: REQ-LEARN-3697, SCENARIO-LEARN-3697

### SCENARIO-LEARN-3697: v12 Drift Reset Persists Structure Without Collapse

**Given** FR-11 cached verifier traces with labels and distributionally distinct
slices
**When** Exp 3697 runs the deploy reset arm and the v11 continuous
re-estimation control arm across recoverable and transient drift
**Then** it writes
`results/experiment_3697_fr11_continuous_self_learning_v12.json`
**And** the artifact honestly classifies one of success, no-gain, or blocked
without hard-coding a success verdict
**And** success requires drift detection, transient reset, SHA256 persistence
round-trip, no deploy collapse, distinct pass-rate and true-accuracy arrays,
and maintained ensemble quality.

### REQ-AUTO-015: Advisory Anomaly-Escalation Classifier

The autoresearch system SHALL provide a standalone, deterministic classifier
(`scripts/anomaly_escalation_classifier.py`) that reads an experiment artifact's
`honest_verdict` and optional prior-expectation / kill-gate metadata and returns
one of:

- `clean_bounded_negative`
- `frame_violating_anomaly`
- `clean_positive`

The classifier SHALL treat a negative as `clean_bounded_negative` only when it
matches a declared expected kill-gate, expected negative verdict, or known
bounded lineage. It SHALL treat a result as `frame_violating_anomaly` when a
load-bearing positive control fails, a stated assumption is contradicted, or an
observed direction/magnitude falls materially outside the experiment's declared
prediction envelope. The classifier SHALL emit only an advisory recommendation
and rationale; it MUST NOT prune, edit artifacts, modify the conductor, or
recommend relaxing verification. Any anomaly recommendation SHALL be "pause
pruning and ask a human," preserving human-gated valley funding.

#### SCENARIO-AUTO-012: Anomaly Classifier Separates Bounded Negatives From Frame Violations

**Given** a planned kill-gate artifact whose negative `honest_verdict` matches
its expected bounded-negative metadata
**When** the anomaly-escalation classifier evaluates it
**Then** it classifies the artifact as `clean_bounded_negative` and recommends
standard auto-reconciliation.

**Given** an inconclusive artifact with a load-bearing positive-control failure
**When** the anomaly-escalation classifier evaluates it
**Then** it classifies the artifact as `frame_violating_anomaly` and recommends
halting pruning for human review without relaxing verification.

**Given** a terminal positive artifact with no anomaly signals
**When** the anomaly-escalation classifier evaluates it
**Then** it classifies the artifact as `clean_positive`.

### REQ-AUTO-016: Historical Validation For The Advisory Anomaly Classifier

The autoresearch system SHALL provide a deterministic validation runner
(`scripts/experiment_3791_anomaly_escalation_classifier_validation.py`) that
evaluates the shipped advisory classifier against at least 30 historical
artifacts drawn from `results/operational_retro_*.json`,
`results/experiment_*.json`, and known P1 v1/v2 positive-control failure
artifacts. The runner SHALL label each sample as `clean_bounded_negative`,
`frame_violating_anomaly`, or `clean_positive` by explicit corpus rules, run
the real classifier over those samples, and write
`results/experiment_3791_anomaly_escalation_classifier_validation.json` with a
confusion matrix, false-escalation rate for clean bounded negatives, recall on
known frame-violating anomalies, upstream artifact provenance, deterministic
checksum, and a bare `never_relaxes_verification=true` guarantee. The
validation SHALL remain advisory-only and MUST NOT modify
`scripts/research_conductor.py`.

#### SCENARIO-AUTO-013: Historical Validation Quantifies Advisory Hook Risk

**Given** the historical results corpus and an importable advisory classifier
**When** the Exp 3791 validation runner evaluates the labeled sample
**Then** the output artifact reports at least 30 validation artifacts, includes
a prediction-vs-expected confusion matrix, reports false-escalation rate and
frame-violating recall, cites the upstream artifacts, confirms that no
recommendation relaxes verification, and leaves the conductor unmodified.

### REQ-AUTO-017: Tuned Advisory Anomaly Classifier False-Escalation Control

The autoresearch system SHALL tune the advisory anomaly-escalation classifier
against the same Exp 3791 labeled validation sample before recommending any
operator wiring. The tuned classifier SHALL reduce false escalation of
`clean_bounded_negative` rows to at most 0.2 while preserving recall 1.0 on
the known P1 v1/v2 `frame_violating_anomaly` positive-control failures. The
tuning SHALL remain recommend-only: frame violations may recommend pause plus
human escalation, but the classifier MUST NOT modify the conductor, prune
research state, edit artifacts, or recommend relaxing verification.

#### SCENARIO-AUTO-014: Tuned Classifier Keeps Earned Negatives Clean And P1 Failures Escalated

**Given** the same 32-row Exp 3791 labeled validation sample
**When** the tuned anomaly-escalation classifier re-validates the sample
**Then** the output artifact reports the post-tuning confusion matrix,
false-escalation rate at most 0.2, frame-violating recall 1.0, upstream
artifact provenance, and `never_relaxes_verification=true`.

**Given** an earned bounded or kill-gate negative whose verdict text records
bounded lineage, headroom, earned negative evidence, or a planned kill gate
without a frame-violation signal
**When** the tuned classifier evaluates the artifact
**Then** it classifies the artifact as `clean_bounded_negative`.

**Given** a P1 v1 or P1 v2 artifact whose verdict text records a load-bearing
positive-control failure
**When** the tuned classifier evaluates the artifact
**Then** it classifies the artifact as `frame_violating_anomaly` and still only
recommends human escalation without relaxing verification.

### REQ-AUTO-018: Recommend-Only Anomaly Escalation Advisory Hook

The autoresearch system SHALL provide a standalone advisory wrapper around the
tuned Exp 3802 anomaly-escalation classifier. The wrapper SHALL expose a pure
`classify_negative(artifact_or_verdict)` function returning only
`recommendation`, `reason`, and `frame_violation`, where `recommendation` is
one of `auto_reconcile` or `escalate_to_human`. It SHALL call the tuned
classifier, SHALL have no side effects, and MUST NOT relax verification, edit
artifacts, prune research state, or modify `scripts/research_conductor.py`.

The Exp 3809 workflow SHALL validate the wrapper by replaying the Exp 3791/3802
historical labeled sample with at least 30 artifacts. It SHALL report a
confusion matrix, false-escalation rate for clean bounded negatives, recall on
frame-violating anomalies, upstream artifact provenance, deterministic checksum,
and `never_relaxes_verification=true` in
`results/experiment_3809_anomaly_escalation_advisory_hook.json`. The workflow
SHALL update only the conductor-hook change proposal, describing where an
operator may call the advisory in the reconciliation path.
It SHALL leave the actual conductor unmodified.

#### SCENARIO-AUTO-015: Advisory Wrapper Replays Tuned Classifier Without Relaxing Verification

**Given** the tuned Exp 3802 classifier artifact, the Exp 3791/3802 historical
sample, and a clean bounded negative artifact
**When** `classify_negative` evaluates the clean negative
**Then** it recommends `auto_reconcile`, reports `frame_violation=false`, and
does not recommend any verification relaxation.

**Given** a P1 v1 or P1 v2 frame-violating positive-control failure artifact
**When** `classify_negative` evaluates the anomaly
**Then** it recommends `escalate_to_human`, reports `frame_violation=true`, and
does not recommend any verification relaxation.

**Given** the Exp 3791/3802 labeled replay corpus
**When** the Exp 3809 workflow replays the advisory wrapper
**Then** the artifact reports at least 30 replay negatives, false-escalation
rate at most 0.2, frame-violating recall 1.0, `conductor_unmodified=true`,
`integration_proposal_emitted=true`, upstream artifact provenance, no live-model
markers, and a terminal verdict that records recommend-only wiring readiness.


### REQ-AUTO-019: Unattended Conductor Integration

**Origin:** 2026-09-11 operator directive: "I mostly want to allow Carnot to
pursue and try things on its own during conductor loops and not require my
involvement," refined to specifically the real AVO-style mutation-operator
loop (agent proposes a change, a mechanical benchmark scores it, keep only
if it beats the incumbent, persist a scored lineage). REQ-AUTO-001 through
REQ-AUTO-015 already specify and implement that entire loop; nothing in
`scripts/research_conductor.py` ever called any of it.

`scripts/research_conductor.py` SHALL invoke one bounded autoresearch round
at milestone-close, as a sibling step to the existing adversarial audits
(`pages_adversarial_audit.py`, `verifier_authenticity_audit.py`, etc.),
gated by `CARNOT_AUTORESEARCH_UNATTENDED=1` (default off). The round SHALL
use `run_loop_with_generator` (REQ-AUTO-003) against a hypothesis generator
invoked via the `codex` CLI (`scripts/autoresearch_conductor_round.py:
call_codex`, mirroring `pages_adversarial_audit.py:call_codex` and
`scripts/research_conductor.py`'s own `_build_agent_command` codex branch --
same flags, prompt piped via stdin), model `gpt-6-astra` (2026-09-12
operator directive; supersedes an earlier draft of this REQ that specified a
locally-served OpenAI-compatible endpoint via `hypothesis_generator.py`'s
`GeneratorConfig` -- see the Decentralization note below for why the model
choice differs from `hypothesis_generator.py`'s own docstring, which still
describes the HTTP path as a generic, reusable option). The round SHALL be
bounded (a small `max_iterations`, the existing `max_consecutive_failures`
circuit breaker, a per-call codex timeout) and SHALL pass a
`ConstitutionChecker` (REQ-AUTO-015) so every sandbox execution and every
file-system action it takes is gated by the existing three-tier policy. If
the `codex` CLI is unavailable (`codex_available()`, a `shutil.which`
precondition check per the project's Pre-Launch Preconditions Discipline),
the round SHALL write a clean, non-fatal receipt and exit 0 rather than
blocking milestone-close -- the same contract every sibling audit already
has.

**Decentralization note.** This generator choice is the project's own
internal R&D tooling calling a closed-weight model via `codex exec` --
architecturally the same category as the planner/retro/audit tiers
`scripts/research_conductor.py` already runs on codex, never a locally-served
model. It is NOT a Carnot CAPABILITY (the thing `python/carnot/verify`/
`pipeline`/`samplers` ship to users), so the Decentralization-Respecting
Design Constraints' local-first mandate (which targets shipped capabilities)
does not bind it, any more than it binds the planner's own codex calls. This
mutation loop is also architecturally separate from, and never runs as part
of, the ARC live agent's own Kaggle-submitted code path (`scripts/kaggle/
kernel/main.py`, `arc_competition_agent.py`) -- confirmed by grep, zero
cross-imports in either direction -- so this REQ's model choice has no
bearing on the scored submission, which stays pinned to the local
Qwen3.8-27B model regardless of what this REQ does.

The fitness target for the first integration SHALL be the existing
DoubleWell/Rosenbrock energy benchmarks (`scripts/demo_autoresearch.py:
create_initial_baselines`) -- cheap, deterministic, no live LLM inference
needed to SCORE a candidate. The ARC live agent is explicitly OUT of scope
for this REQ (see `docs/research-notes/avo-adaptation-for-local-generator-
2026-08-21.md` Part 3 for why an evolutionary mutation loop was rejected
there), as is a verifier-ensemble-AUROC target (no reusable scoring harness
exists yet).

**Fitness is independently measured, not self-reported (fixed 2026-09-12 by
REQ-AUTO-021).** The energy compared against the baseline was originally
SELF-REPORTED by the hypothesis's own `run()` return value
(`sandbox.py:run_in_sandbox` took it verbatim) -- an adversarial review found
"keep only if it beats the incumbent" therefore meant "keep only if the
hypothesis CLAIMS to beat the incumbent". REQ-AUTO-021 closes this properly:
the hypothesis contract now returns a `final_state`, never a bare energy
number, and trusted harness code (not the sandboxed hypothesis) recomputes
the energy from that state via a real potential function. See REQ-AUTO-021
for the full mechanism and its own scenarios.

#### SCENARIO-AUTO-019-A: A bounded round runs unattended and is non-fatal when codex is unavailable

**Given** `CARNOT_AUTORESEARCH_UNATTENDED=1` and no `codex` binary on `PATH`
**When** the conductor reaches milestone-close
**Then** `scripts/autoresearch_conductor_round.py` runs, writes a receipt
containing `BLOCKED`, and returns exit code 0 -- milestone-close proceeds
to the planning step exactly as if the step had not run.

**Given** `CARNOT_AUTORESEARCH_UNATTENDED` unset (the default)
**When** the conductor reaches milestone-close
**Then** the autoresearch round is never invoked.

#### SCENARIO-AUTO-019-B: The generator-driven loop respects the constitution exactly like the static-list loop

**Given** a `ConstitutionChecker` configured to forbid `run_sandbox`, and an
`AutoresearchConfig` carrying it, passed to `run_loop_with_generator`
**When** the generator proposes a hypothesis
**Then** the hypothesis is rejected before the sandbox executes, exactly as
`run_loop` (REQ-AUTO-003) already behaves -- this closes a real gap found
while building this REQ: `run_loop_with_generator` and `run_loop_with_skills`
(REQ-AUTO-011) had no constitution check at all before this fix, so an
unattended run (the only caller that matters for this REQ) silently ignored
a configured `constitution_checker`.


### REQ-AUTO-020: Git-Committed Lineage Per Accepted Hypothesis

**Origin:** same 2026-09-11 directive as REQ-AUTO-019. AVO's own mechanism
(`docs/research-notes/avo-adaptation-for-local-generator-2026-08-21.md` Part
1) persists "a git commit per version with its score" -- the orchestrator
(REQ-AUTO-002, REQ-AUTO-008) only updates an in-memory `BaselineRecord` and
appends to a JSON `ExperimentLog`; neither is a durable, git-visible record.

When an unattended round (REQ-AUTO-019) accepts a hypothesis, the system
SHALL persist it as a self-contained JSON record under
`ops/autoresearch_discoveries/<benchmark_name>/<experiment_id>.json`
(hypothesis code, description, sandbox metrics, eval verdict/reason, and the
baseline's `final_energy` immediately before and after acceptance), and
SHALL commit that ONE file via an explicit `git add <path>` -- never
`git add -A` and never sweeping any other file -- with a commit message
naming the benchmark, the experiment id, and the before/after score. A
commit that a pre-commit hook or `git commit` itself refuses SHALL be
treated as "accepted by the evaluator, not persisted to lineage" -- it MUST
NOT retroactively change the round's accepted/rejected counts, and MUST NOT
leave a partially-staged file behind (a failed commit attempt SHALL reset
the file out of the index).

The record SHALL be a JSON sidecar rather than a live, importable `.py`
module: raw LLM-generated hypothesis code is not expected to satisfy this
repository's strict ruff/mypy pre-commit gates, and the lineage's purpose is
an audit trail (like a `results/*.json` artifact), not a shippable module.
The fast-resume caches this round reads and writes
(`ops/.autoresearch_baselines.json`, `ops/.autoresearch_experiment_log.json`)
are local, gitignored state, not part of this durable record.

#### SCENARIO-AUTO-020-A: An accepted hypothesis lands as its own scoped commit

**Given** a hypothesis that improves on the current `double_well` baseline
**When** the round accepts it
**Then** exactly one new file,
`ops/autoresearch_discoveries/double_well/<experiment_id>.json`, is
committed; `git show --stat` on that commit names no other path; and the
commit message contains the benchmark name, the experiment id, and both the
before and after `final_energy` values.

#### SCENARIO-AUTO-020-B: A constitution-forbidden action never reaches git

**Given** a `ConstitutionChecker` that forbids `create_file:ops/
autoresearch_discoveries/...` or forbids `git_commit`
**When** an otherwise-accepted hypothesis would be persisted
**Then** no file is written under `ops/autoresearch_discoveries/`, no
commit is made, and the repository's commit count is unchanged.

#### SCENARIO-AUTO-020-C: Local caches never enter git status as staged or committed

**Given** a completed round, whether or not any hypothesis was accepted
**When** the round finishes
**Then** `ops/.autoresearch_baselines.json` and
`ops/.autoresearch_experiment_log.json` exist on disk (or are updated) but
are never `git add`-ed or committed by this round.


### REQ-AUTO-021: Independently-Measured Fitness for the Toy Benchmarks

**Origin:** 2026-09-12 adversarial review (Fable 5.1), finding 1, followed by
an operator directive: "build the real energy function fix." The review
reproduced that `run_round`'s fitness gate trusted a hypothesis's own
self-reported `final_energy` verbatim -- `def run(d): return {'double_well':
{'final_energy': -999999.0}}` was accepted and committed as if it were a
real result. `orchestrator.py`'s own docstring states the design intent this
violated: "the energy function is the objective judge (can't be gamed by an
LLM)". For the two toy benchmarks (REQ-AUTO-019), no such judge existed.

The system SHALL provide real, deterministic potential functions for
`double_well` (`E(x) = sum_i (x_i^2 - 1)^2`) and `rosenbrock`
(`E(x) = sum_i [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]`) in
`python/carnot/autoresearch/toy_benchmarks.py`, and a
`recompute_final_energy(benchmark_name, final_state)` function that computes
the true energy from a state the hypothesis reports having reached, never
raising -- returning `None` on any unscoreable input (unknown benchmark
name, missing/malformed/non-finite state, an oversized state) rather than a
value that could itself leak information back to the hypothesis.

The autoresearch conductor round (REQ-AUTO-019) SHALL change the hypothesis
contract to require a `final_state` (the point the procedure converged to)
instead of a self-reported `final_energy`, and SHALL make every sandboxed
hypothesis execution's returned metrics pass through
`recompute_final_energy` BEFORE the evaluator (REQ-AUTO-005) ever sees them
-- discarding any self-reported `final_energy` entirely and replacing it
with the independently recomputed value, or removing the key altogether when
recomputation is not possible. This interception SHALL be implemented
without modifying `orchestrator.py`, `evaluator.py`, or `sandbox.py`
(REQ-AUTO-002 through REQ-AUTO-010's shared, independently-tested contract)
-- a request-scoped substitution of the `execute_hypothesis` name
`orchestrator.py` calls, active only for the duration of one autoresearch
round, is sufficient and keeps every existing accept/reject/circuit-breaker
test in those shared modules meaningful and unmodified.

A benchmark name `recompute_final_energy` does not recognize (including any
name an LLM invents) SHALL never receive a `final_energy` value, so it can
never register as an improvement or a regression under REQ-AUTO-005's
evaluation.

#### SCENARIO-AUTO-021-A: A fabricated energy claim is never committed

**Given** a hypothesis `def run(d): return {'double_well': {'final_energy':
-999999.0}}` (a `final_energy` with no `final_state`)
**When** a bounded round runs it through the real sandbox and evaluator,
with nothing mocked below the hypothesis generator
**Then** the hypothesis is still sandboxed successfully, but its
`final_energy` claim is discarded before evaluation; no lineage commit is
made and the repository's commit count is unchanged.

#### SCENARIO-AUTO-021-B: A real state recomputes to the true energy

**Given** a hypothesis that returns `{'double_well': {'final_state': [1.0,
1.0]}}`
**When** the round evaluates it
**Then** the committed record's `final_energy` is exactly `0.0` (the true
global minimum of the double-well potential at that state), never a
value the hypothesis chose.

#### SCENARIO-AUTO-021-C: An unscoreable claim is dropped, not treated as zero or as failure

**Given** any of: an unknown benchmark name, a missing `final_state`, a
non-numeric or non-finite state value, or a state longer than the supported
dimension
**When** `recompute_final_energy` is called
**Then** it returns `None` without raising, and the corresponding
`final_energy` key is absent from the recomputed metrics passed to the
evaluator.


### REQ-AUTO-022: Persisted Generator-Failure Diagnostics

**Origin:** 2026-09-12 known-issues entry, filed the same day REQ-AUTO-019's
"always run" fix went live: the round fired for real twice in production and
both times finished with zero iterations, but the receipt
(`ops/autoresearch_conductor_report.md`) carried no diagnostic for *why*.
`codex_generate_hypotheses` and `fable_generate_hypotheses` each append a
`{"description": ..., "reason": ...}` entry to a `recent_failures` list on
failure, but that list was only ever read back INTO the next iteration's
prompt (so the next generator call sees what went wrong) -- never read OUT
into the receipt. The operator could see a round produced nothing, but not
whether codex timed out, returned a non-zero exit, or crashed with an
`OSError`, without re-running by hand.

The system SHALL capture the same `recent_failures` list object
`orchestrator.run_loop_with_generator` passes to the round's `generator`
callback on every iteration (the orchestrator creates this list once, outside
the loop, and clears it only when a hypothesis is accepted -- so for a round
that never accepts anything, the object holds every failure the round saw).
When that captured list is non-empty at the end of the round,
`ops/autoresearch_conductor_report.md` SHALL include a `## Generator failure
reasons` section listing each entry's `description` and `reason` (the raw
`call_codex`/`call_fable` failure string -- a timeout message, an exit code
plus truncated stderr, or an `OSError`), each `reason` capped at 300
characters so a large stderr blob cannot make the receipt unreadable.

This requirement adds no new generator behavior and does not change
`orchestrator.py`, `evaluator.py`, or `sandbox.py` -- it only threads an
already-computed diagnostic from an in-memory list to the receipt file that
already exists per REQ-AUTO-019.

#### SCENARIO-AUTO-022-A: Both generators failing leaves a reason in the receipt

**Given** a round where `call_codex` fails with `"codex exit 1: some stderr"`
and the Fable fallback then also fails with `"claude exit 1: some other
stderr"`
**When** the round completes with `iterations == 0`
**Then** the receipt's `## Generator failure reasons` section contains both
`codex_call_failed: codex exit 1: some stderr` and `fable_call_failed: claude
exit 1: some other stderr`, in the order the failures occurred.

#### SCENARIO-AUTO-022-B: A clean round with no failures omits the section

**Given** a round where the first generator call returns a usable hypothesis
on iteration 0
**When** the round completes
**Then** the receipt contains no `## Generator failure reasons` section,
since `recent_failures` never received an entry.

### REQ-AUTO-023: Bounded Retry After An Empty Generator Response

**Origin:** the 2026-09-12 known-issues entry "two-for-two zero-iteration
production fires": `run_loop_with_generator` broke out of the ENTIRE round
the instant the `generator` callback returned an empty hypothesis list on
ANY iteration, including iteration 0. A configured `max_iterations=5` meant
"up to 5 attempts" only if the FIRST attempt succeeded -- a single transient
generator hiccup (a codex timeout, a malformed response, a CLI error) ended
the whole round with zero retries, and both real production fires this
session hit exactly that path.

The system SHALL treat a generator returning zero hypotheses the same way it
already treats a generator raising an exception (an existing, already-correct
pattern in the same function): log the failure, append a diagnostic entry to
`recent_failures` (`{"description": "generator_empty", "reason": ...}`),
advance the iteration counter, and continue the loop -- rather than breaking
immediately. This SHALL apply to both `run_loop_with_generator` and
`run_loop_with_skills` (the two functions sharing the pre-existing bug).

Because a generator backed by an expensive external call (an LLM subprocess)
should not share a retry budget tuned for cheap, local, sandboxed hypothesis
rejections, `AutoresearchConfig` SHALL gain a dedicated
`max_consecutive_empty_generations: int = 3` field, independent of
`max_consecutive_failures` (which continues to bound consecutive REJECTED
*evaluated* hypotheses only). When a generator returns empty
`max_consecutive_empty_generations` times in a row, the loop SHALL stop and
set a new `LoopResult.generator_exhausted: bool` field to `True`, distinct
from `circuit_breaker_tripped` (which requires the generator to have
produced something that was then rejected). The outer `max_iterations` bound
continues to apply regardless -- whichever limit is reached first ends the
loop.

A caller whose generator's per-call cost is non-trivial (REQ-AUTO-019's
`scripts/autoresearch_conductor_round.py`, where one empty generator call
means a full codex timeout followed by a full Fable fallback timeout) SHALL
pass an explicit `max_consecutive_empty_generations` and size its own outer
process timeout (`research_conductor.py:_run_autoresearch_round`'s
`_run_audit_with_receipt` call) to comfortably exceed
`max_consecutive_empty_generations * (codex_timeout + fable_timeout)` in the
worst case, so the retry budget this requirement grants is not silently cut
short by an unrelated parent-process timeout.

#### SCENARIO-AUTO-023-A: An empty iteration 0 is followed by a successful iteration 1

**Given** a generator that returns no hypotheses on iteration 0 and a
winning hypothesis on iteration 1
**When** the loop runs with `max_iterations >= 2`
**Then** the loop does not stop after iteration 0; the hypothesis from
iteration 1 is evaluated and, if it beats baseline, accepted --
`result.accepted == 1` and `result.generator_exhausted` is `False`.

#### SCENARIO-AUTO-023-B: A permanently-empty generator still gives up, bounded

**Given** a generator that always returns no hypotheses
**When** the loop runs with `max_iterations=10` and the default
`max_consecutive_empty_generations=3`
**Then** the generator is called exactly 3 times (not 1, not 10),
`result.generator_exhausted` is `True`, and `result.iterations == 0`.

#### SCENARIO-AUTO-023-C: max_iterations still bounds the retry loop

**Given** a generator that always returns no hypotheses, with
`max_consecutive_empty_generations=10` but `max_iterations=2`
**When** the loop runs
**Then** the generator is called exactly 2 times (the outer bound wins),
and `result.generator_exhausted` is `False` (the empty-streak threshold was
never reached; the loop ended on `max_iterations` instead).

### REQ-AUTO-024: Correct Generator Attribution in Commit Messages

**Origin:** the first two real production autoresearch rounds after REQ-AUTO-023
shipped both ran entirely on the Fable 5.1 fallback (`fable_fallback_iterations:
[0, 1, 2, 3, 4]` -- codex failed every single iteration), yet all 5 resulting
git commits said "Hypothesis proposed via codex exec (see call_codex)". The
text was a hardcoded literal in `commit_accepted_hypothesis`, never checked
against which generator actually produced the winning hypothesis.

`ExperimentEntry` (the shared, REQ-AUTO-008 dataclass `orchestrator.py`
returns) carries no generator-provenance field, and this script does not own
that dataclass -- widening it is out of scope here. Instead, the system SHALL
recover which generator won from the entry id's own shape:
`run_loop_with_generator` names each entry `llm-<timestamp>-<iteration:03d>`,
and `fable_fallback_iterations` (already tracked per REQ-AUTO-022) records
which iterations needed the fallback. `generator_label_for_entry(entry_id,
fable_fallback_iterations)` SHALL parse the trailing iteration number and
return a label naming Fable when that iteration is in the fallback list,
codex otherwise -- falling back to the codex label (never raising) if the id
does not match the expected shape. `commit_accepted_hypothesis` SHALL accept
this label and interpolate it into the commit message in place of the
hardcoded text.

#### SCENARIO-AUTO-024-A: A Fable-produced acceptance is attributed to Fable

**Given** a round where the generator reports iteration 0 as a fallback
iteration and iteration 0's hypothesis is accepted
**When** the round commits that hypothesis
**Then** the commit message names Fable, not codex.

#### SCENARIO-AUTO-024-B: A codex-produced acceptance is attributed to codex

**Given** a round where no iteration needed the fallback
**When** an accepted hypothesis is committed
**Then** the commit message names codex exec, matching the historical text.

### REQ-AUTO-025: Verifier-AUROC as a Second, Headroom-Having Fitness Target

**Origin:** 2026-09-14 research note
(`docs/research-notes/rsi-levels-and-autoresearch-fitness-target-2026-09-14.md`)
found that fitness target #1 (`double_well`/`rosenbrock`, REQ-AUTO-021) had
been driven to machine-precision zero on the very first production
autoresearch fire, and every fire since landed `accepted > 0, committed = 0`
-- expected benchmark saturation, not a defect, but a dead end for further
measurement. The note, and `autoresearch_conductor_round.py`'s own module
docstring, named the reason a verifier-ensemble AUROC target had not been
built instead: "no reusable AUROC harness exists yet." A 2026-09-16 operator
directive asked for that gap to be scoped and closed.

The system SHALL provide `python/carnot/autoresearch/
verifier_auroc_benchmark.py`, a second benchmark named `verifier_auroc`,
built the same way REQ-AUTO-021 built the first two: a real, independently
computed metric a hypothesis cannot influence except by actually finding a
better answer.

**The corpus and split.** The benchmark scores against
`data/fover_corpus_v4.json` (6548 labeled reasoning-step rows, `label` in
`{"correct", "incorrect"}`), split by a fixed SHA-256 hash of `question_id`
into a ~30% TRAINING slice and a disjoint ~70% HELD-OUT slice. The split
SHALL be computed once and never re-randomized -- a hypothesis across
different rounds always trains against the same rows and is always scored
against the same held-out rows it never sees. Splitting by `question_id`
(not by individual row) SHALL keep every reasoning step from the same
problem on one side of the split, preventing train/held-out leakage through
a shared question.

**The tunable parameters.** A hypothesis proposes two weights,
`final_state = [entity_weight, falsifiability_weight]`, for
`carnot.verify.pcib_probe.PCIBProbe` -- an already-shipped, honestly
disclosed text-statistical hallucination probe (no GPU or LLM call needed to
score a candidate). The hypothesis MAY read
`benchmark_data["verifier_auroc_train_rows"]` (the training slice only, as
plain `{"step_text", "label"}` dicts) and import `PCIBProbe` directly to
search for good weights by any real method against that training data; it
is never shown the held-out slice.

**The trust boundary (same shape as REQ-AUTO-021, unchanged).**
`recompute_verifier_auroc_energy(final_state)` is the one function trusted
harness code calls to turn a hypothesis's claimed weights into a real
energy: it rebuilds `PCIBProbe` from the claimed weights, scores every row
of the FIXED held-out slice, computes AUROC via a local Mann-Whitney
implementation, and returns `1.0 - auroc` (lower is better, matching
REQ-AUTO-021's minimization convention). It SHALL never raise -- returning
`None` (dropped entirely, same as a missing `final_state` under REQ-AUTO-021)
for a malformed state (wrong length, non-numeric, non-finite, or a weight
whose magnitude exceeds `MAX_ABS_WEIGHT`), an empty/unreadable corpus, or a
held-out slice missing either class after any of those checks. A
hypothesis's own computed AUROC on the training rows, if it reports one, is
never read by the evaluator.

**The dispatch wiring.** `autoresearch_conductor_round.py:_recompute_metrics`
SHALL recognize the `verifier_auroc` benchmark name alongside the two
existing `toy_benchmarks.py` names, dispatching to
`recompute_verifier_auroc_energy` the same way it dispatches double_well/
rosenbrock to `toy_benchmarks.recompute_final_energy` -- without modifying
`toy_benchmarks.py` itself, keeping REQ-AUTO-021's own scope (the two toy
benchmarks) unchanged. `AUTORESEARCH_SYSTEM_PROMPT` SHALL describe this third
benchmark's contract in the same message as the first two, so a single
generator call can propose a hypothesis for any of the three.

**The baseline seed is a real measurement, not a placeholder.**
`measure_default_weight_energy()` SHALL compute the seed baseline by calling
`recompute_verifier_auroc_energy([0.5, 0.5])` -- `PCIBProbe`'s own documented
default weights -- against the real corpus, rather than a hand-typed number
that could silently drift from the corpus or the probe's logic.
`seed_baselines()` SHALL use this measurement for the `verifier_auroc` entry.

**Real, measured headroom (not assumed).** Confirmed once against the
checked-in corpus, not asserted: the default weights (0.5, 0.5) score AUROC
0.3465 on the held-out slice -- worse than chance, because `entity_uptake`
(the probe's "novel numbers are suspicious" signal) does not track this
corpus's actual error class. A single sign flip (entity_weight=-1.0,
falsifiability_weight=1.0) reaches AUROC 0.7219. A hypothesis that discovers
this by searching, rather than trusting the probe's own default weighting,
is a genuine win against real headroom -- unlike REQ-AUTO-021's two toy
benchmarks, which had no headroom left after the first production fire.

#### SCENARIO-AUTO-025-A: A fabricated AUROC claim is never committed

**Given** a hypothesis that returns `{"verifier_auroc": {"final_energy":
0.0}}` with no `final_state`
**When** a bounded round runs it through the real sandbox and evaluator
**Then** the hypothesis is sandboxed successfully, but its `final_energy`
claim is discarded before evaluation; no lineage commit is made.

#### SCENARIO-AUTO-025-B: Real weights recompute to the true held-out AUROC

**Given** a hypothesis that returns `{"verifier_auroc": {"final_state":
[-1.0, 1.0]}}`
**When** the round evaluates it
**Then** the committed record's `final_energy` equals
`1.0 - auroc`, where `auroc` is independently computed by
`recompute_verifier_auroc_energy` against the fixed held-out slice --
never a value the hypothesis chose or reported.

#### SCENARIO-AUTO-025-C: Training and held-out rows never overlap

**Given** the fixed corpus split
**When** `default_benchmark_data()["verifier_auroc_train_rows"]` and the
module's internal held-out slice are compared by `step_text`
**Then** their intersection is empty.

#### SCENARIO-AUTO-025-D: An unscoreable claim is dropped, not treated as zero or as failure

**Given** any of: a `final_state` of the wrong length, a non-numeric or
non-finite weight, or a weight whose magnitude exceeds `MAX_ABS_WEIGHT`
**When** `recompute_verifier_auroc_energy` is called
**Then** it returns `None` without raising, and the corresponding
`final_energy` key is absent from the recomputed metrics passed to the
evaluator.

### CORRECTION 2026-09-16 (same day, adversarial review before first production
fire): the trust boundary above was incomplete, and the baseline seed never
reached production. Both fixed; both preserved here per never-prune.

**Origin:** an adversarial review (Fable 5.1), commissioned per operator
directive before the first real production fire of REQ-AUTO-025, found two
CRITICAL findings and three REAL_BUG findings against the implementation
above -- all reproduced through the real code, not hand-traced. The 82 tests
shipped with the original REQ-AUTO-025 stayed green through every one of
them, because none of the tests exercised a PRE-EXISTING production baseline
cache, an in-process monkeypatch, or a degenerate weight pair.

**CRITICAL-1 (fixed): the production baseline cache predates this benchmark,
so the seed in `seed_baselines()` never actually reaches it.** `load_baselines`
returned a cached `BaselineRecord` verbatim when the cache file existed.
`ops/.autoresearch_baselines.json` already existed before REQ-AUTO-025
shipped and holds only `double_well`/`rosenbrock`. Reproduced end to end: the
FIRST hypothesis to report ANY `verifier_auroc` weights -- reproduced with a
genuinely bad pair (held-out energy ~0.72) -- gets accepted with verdict
"PASS: No regression" (the evaluator has nothing in `baselines.benchmarks` to
compare against) and `orchestrator._update_baselines` writes THAT number as
the baseline. The measured seed (`measure_default_weight_energy()`) is never
consulted in production. **The fix:** `load_baselines` now calls
`_merge_missing_seed_benchmarks(record)` after loading a cache, which adds
any benchmark `seed_baselines()` knows about but the loaded record does not
-- a schema migration, not a one-off patch, so a future fitness target #3
added the same way does not repeat this exact bug.

**CRITICAL-2 (fixed): the trust boundary was void in-process.** `sandbox.py`'s
own docstring already says its isolation is "not a security boundary";
hypotheses execute in the SAME interpreter as the harness's own recompute
call, immediately afterward. Three reproduced variants, all through the real
`run_round` -> real sandbox -> real evaluator -> real git commit:

- **Variant B1 (monkeypatch):** a hypothesis does `import carnot.autoresearch.
  verifier_auroc_benchmark as vab; vab._binary_auroc = lambda l, s: 1.0`, then
  reports ANY weights. The patched function is still installed when the
  harness's "independent" recompute runs next in the same process --
  `final_energy` comes back `0.0` and IS git-committed as a real discovery.
  The identical class of bug was reproduced against REQ-AUTO-021's own
  `toy_benchmarks.BENCHMARK_ENERGY_FUNCTIONS` too (the 2026-09-12 reviewer's
  exact `-999999.0` reproduction still works with one added import line) --
  this correction covers both.
- **Variant B2 (cache mutation):** `_split_corpus()`'s `lru_cache` returns the
  SAME dict objects on every call; a hypothesis mutating a held-out row's
  `step_text`/`label` in place corrupts the harness's own subsequent scoring
  of that row.
- **Variant A (answer-key read):** nothing stopped a hypothesis from
  importing `carnot.autoresearch.verifier_auroc_benchmark` directly and
  calling `_split_corpus()` / `recompute_verifier_auroc_energy()` itself to
  read the held-out rows and report the held-out-optimal weights directly --
  the ORIGINAL `train_rows_for_prompt()` docstring's claim that the held-out
  split "is never exposed this way; it exists only inside this module" was
  false as long as the module itself was importable from inside the sandbox.

**The fix, two parts, closing all three variants together:**

1. The system SHALL recompute every benchmark's post-sandbox energy in a
   FRESH interpreter process, never the process that just executed the
   hypothesis. `_subprocess_recompute_energy` (in
   `autoresearch_conductor_round.py`) spawns
   `scripts/_autoresearch_energy_recompute_worker.py` per recompute, passing
   `{"benchmark_name": str, "final_state": Any}` as JSON on stdin and reading
   `{"energy": float | None}` as JSON from stdout -- never a live Python
   object across the boundary, so nothing a hypothesis constructed (a
   patched function, a mutated dict, a class instance) can reach the
   recompute. This closes Variant B1 and B2 completely, and retroactively
   hardens REQ-AUTO-021's own toy benchmarks against the identical class of
   bug, without modifying `toy_benchmarks.py`, `orchestrator.py`,
   `evaluator.py`, or `sandbox.py`.
2. The system SHALL block every `carnot` import for sandboxed hypothesis
   code in a real autoresearch round, and SHALL hand `PCIBProbe` to the
   hypothesis directly as a value in `benchmark_data["PCIBProbe"]` instead of
   requiring `import carnot.verify.pcib_probe`. `run_round`'s
   `AutoresearchConfig` now carries `sandbox_config=SandboxConfig
   (blocked_modules=BLOCKED_MODULES | frozenset({"carnot"}))`. This closes
   Variant A: a hypothesis can no longer import
   `verifier_auroc_benchmark.py` (or any other `carnot.*` module) to read
   the held-out split or reach the same globals the subprocess fix (above)
   protects, while the intended workflow (constructing `PCIBProbe` and
   calling `.score()` against the training rows) is unaffected. This does
   NOT require modifying `sandbox.py`'s own guarded-import mechanism --
   `SandboxConfig.blocked_modules` was already a caller-supplied parameter;
   `run_round` simply supplies a stricter one for this specific round.

**REAL_BUG (fixed): "never raises" was false.** `_validate_weights` (this
module) and `toy_benchmarks.recompute_final_energy` both caught only
`(TypeError, ValueError)` around their `float(...)` conversions. A value
like `10**400` (a plain Python int too large for a float) raises
`OverflowError` instead, which propagated past both functions uncaught and
killed the whole round before its receipt was written -- contradicting both
functions' own docstring claims. Both now catch `Exception` broadly, with a
comment explaining why that breadth is deliberate for untrusted,
LLM-generated input.

**REAL_BUG (fixed): a constant scorer was an accepted, committable
"improvement."** Weights `(0.0, 0.0)` (or any pair producing an identical
score for every held-out row) make `_binary_auroc` return exactly `0.5` by
its own tie-counting rule -- which happened to beat this benchmark's
worse-than-chance seed baseline (0.6535) and so was accepted as a genuine
"improvement" that in fact discriminates nothing.
`recompute_verifier_auroc_energy` now rejects (returns `None` for) any
weight pair whose held-out scores have zero variation, before computing
AUROC at all. The `MAX_ABS_WEIGHT` bound's docstring, which had incorrectly
claimed to guard against exactly this case, is corrected: AUROC is
invariant to uniformly rescaling both weights, so that bound is a numeric-
stability guard only, not a degeneracy guard.

**REAL_BUG (named, NOT structurally fixed here -- see verifier_auroc_
benchmark.py's own module docstring for the honest, corrected framing):**
the score is a linear function of two weights, so its AUROC depends only on
their angle; one honest search finds essentially the whole landscape. The
review's own Hanley-McNeil calculation put the held-out slice's standard
error at ~0.034 (71 positive / 4488 negative rows) against an acceptance
tolerance roughly 125x smaller, meaning ACCEPTED "improvements" after the
first are plausibly sampling noise on a fixed held-out set, not genuine
generalization. Candidate fixes (not attempted in this pass): widen the
tolerance for this benchmark to roughly one Hanley-McNeil standard error, or
stop echoing the exact held-out energy value back into the hypothesis-
generator prompt across rounds. Tracked as an open item in
`ops/known-issues.md`.

#### SCENARIO-AUTO-025-E: An in-process monkeypatch cannot fabricate an energy

**Given** a hypothesis that does `import carnot.autoresearch.
verifier_auroc_benchmark as vab; vab._binary_auroc = lambda l, s: 1.0` and
reports any weights
**When** the round evaluates it through the real sandbox and the real
subprocess-isolated recompute
**Then** the resulting `final_energy` is the same as an honest, unpatched
recompute of those weights would produce -- never the fabricated value the
monkeypatch would have produced in-process.

#### SCENARIO-AUTO-025-F: A hypothesis cannot import its way to the held-out split

**Given** a hypothesis that does `import carnot.autoresearch.
verifier_auroc_benchmark as vab; vab._split_corpus()`
**When** it runs inside a real autoresearch round's sandbox
**Then** the import raises `ImportError` (the sandbox's blocked-modules set
includes `carnot` for this round), and the hypothesis's execution fails
rather than succeeding with the held-out rows in hand.

#### SCENARIO-AUTO-025-G: PCIBProbe remains usable without any carnot import

**Given** a hypothesis that reads `benchmark_data["PCIBProbe"]`, constructs
it, and calls `.score(...)` against the training rows, without writing any
`import carnot` statement
**When** it runs inside a real autoresearch round's sandbox with `carnot`
imports blocked
**Then** it succeeds and produces a real, independently recomputed
`final_energy` for `verifier_auroc`.

#### SCENARIO-AUTO-025-H: A degenerate constant scorer is never accepted

**Given** a `final_state` such as `[0.0, 0.0]` that scores every held-out row
identically
**When** `recompute_verifier_auroc_energy` is called
**Then** it returns `None` (unscoreable), never `0.5` treated as a real
energy.

#### SCENARIO-AUTO-025-I: A missing benchmark in a pre-existing baseline cache is seeded, not silently absent

**Given** a `BaselineRecord` loaded from a cache file that predates
`verifier_auroc` (contains only `double_well`/`rosenbrock`)
**When** `load_baselines` loads it
**Then** the returned record also contains a `verifier_auroc` entry equal to
`measure_default_weight_energy()`, not an absent key.

### REQ-AUTO-026: Scope the rejection circuit breaker to one loop invocation

Each real autoresearch loop invocation SHALL save the append position of its
experiment log before it evaluates a hypothesis. Every circuit-breaker check in
that invocation SHALL count only the consecutive rejected entries at or after
that position. Persisted rejections before the saved position SHALL remain in
the log and in the rejected-ID registry, but SHALL NOT stop a later invocation
before its first proposal or evaluation.

The configured `max_consecutive_failures` threshold SHALL remain unchanged. A
rejected sandbox execution SHALL count as a rejection. An accepted or
pending-review entry SHALL break the local rejection streak. Empty logs and
invocations with no new entries SHALL have a local count of zero. The existing
all-time `consecutive_failures()` query SHALL retain its historical behavior for
callers that do not supply an invocation boundary.

The bounded conductor round SHALL use the same invocation-local boundary as the
three shared loop entrypoints. It SHALL preserve the original bytes of all old
log records, timeout and error handling, review outcomes, and rejected-ID
deduplication. Recovery means that a later round reaches proposal and evaluation;
it does not turn a scientific rejection into an acceptance.

#### SCENARIO-AUTO-026-A: Historical rejections do not lock a new invocation

**Given** a persisted log whose tail contains ten valid rejected entries
**When** a new loop invocation starts with a threshold of ten
**Then** its first proposal reaches evaluation
**And** the ten historical entries remain byte-identical.

#### SCENARIO-AUTO-026-B: Fresh rejections still stop a stuck invocation

**Given** a new invocation with a rejection threshold of ten
**When** that invocation appends ten consecutive rejected evaluations
**Then** its circuit breaker trips before an eleventh evaluation.

#### SCENARIO-AUTO-026-C: A non-rejection breaks only the local streak

**Given** a new invocation that records rejections followed by an accepted or
pending-review result
**When** later rejections are counted
**Then** only the trailing rejections after that result contribute to the
invocation-local circuit breaker.

#### SCENARIO-AUTO-026-D: Successive invocations have independent budgets

**Given** two loop invocations that append to the same persisted history
**When** the first invocation exhausts its consecutive-rejection budget
**Then** the second invocation can still propose and evaluate
**And** every rejection from both invocations remains recorded.

### REQ-AUTO-027: Start with agy and fail over to codex

The bounded conductor round SHALL try the agy CLI (Google Antigravity, Gemini
Flash models) first for every iteration. It SHALL call codex only when agy
returned no hypotheses for that iteration. It SHALL NOT call the Fable 5.1
(`claude`) generator. The Fable functions MAY stay in the file as dormant code.
The default agy model SHALL be a Gemini 3.8 Flash level. The codex fallback model
SHALL stay the round's `--model` value (default `gpt-6-astra`).

The agy call SHALL pass the prompt as a literal `--print` argument, because
`agy --print` does not read stdin. It SHALL run in a fresh empty scratch
directory and SHALL NOT use a permission-bypass flag. It SHALL resolve the
binary with `shutil.which` and fall back to `~/.local/bin/agy`, because the
conductor service PATH does not include `~/.local/bin`. The round SHALL be
blocked only when neither agy nor codex is available.

The round SHALL record each iteration where agy returned nothing and codex ran in
the receipt under `fallback_iterations`. The commit message of a hypothesis that
codex produced after that failover SHALL say so. When both generators return
nothing for enough consecutive iterations, the receipt SHALL say that agy and
codex both produced nothing.

The receipt SHALL show the TAIL of each generator failure text, not the head.
codex prints a banner and then echoes the whole prompt before its real error,
so the head holds no diagnosis.

#### SCENARIO-AUTO-027-A: agy empty, codex supplies the hypothesis

**Given** agy returns no hypotheses for an iteration
**When** the round runs that iteration
**Then** codex is called once with the same prompt
**And** the iteration is recorded in `fallback_iterations`
**And** Fable is never called.

#### SCENARIO-AUTO-027-B: both fail, the real error stays visible

**Given** agy and codex both exit non-zero with long stderr ending in a real error
**When** the round ends with the generator exhausted
**Then** the receipt lists `agy_call_failed` and `codex_call_failed` with the
error tails visible
**And** no `claude` process ran.

### REQ-AUTO-016: Headroom Gate Corpus for Grid Tasks
The system MUST generate a difficulty-stratified grid corpus (n >= 50) and measure matched-compute AR greedy, AR+SC32, and oracle solve rates. It must compute the headroom band (oracle - AR+SC32).
If AR+SC32 > 0.75, it must ABORT as ceiling-polluted. If AR_greedy ~ 0.20 and AR_SC32 < 0.50 and oracle materially > AR+SC32, it must set headroom_confirmed = true and CONFIRM.

#### SCENARIO-AUTO-016: Headroom Gate Output
**Given** a generator for grid tasks
**When** the headroom gate is evaluated
**Then** it outputs a valid json artifact with bare boolean headroom_confirmed, and principle-annotated values for ar_greedy_solve_rate, ar_sc32_solve_rate, oracle_solve_rate, headroom_margin, corpus_path, n_instances, difficulty_strata, preconditions_checked, inference_substrate, random_seed, reproducibility_checksum, and duration_s.


### REQ-AUTO-5194: Standalone Poison-Test-Cascade Pretest-Triage Module

The autoresearch system SHALL provide a standalone module
`scripts/pretest_triage.py` that the conductor's smart-subset pretest gate can
import to recognize the *poison-test-cascade* failure signature: a task's own
`tests/python/test_experiment_*.py` (or `test_exp*.py`) that reads a
`results/*.json` deliverable a SIBLING module's `main()` would produce, but the
task failed (wall-clock timeout or otherwise) BEFORE `main()` wrote that file,
leaving one red test whose `FileNotFoundError` (or `JSONDecodeError`) poisons the
SHARED pretest gate and SKIP/GATE_BLOCK-cascades every remaining task in the
milestone (milestone `2026.07.475` lost 10 of 12 tasks this way).

The module SHALL expose a pure `detect_poison_cascade(pytest_output, repo_root)`
function that returns which failing test node ids match this NARROW signature and
which do not. A failure matches ONLY when ALL of: (1) the failing test file is an
experiment-specific test (`test_experiment_*` / `test_exp*`, never a core/shared
test); (2) its failure block references a `results/*.json` path P; (3) P is a
declared `deliverable` of a task in the current `research-roadmap.yaml`; (4) P
does NOT yet exist on disk (the producing task has not delivered); and (5) the
failure block carries a file-absence marker (`FileNotFoundError`,
`No such file or directory`, or `JSONDecodeError`). A failure that lacks any of
these — an unrelated assertion, an import error, a stale/typo path not in the
roadmap, a deliverable already present on disk, or a core test — MUST NOT match,
so genuine regressions keep blocking the gate. The remediation the module
recommends/applies SHALL be a per-node-id `xfail` (never a blanket `skip` and
never file removal), gated on `condition=not os.path.exists(<deliverable>)` so it
SELF-EXPIRES the moment the deliverable lands and the test runs live again. The
module MUST NOT modify `scripts/research_conductor.py`; it carries documented
wiring instructions in its own docstring, mirroring
`scripts/retro_timing_fallback.py`.

#### SCENARIO-AUTO-5194-PRIMARY: The .475 Signature Is Detected And Scoped To One Node

**Given** the milestone `2026.07.475` pretest output where
`test_experiment_5182_diffusiongemma_meta_tensor_rootcause_fix_v475.py::test_ondisk_deliverable_is_valid`
fails with `FileNotFoundError` on
`results/experiment_5182_diffusiongemma_meta_tensor_rootcause_fix_v475.json`, a
roadmap declaring that path as exp5182's deliverable, and a repo root where the
deliverable does not exist
**When** `detect_poison_cascade` evaluates the output
**Then** it reports `matched=true` with exactly that one node id scoped for
`xfail`, names exp5182 as the producing task, and renders a self-expiring
`xfail` whose condition is the deliverable's on-disk absence.

#### SCENARIO-AUTO-5194-PRECISION: Genuinely Broken Tests Are Not Masked

**Given** pretest output containing an unrelated failing test (an assertion or
import error with no `results/*.json` reference), a test referencing a
`results/*.json` path that is not a roadmap deliverable, a test referencing a
declared deliverable that already exists on disk, and a core test
(`test_pipeline_extract.py`) referencing a pending deliverable
**When** `detect_poison_cascade` evaluates the output
**Then** none of those failures match the poison signature, they are reported as
unmatched failures that still block the gate, and `all_failures_explained` is
false whenever any real failure remains.

#### SCENARIO-AUTO-5194-HISTORICAL: Retrospective Classification Is Honest

**Given** the four cited incidents (exp3521/.325, exp3544/.326, exp3612/.332,
exp5182/.475)
**When** the module's historical validation runs
**Then** it confirms exp5182/.475 matches the deliverable-read signature exactly,
and honestly classifies exp3521/.325, exp3544/.326, exp3612/.332 as the sibling
verdict-assertion poison sub-class (zero `results/*.json` references) covered by
the pre-existing consecutive-fail auto-quarantine guard, rather than falsely
claiming a 4/4 narrow-signature match.

### REQ-AUTO-7440: Prospective Mixture Learning With A Frozen Reference

The experiment SHALL replay all authenticated source-group representatives from
the sealed `prospective_stream`. It SHALL use the label-blind hash order and the
domain-blocked shift order. Each order SHALL run with feedback delays zero and
eight. All arms in a cell SHALL share one uniform reveal schedule that selects
eight of each complete block of 32 and `floor(n/4)` from the final block. Every
revealed row SHALL have a positive known propensity.

Before a label arrives, the experiment SHALL persist each arm's probability,
shadow typed action, prediction-time state hash, and mixture weights. After the
label arrives, it SHALL update mixture weights from the stored prediction-time
losses. It SHALL then update each adaptive expert exactly once. A restart or
freeze SHALL NOT replay a committed update. The experiment SHALL retain request
order, reveal probability, arrival order, parent/event/child state hashes,
losses, update costs, feedback rows, weight trajectories, and checkpoint
lineage in hash-bound row shards.

The registered arms SHALL be a learned four-expert mixture, equal fixed weights
over the same adaptive experts, a frozen spline, an adaptive spline, and a
no-feedback frozen-prior mixture. Two fixed shuffled-label permutations SHALL
act as negative controls. All arms SHALL start from the authenticated Exp7439
fit checkpoints and use the Exp7438 mixture protocol. Typed action thresholds
SHALL remain fixed from Exp7439. Adaptive actions SHALL be shadow-only and SHALL
NOT inherit a static safety certificate or production permission.

The experiment SHALL report full-stream Brier and log loss, revealed-only and
inverse-probability-weighted estimates, domain-change harm, and read, predict,
persist, and update latency. It SHALL use 10,000 moving-block bootstrap draws at
block lengths 32 and 64. The four order-delay cells and three primary
comparisons SHALL form one prespecified family. Seed-level deltas SHALL be
averaged within source group before resampling because five fit seeds do not
create five independent corpora.

`online_value_score` SHALL be one only if the learned mixture has an upper
simultaneous log-loss delta below zero against the frozen spline, adaptive
spline, and equal-weight adaptive mixture in every cell. It SHALL also require
Brier non-inferiority within 0.001, no higher defined empirical harmful-action
rate, no coverage loss, no label-cost increase, no-feedback equality, and both
shuffled-label controls. A valid run with insufficient benefit SHALL be
`complete_null`, not disqualified. `online_capture_complete_score` SHALL be one
after valid terminal capture. `promotion_score` SHALL always be zero.

#### SCENARIO-AUTO-7440-01: Feedback Is Causal And Exactly Once

**Given** a prediction with delayed revealed feedback
**When** the replay commits the feedback
**Then** all arm probabilities and actions predate the reveal
**And** the weight update uses stored prediction-time losses
**And** each adaptive expert has one parent/event/child lineage edge
**And** a duplicate or restarted commit cannot apply the update twice.

#### SCENARIO-AUTO-7440-02: Uniform Budget Preserves Positive Propensity

**Given** a stream whose length is not a multiple of 32
**When** the uniform reveal schedule is sealed
**Then** each full block reveals exactly eight rows
**And** the final block reveals `floor(n/4)` rows
**And** every row has the block's positive selection propensity
**And** all arms and both delays use the same selected identities per order.

#### SCENARIO-AUTO-7440-03: Registered Evidence Uses One Corpus Unit

**Given** five fitted checkpoint seeds for each registered cell
**When** moving-block intervals are computed at lengths 32 and 64
**Then** seed deltas are first averaged within source group
**And** the bootstrap resamples ordered source groups rather than treating seed
rows as independent observations
**And** all 12 registered cell-comparison contrasts share one simultaneous
family correction.

#### SCENARIO-AUTO-7440-04: Adaptive Decisions Stay Shadow Only

**Given** fixed Exp7439 typed action thresholds and changing online probabilities
**When** the experiment reports actions and empirical risks
**Then** it marks every action as shadow-only
**And** it reports all-escalate deployment coverage as zero
**And** it reports selected risk as null when no shadow action is selected
**And** it never reports `certified_safe=true`.

#### SCENARIO-AUTO-7440-05: Terminal Value Is Conjunctive

**Given** complete valid replay rows and independently recomputed metrics
**When** one registered benefit or control gate fails
**Then** `online_capture_complete_score` is one
**And** `online_value_score` is zero
**And** the verdict is `complete_null` unless a validity defect disqualifies it.

Spec: REQ-AUTO-7440, SCENARIO-AUTO-7440-01, SCENARIO-AUTO-7440-02, SCENARIO-AUTO-7440-03, SCENARIO-AUTO-7440-04, SCENARIO-AUTO-7440-05

### REQ-AUTO-7449: Seal source-conditioned inputs and an external human challenge

Exp7449 SHALL seal a model-independent protocol for later source-conditioned
representation work. It SHALL make no current LLM call and SHALL declare
`MODEL_SPECS=[]`, `model_invoked=false`, zero current invocation counts,
`inference_substrate_class=no_model_load`, and `execution_venue=host`. Numeric
selector fitting is deferred and SHALL be reported separately as
`small_ebm_training` by its future producer. Historical model-shaped evidence
SHALL remain typed and hash-bound rather than becoming current inference.

The RAGTruth panel SHALL group rows by normalized complete source hash before
selection and keep every sibling response in one group. Label-blind hash order
SHALL select one response per group and at most 180 original-train groups for
training, 60 other original-train groups for calibration and tuning, and 60
original-test groups for internal testing. A shortage SHALL not be cross-filled
from another official role. Selection SHALL occur before evaluator labels are
read and the realized count SHALL be recorded before later fitting.

FaithBench SHALL be pinned to revision
`cf89797d82812c23b5d5e5c121f1d9b8983bbbce`. Only the bounded public release
license, README, schema, official aggregation script, and release batch files
SHALL enter an external cache. The protocol SHALL record immutable URLs, byte
hashes, sizes, CC BY-NC-SA 4.0 attribution, the release schema, and the official
worst-case binary label policy. It SHALL preserve ambiguous annotation labels
in the evaluator view while excluding annotator notes, detector scores, and
summarizer identity from predictor features. FaithBench SHALL remain an
external, disagreement-selected challenge of at most 100 normalized-source
groups and SHALL not estimate deployment prevalence.

The predictor view SHALL contain only stable identity and role fields, complete
source and response text, and the six existing source features. Human labels
and annotation details SHALL remain in a separately hash-bound evaluator view.
The protocol SHALL deduplicate the external challenge against authenticated
local V649 through V652 source corpora and disclose residual public-data
contamination uncertainty. Permuting evaluator labels SHALL leave group
eligibility, response selection, prompt bytes, and feature bytes unchanged.

Two representation views SHALL be frozen: complete RESPONSE alone and complete
SOURCE followed by RESPONSE. Neither view may contain a label or request a
yes/no answer. The GGUF tokenizer SHALL later enforce a 2,048-token complete-
input ceiling; evidence SHALL never be truncated to make a row eligible. This
model-independent protocol MAY be ready before exact tokenizer eligibility is
known, but it SHALL retain one unstarted eligibility row per group and view.

The future comparison SHALL freeze final-layer last-token pooling, a seeded
32-dimensional random projection, five fit seeds, matched-capacity Gibbs and
logistic heads, a prevalence control, and the old lexical-feature arm.
Source conditioning SHALL have response-only and source-shuffled matched
controls. Primary external Brier and log-loss contrasts SHALL use 10,000
source-group bootstrap draws and Holm correction. Risk and coverage intervals
SHALL remain descriptive. Confirmatory minima SHALL be 150 training, 40
calibration, 40 internal-test, and 60 external groups; a shortfall blocks only
confirmatory value, not an honest coverage report.

`source_protocol_ready_score=1` SHALL require frozen group roles, the feature
allowlist, authenticated release hashes and attribution, all controls, evaluator
isolation, and passing required validation. It SHALL not require token counts or
claim predictive benefit. `promotion_score` SHALL remain zero. The terminal
artifact SHALL publish atomically only after affected checks, fresh-process
cold replay, independent raw reduction, adversarial verification, and strict
row consistency pass. These readers form the capability end-to-end check; no
numbered end-to-end scenario applies because shared training, sampling,
bindings, and ARC code do not change.

#### SCENARIO-AUTO-7449-01: Official roles and shortages stay fixed

**Given** duplicate sources, sibling responses, and original train/test roles
**When** the label-blind group selector seals the RAGTruth panel
**Then** normalized duplicate sources and siblings cannot cross roles
**And** each role stops at its own cap without cross-filling a shortage.

#### SCENARIO-AUTO-7449-02: External annotations stay evaluator-only

**Given** pinned FaithBench rows with detector metadata, notes, and disagreeing labels
**When** predictor and evaluator views are projected
**Then** predictor bytes contain only source, response, existing features, identity, and role
**And** ambiguity remains in evaluator bytes without notes, scores, or summarizer identity.

#### SCENARIO-AUTO-7449-03: Label permutation cannot alter model inputs

**Given** one sealed predictor panel and its separate evaluator labels
**When** evaluator labels are permuted across the same row identities
**Then** prompt bytes, feature bytes, group eligibility, and selected response identities match
**And** no label-derived replacement can enter either representation view.

#### SCENARIO-AUTO-7449-04: Complete evidence controls token eligibility

**Given** RESPONSE and SOURCE-followed-by-RESPONSE representation bytes
**When** the later GGUF tokenizer applies the 2,048-token ceiling
**Then** over-limit rows are excluded without truncating evidence
**And** protocol readiness can precede exact token counts while eligibility stays unstarted.

#### SCENARIO-AUTO-7449-05: Source value has matched controls

**Given** a fixed final-layer pooling surface and 32-dimensional projection
**When** the future selector comparison is executed
**Then** five-seed Gibbs is compared with matched logistic, prevalence, lexical, response-only, and source-shuffled arms
**And** primary external proper-score contrasts use grouped resampling and Holm correction.

#### SCENARIO-AUTO-7449-06: Challenge coverage is not a safety certificate

**Given** complete external FaithBench coverage selected for detector disagreement
**When** protocol readiness and future value are interpreted
**Then** readiness claims only a reproducible isolated protocol
**And** risk, coverage, and shortfall results cannot authorize deployment prevalence or promotion.

Spec: REQ-AUTO-7449, SCENARIO-AUTO-7449-01, SCENARIO-AUTO-7449-02, SCENARIO-AUTO-7449-03, SCENARIO-AUTO-7449-04, SCENARIO-AUTO-7449-05, SCENARIO-AUTO-7449-06

### REQ-AUTO-7450: Seal A Replayable Prediction-Time Mixture Ledger

Exp7450 SHALL establish the causal record that Exp7440 omitted. Before a label
is available, a prediction event SHALL persist its event and source-group
identities, request order, feedback delay, fit seed, four named expert
probabilities, mixture weights, mixture probability, one hash for each expert
checkpoint, label propensity, and pre-feedback numeric state hash. The event
SHALL contain no label or label-derived value. Its content hash SHALL cover all
prediction-time fields. A feedback event SHALL reference that exact prediction
event hash.

Each feedback event SHALL record reveal order, label origin, the revealed binary
label, four losses computed from the saved expert probabilities, old and new log
weights, the unshared numeric update, stable normalizer, fixed-share update,
and parent and child state hashes. The ledger SHALL include an initial numeric
state manifest with the expert order, update rate, fixed-share rate, initial log
weights, checkpoint byte hashes, and initial state hash. A cold reader SHALL
replay the complete trajectory without importing the producer's update logic.

The cold reader SHALL reject a changed saved expert prediction, a missing or
changed checkpoint, label access before the registered reveal order, duplicate
feedback, reordered feedback, and a state/hash mismatch. Recovery from a crash
after a persisted prediction SHALL produce the same final state as an
uninterrupted replay. The registered no-feedback arm SHALL emit predictions but
SHALL keep its initial numeric state unchanged.

The analytic fixture SHALL compare the recorded update with a separately coded
scalar calculation. Exact success on this synthetic oracle SHALL be classified
`circular_positive`, with `verifier_is_oracle=true`. It establishes ledger
mechanics only, not online scientific value.

The collector protocol SHALL freeze Exp7440's two orders, delays zero and eight,
five fit seeds, seven arms, one-quarter reveal schedule, block sizes 32 and 64,
10,000 bootstrap draws, three primary comparisons, labels, and success bars.
This prototype SHALL not execute the full scientific stream. It SHALL qualify
read, predict, persist, reveal, and update as the five service-time stages for a
later cost study.

The task SHALL make no current LLM call and SHALL declare `MODEL_SPECS=[]`,
`model_invoked=false`, zero current invocation counts,
`inference_substrate_class=no_model_load`, and `execution_venue=host`.
Historical model-shaped evidence SHALL remain typed and hash-bound. Small
numeric-head fitting is historical `small_ebm_training`, not current inference.

`prediction_ledger_ready_score=1` SHALL require independent cold replay and all
causal mutation checks. `promotion_score` SHALL remain zero. The terminal JSON
SHALL publish atomically only after affected validation, fresh-process replay,
independent raw reduction, adversarial verification, and strict row consistency
pass. These readers form the capability end-to-end check. No numbered end-to-end
scenario applies because shared training, sampling, bindings, and ARC code do
not change.

#### SCENARIO-AUTO-7450-01: Prediction Bytes Precede Labels

**Given** a four-expert prediction whose feedback has a registered delay
**When** the collector persists the prediction event
**Then** all probabilities, weights, checkpoint hashes, and the pre-feedback
state hash are content-bound before reveal
**And** no label or label-derived loss is present in the prediction event.

#### SCENARIO-AUTO-7450-02: Feedback Replays Stored Expert Losses

**Given** one persisted prediction and a later revealed binary label
**When** feedback updates the learned mixture
**Then** each loss is computed from the saved expert probability
**And** old log weights, numeric update, normalizer, fixed-share result, and
state hashes permit an independent scalar replay.

#### SCENARIO-AUTO-7450-03: Causal Mutations Fail Closed

**Given** an otherwise valid prediction and feedback ledger
**When** an expert probability changes, a checkpoint disappears, feedback is
early or duplicated, or two feedback events are reordered
**Then** the cold reader rejects the ledger with a specific integrity error.

#### SCENARIO-AUTO-7450-04: Crash Recovery Is Exact

**Given** a prediction persisted before its label is revealed
**When** the collector restarts from the initial manifest and durable events
**Then** later feedback produces the same final state hash as uninterrupted
execution
**And** the prediction event hash remains unchanged.

#### SCENARIO-AUTO-7450-05: No Feedback Means No State Change

**Given** the registered no-feedback frozen-prior arm
**When** it emits predictions without feedback events
**Then** its final state hash equals its initial state hash exactly.

#### SCENARIO-AUTO-7450-06: Readiness Is Mechanical And Circular

**Given** the analytic fixture, frozen Exp7440 protocol, five service stages,
and passing terminal readers
**When** an independent reducer evaluates readiness
**Then** `prediction_ledger_ready_score` is one only if every replay and
mutation control passes
**And** the terminal class is `circular_positive` with promotion disabled.

Spec: REQ-AUTO-7450, SCENARIO-AUTO-7450-01, SCENARIO-AUTO-7450-02, SCENARIO-AUTO-7450-03, SCENARIO-AUTO-7450-04, SCENARIO-AUTO-7450-05, SCENARIO-AUTO-7450-06

### REQ-AUTO-7454: Replay continuous mixture learning with complete causal evidence

Exp7454 SHALL repeat the unchanged Exp7440 scientific comparison only after the
Exp7450 ledger reports `prediction_ledger_ready_score=1`, a verdict class in
`null`, `positive`, or `circular_positive`, and `flagged_adversarial=false`.
It SHALL authenticate the ten selected Exp7439 spline and Gibbs checkpoints and
the Exp7438 protocol bytes. It SHALL use the same 753 source groups, five fit
seeds, two label-blind orders, delays zero and eight, seven arms, randomized
one-quarter reveal schedule, action thresholds, and success bars. The result
SHALL identify this evidence as a replay of the old corpus and protocol. It
SHALL NOT describe the result as fresh deployment evidence.

Before the evaluator label is read, each arm SHALL durably persist a
content-hashed prediction event. The event SHALL include all four named expert
probabilities, its mixture weights, mixture probability, shadow action,
propensity, checkpoint hashes, and pre-feedback numeric state. Prediction bytes
SHALL contain no label or label-derived value. A later outcome event SHALL bind
the label to those prediction hashes. Each delivered-feedback event SHALL bind
its prediction hash, four losses recomputed from the saved probabilities, old
and new numeric state, normalization operands, update receipt, and lineage.
Duplicate feedback SHALL fail closed. Revoked feedback SHALL replay from the
initial state without the revoked event. The no-feedback arm SHALL emit no
update and SHALL keep its initial state.

All prespecified cells SHALL run the learned mixture, equal-weight mixture,
frozen spline, adaptive spline, no-feedback mixture, and two fixed shuffled-label
controls. Numeric checkpoints SHALL occur after each 32 source groups and at the
final group. A cold reader SHALL rehash all shards, reconstruct every mixture
weight transition without producer update code, verify checkpoint lineage, and
show that delivered feedback changes at least one later prediction state.
Segmented restart replay SHALL produce the same final state as uninterrupted
replay.

The experiment SHALL recompute full-stream and revealed proper scores,
inverse-propensity estimates, harmful-action rates, coverage, label cost, and
domain-change harm. It SHALL compare the learned arm with every registered
comparator. It SHALL use the unchanged moving-block lengths 32 and 64, 10,000
draws, and one simultaneous 12-contrast family after averaging fit seeds within
each source group. The unchanged no-harm, benefit, no-feedback, and shuffled-label
control bars SHALL decide `online_value_score`. A complete valid repeated null
SHALL retire this unchanged mixture construction. It SHALL not rename the null.

The artifact SHALL expose read, predict, durable persist, reveal, and update
timings. It SHALL also expose checkpoint hashing cost, CPU work counts, and
persisted bytes. The 100x hardware goal SHALL remain an engineering target.
The task SHALL declare `MODEL_SPECS=[]`, `model_invoked=false`, zero current LLM
calls, `inference_substrate_class=no_model_load`, `execution_venue=host`,
`continuous_self_learning_task=true`, and `no_model_weight_mutation=true`.
Historical model events SHALL remain typed, hash-bound inputs. Prior numeric
head fitting SHALL remain a separate `small_ebm_training` receipt.

`online_capture_complete_score=1` SHALL require complete cells, exact cold
replay, later-query causality, restart equality, checkpoint coverage, and all
required validation. `online_value_score` SHALL use only the unchanged Exp7440
bars. `promotion_score` SHALL remain zero. The terminal artifact SHALL publish
atomically only after scoped affected checks, declared-entrypoint cold replay,
independent raw reduction, adversarial verification, and strict row consistency
pass. These readers form the capability end-to-end check. No numbered end-to-end
scenario applies because shared training, sampling, bindings, and ARC code do
not change.

#### SCENARIO-AUTO-7454-01: Prediction evidence is label-free and durable

**Given** one source group in a registered replay cell
**When** all seven arms predict
**Then** each prediction commits four probabilities, weights, state, and propensity before label access
**And** a later outcome references the immutable prediction hashes.

#### SCENARIO-AUTO-7454-02: Stored losses replay every mixture update

**Given** delivered feedback for a persisted mixture prediction
**When** an independent reader applies the frozen scalar update
**Then** all four stored losses and normalization operands match
**And** the parent, event, and child state hashes form one exactly-once edge.

#### SCENARIO-AUTO-7454-03: Feedback controls preserve causal state

**Given** duplicate, revoked, delayed, no-feedback, and segmented-restart cases
**When** the ledger replays from its initial numeric manifest
**Then** duplicate feedback is rejected and revoked feedback is excluded
**And** no-feedback stays fixed and segmented restart equals uninterrupted replay.

#### SCENARIO-AUTO-7454-04: Later predictions depend on earlier feedback

**Given** a revealed update followed by another query in the same cell
**When** the later prediction is reduced
**Then** its pre-feedback state descends from the earlier update child
**And** the matching no-feedback state remains unchanged.

#### SCENARIO-AUTO-7454-05: Scientific reduction keeps the registered family

**Given** all 753 groups in every order, delay, seed, and arm
**When** the independent reducer computes value
**Then** it averages fit seeds within source group and uses 10,000 moving-block draws at 32 and 64
**And** benefit, no-harm, cost, no-feedback, and shuffled-label gates remain conjunctive.

#### SCENARIO-AUTO-7454-06: A valid repeated null retires the construction

**Given** complete replay and passing mechanical and validation gates
**When** any registered benefit gate fails
**Then** capture is complete, value is zero, and the verdict is `complete_null`
**And** the artifact records retirement of the unchanged mixture construction without promotion.

Spec: REQ-AUTO-7454, SCENARIO-AUTO-7454-01, SCENARIO-AUTO-7454-02, SCENARIO-AUTO-7454-03, SCENARIO-AUTO-7454-04, SCENARIO-AUTO-7454-05, SCENARIO-AUTO-7454-06

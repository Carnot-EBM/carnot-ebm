# Carnot Research Roadmap vNEXT: Certified Mapping, Convex Energy, and Queue-Regulated Learning

**Created:** 2026-09-03  
**Milestone:** 2026.09.609  
**Status:** Planned; activates after milestone 2026.09.608 closes  
**Supersedes:** milestone 2026.09.608, experiments exp6941-exp6952  
**Task contract:** exactly 12 tasks, exp6953-exp6964, in the order below  
**Research basis:** `research-program.md` and the V609 section of
`research-references.md`

## What Milestone 2026.09.608 Proved

Milestone 2026.09.608 ended with terminal records for all 12 task slots. It did
not produce science evidence for causal prefix energy, hidden-state selection,
ARC branch energy, or trace learning.

| Evidence | Result | V609 consequence |
|---|---|---|
| Roadmap contract | The Markdown and YAML both held 12 tasks, exp6941-exp6952, in the same order. | Keep a 12-task contract and check it again. |
| Source audit | Exp6941 completed its dated source coverage. | Keep one advisory post-marker source slot. |
| Contract preflight | Exp6942 returned `blocked_v608_contract_preflight`; bounded-scope and lint rows kept readiness at zero. | Make the audit advisory. No science task may gate on it. |
| Science branches | Exp6943 and exp6948 gate-blocked three times. Their descendants then skipped because the roots retired. | Use only narrow producer gates. Gate on completed data, not on a global milestone score. |
| Capstone | Exp6952 classified the contract as matched and the science as incomplete. Its partial prefix and class also exposed a verdict-guard edge. | The V609 capstone accepts incomplete evidence but emits a class-consistent terminal verdict. |

The milestone therefore proved an operational point: a correct plan can still
lose all science when one global gate controls every branch. It did not validate
the V608 scientific hypotheses.

## Three Largest Gaps to the PRD Vision

### Gap 1: Language-to-constraint mapping has no exact, current SOTA receipt

Carnot has exact solvers and many constraint representations. It still lacks a
bounded receipt showing that current local models can map two equivalent or
non-equivalent formulations into a schema that an independent solver can
certify. This blocks PRD FR12 and weakens every later verifier.

V609 uses the SOVER split: the LLM proposes a mapping; Z3 and exact enumeration
decide whether the mapping preserves domains and objective order.

### Gap 2: Energy scores have not shown causal value over strong controls

Past work found decodable or structural signals, but a score is useful only if
it changes a decision and beats likelihood, syntax, confidence, and shuffled
controls. Non-convex factor composition also creates a search confound.

V609 builds a small CCEM-inspired input-convex factor energy. It tests convexity
and local ordering first, then measures paired top-1 candidate selection on the
frozen three-family proposal bank.

### Gap 3: Continuous self-learning still lacks safe prospective utility

Exp6856 found harmful writes and weak abstention. Exp6873 did not show sealed
prospective utility. Model self-reports cannot serve as write authority.

V609 freezes model weights and learns through an external trace store. Exact
SMT certificates authorize writes. A drift-plus-penalty debt queue regulates
admission and retrieval. The test compares no memory, fixed FIFO memory, and
queue-regulated memory in chronological order with hard process resets.

## Research Inputs Added Before Design

- SOVER, arXiv:2609.00728, separates semantic mapping from SMT certification.
- Convex Compositional Reasoning Models, arXiv:2605.23395, keeps nonnegative
  factor sums convex and uses projected first-order inference.
- Self-Reports Are Not Verification, arXiv:2609.00652, shows why exact external
  outcomes must outrank confidence and rationale.
- FedQCL, arXiv:2608.21539, supplies the virtual-debt control shape used for the
  external trace store. V609 does not claim to reproduce its federated results.
- SFAD, Kona, Lean EBM, and Extropic remain watch items. They have no matching
  local runner or authenticated hardware path for this milestone.

## V609 Architecture

```text
  Exact reformulation pairs (Exp6955)
               |
       +-------+------------------+
       |                          |
       v                          v
  Three-family local GGUF     Convex factor energy
  mapping proposals           canary (Exp6958)
  (Exp6956)                        |
       |                          |
       v                          |
  Independent Z3 + exact          |
  enumeration certificates        |
  (Exp6957)                        |
       +-------------+------------+
                     v
         Label-blind energy selection
                 (Exp6959)
                     |
                     v
         Fresh-process causal audit
                 (Exp6960)

  Certified proposal rows (Exp6957)
                     |
                     v
       Sealed chronological sequence
                 (Exp6961)
                     |
                     v
   +-----------------+------------------+
   |                 |                  |
 no memory        exact FIFO       DPP debt queue
   |                 |                  |
   +-----------------+------------------+
                     v
       frozen-weight SOTA comparison
                 (Exp6962)
                     |
                     v
          cold safety audit (Exp6963)

  Exp6953 source delta ── advisory only
  Exp6954 contract audit ─ advisory only
  Exp6964 capstone ─────── ungated reconciliation
```

The exact certifier is the authority. LLM confidence, rationale, and learned
energy are candidate signals only. Exact labels stay outside prompts and
selection inputs.

## Phase A: Evidence and Exact Contract Roots

### Exp6953: V609 post-marker source delta and compatibility audit

- Recheck all requested source families after the V609 marker.
- Append only new, dated, primary-source facts.
- Record explicit no-update and unavailable rows.
- Keep this task advisory. No science task gates on it.
- Deliverable: `results/experiment_6953_v609_source_delta.json`

### Exp6954: V609 advisory execution-contract and gate-cascade audit

- Compare this document with active `research-roadmap.yaml`, which is the
  activated copy of `research-roadmap-next.yaml`.
- Require exactly 12 tasks, exp6953-exp6964, with matching titles,
  deliverables, order, and structured gates.
- Check prompt endings, producer fields, retired upstreams, prior-failure
  blocks, model contracts, and bounded scopes.
- Report failures without blocking science. No task gates on its score.
- Deliverable: `results/experiment_6954_v609_contract_advisory.json`

### Exp6955: Exact optimization-reformulation mapping fixture

- Build 120 bounded rational problems: 72 equivalent and 48 hard negatives.
- Cover integer linear, Boolean-cardinality, and bounded piecewise-linear
  formulation families.
- Freeze canonical mapping JSON, domain correspondence, objective direction,
  objective-order witnesses, splits, and hard-negative edits.
- Validate every pair with Z3 and a separate bounded exact enumerator.
- Deliverable: `results/experiment_6955_reformulation_fixture.json`

## Phase B: SOTA Mapping, Certification, and Convex Factors

### Exp6956: Three-family SOTA reformulation mapping bank

- Gate only on `exp6955.reformulation_fixture_ready_score == 1`.
- Run the same 18 held-out pairs with three candidates per pair on all three
  required GGUF families. The fixed headline budget is 162 attempts.
- Preserve raw output before parsing or exact checking.
- Do not expose labels, solver messages, or prior candidate outcomes.
- Deliverable: `results/experiment_6956_three_family_reformulation_bank.json`

### Exp6957: Independent SMT certification of SOTA mappings

- Gate only on `exp6956.reformulation_bank_complete_score == 1`.
- Parse frozen proposals and certify domain cross-feasibility and objective
  order with Z3 and bounded exact enumeration.
- Compare certification with syntax, model confidence, and self-report fields.
- Keep every malformed, timeout, unknown, equivalent, and hard-negative row.
- Deliverable: `results/experiment_6957_smt_mapping_certification.json`

### Exp6958: Convex compositional factor-energy canary

- Gate only on `exp6955.reformulation_fixture_ready_score == 1`.
- Implement a small input-convex factor model over mapping structure.
- Compare it with a parameter-matched unconstrained MLP, a linear score, and
  shuffled labels under the same splits and budgets.
- Test numerical convexity, projected-solver convergence, ordering, and size
  transfer. This is a local method canary, not a paper reproduction.
- Deliverable: `results/experiment_6958_convex_factor_energy_canary.json`

## Phase C: Causal Selection and Independent Replay

### Exp6959: Causal certified-energy candidate selection

- Gate on both `exp6957.smt_certification_run_complete_score == 1` and
  `exp6958.convex_factor_run_complete_score == 1`.
- Score frozen candidates without exact labels.
- Compare convex energy, unconstrained energy, likelihood, syntax, model
  confidence, shuffled energy, fixed order, and an oracle upper bound.
- Require paired top-1 gain over the strongest non-oracle baseline with CI95
  above zero and at least 20% of available oracle headroom for a positive.
- Deliverable: `results/experiment_6959_certified_energy_selection.json`

### Exp6960: Fresh-process certified-selection audit

- Gate only on `exp6959.certified_selection_run_complete_score == 1`.
- Recompute every arm and headline from raw proposal and certificate rows.
- Reload checkpoints in a fresh process and verify split, label, tie, budget,
  and candidate-order isolation.
- Never upgrade the upstream result.
- Deliverable: `results/experiment_6960_certified_selection_cold_audit.json`

## Phase D: Certified Self-Learning and Reconciliation

### Exp6961: Sealed chronological outcome-certificate sequence

- Gate only on `exp6957.smt_certification_run_complete_score == 1`.
- Build a prospective sequence of related but non-identical reformulations.
- Freeze event order before later outcomes are visible.
- Keep only prior exact-success certificates eligible for later retrieval.
- Require enough real opportunity for memory to help or harm. Do not fabricate
  headroom when certified rows are insufficient.
- Deliverable: `results/experiment_6961_certified_event_sequence.json`

### Exp6962: Queue-regulated continuous self-learning

- Gate only on `exp6961.certified_event_sequence_ready_score == 1`.
- Run no-memory, fixed-capacity FIFO, and drift-plus-penalty debt-queue arms.
- Use Qwen3.6-35B-A3B and Gemma-4-26B-A4B in separate fresh processes.
- Write only after an external exact certificate. Keep weights frozen.
- Charge failed or contradictory memory use as debt. Test poison, retention,
  restart, tombstone, and rollback.
- Deliverable: `results/experiment_6962_queue_regulated_self_learning.json`

### Exp6963: Fresh-process queue-memory safety audit

- Gate only on `exp6962.queue_learning_run_complete_score == 1`.
- Recompute prospective utility, debt evolution, writes, retrievals, safety,
  retention, and model immutability from raw rows and stores.
- Verify that no current or future outcome affected its own prompt or write.
- Deliverable: `results/experiment_6963_queue_memory_cold_audit.json`

### Exp6964: V609 independent capstone and V610 handoff

- Stay ungated so it can classify missing, blocked, null, disqualified, and
  positive branches.
- Recheck the exact 12-task document/YAML contract.
- Recompute all comparative claims from per-unit rows.
- Separate exact-certifier authority from circular conformance checks.
- Produce the next three evidence gaps without inventing unavailable science.
- Deliverable: `results/experiment_6964_v609_capstone.json`

## Dependency Graph

```text
exp6953  advisory source delta
exp6954  advisory contract audit

exp6955  exact fixture
  ├─> exp6956  three-family mapping bank
  │     └─> exp6957  exact mapping certification
  │             ├─> exp6959  causal selection <─ exp6958
  │             │      └─> exp6960  cold selection audit
  │             └─> exp6961  sealed certificate sequence
  │                    └─> exp6962  queue self-learning
  │                           └─> exp6963  cold memory audit
  └─> exp6958  convex factor canary

exp6964  ungated capstone
```

No task depends on exp6953 or exp6954. No task references a retired V608
upstream. Gates use run-complete or fixture-ready fields, not positive science
scores.

## Exact Task Contract

| Order | ID | Title | Deliverable | Structured gate |
|---:|---|---|---|---|
| 1 | exp6953-v609-source-delta | V609 post-marker source delta and compatibility audit | `results/experiment_6953_v609_source_delta.json` | none |
| 2 | exp6954-v609-contract-advisory | V609 advisory execution-contract and gate-cascade audit | `results/experiment_6954_v609_contract_advisory.json` | none |
| 3 | exp6955-reformulation-fixture | Exact optimization-reformulation mapping fixture | `results/experiment_6955_reformulation_fixture.json` | none |
| 4 | exp6956-three-family-reformulation-bank | Three-family SOTA reformulation mapping bank | `results/experiment_6956_three_family_reformulation_bank.json` | exp6955 `reformulation_fixture_ready_score == 1` |
| 5 | exp6957-smt-mapping-certification | Independent SMT certification of SOTA mappings | `results/experiment_6957_smt_mapping_certification.json` | exp6956 `reformulation_bank_complete_score == 1` |
| 6 | exp6958-convex-factor-energy-canary | Convex compositional factor-energy canary | `results/experiment_6958_convex_factor_energy_canary.json` | exp6955 `reformulation_fixture_ready_score == 1` |
| 7 | exp6959-certified-energy-selection | Causal certified-energy candidate selection | `results/experiment_6959_certified_energy_selection.json` | exp6957 `smt_certification_run_complete_score == 1`; exp6958 `convex_factor_run_complete_score == 1` |
| 8 | exp6960-certified-selection-cold-audit | Fresh-process certified-selection audit | `results/experiment_6960_certified_selection_cold_audit.json` | exp6959 `certified_selection_run_complete_score == 1` |
| 9 | exp6961-certified-event-sequence | Sealed chronological outcome-certificate sequence | `results/experiment_6961_certified_event_sequence.json` | exp6957 `smt_certification_run_complete_score == 1` |
| 10 | exp6962-queue-regulated-self-learning | Queue-regulated continuous self-learning | `results/experiment_6962_queue_regulated_self_learning.json` | exp6961 `certified_event_sequence_ready_score == 1` |
| 11 | exp6963-queue-memory-cold-audit | Fresh-process queue-memory safety audit | `results/experiment_6963_queue_memory_cold_audit.json` | exp6962 `queue_learning_run_complete_score == 1` |
| 12 | exp6964-v609-capstone | V609 independent capstone and V610 handoff | `results/experiment_6964_v609_capstone.json` | none |

This table is the milestone contract. `research-roadmap-next.yaml` must contain
the same 12 IDs, titles, deliverables, order, and gates.

## Hardware and Runtime Requirements

| Tasks | Substrate | Estimated time | Requirement |
|---|---|---:|---|
| exp6953-exp6955 | CPU, network only for exp6953 | 1.5-3 hours each | Python environment, Z3, writable checkpoints |
| exp6956 | Dual RTX 3090, sequential model ownership | up to 12 hours | All three mandated GGUF files, CUDA offload receipt, per-attempt checkpoints |
| exp6957-exp6961 | CPU; one GPU optional for exp6958 training | 3-6 hours each | Exact solver, bounded enumeration, saved factor checkpoint |
| exp6962 | Dual RTX 3090, sequential arm and model processes | up to 12 hours | Qwen3.6-35B-A3B and Gemma-4-26B-A4B GGUF, transactional trace store |
| exp6963-exp6964 | CPU | 3-4 hours each | Read-only artifact replay and fresh-process store load |

Each GGUF task must resolve cached files, run llama.cpp `vocab_only` probes,
verify CUDA offload, record model hashes, close the model before loading the
next family, and checkpoint every unit. Legacy small models may appear only in
CPU smoke rows.

KV260, GateMate, PolarFire, and Extropic hardware are outside the blocking
graph. V609 makes no hardware speed, power, or availability claim. A later
milestone may export a positive convex factor representation to Ising or QUBO
only after exp6958 and exp6960 justify the representation.

## Milestone Exit Criteria

- The document and YAML match on all 12 task contracts.
- The exact fixture and proposal bank have terminal per-unit rows.
- SMT certificates remain the sole correctness authority.
- Any positive energy claim beats the strongest non-oracle control on paired
  rows and survives fresh-process replay.
- The continuous-learning task uses immutable external certificates, frozen
  weights, prospective order, hard resets, debt rows, and rollback.
- Every task emits `verdict_class`, a class-consistent `honest_verdict`,
  `inference_substrate`, duration, source hashes, random seed, checksum, and
  `gate_check_summary` when blocked.
- The capstone classifies every task even if a science branch is unavailable.

## Explicit Deferrals

- Live ARC policy changes or new game-level solve claims.
- Weight updates, LoRA, GRPO, verifier-as-reward, and self-certified writes.
- Hidden-state intervention until a separate local runner receipt justifies it.
- dReal nonlinear certification until the dependency and tolerance contract are
  proven locally.
- FPGA or TSU execution, and any board speed or energy claim.
- Production default-on adoption before the independent audits pass.

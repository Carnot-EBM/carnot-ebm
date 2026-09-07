# Carnot Research Roadmap vNEXT: Generalization, Portable Memory, and Multiscale Sampling

**Milestone:** 2026.09.625  
**Date:** 2026-09-07  
**Status:** Proposed  
**Task contract:** Exactly 13 tasks, `exp7121` through `exp7133`, in the order below.

## What Milestone 2026.09.624 Proved

Milestone 2026.09.624 executed six active tasks, `exp7109` through `exp7114`.
It did not execute the six extra tasks promised by its stale Markdown design.

- Exp7109 detected the 12-row Markdown versus six-row YAML mismatch. It returned
  `complete_disqualified_v624_markdown_yaml_contract_mismatch`.
- Exp7110 shipped a fail-closed evidence-ingress quarantine. It rejects flagged,
  missing-date, malformed, absent, duplicate, and hash-drifted evidence.
- Exp7111 shipped forward ARC solve-provenance validation. It prevents a
  development proxy from becoming a credited live solve.
- Exp7112 completed the execution-time SOTA source and model check.
- Exp7113 did not measure ARC generation. Its idle-GPU check converted a valid
  zero-percent utilization value to 999 and reported both GPUs unavailable.
- Exp7114 was then blocked by its structured preflight gate. It produced no
  adapter-withheld ARC result.

The milestone improved evidence integrity. It did not establish adapter-free
ARC generalization, SOTA verifier-guided improvement, or model-facing continual
learning.

## The Three Largest Gaps to the PRD Vision

1. **Live generalization is unmeasured.** The registry records 183 reproduced
   development-proxy levels. Carnot still lacks a clean adapter-withheld number
   from the scored E3 action path.
2. **Verified reasoning has not improved current local models.** Exact solvers
   and provenance controls work. The system has not yet shown that exact,
   non-answer-revealing feedback improves outputs from the three mandated GGUF
   families.
3. **Continuous learning remains synthetic and substrate-local.** V623 showed
   delayed procedural-memory value on a deterministic stream. It did not show
   value on local-model decisions, transfer across model upgrades, or a faster
   exact sampling path for future online energy updates.

## vNEXT Architecture

```text
                         evidence quarantine
                                  |
             +--------------------+--------------------+
             |                                         |
             v                                         v
  adapter-withheld ARC E3 path              exact constraint stream
  +---------------------------+              +-----------------------+
  | observation -> SOTA GGUF  |              | SOTA GGUF candidate   |
  | -> candidate action       |              | -> grammar check      |
  | -> exact legal commit     |              | -> exact solver       |
  | -> environment outcome    |              | -> commit or revise   |
  +-------------+-------------+              +-----------+-----------+
                |                                            |
                v                                            v
       removal and leakage audit                 delayed signed memory
       no game-source access                     fixed schema + raw source
                |                                            |
                +-------------------+------------------------+
                                    |
                                    v
                       independent cold reconstruction
                                    |
                    +---------------+----------------+
                    |                                |
                    v                                v
          WCRG-inspired software path      evidence-quarantined capstone
          exact finite parity first        bounded claims only
```

Exact environment outcomes and exact constraint solvers remain authorities.
Model scores, reviewer opinions, energy values, and memory retrieval are advice.

## Exact Task Contract

| Order | Task ID | Title | Deliverable | Structured gate |
|---:|---|---|---|---|
| 1 | `exp7121-v625-contract-preflight` | V625 Markdown and YAML task-contract preflight | `results/experiment_7121_v625_contract_preflight.json` | none |
| 2 | `exp7122-v625-sota-ingestion` | V625 execution-time SOTA ingestion and method map | `results/experiment_7122_v625_sota_ingestion.json` | none |
| 3 | `exp7123-adapter-withheld-arc-loo-shard-a` | Adapter-withheld ARC leave-one-game-out shard A | `results/experiment_7123_v625_arc_loo_shard_a.json` | none |
| 4 | `exp7124-adapter-withheld-arc-loo-shard-b` | Adapter-withheld ARC leave-one-game-out shard B | `results/experiment_7124_v625_arc_loo_shard_b.json` | none |
| 5 | `exp7125-arc-loo-causal-audit` | Independent ARC leave-one-game-out causal and leakage audit | `results/experiment_7125_v625_arc_loo_causal_audit.json` | none |
| 6 | `exp7126-sota-constraint-episode-bank` | Three-family exact SOTA constraint episode bank | `results/experiment_7126_v625_sota_constraint_bank.json` | none |
| 7 | `exp7127-verifier-committed-revision` | Verifier-committed SOTA revision, gated on Exp7126 bank | `results/experiment_7127_v625_verifier_committed_revision.json` | `exp7126-sota-constraint-episode-bank.sota_constraint_bank_ready_score == 1` |
| 8 | `exp7128-delayed-fixed-schema-csl` | Delayed fixed-schema continuous learning, gated on Exp7126 bank | `results/experiment_7128_v625_model_facing_csl.json` | `exp7126-sota-constraint-episode-bank.sota_constraint_bank_ready_score == 1` |
| 9 | `exp7129-cross-model-memory-portability` | Cross-model memory portability, gated on Exp7128 learning | `results/experiment_7129_v625_memory_portability.json` | `exp7128-delayed-fixed-schema-csl.model_facing_csl_complete_score == 1` |
| 10 | `exp7130-continual-memory-cold-audit` | Independent continual-memory cold audit | `results/experiment_7130_v625_memory_cold_audit.json` | none |
| 11 | `exp7131-wcrg-multiscale-sampler-prototype` | WCRG-inspired multiscale sampler prototype | `results/experiment_7131_v625_multiscale_sampler_prototype.json` | none |
| 12 | `exp7132-frustrated-ising-sampler-benchmark` | Frustrated-Ising sampler benchmark, gated on Exp7131 prototype | `results/experiment_7132_v625_multiscale_sampler_benchmark.json` | `exp7131-wcrg-multiscale-sampler-prototype.multiscale_sampler_ready_score == 1` |
| 13 | `exp7133-v625-capstone` | V625 evidence-quarantined capstone | `results/experiment_7133_v625_capstone.json` | none |

The YAML must contain these 13 full IDs in this order. Titles, deliverables, and
structured gates must match this table exactly.

## Phase 0: Contract and Freshness

### Exp7121: V625 Markdown and YAML task-contract preflight

**Purpose:** Prevent another roadmap activation mismatch.

**Method:** Parse the active YAML and this Markdown independently. Compare all
13 IDs, titles, deliverables, gates, producer fields, model contracts, prior
failures, prompt tails, and the ungated capstone.

**Success metric:** `v625_task_contract_conforms_score=1` only when both views
match exactly. A mismatch is disqualified and does not gate science.

### Exp7122: V625 execution-time SOTA ingestion and method map

**Purpose:** Catch source, model-cache, and claim-boundary drift at execution.

**Method:** Recheck arXiv, OpenReview, Semantic Scholar, Hugging Face, GitHub,
Extropic, and Logical Intelligence. Verify the three mandated model repositories
and resolve cached files without downloading them.

**Success metric:** Complete source rows, model-path receipts, and a bounded
method map. This task does not gate later tasks.

## Phase 1: Adapter-Withheld ARC Generalization

### Exp7123: Adapter-withheld ARC leave-one-game-out shard A

**Purpose:** Produce the first bounded adapter-withheld E3 measurement after the
V623 and V624 blocks.

**Method:** Select eligibility rank one from a deterministic registry-only
predicate. Run one held-out game with Qwen3.6-35B-A3B. Compare the withheld arm
with a separately isolated adapter-visible control under the same request and
action budget. Perform runtime checks inside the task. Do not use a structured
preflight gate.

**Success metric:** Complete per-attempt receipts within 70 minutes. Zero levels
is a valid terminal null. The task cannot register or headline a solve.

### Exp7124: Adapter-withheld ARC leave-one-game-out shard B

**Purpose:** Add an independent second game and model-family cell.

**Method:** Select eligibility rank two with the same frozen predicate. Run the
Gemma4-26B-A4B model. Keep all isolation, budget, and control rules identical to
Exp7123.

**Success metric:** Complete receipts for the second disjoint game within 70
minutes. Zero remains valid.

### Exp7125: Independent ARC leave-one-game-out causal and leakage audit

**Purpose:** Decide whether the generic branch acted and whether any advisory
signal changed behavior.

**Method:** Reconstruct both shards from raw hashes and receipts. Audit imports,
file access, registry stability, model execution, and provenance. Apply
deterministic removal replay to credited world-model and verifier signals.

**Success metric:** Report level rows separately for each shard. Credit a signal
only when removal changes the selected action on the preserved candidate set.
Missing upstream evidence is blocked, not partial.

## Phase 2: Verified SOTA Reasoning and Continuous Learning

### Exp7126: Three-family exact SOTA constraint episode bank

**Purpose:** Put current local models on a small, exact, reusable stream.

**Method:** Freeze 36 chronological episodes across SAT, graph coloring,
arithmetic, and temporal constraints. Include verified paraphrases, randomized
answer codes, exact labels, and held groups. Run all three mandated GGUF models
with short deterministic outputs and no memory.

**Success metric:** All 108 headline cells have request, parse, solver, timing,
model, and exact-outcome receipts. Empty or incomplete model cells disqualify
the bank. Low accuracy is a valid terminal null.

### Exp7127: Verifier-committed SOTA revision

**Purpose:** Test the verified-turn commitment idea from Persistent Teacher
Anchoring on exact constraint actions.

**Method:** Compare single shot, same-model self-review, cross-family review,
and exact non-answer-revealing violation feedback. Commit only grammar-valid,
solver-valid actions. Preserve candidate support and rejected attempts.

**Success metric:** Paired exact-accuracy, repair, damage, abstention, and cost
rows on baseline errors with real headroom. Gate compliance alone cannot support
a positive verdict.

### Exp7128: Delayed fixed-schema continuous learning

**Purpose:** Advance PRD FR-11 on real local-model decisions.

**Method:** Process the sealed stream chronologically. Compare verifier-signed
fixed-schema procedural memory with free-form notes, equal-context raw replay,
and no memory. Freeze the decision snapshot. Reveal exact feedback afterwards.
Commit memory only between episodes. Keep model weights frozen.

**Success metric:** Report online, future-group, replay, repair, retention,
forgetting, negative-transfer, and capacity rows. A positive claim requires
paired future-group improvement with no protected-group regression.

### Exp7129: Cross-model memory portability

**Purpose:** Test whether learned constraint memory survives a model upgrade.

**Method:** Run directional writer-to-reader pairs across Qwen3.6-35B-A3B,
Gemma4-31B, and Gemma4-26B-A4B. Compare fixed-schema memory with notes, no
memory, and source-backed repair. Keep representation versions isolated.

**Success metric:** Report every direction separately. A pooled gain cannot hide
a harmful migration direction. A portability claim requires retained exact
accuracy and successful source-backed repair.

### Exp7130: Independent continual-memory cold audit

**Purpose:** Recompute the model-facing learning claims without trusting their
headlines.

**Method:** Use evidence ingress, raw rows, exact validators, and source hashes.
Recompute online, transfer, replay, repair, retention, forgetting, and
portability metrics. Check decision-before-feedback order and equal budgets.

**Success metric:** Reconstructed metrics match within declared tolerance.
Externally absent evidence produces a terminal blocked verdict.

## Phase 3: Sampling Bridge and Synthesis

### Exp7131: WCRG-inspired multiscale sampler prototype

**Purpose:** Test whether a small hierarchical proposal is technically sound on
frustrated Ising fixtures.

**Method:** Add a narrow software sampler behind the existing interface. Learn
coarse-to-fine conditional proposals on training fixtures. Use a
Metropolis-Hastings correction so the target distribution remains explicit.
Test deterministic seeding, detailed balance, and exact finite parity.

**Success metric:** `multiscale_sampler_ready_score=1` requires unit and spec
tests plus exact parity on enumerated small lattices. This is not a WCRG
replication or hardware result.

### Exp7132: Frustrated-Ising sampler benchmark

**Purpose:** Measure whether the prototype improves mixing without losing
distribution quality.

**Method:** Compare corrected multiscale proposals, local Gibbs, and the current
promoted sampler on frozen frustrated and unfrustrated fixtures. Use five seeds.
Report total variation or KL where exact enumeration is available. Report ESS,
autocorrelation, acceptance, wall time, and degree-16 placement overhead.

**Success metric:** Any speed or mixing claim requires exact-quality parity and
per-instance rows. Host software cannot support FPGA, TSU, power, or asymptotic
claims.

### Exp7133: V625 evidence-quarantined capstone

**Purpose:** Reconcile the milestone without repeating a flagged-input claim.

**Method:** Ingest all expected artifacts through the V624 quarantine. Recompute
task counts, gates, verdict classes, model use, ARC provenance, learning claims,
sampling claims, exclusions, and unresolved blockers.

**Success metric:** Produce a claim-to-evidence matrix. Exclude flagged or
malformed artifacts. Keep the capstone ungated. Use blocked for external gaps
and partial only for unfinished capstone work.

## Dependency Graph

```text
exp7121 contract preflight (advisory)        exp7122 SOTA ingestion (independent)

exp7123 ARC shard A ----+
                         +--> exp7125 ARC causal and leakage audit
exp7124 ARC shard B ----+

exp7126 exact SOTA bank --+--> exp7127 verifier-committed revision
                          |
                          +--> exp7128 delayed fixed-schema CSL
                                      |
                                      +--> exp7129 memory portability

exp7126/27/28/29 ---------------------------> exp7130 cold audit

exp7131 multiscale prototype --> exp7132 frustrated-Ising benchmark

all available clean artifacts ----------------> exp7133 ungated capstone
```

Only Exp7127, Exp7128, Exp7129, and Exp7132 have structured gates. Their
producer fields appear as bare top-level required artifact fields in the
upstream tasks. Neither ARC shard has a preflight gate.

## Hardware Requirements

| Resource | Tasks | Requirement and boundary |
|---|---|---|
| Dual RTX 3090 GPUs | Exp7123, Exp7124, Exp7126-Exp7129 | Use cached GGUF files, one bounded server per lease, telemetry, cleanup, and no downloads. |
| Host CPU and RAM | All tasks; especially Exp7131-Exp7132 | Run exact solvers, evidence audits, finite enumeration, and software sampling. |
| Local storage | Model and raw-trace tasks | Keep bulky raw generations outside `results/`. Store paths, hashes, sizes, and dates in artifacts. |
| KV260 and PolarFire | None on the critical path | Existing terminal receipts remain valid. No new physical-board claim is needed. |
| GateMate | None | The board remains physically blocked. |
| Extropic Z1 or TSU | None | No authenticated device is attached. Degree-16 software rows are not hardware evidence. |

## Claim and Failure Boundaries

- Every task writes `run_date`, `verdict_class`, `field_principles`, substrate,
  venue, duration, source hashes, and blocked diagnostics.
- Every comparative task emits per-unit rows.
- Every LLM task declares `MODEL_SPECS` and runs at least one mandated model in
  a headline cell. Legacy small models may only smoke-test transport.
- ARC rows declare `solve_provenance=development_proxy`,
  `headline_solve_eligible=false`, and `arc_registry_delta=0`.
- A complete zero-level, zero-gain, or no-headroom result is `null`, not
  `partial`.
- A missing upstream artifact or failed external prerequisite is `blocked`, not
  `partial`.
- Exact verifiers decide validity. Learned energy, reviewers, and memory do not
  become proof authorities.
- Physical hardware speed, energy, and execution claims are out of scope.

## Explicitly Deferred

- Public ARC solve claims or registry updates from development-proxy LOO runs.
- Foundation-weight updates or LoRA continual training.
- Another external-text Phase D energy scorer.
- Reopening retired PWA-KAN ranking or finite-ID answer transport.
- Full WCRG replication, asymptotic scaling, FPGA deployment, or Z1 execution.

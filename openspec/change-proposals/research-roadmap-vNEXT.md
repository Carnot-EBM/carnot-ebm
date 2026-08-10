# Carnot Research Roadmap vNEXT: Executable Constraints and Certified Online Memory

**Created:** 2026-08-10
**Milestone:** 2026.08.541
**Status:** Planned
**Supersedes:** milestone 2026.08.540 (Exp6260-Exp6271, terminal)
**Informed by:** the V540 exact-path capstone, Exp6263 replay bridge,
Exp6264 familiarity controls, Exp6268 exact fixtures, Exp6269 backend failure,
the V541 source refresh, and the current ARC root-cause audit

The architecture document was last reconciled on 2026-07-03. It is more than
30 days old. V541 uses exact artifacts and current ops evidence when they
conflict with that document.

## What the Previous Milestone Proved

| Approach | Evidence | Finding |
|---|---|---|
| V540 transition and source freeze | Exp6260-Exp6261 | Both tasks wrote honest blocked artifacts. Broad validation and timeout failures prevented readiness. |
| Terminal-artifact classifier | Exp6262 | Focused classifier, coverage, spec, and lint checks passed. A repository-wide Python suite timed out, so the readiness field stayed zero. |
| Immutable SOTA replay | Exp6263 | The replay bridge passed current adversarial rules. It preserved clean Exp6160 and Exp6162 rows and quarantined warned Exp6146 rows. |
| Energy familiarity | Exp6264 | A global threshold gave positive utility and zero unsafe shifted advice. The task-conditioned gate abstained on all 160 shifted test rows. A broad suite failure closed the gate. |
| Continuous learning chain | Exp6265-Exp6267 | The readiness gate failed closed. No chronological learner, holdout audit, or shadow consumer ran. |
| Exact sampler suite | Exp6268 | The suite was ready across Ising, Potts, and typed-factor fixtures. |
| Multi-family mode jumping | Exp6269 | The backend supported only the original six-state vector. Seven of eight fixture types were unsupported. No safety or workload claim was promoted. |
| Capstone | Exp6271 | The capstone kept every blocked, skipped, null, and ready state separate. It promoted no continuous-learning or sampler claim. |

**The gap:** V540 supplied two strong foundations: immutable verifier-labeled
events and exact multi-family sampler fixtures. The next failures are now
mechanism failures. The familiarity policy lacks certified admission under
shift. The sampler ABI assumes one fixed vector shape. Carnot also lacks an
executable declarative bridge between natural-language constraint tasks and an
inspectable energy.

## The Three Largest PRD Gaps

| Gap | Current state | PRD target | V541 response |
|---|---|---|---|
| Executable constraint semantics for live local models | Exact validators exist, but prior structured-output lanes often failed to parse or had zero semantic success. | FR12 requires deterministic verification. The long-term vision requires real constraint handling rather than prompt-only confidence. | Build a bounded ASP-to-energy compiler. Check it against an independent ASP solver. Run all three mandated flagship GGUF families on ordinary candidate assignments with sealed formal sidecars. |
| Continuous self-learning reaches a safe decision path | Exp6263 is ready. Exp6264 found a useful global control, but its task gate abstained under shift. The learning chain never ran. | FR11 requires online adaptation, immutable validation, rollback, and retained capability. | Replace the failed threshold with positive and negative caches, entropy and diversity admission, a frozen reserve, and a certified impurity-slope bound. Then run chronological learning, heldout audits, and a default-off shadow consumer. |
| Exact sampling supports real fixture shapes | Exp6268 is ready, but Exp6269 used a fixed six-state backend. | FR07 needs general exact or approximate sampling with auditable quality. The hardware path needs a portable sampler ABI. | Add typed variable-cardinality state support to Rust and PyO3. Re-run the exact multi-family A/B only after parity and shape controls pass. |

The ARC live-agent floor remains separate from these gaps. V541 adds one
mechanic-class detector to the live path. It makes no game-level solve claim.

## V541 Architecture: Formal Semantics to Certified Memory

```text
               Sealed natural-language constraint tasks
                              │
                  formal ASP sidecar + exact answer
                              │
                              ▼
              ASP-to-energy compiler (Exp6274)
                │                         │
                │                         └──► independent ASP solver
                │                                  exact certificate
                ▼
   Qwen3.6-35B / Gemma-4-31B / Gemma-4-26B-A4B
              candidate assignments (Exp6275)
                              │
                              ▼
          parse margin + semantic margin + repair receipt
                              │
                              └──────────────┐
                                             ▼
  Clean V540 verifier events ──► certified dual cache (Exp6276)
                                             │
                                             ▼
                               chronological CSL (Exp6277)
                                             │
                                             ▼
                              family/task holdout (Exp6278)
                                             │
                                             ▼
                              default-off shadow (Exp6279)

  Exp6268 exact fixtures ──► variable-cardinality ABI (Exp6280)
                                             │
                                             ▼
                               exact multi-family A/B (Exp6281)

  ARC live traces ──► mechanic-class router (Exp6282; no solve claim)

  All exact paths ───────────────────────────► capstone (Exp6283)
```

Exp6275 and Exp6282 run live local LLM inference. Exp6275 uses all three
mandated flagship families. Exp6282 uses the mandated Qwen flagship. The other
tasks use exact CPU methods or immutable upstream evidence. Legacy small models
can run smoke tests only. They cannot supply headline rows.

## Phase 0: Evidence Handoff and Executable Semantics (Exp6272-Exp6274)

### Exp6272: V540-to-V541 exact terminal transition

Archive V540 by exact declared paths. Preserve the blocked broad-suite results
without hiding focused passes. Reserve Exp6272-Exp6283 after a tracked and
untracked collision scan. Validate the staged roadmap without editing the active
roadmap or conductor.

**Deliverable:** `results/experiment_6272_v541_terminal_transition.json`

### Exp6273: Post-marker source delta and scope freeze

Search only evidence later than the V541 planner marker. Append only stable,
non-duplicate sources. Freeze the ASP semantics, flagship benchmark, certified
cache, sampler ABI, ARC provenance, and hardware claim boundaries. A null delta
is a valid terminal result.

**Deliverable:** `results/experiment_6273_v541_post_marker_source_scope_freeze.json`

### Exp6274: Bounded ASP-to-energy semantic compiler

Implement a bounded ASP subset with facts, grounded rules, default negation,
integrity constraints, and cardinality constraints. Compile each ground rule to
an inspectable energy term. Compare every candidate state with an independent
ASP solver. Use at least 40 fixtures across graph, scheduling, default,
contradiction, and control families. The solver and compiler are oracles derived
from the same formal theory. Make no verifier-moat claim.

**Deliverable:** `results/experiment_6274_asp_energy_semantic_compiler.json`

## Phase 1: Live Flagship Constraint Verification (Exp6275)

### Exp6275: Three-family local-GGUF ASP constraint benchmark

Run at least 30 sealed tasks per model. Use
`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`. Models return ordinary candidate
assignments. They do not author ASP theories or ConstraintIR. The exact formal
sidecar stays hidden from the model and supplies the oracle.

Compare one-shot output, fixed-budget self-consistency, and energy-guided
repair. Report format recovery and semantic recovery separately. Record exact
certificate coverage, residual violations, abstention, latency, GPU offload,
VRAM, seeds, and raw hashes. Run models serially. Do not use a text scorer or a
legacy model for a headline result.

**Deliverable:** `results/experiment_6275_flagship_asp_constraint_verification_benchmark.json`

## Phase 2: Certified Continuous Self-Learning (Exp6276-Exp6279)

### Exp6276: Certified dual-cache admission

Replace the failed task threshold. Keep separate positive and negative caches.
Admit rows through entropy and diversity gates. Calibrate only from a frozen
reserve. Estimate the adaptive-memory impurity reproduction number and its
upper confidence bound. Readiness requires the upper bound below one, useful
coverage, zero unsafe held-shift advice, and exact rollback.

**Deliverable:** `results/experiment_6276_certified_dual_cache_admission.json`

### Exp6277: Chronological certified self-learning A/B

Run the required continuous self-learning experiment. Keep GGUF weights frozen.
Use a read-only pre-decision snapshot. Commit only after the exact outcome.
Compare no memory, the V540 global threshold, unconditional memory, and the
certified dual cache. Use the clean Exp6263 stream. Add clean Exp6275 events only
when their corpus readiness field equals one.

Measure forward transfer, retention, forgetting, negative transfer, cache
purity, quarantine, memory cost, and byte-identical rollback. Preserve results
by model family and task class before any pooled summary.

**Deliverable:** `results/experiment_6277_chronological_certified_csl_ab.json`

### Exp6278: Model-family and task-family holdout audit

Audit the promoted policy without refitting held partitions. Run
leave-one-model-family-out and leave-one-task-family-out tests. Preserve small
or failing strata. Readiness requires positive transfer, retained known-task
quality, bounded unsafe advice, stable impurity certification, and exact replay.

**Deliverable:** `results/experiment_6278_model_family_task_holdout_csl_audit.json`

### Exp6279: Default-off certified-memory shadow consumer

Wire the audited policy into the canonical constraint decision path in shadow
mode. Keep the live decision byte-identical to control. Read one immutable cache
snapshot per decision. Write memory only after the exact outcome boundary.
Prove restart, quarantine, and rollback identity. Keep the default off.

**Deliverable:** `results/experiment_6279_certified_memory_shadow_consumer.json`

## Phase 3: Portable Sampling, ARC Floor, and Reconciliation (Exp6280-Exp6283)

### Exp6280: Variable-cardinality Rust/PyO3 mode-jump backend

Replace the fixed `shape (6,)` contract with typed state metadata. Support
binary Ising, multistate Potts, and bounded typed-factor fixtures. Add explicit
cardinality, shape, encoding, and proposal-domain checks. Prove Rust/Python
parity, treatment activation, and rejection of malformed metadata. This task
does not compare scientific outcomes.

**Deliverable:** `results/experiment_6280_variable_cardinality_mode_jump_backend.json`

### Exp6281: Exact multi-family mode-jump A/B

Re-run the V540 scientific comparison only after Exp6280 passes. Use the frozen
Exp6268 fixture hashes. Hold targets, seeds, burn-in, retained samples, proposal
budgets, and schedules fixed. Report exact distribution error, energy error,
mode occupancy, barrier crossings, autocorrelation, ESS, acceptance, and
descriptive wall time. Use equivalence-aware intervals. Make no hardware or
speedup claim.

**Deliverable:** `results/experiment_6281_mode_jump_multifamily_rerun.json`

### Exp6282: ARC live-path mechanic-class router

Add one detector for the current missing mechanic class. Use push-block and
toggle/move synthetic controls. Route features into the existing live inducer
without hidden game source access, offline ground-truth search, or a per-game
adapter. Run a bounded live-agent canary with the mandated Qwen flagship. Score
mechanic classification, proposal coverage, invalid action rate, and treatment
activation. Make no level solve claim. Do not update the solve registry.

**Deliverable:** `results/experiment_6282_arc_mechanic_class_live_router.json`

### Exp6283: V541 adversarial capstone

Read every declared deliverable by exact path. Keep terminal, nonterminal,
blocked, skipped, null, flagged, and ready states separate. Re-run current
adversarial rules. Reconcile specs and ops evidence only to files that exist.
Report ASP, self-learning, sampler, and ARC branches independently.

**Deliverable:** `results/experiment_6283_v541_adversarial_capstone.json`

## Dependency Graph

```text
Exp6272 transition
   └── Exp6273 source/scope freeze
          ├── Exp6274 ASP semantic compiler
          │      └── Exp6275 live flagship benchmark
          ├── Exp6276 certified cache admission
          │      └── Exp6277 chronological CSL
          │             └── Exp6278 holdout audit
          │                    └── Exp6279 shadow consumer
          ├── Exp6280 variable-cardinality backend
          │      └── Exp6281 exact multi-family A/B
          └── Exp6282 ARC mechanic-class router

Exp6273-Exp6282 ───────────────────────────────► Exp6283 capstone
```

Structured gates:

- Exp6275 requires `asp_energy_semantic_ready_score == 1` from Exp6274.
- Exp6277 requires `certified_admission_ready_score == 1` from Exp6276.
- Exp6278 requires `continuous_learning_promotion_ready_score == 1` from Exp6277.
- Exp6279 requires `heldout_certified_transfer_ready_score == 1` from Exp6278.
- Exp6281 requires `variable_cardinality_backend_ready_score == 1` from Exp6280.
- Exp6283 is ungated. It treats gate skips as evidence.

The self-learning chain does not depend on Exp6275. This prevents a local GPU
runtime failure from hiding the central FR11 experiment. Exp6277 can consume a
ready Exp6275 corpus as a held-forward stratum.

## Hardware Requirements

| Experiments | Compute | Memory | Expected time | Claim boundary |
|---|---|---:|---:|---|
| 6272-6274 | CPU | 8-16 GB | 20-120 min each | Exact software and source evidence only |
| 6275 | 2x RTX 3090, serial model loads, CPU ASP solver | 24 GB VRAM per GPU, 32 GB RAM | 3-6 h | Live local-GGUF quality and descriptive cost only |
| 6276-6279 | CPU, immutable result files | 16 GB | 30-120 min each | Frozen-weight external memory only |
| 6280-6281 | CPU and Rust/PyO3 build | 8-16 GB | 45-180 min each | Software sampler quality; no hardware speedup |
| 6282 | 1x RTX 3090 plus ARC runtime | 24 GB VRAM, 16 GB RAM | 1-3 h | Live-path mechanism evidence; no level solve |
| 6283 | CPU | 8 GB | 30-60 min | Reconciliation only |

The two RTX 3090 GPUs are the only scheduled accelerators. KV260 is terminal
for current work. GateMate lacks a new physical JTAG receipt. PolarFire remains
opportunistic. Extropic exposes no authenticated device or simulator route.
V541 schedules no FPGA or TSU task. It makes no power or energy-efficiency
claim.

## Explicitly Deferred

- LLM-authored ASP theories. The retired structured-output lane stays closed.
- Hidden-state or external-text learned scorers. Deterministic ASP is the only
  verifier in the live flagship benchmark.
- Parameter updates to any GGUF model. V541 self-learning uses external memory.
- Descriptor routing for mode jumping. V541 first proves shape support and
  multi-family value.
- PTT, MetaDNS, NCE, and new learned-sampler training. The fixed backend failure
  comes first.
- ARC level solve claims. Exp6282 changes the live path and does not solve for
  the agent.
- FPGA, TSU, Kona, speed, power, or availability claims without a new receipt.

## Milestone Exit Criteria

V541 is successful when it produces these honest decisions:

1. The ASP compiler matches an independent solver on all supported fixtures.
   The live flagship benchmark then reports exact format and semantic margins,
   or records a terminal runtime block.
2. The certified cache either keeps its impurity upper bound below one and
   reaches chronological, heldout, and shadow evaluation, or records the unsafe
   stratum that blocks promotion.
3. The variable-cardinality backend supports every frozen fixture family. The
   exact A/B then finds workload value, equivalence, harm, or an honest block.
4. The ARC detector reaches the live agent and fires on its positive controls.
   No hidden-source, offline solve, or game-level claim enters the evidence.

No branch needs a positive result for the capstone to complete.

# Carnot Research Roadmap vNEXT: Diagnostic Verification, Transactional Learning, and Compiler Fidelity

**Milestone:** `2026.08.587`  
**Created:** 2026-08-29  
**Status:** Proposed  
**Supersedes:** milestone `2026.08.586`  
**Research basis:** `research-program.md`, `_bmad/prd.md`, `_bmad/architecture.md`,
`ops/status.md`, `ops/changelog.md`, `research-complete.yaml`, `ops/conductor-log.md`, and the
`V587 Planner Refresh` in `research-references.md`

## What milestone 2026.08.586 proved

Milestone `.586` completed honestly but did not execute its paired science run. Its only task,
`exp6716-object-table-fetch-on-demand-ab`, terminated with:

> `blocked_context_window: CARNOT_ARC_INDUCE_N_CTX is unset; the paired A/B did not start`

That is useful operational evidence, not a null result on fetch-on-demand. It proves four bounded facts:

1. The object table remains the one large ARC induction block worth testing: about 54-56% of the
   measured prompt, while keyframe/delta frame compression is already shipped.
2. The default-ON table cannot be removed casually because the prior 20-game held-out experiment
   measured `mean_delta_on_minus_off=+0.0720` on `change_fidelity`.
3. The selfparse path still has no live code-carrying, multi-parameter tool receipt. Its 20/20 live
   success covered only zero-argument `list_transitions`; `find_objects` remains the first real test.
4. The conductor prompt knew that a single-stream run requires 32K context, but the task did not own
   that setting. A known, sufficient local resource became an external shell precondition. `.587`
   fixes the contract by setting and validating `CARNOT_ARC_INDUCE_N_CTX=32768` inside the task.

The immediate inherited baseline is equally important. `.580-.585` produced a 48-row direct exact
certificate corpus with only 8/48 exact successes and two rows of measured regeneration headroom;
an exact finite-horizon planning fixture exists, but repeated audit/proposal chains died at missing
artifacts and hard walls; prospective repair memory had a positive point estimate, but its cold audit's
order-level interval included zero; and Torx exact references existed while installed-runtime parity
failed. The next milestone therefore changes the experimental substrates instead of replaying those
chains.

## The three largest gaps to the PRD vision

| Rank | PRD gap | Current evidence | `.587` response |
|---|---|---|---|
| 1 | **FR12 lacks an end-to-end, non-circular verification path on current local SOTA models.** | Exact validators exist, but direct proposals are mostly invalid; planning-energy chains were gate-blocked; scalar failures do not separate bad reasoning from bad neural-to-symbolic translation. | Build a hardness-controlled proof-carrying corpus, dual symbolic encodings, a held-family oracle-distinct diagnostic energy, and a localized repair A/B whose final authority remains an exact solver. |
| 2 | **FR11 continuous self-learning is not yet prospective, durable, and statistically credible.** | A repair-memory point estimate was positive, but the independent order-level interval included zero; a later prequential branch never ran. | Use read-only active episodes, exact between-episode commit receipts, byte-exact rollback, multiple chronological orders, best@k support, retention, and poison audits. |
| 3 | **The live ARC and hardware-portability bridges are incomplete.** | ARC selfparse has not carried live code arguments; `.586` never ran. Torx parity failed and Extropic silicon is not locally reachable. | Run the first code-carrying `find_objects` call and the paired object-table A/B on `E3AgentPolicy`; separately measure Thermalizers-style factor KL and trajectory drift in a simulator-only compiler-fidelity experiment. |

## Research deltas adopted in this milestone

- **SymDiag (`2608.08786`)** supplies the dual-encoding `TranslationError` versus
  `ReasoningError` split used in Phase 2.
- **Solver-Hard Is Not Model-Hard (`2607.17047`)** supplies proof-hard/proof-easy,
  density-matched families and proof-preserving surface relabeling.
- **PARTAB (`2608.24082`)** strengthens the object-table plan: hierarchical, row-linked evidence is
  fetched when needed instead of shipping a flat full table.
- **Memoir (`2607.20792`)** motivates read-only within-episode memory and delayed commits.
- **Verifier-Induced Support Reshaping (`2608.00220`)** makes best@k/effective support and rollback
  mandatory self-learning outcomes.
- **Thermalizing Stochastic Programs (`2608.01615`)** and **Torx (`2608.01612`)** define the
  factor-error-to-trajectory-error hardware-preparation target.
- **Parser Stack Classification (`2608.03065`)** is a measured sidecar only. It does not become a
  dependency unless the local runtime actually invokes it.

## vNEXT architecture

```text
                         EXACT AUTHORITY BOUNDARY
                         (never a learned self-score)

  Local SOTA GGUF ──► proof-carrying candidate ──► exact SAT/certificate checker
        │                         │                         │
        │                         ├── encoding A ──────────┤
        │                         └── encoding B ──────────┤
        │                                      disagreement│
        ▼                                                  ▼
  structural features ──► oracle-distinct diagnostic energy ──► error locus
                                                                  │
                                                                  ▼
                                                     localized backtracking
                                                                  │
                                                                  └──► exact recheck

  chronological exact outcome stream
        │
        ├── active episode: READ ONLY
        ▼
  exact admission receipt ──► atomic between-episode commit ──► versioned memory
        ▲                                                        │
        └──────── support/retention/poison gate ◄── rollback ─────┘

  E3AgentPolicy / make_carnot_agent                 Hardware-preparation sidecar
        │                                                      │
  inline object table ── A/B ── fetch find_objects       typed stochastic kernels
        │                                                      │
  same games/seeds, Qwen3.8 live pin                     factor-to-EBM compilation
        │                                                      │
  tokens + change_fidelity + transition utility          exact KL + trajectory TV
```

The architecture preserves two non-negotiable boundaries. Exact solvers certify current candidates;
the learned energy may prioritize or localize but cannot certify itself. ARC credit stays on the live
`E3AgentPolicy` / `make_carnot_agent` path using the agent's own observations and runtime tools. No
game source, exhaustive ground-truth BFS, per-game adapter, or duplicate level solve is introduced.

## Phase 1: Evidence contract and live ARC tool path

### Exp 6729: V587 activation and evidence contract

Freeze the 13-task doc-to-manifest map, primary-source receipts, unique IDs/deliverables, structured
gate fields, prior-failure declarations, model policy, and claim boundaries. This is the required
bleeding-edge ingestion slot and the first of two infrastructure slots.

**Deliverable:** `results/experiment_6729_v587_activation_evidence_contract.json`

### Exp 6730: Owned 32K context and code-carrying selfparse preflight

Move the 32K setting into the subprocess that owns the run. Verify CUDA offload, both cached model
paths, context admission, XML parse, multi-parameter dispatch, tool response, and a bounded
`find_objects` return. Qwen3.6-35B-A3B is the mandated SOTA transport canary; Qwen3.8-27B remains
the immutable scored ARC generator. No game-level result is measured.

**Deliverable:** `results/experiment_6730_arc_context_tool_preflight.json`

### Exp 6731: Object-table fetch-on-demand A/B

Rerun `.586` on the same 20 held-out games and seeds as the 2026-08-01 object-perception design.
Compare default inline-table ON with table-absent plus live `find_objects` fetch. The adoption gate is
non-inferiority in `change_fidelity` within the preregistered noise floor, positive realized token
savings, and a non-zero successful fetch rate. A loss retires the direction. This is an ARC path-quality
experiment, not a solve experiment.

**Deliverable:** `results/experiment_6731_object_table_fetch_on_demand_ab.json`

### Exp 6732: Cold object-table row and provenance audit

Recompute every pair from raw prompts/tool transcripts and verify games, seeds, model identity,
context settings, token counts, tool arguments/results, transition outcomes, and paired statistics.
The audit cannot replace missing rows and cannot convert a negative A/B into a positive result.

**Deliverable:** `results/experiment_6732_object_table_ab_cold_audit.json`

## Phase 2: Certified diagnostic energy and guided repair

### Exp 6733: Hardness-controlled exact certificate stream

Generate a task-owned stream of near-density-matched expander-Tseitin, ladder-Tseitin, and
pigeonhole-anchor instances across fixed size bins. Produce proof-preserving relabelings, exact
SAT/UNSAT labels, satisfying assignments or independently checkable UNSAT certificates, solver
conflict receipts, and immutable train/dev/test family splits. Solver conflicts are metadata, never a
model-hardness label.

**Deliverable:** `results/experiment_6733_hardness_controlled_certificate_stream.json`

### Exp 6734: Three-family SOTA dual-encoding proposal corpus

Run all three mandated local GGUFs sequentially on the frozen stream. Require a proof-carrying DSL,
parse each response through two independently implemented encoders, retain every failure and
abstention, and exact-check the proposed certificate. The corpus is useful even if exact success is low;
readiness means complete, attributable rows rather than a flattering success threshold.

**Deliverable:** `results/experiment_6734_sota_dual_encoding_proposal_corpus.json`

### Exp 6735: Oracle-distinct diagnostic energy

Train a compact energy on structural, pre-oracle features from one set of families and evaluate on
held-out families and proof-preserving relabelings. Compare dual-encoding features with single-encoding
and undifferentiated scalar-failure baselines. The exact checker supplies training/evaluation labels but
no current-row exact outcome, solver conflict count, answer key, or certificate validity bit may enter
the energy input. The downstream gate requires held-out reasoning-error AUROC at least 0.65 and zero
detected oracle leakage.

**Deliverable:** `results/experiment_6735_oracle_distinct_diagnostic_energy.json`

### Exp 6736: Diagnostic-energy localized repair A/B

On frozen failed proposals, compare no repair, full regeneration, and localized backtracking driven by
the diagnostic energy. Use the same model, prompt budget, candidate budget, and exact final verifier per
row. Report exact success, harmful flips, tokens, latency, abstention, and error-type movement. Positive
credit requires the paired lower confidence bound for localized repair over full regeneration to exceed
zero without higher harmful-flip rate.

**Deliverable:** `results/experiment_6736_diagnostic_energy_localized_repair_ab.json`

## Phase 3: Transactional continuous self-learning

### Exp 6737: Read-only episode and atomic commit fixture

Construct a chronological cross-family stream from immutable Phase 2 instances plus unseen variants.
The active episode sees a read-only memory snapshot. Only an exact-certified future-useful repair recipe
may commit between episodes; every commit has parent hash, evidence hash, scope, TTL, and inverse patch.
Crash/restart, duplicate, contradiction, stale evidence, poison, and byte-exact rollback tests must pass
before live SOTA evaluation.

**Deliverable:** `results/experiment_6737_transactional_constraint_memory_fixture.json`

### Exp 6738: Prospective support-preserving self-learning A/B

Compare frozen/no-memory with transactional memory on preregistered chronological orders. Use
Qwen3.6-35B-A3B for acquisition and both Qwen3.6 and Gemma4-31B for held-out transfer. Measure
prequential exact yield, best@k/effective rewardable support, joint correct-and-constraint-following
support, retention anchors, negative transfer, commits, rejects, and rollbacks. The memory never writes
during the active generation episode and no weights change.

**Deliverable:** `results/experiment_6738_prospective_support_preserving_csl_ab.json`

### Exp 6739: Cold self-learning durability and poison audit

Recompute the chronological result from rows, run order-level intervals, verify no future leakage,
replay poison/stale/conflict attacks, restart from every commit boundary, and prove rollback byte
identity. Positive credit requires an order-level lower confidence bound above zero, no meaningful
best@k support contraction, no anchor forgetting, and zero admitted poison.

**Deliverable:** `results/experiment_6739_csl_support_durability_audit.json`

## Phase 4: Stochastic compiler fidelity and milestone synthesis

### Exp 6740: Thermalizers-style factor-to-EBM compiler fidelity

Implement a bounded, exact-enumeration reference for typed stochastic kernels and compile them into
sparse Ising/EBM factors. Compare independent factor fitting, context matching, and trajectory-level
refinement over depths 1/2/4/8. Report every factor's conditional KL and every circuit's accumulated
trajectory total variation, topology, precision, seeds, and optional official-Torx conformance. This is
simulator-only hardware preparation. It makes no X0/Z1, FPGA, speed, power, or energy-efficiency claim.

**Deliverable:** `results/experiment_6740_thermalizer_compiler_fidelity.json`

### Exp 6741: V587 branch disposition and PRD gap update

Independently synthesize the four branches from their rows and receipts, run adversarial verification on
all milestone artifacts, classify every branch as positive/null/blocked/disqualified/partial, and state
which PRD gaps actually narrowed. Missing or blocked branches remain missing or blocked; no pooling can
manufacture a milestone-level positive.

**Deliverable:** `results/experiment_6741_v587_branch_disposition.json`

## Dependency graph

```text
Exp6729 activation/evidence contract
  └── Exp6730 32K + selfparse preflight
        └── Exp6731 object-table A/B
              └── Exp6732 cold ARC audit

Exp6733 hardness-controlled certificate stream
  └── Exp6734 SOTA dual-encoding corpus
        └── Exp6735 diagnostic energy
              └── Exp6736 localized repair A/B

Exp6737 transactional memory fixture
  └── Exp6738 prospective CSL A/B
        └── Exp6739 durability/poison audit

Exp6740 compiler fidelity  (independent after Exp6729)

Exp6732 ─┐
Exp6736 ─┼──► Exp6741 branch disposition
Exp6739 ─┤
Exp6740 ─┘
```

Conductor gates use field names emitted verbatim by the upstream task:

- `v587_contract_ready`
- `arc_context_tool_preflight_ready`
- `object_table_ab_completed`
- `hardness_stream_ready`
- `dual_encoding_corpus_ready`
- `heldout_reasoning_error_auroc` and `oracle_leakage_detected`
- `transaction_stream_ready`
- `csl_run_completed`

The capstone is deliberately ungated so it can synthesize honest blocked branches and missing-artifact
diagnostics rather than disappearing behind the first failure.

## Local model policy

Every task that invokes an LLM includes a mandated current GGUF in `MODEL_SPECS`:

| Role | Model |
|---|---|
| Flagship MoE | `unsloth/Qwen3.6-35B-A3B-GGUF` |
| Flagship dense | `unsloth/gemma-4-31B-it-GGUF` |
| Middle MoE | `unsloth/gemma-4-26B-A4B-it-GGUF` |
| Immutable live ARC generator | `unsloth/Qwen3.8-27B-GGUF` (used only with a mandated Qwen3.6 transport canary in the same experiment) |

Phase 2 uses all three mandated models. Phase 3 uses Qwen3.6-35B-A3B for acquisition and
Gemma4-31B for cross-family transfer. Legacy Qwen3.5-0.8B and Gemma4-E4B are permitted only for
CPU smoke tests and cannot contribute headline rows. All model loads are sequential; a missing model
writes a blocked artifact rather than silently substituting a legacy model.

## Hardware requirements

| Experiments | Required resources | Expected wall time | Claim boundary |
|---|---|---:|---|
| 6729, 6732, 6733, 6735, 6737, 6739, 6741 | CPU, 16-32 GB RAM | 5-90 min each | No new LLM inference; deterministic generation/replay/audit. |
| 6730 | One RTX 3090, CUDA-capable `llama.cpp`, cached Qwen3.8 and Qwen3.6 GGUFs, 32K context | 30-60 min | Transport/preflight only; no ARC quality or solve claim. |
| 6731 | One RTX 3090 for sequential Qwen3.8 inference; second 3090 optional for non-overlapping canary/setup | 3-6 h | Same 20 games/seeds; no game-level solve claim. |
| 6734, 6736 | Dual RTX 3090 available, but models loaded sequentially to avoid shared-memory/OOM confounds | 3-8 h each | Local GGUF generation; exact CPU verifier is final authority. |
| 6738 | Dual RTX 3090, sequential Qwen/Gemma runs, checkpointed chronological stream | 6-12 h | External transactional memory only; no weight learning claim. |
| 6740 | CPU/JAX; CUDA optional for matched simulator timing; official Torx optional | 1-3 h | Simulator/compiler fidelity only; no physical-hardware or speedup claim. |

No attached FPGA board is required for `.587`. KV260 is near terminal, PolarFire's current workload
validation is terminal for its declared scope, and GateMate remains blocked on physical JTAG. Repeating
unchanged board smokes would violate continuity discipline. Extropic reports Z1 early access in 2027;
Carnot has no authenticated X0/Z1 execution path today.

## Pre-registered stop and retirement rules

- If Exp6730 reproduces `.586`'s context block after the subprocess sets 32K and verifies CUDA
  offload, retire the local object-table rerun scope pending a changed runtime resource.
- If Exp6731 loses beyond the prior within-arm noise floor, keep the table inline and retire
  fetch-on-demand for this prompt/tool design.
- If Exp6735 misses held-out AUROC 0.65 or leaks any exact current-row signal, Exp6736 is gate-blocked
  and the diagnostic energy is not used for generation.
- If Exp6739 repeats the order-level null, retire this transactional external-memory recipe instead of
  increasing seeds without a technique change.
- If Exp6740 repeats the Torx/parity block, retire this compiler-fidelity route until the dependency or
  factor representation materially changes.

## Explicitly deferred

- Weight-updating continual learning, LoRA, or RLVR. `.587` first needs a valid external-memory causal
  result with preserved support.
- EBT/Kona-scale latent reasoning training. No public Kona runner exists, and local EBT scale is not a
  one-milestone dependency.
- Physical TSU speed/power claims. Z1 access is not available locally.
- PSC integration into llama.cpp. Measure it only if the runtime implementation is actually reachable.
- New ARC game solves, offline ground-truth BFS, game-source inspection, and hand-built adapters.

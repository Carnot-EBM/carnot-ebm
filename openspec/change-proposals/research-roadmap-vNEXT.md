# Research Roadmap vNEXT

**Milestone:** `2026.07.514`
**Title:** Lossless Evidence Gates, Solver-Certified Constraint Acquisition, Final Rust Crossover, and ARC Generalization
**Status:** Proposed
**Date:** 2026-07-20
**Supersedes:** milestone `2026.07.513` planning document
**Task range:** `exp5755`-`exp5768` (14 experiments, collision-checked)
**Conductor file:** `research-roadmap-next.yaml`

## Milestone thesis

Milestone `.513` answered two scientific questions and exposed two orchestration defects. The
render-matched KAN residual was genuinely negative (`-0.084269`), so KAN-specific scale-up is not
the next continuous-learning move. The game-blind ARC primitive A/B was genuinely null, so repeating
the same primitive is not justified. In contrast, the exact proposal benchmark and the repaired Rust
restart path were both scientifically ready, but their dependent experiments never ran because the
producer artifacts stored gate evidence in shapes the conductor cannot compare as scalars.

Milestone `.514` separates evidence transport from scientific evaluation. It first derives and tests
lossless bare-scalar bridge artifacts from the already-sealed Exp5746 and Exp5751 evidence. Only then
does it run the SOTA proposal-utility panel and the optimized Rust crossover. It also replaces KAN
reweighting/growth with a materially different FR-11 mechanism: solver-certified constraint
acquisition that can add, refine, quarantine, supersede, and roll back typed constraints from an
exact chronological stream. Finally, it satisfies the ARC generalization floor with leave-one-game-out
component-interaction attribution and, only on positive held-out evidence, one game-blind live-path
composition hardening.

The milestone does not reopen free-form answer-envelope repair, native JSON grammar, PHASE-D
generated-text/logprob scoring, LLM judges as truth, GGUF weight writes, broad RL, KAN scale-up,
two-axis replica exchange, per-game ARC adapters, source-derived games, offline ground-truth solves,
or unauthenticated FPGA/TSU/Kona speed claims.

## What milestone `.513` proved

| Evidence | Terminal result | Consequence for `.514` |
|---|---|---|
| Exp5743 transition | Terminal `.512` evidence and the `exp5743`-`exp5754` allocation were preserved. | Archive `.513` exactly once; do not rewrite duplicated older history. |
| Exp5744 source ingestion | No non-duplicate execution-time source delta changed the graph. | Keep one bounded post-planner ingestion slot; zero accepted findings is a valid terminal result. |
| Exp5745 ARC gate corrigendum | Frozen Exp5740 evidence normalized cleanly: coverage `1.0`, admitted leaks `0`, seven positive deletion primitives. | The prior gate was fixed; the later live null is scientific, not another schema excuse. |
| Exp5746 exact proposal benchmark | A disjoint 180-row hard/soft benchmark, complete candidate pools, exact optima, structure receipts, and adversarial controls are ready. `benchmark_ready_score=1.0`. | Reuse the sealed manifest. Do not rebuild or contaminate its science split. |
| Exp5747 proposal panel | Pre-gate blocked because two required fields were absent at top level even though the benchmark itself was ready. | Emit a tested scalar bridge, then run the panel as an authorized continuation. |
| Exp5748 selective search | Preemptively skipped after Exp5747 did not run. | Run only behind a positive, scalar one-shot utility gate. |
| Exp5749 CSL mechanism audit | Safe generic FR-11 evidence survived, but `kan_mechanism_residual=-0.084269`; prefix retention passed and unsafe updates stayed zero. | Retire KAN-specific scale-up. Change the learned object from residual weights to the constraint set itself. |
| Exp5750 dependent-task KAN CSL | Correctly gate-skipped on the negative KAN residual. | Do not retry KAN. A dependent-task continuation is admissible only for a different, positive constraint-acquisition mechanism. |
| Exp5751 Rust restart repair | The original `n=96` divergence was reproduced and repaired; n=48/96/192 interruption tests passed. Parity/reachability receipts are structured objects. | Preserve the repair and expose their pass predicates as tested bare booleans before timing. |
| Exp5752 Rust crossover | Pre-gate blocked because the conductor compared whole parity/reachability objects with `true`. | Bridge the objects losslessly; then profile/optimize and run one final 10x-or-retire benchmark. |
| Exp5753 ARC live A/B | The live development-proxy A/B ran and produced `delta=0` with no registry credit. | Retire the single generic-primitive intervention. Test component interactions under held-out folds instead. |
| Exp5754 capstone | Proposal benchmark ready, proposal utility unknown, KAN residual negative, restart parity ready, Rust 10x not retired, ARC gate fixed, ARC live delta zero. | `.514` targets the remaining measurements without relabeling blocked work as a scientific null. |

## The three largest gaps to the PRD vision

### Gap 1: exact evidence exists, but producer-to-conductor reachability is unreliable

FR-12 requires verifiable reasoning and NFR-01 requires a measured production speedup. Exp5746 and
Exp5751 contain the necessary benchmark and parity evidence, yet dependent tasks were skipped because
gate fields were missing or object-valued. Repeating the expensive science without repairing the
transport contract would make the result depend on serialization accidents.

`.514` response: Exp5757 and Exp5758 build lossless bridge artifacts with top-level bare scalar gate
fields and producer-normalizer tests. Exp5759 then runs all three mandated SOTA GGUFs on the frozen
proposal benchmark. Exp5764 profiles and optimizes only after the Rust bridge passes; Exp5765 runs the
final matched-quality crossover.

### Gap 2: continuous self-learning is safe, but it does not yet learn new constraint structure

FR-11 calls for continuous improvement on an exact held-out objective. Carnot has safe sidecar
updates, immutable base models, prefix retention, and rollback. The KAN residual is nevertheless
negative, and changing weights or residual coefficients is not the same as discovering a missing
rule. The research program explicitly prioritizes Tier-1 constraint addition.

`.514` response: Exp5761 creates an MPMMine-shaped exact constraint-acquisition corpus from the sealed
local benchmark. Exp5762 implements query-driven typed constraint birth/refinement/quarantine using
exact solver membership queries and is the milestone's mandatory continuous-self-learning task.
Exp5763 scales only a positive mechanism to dependent chronological tasks with lifecycle repair,
non-forgetting certificates, and exact rollback.

### Gap 3: production scale and ARC live generalization remain below the PRD destination

The Rust one-axis backend has semantic and restart readiness but no 10x result. The ARC development
registry is 183/183 while the submitted live mechanism reproduces only a small fraction; another
public-level solve or a single-primitive retry would not measure generalization.

`.514` response: Exp5764 uses allocation/phase profiling to implement a narrow production-reachable
hot-path optimization; Exp5765 claims 10x only under two consecutive larger-size lower bounds and
retires the technique on the same null. Exp5766 measures leave-one-game-out causal contributions and
interactions among already-reachable live components. Exp5767 hardens at most one positive
game-blind composition rule and runs a matched full-registry development-proxy A/B; no solve or
registry credit is available.

## Research incorporated before design

The complete search and dispositions are recorded in the `V514 Planner Refresh` block of
`research-references.md`.

| 2025-2026 source | Actionable idea | `.514` use |
|---|---|---|
| Query-Driven Interactive Constraint Acquisition, arXiv:2509.24489 | Use discriminating membership queries to remove over-fit constraints and recover valid substructure. | Exp5762 adds/refines/quarantines typed constraints using exact solver queries instead of KAN weight changes. |
| MPMMine / Constraint acquisition needs better benchmarks, arXiv:2605.26279 | CA benchmarks need multiple models, domain artifacts, and both solutions and non-solutions. | Exp5761 seals faithful, incomplete, and over-fit model variants plus positive/negative rows and exact receipts. |
| LLM-as-a-Verifier, arXiv:2607.05391 | Verification quality depends on granularity, repetition, criteria, and candidate-ranking cost. | Boundary only: Exp5759 reports model/family stratification and matched ranking budgets, but does not reopen PHASE-D text scoring or treat an LLM as authority. |
| The Verification Horizon, arXiv:2606.26300 | Fixed reward signals can become unfaithful or gameable as generators change. | Exp5759 reports per-model residuals and adversarial controls; any family-specific shortcut blocks promotion. |
| Selective Verification, arXiv:2606.19808 | Compare selective intervention with always-on and matched-budget longer generation. | Exp5760 runs only after positive one-shot utility and measures harmful flips, exact calls, model tokens, and wall time. |
| Opt-Verifier, arXiv:2605.29556 | Separate model/formulation structure from solution validity. | Exp5757 preserves both receipt axes; Exp5759/Exp5760 require exact structure and solution receipts on every credited row. |
| Fixed-Point Reasoners, arXiv:2606.18206 | Adaptive compute should stop on measured convergence rather than a fixed depth. | Watch-only for future latent reasoning; `.514` has no locally trained FPRM and makes no imported ARC claim. |

The Semantic Scholar query exposed 28 EBT and 8 ARM-EBM citation records at planning time, but no
new citation supplied a stronger open local implementation. Extropic's public writing still ends at
the 2025 X0/XTR-0/TSU launch, and Kona remains proprietary without local weights or authenticated
receipts. GitHub discovery found no replacement for Carnot's exact validators, live ARC path, or
one-axis sampler. Therefore `.514` requires the existing dual RTX 3090 host only and contains no
board integration task.

## Target architecture after `.514`

```text
                     LOSSLESS EVIDENCE / PROPOSAL PLANE

 Exp5746 sealed benchmark ─► Exp5757 scalar bridge ─► Exp5759 SOTA utility panel
   structure + solution          bare fields                    │ utility LCB > 0
   exact receipts                hash-preserving                ▼
                                                       Exp5760 selective exact feedback
                                                       exact validators remain authority

                     CONTINUOUS SELF-LEARNING PLANE

 Exp5746 exact rows ─► Exp5761 CA corpus ─► Exp5762 query-driven constraint lifecycle
 faithful/incomplete/      sealed folds          add / refine / quarantine / rollback
 over-fit model variants                              │ recovery gain > 0 + retention
                                                     ▼
                                         Exp5763 dependent-task longitudinal CSL
                                         immutable GGUF + exact non-forgetting

                     PRODUCTION / GENERALIZATION PLANE

 Exp5751 repaired parity ─► Exp5758 scalar bridge ─► Exp5764 profiled hot path
                                                        │ parity + readiness
                                                        ▼
                                                Exp5765 10x-or-retire

 ARC live components ─► Exp5766 leave-one-game-out interaction audit
                               │ positive held-out LCB, no leaks
                               ▼
                       Exp5767 game-blind composition hardening
                       development_proxy; no public solve credit

 Exact solvers, state hashes, Rust parity receipts, and live observation/action traces are
 authoritative. Learned/model scores only propose, order, or allocate work.
```

## Phase 1 - Terminal transition and lossless gate contracts

### Exp5755 - Transition terminal `.513` evidence

Archive Exp5743-Exp5754 outcomes exactly once, including blocked-versus-null distinctions. Preserve
the negative KAN residual, repaired Rust parity, proposal benchmark readiness, and zero ARC live
delta. Collision-scan Exp5755-Exp5768.

**Deliverable:** `results/experiment_5755_transition_v514.json`

### Exp5756 - Post-V514 source-delta ingestion

Search only after the V514 planner marker. Classify each source as accepted, duplicate, watch-only,
or excluded, record real bibliographic wall time, and require operator review before changing the
active graph.

**Deliverable:** `results/experiment_5756_v514_source_delta_ingestion.json`

### Exp5757 - Proposal benchmark scalar bridge

Read Exp5746 without regenerating a row. Verify its manifest, row hashes, exact receipts, and
adversarial controls, then emit top-level bare scalars for benchmark readiness, structure failures,
validator disagreements, and bridge readiness. Test the exact conductor predicates.

**Deliverable:** `results/experiment_5757_proposal_benchmark_scalar_bridge.json`

### Exp5758 - Rust parity scalar bridge

Read Exp5751 without changing sampler code. Verify the repaired trace hashes and derive bare pass
booleans from `distributional_parity.passed`, `fallback_equivalence.exact_fallback_equivalence`, and
`production_backend_reachable.passed`, plus the existing restart readiness score. Test the exact
downstream predicates.

**Deliverable:** `results/experiment_5758_rust_parity_scalar_bridge.json`

## Phase 2 - Decision-useful SOTA proposals

### Exp5759 - SOTA exact proposal-utility panel

**Gate:** Exp5757 bridge ready, zero structure failures, and zero validator disagreements.

Run all three mandated local GGUFs on the frozen Exp5746 science split through the already-qualified
finite-choice proposal channel. Compare model ordering with random, deterministic energy, and
solver-native controls under matched candidate and exact-validator budgets. Promotion requires a
positive paired lower confidence bound, no hard-feasibility regression, and non-regression in both
flagship families. This is candidate proposal/order evidence, not PHASE-D text scoring.

**Deliverable:** `results/experiment_5759_sota_exact_proposal_utility_panel.json`

### Exp5760 - Selective exact-feedback search

**Gate:** Exp5759 utility lower bound positive, both flagships non-regressing, and zero validator or
authority violations.

Use exact conflict/minimal-correction feedback to eliminate validator-proven dead candidates and
allocate a second finite-choice proposal only to preregistered recoverable states. Compare selective,
always-feedback, one-shot, matched-token longer proposal, random, energy, and solver-native controls.
Exact validators alone admit candidates.

**Deliverable:** `results/experiment_5760_selective_exact_feedback_search.json`

## Phase 3 - Solver-certified continuous self-learning

### Exp5761 - Exact constraint-acquisition benchmark

Derive a disjoint CA corpus from the sealed finite-domain families. For every faithful model, create
hash-sealed incomplete and over-fit variants, positive/negative assignments, minimal distinguishing
queries, and independent exact validator receipts. No LLM or learner sees the science fold during
construction.

**Deliverable:** `results/experiment_5761_exact_constraint_acquisition_benchmark.json`

### Exp5762 - Query-driven constraint lifecycle

**Gate:** Exp5761 benchmark ready, disjoint, and validator-clean.

This is the mandatory continuous-self-learning experiment. Start from incomplete or over-fit typed
constraint sets, propose constraint birth/refinement/quarantine operations from observed exact
failures, and spend a bounded solver membership-query budget on discriminating assignments. Compare
query-driven refinement with passive-only, frozen, safe generic residual, and random-query controls.
Require positive held-out constraint-recovery gain, exact protected-prefix retention, zero unsafe
updates, and exact rollback.

**Deliverable:** `results/experiment_5762_query_driven_constraint_lifecycle.json`

### Exp5763 - Dependent-task longitudinal constraint acquisition

**Gate:** Exp5762 recovery lower bound positive, prefix retention passing, zero unsafe updates, and
zero rollback mismatch.

Scale the qualified constraint-set learner to chronological tasks with dependencies, supersession,
concept drift, conflicts, delayed counterexamples, crash/restart, and recovery. Compare matched
baselines on forward transfer, dynamic regret, query efficiency, retained exactness, and lifecycle
cost. GGUF weights and production defaults remain immutable.

**Deliverable:** `results/experiment_5763_dependent_task_constraint_acquisition.json`

## Phase 4 - Production crossover, ARC generalization, and capstone

### Exp5764 - Profiled allocation-free Rust hot path

**Gate:** Exp5758 bridge ready and all parity/fallback/reachability booleans true.

Profile serialization, PyO3 crossing, batch allocation, worker scheduling, kernel update, validation,
and restart phases. Optimize only the measured dominant production-reachable phase using contiguous
buffers and a fixed worker pool. Prove semantic, distributional, fallback, restart, and production
reachability parity. Make no 10x claim.

**Deliverable:** `results/experiment_5764_one_axis_profiled_allocation_free_hot_path.json`

### Exp5765 - Final Rust/Python 10x-or-retire crossover

**Gate:** Exp5764 optimized path ready with semantic/distributional parity and production reachability.

Benchmark matched Python and Rust release paths at n=48/96/192 and one larger feasible size with at
least 30 paired batches per cell. A 10x claim requires a paired lower confidence bound at least 10.0
at two consecutive larger sizes with matched quality. Otherwise retire this 10x technique and retain
the measured speedup without claim inflation.

**Deliverable:** `results/experiment_5765_one_axis_final_10x_crossover.json`

### Exp5766 - ARC leave-one-game-out component-interaction audit

Use agent-owned traces and the submitted live path to measure marginal and pairwise interaction
effects of existing generic components under leave-one-game-out folds across all 25 public games.
Hold action budgets and replay seeds fixed; include negative source/game-identity leak canaries and
a positive causal-control trace. This is `development_proxy` generalization measurement, not a solve.

**Deliverable:** `results/experiment_5766_arc_loo_component_interaction_audit.json`

### Exp5767 - Game-blind ARC composition hardening

**Gate:** Exp5766 has a positive held-out generalization lower bound, at least one causal interaction,
and zero source/game-identity leaks.

Harden at most one runtime-observable, game-blind composition rule in the reachable shared live path,
then run a paired full-registry A/B under identical budgets. No game ID, source, adapter, banked plan,
offline BFS, or public-level credit is permitted. The same zero/negative live delta retires this
composition intervention.

**Deliverable:** `results/experiment_5767_arc_game_blind_composition_hardening.json`

### Exp5768 - Milestone capstone reconciliation

Reconcile all 14 tasks, gates, skips, terminal verdicts, source changes, capability specs, traceability,
status, and changelog. Record scientific nulls separately from conductor/schema skips. State whether
proposal utility, selective feedback, constraint acquisition, Rust 10x, and ARC live generalization
actually promoted. No publication or external deployment occurs.

**Deliverable:** `results/experiment_5768_v514_capstone_reconciliation.json`

## Dependency graph

```text
exp5755 transition ─────────────────────────────────────────────────────────────┐
exp5756 source ingestion ──────────────────────────────────────────────────────┤
                                                                               │
exp5757 proposal scalar bridge ─► exp5759 SOTA utility ─► exp5760 selective ──┤
                                                                               │
exp5761 CA benchmark ─► exp5762 constraint lifecycle ─► exp5763 dependent CSL ┤
                                                                               │
exp5758 Rust scalar bridge ─► exp5764 hot path ─► exp5765 10x-or-retire ──────┤
                                                                               │
exp5766 ARC LOO audit ─► exp5767 composition hardening ────────────────────────┤
                                                                               ▼
                                                                     exp5768 capstone
```

All gates are conjunctive. A failed scientific gate writes a blocked artifact and skips downstream
agent cost. The capstone still runs and records every skip. The bridge tasks may normalize only
unambiguous existing evidence; they may not invent missing methodology or modify upstream artifacts.

## Task inventory and execution order

| Order | ID | Track | Agent/model | Expected wall time | Gate |
|---:|---|---|---|---:|---|
| 1 | exp5755 | transition | Codex / gpt-5.5 | 45 min | none |
| 2 | exp5756 | research | Claude / Sonnet | 90 min | none |
| 3 | exp5757 | infrastructure | Claude / Opus | 120 min | none |
| 4 | exp5758 | infrastructure | Claude / Opus | 120 min | none |
| 5 | exp5759 | verification | Claude / Sonnet | 420 min | exp5757 |
| 6 | exp5760 | verification | Claude / Sonnet | 360 min | exp5759 |
| 7 | exp5761 | continuous-learning | Codex / gpt-5.5 | 180 min | none |
| 8 | exp5762 | continuous-learning | Codex / gpt-5.5 | 300 min | exp5761 |
| 9 | exp5763 | continuous-learning | Codex / gpt-5.5 | 360 min | exp5762 |
| 10 | exp5764 | performance | Codex / gpt-5.5 | 300 min | exp5758 |
| 11 | exp5765 | performance | Codex / gpt-5.5 | 420 min | exp5764 |
| 12 | exp5766 | arc-generalization | Claude / Sonnet | 300 min | none |
| 13 | exp5767 | arc-generalization | Codex / gpt-5.5 | 360 min | exp5766 |
| 14 | exp5768 | reconciliation | Claude / Opus | 180 min | none |

## Hardware and model requirements

### Required local hardware

- Dual RTX 3090 GPUs for Exp5759 and Exp5760. Use both cards when the runner supports it, authenticate
  CUDA/offload receipts, and release models on all paths.
- CPU/RAM/disk for exact solvers, constraint-acquisition streams, Rust/PyO3 builds, and ARC replay.
- Fixed CPU affinity and thread counts for Exp5764/Exp5765. No CPU result may be labeled hardware
  acceleration.
- No FPGA, KV260, PolarFire, GateMate, TSU, photonic, or Kona hardware is required or claimed.

### Mandated GGUF policy

Every experiment that invokes an LLM must declare at least one of:

- `unsloth/Qwen3.6-35B-A3B-GGUF` - flagship MoE;
- `unsloth/gemma-4-31B-it-GGUF` - flagship dense;
- `unsloth/gemma-4-26B-A4B-it-GGUF` - middle MoE.

Exp5759 uses all three. Exp5760 uses the flagship Qwen and dense Gemma at minimum. Resolution must
start with `cached_sota_pair()` or the repository's equivalent cached SOTA resolver and record exact
hub IDs, paths, hashes, quantization, llama.cpp build, GPU assignment, and offload receipts. Legacy
Qwen3.5-0.8B or Gemma E4B models are smoke tests only and cannot supply headline numbers.

## Promotion, retirement, and claim rules

- **Proposal branch:** promote only if `proposal_utility_lcb > 0`, both flagship families do not
  regress, and exact authority violations/disagreements are zero. Otherwise preserve the parse-safe
  channel but retire the decision-utility claim.
- **Selective branch:** promote only if selective feedback beats one-shot and is non-inferior to
  always-feedback and matched-token longer proposal while reducing exact calls or model tokens and
  not increasing harmful flips.
- **CSL branch:** KAN-specific scaling is closed. Promote query-driven CA only on positive held-out
  recovery, exact prefix retention, zero unsafe propagation, and exact rollback. Constraint additions
  remain sidecar state until separately production-qualified.
- **Rust branch:** claim 10x only under the preregistered consecutive-size paired lower-bound rule.
  The same null as Exp5739 permanently retires this allocation-free 10x technique.
- **ARC branch:** no level solve is proposed. `solve_provenance=development_proxy`, registry delta and
  solve credit must remain zero. A zero/negative composition delta retires the intervention.
- **Verifier authority:** exact solvers and environment transitions are oracles only for the scoped
  checks they actually decide. Learned/model scores never become truth.
- **External actions:** do not publish, deploy, push, edit public docs, or claim unavailable hardware.

## Expected milestone outputs

1. A durable bare-scalar producer contract demonstrated on both proposal and Rust artifacts.
2. The first valid decision-utility result on the sealed exact proposal benchmark, or an honest
   retirement of the proposal-utility claim.
3. A solver-certified Tier-1 continuous learner that changes constraint structure, with a gated
   dependent-task result or an honest negative.
4. A final matched-quality Rust crossover and a mechanically clear 10x promotion/retirement outcome.
5. A held-out ARC component-interaction matrix and at most one reachable game-blind composition A/B,
   with no duplicate solve or off-path credit.
6. Reconciled OpenSpec, BMAD traceability, status, changelog, and a terminal capstone artifact.

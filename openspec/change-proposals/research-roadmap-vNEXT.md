# Research Roadmap vNEXT

**Milestone:** `2026.07.513`
**Title:** Decision-Useful Exact Proposals, Mechanism-Distinct Self-Learning, Restart-Safe Rust Scaling, and ARC Gate Recovery
**Status:** Proposed
**Date:** 2026-07-20
**Task range:** `exp5743`-`exp5754` (12 experiments, collision-checked)
**Conductor file:** `research-roadmap-next.yaml`

## Milestone thesis

Milestone `.512` converted several speculative interfaces into real, replayable mechanisms. All
three mandated local GGUF families can select sealed one-token candidate labels; exact validators
attested a 96-row chronological proposal stream; zero-gated KAN updates preserved the old function
at insertion and survived lifecycle rollback; and the Rust `sample_batch` boundary is semantically
and distributionally ready. The ARC causal audit also found seven game-blind primitives with
positive deletion utility.

Those are mechanism-readiness results, not yet PRD-level value claims. The GGUF proposal stream is
conflict-heavy and near chance, so Carnot does not know whether it reduces exact-search work. The
KAN sidecar was safe, but its parameter-matched MLP control had lower suffix error, so there is no
KAN-specific learning claim. The Rust benchmark reached only about 2x and exposed restart mismatch
at the larger sizes. Finally, the ARC live A/B never ran because the upstream artifact represented
detected-and-rejected leak canaries as positive leak counts and encoded receipt coverage as an
object rather than the required scalar.

Milestone `.513` therefore measures marginal value and repairs blockers before adding scope:

1. test whether exact-attested SOTA proposals improve feasible discovery or exact-search effort
   over random, deterministic heuristic, and solver-only controls;
2. admit adaptive exact-verifier feedback only if one-shot proposal utility is positive;
3. test whether zero-gated KAN learning has a render- and parameter-matched mechanism residual
   before scaling continuous self-learning to dependent tasks;
4. repair restart parity before one final allocation-free Rust 10x-or-retire benchmark; and
5. normalize the ARC causal-gate schema, then run the already-designed generic live A/B without
   solving public games off-path.

The milestone does not reopen free-form answer envelopes, native JSON grammar, external
generated-text/logprob scoring, LLM judges, token scores as semantic authority, GGUF weight
writes, broad RL, two-axis exchange, per-game ARC adapters, source-derived games, offline
ground-truth solves, or unauthenticated TSU/Kona/FPGA speed claims.

## What milestone `.512` proved

| Evidence | Terminal result | Consequence for `.513` |
|---|---|---|
| Exp5731 transition | `.511` nulls and retirements were preserved and a collision-free `.512` graph was allocated. | Archive `.512` without rewriting the duplicated historical blocks already present in `research-complete.yaml`. |
| Exp5732 source ingestion | Three actionable deltas were accepted, but adversarial review flagged the bibliographic run as too short for its declared duration. | Keep one bounded source slot, explicitly classify its substrate as web/bibliographic, and measure only real wall-clock search time. |
| Exp5733 finite-choice channel | All three mandated GGUFs qualified on 42 controls with zero receipt failures, label collisions, or validator disagreements. Accuracy ranged from 0.476 to 0.571. | The interface is parse-safe. Do not repeat qualification; ask whether proposals help search. |
| Exp5734 sealed proposal stream | 96 complete flagship rows were sealed with exact labels; 53 rows had proposal conflicts and no validator disagreement. | Exact authority works, but proposal utility is unproven and cannot be inferred from parseability. |
| Exp5735 zero-gated CSL | Function-preserving insertion, zero unsafe updates, 0.25 ms max update latency, and 0.607 suffix improvement passed. The KAN suffix error was 0.146 versus 0.062 for the parameter-matched MLP. | FR-11 has a safe mechanism, but no KAN-specific advantage. Require mechanism-residual evidence before scale-up. |
| Exp5736 lifecycle canary | 74 lifecycle operations, exact rollback hashes, and zero rejected-update propagation passed. | Reuse the lifecycle machinery; do not repeat safety qualification. |
| Exp5737 SOTA ingress | A 48/48 shadow stream completed with positive suffix improvement, but lifecycle presentation and control semantics remain confounded. | Use render-matched evaluation and exact labels; do not treat the current result as mechanism-specific. |
| Exp5738 Rust batch backend | `sample_batch` achieved semantic, checkpoint, fallback, and distributional readiness. | Keep the API and focus only on the restart mismatch that invalidated larger-size benchmark cells. |
| Exp5739 Rust/Python benchmark | 728 quality-matched pairs produced a terminal null. Only `n=48` qualified; `n=96`/`n=192` had 352 `restart_match` exclusions and the qualified speed bound was about 2x, not 10x. | Diagnose and repair restart parity before one materially changed allocation-free benchmark; retire if the same null repeats. |
| Exp5740 ARC causal audit | Seven generic primitives had positive deletion utility with complete large replay counts. Negative leak canaries were detected and rejected, but their counts were stored in fields interpreted as admitted leaks; coverage was an object. | Correct the gate schema without changing the science, then reuse the frozen causal artifact. |
| Exp5741 ARC live A/B | Conductor skipped the task because three structured gates failed on those schema representations. No live trial or solve occurred. | This is an authorized rerun only after the upstream schema is normalized; the same blocked verdict retires the lane. |
| Exp5742 capstone | Proposal channel, SOTA stream, CSL, and batch backend were ready; strict Rust 10x and ARC registry delta were false. | `.513` targets decision value, KAN mechanism specificity, Rust NFR-01, and live ARC induction without claim inflation. |

## The three largest gaps to the PRD vision

### Gap 1: exact proposal transport exists, but decision utility is unknown

FR-12 calls for verifiable reasoning, not merely a parseable candidate ID. `.512` proved that
GGUF scores can select among sealed candidates while exact validators remain authority, but the
stream contained 53 conflicts and the three control accuracies were near chance. Carnot has not
measured top-k feasible discovery, nodes-to-first-valid, optimality gap, or verifier calls saved
against random ordering, a deterministic energy heuristic, and exact search alone.

`.513` response: Exp5746 creates a disjoint exact hard/soft benchmark with separate structural and
solution-validity receipts. Exp5747 evaluates all three mandated SOTA families under matched
candidate and validator budgets. Exp5748 adds conflict-localized, selectively allocated exact
feedback only if the one-shot panel shows positive proposal utility.

### Gap 2: continuous self-learning is safe, but its claimed mechanism is not distinct

FR-11 now has chronological updates, function-preserving insertion, lifecycle rejection, exact
rollback, and immutable base models. That is meaningful continuous-learning infrastructure.
However, the parameter-matched MLP residual outperformed the zero-gated spline residual on the
reported suffix error, and the SOTA ingress did not fully remove memory-render effects. Scaling
the KAN would therefore confuse safe learning with KAN-specific value.

`.513` response: Exp5749 replays frozen `.512` receipts under render-, update-count-, parameter-,
and chronology-matched controls and defines one signed KAN mechanism residual. Exp5750 is a gated
dependent-task CSL scale-up with exact retention certificates; it runs only if that residual is
positive. Exp5749 itself is the milestone's mandatory continuous self-learning experiment and can
honestly close the KAN branch without cascade-skipping FR-11.

### Gap 3: the production path still misses both scale and live generalization gates

NFR-01 requires a measured Rust throughput advantage, but the batch API currently reaches only a
small-size result and loses larger cells to restart mismatch. Separately, the submitted ARC path
still trails the 183-level development oracle by 179 levels; `.512` found a possible generic
induction primitive but a schema error prevented the live experiment from running. Both are
reachability gaps: the mechanisms exist, but the credited production/live paths cannot yet use
them at the required quality.

`.513` response: Exp5751 repairs restart determinism without making a timing claim, and Exp5752
runs one allocation-free 10x-or-retire benchmark behind exact parity. Exp5745 normalizes the ARC
gate without recomputing causal science, then Exp5753 runs a game-blind full-registry live A/B.
All 25 public games are registry-complete, so Exp5753 improves live reproduction and induction
metrics; it does not claim a new public-game solve.

## Research incorporated before design

The full sweep and source dispositions are recorded in the `V513 Planner Refresh` block of
`research-references.md`.

| 2025-2026 source | Actionable idea | `.513` use |
|---|---|---|
| Opt-Verifier, arXiv:2605.29556 / ICML 2026 | Validate formulation structure and candidate solution separately; solver execution alone is insufficient. | Exp5746 seals a structural-completeness manifest and independent exact solution receipts; Exp5747/Exp5748 report both axes. |
| Selective Verification for Budget-Aware Reasoning, arXiv:2606.19808 | Verification can repair some attempts but may lose to a longer initial solve; compare allocation policies on matched cost. | Exp5748 compares selective feedback with always-feedback and matched-budget one-shot proposals. Exact validators remain authority. |
| Hard Rules, Soft Preferences, arXiv:2607.15562 | Separate hard feasibility from soft preference optimization. | Exp5746 adds exact CSP/MaxSAT/finite-state families with hard-valid and soft-objective receipts. |
| Presentation, Not Mechanism, arXiv:2607.16019 | Memory-format changes can masquerade as lifecycle-mechanism gains. | Exp5749 requires render-matched lifecycle controls before any KAN claim or scale-up. |
| ARC executable-world-model ablations, arXiv:2607.15439 | Simplification/world models count only when exact replay and ablation show causal utility. | Exp5745 preserves the frozen deletion evidence; Exp5753 tests at most one generic primitive on the reachable live path. |
| Adaptive Generate-Rank-Verify, arXiv:2605.17609 | Allocate costly exact checks using cheap proposal signals in a first-positive search. | Exp5748 uses only preregistered conflict/uncertainty signals to allocate verifier calls and benchmarks the full cost frontier. |
| CerCE, OpenReview ICLR 2026 | Constrain online updates with explicit non-forgetting certificates. | Exp5749/Exp5750 keep exact chronological retention and rollback certificates; no foundation-model training is introduced. |
| ARM-EBM, arXiv:2512.15605 v4 / ICML 2026 | Autoregressive lookahead can be represented through an energy/Bellman view. | Architecture context for proposal ordering only; token scores never become truth. |

The Extropic writing index still exposes X0/XTR-0/TSU and THRML but no authenticated local
hardware. Logical Intelligence still exposes Kona only as proprietary architecture/benchmark
context. GitHub trending and EBT/ARM-EBM citation trails supplied no stronger reproducible local
dependency. No TSU, Kona, photonic, or FPGA execution task is justified.

## Target architecture after `.513`

```text
                        exact-attested proposal plane

 mandated local GGUFs        sealed hard/soft domain        exact authority
 Qwen MoE / Gemma pair ───► candidate ordering ───────────► structure receipt
          │                         │                       + solution receipt
          │                         ▼                              │
          │               Exp5747 utility panel                   │
          │               random / heuristic / solver             │
          │                         │ utility > 0                  │
          └─────────────────────────┴────► Exp5748 selective exact feedback
                                          conflict localization only

                        continuous-learning plane

 .512 exact streams + lifecycle receipts ─► Exp5749 render/parameter audit
                                                   │ KAN residual > 0
                                                   ▼
                                           Exp5750 dependent-task CSL
                                           exact retention + rollback

                        production/live plane

 Rust sample_batch ─► Exp5751 restart parity ─► Exp5752 allocation-free 10x/retire

 ARC causal artifact ─► Exp5745 scalar gate corrigendum ─► Exp5753 live registry A/B
                                                               development proxy only

 Every learned score is advisory. Exact validators, state hashes, restart receipts, and
 live observation/action traces remain authority at their respective boundaries.
```

## Phase 1 - Evidence transition and blocker normalization

### Exp5743 - Transition terminal `.512` evidence

Archive every Exp5731-Exp5742 artifact and conductor outcome, including the Exp5732 duration flag,
the successful proposal/CSL/batch mechanisms, the 10x null, and the ARC gate skip. Collision-scan
Exp5743-Exp5754. Reconcile `.512` once without rewriting the pre-existing duplicated historical
blocks in `research-complete.yaml`.

**Deliverable:** `results/experiment_5743_transition_v513.json`

### Exp5744 - Post-V513 source-delta ingestion

Search only after the V513 planner marker. This is web/bibliographic research, not benchmark
compute. Record actual query-to-artifact wall time, accept zero findings as complete, and require
operator review before any roadmap-scope change.

**Deliverable:** `results/experiment_5744_v513_source_delta_ingestion.json`

### Exp5745 - ARC causal-gate schema corrigendum

Verify the Exp5740 hashes and create a lossless normalized view. Preserve the rejected canary
counts as `detected_*_canary_count`, derive scalar `admitted_source_leak_count=0`,
`admitted_game_identity_leak_count=0`, and `counterfactual_receipt_coverage_score=1.0` from the
existing receipts. Do not rerun primitive mining, change the seven causal effects, modify live
policy code, or claim a solve.

**Deliverable:** `results/experiment_5745_arc_causal_gate_schema_corrigendum.json`

## Phase 2 - Decision-useful exact proposals

### Exp5746 - Disjoint dual-receipt exact decision benchmark

Build at least 180 held-out instances across finite-domain CSP, weighted MaxSAT, hard/soft packing,
and finite-state planning. Each row has a sealed complete candidate pool, structural-completeness
receipt, exact feasibility/optimality receipt, solver baseline, deterministic heuristic ordering,
and adversarial omission/shortcut controls. No `.512` control or science row may be reused.

**Deliverable:** `results/experiment_5746_exact_proposal_utility_benchmark.json`

### Exp5747 - SOTA proposal-utility panel

**Gate:** Exp5746 benchmark readiness, zero structural-receipt failures, and zero exact-validator
disagreements.

Use all three mandated GGUF families through the frozen finite-choice interface. Compare model
ordering with random permutations, deterministic energy heuristic, and solver-native branching
under matched candidate and exact-validator budgets. Promotion requires a positive paired lower
confidence bound for at least one decision-utility metric, no hard-feasibility regression, and
non-negative results in both flagship families. Proposal scores stay advisory.

**Deliverable:** `results/experiment_5747_sota_exact_proposal_utility_panel.json`

### Exp5748 - Selective exact-feedback active-set search

**Gate:** Exp5747 has positive overall proposal utility, both flagship families are non-regressing,
and exact authority receipts are clean.

Use exact conflict/MCS feedback to replace eliminated candidates and allocate verifier calls only
to preregistered recoverable states. Compare selective feedback with one-shot, always-feedback,
random, deterministic heuristic, and matched-total-budget longer proposal baselines. Run the
flagship Qwen and Gemma GGUFs locally; exact validators alone admit candidates.

**Deliverable:** `results/experiment_5748_selective_exact_feedback_search.json`

## Phase 3 - Mechanism-distinct self-learning and restart-safe Rust

### Exp5749 - Render-matched CSL mechanism-residual audit

This is the milestone's mandatory continuous self-learning experiment. Replay the sealed Exp5735-
Exp5737 rows through frozen zero-gated KAN, parameter-matched MLP, no-growth, always-open, and
render-matched deprecation-disabled controls. Match chronology, update count, parameter budget,
labels, and presentation. Recompute prefix retention, suffix error, dynamic regret, rollback, and
unsafe propagation. Define positive KAN residual so that greater than zero means KAN beats the best
matched non-KAN control. A non-positive result retires KAN-specific scale-up but preserves the safe
generic FR-11 mechanism.

**Deliverable:** `results/experiment_5749_csl_render_matched_mechanism_audit.json`

### Exp5750 - Dependent-task continuous self-learning scale-up

**Gate:** Exp5749 KAN mechanism residual is positive, prefix retention passes, and unsafe updates
remain zero.

Generate chronological dependent constraint tasks with exact labels, hidden compositions,
supersession, conflicts, and distribution shifts. Compare the qualified zero-gated sidecar with
the same matched controls on forward transfer, recovery time, dynamic regret, old-task retention,
memory/latency growth, and exact rollback. GGUF weights and production defaults remain immutable.

**Deliverable:** `results/experiment_5750_dependent_task_continuous_self_learning.json`

### Exp5751 - Rust restart-parity root cause and repair

Reproduce the `n=96`/`n=192` mismatches from Exp5739, localize the first divergent checkpoint or
RNG/state transition, and repair the production-reachable one-axis batch path. Require exact
Python/Rust scheduler, restart, checkpoint, proposal, energy, sample-count, and fallback parity
under interruption injection. This task makes no throughput or 10x claim.

**Deliverable:** `results/experiment_5751_rust_restart_parity_repair.json`

### Exp5752 - Allocation-free Rust/Python 10x-or-retire benchmark

**Gate:** Exp5751 restart parity, semantic parity, distributional parity, and production-backend
reachability all pass.

Use a contiguous allocation-free batch representation and fixed worker pool, then benchmark
matched Python/Rust release paths at `n=48`, `n=96`, `n=192`, and one larger feasible size with at
least 30 batches per cell. Claim 10x only if quality matches and the paired lower confidence bound
is at least 10.0 at two consecutive larger sizes. The same terminal null permanently retires this
10x technique.

**Deliverable:** `results/experiment_5752_one_axis_allocation_free_10x_crossover.json`

## Phase 4 - ARC live-path recovery and capstone

### Exp5753 - Generic ARC primitive live-registry A/B

**Gate:** Exp5745 normalized coverage is 1.0, admitted source/game-identity leaks are zero, and at
least one frozen positive causal primitive remains.

Registry-precheck all 25 already-complete public games. Add at most one game-blind primitive to the
submitted live E3 induction path and run paired full-registry A/B under identical 400-action
budgets. Measure live levels reproduced, action-effect prediction, valid-action rate, repeated
actions, and budget use. This is a `development_proxy` generalization experiment: no public-game
level is new, no registry solve is credited, and source/adapters/offline solvers are forbidden.

**Deliverable:** `results/experiment_5753_arc_generic_primitive_live_registry_ab.json`

### Exp5754 - `.513` capstone reconciliation

Aggregate every Exp5743-Exp5753 artifact, gate skip, retirement signal, and missing state. Reconcile
OpenSpec, traceability, status, changelog, conductor log, exclusions, known issues, verifier gaps,
north-star, and applicable E2E receipts without changing scientific verdicts. Preserve independent
proposal, CSL, Rust, and ARC branch outcomes.

**Deliverable:** `results/experiment_5754_v513_capstone_reconciliation.json`

## Dependency graph

```text
Phase 1
Exp5743 transition ───────────────────────────────────────────────────────────┐
Exp5744 source delta ─────────────────────────────────────────────────────────┤
Exp5745 ARC gate corrigendum ───────────────────────────────► Exp5753 ARC A/B ┤
                                                                              │
Phase 2                                                                       │
Exp5746 exact benchmark ─► Exp5747 SOTA utility ─► Exp5748 selective feedback ┤
                                                                              │
Phase 3                                                                       │
Exp5749 CSL mechanism audit ───────────────────────► Exp5750 CSL scale-up ────┤
Exp5751 Rust restart repair ───────────────────────► Exp5752 10x/retire ──────┤
                                                                              ▼
                                                                     Exp5754 capstone
```

No `requires:` chain points to a retired experiment. Exp5749 is deliberately independent of the
SOTA proposal branch so the mandatory continuous self-learning evidence cannot cascade-skip. Every
natural-language gate is mirrored by a structured `gated_on` entry.

## Hardware and model requirements

| Resource | Tasks | Requirement and boundary |
|---|---|---|
| RTX 3090 GPU 0/1 | Exp5747-Exp5748 | CUDA-enabled `llama-cpp-python`, positive offloaded-layer and before/during/after memory receipts, one loaded model per device unless VRAM receipts justify otherwise. CPU fallback is smoke-only. |
| Mandated local GGUFs | Exp5747 | `unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and `unsloth/gemma-4-26B-A4B-it-GGUF` in explicit `MODEL_SPECS`. |
| Flagship SOTA pair | Exp5748 | Qwen3.6-35B-A3B plus Gemma-4-31B, immutable cached `.gguf` files, native llama.cpp tokenization/score access, no transformers. |
| CPU/RAM | Exp5743-Exp5746, Exp5749-Exp5754 | Exact solvers, CSL replay/training, Rust/Python sampling, ARC live policy, and reconciliation. Record peak memory and fixed core/thread allocations where relevant. |
| Exact solvers | Exp5746-Exp5748, Exp5749-Exp5750 | Existing deterministic FSM/arithmetic/SAT/MaxSAT/Z3/CP-SAT-compatible validators. Solver execution is not enough: structural manifests and solution receipts are both required. |
| Rust/PyO3 toolchain | Exp5751-Exp5752 | Existing `carnot-samplers` crate and production `SamplerBackend`, release build, deterministic checkpoint/RNG schemas, fixed worker pools, and reproducible end-to-end receipts. |
| NVMe | All phases | Immutable GGUF hashes, benchmark manifests, score vectors, exact receipts, CSL ledgers/checkpoints, sampler traces, ARC traces, and artifact hashes. |
| ARC live environment | Exp5753 | Submitted live E3 observation/action path and registry precheck only. All public games are already solved in the registry; source, adapters, exhaustive BFS, and off-path solvers are forbidden. |
| KV260 / PolarFire / GateMate | None | All three local FPGA lanes are terminal in `ops/status.md`; no board-per-milestone task or speed claim is required. |
| Extropic TSU / Kona / photonic Ising | None | Watch-only architecture context; no authenticated local execution path exists. |

## Promotion, retirement, and claim rules

1. **Benchmark gate:** Exp5746 must independently prove candidate-domain structural completeness
   and candidate solution feasibility/optimality with zero validator disagreement.
2. **Proposal-utility gate:** Exp5747 promotes only a paired, budget-matched positive result that
   survives random, deterministic heuristic, and solver-only controls in both flagship families.
   Parseability, CUDA offload, score margin, and proposal accuracy alone are not utility.
3. **Selective-feedback gate:** Exp5748 may run only after positive one-shot utility and must beat
   both always-feedback and matched-budget longer proposal baselines. Exact validators remain the
   only authority.
4. **Continuous self-learning gate:** Exp5749 must remove render and parameter confounds. Exp5750
   runs only for positive KAN mechanism residual, exact old-prefix retention, zero unsafe updates,
   and exact rollback. A non-positive residual preserves generic FR-11 success but closes KAN scale.
5. **Rust gate:** Exp5751 claims parity only, never speed. Exp5752 may claim 10x only at two
   consecutive larger sizes with matched quality and paired lower confidence bound at least 10.0;
   the same null triggers permanent retirement for this technique.
6. **ARC gate:** Exp5745 may normalize representation but cannot change causal effects. Exp5753 is
   credited only as a live-path development proxy and cannot bank already-solved public levels.
7. **Failed reruns:** every matching carry-forward has all four `prior_failures` fields and
   `retire_if_same_verdict: true`; new experiment IDs avoid retired upstream chains.
8. **No claim inflation:** missing, gate-skipped, malformed, development-proxy, or CPU-smoke
   artifacts never become headline SOTA/live/hardware evidence. A proposal is not a proof, safe
   learning is not KAN advantage, parity is not speedup, and reproduction is not a new ARC solve.

## Expected outputs

- one terminal `.512` transition and one bounded post-planner source-delta artifact;
- one lossless ARC gate-schema corrigendum;
- one disjoint dual-receipt exact benchmark, one all-SOTA proposal-utility panel, and one gated
  selective exact-feedback search;
- one mandatory render-matched CSL mechanism audit and one gated dependent-task scale-up;
- one restart-parity repair and one allocation-free 10x-or-terminal-retirement benchmark;
- one gated game-blind ARC live-registry A/B with explicit `development_proxy` provenance; and
- one capstone reconciling specifications, operations, exclusions, hardware boundaries, and every
  positive, null, blocked, skipped, or retired result without changing its verdict.

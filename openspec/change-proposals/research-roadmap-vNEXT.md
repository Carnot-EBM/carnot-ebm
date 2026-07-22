# Research Roadmap vNEXT — Milestone 2026.07.519

**Milestone:** 2026.07.519  
**Title:** Certified Adaptive Memory, Internal Energy, and World-Feedback Induction  
**Status:** Proposed  
**Task range:** Exp5823-Exp5836 (14 experiments)  
**Execution file:** `research-roadmap-next.yaml`  
**Date planned:** 2026-07-22

## Thesis

Milestone `.518` closed the finite-ID GGUF answer-transport question. The companion audit and
split-budget implementation were clean, and all three mandated models ran with authenticated CUDA
offload, but the changed canary still qualified zero models: only 2 of 144 rows carried an exact
label, 138 rows failed parsing, and 134 truncated. Exp5814 and Exp5815 consequently gate-blocked.
The result is a bounded negative for this transport lane, not a negative result for constraint
learning or local SOTA representations.

Milestone `.519` pivots to evidence paths that are already reachable. Carnot has a successful
four-family exact constraint-acquisition benchmark (Exp5761), a credited query-driven lifecycle
(Exp5762), a dependent-task exact acquisition result (Exp5763), and a sealed hardness/surface
fixture (Exp5785). The next question is whether the learner can acquire *new constraint structure*
outside its frozen template library, survive multiple chronological rule changes, promote only on
sealed future evidence, and transfer selectively without forgetting. GGUF weights remain frozen;
exact solver feedback is explicitly oracle evidence.

In parallel, `.519` uses the only reproducible local hidden-state surface that does not depend on
generated answers: final-token/final-layer embeddings from `llama_cpp.Llama(embedding=True)`. It
builds exact paired candidates for all three mandated SOTA families and tests an oracle-distinct
contrastive/KAN energy verifier with held-model-family-out evaluation. A clean null retires this
bounded GGUF-embedding verifier lane.

The ARC branch changes architecture after two-model CEGIS nulls. It gives the live E3 agent a
bounded, append-only, write-protected world-fact tape and a budget-matched world-feedback probing
policy. Facts may come only from the agent's own legal actions and exact runtime observations.
There is no source inspection, offline BFS, per-game adapter, iterative LLM reinduction, public-game
solve target, or registry credit. The accepted adaptive-memory operations are then expressed as a
bounded Python/Rust microkernel and mapped conservatively to each attached board.

The milestone makes four falsifiable claims:

1. Minimal-core and membership-query evidence can recover held-out constraint structures that the
   successful Exp5762 frozen-template learner cannot express.
2. Versioned write-protected constraint memory can improve sealed future batches across multiple
   changes while preserving old-prefix accuracy, recurrence recovery, bounded growth, and exact
   rollback.
3. Output-free internal embeddings contain model-family-portable energy signal on causal candidate
   pairs; otherwise the bounded final-layer GGUF verifier route is retired.
4. Exact world-feedback acquisition plus a structured fact tape improves held-out live E3
   transition/policy behavior under the same interaction budget; no level solve is required.

## What milestone 2026.07.518 proved

| Branch | Terminal evidence | Consequence for `.519` |
|---|---|---|
| Transition | Exp5809 archived `.517`, preserved Exp5799 as quarantined, tombstoned Exp5800-Exp5808, and reported zero collisions for Exp5809-Exp5822. | Exp5823 archives the seven activated `.518` identities, tombstones proposal-only Exp5816-Exp5822, and allocates Exp5823-Exp5836. |
| Source currency | Exp5810 found zero accepted post-V518 deltas. | Start from the new V519 planner marker; zero accepted execution-time findings remains complete. |
| Evidence integrity | Exp5811 independently replayed Exp5799 and made the split-budget contract safe to test. | Preserve the audit and historical artifacts; do not use them as a reason to rerun output transport. |
| Transport contract | Exp5812 implemented separate reasoning/finalization budgets, a sealed candidate environment, transcript freezing, and exact-validator authority. | The mechanism was valid, so the following failure is scientific rather than a missing contract. |
| Three-family SOTA canary | Exp5813 ran Qwen3.6-35B-A3B, Gemma 4 31B, and Gemma 4 26B-A4B with fresh CUDA receipts. It qualified 0/3 models; exact-label coverage was 2/144, with 138 parser failures and 134 truncations. | Retire the finite-ID generated-answer lane. New SOTA work must use a materially different surface; `.519` uses embeddings, not generated labels. |
| Downstream gate | Exp5814 gate-blocked on `answer_channel_ready_score=0.0`; Exp5815 pre-emptively skipped behind the retired upstream. | Do not re-propose the same generated stream. Resume FR-11 from Exp5761/5762/5763/5785 exact artifacts instead. |

## The three biggest gaps to the PRD vision

### Gap 1 — FR-11 has a credited toy lifecycle but no structural, future-validated endurance

Exp5762 recovered constraints from a matched frozen candidate library with exact membership
queries. Carnot has not shown acquisition of relations absent from that library, multiple rule
additions/supersessions/recurrences, future-batch admission, compatible replay, bounded memory, or
separate forward-transfer and retention guarantees. That is the milestone's primary gap.

### Gap 2 — FR-08/FR-09 lack a local oracle-distinct representation verifier

The frozen FoVer result remains defensible, but the generated-text/logprob PHASE D lane is retired.
Exp5200 showed that one final-layer Gemma embedding probe on MMLU-Pro did not beat tuned
self-consistency or zero-training controls. Carnot has not tested causal exact candidate pairs,
three current SOTA families, fixed dimension alignment, or held-model-family-out transfer. This is
a narrow open verifier question with an explicit retirement outcome.

### Gap 3 — ARC world induction is not reliably live-path reachable or portable

The CEGIS refinement arms on ThinkingCap/Qwen and Gemma 4 31B produced pooled negative deltas, and
the latest generic held-out evidence identifies induced dynamics quality as binding. The live agent
still lacks a deterministic structured memory of its own observations and a principled way to buy
new world evidence. Even accepted adaptive state has no small backend-neutral update/lookup ABI for
Rust and attached-board handoff. `.519` addresses the live path first and maps hardware second.

## 2025-2026 research update and experiment hooks

The dated search ledger is in `research-references.md` under
`V519-PLANNER-REFRESH-20260722-END`.

| Finding | Carnot implication | Experiment hook |
|---|---|---|
| [Learning from World Feedback](https://arxiv.org/abs/2607.16591) | Prediction uncertainty need not identify constraint-boundary risk; direct outcomes are the safer admission signal. | Exp5827-Exp5829 use exact boundary evidence, and Exp5833 selects probes by task structure and runtime outcomes rather than model confidence. |
| [Chain of Computation and Structured Context Windows](https://arxiv.org/abs/2607.17710) | A bounded structured tape plus deterministic support can make planning state explicit. | Exp5832 adds an append-only typed world-fact tape at the live E3 seam. |
| [Ask the World Before Acting](https://arxiv.org/abs/2606.31422) | Evidence acquisition is a budgeted action; criticality, staleness, and dependency can beat periodic probing. | Exp5833 compares a game-blind structural scheduler with periodic and random budget-matched controls. |
| [Write-Protected Discrete Bottlenecks](https://arxiv.org/abs/2607.08312) | Detach semantic memory from model gradients, use non-parametric bindings, and split collisions. | Exp5828 and Exp5832 use write-protected records, collision splitting, quarantine, versioning, and rollback with frozen GGUF weights. |
| [Rethinking Transfer in Continual Learning](https://arxiv.org/abs/2607.15587) | Forward transfer and retention are different; replay should select compatible history. | Exp5829 compares transfer-selective, all, and no replay under family recurrence. |
| [Equilibrium-Based Thermodynamic Computing Blueprint](https://arxiv.org/abs/2607.16183) | Continuous stochastic backends need explicit state, precision, update, and observability capabilities. | Exp5834-Exp5835 extend the bounded ABI/resource map only; there is no Extropic execution claim. |

The July SNN parallel-tempering CSP paper is recorded but not scheduled because that method family
is retired locally. EBT/ARM citation routes yielded no newer reproducible Carnot dependency.

## Target architecture

```text
 terminal .518 evidence
          |
          v
 archive 7 activated tasks + retire answer transport + allocate .519 (Exp5823)

 Exp5761 exact acquisition ----+
 Exp5762 credited lifecycle ---+--> canonical exact-event contract (Exp5825)
 Exp5763 dependent acquisition +              |
 Exp5785 hardness/surface ------+              v
                                  chronological out-of-template stream (Exp5826)
                                                |
                  +-----------------------------+----------------------+
                  |                                                    |
                  v                                                    v
  minimal-core/query structural learner (Exp5827)     exact causal candidate pairs
                  |                                    + three SOTA embeddings (Exp5830)
                  v                                                    |
 sealed-future promote/quarantine/rollback (Exp5828)                   v
                  |                                  held-model-family-out energy/KAN (Exp5831)
                  v                                    oracle-distinct; exact labels only
 compatible replay + recurrence/retention (Exp5829)
                  |
                  v
 bounded update/lookup/supersede/rollback microkernel (Exp5834)
                  |
                  v
 KV260 + PolarFire + GateMate capability/precondition receipts (Exp5835)

 agent-owned live E3 actions/outcomes
                  |
                  v
 write-protected bounded world-fact tape (Exp5832)
                  |
                  v
 structural world-feedback probing A/B (Exp5833)
 held-out games/actions; no solve target or registry credit

 Exp5823-Exp5835 evidence ---------------------------> Exp5836 capstone
                                                       always runs
```

The continual-learning, representation-verifier, and ARC branches are scientifically independent.
Structured `gated_on` fields skip only tasks whose required producer scalar failed. Hardware
continuity and the capstone always run so blocked/null outcomes remain visible.

## Phase 0 — Boundary closure and exact-event substrate (Exp5823-Exp5826)

### Exp5823 — Archive `.518`, retire the transport lane, and allocate `.519`

Resolve the seven activated `.518` tasks by exact `(milestone, task_id, declared_deliverable)`.
Classify Exp5813 as a clean negative and Exp5814/Exp5815 as gate-blocked, tombstone the unactivated
Exp5816-Exp5822 proposal identities, add the same-verdict answer-transport scope to the exclusion
manifest, and prove Exp5823-Exp5836 collision-free.

**Deliverable:** `results/experiment_5823_transition_v519.json`

### Exp5824 — Post-V519 source and implementation refresh

Search only work newer than the V519 planner marker. Accepted deltas may add bounded controls to
allocated tasks but may not reopen retired transport, CEGIS, tempering, public ARC solve, or
unchanged-board scopes. Zero accepted findings is complete.

**Deliverable:** `results/experiment_5824_v519_source_delta_ingestion.json`

### Exp5825 — Certified adaptive-memory event contract

Define one canonical, hash-stable event and state schema over Exp5761/5762/5763/5785: observations,
exact membership outcomes, constraint births, supersessions, recurrences, family/surface axes,
protected prefixes, sealed future batches, quarantine, promotion, rollback, and provenance. This is
an infrastructure adapter and fail-fast preflight, not a new benchmark or learning result.

**Deliverable:** `results/experiment_5825_certified_adaptive_memory_contract.json`

### Exp5826 — Chronological out-of-template structure stream

Generate a deterministic four-family exact stream with at least three preregistered changes per
family. Every science target is absent from the Exp5762 frozen template library and includes
addition, supersession, and recurrence cases crossed with hardness and proof-preserving surfaces.
Ground truth stays sealed from learners; exact solvers provide row labels and minimal-core receipts.

**Deliverables:**

- `results/experiment_5826_out_of_template_constraint_stream.json`
- `results/experiment_5826_out_of_template_constraint_stream.rows.jsonl`

## Phase 1 — Structural continuous self-learning (Exp5827-Exp5829)

### Exp5827 — Minimal-core structural acquisition A/B

Compare frozen state, the successful Exp5762 matched-template learner, passive core extraction,
random-query induction, and active discriminating queries plus minimal-core structure synthesis.
Use at least 30 independent units per primary family/change cell and paired bootstrap intervals.
Credit only exact structural recovery on headroom-present out-of-template rows.

**Deliverable:** `results/experiment_5827_minimal_core_structural_acquisition_ab.json`

### Exp5828 — Sealed-future promotion, quarantine, and rollback

Run the successful Exp5827 learner chronologically. Every proposed constraint is quarantined and
may be promoted only on a sealed future batch with positive paired lower bound, exact protected-
prefix retention, zero unsafe propagation, and hash-exact rollback. Test collision splitting,
stale-rule supersession, multiple changes, restart equivalence, and a fixed memory cap. This is the
milestone's required continuous-self-learning experiment.

**Deliverable:** `results/experiment_5828_future_validated_structural_memory.json`

### Exp5829 — Transfer-selective replay and recurrence audit

Compare no replay, all replay, and signature-compatible replay across held-out constraint families,
proof-preserving surfaces, and recurrence episodes. Predeclare headroom and report forward transfer,
retention, forgetting, recurrence recovery, dynamic regret, and replay cost separately.

**Deliverable:** `results/experiment_5829_transfer_selective_replay_audit.json`

## Phase 2 — Internal energy and live world feedback (Exp5830-Exp5833)

### Exp5830 — Three-family exact paired-embedding corpus

Build causal correct/incorrect candidate pairs from Exp5826 and extract output-free embeddings with
all three mandated local SOTA models. Use native embedded templates, fresh CUDA receipts, a fixed
label-blind projection into a common dimension, at least 30 pairs per constraint family, and
proof-preserving surface controls. No generated answer is consumed or scored.

**Deliverables:**

- `results/experiment_5830_sota_paired_embedding_corpus.json`
- `results/experiment_5830_sota_paired_embedding_corpus.npz`

### Exp5831 — Held-model-family-out contrastive/KAN energy verifier

Train cosine/logistic, MLP, and compact KAN energy controls on fixed projected pair differences.
Hold out each SOTA family in turn, mask model identity and answer-bearing fields, and report AUROC,
paired ranking accuracy, calibration, efficiency, confidence intervals, and within-family versus
cross-family gaps. The learned verifier is not an oracle; exact solvers remain labels and release
authority. If every headroom-present held-family lower bound includes zero, retire this route.

**Deliverable:** `results/experiment_5831_cross_family_embedding_energy_verifier.json`

### Exp5832 — Write-protected ARC world-fact tape

Add a bounded append-only structured context window to the reusable live E3 seam. Records are typed
agent-owned `(state digest, legal action, exact observed delta, dependencies)` events with
non-parametric symbol binding, collision splitting, eviction receipts, and replay hashes. Disable
game adapters, source access, offline BFS, per-game rules, and LLM reinduction in tests.

**Deliverable:** `results/experiment_5832_arc_write_protected_world_fact_tape.json`

### Exp5833 — Budget-matched world-feedback probing A/B

On a preregistered held-out game/action split including `sc25`, compare the current live E3 policy,
random probing, periodic/staleness probing, and structural criticality/staleness/dependency probing.
Use the same legal-action budget, at least 30 independent seeded episodes per arm, and agent-owned
runtime evidence only. Primary outcomes are held-out transition accuracy and policy progress with
paired intervals; incidental levels are disclosed with `solve_provenance=live_agent_self_discovery`
but receive no solve or registry credit.

**Deliverable:** `results/experiment_5833_arc_world_feedback_probe_ab.json`

## Phase 3 — Portable state and reconciliation (Exp5834-Exp5836)

### Exp5834 — Bounded adaptive-memory microkernel

Translate only Exp5828-accepted operations into a backend-neutral update/lookup/supersede/rollback
ABI with Python/Rust parity, deterministic vectors, interruption/restart equivalence, bounded arrays,
fixed-point sensitivity, and verified useful-work accounting. This is correctness/portability work;
the retired allocation-free sampler 10x path is not reused and no speed claim is made.

**Deliverable:** `results/experiment_5834_bounded_adaptive_memory_microkernel.json`

### Exp5835 — Attached-board capability and precondition receipts

Target each attached board: KV260, PolarFire SoC Icicle, and GateMate. Map the bounded kernel into
resource, precision, update, observability, host-transport, and toolchain requirements. Recompute
canonical authenticated precondition hashes and run a bounded non-destructive command only when the
relevant board hash changed. Otherwise emit `cached_unchanged` or `blocked` receipts. Do not touch
flash/block devices and do not claim speed, power, energy, convergence, 10x, Extropic, or Kona
execution.

**Deliverable:** `results/experiment_5835_attached_board_adaptive_memory_receipts.json`

### Exp5836 — Milestone capstone and reconciliation

Aggregate every available Exp5823-Exp5835 artifact by hash. Preserve blocked, negative, null,
positive, flagged, missing, oracle, and oracle-distinct classes; skip flagged evidence from
headlines; apply retire-if-same-verdict rules; run the publication gate without changing the frozen
FoVer 0.9131 result; update internal specs/traceability/status/changelog; and make no submission or
public-document edit.

**Deliverable:** `results/experiment_5836_capstone_v519.json`

## Dependency graph

```text
Exp5823 transition ───────────────────────────────────────────────┐
Exp5824 source refresh ──────────────────────────────────────────┤
                                                                 |
Exp5825 contract --> Exp5826 stream --> Exp5827 learner          |
                           |               |                      |
                           |               v                      |
                           |          Exp5828 lifecycle --> Exp5829 replay
                           |               |
                           |               v
                           |          Exp5834 microkernel --> Exp5835 boards
                           |
                           +--> Exp5830 embeddings --> Exp5831 verifier

Exp5832 ARC fact tape --> Exp5833 ARC world-feedback A/B

all available Exp5823-Exp5835 evidence ------------------------> Exp5836
```

Structured runtime gates:

| Downstream | Upstream field | Condition |
|---|---|---|
| Exp5826 | Exp5825 `adaptive_memory_contract_ready_score` | `== 1.0` |
| Exp5827 | Exp5826 `constraint_event_stream_ready_score` | `== 1.0` |
| Exp5828 | Exp5827 `structural_learner_ready_score` | `== 1.0` |
| Exp5829 | Exp5828 `future_validated_lifecycle_ready_score` | `== 1.0` |
| Exp5830 | Exp5826 `constraint_event_stream_ready_score` | `== 1.0` |
| Exp5831 | Exp5830 `paired_embedding_corpus_ready_score` | `== 1.0` |
| Exp5833 | Exp5832 `arc_fact_tape_ready_score` | `== 1.0` |
| Exp5834 | Exp5828 `future_validated_lifecycle_ready_score` | `== 1.0` |

Exp5835 and Exp5836 are deliberately ungated. Hardware continuity must report unchanged/blocked
state, and the capstone must reconcile every branch even when science gates fail.

## Hardware requirements

| Experiments | Compute | Memory/storage | Expected wall time | Notes |
|---|---|---|---|---|
| Exp5823-Exp5826 | CPU | 8-16 GB RAM, <5 GB new artifacts | 30-120 min each | Exact solver/data and infrastructure work; no LLM. |
| Exp5827-Exp5829 | CPU, optional parallel exact solvers | 16-32 GB RAM | 2-4 h each | N>=30 cells, chronological replay, paired bootstrap intervals. |
| Exp5830 | 2x RTX 3090 via sequential safe placement | >=48 GB aggregate VRAM available, >=48 GB RAM, >=20 GB disk | 3-6 h | All three mandated cached Q4_K_M GGUFs; embeddings only; checkpoint every batch. |
| Exp5831 | CPU; GPU optional for training only | 32 GB RAM | 1-3 h | Small logistic/MLP/KAN controls over cached fixed vectors. |
| Exp5832-Exp5833 | CPU live ARC runtime | 16 GB RAM | 2-5 h | No LLM, no adapters, no source/BFS; at least 30 seeded episodes per arm. |
| Exp5834 | CPU Rust/Python toolchain | 16 GB RAM | 2-4 h | Correctness/parity only; no acceleration claim. |
| Exp5835 | Host plus currently attached boards | Existing SSH/JTAG/USB paths only | 1-3 h | Commands only on changed authenticated preconditions; never flash. |
| Exp5836 | CPU | 8 GB RAM | 1-2 h | Aggregation, adversarial verification, spec/ops reconciliation. |

Required local model IDs for Exp5830:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

Legacy small models may be used only for CPU smoke tests and never for headline numbers.

## Success, stop, and retirement rules

- A gate scalar is emitted bare and only after its preregistered sample, integrity, and paired-CI
  conditions pass. Missing evidence is not zero and never passes a gate.
- Exact solvers in Exp5826-Exp5829 are oracles. Their positive learning results support FR-11
  lifecycle claims, not an oracle-distinct verifier-moat headline.
- Exp5831 is oracle-distinct at inference. If every headroom-present held-model-family comparison is
  clean null, the final-layer GGUF embedding verifier lane retires rather than scaling or retuning.
- Exp5833 is credited only through live-agent self-discovery. It cannot claim a public solve,
  registry delta, or level target; source inspection, offline ground-truth BFS, and per-game adapters
  are critical violations.
- Exp5835 may claim only capability mapping and actual bounded command receipts. Unchanged hashes
  forbid a repeated board probe. No proprietary hardware execution is inferred.
- Exp5836 always runs, excludes flagged artifacts from headlines, and preserves the frozen FoVer
  `AUROC=0.9131`, `paper_ready=true` decision. Only the operator may submit or publish.

## Explicitly deferred

- Any further finite-ID, grammar-only, split-budget, shared-budget, stop-token, or parser retry on
  the same GGUF generated-answer transport.
- Generated-text/logprob external scorers, PHASE D repairs, learned-judge authority, and
  energy-as-generator work.
- ARC CEGIS refinement, public-game re-solves, offline ground-truth BFS, per-game adapters, source
  inspection, registry credit, and novelty/first-contact signal retuning.
- Two-axis or parallel tempering and the retired allocation-free one-axis 10x sampler path.
- TSU, Kona, Extropic, or Aleph execution; no authenticated local dependency exists.
- Production answer-path integration, shadow deployment, leaderboard submission, and public paper
  or README edits.


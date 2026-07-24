# Carnot Research Roadmap vNEXT — Milestone 2026.07.522

**Milestone:** 2026.07.522  
**Title:** Certified Adaptive Memory, Layer-Dynamic Energy, and Live Causal Recurrence  
**Status:** Proposed  
**Task range:** Exp5863-Exp5876 (14 experiments)  
**Execution file:** `research-roadmap-next.yaml`  
**Date planned:** 2026-07-23

## Thesis

Milestone `.521` produced a clean positive continuous-self-learning result and a
substantially implemented Python/Rust state kernel, but it did not produce a
promotion-ready end-to-end system. Exp5856-Exp5858 qualified versioned lifecycle,
selective replay, and reduced-oracle learning with immutable model weights. Exp5859
then demonstrated exact cross-language operation, hash, serialization, rollback, and
invalid-input parity, but its readiness score stayed zero because the repository-wide
Python suite exited 2. Rewriting that kernel would discard evidence; `.522` first
attributes and repairs only the failing end-to-end seam.

The representation branch ended more decisively. Exp5852 extracted paired final
embeddings from the three mandated current GGUF families, but Exp5853 disqualified
the route under claim-flip, disaggregated-cell, label-permutation, norm/length, and
model-identity controls. Exp5854 and Exp5855 correctly gate-skipped. The final
embedding is therefore closed. Recent ICR and HARP work suggests a materially
different surface—cross-layer residual updates and a compact unembedding-derived
reasoning subspace—but the pinned GGUF runtime may not expose those tensors. `.522`
puts a hard public-API feasibility gate before extraction or training and imports
CORVUS-style camouflage controls if the surface exists.

The ARC branch also needs a true mechanism change. Exp5860 was a null, but not a clean
one: its artifact declared `live_llm_inference`, recorded approximately 10 ms model
calls, omitted the seed, ran below the compute-duration floor, and retained a failing
global suite exit. Repeating the active observer would not advance the live path.
`.522` instead builds a game-blind recurrence memory from the live agent's own
observed transitions. It performs no imagined rollout or lookahead. Its job is to
remember repeated causal signatures, contradiction evidence, and action outcomes
within an episode, then route legal proposals under a matched budget. It claims no
public-game solve.

The milestone makes four falsifiable claims:

1. The existing adaptive-state kernel can become promotion-ready by closing its
   actual end-to-end test seam without changing already-passing semantics.
2. A default-off shadow adapter plus a chronological future stream can continuously
   improve bounded external state while certifying retention, safety, rollback, and
   immutable GGUF weights.
3. If the pinned GGUF stack exposes a reproducible intermediate-layer surface,
   cross-layer dynamics can beat hardness, surface, identity, and final-embedding
   controls on held-model and held-constraint splits. If the surface is unavailable
   or the controls survive, the bounded route closes.
4. Observed-transition recurrence improves live E3 evidence efficiency or proposal
   support under matched action and model-call budgets. A clean null retires this
   mechanism.

## What milestone 2026.07.521 proved

| Branch | Terminal evidence | Consequence for `.522` |
|---|---|---|
| Transition | Exp5849 archived the exact `.520` boundary and allocated Exp5849-Exp5862 without collision. | Exp5863 must archive every activated `.521` identity, including blocked, gate-skipped, disqualified, flagged, and no-change outcomes, without laundering them. |
| Source currency | Exp5850 found zero accepted post-marker findings. Its artifact received a methodology warning because non-LLM source aggregation was incorrectly subjected to model-spec expectations. | Exp5864 starts after `V522-PLANNER-REFRESH-20260723-END`, declares external-source aggregation honestly, and treats zero accepted deltas as complete. |
| Replay provenance | Exp5851 established deterministic exact replay as the honest substrate and rejected false compute markers. | All adaptive-memory tasks must use deterministic exact-verifier receipts rather than CUDA/GGUF language when no model runs. |
| Current-model final embeddings | Exp5852 ran all three mandated SOTA GGUF families and emitted the paired corpus. The artifact was usable but warned for missing seed provenance. | Preserve the corpus and warning; do not claim current-model portability from extraction alone. |
| Final-embedding integrity | Exp5853 disqualified the route: claim-flip direction and disaggregated cells failed; label/pair permutation survived; norm/length controls failed; raw dimensions exposed model identity. | Final embeddings are not retried. Exp5870-Exp5872 may proceed only on a distinct, reproducible intermediate-layer surface and carry Exp5200/Exp5853 prior-failure records. |
| Downstream representation work | Exp5854 and Exp5855 gate-skipped after Exp5853. | Preserve the skips. No energy training or release routing may consume the disqualified surface. |
| Adaptive lifecycle | Exp5856 produced clean, deterministic, versioned promotion/quarantine/rollback evidence. | Use it as the semantic upstream for the kernel and adapter; do not rerun lifecycle science. |
| Selective replay | Exp5857 qualified transfer-selective replay with clean provenance. | The prospective stream may use bounded compatible replay with all-replay, no-replay, and shuffled controls. |
| Reduced-oracle learning | Exp5858 qualified continuous self-learning with zero unsafe accepts. | `.522` must test the same scientific claim through the actual pipeline shadow boundary and future chronological batches. |
| Python/Rust kernel | Exp5859 passed focused Python/Rust parity, hash, serialization, rollback, binding, lint, clippy, and invalid-input checks. Readiness remained zero only because `.venv/bin/pytest tests/python -q` exited 2. | Exp5865 attributes the failing suite seam and requalifies the existing kernel; it does not redesign the ABI. |
| ARC active observation | Exp5860 reported a null, but adversarial verification raised `DURATION_TOO_SHORT` and missing-seed flags; its model-call accounting is inconsistent with real local inference. | Do not use the null as clean mechanism evidence and do not repeat active observation. Replace it with observed-transition causal recurrence and require real model/GPU/seed receipts. |
| Hardware | Exp5861 recorded no changed authenticated state-operation route: KV260 retained the programmed-image POC, PolarFire retained a prior physical workload only, and GateMate remained IDCODE-blocked. | Exp5875 is conditional on a requalified kernel and a changed authenticated route; unchanged preconditions produce a no-change receipt without repeated board commands. |
| Capstone | Exp5862 stayed blocked because required checks included the disqualified representation branch, blocked kernel, flagged ARC artifact, and no-change hardware receipt. | `.522` must classify independent branches rather than require every scientific hypothesis to be positive. |

## The three biggest gaps to the PRD vision

### Gap 1 — FR-11 is scientifically positive but not integrated

The PRD requires autonomous ongoing learning, validation, bounded retention, and
rollback. Carnot now has positive exact-stream science and almost-complete
cross-language state semantics, but the state kernel is not promotion-ready and the
production verify/repair path does not consume it through a default-off protocol.
There is no prospective pipeline receipt showing that chronological updates improve
future batches without changing GGUF weights or release authority.

### Gap 2 — the verifier moat has no portable oracle-distinct internal energy

The stable FoVer headline is execution-grounded. PHASE D external text/logprob
rankers are retired, Exp5200's absolute hidden-state probe was negative, and
Exp5853 closed the final-embedding comparative route. The remaining live research
question is whether a reproducible layer-dynamic signal exists on current local
models and survives model-family transfer, constraint-family transfer, surface
relabeling, perturbation, and simple controls. Runtime feasibility is currently
unknown and must be settled before another GPU-scale probe.

### Gap 3 — the live ARC agent does not retain useful causal structure online

All public game levels are already registry-reproduced, yet the adapter-disabled E3
path still fails on hard held-out games. Candidate coverage and lookahead have both
shown limited or zero deliverable headroom, and Exp5860 did not perform verifiable
live LLM observation selection. The live agent needs a reusable, game-blind
within-episode state that learns only from its own actions and exact observations,
routes future proposals, and is evaluated on efficiency and evidence quality rather
than another public solve.

## 2025-2026 research update and experiment hooks

The dated source ledger is in `research-references.md` under
`V522-PLANNER-REFRESH-20260723-END`.

| Finding | Carnot implication | Experiment hook |
|---|---|---|
| [ICR Probe](https://arxiv.org/abs/2507.16488) | Cross-layer residual-stream updates can carry information absent from one isolated state. | Exp5870 first proves a supported GGUF layer surface; Exp5871 uses token-aligned layer differences only if that gate passes. |
| [HARP](https://arxiv.org/abs/2509.11536) | An unembedding-SVD reasoning subspace may remove semantic noise and reduce dimensionality. | Exp5871 freezes the projection before labels; Exp5872 compares it with raw layer differences, logistic, MLP, KAN, and final-embedding controls. |
| [CORVUS](https://arxiv.org/abs/2601.14310) | Internal telemetry can be camouflaged and same-family probe success is insufficient. | Exp5872 adds bounded feature perturbation, evaluator swap, family transfer, and exact-grounding audits. The learned score is never release authority. |
| [Solver-Hard Is Not Model-Hard](https://arxiv.org/abs/2607.17047) | SAT solver conflicts, clause density, surface form, and model difficulty are distinct axes. | Exp5868 builds proof-hard/proof-easy near-density-matched fixtures and proof-preserving relabels; Exp5869 establishes headroom and control behavior before GPU extraction. |
| [CerCE](https://openreview.net/forum?id=Anh6VfNM22) | Continual updates need explicit non-forgetting constraints rather than retention prose. | Exp5867 requires a per-update retention certificate and automatic rollback on any protected-cell regression. |
| [Current Agents Fail to Leverage World Models as Tools](https://arxiv.org/abs/2601.03905) | The bottleneck is often when and how agents use evidence, not simulator availability. | Exp5873-Exp5874 avoid imagined rollouts and instead test observed causal recurrence as a routing primitive. |
| [Structured SAT on Ising Machines](https://arxiv.org/abs/2511.21046) | Tight constraints can distort physical Ising dynamics; hybrid classical preprocessing is safer. | Exp5875 keeps tight bounds and release checks classical and maps only bounded residual operations after authenticated route change. |

## Architecture

```text
                        immutable authority boundary
             exact validators / exact observations / state hashes
                                   │
            ┌──────────────────────┼──────────────────────┐
            │                      │                      │
            ▼                      ▼                      ▼
   Exp5856-5859 evidence   Exp5868 exact CSP corpus   live E3 observations
            │                      │                      │
            ▼                      ▼                      ▼
   Exp5865 kernel          Exp5869 headroom and      Exp5873 causal
   requalification         shortcut qualification   recurrence controller
            │                      │                      │
            ▼                      ▼                      ▼
   Exp5866 default-off     Exp5870 pinned-GGUF       Exp5874 matched live A/B
   pipeline adapter        layer-surface preflight   (no solve credit)
            │                      │
            ▼                      ├── blocked → skip Exp5871/5872
   Exp5867 prospective     │
   certified CSL           ▼
                    Exp5871 three-family layer dynamics
                             │
                             ▼
                    Exp5872 portability + camouflage audit
            │                      │                      │
            └──────────────┬───────┴──────────────────────┘
                           ▼
                 Exp5875 conditional board receipt
                           │
                           ▼
                 Exp5876 capstone reconciliation

Cross-cutting:
  Exp5863 exact transition/archive
  Exp5864 post-marker SOTA ingestion
```

Learned scores may rank or route candidates. Exact solvers and validators remain
label, promotion, and release authority. The adaptive learner changes versioned
external state only. ARC memory contains only the live agent's own transitions.
Hardware never converts a requested topology, simulator, or fallback into physical
execution evidence.

## Phase 0 — Exact Boundary and Promotion Seam (Exp5863-Exp5865)

### Exp5863: Archive `.521` and allocate `.522`

Archive all 14 activated `.521` identities with exact paths and conductor outcomes.
Preserve Exp5853 as disqualified, Exp5854-Exp5855 as gate-skipped, Exp5859 as
blocked, Exp5860 as flagged null, Exp5861 as no-change, and Exp5862 as blocked.
Append `.521` to completion history exactly once if absent and prove
Exp5863-Exp5876 collision-free.

**Deliverable:** `results/experiment_5863_transition_v522.json`

### Exp5864: Post-V522 SOTA delta ingestion

Run the low-concurrency post-marker source refresh required by the SOTA-ingestion
discipline. Accept only newer, non-duplicate findings that sharpen existing task
controls. It cannot change identities, gates, authority, or retired scopes. Zero
accepted findings is complete.

**Deliverable:** `results/experiment_5864_v522_source_delta_ingestion.json`

### Exp5865: Adaptive-state kernel E2E attribution and requalification

Reproduce the Exp5859 repository-wide exit 2, identify exact failing collection
nodes or environment preconditions, and separate task-owned failures from unrelated
workspace debt. Preserve the already-passing ABI and conformance traces. Fix only an
owned seam, rerun focused and applicable end-to-end checks, and emit readiness only
when every claimed command exits zero.

**Deliverable:** `results/experiment_5865_adaptive_state_kernel_requalification.json`

## Phase 1 — Certified Self-Learning and Hardness Controls (Exp5866-Exp5869)

### Exp5866: Default-off adaptive-state pipeline shadow adapter

Behind the Exp5865 gate, add the smallest protocol-based adapter that mirrors
verify/repair events into the requalified kernel. The default path and user-visible
outputs must be byte-identical with the adapter off and on in shadow mode. Exact
validators keep release authority; shadow state cannot change acceptance.

**Deliverable:** `results/experiment_5866_adaptive_state_pipeline_shadow_adapter.json`

### Exp5867: Prospective certified continuous self-learning

Run a chronological future-batch experiment through the shadow boundary. Compare
compatible replay, all replay, no replay, shuffled replay, and reset-state controls.
Every accepted update must carry an exact future-lift receipt and a non-forgetting
certificate over protected prior cells. Regression, unsafe acceptance, capacity
overflow, or restart mismatch forces rollback. GGUF weights remain immutable.

**Deliverable:** `results/experiment_5867_prospective_certified_continuous_learning.json`

### Exp5868: Hardness-controlled constraint fixture

Construct solver-labeled proof-hard and proof-easy Tseitin families with
near-matched clause density, matched width, satisfiable/unsatisfiable balance,
candidate certificates, length controls, and proof-preserving variable relabels.
Store every row and solver receipt. This is exact dataset infrastructure, not a model
claim.

**Deliverable:** `results/experiment_5868_hardness_controlled_constraint_fixture.json`

### Exp5869: Headroom, surface, and oracle-distinctness qualification

Audit the Exp5868 corpus before model compute. Quantify label balance, solver-conflict
separation, density/length leakage, relabel equivalence, trivial surface-control
performance, current exact-verifier circularity, and the remaining oracle-distinct
headroom. Only a clean corpus with non-saturated controls can feed the representation
branch.

**Deliverable:** `results/experiment_5869_hardness_surface_headroom_audit.json`

## Phase 2 — Layer-Dynamic Moat and Live ARC Generalization (Exp5870-Exp5874)

### Exp5870: Pinned GGUF intermediate-layer surface preflight

Inspect the exact installed llama.cpp/llama-cpp-python versions and public C/Python
APIs. Using a mandated cached GGUF and embedded tokenizer, determine whether
deterministic token-aligned intermediate states can be extracted twice with stable
shape, layer identity, device, and checksum. No custom runtime fork,
`AutoTokenizer`, transformer-native weight load, monkey patch, or undocumented
activation hook is allowed. A clean unavailable result closes the branch and skips
Exp5871-Exp5872.

**Deliverable:** `results/experiment_5870_gguf_layer_surface_preflight.json`

### Exp5871: Three-family layer-dynamic causal representations

Only if Exp5869 and Exp5870 pass, run non-generative forward extraction over the
exact paired CSP rows using:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

Freeze layer selection and any HARP-style unembedding-SVD basis without test labels.
Emit token-aligned cross-layer deltas and a complete row ledger. No generated answer,
logprob reranker, or final-embedding reopening is permitted.

**Deliverable:** `results/experiment_5871_three_family_layer_dynamics.json`

### Exp5872: Portable energy and camouflage audit

Train only small interpretable scorers over cached Exp5871 features. Require held-model
and held-constraint lower bounds over logistic, MLP, compact KAN, raw layer-delta,
HARP projection, solver-conflict, density/length, identity, and Exp5853 final-embedding
controls. Run claim flips, relabels, permutations, evaluator swaps, bounded feature
perturbations, and no-information controls. The score never releases an answer.

**Deliverable:** `results/experiment_5872_layer_dynamic_portability_audit.json`

### Exp5873: Game-blind causal recurrence controller

Implement a default-off live-E3 component that stores bounded transition signatures,
legal actions, exact frame-difference outcomes, contradiction evidence, and recurrence
counts from the current episode. It may route legal proposals but cannot simulate
future frames, use registry trajectories, inspect source, invoke adapters, or read
offline BFS. Recorded-tape tests must prove determinism, capacity, reset, shuffle,
and forbidden-channel behavior before a live run.

**Deliverable:** `results/experiment_5873_arc_causal_recurrence_controller.json`

### Exp5874: Matched live causal-recurrence A/B

Run the controller on the canonical adapter-disabled E3 path with real current local
GGUF inference, explicit non-null seeds, GPU receipts, and matched action/model/token
budgets. Compare current E3, recurrence memory, shuffled memory, and reset/no-memory
controls across held-out public games without re-solving or updating the registry.
Primary metrics are verified transition-prediction error, proposal support, no-op and
invalid-action rate, evidence gained per action, and actions to first observed progress.
Any incidental level result is descriptive only and must have
`solve_provenance: live_agent_self_discovery`.

**Deliverable:** `results/experiment_5874_arc_live_causal_recurrence_ab.json`

## Phase 3 — Conditional Hardware and Reconciliation (Exp5875-Exp5876)

### Exp5875: Changed-route attached-board state-operation receipt

Run only after Exp5865 readiness. Recheck whether KV260, PolarFire, or GateMate has a
changed authenticated route. If none changed, avoid repeated programming/probing and
record no-change. If one changed, execute only bounded same-input state operations
against the CPU reference, retain tight constraints and release logic classically, and
report exact state/hash parity. No speed, power, energy, thermalization, TSU, Kona, or
sovereignty claim is authorized.

**Deliverable:** `results/experiment_5875_attached_board_adaptive_state_receipt.json`

### Exp5876: Four-branch capstone and documentation reconciliation

Replay every structured gate, run fresh adversarial verification, recompute load-bearing
metrics, classify each task independently, and apply same-verdict retirement rules.
Reconcile internal OpenSpec, traceability, status, changelog, and conductor records.
External publication remains operator-only.

**Deliverable:** `results/experiment_5876_v522_capstone_reconciliation.json`

## Dependency graph

```text
Exp5863 transition ───────────────────────────────────────────────┐
Exp5864 source ingestion ────────────────────────────────────────┤
                                                                 │
Exp5856-5859 ── Exp5865 kernel requalification                   │
                         │                                       │
                         ├── Exp5866 shadow adapter ── Exp5867 CSL
                         └─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ Exp5875 hardware
                                                                 │
Exp5868 exact CSP fixture ── Exp5869 headroom audit               │
                                      │                           │
Exp5870 GGUF layer preflight ─────────┤                           │
                                      ▼                           │
                             Exp5871 layer dynamics               │
                                      │                           │
                                      ▼                           │
                             Exp5872 portability audit            │
                                                                 │
Exp5873 ARC recurrence controller ── Exp5874 live matched A/B     │
                                                                 │
all terminal outcomes ────────────────────────────────────────────┤
                                                                 ▼
                                                        Exp5876 capstone
```

Structured conductor gates:

- Exp5866 requires `Exp5865.adaptive_state_microkernel_requalified_score == 1.0`.
- Exp5867 requires `Exp5866.adaptive_state_shadow_adapter_ready_score == 1.0`.
- Exp5869 requires `Exp5868.hardness_controlled_fixture_ready_score == 1.0`.
- Exp5871 requires both
  `Exp5869.hardness_surface_headroom_ready_score == 1.0` and
  `Exp5870.gguf_intermediate_layer_surface_ready_score == 1.0`.
- Exp5872 requires `Exp5871.layer_dynamic_representation_ready_score == 1.0`.
- Exp5874 requires `Exp5873.causal_recurrence_controller_ready_score == 1.0`.
- Exp5875 requires `Exp5865.adaptive_state_microkernel_requalified_score == 1.0`;
  board-route change is checked inside the task because it is an external precondition,
  not an upstream artifact field.

## Prior-failure discipline

| New task | Prior evidence | Material change |
|---|---|---|
| Exp5865 | Exp5859 `blocked: adaptive_state_microkernel_conformance_incomplete` | Reproduces and attributes the sole failing global-suite seam; preserves already-passing ABI semantics. Same verdict retires requalification. |
| Exp5866 | Exp5775/Exp5789 `blocked_gate_check_failed` | Clean lifecycle, replay, reduced-oracle, and requalified-kernel artifacts now exist before adapter work. |
| Exp5867 | Exp5709 and Exp5773/Exp5787 `blocked_gate_check_failed` | Uses the clean exact stream and real shadow adapter instead of missing or parse-unqualified generated streams. |
| Exp5871 | Exp5200 absolute hidden-state negative; Exp5853 final-embedding disqualification | Cross-layer dynamics and unembedding subspaces are distinct from absolute/final embeddings and are feasibility-gated. |
| Exp5872 | Exp5853 disqualification | Adds held-family lower bounds, HARP/ICR features, hardness controls, and CORVUS-style perturbations; same control-surviving result retires the layer route. |
| Exp5873/Exp5874 | Exp5860 flagged `complete_null` | Replaces active observation with observed-transition recurrence, real model receipts, non-null seeds, and no imagined rollouts. A clean repeated null retires recurrence. |
| Exp5875 | Exp5861 no-change | Requires a requalified kernel and changed authenticated route; unchanged state is recorded without repeated probing and same verdict retires this mapping attempt. |

No task reuses a retired experiment ID. No `requires` chain references a retired
upstream ID.

## Model policy

Every LLM-using experiment declares `MODEL_SPECS` with at least one mandated current
local GGUF:

- `unsloth/Qwen3.6-35B-A3B-GGUF` — flagship MoE
- `unsloth/gemma-4-31B-it-GGUF` — flagship dense
- `unsloth/gemma-4-26B-A4B-it-GGUF` — middle MoE

Exp5871 uses all three for headline representation rows. Exp5870 uses the cached
Qwen flagship for the runtime feasibility receipt. Exp5874 uses the flagship MoE and
flagship dense families for live and replication rows. GGUF files are loaded by local
path through llama.cpp with their embedded tokenizers. `AutoTokenizer` on a GGUF repo,
mock inference, CPU smoke presented as GPU evidence, and legacy small-model headline
rows are prohibited.

## Hardware requirements

| Experiment | Compute and access | Memory/storage | Estimated time | Hard boundary |
|---|---|---:|---:|---|
| 5863-5864 | CPU, network for 5864 | 8 GB RAM | 1-3 h each | No external writes except the internal source ledger; no publication |
| 5865-5867 | CPU, Rust/PyO3 toolchain | 16 GB RAM, 10 GB free | 2-6 h each | Preserve passing ABI; no GGUF weight mutation |
| 5868-5869 | CPU, PySAT/Z3/current exact solvers | 16 GB RAM | 2-5 h each | Solver is label authority; conflicts are a covariate, not truth confidence |
| 5870 | Dual RTX 3090 llama.cpp stack, one cached flagship GGUF | 24 GB VRAM aggregate minimum, 40 GB model cache | 1-3 h | Public/pinned runtime API only; unavailable is terminal |
| 5871 | Dual RTX 3090, all three cached SOTA GGUFs | 48 GB total VRAM, 120 GB cache/results headroom | 4-12 h | Non-generative forward extraction; embedded tokenizers |
| 5872 | CPU/GPU for small scorers | 16 GB RAM, optional 8 GB VRAM | 3-8 h | No release claim; held-family and perturbation controls |
| 5873 | CPU unit/E2E harness | 16 GB RAM | 3-6 h | Default-off; no source, adapter, BFS, registry replay, or simulator |
| 5874 | Live ARC SDK/network plus dual RTX 3090 and cached Qwen/Gemma | 48 GB total VRAM | 4-12 h | Canonical E3 path, matched budgets, non-null seeds, no registry update |
| 5875 | KV260 SSH, PolarFire/GateMate only if authenticated route changed; Rust/board toolchains | 16 GB RAM | 0.5-8 h | Never touch KV260 `/dev/mmcblk*`; no unchanged probe or performance claim |
| 5876 | CPU aggregation | 16 GB RAM | 2-5 h | No new science and no external publication |

Attached-board state at planning time:

- **KV260:** programmed-image sovereignty POC exists; access is SSH-only. No raw
  block-device command is allowed.
- **PolarFire:** prior physical workload evidence exists, but no current adaptive-state
  mapping is authenticated.
- **GateMate:** IDCODE/toolchain path remains blocked and opportunistic.
- **Extropic XTR-0/Z1 and Kona:** no authenticated local execution route or public
  reproducible weights; they are references only.

## Explicitly deferred or prohibited

- Reopening final-embedding, generated-text/logprob PHASE D, spilled-energy, finite-ID
  answer transport, grammar-only, tempering, or two-axis clamping mechanisms.
- Loading GGUF repos through `AutoTokenizer`, transformers, or an untracked activation
  hook.
- Training or mutating Qwen/Gemma weights as continuous self-learning.
- Public ARC game re-solves, source inspection, offline ground-truth BFS, per-game
  adapters, registry trajectory replay, or solver credit from a development proxy.
- A new FPGA bitstream redesign, unchanged board probing, or speed/power/energy claims
  without authenticated physical execution and matched CPU receipts.
- TSU or Kona execution claims.
- External arXiv, leaderboard, email, or publication action; those remain
  operator-only.

# Research Roadmap vNEXT

**Milestone:** `2026.07.512`  
**Title:** Finite-Choice Exact Proposals, Function-Preserving Self-Learning, Batched Rust Sampling, and ARC Causal Primitives  
**Status:** Proposed  
**Date:** 2026-07-19  
**Task range:** `exp5731`-`exp5742` (12 experiments, collision-checked)  
**Conductor file:** `research-roadmap-next.yaml`

## Milestone thesis

Milestone `.511` closed several ambiguities. The three mandated GGUF families and CUDA
runtime are present, but another free-form answer-envelope repair did not produce a
qualified protocol. The one-axis Rust kernel is production-reachable and quality-matched,
but its speed advantage reverses at larger problem sizes. The ARC epistemic ledger is
live-path safe but null, while a full-registry measurement exposed a 179-level live/oracle
gap concentrated in induction quality. Continuous self-learning did not run because it
was placed behind the failed SOTA stream.

Milestone `.512` changes those boundaries rather than repeating them:

1. replace free-form answers with a sealed finite-choice proposal channel, while keeping
   deterministic exact validators as the only authority;
2. make FR-11 non-cascading by running function-preserving, zero-gated KAN growth on the
   already promoted exact nonstationary stream, then optionally admit a new SOTA stream;
3. diagnose the large-size Rust reversal and expose a parity-tested batched sampler path
   before repeating any 10x software claim; and
4. satisfy the ARC generalization floor through game-blind, deletion-tested causal
   primitive mining and one generic live-path induction hardening attempt.

The milestone does not reopen native JSON grammar, external generated-text/logprob
scoring, token/logit semantic authority, model-weight writes, PTRM generation, learned
cross-game value heads, per-game ARC adapters, two-axis exchange, or TSU/Kona execution.

## What milestone `.511` proved

| Evidence | Terminal result | Consequence for `.512` |
|---|---|---|
| Exp5717 transition | Terminal `.510` evidence was preserved and the narrow parse-failed prospective stream scope was retired. | Start from the `.511` capstone; do not reopen that stream or its free-form protocol. |
| Exp5718 source ingestion | One post-planner continual-learning source sharpened retention controls; the execution itself was bibliographic and duration-flagged. | Keep a bounded freshness slot, but treat no-op ingestion as documentation work, not benchmark evidence. |
| Exp5719 answer-channel forensics | All three mandated GGUF files loaded with CUDA booleans, yet no model/protocol qualified: parse rate 0, 41 truncations, 82 missing answers, and no authenticated offload score. | Retire further free-form `FINAL:` repair. Change the deliverable to a one-step finite-choice proposal interface with randomized sealed labels. |
| Exp5720-Exp5722 SOTA stream and FR-11 chain | Exp5720 gate-blocked; Exp5721 emitted no artifact; Exp5722 gate-blocked. | Do not make continuous self-learning depend on another SOTA channel. Use the promoted exact stream first, then gate only optional SOTA ingress. |
| Exp5723 Rust integration | The one-axis corrected-cDLS kernel is exposed through production `SamplerBackend`, with exact fallback and readiness score 1. | The production boundary is real and can be extended, but semantics must remain unchanged. |
| Exp5724 matched crossover | 178 quality-matched pairs produced a terminal null. Rust was faster at some small sizes but slower at `n=48` and `n=96`; no consecutive larger-size crossover exists. | Profile the large-size reversal and add a batched boundary only if phase evidence identifies a fixable bottleneck. |
| Exp5725 ARC epistemic qualification | The ledger is live reachable, leakage-free, and safe, with no solve claim. | Live wiring alone remains insufficient. |
| Exp5726 ARC epistemic A/B | Six matched pairs were safe but null; the ledger was not promoted. | Do not retry the ledger. Change to causal action-effect primitives and deletion replay. |
| Exp5727 ARC generalization measurement | Across all 25 public games the live agent reproduced 4 of 183 oracle levels, leaving gap 179. `lf52`, `bp35`, and `su15` were worst and induction accuracy was near zero. | Generalization work must target game-blind world-model induction, not another public-game solve. |
| Exp5728 capstone | Honest blocked reconciliation: no answer channel, stream, FR-11 credit, Rust crossover, or ARC delta; Rust production reachability remained positive. | Preserve all nulls and use independent branches so one failure cannot erase the whole milestone. |

## The three largest gaps to the PRD vision

### Gap 1: continuous self-learning is still replay evidence, not safe live capacity growth

FR-11 requires auditable ongoing improvement from verifier feedback. Carnot has a promoted
exact nonstationary stream, an active-spline conformal KAN controller, and anytime-valid
audit evidence, but `.511` placed the next lifecycle experiment behind an unrelated GGUF
format failure. It also lacks a function-preserving rule for adding capacity without
changing the prior safe function at insertion.

`.512` response: Exp5735 adds zero-gated residual spline capacity and proves exact
pre-insertion equivalence before chronological learning. Exp5736 exercises remember,
update, supersede, forget, conflict reject, crash, and rollback under preregistered
`(epsilon, delta)` release bounds. Exp5737 optionally tests the same lifecycle on a sealed
SOTA-proposal stream, but only after both independent branches pass.

### Gap 2: the live ARC agent has a 179-level induction gap

The submitted path is safe and can maintain epistemic state, but only 4 of 183 registry
levels are reproduced under the full-registry live measurement. The worst games fail at
building an accurate action-effect world model, not at accessing an off-path solver. A
static list of seemingly relevant primitives is not enough; their causal contribution to
future agent behavior must be measured.

`.512` response: Exp5740 strips game identities from agent-owned traces, mines generic
action-effect primitives, and scores them by deletion-and-replay counterfactual trajectory
utility. Exp5741 hardens at most one positive generic primitive inside the live E3 policy
and measures the full registry under matched budgets. Only registry-new levels found by
the live agent's own attempts receive solve credit.

### Gap 3: the Rust production core has no large-size throughput win

The PRD calls for a Rust core with a 10x throughput advantage. Production reachability and
semantic parity are now established, but `.511` found an alternating crossover: Rust wins
at smaller sizes and loses at `n=48`/`n=96`. Aggregate timing is insufficient to decide
whether the cause is serialization, the PyO3 boundary, validation, restart handling, or
the kernel's state-update complexity.

`.512` response: Exp5738 performs phase-level profiling, then implements a deterministic
`sample_batch` path only where the profile justifies it. Exp5739 repeats the matched-quality
benchmark with single-core and fixed multi-core receipts, at least 30 batches per cell,
and a strict 10x end-to-end software gate. A null is a valid terminal result.

## Research incorporated before design

The complete planning sweep and dispositions are recorded in the `V512 Planner Refresh`
block of `research-references.md`.

| 2025-2026 source | Actionable idea | `.512` use |
|---|---|---|
| Generative Compilation, arXiv:2607.13921 | Make incomplete artifacts checkable and reject semantic dead ends early while an exact compiler stays authoritative. | Exp5733/Exp5734 use sealed finite candidate/label tables and exact validation; model token scores are proposal signals only. |
| Gate-Zero Growth, arXiv:2607.14571 | Add capacity at an exactly function-preserving zero gate, then bound drift while opening it. | Exp5735 adds zero-gated residual splines to the KAN sidecar, never to GGUF weights. |
| SMC-ES, arXiv:2607.15003 | Pair learned policy synthesis with explicit probabilistic safety/confidence certificates. | Exp5735/Exp5736 preregister `(epsilon, delta)` release bounds while keeping exact row validators and state hashes authoritative. |
| Campaign Diagrams, arXiv:2607.15225 | Analyze compute, memory, and latency phase by phase rather than hiding bottlenecks in one aggregate. | Exp5738 attributes the large-size Rust reversal before changing the boundary; Exp5739 reports end-to-end results. |
| Bridge Evidence, arXiv:2607.15253 | Deletion-and-replay counterfactuals reveal causally useful intermediate evidence. | Exp5740 scores game-blind ARC primitives by downstream trajectory effect, not static plausibility. |
| Verified DPLL transition systems, arXiv:2607.14999 | Separate state-transition rules, strategy, correctness, and termination. | Exp5736 uses deterministic typed lifecycle transitions and well-founded rollback tests; a Rocq port is deferred. |
| KAN-versus-MLP cost study, arXiv:2607.13413 | KAN accuracy must be evaluated against parameter and compute cost. | Exp5735 records parameter growth, update latency, memory, and a no-growth baseline. |
| Photonic Ising perspective, arXiv:2607.13446 | Connectivity, reconfigurability, scale, and end-to-end time remain core hardware constraints. | Watch-only; `.512` makes no photonic, FPGA, TSU, or other hardware-speedup claim. |

OpenReview, Hugging Face Papers, GitHub, Extropic, Logical Intelligence, and Semantic
Scholar checks supplied no stronger locally executable dependency. EBT and ARM-EBM
citation trails remain architecture context; Kona/Aleph and TSU hardware remain non-local.

## Target architecture after `.512`

```text
                    learned proposal, exact authority
                                  │
          ┌───────────────────────┴────────────────────────┐
          │                                                │
          ▼                                                ▼
┌───────────────────────┐                    ┌────────────────────────┐
│ mandated local GGUFs  │ Exp5733            │ exact nonstationary    │ Exp5735
│ one-step label scores │ finite-choice gate │ stream + KAN sidecar   │ zero-gate
└───────────┬───────────┘                    └────────────┬───────────┘
            │ sealed label/candidate map                  │ exact function
            ▼                                             │ preservation
┌───────────────────────┐ Exp5734                         ▼
│ exact-attested SOTA   │                    ┌────────────────────────┐
│ proposal stream       │                    │ lifecycle state machine│ Exp5736
└───────────┬───────────┘                    │ update/forget/rollback │
            │                                └────────────┬───────────┘
            └──────────────────────┬──────────────────────┘
                                   ▼
                         ┌──────────────────────┐ Exp5737
                         │ optional SOTA ingress│
                         │ shadow-only FR-11    │
                         └──────────────────────┘

 energy descriptors                         agent-owned ARC traces
          │                                            │
          ▼                                            ▼
┌───────────────────────┐ Exp5738          ┌────────────────────────┐ Exp5740
│ phase-profiled Rust   │                  │ game-blind primitive   │
│ sample_batch backend  │                  │ deletion/replay audit  │
└───────────┬───────────┘                  └────────────┬───────────┘
            ▼                                           ▼
┌───────────────────────┐ Exp5739          ┌────────────────────────┐ Exp5741
│ matched 10x software  │                  │ one generic live-path  │
│ crossover or null     │                  │ induction hardening    │
└───────────────────────┘                  └────────────────────────┘

 Exact validators remain final authority. No LLM judge, game source, per-game adapter,
 offline solve, learned cross-game value head, or unmatched hardware claim is allowed.
```

## Phase 1 - Evidence transition and exact proposal boundary

### Exp5731 - Transition terminal `.511` evidence

Archive every Exp5717-Exp5728 artifact and conductor outcome, preserving the missing
Exp5721 artifact and all gate/null states. Allocate Exp5731-Exp5742 only after a fresh
collision scan. Preserve the retired free-form stream, epistemic-ledger non-promotion,
two-axis retirement, and the positive Rust production integration.

**Deliverable:** `results/experiment_5731_transition_v512.json`

### Exp5732 - Post-V512 source-delta ingestion

Search only after the V512 planner marker and append only genuine non-duplicate deltas.
This is a bounded bibliographic task; zero accepted findings is a complete outcome and no
benchmark duration or compute claim is permitted.

**Deliverable:** `results/experiment_5732_v512_source_delta_ingestion.json`

### Exp5733 - Finite-choice GGUF proposal-channel qualification

Use all three mandated GGUF families on disjoint exact controls. Precompute candidate
answers, randomize and seal one-token label mappings separately per model tokenizer, and
read the next-token candidate-label scores directly. The model selects a proposal; an
independent exact validator decides correctness. Qualify only if every control has a
complete candidate set, every label is a unique one-token encoding, CUDA offload is
authenticated, and there are no missing/non-finite score rows or validator disagreements.
No generated free-form answer, JSON grammar, external scorer, or semantic logit threshold
is allowed.

**Deliverable:** `results/experiment_5733_sota_finite_choice_proposal_channel.json`

### Exp5734 - Sealed exact-attested SOTA proposal stream

**Gate:** Exp5733 channel readiness, two qualified flagship families, authenticated CUDA,
and zero control receipt failures.

Use the flagship Qwen MoE and Gemma dense GGUFs to choose among preregistered finite
candidates on at least 96 chronological rows. Exact validators mint admitted labels and
store rejected proposals with conflict receipts. Seal prefix/suffix and all model,
candidate, label, score, and validator hashes. Any missing row or provenance break blocks
the stream.

**Deliverable:** `results/experiment_5734_sota_exact_proposal_stream.json`

## Phase 2 - Function-preserving continuous self-learning

### Exp5735 - Zero-gated KAN continuous self-learning canary

This is the milestone's mandatory, non-cascading continuous self-learning experiment.
Replay the promoted Exp5616 chronological exact stream through the Exp5628 active-spline
controller, insert zero-gated residual spline components, prove bitwise/strict-tolerance
pre-insertion function equivalence, then allow gate opening only from verifier-attested
prefix events. Compare zero-gated growth with no-growth, always-open, MLP-residual, and
corrupted-order controls. Measure old-prefix retention, new-suffix improvement, unsafe
updates, parameter/memory growth, and update latency under a preregistered statistical
model-checking certificate.

**Deliverable:** `results/experiment_5735_zero_gate_kan_continuous_self_learning.json`

### Exp5736 - Typed lifecycle, conflict, and rollback canary

**Gate:** Exp5735 exact insertion equivalence, positive suffix improvement, retention
within margin, and zero unsafe updates.

Exercise remember, update, supersede, forget, reject, rollback, and recovery operations
over an untouched chronological suffix. Inject stale/conflicting advice, crashes at each
state transition, and corrupted checkpoints. Require exact state-hash restoration and
zero propagation of rejected constraints. Production remains disabled.

**Deliverable:** `results/experiment_5736_csl_lifecycle_conflict_rollback.json`

### Exp5737 - Optional SOTA-stream lifecycle ingress

**Gate:** Exp5734 stream readiness and Exp5736 lifecycle readiness.

Feed the sealed SOTA proposal-stream prefix through the qualified lifecycle controller in
shadow mode. Compare chronological versus corrupted-order and validator-label versus
model-proposal controls. Only exact-validator labels may update the sidecar; GGUF weights
and production defaults remain immutable. This integration is optional and does not
determine whether `.512` satisfies the continuous-learning floor.

**Deliverable:** `results/experiment_5737_sota_stream_csl_shadow_ingress.json`

## Phase 3 - Batched Rust sampling and 10x evidence

### Exp5738 - Large-size Rust phase profile and batched backend

Start from the production one-axis `SamplerBackend`. Reproduce the `n=48`/`n=96` reversal,
attribute serialization, PyO3, kernel, validation, restart, allocation, and memory phases,
then implement a deterministic `sample_batch` boundary only for the measured dominant
path. Require Python/Rust energy, proposal, exchange, scheduler, checkpoint, restart, and
distributional parity. No timing promotion occurs in this task.

**Deliverable:** `results/experiment_5738_one_axis_rust_batched_backend.json`

### Exp5739 - Matched-quality batched Rust/Python crossover

**Gate:** Exp5738 batch parity, exact fallback equivalence, restart parity, and backend
readiness.

Benchmark identical batches, seeds, ladders, transition budgets, checkpoints, warmups,
and thread/core allocations across Python and Rust. Use at least 30 batches per cell and
report single-core plus fixed multi-core end-to-end throughput, quality, confidence
intervals, memory, and phase receipts. A 10x claim requires two consecutive larger sizes
with matched quality and a lower confidence bound at least 10.0. Otherwise record a null.

**Deliverable:** `results/experiment_5739_one_axis_batched_10x_crossover.json`

## Phase 4 - ARC causal primitives and capstone

### Exp5740 - Game-blind ARC primitive causal-utility audit

Registry-precheck all 25 public games and use only existing agent-owned observation/action
traces. Strip game IDs and source-derived metadata. Mine generic action-effect primitives,
then delete each primitive and replay the same trace to measure changes in next-action
validity, world-model prediction, planning reachability, repeated-action rate, and budget.
No game is solved and `solve_provenance=development_proxy` is explicit.

**Deliverable:** `results/experiment_5740_arc_game_blind_primitive_causal_audit.json`

### Exp5741 - Generic live-path primitive hardening

**Gate:** Exp5740 finds at least one positive game-blind causal primitive, no source leak,
and complete counterfactual receipts.

Add at most one generic primitive to the live E3 world-model/induction path without game
names, IDs, source, adapters, or learned cross-game values. Run a matched full-registry
A/B under fixed 400-action budgets. Only levels beyond the registry precheck count, and
only when `solve_provenance=live_agent_self_discovery` is reproduced from the submitted
live path. A safe null is valid.

**Deliverable:** `results/experiment_5741_arc_generic_primitive_live_ab.json`

### Exp5742 - `.512` capstone reconciliation

Aggregate every Exp5731-Exp5741 artifact and every gate-skip/missing state. Reconcile
OpenSpec, traceability, status, changelog, conductor log, exclusions, known issues,
verifier gaps, north-star, hardware status, and applicable E2E receipts without changing
scientific verdicts. Preserve the independence of the SOTA, FR-11, Rust, and ARC branches.

**Deliverable:** `results/experiment_5742_v512_capstone_reconciliation.json`

## Dependency graph

```text
Phase 1
Exp5731 transition ───────────────────────────────────────────────────────┐
Exp5732 source delta ─────────────────────────────────────────────────────┤
                                                                         │
Exp5733 finite-choice channel ───────► Exp5734 sealed SOTA stream ───┐   │
                                                                    │   │
Phase 2                                                             │   │
Exp5735 zero-gate CSL ──────────────► Exp5736 lifecycle/rollback ────┼──►Exp5737
                                                                    │   │
Phase 3                                                             │   │
Exp5738 Rust batch backend ─────────► Exp5739 matched 10x/null ──────┤   │
                                                                    │   │
Phase 4                                                             │   │
Exp5740 ARC causal audit ───────────► Exp5741 generic live A/B ──────┤   │
                                                                    ▼   ▼
                                                               Exp5742 capstone
```

No `requires:` chain points to a retired experiment. Exp5735 is deliberately independent
of Exp5733/Exp5734 so the mandatory continuous self-learning experiment cannot be
cascade-skipped. Every natural-language gate is mirrored by a structured `gated_on` entry.

## Hardware and model requirements

| Resource | Tasks | Requirement and boundary |
|---|---|---|
| RTX 3090 GPU 0/1 | Exp5733-Exp5734 | CUDA-enabled `llama-cpp-python`, positive offloaded-layer and memory-delta receipts, one loaded model per device unless VRAM proof permits otherwise. CPU fallback is smoke-only. |
| Mandated local GGUFs | Exp5733 | `unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and `unsloth/gemma-4-26B-A4B-it-GGUF` in explicit `MODEL_SPECS`. |
| Flagship SOTA pair | Exp5734 | Qwen3.6-35B-A3B plus Gemma-4-31B, immutable cached `.gguf` files, native llama.cpp tokenization/logit access, no transformers. |
| CPU/RAM | Exp5731-Exp5732, Exp5735-Exp5742 | Exact solvers, KAN lifecycle, Rust/Python sampling, ARC trace replay/live policy, and reconciliation. Record peak memory for KAN growth and sampler benchmarks. |
| Rust/PyO3 toolchain | Exp5738-Exp5739 | Existing `carnot-samplers` crate/bindings, release build, deterministic checkpoint schema, fixed thread pools, and reproducible software receipts. |
| NVMe | All phases | Model hashes, sealed candidate/label tables, score rows, exact stream manifests, lifecycle checkpoints, sampler traces, ARC traces, and artifact hashes. |
| ARC live environment | Exp5740-Exp5741 | Agent-owned frames/actions and submitted live E3 path only. Registry precheck is mandatory; source, adapters, exhaustive BFS, and off-path solvers are forbidden. |
| KV260 / PolarFire / GateMate | None | No board experiment is load-bearing. Existing board continuity remains documented; no board speedup claim is made. |
| Extropic TSU / Kona / photonic Ising hardware | None | Watch-only architecture context with no authenticated local execution path. |

## Promotion, retirement, and claim rules

1. **Finite-choice gate:** Exp5733 qualifies only with authenticated CUDA, at least two
   flagship model families, unique one-token label encodings, complete sealed candidate
   sets, finite score rows, exact validator receipts, and zero receipt disagreement.
2. **Proposal-stream gate:** Exp5734 requires every row, candidate set, label permutation,
   score vector, proposal, exact-validator label, and split commitment to be replayable.
   Model token scores never become exact authority.
3. **Continuous self-learning gate:** Exp5735/Exp5736 may update only the external
   rollback-capable KAN sidecar. They require exact insertion equivalence, positive suffix
   improvement, old-prefix retention within margin, zero unsafe updates, bounded growth,
   and exact crash/restart/rollback replay. GGUF weights and production defaults stay fixed.
4. **Optional ingress gate:** Exp5737 cannot alter the FR-11 milestone verdict. It admits
   only exact-validator labels from Exp5734 and remains shadow-only.
5. **Rust backend gate:** Exp5738 must preserve semantic, distributional, checkpoint, and
   fallback parity. Exp5739 may claim 10x only from matched-quality end-to-end software
   timing at two consecutive larger sizes with the lower confidence bound at least 10.0.
6. **ARC mechanism gate:** Exp5741 implements no primitive unless Exp5740 shows positive
   deletion-replay causal utility and zero source/game identity leakage. It may bank only
   registry-new live-agent self-discovery levels.
7. **Failed reruns:** every matching carry-forward has complete `prior_failures` metadata.
   Repeating the same verdict activates `retire_if_same_verdict: true`.
8. **No claim inflation:** missing, gate-skipped, blocked, malformed, or development-proxy
   artifacts never count as successful live work. Offload is not model quality; a proposal
   is not an exact label; parity is not speedup; a known level is not a new ARC solve.

## Expected outputs

- one terminal `.511` evidence transition and collision-free `.512` allocation;
- one bounded execution-time source-delta artifact;
- one finite-choice qualification across all three mandated SOTA families;
- one gated, exact-validator-attested flagship proposal stream;
- three FR-11 artifacts, with the zero-gated exact-stream task independently satisfying
  the continuous self-learning floor;
- one parity-tested batched Rust backend and one matched 10x-or-null crossover artifact;
- one ARC game-blind causal primitive audit and one generic live-path A/B;
- one capstone reconciling specs, code, operations, exclusions, hardware boundaries, and
  negative/null evidence without changing verdicts.

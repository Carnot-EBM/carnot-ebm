# Research Roadmap vNEXT: Shared-State Energy and Licensed Online Learning

**Milestone:** `2026.08.543`
**Date:** 2026-08-10
**Experiments:** Exp6297-Exp6309
**Phases:** 4
**Primary requirements:** FR7, FR11, FR12
**North-star constraint:** move model-native state toward whole-configuration energy refinement without promoting an oracle, replay, or ARC proxy as the product result

## Milestone thesis

V542 completed its 13-task execution but did not promote a new branch. It did
prove two reusable foundations: bounded ASP energy has a vertex-exact
continuous relaxation, and ordinary model text can be reduced to fail-closed
partial atom evidence. It also produced decisive negative evidence. The
partial-evidence artifact was methodology-flagged, the flagship warm start
showed no value, revocable memory failed its recorded verification command,
and the ARC canary reached `0.91` rather than its exact `1.0` promotion gate.

V543 changes the representation and learning mechanisms. Instead of asking
generated text to carry the state, it uses the complete Exp5852 three-family
embedding corpus to learn one linear adapter pair per model into a shared
activation space. An independent audit must first remove the exact shortcuts
that disqualified Exp5853. Only then may shared model state initialize V542's
exact-vertex ASP relaxation. Cold starts, per-model raw embeddings, and the
exact solver remain controls.

Continuous self-learning updates only a small state initializer. Base GGUF
weights stay frozen. Online updates are anchored to a frozen reference and
committed only after an immutable exact verifier reveals the outcome. Prior
strategies retrieved across families are hypotheses, not knowledge, until the
target-side validator licenses them. The ARC branch applies the same rule to a
mechanic route learned from the live agent's own observations and makes no
level-solve claim.

## What V542 proved

| Branch | V542 evidence | V543 consequence |
|---|---|---|
| Milestone integrity | Exp6284 and Exp6296 preserved exact terminal classes across 13 tasks. The capstone found 6 complete, 1 flagged, 2 null, 3 skipped, and 1 missing/nonterminal branch state, with no promoted new branch. | Start from the exact capstone, repair evidence admission before new science, and keep branch-independent promotion. |
| Source freeze | Exp6285 accepted zero post-marker findings and terminated cleanly. | Search strictly after the V543 marker. Zero findings remain valid. |
| Continuous ASP energy | Exp6287 proved the multilinear relaxation agrees with the discrete ASP energy on every tested vertex and supplies checked gradients. | Reuse it as the editable state substrate. Do not rebuild the relaxation. |
| Partial evidence | Exp6288 implemented a fail-closed adapter, but the artifact was adversarial-flagged for a sub-microsecond compute receipt and missing methodology. | Do not rerun that adapter. Move from text evidence to an explicit cross-model activation interface. |
| Flagship value | Exp6289 loaded the mandated models but recorded failing verification commands and `warm_start_value_ready_score=0`. Exact completion remained an oracle-only repair. | Use a non-generation activation surface, preflight terminal evidence, and require value over cold and raw-embedding controls. |
| Revocable memory | Exp6290 implemented the lifecycle but ended `complete_null` because a recorded test command failed. Exp6291 gate-blocked, Exp6292 was absent, and Exp6293 gate-blocked. | Do not repeat advice memory. Update a small state initializer online with reference anchoring, exact post-outcome commits, and rollback. |
| ARC live path | Exp6294 produced a clean 67.7-second causal canary with no solve claim, but readiness was `0.91`. Exp6295 correctly gate-blocked. | Treat the route as an untrusted hypothesis. Require target-side evidence from the agent's own transitions before activation, then rerun the held-game audit only through a structured gate. |
| Publication state | Exp6296 kept the existing FoVer publication gates true while rejecting flagged, missing, skipped, null, blocked, oracle-only, and replay-only evidence. | Preserve the paper boundary and reconcile architecture claims only for independently promoted branches. |

## The three largest gaps to the PRD vision

### Gap 1: model-native state is not connected to exact global energy

FR12 requires deterministic constraints, while the Phase 3 vision requires a
continuous editable global state. Carnot currently verifies or repairs text
outside the model. Exp5852 extracted real three-family embeddings, but Exp5853
found raw model-dimension identity, claim-flip, pair-permutation, and norm or
length shortcuts. V543 learns an explicit per-model activation contract and
audits those exact failure modes before using it.

### Gap 2: continuous refinement has no measured model value

FR7 requires iterative inference over configurations. V542's ASP relaxation is
semantically valid, but its model warm start did not beat cold exact completion
or show reduced refinement work. V543 maps a shared activation into the
continuous atom state and measures exact validity, refinement steps, exact
fallback work, wall time, and harm on held task and model families. The exact
solver remains an oracle and cannot count as verifier value.

### Gap 3: autonomous learning is neither continuously useful nor safely transferable

FR11 requires online improvement, retention, immutable validation, and
rollback. Advice memories have repeatedly abstained, failed verification, or
gate-blocked. V543 makes the learned object smaller and more direct: the
model-to-state initializer. It compares frozen, unanchored online, and
reference-anchored updates on a chronological stream, then licenses any
cross-family strategy on the target's exact validator. Replay is not transfer.

## Research delta used by this roadmap

- Universal Activation Bus (`arXiv:2608.09521`) motivates one adapter pair per
  model and a frozen shared activation contract. This directly changes the
  mechanism that failed Exp5853.
- VERDI (`arXiv:2608.09537`) motivates target-side evidence licensing and the
  rule that retrieved experience is only a transfer hypothesis.
- SR-OPSD (`arXiv:2608.09745`) motivates reference-anchored online updates. V543
  adapts the geometry to a small initializer and does not fine-tune an LLM.
- Open-Ended Optimization (`arXiv:2608.09629`) supports allowing an adaptive
  update policy inside fixed objectives, budgets, data boundaries, and exact
  validation. Admission and rollback authority stay external.
- Thermodynamic matrix inversion (`arXiv:2608.09743`) motivates a deterministic
  optimizer control: stochastic or thermodynamic language alone is not a
  hardware advantage on a convex single-minimum problem.
- The 2026 KAN language-model audit (`arXiv:2607.15525`) found no consistent
  quality or latency advantage. V543 does not reopen KAN training.

The full source disposition is in `research-references.md` under the V543
planner marker.

## Target architecture

```text
                 sealed matched constraint texts
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
       Qwen3.6 GGUF      Gemma-4 dense     Gemma-4 MoE
            │                 │                 │
            └──── per-model linear encoder/decoder ────┐
                                                        ▼
                                              shared activation bus
                                                        │
                                  independent shortcut audit
                                  ┌─────────────┴─────────────┐
                                  │ ready                     │ closed
                                  ▼                           ▼
                         model-to-state initializer       no value claim
                                  │
                                  ▼
                  vertex-exact ASP continuous relaxation
                                  │
                    bounded gradient refinement / fallback
                                  │
                                  ▼
                   exact certificate ◄── Clingo oracle control
                                  │
          read snapshot ─► decide ─► reveal exact outcome ─► update
                                  │
                 frozen / unanchored / reference-anchored arms
                                  │
                                  ▼
                   target-side transfer license and rollback

ARC live branch:
own attempts ─► mechanic hypothesis ─► runtime transition checks
     │                                      │
     └── router off / retrieval only / licensed route A/B ──► no solve claim
```

## Phase 0: terminal evidence and source freeze

### Exp6297: V542-to-V543 terminal transition

Build the exact handoff from Exp6296 and the operational retro. Validate all 13
staged tasks, gates, prior-failure blocks, model rules, IDs, paths, and prompt
endings. Do not activate the staged roadmap.

**Deliverable:** `results/experiment_6297_v543_terminal_transition.json`

### Exp6298: terminal-evidence preflight linter

Add a standalone preflight that checks required artifact fields, terminal
prefixes, substrate and duration receipts, test-command existence, recorded
exit codes, and downstream gate fields before an artifact is eligible. It may
not modify the conductor. Replay the exact Exp6288-Exp6290 failure shapes.

**Deliverable:** `results/experiment_6298_terminal_evidence_preflight_linter.json`

### Exp6299: post-marker source and scope freeze

Search only after the V543 planner marker. Freeze activation-bus, state
initializer, online-learning, target-license, ARC, and no-hardware-claim
contracts. A zero-source delta is terminal.

**Deliverable:** `results/experiment_6299_v543_post_marker_source_scope_freeze.json`

## Phase 1: shared model state to exact energy

### Exp6300: three-family universal activation bus

Reuse the immutable Exp5852 paired embeddings. Fit one linear encoder-decoder
pair per model into a fixed shared space using unlabeled matched rows. Split by
task family, perturbation family, and template before fitting. Report
cross-model retrieval, reconstruction, neighborhood consistency, and model-ID
leakage. Do not train an energy head yet.

**Deliverable:** `results/experiment_6300_three_family_universal_activation_bus.json`

### Exp6301: independent activation-bus integrity audit

Reconstruct Exp6300 from hashes and replay the Exp5853 adversarial controls:
claim flips, pair swaps, label permutations, norm and length controls,
truncation, duplicates, model identity, and held-family folds. No pooled mean
may hide a failed disaggregated cell.

**Deliverable:** `results/experiment_6301_activation_bus_integrity_audit.json`

### Exp6302: shared activation-to-state initializer, gated on Exp6301 integrity=1

Only after the bus passes its audit, fit a bounded initializer from shared
state to V542's continuous ASP atom vector. Compare cold blank, cold random,
raw per-model, and shared-bus starts with identical refinement budgets. Require
exact validity non-inferiority and lower work on held folds.

**Deliverable:** `results/experiment_6302_shared_activation_state_initializer.json`

### Exp6303: live three-family shared-state benchmark, gated on Exp6302 readiness=1

Extract fresh embeddings from all three mandated GGUF families on a sealed
holdout. Compare the frozen shared initializer against cold and raw per-model
controls. Record CUDA, model, tokenizer, seed, raw-row, exact-work, and timing
receipts. Generated-answer transport is not used.

**Deliverable:** `results/experiment_6303_live_three_family_shared_state_benchmark.json`

## Phase 2: continuous self-learning and licensed transfer

### Exp6304: reference-anchored online state learning

Run the milestone's required continuous self-learning experiment independently
of Phase 1 promotion. On a sealed chronological ASP stream, compare a frozen
initializer, an unanchored on-policy update, and a reference-anchored update.
Reveal exact outcomes only after each decision. Update no GGUF weights. Measure
forward transfer, retention, regret, reversal, poison, rollback, and memory
cost.

**Deliverable:** `results/experiment_6304_reference_anchored_online_state_learning.json`

### Exp6305: evidence-licensed cross-family transfer, gated on Exp6301=1 and Exp6304=1

Build optimization fingerprints from shared probes. Compare no transfer,
retrieval-only transfer, and target-licensed transfer while holding out each
model and task family. A retrieved strategy remains inactive until a frozen
target calibration slice passes exact validation.

**Deliverable:** `results/experiment_6305_evidence_licensed_cross_family_transfer.json`

### Exp6306: independent online-learning safety audit

Audit Exp6304 even if utility is null. Inject false passes, contradictions,
stale references, reversals, poisoned rows, missing validators, process
restarts, and corrupted snapshots. Prove fail-closed behavior and byte-exact
rollback. Safety alone cannot promote utility.

**Deliverable:** `results/experiment_6306_online_state_learning_safety_audit.json`

## Phase 3: ARC live path and capstone

### Exp6307: ARC target-validated mechanic-route causal canary

Treat V542's mechanic route as an untrusted retrieved hypothesis. Compare
router-off, retrieval-only, and target-licensed arms. The licensed arm may
activate only after the live agent's own attempts and runtime transition checks
support the route. Use fresh fixtures, matched budgets, real flagship GGUF
receipts, and no hidden source or level-solve credit.

**Deliverable:** `results/experiment_6307_arc_target_validated_route_canary.json`

### Exp6308: held-game ARC route audit, gated on Exp6307 readiness=1

Freeze the target-license policy. Evaluate held games and mechanic strata with
no refit. Keep the feature default off. Require no adequately powered fold to
show harm. This remains a live-agent path audit, not a solve task.

**Deliverable:** `results/experiment_6308_arc_target_validated_route_holdout.json`

### Exp6309: V543 adversarial capstone and reconciliation

Classify every exact declared artifact. Preserve missing, flagged, null,
blocked, skipped, oracle-only, replay-only, and safety-only states. Promote
branches independently. Reconcile OpenSpec, traceability, status, changelog,
architecture, publication tables, and the next-milestone evidence ledger.

**Deliverable:** `results/experiment_6309_v543_adversarial_capstone.json`

## Dependency graph

```text
6297 transition
 ├─► 6298 terminal-evidence preflight ───────────────────────┐
 └─► 6299 source/scope freeze ────────┬──────────────────────┤
                                      │                      │
                                      ▼                      │
                              6300 activation bus             │
                                      ▼                      │
                              6301 integrity audit            │
                               │                 │             │
                     gate=1 ───┘                 └── gate=1 ─┐ │
                               ▼                              │ │
                       6302 state initializer                  │ │
                               ▼ gate=1                        │ │
                       6303 live benchmark                     │ │
                                                              │ │
6299 ─► 6304 online learning ──► 6306 safety audit            │ │
              │ gate=1                                        │ │
              └──────────────► 6305 licensed transfer ◄───────┘ │
                                                                │
6298 + 6299 ─► 6307 ARC target-license canary ─► 6308 holdout  │
                                                    gate=1      │
                                                                │
all exact declared artifacts ───────────────────────────────► 6309
```

All structured gates are conjunctive. A skipped downstream task is terminal
evidence and must not be reinterpreted as a null scientific result.

## Hardware requirements

| Phase | Hardware | Requirement and boundary |
|---|---|---|
| Phase 0 | CPU, local disk, network for source refresh | No model load. Hash all inputs. Exp6298 must not patch `scripts/research_conductor.py`. |
| Phase 1 offline | CPU plus existing Exp5852 corpus; optional one RTX 3090 for small adapter fitting | Exp6300-Exp6302 may use cached, hash-bound embeddings. They must not claim fresh model inference. |
| Phase 1 live | Dual RTX 3090 GPUs | Exp6303 loads the three mandated GGUF families sequentially or under measured safe placement. Require CUDA/offload and memory receipts before model construction. |
| Phase 2 | CPU or one RTX 3090 for small initializer updates | Base GGUF weights remain frozen. The learned factor/state objective must retain a CPU/GPU path and an explicit factor decomposition suitable for future Ising/THRML compilation. |
| Phase 3 | Dual RTX 3090 GPUs for matched ARC generation; CPU for replay and capstone | Bound every call, avoid heavy concurrent model work, and record the known intermittent server/reaper boundary. No hidden-game source access. |

No FPGA or TSU task is scheduled. KV260 and PolarFire have no new workload
receipt, GateMate remains physically blocked at IDCODE, and Carnot has no
authenticated Extropic device or simulator route. V543 may document factor
compatibility but may make no board, latency, power, speed, or availability
claim. A future hardware experiment requires a new dated physical or
authenticated receipt.

## Explicit exclusions

- No generated-answer transport, parser, grammar, or stop-token retry.
- No external generated-text EBM, uPRM, EBRM, or Phase-D scorer rerun.
- No MMLU-Pro final-state hidden-probe retry; the activation bus uses a changed
  representation and exact configuration domain.
- No KAN replacement or mode-jump sampler extension.
- No LLM weight fine-tuning, GRPO, live LoRA, or verifier-as-reward training.
- No ARC public-level solve target, hidden-source read, offline ground-truth
  BFS, per-game adapter, or registry credit.
- No hardware speed, power, energy, or availability claim without a new
  physical or authenticated route.

## Milestone success condition

V543 succeeds scientifically even if every value gate is null, provided the
artifacts are terminal and honest. A positive architecture result requires:

1. the shared activation bus passes every independent shortcut and identity
   control;
2. the shared-state initializer improves exact-valid refinement work on held
   folds without accuracy harm and without crediting the exact oracle;
3. reference-anchored online learning improves future chronological events,
   retains old capability, survives safety audit, and rolls back exactly; and
4. any cross-family or ARC route is licensed by target-side evidence before it
   affects a decision.

The capstone must keep these branches independent. No aggregate score may hide
a failed model, task family, safety stratum, or ARC fold.

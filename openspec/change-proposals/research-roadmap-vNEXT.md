# Carnot Research Roadmap vNEXT: Error-Structured Constraint Compilation and Dependency-Fenced Learning

**Created:** 2026-09-04  
**Milestone:** 2026.09.610  
**Status:** Planned; activates after milestone 2026.09.609 closes  
**Supersedes:** milestone 2026.09.609, experiments exp6953-exp6964  
**Task contract:** exactly 14 tasks, exp6965-exp6978, in the order below  
**Research basis:** `research-program.md` and the V610 section of
`research-references.md`

## What Milestone 2026.09.609 Proved

Milestone 2026.09.609 completed all 12 task slots. It produced exact fixtures,
live local-model proposals, solver certificates, a factor-energy canary, a
sealed event sequence, and terminal audits. It did not prove useful constraint
mapping, useful energy selection, or continuous self-learning.

| Evidence | Result | V610 consequence |
|---|---|---|
| Roadmap contract | The Markdown and YAML both held 12 tasks, exp6953-exp6964, in the same order. The advisory audit conformed. | Keep an advisory contract audit. Do not gate science on it. |
| Live mapping | All three required GGUF families produced 162 raw mapping proposals. Only 10 were exact. Qwen had 54 parse failures. The full bank had 79 parse and 37 schema failures. | Diagnose every failure. Optimize prompts on calibration rows before one held-out test. |
| Exact authority | Z3 and bounded enumeration certified mappings with zero false accepts. Model confidence had AUROC 0.515 and the top-confidence error rate was 0.88. | Preserve exact certification. Do not use confidence or self-report as authority. |
| Convex energy | The canary completed but failed Jensen and finite-difference checks. Its short duration also triggered an adversarial flag. | Change the mechanism to a hard constraint-affine function. Require feasibility and real runtime receipts before selection. |
| Candidate selection | All 54 candidate groups had zero oracle headroom. Every selection arm therefore tied. | Measure candidate headroom before any new energy-selection task runs. |
| Continuous learning | The sealed event sequence conformed. The live task blocked when the Qwen GGUF raised a `no_memory` load failure. Its audit gate-blocked. | Repair and prove the model load envelope first. Then change the learning rule to verifier-grounded group advantage with dependency fences. |
| ARC live path | V609 contained no ARC task. Separately, the post-refit `n_ctx=98304` live run emitted a complete engine, but the engine appears to memorize shown delta rows and has not received a held-out score. | Add one bounded ARC induction-quality audit. Make no solve claim. |
| Capstone | The capstone classified the milestone as disqualified because evidence was null, blocked, or flagged. | Keep the next capstone ungated and class-preserving. |

## Three Largest Gaps to the PRD Vision

### Gap 1: Exact language-to-constraint mapping remains too weak

Carnot can certify a proposed mapping. It cannot yet obtain reliable mappings
from the current local flagship models. A 6.17% exact rate is not a usable
front end for PRD FR12.

V610 uses ESPO's three-stage shape. It clusters every exact failure, creates
four independently biased prompt policies, and uses bootstrap stability on
calibration rows. It then tests one selected policy on sealed held-out pairs.
Instruction duplication is a control, not the main method.

### Gap 2: Learned energy has no valid decision test

V609 had two separate blockers. Its convex function failed its own shape tests.
Its selection bank also had no candidate headroom. An energy score cannot show
causal value when every candidate has the same exact outcome.

V610 changes the function to a CAffNet-inspired hard constraint-affine head.
It trains only on solver-native structural features and calibration labels. It
must satisfy its architecture constraints on every row. A later selection task
runs only when the held-out prompt experiment exposes at least six candidate
groups with real headroom.

### Gap 3: Continuous self-learning has no executable prospective receipt

The external event sequence exists, but the V609 runtime failed before the
learning arms ran. Earlier milestones also found harmful writes and null
prospective utility. This leaves PRD FR11 open.

V610 first proves all three mandated GGUFs can load, generate, close, and
release memory. It then keeps weights frozen and learns only an external prompt
policy. Exact group advantage controls each update. PlanFence-style dependency
hashes prevent a fresh store from authorizing a stale policy. The experiment
uses chronological events, hard resets, poison tests, rollback, and a cold
audit.

## Research Inputs Added Before Design

- ESPO, arXiv:2609.04197, supplies diagnose, diversify, and bootstrap-select.
- Compile by Training, arXiv:2609.04199, motivates a small reusable local
  constraint function. V610 does not fine-tune the flagship GGUFs.
- FlowBalance, arXiv:2609.03241, supplies verifier-grounded group advantage and
  a minimum-change policy update.
- PlanFence, arXiv:2609.03340, supplies dependency-scoped policy validation.
- CAffNet, arXiv:2605.24437, supplies hard constraint-affine layers.
- Instruction Duplication, arXiv:2609.04024, supplies a cheap formatting
  control with known semantic limits.
- LEAP, arXiv:2609.03294, motivates separate latent or prefix fit and decoded
  held-out state agreement in the ARC audit.
- VFScale, Kona, Extropic Z1, and new GitHub repositories remain watch items.
  They do not provide a matching local, independently certified runtime.

## V610 Architecture

```text
                    +------------------------------+
                    | Exp6965 advisory contract     |
                    | No science task gates on it   |
                    +------------------------------+

  Three cached flagship GGUFs                 V609 exact artifacts
              |                                      |
              v                                      v
  Exp6966 load-envelope canary            Exp6967 error/headroom fixture
       ready_score                                  |
              |                         +------------+-------------+
              |                         |                          |
              |                         v                          v
              |             Exp6969 four-strategy bank   Exp6973 hard-affine
              |                         |                function canary
              |                         v                          |
              +--------------> Exp6970 bootstrap policy           |
                                        |                         |
                                        v                         |
                              Exp6971 held-out prompt A/B          |
                                |       |                          |
                                |       +---- headroom >= 6 -------+
                                v                                  v
                        Exp6972 cold audit              Exp6974 energy selection
                                                                    |
                                                                    v
                                                        Exp6975 cold audit

  Post-refit live ARC engine ---> Exp6968 held-out induction audit
                                  no level or solve claim

  Exp6966 runtime + Exp6967 chronological stream + Exp6970 policy
                                  |
                                  v
                    Exp6976 FlowBalance + PlanFence learning
                                  |
                                  v
                         Exp6977 cold safety audit

                    Exp6978 ungated capstone
```

The exact solver is the final correctness authority. Prompt policies, compact
constraint functions, and external memory are learned signals. They cannot
certify their own outputs. Raw model output is durable before parsing.

## Phase A: Contract, Runtime, Exact Data, and ARC Audit

### Exp6965: V610 advisory execution-contract and retirement audit

- Compare this document with the active copy of `research-roadmap-next.yaml`.
- Require exactly 14 tasks, exp6965-exp6978, with matching titles,
  deliverables, order, gates, producer fields, prior failures, and prompt
  endings.
- Check all model contracts, retired scopes, and ARC solve rules.
- Keep the audit advisory. No science task gates on it.
- Deliverable: `results/experiment_6965_v610_contract_advisory.json`

### Exp6966: Three-family GGUF load-envelope and teardown canary

- Reproduce the V609 Qwen `no_memory` failure before changing configuration.
- Resolve all three mandated cached GGUFs with `cached_sota_pair()`.
- Load one model process at a time across the dual RTX 3090 host.
- Run a vocabulary probe and one short live generation for each family.
- Prove process exit and VRAM release before the next model starts.
- Make the smallest loader or configuration repair outside the conductor when
  the failure is reproducible.
- Produce `gguf_runtime_ready_score` as the narrow downstream gate.
- Deliverable: `results/experiment_6966_gguf_load_envelope_canary.json`

### Exp6967: Certified mapping-error and headroom fixture

- Recompute all V609 parse, schema, domain, and objective errors from raw rows.
- Cluster all 162 proposals without using model confidence as a label.
- Select unused exact pairs from the V609 fixture. Freeze 18 calibration pairs,
  18 held-out pairs, and 24 chronological events.
- Preserve family balance, positive and hard-negative balance, split hashes,
  and exact witnesses.
- Produce `error_fixture_ready_score` and
  `chronological_event_stream_ready_score`.
- Deliverable: `results/experiment_6967_certified_error_headroom_fixture.json`

### Exp6968: ARC post-refit induction held-out quality audit

- Find the completed post-refit `n_ctx=98304` r11l run and its emitted engine.
- Use the existing induction-quality scorer in a fresh process.
- Compare shown-prefix accuracy, held-out accuracy, changing-transition
  accuracy, identity, constant-delta, and memorization controls.
- Record every transition row and all source hashes.
- Make no game-level or level-level solve claim. Do not update the solve
  registry.
- Deliverable: `results/experiment_6968_arc_post_refit_induction_audit.json`

## Phase B: Error-Structured Prompt Compilation

### Exp6969: Three-family error-structured prompt candidate bank

- Gate on `exp6966.gguf_runtime_ready_score == 1` and
  `exp6967.error_fixture_ready_score == 1`.
- Run the 18 calibration pairs through the direct V609 prompt and four fixed,
  independently biased ESPO-style prompt policies.
- Use all three mandated GGUF families. The fixed budget is 270 live attempts.
- Preserve raw output before parsing. Do not reveal exact labels, solver
  messages, future rows, or outcomes from another policy.
- Produce `prompt_candidate_bank_complete_score`.
- Deliverable: `results/experiment_6969_error_structured_prompt_bank.json`

### Exp6970: Bootstrap-stable prompt policy selection

- Gate on `exp6969.prompt_candidate_bank_complete_score == 1`.
- Certify only the calibration outputs.
- Compare the four candidate policies with the direct baseline.
- Select a non-baseline policy only when it improves calibration exactness and
  wins at least 80% of preregistered bootstrap resamples.
- Keep model-family and formulation-family rows.
- Produce `stable_prompt_policy_ready_score`. This is calibration evidence and
  is circular, not a held-out positive result.
- Deliverable: `results/experiment_6970_bootstrap_prompt_policy_selection.json`

### Exp6971: Three-family held-out prompt-policy causal A/B

- Gate on `exp6966.gguf_runtime_ready_score == 1` and
  `exp6970.stable_prompt_policy_ready_score == 1`.
- Run 18 sealed held-out pairs with all three mandated GGUF families.
- Compare the direct V609 prompt, an instruction-duplication control, and the
  selected error-structured policy. The fixed budget is 162 live attempts.
- Certify only after every raw output is durable.
- Report parse, schema, exact mapping, premature commitment, paired deltas,
  and `heldout_headroom_group_count`.
- Deliverable: `results/experiment_6971_heldout_prompt_policy_ab.json`

### Exp6972: Fresh-process prompt-policy audit

- Gate on `exp6971.prompt_ab_run_complete_score == 1`.
- Recompute all prompt arms and headlines from raw output.
- Verify split isolation, prompt hashes, output budgets, model identities,
  parser versions, exact witnesses, and tie rules.
- Never upgrade the upstream verdict.
- Deliverable: `results/experiment_6972_prompt_policy_cold_audit.json`

## Phase C: Hard-Feasible Constraint Energy and Selection

### Exp6973: Hard constraint-affine local function canary

- Gate on `exp6967.error_fixture_ready_score == 1` and
  `exp6969.prompt_candidate_bank_complete_score == 1`.
- Train only on V609 and V610 calibration rows.
- Use solver-native structural features. Do not use raw text embeddings,
  hidden states, held-out labels, or exact same-row outcomes as input.
- Compare a CAffNet-inspired hard-affine head with a parameter-matched MLP, a
  linear score, and shuffled labels across fixed seeds.
- Prove aggregation equality, renaming symmetry, finite outputs, checkpoint
  reload, and sufficient measured duration.
- Produce `constraint_affine_feasibility_score`.
- Deliverable: `results/experiment_6973_constraint_affine_function_canary.json`

### Exp6974: Headroom-qualified constraint-affine energy selection

- Gate on `exp6971.prompt_ab_run_complete_score == 1`,
  `exp6971.heldout_headroom_group_count >= 6`, and
  `exp6973.constraint_affine_feasibility_score == 1`.
- Score the three frozen prompt candidates for each held-out model/pair group.
- Compare hard-affine energy, unconstrained MLP, linear score, syntax,
  likelihood, confidence, shuffled energy, fixed order, and an oracle upper
  bound.
- The learned selectors cannot see exact labels until after each choice.
- Require a paired gain over the strongest non-oracle baseline and at least 20%
  oracle-headroom capture for a positive result.
- Deliverable:
  `results/experiment_6974_headroom_projected_energy_selection.json`

### Exp6975: Fresh-process projected-selection audit

- Gate on `exp6974.projected_selection_run_complete_score == 1`.
- Reload the checkpoint and recompute every arm from immutable rows.
- Verify train, calibration, and held-out separation.
- Check feature, label, order, tie, budget, and exact-outcome leakage.
- Never upgrade the upstream result.
- Deliverable: `results/experiment_6975_projected_selection_cold_audit.json`

## Phase D: Dependency-Fenced Self-Learning and Reconciliation

### Exp6976: Verifier-grounded dependency-fenced continuous self-learning

- Gate on `exp6966.gguf_runtime_ready_score == 1`,
  `exp6967.chronological_event_stream_ready_score == 1`, and
  `exp6970.stable_prompt_policy_ready_score == 1`.
- Use Qwen3.6-35B-A3B and Gemma-4-26B-A4B in separate fresh processes.
- Compare no memory, fixed FIFO, unfenced group-advantage learning, and
  FlowBalance plus PlanFence.
- Update only after exact event certification. Reverse negative advantage.
  Make no update when the group has no outcome preference.
- Attach source certificate hashes to every policy update. Replan once or block
  when a dependency changed.
- Keep weights frozen. Test stale dependencies, poison, retention, tombstones,
  rollback, restart, store size, and future leakage.
- Produce `self_learning_run_complete_score`.
- Deliverable: `results/experiment_6976_dependency_fenced_self_learning.json`

### Exp6977: Fresh-process dependency-fenced learning safety audit

- Gate on `exp6976.self_learning_run_complete_score == 1`.
- Recompute prospective gains, group advantage, policy updates, dependency
  checks, stale-action blocks, rollbacks, and model immutability.
- Replay stores from disk in a fresh process.
- Verify that no event or future outcome affected its own prompt or update.
- Never upgrade the upstream result.
- Deliverable: `results/experiment_6977_self_learning_cold_audit.json`

### Exp6978: V610 independent capstone and V611 handoff

- Stay ungated. Classify missing, blocked, null, partial, circular, positive,
  and disqualified evidence without retries caused by external absence.
- Recheck the exact 14-task document and YAML contract.
- Recompute every comparison from per-unit rows.
- Separate exact authority, calibration circularity, ARC non-solve evidence,
  learned signals, and hardware conformance.
- Produce the three largest remaining gaps and a bounded V611 handoff.
- Deliverable: `results/experiment_6978_v610_capstone.json`

## Dependency Graph

```text
exp6965  advisory contract audit
exp6968  independent ARC audit

exp6966  GGUF runtime
  ├─> exp6969  prompt bank <─ exp6967 exact fixture
  │      ├─> exp6970  stable prompt policy
  │      │      ├─> exp6971  held-out prompt A/B
  │      │      │      ├─> exp6972  cold prompt audit
  │      │      │      └─> exp6974  energy selection <─ exp6973
  │      │      │                                 └─> exp6975 cold audit
  │      │      └─> exp6976  continuous learning <─ exp6967 stream
  │      │                                      └─> exp6977 cold audit
  │      └─> exp6973  hard-affine function <─ exp6967
  └──────────> exp6976

exp6978  ungated capstone
```

No task depends on exp6965. No task references a retired upstream experiment.
Every structured gate names a producer in this milestone. Every producer names
the same field in its own required artifact fields.

## Exact Task Contract

| Order | ID | Title | Deliverable | Structured gate |
|---:|---|---|---|---|
| 1 | exp6965-v610-contract-advisory | V610 advisory execution-contract and retirement audit | `results/experiment_6965_v610_contract_advisory.json` | none |
| 2 | exp6966-gguf-load-envelope-canary | Three-family GGUF load-envelope and teardown canary | `results/experiment_6966_gguf_load_envelope_canary.json` | none |
| 3 | exp6967-certified-error-headroom-fixture | Certified mapping-error and headroom fixture | `results/experiment_6967_certified_error_headroom_fixture.json` | none |
| 4 | exp6968-arc-post-refit-induction-audit | ARC post-refit induction held-out quality audit | `results/experiment_6968_arc_post_refit_induction_audit.json` | none |
| 5 | exp6969-error-structured-prompt-bank | Three-family error-structured prompt candidate bank | `results/experiment_6969_error_structured_prompt_bank.json` | exp6966 `gguf_runtime_ready_score == 1`; exp6967 `error_fixture_ready_score == 1` |
| 6 | exp6970-bootstrap-prompt-policy-selection | Bootstrap-stable prompt policy selection | `results/experiment_6970_bootstrap_prompt_policy_selection.json` | exp6969 `prompt_candidate_bank_complete_score == 1` |
| 7 | exp6971-heldout-prompt-policy-ab | Three-family held-out prompt-policy causal A/B | `results/experiment_6971_heldout_prompt_policy_ab.json` | exp6966 `gguf_runtime_ready_score == 1`; exp6970 `stable_prompt_policy_ready_score == 1` |
| 8 | exp6972-prompt-policy-cold-audit | Fresh-process prompt-policy audit | `results/experiment_6972_prompt_policy_cold_audit.json` | exp6971 `prompt_ab_run_complete_score == 1` |
| 9 | exp6973-constraint-affine-function-canary | Hard constraint-affine local function canary | `results/experiment_6973_constraint_affine_function_canary.json` | exp6967 `error_fixture_ready_score == 1`; exp6969 `prompt_candidate_bank_complete_score == 1` |
| 10 | exp6974-headroom-projected-energy-selection | Headroom-qualified constraint-affine energy selection | `results/experiment_6974_headroom_projected_energy_selection.json` | exp6971 `prompt_ab_run_complete_score == 1`; exp6971 `heldout_headroom_group_count >= 6`; exp6973 `constraint_affine_feasibility_score == 1` |
| 11 | exp6975-projected-selection-cold-audit | Fresh-process projected-selection audit | `results/experiment_6975_projected_selection_cold_audit.json` | exp6974 `projected_selection_run_complete_score == 1` |
| 12 | exp6976-dependency-fenced-self-learning | Verifier-grounded dependency-fenced continuous self-learning | `results/experiment_6976_dependency_fenced_self_learning.json` | exp6966 `gguf_runtime_ready_score == 1`; exp6967 `chronological_event_stream_ready_score == 1`; exp6970 `stable_prompt_policy_ready_score == 1` |
| 13 | exp6977-self-learning-cold-audit | Fresh-process dependency-fenced learning safety audit | `results/experiment_6977_self_learning_cold_audit.json` | exp6976 `self_learning_run_complete_score == 1` |
| 14 | exp6978-v610-capstone | V610 independent capstone and V611 handoff | `results/experiment_6978_v610_capstone.json` | none |

This table is the milestone contract. `research-roadmap-next.yaml` must contain
the same 14 IDs, titles, deliverables, order, and gates.

## Hardware and Runtime Requirements

| Tasks | Substrate | Estimated time | Requirement |
|---|---|---:|---|
| exp6965, exp6967, exp6968 | CPU | 1-3 hours each | YAML and Markdown parsers, Z3, bounded enumeration, ARC replay assets |
| exp6966 | Dual RTX 3090 | up to 4 hours | All three cached mandated GGUFs, llama.cpp, task-owned GPU and teardown receipts |
| exp6969 | Dual RTX 3090, sequential family ownership | up to 12 hours | All three mandated GGUFs, batched live generation, checkpoint after every pair/model block |
| exp6970, exp6972 | CPU | 2-4 hours each | Exact certifier, bootstrap reducer, immutable raw rows |
| exp6971 | Dual RTX 3090, sequential family ownership | up to 12 hours | All three mandated GGUFs, sealed prompts, 162 live attempts |
| exp6973-exp6975 | CPU; CUDA optional for compact training only | 3-6 hours each | PyTorch or NumPy, saved compact checkpoint, exact row replay |
| exp6976 | Dual RTX 3090, sequential arm and family processes | up to 12 hours | Qwen3.6 and Gemma-4-26B-A4B cached GGUFs, transactional policy store |
| exp6977-exp6978 | CPU | 2-4 hours each | Read-only artifact and store replay |

Every GGUF task uses the embedded GGUF tokenizer through llama.cpp. It must not
call `AutoTokenizer.from_pretrained()` on a GGUF repository ID. Legacy small
models may appear only in labeled CPU smoke rows. They cannot supply a headline
result. A missing flagship cache produces a blocked artifact.

The AMD APU, NPU, KV260, GateMate, PolarFire, and Extropic hardware stay outside
the blocking graph. V610 makes no FPGA, TSU, throughput, power, or energy claim.

## Milestone Exit Criteria

- The document and YAML match on all 14 task contracts.
- All gates resolve to fields declared by producers in this roadmap.
- Raw model output is durable before parsing or certification.
- The prompt result uses all three mandated GGUF families and sealed held-out
  pairs.
- An energy-selection run occurs only with at least six real headroom groups
  and a hard-feasible compact checkpoint.
- The ARC task emits held-out transition evidence and no solve claim.
- The self-learning task uses immutable exact certificates, chronological
  order, hard resets, dependency fences, rollback, and frozen weights.
- Every comparative task emits per-unit rows.
- Every task emits `verdict_class`, a class-consistent `honest_verdict`,
  provenance, duration, hashes, random seed, checksum, and
  `gate_check_summary` when blocked.
- The ungated capstone classifies every task without laundering missing or
  circular evidence.

## Explicit Deferrals

- Grammar-constrained generated-answer transport and finite answer IDs.
- Schema-reprompt retries with zero exact semantic gain.
- Free-text external reward models and hidden-state scoring.
- Same-model self-verification as final authority.
- Flagship GGUF weight updates, LoRA, GRPO, or teacher distillation.
- ARC game-level or level-level solve claims.
- Production default-on prompt, energy, or memory policy changes.
- FPGA, TSU, or remote hardware execution and performance claims.

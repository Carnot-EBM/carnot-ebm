# Carnot Research Roadmap vNEXT: Grounded Learning and Internal Energy

**Created:** 2026-09-03  
**Milestone:** 2026.09.607  
**Status:** Planned  
**Supersedes:** milestone 2026.09.606, experiments 6923-6926  
**Task contract:** exactly 14 tasks, Exp6927 through Exp6940, in the order below  
**Execution manifest:** research-roadmap-next.yaml

## What Milestone 2026.09.606 Proved

| Result | Experiment | Finding |
|---|---:|---|
| Lifecycle evidence contract | 6923 | The runtime audit still blocked on roadmap lifecycle state. A science branch must not depend on this class of contract. |
| Task-owned runtime receipt helper | 6924 | The reusable adoption path is ready, but its result was null because no live SOTA GPU receipt proved adoption. |
| Execution-time SOTA refresh | 6925 | All three required local GGUF families remain the mandated baseline. The public ISM runner is hosted-API-bound. Only its schema and update patterns transfer directly. |
| Span-first exact fixture | 6926 | Fifteen family-label cells and 69 rows now preserve byte spans, direction, aliases, and exact labels without disagreement. The result is circular-positive because it proves fixture integrity, not live model acquisition. |

Milestone 2026.09.606 did not run the ten science tasks described by its stale
design document. Its active YAML contained four tasks only. V607 treats Exp6926
as the last executed ID. This document and the next-roadmap YAML carry the same
14-task contract.

## Three Largest Gaps to the PRD

### Gap 1: Live language still does not become trusted constraints

The exact span fixture exists. Carnot has not shown that each required SOTA
model can copy grounded spans, preserve direction, and form exact held-out
relations at useful accuracy. This blocks FR01, FR02, and the natural-language
side of the Kona-style interface.

### Gap 2: Continuous self-learning has no clean prospective utility result

Carnot has persistent stores and certified update machinery. Recent sealed
tests found harmful writes, weak abstention, or no utility. FR11 needs a clean
ordered experiment with immutable exact receipts, delayed writes, hard resets,
retention, rollback, and a no-memory control.

### Gap 3: Energy has not improved a non-saturated SOTA decision

Exact-guided generation saturated in Exp6920. Earlier hidden-state probes tied
or lost to simple controls. Carnot still lacks a non-circular result in which a
structural energy or internal-state verifier improves candidate choice under a
matched budget. This blocks FR03, FR06, FR08, FR09, and the Phase-D bridge.

## Research Basis Added for V607

- HSRM, arXiv:2608.30841, reads hidden states at reasoning-step boundaries and
  trains a compact candidate ranker from outcome labels.
- Sparse Reward Subsystem, arXiv:2602.00986, reports that reward signals can
  concentrate in a small neuron subset.
- NSVIF, ICLR 2026 OpenReview RZGs4OAH6g, supports a hybrid span-to-constraint
  path with exact symbolic checks.
- Sampling for Quality, arXiv:2604.16453, motivates matched-budget population
  resampling with soft rewards.
- Extropic Z1 remains a future-access target. Kona remains a non-executable
  architecture comparator. Neither belongs in the blocking graph.

The dated source notes and compatibility limits are in
research-references.md under the V607 planner refresh.

## V607 Architecture

~~~text
                    advisory roots
       ┌──────────────────┐  ┌────────────────────┐
       │ literature delta │  │ SOTA runtime proof │
       │      Exp6927     │  │      Exp6928       │
       └──────────────────┘  └────────────────────┘

User text + hidden reference labels
                 │
                 ▼
       ┌────────────────────┐
       │ span-first SOTA     │
       │ acquisition Exp6929│
       └─────────┬──────────┘
                 │ exact held-out qualification
                 ▼
       ┌────────────────────┐
       │ relation bank      │
       │      Exp6930       │
       └────────────────────┘

Exact strategy episodes                    Fresh candidate groups
          │                                          │
          ▼                                          ▼
┌────────────────────┐                     ┌────────────────────┐
│ fixture Exp6931    │                     │ headroom Exp6934   │
└─────────┬──────────┘                     └─────────┬──────────┘
          ▼                                          ▼
┌────────────────────┐                     ┌────────────────────┐
│ episodic memory    │                     │ energy canary      │
│ Exp6932            │                     │ Exp6935            │
└─────────┬──────────┘                     └─────────┬──────────┘
          ▼                                          ▼
┌────────────────────┐                     ┌────────────────────┐
│ cold audit Exp6933 │                     │ SMC Exp6936        │
└────────────────────┘                     └────────────────────┘

Mandated local GGUF runtime
          │
          ▼
┌────────────────────┐       ready only       ┌────────────────────┐
│ state reachability │───────────────────────▶│ compact ranker     │
│ Exp6937            │                        │ Exp6938            │
└────────────────────┘                        └────────────────────┘

Live ARC attempts ─────────▶ induction emission audit Exp6939

All available receipts ───▶ ungated capstone Exp6940 ───▶ V608 handoff
~~~

Exact checkers remain the update and release authority. Learned scores may
rank candidates. They may not certify their own labels. Outcome labels stay out
of model inputs. Every comparison emits per-unit rows.

## Phase A: Grounded Inputs and Runtime Evidence

### Exp6927: V607 execution-time literature delta and compatibility map

Recheck the named source families at execution time. Append only verified
deltas. Record incompatible code paths and explicit no-update results. This is
the required SOTA-ingestion slot. It is advisory and ungated.

### Exp6928: Dual-GPU SOTA runtime receipt qualification

Adopt the Exp6924 helper in one bounded real model run. Exercise all three
required GGUF families sequentially on the dual RTX 3090 host. Prove process
ownership, device identity, offload, phase intervals, cache state, teardown,
and fresh-process replay. This is infrastructure slot 1 of 2. It is advisory
and ungated.

### Exp6929: Three-family live span-first relation acquisition

Use the Exp6926 fixture contract on held-out prompts. Run Qwen3.6-35B-A3B,
Gemma-4-31B-it, and Gemma-4-26B-A4B-it. Require verbatim spans before a typed
directed relation. Preserve every malformed and failed row. Do not reprompt
with hidden labels.

### Exp6930: Independent exact relation qualification

This task runs only when Exp6929 emits
span_acquisition_bank_ready_score == 1. A fresh process checks offsets,
aliases, direction, schema, and hidden reference labels. The task reports
coverage and accuracy by model and relation family. It may not infer missing
rows or repair model outputs.

## Phase B: Continuous Self-Learning

### Exp6931: Exact strategy-episode and delayed-write fixture

Create ordered episodes with exact pre-action state, action, outcome,
constraint receipt, and reusable strategy fields. Include useful, irrelevant,
contradictory, poisoned, and rollback cases. Freeze train, validation, and
retention order before any model run.

### Exp6932: Prospective selective episodic self-learning

This task runs only when Exp6931 emits
strategy_episode_fixture_ready_score == 1. Compare three complete systems:
no memory, read-only retrieval, and exact-admitted delayed writes. Run all
three required SOTA model families. Reset processes between arms. Admit writes
only after exact outcomes. Measure next-event utility, abstention, retention,
rollback, and contamination. This is the milestone's FR11 continuous
self-learning task.

### Exp6933: Fresh-process episodic-memory safety audit

This task runs only when Exp6932 emits
episodic_memory_run_complete_score == 1. Recompute every headline from raw
rows in a fresh process. Test restart, provenance, delayed-copy poison,
tombstone reappearance, old-family retention, and rollback. Audit completion,
not a positive effect, is the gate.

## Phase C: Non-Saturated Energy and Hidden-State Verification

### Exp6934: Three-family non-saturated proposal bank

Generate a frozen bank with all three required SOTA families. Select held-out
items only when the baseline has real headroom. Keep groups with both correct
and incorrect candidates. Save token budgets, seeds, runner receipts, exact
labels, and every rejected group. Do not train on the evaluation groups.

### Exp6935: Structural-energy headroom admission canary

This task runs only when Exp6934 emits
headroom_proposal_bank_ready_score == 1. Score typed candidate structures with
Carnot energy. Compare top-1 selection with random, majority, likelihood, and
simple exact-feature controls. Admit population guidance only when the paired
held-out improvement is positive with a confidence interval above zero and the
result survives permutation and label-leak checks.

### Exp6936: Population SMC with exact structural energy

This task runs only when Exp6935 emits
population_guidance_admitted_score == 1. Implement resample-move population
search over the same frozen item distribution. Compare guided SMC, unguided
SMC, and fixed-budget Best-of-N. Match model calls, candidate count, token
budget, and stopping rules. Exact outcomes evaluate candidates but never enter
the proposal context.

### Exp6937: GGUF hidden-state reachability and sparse-signal canary

Test Qwen3.6-35B-A3B and Gemma-4-31B-it through the local GGUF runner. Record
which token, layer, chunk, cache, embedding, and logit states are actually
available without re-decoding text. Compare full accessible state, sparse
coordinates, final embedding, logits, and content-blind controls on exact
candidate labels. Unsupported intermediate layers are a valid blocked result.
This is a new HSRM compatibility question, not a rerun of the retired MMLU-Pro
probe path.

### Exp6938: Compact hidden-state candidate ranker

This task runs only when Exp6937 emits
hidden_state_capture_ready_score == 1. Train a small tie-safe ranker on cached
states from disjoint candidate groups. Compare it with likelihood,
final-embedding, sparse-coordinate, and structural-energy controls. Report
Best-of-N accuracy, ranking AUROC, calibration, latency, and paired intervals.
Do not use a text verifier or model judge as ground truth.

## Phase D: Live-Path Audit and Synthesis

### Exp6939: ARC post-fix induction emission and held-out accuracy audit

Use fresh live attempts on r11l after the 98,304-token context repair. Measure
whether the agent emits a usable induced transition model and whether that
model predicts held-out transitions. Compare the required Qwen3.6 flagship
with the current production control. This task does not seek or claim a new
game-level solve. It does not repeat the zero-delta tool/no-tool credit test.

### Exp6940: V607 independent capstone and V608 handoff

Aggregate available branch receipts without gating on a positive result. Skip
missing, blocked, disqualified, and adversarially flagged evidence from positive
headlines. Recompute task count, ID order, gate fields, model coverage,
self-learning status, verifier circularity, and hardware provenance. This is
infrastructure slot 2 of 2 and the only handoff task.

## Exact Task Contract

| Order | Task ID | Title | Deliverable | Structured gate |
|---:|---|---|---|---|
| 1 | exp6927-v607-literature-delta | V607 execution-time literature delta and compatibility map | results/experiment_6927_v607_literature_delta.json | none |
| 2 | exp6928-sota-runtime-receipt-qualification | Dual-GPU SOTA runtime receipt qualification | results/experiment_6928_sota_runtime_receipt_qualification.json | none |
| 3 | exp6929-three-family-span-acquisition | Three-family live span-first relation acquisition | results/experiment_6929_three_family_span_acquisition.json | none |
| 4 | exp6930-independent-relation-qualification | Independent exact relation qualification | results/experiment_6930_independent_relation_qualification.json | exp6929-three-family-span-acquisition.span_acquisition_bank_ready_score == 1 |
| 5 | exp6931-strategy-episode-fixture | Exact strategy-episode and delayed-write fixture | results/experiment_6931_strategy_episode_fixture.json | none |
| 6 | exp6932-selective-episodic-memory | Prospective selective episodic self-learning | results/experiment_6932_selective_episodic_memory.json | exp6931-strategy-episode-fixture.strategy_episode_fixture_ready_score == 1 |
| 7 | exp6933-memory-cold-audit | Fresh-process episodic-memory safety audit | results/experiment_6933_memory_cold_audit.json | exp6932-selective-episodic-memory.episodic_memory_run_complete_score == 1 |
| 8 | exp6934-headroom-proposal-bank | Three-family non-saturated proposal bank | results/experiment_6934_headroom_proposal_bank.json | none |
| 9 | exp6935-structural-energy-headroom-canary | Structural-energy headroom admission canary | results/experiment_6935_structural_energy_headroom_canary.json | exp6934-headroom-proposal-bank.headroom_proposal_bank_ready_score == 1 |
| 10 | exp6936-population-smc-guidance | Population SMC with exact structural energy | results/experiment_6936_population_smc_guidance.json | exp6935-structural-energy-headroom-canary.population_guidance_admitted_score == 1 |
| 11 | exp6937-gguf-hidden-state-reachability | GGUF hidden-state reachability and sparse-signal canary | results/experiment_6937_gguf_hidden_state_reachability.json | none |
| 12 | exp6938-compact-hidden-state-ranker | Compact hidden-state candidate ranker | results/experiment_6938_compact_hidden_state_ranker.json | exp6937-gguf-hidden-state-reachability.hidden_state_capture_ready_score == 1 |
| 13 | exp6939-arc-induction-emission-audit | ARC post-fix induction emission and held-out accuracy audit | results/experiment_6939_arc_induction_emission_audit.json | none |
| 14 | exp6940-v607-capstone | V607 independent capstone and V608 handoff | results/experiment_6940_v607_capstone.json | none |

## Dependency Graph

~~~text
Exp6927                           advisory root
Exp6928                           advisory infrastructure root
Exp6929 ──ready──▶ Exp6930        grounded acquisition
Exp6931 ──ready──▶ Exp6932 ──run-complete──▶ Exp6933
Exp6934 ──ready──▶ Exp6935 ──admitted──▶ Exp6936
Exp6937 ──capture-ready──▶ Exp6938
Exp6939                           independent live ARC audit
Exp6940                           ungated aggregate of available receipts
~~~

No science task gates on Exp6927, Exp6928, or Exp6940. A blocked producer must
write gate_check_summary and its declared ready score. The conductor can then
skip only the direct dependent task.

## Model Contract

Every experiment that invokes an LLM must declare MODEL_SPECS through the
helpers in scripts/experiment_template.py.

| Task | Required headline model set |
|---|---|
| Exp6928 | Qwen3.6-35B-A3B, Gemma-4-31B-it, Gemma-4-26B-A4B-it |
| Exp6929 | Qwen3.6-35B-A3B, Gemma-4-31B-it, Gemma-4-26B-A4B-it |
| Exp6932 | Qwen3.6-35B-A3B, Gemma-4-31B-it, Gemma-4-26B-A4B-it |
| Exp6934 | Qwen3.6-35B-A3B, Gemma-4-31B-it, Gemma-4-26B-A4B-it |
| Exp6936 | Qwen3.6-35B-A3B, Gemma-4-31B-it, Gemma-4-26B-A4B-it |
| Exp6937 | Qwen3.6-35B-A3B and Gemma-4-31B-it |
| Exp6939 | Qwen3.6-35B-A3B plus the current production control |

Legacy small models may run CPU smoke tests. They may not support a headline.
GGUF tokenizer metadata comes from the GGUF or runner. Do not call
AutoTokenizer.from_pretrained() on a GGUF repository ID.

## Acceptance and Reporting Rules

- Every task writes its artifact even when blocked.
- Every artifact declares the closed verdict_class enum.
- Every blocked artifact names the failed check, expected value, and observed
  value in gate_check_summary.
- Every comparative task emits one per-unit row for every model, item, arm,
  seed, and condition used by its headline.
- Every model-bearing task records the selected model, quantization, runner,
  device path, offload, cache state, duration, and teardown.
- Every learned result records split hashes and proves label isolation.
- Every task annotates each required field with a scientific principle.
- A verifier used as its own oracle cannot receive a positive verdict class.
- Exp6939 makes no game-level solve claim, so it has no ARC solve provenance.
- No task may modify scripts/research_conductor.py.

## Hardware Requirements

| Tasks | Hardware | Budget and boundary |
|---|---|---|
| 6927, 6930, 6931, 6933, 6940 | CPU and network where stated | Bounded audits and exact replay. No accelerator claim. |
| 6928, 6929, 6932, 6934, 6936, 6937, 6939 | Dual RTX 3090 CUDA | Run required GGUF models sequentially unless measured memory allows safe concurrency. Record both GPU UUIDs and peak VRAM. |
| 6935, 6938 | CUDA for training or scoring, CPU replay | Use frozen rows. Keep training bounded. Save checkpoints and split hashes. |
| all tasks | Disk | Reuse cached GGUF files. Store large hidden arrays or proposal rows under results/checkpoints/, not in Git. |
| none | KV260, GateMate, PolarFire, XTR-0, or Z1 | Attached boards and future TSUs are outside the blocking graph. Make no hardware speed or power claim. |

## Estimated Execution Budget

| Phase | Tasks | Expected wall time |
|---|---|---:|
| A | 6927-6930 | 12-20 hours |
| B | 6931-6933 | 12-20 hours |
| C | 6934-6938 | 22-36 hours |
| D | 6939-6940 | 8-14 hours |
| Total | 14 tasks | 54-90 hours |

The conductor executes tasks serially. Model caches and proposal checkpoints
must make restarts safe. Wall-time estimates include bounded reruns, not
unlimited retries.

## Explicitly Deferred

- Weight updates and LoRA continual learning remain deferred until external
  exact-memory utility is positive and safe.
- Full EBT or Kona-scale pretraining remains deferred. V607 tests internal
  verification signals, not a foundation-model training claim.
- MMLU-Pro hidden-state probing remains retired. V607 uses exact grouped
  candidates and a new HSRM-style method.
- New ARC game solves remain outside V607. Exp6939 audits the live induction
  path only.
- FPGA or TSU acceleration remains opportunistic. No unavailable device sits
  on a dependency path.

## Completion Contract

V607 is complete when exactly these 14 task IDs have terminal artifacts or
conductor-written blocked artifacts in this order. Exp6940 must report branch
truth without converting a null, blocked, circular, partial, or disqualified
result into a positive claim. Before activation, validate this document against
research-roadmap-next.yaml for exact task count, ID order, titles,
deliverables, and gates.

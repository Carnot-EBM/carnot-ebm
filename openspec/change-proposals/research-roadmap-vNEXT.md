# Research Roadmap vNEXT — Milestone 2026.07.517

**Milestone:** 2026.07.517  
**Title:** Channel-Qualified Continuous Self-Learning and Reachable ARC World Models  
**Status:** Proposed  
**Task range:** Exp5796-Exp5808 (13 experiments)  
**Execution file:** `research-roadmap-next.yaml`  
**Date planned:** 2026-07-22

## Thesis

Milestone `.516` reached the real SOTA models and produced useful negative evidence. The exact
evidence index and sealed constraint fixture are ready, and all three mandated GGUFs ran with
authenticated CUDA offload. The stream was not promotable because Qwen3.6 used all 48 output tokens
without reaching the finite-choice answer on every one of its 360 rows. The two Gemma families were
fully parseable, but the combined parser-failure rate was `0.333333`; only 25 satisfiable-drift rows
were available, below the learning gate. The continuous-self-learning chain therefore did not test
learning.

The ARC admission contract also shipped, but the three-family single-shot panel blocked before
generation. Its preflight demanded GPU-load receipts, fresh hypotheses, and a resume checkpoint
before the code path that creates those artifacts. That is a bootstrap-ordering defect, not evidence
against single-shot world-model induction. Downstream calibration and live E3 influence were not
tested.

Milestone `.517` repairs these two evidence boundaries without relaxing them. It first diagnoses and
qualifies per-model answer channels on small real-GGUF canaries. The Qwen arm explicitly tests the
upstream reasoning/grammar interaction rather than assuming grammar is safe or sufficient. Only a
three-family, parser-complete, exact-label-complete canary may trigger a scaled chronological stream.
That stream then supports validation-gated, versioned constraint skills with future-batch admission,
rollback, endurance, and out-of-distribution audits while every GGUF weight remains immutable.

In parallel, `.517` moves ARC load receipts, checkpoint creation, and hypothesis existence to their
correct place in execution. It runs independent single-shot hypotheses from the three mandated SOTA
families through the already-qualified admission contract, selects only among immutable admissible
hypotheses using agent-owned calibration transitions, and permits one held-out live E3 A/B only if
the selector has a positive lower bound. This is live-path generalization, not a public-game solve.

The milestone makes two falsifiable claims:

1. A bounded, model-specific transport layer can recover exact finite-choice responses from all
   three mandated local SOTA GGUFs without hiding reasoning truncation, empty content, unsafe schema
   control, or exact-answer errors.
2. Frozen-model sidecar learning and immutable ARC world-model selection can improve sealed future
   behavior under exact/live authority, with rollback and abstention when evidence is insufficient.

## What milestone 2026.07.516 proved

| Branch | Terminal evidence | Consequence for `.517` |
|---|---|---|
| Transition and reporting | Exp5782 archived `.515` by exact declared identity. Exp5784 qualified the exact-deliverable index and Exp5795 reconciled all `.516` outcomes. | Reuse the qualified identity path. Do not reopen evidence-index repair or infer task state from numeric-prefix aliases. |
| Source currency | Exp5783 completed with no accepted post-marker delta. | Execute one bounded V517 refresh; zero accepted findings remains a complete result. |
| Exact constraint fixture | Exp5785 shipped the sealed hardness/surface fixture with exact labels and parser controls ready. | Reuse it byte-for-byte. No new benchmark generation is needed before channel qualification. |
| Real SOTA stream | Exp5786 produced 1,080 rows with authenticated CUDA offload for Qwen3.6-35B-A3B, Gemma4-31B, and Gemma4-26B-A4B. Gemma4-26B scored `0.927778`, Gemma4-31B scored `0.75`, and both had zero parser failures. Qwen had 360/360 truncations at `max_tokens=48`, so `parser_failure_rate=0.333333` and `stream_ready_score=0`. | Preserve the Gemma evidence, diagnose Qwen reasoning/final-content behavior, and qualify a bounded channel before another scaled run. Do not call the Qwen result a competence failure. |
| Continuous self-learning | Exp5787 was gate-blocked, Exp5788 had no exact deliverable, and Exp5789 repeated its gate-blocked verdict. No learning A/B, transfer, or integration evidence exists. | A future clean stream may reopen exact skill learning, but the retired shadow-integration shape remains closed. Test lifecycle durability and OOD safety without production wiring. |
| ARC admission | Exp5790 shipped `admission_contract_ready_score=1.0`, with pivotal-dynamics and no-source-leak controls. It claimed no solve. | Reuse the immutable admission contract; do not rebuild or weaken it. |
| ARC hypothesis panel | Exp5791 verified all three GGUFs were cached and two RTX 3090s were visible, then blocked on `headline_gpu_offload_receipts_present`, `fresh_matched_hypotheses_present`, and `resume_checkpoint_file_present` before loading or generating. | Repair execution ordering. Preconditions may check inputs and capacity; load receipts, checkpoints, and hypotheses are outputs. |
| ARC selector/live A/B | Exp5792 produced no artifact and Exp5793 was gate-blocked. | Run them only behind new panel/admission evidence. No public solve, source, offline BFS, GameAdapter, CEGIS, or registry credit. |
| Hardware | Exp5794 reconciled cached KV260, PolarFire, and GateMate state and made no speed/energy claim. No physical precondition changed. | Do not repeat probes. Produce a bounded self-learning microkernel/ABI handoff that can later be mapped to attached boards. |

## The three biggest gaps to the PRD vision

### Gap 1 — Real local models run, but a verified answer channel is not portable across families

FR-12 requires exact verification, yet exact validators cannot act on a response that never reaches
its answer boundary. `.516` demonstrated that a common `max_tokens=48` transport is invalid for the
Qwen3.6 reasoning model even though the same fixture and parser work for Gemma. Recent llama.cpp
receipts also show that grammar constraints can leak into Qwen reasoning content, produce empty
final content, or loop to the token limit. Carnot needs a model-aware but semantically identical
transport contract with raw reasoning/final-content/stop/token receipts and fail-closed parsing.

### Gap 2 — FR-11 self-learning has a positive synthetic mechanism but no prospective durability

The PRD calls for autonomous propose-measure-update loops that retain prior knowledge. Carnot has
solver-authoritative typed constraint acquisition and positive dependent-task transfer from `.514`,
but no clean prospective stream, fixed future-batch admission, multi-change endurance, memory cap,
restart equivalence, or OOD audit. `.517` must show that versioned sidecar skills improve a sealed
future suffix without changing protected facts, GGUF weights, exact validators, or old-prefix
behavior. A null is acceptable; unsafe propagation is terminal.

### Gap 3 — ARC world-model selection is specified but not reachable by the live agent

The admission contract is ready, while the producer blocked on outputs mislabeled as preconditions.
The live path therefore has no matched three-family panel, no calibration-only selector, and no
held-out E3 evidence. `.517` must create immutable hypotheses from the agent's own transitions,
admit and select them without test leakage, and measure policy influence at the scored
`E3AgentPolicy` entrypoint. The target is held-out action/game generalization, not another solve of
the complete 25-game public registry.

NFR-01's production 10x goal remains open. The allocation-free software route is retired, and no
attached-board state changed. This milestone narrows the hardware gap by defining and benchmarking
a portable bounded update/lookup microkernel for accepted self-learning state; it makes no FPGA,
thermodynamic, energy, or 10x claim.

## 2025-2026 research update and experiment hooks

The dated search ledger is in `research-references.md` under
`V517-PLANNER-REFRESH-20260722-END`.

| Finding | Carnot implication | Experiment hook |
|---|---|---|
| [Self-Evolving World Models for LLM Agent Planning](https://arxiv.org/abs/2606.30639) | Frozen parameters can support deployment-time learning through episodic memory, persistent rules derived from mismatch, and selective use of uncertain foresight. | Exp5801-Exp5803 use exact outcomes as observations, version typed skills, future-gate edits, cap memory, and abstain/rollback under uncertainty. This does not reopen ARC CEGIS. |
| [Reasoning-Aware SLOs](https://openreview.net/forum?id=eEZ7uze9kf) | First-token latency is not useful when reasoning branches truncate or fail verification. | Exp5799/Exp5800 report verified accepted outputs per wall-second/token and wasted/truncated token work; Exp5807 benchmarks verified useful update/lookup operations. |
| [llama.cpp Qwen grammar/thinking issue #20345](https://github.com/ggml-org/llama.cpp/issues/20345) | Grammar may constrain reasoning content, leave final content empty, or loop; reasoning-disable support is model/template dependent. | Exp5798 builds the forensic mode matrix; Exp5799 treats grammar as one guarded candidate, uses embedded GGUF templates, and requires raw reasoning/final-content receipts. |
| [Verifiable Self-Evolution via Future-Feedback Prediction](https://arxiv.org/abs/2607.18973) | Textual skills can be validation-gated against fixed downstream feedback while the base model stays frozen. | Exp5801 admits typed skill edits only on an immutable future batch with exact solver authority and rollback. |
| [Solver-Hard Is Not Model-Hard](https://arxiv.org/abs/2607.17047) | Solver conflicts and proof-preserving surfaces are separate causal axes. | Exp5800 preserves the `.516` fixture's crossed reporting; Exp5803 audits surface and family transfer rather than averaging them away. |
| [When a Verified World Model Still Loses](https://arxiv.org/abs/2607.14169) | Transition accuracy alone misses pivotal dynamics needed for play. | Exp5804 reuses Exp5790's pivotal admission; Exp5805 cannot select a high-average model that fails play-cost coverage. |
| Extropic TSU and Logical Intelligence Kona public pages | Both remain architecture-relevant but unavailable as authenticated local execution surfaces. | No proprietary execution or performance claim; the microkernel handoff stays open, local, and backend neutral. |

## Target architecture

```text
                 exact sealed constraint fixture (Exp5785)
                                |
                                v
          offline failure forensics on Exp5786 raw responses
          reasoning | final content | stop | token exhaustion
                                |
                                v
 Qwen3.6-35B-A3B ---- answer-channel canary ---- Gemma4-31B
               \            |                 /
                \---- Gemma4-26B-A4B --------/
             embedded templates; guarded grammar/no-think arms
                                |
                    three-family clean-channel gate
                                |
                                v
        sealed chronological exact response stream (no learning)
                                |
             solver-certified typed constraint proposals
                                |
                                v
 versioned sidecar: quarantine -> fixed future validate -> commit/rollback
                                |
              endurance + OOD family/surface safety audits
                                |
           packed bounded update/lookup microkernel handoff


 agent-owned ARC transitions -> immutable admission contract (Exp5790)
                                |
                                v
       3 SOTA families x independent single-shot hypotheses
       checkpoint/create -> load receipt -> generate -> freeze -> score
                                |
                                v
        calibration-only selector over admissible hypotheses
                                |
                    positive lower-bound gate
                                |
                                v
       held-out scored E3AgentPolicy A/B on live observations
       no source / BFS / adapter / iterative patch / solve credit
```

The constraint and ARC branches share exact evidence, local execution, and fail-closed gate
principles but have no scientific dependency. Phase 3 consumes accepted constraint state only for
the microkernel; the capstone always runs.

## Phase 0 — Transition and answer-channel qualification (Exp5796-Exp5799)

### Exp5796 — Transition terminal `.516` evidence and allocate `.517`

Archive every `.516` task by exact declared path and conductor outcome. Preserve positive,
scientific-null, negative, blocked-precondition, blocked-gate, missing, and no-solve classes as
distinct. Append `.516` completion exactly once and allocate Exp5796-Exp5808 collision-free.

**Deliverable:** `results/experiment_5796_transition_v517.json`

### Exp5797 — Post-V517 time-windowed source refresh

Search only work newer than the V517 planner marker. Findings may add bounded controls inside the
allocated tasks, but may not rewrite IDs, gates, retired scopes, model requirements, or hardware
claims. Zero accepted findings is complete.

**Deliverable:** `results/experiment_5797_v517_source_delta_ingestion.json`

### Exp5798 — Forensic answer-channel diagnosis

Analyze Exp5786's complete raw rows without calling an LLM. Attribute each Qwen truncation to
reasoning/final-content/token/stop/template behavior, compare Gemma controls, inspect embedded GGUF
template metadata and pinned runtime receipts, and preregister a bounded candidate-mode matrix.
Grammar is a guarded arm, not the presumed repair.

**Deliverable:** `results/experiment_5798_sota_answer_channel_diagnostic.json`

### Exp5799 — Three-family real-GGUF channel canary

Run a small matched canary on all three mandated models. Test only the preregistered modes, preserve
raw reasoning and final content separately, and include empty-content, token-exhaustion, invalid-ID,
schema-injection, stop, and exact-answer controls. Select a per-model transport only when it produces
the same finite-choice semantic contract with zero parser failures and authenticated CUDA offload.

**Deliverables:**

- `results/experiment_5799_sota_answer_channel_canary.json`
- `results/experiment_5799_sota_answer_channel_canary.rows.jsonl`

## Phase 1 — Prospective continuous self-learning (Exp5800-Exp5803)

### Exp5800 — Channel-qualified three-family prospective stream

Reuse the sealed Exp5785 fixture and selected Exp5799 transport for each model. Run all three
mandated SOTA families at N>=30 independent units per primary cell, preserve checkpoint/resume and
raw response receipts, and report solver hardness, surface sensitivity, parser failure, truncation,
exact error, satisfiable drift, and protected-fact distortion separately. This task learns nothing.

**Deliverables:**

- `results/experiment_5800_channel_qualified_constraint_stream.json`
- `results/experiment_5800_channel_qualified_constraint_stream.rows.jsonl`

### Exp5801 — Future-validated typed-skill learning A/B

Compare frozen state, contradiction-only feedback, and query-driven typed skills. Proposed updates
come only from chronological prefix evidence, remain quarantined, and are committed only if an
immutable future batch yields a positive paired lower bound with zero unsafe propagation, exact
old-prefix retention, and state-hash rollback. Model weights and exact validators remain immutable.

**Deliverable:** `results/experiment_5801_future_validated_constraint_skill_ab.json`

### Exp5802 — Multi-change endurance, retirement, and restart

Replay at least two preregistered distribution changes through the credited lifecycle. Measure
dynamic regret, delayed retention, rule interference, memory growth, selective abstention,
quarantine, retirement, rollback, and interruption/resume equivalence under a fixed sidecar memory
cap. This is the required continuous-self-learning durability experiment.

**Deliverable:** `results/experiment_5802_constraint_skill_endurance.json`

### Exp5803 — Leave-one-family/surface OOD audit

Hold out generator family, constraint family, and proof-preserving surface axes. Compare reset,
frozen, transferred, and retired-rule states; separate useful transfer from model-identity shortcuts
and safety from benefit. No shadow adapter or production answer path is introduced.

**Deliverable:** `results/experiment_5803_constraint_skill_ood_audit.json`

## Phase 2 — Reachable ARC world-model generalization (Exp5804-Exp5806)

### Exp5804 — Bootstrap-safe matched single-shot ARC panel

Reuse Exp5790's admission contract. Create the checkpoint/seed manifest, load each real model and
capture offload, then generate and freeze independent single-shot hypotheses before scoring. Run
Qwen3.6-35B-A3B, Gemma4-31B, and Gemma4-26B-A4B under matched agent-owned splits and budgets. No
hypothesis may be iteratively patched or see source, a GameAdapter, offline BFS, or test transitions.

**Deliverables:**

- `results/experiment_5804_arc_bootstrap_safe_sota_panel.json`
- `results/experiment_5804_arc_bootstrap_safe_sota_panel.hypotheses.jsonl`

### Exp5805 — Calibration-only immutable-hypothesis selector

Choose only among admissible frozen hypotheses using calibration transitions, complexity, stability,
horizon-matched disagreement, and pivotal coverage. Compare first, random, and a disclosed
non-deployable held-out oracle upper bound. Evaluate once on the sealed test split; never patch a
rejected hypothesis.

**Deliverable:** `results/experiment_5805_arc_immutable_selector.json`

### Exp5806 — Held-out live E3 generalization A/B

If the selector has a positive paired lower bound and zero leakage, compare baseline E3 with
selected-world-model E3 at the scored `E3AgentPolicy` entrypoint using identical held-out
games/actions, seeds, and budgets. Report policy utility, validity, abstention, and pivotal outcomes.
No public-game level is targeted; any incidental live success remains
`live_agent_self_discovery` provenance and does not update the solve registry in this task.

**Deliverable:** `results/experiment_5806_arc_live_heldout_world_model_ab.json`

## Phase 3 — Hardware path and reconciliation (Exp5807-Exp5808)

### Exp5807 — Bounded self-learning update/lookup microkernel handoff

If Exp5801 credits self-learning, translate only its accepted versioned-rule operations into a
backend-neutral packed ABI: lookup, quarantine insert, validate/commit, retire, rollback, and state
hash. Prove Python/Rust exact parity, benchmark verified useful operations under a preregistered
memory cap, and map resource envelopes to cached KV260/PolarFire/GateMate constraints. Run no board
command unless a precondition hash changed; claim no 10x, board, speed, power, or energy result.

**Deliverable:** `results/experiment_5807_self_learning_microkernel_handoff.json`

### Exp5808 — `.517` capstone reconciliation

Resolve all tasks by exact declared identity and conductor outcome, replay gates, apply repeated-
failure retirement, compute phase telemetry, and reconcile OpenSpec, traceability, research history,
status, changelog, known issues, and architecture only where evidence warrants it. Keep public claims,
production sidecar influence, ARC registry credit, and hardware claims closed unless their exact
promotion gates pass.

**Deliverable:** `results/experiment_5808_v517_capstone_reconciliation.json`

## Dependency graph

```text
Exp5796 transition ----> Exp5797 source refresh

Exp5786 rows ----------> Exp5798 diagnosis
                              |
                              v
                         Exp5799 canary
                              |
                              v
                         Exp5800 stream
                              |
                              v
                         Exp5801 skill A/B
                              |
                              v
                         Exp5802 endurance
                              |
                              v
                         Exp5803 OOD audit
                              |
                accepted skill state only
                              v
                         Exp5807 microkernel

Exp5790 admission -----> Exp5804 SOTA panel
                              |
                              v
                         Exp5805 selector
                              |
                              v
                         Exp5806 live E3 A/B

Exp5796-Exp5807 -------> Exp5808 capstone (always runs)
```

The conductor executes in numeric order. Structured `gated_on` fields skip expensive downstream
agent calls when producer scalars fail. Exp5808 has no scientific gate.

## Hardware and execution requirements

| Resource | Use | Requirements and boundaries |
|---|---|---|
| Dual NVIDIA RTX 3090, 24 GB each | Exp5799, Exp5800, Exp5804 | Real local GGUF inference through CUDA-enabled llama.cpp. Each headline row records model path/hash, quantization, embedded template hash, runtime/binary hash, requested and actual offload layers, VRAM delta/peak, token counts, stop reason, seed, and checkpoint. CPU-only smoke evidence is non-headline. |
| Mandated GGUF cache | Exp5799, Exp5800, Exp5804 | `unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and `unsloth/gemma-4-26B-A4B-it-GGUF`. Use cached `.gguf` files and embedded tokenizer/templates; never call `AutoTokenizer` on a GGUF repository. |
| CPU/RAM/disk | All tasks | At least 64 GB host RAM recommended, enough free disk for raw rows/checkpoints, deterministic local exact validators, and atomic artifacts. Every task records precondition capacity and hashes. |
| Z3/exact validators | Exp5798-Exp5803 | Exact labels and protected-fact authority. LLM or learned scores never replace them. |
| KV260 | Exp5807 mapping only | Existing SSH/bitstream receipts are cached. No `/dev/mmcblk*` access, flashing, or storage writes. A bounded non-destructive check is allowed only if the canonical precondition hash changed. |
| PolarFire SoC Icicle | Exp5807 mapping only | Existing SSH/hash smoke is cached; terminal workload validation remains absent. No performance claim without authenticated workload execution. |
| GateMate | Exp5807 mapping only | Toolchain exists, while physical DirtyJTAG/cable state remains blocked. No repeated probe or board claim without changed physical preconditions. |
| Extropic XTR-0/Z1 and Kona | Context only | No authenticated local route or open reproducible comparator. No execution, power, energy, certainty, or speedup claim. |
| Network | Exp5797 only | Primary arXiv/OpenReview/official-project metadata and named secondary checks. Unavailable routes are reported, not guessed. Science inference remains local/offline. |

## Promotion criteria

| Branch | Promotion gate |
|---|---|
| Answer channel | All three mandated SOTA families have authenticated real-GGUF CUDA receipts, raw reasoning and final-content coverage `1.0`, exact-label coverage `1.0`, parser/truncation/empty-content rate `0.0`, and adversarial channel controls passing. |
| Prospective stream | Channel gate passes at preregistered sample size; chronology and row hashes are sealed; exact taxonomy is complete; at least 30 independent solver-certified learnable errors exist for the learning A/B. |
| Continuous self-learning | Paired satisfiable-drift reduction lower bound `>0`, unsafe propagation `=0`, protected-fact distortion delta `<=0`, old-prefix retention `=1.0`, rollback/restart equivalence `=1.0`, immutable GGUF hashes, and bounded state growth. |
| OOD learning safety | Every leave-one-family/surface cell has zero unsafe propagation and exact rollback; transfer benefit is reported honestly and may be null. No shadow adapter promotion occurs in `.517`. |
| ARC panel | All three real SOTA families actually load and generate independent immutable hypotheses; admission scores are complete; source/offline/test leakage counts are zero. Panel completion does not require an admissible model. |
| ARC selector | At least one admissible hypothesis, selector-vs-random paired lower bound `>0`, zero test leakage, and selected hypothesis passes pivotal/play-adequacy requirements. |
| Live ARC influence | Scored `E3AgentPolicy` utility paired lower bound `>0`, validity non-inferiority, zero source/BFS/adapter leakage, and agent-owned live provenance. No public solve or registry claim. |
| Hardware path | Exact Python/Rust operation parity, bounded memory/state, deterministic rollback hash, and truthful verified-useful-op telemetry. No 10x or board claim. |

## Retired-scope and safety boundaries

- Do not rerun PHASE-D generated-text/logprob external scoring, LNN within-chain adaptation,
  allocation-free one-axis 10x scaling, two-axis tempering, ARC transition-cycle accreditation,
  counterexample-patched/CEGIS world models, generic component composition, cross-game value
  transfer, or public-game solve tasks.
- Do not recreate the retired constraint shadow-integration task. `.517` ends at endurance, OOD
  audit, and a backend-neutral microkernel handoff.
- Grammar-constrained decoding is not exact semantic authority and may be a safety/control-plane
  hazard. Raw outputs, exact candidate membership, protected facts, and independent validators remain
  authoritative.
- A gate skip, missing artifact, bootstrap block, cached hardware receipt, development proxy, or
  non-deployable oracle is not a scientific null or promoted result.
- No task may modify `research-roadmap.yaml` or `scripts/research_conductor.py`, push, publish, flash
  hardware, write board storage, or update the ARC solve registry.

## Decentralization and local-first implications

The milestone uses open local GGUFs, exact local validators, content-addressed rows, versioned sidecar
state, and a backend-neutral microkernel contract. A result can be reproduced without a proprietary
inference or verifier service, and the accepted learning state can move with its hashes, rollback
log, and exact authority. Extropic and Kona remain optional future substrates, not dependencies.
This keeps Carnot's verification and learning control plane inspectable, self-hostable, and portable
across CPU, GPU, Rust, and eventual attached hardware.


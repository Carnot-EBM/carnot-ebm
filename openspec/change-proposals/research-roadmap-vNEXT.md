# Research Roadmap vNEXT

**Milestone:** `2026.07.511`  
**Title:** Attested Exact Streams, Lifecycle-Audited Self-Learning, Production Rust Sampling, and ARC Epistemic State  
**Status:** Proposed  
**Date:** 2026-07-15  
**Task range:** `exp5717`-`exp5728` (12 experiments, collision-checked)  
**Conductor file:** `research-roadmap-next.yaml`

## Milestone thesis

Milestone `.510` proved that Carnot can run a mandated current GGUF with authenticated
CUDA offload and that the promoted one-axis sampler survives an exact Rust/PyO3 port.
It also exposed two sharper failures: the generated exact stream was unusable because
47 of 50 responses failed parsing, and a correctly wired relational ARC goal route was
null in a matched live A/B. Milestone `.511` therefore moves each positive substrate one
layer closer to the PRD while repairing failures at their actual boundaries:

1. qualify the local-GGUF answer channel before generating another sealed stream;
2. let only exact-validator-attested envelopes enter a lifecycle-audited FR-11 canary;
3. wire the promoted Rust one-axis kernel into the production `SamplerBackend` contract
   and measure a matched quality/throughput crossover; and
4. replace the null ARC relational-score extension with a generic, agent-owned epistemic
   ledger over confirmed observations, ranked hypotheses, and open questions, then run
   the standing unconditional live self-discovery attempt.

The milestone does not reopen the retired native three-model/JSON-grammar certificate,
external generated-text scoring, PTRM-as-generator, counterexample transition patcher,
generic exploration-signal family, two-axis beta-lambda exchange, or TSU/Kona execution.

## What milestone `.510` proved

| Evidence | Terminal result | Consequence for `.511` |
|---|---|---|
| Exp5706 transition | `.509` evidence and the Exp5641/Exp5645 retirements were preserved. | Start from the terminal capstone and add only the one remaining narrow Exp5709 retirement. |
| Exp5707 source ingestion | No non-duplicate execution-time source changed the graph. | Retain a bounded SOTA-ingestion slot; a clean no-op remains success. |
| Exp5708 SOTA exact canary | Authenticated `llama-cpp-python` CUDA offload succeeded, but `parse_failure_count=47/50`: 21 truncations and 26 missing answer lines. | The runtime is usable; the answer channel is not. Diagnose chat-template, token-budget, stop, repetition, and finish-reason behavior before another panel. |
| Exp5709/Exp5710 FR-11 | The prospective shadow task gate-blocked; the isolated canary was pre-emptively skipped and emitted no artifact. | Preserve the `.509` controller positive, but admit no `.510` stream evidence. A clean attested stream is a new prerequisite. |
| Exp5711 ARC qualification | The zero-variance-safe relational route was live-path reachable, discriminative on controls, and leakage-free. | Live reachability alone is no utility claim. |
| Exp5712 ARC matched A/B | Six matched pairs, zero regressions and zero unsafe routes, but no frozen primary benefit; `relational_live_ab_ready_score=0`. | Do not promote or rerun the relational route. Change the mechanism from a scalar goal feature to explicit epistemic state and commitment. |
| Exp5713 ARC live attempt | Honest bounded attempt at `lf52` L9, `solve_provenance=live_agent_self_discovery`, registry delta 0. | Rotate away from every reproduced/recently attempted level and run another unconditional live attempt. |
| Exp5714 Rust parity | One-axis corrected-cDLS Rust/Python energy, proposal, swap, scheduler, checkpoint, and restart parity passed. | The Rust kernel is semantically qualified. |
| Exp5715 Rust quality/restart | Five-seed hard-instance quality and cross-language restart passed with zero material regressions. No timing claim. | Production adapter integration and a matched timing study are now admissible. |
| Exp5716 capstone | Honest blocked reconciliation: no prospective FR-11 promotion, ARC delta 0, Rust parity/quality positive, no speedup claim. | `.511` must preserve all boundaries and reconcile missing/gate-skipped artifacts explicitly. |

## The three largest gaps to the PRD vision

### Gap 1: no current-model stream can reach continuous self-learning

FR-11 calls for continuous, auditable learning from verifier feedback. The active-spline
controller has strong replay evidence, but the only current mandated-GGUF stream was
unusable. CUDA was not the blocker: the failure was at the completion/chat boundary.
Without a lossless chronological stream, no prospective or act-on-advice claim is
admissible.

`.511` response: Exp5719 diagnoses and freezes a non-retired answer-channel protocol;
Exp5720 generates a sealed exact stream whose envelopes are attested by deterministic
validators; Exp5721 and Exp5722 perform operation-level prospective and isolated
act-on-advice evaluation. GGUF weights and the production default remain immutable.

### Gap 2: the live ARC agent still lacks useful organized state for hidden-game discovery

Carnot's ARC north star is not an offline solve. It is the live submitted agent discovering
hidden-game dynamics from its own observations and actions. The `.510` relational goal
route reached the live policy but changed no matched primary metric, and the registry stayed
at 177. The next gap is not another scalar score; it is preserving what the agent has
confirmed, what remains hypothetical, and which observation would discriminate those
hypotheses before budget is exhausted.

`.511` response: Exp5725 adds a generic epistemic ledger reachable from `E3AgentPolicy`;
Exp5726 tests it under a matched known-level A/B; Exp5727 runs a registry-prechecked,
unconditional live self-discovery attempt whether or not the A/B promotes.

### Gap 3: the promoted Rust sampler is still an experiment module, not a production backend

The PRD requires a Rust production core and swappable sampler backends. `.510` proved
semantic and restart parity but did not expose the one-axis kernel through the production
`SamplerBackend` factory or measure the Python/Rust crossover under matched work and
quality. A portability proof is not yet an integration or performance result.

`.511` response: Exp5723 adds the production adapter and exact fallback; Exp5724 measures
matched Python/Rust quality, work, warmup, serialization, and wall-clock distributions.
No speed claim is made unless the preregistered interval and quality gates pass.

## Research incorporated before design

The complete planning sweep and dispositions are recorded in the `V511 Planner Refresh`
block of `research-references.md`.

| 2025-2026 source | Actionable idea | `.511` use |
|---|---|---|
| EG-VAR, arXiv:2607.12650 | Only a kernel/tool-attested path may mint a verified claim; unsupported cases abstain with replayable evidence. | Exp5720 separates model proposals from exact-validator authority and hashes every envelope, source, model, and validation receipt. |
| MemOps, arXiv:2607.12893 | Audit remember, forget, update, reflect, and composed lifecycle operations rather than final accuracy alone. | Exp5721 records each FR-11 update, supersession, rejection, rollback, and forgetting operation with pre/post state hashes. |
| Compliance Trap, arXiv:2607.10608 | Measure memory Entry, Propagation, and Recovery under conflicting or stale memory. | Exp5722 injects conflicts and measures first changed decision, downstream propagation, and exact rollback recovery. |
| SLEUTH, arXiv:2607.12267 | Externalize confirmed facts, ranked hypotheses, and open questions; commit only when evidence is sufficient. | Exp5725/Exp5726 implement and test a generic agent-owned ARC epistemic ledger and bounded commitment trigger. |
| MaxSAT feedback for Sudoku, arXiv:2607.12711 | Keep rules hard and return the largest mutually consistent subset of proposed assignments as repair feedback. | Exp5720 emits exact conflict sets for inconsistent envelopes; the solver remains authoritative and no Sudoku/VLM claim is made. |
| Continual facts in weights, arXiv:2607.11020 | Sequential weight writes can make older facts behaviorally unreachable; context remains the reliable recovery channel. | All FR-11 experiments keep GGUF weights frozen and credit only external rollback-capable constraint state. |
| Calibrated e-CUSUM, arXiv:2607.11317 | Centered token log-probability is not a reliable decoder-health observable; repetition and termination need direct telemetry. | Exp5719 diagnoses finish reason, repetition, truncation, stop, and template behavior instead of reviving token-energy authority. |
| Energy-guided Recursive Model, arXiv:2607.10128 | Hopfield energy can rank recursive reasoning trajectories. | Watch-only: the dedicated PTRM-generator scope is retired and is not reopened in `.511`. |
| OpenReview XSkill / SDFT / Training as Computation | Continual systems need bounded verifier-buffer loops, retention, and explicit experience/skill lifecycles. | Architectural support for Exp5721/Exp5722; no GGUF fine-tuning or broad RL task. |

Secondary checks found no new authenticated Extropic TSU path, no reproducible Kona
weights/runtime, no GitHub project that supersedes Carnot's current backends, and no EBT
or ARM-EBM citation that changes this execution order.

## Target architecture after `.511`

```text
                 exact authority and immutable evidence
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Mandated local GGUFs                                                   │
│  Qwen3.6-35B-A3B / Gemma-4-31B / Gemma-4-26B-A4B                      │
└────────────────┬─────────────────────────────────────────────────────────┘
                 │ authenticated CUDA + native chat-template receipts
                 ▼
        ┌──────────────────────┐       Exp5719
        │ answer-channel gate  │────── forensics / positive controls
        └──────────┬───────────┘
                   │ frozen qualified protocol
                   ▼
        ┌──────────────────────┐       Exp5720
        │ attested envelopes   │────── raw response + exact conflict set
        └──────────┬───────────┘
                   │ sealed chronological exact stream
                   ▼
        ┌──────────────────────┐       Exp5721 / Exp5722
        │ FR-11 lifecycle KAN  │────── prequential operations + E/P/R audit
        │ isolated checkpoint  │────── rollback; production remains disabled
        └──────────────────────┘

 ARC visible observations/actions                      energy descriptors
                 │                                             │
                 ▼                                             ▼
        ┌──────────────────────┐                    ┌──────────────────────┐
        │ epistemic ledger     │ Exp5725            │ SamplerBackend       │ Exp5723
        │ confirmed/hyp/open   │                    │ Rust one-axis PyO3   │
        └──────────┬───────────┘                    └──────────┬───────────┘
                   │ E3AgentPolicy live path                    │ exact fallback
                   ▼                                             ▼
        ┌──────────────────────┐                    ┌──────────────────────┐
        │ matched A/B + live   │ Exp5726/5727       │ matched quality/time │ Exp5724
        │ self-discovery       │                    │ crossover receipt    │
        └──────────────────────┘                    └──────────────────────┘

 Exact validators remain final authority across all three lanes.
 No LLM judge, game source, per-game adapter, outer-loop BFS, or TSU/Kona claim.
```

## Phase 1 - Terminal evidence, source freshness, and answer-channel qualification

### Exp5717 - Transition terminal `.510` evidence

Archive every Exp5706-Exp5716 terminal artifact and conductor outcome, apply the capstone's
narrow missing retirement for the parse-failed Exp5709 stream scope, preserve future clean
prospective streams, and allocate the collision-free Exp5717-Exp5728 graph. Emit the current
ARC registry count, FR-11 boundaries, Rust readiness scalars, and exact retirement/preservation
map.

**Deliverable:** `results/experiment_5717_transition_v511.json`

### Exp5718 - Post-V511 source-delta ingestion

Search only after the V511 planner marker, deduplicate against the research history, and map
genuine new hooks to existing tasks without silently changing IDs or gates. A no-op is valid.

**Deliverable:** `results/experiment_5718_v511_source_delta_ingestion.json`

### Exp5719 - Mandated-GGUF answer-channel forensics

Reproduce the `.510` raw-completion control, then compare a small preregistered matrix of
native GGUF chat-template invocation, stop policy, answer budget, and reasoning/answer
separation across all three mandated model families. Use only exact positive controls and
negative malformed/truncation controls. Freeze one protocol only if at least two mandated
models attain 100% control parse success with authenticated CUDA and no repetition,
truncation, missing rows, or validator disagreement. No native JSON grammar or external
scorer is allowed.

**Deliverable:** `results/experiment_5719_sota_answer_channel_forensics.json`

## Phase 2 - Attested exact stream and continuous self-learning

### Exp5720 - Sealed attested exact-envelope canary

**Gate:** Exp5719 answer-channel readiness, 100% positive-control parse success, at least two
qualified mandated models, and authenticated CUDA.

Generate a sealed, chronological, balanced hard/soft exact stream with the flagship Qwen MoE
and flagship Gemma dense GGUFs using the frozen protocol. Store raw responses losslessly.
Each usable envelope receives an independent deterministic validator receipt; inconsistent
proposals receive a bounded exact conflict set. Any missing row, parse failure, validator
disagreement, or provenance break blocks the stream.

**Deliverable:** `results/experiment_5720_sota_attested_exact_envelope_canary.json`

### Exp5721 - MemOps lifecycle prospective FR-11 shadow stream

**Gate:** Exp5720 stream readiness, zero parse failures, and zero validator disagreements.

This is the milestone's first continuous self-learning experiment. Consume the committed
prefix once in chronological order. Record pre-label decisions, then every accepted,
rejected, superseded, reflected, forgotten, and rolled-back sidecar operation with exact
evidence and pre/post hashes. Compare the active-spline controller with frozen, no-memory,
last-window, always-adapt, and corrupted-order controls. GGUF weights do not change.

**Deliverable:** `results/experiment_5721_fr11_memops_lifecycle_shadow_stream.json`

### Exp5722 - Compliance-Trap recovery and rollback canary

**Gate:** Exp5721 lifecycle readiness and zero unsafe false accepts.

Open the untouched suffix inside an isolated controller checkpoint. Let admissible advice
change only sidecar/KAN state, inject stale and conflicting memory, and measure Entry,
Propagation, and Recovery. Exercise crash boundaries and exact rollback. Production remains
disabled; exact rejection cannot be overridden.

**Deliverable:** `results/experiment_5722_fr11_compliance_recovery_rollback_canary.json`

## Phase 3 - Production Rust sampler and matched crossover

### Exp5723 - One-axis Rust `SamplerBackend` adapter

**Gate:** Exp5717 confirms the `.510` Rust quality/restart promotion.

Expose the promoted Rust one-axis corrected-cDLS kernel through the production Python
`SamplerBackend` protocol and factory. Preserve deterministic checkpoints, exact fallback,
seed semantics, energy accounting, and Python/Rust round trips. Broken-binding and corrupted
checkpoint controls must fail closed. This task makes no timing claim.

**Deliverable:** `results/experiment_5723_one_axis_rust_samplerbackend_integration.json`

### Exp5724 - Matched Rust/Python quality-throughput crossover

**Gate:** Exp5723 production-backend readiness and exact fallback equivalence.

Benchmark the production Rust and Python one-axis implementations on identical workloads,
seeds, ladders, transition budgets, restart schedules, checkpoints, warmups, and CPU affinity.
Report quality, work, serialization/PyO3 overhead, wall-clock distributions, confidence
intervals, and the first qualified crossover size if one exists. A null is a valid terminal
result. This is CPU software timing, not FPGA/TSU/GPU hardware acceleration.

**Deliverable:** `results/experiment_5724_one_axis_rust_python_matched_crossover.json`

## Phase 4 - ARC epistemic state, live attempt, and reconciliation

### Exp5725 - ARC epistemic-ledger live-path qualification

Implement a generic ledger of confirmed observation/action facts, ranked transition/goal
hypotheses, open discriminating questions, and evidence-linked contradiction/supersession
events. Prove that current `E3AgentPolicy` reads and updates it using only agent-owned receipts.
Use synthetic controls and registry-prechecked reproduced levels, with leave-one-game-out,
stale-evidence, false-commit, and orphan-solver tests. Claim no new solve and record
`solve_provenance=development_proxy`.

**Deliverable:** `results/experiment_5725_arc_epistemic_ledger_live_qualification.json`

### Exp5726 - Matched known-level live A/B

**Gate:** Exp5725 readiness, live reachability, and zero per-game leakage.

Compare the current full stack against the identical stack plus the epistemic ledger and
bounded commitment trigger on registry-prechecked reproduced levels. Match all budgets and
policy knobs. Promote only for a frozen benefit with no level regression or unsafe commit.
This task does not bank a level and records `solve_provenance=development_proxy`.

**Deliverable:** `results/experiment_5726_arc_epistemic_ledger_live_ab.json`

### Exp5727 - Unconditional ARC live self-discovery level-up attempt

Registry-precheck before target selection. Exclude every reproduced level and recent failed
target, including `lf52` L9. Use Exp5726 only if promoted and target-locally learned; otherwise
run the unchanged baseline. Only the live agent's own observations, attempts, and runtime RE
can receive credit. The required artifact field is
`solve_provenance=live_agent_self_discovery`; no source read, per-game adapter, exhaustive
offline BFS, hand model, or off-path solver is permitted.

**Deliverable:** `results/experiment_5727_arc_live_self_discovery_levelup_v511.json`

### Exp5728 - `.511` capstone reconciliation

Aggregate every Exp5717-Exp5727 artifact plus gate-skip/missing states; reconcile OpenSpec,
traceability, status, changelog, conductor log, exclusions, known issues, verifier gaps,
north-star, and applicable E2E receipts. Preserve negative/null results. A clean capstone may
remain `blocked:` if the attested stream or live solve does not promote.

**Deliverable:** `results/experiment_5728_v511_capstone_reconciliation.json`

## Dependency graph

```text
Phase 1
Exp5717 transition ──────────────────────┬──────────────► Exp5723
                                        │                    │
Exp5718 source delta ────────────────────┼────────────────────┼──────────┐
                                        │                    ▼          │
Exp5719 answer-channel forensics ───────► Exp5720 ─────► Exp5721       │
                                              │             │          │
                                              │             ▼          │
                                              │          Exp5722       │
                                              │                        │
                                              └────────────────────────┤
Exp5723 SamplerBackend ────────────────► Exp5724                       │
                                                                       │
Exp5725 ARC epistemic qualification ──► Exp5726                       │
                                                                       │
Exp5727 ARC live attempt (UNCONDITIONAL; Exp5726 advisory only) ──────┤
                                                                       ▼
                                                             Exp5728 capstone
```

No `requires:` chain points to a retired experiment. Structured gates are conjunctive and
refer only to Exp5717-Exp5726 artifacts. Exp5727 is deliberately ungated.

## Hardware and model requirements

| Resource | Tasks | Requirement and boundary |
|---|---|---|
| RTX 3090 GPU 0/1 | Exp5719-Exp5720 | Authenticated CUDA-enabled `llama-cpp-python`, positive offloaded-layer receipt, GPU memory deltas, and one loaded model per device unless a recorded VRAM proof permits otherwise. CPU fallback is diagnostic only and never headline. |
| Mandated local GGUFs | Exp5719 | `unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and `unsloth/gemma-4-26B-A4B-it-GGUF` in the explicit `MODEL_SPECS`. |
| Flagship SOTA pair | Exp5720 | `unsloth/Qwen3.6-35B-A3B-GGUF` plus `unsloth/gemma-4-31B-it-GGUF`, resolved from immutable local cache and executed through GGUF-native llama.cpp APIs. |
| CPU/RAM | Exp5721-Exp5726, Exp5728 | KAN lifecycle replay, exact solvers, Rust/Python sampling, ARC policy, tests, and reconciliation. Record peak memory for the crossover. |
| Rust/PyO3 toolchain | Exp5723-Exp5724 | `cargo`, existing `carnot-samplers` crate, bindings, deterministic checkpoint schema, and reproducible optimized build profile. |
| NVMe | All phases | GGUF hashes, sealed row manifests, raw responses, lifecycle ledgers, checkpoints, benchmark samples, ARC traces, and artifact hashes. |
| ARC live environment | Exp5725-Exp5727 | Submitted live path and agent-owned receipts only. Registry precheck is mandatory; game source and per-game adapters are forbidden. |
| KV260 / PolarFire / GateMate | None | No board task is load-bearing in `.511`; no FPGA reachability or speedup claim is made. |
| Extropic TSU / Kona | None | Watch-only architecture context; no authenticated local execution path. |

## Promotion, retirement, and claim rules

1. **Answer-channel gate:** Exp5719 promotes only with authenticated CUDA, at least two
   qualified mandated models, 100% positive-control parse success, zero missing/truncated
   controls, zero validator disagreements, and a frozen non-grammar protocol.
2. **Attested stream gate:** Exp5720 promotes only with zero missing rows, zero parse
   failures, zero validator disagreements, verified commitments, and exact-validator
   receipts for every admitted envelope.
3. **Continuous self-learning gate:** Exp5721/Exp5722 may update only an external
   rollback-capable controller/KAN sidecar. They require zero unsafe accepts/updates,
   retention within margin, and exact crash/restart/rollback replay. GGUF weights and the
   production default remain unchanged.
4. **Rust backend gate:** Exp5723 must preserve exact semantics and fallback. Exp5724 may
   report a software speedup only when matched quality/work pass and the preregistered
   confidence interval excludes 1.0. Otherwise record the terminal null.
5. **ARC mechanism gate:** Exp5726 promotes the ledger only for a matched benefit with no
   level regression, unsafe commit, leakage, or off-path behavior. A null does not block
   Exp5727.
6. **ARC solve credit:** only a registry-prechecked new level with
   `solve_provenance=live_agent_self_discovery`, reproduced from the live agent's own
   attempt/runtime-RE trace, changes the solve registry or north-star.
7. **Failed reruns:** every matching carry-forward has all four `prior_failures` fields.
   If it repeats the same terminal verdict, `retire_if_same_verdict: true` activates the
   mechanical retirement rule.
8. **No claim inflation:** missing, gate-skipped, blocked, or malformed artifacts never
   count as successful work. Offload is not model quality; portability is not integration;
   integration is not speedup; known-level reproduction is not a new ARC solve.

## Expected outputs

- one terminal transition artifact with the Exp5709 narrow retirement applied;
- one bounded execution-time source-delta artifact;
- one answer-channel forensics report across all three mandated GGUF families;
- one sealed, exact-validator-attested flagship SOTA stream;
- two FR-11 lifecycle/rollback artifacts satisfying the continuous self-learning floor;
- one production Rust `SamplerBackend` integration and one matched crossover artifact;
- one ARC epistemic-ledger qualification, one matched live A/B, and one unconditional
  live self-discovery attempt;
- one capstone that reconciles code, specs, traceability, operations, exclusions, and E2E
  evidence without changing negative verdicts.


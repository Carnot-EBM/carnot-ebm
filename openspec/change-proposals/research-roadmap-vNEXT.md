# Research Roadmap V560: Trajectory Energy, Anytime Factor Memory, and ARC Decision Alignment

**Milestone:** `2026.08.560`
**Created:** 2026-08-21
**Status:** proposed
**Supersedes:** completed milestone `2026.08.559`
**Execution manifest:** `research-roadmap-next.yaml`

## Executive summary

V559 closed the forced-candidate hidden-representation branch honestly. Its receipt and transition contracts worked, but the prospective representation corpus did not support a scientific selector: candidate identifier length, candidate identity, and row-order parity each predicted the label perfectly, while token, prompt, and candidate lengths were absent from the raw evidence. That result is a lineage boundary, not a request to tune another hidden-state head.

V560 moves to a different observable: exact solver trajectories. It asks whether early solver state predicts which constraints and assignments persist to a final exact solution, whether SOTA local models can propose atomic reusable factors whose causal value survives exact counterfactual replay, and whether an anytime-valid bounded factor pool can learn continuously without eroding future support. A separate ARC branch first measures whether the existing conservative prefix energy aligns with later live-agent progress; only a positive diagnostic may activate a default-off policy A/B.

The milestone has 14 experiments in four phases:

1. commit a leakage-resistant solver-trajectory stream and evaluate early-to-final energy;
2. use exact replay to validate model-proposed factors and route exact checks under charged budgets;
3. execute restarted reuse/spawn/defer continuous learning with bounded capacity and future-support audits;
4. diagnose ARC decision alignment, gate a live policy A/B, and independently reconcile the milestone.

No task depends on FPGA, TSU, Kona, remote APIs, or unavailable model weights. The dual RTX 3090 host is the required accelerator for the single local-GGUF proposal task; the exact-solver, controller, audit, and ARC tasks remain runnable on the current CPU/GPU stack.

## What V559 proved

| Experiment | Durable result | V560 consequence |
|---|---|---|
| Exp6483 | The local SOTA source map and evidence boundary were complete. | Preserve source and model provenance; do not confuse literature support with local evidence. |
| Exp6484 | A non-generation representation receipt can bind model, runtime, raw bytes, prompt/candidate identity, and checksums. | Reuse the receipt discipline for any local-GGUF proposal event. |
| Exp6485 | Cache transition, event-time action, multiplicity, restart, rollback, and held-future evidence have an executable contract. | Extend this contract with paired restarted reuse/spawn evidence processes. |
| Exp6486 | Three exact-label families produced 432 prospective raw records. | The infrastructure can commit prospective evidence, but the rows are not selector-eligible. |
| Exp6487 | Readiness was 0. Three shortcuts perfectly predicted labels; required length evidence was unavailable. | Retire the forced-candidate hidden-representation selector scope. Do not headline, repair, or silently reuse those rows. |

The strongest adjacent completed results remain relevant:

- Exp6478 established exact finite energy selection with a positive lower confidence bound and zero harmful flips.
- Exp6479 provided a factor-cache adapter.
- Exp6480 froze the failed Exp6463 fixed-policy lineage while allowing genuinely prospective replacement lineages.
- Exp6481 defined phase receipts, and Exp6482 committed a 48-unit prospective exact stream across quota allocation, route ordering, and Boolean guards.
- Exp6471 produced a generic default-off ARC safety shield reachable by the live agent, but Exp6458 showed that a broad policy claim without direct objective alignment was premature.
- Exp6472 found continuous-learning and ARC engineering components present while scientific and hardware conclusions remained unearned.

## The three largest gaps to the PRD vision

### Gap 1: exact energy lacks a leakage-resistant learned signal on an authentic constraint process

FR-12 requires verifiable reasoning, not a scorer whose apparent accuracy comes from answer position, identifier form, or corpus construction. Exact finite energy works, but V559 showed that forced candidates were a poor scientific substrate. V560 therefore learns only from early states emitted by exact solvers and labels them by persistence to the final exact solution. The final solver remains the oracle; learned energy may prioritize work but never release an answer.

### Gap 2: FR-11 lacks executed, anytime-valid continuous self-learning

Carnot has transition and cache contracts, but it has not yet shown an online factor pool making reuse/spawn/defer decisions over a chronological stream while controlling false actions across time, factor multiplicity, and restarts. V560 must execute that lifecycle, charge updates fairly, bound capacity, support rollback, and test whether current gains reshape future support adversely.

### Gap 3: ARC energy is not yet a validated live-agent decision metric

The generic safety shield is reachable, but the prior broad ARC objective did not clear its roster. Before changing policy, V560 measures whether prefix energy predicts later live-agent progress beyond simple controls on a frozen public roster. A policy A/B runs only if that direct alignment gate is positive. No offline ground-truth BFS, game-source reading, or hand-built per-game adapter can receive solve credit.

Hardware acceleration remains strategically important, but it is not one of this milestone's three executable gaps: the local dual-GPU system is operational, while Extropic access, Kona weights, and FPGA integration evidence are unavailable or terminal for the present branch.

## Research findings adopted for V560

The 2026-08-21 literature refresh is recorded in `research-references.md`. The design adopts the following ideas without treating paper claims as local results:

- **Evidence Before Expansion** ([arXiv:2608.19888](https://arxiv.org/abs/2608.19888)) frames reuse, spawn, and defer as two one-sided sequential tests separated by an indifference zone, with restarted anytime-valid evidence. Exp6495 adapts this to factor-memory actions.
- **Learning Early-to-Final Solution Consistency for MILP** ([arXiv:2608.19953](https://arxiv.org/abs/2608.19953)) predicts whether early assignments survive to a final solve rather than guessing final assignments from static instance features. Exp6489-6490 use that boundary for exact solver trajectories.
- **Credit Without Ground Truth** ([arXiv:2608.19760](https://arxiv.org/abs/2608.19760)) reports that judge scores, log probabilities, and confidence are unreliable substitutes for executed causal replay and warns about dose confounding. Exp6492 and Exp6496 use exact add/drop replay and matched admitted-event counts.
- **Stopping and Routing LLM Judge Panels** ([arXiv:2608.19802](https://arxiv.org/abs/2608.19802)) separates copies, complements, and specialists under held cost. Exp6494 applies the complementarity question to exact checkers, not to model judges.
- **Verifier-Induced Support Reshaping** ([arXiv:2608.00220](https://arxiv.org/abs/2608.00220)) shows that immediate verifier gains can reduce future support. Exp6496-6497 measure held-future quality and best-of-k support after online admission.
- **ROC-n-reroll** ([arXiv:2507.12399](https://arxiv.org/abs/2507.12399)) cautions that low-budget verifier results need not predict high-budget selection. Exp6493-6494 report charged budget curves rather than a single operating point.
- **Bandit-MoE** ([OpenReview forum](https://openreview.net/forum?id=o7HKyfd5mk)) motivates bounded expert pools because small KAN pools can resist forgetting while larger pools can become difficult to manage. Exp6497 treats capacity as an experimental variable.
- **EVADE** ([arXiv:2608.18833](https://arxiv.org/abs/2608.18833)) motivates independent views and abstention. V560 uses independent exact replay and defer decisions rather than self-certification.
- **Orthogonal JEPA** ([arXiv:2608.20065](https://arxiv.org/abs/2608.20065)) is retained as a Tier-3 predictive-verification direction, but V560 does not schedule another latent-representation experiment after the V559 integrity failure.
- Extropic's Z1 update describes future high-connectivity probabilistic hardware, but availability is announced for 2027. It is architectural context only, not a local hardware claim.

## Target architecture

```text
 prospective exact instances (Exp6482 lineage)
                    |
                    v
       exact solver trajectory recorder
       early state ----------> final exact solution/oracle
                    |                    |
                    +---- persistence ---+
                              labels
                                |
                 +--------------+---------------+
                 |                              |
                 v                              v
      leakage-neutral trajectory       local SOTA GGUF models
      energy (linear/MLP/KAN)           propose atomic factors only
                 |                              |
                 |                     exact compile + add/drop replay
                 |                              |
                 +--------------+---------------+
                                v
                    decomposed exact-grounded energy
                                |
                    exact checker VOI router
                                |
                 full exact solver remains authority
                                |
                                v
          chronological restarted reuse/spawn/defer controller
          bounded pool | rollback | multiplicity | future support

 frozen public ARC prefixes --> alignment diagnostic
                                |
                      positive alignment gate?
                         no /          \ yes
                    report null      default-off live A/B
                                          |
                               generic shield + live agent
                               exact/public game feedback
```

The architecture has four hard boundaries:

1. local models may propose factors but cannot label, verify, or release solutions;
2. exact trajectory outcomes and exact counterfactual replay are the scientific authority;
3. continuous updates are chronological, default-off until admitted, and evaluated on future held units;
4. ARC solve provenance, if any level is incidentally reached, must be `live_agent_self_discovery`.

## Phase A: evidence boundary and solver-trajectory substrate

### Exp6488 - V559 decision ledger and V560 lineage lock

Aggregate Exp6483-6487 from their raw evidence, freeze the invalid representation-selector scope, and state the allowed V560 lineage. The deliverable must distinguish reusable infrastructure from scientifically disqualified rows and give downstream tasks an exact `v560_lineage_lock_ready_score` field.

**Deliverable:** `results/experiment_6488_v559_decision_ledger.json`

### Exp6489 - Immutable early-to-final solver trajectory commitment

Instrument the exact backends already used by the Exp6482 three-family commitment. Persist early states, chronological decisions, final exact outcomes, identity-free features, split commitments, checksums, and a leakage audit before fitting any model. Labels express persistence to the final solution; they are not generated by an LLM.

**Deliverable:** `results/experiment_6489_solver_trajectory_commitment.json`

### Exp6490 - Leakage-neutral early-to-final trajectory energy baselines

Compare analytical early energy, regularized linear, compact MLP, and compact KAN persistence heads on committed held trajectories and seeds. Audit instance identity, ordering, serialized length, norm, backend, and family shortcuts. Exp5853 and Exp6487 are explicit prior failures; a repeated no-signal or shortcut result retires this learned-trajectory branch.

**Deliverable:** `results/experiment_6490_trajectory_energy_baselines.json`

### Exp6491 - Local-SOTA atomic factor proposal stream

Use the mandated local GGUF models to propose one atomic, compilable factor at each development-only divergence event. Raw request/response bytes and model receipts are immutable. The model does not emit answers, see held labels, use a constrained answer grammar, or retry until accepted. Exact compilation may reject proposals. This is a new factor-proposal scope, not a rerun of the retired Exp6463 answer-policy corpus.

**Deliverable:** `results/experiment_6491_sota_factor_proposal_stream.json`

## Phase B: exact-grounded factor energy and checker routing

### Exp6492 - Executed-replay factor causal audit

For every accepted proposal, execute exact add/drop counterfactual replays on solver trajectories. Match random and structural-control factors, event counts, and exposure dose. Report per-event changes in exact search work, final validity, and harmful flips. No model-based judge may substitute for executed replay.

**Deliverable:** `results/experiment_6492_factor_causal_replay.json`

### Exp6493 - Gated decomposed trajectory-energy A/B

Run only if Exp6490 finds a held trajectory signal and Exp6492 finds positive causal factor value. Compare exact analytical energy, learned persistence energy, decomposed energy, and matched controls across charged budget points. Full exact solving remains authority; the outcome is search-work reduction with validity parity, not a hardware-speed claim.

**Deliverable:** `results/experiment_6493_decomposed_trajectory_energy_ab.json`

### Exp6494 - Gated exact-checker complementarity and value-of-information routing

Run only if Exp6493 establishes usable decomposed energy. Measure which exact checks behave as copies, complements, or specialists, including authentic runtime cost. Compare always-all, uniform, static, and value-of-information routing across fixed charged budgets. False release is forbidden.

**Deliverable:** `results/experiment_6494_exact_checker_voi_router.json`

## Phase C: continuous self-learning

### Exp6495 - Restarted reuse/spawn/defer factor-pool controller

Extend the Exp6479 cache adapter and Exp6485 transition contract with paired one-sided restarted evidence processes and an explicit indifference zone. Implement event-time spending, factor/restart multiplicity, bounded capacity, rollback, defer, and deterministic corruption fixtures. This is a contract and implementation task, not yet a learning-effect claim.

**Deliverable:** `results/experiment_6495_restarted_factor_pool_controller.json`

### Exp6496 - Chronological continuous factor learning

Execute the immutable proposal stream chronologically. Compare frozen, always-update, fixed-threshold, and restarted reuse/spawn/defer arms with matched admitted-event counts and exposure dose. Exact replay controls every write. Evaluate future held quality, best-of-k support, safety, pool actions, rollback, and restart behavior. This is the milestone's required continuous self-learning experiment.

**Deliverable:** `results/experiment_6496_continuous_factor_learning.json`

### Exp6497 - Bounded-capacity recurrence and support stress

Stress small and larger pool capacities under recurring, shifted, and corrupted factor streams. Measure negative transfer, eviction quality, recovery after restart, future support, and validity. A null or adverse result is valid evidence and constrains the production design.

**Deliverable:** `results/experiment_6497_factor_pool_support_stress.json`

### Exp6498 - Independent continuous-learning replay audit

Using an independent reducer, recompute chronological actions, evidence spending, multiplicity, matched dose, rollback, and future-support results from raw rows. The audit must distinguish execution completeness from eligibility for a continuous-learning scientific claim.

**Deliverable:** `results/experiment_6498_csl_independent_audit.json`

## Phase D: ARC decision alignment and capstone

### Exp6499 - ARC prefix-energy-to-progress alignment diagnostic

On a frozen public roster, correlate conservative prefix energy with later live-agent progress and compare against step count, action count, validity, and simple state-size controls. Perform the solve-registry precheck. This task changes no policy and claims no new game or level solve; its purpose is to establish whether energy is a useful decision metric.

**Deliverable:** `results/experiment_6499_arc_energy_progress_alignment.json`

### Exp6500 - Gated default-off live ARC policy A/B

Run only if Exp6499's held alignment gate is positive. Compare the generic energy shield off and on under matched games, seeds, and budgets using the reachable live-agent path. Measure progress, invalid actions, regressions, cost, and any incidental level reachability. Any solve provenance must be `live_agent_self_discovery`; source inspection, offline BFS, and per-game adapters are prohibited.

**Deliverable:** `results/experiment_6500_arc_live_energy_policy_ab.json`

### Exp6501 - V560 independent capstone and V561 handoff

Recompute every headline from per-unit rows, verify gates and frozen-lineage exclusions, classify blocked versus valid-null outcomes, and report closure against the three gaps. Reconcile research, spec, traceability, status, and changelog documents. This task runs even when a scientific gate closes.

**Deliverable:** `results/experiment_6501_v560_capstone.json`

## Dependency graph

```text
6488 lineage lock
 |-- 6489 trajectory commitment
 |    |-- 6490 trajectory signal -----------+
 |    |-- 6491 SOTA factor proposals        |
 |         `-- 6492 exact causal replay ----+--> 6493 decomposed A/B
 |                                                 `-- 6494 VOI router
 |
 |-- 6495 reuse/spawn/defer controller
 |    `-- 6496 continuous learning
 |         `-- 6497 capacity/support stress
 |              `-- 6498 independent CSL audit
 |
 `-- 6499 ARC alignment
      `-- [positive gate] 6500 live ARC policy A/B

6488..6500 ---------------------------------------> 6501 capstone
```

Structured gates in `research-roadmap-next.yaml` are authoritative. Each gated field is named identically in the upstream task's required artifact fields. Exp6501 is deliberately ungated so a closed branch still receives a capstone record.

## Hardware and model requirements

| Resource | Tasks | Requirement |
|---|---|---|
| Dual RTX 3090, 48 GB aggregate VRAM | Exp6491 | Load one local GGUF at a time through `llama_cpp`; prefer GPU offload and record actual layers/backend. |
| Mandated local models | Exp6491 | Include `unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and `unsloth/gemma-4-26B-A4B-it-GGUF` in `MODEL_SPECS`; at least two distinct SOTA model families must produce evidence. |
| CPU and existing exact backends | Exp6489-6490, Exp6492-6498 | Exact solver traces, replays, sequential evidence, and audits. Record real wall time but make no special-hardware speed claim. |
| Existing ARC live runtime | Exp6499-6500 | Frozen public games, live-agent actions, generic shield, and registry precheck. |
| FPGA / KV260 / ZCU104 | none | Not required; no milestone claim depends on unavailable bring-up. |
| Extropic TSU / Z1 | none | No authenticated local access; paper and vendor claims remain external provenance only. |
| Kona | none | No public runnable weights or local inference path assumed. |

For local GGUF inference, `llama_cpp` is the primary substrate and its embedded tokenizer is authoritative. `transformers.AutoTokenizer` must not be used against a GGUF repository. Legacy Qwen3.5-0.8B and gemma-4-E4B models may appear only as labeled CPU smoke checks and cannot support a headline.

## Evidence, safety, and retirement rules

- Every comparison emits per-unit rows sufficient to recompute its headline without rerunning the experiment.
- Every artifact records `inference_substrate`, `verifier_is_oracle`, field provenance and principles, random seed, checksum, duration, tests, preconditions, protected-file checks, and `gate_check_summary`.
- A blocked artifact uses `honest_verdict: blocked_*` and names the exact failed check and observed value in `gate_check_summary`.
- Gated tasks do not reinterpret a missing artifact or missing field as a scientific null.
- Exact solvers and exact replay are authorities. Learned energy, SOTA proposals, and cache actions are advisors.
- Exp6490 carries the Exp5853 and Exp6487 prior failures. If the changed trajectory substrate produces the same no-signal/shortcut verdict, the branch retires mechanically.
- Exp6491 carries Exp6463 as a prior failed scope and is permitted only because the deliverable has changed from answer-policy generation to atomic factor proposals on a new prospective solver stream.
- Exp6493 and Exp6494 carry the blocked Exp6464/Exp6466 ancestry and must retire on the same verdict.
- Exp6499 and Exp6500 carry Exp6458. They change the scope by measuring direct alignment before policy intervention and by relying on the shipped generic live-path shield.
- ARC tasks run `ops/arc_solve_registry.yaml` and live-path prechecks before execution. They do not duplicate existing level solves.
- Tests and probes may not leave scratch files in the repository root or mutate unrelated tracked files.

## Spec-anchored execution contract

Before implementation work, each task must read `CODEX.md`, `CLAUDE.md`, the relevant capability spec, the upstream artifacts named in its prompt, and the cited existing modules/tests. If a required `openspec/capabilities/*/spec.md` or `REQ-*` anchor is absent, add or amend the spec before implementation and then add failing tests.

Before reporting completion, each task must run its focused tests, the applicable lint and spec-coverage checks, relevant commands from `ops/e2e-test-plan.md`, and `git status --short`. Implementation-bearing tasks reconcile the applicable OpenSpec, `_bmad/traceability.md`, `ops/status.md`, and `ops/changelog.md`. No task may modify `research-roadmap.yaml` or `scripts/research_conductor.py`, and no task may push.

## Acceptance criteria

V560 is complete when all 14 task artifacts exist and the capstone can verify their gates and rows. Scientific success is deliberately separable from execution success:

- **Trajectory-energy success:** Exp6490 finds held early-to-final signal that survives shortcut controls, and Exp6492 finds positive exact causal factor value without harmful flips.
- **Routing success:** gated Exp6493 reduces charged exact search work with validity parity, and Exp6494 improves cost-adjusted checking without false release.
- **Continuous-learning success:** Exp6496 completes chronological learning and improves future held utility without safety or support loss; Exp6497 and Exp6498 confirm capacity, recurrence, sequential evidence, and row integrity.
- **ARC success:** Exp6499 finds held energy-to-progress alignment, then Exp6500 improves the frozen live objective without regression and with credited live-path provenance.
- **Valid negative outcome:** any clean null, closed gate, support-loss finding, or retired branch is accepted if the contract, rows, provenance, and verdict remain complete.

## Explicit non-goals

- repairing or reusing the V559 forced-candidate representation corpus for a headline;
- training a free-form answer generator or treating an LLM judge as an oracle;
- claiming hardware acceleration from software wall-clock timing;
- FPGA or thermodynamic-hardware bring-up without authenticated access;
- public deployment, remote service calls, or pushing changes;
- ARC source reading, offline exhaustive ground-truth BFS, per-game adapters, or duplicate level solves;
- claiming autonomous continual learning from a controller fixture alone.

## Expected V561 decision boundary

The capstone should leave one of four explicit handoffs:

1. **trajectory signal and continuous learning both positive:** scale the exact-grounded factor pool to a larger prospective constraint stream and begin a production-shadow integration plan;
2. **trajectory signal positive, online learning unsafe or support-eroding:** freeze the learned scorer and research admission/capacity controls rather than scaling updates;
3. **trajectory signal null after integrity passes:** retire compact learned energy for these families and invest in exact structural features or a new task distribution;
4. **ARC alignment null or policy gate closed:** retain the generic safety shield only and stop treating the current energy as a progress objective.

Hardware work becomes a next milestone only when an authenticated device or runnable public substrate changes the execution boundary.

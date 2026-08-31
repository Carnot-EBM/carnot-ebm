# Carnot Research Roadmap vNEXT: Verified Action Memory and Live Causal Control

**Created:** 2026-08-31  
**Milestone:** 2026.08.595  
**Status:** Planned. Activate after milestone 2026.08.594 is terminal.  
**Supersedes:** The 2026.08.594 plan in this path.  
**Experiment range:** Exp6810-Exp6823  
**Informed by:** Exp6776-Exp6779, Exp6797-Exp6809, Selective Value-Filtered
Decoding (`2605.14746`), RCML (`2606.07088`), AdaMEM (`2606.05684`), VerMem
(`2608.03137`), CEDAR (`2608.27797`), PLVR (`2608.28421`), and xTRUCE
(`2608.28532`).

## What 2026.08.594 Proved

| Result | Experiment | Finding |
|---|---:|---|
| The deterministic prechecks work | 6802 | The v2 automaton task verified its code, source artifacts, registry, and runtime inputs before implementation. |
| The roadmap named the wrong spec owner | 6802 | The task required `openspec/capabilities/agentic-verification/spec.md`. That file does not exist. The readiness field was correctly set to false. |
| Structured gates fail closed | 6803 | The conductor read `operational_automaton_fixture_ready=false` and wrote a terminal blocked artifact before model load. |
| One bad root gate erased the active milestone | 6804-6809 | The conductor preemptively skipped six tasks after the root branch retired. No arbitration, route-learning, portability, or cold-audit science ran. |
| The plan and active manifest diverged | Milestone | The vNEXT document described 13 tasks through Exp6814. The activated YAML contained eight tasks through Exp6809. The live ARC and disposition phases never entered the execution manifest. |

Milestone .594 did not test its scientific claims. It exposed two planning defects: the spec path
had no owner, and the active manifest did not match the design. Those defects are now explicit
inputs to V595. They do not count as negative evidence for the priority arbiter, typed route
learning, or the live ARC mechanism.

## The Three Largest Gaps to the PRD Vision

### Gap 1: exact verification does not yet improve authentic actions

FR12 requires a constraint layer that changes behavior and keeps exact authority. Carnot has
deterministic validators and authentic local-model outputs. It has not shown that an action arbiter
can prevent hard violations, retain already-safe actions, and improve progress or retry cost.

### Gap 2: continuous self-learning is causal but not authentic and portable

FR11 has a byte-replayable transactional learner. Its strongest causal result still comes from a
frozen routing stream. Carnot has not shown continuous, verifier-gated improvement on authentic
mandated-GGUF proposals. It also lacks cross-model portability and a global cold memory audit.

### Gap 3: live ARC mechanisms have transport receipts but no causal progress result

The production agent can emit selfparse tool calls. The supervisor can write refinement receipts.
The last window-120 accrual blocked on an occupied GPU. The tool-gap and actions-to-progress tasks
then gate-blocked. Carnot still needs a process-isolated, row-complete A/B on the live path.

## Planning Repair Before Research

V595 does not create another vague capability name. It maps each contract to an existing owner:

| Contract | Existing owner |
|---|---|
| Live operational obligations, action seam, process isolation | `openspec/capabilities/agentic-harness/spec.md` |
| Exact priority arbitration and conflict certificates | `openspec/capabilities/constraint-verification/spec.md` |
| Transactional route memory and rollback | `openspec/capabilities/continuous-learning/spec.md` |

Exp6810 verifies these paths, writes the missing REQ anchors, and proves that the planned task set,
experiment range, gates, and deliverables match the activated YAML. Exp6811 then reruns the blocked
automaton scope against those real owners. This separates contract repair from the scientific run.

## Research Delta

The new source sweep changes four parts of the design.

- Selective Value-Filtered Decoding makes false intervention a first-class metric. The arbiter must
  preserve safe proposals as well as reject unsafe ones.
- RCML separates current correction pressure from finite-gain memory. The route learner will compare
  residual-pressure updates with raw violation accumulation. Stale pressure must decay after
  feasibility returns.
- AdaMEM uses transient stepwise strategy state. The live ARC branch will compare static
  episode-start guidance with stepwise read-only strategy retrieval. It will not persist writes
  during the active episode.
- VerMem combines local transition checks with global memory checks. Carnot will expose a small
  typed memory-operation set. Exact local receipts control commits. A fresh-process global audit
  checks terminal coherence, rollback, retention, and poison behavior.

These ideas extend the V594 pivot. They do not reopen learned text-energy repair. They keep the
frozen GGUF models as generators and keep exact validators as release authorities.

## V595 Architecture

```text
Existing OpenSpec owners
 agentic-harness │ constraint-verification │ continuous-learning
        └──────────────────┬───────────────────────┘
                           ▼
             owned REQ map + manifest preflight
                           │
                           ▼
        typed obligation schema + exact event automaton
                           │
            ┌──────────────┴────────────────┐
            │                               │
            ▼                               ▼
  Three mandated GGUF families       Production E3AgentPolicy
  structured action proposals        live attempts, no adapters
            │                               │
            ▼                               ▼
  exact selective priority arbiter   stepwise read-only strategy
  hard > binding > soft              selfparse tool-gap transport
  conflict + no-op certificates      exact next-action outcomes
            │                               │
            ▼                               ▼
  typed verified memory operations   actions-to-progress A/B
  local receipt before commit               │
  residual-pressure update                  │
  between-episode transaction               │
            └──────────────┬────────────────┘
                           ▼
             independent fresh-process audits
                           │
                           ▼
              adopt │ narrow │ retire │ blocked
```

The exact automaton and exact post-action receipts remain authoritative. The learned external
program may rank legal proposals or select legal memory operations. It may not override a hard
constraint. The live agent must discover progress from its own attempts and runtime reverse
engineering.

## Phase 1: Contract Recovery and Selective Action Arbitration

### Exp6810: Owned-spec and execution-manifest recovery preflight

Audit the .594 failure, map every planned contract to an existing capability spec, and add the
needed REQ anchors. Verify that V595 has exactly 14 tasks, Exp6810 through Exp6823, with unique
deliverables and resolvable gate fields. This task writes no scientific claim.

**Deliverable:** `results/experiment_6810_v595_contract_manifest_preflight.json`

### Exp6811: Typed operational-obligation automaton v3

Recover Exp6802 against the owned specs. Add the canonical prerequisite, authority, fallback,
consequence, and priority record to the existing supervisor. Compile it to a deterministic event
automaton and exact lexicographic energy. Prove v1 compatibility, canonical replay, fail-closed
attacks, and exact field readiness on frozen source-free traces.

**Deliverable:** `results/experiment_6811_operational_obligation_automaton_v3.json`

### Exp6812: Three-model operational-handoff proposal corpus v2

Invoke all three mandated local GGUF families. Generate paired proposals under direct typed and
length-matched compressed handoffs. Freeze prompts, model files, seeds, raw bytes, and exact checks.
Measure operational preservation, hard violations, parse completion, safe-proposal headroom, and
retry demand. This is not a live ARC solve.

**Deliverable:** `results/experiment_6812_sota_operational_handoff_corpus_v2.json`

### Exp6813: Selective priority arbiter A/B

Compare the exact priority arbiter with flat reject-and-retry under matched candidate, check, and
wall budgets. Add a frozen no-intervention path for already-valid proposals. Measure hard-violation
prevention, accepted progress, retry cost, false intervention, safe-action identity, certificate
coverage, abstention, and latency. Completion does not require a positive effect.

**Deliverable:** `results/experiment_6813_selective_priority_arbiter_ab.json`

### Exp6814: Independent selective-arbiter authority audit

Reparse raw bytes and replay both arms in a fresh module. Recompute every headline. Attack priority,
authority, stale prerequisites, missing fallback, consequence weakening, model identity, future
outcomes, and unnecessary intervention accounting. Separate hard safety from utility.

**Deliverable:** `results/experiment_6814_selective_priority_arbiter_cold_audit.json`

## Phase 2: Verifier-Gated Continuous Self-Learning

### Exp6815: Chronological verified-memory operation stream

Convert authentic proposal and arbiter receipts into a chronological stream. Represent route memory
with typed add, revise, soft-delete, retrieve, filter, and restore operations. Seal future outcomes.
Freeze development, held-future, hard-case, and leave-one-model-family-out splits. Require headroom,
legal alternative routes, and both admitted and rejected operations.

**Deliverable:** `results/experiment_6815_verified_memory_operation_stream.json`

### Exp6816: Residual-pressure transactional route learning A/B

Compare four equal-capacity arms: frozen memory, residual-pressure route learning, raw violation
accumulation, and random valid updates. Make the active episode read-only. Commit only between
episodes after an exact local receipt. Store canonical parent and new bytes. Roll back support,
retention, or hard-case harm. Credit only later action changes with exact utility witnesses.

**Deliverable:** `results/experiment_6816_residual_pressure_route_learning_ab.json`

### Exp6817: Leave-one-model-family-out route-memory portability

Learn on two mandated model families and evaluate the third, then rotate. Freeze program bytes
before held-family outcomes. Compare residual-pressure, frozen, and raw-accumulation arms. Measure
exact utility, reachable support, constraint preservation, retention, cost, and stale-pressure
release for every rotation.

**Deliverable:** `results/experiment_6817_route_memory_portability.json`

### Exp6818: Independent local-and-global route-memory audit

Cold-decode every transaction and operation. Recompute local transition validity and global terminal
memory coherence. Remove each credited write and disable later retrieval. Attack poison, future
leakage, stale parents, cross-family labels, duplicate operations, restart, rollback, capacity, and
aggregate row deletion.

**Deliverable:** `results/experiment_6818_route_memory_global_cold_audit.json`

## Phase 3: Live ARC Stepwise Strategy and Causal Control

### Exp6819: Stepwise read-only strategy-supervisor accrual

Reuse compatible durable external-evaluation rows first. Then run only the bounded top-up needed for
the frozen evidence floor. Compare episode-start static guidance with stepwise read-only strategy
retrieval in shadow mode. Preserve action hashes in shadow rows. Active rows need exact next-action
outcomes. Use the production Qwen3.6 flagship MoE.

**Deliverable:** `results/experiment_6819_arc_stepwise_strategy_accrual.json`

### Exp6820: Live selfparse tool-gap obligation transport v2

Run one production selfparse cell and a typed canary. Prove that all operational fields and strategy
receipts survive live serialization, analyzer ingestion, and ledger update. Keep default-off behavior
unchanged. An empty natural event list is valid only when the typed canary proves the path.

**Deliverable:** `results/experiment_6820_arc_tool_gap_obligation_transport_v2.json`

### Exp6821: Obligation-routed actions-to-progress A/B v2

Run matched live control and treatment processes. The control has the tool loop unset. The treatment
uses selfparse, stepwise read-only strategy retrieval, and the audited selective arbiter. Match games,
seeds, actions, model, token limits, and censoring. Measure progress rate, restricted mean actions,
harm, false interventions, and hard violations. Make no game-level solve claim.

**Deliverable:** `results/experiment_6821_arc_obligation_actions_to_progress_ab_v2.json`

### Exp6822: Independent ARC causal adoption audit

Use fresh reducers to replay Exp6819 through Exp6821. Verify process isolation, exact outcomes,
source prohibition, adapter absence, solve provenance, ledger updates, row-derived headlines, and
protected files. Issue one closed decision for the supervisor, stepwise strategy, tool transport,
and action arbiter.

**Deliverable:** `results/experiment_6822_arc_causal_adoption_audit.json`

## Phase 4: Milestone Disposition

### Exp6823: V595 evidence synthesis and branch disposition

Read all 14 terminal artifacts, including blocked and disqualified results. Recompute claims from
rows. Compare the executed manifest with this plan. Decide the future of the selective arbiter,
verified route memory, and live ARC path. Reconcile OpenSpec, traceability, status, changelog, and
known issues. This task is deliberately ungated.

**Deliverable:** `results/experiment_6823_v595_branch_disposition.json`

## Dependency Graph

```text
Exp6810 owned-spec + manifest preflight
   └── Exp6811 obligation automaton v3
         ├── Exp6812 three-model handoff corpus
         │     └── Exp6813 selective-arbiter A/B
         │           └── Exp6814 cold authority audit
         │                 └── Exp6815 verified-memory stream
         │                       └── Exp6816 residual route learning
         │                             ├── Exp6817 portability
         │                             └──────────────┐
         │                                            ▼
         │                                      Exp6818 global cold audit
         │
         └── Exp6819 live stepwise-strategy accrual
               └── Exp6820 live tool-gap transport
                     └── Exp6821 actions-to-progress A/B ◄── Exp6814 audit
                           └── Exp6822 ARC adoption audit

All terminal results, including blocked branches ─────────► Exp6823 disposition
```

Every structured gate targets a field declared verbatim in the upstream task. Gates test contract,
artifact, or transport completeness. They do not require a positive scientific result. Exp6823 has
no gate, so a blocked branch still receives a terminal disposition.

## Acceptance Rules

- Every task writes its declared artifact on success, null, block, disqualification, or partial
  completion.
- Every comparative task emits one row for each model, scenario, arm, seed, order, game, operation,
  or attack needed to recompute the claim.
- Every artifact carries `verdict_class` with the closed enum `positive | circular_positive | null |
  blocked | disqualified | partial` next to `honest_verdict`.
- Every blocked verdict records the failed check, expected value, and observed value in
  `gate_check_summary`.
- Every LLM task includes at least one mandated local GGUF in `MODEL_SPECS`. Exp6812 uses all three.
  Exp6819 through Exp6821 use `unsloth/Qwen3.6-35B-A3B-GGUF`.
- Legacy small models may run CPU smoke tests only. Their rows cannot support a headline.
- GGUF tasks use llama.cpp embedded tokenizer and native chat templates. They never send a GGUF
  repository to `AutoTokenizer`.
- Exact validators remain external release authorities. A learned score, route program, memory
  policy, or same-model vote may not certify itself.
- The route-memory learner keeps model weights frozen. It commits external state only between
  episodes after exact receipts.
- Live ARC tasks run a registry precheck and record `solve_provenance`. Only
  `live_agent_self_discovery` could support a solve, but V595 makes no game-level solve claim.
- No task reads game source, runs offline ground-truth BFS, builds a per-game adapter, or re-solves a
  registry-complete level.
- Every repeated scope carries complete `prior_failures`. The same verdict retires the new attempt.

## Hardware Requirements

| Experiments | Substrate | Requirement | Estimate |
|---|---|---|---:|
| 6810-6811 | CPU | Existing specs, supervisor, frozen traces, schema and manifest checks | 1-3 h each |
| 6812 | Dual RTX 3090, sequential phases | Local llama.cpp CUDA; Qwen3.6-35B-A3B, Gemma-4-31B, Gemma-4-26B-A4B; task-owned leases | 4-8 h |
| 6813-6815 | CPU | Frozen raw bytes, exact simulators, independent reducers | 2-4 h each |
| 6816-6818 | CPU | Transaction store, canonical bytes, chronological replay | 3-6 h each |
| 6819-6821 | RTX 3090 | Production Qwen3.6 GGUF, task-owned server, durable cell checkpoints, bounded top-up | 4-12 h each |
| 6822-6823 | CPU | Fresh-process audits and documentation reconciliation | 2-4 h each |

The two RTX 3090 boards are the only headline inference substrate. Tasks must not kill or reuse an
external operator process. They wait through the lease mechanism and stop with a terminal blocked
artifact if the bounded lease fails. KV260 and GateMate already have terminal receipts. PolarFire is
opportunistic. Extropic Z1 access targets 2027. No FPGA, ROCm, NPU, WebGPU, or TSU task is on the
V595 blocking path.

## Explicitly Deferred

- Another text-energy, fixed-point, or learned-score repair on the Exp6799 corpus.
- Finite-ID proof transport and grammar-decoding branches.
- Weight updates to any mandated GGUF model.
- Same-model verifier acceptance or neural replacement of exact certificates.
- New ARC level solves, game-source inspection, offline ground-truth search, or per-game adapters.
- New FPGA or thermodynamic-hardware claims without new physical access.
- Open-ended memory-schema search. V595 tests a frozen typed operation set.

## Milestone Exit

V595 is terminal when all 14 tasks have terminal artifacts and Exp6823 gives a closed disposition
for every branch. A positive product claim additionally needs one row-derived, cold-audited result:

1. the selective arbiter preserves every hard obligation, stays within the false-intervention bound,
   and improves accepted progress or retry cost;
2. residual-pressure route memory improves held-future exact utility without support, retention, or
   hard-case harm and survives byte, poison, restart, rollback, and portability audits; or
3. the live ARC treatment improves actions-to-progress with zero hard-obligation harm under
   process-isolated `live_agent_self_discovery` provenance.

A clean null or a diagnosed resource block is valid evidence. It narrows or retires the branch. It
does not become a positive claim.

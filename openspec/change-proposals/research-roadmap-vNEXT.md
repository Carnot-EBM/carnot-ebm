# Research Roadmap vNEXT: Prospective ARC Belief Memory and Live Policy Utility

**Milestone:** `2026.09.615`

**Status:** Proposed

**Task contract:** exactly 12 tasks, `exp7016` through `exp7027`, in the order in this document

**Executable contract:** `research-roadmap-next.yaml`

**North star:** improve the live ARC-AGI-3 discovery agent's accuracy or action efficiency without game source, offline ground-truth search, or per-game adapters

## 1. Executive Summary

Milestone `2026.09.614` finished all seven tasks in its executable YAML. It
also exposed a planning defect: the Markdown promised 14 tasks while the YAML
contained only seven. Exp7009 detected the mismatch and was disqualified.

The scientific result was also decisive. Exp7012 built 48 exact minimal
intervention pairs across four constraint families. Exp7013 scored all 576
model-by-pair rows with the three required GGUF families. The signed response
was null. Exp7014 found no causal feature bank that met its cold release rules.
Exp7015 was therefore gate-blocked. V615 does not rerun that latent-energy
line.

V615 restores the ARC belief branch that the stale V614 document described
but its executable YAML never ran. It builds a small, explicit belief ledger
from the live agent's own chronological observations. A contradiction becomes
a counterexample. The ledger updates only after the next observation. The
milestone then asks the load-bearing question: does that memory improve later
decisions under a fixed action and model budget?

The milestone contains 12 tasks. It has three independent safeguards:

1. A contract task checks exact Markdown/YAML parity but does not gate the
   science branch.
2. A compute-receipt task records task-linked phase, GPU, model, and runner
   evidence before the live comparison.
3. A cold audit can block live wiring without blocking the capstone.

## 2. What V614 Proved

V614 proved five facts that govern this plan.

1. The executable milestone was seven tasks, `exp7009` through `exp7015`.
   The stale 14-task Markdown contract was false.
2. The ARC evaluation provenance builder and consumer now agree on one
   schema. Exp7010 produced a stable round-trip hash and rejected malformed
   fixtures.
3. The exact intervention fixture is usable. It contains 48 balanced minimal
   pairs across four exact authority families.
4. All three mandated local GGUF families completed the intervention surface.
   The result was a terminal null, not an environment block.
5. No causal feature bank passed the cold release rules. Another
   pair-centered PWA-KAN fit would be an unchanged rerun.

The operational retrospective added one systems finding. Exp7013 used almost
all measured milestone time, but the artifact did not locate time inside
setup, model load, inference, write, and cleanup phases. It also lacked
task-linked GPU samples and a model-concurrency/runner receipt.

## 3. The Three Biggest Gaps

### Gap 1: The live agent simulates actions but has no queryable current-state belief

`E3AgentPolicy` rebuilds active world-model inputs from a post-level-boundary
transition suffix. It does not maintain a durable, explicit statement of what
is known, possible, contradicted, or uncertain. This is a direct gap between
the current architecture and the PRD's autonomous, adaptive constraint
reasoner.

### Gap 2: Continuous self-learning has storage safety but no proven future utility

Earlier self-learning tasks proved transactions, rollback, restart, retention,
and poison controls. They did not beat a read-only control on every sealed
future order. V615 changes the learned object from generated constraint memory
to game-blind ARC transition beliefs. It keeps the same safety bar and makes
later decision value the primary gate.

### Gap 3: The live path lacks a matched, provenance-complete belief-value result

The current ARC engine did not beat both inert and action-delta controls in
Exp7005. No result compares base, simulation-only, belief-only, and combined
live policies with the same model calls and action budget. No result ties
phase time and GPU use to those arms. V615 must produce that comparison or an
honest null.

## 4. Research Inputs

The full source record is in `research-references.md`, section "V615 Planner
Refresh - 2026-09-05". The main inputs are:

- BB-WM (`arXiv:2609.00455`): expose current-state belief as a separate policy
  interface and compare it with simulation.
- Counterexample Guided Learning in the Large (`arXiv:2606.11521`): cluster
  exact counterexamples and use them to revise symbolic hypotheses.
- Scaling Flaws of Verifier-Guided Search (`arXiv:2502.00271`): retain an
  equal-budget repeated-search control because a weak scorer can prune valid
  paths.
- When To Solve, When To Verify (`arXiv:2504.01005`): compare under fixed
  compute, not candidate count alone.
- KAC (`arXiv:2503.21076`) and the KAN forgetting study
  (`arXiv:2511.12828`): defer learned KAN compression until an explicit belief
  substrate has prospective value.

Extropic Z1T and Logical Intelligence Kona remain architecture references.
Carnot has no authenticated Z1/TSU runner and no public Kona checkpoint. V615
makes no hardware speed, power, or external-model performance claim.

## 5. Target Architecture

```text
                         provenance-complete live attempt
                                      |
                                      v
ARC observation --> chronological transition recorder --> immutable stream
      ^                                                       |
      |                                                       v
      |                                  counterexample cluster + tombstone
      |                                                       |
      |                                                       v
      |                  +----------------------> belief ledger
      |                  |                    known / possible /
      |                  |                    contradicted / uncertain
      |                  |                           |
      |                  |                           v
      |          simulation world model ----> belief query API
      |                  |                           |
      |                  +-------------+-------------+
      |                                v
      +--- selected action <--- E3AgentPolicy selector <--- SOTA GGUF proposer
                                       |
                                       v
                        task-linked phase/GPU/model receipts
```

The environment observation is delayed supervision. It is visible only after
the action. The updater may use past observations and past contradictions. It
must not use game source, hidden future rows, offline BFS, a hand-built game
adapter, or the answer that the current decision is being scored against.

The belief path stays default-off until the cold audit and live-path tests
pass. The existing production action path remains unchanged when the flag is
off.

## 6. Experimental Design

### Primary hypothesis

A compact counterexample-updated belief ledger gives the live agent useful
current-state information that action simulation alone does not provide.

### Primary outcomes

- Progress per 100 actions on held-mechanic live episodes.
- Actions to the next observed progress event.
- Invalid or contradicted action proposals.
- Model requests, completions, wall time, and GPU energy proxies per arm.

### Required controls

- Base live agent with no belief and no added simulation query.
- Recency-only memory with the same capacity as the belief ledger.
- Simulation-only policy.
- Belief-only policy.
- Combined belief plus simulation policy.

The offline prospective task uses frozen, recency-only, and belief-ledger
arms. The live task uses base, simulation-only, belief-only, and combined
arms. Every comparative artifact emits rows for each unit and arm. Pooled
results cannot override contradictory rows.

### Claim boundary

The belief ledger is oracle-distinct only if it uses information that existed
before the scored action. Later environment feedback may train the next
decision, but it may not score the current decision at selection time.
`verifier_is_oracle` must be false for a positive belief-value claim. Any
incidental game-level solve is creditable only with
`solve_provenance=live_agent_self_discovery` and a registry precheck.

## 7. Phases and Exact Task Contract

The following table is the complete task contract. It contains exactly the 12
tasks in `research-roadmap-next.yaml`. There are no aspirational tasks outside
this table.

| Order | Experiment | Title | Deliverable | Structured prerequisites |
|---:|---|---|---|---|
| 1 | `exp7016-v615-source-contract-preflight` | V615 source delta and exact task-contract preflight | `results/experiment_7016_v615_source_contract_preflight.json` | None |
| 2 | `exp7017-task-linked-compute-receipts` | Task-linked phase, GPU, and runner receipt contract | `results/experiment_7017_task_linked_compute_receipts.json` | None |
| 3 | `exp7018-v615-sota-ingestion` | V615 recent-source ingestion and architecture map | `results/experiment_7018_v615_sota_ingestion.json` | None |
| 4 | `exp7019-arc-belief-stream-fixture` | Immutable ARC chronological belief-stream fixture | `results/experiment_7019_arc_belief_stream_fixture.json` | None |
| 5 | `exp7020-counterexample-belief-ledger` | Counterexample-updated ARC belief ledger | `results/experiment_7020_counterexample_belief_ledger.json` | Exp7019 `arc_belief_stream_ready_score == 1` |
| 6 | `exp7021-prospective-belief-utility` | Prospective held-future belief utility comparison | `results/experiment_7021_prospective_belief_utility.json` | Exp7020 `belief_ledger_ready_score == 1` |
| 7 | `exp7022-belief-ledger-cold-audit` | Fresh-process belief isolation, retention, and poison audit | `results/experiment_7022_belief_ledger_cold_audit.json` | Exp7020 `belief_ledger_ready_score == 1`; Exp7021 `belief_utility_comparison_complete_score == 1` |
| 8 | `exp7023-belief-query-api` | Bounded belief-query API for the ARC policy | `results/experiment_7023_belief_query_api.json` | Exp7022 `belief_shadow_safe_score == 1` |
| 9 | `exp7024-belief-aware-e3-selector` | Default-off belief-aware E3 selector wiring | `results/experiment_7024_belief_aware_e3_selector.json` | Exp7023 `belief_query_api_ready_score == 1` |
| 10 | `exp7025-belief-shadow-live-trace` | Provenance-complete live belief shadow trace | `results/experiment_7025_belief_shadow_live_trace.json` | Exp7017 `task_compute_receipt_ready_score == 1`; Exp7024 `belief_selector_live_path_ready_score == 1` |
| 11 | `exp7026-held-mechanic-belief-ab` | Held-mechanic live belief and simulation A/B | `results/experiment_7026_held_mechanic_belief_ab.json` | Exp7017 `task_compute_receipt_ready_score == 1`; Exp7025 `belief_shadow_trace_ready_score == 1` |
| 12 | `exp7027-v615-capstone` | V615 independent evidence capstone and V616 handoff | `results/experiment_7027_v615_capstone.json` | None; structurally ungated |

### Phase I: Contract, observability, and source delta (`exp7016`-`exp7018`)

Exp7016 parses this document and the YAML independently. It checks exact
count, order, titles, deliverables, gates, producer field spelling, prior
failures, model rules, artifact fields, and prompt tails. It reports the
contract defect but never edits either source silently.

Exp7017 implements a reusable task-linked receipt. It records setup, model
load, inference, write, and cleanup time. It samples GPU identity,
utilization, memory, and power during the owned interval. It also records
model concurrency and why a sequential or dual-GPU runner was selected.

Exp7018 performs the reserved recent-source ingestion. It records primary and
secondary access results and maps only reproducible changes into Carnot. It
does not make a TSU, Kona, EBT, or ARM-EBM readiness claim from prose.

### Phase II: Prospective belief self-learning (`exp7019`-`exp7022`)

Exp7019 builds an immutable stream from provenance-complete live-agent
attempts. It freezes the observation-time boundary and creates game-blind
mechanic signatures. No held-future field is visible at update time.

Exp7020 implements the continuous self-learning substrate. It maintains
known, possible, contradicted, and uncertain beliefs. It updates after an
observation, groups contradictions, tombstones invalid hypotheses, limits
capacity, and supports exact restart and rollback.

Exp7021 compares frozen, recency-only, and belief-ledger policies on later
events. It reports value on distinct future units. It cannot claim success
from replay fit, write rate, or retention alone.

Exp7022 runs in a fresh restricted process. It audits future leakage,
game/source identity leakage, authority conflicts, supersession, retrieval
collision, poison, capacity, retention, restart, and rollback. It exposes two
separate outputs: safe for shadow use and ready for value promotion.

### Phase III: Live policy integration and generalization (`exp7023`-`exp7026`)

Exp7023 creates a bounded query interface. It returns compact belief evidence
with support, contradiction, uncertainty, age, and provenance. It never
returns a future outcome or a game-specific adapter field.

Exp7024 wires that interface behind a default-off flag in
`make_carnot_agent` and `E3AgentPolicy`. Flag-off behavior must be byte-stable
at the action-decision boundary. The policy records whether belief was
queried, what evidence was available, and whether it changed a ranking.

Exp7025 runs a short live shadow trace with
`unsloth/Qwen3.6-35B-A3B-GGUF`. The query fires in the scored policy, but it
does not change actions. The task proves model, GPU, context, lease, policy,
factory, completion, solve-provenance, and task-time receipts before the A/B.

Exp7026 is the milestone's ARC generalization-floor task. It compares base,
simulation-only, belief-only, and combined live policies under matched action
and LLM budgets. Qwen3.6-35B-A3B is the primary model and
Gemma-4-26B-A4B is a family canary. The roster is held by mechanic rather than
chosen after seeing results. Per-game rows report wins, losses, ties,
no-headroom cells, progress, actions, model calls, latency, and GPU receipts.

### Phase IV: Independent synthesis (`exp7027`)

Exp7027 reads every available V615 artifact directly. It recomputes claims
from rows, records upstream SHA-256 hashes and imported fields, runs
adversarial verification, and reconciles the milestone record. Missing or
gate-blocked inputs produce `verdict_class: blocked`, not `partial`. The
capstone cannot turn a circular, null, blocked, disqualified, or absent result
into a positive claim.

## 8. Dependency Graph

```text
exp7016  contract audit -----------------------------------------+
                                                                  |
exp7018  source ingestion ---------------------------------------+--> exp7027
                                                                  |
exp7019 --> exp7020 --> exp7021 --> exp7022 --> exp7023 --> exp7024
                              |                         |          |
                              +-------------------------+          v
exp7017 ------------------------------------------------------> exp7025
                                                                  |
                                                                  v
                                                              exp7026
                                                                  |
                                                                  v
                                                              exp7027
```

Exp7016 and Exp7018 are advisory and independent. Exp7027 is structurally
ungated so it always records the terminal milestone state. The science chain
has a cold safety gate before policy wiring. The live A/B has a separate
shadow-path gate and a compute-receipt gate.

## 9. Acceptance Gates

The milestone can produce a positive live belief result only if all of the
following hold:

1. Every compared row uses a provenance-complete `E3AgentPolicy` path.
2. The belief updater sees no current or future outcome before its action.
3. Game source, offline BFS, per-game adapters, and hand-built models are
   absent.
4. The cold audit reports zero leakage and zero protected-case regression.
5. The live comparison uses matched action and model-call budgets.
6. Belief-only or combined improves progress per 100 actions over both the
   base and simulation-only controls, with row-derived uncertainty and no
   material safety regression.
7. At least two held mechanic groups contribute non-negative evidence; one
   outlier game cannot carry the result.
8. Every solve claim has `solve_provenance=live_agent_self_discovery` and
   passes the solve-registry precheck.
9. Every numerical headline recomputes from per-game rows.
10. A result that uses an exact outcome at selection time declares
    `verifier_is_oracle=true` and cannot have `verdict_class: positive`.

Failure of a scientific gate is a useful terminal null or disqualification.
It is not a reason to weaken the gate after seeing the rows.

## 10. Hardware and Runtime Requirements

### Available local hardware

- Two RTX 3090 GPUs with about 24 GB VRAM each.
- CUDA-capable `llama.cpp` and the existing lease/port ownership helpers.
- Local cache for the three mandated GGUF families.
- CPU and local storage for deterministic replay, cold audits, and receipts.

### Model plan

Only Exp7025 and Exp7026 require LLM inference.

- Exp7025: `unsloth/Qwen3.6-35B-A3B-GGUF`.
- Exp7026 primary: `unsloth/Qwen3.6-35B-A3B-GGUF`.
- Exp7026 canary: `unsloth/gemma-4-26B-A4B-it-GGUF`.

`unsloth/gemma-4-31B-it-GGUF` remains an available dense flagship but is not
required for the live ARC critical path. Legacy Qwen3.5-0.8B and
Gemma-4-E4B-it may run only as labeled CPU smoke tests. They cannot support a
headline result.

### Resource rules

- Every GPU task checks cached files, CUDA offload, free VRAM, lease
  ownership, port ownership, writable checkpoints, and cleanup before work.
- Each model process runs on one GPU. No single process tensors both GPUs.
- Use the dual-GPU runner only when at least two model processes run at the
  same time. Record the decision.
- Exp7026 checkpoints after every game/seed/arm cell and resumes without
  repeating complete cells.
- The milestone makes no XTR-0, Z1, TSU, Kona, FPGA speed, or board-power
  claim. Z1T is software evidence only.

## 11. Prior-Failure Discipline

The YAML carries explicit `prior_failures` for every materially related null,
blocked, or disqualified scope.

- Exp7016 cites Exp7009. The new attempt checks a 12-task contract that is
  actually present in both sources and adds exact parity tests.
- Exp7020 cites Exp6978. The new learned object is an ARC current-state belief,
  not generated constraint memory.
- Exp7021 and Exp7022 cite Exp6873 and Exp6978 where applicable. The new gate
  is later action utility on a game-blind chronological stream.
- Exp7026 cites Exp7005. The new mechanism queries explicit belief during the
  live policy and compares against simulation-only under matched budgets.

Every entry contains `retire_if_same_verdict: true`. The capstone uses the
standing routine-capstone operator override only to prevent a false scope
match against old capstones. No retired experiment ID is reused and no task
depends on a retired upstream ID.

## 12. Required Verification

Each task follows spec-first and test-first development. Before it reports
done, it runs the focused unit tests, relevant lint and spec-coverage checks,
the applicable steps in `ops/e2e-test-plan.md`, artifact validation,
`adversarial_verify.py`, row-consistency lint for comparative artifacts, and
root-clutter checks.

Before V615 activation, the planner output must pass:

```bash
.venv/bin/python -c "import yaml; yaml.safe_load(open('research-roadmap-next.yaml'))"
.venv/bin/python scripts/exclusion_manifest_lint.py research-roadmap-next.yaml
.venv/bin/python scripts/arc_levelup_guarantee_lint.py research-roadmap-next.yaml
```

A direct parity check must also confirm exactly 12 YAML tasks with the exact
ordered IDs, titles, deliverables, and gates in the task table above.

## 13. Expected Outcomes

### Strong positive

The explicit belief ledger passes the cold audit and belief-only or combined
live policy improves held-mechanic progress or action efficiency over both
base and simulation-only controls under matched compute.

### Useful null

The ledger is safe and queryable but does not improve later decisions. Retire
this explicit belief mechanism. Do not hide the result behind a KAN compressor
or a larger prompt.

### Diagnostic disqualification

The updater leaks future outcomes, uses game identity, cannot preserve
protected beliefs, or the live path cannot prove its model and solve
provenance. Stop before a promotion claim and record the exact failed gate.

### Operational success

The Markdown and YAML task contracts agree exactly, every live cell has
task-linked compute receipts, and the capstone records all terminal outcomes
without a cascade-driven retry.

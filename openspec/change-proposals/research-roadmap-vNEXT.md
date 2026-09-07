# Research Roadmap vNEXT: Milestone 2026.09.623

**Milestone:** `2026.09.623`
**Contract:** exactly 12 tasks, `exp7097` through `exp7108`, in the order
defined below and in `research-roadmap-next.yaml`  
**Theme:** Measure adapter-withheld ARC generalization, test exact feasibility
before action energy, and establish delayed-commit continuous self-learning on
a sealed exact stream.

## What V622 Proved

V622 closed its actual six-task execution contract with a useful null and a
repeat of the planning-contract failure.

- Exp7091 independently found that the Markdown promised 12 tasks while the
  activated YAML held six. Its
  `complete_disqualified_v622_markdown_yaml_contract_mismatch` verdict is valid
  evidence. V623 therefore makes the exact 12-row table below the normative
  task contract and gives no science task a dependency on the advisory check.
- Exp7092 completed the requested execution-time source sweep with a verified
  empty delta. The planner refresh in `research-references.md` now supplies the
  current research boundary for V623.
- Exp7093 survived its first hard wall-clock cap, completed 169 tests on the
  next attempt, and produced a terminal null. The cold audit found
  `entrance_support_audit_ready_score=0` and
  `entrance_selector_headroom_ready_score=0`; 26 headroom units did not repair
  missing family support.
- Exp7094 was correctly gate-blocked because Exp7093's support score was zero.
  Exp7095 and Exp7096 were then pre-emptively skipped. No entrance-energy,
  abstention, or downstream continuous-learning claim was tested.
- The repository's ARC registry still reproduces 183 of 183 known levels
  through game-specific adapters. That is a strong regression baseline, but it
  does not measure transfer to a game whose adapter and trajectory recipe are
  withheld.

V623 accepts the entrance branch's null and does not rerun it. It also does not
claim that registry replay is hidden-game generalization.

## Three Largest Gaps to the PRD Vision

1. **Adapter-withheld ARC generalization is unmeasured.** Seven successive
   milestones missed the required generalization floor while adapter-backed
   registry coverage reached 183 of 183 levels. Carnot needs an honest
   leave-one-game-out measurement in which the live E3 path acts without a
   per-game adapter, game source, registry trajectory, or hand-built solution.
2. **Energy is not causally connected to feasible live action.** Earlier
   relational and entrance-energy branches were null or blocked. Carnot still
   lacks a matched receipt showing that an exact generic feasibility projection
   preserves useful action support, that a fixed analytic energy changes the
   chosen action, and that any world-model forecast is actually consumed.
3. **Continuous self-learning remains underpowered and unverified.** The failed
   BCIT attempt exposed only eight events across five groups, and the earlier
   transactional-memory experiment was null. The PRD requires a sealed stream,
   exact post-decision supervision, bounded procedural memory, prospective
   improvement, protected retention, and transactional rollback.

## Research Findings That Shape V623

- The ACL 2026 world-model study (`arXiv:2601.03905`) separates simulation
  invocation, interpretation, and action integration. V623 records all three;
  a forecast call that cannot be linked to a later action earns no causal
  credit.
- FSNet (NeurIPS 2025) applies a feasibility-seeking step before optimization.
  V623 turns that ordering into a hard control: exact generic legality and
  invariant projection precedes optional action ranking.
- The 2026 continual-memory study (`arXiv:2604.27003`) reports that abstract
  procedures transfer more reliably than raw trajectories and that negative
  transfer concentrates on hard cases. V623 compares procedural memory with
  raw traces, equal-context replay, write-while-deciding, and no memory, then
  reports every group and capacity slice.
- Memoir (`arXiv:2607.20792`) finds an early penalty when an agent writes the
  memory it is currently using. V623 freezes the decision snapshot and commits
  only after exact feedback; a coupled write-while-deciding arm is a negative
  control.
- Distributional EBMs (`arXiv:2605.18871`) motivate a deterministic constraint
  channel next to a learned score and warn about model-identity shortcuts. V623
  uses hard feasibility as authority, fixed analytic energy only as a ranker,
  and matched model-family rows.
- Extropic's 2026 Z1T update fixes a useful host-software interface at degree
  16. V623 checks exact finite distributions and placement overhead on CPU. It
  makes no Z1, FPGA, power, latency, or attached-hardware claim.

The verified citations, code checks, product checks, and claim boundaries are
in `research-references.md` under `V623 planner refresh - 2026-09-07`.

## V623 Architecture

```text
                              V623 contract boundary
                  +---------------------------------------+
                  | Exp7097 contract preflight (advisory) |
                  | Exp7098 current-source ingestion      |
                  +-------------------+-------------------+
                                      |
                              evidence context only

  adapter-withheld ARC branch                         self-learning branch
  ===========================                         ====================

  Qwen3.8-27B live generator +                        exact local fixtures
  Qwen3.6-35B headline control                         SAT / graph / arithmetic /
               |                                      temporal constraint groups
               v                                                  |
  Exp7099 live-path preflight                                      v
  - E3AgentPolicy acts                                Exp7105 sealed 144-event stream
  - no adapter/source/recipe                                      |
  - forecast-to-action receipt                                   v
               |                                      Exp7106 delayed-commit memory A/B
               v                                      - procedural abstraction
  Exp7100 adapter-withheld LOO                        - raw trace
  - at least four games                               - equal context
  - three fixed seeds                                 - write while deciding
  - registry comparison                               - no memory
  - zero is a valid result                                        |
               |                                                  v
               v                                      Exp7107 fresh-process retention,
  Exp7101 cold provenance audit                       poison, crash, and rollback audit
               |
               v
  Exp7102 exact feasibility + fixed analytic action energy
             /   \
            v     v
  Exp7103 live A/B  Exp7104 degree-16 host portability
             \     /
              v   v
        Exp7108 ungated evidence matrix and branch disposition
```

The live ARC generator remains the local E3 route. Its primary generator is
`unsloth/Qwen3.8-27B-GGUF` as required by the live-path policy; every ARC
task that generates actions also runs the mandated headline control
`unsloth/Qwen3.6-35B-A3B-GGUF`. Both use the approved cached llama.cpp path.
Legacy-small models may smoke-test transport only and may not supply a
headline row.

No learned component is an authority. Exact environment transitions determine
ARC outcomes. Exact local solvers determine constraint-stream outcomes. The
action energy is a fixed, game-agnostic ranker over candidates that survive the
generic feasibility projector. Continuous learning changes only bounded
external memory after the event outcome is sealed; model weights remain
frozen.

## Dependency Graph

```text
exp7097   exp7098                    (independent, advisory)

exp7099 --ready==1--> exp7100 --complete==1--> exp7101
                             \                    |
                              +-------------------+
                                      |
                                      v
                                   exp7102
                                  /       \
                                 v         v
                              exp7103    exp7104

exp7105 --stream_ready==1--> exp7106 --comparison_complete==1--> exp7107

exp7108                                (ungated; reads every available artifact)
```

A completion gate means that the upstream experiment ran the declared cells;
it does not require a positive scientific result. Every gate field is a bare
top-level field named verbatim in the producer's required artifact fields.
External incompleteness is `blocked`, not `partial`.

## Exact Task Contract

The following five-column table is normative. `research-roadmap-next.yaml`
must contain these 12 full IDs, titles, deliverables, and gates in this exact
order.

| Order | Full task ID | Title | Deliverable | Structured gate |
|---:|---|---|---|---|
| 1 | `exp7097-v623-contract-preflight` | V623 Markdown and YAML task-contract preflight | `results/experiment_7097_v623_contract_preflight.json` | none |
| 2 | `exp7098-v623-execution-sota-ingestion` | V623 execution-time SOTA ingestion and claim-boundary audit | `results/experiment_7098_v623_sota_ingestion.json` | none |
| 3 | `exp7099-adapter-withheld-live-path-preflight` | Adapter-withheld ARC live-path preflight | `results/experiment_7099_v623_adapter_withheld_preflight.json` | none |
| 4 | `exp7100-adapter-withheld-arc-loo-measurement` | Mandatory adapter-withheld ARC leave-one-game-out measurement | `results/experiment_7100_v623_adapter_withheld_loo.json` | `exp7099-adapter-withheld-live-path-preflight.adapter_withheld_live_path_ready_score == 1` |
| 5 | `exp7101-adapter-withheld-arc-cold-audit` | Independent adapter-withheld ARC provenance and leakage audit | `results/experiment_7101_v623_adapter_withheld_cold_audit.json` | `exp7100-adapter-withheld-arc-loo-measurement.adapter_withheld_loo_complete_score == 1` |
| 6 | `exp7102-feasibility-projected-action-energy` | Exact feasibility projection and analytic ARC action-energy comparison | `results/experiment_7102_v623_feasibility_action_energy.json` | `exp7100-adapter-withheld-arc-loo-measurement.adapter_withheld_loo_complete_score == 1`; `exp7101-adapter-withheld-arc-cold-audit.adapter_withheld_audit_ready_score == 1` |
| 7 | `exp7103-adapter-withheld-energy-live-ab` | Adapter-withheld feasibility-energy live A/B | `results/experiment_7103_v623_adapter_withheld_energy_live_ab.json` | `exp7102-feasibility-projected-action-energy.projected_action_energy_comparison_complete_score == 1` |
| 8 | `exp7104-degree16-action-energy-portability` | Degree-16 action-energy software portability receipt | `results/experiment_7104_v623_degree16_action_energy_portability.json` | `exp7102-feasibility-projected-action-energy.projected_action_energy_comparison_complete_score == 1` |
| 9 | `exp7105-sealed-exact-constraint-stream` | Sealed 144-event exact constraint stream | `results/experiment_7105_v623_exact_constraint_stream.json` | none |
| 10 | `exp7106-delayed-commit-procedural-memory-csl` | Delayed-commit procedural-memory continuous self-learning A/B | `results/experiment_7106_v623_procedural_memory_csl.json` | `exp7105-sealed-exact-constraint-stream.exact_constraint_stream_ready_score == 1` |
| 11 | `exp7107-continual-memory-cold-audit` | Fresh-process continual-memory retention and rollback audit | `results/experiment_7107_v623_continual_memory_cold_audit.json` | `exp7106-delayed-commit-procedural-memory-csl.procedural_memory_comparison_complete_score == 1` |
| 12 | `exp7108-v623-capstone` | V623 independent evidence matrix and branch disposition | `results/experiment_7108_v623_capstone.json` | none |

## Phase 0: Contract and Current Evidence

Phase 0 protects the execution boundary and catches new sources. Both tasks
are advisory. No science task gates on either result.

### Exp7097 - V623 Markdown and YAML task-contract preflight

**Purpose:** Stop the recurring stale-document and truncated-YAML failure
before its result can be mistaken for science.

**Method:** Independently parse the active YAML and this five-column Markdown
table. Compare all 12 IDs, order, titles, deliverables, structured gates,
producer fields, prior-failure blocks, model requirements, routing, prompt
tails, and the ungated capstone.

**Acceptance:** `v623_task_contract_conforms_score=1` only when the two
independent views match exactly. Any task-contract mismatch is a terminal
`disqualified` result.

**Gate:** None.

**Prior failures:** Exp7076, Exp7084, and Exp7091 each found a real task-count
or row mismatch. V623 replaces both planning surfaces with one exact 12-row
contract.

### Exp7098 - V623 execution-time SOTA ingestion and claim-boundary audit

**Purpose:** Detect work published after the planning snapshot without
silently widening the milestone.

**Method:** Re-run the requested arXiv and secondary-source sweep. Verify every
promoted item from a primary paper or official project page. Append only novel,
decision-relevant deltas to `research-references.md` and classify each item as
adopt, control, watch, reject, or duplicate.

**Acceptance:** `v623_sota_ingestion_complete_score=1` means every requested
source class has a dated receipt and each adoption has a bounded Carnot hook
and claim boundary. A verified empty delta is complete.

**Gate:** None.

**Prior failures:** Exp6198 completed a narrower post-marker source audit with
no accepted findings. V623 advances the date boundary, covers the operator's
full requested source matrix, and preserves verified empty-delta completion.
Exp7092 is a successful method reference.

## Phase 1: Mandatory Adapter-Withheld ARC Floor

Phase 1 answers the question that adapter-backed registry coverage cannot:
does the live reusable mechanism do anything on a game whose adapter and known
trajectory recipe are unavailable? No task reads game source or builds a
per-game adapter. The run is a `development_proxy`, not new solve credit.

### Exp7099 - Adapter-withheld ARC live-path preflight

**Purpose:** Prove that the intended E3 route can emit and execute a valid game
action under a real adapter-withheld import boundary.

**Method:** Select at least four public games before outcomes, stratified by
mechanic and registry depth. In isolated processes, block game adapters, game
source, registry trajectories, and per-game recipes. Run one bounded action
cell with Qwen3.8-27B and one matched cell with Qwen3.6-35B. Trace simulation
requests, forecast returns, selected actions, and exact environment outcomes.

**Acceptance:** `adapter_withheld_live_path_ready_score=1` requires actual
valid action emission through `E3AgentPolicy`, an exact transition, clean
forbidden-import receipts, and a complete forecast-to-action ledger for both
models. Advice text, an offline BFS, or an unreachable solver fails readiness.

**Gate:** None.

**Prior failures:** Exp6122 found no supported reusable primitive with direct
held-out causal receipts. Exp5766 found no leave-one-out component gain. This
attempt changes the deliverable to a concrete live-path reachability preflight
using the now-existing E3 route and strict import isolation.

### Exp7100 - Mandatory adapter-withheld ARC leave-one-game-out measurement

**Purpose:** Establish the missing ARC generalization floor without promising
that any held game advances.

**Method:** Freeze the games, model builds, seeds, budgets, and success rules
before execution. Run Qwen3.8-27B and Qwen3.6-35B on at least four games and
three seeds through the preflighted E3 route. Each game is one fold. Any
generic calibration uses only the other games; a frozen preexisting policy
declares an empty calibration set. Count only levels reproduced by the live
agent's own attempts and runtime observations. Compare
`levels_reached_without_adapter` with registry `levels_reproduced`; replay any
candidate advance in a fresh exact environment.

**Acceptance:** `adapter_withheld_loo_complete_score=1` requires every frozen
cell and provenance check. All-zero reached levels are a complete null.
`adapter_withheld_any_level_score=1` is a separate scientific outcome and is
never a completion gate.

**Gate:** `exp7099-adapter-withheld-live-path-preflight.adapter_withheld_live_path_ready_score == 1`

**Prior failures:** Exp4330's shallow-tail sweep made no advance, Exp5766 found
no held-out gain, and Exp6122 found no supported reusable primitive receipt.
V623 uses a live E3 policy, two current local GGUFs, fixed multi-game cells,
and an explicit zero-valid measurement.

### Exp7101 - Independent adapter-withheld ARC provenance and leakage audit

**Purpose:** Determine whether the LOO rows really came from the live route and
whether any credited transition survives fresh reproduction.

**Method:** Reconstruct sampled rows in a fresh process from immutable traces.
Audit import closures, prompts, model IDs, seeds, budgets, attempted actions,
forecast consumption, adapter and source access, registry access, and exact
environment replays. Mutate trace bytes and forbidden-path receipts to prove
the audit fails closed.

**Acceptance:** `adapter_withheld_audit_ready_score=1` requires provenance and
leakage integrity for all required cells. It does not require a reached level.
Any game-level outcome must declare `solve_provenance=development_proxy`,
`offline_reproduced=true`, and zero registry promotion.

**Gate:** `exp7100-adapter-withheld-arc-loo-measurement.adapter_withheld_loo_complete_score == 1`

**Prior failures:** Exp6122 lacked direct causal held-out receipts. V623 audits
the immutable attempt and transition ledger emitted by Exp7100.

## Phase 2: Feasibility, Action Energy, and Sparse Portability

Phase 2 uses the measured adapter-withheld support whether it is positive or
zero. It tests generic action filtering and ranking, not per-game knowledge or
cross-game value transfer.

### Exp7102 - Exact feasibility projection and analytic ARC action-energy comparison

**Purpose:** Test whether hard generic feasibility before ranking preserves
valid support and improves immediate action quality over soft penalties.

**Method:** Replay frozen candidate sets from Exp7100. Compare uniform legal
choice, random legal choice, soft constraint penalty, exact feasibility only,
and exact projection followed by a fixed analytic energy. The projector may use
only generic action-schema and observed-state invariants. The energy is fixed
before held rows and cannot learn model identity, game identity, or registry
outcomes. Post-decision exact counterfactual transitions label each candidate.

**Acceptance:** `projected_action_energy_comparison_complete_score=1` requires
all declared arms and rows. A value score is separate and requires increased
valid-action support plus a positive paired immediate-progress delta against
the strongest non-oracle control without a required-model regression.

**Gates:**

- `exp7100-adapter-withheld-arc-loo-measurement.adapter_withheld_loo_complete_score == 1`
- `exp7101-adapter-withheld-arc-cold-audit.adapter_withheld_audit_ready_score == 1`

**Prior failures:** Exp5712's relational live route was null, and Exp6949's
energy task was gate-blocked. V623 replaces learned cross-game scoring with an
auditable fixed analytic energy after an exact generic projector and uses a
completed, provenance-audited candidate ledger.

### Exp7103 - Adapter-withheld feasibility-energy live A/B

**Purpose:** Check whether the best replay policy changes real live behavior
rather than only scoring counterfactual candidate sets.

**Method:** On the same frozen games, budgets, models, and seeds, compare
baseline E3, exact feasibility only, and feasibility plus fixed energy. Record
every generated candidate, rejection reason, energy term, selected action,
simulation request, returned forecast, consuming action, transition, and level
outcome. Exact environment outcomes remain final authority.

**Acceptance:** `adapter_withheld_energy_live_ab_complete_score=1` requires all
matched cells. `adapter_withheld_energy_live_value_ready_score=1` additionally
requires a positive paired progress delta against feasibility-only and baseline
with no required-model regression. Zero or negative value is a complete null.

**Gate:** `exp7102-feasibility-projected-action-energy.projected_action_energy_comparison_complete_score == 1`

**Prior failures:** Exp5712 produced no live-route promotion, and Exp5766 found
no held-out gain. V623 tests the predeclared generic projector and fixed energy
inside the reachable E3 action loop.

### Exp7104 - Degree-16 action-energy software portability receipt

**Purpose:** Determine whether the exact projected action energy has a
lossless, finite host-software Ising representation compatible with a degree-16
parent graph.

**Method:** Compile small frozen candidate sets to QUBO and Ising form. Verify
affine energy, rank, tie, and decoded-action parity exhaustively. Compare exact
Boltzmann probabilities with seeded CPU Gibbs and parallel tempering at three
or more temperatures. Report routing, auxiliaries, quantization, schedule,
autocorrelation, effective sample size, and excluded device costs.

**Acceptance:** `degree16_action_energy_portability_ready_score=1` requires
exact algebra, finite-distribution agreement within the predeclared tolerance,
and explicit degree-16 placement overhead. The result is host software only.

**Gate:** `exp7102-feasibility-projected-action-energy.projected_action_energy_comparison_complete_score == 1`

**Prior failures:** Exp7083 and Exp7090 were blocked by retired entrance-energy
chains. V623 maps a current fixed analytic action energy and has no unavailable
hardware dependency.

## Phase 3: Continuous Self-Learning and Synthesis

This branch is independent from the failed entrance bank and ARC result. It
tests the PRD's continuous self-learning requirement with exact outcomes and
bounded external memory.

### Exp7105 - Sealed 144-event exact constraint stream

**Purpose:** Supply enough prospective, independently verifiable events to
measure memory learning and negative transfer.

**Method:** Build exactly 144 immutable chronological events across at least 12
groups drawn from existing SAT, graph-coloring, arithmetic, temporal, and other
exact local fixtures. Freeze all candidate decisions, difficulty labels, group
boundaries, hashes, memory capacity schedules, and feedback-release order
before exposing exact outcomes. Include reusable structure, matched decoys,
hard groups, and protected retention probes.

**Acceptance:** `exact_constraint_stream_ready_score=1` requires 144 unique
events, at least 12 groups, no conflicts or future-label leakage, exact witness
replay, and complete sealed hashes. It does not depend on an LLM.

**Gate:** None.

**Prior failures:** Exp7070 was blocked with eight frozen events and five
groups. V623 uses a new independent stream with 144 events and at least 12
groups rather than extending the insufficient BCIT source.

### Exp7106 - Delayed-commit procedural-memory continuous self-learning A/B

**Purpose:** Test prospective improvement from exact verifier-signed external
memory while separating abstraction, context replay, and write timing.

**Method:** Run procedural abstraction with delayed commits, raw-trace memory,
equal-context replay without persistence, procedural memory that writes while
deciding, and no memory. Freeze model weights. Give all arms equal information,
context, capacity, decision budget, and validation rules. Make a decision from
a read-only snapshot, reveal exact signed feedback, then commit between events.
Report early, middle, late, hard-group, decoy, capacity, and retention rows.

**Acceptance:** `procedural_memory_comparison_complete_score=1` requires all
arms, groups, capacity slices, and transaction receipts.
`procedural_memory_value_ready_score=1` additionally requires a predeclared
positive later-event delta over every strong control, no hard-group regression,
bounded memory, and protected retention. A complete null is terminal.

**Gate:** `exp7105-sealed-exact-constraint-stream.exact_constraint_stream_ready_score == 1`

**Prior failures:** Exp6978 found no transactional self-learning value, and
Exp7070 lacked a sufficient frozen stream. V623 changes both the memory form
and the evidence base: delayed abstract procedures are tested against four
matched controls on 144 sealed exact events.

### Exp7107 - Fresh-process continual-memory retention and rollback audit

**Purpose:** Verify that any measured memory effect survives reconstruction and
that unsafe or partial updates fail closed.

**Method:** Rebuild selected states from an empty process. Replay event order,
reorder events, inject poisoned feedback, exceed capacity, simulate stale
parents, truncate writes, crash between prepare and commit, and roll back.
Recompute protected retention and hard-group results from immutable decisions
and exact outcomes. Compare final memory hashes and decisions with Exp7106.

**Acceptance:** `continual_memory_cold_audit_ready_score=1` requires exact
reconstruction, atomic commits, deterministic rollback, capacity enforcement,
poison rejection, and metric parity. It audits integrity whether the learning
result was positive or null.

**Gate:** `exp7106-delayed-commit-procedural-memory-csl.procedural_memory_comparison_complete_score == 1`

**Prior failures:** Exp6979 found no cold self-learning value, and Exp7071 was
gate-blocked. V623 audits a completed, sufficiently powered delayed-commit
comparison with immutable event and transaction logs.

### Exp7108 - V623 independent evidence matrix and branch disposition

**Purpose:** Recompute every V623 claim and decide what is promoted, retained
as null evidence, blocked, disqualified, or retired.

**Method:** Discover every current-milestone task by exact ID, validate hashes
and field principles, rerun headline calculations from per-unit rows, enforce
gate and provenance consistency, compare ARC rows with the registry, separate
completion from value fields, and write branch dispositions. Missing upstream
artifacts produce a terminal blocked matrix entry rather than a retrying
`partial` result.

**Acceptance:** `v623_evidence_matrix_complete_score=1` requires a disposition
for all 12 slots and no row/headline contradiction. Positive science is not
required. The capstone remains useful when a branch is blocked or null.

**Gate:** None. The capstone is intentionally ungated.

**Prior failures:** Exp6952 declared the V608 capstone partial because external
science was incomplete. V623 is ungated, discovers the actual current contract,
and assigns terminal blocked entries to stable external gaps instead of
retrying the capstone as partial.

## Hardware Requirements and Claim Boundary

| Resource | V623 use | Requirement | Claim boundary |
|---|---|---|---|
| Dual RTX 3090, 24 GB each | Local GGUF ARC action generation | Available; use approved cache and one run owner per GPU | Report model, quantization, GPU assignment, peak memory, and wall time. No cloud substitution. |
| Host CPU and RAM | Exact ARC replay, stream generation, QUBO/Ising enumeration, Gibbs/PT, audits | Available | CPU sampling and placement are software receipts, not accelerator performance. |
| NVMe model cache | Pinned Qwen3.8 and Qwen3.6 GGUF artifacts | Required and preflighted before generation | A missing mandated model blocks the task. A legacy-small smoke test cannot replace a headline cell. |
| AMD KV260 | None | Terminal prior receipt | Do not rerun or claim V623 evidence. |
| Microchip PolarFire SoC | None | Terminal prior receipt | Do not rerun or claim V623 evidence. |
| GateMate | None | Physically blocked | Not a milestone dependency. |
| Extropic Z1 or TSU | None | Not attached | Degree-16 results are host-software compatibility only; no speed, power, or device-execution claim. |

## Milestone-Wide Scientific Rules

- All comparative tasks emit recheckable per-game or per-event rows; pooled
  headlines alone are invalid.
- Every artifact declares `field_principles`, the current six-value
  `inference_substrate_class`, `execution_venue`, `verdict_class`, and a
  terminal-prefix `honest_verdict`.
- Every blocked artifact uses the exact field `gate_check_summary` and names
  the failed check, expected value, and observed value.
- ARC outcome rows declare `solve_provenance`. V623 uses
  `development_proxy`, requires `offline_reproduced=true` for counted outcomes,
  and sets `arc_registry_delta=0`. No result is presented as a new solve.
- No ARC task reads hidden game source, runs an outer-loop ground-truth BFS,
  builds a per-game adapter, or credits an action path that the live E3 agent
  cannot reach.
- Every task that generates LLM output executes at least one mandated SOTA
  GGUF headline cell. The ARC generation tasks execute both the required live
  Qwen3.8 model and the mandated Qwen3.6 control.
- Continuous learning updates bounded external memory only after exact
  feedback. No LLM weights are changed.
- Null results are terminal scientific results. `partial` is reserved for a
  task's own recoverable incomplete work, never an unchanged external block.
- V623 does not modify `scripts/research_conductor.py`, does not activate
  `research-roadmap-next.yaml`, and does not push.

## Milestone Success Criteria

V623 succeeds as a research milestone when it produces all possible terminal
evidence under the declared gates, even if both scientific branches are null.
The minimum informative outcome is:

1. an adapter-withheld live-path receipt and multi-game LOO measurement with an
   honest zero or nonzero `levels_reached_without_adapter` result;
2. a provenance-audited comparison of exact feasibility and fixed action
   energy, plus a bounded degree-16 host-software receipt when its completion
   gate passes;
3. a sealed 144-event stream, a five-arm delayed-commit memory comparison, and
   a fresh-process safety audit when their completion gates pass; and
4. an ungated capstone that reconciles the exact 12-task contract without
   inflating blocked, circular, or development-proxy evidence.

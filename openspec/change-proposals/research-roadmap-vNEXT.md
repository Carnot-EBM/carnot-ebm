# Carnot Research Roadmap vNEXT: Exact Fixed Points and Attributed Online Learning

**Created:** 2026-08-30
**Milestone:** 2026.08.592
**Status:** Planned
**Supersedes:** milestone 2026.08.591
**Experiments:** Exp6784-Exp6795 (12 tasks, four phases)
**Research refresh:** `research-references.md`, `V592 Planner Refresh - 2026-08-30`

## What Milestone 2026.08.591 Proved

Milestone completion was terminal, not scientifically positive. The active manifest contained three
tasks, although the prior design document described thirteen. The three terminal outcomes narrow the
next plan.

| Task | Terminal evidence | What it proved | What it did not prove |
|---|---|---|---|
| Exp6781 | No artifact after three dispatch failures: Codex could not resolve `gemini-3.1-pro-preview` | Roadmap `agent_type` and `model` must be validated as one dispatch contract before activation | No source or method result |
| Exp6782 | `complete_blocked_sequential_sota_runtime`; all static preconditions passed, but Qwen lease admission ended at `lease_wait_deadline_expired`; all model readiness fields were false | Static visibility of two RTX 3090 cards and cached GGUF files does not establish an owned runtime window | No model-quality or grammar result |
| Exp6783 | `blocked_gate_check_failed` because `all_mandated_runtime_ready=false` | The structured gate prevented an unowned fallback and preserved the intended evidence standard | No proof-generation A/B result |

Two older positive results remain valid inputs: Exp6768 produced a 126-row targetable exact-invalid
proof panel, and Exp6769 proved a deterministic environment-indexed grammar with SAT and UNSAT
support. They are not repeated here. Exp6751 and Exp6766 provide simulator-only stochastic-program
evidence; their exact-objective circularity remains explicit.

## The Three Biggest PRD Gaps

### Gap 1: Research execution is not fail-closed before dispatch

The roadmap can express an agent/model pair that the selected CLI cannot serve. Long experiments can
also lose every row when checkpoints live inside temporary directories. Both failures occur before a
scientific conclusion and can cascade into missing artifacts. FR12 needs attributable verification
evidence, so dispatch compatibility and durable row ownership are part of the evidence boundary.

### Gap 2: Carnot has exact fixtures but no validated neural constraint-dynamics bridge

The project has symbolic checkers, constraint groups, targetable proof errors, and a runtime grammar.
It does not yet have a bounded, oracle-distinct learned dynamics module that preserves declarative
group topology and beats a parameter-matched non-structural control on held-out constraint graphs.
AS2, SymbolLKG, and verifier-first counterfactual negatives motivate a small CPU fixed-point test.
The exact checker evaluates candidates but never supplies proposal-time features or gradients.

### Gap 3: Continuous self-learning has not earned causal credit

Exp6749 completed with no support-preserving gain and no prospective transaction activity. Exp6762
and the later live variants blocked on GPU ownership. The PRD's FR11 goal requires more than a memory
file that can round-trip. Carnot needs prospective writes, later reads, changed actions, held-future
improvement, retention, poison rejection, and byte-exact rollback. V592 moves the action boundary to
a CPU constraint-routing controller so these causal requirements can be tested without another model
lease.

## Research Leads Promoted into Experiments

| Source | Mechanism used in V592 | Boundary |
|---|---|---|
| AS2, arXiv:2603.18436 | Constraint-group-aware soft fixed-point dynamics | Soft residual is not correctness |
| SymbolLKG, arXiv:2608.26836 | Rules, dependencies, and constraints as first-class graph nodes | Existing exact checker remains release authority |
| Neuro-symbolic PRM, arXiv:2608.26329 | Constraint-preserving, dependency-breaking hard negatives | No PRM training and no learned authority |
| Compositional Online Learning, arXiv:2608.27244 | Separately attributable admission, retrieval, and routing updates | Updates occur only after an event's exact receipt |
| Time-Dimensional Exchange Coupling, arXiv:2608.21753 | Temporal coupling of successive configurations in one Ising network | Exact-enumerable simulation only; no board or TSU claim |

The requested OpenReview, Hugging Face Papers, Semantic Scholar, GitHub, Extropic, and Logical
Intelligence checks found no public checkpoint, exact authority, or device interface that supersedes
Carnot's pinned stack. The current EBT and ARM-EBM citation trails remain useful context but do not
add an executable matching-base system.

## vNEXT Architecture

```text
                         FAIL-CLOSED EVIDENCE BOUNDARY
                ┌────────────────────────────────────────┐
Roadmap YAML ──►│ agent/model dispatch lint              │
Long runners ──►│ parent-owned atomic checkpoint + resume│
                └──────────────────┬─────────────────────┘
                                   │
             ┌─────────────────────┼─────────────────────────┐
             │                     │                         │
             ▼                     ▼                         ▼
  Exact constraint graph    Chronological event stream   Exact Ising fixtures
  + dependency negatives    + hidden future receipts     + target distributions
             │                     │                         │
      ┌──────┴──────┐       ┌──────┴─────────┐          ┌────┴─────┐
      │ soft grouped│       │ factor admission│          │ Gibbs    │
      │ fixed point │       │ memory retrieval│          │ baseline │
      ├─────────────┤       │ route selection │          ├──────────┤
      │ matched flat│       └──────┬─────────┘          │ temporal │
      │ control     │              │ action              │ exchange │
      └──────┬──────┘              ▼                     └────┬─────┘
             │ candidates      Exact post-action receipt       │ samples
             │                     │                            │
             ▼                     ▼                            ▼
       independent exact      between-event commit       exact distribution,
       checker + cold audit   or byte-exact rollback     ESS, and cost audit
             │                     │                            │
             └─────────────────────┴────────────────────────────┘
                                   │
                                   ▼
                         Ungated branch disposition
```

No V592 experiment invokes an LLM. Frozen model-derived rows retain their original model and hash
provenance when used as observations, but V592 makes no new model-generation claim. This is a
deliberate response to the repeated exclusive-runtime verdict, not a small-model substitution. A
future live experiment must use at least one of the mandated Qwen3.6-35B-A3B, Gemma-4-31B, or
Gemma-4-26B-A4B GGUF families after a genuinely changed execution substrate is available.

## Phase 1: Evidence-Boundary Repair and Exact Fixture (Exp6784-Exp6786)

### Exp6784: Fail-closed roadmap agent-model dispatch contract

Extend the existing roadmap gate audit with a maintained agent-family/model-family compatibility
matrix. Reproduce the Exp6781 cross-vendor mismatch as a negative test, accept supported Codex model
variants including `gpt-5.6-sol`, and emit task ID, agent, model, expected family, and reason. Wire the
check into the documented pre-activation audit command without modifying the research conductor.

**Success:** every task is classified; the Exp6781 pair is rejected before dispatch; supported pairs
round-trip; malformed or unknown pairs fail closed.

### Exp6785: Parent-owned durable row checkpoint and resume contract

Build a small reusable utility for atomic row checkpoints outside worker temporary directories. Test
interruption after a deterministic prefix, restart in a new process, duplicate suppression, manifest
hash refusal, and final byte-stable aggregation. Patch only a synthetic probe or a reusable helper;
do not rerun the Exp6753 ARC comparison.

**Success:** an injected interruption preserves the prefix, a fresh process resumes exactly once,
and a changed manifest is refused with an attributable blocked artifact.

### Exp6786: Constraint-dependency graph and hard-negative fixture

Create exact-enumerable constraint graphs from the frozen proof and declarative-group infrastructure.
Each unit records local groups, cross-group dependencies, exact assignments, topology family, and a
counterfactual negative that passes local checks while breaking one dependency. Freeze disjoint
topology splits and a no-answer feature denylist.

**Success:** at least 96 unique rows span all preregistered families and negative classes; exact cold
replay agrees; local-pass/cross-group-fail negatives are nonempty in every split.

## Phase 2: Oracle-Distinct Soft Fixed Points (Exp6787-Exp6789)

### Exp6787: Group-aware soft fixed-point proposer

Implement an AS2-inspired recurrent operator whose state updates attend to declarative constraint
groups and dependencies. Train on the fixture's train split only. Freeze convergence tolerance,
iteration cap, optimizer, seeds, and candidate decoder. The exact solver is excluded from forward
features, loss inputs, stopping, and repair.

**Success:** the implementation produces attributable candidates for every development and held-out
unit, with finite residuals and deterministic replay. This is mechanism readiness, not a quality win.

### Exp6788: Fixed-point proposer versus parameter-matched flat control

Compare the grouped fixed-point proposer with a flat, non-structural recurrent control matched on
trainable parameters, optimization steps, data, seeds, and candidate budget. Report every
unit-seed-arm row. Headline metrics are held-topology exact-valid rate, dependency-violation rate,
Hamming distance to the nearest valid assignment, convergence, and hard-negative discrimination.

**Positive gate:** the paired lower confidence bound for grouped minus flat exact-valid rate is above
zero, with no preregistered support or convergence harm. Completion is recorded even when the delta
is null or negative.

### Exp6789: Cold fixed-point authority and shortcut audit

In a fresh process, recompute parameters, row hashes, aggregates, and exact outcomes. Permute labels,
group IDs, and dependency edges separately. Test whether the claimed gain survives topology holdout
and disappears under the appropriate destructive controls. Verify that no exact answer, oracle
residual, or exact-check feedback entered the proposal path.

**Success:** all planned rows and destructive controls are attributable. A source result can remain
null or disqualified; the audit must not promote it.

## Phase 3: Prospective Component-Level Self-Learning (Exp6790-Exp6792)

### Exp6790: Chronological constraint-routing opportunity stream

Turn the exact fixture into a chronological decision stream. Before each exact receipt is revealed,
the controller must choose a bounded constraint-check route. The stream includes reusable topology,
novel held-future families, drift, hard cases, poison candidates, and non-saturating headroom. Freeze
five order replicates. Labels and future receipts are unavailable at action time.

**Success:** every order has genuine action alternatives, reusable signal, held-future events, poison
rows, and measurable headroom. Frozen-policy performance is neither zero nor saturated.

### Exp6791: Compositional online constraint-routing comparison

Run isolated frozen, online, random-update-placebo, and retrieval-disabled arms. The online arm may
update factor admission, memory retrieval, and route selection only after the current event's exact
receipt. Record writes, later reads, action changes, component attribution, reward, cost, retention,
and rollback state per event.

**Positive gate:** all orders show actual write/read/action influence, and the order-level lower
confidence bound for online minus frozen held-future utility is above zero without hard-case,
retention, or support harm. This is the mandatory continuous self-learning experiment.

### Exp6792: Causal-use, forgetting, poison, restart, and rollback audit

Cold-replay the comparison. Disable each admitted update and each retrieval in turn. Inject poisoned
receipts, restart at transaction boundaries, enforce capacity, and trigger rollback on preregistered
retention or hard-case harm. Recompute all order-level effects from event rows.

**Success:** every positive credit has a changed-action causal witness; poison is never admitted;
restart preserves bytes; failed safety gates restore the prior bytes exactly. Null is an acceptable
scientific result.

## Phase 4: Temporal Ising Simulation and Disposition (Exp6793-Exp6795)

### Exp6793: Matched-update temporal exchange Ising comparison

Implement the arXiv:2608.21753 temporal exchange schedule in the existing CPU sampling interface.
Compare it with ordinary single-site Gibbs under identical spin-update calls, initial-state schedules,
temperatures, seeds, and graph fixtures. Use exact-enumerable target distributions for headline
fidelity; larger stress rows are diagnostic only.

**Positive gate:** the temporal schedule improves preregistered effective-sample or optimum-hitting
metrics with a positive paired lower bound and does not exceed the target-law error margin.

### Exp6794: Independent sampler fidelity and hardware-cost audit

Recompute target laws and aggregates in a fresh process. Audit burn-in, update accounting, temporal
state initialization, coefficient range, sensitivity, and stationarity. Map added state and arithmetic
to the already documented KV260 and GateMate resource envelopes without synthesis or timing claims.

**Success:** the audit reproduces or rejects the source comparison and clearly separates simulator
fidelity, estimated mapping cost, and physical hardware evidence.

### Exp6795: V592 branch disposition and PRD reconciliation

Read every V592 artifact whether positive, null, blocked, disqualified, circular, or partial. Recompute
the three branch summaries from rows and audit receipts. Update the research ledger and operational
documents with explicit follow-up or retirement conditions. Do not pool infrastructure readiness,
soft fixed-point quality, self-learning, and sampling into one success score.

## Dependency Graph and Conductor Order

```text
Exp6784  dispatch contract ───────────────────────────────────────────────┐
Exp6785  durable checkpoint ─────────────────────────────────────────────┤
                                                                         │
Exp6786  exact graph fixture ──► Exp6787 soft fixed point                 │
                                      │                                  │
                                      ▼                                  │
                                  Exp6788 paired comparison               │
                                      │                                  │
                                      ▼                                  │
                                  Exp6789 cold audit ─────────────────────┤
                                                                         │
Exp6786 ──► Exp6790 chronological stream ──► Exp6791 online comparison    │
                                                  │                      │
                                                  ▼                      │
                                              Exp6792 causal audit ───────┤
                                                                         │
Exp6793 temporal Ising comparison ──► Exp6794 sampler audit ──────────────┤
                                                                         ▼
                                                                  Exp6795 capstone
```

Structured gates test *completion/readiness*, not positive scientific outcomes. Exp6788 may run when
Exp6787 produced a complete attributable proposal set. Exp6789 may run when Exp6788 completed all
planned rows. The same rule applies to the self-learning and sampler audits. Exp6795 is ungated so a
blocked branch cannot suppress the milestone disposition.

## Hardware Requirements

| Tasks | Substrate | Expected memory | Expected time | Claim boundary |
|---|---|---:|---:|---|
| Exp6784-Exp6786 | CPU, local files | 2-4 GB RAM | 20-90 min each | Infrastructure and exact-fixture evidence |
| Exp6787-Exp6789 | CPU PyTorch/NumPy; GPU optional but not required | 4-8 GB RAM | 1-3 h each | Soft proposer plus independent exact evaluation |
| Exp6790-Exp6792 | CPU, transactional local storage | 2-4 GB RAM | 1-3 h each | Tier-2 online constraint learning |
| Exp6793-Exp6794 | CPU exact enumeration and sampling | 4-8 GB RAM | 1-4 h each | Simulator fidelity and estimated mapping cost only |
| Exp6795 | CPU, artifact readback | 2 GB RAM | 30-90 min | Receipt-only synthesis |

- **RTX 3090 pair:** not required. The cards may remain occupied by unrelated owned work.
- **Qwen3.6/Gemma GGUF cache:** retained but not invoked in V592.
- **Strix `gfx1150`:** not used as a substitute inference substrate.
- **KV260:** prior terminal bring-up result stands. No duplicate board task.
- **GateMate:** prior terminal bitstream and sampler result stands. No new physical receipt is assumed.
- **PolarFire:** reachable smoke status remains opportunistic; no milestone gate.
- **Extropic Z1/X0:** no authenticated hardware path. Torx or CPU simulation cannot establish device
  speed, power, or availability.

## Milestone-Level Acceptance and Stop Rules

1. All 12 tasks write their declared artifact or a diagnostic blocked artifact with
   `gate_check_summary`.
2. Every comparative task emits per-unit rows and passes row-to-headline consistency checks.
3. Every artifact declares `verdict_class` from the closed enum and a terminal-prefix
   `honest_verdict`.
4. Exact authority is absent from proposer inputs, loss, stopping, memory action selection, and
   sampler acceptance claims except where the preregistered exact algorithm itself defines the
   baseline or target distribution.
5. The self-learning branch records nonzero prospective writes, later reads, and action influence
   before any positive claim.
6. No blocked branch is replaced by a smaller model, CPU LLM row, remote API, reduced unit set, or
   changed endpoint metric.
7. A repeated prior verdict triggers the task's declared retirement signal. The capstone records
   retirement instead of proposing another unchanged rerun.

## Explicit Non-Goals

- No rerun of the Exp6782 exclusive-GPU admission or Exp6783 proof-generation chain.
- No ARC game solve, duplicate live-level claim, outer-loop reverse engineering, or offline solver.
- No generated-text external energy scorer, learned-verifier release authority, or exact-solver
  feature leakage.
- No KAN scaling branch before a causal self-learning signal exists.
- No physical FPGA or TSU performance claim.
- No modification to `scripts/research_conductor.py` and no publication or push action.

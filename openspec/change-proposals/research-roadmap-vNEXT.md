# Research Roadmap V563: Certified Solver Guidance and Exact Conflict Memory

**Milestone:** 2026.08.563  
**Date:** 2026-08-22  
**Status:** Planned  
**Experiments:** Exp6510-Exp6522  
**Execution file:** `research-roadmap-next.yaml`

## Purpose

V563 tests one narrow hypothesis:

> A fresh exact evidence root can support causal branch-ordering tests, and a
> sound conflict memory can learn across refined constraint queries without
> changing exact answers or reducing held-future support.

This milestone repairs the V562 execution chain before it expands the method.
It does not depend on a retired experiment ID. It does not reopen answer-level
energy scoring, natural-language constraint extraction, external-text
verification, hidden-state scoring, factor spawning, or a hardware speed claim.

## What V562 proved

V562 reached a terminal state for Exp6506-Exp6509. It did not complete its
planned scientific chain.

1. **The Exp6506 content is usable historical evidence.** Its artifact contains
   a completed independent replay, a corrected `partial` class, immutable input
   hashes, and `v562_exact_branch_ready_score=1.0`.
2. **The Exp6506 conductor contract failed.** The conductor returned
   `artifact_not_updated_past_bootstrap` three times and retired the task even
   though the final artifact content exists. V563 must not require that task ID.
3. **The branch dataset never ran.** Exp6507 was skipped because its upstream
   was retired. It produced no scientific artifact.
4. **The method comparisons never ran.** Exp6508 contains only a gate-block
   artifact because Exp6507 was missing. Exp6509 was skipped because Exp6507
   was retired.
5. **The cascade was diagnostic.** It showed that a correct artifact is not a
   valid dependency when the conductor retires its task. A new root must have a
   new ID, a new small deliverable, an atomic terminal write, and no structured
   dependency on Exp6506 or Exp6507.

V562 therefore proved an execution-contract failure and preserved a usable
historical evidence file. It did not prove structural headroom, learned branch
value, exact-repair value, or continuous self-learning.

## Three largest gaps to the PRD vision

| Gap | Current evidence | V563 response |
|---|---|---|
| The research chain has no live eligible root | V562 retired its first task and cascade-blocked all science | Create a new, small independent qualifier from immutable file content; never require a retired task ID |
| Learned guidance has no causal held benefit inside an exact solver | Exact benchmark rows exist, but branch counterfactuals and controls did not run | Build audited branch rows, compare analytical, consistency-aware, refocus, and enumeration controls, then gate compact rankers |
| FR-11 has no sound continuous memory for exact constraint work | Prior factor and weight learners lacked held-future benefit or used retired scopes | Reuse exact conflicts only under proved refinement, then test chronological retention, invalid-reuse veto, rollback, restart, and bounded capacity |

The ARC generalization gap remains a standing project requirement. V563 gives
it one bounded supervisor-refinement slot based on the live redirect ledger.
That task cannot claim a game or level solve.

## Research findings used by V563

The full source refresh is in `research-references.md`.

- **DiBS, arXiv:2606.06518:** learned global structure can order candidate
  values while a complete symbolic solver keeps every candidate. Exp6513 adds
  a consistency-aware control. Exp6515 adds the same feature family to compact
  rankers. No learned advice can prune a candidate.
- **Incremental Neural Network Verification via Learned Conflicts,
  arXiv:2603.12232:** conflict reuse is sound only under a proved query
  refinement. Exp6518-Exp6520 use that relation as the continuous-learning
  write gate.
- **Composing Flow-Matching Energies with Known Physics, arXiv:2608.18004:** a
  learned energy and an independent known residual can be complementary.
  Exp6515 must report analytical-only, learned-only, and additive arms. Both
  terms need independent marginal value before the composed arm can headline.
- **HalluTracer, arXiv:2608.16353:** depth aggregation is promising, but the
  qualified GGUF backend does not expose hidden states. V563 records this as a
  watch item and does not reopen the retired hidden-state lane.
- **Judge, Retrieve, or Abstain, arXiv:2608.17994:** held calibration and
  abstention can control accepted risk. V563 applies the abstention rule to
  learned routing only. Exact solving remains release authority.
- **OpenReview l7GZ3vswuD:** one-shot critical-variable enumeration is included
  as a local structural control, not imported as a result.

Current Extropic, Kona, KAN, Ising, FPGA, Hugging Face, OpenReview, GitHub, and
Semantic Scholar checks do not change the authority boundary or hardware
access state.

## Scientific invariants

1. The installed exact SAT or CSP solver owns labels, accepted solutions, and
   release decisions.
2. Advice can order variables, order values, request bounded refocus, select a
   bounded neighborhood, or abstain. It cannot prune a candidate or accept a
   result.
3. Every data split stays sealed by base-instance lineage. Held rows cannot be
   changed after any held result is read.
4. Every comparative claim emits one row for every unit, arm, seed, shift,
   budget, and terminal disposition.
5. Feature, model, solver, enumeration, repair, and fallback costs are charged.
6. Every arm must return the same exact answer on each admitted unit.
7. Exact solver self-checks are `circular_positive` at most. They cannot create
   an oracle-distinct method claim.
8. Learned advice keeps all candidates and falls back to native exact search.
9. Continuous learning may persist only conflicts with a refinement witness
   and exact replay receipt. It cannot change the solver, feature schema,
   validation split, or release rule.
10. A blocked artifact names the failed field, expected value, observed value,
    and source path in `gate_check_summary`.
11. No structured gate or `requires` chain names Exp6506, Exp6507, Exp6508, or
    Exp6509.
12. No ARC task claims a public game or level solve. Exp6521 reports supervisor
    selection evidence only.

## Local model policy

The planned scientific path does not need an LLM. The exact benchmark,
counterfactuals, controls, rankers, conflict memory, audits, and ARC ledger
reduction are local non-LLM work.

If any implementation adds an LLM arm, its `MODEL_SPECS` must include at least
one of:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

The arm must use `cached_sota_pair()`, `llama_cpp`, and the GGUF embedded
tokenizer. Qwen3.5-0.8B and gemma-4-E4B-it are CPU smoke tests only. They cannot
support a headline. Exp6521 does not invoke the ARC generator; it reads already
recorded live-path supervisor receipts.

## Architecture

```text
Immutable historical files (not structured dependencies)
  Exp6504 exact benchmark rows + Exp6506 completed content
                         |
                         v
           Exp6510 fresh independent V563 root
                         |
                         v
           Exp6511 branch-counterfactual dataset v2
                         |
                         v
           Exp6512 independent dataset audit
                  /                         \
                 v                           v
   Exp6513 structural controls      Exp6518 exact-conflict controller
                 |                           |
                 v                           v
   Exp6514 control certificate      Exp6519 chronological conflict CSL
                 |                           |
                 v                           v
   Exp6515 compact ranker A/B        Exp6520 independent CSL audit
                 |
                 v
   Exp6516 independent ranker audit
                 |
                 v
   Exp6517 learned destroy + exact-repair LNS

   Exp6521 ARC redirect-ledger supervisor refinement (independent)

   Exp6510-Exp6521 terminal records ---> Exp6522 V563 capstone
```

The exact solver remains in the acceptance path in every constraint task.
Advice is a sidecar. Conflict memory is a cache of proved reusable facts, not a
new solver.

## Phase A: Recover the root and prove structural headroom

### Exp6510 - Fresh independent V563 evidence root

Read the immutable Exp6504 and Exp6506 files directly. Recompute their hashes,
row counts, exact-label receipts, verdict contract, and allowed lineage. Write
a small new terminal artifact atomically. Do not rerun the large corrigendum and
do not require the retired Exp6506 task.

**Deliverable:**
`results/experiment_6510_v563_independent_exact_root.json`

**Gate:** `v563_independent_root_ready_score == 1.0`.

### Exp6511 - Exact branch-counterfactual dataset v2

Use the fresh root. Precommit solver checkpoints and candidate rules. Replay
both polarities or values under equal exact budgets. Write bounded shards and a
final manifest so a long computation cannot end as a bootstrap-only artifact.

**Deliverable:**
`results/experiment_6511_exact_branch_counterfactual_dataset_v2.json`

### Exp6512 - Independent branch-dataset audit

Replay dataset rows, split commitments, exact receipts, feature timing, and
leakage attacks. This task always emits a closed readiness score, including
when Exp6511 is missing or blocked.

**Deliverable:**
`results/experiment_6512_branch_dataset_independent_audit.json`

**Gate:** `branch_dataset_audited_ready_score == 1.0`.

### Exp6513 - Combined structural-control A/B

Compare native dynamic branching, shuffled order, analytical order,
consistency-aware order, periodic bounded refocus, random critical-variable
enumeration, and analytical critical-variable enumeration. Charge all costs and
record live influence.

**Deliverable:**
`results/experiment_6513_structural_controls_ab.json`

### Exp6514 - Independent structural-headroom certificate

Recompute Exp6513 from rows. Reject correctness drift, inert advice, identity
shortcuts, uncharged work, one-cell wins, and unsupported transfer from DiBS or
the OpenReview enumeration study. Emit a closed learned-phase gate.

**Deliverable:**
`results/experiment_6514_structural_headroom_certificate.json`

**Gate:** `certified_structural_headroom_score == 1.0`.

## Phase B: Learn advice and test exact repair

### Exp6515 - Compact branch-ranker and energy-composition A/B

Train matched linear, MLP, compact KAN, and small graph rankers. Compare native,
analytical-only, learned-only, and additive analytical-plus-learned scores. Add
the DiBS-style partial-assignment consistency feature as an ablation. Keep all
candidates and exact fallback.

**Deliverable:**
`results/experiment_6515_compact_branch_ranker_ab.json`

### Exp6516 - Independent learned-ranker audit

Replay held rows and charged cost. Test identity, order, family, length,
solver-effort, and label-balance shortcuts. Verify candidate preservation,
abstention, and exact answer equality. Always emit a closed score.

**Deliverable:**
`results/experiment_6516_learned_branch_ranker_audit.json`

**Gate:** `learned_branch_signal_ready_score == 1.0`.

### Exp6517 - Learned destroy with exact-repair LNS

Compare native exact search, random destroy, analytical destroy, and learned
destroy. Use bounded neighborhoods and exact repair. Separate destroy,
repair, fallback, and validity receipts.

**Deliverable:**
`results/experiment_6517_exact_repair_lns_ab.json`

## Phase C: Continuous self-learning through sound conflict memory

### Exp6518 - Transactional exact-conflict controller

Implement a versioned memory for exact conflicts. Admit a conflict only when a
formal refinement witness and replay receipt show it remains valid. Include
invalid-reuse veto, bounded capacity, deterministic eviction, transactional
commit, rollback, restart, corruption quarantine, and native fallback. Define a
fixed-width record and CPU reference mapping as the hardware path.

**Deliverable:**
`results/experiment_6518_transactional_exact_conflict_controller.json`

**Gate:** `exact_conflict_controller_ready_score == 1.0`.

### Exp6519 - Chronological exact-conflict self-learning

Run a sealed chronological stream of related and unrelated queries. Compare
scratch solving, frozen memory, valid reuse, bounded reuse with eviction, and an
invalid-reuse attack arm that must be vetoed. Measure current cost, retention,
future support, interference, restart, and rollback.

**Deliverable:**
`results/experiment_6519_chronological_exact_conflict_csl.json`

### Exp6520 - Independent continuous-learning audit

Replay chronological rows and every memory transaction. Recompute refinement
witness validity, matched exposure, current gain, old-query retention,
held-future support, invalid-reuse veto, restart, rollback, and bounded-capacity
effects. Emit the final CSL claim gate.

**Deliverable:**
`results/experiment_6520_exact_conflict_csl_independent_audit.json`

**Gate:** `csl_claim_eligible_score == 1.0`.

## Phase D: ARC generalization floor and capstone

### Exp6521 - ARC redirect-ledger supervisor refinement

Read live-path `trajectory_supervisor` receipts that carry REQ-ARC-WMTE-6640
outcomes. Retire never-helped arms only after the preregistered firing floor,
raise priority only with repeated progress, and specify a new curated arm only
when all existing arms were exhausted. If the ledger has no valid firings,
report `no_firings_nothing_to_refine`. Do not invoke an LLM and do not solve a
game.

**Deliverable:**
`results/experiment_6521_arc_redirect_ledger_supervisor_refinement.json`

### Exp6522 - V563 adversarial capstone

Recompute every gate from per-unit rows. Separate operational readiness,
circular exact checks, eligible science, nulls, blocked tasks, disqualified
claims, and deferred work. Publish a next-step license and a retirement ledger.

**Deliverable:** `results/experiment_6522_v563_capstone.json`

## Dependency graph

| Experiment | Direct evidence dependency | Structured execution gate |
|---|---|---|
| Exp6510 | Immutable Exp6504 and Exp6506 files | None; historical task IDs are not dependencies |
| Exp6511 | Exp6510 | Exp6510 root score equals 1.0 |
| Exp6512 | Exp6511 terminal file or missing-file receipt | None; always closes the dataset gate |
| Exp6513 | Exp6512 | Exp6512 audit score equals 1.0 |
| Exp6514 | Exp6513 terminal file or gate block | None; always closes the structural gate |
| Exp6515 | Exp6511 and Exp6514 | Exp6514 certificate equals 1.0 |
| Exp6516 | Exp6515 terminal file or gate block | None; always closes the learned gate |
| Exp6517 | Exp6516 | Exp6516 learned score equals 1.0 |
| Exp6518 | Exp6511 and Exp6512 | Exp6512 audit score equals 1.0 |
| Exp6519 | Exp6518 | Exp6518 controller score equals 1.0 |
| Exp6520 | Exp6519 terminal file or gate block | None; always closes the CSL gate |
| Exp6521 | Live supervisor receipts | None |
| Exp6522 | All V563 terminal records | None |

No structured gate or `requires` field refers to Exp6506-Exp6509. Historical
failures appear only in `prior_failures` blocks and direct immutable file
receipts.

## Prior-failure discipline

| New task | Prior terminal task | Prior result | Material change |
|---|---|---|---|
| Exp6510 | Exp6506 | `artifact_not_updated_past_bootstrap` | New ID and small terminal file; consume finished content by hash; do not rerun the corrigendum or require its task |
| Exp6511 | Exp6507 | `blocked_preemptive_skip_upstream_retired` | Fresh eligible upstream; bounded row shards and final manifest |
| Exp6513 | Exp6508 | `blocked_gate_check_failed` | Fresh audited dataset and one combined control harness |
| Exp6513 | Exp6509 | `blocked_preemptive_skip_upstream_retired` | Same fresh dataset; analytical and random enumeration are controls in the combined harness |

Each matching YAML task carries all required `prior_failures` fields and
`retire_if_same_verdict: true`.

## Acceptance logic

### Structural advice claim

A structural result can be positive only when:

- every arm returns the same exact answer;
- at least one preregistered held cell improves exact search work;
- charged wall time is non-inferior to native search;
- the advice changes live branch decisions;
- per-unit replay and shortcut attacks pass; and
- the result is not an exact solver self-consistency claim.

### Learned branch claim

A learned result also requires:

- Exp6514 and Exp6516 scores equal 1.0;
- benefit on at least two held shift axes;
- a compact learned arm beats the analytical control;
- both analytical and learned terms have marginal value before an additive
  energy can headline;
- all candidates remain reachable; and
- abstention returns control to native exact search.

### Continuous self-learning claim

A continuous-learning result also requires:

- Exp6518 readiness and Exp6520 eligibility equal 1.0;
- every persisted conflict has a valid refinement witness and exact replay;
- invalid reuse is vetoed before it reaches the solver;
- the persistent arm beats scratch and frozen-memory controls under matched
  exposure;
- old-query retention and held-future support stay within preregistered bounds;
- capacity, eviction, restart, corruption, and rollback tests pass; and
- the exact solver and immutable evaluation boundary do not change.

Failure of any condition yields `null`, `blocked`, `disqualified`, or `partial`.
It cannot yield `positive`.

## Hardware requirements

| Resource | Use in V563 | Boundary |
|---|---|---|
| CPU and system RAM | Exact solving, counterfactual replay, compact rankers, conflict memory, audits, and ARC ledger reduction | Required |
| Dual RTX 3090 | Optional acceleration for compact KAN or graph-ranker training | No claim may depend on unavailable multi-GPU behavior |
| Local GGUF cache | Not required by the planned path | Any added LLM arm must include a mandated SOTA GGUF and use its embedded tokenizer |
| Fixed-width conflict record | CPU reference in Exp6518; future FPGA pattern-match path | Interface and bit-exact replay only; no board claim |
| KV260 | No board run | Recorded terminal state remains unchanged |
| PolarFire SoC | No board run | Recorded terminal workload state remains unchanged |
| GateMate | No board run | JTAG and physical access remain changed-state blocked |
| XDNA/NPU | No run | Unsupported runtime boundary remains unchanged |
| Extropic TSU/Z1 | No run | No authenticated device or API route; 2027 early access is watch-only |

The continuous-learning hardware answer is explicit: memory updates and lookup
run on CPU now; the frozen fixed-width record can map to an FPGA pattern-match
engine later. V563 does not infer a speedup from that mapping.

## Retired and deferred work

V563 must not:

- reactivate Exp6506, Exp6507, Exp6508, or Exp6509 as a dependency;
- rerun Exp6505 free-form formal mutation generation;
- use DiBS checkpoints that the public repository does not provide;
- prune solver candidates from learned advice;
- treat a composed energy as the exact validity checker;
- reopen answer-level, external-text, or hidden-state verifier lineages;
- let continuous learning add features, change validation data, or change the
  exact solver;
- generate ARC supervisor arms with an LLM;
- claim an ARC game or level solve;
- run an unchanged FPGA, NPU, or thermodynamic hardware probe; or
- publish a hardware latency, power, energy, or acceleration claim.

## Planned outputs

- 13 experiment artifacts in `results/`.
- New REQ-* and SCENARIO anchors before implementation code.
- Focused tests for every new module.
- Per-unit rows for every comparison.
- Row-consistency, exclusion-manifest, and adversarial-verification receipts.
- Applicable exact-solver E2E receipts from `ops/e2e-test-plan.md`.
- Reconciled capability specs, `_bmad/traceability.md`, `ops/status.md`, and
  `ops/changelog.md` during milestone execution.

## Terminal decision

V563 succeeds as a research milestone when it reaches an honest decision:

- promote audited structural advice, exact-repair LNS, or sound conflict
  memory only when their rows meet every gate; or
- retire a repeated scope after the same verdict and preserve only the exact
  benchmark, sound controller mechanics, and negative evidence that survive
  independent audit.

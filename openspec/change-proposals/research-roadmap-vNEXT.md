# Research Roadmap V562: Certified Branch Advice and Continuous Structural Learning

**Milestone:** 2026.08.562  
**Date:** 2026-08-22  
**Status:** Planned  
**Experiments:** Exp6506-Exp6518  
**Execution file:** `research-roadmap-next.yaml`

## Purpose

V562 tests one narrow hypothesis:

> Exact branch-counterfactual evidence can support useful structural search
> advice, and a fixed safe branch policy can improve online without weakening
> exact correctness or held-future support.

This milestone does not reopen answer-level energy scoring, external-text
verification, natural-language constraint translation, learned trajectory
energy, factor spawning, ARC policy work, or hardware acceleration claims.

## What V561 proved

V561 completed Exp6502-Exp6505.

1. **The old trajectory and factor lineages are closed.** Exp6502 permits one
   fresh branch: a new exact SAT/CSP distribution, solver-native structural
   advice, exact branch counterfactuals, fixed-feature weight updates, and a
   fixed-width mapping.
2. **The method boundary is frozen.** Exp6503 records the source receipts and
   keeps an exact local solver above every learned signal.
3. **A useful procedural benchmark exists.** Exp6504 sealed raw SAT/CSP
   instances, exact labels, replay receipts, lineage-separated splits, held
   cells, strata, leakage attacks, and checksums.
4. **The Exp6504 headline is not eligible yet.** The conductor quarantined the
   artifact because `verifier_is_oracle=true` and `verdict_class=positive` are
   inconsistent. The data can be reused only after an independent corrigendum
   replays the rows and emits a valid evidence class.
5. **Free-form local-SOTA mutation is a measured null.** Exp6505 issued one
   request to each mandated GGUF family and accepted zero mutations. Two rows
   failed parsing and one row produced prohibited output. V562 does not retry
   that mechanism and does not require its empty challenge pool.

V561 therefore proved benchmark construction and lineage control. It did not
prove that structural advice improves exact search, that a learned branch
policy generalizes, or that online policy updates are safe and useful.

## Three largest gaps to the PRD vision

| Gap | Current evidence | V562 response |
|---|---|---|
| Verifiable reasoning lacks a causal learned search signal | Exact outcomes exist, but prior trajectory and factor signals failed or were disqualified | Create exact branch counterfactuals, test analytical and enumerative controls, and certify headroom before learning |
| Learned guidance has no held, charged, shortcut-resistant result | Previous learned signals used retired scopes or had no held causal benefit | Compare linear, MLP, KAN, and GNN rankers on sealed shifts with exact fallback and charged overhead |
| FR-11 continuous self-learning is not eligible | The last chronological factor learner had no held-future benefit | Update only fixed safe branch weights, compare replay/co-observation and recency controls, and audit drift, rollback, and future support |

The hardware gap remains important, but current boards are at recorded terminal
or changed-state-blocked states. V562 can define a CPU reference mapping ABI.
It cannot make a hardware execution or acceleration claim.

## Research findings used by V562

The full refresh is in `research-references.md`.

- arXiv:2608.16003 shows that audit and repair history can shift an LLM
  verifier's threshold. V562 keeps proposal and verification contexts separate
  and uses no LLM as a release checker.
- arXiv:2608.18803 separates replay's co-observation benefit from forgetting
  control. Exp6515 includes current-only, replay/co-observation, and frozen
  arms under matched update dose.
- arXiv:2608.16141 shows that zero representational drift can reduce
  plasticity. Exp6515 records weight drift and new-shift acquisition together.
- arXiv:2608.03874 shows that recent context can match explicit skill state.
  Exp6515 includes an ephemeral recent-window control with no persistent
  learned weights.
- OpenReview l7GZ3vswuD uses one-shot critical-variable prediction followed by
  exact enumeration. Exp6509 tests the solver structure with analytical and
  random critical-variable controls. Exp6511 later supplies learned rankers if
  structural headroom is certified.
- Current Extropic and Kona material changes no local authority boundary.
  Exp6517 is a CPU-only mapping contract. It makes no device claim.

## Scientific invariants

1. The installed exact SAT or CSP backend owns labels, accepted solutions, and
   release decisions.
2. Learned or analytical advice may order variables, choose a bounded refocus,
   select a bounded neighborhood, or abstain. It may not accept a solution.
3. All benchmark splits remain sealed by base-instance lineage. No task may
   repair held data after seeing a result.
4. Every comparative claim emits one row for every instance, arm, seed, shift,
   and terminal disposition.
5. Neural overhead, feature cost, model load, and fallback cost are charged.
6. Correctness is a hard invariant. A search reduction cannot offset a changed
   SAT, UNSAT, feasible, or infeasible answer.
7. Positive performance claims require an independent row replay. Exact solver
   self-consistency alone is `circular_positive`, not `positive`.
8. Continuous learning can update weights over a frozen feature schema only.
   It cannot add factors, create features, or change the exact solver.
9. The SOTA mutation lineage stays retired for this milestone. No task consumes
   Exp6505's `challenge_pool_ready_score=0.0` result.
10. A blocked artifact names the failed check and observed value in
    `gate_check_summary`.

## Local model policy

V562 does not need an LLM for its scientific path. That choice follows the
Exp6505 null and the exact-solver authority boundary. It is not an exemption
for later ad hoc model use.

If an implementation task adds any LLM arm, `MODEL_SPECS` must contain at least
one of these repositories:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

The task must use `llama_cpp` and the GGUF embedded tokenizer. It must not use
`AutoTokenizer` on a GGUF. Qwen3.5-0.8B and gemma-4-E4B-it are smoke-test models
only and cannot support a headline result.

## Architecture

```text
V561 immutable evidence
  Exp6504 benchmark ----> Exp6506 independent corrigendum and V562 lock
  Exp6505 mutation null -X           |
                                     v
                         Exp6507 exact branch-counterfactual dataset
                              /                    \
                             v                      v
                 Exp6508 analytical advice   Exp6509 one-shot enumeration
                              \                    /
                               v                  v
                         Exp6510 independent signal certificate
                              |                    |
                    positive |                    | structural mapping
                              v                    v
                  Exp6511 learned ranker A/B   Exp6517 fixed-width ABI
                              |
                              v
                  Exp6512 independent ranker audit
                       |                     \
              positive|                      \ exact dataset
                       v                       v
                  Exp6513 exact-repair LNS   Exp6514 safe online controller
                                                |
                         learned signal + controller
                                                v
                              Exp6515 chronological continuous learning
                                                |
                                                v
                                   Exp6516 independent CSL audit

              Exp6506-Exp6517 ----> Exp6518 V562 capstone
```

The exact solver stays on the acceptance path in every phase. Advice is a
sidecar that can change search order. Failure or abstention returns control to
the native solver.

## Phase A: Repair evidence and measure structural headroom

### Exp6506 - V561 evidence corrigendum and V562 lineage lock

Recompute the Exp6504 artifact from raw rows. Preserve the immutable benchmark
and emit a separate corrigendum. Replace the invalid positive class with an
eligible non-scientific readiness disposition. Freeze the Exp6505 zero-yield
mutation stream as a terminal null and prohibit downstream dependence on it.

**Deliverable:**
`results/experiment_6506_v561_evidence_corrigendum_v562_lineage_lock.json`

**Gate:** `v562_exact_branch_ready_score == 1.0`.

### Exp6507 - Exact branch-counterfactual dataset

Select branch checkpoints before evaluation. For each eligible variable and
polarity, run bounded exact counterfactual replays. Record conflicts,
propagations, decisions, proof or model validity, censoring, and fallback. Keep
the original split and lineage commitment.

**Deliverable:**
`results/experiment_6507_exact_branch_counterfactual_dataset.json`

**Gate:** `branch_counterfactual_dataset_ready_score == 1.0`.

### Exp6508 - Analytical branch and refocus comparison

Compare native dynamic branching, a static analytical structural order,
periodic bounded refocus, and a shuffled structural control. Record advice
influence duration and charged cost. A positive signal needs fewer exact search
events on held units without a correctness change or a wall-clock regression.

**Deliverable:**
`results/experiment_6508_analytical_branch_refocus_ab.json`

### Exp6509 - One-shot critical-variable enumeration

Test the system design from the OpenReview enumeration paper without importing
its claim. Select a fixed critical-variable subset once with an analytical
score. Compare it with random subsets and native exact search. Enumerate the
chosen prefix and use the exact solver for every remaining branch.

**Deliverable:**
`results/experiment_6509_critical_variable_enumeration_ab.json`

### Exp6510 - Independent structural-signal certificate

Replay Exp6508 and Exp6509 from per-unit rows. Reject correctness drift,
identity shortcuts, outcome-derived features, undercharged overhead, and
single-cell wins. Open the learned phase only if at least one preregistered
non-learned method has held headroom.

**Deliverable:**
`results/experiment_6510_structural_signal_certificate.json`

**Gate:** `certified_structural_signal_ready_score == 1.0`.

## Phase B: Learn branch advice under exact authority

### Exp6511 - Linear, MLP, KAN, and GNN branch-ranker A/B

Train compact rankers on the exact counterfactual labels. Match data, update
dose, feature schema, and tuning access. Compare native, analytical, random,
linear, MLP, KAN, and GNN arms. Evaluate family, scale, surface, and source
shifts. The exact solver accepts every final result.

**Deliverable:**
`results/experiment_6511_learned_branch_ranker_ab.json`

### Exp6512 - Independent learned-ranker audit

Replay ranker rows without using task aggregates. Test identifier, row-order,
serialization-length, family, label-balance, solver-effort, and leakage
controls. Recompute charged cost and exact correctness. Emit a closed score
even when Exp6511 was blocked or null so later gates do not depend on a missing
field.

**Deliverable:**
`results/experiment_6512_learned_branch_ranker_audit.json`

**Gate:** `learned_branch_signal_ready_score == 1.0`.

### Exp6513 - Learned destroy with exact-repair LNS

If the ranker audit passes, compare native exact search, random destroy,
analytical destroy, and learned destroy. Use bounded neighborhoods and exact
repair. Record destroy, repair, fallback, validity, and total charged cost as
separate events.

**Deliverable:**
`results/experiment_6513_exact_repair_lns_ab.json`

## Phase C: Continuous self-learning on a fixed safe policy

### Exp6514 - Transactional fixed-feature branch controller

Build a controller that updates bounded weights over the frozen structural
feature schema. It must provide read-before-write receipts, holdout prediction,
transactional commit, rollback, restart, corruption quarantine, capacity
bounds, and exact-solver fallback. This task proves mechanism only.

**Deliverable:**
`results/experiment_6514_transactional_branch_controller.json`

**Gate:** `continual_controller_ready_score == 1.0`.

### Exp6515 - Chronological replay and co-observation study

Run a sealed chronological stream across family, scale, surface, and source
shifts. Compare frozen advice, current-only updates, replay/co-observation,
strong anchoring, and an ephemeral recent-window fit with no persistent state.
Match admitted update count and optimizer dose. Measure current gain,
retention, drift, plasticity, rollback, and held-future exact-satisfying
support.

**Deliverable:**
`results/experiment_6515_chronological_branch_csl.json`

### Exp6516 - Independent continuous-learning audit

Replay chronological rows and transactional receipts. Recompute the frozen,
recency, replay, anchoring, and current-only controls. Reject dose mismatch,
future access, unsafe writes, support loss, hidden feature creation, and claims
based only on current-task gain.

**Deliverable:**
`results/experiment_6516_branch_csl_independent_audit.json`

**Gate:** `csl_claim_eligible_score == 1.0` for a continuous-learning claim.

## Phase D: Portability contract and capstone

### Exp6517 - Fixed-width structural mapping ABI

Map the frozen structural features and branch scores to a fixed-width integer
ABI. Provide a CPU reference encoder, scorer, checksum, saturation rules, and
bit-exact replay. Do not access a board and do not report latency, power, or
acceleration.

**Deliverable:**
`results/experiment_6517_fixed_width_structural_mapping_abi.json`

### Exp6518 - V562 adversarial capstone

Recompute every gate and claim from Exp6506-Exp6517. Separate operational
readiness, circular exact self-checks, positive science, nulls, blocked tasks,
and deferred work. Publish the exact next-step license and retirement ledger.

**Deliverable:** `results/experiment_6518_v562_capstone.json`

## Dependency graph

| Experiment | Direct evidence dependency | Structured execution gate |
|---|---|---|
| Exp6506 | Exp6502-Exp6505 completed artifacts | None |
| Exp6507 | Exp6504 data through Exp6506 corrigendum | Exp6506 lock equals 1.0 |
| Exp6508 | Exp6507 branch dataset | Exp6507 readiness equals 1.0 |
| Exp6509 | Exp6507 branch dataset | Exp6507 readiness equals 1.0 |
| Exp6510 | Exp6508 and Exp6509 terminal artifacts | None; it closes the OR decision |
| Exp6511 | Exp6507 and Exp6510 | Exp6510 certificate equals 1.0 |
| Exp6512 | Exp6511 terminal artifact or gate block | None; it always emits a closed score |
| Exp6513 | Exp6512 audited learned signal | Exp6512 score equals 1.0 |
| Exp6514 | Exp6507 fixed feature schema | Exp6507 readiness equals 1.0 |
| Exp6515 | Exp6512 and Exp6514 | Both scores equal 1.0 |
| Exp6516 | Exp6515 terminal artifact or gate block | None; it always emits a closed score |
| Exp6517 | Exp6510 structural certificate | Exp6510 certificate equals 1.0 |
| Exp6518 | All V562 terminal artifacts | None |

No task requires a retired experiment. Prior failed artifacts appear only in
`prior_failures` receipts and changed-technique comparisons.

## Acceptance logic

### Structural advice claim

A structural method can be positive only when:

- every accepted answer matches exact authority;
- a held preregistered cell improves exact search work;
- the effect survives per-unit replay and shortcut attacks;
- feature, inference, and fallback overhead are charged;
- the result is not only a solver-oracle self-check.

### Learned branch claim

A learned branch claim also requires:

- Exp6510 and Exp6512 scores equal 1.0;
- benefit on at least two held shift axes;
- no result depends on unit identity, row order, family label, or solver outcome;
- a compact learned model beats the analytical control, not only random order;
- abstention returns to the native exact solver.

### Continuous self-learning claim

A continuous-learning claim also requires:

- Exp6514 mechanism readiness and Exp6516 claim eligibility equal 1.0;
- a persistent state arm beats frozen and ephemeral recency controls;
- replay or co-observation benefit is measured apart from retention;
- held-future support does not decline beyond the preregistered tolerance;
- all writes are transactional, bounded, restart-safe, and reversible;
- drift and new-shift plasticity are reported together.

Failure of any condition produces `null`, `blocked`, `disqualified`, or
`partial`. It cannot produce `positive`.

## Hardware requirements

| Resource | Use in V562 | Boundary |
|---|---|---|
| CPU and system RAM | Exact solving, replay, ranker training, audits, and ABI reference | Required |
| Dual RTX 3090 | Optional for compact GNN or KAN training when CPU cost is excessive | No result may depend on unavailable multi-GPU behavior |
| Local GGUF cache | Not required by the planned science path | Any added LLM arm must use a mandated SOTA repository and embedded tokenizer |
| KV260 | No board run | Terminal hardware state remains unchanged |
| PolarFire SoC | No board run | Smoke boundary remains unchanged |
| GateMate | No board run | Changed-state hardware work remains blocked |
| XDNA/NPU | No run | Unsupported runtime boundary remains unchanged |
| Extropic TSU | No run | No authenticated device or API; watch and ABI only |

Exp6517 is an interface experiment. It is not a hardware experiment and cannot
headline latency, throughput, energy, power, or acceleration.

## Retired and deferred work

V562 must not:

- rerun Exp6505 free-form formal mutation generation;
- use an LLM to emit answers, labels, release decisions, or natural-language
  constraints;
- retrain learned trajectory energy or factor causal-value models;
- spawn or synthesize new factors during continuous learning;
- route ARC actions or claim an ARC level solve;
- run unchanged FPGA, NPU, or thermodynamic hardware probes;
- compare hardware speed through a simulator-only path;
- promote Kona or paper results as local evidence;
- convert a blocked downstream task into a positive capstone claim.

## Planned outputs

- 13 experiment artifacts in `results/`.
- New REQ-* and SCENARIO anchors before each implementation.
- Focused tests for every new module.
- Per-unit rows for every comparison.
- Row-consistency and adversarial-verification receipts.
- Reconciled capability specs, `_bmad/traceability.md`, `ops/status.md`, and
  `ops/changelog.md` during milestone execution.

## Terminal decision

V562 succeeds as a research milestone even if every performance gate stays
closed. The required result is a correct decision:

- promote certified structural advice and safe continuous learning when the
  rows support them;
- otherwise retire the repeated scope and preserve only the exact benchmark,
  controller mechanics, and mapping contract that survive audit.

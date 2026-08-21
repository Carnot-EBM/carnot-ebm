# Research Roadmap V561: Certified Structural Search and Continual Branch Advice

**Milestone:** `2026.08.561`  
**Status:** proposed  
**Predecessor:** `2026.08.560`  
**Tasks:** 13 experiments in four phases  
**Execution manifest:** `research-roadmap-next.yaml`

## Executive decision

V560 closed three branches. Compact learned trajectory energy is retired.
The factor-proposal and decomposed-energy chain has no held causal value.
The current ARC prefix energy does not align with later progress. V561 does
not repair or rename those methods.

V561 opens one new branch. It moves learned energy inside an exact solver. The
new energy ranks clauses, variables, and search neighborhoods. It never accepts
a solution. Exact CDCL, CSP repair, and executable checks remain authority.

The milestone also tests continuous self-learning on a different object. The
learner can update weights over a fixed, preapproved set of structural branch
features. It cannot create factors, edit the verifier, change the held split,
or write new solver rules. Every update has exact cost feedback, rollback, and
future-support checks.

## What V560 proved

V560 completed Exp6488-Exp6501. Its capstone classified all upstream outcomes
and made four claim decisions:

- `trajectory_energy_claim_eligible=false`;
- `continuous_learning_claim_eligible=false`;
- `arc_policy_claim_eligible=false`;
- `hardware_claim_eligible=false`.

The detailed findings are more useful than the four booleans:

1. The exact solver-trajectory commitment was sound, but the compact learned
   heads were disqualified because a shortcut control survived.
2. Local SOTA models produced valid one-shot factor proposals. Exact add/drop
   replay found no positive held causal value.
3. The restarted reuse, spawn, and defer controller worked as a mechanism.
   Chronological learning completed, but held-future benefit stayed closed.
4. Medium bounded capacity preserved support and exact safety under stress.
   This is a mechanism result, not a learning-benefit result.
5. The independent audit replayed the rows and kept the continuous-learning
   claim ineligible.
6. ARC prefix energy had no held alignment with later progress. The policy A/B
   stayed gate-blocked.
7. No authenticated special-hardware result supported an acceleration claim.

The V560 handoff is explicit: retire learned trajectory energy, defer ARC policy
work, and scale only a fresh exact structural signal or a new task distribution.
V561 follows that handoff.

## The three largest gaps to the PRD vision

### Gap 1: Carnot has no causal solver-native energy signal

The PRD calls for learned energy that improves verifiable reasoning. V560 showed
that whole-trajectory and proposed-factor signals are not enough. Carnot needs a
signal tied to exact search actions. V561 uses clause pressure, propagation,
graph structure, branch counterfactuals, and solver influence receipts. It first
tests analytical advice. Learned rankers run only after that structural signal
has held value.

### Gap 2: Carnot lacks a clean shifted benchmark for structural reasoning

Past corpora leaked identity, order, length, model family, or construction
rules. V561 commits a new exact SAT/CSP benchmark before fitting. It separates
formula family, graph scale, surface relabeling, solver hardness, and source.
Procedural rows are sufficient for the benchmark. Local SOTA models may add
challenge mutations, but exact parsing and solving provide all labels.

### Gap 3: Continuous self-learning has not improved held future work

Carnot has lifecycle controls, receipts, and rollback. It does not have a
positive held-future learning result. The V560 factor-pool learner is retired.
V561 tests a narrower learner. It updates only weights over a fixed safe branch
feature set. Exact solver cost is the feedback. The learner must beat frozen and
matched-dose controls after shifts and restarts without losing future support.

ARC policy and physical hardware acceleration remain important. They are not
active V561 gaps because their direct V560 prerequisites are closed. V561 keeps
the hardware ABI visible and keeps ARC unchanged.

## Research findings adopted for V561

The 2026-08-21 source refresh is in `research-references.md`. V561 adopts four
methods and two warnings:

- **Certified Correctness in Neural Constraint Reasoning Requires Symbolic
  Integration** ([arXiv:2608.14569](https://arxiv.org/abs/2608.14569)) argues
  that neural-only constraint systems can violate hard constraints under
  shift. Exp6507-Exp6509 keep neural advice inside an exact solver.
- **Learning to Rank the Initial Branching Order of SAT Solvers**
  ([arXiv:2603.07176](https://arxiv.org/abs/2603.07176)) shows useful static
  advice on random and pseudo-industrial SAT. It also shows that dynamic
  heuristics can erase that advice. Exp6506-Exp6508 record influence duration
  and compare initialization with bounded refocus.
- **Using Clause Predictions for Learning-Augmented Constraint Satisfaction**
  ([OpenReview](https://openreview.net/forum?id=xvcqXxw4Le)) motivates uncertain
  clause-level advice with an exact fallback. Exp6506 creates clause and branch
  counterfactual labels.
- **Large Neighborhood Search meets Iterative Neural Constraint Heuristics**
  ([arXiv:2603.20801](https://arxiv.org/abs/2603.20801)) separates destroy and
  repair. Exp6509 uses stochastic destroy and exact greedy repair only after a
  learned branch signal clears its gate.
- **Solver-Hard Is Not Model-Hard**
  ([arXiv:2607.17047](https://arxiv.org/abs/2607.17047)) warns that solver effort
  is not a proxy for model difficulty. Exp6504-Exp6506 stratify both structure
  and surface form.
- Extropic's current Z1 update defines a sparse 16-neighbor Ising target and a
  future compiler stack. Exp6513 tests only a fixed-width mapping ABI on local
  CPU reference code. It makes no device, power, or speed claim.

## Target architecture

```text
 procedural exact families             local SOTA GGUF models
 SAT | coloring | scheduling            propose edit scripts only
             |                                      |
             +-------------+------------------------+
                           v
              exact parser + SAT/CSP authority
              validity | SAT/UNSAT | proof receipt
                           |
                  sealed shifted benchmark
            family | scale | surface | hardness | source
                           |
                           v
              exact branch counterfactual recorder
              clause advice + variable order labels
                           |
           +---------------+----------------+
           |                                |
           v                                v
 analytical structural energy       learned branch rankers
 native solver control              linear | MLP | KAN | GNN
           |                                |
           +---------------+----------------+
                           v
             exact CDCL / CSP repair remains authority
                           |
              [positive held causal gate only]
                           v
          stochastic-destroy LNS + exact greedy repair
                           |
                           v
        fixed-feature continual branch-policy learner
        matched dose | rollback | restart | future support
                           |
             independent row and safety audit

 structural factor graph --> fixed-width Ising/TSU mapping ABI
                              CPU reference only
```

The architecture has six hard boundaries:

1. SOTA models may mutate a formal instance. They do not answer it.
2. Exact solvers create labels and certify every accepted output.
3. Learned advice changes search order only. It cannot change satisfiability.
4. Dynamic solver overrides are measured. An inert initializer is not a win.
5. Continuous learning updates fixed weights only. It cannot spawn factors.
6. The mapping ABI is a portability check. It is not hardware execution.

## Phase A: boundary and certified shifted substrate

### Exp6502 - V560 retirement ledger and V561 lineage lock

Recompute the V560 capstone from rows. Freeze the learned trajectory-energy,
factor-causal, decomposed-energy, checker-router, factor-pool learner, and ARC
policy branches. Permit only exact solver-native structural advice, a new
certified task distribution, fixed-feature learning, and fixed-width mapping.

**Deliverable:** `results/experiment_6502_v560_retirement_v561_lineage_lock.json`

### Exp6503 - V561 source delta and method preregistration

Recheck the planner sources at execution time. Record stable paper and product
receipts. Convert the selected methods into a frozen comparison contract. This
task adds no runtime dependency and makes no paper claim a local result.

**Deliverable:** `results/experiment_6503_v561_source_delta_method_contract.json`

### Exp6504 - Immutable exact structural benchmark commitment

Create procedural SAT and CSP families with exact labels. Include random and
pseudo-industrial CNF, Tseitin and pigeonhole controls, graph coloring, and a
small scheduling family. Freeze source, family, scale, surface, hardness, and
split strata before any ranker sees the rows. Use at least 30 held units in each
headline comparison cell.

**Deliverable:** `results/experiment_6504_exact_structural_benchmark_commitment.json`

### Exp6505 - Local-SOTA formal challenge mutation stream

Use all three mandated GGUF families, one at a time, to propose one-shot edit
scripts against already formal development instances. An edit script may add,
remove, or relabel bounded clauses, edges, or jobs. It may not emit an answer,
label, natural-language-to-ConstraintIR translation, solver heuristic, or
release decision. Exact parsing and solving accept or reject each mutation.

**Deliverable:** `results/experiment_6505_sota_formal_challenge_mutations.json`

### Exp6506 - Sealed multi-shift benchmark and exact branch labels

Merge the procedural commitment with any valid model mutations. Freeze held
source, family, scale, surface, and hardness shifts. At precommitted solver
checkpoints, run exact branch counterfactuals and record clause, variable,
propagation, conflict, and influence-duration labels. The labels come only from
exact replay.

**Deliverable:** `results/experiment_6506_sealed_structural_branch_labels.json`

## Phase B: exact-solver structural advice

### Exp6507 - Analytical structural advice A/B

Compare native dynamic branching, one-shot static structural order, and bounded
periodic refocus. Use fixed analytical features only. Measure solve parity,
conflicts, propagations, decisions, wall time, and how long advice stays active.
This experiment establishes whether any fresh structural signal exists before
training.

**Deliverable:** `results/experiment_6507_analytical_structural_advice_ab.json`

### Exp6508 - Gated learned branch-ranker comparison

Run only if Exp6507 finds held structural benefit with exact validity parity.
Compare regularized linear, compact MLP, compact KAN, and small GNN rankers at
matched parameter and inference budgets. Use exact counterfactual labels. Test
family, scale, surface, source, identity, order, and serialization shortcuts.

**Deliverable:** `results/experiment_6508_learned_branch_rankers.json`

### Exp6509 - Gated structural LNS destroy and exact repair

Run only if Exp6508 finds a held learned branch signal. Compare random destroy,
analytical destroy, and learned destroy. Use exact greedy repair and native
solver fallback. Report every destroy, repair, fallback, validity, and cost
event. A search-work gain cannot trade away exact validity.

**Deliverable:** `results/experiment_6509_structural_lns_exact_repair.json`

## Phase C: continuous self-learning

### Exp6510 - Fixed-feature continual branch-policy controller

Build the default-off update mechanism for branch-ranker weights. Freeze the
feature set and capacity. Add exact-cost feedback, event-time evidence,
transactional writes, rollback, restart, dose accounting, and corruption
fixtures. This is a mechanism task. It makes no learning-benefit claim.

**Deliverable:** `results/experiment_6510_continual_branch_policy_controller.json`

### Exp6511 - Chronological continuous branch learning

Run only if Exp6508 and Exp6510 clear their gates. Compare frozen, always-update,
fixed-threshold, and anytime-valid guarded updates. Match exposure and accepted
update dose. Evaluate later families, larger scales, surface relabeling,
recurrence, restart, exact validity, negative transfer, and future support.
This is V561's required continuous self-learning experiment.

**Deliverable:** `results/experiment_6511_continuous_branch_learning.json`

### Exp6512 - Independent continual-learning and support audit

Run after Exp6511 completes, including after a valid null. Use an independent
reducer to replay every update, rollback, restart, cost, and future-support row.
Attack held access, dose mismatch, update leakage, seed pooling, capacity drift,
and silent solver override. Keep execution completeness separate from claim
eligibility.

**Deliverable:** `results/experiment_6512_continuous_branch_learning_audit.json`

## Phase D: portability boundary and capstone

### Exp6513 - Fixed-width structural Ising mapping ABI

Map the exact structural feature graph and analytical branch energy into a
bounded fixed-point Ising-style descriptor. Check quantization, degree, spin
budget, coefficient range, CPU energy parity, and round-trip hashes. Use no
board and make no performance claim. The result is a future portability
contract for FPGA, THRML, Thermalizers, or Z1 work.

**Deliverable:** `results/experiment_6513_structural_ising_mapping_abi.json`

### Exp6514 - V561 independent capstone and V562 handoff

Recompute all gates and headlines from rows. Verify retirements, exact authority,
model boundaries, continuous-learning receipts, and hardware-claim limits.
Classify each of the three gaps and produce one explicit V562 branch decision.
This task is ungated so it runs after clean nulls and gate closures.

**Deliverable:** `results/experiment_6514_v561_capstone.json`

## Dependency graph

```text
6502 V560 retirement / V561 lock
 |
 +-- 6503 source delta and method contract
 |      `-- 6504 exact benchmark commitment
 |              `-- 6505 SOTA challenge mutations
 |                      `-- 6506 sealed shifts + exact branch labels
 |                              |
 |                              +-- 6507 analytical advice
 |                              |      `-- [positive] 6508 learned rankers
 |                              |                    `-- [positive] 6509 LNS
 |                              |
 |                              +-- 6510 continual controller
 |                                     `-- [6508 positive] 6511 learning
 |                                                         `-- 6512 audit
 |                              |
 |                              `-- 6513 fixed-width mapping ABI
 |
 `-- 6502..6513 --------------------------------------> 6514 capstone
```

The structured gates in `research-roadmap-next.yaml` are authoritative. Every
gate field appears with the same spelling in the upstream task's required
artifact fields. Exp6514 is deliberately ungated.

## Hardware and model requirements

| Resource | Tasks | Requirement |
|---|---|---|
| Dual RTX 3090, 48 GB aggregate VRAM | Exp6505 | Load one GGUF at a time through `llama_cpp`. Record offload, VRAM, backend, tokenizer, and timing receipts. |
| `unsloth/Qwen3.6-35B-A3B-GGUF` | Exp6505 | Required in `MODEL_SPECS`; flagship MoE challenge generator. |
| `unsloth/gemma-4-31B-it-GGUF` | Exp6505 | Required in `MODEL_SPECS`; flagship dense challenge generator. |
| `unsloth/gemma-4-26B-A4B-it-GGUF` | Exp6505 | Required in `MODEL_SPECS`; middle MoE challenge generator. |
| CPU and installed exact solvers | Exp6504, Exp6506-Exp6514 | Exact labeling, CDCL/CSP search, replay, continual updates, audits, and CPU mapping parity. |
| FPGA boards | none | KV260 is terminal. PolarFire is at a recorded terminal smoke boundary. GateMate's last authorized changed-state detect failed. No unchanged probe. |
| Extropic XTR-0 / Z1 | none | No authenticated local device or API. Vendor material is architecture context only. |
| Kona | none | No public weights or reproducible local runner. |

`llama_cpp` and each GGUF's embedded tokenizer are authoritative for Exp6505.
Do not use `transformers.AutoTokenizer` with a GGUF repository. Legacy
Qwen3.5-0.8B and gemma-4-E4B models may run only as labeled CPU smoke checks.
They cannot support a headline.

## Evidence, safety, and retirement rules

- Every comparison emits one row for every unit, arm, seed, family, scale,
  surface, and source needed to recompute its claim.
- Every task declares `verdict_class` as one of `positive`,
  `circular_positive`, `null`, `blocked`, `disqualified`, or `partial`.
- Every blocked verdict uses `honest_verdict: blocked_*` and an exact
  `gate_check_summary` with the failed check and observed value.
- Every artifact records `inference_substrate`, `verifier_is_oracle`, field
  principles, field provenance, random seed, duration, tests, and checksum.
- Exact solvers are oracles only for the formal instance they execute. Learned
  rankers, model mutations, and mapping descriptors are never oracles.
- Exp6505 carries the failed generated-answer, schema-reprompt, and fixed-policy
  corpus ancestry. It is allowed only because the model emits instance edits,
  never answers or semantic translations.
- Exp6508 carries Exp6490. If shortcut survival repeats, the learned structural
  branch-ranker scope retires.
- Exp6509 carries Exp6493. A repeated gate-blocked verdict retires this LNS
  continuation.
- Exp6511 carries Exp6496 and Exp5895. If held-future benefit stays null, the
  fixed-feature continuous branch-learning scope retires.
- Exp6512 carries the prior independent audit ancestry. It cannot upgrade a
  claim from aggregate fields when its own rows disagree.
- No V561 task depends on a retired upstream experiment ID.
- No task may modify `research-roadmap.yaml` or
  `scripts/research_conductor.py`. No task may push.

## Spec-anchored execution contract

Each task reads `CODEX.md`, `CLAUDE.md`, this roadmap, relevant upstream
artifacts, capability specs, exclusion records, and `ops/e2e-test-plan.md`.
Before implementation, the task adds or confirms REQ-* and SCENARIO anchors.
It writes failing focused tests first.

Before completion, each task runs focused tests, lint, spec coverage, row
consistency checks, adversarial verification, applicable end-to-end checks,
and `git status --short`. Implementation tasks reconcile OpenSpec,
`_bmad/traceability.md`, `ops/status.md`, and `ops/changelog.md`.

## Acceptance criteria

V561 execution is complete when all 13 artifacts have a terminal record and
Exp6514 can replay their rows and gates. Scientific success is separate:

- **Structural signal success:** Exp6507 improves exact search work with solve
  and validity parity on held shifts.
- **Learned advice success:** Exp6508 beats analytical, native, and shuffled
  controls without a shortcut or family failure.
- **LNS success:** Exp6509 reduces exact search work at matched validity and
  charged cost.
- **Continuous-learning success:** Exp6511 improves later held work and support
  without negative transfer; Exp6512 confirms the result independently.
- **Portability success:** Exp6513 achieves exact CPU energy parity within the
  frozen quantization tolerance. This is not a hardware-speed success.
- **Valid negative outcome:** a clean null, disqualification, or closed gate is
  accepted when rows, provenance, and retirement mechanics are complete.

## Explicit non-goals

- repairing compact learned trajectory energy;
- proposing or admitting new learned factors;
- generated answers, finite-ID answer transport, or NL-to-ConstraintIR
  reprompting;
- external-text, hidden-state, or model-identity verifier scoring;
- changing ARC policy or claiming a new ARC solve;
- reading ARC game source, offline ground-truth BFS, or per-game adapters;
- probing unchanged FPGA state or claiming special-hardware speed;
- using Kona or Extropic product claims as local evidence;
- editing the conductor, activating the roadmap, pushing, or deploying.

## Expected V562 decision boundary

Exp6514 must choose one branch:

1. **Structural and continual results are positive:** shadow the fixed-feature
   branch policy on a larger exact stream. Keep exact fallback.
2. **Structural advice is positive but learning is null or unsafe:** freeze the
   best offline ranker. Research update objectives and support preservation.
3. **Analytical advice is positive but learned rankers fail:** ship no learner.
   Keep the transparent structural heuristic as an experimental baseline.
4. **Structural advice is null:** retire this branch-order lineage and move to
   a different exact task distribution or proof-local objective.
5. **Mapping succeeds without a solver benefit:** retain the ABI only. Do not
   schedule hardware work until an algorithmic workload and authenticated
   execution surface both exist.

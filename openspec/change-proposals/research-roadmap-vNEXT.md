# Carnot Research Roadmap vNEXT: Verifier-Bounded Energy and Continuous Learning

**Created:** 2026-08-15  
**Milestone:** 2026.08.555  
**Status:** Planned  
**Supersedes:** milestone 2026.08.554, experiments 6436-6447 as proposed and
experiments 6436-6444 as activated  
**Primary evidence:** terminal artifacts for experiments 6426-6444,
`ops/conductor-log.md`, `ops/exclusion_manifest.yaml`, and the V555 source
refresh in `research-references.md`

The architecture document was last reconciled on 2026-07-03. It is stale under
the repository's 30-day rule. This roadmap uses it for stable component names.
It uses newer terminal artifacts, status records, and change proposals for
current behavior.

## 1. What milestone 2026.08.554 proved

Milestone 2026.08.554 reached a terminal state, but it did not execute its main
science chain. The activated YAML contained 9 tasks. The design document
expected 12. Exp6436 measured this mismatch and set
`v554_queue_ready_score=0.0`. The conductor then blocked or skipped the gated
chain. This is evidence about the workflow, not evidence for causal factors or
continuous self-learning.

| Area | Terminal evidence | What it proves | What it does not prove |
|---|---|---|---|
| Queue integrity | Exp6436 found 9 activated tasks where 12 were expected. | The handoff detected an incomplete queue and failed closed. | It does not validate any V554 science claim. |
| Path receipts | Exp6437 ended `blocked_gate_check_failed`. | The conductor gate preserved the failed queue decision. | No generation-to-verdict receipt implementation ran. |
| Factor influence | Exp6438-Exp6440 did not produce eligible science evidence. Exp6440 is a gate-block artifact. | Missing and blocked evidence remained visible. | No active-versus-ablated causal factor result exists. |
| Continuous self-learning | Exp6441 and Exp6443 have no science artifacts. Exp6442 is gate-blocked. | The conductor did not promote absent upstream evidence. | No prospective or held V554 CSL claim exists. |
| Independent audit | Exp6444 found 11 failed checks and set `csl_audit_ready_score=0.0`. | The independent audit preserved missing artifacts and the prior duration flag. | It cannot validate a CSL effect that never ran. |
| ARC | The proposed sharded Exp6445 never entered the active queue. Exp6434 remains a zero-byte artifact. | The ARC evidence gap is still explicit. | There is no new reachability, generalization, game, or level claim. |

The result changes the milestone design. V555 does not put all work behind one
queue or receipt gate. Infrastructure tasks still report defects. Science tasks
start from independent, known-good inputs and produce their own receipts.

## 2. The three largest gaps to the PRD vision

### Gap 1: Energy and constraints do not yet cause a verified decision gain

Carnot has exact checkers, factor admission, and several narrow positive factor
results. It does not yet have a clean held result where a fixed constraint
representation changes candidate selection and improves the final exact
outcome. A null downstream policy may also hide a useful representation. V555
must separate representation quality from objective quality.

### Gap 2: FR-11 lacks an independently eligible continuous learning result

Carnot has promising memory and factor experiments. The recent prospective
claims remain blocked by missing artifacts, underpowered cells, or duration
flags. The PRD requires a closed loop with immutable validation, rollback, and
bounded forgetting. V555 must update persistent state from exact future
outcomes, survive restart, reject corrupt feedback, and pass an independent row
recomputation.

### Gap 3: ARC live-path generalization lacks a bounded causal measurement

The public ARC registry is complete. More public game solves are not useful.
The open question is whether a generic state representation and a better
objective change live policy reachability on held mechanics. Exp6434 failed
before it wrote evidence. V555 must run bounded, resumable shards and make no
game or level solve claim.

## 3. Research hooks from the V555 source refresh

| Source | Carnot experiment hook | Boundary |
|---|---|---|
| CrEST, arXiv:2608.13179 | Let exact verifier feedback choose update direction. Let model evidence change update magnitude only. | No in-loop LoRA or GRPO. The base GGUF stays frozen. |
| Objective Is the Bottleneck, arXiv:2608.12959 | Freeze the representation and replace only the planning or ranking objective. | A better probe score is not a release claim. Final exact outcomes decide. |
| Sampling Luck Masquerades as Allocation Gain, arXiv:2608.13087 | Tune a verifier allocation policy on development units and test it on a sealed held split. | Charge probe cost. Include a constructed zero-gain control. |
| Policy-as-logic, arXiv:2608.11905 | Use an LLM only to ground facts into fixed typed slots. Run rules in an exact solver. | Do not ask the LLM to invent the rule language or solver. |
| Training Under Challenge, arXiv:2608.12655 | Preserve executable counterexamples and state suite-relative conclusions. | Passing a finite attack suite is not a global proof. |
| MemoPilot, OpenReview ICML 2026 | Assign credit from future task outcomes, not memory prose quality. | Borrow the temporal credit boundary, not its GRPO stack. |

The refresh found no authenticated Extropic hardware route, no new local Kona
runner, and no board state change. V555 schedules no TSU, Kona, NPU, FPGA, KAN,
or Ising hardware experiment.

## 4. V555 architecture

```text
                         exact authority boundary
                                  │
Natural-language policy task      │
          │                       │
          ▼                       │
┌──────────────────────────┐      │
│ Mandated local GGUFs     │      │
│ Qwen3.6 35B-A3B          │      │
│ Gemma 4 31B dense        │      │
│ Gemma 4 26B-A4B          │      │
└────────────┬─────────────┘      │
             │ raw candidate bytes│
             ▼                    │
┌──────────────────────────┐      │
│ Immutable path receipts  │      │
│ model → bytes → parser   │      │
│ → facts → energy → check │      │
└────────────┬─────────────┘      │
             │                    │
             ▼                    │
┌──────────────────────────┐      │
│ Fixed typed fact slots   │      │
│ + fixed rule program     │      │
└────────────┬─────────────┘      │
             │                    │
             ▼                    │
┌──────────────────────────┐      │
│ Constraint energy        │      │
│ objective + budget route │      │
└────────────┬─────────────┘      │
             │ proposal/ranking   │
             ▼                    ▼
┌──────────────────────────┐   ┌──────────────────────┐
│ Selected candidate       │──▶│ Exact local checker  │──▶ release or abstain
└────────────┬─────────────┘   └──────────┬───────────┘
             │                            │ exact outcome
             ▼                            ▼
┌──────────────────────────┐   ┌──────────────────────┐
│ Verifier-bounded online  │◀──│ Immutable event log  │
│ factor-weight update     │   │ + rollback head      │
└──────────────────────────┘   └──────────────────────┘

Independent ARC lane:
live observation traces → frozen state representation → objective A/B
→ legal-action and held-reachability metrics; no solve or registry update
```

The learned or heuristic parts can ground, rank, route, abstain, or update
persistent weights. Only deterministic local checkers can authorize release.

## 5. Phase 0 - Independent evidence contracts and SOTA corpus

### Exp6448 - V554 terminal handoff and V555 queue integrity

Question: Is the V555 queue complete, internally consistent, and executable?

The task freezes the true V554 terminal state. It validates exactly 12 V555
tasks, unique IDs and deliverables, gate producer fields, prompt endings,
mandatory model declarations, prior-failure blocks, and exclusion rules. It is
ungated. No science task depends on its readiness score. This prevents a queue
defect from silently suppressing the full milestone again.

Deliverable:
`results/experiment_6448_v555_terminal_handoff_and_queue_integrity.json`

### Exp6449 - Ungated generation-to-verdict path receipt contract

Question: Can Carnot localize a changed verdict to the exact boundary that
changed the bytes?

This is a changed recovery of gate-blocked Exp6437. It runs directly on
immutable V553 fixtures. It binds raw bytes, parse output, typed facts, energy
input, checker transport, checker output, and final verdict. Identity,
injected-wrapper, and restored-wrapper controls must localize the changed
boundary. The task is infrastructure and remains independent of Exp6448.

Deliverable:
`results/experiment_6449_generation_to_verdict_path_receipt_contract.json`

### Exp6450 - Fresh SOTA fixed-policy candidate corpus

Question: Can all three mandated local GGUFs produce a fresh, replayable
candidate pool with real exact-selection headroom?

The task seals 36 fixed-policy tool-use problems before inference. It creates
development, allocation-held, and selection-held partitions. Every model
produces matched candidate action plans. A fixed simulator labels final
legality and task success. The artifact stores every raw output and path
receipt. It does not claim that one model is better.

Models:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

Deliverable:
`results/experiment_6450_sota_fixed_policy_candidate_corpus.json`

## 6. Phase 1 - Fixed logic, causal objectives, and held allocation

### Exp6451 - Typed fact grounding into fixed policy logic

Question: Does fixed-slot fact grounding plus exact rule execution beat
policy-as-prompt without creating false accepts?

The task uses each mandated GGUF only to fill fixed predicate slots. The rule
program and solver already exist before model output. It compares
policy-as-prompt, policy-as-code, gold-fact upper bound, and policy-as-logic on
the development split. This is not another ConstraintIR generation attempt.

This task is gated only on `sota_corpus_ready_score == 1.0` from Exp6450.

Deliverable:
`results/experiment_6451_typed_fact_grounding_fixed_policy_logic_ab.json`

### Exp6452 - Representation-versus-objective causal A/B

Question: Is a null candidate-selection result caused by the grounded facts or
by the ranking objective?

The task freezes Exp6451 fact rows and candidate bytes. It compares the current
scalar violation objective, a reachability-aware lexicographic energy, an
active-versus-ablated objective, and a shuffled placebo. It changes no model
weights and generates no new candidates. Exact final-state outcomes determine
the result.

This task is gated on `typed_grounding_ready_score == 1.0` from Exp6451.

Deliverable:
`results/experiment_6452_representation_objective_causal_ab.json`

### Exp6453 - Held-out verifier budget allocation

Question: Does a development-frozen allocation policy improve exact yield on a
sealed held split at equal total checker cost?

The task compares uniform allocation, development-frozen adaptive allocation,
an in-sample oracle diagnostic, and a zero-gain shuffled control. It charges
probe work and holds the total exact-checker budget fixed. The allocation-held
partition remains hidden until the policy and analysis are frozen.

This task is gated only on `sota_corpus_ready_score == 1.0` from Exp6450. It is
independent of Exp6451 and Exp6452.

Deliverable:
`results/experiment_6453_held_verifier_budget_allocation_ab.json`

### Exp6454 - Held exact-constraint energy selection

Question: Does the changed energy objective select more exact-valid action
plans than first-candidate, vote, and shuffled-energy controls?

The task uses the untouched selection-held partition. It freezes the Exp6452
objective before unsealing the rows. The constraint energy is a proposal and
ranking signal. The exact simulator remains the only authority. No external
generated-text scorer or LLM judge is used.

This task is gated on `objective_causal_ready_score == 1.0` from Exp6452.

Deliverable:
`results/experiment_6454_held_exact_constraint_energy_selection_ab.json`

## 7. Phase 2 - Verifier-bounded continuous self-learning

### Exp6455 - Prospective verifier-bounded factor-weight learning

Question: Can Carnot improve future exact yield by updating persistent factor
weights online while the base LLM stays frozen?

This is the milestone's mandatory continuous self-learning experiment. It is a
Tier-1 online weight update over an external constraint-energy layer. It seals
a fresh chronological stream and generates matched candidate pools with all
three mandated GGUFs. It compares frozen weights, self-teacher-signed updates,
and verifier-bounded updates. Exact checker feedback chooses the update
direction. Model confidence can only scale its magnitude. Updates at time `t`
may affect only future units.

The task is ungated. It does not depend on the Phase 1 gate chain.

Deliverable:
`results/experiment_6455_prospective_verifier_bounded_factor_weight_csl.json`

### Exp6456 - Corrupt-feedback quarantine and held restart replication

Question: Does the learned weight ledger keep its benefit after binding shift
and process restart, while a corrupt feedback event is quarantined and rolled
back?

The task seals a new held stream. It compares frozen weights, the clean
verifier-bounded learner, and a learner exposed to a transport-corrupted
feedback event. Path receipts must detect the corrupt event. The governed arm
must quarantine it, write a tombstone, roll back, restart a real process, and
recover the last good head. All held model outputs are new.

This task is gated on `verifier_bounded_csl_ready_score == 1.0` from Exp6455.

Deliverable:
`results/experiment_6456_corrupt_feedback_held_restart_csl_replication.json`

### Exp6457 - Independent CSL row and lifecycle audit

Question: Do the prospective and held CSL headlines recompute from immutable
events without importing upstream aggregate functions?

The audit is ungated. It independently recomputes future exact yield, update
chronology, contamination, forgetting, protected retention, quarantine,
rollback, restart recovery, and cost. Missing, blocked, malformed, flagged, or
underpowered evidence stays visible. A repeated null or blocked result retires
this exact audit scope under the declared prior-failure contracts.

Deliverable:
`results/experiment_6457_independent_verifier_bounded_csl_audit.json`

## 8. Phase 3 - ARC generalization and adversarial close

### Exp6458 - Bounded ARC representation-objective generalization audit

Question: Does a collision-certified state suffix plus a reachability-aware
objective improve held live-policy decisions without a game-specific adapter?

This is a changed recovery of zero-byte Exp6434. It uses bounded CPU shards,
atomic checkpoints, resume, and a terminal partial artifact. It compares the
current state key and objective, the new state suffix with the old objective,
the new suffix with the new objective, and a shuffled placebo. Tuning and held
game rosters are disjoint.

The task measures state collisions, legal-action coverage, held next-state
reachability, action cost, and policy influence. It does not claim or attempt a
game or level solve. It may not read game source, use offline ground-truth BFS,
add a per-game adapter, or change `ops/arc_solve_registry.yaml`.

Deliverable:
`results/experiment_6458_arc_representation_objective_generalization_ab.json`

### Exp6459 - V555 adversarial capstone and reconciliation

Question: Which V555 claims remain eligible after independent row
recomputation, current adversarial checks, dependency review, and determination
preservation?

The capstone is ungated. It audits all 12 task slots and preserves separate
determinations for typed grounding, objective causality, held budget
allocation, energy selection, prospective CSL, held CSL safety, internal ARC
generalization, public ARC, and hardware. It also computes joint failure
moments where row data support them. It may not multiply marginal reliability
or average away missing evidence.

Deliverable:
`results/experiment_6459_v555_adversarial_capstone.json`

## 9. Dependency graph

```text
Phase 0
  Exp6448  terminal handoff and queue integrity       (ungated)
  Exp6449  path receipt contract                      (ungated)
  Exp6450  fresh SOTA candidate corpus                (ungated)
       │
       ├──────────────▶ Exp6451 typed grounding
       │                      │
       │                      ▼
       │               Exp6452 objective causal A/B
       │                      │
       │                      ▼
       │               Exp6454 held energy selection
       │
       └──────────────▶ Exp6453 held budget allocation

Phase 2 independent branch
  Exp6455 prospective verifier-bounded CSL             (ungated)
       │
       ▼
  Exp6456 corrupt-feedback held restart

  Exp6455 + Exp6456 ──▶ Exp6457 independent audit      (audit is ungated)

Phase 3
  Exp6458 ARC representation-objective audit           (ungated)

All task artifacts ──▶ Exp6459 capstone                (capstone is ungated)
```

Only four tasks have runtime gates. Every gate names an upstream task in this
roadmap. Every producer prompt declares the exact gate field in its required
artifact fields.

## 10. Failed-experiment rerun discipline

| New task | Prior scope | What changes |
|---|---|---|
| Exp6449 | Exp6437 `blocked_gate_check_failed` | The receipt task is ungated and runs on immutable known-good fixtures. |
| Exp6451 | Exp5923 retired schema-supported ConstraintIR | The model fills fixed predicate slots. It does not generate a schema, solver, or answer channel. |
| Exp6453 | Exp6429 positive but duration-flagged selective verification | The policy is frozen before a sealed held test, probe cost is charged, and a zero-gain control detects allocation bias. |
| Exp6456 | Exp6432 duration-flagged held restart | The task uses a different external weight learner, new bytes, corrupt-feedback quarantine, and real restart receipts. |
| Exp6457 | Exp6433 null audit and Exp6444 blocked audit | The upstream mechanism is verifier-bounded weight learning, and the audit remains ungated. |
| Exp6458 | Exp6434 zero-byte hard failure | The work is split into bounded shards with atomic checkpoints and a terminal partial artifact. |

Every matching YAML task carries all four required `prior_failures` fields.
Exp6448 carries the standing routine-transition operator override.

## 11. Model and measurement policy

Every task that executes an LLM must use `cached_sota_pair()` or the same
cache resolver and declare at least one of the mandated GGUFs. Exp6450,
Exp6451, Exp6455, and Exp6456 use all three. They use the GGUF-embedded
tokenizer. They do not call `AutoTokenizer` on a GGUF. Legacy small models may
run a CPU smoke test only. Their rows cannot support a headline.

Every comparison emits one `per_unit_rows` entry for each unit, model, arm,
condition, and seed that supports the claim. Every aggregate must recompute
from those rows. Every blocked artifact uses `gate_check_summary`. Every
artifact records `inference_substrate`, `random_seed`, `duration_s`, and
`reproducibility_checksum`. Each artifact maps every required field and gate to
its principle in `field_principles` and classifies its provenance.

## 12. Hardware requirements

| Experiments | Hardware | Expected use | Fail-closed boundary |
|---|---|---|---|
| 6448, 6449 | CPU, 8 GB RAM | Schema, hashes, replay, focused tests | No GPU or science claim. |
| 6450, 6451 | Dual RTX 3090, cached GGUFs, 64+ GB host RAM | Fresh corpus and fact-grounding inference | No headline from CPU fallback or a missing model family. |
| 6452-6454 | CPU, 16 GB RAM | Deterministic replay, solver work, bootstrap intervals | Exact-checker budgets and held partitions must stay sealed. |
| 6455, 6456 | Dual RTX 3090, cached GGUFs, 64+ GB host RAM | Fresh chronological and held candidate generation | No replayed development bytes, synthetic timing, or fake restart. |
| 6457-6459 | CPU, 16-32 GB RAM | Independent reduction, ARC shards, adversarial audit | Partial ARC work must still write an atomic terminal artifact. |

The three mandated model files are already cached and Exp6413 authenticated the
CUDA path. Each live task must obtain a fresh task-scoped receipt. No local
board or authenticated TSU state changed after V554. V555 makes no hardware
speedup, power, or availability claim.

## 13. Success and retirement conditions

V555 succeeds only if the terminal capstone can support at least one of these
narrow claims from row-derived held evidence:

1. Fixed typed grounding plus exact policy logic improves exact outcomes with
   no false-accept increase.
2. A changed constraint-energy objective improves held candidate selection at
   fixed candidate bytes and checker authority.
3. A development-frozen verifier allocation improves held exact yield at equal
   charged budget and passes the zero-gain control.
4. Verifier-bounded online factor weights improve fresh future exact yield,
   survive restart, and contain corrupt feedback without safety regression.
5. The generic ARC representation-objective pair improves held live-policy
   reachability without source access, adapters, or solve credit.

Any branch can fail honestly without invalidating independent branches. A same-
verdict rerun activates its `retire_if_same_verdict` rule. The capstone must
leave public ARC and hardware claims blocked unless separate eligible evidence
exists. V555 does not contain such a task.

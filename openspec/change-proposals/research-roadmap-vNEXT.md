# Carnot Research Roadmap vNEXT: Behavior-Aware Verification and Guarded Continual Reasoning

**Created:** 2026-08-28  
**Milestone:** `2026.08.585`  
**Status:** Planned; activates after milestone `2026.08.584` is archived  
**Experiments:** Exp6715-Exp6728  
**Execution manifest:** `research-roadmap-next.yaml`  
**Informed by:** Exp6702-Exp6705 terminal states, the V585 literature refresh,
FR-11, FR-12, and the live ARC generalization floor

## Milestone decision

V585 keeps the exact planning fixture that V584 recovered. It does not rebuild
that fixture. It replaces one monolithic cold audit with two bounded,
behavior-specific audits and a receipt-only merge. This change addresses the
three Exp6703 hard-wall-clock failures without weakening the audit.

The milestone then asks four questions:

1. Can all three required local GGUF families produce one complete, replayable
   plan bank, and can a typed oracle-distinct energy improve held-family plan
   selection?
2. If the energy transfers, does it improve matched-budget prefix search over
   direct choice, fixed best-of-N, and hard-prefix backtracking?
3. Can a guarded external evidence forest improve future exact outcomes while
   the GGUF weights remain frozen and poison, retention, restart, and rollback
   checks remain clean?
4. Can Carnot close the held-family ARC outcome claim and the Torx raw-chain
   schedule claim without a game solve, FPGA result, or TSU claim?

The final synthesis is ungated. It records positive, null, partial, blocked,
disqualified, and missing evidence.

## What V584 proved

V584 completed in the conductor sense. Only one scientific producer finished.
The remaining tasks reached terminal failure, preemption, or gate states.

| V584 evidence | Measured result | V585 consequence |
|---|---|---|
| Exp6702 exact planning fixture | `planning_fixture_ready=true`; the artifact contains four finite-horizon families, sealed exact labels, task-owned checks, and 92 passing focused tests. | Reuse this immutable fixture. Do not regenerate it. |
| Exp6703 cold fixture audit | Three attempts reached the 4,800-second hard wall. No terminal artifact exists. | Split exact replay from leakage, seal, metamorphic, and mutation checks. Give each task a fixed sample and row budget. |
| Exp6704 proposal bank | The conductor preempted it after Exp6703 retired. No proposal-bank artifact exists. | Reattempt only after both bounded audit shards pass and a receipt-only merge owns the gate field. |
| Exp6705 structural energy | It emitted `blocked_gate_check_failed` because the Exp6704 artifact was absent. No energy training or held-family comparison ran. | Reattempt on typed plan features with the same oracle boundary and explicit prior-failure discipline. |
| Active V584 manifest | The active YAML contained four tasks although its design document described thirteen. | Validate exact task-set and order parity before activation. V585 has 14 tasks in both files. |

V584 did not prove a planning-energy gain, a search gain, a continuous-learning
gain, an ARC solve, a Torx schedule gain, or hardware acceleration.

## The three largest gaps to the PRD vision

### Gap 1: FR-12 lacks an end-to-end local-SOTA verification path

Carnot has an exact finite-horizon fixture and sealed labels. It does not have
an authentic three-family proposal bank, a held-family typed energy result, or
a matched-budget prefix-search result. Generated-text and logprob energy
scoring remain retired. V585 uses only typed state, action, resource,
transition, and violation features for learned scoring.

### Gap 2: FR-11 lacks robust prospective self-learning evidence

Recent repair-memory point estimates have not survived every cold audit. The
V583 online-energy branch never reached its producer. V585 tests a distinct
Tier-1 mechanism: a writable external evidence forest around frozen GGUF
models. Exact checker receipts authorize writes between events. The learner
must survive copied poison, relation poison, future-label leakage, retention,
restart, tombstone, and byte-exact rollback attacks.

### Gap 3: live benefit and stochastic portability remain incomplete

The canonical ARC supervisor reaches live actions and now records outcomes, but
Exp6682 remained partial because its broad verification command failed. It made
no benefit or solve claim. Torx matched most factor and state rows in Exp6684,
but one applicable E2E check failed, so the schedule comparison blocked. V585
uses task-owned verification receipts and preserves the known global-suite
state as a diagnostic.

## Research findings incorporated

The dated record appears in `research-references.md` under “V585 planner
refresh.” V585 uses four new findings directly:

- **HarnessLens** (`2608.27311`) selects behavior-relevant verification tasks
  and requires attributable evidence. V585 splits the failed fixture audit into
  independent behavior owners.
- **ABE-Ralph** (`2608.26753`) treats claims, protocols, components, baselines,
  and metrics as executable constraints. V585 audits method substitutions,
  oracle access, shortened budgets, and missing rows as fidelity failures.
- **GraphMemix** (`2608.26983`) selects a query-aware evidence forest under a
  fixed evidence budget. V585 compares it with no-memory and deterministic
  similarity controls.
- **EVOMAL** (`2608.25776`) demonstrates persistent self-poisoning through
  copied retrieved skills. V585 attacks copied experience records, relation
  edges, provenance, deletion, and tombstone behavior.

PoP (`2608.27165`), BayesPO (`2607.16001`), and VFScale remain watch items.
PoP needs a frozen inter-layer extraction contract. BayesPO changes prompts and
is expensive. VFScale uses a diffusion substrate. None is an executable V585
dependency. The Semantic Scholar citation sweep found no reproducible
matching-base EBT or ARM-EBM checkpoint. Extropic still places public Z1 access
in 2027. Kona still has no public local runner.

## Target architecture

```text
                     PRIOR MILESTONE EVIDENCE

       Exp6702 immutable exact planning fixture (ready)
                         |                 |
                         v                 v
          [6715 exact-replay shard] [6716 seal/attack shard]
                         \                 /
                          \               /
                           v             v
                         [6717 audit merge]
                                  |
                                  v
                  [6718 three-family proposal bank]
                         |                 |
                         v                 v
              [6719 typed plan energy] [6722 guarded forest]
                         |                 |
          positive held-family gate       v
                         |       [6723 prospective CSL A/B]
                         v                 |
              [6720 prefix search]         v
                         |       [6724 poison/durability audit]
                         v
              [6721 search fidelity audit]

                    INDEPENDENT LIVE BRANCH

  canonical ARC E3 seam + Exp6681 outcomes -> [6725 held-family replay]
       no source access, no offline BFS, no game or level solve claim

                 INDEPENDENT STOCHASTIC BRANCH

  Exp6683 exact Ising rows + Exp6684 raw rows -> [6726 Torx parity repair]
                                                   |
                                                   v
                                           [6727 raw-chain A/B]

      every terminal artifact and missing row -> [6728 V585 synthesis]
```

The exact planner is an evaluator and post-event update authority. It cannot
select the current plan. The evidence forest can read immutable past receipts.
It cannot change them. The ARC branch uses only the canonical live E3 path. The
sampler branch measures software semantics only.

## Pre-activation contract

V585 has no runtime manifest-parity experiment. Before activation:

- Confirm that this document and `research-roadmap-next.yaml` contain exactly
  Exp6715-Exp6728 in the same order.
- Run schema, prior-failure, exclusion-manifest, gate-cross-reference, ARC
  level-up, and overdue-priority lints.
- Confirm that every structured gate points to a task in this milestone and to
  a field declared by that task with identical spelling.
- Confirm that every comparative task requests per-unit rows.
- Confirm that every prompt protects `research-roadmap.yaml` and
  `scripts/research_conductor.py`.

Activation must fail before agent time if any contract is false.

## Phase 1: behavior-aware planning verification

Phase 1 recovers the blocked V584 planning chain. It has two infrastructure
audit slots, one bounded merge, one model producer, and three conditional
verification experiments.

### Exp6715: bounded independent exact-replay audit

Recompute a frozen stratified subset of Exp6702 with a separately implemented
exhaustive solver. Audit transitions, legal actions, feasibility, optima, ties,
action values, and action gaps. Cap the sample and enumeration work before
execution. Set `exact_replay_audit_passed=true` only from raw rows.

**Deliverable:** `results/experiment_6715_bounded_exact_replay_audit.json`

### Exp6716: bounded seal, leakage, and mutation audit

Audit prompt-answer leakage, family and split isolation, label-access timing,
stale seals, metamorphic invariants, and mutation detection. Do not enumerate
full plan spaces. Set `seal_attack_audit_passed=true` only from raw attack rows.

**Deliverable:** `results/experiment_6716_bounded_seal_attack_audit.json`

### Exp6717: planning audit receipt merge

Read only the two terminal shard artifacts. Recompute their gate fields and
hashes. Do not rerun a solver, scanner, test suite, or mutation campaign. Set
`planning_fixture_audit_passed=true` only if both shard fields are true and the
fixture identity matches Exp6702.

**Deliverable:** `results/experiment_6717_planning_audit_receipt_merge.json`

### Exp6718: three-family frozen planning proposal bank

Run the exact mandated models:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

Make one planned call per model-instance-seed unit. Store raw output before
parsing. Keep timeout, malformed, empty, and partial rows. Reveal exact labels
only after response commit. `proposal_bank_ready` is a completeness field, not
a quality gate.

**Deliverable:** `results/experiment_6718_sota_planning_proposal_bank.json`

### Exp6719: held-family typed structural plan energy

Fit a small pairwise energy on development rows. Use typed features only. Ban
rationales, answer text, model identity, task IDs, proposal order as a learned
feature, and exact future values. Compare proposal order, hard violations,
immediate cost, a fixed linear rule, randomized controls, and the learned
energy under leave-one-family-out evaluation.

Set `energy_generalization_supported=true` only if regret improves over the
strongest nonlearned baseline, the independent-fold interval excludes zero,
and every leakage test passes.

**Deliverable:** `results/experiment_6719_structural_plan_energy.json`

### Exp6720: matched-budget energy-guided prefix search

Run Qwen3.6-35B-A3B and Gemma-4-31B. Compare direct choice, fixed best-of-N,
hard-prefix backtracking, and hard-prefix plus learned energy. Match model,
instance, seed, candidate, token, verifier-call, retry, and stop budgets. Keep
the exact planner sealed until commitment. This task runs only after a positive
held-family energy gate.

**Deliverable:** `results/experiment_6720_energy_guided_prefix_search.json`

### Exp6721: cold search fidelity audit

Recompute every headline from raw transcripts. Audit exact-evaluator timing,
hidden retries, method substitution, token and query drift, stopping drift,
missing rows, model identity, task identity, and independence units.

**Deliverable:** `results/experiment_6721_prefix_search_fidelity_audit.json`

## Phase 2: guarded continuous self-learning

Phase 2 is the required continuous-self-learning branch. It implements Tier 1
from `research-program.md`: external state changes between events while the
GGUF weights remain frozen.

### Exp6722: guarded evidence-forest fixture

Build a typed event store with state, action, constraints, exact outcome,
alternative branch, relation, provenance, version, support, and tombstone
records. Freeze similarity and evidence-forest retrieval policies. Enforce a
shared evidence-token budget. Add copied-poison, relation-poison,
provenance-loss, future-label, duplicate, retention, partial-write, restart,
rollback, deletion, and tombstone-reappearance attacks.

**Deliverable:** `results/experiment_6722_guarded_evidence_forest_fixture.json`

### Exp6723: prospective evidence-forest CSL A/B

Use Qwen3.6-35B-A3B and Gemma-4-26B-A4B. Run identical prospective orders with
no memory, read-only similarity memory, and guarded evidence-forest memory.
Event `t` may read only records committed before `t`. A write occurs only after
the current response commits and the exact checker authorizes it. Report exact
yield, regret, retrieval precision, influence, tokens, forgetting, retention,
poison rejection, and order-level intervals.

**Deliverable:** `results/experiment_6723_prospective_evidence_forest_csl.json`

### Exp6724: cold CSL poison and durability audit

Replay each order from a clean initial store in a fresh process. Recompute all
comparisons. Attack delayed copied poison, relation edges, provenance,
tombstones, future outcomes, cross-arm contamination, restart, and rollback.
No positive CSL claim survives if the order-level interval includes zero or a
safety check fails.

**Deliverable:** `results/experiment_6724_evidence_forest_csl_audit.json`

## Phase 3: live ARC held-family generalization

### Exp6725: task-owned ARC supervisor repair and held-family replay

Repair only the verification scope that left Exp6682 partial. Reuse the
canonical `make_carnot_agent` and E3 policy seam plus Exp6681 post-redirect
outcomes. Run a prospective held-family off/on comparison with exact
post-action outcomes, matched budgets, protected-state receipts, and task-owned
tests. Preserve the repository-wide suite result as a diagnostic.

This experiment makes no game or level solve claim. It may report only policy
influence, transition benefit, safety, and held-family generalization on live
agent-generated rows. Source reads, offline ground-truth BFS, and per-game
adapters are forbidden.

**Deliverable:** `results/experiment_6725_arc_supervisor_held_family_replay.json`

## Phase 4: stochastic portability and synthesis

### Exp6726: Torx factor and state parity repair

Replay Exp6683 exact reference rows and Exp6684 Torx rows. Repair the
task-owned applicable E2E command or semantic mismatch. Compare factor energy,
normalized probability, conditionals, state enumeration, update order,
coefficient width, and dtype. Set `torx_factor_parity_ready=true` only if every
applicable row and task-owned check passes.

**Deliverable:** `results/experiment_6726_torx_factor_parity_repair.json`

### Exp6727: raw-chain autocorrelation schedule A/B

Compare fixed chromatic Gibbs and the preregistered autocorrelation-aware
schedule on identical bounded-treewidth fixtures, seeds, burn-in, sample
counts, and initial states. Save raw chains. Report likelihood error,
correlations, integrated autocorrelation time, effective sample size, wall
time, and update count. Make no FPGA, TSU, asymptotic, power, or speedup claim.

**Deliverable:** `results/experiment_6727_raw_chain_schedule_ab.json`

### Exp6728: V585 adversarial synthesis

Recompute branch dispositions from terminal artifacts. Reconcile the three PRD
gaps, exclusions, requirements, tests, model and hardware receipts, and next
actions. Record blocks and missing artifacts. Do not infer a downstream result
from a planned task or an upstream readiness flag.

**Deliverable:** `results/experiment_6728_v585_adversarial_synthesis.json`

## Dependency graph

```text
6715 ----\
          >-- 6717 --> 6718 --> 6719 --positive--> 6720 --> 6721
6716 ----/                  \
                             --> 6722 --> 6723 --> 6724

6725                         independent live ARC branch

6726 --> 6727                independent sampler branch

6728                         ungated synthesis of every terminal state
```

The YAML order follows the numeric order. The dependency graph does not force
execution branches to share a fragile scientific root. Exp6722 needs the
audited fixture and Exp6723 needs the proposal bank, but it does not need a
positive energy result. Exp6725 and Exp6726 are roots.

## Hardware requirements

| Resource | Tasks | Requirement |
|---|---|---|
| Dual RTX 3090 CUDA host | Exp6718, Exp6720, Exp6723 | Use owner-bound exclusive leases and independent llama.cpp processes. Run large models sequentially when VRAM requires it. |
| Local GGUF cache | Exp6718, Exp6720, Exp6723 | Resolve exact hub IDs, validate GGUF magic and hashes, and use embedded tokenizers. No legacy model may produce a headline row. |
| Host CPU and RAM | Exp6715-Exp6717, Exp6719, Exp6721-Exp6722, Exp6724-Exp6728 | Use bounded exhaustive planning, exact Ising references, raw-chain reducers, and cold audits. Preserve disk and RAM preconditions. |
| Attached FPGA boards | None required | KV260, PolarFire, and GateMate remain opportunistic. No branch may block on them. |
| Extropic TSU | None available | Torx is a software portability boundary only. No Z1 result is allowed. |

## Milestone success conditions

V585 succeeds operationally when all 14 tasks reach honest terminal states and
the final synthesis reconciles them. Scientific success is branch-specific:

- Planning: a replayable three-family bank plus an oracle-distinct held-family
  energy result; prefix-search benefit is optional and gated.
- Continuous learning: order-level benefit over the strongest control with all
  poison, retention, restart, and rollback checks passing.
- ARC: complete live held-family off/on rows with task-owned verification and
  no source, BFS, adapter, or solve claim.
- Sampling: exact Torx parity and replayable raw-chain schedule evidence.

A null, blocked, partial, or disqualified result is valid milestone evidence.
It must not be rewritten as support.

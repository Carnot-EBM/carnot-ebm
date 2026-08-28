# Carnot Research Roadmap vNEXT: Verifier-Grounded Planning Energy

**Created:** 2026-08-27  
**Milestone:** 2026.08.583  
**Status:** Planned; activates after milestone 2026.08.582 is archived  
**Experiments:** Exp6688-Exp6701  
**Supersedes:** the V582 `research-roadmap-vNEXT.md` design  
**Informed by:** Exp6674-Exp6687, the V583 literature refresh, FR-11, and FR-12

## Milestone Decision

V583 moves Carnot from output-format and intervention experiments to a bounded
but real planning corridor. Local flagship models propose action sequences.
Carnot assigns structured energy to typed plan states. An action-level search
may revise earlier choices. An exact dynamic program stays hidden from the live
selector and serves only as the evaluator and post-event learning authority.

The milestone also closes the installed-Torx sampler branch. Exp6684 already
proved mathematical factor parity to floating-point tolerance. Its only failed
check was an applicable end-to-end suite that now independently passes 5/5 on
the current tree. V583 qualifies that changed condition before generating raw
chains.

V583 does not rerun two V582 methods unchanged:

- The ARC supervisor produced zero intervention wins, three paired losses, and
  nine false interventions on nine eligible rows. It is paused until a new
  mechanism and genuine headroom exist.
- The three-family syntax-tail experiment reached no inference row because no
  exclusive accelerator lease was available. V583 uses a new scientific
  mechanism and owner-bound leases. It does not claim that the syntax-tail
  question was answered.

## What V582 Proved

| Branch | Experiments | Measured result | V583 consequence |
|---|---:|---|---|
| Execution integrity | 6674 | The 14-task V582 document and active manifest matched. Gate fields and prior-failure records were mechanically auditable. | Retain a first-task parity contract. |
| Triggered-tail fixture | 6675-6677 | The task-owned fixture was ready, but the GPU A/B produced zero inference rows because the exclusive-workload precondition was false. | Do not treat transport as tested. Leave it outside the new critical path. |
| Prospective repair memory | 6678-6680 | The fixture blocked on its own tests. The A/B gate-blocked and the audit never ran. | Change the learning surface from repair prose to typed plan-energy weights. |
| Live ARC outcomes | 6681-6682 | Exact post-action outcomes reached the canonical live seam for 30 eligible events. The supervisor treatment lost on three of nine paired rows and never won. | Retire the unchanged supervisor. Preserve the outcome transport for later mechanisms. |
| Exact Ising reference | 6683 | Eleven supported fixtures, 294 states, partition functions, probabilities, marginals, and correlations recomputed to at most `1.11e-16` probability error. | Reuse as the immutable sampler reference. |
| Torx typed factors | 6684 | All 97 factor rows and all 294 state rows matched the exact reference to floating-point tolerance. Readiness was false only because the then-applicable E2E command exited 1. | Requalify the changed E2E state before sampling. |
| Sampler schedule | 6685-6686 | Both tasks gate-blocked, so no raw chain or ESS claim exists. | Run only after producer-owned Torx qualification. |
| Milestone synthesis | 6687 | The milestone was partial: exact outcome transport landed; the supervisor was adverse; output, CSL, and stochastic comparisons remained blocked. | Pivot to one coherent planning-energy spine plus an independent sampler branch. |

## The Three Largest PRD Gaps

### Gap 1: FR-12 lacks a live, oracle-distinct planning path

Carnot can verify bounded artifacts and can score many offline candidates. It
has not shown that a learned structured energy changes the behavior of current
local flagship models on held-out constraint families. Prior generated-text
energy scorers are retired after repeated nulls. V583 uses typed state, action,
resource, and constraint features. It never trains on free-form rationales.

### Gap 2: FR-11 lacks successful prospective weight adaptation

Carnot has memory and rollback infrastructure, but V582 produced no prospective
learning comparison. The next experiment must update a real energy policy
between events, evaluate predictions before revealing labels, prevent future
label leakage, preserve old-family performance, and restore byte-exact state on
rejected updates.

### Gap 3: the stochastic substrate has parity but no chain-level result

The exact Ising reference and Torx energy map are mathematically valid. Carnot
still has no completed fixed-Gibbs versus autocorrelation-aware schedule result
with raw chains, likelihood error, autocorrelation, integrated autocorrelation
time, and effective sample size. This blocks the software-to-TSU portability
story in the PRD architecture.

## Research Findings Incorporated

The full dated sweep is recorded in `research-references.md` before this design.
V583 directly incorporates four findings:

- **Exact DP action values as dense verifier signals** (`2607.12856`). V583
  generates finite-horizon tasks whose exact action gaps can supervise a small
  local energy policy.
- **Verifier-assisted backtracking** (ICML 2025 OpenReview `9oIjvaDhoN`). V583
  tests action-level rollback under matched query budgets.
- **Verifier-guided optimization without weight changes to the LLM**
  (`2607.20478`). V583 updates only Carnot's compact energy policy. The mandated
  GGUF generators remain frozen.
- **Reasoning-trace noise can hide reliability signals** (`2607.22098`). V583
  excludes free-form rationales from learned energy features and release
  decisions.

DVI (`2510.05421`) and VeRA (`2602.13217`) remain supporting patterns. DVI
freezes the verifier while turning decisions into online supervision. VeRA uses
generated executable specifications to resist benchmark memorization.

## V583 Architecture

```text
Generated executable planning specification (four families)
         │
         ├── natural-language state + constraints
         ├── typed transition and cost functions
         └── exact DP tables and action gaps (sealed from live selection)
         │
         ▼
┌──────────────────────────────────────────────────────────────┐
│ Frozen local proposers                                      │
│ Qwen3.6-35B-A3B │ Gemma-4-31B │ Gemma-4-26B-A4B            │
└──────────────────────────────────────────────────────────────┘
         │ action proposals and untrusted rationales
         ▼
┌──────────────────────────────────────────────────────────────┐
│ Typed plan-state adapter                                    │
│ actions │ resource slack │ local violations │ objective rows │
└──────────────────────────────────────────────────────────────┘
         │
         ▼
┌───────────────────────────────┐
│ Carnot structural energy      │
│ learned on development rows   │
│ no exact future feature       │
└───────────────────────────────┘
         │
         ├──────────────┐
         ▼              ▼
 direct/search       action-level reject + backtrack
         │              │
         └──────┬───────┘
                ▼
       completed candidate plan
                │
                ▼
┌──────────────────────────────────────────────────────────────┐
│ Sealed exact evaluator                                      │
│ validity │ optimum │ regret │ per-action DP value            │
└──────────────────────────────────────────────────────────────┘
                │ post-event receipt only
                ▼
┌──────────────────────────────────────────────────────────────┐
│ Continuous energy learner                                   │
│ prequential update │ retention gate │ journal │ rollback     │
└──────────────────────────────────────────────────────────────┘

Independent portability branch:
exact Ising reference → installed Torx parity qualification → raw chains
→ likelihood/ACF/IAT/ESS comparison → cold audit
```

The exact evaluator is deliberately one-way. A plan must be selected before its
exact action values are revealed. Exact values may train the next event's energy
policy. They may not select the current event's answer.

## Phase 1: Execution Contract and Exact Planning Fixture

Phase 1 creates a dynamic, contamination-resistant benchmark. It contains four
finite-horizon families, generated parameters, natural-language templates,
typed transitions, hard constraints, costs, exact optima, and exact per-action
values. Horizons and action sets remain small enough for independent exhaustive
checks.

### Exp6688: V583 document-to-manifest parity contract

- Compare the design and YAML task IDs, order, deliverables, routes, gates,
  producer fields, prior-failure records, and run commands.
- Require exactly Exp6688 through Exp6701.
- Set `v583_manifest_parity_ready=true` only after every activation-hard check
  passes.
- **Deliverable:** `results/experiment_6688_v583_manifest_parity_contract.json`

### Exp6689: generated exact planning fixture

- Generate 32 headline instances across inventory, battery dispatch, job-slot,
  and reservoir-control families. Generate separate development instances.
- Cap horizons at eight and action sets at five. Retain every transition and
  dynamic-programming row.
- Emit natural-language prompts without answers, typed executable specs,
  optimum plans, action gaps, family splits, and mutation attacks.
- Add `REQ-*` and `SCENARIO-*` anchors before implementation.
- Set `planning_fixture_ready=true` only when all exact and metamorphic checks
  pass.
- **Deliverable:** `results/experiment_6689_exact_planning_fixture.json`

### Exp6690: cold planning-fixture audit

- Recompute a blinded subset with an independent exhaustive solver.
- Audit family separation, prompt-answer leakage, label sealing, dynamic
  consistency, optimum ties, infeasible states, and mutation detection.
- Set `planning_fixture_audit_passed=true` only from raw audit rows.
- **Deliverable:** `results/experiment_6690_exact_planning_fixture_audit.json`

## Phase 2: Flagship Proposals and Energy-Guided Backtracking

Phase 2 measures all three mandated local GGUF families on the same frozen
planning instances. It then trains an oracle-distinct structural energy and
tests whether that energy helps action-level search under fixed query budgets.

### Exp6691: three-family SOTA planning proposal corpus

- Use `cached_sota_pair()` and `resolve_cached_gguf()` with:
  `unsloth/Qwen3.6-35B-A3B-GGUF`,
  `unsloth/gemma-4-31B-it-GGUF`, and
  `unsloth/gemma-4-26B-A4B-it-GGUF`.
- Run one planned call per model-instance unit. Retain up to four action
  proposals from the unmodified response. Do not retry malformed outputs.
- Record parse yield, candidate diversity, exact headroom, regret, token cost,
  latency, CUDA process identity, and every raw response.
- Set `proposal_corpus_ready=true` on completeness, not on model success.
- **Deliverable:** `results/experiment_6691_sota_planning_proposal_corpus.json`

### Exp6692: oracle-distinct structural plan energy

- Fit a small pairwise-margin energy from typed plan features and sealed
  development labels. Exclude rationales, model identity, answer strings, task
  IDs, and exact future values from inputs.
- Compare against proposal order, hard-violation count, immediate cost, and
  randomized controls with leave-one-family-out evaluation.
- Report AUROC, rank correlation, top-1 regret, calibration, and paired
  intervals per family.
- Set `energy_generalization_supported=true` only if the preregistered held-
  family regret improvement over the strongest nonlearned baseline is positive
  with its interval excluding zero and all leakage attacks pass.
- **Deliverable:** `results/experiment_6692_structural_plan_energy.json`

This task acknowledges the retired Exp5163 external-text scorer. The new attempt
changes both the representation and the inference seam: typed executable plan
features replace generated-text and logprob scoring, and the energy guides
action search rather than ranking prose answers.

### Exp6693: energy-guided action backtracking A/B

- Gate on `energy_generalization_supported=true`.
- Use the Qwen3.6-35B-A3B and Gemma-4-31B flagship pair.
- Compare direct sequential proposals, hard-prefix rejection with backtracking,
  and hard-prefix plus learned-energy backtracking.
- Match instances, seeds, action-query budgets, token budgets, stopping rules,
  and parser behavior. The exact DP evaluator stays sealed until plan commit.
- Report exact validity, normalized regret, queries, backtracks, tokens, harmful
  flips, no-headroom units, wins, losses, ties, and intervals.
- **Deliverable:** `results/experiment_6693_energy_backtracking_ab.json`

### Exp6694: cold search and leakage audit

- Recompute all claims from raw model transcripts and plan rows.
- Prove the exact evaluator was not called before commitment.
- Attack model/order leakage, label access, task IDs, rationale features,
  budget drift, silent retries, missing rows, and post-hoc stopping.
- Set `energy_search_audit_passed=true` only when every claim survives.
- **Deliverable:** `results/experiment_6694_energy_backtracking_audit.json`

## Phase 3: Continuous Self-Learning of the Energy Policy

Phase 3 is the required continuous-self-learning experiment. It implements
Tier 1 from `research-program.md`: online updates to Carnot's small energy
policy. The LLM weights and exact evaluator remain frozen.

### Exp6695: immutable online-energy update fixture

- Add event-boundary snapshots, append-only update receipts, exact admission,
  protected-family retention, restart replay, and byte-exact rollback.
- Reject duplicate, reordered, poisoned, future-labeled, and retention-breaking
  updates.
- Set `online_energy_fixture_ready=true` only when every update and rollback
  attack passes task-owned tests and applicable E2E checks.
- **Deliverable:** `results/experiment_6695_online_energy_update_fixture.json`

### Exp6696: prospective prequential online-energy A/B

- Generate and seal a new planning stream before exposing any exact label.
- Use all three mandated GGUF families to create the matched proposal pool.
- Compare frozen energy, immediate write-through online energy, and exact-
  admitted retention-safe online energy.
- For each event: predict first, hash the pre-event state, reveal exact values,
  update between events, then probe old-family retention.
- Replicate family orders. Report prequential regret, exact validity, learning
  curves, update acceptance, rollback, old-family retention, negative transfer,
  restart equivalence, and intervals.
- Set `csl_improvement_supported=true` only if the safe online arm improves the
  primary prequential metric with its interval excluding zero, has no protected-
  family regression, and passes every safety attack.
- **Deliverable:** `results/experiment_6696_prequential_online_energy_ab.json`

### Exp6697: cold prequential and rollback audit

- Replay the sealed event stream and journal in a fresh process.
- Recompute every pre-event prediction, update, learning curve, retention
  probe, rollback, and aggregate from raw bytes.
- Attack future-label access, snapshot aliasing, duplicate admission, order
  dependence, restart divergence, poisoned rewards, selective family omission,
  and missing events.
- Set `csl_audit_passed=true` only when the result is reproducible.
- **Deliverable:** `results/experiment_6697_online_energy_csl_audit.json`

## Phase 4: Torx Sampler Qualification and Milestone Synthesis

Phase 4 is independent of Phases 2 and 3 after the manifest contract. It
qualifies the changed E2E state, produces raw chains, audits statistics, and
closes the milestone without coercing blocked branches to zero.

### Exp6698: installed-Torx factor qualification

- Reuse the frozen Exp6683 exact reference and Exp6684 factor/state rows.
- Rerun task-owned parity tests, scoped coverage, and the applicable E2E command.
- Diagnose any changed failure. Do not weaken an applicable check.
- Set `torx_factor_parity_qualified=true` only when mathematics, runtime API,
  focused tests, and applicable E2E checks pass.
- **Deliverable:** `results/experiment_6698_torx_factor_qualification.json`

### Exp6699: autocorrelation-aware schedule A/B

- Gate on `torx_factor_parity_qualified=true`.
- Compare fixed-temperature Gibbs and the preregistered schedule at matched
  update budgets across fixture-arm-seed chains.
- Retain every raw state, energy, temperature, transition, update count, and
  monotonic timestamp.
- Report distribution error, marginal and correlation error, ACF, IAT, ESS,
  ESS/update, ESS/second, sensitivity, and paired intervals.
- **Deliverable:** `results/experiment_6699_autocorrelation_schedule_ab.json`

### Exp6700: cold raw-chain sampler audit

- Recompute exact probabilities and all chain statistics independently from raw
  bytes.
- Audit chain independence, seeds, burn-in, truncation, temperature labels,
  update order, coefficient precision, fixture coverage, and missing chains.
- Set `sampler_audit_passed=true` only when every reported statistic matches.
- **Deliverable:** `results/experiment_6700_stochastic_portability_audit.json`

### Exp6701: V583 branch synthesis

- Enumerate all 14 task states and recompute every available headline from raw
  rows.
- Preserve `positive`, `circular_positive`, `null`, `blocked`, `disqualified`,
  and `partial` without pooling branches.
- Reconcile capability specs, traceability, status, changelog, and completed
  research only after the synthesis artifact exists.
- **Deliverable:** `results/experiment_6701_v583_branch_synthesis.json`

## Dependency Graph

```text
Exp6688 manifest parity
  ├── Exp6689 exact planning fixture
  │     └── Exp6690 cold fixture audit
  │            └── Exp6691 all-three-GGUF proposal corpus
  │                   └── Exp6692 structural plan energy
  │                          └── [energy_generalization_supported]
  │                                  └── Exp6693 backtracking A/B
  │                                         └── Exp6694 search audit
  │
  │                   └── Exp6695 online-update fixture
  │                          └── Exp6696 prequential CSL A/B
  │                                 └── Exp6697 CSL audit
  │
  └── Exp6698 Torx factor qualification
          └── Exp6699 schedule A/B
                 └── Exp6700 sampler audit

Exp6701 synthesis is deliberately ungated and reads every terminal state.
```

All structured gate fields are declared with identical spelling in the
producer's required artifact fields. No gate references a task outside V583.

## Conductor Order

| Order | ID | Phase | Route | GPU | Estimated wall time |
|---:|---|---|---|---:|---:|
| 1 | Exp6688 | Contract | Claude Opus | no | 45 min |
| 2 | Exp6689 | Fixture | Codex gpt-5.6-sol | no | 120 min |
| 3 | Exp6690 | Fixture audit | Claude | no | 60 min |
| 4 | Exp6691 | Proposal corpus | Claude | yes | 240 min |
| 5 | Exp6692 | Structural energy | Codex gpt-5.6-sol | no | 150 min |
| 6 | Exp6693 | Backtracking A/B | Codex gpt-5.6-sol | yes | 300 min |
| 7 | Exp6694 | Search audit | Claude | no | 90 min |
| 8 | Exp6695 | Online fixture | Codex gpt-5.6-sol | no | 120 min |
| 9 | Exp6696 | Prequential CSL | Claude | yes | 300 min |
| 10 | Exp6697 | CSL audit | Claude | no | 90 min |
| 11 | Exp6698 | Torx qualification | Claude Opus | no | 90 min |
| 12 | Exp6699 | Schedule A/B | Codex gpt-5.6-sol | no | 150 min |
| 13 | Exp6700 | Sampler audit | Claude | no | 90 min |
| 14 | Exp6701 | Synthesis | Claude Opus | no | 150 min |

## Hardware Requirements

| Tasks | Required substrate | Resource contract |
|---|---|---|
| 6688-6690 | CPU, 128 GB host RAM, local disk | No LLM. Exact DP and audit only. |
| 6691 | Dual RTX 3090, owner-bound llama.cpp CUDA processes | All three mandated GGUF families. Sequential per-device model ownership is allowed. No CPU headline fallback. |
| 6692 | CPU; optional CUDA for small energy training | No LLM call. Training must fit on the local host. |
| 6693 | Dual RTX 3090, owner-bound llama.cpp CUDA processes | Qwen3.6-35B-A3B plus Gemma-4-31B. Stop with a blocked artifact if a flagship cannot run on CUDA. |
| 6694-6695 | CPU, local raw stores | No LLM. Audit and update fixture only. |
| 6696 | Dual RTX 3090, owner-bound llama.cpp CUDA processes | All three mandated GGUF families create the sealed proposal stream. Energy updates are local and small. |
| 6697 | CPU, immutable stream and journal | Fresh-process audit. No new model generation. |
| 6698-6700 | CPU, installed `extro-torx`/JAX, exact fixtures | Software-only. No TSU or FPGA claim. Raw chains may use local CPU/JAX. |
| 6701 | CPU and repository artifacts | Synthesis only. No LLM inference. |

KV260, PolarFire, and GateMate are not blocking resources. Their state has not
changed. KV260 and PolarFire remain opportunistic. GateMate remains blocked on
physical/JTAG identification. Extropic Z1 access remains planned for 2027.

## Model Contract

Every task that invokes an LLM declares `MODEL_SPECS` and uses at least one of:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

Exp6691 and Exp6696 use all three. Exp6693 uses the flagship MoE and flagship
dense pair. Legacy Qwen3.5-0.8B and Gemma-4-E4B may run CPU smoke checks only.
They cannot populate headline rows. GGUF tokenizers come from embedded metadata.

## Claim Boundaries

- Exact DP defines correctness and post-event learning labels. It is sealed from
  current-event selection.
- A structural-energy claim is oracle-distinct only when features exclude exact
  future values, labels, answers, task IDs, model IDs, and rationales.
- Ready infrastructure normally has `verdict_class: null`.
- A positive claim with `verifier_is_oracle: true` is
  `circular_positive`, never `positive`.
- Every comparison emits per-unit rows, wins, losses, ties, no-headroom rows,
  missing rows, intervals, and row-derived aggregates.
- A blocked artifact names the exact failed check and value in
  `gate_check_summary`.
- No software result implies TSU, FPGA, latency, power, or asymptotic hardware
  performance.
- V583 makes no ARC game or level solve claim.

## Prior-Failure Discipline

The execution manifest records changed conditions for all matching failed or
retired scopes:

- Exp5163 external generated-text scoring → Exp6692 typed executable plan
  energy with a different inference seam.
- Exp6678 blocked repair-memory stream → Exp6695 typed online-energy fixture.
- Exp6679 gate-blocked prospective memory A/B → Exp6696 new weight-adaptation
  surface after its own fixture.
- Exp6684 parity blocked by E2E → Exp6698 reruns after the independently observed
  5/5 changed E2E state.
- Exp6685 schedule gate block → Exp6699 runs only after a producer-owned V583
  qualification field.
- Exp6686 missing audit → Exp6700 runs only after V583 raw chains exist.

Every entry carries `retire_if_same_verdict: true`.

## Promotion Gates

V583 succeeds scientifically only if raw rows support the claim. Execution of
all tasks is not itself success.

- **Planning fixture:** exact and cold solvers agree on every audited unit.
- **Energy:** held-family top-1 regret improves over the strongest nonlearned
  baseline with an interval excluding zero and no leakage attack.
- **Backtracking:** treatment beats hard-prefix backtracking under matched
  budgets without harmful-flip or query-cost concealment.
- **Continuous self-learning:** the retention-safe online arm improves
  prequential regret with an interval excluding zero, no protected-family
  regression, and byte-exact rollback.
- **Sampler:** raw-chain distribution metrics and ESS recompute in the cold
  audit. Ground-state frequency alone is insufficient.

## Deferred

- General open-domain natural-language constraint extraction beyond the four
  executable planning families.
- Training or modifying the GGUF model weights.
- Hidden-state verifier work until a local extractor exists for a mandated
  flagship base.
- Another ARC supervisor comparison until a new mechanism has genuine
  intervention headroom.
- Another syntax-tail run until the operator wants to resolve the environment-
  lease question directly.
- Any attached-board or Z1 performance claim without changed authenticated
  hardware evidence.

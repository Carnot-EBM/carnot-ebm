# Carnot Research Roadmap vNEXT: Recovery, Experience Graphs, and Exact Replay

**Created:** 2026-08-28  
**Milestone:** 2026.08.584  
**Status:** Planned; activates after milestone 2026.08.583 is archived  
**Experiments:** Exp6702-Exp6714  
**Supersedes:** the V583 `research-roadmap-vNEXT.md` design  
**Informed by:** Exp6688-Exp6697 terminal states, Exp6683-Exp6686 sampler evidence,
the V584 literature refresh, FR-11, and FR-12

## Milestone Decision

V584 recovers the scientific questions that V583 did not reach. It removes
runtime manifest parity from the experiment graph. Document-manifest parity is
a pre-activation property. The planner and conductor validators must establish
it before any experiment starts.

The milestone has three scientific branches:

1. Build an exact finite-horizon planning fixture. Use all three required local
   GGUF families to create one frozen proposal bank. Test an oracle-distinct
   structural energy and, only if it transfers, energy-guided prefix search.
2. Test continuous self-learning through a bounded typed experience graph. The
   graph stores exact-checker receipts while the GGUF models stay frozen.
3. Finish the software sampler question. Qualify Torx factors against the exact
   Ising reference, then compare raw chains and audit every statistic.

The branches share evidence where useful. They do not share one fragile root.
The final synthesis is ungated and records every terminal state.

## What V583 Proved

V583 reached terminal conductor states. It did not produce a scientific result.
That distinction defines this milestone.

| V583 evidence | Measured result | V584 consequence |
|---|---|---|
| Active manifest | The activated YAML contained 10 tasks, Exp6688-Exp6697. The design document described 14 tasks, Exp6688-Exp6701. | Validate task IDs, order, deliverables, routes, gates, and producer fields before activation. Do not spend a research task on parity. |
| Exp6688 runtime parity task | Three attempts hit the hard wall clock after about 4,800 seconds. No terminal artifact exists. | Remove the task and its dependency edge. Keep activation validation outside the science graph. |
| Exp6689, Exp6691, Exp6693, Exp6695, Exp6697 | The conductor preempted them after their upstream task retired or blocked. They emitted no scientific artifact. | Reattempt only with explicit `prior_failures` and a changed upstream contract. |
| Exp6690, Exp6692, Exp6694, Exp6696 | Each artifact reported `blocked_gate_check_failed` because its producer artifact was absent. | Keep exact producer-field spelling. Gate only on producers in V584. |
| Planned sampler tasks | Exp6698-Exp6701 were present in the document but absent from the active manifest. | Allocate new IDs. Put all sampler tasks in both files. Validate exact task-set equality now. |
| Scientific claims | No planning fixture, model proposal bank, structural energy, backtracking comparison, or prospective CSL stream ran. | Preserve the pre-V583 evidence boundary. V584 starts from the last authentic artifacts, not from planned V583 claims. |

V583 also proved that `gate_check_summary` localizes missing producers. That
diagnostic worked. The failure was the task graph and activation contract, not
the gate evaluator.

## The Three Largest PRD Gaps

### Gap 1: FR-12 lacks live oracle-distinct planning evidence

Carnot has exact verifiers and many offline discriminators. It has not shown
that a learned typed energy improves a current local flagship model on held-out
constraint families. Generated-text energy scoring is retired. V584 uses typed
state, action, resource, and violation features. It excludes rationales, answer
strings, model identity, task IDs, and exact future values.

### Gap 2: FR-11 lacks independently audited prospective self-learning

Some recent memory experiments produced positive point estimates. Their cold
audits narrowed or nullified the claim. V583 did not run its online learner.
V584 tests a different mechanism: a typed experience graph with immutable
exact-checker receipts, bounded retrieval, between-event writes, retention
checks, restart replay, and byte-exact rollback. The LLM weights stay frozen.

### Gap 3: the stochastic substrate lacks raw-chain evidence

Exp6683 established an exact bounded-treewidth Ising reference. Exp6684 matched
97 factor rows and 294 state rows to floating-point tolerance, but its
applicable end-to-end command failed. The schedule and audit then blocked.
Carnot still lacks a complete fixed-Gibbs versus autocorrelation-aware schedule
comparison with raw chains, likelihood error, correlations, integrated
autocorrelation time, and effective sample size.

## Research Findings Incorporated

The dated source sweep appears in `research-references.md` before this design.
V584 directly incorporates these findings:

- **KOPE experience graph memory** (`2608.25570`). Store decisions, observed
  exact outcomes, later uses, and alternative branches. Retrieve a bounded
  context. Keep the foundation model frozen. V584 applies this to typed plan
  events and exact checker receipts.
- **Test-time scaling regime and replay discipline** (`2608.04001`). Compare
  the full inference system under matched budgets. Save proposal banks, seeds,
  verifier calls, stopping rules, and raw per-instance replay data.
- **Verifier-assisted backtracking** (OpenReview `9oIjvaDhoN`). Test prefix
  search only after the structural energy transfers on held families. Keep the
  exact solver sealed until plan commitment.
- **Exact action-value supervision** (`2607.12856`). Use exact dynamic-program
  action gaps as development labels and post-event learning receipts. Never
  expose them to current-event selection.
- **Executable generated fixtures** (`2602.13217`) and immutable verifier
  feedback (`2510.05421`). Generate coherent tasks, freeze the verifier, and
  use its decisions as update authority.

DCGC (`2608.25428`) is watch-only. Its masked-diffusion substrate does not match
the required autoregressive GGUF models. No substitute checkpoint is allowed.

## V584 Architecture

```text
Generated exact planning families
  natural language + typed transitions + hard constraints
  exact optima and action values sealed from live selection
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│ Frozen local proposal models                                │
│ Qwen3.6-35B-A3B │ Gemma-4-31B │ Gemma-4-26B-A4B            │
└──────────────────────────────────────────────────────────────┘
                         │
                         ▼
              frozen raw proposal bank
                  │                 │
                  ▼                 ▼
┌────────────────────────────┐  ┌─────────────────────────────┐
│ Typed structural energy    │  │ Typed experience graph      │
│ development labels only    │  │ exact receipt on each write │
│ no future-value feature    │  │ bounded retrieval context   │
└────────────────────────────┘  └─────────────────────────────┘
          │                              │
          ▼                              ▼
 direct / fixed best-of-N /        prospective no-memory /
 energy-guided prefix search       read-only / admitted-memory
          │                              │
          └──────────────┬───────────────┘
                         ▼
┌──────────────────────────────────────────────────────────────┐
│ Sealed exact evaluator                                      │
│ validity │ optimum │ regret │ action gaps │ update authority │
└──────────────────────────────────────────────────────────────┘
                         │ post-event receipts only
                         ▼
             retention │ journal │ rollback │ replay

Independent stochastic branch:
Exp6683 exact Ising rows → Torx factor qualification → raw chains
→ fixed versus scheduled Gibbs → cold statistic audit
```

The exact evaluator is one-way. A plan must commit before exact values become
readable. Exact values may authorize a later graph write. They may not select
the current answer. A positive learned-verifier result therefore remains
oracle-distinct. Any accidental live exact access disqualifies the claim.

## Pre-Activation Contract

V584 has no runtime parity experiment. Before activation, run the repository
schema, prior-failure, gate-cross-reference, and custom task-set checks. The two
planning files must contain exactly the same 13 IDs in the same order. All
gated producers must exist in V584 and declare the exact gate field in their own
required artifact fields.

This contract changes the V583 failure mode. A bad manifest is refused before
agent time. It cannot retire the first scientific producer and cascade through
the milestone.

## Phase 1: Exact Planning Recovery

Phase 1 creates and independently audits a contamination-resistant planning
fixture. It has four finite-horizon families: inventory, battery dispatch,
job-slot allocation, and reservoir control. Horizons and action spaces remain
small enough for an independent exhaustive solver.

### Exp6702: generated exact planning fixture recovery

- Generate at least 32 headline instances and separate development instances.
- Retain prompts, typed specs, transitions, feasibility, objectives, exact
  optima, every action value, ties, and held-family splits.
- Add metamorphic and mutation rows. Do not depend on an operational preflight
  experiment.
- Set `planning_fixture_ready=true` only when exact, leakage, and task-owned
  test checks pass.
- **Deliverable:** `results/experiment_6702_exact_planning_fixture_recovery.json`

### Exp6703: cold planning fixture audit

- Recompute a blinded subset with an independent exhaustive solver.
- Audit prompt-answer leakage, family separation, infeasible states, optimum
  ties, action gaps, label seals, and mutations.
- Set `planning_fixture_audit_passed=true` only from raw audit rows.
- **Deliverable:** `results/experiment_6703_exact_planning_fixture_audit.json`

## Phase 2: Flagship Proposals and Oracle-Distinct Search

Phase 2 creates one reusable proposal bank from all three required GGUF models.
It then tests typed structural energy on held families. The expensive prefix
search runs only if that held-family result is positive under its preregistered
interval and leakage rules.

### Exp6704: three-family frozen planning proposal bank

- Use all three mandated model families through owner-bound llama.cpp CUDA
  processes.
- Make one call per model-instance-seed unit. Keep malformed and failed rows.
  Do not silently retry.
- Retain raw responses, parsed actions, proposal sets, tokens, latency, process
  receipts, exact headroom, validity, and regret.
- Set `proposal_bank_ready=true` on complete execution and replayability, not on
  model quality.
- **Deliverable:** `results/experiment_6704_sota_planning_proposal_bank.json`

### Exp6705: held-family structural plan energy

- Fit a small pairwise energy from typed development rows.
- Exclude rationales, text embeddings, answer strings, task IDs, model IDs, and
  exact future values.
- Compare against proposal order, hard-violation count, immediate cost, and a
  randomized control with leave-one-family-out evaluation.
- Report AUROC, calibration, rank correlation, top-choice validity, regret,
  independent family deltas, and intervals.
- Set `structural_energy_ready=true` on complete replayable evaluation. Set
  `energy_generalization_supported=true` only when held-family regret improves
  over the strongest nonlearned baseline, its interval excludes zero, and all
  leakage attacks pass.
- **Deliverable:** `results/experiment_6705_structural_plan_energy.json`

### Exp6706: matched-budget energy-guided prefix search

- Gate on `energy_generalization_supported=true`.
- Use Qwen3.6-35B-A3B and Gemma-4-31B. Compare direct choice, fixed-budget
  best-of-N, hard-prefix backtracking, and hard-prefix plus learned energy.
- Match model-instance seeds, candidate calls, token ceilings, verifier calls,
  stopping rules, and parser behavior.
- Keep the exact dynamic program sealed until final commitment.
- Report validity, normalized regret, tokens, queries, backtracks, harmful
  flips, wins, losses, ties, no-headroom rows, and intervals.
- Set `prefix_search_ready=true` on complete matched-budget evidence.
- **Deliverable:** `results/experiment_6706_energy_guided_prefix_search.json`

### Exp6707: cold search, budget, and leakage audit

- Recompute all search headlines from raw transcripts and replay rows.
- Audit exact-evaluator timing, hidden retries, model or order leakage, task ID
  leakage, budget drift, stopping drift, missing rows, and independence units.
- Set `prefix_search_audit_passed=true` only when every claim survives.
- **Deliverable:** `results/experiment_6707_prefix_search_audit.json`

## Phase 3: Continuous Self-Learning with Experience Graph Memory

Phase 3 is the required continuous-self-learning branch. It implements Tier 1
from `research-program.md`. Carnot updates an external typed memory between
events. It does not update GGUF weights. The branch depends on the exact fixture
and frozen proposal bank. It does not depend on a positive structural-energy
result.

### Exp6708: verifier-backed experience graph fixture

- Store typed state, action, constraint, outcome, exact receipt, alternative
  branch, parent, later-use, and provenance edges.
- Enforce a fixed retrieval-token budget and deterministic relevance policy.
- Admit only post-commit exact-checker-backed rows between events.
- Reject future labels, duplicates, reordered parents, poison, unsupported
  edits, retention breaks, corrupt journals, and partial writes.
- Prove restart replay and byte-exact rollback.
- Set `experience_graph_fixture_ready=true` only when every safety and
  durability check passes.
- **Deliverable:** `results/experiment_6708_experience_graph_fixture.json`

### Exp6709: prospective experience-graph CSL A/B

- Generate and seal new planning-event orders before inference.
- Use all three required GGUF families on matched event pools.
- Compare no memory, read-only retrieval, unverified write-through memory, and
  exact-admitted bounded graph memory.
- For each event: predict, commit, reveal exact values, consider the write,
  check protected-family retention, then advance.
- Use independent order replicates. Report prequential regret, exact validity,
  learning speed, accepted writes, retrieval use, false retrievals, token cost,
  negative transfer, rollback, restart equivalence, and intervals.
- Set `experience_graph_csl_ready=true` on complete prospective evidence. Set
  `csl_improvement_supported=true` only if the admitted graph improves the
  primary order-level metric with its interval excluding zero, causes no
  protected-family regression, and passes every safety attack.
- **Deliverable:** `results/experiment_6709_experience_graph_csl_ab.json`

### Exp6710: cold experience-graph CSL audit

- Replay the sealed streams, proposal stores, journals, retrievals, retention
  probes, rollbacks, and restarts in a fresh process.
- Recompute every pre-event decision and order-level statistic.
- Attack future access, aliasing, duplicate admission, dropped events, family
  omission, poison, retrieval overflow, rollback mismatch, and independence
  inflation.
- Set `experience_graph_csl_audit_passed=true` only when raw bytes reproduce
  the reported claim.
- **Deliverable:** `results/experiment_6710_experience_graph_csl_audit.json`

## Phase 4: Torx Raw Chains and Milestone Synthesis

The sampler branch is independent of Phases 1-3. It uses CPU/JAX and the
installed Torx stack. It makes no TSU or FPGA performance claim.

### Exp6711: installed-Torx factor qualification

- Reuse Exp6683 exact reference rows and Exp6684 factor/state rows.
- Diagnose the prior applicable-E2E failure. Do not weaken the check.
- Recompute energy and probability parity through the installed Torx API.
- Set `torx_factor_parity_qualified=true` only when mathematics, runtime API,
  focused tests, scoped coverage, and applicable E2E checks pass.
- **Deliverable:** `results/experiment_6711_torx_factor_qualification.json`

### Exp6712: autocorrelation-aware schedule A/B

- Gate on `torx_factor_parity_qualified=true`.
- Compare fixed-temperature Gibbs and the preregistered schedule at matched
  update budgets over fixture-arm-seed chains.
- Retain every state, energy, temperature, transition, update, seed, and
  monotonic timestamp.
- Report likelihood error, marginal and pair-correlation error, ACF, IAT, ESS,
  ESS/update, ESS/second, sensitivity, and paired intervals.
- Set `sampler_schedule_ready=true` on complete raw-chain evidence. Set
  `sampler_schedule_supported=true` only when the primary ESS/update interval
  excludes zero and exact-distribution error stays within tolerance.
- **Deliverable:** `results/experiment_6712_autocorrelation_schedule_ab.json`

### Exp6713: cold raw-chain sampler audit

- Recompute exact probabilities and all chain statistics from raw bytes.
- Audit seeds, chain independence, burn-in, truncation, temperature labels,
  update order, coefficient precision, fixture coverage, and missing chains.
- Set `sampler_audit_passed=true` only when every statistic and verdict matches.
- **Deliverable:** `results/experiment_6713_stochastic_portability_audit.json`

### Exp6714: V584 branch evidence synthesis

- Enumerate all 13 planned task states and all expected deliverables.
- Recompute each available headline from per-unit rows.
- Keep planning, search, CSL, and sampler claims separate. Never coerce blocked
  or missing branches to zero.
- Preserve the closed `verdict_class` enum and oracle-circularity rules.
- Reconcile capability specs, `_bmad/traceability.md`, `ops/status.md`,
  `ops/changelog.md`, and `research-complete.yaml` after the synthesis artifact
  exists.
- **Deliverable:** `results/experiment_6714_v584_branch_synthesis.json`

## Dependency Graph

```text
Exp6702 exact planning fixture
  └── Exp6703 cold fixture audit
        ├── Exp6704 all-three-GGUF proposal bank
        │     ├── Exp6705 structural plan energy
        │     │      └── [energy_generalization_supported]
        │     │              └── Exp6706 prefix search A/B
        │     │                    └── Exp6707 search audit
        │     └──────────────────────────────┐
        └── Exp6708 experience graph fixture│
                         └───────────────────┴── Exp6709 CSL A/B
                                                   └── Exp6710 CSL audit

Exp6711 Torx factor qualification
  └── Exp6712 schedule A/B
        └── Exp6713 sampler audit

Exp6714 synthesis is deliberately ungated and reads every terminal state.
```

No gate references a retired task or a task outside V584. Every gate field uses
the spelling declared in its producer's required artifact fields.

## Conductor Order

| Order | ID | Phase | Route | GPU | Estimated wall time |
|---:|---|---|---|---:|---:|
| 1 | Exp6702 | Exact fixture | Codex gpt-5.6-sol | no | 120 min |
| 2 | Exp6703 | Fixture audit | Codex gpt-5.6-sol | no | 75 min |
| 3 | Exp6704 | Proposal bank | Claude | yes | 300 min |
| 4 | Exp6705 | Structural energy | Codex gpt-5.6-sol | no | 150 min |
| 5 | Exp6706 | Prefix search | Claude | yes | 300 min |
| 6 | Exp6707 | Search audit | Claude | no | 90 min |
| 7 | Exp6708 | Experience graph fixture | Codex gpt-5.6-sol | no | 150 min |
| 8 | Exp6709 | Prospective CSL | Claude | yes | 360 min |
| 9 | Exp6710 | CSL audit | Claude | no | 120 min |
| 10 | Exp6711 | Torx qualification | Claude Opus | no | 120 min |
| 11 | Exp6712 | Schedule A/B | Codex gpt-5.6-sol | no | 180 min |
| 12 | Exp6713 | Sampler audit | Claude | no | 90 min |
| 13 | Exp6714 | Synthesis | Claude Opus | no | 150 min |

## Hardware Requirements

| Tasks | Required substrate | Resource contract |
|---|---|---|
| 6702-6703 | CPU, 128 GB host RAM, local disk | No LLM. Exact dynamic programming and independent audit. |
| 6704 | Two RTX 3090 GPUs, owner-bound llama.cpp CUDA processes | All three mandated GGUF families. Sequential model ownership is allowed. No CPU headline fallback. |
| 6705 | CPU; optional CUDA for the small energy | No LLM call. Training must fit locally. |
| 6706 | Two RTX 3090 GPUs, owner-bound llama.cpp CUDA processes | Qwen3.6-35B-A3B and Gemma-4-31B. Block if a required headline model cannot run on CUDA. |
| 6707-6708 | CPU and immutable local stores | No new LLM inference. Audit and graph fixture only. |
| 6709 | Two RTX 3090 GPUs, owner-bound llama.cpp CUDA processes | All three mandated GGUF families. Freeze proposal pools before arm-specific updates. |
| 6710 | CPU and immutable stream, graph, and journal bytes | Fresh-process audit. No new model generation. |
| 6711-6713 | CPU, installed `extro-torx`/JAX, exact fixtures | Software-only. No TSU or FPGA claim. |
| 6714 | CPU and repository artifacts | Synthesis only. No LLM inference. |

KV260, PolarFire, and GateMate are not blocking resources. Their receipts have
not changed. KV260 and PolarFire remain opportunistic. GateMate remains blocked
on physical and JTAG identification. Extropic Z1 access remains planned for
2027. No V584 claim may substitute simulator results for device measurements.

## Model Contract

Every task that invokes an LLM must declare `MODEL_SPECS` and use at least one
of these exact hub IDs:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

Exp6704 and Exp6709 use all three. Exp6706 uses the flagship MoE and flagship
dense model. Resolve local files through `cached_sota_pair()` and
`resolve_cached_gguf()`. `Qwen3.5-0.8B` and `gemma-4-E4B-it` may run CPU smoke
tests only. They may not support headline results. A missing CUDA-capable
headline model produces a blocked artifact. It never triggers a legacy-model
substitution.

## Prior-Failure Discipline

Every reattempted scope declares the prior experiment, its honest verdict, the
changed condition, and `retire_if_same_verdict: true` in the YAML. The main
changed conditions are:

- V583's runtime parity dependency is gone. The exact fixture is a root task.
- Every V583 missing-producer gate now points to a same-milestone producer.
- Experience graph memory replaces free-form repair memory and the unexecuted
  online-energy plan. It uses typed exact receipts and bounded retrieval.
- Torx qualification diagnoses the prior applicable-E2E failure before it
  permits raw chains.

No task reuses a retired experiment ID. No `requires:` edge names a retired
upstream task.

## Evidence and Verdict Rules

Every task writes a terminal artifact, including blocked tasks. Every task
declares `verdict_class` as one of `positive`, `circular_positive`, `null`,
`blocked`, `disqualified`, or `partial`. Every blocked artifact uses the exact
field `gate_check_summary` and names the failed check plus observed value.

Every comparative task emits one row per model, instance, seed, arm, order, or
chain as applicable. Aggregates are recomputed from those rows. Missing and
blocked units remain missing or blocked. They are never converted to losses or
zeros.

Successful terminal verdicts begin with `complete_`, `success_`, `passed_`, or
`shipped_`. Genuine precondition failures begin with `blocked_`. A positive
structural-energy or CSL claim is forbidden if the exact evaluator influenced
the current decision. Such a run is disqualified, not circular-positive,
because the milestone preregisters oracle-distinct behavior.

## Verification and Reconciliation

Before handoff, validate the planning files with:

```bash
.venv/bin/python scripts/validate_prior_failures.py research-roadmap-next.yaml
.venv/bin/python scripts/audit_roadmap_gates.py research-roadmap-next.yaml
```

Also parse the YAML through `scripts/roadmap_schema.py`, compare document and
YAML IDs and deliverables, confirm all 13 prompts end with the mandated run and
do-not-modify lines, and run the relevant roadmap-schema and gate-audit tests.

During execution, each task must run focused unit tests, scoped coverage,
spec-coverage checks, artifact and row consistency checks, adversarial
verification, and applicable checks from `ops/e2e-test-plan.md`. Exp6714 updates
the long-lived specs and operations documents only from terminal artifacts.

## Explicit Non-Goals

- No ARC game or level solve claim.
- No generated-text or answer-logprob Phase D scorer retry.
- No GGUF weight update.
- No diffusion-model substitution for the required GGUF stack.
- No TSU, FPGA, latency, power, or device-availability claim from simulation.
- No learned verifier as release authority.
- No activation of `research-roadmap-next.yaml` in this planning task.


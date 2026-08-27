# Carnot Research Roadmap vNEXT — V582: Exact Transport and Prospective Adaptation

**Status:** Proposed  
**Planning date:** 2026-08-27  
**Milestone:** `2026.08.582`  
**Execution manifest:** `research-roadmap-next.yaml`  
**Task contract:** 14 tasks, `exp6674` through `exp6687`, in conductor order

## Executive decision

V581 reached a terminal conductor state, but it did not execute its documented
research plan. The design document declared 14 tasks. The active YAML contained
four tasks. Exp6660 failed three times because it used an unavailable Gemini
route. Exp6661 then built a valid fixture and passed every task-owned check, but
its readiness reducer included the known-red repository suite. Exp6662 blocked
on that false readiness field. Exp6663 was skipped.

V582 repairs the execution contract first. It then resumes the three research
branches that remain natural after V580 and the usable part of V581:

1. exact semantic transport from local SOTA reasoning to executable checks;
2. prospective continuous self-learning across independent constraint families;
3. outcome-bearing live control and exact stochastic portability contracts.

V582 does not rerun V581 unchanged. It changes the fixture gate to the approved
task-owned verification scope. It adds document-to-YAML task-set parity. It also
uses current Codex routing and removes Gemini from the plan.

## What V581 proved

| V581 scope | Terminal evidence | What is proved | V582 consequence |
|---|---|---|---|
| Design-to-manifest activation | The V581 document named 14 tasks. The active roadmap loaded four. | A complete design document does not ensure a complete conductor manifest. | Exp6674 compares exact task IDs, order, deliverables, gates, and run commands before activation. |
| Exp6660 evidence contract | Three dispatches failed because the Gemini model metadata was unavailable. No terminal artifact exists. | The selected backend was not executable on this host. No contract audit ran. | Use Codex for formulaic tasks. Record the stale `gpt-5.5` audit rule as a validator mismatch because the operator now requires `gpt-5.6-sol`. |
| Exp6661 triggered-tail fixture | 33 focused tests passed. Scoped coverage was 100%. Ruff, formatting, and spec coverage passed. The full repository suite failed with 3,923 cached nodes, none owned by Exp6661. | The fixture and its exact checkers are usable. The readiness reducer used the wrong test scope. | Exp6675 replays task-owned receipts and publishes a corrected readiness field. It does not rebuild the fixture or hide the global-suite state. |
| Exp6662 structured-tail A/B | The conductor wrote `blocked_gate_check_failed` because Exp6661 reported readiness false. | No model comparison ran. No output-transport claim exists. | Exp6676 runs only after Exp6675 owns a true readiness field. |
| Exp6663 independent audit | The conductor skipped the task after the upstream gate chain terminated. | No independent audit exists. | Exp6677 audits only a complete Exp6676 row set. |
| V580 scientific base | All three mandated GGUF families ran. Direct exact outputs parsed on 10 of 48 rows. Repair memory was safe but had a null independent interval. ARC redirects lacked exact post-action outcomes. The exact Ising reference passed algorithm rows but failed an overbroad readiness receipt. | The execution paths exist. Transport and independent outcome evidence remain the limiting gaps. | V582 repairs these boundaries instead of adding another model family or hardware target. |

## Three largest gaps to the PRD vision

| Rank | Gap | Current evidence | V582 closure attempt |
|---:|---|---|---|
| 1 | FR-12 has no trustworthy local-SOTA path from free reasoning to exact machine semantics. | The models run, but V580 parsed only 10 of 48 direct rows. V581 did not run the planned transport comparison. | Qualify the existing semantic-free triggered-tail fixture. Then compare natural extraction, immediate JSON, and delayed syntax tails on all three mandated GGUF families. |
| 2 | FR-11 continuous self-learning has no independent cross-family benefit. | Verified memory was safe and reversible on one fixture. Its order-level interval included zero. | Run a reset-free prequential stream across four independent families. Separate read-only reasoning from between-event memory writes. Use held-family exact validity, retention, retirement, restart, and rollback gates. |
| 3 | Objective feedback does not cross two live system boundaries. | ARC policy redirects lack exact next outcomes. Ising schedules lack a clean exact reference and typed portability receipt. | Carry exact post-action outcomes through the canonical live seam. Separately close the task-owned exact Ising receipt, Torx parity, and chain-level schedule evidence. |

## Exact execution contract

| Order | Task ID | YAML title | Deliverable |
|---:|---|---|---|
| 1 | `exp6674-v582-manifest-parity-contract` | V582 document-to-manifest parity contract | `results/experiment_6674_v582_manifest_parity_contract.json` |
| 2 | `exp6675-triggered-tail-scope-receipt` | Triggered-tail task-owned verification receipt | `results/experiment_6675_triggered_tail_scope_receipt.json` |
| 3 | `exp6676-three-family-triggered-tail-ab` | Three-family delayed syntax-tail A/B | `results/experiment_6676_three_family_triggered_tail_ab.json` |
| 4 | `exp6677-triggered-tail-independent-audit` | Delayed syntax-tail blinded row audit | `results/experiment_6677_triggered_tail_independent_audit.json` |
| 5 | `exp6678-constraint-family-stream` | Independent constraint-family prequential stream | `results/experiment_6678_constraint_family_stream.json` |
| 6 | `exp6679-prequential-cross-family-csl-ab` | Prequential cross-family continuous self-learning A/B | `results/experiment_6679_prequential_cross_family_csl_ab.json` |
| 7 | `exp6680-csl-durability-audit` | Cross-family CSL chronological durability audit | `results/experiment_6680_csl_durability_audit.json` |
| 8 | `exp6681-arc-post-redirect-outcomes` | Canonical ARC post-redirect outcome transport | `results/experiment_6681_arc_post_redirect_outcomes.json` |
| 9 | `exp6682-arc-held-family-supervisor-ab` | Held-family ARC supervisor outcome A/B | `results/experiment_6682_arc_held_family_supervisor_ab.json` |
| 10 | `exp6683-ising-reference-scope-receipt` | Bounded-treewidth Ising task-owned receipt | `results/experiment_6683_ising_reference_scope_receipt.json` |
| 11 | `exp6684-torx-typed-factor-parity` | Torx energy-distribution conformance | `results/experiment_6684_torx_typed_factor_parity.json` |
| 12 | `exp6685-autocorrelation-schedule-ab` | Autocorrelation-aware stochastic schedule A/B | `results/experiment_6685_autocorrelation_schedule_ab.json` |
| 13 | `exp6686-stochastic-portability-audit` | Cold exact and raw-chain review | `results/experiment_6686_stochastic_portability_audit.json` |
| 14 | `exp6687-v582-branch-synthesis` | V582 five-branch disposition | `results/experiment_6687_v582_branch_synthesis.json` |

## External research incorporated

The dated source record is in `research-references.md` under “V582 planner
refresh.” V582 uses these findings:

- **Generalization Gap in LLM Planning (arXiv:2601.14456):** high in-domain
  validity can coexist with zero validity on unseen domains. V582 makes
  held-family validity a primary metric.
- **Memoir (arXiv:2607.20792):** write-while-thinking memory slowed fixed-budget
  learning. V582 stages memory updates between events.
- **Self-Trained Verification (arXiv:2605.30290):** verifier scores can rise
  while accuracy stays flat. V582 keeps executable checkers authoritative.
- **AREX (arXiv:2607.21461):** verified evidence and unresolved constraints can
  form a compact improvement state. V582 stores typed repair obligations with
  exact receipt links.
- **Finite-automaton diffusion decoding (arXiv:2607.07026):** syntax constraints
  can be exact without carrying answer semantics. V582 applies that contract as
  an audit principle, not as a direct method claim for autoregressive GGUFs.
- **Pipelined p-computer (arXiv:2607.21077):** update order, coefficient width,
  and memory movement define later hardware fidelity. V582 preserves those
  receipts in the software reference.

OpenReview adds evidence that rule-based equivalence and learned-verifier
hacking need separate tests. Hugging Face work supports selective intervention
and harmful-flip accounting. GitHub supplies interface patterns but no required
dependency. Extropic still schedules Z1 access for 2027. Kona still has no
public weights or local runner. No attached FPGA has a changed-state receipt.

## Target architecture

```text
design task set ----+
YAML task set ------+--> parity receipt

V581 frozen fixture --> task-owned receipt --> three mandated GGUFs
                                              | natural
                                              | immediate JSON
                                              + delayed syntax tail
                                                        |
                                                exact executors
                                                        |
                                                blinded row audit

typed event stream --> frozen/context/memory arms --> exact admission
        ^                       |                         |
        |                       +---- future events <----+
        +---------- sealed initial state + rollback -----+

live E3 proposal --> canonical action seam --> applied action --> environment
                              ^                                  |
                              |                                  v
                       trace supervisor <--- exact next outcome receipt

Ising fixtures --> exact treewidth reference --> Torx typed factors
                                                |
                                        schedule A/B chains
                                                |
                                  likelihood + ACF + ESS audit
```

The exact checker, environment, or exact probability reference is release
authority. Learned energies, LLM scores, memories, and supervisors may propose,
rank, route, repair, or abstain. They may not certify themselves.

## Phase I — Execution integrity and fixture qualification

### Exp6674 — V582 manifest-parity contract

**Question:** Does the execution YAML contain the same 14-task contract as this
document, with valid gates, routes, deliverables, prior failures, and commands?

**Deliverable:** `results/experiment_6674_v582_manifest_parity_contract.json`

Compare ordered task IDs, titles, tracks, deliverables, dependencies, gates,
model routes, and run commands. Run schema, prior-failure, exclusion, and gate
checks. Treat the repository rule that still pins Codex to `gpt-5.5` as a dated
validator mismatch because this roadmap follows the operator's `gpt-5.6-sol`
instruction. Do not patch any validator in this task.

### Exp6675 — Triggered-tail verification-scope receipt

**Question:** Is the existing Exp6661 fixture ready under task-owned checks while
the unrelated repository-suite failure remains visible and non-gating?

**Deliverable:** `results/experiment_6675_triggered_tail_scope_receipt.json`

Replay the 33 task-owned tests, scoped coverage, lint, format, spec coverage,
exact controls, grammar leakage attacks, and row recomputation. Preserve the
full-suite failure as a diagnostic field. Set `triggered_tail_fixture_ready`
from owned checks only. If an Exp6661-owned node fails, block.

## Phase II — Exact local-SOTA output transport

### Exp6676 — Three-family triggered-tail A/B

**Question:** At matched task and token budgets, does a delayed syntax-only tail
improve exact semantic success over natural extraction and immediate JSON?

**Deliverable:** `results/experiment_6676_three_family_triggered_tail_ab.json`

Run all three mandated models:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

Use `cached_sota_pair()` and `resolve_cached_gguf()`. Use embedded GGUF
tokenizers. Emit every model-task-arm row. The primary metric is exact semantic
success on held-family tasks. Secondary metrics are parse yield, harmful flips,
trigger failures, proposal tokens, latency, and accelerator use. Immediate JSON
is a control. It does not reopen the retired ConstraintIR method.

### Exp6677 — Triggered-tail independent audit

**Question:** Do cold-process exact checks reproduce the A/B result, and does the
tail grammar remain free of task-answer semantics?

**Deliverable:** `results/experiment_6677_triggered_tail_independent_audit.json`

Blind task labels. Recompute all aggregates from raw rows. Attack answer
permutations, label renaming, trigger collisions, missing rows, duplicate rows,
unequal budgets, parser coercion, and grammar-only recovery. Report per-model
and held-family uncertainty. A syntax gain without exact gain is a transport
result only.

## Phase III — Continuous self-learning

### Exp6678 — Independent constraint-family stream

**Question:** Can one typed event contract cover four independent constraint
families without future leakage or family-specific answer keys?

**Deliverable:** `results/experiment_6678_constraint_family_stream.json`

Build scheduling, graph, arithmetic or logic, and plan-state families. Freeze
calibration, held-family, and at least five task-order manifests. Each event
contains visible state, exact violation witnesses, candidate repair operators,
support, anchors, provenance, version, and inverse patch. Future outcomes may
not enter retrieval keys.

### Exp6679 — Prequential cross-family CSL A/B

**Question:** Does exact-checker-admitted memory improve future exact outcomes
over frozen and context-only controls across orders and held families?

**Deliverable:** `results/experiment_6679_prequential_cross_family_csl_ab.json`

This is the required Continuous Self-Learning experiment. Run frozen,
context-only, and verified-memory arms on identical events. Use at least
`unsloth/Qwen3.6-35B-A3B-GGUF`; use the full three-model set when resources
permit, but never replace the headline model with a legacy smoke model. Event
`t` can read only state committed before `t`. The model reasons read-only. The
system admits patches between events after source repair, held-anchor retention,
support, restart, and inverse-patch checks.

### Exp6680 — CSL independent durability audit

**Question:** Does the CSL result survive chronological replay, order-level
uncertainty, poison attacks, restart, rollback, and family-held-out analysis?

**Deliverable:** `results/experiment_6680_csl_durability_audit.json`

Replay from a sealed initial state. Recompute future exact yield, forgetting,
regret, negative transfer, retirement, and support. Attack future-outcome
leakage, family labels in keys, duplicate events, poison patches, anchor
regression, and non-atomic state. An interval that includes zero is null.

## Phase IV — Live outcomes and stochastic portability

### Exp6681 — Canonical live ARC outcome transport

**Question:** Can the canonical live E3 action seam attach an exact next outcome
to every applied game-agnostic supervisor redirect?

**Deliverable:** `results/experiment_6681_arc_post_redirect_outcomes.json`

Instrument `make_carnot_agent` and `E3AgentPolicy` at the environment seam.
Record state hash, proposed action, applied action, reason, observations before
and after, reward, termination, family, attempt, and receipt lineage. Validate
on held-family live episodes. Archived replay is diagnostic only. This task
makes no game or level solve claim.

### Exp6682 — Held-family live supervisor A/B

**Question:** With exact outcomes present, does the existing game-agnostic trace
supervisor reduce forbidden actions or improve exact transitions without
blocking valid actions?

**Deliverable:** `results/experiment_6682_arc_held_family_supervisor_ab.json`

Freeze the supervisor before the held-family run. Compare supervisor-off and
supervisor-on at matched episode, seed, and action budgets. Emit every proposed
and applied action with exact next outcome. Primary metrics are forbidden-action
delta and exact transition utility. Report false interventions and actions.
Make no solve-rate, game-level, or level-level claim.

### Exp6683 — Bounded-treewidth exact-reference scope receipt

**Question:** Is the Exp6657 exact Ising reference ready under task-owned tests
and exact probability checks?

**Deliverable:** `results/experiment_6683_ising_reference_scope_receipt.json`

Replay decomposition, partition function, marginal, correlation, normalization,
sampling, unsupported-input rejection, scoped coverage, lint, and spec checks.
Record update order and coefficient precision. Keep the known global-suite state
diagnostic. Do not rebuild the reference unless an owned check fails.

### Exp6684 — Torx typed-factor parity

**Question:** Does the installed Torx CPU path reproduce the exact fixture's
energy and log-probability contract on supported graphs?

**Deliverable:** `results/experiment_6684_torx_typed_factor_parity.json`

Map binary variables, couplings, biases, temperatures, and update schedules to
typed Torx factors. Compare factor energy, total energy, normalized probability,
and rejected cases against Exp6683. Measure software only. Make no TSU, FPGA,
power, or asymptotic claim.

### Exp6685 — Autocorrelation-aware schedule A/B

**Question:** On exact supported fixtures, does the preregistered schedule
improve effective independent samples or likelihood accuracy at matched updates?

**Deliverable:** `results/experiment_6685_autocorrelation_schedule_ab.json`

Compare fixed-temperature Gibbs with the preregistered schedule. Freeze burn-in,
chain length, update count, seeds, and stopping rules. Emit every chain row with
energy, likelihood error, autocorrelation, integrated autocorrelation time,
effective sample size, and wall time. Do not use energy minimum as a sampling
quality proxy.

### Exp6686 — Stochastic portability independent audit

**Question:** Do cold exact recomputation and raw-chain analysis support the Torx
and schedule claims without hidden filtering or correlated-sample inflation?

**Deliverable:** `results/experiment_6686_stochastic_portability_audit.json`

Recompute exact probabilities and chain statistics. Attack seed reuse, chain
truncation, selective fixture omission, unsupported graphs, precision drift,
update-order drift, and ESS inflation. Separate algorithm quality from runtime.

### Exp6687 — V582 branch synthesis

**Question:** Which branches produced positive, null, blocked, disqualified, or
partial evidence, and what exact next action follows from each branch?

**Deliverable:** `results/experiment_6687_v582_branch_synthesis.json`

Read every terminal artifact and conductor state. Recompute headline metrics
from rows. Reconcile the V582 change proposal, roadmap status, traceability,
status, changelog, and research-complete records only to the evidence. Preserve
missing artifacts as missing. Make no pooled milestone success claim.

## Dependency graph

```text
6674                                      manifest parity, independent

6675 -> 6676 -> 6677                     exact output transport

6678 -> 6679 -> 6680                     continuous self-learning

6681 -> 6682                             live ARC outcomes

6683 -> 6684 -> 6685 -> 6686             exact stochastic portability

6674-6686 terminal states -> 6687         ungated branch synthesis
```

Structured gates use producer-owned fields:

| Consumer | Producer field | Condition |
|---|---|---|
| Exp6676 | `exp6675.triggered_tail_fixture_ready` | `in [true]` |
| Exp6677 | `exp6676.triggered_tail_ab_ready` | `in [true]` |
| Exp6679 | `exp6678.constraint_family_stream_ready` | `in [true]` |
| Exp6680 | `exp6679.prequential_csl_ab_ready` | `in [true]` |
| Exp6682 | `exp6681.arc_outcome_transport_ready` | `in [true]` |
| Exp6682 | `exp6681.eligible_redirect_outcome_rows` | `>= 30` |
| Exp6684 | `exp6683.ising_reference_ready` | `in [true]` |
| Exp6685 | `exp6683.ising_reference_ready` | `in [true]` |
| Exp6685 | `exp6684.torx_factor_parity_ready` | `in [true]` |
| Exp6686 | `exp6685.schedule_ab_ready` | `in [true]` |

Every producer prompt declares the field with identical spelling in its required
artifact fields. A blocked artifact uses `gate_check_summary` and records the
failed check, expected value, and observed value.

## Failed-scope and retirement boundaries

- Exp6675 addresses Exp6661 by changing verification scope, not the fixture.
  The full-suite state remains visible.
- Exp6676 addresses Exp6662 and Exp5923 through a qualified delayed syntax tail.
  It does not use finite answer IDs or a schema-supported ConstraintIR reprompt.
- Exp6679 addresses Exp6655 with independent families, held-family outcomes, and
  between-event writes. A repeated null retires this fixed-schema replication.
- Exp6681 addresses Exp6656 at the canonical live environment seam. Archived
  replay cannot support a benefit claim.
- Exp6683 addresses Exp6657 through owned checks. Exp6685 addresses Exp6658 only
  after exact-reference and Torx-parity readiness.
- No task references a retired upstream experiment in `requires` or `gated_on`.

## Hardware requirements

| Resource | Tasks | Requirement and claim boundary |
|---|---|---|
| Dual RTX 3090, 24 GB each | Exp6676, Exp6679 | Use owner-bound llama.cpp CUDA processes. Sequential model reuse is allowed. Record UUID, PID/start, model hash, load phases, VRAM, inference count, unload, and release. |
| Local GGUF cache | Exp6676, Exp6679 | Resolve mandated models through `cached_sota_pair()` and `resolve_cached_gguf()`. Use embedded tokenizers. Legacy small models can run CPU smoke tests only. |
| CPU and system RAM | Exp6674-6675, Exp6677-6678, Exp6680-6687 | Run exact checkers, memory replay, live ARC orchestration, exact Ising inference, Torx parity, chain analysis, and audits. Record the measured substrate. |
| Torx 0.0.1 | Exp6684-6686 | Use the installed API. If it is absent, block with the missing symbol. Do not install a replacement in the experiment. |
| Extropic TSU/Z1 | None | No authenticated runner is available. Make no latency, throughput, power, or availability claim. |
| KV260, GateMate, PolarFire | None | No changed-state receipt exists. Keep the boards opportunistic and outside V582. |
| AMD iGPU/ROCm | None required | Do not use the unstable path as a headline substrate. |

## Measurement and claim rules

- Every comparative task emits a per-unit row for every model, arm, event,
  order, family, episode, chain, seed, and condition.
- Every runtime LLM task includes a mandated GGUF in `MODEL_SPECS`. Exp6676 uses
  all three. Exp6679 uses at least Qwen3.6-35B-A3B.
- Exact executable checks authorize semantic validity and memory admission.
- Missing parses, outcomes, rows, chains, or artifacts remain missing.
- `verdict_class` is one of `positive`, `circular_positive`, `null`, `blocked`,
  `disqualified`, or `partial`.
- A terminal completed verdict starts `complete:`, `success:`, `passed:`, or
  `shipped:`. A blocked verdict starts `blocked_` and records
  `gate_check_summary`.
- ARC tasks make no solve claim and do not modify the solve registry.
- Hardware claims remain local to measured CUDA, CPU, or software paths.

## Milestone success criteria

V582 is operationally complete when all 14 tasks have terminal artifacts or
explicit conductor terminal states, and Exp6687 reconciles them. Scientific
success is branch-specific:

- output transport needs complete three-model rows, exact held-family outcomes,
  no semantic grammar leakage, and independent recomputation;
- continuous self-learning needs positive future exact yield with order-level
  uncertainty excluding zero, no anchor regression, restart, and rollback;
- ARC needs exact post-redirect outcomes before any supervisor utility claim;
- stochastic portability needs exact-reference readiness, Torx factor parity,
  and raw chain evidence for likelihood, autocorrelation, and ESS;
- process integrity needs exact document/YAML task-set parity, unchanged active
  `research-roadmap.yaml`, unchanged conductor, and no push.

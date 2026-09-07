# Research Roadmap vNEXT: Milestone 2026.09.624

**Milestone:** `2026.09.624`  
**Contract:** exactly 12 tasks, `exp7109` through `exp7120`, in the order
defined below and in `research-roadmap-next.yaml`  
**Theme:** Repair evidence ingress, recover the ARC generator path that did not
run in V623, and test continuous procedural learning on exact-verified outputs
from current local SOTA GGUF models.

## What V623 Proved

V623 closed its exact 12-task execution contract with one positive science
branch, one failed ARC preflight, and a newly measured evidence-consumer gap.

- Exp7097 proved that the V623 Markdown and YAML contracts agreed. Exp7098
  completed the execution-time source sweep. The stale-document failure from
  V620-V622 did not recur.
- Exp7099 passed its declared model, GPU, offline-environment, and registry
  preconditions, but its planned two-model/two-game generation cells did not
  execute. It finished in 41.4 seconds, set
  `adapter_withheld_live_path_ready_score=0`, and was stamped
  `flagged_adversarial: true` for duration and substrate-class contradictions.
  Exp7100-Exp7104 then gate-blocked. V623 produced no adapter-withheld ARC
  generalization or action-energy result.
- Exp7105 produced a sealed 144-event exact constraint stream. Exp7106 showed
  positive later value from delayed-commit abstract procedural memory against
  its matched controls while preserving protected retention. Exp7107 rebuilt
  and audited that result in a fresh process.
- The continuous-learning result used deterministic sealed events and did not
  exercise a local LLM. It establishes a sound transaction and evaluation
  substrate, not continuous improvement of a deployed model-facing system.
- Exp7108 reconciled all 12 task slots, but it ingested the flagged Exp7099
  artifact and marked 57 derived leaf checks as passed. Separately, 28% of new
  artifacts lacked a parseable `run_date`, and 19 existing E3 per-game rows
  lacked `solve_provenance`. A working gate is not enough when producers omit
  its key or consumers ignore its stamp.

V624 preserves the positive self-learning substrate, does not rerun the
blocked action-energy chain, and changes the failed ARC preflight into a
bounded liveness-and-receipt diagnosis before attempting the measurement.

## Three Largest Gaps to the PRD Vision

1. **Evidence cannot yet be trusted transitively.** The PRD requires auditable,
   verifiable reasoning, but new artifacts can evade dated checks, ARC rows can
   omit solve provenance, and capstones can aggregate artifacts already marked
   adversarial. Producer fields and consumer quarantine must agree.
2. **The live ARC generalization path still has no usable measurement.** V623
   showed that the prerequisites existed but not that model generation ran.
   Carnot needs nonzero request, token, action-channel, and duration receipts
   before it spends hours on an adapter-withheld leave-one-game-out matrix.
3. **Continuous self-learning has not crossed the model-facing boundary.** The
   delayed memory result is positive on an exact synthetic stream, but the PRD
   calls for a system that learns from actual interactions. The next test must
   use frozen SOTA GGUF outputs, exact post-decision feedback, multiple
   chronological iterations, future groups, paraphrases, and negative-transfer
   controls while model weights remain frozen.

## Research Findings That Shape V624

- Think-Verify-Revise (`arXiv:2609.05388`) couples grammar-constrained rule
  induction to verification feedback. V624 uses a strict constraint grammar,
  exact executable validation, one bounded revision, and delayed memory writes.
- ROBORMBENCH (`arXiv:2609.05401`) shows that a reward model can reverse its
  decision when only the instruction paraphrase changes. V624 freezes verified
  paraphrase pairs and reports decision flips per model, family, and memory arm.
- The experience-internalization study (`arXiv:2606.04703`) finds progressive
  multi-iteration collapse and favors principle-level experience, step-aligned
  injection, and off-policy successful trajectories. V624 directly compares
  principle/step memory with instance/global memory under equal token and
  capacity budgets. It tests external memory before any parametric update.
- The EBT and ARM-EBM citation trails still support energy and lookahead as
  useful ranking views, not exact authorities. V624 does not revive a learned
  action-energy claim before the live generator produces valid receipts.
- Extropic's Z1T update retains a degree-16 Z1 graph and FPGA/TSU partition, but
  Z1 early access is stated for 2027. V624 keeps physical hardware off the
  critical path and makes no device, power, or latency claim.
- Kona remains a proprietary globally scored constraint architecture without a
  public compatible checkpoint or runner. It remains a design comparator.

The source receipts and claim boundaries are recorded in
`research-references.md` under `V624 Planner Refresh - 2026-09-07`.

## V624 Architecture

```text
                         V624 evidence boundary
              +----------------------------------------+
              | Exp7109 exact task contract (advisory) |
              | Exp7112 current-source ingestion       |
              +----------------------------------------+

  evidence integrity                    ARC generalization recovery
  ==================                    ===========================

  Exp7110 shared ingress policy         Qwen3.8 pinned live model
  - reject flagged artifacts            + Qwen3.6 mandated SOTA control
  - require forward run_date                         |
  - record exclusions                                v
           |                             Exp7113 bounded generation liveness
           v                             request -> tokens -> action channel
  Exp7111 forward ARC provenance                      |
  - per-game solve_provenance                         v
  - dashboard consumption                Exp7114 adapter-withheld LOO matrix
  - no historical backfill               - two games x two models x two seeds
                                                - exact outcome replay
                                                         |
                                                         v
                                             Exp7115 independent cold audit

  verified continuous self-learning
  =================================

  Qwen3.6-35B-A3B + Gemma-4-26B-A4B
                     |
                     v
  Exp7116 frozen exact-labeled proposal/paraphrase bank
                     |
                     v
  Exp7117 think -> exact verify -> one bounded revise
                     |
                     v
  Exp7118 three-iteration matched memory comparison
  principle/step | instance/global | raw equal-context | no-memory | coupled-write
                     |
                     v
  Exp7119 fresh-process retention, flip, poison, crash, rollback audit

  Every available artifact --quarantine at ingress--> Exp7120 ungated capstone
```

Exact solvers and executable environments remain the authorities. Local models
propose rules or actions. Memory changes only a bounded external state after
the exact outcome is sealed. V624 does not change GGUF weights.

## Dependency Graph

```text
exp7109   exp7110   exp7111   exp7112       (independent infrastructure)

exp7113 --arc_generation_liveness_ready_score==1--> exp7114
exp7114 --adapter_withheld_loo_complete_score==1---+
exp7110 --evidence_ingress_quarantine_ready_score==1--> exp7115

exp7116 --sota_constraint_episode_bank_ready_score==1--> exp7117
exp7117 --verify_revise_loop_complete_score==1----------> exp7118
exp7118 --procedural_memory_comparison_complete_score==1-+
exp7110 --evidence_ingress_quarantine_ready_score==1-----> exp7119

exp7120                                      (ungated; reads every available slot)
```

Completion gates mean that all declared cells and evidence checks ran. They do
not require a positive scientific result. Every gate field below is a bare
top-level field named exactly in its producer's required artifact fields.
Stable external incompleteness is `blocked`, not retrying `partial`.

## Exact Task Contract

The following five-column table is normative. `research-roadmap-next.yaml`
must contain these 12 full IDs, titles, deliverables, and structured gates in
this exact order.

| Order | Full task ID | Title | Deliverable | Structured gate |
|---:|---|---|---|---|
| 1 | `exp7109-v624-contract-preflight` | V624 Markdown and YAML task-contract preflight | `results/experiment_7109_v624_contract_preflight.json` | none |
| 2 | `exp7110-evidence-ingress-quarantine` | Forward evidence ingress quarantine and date contract | `results/experiment_7110_v624_evidence_ingress_quarantine.json` | none |
| 3 | `exp7111-forward-arc-provenance-canary` | Forward ARC evaluation provenance and dashboard canary | `results/experiment_7111_v624_arc_provenance_canary.json` | none |
| 4 | `exp7112-v624-execution-sota-ingestion` | V624 execution-time SOTA ingestion and claim-boundary audit | `results/experiment_7112_v624_sota_ingestion.json` | none |
| 5 | `exp7113-arc-generation-liveness-recovery` | Bounded ARC local-generation liveness and receipt recovery | `results/experiment_7113_v624_arc_generation_liveness.json` | none |
| 6 | `exp7114-adapter-withheld-arc-loo-measurement` | Mandatory adapter-withheld ARC leave-one-game-out measurement | `results/experiment_7114_v624_adapter_withheld_loo.json` | `exp7113-arc-generation-liveness-recovery.arc_generation_liveness_ready_score == 1` |
| 7 | `exp7115-adapter-withheld-arc-cold-audit` | Independent adapter-withheld ARC provenance and leakage audit | `results/experiment_7115_v624_adapter_withheld_cold_audit.json` | `exp7110-evidence-ingress-quarantine.evidence_ingress_quarantine_ready_score == 1`; `exp7114-adapter-withheld-arc-loo-measurement.adapter_withheld_loo_complete_score == 1` |
| 8 | `exp7116-sota-constraint-episode-bank` | SOTA constraint proposal and verified paraphrase episode bank | `results/experiment_7116_v624_sota_constraint_episode_bank.json` | none |
| 9 | `exp7117-exact-verify-revise-loop` | Exact think-verify-revise constraint loop on held-out paraphrases | `results/experiment_7117_v624_exact_verify_revise_loop.json` | `exp7116-sota-constraint-episode-bank.sota_constraint_episode_bank_ready_score == 1` |
| 10 | `exp7118-principle-step-memory-csl` | Principle-level step-aligned continuous memory versus matched controls | `results/experiment_7118_v624_principle_step_memory_csl.json` | `exp7117-exact-verify-revise-loop.verify_revise_loop_complete_score == 1` |
| 11 | `exp7119-multi-iteration-memory-cold-audit` | Fresh-process multi-iteration memory retention and paraphrase audit | `results/experiment_7119_v624_multi_iteration_memory_cold_audit.json` | `exp7110-evidence-ingress-quarantine.evidence_ingress_quarantine_ready_score == 1`; `exp7118-principle-step-memory-csl.procedural_memory_comparison_complete_score == 1` |
| 12 | `exp7120-v624-capstone` | V624 independent evidence matrix and branch disposition | `results/experiment_7120_v624_capstone.json` | none |

## Phase 0: Evidence Contract and Current Sources

Phase 0 closes the three forward evidence gaps and repeats the standard source
check. These tasks do not gate the ARC liveness probe or episode-bank creation.

### Exp7109 - V624 Markdown and YAML task-contract preflight

**Purpose:** Prevent stale prose, truncated YAML, mismatched gates, and missing
producer fields from becoming the milestone's execution contract.

**Method:** Independently parse this table and the active roadmap. Compare all
12 IDs, titles, order, deliverables, gates, producer fields, prior failures,
model rules, `run_date`, verdict fields, prompt tails, and the ungated capstone.

**Acceptance:** `v624_task_contract_conforms_score=1` only when both independent
views agree. A mismatch is terminal `disqualified`. The task is advisory.

**Gate:** None.

**Prior failures:** Exp7076, Exp7084, and Exp7091 each found real contract
mismatches. V624 replaces both surfaces with one exact 12-row contract.

### Exp7110 - Forward evidence ingress quarantine and date contract

**Purpose:** Make an adversarial flag and a dated cutover effective at every
aggregation boundary.

**Method:** Build one reusable evidence-ingress helper. It hashes each artifact,
rejects `flagged_adversarial: true`, enforces a parseable `run_date` for V624
artifacts, and records every exclusion. Classify the 55 known capstone-to-flagged
references without rewriting history. Mutation tests must prove consumers
cannot silently aggregate a flagged or undated current artifact.

**Acceptance:** `evidence_ingress_quarantine_ready_score=1` requires fail-closed
classification, exact exclusion reasons, current-date coverage, and tests in at
least two independent consumer fixtures.

**Gate:** None.

### Exp7111 - Forward ARC evaluation provenance and dashboard canary

**Purpose:** Prove that future E3 per-game rows carry and preserve the field
that separates live self-discovery from development or outer-loop work.

**Method:** Exercise the real ARC row builder with deterministic stubbed policy
and environment boundaries. Require a legal `solve_provenance` on every new
per-game row and verify the dashboard groups it by that value. Do not label or
rewrite the 19 historical rows whose provenance is unknown.

**Acceptance:** `arc_forward_provenance_ready_score=1` requires 100% provenance
coverage on canary rows, consumer parity, and mutation failures for missing,
unknown, or consumer-dropped values.

**Gate:** None.

### Exp7112 - V624 execution-time SOTA ingestion and claim-boundary audit

**Purpose:** Catch decision-changing work published after this planning
snapshot without silently widening the fixed task contract.

**Method:** Repeat the requested source matrix, verify primary receipts, append
only novel material to `research-references.md`, and emit a valid empty-delta
artifact when nothing changes the plan.

**Acceptance:** `v624_sota_ingestion_complete_score=1` requires every requested
source class and explicit adoption, defer, duplicate, or inaccessible status.

**Gate:** None.

**Prior failure:** Exp6198 ended with a valid empty source delta. V624 searches
the broader dated matrix and treats a verified empty delta as completion.

## Phase 1: ARC Generation Recovery and Generalization

Phase 1 spends the mandatory ARC slot on the scored E3 path. It diagnoses the
unexecuted V623 cells before a bounded LOO matrix. It does not reopen action
energy or alter the ARC solve registry.

### Exp7113 - Bounded ARC local-generation liveness and receipt recovery

**Purpose:** Determine why V623 declared full generation without executing its
planned cells, and establish a small trustworthy entrance receipt.

**Method:** Run one bounded isolated generator request per model on the same
frozen ordinary observation. Capture lease, process, argv, stdout/stderr,
request, completion, token, parsed-channel, action, timeout, cleanup, and wall
time rows. Use `model_bounded_generation`, not a full-generation declaration.

**Acceptance:** `arc_generation_liveness_ready_score=1` requires both the pinned
Qwen3.8 live model and mandated Qwen3.6 control to load, receive a request,
produce nonzero tokens, expose a nonempty E3 generator channel, and exit cleanly.

**Gate:** None.

**Prior failure:** Exp7099 returned
`complete_null_adapter_withheld_live_path_not_ready_no_solve_claim`. Exp7113
changes the technique to bounded request-level instrumentation and an accurate
substrate class before any game measurement.

### Exp7114 - Mandatory adapter-withheld ARC leave-one-game-out measurement

**Purpose:** Produce the generalization measurement V623 never ran.

**Method:** Before outcomes, freeze two games, two models, two seeds, budgets,
and stop rules. Run only through the scored E3 live path without adapters, game
source, registry trajectories, or per-game recipes. Bank each cell immediately.
Join the registry only after live rows are sealed and replay every counted
transition or level in an exact environment. Timeouts and zeros stay in scope.

**Acceptance:** `adapter_withheld_loo_complete_score=1` means all eight cells and
their receipts completed, including honest all-zero results. Value is reported
separately in `adapter_withheld_any_level_score`.

**Gate:** `exp7113-arc-generation-liveness-recovery.arc_generation_liveness_ready_score == 1`

**Prior failure:** Exp7100 was `blocked_gate_check_failed` because Exp7099 wrote
readiness zero. Exp7114 gates on a new request-level liveness producer and
requires forward provenance rows.

### Exp7115 - Independent adapter-withheld ARC provenance and leakage audit

**Purpose:** Decide whether Exp7114 is a valid development measurement and
whether any result crossed the forbidden information boundary.

**Method:** In a fresh process, quarantine flagged inputs, reconstruct the
matrix, replay outcomes, compare raw hashes, scan accesses, and attack adapter,
source, registry-recipe, outcome, and provenance boundaries. Recompute every
headline from per-game rows.

**Acceptance:** `adapter_withheld_audit_ready_score=1` requires matrix parity,
no forbidden access, exact reproduction, model and budget parity, legal
`solve_provenance`, and `arc_registry_delta=0`. It can validate a null result.

**Gates:** `exp7110-evidence-ingress-quarantine.evidence_ingress_quarantine_ready_score == 1` and
`exp7114-adapter-withheld-arc-loo-measurement.adapter_withheld_loo_complete_score == 1`

## Phase 2: Verified SOTA Continuous Self-Learning

Phase 2 moves the V623 transaction mechanism onto model-generated constraint
proposals while exact checks retain authority. It is the milestone's explicit
Continuous Self-Learning track.

### Exp7116 - SOTA constraint proposal and verified paraphrase episode bank

**Purpose:** Create immutable real-model episodes suitable for a prospective,
multi-iteration learning test.

**Method:** Freeze 36 cases across 12 constraint groups. For each case and each
of Qwen3.6-35B-A3B and Gemma-4-26B-A4B, run canonical and independently
verified paraphrase prompts under matched settings. Parse only a strict grammar,
label proposals with two exact validators, and seal chronological train,
adaptation, future-group, and retention partitions before learning.

**Acceptance:** `sota_constraint_episode_bank_ready_score=1` requires all 144
planned model-prompt cells, exact-validator agreement, immutable hashes, no
outcome-based replacement, and sufficient rows in every group and partition.

**Gate:** None.

### Exp7117 - Exact think-verify-revise constraint loop on held-out paraphrases

**Purpose:** Test whether exact counterexample feedback repairs model proposals
and whether the effect survives a surface-only paraphrase.

**Method:** Compare one-shot generation with one bounded revision using the
same model, token budget, case, and prompt surface. The verifier returns only a
minimal violated constraint or counterexample after the first decision; it
never supplies a valid answer. Report canonical/paraphrase flips and every
repair or regression per unit.

**Acceptance:** `verify_revise_loop_complete_score=1` requires every matched
cell and information-isolation check. `verify_revise_value_ready_score=1`
requires a positive paired exact-validity delta for both model families without
increased paraphrase flips or protected-retention regression. Null is terminal.

**Gate:** `exp7116-sota-constraint-episode-bank.sota_constraint_episode_bank_ready_score == 1`

### Exp7118 - Principle-level step-aligned continuous memory versus matched controls

**Purpose:** Test the V623 procedural-memory result on actual SOTA-model
interactions and measure whether it compounds or collapses over time.

**Method:** Run three chronological iterations with five frozen arms:
principle-level step-aligned delayed memory, instance-level global memory, raw
equal-context replay, no memory, and write-while-deciding. Match token, capacity,
model, case, revision, and retrieval budgets. Decide before exact feedback and
commit only after it. Evaluate future groups, protected old groups, hard cases,
and paraphrase pairs for both models.

**Acceptance:** `procedural_memory_comparison_complete_score=1` requires every
declared cell, transaction, and per-episode row.
`procedural_memory_value_ready_score=1` requires the principle/step arm to beat
both no-memory and instance/global controls on later and future windows for
both models, without retention, flip-rate, hard-group, or poison regressions.

**Gate:** `exp7117-exact-verify-revise-loop.verify_revise_loop_complete_score == 1`

**Prior failures:** Exp6978 was a transactional-memory null and Exp7070 blocked
on an eight-event stream. V624 uses a 144-cell frozen SOTA bank, three temporal
iterations, exact delayed feedback, paraphrase pairs, and matched memory arms.

### Exp7119 - Fresh-process multi-iteration memory retention and paraphrase audit

**Purpose:** Verify that the self-learning result is reconstructable and that
the selected memory design does not trade short-term gain for forgetting,
surface fragility, or unsafe state mutation.

**Method:** Quarantine flagged inputs, rebuild all five arms from empty state,
replay event and transaction order, recompute metrics, and attack future-label
leakage, paraphrase mapping, poison, stale parents, partial writes, crashes,
capacity, eviction, rollback, and model-family aggregation.

**Acceptance:** `continual_memory_cold_audit_ready_score=1` requires exact
reconstruction, atomic delayed commits, deterministic rollback, producer-auditor
metric parity, and correct terminal classification of positive or null value.

**Gates:** `exp7110-evidence-ingress-quarantine.evidence_ingress_quarantine_ready_score == 1` and
`exp7118-principle-step-memory-csl.procedural_memory_comparison_complete_score == 1`

**Prior failures:** Exp6979 was a cold-audit null and Exp7071 was gate-blocked.
V624 audits a current completed producer with immutable SOTA episode,
transaction, paraphrase, and memory-hash rows.

## Phase 3: Independent Synthesis

### Exp7120 - V624 independent evidence matrix and branch disposition

**Purpose:** Recompute every V624 claim and state what advances, remains null,
is blocked, is disqualified, or must retire.

**Method:** Discover the exact 12 slots, pass every input through the shared
quarantine policy, record excluded flagged or invalid-date artifacts, validate
hashes and principles, replay gates, recompute headlines from per-unit rows,
and reconcile ARC provenance and self-learning comparisons. Never treat missing
external evidence as retryable `partial` work.

**Acceptance:** `v624_evidence_matrix_complete_score=1` requires a disposition
for all 12 slots, visible exclusions, and no row/headline contradiction.
Positive science is not required.

**Gate:** None. The capstone is intentionally ungated.

**Prior failures:** Exp6952 retried a capstone over stable external gaps.
Exp7108 completed but consumed a flagged input. Exp7120 is ungated, uses
terminal blocked dispositions, and records every quarantined upstream.

## Hardware Requirements and Claim Boundary

| Resource | V624 use | Requirement | Claim boundary |
|---|---|---|---|
| Dual RTX 3090, 24 GB each | Qwen3.8/Qwen3.6 ARC work; Qwen3.6/Gemma-4-26B constraint work | Available; approved cache, isolated lease, one owner per GPU | Report exact hub ID, quantization, path hash, GPU assignment, peak memory, and wall time. No cloud substitution. |
| Host CPU and RAM | Exact solvers, ARC replay, memory transactions, audits, aggregation | Available | Exact host checks are authorities for declared fixtures, not accelerator benchmarks. |
| NVMe model cache | Three referenced GGUF repositories | Required before each generating task; no download during an experiment | Missing required models block. Legacy-small models may only smoke-test transport and never supply headline rows. |
| AMD KV260 | None | Terminal prior receipt | No rerun or V624 claim. |
| Microchip PolarFire SoC | None | Terminal prior receipt | No rerun or V624 claim. |
| GateMate | None | Physically blocked | Not a dependency. |
| Extropic Z1/TSU | None | Not attached; vendor says early access 2027 | No device, speed, power, or execution claim. The bounded procedural memory has a plausible sparse counter/lookup path, but V624 does not validate it in hardware. |

## Milestone-Wide Scientific Rules

- Every task requires a parseable `run_date`, a principle annotation for every
  required field, one of the six current `inference_substrate_class` values,
  `execution_venue`, the closed `verdict_class`, and a terminal-prefix
  `honest_verdict`.
- Every comparative task emits a per-unit list. Pooled metrics without the rows
  that determine them are invalid.
- Every blocked artifact uses the exact field `gate_check_summary` and names
  the failed check, expected value, and observed value.
- Every capstone or evidence consumer excludes `flagged_adversarial: true`
  inputs and records them in `excluded_flagged_upstreams`.
- ARC game-level rows declare `solve_provenance`. V624's public-game LOO
  measurement is `development_proxy`, requires exact replay for every counted
  level, keeps `arc_registry_delta=0`, and is never promoted as a new solve.
- No ARC task reads target-game source, adapters, registry trajectories,
  per-game recipes, or exhaustive outer-loop ground truth.
- Every LLM-generating task executes at least one mandated SOTA GGUF headline
  cell. V624 uses `unsloth/Qwen3.6-35B-A3B-GGUF` in every such task and
  `unsloth/gemma-4-26B-A4B-it-GGUF` in the self-learning branch. Qwen3.8 is an
  additional pinned ARC live-model arm.
- Exact validation happens after each decision. Memory writes happen after
  exact feedback. Model weights remain frozen.
- Null results are terminal science. `partial` is reserved for the task's own
  recoverable incomplete work, never an unchanged upstream absence or block.
- V624 does not modify `scripts/research_conductor.py`, does not activate
  `research-roadmap-next.yaml`, and does not push.

## Milestone Success Criteria

V624 succeeds when it produces all possible terminal evidence under the fixed
gates, including honest nulls. The minimum informative close is:

1. a forward evidence-ingress policy, dated current artifacts, and a provenance
   canary that proves no historical provenance was invented;
2. bounded nonzero ARC generation receipts followed, when ready, by an honest
   adapter-withheld eight-cell measurement and independent audit;
3. a 144-cell SOTA proposal/paraphrase bank, exact verify-revise comparison,
   three-iteration five-arm continuous-memory test, and cold safety audit; and
4. an ungated 12-slot capstone that visibly quarantines flagged artifacts and
   never inflates blocked, circular, or development-proxy evidence.

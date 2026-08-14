# Research Roadmap vNEXT — Evidence-Complete Energy and Safe Online Memory

**Milestone:** 2026.08.553  
**Planning date:** 2026-08-14  
**Status:** Proposed  
**Predecessor:** 2026.08.552  
**Execution queue:** `research-roadmap-next.yaml`  
**Task count:** 12 experiments in four phases

## 1. Decision

Milestone 2026.08.553 will repair the evidence boundary that blocked V552. It
will then test two new research ideas on clean data:

1. verification cost under compositional constraint saturation; and
2. capacity-limited continuous memory under controlled interference.

The milestone keeps exact validators as the only release authority. Learned
energy, model output, memory, and routing may propose, rank, abstain, or direct
work. They may not accept an invalid result.

One ARC task preserves the standing generalization floor. It tests a generic
state-key reachability repair on the canonical live path. It makes no game or
level solve claim and does not modify the solve registry.

No board experiment is scheduled. The local FPGA and TSU state did not change.
V553 uses dual RTX 3090 CUDA execution for authenticated GGUF work and records
task-linked resource receipts that can support later hardware comparisons.

## 2. What V552 proved

V552 produced real progress, but its strongest public claims did not survive
the full evidence chain.

| Evidence | What is proved | Boundary carried into V553 |
|---|---|---|
| Exp6412 | The V551 powered claims were not supported by live execution receipts. The additive correction trail works. | Historical artifacts stay immutable. New claims must start from fresh execution. |
| Exp6413 | All three mandated GGUF families produced authenticated local CUDA receipts. | The reusable process, model-byte, tokenizer, raw-output, and GPU receipt layer survives. |
| Exp6414 | A fresh three-family factor corpus was created and exact-checker bound. | The artifact is adversarial-flagged because `duration_s=48.151455055` is below the declared 60-second live-model floor. It has no claim-ready standing. |
| Exp6415 | Exact Boolean WCSP CCG kernelization reduced work while preserving optima on its frozen cases. | This is a clean deterministic control. It does not prove learned energy value. |
| Exp6416 | Exact-triggered selective refinement matched always-refine accuracy with less work and no false accepts. | Routing is useful when exact triggers retain authority. |
| Exp6417 | The write-time admission replay reported positive future yield and no retention harm. | The artifact is adversarial-flagged because its measured duration is implausibly small. The public factor claim remains false. |
| Exp6418–6419 | Dual-path memory and held restart runs recorded valid exact bindings, retention, rollback, and restart structures. | Their reported utility deltas do not recompute from event rows. Development and held recomputed deltas are both zero. |
| Exp6420 | The independent audit found eight metric mismatches, 24 reused development outputs, cache resurrection, and four underpowered cells. | Prospective CSL eligibility is false. A fresh write-once stream is required. |
| Exp6421–6422 | A default-off ARC route changed legal executed policy and passed a held-family safety audit. | This is internal policy influence only. No solve, registry, or public ARC claim exists. |
| Exp6423 | The capstone preserved the authentic receipt layer, blocked flagged factor claims, nullified the CSL public claim, and preserved the narrow ARC policy result. | V553 must not pool positive upstream summaries across flagged or non-recomputable evidence. |

## 3. The three largest PRD gaps

### Gap 1 — Scientific provenance does not yet support a public energy claim

The PRD requires trustworthy verification and reproducible local execution.
V552 has authentic model receipts, exact CCG controls, and safe selective
refinement. It does not have one clean chain from fresh model bytes to an
untouched future outcome. Exp6414 and Exp6417 remain flagged. Comparative
artifacts also lack a common `per_unit_rows` surface.

V553 response:

- add a reusable task-scoped runtime receipt contract;
- create a new constraint-count-stratified factor corpus;
- recompute every aggregate from immutable event rows; and
- rerun the write-time admission question with honest timing and current
  adversarial checks.

### Gap 2 — FR11 continuous self-learning has structure but no verified utility

The PRD calls for online improvement with immutable validation, rollback, and
bounded forgetting. V552 recorded the right transaction shapes, but the audit
found zero recomputed utility delta, raw-output reuse, cache resurrection, and
underpowered cells. This is not a verified self-learning result.

V553 response:

- generate a fresh write-once event stream;
- measure a memory capacity-versus-utility frontier;
- attack authority conflicts, supersession, and retrieval collisions;
- replicate on a sealed held shift after real process restarts; and
- run an independent audit that derives every headline from rows.

### Gap 3 — Live autonomy and deployment cost remain below the PRD vision

The live ARC route can affect policy, but it has not improved a public-score-
relevant outcome. The adapter-bypassed public baseline clears at least one
level in 8 of 25 games. A generic state key collapses `sc25` after 24 actions,
so the current live search cannot even expose that game to later policy work.
Hardware acceleration also remains proof-of-concept only.

V553 response:

- diagnose and repair the generic state-key reachability invariant;
- test it on the full adapter-bypassed public roster with per-game regression
  gates; and
- record task-linked verification cost and GPU use for all powered work.

The ARC task does not claim or register a solve. The hardware part is evidence
infrastructure only. It makes no speed, power, or board-availability claim.

## 4. Research refresh and experiment hooks

The dated source ledger is in `research-references.md` under the V553 planner
marker. Four findings change the experiment design.

| Source | Carnot use in V553 | Boundary |
|---|---|---|
| Constraint Saturation Evaluation, arXiv:2608.12426 | Stratify factor rows by constraint count and interaction class. Report per-constraint and joint exact success. | No LLM judge. No claim outside the frozen families. |
| Verification Cost, arXiv:2608.08709 | Measure exact correctness and cost-to-verdict under fixed budgets as separate outcomes. | Cost does not weaken the exact release veto. |
| Formal Agent Memory, arXiv:2608.11654 | Measure coverage, precision, retention, and query value across fixed memory capacities. | Memory is not an oracle. |
| Controlled Memory Interference, arXiv:2608.07622 | Add authority conflict, supersession, temporal-validity, and retrieval-collision attacks. | A failed attack blocks promotion. |

Recent vector-spin hardware, trie decoding, and neural drift bounds remain
controls. Carnot has no local vector-spin device. The generated-answer grammar
lane remains retired. Neural drift claims wait until row-level memory effects
are real.

Semantic Scholar still exposes 33 EBT citations and eight ARM-EBM citations.
OpenReview and Hugging Face Papers add no executable exact-authority baseline.
GitHub Trending adds no required runtime dependency. Extropic Z1 remains a
future access target. Kona still has no public weights or documented local API.

## 5. Target architecture

```text
                   task-scoped resource receipt contract
                   phase clocks | PID | GPU | runner | hashes
                                      |
                                      v
  mandated local GGUFs ---> immutable per-unit event rows
       Qwen MoE             raw bytes | prompt | model | source | cost
       Gemma dense                       |
       Gemma MoE                         v
                              deterministic exact checks
                                      |
                    +-----------------+------------------+
                    |                                    |
                    v                                    v
        constraint saturation + cost          exact write-time admission
          never / always / selective           commit / reject / quarantine
                    |                                    |
                    +-----------------+------------------+
                                      |
                                      v
                         capacity-limited online memory
                    proposal coverage | selection utility
                                      |
                    +-----------------+------------------+
                    |                                    |
                    v                                    v
          authority-interference attacks       held shift + process restart
                    |                                    |
                    +-----------------+------------------+
                                      |
                                      v
                         independent row recomputation
                                      |
                           exact release veto remains final

  canonical live ARC path
        observation ---> collision certificate ---> generic state-key suffix
             |                                         |
             +---------- full-roster matched A/B <-----+
                  default off | no solve credit
```

Three rules hold across the diagram:

1. Every comparative claim has immutable `per_unit_rows`.
2. Every aggregate is a deterministic function of those rows.
3. A model, score, router, or memory cannot override an exact rejection.

## 6. Phase design

### Phase 0 — Evidence and execution controls

#### Exp6424 — V552 terminal handoff and V553 queue preflight

Question: Is the V553 queue complete, schema-valid, non-retired, and anchored to
the actual V552 terminal evidence?

The task records all V552 artifacts, conductor outcomes, current adversarial
findings, and scientific eligibility. It validates all 12 V553 tasks before
research starts. It also checks prompts, gates, model policy, prior failures,
deliverables, and protected files.

Deliverable:
`results/experiment_6424_v553_terminal_handoff_and_queue_preflight.json`

#### Exp6425 — Recurring gate-block root cause and diagnostic contract

Question: Why did `blocked_gate_check_failed` stop 31 tasks across recent
milestones without identifying the failed gate?

This is the mandatory-next issue from `ops/known-issues.md`. The task traces the
31 cases to their producer fields and structured gates. It separates correct
scientific refusals from missing data, wrong field names, wrong types, retired
dependencies, and stale artifact reads. It may repair shared producer or
artifact diagnostics outside the conductor. It must not rerun the blocked
experiments.

Deliverable:
`results/experiment_6425_recurring_gate_block_root_cause.json`

#### Exp6426 — Task-scoped runtime and resource receipt contract

Question: Can each powered experiment prove which task used which model,
runner, process, GPU, and wall-clock interval?

The contract captures monotonic phase clocks, subprocess IDs, command and
config hashes, model-byte hashes, raw-output hashes, PID-linked GPU samples,
runner selection, concurrency, exit status, and attribution failures. It must
work for successful, blocked, interrupted, and CPU-only tasks. It does not
modify the conductor.

Deliverable:
`results/experiment_6426_task_scoped_runtime_receipt_contract.json`

### Phase 1 — Clean factor evidence and verification cost

#### Exp6427 — Fresh constraint-saturation factor corpus

Question: Can the three mandated local model families produce a clean,
row-recomputable factor corpus with no current adversarial flag?

This task is a changed rerun of Exp6414. It uses Exp6426 receipts, new prompts,
new raw bytes, and monotonic task timing. The corpus is balanced across three
factor families, three model families, constraint counts, and interaction
classes. Exact checkers score both each constraint and the joint result.

Promotion gates:

- all three mandated models execute through `cached_sota_pair()`;
- every row has a unique raw-output hash and task receipt;
- the corpus and future partition are sealed before analysis;
- all headline metrics recompute from `per_unit_rows`; and
- current adversarial flag count is zero.

If the same duration-flagged verdict recurs, retire this exact rerun scope.

Deliverable:
`results/experiment_6427_fresh_constraint_saturation_factor_corpus.json`

#### Exp6428 — Clean exact write-time factor admission A/B

Question: Does exact write-time admission improve untouched future exact yield
over a frozen arm without increasing contamination or reducing retention?

This is a changed rerun of Exp6417. It consumes only the clean Exp6427 rows and
records real nonzero execution time. Frozen, write-everything, and exact-
admission arms receive matched evidence and work. The future partition is
evaluated once.

Promotion gates:

- `delta_future_exact_yield > 0`;
- contamination does not increase over frozen and stays below write-everything;
- protected retention does not regress;
- every aggregate matches row recomputation; and
- the current adversarial flag count is zero.

If the same flagged positive verdict recurs, retire this exact rerun scope.

Deliverable:
`results/experiment_6428_clean_write_time_factor_admission_ab.json`

#### Exp6429 — Constraint saturation and verification-cost A/B

Question: Under a fixed verification budget, where does exact joint success
collapse, and can exact-triggered selective verification reduce cost without
adding false accepts?

The task compares never-refine, always-refine, and exact-triggered selective
arms on the sealed Exp6427 rows. It reports results by constraint count,
interaction class, model family, and verifier budget. It separates correctness,
abstention, time-to-verdict, checker calls, and verification-cost errors.

This task does not require a positive Exp6428 result. It asks a distinct
measurement question about the clean corpus.

Deliverable:
`results/experiment_6429_constraint_saturation_verification_cost_ab.json`

### Phase 2 — Continuous self-learning with independent recomputation

#### Exp6430 — Prospective write-once memory capacity frontier

Question: Does exact-governed continuous memory improve future exact outcomes,
and how does value change with memory capacity?

This is the milestone's required continuous self-learning task. It is gated on
a clean positive Exp6428 result. It generates a new chronological stream with
the three mandated GGUF families. Every event gets a fresh raw output. The task
proves the manifest path did not exist before generation. It compares frozen
memory with fixed capacities and reports coverage, precision, exact future
yield, retention, forgetting, growth, restart recovery, and cost.

The task addresses Exp6420 by construction:

- no raw output may serve more than one event;
- all rows are written before aggregate calculation;
- no cached held manifest may be resurrected;
- effects are reported with counts and uncertainty; and
- exact validators remain the release authority.

Deliverable:
`results/experiment_6430_prospective_write_once_memory_capacity_frontier.json`

#### Exp6431 — Controlled memory-interference A/B

Question: Can the frozen Exp6430 memory policy resist authority conflict,
supersession, temporal invalidity, retrieval collision, and poisoning?

The task applies a pre-registered interference matrix to the sealed stream. It
compares capacity-matched memory with and without authority-aware retrieval and
write controls. It measures target exposure, downstream use, plasticity,
protected stability, contamination, rollback, and exact future yield.

An attack that reaches release authority or survives rollback blocks promotion.

Deliverable:
`results/experiment_6431_controlled_memory_interference_ab.json`

#### Exp6432 — Held-shift process-restart replication

Question: Does the frozen memory policy retain any positive effect on a new
distribution after real process restarts?

The task uses new prompts and new raw outputs. It does not replay development
bytes. It seals the held manifest before model execution, restarts the process
between sessions, and verifies memory heads from disk. It reports row-derived
effects and negative transfer by model and family.

Deliverable:
`results/experiment_6432_held_shift_process_restart_csl_replication.json`

#### Exp6433 — Independent CSL row-recomputation and safety audit

Question: Do Exp6430 through Exp6432 support a prospective CSL claim when an
independent implementation recomputes every metric and replays every attack?

This audit is deliberately ungated. Missing, skipped, null, flagged, or
underpowered upstream results must remain visible. The audit cannot import the
upstream aggregate functions. It reads immutable rows and derives counts,
rates, deltas, uncertainty, retention, forgetting, contamination, and cost.

If it repeats Exp6420's null verdict, the exact repeated CSL claim scope is
retired. Transaction mechanics may remain valid as non-utility infrastructure.

Deliverable:
`results/experiment_6433_csl_row_recomputation_safety_audit.json`

### Phase 3 — Live-path reachability and adversarial close

#### Exp6434 — ARC state-key reachability invariant A/B

Question: Does a generic collision-certified state key prevent premature
frontier collapse without causing per-game regressions?

The task starts from the current 25-game adapter-bypassed public baseline. It
detects observation-history collisions generically and adds the smallest state
suffix only after a collision certificate. It runs matched baseline and arm
cells across the full roster and multiple seeds.

The success surface is search reachability, regression safety, and action cost.
The task makes no game or level solve claim. It does not update
`ops/arc_solve_registry.yaml`. Public games are a development proxy, not hidden
game evidence.

Deliverable:
`results/experiment_6434_arc_state_key_reachability_ab.json`

#### Exp6435 — V553 adversarial capstone and reconciliation

Question: Which V553 claims remain eligible after current adversarial checks,
row recomputation, dependency review, and determination preservation?

The capstone audits all 12 tasks. It cannot average away missing, skipped,
flagged, null, or underpowered cells. It records separate eligibility for:

- the factor and verification-cost result;
- prospective continuous self-learning;
- the narrow ARC reachability result; and
- hardware or deployment claims.

It reconciles OpenSpec, traceability, status, changelog, known issues, the
exclusion manifest, and the claim ledger. Hardware eligibility remains false
unless an authenticated hardware artifact exists, which V553 does not plan.

Deliverable:
`results/experiment_6435_v553_adversarial_capstone.json`

## 7. Dependency graph

```text
Exp6424  terminal handoff and queue preflight
   |
   +------------------------+------------------------+
   |                        |                        |
   v                        v                        v
Exp6425                  Exp6426                 later capstone
gate-block audit         runtime receipts
                            |
                            v
                         Exp6427
                      clean factor corpus
                       /             \
                      v               v
                   Exp6428          Exp6429
                clean admission   saturation + cost
                      |
                      v
                   Exp6430
               memory capacity frontier
                    /   \
                   v     v
                Exp6431  |
              interference|
                   \      |
                    v     v
                     Exp6432
                 held restart replication
                         |
                         v
                     Exp6433
                  independent audit

Exp6434  ARC reachability A/B  ----------------------+
Exp6425  gate-block result     ----------------------+--> Exp6435
Exp6429  verification-cost result -------------------+
Exp6433  CSL audit ----------------------------------+
```

Structured conductor gates exist only where a failed prerequisite makes the
downstream experiment meaningless. Exp6433 and Exp6435 remain ungated so they
can report missing or blocked evidence.

## 8. Models and inference policy

Experiments that execute an LLM are Exp6427, Exp6430, and Exp6432. Their
`MODEL_SPECS` must include:

- `unsloth/Qwen3.6-35B-A3B-GGUF`;
- `unsloth/gemma-4-31B-it-GGUF`; and
- `unsloth/gemma-4-26B-A4B-it-GGUF`.

Each task must use `cached_sota_pair()` and the embedded GGUF tokenizer. It
must not call `AutoTokenizer`. It must bind model bytes, prompt, runner,
process, raw output, GPU samples, and exit status to each event row.

`Qwen3.5-0.8B` and `gemma-4-E4B-it` may appear only in CPU smoke tests. They
cannot support a headline cell.

## 9. Hardware requirements

| Resource | Tasks | Requirement and boundary |
|---|---|---|
| Dual RTX 3090, 48 GB total VRAM | Exp6427, Exp6430, Exp6432 | Run mandated GGUF families with CUDA offload. Record task-linked PIDs, GPU samples, concurrency, runner selection, and model-byte hashes. A CPU fallback blocks headline eligibility. |
| CPU and system RAM | All tasks | Exact checkers, CCG, row recomputation, ARC search, and audit work. Record CPU-only substrate honestly. |
| Local persistent disk | Exp6427–Exp6433 | Store immutable raw outputs, per-unit JSONL rows, manifests, memory heads, restart receipts, and hashes. Pre-existing held paths block freshness. |
| ARC live environment | Exp6434 | Use the canonical adapter-bypassed public benchmark and exact game interface. No source reads, exhaustive ground-truth search, or per-game adapter. |
| KV260, GateMate, PolarFire | None | No state change justifies another probe. Existing POC boundaries remain. |
| Extropic XTR-0 or Z1 | None | No authenticated local route exists. No execution, power, latency, or availability claim. |

The milestone does not claim hardware acceleration. Exp6426 creates the cost
and attribution evidence needed for a later matched hardware comparison.

## 10. Measurement contract

Every comparative artifact must include `per_unit_rows`. Each row must carry a
stable unit ID, arm, partition, model or substrate, source hashes, outcome,
work, timing, and exclusion reason. Aggregate fields must identify their row
filter and deterministic reduction.

Every task must include:

- `blocked_reason`, even when null;
- `inference_substrate`;
- `field_principles` for every required field and gate;
- `field_provenance` classified as measured, derived, constant, or upstream;
- `random_seed`, `duration_s`, `tests_run`, and `reproducibility_checksum`;
- current adversarial and determination-preservation results when a scientific
  claim is made.

The main claim ladder is:

```text
authenticated execution
  -> immutable per-unit evidence
  -> exact row recomputation
  -> positive held effect with uncertainty
  -> adversarial attacks fail closed
  -> narrow claim eligibility
```

Failure at one rung blocks every higher rung. It does not erase valid lower-
rung engineering evidence.

## 11. Success and retirement rules

V553 is scientifically successful if it closes questions honestly. A clean
null is acceptable. A fabricated or non-recomputable positive is not.

Public factor eligibility requires:

- clean Exp6427 execution and rows;
- positive and clean Exp6428 held future yield;
- zero false-accept increase and no retention regression; and
- an eligible Exp6435 determination.

Prospective CSL eligibility requires:

- fresh write-once event outputs;
- positive row-recomputed future value on development and held streams;
- bounded memory growth and protected retention;
- no surviving critical interference attack; and
- an eligible Exp6433 and Exp6435 determination.

ARC eligibility is limited to a generic reachability result. Exp6434 cannot
create solve credit or support a hidden-game claim.

The following repeated scopes carry mechanical retirement:

| New task | Prior task | Retire if repeated |
|---|---|---|
| Exp6427 | Exp6414 | The clean corpus again ends with the same flagged evidence verdict. |
| Exp6428 | Exp6417 | The admission A/B again ends with the same flagged positive verdict. |
| Exp6430–Exp6433 | Exp6420 | The fresh CSL chain again ends with non-recomputable rows or open raw-output/cache attacks. |

No retired experiment ID is reused. No task depends on a retired upstream ID.

## 12. Reconciliation obligations

Before V553 is complete:

1. add or update relevant `REQ-*` and `SCENARIO-*` entries before code;
2. run focused unit tests and applicable E2E checks from
   `ops/e2e-test-plan.md`;
3. run roadmap schema, prior-failure, gate, exclusion, spec-coverage,
   adversarial, determination-preservation, and root-clutter checks;
4. preserve the three unrelated dirty audit reports present at planning time;
5. reconcile `openspec/`, `_bmad/traceability.md`, `ops/status.md`,
   `ops/changelog.md`, `ops/known-issues.md`, and claim records; and
6. write one honest next research question from the surviving evidence.

The milestone must not modify `research-roadmap.yaml` during planning. It must
not modify `scripts/research_conductor.py`. It must not push.

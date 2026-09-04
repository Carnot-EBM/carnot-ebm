# Research Roadmap V611: Certified Constraint Adaptation

**Milestone:** 2026.09.611
**Status:** Planned
**Date:** 2026-09-04
**Task contract:** exactly 12 tasks, exp6972 through exp6983, in the order below
**Program anchors:** `research-program.md`, `_bmad/prd.md`, `ops/north-star.md`
**Execution contract:** `research-roadmap-next.yaml`

## Milestone thesis

Carnot has a certified error fixture and a chronological learning stream, but it has not yet run a
clean live experiment that turns those assets into better exact constraint mappings. V611 makes
that transition in one bounded chain:

1. establish a lease-aware three-family GGUF runtime and repair the evidence verifier that falsely
   quarantined the deterministic fixture;
2. compare delayed constraint schedules on immutable exact pairs;
3. certify candidates once, learn a compact PWA-KAN residual, and requalify logit energy only as a
   diagnostic;
4. execute transactional continuous self-learning and audit the live ARC world-model path without
   claiming a solve.

The milestone is successful if it leaves a trustworthy result at every branch boundary. A null is
valid. A blocked task names its failed check and does not retry an external condition as `partial`.

## What V610 proved

Milestone 2026.09.610 completed its executable seven-task YAML, although its design document
incorrectly promised 14 tasks. Its artifacts narrow the next step:

- Exp6965 proved the contract defect: the document held 14 IDs while the active YAML held seven.
  This is the third 14-task underfill in the recent record. V611 therefore fixes the contract at 12
  tasks and makes the parity audit advisory.
- Exp6966 found all three mandated GGUF files, two visible RTX 3090s, and CUDA-capable
  `llama-cpp-python` 0.3.33. It stopped because foreign GPU processes held memory, including an
  orphaned server that stop authority later reaped. The blocker was ownership, not model absence or
  CPU-only inference.
- Exp6967 recomputed the prior mapping failures and froze disjoint calibration, held-out, and 24
  chronological event rows with exact witnesses. Its scientific fields are usable, but the shared
  verifier falsely applied a live-model duration floor because model IDs appeared inside source row
  identifiers.
- Exp6968 correctly refused an ARC generalization claim because the selected live run lacked an
  immutable transition source. New engine files appeared after that artifact, so one changed-state
  audit is justified; another prompt, token-budget, or public-game solve attempt is not.
- Exp6969 was pre-gate blocked on Exp6966. Exp6970 and Exp6971 were skipped downstream. No prompt
  policy, held-out comparison, energy selection, or self-learning science ran.

V610 therefore proved the data substrate and diagnosed two infrastructure boundaries. It did not
prove improved constraint extraction, a verifier moat, continuous learning, or ARC progress.

## Three largest gaps to the PRD

### 1. Natural language still does not compile reliably into exact constraints

The PRD requires LLM proposals to become executable, checkable constraints. V609 certified only
10 of 162 direct mappings, with parse, schema, domain-correspondence, and objective errors. V610
froze a clean error fixture but never generated the changed candidates. V611 tests delayed and
draft-conditioned constraint schedules, then certifies every output with the existing exact
executor.

### 2. FR-11 continuous self-learning is not demonstrated on the current constraint path

The program requires learning from verified outcomes without catastrophic forgetting. The newest
24-event stream is ready, but V610's learning branch never ran. Older exact-slot requalification is
retired, and prior prospective attempts lacked a clean stream. V611 uses transactional post-outcome
writes, hard episode resets, rollback, and a held-future no-forgetting audit. It adapts policy memory,
not model weights.

### 3. The live ARC agent has generated engines without proving useful, reachable foresight

The north star is a live hidden-game discovery process. A generated Python engine is not progress
unless the live path can reach it and its held-out transition or first-step rankings beat controls.
V611 audits a newly available frozen live engine, records reachability, and makes no game or level
solve claim.

## Research inputs selected for this milestone

The 2026-09-04 refresh is recorded at the top of `research-references.md`. The executable hooks are:

- DCCD (`2603.03305`) and In-Writing (`2601.07525`) for separating semantic reasoning from the
  constrained certificate tail;
- Spilled Energy (`2602.18671`) for a single, explicitly prior-failure-bound logit diagnostic;
- optimal PWA abstractions for KAN verification (`2602.06737`) for a compact residual whose bounds
  can be checked by MILP;
- MARCH (`2603.24579`) for information separation between proposal, exact outcome, and memory
  writer;
- world-model-use and false-first-step diagnostics (`2601.03905`, `2602.02991`) for ARC;
- LagONN (`2505.07179`) for keeping feasibility as a hard state rather than blending every
  constraint into a soft score.

EBT, ARM-as-EBM, Kona, Extropic, and recent GitHub EBM repositories remain architectural inputs.
They do not provide a local checkpoint, device, or authenticated service needed by this milestone.

## Target architecture

```text
                         advisory evidence plane
  exp6972 contract audit -------------------------------> exp6983 capstone
  exp6974 claim-provenance verifier ----+----------------> exp6983 capstone
                                       |
                                       v
  Exp6967 frozen exact fixture --> exp6975 delayed-constraint candidate bank
                                           ^
                                           |
  exp6973 lease-aware GGUF runtime ----------+
                                           |
                                           v
                                exp6976 exact certification
                                  |          |          |
                                  |          |          +--> exp6980 spilled energy
                                  |          |
                                  |          +--> exp6977 certified PWA-KAN energy
                                  |                         |
                                  |                         v
                                  |              exp6982 hard-feasible selection
                                  |
  Exp6967 chronological stream ---+--> exp6978 transactional self-learning
                                                    |
                                                    v
                                           exp6979 cold safety audit

  new frozen live ARC engine ----------------> exp6981 reachability/generalization audit

  all terminal artifacts --------------------> exp6983 ungated reconciliation
```

The exact verifier stays outside the generator and memory writer. Learned energy may rank candidates
only after hard feasibility has been represented separately. The capstone reads every terminal
artifact but gates no experiment.

## Phase A: Evidence and runtime boundaries

### Exp6972: V611 advisory task-contract and gate audit

Verify that this document and `research-roadmap-next.yaml` contain exactly the same 12 tasks,
exp6972 through exp6983, with matching order, titles, deliverables, model contracts, prior failures,
gate fields, and prompt endings. The result is advisory and cannot block science.

### Exp6973: Lease-aware three-family GGUF runtime handoff

Repeat the narrow runtime canary only because Exp6966's foreign owners have changed. Acquire the
existing GPU lease, distinguish owned from foreign processes, load one required model at a time,
generate a short fixed completion, tear down, and prove VRAM return. Do not kill or reuse an
unowned server. The ready score is 1 only when all three required families complete on CUDA.

### Exp6974: Claim-provenance-aware duration verification

Repair the shared artifact verifier's category error: model names in immutable source rows are not
evidence that the current task invoked a model. Use explicit invocation and provenance fields rather
than a substrate allowlist. Preserve detection for genuinely impossible live durations, add mutation
tests against field hiding, and issue a clean admissibility receipt for the Exp6967 fixture without
rewriting its historical artifact.

## Phase B: Constraint compilation and certified energy

### Exp6975: Three-family delayed-constraint candidate bank

On six calibration and six held-out pairs sampled before generation, compare three frozen schedules:
direct ConstraintIR, trigger-switched structured output, and draft-conditioned structured output.
Run every cell on Qwen3.6-35B-A3B, Gemma-4-31B-it, and Gemma-4-26B-A4B. Capture raw text and
adjacent-step energy scalars before parsing. Labels and solver feedback remain hidden during
generation. This is a 108-attempt bounded pilot, not a benchmark headline.

### Exp6976: Exact candidate certification and policy selection

Certify every Exp6975 output with Z3 and bounded enumeration. Freeze a policy using calibration
rows only, then open held-out labels once. Report parse, schema, semantic, and objective correctness,
per-family headroom, and paired schedule deltas. A positive result is `circular_positive` because
the exact verifier supplies selection and assessment labels.

### Exp6977: Certified PWA-KAN residual energy

Fit a compact KAN residual to calibration candidate features and exact error labels. Convert each
unit to a piecewise-affine abstraction, allocate a finite piece budget, and use MILP to prove stated
bounds and invariants. Evaluate untouched held-out rows. The exact executor remains the external
assessor; the learned scorer cannot override infeasibility.

## Phase C: Continual learning and hallucination signals

### Exp6978: Transactional verifier-grounded continuous self-learning

Run the selected constraint schedule over Exp6967's sealed 24-event chronological stream with the
Qwen3.6 flagship. Compare frozen, read-only-memory, and transactional-write arms under matched
generation budgets and fresh contexts. The writer sees atomic error certificates only after the
exact outcome, commits through a journal, and rolls back a harmful update. Report plasticity,
stability, forgetting, state growth, and held-future exact success.

### Exp6979: Fresh-process self-learning safety audit

In a process that cannot call an LLM or mutate the store, replay the event order, journal, rollback,
and headline arithmetic from Exp6978. Confirm that no future label or later outcome influenced an
earlier choice. Preserve a negative result if the learning arm did not beat read-only memory.

### Exp6980: Span-localized spilled-energy requalification

Recompute spilled and marginalized energy from Exp6975's captured adjacent-step scalars, localized
to the exact emitted mapping span. Compare against entropy and token-confidence baselines on the
locked held-out rows. This is the one permitted requalification after Exp2497's AUROC 0.4903 null.
If the preregistered held-out gate fails, record a terminal null and retire this diagnostic again.

## Phase D: Live-agent and system-level decisions

### Exp6981: ARC live-engine reachability and generalization audit

Select the first complete post-Exp6968 live engine whose manifest includes immutable prompt and
transition hashes. Execute it in a restricted fresh process, score held-out changing and no-op
transitions, compare identity, constant-delta, nearest-shown, and memorization controls, and verify
that the live agent can actually route to the engine. Measure first-step counterfactual ranking.
Do not generate a replacement engine, inspect game source, claim a solve, or update the registry.

### Exp6982: Hard-feasible hybrid energy selection

Where Exp6976 exposes real held-out headroom, compare likelihood, exact-feasibility-only,
PWA-residual-only, and hard-feasible-plus-PWA selection on identical candidate groups. Feasibility is
a hard projection; energy ranks only within it. Report every group, abstention, and selection cost.
Any improvement is circular because the exact feasibility filter participates in selection.

### Exp6983: V611 independent capstone and V612 handoff

Recompute the document/YAML contract and all scientific headlines from unflagged per-unit rows.
Separate positive, circular-positive, null, blocked, disqualified, and partial classes. Do not count
pre-gate placeholders, advisory receipts, or model availability as science. Recommend V612 from the
first unresolved causal boundary, not from task completion counts.

## Dependency graph

```text
exp6972  advisory only -----------------------------------------------> exp6983

exp6973 lease-aware runtime ----+--> exp6975 candidate bank
                                +--> exp6978 self-learning

exp6974 fixture admissibility --+--> exp6975 candidate bank
                                +--> exp6978 self-learning

exp6975 --> exp6976 exact certification --+--> exp6977 certified PWA energy
                                          +--> exp6978 self-learning --> exp6979
                                          +--> exp6980 spilled energy
                                          +--> exp6982 hybrid selection

exp6977 certified PWA energy ----------------> exp6982 hybrid selection

exp6981  independent ARC branch
exp6983  ungated; reads every artifact or absence
```

Every structured gate names a field required by its producer. Cross-milestone evidence such as
Exp6967 is checked as a task-local precondition because structured gates may reference only tasks in
this roadmap. No task gates on Exp6972 or the capstone.

## Exact task contract

| Order | Task ID | Exact title | Deliverable | Structured prerequisites |
|---:|---|---|---|---|
| 1 | exp6972-v611-contract-advisory | V611 advisory task-contract and gate audit | `results/experiment_6972_v611_contract_advisory.json` | none |
| 2 | exp6973-lease-aware-gguf-runtime | Lease-aware three-family GGUF runtime handoff | `results/experiment_6973_lease_aware_gguf_runtime.json` | none |
| 3 | exp6974-claim-provenance-duration-lint | Claim-provenance-aware duration verification | `results/experiment_6974_claim_provenance_duration_lint.json` | none |
| 4 | exp6975-delayed-constraint-candidate-bank | Three-family delayed-constraint candidate bank | `results/experiment_6975_delayed_constraint_candidate_bank.json` | exp6973 `lease_aware_runtime_ready_score == 1`; exp6974 `fixture_admissibility_ready_score == 1` |
| 5 | exp6976-exact-candidate-certification | Exact candidate certification and policy selection | `results/experiment_6976_exact_candidate_certification.json` | exp6975 `candidate_bank_complete_score == 1` |
| 6 | exp6977-certified-pwa-kan-energy | Certified PWA-KAN residual energy | `results/experiment_6977_certified_pwa_kan_energy.json` | exp6976 `candidate_certification_complete_score == 1` |
| 7 | exp6978-transactional-constraint-self-learning | Transactional verifier-grounded continuous self-learning | `results/experiment_6978_transactional_constraint_self_learning.json` | exp6973 `lease_aware_runtime_ready_score == 1`; exp6974 `fixture_admissibility_ready_score == 1`; exp6976 `selected_policy_ready_score == 1` |
| 8 | exp6979-self-learning-cold-audit | Fresh-process self-learning safety audit | `results/experiment_6979_self_learning_cold_audit.json` | exp6978 `self_learning_run_complete_score == 1` |
| 9 | exp6980-spilled-energy-requalification | Span-localized spilled-energy requalification | `results/experiment_6980_spilled_energy_requalification.json` | exp6975 `candidate_bank_complete_score == 1`; exp6976 `candidate_certification_complete_score == 1` |
| 10 | exp6981-arc-live-engine-generalization-audit | ARC live-engine reachability and generalization audit | `results/experiment_6981_arc_live_engine_generalization_audit.json` | none |
| 11 | exp6982-hard-feasible-hybrid-selection | Hard-feasible hybrid energy selection | `results/experiment_6982_hard_feasible_hybrid_selection.json` | exp6976 `candidate_certification_complete_score == 1`; exp6976 `heldout_headroom_group_count >= 2`; exp6977 `certified_pwa_energy_ready_score == 1` |
| 12 | exp6983-v611-capstone | V611 independent capstone and V612 handoff | `results/experiment_6983_v611_capstone.json` | none |

This table is the design-document contract. `research-roadmap-next.yaml` must contain these 12 rows,
in this order, without reserved or omitted IDs.

## Failed-scope boundaries

- Exp6973 cites Exp6966 and the old no-offload receipt. Its changed mechanism is lease-aware
  ownership plus the now-confirmed CUDA runtime; it is not another CPU-only offload rerun.
- Exp6975 cites the blocked Exp6969 bank, retired Exp5923 schema reprompt, and Exp5813 finite-ID
  transport. Delayed/draft-conditioned generation is the changed technique. Exact semantics remain
  external, and finite answer IDs are not revived.
- Exp6978 cites all matching continuous-learning failures. Its clean Exp6967 stream,
  post-outcome journal, information-separated writer, and rollback are the changed prerequisites and
  technique. It does not reuse the retired Exp5895 exact slot.
- Exp6980 cites Exp2497 and has `retire_if_same_verdict: true`. Another null closes the diagnostic;
  no prompt or pooling retune follows.
- Exp6981 cites Exp6968. The only reopening evidence is a newer live engine plus immutable source
  hashes. Absence yields `blocked`, not a generated substitute.
- Exp6982 cites Exp6958 and Exp6959. A learned certified PWA residual inside hard feasibility is the
  new mechanism; hand-weight retuning and headroom-free selection stay closed.
- No task reopens external generated-text scorers, schema-only retries, prefix rejection, public ARC
  re-solves, token-budget-only ARC induction, or a physical hardware scope without changed state.

## Hardware requirements

| Tasks | Required hardware | Runtime contract |
|---|---|---|
| exp6973 | Dual RTX 3090 | Existing lease, one required GGUF loaded at a time, CUDA offload receipt, owned-process teardown, VRAM return within 512 MiB |
| exp6975 | Dual RTX 3090 | Three required GGUF families, sequential family workers, 108 bounded attempts, 128-token cap, checkpoint after each pair block |
| exp6978 | One RTX 3090 minimum; second available for clean handoff | Qwen3.6-35B-A3B flagship, 24 chronological events, three matched arms, fresh context per event, transaction journal |
| exp6972, exp6974, exp6976-exp6977, exp6979-exp6983 | CPU and RAM | Deterministic reducers, Z3/bounded enumeration, small PyTorch KAN training if needed, MILP, fresh-process replay |

Required local model IDs:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

Exp6973 and Exp6975 use all three. Exp6978 uses the Qwen flagship. Any legacy small model is a
clearly labeled CPU smoke test and cannot contribute a headline row. GGUF tokenization uses the
embedded llama.cpp tokenizer path, never `AutoTokenizer.from_pretrained()` on a GGUF-only repo.

No FPGA, XDNA, TSU, or Kona access is required. GateMate, KV260, PolarFire, and XDNA have no new
changed-state receipt; Extropic has no authenticated device route; Kona has no public local runner.

## Measurement and verdict rules

- Every task declares `verdict_class` from the closed enum `positive | circular_positive | null |
  blocked | disqualified | partial` beside `honest_verdict`.
- Every comparison emits per-unit rows. Aggregate claims must recompute from those rows.
- Every blocked artifact emits `gate_check_summary` with failed check, expected value, and observed
  value. External absence is `blocked`, never `partial`.
- Every live task records exact `MODEL_SPECS`, model IDs, file hashes, CUDA evidence, task-owned
  duration, checkpoints, and process ownership.
- Exact solver participation in selection makes a positive result `circular_positive`. Learned
  PWA-KAN and spilled-energy diagnostics are oracle-distinct only when the exact solver is used for
  labels and evaluation, not inference-time selection.
- ARC Exp6981 emits `solve_claimed=false`, `level_claimed=false`, `registry_updated=false`, and
  `submitted_to_leaderboard=false`. It cannot change `ops/arc_solve_registry.yaml`.
- The capstone is ungated and excludes flagged artifacts, missing per-unit evidence, pre-gate
  placeholders, and advisory infrastructure from scientific success counts.

## Success criteria

V611 is complete when all 12 tasks have terminal outcomes and the document/YAML contract still
matches. Scientific promotion requires, independently:

1. at least two held-out groups with candidate headroom and a preregistered delayed-constraint
   policy result;
2. a non-tautological PWA/MILP certificate before the learned residual enters hybrid selection;
3. transactional learning that improves held-future exact success over read-only memory without a
   forgetting or rollback violation;
4. either an ARC engine that beats the strongest held-out control through a reachable live path, or
   a precise blocked/null receipt that identifies the next causal boundary.

None is forced. The honest fallback is a reconciled null or blocked milestone with a narrower V612
handoff.

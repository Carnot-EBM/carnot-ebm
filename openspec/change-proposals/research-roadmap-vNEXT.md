# Carnot Research Roadmap vNEXT — V580: Receipt-Bound Verification and Prospective Memory

**Status:** Proposed  
**Planning date:** 2026-08-26  
**Milestone:** `2026.08.580`  
**Execution manifest:** `research-roadmap-next.yaml`  
**Task contract:** 13 tasks, `exp6647` through `exp6659`, in conductor order

## Executive decision

V580 separates infrastructure admission from scientific conclusions and gives
three independent research branches their own execution paths. One narrow
receipt reducer repairs the V579 admission contract; it does not rewrite the
working lease implementation and it does not make model-quality claims. The
verification branch then establishes real SOTA direct headroom before trying
failure-localized suffix regeneration. The continuous-learning, ARC, and Ising
branches do not depend on that GPU chain, so an admission block cannot erase the
whole milestone again.

The milestone asks four bounded questions:

1. Can the three mandated local GGUF families be admitted with task-owned
   identity, process, phase, accelerator, and unload receipts without treating
   the unrelated repo-wide test baseline as a model failure?
2. On a frozen exact-checkable constraint corpus, does exact failure-localized
   suffix regeneration beat full retry at a matched proposal-token budget?
3. Can state-grounded, validation-gated repair memory improve future exact
   outcomes across prospective task orders while preserving held anchors,
   recoverable support, restart, and rollback?
4. Do a game-agnostic trace automaton and an independent bounded-treewidth Ising
   reference produce useful live-policy and sampler evidence without a solve,
   hardware, or asymptotic claim?

## What V579 actually proved

V579 reached terminal conductor outcomes, but it did not answer its scientific
questions.

| V579 scope | Terminal evidence | What is proved | V580 consequence |
|---|---|---|---|
| Exp6633 task-scoped lease journal | The artifact reports `blocked_gpu_lease_scheduler_not_ready`; owner/race/recovery/tamper fixtures and focused checks passed, while an unrelated repo-wide xdist/CWD failure made `focused_tests` false in the reducer. | The lease mechanism is substantially implemented; the admission reduction boundary is wrong. | Recompute readiness from preregistered task-owned receipts and record the global baseline separately. Do not rebuild the lease. |
| Exp6634 mandated-model admission | `blocked_gate_check_failed` on Exp6633's zero score. | No GGUF family was scientifically admitted. | Repeat only after the changed receipt reducer, with all three mandated families and independent fresh processes. |
| Exp6635 direct headroom | No terminal artifact after the admission block. | No direct-headroom conclusion exists. | Create a smaller exact-certificate proposal corpus after admission. |
| Exp6636 delayed two-level decoding | `blocked_gate_check_failed`; its expected upstream artifact did not exist. | No constrained-generation conclusion exists. | Replace token masking with SRD-inspired post-prefix suffix regeneration. This is not the retired finite-ID transport mechanism. |
| Exp6637/6638 verifier calibration and audit | No terminal artifacts after the upstream cascade. | No verifier-unit or independent-audit conclusion exists. | Use the completed candidate corpus as the only gate field and spell that field identically in producer and consumers. |
| Exp6639 Kac-Ward reference | Three consecutive hard wall-clock caps and no artifact. | The proposed implementation scope was too broad for one task; no exact-reference result exists. | Change technique to a bounded-treewidth junction-tree reference, with a separate schedule experiment. |

Two contract lessons are load-bearing. First, the active V579 YAML contained
seven tasks (`exp6633`-`exp6639`) while its prior design document described
fourteen (`exp6633`-`exp6646`). V580's document and YAML therefore declare the
same 13 IDs, titles, deliverables, order, and gates. Second, a task-owned
readiness field must be based on task-owned evidence. A known global regression
is still recorded honestly, but it cannot silently relabel passing lease
fixtures as a failed model.

## Three largest gaps to the PRD vision

| Rank | Gap | Current evidence | V580 closure attempt |
|---:|---|---|---|
| 1 | No trustworthy local-SOTA path from proposal through exact intervention evidence | The exact two-level corpus exists, but V579 admitted no mandated model and produced no direct or intervention rows. External generated-text verifier scoring is retired. | Receipt-scoped admission, a complete exact-certificate proposal corpus, twin verifier-unit calibration, SRD-style suffix regeneration, and independent row audit. |
| 2 | FR-11 continuous self-learning is not yet prospective, state-grounded, and rollback-safe in the current method chain | Revocable and verifier-bounded memory primitives exist, while V576's invariant-memory path had zero prospective benefit and its self-learning task blocked on tests. | Recuris-inspired working/experiential memory separation, targeted patch admission, frozen and context-only controls, multiple task orders, support checks, restart, poison, and rollback attacks. |
| 3 | Transferable reasoning and sampling primitives lack fresh independent evidence | ARC's live supervisor path is reachable but lacks enough outcome-bearing selection evidence; spectral k-block scaling lacks an independent exact reference; attached boards have no changed-state receipts. | A trace-derived game-agnostic FSM tested through the live E3 policy and a bounded-treewidth exact Boltzmann reference followed by autocorrelation-aware schedule comparisons. |

## External research incorporated

The dated findings and product checks are recorded in
`research-references.md` under “V580 planner refresh.” The experiments use the
following ideas:

- **Recuris (arXiv:2608.24876):** separate verified working state from
  experiential memory; localize updates; admit them only on paired held-out
  evidence.
- **Selective Regenerative Decoding (arXiv:2608.24338):** preserve useful
  prefixes and regenerate only degraded suffixes, measured against full retry
  at a fixed compute budget.
- **AutoSaddler (arXiv:2608.23041):** diagnose the concrete failing receipt and
  make targeted, validation-selected harness changes instead of broad rewrites.
- **Automata from Agent Traces (arXiv:2608.23670):** derive a compact behavioral
  state machine from cross-run traces and test it on held-out live-policy paths.
- **StepGuard (arXiv:2608.24777) and V579 verifier-unit work:** pair unsafe
  actions with clean twins and report prevented violations together with
  blocked-valid actions.
- **Scaling Up Thermodynamic AI Models (arXiv:2607.00170):** make schedule and
  autocorrelation cost explicit; do not equate raw transitions with independent
  samples.

OpenReview and Hugging Face reinforce selective intervention and independent
executable checking, but supply no matching-base local EBT/ARM-EBM checkpoint.
Semantic Scholar returned the same dated discovery counts—35 visible EBT
citations and eight ARM-EBM citations—and no reproducible matching-base local
checkpoint. Extropic still schedules Z1 access for 2027. Kona describes
whole/partial-trace energy reasoning and Spec-Code-Proof alignment but exposes
no public weights or local runner. None is an executable V580 baseline.

## Target architecture

```text
                       PHASE A — RECEIPT-BOUND ADMISSION

  Exp6633 artifact + task-owned fixtures + known global-suite receipt
                              |
                              v
                 [6647 admission-boundary reducer]
                              |
                     task_owned_admission_ready_score == 1
                              v
               [6648 three-family GGUF canaries]
                              |
                    all_mandated_models_admitted == true
                              v
               [6649 exact proposal/headroom corpus]
                         /                     \
                        v                       v
        PHASE B — VERIFICATION                  regeneration_headroom >= 8
        [6650 twin verifier-unit map]            |
                                                  v
                                 [6651 failure-localized suffix A/B]
                                                  |
                                                  v
                                 [6652 independent row/claim audit]

        PHASE C — INDEPENDENT ADAPTATION AND ARC

  existing exact repair events              archived live E3 traces
              |                                      |
              v                                      v
  [6653 state-grounded memory fixture]    [6656 trace-FSM live LOO A/B]
              |
              v
  [6654 prospective memory evolution]
              |
              v
  [6655 poison/restart/rollback audit]

        PHASE D — INDEPENDENT SAMPLING AND SYNTHESIS

  bounded-treewidth Ising fixtures
              |
              v
  [6657 exact junction-tree reference]
              |
              v
  [6658 autocorrelation-aware schedule A/B]

  all terminal artifacts, including honest blocks/nulls
              |
              v
  [6659 V580 capstone and claim reconciliation]
```

The three independent roots—Exp6647, Exp6653/6656, and Exp6657—prevent a
single missing GPU receipt from blocking continuous self-learning, the required
ARC generalization slot, or the sampling reference.

## Phase A — Receipt-bound admission and direct evidence

### Exp6647 — Receipt-scoped model admission boundary

**Question:** Can V579's passing task-owned lease receipts be reduced
independently of the unrelated repo-wide xdist/CWD baseline?

**Deliverable:** `results/experiment_6647_receipt_scoped_admission_boundary.json`

This is a changed rerun of Exp6633's terminal block, not a lease rewrite. It
must replay owner, race, PID-start, heartbeat, phase, unload, recovery, and
tamper fixtures; enumerate the exact task-owned gate set before execution; and
record the repo-wide suite result under a separate non-gating field. It may set
`task_owned_admission_ready_score=1.0` only if every preregistered owned check
passes. Ready infrastructure uses `verdict_class=null`; no model-quality claim
is allowed.

### Exp6648 — Three-family GGUF accelerator canaries

**Question:** Are all three mandated GGUF families loadable and inferencing in
independent owned processes on the measured RTX path?

**Deliverable:** `results/experiment_6648_three_family_gguf_canaries.json`

The task resolves the Qwen/middle-Gemma pair through `cached_sota_pair()` and
the dense Gemma through `resolve_cached_gguf()`. Each family gets its own fresh
process and owner-bound receipt: exact model ID and file hash, embedded GGUF
tokenizer probe, PID/start, device UUID, phase transitions, VRAM before/resident/
after, prompt hash, non-empty output, exit, unload, and release. The task is
infrastructure admission, not a quality comparison.

### Exp6649 — Exact certificate proposal corpus

**Question:** On preregistered exact-checkable tasks, do the flagship MoE and
middle MoE produce a complete direct candidate corpus with enough
prefix-repairable failures to test localized regeneration?

**Deliverable:** `results/experiment_6649_exact_certificate_proposal_corpus.json`

Use at least 24 fixed tasks and both
`unsloth/Qwen3.6-35B-A3B-GGUF` and
`unsloth/gemma-4-26B-A4B-it-GGUF`. Every row stores task, prompt, model, seed,
raw output hash, parsed plan, exact outcome, first failing step, valid-prefix
length, token count, and receipt lineage. `candidate_corpus_complete` depends on
row completeness and exact recheckability, not on whether headroom exists.
`regeneration_headroom_count` is a separate measured field.

## Phase B — Verification unit and localized regeneration

### Exp6650 — Twin-prefix verifier discrimination map

**Question:** Which verification unit—one step, two steps, or full remaining
suffix—best separates paired clean/error candidates without merely rejecting
more valid work?

**Deliverable:** `results/experiment_6650_twin_prefix_verifier_map.json`

Build byte-matched clean/error twins from Exp6649 rows. Report catch rate,
false-reject rate, informedness, AUROC/AUPRC where defined, abstention, and
latency per unit. Exact labels remain authority; learned signals only measure or
route. This task runs on frozen rows and invokes no new LLM.

### Exp6651 — Failure-localized suffix regeneration A/B

**Question:** When an exact checker identifies the first invalid step, does
preserving the valid prefix and regenerating only the suffix outperform a full
retry at the same proposal-token budget?

**Deliverable:** `results/experiment_6651_failure_localized_suffix_regeneration.json`

Use the fixed invalid/headroom rows and both
`unsloth/Qwen3.6-35B-A3B-GGUF` and
`unsloth/gemma-4-31B-it-GGUF`. Compare direct candidate, full retry, and
localized suffix regeneration. The method may prompt from an accepted prefix
plus exact failure information; it may not mask finite answer IDs, force a
grammar carrying answer semantics, or reuse the retired generated-answer
transport path. Match total generated proposal tokens per row. Because the same
exact checker localizes the intervention, any favorable intervention result is
`circular_positive`, not an independent positive claim.

### Exp6652 — Constraint intervention independent row audit

**Question:** Do the raw rows support the claimed validity, token-budget,
prefix-preservation, and model-identity aggregates?

**Deliverable:** `results/experiment_6652_constraint_intervention_audit.json`

Recompute every headline from raw rows, rerun exact checks from blinded task
inputs, verify model/process/accelerator receipts, and attack leakage, duplicate
rows, unequal budgets, parser nulls, impossible prefix claims, and oracle
circularity. The audit cannot upgrade an oracle-assisted result to independent
positive evidence.

## Phase C — Continuous self-learning and ARC generalization

### Exp6653 — State-grounded repair memory fixture

**Question:** Can exact repair events be represented as separate working-state
and experiential-repair records with targeted, revocable updates?

**Deliverable:** `results/experiment_6653_state_grounded_repair_memory_fixture.json`

Build a deterministic fixture from existing exact constraint/repair artifacts,
not from new LLM generation. Each event records the visible state, violated
constraint, exact witness, candidate repair, applicability key, support before
and after, held-anchor split, provenance, version, and inverse rollback patch.
No future outcome may enter a lookup key. This is data/schema readiness and uses
`verdict_class=null` when ready.

### Exp6654 — Prospective experiential repair evolution

**Question:** Across at least three preregistered task orders, does
validation-gated experiential memory improve future exact outcomes relative to
frozen and context-only controls without collapsing recoverable support?

**Deliverable:** `results/experiment_6654_prospective_repair_memory_evolution.json`

Run frozen, context-only, and verified-memory arms over identical exact repair
operator candidates. Patches may change only one typed memory component and are
admitted only after source-event repair, paired held-anchor non-regression, and
support-floor checks. Evaluation is prequential: event `t` may use only
information committed before `t`. Report per-event retrieval and action
influence, retained/retired patches, regret, future exact yield, forgetting,
support, and order sensitivity. This is V580's required Continuous
Self-Learning experiment and invokes no LLM.

### Exp6655 — Adversarial durability of repair memory

**Question:** Does the prospective memory result survive corruption, duplicate,
unsupported-update, restart, and rollback attacks?

**Deliverable:** `results/experiment_6655_repair_memory_safety_audit.json`

Replay artifacts from disk, inject conflicting and low-support events, verify
checksum and version failures close, and prove that rejected or harmful patches
restore the exact prior state. Recompute all multi-order arm deltas from rows.
No same-policy self-grade may authorize a memory update.

### Exp6656 — Trace-automaton ARC live supervisor LOO A/B

**Question:** Can a game-agnostic finite-state supervisor learned from archived
live E3 traces reduce held-family unproductive or forbidden actions when it is
actually reachable by the scored `make_carnot_agent`/`E3AgentPolicy` path?

**Deliverable:** `results/experiment_6656_arc_trace_automaton_live_loo.json`

Derive a compact FSM only from policy-visible trace fields. Freeze topology and
thresholds before held-family evaluation. Run paired supervisor-off/on cells
through the actual live E3 path across at least three held-out game families
and three seeds, recording redirect firings, actual action changes, valid-action
blocks, violations prevented, actions to exact observed progress, and receipt
lineage. Do not read game source, use per-game adapters, run offline BFS, or
claim any game or level solve. The reusable primitive and held-family design
satisfy the ARC generalization floor even if the result is null.

## Phase D — Exact Ising reference, schedule test, and capstone

### Exp6657 — Bounded-treewidth exact Ising reference

**Question:** Can a small junction-tree/dynamic-programming implementation
produce normalized likelihoods and independent exact samples on preregistered
bounded-treewidth Ising graphs?

**Deliverable:** `results/experiment_6657_bounded_treewidth_ising_reference.json`

This changes technique from the failed Kac-Ward attempt. Use at least 12 graph
fixtures with treewidth at most four, including fields and frustration where
the representation supports them. Compare partition function, marginals,
likelihoods, and samples against brute-force enumeration at small `n`. Reject
unsupported graphs explicitly. A ready reference is null infrastructure
evidence, not a sampler speed claim.

### Exp6658 — Autocorrelation-aware thermodynamic schedule A/B

**Question:** Relative to the independent exact reference, does an
autocorrelation-aware temperature/transition schedule improve effective samples
per measured wall-second over fixed schedules on bounded graphs?

**Deliverable:** `results/experiment_6658_thermodynamic_schedule_ab.json`

Compare fixed-transition, fixed-temperature-ladder, and
autocorrelation-aware schedules across graph and seed rows. Report setup,
transition, and end-to-end wall time separately; energy and marginal error;
normalized-likelihood diagnostics; ESS; integrated autocorrelation time; and
failure/support rows. This is CPU/JAX or local CUDA software evidence only—no
TSU, FPGA, asymptotic, or energy-efficiency claim.

### Exp6659 — V580 evidence synthesis

**Question:** Which V580 claims are positive, circular-positive, null, blocked,
disqualified, or partial after independent row recomputation?

**Deliverable:** `results/experiment_6659_v580_capstone.json`

Aggregate every available artifact without inventing missing zeros. Recompute
gates and comparisons from rows, preserve circularity and solve provenance,
list retired reruns, and update only the relevant capability specs,
`_bmad/traceability.md`, `ops/status.md`, and `ops/changelog.md`. A terminal
capstone must run even when an upstream branch blocks.

## Dependency graph and conductor order

```text
6647
  └─6648 [task_owned_admission_ready_score == 1.0]
      └─6649 [all_mandated_models_admitted == true]
          ├─6650 [candidate_corpus_complete == true]
          └─6651 [candidate_corpus_complete == true AND regeneration_headroom_count >= 8]
              └─6652 [repair_comparison_complete == true]

6653
  └─6654 [memory_fixture_ready == true]
      └─6655 [prospective_memory_comparison_complete == true]

6656  (independent ARC root)

6657
  └─6658 [ising_reference_ready == true]

6659  (last in YAML; aggregates all terminal/available artifacts, no science gate)
```

All structured gate fields above are declared with the identical spelling in
the upstream task's required artifact fields. No task requires a retired
experiment ID. Prior failures are documentary inputs only.

## Experimental model policy

Only three tasks invoke an LLM. Their `MODEL_SPECS` are mandatory and explicit.

| Task | Experimental models | Resolution contract | Headline use |
|---|---|---|---|
| Exp6648 | Qwen3.6-35B-A3B, Gemma-4-31B-it, Gemma-4-26B-A4B-it | `cached_sota_pair()` for Qwen + middle MoE; `resolve_cached_gguf()` for dense Gemma; embedded GGUF tokenizer | Admission only |
| Exp6649 | Qwen3.6-35B-A3B + Gemma-4-26B-A4B-it | `cached_sota_pair(gpu_indices=(0,1), model_indices=(0,1))` | Direct proposal/headroom rows |
| Exp6651 | Qwen3.6-35B-A3B + Gemma-4-31B-it | cached Qwen plus `resolve_cached_gguf()` dense path | Matched-budget repair comparison |

Full hub IDs:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

Legacy `Qwen/Qwen3.5-0.8B` and `google/gemma-4-E4B-it` may run only as labeled
CPU smoke tests. Their rows cannot satisfy admission, headroom, repair, or
headline gates. No experiment may call Hugging Face `AutoTokenizer` on a GGUF
repository; the tokenizer is embedded in the GGUF and must be checked through
the llama.cpp-backed path.

## Hardware requirements

| Resource | Tasks | Requirement and claim boundary |
|---|---|---|
| Dual RTX 3090, 24 GB each | Exp6648, Exp6649, Exp6651 | Required for independent local GGUF processes. Record GPU UUID, PID/start, model hash, phase, VRAM, exit, unload, and lease release. Do not infer quality from accelerator admission. |
| Host CPU and RAM | All; especially Exp6647, Exp6650, Exp6652-6658 | Current host is sufficient for reducers, exact checks, memory streams, ARC fixtures, and bounded-treewidth graphs. Record measured resources in each artifact. |
| Local disk/model cache | Exp6648, Exp6649, Exp6651 | All mandated Q4_K_M GGUFs must resolve locally. No download is part of the experiment. |
| ARC/Arcade runtime | Exp6656 | Use only canonical live E3 entrypoints and policy-visible observations. No game source or per-game adapter. |
| KV260, GateMate, PolarFire | None | No changed-state receipt exists. Repetitive continuity or bitstream redesign is outside V580. |
| Extropic TSU/Z1 | None | No authenticated runner; 2027 early-access statements are not hardware evidence. |

## Artifact and evidence contract

Every task must:

- write exactly one declared JSON deliverable atomically;
- declare `honest_verdict` and closed `verdict_class` in
  `{positive, circular_positive, null, blocked, disqualified, partial}`;
- make any blocked verdict start `blocked_` and populate
  `gate_check_summary` with the failed check and observed value;
- emit `per_unit_rows` for every compared model, arm, task, graph, seed,
  condition, task order, or attack;
- recompute aggregates from rows and record provenance, hashes, seed, duration,
  commands, exit codes, and inference substrate;
- preserve `research-roadmap.yaml` and `scripts/research_conductor.py` byte-for-byte;
- run focused unit tests, spec coverage, row consistency, adversarial
  verification, and the applicable checks in `ops/e2e-test-plan.md`;
- update the relevant `openspec/capabilities/*/spec.md` before implementation
  and reconcile traceability/status/changelog only for code actually changed;
- never translate a missing artifact, null row, parser failure, or unavailable
  denominator into zero.

ARC Exp6656 makes no level-solve claim, so it must record a no-solve receipt and
must not populate a credited solve. If any implementation unexpectedly adds a
level-solve claim, it must add `solve_provenance` and only
`live_agent_self_discovery` can receive credit.

## Failed-experiment discipline

The YAML carries `prior_failures` for every materially overlapping terminal or
failed scope: Exp6633, Exp6634, Exp6635, Exp6636, Exp6637, Exp6638, Exp5913,
Exp6614, Exp6290, Exp6524, Exp6639, and Exp6612. Each entry records the observed terminal result,
the changed technique or newly shipped prerequisite, and
`retire_if_same_verdict: true`. V580 reuses none of those experiment IDs and
requires none as a conductor dependency. No operator override is needed.

## Milestone acceptance boundary

V580 is successful as a research milestone if it produces honest terminal
evidence for all independent branches, not only if every method wins.

- Phase A succeeds when task-owned admission is either proven or blocked with a
  named owned check; global-suite truth remains visible.
- Phase B succeeds when complete raw candidate rows support either a bounded
  regeneration comparison or an honest no-headroom closure.
- Phase C succeeds when the memory experiment runs prospectively across task
  orders and the live E3 ARC primitive records real policy influence or an
  honest null/no-firing result.
- Phase D succeeds when the exact Ising reference is independently checked and
  the schedule task reports ESS/autocorrelation or an explicit unsupported
  boundary.
- Exp6659 reconciles all claims and does not promote circular-positive, blocked,
  missing, or off-path evidence.

The conductor must execute `research-roadmap-next.yaml`; it must not modify or
activate `research-roadmap.yaml` as part of this planning task.

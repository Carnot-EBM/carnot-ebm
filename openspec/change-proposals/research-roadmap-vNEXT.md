# Carnot Research Roadmap vNEXT: Exact Proof Transport and Active Procedural Memory

**Milestone:** `2026.08.589`
**Created:** 2026-08-29
**Status:** Proposed
**Supersedes:** milestone `2026.08.588`
**Research basis:** `research-program.md`, `_bmad/prd.md`, `_bmad/architecture.md`,
`ops/status.md`, `ops/changelog.md`, `research-complete.yaml`, `research-roadmap.yaml`, prior
roadmap proposals, `ops/conductor-log.md`, `research-hardware-wishlist.md`, and the
`V589 Planner Refresh` in `research-references.md`.

## What milestone 2026.08.588 proved

Milestone `.588` completed all thirteen planned tasks. It restored broad execution after `.587`,
but its terminal capstone was correctly partial. The useful evidence is specific:

- All three mandated local GGUF families loaded on CUDA and produced attributable output. The
  exact 72-instance certificate stream also passed deterministic replay.
- Exp6745 retained 216/216 planned proposal rows, but exact-valid yield was 0/216. Every row was
  labeled `malformed_certificate`, so Exp6746 could not form a two-class diagnostic panel and
  Exp6747 gate-blocked.
- The malformed result is now locally explainable. All 216 `raw_output` fields begin as Python byte
  literal envelopes such as `b'SAT ...'`. Both the Qwen and Gemma families show this boundary form.
  The frozen parser therefore saw `b'SAT` instead of `SAT` and returned `unknown_claim` on every
  row. `.588` did not test a lossless decode-and-reparse path.
- The transactional memory fixture passed. The prospective self-learning comparison then produced
  a null result. The cold audit recorded zero commits and zero rejects, so the branch did not test
  an active learning mechanism.
- The Thermalizers-style compiler produced bounded simulator results, but its artifact declared
  `verifier_is_oracle=true` with `verdict_class=positive`. The live adversarial check correctly
  treats that combination as circular. The result cannot support a non-circular portability claim.
- The 32K code-carrying ARC preflight passed on Qwen3.8 and Qwen3.6. The paired object-table A/B did
  not start because GPU 0 had 6,235 MiB free while the frozen load required 22,610 MiB. This is a
  resource-admission block, not a negative result about fetch-on-demand.
- The activated handoff audit blocked because it expected rendered paths while the roadmap contract
  intentionally preserved `{project_root}` and `{date}` placeholders. That validator mismatch did
  not suppress unrelated science and should not become another milestone root.

The natural next step is therefore not a broad new architecture. First recover the proof evidence
already paid for. Then compare a new environment-indexed decoding mechanism on the remaining
semantic failures. In parallel, make memory admission active on a controlled non-saturating stream.
Finally, obtain the two missing independent receipts: a resource-owned live ARC comparison and a
non-circular stochastic compiler audit.

## The three largest gaps to the PRD vision

| Rank | Gap | Current evidence | `.589` response |
|---|---|---|---|
| 1 | **FR12 has exact authority but no reliable proof transport from current local SOTA models.** | Exact stream and checkers work, but a byte-envelope boundary made 216/216 proposals look malformed. No current result separates transport loss, invalid references, and false but parseable proofs. | Repair and replay the output boundary without regenerating. Then compare one-shot, static grammar, and draft-conditioned environment-indexed grammar. Cold-audit exact validity, semantic accuracy, support, and leakage before diagnostic energy or repair. |
| 2 | **FR11 memory exists as storage, not as continuous self-learning.** | The `.588` fixture passed, but the prospective run made no commits or rejects and found no support-preserving gain. It did not exercise a learning loop. | Build a non-saturating stream with exact-admissible opportunities. Compare no memory, detailed trajectories, and abstract procedural constraint memory under fixed storage, retrieval, and context budgets. Audit hard-case negative transfer, forgetting, poison, restart, and rollback. |
| 3 | **Live and hardware-facing evidence still breaks at the execution boundary.** | ARC tool transport works but the paired A/B was blocked by VRAM contention. The stochastic compiler result is simulator-only and circularly classified. | Use the existing task-owned lease on the least-used eligible GPU, prove full load and teardown, and rerun the live A/B only after that receipt. Recompute compiler fidelity with an independently implemented reference and correct circularity semantics. |

## Research deltas adopted in this milestone

- **Decode-Time Grammars (`2607.18357`)** supplies an environment-indexed grammar. The current CNF
  defines the only valid variable names, clause IDs, and values. Prefix-generated declarations can
  tighten later holes. This is materially different from the retired schema-reprompt route.
- **Draft-Conditioned Constrained Decoding (`2603.03305v2`)** separates semantic planning from
  structural rendering. `.589` measures whether draft conditioning reduces the projection tax of a
  hard grammar without merely increasing parseability.
- **When Continual Learning Moves to Memory (`2604.27003`)** shifts the research question from
  storage to representation and retrieval. `.589` compares detailed traces against abstract
  procedural lessons under the same finite capacity and measures harm on hard cases.
- **Memoir (`2607.20792`)** supplies a negative control. Live inference remains read-only. Exact-
  approved memory changes happen only between episodes. No branch writes memory while it reads it.
- **Solver-Hard Is Not Model-Hard (`2607.17047`)** motivates matched surface forms, exact family
  strata, and proof-preserving relabeling. Solver conflicts remain metadata and never stand in for
  model difficulty.
- **Extropic's summer 2026 update** keeps physical TSU claims out of scope. Z1 early access remains
  targeted for 2027. `.589` asks only whether the existing compiler result survives an independent
  local reference.
- **Kona 1.0** remains architecture evidence for global editable trace energies. It exposes no
  weights or local runner, so it is not an executable baseline.

## vNEXT architecture

```text
                           EXACT AUTHORITY BOUNDARY
                   learned scores propose; exact checks certify

 Local SOTA GGUF ──► output-text boundary ──► semantic draft
                           │                       │
                           │ lossless reparse      ▼
                           └──────────────► environment-indexed grammar
                                                  │
                              typed symbols only  ▼
                                           proof candidate
                                                  │
                      dual encoders ───────► exact checker
                              │                   │
                              ▼                   ▼
                    diagnostic energy ──► prefix backtracking
                                                  │
                                                  └──► exact recheck

 Chronological event ──► read-only memory snapshot ──► proposal + exact result
        ▲                                                       │
        │             fixed capacity and retrieval budget       ▼
        └──── next episode ◄── atomic procedural commit ◄── admission gate
                               │
                               └── restart / poison / rollback audit

 Task-owned GPU lease ──► full 32K load receipt ──► live ARC object fetch A/B
                                 │
                                 └── unload and VRAM-release receipt

 Typed stochastic program ──► factor compiler under test ──► trajectory samples
           │                                                      │
           └──── independent enumerator and sampler audit ◄────────┘
```

The exact checker may label rows and certify final candidates. It may not enter the feature vector
of the diagnostic energy or procedural memory query. The current-row label, answer key, exact-valid
bit, solver-work counters, and equivalent proxies remain prohibited features. ARC remains on the
production `E3AgentPolicy` / `make_carnot_agent` path. No game source, offline ground-truth BFS,
hand-built per-game adapter, or game-level solve claim is introduced.

## Phase 1: Recover and harden proof transport

### Exp 6755: Lossless GGUF output boundary and 216-row reparse

Prove the byte-envelope failure from the frozen Exp6745 rows. Add one explicit output-text boundary
that accepts bytes or text, decodes bytes once, and rejects ambiguous repr coercion. Reparse every
existing row through both symbolic encoders and the exact checker. Do not call an LLM. Preserve the
original text and hash beside the normalized text and hash. Report pre/post diagnosis for every row,
the exact-valid count, invalid-symbol and invalid-domain counts, and the number of rows that an
environment-indexed grammar can address.

**Deliverable:** `results/experiment_6755_lossless_gguf_output_reparse.json`

### Exp 6756: Environment-indexed proof grammar fixture

Implement a bounded decode-time grammar for the certificate DSL. Instantiate valid `xN` variables,
`cN` clause IDs, binary values, uniqueness state, and prefix-dependent remaining slots from each
CNF environment. Add a draft-conditioned rendering interface, but do not run the SOTA panel yet.
Test static CFG, environment-indexed, and draft-conditioned paths on exact fixtures and adversarial
ghost references. The fixture passes only if every emitted string parses, no out-of-environment
reference is reachable, valid exact certificates remain representable, and the runtime mask path is
actually invoked.

**Deliverable:** `results/experiment_6756_environment_indexed_proof_grammar_fixture.json`

### Exp 6757: Three-model DCCD environment-grammar A/B

On a frozen hardness- and family-stratified subset, run Qwen3.6-35B-A3B, Gemma-4-31B, and
Gemma-4-26B-A4B sequentially. Compare three matched arms: repaired one-shot output, static grammar,
and draft-conditioned environment-indexed grammar. Match instance, seed, maximum total generated
tokens, context, and exact-check budget. Emit every model-instance-arm row. Measure transport
validity, exact certificate validity, semantic correctness, abstention, invalid references,
latency, and generated tokens. Positive credit requires exact-valid improvement, not parseability
alone.

**Deliverable:** `results/experiment_6757_dccd_environment_grammar_ab.json`

### Exp 6758: Independent proof-transport audit

Cold-recompute Exp6757 with an independently implemented parser and exact checker. Verify that the
grammar runtime was invoked, that no answer or current-row exact feature entered the prompt or mask,
and that budget matching held. Measure paired exact-valid deltas, semantic error, abstention,
support contraction, and proof-preserving relabeling consistency. Open the diagnostic branch only
when each held family has at least two diagnosis classes and the panel has at least 24 parseable
exact-invalid rows.

**Deliverable:** `results/experiment_6758_proof_transport_independent_audit.json`

## Phase 2: Diagnostic energy and exact repair

### Exp 6759: Held-family oracle-distinct diagnostic energy v2

Train a compact structural energy on the audited proof rows. Compare dual-encoding structure,
single-encoding structure, and an undifferentiated scalar baseline on family-disjoint tests and
proof-preserving relabelings. The model may use prefix state, grammar rejection location, encoder
disagreement, and structural counts. It may not use the answer, label, exact-valid bit, exact checker
trace, solver conflicts, or current-row repair outcome. The repair branch opens only if held-family
reasoning-error AUROC is at least 0.65 and the leakage audit finds zero prohibited features.

**Deliverable:** `results/experiment_6759_oracle_distinct_diagnostic_energy_v2.json`

### Exp 6760: Diagnostic prefix-backtracking repair A/B

On a frozen set of at least 24 parseable exact-invalid candidates, compare no repair, matched full
regeneration, and diagnostic prefix backtracking. Pair model, instance, seed, prompt, original
candidate, total token budget, and exact-verifier budget. Use Qwen3.6-35B-A3B and Gemma-4-31B as
headline models. The learned energy selects a prefix; exact authority only checks the final result.
Positive credit requires the paired lower confidence bound over full regeneration to exceed zero
without a higher harmful-flip rate or support contraction.

**Deliverable:** `results/experiment_6760_prefix_backtracking_repair_ab.json`

## Phase 3: Continuous self-learning through procedural memory

### Exp 6761: Capacity-controlled procedural memory stream

Build a chronological, non-saturating constraint stream with known reusable procedures, detailed
trajectory counterparts, naive distractors, held families, hard cases, poison candidates, and six
preregistered orders. Freeze exact admissibility opportunities so the experiment cannot finish with
zero eligible commits. Keep active episodes read-only. Each between-episode transaction records a
parent hash, evidence hash, representation type, scope, TTL, admission reason, inverse patch, and
restart receipt. The fixture must prove nonzero accept and reject opportunities before evaluation.

**Deliverable:** `results/experiment_6761_procedural_memory_stream.json`

### Exp 6762: Procedural versus trace memory prospective A/B

Compare frozen no-memory, detailed trajectory memory, and abstract procedural constraint memory on
the same six chronological orders. Fix storage bytes, top-k retrieval, context tokens, update
opportunities, and exact authority across memory arms. Use Qwen3.6-35B-A3B for acquisition and both
Qwen3.6 and Gemma-4-31B for held-out transfer. Record prequential exact yield, hard-case yield,
best@k and effective support, actual retrieval, action influence, commits, rejects, negative
transfer, forgetting, token cost, restarts, and rollbacks. No weights change.

**Deliverable:** `results/experiment_6762_procedural_vs_trace_csl_ab.json`

### Exp 6763: Cold hard-case, forgetting, and poison audit

Recompute Exp6762 from raw rows in a fresh process. Verify chronological isolation, order-level
intervals, capacity equality, actual memory use, commit hashes, future-evidence denial, restart from
each boundary, poison rejection, byte-exact rollback, support preservation, and hard-case
performance. Positive credit requires a procedural-memory order-level lower bound above zero, no
anchor forgetting, no hard-case regression beyond the preregistered margin, zero admitted poison,
and nonzero exact commits and rejects.

**Deliverable:** `results/experiment_6763_csl_hard_case_forgetting_audit.json`

## Phase 4: Live-path and hardware-facing evidence

### Exp 6764: Exclusive full-load ARC preflight

Use the existing receipt-scoped `GpuLease`. Inspect both RTX 3090 devices and acquire the least-used
eligible device without killing or preempting unrelated work. In the leased subprocess, set the
32K context, load the immutable Qwen3.8 ARC generator, run one bounded production selfparse tool
call, unload, release, and prove VRAM recovery. Run a Qwen3.6-35B-A3B transport canary under the same
lease policy. This is an admission and teardown receipt, not an ARC quality result.

**Deliverable:** `results/experiment_6764_arc_exclusive_load_preflight.json`

### Exp 6765: Live object-table fetch-on-demand A/B v2

Rerun the frozen 20-game paired comparison only after Exp6764 proves an exclusive full load. Compare
the default inline object table against table-absent plus production `find_objects` fetch. Keep the
immutable Qwen3.8 generator, 32K context, games, seeds, budgets, and public agent route fixed. Use
Qwen3.6 only as a non-headline transport canary. Measure prompt tokens, tool calls, useful fetch
rate, transition utility, and `change_fidelity`. Adoption requires positive realized token savings
and non-inferiority within the frozen within-arm noise floor. This makes no game-level solve claim.

**Deliverable:** `results/experiment_6765_object_table_fetch_ab_v2.json`

### Exp 6766: Independent Thermalizer trajectory audit

Cold-recompute the Exp6751 factor and trajectory claims with a separately implemented exact
enumerator and direct sampler. Verify topology, precision, normalization, seed, and compiler
provenance. Compare independent factor fitting, context matching, and trajectory refinement at
depths 1, 2, 4, and 8. The compiler under test must not share the evaluator implementation. Report
conditional KL and trajectory total variation per factor and seed. Classify circular evidence
correctly. This remains simulator-only and makes no speed, power, FPGA, X0, or Z1 claim.

**Deliverable:** `results/experiment_6766_thermalizer_independent_trajectory_audit.json`

### Exp 6767: V589 branch disposition and PRD gap update

Read every available milestone artifact, including gate-blocked branches. Recompute headlines from
rows, run adversarial and row-consistency checks, classify each branch, and state which of the three
PRD gaps narrowed. Keep proof transport, diagnostic repair, continuous memory, ARC, and stochastic
portability separate. The capstone is ungated and may not convert missing evidence into success.

**Deliverable:** `results/experiment_6767_v589_branch_disposition.json`

## Dependency graph

```text
Exp6755 lossless output reparse
  └── Exp6756 environment-indexed grammar fixture
        └── Exp6757 three-model grammar A/B
              └── Exp6758 independent transport audit
                    └── Exp6759 diagnostic energy v2
                          └── Exp6760 prefix-backtracking A/B

Exp6761 procedural memory stream
  └── Exp6762 procedural-versus-trace CSL A/B
        └── Exp6763 hard-case/forgetting/poison audit

Exp6764 exclusive full-load ARC preflight
  └── Exp6765 live object-table A/B v2

Exp6766 independent Thermalizer audit       (independent)

Exp6760 ─┐
Exp6763 ─┼──► Exp6767 branch disposition    (ungated)
Exp6765 ─┤
Exp6766 ─┘
```

Structured conductor gates use fields emitted verbatim by their upstream prompts:

- `transport_reparse_ready`
- `environment_grammar_targetable_rows`
- `dynamic_proof_grammar_ready`
- `proof_transport_ab_completed`
- `proof_transport_audit_ready`
- `diagnostic_panel_ready`
- `heldout_reasoning_error_auroc`
- `oracle_leakage_detected`
- `procedural_memory_stream_ready`
- `prospective_csl_completed`
- `arc_exclusive_load_ready`

Every blocked task-owned outcome records the failed check and observed value in
`gate_check_summary`. Every task declares the closed `verdict_class` enum. Every comparison emits
one `rows` entry per model, instance, arm, order, event, game, factor, depth, or seed as applicable.

## Local model policy

Every experiment that invokes an LLM names at least one mandated local model in `MODEL_SPECS`:

| Role | Model |
|---|---|
| Flagship MoE | `unsloth/Qwen3.6-35B-A3B-GGUF` |
| Flagship dense | `unsloth/gemma-4-31B-it-GGUF` |
| Middle MoE | `unsloth/gemma-4-26B-A4B-it-GGUF` |
| Immutable scored ARC generator | `unsloth/Qwen3.8-27B-GGUF` |

The proof panel uses all three mandated families as headline models. The CSL comparison uses Qwen
and the dense Gemma. ARC keeps Qwen3.8 as the immutable scored generator and uses Qwen3.6 only for
transport. Legacy `Qwen/Qwen3.5-0.8B` and `google/gemma-4-E4B-it` may appear only as labeled CPU
smoke tests. They cannot satisfy readiness, comparative, or headline gates.

## Hardware requirements

| Work | Required substrate | Expected use | Claim boundary |
|---|---|---|---|
| Exp6755, Exp6756, Exp6758, Exp6759 | CPU, local disk, exact solver toolchain | Lossless replay, grammar fixtures, exact checks, compact energy | No LLM or accelerator claim |
| Exp6757, Exp6760 | Dual RTX 3090 host; cached three mandated GGUFs; llama.cpp CUDA | Sequential model loads with explicit device, VRAM, and runtime-mask receipts | No cross-model speed claim |
| Exp6761, Exp6763 | CPU, local disk, atomic memory store | Stream generation and cold replay | No weight-learning claim |
| Exp6762 | Dual RTX 3090 host; cached Qwen3.6 and Gemma-4-31B GGUFs | Sequential acquisition and transfer panels | Tier-2 memory only; no weight update |
| Exp6764, Exp6765 | One exclusively leased RTX 3090; cached Qwen3.8 and Qwen3.6; 32K context | Full load, selfparse transport, paired live ARC run, teardown | No game-level solve or source-derived claim |
| Exp6766 | CPU/JAX or installed Torx/THRML simulator; exact enumeration | Small factors and trajectories at depth at most 8 | Simulator/compiler fidelity only; no physical TSU |
| Exp6767 | CPU and local artifacts | Row replay and audits | No pooled scientific success claim |

KV260, GateMate, and PolarFire remain continuity tracks. Their reachable state has not changed, so
`.589` schedules no unchanged board probe. Extropic Z1 access remains a 2027 prospect. No task may
infer physical acceleration, power, or energy from simulator timing.

## Execution order and failure isolation

The conductor order is Exp6755 through Exp6767. The order is not a global dependency chain. The
proof, memory, ARC, and stochastic roots are independent. Only scientifically necessary producer-
consumer pairs use `gated_on`. Exp6767 is deliberately ungated and must preserve every blocked or
missing branch.

Fresh IDs are used throughout. Every recovered scope declares the exact prior verdict, explains
the changed mechanism or newly shipped prerequisite, and sets `retire_if_same_verdict: true`. No
task references a retired upstream ID. The capstone uses the standing 2026-05-29 continuation
override.

The roadmap keeps `{project_root}` and `{date}` placeholders because the conductor contract requires
them. No new task treats unrendered planner placeholders as a scientific blocker. Hardware tasks do
not kill, preempt, or adopt unrelated processes. If an exclusive device is unavailable, Exp6764
writes a blocked artifact and Exp6765 skips before its agent call.

The legacy `audit_roadmap_gates.py` still hardcodes every Codex task to `gpt-5.5`. It therefore
reports routing-only findings for the formulaic tasks that the current operator directive assigns to
`gpt-5.6-sol`. The roadmap schema accepts those routes, the gate cross-reference audit reports zero
field failures, and the exclusion-manifest lint is authoritative for failed-scope activation. This
milestone records the validator mismatch and does not edit the audit tool.

## Exit criteria

Milestone `.589` is complete when all thirteen tasks reach a terminal artifact or honest conductor
gate record and Exp6767 classifies every branch. Scientific success is branch-specific:

- **FR12 transport:** the environment-indexed or draft-conditioned arm improves paired exact-valid
  yield over repaired one-shot output without increased semantic error, prohibited leakage, or
  preregistered support contraction.
- **FR12 repair:** held-family AUROC is at least 0.65 with zero oracle leakage, followed by a prefix-
  backtracking lower confidence bound above zero over full regeneration and no harmful-flip rise.
- **FR11:** procedural memory earns a positive order-level lower bound over both no memory and
  detailed traces, with nonzero commits and rejects, retained support, no anchor or hard-case
  regression, zero admitted poison, durable restart, and byte-exact rollback.
- **ARC:** an exclusive full-load receipt completes, then fetch-on-demand saves tokens while
  remaining non-inferior on `change_fidelity`. No solve claim is needed.
- **Portability:** the independent evaluator reproduces a trajectory-error reduction and the tested
  compiler is distinct from its evaluator. Otherwise the branch remains circular, null, or partial.

## Explicitly deferred

- Weight-updating continual learning, LoRA, RLVR, or foundation-model retraining.
- Adaptive KAN knot insertion or deletion until active Tier-2 memory earns a clean prospective
  result and supplies a non-circular signal.
- EBT- or Kona-scale latent reasoning training and proprietary Kona comparisons.
- External-text Phase-D energy scoring, which remains retired.
- Physical TSU, FPGA, NPU, or board speed and power claims.
- A second schema-only or prompt-only ConstraintIR rerun.
- New ARC level solves, duplicate registry targets, game-source inspection, offline ground-truth
  BFS, or hand-built per-game adapters.
- External publication, model upload, leaderboard submission, or other operator-only action.

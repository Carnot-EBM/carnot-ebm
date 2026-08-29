# Carnot Research Roadmap vNEXT: Handoff-Preserved Verification and Transactional Learning

**Milestone:** `2026.08.588`
**Created:** 2026-08-29
**Status:** Proposed
**Supersedes:** milestone `2026.08.587`
**Research basis:** `research-program.md`, `_bmad/prd.md`, `_bmad/architecture.md`,
`ops/status.md`, `ops/changelog.md`, `research-complete.yaml`, `research-roadmap.yaml`,
`ops/conductor-log.md`, `research-hardware-wishlist.md`, and the `V588 Planner Refresh` in
`research-references.md`.

## What milestone 2026.08.587 proved

Milestone `.587` ended terminally, but it did not execute a scientific comparison. The design
document described 13 experiments, while the activated manifest contained seven and
`research-roadmap-next.yaml` was absent. Exp6729 recorded
`blocked_v587_activation_contract`. The consequences were mechanical:

- Exp6730, Exp6733, and their retries gate-blocked on `v587_contract_ready=false`.
- Exp6731 and Exp6734 were preempted after their upstream tasks retired.
- Exp6732 and Exp6735 wrote blocked artifacts for missing upstream evidence.
- Exp6736-Exp6741 never entered the active manifest.
- Five logged tasks rounded to the 0.0-minute duration floor, including tasks classified as
  compute-bearing. The milestone lacks task-owned phase and accelerator activity receipts.

This proves a systems fact, not a negative result about diagnostic energy, continuous learning,
ARC object-table fetch, or stochastic compilation. A global activation result was placed at the
root of unrelated branches, and a handoff mismatch then suppressed all science. `.588` repairs the
evidence path in two ways. The planner statically validates the design and next-manifest before
handoff. At runtime, two independent infrastructure tasks audit the activated contract and task
activity, but neither gates unrelated science.

The inherited scientific evidence remains unchanged:

- The 48-row exact certificate corpus has 8/48 exact model successes. Exact validators work, but
  current SOTA proposal transport leaves substantial invalid-output mass.
- Scalar failure labels do not distinguish translation failure from reasoning failure. No held-
  family, oracle-distinct diagnostic energy has yet earned positive credit.
- Recent prospective self-learning branches were blocked or statistically null. No current result
  proves read-only within-episode learning, exact delayed commit, retained support, and durable
  rollback together.
- ARC selfparse transported 20/20 zero-argument `list_transitions` calls, but no code-carrying,
  multi-parameter `find_objects` call has a live receipt. The fetch-on-demand A/B never started.
- Exact stochastic references exist, but installed Torx parity failed its end-to-end check.
  Extropic silicon is not locally reachable.

## The three largest gaps to the PRD vision

| Rank | Gap | Current evidence | `.588` response |
|---|---|---|---|
| 1 | **The research harness does not reliably preserve an executable evidence contract.** | `.587` activated seven of thirteen designed tasks, omitted the staged manifest, weakened several closed-enum declarations, and put all science behind one false root gate. | Preserve prerequisite, authority, fallback, and consequence as typed handoff rows; audit activated doc/manifest parity; add task-owned monotonic phase and accelerator receipts; keep science branches independent of infrastructure outcomes. |
| 2 | **FR12 lacks a non-circular diagnostic-and-repair path on current local SOTA models.** | Exact solvers certify outputs, but direct exact yield is low. Existing learned scores do not cleanly separate malformed output, translation disagreement, and reasoning error on held families. | Build a hardness-controlled exact stream, collect all-three-model dual-encoding proposals, learn a feature-denylised diagnostic energy, and test localized repair against matched full regeneration with exact final authority. |
| 3 | **FR11 and the live-agent bridge do not yet demonstrate safe prospective adaptation.** | Continuous self-learning has not jointly passed transfer, retention, support, poison, restart, and rollback checks. ARC cannot yet fetch the largest prompt block through a live code-carrying tool call. Hardware portability remains simulator-only. | Run transactional read-only-episode self-learning across controlled chronological orders; cold-audit durability; recover live ARC tool transport and object-table A/B; separately measure factor-to-trajectory compiler error without hardware claims. |

## Research deltas adopted in this milestone

- **HarnessLens (`2608.27311`)** motivates behavior-scoped, attributable checks. `.588` has two
  independent infrastructure receipts instead of one global science gate.
- **ABE-Ralph (`2608.26753`)** motivates frozen method contracts, actual-budget receipts,
  per-unit rows, and disqualification for silent substitutions.
- **When “Must” Becomes “Maybe” (`2608.24569`)** motivates a typed handoff tuple containing the
  prerequisite, authority, fallback, and execution consequence.
- **SymDiag (`2608.08786`)** supplies the dual-encoding split between translation disagreement and
  reasoning error.
- **Solver-Hard Is Not Model-Hard (`2607.17047`)** supplies proof-hard/proof-easy controls and
  proof-preserving relabeling without treating solver conflicts as model difficulty.
- **Memoir (`2607.20792`)** and **AgentCL (`2606.02461`)** motivate read-only active episodes,
  between-episode commits, controlled reusable streams, naive streams, held-out transfer, and
  order-replicated evaluation.
- **Thermalizing Stochastic Programs (`2608.01615`)** and the **PSC/Torx framework
  (`2608.01612`)** define the factor-error-to-trajectory-error portability target.
- **PARTAB (`2608.24082`)** remains the ARC evidence-selection lead: fetch bounded, query-linked
  object partitions instead of inlining the entire table.

## vNEXT architecture

```text
                         STATIC HANDOFF BOUNDARY
              design doc ──► next YAML ──► active YAML
                   │       typed binding rows       │
                   └──────── runtime audit only ────┘
                                  (not a science gate)

                         EXACT AUTHORITY BOUNDARY
                         (learned scores never certify)

 Local SOTA GGUF ──► proof-carrying candidate ──► exact certificate checker
       │                         │                           │
       │                 dual independent encoders          │
       │                         │                           │
       └──────────────► structural diagnostic energy ◄──────┘ labels only
                                  │
                                  └──► localized repair ──► exact recheck

 Chronological event ──► read-only memory snapshot ──► proposal + exact check
        ▲                                                    │
        └──── next episode ◄── atomic admitted commit ◄──────┘
                              retention/support/poison gate

 Live ARC observations ──► Qwen3.8 E3AgentPolicy ──► selfparse find_objects
          │                         │                         │
          └──────── same games/seeds paired A/B ◄────────────┘

 Typed stochastic factors ──► factor EBM compiler ──► short trajectories
          │                         │                         │
          └──── exact enumeration: conditional KL + trajectory TV ───────┘
                                  simulator only
```

The exact checker may label training rows and certify final candidates. Its current-row outcome,
answer key, solver-work counters, certificate-validity bit, or equivalent proxy may not enter the
learned energy feature vector. A positive verifier result is therefore oracle-distinct. ARC remains
on the production `E3AgentPolicy` / `make_carnot_agent` path. No game source, exhaustive offline
BFS, hand-built per-game adapter, or new level-solve claim is introduced.

## Phase 1: Handoff integrity, activity receipts, and exact data

### Exp 6742: Activated handoff and binding-contract audit

Audit exactly 13 activated tasks against this document. For every task and gate, preserve the
prerequisite, authority, fallback, and execution consequence; verify unique IDs and deliverables,
closed verdict classes, exact gate-field spelling, prior-failure retirement mechanics, model policy,
and claim boundaries. This task audits the active snapshot after handoff. It does not gate science.

**Deliverable:** `results/experiment_6742_v588_handoff_contract_audit.json`

### Exp 6743: Task-owned phase and accelerator canary

Run a bounded sequential canary on all three mandated GGUF families. Emit monotonic phase markers
for preflight, cache resolution, model load, first token, completed decode, teardown, and artifact
write. Capture CUDA offload, actual device, peak VRAM, tokens, and measured duration. This is an
activity/provenance receipt, not a quality or cross-model speed claim, and it does not gate science.

**Deliverable:** `results/experiment_6743_task_owned_phase_accelerator_canary.json`

### Exp 6744: Hardness-controlled exact certificate stream

Generate a fixed 72-instance stream across expander-Tseitin, ladder-Tseitin, and pigeonhole-anchor
families, size bins, labels, and seeds. Pair instances with proof-preserving symbol relabelings.
Produce exact SAT assignments or independently checkable UNSAT certificates, family-disjoint
train/dev/test splits, and solver-work metadata. Solver conflicts remain metadata, never a model-
difficulty label.

**Deliverable:** `results/experiment_6744_hardness_controlled_certificate_stream.json`

## Phase 2: Diagnostic energy and exact-authority repair

### Exp 6745: Three-family SOTA dual-encoding proposal corpus

Run Qwen3.6-35B-A3B, Gemma-4-31B, and Gemma-4-26B-A4B sequentially on the frozen stream. Require a
small proof-carrying DSL. Retain every success, malformed output, timeout, and abstention. Parse each
response through two independently implemented symbolic encoders, exact-check the result, and emit a
closed diagnosis: exact-valid, malformed-certificate, translation-disagreement, reasoning-error, or
abstention. Corpus readiness means attributable row completeness, not high accuracy.

**Deliverable:** `results/experiment_6745_sota_dual_encoding_proposal_corpus.json`

### Exp 6746: Held-family oracle-distinct diagnostic energy

Train a compact structural energy on pre-oracle features. Compare dual-encoding features against
single-encoding and undifferentiated-scalar baselines on family-disjoint tests and relabeled pairs.
Explicitly deny current-row exact outcomes, labels, answer keys, solver conflicts, and validity bits
from model inputs. The repair branch opens only if reasoning-error AUROC is at least 0.65 and the
leakage audit finds zero prohibited features.

**Deliverable:** `results/experiment_6746_oracle_distinct_diagnostic_energy.json`

### Exp 6747: Diagnostic-energy localized repair A/B

On a frozen set of at least 24 exact-invalid but parseable proposals, compare no repair, full
regeneration, and diagnosis-localized suffix/backtracking repair. Pair model, row, seed, prompt,
candidate, token, and exact-verifier budgets. Positive credit requires the paired lower confidence
bound for localized repair over full regeneration to exceed zero without a higher harmful-flip rate.
The exact checker remains final authority.

**Deliverable:** `results/experiment_6747_diagnostic_energy_localized_repair_ab.json`

## Phase 3: Transactional continuous self-learning

### Exp 6748: Read-only episode and atomic commit fixture

Build a controlled chronological constraint stream with reusable repair structure, naive distractor
events, held-out families, and six preregistered orders. The active episode receives a read-only
snapshot. Only exact-certified, future-eligible records may commit between episodes. Each commit has
a parent hash, evidence hash, scope, TTL, admission reason, inverse patch, and atomic restart receipt.
Duplicate, contradiction, stale, provenance-loss, poison, crash, and byte-exact rollback tests must
pass before live evaluation.

**Deliverable:** `results/experiment_6748_transactional_constraint_memory_fixture.json`

### Exp 6749: Prospective support-preserving self-learning A/B

Compare frozen no-memory against transactional memory on the six chronological orders. Use
Qwen3.6-35B-A3B for acquisition and both Qwen3.6 and Gemma-4-31B for held-out transfer. Measure
prequential exact yield, best@k and effective rewardable support, joint correct-and-constraint-
following support, retention anchors, negative transfer, token cost, commits, rejects, and rollbacks.
No weights change and no memory write occurs inside an active episode.

**Deliverable:** `results/experiment_6749_prospective_support_preserving_csl_ab.json`

### Exp 6750: Cold durability, support, and poison audit

Recompute the prospective result from raw rows in a fresh process. Verify chronological isolation,
order-level intervals, commit hashes, future-evidence denial, restart from every boundary, byte-exact
rollback, poison rejection, support preservation, and retained anchor performance. Positive credit
requires an order-level lower confidence bound above zero, zero admitted poison, no anchor forgetting,
and no preregistered support contraction.

**Deliverable:** `results/experiment_6750_csl_durability_support_poison_audit.json`

## Phase 4: Portability, live ARC recovery, and synthesis

### Exp 6751: Thermalizers-style factor-to-trajectory compiler fidelity

Implement bounded typed binary/categorical stochastic kernels and compile them to sparse EBM/Ising
factors. Compare independent factor fitting, context matching, and trajectory-level refinement at
depths 1, 2, 4, and 8. Exact enumeration supplies per-factor conditional KL and trajectory total
variation. Record topology, precision, seeds, compiler provenance, and optional official-Torx
conformance. This is simulator-only hardware preparation and makes no speed, power, FPGA, X0, or Z1
claim.

**Deliverable:** `results/experiment_6751_thermalizer_factor_trajectory_fidelity.json`

### Exp 6752: Owned 32K code-carrying ARC tool preflight

Set `CARNOT_ARC_INDUCE_N_CTX=32768` inside the task-owned subprocess. On the immutable scored
Qwen3.8-27B generator and a Qwen3.6-35B-A3B transport canary, prove CUDA admission and exercise a
multi-parameter `find_objects` request carrying a bounded predicate through production XML parse,
dispatch, response bounding, and transcript capture. This measures transport only and claims no
game solve.

**Deliverable:** `results/experiment_6752_arc_code_carrying_tool_preflight.json`

### Exp 6753: Live object-table fetch-on-demand A/B

On the same 20 held games and seeds used by the 2026-08-01 object-perception study, compare the
default inline object table against table-absent plus production `find_objects` fetch. Keep Qwen3.8,
context, budgets, public agent route, and seeds fixed. Measure prompt tokens, tool calls, useful fetch
rate, transition utility, and `change_fidelity`. Adoption requires positive realized token savings and
non-inferiority within the preregistered within-arm noise floor. This is live-path quality evidence,
not a level solve.

**Deliverable:** `results/experiment_6753_object_table_fetch_on_demand_ab.json`

### Exp 6754: V588 branch disposition and PRD gap update

Read every available milestone artifact, including blocked or missing branches. Recompute headline
claims from rows, run adversarial and row-consistency checks, classify each branch, and state which of
the three PRD gaps narrowed. The capstone is deliberately ungated. It may not pool unrelated branches
or convert missing evidence into success.

**Deliverable:** `results/experiment_6754_v588_branch_disposition.json`

## Dependency graph

```text
Exp6742 handoff contract audit       (independent; not a science gate)
Exp6743 phase/accelerator canary     (independent; not a science gate)

Exp6744 exact certificate stream
  └── Exp6745 SOTA dual-encoding corpus
        └── Exp6746 diagnostic energy
              └── Exp6747 localized repair A/B

Exp6748 transactional memory fixture
  └── Exp6749 prospective CSL A/B
        └── Exp6750 cold durability/poison audit

Exp6751 compiler fidelity            (independent)

Exp6752 ARC tool preflight
  └── Exp6753 object-table A/B

Exp6742 ─┐
Exp6743 ─┤
Exp6747 ─┤
Exp6750 ─┼──► Exp6754 branch disposition (ungated)
Exp6751 ─┤
Exp6753 ─┘
```

Structured conductor gates use fields emitted verbatim by their upstream prompts:

- `hardness_stream_ready`
- `dual_encoding_corpus_ready`
- `heldout_reasoning_error_auroc`
- `oracle_leakage_detected`
- `transaction_memory_ready`
- `prospective_csl_completed`
- `arc_context_tool_preflight_ready`

Every blocked task-owned outcome records the failed check and observed value in
`gate_check_summary`. Every task declares the closed `verdict_class` enum. Every comparison emits
one `rows` entry per model, instance, order, arm, factor, trajectory, game, or seed as applicable.

## Local model policy

Every experiment that invokes an LLM names at least one mandated local model in `MODEL_SPECS`:

| Role | Model |
|---|---|
| Flagship MoE | `unsloth/Qwen3.6-35B-A3B-GGUF` |
| Flagship dense | `unsloth/gemma-4-31B-it-GGUF` |
| Middle MoE | `unsloth/gemma-4-26B-A4B-it-GGUF` |
| Immutable scored ARC generator | `unsloth/Qwen3.8-27B-GGUF` |

The three mandated families are headline models for the proof and self-learning branches. Legacy
`Qwen/Qwen3.5-0.8B` and `google/gemma-4-E4B-it` may appear only as explicitly labeled CPU smoke tests.
They cannot satisfy model readiness or headline gates. ARC keeps Qwen3.8 as the production generator;
Qwen3.6 is a transport canary and does not replace or pool with the scored model.

## Hardware requirements

| Work | Required substrate | Expected use | Claim boundary |
|---|---|---|---|
| Exp6742, Exp6744, Exp6746, Exp6748, Exp6750 | CPU, local disk, exact solver toolchain | 32-64 GB RAM; bounded parallel exact checks | No GPU or hardware claim |
| Exp6743, Exp6745, Exp6747, Exp6749 | Dual RTX 3090 host; cached mandated GGUFs; llama.cpp CUDA offload | Sequential model loads, explicit device/VRAM/phase receipts | No cross-model speed claim from the canary |
| Exp6751 | CPU/JAX or installed Torx simulator; exact enumeration | Small typed factors and depth <=8 circuits | Simulator/compiler fidelity only; no physical TSU |
| Exp6752, Exp6753 | Dual RTX 3090 host; cached Qwen3.8 and Qwen3.6; 32K context | One model at a time; bounded live ARC runs | No new solve, source, BFS, or per-game adapter |
| Exp6754 | CPU and local artifacts | Row replay and audits | No external publication |

KV260, GateMate, and PolarFire remain opportunistic continuity tracks. Their state has not changed,
so `.588` schedules no unchanged board probe. Extropic Z1 access remains a 2027 prospect. No task
may infer physical acceleration, power, or energy from simulator timing.

## Execution order and failure isolation

The conductor order is Exp6742 through Exp6754. The order is not a global dependency chain. Phase 1
infrastructure tasks may fail honestly without suppressing the verifier, self-learning, compiler, or
ARC roots. Only scientifically necessary producer-consumer pairs use `gated_on`. The capstone is
ungated and must preserve every blocked or missing branch.

Fresh IDs are used throughout. Each recovered scope declares the exact prior verdict, explains the
changed mechanism or prerequisite, and sets `retire_if_same_verdict: true`. No task references a
retired upstream ID. The capstone carries the standing 2026-05-29 continuation override.

The current `audit_roadmap_gates.py` still hardcodes Codex tasks to `gpt-5.5`. It reports a
routing-only incompatibility for the formulaic tasks that this roadmap intentionally assigns to
`gpt-5.6-sol` under the operator's current routing directive. Roadmap schema validation, gate-field
cross-reference checks, prompt checks, and exclusion-manifest lint are authoritative for this
handoff. Exp6742 records the legacy-validator mismatch as evidence and does not edit the audit tool.

## Exit criteria

Milestone `.588` is complete when every task reaches a terminal artifact or an honest conductor gate
record and Exp6754 classifies all branches. Scientific success is branch-specific:

- **FR12:** held-family AUROC >=0.65 with zero leakage, followed by a localized-repair paired lower
  confidence bound above zero and no harmful-flip increase.
- **FR11:** prospective order-level lower bound above zero with retained support, no anchor
  forgetting, zero admitted poison, durable restart, and byte-exact rollback.
- **ARC:** code-carrying transport completes and fetch-on-demand saves tokens while remaining
  non-inferior on `change_fidelity`; no solve claim is needed.
- **Portability:** context or trajectory refinement reduces accumulated trajectory error against
  independent factor fitting on exact-enumerable circuits; this remains simulator evidence.
- **Harness:** active doc/manifest contract is preserved and real model work has monotonic phase and
  accelerator receipts. Harness success cannot substitute for a scientific result.

## Explicitly deferred

- Weight-updating continual learning, LoRA, RLVR, or foundation-model retraining.
- Adaptive KAN knot insertion/deletion. It follows a clean transactional-memory result and a
  non-circular learned energy signal.
- EBT/Kona-scale latent reasoning training or proprietary Kona comparison.
- Physical TSU, FPGA, NPU, or board speed/power claims.
- PSC integration into llama.cpp or grammar-mask speed claims without a live invoked runtime path.
- New ARC level solves, duplicate registry targets, game-source inspection, offline ground-truth BFS,
  or hand-built per-game adapters.
- External publication, model upload, leaderboard submission, or other operator-only action.

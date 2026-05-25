# Research Roadmap vNEXT - Milestone 2026.05.287

**Title:** Verifier-Gain Recovery + Soundness-Bounded FR-11 + Blocker Reconciliation
**Created:** 2026-05-25
**Status:** Planned
**Supersedes:** 2026.05.286 "Retire Gate-Rerun Blockers + Solver-Grounded Verification + FR-11 Promotion Boundary"
**Execution queue:** `research-roadmap-next.yaml`

## What 2026.05.286 Proved

Milestone `.286` completed all planned tasks, but the authoritative capstone
kept the research program below paper-ready status. The authority artifact is
`results/experiment_3066_capstone_v286.json`:

- `capstone_ready=true`
- `paper_ready=false`
- `repair_claim_status=bounded_and_gated_skipped`
- `solver_grounding_status=flagged_solver_grounded_no_gain`
- `fr11_self_learning_status=controller_only_delayed_regression_ready_flagged`
- `kan_pwa_status=bounded_controller_anchor_audit_not_promoted`
- `gatemate_status=blocked_no_rerun_operator_actions_required`
- `ssqa_status=gated_skipped_host_visible_smoke_missing`
- `publication_blocker_count=33`
- `next_milestone_recommendation=2026.05.287_verifier_gain_recovery_and_blocker_reconciliation`

The positive result is that `.286` stopped several bad loops. It retired
unsupported repair headline wording, built a de-tautology protocol, preserved
formal SMT fallback, added FR-11 trace and delayed-regression artifacts, kept
KAN/PWA bounded, and converted GateMate/SSQA into no-rerun ledgers until
operator evidence exists.

The hard negative result is that the core paper claim still lacks a measured
gain. `exp3057` produced a local SOTA solution-verifier panel, but
`verifier_gain_delta=-0.125`, `false_positive_rate=0.0`, and
`false_negative_rate=1.0`. `exp3058` showed LLM-guided SMT proposals matching
solver-only success with no measurable improvement. The gated repair rerun
`exp3059` correctly did not run because the verifier-gain gate failed.
Matrix v20 also surfaced blocker hygiene problems, including artifact aliasing
around `exp3059`, remaining missing-source rows, and 33 publication blockers.

| Area | `.286` result | `.287` consequence |
| --- | --- | --- |
| Verifier gain | Local SOTA verifier panel was calibrated but negative | Diagnose confidence/entropy/abstention before another repair gate |
| Solver guidance | AquaForte-style SMT preserved fallback but showed no lift | Add MCS/refinement feedback and keep exact solvers authoritative |
| Repair | De-tautology protocol exists; gated rerun skipped | Try grammar-constrained repair only after positive verifier-gain evidence |
| FR-11 | Controller-side delayed regression worked but remained flagged | Add explicit soundness/completeness mistake budgets before promotion |
| KAN/EBM theory | KAN/PWA stayed controller-anchor only | Audit EBT/ARM-EBM adapter feasibility without implementation claims |
| Hardware | GateMate/SSQA no-rerun ledgers are correct | Refresh operator-action ledgers only; no flash/timing/speedup reruns |
| Matrix/capstone | Matrix v20 ready, capstone not paper-ready | Normalize aliases/blockers and build matrix v21 as the capstone authority |

## Three Biggest Gaps To PRD Vision

1. **Verifier-gain and abstention gap.** The PRD requires verifiable reasoning
   gains, but the latest local SOTA verifier made selection worse and missed all
   wrong cases. `.287` must treat confidence, first-token entropy, abstention,
   and formal correction feedback as calibration signals before repair can run.

2. **Continuous self-learning soundness gap.** FR-11 has controller-side delayed
   regression evidence, but not a soundness/completeness accounting model. `.287`
   must define mistake budgets, delayed-regression windows, no-feedback controls,
   shuffled-feedback controls, and rollback criteria before claiming any
   continuous self-learning improvement.

3. **Claim-blocker reconciliation gap.** The framework is improving its
   honesty, but matrix v20 still has alias, missing, bounded, blocked, and
   projection-only rows that prevent a paper-ready capstone. `.287` must reduce
   ambiguity by evidence, retirement, or explicit bounded status, not by wording.

## New Research Integrated

The post-`.286` sweep was added to `research-references.md` before this
milestone was designed. The findings that materially shape `.287` are:

| Finding | Source | Milestone use |
| --- | --- | --- |
| First-token entropy can flag hallucination risk | arXiv:2605.05166 / Hugging Face Papers | Add first-token confidence and abstention to the verifier-gain recovery panel |
| Lyapunov probes expose hallucination sensitivity | arXiv:2603.06081 | Use as a diagnostic design reference; no probe implementation claim without hidden-state access |
| HALT-RAG separates abstention, acceptance, and rejection | arXiv:2509.07475 / Hugging Face Papers | Require separate acceptance precision, rejection recall, and abstention coverage fields |
| Energy-guided decoding can suppress hallucinated objects | arXiv:2507.07731 | Treat energy-guided decoding as a repair/selection design pattern, not a multimodal local claim |
| VERGE combines formal refinement with MCS-style feedback | arXiv:2601.20055 / Hugging Face Papers | Add correction-subset SMT/SAT feedback instead of only solver success/failure |
| Online CoT verifier learnability needs mistake bounds | arXiv:2603.03538 | Define FR-11 soundness and completeness mistake budgets |
| EBT and ARM-EBM provide relevant theory but not local integration | arXiv:2507.02092, arXiv:2512.15605, `alexiglad/EBT` | Add a feasibility audit before any adapter implementation |
| LLGuidance supports grammar-constrained decoding | `guidance-ai/llguidance` | Add syntax-constrained repair candidate emission before a repair micro-panel |
| Ising/thermodynamic hardware remains promising but externally scoped | NVIDIA Ising-Decoding, Extropic writing, arXiv:2601.04358 | Keep hardware as future architecture context until local host-visible evidence exists |
| Kona and thermodynamic accelerators are architecture signals only | logicalintelligence.com, extropic.ai/writing | Compare architecture boundaries without borrowing external performance claims |

## Architecture Snapshot

```text
                           +----------------------------------+
                           | Mandated local SOTA GGUF models  |
                           | Qwen3.6-35B-A3B, Gemma-4-31B,   |
                           | Gemma-4-26B-A4B                 |
                           +----------------+-----------------+
                                            |
                                            v
 +----------------------+      +----------------------------+      +----------------------+
 | exp3069 failure      | ---> | exp3070 confidence, first- | ---> | exp3072 verifier    |
 | autopsy and protocol |      | token entropy, abstention  |      | calibration v2      |
 +----------------------+      +----------------------------+      +----------+-----------+
              |                              |                               |
              |                              v                               v
              |                 +----------------------------+      +----------------------+
              |                 | exp3071 VERGE/MCS formal   | ---> | exp3075 repair only |
              |                 | correction feedback        |      | if gates pass       |
              |                 +----------------------------+      +----------------------+
              |
              v
 +----------------------+      +----------------------------+      +----------------------+
 | exp3073 EBT/ARM-EBM  | ---> | future adapter backlog,    |      | no implementation   |
 | feasibility audit    |      | no local claim             |      | claim in .287       |
 +----------------------+      +----------------------------+      +----------------------+

 +----------------------+      +----------------------------+      +----------------------+
 | exp3076 FR-11        | ---> | exp3077 online self-       | ---> | bounded promotion   |
 | mistake-budget spec  |      | learning pilot with exact  |      | only if controls    |
 |                      |      | feedback and rollback      |      | pass                |
 +----------------------+      +----------------------------+      +----------------------+

 +----------------------+      +----------------------------+      +----------------------+
 | exp3078 GateMate/    | ---> | no-rerun operator-action   | ---> | no speedup or       |
 | SSQA refresh         |      | ledger                     |      | hardware claim      |
 +----------------------+      +----------------------------+      +----------------------+

                                            |
                                            v
                           +----------------------------------+
                           | exp3079 matrix v21               |
                           | exp3080 capstone .287            |
                           +----------------------------------+
```

## Phase Plan

### Phase A - Archive, Matrix Hygiene, and Failure Autopsy

Tasks: `exp3067`-`exp3069`

- Archive `.286` and pre-stage `.287` without modifying the active roadmap.
- Normalize matrix v20 source-artifact aliases, especially the `exp3059`
  `_v1` mismatch, and separate missing artifacts from bounded artifacts.
- Turn the negative verifier and no-lift SMT results into a calibration
  protocol with explicit abstention and disqualifier fields.

Exit condition: `.287` starts from a clean blocker ledger and a concrete
verifier-gain recovery protocol.

### Phase B - Verifier-Gain Recovery and EBM Feasibility

Tasks: `exp3070`-`exp3073`

- Run a local SOTA first-token entropy and abstention panel using the mandated
  GGUF models.
- Add VERGE/MCS-style formal correction feedback while preserving solver
  authority.
- Run a gated verifier calibration v2 only after the confidence and formal
  correction artifacts exist.
- Audit EBT/ARM-EBM adapter feasibility without claiming implementation or
  benchmark performance.

Exit condition: Carnot either has positive verifier-gain evidence under exact
ground truth, or the repair branch remains mechanically blocked.

### Phase C - Grammar-Constrained Repair and Soundness-Bounded FR-11

Tasks: `exp3074`-`exp3077`

- Define a grammar-constrained repair protocol based on LLGuidance-style
  syntax control and AprAD-style intent preservation.
- Run a tiny SOTA repair micro-panel only if verifier-gain recovery is positive.
- Define FR-11 soundness/completeness mistake budgets.
- Run the mandatory continuous self-learning pilot with exact feedback,
  delayed-regression checks, no-feedback controls, shuffled-feedback controls,
  and rollback.

Exit condition: repair is either cleanly improved under strict gates or skipped,
and FR-11 has a soundness-bounded controller-side result.

### Phase D - Hardware Boundaries, Matrix v21, and Capstone

Tasks: `exp3078`-`exp3080`

- Refresh GateMate and SSQA operator-action ledgers without rerunning hardware
  or making speedup claims.
- Build cross-corpus matrix v21 from `.286` and `.287` artifacts.
- Write the `.287` capstone, preserving `paper_ready=false` unless every
  required matrix row is clean.

Exit condition: matrix v21 is the authority for paper readiness, and the next
milestone inherits fewer ambiguous blockers.

## Dependency Graph

```text
exp3067 archive
  |
  v
exp3068 matrix v20 alias/blocker normalization
  |
  v
exp3069 solver-verifier failure autopsy

exp3069
  |
  +--> exp3070 first-token abstention SOTA panel
  |        |
  |        v
  |    exp3072 verifier calibration v2
  |
  +--> exp3071 VERGE/MCS SMT correction pilot
           |
           v
       exp3072 verifier calibration v2

exp3073 EBT/ARM-EBM feasibility audit

exp3074 grammar-constrained repair protocol
  |
  v
exp3075 repair micro-panel

exp3072(verifier_gain_delta > 0) + exp3074(protocol ready)
  |
  v
exp3075 repair micro-panel

exp3076 FR-11 soundness/completeness budget
  |
  v
exp3077 FR-11 soundness-bounded online self-learning pilot

exp3078 GateMate/SSQA no-rerun refresh

exp3068, exp3072, exp3073, exp3075, exp3077, exp3078
  |
  v
exp3079 matrix v21
  |
  v
exp3080 capstone .287
```

Structured conductor gates:

- `exp3072` gates on `exp3070.first_token_panel_ready == true` and
  `exp3071.mcs_feedback_ready == true`.
- `exp3075` gates on `exp3074.grammar_constrained_repair_protocol_ready == true`
  and `exp3072.verifier_gain_delta > 0.0`.
- `exp3077` gates on `exp3076.soundness_completeness_budget_ready == true`.
- `exp3080` gates on `exp3079.matrix_v21_ready == true`.

## Hardware Requirements

| Task group | Requirement | Claim boundary |
| --- | --- | --- |
| `exp3070`, `exp3071`, `exp3072`, `exp3075` | Local GPU capable of mandated SOTA GGUF inference; dual RTX 3090 preferred | These tasks may claim local model transcript evidence only if model ID, quantization, prompt hash, seed, timing, and output path are recorded |
| `exp3078` | No hardware execution required | Must not flash GateMate, run SSQA, claim Boltzmann sampling, or claim speedup unless operator-provided host-visible evidence already exists |
| KV260, PolarFire, Extropic/TSU, Kona | Context only | No local benchmark, dispatch, sampling, or thermodynamic claim in `.287` |

All compute-bound tasks must check preconditions before generation and must
write an honest gate-blocked artifact if the required local model, GPU, or
prompt-output path is unavailable.

## Required SOTA Model Policy

Every `.287` experiment that invokes an LLM must include at least one mandated
local GGUF model in `MODEL_SPECS`:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

Legacy small models may appear only as fast CPU smoke tests and never as
headline-result models.

## Failed-Experiment Rerun Compliance

The roadmap deliberately avoids retired hardware and repair rerun scopes. Tasks
that touch failed or bounded areas include explicit `prior_failures` entries
with `retire_if_same_verdict: true`:

- `exp3068` covers matrix/capstone blocker carryforward from `exp3065` and
  `exp3066`.
- `exp3069`, `exp3070`, and `exp3072` address the negative verifier-gain result
  from `exp3057`.
- `exp3071` addresses the no-lift LLM-guided SMT result from `exp3058`.
- `exp3075` addresses the gate-blocked repair rerun from `exp3059` and the
  bounded repair lineage.
- `exp3076` and `exp3077` address the flagged controller-only FR-11 lineage from
  `exp3061`.
- `exp3078` addresses the GateMate/SSQA no-rerun ledgers from `exp3063` and
  `exp3064`.

No task depends on a retired upstream experiment ID. Hardware tasks remain
ledger-only unless operator evidence changes the preconditions.

## Acceptance Criteria

- `research-roadmap-next.yaml` validates against the roadmap schema.
- `scripts/validate_prior_failures.py research-roadmap-next.yaml` passes.
- `scripts/exclusion_manifest_lint.py research-roadmap-next.yaml` passes.
- `scripts/audit_roadmap_gates.py research-roadmap-next.yaml` passes.
- Every prompt contains `CONTEXT`, `EXISTING CODE TO READ FIRST`, `TASK`, and
  `CONCRETE STEPS`.
- Every prompt ends with `Run command, 'Do NOT push. Do NOT modify scripts/research_conductor.py.'`
- Every LLM-using experiment includes mandated SOTA GGUF `MODEL_SPECS`.
- At least one experiment targets continuous self-learning with exact feedback,
  controls, delayed regression, and rollback.
- Repair runs only after a positive verifier-gain gate.
- Hardware remains no-rerun/no-claim unless host-visible operator evidence is
  already present.

## Out Of Scope

- Updating the active `research-roadmap.yaml`.
- Modifying `scripts/research_conductor.py`.
- Pushing branches or tags.
- Claiming GateMate, SSQA, KV260, PolarFire, Extropic, Kona, Ising, or
  thermodynamic speedups.
- Claiming native KAN/PWA, EBT, ARM-EBM, or model-weight FR-11 integration
  without a dedicated implementation and verification milestone.

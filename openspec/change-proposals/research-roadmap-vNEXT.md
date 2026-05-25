# Research Roadmap vNEXT - Milestone 2026.05.286

**Title:** Retire Gate-Rerun Blockers + Solver-Grounded Verification + FR-11 Promotion Boundary
**Created:** 2026-05-25
**Status:** Planned
**Supersedes:** 2026.05.285 "GateMate Output Unblock + Repair Flag Hygiene + Governed FR-11 Self-Learning"
**Execution queue:** `research-roadmap-next.yaml`

## What 2026.05.285 Proved

Milestone `.285` completed all planned tasks, but the authoritative capstone
kept the research program below paper-ready status. The authority artifact is
`results/experiment_3053_capstone_v285.json`:

- `capstone_ready=true`
- `paper_ready=false`
- `repair_claim_status=bounded`
- `fr11_self_learning_status=controller_only_solver_feedback_and_locality_ready`
- `gatemate_status=blocked_output_contract`
- `ssqa_status=gated_skipped_host_visible_smoke_missing`
- `next_milestone_recommendation=2026.05.286_retire_gate_rerun_blockers`

The useful positive result is that `.285` converted FR-11 from a vague
self-learning story into governed controller-side evidence: exact solver
feedback, rollback, held-out family checks, and KAN locality were all measured
without claiming model-weight learning. It also proved local SOTA transcript
fingerprinting with `unsloth/gemma-4-26B-A4B-it-GGUF` and added exact SMT/SAT
validator-tree evidence.

The hard negative result is that repair and hardware cannot advance by another
ungated rerun. Repair remained bounded because `exp3028` still carries
adversarial/methodology blockers, including tautological deltas,
implausibly-perfect zero-delta rows, duration too short for the claimed live
SOTA run, and missing random-seed metadata. GateMate remained blocked because
the output contract lacks operator authority, physical binding, and a host
reader transcript. SSQA therefore correctly stayed gate-skipped.

| Area | `.285` result | `.286` consequence |
| --- | --- | --- |
| Repair | `exp3042` kept repair bounded with 9 blockers | Retire unsupported repair headline wording before any live rerun; rerun only under de-tautology and verifier-gain gates |
| Reproducibility | `exp3043` produced a small live SOTA transcript fingerprint | Use the fingerprint as a precondition for any new live repair or verifier panel |
| Validator frontier | `exp3044` separated verified, unresolved, fallback-only, and correction-set cases | Add local solver-verifier gain and LLM-guided SMT with formal fallback |
| FR-11 | `exp3046` and `exp3047` are controller-only ready | Add solver self-model traces and delayed-regression checks without model-weight claims |
| Hardware | `exp3048` blocked GateMate on output-contract authority | Stop rerunning RTL/flash/SSQA branches until operator evidence exists |
| Matrix/capstone | `exp3052` matrix v19 and `exp3053` capstone are complete but not paper-ready | Matrix v20 must reduce publication blockers by retirement or exact bounded status, not wording |

## Three Biggest Gaps To PRD Vision

1. **Claim-retirement gap.** The PRD requires verifiable, honest reasoning
   claims. Current repair and hardware rows still invite expensive reruns whose
   prerequisites are missing. `.286` must retire unsupported headline wording
   and create explicit no-rerun gates before any additional live work.

2. **Solver-grounding gap.** Carnot has exact validators, but it has not yet
   measured when a local SOTA verifier actually improves solver output
   selection or when LLM-guided SMT preserves formal fallback. `.286` should
   treat LLMs as proposal mechanisms and solvers as authority.

3. **Continuous self-learning promotion gap.** FR-11 is now controller-only
   ready, but still lacks a durable self-model trace with delayed regression
   and family-local credit assignment. `.286` should advance FR-11 by adding
   solver-grounded process traces, not by escalating to model-weight learning.

## New Research Integrated

The post-`.285` sweep was added to `research-references.md` before this
milestone was designed. The findings that materially shape `.286` are:

| Finding | Source | Milestone use |
| --- | --- | --- |
| Solver-verifier gain | OpenReview ICLR 2026 Workshop "Beyond Solving" | Add a local SOTA verifier-gain panel before repair candidate selection |
| Approximately Aligned Decoding | OpenReview NeurIPS 2025 / arXiv:2410.01103 | Use as a repair acceptance design reference to preserve draft intent under hard gates |
| AquaForte LLM-guided SMT | arXiv:2601.04675 / Hugging Face papers | Let SOTA LLMs propose SMT instantiations while Z3/CVC5-style checks remain authority |
| StepORLM external solver feedback | arXiv:2509.22558 | Add solver self-model traces and delayed-regression windows for FR-11 |
| KAN verification and forgetting caveats | arXiv:2602.06737, arXiv:2605.12306, arXiv:2511.12828 | Keep KAN locality bounded unless PWA/MILP and nonforgetting evidence exist |
| FPGA probabilistic sampling | arXiv:2512.24558 | Keep hardware acceleration as long-term context; no local speedup claim before GateMate host-visible output |
| Extropic/Kona public updates | extropic.ai, logicalintelligence.com | Architecture context only; no external-access or benchmark borrowing |

## Architecture Snapshot

```text
                        +--------------------------------+
                        |  Mandated local SOTA GGUFs     |
                        |  Qwen3.6-35B-A3B               |
                        |  Gemma-4-31B dense             |
                        |  Gemma-4-26B-A4B               |
                        +----------------+---------------+
                                         |
                                         v
 +--------------------+       +-------------------------+       +-------------------+
 | Retire unsupported | ----> | Verifier-gain + AprAD   | ----> | Repair rerun only |
 | repair headlines   |       | de-tautology protocol   |       | if gates are met  |
 +--------------------+       +-------------------------+       +-------------------+

 +--------------------+       +-------------------------+       +-------------------+
 | Exact SMT/SAT      | ----> | LLM-guided SMT proposals| ----> | Solver authority  |
 | validator tree     |       | with formal fallback    |       | matrix evidence   |
 +--------------------+       +-------------------------+       +-------------------+

 +--------------------+       +-------------------------+       +-------------------+
 | Governed FR-11     | ----> | Solver self-model trace | ----> | Delayed regression|
 | controller evidence|       | and rollback ledger     |       | + KAN/PWA audit   |
 +--------------------+       +-------------------------+       +-------------------+

 +--------------------+       +-------------------------+       +-------------------+
 | GateMate blocked   | ----> | No-rerun operator ledger| ----> | SSQA remains gated|
 | output contract    |       | until host-visible IO   |       | until smoke exists|
 +--------------------+       +-------------------------+       +-------------------+

                                         |
                                         v
                        +--------------------------------+
                        | Cross-corpus matrix v20        |
                        | Capstone .286                  |
                        +--------------------------------+
```

## Phase Plan

### Phase A - Archive, Claim Retirement, Repair De-Tautology

Tasks: `exp3054`-`exp3056`

- Archive `.285` and pre-stage `.286` without modifying the active roadmap.
- Retire unsupported repair headline wording and record which rows cannot be
  rerun without new evidence.
- Build a repair de-tautology protocol that consumes the `.285` transcript
  fingerprint and prevents tautological, too-fast, or seedless live rows.

Exit condition: repair has a clean rerun protocol and retired unsupported
claims, or remains bounded with machine-readable blockers.

### Phase B - Solver-Grounded Verification and SOTA Repair Gate

Tasks: `exp3057`-`exp3059`

- Measure a tiny local SOTA solver-verifier panel with verifier gain, false
  positives, and cross-family selection.
- Pilot AquaForte-style LLM-guided SMT instantiation with exact fallback.
- Run a tightly gated SOTA repair de-tautology rerun only if the protocol and
  verifier-gain gates both pass.

Exit condition: local SOTA repair evidence is either cleanly produced with
mandated GGUF models and reproducibility metadata, or skipped before expensive
generation because the structured gates failed.

### Phase C - FR-11 Solver Self-Model and KAN/PWA Boundary

Tasks: `exp3060`-`exp3062`

- Define a solver self-model trace schema for FR-11.
- Run the mandatory continuous self-learning pilot with delayed-regression,
  rollback, and solver-process feedback.
- Audit whether KAN locality can be bounded by PWA/MILP-style verification
  without claiming native KAN model-weight learning.

Exit condition: FR-11 advances only as governed controller-side learning unless
stronger evidence is actually produced.

### Phase D - Hardware Gate Retirement, Matrix, and Capstone

Tasks: `exp3063`-`exp3066`

- Convert the GateMate output-contract failure into a no-rerun operator ledger.
- Keep SSQA bounded until host-visible GateMate smoke evidence exists.
- Build cross-corpus matrix v20 from `.285` and `.286` artifacts.
- Write a capstone that can only set `paper_ready=true` if repair, solver
  grounding, FR-11, GateMate, and SSQA claims are clean under matrix v20.

Exit condition: no GateMate flash, timing, SSQA, Boltzmann, or speedup claim can
run through the roadmap unless the missing output-contract evidence is present,
and matrix/capstone reduce blockers by evidence or retirement, not wording
changes.

## Dependency Graph

```text
exp3054 archive
  |
  v
exp3055 repair headline retirement
  |
  v
exp3056 repair de-tautology protocol

exp3057 local SOTA verifier-gain panel
  |
  v
exp3058 AquaForte-style SMT pilot

exp3056 + exp3057(verifier_gain_delta > 0)
  |
  v
exp3059 gated SOTA repair rerun

exp3060 FR-11 solver self-model trace schema
  |----------------------.
  v                      v
exp3061 FR-11 delayed regression pilot
  |
  v
exp3062 KAN/PWA locality audit

exp3063 GateMate no-rerun ledger
  |
  v
exp3064 SSQA boundary ledger

exp3055, exp3057, exp3058, exp3059, exp3061, exp3062, exp3063, exp3064
  |
  v
exp3065 matrix v20
  |
  v
exp3066 capstone .286
```

Structured conductor gates are present where they save expensive agent calls:

- `exp3058` gates on `exp3057.solution_verifier_calibration_ready`.
- `exp3059` gates on `exp3056.repair_de_tautology_protocol_ready` and
  `exp3057.verifier_gain_delta > 0.0`.
- `exp3061` gates on `exp3060.solver_self_model_trace_ready` and
  `exp3058.formal_fallback_preserved`.
- `exp3062` gates on `exp3061.fr11_delayed_regression_ready`.
- `exp3066` gates on `exp3065.matrix_v20_ready`.

`exp3065` is intentionally not gated on live repair or hardware success. The
matrix must record clean, bounded, blocked, gate-skipped, missing, and retired
rows.

## Hardware Requirements

| Resource | Required by | Requirement | Claim boundary |
| --- | --- | --- | --- |
| Dual RTX 3090 CUDA host | `exp3057`, `exp3058`, `exp3059` | At least one mandated GGUF loadable through `cached_sota_pair()` or the repo's SOTA cache path | No SOTA headline if only legacy smoke models run |
| GateMate A1-EVB-2M | none for execution in `.286` unless operator evidence is already present | Output pinout, CCF binding, and host reader transcript remain preconditions | No flash, latency, sampler, Boltzmann, SSQA, or speedup claim without host-visible transcript |
| KV260 | none in `.286` | Available but out of scope | Do not use host SD-card preconditions |
| PolarFire | none in `.286` | Available but out of scope | No PolarFire claims |
| Extropic TSU/XTR-0/XTR-1 | none | Public context only | No access or performance implication |

## Acceptance Criteria

1. `research-roadmap-next.yaml` validates against YAML parsing, gate audit, and
   prior-failure/exclusion-manifest lint.
2. Every task has a concrete JSON deliverable path.
3. Every prompt includes CONTEXT, EXISTING CODE TO READ FIRST, TASK, and
   CONCRETE STEPS.
4. Every prompt ends with: `Run command, 'Do NOT push. Do NOT modify scripts/research_conductor.py.'`
5. Every LLM-using task names at least one mandated local SOTA GGUF in
   `MODEL_SPECS`.
6. `exp3061` satisfies the mandatory continuous self-learning requirement.
7. Repair rerun work is structurally gated on the de-tautology protocol and
   positive verifier gain.
8. GateMate and SSQA do not execute hardware or claim performance without
   host-visible output evidence.
9. Prior-failure metadata includes all four mandatory fields, including
   `retire_if_same_verdict: true`, for carry-forward scopes.
10. The capstone does not set `paper_ready=true` unless matrix v20 supports the
    promoted claims with source artifacts and honest boundaries.

## Failed-Experiment Rerun Compliance

This milestone intentionally revisits scopes from `.285` and earlier. Every
carry-forward task in `research-roadmap-next.yaml` includes `prior_failures`
with the four mandatory fields:

- `experiment_id`
- `verdict`
- `addressed_by`
- `retire_if_same_verdict: true`

The roadmap avoids `requires` chains that point to retired upstream experiment
IDs. Where a downstream task is conditional, it uses same-milestone structured
`gated_on` fields.

## Out Of Scope

- No modification to `research-roadmap.yaml`.
- No modification to `scripts/research_conductor.py`.
- No public submission, arXiv action, PyPI action, or external publication.
- No Extropic, Kona, KV260, PolarFire, or external-hardware performance claim.
- No model-weight self-learning claim unless an experiment actually trains and
  verifies model weights, which this milestone does not plan.
- No GateMate flash, SSQA readback, hardware speedup, or Boltzmann sampling
  claim until a host-visible output contract exists.

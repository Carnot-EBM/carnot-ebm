# Research Roadmap vNEXT - Milestone 2026.05.297

**Title:** CUDA Receipt Recovery + Context-Shortcut Verification + Evidence-Gated FR-11 Replay

**Created:** 2026-05-27
**Status:** Proposed, staged in `research-roadmap-next.yaml`
**Supersedes:** Milestone 2026.05.296
**Execution queue:** `exp3205` through `exp3218`

## What Milestone 2026.05.296 Proved

Milestone `.296` did not unblock headline verifier or repair claims. It did
make the critical path much sharper:

- `exp3193` proved the local SOTA runtime blocker is no longer "model file
  missing" in the abstract. The machine had an RTX 3090 visible to
  `nvidia-smi`, but the selected Python/runtime path reported
  `torch.cuda.is_available() == false`, `llama_cpp` did not expose usable GPU
  offload, and stderr contained `ggml_cuda_init: failed to initialize CUDA:
  unknown error`.
- `exp3194` clean live SOTA verifier v11 correctly gate-skipped because
  `clean_rerun_allowed=false`.
- `exp3198` repair gate v5 remained blocked on the skipped clean verifier,
  and `exp3199` repair ladder v6 gate-skipped behind the blocked repair gate.
- `exp3200` promoted FR-11 VeriFY-style trace memory as a controller policy
  without model-weight updates, with no negative-control regression.
- `exp3201` kept KAN-CL as a non-promoted sidecar; `exp3202` kept
  Sparse Potts/PAOA/THRML as a diagnostic factor boundary with no authenticated
  speedup claim.
- `exp3203`/`exp3204` reported `paper_ready=false`,
  `publication_blocker_count=85`, and
  `next_top_gap=cuda_offload_full_local_sota_receipt_clean_rerun_allowed_repair_gate_unblock`.

The next milestone should therefore avoid another expensive ungated repair
rerun. It should first turn CUDA/offload failure into an environment receipt,
then run clean verifier and repair only behind structured gates. In parallel it
should add exact context-shortcut and constraint-optimization fixtures so the
eventual clean SOTA rerun is more diagnostic than a raw benchmark replay.

## Three Biggest Gaps To PRD Vision

1. **Local SOTA verifier authority is still missing.** The PRD requires
   reproducible, verifiable reasoning claims over current strong local models.
   Carnot currently has CPU-fallback smoke evidence for a mandated GGUF family,
   not a full CUDA/offload receipt that can unlock clean verifier and repair
   claims.

2. **The verifier/repair loop is gated but not yet recovered.** Exact fixtures,
   counterexamples, adaptive granularity, GenCP preview, and ExVerus-style
   invariant expansion exist as supporting artifacts, but the headline loop is
   still blocked behind `clean_rerun_allowed=false`.

3. **FR-11 self-learning is controller memory, not a full governed learning
   loop.** `.296` promoted trace memory without model-weight updates. The PRD
   vision needs ongoing self-learning that admits evidence-backed traces,
   suppresses redundant checks, regulates forgetting, and remains rollback-safe.

## New Research Integrated

The post-`.296` sweep was added to `research-references.md` before this design.
The most relevant additions are:

- llama.cpp CUDA build/runtime documentation: CUDA receipt recovery must become
  environment forensics, not another generic smoke test.
- Context-CoT / CL-Bench: add context-dependent, parametric-shortcut fixtures
  that distinguish following supplied context from relying on pretrained priors.
- ConstraintBench: report feasibility, objective gap, and hallucinated entities
  separately for direct constrained optimization.
- Reward-Weighted On-Policy Distillation with an open property-equivalence
  verifier: use verifier utility as FR-11 replay labels without fine-tuning.
- Evidence Over Plans / SPARK: admit only environment- or verifier-backed
  trajectories into self-learning promotion.
- Grounded Continuation: model multi-turn stale-premise handling as an explicit
  dependency graph.
- KAN PWA/MILP verification: keep KAN sidecars bounded unless they emit
  abstraction certificates.
- llguidance and XGrammar-2: constrained decoding can remove repair syntax
  failures, but exact verifiers must still score semantics.
- Extropic TSU/THRML and Logical Intelligence Kona: strategic architecture
  signals only; no Carnot speedup or execution claim without local receipts.

## Architecture Diagram

```text
                 research-complete.yaml / conductor-log
                               |
                               v
                  exp3205 archive + activate .297
                               |
                               v
          +---------------- exp3206 ----------------+
          | CUDA env forensics + import/order ledger |
          +---------------------+-------------------+
                                |
                                v
          +---------------- exp3207 ----------------+
          | llama.cpp CUDA rebuild / clean subprocess |
          +---------------------+-------------------+
                                |
                 cuda_receipt_ready == true
                                |
                                v
          +---------------- exp3208 ----------------+
          | full local SOTA GGUF receipt v5          |
          | Qwen3.6-35B-A3B / Gemma-31B / Gemma-26B |
          +---------------------+-------------------+
                                |
                 clean_rerun_allowed == true
                                |
             +------------------+------------------+
             v                                     v
       exp3209 clean verifier v12          exp3212 structured
       exact rows + SOTA calls             repair proposal preflight
             |                                     |
             +------------------+------------------+
                                v
          +---------------- exp3213 ----------------+
          | repair gate v6                          |
          +---------------------+-------------------+
                                |
                 repair_gate_state == unblocked
                                |
                                v
          +---------------- exp3214 ----------------+
          | multi-turn repair ladder v7             |
          +-----------------------------------------+

 Parallel exact-fixture and FR-11 tracks:

   exp3210 Context-CoT/CL-Bench fixture bank
   exp3211 ConstraintBench feasibility/objective pilot
   exp3215 evidence-gated FR-11 replay controller
   exp3216 grounded-continuation trace graph and nonforgetting queue
                  \              |              /
                   \             v             /
                    +------ exp3217 matrix v31
                                   |
                                   v
                            exp3218 capstone
```

## Phase Plan

### Phase 1 - CUDA Receipt Recovery And Gate Hygiene

Goal: convert the `.296` CUDA failure into a reproducible environment ledger
and only then attempt a full local SOTA receipt.

- `exp3205-archive-v296-activate-v297`
- `exp3206-cuda-env-forensics-ledger-v1`
- `exp3207-llama-cpp-cuda-rebuild-clean-subprocess-v1`
- `exp3208-full-local-sota-receipt-v5`

Success means `exp3208` emits `clean_rerun_allowed=true` with a full local SOTA
receipt. If not, downstream clean verifier and repair tasks must gate-skip
without spending an LLM call.

### Phase 2 - Context/Constraint Fixtures, Clean Verifier, Repair Gate

Goal: make the verifier rerun more diagnostic while preserving exact authority.

- `exp3209-clean-live-sota-verifier-rerun-v12`
- `exp3210-context-cot-clbench-parametric-shortcut-fixtures-v1`
- `exp3211-constraintbench-feasibility-objective-pilot-v1`
- `exp3212-structured-repair-proposal-preflight-v1`
- `exp3213-repair-gate-decision-v6`
- `exp3214-multi-turn-repair-ladder-v7`

Success means either the repair ladder is honestly unblocked and executed, or
the artifact chain cleanly reports which gate still blocks it. Context and
ConstraintBench fixtures should be useful even when CUDA remains blocked.

### Phase 3 - Continuous Self-Learning Without Weight Updates

Goal: advance FR-11 from trace-memory promotion to evidence-gated replay with
forgetting regulation and stale-premise handling.

- `exp3215-fr11-evidence-gated-trace-replay-controller-v2`
- `exp3216-fr11-grounded-continuation-nonforgetting-queue-v1`

Success means FR-11 reports verifier-backed replay utility, redundant-check
suppression, negative-control stability, rollback metadata, and a virtual
forgetting/nonforgetting queue. No task may claim model-weight updates.

### Phase 4 - Aggregation And Claim Boundaries

Goal: reconcile all artifacts into the matrix and capstone without inflating
claims.

- `exp3217-cross-corpus-matrix-v31`
- `exp3218-capstone-v297`

Success means the matrix reports all blocker deltas, gate skips, SOTA receipt
state, FR-11 promotion state, and hardware claim boundary. The capstone must
name the next top gap.

## Dependency Graph

```text
exp3205
  -> exp3206
      -> exp3207
          -> exp3208 [gated on exp3207.cuda_receipt_ready == true]
              -> exp3209 [gated on exp3208.clean_rerun_allowed == true]
              -> exp3212 [gated on exp3208.clean_rerun_allowed == true]
                  -> exp3213
                      -> exp3214 [gated on exp3213.repair_gate_state == unblocked]

exp3210 -> exp3213 -> exp3217
exp3211 -> exp3213 -> exp3217
exp3215 -> exp3217
exp3216 -> exp3217
exp3217 -> exp3218
```

## Hardware Requirements

### Required For Headline Local SOTA Claims

- One or both local RTX 3090 GPUs visible to `nvidia-smi`.
- A CUDA-capable PyTorch/runtime path that reports nonzero device count in a
  clean subprocess.
- A CUDA-built `llama_cpp` or `llama.cpp` binary with GPU offload support.
- Local GGUF cache for at least one mandated SOTA model:
  - `unsloth/Qwen3.6-35B-A3B-GGUF`
  - `unsloth/gemma-4-31B-it-GGUF`
  - `unsloth/gemma-4-26B-A4B-it-GGUF`

### Allowed Diagnostic-Only Hardware Work

- KV260, GateMate A1, PolarFire, THRML, Extropic TSU, and Kona references may
  appear only as boundary or architecture diagnostics unless a local transcript
  proves execution. No speedup or board-execution claim may be made from
  simulation or vendor writing alone.

## SOTA Local GGUF Policy

Any `.297` experiment that invokes an LLM must include at least one mandated
model in its `MODEL_SPECS`:

- `unsloth/Qwen3.6-35B-A3B-GGUF` as the flagship MoE model.
- `unsloth/gemma-4-31B-it-GGUF` as the flagship dense model.
- `unsloth/gemma-4-26B-A4B-it-GGUF` as the middle MoE model.

Legacy small models may be used only as fast CPU smoke tests and must not
populate headline result fields. Prompts should reference
`scripts/experiment_template.py` and the `cached_sota_pair()` pattern.

## Failed-Rerun Discipline

Every task whose scope matches a prior failed or blocked chain includes a
`prior_failures` entry with:

- `experiment_id`
- `verdict`
- `addressed_by`
- `retire_if_same_verdict: true`

The gating chain avoids references to retired upstream experiment IDs and uses
structured `gated_on` entries for expensive downstream tasks.

## Success Criteria

- `research-roadmap-next.yaml` validates under roadmap schema,
  prior-failure lint, exclusion-manifest lint, and gate audit.
- `research-roadmap.yaml` and `scripts/research_conductor.py` remain untouched.
- `research-references.md` records the post-`.296` sweep before experiment
  design.
- At least one continuous self-learning task is included and explicitly avoids
  model-weight-update claims.
- If CUDA remains blocked, the milestone still yields a precise environment
  ledger, exact fixture artifacts, FR-11 replay/nonforgetting artifacts, and an
  honest capstone with no inflated SOTA or repair claims.

## Out Of Scope

- Pushing commits.
- Modifying `scripts/research_conductor.py`.
- Replacing `research-roadmap.yaml` during planning.
- Claiming local SOTA verifier success from CPU fallback, partial offload, or
  missing transcript evidence.
- Claiming FPGA/TSU/Kona execution or speedup without a local authenticated
  transcript.

# Research Roadmap vNEXT - Milestone 2026.05.298

**Title:** Hermetic CUDA Receipt Repair + Distributional Exact-Row Triage + FR-11 Nonforgetting Promotion

**Created:** 2026-05-27
**Status:** Proposed, staged in `research-roadmap-next.yaml`
**Supersedes:** Milestone 2026.05.297
**Execution queue:** `exp3219` through `exp3232`

## What Milestone 2026.05.297 Proved

Milestone `.297` completed the conductor queue, but the terminal state is still
not paper-ready. Its useful result is a sharper failure boundary:

- `exp3205` archived `.296` and activated `.297` without modifying the
  protected conductor or active roadmap.
- `exp3206` produced the key CUDA ledger: the selected Python was the repo
  `.venv` Python 3.14 path, `nvidia-smi` saw an RTX 3090, PyTorch reported
  `2.11.0+cu128`, but a clean subprocess still returned
  `torch.cuda.is_available() == false`, while `llama_cpp` emitted
  `ggml_cuda_init: failed to initialize CUDA: unknown error`.
- `exp3207` correctly refused a blind rebuild. `cuda_receipt_ready=false`,
  `clean_rerun_allowed_candidate=false`, and the selected Python CUDA failure
  remained the blocker.
- `exp3208`, `exp3209`, and `exp3212` were gate-blocked or skipped behind the
  missing CUDA/offload receipt. No clean local SOTA verifier or structured
  repair headline result exists for `.297`.
- `exp3210` and `exp3211` created useful exact fixture banks: 30 context
  shortcut fixtures and 15 ConstraintBench-style feasibility/objective pilot
  fixtures.
- `exp3213` kept the repair gate blocked, and `exp3214` correctly skipped the
  repair ladder.
- `exp3215` promoted verifier-backed FR-11 trace replay only as a controller
  policy, with no model-weight update claim.
- `exp3216` materialized a grounded-continuation trace graph and a
  nonforgetting queue, but kept promotion audit-only.
- `exp3217` and `exp3218` reported `paper_ready=false`,
  `publication_blocker_count=92`, and
  `next_top_gap=cuda_offload_full_local_sota_receipt_clean_rerun_allowed_repair_gate_unblock`.

The next milestone should therefore spend its first phase on a hermetic CUDA
repair/receipt path rather than another ungated verifier rerun. Exact fixture
work and FR-11 governance can progress in parallel because they do not require
the full local SOTA receipt.

## Three Biggest Gaps To PRD Vision

1. **Local SOTA authority is still blocked at the selected Python runtime.**
   The PRD requires reproducible verifier claims over current local SOTA GGUF
   models. Carnot currently has a visible RTX 3090 and model-cache discipline,
   but no selected-runtime CUDA receipt that can unlock clean verifier and
   repair claims.

2. **Exact fixtures exist, but the clean verifier and repair loop remain
   gated.** `.297` added context and constrained-optimization fixtures, yet the
   headline live-SOTA verifier, schema-constrained repair preflight, repair
   gate, and repair ladder are still blocked behind `clean_rerun_allowed=false`.

3. **Continuous self-learning remains controller-local and audit-heavy.**
   FR-11 now has trace replay and a nonforgetting queue, but not a governed
   promotion/rollback policy or a certified KAN sidecar boundary that can
   safely admit new evidence over time.

## New Research Integrated

The 2026-05-27 external sweep was added to `research-references.md` before this
roadmap was designed. The most relevant findings are:

- Distributional EBMs for structured LLM reasoning
  (`https://arxiv.org/abs/2605.18871`): useful as an uncertainty and
  abstention sidecar over exact rows, not as a replacement for exact scoring.
- Logitext / neurosymbolic language reasoning as SMT
  (`https://arxiv.org/abs/2602.18095`): motivates a partial SMT coverage pilot
  that measures which natural-language constraints can be formalized exactly.
- KAN-CL (`https://arxiv.org/abs/2605.12306`): relevant for continual learning
  only if Carnot keeps explicit certificate, nonforgetting, and rollback
  boundaries.
- Fully parallel p-bit Ising with inertia
  (`https://arxiv.org/abs/2604.17109`): a future hardware-boundary signal, but
  not enough to justify rerunning retired PIMI-style hardware claims in `.298`.
- EBT/ARM-EBM citation watch
  (`https://www.semanticscholar.org/paper/Energy-Based-Transformers-are-Scalable-Learners-and-Gladstone-Nanduru/2da9163730998a4368c609972ccff0582518b36b`,
  `https://arxiv.org/abs/2512.15605`): keep monitoring for verifier-style
  EBM uses, but local receipt and exact rows are the immediate bottleneck.
- Structured decoding libraries (`https://github.com/guidance-ai/llguidance`,
  `https://github.com/mlc-ai/xgrammar`): good candidates for schema-valid
  repair proposals once clean SOTA receipt is unblocked; they do not certify
  semantic correctness.
- Extropic TSU and Logical Intelligence Kona
  (`https://extropic.ai/writing/thermodynamic-computing-from-zero-to-one`,
  `https://logicalintelligence.com/blog/energy-based-models-for-reasoning`):
  strategic architecture signals only. `.298` makes no thermodynamic or Kona
  hardware speedup claim.

## SOTA Local GGUF Policy

Any `.298` task that invokes an LLM for headline evidence must include at least
one mandated local SOTA GGUF in `MODEL_SPECS`:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

Legacy small models may appear only as fast CPU smoke tests. They cannot
populate headline result fields and cannot unblock clean verifier, repair, or
publication-readiness claims.

## Architecture Diagram

```text
                    .297 terminal artifacts
     exp3218 capstone / exp3217 matrix / conductor-log / references
                                  |
                                  v
                    exp3219 archive + activate .298
                                  |
                                  v
         +------------------- exp3220 -------------------+
         | Hermetic CUDA runtime repair ledger            |
         | selected .venv vs CUDA-only isolated runtime   |
         +------------------------+-----------------------+
                                  |
              cuda_receipt_ready_candidate == true
                                  |
                                  v
         +------------------- exp3221 -------------------+
         | llama.cpp CUDA/offload receipt smoke           |
         +------------------------+-----------------------+
                                  |
                    cuda_receipt_ready == true
                                  |
                                  v
         +------------------- exp3222 -------------------+
         | Full local SOTA GGUF receipt v6                |
         | Qwen3.6-35B-A3B / Gemma-31B / Gemma-26B        |
         +------------------------+-----------------------+
                                  |
                    clean_rerun_allowed == true
                                  |
              +-------------------+-------------------+
              v                                       v
      exp3225 clean live SOTA verifier       exp3226 structured repair
      exact rows + mandated GGUFs            proposal preflight
              |                                       |
              +-------------------+-------------------+
                                  v
                         exp3227 repair gate
                                  |
                    repair_gate_state == unblocked
                                  |
                                  v
                         exp3228 repair ladder

 Parallel exact-row and FR-11 tracks:

   exp3223 distributional EBM uncertainty sidecar over exact rows
   exp3224 Logitext-style partial SMT coverage pilot
   exp3229 FR-11 nonforgetting promotion/rollback controller
   exp3230 KAN-CL certificate boundary audit
            \              |                 /
             \             v                /
              +------ exp3231 matrix v32 ---+
                              |
                              v
                       exp3232 capstone .298
```

## Phase Plan

### Phase 1 - Hermetic CUDA Receipt Repair

Goal: isolate whether CUDA failure is specific to the selected `.venv` Python,
the system driver/runtime boundary, llama.cpp linkage, import order, or an
environment variable conflict.

- `exp3219-archive-v297-activate-v298`
- `exp3220-hermetic-cuda-runtime-repair-ledger-v1`
- `exp3221-llama-cpp-cuda-offload-receipt-smoke-v1`
- `exp3222-full-local-sota-receipt-v6`

Success means `exp3222` emits `clean_rerun_allowed=true` with a non-CPU CUDA or
GPU-offload receipt for at least one mandated local SOTA GGUF. If the selected
runtime remains blocked, downstream clean verifier and repair tasks should skip
through structured gates without burning synthesis time.

### Phase 2 - Exact-Row Triage, Clean Verifier, And Repair Gate

Goal: improve the evidence quality of the eventual live-SOTA rerun, then run
the verifier and repair tasks only when the receipt gate is satisfied.

- `exp3223-distributional-ebm-exact-row-uncertainty-sidecar-v2`
- `exp3224-logitext-partial-smt-context-coverage-pilot-v1`
- `exp3225-clean-live-sota-verifier-rerun-v13`
- `exp3226-structured-repair-proposal-preflight-v2`
- `exp3227-repair-gate-decision-v7`
- `exp3228-multi-turn-repair-ladder-v8`

Success means the clean verifier either reports a real local SOTA delta over
exact rows or a blocked artifact with the precise runtime gate. Structured
repair may claim schema validity and proposal hygiene, but not semantic repair
success unless the exact verifier scores it.

### Phase 3 - Continuous Self-Learning Governance

Goal: advance FR-11 from trace replay and audit queues into a governed
controller-level promotion loop with rollback metadata and explicit
nonforgetting budgets.

- `exp3229-fr11-nonforgetting-promotion-controller-v3`
- `exp3230-kan-cl-certificate-boundary-audit-v2`

Success means Carnot can admit evidence-backed traces to controller memory,
reject stale-premise regressions, define rollback, and keep KAN-CL sidecars
non-promoted unless they provide certificate-ready boundaries. No task may
claim model-weight updates.

### Phase 4 - Matrix And Capstone

Goal: reconcile runtime, verifier, repair, self-learning, hardware, and
publication claims without inflating blocked paths.

- `exp3231-cross-corpus-matrix-v32`
- `exp3232-capstone-v298`

Success means the matrix and capstone report `paper_ready`, publication
blocker deltas, SOTA receipt state, repair gate state, FR-11 promotion state,
and the next top gap from actual artifacts.

## Dependency Graph

```text
exp3219
  -> exp3220
      -> exp3221 [gated on exp3220.cuda_receipt_ready_candidate == true]
          -> exp3222 [gated on exp3221.cuda_receipt_ready == true]
              -> exp3225 [gated on exp3222.clean_rerun_allowed == true]
              -> exp3226 [gated on exp3222.clean_rerun_allowed == true]
                  -> exp3227
                      -> exp3228 [gated on exp3227.repair_gate_state == unblocked]

exp3223 -> exp3225 -> exp3231
exp3224 -> exp3225 -> exp3231
exp3226 -> exp3227 -> exp3231
exp3228 -> exp3231
exp3229 -> exp3231
exp3230 -> exp3231
exp3231 -> exp3232
```

## Hardware Requirements

### Required For Headline Local SOTA Claims

- RTX 3090 visible to `nvidia-smi`.
- Selected Python runtime with PyTorch CUDA initialization working in a clean
  subprocess, or an isolated CUDA-only runtime with an explicit handoff path.
- llama.cpp or llama-cpp-python build with CUDA/GPU offload metadata and a
  clean subprocess smoke receipt.
- Local access to at least one mandated SOTA GGUF:
  `unsloth/Qwen3.6-35B-A3B-GGUF`,
  `unsloth/gemma-4-31B-it-GGUF`, or
  `unsloth/gemma-4-26B-A4B-it-GGUF`.

### Not Required For `.298`

- KV260, GateMate, PolarFire, AMD XDNA, TSU, or Kona hardware is not required
  for this milestone.
- `.298` makes no FPGA, thermodynamic, p-bit, or Kona speedup claim. Hardware
  references remain architecture guidance unless an authenticated local
  transcript exists.

## Claim Boundaries

- A visible GPU is not a local SOTA receipt.
- CPU fallback is not headline evidence.
- Schema-valid repair proposals are not semantic repair success.
- Distributional EBM uncertainty is triage, not exact verification.
- Logitext-style SMT coverage is partial coverage, not full natural-language
  understanding.
- FR-11 controller-memory promotion is not model-weight training.
- KAN-CL remains a bounded sidecar unless certificate boundaries are emitted.

## Expected Terminal Outcomes

At the end of `.298`, one of these should be true:

1. **Receipt recovered:** `exp3222.clean_rerun_allowed=true`; clean verifier and
   repair preflight run with mandated local SOTA models; the capstone reports
   the next blocker after live SOTA evidence.
2. **Runtime still blocked, but isolated:** `exp3220` or `exp3221` identifies a
   precise selected-runtime or driver/toolchain blocker; exact-row and FR-11
   work still improves the next retry.
3. **Repair remains blocked by evidence, not planning:** clean verifier, repair
   proposal, gate, and ladder statuses are all structured and auditable, with
   no wasted ungated rerun.

## Out Of Scope

- Pushing changes.
- Modifying `scripts/research_conductor.py`.
- Modifying the active `research-roadmap.yaml`.
- Retrying retired PIMI, WOPR, KV260 SD-card, or wrong-mechanism hardware
  scopes.
- Claiming publication readiness without the capstone artifact setting
  `paper_ready=true`.

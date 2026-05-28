# Research Roadmap vNEXT - Milestone 2026.05.300

**Title:** Runtime Receipt Recovery + Prompt-Injection KAN Split-Run + FR-11 Failure Memory

**Created:** 2026-05-28
**Status:** Proposed, staged in `research-roadmap-next.yaml`
**Supersedes:** Milestone 2026.05.299
**Execution queue:** `exp3233` through `exp3245`

## What Milestone 2026.05.299 Proved

Milestone `.299` was intentionally narrow. It proved that the single-focus
Prompt-Injection KAN v4 plan is not ready to run as one large task:

- `exp3221` archived `.298` and activated `.299`.
- `exp3222` attempted Prompt-Injection KAN Distillation v4 three times and did
  not produce `results/experiment_3222_prompt_injection_kan_distill_v4_15k.json`.
  The conductor log records three CLI failures.
- `exp3223` completed the capstone and reported
  `paper_ready=false`, `publication_blocker_count=100`,
  `v4_outcome=blocked_missing_exp3222_result`, and
  `next_top_gap=cuda_chain_for_full_local_sota_receipts`.
- No new KAN corpus, teacher-label shard, verifier rerun, repair preflight,
  or hardware receipt was produced in `.299`.

The next milestone should therefore split the failed monolith into receipt,
manifest, teacher-label, training, and aggregation artifacts. It should also
resume the real blocker identified by `.298` and `.299`: the selected Python /
CUDA / llama.cpp boundary that prevents trustworthy local SOTA GGUF receipts.

## Three Biggest Gaps To PRD Vision

1. **Local SOTA authority is still missing.** The PRD requires reproducible
   verifier evidence over current local GGUF models. Carnot has GPU-visible
   host evidence in prior artifacts, but selected Python still reported CUDA
   unavailable and the llama.cpp/offload chain did not produce a clean receipt.

2. **Prompt-injection KAN work lacks staged, auditable artifacts.** Existing
   prompt-injection KAN artifacts are older and smaller. `.299` attempted a
   15k v4 distillation directly, but failed before producing even a manifest.
   The next attempt must start with corpus, teacher, model-spec, power-analysis,
   and garak/config receipts before any headline metric.

3. **FR-11 is still controller-memory only.** `.298` established useful
   nonforgetting governance, but Carnot still lacks a continuous self-learning
   loop that learns from failed runs, gate blocks, and stale premises without
   claiming foundation-model weight updates.

## External Research Integrated

The 2026-05-28 external sweep was added to `research-references.md` before this
roadmap was designed. The most relevant updates are:

- Distributional EBMs for structured reasoning (`https://arxiv.org/abs/2605.18871`)
  support uncertainty and abstention sidecars over exact rows.
- Draft-Conditioned Constrained Decoding (`https://arxiv.org/abs/2603.03305`)
  motivates repair preflight that drafts freely but validates structure under
  constraints.
- Vectorized constrained decoding (`https://arxiv.org/abs/2602.22647`) suggests
  accelerator-friendly schema constraints if repair decoding becomes a latency
  bottleneck.
- CiteTracer and large-scale citation hallucination audits
  (`https://arxiv.org/abs/2605.08583`, `https://arxiv.org/abs/2605.07723`)
  reinforce field-level evidence artifacts and deterministic matching.
- KAN-CL and KAC (`https://arxiv.org/abs/2605.12306`,
  `https://arxiv.org/abs/2503.21076`) support locality-aware continual-learning
  sidecars only when nonforgetting and rollback fields are explicit.
- P-bit Ising guidance (`https://arxiv.org/abs/2604.17109`,
  `https://arxiv.org/abs/2605.04033`) should remain guidance with exact fallback,
  not a correctness authority.
- Extropic THRML and Logical Intelligence Kona are strategic alignment signals
  (`https://extropic.ai/software`,
  `https://logicalintelligence.com/kona-ebms-energy-based-models`), not `.300`
  speedup or certification claims.

## SOTA Local GGUF Policy

Any `.300` experiment that invokes an LLM for evidence must include at least
one mandated local SOTA GGUF model in `MODEL_SPECS`:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

Legacy small models may appear only as fast CPU smoke tests. They cannot
populate headline result fields and cannot unblock clean verifier, repair, or
publication-readiness claims.

## Architecture Diagram

```text
                 .299 terminal state
  capstone_v299: paper_ready=false, blockers=100,
  v4_outcome=blocked_missing_exp3222_result,
  next_top_gap=cuda_chain_for_full_local_sota_receipts
                              |
                              v
              exp3233 archive + activate .300
                              |
              +---------------+----------------+
              |                                |
              v                                v
  exp3234 backend failure ledger      exp3235 runtime boundary package
              |                                |
              +---------------+----------------+
                              |
                              v
              exp3236 isolated CUDA/Python smoke
                              |
                              v
              exp3237 llama.cpp CUDA receipt smoke
                              |
                              v
              exp3238 mandated SOTA GGUF receipt
                              |
        +---------------------+----------------------+
        |                                            |
        v                                            v
 exp3240 teacher-label shard                 exp3242 DCCD exact-row
    ^        |                                structured proposal preflight
    |        v
 exp3239 -> exp3241 prompt-injection
 manifest    KAN shard train/eval

 exp3243 FR-11 failure-memory controller
          learns from .295-.300 gate and failure traces
          without model-weight update claims

 exp3244 matrix -> exp3245 capstone
```

## Phase Plan

### Phase 1 - Milestone Hygiene and Failure Decomposition

- `exp3233` archives `.299`, records the missing v4 artifact, and activates
  `.300`.
- `exp3234` turns the `.299` CLI failure into a structured root-cause ledger so
  the prompt-injection work is split before rerun.
- `exp3235` writes the runtime-boundary operator package from `.296-.299`
  evidence and names the exact handoff conditions for CUDA, selected Python,
  and llama.cpp.

### Phase 2 - Runtime Receipt Chain

- `exp3236` runs isolated selected-Python and CUDA smoke probes and emits
  `cuda_python_smoke_passed`.
- `exp3237` runs only if `exp3236` passes and emits
  `llama_cpp_cuda_receipt_ready`.
- `exp3238` runs only if `exp3237` passes and attempts the mandated SOTA GGUF
  receipt over the three required local GGUF models.

### Phase 3 - Prompt-Injection and Structured Repair Split-Runs

- `exp3239` builds the prompt-injection v4 manifest, shard plan, power plan,
  and garak/config receipts without invoking an LLM.
- `exp3240` runs a small teacher-label shard only if the manifest and SOTA
  receipt are present.
- `exp3241` trains/evaluates the KAN sidecar on the shard only if teacher
  labels exist, and explicitly marks results as non-headline.
- `exp3242` runs a DCCD-style structured proposal preflight only if the clean
  SOTA rerun gate is open; exact-row verification remains the authority.

### Phase 4 - Continuous Self-Learning and Aggregation

- `exp3243` implements the required continuous self-learning task: a
  failure-memory controller that learns from gate blocks, stale premises, and
  missing artifacts while preserving no-weight-update governance.
- `exp3244` aggregates `.300` evidence into cross-corpus matrix v33.
- `exp3245` produces the `.300` capstone and names the next top gap.

## Dependency Graph

```text
exp3233
  -> exp3234
  -> exp3235
      -> exp3236
          -> exp3237 [gate: cuda_python_smoke_passed == true]
              -> exp3238 [gate: llama_cpp_cuda_receipt_ready == true]
                  -> exp3240 [also requires exp3239.v4_manifest_ready == true]
                  -> exp3242 [gate: clean_rerun_allowed == true]

exp3239 -> exp3240 -> exp3241
exp3243 independent after exp3233, reads .295-.300 traces
exp3244 reads all available .300 artifacts
exp3245 reads exp3244 and all available .300 artifacts
```

## Hardware Requirements

- **Required for Phase 2 success:** one visible NVIDIA GPU, selected Python
  CUDA initialization, and llama.cpp CUDA/offload support. If these fail, the
  milestone still produces a useful boundary artifact and skips downstream live
  SOTA tasks through structured gates.
- **Required for Phase 3 live LLM tasks:** local cached GGUF models from the
  mandated model list. If the full SOTA receipt is unavailable, prompt-injection
  teacher labeling and structured proposal preflight skip rather than falling
  back to legacy small headline models.
- **Not required for `.300`:** KV260, GateMate, PolarFire, Extropic TSU, or
  Kona hardware. Do not use a host `/dev/mmcblk*` KV260 precondition; the KV260
  path, if reopened later, uses SSH and `xmutil`.

## Experiment Queue

1. `exp3233-archive-v299-activate-v300`
2. `exp3234-cli-backend-failure-root-cause-ledger-v1`
3. `exp3235-cuda-driver-boundary-operator-package-v1`
4. `exp3236-isolated-cuda-python-smoke-v1`
5. `exp3237-llama-cpp-cuda-receipt-smoke-v2`
6. `exp3238-sota-gguf-receipt-v7`
7. `exp3239-prompt-injection-kan-v4-resource-manifest-v1`
8. `exp3240-prompt-injection-kan-teacher-label-shard-v1`
9. `exp3241-prompt-injection-kan-train-eval-shard-v1`
10. `exp3242-dccd-exact-row-structured-proposal-preflight-v1`
11. `exp3243-fr11-failure-memory-controller-v1`
12. `exp3244-cross-corpus-matrix-v33`
13. `exp3245-capstone-v300`

## Done Criteria

- `research-roadmap-next.yaml` validates against roadmap schema,
  prior-failure discipline, exclusion-manifest lint, and gate audit.
- Every live-LLM task includes `MODEL_SPECS` with at least one mandated SOTA
  GGUF model.
- Every gated task includes matching `gated_on` metadata and upstream required
  artifact fields.
- At least one experiment (`exp3243`) directly advances continuous
  self-learning under FR-11 without claiming foundation-model weight updates.
- No task modifies `scripts/research_conductor.py`, no task modifies
  `research-roadmap.yaml`, and no task pushes.

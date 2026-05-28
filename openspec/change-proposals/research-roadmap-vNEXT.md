# Research Roadmap vNEXT - Milestone 2026.05.301

**Title:** Selected-Python CUDA Repair + Constraint-Tax Prompt Injection + Lifelong FR-11 Retention

**Created:** 2026-05-28
**Status:** Proposed, staged in `research-roadmap-next.yaml`
**Supersedes:** Milestone 2026.05.300
**Execution queue:** `exp3246` through `exp3258`

## What Milestone 2026.05.300 Proved

Milestone `.300` completed and narrowed Carnot's next blocker:

- `exp3233` archived `.299` and activated `.300`.
- `exp3234` produced the CLI backend root-cause ledger and confirmed the
  failed prompt-injection monolith should stay split into smaller artifacts.
- `exp3235` produced the CUDA boundary package, but `exp3236` reported
  `cuda_python_smoke_passed=false` with
  `selected_python_torch_cuda_unavailable` and
  `cuda_bindings_runtime_no_devices`.
- `exp3237`, `exp3238`, `exp3240`, `exp3241`, and `exp3242` were correctly
  gate-blocked or skipped because the local SOTA receipt chain was not ready.
- `exp3239` produced the Prompt-Injection KAN v4 resource manifest.
- `exp3243` produced the FR-11 failure-memory controller artifact with held-out
  replay and doomed-rerun avoidance evidence, while preserving the
  no-foundation-weight-update boundary.
- `exp3245` reported `paper_ready=false`, `publication_blocker_count=106`,
  `local_sota_receipt_status=blocked`, and
  `next_top_gap=repair_selected_python_torch_cuda_before_exp3237`.

The next milestone should therefore focus on a selected-Python CUDA repair
attempt, then reopen local SOTA receipts and downstream prompt-injection work
only through structured gates. It should also convert FR-11 memory from a
single controller artifact into a lifelong retention/adaptation/forgetting
audit.

## Three Biggest Gaps To PRD Vision

1. **Local SOTA authority is still blocked at the selected-Python CUDA layer.**
   The PRD requires current local GGUF model evidence, but Carnot cannot yet
   make a trustworthy headline claim because selected Python cannot initialize
   CUDA consistently. This blocks live SOTA receipts, teacher labels, and
   structured repair evaluation.

2. **Prompt-injection KAN work has a manifest but no current teacher labels or
   constraint-tax control.** `.300` proved the manifest can be built, but
   downstream labeling and training were skipped. New 2026 work on constraint
   tax means `.301` must compare free-reasoning and schema-constrained arms
   before trusting structured labels or DCCD-style proposals.

3. **Continuous self-learning is controller-memory only.** `.300` showed that
   failure memory can avoid doomed reruns, but PRD FR-11 needs longer-lived
   evidence: retention across sessions, adaptation to new gate blocks, and
   negative-control checks for forgetting without claiming foundation-model
   weight updates.

## External Research Integrated

The 2026-05-28 post-`.300` sweep was added to `research-references.md` before
this roadmap was designed. The most relevant updates are:

- Constraint Tax (`https://arxiv.org/abs/2605.26128`) warns that formalized
  output constraints can reduce reasoning accuracy even while improving
  parseability.
- ConstrainPrompt (`https://huggingface.co/papers/2603.25111`) gives a cheap
  prompt-only constraint baseline for format, lexical, structural, and
  syntactic controls.
- SEVerA (`https://arxiv.org/abs/2603.22471`) supports verifier-guided
  adaptive test-time search as a proposal-ranking idea, not a correctness
  authority.
- LifelongAgentBench (`https://arxiv.org/abs/2605.05135`) motivates explicit
  retention, adaptation, and forgetting metrics for FR-11.
- P-Dit probabilistic units (`https://arxiv.org/abs/2506.00269`) suggest a
  multi-state probabilistic computing diagnostic path for partial-credit rows.
- Extropic and Kona updates remain strategic alignment signals for
  energy-based inference and hardware-accelerated sampling, but `.301` makes
  no TSU, Kona, or hardware speedup claim.

## SOTA Local GGUF Policy

Any `.301` experiment that invokes an LLM for evidence must include at least
one mandated local SOTA GGUF model in `MODEL_SPECS`:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

Legacy small models may appear only as fast CPU smoke tests. They cannot
populate headline result fields and cannot unblock clean verifier, repair, or
publication-readiness claims.

## Architecture Diagram

```text
                 .300 terminal state
  capstone_v300: paper_ready=false, blockers=106,
  local_sota_receipt_status=blocked,
  next_top_gap=repair_selected_python_torch_cuda_before_exp3237
                              |
                              v
              exp3246 archive + activate .301
                              |
                              v
              exp3247 selected-Python CUDA root-cause surgery
                              |
                              v
              exp3248 isolated selected-Python CUDA smoke
                              |
                              v
              exp3249 llama.cpp CUDA receipt smoke
                              |
                              v
              exp3250 mandated SOTA GGUF receipt v8
                              |
        +---------------------+----------------------+
        |                                            |
        v                                            v
 exp3252 teacher-label shard                 exp3254 DCCD/SEVerA
    ^        |                                structured proposal preflight
    |        v
 exp3251 -> exp3253 prompt-injection
 constraint  KAN train/eval shard
 tax plan

 exp3255 FR-11 lifelong failure-memory retention audit
 exp3256 p-dit/Potts multi-state sampler diagnostic manifest
 exp3257 matrix v34 -> exp3258 capstone v301
```

## Phase Plan

### Phase 1 - Archive and Selected-Python CUDA Repair

- `exp3246` archives `.300`, records the capstone blockers, and activates
  `.301`.
- `exp3247` performs selected-Python CUDA root-cause surgery using current
  artifacts and live environment probes, then emits a repair ledger and
  `next_smoke_allowed`.
- `exp3248` runs only if `exp3247` allows it and tests the selected Python,
  PyTorch CUDA, CUDA bindings, and minimal kernel/runtime boundary.

### Phase 2 - Runtime Receipt Chain

- `exp3249` runs only if `exp3248` reports `cuda_python_smoke_passed=true` and
  produces a llama.cpp CUDA/offload receipt.
- `exp3250` runs only if `exp3249` reports
  `llama_cpp_cuda_receipt_ready=true` and attempts the mandated SOTA GGUF
  receipt over the three required local models.

### Phase 3 - Prompt-Injection Constraint-Tax Split-Runs

- `exp3251` refreshes the Prompt-Injection KAN v4 manifest with a
  constraint-tax control plan and ConstrainPrompt baseline.
- `exp3252` runs only if the SOTA receipt and refreshed manifest are ready. It
  creates a small teacher-label shard with free-reasoning and
  schema-constrained arms.
- `exp3253` trains and evaluates the KAN sidecar only if teacher labels exist,
  and explicitly marks the result as a shard-level, non-headline metric.
- `exp3254` runs DCCD/SEVerA-style structured proposal preflight only behind
  the clean SOTA receipt gate, with exact verifier authority preserved.

### Phase 4 - Continuous Self-Learning and Hardware-Aligned Diagnostics

- `exp3255` is the required continuous self-learning experiment. It converts
  the FR-11 failure-memory controller into a LifelongAgentBench-style
  retention/adaptation/forgetting audit over held-out traces.
- `exp3256` maps p-dit/Potts multi-state probabilistic units to Carnot
  partial-credit rows as a diagnostic manifest. It keeps exact fallback and
  makes no hardware speedup claim.

### Phase 5 - Aggregation and Capstone

- `exp3257` aggregates all available `.301` artifacts into cross-corpus matrix
  v34, including gated skips and publication blockers.
- `exp3258` produces the `.301` capstone, determines whether blockers
  decreased, and names the next top gap.

## Dependency Graph

```text
exp3246
  -> exp3247
      -> exp3248 [gate: next_smoke_allowed == true]
          -> exp3249 [gate: cuda_python_smoke_passed == true]
              -> exp3250 [gate: llama_cpp_cuda_receipt_ready == true]
                  -> exp3252 [also requires exp3251.v4_manifest_v2_ready == true]
                  -> exp3254 [gate: clean_rerun_allowed == true]

exp3251 -> exp3252 -> exp3253
exp3255 reads exp3243 and .295-.301 failure traces
exp3256 independent diagnostic manifest after exp3246
exp3257 reads all available .301 artifacts
exp3258 reads exp3257 and all available .301 artifacts
```

## Hardware Requirements

- **Required for Phase 1 and Phase 2 success:** one visible NVIDIA GPU,
  selected Python CUDA initialization, PyTorch CUDA visibility, CUDA bindings
  runtime visibility, and llama.cpp CUDA/offload support. If any of these fail,
  downstream live SOTA tasks skip through structured gates.
- **Required for live LLM tasks:** cached local GGUF files or resolvable
  Hugging Face access for `unsloth/Qwen3.6-35B-A3B-GGUF`,
  `unsloth/gemma-4-31B-it-GGUF`, and
  `unsloth/gemma-4-26B-A4B-it-GGUF`. Any task that invokes an LLM must record
  `MODEL_SPECS`, `preconditions_checked`, and `models_used`.
- **Allowed diagnostics:** CPU-only p-dit/Potts mapping and THRML/Kona
  literature alignment notes with no hardware access or speedup claim.
- **Not required for `.301`:** KV260, GateMate, PolarFire, Extropic TSU, or
  Kona hardware. Do not use a host `/dev/mmcblk*` KV260 precondition; any future
  KV260 work uses SSH reachability and board-side `xmutil`.

## Experiment Queue

1. `exp3246-archive-v300-activate-v301`
2. `exp3247-selected-python-cuda-root-cause-surgery-v1`
3. `exp3248-isolated-cuda-selected-python-smoke-v2`
4. `exp3249-llama-cpp-cuda-receipt-smoke-v3`
5. `exp3250-sota-gguf-receipt-v8`
6. `exp3251-prompt-injection-v4-constraint-tax-manifest-v2`
7. `exp3252-prompt-injection-teacher-label-shard-v2`
8. `exp3253-prompt-injection-kan-train-eval-shard-v2`
9. `exp3254-dccd-severa-structured-proposal-preflight-v2`
10. `exp3255-fr11-lifelong-failure-memory-retention-audit-v1`
11. `exp3256-pdit-potts-multistate-sampler-diagnostic-v1`
12. `exp3257-cross-corpus-matrix-v34`
13. `exp3258-capstone-v301`

## Done Criteria

- `research-roadmap-next.yaml` validates against roadmap schema,
  prior-failure discipline, exclusion-manifest lint, and gate audit.
- Every live-LLM task includes `MODEL_SPECS` with at least one mandated SOTA
  GGUF model.
- Every gated task includes matching `gated_on` metadata and upstream required
  artifact fields.
- Every compute-bound task records `preconditions_checked`,
  `inference_substrate`, `random_seed`, and `reproducibility_checksum`.
- At least one experiment (`exp3255`) directly advances continuous
  self-learning under FR-11 without claiming foundation-model weight updates.
- No task modifies `scripts/research_conductor.py`, no task modifies
  `research-roadmap.yaml`, and no task pushes.

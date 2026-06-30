# Research Roadmap vNEXT - 2026.06.464 - Power the LoRA-EBM signal, repair the confirmation axes, and start verifier self-learning

**Milestone:** 2026.06.464
**Planner:** Codex GPT-5, 2026-06-30 (UTC)
**Prior milestone:** 2026.06.463 (PHASE D third execution)
**Theme:** .463 finally produced the first real trained-verifier number: a clean LoRA-EBM selector at
0.665 vs genuine tuned self-consistency 0.585 on MuSR (+0.080), but the confidence interval touched
zero. The scalar uPRM arm went negative, the uncertainty wrapper matched D1 without improving it, and
the two confirmation axes still did not execute (`blocked_judge_server`, `blocked_second_corpus_unavailable`).
.464 therefore stops treating the moat as an infra-only rescue and asks the next scientific question:
does the +8pp signal survive power, SOTA local GGUF candidates, a repaired cross-model cascade, and a
solver-backed second corpus? It also adds the first verifier-trace self-learning loop required by FR-11.

---

## 1. What .463 proved

| Area | .463 result | Read for .464 |
|---|---|---|
| D1 LoRA-EBM | `complete_lora_ebm_no_win_musr_plus_0p080_ci_incl_0`; 0.665 vs 0.585, CI `[0.0, 0.165]`, McNemar p=0.076369 | Real signal, underpowered. Power it and add margin/uncertainty shaping. |
| D2 uPRM | `complete_uprm_no_win_musr_minus_0p110_mcnemar_or_headroom_gate` | Scalar logprob/uPRM should not be repeated as-is. Replace with dense process rewards/VPR. |
| D3 EBRM | `complete_ebrm_no_win_musr_plus_0p080_ci_incl_0` | The wrapper did not beat D1. Try a calibrated KAN/FIS or PURM readout over the powered scorer. |
| D6 cascade | `blocked_judge_server` | Confirmation axis missing. Add SOTA GGUF judge preflight and cross-model fallback. |
| D4 second corpus | `blocked_second_corpus_unavailable` | Confirmation axis missing. Build the second corpus first, preferably PPBench/solver-backed. |
| KV260 | `success_kv260_reachable_overlay_loaded_energy_ok` | Board path is alive. Next evidence should be timing/ratio parity for p-bit style workloads. |
| Self-play | checkpoint refreshed, no new ARC level bank | Continuous self-learning needs to leave ARC-only dry runs and consume verifier traces. |

The .463 capstone correctly stayed `moat_execution_incomplete`: a real positive margin appeared, but
it did not satisfy the falsifiable gate because it was not statistically decisive and the cross-corpus
and cascade confirmations were blocked.

## 2. The three biggest gaps to the PRD vision

1. **Evidence gap: the verifier moat is suggestive, not decisive.** The PRD wants verifiable constraint
   reasoning that improves local reasoning systems. .463's +8pp D1 result is the first credible sign,
   but it is still MuSR-scoped and CI-touching-zero.
2. **Confirmation gap: the independent axes are still missing.** The architecture needs domain
   transfer and compute/latency Pareto evidence. D4 and D6 did not run, so the current proof is
   single-corpus and single-arm.
3. **Learning-loop gap: FR-11 is not yet closed.** The system has self-play checkpoints and verifier
   artifacts, but it does not yet convert verifier near-misses into a verified, online/self-learning
   improvement loop with contamination controls.

## 3. Fresh research incorporated before experiment design

The `.464` planning sweep updated `research-references.md` with 2025-2026 references and hooks. The
most actionable additions are:

- REVES (arXiv:2606.18910) and Reliable Self-Improvement by Verifying Reasoning (arXiv:2603.21558):
  use verified near-miss reasoning traces for self-learning, not final-answer correctness alone.
- Pencil Puzzle Bench (arXiv:2603.02119 plus `approximatelabs/pencil-puzzle-bench`): a deterministic,
  solver-checkable second corpus for constraint-reasoning verification.
- ETS (arXiv:2601.21484, OpenReview): energy-guided test-time scaling and importance sampling are the
  right comparator family for D1/D3 reranking.
- DISC (arXiv:2606.21724): repair D6 with explicit cross-model verify/judge/correct loops.
- ORLA (arXiv:2606.29366) and Formalize, Don't Optimize (arXiv:2605.12421): favor solver-backed
  formulation/candidate generation over unverified LLM heuristics for the second corpus.
- KANFIS (arXiv:2602.03034): use interpretable KAN/fuzzy readouts for uncertainty calibration.
- FPGA p-bit paper (arXiv:2606.25313): local hardware claims should be modest timing-ratio/parity
  packets, not scale claims.

## 4. Architecture and dependency graph

```
                         exp5042 PHASE 0
            archive .463, activate .464, record close-state
                                      |
            +-------------------------+-------------------------+
            |                                                   |
   exp5043 B1 SOTA GGUF + judge preflight           exp5044 B2 second corpus cache
   - mandated local GGUF availability               - PPBench/solver-backed or fallback
   - top_logprobs/judge endpoint                     - SOTA candidate rows + genuine SC
            |                                                   |
            | sota_models_ready / sota_judge_ready              | second_corpus_cache_built
            v                                                   v
   exp5045 D1 powered EORM/LoRA-EBM                 exp5049 D4 second-corpus confirmation
   - MuSR n>=400 or all cached                       - best powered arm on second corpus
   - SOTA candidate pool
            |
            | powered_scorer_available
            +--------------------+--------------------+
                                 |                    |
                       exp5047 D3 KAN/PURM       exp5046 D2 VPR/ProcessThinker
                       calibration readout        dense process reward repair
                                 |                    |
                                 +---------+----------+
                                           |
                 exp5048 D6 cross-model cascade (also gated on exp5043 judge)
                                           |
                                           v
                           exp5050 D5 moat gate resolution

   Parallel continuity / reserved:
     exp5051 FR-11 verifier-trace self-learning (REVES/VSI)
     exp5052 KV260 p-bit timing-ratio parity packet
     exp5053 SOTA ingestion for .465
     exp5054 opportunistic ARC live-path self-discovery
                                           |
                                           v
                                  exp5055 capstone
```

## 5. Phases

### Phase 0 - Transition

- **exp5042:** archive .463 -> activate .464; record the actual .463 state: real D1 +8pp, D2 negative,
  D3 no improvement, D4/D6 blocked, KV260 live, ARC no-bank.

### Phase B - Preflight and data repair

- **exp5043:** preflight the mandated local SOTA GGUF models and repair the brittle judge-server axis:
  `unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
  `unsloth/gemma-4-26B-A4B-it-GGUF`. At least one must be usable for headline evidence; the judge path
  must support top-logprob/confidence or a documented offline fallback.
- **exp5044:** build the second-corpus candidate cache before D4 runs. Prefer PPBench/solver-verified
  constraint puzzles; fall back only to already cached GPQA/MMLU-Pro style data with genuine SC,
  oracle@K, and solver/verifier labels.

### Phase D - Verifier moat

- **exp5045:** power D1. Rerun the LoRA-EBM/EORM-style energy selector with a larger MuSR candidate set,
  SOTA GGUF candidate generation, energy margins, uncertainty telemetry, and paired statistics.
- **exp5046:** replace the failed scalar uPRM axis with VPR/ProcessThinker-style dense process rewards
  or rollout process labels.
- **exp5047:** calibrate the powered D1/D3 signal with a KAN/FIS or PURM readout; this must improve over
  D1, not just tie it.
- **exp5048:** repair D6 with a DISC-style cross-model cascade and explicit generator/judge separation.
- **exp5049:** confirm the best powered arm on a second corpus. No second corpus means the task blocks
  quickly via `gated_on`, not after a full agent call.
- **exp5050:** resolve the moat gate: realized, bounded-retired, MuSR-scoped, or execution-incomplete.

### Phase E/C - Continuity and reserved research slots

- **exp5051:** FR-11 continuous self-learning from verifier traces. Convert near-misses into verified
  revision/training examples, run a small update, and report held-out delta with contamination controls.
- **exp5052:** KV260 continuity: p-bit/timing-ratio parity packet for the local overlay.
- **exp5053:** reserved SOTA ingestion for .465.
- **exp5054:** opportunistic ARC live-path self-discovery. Any solve claim must be live-agent provenance,
  not offline source-reading or per-game BFS.
- **exp5055:** capstone and .465 pointer.

## 6. Falsifiable gates

- **Moat realized:** at least one oracle-distinct verifier arm beats genuine tuned self-consistency on
  MuSR with CI excluding zero, and either confirms on the second corpus or D6 reaches accuracy parity at
  materially lower judge cost.
- **MuSR-scoped positive:** D1/D3 remains positive on MuSR but D4 or D6 is still blocked or negative.
  This is progress, not a moat claim.
- **Bounded retirement:** only if properly executed powered D1 plus the repaired D2/D3 arms clean-null
  on headroom-present data and D6 has no efficiency win.
- **Execution incomplete:** any blocked SOTA preflight, blocked second corpus, blocked judge, skeleton
  training, degenerate abstention, or missing statistics.

## 7. Hardware requirements

- **Dual RTX 3090 CUDA path:** SOTA GGUF local inference, candidate generation, judge/cascade scoring,
  and any LoRA/EORM training. Do not iGPU-pin offline PHASE D runs.
- **KV260 over SSH (`ssh kria`):** board-only overlay/timing checks. Never host SD-card operations.
- **GateMate/PolarFire:** not required in .464 unless exp5052 finds an immediate parity extension.

## 8. Model policy

Every experiment that uses an LLM must include at least one mandated SOTA local GGUF model in
`MODEL_SPECS`:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

Legacy small models may appear only as CPU smoke tests and must not be headline-result models.

## 9. Experiment order

1. exp5042 transition
2. exp5043 SOTA/judge preflight
3. exp5044 second-corpus cache
4. exp5045 powered D1
5. exp5046 repaired D2
6. exp5047 D3 calibration
7. exp5048 D6 cascade
8. exp5049 D4 second-corpus confirmation
9. exp5050 gate resolution
10. exp5051 continuous self-learning
11. exp5052 hardware continuity
12. exp5053 SOTA ingestion
13. exp5054 ARC live-path continuity
14. exp5055 capstone

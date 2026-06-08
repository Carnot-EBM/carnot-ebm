# Step-level process-reward: the rescue path that uses the verifier where it's strong

**2026-06-07.** Phase 0 found Carnot's verifier is a strong PROCESS verifier (96.7%
per-step precision in-format) but a weak OUTCOME certifier (56% trace-level) — so
trace-level RFT poisons. This path uses the verifier the way it's actually strong:
as a **dense per-step process reward**, not a trace-level pass/fail filter.

## The idea

Instead of "keep whole traces the verifier certifies correct" (failed: process !=
outcome), give the generator a **dense per-step reward** from the verifier and train
it to produce more locally-valid reasoning — the standard process-reward-model (PRM)
recipe (Lightman et al. "Let's Verify Step by Step"; the FoVer line). The 96.7%
per-step precision is exactly the signal a PRM needs.

Mechanism options (simplest first):
1. **Process-reward-weighted SFT** — weight each generated trace (or each step's
   contribution) by its verifier process-reward; up-weight high-reward reasoning.
   Cheapest; no RL.
2. **Best-prefix distillation** — keep the longest verifier-certified-correct PREFIX
   of each trace and train on it (extract the maximal-valid portion).
3. **Step-level RL (process-reward PPO/GRPO)** — per-step reward in an RL loop.
   Heaviest; the full PRM-as-reward.

## The central risk (must de-risk before building)

**Process-reward hacking / process != outcome, as a SOFT signal.** Phase 0 already
showed local step-validity only weakly predicts outcome (trace-level 56%). So
rewarding per-step validity risks producing **locally-valid-but-unproductive**
reasoning — valid-looking steps that don't progress to the right answer. The PRM
literature shows a STRONG PRM improves outcomes, but a weak/miscalibrated one can be
gamed. The question is whether Carnot's per-step signal is strong ENOUGH, as a dense
reward, to move outcome accuracy.

## The cheap de-risk (Phase 0.5 — gates the build, runnable NOW)

Before any training, measure whether the verifier's aggregated per-step process-reward
**ranks correct traces above incorrect ones** — i.e., the trace-OUTCOME AUROC of the
aggregate process-reward, on existing p01 traces (we already have the per-step scores
from the chunking-bridge run). No training, no new infra.

- **GATE:** trace-outcome AUROC of the aggregate process-reward >= ~0.65 (meaningfully
  above chance). If yes, the dense reward carries real outcome signal -> process-reward
  training is worth the harness build. If ~0.5-0.55, the per-step signal does NOT rank
  correct traces above incorrect at the trace level -> process-reward would reward-hack;
  the path needs a stronger/outcome-aware verifier first.

Aggregation variants to report: mean per-step reward, min per-step reward (one bad step
sinks the trace), and fraction-of-steps-certified. The best aggregation's AUROC is the
gate.

## Infra (if the gate passes)

The fine-tune harness is still the gap (no peft/LoRA/RL in the tree). Start with option
1 (process-reward-weighted SFT) on Qwen3.5-0.8B (trainable, cached, the
project-native family; NOT Qwen2.5) — simplest, no RL. Arms: base /
process-reward-weighted / gold-RFT (upper bound) / SC. Headroom corpus, multi-seed,
held-out eval. Only escalate to step-level RL (option 3) if weighted-SFT shows signal.

## Honest framing

This is the most promising rescue because it matches the verifier's demonstrated
strength (per-step), and it's the established PRM-as-reward recipe. But the same
process-vs-outcome tension applies as a soft signal — the Phase-0.5 de-risk measures
whether the soft per-step reward is strong enough at the trace-outcome level before any
build. Same discipline as Phase 0: cheapest decisive measurement first.

---

# Phase 0.5 RESULT (2026-06-07) — de-risk PASSES: dense process-reward ranks outcomes at AUROC 0.73

Trace-OUTCOME AUROC of the aggregated per-step process-reward on 558 free-form p01
traces (6038 chunks, 34% base rate):

| aggregation | trace-outcome AUROC |
|---|---|
| fraction_certified | **0.730** |
| mean_reward | 0.722 |
| min_reward | 0.673 |

**GATE PASSES (best 0.73 >= 0.65).** The dense/soft process-reward carries real outcome
signal -- unlike the HARD trace-certification (56% precision, Phase 0). The difference
is information: all-steps-clean is all-or-nothing (one flagged step kills a mostly-
correct trace), but the soft aggregate (fraction-certified / mean-reward) keeps the
dense per-step signal and ranks correct traces above incorrect at 0.73.

**Conclusion: the step-level process-reward path is VIABLE.** Use Carnot's verifier as a
DENSE per-step reward, NOT a hard trace certifier. This is the actionable resolution of
the Phase-0 process-vs-outcome bound: hard certification fails, soft process-reward
works (0.73 outcome ranking). Caveats: 0.73 is moderate (not the 0.967 in-format per-
step -- process-vs-outcome + OOD attenuation remain), so a process-reward-trained
generator would improve under a moderately-noisy reward (some reward-hacking risk); and
this is correlational (reward ranks outcome) -- the causal lift needs the training run.

**Next (the build, now justified):** process-reward-weighted SFT (option 1) on
Qwen3.5-0.8B (PoC) -> scale to Qwen3.5-9B/27B (headline); arms base / process-reward-weighted / gold-RFT (UB) / SC; multi-seed, held-out
eval; gate on the weighted arm beating base. The fine-tune harness is the remaining
infra to build. This is the concrete forward experiment.


## Base-model note (2026-06-07)

For FINE-TUNING we need trainable HF safetensors, NOT the GGUF SOTA models. The SOTA
`unsloth/Qwen3.6-35B-A3B-GGUF` is GGUF-only (llama.cpp inference) -> NOT fine-tunable;
it is the verifier/inference-side SOTA, not a trainable generator base. Cached TRAINABLE
options: Qwen3.5 {0.8B,2B,4B,9B,27B} (the project's own per-token-EBM family) and
gemma-4-{E2B,E4B}. Plan: PoC on Qwen3.5-0.8B (recent, trainable, cached -- NOT the
outdated Qwen2.5), scale to Qwen3.5-9B/27B (LoRA on the 2x3090). To use literally Qwen3.6
as the generator, download its non-GGUF base and LoRA the 35B-A3B MoE (heavier; do only
after the mechanism is proven on Qwen3.5).

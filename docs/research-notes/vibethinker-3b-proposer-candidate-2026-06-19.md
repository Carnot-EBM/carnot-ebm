# SOTA ingestion: VibeThinker-3B as a candidate ARC induce/refactor proposer (2026-06-19)

**Date:** 2026-06-19 · **Trigger:** operator handed the model card for ingestion.
**Source (VERIFIED):** WeiboAI/VibeThinker-3B, https://huggingface.co/WeiboAI/VibeThinker-3B
(fetched 2026-06-19). GGUFs exist: **Q4_K_M = 1.93 GB** (`oussaber/VibeThinker-3B-Q4_K_M-GGUF`),
Q8_0 = 3.29 GB (`JohnRoger/...`, `bms22/...`), Q2_K..F32 (`prithivMLmods/VibeThinker-3B-GGUF`).

## What it is

A **3B**, **MIT-licensed** reasoning model fine-tuned from **Qwen2.5-Coder-3B** via the SSP recipe
(curriculum SFT with Diversity-Exploring Distillation → MaxEnt-Guided Policy Optimization across
math/code/STEM, 64K ctx → offline self-distillation → instruct RL), with a test-time
Claim-Level-Reliability self-assessment (CLR).

**Headline:** IMO-AnswerBench **76.4% (80.6% with CLR)** — "approaching DeepSeek V3.2 (671B), GLM-5
(744B), Kimi K2.5 (1T)"; LeetCode **96.1% first-attempt acceptance** (123/128). A 3B rivaling 671B–1T
on *verifiable* math/code.

## Why it's relevant to the ARC live generator (it may beat Qwen3.5-9B-MTP at the proposer role)

Carnot's design: the LLM is a swappable **proposer** that writes one `world_model.py` / win-predicate
per prompt; the Carnot harness (`StepwiseExplorer` + verifier + search) IS the agent. So we need a
strong small **coder/reasoner on a focused, verifiable task**, not an agent.

| Factor | Implication |
|---|---|
| **3B (~1.93 GB Q4)** vs Qwen3.5-9B (5.9 GB) | far more 16 GB-Kaggle KV/context headroom; possibly faster overall |
| **MIT license** | more permissive than our Apache-2.0; aligns with the MIT-0 ARC-Prize relicense |
| **Qwen2.5-Coder base, 96% LeetCode** | the E3 induce/refactor step IS Python-code writing — its core strength |
| **Diversity-Exploring Distillation (multiple solution paths)** | good for **propose-then-ground**: diverse candidates → higher chance one grounds |
| verifiable-domain reasoner + external verifier | matches ARC config-rule induction (verifier grounds the output) — the FinAcumen/small-model-rivals-frontier pattern, third corroboration |

**Its stated limitation is the part our design removes.** The card warns it is "not trained on
tool-calling or agent-based programming... unsuitable for autonomous agents." That sounds
disqualifying — but we never use the LLM agentically; we use it as a focused code/rule generator
inside a harness that is the agent. Its weakness is exactly the job our architecture takes away from
the LLM.

## The trade-off to MEASURE (do NOT assume it wins)

- It is a **long-CoT "thinking" model** (max_new_tokens up to 102,400, temp=1.0, top_p=0.95). We run
  Qwen3.5-9B-MTP with **`/no_think`** for fast, direct, deterministic code. VibeThinker's long
  reasoning + high temperature could be **slow + less deterministic** for the fast per-step
  induction, and it has **no MTP** speculative decoding. Stronger per-call reasoning, possibly fewer
  calls within the 8 h / 10-steps-per-sec budget.
- math/code/STEM focus, **not broad world-knowledge** — fine for verifiable induction; no help for
  open-ended perception/exploration.

## Benchmark spec — flagged for `.412 (head-to-head proposer eval)

**PRECONDITIONS:** fetch `oussaber/VibeThinker-3B-Q4_K_M-GGUF` (1.93 GB) to the HF cache; the live
generator GGUF Qwen3.5-9B-MTP already cached. Run on the iGPU or the now-available 3090
(`CARNOT_ARC_GENERATOR_CUDA_GPU`); NEVER fabricate.

**Method:** load EACH model via `LocalGGUFProposer` (VibeThinker with a thinking budget; Qwen with
`/no_think`+MTP) and run the SAME E3 induce/refactor step (write `world_model.py` / config-rule
predicate) on a fixed set of solved + held-out config games. **Metrics (all three, not just
accuracy):** (1) **grounding rate** — % of proposed predicates/models the verifier grounds; (2)
**events/actions-to-solve** + reproduction-gated `reproduced_levels`; (3) **wall-time per induction**
+ tokens. Decision: VibeThinker replaces/augments Qwen as the proposer ONLY if it grounds ≥ as well
at ≤ the wall-time (or a clearly better grounding-per-second). `inference_substrate=live_llm_inference`
(real model load). `verifier_is_oracle=true`.

**flagged_for_v412:** the proposer head-to-head above. If VibeThinker wins, it is a strictly better
Kaggle generator (3× smaller, MIT, more KV headroom). The LLM is swappable (the verifier is the moat),
so this is low-risk / high-upside.

## Honest caveats
- A candidate **proposer swap**, NOT an architecture change. The verifier + harness are unchanged.
- Long-CoT slowness is a real risk against the live time budget — the benchmark must weight
  grounding-per-second, not raw grounding.
- "Approaching 671B–1T" is on *math* benchmarks; ARC induction is a different distribution — the
  benchmark measures ARC grounding directly, not transfer from IMO scores.

## Cross-refs
- `feedback_sota_ingestion_cycle` (memory) · CLAUDE.md "SOTA-Ingestion Cycle Discipline" + "SOTA Local Models"
- [[project_arc_live_generator]] (the current Qwen3.5-9B-MTP pick) · [[project_kaggle_16gb_gemma12b]] (the 16 GB constraint)
- `docs/research-notes/finacumen-experience-memory-ingestion-2026-06-19.md` (sibling: small-model-rivals-frontier)
- `python/carnot/agentic/arc_executable_world_model.py:LocalGGUFProposer` (the swappable proposer)

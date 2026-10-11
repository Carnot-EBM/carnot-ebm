# Limite 1B "Violetto": review for Carnot (2026-10-10)

Operator question: what can we learn from, or how can we use,
https://huggingface.co/paradigma-inc/limite-1b-violetto.

Basis: the model card only. I did not run the model, read its code, or check any claim.
No paper is linked. A fetch tool summarized the page, so quotes below are from that summary.

## What the card says

| Item | Card claim |
|---|---|
| Model | 1B dense autoregressive transformer, built for math reasoning |
| License | Apache-2.0 (weights and vLLM serving code) |
| Training | From scratch on under 300B curated tokens, then synthetic data, SFT and RL |
| Architecture | "Strongly inspired" by pre-training speedrun work. Hybrid local/global attention. No layer or width numbers |
| Context | Up to 131k tokens |
| Headline result | 74.25% average on BeyondAIME, against 70% for a 30B model |
| Intended use | Single-turn math. Not a general assistant. It can "lose the scope of a prompt" and turn it into a math task |
| Serving | vLLM through a GitHub plugin. Transformers validated on an H100 only |

## What is not supported by the card

- The benchmark table is an image. Some scores come from other model cards or MathArena and were not re-run.
- "Rivals models tens of times larger" has no FLOP counts.
- The data sources, the RL setup and the "canonical system prompt" are not described.
- No ablation or training curve backs the sample-efficiency claim.

## Possible uses for Carnot

1. **Math candidate generator for verifier tests.**
   - The verifier works on math (FoVer 0.9131).
   - The energy-selection tests lost to self-consistency (SC). SC was already near-optimal, so no method had headroom.
   - A 1B model makes more errors and is cheap to sample many times. That may give a corpus where the oracle beats SC.
   - That is the positive control the false-negative rule requires before any null result counts.
   - It would be a comparator. The mandated headline model stays `unsloth/Qwen3.8-27B-GGUF`.
2. **Cheap candidate pools** for the calibrated-decision floor, where many samples per problem are needed.
3. **Method hint, unverified.** Curated data and a speedrun-style design may matter more than scale at this size. We cannot check this from the card.

## Not useful for

- ARC. The model drifts off the prompt and is math-only.
- General instruction following.

## Costs and risks

- The vLLM plugin is custom code from GitHub. The "audit untrusted code" rule applies before any run.
- Loading through transformers needs `CARNOT_TRUST_REMOTE_CODE=1`.
- A new architecture probably has no llama.cpp or GGUF path. That weakens the local-first claim. I did not check for one.
- Transformers inference is validated on an H100 only. Our RTX 3090s are untested.
- Training and LoRA are not validated by the authors.

## Recommended test (not started)

Start only after the ARC induction run frees card 1.

1. Audit the plugin code.
2. Serve the model on card 1 and sample 8 to 16 answers per problem on a math corpus we already use.
3. Gate: the oracle (any sample correct) must beat SC by a clear margin, with the sample size stated in advance.
4. If the gate fails, stop. The model gives no headroom.
5. If it passes, run the verifier ensemble against SC on the same pools, with the artifact fields the fabrication gate requires.

## Decision

No integration now. This note records the idea and the gate so the next reader does not re-derive them.

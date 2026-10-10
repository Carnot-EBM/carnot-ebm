# Dust: pretraining transformers without backpropagation (read 2026-10-10)

Source: Dahal, Mandal, Gülbahar, Vegesna, "Dust: Pretraining Transformers Without
Backpropagation", Q Labs Research, October 2026. https://qlabs.sh/research/dust
(appendix: https://qlabs.sh/research/dust/appendix.html). Code (MIT):
https://github.com/qlabs-eng/dust. There is no arXiv or PDF version. The page is the paper.

I read the full page, the full appendix and the repo README as raw text. I did not run the
code. No claim here is reproduced by us.

## What the method does

- Dust adds Gaussian noise to each linear layer's output, separately at every token.
- Each token's noise gets a reward: the loss reduction at that token and, with a decay, at the
  later tokens. The reward-weighted noise estimates the error at the layer output.
- That estimate times the layer input gives the weight gradient. Backprop forms the same
  product. Only the source of the output error differs.
- One forward pass tests thousands of perturbations, one per token. The paper calls this a
  "virtual population".
- The key input is a dense loss at every token.

## What the numbers show (test loss, three seeds, 8 layers, width 512, 37.7M parameters)

| Tokens | Dust at 16k draws | Backprop | Dust minus backprop |
|---|---|---|---|
| 100k | 7.170 | 7.216 | -0.046 |
| 1M | 5.916 | 5.959 | -0.043 |
| 10M | 5.049 | 4.989 | +0.060 |
| 20M | 4.802 | 4.633 | +0.169 |

- Dust beats backprop only at 100k and 1M tokens. At 10M and 20M it is behind, and the gap
  grows with the token budget.
- The claim "below backprop at 20M" is a power-law limit: 4.431, 95% interval 3.89 to 4.58. The
  authors write that it is "loosely constrained" and not "a measured limit". At 10M the fitted
  limit, 5.013, is above backprop's 4.989.
- Dust at 256 draws is worse than backprop at 1M tokens (6.067 against 5.959). It needs about
  1k draws to win there.
- The claim of "10^3 to 10^4 times more efficient" than EGGROLL comes from extrapolating
  EGGROLL's ladder. Appendix D gives 3,100 to 23,000 times, and some leave-one-out refits give
  an infinite multiple.
- Model size, 10M tokens, 16k draws, earlier settings (Figure 4 table): Dust 5.171 / 5.065 /
  5.036 / 5.086 for 2M / 7M / 38M / 243M parameters. Backprop 5.180 / 5.066 / 5.015 / 5.048.
  Dust is ahead only at 2M, level at 7M, and behind at 38M and 243M. At 64 draws the 243M model
  (5.719) is slightly worse than the 2M model (5.705).
- Cost: the base 16k run uses 8 GPUs (README). The authors say they do "not attempt to make it
  compute-efficient enough to replace backprop today".

## Correction to an earlier chat summary

An earlier summary of the page, made through a summarizer, said Dust got 5.036 and backprop
5.048 at 10M tokens. That was wrong. 5.048 is backprop at 243M parameters. At the base model
size, Dust is 0.021 behind in the model-size table and 0.060 behind in Table 1.

## What this means for Carnot (my reading, not the paper's)

1. No adoption now. The compute per step is orders of magnitude above backprop, and the best
   cells need 8 GPUs. We have two RTX 3090 cards.
2. The speed-up comes from dense per-token loss. A verifier gives one score per answer. Dust's
   per-token credit would not carry over, so the gain over plain evolution strategies likely
   shrinks. This is untested.
3. Our small selector models (GibbsModel with NCE, in JAX) have exact gradients. Dust adds cost
   there with no benefit.
4. The hardware angle is weak. Dust removes the backward pass, but still needs a bf16
   transformer forward on a GPU. It does not help an Ising or FPGA sampler.
5. A cheap check exists if we ever want it. The repo runs 1M tokens at 1,024 draws on one GPU.
   The paper claims a win over backprop at that size (5.934 against 5.959). It is a 0.025
   difference over three seeds, so a reproduction needs several seeds.

## Status

Recorded as a reference only. No task is queued.

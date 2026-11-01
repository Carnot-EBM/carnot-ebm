# Grassmann flows as an attention-free sequence model: reading note (2026-10-31)

Status: reading note. No experiment was run. Nothing here is a Carnot result.

Source read in full: Zhang Chong, "Attention Is Not What You Need", arXiv 2512.19428 v1 (22 Dec 2025, CC BY 4.0). Text extracted from the PDF with `pdftotext`. One author, one version, no code link found in the text.

## What the method is

The paper replaces self-attention with a local geometric mixing step. Per layer:

1. Reduce each hidden state to r = 32 dimensions.
2. Pair each position t with later positions t + D for D in a fixed window set, for example {1, 2, 4, 8, 12, 16}.
3. For each pair, compute the Plucker coordinates (all 2x2 minors of the two vectors) and L2-normalise them.
4. Project back to model width, then average over valid offsets.
5. Mix with the original state through a learned sigmoid gate, then LayerNorm and a normal feed-forward block.

Cost is linear in sequence length for fixed r and window count. The authors state their implementation is slower per step than optimised attention up to length 256.

## What the paper reports

| Task | Transformer | Grassmann |
|---|---|---|
| WikiText-2, 6 layers, block 128 (val PPL) | 248.4 and 253.6 | 275.7 and 282.3 |
| WikiText-2, 12 layers, block 256 (val PPL) | 235.2 | 261.1 |
| SNLI, DistilBERT backbone, val / test accuracy | 0.8545 / 0.8511 | 0.8550 / 0.8538 |

Model sizes are 12.6 to 18.2M parameters. The language model trains from scratch for 30 epochs.

## Problems in the paper itself

I checked the tables against the text. Four things do not hold.

- **Table 1 and the text disagree.** The text says the Transformer scores "241.0-253.6". Table 1 has no 241.0. The text also pairs Grassmann 282.3 with Transformer 248.4. In Table 1, 282.3 sits beside 253.6. The two Table 1 row pairs have the same labels and differ only in window schedule, which the table does not say.
- **"The gap narrows with depth" is not supported.** Gaps from the table: 275.7 / 248.4 = 11.0%, 282.3 / 253.6 = 11.3%, 261.1 / 235.2 = 11.0% (12 layers). They are flat. The 6-layer and 12-layer runs also differ in block size, so they are not a depth comparison.
- **The SNLI margin is inside noise.** The test gap is 0.27 points on one run per head. SNLI test has about 9,800 examples, so the standard error of one accuracy is about 0.36 points. No seeds or variance are reported.
- **The SNLI backbone is ambiguous.** The text says DistilBERT is "fixed as a feature extractor". It also says both heads train "from the same initialization of the backbone". These two statements do not agree on whether the backbone is frozen.

Perplexity near 250 on WikiText-2 with a 30k WordPiece vocabulary is also not comparable to word-level numbers in other papers. The optimiser and learning-rate schedule are not named.

## Limits the authors state

- Slower than attention at the tested lengths.
- Local windows only. Long-range structure comes from depth.
- "Our aim is not to claim that attention is obsolete."
- Small scale, no hyperparameter tuning.

## What this means for Carnot

### Nothing to build

The paper does not test verification, calibration, search, or any decision problem. It is an architecture proposal at 13 to 18M parameters. Carnot's generator is fixed by mandate (`unsloth/Qwen3.8-27B-GGUF`), and changing it is out of scope. A new sequence mixer cannot reach the live ARC agent or the scored path.

### One loose thread, not a lead

Plucker coordinates give a fixed-size, normalised, algebraically constrained feature of a pair of vectors. That is a possible feature type for a small selector over embeddings. This note makes no claim that it helps. It would need a baseline that uses plain concatenation or cosine of the same pair, and a positive control. Do not start it without a task that needs it.

### The interpretability claim

The paper argues Plucker vectors are easier to analyse than attention maps. It gives no experiment. Treat it as a hypothesis.

## Circularity and rule checks

- No verifier-value claim is made, so the Circularity / Oracle-Distinctness rule does not apply.
- No self-consistency baseline exists, and none would apply to a perplexity or classification-head comparison.
- The paper does not touch the north star, the calibrated-decision floor, or ARC.

## Not checked

- Whether code exists (the TeX source was not fetched).
- Whether anyone replicated the result.
- Whether later versions fix the table mismatch.

## Verdict

Low value for Carnot. An unreplicated, single-author, small-scale result with internal inconsistencies, on a topic outside the open gaps. No experiment follows.

# DeepSeek-OCR and optical context compression: review and gated test plan (2026-10-11)

Operator question: what can we learn from DeepSeek-OCR, mainly how it addresses the AI memory
problem. Source: https://github.com/deepseek-ai/DeepSeek-OCR

Basis: the repository README only (fetched 2026-10-11). Paper details below come from my
memory of arXiv 2510.18234. I did not fetch the paper. Treat them as unverified until read.

## What the README says

- The project studies "Contexts Optical Compression": text is shown to the model as an image,
  so a page costs vision tokens instead of text tokens.
- Vision tokens per input mode: Tiny 64 (512x512), Small 100 (640x640), Base 256 (1024x1024),
  Large 400 (1280x1280), Gundam (tiled).
- It gives one throughput figure: about 2,500 tok/s on an A100-40G with vLLM for PDFs.
- It gives no ratio, no accuracy table, no encoder description, no memory claim, no limits.

## From memory, unverified

- Reported: about 10x compression at about 97% decoding precision, and about 20x at about 60%.
- Memory idea: older context is re-rendered at lower resolution, so it fades step by step.
  The paper discusses this as a possible forgetting curve. It is a proposal, not a result.

## What could carry over to Carnot

| Idea | Carnot problem it touches | Fit |
|---|---|---|
| Lossy tier for old context | The induction prompt is about 62,000 tokens. It overflowed a 106,496-token cache when 4 ran at once | Real problem. The lossy part is the risk |
| Render grids as images | ARC frames are 64x64 grids. They are already images | Natural. But the generator has no vision path in our GGUF setup |
| Graded forgetting | Long runs keep every transition | Same as a summary policy. Can be tested without images |

## Why not now

- ARC needs exact cells. A 20x lossy tier may drop the one cell that matters.
- The shipped generator (Qwen3.8-27B GGUF) is text-only in our setup. Vision needs an mmproj
  and a second model path. That conflicts with the one-generator rule.
- The README gives no evidence on grids or on reasoning over compressed context.

## Cheaper first test (text only)

The same problem has a text-side answer: compact old transitions before the induction call.

1. Measure the induction prompt: what share of tokens is old transitions.
2. Build a compaction (drop duplicate transitions, keep the first and last of each action
   effect). Keep it deterministic.
3. Run the induction on the same games with one cell at a time (`--k 1`).
4. Gate: installed-plan rate with compaction is no worse than without, at a state sample size
   fixed in advance, and the prompt is under half the tokens. If it is worse, stop.

Only if text compaction fails for lack of room should an image tier be considered, and only
after the Clef spike shows whether our serving stack can take images at all.

## Decision

No integration. Read the paper before citing any number here. The text-compaction test above is
the only item worth queueing, and it waits for the `--k 1` induction rerun to give a baseline.

## Cross-references

- `docs/research-notes/arc-heldout-test-scope-2026-10-10.md` (induction run and the overflow)
- `docs/research-notes/clef-system1-decision-api-review-2026-10-11.md` (image support in serving)

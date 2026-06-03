# Phase-3 Thesis A — Part (b) design: does energy-as-GENERATOR beat autoregression at MATCHED COMPUTE?

**Status:** DESIGN / operator-decision doc, drafted 2026-06-03 by the outer-loop
after part-(a) PASSED. Earns part-(b) by clearing the cheap gate.

## Where we are

**Part (a) PASSED** (`results/thesis_a_direct_definitive_run.json`, 2026-06-03): a
tiny 38M byte-EBT trained STABLY on GSM8K (no NaN/divergence, bounded gradients)
AND learned a *generalizing* held-out energy landscape (pos/neg margin 0.723 vs
0.084 untrained, ~8.6x). AR sanity passed (held-out CE 5.71→1.55). So
energy-as-generator is viable at the stability gate — worth building the decoder
and running the real test.

**Part (b) is the actual thesis:** *at EQUAL inference compute, does EBT
energy-descent GENERATION beat autoregressive generation on held-out reasoning
accuracy?* Part-(a) only showed the EBT learns an energy landscape; it did not
show that landscape produces better *generations* than AR per unit compute —
which is the whole Phase-3 claim.

## Problem 1 — the decoder (EBT outputs embeddings, not tokens)

The EBT scores `(context_emb, predicted_emb) -> scalar energy`. To generate and
measure token accuracy we need `energy-minimized embedding -> token`. Options:

- **(A) Nearest-neighbour** in the `token_embedding` table — decode the
  energy-descended embedding to its nearest token vector. Zero new params; a good
  sanity baseline, but the continuous landscape may not land cleanly on table
  vectors.
- **(B) Tied-embedding softmax (RECOMMENDED)** — `logits = emb @ token_embedding.weight.T`;
  decode `argmax`/sample. Parameter-efficient (reuses the table the EBT already
  has), principled, and differentiable so it can be lightly fine-tuned with a
  reconstruction loss if (A) is too lossy.
- **(C) Learned MLP head** `emb -> vocab` — most faithful, most new params/training.

Plan: implement (B) as primary, keep (A) as a control. Only escalate to (C) if
both underperform a trivial decode baseline.

**EBT generation loop:** autoregressively, for each next token — initialise
`predicted_emb` (random or a cheap proposal), run **K Langevin energy-descent
steps** to minimise energy given the context, decode via (B), append, repeat. K
is the "System-2 thinking" budget and the unit of EBT compute.

## Problem 2 — the task regime (the load-bearing design choice)

**Risk:** at 38M–300M scale, full GSM8K is too hard — both AR and EBT score ~0%,
so "accuracy" carries no signal and any "EBT doesn't win" verdict is a FALSE
NEGATIVE (the exact trap from `adversarial_verify.py:FALSE_NEGATIVE_RISK` and the
P0.1 grind: a null result is only meaningful if a positive control had headroom).

**The comparison needs a regime where (i) a small AR model gets measurable,
non-trivial accuracy, and (ii) there is headroom above it.** Recommended:
**multi-digit arithmetic / a controllable synthetic reasoning task** with tunable
difficulty —

- AR's left-to-right commitment is a *known* weakness on long-carry arithmetic
  and backtracking-needed structure; an energy-based GLOBAL view has a plausible
  structural advantage (cf. the "Beyond Autoregression" precedent the loop itself
  cited — global inference beat AR on Sudoku/Countdown).
- Tune difficulty so the AR baseline lands ~40–70% — measurable, with headroom.
- **Mandatory positive control (FALSE_NEGATIVE_RISK guard):** include a split
  where an oracle/optimal solver clearly exceeds the AR baseline. If oracle ≈ AR,
  the corpus has no headroom and NO method could win — a "bounded" verdict there
  is uninformative and must be rejected, not propagated.

GSM8K stays as a secondary, harder probe only — not the primary signal corpus.

## Problem 3 — matched COMPUTE, not matched params (the P0.1 lesson)

Reuse the harness already built in `scripts/experiment_3727_matched_compute_eval_harness.py`:

- EBT cost/token ≈ K energy-descent forward passes × model FLOPs.
- Give AR an **equal total-inference-FLOP budget** — best-of-N sampling or
  self-consistency with N tuned so `AR_FLOPs ≈ EBT_FLOPs` (within the harness's
  tolerance).
- Report held-out accuracy for BOTH **at equal compute**. A win only counts at
  equal inference FLOPs; a params-matched "win" that just spends more passes is
  the P0.1 trap and does not count.

## Kill-gate (part b)

- **PASS:** EBT beats AR on held-out accuracy **at equal inference compute**, the
  positive control confirms headroom (oracle > AR, flips > 0), and the win
  replicates across ≥3 seeds → STRONG Phase-3 signal → scale further.
- **FAIL / BOUNDED:** EBT does not beat AR at equal compute AND the gap does not
  close with 2× training/compute, on a corpus that DOES have headroom →
  energy-as-generator is bounded for reasoning at this scale → honest STOP.
- **REJECTED (not a verdict):** any "bounded" reading on a corpus where the
  positive control fails (oracle ≈ AR) — uninformative, re-run with headroom.

## Scale-up knobs (from the part-(a) 38M baseline)

- Model: 38M → ~100–300M (still single-3090-trainable).
- Tokenizer: byte-level is fine for arithmetic; a small subword tokenizer for
  GSM8K probes.
- Data/steps: large generated arithmetic corpus; train to held-out convergence
  (monitor, don't fix a step count blindly).
- Compute budget: bounded GPU-hours with checkpointing; the now-fixed infra
  (`.venv` on PATH + the inode reaper) means the loop can run training-class
  tasks without the false-negative faults that blocked part-(a).

## Milestone task breakdown (`.344`-class)

1. **Decoder + generation loop** — tied-embedding softmax decode (B) + the
   K-step energy-descent autoregressive generator; unit-tested on the part-(a)
   checkpoint (does it decode to *something* coherent?).
2. **Task corpus + positive control** — generate the tunable arithmetic corpus
   with an AR-difficulty knob and an oracle-vs-AR headroom split.
3. **Scale + train** EBT + matched AR (checkpointed) on the corpus.
4. **Matched-compute comparison** — reuse exp3727; EBT-gen vs AR best-of-N at
   equal FLOPs; held-out accuracy + the positive control; ≥3 seeds.
5. **Part-(b) verdict + kill-gate** — PASS/BOUNDED/REJECTED per above.

## Honest framing

Part (b) is where the thesis lives or dies. A clean negative — EBT can't beat AR
at equal compute even on AR-hostile arithmetic *with* confirmed headroom — would
bound energy-as-generator for reasoning, which is as valuable a result as a win.
The kill-gate + the mandatory positive control make it a disciplined bet, not a
P0.1-style grind. Invariants unaffected: `paper_ready` stays TRUE, frozen FoVer
0.9131 unchanged, P0.1/energy-SELECTION stays honest-negative (this is a
different mechanism — generation, not selection).

## Execution options (operator picks)

- **(i) I build + run it directly**, like part-(a) — I write the decoder +
  generation + comparison harness and drive the training on the rig, for a
  definitive answer without the loop's cadence limits. Best for the tricky
  decoder/generation code.
- **(ii) Pre-stage `.344` as a conductor milestone** — now that the infra is
  fixed, the loop can run it; slower but hands-off.
- **(iii) Hybrid (recommended)** — I build + validate the decoder/generation/
  comparison harness directly (the hard ML), then the scale-train + comparison
  runs as a milestone or a direct run.

**Operator decisions:** (1) task regime — arithmetic (recommended) vs a synthetic
CSP vs GSM8K-only; (2) execution path (i/ii/iii).

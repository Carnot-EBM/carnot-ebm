# Verifier-as-self-improvement-reward: scaling the #4 v4 result to reasoning

**2026-06-07, outer-loop session.** Scopes the forward hook from the retired
energy-as-generator line: the #4 v4 Sudoku result showed a verifier can *teach* a
generator (RFT on verifier-certified self-samples beat gold-SFT on a headroom base).
This is the **Phase-3 / foundation-model self-improvement loop** — the most
strategically load-bearing direction the project has. This doc defines the program
and, crucially, the **cheap decisive de-risk** that must pass before any training.

## The hypothesis

Carnot's verifier ensemble, used as the **correctness filter** in a STaR/RFT loop
(generate K traces -> verifier certifies the correct ones -> fine-tune the generator
on them, NO gold labels), can drive **self-improvement** of an LLM on reasoning —
comparably to gold-RFT — on a corpus with headroom.

## The ONE thing that's different from Sudoku (the actual novel question)

The Sudoku #4 v4 used the **perfect** verifier: `violations==0` certifies a trace as
*certainly* correct (a valid givens-consistent Sudoku is unique). Carnot's reasoning
verifier is **imperfect** — FoVer AUROC 0.9131. So the central question is:

> **Is a 0.91-AUROC verifier a good enough CERTIFIER to drive self-improvement, or do
> its false-positive certifications poison the training set and collapse it** (the
> Sudoku soft-argmin failure mode, at scale)?

RFT cares about **certification PRECISION**, not AUROC: of the traces the verifier
*certifies as correct*, what fraction actually are? If precision is high at usable
recall, RFT trains on mostly-correct data and improves. If precision is low, it
distills wrong traces and degrades. **AUROC 0.91 does not tell us the precision at a
high-confidence threshold — that is exactly what Phase 0 measures.**

## Infra inventory (what exists vs the gap)

- **Generated traces — READY.** `data/p01_*_generations.jsonl` (difficulty-matched
  558 rows, GSM8K 5.9MB, hardmath 3.4MB): each row has `text` (full trace),
  `reasoning_steps`, `extracted_answer`, and `is_correct` (the gold label). K traces
  per problem already sampled — the expensive generation step is done.
- **Verifier ensemble — READY.** `score_carnot_ensemble` and the FoVer panel scorers
  (`python/carnot/eval/verifier_error_independence_scissor_at_scale.py` and siblings,
  used by the moat + efficiency panel). Scores reasoning steps.
- **Small trainable base models — READY.** Qwen2.5-0.5B(-Instruct), gemma-4-E2B,
  Qwen1.5-0.5B, and a Carnot per-token-EBM on Qwen3-0.6B are cached (non-GGUF,
  trainable). A 0.5B is ideal for a fast PoC.
- **THE GAP: no LLM fine-tune / LoRA / RFT harness** (no peft/trl/lora in the tree).
  Phase 1 must build it. Phase 0 does NOT need it.

## Phase 0 — verifier CERTIFICATION PRECISION (cheap, decisive, de-risks everything)

The smart first step (mirrors how #3 de-risked #4): NO fine-tuning, NO generation —
reuse existing p01 traces + the existing ensemble.

1. Load p01 traces (trace, reasoning_steps, is_correct).
2. Score each with the Carnot verifier ensemble -> a per-trace correctness score.
3. At a high-confidence certification threshold, measure **precision** (of certified
   traces, fraction gold-correct) and **recall** (fraction of correct traces
   certified), and the precision-recall curve.
4. Compare to the gold-certified set (the RFT training set you'd get from labels).

**GATE:** certification precision >= ~85% at a recall that yields enough traces to
train on (say >= 20% of correct traces certified). If it passes, Phase 1 (RFT) is
viable and worth building the harness for. If precision is low, the imperfect
verifier is NOT a usable certifier — and that is itself the decisive finding
(bounds the self-improvement claim), saving the Phase-1 build.

This is CPU/light-GPU, uses only existing assets, and answers the load-bearing
question directly. **Build Phase 0 first; gate Phase 1 on it.**

## Phase 1 — the RFT loop (gated on Phase 0)

Only if Phase 0 passes. Build a minimal LoRA fine-tune harness (peft) on a 0.5B base.
Arms on a headroom corpus: base / **verifier-RFT (no gold)** / gold-RFT (upper bound)
/ SC-baseline. Gates: verifier-RFT > base (multi-seed, significant), and verifier-RFT
recovers a meaningful fraction of the gold-RFT lift. This is the direct scale-up of
#4 v4 (Sudoku perfect verifier -> reasoning imperfect verifier-ensemble).

## Decentralization / discipline notes

Local open models only for the headline (Qwen/gemma small bases). PRECONDITIONS-gate
the verifier-scoring (does the ensemble need GPU? cached?). Headroom must be confirmed
(oracle > SC) on the chosen corpus before training — the P0.1 lesson — else it's the
no-headroom null again. Multi-seed + held-out eval for any lift claim.

## Status

Scoped 2026-06-07. Phase 0 is the immediate concrete next experiment (cheap,
decisive, no new infra). Phase 1 gated on Phase 0. This is the real test of
verifier-as-self-improvement-reward — the Phase-3 endgame in miniature.

---

# Phase 0 VERDICT (2026-06-07) — process verifier != outcome certifier

The de-risk ran v1->v3 + a chunking bridge and found the binding constraint BEFORE any
fine-tune-infra build. Arc:

| test | what | result |
|---|---|---|
| v1 | whole-trace certification | base-rate (format mismatch, discarded) |
| v2 | fine p01 steps, TRACE-level | 56% precision @ 24% recall |
| v3 | FoVer corpus, PER-STEP, balanced | **96.7%** precision @ 78% recall |
| bridge | re-chunked p01 (paragraph), TRACE-level | 56% (same granularity ~11, same precision) |

**Decisive conclusion.** v3's 96.7% is **per-STEP** precision; v2/bridge's 56% is
**TRACE-level** (does "all steps clean" predict the ANSWER is correct?). Re-chunking
does not close the gap. The gap is **process vs outcome**: Carnot's verifier checks
**local step validity** excellently (96.7% in-format = the FoVer/moat result), but RFT
self-improvement needs **outcome correctness**, and *all-steps-locally-valid != correct
answer* (a trace can be locally valid yet reach a wrong answer). So a strong PROCESS
verifier yields noisy OUTCOME certification (~56% -> ~44% of certified traces have wrong
answers = RFT-poisoning).

**The honest bound.** Verifier-as-self-improvement-reward, with Carnot's PROCESS
verifier as the certifier, is NOT clean on free-form generated traces: process
certification is insufficient for outcome labels. The Sudoku #4 v4 positive used a
PERFECT verifier where local validity == global correctness (a unique solution); that
equivalence does NOT hold for free-form reasoning.

**Rescue paths (the forward directions, each a real build).**
1. Pair the process verifier with an OUTCOME verifier (answer-consistency / a learned
   outcome head) so certification reflects correctness, not just local validity.
2. Re-calibrate/fine-tune the verifier on the GENERATOR's own trace distribution so it
   certifies outcomes (closes the in-dist-vs-OOD axis too, which is entangled here).
3. Use the high-precision PER-STEP signal differently: step-level RFT / process-reward
   (reward correct steps) rather than trace-level certification -- closer to how the
   96.7% per-step precision would actually be usable.

**Why this de-risk was worth it.** It found a fundamental obstacle (process != outcome)
for ~0 GPU and no fine-tune-infra build -- exactly the value of a cheap Phase 0. The
direction is not dead; it is precisely bounded, with three concrete rescue paths.

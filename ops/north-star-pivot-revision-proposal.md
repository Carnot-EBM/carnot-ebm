# PROPOSAL — North-Star revision for the verifier-as-reward pivot (2026-06-11)

**Status: PROPOSAL for operator review + merge.** `ops/north-star.md` is operator-curated
(the autonomous loop must not edit it). This file drafts the proposed revision; the operator
applies it when satisfied. Authored by the outer-loop after the 2026-06-11 pivot decision.

---

## Why a revision

The current north-star §0 ("division of labor") says the verifier's job is to make the
*search* accurate + efficient at **inference time** — routing (Meta-EBM Cascade Router),
pruning hopeless actions, verifying state/trajectory. The overnight .371–.375 evidence
sharpened this: the verifier-as-**inference-selector** is a *commodity* (proven, not a moat),
while the verifier's scarce, hack-resistant, durable value is as a **training/search-time
reward signal**. The pivot does not discard the inference-time role (the demo-fit safety-gate
still ships) — it **adds** the load-bearing training-time role. The north star should name it.

**Evidence behind the revision (decision-grade, .371–.375):**
- Selection moat dead — 5×-confirmed (GAP-3 lineage retired); the one GAP-4 positive is
  generator-attributable + contamination-confounded.
- Frontier-as-selector questions perpetually deferred (off-ARC directional-not-significant;
  vc33 a WM-fidelity wall; decentralization-as-base-selection underpowered).
- The constructive direction (Deep-Think Q3 + the standing Sudoku RFT-beats-SFT beachhead,
  3/3 seeds): the un-hallucinating execution verifier is an **automated ground-truth engine**
  → use it to **generate training data**, not to filter at inference.

---

## Proposed edit 1 — extend the §0 "division of labor" paragraph

REPLACE the sentence *"The energy verifier does NOT induce; it makes the search accurate and
efficient — routing ... pruning ... verifying state/trajectory at scale."* with:

> The energy verifier does NOT induce. Its job is **two-fold**, and the overnight .371–.375
> evidence reordered the two by durability:
> 1. **(training/search-time — the durable, scarce role) Verifier-as-REWARD.** The verifier is
>    an un-hallucinating, hack-resistant ground-truth signal, so it is an *automated reward/
>    label engine*: certify which generator traces are correct (pass execution / demos / tests)
>    and **train the generator on them** (verifier-certified RFT). This is how the *generator*
>    gets stronger — including the *sovereign local* generator (decentralization-as-distillation:
>    does training on verifier-certified traces close the local induction gap?). The Sudoku
>    RFT-beats-SFT result (verifier-certified RFT ≥ gold-SFT, 3/3 seeds) is the beachhead.
> 2. **(inference-time — the shipped commodity role) Verifier-as-GATE.** The model-free demo-fit
>    execution gate is the shipped Phase-1 "second pair of eyes" trust product (zero-loss
>    abstention wrapper) and an action-pruner/router for efficiency. This role is *useful and
>    shipped* but is a commodity (an execution sandbox), proven across .371–.375 to add no
>    independent *selection* signal beyond catching syntax/demo faults — so it is the product,
>    not the research frontier.

## Proposed edit 2 — add to the §0 metrics table (or as a §0.1 note)

The two north-star metrics (ARC-AGI-3 accuracy + efficiency) are unchanged as the *destination*.
Add the verifier's **load-bearing contribution metric** for the pivot era:

| Axis | Metric | Why |
|---|---|---|
| **VERIFIER-AS-REWARD (new, load-bearing)** | held-out gain of verifier-certified RFT over (a) the cold base and (b) gold-SFT, on ≥2 domains (Sudoku ✓ beachhead + ARC-induction, .377) | The verifier's scarce value is training a better generator from a *label-free* signal. Beating gold-SFT = you can self-improve without oracle labels — the hack-resistant-evals bottleneck (Anthropic-W2S) that the verifier uniquely fills. Also *is* the decentralization answer (latent-vs-absent as a training outcome). |

## Proposed edit 3 — G1–G4 publication-gate note (§2)

The G1 headline ("FoVer / the verifier's measured value") should track, going forward, the
**RFT-beats-SFT generalization** result (≥2 domains) as the verifier's primary measured value,
with the inference-time selection numbers relegated to "shipped commodity gate" supporting
status. The G2 (independent reproducer) / G3 (prose narrowing) / G4 (numbers trace) gates are
unchanged in form; only the G1 *headline claim* shifts from "verifier selects" → "verifier
trains a better generator (verifier-certified RFT ≥ gold-SFT, held-out, ≥2 domains)."

---

## What does NOT change

- ARC-AGI-3 (accurate + efficient) remains the destination (§0).
- The hybrid architecture (generator induces; verifier is the load-bearing energy layer) is
  unchanged — the *verifier's* contribution just gained its primary, durable form (reward).
- The shipped demo-fit safety-gate stays as the Phase-1 trust product.
- The discipline machinery (adversarial-verify, G-gates, never-prune) is unchanged.

## Operator action

Merge the three edits into `ops/north-star.md` §0/§2 when satisfied (or adjust). The .377
roadmap (`research-roadmap-next.yaml`, pre-staged) already operationalizes the pivot as the
verifier-certified-RFT headline; this revision aligns the north star with it.

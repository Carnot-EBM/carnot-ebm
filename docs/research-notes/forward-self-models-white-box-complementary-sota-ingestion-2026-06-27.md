# SOTA ingestion: "Forward Self-Models" → white-box-complementary tier (state / feature / **computation**)

**Date:** 2026-06-27 · outer-loop (interactive, operator-requested) · SOTA-Ingestion Cycle Discipline.
**Source:** J. Gilley, "Forward Self-Models Learn an Empirical Approximation of Neural Network
Computation," `https://jagilley.github.io/forward-self-models.html` (a github.io writeup).
**Provenance — FLAGGED:** single-pass WebFetch 2026-06-27; **not on arXiv** (verified via WebSearch
2026-06-27 — no arXiv ID exists). Cite as a web writeup only; re-verify against the live page before
any paper-v6 use. The numeric findings below (r=+0.332, d≈±0.03, cosine/KL figures) are reproduced
from a single WebFetch summary and are **not** independently confirmed against the rendered source.

This note was produced with a 3-lens mapping + adversarial-critic workflow (`wf_fe49dab5-788`); the
critic verdict was **clean** with three required fixes, all incorporated here.

---

## 1. What the source establishes

A **forward self-model** is a deliberately **small** auxiliary transformer (1–3% of main-model params;
e.g. 330K for a 28.9M toy GPT, 26.2M for Llama-3.2-1B) trained with MSE loss to predict a main model's
**later-layer** residual-stream activations from its **earlier-layer** activations (layer *i* → layer
*j*, *j*>*i*), with causal attention and **no gradients back to the main model**. It reconstructs the
activation manifold well (toy GPT: cosine 0.972, causal substitution recovers 94% of the replaced
layer's KL; Llama-3.2-1B layer 7→8: cosine 0.937, 74% KL), degrading with layer-gap depth and
saturating near ~10% capacity.

**The load-bearing finding (a dissociation):** because the model is deliberately small, its residual
`r = a_j − â_j` is the *incompressible* part of the layer's computation — what the author calls
**"computational novelty."** The residual:

- **tracks COMPUTATIONAL COMPLEXITY** — correlates with attention entropy (**r = +0.332**); highest
  before closing delimiters (d=+0.84), lowest at sentence starts (d=−0.85), lower under focused
  attention (d=−0.61);
- **does NOT track PREDICTION DIFFICULTY** — easy vs hard predictions, **d ≈ ±0.03** (negligible);
- the author makes **no claim about uncertainty, hallucination, or correctness.**

Conceptual core: a "dissociation between representation and computation" — the forward model is *given*
the optimal representation, so what it learns is the *computation*, and its residual measures how hard
that computation is to compress, **not** whether the output is right.

**This null is trustworthy (positive-control check).** Per Carnot's FALSE_NEGATIVE_RISK discipline a
null is only informative with a positive control. Here the positive control is built in: the residual
*does* correlate with attention entropy (r=+0.332), proving it measures something real — so the
~zero correlation with prediction difficulty (d≈±0.03) is a genuine dissociation, not a degenerate
measurement. The "residual ≠ correctness" result is real, not an artifact.

---

## 2. The headline takeaway: a THIRD white-box primitive *type*

Carnot already files two white-box mechanistic-interpretability tools as **complementary, never core**
(see `reference_cognometry.md`, `reference_goodfire_silico.md`). The forward self-model adds a third —
and the genuine contribution is **taxonomic**: it probes a different *thing*.

| Primitive | Probes… | Example | Carnot status |
|---|---|---|---|
| **Cognometry** (Fathom/Styxx) | cognitive **STATE** (refusal / confabulation / retrieval directions over residual+logprob) | AUC 0.998 HaluEval-QA | white-box complementary tier |
| **Goodfire Silico** | **FEATURE / CIRCUIT** (which neuron/pathway fired; the causal *why*) | 9.11>9.9 traced to a version-number neuron | white-box complementary tier |
| **Forward self-model** (this source) | **COMPUTATION** (the activation→activation transform; its complexity) | residual ~ attention-entropy r=+0.332 | white-box complementary tier (lowest priority — §4) |

**The only fully-defensible Carnot use of this source is a one-paragraph `state / feature /
computation` taxonomy entry in the position paper's white-box-complementary section** — explicitly NOT
in the black-box verifier core. Everything else is non-application (§3, §5).

---

## 3. Verifier-core mapping: NON-APPLICABLE

The forward self-model does **not** help Carnot's black-box energy/constraint verifier. Four honest
non-applications, in descending strength:

1. **Residual ≠ a hallucination/uncertainty signal — CONTRADICTED BY THE SOURCE.** The single most
   tempting bridge ("the residual is the model's white-box 'I'm unsure / about to confabulate' cue,
   feed it to the verifier") is refuted by the source's own dissociation finding: residual ~
   computational complexity (r=+0.332), residual ⊥ prediction difficulty (d≈±0.03), no uncertainty
   claim by the author. This bridge must be **permanently retired** — and the dissociation number is
   the kill-shot to cite every time it recurs (§6).
2. **Residual ≠ a core verifier feature — forbidden by construction.** It reads model internals
   (a forward model over the residual stream). Carnot's core is **black-box / API-only** (verifies
   *outputs*, no internals) per decentralization rules 1 (local-first open models) + 7 (no
   vendor/internals dependency in the core). A white-box signal is a labeled complementary tier at
   most, never the core.
3. **Even as a complementary signal it is the weakest of the three — two empirical demotions.**
   (a) **Cross-vendor transfer ceiling:** Cognometry already showed white-box directions transfer
   *within* a model family (cos 0.464) but **not across vendors** (cos 0.043). A per-model forward
   self-model inherits the same ceiling — it would not transfer to the next open-weight model, let
   alone Carnot's any-LLM goal. (b) **No black-box-accessible spillover:** Cognometry's logprob
   trajectories are at least readable through most APIs (a candidate black-box feature); the forward
   self-model needs *full activations* and has **no** black-box-accessible spillover — making it the
   **most decentralization-degraded** of the three white-box references on file.
4. **Residual ≠ a repair signal.** Verify-and-repair must localize *which* output is wrong and *why*;
   a complexity correlate that is uncorrelated with prediction difficulty cannot localize an error.

A cascade-routing idea (escalate to the expensive verifier on high-residual inputs) is **note-only,
not a build**: routing on compute-complexity does not target the cases where the cheap verifier is
actually *wrong* (that's the dissociation, again), it is white-box, and the Meta-EBM Cascade Router
already has cheaper black-box routing features.

---

## 4. White-box-complementary bucket: where it fits

Slots beside Cognometry and Silico as the **COMPUTATION** probe (§2). The "a mature reliability stack
uses both/all" framing is already explicit in both precedent files
(`reference_cognometry.md`: "a mature safeguard stack uses both"; `reference_goodfire_silico.md`:
"A complete LLM-reliability stack uses both"). Adding the third axis is the real, honest contribution.

**Decentralization placement:** lowest priority of the three (rules 1 + 7). It is open-weight-only,
needs full activation access, has no black-box spillover, and (per §3.3a) would not transfer across
vendors. Useful only as a labeled local-first diagnostic on a model the operator fully controls, never
a verifier feature and never an AUROC contributor.

---

## 5. Deeper connections + honest non-applications

- **Amortized-inference theme — THEMATIC ECHO ONLY; do NOT add to the corroboration list.** A forward
  self-model is an amortized one-pass predictor of the main model's *own computation*, which superficially
  rhymes with Carnot's "amortized inference > energy-descent" corroborators (NVIDIA Ising-QEC: a tiny
  3D-CNN beats MWPM energy-minimization; BES: arXiv:2605.28814). But those are load-bearing because the
  amortized net **replaces an energy/search step that defines a task solution** (decode the syndrome /
  find a correct trajectory). The forward self-model amortizes a model's *introspection of its own
  compute* — it generates nothing, selects nothing, verifies nothing, and its residual tracks
  attention-entropy, not solution quality. Citing it alongside NVIDIA-QEC / BES would **dilute** the
  genuine corroborators. (Cf. the NVIDIA-Ising name-collision lesson — shared vocabulary, false friend.)
- **ARC generation wall — NO HELP (high confidence).** The binding ARC wall is candidate **generation**
  (the winning L1 trajectory never enters the pool; `project_arc_l1_first_contact_wall`,
  `project_arc_generation_not_selection`). A white-box probe of the generator's own activations produces
  no new candidates, proposes no actions, induces no dynamics — and would not run in the iGPU-pinned
  live submission stack. No defensible mechanism; claiming ARC benefit would violate the anti-overclaim rule.
- **Phase-3 deep-EBM — ORTHOGONAL.** The source frames forward self-models as "a primitive for future
  architectures that model their own computational dynamics," superficially adjacent to Phase-3
  energy-over-activations. But Phase-3's energy is *normative* (minimized by a sampler to verify/repair,
  hardware-acceleratable Boolean/Ising); a forward self-model is a *descriptive* predictor of an
  activation trajectory. Worse, Carnot's own result is that energy-**descent** does not generate
  (Sudoku rand-init descent ≈ 0); a descriptive self-model of computation is further still from a
  generator. Shared phrasing, no shared mechanism — do not let it imply a roadmap dependency.

---

## 6. The recurring overclaim to permanently retire

> **"The forward-self-model residual is a white-box uncertainty/hallucination signal — feed it to the
> verifier."** REFUTED by the source's own dissociation finding: residual ~ computational complexity
> (attention entropy **r=+0.332**) but ⊥ prediction difficulty (**d≈±0.03**), and the author makes no
> uncertainty/hallucination claim. The null is trustworthy because it has a built-in positive control
> (the r=+0.332 correlation proves the residual measures something real). Cite this every time the
> bridge recurs; treat any artifact reviving it as an overclaim.

---

## 7. Citations

- **[FLAGGED]** J. Gilley, "Forward Self-Models Learn an Empirical Approximation of Neural Network
  Computation," `https://jagilley.github.io/forward-self-models.html` — web writeup, **not on arXiv**
  (WebSearch 2026-06-27). Single-pass WebFetch; numbers (r=+0.332, d≈±0.03, cosine/KL) not
  independently re-verified against the rendered page.
- `reference_cognometry.md` — Cognometry / Styxx (fathom.darkflobi.com/cognometry): white-box
  residual-stream + logprob **STATE** probe; within-family transfer cos 0.464, cross-vendor cos 0.043.
- `reference_goodfire_silico.md` — Goodfire Silico (MIT Tech Review 2026-04-30): white-box
  neuron/circuit **FEATURE** inspection.
- `reference_nvidia_ising_qec_amortized.md`, `reference_bes_bidirectional_evolutionary_search.md`
  (arXiv:2605.28814) — the genuine "amortized > energy-descent" corroborators this source only
  thematically echoes (§5).
- CLAUDE.md "Decentralization-Respecting Design Constraints" rules 1 + 7 — the black-box-core invariant.
- `project_arc_l1_first_contact_wall` / `project_arc_generation_not_selection` — the ARC
  generation-wall non-application.

---

## 8. Flagged for the next roadmap

- **Strongest (and only fully-defensible) use — NOTE-ONLY:** add the `state / feature / computation`
  taxonomy paragraph (§2) to the position paper's white-box-complementary tier, plus the
  `reference_forward_self_models.md` memory entry. No code, no GPU.
- **Do NOT build** a verifier feature, a cascade router on the residual, or an ARC aid. A residual-as-
  verifier-feature A/B is **not recommended** — the source already ran the relevant dissociation and
  found a null; re-running would re-derive a published null (Failed-Experiment-Rerun / shoulders-of-
  giants), and even a positive would be open-weight-only complementary-tier, never core.
- **Cite in paper-v6** only in the *complementary white-box tier* discussion (alongside Cognometry /
  Silico), never as a verifier-reliability result.
- Marked ingested in `research-studying.md`.

Cross-refs: `reference_cognometry.md`, `reference_goodfire_silico.md`,
`thinking-to-recall-verifier-gated-reasoning-sota-ingestion-2026-06-27.md` (sibling ingestion this
session), `reference_nvidia_ising_qec_amortized.md`, `reference_bes_bidirectional_evolutionary_search.md`.

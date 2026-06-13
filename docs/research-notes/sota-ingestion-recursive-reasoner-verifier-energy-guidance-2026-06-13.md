# SOTA ingestion 2026-06-13: recursive reasoner verifier energy-guidance map

**Artifact fields**
- honest_verdict: `complete: sota_ingestion_recursive_reasoner_verifier_energy_guidance_mapped`
  - principle: Terminal-prefixed. Records ingestion completed with verifiable citations.
- methods_mapped:
  - {name: `TRM nano-trm recursive baseline gate`, arxiv_id_or_url: `2510.04871`, url: `https://arxiv.org/abs/2510.04871`}
  - {name: `TTA-TRM adaptation-control arm`, arxiv_id_or_url: `2511.02886`, url: `https://arxiv.org/abs/2511.02886`}
  - {name: `V-STaR accepted/rejected trace selector`, arxiv_id_or_url: `2402.06457`, url: `https://arxiv.org/abs/2402.06457`}
  - {name: `SEDD discrete diffusion score-energy formalism`, arxiv_id_or_url: `2310.16834`, url: `https://arxiv.org/abs/2310.16834`}
  - {name: `Classifier-guided diffusion energy precedent`, arxiv_id_or_url: `2105.05233`, url: `https://arxiv.org/abs/2105.05233`}
  - {name: `Classifier-free diffusion guidance control`, arxiv_id_or_url: `2207.12598`, url: `https://arxiv.org/abs/2207.12598`}
  - {name: `DiffusionGemma queued discrete-text substrate`, arxiv_id_or_url: `https://ai.google.dev/gemma/docs/diffusiongemma`, url: `https://ai.google.dev/gemma/docs/diffusiongemma`}
  - principle: Each method/source MUST carry a real arXiv ID/URL; an ingestion note without verifiable citations is treated as fabrication.
- flagged_for_v385: `diffusiongemma_sedd_verifier_energy_guidance_probe_v385`
  - principle: Closes the discover->ingest->plan loop: names the strongest method for the next planner.

**Fresh-pass provenance**

Read `research-studying.md` and `research-references.md` filtered to
verifier-guided-training and energy-guided-generation, including the prior
Exp 4102, 4111, 4121, 4130, 4141 entries and the 2026-06-13
DiffusionGemma operator-requested note. Ran the reliable helpers, not
`/deep-research`:

- `.venv/bin/python scripts/sweep_clusters.py 0 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 1 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 3 --max-results 8`
- `.venv/bin/python scripts/sweep_semscholar.py "verifier guided training TRM V-STaR recursive reasoning" --limit 8`
- `.venv/bin/python scripts/sweep_semscholar.py "energy guided generation discrete diffusion classifier guidance SEDD" --limit 8`
- `.venv/bin/python scripts/sweep_semscholar.py "DiffusionGemma energy guidance discrete token diffusion verifier" --limit 8`

The arXiv cluster helper emitted the broadened verifier, EBM, and
active-inference query URLs. Semantic Scholar returned HTTP 429 for the first
two focused queries and returned `arXiv:2605.04040` for the DiffusionGemma
guidance query; that paper is adjacent evidence for verification feedback as
generation guidance, but the `.385` map keeps the requested TRM/TTA/V-STaR/
SEDD/guidance anchors as the load-bearing sources. Low-concurrency
WebSearch/WebFetch verified `arXiv:2510.04871`, `arXiv:2511.02886`,
`arXiv:2402.06457`, `arXiv:2310.16834`, `arXiv:2105.05233`,
`arXiv:2207.12598`, and `https://ai.google.dev/gemma/docs/diffusiongemma`.

## Current .385 verifier-guided-generation anchor

The local DiffusionGemma note says the model is a DEPTH scale-up of the
verifier-as-guidance bet, not a new proof that the verifier works. The same
note keeps it gated on the TRM verifier graft reporting `verifier_value_added
== true`. Therefore the `.385` handoff should connect two tracks without
collapsing them: recursive `nano-trm` still measures candidate quality and
verifier discrimination, while DiffusionGemma/SEDD supplies the generation-time
surface where Carnot energy could act before final text or grid selection.

## TRM nano-trm recursive baseline gate

**Method/source:** TRM, `arXiv:2510.04871`
(https://arxiv.org/abs/2510.04871), is the baseline recursive substrate: a
small two-layer recursive model with 7M parameters and strong Sudoku/ARC
generalization claims relative to larger systems.

**Implementation over nano-trm + Carnot-verifier stack:** Keep `nano-trm` as
the local Sudoku baseline that measures oracle(best-of-K) headroom,
pass-at-one, checkpoint lineage, and candidate diversity before any
energy-guided generation claim. The verifier is only meaningful when the
generator emits alternatives the verifier can discriminate.

**Pitfalls / where it fails:** If the baseline is undertrained, lacks oracle
headroom, or majority vote already captures all selectable support, then a
verifier-training or diffusion-guidance result is uninformative rather than
negative evidence against the method.

## TTA-TRM adaptation-control arm

**Method/source:** TTA-TRM, `arXiv:2511.02886`
(https://arxiv.org/abs/2511.02886), shows that bounded full fine-tuning of a
tiny recursive model can change results within a competition-style compute
budget.

**Implementation over nano-trm + Carnot-verifier stack:** Keep a no-verifier
adaptation arm with the same checkpoint, optimizer-step budget, LR schedule,
candidate pool, and receipts as any Carnot-verifier-labeled arm. This is the
control that prevents ordinary adaptation compute from being mistaken for
verifier reward.

**Pitfalls / where it fails:** Full fine-tuning can win through compute alone.
If the verifier arm gets more steps, cleaner labels, or a different schedule,
the experiment cannot attribute a delta to Carnot verifier information.

## V-STaR accepted/rejected trace selector

**Method/source:** V-STaR, `arXiv:2402.06457`
(https://arxiv.org/abs/2402.06457), trains a verifier from both correct and
incorrect self-generated solutions and uses it to select among candidates at
inference time.

**Implementation over nano-trm + Carnot-verifier stack:** Retain accepted and
rejected `nano-trm` Sudoku traces, then train or evaluate a selector over that
paired evidence before spending on another generator pass. The clean handoff is
a pairwise selector that can be compared against fixed vote, oracle(best-of-K),
and the current Carnot verifier order.

**Pitfalls / where it fails:** V-STaR needs real contrast. If the saved pool is
dominated by near-identical wrong candidates or the executable oracle exposes
false-positive Carnot labels, the selector will learn trace artifacts instead
of correctness.

## SEDD discrete diffusion score-energy formalism

**Method/source:** SEDD, `arXiv:2310.16834`
(https://arxiv.org/abs/2310.16834), extends score matching to discrete spaces
through score entropy and provides the clearest language-model bridge from
token denoising to score/energy reasoning.

**Implementation over nano-trm + Carnot-verifier stack:** Use SEDD as the
formal scaffold for the queued DiffusionGemma guidance experiment: Carnot
verifier energy should alter denoising choices while the canvas is still
mutable, then the resulting candidate is scored by the same executable and
non-oracle receipts used for `nano-trm`. This moves verifier signal upstream
from rerank-after-generation to generation-time guidance.

**Pitfalls / where it fails:** SEDD itself is a generator loss, not a Carnot
verifier. If the external energy is badly scaled or applied too late, it can
collapse diversity, harm fluency, or merely reproduce post-hoc reranking under
a more expensive sampler.

## Classifier-guided diffusion energy precedent

**Method/source:** Classifier-guided diffusion, `arXiv:2105.05233`
(https://arxiv.org/abs/2105.05233), shows that diffusion samples can be steered
by an external classifier gradient. Classifier-free guidance,
`arXiv:2207.12598` (https://arxiv.org/abs/2207.12598), is the matching control:
guidance can also be produced by mixing model scores without an external
classifier.

**Implementation over nano-trm + Carnot-verifier stack:** Treat Carnot verifier
scores as the discrete-token analogue of an external guidance energy. The
`.385` probe should sweep small guidance weights, include a no-external-energy
classifier-free-style control, and report both generation quality and exact
Sudoku validity so an over-guided sample cannot pass as a verifier win.

**Pitfalls / where it fails:** Guidance is a tradeoff. Too much verifier energy
can reduce diversity or create samples optimized for a proxy rather than for
validity. A win against no-guidance is not enough unless it also beats the
ordinary conditional/unconditional guidance control.

## DiffusionGemma queued discrete-text substrate

**Method/source:** DiffusionGemma official documentation
(https://ai.google.dev/gemma/docs/diffusiongemma) describes an experimental
open model that generates text with discrete diffusion over block canvases,
including parallel denoising, bidirectional attention over the generation
canvas, entropy-bounded denoising, adaptive stopping, and a Sudoku fine-tuning
recipe.

**Implementation over nano-trm + Carnot-verifier stack:** Queue DiffusionGemma
as the open-weight substrate for the `.385` guidance probe, not as a replacement
headline. The testable path is: establish verifier discrimination on the
`nano-trm` domain, attach Carnot energy during DiffusionGemma denoising, and
compare against no-guidance plus classifier-free-style guidance controls.

**Pitfalls / where it fails:** DiffusionGemma is a generator substrate. Its base
or SFT Sudoku behavior does not prove Carnot verifier value, and the official
docs describe it as experimental. The result must be labeled as guidance only
if Carnot energy changes the denoising outcome and improves held-out exact
validity under matched compute.

## Flagged for the .385 roadmap

`diffusiongemma_sedd_verifier_energy_guidance_probe_v385` is the strongest
candidate. It directly tests the queued energy-guided-generation hypothesis:
SEDD gives the discrete score/energy formalism, classifier/classifier-free
guidance gives the ablation structure, and DiffusionGemma gives the open
block-diffusion substrate. Keep it gated on measured Carnot-verifier
discrimination; otherwise spend `.385` on improving the trace selector and
candidate diversity before launching a guided-generation probe.


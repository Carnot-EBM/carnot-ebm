# SOTA ingestion 2026-06-13: verifier-moat guidance map for .387

**Artifact fields**
- honest_verdict: `complete: sota_ingestion_verifier_moat_guidance_mapped_v387`
  - principle: Terminal-prefixed. Records ingestion completed with verifiable citations.
- methods_mapped:
  - {name: `TRM nano-trm baseline and headroom gate`, arxiv_id_or_url: `2510.04871`, url: `https://arxiv.org/abs/2510.04871`}
  - {name: `TTA-TRM adaptation-control arm`, arxiv_id_or_url: `2511.02886`, url: `https://arxiv.org/abs/2511.02886`}
  - {name: `V-STaR accepted/rejected trace selector`, arxiv_id_or_url: `2402.06457`, url: `https://arxiv.org/abs/2402.06457`}
  - {name: `SEDD discrete diffusion score-energy formalism`, arxiv_id_or_url: `2310.16834`, url: `https://arxiv.org/abs/2310.16834`}
  - {name: `Classifier-guided diffusion external-energy precedent`, arxiv_id_or_url: `2105.05233`, url: `https://arxiv.org/abs/2105.05233`}
  - {name: `Classifier-free guidance control`, arxiv_id_or_url: `2207.12598`, url: `https://arxiv.org/abs/2207.12598`}
  - {name: `EntRGi entropy-aware reward guidance`, arxiv_id_or_url: `2602.05000`, url: `https://arxiv.org/abs/2602.05000`}
  - {name: `EDLM sequence-level diffusion energy comparator`, arxiv_id_or_url: `2410.21357`, url: `https://arxiv.org/abs/2410.21357`}
  - principle: Each method/source MUST carry a real arXiv ID/URL; an ingestion note without verifiable citations is treated as fabrication.
- flagged_for_v387: `vstar_rejected_trace_selector_headroom_gate_before_diffusiongemma_v387`
  - principle: Closes the discover->ingest->plan loop: names the strongest method for the next planner.

## Fresh-pass provenance

Read `research-studying.md` and `research-references.md` filtered to
verifier-as-reward, verifier-guided trace selection, and energy-guided
generation. Also checked the latest gate artifact,
`results/experiment_4168_decisive_verifier_graft_defensive.json`: it records
`verifier_value_added=false` because the graft was deferred while the baseline
was not faithful/stable, not because DiffusionGemma guidance was tested.

Reliable-channel helper pass, not `/deep-research`:
- `.venv/bin/python scripts/sweep_clusters.py 0 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 1 --max-results 8`
- `.venv/bin/python scripts/sweep_semscholar.py "verifier as reward recursive reasoning V-STaR TRM" --limit 8`
- `.venv/bin/python scripts/sweep_semscholar.py "energy guided generation discrete diffusion language model classifier guidance SEDD" --limit 8`

The cluster helper emitted the broadened verifier and energy arXiv API URLs.
Semantic Scholar returned HTTP 429 for both focused queries during this run.
Low-concurrency WebSearch/WebFetch then verified the mapped paper set:
arXiv:2510.04871, arXiv:2511.02886, arXiv:2402.06457, arXiv:2310.16834,
arXiv:2105.05233, arXiv:2207.12598, arXiv:2602.05000, and arXiv:2410.21357.
The DiffusionGemma official documentation was also checked as queued substrate
context, but it is not counted as evidence that the verifier works.

## SOTA -> experiment mapping

## TRM nano-trm baseline and headroom gate

**Method/source:** TRM, arXiv:2510.04871
(https://arxiv.org/abs/2510.04871), is the tiny recursive baseline: a 7M
parameter two-layer recursive model with strong puzzle generalization claims.

**Carnot-verifier implication:** The verifier is only meaningful when the TRM
candidate pool is faithful, stable, and has oracle headroom. If the generator
does not emit selectable correct alternatives, reranking and reward guidance
are uninformative.

**Queued DiffusionGemma implication:** DiffusionGemma should inherit the same
headroom gate. A faster or deeper denoising substrate cannot fix an unmeasured
verifier signal.

**Experiment mapping:** For .387, preserve checkpoint lineage, candidate
diversity, oracle(best-of-K), and vote baselines before any verifier graft or
diffusion guidance run.

## TTA-TRM adaptation-control arm

**Method/source:** TTA-TRM, arXiv:2511.02886
(https://arxiv.org/abs/2511.02886), shows that bounded full fine-tuning of a
tiny recursive model can change ARC outcomes inside a competition-style budget.

**Carnot-verifier implication:** A verifier-labeled arm needs a same-budget
no-verifier adaptation arm. Otherwise ordinary adaptation compute can masquerade
as verifier reward.

**Queued DiffusionGemma implication:** A guidance result must include a
no-external-verifier adaptation or conditioning control, because a generator can
improve from its own update path without Carnot information.

**Experiment mapping:** Keep identical checkpoint, optimizer-step, LR schedule,
candidate-pool, and wall-clock receipts across verifier and no-verifier arms.

## V-STaR accepted/rejected trace selector

**Method/source:** V-STaR, arXiv:2402.06457
(https://arxiv.org/abs/2402.06457), trains a verifier from both correct and
incorrect self-generated solutions and uses it to select among candidates.

**Carnot-verifier implication:** This is the strongest .387 method because it
acts on the current bottleneck: Carnot needs paired accepted/rejected traces and
a selector before spending on a generator-side guidance stack.

**Queued DiffusionGemma implication:** The same rejected evidence should
calibrate Carnot energy before it is attached to DiffusionGemma. Without
rejected traces, guidance can optimize superficial trace artifacts.

**Experiment mapping:** Build the rejected-trace selector and headroom gate
first. Unlock EntRGi-style DiffusionGemma guidance only if verifier
discrimination turns positive.

## SEDD discrete diffusion score-energy formalism

**Method/source:** SEDD, arXiv:2310.16834
(https://arxiv.org/abs/2310.16834), extends score matching to discrete spaces
through score entropy and provides the bridge from token denoising to
score/energy reasoning.

**Carnot-verifier implication:** SEDD is not verifier evidence. It is the
formal scaffold for moving Carnot scores upstream from post-hoc reranking into
generation-time energy.

**Queued DiffusionGemma implication:** Apply verifier energy while the
DiffusionGemma canvas remains mutable, then check the committed candidate with
the same exact validation receipts.

**Experiment mapping:** After a positive discrimination gate, sweep small
guidance weights and report exact validity, diversity, and reward-call cost.

## Classifier-guided diffusion external-energy precedent

**Method/source:** Classifier-guided diffusion, arXiv:2105.05233
(https://arxiv.org/abs/2105.05233), demonstrates steering diffusion samples
with an external classifier signal and shows the fidelity/diversity tradeoff.

**Carnot-verifier implication:** Carnot verifier scores can play the role of an
external guidance signal, but the experiment must audit proxy over-optimization.

**Queued DiffusionGemma implication:** The DiffusionGemma probe should expose
guidance strength and refuse to count verifier-shaped but invalid samples as a
win.

**Experiment mapping:** Include no-guidance, weak-guidance, and strong-guidance
arms under matched denoising steps and exact downstream validation.

## Classifier-free guidance control

**Method/source:** Classifier-free guidance, arXiv:2207.12598
(https://arxiv.org/abs/2207.12598), mixes conditional and unconditional model
scores to obtain guidance without an external classifier.

**Carnot-verifier implication:** This is the mandatory no-external-verifier
control: it distinguishes generic score mixing from actual Carnot verifier
value.

**Queued DiffusionGemma implication:** DiffusionGemma guidance must beat or
complement this internal-score control before making a verifier-moat claim.

**Experiment mapping:** Treat classifier-free-style control as a required arm
in any future guidance experiment.

## EntRGi entropy-aware reward guidance

**Method/source:** EntRGi, arXiv:2602.05000
(https://arxiv.org/abs/2602.05000), studies reward guidance for discrete
diffusion language models by interpolating between continuous token relaxations
and hard tokens according to predictive entropy.

**Carnot-verifier implication:** EntRGi is the best concrete guidance mechanism
only after the verifier is known to be discriminative. It is not evidence that
the Carnot verifier already has value.

**Queued DiffusionGemma implication:** Use EntRGi's entropy-aware soft/hard
token interpolation as the DiffusionGemma implementation template after the
gate flips positive.

**Experiment mapping:** Keep EntRGi queued behind the .387 V-STaR/headroom gate
rather than launching it from the current deferred graft state.

## EDLM sequence-level diffusion energy comparator

**Method/source:** EDLM, arXiv:2410.21357
(https://arxiv.org/abs/2410.21357), adds a residual sequence-level energy model
to diffusion language modeling and uses parallel importance sampling.

**Carnot-verifier implication:** EDLM is the internal-energy comparator. Carnot
should measure whether the external executable verifier adds information beyond
diffusion-model energy.

**Queued DiffusionGemma implication:** For DiffusionGemma, compare external
Carnot energy with an internal sequence-level energy baseline, so guidance is
not merely relabeled EBM behavior.

**Experiment mapping:** Add EDLM-style internal-energy control after the
external verifier discrimination gate is satisfied.

## Flagged for .387

`vstar_rejected_trace_selector_headroom_gate_before_diffusiongemma_v387` is the
strongest follow-on. Exp 4168 did not prove the verifier negative; it deferred
because the baseline was not yet faithful/stable. Therefore .387 should bank
paired accepted/rejected traces and a selector/headroom gate before activating
EntRGi-style DiffusionGemma energy guidance. EntRGi remains the strongest
guidance template, but it should stay queued unless the verifier discrimination
gate flips positive.


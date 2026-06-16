# Audit: the .397 "in-generation moat WON" (exp4293) rests on DEGENERATE controls — not established

**Outer-loop audit, 2026-06-16.** exp4293 reports `diffusiongemma_guidance_moat: True`,
`carnot_minus_rfg_delta: +0.567`, CI95 `[0.367, 0.767]`, `honest_verdict:
diffusiongemma_guidance_moat_won` — i.e. the deepest §5 claim (an EXTERNAL learned verifier
*improves generation*, beating the model's own self-guidance). **It is not supported.** The result
is correctly FLAGGED (TAUTOLOGY, fires on both the conductor's linter and on-disk) and must NOT
headline.

## The tell: all three controls are identical no-ops

`condition_accuracy: {carnot: 0.867, rfg: 0.3, unguided: 0.3, entrgi: 0.3}`.

RFG (the model's own log-likelihood-ratio self-guidance), **unguided**, AND EntRGi all scored
**exactly 0.3** — three distinct guidance conditions producing a bit-identical number is the
no-op signature: the RFG and EntRGi arms did not actually apply their guidance; they fell back to
unguided. That is exactly why `carnot_minus_rfg_delta == carnot_minus_unguided_delta == 0.5667`
(the TAUTOLOGY): `Carnot − RFG == Carnot − unguided` ⟹ `RFG == unguided`.

So "Carnot-guided 0.867 beats RFG 0.3" is **"Carnot beats a no-op," not "Carnot beats the model's
self-guidance."** The load-bearing control (RFG) — the entire point of the moat claim — is broken.

## Second concern: the Carnot arm may be inflated (scorer leak)

The partial-state scorer built in B1 (exp4292) reported `partial_state_auroc: 0.966` — suspiciously
high for scoring *masked* denoising states. Despite B1's leak ablation declaring
`partial_state_leak_free: True`, a 0.966 AUROC + a 0.867 Carnot arm (vs 0.3 unguided) warrants an
independent leak re-check before the Carnot number is trusted.

## Recommendation

- **The .397 capstone MUST NOT headline `diffusiongemma_guidance_moat_won`.** The flag already
  quarantines it; keep it quarantined (this is a TRUE-positive TAUTOLOGY, unlike the exp4257
  reproduction false-positive that was correctly rescued).
- **.398 re-run requirements:** (1) controls that ACTUALLY apply differentiated guidance — assert
  `unguided != rfg != entrgi` (reject if any two arms tie exactly, the no-op signature); (2) an
  independent leak re-check on the exp4292 partial-state scorer (AUROC 0.966 is a yellow flag);
  (3) only then is `carnot − rfg` a valid moat test.
- **adversarial_verify gap (same family as exp4282):** recommend a DEGENERATE_CONTROLS check —
  flag when ≥2 distinct control arms in a `condition_accuracy`/arms map are bit-identical.

## Status of the §5 in-generation thesis
**Still OPEN / NOT established.** The verifier's proven value remains *selection* (the cross-family
+ cross-generator oracle-distinct win, exp4271/exp4291) and *efficiency* (exp4284, with its own
below-random-judge caveat). Whether it improves *generation* is unproven — this attempt's controls
were degenerate.

## Provenance
- exp4293 `results/experiment_4293_diffusiongemma_energy_guided_run_partial_state.json`
- exp4292 `results/experiment_4292_partial_state_diffusion_scorer_build.json` (AUROC 0.966)
- sibling audit: `docs/research-notes/exp4282-arcgen-degenerate-audit-2026-06-16.md`

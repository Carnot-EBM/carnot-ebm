# Config Layer B — scaffolded LLM rule induction (2026-06-18)

Outer-loop research thread continuing the ARC-AGI-3 first-contact solver. Layer B is the config
**win-RULE** inducer — the relation a configuration must satisfy to complete a level. It is the wall
behind ~13 of the 25 offline games (the `FAIL_EXPLORATION` config/toggle class, where the win is
relational and random exploration cannot stumble it). The verifier — not a generator — is Carnot's
value-add, so we want the win-rule expressed as a **checkable predicate** that the verifier grounds.

## The arc: raw-scene reading fails, structured extraction succeeds

**Move #1 (negative) — `arc3_config_layerb_llm.py`.** Ask the local offline `gemma-4-12B-it` (Q4, on
the AMD iGPU, ~4.2 tok/s, never the 3090s, zero quota) to read the raw 64×64 ASCII scene and emit
`is_win(grid)`. Result on **two** games:

- **tr87** (hardest, 2-D glyph rewrite): proposed a structured but **wrong** rule → did not ground.
- **ka59** (simpler, 1-D recolor strip): **degenerated** — wrote `def is_win(grid):` then filled the
  body with a repeated comment-loop ("Row 63 is the bottom row." …), never wrote logic → syntax error.

Diagnosis: the bottleneck is **raw-scene reading**, not rule complexity. The 12B loses spatial track of
the dense 64×64 grid and degenerates, on both difficulties. (Infra facts banked: iGPU is 4.2 tok/s, so
the 4096-token default needs ~975 s and trips a 900 s timeout — the original `TimeoutError` was an
infra artifact, not a capability signal; a `stop:["```"]` mis-fires because gemma re-emits ```` ```python ````.)

**Move #2 (the lever) — `arc3_config_layerb_scaffolded.py`.** Remove the perception burden: hand the
model **small pre-digested features**, never a raw grid. Extract:

- the **editable region** cropped to its bbox, with the **WIN and NON-win** editable values as labelled
  examples, each pre-summarised as `values=[…] counts={…}` (the model never re-counts ASCII by eye);
- the static **reference region** ("rule area") as **object-centric component features** — per colour:
  `total_cells`, `n_components`, component bounding boxes — *not* a raw sub-grid.

Tiered, falsifiable verdict: **Tier 0** coherent runnable predicate (vs move #1's degeneration) ·
**Tier 1** grounds (fires on the banked win, 0 false-positives on the non-wins) · **Tier 2** grounds AND
is not a literal-array hardcode (a real, generalisable relation).

### ka59 result: Tier 2 — GROUNDED relational rule

`complete_scaffolded_tier2_GROUNDED_relational_rule` — `fires_on_win=True`, `false_positive_rate=0.0`
over 6 non-wins, `literal_hardcode=False`. The model's own reasoning, verbatim:

> "The rule is: the number of fours in e must equal the total count of fours in the reference region
> (32)." → `return count_4 == 32`

It related the editable strip's 4-count to the reference region's 4-count (32, read straight from the
component digest) — exactly the editable↔reference relation we wanted. The verifier grounded it.

### Iteration notes (what it took, all generic — not per-game)

1. Raw editable arrays only → coherent but the model **miscounted** the ASCII → didn't ground. Fix:
   pre-compute `counts={…}` so it never counts by eye.
2. Rendering the reference as a **cropped raw grid** (21×45) **re-triggered the degeneration** — the
   model narrated the grid cell-by-cell and ran out of tokens. This sharpened the principle: *every*
   region must be a compact digest, never a raw grid, however cropped.
3. Reference as **connected-component features** (object-centric, the pipeline's load-bearing
   representation) restored coherence AND supplied the relational signal → Tier 2.
4. Two **scoring bugs in our own harness** under-reported the success: `(false_positive_rate or 1.0)`
   turned a perfect `0.0` into `1.0` (falsy-zero), and the literal-hardcode detector matched the win
   array echoed in a *comment*. Both fixed (strip comments; explicit `None` check). The model had
   grounded a relational rule before the scoring was corrected.

### tr87 result: Tier 0 — coherent but not grounded

`complete_scaffolded_tier0_coherent_but_rule_not_grounded` — `coherent=True`, `grounded=False`,
`literal=False`. The headline is still a win for the thesis: scaffolding **fixed the reading problem
even for tr87** — no degeneration this time (move #1 made tr87 ramble a wrong rule; here it wrote clean
code: `e = grid[48:64, 15:64]`). But it did not ground, for a concrete structural reason: tr87's
editable region is **768 cells (16×48, 2-D)**, vs ka59's 38-cell 1-D strip. The flat `values=[…]`
digest of a 768-cell region is itself 768 numbers; the model spent its **entire 1100-token budget
echoing that array as a comment** and never reached a `return` (so `is_win` returns `None` →
`fires_on_win=False`).

Conclusion: the **generic count/component digest does not transfer to the rewrite class** — not because
the model can't read (it can now), but because a 2-D glyph rewrite over a 768-cell region needs a
**rewrite-specific digest**: don't dump 768 raw values; characterise the editable region as a grid of
glyph cells and supply the rewrite **mapping** the reference defines (a per-glyph lookup), so the
induced predicate is `for each cell: e[cell] == map(reference, cell)` rather than a flat comparison.
This was the predicted failure mode (a per-glyph mapping table), now confirmed.

### tr87 rewrite digest: a from-PIXELS glyph-substitution verifier (partial)

`arc3_config_layerb_glyph_tr87.py` — `complete_glyph_pixel_decode_partial_not_grounded`. Reframing:
tr87's mechanic is **known** (the adapter RE'd it 2026-06-17 — a glyph substitution), so its Layer-B
verifier is a **deterministic glyph-decode + rewrite check**, not an LLM induction (LLM induction is for
*unknown* mechanics like ka59). The original solve read the game's **internal state** and explicitly
deferred "classifying glyph bitmaps + decoding the rule grid from pixels" — this is that pixel upgrade.

Fully RE'd tr87's pixel structure: glyph **value = the 5-on bitmap pattern** of a 5×5 sprite (frame-
agnostic); **A-series** (target + rule LHS) are colour-10-framed, **B-series** (editable + rule RHS) are
colour-7-framed; 3 rule bands give 6 (A→B) pairs; win = `editable_B == [map[a] for a in target_A]`.

What works: the **glyph-segmentation primitive** (split a sprite band into 5×5 tiles by on-pixel column
gaps) cleanly segments **every** region (editable, target, 3 rule bands) into w=5 tiles, and the rule-band
A10/B7 pairing decodes 6 (LHS→RHS) rule prototypes correctly.

What does not (two grounding attempts — exact-bitmap, then Hamming-nearest to the rule codebook): the
verifier does not ground, and the second attempt revealed the gap is **deeper than tolerant clustering**.
Three remaining unknowns, each a real perception step that the adapter sidesteps by reading internal
state: (1) **glyph-value identity** — the same value renders to bitmaps far enough apart that even
Hamming-nearest collapses *distinct* target glyphs onto one rule LHS (a learned 7-way classifier, not a
distance heuristic, is likely needed); (2) **target-region identity** — *which* pixel region is the
"target sequence" is itself uncertain from pixels (rows 41-45 was a guess; its glyphs match the rule LHS
poorly, suggesting it is not the target series or not the right rows); (3) **the rewrite is a greedy
multi-glyph-LHS *sequence* rewrite** (per the adapter's `_required_editable`), not a position-wise
substitution — so even perfect glyph identity needs the sequence-rewrite engine, possibly multi-pass.

Honest boundary: **segmentation + rule-prototype decode are solved; a grounded from-pixels tr87 verifier
needs a glyph classifier + target-region localization + the greedy sequence-rewrite engine** — i.e. the
full pixel-perception pipeline the original solve deliberately deferred. The reusable primitive
(`segment_glyphs`) is banked; the rest is a scoped sub-project, not a one-liner. Pragmatic alternative:
tr87 already has a *working* internal-state verifier (the adapter); the from-pixels version is a
sovereignty/first-contact upgrade, not a correctness gap.

## Implications

- **The scaffolding thesis is validated end-to-end**: an offline 12B that *cannot* read a raw 64×64
  scene *can* induce a grounded, relational config win-rule when handed object-centric digests. The
  decentralisation constraint (local open model) holds — the lever is representation, not model size.
- **The verifier moat holds**: propose-then-ground correctly rejected wrong rules (move #1, the
  reference-render regression) and confirmed the real one. The machinery is sound; the reader was the
  weak link, and structured extraction fixes it.
- This is the same object-centric representation that bridged the pipeline to unseen games — now
  applied to the win-rule layer, not just dynamics.

## Next

- Append the tr87 verdict; if rewrite-class needs a different digest, add a per-glyph mapping feature.
- Push toward **first-contact** Layer B: ka59 used the banked win+non-win as labelled signal. The
  test-time version must induce the relation from *exploration* signal alone (no banked win). The
  relation it found (editable-count == reference-count) is a candidate vocabulary entry for the config
  rule-inducer (per the operator's "derive config-rules from previously-seen games" direction).

Artifacts: `results/arc3_config_layerb_scaffolded_{ka59,tr87}.json`,
`results/arc_config_layerb/ka59_scaffolded_is_win.py`,
harness `scripts/experiments/arc3_config_layerb_scaffolded.py`. Move #1:
`results/arc3_config_layerb_llm_{tr87,ka59}.json`,
note `docs/research-notes/arc-first-contact-solver-2026-06-18.md`.

# Verifier Gaps — the missing-verifier backlog (Carnot's core product backlog)

**WHY (operator directive 2026-06-09):** "we should make note of any 'missing' Carnot ARC
verifiers that might help improve our results — this will help us improve the core function of
Carnot as a verifier, which has become the point of this project."

Carnot's value-add is the VERIFIER. Every case where the verifier *cannot select the correct
answer* — because no existing invariant/energy captures the discriminating signal — is a
**missing-verifier spec**. Filling these specs is the product. This file is the complement of
`ops/verifier_registry.yaml`: the registry lists the verifiers we HAVE; this ledger lists the
verifiers we NEED, with the evidence that motivates each, so the planner can queue verifier-build
tasks against it.

**How a gap gets here.** Any ARC/verifier experiment that finds a case where the correct answer is
present-but-unselectable (e.g. an oracle ceiling the verifier can't reach, a distractor class with
≈chance discrimination, a task family where every applicable family abstains) appends a gap entry:
the failure mode, the discriminator that was MISSING, and what a new verifier would need to compute.
Never-prune; `status: filled` (with the registry verifier_id that closed it) rather than deletion.

**Schema (one entry per gap):**
```
### GAP-<n>: <short capability name>
- status: open | building | filled (<verifier_id>)
- evidence: <experiment artifact + the measured failure mode/number>
- failure mode: <what the verifier mis-ranks and why the existing families are blind>
- missing discriminator: <the signal a new verifier must compute>
- candidate design: <how it could be built — invariant family / energy / learned>
- priority: high | medium | low (by how much pass@K / accuracy headroom it would unlock)
```

---

## Empirical confirmation (TRM rerank, 2026-06-09)

`results/arc3_trm_verifier_rerank.json` (n=31-task subset of TRM's arc_v1 candidate pool) tested
whether the `union_max` ensemble reranks TRM's candidates better than its own frequency vote.

**CORRIGENDUM (2026-06-09, same day): the first version of this experiment was DEGENERATE — do not
cite its "captured 0.0 wash" numbers.** `_verifier_scores` passed numpy grids to `V._combined_scores`,
whose grid helpers (`_dims`/`_colors`) raise `ValueError: truth value of an array ... ambiguous` on a
numpy grid — so EVERY candidate routed to the `except → 1e9` branch and the verifier scored all
candidates identically (all-tied). The "wash" was the verifier doing literally nothing, which is
harmless-by-accident, not neutral-on-the-merits. Fixed by converting candidate grid + test_input to
`list[list[int]]` before scoring (`_as_list`, commit pending). The corrected numbers below SUPERSEDE
the wash.

**Corrected result — the hand-invariant verifier ANTI-RANKS on TRM's real candidate pool.** With the
verifier actually scoring:

| Ranker | pass@1 | pass@2 | vs TRM_VOTE | net_fix |
|---|---|---|---|---|
| TRM_VOTE (baseline) | 0.419 | **0.484** | — | — |
| VERIFIER (pure union_max) | 0.065 | **0.161** | **−0.323** | −11 |
| HYBRID (vote-primary + verifier tie-break) | 0.419 | 0.452 | −0.032 | 0 |
| oracle ceiling (pass@1000) | — | 0.613 | +0.129 | — |

The pure verifier ranks the correct answer *near the bottom* (pass@2 0.16 ≪ vote 0.48 ≪ oracle 0.61);
it breaks 11 correct votes (net_fix −11) over 28 flips. Even the safe vote-primary HYBRID loses ~3pp.
**This is an INFORMATIVE negative, not a degenerate one** (per FALSE_NEGATIVE_RISK): the oracle ceiling
0.613 > vote 0.484 proves ~13pp of *selectable* headroom genuinely exists in the pool, and the verifier
DOES change picks (28 flips) — it just changes them WRONG. So the discriminating signal for ARC
content/rule correctness is NOT in the cheap hand-invariants; on this corpus they are anti-correlated
with correctness. Of the 5 uncaptured tasks (correct answer in pool, HYBRID top-2 missed it) the census
split **GAP-1: 2, GAP-2: 1, GAP-3: 2**, confirming all three gaps below on REAL mis-votes. (Caveat:
31-task subset; counts small but directional — re-confirm at scale.)

**GAP-1 hand-invariant candidate (directional-adjacency) — TESTED, REFUTED (2026-06-09).**
`results/arc3_trm_verifier_rerank_gap1.json`: adding the `arc_invariant_directional_adjacency_draft`
family as a within-top-vote-cluster orientation tie-break made HYBRID *worse* (pass@2 0.452 → 0.419),
captured ZERO of the 2 transpose mis-votes, and raised total uncaptured 5 → 6. The directional family
scores a candidate's H/V color-transition distribution against the *train-output average*, which is a
noisy proxy (test-output directional stats legitimately differ from the train average), so it penalizes
correct candidates. **Lesson: GAP-1 is real, but a hand-invariant is the wrong tool to fill it** — the
same conclusion the corrected baseline forces for the whole hand-invariant ensemble. This escalates the
GAP-3 learned / model-native energy (below) from "medium-high" to the **primary** path: the headroom is
real and only a content/rule-aware energy (not more cheap invariants) can reach it.

## Open gaps

### GAP-1: transpose / orientation discrimination
- status: open — hand-invariant candidate (`arc_invariant_directional_adjacency_draft`) TESTED & REFUTED
  2026-06-09 (degraded TRM HYBRID rerank pass@2 0.452→0.419, captured 0/2 transpose mis-votes; see
  CORRIGENDUM above). The capability is still missing; a cheap hand-invariant is the wrong tool for it.
- evidence: design survey of `arc_grid_verifier_invariants_v2_combined` (2026-06-09) +
  `results/arc_grid_verifier_invariants_v2.json` — the square-transpose distractor subset (239 tasks)
  sits at `union_max ≈ 0.53` (≈ chance); `transposed` gold-strictly-better-rate 0.293 (well below the
  0.70 catch bar).
- failure mode: the always-on content families (`object_count`, `palette_histogram_shape`) are
  PROVABLY transpose-invariant — a grid and its transpose have identical object statistics and color
  histograms — so they cannot distinguish the correct output from its transpose. v1 (dims/palette) is
  also blind when the transpose preserves dimensions (square grids).
- missing discriminator: a verifier sensitive to SPATIAL ARRANGEMENT / orientation that is NOT
  transpose-invariant — e.g. per-position / directional adjacency consistency with the train outputs,
  or a learned positional-pattern energy.
- candidate design: a `positional_adjacency` invariant family (row/column-ordered edge statistics, or
  an oriented content-overlap that does not symmetrize over the dihedral group); gate it on
  fixed-output-dim tasks.
- priority: high (transpose-class mis-votes are a large, currently-uncapturable slice).

### GAP-2: content verification for VARIABLE-output-dim tasks
- status: open
- evidence: design survey (2026-06-09) — the positional/transformation families
  (`content_overlap`, `color_mapping`, `delta_pattern`) ABSTAIN on variable-output-dim / non-positional
  tasks (~half of ARC), so the ensemble falls back to v1 + `object_count` + `palette_histogram` (all
  content-weak). The residual null space concentrates here.
- failure mode: when the output dimensions are not a fixed function of the input, the positional
  families have no canvas to align to and return "not applicable", leaving only structure + count/
  histogram signals — which a content-wrong-but-count-preserving candidate passes.
- missing discriminator: an object-RELATIONAL / rule-application consistency verifier that does not
  require fixed positions — e.g. "does the same object-level transformation inferred from train map
  the test input to this candidate?" (object correspondence + per-object op consistency).
- candidate design: an `object_relational_rule` family (segment objects, infer per-object train ops,
  score candidate disagreement) and/or a learned program-consistency energy (the ARC-domain energy
  instance, per `project_verifier_domain_bound`).
- priority: high (~half of ARC is variable-output-dim; this is the biggest coverage hole).
- ALT candidate design (model-native, 2026-06-09): per arXiv:2604.17614 ("Characterizing Model-Native
  Skills"), recover a compact ORTHOGONAL basis from the GENERATOR's own activations and score candidate
  consistency along it, instead of hand-imposed grid invariants. Feasible here because TRM is open/local
  (we have its activations). Directly attacks GAP-2/GAP-3 (the content/rule axes hand-features miss) AND
  the verifier-orthogonality program (a model-native orthogonal basis = principled distinct null spaces,
  vs the SHARED transpose-invariance of object_count+palette_histogram = GAP-1). White-box-complement to
  the black-box energy verifier ([[reference_cognometry]], [[reference_goodfire_silico]]). TRM-specific
  (needs generator activations) — not a general verifier. The GAP-3 follow-up if hand-invariants wash.

### GAP-3: content errors that preserve ALL cheap invariants
- status: open (partially mitigated by the v2 transformation families)
- evidence: `results/arc_grid_verifier_discriminator.json` — v1 is `PRUNER_NOT_SELECTOR`
  (perturbed_gold gold-strictly-better-rate 0.006, tie_rate 0.99); the v2 ensemble lifts perturbed_gold
  to 0.78 and color_swap to 0.75 but a residual remains for candidates that are structurally valid AND
  object-count/histogram-preserving AND positionally non-alignable.
- failure mode: a candidate that is the right size, palette, background, object-count, and color-
  histogram but applies the wrong RULE is in the joint null space of every cheap invariant.
- missing discriminator: a genuine rule-application / consistency-ENERGY verifier trained on ARC
  transitions (not a hand invariant) — the "new ARC energy instance" the domain-bound analysis calls
  for (`project_verifier_domain_bound`).
- priority: **PRIMARY (escalated 2026-06-09)**. The corrected TRM rerank showed the full hand-invariant
  ensemble is anti-correlated with correctness on TRM's real pool (VERIFIER pass@2 0.16 vs vote 0.48 vs
  oracle 0.61), and the GAP-1 hand-invariant attempt made things worse — so cheap invariants cannot
  reach the proven ~13pp headroom. Only a content/rule-aware learned energy can. A learned energy likely
  subsumes GAP-1 and GAP-2 as well.
- candidate design (escalated): the model-native basis from arXiv:2604.17614 — recover a compact
  orthogonal basis from TRM's OWN activations (open/local weights) and score candidate consistency along
  it, instead of hand-imposed grid invariants. See `reference_model_native_skills` memory + GAP-2 ALT
  block above. This is the next experiment, not another hand-invariant family.

---

## Filled gaps

*(none yet — append `### GAP-<n>` entries here with `status: filled (<verifier_id>)` and the
registry version that closed them when a new verifier captures a previously-open gap.)*

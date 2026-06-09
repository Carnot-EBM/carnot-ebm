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

## Diagnosis: WHY the hand-invariant ensemble anti-ranks (2026-06-09, operator: diagnose before building)

Two follow-up experiments decomposed the anti-ranking on TRM's real pool (n=31):

**`results/arc3_verifier_antirank_diagnosis.json`** — per-family discrimination AUROC (>0.5 = ranks gold
better) on TRM's REAL candidates (not the synthetic distractors the v2 verifier was tuned on):

| family | AUROC | applicable tasks |
|---|---|---|
| tiling_scaling | 0.91 | 3 |
| color_mapping | 0.71 | 6 |
| object_count | 0.67 | 31 (always-on) |
| content_overlap | 0.67 | 16 |
| palette_histogram_shape | 0.56 | 31 (always-on) |
| **delta_pattern** | **0.43** | 18 (ANTI) |
| **v1 structural** | **0.42** | 31 (ANTI) |
| symmetry | n/a | 0 |

The `union_max` = max() aggregation is the proximate killer (pass@2 0.19): it scores each candidate by
its WORST applicable family, so any family mis-firing on gold sinks it, and v1+delta_pattern are
anti-discriminative. `min_defined` recovers to ~vote (0.42) but no FIXED aggregation beats vote (0.45).
Sinking census (which family drives gold's max-violation on mis-ranks): content_overlap 7,
palette_histogram 4, object_count 2 — discriminative-on-average families that catastrophically mis-fire
on a subset, amplified by max().

**`results/arc3_verifier_learned_combiner_ceiling.json`** — the definitive test: an OUT-OF-FOLD learned
linear combiner (class-balanced leave-one-task-out logistic) that CAN down-weight the anti-families.
It learned sensible weights (object_count +1.48, tiling +0.54, content_overlap +0.44, color_mapping
+0.19; v1 −0.52, palette_histogram −0.23 inverted) — and STILL scored pass@2 0.23, well below vote 0.45.

**CONCLUSION — cheap hand-features are EXHAUSTED on TRM's real candidate pool.** Three independent
measurements agree (pure union_max 0.16, best fixed aggregation 0.42, best learned combiner 0.23 — none
beat vote 0.45). The structural reason: the only strongly-discriminative families (tiling 0.91,
color_mapping 0.71) fire on <20% of tasks; the always-on families top out at AUROC 0.67 — insufficient
to seat gold in top-2 against dozens of structurally-plausible candidates. **GAP-3 (a learned /
model-native ARC energy) is the CONFIRMED only path** to the proven ~13pp oracle headroom; GAP-1/GAP-2
(more hand-invariants) cannot close it. **CAVEAT (sample-size rigor):** 31-task/19-gold is tiny — the
direction is strong and triply-confirmed but re-confirm at full 400-task scale before any irreversible
strategy commitment.

## GAP-3 build log (2026-06-09)

**Stage 0 — TRM-native q_halt energy: NEGATIVE, adversarially confirmed → advance to Stage 1.**
`results/arc3_gap3_stage0_qhalt_energy.json`: TRM's own halting confidence (scalar q_halt, already in
the dump, zero GPU) does NOT beat its frequency vote as a model-native PRIMARY selection signal on the
real pool (n=31). Q_MEAN pass@2 0.290 < vote 0.452; the vote-residual ranker collapses to 0.097;
bootstrap of (Q_MEAN−vote) is [−0.290, −0.032], entirely below zero. The A0 vote-mimicry control caught
that the only "wins" (HYBRID/Q_LSE, +1 task) are vote-primary tie-break / vote-confounded, not
model-native. Nuance: q_mean has real SOFT within-task discrimination (AUROC 0.86 — gold beats most
non-golds) but the lossy 1-D scalar is neither sharp enough for top-2 nor vote-orthogonal.

**Adversarial-verify round** (`results/arc3_gap3_stage0_adversarial_verify.json`, GAP-3 design §4):
5 independent reviewers re-derived every number from the 8041-candidate table (no torch) and tried to
refute. Unanimous **NEGATIVE_CONFIRMED**, worst severity cosmetic. No reviewer found a de-confounded q
ranker that beats vote; the decisive fact — `headroom_capture_fraction=0` — is that on the 4 tasks where
gold is present but vote misses top-2, q_mean recovers ZERO (the scalar is redundant with vote exactly
where it would need to add value). Positive control holds (oracle 0.613 > vote 0.452, ~16pp real
headroom), so this is an honest negative. **Recommendation: GO for Stage 1** — the full
penultimate-activation latent (not collapsed to a scalar, not vote-redundant); the 0.86 soft-AUROC is a
leading indicator the latent carries sharper, vote-orthogonal signal. (Caveat: n=31; re-confirm at 400.)

**Stage 1 — TRM penultimate-latent energy (z_H[:,0], hidden=512): NEGATIVE, adversarially confirmed →
advance to Stage 2.** Re-run completed 2026-06-09 after the Fable-5 switch abort (fresh 29.4-min capped
latent dump, 100 batches, GPU 1; `results/arc3_gap3_stage1_latent_energy.json`). The out-of-fold LOTO
PCA-24 + balanced-logistic probe over z_mean does NOT beat vote: probe pass@2 0.4194 < vote 0.4516;
vote-residualized probe collapses to 0.0645; HYBRID adds nothing (0.4516); headroom_capture = −0.20
(probe recovers 1 of 5 recoverable tasks but LOSES 2 vote-wins, net −1); bootstrap Δpass@2 point −0.0323,
CI95 [−0.129, +0.0645].

**Adversarial-verify round** (`results/arc3_gap3_stage1_adversarial_verify.json`): 5 independent
reviewers re-derived every number from the exported 8041-candidate table + (8041,512) latents and swept
~300 alternative OOF/LOTO constructions (PCA-k 4–192, raw-512 logistic/ridge/LDA/Mahalanobis, kNN-to-gold,
centroid/softmin, RankNet, GBM stumps, feature unions, shortlist re-ranks). Unanimous **NEGATIVE_CONFIRMED**,
worst severity minor, zero de-confounded rankers beating vote. The round SHARPENED the Stage-0 leading
indicator into a refutation of it: the latent is a **partial vote shadow**, not an independent signal —
vote's own within-task AUROC 0.9235 > probe 0.8689, the vote-orthogonal residual has AUROC 0.3176 (below
chance), and the hard-negative (votes≥5) probe AUROC is 0.6646 (fails the 0.70 gate the macro number
passed). Mechanism: per-candidate mean-pooling of the latent over its `votes` augmentation views bakes
vote count into the feature (corr(‖z_mean‖, log-votes) ≈ −0.80 raw / −0.41 Spearman). On the deep-headroom
tasks gold is MORE latent-buried than vote-buried (t2: latent rank ~329 vs vote 65 of 599). Steelman
ceiling across the whole sweep: 0.4839 (+1 task, CI touches 0, equals vote's optimistic tie-break ceiling).
**Recommendation: GO for Stage 2** (trained generator-INDEPENDENT ARC transition-EBM) with design
constraints from this round: compute energy from grid CONTENT (no generator activations, no augmentation
pooling — the structural vote leak), use task-GROUPED LOTO at 400-task scale, and the baseline-to-beat is
the no-latent votes+q_mean+vote_share union (pass@2 0.4839; with-latent 0.5161 only ties Stage-0's
top-3-then-q_mean shortlist ceiling). Corrigendum details added in-place to the Stage-1 artifact
(`corrigendum_2026_06_09` block). (Caveat: n=31; CI upper edge +0.0645 cannot exclude a small positive —
re-confirm at 400.)

**Stage 2 v1 — trained generator-INDEPENDENT ARC transition-EBM: NEGATIVE, adversarially confirmed;
failure attributed to RECIPE, not concept.** Same-day build + run (2026-06-09,
`results/arc3_gap3_stage2_transition_ebm.json` + `..._adversarial_verify.json`, 5/5 NEGATIVE_CONFIRMED,
worst severity minor). A ~1.1M-param relation-network EBM (shared pair-encoder CNN; rule = mean over
demo-pair embeddings; E = MLP([r,p,r·p,|r−p|])), trained with InfoNCE (gold vs 10 synthetic corruption
negatives) on ARC-1 training (400) + ConceptARC (160) — `training2` EXCLUDED after an overlap audit
found 376/400 ARC-1 eval tasks inside it (29/30 pool tasks) — then evaluated zero-shot as a re-ranker on
TRM's real pool. It learns its curriculum (val top-1 0.59 vs 0.09 chance; on the REAL pool the lone
identity-copy ranks dead last, demo-copy AUROC 0.833) but is **statistically indistinguishable from a
RANDOM ranker on real candidates**: pass@2 0.1613 vs random-baseline 0.1432 (P(X≥5)=0.468) vs vote
0.4516; macro AUROC 0.5442; union value-add zero (31/31 top-2 sets unchanged); E's hit-set is a strict
subset of vote's, and complementarity is structurally impossible (Pareto-dominated gold on 4/5
vote-missed entries — no monotone vote+E combination can exceed the no-E union's 0.4839).

**Why (the quantified gap — this is the missing-discriminator spec):** only 3.8% of TRM's real wrong
candidates (28.2% by vote mass) belong to the 8 trained corruption families. **91.5% of real wrong
candidates share gold's SHAPE**; the dominant error classes are structured near-misses (29.7%, median
Hamming 0.10) and plausible-but-wrong rule applications (59.1%). On these the energy is at/below chance
(gold-vs-near-miss(≤5% cells) AUROC 0.481) and the global-mean-pooled encoder cannot even resolve
1–2-cell differences (c3202e5a: gold differs from the vote leader by ONE cell, ranks 747/754).

- **missing discriminator:** same-shape near-miss rule-consistency — score whether a candidate that
  matches gold's shape applies the demonstrated rule correctly at the 1–5%-cells-wrong resolution. The
  91.5% class where the trained E sits at ~0.52 pair-weighted AUROC and vote sits at 0.92.
- **candidate design (Stage-2 v2, ONE run behind hard gates):** (1) negatives = REAL wrong candidates
  mined from a generator run on TRAINING-split tasks only (different TRM seed/early checkpoint or a
  weaker non-TRM model — preserves eval-generator independence + task hygiene) + STRUCTURED near-miss
  synthesis (single-object recolor/move/delete, off-by-one placement) so same-shape ≤5%-cell errors
  dominate; (2) architecture drops global mean-pooling for spatial features / per-cell cross-attention
  (cellwise-diff resolvable); keep dihedral-8 TTA at eval (demonstrated free +0.03 pass@2). (3) GATES:
  gold-vs-same-shape-near-miss AUROC > 0.70 on held-out TRAINING tasks BEFORE any selection eval; at
  eval require pass@2 > 0.1432 (random), then > vote. If v2 still lands at random → RETIRE the
  trained-content-energy lineage and close GAP-3 with the honest bound below.
- **honest bound after three stages:** the 16pp oracle headroom (0.6129 vs vote 0.4516; 5
  present-but-mis-voted entries) is real, replicated, and untouched by scalar (Stage 0), latent
  (Stage 1), and trained content-energy v1 (Stage 2) selectors. Vote's own within-task AUROC 0.9235 is
  effectively a 1000-sample TRM self-ensemble — the bar any content energy must clear. The remaining
  serious candidates beyond v2 are execution/program-synthesis verification and hybrids, not cheaper
  energies.

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
- DESIGN (2026-06-09): full staged build plan at `docs/research-notes/gap3-learned-arc-energy-design.md`
  — Stage 0 TRM-native q_halt-confidence energy (NO new GPU; data already dumped), Stage 1 TRM
  penultimate-activation energy (model-native basis / probe; GPU dump), Stage 2 trained ARC
  transition-EBM (generator-independent). Per-stage empirical gates (selection > vote, AUROC > 0.70,
  coverage ≥ 80%, headroom-capture ≥ 30%) + adversarial checks (vote-mimicry, activation-shortcut,
  task-leak, oracle-leak, sample-size). Recommended first experiment: Stage 0 (runnable now, zero GPU).

---

## Filled gaps

*(none yet — append `### GAP-<n>` entries here with `status: filled (<verifier_id>)` and the
registry version that closed them when a new verifier captures a previously-open gap.)*

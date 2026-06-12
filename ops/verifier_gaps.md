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

**Stage 2 v2 — RAN SAME-DAY (operator "run it"); NEGATIVE, adversarially confirmed; LINEAGE RETIRED
per the pre-registered condition.** (`results/arc3_gap3_stage2v2_transition_ebm.json` +
`corrigendum_2026_06_09_stage2v2` + `..._adversarial_verify.json`, 5/5 NEGATIVE_CONFIRMED,
retire_lineage=true.) The v2 spatial FiLM/max energy + structured near-miss curriculum PASSED its gate
(synthetic near-miss AUROC 0.7893; seed-stable 0.79–0.83) and the model demonstrably resolves what it
trains on — yet selection landed statistically AT random AGAIN (pass@2 0.1613 vs random 0.1432,
P(X≥5)=0.468; real-candidate AUROC 0.46–0.49; every steelman tops out at exactly the no-E union 0.4839).
The round's forensics (all committed numbers reproduced bit-for-bit; severity "major" attaches to the
artifact's self-description):

- **The mined-real axis was a silent no-op** — miner de-aug bug keyed 240/250 mined negatives to empty
  input grids (unjoinable); effective real-error exposure was 10 negatives / 3 tasks / 0.04% of the
  curriculum. v2 was therefore a CLEAN test of the structured-synthetic arm alone.
- **The mining premise is structurally barren** — the "59.9% wrong" yield was a padding artifact off
  ~500×; TRM's genuine train-split error rate is 10/8,259 = 0.12% (memorization). No miner fix can
  supply a real-error-dominant curriculum from this checkpoint's training split.
- **Gate-1 was a self-licking ice cream cone** — the v1 model passes the identical synthetic gate at
  0.7291. Synthetic-negative gates measure curriculum-fitting, not transfer; v2's REAL-pool ≤5%-cell
  near-miss AUROC is 0.58–0.60 (vote: 0.95 on the same class).
- **The near-miss curriculum REGRESSED the dominant class** — same-shape plausible-but-wrong rule
  applications (59.1% of errors, 81.4% of wrong-pair mass) scored 0.502 macro / 0.3283 pair-weighted,
  below chance and worse than v1 (0.6283/0.4707): the curriculum teaches "gold + small defect = bad,"
  ranking coherent wrong rule-applications BELOW gold.
- Also recorded: TTA non-replication (hurt v2); independence violation (mined from the eval generator's
  own checkpoint — in the direction that should HELP, and it still landed at random).

**HONEST BOUND (lineage closed; full wording in the v2 synthesis):** the ~16pp present-but-mis-voted
oracle headroom (0.6129 vs vote 0.4516) is REAL but UNREACHED by scalar (q_halt), latent (z_H probe),
or trained-content-energy selectors (v1 + v2, two architectures, two curricula). These selectors master
what they train on but score AUROC 0.43–0.50 on the dominant real-error class; matching vote needs
per-task AUROC ~0.92–0.99; the lineage ceiling is ~0.63–0.69 with ~36% synthetic→real transfer. The
lineage is on `ops/exclusion_manifest.yaml` (`gap3_trained_content_energy_selector_retired_stage2v2_
2026_06_09`); re-open requires operator authorization + the pre-registered gate-1R (REAL-mined-negative
AUROC ≥ 0.70, held out from training AND checkpoint selection) passed BEFORE any selection eval.

### GAP-4: same-shape rule-application consistency (the missing discriminator, quantified by Stages 2v1+2v2)
- status: **open — FIRST POSITIVE LANDED (2026-06-09/10, PRELIMINARY, adversarially confirmed as
  scoped).** The rule-execution verifier (`results/arc3_gap4_rule_exec_verifier.json` +
  `corrigendum_2026_06_10_gap4` + `..._adversarial_verify.json`, 5/5 POSITIVE_CONFIRMED) implements
  exactly this gap's candidate design: codex (gpt-5.5, 44 calls/23 min) induces `def transform(grid)`
  from demo pairs only (no task id, no candidates, no gold in prompt); a model-free verifier requires
  exact reproduction of ALL demos (29/31 entries demo-perfect); the demo-perfect program executes on
  the test input and the GATED rerank promotes the exact-matching candidate else no-ops to vote.
  **Pool-restricted pass@2: vote 0.4516 → gated 0.5806 (+4 recovered / 0 lost at pass@2; ~0.80 of the
  oracle headroom), pass@1 0.4194 → 0.5484.** The recovered set is exactly the GAP-4 class the whole
  GAP-3 lineage scored ~chance on — including 17cae0c1 (gold at vote-rank 65) and c3202e5a (the
  one-cell-diff task). All numbers re-derived bit-exact by 5 hostile reviewers from the SAVED programs
  (`results/arc3_gap4_induced_programs.json`) + frozen pool with zero API calls; programs audited
  generic (no hardcoded grids; mutation-battery clean).
- **ESTABLISHED:** the mechanism (demo-fit execution verification reranks a candidate pool with zero
  pass@2 downside on this pool) and the gate's safety behavior. **NOT (yet) ESTABLISHED:** statistical
  significance (exact sign test p=0.0625 one-sided; tie-robust recovery count 3, delta +0.0968 under
  gold-favorable vote ties); verifier value independent of the generator (codex-standalone scores
  26/31 = 0.8387 against TRUE gold — ABOVE the 0.6129 pool ceiling; the trivial codex-first arm scores
  0.903; the lift is GENERATOR-attributable and the gate is the safety wrapper); the induction rate on
  uncontaminated tasks (30/31 pool tasks appear verbatim in public ARC-AGI-2 TRAINING data; gpt-5.5
  generator; the 29/31 rate has near-zero transfer value); 400-task scale; any local-model generator.
  Pass@1 caveat: one gate mis-fire (25094a63) — zero-loss is a pass@2 property.
- **Forward protocol (before `status: filled`):** sandboxed (no filesystem bypass) 400-task
  re-confirmation on a host without ARC solutions on disk, with archived transcripts, genuinely
  held-out tasks (ARC-AGI-2 eval / ConceptARC holdout / post-cutoff), the codex-first arm, cluster
  bootstrap + exact tests (zero-loss design needs ≥6 discordant wins for two-sided p<0.05), a hardened
  exec sandbox (timeout; block np.load/np.save/np.fromfile/type), and a local open-weight generator
  arm (Gemma-4/Qwen3.6) for the decentralization tier.

**ARC-2 TRANSFER PROBE (2026-06-10, 5/5 adversarially CONFIRMED, worst severity minor) — the
two-point transfer curve.** Same pipeline on a fresh pool from TRM arc_v2 scored on ARC-AGI-2 eval
(locally rebuilt arc2concept dataset, identifier-aligned; 31 entries / 23 unique tasks / 20,379
candidates; only 4 oracle entries — TRM arc_v2 is far weaker here). Sandbox hardened first (exec
timeout + numpy-I/O blocks; then a word-boundary fix after the round caught 'type('/'os.' substring
false-rejections). Full per-call transcripts archived — all 56 leak-clean (the no-oracle invariant is
now AUDITED, not just asserted). `results/arc3_gap4_arc2_rule_exec_verifier.json` (+
`corrigendum_2026_06_10_arc2`) + `..._adversarial_verify.json` + transcripts.

| | ARC-1 (contaminated/easy) | ARC-2 eval (reduced-exposure/hard) |
|---|---|---|
| induction (demo-perfect, unique tasks) | 28/30 (0.93) | 13/23 (0.57, post-regrade; Fisher p=4e-4) |
| precision P(true-gold \| demo-perfect) | 26/29 (0.897) | 8/17 (0.47; p=0.0085) |
| true-gold overall (entries) | 26/31 (0.839) | 8/31 (0.258; end-to-end p=8.2e-6) |
| rerank effect | +4/−0 (gate fired 16×) | nil (gate fired 0× — pred_in_pool=0; venue degenerate) |
| cost | 46 s/task | 251 s/task (5.4×) |

- **The decay is honest induction strain, NOT collapse-to-recall:** demo-overfit (demo-perfect but
  test-wrong) rises 10.3% → 50% — memorization predicts the OPPOSITE signature — and the maximally
  exposed task (16b78196, content-identical 5-years-public ARC-1-eval reuse) fails completely at
  demo_fit 0.0. Exposure does not function as recall in this pipeline. Corollary: ARC-1's 0.897
  precision is an UPPER BOUND on genuine induction; difficulty vs residual contamination cannot be
  decomposed by this design (evaluation2 is public-era; 2/23 tasks are verbatim ARC-1-eval reuses —
  say "reduced-exposure", never "held-out").
- **Where the rerank venue pays:** only where the candidate generator is strong-but-mis-voted (ARC-1).
  Where the generator is weak (ARC-2), value migrates to codex-standalone (0.258, 2× the pool oracle
  0.129 — descriptive; vs TRM arc_v2's global 0.0125, p~3e-7) and the gate is inert. The '+4/−0'
  safety record is ARC-1-specific: at ~0.5 precision the wrong predictions are near-misses (hamming
  0.002–0.032) and must not be promoted by an exact-match gate fed a richer pool.
- **Next moves (panel-ranked):** (1) consistency-ensemble precision fix — k=2–3 INDEPENDENT inductions
  + agreement gate (micro-evidence from saved histories: agreed pairs gold 3/3 vs 0.50 unconditional);
  (2) graded execution-energy gate (demo_fit==1 AND min-hamming ≤ ~0.005) — the round found the one
  live signal: a 1-cell-off prediction GRADED promoted to rank 1 where exact-match missed; validate on
  the non-degenerate ARC-1 venue first; (3) adaptive iteration caps + ≥600s timeouts (67% of ARC-2
  codex-seconds burned on never-perfect tasks; an oracle-hit task lost all 3 iterations to 300s
  timeouts); (4) the local open-weight generator arm; (5) a 400-task run is NOT yet worth it — the
  gate is structurally inert where TRM is weak; it becomes worth it only after (1)/(2), framed as
  generator+selector precision, not pool-rerank.

**PRECISION FIXES RAN (2026-06-10, 5/5 adversarially CONFIRMED with narrowings; the round also caught
a live artifact-clobber bug — record reconstructed from transcripts, guard added).**
(`results/arc3_gap4_arc2_consistency_ensemble.json` + `corrigendum_2026_06_10_precision_fixes` +
`..._precision_fixes_adversarial_verify.json`.)

- **Graded min-hamming gate (Part A): harmless everywhere, pays only at ARC-2 pass@1.** ARC-1: pass@2
  0.5806 FLAT from τ=0 to τ=2.0, zero pass@2 vote-wins lost — but all 4 recoveries are τ=0 exact-match
  fires, the relaxation adds nothing there, and "safety" is pool structure (the contaminated
  exact-match bimodality), not a measured τ boundary (band precision at τ≤0.02 is 1/7 = 14%). ARC-2
  τ=0.005 fires once (the 1/493-cells near-miss) doubling pass@1 0.0323→0.0645. **Production: τ=0.005
  ONLY + a vote-aware guard** (the 25094a63 exact-match mis-promotion over a 945-vote gold is
  τ-untouchable; the gate's sole pass@1 loss, gross +5/−1).
- **k=3 independent-induction agreement gate (Part B): the precision mechanism is real; the coverage
  collapse is arithmetic.** Any-of-3 demo-perfect 18/31; agreement on 1/31 (gold — but n=1, CP95
  [0.025,1.0], and the event is one of the 2 flagged ARC-1-reuse tasks; deduped with the probe: 2/2
  task-level, CP95 [0.158,1.0] — nothing claimable above the 0.52 baseline yet). Bottleneck =
  singletons (14/31 entries had exactly one demo-perfect sample, 8 of them gold); binomial predicts
  1.42 agreement events vs 1 observed. Single-shot per-call induction 0.196 vs the 3-iter chain's
  ~0.52-0.57 per lineage (chain-vs-variance confound: Fisher p=0.092). Disagreements = 2/3
  junk-killing, 1/3 recall cost. Correlated arms restore coverage at exactly the unconditional 0.5
  precision — **independence IS the precision signal**. Near-miss output clustering REFUTED (the only
  near-pair is wrong-wrong); no cheap observable separates gold from wrong demo-perfect programs
  (AUCs 0.43–0.62) — agreement is the only measured discriminator.
- **Banked successor (zero new codex):** tiered harness = chain+agreement hybrid (saved chained
  program + 2 archived fresh singles: measured 2/31 at 2/2 gold, doubles Part-B coverage free;
  exclude chain-iter0 pairs — provenance ambiguity) → graded gate τ=0.005 → vote. On weak-vote venues
  the EV-optimal composite is promote-ANY-demo-perfect (~0.306 expected pass@1 at 0.52 precision,
  ~9.5× vote) — measurable offline. **Next codex purchase when bought: k=3 CHAIN arms on the ~13
  chain-feasible tasks** (projects ~5/31 agreement-gold, ~2.5× the graded gate; also disambiguates
  the chain-vs-variance confound), pre-registered on clean tasks, ≥5/5 clean agreement-gold to beat
  0.52 at α=0.05. NOT k=5–7 singles (the binomial model overpredicts k=3 by ~3×; 21/31 entries have
  zero single-shot-reachable gold at any k).
- failure mode: candidates that match gold's SHAPE and look structurally coherent but apply the WRONG
  rule (59.1% of TRM's real errors; 81.4% of wrong-candidate pairs; median Hamming-to-gold 0.40).
  Every content signal tested scores ~chance here (hand-invariants ≤0.67; trained content-EBMs
  0.43–0.50 across v1+v2) while frequency vote scores 0.92–0.98 via generator-marginal signal.
- missing discriminator: **does the candidate output follow from applying the task's induced rule to
  the test input** — "is this the right transformation," not "is this grid damaged."
- candidate design: execution/program-synthesis verification — induce the rule as a program from the
  demo pairs, execute it on the test input, compare to the candidate (the M2-v3/v4 codex+consistency-
  energy stack already built for ARC-AGI-3 is the in-house precedent); or vote-primary hybrids gated by
  an executed-rule consistency check. Synthesizing the missing negative class IS program synthesis —
  which is why no cheaper energy can fake it.
- priority: HIGH — it is the entire residual 16pp headroom on the TRM rerank pool, and the same class
  blocks any generator-side verifier moat claim on ARC-AGI-1.

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

### GAP-5: demo-underdetermination detection (from the chain-arms round, 2026-06-10)
- status: open — distilled from the 446ef5d2 wrong-agreement anatomy
  (`results/arc3_gap4_chain_arms_adversarial_verify.json`).
- failure mode: THREE structurally disjoint demo-perfect programs (difflib 0.02–0.10)
  unanimously produce the SAME wrong test output (hamming-to-gold 0.459) — convergent
  wrong inference from demos that under-determine the rule. No per-entry agreement/
  ensemble policy can rescue it (even unanimity agrees on the wrong answer). This is the
  irreducible failure mode of the program-induction line: structural disjointness of the
  programs (difflib 0.020–0.101) confirms the convergence is due to demo ambiguity, not
  correlated sampling — no quorum design can rescue it.
- missing discriminator: a demo-underdetermination detector — "do these demos pin down
  the rule on THIS test input?" The one measured tripwire: task-level sibling-input
  disagreement (the same arms DISAGREE on the task's other test input) — post-hoc 5/5
  gold at 0.3125 coverage; needs fresh pre-registered confirmation (conductor follow-up
  #2 carries it as the tertiary gate).
- candidate design: task-level sibling-input disagreement tripwire — when the induced
  programs disagree on the task's SECOND test input (idx1) while agreeing on idx0, flag
  the task as underdetermined and abstain (return None / confidence=low). Measured
  performance: post-hoc 5/5 gold, 0.3125 coverage — strong precision, low recall.
  Alternative: a formal demo-coverage checker that enumerates whether the demo set has
  ambiguities (requires program synthesis over the demo manifold; expensive but principled).
  Near-term implementation: extend tiered_select() in gap4_program_induction_stack.py to
  accept multi-input per task and return agreement=False when sibling-input programs diverge.
- priority: MEDIUM-HIGH — it is the irreducible failure mode of the entire
  program-induction verifier line; everything else (coverage, arm quality, quorum
  design) is now engineering.

### GAP-4 Agreement Selector Closure (Exp 4023)
- status: retired as selector R&D; agreement is a CONFIDENCE LABEL ONLY, not a precision selector.
- evidence aggregated: chain-arms adversarial report narrowed the result to unestablished precision
  uplift; exp3988 was poison-skipped; exp3999 wrote `protocol_preregistered_pending_execution`
  with 0 Codex calls and 0 agreement events; exp4009 wrote `blocked_execution_floor_unmet` with
  0 Codex calls and 0 agreement events; the known-issues unfeedable-power finding says monolithic
  powered multi-call confirmations must be task-split or they will hit the conductor hard cap.
- retirement trigger: `retire_if_same_verdict` triggered after the repeated non-execution /
  non-confirmation lineage; no precision-confirmation v4 is proposed.
- product boundary: the shipped demo-fit execution safety-gate is KEPT. This retires the
  smart-selector agreement-precision R&D line; it does not delete the deployed trust gate in
  `gap4_program_induction_stack`.
- next work belongs in logged verifier gaps such as GAP-5 demo-underdetermination or in generator /
  execution-safety hardening, not another agreement-as-selector confirmation.

### GAP-CODE-EXEC-DEMOFIT: code hidden-semantic execution discriminator
- status: open
- evidence: `results/experiment_4032_offarc_exec_verifier_transfer_raw.json` measured no OFF-ARC demo-fit transfer (delta_pp=5.0, CI95=[0.0, 12.5]).
- failure mode: candidates can pass visible demo tests while failing hidden semantic tests.
- missing discriminator: code_demo_fit_visible_tests_do_not_discriminate_hidden_semantics.
- candidate design: enrich the code verifier with hidden-property synthesis, stronger metamorphic tests, or formal/runtime oracles beyond visible examples.
- priority: high

### GAP-DECENTRALIZATION-MOE-BASE-4048: MoE-base local support gaps
- status: open
- evidence: /home/ianblenke/github.com/ianblenke/carnot/results/experiment_4048_decentralization_moe_base.json; n_tasks_scored=6; coverage=0.5; diagnosis=uninformative
- failure mode: Qwen3.6-35B-A3B best-of-N did not surface a demo-perfect local program for the listed ARC tasks under the cached verifier ensemble.
- missing discriminator: a verifier-side signal that can recognize the demonstrated rule when the local generator has not already produced a demo-perfect candidate.
- candidate design: extend GAP-4 execution/program induction or add a larger-base candidate source before distillation.
- priority: high
- missing_verifier_gaps: 17cae0c1, 1a2e2828, 1a6449f1

<!-- exp4051-g1:start -->
#### Exp 4051 G1 off-ARC power update for GAP-CODE-EXEC-DEMOFIT
- status: g1_off_arc_power_pending
- evidence: `results/experiment_4045_offarc_transfer_power.json`; n_tasks=22; powered_task_floor=160; demo_fit_CI95=[0.0, 0.0]; best_arm=armC_symbolic; best_arm_CI95=[0.0, 0.0].
- failure mode: visible/demo-fit code tests are not yet a powered hidden-semantic discriminator.
- missing discriminator: code_demo_fit_visible_tests_do_not_discriminate_hidden_semantics.
- candidate design: finish the powered code run or add hidden-property, symbolic, or formal/runtime oracles.
- priority: high
<!-- exp4051-g1:end -->

<!-- exp4051-g2:start -->
### GAP-ARC3-VC33-SIM2REAL-CEILING: vc33 verified-WM closed-loop sim2real ceiling
- status: g2_sim2real_ceiling_gap_logged
- evidence: `results/experiment_4046_closed_loop_replan_over_vc33_wm.json`; per_step_wm_real_divergence_rate=0.207031; divergence_gate_fired_count=1.
- failure mode: bounded WM search produced a plan whose predicted next state diverged from the real environment.
- missing discriminator: per-step WM-to-real transition trust signal strong enough to plan past vc33's wall.
- candidate design: improve the verified world model or add a conservative real-env grounding/replan guard.
- priority: high
<!-- exp4051-g2:end -->

<!-- exp4063-g1:start -->
#### Exp 4063 G1 EvalPlus update for GAP-CODE-EXEC-DEMOFIT
- status: g1_evalplus_accumulating
- evidence: `results/experiment_4057_offarc_power_evalplus.json`; accumulated_n_tasks=0; powered_task_floor=160; oracle_headroom_present=False; demo_fit_CI95=[0.0, 0.0]; best_arm=armC_symbolic; best_arm_CI95=[0.0, 0.0].
- failure mode: visible/demo-fit code tests are not yet a powered EvalPlus hidden-semantic discriminator.
- missing discriminator: code_demo_fit_visible_tests_do_not_discriminate_hidden_semantics.
- candidate design: continue accumulation or add hidden-property, symbolic, formal, or runtime oracles.
- priority: high
<!-- exp4063-g1:end -->

<!-- exp4063-g3:start -->
#### Exp 4063 G3 accumulated update for GAP-DECENTRALIZATION-MOE-BASE-4048
- status: g3_decentralization_moe_base_4048_pending
- evidence: `results/experiment_4059_decentralization_moe_resume.json`; accumulated_coverage=0.0; n_tasks_scored=0; ACCUMULATED-N=0; diagnosis=pending; bootstrap_CI95=[0.0, 0.0].
- failure mode: local MoE best-of-N support has not established a sovereign GAP-4 replacement.
- missing discriminator: verifier-side signal or stronger local base that surfaces demo-perfect programs without Codex.
- candidate design: continue accumulation, use a stronger local base, or add verifier-guided generation before distillation.
- priority: high
<!-- exp4063-g3:end -->

<!-- exp4063-g2:start -->
### GAP-ARC3-VC33-SIM2REAL-CEILING: vc33 verified-WM closed-loop sim2real ceiling
- status: g2_vc33_sim2real_ceiling_logged
- evidence: `results/experiment_4046_closed_loop_replan_over_vc33_wm.json`; per_step_wm_real_divergence_rate=0.207031; divergence_gate_fired_count=1.
- failure mode: bounded WM search produced a plan whose predicted next state diverged from the real environment.
- missing discriminator: per-step WM-to-real transition trust signal strong enough to plan past vc33's wall.
- candidate design: improve the verified world model or add a conservative real-env grounding/replan guard.
- priority: high
<!-- exp4063-g2:end -->

### GAP-DECENTRALIZATION-MOE-SYNC-4069: Synchronous MoE local support gaps
- status: open
- evidence: /home/ianblenke/github.com/ianblenke/carnot/results/experiment_4069_decentralization_moe_sync.json; accumulated_n=30; coverage=0.2333; diagnosis=absent
- failure mode: Qwen3.6-35B-A3B best-of-N did not surface a demo-perfect local program for the listed ARC tasks under the unchanged GAP-4 verifier.
- missing discriminator: a local candidate source or verifier-side signal that recovers the demonstrated rule before distillation.
- priority: high
- missing_verifier_gaps: 17cae0c1, 1a2e2828, 1a6449f1, 25094a63, 2f0c5170, 4e469f39, 505fff84, 50a16a69, 5833af48, 692cd3b6, 712bf12e, 79fb03f4, 81c0276b, 8719f442, 96a8c0cd, a57f2f04, ac3e2b04, b9630600, c3202e5a, c7d4e6ad, f0df5ff0, f3e62deb, fafd9572

<!-- exp4073-g1:start -->
#### Exp 4073 G1 corpus-routed update for GAP-CODE-EXEC-DEMOFIT
- status: g1_evalplus_accumulating
- evidence: `results/experiment_4068_offarc_transfer_power_sync.json`; corpus=evalplus; accumulated_n_tasks=160; powered_task_floor=160; oracle_headroom_present=False; oracle_passrate=0.9625; demo_fit_CI95=[0.0, 3.125]; best_arm=armC_symbolic; best_arm_CI95=[0.0, 3.125]; route=12B oracle headroom present on EvalPlus (0.7500 < 0.95); route stays on cheap hidden tests..
- failure mode: visible/demo-fit code tests are not yet a powered hidden-semantic discriminator.
- missing discriminator: code_demo_fit_visible_tests_do_not_discriminate_hidden_semantics.
- candidate design: continue accumulation on a corpus with oracle headroom or add hidden-property, symbolic, formal, or runtime oracles.
- priority: high
<!-- exp4073-g1:end -->

<!-- exp4073-g3:start -->
#### Exp 4073 G3 synchronous update for GAP-DECENTRALIZATION-MOE-BASE-4048
- status: g3_decentralization_absent_coverage_0.2333
- evidence: `results/experiment_4069_decentralization_moe_sync.json`; accumulated_coverage=0.2333; ACCUMULATED-N=30; n_demo_perfect_tasks=7; diagnosis=absent; bootstrap_CI95=[-0.1581, 0.1419]; missing_verifier_gaps=['17cae0c1', '1a2e2828', '1a6449f1', '25094a63', '2f0c5170', '4e469f39', '505fff84', '50a16a69', '5833af48', '692cd3b6', '712bf12e', '79fb03f4', '81c0276b', '8719f442', '96a8c0cd', 'a57f2f04', 'ac3e2b04', 'b9630600', 'c3202e5a', 'c7d4e6ad', 'f0df5ff0', 'f3e62deb', 'fafd9572'].
- failure mode: local MoE best-of-N support has not established a sovereign GAP-4 replacement.
- missing discriminator: verifier-side signal or stronger local base that surfaces demo-perfect programs without Codex.
- candidate design: continue accumulation, use a stronger local base, or add verifier-guided generation before distillation.
- priority: high
<!-- exp4073-g3:end -->

<!-- exp4073-g2:start -->
### GAP-ARC3-VC33-SIM2REAL-CEILING: vc33 verified-WM closed-loop sim2real ceiling
- status: g2_vc33_sim2real_ceiling_logged
- evidence: `results/experiment_4046_closed_loop_replan_over_vc33_wm.json`; per_step_wm_real_divergence_rate=0.207031; divergence_gate_fired_count=1.
- failure mode: bounded WM search produced a plan whose predicted next state diverged from the real environment.
- missing discriminator: per-step WM-to-real transition trust signal strong enough to plan past vc33's wall.
- candidate design: improve the verified world model or add a conservative real-env grounding/replan guard.
- priority: high
<!-- exp4073-g2:end -->

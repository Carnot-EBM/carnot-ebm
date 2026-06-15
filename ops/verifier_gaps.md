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

<!-- exp4083-pivot:start -->
### GAP-TRAINING-VERIFIER-AS-REWARD-RFT-4079: training-time verifier role for verifier-as-reward RFT
- status: verifier_as_reward_rft_blocked
- evidence: `results/experiment_4079_verifier_reward_rft_eval_collect.json`; status=blocked; honest_verdict=blocked_gate_check_failed; blocked_at_layer=conductor_pre_gate; gate_check_summary=1 of 1 gate(s) failed; first failure: exp4078-verifier-reward-rft-train-launch.train_launched (actual=False == expected=True).
- failure mode: the verifier-as-reward training-time role is not yet a decision-grade win; Exp 4079 must beat the cold base and the label ablation on held-out induction before promotion.
- missing discriminator: decision-grade evidence that verifier-certified training carries signal beyond codex-distillation and gold-SFT controls.
- candidate design: unblock the train launch, rerun Exp 4079 on the held-out ladder, and keep the RFT-correct vs RFT-ablation contrast load-bearing.
- priority: high
<!-- exp4083-pivot:end -->

<!-- exp4087-gap5:start -->
#### Exp 4087 certification precision-rescue update for GAP-5
- status: precision_rescue_succeeded
- evidence: `results/experiment_4087_certification_precision_rescue.json`; best_certified_precision=0.8824; best_op_point_recall=0.7143; n_tasks_scored=21; n_codex_calls=0.
- outcome: complete: precision_rescue_succeeded_best_0.8824_at_recall_0.7143.
- implication: GAP-5 demo-underdetermination remains the Phase B RFT gate unless the recorded frontier contains a qualifying precision/recall operating point.
<!-- exp4087-gap5:end -->

<!-- exp4095-precision-rescue:start -->
### GAP-5-CERTIFICATION-PRECISION-RESCUE-4087: Exp 4095 precision-rescue registry update
- status: precision_rescue_succeeded
- evidence: `results/experiment_4087_certification_precision_rescue.json`; best_certified_precision=0.8824; best_op_point_recall=0.7143; filter_stack=k_of_n_agreement; threshold=k=1; n_certified=17; any_stack_reached_0_85=true.
- failure mode: raw demo-perfect certification can carry false positives when visible demos underdetermine the hidden test transformation.
- missing discriminator: certification-precision calibration strong enough for reward-data use.
- candidate design: retain the Exp 4087 operating point until a stronger calibrated filter beats its precision/recall tradeoff on held-out tasks.
- priority: high
<!-- exp4095-precision-rescue:end -->

<!-- exp4095-rft-a-vs-b:start -->
### GAP-RFT-A-VS-B-4090: Exp 4095 RFT A-vs-B outcome update
- status: rft_a_vs_b_pending_absent
- evidence: `results/experiment_4090*.json`; present=false; status=pending; honest_verdict=; arm_a_vs_b_delta=None.
- failure mode: verifier-as-reward RFT has not produced a decision-grade A-vs-B win unless the held-out Exp 4090 artifact exists and reports it.
- missing discriminator: measured evidence that the verifier-certified arm beats the ablation arm under the same training/eval pipeline.
- candidate design: run or consume Exp 4090 only after the precision-calibrated corpus and training artifacts exist; keep absent/pending state out of headline claims.
- priority: high
<!-- exp4095-rft-a-vs-b:end -->

<!-- exp4103-trm-grid-discrimination:start -->
### GAP-TRM-GRID-DISCRIMINATION: Exp 4112 .379 TRM-grid anti-discrimination update
- status: open_anti_discrimination_captured_pp_-0.2258
- evidence: `results/experiment_4099_trm_pool_verifier_discrimination_probe.json`; best_reranker=K_OF_N_AGREEMENT; best_captured_pp=0.0; captured_pp=-0.2258; captured_pp_rounded=-0.23; captured_pp_ci95=[-0.3387, -0.1129]; anti_discriminating_rerankers=['AUG_INVARIANCE', 'DEMO_FIT', 'STACK_DEMO_AUG']; verifier_beats_trm_vote=false; pool_n_tasks=62.
- failure mode: the correct TRM grid can be present in the candidate pool but remain unselectable; the measured verifier rerankers either tie vote or actively anti-rank against it.
- missing discriminator: signal separating a correct TRM grid from a confident-wrong TRM grid on the candidate pool.
- candidate design: treat the anti-discrimination as a missing-verifier spec until a held-out discriminator beats TRM vote without relying on neutral vote fallback.
- priority: high
<!-- exp4103-trm-grid-discrimination:end -->

<!-- exp4103-trm-rft-outcome:start -->
### GAP-TRM-VERIFIER-RFT-4100: Exp 4100 TRM verifier-RFT outcome
- status: smoke_checkpoint_ok
- evidence: `results/experiment_4100_trm_verifier_rft_conditional.json`; branch_taken=smoke; trm_native_trainer_checkpoint_ok=true; rft_vs_ablation_delta={'ci95': [0.0, 0.0], 'delta': 0.0, 'metric': 'heldout_pass@2', 'status': 'not_run_no_verifier_signal'}; bottleneck=verifier_discrimination_on_trm_grids.
- failure mode: verifier-as-reward RFT cannot be promoted when the upstream grid reranker captured 0.0pp; the mechanism smoke proves checkpoint plumbing, not a reward win.
- missing discriminator: decision-grade evidence that verifier-certified TRM training beats the vote-label ablation on held-out grid induction.
- candidate design: rerun full RFT only after a non-TRM grid reranker clears the Exp 4099 discrimination gate.
- priority: high
<!-- exp4103-trm-rft-outcome:end -->

<!-- exp4112-sudoku-executable-verifier:start -->
### GAP-SUDOKU-EXECUTABLE-VERIFIER-4109: Exp 4112 .380 Sudoku executable-verifier update
- status: open_honest_null_no_value_added
- evidence: `results/experiment_4109_carnot_verifier_graft_sudoku.json`; verifier_value_added=false; native_training_launched=false; rft_vs_ablation_delta={'a_exact_accuracy': 1.0, 'b_exact_accuracy': 1.0, 'ci95': [0.0, 0.0], 'delta': 0.0, 'metric': 'heldout_exact_accuracy', 'n_matched': 15, 'status': 'honest_null_ci95_includes_zero'}; rerank_delta=0.0; n_matched=15.
- failure mode: executable Sudoku constraints can score candidate validity, but Exp 4109 did not show value over the vote-label ablation on the matched executable-domain corpus.
- missing discriminator: decision-grade evidence that executable verifier labels add training-time value beyond vote labels on held-out TRM Sudoku induction.
- candidate design: rerun only with native training launched or a corpus where the vote baseline leaves measurable headroom; keep the .380 result as an honest null meanwhile.
- priority: high
<!-- exp4112-sudoku-executable-verifier:end -->

<!-- exp4122-sudoku-baseline-reproduction:start -->
### GAP-SUDOKU-BASELINE-REPRODUCTION-4118: Exp 4122 .381 Sudoku baseline reproduction status
- status: open_baseline_not_reproduced_val_0.1060
- evidence: `results/experiment_4116_sudoku_extreme_resume_pass1.json`, `results/experiment_4117_sudoku_extreme_resume_pass2.json`, `results/experiment_4118_sudoku_extreme_resume_pass3.json`; val_trajectory=[0.0854, 0.0966, 0.106]; final_val=0.106; matches_published_087=false; published_target=0.87; total_cumulative_epochs=4300.
- failure mode: the nano-TRM Sudoku checkpoint resumed and improved, but the validation exact accuracy remains far below the published baseline, so verifier training-time claims over this pool are underpowered.
- missing discriminator: faithful resumed TRM Sudoku candidate source before training-time verifier claims.
- candidate design: continue the stable baseline reproduction or move the executable verifier into adaptive candidate expansion before treating it as a reward-training signal.
- priority: high
<!-- exp4122-sudoku-baseline-reproduction:end -->

<!-- exp4122-sudoku-verifier-graft:start -->
### GAP-SUDOKU-EXECUTABLE-VERIFIER-4119: Exp 4122 .381 Sudoku executable-verifier graft status
- status: open_graft_deferred_verifier_value_added_false
- evidence: `results/experiment_4119_carnot_verifier_graft_sudoku.json`; graft_deferred=true; verifier_value_added=false; flagged_adversarial=true; baseline_final_val=0.106; baseline_trajectory=[0.0854, 0.0966, 0.106].
- failure mode: Exp 4119 did not run a meaningful graft because the .381 baseline was not reproduced; the executable verifier therefore has no decision-grade training-time TRM value-added result.
- missing discriminator: decision-grade training-time value from executable verifier labels beyond vote labels on held-out TRM Sudoku induction.
- candidate design: rerun the graft only after baseline reproduction or after verifier-guided candidate expansion creates a candidate pool with measurable oracle headroom.
- priority: high
<!-- exp4122-sudoku-verifier-graft:end -->

<!-- exp4131-lr-resume-fix:start -->
### GAP-SUDOKU-LR-RESUME-FIX-4126: Exp 4131 .382 LR resume correctness status
- status: fixed_lr_resume_continuous
- evidence: `results/experiment_4126_lr_resume_correctness_fix.json`; lr_continuous_across_resume=true; validation_first_lr=9.998933091992512e-05; fresh_warmup_lr=2.4500000108673703e-06; manual_lr_step_restored=4300; full_batch_validation_attempt=blocked_cuda_oom_before_metrics.
- failure mode: previous bounded Sudoku resumes rewarmed the manual LR schedule, making verifier training-time claims underpowered and hard to interpret.
- missing discriminator: faithful LR-schedule continuity before treating resumed TRM candidate pools as reward-training evidence.
- candidate design: build on the fixed stable checkpoint lineage and keep measuring validation until the baseline is faithful enough for grafting.
- priority: high
<!-- exp4131-lr-resume-fix:end -->

<!-- exp4131-sudoku-baseline-reproduction:start -->
### GAP-SUDOKU-BASELINE-REPRODUCTION-4127: Exp 4131 .382 Sudoku fixed-LR baseline status
- status: open_baseline_not_reproduced_val_0.2782
- evidence: `results/experiment_4127_sudoku_extreme_accumulate_fixed.json` with LR fix `results/experiment_4126_lr_resume_correctness_fix.json`; val_trajectory=[0.106, 0.2782]; final_val=0.2782; matches_published_087=false; published_target=0.87; lr_continuous_across_resume=true; per_pass_delta_vs_v381={'beats_v381': True, 'comparison': 'faster_than_v381', 'deltas': [0.172182761133], 'mean_delta': 0.172182761133, 'reference_delta': 0.01}.
- failure mode: the fixed-LR nano-TRM Sudoku checkpoint improved much faster than the .381 rewarm runs but remains far below the published baseline, so verifier training-time claims over this pool are still underpowered.
- missing discriminator: faithful fixed-LR TRM Sudoku candidate source before training-time verifier claims.
- candidate design: continue the fixed checkpoint lineage or use verifier-guided candidate expansion before treating executable constraints as a reward-training win.
- priority: high
<!-- exp4131-sudoku-baseline-reproduction:end -->

<!-- exp4131-sudoku-verifier-graft:start -->
### GAP-SUDOKU-EXECUTABLE-VERIFIER-4128: Exp 4131 .382 Sudoku executable-verifier graft status
- status: open_graft_deferred_verifier_value_added_false
- evidence: `results/experiment_4128_carnot_verifier_graft_sudoku.json`; graft_deferred=true; verifier_value_added=false; verifier_value_added_meaningful=false; flagged_adversarial=true; baseline_final_val=0.2782; baseline_trajectory=[0.106, 0.2782]; estimated_passes_to_converge_for_383={'basis': 'latest_fixed_lr_pass_delta', 'current_val_exact_accuracy': 0.278172343969, 'destination': '.383', 'estimated_additional_passes': 4, 'observed_delta_per_pass': 0.172183, 'previous_val_exact_accuracy': 0.105989582837, 'target_val_exact_accuracy': 0.85}.
- failure mode: Exp 4128 did not run a meaningful graft because the .382 baseline was still not reproduced; the executable verifier therefore has no decision-grade training-time TRM value-added result.
- missing discriminator: decision-grade training-time value from executable verifier labels beyond vote labels on held-out TRM Sudoku induction.
- candidate design: rerun the graft only after baseline reproduction or after verifier-guided candidate expansion creates a candidate pool with measurable oracle headroom.
- priority: high
<!-- exp4131-sudoku-verifier-graft:end -->

<!-- exp4142-sudoku-baseline-reproduction:start -->
### GAP-SUDOKU-BASELINE-REPRODUCTION-4138: Exp 4142 .383 Sudoku baseline trajectory status
- status: open_baseline_config_blocked_val_0.2782
- evidence: `results/experiment_4138_sudoku_accumulate_pass4_convergence_check.json`; baseline_status=config-blocked; val_trajectory_383=[0.2782, None, None, None, None]; measured_val_trajectory=[0.2782]; final_val=0.2782; matches_published_087=false; near_faithful_080=false; published_target=0.87; estimated_passes_to_converge=None.
- failure mode: the .383 continuation did not produce new validation progress because the baseline lineage was config-blocked before pass4, so the Sudoku candidate source remains far below the published 0.87 target.
- missing discriminator: faithful Sudoku baseline candidate source before DiffusionGemma scale-up or verifier-as-reward claims.
- candidate design: fix the Timer/config-blocked resume path or run a clean contiguous baseline before rerunning the graft.
- priority: high
<!-- exp4142-sudoku-baseline-reproduction:end -->

<!-- exp4142-sudoku-decisive-graft:start -->
### GAP-SUDOKU-EXECUTABLE-VERIFIER-4139: Exp 4142 .383 Sudoku decisive executable-verifier graft status
- status: open_graft_deferred_no_transferable_value_added
- evidence: `results/experiment_4139_decisive_verifier_graft_sudoku.json` with baseline `results/experiment_4138_sudoku_accumulate_pass4_convergence_check.json`; baseline_final_val=0.2782; headroom_present=false; graft_deferred=true; executable_verifier_is_oracle=true; executable_oracle_upper_bound_delta=0.0; ensemble_rerank_lift_vs_vote_delta=0.0; ensemble_rerank_lift_vs_vote_status=uninterpretable_no_headroom; rft_vs_ablation_delta=0.0; rft_vs_ablation_delta_status=deferred_baseline_below_0.80; verifier_value_added=false; diffusiongemma_gate_state=kept_gated.
- failure mode: executable Sudoku validity is an oracle upper bound on unique-solution Sudoku, not a transferable verifier reward. The non-oracle ensemble rerank added no measured lift, and the RFT label contrast was deferred because the baseline was below the near-faithful gate.
- missing discriminator: transferable training-time value from non-oracle Sudoku verifier labels beyond vote labels.
- candidate design: keep DiffusionGemma gated until the transferable ensemble rerank or RFT A-vs-B label contrast shows value with selectable headroom.
- priority: high
<!-- exp4142-sudoku-decisive-graft:end -->

<!-- exp4153-sudoku-baseline-reproduction:start -->
### GAP-SUDOKU-BASELINE-REPRODUCTION-4149: Exp 4153 .384 Sudoku baseline trajectory status
- status: open_baseline_blocked_pass3_noop_val_0.2782
- evidence: `results/experiment_4149_sudoku_accumulate_pass4_convergence.json`; baseline_status=blocked_pass3_noop_unresolved; raw_val_trajectory_v384=[0.2782, None, None, None, 0.2782]; effective_val_trajectory_v384=[0.2782, 0.2782, 0.2782, 0.2782, 0.2782]; final_val=0.2782; matches_published_087=false; faithful_for_graft_085=false; published_target=0.87.
- failure mode: the .384 continuation did not produce real training progress; the pass1/pass2/pass3 no-op lineage carried forward and pass4 preserved the 0.2782 baseline rather than approaching the published target.
- missing discriminator: faithful Sudoku baseline candidate source before DiffusionGemma scale-up or verifier-as-reward claims.
- candidate design: resolve the timer/no-op checkpoint lineage or create a clean contiguous baseline before rerunning the graft.
- priority: high
<!-- exp4153-sudoku-baseline-reproduction:end -->

<!-- exp4153-sudoku-decisive-graft:start -->
### GAP-SUDOKU-EXECUTABLE-VERIFIER-4150: Exp 4153 .384 Sudoku decisive executable-verifier graft status
- status: open_graft_deferred_baseline_below_0.85
- evidence: `results/experiment_4150_decisive_verifier_graft_sudoku.json` with baseline `results/experiment_4149_sudoku_accumulate_pass4_convergence.json`; baseline_final_val=0.2782; baseline_faithful_085=false; graft_deferred=true; candidate_source=none_baseline_below_0.85; n_candidate_pools=0; rerank_lift_vs_vote_delta=0.0; rerank_lift_vs_vote_status=deferred_baseline_below_0.85; rft_vs_ablation_delta=0.0; rft_vs_ablation_delta_status=deferred_baseline_below_0.85; verifier_value_added=false; diffusiongemma_gate_state=kept_gated.
- failure mode: Exp 4150 correctly deferred the graft because the baseline was below the faithful 0.85 gate; no rerank or RFT candidate source was created, so there is no transferable verifier-value-added evidence.
- missing discriminator: transferable training-time value from non-oracle Sudoku verifier labels beyond vote labels.
- candidate design: keep DiffusionGemma gated until rerank or RFT A-vs-B label contrast shows value on a faithful baseline.
- priority: high
<!-- exp4153-sudoku-decisive-graft:end -->

<!-- exp4163-sudoku-baseline-reproduction:start -->
### GAP-SUDOKU-BASELINE-REPRODUCTION-4157: Exp 4163 .385 Sudoku baseline trajectory status
- status: open_baseline_blocked_noop_step_unchanged_val_0.5010_flagged
- evidence: `results/experiment_4157_baseline_harvest_contiguous_continue.json`; honest_verdict=blocked_noop_step_unchanged; current_val=0.501; max_val=0.501; baseline_faithful=false; val_trajectory_385=[0.2005, 0.1992, 0.1995, 0.2216, 0.2042, 0.2211, 0.2443, 0.257, 0.2823, 0.3008, 0.3198, 0.3385, 0.3555, 0.3721, 0.3862, 0.4, 0.412, 0.4263, 0.4354, 0.4427, 0.4521, 0.4648, 0.4776, 0.4857, 0.4906, 0.4938, 0.501]; native_trainer_launched=true; flagged_adversarial=true.
- failure mode: the .385 continuation advanced the visible validation trajectory to about 0.5010 but still did not reach the faithful 0.85 gate, and the source artifact carries flagged caveats plus a blocked/no-op/OOM verdict.
- missing discriminator: faithful Sudoku baseline candidate source before DiffusionGemma scale-up or verifier-as-reward claims.
- candidate design: continue or relaunch the baseline under a clean resource envelope, then rerun rerank/graft only after the candidate source has faithful accuracy and selectable headroom.
- priority: high
<!-- exp4163-sudoku-baseline-reproduction:end -->

<!-- exp4163-sudoku-rerank-moat:start -->
### GAP-SUDOKU-RERANK-RECOVERY-MOAT-4158: Exp 4163 .385 Sudoku executable-verifier rerank moat status
- status: open_rerank_uninformative_no_headroom_flagged
- evidence: `results/experiment_4158_verifier_rerank_recovery_moat.json`; headroom_present=false; oracle_at_k=0.140625; vote_at_1=0.140625; verifier_recovers_outvoted=0; rerank_lift_vs_vote_delta=0.0; rerank_lift_vs_vote_ci95=[0.0, 0.0]; ci_excludes_zero_positive=false; flagged_adversarial=true; diffusiongemma_gate_state=kept_gated.
- failure mode: Exp 4158 reported no selectable rerank headroom and zero outvoted recoveries, so the executable checker did not produce a decision-grade rerank moat signal.
- missing discriminator: decision-grade executable Sudoku rerank signal with selectable headroom and a positive CI excluding zero.
- candidate design: rerun on a faithful checkpoint/pool where oracle@K exceeds vote@1, then promote only if rerank lift has a positive confidence interval.
- priority: high
<!-- exp4163-sudoku-rerank-moat:end -->

<!-- exp4163-sudoku-decisive-graft:start -->
### GAP-SUDOKU-EXECUTABLE-VERIFIER-4159: Exp 4163 .385 Sudoku decisive executable-verifier graft status
- status: open_graft_deferred_baseline_below_0.85_flagged
- evidence: `results/experiment_4159_decisive_verifier_reward_graft.json` with baseline `results/experiment_4157_baseline_harvest_contiguous_continue.json`; baseline_current_val=0.501; baseline_faithful=false; graft_deferred=true; candidate_source=none_baseline_below_0.85; n_candidate_pools=0; rft_vs_ablation_delta=0.0; rft_vs_ablation_delta_status=deferred_baseline_below_0.85; verifier_value_added=false; flagged_adversarial=true; diffusiongemma_gate_state=kept_gated.
- failure mode: Exp 4159 deferred the training-time graft because the baseline remained below the faithful 0.85 threshold; no verifier-as-reward value-added claim is available.
- missing discriminator: transferable training-time value from non-oracle Sudoku verifier labels beyond vote labels.
- candidate design: keep DiffusionGemma gated until rerank or RFT A-vs-B label contrast shows value on a faithful baseline.
- priority: high
<!-- exp4163-sudoku-decisive-graft:end -->

<!-- exp4171-sudoku-baseline-reproduction:start -->
### GAP-SUDOKU-BASELINE-REPRODUCTION-4167: Exp 4171 .386 outer-loop Sudoku baseline trajectory status
- status: open_outerloop_training_alive_val_0.5042
- evidence: `results/experiment_4167_outerloop_training_monitor.json`; honest_verdict=complete: outerloop_training_alive_val_0.5042_below_0.85; current_val=0.5042; max_val=0.5042; baseline_faithful=false; outerloop_train_alive=true; checkpoint_mtime=2026-06-13T04:41:29.093138Z; val_trajectory_386_rounded=[0.2005, 0.1992, 0.1995, 0.2216, 0.2042, 0.2211, 0.2443, 0.257, 0.2823, 0.3008, 0.3198, 0.3385, 0.3555, 0.3721, 0.3862, 0.4, 0.412, 0.4263, 0.4354, 0.4427, 0.4521, 0.4648, 0.4776, 0.4857, 0.4906, 0.4938, 0.501, 0.2005, 0.2193, 0.2388, 0.2586, 0.2776, 0.2992, 0.199, 0.2182, 0.2424, 0.2635, 0.2794, 0.3036, 0.324, 0.3383, 0.3555, 0.2021, 0.224, 0.1992, 0.2219, 0.2432, 0.2604, 0.2784, 0.3073, 0.3214, 0.3341, 0.3521, 0.3727, 0.3852, 0.4005, 0.4109, 0.4253, 0.4352, 0.4458, 0.4523, 0.4604, 0.4685, 0.4753, 0.4815, 0.493, 0.4966, 0.5042]; training_launched=false; train_process_stop_attempted=false; stable_checkpoint_written=false.
- failure mode: the outer-loop baseline is still below the faithful 0.85 gate, so verifier-graft claims remain deferred even though validation progress continued into the .386 window.
- missing discriminator: faithful outer-loop Sudoku baseline candidate source before DiffusionGemma scale-up or verifier-as-reward claims.
- candidate design: keep the outer-loop run owner authoritative, continue monitoring read-only status, and rerun graft only after the checkpoint is faithful and stable.
- priority: high
<!-- exp4171-sudoku-baseline-reproduction:end -->

<!-- exp4171-sudoku-decisive-graft:start -->
### GAP-SUDOKU-EXECUTABLE-VERIFIER-4168: Exp 4171 .386 defensive executable-verifier graft status
- status: open_graft_deferred_outerloop_training_val_0.5148
- evidence: `results/experiment_4168_decisive_verifier_graft_defensive.json` with monitor `results/experiment_4167_outerloop_training_monitor.json`; outerloop_monitor_current_val=0.5042; baseline_current_val=0.5148; baseline_faithful=false; faithful_stable=false; graft_deferred=true; checkpoint_copy_performed=false; candidate_source=none_deferred_outerloop_training; n_candidate_pools=0; rerank_lift_vs_vote_status=deferred_outerloop_training; rft_vs_ablation_delta=0.0; rft_vs_ablation_delta_status=deferred_outerloop_training; verifier_value_added=false; diffusiongemma_gate_state=kept_gated.
- failure mode: Exp 4168 deferred before checkpoint copy, candidate sampling, or training because the baseline was not faithful and stable; no verifier-as-reward value-added claim is available.
- missing discriminator: transferable training-time value from non-oracle Sudoku verifier labels beyond vote labels.
- candidate design: keep DiffusionGemma gated until rerank or RFT A-vs-B label contrast shows value on a faithful stable baseline copy.
- priority: high
<!-- exp4171-sudoku-decisive-graft:end -->

<!-- exp4181-headroom-controlled-moat:start -->
### GAP-MOAT-HEADROOM-CONTROLLED-4177: Exp 4181 .387 headroom-controlled moat verdict
- status: filled_headroom_controlled_verifier_value_added
- evidence: `results/experiment_4177_decisive_headroom_controlled_moat_test.json` with headroom census `results/experiment_4175_headroom_gate_executable_census.json`; headroom_present_domain=code; verifier_value_added=true; positive_control_confirmed=true; moat_delta_vs_vote_delta=0.18; moat_delta_vs_vote_ci95=[0.08, 0.3]; moat_vs_matched_control_delta=0.18; max_selectable_headroom=0.18; inference_substrate=deterministic_verifier_plus_replay.
- failure mode: closed for this headroom-controlled code-domain moat test; the verifier-plus-selector arm beats self-consistency vote with a positive CI.
- missing discriminator: none for the measured .387 code-domain moat verdict; continue to require headroom-positive domains before interpreting moat nulls.
- candidate design: preserve the objective headroom gate and matched-control arm for future verifier-value tests.
- priority: medium
<!-- exp4181-headroom-controlled-moat:end -->

<!-- exp4181-gap3-stage1:start -->
### GAP-3-STAGE1-MODEL-NATIVE-LATENT-4178: Exp 4181 .387 GAP-3 Stage-1 latent-energy result
- status: open_stage1_honest_negative_does_not_advance
- evidence: `results/experiment_4178_gap3_stage1_model_native_arc_energy.json`; selected_energy=model_native_basis_pca_gold_mahalanobis; pass2_energy_vs_vote=0.0; energy_pass2=0.451613; vote_pass2=0.451613; oracle_pass2=0.612903; bootstrap_ci95=[-0.096774, 0.096774]; headroom_capture_fraction=0.0; oracle_minus_vote=0.16129; candidate_auroc=0.893651; coverage_fraction=1.0; all_four_gates_pass=false; advances_toward_filled=false.
- failure mode: the model-native latent energy ties vote at pass@2 and captures none of the oracle-minus-vote headroom, so Stage 1 is an honest negative rather than a filled GAP-3 discriminator.
- missing discriminator: a model-native ARC energy that improves pass@2 over vote and captures real headroom without oracle leakage.
- candidate design: keep GAP-3 open for a stronger generator-independent content energy; do not promote Stage 1 toward filled from this result.
- priority: high
<!-- exp4181-gap3-stage1:end -->

<!-- exp4193-efficiency-moat:start -->
### GAP-MOAT-EFFICIENCY-JUDGE-4186: Exp 4193 .388 efficiency moat versus LLM judge
- status: filled_verifier_efficiency_win
- evidence: `results/experiment_4186_efficiency_moat_verifier_vs_llm_judge.json`; verifier_efficiency_win=true; accuracy_parity_vs_judge_delta=0.18; accuracy_parity_vs_judge_ci95=[0.08, 0.3]; cost_ratio_wall_clock=1.9986e-06; wall_clock_x_cheaper=500351.5303458394; ten_x_cheaper_on_both_axes=true; strictly_pareto_dominant=true; positive_control_confirmed=true.
- failure mode: closed for the measured .388 efficiency moat only when the cheap verifier matches or beats the judge on accuracy while dominating real cost; otherwise GAP-MOAT remains open.
- missing discriminator: none for this measured code-domain efficiency moat because verifier_efficiency_win=true.
- candidate design: preserve the real-cost LLM-judge comparator and the objective headroom-positive pool for future moat checks.
- priority: medium
<!-- exp4193-efficiency-moat:end -->

<!-- exp4193-gap4-graded-gate:start -->
### GAP-4-GRADED-GATE-4187: Exp 4193 .388 GAP-4 guarded graded gate
- status: filled_guarded_graded_gate_holds_plus4_minus0
- evidence: `results/experiment_4187_gap4_graded_execution_gate_hardening.json`; graded_gate_pass2_vs_vote=0.129; gross_recovery_ledger.recovered=4; gross_recovery_ledger.lost=0; pass2_vote_wins_lost=0; vote_aware_guard_blocked_mispromotion=true; agreement_confidence_label_only=true.
- failure mode: the graded relaxation adds no ARC-1 recovery beyond the exact baseline, but the guarded policy preserves the +4/-0 pass@2 safety record and blocks the recorded high-vote-gold mispromotion.
- missing discriminator: none for the guarded ARC-1 pass@2 baseline; keep vote-aware guarding load-bearing for future graded relaxations.
- candidate design: use the guarded tau=0.005 graded execution gate while recording agreement only as a confidence label.
- priority: high
<!-- exp4193-gap4-graded-gate:end -->

<!-- exp4193-sovereign-generator:start -->
### GAP-SOVEREIGN-LOCAL-GAP4-GENERATOR-4188: Exp 4193 .388 sovereign local GAP-4 generator
- status: building_sovereign_local_generator_positive_flagged
- evidence: `results/experiment_4188_sovereign_local_generator_gap4_self_distill.json`; local_induction_rate=0.2258; local_demo_perfect=7; local_total=31; sovereign_pool_pass2.LOCAL_HARDENED_GATE=0.4839; sovereign_pool_pass2.TRM_VOTE=0.4516; sovereign_pool_pass2.delta_vs_vote=0.0323; self_distillation_corpus_size=7; no_closed_weight_call=true; flagged_adversarial=true.
- failure mode: the local generator recovers a positive guarded pass@2 lift and banks a verifier-labeled corpus, but its induction rate remains far below the Codex reference and the source artifact carries adversarial caveats.
- missing discriminator: stronger local program induction or verifier-guided generation that surfaces demo-perfect programs without closed-weight calls.
- candidate design: continue sovereign generator improvement and self-distill from verifier-labeled demo-perfect programs before any clean filled claim.
- priority: high
<!-- exp4193-sovereign-generator:end -->

<!-- exp4204-gap-reward:start -->
### GAP-REWARD: Exp 4204 .389 verifier-as-reward A-vs-B axis
- status: blocked_a_vs_b_not_collected_training_not_launched
- evidence: `results/experiment_4199_verifier_reward_decisive_a_vs_b_collect.json` with Phase-0 `results/experiment_4197_verifier_reward_phase0_headroom_harness_build.json`; phase0_precision=0.9561855670103093; youden_j=0.4137931034482759; phase0_gate_clean=true; training_launched=false; verifier_label_carries_signal=false; a_vs_b_delta=None; a_vs_b_ci95=None; honest_verdict=blocked_gate_check_failed.
- failure mode: the clean Phase-0 operating point exists, but the decisive A-vs-B collection is blocked because the 3-arm training launch did not produce a live checkpoint; no reward-signal win is claimable.
- missing discriminator: decision-grade evidence that verifier-certified labels beat same-generator random-label controls on held-out hidden tests.
- candidate design: relaunch/resume the stable 3-arm training run, require gold-control and truncation guards, and promote only if the A-vs-B CI excludes zero.
- priority: high
<!-- exp4204-gap-reward:end -->

<!-- exp4204-certified-corpus:start -->
### GAP-REWARD-CERTIFIED-CORPUS-4200: Exp 4204 certified ARC corpus distill-lift note
- status: certified_corpus_built_distill_lift_uninformative
- evidence: `results/experiment_4200_certified_arc_corpus_distill_lift.json`; certified_corpus_size=16; certification_precision.rate=0.9375; distill_lift_ci95=[0.0, 0.0]; invisible_leash_diagnosis=uninformative; seeded_generation_status=missing_seeded_checkpoint_conservative_flat; flagged_adversarial=true.
- failure mode: the GAP-4 certified corpus exists and is high precision, but the cheap seeded-vs-cold local induction read is uninformative because the seeded checkpoint is missing; no latent distillation lift is established.
- missing discriminator: measured seeded local generation or LoRA distillation lift from verifier-certified ARC programs.
- candidate design: materialize the seeded checkpoint or run the bounded LoRA-distill follow-up before claiming the certified labels train a better local generator.
- priority: high
<!-- exp4204-certified-corpus:end -->

<!-- exp4216-oracle-distinct:start -->
### GAP-ORACLE-DISTINCT: Exp 4216 .390 oracle-distinct A3 frontier
- status: open_a3_blocked_selector_not_trained
- evidence: `results/experiment_4210_oracle_distinct_arc_verifier_beats_vote.json` with build `results/experiment_4209_oracle_distinct_arc_verifier_build.json`; oracle_distinct_beats_vote=false; oracle_distinct_delta=None; oracle_distinct_ci95=None; verifier_is_oracle=false; selector_trained=false; oracle_distinct_auroc=0.0; oracle_distinct_auroc_ci95=[0.0, 0.0]; honest_verdict=blocked_gate_check_failed. GAP-MOAT unchanged.
- failure mode: A3 did not execute because the learned oracle-distinct ARC selector was not trained from labeled candidates; no vote-beating off-oracle verifier result is claimable.
- missing discriminator: a learned verifier that beats vote where execution is not the oracle.
- candidate design: materialize labeled per-candidate ARC rows, train the oracle-distinct selector out of fold, then rerun A3 only if selector_trained=true.
- priority: high
<!-- exp4216-oracle-distinct:end -->

<!-- exp4216-detector-auroc:start -->
### GAP-DETECTOR-AUROC-4208: Exp 4216 detector AUROC status note
- status: detector_auroc_recorded_all_domains_ci_exclusive
- evidence: `results/experiment_4208_verifier_as_detector_auroc.json`; sudoku=1.0; code=1.0; math=1.0; arc=0.9016; ci95_by_domain={'sudoku': [1.0, 1.0], 'code': [1.0, 1.0], 'math': [1.0, 1.0], 'arc': [0.7828, 0.9984]}; verifier_is_oracle_by_domain={'sudoku': True, 'code': True, 'math': True, 'arc': False}.
- failure mode: detector AUROC says the verifier can separate good from bad candidates, but it does not by itself prove a selector beats vote.
- missing discriminator: a selection policy that converts detector signal into a vote-beating ranker on a headroom-present pool.
- candidate design: use the detector as a training or calibration source for the oracle-distinct selector, then measure selection lift with bootstrap CI.
- priority: medium
<!-- exp4216-detector-auroc:end -->

<!-- exp4216-gap-reward:start -->
### GAP-REWARD: Exp 4216 .390 verifier-as-reward A-vs-B axis
- status: open_accumulating_reward_no_eval_yet
- evidence: `results/experiment_4211_verifier_as_reward_finish_synchronous.json`; verifier_label_carries_signal=false; a_vs_b_delta=None; a_vs_b_ci95=None; youden_j=0.4137931034482759; positive_control_confirmed=false; accumulated_n={'eval': 0, 'train_A': 0, 'train_B': 0, 'train_C': 0, 'train_D': 0}; verifier_is_oracle=true; honest_verdict=progress: accumulating_verifier_reward_training_no_eval_yet.
- failure mode: the synchronous reward run has not produced held-out A-vs-B eval rows yet, so the verifier-label reward signal remains unproven.
- missing discriminator: decision-grade evidence that verifier-certified labels beat same-generator random-label controls on held-out hidden tests.
- candidate design: continue the accumulate/resume path until eval rows exist, then promote only if the A-vs-B CI excludes zero with a valid positive control.
- priority: high
<!-- exp4216-gap-reward:end -->

<!-- exp4216-certified-corpus:start -->
### GAP-REWARD-CERTIFIED-CORPUS-4212: Exp 4216 certified ARC corpus distill-lift note
- status: certified_corpus_built_distill_lift_absent
- evidence: `results/experiment_4212_certified_arc_corpus_distill_lift.json`; certified_corpus_size=16; certification_precision.rate=0.9375; distill_lift_delta=0.0; distill_lift_ci95=[0.0, 0.0]; invisible_leash_diagnosis=absent; seeded_generation_status=missing_seeded_checkpoint_conservative_flat; verifier_is_oracle=true; flagged_adversarial=true.
- failure mode: the certified corpus is high precision, but the seeded-vs-cold read is flat and flagged, so no local distillation lift is established.
- missing discriminator: measured seeded local generation or LoRA distillation lift from verifier-certified ARC programs.
- candidate design: materialize a real seeded checkpoint or LoRA-distill follow-up before claiming certified labels improve a local generator.
- priority: high
<!-- exp4216-certified-corpus:end -->

<!-- exp4227-oracle-distinct:start -->
### GAP-ORACLE-DISTINCT: Exp 4227 .391 oracle-distinct frontier
- status: open_a2_ties_vote_with_headroom
- evidence: `results/experiment_4221_oracle_distinct_arc_verifier_beats_vote.json` with build `results/experiment_4220_oracle_distinct_arc_verifier_build_labeled.json`; oracle_distinct_beats_vote=false; verifier_minus_vote_delta=-0.0714285714; verifier_minus_vote_ci95=[-0.2142857143, 0.0]; verifier_is_oracle=false; selector_trained=true; oracle_distinct_auroc=0.778980279; wrong_majority_n=5; honest_verdict=complete: oracle_distinct_verifier_ties_vote_with_headroom. GAP-MOAT unchanged.
- failure mode: the trained non-oracle A2 verifier did not produce a CI-exclusive vote-beating read on the headroom-present ARC slice.
- missing discriminator: a learned verifier that beats vote where execution is not the oracle.
- candidate design: use the wrong-majority rows and detector signal to improve the oracle-distinct selector before re-testing A2.
- priority: high
<!-- exp4227-oracle-distinct:end -->

<!-- exp4227-oracle-distinct-a2:start -->
### GAP-ORACLE-DISTINCT-A2-4221: Exp 4227 .391 A2 beats-vote read
- status: open_a2_ties_vote_with_headroom
- evidence: `results/experiment_4221_oracle_distinct_arc_verifier_beats_vote.json`; oracle_distinct_beats_vote=false; verifier_minus_vote_delta=-0.0714285714; verifier_minus_vote_ci95=[-0.2142857143, 0.0]; arbiter_override_minus_vote=0.0; matched_control_delta=0.0; oracle_at_k=1.0; verifier_is_oracle=false; oracle_distinct_auroc=0.778980279; oracle_distinct_auroc_ci95=[0.6146676853, 0.9174508427]; wrong_majority_n=5.
- failure mode: A2 is the first trained oracle-distinct read, but the held-out rerank still ties/underperforms vote rather than capturing the wrong-majority headroom.
- missing discriminator: a non-oracle ARC selector that converts off-fold candidate discrimination into vote-beating top-1 selection.
- candidate design: reweight or enrich the learned selector, then require a positive bootstrap CI before upgrading the frontier.
- priority: high
<!-- exp4227-oracle-distinct-a2:end -->

<!-- exp4227-gap-reward:start -->
### GAP-REWARD: Exp 4227 .391 verifier-as-reward A-vs-B axis
- status: open_accumulating_reward_no_eval_yet
- evidence: `results/experiment_4223_verifier_as_reward_3arm_synchronous.json`; verifier_label_carries_signal=false; a_vs_b_delta=None; a_vs_b_ci95=None; youden_j=0.4137931034482759; positive_control_confirmed=false; accumulated_n={'eval': 0, 'train_A': 0, 'train_B': 0, 'train_C': 0, 'train_D': 0}; verifier_is_oracle=true; honest_verdict=progress: accumulating_verifier_reward_training_no_eval_yet.
- failure mode: the synchronous reward path still has no held-out A-vs-B eval rows, so verifier-label reward signal remains unproven.
- missing discriminator: decision-grade evidence that verifier-certified labels beat same-generator random-label controls on held-out hidden tests.
- candidate design: continue only until eval rows exist, then promote only if the A-vs-B CI excludes zero with a valid positive control.
- priority: high
<!-- exp4227-gap-reward:end -->

<!-- exp4227-detector-auroc:start -->
### GAP-DETECTOR-AUROC-4208: Exp 4227 detector AUROC status note
- status: detector_auroc_recorded_all_domains_ci_exclusive
- evidence: `results/experiment_4208_verifier_as_detector_auroc.json`; sudoku=1.0; code=1.0; math=1.0; arc=0.9016; ci95_by_domain={'sudoku': [1.0, 1.0], 'code': [1.0, 1.0], 'math': [1.0, 1.0], 'arc': [0.7828, 0.9984]}; verifier_is_oracle_by_domain={'sudoku': True, 'code': True, 'math': True, 'arc': False}.
- failure mode: detector AUROC separates good from bad candidates but has not yet become a selector that beats vote.
- missing discriminator: a selection policy that converts detector signal into a vote-beating ranker on a headroom-present pool.
- candidate design: use the detector as a training or calibration source for the oracle-distinct selector, then measure selection lift with bootstrap CI.
- priority: medium
<!-- exp4227-detector-auroc:end -->

<!-- exp4239-oracle-distinct:start -->
### GAP-ORACLE-DISTINCT: Exp 4239 .392 oracle-distinct frontier
- status: open_a2_ties_vote_with_headroom_at_power
- evidence: `results/experiment_4232_oracle_distinct_arc_aggregator_beats_vote.json` with build `results/experiment_4231_oracle_distinct_arc_aggregator_build.json`; oracle_distinct_beats_vote=false; aggregator_minus_vote_delta=0.0; aggregator_minus_vote_ci95=[0.0, 0.0]; held_out_task_n=52; matched_control_delta=0.0384615385; oracle_at_k=0.3653846154; verifier_is_oracle=false; oracle_distinct_auroc=0.7865558646; oracle_distinct_auroc_ci95=[0.6319719028, 0.9258842843]; wrong_majority_n=9; build_flagged_adversarial=true; honest_verdict=complete: oracle_distinct_aggregator_ties_vote_with_headroom_at_power. GAP-MOAT unchanged: .392 stronger build + power did not change the .391 ties-vote read.
- failure mode: the strengthened non-oracle ARC aggregator still tied vote despite headroom and a larger held-out task count.
- missing discriminator: a learned non-oracle ARC selector whose vote-beating delta has a positive CI on the headroom-present slice.
- candidate design: grow the ARC pool or feature set using the code-domain disambiguation result before re-testing A2.
- priority: high
<!-- exp4239-oracle-distinct:end -->

<!-- exp4239-oracle-distinct-a2:start -->
### GAP-ORACLE-DISTINCT-A2-4232: Exp 4239 .392 strengthened A2 read
- status: open_a2_ties_vote_with_headroom_at_power
- evidence: `results/experiment_4232_oracle_distinct_arc_aggregator_beats_vote.json`; oracle_distinct_beats_vote=false; aggregator_minus_vote_delta=0.0; aggregator_minus_vote_ci95=[0.0, 0.0]; held_out_task_n=52; margin_override_minus_vote=0.0; matched_control_delta=0.0384615385; oracle_at_k=0.3653846154; verifier_is_oracle=false; oracle_distinct_auroc=0.7865558646; oracle_distinct_auroc_ci95=[0.6319719028, 0.9258842843]; wrong_majority_n=9.
- failure mode: the power increase made the null cleaner rather than closing the oracle-distinct frontier.
- missing discriminator: an ARC candidate scorer that converts off-fold discrimination into vote-beating top-1 selection.
- candidate design: scale ARC positives or use richer candidate-set features, then require CI-exclusive lift before upgrading the frontier.
- priority: high
<!-- exp4239-oracle-distinct-a2:end -->

<!-- exp4239-code-disambiguation:start -->
### GAP-CODE-DISAMBIGUATION-4233: Exp 4239 .392 code disambiguation note
- status: filled_code_oracle_distinct_beats_vote
- evidence: `results/experiment_4233_oracle_distinct_code_beats_vote.json`; disambiguation_read=ARC_null_is_data_sparsity; code_oracle_distinct_beats_vote=true; code_predictor_minus_vote_delta=0.03125; code_predictor_minus_vote_ci95=[0.00625, 0.0625]; held_out_task_n=160; verifier_is_oracle=false; off_fold_auroc=0.9739318159; oracle_at_k=0.9625.
- failure mode: ARC's null does not generalize to high-power code; the ARC frontier is more likely data/positive-sparsity bound.
- missing discriminator: ARC-scale positives or features with the power that made code vote-beating.
- candidate design: build a larger ARC oracle-distinct candidate pool before retiring the selection thesis.
- priority: high
<!-- exp4239-code-disambiguation:end -->

<!-- exp4239-gap-reward:start -->
### GAP-REWARD: Exp 4239 .392 verifier-as-reward A-vs-B axis
- status: open_live_lora_blocked_pre_gate
- evidence: `results/experiment_4235_verifier_as_reward_3arm_window_boxed.json`; verifier_label_carries_signal=false; a_vs_b_delta=None; a_vs_b_ci95=None; youden_j=None; live_lora_retired=false; blocked_at_layer=conductor_pre_gate; honest_verdict=blocked_gate_check_failed.
- failure mode: Exp 4235 did not reach a held-out A-vs-B measurement because the real-training smoke pre-gate failed.
- missing discriminator: decision-grade evidence that verifier-certified labels beat same-generator random-label controls, or an explicit live-LoRA retirement artifact.
- candidate design: re-scope to an offline reward-weighted form or land a valid non-blocked A-vs-B eval before promotion.
- priority: high
<!-- exp4239-gap-reward:end -->

<!-- exp4252-oracle-distinct:start -->
### GAP-ORACLE-DISTINCT: Exp 4252 .393 oracle-distinct frontier
- status: filled_arc_a3_set_encoder_beats_vote_non_oracle
- evidence: `results/experiment_4245_arc_set_encoder_beats_vote.json` with build `results/experiment_4244_arc_set_encoder_aggregator_build.json`; oracle_distinct_beats_vote=true; set_encoder_minus_vote_delta=0.4423076923; set_encoder_minus_vote_ci95=[0.3076923077, 0.5961538462]; held_out_task_n=52; matched_control_delta=0.4807692308; oracle_at_k=0.8269230769; oracle_minus_vote=0.5769230769; verifier_is_oracle=false; oracle_distinct_auroc=0.9633173387; oracle_distinct_auroc_ci95=[0.9185239956, 0.9911212918]; set_encoder_vs_logistic_auroc_delta=-0.0161846276; wrong_majority_n=30; honest_verdict=complete: arc_oracle_distinct_set_encoder_beats_vote. This changed the .392 ties-vote read on the grown-pool set-encoder path. GAP-MOAT unchanged: registry hygiene records the frontier result but does not silently upgrade a moat claim.
- failure mode: closed for the measured ARC A3 oracle-distinct selection read; other reward and replication axes remain separate.
- missing discriminator: none for the measured non-oracle ARC A3 vote-beating read.
- candidate design: preserve the grown-pool set-encoder methodology and retest only with explicit non-oracle and positive-CI gates.
- priority: high
<!-- exp4252-oracle-distinct:end -->

<!-- exp4252-oracle-distinct-a3:start -->
### GAP-ORACLE-DISTINCT-A3-4245: Exp 4252 .393 ARC A3 set-encoder read
- status: filled_arc_a3_set_encoder_beats_vote_non_oracle
- evidence: `results/experiment_4245_arc_set_encoder_beats_vote.json`; oracle_distinct_beats_vote=true; set_encoder_minus_vote_delta=0.4423076923; set_encoder_minus_vote_ci95=[0.3076923077, 0.5961538462]; held_out_task_n=52; margin_override_minus_vote=0.4230769231; matched_control_delta=0.4807692308; oracle_at_k=0.8269230769; verifier_is_oracle=false; oracle_distinct_auroc=0.9633173387; oracle_distinct_auroc_ci95=[0.9185239956, 0.9911212918]; set_encoder_vs_logistic_auroc_delta=-0.0161846276; wrong_majority_n=30.
- failure mode: no ARC A3 failure for this measured read; the set-encoder converted the grown-pool candidate signal into a CI-positive vote-beating selector.
- missing discriminator: none for this measured read; keep the non-oracle and held-out task gates load-bearing.
- candidate design: compare future variants against both vote and same-pool controls.
- priority: high
<!-- exp4252-oracle-distinct-a3:end -->

<!-- exp4252-code-replication:start -->
### GAP-CODE-REPLICATION-4246: Exp 4252 .393 code replication status
- status: blocked_code_second_corpus_missing
- evidence: `results/experiment_4246_code_oracle_distinct_replication.json`; replication_read=blocked_code_second_corpus_missing; code_replication_beats_vote=false; code_predictor_minus_vote_delta=0.0; code_predictor_minus_vote_ci95=[0.0, 0.0]; held_out_task_n=0; verifier_is_oracle=false; off_fold_auroc=0.0; oracle_at_k=0.0.
- failure mode: no second distinct code corpus was available, so the code oracle-distinct win was neither replicated nor refuted.
- missing discriminator: a source-distinct code candidate corpus for replication.
- candidate design: rerun only after the source-distinctness gate has a nonempty corpus.
- priority: medium
<!-- exp4252-code-replication:end -->

<!-- exp4252-gap-reward:start -->
### GAP-REWARD: Exp 4252 .393 offline verifier-as-reward A-vs-B axis
- status: blocked_offline_reward_gate_failed_live_lora_retired
- evidence: `results/experiment_4248_verifier_as_reward_offline_3arm.json` with retirement artifact `results/experiment_4247_verifier_reward_offline_harness_retire_livelora.json`; verifier_label_carries_signal=false; a_vs_b_delta=None; a_vs_b_ci95=None; youden_j=None; live_lora_retired=true; blocked_at_layer=conductor_pre_gate; honest_verdict=blocked_gate_check_failed.
- failure mode: Exp 4248 was blocked by the Exp 4247 harness smoke gate, so no held-out A-vs-B reward signal exists.
- missing discriminator: decision-grade offline evidence that verifier-certified labels beat same-generator random-label controls.
- candidate design: repair or replace the offline smoke harness before any reward-signal promotion; do not reopen the live-LoRA path without operator approval.
- priority: high
<!-- exp4252-gap-reward:end -->

<!-- exp4252-live-lora-retirement:start -->
### GAP-REWARD-LIVE-LORA-RETIREMENT-4247: Exp 4252 live-LoRA retirement note
- status: retired_live_lora_path_after_6_infra_failures
- evidence: `results/experiment_4247_verifier_reward_offline_harness_retire_livelora.json`; live_lora_retired=true; harness_smoke_passed=false; infra_failure_count=6; operator_reopen_required=true; retire_if_same_verdict=true; honest_verdict=blocked_offline_reward_weighted_training_cannot_run_in_window.
- failure mode: the live-LoRA verifier-as-reward path accumulated 6 infra failures and is retired so it is not re-proposed as another live run.
- missing discriminator: none for live-LoRA retirement; future reward work must use the offline path unless an operator explicitly reopens live-LoRA.
- candidate design: keep the exclusion manifest entry authoritative with operator_reopen_required=true and retire_if_same_verdict=true.
- priority: high
<!-- exp4252-live-lora-retirement:end -->

<!-- exp4266-gap-arc-cross-game-selection-4266:start -->
### GAP-ARC-CROSS-GAME-SELECTION-4266: Exp 4266 .394 missing-verifier gap
- status: open
- evidence: results/experiment_4258_arc_oracle_distinct_cross_game_transfer.json; honest_verdict=blocked_arc_game_ids_unrecoverable; held_out_game_n=0; held_out_task_n=0.
- failure mode: The ARC oracle-distinct selector is hardened within-pool, but cross-game transfer could not be measured because game/family ids were unrecoverable.
- missing discriminator: game/family provenance for every ARC candidate row so selection can be evaluated on family-disjoint held-out games.
- candidate design: Persist source-kind, generator family, game id, fold id, and target hash in the ARC candidate manifest, then rerun the set-encoder against vote on held-out families.
- priority: high
<!-- exp4266-gap-arc-cross-game-selection-4266:end -->

<!-- exp4266-gap-arc-supra-oracle-k-synthesis-4266:start -->
### GAP-ARC-SUPRA-ORACLE-K-SYNTHESIS-4266: Exp 4266 .394 missing-verifier gap
- status: open
- evidence: results/experiment_4259_arc_agglm_grid_synthesis.json; synthesis_beats_selection=False; synthesis_breaks_oracle_ceiling=False; synthesis_minus_oracle_delta=-0.2826086957.
- failure mode: Score-weighted grid synthesis underperformed selection and did not solve tasks beyond oracle@K candidate availability.
- missing discriminator: a supra-oracle@K verifier signal that can infer missing output cells or shapes when no cached candidate is correct.
- candidate design: Add rule-consistency or latent task-family constraints to propose cells outside the selected candidate family, with exact-match and selector-only controls.
- priority: high
<!-- exp4266-gap-arc-supra-oracle-k-synthesis-4266:end -->

<!-- exp4266-gap-diffusiongemma-loader-guidance-4266:start -->
### GAP-DIFFUSIONGEMMA-LOADER-GUIDANCE-4266: Exp 4266 .394 missing-verifier gap
- status: open
- evidence: results/experiment_4260_diffusiongemma_energy_guided_preflight.json; honest_verdict=blocked_diffusiongemma_gguf_loader_failed; preflight_go=False; guidance_changes_selection=False.
- failure mode: DiffusionGemma energy guidance remained blocked before the guidance hook could demonstrate verifier-shaped token selection.
- missing discriminator: a loader-validated diffusion guidance path that proves verifier energy changes denoising selections before a full run is scheduled.
- candidate design: Repair the GGUF vocab loader path, then run a tiny deterministic guidance smoke that records changed selections and exact verifier controls.
- priority: medium
<!-- exp4266-gap-diffusiongemma-loader-guidance-4266:end -->

<!-- exp4266-gap-code-oracle-distinct-robustness-4266:start -->
### GAP-CODE-ORACLE-DISTINCT-ROBUSTNESS-4266: Exp 4266 .394 missing-verifier gap
- status: open
- evidence: results/experiment_4264_code_oracle_distinct_replication_retry.json; replication_read=corpus_specific; code_replication_beats_vote=False; code_predictor_minus_vote_delta=-0.00625.
- failure mode: The code oracle-distinct read is corpus-specific; the second corpus did not replicate a vote-beating predictor despite oracle headroom.
- missing discriminator: source-robust code candidate features that distinguish real hidden-test pass signal from corpus-specific lexical or vote-signature artifacts.
- candidate design: Evaluate source-disjoint code pools with normalized-code, AST, agreement, and self-consistency features, plus per-source ablations before any robustness claim.
- priority: medium
<!-- exp4266-gap-code-oracle-distinct-robustness-4266:end -->

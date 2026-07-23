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

## 2026-07-02 retirement note: Phase D external-text scorer moat closed, hidden-state verifiers remain open

`results/experiment_5170_retire_phase_d_external_text_scorer_v474.json` retires only the
external generated-text/logprob scorer construction class from Phase D: LoRA-EBM holistic
scorers, uPRM-style text/logprob process rewards, EBRM-style post-hoc reward refinement, and
closely equivalent distributional-energy rankers when they are evaluated against genuine tuned
self-consistency on off-ARC reasoning corpora. The consolidation covers the Exp 4940 plus
Exp 5001-5163 Phase D trail, including the live MuSR negative and the Exp 5163 MMLU-Pro
continuation whose CI still includes zero and is adversarially tautology-flagged.

This retirement does not close hidden-state/internal-representation verifier research. In
particular, TrajSelector-style hidden-state scoring and VerifySteer-style hidden-state steering
remain open, sanctioned mechanisms for `exp5178`, because they score the generator's own
internal representations rather than reranking generated text or logprobs. The retirement also
does not apply to future ARC-domain oracle-distinct verifier work or to Carnot's FoVer production
ensemble.

## Exp 5007 off-ARC EBRM selection residual (2026-06-30)

`results/experiment_5007_moat_gate_resolution.json` aggregated the D1-D4 off-ARC verifier-moat
artifacts after skipping `flagged_adversarial=true` inputs. The only clean aggregated arm was
Exp 5005 EBRM on MuSR: `verifier_is_oracle=false`, `headroom_present=true`, oracle@K `0.93`,
TUNED-SC `0.585`, EBRM selection accuracy `0.585`, delta `0.0`, CI95 `[-0.03, 0.025]`,
McNemar `p=1.0`. The correct-answer headroom is present but the current uncertainty-aware
energy selector cannot choose beyond TUNED-SC.

### GAP-MUSR-EBRM-HEADROOM-SELECTION: reasoning-quality energy does not capture selectable MuSR headroom
- status: open
- evidence: `results/experiment_5007_moat_gate_resolution.json`; clean D3/Exp 5005 null on
  headroom-present MuSR, with D1/D2/D4 skipped as flagged and no cross-corpus confirmation.
- failure mode: the EBRM reward-distribution selector distinguishes neither the TUNED-SC pick
  nor the oracle-recoverable candidate strongly enough to move accuracy above the matched
  self-consistency baseline.
- missing discriminator: an oracle-distinct process/semantic signal that separates genuinely
  entailed multi-hop MuSR reasoning from fluent but wrong candidate rationales under the same
  answer-choice pool.
- candidate design: mine MuSR oracle-recoverable rows where TUNED-SC misses, label candidate
  rationales with contrastive entailment/decomposition features, and train a process-aware
  selector whose evaluation remains paired against TUNED-SC with CI95 and McNemar gates.
- priority: medium

## Exp 4381 FoVer BiPRM localization residual (2026-06-18)

`results/experiment_4381_biprm_detector_localization_abstention.json` scored the cached
step-labeled FoVer corpus with causal L2R and offline suffix-aware R2L verifier passes.
The detector remained oracle-distinct and post-hoc bidirectional localization did not improve
over causal L2R: both localized 11/114 first-error traces (accuracy/F1 0.096491), with
bootstrap delta CI95 [0.0, 0.0]. The artifact records 103 missed first-error traces in the
untyped residual class.

### GAP-FOVER-BIPRM-LOCALIZATION-untyped: earliest causal error vs downstream consequence
- status: open
- evidence: `results/experiment_4381_biprm_detector_localization_abstention.json`;
  clean powered null for bidirectional fusion, 103 missed first-error traces out of 114
  error traces, no bidirectional-minus-L2R localization gain.
- failure mode: the current FoVer verifier ensemble can detect trace-level error risk, but
  it often assigns the highest step-error score to a later downstream consequence rather than
  the earliest causal error.
- missing discriminator: a per-step feature that distinguishes the first causal invariant break
  from later steps that merely inherit the earlier mistake.
- candidate design: add typed step-error labels and a contrastive earliest-error objective over
  cached FoVer traces, then require separate causal L2R and offline R2L reporting so future-context
  gains are not claimed as online-actionable.
- priority: medium

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

<!-- experiment_5205_autopyverifier_gap1_pilot_v476 start -->
- experiment_5205 AutoPyVerifier-inspired deterministic set search (2026-07-03): best_subset=['border_ordered_profile', 'color_centroid_orientation', 'row_column_run_profile'], pass@2 baseline=0.087866, pass@2 best=0.221757, transpose captures=47 out of 239. Candidate singletons: directional_adjacency_refuted_20260609=helped, row_ordered_edge_profile=helped, column_ordered_edge_profile=helped, diagonal_adjacency_asymmetry=helped, corner_anchored_quadrant_histogram=helped, border_ordered_profile=helped, color_centroid_orientation=helped, row_column_run_profile=helped. Verdict: complete: set_search_beats_always_on_beats_single_refuted_baseline_0.0879_best_0.2218_single_refuted_0.1506_captured_47_of_239_gap1_candidate_positive
<!-- experiment_5205_autopyverifier_gap1_pilot_v476 end -->
<!-- experiment_5209_gap1_set_search_holdout_hardening_v477 start -->
- experiment_5209 GAP-1 set-search holdout hardening (2026-07-04): gap1_hardened_positive=True, heldout pass@2 mean=0.189584, always-on baseline=0.088976, single refuted directional=0.147787, paired delta CI95=[0.023148, 0.060446], leakage_audit_passed=True, best_subset_stable=False. Do not promote to registry here. Verdict: complete: set_search_remains_positive_after_hardening_heldout_0.1896_always_0.0890_single_refuted_0.1478_paired_delta_ci95_0.0231_0.0604_best_subset_not_stable_do_not_promote_to_registry_here
<!-- experiment_5209_gap1_set_search_holdout_hardening_v477 end -->
<!-- experiment_5222_gap1_gate_field_registry_promotion_v478 start -->
- experiment_5222 GAP-1 registry promotion decision (2026-07-04): decision=blocked_instability, gap1_registry_promoted=False, exp5209_gate_parsed_from_value=True, frozen_subset=None. Follow-up criterion: Reconsider registry promotion only after a predeclared frozen subset is selected from training evidence alone, one exact subset wins at least half of grouped splits, leakage guards pass, and directional_adjacency_refuted_20260609 remains excluded from the promoted frozen verifier. Verdict: complete: GAP-1 registry promotion blocked_instability; exp5209 gate parsed from gap1_hardened_positive.value=True, but the selected subset is not stable enough to freeze without held-out tuning; this is not the exp5210 gate-shape failure alone.
<!-- experiment_5222_gap1_gate_field_registry_promotion_v478 end -->
<!-- experiment_5237_gap1_stability_freeze_or_retire_v479 start -->
- experiment_5237 GAP-1 stability freeze-or-retire decision (2026-07-04): decision=blocked_instability, gap1_registry_promoted=False, frozen_subset=None, stability_rule_predeclared=True, no_new_broad_search=True. Block condition: Freeze is blocked unless the exact-subset and invariant-inclusion stability checks both pass under the no-held-out-tuning leakage guard. Minimum evidence to reopen: Minimum evidence to reopen: choose one frozen subset from training evidence alone, before held-out scoring, and have the exact subset win at least 10 of 20 grouped splits; every frozen invariant must appear in at least half of those splits, all leakage/no-held-out-tuning guards must pass, and directional_adjacency_refuted_20260609 must remain excluded. Verdict: complete: GAP-1 blocked_instability; the existing Exp 5209 positive result is non-leaky, but the exact selected subset is not stable enough to freeze.
<!-- experiment_5237_gap1_stability_freeze_or_retire_v479 end -->

- Exp 5214 verifier-memory pointer: `results/verifier_memory_v477.json` stores the GAP-1 orientation-discriminator set as a memory-only promotion (not a registry fill) because deterministic guardrails passed and held-out delta cleared the memory threshold.

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

### GAP-4: Exp 5161 .473 forward-protocol pilot (n=60, bounded scale)

**2026-07-02, outer-loop-corrected.** `results/experiment_5161_gap4_protocol_execution_pilot_v473.json`
ran a bounded n=60 pilot of GAP-4's own "Forward protocol" (see above): rescored the existing cached
candidate pool against a fresh held-out split, cluster-bootstrapped, ran the exact sign test. Result:
`replicated_prior_direction=true`, `exact_test_discordant_wins=4`, `exact_test_discordant_losses=0`,
`exact_test_p_value_two_sided=0.125` (independently reverified via `scipy.stats.binomtest(4, 4, 0.5)`
-- matches exactly), `exact_test_passes_min6_rule=false` (short of the protocol's own documented
≥6-discordant-win floor for two-sided p<0.05 at zero-loss design). **Honest verdict: direction
replicates at this bounded scale, not yet statistically significant, scale-up recommended.**

Was initially FLAGGED CRITICAL (DURATION_TOO_SHORT) and quarantined -- a substrate-mislabeling bug,
not fabrication (the artifact's own `inference_substrate` field was structured as a `{principle,
value}` dict, which `adversarial_verify.py`'s comparison logic cannot parse, so the check fell
through to the generic compute-bound-marker fallback regardless of the declared value). Corrected and
un-quarantined the same day (`linter_flag_corrigendum` in the artifact; live re-check clean, 0
CRITICAL flags).

**Still genuinely open:** the local-generator-arm (decentralization tier) was only cache-checked (one
smoke call, an identity-function response), not run at real pilot scale -- GAP-4's decentralization
requirement (CLAUDE.md rule 1) remains untested. The 400-task sandboxed re-confirmation on genuinely
held-out ARC-AGI-2/ConceptARC tasks (the protocol's full bar for `status: filled`) also has not run.
`status` stays **open — FIRST POSITIVE LANDED, now with a bounded-scale (n=60) directional replication
on top, still short of the significance floor and the decentralization tier.**

### GAP-4: Exp 5212 .477 expanded-pool scale validation (blocked by protocol labels)

**2026-07-04.** `results/experiment_5212_gap4_scale_validation_gated_v477.json`
loaded the usable Exp 5211 expanded pool (`candidate_pool_n=120`,
`gap4_expansion_usable=true`) and kept the Exp 5161/5177/5197 protocol fixed.
All 120 rows passed Exp 5211's feasible/demo-perfect pool gate but lacked the
registered `vote_top2` / `gated_top2` pass@2 labels required to compare vote vs
rule-execution gated pass@2 without changing the method mid-test. Result:
`n_scored=0`, `excluded_rows=120`, exclusion summary
`missing_protocol_pass2_fields=120`, `exact_test_discordant_wins=0`,
`exact_test_discordant_losses=0`, `exact_test_p_value_two_sided=1.0`,
`exact_test_passes_min6_rule=false`, CI95 `[0.0, 0.0]`,
`gap4_status_recommendation=blocked`.

GAP-4 remains **open**. The failure mode is not a negative significance result
on 120 protocol-scored rows; it is a blocked validation because the expanded
local-generation pool is not yet a vote-vs-gated pass@2 evaluation pool. A
future fill still requires the unchanged floor: at least six discordant wins,
zero discordant losses, and two-sided exact `p < 0.05` on rows that already
carry the established protocol labels.

Exp 5214 verifier-memory pointer: `results/verifier_memory_v477.json` records
the Exp 5211 GAP-4 guarded candidate-pool expansion as a rollback entry because
deterministic feasibility/leakage guards are not enough without a held-out
selection delta.

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
### GAP-ARC-CROSS-GAME-SELECTION-4266: Exp 4277 .395 filled provenance gap
- status: filled (arc_family_provenance_recovery_4270_cross_family_4271)
- evidence: results/experiment_4270_arc_family_provenance_recovery.json; family_split_feasible=True; distinct_family_n=52. results/experiment_4271_arc_cross_family_transfer_existing_pool.json; cross_family_win_holds=True; cross_family_delta=0.4038461538; cross_family_ci95=[0.25, 0.5576923077]; held_out_task_n=52.
- failure mode: filled for the original missing game/family provenance blocker; the recovered family manifest made a held-out-family test possible and the win held.
- missing discriminator: none for provenance recovery; future work should preserve family_id, fold, source_kind, and target hash on every candidate row.
- candidate design: keep the manifest join as a required input to any future cross-family selector evaluation.
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
### GAP-DIFFUSIONGEMMA-LOADER-GUIDANCE-4266: Exp 4277 .395 filled loader-guidance gap
- status: filled (diffusiongemma_loader_fix_preflight_4274)
- evidence: results/experiment_4274_diffusiongemma_loader_fix_preflight.json; loader_repaired=True; preflight_go=True; guidance_changes_selection=True; guidance_selection_change_count=12.
- failure mode: filled for the loader/preflight blocker; the .396 full run is now gated on hardened_win and this preflight_go result rather than loader reachability.
- missing discriminator: none for the loader-guidance preflight; full-run quality remains a separate .396 measurement.
- candidate design: use the repaired GGUF metadata loader and tiny guidance smoke as the preflight before any full benchmark.
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

<!-- exp4277-gap-arc-online-adaptation-calibration-4277:start -->
### GAP-ARC-ONLINE-ADAPTATION-CALIBRATION-4277: Exp 4277 .395 missing-verifier gap
- status: open
- evidence: results/experiment_4273_arc_cross_family_online_adaptation.json; online_adaptation_helps=False; online_minus_static_delta=0.0961538462; online_minus_static_ci95=[0.0, 0.1923076923].
- failure mode: Tier-1 online adaptation improved the point estimate but its CI touched zero, so static cross-family selection remains the decision-grade ceiling.
- missing discriminator: uncertainty-aware family-transfer calibration that tells when online feature and subverifier precision counters should override the static selector.
- candidate design: Use a hierarchical family calibrator with frozen static-selector controls, per-family uncertainty intervals, and a pre-registered online-minus-static CI gate.
- priority: medium
<!-- exp4277-gap-arc-online-adaptation-calibration-4277:end -->

<!-- exp4287-gap-diffusiongemma-partial-state-scorer-4287:start -->
### GAP-DIFFUSIONGEMMA-PARTIAL-STATE-SCORER-4287: Exp 4287 .396 missing-verifier gap
- status: open
- evidence: results/experiment_4281_diffusiongemma_energy_guided_full_run.json; honest_verdict=complete_diffusiongemma_learned_verifier_cannot_score_partial_states; diffusiongemma_guidance_moat=False; learned_partial_state_can_score=False; guidance_changes_selection=True.
- failure mode: DiffusionGemma guidance can reweight token choices in smoke tests, but the headline learned-verifier arm cannot score masked/partial diffusion token states, so the moat cannot be measured without falling back to circular execution-grounded verification.
- missing discriminator: learned partial-state diffusion scorer that assigns non-oracle energy to incomplete or masked token canvases before a full candidate exists.
- candidate design: Add a score_partial_state or score_masked_canvas verifier interface, train it on masked diffusion-token canvases with final-answer labels held out by task family, and require a non-circular guidance-vs-unguided CI gate before any moat claim.
- priority: high
<!-- exp4287-gap-diffusiongemma-partial-state-scorer-4287:end -->

<!-- exp4310-gap-cross-domain-family-invariant-selection-4305:start -->
### GAP-CROSS-DOMAIN-FAMILY-INVARIANT-SELECTION-4305: Exp 4310 .398 missing-verifier gap
- status: open
- evidence: results/experiment_4305_cross_domain_selector_generalization.json; upstream_missing_verifier_gap=true; failure_mode=cross_domain_selection_collapses_domain_bound.
- failure mode: cross_domain_selection_collapses_domain_bound
- missing discriminator: domain-invariant selector features that preserve wrong-majority recovery across ARC, ARC-GEN, and FoVer/math step candidates without using domain labels
- candidate design: DG-PRM-style multi-invariant verifier dimensions with a learned task-structure router validated on held-out fover
- priority: high
<!-- exp4310-gap-cross-domain-family-invariant-selection-4305:end -->

### GAP-4318: Game-invariant ARC value representation
- status: open
- evidence: `results/experiment_4318_arc_cross_game_learned_verifier_transfer.json` reports cross_game_state_reduction=1 with baseline_solves_held_out=true.
- failure mode: a value-head trained on other solved games did not produce a decision-grade held-out search-state reduction.
- missing discriminator: game-invariant ARC value representation that recognizes progress across navigation, click-placement, and rotation mechanics.
- candidate design: learned frame encoder or adapter-conditioned value head trained on more reproduced solve traces, with hardware-portable CPU features first and an accelerator path later.
- priority: medium

<!-- exp4321-gap-cross-domain-family-invariant-selection-4305:start -->
### GAP-CROSS-DOMAIN-FAMILY-INVARIANT-SELECTION-4305: Exp 4321 .399 verifier gap update
- status: open
- evidence: results/experiment_4314_cross_domain_selector_ir3de_cascal.json; upstream_missing_verifier_gap=true; failure_mode=powered_collapse_cross_domain_domain_bound.
- failure mode: powered_collapse_cross_domain_domain_bound
- missing discriminator: domain-invariant selector features that preserve wrong-majority recovery across ARC, ARC-GEN, and FoVer/math step candidates without using domain labels
- candidate design: stronger family-invariant verifier dimensions beyond IR3DE+CASCAL+ContextPRM, validated on held-out fover
- priority: high
<!-- exp4321-gap-cross-domain-family-invariant-selection-4305:end -->

<!-- exp4321-gap-4318:start -->
### GAP-4318: Exp 4321 .399 verifier gap update
- status: open
- evidence: results/experiment_4318_arc_cross_game_learned_verifier_transfer.json; upstream_missing_verifier_gap=true; failure_mode=transferred linear value-head did not reduce held-out OfflineSolver states.
- failure mode: transferred linear value-head did not reduce held-out OfflineSolver states
- missing discriminator: game-invariant ARC value representation
- candidate design: learned frame encoder or per-game adapter-conditioned value head
- priority: medium
<!-- exp4321-gap-4318:end -->

<!-- exp4321-gap-code-exec-demofit:start -->
### GAP-CODE-EXEC-DEMOFIT: Exp 4321 .399 verifier gap update
- status: filled (gap4_code_demo_fit_execution_transfer_4319)
- evidence: results/experiment_4319_off_arc_execution_verifier_transfer_accumulate.json; off_arc_demofit_beats_vote=True; off_arc_demofit_minus_vote_delta=0.02; off_arc_delta_ci95=[0.005, 0.04]; accumulated_n=200.
- failure mode: candidates can pass visible demo tests while failing hidden semantic tests
- missing discriminator: code_demo_fit_visible_tests_do_not_discriminate_hidden_semantics
- candidate design: Use the accumulated GAP-4 visible-test demo-fit execution selector as the filled cheap execution layer; reopen only if a future powered replay loses the positive hidden-test CI.
- priority: high
<!-- exp4321-gap-code-exec-demofit:end -->

### 2026-06-17 Exp4327 ar25 E3 residual gap
- Spec: REQ-PHASE4-074 / SCENARIO-PHASE4-074
- Best verifier accuracy: 0.8875
- Residual mismatch class: `missing_world_model_rule_gap_hidden_undo_stack_action7`
- Reproducibility checksum: `1e926634c023e7d82b793a890ea825f08d82d8ec6e5ebf89b6f2f8fde67af7ad`
- Gap: bounded executable-world-model run did not satisfy the offline reproduced L1 gate.

### 2026-06-17 Exp4328 ka59 E3 residual gap
- Spec: REQ-PHASE4-075 / SCENARIO-PHASE4-075
- Best verifier accuracy: 0.5625
- Residual mismatch class: `missing_world_model_rule_gap_actions_1_2_4_6`
- Reproducibility checksum: `c1cc68b82115b2006f1360fe20fa207a467ad74ccb037ce046196c763a255532`
- Gap: bounded executable-world-model run did not satisfy the offline reproduced L1 gate.

### 2026-06-17 Exp4329 tr87 E3 residual gap
- Spec: REQ-PHASE4-076 / SCENARIO-PHASE4-076
- Game: `tr87`
- Best verifier accuracy: 0.0000
- Residual mismatch class: `missing_world_model_rule_gap_actions_1_2_4`
- Reproducibility checksum: `9d7969995becae6b7704dd94148870d6eee2f7ff7f3c6ffaf036eefcf514ee00`
- Gap: bounded executable-world-model run did not satisfy the offline reproduced L1 gate.

### 2026-06-17 Exp4329 ft09 E3 residual gap
- Spec: REQ-PHASE4-076 / SCENARIO-PHASE4-076
- Game: `ft09`
- Best verifier accuracy: 0.1000
- Residual mismatch class: `missing_world_model_rule_gap_actions_6`
- Reproducibility checksum: `9d7969995becae6b7704dd94148870d6eee2f7ff7f3c6ffaf036eefcf514ee00`
- Gap: bounded executable-world-model run did not satisfy the offline reproduced L1 gate.

### GAP-4331: Game-invariant ARC value representation - small encoder insufficient
- status: open
- evidence: `results/experiment_4331_self_learning_learned_frame_encoder_cross_game_transfer.json` reports cross_game_state_reduction=1.00849 with baseline_solves_held_out=true.
- failure mode: small learned frame encoder over the current solved set is insufficient to produce a decision-grade held-out search-state reduction.
- missing discriminator: game-invariant ARC value representation that recognizes progress across navigation, click-placement, rotation, and shallow-tail mechanics.
- candidate design: larger encoder with more reproduced games, adapter-conditioned value head, or experience-gated source-relevance features; preserve a CPU/hardware path.
- priority: medium

<!-- exp4333-gap-diffusiongemma-second-corpus-leak-free-scorer-4325:start -->
### GAP-DIFFUSIONGEMMA-SECOND-CORPUS-LEAK-FREE-SCORER-4325: Exp 4344 .401 filled verifier gap update
- status: filled (leak_robust_in_generation_partial_state_scorer_exp4338)
- evidence: results/experiment_4338_in_generation_moat_replicate_leak_robust.json; in_generation_moat_replicates=True; controls_differentiated=True; scorer_leak_recheck_passed=True; benchmark_n=240; replication_ci95=[0.283333, 0.4375].
- failure mode: the prior second-corpus leak-free scorer gap is filled by the leak-robust .401 replication.
- missing discriminator: none; the scorer now passes the answer-masked held-out leak recheck for this replication scope.
- candidate design: preserve the exp4338 leak-robust scorer protocol as the in-generation moat gate.
- priority: high
<!-- exp4333-gap-diffusiongemma-second-corpus-leak-free-scorer-4325:end -->

<!-- exp4333-gap-arc-grid-generation-scorer-4326:start -->
### GAP-ARC-GRID-GENERATION-SCORER-4326: Exp 4333 .400 verifier gap update
- status: open
- evidence: results/experiment_4326_adaptive_guided_generation_scaleup.json; adaptive_guidance_beats_control=False; domain_used=reasoning_corpus_fallback; adaptive_ci95=[-0.075, 0.35]; carnot_minus_best_control_delta=0.15.
- failure mode: adaptive DiffusionGemma guidance stayed bounded to a reasoning-corpus null and did not establish an ARC-grid generation scorer
- missing discriminator: oracle-distinct ARC-grid partial-state generation scorer that can rank candidate grid states during denoising rather than post-hoc reasoning choices
- candidate design: build a grid-native canvas scorer with masked-cell leak checks, no-adaptation controls, and ARC environment reproduction gates before rerunning adaptive guidance
- priority: high
<!-- exp4333-gap-arc-grid-generation-scorer-4326:end -->

<!-- exp4333-gap-e3-world-model-rule-ar25-4327:start -->
### GAP-E3-WORLD-MODEL-RULE-AR25-4327: Exp 4333 .400 verifier gap update
- status: open
- evidence: results/experiment_4327_e3_executable_world_model_ar25.json; game=ar25; offline_reproduced=False; reproduced_levels=0; verifier_best_accuracy=0.8875; residual_mismatch_class=missing_world_model_rule_gap_hidden_undo_stack_action7.
- failure mode: E3 induced world model for ar25 remained partial and could not execute a reproduced level through the real offline environment
- missing discriminator: ar25 executable world-model rule coverage for missing_world_model_rule_gap_hidden_undo_stack_action7
- candidate design: mine the divergent transition traces, add the missing action/rule cases to the executable model, and keep halt-on-divergence plus reproduce() as the gate
- priority: high
<!-- exp4333-gap-e3-world-model-rule-ar25-4327:end -->

<!-- exp4333-gap-e3-world-model-rule-ka59-4328:start -->
### GAP-E3-WORLD-MODEL-RULE-KA59-4328: Exp 4355 .402 filled verifier gap update
- status: filled (exp4350_ka59_l1_world_model)
- evidence: results/experiment_4350_e3_explore_verify_plan_ka59.json; offline_reproduced=True; reproduced_levels=1; verifier_best_accuracy=0.6375; world_model_path=results/arc_e3/ka59/world_model.py.
- failure mode: the prior ka59 action-rule blocker no longer prevents an offline reproduced L1 solve.
- missing discriminator: none for the L1 push-through-wall action plan; the remaining hidden HUD residual is tracked separately as GAP-E3-WORLD-MODEL-RULE-KA59-4350.
- candidate design: preserve the Exp 4350 adaptive transition tests and reproduce() gate for ka59 L1.
- priority: high
<!-- exp4333-gap-e3-world-model-rule-ka59-4328:end -->

<!-- exp4333-gap-e3-world-model-rule-tr87-4329:start -->
### GAP-E3-WORLD-MODEL-RULE-TR87-4329: Exp 4366 .403 filled verifier gap update
- status: filled (exp4363_tr87_ft09_world_models)
- evidence: results/experiment_4363_e3_mechanic_limited_tails_tr87_ft09.json; game=tr87; offline_reproduced=True; reproduced_levels=1; verifier_accuracy=1.0; residual_mismatch_class=none; world_model_path=results/arc_e3/tr87/world_model.py.
- failure mode: the prior tr87 partial world-model rule blocker no longer prevents an offline reproduced L1 gate.
- missing discriminator: none for the reproduced L1 tail-game plan; deeper future mechanics remain separate gaps if exposed.
- candidate design: preserve the Exp 4363 mechanic checks and reproduce() gate for this game.
- priority: high
<!-- exp4333-gap-e3-world-model-rule-tr87-4329:end -->

<!-- exp4333-gap-e3-world-model-rule-ft09-4329:start -->
### GAP-E3-WORLD-MODEL-RULE-FT09-4329: Exp 4366 .403 filled verifier gap update
- status: filled (exp4363_tr87_ft09_world_models)
- evidence: results/experiment_4363_e3_mechanic_limited_tails_tr87_ft09.json; game=ft09; offline_reproduced=True; reproduced_levels=1; verifier_accuracy=1.0; residual_mismatch_class=none; world_model_path=results/arc_e3/ft09/world_model.py.
- failure mode: the prior ft09 partial world-model rule blocker no longer prevents an offline reproduced L1 gate.
- missing discriminator: none for the reproduced L1 tail-game plan; deeper future mechanics remain separate gaps if exposed.
- candidate design: preserve the Exp 4363 mechanic checks and reproduce() gate for this game.
- priority: high
<!-- exp4333-gap-e3-world-model-rule-ft09-4329:end -->

<!-- exp4333-gap-4331:start -->
### GAP-4331: Exp 4333 .400 verifier gap update
- status: open_small_encoder_insufficient
- evidence: results/experiment_4331_self_learning_learned_frame_encoder_cross_game_transfer.json; upstream_missing_verifier_gap=true.
- failure mode: small learned frame encoder over the current solved set did not produce a decision-grade held-out OfflineSolver state reduction
- missing discriminator: game-invariant ARC value representation
- candidate design: larger learned frame encoder, more reproduced solved traces, or adapter-conditioned value head with a hardware-portable path
- priority: medium
<!-- exp4333-gap-4331:end -->

<!-- arc-search-heuristic-gaps-2026-06-17:start -->
## Missing SEARCH-HEURISTIC classes (move-distance program, 2026-06-17)

The goal-distance heuristic is a cheap, oracle-distinct PROGRESS verifier for the search (it
ranks states by estimated moves-to-win). We now have two — `cell_count_distance` (low cell-impact
games) and `misplaced_region_distance` (high cell-impact games) — selected dynamically by
`arc_heuristic_select`. The 8-game validation surfaced the heuristic CLASSES we still lack:

### GAP-ARC-HEURISTIC-DEEP-SPARSE: deep-sparse-win progress signal (wa30-class)
- status: open
- evidence: /tmp move-dist A/B + registry HARD TAIL note — wa30 (~33-deep keyboard nav) is NOT
  solved by BFS, cell_count, OR region_count at budget 8000 (all no-advance); cross-game verifier
  also did not transfer (66s vs 77s, both no-advance).
- failure mode: region-count is flat early in a long sparse game (every far-from-win state has a
  similar number of wrong regions), so it gives no usable gradient until near the goal; BFS
  exhausts the budget before reaching the win.
- missing discriminator: a progress signal that increases monotonically along a LONG plan before
  the grid visibly approaches the win (sub-goal / bottleneck / reachability distance, not surface).
- candidate design: learned game-specific value head (per-game RE / adapter), OR a sub-goal
  decomposition heuristic, OR depth-biased novelty (v3) with a much larger budget.
- priority: high (the deep tail is the bulk of the unsolved games)

### GAP-ARC-HEURISTIC-OBJECT-DISPLACEMENT: manhattan-to-target for piece-moving games
- status: open
- evidence: r11l region-count finds the optimal path, but for games where a specific agent/object
  must REACH a target cell, region-count only counts "wrong regions" — it cannot tell a piece 2
  cells from its goal from one 20 cells away (both = 1 wrong region).
- failure mode: region-count is move-aligned for FIX-IN-PLACE edits but blind to DISTANCE for
  move-an-object-to-a-target mechanics; it under-discriminates within an equal region-count tier.
- missing discriminator: summed manhattan distance of each misplaced object to its WIN position
  (object correspondence + displacement), as a tie-breaker within equal region counts.
- candidate design: a `manhattan_displacement_distance(win)` heuristic (match components grid↔win
  by color/shape, sum centroid manhattan); register in arc_heuristic_select; select for spatial
  piece-moving games (survey is_spatial_planning + low region-count variance).
- priority: medium

### GAP-ARC-HEURISTIC-CONDITION-WIN: predicate-distance for non-target-grid wins
- status: open
- evidence: all current goal-distance heuristics need a concrete win GRID to diff against; games
  whose win is a CONDITION ("all X collected", "counter == N") have no single target grid, so the
  heuristics are inapplicable (and a banked L1 win-grid mis-grounds a different level).
- failure mode: a grid-diff heuristic grounded on one level's win-state does not transfer to other
  levels of the same game when levels are different puzzles sharing only the win CONDITION.
- missing discriminator: distance measured against the win PREDICATE (count of unmet sub-conditions
  / distance to the condition boundary), not a fixed grid — ties this to the GameAdapter
  hand_verifier as the goal signal.
- candidate design: a `condition_distance(hand_verifier)` heuristic that scores how close a frame
  is to satisfying the adapter's win predicate; level-general by construction.
- priority: medium
<!-- arc-search-heuristic-gaps-2026-06-17:end -->

<!-- arc-router-training-gaps-2026-06-17:start -->
## TRAINED-ROUTER training-data gaps (2026-06-17)

The trained router (`arc_router`, 8/8 leave-one-out) routes the goal-distance HEURISTIC because
that is where we have labelled outcomes. The operator's full ask — "properly train our TRMs and
BFS and DFS and routers" — needs two more label sets the router schema already accommodates:

### GAP-ARC-ROUTER-ENGINE-LABELS: engine-vs-engine A/B per game
- status: answered 2026-06-17 (engine choice ruled out as the deep-tail lever; v2-BFS dominant)
- evidence: engine A/B (results/arc_engine_ab.json; docs/research-notes/arc-engine-ab-2026-06-17.md)
  ran v2-BFS vs v3-novelty vs v3+learned-verifier over 3 solved + 3 deep games.
- finding: v2-BFS WINS every solvable game on path-optimality (v3-novelty solves but with 4-15x
  longer paths: 29/58/15 vs 7/3/5 actions). v3 cracks NO deep game (wa30/g50t/sb26 fail under BOTH
  at 20k). A learned verifier guiding v3 cuts expansions -52% (r11l 1064 vs 2236) but loses
  path-optimality -> the efficiency lever is a GOOD per-game verifier (= the OfflineSolver/
  verifier-routed path), NOT a generic engine swap.
- resolution: engine routing collapses to "v2-BFS for first-contact"; no separate engine-selector
  needed. The deep tail's lever is per-game RE + GAP-ARC-TRM-TRAINED-ON-ARC (a well-trained per-game
  verifier/representation), not engine choice. graph_explore_solve_v3 gained a stats param for the
  measurement.

### GAP-ARC-TRM-TRAINED-ON-ARC: an ARC-trained TRM generator/refiner
- status: open
- evidence: the running TRM (nano-trm) is sudoku-trained; no TRM is trained on ARC solve traces.
- failure mode: the hybrid architecture's generator/refiner slot has no ARC-specialised TRM, so the
  router has no TRM-guided engine to route to and first-contact exploration is search-only.
- missing discriminator: a TRM trained on the accumulated captured ARC trajectories (gap_fills/ +
  results/arc_explore_trajectory_*.json) that proposes/refines candidate action sequences.
- candidate design: prepare an ARC trajectory dataset from the captured solves; train a TRM
  (full-FT > LoRA per the TTA-TRM finding) on a 3090; wire as a TRM-guided rollout engine.
- priority: medium (heavier GPU track; gated on enough captured trajectories)
- **UPDATE 2026-07-04 (outer-loop):** two new supporting arguments, full writeup at
  `docs/research-notes/trm-arc-action-sequence-generator-2026-07-04.md`. (1) Verified
  `results/experiment_sudoku_energy_vs_ar_v1.json`'s "recursive refinement beats AR" result IS
  literally the TRM recipe (z←block(x,y,z); y←head(z), 8 cycles, deep supervision) — 18.2% solve rate
  vs AR's ~0-0.2%, while a near-perfect energy scorer still failed to generate (0%, worse than
  random) — the exact generation-vs-scoring failure shape GAP-4891 diagnosed for ARC's own
  enumeration wall, on a different combinatorial domain. (2) arXiv:2604.07822 ("Loop, Think, &
  Generalize") gives a mechanistic reason recurrent-depth computation should help with ARC's specific
  compositional-generalization challenge, plus two concrete design rules for any build attempt: train
  with dynamic (not fixed) recursion depth, and instrument explicitly for "overthinking" (accuracy
  peaks at some recursion depth, then degrades). Priority should move to first running the note's
  proposed leave-one-game-out overfit-generalization pilot (a cheap, falsifiable test of whether the
  current 69-level trajectory corpus carries enough signal at all) BEFORE committing to the full
  build this entry already scopes.
- **UPDATE 2026-07-04 #2 (outer-loop, operator asked "don't we have the human generated game event
  solutions handy?"):** yes — full writeup at `docs/research-notes/human-replay-corpus-staging-bug-
  and-opportunity-2026-07-04.md`. A real, licensed human replay corpus already exists
  (`data/arc_public_demo_human_replay_corpus/`, CC BY 4.0) with **144 complete winning human
  trajectories across all 25 public games** (~90,000+ actions), each with validated per-step
  level-completion signal in the raw source — this supersedes the 69-Carnot-agent-level corpus above
  as the priority training-data source. BUT the *staged* form (the shards actually wired for
  training) has a real bug: `level_progress` reads `0.0` in all 14,797 staged rows, including full
  winning sessions — the staging conversion (`exp4495`) dropped `won`/`levels_completed`/`state`/
  `available_actions` when building the training shards. Fixing this also unblocks `exp4490` (the
  original frame-change-predictor consumer), which has sat `blocked_human_replay_corpus_not_cached`
  since 2026-06-20 (ran ~1.5h before staging completed, never retried since). Concrete next step:
  re-stage the corpus preserving the win-segmentation fields (and resolve the GAME_OVER/retry
  discontinuity question the note documents) BEFORE either exp4490 or a TRM pilot consumes it.
<!-- arc-router-training-gaps-2026-06-17:end -->

### 2026-06-17 Exp4340 ka59 E3 residual gap
- Spec: REQ-PHASE4-079 / SCENARIO-PHASE4-079
- Best verifier accuracy: 0.5625
- Residual mismatch class: `hidden_step_counter_hud_gap`
- Reproducibility checksum: `9c238354e5ff765c0291ff09777e55cc0dd8661416b71bd69bc4692b7af4d6ea`
- Gap: bounded explore-verify-plan run did not satisfy the offline reproduced L1 gate.

<!-- exp4344-gap-e3-world-model-rule-ar25-4339:start -->
### GAP-E3-WORLD-MODEL-RULE-AR25-4339: Exp 4344 .401 verifier gap update
- status: open_residual_after_l1_reproduction
- evidence: results/experiment_4339_e3_explore_verify_plan_ar25.json; game=ar25; offline_reproduced=True; reproduced_levels=1; verifier_best_accuracy=0.8875; residual_mismatch_class=missing_world_model_rule_gap_hidden_undo_stack_action7.
- failure mode: E3 induced world model for ar25 still exposes residual mismatch missing_world_model_rule_gap_hidden_undo_stack_action7 after the .401 run
- missing discriminator: ar25 executable world-model rule coverage for missing_world_model_rule_gap_hidden_undo_stack_action7
- candidate design: mine the divergent transition traces, add the missing action/rule cases to the executable model, and keep halt-on-divergence plus reproduce() as the gate
- priority: high
<!-- exp4344-gap-e3-world-model-rule-ar25-4339:end -->

<!-- exp4344-gap-e3-world-model-rule-ka59-4340:start -->
### GAP-E3-WORLD-MODEL-RULE-KA59-4340: Exp 4344 .401 verifier gap update
- status: open
- evidence: results/experiment_4340_e3_explore_verify_plan_ka59.json; game=ka59; offline_reproduced=False; reproduced_levels=0; verifier_best_accuracy=0.5625; residual_mismatch_class=hidden_step_counter_hud_gap.
- failure mode: E3 induced world model for ka59 still exposes residual mismatch hidden_step_counter_hud_gap after the .401 run
- missing discriminator: ka59 executable world-model rule coverage for hidden_step_counter_hud_gap
- candidate design: mine the divergent transition traces, add the missing action/rule cases to the executable model, and keep halt-on-divergence plus reproduce() as the gate
- priority: high
<!-- exp4344-gap-e3-world-model-rule-ka59-4340:end -->

<!-- exp4344-gap-4342:start -->
### GAP-4342: Exp 4344 .401 verifier gap update
- status: open_third_null_retired_direction
- evidence: results/experiment_4342_self_learning_action_role_cross_game_encoder.json; upstream_missing_verifier_gap=true.
- failure mode: game-agnostic action-role interaction value head did not produce a decision-grade held-out OfflineSolver state reduction
- missing discriminator: transferable object-interaction value representation
- candidate design: larger interaction encoder, richer affordance discovery, or more reproduced traces before retiring cross-game value transfer
- priority: high
<!-- exp4344-gap-4342:end -->

<!-- arc-live-generalization-gaps-2026-06-17:start -->
## ARC live-generalization gaps (2026-06-17) — per-game RE does not transfer

The tn36 deep-RE climb solved L1-L6 by reading INTERNAL game state + manual RE. It does NOT
generalize to live unseen games (only rendered frames available live). See
docs/research-notes/arc-live-generalization-gap-2026-06-17.md. The two builds that close it:

### GAP-ARC-LIVE-FRAME-ONLY-INDUCTION: discover a game's mechanic from FRAMES alone
- status: open (E3 world-model substrate exists; deep mechanics not yet induced frame-only)
- evidence: scripts/arc3_tn36_offline_solver.py has 12 internal-state accesses
  (env._game.fdksqlmpki...); a live submission exposes only 64x64 frames, so this computation
  is non-transferable. Manual LLM RE is not an automated process.
- missing discriminator: an automated probe->observe->induce loop that derives the transition
  model + win predicate + control mechanic (program-editor, timed-trap, checkpoint) from frame
  transitions only, replacing the internal-state reads + manual RE.
- candidate design: extend arc_executable_world_model.py (E3, frame-only) to detect the discovered
  mechanic CLASSES from observable frame signals (palette-of-glyphs->program-editor; periodic-cell-
  visibility->timed-trap; base-advance-on-region->checkpoint).
- priority: high (the foundation of any live-submission capability)

### GAP-ARC-STRATEGY-ROUTER: route to a solving STRATEGY-class, not just a heuristic
- status: building (routing layer + ALL FOUR class solvers SHIPPED 2026-06-17 —
  arc_strategy_router.py + arc_maze_planner.py {checkpoint_multirun_plan reproduces tn36 L6,
  timed_trap_plan reproduces tn36 L7 = first L7 solve}; remaining: frame-only induction of each
  class's MODEL — the program-editor transition verifier + the maze MazeModel from frames)
- evidence: arc_router routed only {bfs,cell_count,region_count} goal-distance heuristics with no
  awareness of mechanic CLASSES. NOW: arc_strategy_router is the Tier-1 STRATEGY layer above it —
  route_for_game(game, mechanic=...) maps a detected mechanic to its strategy and SHORT-CIRCUITS the
  goal-distance heuristic for program-editor (a category error: tn36 ran graph_explore clean but
  every heuristic NO-ADVANCEs). recommend_approach() now returns `strategy` as the first decision.
- missing discriminator (now SUPPLIED at the routing level): mechanic-class detection with live-
  correct precedence — injected frame-only verdict (arc3_frame_induction.induce, zero internal state)
  > registry mechanic_class > default graph_explore; tn36 records mechanic_class: program_editor.
- candidate design (DONE for routing): strategy-class library (program_editor / graph_explore wired;
  checkpoint_multirun / timed_trap_aware declared, wired: False) each declaring applicability features
  + solver entrypoint. Unit-tested (tests/python/test_arc_strategy_router.py, 6 tests).
- priority: high (turns per-game wins into dynamic transfer) — routing turned wins into transfer;
  the per-class SOLVER builds (esp. the program_editor offline transition verifier,
  GAP-ARC-PROGRAM-EDITOR-NO-GRADED-FEEDBACK) are the remaining work.

### GAP-ARC-PROGRAM-EDITOR-NO-GRADED-FEEDBACK: program-editor games are blind-search-only frame-only
- status: filled at the MODEL level (python/carnot/agentic/arc_program_editor_model.py, 2026-06-17) —
  the offline transition model supplies the distance-to-target gradient the frames withhold: 100%
  win-bit agreement vs the tn36 env (105/105), 5/5 model-guided env-confirmed solves L1-L5, ~47M-blind
  -> ~5-32 directed expansions (results/experiment_program_editor_transition_model_validation.json).
  RESIDUAL: frame-only induction of the model INPUTS (object + TARGET attrs) — target attrs are the
  frame residual (GAP-ARC-MAZE-MODEL-FRAME-INDUCTION); known games read them from internal state.
- evidence: scripts/arc3_frame_induction.py + three frame-only probes (2026-06-17, design note
  "general winner-discovery is BLIND search"). On tn36: (1) the run is ATOMIC — one env.step runs
  the whole move-program and resets the object, so per-move motion is frame-invisible (object region
  (31,21,2268) unchanged across 8 post-run inert steps); (2) a wrong edit echoes the same ~4 cells as
  a correct edit (no closeness signal); (3) a losing run renders exactly 1 attempt-counter cell,
  identical for k=0/1/2 slots-correct (no partial-attribute-match). Only a FULL win advances
  levels_completed. frame_only_winner_search solves L1 in 4 blind runs but the reachable space is
  already 1024 (5 slots × 4 glyphs, 2 located bit-rows) and exponential in program length.
- failure mode: with only a binary win bit and no gradient, frame-only winner-discovery is blind
  program-space search with no pruning — tractable on L1 only via a uniform-program prior + small
  alphabet; it does not scale, and the code semantics cannot be induced online (atomic run hides the
  dynamics). The verifier has nothing to rank intermediate programs by.
- missing discriminator: a per-class DYNAMICS/transition verifier ("editor code → object transform")
  that scores a candidate program's predicted end-state against the target's frame-read attributes —
  supplying the gradient the live frame stream withholds.
- candidate design: train the program-editor transition model OFFLINE (from the banked internal-state
  RE + multiple program-editor games), package it as the program_editor STRATEGY-class model
  (GAP-ARC-STRATEGY-ROUTER); the live agent reads editor glyphs + target attributes from frames and
  PLANS with the learned model instead of blind online search.
- priority: high (blind search caps live program-editor solvability; the offline-learned transition
  verifier is the unlock — and is squarely Carnot's verifier-as-product thesis)
### GAP-ARC-MAZE-MODEL-FRAME-INDUCTION: a planner-ready MazeModel can't be induced from atomic-run frames
- status: RESOLVED end-to-end (2026-06-17). frame_to_maze_model assembles a complete
  arc_maze_planner.MazeModel from ONE frame -> the maze planner -> executed -> the real env WINS on tn36
  L6 (checkpoint-multirun, 2 legs) AND L7 (timed-trap, 3 legs); zero internal state on the perception
  path (move-codes + cadence from the offline transition model). Key fix: walls are ROW-RUNS not bboxes
  (a bbox fills a concave passage and over-blocks the planner — the L7 failure mode).
  results/experiment_full_frame_only_maze_solve_validation.json. The live-generalization chain (detect
  -> route -> perceive -> plan -> env-confirmed solve) is now complete frame-only for BOTH the
  program-editor transform class (L1-L5) and the maze class (L6-L7).
  --- field-level detail (still valid): ---
  (1) TARGET attrs: induce_object_target_attrs reads object+target x/y/scale/rotation/property
  EXACTLY (the target is a HOLLOW OUTLINE sprite) -> frame-induced attrs plan to an env-confirmed win on
  tn36 L1-L5 (5/5; results/experiment_frame_induced_target_attrs_validation.json). (2) MAZE sub-fields:
  induce_maze_sub_fields reads CHECKPOINTS (a DITHERED 4x4 of the object colour) + the HAZARD band
  (distinct low-area marker colours) EXACTLY vs internal truth on L6 (3/3 cp, no hazard) and L7 (3/3 cp +
  exact spike band); results/experiment_frame_induced_maze_subfields_validation.json. RESIDUAL: the full
  frame-only maze SOLVE needs COMPLETE wall geometry from one frame + move-code induction + the
  spikes_hidden hitbox — a separate integration step, not a field-induction gap.
- (was) status: open (frame-only inducer ships for the OBSERVABLE parts; tn36's critical fields are not frame-rendered)
- evidence: scripts/arc3_frame_induction.py:induce_maze_model + the 2026-06-17 frame probes (design
  note "frame-only MazeModel induction"). The behavioral inducer recovers the OBJECT (the colour whose
  centroid moves across frames) and WALLS (static non-floor structure) — validated on synthetic frames
  + observed on tn36 (object colour 11 moves; walls colour 6). BUT the planner-CRITICAL fields are not
  frame-distinct in tn36: the TARGET draws on floor colour 4, the CHECKPOINTS draw on the floor
  checkerboard (colour 5), and the spike HAZARDS are invisible at rest (and the run is atomic, hiding
  the mid-run flash). So induce_maze_model returns usable_model=False for tn36 — the maze model falls
  back to internal state.
- failure mode: the maze planners (arc_maze_planner) need target + checkpoints + hazard boxes; for the
  atomic-run program-editor maze (tn36 L6/L7) those are not in the frames, so a fully frame-only live
  solve of that class is blocked. (A DIRECT-CONTROL maze that renders a distinct target + walls IS
  fully inducible by the same primitives — usable_model=True on the synthetic clean case.)
- missing discriminator: a perception/verifier that recovers the floor-blended checkpoints + the
  invisible hazards — e.g. EFFECT-probing (a cell that, when a run ends on it, persists base advance =
  checkpoint) and TEMPORAL hazard detection across run-animation frames — or, for the program-editor
  class, the offline transition verifier (GAP-ARC-PROGRAM-EDITOR-NO-GRADED-FEEDBACK) that supplies the
  model the frames withhold.
- candidate design: (1) effect-probe checkpoints (run-end base-persistence); (2) temporal blink
  detection over the run-animation frame stream for hazards; (3) for atomic-run editors, source the
  MazeModel from the offline-trained transition model rather than live frames.
- priority: medium (object+walls induce today; the residual is the same atomic-run/rendering limit as
  the program-editor gap — both point at the offline transition verifier as the durable unlock)
<!-- arc-live-generalization-gaps-2026-06-17:end -->

### 2026-06-17 Exp4350 ka59 E3 residual gap
- Spec: REQ-PHASE4-082 / SCENARIO-PHASE4-082
- Best verifier accuracy: 0.5687
- Residual mismatch class: `hidden_step_counter_hud_gap`
- Reproducibility checksum: `8e6e766c670f6890f2a8e3c704e6c5847bb40381e11a51285f283e83849572ee`
- Gap: bounded explore-verify-plan run did not satisfy the offline reproduced L1 gate.

### 2026-06-17 Exp4352 tr87 E3 residual gap
- Spec: REQ-PHASE4-084 / SCENARIO-PHASE4-084
- Game: `tr87`
- Best verifier accuracy: 0.0000
- Residual mismatch class: `missing_world_model_rule_gap_actions_1_2_3_4`
- Reproducibility checksum: `88ebeb795a2569c299b14d073f12934f23c2738279d6c561623f8af4293b7f28`
- Gap: bounded explore-verify-plan did not satisfy the offline reproduced L1 gate.

### 2026-06-17 Exp4352 ft09 E3 residual gap
- Spec: REQ-PHASE4-084 / SCENARIO-PHASE4-084
- Game: `ft09`
- Best verifier accuracy: 0.0500
- Residual mismatch class: `missing_world_model_rule_gap_actions_6`
- Reproducibility checksum: `88ebeb795a2569c299b14d073f12934f23c2738279d6c561623f8af4293b7f28`
- Gap: bounded explore-verify-plan did not satisfy the offline reproduced L1 gate.

<!-- exp4355-gap-e3-world-model-rule-ka59-4350:start -->
### GAP-E3-WORLD-MODEL-RULE-KA59-4350: Exp 4355 .402 verifier gap update
- status: open_residual_after_l1_reproduction
- evidence: results/experiment_4350_e3_explore_verify_plan_ka59.json; offline_reproduced=True; reproduced_levels=1; verifier_best_accuracy=0.6375; residual_mismatch_class=hidden_step_counter_hud_gap.
- failure mode: ka59 L1 now reproduces, but the executable world model still has residual mismatch hidden_step_counter_hud_gap
- missing discriminator: ka59 hidden StepCounter HUD dynamics
- candidate design: model the hidden bottom-row HUD counter separately from win-state movement so exact transition tests can pass without corrupting L1 solve logic
- priority: high
<!-- exp4355-gap-e3-world-model-rule-ka59-4350:end -->

<!-- exp4355-gap-e3-world-model-rule-sc25-l2-4351:start -->
### GAP-E3-WORLD-MODEL-RULE-SC25-L2-4351: Exp 4355 .402 verifier gap update
- status: open_deeper_level_residual
- evidence: results/experiment_4351_e3_deeper_solved_games.json; game=sc25; offline_reproduced=False; new_reproduced_level=1; residual=sc25_l2_live_recorded_not_offline_reproduced_spell_delta_gap.
- failure mode: sc25 deeper level remains unreproduced due to sc25_l2_live_recorded_not_offline_reproduced_spell_delta_gap
- missing discriminator: sc25 executable rule coverage for sc25_l2_live_recorded_not_offline_reproduced_spell_delta_gap
- candidate design: mine divergent deeper-level traces, add the missing executable transition cases, and keep reproduce() as the only solved-level gate
- priority: high
<!-- exp4355-gap-e3-world-model-rule-sc25-l2-4351:end -->

<!-- exp4355-gap-e3-world-model-rule-ar25-l2-4351:start -->
### GAP-E3-WORLD-MODEL-RULE-AR25-L2-4351: Exp 4355 .402 verifier gap update
- status: open_deeper_level_residual
- evidence: results/experiment_4351_e3_deeper_solved_games.json; game=ar25; offline_reproduced=False; new_reproduced_level=1; residual=ar25_l2_hidden_rule_delta_not_reproduced_action7_undo_stack_gap.
- failure mode: ar25 deeper level remains unreproduced due to ar25_l2_hidden_rule_delta_not_reproduced_action7_undo_stack_gap
- missing discriminator: ar25 executable rule coverage for ar25_l2_hidden_rule_delta_not_reproduced_action7_undo_stack_gap
- candidate design: mine divergent deeper-level traces, add the missing executable transition cases, and keep reproduce() as the only solved-level gate
- priority: high
<!-- exp4355-gap-e3-world-model-rule-ar25-l2-4351:end -->

<!-- exp4355-gap-e3-world-model-rule-tr87-4352:start -->
### GAP-E3-WORLD-MODEL-RULE-TR87-4352: Exp 4366 .403 filled verifier gap update
- status: filled (exp4363_tr87_ft09_world_models)
- evidence: results/experiment_4363_e3_mechanic_limited_tails_tr87_ft09.json; game=tr87; offline_reproduced=True; reproduced_levels=1; verifier_accuracy=1.0; residual_mismatch_class=none; world_model_path=results/arc_e3/tr87/world_model.py.
- failure mode: the prior tr87 partial world-model rule blocker no longer prevents an offline reproduced L1 gate.
- missing discriminator: none for the reproduced L1 tail-game plan; deeper future mechanics remain separate gaps if exposed.
- candidate design: preserve the Exp 4363 mechanic checks and reproduce() gate for this game.
- priority: high
<!-- exp4355-gap-e3-world-model-rule-tr87-4352:end -->

<!-- exp4355-gap-e3-world-model-rule-ft09-4352:start -->
### GAP-E3-WORLD-MODEL-RULE-FT09-4352: Exp 4366 .403 filled verifier gap update
- status: filled (exp4363_tr87_ft09_world_models)
- evidence: results/experiment_4363_e3_mechanic_limited_tails_tr87_ft09.json; game=ft09; offline_reproduced=True; reproduced_levels=1; verifier_accuracy=1.0; residual_mismatch_class=none; world_model_path=results/arc_e3/ft09/world_model.py.
- failure mode: the prior ft09 partial world-model rule blocker no longer prevents an offline reproduced L1 gate.
- missing discriminator: none for the reproduced L1 tail-game plan; deeper future mechanics remain separate gaps if exposed.
- candidate design: preserve the Exp 4363 mechanic checks and reproduce() gate for this game.
- priority: high
<!-- exp4355-gap-e3-world-model-rule-ft09-4352:end -->

### 2026-06-17 Exp4361 sc25 E3 residual gap
- Spec: REQ-PHASE4-085 / SCENARIO-PHASE4-085
- Game: `sc25`
- Best verifier accuracy: 1.0000
- Residual mismatch class: `sc25_l2_live_recorded_not_offline_reproduced_spell_delta_gap`
- Reproducibility checksum: `bce9878e3d5396e127ea1342fd0452b841b61935eb539e1da466055962842a90`
- Gap: sc25 L1 remains reproduced, but L2 from the live-recorded spell-cast/cast-grid/tank-controls sequence did not satisfy the offline reproduced-level gate.

### 2026-06-17 Exp4361 tn36 E3 residual gap
- Spec: REQ-PHASE4-085 / SCENARIO-PHASE4-085
- Game: `tn36`
- Best verifier accuracy: 0.8750
- Residual mismatch class: `tn36_l8_program_editor_maze_delta_gap`
- Reproducibility checksum: `bce9878e3d5396e127ea1342fd0452b841b61935eb539e1da466055962842a90`
- Gap: tn36 L7 remains reproduced, but the program-editor solver did not extend through L8 under the offline reproduction gate.

### 2026-06-17 Exp4361 lp85 E3 residual gap
- Spec: REQ-PHASE4-085 / SCENARIO-PHASE4-085
- Game: `lp85`
- Best verifier accuracy: 1.0000
- Residual mismatch class: `lp85_l5_search_path_not_offline_reproduced_reset_replay_gap`
- Reproducibility checksum: `bce9878e3d5396e127ea1342fd0452b841b61935eb539e1da466055962842a90`
- Gap: lp85 search reached L5, but reset replay via `arc_solver_kit.reproduce()` only reproduced through L4, so L5 does not count.

### 2026-06-17 Exp4362 ar25 named hidden-rule residual gap
- Spec: REQ-PHASE4-086 / SCENARIO-PHASE4-086
- Best verifier accuracy: 0.8688
- Residual gap class: `ar25_l2_hidden_rule_delta_not_reproduced_action7_undo_stack_gap`
- Reproducibility checksum: `3f26809ef8c93a4f0dab633c35a36a516ffe4cc1b7d2d49fb67982681570c77d`
- Gap: bounded explore-verify-plan did not reproduce a new level beyond L1.

### 2026-06-17 Exp4362 ka59 named hidden-rule residual gap
- Spec: REQ-PHASE4-086 / SCENARIO-PHASE4-086
- Best verifier accuracy: 0.6438
- Residual gap class: `ka59_l2_hidden_step_counter_hud_register_gap`
- Reproducibility checksum: `3f26809ef8c93a4f0dab633c35a36a516ffe4cc1b7d2d49fb67982681570c77d`
- Gap: bounded explore-verify-plan did not reproduce a new level beyond L1.

<!-- exp4366-gap-e3-world-model-rule-sc25-l2-4361:start -->
### GAP-E3-WORLD-MODEL-RULE-SC25-L2-4361: Exp 4377 .404 verifier gap update
- status: open
- evidence: results/experiment_4372_e3_deeper_high_headroom_games.json; game=sc25; offline_reproduced=False; prior_best_level=1; new_reproduced_level=1; verifier_accuracy=1.0; residual=sc25_l2_spell_delta_gap.
- failure mode: sc25 L2 remains unreproduced due to sc25_l2_spell_delta_gap
- missing discriminator: sc25 executable world-model rule coverage for sc25_l2_spell_delta_gap
- candidate design: mine divergent active traces for the named residual, add transition tests, and count progress only through the offline reproduce() gate
- priority: high
<!-- exp4366-gap-e3-world-model-rule-sc25-l2-4361:end -->

<!-- exp4366-gap-e3-world-model-rule-tn36-l8-4361:start -->
### GAP-E3-WORLD-MODEL-RULE-TN36-L8-4361: Exp 4377 .404 verifier gap update
- status: open
- evidence: results/experiment_4372_e3_deeper_high_headroom_games.json; game=tn36; offline_reproduced=False; prior_best_level=7; new_reproduced_level=7; verifier_accuracy=0.875; residual=tn36_l8_program_editor_object_control_gap.
- failure mode: tn36 L8 remains unreproduced due to tn36_l8_program_editor_object_control_gap
- missing discriminator: tn36 executable world-model rule coverage for tn36_l8_program_editor_object_control_gap
- candidate design: mine divergent active traces for the named residual, add transition tests, and count progress only through the offline reproduce() gate
- priority: high
<!-- exp4366-gap-e3-world-model-rule-tn36-l8-4361:end -->

<!-- exp4366-gap-e3-world-model-rule-lp85-l5-4361:start -->
### GAP-E3-WORLD-MODEL-RULE-LP85-L5-4361: Exp 4377 .404 filled verifier gap update
- status: filled (exp4372_lp85_l5_world_model)
- evidence: results/experiment_4372_e3_deeper_high_headroom_games.json; game=lp85; offline_reproduced=True; new_reproduced_level=5; verifier_accuracy=1.0; world_model_path=python/carnot/agentic/arc_game_adapters.py.
- failure mode: the prior lp85 L5 reset-replay blocker no longer prevents an offline reproduced L5 gate.
- missing discriminator: none for the reproduced L5 plan; deeper future mechanics remain separate gaps if exposed.
- candidate design: preserve the Exp 4372 reproduce() gate for lp85 L5.
- priority: high
<!-- exp4366-gap-e3-world-model-rule-lp85-l5-4361:end -->

<!-- exp4366-gap-e3-world-model-rule-ar25-l2-4362:start -->
### GAP-E3-WORLD-MODEL-RULE-AR25-L2-4362: Exp 4377 .404 verifier gap update
- status: open
- evidence: results/experiment_4373_e3_blocked_mechanic_levels_ar25_ka59_ft09.json; game=ar25; offline_reproduced=False; prior_best_level=1; new_reproduced_level=1; verifier_accuracy=0.958333; residual=ar25_l2_action7_undo_stack_hidden_rule_gap.
- failure mode: ar25 L2 remains unreproduced due to ar25_l2_action7_undo_stack_hidden_rule_gap
- missing discriminator: ar25 executable world-model rule coverage for ar25_l2_action7_undo_stack_hidden_rule_gap
- candidate design: mine divergent active traces for the named residual, add transition tests, and count progress only through the offline reproduce() gate
- priority: high
<!-- exp4366-gap-e3-world-model-rule-ar25-l2-4362:end -->

<!-- exp4366-gap-e3-world-model-rule-ka59-l2-4362:start -->
### GAP-E3-WORLD-MODEL-RULE-KA59-L2-4362: Exp 4377 .404 verifier gap update
- status: open
- evidence: results/experiment_4373_e3_blocked_mechanic_levels_ar25_ka59_ft09.json; game=ka59; offline_reproduced=False; prior_best_level=1; new_reproduced_level=1; verifier_accuracy=0.15625; residual=ka59_l2_hidden_step_counter_hud_register_gap.
- failure mode: ka59 L2 remains unreproduced due to ka59_l2_hidden_step_counter_hud_register_gap
- missing discriminator: ka59 executable world-model rule coverage for ka59_l2_hidden_step_counter_hud_register_gap
- candidate design: mine divergent active traces for the named residual, add transition tests, and count progress only through the offline reproduce() gate
- priority: high
<!-- exp4366-gap-e3-world-model-rule-ka59-l2-4362:end -->

### GAP-4370: ARC LLM-generated action-cost residual
- status: open
- evidence: `results/experiment_4370_llm_generated_action_cost_heuristics.json` reports unreduced held-out levels: lp85:L3, lp85:L4, tu93:L3, tu93:L4, tr87:L3, tr87:L4, tr87:L5, tr87:L6, tn36:L7.
- failure mode: no clean generated heuristic reduced reproduced held-out actions below the deployed linear action-cost baseline.
- missing discriminator: observable grid/action feature that predicts a strictly shorter valid plan than the current linear cost.
- candidate design: add richer per-game transition features or collect more reproduced levels before re-running generated-program selection.
- priority: medium

### 2026-06-18 Exp4373 ar25 named active-data residual gap
- Spec: REQ-PHASE4-4373 / SCENARIO-PHASE4-4373
- Best verifier accuracy: 0.9583
- Active transitions collected: 96
- Target action counts: `{"7": 10}`
- Residual gap class: `ar25_l2_action7_undo_stack_hidden_rule_gap`
- Reproducibility checksum: `4c775367e4b09eaf060085f9df2e1617bb79af96c87fcd0996ef15bce2b79310`
- Gap: bounded targeted active-data pass did not reproduce a new level beyond L1.

### 2026-06-18 Exp4373 ka59 named active-data residual gap
- Spec: REQ-PHASE4-4373 / SCENARIO-PHASE4-4373
- Best verifier accuracy: 0.1562
- Active transitions collected: 96
- Target action counts: `{"1": 13, "2": 15, "3": 18, "4": 11, "6": 39}`
- Residual gap class: `ka59_l2_hidden_step_counter_hud_register_gap`
- Reproducibility checksum: `4c775367e4b09eaf060085f9df2e1617bb79af96c87fcd0996ef15bce2b79310`
- Gap: bounded targeted active-data pass did not reproduce a new level beyond L1.

### 2026-06-18 Exp4373 ft09 named active-data residual gap
- Spec: REQ-PHASE4-4373 / SCENARIO-PHASE4-4373
- Best verifier accuracy: 0.1562
- Active transitions collected: 95
- Target action counts: `{"6": 95}`
- Residual gap class: `ft09_l2_residual_world_model_mismatch_gap`
- Reproducibility checksum: `4c775367e4b09eaf060085f9df2e1617bb79af96c87fcd0996ef15bce2b79310`
- Gap: bounded targeted active-data pass did not reproduce a new level beyond L1.

<!-- exp4377-gap-e3-world-model-rule-ft09-l2-4373:start -->
### GAP-E3-WORLD-MODEL-RULE-FT09-L2-4373: Exp 4377 .404 verifier gap update
- status: open
- evidence: results/experiment_4373_e3_blocked_mechanic_levels_ar25_ka59_ft09.json; game=ft09; offline_reproduced=False; prior_best_level=1; new_reproduced_level=1; verifier_accuracy=0.15625; residual=ft09_l2_residual_world_model_mismatch_gap.
- failure mode: ft09 L2 remains unreproduced due to ft09_l2_residual_world_model_mismatch_gap
- missing discriminator: ft09 executable world-model rule coverage for ft09_l2_residual_world_model_mismatch_gap
- candidate design: mine divergent active traces for the named residual, add transition tests, and count progress only through the offline reproduce() gate
- priority: high
<!-- exp4377-gap-e3-world-model-rule-ft09-l2-4373:end -->

### 2026-06-18 Exp4384 ar25 Mind-Studio lookahead residual gap
- Spec: REQ-PHASE4-4384 / SCENARIO-PHASE4-4384
- Best verifier accuracy: 0.9062
- K-step lookahead fidelity: 0.7333
- Active transitions collected: 96
- Target action counts: `{"7": 12}`
- Residual gap class: `ar25_l2_action7_undo_stack_hidden_rule_gap`
- Reproducibility checksum: `a0661e0d5b21d51275b5700ffdbe9e012c306fcf26f39a27afba305ad11eb3b9`
- Gap: bounded active-data plus K-step named-register fidelity did not reproduce a new level beyond L1.

### 2026-06-18 Exp4384 ka59 Mind-Studio lookahead residual gap
- Spec: REQ-PHASE4-4384 / SCENARIO-PHASE4-4384
- Best verifier accuracy: 0.1250
- K-step lookahead fidelity: 0.1123
- Active transitions collected: 96
- Target action counts: `{"1": 12, "2": 18, "3": 14, "4": 10, "6": 42}`
- Residual gap class: `ka59_l2_hidden_step_counter_hud_register_gap`
- Reproducibility checksum: `a0661e0d5b21d51275b5700ffdbe9e012c306fcf26f39a27afba305ad11eb3b9`
- Gap: bounded active-data plus K-step named-register fidelity did not reproduce a new level beyond L1.

### 2026-06-18 Exp4384 ft09 Mind-Studio lookahead residual gap
- Spec: REQ-PHASE4-4384 / SCENARIO-PHASE4-4384
- Best verifier accuracy: 0.1667
- K-step lookahead fidelity: 0.3475
- Active transitions collected: 95
- Target action counts: `{"6": 95}`
- Residual gap class: `ft09_l2_residual_world_model_mismatch_gap`
- Reproducibility checksum: `a0661e0d5b21d51275b5700ffdbe9e012c306fcf26f39a27afba305ad11eb3b9`
- Gap: bounded active-data plus K-step named-register fidelity did not reproduce a new level beyond L1.

<!-- exp4388-gap-e3-world-model-rule-lp85-l6-4383:start -->
### GAP-E3-WORLD-MODEL-RULE-LP85-L6-4383: Exp 4388 .405 verifier gap update
- status: open
- evidence: results/experiment_4383_e3_deeper_high_headroom_lookahead.json; game=lp85; offline_reproduced=False; prior_best_level=5; new_reproduced_level=5; target_level=6; verifier_accuracy=0.833333; lookahead_fidelity=0.833333; residual=wall_time_cap_exhausted.
- failure mode: lp85 L6 remains unreproduced due to wall_time_cap_exhausted
- missing discriminator: lp85 executable world-model rule coverage for wall_time_cap_exhausted
- candidate design: mine divergent active traces for the named residual, add transition tests, and count progress only through the offline reproduce() gate
- priority: high
<!-- exp4388-gap-e3-world-model-rule-lp85-l6-4383:end -->

<!-- exp4388-gap-e3-world-model-rule-tu93-l5-4383:start -->
### GAP-E3-WORLD-MODEL-RULE-TU93-L5-4383: Exp 4388 .405 verifier gap update
- status: open
- evidence: results/experiment_4383_e3_deeper_high_headroom_lookahead.json; game=tu93; offline_reproduced=False; prior_best_level=4; new_reproduced_level=4; target_level=5; verifier_accuracy=0.8; lookahead_fidelity=0.8; residual=wall_time_cap_exhausted.
- failure mode: tu93 L5 remains unreproduced due to wall_time_cap_exhausted
- missing discriminator: tu93 executable world-model rule coverage for wall_time_cap_exhausted
- candidate design: mine divergent active traces for the named residual, add transition tests, and count progress only through the offline reproduce() gate
- priority: high
<!-- exp4388-gap-e3-world-model-rule-tu93-l5-4383:end -->

<!-- exp4388-gap-e3-world-model-rule-tn36-l8-4383:start -->
### GAP-E3-WORLD-MODEL-RULE-TN36-L8-4383: Exp 4388 .405 verifier gap update
- status: open
- evidence: results/experiment_4383_e3_deeper_high_headroom_lookahead.json; game=tn36; offline_reproduced=False; prior_best_level=7; new_reproduced_level=7; target_level=8; verifier_accuracy=0.875; lookahead_fidelity=0.875; residual=tn36_l8_program_editor_object_control_gap_sxhtkytekm_palette_population.
- failure mode: tn36 L8 remains unreproduced due to tn36_l8_program_editor_object_control_gap_sxhtkytekm_palette_population
- missing discriminator: tn36 executable world-model rule coverage for tn36_l8_program_editor_object_control_gap_sxhtkytekm_palette_population
- candidate design: mine divergent active traces for the named residual, add transition tests, and count progress only through the offline reproduce() gate
- priority: high
<!-- exp4388-gap-e3-world-model-rule-tn36-l8-4383:end -->

<!-- exp4388-gap-e3-world-model-rule-tr87-l7-4383:start -->
### GAP-E3-WORLD-MODEL-RULE-TR87-L7-4383: Exp 4388 .405 verifier gap update
- status: open
- evidence: results/experiment_4383_e3_deeper_high_headroom_lookahead.json; game=tr87; offline_reproduced=False; prior_best_level=6; new_reproduced_level=6; target_level=7; verifier_accuracy=0.857143; lookahead_fidelity=0.857143; residual=wall_time_cap_exhausted.
- failure mode: tr87 L7 remains unreproduced due to wall_time_cap_exhausted
- missing discriminator: tr87 executable world-model rule coverage for wall_time_cap_exhausted
- candidate design: mine divergent active traces for the named residual, add transition tests, and count progress only through the offline reproduce() gate
- priority: high
<!-- exp4388-gap-e3-world-model-rule-tr87-l7-4383:end -->

<!-- exp4388-gap-e3-world-model-rule-ar25-l2-4384:start -->
### GAP-E3-WORLD-MODEL-RULE-AR25-L2-4384: Exp 4388 .405 verifier gap update
- status: open
- evidence: results/experiment_4384_e3_blocked_mechanic_tails_ar25_ka59_ft09.json; game=ar25; offline_reproduced=False; prior_best_level=1; new_reproduced_level=1; target_level=None; verifier_accuracy=0.90625; lookahead_fidelity=0.733333; residual=ar25_l2_action7_undo_stack_hidden_rule_gap.
- failure mode: ar25 L2 remains unreproduced due to ar25_l2_action7_undo_stack_hidden_rule_gap
- missing discriminator: ar25 executable world-model rule coverage for ar25_l2_action7_undo_stack_hidden_rule_gap
- candidate design: mine divergent active traces for the named residual, add transition tests, and count progress only through the offline reproduce() gate
- priority: high
<!-- exp4388-gap-e3-world-model-rule-ar25-l2-4384:end -->

<!-- exp4388-gap-e3-world-model-rule-ka59-l2-4384:start -->
### GAP-E3-WORLD-MODEL-RULE-KA59-L2-4384: Exp 4388 .405 verifier gap update
- status: open
- evidence: results/experiment_4384_e3_blocked_mechanic_tails_ar25_ka59_ft09.json; game=ka59; offline_reproduced=False; prior_best_level=1; new_reproduced_level=1; target_level=None; verifier_accuracy=0.125; lookahead_fidelity=0.112281; residual=ka59_l2_hidden_step_counter_hud_register_gap.
- failure mode: ka59 L2 remains unreproduced due to ka59_l2_hidden_step_counter_hud_register_gap
- missing discriminator: ka59 executable world-model rule coverage for ka59_l2_hidden_step_counter_hud_register_gap
- candidate design: mine divergent active traces for the named residual, add transition tests, and count progress only through the offline reproduce() gate
- priority: high
<!-- exp4388-gap-e3-world-model-rule-ka59-l2-4384:end -->

<!-- exp4388-gap-e3-world-model-rule-ft09-l2-4384:start -->
### GAP-E3-WORLD-MODEL-RULE-FT09-L2-4384: Exp 4388 .405 verifier gap update
- status: open
- evidence: results/experiment_4384_e3_blocked_mechanic_tails_ar25_ka59_ft09.json; game=ft09; offline_reproduced=False; prior_best_level=1; new_reproduced_level=1; target_level=None; verifier_accuracy=0.166667; lookahead_fidelity=0.347518; residual=ft09_l2_residual_world_model_mismatch_gap.
- failure mode: ft09 L2 remains unreproduced due to ft09_l2_residual_world_model_mismatch_gap
- missing discriminator: ft09 executable world-model rule coverage for ft09_l2_residual_world_model_mismatch_gap
- candidate design: mine divergent active traces for the named residual, add transition tests, and count progress only through the offline reproduce() gate
- priority: high
<!-- exp4388-gap-e3-world-model-rule-ft09-l2-4384:end -->

### GAP-4392-FIRST-ERROR-GAP-4-ARC-arc_candidate_process_proxy: Exp 4392 first-error residual
- status: open
- evidence: `results/experiment_4392_verifiable_process_data_localizer.json`; missed_first_error_traces=16 on GAP-4 ARC / arc_candidate_process_proxy.
- failure mode: the synthetic-trained earliest-error localizer ranked a downstream inheritor or proxy candidate ahead of the first error.
- missing discriminator: A domain feature that separates the first causal process break from later inherited or candidate-order artifacts.
- candidate design: Add typed domain-specific prefix checks and train a leave-domain-out contrastive earliest-error objective.
- priority: medium

### GAP-4393-LOCALIZER-POSITION-OR-TEMPLATE-CONFOUND: Exp 4393 localizer skeptic-proof residual
- status: open
- evidence: `results/experiment_4393_localizer_skeptic_proof.json`.
- confounder: position_only_baseline_ties_a1
- missing discriminator: A held-out real split with non-degenerate first-error positions.
- candidate design: Collect or construct REAL first-error traces with varied first-error positions and retrain the localizer with template-family holdouts.
- priority: high

<!-- exp4399-gap-fover-biprm-localization-untyped:start -->
### GAP-FOVER-BIPRM-LOCALIZATION-untyped: Exp 4399 .406 verifier gap update
- status: open
- evidence: results/experiment_4392_verifiable_process_data_localizer.json; localizer_beats_ensemble_baseline=True; results/experiment_4393_localizer_skeptic_proof.json; localizer_win_is_genuine=False; results/experiment_4396_localizer_self_learning_compounds.json; localizer_compounds=False; compounding_delta_ci95=[0.0, 0.0].
- failure mode: Exp 4392 solved the original FoVer split, but Exp 4393 quarantined the A1 headline unless position/template controls pass.
- missing discriminator: held-out real first-error labels with non-degenerate position/template variation
- candidate design: collect varied real first-error traces, type the residual error classes, and retrain with template-family holdouts before marking the gap filled
- priority: high
<!-- exp4399-gap-fover-biprm-localization-untyped:end -->

<!-- exp4399-gap-4392-first-error-gap-4-arc-arc_candidate_process_proxy:start -->
### GAP-4392-FIRST-ERROR-GAP-4-ARC-arc_candidate_process_proxy: Exp 4399 .406 verifier gap update
- status: open
- evidence: results/experiment_4392_verifiable_process_data_localizer.json; domain=GAP-4 ARC; error_class=arc_candidate_process_proxy; missed_first_error_traces=16.
- failure mode: synthetic-trained earliest-error localizer still ranks a later inheritor or ARC proxy artifact ahead of the first break
- missing discriminator: A domain feature that separates the first causal process break from later inherited or candidate-order artifacts.
- candidate design: Add typed domain-specific prefix checks and train a leave-domain-out contrastive earliest-error objective.
- priority: medium
<!-- exp4399-gap-4392-first-error-gap-4-arc-arc_candidate_process_proxy:end -->

<!-- exp4399-gap-4393-localizer-position-or-template-confound:start -->
### GAP-4393-LOCALIZER-POSITION-OR-TEMPLATE-CONFOUND: Exp 4399 .406 verifier gap update
- status: open
- evidence: results/experiment_4393_localizer_skeptic_proof.json; beats_position_only_baseline=False; template_ablation_drop=0.0; delta_ci95=[0.904, 0.904].
- failure mode: position/template controls quarantine the A1 localizer headline
- missing discriminator: real held-out split with varied first-error positions
- candidate design: construct real first-error traces with template-family holdouts and require a positive A1-vs-position and A1-vs-ablation CI
- priority: high
<!-- exp4399-gap-4393-localizer-position-or-template-confound:end -->

<!-- exp4399-gap-e3-world-model-rule-lp85-l6-4383:start -->
### GAP-E3-WORLD-MODEL-RULE-LP85-L6-4383: Exp 4399 .406 verifier gap update
- status: open
- evidence: results/experiment_4394_e3_deeper_fidelity_gate.json; game=lp85; offline_reproduced=False; target_level=6; new_reproduced_level=5; verifier_accuracy=0.833333; lookahead_fidelity=0.833333; residual=lookahead_fidelity_below_gate.
- failure mode: lp85 L6 remains unreproduced due to lookahead_fidelity_below_gate
- missing discriminator: lp85 executable world-model rule coverage for lookahead_fidelity_below_gate
- candidate design: mine divergent active traces for the named residual, add transition tests, and count progress only through the offline reproduce() gate
- priority: high
<!-- exp4399-gap-e3-world-model-rule-lp85-l6-4383:end -->

<!-- exp4399-gap-e3-world-model-rule-tu93-l5-4383:start -->
### GAP-E3-WORLD-MODEL-RULE-TU93-L5-4383: Exp 4399 .406 verifier gap update
- status: open
- evidence: results/experiment_4394_e3_deeper_fidelity_gate.json; game=tu93; offline_reproduced=False; target_level=5; new_reproduced_level=4; verifier_accuracy=0.8; lookahead_fidelity=0.8; residual=lookahead_fidelity_below_gate.
- failure mode: tu93 L5 remains unreproduced due to lookahead_fidelity_below_gate
- missing discriminator: tu93 executable world-model rule coverage for lookahead_fidelity_below_gate
- candidate design: mine divergent active traces for the named residual, add transition tests, and count progress only through the offline reproduce() gate
- priority: high
<!-- exp4399-gap-e3-world-model-rule-tu93-l5-4383:end -->

<!-- exp4399-gap-e3-world-model-rule-tn36-l8-4383:start -->
### GAP-E3-WORLD-MODEL-RULE-TN36-L8-4383: Exp 4399 .406 verifier gap update
- status: open
- evidence: results/experiment_4394_e3_deeper_fidelity_gate.json; game=tn36; offline_reproduced=False; target_level=8; new_reproduced_level=7; verifier_accuracy=0.875; lookahead_fidelity=0.875; residual=lookahead_fidelity_below_gate.
- failure mode: tn36 L8 remains unreproduced due to lookahead_fidelity_below_gate
- missing discriminator: tn36 executable world-model rule coverage for lookahead_fidelity_below_gate
- candidate design: mine divergent active traces for the named residual, add transition tests, and count progress only through the offline reproduce() gate
- priority: high
<!-- exp4399-gap-e3-world-model-rule-tn36-l8-4383:end -->

<!-- exp4399-gap-e3-world-model-rule-tr87-l7-4383:start -->
### GAP-E3-WORLD-MODEL-RULE-TR87-L7-4383: Exp 4399 .406 verifier gap update
- status: open
- evidence: results/experiment_4394_e3_deeper_fidelity_gate.json; game=tr87; offline_reproduced=False; target_level=7; new_reproduced_level=6; verifier_accuracy=0.857143; lookahead_fidelity=0.857143; residual=lookahead_fidelity_below_gate.
- failure mode: tr87 L7 remains unreproduced due to lookahead_fidelity_below_gate
- missing discriminator: tr87 executable world-model rule coverage for lookahead_fidelity_below_gate
- candidate design: mine divergent active traces for the named residual, add transition tests, and count progress only through the offline reproduce() gate
- priority: high
<!-- exp4399-gap-e3-world-model-rule-tr87-l7-4383:end -->

<!-- exp4399-gap-e3-world-model-rule-ar25-l2-4384:start -->
### GAP-E3-WORLD-MODEL-RULE-AR25-L2-4384: Exp 4399 .406 verifier gap update
- status: open
- evidence: results/experiment_4395_e3_blocked_mechanic_tails_ar25_ka59_ft09.json; game=ar25; offline_reproduced=False; target_level=2; new_reproduced_level=1; verifier_accuracy=0.90625; lookahead_fidelity=0.733333; residual=ar25_l2_action7_undo_stack_hidden_rule_gap.
- failure mode: ar25 L2 remains unreproduced due to ar25_l2_action7_undo_stack_hidden_rule_gap
- missing discriminator: ar25 executable world-model rule coverage for ar25_l2_action7_undo_stack_hidden_rule_gap
- candidate design: mine divergent active traces for the named residual, add transition tests, and count progress only through the offline reproduce() gate
- priority: high
<!-- exp4399-gap-e3-world-model-rule-ar25-l2-4384:end -->

<!-- exp4399-gap-e3-world-model-rule-ka59-l2-4384:start -->
### GAP-E3-WORLD-MODEL-RULE-KA59-L2-4384: Exp 4399 .406 verifier gap update
- status: open
- evidence: results/experiment_4395_e3_blocked_mechanic_tails_ar25_ka59_ft09.json; game=ka59; offline_reproduced=False; target_level=2; new_reproduced_level=1; verifier_accuracy=0.125; lookahead_fidelity=0.112281; residual=ka59_l2_object_relevance_step_counter_hud_register_gap.
- failure mode: ka59 L2 remains unreproduced due to ka59_l2_object_relevance_step_counter_hud_register_gap
- missing discriminator: ka59 executable world-model rule coverage for ka59_l2_object_relevance_step_counter_hud_register_gap
- candidate design: mine divergent active traces for the named residual, add transition tests, and count progress only through the offline reproduce() gate
- priority: high
<!-- exp4399-gap-e3-world-model-rule-ka59-l2-4384:end -->

<!-- exp4399-gap-e3-world-model-rule-ft09-l2-4384:start -->
### GAP-E3-WORLD-MODEL-RULE-FT09-L2-4384: Exp 4399 .406 verifier gap update
- status: open
- evidence: results/experiment_4395_e3_blocked_mechanic_tails_ar25_ka59_ft09.json; game=ft09; offline_reproduced=False; target_level=2; new_reproduced_level=1; verifier_accuracy=0.166667; lookahead_fidelity=0.347518; residual=ft09_l2_residual_world_model_mismatch_gap.
- failure mode: ft09 L2 remains unreproduced due to ft09_l2_residual_world_model_mismatch_gap
- missing discriminator: ft09 executable world-model rule coverage for ft09_l2_residual_world_model_mismatch_gap
- candidate design: mine divergent active traces for the named residual, add transition tests, and count progress only through the offline reproduce() gate
- priority: high
<!-- exp4399-gap-e3-world-model-rule-ft09-l2-4384:end -->

### GAP-4403-REAL-INTERVENTION-LOCALIZER-POSITION-ONLY: Exp 4403 real-intervention localizer residual
- status: open
- evidence: `results/experiment_4403_real_intervention_localizer_deconfound.json`.
- confounder: position_only_or_template_family_control_failed
- failure mode: position-only first-error localization ties or beats the localizer.
- missing discriminator: Real multi-step intervention labels with non-degenerate first-error positions and suffix redirects.
- candidate design: Collect typed multi-step FoVer interventions where correction at k is checked against a non-empty suffix, then re-run held-out-family position-only and template-family controls.
- priority: high

<!-- exp4405-gap-lp85-l6:start -->
### GAP-4405-LP85-L6: Exp 4405 mechanic-unit-test residual
- status: open
- evidence: results/experiment_4405_e3_deeper_mechanic_unit_tests.json; mechanic_unit_tests=1/1; offline_reproduced=False.
- residual failing mechanic: lp85_l6_permutation_bfs_reproduction_not_proven_after_unit_transition
- candidate design: extend the target solver only after the residual transition has an executable unit test and the offline reproduce() gate proves a new level.
- priority: high
<!-- exp4405-gap-lp85-l6:end -->

<!-- exp4405-gap-tu93-l5:start -->
### GAP-4405-TU93-L5: Exp 4405 mechanic-unit-test residual
- status: open
- evidence: results/experiment_4405_e3_deeper_mechanic_unit_tests.json; mechanic_unit_tests=1/1; offline_reproduced=False.
- residual failing mechanic: tu93_l5_fresh_env_branch_mode_reproduction_not_proven_after_unit_transition
- candidate design: extend the target solver only after the residual transition has an executable unit test and the offline reproduce() gate proves a new level.
- priority: high
<!-- exp4405-gap-tu93-l5:end -->

<!-- exp4405-gap-tn36-l8:start -->
### GAP-4405-TN36-L8: Exp 4405 mechanic-unit-test residual
- status: open
- evidence: results/experiment_4405_e3_deeper_mechanic_unit_tests.json; mechanic_unit_tests=1/1; offline_reproduced=False.
- residual failing mechanic: tn36_l8_sxhtkytekm_palette_population_reproduction_not_proven_after_unit_transition
- candidate design: extend the target solver only after the residual transition has an executable unit test and the offline reproduce() gate proves a new level.
- priority: high
<!-- exp4405-gap-tn36-l8:end -->

<!-- exp4405-gap-tr87-l7:start -->
### GAP-4405-TR87-L7: Exp 4405 mechanic-unit-test residual
- status: open
- evidence: results/experiment_4405_e3_deeper_mechanic_unit_tests.json; mechanic_unit_tests=1/1; offline_reproduced=False.
- residual failing mechanic: tr87_l7_no_offline_level_available_after_two_pass_rewrite_unit_transition
- candidate design: extend the target solver only after the residual transition has an executable unit test and the offline reproduce() gate proves a new level.
- priority: high
<!-- exp4405-gap-tr87-l7:end -->

<!-- exp4406-gap-ar25-l2:start -->
### GAP-4406-AR25-L2: Exp 4406 named-register residual
- status: open
- evidence: results/experiment_4406_e3_blocked_mechanic_tails_unit_tests.json; register_unit_tests=1/1; offline_reproduced=False.
- residual gap class: ar25_l2_action7_undo_stack_plan_not_reproduced_after_register_test
- candidate design: only plan/claim L2 after the register-level transition and a replayable offline reproduction gate both pass.
- priority: high
<!-- exp4406-gap-ar25-l2:end -->

<!-- exp4406-gap-ka59-l2:start -->
### GAP-4406-KA59-L2: Exp 4406 named-register residual
- status: open
- evidence: results/experiment_4406_e3_blocked_mechanic_tails_unit_tests.json; register_unit_tests=1/1; offline_reproduced=False.
- residual gap class: ka59_l2_object_relevance_step_counter_hud_register_gap
- candidate design: only plan/claim L2 after the register-level transition and a replayable offline reproduction gate both pass.
- priority: high
<!-- exp4406-gap-ka59-l2:end -->

<!-- exp4406-gap-ft09-l2:start -->
### GAP-4406-FT09-L2: Exp 4406 named-register residual
- status: open
- evidence: results/experiment_4406_e3_blocked_mechanic_tails_unit_tests.json; register_unit_tests=1/1; offline_reproduced=False.
- residual gap class: ft09_l2_residual_world_model_mismatch_gap_after_component_transition
- candidate design: only plan/claim L2 after the register-level transition and a replayable offline reproduction gate both pass.
- priority: high
<!-- exp4406-gap-ft09-l2:end -->

### GAP-4407-ACTIVE-LOCALIZER-POSITION-BOUND: Exp 4407 active localizer residual
- status: open
- evidence: `results/experiment_4407_active_learning_self_learning_compounds.json`.
- failure mode: complete: clean_null_position_bound_or_saturated
- missing discriminator: Non-degenerate multi-position real intervention labels with content features that beat a position-only first-error baseline.
- candidate design: Collect multi-step FoVer interventions with non-empty suffix redirects and typed first-error families before retrying active trace selection.
- priority: high

### GAP-4408-CODE-HUMANEVAL-DECONFOUNDED-DETECTOR-CHANCE
- status: open
- domain: code_humaneval
- failure_mode: Deconfounded detection AUROC CI95 includes chance on code_humaneval after SCA; n=539.
- missing_discriminator: A domain-native oracle-distinct verifier feature that separates semantically grouped correct answers from plausible wrong answers.
- candidate_design: Add a verifier score that targets the residual wrong mode, then rerun Exp 4408 with the same SCA and LODO calibration gate.
- priority: high

<!-- exp4415-gap-ar25-l2:start -->
### GAP-4415-AR25-L2: Exp 4415 adaptive residual behavior
- status: open
- evidence: results/experiment_4415_agent2world_adaptive_e3_repair.json; adaptive_tests=1/2; offline_reproduced=False.
- residual failing behavior: ar25_l2_hidden_undo_stack_state_not_visible_in_rollout
- leakage control: held-out mechanic test and fresh-agent state are reported separately from the solve claim.
- priority: high
<!-- exp4415-gap-ar25-l2:end -->

<!-- exp4415-gap-tn36-l8:start -->
### GAP-4415-TN36-L8: Exp 4415 adaptive residual behavior
- status: open
- evidence: results/experiment_4415_agent2world_adaptive_e3_repair.json; adaptive_tests=1/2; offline_reproduced=False.
- residual failing behavior: tn36_l8_palette_population_or_later_program_state_still_wrong
- leakage control: held-out mechanic test and fresh-agent state are reported separately from the solve claim.
- priority: high
<!-- exp4415-gap-tn36-l8:end -->

<!-- exp4415-gap-lp85-l6:start -->
### GAP-4415-LP85-L6: Exp 4415 adaptive residual behavior
- status: open
- evidence: results/experiment_4415_agent2world_adaptive_e3_repair.json; adaptive_tests=1/2; offline_reproduced=False.
- residual failing behavior: lp85_l6_button_permutation_search_reproduction_still_wrong
- leakage control: held-out mechanic test and fresh-agent state are reported separately from the solve claim.
- priority: high
<!-- exp4415-gap-lp85-l6:end -->

### GAP-4419-CODE-HUMANEVAL-STEERCONF-DETECTOR-CHANCE
- status: open
- domain: code_humaneval
- failure_mode: SteerConf cached-feature detection AUROC CI95 includes chance on code_humaneval; n=539, baseline_auroc=0.577374, steered_delta=0.024536.
- missing_discriminator: A domain-native verifier feature beyond self-reported or cached-feature confidence consistency that separates correct outputs from plausible wrong outputs.
- candidate_design: Build a domain-specific oracle-distinct verifier feature, then rerun Exp 4419's same cached-pool SteerConf and LODO calibration gate.
- priority: high

<!-- exp4421-gap-4414-ka59-config-rule-grounded:start -->
### GAP-4414-KA59-CONFIG-RULE-GROUNDED: Exp 4421 .408 verifier gap update
- status: filled (ka59_config_win_rule_predicate)
- evidence: results/experiment_4414_config_rule_induction_solve.json; predicate=editable_count_4_equals_reference_count_4_32; tier=2; false_positive_rate=0.0.
- failure mode: ka59 had an ungrounded config-game win-rule predicate
- missing discriminator: grounded config win rule checked against the offline state
- candidate design: registry-backed config-rule predicate reused for future ka59 targets
- priority: medium
- movement: filled
<!-- exp4421-gap-4414-ka59-config-rule-grounded:end -->

<!-- exp4421-gap-4414-config-rule-induction-bp35:start -->
### GAP-4414-CONFIG-RULE-INDUCTION-BP35: Exp 4421 .408 verifier gap update
- status: open
- evidence: results/experiment_4414_config_rule_induction_solve.json; game=bp35; search_blocker=blocked_local_model_unavailable.
- failure mode: fresh config-rule induction did not run or did not reproduce a new level
- missing discriminator: grounded reusable config win-rule for bp35
- candidate design: run local symbolic/config-rule induction once the local proposer is available
- priority: high
- movement: newly_logged
<!-- exp4421-gap-4414-config-rule-induction-bp35:end -->

<!-- exp4421-gap-4414-config-rule-induction-dc22:start -->
### GAP-4414-CONFIG-RULE-INDUCTION-DC22: Exp 4421 .408 verifier gap update
- status: open
- evidence: results/experiment_4414_config_rule_induction_solve.json; game=dc22; search_blocker=blocked_local_model_unavailable.
- failure mode: fresh config-rule induction did not run or did not reproduce a new level
- missing discriminator: grounded reusable config win-rule for dc22
- candidate design: run local symbolic/config-rule induction once the local proposer is available
- priority: high
- movement: newly_logged
<!-- exp4421-gap-4414-config-rule-induction-dc22:end -->

<!-- exp4421-gap-4415-adaptive-e3-ar25-l2:start -->
### GAP-4415-ADAPTIVE-E3-AR25-L2: Exp 4421 .408 verifier gap update
- status: open
- evidence: results/experiment_4415_agent2world_adaptive_e3_repair.json; game=ar25; target_level=2; adaptive_tests=1/2; verifier_accuracy=0.5; lookahead_fidelity=0.5; residual=ar25_l2_hidden_undo_stack_state_not_visible_in_rollout.
- failure mode: ar25 L2 remains unreproduced after adaptive E3 repair
- missing discriminator: state-grounded executable rule for ar25_l2_hidden_undo_stack_state_not_visible_in_rollout
- candidate design: convert the residual behavior test into an offline reproduce() plan
- priority: high
- movement: newly_logged
<!-- exp4421-gap-4415-adaptive-e3-ar25-l2:end -->

<!-- exp4421-gap-4415-adaptive-e3-tn36-l8:start -->
### GAP-4415-ADAPTIVE-E3-TN36-L8: Exp 4421 .408 verifier gap update
- status: open
- evidence: results/experiment_4415_agent2world_adaptive_e3_repair.json; game=tn36; target_level=8; adaptive_tests=1/2; verifier_accuracy=0.875; lookahead_fidelity=0.875; residual=tn36_l8_palette_population_or_later_program_state_still_wrong.
- failure mode: tn36 L8 remains unreproduced after adaptive E3 repair
- missing discriminator: state-grounded executable rule for tn36_l8_palette_population_or_later_program_state_still_wrong
- candidate design: convert the residual behavior test into an offline reproduce() plan
- priority: high
- movement: newly_logged
<!-- exp4421-gap-4415-adaptive-e3-tn36-l8:end -->

<!-- exp4421-gap-4415-adaptive-e3-lp85-l6:start -->
### GAP-4415-ADAPTIVE-E3-LP85-L6: Exp 4421 .408 verifier gap update
- status: open
- evidence: results/experiment_4415_agent2world_adaptive_e3_repair.json; game=lp85; target_level=6; adaptive_tests=1/2; verifier_accuracy=0.833333; lookahead_fidelity=0.833333; residual=lp85_l6_button_permutation_search_reproduction_still_wrong.
- failure mode: lp85 L6 remains unreproduced after adaptive E3 repair
- missing discriminator: state-grounded executable rule for lp85_l6_button_permutation_search_reproduction_still_wrong
- candidate design: convert the residual behavior test into an offline reproduce() plan
- priority: high
- movement: newly_logged
<!-- exp4421-gap-4415-adaptive-e3-lp85-l6:end -->

<!-- exp4421-gap-fover-biprm-localization-untyped:start -->
### GAP-FOVER-BIPRM-LOCALIZATION-untyped: Exp 4421 .408 verifier gap update
- status: open (sharpened by exp4416 hidden-state null)
- evidence: results/experiment_4416_hidden_state_localizer_falsification_audit.json; hidden_state_localizer_has_nonposition_signal=False; position_only_baseline_f1=1.0.
- failure mode: hidden-state localizer tied the content-blind position baseline
- missing discriminator: non-position earliest causal error signal under non-degenerate traces
- candidate design: collect typed multi-step first-error traces before reviving localization
- priority: medium
- movement: sharpened
<!-- exp4421-gap-fover-biprm-localization-untyped:end -->

<!-- exp4421-gap-fover-hidden-state-localization-position-saturated:start -->
### GAP-FOVER-HIDDEN-STATE-LOCALIZATION-POSITION-SATURATED: Exp 4421 .408 verifier gap update
- status: open
- evidence: results/experiment_4416_hidden_state_localizer_falsification_audit.json; hidden-state localizer null.
- failure mode: The hidden-state transport/margin probe tied the content-blind position-only first-error baseline; the available FoVer failed traces are position-saturated at first step.
- missing discriminator: A model-native signal that separates earliest causal error from position and downstream consequence under non-degenerate multi-step FoVer labels.
- candidate design: Collect typed multi-step FoVer traces with non-first-position first errors before any localizer redeployment; do not revive the position-saturated text or hidden-state localizer line.
- priority: medium
- movement: sharpened
<!-- exp4421-gap-fover-hidden-state-localization-position-saturated:end -->

<!-- exp4421-gap-4417-sovereign-gap4-local-generator-zero-fires:start -->
### GAP-4417-SOVEREIGN-GAP4-LOCAL-GENERATOR-ZERO-FIRES: Exp 4421 .408 verifier gap update
- status: open
- evidence: results/experiment_4417_gap4_local_generator_sovereign_arm.json; sovereign_gap4_gate_holds=True; graded_gate_fires=0; gated_pass2=0.4516; vote_pass2=0.4516.
- failure mode: local-generator sovereign arm holds the safety gate but fires zero graded wins
- missing discriminator: local open-weight generator proposal that creates verifier-actionable GAP-4 candidates
- candidate design: separate reusable symbolic induction from another static local-generator replay
- priority: high
- movement: newly_logged
<!-- exp4421-gap-4417-sovereign-gap4-local-generator-zero-fires:end -->

<!-- exp4421-gap-4418-config-rule-vocabulary-local-model-unavailable:start -->
### GAP-4418-CONFIG-RULE-VOCABULARY-LOCAL-MODEL-UNAVAILABLE: Exp 4421 .408 verifier gap update
- status: open
- evidence: results/experiment_4418_config_rule_vocabulary_transfer.json; transfers=False; local_model_status=blocked_local_model_unavailable.
- failure mode: config-rule vocabulary transfer was blocked by local model unavailability
- missing discriminator: local config-rule proposer that transfers grounded vocabulary to unsolved games
- candidate design: cache or start the declared local iGPU proposer and rerun vocabulary transfer
- priority: high
- movement: newly_logged
<!-- exp4421-gap-4418-config-rule-vocabulary-local-model-unavailable:end -->

<!-- exp4421-gap-4419-code-humaneval-steerconf-detector-chance:start -->
### GAP-4419-CODE-HUMANEVAL-STEERCONF-DETECTOR-CHANCE: Exp 4421 .408 verifier gap update
- status: open
- evidence: results/experiment_4419_steerconf_code_detection_calibration_repair.json; SteerConf code detection chance.
- failure mode: SteerConf cached-feature detection AUROC CI95 includes chance on code_humaneval; n=539, baseline_auroc=0.577374, steered_delta=0.024536.
- missing discriminator: A domain-native verifier feature beyond self-reported or cached-feature confidence consistency that separates correct outputs from plausible wrong outputs.
- candidate design: Build a domain-specific oracle-distinct verifier feature, then rerun Exp 4419's same cached-pool SteerConf and LODO calibration gate.
- priority: high
- movement: newly_logged
<!-- exp4421-gap-4419-code-humaneval-steerconf-detector-chance:end -->

<!-- exp4427-gap-4421-s5i5-marker-coverage:start -->
### GAP-4421-S5I5-MARKER-COVERAGE: Exp 4427 .409 verifier gap hygiene
- status: filled (exp4421_s5i5_marker_coverage)
- evidence: results/experiment_4421_config_rule_solve_unseen.json; offline_reproduced=True; reproduced_levels=1; new_levels_reproduced=1
- failure mode: s5i5 marker-coverage config rule is now grounded and offline reproduced
- missing discriminator: grounded marker-coverage predicate
- candidate design: reuse Exp 4421 marker-coverage verifier for related marker-toggle games
- priority: medium
- headroom: 0
- build target for .410 planner: false
- movement: filled
<!-- exp4427-gap-4421-s5i5-marker-coverage:end -->

<!-- exp4427-gap-4422-tr87-glyph-rewrite-perception:start -->
### GAP-4422-TR87-GLYPH-REWRITE-PERCEPTION: Exp 4427 .409 verifier gap hygiene
- status: filled (exp4422_tr87_glyph_rewrite_perception)
- evidence: results/experiment_4422_glyph_rewrite_perception.json; offline_reproduced=True; reproduced_levels=6; fires_on_win=True; false_positive_rate=0.0
- failure mode: tr87 glyph rewrite perception is now grounded through L6 replay
- missing discriminator: glyph rewrite perception predicate
- candidate design: reuse segmented glyph rewrite predicates for future rewrite games
- priority: medium
- headroom: 0
- build target for .410 planner: false
- movement: filled
<!-- exp4427-gap-4422-tr87-glyph-rewrite-perception:end -->

<!-- exp4427-gap-4423-g50t-unselectable-first-contact:start -->
### GAP-4423-G50T-UNSELECTABLE-FIRST-CONTACT: Exp 4449 .411 registry gap hygiene
- status: filled (exp4443_bank_g50t_example_conditioned_win)
- evidence: results/experiment_4443_bank_g50t_example_conditioned_win.json; target_game=g50t; offline_reproduced=True; reproduced_levels=1; reproducible_total_levels=38
- failure mode: prior first-contact route could not select the winning target-offset predicate
- missing discriminator: filled by execution-grounded target-offset verifier
- candidate design: keep the target-offset config-rule verifier in the generic bank
- priority: high
- source artifact: results/experiment_4443_bank_g50t_example_conditioned_win.json
- movement: filled
<!-- exp4427-gap-4423-g50t-unselectable-first-contact:end -->

<!-- exp4427-gap-4424-sc25-l2-route-search:start -->
### GAP-4424-SC25-L2-ROUTE-SEARCH: Exp 4427 .409 verifier gap hygiene
- status: open
- evidence: results/experiment_4424_deeper_solved_game.json; game=sc25; target_level=2; offline_reproduced=False; reproduced_levels=1; residual=sc25_l2_route_search_still_missing_after_hud_cleanup
- failure mode: sc25 L2 remains unreproduced after .409 HUD/mechanic cleanup
- missing discriminator: sc25 route-search verifier that proves the complete L2 path after the recorded mechanic cleanup
- candidate design: build an executable route-search verifier over the sc25 world model and count only arc_solver_kit.reproduce success
- priority: medium
- headroom: 1
- build target for .410 planner: false
- movement: newly_logged
<!-- exp4427-gap-4424-sc25-l2-route-search:end -->

<!-- exp4438-gap-4432-loo-tr87-missing-glyph-rewrite-rule-verifier-without-tr87-adapter:start -->
### GAP-4432-LOO-TR87-MISSING-GLYPH-REWRITE-RULE-VERIFIER-WITHOUT-TR87-ADAPTER: Exp 4461 .412 registry gap hygiene
- status: filled (exp4456_generic_glyph_rewrite_operator)
- evidence: results/experiment_4456_generic_glyph_rewrite_operator.json; tr87_resolved_generically=True; offline_reproduced=True; tr87_generic_level_reproduced=1
- failure mode: prior missing_glyph_rewrite_rule_verifier residual is closed for tr87
- missing discriminator: filled by generic glyph_rewrite_rule_verifier
- candidate design: keep the glyph rewrite verifier in the generic leave-one-out loop
- priority: high
- source artifact: results/experiment_4456_generic_glyph_rewrite_operator.json
- movement: filled
<!-- exp4438-gap-4432-loo-tr87-missing-glyph-rewrite-rule-verifier-without-tr87-adapter:end -->

<!-- exp4438-gap-4432-loo-sc25-missing-cast-grid-spell-shrink-tank-exit-verifier:start -->
### GAP-4432-LOO-SC25-MISSING-CAST-GRID-SPELL-SHRINK-TANK-EXIT-VERIFIER: Exp 4474 .413 registry gap hygiene
- status: filled (experiment_4469_generic_cast_grid_fsm_operator)
- evidence: results/experiment_4469_generic_cast_grid_fsm_operator.json; sc25_resolved_generically=True; sc25_generic_level_reproduced=1; offline_reproduced=True
- failure mode: closed_by_cast_grid_phase_fsm_world_model
- missing discriminator: filled by execution-grounded cast_grid_phase_fsm_world_model
- candidate design: reuse two-phase cast/config toggle then navigation FSMs for future cast-grid games
- priority: high
- source artifact: results/experiment_4469_generic_cast_grid_fsm_operator.json
- movement: filled
<!-- exp4438-gap-4432-loo-sc25-missing-cast-grid-spell-shrink-tank-exit-verifier:end -->

<!-- exp4438-gap-4432-loo-ka59-missing-push-block-world-model-and-dynamic-selection:start -->
### GAP-4432-LOO-KA59-MISSING-PUSH-BLOCK-WORLD-MODEL-AND-DYNAMIC-SELECTION: Exp 4449 .411 registry gap hygiene
- status: filled (exp4445_generic_object_motion_world_model_operator)
- evidence: results/experiment_4445_generic_object_motion_world_model_operator.json; residuals_closed_generically includes ka59; offline_reproduced=True; reproduced_levels=2; operator=object_motion_world_model; target_recipe_withheld=ka59; world_model_accuracy_with_examples=1.0; world_model_accuracy_cold=0.25
- failure mode: missing_push_block_world_model_and_dynamic_selection
- missing discriminator: filled by generic object-slot translate/push transition model with dynamic selection
- candidate design: keep the generic operator in the standing loop
- priority: high
- source artifact: results/experiment_4445_generic_object_motion_world_model_operator.json
- movement: filled
<!-- exp4438-gap-4432-loo-ka59-missing-push-block-world-model-and-dynamic-selection:end -->

<!-- exp4438-gap-4432-loo-ar25-missing-reflection-world-model-and-object-motion-plan:start -->
### GAP-4432-LOO-AR25-MISSING-REFLECTION-WORLD-MODEL-AND-OBJECT-MOTION-PLAN: Exp 4449 .411 registry gap hygiene
- status: filled (exp4445_generic_object_motion_world_model_operator)
- evidence: results/experiment_4445_generic_object_motion_world_model_operator.json; residuals_closed_generically includes ar25; offline_reproduced=True; reproduced_levels=2; operator=object_motion_world_model; target_recipe_withheld=ar25; world_model_accuracy_with_examples=1.0; world_model_accuracy_cold=0.25
- failure mode: missing_reflection_world_model_and_object_motion_plan
- missing discriminator: filled by generic object-slot translate/reflect transition model
- candidate design: keep the generic operator in the standing loop
- priority: high
- source artifact: results/experiment_4445_generic_object_motion_world_model_operator.json
- movement: filled
<!-- exp4438-gap-4432-loo-ar25-missing-reflection-world-model-and-object-motion-plan:end -->

<!-- exp4438-gap-4432-loo-ft09-missing-local-constraint-color-cycle-verifier:start -->
### GAP-4432-LOO-FT09-MISSING-LOCAL-CONSTRAINT-COLOR-CYCLE-VERIFIER: Exp 4449 .411 registry gap hygiene
- status: filled (exp4444_generic_config_rule_verifier_operator)
- evidence: results/experiment_4444_generic_config_rule_verifier_operator.json; ft09_resolved_generically=True; offline_reproduced=True; reproduced_levels=1; operator=config_rule_verifier; target_recipe_withheld=ft09
- failure mode: prior missing_local_constraint_color_cycle_verifier residual is closed for ft09 L1
- missing discriminator: filled by generic execution-grounded local_constraint_color_cycle verifier
- candidate design: reuse config_rule_verifier for future local-constraint/toggle digests
- priority: high
- source artifact: results/experiment_4444_generic_config_rule_verifier_operator.json
- movement: filled
<!-- exp4438-gap-4432-loo-ft09-missing-local-constraint-color-cycle-verifier:end -->

<!-- exp4438-gap-4423-dc22-unselectable-first-contact:start -->
### GAP-4423-DC22-UNSELECTABLE-FIRST-CONTACT: Exp 4474 .413 registry gap hygiene
- status: filled (experiment_4467_solve_dc22_cegis_nocov)
- evidence: results/experiment_4467_solve_dc22_cegis_nocov.json; target_game=dc22; offline_reproduced=True; dc22_grounded=True; reproduced_levels=1
- failure mode: closed_by_dc22_cegis_config_rule
- missing discriminator: filled by execution-grounded buezna toggle plus jfva->goknoi navigation predicate
- candidate design: keep dc22_toggle_navigation in config_rule_verifier and the dc22 GameAdapter
- priority: high
- source artifact: results/experiment_4467_solve_dc22_cegis_nocov.json
- movement: filled
<!-- exp4438-gap-4423-dc22-unselectable-first-contact:end -->

<!-- exp4446-gap-4423-vc33-unselectable-first-contact:start -->
### GAP-4423-VC33-UNSELECTABLE-FIRST-CONTACT: Exp 4449 .411 registry gap hygiene
- status: filled (exp4446_drive_generic_first_contact_bank)
- evidence: results/experiment_4446_drive_generic_first_contact_bank.json; target_game=vc33; routed_to=s5i5; offline_reproduced=True; reproduced_levels=1
- failure mode: closed_by_support_clearance_config_rule
- missing discriminator: filled by generic config_rule_verifier support-clearance digest
- candidate design: reuse routed config-rule support-clearance predicates
- priority: high
- source artifact: results/experiment_4446_drive_generic_first_contact_bank.json
- movement: filled
<!-- exp4446-gap-4423-vc33-unselectable-first-contact:end -->

<!-- exp4458-gap-sb26-color-match-slot-sequence:start -->
### GAP-4458-SB26-COLOR-MATCH-SLOT-SEQUENCE: Exp 4474 .413 registry gap hygiene
- status: filled (experiment_4470_color_match_slot_operator_solve_sb26)
- evidence: results/experiment_4470_color_match_slot_operator_solve_sb26.json; color_match_operator_built=True; offline_reproduced=True; reproduced_levels=1; counterexample_rounds=2
- failure mode: closed_by_color_match_slot_sequence_verifier
- missing discriminator: filled by execution-grounded ordered color-match item-slot verifier with undo-aware grounding
- candidate design: reuse color_match_slot_sequence_verifier for ordered item-slot color puzzles
- priority: high
- source artifact: results/experiment_4470_color_match_slot_operator_solve_sb26.json
- movement: filled
<!-- exp4458-gap-sb26-color-match-slot-sequence:end -->

<!-- exp4469-gap-sc25-cast-grid:start -->
### GAP-4432-LOO-SC25-MISSING-CAST-GRID-SPELL-SHRINK-TANK-EXIT-VERIFIER: Exp 4469 generic cast-grid FSM
- status: filled
- evidence: results/experiment_4469_generic_cast_grid_fsm_operator.json; sc25_resolved_generically=True; sc25_generic_level_reproduced=1; offline_reproduced=True
- failure mode: closed_by_cast_grid_phase_fsm_world_model
- missing discriminator: filled by execution-grounded cast_grid_phase_fsm_world_model
- candidate design: reuse two-phase cast/config toggle then navigation FSMs for future cast-grid games
- priority: high
<!-- exp4469-gap-sc25-cast-grid:end -->

<!-- exp4471-gap-re86-pattern-match-sprite-resize:start -->
### GAP-4471-RE86-MISSING-PATTERN-MATCH-SPRITE-RESIZE-VERIFIER: Exp 4479 sprite overlay solve
- status: filled
- evidence: results/experiment_4479_solve_re86.json; target_game=re86; operator=sprite_overlay_resize_verifier; offline_reproduced=True; reproduced_levels=1
- failure mode: closed_by_sprite_overlay_resize_verifier
- missing discriminator: filled by generic sprite-overlay pattern-match and resize verifier
- candidate design: reuse exact overlay coverage plus explicit resize variants for future games
- priority: high
- movement: filled
<!-- exp4471-gap-re86-pattern-match-sprite-resize:end -->

---

## Architecture/discovery gaps from the 2026-06-19 step-back audit (parallel codebase mine)

These came from a systematic gap audit (not a single experiment), grounded in code+artifacts. They are
the *structural* verifier/solver gaps behind the 0.08 first live score. Several are inter-dependent.

### GAP-ARCH-VERIFIER-REGRESSION-ONLY: no discriminative win/loss classifier (PARTIALLY ADDRESSED 2026-06-19)
- status: building (per-game lever shipped; cross-game blocked on GAP-ARCH-FEATURES)
- evidence: `arc_value_learner.py:98-104` (LearnedVerifier = lstsq steps-to-go regression); `arc_value_net.py:1-7`
  docstring admits the linear head is "actively misleading when given control". NEW: built
  `DiscriminativeVerifier` + off-path-negatives (results/arc_discriminative_verifier.json) — in-sample AUROC
  0.726 but **leave-one-game-out AUROC 0.503 == chance**.
- failure mode: a distance-along-the-gold-path value cannot tell an off-path TRAP from a near-win (identical
  marginal features + value). A classifier separates them PER-GAME (0.726) but does NOT transfer cross-game.
- missing discriminator: a win-reachability classifier — usable when trained ONLINE per game during
  exploration (the off-path negatives the solver already produces); a cross-game pre-trained head is not.
- candidate design: SHIPPED DiscriminativeVerifier (logistic, off-path negatives). .414 A-task: wire the
  per-game online-trained head into the live explorer to prune traps the value head misses.
- priority: high

### GAP-ARCH-FEATURES: frame-only order-1 features (the real cross-game blocker)
- status: open
- evidence: `arc_value_learner.py:26-70` (5 scalars + 6x6 occupancy, all single-frame marginals). The
  LOO-chance result above + the value-head's inertness (value_weight=0.0) both trace here.
- failure mode: linear-over-marginals provably can't represent relational/XOR/counting win-rules; one-action-
  off-path states are near-identical to on-path in these features (mean feature gap 1.73 over standardized);
  nothing transfers cross-game.
- missing discriminator: object-relational (pairwise correspondence + manhattan displacement), frame-DELTA
  (Δ between consecutive frames = progress signal), action-conditioned (v(frame|action)), symmetry-invariant,
  and predicate-distance (tied to the adapter win condition) features.
- candidate design: add the above feature classes to cross_game_features_v3; re-run the LOO-AUROC gate (the
  harness exists: `arc_cross_game_verifier_train.py --discriminative`). THIS is the highest-leverage verifier
  research item — it unblocks both the value head and the discriminative head cross-game.
- priority: high

### GAP-ARCH-DEADENDS-AS-PROSE: recorded dead-ends used as LLM prose, never as negative labels (ADDRESSED 2026-06-19)
- status: building (off-path negatives shipped; true game-over negatives still unused)
- evidence: `arc_solve_learning.py:121-147` (dead-ends -> prose briefing); `arc_competition_agent.py:172-187`
  (game-over frame skipped, never emitted as a negative pair).
- failure mode: a free, growing corpus of true negatives the verifier never sees.
- missing discriminator: emit registry/explorer game-over states as (features, 0) negatives (stronger than the
  one-action-off-path negatives already shipped, which the LOO result shows are weak).
- candidate design: persist explorer game-over frames during solve; feed to DiscriminativeVerifier.
- priority: medium

### GAP-ARCH-GOAL-NOT-VERIFIED: world-model verifier scores DYNAMICS but never the GOAL predicate
- status: open
- evidence: `arc_executable_world_model.py:155-184` (WorldModelVerifier.score = grid-transition accuracy only);
  `is_level_complete` is loaded (line 198) but graded by nothing; refactor loop feeds back only transition
  mismatches. exp4020 induces `is_goal` at held-out precision 1.0 but is NOT wired into the E3 verifier.
- failure mode: a 99%-dynamics model with a wrong win-predicate is "trusted" -> the planner plans confidently
  toward the wrong win state. The deep-research-named goal-vs-dynamics gap.
- missing discriminator: held-out scoring of `is_level_complete` against recorded level-up transitions, as a
  first-class refactor signal separate from dynamics.
- candidate design: wire exp4020's goal-induction-and-verification into the E3 path.
- priority: high

### GAP-ARCH-GRID-ONLY-STATE: E3 state is grid-only; hidden HUD registers unrepresentable (deepening-tail root cause)
- status: open
- evidence: induce prompt fixes state as "HxW integer grid" (`arc_executable_world_model.py:276-292`); the
  L2-stall artifacts are register failures: ka59 `hud_count` diverges (exp4384, fidelity 0.112), ar25
  `action7_undo_stack` hidden (exp4395, fidelity 0.733), ft09 residual mismatch.
- failure mode: the lookahead-fidelity gate (0.73-0.875) is a SYMPTOM — the model predicts win-relevant
  behavior from a state that physically omits the deciding variable. ar25/ka59/ft09 stall at L2 on state
  representation, NOT search/depth.
- missing discriminator: extend E3 state to (grid, registers) with induced HUD/counter scalars; let
  is_level_complete read them.
- priority: high

### GAP-ARCH-NO-HIERARCHICAL-SEARCH: no subgoal/landmark/MCTS engine wired (deep-research's "single biggest lever")
- status: DOWN-WEIGHTED (2026-07-20, REQ-ARC-FCP-5757 empirical basis) -- was "open (single biggest lever)"
- down-weight basis (REQ-ARC-FCP-5757 candidate-coverage attribution,
  `results/experiment_5757_candidate_coverage_attribution.json`): a structural attribution of all 92
  known-winning-path progress actions across the 9 stalled adaptered games, classifying each vs the live
  `rich_action_candidates` generator, ran the FAIR pre-registered fire-once test of the search/lookahead lever
  (bucket b = "in-set but not frame-changing in isolation, on a known level-up path"). Result: **bucket b == 0**
  -- NO winning-path action is an in-set no-op-in-isolation that only pays off downstream, so there is ZERO
  evidence a lookahead planner would recover an action a greedy ranking cannot. This is the direct empirical
  confirmation of the 2026-07-20 top-project search-architecture audit's §3 finding: all three Milestone-1
  winners (Duck 1st/Reki 2nd/forge 3rd) are greedy single-commit generators with NO tree/beam/MCTS search, and
  Carnot already has strictly MORE search machinery (StepwiseExplorer graph search + `plan_in_model` lookahead +
  the vc33 hierarchical prototype). The gap is upstream of search.
- what the attribution found the gap IS instead: single-action candidate coverage is 98.9% exact / 100%
  tolerant (generation is not the single-action gap either); the only single-action gap is SELECTION/RANKING of
  object clicks (all 6 gap actions are action-6 clicks in-set + frame-changing but ranked >= 12: r11l x1, su15
  x4, + one cn04 <=2px near-miss). ~93% of winning-path actions are already individually handled yet the games
  stall, so the binding constraint is SEQUENCE-level routing without a goal signal -> the world-model INDUCTION
  gap (REQ-ARC-WMTE-5724, induce 0/12), NOT search depth.
- ORIGINAL evidence (preserved, never-prune): production search is flat BFS / weighted-A*
  (`arc_graph_explore.py:239-307`, OfflineSolver best-first); hierarchical search exists only as the single-game
  `arc_vc33_hierarchical_search.py` (vc33 still L1). `search-layer-literature-2026-06-11.md:42` named subgoal
  decomposition as the single biggest lever -- a claim NOT supported by the winners' evidence or by 5757's b==0.
- failure mode: no within-level subgoal decomposition; combinatorial config spaces (the OOD-dominant class)
  are intractable to flat search.
- candidate design (deferred, not retired): promote the vc33 hierarchical best-first into a generic,
  router-selectable engine for `is_spatial_planning:true` games -- but only AFTER world-model induction improves
  enough to give that planner a correct model + goal predicate to plan over (a lookahead planner is only as good
  as the induced model it consumes; that induction is the true bottleneck, per 5757 + the 0/12 induce nulls).
- priority: LOW (was medium) -- re-ordered below induction/perception per REQ-ARC-FCP-5757. Re-opens ONLY if a
  future well-powered measurement finds bucket b > 0.3 (5757's b branch can fire once on evidence).

### GAP-LIVE-INTEGRATION: the SUBMITTED agent runs a weaker generic path than the repo's own research (HIGHEST score lever)
- status: re-scoped (2026-07-02; stale wiring/config evidence corrected, residual provenance-mirage audit remains)
- evidence: Original 2026-06-19 evidence, now stale in part: `make_carnot_agent -> E3AgentPolicy -> StepwiseExplorer`
  was described as bare BFS (measured 8/32 in-distribution, ~0 OOD — `results/arc_offline_to_live_bridge_v2.json:5,18`)
  + an LLM tier with 0/6 measured value-added (:13); `target_levels=1`; `value_weight=0.0`; and
  `arc_strategy_router.py` / `arc_world_model_dsl.py` NOT imported by `arc_competition_agent.py`.
  2026-07-02 correction: current source refutes the three narrow wiring claims. `arc_competition_agent.py:30`
  imports `arc_strategy_router`, `:51` imports `ObjectDeltaModel`, `:2157-2164` uses both in `E3AgentPolicy`;
  `SUBMITTED_TARGET_LEVELS = 3` at `:88` and `SUBMITTED_AGENT_CONFIG["target_levels"]` reads it at `:3132`;
  `SUBMITTED_VALUE_WEIGHT = 1e-12` at `:83` and `SUBMITTED_AGENT_CONFIG["value_weight"]` reads it at `:3131`.
  Exp4605 records the submitted config with `target_levels=3`, router/DSL wired, and `value_weight_used=1e-12`.
  Exp4652 records a matched `value_weight=0.0` baseline plus a cost-fixed nonzero run (`value_weight_set=1e-12`)
  with zero live lift and `residual_cause_hypothesis=distribution_shift_or_calibration`; this is "tried and did
  not help", not "never wired". `scripts/arc_orphan_solver_lint.py` now passes (`OK: all solver-like ARC modules
  are reachable from the live agent path (46 modules in the live closure).`). Residual 2026-07-02 audit of
  `ops/arc_solve_registry.yaml`: registry declares `reproducible_total_games: 24` but 25 rows carry
  `levels_reproduced>0`; using the declared-24 view, 4 current-depth banked games are
  `live_agent_self_discovery` and 20 are `development_proxy` by banking artifact/mechanism (row-level view:
  4 live, 21 development_proxy, with `wa30` the extra legacy L1 row).
- failure mode: `reproducible_total_levels` (what the sprint optimizes) is largely a MIRAGE for the leaderboard;
  the score is driven by generic OOD solve-rate + action efficiency, which the submitted agent barely has.
- missing discriminator: n/a — this remains INTEGRATION / provenance hygiene, not modeling. Do NOT re-propose
  "wire the router/DSL" or "raise target_levels" as new work; those are already current-code facts. The real
  residual scope for exp5176 or a future milestone is to reduce the registry's development-proxy/current-depth
  ratio by making banked progress arise from the live agent's own runtime self-discovery path, and to keep
  `reproducible_total_games` aligned with the per-game rows.
- priority: medium (the stale highest-lever rebuild is closed; the provenance-mirage ratio is still operationally
  important but is a narrower residual than the original wiring gap)

### GAP-ARCH-WORLD-MODEL-TRUST-ENERGY: learned oracle-distinct energy for hidden-state world-model trust
- status: open
- evidence: the live agent gates planning on a BINARY `WorldModelVerifier(...).score(engine).accuracy <
  0.5` cutoff (`arc_competition_agent.py:779-780`) + `consistency_energy` (`:698`,
  `arc_world_model_dsl.py:305`). `results/arc3_m2_world_model.json` shows the energy already SEPARATES
  hidden-state from Markov games (0.88 vs 0.75, separation 0.8) but the agent uses only the hard
  threshold, not a ranking.
- failure mode: when the E3 proposer emits several candidate world-models, the agent trusts the FIRST to
  clear 0.5, not the one that GENERALIZES on held-out transitions; and it pays no energy benefit on the
  ~11 hidden-state games where there is NO cheap execution oracle (the moat slot).
- missing discriminator: a learned/calibrated trust energy `E(transitions, engine) -> trust` that ranks
  candidates by held-out (not prefix) misprediction, discriminating specifically on hidden-state games.
- candidate design: prefix-vs-held-out energy gap features -> logistic/isotonic ranking; replace the 0.5
  cutoff behind a flag for hidden-state games only (keep execution check for Markov). Full spec:
  docs/research-notes/arc-world-model-trust-energy-spec.md.
- priority: medium (real oracle-distinct MOAT work; sequence AFTER the .414 integration/feature
  score-drivers — the EBM thesis's one genuinely load-bearing ARC slot, verifier_is_oracle: false)

### GAP-ARCH-FRAME-CHANGE-PREDICTOR: learned CNN action-effect / clickability model for action efficiency
- status: attempted_and_falsified (2026-07-19) -- candidate design WAS built and shipped
  (`SmallFrameChangeCNN` / `FrameChangeScorer` / `LiveActionEffectScorer`, `python/carnot/agentic/
  arc_frame_change_predictor.py`, live-default-on since REQ-ARC-FCP-4490/4629), then rigorously
  A/B'd across 4 experiments in one session and found to carry NO robust non-base-rate signal. Not
  filled -- the underlying need (action-efficient candidate ranking) remains genuinely open; this
  specific candidate design is the part that's closed.
- evidence (original, 2026-06-20): leaderboard competitive intel -- the leader (Tufa StochasticGoose,
  1.21) + 2nd (Blind Squirrel ResNet18) win on a CNN that predicts which actions cause a frame change;
  the 30-day report shows StochasticGoose wasted ~350 no-op clicks before learning clickability. Our
  explorer (`arc_graph_explore.py:44 rich_action_candidates`) enumerates centroid-clicks + keys with NO
  effect prioritization -> action-inefficient (the scoring lever min(human/agent,1)^2; we are at 0.08).
- failure mode (original): every candidate action is equally likely to be a no-op; the explorer burns
  its action budget on no-effect actions, collapsing the squared efficiency term even on solved levels.
- **falsification chain (2026-07-19, 4 experiments, all adversarial_verify-clean, 0 flags each):**
  - REQ-ARC-FCP-5590 fixed a real dict-candidate crash that silently zeroed the CNN term in live
    ranking -- clean null (byte-identical control/treatment across an 11-game roster).
  - REQ-ARC-FCP-5728 swept the CNN's blend weight 0.05->2.0 (40x range, memory_weight fixed at 1.0) --
    clean null (`any_weight_beats_baseline_levels: false`), but localized WHY: a validation gate
    (`GroundTruthValidatedFrameChangeScorer`) returns 0.0 upstream of the weight multiply on 7/11 games,
    so the CNN was rarely-to-never consulted regardless of magnitude.
  - REQ-ARC-FCP-5729 tested loosening that gate (rate-tolerance + reset-on-levelup vs baseline) --
    the gate loosening is SAFE (no states_expanded regression, genuinely miscalibrated scorers still
    correctly blocked) and DOES turn the scorer on (3/11 -> 9-10/11 games validated, ranking consults
    ~3x), but STILL zero level gain -- even consulted ~27,000 times on lp85, search was byte-identical
    to not having the CNN at all. Localized the blocker one layer further: the scorer's own
    discriminative quality, not the gate admitting it.
  - REQ-ARC-FCP-5730 went straight at that: applied REQ-ARC-WMTE-5727's action-id-only base-rate
    control (the SAME adversarial control that found the hand-crafted linear dynamics features were a
    base-rate mirage, `frame_adds_over_action_id = -0.039`) to the CNN's own held-out AUROC. Result:
    the CNN's headline 0.709 AUROC (exp4547, stale corpus) does not reproduce on the current
    ~11.8x-larger corpus (0.539 mean over 5 seeds); an action-id-only baseline scores HIGHER (0.549,
    `frame_adds_over_action_id = -0.010`). The one promising sub-finding (click-location discrimination
    within a fixed action-id, 0.918 AUROC on one seed -- the one thing a base-rate baseline structurally
    cannot explain) did not survive a 5-seed re-run (0.444-0.918) or its own untrained/random-init
    structural control (mean 0.580 untrained > 0.570 trained) -- seed luck, not a learned signal.
- missing discriminator (revised): what's still missing is a representation that captures WHICH
  specific action, at WHICH specific frame location/state, will produce a change -- i.e. a genuine
  action x frame INTERACTION term. Both tested representations (REQ-ARC-WMTE-5727's hand-crafted
  linear features, and this CNN's learned conv features) reduce to the SAME degenerate structure: a
  per-action-TYPE base rate that the `PersistentAEM` memory term already captures for free, making the
  CNN redundant with a signal the live agent already has. A fix needs to break that redundancy, not
  retrain the same frame-only architecture on more data or tune its weight/gate.
- candidate design (revised, NOT yet built or tested -- open): (a) an explicit local-neighborhood /
  patch-level interaction feature (what's immediately AROUND a click target, not just the global frame)
  rather than global-pooled conv features feeding a single sigmoid per action-id; (b) a
  contrastive/pairwise training objective that specifically penalizes base-rate-only prediction (forces
  the model to discriminate WITHIN an action-id, not just across them) instead of the current pointwise
  changed/no-op classification loss, which a base-rate-matching model can trivially minimize; (c) more
  history than a single before/after frame pair -- multi-step context so the model can condition on
  what's already been tried/observed at a specific location, closer to what a real "is this clickable"
  judgment needs. None of these are validated; they are candidate directions for a genuinely new
  representation, not incremental tuning of the falsified design.
- priority: high (the underlying need -- action-efficient candidate ranking, the scoring lever
  min(human/agent,1)^2 -- is unaddressed by anything currently live; this is the most direct steal from
  the leaderboard leader and remains open) but LOWER urgency for immediate re-attempt than the priority
  tag alone suggests: per the Failed-Experiment Rerun Discipline, any follow-up MUST target one of the
  revised candidate designs above (a genuine representation change), NOT a retrain/re-tune of
  `SmallFrameChangeCNN` on the same frame-only architecture -- that specific path is now closed by 4
  independent, adversarially-verified negative results in one session.

---

## GAP-WM-TRUST-GATE: world-model trust gate gameable by identity on no-op-heavy corpora (2026-06-21)

### GAP-WM-TRUST-GATE: change-weighted world-model verification
- status: open
- evidence: outer-loop induction-quality investigation 2026-06-21
  (docs/research-notes/arc-l1-l2-barrier-diagnosis-2026-06-20.md UPDATE). lp85 L1 exploration via
  `collect_transitions("lp85", n=120, seed=0)` yields 120 transitions, ALL ACTION6 clicks, 87 no-ops /
  33 grid-changing. An IDENTITY engine (`return grid`) scores 0.725 on `WorldModelVerifier.score` and
  PASSES the `accuracy >= 0.5` trust gate in `arc_competition_agent.py:_induce_and_plan`.
- failure mode: the WorldModelVerifier counts the fraction of ALL transitions reproduced; on a
  click-driven game where most candidate clicks are no-ops, a do-nothing engine reproduces every no-op
  and clears 0.5 — but it predicts no change, so `plan_in_model` finds no path to a win state and the
  induction tier contributes nothing. The gate trusts a useless model. (Both Qwen3.5-9B and
  Qwen2.5-Coder-14B in fact score 0.0 here by OVER-transforming every click, the opposite failure — but
  the gate's blind spot is the identity direction: a model that learns "clicks usually do nothing" would
  be trusted while being unable to plan.)
- missing discriminator: a CHANGE-WEIGHTED world-model score — accuracy restricted to (or up-weighted on)
  the grid-CHANGING transitions, plus a minimum count of correctly-predicted CHANGES, so the gate measures
  whether the engine models the MECHANIC rather than reproducing inaction. Equivalently: require the engine
  to be non-degenerate (predict >=1 real change correctly) before it is trusted for planning.
- candidate design: extend `WorldModelVerifier` to return `change_accuracy` (n_correct over the
  grid-changing transitions only) and `n_changes_correct`; the `_induce_and_plan` trust gate becomes
  `change_accuracy >= T AND n_changes_correct >= k` instead of overall `accuracy >= 0.5`. Cheap, pure
  Python, no model. NOTE: needs adversarial review before landing (standing rule) — changing a trust gate
  can mask or unmask induced engines in the live path.
- priority: medium (does not by itself unlock L2 — induction quality (GAP not here) is the binding
  constraint — but it removes a false-trust hole that would let a degenerate identity engine through, and
  it gives the induction loop an honest non-degeneracy signal to optimize against)

## GAP-ARC-4713-SURFACING-r11l

- Date: 2026-06-25
- Residual: present_winner_not_separable_from_distractors
- Context: object-centric coverage kept the L1 winner present, but the off-path-calibrated structural ranker did not lift it into actionable top-k.
- Evidence: no_surfacing_precision={'k': 8, 'hits': 0, 'total': 4, 'precision': 0.0}; surfacing_precision={'k': 8, 'hits': 0, 'total': 4, 'precision': 0.0}.
- Needed verifier: a non-circular discriminator that separates the present winning slot from same-depth distractors before the live explorer exhausts budget.


- EXP4727 active_probe_disambiguation bp35 budget_insufficient: active probing did not bank a new reproduced level. active_level=0, no_probe_level=0, probe_actions=0, posterior_entropy_reduction=0.000000. Needed verifier/probe gap: an oracle-distinct discriminator whose transition buckets separate the true mechanic at logical-grid resolution and still imply a level-completion policy.

### GAP-4890: within-game L2->L3 goal re-induction from a SINGLE win exemplar
- status: building (FLOOR ADDRESSED 2026-06-27 -- induce_goal_energy_single_positive shipped + unit-tested: arc_agi3_goal_induction.py + tests/python/test_arc_single_positive_goal_energy.py; fires from ONE win iff strictly separated from negatives, drops into the graph_explore_solve_v2 goal_energy hook. Residual binding ceiling is NOT the floor but the goal-REPRESENTATION -> GAP-4891.)
- evidence: results/arc_within_game_l3_self_induction_cd82_stage1.json (2026-06-27, outer-loop). cd82 reached the solved frontier (level sequence over the 20-label adapter trajectory = [0,0,0,0,0,1,1,...,1], exactly ONE level-completion). induce_goal_energy (arc_agi3_goal_induction.py:61) requires >=2 win grids and returned None (n_win=1) -> goal re-induction could not even FIRE at the L2->L3 transition. The new single-positive operator clears that floor (it fires from 1 win); see GAP-4891 for why cd82 still does not deepen.
- failure mode: 15 of 24 reproduced games stall at exactly L2 NOT because of a search/perception failure but because the goal-induction operator structurally cannot bootstrap: at the solved frontier the live agent has only ONE level-completion exemplar, below induce_goal_energy's >=2 floor. So the next level's goal predicate is never induced, and graph_explore_solve_v2 degrades to blind BFS with no goal bias.
- missing discriminator: a goal-induction operator that derives a graded next-level goal-energy from a SINGLE win exemplar + many self-played non-win negatives (or that harvests multiple positives per level: multiple distinct solution paths to the same completion, near-win/sub-goal frames, or cross-game prior pseudo-positives of the same mechanic_class).
- candidate design: (a) single-positive energy via contrastive ranking of the one win grid vs the negative distribution (vs the current >=2-positive object/color hypotheses); (b) positive-harvesting wrapper that re-solves a banked level via N distinct paths to collect N completion grids before inducing; (c) mechanic_class-conditioned prior that seeds positives from already-solved same-class games. Reuse the existing graph_explore_solve_v2 goal_energy hook once an energy can be produced.
- priority: high (it is the single binding ceiling on multi-level deepening = the only lever that grows reproducible_total_levels via live self-discovery; 15 games gated on it)

### GAP-4891: goal-induction REPRESENTATION beyond object/colour COUNTS (spatial/value/order goals)
- status: building (GOAL-DETECTION gap CLOSED for cd82/sk48/sp80 via RELATIONAL target-match; but STAGE-2
  shows it does NOT unlock deepening, and Stage-3 relational-mask pruning still does NOT bank a level or
  reduce applied states_expanded under the same 4000-expansion budget. Stage-4 MAP-style landmark
  prestage also seeds the frontier but banks zero levels. Binding wall remains trajectory
  ENUMERATION/generation, not goal-detection. NEXT specific lever must move beyond this bounded
  MAP-landmark prestage because it did not alter the banked-level outcome. Ladder:
  counts[GAP-4890]=fail -> richer-scalar=fail -> RELATIONAL target-match=SEPARATES 3/4 -> stage-2
  guidance=NEITHER arm banks L3 -> stage-3 mask-pruner prunes edges but still NEITHER arm banks L3 ->
  stage-4 MAP landmark prestage seeds frontier but still zero banks. See UPDATE-5 then UPDATE-4 then
  UPDATE-3 [decisive negative] then UPDATE-2.)
- UPDATE-5 2026-07-03 (STAGE-4 decisive: bounded MAP-style landmark prestage does NOT close the
  enumeration wall): built python/carnot/agentic/arc_map_landmark_prestage.py, added the
  `frontier_seed_bank` hook to graph_explore_solve_v2, and wrote
  python/carnot/experiment_5198_map_landmark_prestage_prototype_v476.py plus
  results/experiment_5198_map_landmark_prestage_prototype_v476.json. Protocol: pruner-only uses the
  exact Exp5175 pruned baseline; map-only and map-plus-pruner build a 750-step state-novelty cognitive
  map from the post-prefix offline state, record reachable regions/action-effect deltas/relational
  landmarks, seed replayable landmark trajectories into graph_explore_solve_v2, and run the same
  4000-expansion reproduction-gated search. RESULT: lever_validated=false; levels_banked=[]; cd82,
  sk48, sp80, and cn04 all ended at L1 with states_expanded=4000 for pruner_only, map_only, and
  map_plus_pruner. Map overhead was 750 exploration steps per MAP arm (wall-clock roughly cd82 2.4-2.6s,
  sk48 7.5-8.6s, sp80 7.3s, cn04 4.9-5.4s). The MAP seed hook fired (frontier seed injected once per
  MAP arm) and map-plus-pruner still exercised the relational mask (cd82 358 prunes, sk48 22808, sp80 0,
  cn04 360), but zero reproduction-gated levels were banked. cn04 negative control stayed clean and
  arc_orphan_solver_lint passed. DECISIVE READ: this bounded MAP-landmark prestage does not enumerate the
  missing winning trajectory either; GAP-4891 remains a trajectory-enumeration/generation wall under this
  lever too.
- UPDATE-4 2026-07-02 (STAGE-3 decisive: relational-mask move-pruner is exercised but does NOT close the
  enumeration wall): built python/carnot/experiment_5175_gap4891_relational_mask_pruner_ab_v474.py and
  wired graph_explore_solve_v2 to accept a live-path move_pruner with the same should_prune/observe
  lifecycle as OfflineSolver. Ran results/experiment_5175_gap4891_relational_mask_pruner_ab_v474.json on
  cd82/sk48/sp80 plus cn04 negative control, each with the same 4000-expansion Stage-2 budget and the same
  relational goal-energy control vs treatment+RelationalMaskMovePruner. Precondition passed
  (tests/python/test_arc_relational_mask_pruner.py: 8 passed), and arc_orphan_solver_lint passed
  (live_path_reachable=true). RESULT: no reproduction-gated new levels banked on any game; levels_banked=[];
  states_expanded_pruned == states_expanded_unpruned == 4000 for cd82/sk48/sp80/cn04; cn04 negative control
  stayed clean. The pruner did fire on candidate edges for cd82 (358), sk48 (22807), and cn04 (375), but
  that did not reduce the applied expansion count or enumerate the winning trajectory. DECISIVE READ:
  pruning alone does not close GAP-4891's enumeration wall; the next lever must generate/structure the
  trajectory space before flat frontier search, specifically the MAP-style map-then-act / hierarchical
  pre-search lane flagged by exp5172.
- UPDATE-3 2026-06-28 (STAGE-2 decisive: relational energy SEPARATES but does NOT GUIDE the search ->
  enumeration wall): built scripts/experiments/arc_relational_goal_energy_stage2.py -- wires
  induce_goal_energy_relational into graph_explore_solve_v2's goal_energy hook (energy induced live by
  replaying the game's banked L1 self-discovery seed as the prefix; the relational offset is level-invariant
  so an energy induced at level k detects level k+1's target), then runs energy-guided search vs a BFS
  ablation (goal_energy=None), reproduction-gating both -- the goal-induction doctrine's mandatory control.
  RESULT on all 3 separating games (results/arc_relational_goal_energy_stage2_{cd82,sk48,sp80}.json,
  adversarial_verify clean 3/3): induce_fired=True (the relational energy SEPARATES, confirming Stage 1) but
  NEITHER arm banks a new reproduction-gated level -- both energy-guided and BFS stall at L1 with 4000
  expansions, traj_len=0, reproduced=False. Verdict on each: complete_stage2_neither_banks_new_level_*_
  energy_L1_bfs_L1_search_wall_not_goal_energy. DECISIVE READ: separation is necessary but NOT sufficient --
  ordering the frontier by a correct goal-energy does NOT let the search REACH the next level, because the
  winning trajectory is never ENUMERATED into the frontier in the first place. The binding wall for within-
  game deepening is the SAME trajectory-enumeration wall as L1 first-contact, NOT goal-detection or value-
  prediction. This converges INDEPENDENTLY with the conductor's .452 A1 env-grounded search finding
  (WALL_DEEPER_THAN_VALUE_PREDICTION). CAVEAT (honest, keeps this from over-claiming): Stage-2 isolated the
  goal_energy lever WITHOUT the change-location action-pruning prior, so it is somewhat underpowered vs a
  fully-instrumented search -- but the .452 A1 result (complete env-grounded search WITH action-pruning) ALSO
  hit WALL_DEEPER, so the enumeration wall holds under the stronger setup too. CONSEQUENCE: the relational
  goal-energy is a genuine GOAL-DETECTION / verifier contribution (survives the post-6/30 verifier-moat
  pivot -- it is an oracle-distinct discriminator that separates win from near-win), but it is NOT the lever
  that unlocks reproducible_total_levels growth. The deepening lever is whatever makes the winning trajectory
  appear in the candidate pool (directed exploration / generation), which is the project-wide generation wall.
- UPDATE-2 2026-06-27 (RELATIONAL target-match WORKS on 3/4): built induce_goal_energy_relational
  (arc_agi3_goal_induction.py) = translational self-similarity with a background-excluded induced mask
  (find offset where the win's non-bg content matches its own translate = canvas==target-shown-at-offset;
  mask = that match-set; energy = mask-cells where g != g-shifted; 0 at win, >0 at near-win negatives).
  3 unit tests pass (10 total). EMPIRICAL on the 4 stalled games (results/arc_within_game_l3_self_
  induction_*_stage1.json, adversarial_verify clean 4/4): cd82 SEPARATES (winE=0, mean-nonE=12.8,
  frac_nonwin_above_max_win=1.0), sk48 SEPARATES (winE=0, nonE=29.0), sp80 SEPARATES (winE=0, nonE=3.27),
  cn04 FAILS (alignment -- count+scalar+relational all fail -> needs a learned/masked predicate). So the
  goal-ENERGY representation gap is CLOSED for 3 of 4 (the energy is 0 on the win and >0 on EVERY non-win,
  incl. near-win) -- the first POSITIVE in the ladder. RESIDUALS: (a) confirm the energy GUIDES the search
  (stage-2 graph_explore_solve_v2 to a reproduction-gated L3 -- separation is necessary not sufficient);
  (b) cn04 alignment needs a relational predicate beyond translation (object-centroid relative position).
- UPDATE 2026-06-27 (richer-scalar candidate refuted -> the gap is RELATIONAL, not scalar): built
  induce_goal_energy_richer (arc_agi3_goal_induction.py) adding a value/fill/spatial scalar ladder
  (nonbg_cells, max_color_count, nonbg_bbox_area, color_entropy) under the SAME strict-separation guard;
  3 unit tests pass (test_arc_single_positive_goal_energy.py). RE-RAN the stage-1.6 probe on all 4 games:
  STILL None on cd82/sk48/sp80/cn04 (operator_used=None, separating_feature=None; adversarial_verify clean
  4/4). ROOT CAUSE (the sharper finding): the negatives include NEAR-WIN frames (the penultimate frame
  differs from the win by ~one cell), so NO global scalar -- count, fill, OR spatial-extent -- can strictly
  separate the win from them. The goal is RELATIONAL/CELL-LEVEL (the configuration matches a within-frame
  TARGET; the win differs from the penultimate frame only in specific cell VALUES), which no global scalar
  can express. So the missing feature is a WITHIN-FRAME RELATIONAL TARGET-MATCH, not richer scalars.
  Re-scoped candidate design below.
- evidence: results/arc_within_game_l3_self_induction_{cd82,sk48,sp80,cn04}_stage1.json (2026-06-27, outer-loop). With the GAP-4890 win-exemplar floor cleared (induce_goal_energy_single_positive fires from 1 win), ALL 4 grid-based stalled games returned None with verdict complete_self_induction_gap4890_single_positive_no_separation_representation_ceiling: NO object-count / unique-colour-count feature strictly separates the lone level-completion win from the self-played non-win grids. adversarial_verify clean (4/4).
- failure mode: the goal-induction operator (both the >=2-win and the new single-positive variants) only has object-COUNT and colour-COUNT hypotheses (H1-H5). The stalled games' goals are SPATIAL/VALUE/ORDER, not count: cd82 palette_region_fill (canvas==target under a mask -> same counts), sk48 chain_color_reorder (same colours, reordered), sp80 spill_splitter (placement), cn04 marker_pair_shape_alignment (alignment). So the win and non-win states have identical object/colour counts and the energy reads ~0 on both -> cannot induce a usable goal -> graph_explore degrades to blind BFS -> no L3. This, NOT the win-exemplar floor, is the binding ceiling on multi-level deepening for >=4 of the 15 L2-stalled games.
- missing discriminator: a richer goal-feature family for goal-energy induction -- cell-VALUE match against an induced target/mask, spatial-pattern / alignment match, set/multiset ORDER match -- so the induced energy is 0 on the win and >0 on non-wins for value/spatial/order goals. The change-LOCATION prior (which transfers) can supply the candidate mask region; the gap is the VALUE/spatial predicate over it.
- candidate design (RE-SCOPED 2026-06-27 to RELATIONAL after the scalar refutation): the energy must be a
  WITHIN-FRAME RELATIONAL TARGET-MATCH, not a global scalar. Concretely: (a) induce a target by finding a
  region-PAIR (or region + reference) that is EQUAL in the win exemplar but UNEQUAL in the negatives (the
  canvas vs the displayed target) -> energy = Hamming mismatch of that region pair in the candidate grid
  (this generalises across levels: the target is re-read from each level's frame, not the win's exact
  values); (b) for reorder games, a sequence/permutation-match energy over the ordered colour run; (c) for
  alignment, an object-centroid relative-position match. The strict-separation guard still selects which
  relational predicate fires. NOTE: global-scalar features (counts AND value/fill/spatial) are now
  EMPIRICALLY RULED OUT (this UPDATE) because near-win negatives bracket the win on every global statistic.
  Then re-run the stage-1.6 probe; success = the relational energy is 0 on the win and >0 on near-win
  negatives -> proceed to the graph_explore_solve_v2 L3 search.
- priority: high (now THE binding ceiling on within-game deepening; confirmed 2026-06-28 when the .451 A2
  level-up STALLED on a stalled-game with repro=False/newlvls=0 -- the reliable +1/milestone lane is now
  blocked on exactly this; growing reproducible_total_levels via live self-discovery depends on it)

<!-- experiment_5200_hidden_state_verifier_v2_mmlu_pro_v476:start -->
### experiment_5200_hidden_state_verifier_v2_mmlu_pro_v476
- status: open
- evidence: `results/experiment_5200_hidden_state_verifier_v2_mmlu_pro_v476.json`; honest_verdict=complete_hidden_state_probe_does_not_beat_tuned_sc_probe0.100_sc0.075_self0.075_clue0.100_rcs0.100; probe_accuracy=0.1; tuned_sc_accuracy=0.075; self_certainty_accuracy=0.075; clue_accuracy=0.1; radial_consensus_score_accuracy=0.1.
- failure mode: probe_missed_oracle_recoverable_candidates.
- missing discriminator: candidate-internal correctness signal that separates correct MMLU-Pro traces from dense wrong-answer clusters.
- candidate design: add a stronger supervised hidden-state probe or transformer-layer sweep once output_hidden_states access is practical.
- priority: medium; oracle-recoverable probe misses=8 on this eval split.
<!-- experiment_5200_hidden_state_verifier_v2_mmlu_pro_v476:end -->

<!-- experiment_5213_hidden_state_verifier_v3_layer_chunk_sweep_v477:start -->
### experiment_5213_hidden_state_verifier_v3_layer_chunk_sweep_v477
- status: retired
- evidence: `results/experiment_5213_hidden_state_verifier_v3_layer_chunk_sweep_v477.json`; honest_verdict=complete_hidden_state_v3_signal_does_not_beat_all_controls_retires_mmlu_hidden_state_path_probe0.075_sc0.075_self0.075_clue0.025_rcs0.025; best_probe_accuracy=0.075; tuned_sc_accuracy=0.075; self_certainty_accuracy=0.075; clue_accuracy=0.025; radial_consensus_score_accuracy=0.025.
- failure mode: mmlu_hidden_state_path_retired_no_positive_ci_vs_all_controls.
- residual gap: richer hidden-state access did not provide a positive-CI selector win over tuned SC, self-certainty, CLUE, and RCS on the headroom-confirmed MMLU-Pro pool.
- recommendation: retire MMLU-Pro hidden-state verifier path; do not rerun this path without a new non-final-layer internal signal or a different corpus-level mechanism.
<!-- experiment_5213_hidden_state_verifier_v3_layer_chunk_sweep_v477:end -->

### GAP-ARC-INERT-CLICK-PRUNER-5xxx: no live-path pruner for inert/no-op click targets
- status: built_and_wired_but_no_live_efficiency_benefit_at_budget200 (2026-07-20, exp5756). The
  component was BUILT + unit-tested + live-wired on 2026-07-13 (`InertClickSigPruner`,
  `arc_inert_click_pruner.py`; `rank_candidates` drop-filter in `StepwiseExplorer._candidates` + a
  real `observe()` from `_ingest`; gated OFF by default `SUBMITTED_INERT_CLICK_PRUNER_ENABLED=False`).
  The validation A/B that was missing is now done — exp5756
  (`results/experiment_5756_inert_click_pruner_11game_ab.json`), the 11-game roster / budget=200 /
  no-LLM matched-budget A/B extending the earlier single-game (`m0r0`, budget~37) never-fired nulls
  exp5595 + exp5602. RESULT: at budget=200 the pruner FIRES (unlike the priors) — 2643 candidates
  pruned across treatment arms, all on `sk48` — with NO missed win (`suppressed_a_winnable_click:
  false`, the trust+specificity+sacred gate held, every game kept its baseline level) BUT NO
  efficiency benefit: `states_expanded` INCREASED (baseline 931 -> treatment_default 953 (+22, all
  from `sk48` 26->48) -> treatment_aggressive 960), the OPPOSITE of `HazardMovePruner`'s tu93 win
  (2947->2859). On the one game where it pruned, dropping frame-inert clicks reshaped the search
  frontier so more states were expanded for the same banked levels (a frame-inert click can still be
  a necessary traversal step the search re-routes around). Recommendation (operator-only): do NOT
  flip the live default; another plain budget/roster A/B is not warranted without a NEW mechanism
  that makes pruning REDUCE rather than reshape search cost. Cache-hygiene fix shipped alongside
  (route the pruner's `connected_color_blobs` through the shared `_cached_blobs_and_counts` cache,
  behavior-preserving). Spec: REQ-ARC-FCP-5756.
- evidence: read-only audit of the ARC-AGI-3 Milestone-1 2nd-place team's ("Reki") open-sourced
  code (`external/arc-m1-2nd-reki/milestone1-2nd-solution.ipynb`, 2026-07-11, operator directive).
  Full writeup: `docs/research-notes/arc-agi3-milestone1-winners-sota-ingestion-2026-07-11.md` (O1).
- failure mode: our live candidate generation (`StepwiseExplorer._candidates`) has a pruner for
  LETHAL moves (`python/carnot/agentic/arc_hazard_pruner.py:HazardMovePruner`, trust+specificity
  gated on observed avatar-removal deaths) but no equivalent pruner for INERT clicks — click
  targets whose structural signature has repeatedly produced zero frame change. Reki independently
  built exactly this ("dead-signature": structural sig `(color, size, is_rect, twin_count)`,
  suppressed after 2 no-op clicks, permanently protected if ever effective) and found it valuable
  enough to keep in their winning configuration.
- missing discriminator: a per-signature "has this structural click-target class ever changed the
  frame" trust score, keyed the same way our object descriptors already key blob salience.
- candidate design: `InertClickSigPruner`, same trust+specificity-gating discipline as
  `HazardMovePruner` (NOT Reki's greedy K=2 threshold, which the audit flagged as over-aggressive
  and prone to mis-protecting context-dependent signatures / over-suppressing "twin" tiles that
  behave differently by position depending on game state).
- priority: RESOLVED-NEGATIVE (2026-07-20) — build was cheap and reused `HazardMovePruner`'s shape
  as intended, but the exp5756 live A/B shows it adds no live-path capability AND no efficiency at
  budget=200 (fires + holds levels but grows `states_expanded`), so it stays gated OFF. Not a live
  lever on this evidence; do not re-queue a plain budget/roster A/B without a new mechanism. (Original
  build was known-issues.md active ARC priority list task 9.)

### GAP-ARC-CLAIMED-VS-MEASURED-DIFF-5xxx: generator self-reports of "what changed" are never checked against the real pixel diff
- status: open_confirmed_no_hook_point_current_architecture (2026-07-14, outer-loop, task 11 final assessment)
- evidence: read-only audit of the ARC-AGI-3 Milestone-1 2nd-place team's ("Reki") open-sourced
  code, same source as above. Full writeup: same SOTA-ingestion note (O3).
- failure mode: Reki's policy LLM self-reports a `board_change_assessment` field (what it believes
  changed) in the same turn where the harness independently computes `changed_pixels` (the real
  pixel diff) — the two values are computed side by side but NEVER cross-checked, so a confabulated
  "what changed" narrative can persist uncorrected. We do not currently have any generator-facing
  component that makes this specific claim, but any future generation-time reasoning/planning step
  that asks a model to narrate observed state changes would have the identical unverified-claim gap.
- missing discriminator: a cheap distance/consistency score between a generator's natural-language
  description of an observed transition and the actual measured pixel/state diff for that
  transition.
- candidate design: `distance(claimed_diff_description, measured_pixel_diff)` — deterministic, no
  second LLM call required. Directly in-thesis (verify a claim against ground truth); see the
  SOTA-ingestion note's fragility section for why this must ship BEFORE any persistent NL
  hypothesis memory (O5) is built, not after.
- priority: medium-high — genuinely new build (not a code-shape reuse like the pruner above), but
  small and squarely in-thesis. See known-issues.md active ARC priority list task 11.
- **2026-07-14 final assessment (outer-loop, task 11's second half):** re-checked every LLM
  touchpoint in the current ARC induction pipeline (`arc_executable_world_model.py`'s
  `induce_prompt`/`refactor_prompt`/`CodexProposer`/`LocalGGUFProposer`, and
  `arc_llm_strategy_proposer.py`'s `LLMStrategyProposer.propose_one`/`reflect`) for any
  natural-language "what changed" self-report analogous to Reki's `board_change_assessment`.
  Confirmed none exists: our LLM touchpoints either (a) write executable CODE
  (`engine()`/`is_level_complete()` — `refactor_prompt` feeds real mismatches back and asks for a
  CODE fix, never a prose diff claim) or (b) state a forward-looking exploration STRATEGY
  (SGE's `propose_one`/`reflect` — "what I'm trying to learn next," not "what I think just
  changed"). Neither is the shape this gap needs. Building a NEW self-report solely to have
  something to cross-check would cost an extra LLM call per transition, directly contradicting
  this gap's own corroborating evidence (forge's 3rd-place team disabled their LLM judge/arbiter
  for cost while keeping only the free `changed_pixels==0` deterministic filter). Conclusion:
  this remains a genuine, confirmed dead-end in the CURRENT architecture, not merely an
  unstarted build — re-open only if some other component starts organically producing a
  natural-language change-claim as a byproduct of work it does for another purpose (at which
  point the candidate design above is still the right shape to hook it with).

### GAP-ARC-GOAL-HYPOTHESIS-VS-TRANSITIONS-5xxx: free-text goal/rule hypotheses are never checked against observed level-up/no-op transitions
- status: filled (score_goal_predicate_consistency)
- evidence: read-only audit of the ARC-AGI-3 Milestone-1 1st-place team's ("Duck Harness", Tufa
  Labs) open-sourced code (`external/duck-harness/`, 2026-07-11). Full writeup: same SOTA-ingestion
  note (O3/O5).
- failure mode: Duck's "scientist note" world model carries a free-text Goal/Action-model hypothesis
  that is regenerated by the LLM each turn and re-injected into the next prompt, but is never
  checked against the actual observed reward signal (level-up vs no-change transitions accumulated
  so far). An incorrect hypothesis can persist and compound. Reki's periodic reflection-memory
  rewrite (`_run_reflection`) has the identical gap: it fully replaces prior memory with a new LLM
  self-summary with no grounding check.
- missing discriminator: "does this goal-hypothesis correctly predict the sign (level-up vs
  no-change) of the last N observed action outcomes."
- candidate design: a lightweight consistency scorer gating whether a candidate goal-hypothesis is
  retained/carried-forward vs discarded, based on its predictive accuracy over recent transitions.
  This is the natural prerequisite for any persistent NL hypothesis memory (O5 in the SOTA note) —
  do not build the memory feature ahead of this check, per the note's fragility section (both source
  implementations let unverified hypotheses become "authoritative" context, which is confabulation
  with extra steps).
- priority: medium — real, but should follow GAP-ARC-CLAIMED-VS-MEASURED-DIFF-5xxx rather than being
  built in parallel. See known-issues.md active ARC priority list task 11.
- **2026-07-13/14 fill (outer-loop, task 11):** `score_goal_predicate_consistency`/
  `GoalPredicateConsistency` shipped in `arc_executable_world_model.py` (2026-07-13, `f4aa99c24`) —
  the deterministic sign check this gap calls for (`is_level_complete(next_grid)` vs real
  `level_after > level_before`), no second LLM call, matching forge's own competitive-pressure
  finding. Validated with 5 unit tests on synthetic data at ship time, then a REAL end-to-end
  positive-control demo on 2026-07-14 (exp5593 rerun, after the `induce_prompt` scalability fix
  below unblocked it): a REAL Qwen3.5-9B-MTP induction on lp85's real 8-transition window produced
  a real `is_level_complete`, scored against the real observed transitions —
  `goal_predicate_accuracy=0.75` (6/8 correct, 2 false-negative mismatches where a real level-up
  was missed), `real_levelup_present_in_sample=true` (interpretable per the FALSE_NEGATIVE_RISK
  discipline — not an all-no-op degenerate test). The check ITSELF is now demonstrated working on
  real induced output, not just synthetic fixtures; still NOT wired into any live decision
  (vetoing a goal predicate before planning) — that remains a distinct, separately-scoped step.

### GAP-5703: `GoalSatisfactionEnergy` (Exp4020) is structurally blind on sp80 — constant score, zero variance, live path (not just offline self-induction)
- status: open
- evidence: `results/experiment_5703_sp80_candidate_stack_mechanism_trace.json` (2026-07-14,
  outer-loop, task 10 — follow-up to exp5701's finding that sp80 was the one game where
  `bare_control` beat `full_stack` by a level in the candidate-scoring-stack ablation). Directly
  instrumented all three "richer stack" mechanisms live during a real sp80 episode
  (`E3AgentPolicy(game)`, budget=500, offline_arcade substrate, no LLM): `goal_bias`
  (`arc_goal_energy_live.GoalSatisfactionEnergy`, source `exp4020_graded_goal_satisfaction_energy`)
  scored **every one of 771 real frontier-node invocations at exactly 1.0** (`goal_bias_score_
  variance=0.0`); `goal_candidate_guidance` (the same energy source, applied to the immediate
  14-candidate pool) also scored uniformly (`score_min=score_max=1.0`) and correctly self-detected
  its own degeneracy (`arms_non_degenerate=False`) and no-op'd (`candidate_pool_differs_from_
  baseline=False`, by the existing design in `arc_goal_energy_live.py:274-275`); `candidate_router`
  was genuinely invoked 33 times but never once changed the candidate ordering it was given
  (`candidate_router_changed_order_count=0`). adversarial_verify clean.
- failure mode: `GoalSatisfactionEnergy.__call__` (`arc_goal_energy_live.py:338-353`) falls back to
  a constant `1.0` ("totally unsatisfied") whenever `visible_state(value)` returns `None` or the
  extracted state's `total_targets <= 0.0`. A constant score across an entire real episode means
  `_state_from_visible()` extracts no usable target-state from sp80's frames at all — the energy
  source has literally zero information about sp80's spill-splitter/placement mechanic, not just
  weak/noisy information. Since `_goal_bias_key` maps a constant score to a constant sort key
  (`arc_competition_agent.py:1139-1144`), this is not merely a weak signal, it is a mathematically
  guaranteed no-op on frontier ordering — confirmed by direct instrumentation, not inferred.
- **corroborates GAP-4891 via an independent code path.** GAP-4891 (above) found the SAME
  underlying problem — sp80's goal is SPATIAL/placement, not discriminable by
  object-count/colour-count style features — via the OFFLINE self-induction operator
  (`induce_goal_energy_*` in `arc_agi3_goal_induction.py`), a completely different module from
  `GoalSatisfactionEnergy`/`arc_goal_energy_live.py`. This entry shows the SAME game's SAME class
  of failure also reaches the LIVE submitted agent's own search behavior (`goal_bias` +
  `goal_candidate_guidance`, both wired into `E3AgentPolicy`'s default "full stack" per
  `SUBMITTED_AGENT_CONFIG`) — not just an offline diagnostic tool. The gap is broader and more
  load-bearing than GAP-4891 alone suggested: it affects what the scored agent actually does on
  this game class, not only a research-side self-induction experiment.
- missing discriminator: a placement/spatial-aware `visible_state()` extraction (or an alternate
  goal-energy source entirely) that can represent progress toward a target CONFIGURATION — cell
  values/positions matching an induced or hard-coded target — for games whose win condition is
  "put things in the right place" rather than "make a count/fraction go to zero." The relational
  target-match representation GAP-4891's UPDATE-2 already built and validated
  (`induce_goal_energy_relational`, separates sp80's win from every non-win, `winE=0, nonE=3.27`)
  is a candidate signal source that could replace or supplement `GoalSatisfactionEnergy` for games
  in this class — it already exists and is proven to discriminate on this exact game, just not
  wired into the LIVE `goal_bias`/`goal_candidate_guidance` path.
- candidate design: (a) a degenerate-score self-audit on `goal_bias`'s frontier-node scoring,
  mirroring the audit `goal_candidate_guidance` already has (`arms_non_degenerate`) — so a
  zero-variance goal-energy source safely falls back to no-goal-bias search instead of silently
  contributing an inert-but-unaudited key, closing the asymmetry this investigation found between
  the two mechanisms; (b) route `goal_bias`/`goal_candidate_guidance` through GAP-4891's
  already-validated relational target-match energy on games flagged as placement/spatial-class
  (the mechanic-class routing infrastructure already exists per `arc_solve_learning.recommend_
  approach` / `_recommend_live_approach`), rather than the generic Exp4020 predicate-fraction
  formula.
- what does NOT explain the sp80 regression: since all three learned mechanisms are proven inert
  on this game (constant score / no reordering / self-audited no-op), they cannot be the cause of
  `full_stack` losing a level to `bare_control` there. The regression's real cause is one of the
  remaining differing knobs (`value_weight`/DAgger value head, `navigation_cost_tiebreak`,
  `action_effect_expansion_prior`) — NOT further isolated in exp5703's scope; a natural follow-up
  if sp80 specifically becomes headline-relevant.
- priority: medium — real and load-bearing for the live agent's search quality on placement-class
  games (not just a research diagnostic), but the exp5701 sweep found sp80 is the ONLY game (of 5
  with measured headroom) where this class of degeneracy corresponded to a measured regression;
  candidate design (a) is cheap and should be paired with any future goal_bias work regardless of
  priority tier, since it is a general robustness fix (fail-safe on ANY out-of-distribution game),
  not sp80-specific.

### GAP-OBJECT-HISTORY-SALIENCE-ONLINE-SIGNAL-DOES-NOT-REACH-LIVE-FRONTIER: object-identity change-history is real online but does not convert to a live-path level
- status: open
- evidence: `results/experiment_5740_object_history_salience_11game_ab.json` (REQ-ARC-FCP-5740).
  A properly-powered 4-arm 11-game live A/B of `ObjectHistorySaliencePrior` (the `object_hash`-keyed
  change-history `action_prior`) found NO capability gain: baseline/`blob_only`/`treatment_default`/
  `treatment_rescaled` all bank exactly `1` level (`lp85` L1), `any_config_beats_baseline_levels:
  false`, no safety regression (states 931 -> 812). Isolated against a `blob_only` (ColorBlob prior,
  no bonus) control, the object-history bonus's TRUE marginal behavioral effect is small -- it
  reorders search on only 4 games at default weight / 5 at the rescaled weight (vs the ColorBlob
  prior's own 8-game effect) -- and banks no level anywhere. This is the LIVE-path counterpart to
  exp5732 (REQ-ARC-FCP-5732), which measured a genuinely predictive ONLINE prefix-causal
  `object_hash`-keyed memory (within-game AUROC 0.844 vs click-bucket 0.711, +0.133).
- failure mode: the object-identity change-history signal is real and predictive as an OFFLINE/online
  reranking key (exp5732), but wiring it as a live `action_prior` frontier-ordering bonus does not
  convert that predictive signal into a banked level at budget=200 on the 11-game roster -- the search
  re-orders a few candidates (4-5 games) but the re-ordering never reaches a solve. The existing
  action_prior families (ColorBlob tier/score, this bonus) rank WHICH click to try, but ranking is not
  the binding constraint here: even a perfect object-affordance ranking does not close a level if the
  path to level-up needs multi-step planning the frontier order alone cannot supply.
- missing discriminator: a mechanism that converts a per-object change-affordance PREDICTION into
  multi-step level-progress, not just a single-candidate frontier bonus -- e.g. an object-affordance-
  conditioned world-model/plan step (`plan_in_model` consuming the object_hash change-rate as an
  action-effect prior) rather than a myopic score bump, OR an online object-affordance memory consumed
  by the SCORED `E3AgentPolicy.plan_in_model` path (per ARC Live-Path Reachability Discipline) instead
  of only the `StepwiseExplorer` frontier order.
- candidate design: (a) route exp5732's `object_hash`-memory as an action-effect prior INTO the
  world-model planning step (`e3.plan_in_model`), where a predicted-change object gets tried as part
  of a multi-step plan, not a single-frontier reorder; (b) a deeper-budget / per-game-headroom probe
  to test whether the small live divergence ever converts on games with real level headroom beyond
  budget=200 (the 11-game roster's only banked level, `lp85` L1, is reached by all arms, so this
  roster has little level headroom to reveal a conversion). Both are NEW mechanisms, not another
  `ObjectHistorySaliencePrior` weight/roster A/B (which the exp5740 retire condition closes).
- priority: low-medium -- the underlying object-identity signal is real (exp5732), so this is not a
  dead direction, but exp5740 shows the current frontier-bonus wiring is not the live-path lever;
  do NOT re-run the same action_prior A/B, and do NOT flip `SUBMITTED_OBJECT_HISTORY_SALIENCE_ENABLED`
  (operator-only) on this evidence. The exclusion decision (retire the object-history LIVE-PATH
  action_prior lineage vs keep it for a future world-model consumer) is operator-only; this entry is
  the falsification record, NOT an edit to `ops/exclusion_manifest.yaml`.

### GAP-ARC-CLICK-SELECTION-5758: the winning object-click is unselectable within a dense single-pixel field — no perceptual salience feature discriminates it
- status: open
- evidence: `results/experiment_5758_click_ranking_fix_ab.json` (REQ-ARC-FCP-5758) + the motivating
  attribution `results/experiment_5757_candidate_coverage_attribution.json` (REQ-ARC-FCP-5757). exp5757
  localized Carnot's ONLY residual single-action gap to SELECTION/RANKING of object clicks (bucket c: 6
  action-6 clicks that ARE generated + frame-changing but rank >= 12 of ~27-34; r11l x1, su15 x4). exp5758
  diagnosed the frames and A/B-tested a fix: the low-ranked winning clicks are consistently VERY SMALL
  objects (r11l's rank-22 winner is a single pixel area=1; su15's repeatedly-clicked winner is area=1
  color=3; r11l's others area 4/12), and the shipped default salience `area*(1+1/(1+global_color_px))` is
  AREA-DOMINATED so a 1-pixel target always loses to large decorative regions.
- failure mode: an opt-in small-object-first reorder (`CARNOT_ARC_SMALL_OBJECT_FIRST`; small band area<=8
  by colour-rarity first, then the proven salience order) was A/B-tested on the 11-game roster at budget=200
  and is a clean NULL: both arms bank 1 level (lp85, incidental — headroom present), r11l/su15 stay at 0 in
  BOTH arms (`r11l_su15_specifically_fixed: false`), search re-orders on 7/11 games but banks no extra level,
  and search cost actually DROPS 931->807 states (no regression). The offline rank-shift is a WASH (0 of 2
  decisive games reduce their low-rank count): promoting small objects reshuffles the single-pixel field
  WITHOUT picking THE winning pixel, because the winner is outranked by many EQUAL-size, rarer-coloured
  pixels — no monotonic-in-area (or rarity) formula surfaces it. This extends the two prior reorder nulls
  (`CARNOT_ARC_TIER_SCHEDULE`/`proto_tier_ab.json` TIER_NULL_no_win, which front-loads MEDIUM-width objects
  and EXCLUDES the 1x1 winners; exp4556's learned DiscriminativeVerifier `candidate_router`, no value on
  colour-variant first-contact): STATIC PERCEPTUAL click ordering is not the binding constraint.
- missing discriminator: a GOAL-CONDITIONED / LEARNED click discriminator that identifies THE
  progress-making object among a field of perceptually-identical small objects — i.e. a signal keyed on the
  object's ROLE in reaching the goal (predicted level-progress under a world model), not its static
  appearance (area/colour-rarity/size-band, all of which are at chance here). This is the click-selection
  specialisation of the world-model INDUCTION bottleneck exp5757's recommendation named (the deeper lever):
  the winner is distinguished by what it DOES, which requires the induced action-effect/goal model, not the
  frame alone.
- candidate design: (a) consume a learned action-effect / goal-satisfaction prediction (the induced world
  model) as the click SELECTOR inside the SCORED `E3AgentPolicy.plan_in_model` path (per ARC Live-Path
  Reachability Discipline), so a click is ranked by predicted multi-step progress, not appearance; (b) a
  per-object change-affordance memory keyed to goal-progress rather than mere frame-change (extends the
  exp5732 `object_hash` signal past the exp5740/exp5756 frontier-bonus nulls); NOT another static perceptual
  reorder (that class is now falsified across tier-schedule, learned-router, and small-object-first).
- priority: medium — this is Carnot's ONLY measured residual single-action gap on the stalled roster, but
  exp5758 shows it is a discriminating-SIGNAL absence (a verifier we NEED), not a ranking-formula bug, so
  closing it is a world-model/goal-conditioned build, not a cheap salience tweak. Keep
  `CARNOT_ARC_SMALL_OBJECT_FIRST` OFF (operator-only whether to flip); this entry is the falsification
  record, NOT an edit to `ops/exclusion_manifest.yaml`.

## GAP-ARC-INDUCTION-REFINEMENT-NULL: verifier-grounded CEGIS refinement does not lift world-model induction quality, on either a small or a substantially larger frozen local model (2026-07-21)

### GAP-ARC-INDUCTION-REFINEMENT-NULL: CEGIS refinement is not the fix for near-zero induction accuracy
- status: tested_and_null — the design doc's own HONEST-NEGATIVE branch, but disclosed with real nuance
  (neither run cleanly satisfies the branch's literal criteria; see below)
- evidence: `results/experiment_5760_cegis_refinement_induction_ab.json` (REQ-ARC-WMTE-5760,
  ThinkingCap-Qwen3.6-27B + Qwen3.5-9B-MTP, 78 cells) and `results/experiment_5766_gemma31b_cegis_refinement_ab.json`
  (REQ-ARC-WMTE-5766, gemma-4-31B-it, 39 cells, run in PARALLEL on a second GPU per operator directive to
  test whether a substantially more capable model changes the answer). Both ran the SAME 13-game
  pre-registered roster through the live agent's OWN CEGIS refinement loop
  (`execute_bounded_llm_reinduction`, `arc_llm_reinduction.py:654`, wired into both live call sites
  `arc_competition_agent.py:3885,4005`) instead of single-shot induction, measuring the WITHIN-loop delta
  (round0 vs best-refined round) to isolate the refinement mechanism's own effect. Result on BOTH models,
  independently re-verified against each artifact's raw fields: `positive_game_frac = 0.0`, `wins = 0` —
  literally zero games improved on either model. Pooled `delta_heldout`: `-0.0128` (ThinkingCap+Qwen) and
  `-0.0598` (gemma-4-31B). `degradation_guard` holds on both (no corruption of an already-correct model).
- failure mode: this closes the design doc's own open architectural question
  (`docs/research-notes/arc-induction-quality-improvement-design-2026-07-20.md` §2) with a real
  measurement rather than an assumption. Model CAPACITY genuinely helps single-shot induction quality
  (REQ-ARC-WMTE-5764: gemma-4-31B pooled single-shot heldout `0.378` vs ThinkingCap-27B's `0.188`, both
  independently verified) — but applying MORE COMPUTE VIA REFINEMENT to either model's guesses does not.
  This corroborates arXiv:2606.31511 ("Falsification, Not Exposure") on this exact task: for frozen local
  models, self-repair feedback content does not improve correctness; only comparison against external
  executable ground truth would (the falsification/filter role, not the correction role). Two real,
  disclosed nuances keep this from being a CLEAN HONEST-NEGATIVE: (1) window-memorization rate dropped
  substantially on ThinkingCap-27B (`0.73 -> 0.32`) and gemma-4-31B (`0.26` absolute drop) even without a
  matching correctness gain — refinement changes the induced code's SURFACE structure (stops hardcoding
  literal observed coordinates) without fixing its DYNAMICS understanding; (2) Qwen-9B's refactor-round
  emission rate (`0.55`) fell just under the `0.6` healthy threshold, partially confounding its own null
  (ThinkingCap's and gemma's emission rates were both healthy, `>0.98`).
- missing discriminator: per the design doc's own pre-registered consequence for this outcome — the gap is
  NOT "the refinement loop needs tuning" (already tested: `min_heldout_accuracy=1.0` forces genuine
  refactor attempts, not early-accept; emission is healthy on 2 of 3 model arms; the memorization detector
  confirms refactor DOES act on the code, just not on its correctness). The missing thing is either (a) a
  genuinely bigger/different-class offline induction model (permitted on the conductor's 3090s for offline
  work), to test whether the capacity-helps-single-shot trend continues past 31B, or (b) reconsidering
  whether Carnot should induce an explicit, verifiable world model up front AT ALL versus a Duck-Harness-style
  reactive loop where the verifier FILTERS a capable model's turn-by-turn choices instead of trying to
  correct a wrong symbolic model after the fact — the falsification-not-correction distinction the cited
  literature makes precisely.
- candidate design: (a) single-shot (NOT CEGIS-refined) induction with a materially larger/different-family
  model than gemma-4-31B, same 13-game roster, same metric, to extend the REQ-ARC-WMTE-5764 capacity trend
  and see where (or whether) it plateaus; (b) a reactive-with-verifier-as-filter prototype on 2-3 games,
  scoped adversarially per the Phase Prototype + Adversarial Check discipline before any broader investment,
  since this is a real architecture-level bet, not a parameter tweak.
- priority: high — this is the decisive result of the whole 2026-07-19/20 wiring-layer-exhaustion ->
  induction-quality-convergence investigation arc (`ops/status.md` session entry). Do NOT re-propose a CEGIS
  refinement-loop variant on a frozen local model without a NEW mechanism (per `retire_if_same_verdict` in
  both REQs' `preregistration` blocks) — the next move is model class or architecture, both operator-only
  decisions given the GPU-day-scale cost of either.

### GAP-ARC-REACTIVE-FILTER-MYOPIC: verifier-filtered reactive loop is real but LOCAL, not GOAL-DIRECTED
- status: tested_and_null — the candidate design (b) above from GAP-ARC-INDUCTION-REFINEMENT-NULL, run for
  real, honest negative, with a genuine architectural diagnosis of WHY (not just "it didn't work")
- evidence: `results/outer_loop_reactive_verifier_filter_ab_20260722.json`
  (`carnot.agentic.arc_reactive_verifier_filter.run_reactive_verifier_filter_progress`, REQ-ARC-WMTE-5827,
  operator-directed 2026-07-22). Real, non-mocked run on the 3 worst live/oracle-gap games
  (sc25 oracle=6, lf52 oracle=10, bp35 oracle=9, all baseline live=0), frozen Qwen3.5-9B-MTP, 150 LLM calls /
  ~174 actions per game (the LLM-call cap bound before the 400-action budget did — disclosed, not hidden).
  `levels_gained=0` on all three, `error=None` on all three (clean runs, no crashes). `adversarial_verify.py`
  0-flagged. The filter mechanism itself demonstrably ran for real: `frame_change_rejections` 518/596/600
  (the already-trained, live-validated FrameChangeScorer CNN genuinely ranked ~4 candidates per round every
  round) and `dead_end_rejections` 85/0/0 (the zero-cost history filter fired where relevant, e.g. sc25's
  known non-idempotent-reset/state-cycling gotchas, and correctly did NOT fire on games without that
  pattern).
- failure mode: this is a DIFFERENT failure mode from GAP-ARC-INDUCTION-REFINEMENT-NULL's, not the same gap
  restated. The induce-then-plan mechanism fails because the SYNTHESIZED symbolic model is wrong
  (`heldout_accuracy≈0`). The reactive-filter mechanism's per-step filtering signals are demonstrably
  correct/real (they visibly discriminate candidates every round) — but the mechanism is structurally
  **MYOPIC**: dead-end rejection and change-prediction are both LOCAL, single-step signals ("is this action
  a known repeat of nothing happening" / "does this action plausibly change the frame at all"). NEITHER
  signal encodes DIRECTION toward a distant goal. Per the project's own hard-tail characterization
  (`docs/research-notes/arc-improve-bridge-result-2026-06-23.md`): winning sequences on these games are deep
  (13-33+ actions), narrow, and specifically ordered, with no intermediate reward signal a purely local
  filter could climb. The induce-then-plan architecture's `plan_in_model` step was SUPPOSED to supply exactly
  this multi-step lookahead (search inside the induced model to find a sequence reaching the goal) — but the
  induced model is too inaccurate for that search to be meaningful. The reactive-filter architecture,
  meanwhile, gives up lookahead ENTIRELY in exchange for per-step correctness. Neither, as currently built,
  has BOTH a reliable per-step signal AND multi-step goal-directed lookahead at once — that combination is
  the actual missing capability, not "induction" or "reactive filtering" as separate architecture choices.
- missing discriminator: a GOAL-DIRECTED signal usable WITHOUT a per-game hand-crafted `hand_verifier`
  (which only exists for adaptered games, defeating generalization) -- something that estimates progress
  toward whatever the level's win condition turns out to be, learned or inferred FROM the game's own
  observed structure (e.g. object interactions, color-state changes that correlate with prior wins across
  games), not authored per-game. This is the same shape as `GAP-ARCH-GOAL-NOT-VERIFIED` (world-model verifier
  scores dynamics but never the goal predicate) and `GAP-ARCH-NO-HIERARCHICAL-SEARCH` (no subgoal/landmark
  engine wired) — this result is fresh empirical evidence that BOTH of those pre-existing gaps are still
  load-bearing, now demonstrated on the reactive-filter architecture specifically, not just the induce-plan
  one.
- candidate design: (a) combine the two architectures rather than choosing between them — keep the
  per-step verifier-filter (it is real and correct) but ALSO give it multi-step lookahead via a cheap
  n-step rollout search using the filter itself as the step-evaluator (best-first search over filtered
  candidate sequences, not a symbolic engine, not pure greedy either); (b) a general, cross-game LEARNED
  goal-progress signal (the `GAP-ARCH-GOAL-NOT-VERIFIED` fix) that the search in (a) could optimize toward,
  trained on the growing self-play/registry corpus rather than authored per-game.
- priority: high — directly informs the next architectural step after both the induce-then-plan AND the
  naive reactive-filter approaches have now been empirically closed as standalone fixes. `retire_if_same_
  verdict: true` for a bare reactive-filter variant with no lookahead addition — the next attempt in this
  family must add multi-step search/lookahead or a goal-progress signal, not just retune the local filter.

### GAP-ARC-TOOL-LOOP-LOOKAHEAD-BUDGET-INCONCLUSIVE: candidate design (a) built and tested — null, but the
### test almost certainly never reached the depths where the fix would show up
- status: budget_inconclusive — NOT a clean architectural refutation of candidate design (a) from
  `GAP-ARC-REACTIVE-FILTER-MYOPIC` above. The mechanism itself was built correctly (verified via direct
  tracing, not assumed) and genuinely exercises real multi-step search with backtracking, but the tested
  search budget was almost certainly too shallow to reach these games' known winning-sequence depths.
- evidence: `results/outer_loop_tool_loop_lookahead_ab_20260723.json`
  (`carnot.agentic.arc_tool_loop_lookahead.ToolLoopLookaheadSession`, REQ-ARC-WMTE-5828, operator-directed
  2026-07-23: "add real multi-step lookahead, and allow up to 12 tool-calling/REPL turns per decision,
  inspect history, reason, then commit one action"). Real, non-mocked run on the SAME 3 worst live/
  oracle-gap games as REQ-ARC-WMTE-5827 (sc25 oracle=6, lf52 oracle=10, bp35 oracle=9), `error: null` on
  all three, `adversarial_verify.py` 0-flagged (including `random_seed=5828`, a real llama.cpp `/completion`
  seed threaded through per search-node decision and per tool-loop turn — added after the first run of
  this experiment correctly drew a `METHODOLOGY_MISSING` WARN for lacking one). `levels_gained=0` on all
  three, matching BOTH the induce-then-plan baseline AND REQ-ARC-WMTE-5827's bare reactive-filter baseline
  exactly.
- what actually got built and verified working (not assumed): `ToolLoopLookaheadSession` wires the
  tool-calling orientation loop into `arc_solver_kit.OfflineSolver`'s existing best-first search with
  real branching/backtracking — reusing the project's already-tested-across-25-games engine, not a new
  algorithm. THREE distinct, non-obvious integration bugs were found and fixed via direct empirical
  tracing during construction: (1) `OfflineSolver` calls `apply()` with `frame=None` during search
  expansion (`arc_solver_kit.py:5275,5329` — only `_replay()`'s own top-level call passes the real frame),
  requiring the pre-expansion frame to be captured in `action_labels()` instead; (2) `_replay()` re-applies
  `warmup_label` on every node visit and every sibling-restoration replay, and a naive fix comparing
  applied-action VALUES against the warmup action silently swallowed genuine model choices that happened
  to pick the same action id as the warmup step (found directly: an early attempt's `recent` history came
  back completely empty) — fixed with a non-JSON sentinel string (`WARMUP_LABEL`) that can never collide
  with a genuine `_json_action_label()`-encoded candidate; (3) THE search-starving bug: when the tool
  loop's only proposed candidate was a genuine no-op (observed directly: a click on an empty background
  cell), its resulting state hashed IDENTICAL to the parent and correctly never reached the search
  frontier — with no alternative offered, the search died after exactly one node regardless of the
  12-turn/15-node budgets. Fixed by always padding a thin (<2) candidate set with `rich_action_candidates()`
  structured fallbacks at low confidence. Verified via three successive diagnostic trace runs, the last
  showing the search genuinely using its full node budget with distinct state hashes reached at multiple
  depths.
- **why this is budget-inconclusive, not a clean negative (the important caveat):** `states_expanded` came
  back 16/16/18 and `tool_loop_calls` 4/4/5 across the three games — i.e. the ENTIRE search tree, across
  all branches, visited on the order of 16-18 nodes total. Per the project's own hard-tail
  characterization (`docs/research-notes/arc-improve-bridge-result-2026-06-23.md`), winning sequences on
  these exact games are deep (13-33+ actions) and narrow. A search that only ever expands ~16-18 nodes
  total, most of them near the root, has essentially no chance of reaching a node 13+ actions deep even
  with correct branching and a genuinely helpful confidence signal — the tree is far too shallow for the
  claimed mechanism to plausibly show its effect at this budget. This is a materially different, more
  honest conclusion than "tool-calling + lookahead also doesn't help": the correct statement is "this
  specific budget was too small to test the hypothesis," not "the hypothesis was tested and failed." Each
  search-node decision costs up to 12 real LLM completions, so scaling the node budget by even 5-10x
  (bringing total nodes into the hundreds, closer to plausibly reaching depth 13+) costs a proportional
  5-10x more wall-clock/LLM calls per game — a real, disclosed cost, not free to just "turn up."
- missing discriminator: unchanged from `GAP-ARC-REACTIVE-FILTER-MYOPIC` above — a genuine, cross-game
  goal-progress signal not authored per-game. This experiment supplied the LLM's own self-reported
  confidence as a stand-in, which IS a real (if noisy) per-node signal, but the shallow-budget result here
  cannot yet say whether that signal is good enough to guide a properly-scaled search, because the search
  never ran deep enough to find out.
- candidate design: run the SAME mechanism with a substantially larger node budget (target: enough nodes
  to plausibly reach 13+ action depth given the observed branching factor — likely low hundreds of nodes,
  not 15) on at least one of these three games, so the honest answer becomes "tested at a depth where the
  hypothesis could show up" one way or the other. This is a budget/cost question for the operator to weigh
  (up to 12 LLM calls per node), not a further architecture change — the mechanism itself does not need
  more work before that test, per the three bugs above already being fixed and verified.
- priority: high — this is the direct, cheap-to-run follow-up to `GAP-ARC-REACTIVE-FILTER-MYOPIC`'s
  candidate design (a), and the current result neither confirms nor refutes it. `retire_if_same_verdict`
  does NOT apply here in the usual sense (this was not a doomed-rerun scenario) — a larger-budget rerun of
  the SAME mechanism is the correct next step, not a rejected repeat, because the small-budget run was
  never a fair test of the hypothesis in the first place.

**UPDATE 2026-07-23, larger-budget follow-up run (operator-directed: "run that larger budget test"):**
`results/outer_loop_tool_loop_lookahead_ab_largebudget_20260723.json`
(`scripts/arc_tool_loop_lookahead_ab_largebudget.py`, `max_nodes=300` — 20x the first run's 15,
`depth_cap=40` — up from 15, `random_seed=58280`, a genuinely different seed, not an extension of the
first run's trajectory). `error: null` on all three, `adversarial_verify.py` 0-flagged, `levels_gained=0`
on all three — still an honest negative. Per-game results are now DIFFERENTIATED, not a uniform shallow
result, and each tells a different story:
- **sc25: `states_expanded=95` out of the 300-node budget — the search CONVERGED (exhausted its own
  reachable frontier under the dead-end pruner) rather than being budget-capped.** This is a materially
  stronger negative than "budget-inconclusive" for sc25 specifically: the mechanism explored everything it
  could discover from this starting state and found nothing, so more budget alone would not have helped —
  the discriminator gap (no real goal-progress signal) is the binding constraint here, not search depth.
- **bp35: `states_expanded=302` — used the FULL budget without converging**, i.e. still genuinely
  budget-bound, not frontier-exhausted. Scaling the budget further remains a live, untested lever for this
  specific game.
- **lf52: `states_expanded=3`, `tool_loop_calls=2` — anomalously SHALLOWER than the first (15-node-budget)
  run's 16.** Reported honestly as observed, not explained away: this is surprising and NOT yet diagnosed
  (candidate causes not yet distinguished: the dead-end pruner firing unusually aggressively near lf52's
  start state, a genuinely very constrained local neighborhood before any productive action becomes
  available, or something lf52-specific interacting with the tool loop's own convergence — `tool_loop_calls
  =2` means the model committed almost immediately both times, so this is NOT the earlier no-op-starves-
  search bug recurring, since the fallback-padding fix demonstrably still fires here). Flagged for a
  follow-up look, not resolved.

**Recalibration that changes how "conclusive" should be read here (found while preparing this run, not
known when the first budget was picked):** `ops/arc_solve_registry.yaml` shows all three games are
`full_game_clear: true` — but ONLY via hand-crafted, per-game `GameAdapter`s built across many outer-loop
sessions of deliberate reverse-engineering (`development_proxy` provenance), e.g. lf52's registry entry
cites a 146-action L3 sequence and a 927-action L8+ probe. A domain-agnostic mechanism attempting level 1
from raw pixels alone, with zero game-specific hints, is not attempting a small problem — even a
20x-larger budget is not provably sufficient for these SPECIFIC games' actual difficulty. sc25's
frontier-exhaustion result is the one clean, unambiguous negative in this update; bp35 and lf52 remain
open in different ways (bp35 genuinely budget-bound, lf52 anomalously unexplored).

**Live-path wiring gap found and fixed the same session (operator question: "why did we only wire that
into offline and not live agent?"):** `arc_tool_loop_lookahead.py` was reachable from NEITHER canonical
live entrypoint (`scripts/arc_orphan_solver_lint.py`'s own closure check confirmed this) — not even the
offline dev-twin (`scripts/arc_loop_solve.py`), only from its own standalone A/B script. No principled
reason for this; it was simply never plumbed through. Fixed: `solve_via_tool_loop_lookahead()` added to
`arc_loop_solve.py` as a `--mechanism tool_loop_lookahead` alternative to the existing `solve_via_explore`
(graph_explore) adapter-free first-contact strategy, dispatched from `main()`, real end-to-end smoke-tested
(`--game sc25 --mechanism tool_loop_lookahead --ignore-adapter`, clean run, correctly fell through to
`needs_re()` on no advance). `arc_orphan_solver_lint.py` passes clean post-fix. Wiring into the SCORED
`E3AgentPolicy` cascade itself was deliberately NOT done in this pass: the mechanism is still unproven
(every measurement to date is null), and it costs up to 12 real LLM calls per decision — adding an
unproven, expensive mechanism to the actual scored path risks hurting the live agent's efficiency metric
(RHAE) for no demonstrated correctness benefit. The offline dev-twin wiring is the correct sequencing per
the Phase-Prototype-and-Validation discipline; scored-path wiring is the right NEXT step only once a
result here actually proves out.

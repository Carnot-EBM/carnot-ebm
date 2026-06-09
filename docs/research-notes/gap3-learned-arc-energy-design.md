# GAP-3 Design: a learned / model-native ARC energy (the verifier that reaches the headroom)

**Status:** design (operator chose "design GAP-3 first" 2026-06-09). No code yet; this doc is the
prototype + empirical-criteria + adversarial-check trio mandated by the Phase Prototype + Empirical
Validation discipline (CLAUDE.md). It spawns experiments; the OpenSpec REQ/SCENARIO entries get added
when the first GAP-3 experiment is implemented (anchored to the `verification` / `research-harnesses`
capabilities).

**Audience:** an engineer who has not followed the ARC verifier thread. Read `ops/verifier_gaps.md`
first for the gap ledger; this doc is GAP-3's build plan.

---

## 0. Why a learned energy, not another hand-invariant (the evidence)

Carnot's product is the **verifier**: given a generator's candidate outputs, select the correct one
cheaply so the generator is invoked less (the ARC-AGI-3 efficiency axis). We tested the existing Carnot
hand-invariant ensemble as a re-ranker on TRM (a SOTA open ARC-AGI-1 reasoner) — does it select TRM's
correct-but-mis-voted answer where TRM's own augmentation-frequency vote mis-picks it?

The answer, triple-confirmed on TRM's real candidate pool (n=31; oracle headroom is real: pass@1000
0.61 > vote 0.45, so the correct answer IS in the pool ~13pp more often than vote captures it):

| approach | pass@2 | artifact |
|---|---|---|
| TRM_VOTE (baseline to beat) | **0.452** | `results/arc3_trm_verifier_rerank.json` |
| union_max hand-invariant ensemble | 0.194 | (anti-ranks: max() takes each candidate's worst family) |
| best fixed re-aggregation (min) | 0.419 | `results/arc3_verifier_antirank_diagnosis.json` |
| best learned linear combiner (LOTO logistic, OOF) | 0.226 | `results/arc3_verifier_learned_combiner_ceiling.json` |
| oracle ceiling (target) | 0.613 | — |

The diagnosis (`arc3_verifier_antirank_diagnosis.json`) showed the structural reason: the only
strongly-discriminative hand-families (`tiling_scaling` AUROC 0.91, `color_mapping` 0.71) fire on <20%
of tasks; the always-on families top out at `object_count` 0.67 — insufficient to seat gold in top-2
against dozens of structurally-plausible candidates. A learned combiner that *correctly* down-weights
the anti-discriminative families (`v1` −0.52, `palette_histogram` −0.23) still cannot beat vote. **Cheap
hand-features are exhausted.** Only a content/rule-aware *learned* energy can reach the proven headroom.

GAP-1 (transpose/orientation) and GAP-2 (variable-dim) are both hand-invariants → the same wall. GAP-3
is the learned energy that subsumes them.

---

## 1. Capability requirement (falsifiable)

**REQ-GAP3-1 (selection).** A verifier energy `E(candidate | input, train_pairs)` — LOWER = more
likely the correct rule-application — that ranks TRM's correct candidate into top-2 **more often than
TRM's frequency vote**, on a **held-out** task split, **without ever reading the test gold output**.

- **Primary acceptance gate:** `pass@2(energy) > pass@2(TRM_VOTE)` on held-out tasks where the correct
  answer is present in the candidate pool. Current bar: 0.45 at n=31; **the bar is re-measured at full
  400-task scale before any "GAP-3 filled" claim** (n=31 is directional only).
- **Stretch target:** approach the oracle ceiling (0.61 @ n=31), i.e. capture the full present-but-
  mis-voted headroom.

**REQ-GAP3-2 (no-oracle invariant).** The energy MUST be computable from `(candidate, test_input,
train_pairs)` and the generator's own state only. It must never see the held-out solution. This is the
load-bearing honesty constraint (the entire rerank program is oracle-free).

**REQ-GAP3-3 (coverage).** The energy must be DEFINED (non-abstaining) on ≥80% of tasks. The structural
failure of the hand-features was coverage: the discriminative families fired on <20% of tasks. A learned
energy over a dense representation (activations or grid pixels) is always defined — this requirement
makes the coverage win explicit and measurable.

---

## 2. Three staged approaches (increasing complexity & compute cost)

Staged deliberately so the cheapest stage runs on data we ALREADY have, and each stage's result decides
whether the next is worth its compute.

### Stage 0 — TRM-native halting-confidence energy (NO new GPU; data already dumped) — DONE: NEGATIVE (2026-06-09)

> **RESULT (2026-06-09): NEGATIVE, adversarially confirmed → advance to Stage 1.**
> `results/arc3_gap3_stage0_qhalt_energy.json` + `..._adversarial_verify.json`. Scalar q_halt does NOT
> beat frequency vote (Q_MEAN pass@2 0.290 < vote 0.452; vote-residual collapses to 0.097; bootstrap
> CI entirely <0; `headroom_capture_fraction=0`). 5-reviewer round = unanimous NEGATIVE_CONFIRMED. The
> 0.86 within-task soft-AUROC says the signal EXISTS in TRM's confidence but the 1-D scalar projects it
> away → Stage 1 (full latent) is the GO. Original design text follows.


The TRM eval dump (`eval_out/arc_v1/step_0_all_preds.*`) already contains `q_halt_logits` (shape
`(25600,)`, float32): TRM's own per-augmentation halting confidence. This is a *model-native* scalar we
have for free.

- **Energy:** aggregate `q_halt` per de-augmented candidate (mean / max / logsumexp over the
  augmentations that produced it) → `E = -agg_q_halt` (more confident = lower energy). Rank by it.
- **Cost:** zero GPU; it's another ranker on the existing offline dump (like the diagnosis).
- **What it answers:** does TRM's own confidence carry selection signal BEYOND frequency vote? TRM's
  published pipeline uses frequency vote and IGNORES q_halt for selection (the halting head is used to
  stop refinement, not to vote). If q_halt re-ranks better than vote, that is a free model-native win
  AND direct evidence that the richer latent (Stage 1) will help.
- **The catch (→ adversarial check A0):** `q_halt` may be collinear with vote-count (confident
  augmentations cluster on the popular candidate). The lift must be measured OVER vote, not in absolute.

### Stage 1 — TRM penultimate-activation energy (needs an activation-dump re-run; GPU)

The scalar `q_halt` is a lossy projection of TRM's latent state. The full signal is the penultimate
latent `z` (the recursive-refiner's carry state before the output head), per augmentation.

- **Build:** add a forward hook in the eval harness to dump `z` alongside `preds` (extend
  `cfg_dict["eval_save_outputs"]`; may require exposing the carry in TRM's `evaluate()` — a harness
  change, not a TRM retrain). One GPU eval pass (~1hr, conductor pause).
- **Energy, two variants:**
  - **(1a) model-native basis** (arXiv:2604.17614 "Characterizing Model-Native Skills"): recover a
    compact ORTHOGONAL basis from the pooled `z` activations on held-out tasks; score a candidate's `z`
    consistency along it. No new training — basis recovery is SVD-class. The paper's thesis: the
    generator's own activation basis discriminates correctness better than external ontologies (exactly
    our hand-invariants).
  - **(1b) learned probe energy** `E(z)`: a tiny MLP trained out-of-fold on held-out tasks to map `z` →
    correctness. Bigger than (1a) but directly optimizes the gate.
- **Why it should work where hand-features failed:** `z` is dense (defined for every candidate →
  coverage gate satisfied by construction) and carries the model's internal "is this the rule"
  computation, which the discrete grid output hides.

### Stage 2 — Trained ARC transition-EBM (biggest; GPU + ARC corpus; generator-independent)

A from-scratch energy `E(input, candidate_output | train_pairs)`, low when the candidate is the correct
rule-application — the "new ARC energy instance" the domain-bound analysis calls for
(`project_verifier_domain_bound`).

- **Build:** a small CNN/transformer over the (input grid, candidate grid, train-pair context),
  trained on ARC transitions (correct vs perturbed/wrong outputs) on held-out tasks. GPU training.
- **Trade-off:** most general (works for ANY generator, not just TRM — the true Carnot verifier asset)
  but most expensive and most prone to overfit the ARC training distribution. Pursue only if Stage 1's
  model-native energy underperforms or we need generator-independence.

---

## 3. Empirical gates (per stage — every stage must pass ALL before it is called "working")

| gate | threshold | guards against |
|---|---|---|
| **selection** | `pass@2(energy) > pass@2(vote)` on held-out tasks | the whole point — beating the baseline that every hand-feature lost to |
| **discrimination** | per-candidate energy AUROC(gold vs non-gold) > 0.70 | beating the hand-features' best always-on family (object_count 0.67) |
| **coverage** | energy defined on ≥80% of tasks | the structural killer of the hand-features (<20% coverage on the good families) |
| **headroom-capture** | capture ≥ 30% of `(oracle − vote)` | partial credit toward the 0.61 ceiling; a non-trivial dent |
| **no-oracle audit** | feature pipeline provably never reads test gold | REQ-GAP3-2; the honesty invariant |

All gates are first measured on the n=31 capped pool (fast iteration), then **re-confirmed on a full
400-task TRM dump** before a `status: filled` claim in `ops/verifier_gaps.md`.

---

## 4. Adversarial checks (hostile-reviewer round — run BEFORE scaling, per the discipline)

Each attack tries to make a stage PASS the gates without actually working. The check is the
instrumentation that detects it.

- **A0 — vote-mimicry (Stage 0 critical).** Does `q_halt` only re-rank well because it's collinear with
  vote-count? *Check:* partial correlation of `q_halt` with correctness controlling for vote; and a
  "residual q_halt" ranker (regress out vote, rank by the residual). If the residual carries no signal,
  Stage 0's lift is just frequency in disguise → not model-native.
- **A1 — degenerate basis / activation shortcut (Stage 1).** Could the `z`-basis be reading augmentation
  identity, grid size, or position rather than rule-correctness? *Check:* permutation control — shuffle
  the correctness labels within augmentation-strata; a real energy collapses to chance, a shortcut
  survives. Also verify the basis discriminates ACROSS tasks it was not fit on.
- **A2 — task leakage / overfit (all learned stages).** *Check:* strictly out-of-fold / held-out-task
  evaluation; the energy is fit on a disjoint task set from where pass@2 is measured. Report both
  in-fold and OOF; a large gap = overfit.
- **A3 — oracle leak (all stages).** *Check:* audit that no feature derives from the test solution;
  assert the energy is identical whether or not the solutions file is present.
- **A4 — sample-size mirage.** n=31 with 19 gold is tiny; a 1-task swing is ~3pp. *Check:* bootstrap CI
  on the pass@2 delta vs vote; re-confirm at 400 before any irreversible claim (FALSE_NEGATIVE_RISK +
  sample-size rigor). A positive control already exists (oracle 0.61 > vote 0.45 proves selectable
  headroom), so a negative here is informative, not degenerate.

---

## 5. Recommended sequencing

1. **Stage 0 first (no GPU, runnable now).** Cheapest possible model-native probe on data we have. Its
   A0 check (vote-mimicry) is the pivotal question: is there ANY model-native signal beyond frequency?
   - If Stage 0 beats vote with real residual signal → strong green light for Stage 1.
   - If Stage 0 ties/loses → the scalar q_halt is too lossy; Stage 1's full latent is required (the
     scalar projecting away the signal does NOT mean the latent lacks it).
2. **Stage 1 (model-native basis 1a before probe 1b).** The cheaper basis-recovery first; only train the
   probe if the basis underperforms.
3. **Stage 2 only if** generator-independence is needed or Stages 0–1 underperform.

Stop-and-report after each stage (the operator stays in the loop; no autonomous scale-up).

---

## 6. Decentralization (CLAUDE.md rule 1/5 — local-first, sovereignty)

All three stages are sovereignty-clean: Stages 0–1 read **local** TRM weights/activations (open
checkpoint, on this box); Stage 2 trains a **local** EBM on the open ARC corpus. No closed-weight vendor
is in any path. The model-native approach is in fact MORE sovereign than the hand-invariants because it
needs nothing but the open generator we already run.

---

## 7. Definition of done (GAP-3 → `status: filled`)

`ops/verifier_gaps.md` GAP-3 flips to `filled (<verifier_id>)` when one stage:
(a) passes all §3 gates on the **400-task** scale, (b) survives the §4 adversarial round, (c) is
registered in `ops/verifier_registry.yaml`, and (d) the energy is wired into the rerank harness as a
deployable, oracle-free ranker. Until then GAP-3 stays `open` and each stage's result is appended to the
ledger (never-prune).

---

## Cross-references

- `ops/verifier_gaps.md` — the gap ledger (GAP-1 refuted, GAP-2 open, GAP-3 this doc)
- `results/arc3_trm_verifier_rerank.json` / `..._antirank_diagnosis.json` / `..._learned_combiner_ceiling.json`
  — the evidence chain that makes GAP-3 the confirmed path
- `reference_model_native_skills` memory (arXiv:2604.17614) — Stage 1a basis method
- `project_verifier_domain_bound` — Stage 2 the domain-bound "new ARC energy instance"
- CLAUDE.md "Phase Prototype + Empirical Validation + Adversarial Check Discipline" — the trio this doc
  instantiates
- CLAUDE.md "Adversarial Artifact Verification + Sample-Size Rigor" — the §4/§A4 guards

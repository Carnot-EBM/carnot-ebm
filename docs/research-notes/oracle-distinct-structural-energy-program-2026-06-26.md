# Oracle-Distinct Structural Energy on ARC — the staged build that unlocks "extend with energy"

**Date:** 2026-06-26 · **Author:** outer-loop (Claude Opus 4.8), operator-directed.
**Operator directive (2026-06-25):** *"Ignoring the deadline, we must pursue the only direction
that would unlock 'extend with energy' on ARC with oracle-distinct structural energy."*
**Provenance:** map→design→adversarial-verify workflow `wf_f046ff34-c0a` (12 agents, 2.1M tokens):
5 record-maps → 3 independent designs → judge-synthesis → 3 adversarial refuters
(**refuted_count=3** — all three refuted the *first experiment as drafted*; none refuted the
direction; the corrections are folded in below). This note supersedes the earlier
energy-as-RFT-teacher recommendation (RETIRED — see the levers ledger DO-NOT-RE-OPEN).

---

## 0. Why this is the only direction left (settled, not re-derived)

Every prior energy attempt on ARC nulled for ONE shared root cause: the energy is computed over
**frame-marginal features, not game-agnostic STRUCTURE**. The discriminative verifier's cross-game
leave-one-game-out (LOO) AUROC on frame-marginals is **0.503 == chance** (v1/v2,
`models/arc_discriminative_verifier_v2.json`). "Extend with energy" stays at chance until an
energy is built over structure that transfers across games (the **GAP-ARCH-FEATURES** gate,
`ops/verifier_gaps.md:2360`). This is the single open strategic item in
`ops/north-star.md` §5 and the only direction not already nulled-and-retired.

## 1. The foothold (the record already crossed chance — on the WRONG target)

`exp4545` (`results/experiment_4545_cross_game_discrimination_v3.json`) is the first non-circular
above-chance cross-game result: adding four **structural** feature families to the frame-marginal
baseline lifts cross-game **LOO AUROC 0.503 → 0.674, CI [0.606, 0.745] EXCLUDING 0.5**, with
`verifier_is_oracle=False`. Per-family ablation is the decisive evidence of *which* structure
transfers:

| feature set | LOO AUROC | reading |
|---|---|---|
| v2 frame-marginal (41 feats) | 0.494 | chance — the documented root cause |
| + action_conditioned | 0.512 | ~dead |
| + predicate_distance | 0.517 | ~dead |
| + object_relational | 0.620 | **carries transfer** |
| + frame_delta | 0.666 | **biggest single lever (+0.172)** |
| v3 full (82 feats) | 0.674 | CI excludes 0.5 |

**So object-relational + frame-delta structure is what flips the gate; pure frame-marginals and the
action/predicate context do not.** This is the spine to build on (`arc_value_learner.py:433`
`cross_game_features_v3`).

**The catch this program must beat:** exp4545's 0.674 is on a **win-reachability** label
(reachable-state vs dead-end). exp4700 proved win-reachability is *unselectable* even at coverage
1.0 — a selector signal, not a generation signal. The bet of THIS program is that the same
structural representation discriminates the **right** target — **held-out transition-correctness**
— cross-game, and that *that* signal can be wired to put a winner INTO the explorer's pool
(generation), not merely re-rank a pool that already contains it.

## 2. The two killers every stage must survive (from the adversarial pass)

1. **Circularity / oracle-dependence.** The energy must NEVER call the env win-check, run
   `is_level_complete`/unit-tests, or be trained on ground-truth-corrupted next-grids (that needs
   the observed next-grid = the executable oracle, and makes the head a corrupted-vs-real
   *provenance* discriminator — the documented `.393/GAP-3 leak). `verifier_is_oracle=false`,
   must pass `check_circular_moat_overclaim`.
2. **Frame-marginal collapse / no transfer.** Features must be functions of the (s,a,s')
   TRANSITION and of cross-frame object correspondences (differences and relations), never
   absolute color/position — so they cannot collapse to the 0.503 marginals, and they must clear
   chance specifically on **real near-miss** negatives (≤5% cells changed), because GAP-3 proved
   *synthetic* corruptions are aced while real near-misses (91.5% of real errors) score below
   chance.

## 3. The corrected staged program

### S0 — Core-bet probe (1 day, no GPU, no LLM, decisive; retires the direction cheaply if null)

**The fix the refuters converged on:** do NOT corrupt the ground-truth next-grid (oracle-dependent
+ provenance-leaky + a GAP-3 rerun). Instead define transition-correctness via the agent's OWN
induced engine's **held-out generalization** — the oracle-distinct, live-computable target
(`exp4604 world_model_trust_energy`, `verifier_is_oracle=False`):

- **Dataset (live-reproducible, oracle-free):** from the existing banked transitions, fit an
  induced engine on a prefix and evaluate it on **held-out** (not-fit-on) off-path transitions.
  - **positive** = (s,a,s') where the induced engine correctly predicts the held-out s'.
  - **negative** = (s,a,ŝ') where the engine **mispredicts** — a REAL near-miss (engine error,
    not a synthetic teleport/vanish corruption). This is what the live agent actually sees
    off-path; no ground-truth corruption, no env win-check.
- **Features:** ONLY the proven-transferring structural families (`object_relational`,
  `frame_delta` from `cross_game_features_v3`), computed over (s, predicted-s', real-s'). Exclude
  the dead families (action_conditioned, predicate_distance) and all frame-marginals.
- **Gate (cross-game leave-one-GAME-out, the only non-circular evidence):** LOO AUROC on the
  real-near-miss transition-correctness label, with bootstrap CI95, must clear **BOTH**:
  (a) TRUE chance 0.5 (CI95 lower bound > 0.5 — note 0.5442 is NOT chance, it is the recorded
  AUROC of the retired GAP-3 stage-2 EBM; we must beat it as a *harder ceiling*, not treat it as
  the floor), and (b) the v2 frame-marginal control on identical folds/seeds (structural−marginal
  delta CI95 excludes 0).
- **Anti-single-lever:** per-family ablation must show ≥1 genuinely-structural family
  (object_relational or a conservation term) independently clearing **0.55 LOO** — so the lift
  is not all frame_delta (which is near-marginal: changed-fraction + centroid-shift).
- **Leak audit (mandatory):** an origin-probe (induced-vs-real classifier on the same features)
  must score AUROC < 0.6 — else the head is a provenance discriminator, not a correctness one.
- **Positive control:** in-sample AUROC > 0.60 (else harness broken, the LOO null is uninformative
  per the FALSE_NEGATIVE_RISK guard).
- **`prior_failures` (MANDATORY — GAP-3 stage2/stage2v2 is on `ops/exclusion_manifest.yaml`):**
  cite `arc3_gap3_stage2_transition_ebm` (macro-AUROC 0.5442, retired 2026-06-09). What is
  different: (1) **real induced-engine-misprediction near-miss negatives on LIVE game (s,a,s')
  transitions**, not synthetic-corruption negatives on ARC-1 puzzle candidate grids; (2)
  **held-out-generalization target**, not win-reachability; (3) structural features only.
  `retire_if_same_verdict: true` — if the structural head's CI95 includes 0.5 OR the
  structural−marginal delta CI95 includes 0 OR the origin-probe leaks, the **entire
  energy-guided direction retires** (one day of cost). `inference_substrate:
  verifier_ensemble_against_cached_candidates`.

### S1 — Contrastive ENERGY landscape (not a classifier) + multi-seed hardening

Promote the S0 logistic into a contrastively-trained energy E(s,a,s') (NCE/margin on
real-vs-near-miss pairs) so search can DESCEND −ΔE. **Gate:** cross-game LOO transition-correctness
ranking AUROC ≥ **0.70**, CI95 excluding chance, across ≥10 seeds; **≥2 independent
non-frame-delta families** each individually clearing 0.60 LOO (kills the single-lever risk);
energy passes a denoising-direction test the point-classifier fails. **Kill:** < 0.70 after 10
seeds, OR lift collapses to frame_delta alone.

### S2 — Off-path TRUST gate on induced world-models (the oracle-distinct moat slot, live-path-reachable)

Graft E into the LIVE Family-B loop — wired into `WorldModelVerifier`/`E3AgentPolicy` import
closure (NOT an orphan `scripts/experiments` solver; `arc_orphan_solver_lint`). Replace the binary
accuracy<0.5 cutoff with an E-ranking of candidate induced engines by **off-path** structural
energy (where the executable oracle is silent — the oracle-distinct headroom). **Gate:** on
held-out hidden-state games, the energy-selected engine has higher held-out off-path transition
`cell_recall` than the binary-accuracy-gate-selected engine, CI95 excluding 0. **Control:** the
incumbent binary `WorldModelVerifier` (`verifier_is_oracle=TRUE`) on identical engine sets.
**Kill:** energy-ranked selection does not beat the binary gate on held-out off-path cell_recall.

### S3 — GENERATION lift (the second killer): the energy puts a winner INTO the pool the explorer never proposed

Wire E as `goal_energy=` into `plan_in_model()` + the S2 trust gate on the live E3AgentPolicy path.
On held-out games whose winner the order-1/object-centric explorer does NOT enumerate (the exp4700
r11l class: winner-rank deep 59/161/12/77 of 186, 0 banked despite coverage 1.0), run the E-guided
induce→verify→plan loop offline. **Gate:** on ≥5 held-out games where the bare explorer banks 0
levels, the E-guided loop banks ≥1 NEW offline-reproduced level
(`arc_solver_kit.reproduce`, `offline_reproduced=true`, `solve_provenance=live_agent_self_discovery`)
the matched λ=0 control does NOT; decisive metric = fraction of winners NEWLY entering the pool.
**Kill:** guidance does not increase the rate of winners newly entering the pool (CI95 includes 0),
OR every banked level was already in the bare explorer's pool (re-ranking, not generation).

### S4 — Cross-GAME/FAMILY transfer survival + live solve-rate confirmation (the moat / publication gate)

Stress S2+S3 exactly where prior wins died (the `.393 oracle-distinct win did NOT transfer; GAP-4
collapsed ARC-1→ARC-2 0.93→0.57). Re-run S1 LOO + S3 generation lift on a held-out FAMILY of games
disjoint from training (recolor/trail/no-anchor lp85/tu93/r11l vs movement cn04/sp80), multi-seed,
leak-clean. **Gate:** cross-FAMILY held-out LOO transition-correctness AUROC > 0.65, CI95 excluding
chance (degrades <0.10 from in-family, vs the `.393 catastrophic 0.36 drop); S3 lift reproduces on
≥2 held-out-FAMILY games multi-seed with a λ=0-fails / λ>0-banks ablation; beats the prior best
Carnot live run + TRM baseline. **Kill:** cross-family LOO collapses (transfer failure like
`.393/GAP-4 — the energy is a learned-value in disguise), OR the generation lift does not reproduce
held-out-family.

## 4. Honest base rate (the risk that this nulls like every prior energy attempt)

The `.393 oracle-distinct selector win did not transfer; GAP-4 collapsed 0.93→0.57; the GAP-3
transition-EBM scored 0.5442 (subset of vote) and is retired; DiffusionGemma's CODILA control
did not differentiate guided-vs-unguided (`v404). This program is a real, falsifiable attempt with
the failure modes named — but the honest prior is that structural energy may itself null. The value
is that **S0 retires the whole direction in one day if the core bet is false**, and each stage has
a numeric kill criterion. We are not betting multi-week effort before the cheap decisive probe.

## 5. Cross-references

- `ops/north-star.md` §5 (the oracle-distinct moat — the only open strategic item)
- `ops/verifier_gaps.md` GAP-ARCH-FEATURES (the gate) + GAP-ARCH-VERIFIER-REGRESSION-ONLY
- `results/experiment_4545_cross_game_discrimination_v3.json` (the foothold: LOO 0.674 CI-excl-0.5)
- `python/carnot/agentic/arc_value_learner.py:433` (`cross_game_features_v3` — the structural spine)
- `results/experiment_4604_world_model_trust_energy.json` (the oracle-distinct live-computable seam)
- `results/arc3_gap3_stage2_transition_ebm.json` + `ops/exclusion_manifest.yaml:485` (the RETIRED
  rerun S0's `prior_failures` must cite)
- `docs/research-notes/diffusiongemma-energy-guided-diffusion-spec.md` (THE GATE — STILL-PENDING)
- `docs/research-notes/arc-agi3-levers-tried-x-verdict-2026-06-25.md` (the levers ledger;
  DO-NOT-RE-OPEN energy-as-RFT-teacher + transition-EBM)
- CLAUDE.md "Circularity / Oracle-Distinctness Discipline" + "Failed-Experiment Rerun Discipline"
- workflow `wf_f046ff34-c0a` (this program's provenance; refuted_count=3, corrections folded)

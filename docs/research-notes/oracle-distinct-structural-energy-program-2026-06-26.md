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

## RESULTS — S0 (leaky→retired) then S0' (origin-matched→REOPENS to S1)

> **STATUS (2026-06-26): the direction is ALIVE. S0' passed every gate; S1 is authorized.**
> (Recorded here durably because the conductor's stale in-process linter TAUTOLOGY-flagged the
> S0' artifact on a chance-floor false-positive, so the `.439 capstone will skip it — the true
> result lives in the exp4771 artifact + this note.)

- **S0 (`.438 A1, exp4761):** structural cross-game LOO **0.746** (CI [0.660, 0.831] excl chance),
  +0.28 over frame-marginals, frame_delta 0.725 + object_relational 0.661 both >0.55, 92% real
  near-misses — the strongest ARC energy signal yet — BUT the leak audit FAILED (origin-probe
  **0.733**) because positives were REAL recordings and negatives were INDUCED mispredictions
  (origin confounded with correctness). Retired per the pre-registered gate as a *fixable confound*.
- **S0' (`.439 A1, exp4771) — origin-matched (both classes induced):**
  `honest_verdict: success_structural_energy_s0prime_reopens_s1`, `verifier_is_oracle: false`.
  - origin_probe_auroc **0.733 → 0.500** (origin-matching removed the leak — the decisive test)
  - shuffled_label_control_auroc **0.503** (no residual nuisance leak)
  - loo_auroc_structural **0.739**, CI **[0.636, 0.833]** — still excludes chance with origin matched
  - structural − frame-marginal delta CI **[0.127, 0.332]** — excludes 0 (marginals 0.511 = chance)
  - per-family: frame_delta 0.739, object_relational 0.660 — both independently clear 0.55
  - in-sample 0.848 (positive control passes); 16/18 games contribute both classes to the LOO
  - `retire_energy_guided_direction: False`

**Conclusion.** The S0 0.746 was **NOT an origin leak** — when origin is matched (probe driven to
exactly chance), the structural energy STILL discriminates held-out transition-correctness
cross-game at 0.739, oracle-distinct and leak-controlled. This is the **first clean,
leak-controlled, oracle-distinct cross-game energy signal on ARC** — "extend with energy" is no
longer at chance. **S1 (the contrastive energy landscape) is now authorized** per the staged gate;
pre-stage it for `.440.

- **S1 (`.440 A1, exp4781) — PASSED; the energy is now a usable LANDSCAPE → S2 authorized.**
  `honest_verdict: success_structural_energy_s1_landscape_authorizes_s2`, NOT flagged (live re-check
  clean). The contrastive energy E(s,a,s') (margin/NCE over the S0' origin-matched induced-correct vs
  induced-wrong pairs):
  - energy-ranking LOO AUROC **0.713** across **10 seeds**, CI **[0.7133, 0.7137]** (excludes chance,
    very tight multi-seed — the robustness S0' single-seed did not establish)
  - **denoising-direction agreement 0.622** — −ΔE descent points toward correctness (the energy-vs-
    classifier distinction; a point classifier cannot pass this)
  - leak controls hold under contrastive training: origin-probe **0.500**, shuffled-label **0.493**
  - per-family: frame_delta 0.726, object_relational 0.660 — both ≥ 0.60 (no single-lever); marginal
    control 0.484 (structure +0.23 over marginals); in-sample 0.823 (positive control passes)

**Two consecutive headline successes (S0' reopen → S1 landscape).** The structural energy is real,
leak-clean, multi-seed-robust, and forms a descent landscape. **S2 (the off-path trust gate on
induced world-models — the FIRST live-path-reachable stage, where the energy starts adding value to
the actual agent) is now authorized** per the staged gate; pre-stage it for `.441.

- **S2 (`.441 A1, exp4791) — INCONCLUSIVE (a DEGENERATE non-test), NOT bounded.** S2 reported
  `energy_minus_accuracy_delta = 0.0` (CI [0,0]) and a `no_live_trust_value` verdict — but the
  candidate engine pools were behaviorally degenerate: 2/5 held-out games had bit-identical candidate
  predictions (same off-path energy + cell_recall), a 3rd had equal recalls, so only **2/5 games**
  genuinely tested the selection (and in those the energy agreed with the accuracy gate). The 0-delta
  is an artifact of non-diverse candidates, not evidence the energy lacks value. **The operator caught
  the over-claim ("are we sure about S2?").** Outcome: (1) a `DEGENERATE_CANDIDATE_POOL` detector
  shipped in `adversarial_verify.py` (`check_engine_selection_candidate_diversity`, adversarially
  hardened to be non-gameable — independent floor, PASS not exempt, meaningful-spread, negative-delta);
  (2) **S2-v2** (`.442 A1, exp4801) re-runs with an ENFORCED behaviorally-diverse candidate pool +
  per-game logging + an effective-games gate + positive-control for any BOUNDED verdict. **The energy
  direction is NOT paused on S2** — it is awaiting the S2-v2 real test. (The `.441 capstone's
  honest_verdict string says "bounded"; that is a mislabel — it correctly SKIPPED the flagged artifact
  and imported no numbers, but the true status is INCONCLUSIVE; the `.442 transition records it as such.)
- **S2-v2 (`.442 A1, exp4801) — GENUINE BOUNDED (the hardening worked; energy does NOT beat cheap
  accuracy at engine selection).** A real test this time: **n_effective_games = 5/5** (the enforced
  diverse pool), `positive_control_passed=True` (the pool contains a candidate the accuracy gate
  misses — BOUNDED means "energy could have won but didn't"), `candidates_genuinely_induced=True`, and
  the `DEGENERATE_CANDIDATE_POOL` detector does NOT fire (verified live). Result:
  `complete_structural_energy_s2v2_bounded_diverse_pool`, **energy−accuracy off-path cell_recall delta
  = −0.158, CI95 [−0.478, +0.004]** (includes 0, point estimate slightly NEGATIVE). On a genuine
  diverse test the off-path structural energy ranks induced engines **no better (slightly worse)** than
  the cheap execution-grounded accuracy gate. **The energy is a real OFFLINE discriminator (S0'/S1) but
  adds no live value at engine SELECTION.** Per the pre-registered S2 gate, BOUNDED → the next test is
  **S3 (generation lift)** — a DIFFERENT use (does the energy put a winner INTO the pool the explorer
  never proposed) and the actual ARC wall (selection ≠ generation; the ledger's finding is ARC is
  generation-bound). S2-v2 bounded does NOT refute S3, but it IS the first live null — a genuine
  decision point on whether to proceed to S3 or reconsider.

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

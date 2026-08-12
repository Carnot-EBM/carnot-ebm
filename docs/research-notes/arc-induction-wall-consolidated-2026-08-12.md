# The ARC induction wall: what nine experiments settled, and what they did not

**Status:** RESEARCH SYNTHESIS. No live-path file changed by this note. It consolidates
exp6248 through exp6259, run 2026-08-11 and 2026-08-12 under the operator directives
"do all of the plans: P1-P6, unattended", "fix collect_transitions first", and "run the veto
sensitivity A/B".

**Read this first if you read nothing else.** Nine experiments tested levers around
world-model induction. One worked. The rest were null, negative, or retracted. The pattern
across all of them says the binding constraint is how the world model is PRODUCED, and every
lever tested operated on how it is selected, searched, ranked, or filtered instead.

---

## 1. What was tested, and what happened

| # | lever | result |
|---|---|---|
| exp6250 | ensemble: pick the better of two search shapes by VALID score | **WORKED.** 3/3 prospective, pooled 0.5118 beat both pure arms |
| exp6251 | best-of-N sampling, same selector | no reliable gain; 1 win, 1 loss, 2 no-headroom |
| exp6252 | goal gradient as the planner's heap key | positive **RETRACTED**, corrected run is a clean negative |
| exp6254 | MoE-many vs dense-few at matched wall-clock | dense wins 2-0; confirms the existing generator pin |
| exp6255 | fill the induce prompt's win-exemplar slot | **NEGATIVE** on dynamics, pooled -0.2476 |
| exp6256 | same, scored on the goal predicate instead | no effect: 0 of 8 predicates fire on a real win |
| exp6257 | sweep the 25 stored goal predicates | **14 of 21 degenerate**, 4 planner solves verified hollow |
| exp6258 | confusion matrix of the live veto | admits 21 of 26 useless; acceptance precision 0.19 |
| exp6259 | add sensitivity to the veto | **kill condition fired**: 14 admitted, 0 sensitive, nothing to select |

## 2. The one thing that worked

REQ-ARC-WMTE-6250. Running both the linear and REx refinement arms and keeping whichever
scores higher on VALID matched the held-out-optimal arm on 3 of 3 prospective games, and the
ensemble's pooled held-out fidelity (0.5118) beat BOTH pure arms (linear 0.2032, rex 0.4810).
tu93 is the case it exists for: linear returned a worthless engine (0.0) and REx an excellent
one (0.9259) at equal budget, and the VALID score identified it in advance.

It is a SELECTION improvement, not an induction improvement. It picks better from what the
generator already produces. exp6251 then showed the same selector missing on g50t -- VALID
doubled while held-out fell -- so "VALID predicts HELD" must not be generalised beyond
choosing between two structurally different arms.

## 3. The goal predicate is broken, systemically

This was not on the original plan. It emerged from following exp6255's ambiguity.

- **14 of 21 stored goal predicates never fire on a real win** (exp6257). All 14 score a
  PERFECT 1.0 on the project's own `score_goal_predicate_consistency`, because held-out data
  contains no level-ups and a constant-False predicate is 100% correct against it.
- **22 of 22 freshly induced predicates** across exp6256 and exp6259 also fail to fire.
- `plan_in_model` terminates on this predicate. Ten of the 14 degenerate games found no plan
  at all; four found one and all four were verified **HOLLOW** -- the predicate accepts the
  terminal grid reached inside the induced model and rejects the real win grid.
- The live veto (`min_goal_predicate_consistency=1.0`) admits 21 of 26 useless predicates,
  acceptance precision 0.19 (exp6258).
- Adding sensitivity to that veto rejects everything and leaves nothing to select: 14
  admitted, 0 sensitive, arm B empty on 4 of 4 games (exp6259).

**The gate is not the lever.** It is not selecting badly from a good pool. The pool has
nothing good in it.

## 4. Two corrections this batch had to make

Both are recorded because the reasoning error is more reusable than the result.

**The exp6252 retraction.** A positive was reported, then destroyed by adversarial review in
three independent ways: the mandated random ablation was a no-op (numpy abbreviates `repr`
above 1000 elements, so every 4096-cell frame hashed identically, making the "random" arm
literally the flat baseline); a zero-information control reproduced the whole effect
byte-identically on two games; and the win metric was `nodes_expanded` when the benchmark
charges ACTIONS -- the biggest "win" had traded a 7-action plan for 71. The `UniformGoalEnergy`
bug was real shared-code damage and is fixed.

**The false-reject claim.** After analysing the veto I wrote that it both admits
constant-False predicates AND rejects discriminating ones. Measurement supported the first
(21 of 29) and gave **zero** support to the second. The defect is one-sided: it over-admits,
it does not over-reject. That matters practically -- it means LOWERING the threshold makes
things strictly worse.

**The common thread:** in both cases the headline was wrong and only the per-row data
revealed it. Three separate times this batch, a verdict string said something the rows did
not support. Reading rows over verdicts is the habit that caught all three.

## 5. What this says about where the headroom is

Every lever tested operates DOWNSTREAM of induction:

- exp6250/6251 select among produced candidates
- exp6252 reorders the search over a produced model
- exp6254 changes which model produces them
- exp6255/6256 change the prompt's framing
- exp6258/6259 filter what is produced

The dynamics half sits at roughly 0.38-0.5 held-out fidelity against the near-1.0 a 14-step
plan needs. The goal half is degenerate in 22 of 22 fresh inductions. Nothing downstream can
repair either. The next real attempt has to change how the world model is PRODUCED.

**What that does not mean.** It does not mean "use a bigger model" -- exp5722 already moved
zero live levels and exp6254 confirms the current pin under a new frame. It means the open
question is a different production mechanism, not a bigger or better-prompted version of the
present one.

## 6. Honest limits on everything above

- Rosters are 3-6 games; the project's bar for a percentage-point claim is n>=30.
- All win grids are development proxies from replayed banked solves. A hidden game has no
  banked solve and no adapter, so at level 1 the live agent cannot obtain one at all. Several
  of these experiments therefore measured the BEST case for a fix and it still failed.
- Held-out change-fidelity and predicate sign-agreement are offline proxies. No live level
  was solved or claimed by any experiment here.
- exp6252's corrected negative is further weakened by exp6257: most of its roster had a
  termination condition that was unreachable or wrong, so it could not have detected a
  benefit on those games even if one existed.

## 7. Open, in rough order of expected value

1. **A different production mechanism for the goal predicate.** 22 of 22 fresh inductions are
   degenerate. Prompting has been tried (exp6255/6256). Filtering has been tried
   (exp6258/6259). Neither addresses production.
2. **Wire exp6250's ensemble into the live path, default-OFF, then shadow A/B.** The one
   positive, still unwired.
3. **Use the free Kaggle preview channel.** The P5 harness is built and unpushed; it measures
   on the real scored card at zero submission cost.
4. **Fix the conductor's `--no-verify` asymmetry.** It creates guard debt only other
   committers ever have to clear, and it nearly published stripped fabrication-gate
   determinations on 2026-08-12.

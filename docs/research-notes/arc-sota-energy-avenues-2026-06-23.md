# Next avenues for the LIVE ARC agent: embrace + extend the cloned SOTA with energy

**2026-06-23, outer-loop (operator-requested).** Method: deep-read the two cloned SOTA solutions
→ propose energy extensions per lens → adversarially verify each lifts a LIVE metric → synthesize
(16-agent workflow `arc-sota-energy-avenues`). **All load-bearing repo claims independently
verified; one synthesizer claim CORRECTED (see §0).**

## 0. The headline correction (read first)

The workflow's #1 recommendation — "no milestone has softened the exact-match trust gate; build a
graded cell-recall gate" — is **wrong on the prescription, right on the diagnosis.** The graded gate
**already exists**: commit `beb9432e9` (`[outer-loop] arc: coordinated redesign piece 1 — cell-recall
verify gate`) added `CARNOT_ARC_TRUST_METRIC=cell_recall` (`arc_competition_agent.py:1716-1728`),
which gates on graded changed-cell recall instead of full-grid `np.array_equal`. It is **defaulted
OFF** (`exact`, to preserve the submitted-agent parity test) and — the genuinely-open fact — **has
never been A/B-measured live** (no `trust_metric` value appears in any `results/*.json`). So the
highest-value next move is **measurement, not construction**: turn the existing flag on and measure
whether live first-win lifts off 0.08. (The synthesizer missed this because the `carnot-live` reader
died on an API-overload mid-run.)

## 1. The two SOTA solutions and what we've already embraced

- **StochasticGoose** (Tufa Labs, leaderboard LEADER, `ARC3-solution/`): online CNN self-supervising
  a per-pixel clickability/frame-change prior (BCE on "did the frame change", hash-deduped 200k
  buffer, reset-on-level-up), biasing exploration toward predicted frame-changers — **pure
  exploration, no goal/value/world-model.** Carnot already ships this as `SmallFrameChangeCNN` /
  `FrameChangeScorer` (`cnn_weight=0.05`); `.427` PHASE A2 graduates it into live candidate ranking.
- **"Explore It Till You Solve It"** (`dolphin-in-a-coma`, arXiv:2512.24156, 3rd, 17 median levels
  post-bugfix): FrameProcessor + Level GraphExplorer hashing status-bar-masked frames into a finite
  per-level graph, draining the untested-edge frontier in 5 salience tiers — **no induction, no
  model, env-score the only oracle.** Carnot already has this as the live `StepwiseExplorer` /
  `E3AgentPolicy` (RESET-replay navigation, salience-tiered `rich_action_candidates`).

**The uncomfortable lesson:** just-explore reaches ~17 levels with **zero learned model**. Any energy
extension must beat *pure exploration*, not merely add per-node cost.

## 2. The honest wall (independently re-derived by this SOTA analysis → corroborates arc-008)

Offline reproduces 56 levels; **live first-win-rate does not lift (stuck at 0.08 = 1/11; 8/32 larger
sample).** The `.425/.426` capstones falsified the value-head-reranker-into-search lever twice
(`first_win_delta=0`, solve-rate `0.04` across bare==graduated==linear). The wall is **NOT the search
loop** and **NOT representation** (cross-game LOO-AUROC already at 0.725). Per
`docs/research-notes/arc-008-wall-root-cause-2026-06-21.md` it is **overdetermined — three compounding
failures, no single lever**:

1. **The exact-match trust gate** (`WorldModelVerifier`, `np.array_equal(pred,next_grid)` full-grid):
   a ~55%-changed-cell-accurate model scores ~0 → gated out 0/5 (TTT) and 0/6 (e3 induction) → the
   induce→verify→plan superstructure is a **no-op** → fall back to the bare exploration floor (=0.08).
2. **The offline→live bridge / distribution-shift:** a 0.725 offline value head *regressed* live
   (`value_weight` reverted 5→0) because it was trained on winning-path states, not the off-path live
   frontier.
3. **EXPLORE_SAW_NO_WIN:** on hard games the explorer only reaches L1 ~2/25, so induction has no win
   to condition on regardless of the gate.

**The verify pass killed all 8 proposed energy grafts** as frontier-reordering / offline-only /
subsumed — *consistent with* "the wall is overdetermined; no single energy lever moves it." That is
itself the finding: more energy machinery on the world-model path is dead-on-arrival until the gate
and the first-win generation are unblocked.

## 3. The three avenues that survive (corrected)

### A. PURSUE_HIGH — **Measure** the already-built graded trust gate live (then default-on if it lifts)
- **Not "build"** — `CARNOT_ARC_TRUST_METRIC=cell_recall` exists, defaulted off, unmeasured. Turn it
  on; measure trust-pass rate, plan-fire rate, and **live first-win** on the 0/6 gap-1 games vs the
  bare-floor matched control. Add the **energy-bounded divergence-halt** (halt a plan on a cell-recall
  energy spike, not on first `pred != obs`) if not already wired. `verifier_is_oracle: false` (the
  energy is transition-consistency, not the env win-oracle).
- **Gate:** a ≥0.5-changed-cell-accurate model passes trust on ≥3/6 previously-0/6 games AND drives
  ≥1 executed plan of >2 model-inferred actions; **live first-win > bare 0.08** on matched control.
  Else the gate was not the binding lever (the wall is downstream) → record and move to B.
- **Moves:** live first-win-rate. **Attacks:** failure #1 (the gate chokepoint). **Cheap** (flag flip
  + measurement + a parity-test update if defaulted on).

### B. PURSUE_MED — Energy-directed exploration-to-first-win (free-energy/surprise frontier bias)
- Because of failure #3, even a softened gate has no win to condition on hard games. Prioritize
  frontier nodes by **model-surprise / free-energy** (high prediction-error under the induced
  dynamics) for epistemic exploration; once any partial goal structure exists, bias toward
  **goal-energy descent** (the just-merged `induce_goal_energy`). Pairs with `.427` A1's dense
  curiosity loop. `verifier_is_oracle: false`.
- **Gate:** L1-reach-rate on the ~2/25 hard-game set rises to ≥4/25 within the same action budget —
  measured as **L1-REACH (a generation win), NOT unique-states-visited** (that's the killed reordering
  proxy). Else it is the `.426` reordering null again.
- **Moves:** live first-win (via L1-reach). **Attacks:** failure #3. **Partly downstream of A** (a
  fired plan needs a gate that passes).

### C. PURSUE_MED — DAgger search-distribution value retraining as a *pruner* (not an A* value)
- Retrain the verifier/value-energy on the **actual live off-path frontier** (DAgger-style; WM-DAgger
  arXiv:2604.11351 + DAgger arXiv:1011.0686, already flagged for `.427` ingestion), calibrate
  energy→cost (isotonic/Platt, arXiv:1706.04599), and deploy it as **bounded high-energy action
  masking** (the exp1165 ~4× action-efficiency lever), NOT as a search value (that was the 5→0
  failure). `verifier_is_oracle: false`.
- **Gate:** live action-count-to-first-win drops ≥15% vs bare exploration on matched seeds, **no
  solve-rate regression**. Else retire.
- **Moves:** live action-efficiency (the squared score term we have nothing for). **Attacks:** failure
  #2 (distribution-shift bridge). Improves efficiency of wins we already get, not first-win on stuck
  games — lowest-risk live mover, pairs with A.

## 4. Honest dead-ends (KILLED — do not re-propose)

Boltzmann/Gibbs click-sampler (subsumed by A2; attacks a non-existent per-pixel sampler — we
enumerate object-centroid clicks); amortized/O(1) frontier-energy (the v3 featurizer is already
cached+lazy; cheaper chance-level routing changes no live metric); transition-trust SKIP-allocator
(already the live induce→verify→plan cascade); relational dynamics transfer prior (already in
`cross_game_features_v3`, LOO already measured, `cross_game_transfer_helps=False`); energy-prioritized
experience replay (offline-only; the default `dynamics_backend="dsl"` has zero gradient steps);
energy-pooled parallel swarm (live harness is strictly single-trajectory — K rollouts physically
cannot run live). **Pattern:** any avenue whose win is frontier-reordering, making-a-known-dead-signal-
cheaper, or an offline proxy metric is falsified — only *generating a win* or *unblocking the gate*
survives.

## 5. The single most important caveat

The wall is **overdetermined**: softening the gate (A) alone may just expose failure #2/#3. The honest
sequence is **A then B then C, each gated on a LIVE first-win/efficiency delta with retire-if-null** —
not all three at once, and not more energy grafts on a world-model path that is gated to a no-op. The
good news: the most-load-bearing piece (A) is already built and merely unmeasured, so the next move is
**cheap measurement**, not construction.

## Cross-references
- `docs/research-notes/arc-008-wall-root-cause-2026-06-21.md` — the overdetermined-wall diagnosis this corroborates
- `arc_competition_agent.py:1716-1728` — the built-but-defaulted-off `CARNOT_ARC_TRUST_METRIC=cell_recall` gate
- commit `beb9432e9` — "coordinated redesign piece 1 — cell-recall verify gate"
- `results/experiment_4626_capstone_v426.json` — `bridge_characterized_cause_isolated_no_live_lift`; reranker falsified twice
- SOTA repos: `StochasticGoose/ARC3-solution` (leader), `dolphin-in-a-coma` / arXiv:2512.24156 (3rd)
- `ops/arc_solve_registry.yaml` — `reproducible_total_levels: 56`

# ARC-AGI-3 Agent Research Plan: Getting Past Zero

Status: research plan (outer-loop drafted, not yet committed). Date: 2026-06-08.
North star: a non-zero score on ARC-AGI-3 (the interactive ARC Prize 2026 benchmark;
act -> observe -> act; solve levels by inducing each game's novel rule from interaction).
Scope: this is the deliberate next research PHASE, not more quick model runs.

This plan adversarially synthesizes three candidate agent architectures (LLM world-model,
program/DSL synthesis, active-inference verifier) against a verified SOTA scan, and commits
to one staged, offline-first, falsifiable path. Every milestone ties to the existing harness
(`scripts/experiments/arc3_offline_eval.py`), develops fully offline, and respects the
operator QUOTA-GATE: an online/scored run is permitted ONLY when an offline result beats BOTH
a TRM baseline AND our best prior submitted Carnot run.

---

## 0. The two facts that reorganize everything (verified from the SOTA scan)

**Fact 1 — The score is RHAE (Relative Human Action Efficiency), not solve-rate.** Per level,
`(human_actions / AI_actions)^2`, weighted toward later levels, averaged per game (ARC-AGI-3
tech report, arXiv 2603.24621, high confidence). Two load-bearing consequences:
- You must reach `GameState.WIN` on a level to score *anything* on it — efficiency on an
  unsolved level is 0. **Inducing the rule well enough to win is the GATE.**
- Given a win, fewer actions squares the score. **Carnot's verifier-pruner attacks the score
  multiplier directly** — this is the venue where "verifier = efficiency" is the actual metric,
  not a decoration. Our measured pruner gives **1.96x fewer actions (CI 1.74-2.19)** on the
  synthetic harness (exp3929, verified in `results/experiment_3929_arc_agi3_action_efficiency.json`).
  NB: the "~4x" figure quoted in two of the candidate designs is INFLATED; the honest, replicated
  number is 1.96x. Use 1.96x in all claims.

**Fact 2 — Per-step reactive policies are confirmed structurally incapable, and so are pure
LLM harnesses.** Frontier LLMs on the official semi-private leaderboard (Mar 2026): Gemini 3.1
Pro 0.37%, GPT-5.4 High 0.26%, Opus 4.6 Max 0.25%, Grok-4.20 0.00% (high confidence). Our own
0/183 across random / object_click / Gemma E2B/E4B direct / Gemma E4B reasoning / codex-text is
the same finding in microcosm. **What beats zero ALWAYS replaces "LLM picks the next action per
step" with a persistent structure that accumulates across the horizon.** Two families do this:

- **Family A — Informed search + cheap learned value model, NO rule induction.** StochasticGoose
  (1st preview, 12.58%, 18 levels; CNN frame-change predictor + experience buffer), Blind Squirrel
  (2nd, 6.71%; directed state graph + back-labeled ResNet value model), Graph-Based Exploration
  (3rd, arXiv 2512.24156; pure structured exploration that *explicitly makes no attempt to learn
  mechanics* and still beat an LLM+DSL baseline ~4x). **A no-induction explorer ranked 3rd** —
  this is the single most important de-risking fact for M1.
- **Family B — Executable world models verified against transitions (current SOTA, 32.58% mean
  RHAE, 106/209 levels, arXiv 2605.05138).** A coding agent (GPT-5.4 via Codex CLI) maintains
  executable Python transition dynamics; the loop is: edit model -> **run verifiers that check the
  model reproduces recorded transitions** + re-solves already-solved levels -> **refactor toward
  simpler abstractions (explicit MDL/Occam)** -> plan-executor simulates the action sequence
  in-model and **stops the instant predicted frame diverges from reality** (a wrong model never
  burns real actions). High run-to-run variance (cn04: 62% vs 0.01%).

**The strategic punchline: the SOTA architecture IS the Carnot thesis.** Executable world model =
a *hypothesis*; "verify it reproduces recorded transitions" = Carnot's energy verifier as a
consistency check; "stop the instant prediction diverges" = the Meta-EBM cascade router's
verify-then-escalate; "refactor toward simpler abstractions" = MDL/Occam, which our
`arc_grid_verifier_invariants_v2_combined` invariants already encode. The winning division of
labor is exactly ours: **a strong generator induces the world model; the verifier prunes wrong
models and prunes wasteful actions.** We do not need to invent the shape — we need to build the
cheapest credible instance of it and make the verifier load-bearing.

**One thing the scan rules OUT:** the "Seed IQ / active inference 95.49%" claim (Themesis /
Denise Holt) is UNVERIFIED — not on the official leaderboard, no code, no scores/dates, sourced
to promo posts. Do NOT use it as a benchmark anchor or quota-gate target. Active inference as a
*direction* is legitimate; this specific result is not evidence.

---

## 1. Honest verdict on the three candidate architectures

Ranked by probability of beating zero on the easiest games, offline-first, with our assets.

### Rank 1 (best near-term): Program/DSL synthesis (`program_synthesis`)
**Verdict: this one can actually beat zero, and it is the closest match to verified SOTA.** It
is Family B (executable world model) reduced to a tractable, sovereign, offline core. Its key
structural advantage is real and confirmed against our own data: on vc33 a click produces a
**deterministic, sparse, local 1-cell delta**, which collapses the version-space search from
intractable to seconds (least-general-generalization over the observed delta). The goal predicate
is induced from the discrete reward we get for free (`FrameData.levels_completed` incrementing),
which every reactive policy throws away. The verifier role is genuinely load-bearing (consistency
energy = count of mispredicted cells; MDL/Occam = arc invariants), not decorative.

**Its real fatal-gap risk: hidden latent state.** A grid->grid DSL has NO consistent program when
the same (state, action) maps to different next-states because of off-screen/armed/inventory state.
This is the dominant failure mode and the candidate names it. Mitigation (detect non-determinism in
the store, add latent boolean registers, the key/switch/gate pattern already prototyped in
`arc_agi3_action_efficiency.py`) is plausible but unproven on the live games. **This is why M1 is
deliberately the single easiest, click-only, deterministic game (vc33) — if program synthesis can't
induce vc33's rule, the approach is refuted before we spend on harder games.**

### Rank 2 (best generalizer, more risk): LLM world-model + active exploration (`llm_world_model`)
**Verdict: would beat zero on more games than program synthesis IF the codex induction step
reliably writes a checkable goal predicate — but its differentiator is the LLM, not the verifier,
so it de-risks the project LESS.** This is Family B with a neural-ish persistent `GameModel` and
codex as the periodic inducer. It is the most likely to generalize across the 25 games because the
inducer is a strong general reasoner, and its explore -> model -> exploit loop with cross-episode
persistence is a faithful Voyager/Reflexion lever (persistent skill memory + periodic reflection),
which is exactly what gets past the reactive ceiling.

**Its real fatal-gap risk is honesty-of-attribution, not capability.** If codex induces the rule,
the *verifier added nothing to the accuracy* — it only pruned actions (efficiency). For the
broader project thesis ("the verifier is Carnot's value-add"), a solve attributable to codex is a
weaker result than a solve attributable to the verifier. Two concrete sub-risks the candidate
names: perception noise (mitigated by computing deltas DETERMINISTICALLY in numpy, using Gemma only
for object semantics — this is correct and we adopt it), and codex producing an unfalsifiable goal
predicate (mitigated by requiring the predicate be CHECKED against the transition log before it is
trusted — also correct, we adopt it). Quota is the operational risk: even O(level-ups + surprises)
codex calls across 25 games can strain the 23% weekly headroom; the offline-first gate protects us.

### Rank 3 (most elegant, highest single-point-of-failure): Active-inference verifier (`active_inference_verifier`)
**Verdict: the cleanest expression of "verifier as Bayesian rule-inducer," and the only design
where the energy verifier does induction (posterior refutation) rather than just pruning — but it
has the SAME fatal gap as per-step-reactive in a new disguise: COVERAGE.** The honest core is
right: energy alone cannot invent a goal, so it is given a goal-HYPOTHESIS SPACE (ARC core-knowledge
priors) and does what it provably can — refute hypotheses against observation (Boolean-E exact +
grid-consistency soft) and concentrate the posterior. The epistemic/pragmatic EFE action selection
is a principled way to choose disambiguating probes, and the GAME_OVER-as-large-energy survival
penalty is a real improvement over naive exploration.

**Its fatal gap: if the true game-rule is not in the ~30-60 hypothesis bank, the posterior collapses
onto the least-wrong-but-wrong hypothesis and the agent plans toward a false goal — and then ALL
induction falls to the LLM proposer fallback, making the energy layer decorative.** ARC-AGI-3 games
are explicitly designed so each game's rule is NOVEL; a fixed bank is exactly the wrong bet against
an adversary whose whole point is to be off-distribution. The candidate itself names this as "the
dominant risk." Program synthesis grows its DSL from observed deltas (open-ended within the operator
set); the AIF bank is closed. **We keep the AIF machinery as a SCORING/ROUTING layer inside the
recommended hybrid (the posterior-over-hypotheses and EFE probe-selection are excellent), but we do
NOT bet the induction on a fixed bank.**

**Ranking summary:** program_synthesis (induction is open-ended + verifier load-bearing + matches
SOTA + cheapest falsification) > llm_world_model (best generalizer, but credit goes to the LLM) >
active_inference_verifier (most elegant, but closed-bank coverage is a reactive-class gap).

---

## 2. Recommended path: a verifier-centric explore -> induce -> exploit hybrid

**One-liner:** Build a persistent per-game state-action graph by structured exploration (Family A,
the no-induction path that already ranks 3rd), have the Carnot verifier prune looping/null-effect
actions and score progress (the efficiency thesis, measured against a no-verifier ablation), then
layer a program-synthesis inducer on top (Family B) — deterministic-delta DSL enumeration verified
by the energy/consistency ensemble, with codex as the rare heavy inducer and Gemma as cheap
perception — so the verifier prunes wrong rules and wasteful actions while a strong generator
induces the rule.

This is a hybrid because the SOTA scan is unambiguous: the no-induction state-graph explorer
(Family A) is the cheapest thing that beats zero, and the executable-world-model (Family B) is the
SOTA. We build A first (it de-risks M1 with the lowest variance), then graft B's induction onto the
same graph (it is the path to RHAE that scales). The active-inference machinery contributes its
posterior-over-hypotheses scoring and EFE probe-selection as the glue between explore and induce.

### The roles (each asset load-bearing, none decorative)

**(a) Local multimodal Gemma-4 (E2B/E4B-it) — cheap perception + semantic labeling, NEVER the
inducer and NEVER per-step-reactive in isolation.** Concrete jobs:
- Object semantics on demand: when the deterministic perceptual delta is ambiguous, render the
  frame (`_render`) and ask Gemma "what kind of object is at (x,y), and what changed" — Gemma
  supplies object class / role labels that sharpen the DSL's SELECT predicates. Reuses
  `scripts/experiments/arc3_gemma_policy.py:_render`, `gemma_action`.
- DSL-program proposer when enumeration's pool is empty: re-prompt Gemma (or codex) with rendered
  before/after frames to OUTPUT a candidate DSL program; it is parsed to AST and handed to the
  verifier. **Gemma proposes, the verifier decides.** This is the only place Gemma touches actions,
  and it is rare (enumeration-empty), not per-step.
- 0.3-1.3s/call, fully local/sovereign. The deterministic delta is computed in numpy, never from
  Gemma prose (adopting the `llm_world_model` perception-bottleneck mitigation).

**(b) Codex bridge (gpt-5.5) / strong reasoner — PERIODIC heavy rule-induction, never per-step.**
Concrete jobs:
- The expensive abductive leap: serialize the compact transition log + object inventory + current
  goal hypothesis (`_serialize` from `arc3_codex_policy_test.py`) and ask codex for a STRUCTURED
  world-model update — refined per-action effects, an explicit CHECKABLE goal predicate, and a
  candidate plan. Reuses `_ask_codex` with the timeout->object-click fallback so a codex stall
  never ends the game.
- Fires only at escalation points (level-up, surprise = observed != predicted, plan stall, or
  enumeration-empty), bounded to O(level-ups + surprises) per game, not O(steps). Gemma-E4B-reasoning
  is the local fallback inducer when quota is constrained.
- 23% weekly quota headroom is viable precisely because the verifier carries the per-step inner loop.

**(c) Carnot verifier ensemble — the differentiator: candidate-rule verification + action-pruning +
state-value.** Three concrete, measurable plug-ins, all reusing shipped code:
1. **Candidate-rule verifier (accuracy-side, the SOTA-matching job):** `consistency_energy(program,
   store)` runs `apply(program, s_t, a_t)` over every stored transition; per-cell mismatch is encoded
   via `python/carnot/verify/sat.py` + `and_composition_verifier.py`; energy == 0 means provably
   consistent on ALL evidence; energy > 0 ranks near-misses for repair. Malformed LLM-proposed
   programs rejected by `ast_structure_verifier.py`. This is the Meta-EBM cascade: run the cheap
   `arc_grid_verifier_invariants_v2_combined` first (sub-ms; the 7 invariant families double as
   goal-predicate hypotheses AND structural filters), escalate to the LLM proposer only when the
   cheap pool empties.
2. **Action-pruner (efficiency-side, the RHAE multiplier):** `select_verifier_pruned_action` /
   `task_potential` / `encode_rich_candidate_action` from `arc_agi3_action_efficiency.py` prune
   looping, null-effect, and deadly actions before any model call. Measured against a no-verifier
   ablation (M3). The replicated win is 1.96x (exp3929), not 4x.
3. **State-value / progress signal:** once a goal predicate exists, a `task_potential`-style
   goal-distance + the arc invariants give a per-step progress score to greedily order plan steps
   and detect non-progress WITHOUT an LLM call. This is also the cascade's "stop the instant
   prediction diverges" trigger.

### The persistent data structure (one graph, both families read/write it)
A per-game `GameGraph` (new module `python/carnot/agentic/arc_agi3_world_model.py`):
- nodes = frame hashes; edges = (action, data) with the deterministic `grid_delta`, `level_delta`,
  `became_game_over`. (Family A: Blind-Squirrel/Graph-Explore directed state graph.)
- `transition_store`: append-only (s, a, s') triples for the synthesizer. (Family B.)
- `action_effects`: per discrete action / per object-class-clicked aggregated effect signature.
- `deadly_actions`: state->action pairs that caused GAME_OVER (never re-selected).
- `rule_posterior`: categorical over surviving candidate DSL programs (AIF contribution — the
  posterior is over SYNTHESIZED programs, not a fixed bank, which fixes the AIF coverage gap).
- `goal_hypothesis`: {text, predicate_sketch, confidence}, committed only after >=2 level-up
  examples (fixes the `program_synthesis` goal-mis-induction risk).
- Persists to `results/world_model_<game_id>.json` keyed by game_id; resets become cheap re-attempts
  (cross-episode persistence). Effects/goal stored in OBJECT-RELATIVE terms (not absolute cells) so a
  re-randomized layout still transfers (fixes the cross-episode-determinism risk).

---

## 3. Staged milestone ladder (offline-first, each falsifiable, each tied to the harness)

Every stage runs through `scripts/experiments/arc3_offline_eval.py` by registering a new policy in
its `POLICIES` dict, so it is measured by the SAME quota-gate harness (ACCURACY = levels solved,
EFFICIENCY = mean action ratio on solved) as random / object_click / Gemma. No new measurement loop.

> Harness note (load-bearing for M1): `run()` computes `budget = sum(baseline_actions) * budget_factor`.
> For a single-level test, run vc33 with a budget large enough for multi-episode accumulation —
> use `--budget_factor` so the effective budget reaches ~200 actions across resets, or add a
> per-game budget override. vc33 `baseline_actions = [7, 18, 44, 61, 131, 34, 152]`; level-0 = 7.

### M0 (infrastructure, ~0.5 day): the GameGraph + offline policy skeleton
- Create `python/carnot/agentic/arc_agi3_world_model.py` with `GameGraph` (nodes/edges/transition_store/
  deadly_actions/JSON persistence), lift `_objects` flood-fill perception into the module, add
  `compute_grid_delta(prev, next)` (deterministic numpy changed-cell set + appeared/disappeared/moved).
- Register `graph_explore_policy` in `arc3_offline_eval.py:POLICIES`.
- `tests/python/test_arc_agi3_world_model.py`: a transition is logged with the correct delta; a deadly
  action is never re-selected; the graph round-trips through JSON. Every test asserts (no skips).
- **Falsifiable:** the policy runs end-to-end offline on vc33 and emits a valid quota-gate artifact
  with a non-fabricated `duration_s` and `inference_substrate=offline_arc_agi3_graph_explore`. (No
  solve required yet — this is the harness wiring gate.)

### M1 (the decisive first non-zero): solve vc33 level-0 offline by ANY means
- Approach: Family-A structured exploration on the GameGraph — sample untested actions at the current
  perceptual-priority tier, navigate shortest-path to the nearest frontier state otherwise, prune
  actions that loop or don't change state, never repeat a deadly action. This is the no-induction path
  that ranked 3rd in the preview competition; it is the lowest-variance way to a first win.
- **Falsifiable gate:** `ACCURACY_total_levels_solved >= 1` on `vc33-5430563c` in `arc3_offline_eval.py`
  with the `graph_explore` policy, within a generous budget (~200 actions across up to ~10 episode
  resets so the persistent graph accumulates). Every prior policy scored 0 here; a single level-up is
  the FIRST non-zero and proves a persistent horizon structure beats reactivity.
- **Kill-criterion:** if after 200 actions x 10 resets vc33-L0 is still 0, pure structured exploration
  is refuted for the cheapest case -> escalate to M1b (graft the program-synthesis inducer immediately,
  do not move to harder games).
- **M1b (contingency, same game):** add the deterministic-delta DSL enumerator + `consistency_energy`
  verifier + level-up goal induction; re-test vc33-L0. If THIS also fails to solve the single easiest
  deterministic click-game, the whole world-model direction is refuted (see Section 4 kill-criteria).

### M2 (generalize the loop): solve level-0 of the 3 easy games
- Add the program-synthesis inducer (Family B) and the codex/Gemma proposer fallback so the loop is
  explore -> induce -> exploit, not just explore.
- Targets: vc33 (L0=7), lp85 (L0=17), s5i5 (L0=20). Note the SOTA paper (arXiv 2605.05138) fully solves
  lp85 (100%) and the ar25/vc-family, confirming these are tractable.
- **Falsifiable gate:** `levels_solved >= 1` on each of the 3 games offline (>=3 distinct level-0 wins),
  AND the DreamCoder-style DSL library shows >=1 fragment reused across games (cross-game transfer
  measured, not assumed).
- **Kill-criterion:** if the loop solves vc33 but NOTHING generalizes to lp85/s5i5 after the inducer is
  added, the DSL is overfit to vc33 -> the executable-world-model path does not scale for us; fall back
  to the Family-A-only explorer as the submission vehicle (it still scores, per Graph-Explore 3rd place).

### M3 (the efficiency thesis — the project's real venue): verifier measurably cuts actions
- Run the recommended hybrid in two configurations on the games solved in M2: (i) WITH the verifier
  action-pruner + state-value, (ii) ablation with the pruner DISABLED (random/uninformed action order
  among legal actions).
- **Falsifiable gate:** on solved levels, `EFFICIENCY_mean_action_ratio_on_solved` is significantly lower
  WITH the verifier than the ablation, with non-overlapping bootstrap 95% CIs, replicating the synthetic
  exp3929 direction (1.96x, CI 1.74-2.19) on REAL games. Report `actions_to_solve` and
  `llm_calls_per_solved_level` for both arms.
- **This is where "verifier = efficiency" gets its real benchmark.** RHAE squares action efficiency, so
  a measured pruning win on real games is a direct score improvement AND the cleanest external evidence
  for the broader project thesis.
- **Kill-criterion:** if the verifier-pruned arm is not significantly more efficient than the ablation on
  real games (CIs overlap), the efficiency thesis does not transfer from synthetic to ARC-AGI-3 -> the
  verifier's contribution is downgraded to candidate-rule verification only (accuracy-side), and the
  efficiency claim is retracted from forward-facing docs (per the false-negative discipline).

### M4 (quota-gate-passing offline run worth a scored submission)
- Scale the hybrid across the start_here_top8, accumulating the cross-game DSL library.
- **Falsifiable gate (the operator QUOTA-GATE):** the offline `ACCURACY_total_levels_solved` (then
  `EFFICIENCY`) beats BOTH (a) the TRM baseline and (b) our best prior submitted Carnot run. ONLY when
  this holds does the harness permit an online scored run. Capture
  `inference_substrate=offline_arc_agi3_hybrid_graph_synth_gemma_codex` and non-fabricated `duration_s`;
  honest_verdict prefixed `complete:`.
- **Online submission is operator-only** (per CLAUDE.md Operator-Only External Publication): the task
  prepares the scored-run package; the operator triggers the online run.

---

## 4. Honest risk + kill-criteria + fallback

**The single largest risk (kills the whole world-model direction):** ARC-AGI-3 games have HIDDEN
LATENT STATE not visible in the single rendered grid (armed switches, inventory, off-screen goals).
A grid->grid model is then fundamentally under-determined and NO consistent program exists.
- Detection: in the transition store, the same (frame_hash, action) yields different deltas.
- Mitigation: extend state with latent boolean registers (the key/switch/gate pattern already
  prototyped in `arc_agi3_action_efficiency.py`); model in object-relative terms.
- **Kill-criterion:** if M1b (vc33-L0 with the full inducer) cannot solve the single easiest
  DETERMINISTIC click-game, the executable/program-synthesis world-model is refuted for our setup.
- **Fallback:** the Family-A state-graph explorer (M1's graph_explore policy) WITHOUT induction. It
  ranked 3rd in the preview competition with zero rule-learning; it is fully offline-developable and
  still makes the verifier load-bearing for action-efficiency (RHAE). This is the floor we never drop
  below — it should itself beat zero on at least vc33.

**Other named risks and their kills:**
- *Verifier mis-scores ARC novelty* (energy verifier built for math/CSP; its "progress proof" 1+0=1
  encoding is a synthetic proxy and may prune the right action when the goal is unknown). Mitigation:
  in EXPLORE, prune only deadly + null-effect actions, never by goal-energy (goal unknown); let the
  verifier state-value gate EXPLOIT only after a goal predicate exists; keep a fallback that never
  fully prunes (`VerifierRouter.fallback_action`). Kill: if the verifier-gated explore is WORSE than
  unguided explore at reaching the first win (M1 ablation), disable verifier gating during explore.
- *Goal mis-induction from a single level-up* (a spurious invariant fires once). Mitigation: require
  >=2 level-up examples before committing a goal predicate; act for information gain until then. This
  is the highest risk specifically at M1/M2 where evidence is thin.
- *Codex induction quota blowout across 25 games.* Mitigation: offline-first gate means no online
  spend until baselines are beaten; Gemma-E4B-reasoning is the local fallback inducer; cache induced
  models so re-runs don't re-induce. Kill: if a single offline M4 sweep needs > the 23% weekly headroom
  in codex calls, drop codex from the per-game loop and rely on the local Gemma proposer + enumeration.
- *Closed-bank coverage (AIF).* Already mitigated by making the posterior range over SYNTHESIZED DSL
  programs (open-ended), not a fixed hypothesis bank. The AIF machinery is scoring/routing only.
- *Fabrication/measurement integrity.* Every artifact declares `inference_substrate`, a real
  `duration_s`, `random_seed`, `reproducibility_checksum`, and `preconditions_checked` (Gemma cached?
  arc_agi OFFLINE SDK + environment_files/ present?). A win with implausibly short duration is a flag,
  not a result (per the adversarial-verify + fabrication-gate discipline).

**What tells us a DIRECTION is dead (summary):**
- M1 + M1b both fail on vc33-L0 -> world-model/induction direction dead; ship Family-A explorer only.
- M2 solves vc33 but nothing transfers -> executable-world-model does not scale for us; ship Family-A.
- M3 ablation CIs overlap on real games -> efficiency thesis doesn't transfer; retract the efficiency
  claim, keep verifier as accuracy-side candidate-rule verifier only.

---

## 5. How this de-risks the broader project

1. **The verifier-as-efficiency thesis finally gets its real venue.** RHAE *is* action efficiency,
   squared. The project has measured a 1.96x pruning win on a synthetic harness (exp3929); M3 tests
   whether that transfers to a real, externally-scored, novel-task benchmark with a clean no-verifier
   ablation. A positive M3 is the strongest external evidence to date that Carnot's verifier delivers
   measurable value on accuracy-adjacent tasks where self-consistency is NOT already near-optimal —
   precisely the regime the Depth-Over-Breadth retro flagged as missing. (Contrast: on NL-math/CSP
   corpora, SC is near-optimal and the verifier showed no headroom; ARC-AGI-3 is the headroom venue.)

2. **The verifier-prunes-wrong-models role is validated externally, not internally.** The SOTA paper
   independently lands the exact Carnot division of labor (LLM induces, verifier checks transitions +
   stops on divergence). Building the cheapest instance of it lets us claim the architecture with an
   external comparator instead of a self-graded one — which is exactly the G2-class independent-
   reproduction posture the publication gate wants.

3. **It converts a north-star aspiration into a falsifiable ladder.** ARC-AGI-3 was named THE north
   star but had no offline, air-gapped, milestone-by-milestone path; this plan gives one where every
   rung is measured by the existing quota-gate harness and the first decisive test (vc33-L0) is cheap
   to run and cheap to refute. That is the antidote to breadth-churn: one deep, load-bearing question
   (does explore->induce->exploit with a verifier beat zero?) instead of another vN+1 re-measurement.

4. **It keeps sovereignty intact.** The inner loop is local Gemma-4 perception + CPU energy verifier;
   codex is an optional, rare, quota-bounded inducer with a local Gemma-E4B-reasoning fallback. The
   agent runs air-gapped against the OFFLINE SDK and `environment_files/` with no online dependency
   until the operator gate opens.

---

## Appendix: reused modules (verified to exist in-repo, 2026-06-08)

- Harness + quota-gate: `scripts/experiments/arc3_offline_eval.py` (`POLICIES`, `play_game`, `_objects`,
  `_render` via gemma policy, ACCURACY/EFFICIENCY artifact). `budget = sum(baseline_actions)*budget_factor`.
- Verifier action-pruner + state-value + synthetic ground-truth: `python/carnot/agentic/arc_agi3_action_efficiency.py`
  (`select_verifier_pruned_action`, `task_potential`, `encode_rich_candidate_action`, `RichSyntheticArcEnv`).
  Replicated efficiency = 1.96x, CI 1.74-2.19 (exp3929) — NOT 4x.
- Energy verifier + router types: `python/carnot/agentic/arc_agi3_harness.py` (`VerifierRouter`,
  `default_energy_verifier` -> `carnot.verify.cost_instrumented_verification.run_energy_verifier`,
  `encode_candidate_action`, `stable_reproducibility_checksum`, `Action`, `Coordinate`).
- Candidate-rule verification: `python/carnot/verify/{sat.py, constraint.py, and_composition_verifier.py,
  ast_structure_verifier.py, cost_instrumented_verification.py, verification_compute_router.py}`.
- Grid-consistency / MDL invariants (goal-predicate library + cheap filter): `scripts/experiments/
  arc_grid_verifier_invariants_v2_combined.py` + the 7 `arc_invariant_*_draft.py`.
- LLM perception + proposer: `scripts/experiments/arc3_gemma_policy.py` (`_render`, `gemma_action`),
  codex bridge `scripts/experiments/arc3_codex_policy_test.py` (`_ask_codex`, `_serialize`, timeout->
  object-click fallback).
- Game targeting (verified counts): `vc33-5430563c` baseline_actions `[7,18,44,61,131,34,152]` (L0=7);
  `lp85-305b61c3` `[17,38,31,16,41,60,26,159]` (L0=17); `s5i5-18d95033` `[20,89,106,54,162,38,86,83]`
  (L0=20); from `results/arc_agi3_game_characterization.json`.
- Free-energy precedent: exp1165 (`results/experiment_1165_phase4_active_inference_pilot_v1.json`).
- New modules to create: `python/carnot/agentic/arc_agi3_world_model.py` (GameGraph + graph_explore +
  program-synth inducer), `tests/python/test_arc_agi3_world_model.py`.

## Appendix: SOTA citations (with confidence)

- ARC-AGI-3 tech report, arXiv 2603.24621 — RHAE metric, action space, SDK. HIGH.
- Executable World Models, arXiv 2605.05138 — 32.58% mean RHAE, 106/209 levels, verify-transitions +
  stop-on-divergence + MDL refactor. Current SOTA, matches Carnot thesis. HIGH (high run-to-run variance).
- Graph-Based Exploration, arXiv 2512.24156 — pure structured exploration, no mechanic-learning, 3rd
  place. HIGH (the no-induction floor for M1).
- StochasticGoose (1st preview, 12.58%), GitHub DriesSmit/ARC3-solution + Medium writeup — CNN
  frame-change predictor + experience buffer. HIGH.
- Blind Squirrel (2nd, 6.71%) — directed state graph + back-labeled ResNet value model. MEDIUM-HIGH.
- ARC Prize 30-day learnings — "optimize action efficiency, not completion alone." HIGH.
- Frontier LLM leaderboard (Gemini 3.1 Pro 0.37%, GPT-5.4 0.26%, Opus 4.6 0.25%) — per-step reactive
  fails. HIGH.
- "Seed IQ / active inference 95.49%" (Themesis) — UNVERIFIED, no code/scores/dates. DO NOT anchor on it.

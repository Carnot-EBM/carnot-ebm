# ARC-AGI-3 Generation Wall — Higher Abstractions and the Energy-Config-Space Architecture

**Date:** 2026-06-22
**Author:** outer-loop (Claude Opus 4.8), operator-directed
**Status:** SOTA-ingestion + strategy note. Flagged for the `.425` roadmap (see
`ops/known-issues.md` MANDATORY-NEXT-MILESTONE). Supersedes nothing; complements
`docs/research-notes/arc-008-wall-root-cause-2026-06-21.md`.
**Provenance:** two adversarial research workflows over six higher-abstraction
families (intrinsic motivation/empowerment, unsupervised skill discovery, active
inference/expected-free-energy, program synthesis + ARC-AGI-3 SOTA, object-centric
+ latent planning, LLM-hypothesis-search + evolutionary generation), each
cross-checked against the repo's refuted-approaches ledger and reusable assets.
All claims about repo files were verified against source. Real arXiv IDs only;
equation-level and per-game-number claims (2504.14898 eqs, 2605.05138 RHAE
figures) must be re-read from source before any *paper* citation.

---

## 0. Operator framing (the organizing principle)

> "I want to make sure we are working energy judgement into the live agent so
> that it can refine and embrace an energy config space within each game and
> shared amongst the games to provide guidance to the agent loops as it tries to
> tackle each game level iteratively." — operator, 2026-06-22

This elevates the research below from "use energy in three roles" to one
architecture: **the live agent IS an energy-refinement loop.** Concretely, the
agent maintains and refines an **energy configuration space** with two coupled
tiers:

1. **Per-game (online) energy landscape** — refined *within* a game as the agent
   observes transitions and level-ups. The energy scores configurations (states /
   partial plans / candidate goals) and is updated test-time from the agent's own
   exploration. This is the within-game self-improvement loop.
2. **Shared (cross-game) energy prior** — a library of energy structure that
   transfers *across* games (reusable affordance/goal/macro priors), seeding each
   new game's per-game landscape so the agent does not start from zero, and
   accumulating what each solved game teaches.

The energy then **guides the agent's iterative level attempts**: it is the dense
signal that (a) shapes exploration toward goal-consistent states, (b) ranks/grows
candidate multi-action plans, and (c) measures the agent's own uncertainty about
the unknown game. This is the project's invariant — "the energy function is
ground truth" — instantiated as the live ARC mechanism, and it is the
decentralization-respecting bet (open local generator + energy verifier, no
closed-vendor dependency).

This note maps the SOTA onto that architecture and flags the cheapest concrete
unlocks.

---

## 1. The reframe that reorganizes the program

The wall has been mis-scoped as a *selection* problem for ~30 milestones. The
diagnosis is now settled (quadruply confirmed: `.420`–`.423`): the winning action
sequence is **absent from the candidate pool for 24/25 held-out variants**
(`winner_generated = 1/25`), so nothing downstream of generation can help. The
hidden-leaderboard score is stuck at ~0.08 — a generation/generalization wall, not
a wiring or OOM bug.

The ladder of abstraction:

> **select-the-winner** (dead) → **make-a-winner-appear** → **discover-the-goal-predicate**

- **select-the-winner** is refuted in every form: re-ranking a fixed pool
  (ordering_gain = 0), feature-routing (== random on held-out, but it *diagnosed*
  generation-not-selection), verifier-guided best-first expansion (regressed
  −0.04). You cannot rank a winner that was never generated.
- **make-a-winner-appear** is the real frontier. It has two *non-autoregressive*
  escape hatches the refuted single-action greedy search never had:
  **recombination** (splice a reach-prefix to a goal-suffix — the BES /
  FunSearch shape) and **planning-through-a-trusted-model** (induce → plan, the
  ARC-AGI-3 SOTA Family-B shape).
- **discover-the-goal-predicate** is the deepest and most under-addressed layer.
  A 99%-accurate *dynamics* model pointed at the *wrong win-condition* plans
  confidently to the wrong state — indistinguishable from "search failed." So
  **GAP-ARCH-GOAL-NOT-VERIFIED dominates GAP-dynamics.** Critically, the repo
  *already* has a goal-predicate inducer (`exp4020`) at **held-out precision 1.0**
  that was **never wired into the planner**.

### Where the energy moat finally becomes load-bearing

The strategic payoff is not ARC-specific: Carnot's energy verifier only earns its
keep once it stops being a terminal *judge* and becomes a generative *driver*.
The three new roles, mapped onto the operator's two-tier energy-config-space:

| Energy role | Per-game (online) | Shared (cross-game) |
|---|---|---|
| **Goal target** (pragmatic well) | refine `is_goal` energy from observed level-ups | reusable goal-predicate templates (H1–H5) |
| **Generative fitness** (non-AR) | energy ranks recombined exploration fragments | shared macro vocabulary + affordance priors |
| **Epistemic signal** (uncertainty) | ensemble disagreement directs the ~5n budget | trust-energy that ranks induced models by held-out generalization |

The unifying move: the *same* plan-time energy that used to rank candidates now
**generates** the multi-action plan, and it is *refined online per game* while
*seeded from a shared prior*.

---

## 2. Unified ranked menu (energy-angle foregrounded)

Ranked by (leverage on `winner_generated`) × (offline cheapness) × (energy-moat
reuse). Every gate is `offline_reproduced=true` via `arc_solver_kit.reproduce()`
PLUS a uniform-energy ablation control (rules out lift-from-search-alone) PLUS
beating the 0.04 transfer baseline / 0.08 leaderboard wall. `verifier_is_oracle:
false` on every value claim. No live/scored quota until the offline variant-
transfer rate clears > 0.08 AND beats the best prior submitted Carnot run (per
`feedback_arc3_online_gated_on_offline_beating_baselines`).

### #1 — Wire `exp4020`'s `is_goal` as a GRADED pragmatic-value energy target
**Leverage HIGH · Cheapness HIGH · closes GAP-ARCH-GOAL-NOT-VERIFIED.**
Make the search heuristic in `graph_explore_solve_v2` a convex combination of
`arc_goal_distance` (navigation energy) AND a **graded** goal-satisfaction energy
compiled from `exp4020`'s induced predicate (fraction of target-groups satisfied,
not the binary `unsatisfied_targets == 0`). Emit a plan to the pool only when the
predicate fires. Energy angle: `is_goal` is the per-game goal-energy well the
rollout descends; it is oracle-distinct (predicts win from visible state, never
reads the env counter). Asset: `results/experiment_4020_goal_induction_separation.json`
(`goal_predicate_heldout_precision: 1.0`), `arc_goal_distance.make_goal_distance`,
`graph_explore_solve_v2(heuristic=…)`. Gate: on ≥3 games with `cell_recall > 0.8`
but currently 0 reproduced, wiring graded `is_goal` yields `offline_reproduced=true`
with fewer actions-to-win than the navigation-only baseline. Cites: 2009.08111,
2504.14898, 2605.05138.

### #2 — Energy-as-fitness quality-diversity evolution over action-sequences
**Leverage MEDIUM · Cheapness HIGH · the non-AR generator the menu lacked.**
MAP-Elites over multi-action sequences seeded from `graph_explore_solve_v2`
frontier trajectories. Mutate (insert/delete/swap/splice) and **recombine via
crossover at a shared visited-state hash** (concat reach-prefix P→s with
goal-suffix s→win). Fitness = dense energy along the rollout (`arc_goal_distance`
delta + `WorldModelVerifier.cell_recall` + frame-change novelty). Behavior
descriptor = (objects-moved-set, max-level, avatar-region) keeps a diverse archive
rather than collapsing onto one near-miss. Energy angle: the cleanest realization
of "energy as a non-AR generator" — energy that *fails de-novo* (P0.1) works here
because it operates over a population *seeded by real exploration*. Asset:
`experiment_4472.run_variant_attempt`, `experiment_4550` accounting,
`WorldModelVerifier.cell_recall`. Gate: on ≥2 of 8 hard games, QD puts a winner in
the offline-reproducible pool that pure-BFS does not at equal budget. Cites:
2605.28814, FunSearch (Nature s41586-023-06924-6), 2308.05483, 2504.01915,
2605.05138.

### #6-merged — Macro-action vocabulary induction (empowerment/affordances)
**Leverage MEDIUM · Cheapness MEDIUM-HIGH · the horizon-collapse ENABLER.**
`rich_action_candidates` emits ONE action; the hard games need multi-step plans.
Induce a per-game macro vocabulary by clustering exploration action-sequences by
frame-delta effect, keeping high-**empowerment**, controllable-and-reachable ones
(`push-until-blocked`, `cycle-color`, `toggle-then-step`). Run any plan search over
MACROS — a 13-action plan becomes a 3-macro plan, collapsing the 4^13 horizon into
the ~5n budget. Eigenoptions over the Go-Explore successor representation provide a
principled basis when frame-delta clustering is ambiguous. Energy angle:
empowerment IS an information-theoretic energy (channel capacity); the shared macro
library is the cross-game tier of the energy-config-space. Asset: `candidate_router`
hook, `rich_action_candidates`. Gate: macro-plan search reaches ≥1 hard-tail level
at LOWER expansion count AND banks ≥1 level primitives miss in budget. Cites:
2107.07031, 2502.02962, 2302.04693, 1810.04586, 2408.11816.

### #3 — Discriminative affordance world-model + Dijkstra over a success-graph
**Leverage HIGH · Cheapness HIGH.** Induce per-behaviour success probabilities
`q(state, behaviour) ∈ [0,1]` (behaviour = object-targeted primitive) with the
structural prior that a successful behaviour changes *exactly one* object-attribute;
edge-weight `−log q`; Dijkstra/A* to the induced goal; pool = Dijkstra paths.
`−log q` *is* an energy and Dijkstra is exact energy-minimizing path search. The
near-identity that *killed* exact-grid induction is here the *correct* prior (most
behaviours are no-ops). Asset: object-centric segmentation + `graph_explore_solve_v2`.
Gate: pool contains an offline-reproduced winner for ≥6/25 (6× baseline); retire if
< 3/25. Cites: 2408.11816, 2605.05138.

### #4 — PoE-World: divergence-TOLERANT product-of-programmatic-experts model + plan-through-model
**Leverage MEDIUM · Cheapness MEDIUM · the answer to why exact-match died.**
Replace monolithic exact-grid induction with a PRODUCT of small per-object Python
experts (`p(o_{t+1}) ∝ ∏ p_i^{θ_i}`), each induced from the Go-Explore trajectory by
the iGPU Qwen3.5-9B reasoner; prune low-weight experts; plan (MCTS/best-first)
through this probabilistic model toward `is_goal`. A wrong expert merely *lowers a
branch's probability* instead of gating the whole induction to 0/5. Energy angle:
`arc_world_model_trust_energy.select_trusted_world_model` ranks the expert SET by
held-out generalization (`verifier_is_oracle:false`, overfit-penalty 0.25·gap) —
the trust-regularized weight prior. Asset: `arc_world_model_trust_energy.py` (built,
unwired), `arc_llm_guided_solve.py`. Gate: `offline_reproduced=true` AND
`generic_transfer_rate > 0.08` on manufactured variants; retire if ≤ 0.08
("exact-match in disguise"). Cites: 2505.10819, 2605.05138, 2510.04542.

### #5 — Plan2Explore-as-energy: plan toward expected ensemble DISAGREEMENT, then exploit
**Leverage MEDIUM · Cheapness MEDIUM · purest dense-intrinsic-energy reuse.**
Maintain a small ensemble of induced world-models (trust-energy candidates); dense
per-step intrinsic energy = variance across predicted next-grids. Two-phase per
level: EXPLORE (plan toward max-disagreement = max info-gain about the unknown
mechanic until disagreement collapses) → EXPLOIT (switch to `is_goal`-distance and
plan the winner). The moat is already an ensemble, so disagreement is a free read;
gradient-free (no in-episode training — the constraint that kills RND/ICM). Asset:
`arc_world_model_trust_energy` → `graph_explore_v2.expansion_priority`. Gate:
`winner_generated` 1/25 → ≥4/25 on ≥2 hard games AND explore-phase disagreement
collapses (`final Var < 0.1 × initial`). Cites: 2005.05960, 1906.04161, 2605.05138.

### #7 — EFE planning wiring trust-energy + `is_goal` (one wire closes 3 of 4 gaps)
**Leverage MEDIUM · Cheapness MEDIUM · the AXIOM-style unifier.**
Combine the two built-but-unwired assets into an Expected-Free-Energy planner:
minimize `G = pragmatic (energy-distance of the predicted end-state to the
`is_goal`-satisfying set, rolled through the top trust-ranked model) + epistemic
(info-gain — actions disambiguating the top-2 trust-ranked models)`. The epistemic
term *automatically* spends budget on un-grounded HUD/hidden-state mechanics (where
models disagree). One wiring addresses multi-action plans + goal-not-verified +
grid-only-state. Energy angle: pragmatic + epistemic are both energy terms — makes
the trust-energy load-bearing for *generation*. Cites: 2505.24784, 2502.02962,
2009.08111, 2308.08029.

### #8 — Hidden-state register inference (diagnostic-first) for GAP-ARCH-GRID-ONLY-STATE
**Leverage LOW · Cheapness MEDIUM · run the cheap diagnostic EARLY.**
Augment grid state with a small library of latent symbolic registers (step-counter
mod-k, one-bit toggle, last-action memory, undo-depth); a particle filter / ensemble
infers their values; disagreement among latent-augmented models identifies which
register is real. The ONLY candidate that even *represents* hidden state — for a
HUD-dependent game the winning sequence is structurally absent from any grid-only
pool. First step is a **sub-second `cell_recall` diagnostic**: is the hard tail
search/goal-bound or structurally hidden-state-bound? Asset: `HIDDEN_STATE_GAME_IDS`,
`WorldModelVerifier` over augmented Transitions. Gate (diagnostic): exactly one
augmented model beats grid-only by ≥ +0.2 AND grid-only error does NOT collapse;
else log a missing-verifier gap in `ops/verifier_gaps.md` (honest, not a win).
Cites: 2510.12088, 2206.08332, 2505.10819.

---

## 3. The 3 (+1 diagnostic) to attempt next — sequenced

Sequencing logic: **prove the goal-energy wire first (cheapest, unblocks
everything), enable the horizon collapse second, then the non-AR energy-generator,
with the hidden-state diagnostic run early to remove ambiguity.**

1. **#1 — wire graded `is_goal` as a goal-energy target. FIRST, this is nearly
   free.** Proves whether the dominant gap (goal-not-verified) is real AND whether
   the precision-1.0 predicate *generalizes off its single training game* (its
   biggest unknown). Prerequisite for #2/#5/#7 — a wrong goal poisons every planner.
   This is the per-game goal-energy tier of the architecture.
2. **#6-merged — induce a macro vocabulary. PARALLEL.** Proves whether
   horizon-reduction is the binding constraint on the ~5n budget. The enabler that
   gives #2 fragments worth recombining. This is the shared (cross-game) energy-prior
   tier — the macro library transfers.
3. **#2 — energy-as-fitness QD evolution. THIRD, depends on #1+#6.** Proves whether
   energy can GENERATE (not just select) when seeded by real exploration — directly
   tests the P0.1 "energy fails de-novo" boundary under a population-seeded regime.
   Highest direct leverage on `winner_generated`. This is the generative-fitness tier.
4. **#8 diagnostic — cheap `cell_recall` hidden-state probe. EARLY/PARALLEL.**
   Sub-second measurement that tells us whether the 8-game hard tail is search/goal-
   bound (→ #1/#2/#6 suffice) or has a representational ceiling (#8 becomes
   load-bearing). Either answer prevents wasted effort.

Second wave once the bottleneck (goal vs horizon vs representation) is localized:
#3 (discriminative affordance + Dijkstra), #4 (PoE-World factored model), #5/#7
(Plan2Explore / EFE planner).

---

## 4. Honest risks / dead-end triggers

- **#1 single-game generalization (the dominant risk).** `exp4020`'s held-out
  precision 1.0 is **n = 6 on one game (r11l)** and operates on a *state dict*
  (`unsatisfied_targets == 0`), not the raw grid — wiring needs a state-featurizer
  that may not transfer, and its failure mode is *silent* (confident-wrong-state).
  The uniform-energy ablation control is mandatory. Dead-end trigger: `is_goal`
  fires wrongly on ≥2 non-r11l games → log a verifier-gap, retire as a universal
  target (keep as r11l-class only).
- **#2 lives under the P0.1 shadow.** Energy provably fails to generate de-novo;
  the entire bet is that population-seeding + non-null QD priority escapes it.
  Unproven. Also needs exploration to have *reached* the relevant intermediate
  state (where Go-Explore is 0/4 there may be no fragments to recombine) — which is
  why #6 sequences before/with it. Dead-end trigger: `winner_generated` delta ≤ 0
  across the tail after archive tuning → retire per Failed-Experiment Rerun
  Discipline.
- **#6 may find no real macros.** If induced macros never shorten any winning plan
  (horizon-reduction ≤ 1×), the affordance induction is finding noise; the
  200-step induction phase itself spends budget. Dead-end trigger: retire, fall back
  to primitives.
- **Cross-cutting — perceptual grounding (Sensi, 2603.17683, solved 0 on
  ARC-AGI-3).** All approaches sidestep it by operating on the LOGICAL grid (the
  existing pixel→logical decoder), not raw pixels — a strength, but a mechanic
  invisible at logical resolution defeats all of them.
- **Cross-cutting — action efficiency is unaddressed by generation.** These
  generators make a winner *exist*; they do not make it *efficient* (the actual
  `min(1.15, human/agent)^2` score). Sequence the Tufa-style clickability/
  frame-change predictor (the staged `.415/.416` lever) immediately after the first
  generator lands a winner — otherwise even solved levels score near-0.
- **Cross-cutting — "trust-energy ranks the right model" is itself a hypothesis to
  gate,** not an assumption (load-bearing for #4/#5/#7).

---

## 5. Mapping to the energy-config-space architecture (summary)

| Architecture tier | Concrete mechanism | Menu item | Asset |
|---|---|---|---|
| Per-game goal energy (online) | graded `is_goal` well, refined from level-ups | #1 | `exp4020`, `arc_goal_distance` |
| Per-game generative fitness (online) | QD evolution scored by rollout energy | #2 | `WorldModelVerifier.cell_recall` |
| Per-game epistemic energy (online) | ensemble-disagreement explore→exploit | #5 | `arc_world_model_trust_energy` |
| Shared affordance/macro prior (cross-game) | induced macro vocabulary + eigenoptions | #6 | `candidate_router`, Go-Explore SR |
| Shared goal-template prior (cross-game) | H1–H5 goal-hypothesis library | #7/#8 | `arc_agi3_goal_induction.py` |
| Shared model-trust prior (cross-game) | held-out trust-energy ranker | #4/#5/#7 | `arc_world_model_trust_energy.py` |

The live agent loop becomes: *seed* per-game energies from the shared priors →
*explore* under epistemic energy → *induce* the goal predicate → *refine* the
per-game energy landscape from observed transitions → *generate* multi-action plans
by energy-fitness over macros → *bank* the winner and *update the shared prior*.
That is the self-improving energy loop, applied to a fresh hidden game each time.

---

## 6. Citations (all confirmed real arXiv IDs)

2605.05138 (Executable World Models, ARC-AGI-3 SOTA), 2605.28814 (BES bidirectional
search), 2505.10819 (PoE-World), 2005.05960 (Plan2Explore), 2505.24784 (AXIOM),
2502.02962 (intrinsic motivation as constrained entropy maximization), 2009.08111
(Da Costa et al., reward via discrete active inference), 2504.14898 (EFE planning as
variational inference), 2308.08029 (Sophisticated Learning), 2507.12821 (Adaptive
World Models), 2107.07031 (empowerment in sparse-reward exploration), 2302.04693
(proto-goals), 2110.09514 (LEXA), 1810.04586 (Laplacian in RL), 1710.11089
(eigenoptions), 2408.11816 (discriminative object-centric world model), 2308.05483
(quality-diversity under behavior sparsity), 2504.01915, 1906.04161, 2510.12088,
2206.08332, 2510.04542 (Code World Models), 2309.05660 (Hypothesis Search), 2303.11366
(Reflexion), 2305.16291 (Voyager), 2006.08381 (DreamCoder), 2603.17683 (Sensi,
perceptual-grounding wall on ARC-AGI-3), 2602.10390. FunSearch: Nature
s41586-023-06924-6. NOTE: equation-level (2504.14898 Eq. 3/13) and per-game RHAE
numbers (2605.05138) must be re-read from source before any *paper-v6* citation.

---

## 7. Cross-references

- `docs/research-notes/arc-008-wall-root-cause-2026-06-21.md` — the root-cause
  unification this builds on.
- `ops/known-issues.md` — the `.425` MANDATORY-NEXT-MILESTONE flag derived from
  this note; the 2026-06-20 "energy-augmented ARC is the research spine" directive
  this extends.
- CLAUDE.md "ARC-AGI-3 IS a Live Hidden-Game Discovery Agent", "ARC Solve
  Reproducibility + Solver-Reuse Discipline", "Missing-Verifier Gap Logging",
  "Circularity / Oracle-Distinctness Discipline" (every value claim is
  `verifier_is_oracle:false`).
- `ops/verifier_gaps.md` — where #8's honest-negative outcome is logged.
- Memory: `project_arc_energy_config_space` (the operator's 2026-06-22 directive),
  `feedback_hybrid_pragmatic_architecture` (verifier-moat existential),
  `feedback_arc3_online_gated_on_offline_beating_baselines` (offline gate before
  any quota spend), `project_arc_agi3_north_star`.

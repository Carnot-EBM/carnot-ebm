# ARC-AGI-3 registry genre-prior distillation — scope (2026-07-12)

**Provenance:** operator question after a breadth-solving session: "do we have to do
anything to prepare the A/B verify to be able to use the updated registry? I am trying
to understand why the live mechanism was unable to classify the kind of game and find
an efficient search path within budget particularly as we have a pretty complete
registry of solved game levels now that should have helped. there is a gap in the
live agent here that deserves to be explored." Answer given inline: no prep is needed
(the registry has no caching anywhere it's read), but that was not why the live agent
underperforms -- there is a real, code-grounded architectural gap. This note scopes
what closing that gap would look like, gated the same way every other live-path
addition in this project is gated (see
`docs/research-notes/arc-agi3-run-local-cross-game-adaptation-scope-2026-07-12.md`,
whose structure this note mirrors).

**Explicitly NOT built yet.** This is a scope, not a commit. Nothing here should be
read as shipped or as changing current live-agent behavior.

## 1. What exists today (baseline, for contrast)

Traced directly from source, not inferred:

- `E3AgentPolicy.__init__` (`python/carnot/agentic/arc_competition_agent.py`) calls
  `_recommend_live_approach(self.short)` with `mechanic=None,
  early_play_signature=None` -- no registry-derived identity hint at construction
  time (correct: the game is hidden, there is no identity to look up yet).
- `_recommend_live_approach()` delegates to
  `arc_solve_learning.recommend_approach(target_game, mechanic=..., early_play_signature=...)`,
  falling back to `arc_strategy_router.route_for_game()` on exception.
- `arc_solve_learning.recommend_approach()` (line ~706): its docstring says "Route a
  NEW game to the closest proven recipe... Call this BEFORE reverse-engineering a new
  game." If `target_game` is not in the static 25-game survey
  (`results/arc3_win_condition_survey.json`) -- true for every hidden game -- it
  returns a COLD route (`confident_transfer: False`, `routing_confidence: 0.0`), no
  similarity transfer at all. If the game IS one of the 25 public games, it ranks
  OTHER solved games by feature similarity and **explicitly excludes the target game
  from its own similarity ranking** (`if gid == target_game: continue`). The function
  is built to route an unseen game to a *different* game's recipe; it structurally
  never retrieves a partially-solved game's *own* accumulated registry knowledge
  (`win_condition`, `gotchas`, `action_model`) as guidance for continuing that same
  game.
- `arc_strategy_router.detect_mechanic()` (line ~168) IS registry-self-aware at one
  point: it reads the target game's own structured `mechanic_class` field straight
  from `ops/arc_solve_registry.yaml` if present, else falls back to
  `DEFAULT_MECHANIC = "graph_explore"`. But `route_strategy()` then maps that single
  string to one of ~5 fixed strategy descriptors
  (`default_graph_explore`, `systematic_bfs`, `diversity_graph_explore`,
  `goal_distance_astar`, `llm_reasoner`) -- the entire effect of registry awareness
  here is picking *which generic search algorithm runs*, never injecting the
  mechanic's actual content.
- `E3AgentPolicy._maybe_route_from_transitions()` (the `RunLocalMechanicLedger`
  wiring, commit `ecb2b7bf9`) fires once after `_FEATURE_ROUTER_EARLY_PLAY_K = 8`
  transitions, classifying the mechanic via a **7-bucket, registry-BLIND behavioral
  taxonomy** (`avatar_navigation`, `click_connect`, `config_toggle`,
  `hidden_carry_state`, `keyboard_graph`, `click_graph`, `unknown`) derived purely
  from the game's own observed transitions -- it never reads `ops/arc_solve_registry.yaml`
  at all. Per `results/experiment_5578_run_local_ledger_concurrent_ab.json`'s own
  logged `feature_router_mechanic_class` values, this coarser 7-bucket classifier is
  what actually governs search bias for most of a run, superseding whatever
  registry-derived strategy signal `detect_mechanic()` picked at construction.
- `ops/arc_solve_registry.yaml` is read fresh, uncached, on every call to
  `arc_solve_learning._registry()` -- so **no staleness problem exists**; any consumer
  that wanted to read richer registry content today already could, with zero prep.
  That was the literal question this note answers "no" to.

**The gap, stated precisely.** Even in the best case -- a target game IS one of the 25
public survey games, has a registry entry, and `detect_mechanic()` picks a non-default
strategy class -- the live agent's search is biased only by *which of ~5 generic
algorithms runs*. None of the actual hard-won structural knowledge this project has
accumulated ever reaches the search as a usable prior:

- dc22's checker-teleport-star topology (paired teleport sockets gated by a remote
  toggle's phase)
- bp35's 228-action hard per-level timer (gravity-tower puzzles budget action count,
  not just position)
- lf52's leapfrog-pair jump grammar (chained carrier hand-offs with a specific pairing
  order)
- sk48's exact color-ordering requirement (a nearer reachable position can still be
  wrong because a wrong-color block gets seen first)

A game mapped down to the pixel in `ops/arc_solve_registry.yaml`'s `gotchas` /
`win_condition` / `novel_mechanics_found` prose still leaves the live agent exploring
from a near-blank slate, because nothing distills that prose into anything the search
loop can consult.

## 2. What closing this gap would actually require

### 2.0 Preconditions

**Does this violate the live-path memorization boundary?** No -- this is the same
question the run-local-ledger scope already resolved, and the operator's ruling there
governs here too, quoted again in full because it is the load-bearing rationale for
this note's entire design:

> "knowing the rules of a game or a class of games and how to play them is not the
> same as memorization, it is learning and adapting which is the spirit of the
> challenge. starting from a completely naive position would also preclude the use of
> LLMs entirely. as such, we assert that as long as we are not playing back a
> prerecorded sequence of events into a game that we pretrained offline, we are fine."

The operative distinction, applied to THIS design: **FORBIDDEN** = surfacing a
game-identity-keyed action sequence or a `GameAdapter`-equivalent solved-instance
recipe to the live search for a hidden game (this is already excluded from the live
path per "ARC Live-Path Reachability Discipline" and is not proposed here).
**PERMITTED** = surfacing GENRE/CLASS-level structural knowledge -- "puzzles with a
`config_toggle` mechanic sometimes require a toggle to be hit TWICE to create a
temporary state, not just once" is a rule about a *class* of games, distilled from
having seen it across several registry entries, not a memorized answer to one game.
This note's design must stay on the PERMITTED side of that line by construction:
distilled priors are class-level generalizations mined FROM MULTIPLE games'
`gotchas` text, never a single game's own action sequence or per-instance solution,
and never keyed to resolve to a specific hidden game's identity.

**Does the target game need to be in the 25-game survey for this to help?** No, and
this is the actual point of the design (unlike `recommend_approach()`'s similarity
transfer, which requires the target to be a known survey entry). The distilled priors
this note proposes are indexed by MECHANIC CLASS (the same 7-bucket behavioral
taxonomy `_maybe_route_from_transitions()` already computes from a hidden game's own
transitions, registry-blind), not by game identity -- so they apply to a genuinely
hidden game the instant its mechanic class is classified from its own play, which is
exactly the existing `_FEATURE_ROUTER_EARLY_PLAY_K = 8`-transition trigger point.

### 2.1a PILOT RESULT (2026-07-12) — the mechanical heuristic is a confirmed, decisive negative

Built `python/carnot/agentic/arc_genre_prior_distillation.py` (the harness: load,
group by coarse mechanic class via `arc_solve_learning._coarse_mechanic_class`,
mine, filter, write) with `heuristic_shared_phrase_propose` as the default
`propose_fn` (plain 4-9 word near-verbatim phrase overlap across >= 2 non-near-
duplicate games), and ran it once, by hand, against the live registry, per this
note's own §4 rollout order. Full result:
`results/experiment_5581_genre_prior_distillation_heuristic_pilot.json`.

**604 raw candidates, 140 surviving an 18-token boilerplate-marker filter, 0
(zero) expressing genuine game-mechanic content** after manually reading every
surviving candidate across all 4 populated classes. Every single one is drawn
from the outer loop's own formulaic verification/status narration --
`"round 9 2026 07 12 gpt 5 6 sol via"`, `"frontier stays at level"`,
`"no new level banked"`, `"still honest null"`, `"all 5 confirmed level"` -- not
a color, an object behavior, a spatial relationship, or a timing rule anywhere
in the surviving set.

**Root cause.** Registry `gotchas` prose mixes two kinds of content in the same
free-text fields: game-mechanic description (necessarily game-SPECIFIC in exact
wording -- colors, coordinates, object names differ per game even when the
mechanic class matches) and the outer loop's own verification narration
(written in a HIGHLY repetitive template across every entry, because the same
author writes every gotcha with the same habitual phrasing). Literal n-gram
overlap is structurally biased toward surfacing the second kind, which is
exactly backwards from what a genre-prior index needs.

**Conclusion.** This sharpens the mining-mechanism question from "the
mechanical heuristic is a weak first pass, worth trying before spending LLM
budget" (this note's original framing) to a confirmed, decisive negative: a
lexical/heuristic `propose_fn` on this corpus is NOT a viable path to a useful
genre-prior index, full stop. `ops/arc_genre_priors.yaml` from this pilot run
is deliberately NOT checked in -- its content would be pure noise if a future
session mistook it for usable. The harness itself (load / group / independence
filter / write) is retained as correct, reusable infrastructure: `propose_fn`
was built pluggable specifically so an LLM-backed proposer can be substituted
without touching any of that. **The remaining open work on this scope is
entirely in swapping in a semantic (LLM-assisted) `propose_fn`** -- §2.1 below
describes that mechanism as originally scoped; it is now the ONLY mechanism
worth pursuing further, not one option among several.

### 2.1 Component: `RegistryGenrePriorIndex` (the distillation pass)

An OFFLINE, dev-only build step (development_proxy, same boundary as
`arc_solve_learning._survey_features()` and every `GameAdapter`) that mines
`ops/arc_solve_registry.yaml`'s free-text `gotchas` / `win_condition` /
`novel_mechanics_found` fields across ALL games, and produces a small, structured,
checked-in artifact keyed by the SAME 7-bucket mechanic-class taxonomy
`arc_solve_learning.classify_early_play_mechanic()` already uses:

- Storage: `ops/arc_genre_priors.yaml` (or embedded as a new top-level key in
  `ops/arc_solve_registry.yaml` -- decide at implementation time; a separate file
  keeps the distillation pass's blast radius visible in diffs).
- Per mechanic class, a SHORT list of GENERALIZED, class-level prior statements mined
  from >= 2 DIFFERENT games' gotchas text discussing that class (a prior sourced from
  only one game is a memorization risk in disguise -- see 2.5). Example shape (the
  content is illustrative, not yet built):

  ```yaml
  config_toggle:
    priors:
      - text: "a toggle may need to be activated TWICE at different route phases to
               create a temporary passable state, not just once"
        sourced_from: [dc22, sk48]
      - text: "toggle state can persist across a level-reset action; verify before
               assuming a fresh toggle each attempt"
        sourced_from: [g50t, ka59]
  keyboard_graph:
    priors:
      - text: "hard per-level action-count timers exist independent of the session
               budget; probe the timer cheaply (a no-op action) before committing to
               a long route"
        sourced_from: [bp35, g50t]
  ```

- Mining mechanism: an LLM-assisted extraction pass over each game's `gotchas` text
  (an agent task, not hand-authored) that (a) reads all games' gotchas grouped by
  their registry `mechanic_class` field, (b) proposes candidate generalized priors,
  (c) REQUIRES each candidate prior to be attributable to >= 2 distinct games before
  inclusion (mechanical filter, not just LLM judgment -- see 2.5's adversarial check).
  This mirrors the existing SOTA-ingestion-cycle pattern (an agent synthesis pass
  producing a structured artifact) rather than inventing a new agent-orchestration
  shape.
- This index is REBUILT periodically (each time the registry grows meaningfully),
  not maintained incrementally by hand -- it is a derived artifact, checked in like
  `results/arc3_win_condition_survey.json` already is.

### 2.2 Component: confidence-gated consultation at the live `_maybe_route_from_transitions()` hook

- The SAME hook that already fires the 7-bucket behavioral classification
  (`E3AgentPolicy._maybe_route_from_transitions`, after `_FEATURE_ROUTER_EARLY_PLAY_K`
  transitions) additionally looks up `RegistryGenrePriorIndex[classified_mechanic]`
  and, IF that mechanic class has priors, folds them into the SAME confidence-gated
  nudge path `_maybe_route_from_transitions` already uses -- not a new, separate
  behavior-changing pathway.
- Bounded and reversible by construction, matching the run-local-ledger's own
  discipline: priors can only nudge EXISTING knobs already read by the search
  (`uses_goal_distance_heuristic`, `explore_budget`, and analogous existing
  parameters -- e.g. a `config_toggle` prior about double-activation could bias the
  search to retry a toggle before abandoning a branch that looked exhausted). Priors
  MUST NOT introduce a new search algorithm or a new top-level solving strategy (same
  restriction the run-local-ledger scope's Out-of-Scope section already states for
  itself, §3 below).
- A prior is TEXT, not code -- for a first version, "fold in" means feeding the
  matched prior strings into the SAME budget/heuristic knobs via a small, fixed
  keyword-to-knob mapping (e.g. a prior mentioning "twice" / "double" nudges a
  retry-before-abandon counter), NOT an LLM re-reading prose live inside the search
  loop (that would add real latency and a new failure surface for a benefit this
  design has not yet earned). If the mined priors turn out to need genuine natural-
  language reasoning to apply usefully, that upgrade is a separate, later-gated
  follow-on, not part of a first version.

### 2.3 Component: kill switch

- An env/config flag (`CARNOT_ARC_GENRE_PRIOR_DISTILLATION`, default **off**),
  matching the existing `CARNOT_ARC_RUN_LOCAL_ADAPTATION` /
  `SUBMITTED_*_ENABLED` flag pattern already in `arc_competition_agent.py`. Ships off
  until 2.4 clears.

### 2.4 Empirical validation (REQUIRED before any default-on flip)

Build an offline harness, matched-compute against the SAME `exp4582` /
`experiment_5578_run_local_ledger_concurrent_ab` methodology (bootstrap CI on the
delta, CI-excludes-baseline bar):

- Roster: public games spanning several mechanic classes, WITH the genre-prior index
  built from the OTHER games only (leave-one-game-out per class, so a game's own
  gotchas never leak into its own prior lookup during validation -- this is the
  offline analogue of 2.0's "never a single game's own solution" boundary, made
  testable).
- Measure: median actions-to-first-levelup / levels-per-action with the prior
  consultation ENABLED vs. an identical run with it DISABLED, same budget.
- Positive control: confirm the prior lookup actually fires at least once per
  matching-mechanic-class game in the simulated roster (a correctness check on the
  wiring itself, separate from the efficacy question -- same shape as the run-local
  ledger's "confirm the confidence gate fires at least once" control).
- If null (plausible -- text-mined priors folded through a coarse keyword-to-knob
  mapping is a weak mechanism, and the adjacent `exp4582` transfer-routing result was
  ALSO a null): retire per the Failed-Experiment Rerun Discipline, record the honest
  result, do not silently half-ship it enabled.

### 2.5 Adversarial check (before scaling / before any default-on flip)

- **Memorization leak-through, the primary risk.** Could a "genre-level" prior
  actually be a single game's specific solution in disguise, laundered through the
  >= 2-games-sourced filter by two very similar public games? The mining pass must
  reject priors whose only two sourcing games are near-duplicates of each other
  (same `mechanic_class` AND high feature-similarity per `arc_solve_learning._similarity()`)
  -- require sourcing diversity, not just a source count.
- **Prior mis-application across superficially-similar-but-actually-different
  mechanics.** The 7-bucket taxonomy is coarse; a `config_toggle` prior mined from
  dc22/sk48 might not apply to a hidden game that also classifies as `config_toggle`
  but has a structurally different toggle semantics. The confidence-gated nudge (2.2)
  must stay small enough that a WRONG prior costs little (bounded knob nudge, never a
  strategy swap), and 2.4's validation harness should include at least one held-out
  game per class specifically to measure false-application cost, not just
  true-application benefit.
- Standard QA-layer checks per the sibling discipline in CLAUDE.md: field-shape
  assumptions (principle-wrapped fields in the registry), off-by-one confidence
  thresholds, a silent no-op (the lookup key never matching anything real) reading as
  active behavior in a green test.
- Independent read of whether ANY genre-level prior consultation is within the spirit
  of "discover what winning looks like" per hidden game -- same check the run-local
  ledger scope ran first, cheaply, before its expensive validation work. Given the
  operator ruling already covers genre/class-level learning explicitly (2.0), this
  check here is narrower: confirm the MINING pass itself (2.1) never accidentally
  captures a game-identity-keyed fact as if it were class-level, not re-litigate the
  ruling itself.

## 3. Explicitly out of scope for a first version

- No prior sourced from fewer than 2 distinct, non-near-duplicate games.
- No prior consultation outside the existing confidence-gated nudge path -- no new
  top-level solving strategy, no LLM-in-the-loop prose reasoning during search (2.2).
- No persistence of anything OBSERVED during a live run back into
  `ops/arc_genre_priors.yaml` or `ops/arc_solve_registry.yaml` -- the distillation
  pass (2.1) is an offline, dev-only build step; the live path only ever READS the
  checked-in index, mirroring the run-local-ledger's "never touches the checked-in
  registry from the live path" rule.
- No change to the 7-bucket mechanic taxonomy itself -- this design consumes the
  existing taxonomy as its index key, it does not redesign it.
- No cross-submission memory beyond what is already checked into
  `ops/arc_genre_priors.yaml` between dev sessions (that file IS the durable, git-
  tracked memory; there is no additional live run-to-run persistence layer proposed
  here).

## 4. Rollout order

1. §2.0 preconditions confirmed (already cleared -- this note's 2.0 reduces to citing
   the run-local-ledger scope's existing operator ruling; no new sign-off blocks
   starting the prototype).
2. `RegistryGenrePriorIndex` distillation pass (2.1), run once by hand as a dev-only
   script producing `ops/arc_genre_priors.yaml`; inspect the output for the
   memorization-leak risk (2.5) BEFORE wiring any consumer.
3. Confidence-gated consultation wiring at `_maybe_route_from_transitions()` (2.2),
   shipped OFF by default (2.3).
4. Offline leave-one-game-out A/B harness (2.4); run it.
5. Adversarial check (2.5) on whatever the A/B finds, win or null.
6. Only flip default-on if the A/B clears a real, CI-excludes-baseline bar. Otherwise
   retire per Failed-Experiment Rerun Discipline and record the honest null.
7. Either outcome: log to `ops/verifier_gaps.md`-adjacent tracking or a dated
   corrigendum note so a future session doesn't re-propose the same idea without
   citing this scope + its result.

## Cross-references

- `docs/research-notes/arc-agi3-run-local-cross-game-adaptation-scope-2026-07-12.md`
  -- the sibling scope this note mirrors in structure and whose operator ruling on
  the memorization/genre-learning boundary this note reuses directly
- `results/experiment_4582_feature_router_transfer.json` -- the offline null for
  cross-game SIMILARITY transfer (a different mechanism: routing an unseen game to
  another SPECIFIC game's recipe); this note's design is deliberately at the coarser
  mechanic-CLASS level, not the specific-game level, precisely because the
  specific-game level already has a recorded null
- `results/experiment_5578_run_local_ledger_concurrent_ab.json` -- the concurrent A/B
  whose logged `feature_router_mechanic_class` values were the direct evidence that
  the 7-bucket behavioral classifier (not any registry-derived strategy signal)
  governs most of a run's search bias, motivating this note's choice to index priors
  by that same taxonomy rather than inventing a new one
- `python/carnot/agentic/arc_solve_learning.py` (`recommend_approach`,
  `_registry`, `_solved_games`, `_similarity`, `extract_early_play_signature`,
  `classify_early_play_mechanic`) -- the functions read to establish the baseline in
  §1
- `python/carnot/agentic/arc_strategy_router.py` (`detect_mechanic`, `route_strategy`,
  `_BY_MECHANIC`, `DEFAULT_MECHANIC`) -- the one point registry data reaches the live
  path today, and the ceiling of what it currently accomplishes (picking among ~5
  strategy descriptors)
- `python/carnot/agentic/arc_competition_agent.py`
  (`E3AgentPolicy._maybe_route_from_transitions`, `RunLocalMechanicLedger`,
  `_FEATURE_ROUTER_EARLY_PLAY_K`, `_FEATURE_ROUTER_MIN_CONFIDENCE`) -- the existing
  live hook this note's §2.2 extends rather than replaces
- `ops/arc_solve_registry.yaml` -- the source corpus the distillation pass (2.1)
  mines; also the file whose freshness (no caching anywhere it's read) resolved the
  operator's literal "do we need to prepare anything" question
- CLAUDE.md "ARC-AGI-3 IS a Live Hidden-Game Discovery Agent" -- the foundational
  framing this design must serve: the deliverable is the live runtime discovery
  PROCESS, and this note's priors are class-level generalizations that process can
  apply to a genuinely unseen game, not a memorized per-game answer
- CLAUDE.md "ARC Live-Path Reachability Discipline" -- the source-reading /
  live-vs-development_proxy boundary the distillation pass (offline, dev-only) and
  its consumer (live, reads a checked-in index only) both respect
- CLAUDE.md "ARC Solve Reproducibility + Solver-Reuse Discipline" -- the
  capture-and-reuse ethos this note extends from per-game solver code to cross-game
  TEXTUAL knowledge
- CLAUDE.md "Phase Prototype + Empirical Validation + Adversarial Check Discipline"
  -- the three-part structure (§2.1-2.3 prototype, §2.4 validation, §2.5 adversarial
  check) this scope follows
- CLAUDE.md "Failed-Experiment Rerun Discipline" -- governs what happens if §2.4
  comes back null

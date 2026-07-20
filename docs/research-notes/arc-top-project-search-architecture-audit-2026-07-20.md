# ARC-AGI-3 top-project SEARCH / PLANNING architecture audit (2026-07-20)

**Status:** RESEARCH / ARCHITECTURE AUDIT. No live-path file modified, no experiment code written. This
note deep-reads the actual planning/search-decision code of the three open-sourced Milestone-1 winners
(cloned in `external/`) and compares it, mechanism-for-mechanism, to Carnot's own live search — to answer
one operator question: **are we stuck because we tweak peripheral perception/scoring components while the
winners win via a fundamentally different search/planning architecture we never examined?**

**Reads as required input (per CLAUDE.md):** "ARC-AGI-3 IS a Live Hidden-Game Discovery Agent" (foundational
framing), "ARC Live-Path Reachability Discipline", "Literature Priority Discipline", "SOTA-Ingestion Cycle
Discipline", "Phase Prototype + Empirical Validation + Adversarial Check", "Missing-Verifier Gap Logging".

**Why this note exists.** Tonight's session ran SEVEN consecutive live-path A/Bs (REQ-ARC-FCP-5590/5728/5729/
5730/5732/5740/5756) — a frame-change CNN, an action-effect blend weight, a validation-gate tolerance, an
object-affordance prior, an inert-click pruner — all a clean honest null (zero level-gain delta on an 11-game
roster). None touched the SEARCH/PLANNING layer. `ops/verifier_gaps.md :: GAP-ARCH-NO-HIERARCHICAL-SEARCH`
(line 2544) has sat open since 2026-06-11, asserting subgoal decomposition is "the single biggest lever."
The prior winner audit (`arc-agi3-milestone1-winners-sota-ingestion-2026-07-11.md`) extracted five
perception/scoring opportunities but **never read the winners' actual planning/search-decision code** — this
note is the gap it left.

**Headline finding, stated up front (and it is NOT the expected one).** All three winners are **greedy,
single-commit-per-turn generators with NO model-based lookahead, NO tree/beam/MCTS search, and NO subgoal
decomposition.** Carnot already has **strictly more** search machinery than any of them: a systematic
state-transition graph search (BFS/A*) with RESET-replay navigation *and* a second, model-based
`plan_in_model` lookahead tier over an induced+verified world model, *and* a working (if narrow)
hierarchical subgoal-search prototype. **The winners won WITHOUT the thing GAP-ARCH-NO-HIERARCHICAL-SEARCH
says is the biggest lever.** Their edge is elsewhere — orientation-time inspection compute + object-centric
perception + a stronger local generator — which corroborates this project's own repeatedly-reached
generation-not-selection / perception-binding-constraint diagnosis, not the search-depth hypothesis. Full
evidence below; the honest consequence is to **down-weight the hierarchical-search lever**, not chase it.

---

## 1. Per-project findings

Each project is scored against the five questions from the task: (Q1) multi-step lookahead vs greedy; (Q2)
exploration strategy for an unknown game; (Q3) explicit goal/subgoal representation; (Q4) LLM-vs-search
division of labor; (Q5) cross-level/cross-game persistent learning.

### 1a. Duck Harness — Tufa Labs, 1st place (the real framework, the actual winner)

The "solver" is an orchestration adapter, not the decision-maker. `HarnessSolver`/`_HarnessGameSession.play()`
(`external/duck-harness/ARC3-Inference/inference/framework/solver.py:263`) is a thin event loop: `while not
should_stop(): result = self.analyzer.analyze(...)`. Every actual decision is delegated to the analyzer — a
`ToolAgent` LLM — which itself executes real actions through the `step_env`/`_execute_action` callback
(`solver.py:588,667`). The deterministic change check the prior audit flagged lives here:
`board_changed = previous_grid != _grid_from_state(new_state)` (`solver.py:703`). **There is no search code in
solver.py at all** — no tree, no frontier, no graph.

The planning IS the LLM's tool loop (`inference/agent/tool_agent.py`):

- **The only tool is `python`** (`tool_agent.py:1258-1280`). The model is instructed to inspect the frame and
  "search or score candidate actions or short sequences against the current goal" and may "call `action(actions)`
  more than once in one Python snippet if your search or control loop needs it" (system/user prompt,
  `tool_agent.py:1227,1233-1234,1921`). So Duck's "search" is *the LLM writing Python that embeds a control
  loop*, not a classical planner.
- **Q1 (lookahead): NO model-based lookahead.** `analyze()` runs an inner loop of up to
  `_LOCAL_ANALYZER_TOOL_STEPS = 12` turns (`tool_agent.py:140,1783`), but as soon as ANY real action executes it
  breaks (`if step_executed: break`, `tool_agent.py:1976-1977`). Critically, `_handle_action` calls
  `self._step_env_callback` — **the REAL environment** — with no copy, no simulator, no rollback, no undo
  (`tool_agent.py:1495-1545`). Any multi-`action()` "search" the LLM writes *permanently advances the real
  game*. This is greedy forward execution with real, irreversible steps — not planning against a model. The
  "12 turns" buy INSPECTION time (write code, read `segmentation`, no commit), i.e. orientation-time compute
  before a single greedy commit, not tree search.
- **Q2 (exploration strategy):** entirely prompt-scaffolded — "investigate first," maintain a working world
  model, inspect via Python, "program a small search or scorer over candidate actions" (`tool_agent.py:1907-
  1924`). No coded hypothesis-space data structure, no systematic action-class enumeration in code.
- **Q3 (goal representation): free-text, not structured.** A "scientist note" with `World model / Goal model /
  Action model / Recent findings / Open questions / Plan / Cross-level notes` is parsed from assistant text
  (`_extract_scientist_note`, `tool_agent.py:263-296`), carried across turns, and **cleared on level-up** —
  except `cross_level_notes`, which is deliberately NOT cleared (`_update_summarized_knowledge_from_step_summary`,
  `tool_agent.py:1113-1126`). It is an NL goal *hypothesis*, never a landmark/subgoal graph planned toward.
- **Q4 (LLM/search division): none — the LLM is the entire planner.** Perception is the one thing split out:
  frames are pre-segmented into 4-connected objects with a translation-invariant `_object_hash`, a `children`
  containment tree, and an `adjacency_list` (`inference/utils/segmentation.py:65-73,97,103`), fed to the model
  as `current_frame.segmentation` — "the primary view; the raw numeric grid is not available"
  (`tool_agent.py:163,1225`). This is the object-centric perception O4 flagged, confirmed load-bearing.
- **Q5 (cross-game memory): none.** The session resets per game (`_ensure_session` keys on the runtime dir,
  `tool_agent.py:975-984`); `_history_messages` keeps ~30 assistant turns *within* a game; `cross_level_notes`
  persists across LEVELS of one game. No learned heuristic or memory transfers across games.

**Duck in one line:** a capable local LLM (Qwen 3.6 27B) given rich object-centric perception + up to 12
REPL inspection turns of orientation-time compute, committing exactly one greedy action-sequence per turn to
the real env. Its edge over Reki/forge is *more orientation compute + a bigger generator*, not search depth.

### 1b. Reki — 2nd place (Kaggle notebook, Gemma-4-31B-it)

`MyAgent.choose_action` (`external/arc-m1-2nd-reki/…ipynb`, code cell 3, line 417) is a **single VLM call per
step** feeding a tiny plan queue: build prompt → one `_generate_response` (line 643) → parse 1..`MAX_PLAN_ACTIONS`
JSON actions (`MAX_PLAN_ACTIONS = 4`, line 110) → push to `pending_actions` (line 551) → dequeue one per env
step (`_dequeue_action`, line 849).

- **Q1: greedy, no search.** The model emits a 1-4 action micro-plan and the agent executes them one at a time;
  a reflection can flush the queue (line 1008). No tree, no rollout, no simulator.
- **Q2:** prompt-driven ("Infer the controllable object, causal action effects, and current objective. Prefer
  purposeful new states. A repeated state is not progress." line 620-622) plus deterministic guards:
  `failed_state_actions` (per-frame-hash ineffective-action set, line 938-942,1210-1217), `_record_deadsig`
  inert-click pruning (line 1206 — the O1 mechanism), and `_salient_click_coordinate` saliency-tiered fallback.
- **Q3:** an NL reflection memory Markdown doc with a `## Goal` heading (line 962), rewritten every
  `REFLECTION_INTERVAL = 10` steps by `_run_reflection` (line 108,1004) and re-injected each prompt as
  "authoritative but revisable" (line 630). No structured subgoal.
- **Q4:** the LLM is the policy; deterministic helpers only prune/fallback around it. No search module.
- **Q5:** reflection memory persists to disk per game (`<game_id>.md`, line 182-192), so it survives context
  eviction *within a game*; deadsigs + ineffective-action sets reset per level (line 1157-1168). No cross-game.

### 1c. forge — 3rd place (Kaggle notebook, Gemma-4-31B-it)

Structurally near-identical to Reki (same VLM, same reflection memory, same 1-4 action plan queue,
`choose_action` line 510). Its one distinguishing seam is a **candidate-generate-then-select** step
(`_generate_action_response`, line 784): if `candidate_count > 1`, sample N candidates from one prompt, score
each with `_candidate_static_score` (a hand-tuned keyword heuristic — reward "goal/win/complete/…", penalize
"random/guess/…", reward click-inside-content-bbox, line 894-927), take argmax, then optionally re-select with
`_select_candidate_with_arbiter` (a **second** LLM call judging candidates, line 980-1040).

- **But the WINNING configuration disables all of it.** `PROFILE_ENV` for `gemma31b_public_single` (line 8)
  sets `LLM_ACTION_CANDIDATES=1` and `LLM_CANDIDATE_ARBITER=0`, so `candidate_count <= 1` collapses to a single
  greedy response (line 791-802) with no arbiter and no static scoring. Under competitive pressure the 3rd-place
  team CUT its only selection machinery for cost — leaving a greedy single-candidate policy identical in shape
  to Reki. (This is the exact ablation the prior audit's O2 cited; confirmed at source here.)
- **Q1-Q5:** same answers as Reki — greedy, no lookahead, NL goal in reflection memory, LLM-is-policy, per-game
  memory only. forge's candidate arbiter is the *weakest* form of "search" of the three and it was turned off.

---

## 2. Direct comparison table (Carnot vs each winner)

| Axis | Carnot LIVE (`E3AgentPolicy`) | Duck (1st) | Reki (2nd) | forge (3rd, winning cfg) |
|---|---|---|---|---|
| **Planning depth** | **Two tiers:** (1) systematic graph search w/ RESET-replay nav (`StepwiseExplorer`, `arc_competition_agent.py:678`); (2) **model-based lookahead** `plan_in_model` (bounded ~20000-node search inside an induced+verified world model, `arc_competition_agent.py:3374,3806`) | Greedy 1 commit/turn; up to 12 REPL *inspection* turns before commit; real-env steps, **no simulator/rollback** (`tool_agent.py:1495,1976`) | Greedy; 1-4 action micro-plan queue, executed 1/step (`MAX_PLAN_ACTIONS=4`) | Greedy single candidate (arbiter disabled in winning cfg) |
| **Exploration strategy** | Salience-tiered candidate generation + frontier expansion + online discriminative pruning + go-explore archive + curiosity (`arc_graph_explore.py:117`, `StepwiseExplorer`) | LLM-directed via prompt; writes Python to inspect + score | LLM + deadsig/ineffective-action pruning + salient-click fallback | LLM + hand-tuned static candidate score (disabled in winning cfg) |
| **Goal representation** | Structured: goal-energy bias + goal-predicate gate + verified subgoal predicates (vc33 prototype `arc_vc33_hierarchical_search.py:206`) | Free-text "Goal model" in scientist note (`tool_agent.py:263`) | Free-text `## Goal` in reflection MD (rewritten /10 steps) | Free-text `goal_hypothesis`/reflection MD |
| **LLM / search division** | **Split:** small local LLM (or CNN-prior TTT engine) does world-model INDUCTION only; a separate classical search decides actions (`_induce_and_plan` → `plan_in_model`, `arc_competition_agent.py:3734`) | **Fused:** LLM writes the code that inspects, scores, and acts. Perception (segmentation) split out | Fused (LLM is policy) | Fused (LLM is policy) |
| **Cross-level / cross-game memory** | Offline: reproduction-gated registry + trained cross-game value head/router/frame-change scorer, distilled between runs; go-explore + similarity retrieval online | Within-game NL note; `cross_level_notes` across levels; none cross-game | Per-game reflection MD persisted to disk; none cross-game | Per-game reflection MD; none cross-game |
| **Generator in the ACTION loop?** | **No** — generator only induces the model; search picks actions | **Yes** — LLM picks every action directly | **Yes** | **Yes** |
| **Local generator size** | Qwen3.5-9B-MTP (iGPU/16GB Kaggle constraint) | Qwen 3.6 **27B** | Gemma-4 **31B** | Gemma-4 **31B** |

Two facts jump out of the table: (a) Carnot is the ONLY entry with model-based lookahead and structured
subgoals — it has *more* planning machinery, not less; (b) every winner puts a **larger** generator directly
in the action-decision loop, while Carnot uses a **smaller** generator only for induction and lets search
decide. The differentiator is not search depth; it is where the capable-generator + orientation compute sits.

---

## 3. The actual capability gap, stated precisely

**There is no "we are missing hierarchical/lookahead search" gap. The evidence is the opposite.** Carnot's
live scored path already implements the full EXPLORE → INDUCE → VERIFY → PLAN → EXECUTE machine
(`arc_competition_agent.py:2702-2712`), where PLAN is a bounded best-first search *inside a verified world
model* (`plan_in_model`, model sourced from `LocalGGUFProposer` or the CNN-prior-warm-started TTT engine gated
at ≥0.5 held-out trust, `arc_competition_agent.py:3800-3812`). Its offline dev twin adds A* with a
move-distance heuristic, HUD masking, QD sequences, frontier-seed banks, and move pruners
(`arc_graph_explore.py:351-980`). And `arc_vc33_hierarchical_search.py` is a genuine subgoal-decomposed
best-first planner (`decompose_goal_predicate` line 206, `hierarchical_best_first_search` line 281).

**Why vc33-hierarchical never generalized past L1 is the whole story.** It is not stuck because the *search* is
weak — the search is generic and correct. It is stuck because it consumes two per-game INDUCED inputs it can
only get for vc33: a verified `predict(grid, action)` simulator (`arc3_vc33_world_model_program.py`) and an
induced goal predicate (exp4034), loaded by `load_exp4035_preconditions` (`arc_vc33_hierarchical_search.py:378`).
**Any lookahead planner is only as good as the world model + goal predicate it plans against, and inducing those
from a small local generator is exactly where Carnot fails** — the `plan_in_model` docstring names it directly:
"The induce/verify/plan quality (esp. from a small LOCAL model) is the open milestone" (`arc_competition_agent.py:2710`).
The freshest measurement agrees: REQ-ARC-WMTE-5720's induce-completion rate is **0/12** on the stalled roster
(`results/experiment_5724_…json:165`). The search has nothing correct to plan over.

So the gap is: **candidate GENERATION + world-model INDUCTION + perception, not search/selection/planning depth.**
The winners route around this gap by (a) using a **bigger generator** (27-31B vs our 9B) that needs less
explicit world-model scaffolding, (b) giving it **orientation-time inspection compute** in the action loop
(Duck's 12-turn REPL), and (c) feeding it **object-centric perception** (segmentation as primary view). They
have no lookahead, no subgoals, no verified simulator — and they still beat a flat greedy policy because the
generator itself is doing the "planning" implicitly at commit time. This is the same generation-not-selection
diagnosis this project already reached (corroborated by ZendoWorld arXiv:2607.08233, by
`project_arc_live_agent_learning_gaps`, and by tonight's 7 selection/perception-tweak nulls), now confirmed
from the winners' own source: **a better SELECTOR/PLANNER cannot recover an action the GENERATOR never proposed
or a WORLD MODEL that was never correctly induced.**

**Consequence for `GAP-ARCH-NO-HIERARCHICAL-SEARCH`:** down-weight it. The 2026-06-11 "single biggest lever"
claim (verifier_gaps.md:2548) is not supported by the winners' evidence — the actual winner has strictly less
search than Carnot and won on generator + orientation + perception. Hierarchical search only helps once the
world model and goal predicate it plans over are correctly induced, which is the true bottleneck. (This does
not *close* the gap — a hierarchical planner remains the right consumer of a good induced model — but it
re-orders it below induction/perception, and it stops the "add MCTS" reflex.)

---

## 4. The ONE proposed next experiment

**Do NOT** port a lookahead/MCTS planner (Carnot already has more search than the winners; §3 shows the
binding constraint is upstream). **Do NOT** run another "modify a component, measure level-gain delta" A/B —
that is exactly tonight's null pattern (7×). Instead, run the one measurement that decides *where* the gap
actually is, and that the whole nightly investigation orbited without ever measuring.

**Title (working):** Candidate-coverage attribution — partition Carnot's score gap into GENERATION vs
SELECTION vs PLANNING on the stalled games (offline, LLM-free).

**One-paragraph statement.** For every state on Carnot's own stalled-game offline replays (the games where the
live agent gets 0 new levels), and — where the traces are thick enough — for the states in the winners' own
recorded public-game trajectories (Duck ships per-game transcripts+prompts at
`external/duck-harness/example-run/{transcripts,prompts}/`, e.g. `ar25-0c556536_p*`), classify each
*progress-making* action into exactly one of three buckets relative to Carnot's live perception+candidate
generator (`rich_action_candidates(frame)`, `arc_graph_explore.py:117`): **(a) NOT in the candidate set at all**
(a perception/generation miss — the target object isn't segmented, or the click coordinate isn't among Carnot's
object-centroid candidates); **(b) in the set but not frame-changing in isolation** (a genuine multi-step /
lookahead signal — the action only pays off downstream, which Carnot's greedy-per-turn ranking cannot value);
**(c) in the set AND frame-changing in isolation but ranked low** by the candidate-router/value-head (a pure
SELECTION/ranking miss). The bucket histogram *localizes the gap* and dictates the next build — decisively, and
with no dependence on a noisy "did the tweak help" delta.

**Why this specifically will NOT repeat tonight's null.** Tonight's 7 experiments each MODIFIED a component and
measured `level_gain_delta`, a near-zero-headroom outcome that returns "no delta" whenever the corpus has no
selectable headroom (the FALSE_NEGATIVE_RISK failure mode `adversarial_verify.py` warns about). This experiment
measures a **structural attribution** (where a known-progress action sits relative to Carnot's candidate set),
which has no delta to be null on — it *cannot* come back "no change," only "the gap is mostly (a)/(b)/(c)."
It reuses the existing `arc_actions_to_progress.py` harness (REQ-ARC-WMTE-5720) for the offline replay + the
live `rich_action_candidates`/`PersistentAEM` for membership and in-isolation frame-change, so it is a pure
measurement over existing machinery, not a new scorer build.

**Falsifiable acceptance gate (decisive in all three directions).**
- Primary output: the (a)/(b)/(c) fraction over ≥N progress actions across ≥3 stalled games (pre-register N and
  the game list). Report per-game and pooled, with a bootstrap CI.
- **If fraction(a) > 0.5** → CONFIRMED: the gap is perception/candidate-generation. `GAP-ARCH-NO-HIERARCHICAL-
  SEARCH` is formally down-graded in `ops/verifier_gaps.md`; the next build targets segmentation fidelity +
  click-point generation beyond object centroids (Duck's translation-invariant object hash + containment/
  adjacency, `segmentation.py:65-103`). No search work.
- **If fraction(b) > 0.3** → the search/lookahead lever is RE-OPENED on evidence (in-set actions that only pay
  off downstream = a real planning gap the winners' greedy policies also lack but Carnot's `plan_in_model` could
  exploit *if* induction improves). This is the branch that would justify hierarchical search — and note it can
  only fire once, so the experiment is a *fair* test of the search hypothesis, not a foregone dismissal.
- **If fraction(c) > 0.3** → the gap is SELECTION/ranking; greenlight the candidate-router/value-head retrain
  or Duck's orientation-time re-rank loop over the *existing* candidate set (isolating orientation compute from
  generation).
- **RETIRE condition (Failed-Experiment Rerun Discipline, pre-registered):** if the measurement cannot reach N
  progress actions (too few stalled-game frames with a known progress action), fall back to the self-contained
  coverage test — "at each stall state, is there ANY frame-changing action in the candidate set the search never
  tried within budget?" — needing no winner trace; if THAT is also inconclusive, the honest verdict is
  "attribution not measurable offline; must be measured live," which itself retires the offline-attribution
  lineage rather than re-running it.

**Prior-failure block (rerun-discipline compliance).** Names the priors: the 7 nulls (REQ-ARC-FCP-5590/5728/
5729/5730/5732/5740/5756) + REQ-ARC-WMTE-5720/5724 (induce 0/12). Diagnosed root cause: all measured a
level-gain/induction-completion *delta* on a near-zero-headroom corpus, so a null cannot distinguish "component
useless" from "no headroom." What is different: this is not a delta A/B — it is a structural attribution that
partitions the gap and *produces a build decision in every branch*, including one branch that re-opens the
search hypothesis. Substrate: `verifier_ensemble_against_cached_candidates` (offline replay + candidate
membership; no LLM, no GPU); ~half a day.

**Live-path target (Live-Path Reachability Discipline).** Pure measurement over the live `rich_action_candidates`
+ `E3AgentPolicy` perception; whichever bucket dominates points at a *specific* live-path consumer (segmentation/
click-generation for (a), `plan_in_model`/induction for (b), `candidate_router`/value-head for (c)). No orphan
module — the deliverable is a decision about which existing live component to invest in next.

### 4a. Adversarial self-critique — the most likely ways THIS also fails (up front)

Per the Phase-Prototype + Adversarial-Check discipline, a hostile-reviewer pass on my own proposal:

1. **Public-game traces ≠ hidden-game gap (most likely limitation).** The winners' transcripts and Carnot's
   stalled replays are PUBLIC games; the scored deficit is on HIDDEN games, and by the benchmark's design
   perception/affordance conventions don't transfer. An attribution measured on public games may misestimate
   the hidden-game bucket mix. Mitigation: the primary corpus is Carnot's OWN stalled games (self-contained,
   no winner trace needed); the winner traces are corroborating-only. This is the same
   development-proxy-for-a-live-quantity caveat the whole ARC program lives with.
2. **The (a)/(b)/(c) partition can be gamed by the "no-op that pays off later" ambiguity.** An in-set action
   that is no-op-in-isolation could be either genuine lookahead signal (b) OR just a bad candidate that happens
   to be in the set. I resolve this by requiring bucket (b) actions to lie on a *known* progress path (the
   action is a prefix of a trajectory that DID reach a level-up), not merely "in-set and no-op." If too few such
   verified paths exist, (b) is under-powered and the search hypothesis simply cannot be confirmed by this
   experiment — which I state honestly rather than inflating (b).
3. **Duck's example-run is thin (one game, `ar25`).** If the only winner trace is a single game it cannot
   support a general claim; the mitigation is to lean on Carnot's stalled-game corpus for statistical power and
   treat the Duck trace as a single qualitative cross-check, not a data source for the gate.
4. **"In candidate set" is coordinate-exact and may under-count near-misses.** A winning click one pixel off a
   Carnot centroid would score as (a) when it is really a near-hit. Mitigation: score membership with a small
   tolerance radius (reuse the object-bbox test from `_click_action_score`-style logic) and report both exact
   and tolerant membership, so a near-miss surfaces as "generation resolution" rather than "generation absence."

**Net honest read.** The proposal's value is that it converts an unfalsifiable "are we tweaking the wrong layer?"
worry into a measured bucket histogram that dictates the next build in every branch — and, uniquely among the
options, it gives the search/lookahead hypothesis one fair, pre-registered chance to be RIGHT (bucket b) rather
than assuming the §3 finding. Its biggest risk is not a mirage (there is no delta to fake) but *insufficient
power* on the (b) bucket, which I have pre-committed to report as "search hypothesis untestable here," not as
evidence against it.

---

## 5. Cross-references

- **Winner source (read-only, `external/`):** Duck `ARC3-Inference/inference/framework/solver.py`,
  `inference/agent/tool_agent.py`, `inference/utils/segmentation.py`, `example-run/{transcripts,prompts}/`;
  Reki `arc-m1-2nd-reki/milestone1-2nd-solution.ipynb`; forge
  `arc-m1-3rd-forge/arc-agi-3-lb-0-86-3rd-place-candidate-milestone.ipynb`.
- **Carnot search/planning (live + offline):** `python/carnot/agentic/arc_competition_agent.py`
  (`StepwiseExplorer:678`, `E3AgentPolicy:2701`, phase machine `2702-2712`, `plan_in_model` call `3361/3806`,
  `_induce_and_plan:3734`); `arc_graph_explore.py` (`rich_action_candidates:117`, `graph_explore_solve_v2:351`,
  `misplaced_region_distance:947`); `arc_vc33_hierarchical_search.py` (subgoal decomposition `206`,
  `hierarchical_best_first_search:281`, per-game preconditions `378`); `arc_actions_to_progress.py`
  (REQ-ARC-WMTE-5720, the proposed metric harness).
- **The gap this re-orders:** `ops/verifier_gaps.md :: GAP-ARCH-NO-HIERARCHICAL-SEARCH` (line 2544).
- **Prior notes:** `docs/research-notes/arc-agi3-milestone1-winners-sota-ingestion-2026-07-11.md` (perception/
  scoring opportunities O1-O5, the gap this note fills); `arc-action-effect-representation-redesign-2026-07-19.md`
  (rigor template); `arc-agi3-leaderboard-technique-watch.md` (2026-07-17 Duck/TAAF object-representation entry).
- **The seven nulls that motivated this:** REQ-ARC-FCP-5590/5728/5729/5730/5732/5740/5756;
  `openspec/capabilities/arc-human-replay-frame-change/spec.md`.
- **Corroborating memory:** `project_arc_live_agent_learning_gaps` (perception is the binding constraint),
  `project_arc_actions_to_progress_metric` (dynamics-induction bottleneck; /think + retrieval null on progress),
  `feedback_arc_value_is_process_not_weights`, `reference_zendoworld_hypothesis_uncertainty`
  (generation-not-selection corroboration).

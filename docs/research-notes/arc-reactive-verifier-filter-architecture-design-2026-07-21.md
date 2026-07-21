# ARC reactive-verifier-as-filter architecture — scoping design (2026-07-21)

**Status:** SCOPING / DESIGN ONLY. No code written, no live-path file modified, no experiment script created.
This note proposes ONE concrete, falsifiable, adversarially-self-critiqued small-scale prototype that tests
whether Carnot's ARC-AGI-3 live agent should shift its centre of gravity from "induce an explicit symbolic
world model up front, then search it" toward a REACTIVE turn-by-turn loop in which Carnot's verifier machinery
acts as a FILTER on a proposer's choices, keeping genuine verification in the loop (the falsification role the
literature validates) rather than becoming a Duck-Harness clone that abandons verification entirely.

**Reads as required input (per CLAUDE.md):** "ARC-AGI-3 IS a Live Hidden-Game Discovery Agent" (foundational
framing — the deliverable is a live agent that DISCOVERS unseen games at runtime; an offline null may be a
corpus artifact), "ARC Live-Path Reachability Discipline" (every mechanism must be reachable from the two live
entrypoints; source-reading is public-dev-only, never in the hidden submission), "Phase Prototype + Empirical
Validation + Adversarial Check" (THIS note's governing discipline — a runnable small-scale prototype + explicit
falsifiable pass/fail criteria + a hostile-reviewer pass BEFORE any scaling decision), "Missing-Verifier Gap
Logging", "Literature Priority Discipline" (two-source rule), "Decentralization-Respecting Design Constraints".

**Immediate prerequisites (read in full before this note):**
`docs/research-notes/arc-world-model-induction-quality-diagnosis-2026-07-20.md` (the induction-quality wall),
`docs/research-notes/arc-induction-quality-improvement-design-2026-07-20.md` (the CEGIS-refinement proposal
whose null this note responds to, esp. its §2 architectural question), and
`docs/research-notes/arc-top-project-search-architecture-audit-2026-07-20.md` (the winner-vs-Carnot audit that
found Carnot already has MORE search than any winner). This note builds on those; it does not re-derive them.

---

## 1. The problem, restated precisely (citing the real evidentiary chain — not re-derived)

A ~24-hour investigation (2026-07-19/20) converged decisively on one diagnosis: Carnot's ARC-AGI-3 live agent's
binding constraint is world-model INDUCTION quality, not search depth, not candidate generation/selection, not
click-ranking. The chain, with the freshest evidence:

1. **Not search depth.** The three open-sourced Milestone-1 winners (Duck/1st, Reki/2nd, forge/3rd) are all
   greedy single-commit-per-turn generators with NO model-based lookahead, NO tree/beam/MCTS, NO subgoal
   decomposition; Carnot already has strictly MORE search machinery than any of them (a systematic
   state-transition graph search + a model-based `plan_in_model` lookahead + a working vc33 hierarchical
   prototype) — `docs/research-notes/arc-top-project-search-architecture-audit-2026-07-20.md` §1–§3. Consequence:
   `GAP-ARCH-NO-HIERARCHICAL-SEARCH` was down-weighted on the winners' own evidence.

2. **Not generation/selection.** `results/experiment_5757_candidate_coverage_attribution.json`
   (`honest_verdict: complete_attribution_gate_no_threshold_crossed_dominant_gap_c_n92_games9`) partitioned the
   single-action gap and found candidates are generated + ranked correctly ~99% of the time; the only residual
   is a narrow SELECTION slice (bucket c: 6 object-clicks that ARE generated + frame-changing but rank low).

3. **Not click-ranking.** That residual was A/B-tested three ways and is a clean null —
   `ops/verifier_gaps.md :: GAP-ARC-CLICK-SELECTION-5758`: the winning click is a 1-pixel object outranked by
   many EQUAL-size, rarer-coloured pixels; NO static-perceptual (area/rarity/size-band) reorder surfaces it.
   The missing discriminator is a GOAL-CONDITIONED signal keyed on the object's ROLE in reaching the goal
   (predicted level-progress), i.e. what the action DOES — which requires the induced action-effect/goal model.

4. **It IS induction quality.** `docs/research-notes/arc-world-model-induction-quality-diagnosis-2026-07-20.md`:
   over 37 successful inductions (ThinkingCap-27B + Qwen-9B, 17 games), `heldout_accuracy` (exact full-grid
   transition match) is 0.0 on 29/37, and `reached_levelup = 0/80`. The induced code is near-universally wrong
   about the game's real action→effect DYNAMICS (identity / single-hardcoded-click / confabulated /
   window-memorized engines); the rare near-perfect induction happens only when the mechanic matches a template
   the LLM already knows (2048 = `sp80`, heldout 1.0).

5. **The obvious fix — verifier-grounded CEGIS refinement — is a clean null on TWO models.**
   `results/experiment_5760_cegis_refinement_induction_ab.json` (ThinkingCap-27B + Qwen-9B, pooled
   `Δheldout = -0.0128`, `positive_game_frac = 0.0`) and
   `results/experiment_5766_gemma31b_cegis_refinement_ab.json` (gemma-4-31B-it, pooled `Δheldout = -0.0598`,
   `positive_game_frac = 0.0`) — **literally zero games improved from refinement on either model**
   (`ops/verifier_gaps.md :: GAP-ARC-INDUCTION-REFINEMENT-NULL`). Model CAPACITY does help single-shot induction
   (`results/experiment_5764_gemma31b_singleshot_induction_ab.json`: pooled heldout **0.378** vs
   ThinkingCap-27B's **0.188**, 12/13 games off the floor), but applying MORE COMPUTE VIA SELF-REPAIR does not.
   This corroborates arXiv:2606.31511 ("Falsification, Not Exposure") on this exact task: for frozen local
   models, self-repair FEEDBACK CONTENT does not improve correctness; only comparison against EXTERNAL
   EXECUTABLE ground truth does — **the falsification/filter role, not the correction role.**

The GAP-ARC-INDUCTION-REFINEMENT-NULL entry names this note's exact deliverable as its candidate design (b):
"a reactive-with-verifier-as-filter prototype on 2-3 games, scoped adversarially per the Phase Prototype +
Adversarial Check discipline before any broader investment, since this is a real architecture-level bet, not a
parameter tweak." This note is that scoping.

---

## 2. What "verifier as filter" concretely means for Carnot (the mechanism, one chosen design)

**The chosen design in one sentence.** At each live turn, a proposer emits a small ordered set of candidate
next actions from the current frame (goal-directed, no up-front symbolic world model), and a FILTER *executes*
each candidate one step against a re-executable environment probe and keeps only those whose REAL observed
outcome is non-inert and does not regress the goal-energy / does not die — committing the best surviving
candidate; the induce→plan tier is turned OFF.

**Why this is genuine falsification-grounded filtering, not any of the three failure classes the task warned
against:**

- **Not disguised planning.** The filter does exactly one real forward step per candidate against the env's own
  transition function; it does not search a multi-step tree, does not induce or trust a symbolic model, and
  does not roll a bounded `plan_in_model` lookahead. The "model" it checks against is the REAL environment, one
  step deep. (Carnot already owns the exact machinery: `probe_action_semantics`,
  `python/carnot/agentic/arc_solver_kit.py:5525`, resets a fresh env via `env_factory()`, replays the prefix,
  applies each candidate label, and measures real `changed_cells` / `leveled_up` / `died` / `inert`
  (`inert = changed_cells == 0 and after_level == before_level`). It is not wired as a per-turn pre-commit
  filter — that wiring is the prototype's one new seam.)

- **Not an LLM self-assessing itself.** The keep/reject decision is a deterministic comparison of the candidate's
  real post-action grid against its pre-action grid (a pixel diff and a level/energy read), never the proposer's
  own opinion of its choice. This is the "energy function is ground truth, cannot be gamed" invariant applied at
  the action grain: the proposer PROPOSES, the executable check DECIDES. It matches the E3 module's own stated
  thesis verbatim — "LLMs are most reliable when used not as final authorities, but as PROPOSAL mechanisms
  inside systems that can check their outputs" (E3 docstring, `arc_competition_agent.py:2707`-region).

- **Not abandoning verification like Duck.** Duck has zero verification: `board_changed` is computed
  (`external/duck-harness/ARC3-Inference/inference/framework/solver.py:703`) but is purely advisory — no control
  flow keys off it (the one method that would verbalize it between turns, `_describe_last_outcome`
  `tool_agent.py:1070`, is defined-but-never-called dead code), and the LLM's judgment is the sole final
  authority on every commit (its only pre-commit gate is an `available_actions` LEGALITY check,
  `tufa-arc-agi-framework/src/taaf/game.py:511-524`). Carnot's design keeps a real, non-LLM,
  executable-ground-truth verifier between the proposal and the commit — that is the whole point and the
  thesis-preserving difference from a Duck clone.

**How it is DIFFERENT from what has already been tested and found null:**

- **vs. induce→verify→plan (thrice-falsified, REQ-ARC-WMTE-5760/5766 + the induction diagnosis):** no symbolic
  world model is induced or refined at all. The failed approach spent GPU-days trying to make a frozen model
  WRITE a correct `engine(grid, action)` program; this design never asks for a program — it asks for an action
  and checks the real one-step consequence. The literature counterweight (arXiv:2606.31511) says the
  falsification/filter role is the only one that carries signal for frozen models; this design uses ONLY that
  role and discards the correction role entirely.

- **vs. the REQ-ARC-FCP-5740/5756 frontier-bonus / object-affordance nulls:** those REWEIGHTED candidates by a
  learned/observed-salience PRIOR (a soft additive bonus for objects observed to change) and measured a
  level-gain delta on a near-zero-headroom corpus. This design does not reweight — it EXECUTES each candidate
  and applies a HARD keep/reject on the real outcome before commit. Reweighting a candidate that is never
  frame-changing cannot help; executing it and dropping it if inert is a categorically different operation.

- **vs. the REQ-ARC-FCP-5758 static-perceptual click reorders:** GAP-ARC-CLICK-SELECTION-5758 established that
  the winning click is distinguished by what it DOES (predicted goal-progress), not its appearance, and no
  static reorder surfaces it. Executing the candidate one step is the cheapest possible way to observe what it
  DOES without an induced model — it substitutes a single real forward step for the (failed) induced-model
  prediction of that step.

- **vs. `PersistentAEM` / `ObjectHistorySaliencePrior` (REQ-ARC-FCP-5740/5756):** those learn a PRIOR from the
  agent's observed history and score a candidate BEFORE executing it — `PersistentAEM` is even frozen/offline
  (`arc_online_action_effect_scorer.py:25`). The new element is executing the specific proposed candidate from
  the CURRENT exact state and filtering on THAT real outcome, rather than generalizing from a signature-keyed
  prior that may not fire for a novel state.

---

## 3. What is already built vs. genuinely new (with real file:line — an honest verdict)

**Headline verdict: this is far closer to "promote/re-wire existing components" than "build a new architecture
from scratch." Three of the four ingredients already exist and are live; only one seam is genuinely new.**

### 3a. The reactive per-turn loop ALREADY EXISTS and is the DEFAULT primary mode

`E3AgentPolicy` initializes `self.phase = "explore"` (`arc_competition_agent.py:2890`). In the EXPLORE phase,
`choose_action` calls `self.explorer.next_move(frames, latest)` every turn (`arc_competition_agent.py:3690`,
`:3712`, `:3725`, `:3732`) — a genuine reactive event loop. `StepwiseExplorer.next_move`
(`arc_competition_agent.py:2388`) ingests the latest frame, then commits ONE action per turn: it pops an
untested SALIENT action from the current node (depth-first ride, `:2415-2433`) or navigates to the best frontier
via RESET-replay (`:2436-2457`). The induce→verify→plan→execute machine is the ESCALATION path, entered only
when `_should_enter_induction` fires on a stall or a level-up (`arc_competition_agent.py:3680`, `:3706`;
predicate at `:3160`). **So the "reactive vs. induce-and-plan" reframing is not aspirational — the reactive loop
is already primary, and the now-thrice-falsified induction is already the secondary escalation.** The real
design change is (i) what the reactive loop's PROPOSER is (generic-salience exploration vs. goal-directed), (ii)
what its FILTER is (learned prior vs. executable ground truth), and (iii) whether the induction escalation is
demoted to opportunistic/off.

### 3b. The OBSERVE→FILTER plumbing and reactive falsifiers already exist (some ON, some OFF by default)

- **The OBSERVE fan-out is live:** every realized `(before, action, after)` triple is handed to any component
  exposing `observe_transition`/`observe` (`arc_competition_agent.py:1732-1790`).
- **The FILTER seam is live:** `rich_action_candidates` (`arc_graph_explore.py:164`) applies `prune_arc_actions`
  + `rank_arc_actions`, and any component with a `rank_candidates(frame, rows) -> rows` method can drop/reorder
  proposals per turn.
- **An online-learned falsifier is ON by default:** `StepwiseExplorer(online_discriminative=True)`
  (`arc_competition_agent.py:699`) fits a discriminative pruner from the agent's OWN observed transitions and
  prunes frontier actions predicted inert — a reactive falsifier already in the primary loop.
- **Two stronger reactive falsifiers exist but are OFF by default:** `InertClickSigPruner`
  (`arc_inert_click_pruner.py:136`; `rank_candidates` at `:250`; `click_signature` at `:117`) prunes click
  signatures observed inert ≥0.9 over ≥4 obs (level-up signatures SACRED), and `ObjectHistorySaliencePrior`
  (`arc_object_history_salience.py:66`) boosts objects observed to change online. Both learn only from real
  observed frame changes, both declare `verifier_is_oracle=False`, both gated OFF
  (`SUBMITTED_INERT_CLICK_PRUNER_ENABLED = False`, `arc_competition_agent.py:184`;
  `SUBMITTED_OBJECT_HISTORY_SALIENCE_ENABLED = False`, `:171`).
- **The frame-change scorer is ON but spends its ground truth on validating a PRIOR, not filtering the
  proposal:** `GroundTruthValidatedFrameChangeScorer` (`arc_frame_change_predictor.py:280`) computes the real
  pixel-change fraction per transition but uses it only to gate a learned base scorer's calibration; its
  `candidate_score` returns the LEARNED value, never a ground-truth verdict on the proposed action.

### 3c. The one genuinely NEW element

**Nothing today checks a *proposed, not-yet-executed* action against *real executable ground truth* on the
current turn as a hard pre-commit filter.** The closest primitives each fall short: the online discriminative
pruner and `InertClickSigPruner` learn a signature-keyed prior and filter BEFORE executing (they generalize, so
they can be wrong on a novel state); `LiveTTTWorldModel`'s L0 exact table (`arc_live_ttt.py:350`) is true
own-history ground truth but only fires on an EXACT full-grid repeat; and `probe_action_semantics`
(`arc_solver_kit.py:5525`) IS real per-candidate executable ground truth but is not wired as a per-turn
pre-commit filter. **The new seam is: wire `probe_action_semantics`-style single-step execution as the reactive
loop's pre-commit FILTER, and (for the prototype) replace the generic-salience proposer with a goal-directed
one.** Everything else — the OBSERVE fan-out, the FILTER seam, the executable-probe machinery, the
actions-to-progress metric harness — is reused.

**Honest verdict:** a "promote the reactive loop to primary + turn on the existing observed-signature falsifiers
+ add one executable-ground-truth pre-commit filter seam + demote induction," NOT a from-scratch architecture.
This materially lowers the build risk and is the single most important finding of this scoping pass.

---

## 4. Real precedent from the literature (two-source rule, interactive/embodied — not code-gen self-repair)

The induction-quality design already grounded the correction-vs-filter distinction in the code-generation
self-repair literature (arXiv:2606.31511, "Falsification, Not Exposure"). For the EXACT reactive-turn-filtering
setting — a proposer emitting candidate actions and an external check filtering them before commitment in an
interactive/embodied environment — the closest-fit precedents are:

- **SayCan — "Do As I Can, Not As I Say: Grounding Language in Robotic Affordances," Ahn et al., 2022**
  ([arXiv:2204.01691](https://arxiv.org/abs/2204.01691)). The canonical "LLM-proposes × external-grounding-
  filters" architecture: the LLM (Say) scores how useful each skill is toward the goal, and an affordance value
  function (Can) grounds feasibility from the current state; the product decides the committed skill. This is
  precisely the reactive-proposer + grounding-filter shape. **The distinction Carnot draws:** SayCan's grounding
  is a LEARNED value function (a prior, in principle gameable); Carnot's thesis-consistent filter uses REAL
  executable ground truth (execute the candidate, observe the actual outcome) — a stronger, non-gameable
  falsification, which is exactly the direction arXiv:2606.31511 says carries signal for frozen models.

- **Grounded Decoding — "Guiding Text Generation with Grounded Models for Embodied Agents," Huang et al.,
  NeurIPS 2023** ([arXiv:2303.00855](https://arxiv.org/abs/2303.00855)). Describes itself as "probabilistic
  filtering": decode an action sequence that is both likely under the LM AND "realizable according to grounded
  models of the environment." A second, independent interactive/embodied precedent for the proposer-×-filter
  architecture. **Same Carnot distinction:** Grounded Decoding's grounding is a learned probability; Carnot's is
  a real one-step execution.

- **Duck Harness (the actual ARC-AGI-3 winner) — the anti-precedent that defines what NOT to copy.** Duck has NO
  grounding/affordance/verification filter at all: `board_changed` is advisory-only, `_describe_last_outcome` is
  dead code, and the LLM is the sole final authority (§2 above; agent re-read of
  `external/duck-harness/ARC3-Inference/inference/{framework/solver.py,agent/tool_agent.py}`). Duck wins with a
  27B model + orientation-time REPL compute + object-centric perception, none of which Carnot's Kaggle-16GB /
  frozen-9B live budget can match. Carnot cannot and should not become Duck; the design's value is keeping the
  SayCan/Grounded-Decoding grounding filter that Duck lacks, upgraded to executable ground truth.

**The precise precedent-grounded claim this design makes:** SayCan and Grounded Decoding both validate that a
capable proposer + an external grounding FILTER beats an unfiltered policy in interactive settings — but both
use a LEARNED grounding model. No prior work (that this search surfaced) uses *real one-step execution against
the environment itself* as the per-turn filter in this setting, because most embodied settings have no cheap
re-executable env. ARC-AGI-3's offline dev twin DOES (RESET-replay). That is the specific, thesis-consistent
gap this prototype occupies — and, honestly, also its central live-transfer risk (§6, since the live Kaggle env
is frames-only and not per-candidate re-executable).

---

## 5. The SMALL-SCALE prototype spec (runnable end-to-end, 2-3 games, first-pass adversarial check ONLY)

**Explicitly scoped as a first-pass adversarial check per the Phase-Prototype discipline — NOT a headline
measurement.** It is a development-proxy on PUBLIC games (source-reading allowed for offline dev per the ARC
Live-Path Reachability Discipline; the filter uses the env's transition function, not its source).

**Title (working):** Reactive-proposer + executable-ground-truth filter vs. induce-and-plan — a 3-game offline
first-pass.

**Games (pre-registered, N=3).** `sp80` (the one game where induction ALREADY succeeds — heldout 1.0 — so the
prototype must not REGRESS a game the induce-and-plan path already handles), plus `r11l` and `su15` (the two
games carrying Carnot's only measured residual single-action gap, GAP-ARC-CLICK-SELECTION-5758 — goal-directed
proposal + executable filter is the exact lever that gap's "missing discriminator" calls for). This trio is a
fair test: one induction-favourable game (regression guard) and two reactive-favourable games (signal test).

**Components (all reuse existing machinery; one new wiring seam):**
- **Proposer:** the OFFLINE capable model **gemma-4-31B-it** (single-shot induction showed capacity genuinely
  helps, REQ-ARC-WMTE-5764), run on the conductor's dedicated GPU-0 3090 (permitted for OFFLINE induction/
  proposal per the 2026-06-27 GPU-allocation directive), prompted per-turn for a small ordered candidate set
  (k≤4) of goal-directed next actions from the current frame + a running NL goal-hypothesis — NOT to write an
  `engine()` program. A cheap non-LLM control arm uses the existing `rich_action_candidates`
  (`arc_graph_explore.py:164`) proposer for the same filter (isolates proposer capability from the filter).
- **Filter (the new seam):** for each proposed candidate, run one `probe_action_semantics`-style step
  (`arc_solver_kit.py:5525`) against the re-executable offline env (RESET + replay-prefix + apply candidate);
  KEEP iff the real outcome is non-inert (`changed_cells > 0`) AND not `died` AND does not regress the dense
  goal-energy (reuse the `hv_progress` goal-distance proxy, `arc_actions_to_progress.py:441`); commit the best
  surviving candidate by goal-energy. This is `verifier_is_oracle=False` (the filter checks real dynamics, not
  the win-oracle).
- **Induction:** OFF (the falsified escalation is disabled for the reactive arm).
- **Harness:** extend the existing `run_seeded_progress` actions-to-progress harness
  (`arc_actions_to_progress.py`, REQ-ARC-WMTE-5720) — the same one used all week — to drive this loop and record
  the metrics below. Substrate: `live_llm_inference` for the gemma-31B proposer arm (real GPU generation, 60s
  floor, PRECONDITIONS block required per CLAUDE.md's 2026-07-06 CUDA-build rule:
  `llama_cpp.llama_supports_gpu_offload()` must be True or induction silently runs on CPU);
  `offline_arcade_live_agent_runtime_self_discovery_no_llm` for the cheap `rich_action_candidates` control arm.

**Three arms (pre-registered):**
1. **REACTIVE-FILTER:** gemma-31B goal-directed proposer + executable-ground-truth filter, induction OFF.
2. **NO-FILTER ABLATION (mandatory — the anti-rubber-stamp control):** same gemma-31B proposer, commit its top
   pick directly, NO executable filter. If arm 1 ≈ arm 2, the filter adds nothing (it is rubber-stamping) — this
   arm is what makes the "is the filter real?" question falsifiable.
3. **BASELINE:** the current shipped induce-and-plan primary path (reactive EXPLORE + induction escalation ON),
   same games/budget — the incumbent the reactive architecture must beat or match.

---

## 6. Empirical validation criteria (explicit, falsifiable pass/fail — honest about the proxy)

**Primary metric is a PROXY, stated honestly.** At N=3 games real level-ups will very likely be 0 (tonight's
whole 80-cell run banked 0/80; `reached_levelup` is too rare to gate on at this scale). The primary metric is
therefore the reusable live-path **actions-to-progress / dense goal-energy progress** signal
(`hv_progress`, `arc_actions_to_progress.py:441`; the REQ-ARC-WMTE-5720/5721 metric,
`project_arc_actions_to_progress_metric`), with real level-ups reported as a secondary bonus IF any occur.

Pre-register per-game and pooled (bootstrap 95% CI over the seeds within each game):

- **PROMISE (the reactive-filter architecture warrants a larger test):** ALL of —
  - arm 1 (REACTIVE-FILTER) pooled `hv_progress` **> arm 3 (BASELINE)** by a margin whose bootstrap CI excludes
    0, on at least the two reactive-favourable games (`r11l`, `su15`); AND
  - arm 1 does NOT regress `sp80` below arm 3 (the induction-favourable regression guard); AND
  - **filter selectivity is real:** arm 1's committed action differs from arm 2's (NO-FILTER) top pick on a
    material fraction (pre-register ≥20%) of turns AND the filter's reject rate is in (0, 1) — i.e. it neither
    rubber-stamps everything nor rejects everything.
- **DEAD END (the reactive-filter architecture is also a null):** arm 1 ≈ arm 3 pooled (CI includes 0) on the
  reactive-favourable games, AND/OR arm 1 ≈ arm 2 (the filter changes nothing / rubber-stamps). Log a NEW
  `ops/verifier_gaps.md` entry, do NOT scale, and the architectural conclusion becomes "neither induce-and-plan
  NOR reactive-filter closes the gap at this model class — the lever is model CAPACITY (extend the
  REQ-ARC-WMTE-5764 capacity trend past 31B, operator-only) or a genuinely different representation."
- **INCONCLUSIVE (needs more games):** mixed per-game signs with wide CIs, or `hv_progress` too flat to
  discriminate on all three games. Verdict: "3 games under-powered; the honest next step is a pre-registered
  larger offline roster BEFORE any live-transfer work," not a scale-up on noise.

**Attribution guards (pre-registered):** report the gemma-31B per-turn proposer emission/parse rate (a proposer
that emits no parseable action makes arm 1≈arm 3 a mechanical artifact, not evidence); report the executable
filter's mean probes-per-turn and total probe count (the cost that governs live feasibility, §7); report
whether any real level-up occurred in any arm.

---

## 7. Adversarial check — the ways this prototype could pass its own gate without actually working (BEFORE scaling)

Per the Phase-Prototype + Adversarial-Check discipline, a hostile-reviewer pass on my own proposal, ordered by
how likely and how damaging each is.

1. **The filter rubber-stamps everything → indistinguishable from no filter (highest-priority risk).** On busy
   games most actions produce SOME frame-change, so "keep if non-inert" could keep every candidate, collapsing
   arm 1 to arm 2 (commit the proposer's top pick). Then a "win" would be the PROPOSER's win, falsely credited
   to the filter — and worse, it would be a Duck-style unverified policy wearing a verifier costume. **Mitigation
   (built into the gate):** arm 2 (NO-FILTER) is mandatory, and PROMISE requires the filter to change the
   committed action on ≥20% of turns with a reject rate strictly in (0,1). If the filter's only effect is to
   drop the rare fully-inert click, that is a real-but-tiny contribution and must be reported as such, not
   inflated. The goal-energy-non-regression clause (not merely "changed") is what gives the filter teeth beyond
   inert-dropping — but if it too rarely fires, the honest verdict is "filter ≈ inert-pruner already ON," i.e.
   no new architecture, just turn on `InertClickSigPruner`.

2. **Live-transfer infeasibility even if the OFFLINE prototype works (the most consequential limitation — flag
   loudly).** BOTH of the prototype's real components are OFFLINE-only:
   - The **proposer** is a 31B model called per-turn — feasible on the conductor's GPU-0 3090 offline, but the
     LIVE Kaggle submission is iGPU/~16GB and frozen at Qwen3.5-9B-MTP (`project_arc_live_generator`); a 31B
     per-turn proposer is structurally forbidden live, and even the 9B at ~4 tok/s per turn over hundreds of
     actions may blow the live wall-clock.
   - The **filter** needs a per-candidate re-executable env (RESET + replay-prefix). Offline `environment_files`
     supports this; the LIVE Kaggle env is a frames-only remote gateway (`OPERATION_MODE=online` against
     `gateway:8001`, per the ARC Live-Path Reachability Discipline note) with NO branching/rollback — so
     per-candidate pre-commit execution live would cost RESET + full-prefix-replay + one probe PER candidate,
     which the action budget (RHAE squares efficiency) makes prohibitive.
   **Consequence:** a POSITIVE offline result is a `development_proxy`, NOT proof the live agent self-discovers
   better. It would test the ARCHITECTURE's ceiling with expensive-but-real components; a live instantiation
   MUST degrade to a cheap proposer (the 9B or `rich_action_candidates`) and a cheap filter (the already-live
   observed-signature falsifiers — `InertClickSigPruner` / online discriminative pruner — which approximate the
   executable check WITHOUT re-execution). **Whether the cheap live version preserves the offline signal is a
   SEPARATE, later question that this prototype does NOT answer.** I state this up front rather than letting a
   green offline result imply a live win.

3. **3 games cherry-picked / lucky.** The trio is chosen (not random): `sp80` favours induction, `r11l`/`su15`
   favour the reactive lever. That is deliberate (a fair one-guard-two-signal split), but it means a PROMISE on
   `r11l`/`su15` is evidence for THOSE mechanic classes, not a general claim. **Mitigation:** the gate's scale-up
   is to a pre-registered LARGER offline roster, never a direct live push; and the INCONCLUSIVE branch exists
   precisely so 3-game noise cannot be read as a win. No headline claim is permitted from N=3.

4. **The proxy metric (`hv_progress`) moves without real progress.** `hv_progress` is a dense goal-DISTANCE
   proxy; a policy could reduce it by thrashing near the goal without ever leveling up (the same class as the
   degenerate goal-predicate `plan_len=1` non-wins in the induction diagnosis §6). **Mitigation:** report real
   level-ups alongside, treat `hv_progress` as necessary-not-sufficient, and require the PROMISE margin on the
   proxy to be corroborated by at least a non-worse real-level-up count (arm 1 must not level-up FEWER times
   than arm 3). If the proxy improves but real level-ups regress, the verdict is DEAD END, not PROMISE.

5. **gemma-31B proposer parse/emission failure masquerades as a filter/architecture null.** Tonight 34/40 Qwen
   inductions overran the token budget; a per-turn action proposal is shorter than an `engine()` program (lower
   risk) but non-zero. **Mitigation:** the pre-registered emission/parse-rate attribution guard (§6) quarantines
   this as a mechanical confound, not evidence against the architecture.

**Net honest read.** The prototype's biggest genuine risks are NOT that it fakes a delta (arm 2 makes the filter
falsifiable and the metric has real headroom) — they are (1) the filter contributing nothing beyond the
already-ON inert-pruner, and (2) that even a clean offline win does not transfer to the live budget. Both are
pre-registered as explicit branches with concrete consequences, so the prototype produces an actionable decision
in every outcome — including the one where "reactive-filter" turns out to be "turn on `InertClickSigPruner` and
stop inducing," which would itself be a valuable, cheap, live-legal finding.

---

## 8. Reconciliation with Carnot's core thesis (honest — one clean fit, one real tension)

**Clean fit — the FILTER role strengthens the thesis.** "The energy function is ground truth, cannot be gamed"
(CLAUDE.md Operational Principles) and "LLMs are most reliable as PROPOSAL mechanisms inside systems that can
check their outputs" (E3 docstring / `project_core_motivation`) are BOTH more directly honoured by an executable-
ground-truth filter than by the induce-and-plan path. Induce-and-plan grounds a possibly-wrong SYMBOLIC model
(the verifier certifies a model that then plans wrongly — the 0/80 level-ups); the reactive filter checks the
REAL one-step consequence, which cannot be gamed by a confabulated engine. This design keeps genuine verification
in the loop (the whole reason it is Carnot's approach and not a Duck clone) and moves it to the grain — per-turn
action — where the literature (arXiv:2606.31511) says frozen-model feedback actually carries signal.

**Real tension — DECENTRALIZATION (flag to the operator up front).** Decentralization-Respecting Design
Constraint rule 1 is "local-first using open models, always," and the live submission is deliberately frozen at
a SMALL open model (Qwen3.5-9B) precisely so a sovereignty-respecting user can run it on hardware they own. The
prototype's capable-model per-turn proposer (gemma-31B) is OFFLINE-only and does not itself violate rule 1 (it
runs on the conductor's own GPU, no closed vendor) — but if a POSITIVE result tempted a shift toward a
capable-model-in-the-action-loop LIVE architecture, that would strain the small-model live constraint and
mirrors the exact tension the induction-quality design already flagged (its §2 point 2: every winner puts a
BIGGER generator in the action loop, which Carnot's constraint forbids). **The design contains this tension by
construction:** the prototype tests the architecture's ceiling offline, and the pre-registered live-transfer
path (§7 risk 2) explicitly degrades to the frozen 9B + already-live cheap falsifiers — it does NOT propose a
big-model live loop. The operator should know that a strong offline result would create pressure toward a
larger live proposer, and that the decentralization-preserving answer is the cheap-component live version, whose
signal-preservation is a separate open question. There is no tension with the verifier/energy thesis itself —
only with the small-model live-deployment constraint, and only if the offline ceiling result is mis-read as a
live mandate.

---

## 9. Cross-references

- **The evidentiary chain this builds on:**
  `docs/research-notes/arc-world-model-induction-quality-diagnosis-2026-07-20.md` (induction wall, 29/37
  heldout=0.0); `docs/research-notes/arc-induction-quality-improvement-design-2026-07-20.md` (CEGIS-refinement
  proposal + its §2 architectural question this note answers);
  `docs/research-notes/arc-top-project-search-architecture-audit-2026-07-20.md` (winners are greedy, Carnot has
  more search, bottleneck = induction/generation).
- **The four cited result artifacts:** `results/experiment_5757_candidate_coverage_attribution.json` (gap is
  induction/selection, not generation); `results/experiment_5760_cegis_refinement_induction_ab.json` +
  `results/experiment_5766_gemma31b_cegis_refinement_ab.json` (CEGIS null on both models);
  `results/experiment_5764_gemma31b_singleshot_induction_ab.json` (capacity helps single-shot: 0.378 vs 0.188).
- **The GAPs:** `ops/verifier_gaps.md :: GAP-ARC-INDUCTION-REFINEMENT-NULL` (names this note's deliverable as
  candidate design (b)); `:: GAP-ARC-CLICK-SELECTION-5758` (the goal-conditioned discriminator this design
  supplies via one-step execution); `:: GAP-WM-TRUST-GATE`; `:: GAP-ARCH-NO-HIERARCHICAL-SEARCH` (down-weighted).
- **Live-path code (reactive loop + falsifiers + filter machinery):** `arc_competition_agent.py`
  (`E3AgentPolicy:2701`, `self.phase="explore":2890`, phase dispatch `:3679-3732`, `_should_enter_induction:3160`,
  OBSERVE fan-out `:1732-1790`, live CEGIS call sites `:3885/:4005`, flags `:105/:171/:184`);
  `StepwiseExplorer.next_move` (`arc_competition_agent.py:2388`, `online_discriminative` default `:699`);
  `arc_graph_explore.py` (`rich_action_candidates:164`); `arc_inert_click_pruner.py`
  (`InertClickSigPruner:136`, `rank_candidates:250`, `click_signature:117`);
  `arc_object_history_salience.py` (`ObjectHistorySaliencePrior:66`); `arc_frame_change_predictor.py`
  (`GroundTruthValidatedFrameChangeScorer:280`); `arc_live_ttt.py` (L0 exact table `:350`);
  `arc_solver_kit.py` (`probe_action_semantics:5525` — the executable-ground-truth filter machinery);
  `arc_actions_to_progress.py` (`run_seeded_progress`, `hv_progress:441`, REQ-ARC-WMTE-5720/5721 metric).
- **Duck anti-precedent (read-only, `external/`):** `duck-harness/ARC3-Inference/inference/framework/solver.py`
  (`board_changed:703`), `.../agent/tool_agent.py` (`_describe_last_outcome:1070` dead code),
  `duck-harness/tufa-arc-agi-framework/src/taaf/game.py` (legality gate `:511-524`).
- **Literature:** [arXiv:2204.01691](https://arxiv.org/abs/2204.01691) (SayCan — LLM-proposes × affordance-value
  grounding filter); [arXiv:2303.00855](https://arxiv.org/abs/2303.00855) (Grounded Decoding — "probabilistic
  filtering" of LM proposals against grounded env models); [arXiv:2606.31511](https://arxiv.org/pdf/2606.31511)
  (Falsification, Not Exposure — the filter/falsification role is the only feedback that carries signal for
  frozen models). SayCan/Grounded-Decoding use a LEARNED grounding; Carnot's contribution is REAL one-step
  execution as the filter — the thesis-consistent, non-gameable form.
- **Corroborating memory:** `project_arc_actions_to_progress_metric` (bottleneck = dynamics-induction; heldout
  uniformly 0.0), `project_arc_live_generator` (the frozen Qwen3.5-9B live stack — why a big live proposer is
  forbidden), `project_arc_live_agent_learning_gaps`, `feedback_arc_value_is_process_not_weights` (the
  deliverable is the reusable runtime-discovery PROCESS), `feedback_hybrid_pragmatic_architecture` (open LLM
  generator + energy ensemble as VERIFIER only — the shape this design instantiates at the action grain),
  `reference_zendoworld_hypothesis_uncertainty` (generation-not-selection corroboration).

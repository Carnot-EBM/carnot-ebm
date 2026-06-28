# ARC-AGI-3 Directed-Generation / Hierarchical-Planning research program (2026-06-28)

> **STATUS (2026-06-28): PROGRAM COLLAPSED ON DESIGN — both root-cause branches were already executed
> and refuted by the `.451–.454` chain (the operator-directed not-done-before check caught it BEFORE any
> build). Branch A (inducer strength) = exp4883 `METHOD_IS_CEILING`; Branch B (interaction-history /
> order-state observability) = exp4893 `REPRESENTATION_INVARIANT_HARD`. Conclusion: the generation wall
> is information-theoretic (gradient-free first-contact needle-search), not lever-shaped. Do NOT build a
> probe from this doc without a genuinely new paradigm + a fresh exclusion check. See §5.**

**Origin:** operator directive 2026-06-28 ("let's start the research program for generation —
directed candidate generation / hierarchical planning"), after every *selection*-class lever and the
full oracle-distinct structural-energy program (S0–S3) concluded with no live ARC value, leaving the
**generation wall** as the sole remaining blocker on live multi-level ARC-AGI-3 solving.

This is a **research program**, not a rapid lever. It is post-6/30 (the deadline deliverable is the
standing 0.08 submission + the FoVer paper). The structure mirrors the structural-energy program: a
single **cheapest-decisive probe (G0)** that retires a wrong core bet in ~1 day, then a fork into the
branch the probe selects. Build G0 FIRST.

---

## §1. The wall — pinned mechanistically (do NOT re-derive)

The binding constraint on live multi-level ARC-AGI-3 is **candidate GENERATION**, not selection:

> The winning multi-step L1 action prefix (length 4–33) is **NEVER ASSEMBLED into the candidate pool**.
> The explorer enumerates 488–4766 candidates/game but `matched_winning_prefix_len` reaches only 1–3
> then diverges (`project_arc_l1_first_contact_wall`; exp4851/`.447`; exp4914 buckets all
> `NEVER_ENUMERATED`).

The sharpest characterization (exp4914, `.454` A1 causal-state-abstraction diagnostic):
`fork_verdict: WALL_IS_HIDDEN_STATE`, `minimal_abstraction_is_observable_subset: False`. The
discriminating variable `winning_prefix_order_state` has `observable=False` — *"No ARC frame/env
extractor exposes the banked winning-prefix automaton index; it is interaction-dependent."* So the
proposal distribution is conditioned only on **observable frame features**, while the winning prefix
depends on a **hidden interaction-order automaton** the agent cannot perceive. Positive controls
(tu93 L5, ar25 L3) classify `OBSERVABLE_GAP` and solve. **It is wrong support, not search-depth, not
representability-of-a-given-subgoal.**

Baseline: generic first-win = **0.04 = 1/25** public games (only lp85 self-discovers L1 generically).
The wall is **first-contact (L1)** on 24/25 games; deepening only matters on the ~1 game already past
L1. (`reproducible_total_levels` 69 is a dev-proxy adapter count that scores ~0 on hidden.)

---

## §2. Decisively ruled out — the program must NOT re-run these (Failed-Experiment Rerun Discipline)

| Ruled-out cause | Evidence (verdict) |
|---|---|
| Selection / value-prediction | exp4903 `WALL_DEEPER_THAN_VALUE_PREDICTION`; corrected+baseline value heads both 0.04 |
| Perception representation | v3 LOO-AUROC 0.725 "representation DONE"; exp4904 `representation_invariant_4_classes` |
| Goal-induction / goal-detection | exp4664 `.430 goal_predicate_satisfiable=false` (RETIRED); GAP-4891 relational separates but doesn't guide |
| Primitive-vocab coverage | 99.1% clicks ≤2px of a centroid `guidance_not_coverage`; macro-vocab `guidance_not_depth` |
| Selection-over-archive | IGE LLM Go-Explore cell-selector (2026-06-28) `no_first_win_lift_0.000_vs_0.000` |
| Branch-pruning of the deepening search | relational-mask pruner (2026-06-28) generation-bound/position-dependent, 0 prunes |
| Combination-of-experts | PoE-World weighted consensus (2026-06-28) loses to max-vote (ARC per-cell deterministic) |
| Oracle-distinct structural energy (S0–S3) | real OFFLINE discriminator (S0' LOO 0.739) but S3 GENERATION bounded, 0 winners entered pool; PROGRAM CONCLUDED |
| Plain Go-Explore archive / NGU+RND directed exploration | exp4701/4831 `first_win 0.0`; exp4688 `winning_prefix_still_not_proposed` |
| energy-as-RFT-teacher | exp4100 operator-KILLED; exp4212 distill_lift=0; DO-NOT-RE-OPEN |

**The triangulation is ~8 paths.** A new generation lever is only admissible if it (a) is in the
genuinely-untried set, OR (b) names a NEW root cause AND what is different. "Better selection",
"re-weight the same explorer distribution", and "rely on the frozen 9B to propose the prefix" are all
re-runs.

---

## §3. The two candidate root causes (the program must disambiguate these FIRST)

Given the wall is "the winning prefix is never assembled by an induce→plan loop conditioned on
observable features", there are exactly two live root-cause hypotheses:

- **(A) INDUCER STRENGTH — REFUTED (2026-06-28, do NOT probe).** The hypothesis was: the frozen
  local 9B can't synthesize a world-model accurate enough to plan the prefix, but a STRONG
  coding-agent inducer (executable world models, arXiv:2605.05138) could. **This was already
  attempted AND is logically superseded:**
  - **Already attempted:** `experiment_4883_inducer_ceiling_ab` (`.450` A1b) ran exactly this A/B —
    `local` lane = Qwen3.5-9B-MTP vs `reference` lane = "Family-B reference executable-world-model
    inducer (arXiv:2605.05138)" over 9 held-out games → `inducer_ceiling_attribution: METHOD_IS_CEILING`,
    `complete_inducer_ceiling_neither_lane_lifts_method_is_ceiling`. (Caveat: that artifact is
    `flagged_adversarial: true` / DURATION_TOO_SHORT and its reference lane was a `capability_ceiling_only`
    placeholder, so the *clean* strong-inducer run is technically un-run — but see the logical refutation.)
  - **Logically superseded:** exp4914 `WALL_IS_HIDDEN_STATE`, `minimal_abstraction_is_observable_subset:
    False`. If the winning-prefix-order variable is **unobservable from frames**, NO inducer — however
    strong — can plan a prefix whose ordering depends on a variable absent from *every* inducer's inputs.
    Inducer strength is downstream of an observability problem. A clean strong-inducer re-run would almost
    certainly re-confirm METHOD_IS_CEILING; expected value ~0. **RETIRED — do not build the strong-inducer
    probe.**
- **(B) PERCEPTION → SYMBOLIC-STATE INTERFACE — ALSO REFUTED (2026-06-28 verification pass).** The
  hypothesis was: condition the representation/proposal on interaction-history (a proxy for the hidden
  order-state) and the winning prefix becomes recoverable. **This was already built and nulled:**
  `experiment_4893_action_prefix_latent_adapter` (`.451` A1b) used `latent_substrate:
  "action_prefix_future_state"` — a latent conditioned on the ACTION-PREFIX (interaction history) —
  over the same 9 held-out games → `complete_action_prefix_latent_no_value_lift_representation_invariant_hard`,
  value-accuracy delta median **0.0**, CI95 [−0.135, 0.025] (includes 0). Conditioning on
  interaction-history recovers NO predictive signal. The exp4914 order-state ("interaction-dependent
  classification ground truth") is definable only relative to the *unknown solution* — it is not a
  learnable function of observable experience, so action-history conditioning cannot surface it.
  **RETIRED.**

> **BOTH root-cause branches are already executed and refuted** by the `.451–.454` chain. The
> directed-generation / representation program was, on inspection, ALREADY RUN systematically:
> inducer-strength (exp4883 METHOD_IS_CEILING), decision-need value targets (exp4892
> REPRESENTATION_INVARIANT), action-prefix/history latent (exp4893 REPRESENTATION_INVARIANT_HARD),
> latent-action interface (exp4904 4_CLASSES), env-grounded search (exp4903 WALL_DEEPER), causal-state
> abstraction (exp4914 WALL_IS_HIDDEN_STATE). There is no cheapest-decisive probe left in the
> induce/plan/condition-on-observable family that is not a re-run.

These predict opposite next programs, so the cheapest decisive probe **must fork on them**.

---

## §4. The strong-inducer G0 is RETIRED before building (2026-06-28 not-done-before check)

> **The original §4 proposed a strong-vs-weak inducer probe. The not-done-before check (operator-
> directed) RETIRED it: it is a re-run of `experiment_4883_inducer_ceiling_ab` AND is logically
> superseded by `WALL_IS_HIDDEN_STATE` (see §3-A). Do NOT build it.** Preserved here as the rationale.

The retired probe would have swapped only the inducer (`LocalGGUFProposer` → `CodexProposer`/strong
coding agent), reused `plan_in_model`, and measured winning-prefix migration `NEVER_ENUMERATED →
ENUMERATED` on the frozen split. exp4883 already ran that A/B (`local` 9B vs `reference` Family-B
arXiv:2605.05138) → `METHOD_IS_CEILING`. Even granting exp4883's `flagged_adversarial` weakness, the
hidden-state finding makes a clean re-run's outcome a foregone conclusion (a strong inducer with the
same observable inputs still cannot recover an unobservable ordering variable). Expected value ≈ 0.

## §4'. The ACTUAL cheapest-decisive probe (Branch B): make the hidden ordering state observable

**Core bet (falsifiable):** *The winning prefix is un-enumerable because the proposal distribution is
conditioned only on the current frame; conditioning it ALSO on an explicit interaction-history feature
(a cheap observable proxy for the hidden order-state) makes the winning L1 prefix enumerable.*

**Sketch (to be fully specified + its OWN not-done-before check BEFORE building — do not skip that):**
add an interaction-history descriptor (e.g. a hash/RNN summary of the action-sequence-so-far, or a
visit-count automaton index) to the explorer's node state, then measure whether the winning prefix
migrates `NEVER_ENUMERATED → ENUMERATED` on the frozen failed-game split vs the frame-only control.
Falsifiable gate: prefix-assembly rate lift, CI95 delta excludes 0, reproduction-gated.

**Honest prior:** even this may null — if the order-state is not recoverable from action-history alone,
Branch B collapses to model-free exploration (Family-A "just-explore", arXiv:2512.24156) with an
interaction-history cell key (NOT the already-nulled frame-keyed Go-Explore archive). The expected
outcome is genuinely open — which is why it is the right next probe.

**MANDATORY before building G0' (the lesson of this turn):** run the same exclusion/rerun check against
the record (`arc-agi3-levers-tried-x-verdict`, exclusion manifest, results/) — confirm no prior
interaction-history / order-state / sequence-conditioned proposal experiment exists. The
interaction-history idea overlaps conceptually with the nulled NGU/RND episodic-novelty (exp4688) and
the AMAGO/Algorithm-Distillation in-context-exploration mapping (exp4697, unbuilt) — G0' must be NEW
vs both, or name what is different.

---

## §5. The fork — COLLAPSED: both branches already executed and refuted

The verification pass (2026-06-28, operator-directed "make sure we have not done this before") found
BOTH branches already run:
- **Branch A (inducer strength):** exp4883 `METHOD_IS_CEILING` + logically superseded by
  WALL_IS_HIDDEN_STATE.
- **Branch B (interaction-history / order-state observability):** exp4893 `REPRESENTATION_INVARIANT_HARD`
  (action-prefix latent, delta 0) — history conditioning recovers no signal.

So the program as designed has **no un-run cheapest-decisive probe**. The honest conclusion:

**The generation wall is information-theoretic, not lever-shaped.** For 24/25 games, first-contact is a
gradient-free needle-search: the winning prefix's order-state is definable only relative to the unknown
solution, is absent from the frame (exp4914), and is not recoverable from action-history (exp4893) or
any of 4 representation classes. No induce/plan/condition-on-observable lever can manufacture a gradient
that is not in the observable experience. This is why generic first-win sits at 1/25 (the games whose
prefix is short/salient enough to stumble on blindly).

**Thin residuals (low expected value — do NOT build without a fresh not-done-before check + a named
difference):**
1. **History-keyed model-free Go-Explore dedup** — exp4701/4831 used a FRAME-keyed coarse cell; a
   state-dedup key that treats history-distinct states as distinct is mechanistically different (it
   changes what the search keeps, not what it predicts). BUT the wall is "NEVER ENUMERATED" (the prefix
   isn't *proposed*), and dedup granularity doesn't change the proposal support — so low EV.
2. **AMAGO / Algorithm-Distillation in-context meta-exploration** (exp4697, unbuilt) — a meta-learned
   cross-episode exploration policy; but it conditions on history (exp4893 nulled that signal) and needs
   training the weak 9B, so it likely hits the same wall. Genuinely unbuilt, but low prior.

**The realistic paradigm-change options (not deadline-week levers):**
- A **strong coding-agent inducer AS THE LIVE AGENT** (the SOTA winners' actual mechanism, arXiv:2605.05138)
  — but this is offline-illegal for the competition sandbox, and exp4883's ceiling probe (flagged) +
  WALL_IS_HIDDEN_STATE suggest even it may not recover a hidden order-state. Needs a clean, genuine
  strong-inducer run to settle, but expected value is bounded by the hidden-state logic.
- A **much larger offline-legal local model** (bigger code model on the 3090s) — engineering, not research;
  only worth it if a clean strong-inducer probe first shows inducer strength matters (it probably doesn't,
  per §3-A).

**Recommendation:** treat live multi-level ARC-AGI-3 generation as a closed honest-negative for this
project's current paradigm. The deliverable stays the 0.08 submission (holding) + the FoVer paper. The
energy verifier's proven value is in *verification-bottlenecked* domains, not this *generation*-bottlenecked
one. Re-open only with a genuinely new paradigm (not a re-run of A/B).

---

## §6. Cross-references
- The wall: `project_arc_l1_first_contact_wall`, `project_arc_generation_not_selection`, exp4914
  (`WALL_IS_HIDDEN_STATE`), exp4903 (`WALL_DEEPER_THAN_VALUE_PREDICTION`).
- **Both branches ALREADY DONE (the not-done-before catches):**
  - Branch A: `results/experiment_4883_inducer_ceiling_ab.json` (`.450` A1b) — local 9B vs reference
    Family-B inducer → `METHOD_IS_CEILING`; `flagged_adversarial` (DURATION_TOO_SHORT, ceiling-only
    reference placeholder). Retires the strong-inducer G0.
  - Branch B: `results/experiment_4893_action_prefix_latent_adapter.json` (`.451` A1b) —
    `latent_substrate: action_prefix_future_state` (interaction-history conditioning) →
    `REPRESENTATION_INVARIANT_HARD`, delta 0, CI95 [−0.135, 0.025]. Retires the history/order-state G0'.
  - Supporting: exp4892 (decision-need REPRESENTATION_INVARIANT), exp4904 (latent-action 4_classes).
- Ruled-out census: `docs/research-notes/arc-agi3-levers-tried-x-verdict-2026-06-25.md`,
  `ops/exclusion_manifest.yaml`, `docs/research-notes/oracle-distinct-structural-energy-program-2026-06-26.md`.
- SOTA anchors: executable world models arXiv:2605.05138 (the strong-inducer mechanism G0 tests),
  PoE-World 2505.10819, Just-Explore 2512.24156 (Branch-B model-free fallback), IGE 2405.15143,
  GLoW/Dual-Scale 2509.24116, Strategy-Guided Exploration 2603.02045.
- Reusable infra G0 uses: `arc_executable_world_model.py` (`LocalGGUFProposer`, `CodexProposer`,
  `collect_transitions`, `plan_in_model`, `WorldModelVerifier`), the exp4851 frozen 10-game split.
- Discipline: Failed-Experiment Rerun + Exclusion-Manifest Cross-Check (§2); ARC Live-Path
  Reachability (any deployable Branch-A/B artifact must be live-reachable); solve_provenance honesty
  (G0 = `development_proxy`).

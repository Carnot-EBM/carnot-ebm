# ARC-AGI-3 Directed-Generation / Hierarchical-Planning research program (2026-06-28)

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

- **(A) INDUCER STRENGTH.** The frozen local 9B (Qwen3.5-9B-MTP) cannot synthesize a per-game
  world-model accurate enough that short-horizon planning IN it assembles the winning prefix. The
  SOTA winner (executable world models, arXiv:2605.05138, 58% RHAE) uses a STRONG coding-agent
  inducer (GPT-5.5). Our free-form 9B engine tops out at held-out transition accuracy **0.12**.
- **(B) PERCEPTION → SYMBOLIC-STATE INTERFACE.** Even a strong inducer is starved because ARC's
  glyph/grid perception does not expose the **hidden interaction-order state** (exp4914
  `WALL_IS_HIDDEN_STATE`); no induced model — however strong the inducer — can plan a prefix whose
  ordering depends on a variable absent from its inputs.

These predict opposite next programs, so the cheapest decisive probe **must fork on them**.

---

## §4. G0 — the cheapest decisive probe (build FIRST; ~1 day, offline, no new infra)

**Core bet (falsifiable):** *A strong external coding-agent inducer (not the frozen 9B) can synthesize
an executable per-game world-model accurate enough that short-horizon `plan_in_model` assembles a
winning L1 prefix the live explorer never enumerates.*

**Why it is decisive:** it swaps ONLY the inducer and reuses the existing search, so the result
cleanly attributes the wall to (A) inducer strength vs (B) the perception interface — the §3 fork.

**Procedure (reuses existing code — no new harness):**
1. Dataset: the frozen 10-game held-out split from exp4851 + the already-captured offline transition
   corpus per game (`arc_executable_world_model.collect_transitions`).
2. Induce each game's `(engine, is_level_complete)` TWO ways on the SAME K exploration frames:
   - **weak arm:** `LocalGGUFProposer` (Qwen3.5-9B-MTP) — the live inducer (the control).
   - **strong arm:** `CodexProposer` (gpt-5.5 via CLI — already in `arc_executable_world_model.py`)
     OR a Claude coding agent. (Internet-using → DEV DIAGNOSTIC ONLY; see the legality caveat.)
3. For each induced model, run the EXISTING `plan_in_model` BFS to depth = the game's winning-prefix
   length; record whether the winning L1 prefix migrates `NEVER_ENUMERATED → ENUMERATED` (assembled
   in the planned pool), plus held-out off-path transition accuracy.

**Falsifiable gate (the only non-circular evidence):**
- prefix-assembly rate (strong arm) **≥ 3/10 games**, bootstrap CI95 lower bound **> 0**, AND
- the strong−weak **delta CI95 excludes 0** (matched control, same games), AND
- strong-arm held-out transition accuracy **exceeds the 0.12** free-form-engine ceiling, AND
- anti-circularity: the assembled prefix is verified by the offline reproduction gate, not declared.

**Honest expected outcome:** ~40% it clears the gate (the SOTA result exists, but ARC perception may
starve even a strong inducer). **A NULL is highly informative** — it proves the wall is NOT
inducer-strength but the **perception→symbolic-state interface (B)**, retiring the entire
induce-and-plan family in one day and forcing the pivot to root cause (B) or model-free exploration.

**OFFLINE-LEGALITY CAVEAT (load-bearing).** The live ARC submission MUST run offline in the Kaggle
sandbox; a strong external coding agent is NOT offline-legal. **G0 is a DIAGNOSTIC, not a deployable
solution.** A POSITIVE G0 does not ship codex-as-inducer; it justifies the §5-A program (get a
strong-enough *offline-legal* inducer). `solve_provenance` for any G0 artifact is
`development_proxy` (it uses an off-path strong inducer), NOT `live_agent_self_discovery`.

---

## §5. The fork (which branch G0 selects)

- **G0 POSITIVE → Branch A: stronger OFFLINE-LEGAL inducer.** The 9B is the bottleneck. The program
  becomes: a larger local code model on the dev 3090s (offline-legal), better elicitation/scaffolding
  of the induce→verify→refactor loop (CEGIS, exp4872 lineage), or offline per-game model-building
  distilled into an offline artifact. The energy verifier reverts to its proven routing/pruning role
  over a pool that now contains the winner. Multi-week harness build.
- **G0 NULL → Branch B: perception→symbolic-state interface.** Inducer strength is not the wall; the
  hidden interaction-order state must be made OBSERVABLE (an interaction-history / automaton-index
  feature the proposal distribution can condition on) — or, if that is intractable, pivot the live
  agent to **model-free graph exploration** (Family-A "just-explore", arXiv:2512.24156) whose
  return-then-explore does not need an induced model, accepting the lower ceiling. (NB: plain
  Go-Explore archive already nulled — Branch B's model-free option must add the interaction-history
  cell key, not re-run the frame-keyed archive.)

---

## §6. Cross-references
- The wall: `project_arc_l1_first_contact_wall`, `project_arc_generation_not_selection`, exp4914
  (`WALL_IS_HIDDEN_STATE`), exp4903 (`WALL_DEEPER_THAN_VALUE_PREDICTION`).
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

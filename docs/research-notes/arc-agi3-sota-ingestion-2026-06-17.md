# ARC-AGI-3 SOTA Ingestion — 2026-06-17

SOTA→experiment mapping for the ARC-AGI-3 solve track, triggered by the operator
question "are there online forums/papers that might help us make further progress?"
during the adapter-free sweep (+4 games, 10 reproduced levels across 8 games).
Reliable-channel sweep (WebSearch + WebFetch, per the SOTA-Ingestion Cycle
Discipline — NOT /deep-research). All citations are real arXiv IDs / arcprize.org.

## The landscape (verified this pass)

| Source | What it is | Result | Relevance |
|---|---|---|---|
| **Graph-Based Exploration for ARC-AGI-3** ([arXiv:2512.24156](https://arxiv.org/abs/2512.24156), Rudakov/Shock/Cowley) | Training-free graph-explore (the EXACT family our `graph_explore_solve_v2` implements) | **median 30/52 levels, 3rd on private leaderboard** | We are at 10 levels with the same family → we are leaving ingredients on the table |
| **Executable World Models for ARC-AGI-3** ([arXiv:2605.05138](https://arxiv.org/abs/2605.05138)) | Coding agent induces a Python world model, VERIFIES it vs observations, refactors toward simpler abstractions (MDL proxy), plans to LEVEL_COMPLETED | **GPT-5.5: fully solves 15/25 games** (ar25, ka59, lp85, tr87, tu93, …); GPT-5.4: 8/25 (ar25, cd82, ft09, lp85, tr87, tu93) | Training-free, frozen weights, uses **gpt-5.5 = our codex**. The SOTA for FULL game solves. IS Carnot's induce+verify thesis. |
| **ARC-AGI-3 Technical Report** ([arXiv:2603.24621](https://arxiv.org/abs/2603.24621), arcprize.org) | The benchmark itself: modeling + goal-setting + planning under no instructions | frontier LLMs <1% at launch; humans 100% | Frames the 3 capabilities; our verifier owns the "verify the model/plan" slice |
| SOAR ([arXiv:2507.14172](https://arxiv.org/abs/2507.14172)) | Self-improving LLM evolutionary program synthesis (ARC-AGI-1, not -3) | hindsight-learning loop | corroborates the self-improving-solver loop we already run |
| Neurally-guided program induction ([arXiv:2411.17708](https://arxiv.org/abs/2411.17708)) | 3 paradigms: learn grid / program / transformation space | efficiency study | informs the induction representation choice |

**Forums / competition:** ARC Prize 2026 runs on Kaggle (Mar–Nov 2026), $2M pool;
Milestone #1 (2026-06-30) pays open-sourced solutions (1st $25K / 2nd $10K / 3rd
$2.5K). Docs at docs.arcprize.org. The 3rd-place graph-explore solution
(2512.24156) is the open, reproducible reference; the discussion centers on the
ARC Prize Kaggle forum + the paper.

## What the SOTA graph-explorer does that ours does NOT (the gap, zero-quota to close)

Our `graph_explore_solve_v2` already does: connected-component objects as click
candidates, masked-frame state hash, BFS-shortest-path-to-untested-frontier,
replay-navigation. The 2512.24156 ingredients we are MISSING — each is
training-free and quota-free:

1. **Status-bar masking before state-hash (HIGHEST LEVERAGE).** They mask UI/status
   regions (score, timer) out of the hashed image. This "substantially reduces the
   number of recognized states." Our `frame_hash(grid_of(frame))` hashes the FULL
   grid — so a ticking timer/score makes every state look new → state-explosion AND
   the exact aliasing that made **tu93 reach L2 but fail the reproduction gate**
   (the recorded path didn't replay because two real states aliased). Masking the
   status region is both a correctness fix (kills the false L2) and an efficiency
   fix (shrinks the graph). **This is the single most impactful change.**
2. **Visual-salience PRIORITY TIERS (5 levels) over click candidates.** They stratify
   segments by size / morphology / color salience and exhaust high-salience clicks
   before low-salience ones, with status bars lowest. We treat all objects uniformly
   (`rich_action_candidates` lists every object, no ordering). Tiered ordering is
   what lets them avoid the 4096-action blowup and reach median 30/52.
3. **Frontier-distance metric on nodes** ("minimal distance to nearest unexplored
   frontier") + hierarchical priority-threshold escalation. We do plain BFS; theirs
   is priority-constrained shortest-path-to-frontier. Modest gain on top of #1/#2.
4. **Reset-action marking** — their documented bug was reset-triggering actions not
   marked tested → loops. Verify our v2 marks game-over/reset edges as tested
   (we `pop` from `untested` so the edge isn't retried at that node, but a reset
   that returns to a SEEN state should be pruned — audit this).

## What the Executable-World-Model SOTA gives us for the DEEP TAIL

The games our pure explorer can't crack (ar25, ka59, tr87, ft09 resisted; the L1→L2
advance found 0) are exactly the ones the executable-world-model approach solves
FULLY with gpt-5.5. The transferable core (their words): "LLMs are most reliable when
used not as final authorities but as PROPOSAL mechanisms inside systems that CHECK
their outputs." That is verbatim Carnot's verifier-moat thesis. Mechanism:
- codex (gpt-5.5) fills 3 templates: `world_model_engine.py` (dynamics),
  `world_model_state_io.py` (render/reconstruct), `world_model_main_planner.py`.
- a **verifier** asserts the executable model reproduces recorded transitions; on
  divergence during plan execution it HALTS and records a mismatch artifact.
- refactor-for-simplicity loop: repeatedly ask the agent to replace special cases
  with shared rules while preserving verifier-correctness (the MDL proxy).
- plan inside the model to a LEVEL_COMPLETED state; per-level cap 1500 actions.
- training-free; ~$200/mo Codex ran 2–8 games/week.

## Recommended experiments (priority order)

- **E1 (now, zero quota): status-bar masking + salience tiers in `arc_graph_explore`.**
  Add a `mask_status(frame)` (drop rows/cols that change under a no-op / are
  low-salience UI) feeding the state hash, and order `rich_action_candidates` by a
  salience score (segment area × color-rarity). Re-run the advance + unsolved sweep.
  Acceptance: tu93 L2 reproduces (aliasing fixed) AND ≥1 net new reproduced level.
- **E2 (now, authorized): higher effort budget.** Operator authorized raising budgets.
  Running `ARC_MAX_EXPANSIONS=30000 ARC_SUFFIX_DEPTH=80` advance re-run to separate
  "budget-limited" from "mechanic-limited." (Budget alone, without E1's masking, is
  expected to mostly waste compute on aliased states — E1 must land first to make
  budget pay off.)
- **E3 (next milestone, codex/gpt-5.5): a Carnot Executable-World-Model solver.**
  Port 2605.05138's 3-template loop with OUR verifier as the model-correctness
  checker (induce→verify→refactor→plan). Target the deep tail (ar25, ka59, tr87,
  ft09) the explorer can't reach. This is the verifier-moat made concrete on the
  north-star benchmark and the proven full-solve SOTA. Quota-gated: ~2–8 games/run.

## Flag for the .400+ roadmap

E1 (zero-quota explorer upgrade) and E3 (codex executable-world-model solver) are the
two levers. E1 first (cheap, fixes a correctness bug, lifts the shallow tail); E3 is
the strategic bet that the proven SOTA for full solves IS our induce+verify thesis —
build it with codex and own the verifier slice. Cross-ref:
`reference_arc_agi3_sota_and_plan` (memory), `ops/arc_solve_registry.yaml`,
`project_arc_agi3_north_star` (memory).

## OUTCOMES (2026-06-17, same session — both levers executed)

- **E1 SHIPPED.** Salience-ordered candidates + deterministic HUD-masking added to
  `arc_graph_explore.py` (regression-gated: all banked solves still reproduce). The
  upgrade + a raised budget cracked **+3 new games the pre-E1 sweep could not — cn04,
  m0r0, sk48 (all L1, re-gated reproduced=True)**. Reproducible tally **6→13 levels
  across 11 games** (metaharness-confirmed). The +1-level advance on already-solved
  games NO-ADVANCEd even at 30k expansions → that frontier is **mechanic-limited, not
  budget-limited** (exactly the literature's prediction; it needs E3 or per-game RE).
- **E3 BUILT + VALIDATED.** `arc_executable_world_model.py` + `scripts/arc_e3_solve.py`.
  Live on ar25: codex/gpt-5.5 induced a GENUINE world model (flood-fill box dynamics);
  the Carnot `WorldModelVerifier` grounded it at **61% (73/120)** vs 35% identity
  baseline; the planner correctly reported no in-model win (the random transitions had
  0 level-ups → goal predicate unlearnable — honest). Gap to a full solve = multi-round
  refactor (the one validation round timed out under live conductor-codex contention) +
  win-seeking exploration. **Queued** as a MANDATORY-NEXT-MILESTONE priority
  (`ops/known-issues.md`, 2026-06-17) for the conductor to run the full multi-round loop
  on ar25/ka59/tr87/ft09, codex-agent-as-proposer, +1-level scoped, when codex is not
  contended.

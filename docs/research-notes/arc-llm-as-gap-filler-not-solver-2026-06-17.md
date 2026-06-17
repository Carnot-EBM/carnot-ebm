# ARC-AGI-3: the LLM is a GAP-FILLER, not a solver (2026-06-17)

Operator strategic correction (2026-06-17): "knowing that the local SOTA model is almost
always going to get 0% on ARC games itself, we need to be strategic on where such an LLM
can help dynamically identify and potentially help fill implementation gaps with the other
components that can actually make progress."

## The evidence: LLM-as-direct-solver ≈ 0%

- gemma-4-12B (local) induced world model for cd82: **0%** verifier accuracy (returned a
  scalar, omitted is_level_complete). codex (dev ceiling): **27%** — also too low to plan a
  solve. Frontier LLMs score **<1%** on ARC-AGI-3 directly. The verifier-grounding correctly
  REFUSED to trust these (no plan, no fabricated solve) — but that means E3-as-full-world-
  model-induction is the WRONG frame: the LLM cannot be the solver.

## The reframe: LLM fills the GAPS of the components that DO make progress

What actually makes progress on ARC-AGI-3 (measured this session):
- **Training-free explorer** (`arc_graph_explore` + frontier-distance): **8/11** public games
  solved from scratch at competition budget.
- **Verifier-routed search + learned verifier** (`arc_solver_kit`): efficiency (10.75x fewer
  states on lp85).
- **GameAdapter** (`arc_game_adapters.py`): the per-game DELTA — `state_key` (the load-bearing
  hidden state, e.g. sc25 tank-facing), `hand_verifier` (win/goal-distance predicate),
  salient `action_labels`. This is SMALL, focused, and is what unlocks a stalled game.
- **Verifier ensemble** + `ops/verifier_gaps.md` (the discrimination the ensemble lacks).

**The LLM's RIGHT job** = the focused REASONING + CODE tasks the LLM is good at, applied to
those components — NOT solving the spatial/combinatorial game itself:

1. **Write/fix the GameAdapter DELTA** for a stalled game. From observed transitions, the
   LLM infers the small per-game pieces: the win-condition predicate, the load-bearing
   hidden state for the dedup key, the salient action set / click-data schema (e.g. tn36's
   non-{x,y} ACTION6). This is the "reverse-engineer only the DELTA" of the ARC Solve
   Reproducibility discipline — the harness (search, verifier, gate, routing) is REUSED.
2. **Diagnose WHY a component stalled** and propose the targeted fix. "Explorer reached L0
   because the win needs click #27 that salience deprioritizes" / "dedup merges two states
   that differ only in facing" → a one-line state_key or candidate-ordering fix.
3. **Propose a new verifier INVARIANT** for a missing-verifier gap (`ops/verifier_gaps.md`):
   the discriminator the ensemble can't compute — feeding the GAP-3/GAP-4 verifier program.

## Why this is higher-leverage

- The LLM does what it's GOOD at (reason about why a component fails + write focused code),
  not what it's BAD at (directly solve the game = 0%).
- The per-game cost collapses to the DELTA (the adapter / the fix), which an LLM can write —
  not a whole solver. The heavy lifting (complete search, energy/verifier scoring,
  reproduction gate) stays in the deterministic components that already work.
- It is gain-not-churn: each LLM-filled gap is a durable component (a registered adapter, a
  new verifier) that compounds across games (the self-learning thesis).

## Architecture: the verifier-routed cascade, REFRAMED

The tier-3 LLM escalation does NOT solve the game; it **writes the missing COMPONENT** so the
working search can:
```
explorer / verifier-routed search        (tiers 1-2, deterministic, make progress)
   → STALL on game X
   → CHARACTERIZE the gap  (which component, what signal is missing — done by the verifier/
                            instrumentation: best level reached, where it stuck, what class
                            of transition/state it can't discriminate)
   → LLM (iGPU, local) FILLS the specific gap:  a GameAdapter delta, a state_key/candidate
                            fix, or a verifier invariant  (focused code, NOT a world model)
   → plug the filled component into the harness → re-run → measure (the focused loop)
```
The Carnot verifier remains the ground (it validates the LLM's adapter/invariant against
real transitions, exactly as it grounded — and rejected — the 0% world model).

## Consequence for the E3 work

E3's `plan_in_model`/full-world-model path is DE-PRIORITIZED (LLM-as-solver = 0%). The E3
machinery that KEEPS value: the transition collection, the `WorldModelVerifier` (now used to
GROUND the LLM's adapter/invariant, not a full model), and the local-iGPU LLM substrate —
repurposed for gap-filling. The next build is the **gap-characterizer → adapter-writer**
loop, targeting the games the explorer stalls on (wa30, cn04, sk48) and the tn36 click-schema
delta, with the LLM writing the small per-game piece, not the solver.

Cross-refs: CLAUDE.md "ARC Solve Reproducibility + Solver-Reuse" (RE only the delta) +
"Missing-Verifier Gap Logging"; `python/carnot/agentic/arc_game_adapters.py`,
`arc_solve_learning.py` (recommend_approach), `ops/verifier_gaps.md`;
`docs/research-notes/arc-agi3-focused-loop-and-engine-2026-06-17.md`.

# ARC-AGI-3 multi-level deepening: literature candidates (2026-07-03)

**Provenance:** outer-loop WebSearch pass (direct, low-concurrency — not `/deep-research`, per this
project's own documented rate-limiting history), triggered by the operator asking for papers relevant
to formulating a plan for multi-level solutions on hidden game levels with the live agent. Read the
primary papers before implementation (literature-priority discipline) — this note scopes candidate
pilots, it is not a substitute for reading them.

## Where this sits relative to what's already been tried

GAP-4891's decisive finding (`ops/verifier_gaps.md`): a relational goal-energy CAN correctly detect
the right target state at each new level (separates win from near-win on 3/4 games) but does NOT help
the search reach it — an energy-guided search and a plain BFS control stall identically. The binding
wall for within-game deepening is **trajectory enumeration**, the same wall as L1 first-contact, not
goal-detection or value-prediction. `exp5176` (`.474`) separately tried carrying a warm-started
world-model/belief-state across levels and found no validated lever from its own precursor tasks
(`exp5159`/`.473`, `blocked_gate_check_failed`). Any candidate below is only worth a pilot if it
attacks enumeration specifically — a better detector or a better belief-state representation alone
has already been shown insufficient.

## Candidates, ranked by how directly each attacks the enumeration wall specifically

### 1. CoEx — Co-evolving World-model and Exploration (arXiv:2507.22281)

**Mechanism:** reset-free emulator-state propagation across iterations — each K-step rollout begins
from the *saved emulator state of the previous iteration*, not a fresh reset. Combines a DAgger-style
teacher-relabel step with a pairwise process reward model. Reported to demonstrate agents advancing
through game milestones via within-episode continual adaptation.

**How this differs from what's already been tried:** `exp5176`'s warm-start was a belief-state /
world-model carry-forward (representation-level). CoEx's core move is *emulator-state* propagation
across iterations of the SAME search/training loop, not just carrying forward a learned model's
weights — the search itself resumes from where it left off rather than restarting exploration from
scratch each iteration. This is closer to a search-continuity mechanism than a representation-transfer
mechanism, which is the right layer given GAP-4891 diagnosed the wall as search/enumeration, not
representation.

**Honest risk:** the reported gains are in a different environment class; read the paper closely
before assuming the emulator-state-propagation trick transfers to ARC-AGI-3's turn-based,
non-resettable-mid-episode structure (ARC-AGI-3 games don't offer arbitrary rollback to a saved
emulator state the way a typical RL sim does).

**First pilot shape (public games only):** does resuming the search frontier itself (not just the
learned world-model) at a level transition — i.e., seeding the NEW level's search with the PREVIOUS
level's unexplored-but-visited frontier states, where structurally applicable — reduce actions-to-next-
levelup vs. a cold-restart control, on the already-identified "deepened but stuck" game set
(`ar25/bp35/cd82/cn04/dc22/ft09` per `.474`'s `exp5176`)?

### 2. Self-Evolving World Models for LLM Agent Planning (arXiv:2606.30639)

**Mechanism:** detect mismatches between predicted and observed outcomes, adapt the world model
accordingly — a self-evolution loop for the model itself, not just its use.

**How this differs:** exp5176's world-model warm-start assumed the carried-forward model was already
adequate; this paper's contribution is explicitly about *detecting when it's wrong* and correcting it,
which could matter if part of the deepening failure is a stale/wrong model silently misleading the
search rather than a pure enumeration problem. Read closely to determine whether this is genuinely a
different mechanism or another representation-accuracy fix (which GAP-4891 already showed isn't the
bottleneck) before committing pilot time.

**First pilot shape:** cheap — before any new pipeline work, check whether exp5176's actual failure
mode was "wrong model, confidently used" (this paper's target) vs "correct model, search still
doesn't reach the goal" (GAP-4891's target) by re-reading exp5176's own artifact for a prediction-
error signal. If the latter, this candidate doesn't apply and shouldn't be piloted.

### 3. Hierarchical RL with Landmarks / subgoal decomposition (arXiv:2504.04366)

**Status:** already flagged in this project's literature history (`docs/research-notes/search-layer-
literature-2026-06-11.md`, June 11) as "the key tractability lever for long-horizon games where flat
search blows up" — but never actually tested against the enumeration wall specifically. This is a real
gap between what was flagged as promising and what's been tried.

**Why it's still worth a pilot despite GAP-4891's null:** GAP-4891 tested whether a *scalar energy*
correctly identifying the goal helps search reach it — it did not. Landmark/subgoal decomposition is a
structurally different lever: it doesn't just score states, it partitions the search into smaller
sub-searches with intermediate targets, which can shrink the branching factor a flat best-first search
can't escape even with a perfect terminal-state heuristic. Worth being precise about this distinction
when scoping the task so it doesn't get dismissed as "already tried and nulled" when it hasn't been.

**First pilot shape:** on a deepened-but-stuck game, hand-identify (or induce from the agent's own
transition log) one plausible intermediate landmark between the current state and the next level's
goal; measure whether best-first search restricted to reach-landmark-then-reach-goal (two shorter
searches) succeeds where flat search on the full problem stalls at the same expansion budget.

### 4. Graph-Based Exploration for ARC-AGI-3 (arXiv:2512.24156) + open-source reference

**Status:** already cited in project memory (`project_arc_agi3_sota_and_plan`) as the 3rd-place,
open-source "Family-A" approach — no-induction, directed graph of explored states/transitions,
prioritizes shortest-path-to-untested-state-action-pairs, median 30/52 levels across 6 games.

**What's new here:** the actual reference implementation is public —
[github.com/dolphin-in-a-coma/arc-agi-3-just-explore](https://github.com/dolphin-in-a-coma/arc-agi-3-just-explore).
This project has cited the paper's numbers before but, per available records, never read the actual
code. Given the wall is enumeration/exploration specifically, a concrete open-source exploration
strategy that empirically clears half its levels is worth reading directly rather than re-deriving
from the paper's prose alone — it may contain graph-maintenance or frontier-prioritization
implementation details (data structures, pruning heuristics) not obvious from the paper text.

**First pilot shape:** a reading task, not a build task — read the repo, extract any frontier-
prioritization or graph-maintenance technique not already present in Carnot's own exploration code,
and report a specific delta (not a general "it's similar to what we do").

## What this note is explicitly NOT proposing

- Not re-litigating GAP-4891's finding that a correct goal-detection energy doesn't help search reach
  the goal — that result stands. Every candidate above is scoped to attack enumeration/search
  structure, not detection accuracy.
- Not a claim any of these will work — candidates 1 and 2 in particular need the "does this even
  attack the right layer" gate run before any pipeline investment, per each candidate's own honest-risk
  note above.
- Not touching the live/scored submission stack. Public-games-only piloting, per this project's
  standing offline-first discipline for ARC.

## A note on benchmark validity (context, not a lever)

**"Explore Before You Solve" (arXiv:2605.25931)** found all 25 public ARC-AGI-3 games are solvable via
non-intelligent strategies (10 in a single blind step, 8 via single repeated actions with a large
enough budget). This doesn't change any candidate above, but it's a reason to be skeptical of any
future claim that a technique "solved" a public game unless the solve is compared against a
non-intelligent baseline on the SAME game — a cheap sanity check worth running whenever a public-game
pilot reports a positive result. The paper also formalizes the Speed-Depth trade-off underlying RHAE's
quadratic efficiency penalty — worth reading for that framing even independent of the deepening
question.

## A note on the generation-vs-selection tension (context, not a lever)

**"Scaling Flaws of Verifier-Guided Search"** (arXiv:2502.00271) and **"Exploiting Verification-
Generation Gap"** (arXiv:2606.03608) both formalize the same generation-vs-selection split GAP-4891
found empirically — but "Scaling Flaws" finds the *opposite* dominant failure mode in its own domain
(math reasoning): most search failures there are attributable to selection (a valid path exists but
the verifier misranks it), not generation. Worth remembering the dominant failure mode may be
domain-dependent (dense proof-search trees vs. sparse combinatorial action spaces) before assuming any
future non-ARC verifier work would hit the same generation-wall this project's ARC program found.

## Cross-references

- `ops/verifier_gaps.md` GAP-4891 — the enumeration-wall finding every candidate above is scoped
  against
- `results/experiment_5176_deepen_live_levelup_attempt_v474.json` — the most recent deepening attempt,
  `complete_blocked_no_validated_lever_from_b1_b2_zero_levels_banked`
- `docs/research-notes/search-layer-literature-2026-06-11.md` — the June 11 survey that already
  flagged landmark decomposition (candidate 3) without a direct test
- `project_arc_agi3_sota_and_plan.md` (memory) — prior citation of the graph-exploration paper
  (candidate 4)
- CLAUDE.md "ARC-AGI-3 Incremental-Progress Scoping" — any pilot task drawn from this note must scope
  to +1..+n levels on one game, never "solve everything"
- CLAUDE.md "ARC Live-Path Reachability Discipline" — any pilot that lands a real technique must be
  wired into the live agent path, not a standalone offline script

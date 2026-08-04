# Pre-registration — leave-one-game-out ADAPTER-FREE / held-out-identity measurement

Written **before** the sweep was launched and before any roster-wide result existed. Two
observations made during scoping, before this document, are declared here so they cannot be
mistaken for confirmations of a hypothesis formed after them:

* a `vc33` smoke pair (control vs held-out, seed 1, budget 400) came back **identical** on every
  recorded field;
* an `sb26` control cell reached **0 levels** in 392 actions.

Everything else below is fixed in advance.

## What is being measured

CLAUDE.md's *ARC-AGI-3 Generalization-Testing Floor* (2026-07-17) asks how far the **live** agent
gets on a game when its per-game knowledge is removed and it must rely on the reusable
scaffolding alone. All 25 public survey games are cleared (183/183 levels), but that clear is
`development_proxy` work carried by hand-built per-game `GameAdapter`s. A hidden Kaggle game has
no adapter.

## The correction to the task's premise, stated up front

The adapter is **not** what the scored path would lose. An independently computed import closure
(ast walk, absolute + relative + function-level imports, from both canonical entrypoints)
confirms:

* scored closure (`arc_competition_agent`): **55 files**, `arc_game_adapters` **ABSENT**;
* dev-twin closure (`scripts/arc_loop_solve.py`): 160 files / 69 agentic, `arc_game_adapters`
  **present**.

So on the scored path there is no adapter to remove. What a hidden game actually loses is the
set of **id-keyed lookup sites**. The treatment removes those instead, by handing the POLICY a
synthetic identity while the ENV keeps running the real game.

## Arms

| | `control_identity_on` | `heldout_identity_off` |
|---|---|---|
| `policy_game_id` | the real game id | `"hg" + sha256(game\|heldout)[:6]` |
| engine store (`CARNOT_ARC_E3_DIR`) | private temp dir **seeded with a copy of** `results/arc_e3/<game>/` | private temp dir, **empty** |

Byte-identical between arms: seed, 400-action budget (the shipped `CarnotAgent.MAX_ACTIONS`),
the LLM-off stub proposer, the env (the real game), the measurement instruments.

`explore_budget` is passed as `None` so each arm gets the budget **its own** knowledge state
routes it to (24 when the registry records a goal-distance mechanic class, else 80). This is the
treatment, not a confound: passing an explicit 24 would force the control's routed budget onto
the held-out arm and cancel one of the leaks being removed.

## Seeds and roster

Seeds 1, 2, 3. All 25 games from `ops/arc_solve_registry.yaml` (read from the file, not
hardcoded). 25 x 2 x 3 = 150 cells.

## Metrics

* **PRIMARY — banked levels**: `levels_gained` at budget 400, gated by
  `arc_solver_kit.reproduce` over the recorded `action_trace` with the generic, adapter-free
  `replay_apply`. A live-recorded trajectory is not a banked level (CLAUDE.md ARC Solve
  Reproducibility).
* **SECONDARY — `actions_to_first_solve`**: the axis that is not budget-bound and converts to
  score quadratically.
* **REFERENCE (not budget-matched, and must never be reported as a control)**: the registry's
  per-game `levels_reproduced` (the 183/183 hand-tuned dev-twin reach). It was produced by an
  unbounded best-first search through a hand-built adapter, not by a 400-action live episode.

## Statistical test, fixed in advance

Unit of clustering: the **GAME** (n = 25). Seeds are replicates *within* a game; a game
contributes one paired delta (mean over its seeds per arm). Two-sided **exact sign test** on the
25 per-game deltas.

**MIN REACHABLE p, at game clustering: `2 x 0.5^25 = 5.96e-08`** — attainable only if all 25
games are discordant in the same direction.

Because ties are dropped, the reachable p is governed by the number of **discordant** games *k*:

| discordant games *k* | min reachable two-sided p |
|---|---|
| 6 | 0.03125 |
| 5 | 0.0625 |
| 4 | 0.125 |

**If fewer than 6 games are discordant, no result significant at 0.05 is reachable at all**, and
the honest report is a bounded null, not a negative.

Expected direction: treatment <= control (a drop). A large drop is the expected and most useful
result.

## Honesty guards, fixed in advance

1. **`explored_out` audit.** A cell that used fewer actions **and** has `explored_out = True`
   exhausted its frontier; that is a frontier collapse, not a saving. Reported separately.
2. **Denominator by name.** Any game whose cell errors, times out, or cannot be run leaves the
   denominator **by name**, with the reason. Never silently dropped.
3. **Distributions, never a single mean.** min / q1 / median / q3 / max per arm.
4. **Missing is not zero.**
5. **Vacuity check.** If control and treatment are identical for (nearly) every game, the
   measurement is **vacuous** and must be reported as such — "no drop" would be a false claim.
   The delivery ledger (each id-keyed site instrumented at the CALLEE, caller read off the
   stack) is what distinguishes "the leak was removed and nothing changed" from "the leak was
   never delivered in the first place".

## Ceiling caveat that no result can escape

`environment_files/<game>/` — a perfectly faithful, instantly resettable, zero-latency,
unlimited-retry local simulator — cannot be disabled; it is the definition of the offline twin.
No hidden Kaggle game provides it. Every number this produces is therefore an **upper bound on
hidden-game reach under a generous harness**, never a measurement of it.

## Provenance

`solve_provenance: development_proxy`. No hidden game is played, nothing is submitted, no scored
or online run occurs, `results/arc_e3` is copied from and never written to.

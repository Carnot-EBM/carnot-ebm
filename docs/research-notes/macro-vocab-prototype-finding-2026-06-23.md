# Macro-action vocabulary prototype — honest finding (outer-loop, 2026-06-23)

Operator asked to speed up the `.429` macro-action candidate by prototyping it in the outer loop.
Built it, ran it. **Result: the naive repeat-until-stable macro is a NULL — no banked level, and it
dilutes the search on the reachable tests. NOT merged to main.** This note records why, so the real
`.429` task starts sharper.

## What was built (prototype on the now-purged branch `outer-loop/macro-vocab` — code not retained)
- `arc_macro_actions.repeat_until_stable_macros` — one push-until-blocked macro per non-click keyboard
  action (repeat the action until the frame stops changing).
- `graph_explore_solve_v2(macro_provider=...)` — optional knob, **default None = byte-identical** to
  the proven path (parity preserved). A macro expands as a primitive sequence (trajectory stays
  primitive-faithful/reproducible); each applied primitive is charged as one expansion (fair budget).
  Macros are **additive** candidates (primitives stay in the pool) → cannot regress correctness.
- 4 unit tests pass.

## Empirical result (two gates)

**Gate 1 — L0→L1, 7 games, fair budget (`proto_macro_horizon_ab.json`):** `any_deeper=False`,
`any_cheaper=False`. The macro often did WORSE: tu93 ctrl L1 / macro L0; sp80 ctrl L1 @382 exp /
macro L0 @2500. Two clean reasons:
1. **Wrong regime.** These L1s are not horizon-bound (sp80 L1 = 382 primitive steps). Horizon
   collapse only pays when the level genuinely needs a long action run.
2. **Fair budget + branching dilution.** Counting each macro primitive as one expansion (the honest
   action-cost) means the macro arm does strictly more work per state, and macro-spawned far-states
   add branching — so at equal env-step budget it explores fewer of the states on the short winning
   path.
3. **Overshoot.** "repeat-until-stable" is right for push-until-blocked but WRONG for fixed-count runs
   (g50t L1 = `44445…`: pressing `4` a 5th time is not a no-op, so the macro overshoots the win).

**Gate 2 — prefix-to-L2 (`proto_macro_l2.json`):** pin the stock-v2 L1 trajectory, search L2.
`any_macro_banks_deeper=False`. sp80: both arms exhaust 5000 expansions at L1 (clean null). tu93:
**inconclusive** — the L2 frontier was degenerate (8/16 expansions then empty), a harness limitation
of the prefix→L2 handoff, not a clean macro falsification.

## What the real `.429` task needs (sharpened by this null)
1. **Don't blind-repeat.** Induce macros from the agent's OWN observed winning runs (the action
   sub-sequences that actually recur with a consistent frame-delta), not a blind repeat-until-stable —
   so a macro matches a fixed-count run instead of overshooting it. This is the empowerment-keep
   criterion the flag described; the prototype shortcut (repeat-until-stable) skipped it.
2. **Fix the budget framing.** Either count a macro as ONE decision (test whether horizon collapse
   helps independent of env-step cost) or only deploy macros where the level is *confirmed*
   horizon-bound (cell_recall probe) — don't pay full per-primitive cost on non-horizon-bound levels.
3. **Fix the prefix→L2 harness** before using it as a gate (tu93's degenerate 8-expansion L2 search
   means the gate itself isn't measuring what it should yet).

## Gate 3 — the SMART version, and the definitive verdict (`proto_macro_smart.json`)

Addressed both prior failure modes: **log-length macros** (run of 2/4/8/16 → the right length is among
the candidates, no overshoot) + **per-decision budget** (`macro_unit_cost`: the whole macro = ONE
expansion, isolating horizon collapse from env-step cost). Ran on 5 keyboard-run games (ar25 run-10,
cn04 run-9, m0r0, ls20, dc22), budget 2000. **`any_macro_deeper=False`.** ar25/cn04/m0r0/dc22 reach NO
level in any arm; ls20 the macros HURT (control L1, macros L0).

**Definitive verdict: horizon-collapse-via-macros is dead for this corpus.** The decisive reason: the
known solutions are **interleaved sequences of fixed-count runs of *different* actions** (ar25 =
`3×5, 2×10, 2×8`), so finding a solution needs the right action in the right order — the search is
**breadth / guidance-bound, not depth-bound**. Macros multiply the branching factor (24 candidates vs
4) *without* providing a guiding signal, so they trade depth for breadth when breadth is the binding
constraint → strictly worse. The corpus run-length scan confirms runs are fixed-count (puzzle
lengths), not push-until-blocked, and you cannot know a run's count without already solving the level.

**This corroborates the broader project finding:** the 0.04 live solve-rate is a **generation-GUIDANCE
wall** (which action to take — what the `.427` action-effect predictor and `.428` goal-energy /
expansion-prior attack), NOT a horizon/depth wall. The `.429`-improvement workflow ranked the macro
lever PURSUE_HIGH on the *theory* that 0.04 is a depth wall; the empirics decisively refute that.

## Disposition
Mechanism prototyped on the now-purged branch `outer-loop/macro-vocab` (was parity-safe + 6 tests); **not merged, not retained**
— a definitive null should not land on main as progress. **The `.429` macro candidate is RETIRED** per
its own `retire_if_same_verdict: true` gate (the falsifiable gate — "bank a deeper level at equal
budget" — came back False across blind, run-dominated, prefix-to-L2, and smart-per-decision tests).
Redirect `.429` effort to a GUIDANCE-class lever: the click-heatmap-as-generator (the other PURSUE_MED,
which adds *candidates* the centroid pool omits — a guidance fix, not a depth fix), gated behind its
30-min `winning_click_centroid_coverage` falsifier.

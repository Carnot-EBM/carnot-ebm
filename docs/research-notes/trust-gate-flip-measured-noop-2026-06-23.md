# Trust-gate flip (cell_recall) measured — a provable NO-OP on the live path (outer-loop, 2026-06-23)

Operator asked to run the SOTA-avenues "Avenue A" measurement: does flipping the already-built
`CARNOT_ARC_TRUST_METRIC=cell_recall` trust gate lift live first-win? **Answer: NO — it is a provable
no-op on the path the flag governs.** No LLM/GPU run was needed; the gate-pass data proves it.

## The setup
The live env-var gate (`arc_competition_agent.py:1806`) governs the **e3 LLM/DSL-induction** path: it
scores the INDUCED world-model with `WorldModelVerifier` and skips it when the gate value < 0.5.
Default `exact` (full-grid accuracy); the lever flips it to `cell_recall` (graded changed-cell recall).
The agent's behavior changes ONLY if the gate DECISION changes on some game — i.e., a game that is
exact-FAIL but cell_recall-PASS. So the measurement is a cross-tabulation of the two existing per-game
probes (`results/proto_trust_gate_flip_analysis.json`):

| path | game | exact | cell_recall | gate-flip FAIL→PASS |
|---|---|---|---|---|
| **e3 (flag-governed)** | cn04 | 0.133 | **0.015** | no |
| e3 | sc25 | 0.350 | 0.055 | no |
| e3 | ar25 | 0.667 | 0.857 | no (both pass) |
| e3 | cd82 | 0.192 | **0.000** | no |
| e3 | ka59 | 0.167 | 0.463 | no |
| **TTT (NOT flag-governed)** | ka59 | 0.000 | **0.912** | **YES** |
| TTT | sc25 | 0.000 | **0.797** | **YES** |
| TTT | tn36 | 0.000 | **0.871** | **YES** |
| TTT | lp85 | 0.000 | **0.590** | **YES** |

## Verdict: DEAD on the live path
**e3-path gate-flips = 0.** On the path the flag governs, the LLM/DSL-induced dynamics have
`cell_recall ≈ 0` (cn04 0.015, cd82 0.0, sc25 0.055) — they are WRONG, not imperfect-but-useful, so
NEITHER gate trusts them and flipping the flag changes no decision → identical agent → identical
first-win. **`live_first_win_provably_unchanged: true`.** The wall on the live induce→plan path is
**induction QUALITY, not the trust gate.** This resolves Avenue A: the cheapest, PURSUE_HIGH-ranked
lever ("flip the built flag") is empirically a no-op live.

## The sharper, genuinely-untested lever it surfaces
The **TTT learned-dynamics** path (prior-warmstarted CNN, a DIFFERENT mechanism the env-var gate does
NOT govern) flips **4 games** FAIL→PASS (ka59 0.91, sc25 0.80, tn36 0.87, lp85 0.59 cell_recall) —
those models ARE imperfect-but-useful where the LLM-induced ones are wrong. So the real lever is NOT
"flip the gate on the e3 path" but **"route live trust to the TTT dynamics on the games where they pass
cell_recall,"** then measure whether a trusted TTT model drives `plan_in_model` to a live win. That
needs the TTT CNN wired into the live plan path + a GPU run — genuinely untested, and the honest
next step if this thread is pursued. (Caveat: ka59 is a hidden-state game routed through
`select_trusted_world_model`, so the TTT-route applies cleanly to sc25/tn36/lp85.)

## Disposition
Measurement artifact `results/proto_trust_gate_flip_analysis.json` + this note on main. Avenue A
(cell_recall gate flip) is RESOLVED as a live no-op — do NOT default the flag on expecting a first-win
lift. The induction-quality wall is what `.428` A1/A2 (goal-energy / expansion-prior) attack; the
TTT-route is a separate, GPU-gated follow-on.

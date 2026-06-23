# Head-to-head: just-explore vs our BARE graph explorer — a POSITIVE, extractable signal (outer-loop, 2026-06-23)

Operator asked to build the just-explore head-to-head shim and run it (the decisive gating measurement
from `sota-rapid-accel-levers-2026-06-23.md` #1). Built, run, and **adversarially audited**. **Result: a
genuine positive — just-explore's exploration schedule reaches first-wins / deeper levels our bare
offline graph explorer misses on 5 games (median-of-3), conservatively biased AGAINST just-explore.**
This is the session's first non-null lever and green-lights porting its exploration schedule.

## The shim (verified)
Drives just-explore's `HeuristicAgent` (3rd place, arXiv:2512.24156) on OUR offline arcade by overriding
only `take_action()` → `env.step()`, mapping our `levels_completed → FrameData.score` (bare-package
import bypass for the LLM-template deps; zeroed the agent's 0.31s online rate-limit sleep). **Shim-validity
preflight passes:** replaying lp85's banked 54-action L4 trajectory drives the score 0→4 correctly.
Code: branch `outer-loop/h2h-just-explore`, `scripts/experiments/proto_h2h_just_explore.py`;
data: `results/proto_h2h_just_explore.json`.

## The result (25 games, budget 1000, just-explore best-of-3 vs Carnot bare graph_explore_solve_v2)

| games where just-explore reaches STRICTLY deeper | JE median | JE best | bare-explorer | robust? |
|---|---|---|---|---|
| **bp35** (UNSOLVED first-contact) | 1 | 1 | 0 | yes |
| ft09 | 1 | 1 | 0 | yes |
| m0r0 | 1 | 1 | 0 | yes |
| r11l | 1 | 1 | 0 | yes |
| **vc33** | 2 | 2 | 1 | yes |
| tu93 | 0 | 2 | 0 | best-of-3 only (1/3 seeds) |
| s5i5 | 0 | 1 | 0 | best-of-3 only |

Bare explorer reaches strictly deeper on only **1** game (cd82). Everything else ties. Conservative
**median count = 5** just-explore-deeper; best count = 7. `VERDICT: JE_STRATEGY_VALIDATED_extract`.

## Adversarial fairness audit (hostile reviewer + instrumented env) — the result is a LOWER BOUND
Every fairness axis biases AGAINST just-explore **except** best-of-3:
- **Budget (~4× against JE, measured):** `graph_explore_solve_v2` replays-from-reset, so at nominal-equal
  "1000" the bare explorer did **4.2× (lp85) / 4.4× (bp35)** more real `env.step`+`reset` interaction than
  just-explore — and still lost. JE wins while touching the env ¼ as much.
- **Degradation (strict lower bound):** just-explore threw ~960 graph re-inits / 3 games (HUD-hash
  instability on our offline frames). The upstream README states fixing this exact bug raises it from
  12 → median 17 private levels. The re-inits can only *suppress* JE's reached level, never inflate it
  (a level-up requires a real env `levels_completed` increment). A de-degraded JE wins MORE.
- **Score mapping (sound):** tied to the env's authoritative level counter; cannot inflate.
- **Reset/warmup (neutral→against JE):** RESET wipes progress to L0 (no free levels); JE even forgoes the
  warmup first-action burn the bare explorer gets.
- **Best-of-3 (the one pro-JE axis):** inflates the *count* (7 best vs 5 median), not the *existence* —
  the 5 median-robust games hold under median-of-3. (tu93/s5i5 are best-only and flagged.)

## Two framing constraints (do NOT over-claim)
1. This is just-explore vs our **BARE** `graph_explore_solve_v2`, NOT vs the **live** `E3AgentPolicy`
   (which adds the CNN expansion-prior + LLM induction + world-model trust). Report it as "JE's
   exploration *schedule* reaches first-wins our bare explorer misses" — the right control for isolating
   the *schedule* — NOT "just-explore beats Carnot."
2. Some win-games (ft09/m0r0/r11l) we already solve via per-game adapters; but this is ADAPTER-FREE
   first-contact (what matters for hidden games), and **bp35 is genuinely unsolved** — JE first-contacts
   it to L1 where we get L0.

## What this green-lights (the extractable lever, ~1–2 days)
Port just-explore's **exploration schedule** into our `rich_action_candidates` / explorer and A/B it:
- The **5-tier salience-deferred exhaustion** (`heuristic_agent.py:866-898`, N_GROUPS=5) — bucket
  candidates (T0 = salient color ∈ {6..15} AND medium-width 2≤w,h≤32) and emit tier-ascending, instead of
  our flat `area*(1+1/color_rarity)` sort that up-ranks the giant background blob.
- The strict global frontier-exhaustion ordering (`graph_explorer.py:384`).
Gate: the grafted schedule reaches a first level-up on ≥2 of {bp35, ft09, m0r0, r11l, vc33} that our
current `rich_action_candidates` ordering misses, zero regression on solved games. This is the genuine
SOTA-extraction win the week was looking for — and bp35 (unsolved) is the highest-value target.

## Honest bottom line
The head-to-head did its job: it is **decisive and positive on the conservative measure**. The
exploration schedule is the real, extractable lever (not depth/coverage/gate — those died this week).
Realistic week outcome: graft the tier schedule, A/B on the 5 win-games, aim to first-contact bp35 (and
possibly the other 3 unsolved games re86/sb26/lf52 the degraded JE didn't reach but a de-degraded one
might). Cross-refs: `sota-rapid-accel-levers-2026-06-23.md`, branch `outer-loop/h2h-just-explore`,
`/home/ianblenke/arc-sota-refs/arc-agi-3-just-explore` (arXiv:2512.24156).

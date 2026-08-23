# Trajectory-Supervisor Offline Replay: It Fires, and It Always Would Have (2026-08-23)

## The question

The operator's next loop iterates on the ARC trajectory supervisor
(REQ-ARC-WMTE-6600), but the supervisor has never been OBSERVED to fire, and
exp6524 correctly blocked the refinement task for lack of outcomes. Two
worlds: (a) it fires but does not help; (b) it never fires, and every A/B is
predetermined null. A live A/B (supab4) costs hours of GPU per attempt; this
note answers the question offline, before supab4 lands, so the prediction
can be scored against the real outcome.

## The trigger condition, derived from the code

`arc_trajectory_supervisor.py` + the call site
(`arc_competition_agent.py:_next_move_routed`, one `observe()` per routed
action; enabled only when `CARNOT_ARC_TRAJECTORY_SUPERVISOR=1`, window `W`
from `CARNOT_ARC_TRAJECTORY_SUPERVISOR_WINDOW`, default 400).

Let `s` be the stagnation counter. Per observed action with level `L`:

```
if L > max_level_seen:   s := 0; arms_used := {}          (progress)
else:                    s := s + 1
fire iff s >= W          (then s := 0 whether or not an arm is eligible)
```

At a firing, the FIRST eligible arm redirects, at most once per arm per
level segment:

```
arm1 drop_goal_bias        iff goal_bias_installed
arm2 allow_reinduction     iff induced AND new_transitions_since_induction >= 200
                               AND induction_attempts < 3
arm3 force_exploration_diversity  iff NOT diversity_active
```

Load-bearing fact: `diversity_active` starts False (env
`CARNOT_ARC_EXPLORE_DIVERSITY` defaults off) and NOTHING except arm3 itself
ever sets it. So at the FIRST window expiry of any level segment, arm3 is
unconditionally eligible. The supervisor cannot reach a 400-stagnant window
and stay silent — "fires never" is only possible if the window never
elapses, or the wiring is off.

## Why "never observed to fire" was never evidence

The prior A/Bs could not have observed a firing REGARDLESS of behavior:

- supab (first A/B, ON arm, `tmp/supab/rows_on.json`): every row carries
  `trajectory_supervisor: null` — the run predates the REQ-ARC-WMTE-6640
  receipt plumbing. Firings were UNRECORDABLE, not absent.
- supab2 (VOID, OOM) and supab3: the only rows on disk are the OFF arm
  (`{"enabled": false}`, correct for off).

So the corpus-wide "zero artifacts with any arm outcome" is a record gap,
not a behavioral null. The dichotomy's world (b) was never measured.

## Offline replay over the 25-game baseline

The expiry count is EXACTLY computable from row aggregates: `actions` plus
`level_up_actions` fully determine the counter (assumptions below). Per
game, segments between level-ups of length `g` contribute `floor(g/400)`
expiries. Baseline (supervisor OFF, thinking ON, budget 2000-class,
actions 1280-1986):

| result | value |
|---|---|
| games where the supervisor would have fired | **25 / 25** |
| total window expiries | 94 |
| total redirects, lower bound (arm3 only) | 31 |
| total redirects, mid estimate (arm3 + arm2) | 50 |
| total redirects, upper bound (all arms) | 78 |
| per-game redirects (mid) | 2-4 |

Every game has at least a 1223-action terminal stagnant segment; 14 of 25
never level up at all (1676-1986 stagnant actions = 4 expiries each). The
full per-game table is in the session log; the replay script is 30 lines
over the two rows files.

Assumptions, stated: (A1) one `observe()` per harness-counted action
(`actions == n_actions_counted` on all 25 rows; plan-execution actions that
bypass `_next_move_routed` would shift expiries later without removing
them — the stagnant tails are explore-routed); (A2) transitions accumulate
at roughly one per action, so the 200-new-transitions floor is met within
any 400-action window after an induction attempt (every game records
exactly one stall-triggered attempt, more where levels advanced); (A3)
`level_up_actions` entries are new-maximum level events (consistent with
`levels` on all rows).

## Prediction for supab4 (stated before its rows land)

supab4 = ar25, tu93, sp80; thinking ON; induce timeout 3600; supervisor ON,
window 400; post-6640 harness, so receipts will exist.

1. Every game's `trajectory_supervisor` receipt: `enabled: true` with a
   NON-EMPTY `redirects` list. If any row shows zero redirects with
   `actions >= ~450` since last progress, the wiring — not the detector —
   is broken (env not reaching the agent process, or observe not on the
   executed path).
2. Per game (baseline-shaped trajectories): ar25 ~4 expiries, 2 fired
   (range 1-3), `stagnations_unredirected` ~2; tu93 ~3 expiries in the
   post-L2 tail, 2 fired, 1 unredirected; sp80 ~3 expiries after its ~331
   level-up, 2 fired, 1 unredirected.
3. First redirect per stagnant segment: `force_exploration_diversity`,
   arriving ~400 actions after the last progress. If the 2026-08-22
   induction fixes install a goal bias on these games, `drop_goal_bias`
   fires FIRST and per-segment counts shift up by one.
4. Second redirect: `allow_reinduction` one window later (evidence floor
   met per A2).
5. `arm_outcomes.helped`: predicted **0 across the board**. The redirects
   arrive deep in stagnation; nothing in the baseline suggests a diversity
   draw or a re-induction converts these tails into level-ups within the
   remaining ~1200 actions. The honest expected headline for supab4 is
   world (a): fires, does not (yet) help — which is the supervisor-
   refinement loop's real starting point, not a reason it was unmeasurable.

## The instrumentation that ends the guessing

The detector is replayable offline for its EXPIRY count but not for its ARM
choice (goal-bias/diversity/transition eligibility bits are not recorded
per window). The smallest change that makes every future run fully
replayable — and every OFF arm carry its own counterfactual — is a SHADOW
MODE: instantiate the supervisor on every run, `observe()` as normal, but
when the env flag is off, record the would-have redirect in the receipt
instead of applying it. Zero behavior change on the scored path; the OFF
arm of every future A/B then reports exactly what the ON arm would have
done, and "run it and see" becomes "read the receipt." Not built here —
measure-first, and the agent file is under active iteration — but it is
the recommended next mechanism for the ARC Generalization-Testing Floor's
activity-4 slot.

# E4 frozen game roster (adapter-free pairs), 2026-09-20

**Status: FROZEN before any E4 outcome exists.** No E4 run has happened. The roster is append-only: a game may be added only by a dated entry below, made before that game's outcomes are read. A game may never be removed or swapped after outcomes are read.

**Origin.** Operator answer to question 6 of `semif-ebm-arc-experiment-plan-2026-09-20.md`: "follow your recommendations". The outer loop drafted the roster with the fixed rule below so that no one chose games by hand.

## Selection rule

1. Source: the 25 public games in `ops/arc_solve_registry.yaml` (sha256 of the file at drafting time: `071ecd51939117d9bc5b48491b0649e2ae126e3f848b5af8ff7e53c88e890947`).
2. Stratum: the action interface, from the registry `action_model` text. `mixed` if it names both keyboard actions and clicks. `click` if it names only clicks or ACTION6. `keyboard` otherwise. The registry `mechanic_class` was NOT used: 22 of 25 games have a unique class, so it gives no strata.
3. Within each stratum, rank games by `sha256("carnot-e4-roster-2026-09-20:" + game_id)` ascending and take the first 3.
4. Nine games total, three per stratum.

## Roster

| Stratum | Game | Registry levels reproduced | Rank hash (first 12) |
|---|---|---:|---|
| keyboard | sk48 | 8 | 04cfd9814de9 |
| keyboard | tr87 | 6 | 1e3290725e1e |
| keyboard | tu93 | 9 | 39d28836c0d3 |
| click | s5i5 | 8 | 286a555ffcf6 |
| click | lp85 | 8 | 6ea2c5769562 |
| click | tn36 | 7 | 924fd7df5769 |
| mixed | lf52 | 10 | 0464f19c08fa |
| mixed | cn04 | 6 | 33332b3e96f6 |
| mixed | re86 | 8 | 3c023de8ea15 |

## Full classification (for audit)

| Game | Stratum | Rank hash (first 12) | Picked |
|---|---|---|---|
| s5i5 | click | 286a555ffcf6 | yes |
| lp85 | click | 6ea2c5769562 | yes |
| tn36 | click | 924fd7df5769 | yes |
| vc33 | click | a1c4f397d56c |  |
| r11l | click | c13eb0d217b8 |  |
| ft09 | click | f39497b3b180 |  |
| sk48 | keyboard | 04cfd9814de9 | yes |
| tr87 | keyboard | 1e3290725e1e | yes |
| tu93 | keyboard | 39d28836c0d3 | yes |
| g50t | keyboard | c048299e83fa |  |
| ls20 | keyboard | c6d6441804bf |  |
| lf52 | mixed | 0464f19c08fa | yes |
| cn04 | mixed | 33332b3e96f6 | yes |
| re86 | mixed | 3c023de8ea15 | yes |
| dc22 | mixed | 3c7df69f4bd9 |  |
| ar25 | mixed | 4a278fd82953 |  |
| sb26 | mixed | 6278be5d2e83 |  |
| sc25 | mixed | 634458782056 |  |
| ka59 | mixed | 78ed961d1774 |  |
| wa30 | mixed | 83f03a580b2e |  |
| m0r0 | mixed | 95089f153867 |  |
| cd82 | mixed | 9866a67498ed |  |
| sp80 | mixed | ace3b04bc1f8 |  |
| su15 | mixed | aef6d55f209c |  |
| bp35 | mixed | f5a07b5f9a2e |  |

## Limits

- The stratum comes from a text match on the registry. A game the match misfiled will sit in the wrong stratum. Fix a misfile only by a dated entry here, before outcomes are read.
- Nine games may be too few for the sample-size floors in plan section 6. If E6 or E4 planning shows that, add games by dated entry using the same rule (the next three per stratum by rank), never by hand.
- The registry marks these as public games with solved levels, a development proxy. Adapter-free runs must withhold per-game adapters AND per-game registry knowledge by path (see the LOO lesson in the ARC memory notes).

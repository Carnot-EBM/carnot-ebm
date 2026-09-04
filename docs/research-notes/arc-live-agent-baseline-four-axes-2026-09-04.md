# ARC live agent: a four-axis baseline, and what was changed against it — 2026-09-04

Written by a worktree agent under the 2026-09-04 brief "evaluate the live agent, then
improve efficiency, iteration velocity, accuracy and unattended self-improvement". Every
number below names the population it came from. Measured means run in this session;
inferred means derived from a recorded artifact.

## Axis 1 — efficiency

| quantity | value | population | kind |
|---|---|---|---|
| classical path, LLM off | 272 actions in 1 s (about 4 ms per action) | one r11l run, `--budget 300`, `CARNOT_ARC_DISABLE_INDUCTION=1`, this worktree, pid 3111475; the 1 s is the harness's own stdout timing line, since rows carried no `wall_s` before this session | measured (stdout) |
| whole game, LLM on | 24,998 s for 2,077 actions (12 s per action) | `results/arc_leaderboard_eval_runs/r11l-2491317.json` holds the 2,077 actions; the artifact has no duration field, so the 24,998 s is inferred from the run log and file times as cited in `post-nctx-induce-engines-are-lookup-tables-2026-09-04.md` | actions recorded; seconds inferred |
| generator share of wall clock | about 98 percent by subtraction | the two rows above | inferred |
| generator output per game | 13 chat completions, 2,467,130 reasoning chars, 37,872 final chars (98.5 percent reasoning) | same artifact, `generator_channels` | recorded |
| actions to first level-up | 815 (human baseline 22) | same artifact, `per_level` | recorded |
| per-attempt wall time | not recorded anywhere before this session | all 12 eval artifacts | measured absence |

The 12 s per action is a generator number. With the generator off, the whole 300-action
game took one second. So a 7-hour single-game eval is 13 generator calls; nothing else in
the loop costs wall clock worth measuring.

## Axis 2 — iteration velocity

| quantity | value | population |
|---|---|---|
| search-only regression check | about 6 s including startup | the LLM-off run above |
| one induce, n_ctx 98304 | about 25-30 min (50k tokens at 33-38 tok/s) | `docs/research-notes/post-nctx-induce-engines-are-lookup-tables-2026-09-04.md` |
| one deep single-game eval | 6h57m | `r11l-2491317.json` |
| observability inside a game, before this session | none: no file moved between game start and game end; the game's own wall clock reached the record only as a stdout line | source read of `scripts/arc_leaderboard_eval.py` |

Per-game banking (REQ-ARC-WMTE-6850) exists and works for multi-game sweeps. A
single-game run still banked nothing until its end. That is the shape induction work takes.

## Axis 3 — accuracy

Held-out scoring of every archived r11l engine against 119 fresh level-0 random-walk
transitions (`collect_transitions("r11l", n=120, seed=0)`, all 119 change the grid).
Read-only over `results/arc_e3/r11l/attempts/`. No `results/` artifact was written, by the
brief's rule that `results/**` is evidence. The scores live in this table and in a scratch
JSON (`~/.claude/jobs/ad0c053d/tmp/sp_afd77/r11l_engine_scores.json`, same session); the
regeneration recipe is `WorldModelVerifier(trans).score(engine)` over each attempt file,
which `scripts/arc_e3_induced_model_quality.py` already does for the retained engine. Treat
the table as unrecorded until an experiment artifact carries it.

| engine (emitted) | bytes | exact_acc | cell_recall | note |
|---|---|---|---|---|
| 2026-09-02 23:28 (pre-fix) | 7588 | 0.672 | 0.789 | best of all fifteen |
| 2026-09-04 02:05 (post-fix, engine 1) | 5046 | 0.647 | 0.699 | the "52 literal deltas" engine |
| 2026-09-04 06:24 (post-fix, engine 2) | 3193 | 0.345 | 0.362 | the RLE decoder |
| 2026-09-04 07:12 (post-fix, engine 3) | 6428 | 0.639 | 0.684 | not examined structurally |
| 2026-09-04 07:49 (post-fix, engine 4) | 1741 | 0.008 | 0.017 | |
| 2026-09-03 tool-loop engines (7) | 151-2525 | 0.0-0.244 | 0.0-0.310 | five score exactly 0 |

Two readings, stated separately. The lookup-table engines are NOT zero on unseen
transitions: engine 1 predicts the next frame exactly on 65 percent of transitions it never
saw. That is consistent with the 2026-08-27 fine-read (100 percent prefix, 50-75 percent
held-out) and with a game whose track advances deterministically, so a transcription of the
track IS a partial model. And none of them reaches the planning gate (held-out 1.0), which
is why the live run rejected them (`heldout_transition_verification_failed`). The n_ctx fix
did not raise held-out quality: the best post-fix engine scores below the best pre-fix one.

Caveats. One game, one seed, one transition population. Engines 3 and 4 were emitted after
the level-up at action 815, so they may have been induced from level-1 transitions and are
then scored out of distribution here. Engine-to-attempt attribution needs the
REQ-ARC-WMTE-6643 fields, which this run predates.

Adapter-free first-level acquisition across games stays at the 2026-08-05 measurement
(10 of 25). Counted at 21:10Z, before the coordinator's run began: 12 files in
`results/arc_leaderboard_eval_runs/` (11 complete plus one partial), covering 11 games and
banking levels on r11l, cd82, lp85, sp80, su15 only. A 13th file, `sweep-3113164.partial.json`
(explorer policy, 8 of 11 games), landed at 21:20Z and banks ls20 and tu93 at 1; it is not in
the count above.

## Axis 4 — unattended self-improvement

| quantity | before | after this session | population |
|---|---|---|---|
| refinement ledger | 9 receipts, 6 redirects, untouched since 2026-08-27 | 14 receipts, 31 redirects | `ops/arc_supervisor_refinement_ledger.json` |
| arms at the floor of 10 | none | `drop_goal_bias` 10 fired / 6 helped, `allow_reinduction` 12 / 6 | same |
| tool status | `insufficient_evidence` | `recommendation_available`: a new-arm specification from the r11l cell where every arm fired and 11 windows stayed unredirected | same |
| callers of the refine CLI | none (no timer, no dashboard hook) | unchanged | grep over scripts, ops, systemd user units |
| flag ledger | 136 flags, all `unevaluated`, 0 `off_measured` | unchanged (operator decision) | `ops/arc_flag_ledger.yaml` |

The ledger had not moved because its reader could not read its producer: the tool accepted
harness `rows.json` documents and the live eval writes `{"per_game": [...]}`. Five applied
eval rows holding 25 redirects sat unread. The re-ingest read 13 files and 26 rows: 5 applied
rows became entries, 8 rows carried an error-marker receipt (runs from before the receipt
wiring), and 13 rows had no receipt at all. Note the `helped` numbers above are the pooled,
co-crediting count; `sole=0 share=0.0` on every arm because every existing row predates
the split field.

The refine CLI has no caller. "Unattended" refinement is a tool a human remembers to run.

## What was changed

| REQ | change | proof |
|---|---|---|
| REQ-ARC-WMTE-7010 | in-run heartbeat file per game (level-ups, induce start/end, attempts, every 100 steps); rows carry `wall_s`, `generator_wall_s` | 9 tests; mutations M7-M10 |
| REQ-ARC-WMTE-7011 | every induction attempt carries `started_at` and `wall_s`; policy exposes `induction_progress_hook` | 6 tests; mutations M3-M4 |
| REQ-ARC-WMTE-7012 | the refinement tool ingests `per_game` artifacts and scans `arc_leaderboard_eval_runs/` | 6 tests; mutations M5-M6; the ledger re-ingested |
| REQ-ARC-WMTE-7013 | redirect rows carry `co_credited_count`; receipts carry `arm_credit`; the report shows sole and share | 6 tests; mutations M1-M2, M11 |

Mutation proofs: 11 of 11 RED on real assertion failures, each restored byte-identically,
scored only after every target file passed unmutated. Log:
`~/.claude/jobs/ad0c053d/tmp/sp_afd77/mutate_proofs3.log` (session scratch). The first pass
produced 11 void REDs because the harness passed `-p no:xdist`, which conflicts with the
project's pytest addopts; the baseline gate was added because of that.

## Deliberately not done

- No new solver or world-model module (live-path reachability rule).
- No re-run of the tools A/B (measured null 2026-09-03).
- No cap on induce transitions and no thinking-budget cap: both are accuracy A/Bs that need
  GPU hours, and think-off measured worse once already.
- No `--record-null` on the two tool flags: the status file lists it as an operator judgement.
- No GPU run: GPU 1 was allocated to the coordinator's eval (pid 3114878) at 21:23Z.
- No control-arm (shadow) estimator: no shadow receipt at window 120 with a firing exists yet.

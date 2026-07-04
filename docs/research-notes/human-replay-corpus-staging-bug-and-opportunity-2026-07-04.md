# Human replay corpus: a real staging bug hiding 144 winning trajectories (2026-07-04)

**Provenance:** operator asked whether human-generated ARC-AGI-3 game-event solutions were already
available to help with the TRM-as-generator proposal
(`docs/research-notes/trm-arc-action-sequence-generator-2026-07-04.md`). Investigated the existing
`data/arc_public_demo_human_replay_corpus/` asset directly (raw parquet + staged shards, not just the
capability spec's prose) before answering. Found a real, previously-uncaught staging bug and a
substantially larger training-data opportunity than either this project's TRM note or the original
frame-change-predictor task (`exp4490`) had assumed.

## What already exists (confirmed, not assumed)

`openspec/capabilities/arc-human-replay-frame-change/spec.md` documents a real capability: `exp4495`
(2026-06-20) staged a licensed (CC BY 4.0, Kaggle mirror `jihangli1121/arc-agi-3-replays-v1`) human
replay corpus for the 25 public ARC-AGI-3 games, intended to feed `exp4490` (a frame-change predictor
— a small CNN, not a TRM). The staged corpus lives at
`data/arc_public_demo_human_replay_corpus/shards/*.jsonl` (14,797 rows, schema
`carnot.arc_human_replay.frame_action_delta.v1`) and the raw HuggingFace mirror is preserved
alongside it at `raw_hf_mirror/data/*.parquet` (7 files, 340 sessions total, 144 of them wins).

## The bug: the staged shards carry zero win signal, but the raw source has it

Checked every one of the 14,797 staged rows directly: `level_progress` is `0.0` in **all of them**,
including a full 1564-step end-to-end session (`wa30`) that the raw data confirms actually won. This
is the same class of failure this project has hit before (the `n_win_states=0` corpus-artifact trap
documented in `feedback_arc_value_is_process_not_weights` / the ARC-AGI-3 foundational framing rule) —
an offline corpus silently dropping the exact signal a downstream consumer needs, then that consumer
concluding a false negative about "no win signal available."

The raw parquet source does NOT have this problem. Verified directly, cross-checked against per-step
data, not taken on faith:

- Session-level fields: `won` (bool), `levels_completed` (list), `states` (e.g. `['WIN']`),
  `actions_by_level` (per-level cumulative action-index boundaries), `total_actions`, `resets`.
- Per-trajectory-step fields (`trajectory[i].data`): `frame` (full grid), `action_input` (the exact
  action taken), `state` (`NOT_FINISHED` / `GAME_OVER` / `WIN`), `levels_completed` (a running
  counter), `win_levels` (target level count), `available_actions` (legal actions at that state),
  `guid`.
- **Validated the segmentation is real, not just self-reported metadata**: for one `sk48` session,
  `actions_by_level` claims level 1 completes at action index 37. The raw per-step trajectory
  confirms `levels_completed=0` at step 0 and flips to `levels_completed=1` exactly at step 37, held
  at step 38. The signal is internally consistent.

`exp4495`'s conversion into the `frame_action_delta.v1` schema kept `frame`/`action`/`env`/`guid` but
dropped `won`/`levels_completed`/`win_levels`/`state`/`available_actions` entirely, replacing them
with a `level_progress` field that was never actually wired to a real value. This is a narrow,
well-understood staging bug, not a source-data limitation.

## The opportunity this unlocks: 144 winning human trajectories, all 25 public games

Counted directly from the raw parquet (`won==1` sessions), per game:

| game | won sessions | total actions (won) | max level reached |
|---|---|---|---|
| ar25 | 5 | 5014 | 8 |
| bp35 | 2 | 1191 | 9 |
| cd82 | 8 | 1444 | 6 |
| cn04 | 6 | 6075 | 6 |
| dc22 | 4 | 4430 | 6 |
| ft09 | 4 | 918 | 6 |
| g50t | 2 | 1050 | 7 |
| ka59 | 3 | 2171 | 7 |
| lf52 | 4 | 5311 | 10 |
| lp85 | 15 | 6764 | 8 |
| ls20 | 6 | 4599 | 7 |
| m0r0 | 7 | 7702 | 6 |
| r11l | 10 | 2939 | 6 |
| re86 | 5 | 6070 | 8 |
| s5i5 | 4 | 2520 | 8 |
| sb26 | 5 | 1235 | 8 |
| sc25 | 10 | 3588 | 6 |
| sk48 | 7 | 7530 | 8 |
| sp80 | 2 | 913 | 6 |
| su15 | 3 | 1433 | 9 |
| tn36 | 6 | 2013 | 7 |
| tr87 | 6 | 2751 | 6 |
| tu93 | 9 | 4354 | 9 |
| vc33 | 6 | 3462 | 7 |
| wa30 | 5 | 8709 | 9 |

144 total winning sessions, roughly 90,000+ actions across them, every public game represented,
levels 6-10 reached. This directly answers (far more favorably than assumed) the open question in
the TRM-as-generator note: whether the existing trajectory corpus carries enough supervision signal
to be worth a training pilot at all. It does — provided the corpus is re-staged to actually carry the
win/level signal through.

## A real nuance for anyone building the fix or a training pipeline on top of it

Sessions are not clean single-attempt playthroughs. Checked `lp85`/`tu93`/`wa30` directly: all three
show `GAME_OVER` as an intermediate per-step state alongside `NOT_FINISHED` and the final `WIN` —
meaning the human player died and retried within the same recorded session. A session-level `resets`
field exists (values like `[0, 9]`, `[15]`, `[13]` seen across samples) that appears to mark these
retry points, though `full_reset` never showed `True` in any per-step record checked. **Before
building a training pipeline that treats a session as one continuous sequence, this needs to be
resolved precisely** — naively concatenating a full session's frames risks silently splicing a
death-and-restart discontinuity into what should be a clean win-directed sub-trajectory.

One additional minor, unresolved discrepancy: the raw `wa30` session used above has 1617 trajectory
steps, but the corresponding staged shard rows for `wa30` total 1564. Not chased further here — worth
a quick check during any re-staging work, but not blocking.

## What this note is NOT proposing

- Not proposing to skip straight to TRM training. The concrete next step is narrower and cheaper: fix
  `exp4495`'s staging conversion to preserve `won`/`levels_completed`/`win_levels`/`state`/
  `available_actions` per row (and resolve the `GAME_OVER`/retry segmentation question above) before
  any training work consumes the corpus.
- Not a claim this alone validates TRM-as-ARC-generator. It resolves the *training-data-sufficiency*
  question favorably; the architectural-adaptation question (fixed-tensor grid refinement vs.
  variable-length action-sequence refinement) from the original note is unchanged and still open.
- Not limited to the TRM proposal. Fixing this also directly unblocks `exp4490` (the original
  frame-change-predictor task), which has been sitting in a `blocked_human_replay_corpus_not_cached`
  state since 2026-06-20 04:40 — it ran roughly 1.5 hours *before* `exp4495`'s staging completed, and
  by all available evidence has never been retried in the two weeks since, despite the corpus having
  been available and CC BY 4.0-licensed the whole time.
- Solve-provenance discipline applies unchanged: any pilot built from this corpus is
  `development_proxy` work (offline, public-games-only), not a live hidden-game solve, per "ARC
  Live-Path Reachability Discipline." Any technique that graduates past piloting must be wired into
  the live agent path, never a standalone offline script.

## Cross-references

- `docs/research-notes/trm-arc-action-sequence-generator-2026-07-04.md` — the note whose open
  training-data question this substantially resolves
- `ops/verifier_gaps.md` `GAP-ARC-TRM-TRAINED-ON-ARC` — update with a pointer to this note
- `openspec/capabilities/arc-human-replay-frame-change/spec.md` — the existing capability spec this
  corpus was originally staged for
- `results/experiment_4490_human_replay_frame_change_predictor.json` — the stale, still-blocked
  original consumer task (`blocked_human_replay_corpus_not_cached`, ran before staging completed)
- `results/experiment_4495_human_replay_corpus_staging.json` — the staging task whose schema
  conversion dropped the win-segmentation fields
- `data/arc_public_demo_human_replay_corpus/manifest.json` — the staged shard manifest
- `feedback_arc_value_is_process_not_weights.md` (memory) — the `n_win_states=0` corpus-artifact
  precedent this bug matches structurally
- CLAUDE.md "ARC-AGI-3 IS a Live Hidden-Game Discovery Agent" — the foundational framing warning
  against treating an offline-corpus null as a genuine capability limit before ruling out a corpus
  artifact

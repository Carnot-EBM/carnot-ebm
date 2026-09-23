# B2 v3 induction failure triage (2026-09-23)

## What this note is

Experiment 10009 (B2 v3) measured live-path ARC world-model induction with the
mandated Qwen3.8-27B GGUF in codeonly mode. All 42 induction attempts carried a
failure tag: `proposer_failed` (23) or `exception` (19). The persisted telemetry
dropped the exception text and the proposer note, so the tags did not say what
went wrong.

This note records a read-only triage of every attempt. The triage ran as a
14-agent workflow (`wf_e034f9f2-340`, 2026-09-23):

- 6 agents each took 2 games. They replayed each attempt offline on CPU and named
  a root cause.
- 6 independent agents re-derived each cause from the raw files and tried to
  refute it.
- 1 agent measured the harness budget.
- 1 agent synthesized the results.

The outer loop then checked the five largest claims directly in source. All five
held. See "Claims checked by hand" below.

Raw evidence: `results/raw/experiment_10009_b2_induction_gate_measurement_v3/`.
Code revision at run time: commit f9383abaa0. The run used a worktree that was
removed after the run. HEAD line numbers in `arc_competition_agent.py` are about
32 lines later than at run time.

## Headline

**Every scored attempt ended through a harness mechanism. None of the telemetry
tags describes the model's output.**

Scored seeds are 7491001-7491003: 39 attempts. The other 3 attempts ran on seed
7531001, the last block of the same session, and are excluded here.

| How the attempt ended | Attempts |
|---|---|
| 240 s episode alarm (SIGALRM) fired during or before the attempt | 36 |
| think-mode refactor used all 4,096 tokens in hidden reasoning, empty content (vc33) | 3 |

How the tags were produced:

| Tag | Count | Real mechanism |
|---|---|---|
| `exception` | 17 | `EpisodeTimeout` raised in pre-dispatch code of a fallback `generate()` call (most likely the model-identity hash), escaped to the outer induce handler |
| `proposer_failed` | 19 | the alarm cancelled the in-flight repair call; the fallback `reserve()` was then refused (deadline passed or 2-request limit reached) |
| `proposer_failed` | 3 | vc33 think-mode refactor empty at the 4,096-token cap |

## What the model's first-shot output would have done on merit

The triage replayed each first-shot codeonly response against the same
acceptance checks the live proposer uses, on the observed transitions.

| Merit outcome | Attempts |
|---|---|
| engine scored and predicted wrong (rejected at the 1.0 exact held-out threshold) | 28 |
| engine raised on the observed transitions | 2 |
| a required function was missing (`is_level_complete`) | 4 |
| syntax error | 1 |
| harness-caused: response cut at the 4,096-token cap (bp35, live cap is 131,072) | 1 |
| harness-caused: sp80 level-up re-induction started after the deadline, no request sent | 3 |

So 35 of 39 first shots would not have passed on their own merits, and 0 of 36
first-shot responses would have passed. **This merit claim has no positive
control.** Nobody checked whether any engine, even a hand-written one, reaches
1.0 held-out accuracy on these 12 windows. If some windows have no headroom, the
count overstates model failure.

## B2 was not single-shot

The earlier B2 closing finding (in `ops/known-issues.md`) said the measurement was
single-shot. That was wrong.

- `python/carnot/experiment_7471_v654_arc_seam_observation.py` sets
  `CARNOT_ARC_MAX_REFINEMENT_ROUNDS=str(REQUEST_LIMIT)`, which is 2.
- The live scored default is 1 round
  (`python/carnot/agentic/arc_llm_reinduction.py`, `_max_refinement_rounds_default`,
  capped from 3 to 1 on 2026-08-17).
- In 30 attempts, round 2 was a blind think-mode refactor on the chat template at
  4,096 tokens. 0 of 30 produced code: 17 empty at the cap, 12 cancelled by the
  alarm, 1 never dispatched.

B2 therefore contains a broken instance of a multi-round arm. It is evidence
about the harness, not about refinement.

## The "multi-round refinement" follow-up rested on a misconception

The B2 closing finding named the open question as "does the live path's
multi-round refinement loop beat single-shot codeonly induction". The live path
has no multi-round loop by default. It runs one think-ON induction per stall.

Why the loop was capped at 1: extra rounds measured harmful (exp5760 mean delta
0.0; exp5766 pooled -0.0598, 0 of 83 cells improved). The refactor prompt never
shows the engine it is fixing (REQ-ARC-WMTE-6091). `CARNOT_ARC_REFACTOR_SHOW_ENGINE`
still defaults to `0`, and the Kaggle kernel does not set it. The only test of the
non-blind refactor is exp6091: show-engine 0.518, blind 0.059, single-shot 0.437,
verdict `blocked_underpowered` on 11 gradeable games.

## This harness cannot measure think-ON induction at all

| Budget item | This harness | Live scored path |
|---|---|---|
| episode wall time | 240 s | induce timeout 2,400 s (`scripts/kaggle/submission_kernel/main.py`) |
| tokens per induction call | 4,096 | 131,072 |
| requests per episode | 2 | not limited to 2 |
| model-identity hashing before each call | about 39.5 s (5 SHA-256 passes over 17.1 GB) | cheap path-equality branch on Kaggle |

One median think-ON Qwen3.8 induction is about 62,490 tokens. At the measured
37-40 tokens per second that is about 1,600-1,730 s. A think-ON arm in this harness
would be censored in every case, by construction. That guarantees a false
negative.

## Ranked fixes

| Rank | Fix | Attempts it unblocks | Live scored path? |
|---|---|---|---|
| 1 | Size the harness budget to the work: no 240 s alarm during induction, request limit of at least rounds plus fallbacks, per-call token cap the mode can finish in | 39 | no |
| 2 | Make round 2 a real repair or remove it. Use the live default of 1 round, or route the refactor through the codeonly seam, and disclose the choice | 30 | no |
| 3 | Memoize the model-identity SHA-256 per (path, size, mtime, inode); keep the receipt fail-closed | 15 | yes (local runs only; no speedup on Kaggle) |
| 4 | Raise the request limit to at least 3 so the split-induce fallback can finish (covered by fix 1) | 3 | no |
| 5 | Live agent: do not re-induce after a level-up on 1 transition; check that the "board at the start of the current level" block is the new level's first frame (plausible, not proven) | 3 | yes |
| 6 | Run with tries > 1 so the dry-run defect gate can re-ask when an engine raises (live default is already 3) | 2 | no |
| 7 | Show the engine in the refactor prompt (`CARNOT_ARC_REFACTOR_SHOW_ENGINE=1`); required before any repair result means anything | 0 | yes |
| 8 | Make `EpisodeTimeout` a `BaseException` subclass so a fired alarm records `censored_timeout` and stops the episode | 0 | no |
| 9 | Persist per-attempt failure detail: exception repr and traceback, proposer note, per-round rows, budget refusals, per-call tokens | 0 | yes (telemetry module) |
| 10 | Fix the harness level reader to use `FrameData.levels_completed`, and fix its unit test | 0 | no |

Unverified, reported by one verifier only: the plain single-shot fallback may build
its prompt from all 25 rows, including the 8 held-out rows. Check before acting.

## Record defects found

1. **Level reader.** `python/carnot/experiment_7431_v651_arc_live_sentinel.py`
   `_level()` reads a `levels` attribute that `FrameData` does not have. The real
   field is `levels_completed`. So every level field is 0. An offline replay of
   the recorded action traces shows vc33 and sp80 reached level 1 in all 3 seeds.
   The unit test fakes a `levels` list, so it passes. Several later experiment
   modules carry a function of the same name; check which use this reader.
2. **Swallowed timeout.** `EpisodeTimeout` subclasses `Exception`, so the agent
   swallows it. All 39 started episodes read `disposition: complete` and
   `error: null`. The alarm fired in 33. 18 ran past the cap, up to 407.8 s.
3. **Dropped failure detail.** The agent computes the exception repr, traceback,
   proposer note, and per-round refinement rows. The telemetry row writer drops
   them. Every cause above had to be inferred from timing and orphan staging
   directories.
4. **Last-stage tag.** The `skipped` tag records only the last stage (the fallback
   after the deadline). It overwrites the refinement outcome.
5. **Token fields.** 23 of 42 `completion_tokens` values are refactor usage or a
   stale copy, not induction output.
6. **Effective sample.** The first-call induction prompt is byte-identical across
   the 3 seeds for all 12 games. The effective sample is 12 evidence windows, not
   36.
7. **Starved episodes.** About 39.5 s of hashing per call plus overrun episodes
   left 105 of 144 scheduled episodes unstarted. That is why attempt counts never
   reached the 100-attempt floor.

## A measurable design for the refinement question

1. Separate induction quality from episode timing. Replay the 12 staged windows
   (17 visible plus 8 held-out rows) offline against the same `WorldModelVerifier`.
   Each window is exactly what the live agent saw.
2. Three arms: codeonly single-shot (0 of 30 scored engines passed), live-default
   think-ON with 1 round, think-ON with 2-3 rounds and `SHOW_ENGINE=1`.
3. Give each call the live budget (131,072 tokens, 2,400 s). Memoize the identity
   hash. Persist per-round telemetry.
4. Use a graded primary metric, such as changed-row accuracy or cell recall. At
   the 1.0 exact threshold nearly every cell is 0.
5. Add windows to reach at least 30 cells. Twelve windows repeats exp6091's
   underpowered result.
6. GPU time: about 17 h per think-ON single-round arm over 36 cells, and 2-3 times
   that for multi-round. Pilot first on the 12 windows with one draw per arm.

## Claims checked by hand (outer loop, 2026-09-23)

| Claim | Source checked | Result |
|---|---|---|
| harness caps are 240 s and 2 requests | `experiment_7471_v654_arc_seam_observation.py` `EPISODE_LIMIT_S`, `REQUEST_LIMIT` | confirmed |
| harness overrides refinement rounds to 2 | same file, `CARNOT_ARC_MAX_REFINEMENT_ROUNDS: str(REQUEST_LIMIT)` | confirmed |
| live refinement default is 1 | `arc_llm_reinduction.py` `_max_refinement_rounds_default` | confirmed |
| level reader reads a missing field | `experiment_7431_v651_arc_live_sentinel.py` `_level`; arcengine `enums.py` defines `levels_completed` only | confirmed |
| `EpisodeTimeout` subclasses `Exception` | `experiment_7491_e6_timed_live_profile.py` | confirmed |
| live induce budget is 131,072 tokens and 2,400 s | `scripts/kaggle/submission_kernel/main.py` | confirmed |

## Not checked

- No positive control for the 1.0 held-out threshold on these windows.
- The 3 seed-7531001 attempts were not triaged in depth.
- Think-mode first-shot induction on these windows was never run.
- The plain-fallback held-out leak, the vc33 board-label claim, and a
  repeat-penalty early-stop idea are untested.
- 8 pre-dispatch gaps over 44 s (max 89 s) are unexplained.
- Earlier records that share the level reader or the timeout class (7431, 7457,
  7471, 7491, 7531, 10008) were not re-audited.

## Cross-references

- `results/experiment_10009_b2_induction_gate_measurement_v3.json` (corrigendum
  field `corrigendum_2026_09_23_failure_triage`)
- `ops/known-issues.md`, B2 entry, correction dated 2026-09-23
- Workflow journal: session workflow `wf_e034f9f2-340` (outer-loop scratch, not
  committed)

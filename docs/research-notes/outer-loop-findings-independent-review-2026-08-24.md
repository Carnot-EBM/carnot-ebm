# Independent adversarial review of the outer loop's 21-hour session findings

**Date:** 2026-08-24
**Reviewer:** independent agent, no part in the reviewed work
**Method:** verified against code and artifacts only. The outer loop's summaries
were treated as claims to test, not as evidence.

The outer loop asked for refutation by default, after four other agents found
errors in its reports. This note gives a verdict on each claim with the evidence.
Most claims hold. Two do not. One prediction is unsound and will produce a
confusing result if run as written.

---

## 1. The unwrap helper: blast radius

The helper in question:

```python
g = lambda k: (d.get(k).get('value') if isinstance(d.get(k), dict) else d.get(k))
```

It assumes every dict-valued field is a principle wrapper. For a plain dict it
returns `None`, so the field reads as ABSENT.

**Method.** Parsed every top-level key of all `results/*.json`. 6906 artifacts
parsed, 4 unreadable (oversized or malformed). For each key, counted how often it
is a dict WITH `value` (helper works), a dict WITHOUT `value` (helper corrupts),
or non-dict (helper passes through).

### The fields the outer loop named

| Field | wrapped | **CORRUPT** | plain |
|---|---|---|---|
| `honest_verdict` | 155 | **10** | 5514 |
| `flagged_adversarial` | 0 | **0** | 645 |
| `verdict_class` | 0 | **0** | 60 |
| `inference_substrate` | 168 | **170** | 2895 |
| `duration_s` | 14 | **0** | 4380 |
| `levels` | 0 | **0** | 2 |
| `preconditions_checked` | 44 | **1271** | 880 |
| `model_specs` | 9 | **467** | 357 |
| `offline_reproduced` | 0 | **17** | 254 |
| `random_seed` | 28 | **19** | 2804 |
| `verifier_is_oracle` | 38 | **6** | 966 |
| `acceptance_gate_passed` | 1 | **0** | 172 |

Worst corpus-wide, beyond the named set: `field_principles` 2248,
`field_provenance` 596, `protected_files_unchanged` 363, `test_exit_codes` 277,
`source_checksums` 137, `artifact_metadata` 121, `reproduction_gate` 117.

### The answer to "was quarantine status misreported"

**No. REFUTED, and cleanly.** `flagged_adversarial` is **never** dict-valued
anywhere in the corpus — 0 of 6906, 645 plain values, the rest absent. The helper
cannot corrupt it. The same holds for `verdict_class` (0 of 60) and
`acceptance_gate_passed` (0 of 173).

`honest_verdict` is corrupt in 10 artifacts, all old and all structurally odd —
e.g. `experiment_1798_retro.json`, where `honest_verdict` is a per-file map of
verdicts, and `experiment_256_results.json`, where it is a findings dict. None is
a recent artifact.

### The blast radius in the window that matters

Restricting to the 307 artifacts touched since 2026-08-23 — the population the
30-minute health checks read:

**Exactly one field was ever corrupted: `preconditions_checked`, in 40
artifacts.** Every other named field was a plain scalar in every recent artifact.

Of those 40, only 4 carry `all_required_preconditions_available` at all, and only
**2** have it `False` — exp6581 and exp6582, the two already found. The remaining
36 have no availability key, so no status was inverted by reading them.

**Verdict: CONFIRMED but tightly bounded.** The bug is real and the mechanism is
exactly as described. Its consequences are confined to the one field and, within
the operating window, to the two artifacts already corrected. The worry that
quarantine status was systematically wrong is refuted. The corpus-wide
`inference_substrate` (170) and `model_specs` (467) corruption is real but sits in
older artifacts (exp3026 era) that the session did not read.

---

## 2. Claim: every `run_tests` call site passes `full=False`

**CONFIRMED**, verified independently, with one near-miss and one refinement.

`scripts/research_conductor.py:1467` defines `run_tests(full: bool = False)`.
Call sites: `:6125` and `:6181` bare, `:6671` and `:6798` explicit `full=False`.

- **Near-miss:** `scripts/pretest_triage.py:58` reads `return run_tests(full=full)`
  and passes `full` through. It is **inside the module docstring** (lines 1–68,
  the "HOW TO WIRE THIS IN" block), not live code. `pretest_triage` is imported
  only by its own test — never by the conductor. The documented wiring was never
  applied. The claim survives.
- No indirection: no `getattr`, `partial`, or `globals()` reference to
  `run_tests` exists outside the autoresearch constitution's allow-list string.
- `full=True` appears nowhere in the repo except the docstring describing the
  parameter.

**Refinement.** The repo does contain full-suite invocations — `.venv/bin/pytest
tests/python -q` inside individual experiment scripts (exp2910, exp3002, and
until today exp6581/6582). So "no full-suite reporter" is true of the *conductor*
path, which is what matters, but the phrasing hides that experiments were running
it as verification commands. That is precisely how exp6581/6582 came to be
blocked by it.

### A finding the outer loop did not look for

`ops/known-issues.md` step 3 proposes to "revive the conductor's `full=True`
branch". **As written, that will not work.** The `full=True` branch sets
`timeout=600` (`research_conductor.py:1525-1538`). The same entry's step 1 states
a clean `--no-cov -o addopts= -n 0` run "takes 18+ minutes" — 1080s, comfortably
past the 600s timeout. Revived unchanged, `full=True` returns
`(False, "Command timed out")` on every call, which is a *worse* state than dead
code: a guard that is trusted and silently always-red.

So `full=True` is not merely uncalled. It is non-viable at its current timeout.
Raising it is a precondition of step 3, not an afterthought.

---

## 3. The seed-sweep conclusion

The outer loop's stated inference: the variance was unseeded sampler
nondeterminism, evidenced by the S arm producing `act=389` across all three tu93
seeds while `CARNOT_ARC_GENERATOR_SEED` was unset.

**The conclusion is probably right. The inference given for it is unsound, and
the cited evidence supports a different mechanism.**

### What the code says (CONFIRMED)

`LocalGGUFProposer.sampling_seed()` (`arc_executable_world_model.py:6245`)
returns `None` when `CARNOT_ARC_GENERATOR_SEED` is unset, and the caller then
omits the `seed` field entirely (`:7705`). `llama-server` treats an absent seed
as -1, i.e. fresh random. Temperature is `0.2 + 0.1*attempt`, nonzero. The
harness `--seeds` reaches only `random.seed()` / `np.random.seed()` in the driver
(`arc_scored_path_lever_harness.py:645-646`).

So both of the outer loop's candidate explanations are **simultaneously true and
not competing**: the sampler was unseeded, AND `--seeds` did not reach it. They
are the same fact seen from two ends.

### Why the cited evidence does not support the conclusion

An unseeded sampler predicts **variance**. The S arm shows **none**:

| game | arm | seed | actions | levels | resets | level-up at |
|---|---|---|---|---|---|---|
| tu93 | S | 20260724 | 389 | 1 | 11 | [345] |
| tu93 | S | 20260725 | 389 | 1 | 11 | [345] |
| tu93 | S | 20260726 | 389 | 1 | 11 | [345] |
| tu93 | S_replicate | 20260724 | 389 | 1 | 11 | [345] |
| tu93 | S_replicate | 20260725 | 364 | 2 | 36 | [43, 230] |
| tu93 | S_replicate | 20260726 | 365 | 2 | 35 | [43, 203] |

The three S runs are identical not just in action count but in **reset count and
level-up index**. That is a deterministic trajectory. Yet their LLM token counts
differ by 40% (200721 / 286585 / 280006) and `llm_on_row_valid` is `True` on
every row, with 2–3 induce calls reaching the LLM each time.

So the sampler *was* nondeterministic and the trajectory *was* deterministic. The
resolution is in the induction bookkeeping: in all three S runs every induced
world model was rejected (`world_model_accuracy_below_threshold`,
`no_reachable_plan_after_refinement`, `degenerate_goal_predicate`), so the agent
fell back to a deterministic non-LLM policy — the same 389-action trajectory
every time. In the two divergent S_replicate runs induction succeeded
(`induction_planned` 3 and 2) and the agent leveled up at action 43 instead of
345.

**Therefore:** `act=389 × 3` is evidence that **the LLM's output was being
discarded**, not evidence about seeding. The genuine evidence for sampler
nondeterminism is the *other* arm — S_replicate diverging from S under a config
the harness asserts is byte-identical.

### One suspicion I raised and had to drop

The S/S_replicate split is asymmetric: mode-B (level-up at 43) appears 0/3 in S
and 2/3 in S_replicate. I checked whether the arms actually differ.
`arc_scored_path_lever_harness.py:215` carries an import-time assertion:

```python
assert ARMS["S_replicate"] == ARMS["S"], (
    "S_replicate must be byte-identical to S -- it is the same-config noise floor, not a treatment"
)
```

The arms are identical by construction. 0/3 vs 2/3 under an identical config is
Fisher p = 0.4 — entirely consistent with chance. **No finding here.**

**Verdict: CONFIRMED on the conclusion, REFUTED on the reasoning.** Also note the
sweep is not a clean measurement of anything else: `budget=400` and the S arm
lands at 384–390, so the action metric is saturating against its own ceiling in
most cells.

---

## 4. The retirement of exp6582

**The retirement is procedurally valid, substantively wrong, and now stale.**

### What is confirmed

The third correction in `ops/exclusion_manifest.yaml` is accurate on every point
I could check:

- `preconditions_checked` is a 29-key plain dict in both artifacts.
- `all_required_preconditions_available = False`; `failed_preconditions` is
  `['verification_commands']` for exp6581 and
  `['idle_supported_gpu', 'verification_commands']` for exp6582.
- `model_process_started = False` — no model was loaded in either run.
- The pytest share of runtime is **99.8%** (exp6581, 925.0s of 927.0s) and
  **99.9%** (exp6582, 2409.1s of 2411.0s). The figure is exact.
- The check was `FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"`,
  removed in `696cb32224`.

### Where the diagnosis is wrong

The manifest states the check "reports a TRUE fact" because the suite is red. In
**both on-disk artifacts the full-pytest row carries `exit_code = 124` — a
timeout**, at 900s for exp6581 and 2400s for exp6582. Neither run observed a red
suite. They observed a suite that did not finish.

The red-suite claim itself is separately sourced and I am not disputing it —
`ops/known-issues.md` cites exp6582's *first* run (`b40a131bfd`) reporting
`87 failed, 10326 passed`, plus exp6510, exp6403, exp6426. That entry is careful
and states its own coverage/xdist caveat.

But the two artifacts the retirement rests on evidence a **timeout**, not a red
result. Those are different failure modes with different fixes, and the manifest's
phrasing implies the artifacts show the latter.

### Why the retirement is substantively wrong

exp6582 was retired on `retire_if_same_verdict`, two runs, same verdict. Both
blocking preconditions are environmental and neither is a property of the
experiment:

1. `verification_commands` — a misplaced precondition gating a model load on
   repo-wide suite health, which timed out. Demoted and repaired in `696cb32224`.
2. `idle_supported_gpu` — failed on a contended box. The manifest itself records
   that the outer loop's own sweep held GPU 1 for 19+ hours. Self-inflicted.

CLAUDE.md's Pre-Launch Preconditions Discipline states the intent directly:

> The conductor's reconciler classifies `blocked_*` verdicts as honest
> non-terminal states (NOT as fabrications or partial failures), so the task
> simply retires without burning the doomed-rerun ledger unnecessarily.

Two `blocked_precondition_failed_*` verdicts feeding `retire_if_same_verdict` is
contrary to that intent. An experiment retired because its harness was misbuilt
and its GPU was occupied has not been shown to be doomed.

### Why it is stale

The manifest's own reopen condition is: *"Reopen once someone diagnoses why the
quality precondition fails after a successful load, and states what changed."*
The cause is now diagnosed and the fix is committed. The condition is met.
`exp6582` is still present in `scripts/conductor_exclusion_manifest.json` and
still blocking.

**Recommendation:** reopen exp6582, with the diagnosis and `696cb32224` as the
"what changed". Leaving it retired records "this experiment is doomed" when the
record shows "the harness was misbuilt and the box was busy". Note
`operator_reopen_required: true`, so this is an operator call, not an agent one.

The append-only correction discipline in this manifest entry is, separately,
exemplary — three corrections stacked with the wrong text preserved verbatim.

---

## 5. The scoreboard

**CONFIRMED on the numbers. REFUTED on stability, and the delta is thinner than
it reads.**

Reconstructed from source: `baseline25/rows_first10.json` (10 games, 7 levels) +
`baseline25/rows_rest15.json` (15 games, 9 levels), merged by
`arc_scoreboard.py`'s group-key rule:

- 25 games, **16 levels** — matches.
- mean `efficiency_gateway_charged` = **0.1144** — matches to four decimals.
- 46922 actions across 25 rows.

### The instability

`arc_scoreboard.py:139` globs `rows*.json` and takes the **mtime-newest** run.
When the seed sweep wrote `seedsweep/rows.json` at 08-24 17:59Z, the headline
silently re-pointed. Run now, the same command prints:

```
levels solved           3   across 2 games
efficiency         1.0213   (gateway-charged, mean of 2 rows)
source             seedsweep/rows.json  (08-24 17:59Z)
```

"16 levels / 25 games / 0.1144" and "3 levels / 2 games / 1.0213" are the same
command on different days. Quoting either as "the scoreboard" without naming the
source is how a 25-game baseline gets compared to a 2-game sweep.

### The 24h delta

The tool computes the delta over **shared games only** and flags thin sets. Its
current output says `+0 levels over 2 shared game(s) [thin: <3 games]`.

For the baseline25 headline the outer loop was quoting, the runs inside the
preceding 24h window were `fixrun` (5 games, 5 shared), `supab2_VOID_oom` (3
games, 3 shared) and `supab3` (3 games, 3 shared). So "+0 over 24h" was computed
against **at most 5 shared games**, not 25 — sitting directly beside a headline
that says 25. Comparing shared games only is the right method; presenting its
result next to a 25-game headline without the shared count is what misleads.

To the literal question — is the 24h delta comparing comparable runs? **No.** It
compares the comparable subset, which is a fifth of the headline population, and
the outer loop reported the delta without the caveat the tool itself prints.

---

## MISSED INPUT

Constructed inputs no current check catches. Each is simultaneously a fix and its
regression test.

1. **A guard reading `preconditions_checked` through a principle-unwrap helper.**
   Input: any of the 1271 corpus artifacts where the field is a plain dict, e.g.
   `results/experiment_6581_qwen36_flagship_source_shard.json`. Expected: the
   reader sees `all_required_preconditions_available = False`. Actual under the
   helper: field absent. A shared unwrap helper should return the dict unchanged
   when `'value'` is absent, rather than `None`. The one-line rule: a principle
   wrapper has `'principle'` in it — test for that, not for dict-ness.

2. **`run_tests(full=True)` against a suite needing more than 600s.** Input: the
   current repo. Expected: a pass/fail verdict. Actual: `(False, "Command timed
   out")` at `research_conductor.py` `timeout=600`, indistinguishable from a red
   suite. No test covers the `full=True` branch's timeout budget.

3. **`retire_if_same_verdict` fed by two `blocked_*` verdicts.** Input: exp6582's
   two runs, both `blocked_precondition_failed_without_quality_claim`, both from
   environmental preconditions. Expected per CLAUDE.md: no ledger burn. Actual:
   permanent retirement. A `blocked_*` prefix should be excluded from
   `retire_if_same_verdict`, or require that the failed precondition be a property
   of the experiment rather than of the box.

4. **A precondition row with `exit_code = 124` read as a substantive failure.**
   Input: exp6581/exp6582 `tests_run`. A timeout and a red result need different
   handling; `focused_verification_ok` currently maps both to `False` via
   `exit_code == 0`. Fail-closed is right, but the *reason* is lost, and that lost
   distinction is what produced a manifest claiming the suite was red when it had
   only timed out.

5. **A scoreboard headline quoted without its source and shared-game count.**
   Input: `arc_scoreboard.py` run before and after any 2-game sweep lands. The
   tool already prints both facts; nothing requires a caller to carry them.

---

## What I could not check

- **Whether the outer loop's ~20 reports individually stated the corrupted
  values.** I verified which fields *could* have been corrupted in the artifacts
  available to it. I did not have the report texts, so per-report damage is
  bounded by field, not enumerated by report.
- **Whether the health checks ever read pre-2026-08 artifacts.** If they did, the
  `inference_substrate` (170) and `model_specs` (467) corruption becomes live.
  Everything in the recent window is clean apart from `preconditions_checked`.
- **The queued seeded rerun's actual outcome.** It had not started; GPU 1 was
  still held by the first sweep. I launched no GPU work.
- **4 artifacts** of 6910 were unparseable (oversized or malformed) and are
  excluded from every count above.
- **exp6582's first run** (`b40a131bfd`), the one that reported 87 failures. Its
  artifact has been overwritten by the second run. I read the counts from
  `ops/known-issues.md`, not from the artifact.

---

## Claim 6, unnumbered: the queued seeded rerun's prediction

`$CLAUDE_JOB_DIR/tmp/seedsweep2/run.sh` predicts: with
`CARNOT_ARC_GENERATOR_SEED=20260724` set, "every pair should tie EXACTLY".

**The prediction is unsound as stated, and the run will be hard to interpret.**

1. **It is already true in the seed-OFF baseline for the arm that matters.** S
   ties itself 3/3 and ties S_replicate at seed 20260724 — unseeded. A tie is not
   diagnostic of seeding when the null already produces ties.
2. **A seed does not make two runs converge; it makes one run reproducible.**
   `sampling_seed(attempt) = base*1000 + attempt`. In the observed data the arms
   made different numbers of induce calls (`generate_calls` 3 vs 4, 2 vs 3) and
   reached different attempt counts (2 vs 3). Once two runs differ in *how many*
   attempts they take, attempt N in one arm is not attempt N in the other, the
   seeds desynchronize, and nothing pulls them back together. Seeding is not a
   contraction.
3. **`cache_prompt: True` is in the payload** (`:7659`) and the script runs both
   arms against one server on one port with `--no-spawn`. KV-cache state at
   request start differs between arms. A seed pins the sampler RNG, not the
   cache-hit boundary.

So a divergence would *not* establish "a second nondeterminism source" as the
script claims — items 2 and 3 are sufficient to explain one, and both are
consequences of the design rather than discoveries. And a tie would not establish
that seeding worked, per item 1.

The run is cheap and worth doing as an A/A control. The prediction attached to it
should be weakened before its result is read: the informative comparison is
whether **S_replicate's divergence rate drops** relative to the seed-OFF
baseline, over more than three cells per arm — not whether any single pair ties.

---

## Summary

| # | Claim | Verdict |
|---|---|---|
| 1 | Unwrap helper corrupted other fields | **CONFIRMED, bounded** — 1 field, 40 recent artifacts, 2 with inverted status. `flagged_adversarial` never affected |
| 2 | No full-suite reporter; all call sites `full=False` | **CONFIRMED** — plus `full=True` is non-viable at 600s, not merely uncalled |
| 3 | Seed-sweep variance was unseeded sampling | **CONFIRMED conclusion, REFUTED reasoning** — `389×3` evidences discarded LLM output, not seeding |
| 4 | exp6582's retirement | **REFUTED as justified** — both blocks environmental; reopen condition now met; manifest wrongly says the artifacts show a red suite (they show `exit 124`, timeout) |
| 5 | Scoreboard 16/25/0.1144, +0 over 24h | **Numbers CONFIRMED exactly. Stability and delta framing REFUTED** — source is mtime-newest and re-points; delta covers ≤5 shared games, not 25 |
| 6 | Seeded rerun ties every pair | **REFUTED as a prediction** — already true under the null; seeding is not a contraction |

The outer loop's self-assessment was harsher than the evidence warrants. The
unwrap bug is real but its blast radius is one field, and the specific fear —
misreported quarantine status — is refuted outright. The two genuine problems are
that exp6582 is retired for its environment's faults and should be reopened, and
that a scoreboard whose source silently swaps was quoted about twenty times
without naming it.

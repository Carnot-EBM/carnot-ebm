# Phase 1 (confirm) harness — does `repeat_penalty` + a plain re-ask move the funnel?

Run order. Steps 1 and 2 are offline and cost seconds; step 3 is the only one that needs a GPU,
and it refuses to start if step 2 could not prove the split.

```bash
V=/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/.venv/bin/python
H=results/arc_induce_confirm_20260731/harness

# 1. capture each game's REAL induce prompt + the three-level transition split, LLM-OFF
for g in ft09 tu93 tn36 lp85 sc25 vc33; do CAP_GAME=$g $V $H/capture_p3.py; done

# 2. derive and PROVE the in-sample / held-out split (writes ../split.json)
$V $H/split.py

# 3. the live measurement -- two arms, ONE server process per GPU, attempt-major across games
CONFIRM_GAMES=ft09,tn36,vc33 CONFIRM_PORT=8951 CONFIRM_GPU=0 CONFIRM_TAG=gpu0 $V $H/confirm_ab.py &
CONFIRM_GAMES=tu93,lp85,sc25 CONFIRM_PORT=8952 CONFIRM_GPU=1 CONFIRM_TAG=gpu1 $V $H/confirm_ab.py &
wait

# 4. score OUT-OF-SAMPLE + compute the verdict (writes ../confirm_scored.json)
$V $H/score.py
```

## What each step is for

| file | what it does |
|---|---|
| `capture_p3.py` | Replays the agent LLM-OFF against the offline arcade with a proposer that RECORDS its `induce` arguments. Writes the real induce prompt, the transitions `induce()` received, and — new in this phase — the FULL list before `_proposal_prefix` dropped its suffix. |
| `split.py` | Derives `shown` (the ≤8 rows the prompt renders) and `HELD_OUT = full \ shown`, then PROVES it against the prompt text. Importable: `confirm_ab.py` and `score.py` both call `load_split`, so there is exactly one definition of "held out". |
| `confirm_ab.py` | The live A/B. CONTROL = shipped sampler + shipped accept-first. TREATMENT = `repeat_penalty 1.1` + one plain re-ask on a mechanical defect. Treatment round 0 is recorded separately as the `repeat_penalty`-only sub-arm. |
| `score.py` | Runs the shipped `WorldModelVerifier` over every completion on the held-out rows and on the shown rows, then computes the funnel and the attempt-matched sign tests. |

## The three properties this harness was built to have

**The split is proven, not asserted.** `split.py` derives `shown` by replicating
`_transitions_block`'s selection rule and then checks each row against the prompt TEXT by a
different computation. A row whose rendered line is ambiguous (a duplicate of a shown row) is
DROPPED from held-out rather than scored as unseen.

**No held-out row informs any decision the pipeline makes.** The treatment's dry run, defect check
and non-inert check see only `shown`. Held-out rows are opened for the first time by `score.py`.

**A wall-clock cutoff still leaves a reportable grid.** The loop is attempt-major across games, so
every game has the same number of attempts at any point, and partial JSON is rewritten after every
completed cell (one game's control + treatment round 0 + any re-ask — the smallest unit that can
be scored as a pair). An earlier probe in this repo died at 3.2h with 4 of 12 cells never launched.

## Things that will bite you

* **The sampler seed does not cross server processes.** Both arms must run through one server; two
  processes with identical config give different output.
* **Run from the canonical repo path** (`/home/ianblenke/github.com/ianblenke/carnot`), never a
  `/tmp` worktree or `$PWD` — a worktree run once produced a phantom 5-action regression that
  survived 6 replicates.
* **`n_ctx` must be pinned to 32768.** The shipped `_default_induce_n_ctx()` is 81920, which does
  not fit a 24 GiB card; the 31B then silently binds the iGPU HIP build and runs LLM-OFF while
  REPORTING LLM-ON. `confirm_ab.py` proves the CUDA build from `/proc/<pid>/exe` plus a per-PID
  VRAM row and refuses to run otherwise.
* **Pin a non-default port.** 8919 is the default and a stale server there is silently adopted.
* `results/arc_e3`, `results/arc_logo_snapshot` and `results/arc_e3_origin_fixtures` are EVIDENCE.
  Every script here redirects `CARNOT_ARC_E3_DIR` into its own scratch directory.

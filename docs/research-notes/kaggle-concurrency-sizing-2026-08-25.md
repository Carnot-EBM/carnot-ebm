# Concurrency sizing for the scored card (2026-08-25)

Written for the planner and for any agent sizing the live generator. The numbers
below are measured on this box unless marked ESTIMATE. Memory files are not readable
from the conductor's subprocesses, so this lives in the repo.

## The one-line answer

Concurrency is the only speed lever we have. Per-stream throughput does not change
with K. Raising K raises aggregate throughput almost linearly, and the 96 GiB scored
card fits K=16 with room to spare.

## Measured inputs

| Quantity | Value | How |
|---|---|---|
| KV cost | **36.11 KiB/token** | the server's own `prompt_save` lines, 12 samples, range 35.66-36.66 |
| prompt tokens | median 9,051, max 10,799 | completion log, both runs, identical |
| decode tokens | median 65,507, cap 84,144 | completion log, `eval time` lines only |
| per-stream throughput | ~30-31 tok/s | live run |
| weights | ~16.1 GiB | ESTIMATE for Q4_K_M 27B, not measured |

Do NOT derive the KV cost by subtracting an assumed weights size from `nvidia-smi`.
That method gave 45.55 KiB/token here and was wrong by 26 percent.

**Exclude `prompt eval time` lines when measuring decode.** A grep for
`eval time = ... / N tokens` also matches inside `prompt eval time`, which pools 36
prompt measurements with 36 completion measurements and halves the median. That error
produced a published figure of 40k tokens per call when the real median is 65k.

## The admission rule

From `_default_induce_n_ctx`:

```
n_ctx >= K_concurrent * (prompt_tokens + max_tokens)
```

It reserves against `max_tokens`, the CAP, not the observed median. Per stream today
that is 9,051 + 84,144 = **93,195 tokens**.

## What fits on 96 GiB

| K | KV at cap | plus weights | verdict |
|---|---|---|---|
| 8 | 25.7 GiB | 41.8 GiB | fits |
| 16 | 51.3 GiB | **67.4 GiB** | fits, uses 64 percent of the KV budget |
| 24 | 77.0 GiB | 93.1 GiB | fits on 2.9 GiB margin |
| 32 | 102.7 GiB | 118.8 GiB | does not fit |

K=16 is the safe default. K=24 fits, but its margin is smaller than the error bar on
the weights estimate, so pin the weights figure before choosing it.

## What that buys

| setup | aggregate | inductions in 12h | per game, 25 games |
|---|---|---|---|
| local, 1 stream | 30 tok/s | 20 | 0.8 |
| K=16 | ~500 tok/s | 330 | 13.2 |
| K=32 | ~1000 tok/s | 660 | 26.4 |

A local single-stream run is not a slow version of the scored eval. At 0.8 inductions
per game it is a different activity, and results from it do not transfer.

## How to raise K

Two changes are required together.

1. `CARNOT_ARC_LLAMA_SERVER_SLOTS` (REQ-ARC-WMTE-6253) sizes the pool ARITHMETIC only.
   It does not add `--parallel` to the launch.
2. The launch must pass `--parallel K`. An explicit `--parallel` DIVIDES the pool per
   slot, so `-c` must rise to `K * per-stream` at the same time.

Today the server runs `n_parallel=4` auto with `kv_unified=true` and `-c 106496`.

## Token reduction: what is built, and what it can do

Every token-reduction feature is default-off and none is set on any live run:
`CARNOT_ARC_INDUCE_TOOL_LOOP`, `CARNOT_ARC_INDUCE_TOOL_COMPACT`,
`CARNOT_ARC_INDUCE_TOOL_THINK_BUDGET`, `CARNOT_ARC_PLAYBOOK_RETRIEVAL`,
`CARNOT_ARC_MATM_SIMILARITY_RETRIEVAL`. Prompt size is byte-identical between the
2026-08-24 seed sweep and the live supervisor run, so nothing compacts anything today.

Decode is 88 percent of the per-stream footprint. Prompt is 12 percent. So prompt
compaction alone cannot reach K=32. The think budget is the only lever that moves the
binding number, and `exp6199` measured think-off as WORSE for induction quality
(held-out accuracy 0.100 against 0.3727). That is a real trade, not a free win.

## Not measured

- Whether aggregate throughput scales linearly to K=16 on that card. Memory bandwidth
  may cap it below 500 tok/s.
- The weights size. 16.1 GiB is an estimate and shifts every total by about a GiB.
- What compaction would actually save, since it has never been enabled on a live run.

## Cross-references

- `docs/research-notes/induction-token-accounting-and-rejection-2026-08-25.md` — where
  the 97.6 percent discard figure and the reasoning-channel explanation come from
- `docs/research-notes/induction-rejection-adversarial-read-2026-08-25.md`
- `python/carnot/agentic/arc_executable_world_model.py` — `_default_induce_n_ctx`,
  the slot knob, and the `--parallel` warning

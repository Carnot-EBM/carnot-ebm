# Concurrency sizing for the scored card (2026-08-25)

Written for the planner and for any agent sizing the live generator. The numbers
below are measured on this box unless marked ESTIMATE. Memory files are not readable
from the conductor's subprocesses, so this lives in the repo.

## The one-line answer

Concurrency is the only speed lever we have. Per-stream throughput does not change
with K. Raising K raises aggregate throughput almost linearly, and the 96 GiB scored
card fits K=16 with room to spare.

**CORRECTION 2026-08-25 (same day, measured).** The second and third sentences are
wrong. Both "Not measured" gaps below are now closed, and each one moved the answer.

- Per-stream throughput FALLS hard with K. It does not hold steady.
- Aggregate throughput reaches about 4x at K=16, not 16x. Half of that gain is
  already there at K=6, and the K=6 to K=10 band is unstable.
- This model is a HYBRID. It carries a fixed 149.6 MiB of recurrent state PER SLOT
  that no term in this note accounted for.
- K=24 does NOT fit. It overshoots the card by 6.3 GiB.

The first sentence stands: concurrency is still the only speed lever. **K=12 is the
new recommendation.** It wins on memory and on throughput at the same time. Read
"The measured VRAM model" and "Aggregate throughput against K" below.

## Measured inputs

| Quantity | Value | How |
|---|---|---|
| KV cost | ~~**36.11 KiB/token**~~ **34.00 KiB/token** | superseded 2026-08-25, see below |
| prompt tokens | median 9,051, max 10,799 | completion log, both runs, identical |
| decode tokens | median 65,507, cap 84,144 | completion log, `eval time` lines only |
| per-stream throughput | ~30-31 tok/s | live run |
| weights | ~~16.1 GiB, ESTIMATE~~ **15.26 GiB on the GPU** | MEASURED 2026-08-25 |
| recurrent state | **149.6 MiB per SLOT** | MEASURED 2026-08-25, new term |

Do NOT derive the KV cost by subtracting an assumed weights size from `nvidia-smi`.
That method gave 45.55 KiB/token here and was wrong by 26 percent.

**The 36.11 figure is not a rate (corrected 2026-08-25).** It was read as
`total state size / length` off the `prompt_save` lines. That divides a fixed cost
by a variable length. The state has two parts, and only one of them scales:

```
prompt_save state (MiB) = 149.627 + 0.0332222 * length
```

That line is fitted on **309 `prompt_save` samples** from Qwen3.8-27B runs, lengths
11 to 121,160. The worst residual over all 309 is **0.001 MiB**. The slope is
34.02 KiB/token and the intercept is the per-slot recurrent state.

Dividing the whole thing by length gives 37.4 KiB/token at length 45,000 and
35.7 at length 93,000. The original 12 samples sat in that band, so they produced
"36.11, range 35.66-36.66". The number was read correctly. It was the wrong
quantity: a fixed per-slot cost had been smeared into a per-token rate.

The two halves are confirmed independently, by a method that shares nothing with
`prompt_save`. Fresh server launches on an RTX 3090 report
`llama_kv_cache: size` of exactly 34.00 KiB/token and
`llama_memory_recurrent: size` of exactly 149.625 MiB per slot. Slope agrees to
0.06 percent. Intercept agrees to 0.00 percent.

**Exclude `prompt eval time` lines when measuring decode.** A grep for
`eval time = ... / N tokens` also matches inside `prompt eval time`, which pools 36
prompt measurements with 36 completion measurements and halves the median. That error
produced a published figure of 40k tokens per call when the real median is 65k.

## The measured VRAM model (2026-08-25)

Measured on an RTX 3090, GPU 1 only, with the scored launch shape: `-ngl 999`,
`--cache-type-k q8_0 --cache-type-v q8_0`, no MTP. Five configurations. Each number
below is llama.cpp's own buffer report, cross-checked against per-PID `nvidia-smi`.

| Term | Value | Scales with |
|---|---|---|
| weights on GPU | 15,621.78 MiB | nothing |
| CUDA context | 307 MiB | nothing |
| compute-buffer floor | 80.28 MiB | nothing |
| KV cache | 34.00 KiB/token | total context cells |
| compute buffer | 5.00 KiB/token | total context cells |
| recurrent state | 149.625 MiB | SLOTS, not tokens |

So:

```
VRAM_MiB = 16,009  +  39.00 KiB/token * total_cells  +  149.625 MiB * K
```

The fixed 16,009 MiB is 15.63 GiB. That is the number the old "weights ~16.1 GiB"
estimate was reaching for, and it was close. The estimate was not the problem.

**Why the model has a per-slot term.** Qwen3.8-27B is a hybrid. It runs 16 attention
layers with a KV cache and 64 recurrent layers with a fixed-size state. The recurrent
state costs 149.625 MiB per slot and does NOT depend on context length. Measured at
4 slots (598.50 MiB) and at 16 slots (2,394.00 MiB), which is the same rate twice.
A model with only one per-token term cannot describe this card.

**The accounting closes.** Subtracting llama.cpp's reported buffers from per-PID
`nvidia-smi` leaves 305.9 to 307.4 MiB across all five configurations. That residual
is the CUDA context, and it is constant. The model predicts every measured
configuration to within 0.16 percent, and three of the five exactly:

| n_ctx per seq | slots | total cells | predicted | measured | error |
|---|---|---|---|---|---|
| 4,096 | 4 | 4,096 | 16,764 | 16,790 | -0.16% |
| 65,536 | 4 | 65,536 | 19,104 | 19,104 | 0.00% |
| 131,072 | 4 | 131,072 | 21,600 | 21,600 | 0.00% |
| 172,032 | 4 | 172,032 | 23,160 | 23,160 | 0.00% |
| 2,048 | 16 | 32,768 | 19,651 | 19,620 | +0.16% |

The 4,096 row misses because the compute buffer has a floor it has not yet cleared.
That floor does not matter at the pool sizes this note is about.

**Two checks worth keeping.** The auto-fitter (`-fit`, on by default) changed nothing:
16,790 MiB with it on and 16,790 MiB with it off. And an explicit `--parallel 16` set
`kv_unified = false` and `n_ctx_seq = n_ctx / 16`, which confirms the divide-the-pool
warning in `_default_induce_n_ctx`.

## The admission rule

From `_default_induce_n_ctx`:

```
n_ctx >= K_concurrent * (prompt_tokens + max_tokens)
```

It reserves against `max_tokens`, the CAP, not the observed median. Per stream today
that is 9,051 + 84,144 = **93,195 tokens**.

**Both halves of that 93,195 are local, not scored (found 2026-08-25).** Read the
constants before you size the scored card on this number.

- 84,144 is not a budget. It is `_pool_clamped_n_predict` clamping the request to
  the LOCAL pool: `106,496 - 22,352 = 84,144`. The clamp is a no-op on a large pool.
  The scored budget is `ARC_LIVE_GENERATOR_INDUCE_MAX_TOKENS_DEFAULT = 131,072`.
- 9,051 is the OBSERVED prompt median. The formula in `_default_induce_n_ctx` uses
  `_INDUCE_WORST_CASE_PROMPT_TOKENS = 22,352`, the worst case.

So the reserve is circular as written: it sizes the pool from a clamp that the pool
size produced. The code's own reserve is `22,352 + 131,072 = 153,424` tokens per
stream, 1.65 times larger. Both are priced below, because which one to use is an
operator decision about how much truncation risk to accept, not a measurement.

## What fits on 96 GiB

SUPERSEDED 2026-08-25. Kept per never-prune. The corrected tables follow it.

| K | KV at cap | plus weights | verdict |
|---|---|---|---|
| 8 | 25.7 GiB | 41.8 GiB | fits |
| 16 | 51.3 GiB | **67.4 GiB** | fits, uses 64 percent of the KV budget |
| 24 | 77.0 GiB | 93.1 GiB | fits on 2.9 GiB margin |
| 32 | 102.7 GiB | 118.8 GiB | does not fit |

K=16 is the safe default. K=24 fits, but its margin is smaller than the error bar on
the weights estimate, so pin the weights figure before choosing it.

### Corrected, reserve A: this note's 93,195 tokens per stream

| K | KV + compute | recurrent | fixed | total | margin | verdict |
|---|---|---|---|---|---|---|
| 8 | 27.7 GiB | 1.2 GiB | 15.6 GiB | 44.5 GiB | +51.5 | fits |
| 12 | 41.6 GiB | 1.8 GiB | 15.6 GiB | **59.0 GiB** | +37.0 | fits |
| 16 | 55.5 GiB | 2.3 GiB | 15.6 GiB | 73.4 GiB | +22.6 | fits |
| 22 | 76.3 GiB | 3.2 GiB | 15.6 GiB | 95.1 GiB | +0.9 | fits, barely |
| 24 | 83.2 GiB | 3.5 GiB | 15.6 GiB | 102.3 GiB | **-6.3** | does NOT fit |
| 32 | 110.9 GiB | 4.7 GiB | 15.6 GiB | 131.2 GiB | -35.2 | does not fit |

### Corrected, reserve B: the code's own 153,424 tokens per stream

| K | KV + compute | recurrent | fixed | total | margin | verdict |
|---|---|---|---|---|---|---|
| 8 | 45.7 GiB | 1.2 GiB | 15.6 GiB | 62.5 GiB | +33.5 | fits |
| 12 | 68.5 GiB | 1.8 GiB | 15.6 GiB | **85.9 GiB** | +10.1 | fits |
| 13 | 74.2 GiB | 1.9 GiB | 15.6 GiB | 91.7 GiB | +4.3 | fits, barely |
| 16 | 91.3 GiB | 2.3 GiB | 15.6 GiB | 109.3 GiB | **-13.3** | does NOT fit |
| 24 | 137.0 GiB | 3.5 GiB | 15.6 GiB | 156.1 GiB | -60.1 | does not fit |

**What changed and why.** The old table charged 36.11 KiB/token and no per-slot term.
The correct charge is 39.00 KiB/token plus 149.6 MiB per slot. The per-token error
grows with K, so the old table was most wrong exactly where the decision was tightest.

**K=24 is out.** It was the one row whose margin (2.9 GiB) was smaller than the stated
uncertainty. The measurement resolved that uncertainty against it. K=24 misses by
6.3 GiB under reserve A, and by 60 GiB under reserve B.

**K ceilings.** Reserve A allows K=22, or K=21 with the 1,500 MiB guard margin the code
already uses. Reserve B allows K=13. Choose K=12 and both reserves are safe, with
10 GiB of headroom even under the stricter one.

**One caveat that no local run can remove.** These numbers come from an RTX 3090. The
weights, the KV rate and the recurrent state are properties of the model and the KV
quantisation, so they carry to any card. The 307 MiB CUDA context and the compute-buffer
geometry are properties of the driver and the kernels, so they may differ on the
Blackwell card. Both are small. They cannot move a verdict by 6 GiB.

## Aggregate throughput against K (measured 2026-08-25)

The old projection assumed near-linear scaling. It does not hold.

**Method.** One server on an RTX 3090, GPU 1, `--parallel 16`, `-c 32768`, so every K
sees the same slot geometry. Only the number of simultaneous streams changes. Each
stream decodes a fixed token count with `ignore_eos`, so aggregate is
`total decoded / wall clock`. Prompts differ per stream, so the prompt cache cannot
make a later stream cheap. Two repeats per point, agreeing within 1 percent.

| K | aggregate | speedup vs K=1 | efficiency | per stream |
|---|---|---|---|---|
| 1 | 40.7 tok/s | 1.00x | 100% | 41.1 tok/s |
| 2 | 65.9 tok/s | 1.62x | 81% | 33.5 tok/s |
| 4 | 81.8 tok/s | 2.01x | 50% | 20.8 tok/s |
| 6 | 89.5 tok/s | 2.20x | 37% | 15.2 tok/s |
| 8 | 66.9 tok/s | 1.65x | 21% | 8.5 tok/s |
| 12 | 158.2 tok/s | 3.89x | 32% | 13.6 tok/s |
| 16 | **177.8 tok/s** | **4.37x** | 27% | 11.5 tok/s |

**The headline: 4.4x at K=16, not 16x.** Concurrency still helps, and it is still the
only lever. It buys about a quarter of what the old projection assumed.

**There is an unstable band around K=6 to K=10.** Raising K there can LOWER aggregate
throughput. K=8 measured below K=4 in two of three sweeps. The band is reproducible
within a sweep (repeats agree within 1 percent) but its exact position moves with
decode length: the dip sat at K=6 at 512 tokens and at K=8 at 1024 tokens.

**The cause is NOT the prompt cache.** The server saves idle slot state to a host-RAM
prompt cache, and each slot of this hybrid model holds 149.6 MiB of recurrent state,
so slot-state shuffling was the obvious suspect. It is wrong. Re-running the whole
sweep with `--cache-ram 0` reproduced the same dip and the same K=16 figure to within
2 percent. Do not spend time on this suspect again. The cause is undetermined.

**Do not choose a K in that band.** K=12 delivers 89 percent of K=16's aggregate for
three quarters of the memory, and it fits under both reserves. K=16 is the maximum
useful setting on this evidence, not a step toward K=24.

## What that buys

SUPERSEDED 2026-08-25. Kept per never-prune. Its K=16 and K=32 rows assumed the
near-linear scaling that the table above refutes.

| setup | aggregate | inductions in 12h | per game, 25 games |
|---|---|---|---|
| local, 1 stream | 30 tok/s | 20 | 0.8 |
| K=16 | ~500 tok/s | 330 | 13.2 |
| K=32 | ~1000 tok/s | 660 | 26.4 |

Corrected, using the measured speedups against the same 30 tok/s live baseline:

| setup | aggregate | inductions in 12h | per game, 25 games |
|---|---|---|---|
| local, 1 stream | 30 tok/s | 20 | 0.8 |
| K=12 | ~117 tok/s | 77 | 3.1 |
| K=16 | ~131 tok/s | 86 | **3.5** |
| K=24, K=32 | not available | — | does not fit the card |

A local single-stream run is not a slow version of the scored eval. At 0.8 inductions
per game it is a different activity, and results from it do not transfer.

That paragraph stands, and it now cuts closer. K=16 buys 3.5 inductions per game, not
13.2. That is a real gain over 0.8, but it does not reach the budget the old table
implied. Plan against 3.5.

**How far to trust these numbers off this card.** The SHAPE is the transferable part,
not the absolute rate. Three caveats, all stated rather than corrected for:

- Different card. The 3090 has roughly half the memory bandwidth of the scored
  Blackwell card. The Blackwell may scale better. Nothing local can show that.
- Much shorter context. Slots held 2,048 tokens here against 93,195 on the scored
  card. Longer context makes each decode step read more KV, which normally makes
  concurrency scale WORSE. So this measurement most likely flatters the scored path.
- Shorter decode. 512 and 1,024 tokens per stream, against a live median near 65,000.

The single-stream rate here is 41 tok/s against the live run's 30-31 tok/s, and the
short context is why. Use the speedup column, not the absolute column.

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

Both original bullets were closed on 2026-08-25. They are kept here, struck, with the
answer under each, because both answers changed a decision this note had already made.

- ~~Whether aggregate throughput scales linearly to K=16 on that card. Memory
  bandwidth may cap it below 500 tok/s.~~
  **MEASURED. It does not scale linearly.** K=16 gives 4.4x, not 16x. The projection
  was wrong by about 4x. See "Aggregate throughput against K".
- ~~The weights size. 16.1 GiB is an estimate and shifts every total by about a GiB.~~
  **MEASURED. Weights are 15.26 GiB on the GPU, 15.63 GiB with fixed overhead.** The
  estimate was good to about 0.5 GiB. It was not the real error bar. The real error was
  a MISSING TERM: 149.6 MiB of recurrent state per slot, plus a 5.00 KiB/token compute
  buffer. K=24 does not fit.
- What compaction would actually save, since it has never been enabled on a live run.
  Still not measured.

Newly open, from this pass:

- Why aggregate throughput dips in the K=6 to K=10 band. Reproducible, and NOT the
  prompt cache. Worth a profiler, not another guess.
- Whether that dip and the 4.4x ceiling reproduce on the Blackwell card. Only the
  scored card can answer this.
- Which per-stream reserve the scored run should use, 93,195 or 153,424. That is an
  operator call about truncation risk. It moves the K ceiling from 22 to 13.

## How this was measured (2026-08-25)

Reproduce with the pinned binary
`~/.cache/llama.cpp-master/build/bin/llama-server` and the full GGUF path, on GPU 1
only, `CUDA_VISIBLE_DEVICES=1`. Confirm placement by joining PID to GPU UUID through
`nvidia-smi --query-compute-apps`, never by trusting the env var. Confirm the server
is on the NVIDIA card, not the AMD iGPU, by checking it holds nvidia fds and zero
`/dev/dri` fds.

VRAM terms come from llama.cpp's own `-v` buffer reports: `CUDA0 model buffer size`,
`llama_kv_cache: size`, `llama_memory_recurrent: size`, `CUDA0 compute buffer size`.
Per-PID `nvidia-smi` is the independent cross-check on their sum. Do not read a
weights figure off `nvidia-smi` alone; it also carries the CUDA context.

## Cross-references

- `docs/research-notes/induction-token-accounting-and-rejection-2026-08-25.md` — where
  the 97.6 percent discard figure and the reasoning-channel explanation come from
- `docs/research-notes/induction-rejection-adversarial-read-2026-08-25.md`
- `python/carnot/agentic/arc_executable_world_model.py` — `_default_induce_n_ctx`,
  the slot knob, and the `--parallel` warning

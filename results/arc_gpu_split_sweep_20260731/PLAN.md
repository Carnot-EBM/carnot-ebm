# GPU layer-split sweep — plan (queued 2026-07-31, runs after the QAT head-to-head)

**Question.** Does `-sm layer` across both 3090s give near-single-GPU throughput at roughly
half the per-device VRAM — enough headroom for the MTP draft head and a larger KV cache —
and does it do so EVENLY?

**Why it matters.** The only VRAM lever the agent uses today is `-ot ...=CPU`, which spills
FFN weights to host RAM and is measured to cost 58-86% of decode throughput. Layer-splitting
should cost near-nothing, because each GPU computes its own layers and only activations cross
the bus. If that holds, the CPU lever becomes unnecessary on any multi-GPU host, and the
Kaggle 4xL4 assumption behind `CARNOT_ARC_MTP=1` gets an actual measurement behind it.

## AMENDED 2026-07-31, after the plan was written: the model under test changed

This plan was drafted against `gemma-4-31B-it` **Q4_K_M**. Later the same day the live
generator was repinned to the **QAT** build (`gemma-4-31B-it-qat-UD-Q4_K_XL`), and the harness
now resolves whatever `arc_executable_world_model` pins rather than a hardcoded path — so a
bare run measures QAT, not Q4_K_M.

**That breaks the control as originally specified**, and the break is recorded rather than
papered over. Arms 1 and 2 were written to reproduce the 2026-07-28 figures (36.07 tok/s @
21416 MiB; 15.05 tok/s @ 19072 MiB). Those were measured on Q4_K_M. Run against QAT they
**will not** reproduce, and a mismatch would say nothing about harness health.

Resolution — the arms are unchanged, but they are read differently:

- **Run all five arms on the SHIPPED model (QAT).** The question is whether `-sm layer` is
  ~free, which is a *within-run* comparison: arm 3 against arm 1, arm 5 against arm 4. That
  comparison is valid on any single model, and QAT is the one that will actually be served.
- **Arms 1-2 are now within-run baselines, NOT cross-session reproductions.** They still earn
  their slot: arm 1 anchors the split arms, and arm 2 prices the CPU lever on the same model
  so the two levers are compared like-for-like.
- **To recover the original harness-validation check**, run arms 1-2 a second time with
  `MEAS_GGUF` pinned to the Q4_K_M file. Only then are the 07-28 numbers a legitimate target.
  This is optional; skipping it means the harness is unvalidated against a known-good result,
  which is worth knowing before trusting arms 3 and 5.

Everything below is the plan as originally written.

## Arms (q8_0 KV, -ngl 999, 256 predicted tokens; model resolved from the shipped pin — see
the amendment above, which supersedes the "all Q4_K_M" this line originally read)

| # | label | GPUs | -ts | CPU FFN layers | n_ctx | purpose |
|---|---|---|---|---|---|---|
| 1 | single_ctx32768 | 1 | - | 0 | 32768 | within-session baseline; must reproduce 07-28's 36.07 tok/s @ 21416 MiB |
| 2 | cpu12_ctx32768 | 1 | - | 12 | 32768 | the CURRENT lever; must reproduce 15.05 tok/s @ 19072 MiB |
| 3 | **split2_ctx32768** | **0,1** | **1,1** | 0 | 32768 | **the new arm** |
| 4 | single_ctx81920 | 1 | - | 0 | 81920 | shipped n_ctx, single card (23888 MiB measured) |
| 5 | **split2_ctx81920** | **0,1** | **1,1** | 0 | 81920 | **the new arm at the shipped context** |

Arms 1-2 are deliberately redundant with the 2026-07-28 run. They are the harness's own
control: if they do NOT reproduce, the split numbers are not trustworthy either, and that is
worth knowing before drawing a conclusion from arms 3 and 5.

## What counts as a win

- decode tok/s within ~10% of arm 1 (i.e. splitting is ~free, unlike CPU offload)
- `residency.per_card_mib` roughly halved per device
- `residency.max_minus_min_mib` small -- this is the "EVENLY" claim, reported as a number
  rather than asserted. `-ts 1,1` is passed explicitly because llama.cpp's default splits by
  AVAILABLE VRAM, so a busy card would skew it.

## Preconditions

- BOTH 3090s idle. Arms 3 and 5 use GPU 0, which is normally the conductor's lane; the
  conductor is stopped, so there is no contention. Do not run this while it is running.
- The QAT head-to-head must be finished -- it holds GPU 1.

## Run

```
.venv/bin/python results/arc_gpu_split_sweep_20260731/harness/gpu_split_measure.py \
  "$(cat results/arc_gpu_split_sweep_20260731/arms.json | tr -d '\n')" \
  results/arc_gpu_split_sweep_20260731/gpu_split_results.json
```

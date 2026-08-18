---
type: Claim
title: llama.cpp batches decode 4.37x at k=16, and MTP suppresses it
description: Measured on the scored Blackwell card. MTP wins single-stream 1.81x and loses 2.1x at k=16.
tags: [arc-agi-3, throughput, mtp, concurrency, blackwell]
status: stable
resource: https://www.kaggle.com/code/iancblenke/carnot-arc-blackwell-bench
sources:
  - id: bench-v5
    resource: /home/ianblenke/.claude/jobs/ad0c053d/tmp/kbench5/bench_results.json
    author: outer-loop/claude-opus-5
generated: { by: outer-loop/claude-opus-5, at: 2026-08-18T12:40:00Z }
verified:
  - { by: outer-loop/claude-opus-5, at: 2026-08-18T07:25:00Z }
---

# The measurement

RTX PRO 6000 Blackwell, NVFP4 Qwen3.8-27B, 22000-token prompt and 8192 generated tokens per
request with `ignore_eos` forcing the count. Every arm finished `length`, no truncation.

| arm | k=1 | k=4 | k=16 |
|---|---|---|---|
| MTP off | 52.2 | 134.7 (2.58x) | **228.3 (4.37x)** |
| MTP on | 94.7 | 101.5 (1.07x) | 108.8 (1.15x) |

llama.cpp batches decode. MTP is what was suppressing it -- speculative decoding trades bandwidth
for compute, and 16 slots each running draft-and-verify saturates compute, where batching has
nothing left to amortise.

# The crossover, which is the actionable part

MTP wins single-stream 1.81x (94.7 vs 52.2). MTP-off wins at k=16 by 2.1x (228.3 vs 108.8). With
25 game threads sharing slots, aggregate is the currency, so the scored path should run
`--parallel 16` with MTP OFF.

Against the budget: roughly 100 induction calls at a 61,284-token median need about 148 tok/s
aggregate to fit the 11.5h swarm cap. 228 clears it with ~54% headroom. The shipped config today
gets neither side of the trade -- MTP is off AND `--parallel` is never passed, so it runs the
k=1 arm at 52.2.

# What this closes

The case for migrating to vLLM rested on "llama.cpp does not batch". It does. A 3.7 GB wheel
closure was resolved and uploaded as insurance and is not needed.

# Caveat, stated because it is not yet measured

This ran 8192 generated tokens; production generates ~61k. Longer generations mean more KV
pressure per slot at k=16. The arithmetic fits (16 x 32k cells against 96 GB) but has not been
measured at production length.

# Four invalid predecessors, recorded so the shape is not repeated

v1 ran on a Tesla P100 because the bench kernel omitted `competition_sources` -- the Blackwell
card is provisioned for the competition. v2 and v3 used a 32:1 prompt:output ratio, which is
prefill-dominated; prefill saturates at batch 1 and cannot be batched away, so the observable
ceiling was ~1.7x no matter how well the engine batched. v4 reshaped the prompt to 22000 but
left `max_tokens` as a ceiling with a prompt saying "reply with a short plan", and got ~190
tokens -- a 116:1 ratio, worse than what it replaced. Only v5, with `ignore_eos` forcing the
token count and `finish_reason` captured per request, measured the decode regime production
actually lives in.

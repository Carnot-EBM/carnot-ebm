# The induction tier is truncation-bound, and its own diagnostic is clipped before the numbers

Written 2026-09-02 from two live adapter-free ARC runs. The short version: the local generator is
not failing to reason. It is being cut off mid-reasoning by a shared context pool, so it never
emits the `engine` field the induction step needs, and the record that would let anyone size the
fix is clipped 66 characters too early.

## What was measured

Two runs of `scripts/arc_leaderboard_eval.py --policy e3`, live Qwen3.8-27B on GPU 1.

| Run | Game | Levels | Actions | `chat_completions` | `reasoning_only` | `chars_final` |
|---|---|---|---|---|---|---|
| ls20/wa30 (13.8h) | ls20 | 0 | 2443 | 24 | 24 | 0 |
| ls20/wa30 | wa30 | 0 | 2478 | 24 | 24 | 0 |
| cd82/r11l | r11l | 2 | 2121 | 42 | 41 | 3457 |

`both_channels_empty` is 0 in every case, so the server always answered. The model produced
2.4 million characters of reasoning on r11l alone and 3,457 characters of final output.

## The mechanism

Three induction attempts in the r11l run failed with:

```
split induce: engine failed: local model code unusable after 3 tries
(missing ('engine',) in output [TRUNCATED BY SHARED CONTEXT POOL:
generated only 18431 of the 26800-token budg
```

The other two generated 2,996 and 4,066 tokens of the same 26,800-token budget.

This is not a mystery mode. `arc_executable_world_model.py:_limit_diagnostic` already separates
the two faults that both report `stop_type == "limit"`, and its docstring names this one:

> SHARED-POOL TRUNCATION -- the model was cut off FAR short of max_tokens because the prompt had
> already consumed most of the server's shared context pool, so only the leftover cells were
> available to generate into. The fix is a bigger `-c` / `CARNOT_ARC_INDUCE_N_CTX`, and a bigger
> max_tokens would make it WORSE.

Supporting evidence from the same run: `last_stop_type: "limit"`, `last_raw_completion_len:
12151`, and `last_prompt_truncated: false` — the prompt fits, the generation does not.

Server configuration observed directly from `/proc/<pid>/cmdline`:

```
llama-server -m .../Qwen3.8-27B-Q4_K_M.gguf -ngl 999 -c 49152 --port 8919
             --cache-type-k q8_0 --cache-type-v q8_0
             -ot blk\.(0)\.ffn_(gate|up|down)\.weight=CPU
```

`/slots` reports four slots. The interaction between `-c 49152` and the slot count is the pool
this note is about, and is the first thing to measure before changing anything.

## The defect worth fixing first

`_limit_diagnostic` builds a 216-character message that ends:

```
... in an n_ctx=49152 pool -- the prompt consumed the rest.
RAISE -c / CARNOT_ARC_INDUCE_N_CTX; raising max_tokens would make this worse]
```

That tail never reaches the record. `arc_executable_world_model.py:8314` stores the engine error
as:

```python
return False, f"split induce: engine failed: {str(eng)[:150]}"
```

The prefix is 29 characters, the clip is 150, and every stored instance is exactly 179 characters
— verified against all three observed strings. The message is cut mid-word at `budg`, so the
artifact records THAT truncation happened while dropping the pool size and the instruction naming
the fix.

The irony is documented in the file itself. The docstring above the HTTP-error enricher at line
5535 says two independent sessions already re-derived what an unread error body contained, and
that the change existed to put "the evidence in the message". The evidence is in the message; it
is the storage that clips it.

**Proposed fix, not made here:** raise the clip at 8314 to cover the diagnostic, or store the
diagnostic in its own field rather than concatenated into a capped string. Small, but it needs its
own check that no other consumer depends on the 150-character bound.

## What this means for the tool-use question

Tool use was OFF in both runs. `/proc/727651/environ` carries only
`CARNOT_ARC_GENERATOR_CUDA_GPU`, `CARNOT_ARC_INDUCE_N_CTX`, `CARNOT_ARC_TRAJECTORY_SUPERVISOR` and
`CARNOT_LLAMA_SERVER`; `CARNOT_ARC_INDUCE_TOOL_LOOP`, `CARNOT_ARC_INDUCE_CANDIDATE_TOOLS` and
`CARNOT_ARC_SUPERVISOR_TOOL_ARM` are absent, consistent with `tool_loop_reinduction: fired=0` in
the supervisor receipt. So neither run says anything about whether tools help.

More importantly, an A/B run now would not measure tools. Tool definitions and tool-call
transcripts lengthen the prompt, and a longer prompt leaves fewer cells to generate into — the
exact quantity that is already exhausted. A tool-use arm would likely score WORSE for a reason
that has nothing to do with tools, and that result would be recorded as evidence against them.

**Order of work: fix the pool sizing, confirm `reasoning_only` falls, and only then run the tool
A/B.**

## Prior art, so this is not re-derived a fourth time

- `exp5798` — verdict `answer_channel_diagnostic_ready_qwen_channel_failure_not_competence_failure`
- `exp5799` — `answer_channel_ready_score = 0.0`, `qualified_models = 1`
- `exp5866` mode C — the incident `_limit_diagnostic` was written for, where a 15,754-token prompt
  in a 16,384 pool left 630 cells, produced 2,133 characters, and returned HTTP 200
- REQ-ARC-WMTE-6620 — the pool-clamped `n_predict` this diagnostic compares against

## Limits of this note

Two runs, three games, one model, one server configuration. The pool arithmetic (how `-c 49152`
divides across four slots, and how large the induce prompts actually are) is NOT measured here —
prompt token counts are not recorded anywhere in the artifact, which is its own gap. Nothing in
this note establishes that raising `-c` fixes the solve rate; it establishes only that the
induction tier is currently truncation-bound and that a tool A/B run before fixing it would be
uninterpretable.

## CORRECTION 2026-09-02 15:45Z — the pool is UNIFIED; the cause is prompt SIZE, not slot splitting

Measured by the follow-up agent on GPU 1, Qwen3.8-27B Q4_K_M, `-c 49152`, q8 KV. These supersede
the framing above, which treated the four `/slots` entries as a division of the pool.

**The pool is not divided.** A single fresh stream with a 55-token prompt generated its full
15,000-token budget (`predicted_n=15000`, `stop=limit=budget`, `truncated=false`). One stream owns
all 49,152 tokens. So the section above headed "Server configuration" was right to call the slot
interaction unmeasured, and my hourly summary that day was WRONG to describe the budget as one
"that can't hold a 26,800-token generation across 4 concurrent slots". Sibling slots were not
hoarding anything.

**The prompt is what fills the pool, and it scales steeply with transition count.** The induce
prompt renders ALL transitions when `k=None`, and the live agent passes every transition since
level-start (`_induce_rows = active_transitions`). Measured on cd82: 25 transitions produce 6,589
tokens; 82 transitions produce 20,431. The supervisor's own diagnosis strings cite 220-299
transitions per induction, which puts real induce prompts far above both.

**That reconciles the live numbers.** The r11l truncations of 18,431 / 4,066 / 2,996 tokens against
a 26,800 budget imply prompts of roughly 30-46k tokens — the many-transition prompt, not the ~6k
fresh one. The generation is capped by what the prompt leaves behind against the context wall
(`server-context.cpp:1603`, `ctx_shift=false`).

**Consequence for the fix.** "Raise `-c`" is right in DIRECTION but `--parallel 1` alone does not
solve it: a 30k prompt on a single slot still caps generation near 19k. The real levers are raising
`-c` (bounded by 24GB VRAM, ceiling being measured) and/or capping transitions `k` — and capping k
is an ACCURACY change, so it needs an A/B rather than a config edit.

**Status when this correction was written:** a large-prompt probe (30k prompt, 26,800 budget,
single slot) was running to confirm generation stops near 19,152. No fix is claimed yet. An
env-gated `CARNOT_ARC_LLAMA_SERVER_PARALLEL` launch knob exists, tested and mutation-proven — it
guarantees the whole pool to a single-stream eval, which is a genuine improvement but not the full
fix.

The rest of this note stands: the diagnostic clip (now fixed, REQ-ARC-WMTE-6860), the
`reasoning_only` counts, the prior art, and the conclusion that a tool-use A/B run before this is
resolved would measure truncation rather than tools.

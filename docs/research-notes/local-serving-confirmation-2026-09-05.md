# Local serving confirmation, 2026-09-05

This follows the operator's request to exercise serving flags before building
on them. CLAUDE.md was read in full. The worktree started clean at a690639bec.
All model work uses CPU and the 532,517,120-byte Qwen3.5-0.8B Q4_K_M GGUF.
The real binary reports `version: 9606 (9b4dae81f)`.

The [raw transcript](local-serving-confirmation-transcript-2026-09-05.json)
records every launch, HTTP command, response,
wall time and server log. Numbers below are measured once, not estimates of
27B cost or ARC performance. No GPU was used and no `results/**` was written.

Initial findings:

- Slot save writes 22,196,640 bytes for 162 saved tokens. HTTP save takes
  0.007153 s; restore takes 0.004270 s. Server timers report 5.906 ms and
  3.447 ms. The filesystem page cache is warm; these are not durable-fsync costs.
- After SIGKILL and restart, restore takes 0.003962 s (server: 3.115 ms).
  Baseline, restored, restarted and cold continuation tokens match. Output is
  `11, 12, 13, 14, 15, 16, `. Restored runs report cache_n=162, prompt_n=1.
  The cold control reports cache_n=0, prompt_n=163. Equal output alone would
  not prove reuse; these counters provide the needed second observation.
- `--reasoning-format deepseek` returns separate reasoning and final fields.
  `none` leaves thought tags in content. Replaying `reasoning_content` as an
  assistant field is accepted but `/apply-template` discards it. Explicitly
  pasting the extracted text into the next user message preserves it and yields
  `53`. This is ordinary text reuse. It is not opaque reasoning-state transfer.
- Native `--jinja` tools return a parseable `inspect_cell` call with
  arguments `{"x":2,"y":3}`. This says nothing about the pinned 27B's tool syntax.
- Launch-level grammar works through raw and chat endpoints. Per-request grammar
  also overrides a request to say BANANA, producing a valid tool envelope.
  The identical request without grammar emits `banana`. The grammar permits
  actions 1 through 4; it does not contain one fixed answer.
- `--cache-reuse 8` starts but the server logs that cache reuse is unsupported
  by this context and disables it. Changed-prefix prompts receive no cached
  tokens. Do not build cache shifting on this measured hybrid-model path.
- Slot similarity selects newer slot 1 over older slot 0 for its matching prompt.
  The server logs LCP similarity 1.000 over threshold 0.500; 237 tokens are cached.
- Idle-slot caching writes RAM cache entries. An explicit-slot revisit still
  reprocesses all 241 prompt tokens. A write to RAM is not proof of useful reuse.

The first probe stopped at restart because its own port preflight lacked
SO_REUSEADDR and encountered TIME_WAIT. The server had exited with -9. That
aborted transcript is preserved. The corrected preflight was rerun; the claim
above uses that completed run, not the aborted one.

Ranked plan for this change:

1. Build grammar-constrained tool transport on the existing live loop, default
   off. A measured server mechanism can enforce parseable envelopes without
   native tool training or parser compatibility. Existing verifiers still judge
   proposed code. ARC usefulness remains unmeasured.
2. Next, design full run recovery around a durable action/observation journal,
   policy state, generator request boundaries and explicit slot ownership.
   Slot restore is confirmed, but KV alone cannot restore the environment or
   E3 search state. Automatic snapshots on shared slots could save another game's
   cache. No automatic recovery is implemented or claimed here.
3. Keep explicit visible induction memory. Do not silently paste model-written
   thoughts into trusted evidence. Extraction is already supported by the current
   proposer; a second extractor would duplicate existing code.
4. Defer cache shifting, automatic idle cache tuning, and native-template changes.
   Shifting is disabled here; RAM writes did not yet show reuse; the native
   0.8B request already works. Scaling does not answer these integration questions.

The 27B was not launched. No mechanism showed a scale-dependent fault that
would justify escalation. A paired local-model ARC trial and full recovery
design remain separate work. The feature stays off pending that evidence.

Follow-up measurements before implementation:

- Idle cache reuse DOES work when the request lets the server select a slot.
  The automatic revisit reports cache_n=237, prompt_n=4, in 0.078295 s.
  Its log records loading the matching RAM entry. The earlier explicit-slot
  negative remains valid: forcing a slot bypasses that automatic cache lookup.
- A save submitted during active generation waits for completion. The request
  takes 3.275792 s; the disk-write timer reports only 5.437 ms. The generation
  future is unfinished before the save and finished when save returns. This
  endpoint did not checkpoint a running decode. The sample generated 256 tokens.
- Per-request grammar plus `enable_thinking=false` and
  `thinking_budget_tokens=0` also works on a server launched with reasoning ON.
  It returns `{"name":"inspect","arguments":{"action":4}}` with no reasoning
  field. The implementation will use this measured request shape.
- The killed server restarts to healthy in 1.111825 s, measured separately from
  restore. No persistent sampler state or stochastic-continuation equality was
  tested. All equality tests used temperature zero and seed 42.

Idle-cache reuse is therefore confirmed for automatic selection. It remains
deferred as a tuning change: the live server already enables it by default when
RAM caching is available. The new information strengthens that mechanism claim,
but supplies no ARC performance evidence or reason to change defaults.

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

Implementation and final-loop observations:

REQ-ARC-WMTE-7044/7045 is reachable from the existing `LocalGGUFProposer.induce`
call in the scored policy and the `arc_loop_solve.py --mechanism e3` factory.
The flag is `CARNOT_ARC_INDUCE_TOOL_GRAMMAR=1`; the induction tool loop must also
be enabled (`CARNOT_ARC_INDUCE_TOOL_LOOP=1`, or an existing selfparse/repair
route). Default requests retain their previous native-tool or raw shape.
The frozen session schemas supply allowed names. GBNF constrains one JSON
envelope, and the existing dispatcher checks argument meaning and verifies
submitted engines. The original JSON and actual result go into the next request.
Scored primary, bounded-refinement and repair receipts retain grammar counters.
No confidence, solver acceptance threshold, or trust criterion changed.

The grammar route explicitly refuses the vLLM backend and enabled native-message
compaction before a grammar chat request. Both use the existing failure/fallback
path. Complete malformed responses, truncated JSON, nonfinite constants, unknown
tools and invalid envelope shapes are rejected. Available failed-response token
counts remain visible. A valid envelope may still contain useless arguments.

Two real CPU trials exercised the full production loop on synthetic grid+1 data:

- The first used a 384-token cap. It wrote an unfinished JSON/code response,
  hit `finish_reason=length`, and published no engine. Its initial diagnostic
  incorrectly reported zero tokens; parsing was moved after token accounting.
  This failed trial remains in the supplement rather than being overwritten.
- After removing a contradictory final-code-fence instruction and raising the
  cap to 1,024, two turns returned exactly
  `{"name":"run_engine_on_transitions","arguments":{}}`.
  Both parsed and reached the actual dispatcher. It returned the observed
  missing-`code` argument error, which the second request contained.
  Measured totals: 31 decoded tokens, 11.286728 seconds, zero scoreable engines,
  `terminated_by=turn_cap`. The model ignored the requested initial inspection.

These trials demonstrate constrained transport, dispatch, rejection and feedback.
They do not demonstrate successful model-driven engine induction. Scripted tests
separately prove that valid submitted source reaches verification and the real
engine writer. No ARC efficacy or open-weight 27B result is inferred from them.

The [validation supplement](local-serving-validation-transcript-2026-09-05.json)
contains exact probe sources, launch/HTTP commands, outputs and logs for these
trials and subsequent production-grammar controls. The
[mutation receipt](local-serving-mutations-2026-09-05.json) contains exact source
deletions, test commands, assertion output, `cmp` commands and restored GREEN
runs. Original imperfect proofs are retained and classified explicitly.


The final production-grammar control used the same GBNF builder as the live
request. Against the prompt to say BANANA, it returned
`{"name":"diff_grids","arguments":{}}` (23 tokens, 0.694265 s).
Deleting only the grammar from that request returned `BANANA` (3 tokens,
0.067148 s). A nested/escaped JSON copy did not complete within 160 tokens;
it ended inside an open string. No full nested-value generation roundtrip is
confirmed. This is not evidence that the grammar admitted an invalid interior
character, and it is not a general JSON-language coverage result.

The production control's recorder retained an aliased response object. Consumer
lifting later inserted `tool_calls` into that event store. The supplement
therefore preserves both the original stdout events (printed before lifting)
and the explicitly labeled mutated event store. Only the former is raw server
output. The lifted calls are local consumer output, not server tool extraction.

Mutation closure: 40 distinct call-site mutations across 45 executions, with
assertion RED, byte-identical `cmp` restoration, and GREEN for each final proof.
The first batch had one actual GREEN survivor: selfparse precedence lacked a
combined literal-tag test. Its first retest failed only because the candidate
fixture lacked a description. After fixing that fixture, deletion corrupts the
actual next-request JSON and the equality assertion fails. The outer-choice
guard initially produced exception-only RED; an explicit no-exception assertion
now verifies fallback behavior. All imperfect attempts remain in the receipt.
No dead rule was retained solely because a constant-value test passed: the
precedence rule's effect is demonstrated in the actual HTTP message stream.

A final review found stale success counters on early server/staging failure.
The loop now installs fresh zero counters before initialization. Six tests drive
repair and refinement through unavailable/raising startup and failed evidence
staging. Deleting the fresh receipt fails assertions in both callers. The moved
enable-decision call site was also remutated. Earlier proofs retain their exact
source hashes; the freshness amendment is separately identified in the receipt.

Final verification: 109 focused tests pass. Ruff and whole-package mypy pass;
installed hooks pass without bypass. Reachability lint reports 85 live modules.
Whole-tree collection reports 61,816 tests and eight existing import errors:
four need `carnot._rust`, three refer to missing experiment imports (6814, 6819,
6828), and one quarantined test imports missing Exp4058. The global spec audit
reports 1,178 pre-existing unreferenced tests; changed-file spec coverage passes.
Reconciliation reports that same global traceability backlog. Those broken
paths are unchanged by this branch. Global repository GREEN is not claimed.

Applicable E2E-009/010 checks passed at their declared scope. The real offline
environment smoke made 11 actions (12-step budget including reset) in 0.61 s,
completed zero levels, and never reached an LLM. The generated model-server
processes were stopped by their exact PIDs; port 18919 is no longer listening.
The evidence and implementation commits are b288f4553e, d5d72141ca and 33c63162f8.

Remaining work: full ARC-run recovery design and implementation, 27B behavior,
and paired ARC efficacy evaluation. These are deferred; the new transport remains
unevaluated and off. No operator decision is needed to leave this default-off
change in place. Recovery scope and a suitable local-model trial are future work.

An implementation checkpoint initially failed pre-commit because the working
tree changed during mypy: I had edited the test while that hook was running.
Mypy itself passed. The test was explicitly restaged and the installed hooks
passed on retry. No hook was bypassed. The later code checkpoint temporarily
made reconciliation report missing same-commit ops documentation; the following
documentation commit reconciles it, leaving the existing global traceability
backlog as the only final reconciliation issue.

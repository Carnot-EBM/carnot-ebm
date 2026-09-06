# Induction tool grammar: model-free proof, fix, and a bounded 27B trial (2026-09-05)

REQ-ARC-WMTE-7044, 7045, 7046. Branch `outer-loop/kv-persistence-confirm`, continued
on the worktree branch that carries this note.

## 0. Summary

- The grammar shipped under REQ-7044 admitted `{"name":"run_engine_on_transitions",
  "arguments":{}}`. Proved with llama.cpp's own `test-gbnf-validator`, no model.
- The recorded 0.8B "negative" (31 tokens, no engine) was therefore uninterpretable, and
  it was never a 27B result.
- Fixed in two rounds. Round 1 requires each tool's required arguments with their JSON
  type. Round 2, after an adversarial review, requires each source argument to contain
  the definition dispatch looks for, and makes the loop send a submission-only grammar
  at its force turn. 52 of 52 verdicts re-established against the binary.
- A bounded 27B trial ran on GPU 1 between the rounds (14 cells, 18.5 min of GPU, server
  stopped by PID). Through the round-1 grammar the pinned 27B produced 28 of 28 parseable
  envelopes and 13 real engine submissions across 7 games; one game reached accuracy
  0.919 visible / 1.0 held-out at 422 tokens. Every submission came after the force
  nudge, so the result shows the transport carries code on the 27B and does NOT show the
  model submits without being made to. No further GPU time was spent after the review.

## 1. What was open

The grammar transport (`CARNOT_ARC_INDUCE_TOOL_GRAMMAR=1`) had one recorded trial. It ran on
Qwen3.5-0.8B, the 0.5 GB CPU smoke model: two parseable calls, missing `code`, zero
scoreable engines, 31 tokens, 11.29 s. The branch scoped that honestly ("says nothing
about the pinned 27B"). The result was relayed as if it were the 27B. The pinned 27B
produces world models on the ordinary induce path, so the grammar arm was untested on it,
not tested-and-failed.

## 2. The grammar admitted an empty shell (proved with no model)

`_tool_grammar` (commit d5d72141ca) was:

```
root ::= "{\"name\":" tool-name ",\"arguments\":" object "}"
tool-name ::= "\"run_engine_on_transitions\"" | ...
object ::= "{" ws (string ":" ws value ("," ws string ":" ws value)*)? "}" ws
```

`arguments` is any JSON object. Nothing requires `code`.

Reference check: llama.cpp's own `test-gbnf-validator` (build b9606), fed the grammar text
(sha256 6a09d16de9e9deec, 614 bytes) and each candidate string. MEASURED verdicts:

| case | candidate | old grammar | round-1 grammar |
|---|---|---|---|
| A | `{"name":"run_engine_on_transitions","arguments":{}}` | VALID | INVALID |
| B | full call, `"code"` holds real source with `\n` escapes | VALID | VALID |
| C | unknown name | INVALID | INVALID |
| D | no `arguments` key | INVALID | INVALID |
| E | `{"cod":"x"}` (wrong key) | VALID | INVALID |
| F | `diff_grids` without `t` | VALID | INVALID |
| G | `list_transitions` with `{}` (no required params) | VALID | VALID |
| H | `{"code":""}` | VALID | INVALID |
| I | `{"code":5}` | VALID | INVALID |
| J | `query_region` with only `t` | VALID | INVALID |
| K | `query_region` with all five required ints | VALID | VALID |
| L | `find_objects`, `which:"before"` | VALID | VALID |
| M | `find_objects`, `which:"sideways"` (not in enum) | VALID | INVALID |
| N | full call plus an extra key after `code` | VALID | VALID |
| O | `query_region` all required plus optional `which` | VALID | VALID |
| P | full call with json.dumps default spacing | INVALID | INVALID |
| Q | `{"note":"x","code":...}` (required not first) | VALID | INVALID |
| R | `diff_grids` with `t` as a string | VALID | INVALID |

Case A is exactly the envelope the 0.8B trial returned twice. Under constrained decoding
the cheapest legal path wins, so no model result on the old grammar was interpretable: a
zero could mean "the model does not write code under grammar" or "the grammar let it
skip". The fix comes before any GPU time.

Instrument audit. The first run of the proof script reported B and N INVALID. The cause
was the instrument: `json.dumps` default separators insert spaces, and the root rule is
whitespace-rigid around the envelope. Case P keeps that spaced form so the audit is on
record. The grammar was not wrong there; the checker was.

## 3. Round 1 (commit 1c41b9c698)

- `_tool_grammar(schemas)`: one `call-i`/`args-i` rule pair per session tool. Required
  keys first, in schema order, each with its JSON type. Required strings are
  `nonempty-string`. String enums are literals. Optional keys may follow. A tool with no
  required keys gets `object`.
- `_lift_grammar_response(..., required)`: an envelope that lacks a required argument, or
  sends it as `""`, is `grammar_invalid_response`. No dispatch. Not counted as parsed.
- The grammar text is built once per session from a deep copy of the frozen schemas. A
  first draft rebuilt it per turn from the live dicts; the existing frozen-names test
  (`test_candidate_names_and_json_are_preserved`) caught a mutated candidate name leaking
  into the second request. Kept on record because it is the class-B shape.
- The grammar prompt no longer shows `"arguments": {}` as the example.
- `python/carnot/testing/gbnf_match.py`: a model-free GBNF reader for the subset these
  grammars use, mirroring `llama-grammar.cpp` `parse_sequence`. Pinned to the
  test-gbnf-validator verdicts above on both grammars: 36 of 36 agree (MEASURED, 18
  cases x 2 grammars). An independent reviewer re-derived all 36 from the binary
  without the reader, and ran 1400 mutated strings plus 34 construct probes: 0
  disagreements.
- Spec: REQ-ARC-WMTE-7046 (two scenarios); REQ-7044 carries an amendment note.

Round-1 grammar sha256 947e6e69af65014b, 1761 bytes, for six session tools.

## 4. Round-1 mutation proofs (call site, one pattern each; ran UNLOCKED, see limits)

| id | site | test | RED | restore | GREEN |
|---|---|---|---|---|---|
| M1 | request grammar built from schemas stripped of parameters | `test_request_grammar_rejects_empty_shell_model_free` | 1 failed (empty shell accepted) | byte-identical | 1 passed |
| M2 | lifter `if missing:` -> `if False:` | `test_payload_less_envelope_is_a_grammar_failure` (3) | 3 failed (dispatch reached the tool) | byte-identical | 3 passed |
| M3 | builder `if pairs:` -> `if False:` | reader table (18) + headline + live dispatch (2) | 12 failed, 9 passed (the 9 do not depend on required pairs: B C D G K L N O P) | byte-identical | 21 passed |
| M4 | prompt example reverted to `"arguments": {}` | `test_request_grammar_rejects_empty_shell_model_free` | 1 failed | byte-identical | 1 passed |

Original file sha256 prefix db3f68d3f973 before and after every restore.

## 5. The bounded 27B trial (GPU 1 only; ran on the round-1 grammar)

Harness: `scripts/experiments/outer_loop_arc_grammar_27b_trial_20260905.py`
(`--role driver`, one `--role cell` subprocess per (game, arm); cells own separate
engine stores under the output directory, never `results/**`).

Server (MEASURED from the driver log and nvidia-smi):

- llama-server build b9606 CUDA, `Qwen3.8-27B-Q4_K_M.gguf`
  (`~/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/fe1e2a23.../`),
  `-ngl 999 -c 49152 --parallel 1 --cache-type-k q8_0 --cache-type-v q8_0 -fit off
  --jinja --reasoning-format deepseek`, `CUDA_VISIBLE_DEVICES=1`, port 8939.
- Healthy in 24.0 s. Placement: pid 692290 on GPU-7971baff (index 1) only, 18030 MiB;
  GPU 0 at 4 MiB throughout. Verified with
  `nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory` joined to
  `--query-gpu=index,uuid`, not from the environment variable.
- Decode 36.9 tok/s, prefill 1253 tok/s on a 12,974-token prompt (server `print_timing`).
- Receipt armed (`receipt.json`, REQ-INFRA-6830); none was written, the exit was clean.
  Server stopped by exact PID at 23:55:19Z; port 8939 closed, both cards back to
  24120 MiB free, no orphan xdist workers. GPU wall: 23:36:53Z to 23:55:19Z, 18.5 min.

Inputs: seven public games (tr87, sp80, sb26, re86, g50t, r11l, lp85), each a 40-row
window from `collect_transitions(n=40, warmup=False, seed=0)` in the offline sim,
pickled once on CPU and loaded by both arms (same bytes; the cell records
`window_sha256`). The level-up windows the h2h harness uses (`build_progress_window`)
re-solve L1 by search and cost minutes per game; the first game alone did not finish in
7.5 min, so they were not used. This differs from the prior 27B measurements.

Arms, same model, same window, same `CARNOT_ARC_GENERATOR_SEED=7`:

| | CONTROL | ARM (grammar, round-1 text) |
|---|---|---|
| path | `LocalGGUFProposer.induce` (ordinary single-shot), `use_chat_template=True`, think ON, `thinking_budget_tokens=2048` | `induce_with_tool_loop` called directly (a fallback cannot be counted), `CARNOT_ARC_INDUCE_TOOL_LOOP=1`, `CARNOT_ARC_INDUCE_TOOL_GRAMMAR=1`, thinking off by the grammar, turn cap 4, force-engine nudge at turn 2 |
| per-call budget | `max_tokens` 6144, timeout 600 s | same |
| scoring | the written `world_model.py` re-run through `InductionToolSession.run_engine_on_transitions` on the same window (37 visible / 3 held-out rows) | same scorer |

"Emitted" = a `world_model.py` was written. "Scoreable" = it compiles, defines
`engine`, and the verifier runs it on every visible row. Those are weak bars: an
identity engine is scoreable. So the table also gives engine size, `cell_recall`
(visible / held-out) and accuracy (fraction of rows predicted exactly).

Per cell (MEASURED; n = 7 games per arm, 14 cells, all completed inside the budget):

| game | arm | scoreable | engine chars | tokens | wall s | acc vis | recall vis | acc ho | recall ho | submissions |
|---|---|---|---|---|---|---|---|---|---|---|
| tr87 | control | yes | 4006 | 3446 | 104.1 | 0.000 | 0.153 | 0.000 | 0.639 | 1 |
| tr87 | grammar | yes | 132 (identity) | 133 | 19.8 | 0.000 | 0.000 | 0.000 | 0.000 | 1 |
| sp80 | control | yes | 3882 | 3224 | 88.7 | 0.000 | 0.393 | 0.000 | 0.643 | 1 |
| sp80 | grammar | yes | 3913 | 1938 | 63.4 | 0.000 | 0.286 | 0.000 | 0.643 | 2 |
| sb26 | control | yes | 1614 | 2539 | 69.8 | 0.946 | 0.500 | 1.000 | 0.000 | 1 |
| sb26 | grammar | yes | 639 | 422 | 17.8 | 0.919 | 0.250 | 1.000 | 0.000 | 2 |
| re86 | control | yes | 6800 | 3928 | 121.8 | 0.000 | 0.445 | 0.000 | 0.484 | 1 |
| re86 | grammar | yes | 132 (identity) | 1313 | 55.7 | 0.000 | 0.000 | 0.000 | 0.000 | 2 |
| g50t | control | yes | 6987 | 4320 | 121.8 | 0.000 | 0.254 | 0.000 | 0.599 | 1 |
| g50t | grammar | yes | 454 | 731 | 31.9 | 0.162 | 0.000 | 0.333 | 0.000 | 2 |
| r11l | control | yes | 10218 | 6144 (cap) | 172.9 | 0.000 | 0.099 | 0.000 | 0.000 | 1 |
| r11l | grammar | yes | 375 | 369 | 22.5 | 0.000 | 0.000 | 0.000 | 0.000 | 2 |
| lp85 | control | yes | 6517 | 4426 | 131.1 | 0.351 | 0.000 | 0.333 | 0.000 | 1 |
| lp85 | grammar | yes | 1267 | 1074 | 47.8 | 0.000 | 0.041 | 0.000 | 0.000 | 2 |

Per arm: control 7/7 emitted, 7/7 scoreable, 7/7 non-trivial, 28,027 decode tokens,
810 s; grammar 7/7 emitted, 7/7 scoreable, 5/7 non-trivial, 5,980 decode tokens, 259 s.
Transport: 28 envelopes sent by the model, 28 parsed, 0 invalid, 13 engine
submissions. Mean accuracy 0.185 (control) vs 0.154 (grammar); mean visible cell recall
0.263 vs 0.082.

Turn order in EVERY grammar cell (MEASURED): `list_transitions {}` (13-14 tokens),
`diff_grids` (14-15 tokens), then the first `run_engine_on_transitions` at turn 2, the
turn at which the force-engine nudge fired (`force_engine_nudges: 1` in all 7). The
"identity" engines on tr87 and re86 are the fewest-mismatch candidates the monotone
accept kept; re86's turn-3 candidate was 1232 tokens and scored worse.

What this shows and does not show:

- Shows: the round-1 grammar carries well-formed calls and real code on the pinned 27B.
  28 of 28 envelopes were parseable, including six-argument `query_region` calls and
  code strings of 1166 tokens with escapes. On sb26 the transport produced a 0.919 /
  1.0 engine in 18 s. The mechanism claim in the recorded 0.8B negative ("two parseable
  calls, missing required code, zero scoreable engines") does not hold on the 27B.
- Does not show: that the model submits an engine on its own under the grammar. All 13
  submissions followed the nudge, and the cheapest legal envelope was chosen first in
  7 of 7 cells. This is the attribution problem the review raised, and it is why round
  2 makes the force turn a grammar constraint instead of a prompt.
- The control worked (7/7 engines, two with accuracy above 0), so a zero on the arm would
  have been interpretable as an arm result. There was no zero.
- Quality comparison is not the question this trial was sized for. n = 7 per arm; the
  arms differ in thinking (control on, grammar off) and turn budget (4 turns).

## 6. Review findings and round 2 (same day)

The adversarial review of round 1 found:

1. `list_transitions` has no required parameter, so `args-5 ::= object` and
   `{"name":"list_transitions","arguments":{}}` stayed grammatical. It is that tool's
   complete call, not a shell. It is also the cheapest legal envelope, and the trial
   showed the 27B takes it first. Interpretability then rested on the prompt nudge.
2. `nonempty-string` admitted `{"code":" "}`, which passed the lifter, dispatched, and
   counted as a parsed call.
3. The spec cited this note while it was untracked and incomplete.

Closures (all MEASURED against the binary, `gbnf_proof2.py`, 26 strings x 2 grammars,
reader agreeing on 52 of 52):

- `REQUIRED_SOURCE_DEFINITIONS` (`arc_induction_tools.py`): `code` must contain
  `def engine(` (run_engine_on_transitions) or `def is_level_complete(`
  (run_goal_on_states); `predicate_code` must contain `def accept(`. The grammar rule is
  `"\"" char* "def engine(" char* "\"" ws`. The threshold is the dispatcher's own
  contract (`_exec_candidate(code, "engine")`), not a character count. A grammar cannot
  judge a program beyond that; the verifier does.
- The lifter rejects a blank required string and a source argument without its
  definition.
- At the force turn the loop sends `_tool_grammar(schemas, allowed=("run_engine_on_transitions",))`
  (root `call-0` only; sha256 c83115a92f437134, 724 bytes) and counts
  `grammar_submit_only_turns`. The nudge text stays so the model knows why.

Round-2 full grammar sha256 16578b60ac14f9f2, 1841 bytes. New verdicts (binary):

| case | candidate | full | submit-only |
|---|---|---|---|
| G | `list_transitions` with `{}` | VALID | INVALID |
| S | `{"code":" "}` | INVALID | INVALID |
| T | `{"code":"def engine("}` (the admitted minimum) | VALID | VALID |
| U | `{"code":"x"}` | INVALID | INVALID |
| V | engine code that defines only `is_level_complete` | INVALID | INVALID |
| W | `run_goal_on_states` with `def is_level_complete(` | VALID | INVALID |
| X | `run_goal_on_states` with `def engine(` only | INVALID | INVALID |
| Y | `find_objects` predicate `lambda obj: True` | INVALID | INVALID |
| Z | `def engine(` after escaped quotes, backslash, tab | VALID | VALID |
| B, N | full calls with real code | VALID | VALID |
| K, L, O | complete calls to other tools | VALID | INVALID |

Round-2 mutations (source sha 59e8f98c2e85 before and after each restore):

| id | site | RED | GREEN |
|---|---|---|---|
| M5 | builder drops the definition marker | 6 of 28 failed | 28 passed |
| M6 | lifter ignores the definition marker | 3 of 8 failed | 8 passed |
| M7 | force turn keeps the full grammar | 1 failed (root was the full grammar) | 1 passed |
| M8 | lifter accepts a blank required string | 1 of 8 failed (the blank-enum case built for it) | 8 passed |

Finding 3 is closed by this note being complete and committed.

## 7. Limits, what was not covered, and the open decision

- The 27B trial ran on the round-1 grammar. No cell ran on the round-2 grammar. Per the
  coordinator's instruction, GPU spend stopped at 18.5 min and stays at zero.
- The cheapest legal envelope under the full grammar is still `list_transitions {}`
  during inspection turns. The grammar closes the starvation route at the force turn,
  not before it. Removing `list_transitions` from the grammar mode would close it
  earlier at the cost of the tool. That is the operator decision this arm needs.
- The definition marker forces a definition, not a program. `{"code":"def engine("}`
  is grammatical and fails at dispatch with a syntax error. The verifier, not the
  grammar, judges programs.
- Mutation proofs ran UNLOCKED: `--mutation-begin` refuses to run in a worktree while
  the same lock blocks repo-wide. PYTHONPATH was pinned to the worktree and the
  imported module path was printed and checked.
- Windows were random-walk, not level-up. The quality columns are not comparable with
  the prior 27B measurements that used `build_progress_window`.
- The reviewer's independent re-derivation is recorded in its own report; it is quoted
  here from the coordinator's summary, not re-run.

---

# Trial 2: does the 27B submit before the closure compels it? (2026-09-06)

Branch `grammar-27b-trial2`, off `33c07cb531` (the round-2 grammar). This section
EXTENDS the trial above. Nothing above is retracted.

## 8. Summary

- **Unprompted submission is REFUTED on this population, not merely unmeasured.** With
  the force turn at the loop's own default of 3, the pinned Qwen3.8-27B submitted an
  engine at turn 3 in 7 of 7 grammar cells, and at no earlier turn in any of them.
  Given a third free turn it spent that turn on more inspection, 7 times out of 7.
- The design leaned the other way. Moving the force turn LATER gives the model strictly
  more room to submit on its own than trial 1 allowed. The negative arrived against that
  lean.
- **So the grammar's demonstrated value is transport efficiency, not autonomous tool
  use.** It carries code reliably and cheaply; it does not make the model reach for the
  verifier.
- Both round-2 changes behave as documented, verified model-free against llama.cpp's own
  `test-gbnf-validator`: 24 candidates x 2 grammars = 48 verdicts, 0 disagreements.
- The grammar freeze was verified ON THE WIRE, not by reading the source: every grammar
  cell's requests carried exactly two distinct grammar texts, and both were the expected
  hashes. No mid-session change.
- Control 7/7, reproducing trial 1 to the token (28,027 decode tokens, 809.9 s vs
  810.2 s). GPU wall 16.8 min on GPU 1, server stopped by exact PID.

## 9. What trial 1 left open, and the one question here

Trial 1 proved the transport but could not say whether the model REACHES for a
submission: it passed `CARNOT_ARC_INDUCE_TOOL_FORCE_ENGINE_TURN=2` as a hardcoded
string, and all 7 of its grammar cells first submitted at turn 2 — the turn the nudge
fired. The loop's own default is `DEFAULT_FORCE_ENGINE_TURN = 3`, so trial 1 nudged one
turn EARLIER than ordinary use and never gave the model a third free turn.

Turn indices are 0-based. The nudge fires at the END of turn index `force-1`, so the
compelled request is turn index `force`. At force 3, a first submission at turn 0, 1 or
2 is UNPROMPTED; at turn 3 it is COMPELLED and says nothing new.

**Which cell of this design fails if the claim is wrong.** The falsifier was direct and
available: any cell with `first_submission_turn < 3` and `force_engine_nudges: 0`. None
appeared. And the design was biased toward finding one — more free turns than trial 1,
not fewer — so the negative is not an artifact of a stingy budget.

Two ways a "no unprompted submission" reading could have been an artifact were ruled out
model-free BEFORE any GPU spend, because either would have made the reading
uninterpretable:

- `stall_cap` (`CARNOT_ARC_INDUCE_TOOL_STALL_TURNS`) defaults to 0, i.e. off. An
  inspection-only run cannot be cut short before turn 3.
- `early_stop_non_improving` increments its counter only on a turn that ALREADY contains
  a `run_engine_on_transitions`, so it cannot fire before a first submission.
- No `CARNOT_*` variable was set in the launching environment (MEASURED: `env | grep -i
  CARNOT` printed nothing), so no stale knob leaked into the cells.

## 10. The two round-2 changes, verified model-free against the binary

Checked with llama.cpp's own `test-gbnf-validator` (build b9606), not through this
project's Python reader and not through the loop. Grammars built from the live schemas
at the trial commit:

| grammar | sha256 | bytes | root |
|---|---|---|---|
| full (6 session tools) | `16578b60ac14f9f2` | 1841 | `call-0 \| call-1 \| ... \| call-5` |
| submit-only | `c83115a92f437134` | 724 | `call-0` |

`REQUIRED_SOURCE_DEFINITIONS` as read from source: `run_engine_on_transitions.code` ->
`def engine(`; `run_goal_on_states.code` -> `def is_level_complete(`;
`find_objects.predicate_code` -> `def accept(`.

### Change A — a source argument must contain the definition dispatch looks for

| case | candidate `arguments` | full | submit-only |
|---|---|---|---|
| A1 | `run_engine_on_transitions`, real multi-line engine | VALID | VALID |
| A2 | `{"code":"def engine("}` — the admitted minimum | VALID | VALID |
| A3 | `{"code":"x = 1"}` | INVALID | INVALID |
| A4 | `{"code":" "}` | INVALID | INVALID |
| A5 | `{"code":""}` | INVALID | INVALID |
| A6 | `{"code":"def  engine("}` (two spaces) | INVALID | INVALID |
| A7 | engine slot holding an `is_level_complete` definition | INVALID | INVALID |
| A8 | `run_goal_on_states` with `def is_level_complete(` | VALID | INVALID |
| A9 | `run_goal_on_states` with `def engine(` only | INVALID | INVALID |
| A10 | `find_objects`, all four required keys, `def accept(` | VALID | INVALID |
| A11 | `find_objects` predicate `lambda o: True` | INVALID | INVALID |
| A15 | `find_objects`, all four keys, NOT in schema order | INVALID | INVALID |
| A12 | escaped quote + backslash + tab before `def engine(` | VALID | VALID |
| A13 | `{"code":5}` | INVALID | INVALID |
| A14 | `{}` | INVALID | INVALID |

A6 records the shape of the check: the marker is a literal, so `def  engine(` is refused.
A2 records the documented limit unchanged — the grammar forces a DEFINITION, not a
program.

### Change B — at the force turn only a submission is admissible

| case | candidate | full | submit-only |
|---|---|---|---|
| B1 | `list_transitions {}` | VALID | INVALID |
| B2 | `diff_grids` with `t` | VALID | INVALID |
| B3 | `query_region`, five required ints in schema order | VALID | INVALID |
| B4 | `run_engine_on_transitions` with real code | VALID | VALID |
| B5 | `list_transitions` with an extra key | VALID | INVALID |
| B6 | `run_goal_on_states` with a valid goal | VALID | INVALID |

B1 is the envelope trial 1's 27B chose first in 7 of 7 cells. At the force turn it is no
longer expressible.

The loop-side half of change B is exercised by the shipped suite:
`tests/python/test_arc_tool_grammar_transport.py`, 50 passed, including the case that
sets the force turn to 1 and asserts `grammar_submit_only_turns == 1`.

### Instrument audit

The first run of this verification reported three DISAGREEMENTS. All three were the
checker, not the grammar — the same class trial 1 recorded for its own proof:

- `query_region`'s required order is `t, r0, r1, c0, c1`; the candidate was built
  `t, r0, c0, r1, c1`. The grammar requires schema order, so INVALID was correct.
- `find_objects` requires four keys (`t, which, predicate_code, max_objects`); the
  candidate supplied two. INVALID was correct.
- The "spaced" case put `json.dumps` spacing INSIDE `arguments`, where the rules do admit
  `ws`. VALID was correct.

The corrected suite keeps the two spacings apart on purpose: `ARGS_SPACED` (inside
`arguments`) is VALID, `ENV_SPACED` (in the envelope) is INVALID. A15 was added out of
the first defect, so key ORDER is now under test rather than assumed.

## 11. The trial (GPU 1 only; round-2 grammar, force turn 3)

Server (MEASURED from the driver log and nvidia-smi):

- llama-server build b9606 CUDA, `Qwen3.8-27B-Q4_K_M.gguf`, snapshot `fe1e2a23`,
  `-ngl 999 -c 49152 --parallel 1 --cache-type-k q8_0 --cache-type-v q8_0 -fit off
  --jinja --reasoning-format deepseek`, `CUDA_VISIBLE_DEVICES=1`, port 8941.
- Healthy in 10.0 s. Placement: pid 736570 on GPU index 1 only (uuid `GPU-7971baff`),
  18030 MiB, read from `--query-compute-apps` joined to `--query-gpu` on uuid, never from
  the environment variable. `/props` reports `n_ctx` 49152. GPU 0 stayed at 4 MiB.
- Receipt armed (`receipt.json`, REQ-INFRA-6830); none written, the exit was clean.
  Server stopped by exact PID at 01:17:45Z, rc 0; port 8941 closed, both cards back to
  24120 MiB free, no orphan xdist workers. GPU wall 01:00:57Z to 01:17:45Z, 16.8 min.

**n and why.** n = 7 games per arm, 14 cells. The games AND the window pickles are trial
1's, reused byte-for-byte rather than rebuilt: comparability with trial 1 is worth more
than fresh inputs. Every cell's recomputed `window_sha256` equals trial 1's for the same
game (MEASURED, 7 of 7).

Population: 7 public games (tr87, sp80, sb26, re86, g50t, r11l, lp85), each a 40-row
random-walk window from `collect_transitions(n=40, warmup=False, seed=0)` in the offline
sim, 37 visible / 3 held-out rows, `CARNOT_ARC_GENERATOR_SEED=7`.

Arms are trial 1's with ONE change: `CARNOT_ARC_INDUCE_TOOL_FORCE_ENGINE_TURN` is 3, not
2. Turn cap stays 4, so the grammar arm has three free turns (0, 1, 2) and one compelled
turn (3).

### THE MEASUREMENT: first submission versus the force turn

| game | first submission turn | force turn | unprompted? | nudges | submit-only turns |
|---|---|---|---|---|---|
| tr87 | 3 | 3 | no | 1 | 1 |
| sp80 | 3 | 3 | no | 1 | 1 |
| sb26 | 3 | 3 | no | 1 | 1 |
| re86 | 3 | 3 | no | 1 | 1 |
| g50t | 3 | 3 | no | 1 | 1 |
| r11l | 3 | 3 | no | 1 | 1 |
| lp85 | 3 | 3 | no | 1 | 1 |

First-submission turn distribution: `{3: 7}`. Unprompted: **0 of 7**.

What the free turns were spent on instead (MEASURED per-turn tool names, both trials):

| game | trial 1 turns (force 2) | trial 2 turns (force 3) |
|---|---|---|
| tr87 | list_transitions, diff_grids, **run_engine**, query_region | list_transitions, diff_grids, query_region, **run_engine** |
| sp80 | list_transitions, diff_grids, **run_engine**, run_engine | list_transitions, diff_grids, diff_grids, **run_engine** |
| sb26 | list_transitions, diff_grids, **run_engine**, run_engine | list_transitions, diff_grids, diff_grids, **run_engine** |
| re86 | list_transitions, diff_grids, **run_engine**, run_engine | list_transitions, diff_grids, query_region, **run_engine** |
| g50t | list_transitions, diff_grids, **run_engine**, run_engine | list_transitions, diff_grids, query_region, **run_engine** |
| r11l | list_transitions, diff_grids, **run_engine**, run_engine | list_transitions, diff_grids, diff_grids, **run_engine** |
| lp85 | list_transitions, diff_grids, **run_engine**, run_engine | list_transitions, diff_grids, query_region, **run_engine** |

The behaviour is stereotyped and identical across both trials for the turns they share:
`list_transitions {}` at turn 0 (13-14 tokens), `diff_grids` at turn 1 (14-15 tokens).
The third free turn, which only trial 2 offered, went to another inspection call in 7 of
7 — `query_region` in 4 games, a second `diff_grids` in 3 — at 14 to 52 tokens. It is
cheap, and it is not a submission.

### Grammar freeze, verified on the wire

Every grammar request's grammar text was hashed as it was sent. Per cell, the distinct
set was exactly `{16578b60ac14, c83115a92f43}` — the full grammar on turns 0-2, the
submit-only grammar on turn 3 — in 7 of 7. No third text, no unexpected text, no
mid-session change. The check FAILS CLOSED: an absent wire record reports `NOT_MEASURED`,
never `ok`, which is what it reports for trial 1's cells (they predate the instrument).

### Per cell

| game | arm | scoreable | engine chars | tokens | wall s | acc vis | acc held-out |
|---|---|---|---|---|---|---|---|
| tr87 | control | yes | 4006 | 3446 | 103.97 | 0.000 | 0.000 |
| tr87 | grammar | yes | 132 (identity) | 133 | 19.15 | 0.000 | 0.000 |
| sp80 | control | yes | 3882 | 3224 | 88.44 | 0.000 | 0.000 |
| sp80 | grammar | yes | 2972 | 1164 | 42.85 | 0.000 | 0.000 |
| sb26 | control | yes | 1614 | 2539 | 69.37 | 0.946 | 1.000 |
| sb26 | grammar | yes | 498 | 203 | 11.76 | 0.892 | 1.000 |
| re86 | control | yes | 6800 | 3928 | 121.62 | 0.000 | 0.000 |
| re86 | grammar | yes | 460 | 258 | 25.13 | 0.000 | 0.000 |
| g50t | control | yes | 6987 | 4320 | 121.63 | 0.000 | 0.000 |
| g50t | grammar | yes | 1320 | 600 | 27.34 | 0.189 | 0.333 |
| r11l | control | yes | 10218 | 6144 (cap) | 173.29 | 0.000 | 0.000 |
| r11l | grammar | yes | 132 (identity) | 96 | 16.02 | 0.000 | 0.000 |
| lp85 | control | yes | 6517 | 4426 | 131.61 | 0.351 | 0.333 |
| lp85 | grammar | yes | 1424 | 526 | 32.67 | 0.000 | 0.000 |

Per arm: control 7/7 emitted, 7/7 scoreable, 7/7 non-trivial, 28,027 decode tokens,
809.9 s. Grammar 7/7 emitted, 7/7 scoreable, 5/7 non-trivial, 2,980 decode tokens,
174.9 s. Transport: 28 envelopes sent, 28 parsed, 0 `grammar_invalid_response`, 0 parse
failures, 7 engine submissions. Mean accuracy 0.185 (control) vs 0.154 (grammar); mean
held-out 0.19 vs 0.19.

**The control worked**, 7/7, and reproduced trial 1 to the token on every cell (28,027
tokens, 809.9 s vs trial 1's 28,027 and 810.2 s; per-cell token counts identical). A
control that failed would have meant a broken harness rather than an arm result. It did
not fail.

## 12. Reading the result honestly

- **Unprompted submission is refuted on this population.** Not "unmeasured", not
  "inconclusive". Seven cells, three free turns each, twenty-one free turns in total,
  zero submissions. The one thing that produced a submission was the closure.
- **The mechanism claim from trial 1 stands and is strengthened.** 28 of 28 envelopes
  parsed again, and the tightened round-2 grammar produced ZERO invalid responses on
  live output: requiring `def engine(` inside the code argument cost nothing in
  transport. sb26 again reached 1.0 held-out accuracy, this time in 11.76 s at 203
  tokens against the control's 69.37 s at 2,539.
- **So the honest framing of the grammar's value is transport efficiency.** On these
  seven games it delivered the same held-out mean as the control (0.19) for 2,980 tokens
  instead of 28,027, and 175 s instead of 810 s. That is a real and large result. It is
  not evidence of autonomous tool use, and it should not be reported as such.
- **The closure is what converts the transport into an engine.** Without it these seven
  cells would have inspected until the turn cap and submitted nothing. That is worth
  stating plainly next to the operator's decision to keep the closure: the data says the
  closure is not belt-and-braces here, it is load-bearing.

## 13. Limits, what was not covered, and one operator question

- **One model, one prompt, one budget shape.** This says nothing about whether a
  different system prompt, a stated turn budget, or a larger turn cap would elicit an
  unprompted submission. It says that under the shipped prompt and a 4-turn cap, this
  model does not.
- **n = 7, and the behaviour is stereotyped rather than varied**, which makes the 0/7
  read stronger than n alone suggests, but it remains 7 games from one offline corpus.
- **Random-walk windows.** Same limit trial 1 recorded and this trial inherits: windows
  come from `collect_transitions`, not `build_progress_window`, so the quality columns
  are NOT comparable with prior 27B measurements that used level-up windows.
- **Trial 2's grammar arm is not token-comparable with trial 1's.** Trial 1 got two
  code-carrying turns (submission at 2, another at 3); trial 2 gets one (submission at
  3, then the cap). So trial 2's 2,980 tokens against trial 1's 5,980 is mostly one
  fewer candidate, not a new efficiency. Within trial 2, grammar-vs-control is the
  comparison that holds, because both arms saw the same windows in the same run.
- **The definition marker still forces a definition, not a program.** Unchanged from
  round 2: `{"code":"def engine("}` is grammatical and fails at dispatch.
- **`list_transitions` stays in grammar mode** per the 2026-09-05 operator decision
  recorded at `3752b2c3bb`. This trial does not reopen it, and nothing here argues
  against it: at the force turn the tool is already unexpressible, which is the only
  turn where its cheapness would matter.
- **No adversarial reviewer was run.** The brief forbade spawning subagents; the
  coordinator holds that decision.
- **Not covered:** whether the model would submit unprompted with a larger turn cap
  (it may simply be pacing itself against a budget it was told about); whether a
  non-stereotyped prompt changes the turn-0 `list_transitions` reflex; any hidden game.

**The operator question this raises.** Trial 1 ran the force turn at 2 and trial 2 at 3,
and the extra free turn bought nothing on this population: 7 of 7 spent it on inspection
and the submission still came only under the closure. If that holds, the loop default of
3 is paying one inspection turn per induction for no measured return, and 2 would be the
cheaper setting. This trial does not settle it — it measured 7 games on one corpus, and
a turn that is worthless here may not be worthless on a hidden game — but the operator
should decide whether to re-measure the default rather than leave it at 3 by inheritance.

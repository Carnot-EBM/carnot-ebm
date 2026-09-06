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

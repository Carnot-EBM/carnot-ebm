# Selfparse Tool-Call Transport: Implementation and Gate Measurement (2026-08-28)

Implements the two builds the 2026-08-28 design note
(`arc-context-frame-compression-and-callable-tools-2026-08-28.md`) and its
appended CORRECTION section directed, in order. The object table was NOT
touched (the correction withdrew that recommendation; see "Not built" below).

## 1. Goal-prompt RLE fix (REQ-ARC-WMTE-6740)

`_goal_only_prompt` rendered `previous_level_complete_grid` through `to_ascii`
where `_rle_grid` was available and already used by `induce_prompt`. Fixed:
the win block now renders `_rle_grid` plus a two-sentence encoding
explanation. Lossless either way.

Measured with the pinned generator's own GGUF tokenizer
(`llama_cpp.Llama(vocab_only=True)` on `Qwen3.8-27B-Q4_K_M.gguf`), real
reset frames from the offline arcade (`collect_transitions(game, n=1,
seed=5900)`), full `_goal_only_prompt` output, think mode default:

| Game | Before | After | Saved |
|---|---|---|---|
| tu93 (64x64) | 4,309 | 1,928 | 2,381 |
| ft09 (64x64) | 4,309 | 2,995 | 1,314 |

The grid-level delta on tu93 is the reviewer's 2,450 exactly; the ~69-token
difference is the added encoding explanation. The ft09 number confirms the
reviewer's scope limit (game-shaped RLE output; ~1,383 predicted at the grid
level). Scope limit confirmed unchanged: the block fires only when
`previous_level_complete_grid` is not None, so goal calls before the first
level-up save nothing.

## 2. Selfparse transport (REQ-ARC-WMTE-6730)

New env value `CARNOT_ARC_INDUCE_TOOL_LOOP=selfparse` (default unreachable;
"", "1", and "repair" behave byte-identically to before):

- `_post_chat` sends NO `tools` / `tool_choice` fields (the measured HTTP 400
  fires only on a request carrying `tools`).
- `render_tool_schemas_for_prompt()` carries the schemas as prompt text,
  rendered from `TOOL_SCHEMAS` so prompt and dispatch cannot drift.
- `parse_xml_tool_calls()` lifts the model's Qwen3-coder
  `<tool_call><function=NAME><parameter=K>` XML, scanning only text after the
  last `</think>`, with schema-typed argument coercion, and never dispatches a
  length-truncated block.
- Results return as user-side `<tool_response>` blocks (the Qwen3-coder
  convention); no tool-role message and no assistant `tool_calls` field ever
  reach the chat template.
- `induce()` enters the loop for "selfparse" exactly as for "1"; "repair"
  still enters only through the recall-gated resample.

Cross-backend spot check (the design note's protocol step 2): the parser was
run over the STORED vLLM emission (`offplay_out5/offline_play.json`,
`tool_transport_probe.control_with_parser_flags.content_head` — the exact
string hermes could not lift): 1 parsed / 1 seen / 0 unparsed, name and code
argument exact. Pinned as SCENARIO-ARC-WMTE-6731.

## 3. The pre-registered gate, measured

Protocol (design note section 5): 20 real first-turn induce requests through
`induce_with_tool_loop`, local backend, 5 public games (tu93, lp85, ft09,
vc33, ar25) x 4 seeds (0-3), 25 offline-collected transitions per request,
think ON, turn cap 1, CUDA llama-server on GPU 1 (`n_ctx=49152` — the
pooled 106,496 default needs 26.6 GiB and the guard declines the whole
24 GiB card; single sequential stream needs no pool).

Gate, decided in advance: PASS = >=80% of responses attempt a tool call AND
>=95% of attempts parse to an executed dispatch.

**GATE: PASS, at ceiling on both axes.**

| Metric | Measured | Pre-registered threshold |
|---|---|---|
| Attempt rate | 20/20 = 100% | >=80% |
| Parse-to-dispatch | 20/20 blocks = 100% | >=95% |
| Dispatched calls | 20 (0 name/args failures) | - |

Per-request: every one of the 20 responses attempted exactly one tool call,
every block parsed, every dispatch executed. Wall clock 85.8-157.7s per
request (median 87.9s; the max is the first request, which includes prompt
warm-up); decode 3,088-3,102 tokens per turn (the 3,072 per-turn think
budget binds, plus the call text). Raw rows:
`selfparse_gate_results.jsonl` / `selfparse_gate_summary.json` in the
session scratchpad, and the summary is quoted in full in the changelog
entry for this date.

**Honest scope limit on the live sample.** All 20 live first-turn calls were
`list_transitions` -- the zero-argument index tool the instructions say to
call first. So the LIVE run exercises attempt + parse + dispatch on the
simplest call shape only. The code-carrying, multi-parameter shapes are
covered by the stored vLLM emission (`run_engine_on_transitions` with a
`code` parameter -- parsed 1/1) and by unit fixtures
(SCENARIO-ARC-WMTE-6731/6733), not yet by live multi-turn traffic. The
next measurement that matters is the resumed holdout-equalized A/B
(`results/holdout_equalized_ab_20260820`, resumable per its stop note) with
`CARNOT_ARC_INDUCE_TOOL_LOOP=selfparse`, which drives full multi-turn loops
and will exercise every call shape under the same counters.

Backend note: the local gate ran against the CUDA llama-server build (the
dev twin of the scored path). The scored vLLM backend's emission shape is
covered by the stored-probe spot check above; the emission format is a
property of the model, which both backends serve as the same Qwen3.8-27B.

## Not built (deliberately)

- **Fetch-on-demand object table.** The design note's own largest
  recommendation, withdrawn by the appended correction: the 4,667-token table
  is default-ON because of a measured held-out win (+0.0720 change_fidelity,
  p=0.0192, 19/20 discordant). Removing it to save tokens would un-buy that
  result. Untouched; any change there is an A/B holding change_fidelity flat.
- **Kernel flip.** `submission_kernel/main.py` still sets
  `CARNOT_ARC_INDUCE_TOOL_LOOP=repair`. Moving the scored path to selfparse is
  gated on this note's measurement plus the resumed holdout-equalized A/B
  (quality), not on transport alone.
- **A `repair-selfparse` combination.** "repair" and "selfparse" are mutually
  exclusive values of one env var by construction. If repair draws should use
  selfparse transport, that is a fourth value with its own measurement.

## Reproduction

- Token measurement: `scratchpad selfparse/measure_goal_prompt.py` pattern —
  `collect_transitions(game, 1, seed=5900)`, `_goal_only_prompt`, tokenize
  with the pinned GGUF (`vocab_only=True`, `add_bos=False, special=False`).
- Gate: set `CARNOT_ARC_E3_DIR` to a scratch dir BEFORE import,
  `CARNOT_ARC_INDUCE_TOOL_LOOP=selfparse`, `CARNOT_ARC_INDUCE_TOOL_TURNS=1`,
  `CARNOT_ARC_INDUCE_N_CTX=49152`, `CARNOT_ARC_GENERATOR_CUDA_GPU=<free 3090>`;
  suppress `_write_world_model` and the goal-fallback `generate` in a
  subclass so each request costs exactly one model call; drive
  `induce_with_tool_loop(prop, game, trans, cell)` and read
  `last_tool_loop_stats`.

## Cross-references

- `docs/research-notes/arc-context-frame-compression-and-callable-tools-2026-08-28.md`
  — the design note + correction this implements.
- `openspec/capabilities/arc-world-model-trust-energy/spec.md`
  REQ-ARC-WMTE-6730 / REQ-ARC-WMTE-6740.
- `tests/python/test_arc_induction_selfparse_transport.py` — 12 tests, 8
  mutations each RED then restored byte-identical.
- `docs/research-notes/qwen38-compaction-forward-plan-2026-08-20.md` — Move 2
  Step A is the server-side sibling (`qwen3_xml`, Kaggle-quota-gated).

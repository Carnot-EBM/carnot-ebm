# Context/Frame Compression and Callable Tools for the Live ARC Agent (2026-08-28)

Design note only. No code under `python/carnot/agentic/` or `results/**` was
changed. New measurements in this note were taken with `.venv/bin/python` on
this checkout; each one carries its reproduction command. Claims taken from the
briefing or from prior notes are marked as such, with their source.

The operator's problem statement: the live agent simulates the game "using pure
model thinking, which appears to be working but is just too expensive." Asked
for: (a) context and frame compression, including video-codec-style ideas, and
(b) callable functions (MCP-style tools) so the model reasons less and calls
more.

The one-paragraph answer: most of this system already exists in the repo,
default OFF, and the binding blocker is not design — it is tool-call TRANSPORT
on the scored vLLM backend. Frame compression is ~60% shipped (run-length
keyframe + per-transition deltas, `arc_executable_world_model.py`). A
tool-calling induction loop with five tools, per-turn think budgets, and
mechanical context compaction exists (`arc_induction_tool_loop.py`,
`arc_induction_tools.py`, `arc_induction_compact_state.py`), but the scored
vLLM server rejects tool-bearing requests (HTTP 400) and no parser lifts the
model's tool-call XML. The cheapest decisive next step is an agent-side
tool-call parser, testable locally, that removes the server dependency
entirely. Section 5 specifies that experiment.

## 0. Where the budget actually goes (anchor numbers)

| Quantity | Value | Source | Confidence |
|---|---|---|---|
| Share of decoded characters in the reasoning channel | 97.64% (8,456,727 of 8,661,106 chars, 42 completions; 2 of 42 were `reasoning_only` — full spend, nothing usable) | briefing (REQ-ARC-WMTE-6710 channel accounting, `arc_executable_world_model.py:6457`) | given; CHARACTERS, not tokens; per-channel token split unmeasured |
| Median decode per induction call | 62,490 tokens; max completed 83,444; right-censored draws >= 100,988 | `arc_executable_world_model.py` (comment near `_INDUCE_DEFAULT_MAX_TOKENS`); corroborated in `docs/research-notes/qwen38-compaction-forward-plan-2026-08-20.md` §2 | measured (prior) |
| Decode vs prefill cost on the scored card | decode ~15.4x prefill | `arc_induction_tools.py:6` docstring | measured (prior); I did not re-derive |
| Worst-case induce prompt, shipped defaults | 22,352 tokens (64x64 grid, 25 transitions, object table ON; gemma tokenizer, 2026-08-08) | `arc_executable_world_model.py` `_INDUCE_WORST_CASE_PROMPT_TOKENS` block (~line 4380) | measured (prior) |
| Real tu93 induce prompt, current defaults | 8,678 tokens total | measured in this note, §1 | measured (new) |
| Think-off effect on induction quality | held-out 0.100 vs 0.3727 (worse) | briefing (exp6199) | given |
| Per-slot serving cost | 34.00 KiB/token KV + 149.6 MiB fixed; K=12 recommended | briefing (2026-08-26 correction) | given |
| LLM share of wall clock per game (offline play v3) | 99.8-99.9% | `qwen38-compaction-forward-plan-2026-08-20.md` §2 table | measured (prior) |

Consequence for prioritization: the completion side (~62k decode, 97.6% of it
reasoning) is 3-7x larger than the whole prompt. Frame compression attacks the
smaller term. It still matters — prompt tokens are re-prefilled every tool-loop
turn and occupy KV for the life of the stream — but the mechanism that bounds
the LARGE term is the tool loop's per-turn caps, and think-off is measured
worse, so "just reason less" is not available. Every road below runs through
making the tool loop live on the scored path.

Note also WHERE the LLM appears at all: only in induction, refactor, and
goal-induction calls (`E3AgentPolicy._induce_and_plan`,
`arc_competition_agent.py:7507`). Action selection, search, and planning are
already mechanical. "Functions the model can call" therefore means functions
inside those induction-family calls, not per-action tools.

## 1. Frame representation: measured cost, shipped state, ranked next steps

### 1.1 What a 64x64 frame costs today, measured

Tokenizer: the pinned generator's own GGUF
(`unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf`) via
`llama_cpp.Llama(vocab_only=True)`, per the CLAUDE.md GGUF tokenizer rule.
Frames: real reset-state frames from the offline arcade
(`arc_solver_kit.offline_arcade()`, public games tu93 and lp85, `detect_cell`
returned 1 for both, so logical = raw 64x64). Reproduction: the script inline
in this note's session; the arithmetic is one call each to `to_ascii`,
`_rle_grid`, `_rle_delta`, `_rle_delta_compact`
(`python/carnot/agentic/arc_executable_world_model.py:252,350,2718,2748`) and
`llm.tokenize`.

| Encoding | tu93 (64x64) | lp85 (64x64) | Notes |
|---|---|---|---|
| `to_ascii` (one char/cell) | 4,159 tokens | 4,159 tokens | tokens == chars exactly: this tokenizer emits 1 token per digit char in grid context |
| `_rle_grid` (shipped keyframe form) | 1,709 tokens | 1,857 tokens | lossless; 2.2-2.4x smaller |
| `_rle_delta` per transition (tu93, 1-20 changed cells) | 8-82 tokens | — | lossless per-transition |
| `_rle_delta_compact` per transition | 10-78 tokens | — | shipped delta form; prior lp85 measurement: 8 large deltas = 5,992 tokens vs 9,308 raw (`:2748` docstring) |

A real, current-defaults tu93 `induce_prompt` (25 transitions collected at
seed 5900 via `collect_transitions`, object table ON) decomposes as:

| Block | Tokens | Share |
|---|---|---|
| whole prompt | 8,678 | 100% |
| transitions block (1 RLE keyframe + 25 deltas + labels) | 3,342 | 39% |
| object-structure table (`objects_block`) | 4,667 | 54% |
| of which the RLE keyframe alone | 1,709 | 20% |

So on tu93 the OBJECT TABLE, not the frames, is the largest prompt block —
larger than all 25 transitions together. On the recorded lp85 worst case the
table added +2,141 tokens (17,930 -> 20,071 at k=8; `:4380` block). The
per-game share varies; the table is always additive.

### 1.2 The video-codec analogy is already implemented

The shipped prompt IS keyframe + deltas: one full `_rle_grid` I-frame (the
initial layout), then per-transition P-frame deltas (`_rle_delta_compact`),
plus the win-state grid as a second I-frame when observed
(`_transitions_block`, `:2903`). This shipped after exp5593's context overflow
(18,355 tokens for an 8-transition ASCII window against a 13,824-token budget;
`_rle_grid` docstring). Do not re-propose it.

One accepted information loss already in the shipped scheme, stated so nobody
rediscovers it: transitions shown are a SAMPLE (grid-changing preferred, a
couple of no-ops). Each delta is relative to its own unshown before-grid, so
the model cannot reconstruct every transition's full before-state unless the
sample happens to chain from the keyframe. This is deliberate (data-starvation
in the other direction was the measured failure, REQ-ARC-FCP-5699-22) and
`query_region` / `list_transitions` exist precisely to backfill it on demand.

### 1.3 Ranked remaining moves (tokens saved x risk)

1. **Fetch-on-demand object table** (largest, depends on tool loop). When the
   tool loop is live, drop the static `objects_block` from the prompt and
   expose it as a `find_objects(t, frame)` tool (§2). Saves 4,667 tokens on
   tu93 (54% of that prompt), ~2,141 on the lp85 worst case. Information lost:
   none — same data, pulled instead of pushed. Risk: the model never fetches
   it and induces from pixels alone; that is the same starvation class the k=8
   default had, so gate the change on the §5 experiment's tool-usage rate.
   Cannot ship before the tool loop's transport works.
2. **Fix the one remaining ASCII full-grid render** (small, immediate,
   independent of everything else). `_goal_only_prompt` still renders
   `previous_level_complete_grid` via `to_ascii`
   (`arc_executable_world_model.py:8125`) — 4,159 tokens where `_rle_grid`
   costs 1,709 on a 64x64. Saving: ~2,450 tokens per goal-only call on
   64x64 games. Information lost: none (lossless either way); the prompt must
   explain the encoding, which the induce prompt text already does — reuse it.
   Risk: low; the model already reads this RLE form everywhere else in the
   same prompt family.
3. **Object-level deltas ("motion vectors")** (moderate, unmeasured).
   `_object_delta_perception_block` exists as an opt-in
   (`CARNOT_ARC_OBJECT_DELTA_PERCEPTION`). "Component 3 moved (0,+6)" is both
   smaller than cell runs for moving sprites and closer to the rule the model
   must induce. Information lost: exact cell values when segmentation is
   wrong — and perception is the measured binding constraint of this agent, so
   a wrong segmentation actively misleads induction. Ship only as an ADDITION
   the model can cross-check against lossless deltas, never as a replacement.
4. **Stable prompt prefixes for vLLM prefix caching** (cheap check,
   unverified). If the pinned vLLM wheel has automatic prefix caching active,
   a byte-stable instruction+keyframe prefix across retries and tool turns
   shares KV blocks between streams. I did not verify whether the pinned wheel
   enables it or whether the launch config (`_ensure_vllm_server`, `:6928`)
   would need a flag. Unmeasured; one `/metrics` read on the next scored-shape
   run answers it.
5. **Palette / dictionary remapping: NOT worth doing** (negative finding).
   Measured above, tokens == chars for digit grids — every cell is already one
   token. Remapping colors to letters changes nothing; two-digit colors only
   appear inside RLE values (`14x3`), already compact. Any denser packing
   (base64, hex nibbles) makes the MODEL decode the frame mentally, which
   spends the reasoning channel this whole effort is trying to shrink. Reject.

### 1.4 The KV / concurrency conversion (constraint 4)

At 34.00 KiB/token, per stream: 1,000 prompt tokens = 33.2 MiB of KV.
Worked examples (arithmetic, not measurement):

- Object table (tu93): 4,667 tokens = 155 MiB per stream — slightly more than
  one whole slot's 149.6 MiB fixed cost. At K=12, carrying it in every stream
  costs 1.86 GiB of KV.
- Shipped worst-case prompt: 22,352 tokens = 742 MiB per stream.
- A median single-shot induction decode (62,490 tokens) grows the stream's KV
  by a further 2.03 GiB before it completes.
- The tool loop under compaction bounds peak context near ~25k tokens
  (measured on-arm peaks 22,916-27,245; forward-plan §5 table) = ~830 MiB per
  stream — a ~2.5x per-stream reduction against a completed single-shot draw.

Whether a given saving buys exactly +1 concurrent slot depends on the pool
arithmetic at the moment of admission; the conversion factor above is the
honest, portable statement.

## 2. The tool surface

### 2.1 MCP viability at scored time: answered, and the repo already answered it

The scored eval runs offline in a Kaggle kernel
(`scripts/kaggle/submission_kernel/main.py`); code arrives as an attached
dataset. An MCP server COULD technically run there (a localhost stdio
subprocess needs no internet), but it would add JSON-RPC serialization, a
subprocess, and a dependency (`fastmcp`) that is not in the kernel image — for
tools that live in the SAME Python interpreter as the agent. The repo already
codified the right call: "MCP AS VOCABULARY, NOT TRANSPORT"
(`arc_induction_tools.py` module docstring). Tools are plain functions plus
JSON schemas, called in-process by the loop; `register_mcp_tools`
(`arc_induction_tools.py:582`) exposes the same registry over FastMCP for DEV
use only. This note endorses that split. Proposing an MCP server on the scored
path would be negative value.

The REAL transport problem is between the model and the harness, not between
the harness and the tools:

- The scored vLLM launch (`arc_executable_world_model.py:6928`) passes no
  `--enable-auto-tool-choice` / `--tool-call-parser`. Measured (offline-play
  v5 probe, `/home/ianblenke/.claude/jobs/ad0c053d/tmp/offplay_out5/offline_play.json`,
  `tool_transport_probe`): a request carrying `tools` gets HTTP 400.
- With the flags plus `--tool-call-parser hermes`: request accepted, the model
  emits a well-formed tool call as TEXT, `tool_calls_lifted: false`. The
  captured emission (verbatim):

  ```
  ...</think>

  <tool_call>
  <function=run_engine_on_transitions>
  <parameter=code>
  def engine(grid, action, data): return grid
  </parameter>
  </function>
  </tool_call>
  ```

  This is the Qwen3-coder XML convention; the pinned wheel registers
  `qwen3_xml` and `qwen3_coder` parsers, both untested here (the trial is
  blocked on Kaggle quota — `avo-adaptation-for-local-generator-2026-08-21.md`).
- The loop consumes ONLY server-lifted `tool_calls`
  (`arc_induction_tool_loop.py:566`); unparsed emissions are counted at `:646`
  and never executed. So today, on the scored backend, the tool loop — and the
  compaction that only matters inside it — executes zero tools. The kernel
  even sets `CARNOT_ARC_INDUCE_TOOL_LOOP=repair` (`main.py:225`), routing
  repair draws into a loop that cannot act.

### 2.2 Tools that already exist (default OFF; do not re-invent)

`arc_induction_tools.py:449` `TOOL_SCHEMAS`, dispatched in-process, all
bounded (MAX_MISMATCHES_REPORTED=5, MAX_DIFF_CELLS=200, MAX_REGION_CELLS=400,
MAX_GOAL_PROBE_GRIDS=24), with a memorization AST scan and a 2-3 transition
holdout built into the mismatch report:

1. `run_engine_on_transitions(code)` — runs a candidate engine on the window,
   returns bounded mismatches + visible/held-out split + memorization flags.
2. `query_region(t, frame, r0, c0, r1, c1)` — raw cells of a sub-rectangle.
3. `diff_grids(t)` — changed cells of one transition.
4. `run_goal_on_states(code)` — `is_level_complete` over <=24 observed states.
5. `list_transitions()` — the window index.

The one paired measurement so far (holdout-equalized A/B, deliberately stopped
2026-08-20, `results/holdout_equalized_ab_20260820/STOPPED_2026-08-20.json`):
tu93 single-shot 4,238 s -> holdout 0.0, memorizing; tool arm 598 s mean ->
holdout 0.625/0.75/0.5, non-memorizing. n=2 paired cells — a direction signal,
not a ranking claim. The A/B is resumable per cell.

### 2.3 Proposed additions (signatures, what each replaces, plug point)

All additions plug into `InductionToolSession` + `TOOL_SCHEMAS` +
`dispatch_tool` in `arc_induction_tools.py`; the loop iterates the schema list
and needs no change. Live-path reachability:
`E3AgentPolicy._induce_and_plan` (`arc_competition_agent.py:7507`) ->
`LocalGGUFProposer.induce` (`arc_executable_world_model.py:8181`) -> the
tool-loop hook at `:8197`; the repair route enters via
`_maybe_recall_gated_resample` (`arc_competition_agent.py:8614`). The offline
dev twin reaches the same proposer through `scripts/arc_loop_solve.py`.

6. **`probe_goal_reachability(code: str, max_nodes: int = 2000) ->
   {planned, plan_len, nodes_expanded, frontier_nonempty, plan_head}`**
   Wraps `plan_in_model` (`arc_executable_world_model.py:8577`) over the
   candidate's own `engine` + `is_level_complete`, start grid = the window's
   latest grid. What it replaces: the model reasoning about whether its rules
   ADMIT a path to the goal — today an engine can pass held-out verification
   and still fail planning, and the agent only discovers this after the full
   ~62k-token spend (the briefing's measured case: the 20,000-node probe
   exhausted with a non-empty frontier on a just-verified model). Inside the
   loop, one tool round converts that post-hoc failure into an in-generation
   fix signal. Cost coupling: at the briefing's measured 15.6 ms/engine-call,
   2,000 nodes is 30+ s per probe — only affordable together with §3's
   primitives (0.656 ms -> ~1-7 s).
7. **`simulate_actions(code: str, actions: list[int], start: str = "latest")
   -> {steps: [{changed_cells, level_complete}], final_grid_rle}`** (<=32
   actions). What it replaces: multi-step mental rollout — the single largest
   identifiable reasoning-channel activity ("the model spends it MENTALLY
   SIMULATING grid transforms", `arc_induction_tools.py` docstring). Distinct
   from tool 1: that one grades against OBSERVED transitions; this one rolls
   forward into UNOBSERVED futures, which is what "simulating the game in its
   head" actually is.
8. **`find_objects(t: int, frame: str) -> component table`** (<=40 rows:
   color, bbox, size, centroid; reuses `objects_block`'s existing
   componentry). What it replaces: the static 4,667-token object table (§1.3
   move 1) and the model re-deriving object structure in the think channel.
9. **`note(text: str)` / `notes() -> list[str]`** (bounded, e.g. 10 x 200
   chars) — a hypothesis ledger persisted through compaction by extending the
   carried-state message (`arc_induction_compact_state.py`, EvidenceLedger /
   `build_carried_state`). What it replaces: the model re-deriving rejected
   hypotheses after compaction amputates old turns. This is NVIDIA AVO's
   credited "persistent memory" component in bounded, mechanical form. Lowest
   priority: smallest expected win, wholly unmeasured.

Not proposed: per-action tools at act time (the act-time path is already
LLM-free), and a "read game source" tool (forbidden for hidden games,
CLAUDE.md source-reading directive 2026-06-29).

## 3. The engine-speed generalisation: recommend a primitives library

The question: the LLM writes `world_model.py` engines whose hot loop is a
Python double scan over 3x3 blocks (the briefing's measured case: 15.6
ms/call, `sliding_window_view` rewrite 16.0x faster, zero mismatches on 41
real grids). Three candidate fixes; recommend ONE.

**Recommendation: a vetted primitives module the generated code imports —
`python/carnot/agentic/arc_wm_primitives.py` — steered by one prompt
paragraph.** Contents, each vectorized and unit-tested once:

```python
find_sprites(grid, h=3, w=3, pattern=None)   # the measured 16x case
connected_components(grid, ignore=())        # labels, bboxes, sizes
move_region(grid, bbox, dr, dc, fill=0)
flood_fill(grid, r, c, color)
find_color(grid, color)                      # np.argwhere wrapper
region_equals(grid, r0, c0, block)
```

Delivery is three contained edits (none made in this note): (a) the module
ships inside the same `carnot-agent-code` dataset the kernel already attaches
(`submission_kernel/main.py:24`) — no new dependency; (b) `induce_prompt`'s
"Use only numpy + stdlib" line (`arc_executable_world_model.py:~3460`) gains
"plus these provided helpers", with the signature list; (c) nothing in the
loaders needs to change — `load_engine` imports the file as a real module
(`:2639-2657`) and the tool loop's `_exec_candidate` exec permits imports
(`arc_induction_tools.py:131-143`), so `from carnot.agentic.arc_wm_primitives
import find_sprites` works in both today.

Why this over the alternatives:

- **Steering induction toward vectorised code (rejected as primary).**
  Vectorized numpy is HARDER for a 27B to write correctly than loops; failures
  surface as mismatches and more refactor rounds, and the text-refactor round
  is already measured destructive (0 improvements in 84 cells,
  `arc_llm_reinduction.py:1223-1229` via the forward-plan note). The
  instruction also spends think tokens on code style instead of game rules.
- **Post-induction optimisation pass (rejected).** An LLM pass costs another
  ~62k-token decode with semantic-change risk; a mechanical auto-vectorizer is
  a compiler project; a generic memoization wrapper cannot find helper
  boundaries in arbitrary generated code. All three cost more than shipping
  the six functions above.
- **Primitives win on BOTH budget axes at once.** Calls to a named helper are
  fewer decode tokens than an inline double loop, and the helper is fast by
  construction — which is what makes `probe_goal_reachability` (tool 6) and
  bigger `plan_in_model` budgets affordable (at equal wall clock, 20,000
  nodes -> ~320,000, the briefing's arithmetic). The names also give the
  prompt and the object table a shared vocabulary.

Honest caveats: the win only lands if the model USES the helpers (measurable:
grep induced engines for the import — make it part of the §5 follow-on), and a
prompt-vocabulary change can move induction quality in either direction
(unmeasured until A/B'd).

## 4. Confidence ledger

| Claim | Status |
|---|---|
| Token counts in §1.1 tables | measured this session, Qwen3.8-27B GGUF tokenizer, commands stated |
| tu93 prompt decomposition (8,678 / 3,342 / 4,667 / 1,709) | measured this session; one seed (5900), one game — shares vary by game |
| 22,352 worst case | prior measurement, gemma tokenizer; Qwen-vs-gemma ratio 1.002 on this prompt shape (`:4380` block); not re-measured under Qwen3.8 |
| vLLM 400 / hermes-no-lift / emission shape | prior measurement, offplay_out5 probe, quoted verbatim |
| `qwen3_xml` / `qwen3_coder` parser behavior | UNMEASURED (Kaggle-quota-blocked) |
| Tool arm quality/wall-clock advantage | n=2 paired cells; direction only |
| 15.6 ms engine / 16.0x rewrite / probe exhaustion / 97.64% / exp6199 / KV constants | given by the briefing; not re-derived |
| KV conversions in §1.4 | arithmetic from given constants, not measurement |
| vLLM prefix caching state | unverified |
| Primitives adoption rate by the model | unmeasured |

## 5. What to measure first

**Experiment: agent-side tool-call transport, measured locally — no Kaggle
quota.** The insight that makes it cheap: the HTTP 400 only fires when the
request CARRIES a `tools` field. Put the tool schemas in the prompt text, send
no `tools` field, and parse the model's `<tool_call><function=NAME>
<parameter=K>...` XML in the loop itself (a ~40-line parser slotted exactly
where `_looks_like_unparsed_tool_call` already counts these emissions,
`arc_induction_tool_loop.py:646`). This removes the server-parser dependency
on EVERY backend at once, and the emission format is a property of the model,
which the dev box serves locally as the same Qwen3.8-27B GGUF.

Protocol:

1. Add the fallback parser + prompt-embedded schemas behind a new env value
   (`CARNOT_ARC_INDUCE_TOOL_LOOP=selfparse`), leaving both existing modes
   byte-identical.
2. Run 20 real first-turn induce requests through the loop on the local
   backend across >=4 public games (dev twin roster), think mode ON. Also run
   the parser once over the captured vLLM emission (offplay_out5) as a
   cross-backend spot check.
3. Record, per response: attempted a tool call (yes/no), parsed to an
   executable dispatch (yes/no), dispatch succeeded (yes/no). The loop's
   existing stats fields already count most of this.

Acceptance gate (falsifiable, decided in advance):

- PASS: >=80% of responses attempt a tool call AND >=95% of attempts parse to
  an executed dispatch. Then the tool direction is transport-viable with no
  server flags; resume the stopped holdout-equalized A/B
  (`results/holdout_equalized_ab_20260820`, resumable per its own stop note)
  as the quality gate, and fold §1.3 move 1 and §3 into its tool arm.
- FAIL on parse rate: the server-side `qwen3_xml`/`qwen3_coder` flag trial
  (one launch-arg change, Kaggle-quota-gated) is the only remaining route; if
  that also fails, the tool direction is dead on the scored path, this note's
  §2-§3 are moot, and the fallback is the forward plan's Move 1 envelope caps
  alone. That would be a valuable negative: it would say the 97.6% reasoning
  spend cannot be converted to tool calls with this model and serving stack.
- FAIL on attempt rate (<80%): a prompt problem, not a transport problem;
  iterate the prompt-embedded schemas before concluding anything.

Cost: one bounded dev session; no scored-path risk (new env value, default
unreachable).

## Cross-references

- `docs/research-notes/qwen38-compaction-forward-plan-2026-08-20.md` — the
  ranked-moves plan this note extends; its Move 2 Step A is the server-side
  sibling of §5.
- `docs/research-notes/arc-induction-compacted-carried-state-2026-08-19.md` —
  compaction design; G_M fix plan lives in the forward plan's Move 3.
- `docs/research-notes/induction-token-accounting-and-rejection-2026-08-25.md`
  — the decode-side accounting behind §0.
- `docs/research-notes/avo-adaptation-for-local-generator-2026-08-21.md` — the
  AVO program this tool surface serves; names the quota-blocked parser trial.
- CLAUDE.md "AVO-Method Adoption", "ARC Live-Path Reachability Discipline",
  "ARC-AGI-3 IS a Live Hidden-Game Discovery Agent" — governing rules; every
  plug point above is on the `make_carnot_agent` / `arc_loop_solve` closure.

# Tool-Gap Feedback for the ARC Induction Tool Loop (2026-08-29)

REQ-ARC-WMTE-6770. Design note for the mechanism that identifies tool gaps
from live-run evidence and makes introducing new callable tools safe.

## The problem

The live agent's induction loop calls tools from `TOOL_SCHEMAS` in
`python/carnot/agentic/arc_induction_tools.py`. That set is closed and
hand-authored. Nothing observed live runs and asked: what tool did the model
need and not have?

The project already solved this shape once, for verifiers. The
Missing-Verifier Gap Logging discipline turns each present-but-unselectable
failure into a `ops/verifier_gaps.md` entry, and the gap ledger is a build
backlog. This mechanism is the tool-side analogue, and it reuses that
structure on purpose: `TOOL_SCHEMAS` is the registry of what exists;
`ops/arc_tool_gaps.md` is the ledger of what is demanded and missing.

## What signals exist versus what we wished existed

Measured before design, over every readable tool-loop artifact
(`results/arc_tool_loop_probe_20260817/*.json`,
`results/holdout_equalized_ab_20260820/shard_tool.jsonl`,
`results/tool_loop_compaction_ab_20260820/shard_{off,on}.jsonl`):

- 633 dispatched tool calls. Every name is inside `TOOL_SCHEMAS`
  (query_region 295, diff_grids 199, run_engine_on_transitions 100,
  list_transitions 39). **No live run has yet demanded a nonexistent tool.**
- 8 rows carry `tool_call_parse_failures: 1`. The counter conflates
  "unknown tool" with "unparseable JSON arguments" and keeps NEITHER
  identity. The kind of those 8 failures is unrecoverable. This is the
  finding the brief predicted: the strongest gap signal was not actually
  recorded. The capture had to ship before any mining could mean anything.
- The supervisor's redirect ledger (`arm_outcomes`,
  `stagnations_unredirected`) measures ARM demand, not TOOL demand. It
  connects here in one place: the `tool_loop_reinduction` arm
  (REQ-ARC-WMTE-6760) routes a stagnant level into the tool loop, and any
  gap events from that escalation now land in the same stats rows.

Signals chosen, because they are mechanically computable:

1. `unknown_tool`: the dispatch layer refused a `<function=NAME>` whose NAME
   is outside the active set. The name and the argument keys are the demand.
2. `bad_arguments`: a TypeError calling a real tool — the model imagined a
   signature the schema does not declare. Noisier: a TypeError raised inside
   a tool body is indistinguishable at the dispatch seam, so the analyzer
   labels this kind as weaker evidence.

Signal rejected: "the model needed a tool it could not imagine." Not
computable; not built.

## The mechanism

Three parts, closing the loop the way supervisor refinement does:

1. **Capture (always-on telemetry, live path).** `dispatch_tool` — the
   shared chokepoint both transports and every caller go through — records
   bounded `tool_gap_events` on the session. The loop writes them into
   `last_tool_loop_stats`, which every row consumer already copies, and the
   live E3 policy's `record["tool_loop"]` subset was widened to carry them
   (it was a fixed key list; the events would have died exactly on the
   scored path otherwise). Keys are present-and-empty on clean runs, never
   absent. Telemetry never alters a request payload — the same precedent as
   the REQ-6540 counters.
2. **Analysis (offline, recommends only).**
   `scripts/arc_tool_gap_refine.py` ingests rows into a durable ledger
   (`ops/arc_tool_gap_ledger.json`), with the supervisor-refinement tool's
   own row hash, rows-document shape, and clone-pruned directory scan
   (imported, not copied). Frozen contract: 3 events across 2 distinct rows
   yields a written `tool_gap_specification` for a human, including a
   ready-to-append `ops/arc_tool_gaps.md` entry. Below the floor it says
   `insufficient_evidence` loudly. Rows that predate capture are a named
   population (`no_capture_capable_rows`) — absence of evidence, not
   evidence of absence.
3. **Introduction (human-authored, default off, measurable).** A human
   authors the tool and registers it via `register_candidate_tool`. It is
   served only when named exactly in `CARNOT_ARC_INDUCE_CANDIDATE_TOOLS`
   (registered `unevaluated` in `ops/arc_flag_ledger.yaml`). Enabled, its
   schema reaches the request payload AND the selfparse prompt text, its XML
   calls coerce by its own schema, and its usage is measurable in
   `tool_calls_by_name` — so an A/B can read it before anyone believes it.
   Unset, `active_tool_schemas()` returns the `TOOL_SCHEMAS` object itself:
   byte-identical default by construction. Promotion into `TOOL_SCHEMAS` is
   a later human commit, on measurement.

## The human-versus-machine boundary, and why

CLAUDE.md's AVO rule: arm growth stays human on a 27B generator. The brief
asks whether tools are different. Position taken: the DISTINCTION IS REAL
BUT DOES NOT LICENSE GENERATION. A tool is bounded, typed, and offline
testable where an arm is an open-ended strategy — which is why gap
IDENTIFICATION can be fully mechanical here (the parser literally sees the
demanded name) while arm gaps need `stagnations_unredirected` as a proxy.
But tool INTRODUCTION is authoring live-path code that executes in-process
on the scored path. Three reasons it stays human:

1. Machine-added tools are machine-invented code shipping into the live
   scored path — the exact failure the disciplines exist to stop.
2. A demanded NAME is demand evidence, not a design. `get_full_grid` does
   not bound the response, and an unbounded retrieval tool rebuilds the
   prompt the tool set exists to shrink (the module's own thesis).
3. The verifier-gap precedent works exactly this way and its vocabulary
   transfers; a parallel convention would be invented for no gain.

So: the machine identifies, aggregates, floors, and writes the
specification; the human authors, enables, measures, and promotes. The same
sentence REQ-6720 uses: a human applies changes, or nobody does.

## What was deliberately NOT built

- No auto-generation of tool schemas or implementations from gap evidence.
- No automatic append to `ops/arc_tool_gaps.md` — the analyzer renders the
  entry; a human pastes it.
- No auto-promotion of candidates into `TOOL_SCHEMAS`.
- No seeded candidate tool. The registry ships empty because the evidence
  says no tool has been demanded yet; authoring one now would invert the
  mechanism's own thesis (gap first, then tool). The registry paths are
  proven by test-registered candidates driven through the real induce()
  entry point, and by 14 mutations (all RED).
- No mining of free-text model wishes; only refused calls count.

## How it is measured

- A candidate tool's value: enable it by env in a lever-harness or A/B run;
  read `tool_calls_by_name` for uptake and the run's induction outcomes for
  effect; the flag ledger holds the verdict before any default flip.
- The capture itself: `tool_gap_events` totals across future rows tell
  whether unknown-tool demand exists at all. If it stays zero for a long
  window, the honest reading is that the schema prompt is sufficient and
  this ledger stays empty — that null is a finding, not a failure.

## Cross-references

REQ-ARC-WMTE-6770 (spec) · REQ-ARC-WMTE-6720 (the arm-side sibling) ·
REQ-ARC-WMTE-6760 (the supervisor arm that escalates into the tool loop) ·
CLAUDE.md "Missing-Verifier Gap Logging" (the precedent) · CLAUDE.md
"AVO-Method Adoption" (the human-growth constraint) · `ops/arc_tool_gaps.md`
(the ledger) · `tests/python/test_arc_tool_gap_feedback.py` (entry-point
tests + mutation targets).

## Same-day adversarial review and fixes (appended, not rewritten)

A hostile review of the first committed cut (a613f36bc7) returned 12
findings; the coordinator ordered the freshness-lint rider split into its own
commit and the premise-level defects fixed. Owned corrections:

- **F2, the finding that mattered:** as first built, no env configuration let
  the live agent both run the tool loop AND record gap events —
  `CARNOT_ARC_INDUCE_TOOL_LOOP` runs the loop under "1"/"selfparse" and the
  only widened record was on the "repair" branch of the same variable. The
  claim above that stats reach "every row consumer" was false for the
  primary live induction, which never read them. Fixed: `_induce_and_plan`
  now clears the diagnostic, induces, and copies gap fields onto
  `attempt["tool_gap"]`; the stale-stats direction is pinned by test.
- **F3/F4 (the chokepoint was too narrow):** an unknown name with malformed
  JSON was discarded although the name needed no parse; a tool-call block
  the strict transports refuse held the demanded name in refused text. Both
  now captured (`argument_keys: null`; `loose_tool_call_names` with
  `source: "unparsed_text"`). The "not computable" boundary drawn above was
  wrong for these two populations — they are refused CALLS, not imagined
  wishes — and they are precisely where demand is likeliest to appear.
- **F5/F7:** enablement is now FROZEN per session; within a run, prompt,
  payload, dispatch, and the record read one snapshot. The cross-run limit
  is disclosed in the spec amendment: in-process model-executed code can
  mutate the registry/env, and no in-process guard survives that threat
  model — the sandbox flag is the real containment.
- **F6/F8/F9:** dark candidates no longer affect XML coercion; three
  decorative assertions (dropped-counter finalize, a tautological markdown
  assertion, unbound refusal text) were falsified by the review and replaced
  with biting tests; model-controlled event strings are capped; the analyzer
  surfaces `gap_events_dropped_total` so truncated counts read as floors.
- **F10:** the bootstrap null was over-generalized as "live"; the corpus is
  entirely the server-lifted transport (`selfparse_blocks_seen: 0`), which
  constrains names structurally. Corrected in `ops/arc_tool_gaps.md`.
- **R1/R2/R3 (freshness-lint rider, now its own commit):** the worktree
  fallback could adopt an UNRELATED repo with colliding layout (fail-open on
  a gate whose contract is fail-closed) — now gated on a shared root commit;
  the block-deciding `_sha256_at_head` also needed the GIT_DIR strip; the
  strip was narrowed to the four discovery overrides so GIT_CONFIG_*
  injection keeps working. The first regression test PINNED the R1 bug and
  was rewritten to assert the opposite.

Accepted residuals, stated plainly: the loose-name extraction is best-effort
(two regexes, bounded, deduped against parsed calls) and will miss shapes it
has never seen; R3's keep-config direction cannot be mutation-proven in this
environment (constructing a dubious-ownership repo needs root); and the
earlier "Zero decorative rules" claim was false until this pass — the full
mutation set was re-run (30 runs, labels M1-M28, all RED) after the fixes.

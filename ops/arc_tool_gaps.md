# ARC Tool Gaps — the missing-tool backlog for the induction tool loop

**WHY (REQ-ARC-WMTE-6770, 2026-08-29):** the live agent's induction loop can
call tools, and `TOOL_SCHEMAS` in
`python/carnot/agentic/arc_induction_tools.py` is the closed, hand-authored
set it can call. When the model writes a call for a tool outside that set,
the refusal is now retained with its identity (`tool_gap_events` in the loop
stats). This file is the tool-side sibling of `ops/verifier_gaps.md`: the
registry of tools we HAVE is `TOOL_SCHEMAS`; this ledger lists the tools the
model DEMANDED and we lack, with the evidence, so a human can author them.

**How a gap gets here.** Run
`.venv/bin/python scripts/arc_tool_gap_refine.py <rows.json|dir> ...`.
It ingests `tool_gap_events` into `ops/arc_tool_gap_ledger.json` and, when a
gap crosses the frozen floor (3 events across 2 distinct rows), renders a
ready-to-append entry in the schema below. A HUMAN appends it here and
authors the tool (`register_candidate_tool`, default off behind
`CARNOT_ARC_INDUCE_CANDIDATE_TOOLS`). The analyzer never writes this file
and never generates a tool. Never-prune; close an entry with
`status: filled (<tool name>)`, not deletion.

**Schema (one entry per gap):**
```
### TOOLGAP-<KIND>-<name>: <short demand description>
- status: open | building | filled (<tool name>)
- evidence: <event count> events across <row count> distinct run rows (<ledger path>)
- failure mode: <what the refusal cost the run>
- missing capability: <the tool the model demanded; observed argument keys>
- candidate design: HUMAN-AUTHORED via register_candidate_tool + CARNOT_ARC_INDUCE_CANDIDATE_TOOLS (default off)
- priority: high | medium | low (by refused-call frequency)
```

---

## Bootstrap evidence state (2026-08-29)

No gap entries yet, and that is a measured statement, not an omission. All
readable tool-loop artifacts were mined
(`results/arc_tool_loop_probe_20260817/*.json`,
`results/holdout_equalized_ab_20260820/shard_tool.jsonl`,
`results/tool_loop_compaction_ab_20260820/shard_{off,on}.jsonl`): 633
dispatched calls, every name inside `TOOL_SCHEMAS`
(query_region 295, diff_grids 199, run_engine_on_transitions 100,
list_transitions 39). Eight rows carry `tool_call_parse_failures: 1` whose
KIND is unrecoverable — the pre-capture counter conflated unknown-tool with
malformed-JSON and kept neither identity. That loss is the incident this
ledger exists to end: from 2026-08-29 the loop records `tool_gap_events`
with the requested name, so future entries here carry evidence, not guesses.

**Correction, same day (adversarial review, F10 — appended, not rewritten).**
An earlier draft of this section called the null a "live" measurement. The
mined corpus shows `selfparse_blocks_seen: 0` in every row: all 633 calls
went through the SERVER-LIFTED transport, where the server constrains call
names to the declared schema set, so an unknown name was near-structurally
impossible there. The null is real for that corpus and says nothing about
the selfparse transport — the live scored one — where the model writes
names freely and unknown-name demand becomes possible for the first time.
Read the zero as "no evidence yet", not "no demand".

## Open gaps

*(none yet — the analyzer emits ready-to-append entries when live evidence
crosses the floor)*

## Filled gaps

*(none yet)*

# Tool-gap elicitation — a design considered, not built (2026-09-11)

**Operator question, verbatim:** "are we prompting to include the ability for a model to
imagine any tools that it might need in order to spot the gap tool that needs to be
implemented?"

**Answer: no.** Checked `render_tool_schemas_for_prompt` in
`python/carnot/agentic/arc_induction_tools.py`. The prompt the model sees is a closed,
hand-authored list: "AVAILABLE TOOLS (schemas): ..." followed only by the enabled tool set,
plus the exact `<tool_call>` syntax to use. Nothing invites the model to name a tool that
does not exist. `record_tool_gap` (`arc_induction_tools.py:827-837`) only fires when the
model *spontaneously* requests something nonexistent, unprompted — a passive-observation
channel, not an active elicitation one. This is a plausible reason the tool-gap ledger
(`ops/arc_tool_gaps.md`) has stayed empty across every real measurement to date (n=7 real
inductions as of this date, 0 gap events): nothing has ever asked.

## The idea

Add an explicit elicitation step: "if there is a tool you wish existed that would have
helped here, name it and describe what it would do." Two candidate placements:

1. **In-loop, every action turn.** Cheapest to wire (append to the existing tool-schema
   prompt block), but risks derailing the model from actually playing the game — a model
   asked to imagine will imagine, whether or not there is real unmet need, and every extra
   sentence competes with the model's limited attention on a 27B local generator.
2. **Post-episode reflection turn.** After the real play is over (win, loss, or budget
   exhausted), one dedicated turn asking the same question. Nothing is lost by asking here
   — the episode's outcome is already fixed — so it adds no risk to the measurement.

## Why not built yet

Not decided by the operator; raised as an open question during the 2026-09-11 outer-loop
session and not actioned in that session. The trade-off is real and unresolved: elicited
tool-gap candidates are not the same evidence as spontaneous demand — a model that names a
tool when asked "what would help" is describing a wish, not necessarily encountering a real
block on its actual play path. Any elicited candidate should probably be recorded in a
SEPARATE column from spontaneous `unknown_tool`/`bad_arguments` gap events
(`ops/arc_tool_gaps.md`'s existing two-signal-kind structure could extend to a third:
`elicited_wish`), not merged with the harder passive-observation evidence.

## Cross-references

- `python/carnot/agentic/arc_induction_tools.py:render_tool_schemas_for_prompt` — the current
  closed-list prompt
- `python/carnot/agentic/arc_induction_tools.py:record_tool_gap` — the passive-observation
  detector
- `ops/arc_tool_gaps.md` — the ledger this would feed, if built
- `ops/known-issues.md` 2026-09-11 entries — the same-session tool-use volume follow-up
  (cumulative real inductions, still 0 spontaneous gap events at n=7)
- CLAUDE.md "Missing-Verifier Gap Logging" — the sibling discipline this idea extends from
  verifiers to tools

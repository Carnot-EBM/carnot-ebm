# Holdout-equalized induction A/B, selfparse transport (started 2026-08-29)

A SEPARATE run directory from `holdout_equalized_ab_20260820`, deliberately.

That run banked 13 tool cells and 2 single cells against the NATIVE tool transport -- a `tools`
field in the request, which only the local llama.cpp dev twin accepts. The scored vLLM server is
launched with no `--enable-auto-tool-choice` and returns HTTP 400 on any request carrying
`tools`, so those cells measure a transport the scored path cannot run.

This run uses `--tool-transport selfparse` (REQ-ARC-WMTE-6730): no `tools` field is sent, the
schemas travel as prompt text, and the model's Qwen3-coder XML is parsed agent-side. Its
pre-registered gate passed at ceiling on 2026-08-28 (20/20 attempt, 20/20 parse-to-dispatch),
but all 20 live calls were the zero-argument `list_transitions`, so multi-turn and
code-carrying shapes are covered only by a stored vLLM emission and unit fixtures. Driving the
full loop here is what exercises them under the same counters.

Cells are NOT interchangeable with the 08-20 run and must never be pooled with them.

The single (control) arm must be re-run here rather than borrowed from 08-20: the goal-prompt
RLE change (REQ-ARC-WMTE-6740, 2026-08-28) altered prompt construction after those cells were
banked, so pairing across the two dates would compare across a prompt change as well as across
the arm.

Cost, from the 08-20 stop note: the tool arm averaged 598s per cell, the single arm 4,238s. The
tool arm is therefore hours and the control arm is days; the tool arm runs first.

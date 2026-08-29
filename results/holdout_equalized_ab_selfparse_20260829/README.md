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

## Scope limit of this A/B's metric (operator directive, 2026-08-29)

Operator, on seeing `mem=True` in an early cell:

> "if the live agent is memorizing at run time, that is acceptable: while it is probably
>  generally better for efficiency if our world model understands the game, so long as we are
>  making progress on levels and improving our challenge score in the process, memorization is
>  only a bad thing here at training time before we submit a live agent challenge run."

This corrects a misreading in the run log's early commentary, and it also bounds what this A/B
can decide.

**What the shards record:** `holdout`, `visible_fit`, `is_memorizing`, `memorization_scan`,
`tool_loop_stats`, `elapsed_s`. **What they do not record: level progress, in any form.** The
arm script imports `arc_actions_to_progress` only to BUILD windows; it never scores with it.

So this A/B selects on OFFLINE GENERALIZATION. That is a legitimate dev-time question — a
memorizing engine at selection time tells us the draw did not explain the level — but it is a
PROXY for the deliverable, which is levels banked and challenge score on a hidden game. If a
memorizing engine still banks levels, holdout is not the property that drives the score, and an
A/B selecting on holdout could reject the better live method.

**Consequence for reading the result.** A holdout difference between arms answers "does the tool
loop induce more generalizable engines". It does NOT answer "does the tool loop bank more
levels", which is the question that decides whether to ship it. Do not let this run's number
stand in for that.

The follow-up that would answer it is an actions-to-progress A/B (REQ-ARC-WMTE-5720) over the
same arms — bounded live runs counting actions to level-up, with memorization recorded as an
observation rather than a penalty. Not started; naming it here so it is not rediscovered.

This run continues unchanged: mid-run metric changes would invalidate the cells already banked,
and the generalization question is still worth its answer.

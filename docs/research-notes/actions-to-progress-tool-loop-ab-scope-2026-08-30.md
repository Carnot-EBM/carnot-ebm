# Scope: an actions-to-progress A/B for the callable-tool loop

Written 2026-08-30. Not implemented — GPU 0 and GPU 1 were both occupied when this was written.
This sizes the experiment and states what it can and cannot decide.

## Why this rather than finishing the holdout A/B's control arm

The holdout-equalized A/B banked 38 of 39 tool-arm cells before dying. Completing it needs the
single-shot control arm, which the 2026-08-20 stop note measured at ~4,238s per cell against the
tool arm's ~598s — days of GPU time.

It would buy a paired result on the wrong metric. Operator directive, 2026-08-29:

> "if the live agent is memorizing at run time, that is acceptable: while it is probably
>  generally better for efficiency if our world model understands the game, so long as we are
>  making progress on levels and improving our challenge score in the process, memorization is
>  only a bad thing here at training time before we submit a live agent challenge run."

The holdout A/B scores `holdout`, `visible_fit` and `is_memorizing`, and **no level progress in
any form**. If a memorizing engine still banks levels, holdout is not the property that drives
the score, and an A/B selecting on it can reject the better live method. Spending days to finish
a proxy before running the real measurement is the wrong order.

## What exists already

`python/carnot/agentic/arc_actions_to_progress.py` (REQ-ARC-WMTE-5720) is a bounded
actions-to-progress harness: `run_bounded_progress`, `per_level_hv_progress`, `ProgressResult`,
and `apply_arm(proposer, arm)` which configures env plus proposer fields and returns a restore
closure. Five experiments already drive it, including `experiment_6473_tool_loop_compaction_ab_arm.py`
— so a tool-loop A/B on this metric has precedent rather than needing invention.

## The gap to close

`apply_arm` saves and sets exactly three env vars — `CARNOT_ARC_CODEONLY_INDUCE`,
`CARNOT_ARC_PLAYBOOK_RETRIEVAL`, `CARNOT_ARC_PLAYBOOK_EXEMPLARS_ENABLED` — and does not touch
`CARNOT_ARC_INDUCE_TOOL_LOOP`. A tool-loop arm therefore needs either a new entry in
`ARM_CONFIGS` carrying that variable, or the arm script setting it around the call.

Prefer extending `ARM_CONFIGS`, for one specific reason: `apply_arm`'s restore closure is what
stops one arm leaking configuration into the next on a reused proposer. A variable set outside
that mechanism is not restored by it, and a leaked tool-loop setting would silently turn the
control arm into a second treatment arm — the same defect the holdout A/B's `transport_env()`
was written to prevent, where the control arm receiving the variable would have compared a loop
against a loop.

## The measurement

Two arms on the LIVE path (`E3AgentPolicy` / `make_carnot_agent`, per the ARC Live-Path
Reachability Discipline):

- **control** — single-shot induction, `CARNOT_ARC_INDUCE_TOOL_LOOP` unset
- **treatment** — `CARNOT_ARC_INDUCE_TOOL_LOOP=selfparse`

Metric: actions to level-up, bounded per run. Memorization is RECORDED AS AN OBSERVATION, never
as a penalty — that is the whole point of choosing this metric over holdout.

Pairing: same games, same seeds, same action budget, one arm per process and one card per arm
(the isolation rule the holdout arm script already documents).

## What it can and cannot decide

It answers: does the tool loop reach a level-up in fewer actions? That is the question that
decides whether to ship it, and it is what the supervisor's `tool_loop_reinduction` arm
(REQ-ARC-WMTE-6760, default OFF) is waiting on before it can be promoted or retired on evidence.

It does NOT answer whether the tool loop generalizes better — the holdout data speaks to that,
and 38 unpaired tool-arm cells exist for whoever wants it.

It also does not answer the hidden-game question. All public games are cleared; a public-game
measurement is a development proxy for a live agent that must discover games it has never seen.
Say so in the artifact rather than letting a public-set win read as a submission result.

## Cost and prerequisites

Per the 08-20 note the tool arm runs ~598s per cell. A control arm on the LIVE path is not the
same shape as that note's single-shot induction arm, so its cost is unmeasured — measure one cell
before committing to a full grid.

Prerequisite worth honouring after 2026-08-29: install the death receipt
(`python/carnot/testing/long_run_receipt.py`, REQ-INFRA-6830) so that if this run is killed the
way the last one was, the next investigator gets a signal name instead of a guess.

# B2 think-ON induction pilot (Experiment 10010, 2026-09-24)

## What was run

The live agent's default induction setting was replayed offline on the 10
pre-registered B2 windows:

- think mode ON, 1 round, live retry ladder (3 tries)
- Qwen3.8-27B Q4_K_M on llama.cpp, GPU 1 only, 131,072-token context
- 2,400 s per call (the live timeout)

The design was fixed before any response was scored
(`docs/research-notes/b2-positive-control-2026-09-23.md`, "Pre-registered
design"). Harness: REQ-ARC-WMTE-10010, commit d7a33399e9. Artifact:
`results/experiment_10010_b2_think_on_pilot.json`. Evidence:
`results/raw/experiment_10010_b2_think_on_pilot/`.

Controls passed on every run, before any model output was scored: identity 0.0,
expert 1.0, and the recorded codeonly baseline reproduced at 0.1275. Prompt
fidelity was 10/10: each prompt was byte-equal to the recorded first call, with
no held-out answer in it.

The run used the scorer from before the isolation fix. Every window was then
rescored with the per-row-isolated scorer. No score changed. The as-run artifact
is kept as `artifact_as_run_before_isolation_fix.json`.

## Result

Primary metric: masked held-out change fidelity (identity 0.0, expert 1.0).

| Window | Think-ON | Codeonly baseline | Tokens | Wall time |
|---|---|---|---|---|
| su15 | 0.071 | 0.582 | 75,503 | 2,363 s |
| sp80 | 0.958 | 0.030 | 43,839 | 1,264 s |
| ft09 | 0.500 | 0.156 | 35,974 | 1,043 s |
| g50t | 0.857 | 0.122 | 53,471 | 1,585 s |
| m0r0 | 1.000 | 0.000 | 54,908 | 1,634 s |
| dc22 | 0.0 (cut off at 2,400 s, no answer) | 0.104 | - | 2,400 s |
| wa30 | 0.857 | 0.065 | 76,203 | 2,378 s |
| ka59 | 0.0 (cut off at 2,400 s, no answer) | 0.118 | - | 2,400 s |
| sb26 | 0.500 | 0.000 | 43,882 | 1,282 s |
| ar25 | 1.000 | 0.098 | 69,782 | 2,168 s |
| **Mean** | **0.574** | **0.128** | | |

- 7 of 10 windows beat the baseline. Paired Wilcoxon, one-sided, p = 0.024.
  Sign test, 7 of 10, p = 0.17. Neither test was pre-registered.
- Every call that finished stopped on its own. 2 of 10 calls hit the 2,400 s
  limit.
- **No window passes the live exact 1.0 gate.** On the 3 windows where the gate
  can pass (su15, sp80, ft09), the best was sp80 at 7 of 8 rows.
- Secondary: mean masked exact accuracy 0.62. The no-op guard mean is 0.25; the
  two cut-off windows count as hallucinating on their no-op rows.

## The finding that matters most: the Kaggle vLLM path would discard these engines

The harness also scored each response the way the scored Kaggle path extracts
code. That path runs vLLM with no reasoning parser, so the think text stays in
`content`. Live extraction then takes the first python fence in `content`.

In all 8 windows that produced an engine, the reasoning contains a python fence,
because the model restates the prompt's "return one python block" instruction.
Extracted the Kaggle way, all 8 engines are broken: `...`, prose fragments,
syntax errors. The mean is 0.0 against 0.574.

The outer loop confirmed the cause in source
(`ops/known-issues.md`, 2026-09-23 vLLM entry):

- The vLLM launch argv has no `--reasoning-parser`.
- vLLM's default reasoning parser is empty.
- Live extraction reads `content`.

So on the scored path, think-ON induction probably yields unusable engines in
most calls, even when the model wrote a good one.

## What this means

1. **Think-ON single-shot induction writes much better world models than
   codeonly** on these windows (0.574 against 0.128). B2's codeonly measurement
   understated the live default.
2. **The exact 1.0 gate still rejects every one of them.** Several are near
   misses (sp80 7/8, masked exact 0.875). The gate question from the positive
   control is now concrete: correct-looking engines exist, and the gate turns
   them all away.
3. **The scored path likely throws the engines away before the gate sees them**,
   because of the vLLM extraction defect. This is the cheapest fix with the
   largest expected effect.

## Limits

- 10 windows x 1 draw. Each window has one prompt, so extra draws would measure
  sampling variance only.
- Few changing rows after masking (3 on m0r0; 4 on sb26, dc22, ft09).
- The local path applies a repetition penalty (1.1) that the scored vLLM path
  drops. The model differs too: Q4_K_M on llama.cpp here, NVFP4 on vLLM on
  Kaggle.
- The live split-induce fallback (engine-only, then goal-only) after a failed
  combined call was not replayed.

## Not decided (operator)

- Fix the vLLM extraction and repetition-penalty defects. This needs a Kaggle
  run to confirm.
- Whether the live gate should accept near-exact engines (for example, a
  change-fidelity threshold with a no-op guard) instead of exact 1.0. This
  interacts with the HUD-mask and purity gaps in `ops/verifier_gaps.md`.

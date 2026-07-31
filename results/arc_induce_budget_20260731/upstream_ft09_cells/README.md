# Upstream ft09 evidence, copied so the artifact's claims stay verifiable

These four cells and the one engine file are the 2026-07-30 treatment-activation grid's ft09
output. They lived in a session scratchpad, which is garbage-collected; the Phase-1 artifact
(`results/outer_loop_arc_induce_budget_phase1_20260731.json`) makes three claims that can only be
checked against them, so they are copied here rather than cited by a path that will stop existing:

* `on__ft09__s1.json` / `onb__ft09__s1.json` -- the LLM-ON replicates. Their
  `induction_events[0].refinement_rounds` carry the `[HIT n_predict=4096 OUTPUT LIMIT before
  completing]` diagnostic the budget hypothesis came from, and the round-1 counterexample
  (`real_n=25`, `real_n_correct=6`, `real_accuracy=0.24`, 8 mismatches) that the refactor sweep
  replays.
* `off__ft09__s1.json` / `offb__ft09__s1.json` -- the LLM-OFF replicates, which are what license
  capturing the induce prompt from an LLM-off run at all.
* `on__ft09__s1__world_model.py.frozen` -- the banked engine. The `.frozen` suffix is the
  repo's existing convention for a BYTE-PRESERVED snapshot of LLM-generated Python
  (cf. `results/inducer_h2h_6021/h2h_arm_runner.py.frozen`, and the same reasoning that put
  `results/arc_e3_origin_fixtures` in pyproject.toml's ruff exclude): it must not be linted
  or reformatted, because its exact bytes -- including the 1061-line `#` wall -- ARE the
  evidence. It is: 1144 lines, 11 lines of code, a 1061-line
  contiguous run of bare `#`, and no return on the `action == 6` path. Its two halves tokenize to
  4093 + 81 through the generator's own vocabulary against a 4096 budget, and it carries
  `_combine_world_model()`'s injected `import numpy as np` prefix -- which is how the artifact
  establishes it came from the SPLIT fallback rather than round 1's combined call.

Copies, not originals. Nothing here is rewritten.

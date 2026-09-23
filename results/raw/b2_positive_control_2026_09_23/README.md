# B2 positive control evidence (2026-09-23)

This directory holds evidence from the positive control run on the 12 B2 v3
induction windows. The analysis is in
`docs/research-notes/b2-positive-control-2026-09-23.md`.

| Path | What it is |
|---|---|
| `<game>/expert_engine.py` | A hand-written grid-only engine for that game's window. An agent wrote it from the public game source. It uses only `(grid, action, data)`. It is a ceiling reference, not a live solver. |
| `synth/rescore_all.py`, `synth/rescore_all.json` | The synthesis agent's rescoring of every window under the live gate and under the proposed pilot metric |
| `synth/pilot_baseline.json` | Codeonly first-shot baseline per window under masked change fidelity (mean 0.13) |
| `synth/raise_inflation.*`, `synth/raise_trustmetric.*` | Reproduction of the defect where rows an engine raises on are dropped from graded metrics |
| `workflow_result.json` | The full structured output of workflow `wf_1b23ef1e-379` (12 control reports, 12 verifier audits, synthesis) |

The scripts name absolute paths in a session scratch directory that no longer
exists. They are kept as a record of what was run, not as runnable tools. To
re-run, point them at the engines in this directory and at
`results/raw/experiment_10009_b2_induction_gate_measurement_v3/`.

These engines were written by reading public game source. That is permitted for
offline development on public games. They must never become part of the live
hidden-game agent.

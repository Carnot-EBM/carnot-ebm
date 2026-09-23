# B2 positive control: can any grid-only world model pass the live gate? (2026-09-23)

## Why this was run

The B2 v3 triage (`docs/research-notes/b2-induction-failure-triage-2026-09-23.md`)
left one merit claim standing: 0 of 36 first-shot codeonly responses would have
passed the live acceptance gate. The gate requires exact held-out accuracy 1.0.
That claim had no positive control. If no correct engine can reach 1.0 on a
window, the model's failure there means nothing.

## Method

Workflow `wf_1b23ef1e-379`, 25 agents, CPU only, no model calls. For each of the
12 windows:

1. A control agent reproduced the live split, ran a lookup engine through the
   real `WorldModelVerifier` (plumbing check), counted contradictory rows,
   replayed the recorded actions in the offline simulator, and scored the
   identity engine and the 3 recorded first-shot engines.
2. The same agent wrote an expert engine from the public game source. The
   engine takes only `(grid, action, data)`. It was scored on the held-out rows
   and then on 40 or more fresh simulator transitions, to catch memorization.
3. A second, hostile agent re-derived the split and the plumbing check, audited
   the expert engine for memorization, hidden-state use, and held-out tuning,
   and ran its own fresh transitions.
4. A synthesis agent applied one headroom definition to all 12 windows: an
   engine that is correct on every reachable state. It then rescored every
   window itself.

Evidence is preserved in `results/raw/b2_positive_control_2026_09_23/`.

## Result per window

"Expert" is the source-derived grid-only engine. "Live gate" means unmasked
exact held-out accuracy, which the live agent uses.

| Window | Verdict | Identity | Expert, live gate | Best first shot, live gate |
|---|---|---|---|---|
| su15 | headroom | 0.125 | 1.0 | 0.875 (seed 7491001; missed only the block-move row) |
| sp80 | headroom | 0.0 | 1.0 | 0.0 |
| ft09 | headroom | 0.5 | 1.0 (a visible-rows-only engine also scores 8/8) | 0.375 (no-op rows only) |
| vc33 | degenerate | 0.0 | 1.0 (held-out row is the prompt row) | 0.0 |
| sb26 | capped at 7/8 | 0.375 | 0.875 (row 20 is an undo with hidden history) | 0.375 |
| ar25 | capped at 7/8 | 0.0 | 0.875 (row 19 is an undo with hidden history) | 0.25 |
| g50t | capped by a HUD step counter | 0.125 | 0.5 (1.0 with row 63 masked) | 0.125 |
| m0r0 | capped by a HUD step counter | 0.25 | 0.5 (1.0 with rows 0 and 63 masked) | 0.25 |
| dc22 | capped by a HUD step counter | 0.125 | 0.5 (1.0 with row 63 masked) | 0.375 (never reached the gate) |
| wa30 | capped by a HUD step counter | 0.125 | 0.625 (1.0 with row 63 masked) | 0.125 |
| ka59 | capped by a HUD step counter | 0.0 | 0.625 (1.0 with row 63 masked) | 0.125 |
| bp35 | capped by hidden undo and crusher state | 0.0 | 0.625 | 0.25 |

Plumbing check: the lookup engine passed the real gate on every window. The gate
code itself is not broken for a lookup engine.

## What the B2 merit claim now means

- **On the 3 windows with clean headroom (su15, sp80, ft09), 0 of 9 first shots
  passed.** 7 of the 9 reached the live gate and were rejected. The closest was
  su15 seed 7491001 at 7/8.
- **On 8 windows no correct engine can pass the unmasked gate.** The caps run
  from 0.5 to 0.875. Failures there do not show the model was wrong.
- **vc33 tests copying, not induction.** Its window has 1 row, and the held-out
  row is the row the prompt shows. All 3 engines still failed to reproduce it.
- **A weaker graded result holds on 10 windows.** Under masked change fidelity
  (defined below), the first shots average 0.13, from 0.0 to 0.58 per window.
  Identity scores 0.0 and the expert scores 1.0. The model is far below what is
  achievable on the grid-determined dynamics.
- **The sample is 12 windows with 3 draws of one identical prompt each**, not 36
  independent trials. Read this as a sanity check, not a failure rate.

## Gate defects found (live scored path)

1. **HUD step counters make the gate unpassable for a correct engine** on g50t,
   m0r0, dc22, wa30, and ka59. A step bar shows a hidden action count. A HUD mask
   exists, but `SUBMITTED_WORLD_MODEL_HUD_MASK_ENABLED = False`
   (`arc_executable_world_model.py`) and the Kaggle kernel does not set
   `CARNOT_ARC_WM_HUD_MASK`. With the mask on, the experts pass and the swallow
   check reports `ok`.
   **Context that limits this finding:** exp6015 already ran the mask live in a
   four-arm A/B. It was null: the mask admitted 1 of 25 cells, and that engine
   produced no plan. This control explains why: the mask is necessary for a
   correct engine on these games, but the induced engines are far from correct
   (mean change fidelity 0.13). Turning the mask on alone is not expected to help
   until induction quality improves.
2. **Rows where the engine raises are dropped from every graded metric.**
   `WorldModelVerifier.score` skips a raised row before it counts changed or
   no-op rows. So `cell_recall`, `change_fidelity`, `change_accuracy`, and
   `noop_hallucination_rate` ignore it; only exact accuracy counts it. Reproduced:
   the su15 expert wrapped to raise on 7 of 8 held-out rows gets `cell_recall` 1.0
   and passes the change gate. With `CARNOT_ARC_TRUST_METRIC=cell_recall` the live
   selector accepts it at 1.0. The default metric is `exact` and the kernel does
   not set the flag, so the shipped path is safe today. The defect is latent and
   fails open.
3. **The gate does not enforce engine purity.** It scores rows in trajectory
   order in one process. Verifiers built stateful engines that pass at 1.0 on
   sb26, m0r0, ka59, bp35, and ar25. `plan_in_model` does not replay the
   trajectory, so such a pass would not carry over to planning.
4. **Level-up re-induction on 1 transition measures nothing out of sample.**
   `_split_prefix_heldout` returns `(rows, rows)` when fewer than 2 rows exist.
   With the CEGIS split on, no engine can pass. Either way the gate is
   meaningless there.
5. **Level-up prompt mislabel (proven).** On vc33 the block "BOARD AT THE START
   OF THE CURRENT LEVEL" is the last level-0 board, not the level-1 opening.
6. **Round-2 held-out leak while the CEGIS split is off.** The refactor prompt
   picks mismatch rows from all 25 rows. su15 seed 7491001 and ft09 seed 7491002
   sent held-out row 19 with its true change. Both responses were empty, so no
   scored engine was affected. The live default is 1 round. Fallback prompts that
   include held-out rows were staged but never sent.
7. **Undo rows with hidden history are graded like any other row.** On sb26,
   ar25, and bp35 a tie-break guess decides acceptance.
8. **The run did not record the gate flags** (`CARNOT_ARC_TRUST_METRIC`,
   `CARNOT_ARC_WM_HUD_MASK`, `CARNOT_ARC_CEGIS_ACCEPT_SPLIT`). Telemetry rules out
   the CEGIS split and the mask. Nothing rules out the trust metric, and the g50t
   verdict flips under it.

## Pre-registered design for the pending think-ON pilot

Fix these before any response is scored.

- **Arm:** think-ON, 1 round, live budget (131,072 tokens, 2,400 s per call),
  replayed offline on the recorded first-call prompts. No round 2, because the
  refactor prompt leaks held-out rows while the split is off.
- **Windows (10):** su15, sp80, ft09, g50t, m0r0, dc22, wa30, ka59, sb26 without
  held-out row 20, ar25 without held-out row 19. Drop vc33 (1 row, held-out row
  is the prompt row, mislabelled board). Drop bp35 (6 of 8 held-out rows depend
  on hidden state).
- **Masks:** row 63 on g50t, dc22, wa30, ka59; rows 0 and 63 on m0r0. No mask on
  su15, sp80, ft09: their bar changes follow from the grid, and a mask would
  delete sp80's bar-only rows.
- **Primary metric:** held-out symmetric-union change fidelity (the change gate's
  own quantity), averaged over held-out changing rows, with a raised row counted
  as 0. Identity scores 0.0, the expert 1.0. Compute it in the pilot's own
  wrapper, not from `WorldModelVerifier`'s graded fields (defect 2).
- **Guard metric:** no-op hallucination rate, with a raised no-op row counted as
  hallucinated.
- **Secondary:** masked exact held-out accuracy, plus the live unmasked 1.0 pass
  (meaningful only on su15, sp80, ft09).
- **Purity:** load a fresh engine module per row, or score rows in shuffled
  order.
- **Baseline to beat:** codeonly first shots, mean 0.13. Per window: su15 0.58,
  ft09 0.16, g50t 0.12, ka59 0.12, dc22 0.10, ar25 0.10, wa30 0.07, sp80 0.03,
  m0r0 0.0, sb26 0.0 (`synth/pilot_baseline.json`).
- **Record** every gate flag in the artifact.
- **Power:** 10 windows x 1 draw is a pilot only. Extra draws measure sampling
  variance only, because each window has one prompt. Few held-out rows change
  after masking (3 on m0r0, 4 on sb26, dc22, ft09), so per-window numbers are
  coarse.

## Caveats

- One headroom definition was applied to all 12 windows. For sb26, ar25, g50t,
  and dc22 a verifier used a looser one. The facts agreed in all four cases.
- The expert engines are level-specific and source-derived. They show a correct
  engine exists. They do not show that 17 visible rows identify it. Only ft09 has
  a visible-rows-only engine that passes.
- Several control agents saw held-out rows before writing their engines. The
  verdicts rest on the fresh simulator tests, not on the first held-out score.
- The masks and exclusions above were chosen after the held-out rows had been
  seen. The live agent does not have them.
- The run-time gate modules are inferred (the run worktree is gone). They have
  not changed on main since 8d7ca7bab6, and every recorded telemetry metric that
  was checked reproduces exactly.
- Two control agents imported game source in a way that wrote bytecode caches
  under `environment_files/dc22/` and `environment_files/su15/` (gitignored,
  harmless). This broke the read-only instruction.

## Not checked

- Think-ON first-shot responses (the pilot itself).
- Visible-rows-only identifiability on su15 and sp80.
- Stateful-engine tests on wa30, g50t, and dc22.
- The seed-7531001 attempts.
- How many other public games have a hidden-count HUD bar.
- `plan_in_model` behaviour with an engine accepted under a masked gate.

## Cross-references

- `docs/research-notes/b2-induction-failure-triage-2026-09-23.md`
- `results/experiment_10009_b2_induction_gate_measurement_v3.json`, field
  `corrigendum_2026_09_23_positive_control`
- `results/experiment_6015_wm_hud_mask_change_gate_four_arm_live.json` (prior
  live mask A/B, null)
- `results/raw/b2_positive_control_2026_09_23/` (engines, scripts, full workflow
  output)

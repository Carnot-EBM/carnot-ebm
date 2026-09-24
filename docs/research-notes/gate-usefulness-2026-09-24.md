# Does a looser world-model gate accept useful models? (Experiment 10012, 2026-09-24)

## Question

The live agent trusts an induced world model only if it reproduces 8 held-out
transitions exactly. The think-ON pilot showed good models exist and all fail that
gate. This experiment asked whether a looser gate would accept models that help.

A gate is only right if it predicts usefulness. So each model got a ground-truth
label from the real offline simulator: plan inside the model with the live
planner, play the plan in the real game the way the scored agent does, and count
USEFUL when the real game advances a level.

Artifact: `results/experiment_10012_gate_usefulness.json` (REQ-ARC-WMTE-10012).
Evidence: `results/raw/experiment_10012_gate_usefulness/`.

## A correction first (append-only record)

The first run (v1) did not execute the way the scored agent does. The outer loop's
brief described the scored agent wrongly, and an independent review caught it:

| Point | v1 (wrong for the scored agent) | Scored agent (`E3AgentPolicy`) |
|---|---|---|
| Divergence | stop at the first step where reality differs from the prediction | play the plan to the end; nothing compares prediction and reality |
| Start state for a stall induction | the state after the recent moves | the level's reset board, after sending RESET |

The halt rule belongs to the offline development twin (`plan_and_execute`), not the
scored agent. v1 therefore reported "keep exact 1.0; relaxing only adds harm",
and the outer loop passed that on to the operator. It does not hold. Amendment 1
in the spec made the scored-agent execution the primary arm (LIVE_SCORED). It kept
v1 as a labelled second arm; the v1 artifact is
`results/raw/experiment_10012_gate_usefulness/v1_offline_twin_halt_artifact.json`.

## Result (LIVE_SCORED arm)

- 3 of 10 windows are label-informative, meaning the correct expert model levels up
  from the real start state: su15, ft09, m0r0.
- **One non-expert model is useful: su15 think-ON.** It levels up in 7 real
  actions. Its predicted grids go wrong at step 2, but its plan is the same 7
  clicks as the expert's winning plan.
- **Every candidate gate rejects it.** Its held-out exact accuracy is 0.125.
- Relaxed gates add only models that do not help. Among the real candidates
  (think-ON and codeonly, controls removed):

| Gate | Accepted and useful | Accepted, not useful | Rejected but useful | Rejected, not useful |
|---|---|---|---|---|
| live exact 1.0 | 0 | 0 | 1 | 11 |
| masked exact 1.0, change fidelity >= 0.9, cell recall >= 0.8 | 0 | 1 | 1 | 10 |
| masked exact 0.875, change fidelity >= 0.7 | 0 | 2 | 1 | 9 |
| masked exact 0.75 | 0 | 4 | 1 | 7 |

- With the expert controls included, the exact 1.0 gate also rejects a useful
  model: the m0r0 expert scores 0.5 unmasked because of the step-counter row, and a
  masked gate accepts it.

**In this sample, held-out transition accuracy did not predict usefulness.** The
one useful induced model was scored low; the models the looser gates added did not
help.

## Other findings

- **Wrong win conditions.** Two non-expert models predicted the real moves
  correctly for many steps (su15 codeonly 19/19 from the induction state; ft09
  think-ON 8/8 from the reset board). Their own `is_level_complete` then returned
  True on a real state that had not won. The dynamics were right and the goal was
  wrong. Two cases only.
- **The live goal check cannot catch this here.** The gate's goal-predicate
  consistency veto fires only when the window holds a real level-up, and none of
  the 10 windows does.
- **The scored agent never checks its plan against reality.** A model that is
  accepted but wrong spends real actions until its plan runs out. The artifact
  records these wasted actions per pair.
- **Planner budget matters.** With 150,000 nodes (a NOT-LIVE arm), ka59 and ar25
  become informative: their experts win, but need more search than the live
  20,000 nodes.

## Why 7 windows could not answer

| Window | Reason |
|---|---|
| sp80, dc22, wa30 | the expert's win condition is always False (the positive control built experts for transitions, not goals) |
| sb26 | the expert has no win condition |
| ka59, ar25 | planner budget (informative at 150k nodes, not live) |
| g50t | unresolved: the search empties without budget or depth limits |

## What this means

1. Do not loosen the gate on transition accuracy. On this sample that adds bad
   accepts and still misses the useful model.
2. The signal that mattered is whether the model's plan leads to a real win, and
   whether its win condition is right. That points toward checking the plan and
   the goal, not the per-cell dynamics.
3. The sample is very small: one useful induced model. Any gate choice needs more
   informative windows first. The cheapest way to get them: give the experts real
   win conditions (sp80, dc22, wa30, sb26), and add windows.

## Limits

- 10 windows, one first-call prompt each; 3 informative in the live arm.
- The usefulness label covers one plan from one start state. It does not include
  the explorer fallback a rejected model leads to.
- Masks and exclusions came from the positive control, chosen after the held-out
  rows were seen.

## Amendment 2: repaired controls and stored engines

Amendment 2 preserves the Amendment 1 execution rules. In particular,
`LIVE_SCORED` sends RESET for these stall-type inductions, plans from the reset
board, and plays the complete plan without a divergence stop. The Amendment 1
artifact was preserved byte-for-byte at
`results/raw/experiment_10012_gate_usefulness/v2_amendment1_artifact.json`
(SHA-256 `a3215ae57c1d15d8c5306ac22c7aed0413154e0d5d636369f2ddc4f558d81271`).

### Repaired expert completion predicates

The public level-one winning route was replayed in a fresh real simulator for
each repaired predicate. All four predicates were true on the first level-up
frame, false on every earlier frame, and false on both grids of every frozen
Experiment 10010 window row for that game.

| Game | Grid-only completion predicate | Real route | LIVE_SCORED result |
|---|---|---:|---|
| sp80 | at least 80 colour-8 pixels, the visible platform marker emitted only by the verified successful spill; the real level-two frame has 96 | 4 actions | level-up, 4-action plan, 485 nodes |
| dc22 | player colour 14 occupies the level-one goal block, or the source-derived player/goal signature is present on the atomic level-two frame | 23 actions | no plan, 20,017-node live limit |
| wa30 | the three boxes occupy `(28,28)`, `(32,28)`, `(36,28)` with none carried, or the source-derived level-two boundary signature is present | 33 actions | no plan, 20,001-node live limit |
| sb26 | at least 36 colour-12 pixels, the next-level-visible boundary marker emitted only after the four frame slots equal `(9,14,11,15)` | 9 actions | no plan, 20,025-node live limit |

The new expert files are under
`results/raw/experiment_10012_gate_usefulness/experts_v2/`; the original
positive-control evidence was not edited. The complete validation record,
including solution and expert hashes, is
`results/raw/experiment_10012_gate_usefulness/expert_predicate_validation.json`.

g50t was an expert-dynamics defect. The original model rewound the player but
never created the ghost that replays the stored path and holds the pressure
plate. Its queue exhausted after 712 nodes, far below either node cap. The goal
cell matched the public source, and the registered winning route uses only
uncapped keyboard actions 1-5, excluding the goal and action-candidate cap as
causes. The repaired expert models the ghost's eventual plate effect; it finds
the exact registered 17-action route in 13,931 nodes and levels up under
`LIVE_SCORED`. Evidence is
`results/raw/experiment_10012_gate_usefulness/g50t_investigation.json`.

### Stored-engine qualification survey

The nine Qwen3.8-27B sources use natural inline reasoning with the recorded
neutral prefix and a 102,400-token answer budget. Every qualifying window was
rebuilt twice through the same deterministic registered-solution collector.
Every transition field matched row by row, both canonical row hashes matched,
the historical transition count matched, the reset board was reproduced, and
the head-to-head harness recorded the offline stall-window/root planning
protocol. The full source SHA-256 values and per-pair checks are in
`results/raw/experiment_10012_gate_usefulness/stored_pair_qualification.json`.

| Surveyed source | Found | Qualified | Rejected | Reason |
|---|---:|---:|---:|---|
| `arc_qwen38_h2h_partial_20260817/engine_snapshots` | 3 | 3 | 0 | all strict checks passed |
| `arc_qwen38_h2h_stopped_20260817/engine_archive` | 6 | 6 | 0 | all strict checks passed |
| `exp6091_refine_engine_visible_shard.jsonl` | 26 | 0 | 26 | per-cell engine bytes not persisted or byte-exactly mapped |
| `exp5760_cegis_refinement_induction_shard.jsonl` | 78 | 0 | 78 | per-cell engine bytes not persisted or byte-exactly mapped |
| `exp5766_gemma31b_cegis_refinement_shard.jsonl` | 39 | 0 | 39 | per-cell engine bytes not persisted or byte-exactly mapped |
| `arc_generation_ablation_20260802/out/engines` | 225 | 0 | 225 | no recorded scored-stall reason establishes the required reset start |
| `arc_inert_rejection_ab_20260801/out/engines` | 151 | 0 | 151 | no recorded scored-stall reason establishes the required reset start |
| `arc_object_perception_ab_change_fidelity_20260801/engines` | 116 | 0 | 116 | no recorded scored-stall reason establishes the required reset start |
| `arc_object_perception_ab_20260728/e3` | 240 | 0 | 240 | no recorded scored-stall reason establishes the required reset start |
| `arc_wm_four_arm_20260727/e3_store` | 108 | 0 | 108 | no recorded scored-stall reason establishes the required reset start |
| `arc_engine_validation_20260731` | 7 | 0 | 7 | incomplete window/reason/model/think/budget mapping |
| `arc_logo_snapshot` | 6 | 0 | 6 | incomplete window/reason/model/think/budget mapping |

No new HUD mask was inferred for an added window: no simulator-twin proof of a
hidden-state HUD row was established, so all eight added windows are explicitly
unmasked. Existing Amendment 1 masks and exclusions are unchanged.

### Informativeness and candidate outcomes

`LIVE_SCORED` now has 10 informative windows out of 18: six in the
`expert_live_planner` category and four in the separately reported
`informative_by_registry_solver` category. The offline twin has 7 informative
windows; the NOT-LIVE 150K arm has 13. Of the nine qualified Qwen3.8 engines,
six lie on LIVE-informative windows. Two are useful: tu93 (registry-controlled)
and sp80 (expert-controlled). The other four evaluated pairs are two tr87
engines, lp85, and sk48; none levels up. sb26, ar25, and vc33 remain excluded
from LIVE gate tables because their available expert does not win under the
live planning budget.

### LIVE_SCORED gate tables

Cells are `accepted-positive / accepted-negative / rejected-positive /
rejected-negative`; `n` is the number of pairs on informative windows. The
Qwen3.8-only column is the main model-family result. Complete four-cohort tables
for `OFFLINE_TWIN_HALT` and `BUDGET_150K`, and separate Qwen3.5-family tables,
are stored in the terminal artifact under `gate_tables_by_arm` and
`gate_tables_by_model_family`.

#### USEFUL

| Gate | All pairs (n=38) | Candidates only (n=26) | Qwen3.8 only (n=6) | Expert-controlled windows (n=33) |
|---|---:|---:|---:|---:|
| live exact 1.0 | 4/1/5/28 | 1/1/2/22 | 1/1/1/3 | 3/0/5/25 |
| masked exact 1.0 | 6/2/3/27 | 1/2/2/21 | 1/1/1/3 | 5/1/3/24 |
| masked exact 0.875 | 6/5/3/24 | 1/5/2/18 | 1/1/1/3 | 5/4/3/21 |
| masked exact 0.75 | 6/7/3/22 | 1/7/2/16 | 1/1/1/3 | 5/6/3/19 |
| change fidelity 1.0, no-op 0 | 6/2/3/27 | 1/2/2/21 | 1/1/1/3 | 5/1/3/24 |
| change fidelity 1.0, no-op 0.25 | 6/2/3/27 | 1/2/2/21 | 1/1/1/3 | 5/1/3/24 |
| change fidelity 0.9, no-op 0 | 6/6/3/23 | 1/6/2/17 | 1/4/1/0 | 5/2/3/23 |
| change fidelity 0.9, no-op 0.25 | 6/6/3/23 | 1/6/2/17 | 1/4/1/0 | 5/2/3/23 |
| change fidelity 0.8, no-op 0 | 6/8/3/21 | 1/8/2/15 | 1/4/1/0 | 5/4/3/21 |
| change fidelity 0.8, no-op 0.25 | 6/8/3/21 | 1/8/2/15 | 1/4/1/0 | 5/4/3/21 |
| change fidelity 0.7, no-op 0 | 6/8/3/21 | 1/8/2/15 | 1/4/1/0 | 5/4/3/21 |
| change fidelity 0.7, no-op 0.25 | 6/8/3/21 | 1/8/2/15 | 1/4/1/0 | 5/4/3/21 |
| cell recall 0.9 | 6/7/3/22 | 1/7/2/16 | 1/4/1/0 | 5/3/3/22 |
| cell recall 0.8 | 6/7/3/22 | 1/7/2/16 | 1/4/1/0 | 5/3/3/22 |

#### FAITHFUL

| Gate | All pairs (n=38) | Candidates only (n=26) | Qwen3.8 only (n=6) | Expert-controlled windows (n=33) |
|---|---:|---:|---:|---:|
| live exact 1.0 | 3/2/3/30 | 0/2/2/22 | 0/2/1/3 | 3/0/3/27 |
| masked exact 1.0 | 3/5/3/27 | 0/3/2/21 | 0/2/1/3 | 3/3/3/24 |
| masked exact 0.875 | 3/8/3/24 | 0/6/2/18 | 0/2/1/3 | 3/6/3/21 |
| masked exact 0.75 | 4/9/2/23 | 1/7/1/17 | 0/2/1/3 | 4/7/2/20 |
| change fidelity 1.0, no-op 0 | 3/5/3/27 | 0/3/2/21 | 0/2/1/3 | 3/3/3/24 |
| change fidelity 1.0, no-op 0.25 | 3/5/3/27 | 0/3/2/21 | 0/2/1/3 | 3/3/3/24 |
| change fidelity 0.9, no-op 0 | 3/9/3/23 | 0/7/2/17 | 0/5/1/0 | 3/4/3/23 |
| change fidelity 0.9, no-op 0.25 | 3/9/3/23 | 0/7/2/17 | 0/5/1/0 | 3/4/3/23 |
| change fidelity 0.8, no-op 0 | 3/11/3/21 | 0/9/2/15 | 0/5/1/0 | 3/6/3/21 |
| change fidelity 0.8, no-op 0.25 | 3/11/3/21 | 0/9/2/15 | 0/5/1/0 | 3/6/3/21 |
| change fidelity 0.7, no-op 0 | 3/11/3/21 | 0/9/2/15 | 0/5/1/0 | 3/6/3/21 |
| change fidelity 0.7, no-op 0.25 | 3/11/3/21 | 0/9/2/15 | 0/5/1/0 | 3/6/3/21 |
| cell recall 0.9 | 3/10/3/22 | 0/8/2/16 | 0/5/1/0 | 3/5/3/22 |
| cell recall 0.8 | 3/10/3/22 | 0/8/2/16 | 0/5/1/0 | 3/5/3/22 |

No tested gate separates useful from not-useful candidates. In the main Qwen3.8
table, exact 1.0 accepts useful tu93 but also not-useful lp85 and rejects useful
sp80. The relaxed change-fidelity and recall gates accept tu93 and four failures
while still rejecting sp80. The same no-separation result holds for the broader
candidate-only cohort.

### Amendment 2 conclusion and checks

More windows changed the sample size, not the decision. Transition-fit gates
still do not order planning usefulness. A useful model appears at each extreme:
tu93 passes every gate, while sp80 passes none. The negative candidates overlap
both extremes. No gate or live default is changed.

The terminal run used CPU-only execution and no model call. The focused test
file passed 19 tests. Ruff check, Ruff format check, and mypy passed on the
changed harness and tests. `scripts/adversarial_verify.py` loaded the terminal
artifact and reported zero flags. Scoped spec coverage passed. The repository-wide
spec-coverage audit retains its pre-existing 1,168-test backlog; none is in the
changed Experiment 10012 test file. Exact commands and final outputs are retained
in the implementation handoff; the artifact's reproducibility checksum covers the
tables, source survey, predicate validation, and per-pair executions.

Amendment 2 limits:

- Registry-solver-controlled windows prove real reachability but do not prove an
  expert planner can discover the win; they remain a separate category.
- Three qualified Qwen engines (sb26, ar25, vc33) are excluded from LIVE tables
  because their expert-controlled windows remain uninformative under the live
  budget.
- The historical head-to-head evidence did not store transition-row hashes.
  Exact reconstruction therefore rests on recorded solution bytes, the original
  deterministic collector path, a historical row-count match, and two
  independent row-by-row-equal replays whose new canonical hashes are recorded.
- Added windows are unmasked because no hidden-state HUD twin proof was
  established. This is explicit rather than silently borrowing older masks.

## Correction (amendment 3) — 2026-09-24

Amendment 2's main Qwen3.8 table and its `1/1/1/3` claim are withdrawn. All
nine reused h2h engines are now appendix-only under
`h2h_replay_counterfactual`. Their windows are registered winning routes that
end in level-up, not stall windows; induction used the first two thirds of the
same solution; and RESET reproduces the h2h harness rather than the scored
agent's level-up reinduction rule. Their usefulness labels can therefore
include replay of a seen solution and are not comparable to the Experiment
10010 stall-window labels.

The Amendment 2 artifact is preserved byte-for-byte at
`results/raw/experiment_10012_gate_usefulness/v3_amendment2_artifact.json`
(SHA-256 `90e5f685be8b29fde8b07d83b1e3d5dec6f2708c646307ef9dccec568187f074`).
The corrected terminal artifact is schema v4. `LIVE_SCORED` execution itself
is unchanged.

### Historical qualification correction

None of the nine h2h snapshots meets Amendment 2's historical qualification
rule. Source bytes and filename SHA-12 values are genuine, and reset boards are
freshly reproducible. However, the row check compared two fresh rebuilds rather
than persisted historical rows. `offline_stall_window`, model, thinking mode,
and 102,400 token budget were passed as constants. The shards have no induction
reason or thinking-mode field and do not map the one snapshot per game to one
of three trial rows. They record a 102,400 budget and Qwen3.8 generator on the
available game rows, but `sb26` has no shard row because worker w0 ended with
rc=143. The corrected survey is 0 qualified and 9 rejected; all nine rows are
retained in the appendix rather than deleted.

### Missing measurements and held-out counts

A pair with zero scorable held-out rows now has null decisions for every gate.
It increments `unmeasured`, never a rejected cell. The nine retained Qwen3.8
pairs have these scorable held-out counts:

| Pair | Scorable held-out n | LIVE USEFUL | Gate measurement |
|---|---:|---:|---|
| tu93 `2bbd3f775cd4` | 3 | yes | measured |
| tr87 `d08020e4785f` | 3 | no | measured |
| tr87 `f0677d7eea70` | 3 | no | measured |
| sb26 `7a92b937751b` | 2 | no | window label uninformative |
| sp80 `46a3f1f57925` | 0 | yes | unmeasured |
| ar25 `b9ccf3583853` | 3 | no | window label uninformative |
| lp85 `392aab9f05a4` | 1 | no | measured |
| sk48 `416bf90b75e7` | 3 | no | measured |
| vc33 `555380ff6347` | 0 | no | window label uninformative |

Thus the six LIVE-informative appendix candidates contain five measured pairs
and one unmeasured pair. The two gate-evaluable useful candidates across the
whole experiment are the real-window `su15` THINK model (8 held-out rows, live
exact 0.125) and appendix `tu93` (3 rows, exact 1.0). `sp80` is useful only in
the counterfactual protocol and has no gate measurement. `lp85`'s exact-1.0
false acceptance rests on one row.

### Corrected LIVE_SCORED main tables

Main means only the five LIVE-informative Experiment 10010 stall windows, all
of which have expert live-planner controls. Cells are
`accepted-positive / accepted-negative / rejected-positive / rejected-negative / unmeasured`.
The with-controls table has `n=30, measured=30`; without controls has
`n=20, measured=20`. Every main gate has zero unmeasured pairs.

#### USEFUL

| Gate | With controls | Without controls |
|---|---:|---:|
| live exact 1.0 | 3/0/3/24/0 | 0/0/1/19/0 |
| masked exact 1.0 | 5/1/1/23/0 | 0/1/1/18/0 |
| masked exact 0.875 | 5/4/1/20/0 | 0/4/1/15/0 |
| masked exact 0.75 | 5/6/1/18/0 | 0/6/1/13/0 |
| change fidelity 1.0, no-op 0 | 5/1/1/23/0 | 0/1/1/18/0 |
| change fidelity 1.0, no-op 0.25 | 5/1/1/23/0 | 0/1/1/18/0 |
| change fidelity 0.9, no-op 0 | 5/2/1/22/0 | 0/2/1/17/0 |
| change fidelity 0.9, no-op 0.25 | 5/2/1/22/0 | 0/2/1/17/0 |
| change fidelity 0.8, no-op 0 | 5/4/1/20/0 | 0/4/1/15/0 |
| change fidelity 0.8, no-op 0.25 | 5/4/1/20/0 | 0/4/1/15/0 |
| change fidelity 0.7, no-op 0 | 5/4/1/20/0 | 0/4/1/15/0 |
| change fidelity 0.7, no-op 0.25 | 5/4/1/20/0 | 0/4/1/15/0 |
| cell recall 0.9 | 5/3/1/21/0 | 0/3/1/16/0 |
| cell recall 0.8 | 5/3/1/21/0 | 0/3/1/16/0 |

#### FAITHFUL

| Gate | With controls | Without controls |
|---|---:|---:|
| live exact 1.0 | 3/0/1/26/0 | 0/0/1/19/0 |
| masked exact 1.0 | 3/3/1/23/0 | 0/1/1/18/0 |
| masked exact 0.875 | 3/6/1/20/0 | 0/4/1/15/0 |
| masked exact 0.75 | 4/7/0/19/0 | 1/5/0/14/0 |
| change fidelity 1.0, no-op 0 | 3/3/1/23/0 | 0/1/1/18/0 |
| change fidelity 1.0, no-op 0.25 | 3/3/1/23/0 | 0/1/1/18/0 |
| change fidelity 0.9, no-op 0 | 3/4/1/22/0 | 0/2/1/17/0 |
| change fidelity 0.9, no-op 0.25 | 3/4/1/22/0 | 0/2/1/17/0 |
| change fidelity 0.8, no-op 0 | 3/6/1/20/0 | 0/4/1/15/0 |
| change fidelity 0.8, no-op 0.25 | 3/6/1/20/0 | 0/4/1/15/0 |
| change fidelity 0.7, no-op 0 | 3/6/1/20/0 | 0/4/1/15/0 |
| change fidelity 0.7, no-op 0.25 | 3/6/1/20/0 | 0/4/1/15/0 |
| cell recall 0.9 | 3/5/1/21/0 | 0/3/1/16/0 |
| cell recall 0.8 | 3/5/1/21/0 | 0/3/1/16/0 |

The corrected main candidate-only USEFUL table has one positive, `su15` THINK.
Every tested gate rejects it. Relaxing thresholds adds one to six false accepts
without accepting that useful candidate. This is a five-window, one-positive
observation, not a general separation result.

### Appendix LIVE_SCORED USEFUL tables

The appendix retains only label-informative rows in each listed cohort. Counts
again end with `unmeasured`. `h2h all` has `n=8, measured=5`; `h2h candidates`
has `n=6, measured=5`; registry-solver-controlled has `n=5, measured=5`; and
h2h expert-controlled has `n=3, measured=0` because the only LIVE-informative
expert-controlled h2h window is `sp80` and all three of its rows have zero
scorable held-out transitions.

| Gate or identical gate group | h2h all | h2h candidates | Registry solver | h2h expert |
|---|---:|---:|---:|---:|
| live exact 1.0 | 1/1/0/3/3 | 1/1/0/3/1 | 1/1/0/3/0 | 0/0/0/0/3 |
| masked exact 1.0 | 1/1/0/3/3 | 1/1/0/3/1 | 1/1/0/3/0 | 0/0/0/0/3 |
| masked exact 0.875 | 1/1/0/3/3 | 1/1/0/3/1 | 1/1/0/3/0 | 0/0/0/0/3 |
| masked exact 0.75 | 1/1/0/3/3 | 1/1/0/3/1 | 1/1/0/3/0 | 0/0/0/0/3 |
| change fidelity 1.0, either no-op cap | 1/1/0/3/3 | 1/1/0/3/1 | 1/1/0/3/0 | 0/0/0/0/3 |
| change fidelity 0.9, either no-op cap | 1/4/0/0/3 | 1/4/0/0/1 | 1/4/0/0/0 | 0/0/0/0/3 |
| change fidelity 0.8, either no-op cap | 1/4/0/0/3 | 1/4/0/0/1 | 1/4/0/0/0 | 0/0/0/0/3 |
| change fidelity 0.7, either no-op cap | 1/4/0/0/3 | 1/4/0/0/1 | 1/4/0/0/0 | 0/0/0/0/3 |
| cell recall 0.9 or 0.8 | 1/4/0/0/3 | 1/4/0/0/1 | 1/4/0/0/0 | 0/0/0/0/3 |

These appendix counts do not support the Amendment 2 claim that looser gates
add only bad-model accepts: `sk48` and tr87-d080 can be planner-budget failures,
and registry replay does not resolve that ambiguity. The no-separation pattern
remains visible, but only in this counterfactual, tiny, partially unmeasured
cohort.

### Control and planner corrections

`vc33` is not `planner_budget`: its expert exhausts the reachable reset-board
queue at 814 nodes in both LIVE and 150K, while the induction-state offline arm
finds a plan and levels up. The corrected reason is
`expert_dynamics_or_goal_gap_queue_exhausted_814_nodes`.

The `sp80` expert was not changed. Its validation covered one registered route
and the frozen window, not random play. Reviewer-measured real random play found
10 GAME_OVER frames where the unselected five-cell platform is exactly 80
colour-8 pixels, so the predicate fires at level zero. The expert also hardcodes
the successful bbox `(6,4,5,1)` and does not simulate the spill. Those are now
recorded control limits; no random-play revalidation is claimed.

The following planner explanation is recorded as **reviewer-measured, not
re-run**. Binary goal energy assigns every non-goal state 1.0, so heap ties are
FIFO and the nominal best-first search is BFS. Nodes count engine calls, with
13–37 candidates per state. HUD step/energy bars are in the dedup key (row 53
for sb26 and row 63 for dc22/wa30), so the same board at another step count is a
new state. Expert-only registered-route rollouts reach completion for sb26 in 9
actions, dc22 in 23, and wa30 in 33; every route action is in the candidate set.
sb26's BFS reaches depth 9 after 260,844 engine calls, just above 150K. wa30
reaches about depth 13 at 150K and depth 15 with 42,654 states at 400K. The
150K negative result therefore measures BFS search budget, not failed expert
dynamics, goals, or action candidates.

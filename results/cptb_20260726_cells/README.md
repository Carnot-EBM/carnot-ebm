# Raw cell rows — convention-perturbation transfer battery (2026-07-26)

One gzipped JSONL per game; one line per (arm, condition, seed) cell, 1500 lines total
(4 arms x 3 conditions x 25 games x 5 seeds, zero missing, zero errored).

Rows are produced by `scripts/experiments/cptb_run.py`, which calls the existing
`python/carnot/experiment_5836_frontier_discipline_ab.py:run_cell` so every row carries the
same instrumentation the shipped A/Bs carry (`states_expanded`, `errors`,
`hud_mask_resolved`, `frontier_discipline`, `hud_mask`) on both the success and the crash path.

Companion files:
  cptb_dose.json       static per-game convention-dose witness (reset-frame tier map + HUD mask
                       before/after each perturbation), from cptb_dose.py. REGENERATED
                       2026-07-26 to add the C3_roll_k1 / C3_roll_k2 dose axis; the C1 and C2
                       fields recompute BYTE-IDENTICALLY to the original file (verified
                       field-by-field on all 25 games), so this is a superset, not a revision.
  arm_receipt.json     per-arm requested-vs-resolved value of all seven gated flags, plus a
                       receipt that carnot.agentic.arc_game_adapters is never imported
  cptb_arms_dump.json  the four arms' exact constructor kwargs

Budget sweeps (they separate an efficiency regression from a capability loss — a loss at one
budget can be a budget WALL, as arm B2's cd82 once was). Aggregated into the artifact with
`n_seeds` attached to every bucket, because the first recorded reading rested on a single seed:
  budget_sweep_C0.jsonl.gz            original 1-seed sweep (tn36 + r11l at C0, budgets 4k/8k)
  budget_sweep_r11l_C1.jsonl.gz       original 1-2 seed sweep (r11l at C1, budgets 4k/8k/16k)
  sweep_tn36_C0_b8000.jsonl.gz        5-SEED re-run: tn36 at C0, budget 8000, all four arms
  sweep_r11l_C1_b8000_5seed.jsonl.gz  5-SEED re-run: r11l at C1, budget 8000, all four arms
  sweep_r11l_C1_b16000_5seed.jsonl.gz 5-SEED re-run: r11l at C1, budget 16000, all four arms

Roll-magnitude dose-response probe:
  probe_rollk_r11l_tn36.jsonl.gz      the two games the HUD lever moves, at roll k=1 and k=2,
                                      4 arms x 5 seeds (80 cells). Deliberately NOT part of the
                                      battery: it covers 2 of 25 games, so folding it into the
                                      corpus win sets would unbalance per-condition coverage.
                                      It answers one question -- is there ANY roll magnitude
                                      that violates the HUD edge-adjacency convention while
                                      leaving the support games winnable? Measured answer: no.

Reproduce:
    W=$(mktemp -d)
    CPTB_WORKDIR=$W .venv/bin/python scripts/experiments/cptb_dose.py   # -> $W/cptb_dose.json
    for g in $(ls environment_files); do
      .venv/bin/python scripts/experiments/cptb_run.py --games $g --seeds 5 --budget 2000 \
        --out $W/battery/$g.jsonl
    done
    # budget sweeps (any subset; the artifact aggregates whatever *sweep*.jsonl it finds)
    .venv/bin/python scripts/experiments/cptb_run.py --games tn36 --seeds 5 --budget 8000 \
      --conditions C0_real --out $W/sweep_tn36_C0_b8000.jsonl
    .venv/bin/python scripts/experiments/cptb_run.py --games r11l --seeds 5 --budget 8000 \
      --conditions C1_salience_inversion --out $W/sweep_r11l_C1_b8000_5seed.jsonl
    .venv/bin/python scripts/experiments/cptb_run.py --games r11l --seeds 5 --budget 16000 \
      --conditions C1_salience_inversion --out $W/sweep_r11l_C1_b16000_5seed.jsonl
    # roll-magnitude dose-response probe (*probe_rollk*, kept out of the battery)
    .venv/bin/python scripts/experiments/cptb_run.py --games r11l,tn36 --seeds 5 --budget 2000 \
      --conditions C3_roll_k1,C3_roll_k2 --out $W/probe_rollk_r11l_tn36.jsonl
    CPTB_WORKDIR=$W .venv/bin/python scripts/experiments/cptb_analyze.py
    CPTB_WORKDIR=$W .venv/bin/python scripts/experiments/cptb_artifact.py

Analysis artifact: results/outer_loop_cptb_shipped_lever_convention_transfer_20260726.json

NOTE (2026-07-26): the artifact was REBUILT after a 6-finding adversarial review. One claim was
withdrawn as uninterpretable (the HUD lever's convention verdict) and the honest_verdict changed.
The battery's 1500 cells are unchanged and were not re-run; what changed is the analysis, the gate
preconditions, and the added sweep/probe measurements. See ops/changelog.md 2026-07-26.

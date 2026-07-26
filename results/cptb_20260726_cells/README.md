# Raw cell rows — convention-perturbation transfer battery (2026-07-26)

One gzipped JSONL per game; one line per (arm, condition, seed) cell, 1500 lines total
(4 arms x 3 conditions x 25 games x 5 seeds, zero missing, zero errored).

Rows are produced by `scripts/experiments/cptb_run.py`, which calls the existing
`python/carnot/experiment_5836_frontier_discipline_ab.py:run_cell` so every row carries the
same instrumentation the shipped A/Bs carry (`states_expanded`, `errors`,
`hud_mask_resolved`, `frontier_discipline`, `hud_mask`) on both the success and the crash path.

Companion files:
  cptb_dose.json       static per-game convention-dose witness (reset-frame tier map + HUD mask
                       before/after each perturbation), from cptb_dose.py
  arm_receipt.json     per-arm requested-vs-resolved value of all seven gated flags, plus a
                       receipt that carnot.agentic.arc_game_adapters is never imported
  cptb_arms_dump.json  the four arms' exact constructor kwargs

Reproduce:
    W=$(mktemp -d)
    .venv/bin/python scripts/experiments/cptb_dose.py            # writes $W/cptb_dose.json
    for g in $(ls environment_files); do
      .venv/bin/python scripts/experiments/cptb_run.py --games $g --seeds 5 --budget 2000 \
        --out $W/battery/$g.jsonl
    done
    CPTB_WORKDIR=$W .venv/bin/python scripts/experiments/cptb_analyze.py
    CPTB_WORKDIR=$W .venv/bin/python scripts/experiments/cptb_artifact.py

Analysis artifact: results/outer_loop_cptb_shipped_lever_convention_transfer_20260726.json

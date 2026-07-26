"""THE BUDGET SWEEP on the SCORED path (E3AgentPolicy, shipped flag configuration).

WHY THIS SCRIPT EXISTS RATHER THAN A --budget LOOP AROUND THE EXISTING HARNESS. Two reasons, both
about not perturbing the thing being measured:

  1. `arc_scored_path_lever_harness.main()` REFUSES to run without a live llama-server
     (`if not server_ok: return 2`) even in its `--no-llm` arm, because that harness's whole
     point is
     LLM-ON cost. This sweep is LLM-OFF by design (the LLM-on cost is ~61x per cell at the same
     budget and cannot cover 6 budgets x 25 games x seeds), so it drives `run_cell` directly with
     `proposer=None`, which that function already supports.
  2. Every budget must be measured by the SAME instrumented cell function as the published
     lever A/B, or a budget delta could be an instrumentation delta. So `run_cell` is IMPORTED, not
     reimplemented, and the arm is `ARMS["S"]` -- all EIGHT gated flags pinned, which is what makes
     this the shipped configuration rather than "whatever the module globals said at run time".

MAX_ACTIONS IS NOT TOUCHED. The scored cap lives at `arc_competition_agent.py:6230`
(`CarnotAgent.MAX_ACTIONS = 400`) and is read by the competition framework's own loop. This sweep
varies the OFFLINE harness's `budget` parameter (`arc_leaderboard_eval.run_game(..., budget=N)`,
whose loop is `for step_index in range(budget)`), which is the parameter that SIMULATES a raised cap
without editing the shipped default. Sweep by parameter, not by editing the default.

LOOP ORDER IS DELIBERATE: seed -> game -> budget. Budget is the INNERMOST loop, so a run cut short
can never leave one budget with more measured cells than another within a completed (seed, game)
group -- the round-robin-by-arm discipline applied to the budget axis. And seeds complete one at a
time, so a truncated run still yields whole matched seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path("/home/ianblenke/github.com/ianblenke/carnot")
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts"))

import arc_scored_path_lever_harness as harness  # noqa: E402


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="200,400,1000,2000,4000,8000")
    ap.add_argument("--seeds", default="20260724,20260725")
    ap.add_argument("--games", default="")
    ap.add_argument("--arm", default="S")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    budgets = [int(x) for x in a.budgets.split(",") if x]
    seeds = [int(x) for x in a.seeds.split(",") if x]
    games = [g for g in a.games.split(",") if g] or list(harness.GAMES_25)

    parity = harness.assert_shipped_dict_matches_module_globals()
    if parity["pinned_vs_live_drift"]:
        print(f"[WARNING] pinned-vs-live flag drift: {parity['pinned_vs_live_drift']}", flush=True)

    rows: list[dict] = []
    t0 = time.time()
    total = len(seeds) * len(games) * len(budgets)
    for seed in seeds:
        for game in games:
            for budget in budgets:
                row = harness.run_cell(
                    game,
                    seed,
                    budget=budget,
                    proposer=None,
                    llm=False,
                    extra_kwargs=dict(harness.ARMS[a.arm]),
                    arm=f"{a.arm}_llmoff_b{budget}",
                )
                rows.append(row)
                print(
                    f"[{len(rows):4}/{total}] {game:5} s{seed} b{budget:<6} "
                    f"lv={row.get('levels')} act={row.get('actions')} "
                    f"st={row.get('states_expanded')} wall={row.get('wall_s')}s "
                    f"eff={row.get('efficiency')} atfl={row.get('actions_to_first_levelup')} "
                    f"L1={row.get('lever1_fired')} L2={row.get('lever2_fired')} "
                    f"hudres={row.get('hud_mask_resolved')} "
                    f"hudcells={row.get('hud_mask_cell_count')} "
                    f"hudread={row.get('hud_diagnostics_readable')} "
                    f"L3={row.get('lever3_verdict')} "
                    f"prevframe={row.get('nodes_with_previous_frame')}/{row.get('nodes_total')} "
                    f"ran={row.get('ran')} cum={round(time.time() - t0, 1)}s",
                    flush=True,
                )
                Path(a.out).write_text(
                    json.dumps(
                        {
                            "sweep": "arc_scored_path_budget_sweep",
                            "rows": rows,
                            "budgets_requested": budgets,
                            "seeds_requested": seeds,
                            "games_requested": games,
                            "arm": a.arm,
                            "arm_flags": dict(harness.ARMS[a.arm]),
                            "llm_enabled": False,
                            "scored_agent_max_actions": 400,
                            "flag_parity_vs_live_globals": parity,
                            "elapsed_s": round(time.time() - t0, 1),
                            "rows_checksum": hashlib.sha256(
                                json.dumps(
                                    [
                                        [
                                            r.get("game"),
                                            r.get("seed"),
                                            r.get("budget"),
                                            r.get("levels"),
                                            r.get("actions"),
                                        ]
                                        for r in rows
                                    ],
                                    sort_keys=True,
                                ).encode()
                            ).hexdigest(),
                        },
                        indent=1,
                    )
                )
    print(f"TOTAL {round(time.time() - t0, 1)}s n={len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

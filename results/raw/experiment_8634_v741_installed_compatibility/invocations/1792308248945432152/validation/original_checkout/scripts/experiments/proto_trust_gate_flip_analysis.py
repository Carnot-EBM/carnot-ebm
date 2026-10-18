"""Measure Avenue A: does flipping CARNOT_ARC_TRUST_METRIC=cell_recall lift live first-win?

The live env-var gate (arc_competition_agent.py:1806) governs the e3 LLM/DSL-INDUCTION path: it scores
the INDUCED world-model with WorldModelVerifier and skips it when the gate value < 0.5. Default 'exact'
(full-grid accuracy); the lever flips it to 'cell_recall' (graded changed-cell recall).

A live A/B is unnecessary to answer this: the agent's behavior changes ONLY if the gate DECISION changes
on some game. The gate decision changes only on a game that is exact-FAIL but cell_recall-PASS. So we
cross-tabulate the two existing per-game probes (real measurements) and count gate-flips per path. If
zero gap-1 games flip on the path the env-var governs, flipping the flag is a provable no-op live
(identical gate -> identical plan-fire -> identical first-win), and no LLM run is needed.
"""

from __future__ import annotations

import json

GATE = 0.5


def main() -> int:
    ttt = json.load(open("results/arc_ttt_loo_gate_probe.json"))
    e3 = json.load(open("results/arc_e3_induced_model_quality.json"))

    # e3 LLM/DSL-induction path = the path the live CARNOT_ARC_TRUST_METRIC gate governs
    e3_rows = []
    e3_flips = 0
    for g in e3.get("per_game", []):
        ea, cr = g.get("exact_accuracy"), g.get("cell_recall")
        if ea is None or cr is None:
            continue
        flip = ea < GATE <= cr  # exact-FAIL but cell_recall-PASS
        e3_flips += int(flip)
        e3_rows.append({"game": g["game"], "exact_accuracy": ea, "cell_recall": cr,
                        "exact_gate": "PASS" if ea >= GATE else "FAIL",
                        "cell_gate": "PASS" if cr >= GATE else "FAIL", "flips_fail_to_pass": flip})

    # TTT learned-dynamics path = a DIFFERENT mechanism the env-var gate does NOT govern
    ttt_rows = []
    ttt_flips = 0
    for g in ttt.get("per_game", []):
        ew = g.get("exact_warm", {})
        ea, cr = ew.get("exact"), ew.get("cell_recall")
        if ea is None or cr is None:
            continue
        flip = ea < GATE <= cr
        ttt_flips += int(flip)
        ttt_rows.append({"game": g["game"], "exact": ea, "cell_recall": cr,
                         "flips_fail_to_pass": flip})

    live_noop = e3_flips == 0
    result = {
        "question": "does flipping CARNOT_ARC_TRUST_METRIC=cell_recall lift live first-win?",
        "governed_path": "e3_llm_dsl_induction (the live env-var gate path)",
        "e3_path_gate_flips": e3_flips,
        "e3_path_per_game": e3_rows,
        "ttt_path_gate_flips": ttt_flips,
        "ttt_path_per_game": ttt_rows,
        "live_first_win_provably_unchanged": live_noop,
        "VERDICT": (
            "DEAD_live_gate_flip_is_a_noop_induction_quality_is_the_wall" if live_noop
            else "LIVE_gate_flip_changes_decisions"
        ),
        "why": (
            "On the e3 path the flag governs, ZERO gap-1 games are exact-FAIL+cell_recall-PASS: the "
            "LLM/DSL-induced dynamics have cell_recall ~0 (cn04 0.0146, cd82 0.0, sc25 0.0547) -- they "
            "are WRONG, not imperfect-but-useful, so neither gate trusts them and flipping changes no "
            "decision -> identical agent -> identical first-win. Induction QUALITY is the wall, not the "
            "gate."
        ),
        "sharper_untested_lever": (
            f"The TTT learned-dynamics path flips {ttt_flips} games FAIL->PASS (ka59 0.91, sc25 0.80, "
            "tn36 0.87, lp85 0.59 cell_recall) -- those models ARE imperfect-but-useful. The real lever "
            "is to ROUTE live trust to the TTT dynamics on those games (not flip the gate on the e3 "
            "path), then measure whether a trusted TTT model drives plan_in_model to a live win. That "
            "needs the TTT CNN wired into the live plan path + a GPU run -- genuinely untested."
        ),
    }
    with open("results/proto_trust_gate_flip_analysis.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: result[k] for k in
                      ("e3_path_gate_flips", "ttt_path_gate_flips",
                       "live_first_win_provably_unchanged", "VERDICT")}, indent=2))
    print("\ne3 path (flag-governed):")
    for r in e3_rows:
        print(f"  {r['game']:6} exact_acc={r['exact_accuracy']:.3f} cell_recall={r['cell_recall']:.3f} "
              f"-> exact={r['exact_gate']} cell={r['cell_gate']} flip={r['flips_fail_to_pass']}")
    print("TTT path (NOT flag-governed):")
    for r in ttt_rows:
        print(f"  {r['game']:6} exact={r['exact']:.3f} cell_recall={r['cell_recall']:.3f} "
              f"flip={r['flips_fail_to_pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

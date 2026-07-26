"""RUNNER for the convention-perturbation transfer battery.

Executes (arm x game x condition x seed) cells through the EXISTING, already-instrumented
`experiment_5836_frontier_discipline_ab.run_cell`, so every row carries the same
`states_expanded`, `errors`, `hud_mask_resolved` and frontier/HUD diagnostics the shipped
A/Bs carry -- including on the crash path, which is what stops a partially-crashed arm from
averaging out to a plausible-looking null.

LOOP ORDER IS `for game: for seed: for condition: for arm:` -- ROUND ROBIN BY ARM.  The
upstream harness nests arm-outermost, so a run killed partway leaves the first arm complete
and the last arm empty.  Here a kill at any point leaves every arm covered on the same
(game, seed, condition) prefix, so a truncated run is still a balanced (smaller) experiment.

Rows are appended to a JSONL as they are produced, so a crash loses at most one cell.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cptb_perturb as P  # noqa: E402, N812
from cptb_arms import CPTB_ARMS  # noqa: E402

# (label, variant, reflect)
#
# C3_* are the DOSE AXIS of the geometric perturbation, added 2026-07-25.  C2 (k=3) was
# measured to raze the corpus -- the pre-flip control drops 7 wins -> 1 -- which auto-falsifies
# any narrow-support lever there for reasons unrelated to that lever's convention.  The smaller
# magnitudes exist so the dose at which the HUD Stage-1 predicate stops firing can be separated
# from the dose at which the games stop being winnable at all.  They are additional conditions,
# not replacements: C0/C1/C2 keep their exact meaning and every recorded row stays valid.
CONDITIONS = (
    ("C0_real", 0, None),
    ("C1_salience_inversion", P.VARIANT_SALIENCE_INVERSION, None),
    ("C2_diag_roll", P.VARIANT_IDENTITY_COLOR, P.REFLECT_DIAG_ROLL),
    ("C3_roll_k1", P.VARIANT_IDENTITY_COLOR, P.reflect_code_for_roll_k(1)),
    ("C3_roll_k2", P.VARIANT_IDENTITY_COLOR, P.reflect_code_for_roll_k(2)),
)

BASE_SEED = 20260726


def _palettes(games):
    import numpy as np

    from carnot.agentic import arc_solver_kit as kit

    arc = kit.offline_arcade()
    sc = arc.open_scorecard()
    out = {}
    for g in games:
        env = arc.make(g, scorecard_id=sc)
        st = np.array(env.reset().frame)
        if st.ndim == 2:
            st = st[None, ...]
        out[g] = sorted({int(c) for c in np.unique(st)})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="")
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conditions", default="")
    args = ap.parse_args(argv)

    import carnot.experiment_5836_frontier_discipline_ab as H  # noqa: N812

    games = (
        tuple(g.strip() for g in args.games.split(",") if g.strip()) if args.games else H.ALL_GAMES
    )
    conds = [c for c in CONDITIONS if not args.conditions or c[0] in args.conditions.split(",")]

    # Perturbations must be installed BEFORE any cell runs; the palette map is read from the
    # games' own reset frames (dev-side environment diagnosis, never handed to the agent).
    P.install(_palettes(games))

    # Register the CPTB arms into the harness's ARMS table -- run_cell reads
    # ARMS[arm]["kwargs"], so this is how the new arms reach the constructor.  The existing
    # arms are left untouched so nothing already recorded changes meaning.
    for name, spec in CPTB_ARMS.items():
        H.ARMS[name] = {
            "label": spec["label"],
            "kwargs": dict(spec["kwargs"]),
            "deterministic": spec["deterministic"],
        }

    seeds = [BASE_SEED + i for i in range(max(1, args.seeds))]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    t_start = time.time()
    with out.open("a") as fh:
        for game in games:
            for seed in seeds:
                for cond_label, variant, reflect in conds:
                    # ROUND ROBIN BY ARM -- innermost loop.
                    for arm in CPTB_ARMS:
                        t0 = time.time()
                        try:
                            row = H.run_cell(
                                arm,
                                game,
                                budget=args.budget,
                                seed=seed,
                                variant=variant,
                                reflect=reflect,
                            )
                        except Exception as exc:  # pragma: no cover - attributed to its arm
                            row = {
                                "arm": arm,
                                "game": game,
                                "seed": seed,
                                "ran": False,
                                "reason": f"HARNESS:{type(exc).__name__}:{exc}",
                                "errors": 1,
                                "states_expanded": None,
                            }
                        row["condition"] = cond_label
                        row["variant"] = variant
                        row["reflect"] = reflect
                        row["budget"] = args.budget
                        row["cell_wall_s"] = round(time.time() - t0, 3)
                        fh.write(json.dumps(row, default=str) + "\n")
                        fh.flush()
                        n += 1
                        print(
                            f"[{n}] {game} s{seed} {cond_label} {arm} "
                            f"ran={row.get('ran')} lv={row.get('levels')} "
                            f"act={row.get('actions')} st={row.get('states_expanded')} "
                            f"err={row.get('errors')} hud={row.get('hud_mask_resolved')} "
                            f"{row['cell_wall_s']}s",
                            flush=True,
                        )
    print(f"DONE n_cells={n} wall_s={round(time.time() - t_start, 1)} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

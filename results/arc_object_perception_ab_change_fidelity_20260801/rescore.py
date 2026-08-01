#!/usr/bin/env python3
"""POST-HOC PASS: add the channels `run_ab.py` did not record, and the baselines the metric
was never tested against. Runs AFTER collection stops. No LLM, no GPU, no new generation.

WHAT THIS ADDS AND WHY EACH ONE IS HERE. Every item traces to a specific adversarial-review
finding that was reproduced on frozen data before being acted on.

  1. THE NO-OP HALLUCINATION CHANNEL (`n_noop`, `n_noop_hallucinated`,
     `noop_hallucination_rate`, `noop_channel_measurable`) and `invented_changed_cells`.
     `change_fidelity` averages over CHANGING transitions only, so an engine that models every
     real change correctly AND invents a change on every no-op scores a PERFECT primary.
     Reproduced here on frozen split data: on sc25 that engine scores change_fidelity 1.0000
     at full-grid accuracy 0.0714; on ft09, 1.0000 at 0.2000. The secondary an operator would
     reach for -- `spurious_changed_cells` -- reads 0 on it, because that counter is only
     accumulated inside changing transitions. Only `noop_hallucination_rate` names it, and it
     was not in the pre-registered secondaries. This is a field copy, not new compute.

  2. THE INERTNESS + REPLAY BASELINES (`baseline_worker.py`). The headroom artifact's own
     disqualification criterion for a metric is "does it rank a non-model above a real
     engine", and it was only ever applied to the INERT engine. Delta-replay was not tested.

  3. THE ACTION / COORDINATE BLINDNESS PROBE (`rescore_worker.py`). Neither was blindness.
     This matters most on CLICK games, where an engine can be perfectly correct on the graded
     window while provably unable to see the click.

  4. `hud_mask_status` per cell, so the artifact states rather than implies which arm of
     REQ-ARC-WMTE-6010 produced every number.

WHY IT IS SOUND TO RE-DERIVE RATHER THAN RE-RUN. Both inputs are frozen: the engine text in
`e3_store/`, and a window rebuilt by the SAME deterministic `build_progress_window` +
`_split_prefix_heldout` calls `run_ab.py` makes. The pass therefore RE-DERIVES the two fields
run_ab.py already recorded and compares them. If any cell's `change_fidelity` disagrees, the
rebuild is not deterministic, `reproduction_ok` is False, and the added fields are void rather
than quietly merged -- a mismatch is reported, never averaged over.

WHAT THIS DELIBERATELY DOES NOT DO. It does not re-grade with the HUD mask on. `logical_hud_mask`
needs a FRAME-coordinate mask from the live agent's HUD detector plus the frame/logical cell
size, and the A/B cells record neither -- only transitions. So the masked arm is not
reconstructible from this run's evidence, and the artifact says so instead of pretending the
question was answered. Both arms were graded unmasked, so the mask is a common-mode setting
and cannot by itself flip the direction of a per-game delta.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
SCRATCH = HERE / "rescore_scratch"
REPO = Path("/home/ianblenke/github.com/ianblenke/carnot")
WORKER_TIMEOUT_S = 120.0
# Window rebuilds step a real environment, so they get a longer bound than a scoring pass.
# tr87 exceeded 8 minutes in two separate sweeps without returning, so this is a real
# ceiling rather than a formality.
WINDOW_TIMEOUT_S = 240.0


def build_windows(roster: list[str]) -> tuple[dict[str, Path], dict[str, dict]]:
    """Rebuild every roster window ONCE and pickle it, each in its own KILLABLE process.

    Windows are built once rather than per cell so the ~2 s env build is not paid 124 times.

    THE ISOLATION IS NOT OPTIONAL AND WAS ADDED AFTER IT BIT. The first version of this
    function ran `build_progress_window` inline in the driver, on the reasoning that the
    dangerous thing to execute was LLM-written engine code. `build_progress_window("tr87")`
    then span at 100% CPU without returning and took the entire pass down -- in two
    independently-written sweeps, the same game both times. It steps a real environment and
    creates a scorecard; it is not arithmetic and it has no internal bound. A game that cannot
    be rebuilt inside the timeout is DROPPED from the post-hoc pass with its reason recorded:
    its A/B cells keep the numbers run_ab.py already measured and simply gain no added
    channels. A stated coverage gap, never a silent zero.
    """
    SCRATCH.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    status: dict[str, dict] = {}
    for game in roster:
        p = SCRATCH / f"{game}_window.pkl"
        r = run_worker(
            "window_worker.py",
            {"game": game, "window_pkl": str(p)},
            f"win_{game}",
            timeout=WINDOW_TIMEOUT_S,
        )
        status[game] = r
        if r.get("status") == "ok" and p.exists():
            paths[game] = p
            print(f"  {game}: shown={r['n_shown']} held={r['n_heldout']} cell={r['cell']}")
        else:
            print(f"  {game}: WINDOW NOT REBUILT ({r.get('status')})")
    return paths, status


def run_worker(worker: str, job: dict, tag: str, timeout: float = WORKER_TIMEOUT_S) -> dict:
    jp = SCRATCH / f"job_{tag}.json"
    jp.write_text(json.dumps(job))
    env = dict(os.environ, CARNOT_REPO=str(REPO))
    try:
        pr = subprocess.run(  # noqa: S603
            [sys.executable, str(HERE / worker), str(jp)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        # UNDETERMINED, not a zero. Nothing about this engine was measured, so it leaves both
        # numerator and denominator -- the same rule the frozen best-of-N run applied to its
        # one non-terminating candidate.
        return {"status": "undetermined_worker_timeout"}
    line = (pr.stdout or "").strip().splitlines()
    if not line:
        return {"status": "worker_no_output", "stderr": (pr.stderr or "")[-300:]}
    try:
        return json.loads(line[-1])
    except json.JSONDecodeError:
        return {"status": "worker_bad_output", "stdout": line[-1][:300]}


def main() -> int:
    t0 = time.time()
    rows = json.loads((OUT / "rows.json").read_text())
    meta_p = OUT / "meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    roster = meta.get("roster") or sorted({r["game"] for r in rows})
    # SMOKE-TEST ONLY. Lets this pass be exercised on 2 games while collection is still
    # running, so a defect in it is found before it is the only thing between the frozen
    # cells and the artifact. The real pass runs with the variable unset.
    only = [g for g in os.environ.get("RESCORE_ONLY", "").split(",") if g]
    if only:
        roster = [g for g in roster if g in only]
        rows = [r for r in rows if r["game"] in roster]

    print(f"rebuilding {len(roster)} windows")
    windows, window_status = build_windows(roster)

    # ---- baselines, one killable worker per game ------------------------------------
    print("\nbaselines")
    baselines: dict[str, dict] = {}
    for game, wp in windows.items():
        r = run_worker("baseline_worker.py", {"game": game, "window_pkl": str(wp)}, f"base_{game}")
        baselines[game] = r
        if r.get("status") == "ok":
            b = r["baselines"]
            print(
                f"  {game:<6} identity={b['IDENTITY']['change_fidelity']:<10}"
                f"replay={b['MODAL_SHOWN_DELTA_REPLAY']['change_fidelity']:<10}"
                f"oracle={b['ORACLE_ceiling']['change_fidelity']:<10}"
                f"n_noop={b['IDENTITY']['n_noop']} alias={r['n_aliased_heldout_states']}"
            )
        else:
            print(f"  {game:<6} {r.get('status')}")

    # ---- per-cell rescore -------------------------------------------------------------
    print("\nper-cell rescore")
    rescored: list[dict] = []
    n_mismatch = 0
    for r in rows:
        game, rep, tag = r["game"], r["replicate"], r["tag"]
        cell = f"{game}__r{rep}__{tag}"
        wp = windows.get(game)
        if wp is None:
            rescored.append({"cell": cell, "status": "no_window"})
            continue
        store = HERE / "e3_store" / cell
        cands = sorted(store.rglob("world_model.py"))
        code = cands[0] if cands else store / game / "world_model.py"
        res = run_worker(
            "rescore_worker.py",
            {"cell": cell, "window_pkl": str(wp), "code_path": str(code)},
            cell,
        )
        res["game"], res["replicate"], res["arm"] = game, rep, tag

        # REPRODUCTION CHECK against what run_ab.py recorded for this same cell.
        h = r.get("heldout") or {}
        if res.get("status") == "ok" and h.get("measurable"):
            a, b = h.get("change_fidelity"), res["full"]["change_fidelity"]
            ok = a is not None and abs(float(a) - float(b)) < 1e-6
            res["reproduces_run_ab_change_fidelity"] = bool(ok)
            res["run_ab_change_fidelity"] = a
            if not ok:
                n_mismatch += 1
                print(f"  MISMATCH {cell}: run_ab={a} rescore={b}")
        rescored.append(res)

    ok_cells = [x for x in rescored if x.get("status") == "ok"]
    checked = [x for x in ok_cells if "reproduces_run_ab_change_fidelity" in x]
    repro_ok = all(x["reproduces_run_ab_change_fidelity"] for x in checked)

    # ---- the two disqualifier questions, answered per game ----------------------------
    per_game_disq = {}
    for game in roster:
        b = baselines.get(game, {})
        if b.get("status") != "ok":
            continue
        bl = b["baselines"]
        engines = [
            x for x in ok_cells if x["game"] == game and x.get("full", {}).get("n_changing", 0) > 0
        ]
        eng_cf = [x["full"]["change_fidelity"] for x in engines]
        best_eng = max(eng_cf) if eng_cf else None
        non_model = {
            "IDENTITY": bl["IDENTITY"]["change_fidelity"],
            "MODAL_SHOWN_DELTA_REPLAY": bl["MODAL_SHOWN_DELTA_REPLAY"]["change_fidelity"],
        }
        blind = [x for x in engines if x.get("behaviourally_blind")]
        aware = [x for x in engines if x.get("behaviourally_blind") is False]
        per_game_disq[game] = {
            "oracle_ceiling": bl["ORACLE_ceiling"]["change_fidelity"],
            "n_aliased_heldout_states": b["n_aliased_heldout_states"],
            "best_real_engine": best_eng,
            "non_model_scores": non_model,
            # THE DISQUALIFIER the headroom artifact applied to the object metrics, now
            # applied to the primary on the A/B's OWN roster.
            "a_non_model_outranks_every_real_engine": bool(
                best_eng is not None and max(non_model.values()) > best_eng
            ),
            "n_behaviourally_blind_engines": len(blind),
            "n_action_sensitive_engines": len(aware),
            "best_blind_engine": max((x["full"]["change_fidelity"] for x in blind), default=None),
            "best_aware_engine": max((x["full"]["change_fidelity"] for x in aware), default=None),
            "blind_outranks_aware": bool(
                blind
                and aware
                and max(x["full"]["change_fidelity"] for x in blind)
                > max(x["full"]["change_fidelity"] for x in aware)
            ),
            "noop_channel_measurable": bl["IDENTITY"]["noop_channel_measurable"],
        }

    out = {
        "what_this_is": (
            "post-hoc field addition + baselines over the FROZEN A/B cells. No LLM, no GPU, "
            "no change to the treatment. Every number re-derived from engine text on disk and "
            "a deterministically rebuilt window."
        ),
        "duration_s": round(time.time() - t0, 2),
        "n_cells": len(rescored),
        "n_cells_ok": len(ok_cells),
        "window_rebuild": {
            "n_requested": len(window_status),
            "n_rebuilt": len(windows),
            "not_rebuilt": {
                g: r.get("status") for g, r in window_status.items() if r.get("status") != "ok"
            },
            "why_a_drop_is_not_a_zero": (
                "a game whose window could not be rebuilt inside the worker timeout keeps the "
                "numbers run_ab.py measured and gains no ADDED channels. It is absent from the "
                "post-hoc checks, not scored 0 in them."
            ),
        },
        "reproduction_check": {
            "n_cells_checked": len(checked),
            "n_mismatch": n_mismatch,
            "all_reproduce_run_ab_change_fidelity": bool(repro_ok),
            "why_this_gate_exists": (
                "the added fields are only trustworthy if the rebuilt window is the same "
                "window run_ab.py graded. If any cell's change_fidelity disagrees, the whole "
                "pass is void rather than partly merged."
            ),
        },
        "noop_channel_roster_wide": {
            "n_games_with_measurable_noop_channel": sum(
                1 for v in per_game_disq.values() if v["noop_channel_measurable"]
            ),
            "n_games": len(per_game_disq),
            "reading": (
                "noop_hallucination_rate returns 0.0 when n_noop == 0, and 0.0 is ALSO the "
                "value meaning 'this engine invents nothing'. Where the channel is not "
                "measurable the 0.0 is a dead channel wearing a passing score, not evidence "
                "of a clean engine."
            ),
        },
        "per_game_disqualifier_checks": per_game_disq,
        "baselines": baselines,
        "cells": rescored,
    }
    (OUT / "rescore.json").write_text(json.dumps(out, indent=2))
    print(
        f"\nwrote {OUT / 'rescore.json'}  cells_ok={len(ok_cells)}/{len(rescored)}  "
        f"reproduction_ok={repro_ok} ({len(checked)} checked, {n_mismatch} mismatch)  "
        f"{out['duration_s']}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

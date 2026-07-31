#!/usr/bin/env python3
"""BEST-OF-N, STEP 4 -- score every candidate on criteria (i)/(ii)/(iii) and report yield at N.

THE THREE CRITERIA, in increasing strictness, exactly as the operator posed them:

  (i)   held-out DYNAMICS accuracy -- what the shipped trust gate uses.
  (ii)  (i) AND `goal_predicate_satisfiable` within the shipped depth.
  (iii) (ii) AND a plan actually found by `plan_in_model` at the shipped budget.

WHY ALL THREE AND NOT JUST (i). Selecting on dynamics alone would reliably produce more tn36s:
tn36 round 1 is the best engine measured anywhere in this project -- held-out accuracy 1.0,
trust energy -3.219856 (the grid minimum), 25 of 25 changing transitions exact, cell recall 1.0 --
and it banked ZERO levels across 346 actions, because it died at the goal gate. A criterion that
cannot see that failure will select for it. So each criterion's yield is reported separately, AND
the conditional `select on (i), then check (ii)/(iii)` is reported, which is the direct measurement
of whether the cheap criterion selects for the expensive failure.

THE VACUITY TRAP, and how this scorer avoids it. A `level_up_reinduction` at transition_count=1
passes both gates near-trivially, so a criterion evaluated on the post-bank path selects for
triviality and reproduces the reverse-causal artifact the mediation analysis found. The six
captured games split cleanly and the split is MEASURED, not assumed: five (ft09, lp85, sc25,
tn36, tu93) are STALL inductions -- 25 collected transitions, `levels_gained` 0 -- and vc33 is a
post-bank one, 2 levels gained, 1 transition, 0 held-out rows. The headline is computed on the
five stall games ONLY. vc33 is reported alongside as the contrast that demonstrates the trap
rather than being silently dropped.

SCORING IS OUT-OF-SAMPLE, against the split PROVEN by `split.py`: HELD_OUT = full \\ shown, where
`shown` is the <=8 rows `_transitions_block` actually renders. A row whose rendered delta line is
ambiguous (a duplicate of a shown row) is dropped from held-out rather than scored as unseen.

TWO BARS ARE REPORTED FOR (i), because one of them is known to admit junk. The SHIPPED bar is
`heldout_accuracy >= verifier_threshold` with `verifier_threshold = min_heldout_accuracy = 1.0`
at both live call sites -- so, accuracy exactly 1.0. On a held-out set with no CHANGING rows
(lp85 here) the identity function scores 1.0 on that bar, which is why the strict bar is carried
alongside: at least one held-out changing transition predicted EXACTLY, falling back to zero
hallucinated no-ops where nothing changing is held out. Both are in the payload; where they
disagree, the disagreement is the result.

ACCEPT-FIRST vs VERIFIER-SELECT AT THE SAME N. The shipped path takes the FIRST candidate that
clears a mechanical bar. Best-of-N takes the one the VERIFIER ranks highest. Reporting both over
the SAME N samples separates "more sampling helped" from "verification helped" -- without it a
gain at N=8 could be entirely explained by having drawn more samples, which is not this project's
thesis.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import pickle
import subprocess
import sys
import time

REPO = "/home/ianblenke/github.com/ianblenke/carnot"
HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / "bestofn_scored.json"

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("CARNOT_ARC_E3_DIR", "/tmp/arc_bon_score/e3")
os.makedirs(os.environ["CARNOT_ARC_E3_DIR"], exist_ok=True)
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, str(HERE))

TAGS = [t for t in os.environ.get("BON_TAGS", "gpu1").split(",") if t]
CALL_INDEX = int(os.environ.get("BON_CALL_INDEX", "1"))
GATE_TIMEOUT_S = int(os.environ.get("BON_GATE_TIMEOUT_S", "900"))
GATE_WORKERS = int(os.environ.get("BON_GATE_WORKERS", "6"))
PY = sys.executable
SCRATCH = pathlib.Path(os.environ.get("BON_SCRATCH", "/tmp/arc_bon_score/code"))
SCRATCH.mkdir(parents=True, exist_ok=True)

# The live threshold: `execute_bounded_llm_reinduction(..., min_heldout_accuracy=1.0)` at both
# call sites in arc_competition_agent, and `verifier_threshold = min(1.0, min_heldout_accuracy)`.
SHIPPED_HELDOUT_THRESHOLD = 1.0
NS = [1, 4, 8]


def _run_gates(jobs: list[tuple[str, str, str]]) -> dict[str, dict]:
    """Run gate_worker over `jobs` with bounded parallelism. Each job is (key, code, root)."""
    results: dict[str, dict] = {}
    pending = list(jobs)
    running: list[tuple[str, subprocess.Popen, float]] = []
    while pending or running:
        while pending and len(running) < GATE_WORKERS:
            key, code_path, root_path = pending.pop(0)
            proc = subprocess.Popen(
                [PY, str(HERE / "gate_worker.py"), code_path, root_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            running.append((key, proc, time.monotonic()))
        time.sleep(0.25)
        for item in list(running):
            key, proc, t0 = item
            if proc.poll() is None:
                if time.monotonic() - t0 > GATE_TIMEOUT_S:
                    # A subprocess killed from OUTSIDE cannot be swallowed by the generated
                    # code's own `except Exception` -- see gate_worker's docstring.
                    proc.kill()
                    proc.wait(timeout=30)
                    results[key] = {
                        "status": "gate_timeout",
                        "gate_timeout_s": GATE_TIMEOUT_S,
                        "goal_satisfiable": False,
                        "goal_kind": "gate_timeout",
                        "plan_found": False,
                    }
                    running.remove(item)
                continue
            out, err = proc.communicate()
            running.remove(item)
            try:
                results[key] = json.loads(out.strip().splitlines()[-1])
            except Exception:  # noqa: BLE001
                results[key] = {
                    "status": f"gate_worker_rc{proc.returncode}",
                    "stderr": (err or "")[-300:],
                    "goal_satisfiable": False,
                    "goal_kind": "gate_worker_failed",
                    "plan_found": False,
                }
    return results


def main() -> int:  # noqa: C901
    import numpy as np

    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic.arc_world_model_trust_energy import (
        WorldModelCandidate,
        select_trusted_world_model,
    )
    from split import load_split

    runs: dict[str, dict] = {}
    for tag in TAGS:
        p = HERE / "bon" / tag / "bon.json"
        if p.exists():
            runs[tag] = json.loads(p.read_text())

    if not runs:
        json.dump({"status": "NOT_RUN", "tags": TAGS}, open(OUT, "w"), indent=2)
        print("no runs found")
        return 2

    games = sorted({g for r in runs.values() for g in r.get("games", [])})
    splits = {g: load_split(g, CALL_INDEX) for g in games}
    caps = {
        g: json.loads((HERE / "capture" / g / "capture.json").read_text())
        for g in games
        if (HERE / "capture" / g / "capture.json").exists()
    }
    # THE STALL / POST-BANK PARTITION, measured from the capture rather than assumed.
    stall_games = sorted(g for g in games if int(caps.get(g, {}).get("levels_gained") or 0) == 0)
    postbank_games = sorted(set(games) - set(stall_games))

    # ---- per-candidate scoring -----------------------------------------------------------
    cands: list[dict] = []
    gate_jobs: list[tuple[str, str, str]] = []
    for tag, run in runs.items():
        for row in run.get("rows", []):
            if row.get("status") != "ok":
                cands.append({**row, "tag": tag, "score_status": "generation_failed"})
                continue
            game = row["game"]
            s = splits[game]
            text = (HERE / "bon" / tag / row["completion_file"]).read_text(errors="replace")
            code = e3._extract_python(text) or text.strip()
            m: dict = {
                "tag": tag,
                "game": game,
                "candidate": row["candidate"],
                "seed": row["seed"],
                "temperature": row["temperature"],
                "usable": row["usable"],
                "generate_would_accept": row["generate_would_accept"],
                "defect_kinds": row["defect_kinds"],
                "engine_changes_anything": row["engine_changes_anything"],
                "stop_type": row["stop_type"],
                "predicted_n": row["predicted_n"],
                "wall_s": row["wall_s"],
                "code_sha256_16": row["code_sha256_16"],
            }
            ns: dict = {"np": np, "numpy": np}
            try:
                exec(compile(code, row["completion_file"], "exec"), ns)  # noqa: S102
                engine = ns.get("engine")
                assert callable(engine)
            except Exception as exc:  # noqa: BLE001
                m["score_status"] = f"unrunnable:{type(exc).__name__}"
                cands.append(m)
                continue
            m["score_status"] = "ok"
            m["has_goal_predicate"] = callable(ns.get("is_level_complete"))

            for label, rows_ in (("heldout", s["_heldout"]), ("in_sample", s["_shown"])):
                if not rows_:
                    m[f"{label}_n"] = 0
                    continue
                vr = e3.WorldModelVerifier(list(rows_)).score(engine)
                n_changing = int(getattr(vr, "n_changing", 0) or 0)
                m[f"{label}_n"] = int(getattr(vr, "n", 0) or 0)
                m[f"{label}_accuracy"] = round(float(getattr(vr, "accuracy", 0.0) or 0.0), 6)
                m[f"{label}_n_changing"] = n_changing
                m[f"{label}_n_changes_correct"] = int(getattr(vr, "n_changes_correct", 0) or 0)
                m[f"{label}_cell_recall"] = (
                    round(float(getattr(vr, "cell_recall", 0.0) or 0.0), 4) if n_changing else None
                )
                m[f"{label}_invented_changed_cells"] = int(
                    getattr(vr, "invented_changed_cells", 0) or 0
                )
                m[f"{label}_n_noop"] = int(getattr(vr, "n_noop", 0) or 0)
                m[f"{label}_n_noop_hallucinated"] = int(getattr(vr, "n_noop_hallucinated", 0) or 0)

            # THE SHIPPED TRUST GATE, replicated on the transitions `induce()` actually received.
            # Reported next to the out-of-sample bar because they are NOT the same measurement:
            # this one splits the 17-row prefix internally, the other uses the proven unrendered
            # rows. Where they disagree, the shipped gate is the optimistic one.
            try:
                with open(HERE / "capture" / game / f"transitions{CALL_INDEX}.pkl", "rb") as fh:
                    prefix_trans = pickle.load(fh)
                sel = select_trusted_world_model(
                    list(prefix_trans),
                    [WorldModelCandidate("bon_candidate", engine, ns.get("is_level_complete"))],
                    hidden_state=True,
                )
                m["shipped_gate_heldout_accuracy"] = round(
                    float(sel.selected_score.heldout_accuracy), 6
                )
                m["shipped_gate_prefix_accuracy"] = round(
                    float(sel.selected_score.prefix_accuracy), 6
                )
                m["shipped_gate_trust_energy"] = round(float(sel.selected_score.trust_energy), 6)
                m["shipped_gate_passes"] = bool(
                    sel.selected_score.heldout_accuracy >= SHIPPED_HELDOUT_THRESHOLD
                )
            except Exception as exc:  # noqa: BLE001
                m["shipped_gate_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"

            root_path = HERE / "capture" / game / f"root_grid{CALL_INDEX}.pkl"
            if root_path.exists():
                cp = SCRATCH / f"{game}_k{row['candidate']}.py"
                cp.write_text(code)
                key = f"{game}|{row['candidate']}"
                gate_jobs.append((key, str(cp), str(root_path)))
                m["_gate_key"] = key
            cands.append(m)

    print(f"running {len(gate_jobs)} gate jobs, {GATE_WORKERS} at a time ...", flush=True)
    t_gate = time.monotonic()
    gate_results = _run_gates(gate_jobs)
    gate_wall = round(time.monotonic() - t_gate, 1)
    for m in cands:
        key = m.pop("_gate_key", None)
        if key and key in gate_results:
            for k, v in gate_results[key].items():
                m[f"gate_{k}" if not k.startswith(("goal", "plan")) else k] = v

    # ---- the three criteria --------------------------------------------------------------
    def c_i(m: dict) -> bool | None:
        """SHIPPED BAR, scored OUT-OF-SAMPLE: mechanically usable AND held-out accuracy == 1.0.

        None where nothing is held out -- not a pass, not a fail, NOTHING WAS MEASURED. That is
        the vc33 case and it must not be counted either way."""
        if m.get("score_status") != "ok":
            return False
        if not m.get("usable"):
            return False
        if m.get("heldout_n", 0) == 0:
            return None
        return bool(m.get("heldout_accuracy", 0.0) >= SHIPPED_HELDOUT_THRESHOLD)

    def c_i_strict(m: dict) -> bool | None:
        """ANTI-IDENTITY BAR. `heldout_accuracy == 1.0` is achievable by the identity function on
        a held-out set with no changing rows, so the strict read asks for at least one unseen
        CHANGING transition predicted exactly, falling back to zero hallucinated no-ops."""
        base = c_i(m)
        if base is not True:
            return base
        if m.get("heldout_n_changing", 0) > 0:
            return bool(m.get("heldout_n_changes_correct", 0) >= 1)
        return m.get("heldout_n_noop_hallucinated") == 0

    def c_ii(m: dict) -> bool | None:
        base = c_i(m)
        if base is not True:
            return base
        return bool(m.get("goal_satisfiable"))

    def c_iii(m: dict) -> bool | None:
        base = c_ii(m)
        if base is not True:
            return base
        return bool(m.get("plan_found"))

    CRITERIA = {
        "i_dynamics": c_i,
        "i_dynamics_strict": c_i_strict,
        "ii_goal_satisfiable": c_ii,
        "iii_plan_found": c_iii,
    }
    for m in cands:
        m["criteria"] = {name: fn(m) for name, fn in CRITERIA.items()}

    by_game: dict[str, list[dict]] = {}
    for m in cands:
        by_game.setdefault(m["game"], []).append(m)
    for rows_ in by_game.values():
        rows_.sort(key=lambda r: r.get("candidate", 0))

    def _rank_key(m: dict) -> tuple:
        """The VERIFIER's ranking, on criterion (i)'s own evidence only. Nothing from the goal
        gate or the planner enters here -- that is the whole point of the conditional read."""
        return (
            1 if m.get("score_status") == "ok" else 0,
            1 if m.get("usable") else 0,
            float(m.get("heldout_accuracy") or 0.0),
            int(m.get("heldout_n_changes_correct") or 0),
            -float(m.get("shipped_gate_trust_energy") or 0.0),
            -int(m.get("candidate", 0)),  # ties -> the EARLIER candidate, i.e. cheaper
        )

    # ---- yields --------------------------------------------------------------------------
    def yields_for(games_: list[str]) -> dict:
        out: dict = {}
        n_avail = min((len(by_game.get(g, [])) for g in games_), default=0)
        out["n_candidates_available_per_game"] = n_avail
        out["games"] = games_

        # PER-CANDIDATE MARGINAL RATE. The per-game yield below is what was asked for, but it has
        # n = len(games_) -- five, on the stall path. The marginal rate pools every candidate
        # (n = 5 x N), which is a far better-powered estimate of the underlying per-draw
        # probability, and it is what makes the independence cross-check below meaningful.
        marg: dict = {}
        for name in CRITERIA:
            vals = [
                m["criteria"][name]
                for g in games_
                for m in by_game.get(g, [])[:n_avail]
                if m["criteria"][name] is not None
            ]
            p = (sum(1 for v in vals if v) / len(vals)) if vals else None
            marg[name] = {
                "n_candidates_measured": len(vals),
                "n_pass": sum(1 for v in vals if v),
                "marginal_rate": round(p, 4) if p is not None else None,
                # If draws were INDEPENDENT, this is what any-of-N would be. Comparing it to the
                # OBSERVED any-of-N below separates "sampling cannot find it" (both near zero)
                # from "the pool is correlated / degenerate" (observed << implied).
                "implied_any_of_N_if_independent": {
                    f"N{N}": (round(1.0 - (1.0 - p) ** N, 4) if p is not None else None)
                    for N in NS
                    if max(n_avail, 1) >= N
                },
            }
        out["marginal_per_candidate"] = marg
        for N in NS:
            if n_avail < N:
                out[f"N{N}"] = {"status": "not_reached", "needed": N, "available": n_avail}
                continue
            block: dict = {}
            for name in CRITERIA:
                per_game = {}
                for g in games_:
                    pool = by_game[g][:N]
                    vals = [m["criteria"][name] for m in pool]
                    if any(v is True for v in vals):
                        per_game[g] = True
                    elif all(v is None for v in vals):
                        per_game[g] = None  # unmeasurable on this game
                    else:
                        per_game[g] = False
                measured = [g for g, v in per_game.items() if v is not None]
                block[name] = {
                    "per_game": per_game,
                    "n_measured_games": len(measured),
                    "n_pass": sum(1 for g in measured if per_game[g]),
                    "yield": (
                        round(sum(1 for g in measured if per_game[g]) / len(measured), 4)
                        if measured
                        else None
                    ),
                }
            # SELECT ON (i) ONLY, then ask what you got at (ii)/(iii). This is the direct test of
            # "would selecting on dynamics reliably produce more tn36s".
            sel_rows = {}
            for g in games_:
                pool = [m for m in by_game[g][:N] if m.get("score_status") == "ok"]
                sel_rows[g] = max(pool, key=_rank_key) if pool else None
            block["select_on_i_then_check"] = {
                "selected_candidate_index": {
                    g: (r.get("candidate") if r else None) for g, r in sel_rows.items()
                },
                "per_criterion": {
                    name: {
                        g: (r["criteria"][name] if r else None) for g, r in sel_rows.items()
                    }
                    for name in CRITERIA
                },
            }
            for name in CRITERIA:
                vals = [
                    r["criteria"][name] for r in sel_rows.values() if r and r["criteria"][name] is not None
                ]
                block["select_on_i_then_check"].setdefault("yield", {})[name] = (
                    round(sum(1 for v in vals if v) / len(vals), 4) if vals else None
                )
            # ACCEPT-FIRST over the SAME N samples: the shipped policy, more samples, no verifier.
            af_rows = {}
            for g in games_:
                pool = [m for m in by_game[g][:N] if m.get("score_status") == "ok"]
                af_rows[g] = next((m for m in pool if m.get("usable")), pool[0] if pool else None)
            block["accept_first_over_same_N"] = {
                "selected_candidate_index": {
                    g: (r.get("candidate") if r else None) for g, r in af_rows.items()
                },
                "yield": {},
            }
            for name in CRITERIA:
                vals = [
                    r["criteria"][name] for r in af_rows.values() if r and r["criteria"][name] is not None
                ]
                block["accept_first_over_same_N"]["yield"][name] = (
                    round(sum(1 for v in vals if v) / len(vals), 4) if vals else None
                )
            out[f"N{N}"] = block
        return out

    # ---- cost ----------------------------------------------------------------------------
    gen_wall = sum(float(m.get("wall_s") or 0.0) for m in cands)
    n_ok = sum(1 for m in cands if m.get("score_status") != "generation_failed")
    cost = {
        "generation_wall_s_total": round(gen_wall, 1),
        "generation_wall_s_mean_per_candidate": round(gen_wall / n_ok, 1) if n_ok else None,
        "n_candidates_generated": n_ok,
        "gate_and_plan_wall_s_total_parallel": gate_wall,
        "gate_workers": GATE_WORKERS,
        "note": (
            "generation_wall_s_total is SERIAL GPU time on one card -- every candidate went "
            "through one server process, so it is directly a GPU-hour figure. The gate/plan "
            "wall is CPU and was run with bounded parallelism; it is not GPU cost."
        ),
    }
    for N in NS:
        per_game_cost = {}
        for g in games:
            pool = by_game.get(g, [])[:N]
            per_game_cost[g] = round(sum(float(m.get("wall_s") or 0.0) for m in pool), 1)
        cost[f"N{N}_gpu_seconds_per_induction_mean"] = (
            round(sum(per_game_cost.values()) / len(per_game_cost), 1) if per_game_cost else None
        )

    # ---- diversity: is the candidate pool degenerate? --------------------------------------
    diversity = {}
    for g, rows_ in by_game.items():
        shas = [m.get("code_sha256_16") for m in rows_ if m.get("code_sha256_16")]
        behav = [
            (m.get("heldout_accuracy"), m.get("heldout_n_changes_correct"), m.get("usable"))
            for m in rows_
            if m.get("score_status") == "ok"
        ]
        diversity[g] = {
            "n_candidates": len(rows_),
            "n_distinct_code_sha": len(set(shas)),
            "n_distinct_heldout_behaviour": len(set(behav)),
        }

    payload = {
        "experiment": "outer_loop_arc_induce_bestofn_phase1",
        "schema": "carnot.arc_induce_bestofn.v1",
        "milestone": "2026.07.outer_loop",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "inference_substrate_note": (
            "This SCORING step loads no model: it replays cached LLM completions through the "
            "shipped verifier, goal gate and planner. The GENERATION step that produced those "
            "completions is a separate artifact-bearing process and was live_llm_inference on a "
            "proven CUDA build; its witness is carried under `generation_runs`."
        ),
        "random_seed": 20260731,
        "verifier_is_oracle": False,
        "verifier_is_oracle_note": (
            "The verifier here is the induced world model's held-out prediction accuracy plus the "
            "goal/plan reachability search -- it is NOT the executable oracle that defines a win. "
            "No moat claim is made; this measures whether verifier-based SELECTION among sampled "
            "candidates changes the induce path's yield."
        ),
        "call_index": CALL_INDEX,
        "call_index_note": (
            "call_index 1 is the bounded-reinduction stall site (arc_competition_agent.py:5770 -> "
            "execute_bounded_llm_reinduction), verified by stack capture. All three criteria's "
            "gates live on it and run by default."
        ),
        "shipped_heldout_threshold": SHIPPED_HELDOUT_THRESHOLD,
        "stall_games": stall_games,
        "postbank_games": postbank_games,
        "partition_evidence": {
            g: {
                "levels_gained": caps.get(g, {}).get("levels_gained"),
                "n_transitions_at_induction": next(
                    (
                        c["n_transitions"]
                        for c in caps.get(g, {}).get("captures", [])
                        if c["call_index"] == CALL_INDEX
                    ),
                    None,
                ),
                "heldout_n": splits[g]["n_heldout"],
                "heldout_n_changing": splits[g]["heldout_n_changing"],
            }
            for g in games
        },
        "generation_runs": {
            tag: {
                "status": r.get("status"),
                "witness": r.get("witness"),
                "sampler": r.get("sampler"),
                "temperature": r.get("temperature"),
                "budget": r.get("budget"),
                "seed_base": r.get("seed_base"),
                "n_candidates_requested": r.get("n_candidates_requested"),
                "wall_s": r.get("wall_s"),
            }
            for tag, r in runs.items()
        },
        "splits": {g: {k: v for k, v in splits[g].items() if not k.startswith("_")} for g in games},
        "cost": cost,
        "diversity": diversity,
        "yields_stall_path": yields_for(stall_games),
        "yields_postbank_path": yields_for(postbank_games),
        "candidates": cands,
    }
    payload["reproducibility_checksum"] = hashlib.sha256(
        json.dumps(
            [
                (m.get("game"), m.get("candidate"), m.get("code_sha256_16"))
                for m in sorted(cands, key=lambda r: (r.get("game", ""), r.get("candidate", 0)))
            ],
            sort_keys=True,
        ).encode()
    ).hexdigest()[:32]

    with open(OUT, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
    print(f"wrote {OUT}")

    ys = payload["yields_stall_path"]
    print(f"\nSTALL PATH games={ys['games']} n_avail={ys['n_candidates_available_per_game']}")
    for N in NS:
        b = ys.get(f"N{N}")
        if not b or b.get("status") == "not_reached":
            print(f"  N={N}: {b}")
            continue
        print(f"  N={N}:")
        for name in CRITERIA:
            r = b[name]
            print(
                f"    any-of-N  {name:<22} {r['n_pass']}/{r['n_measured_games']}  "
                f"yield={r['yield']}"
            )
        print(f"    select-on-(i)  {json.dumps(b['select_on_i_then_check']['yield'])}")
        print(f"    accept-first   {json.dumps(b['accept_first_over_same_N']['yield'])}")
    print("\n  MARGINAL per-candidate rate (n = games x N):")
    for name, r in ys.get("marginal_per_candidate", {}).items():
        print(
            f"    {name:<22} {r['n_pass']}/{r['n_candidates_measured']} = {r['marginal_rate']}  "
            f"implied any-of-N if independent: {json.dumps(r['implied_any_of_N_if_independent'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

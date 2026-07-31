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

THREE BARS ARE REPORTED FOR (i), because the first two are known to admit junk. The SHIPPED bar
is `heldout_accuracy >= verifier_threshold` with `verifier_threshold = min_heldout_accuracy = 1.0`
at both live call sites -- so, accuracy exactly 1.0. On a held-out set with no CHANGING rows
(lp85 here) the identity function scores 1.0 on that bar, which is why the strict bar is carried
alongside: at least one held-out changing transition predicted EXACTLY, falling back to zero
hallucinated no-ops where nothing changing is held out. Both are in the payload; where they
disagree, the disagreement is the result.

THE STRICT BAR IS NOT ENOUGH EITHER, and saying otherwise overpromises. `i_dynamics_strict`
falls back to "no hallucinated no-ops" exactly where `heldout_n_changing == 0`, which excludes
only a WHOLLY INERT engine -- so on lp85, whose held-out split is 18 rows of which 0 change,
three candidates clear BOTH bars by predicting "nothing changes" against a split where nothing
does. That is zero discriminative evidence about dynamics, and it is 3 of the 9 (i)-passers on
the stall path. `i_dynamics_change_gradable` is therefore carried as the third bar: identical to
the shipped bar, but None (NOT MEASURED, excluded from both numerator and denominator) on any
game whose held-out split contains no changing transition. It is the only one of the three whose
denominator consists entirely of evidence that could have falsified it. Where the headline (i)
number and the change-gradable number disagree, the gap IS the vacuity.

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


def _wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    """95% Wilson score interval for a binomial proportion.

    WHY THIS IS NOT DECORATION HERE. The headline yield delta -- unconditional (iii) going 0.0 to
    0.4 -- is 2 successes over 5 games. Wilson on 2/5 spans roughly 0.12 to 0.77, so ONE flip in
    either direction moves the point estimate to 0.2 or 0.6 and the interval covers everything
    from "barely better than the baseline" to "most games". Reporting 0.4 bare invites it to be
    read as a rate. Wilson rather than normal-approximation because n is tiny and the proportion
    sits near the boundary, where the normal interval runs outside [0, 1] and is simply wrong."""
    if n <= 0:
        return None
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z / d) * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def _tally(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        k = str(v) if v is not None else "not_evaluated"
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def _run_gates(jobs: list[tuple[str, str]]) -> dict[str, dict]:
    """Run gate_worker over `jobs` with bounded parallelism. Each job is (key, job_json_path)."""
    results: dict[str, dict] = {}
    pending = list(jobs)
    running: list[tuple[str, subprocess.Popen, float]] = []
    while pending or running:
        while pending and len(running) < GATE_WORKERS:
            key, job_path = pending.pop(0)
            proc = subprocess.Popen(
                [PY, str(HERE / "gate_worker.py"), job_path],
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
    from carnot.agentic import arc_executable_world_model as e3
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
    # THIS PROCESS NEVER EXECUTES GENERATED CODE. Every metric that requires running the induced
    # engine -- the held-out verifier, the shipped trust gate, the goal gate, the planner -- is
    # computed by gate_worker.py in its own killable subprocess. See that module's docstring: a
    # non-terminating induced engine wedged the generation loop for 13 minutes on this very run,
    # and `WorldModelVerifier.score` would have executed the same engine here.
    cands: list[dict] = []
    gate_jobs: list[tuple[str, str]] = []
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
                "validation_timed_out": row.get("validation_timed_out"),
                "engine_changes_anything": row["engine_changes_anything"],
                "stop_type": row["stop_type"],
                "predicted_n": row["predicted_n"],
                "wall_s": row["wall_s"],
                "code_sha256_16": row["code_sha256_16"],
            }
            cp = SCRATCH / f"{game}_k{row['candidate']}.py"
            cp.write_text(code)
            hp = SCRATCH / f"{game}_heldout.pkl"
            ip = SCRATCH / f"{game}_shown.pkl"
            if not hp.exists():
                with open(hp, "wb") as fh:
                    pickle.dump(s["_heldout"], fh)
                with open(ip, "wb") as fh:
                    pickle.dump(s["_shown"], fh)
            job = {
                "code_path": str(cp),
                "heldout_pkl": str(hp),
                "in_sample_pkl": str(ip),
                "prefix_pkl": str(HERE / "capture" / game / f"transitions{CALL_INDEX}.pkl"),
                "root_pkl": str(HERE / "capture" / game / f"root_grid{CALL_INDEX}.pkl"),
            }
            jp = SCRATCH / f"{game}_k{row['candidate']}.job.json"
            jp.write_text(json.dumps(job))
            key = f"{game}|{row['candidate']}"
            gate_jobs.append((key, str(jp)))
            m["_gate_key"] = key
            cands.append(m)

    # GATE CACHE, keyed by the candidate's code sha. Gate results are a deterministic function of
    # (code, root grid, transitions), all of which are frozen once generation has finished, so a
    # re-analysis that adds a new CRITERION must not have to re-execute 48 searches -- and must
    # not be tempted to skip them either. Delete the cache file to force a clean re-run.
    cache_path = HERE.parent / "gate_cache.json"
    cache = {}
    if cache_path.exists() and os.environ.get("BON_GATE_CACHE", "1") == "1":
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:  # noqa: BLE001
            cache = {}
    sha_of = {f"{m['game']}|{m['candidate']}": m.get("code_sha256_16") for m in cands}
    todo = [(k, jp) for k, jp in gate_jobs if cache.get(sha_of.get(k) or "") is None]
    print(
        f"running {len(todo)} gate jobs ({len(gate_jobs) - len(todo)} cached), "
        f"{GATE_WORKERS} at a time ...",
        flush=True,
    )
    t_gate = time.monotonic()
    fresh = _run_gates(todo)
    gate_wall = round(time.monotonic() - t_gate, 1)
    for k, v in fresh.items():
        sha = sha_of.get(k)
        if sha:
            cache[sha] = v
    cache_path.write_text(json.dumps(cache, indent=1, sort_keys=True))
    gate_results = {k: cache[sha_of[k]] for k, _ in gate_jobs if sha_of.get(k) in cache}
    for m in cands:
        key = m.pop("_gate_key", None)
        if key and key in gate_results:
            res = dict(gate_results[key])
            m["score_status"] = res.pop("status", "ok")
            m.update(res)
        elif "score_status" not in m:
            m["score_status"] = "no_gate_result"

    # A MEASUREMENT FAILURE IS NOT A CRITERION FAILURE. `unrunnable:*` means the model emitted
    # code that does not parse -- no engine exists, and False is the honest verdict on every
    # criterion. `gate_timeout` is different in kind: ft09 candidate 5 DID produce an engine, and
    # the harness never learned anything about it because the engine did not terminate inside the
    # validation bound. Folding that into the denominator as a False records a missing observation
    # as a zero, which is the exact error this project's artifact discipline names. It is returned
    # as None (UNDETERMINED) instead, so it leaves both numerator and denominator.
    def _undetermined(m: dict) -> bool:
        return str(m.get("score_status", "")) == "gate_timeout"

    # ---- the criteria ----------------------------------------------------------------------
    def c_i(m: dict) -> bool | None:
        """SHIPPED BAR, scored OUT-OF-SAMPLE: mechanically usable AND held-out accuracy == 1.0.

        None where nothing is held out -- not a pass, not a fail, NOTHING WAS MEASURED. That is
        the vc33 case and it must not be counted either way."""
        if _undetermined(m):
            return None
        if m.get("score_status") != "ok":
            return False
        if not m.get("usable"):
            return False
        if m.get("heldout_n", 0) == 0:
            return None
        return bool(m.get("heldout_accuracy", 0.0) >= SHIPPED_HELDOUT_THRESHOLD)

    def c_i_change_gradable(m: dict) -> bool | None:
        """THE ONLY (i) BAR WHOSE DENOMINATOR COULD HAVE FALSIFIED IT.

        Identical to the shipped bar, but None on any game whose held-out split has no CHANGING
        transition. On such a split (lp85: 18 held-out rows, 0 changing) accuracy 1.0 is earned by
        predicting "nothing changes" -- true, worthless, and indistinguishable from a real
        dynamics model. `i_dynamics_strict` does NOT catch this: its fallback where nothing
        changing is held out asks only for zero hallucinated no-ops, which excludes a wholly inert
        engine and nothing else. This bar excludes the game instead of scoring it.

        GRADABILITY IS A PROPERTY OF THE GAME'S SPLIT, NOT OF THE CANDIDATE ROW, and reading it
        off the row is a real bug rather than a style point. An unrunnable candidate carries
        `heldout_n_changing: None` because it was never scored, so a row-level test lets lp85's
        two syntax-error candidates back in as genuine Falses -- which flipped lp85's per-game
        value from None to False at N=8 only, and moved this criterion's yield from a flat 0.25 to
        0.2 for a reason that has nothing to do with change prediction. The split is consulted
        directly so an ungradable game is ungradable at every N."""
        if not splits[m["game"]]["heldout_can_grade_change"]:
            return None
        return c_i(m)

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

    # THE UNCONDITIONAL READS, and why reporting only the conjunctions would have been a false
    # null. The operator posed (ii) and (iii) as CONJUNCTIONS onto (i), and they are computed that
    # way above. But (i) here is the SHIPPED accuracy bar applied OUT-OF-SAMPLE, which is strictly
    # harsher than what the live pipeline computes -- and on this grid the two components turn out
    # to be ANTI-CORRELATED. ft09 candidate 1 and tn36 candidate 1 both reach a satisfiable goal
    # AND a found plan while failing the out-of-sample (i) bar, so the conjunction records 0 for a
    # reason that has nothing to do with the goal or the planner. Reporting only the conjunction
    # would have said "no candidate is plannable" when two are.
    def c_ii_uncond(m: dict) -> bool | None:
        if _undetermined(m):
            return None
        if m.get("score_status") != "ok":
            return False
        return bool(m.get("goal_satisfiable"))

    def c_iii_uncond(m: dict) -> bool | None:
        if _undetermined(m):
            return None
        if m.get("score_status") != "ok":
            return False
        return bool(m.get("goal_satisfiable")) and bool(m.get("plan_found"))

    # THE LIVE PIPELINE'S OWN CONJUNCTION. `select_trusted_world_model` splits the 17-row prefix
    # internally; that -- not the proven unrendered split -- is the gate the agent actually
    # applies before it consults the goal gate. This is therefore what the shipped path would
    # have done with each candidate, and it is the fair test of "could best-of-N have helped the
    # pipeline as it exists" (as opposed to "as it should be graded").
    def c_ii_shipped(m: dict) -> bool | None:
        if _undetermined(m):
            return None
        if m.get("score_status") != "ok":
            return False
        return bool(m.get("shipped_gate_passes")) and bool(m.get("goal_satisfiable"))

    def c_iii_shipped(m: dict) -> bool | None:
        if _undetermined(m):
            return None
        if m.get("score_status") != "ok":
            return False
        return c_ii_shipped(m) and bool(m.get("plan_found"))

    CRITERIA = {
        "i_dynamics": c_i,
        "i_dynamics_strict": c_i_strict,
        "i_dynamics_change_gradable": c_i_change_gradable,
        "ii_goal_satisfiable": c_ii,
        "iii_plan_found": c_iii,
        "ii_goal_satisfiable_unconditional": c_ii_uncond,
        "iii_plan_found_unconditional": c_iii_uncond,
        "ii_shipped_gate_and_goal": c_ii_shipped,
        "iii_shipped_gate_and_plan": c_iii_shipped,
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
                "marginal_rate_wilson95": _wilson(sum(1 for v in vals if v), len(vals)),
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
                n_pass = sum(1 for g in measured if per_game[g])
                block[name] = {
                    "per_game": per_game,
                    "n_measured_games": len(measured),
                    "n_pass": n_pass,
                    "yield": (round(n_pass / len(measured), 4) if measured else None),
                    # The denominator here is GAMES (five on the stall path), not candidates. A
                    # yield reported without this interval reads as a rate; it is 2 of 5.
                    "yield_wilson95": _wilson(n_pass, len(measured)),
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

    # ---- WHY criterion (ii) failed, which is a different question from THAT it failed -------
    # `degenerate_goal_predicate` means the reachable set was searched EXHAUSTIVELY and the goal
    # was never true -- evidence against the predicate. `goal_unreached_within_budget` and
    # `goal_unreached_within_depth` mean the search STOPPED, and whether the goal is reachable is
    # UNKNOWN. Both make criterion (ii) False, but they license opposite conclusions: the first
    # says sampling a better goal could help, the second says the gate could not decide and no
    # amount of sampling is being fairly tested. Flattening them is the exact mislabel the repo
    # fixed on 2026-07-31 in `_goal_satisfiability_check`; it must not be reintroduced here.
    gate_census: dict = {}
    for scope, gs in (("stall", stall_games), ("postbank", postbank_games)):
        rows_ = [m for g in gs for m in by_game.get(g, [])]
        passed_i = [m for m in rows_ if m["criteria"]["i_dynamics"] is True]
        gate_census[scope] = {
            "n_candidates": len(rows_),
            "n_passing_criterion_i": len(passed_i),
            "goal_kind_all_candidates": _tally(m.get("goal_kind") for m in rows_),
            "goal_kind_among_criterion_i_passers": _tally(m.get("goal_kind") for m in passed_i),
            "n_criterion_i_passers_whose_gate_was_UNDECIDED": sum(
                1
                for m in passed_i
                if str(m.get("goal_kind", "")).startswith(
                    ("goal_unreached_within_budget", "goal_unreached_within_depth")
                )
            ),
            "n_criterion_i_passers_whose_goal_was_DISPROVED": sum(
                1 for m in passed_i if m.get("goal_kind") == "degenerate_goal_predicate"
            ),
        }

    # ---- IS DYNAMICS QUALITY ANTI-CORRELATED WITH PLANNABILITY? ----------------------------
    # The operator's design tension, measured rather than argued: "selecting on DYNAMICS accuracy
    # alone would reliably produce more tn36s -- perfect models that die at the goal gate". This
    # block compares held-out accuracy between the candidates whose goal the gate certifies and
    # the candidates whose goal it does not. If the certified ones are WORSE, then criterion (i)
    # is not merely insufficient as a selector, it is actively pointed the wrong way.
    def _acc(m):
        v = m.get("heldout_accuracy")
        return float(v) if isinstance(v, (int, float)) else None

    # ...AND THE ANSWER DEPENDS ON WHICH DYNAMICS BAR YOU ASK WITH, which is why both 2x2s are
    # built below rather than only the one that supports the headline. `i_dynamics` is the
    # OUT-OF-SAMPLE bar this harness computes against the proven unrendered split.
    # `shipped_gate_passes` is what `select_trusted_world_model` ACTUALLY applies in the live
    # pipeline -- it splits the 17-row prefix internally, so it is a different, easier measurement
    # on the same candidate. The two disagree numerically on most candidates (ft09 k1: 0.8 here,
    # 1.0 there), and the disagreement is not cosmetic: it flips the SIGN of this comparison. Any
    # claim about dynamics-vs-plannability must therefore name its measure.
    def _two_by_two(rows_: list[dict], selector) -> dict:
        passers = [m for m in rows_ if selector(m) is True]
        others = [m for m in rows_ if selector(m) is not True]

        def _plannable(ms):
            return sum(1 for m in ms if m["criteria"]["iii_plan_found_unconditional"] is True)

        kp, np_ = _plannable(passers), _plannable(others)
        return {
            "n_selected": len(passers),
            "n_selected_that_are_plannable": kp,
            "n_not_selected": len(others),
            "n_not_selected_that_are_plannable": np_,
            "plannable_rate_when_selected": round(kp / len(passers), 4) if passers else None,
            "plannable_rate_when_not_selected": round(np_ / len(others), 4) if others else None,
            "plannable_rate_when_selected_wilson95": _wilson(kp, len(passers)),
            "plannable_rate_when_not_selected_wilson95": _wilson(np_, len(others)),
            "direction": (
                None
                if not passers or not others
                else (
                    "anti_selective"
                    if (kp / len(passers)) < (np_ / len(others))
                    else "positively_selective"
                    if (kp / len(passers)) > (np_ / len(others))
                    else "no_difference"
                )
            ),
        }

    anti: dict = {}
    for scope, gs in (("stall", stall_games), ("postbank", postbank_games)):
        rows_ = [
            m
            for g in gs
            for m in by_game.get(g, [])
            if m.get("score_status") == "ok" and _acc(m) is not None
        ]
        sat = [m for m in rows_ if m.get("goal_satisfiable")]
        uns = [m for m in rows_ if not m.get("goal_satisfiable")]
        scored_rows = [m for g in gs for m in by_game.get(g, []) if m.get("score_status") == "ok"]
        n_disagree = sum(
            1
            for m in scored_rows
            if m.get("heldout_accuracy") is None
            or m.get("shipped_gate_heldout_accuracy") is None
            or abs(float(m["heldout_accuracy"]) - float(m["shipped_gate_heldout_accuracy"])) > 1e-9
        )
        anti[scope] = {
            "MEASURE_DEPENDENCE": {
                "by_out_of_sample_criterion_i": _two_by_two(
                    scored_rows, lambda m: m["criteria"]["i_dynamics"]
                ),
                "by_shipped_trust_gate": _two_by_two(
                    scored_rows, lambda m: m.get("shipped_gate_passes")
                ),
                "n_scored_candidates": len(scored_rows),
                "n_candidates_where_the_two_accuracies_differ": n_disagree,
                "what_this_means": (
                    "The sign of the dynamics-vs-plannability relationship is NOT a property of "
                    "the candidates -- it is a property of which dynamics bar you select with. "
                    "Under the harsher out-of-sample bar the selected candidates are LESS "
                    "plannable than the rejected ones; under the shipped trust gate, the gate the "
                    "live pipeline actually runs, they are MORE plannable, and ft09 k1 sits in "
                    "BOTH the selected and the plannable set, so the 'empty intersection' is "
                    "specific to the harsher bar. Both directions rest on the same 2 plannable "
                    "candidates, so neither direction is established; what IS established is that "
                    "the claim cannot be stated without naming its measure."
                ),
            },
            "n_candidates_satisfying_both_i_and_goal": sum(
                1 for m in rows_ if m["criteria"]["i_dynamics"] is True and m.get("goal_satisfiable")
            ),
            "n_goal_satisfiable": len(sat),
            "n_goal_not_satisfiable": len(uns),
            "mean_heldout_accuracy_when_goal_SATISFIABLE": (
                round(sum(_acc(m) for m in sat) / len(sat), 4) if sat else None
            ),
            "mean_heldout_accuracy_when_goal_NOT_satisfiable": (
                round(sum(_acc(m) for m in uns) / len(uns), 4) if uns else None
            ),
            "mean_comparison_is_underpowered": (
                "The two pooled means are n=2 against n=29 and pool across games of very "
                "different difficulty, so they are reported but carry no weight. The "
                "load-bearing evidence for the inversion is the EMPTY INTERSECTION and the "
                "per-game tn36 contrast below, neither of which depends on pooling."
            ),
            "the_yield_deltas_are_underpowered_too": (
                "This caveat was originally attached ONLY to the pooled means, which understated "
                "it: the HEADLINE itself -- unconditional (iii) going 0.0 at N=1 to 0.4 at N=4 -- "
                "is 2 successes across 5 games, both at candidate index k1. Its Wilson 95% "
                "interval spans about 0.12 to 0.77 (carried as `yield_wilson95` on every yield "
                "block), so one flip in either direction gives 0.2 or 0.6. The direction of the "
                "gain is attributable to selection rather than to extra sampling -- the "
                "`accept_first_over_same_N` control draws the same N candidates without the "
                "verifier and yields 0.0 at every N -- but the MAGNITUDE is unbounded by this "
                "grid. Read '0.4' as 'it happened on 2 of 5 games', never as a rate."
            ),
            "goal_satisfiable_candidates": [
                {
                    "game": m["game"],
                    "candidate": m["candidate"],
                    "heldout_accuracy": _acc(m),
                    "heldout_n_changes_correct": m.get("heldout_n_changes_correct"),
                    "heldout_n_changing": m.get("heldout_n_changing"),
                    "shipped_gate_passes": m.get("shipped_gate_passes"),
                    "plan_found": m.get("plan_found"),
                    "plan_length": m.get("plan_length"),
                }
                for m in sat
            ],
            # The per-game version is the load-bearing one: a pooled comparison across games
            # confounds "this game is easy" with "this candidate is good".
            "per_game_best_dynamics_vs_satisfiable": {
                g: {
                    "best_heldout_accuracy": max(
                        (_acc(m) for m in by_game.get(g, []) if _acc(m) is not None), default=None
                    ),
                    "heldout_accuracy_of_goal_satisfiable_candidates": [
                        _acc(m) for m in by_game.get(g, []) if m.get("goal_satisfiable")
                    ],
                }
                for g in gs
            },
        }

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
        # WHAT IS IN THE DENOMINATOR, AND WHAT IS NOT. Three distinct states, deliberately not
        # flattened: a candidate that produced working code and was scored; one whose code does
        # not parse (a real, honest zero -- the model failed at the task); and one that produced
        # code the harness could not finish measuring. Only the first two belong in a yield.
        "candidate_disposition": {
            scope: {
                "n_candidates": len([m for g in gs for m in by_game.get(g, [])]),
                "score_status": _tally(m.get("score_status") for g in gs for m in by_game.get(g, [])),
                "n_scored": len(
                    [m for g in gs for m in by_game.get(g, []) if m.get("score_status") == "ok"]
                ),
                "n_genuine_zero_no_engine_produced": len(
                    [
                        m
                        for g in gs
                        for m in by_game.get(g, [])
                        if str(m.get("score_status", "")).startswith("unrunnable")
                    ]
                ),
                "n_undetermined_excluded_from_all_denominators": len(
                    [m for g in gs for m in by_game.get(g, []) if _undetermined(m)]
                ),
                "why_undetermined_is_not_a_zero": (
                    "`gate_timeout` (ft09 candidate 5) produced an engine that did not terminate "
                    "inside the validation bound, so nothing about it was measured -- not its "
                    "dynamics, not its goal, not its plannability. Scoring it False would record "
                    "a MISSING OBSERVATION as a criterion failure and silently deflate every "
                    "denominator by one. It is None everywhere instead. `unrunnable:*` is a "
                    "different case and stays a genuine False: no engine exists to measure."
                ),
            }
            for scope, gs in (("stall", stall_games), ("postbank", postbank_games))
        },
        "change_gradability_of_each_split": {
            g: {
                "heldout_n": splits[g]["n_heldout"],
                "heldout_n_changing": splits[g]["heldout_n_changing"],
                "can_grade_change": splits[g]["heldout_can_grade_change"],
                "consequence": (
                    "Accuracy 1.0 on this split is earned by predicting 'nothing changes' and is "
                    "NOT evidence of a dynamics model; the game is excluded from "
                    "`i_dynamics_change_gradable`."
                    if not splits[g]["heldout_can_grade_change"]
                    else "Held-out changing rows exist, so criterion (i) is falsifiable here."
                ),
            }
            for g in games
        },
        "goal_gate_failure_census": gate_census,
        "dynamics_vs_plannability": anti,
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
    gc = payload["goal_gate_failure_census"]["stall"]
    print(
        f"\n  GOAL-GATE census (stall): {gc['n_passing_criterion_i']} of {gc['n_candidates']} "
        f"candidates pass (i); of those, UNDECIDED={gc['n_criterion_i_passers_whose_gate_was_UNDECIDED']} "
        f"DISPROVED={gc['n_criterion_i_passers_whose_goal_was_DISPROVED']}"
    )
    print(f"    goal_kind among (i)-passers: {json.dumps(gc['goal_kind_among_criterion_i_passers'])}")
    print("\n  MARGINAL per-candidate rate (n = games x N):")
    for name, r in ys.get("marginal_per_candidate", {}).items():
        print(
            f"    {name:<22} {r['n_pass']}/{r['n_candidates_measured']} = {r['marginal_rate']}  "
            f"implied any-of-N if independent: {json.dumps(r['implied_any_of_N_if_independent'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

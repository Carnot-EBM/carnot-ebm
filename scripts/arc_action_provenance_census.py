#!/usr/bin/env python3
"""WHERE DOES THE LIVE AGENT'S ACTION BUDGET ACTUALLY GO? -- a multi-game, replicated
per-action accounting of the SCORED ARC policy, on the OFFLINE arcade.

**Researcher summary (why this exists).**
    The project can say how many levels the live agent banks and how good its induced
    world models look. It could not say WHERE ITS ACTIONS COME FROM. Three independent
    2026-07/2026-08 lines had already concluded -- each by INFERENCE, none by direct
    measurement -- that the induce -> verify -> plan pipeline is not on the causal path to
    banking a level:

      * deleting the LLM induction tier left the action sequence BYTE-IDENTICAL on 5 of 6
        games;
      * 0 of 22 `stall` inductions ever cleared the goal gate, while 4 of 6
        `level_up_reinduction` events did -- so `plan_found` correlates with banking
        because banking TRIGGERS a trivially-passing re-induction, not the other way round;
      * tn36 holds an induced engine with held-out accuracy 1.0 and 25/25 changing
        transitions correct, and banked 0 levels in 346 actions.

    A single-game pilot of the per-action instrument
    (`scripts/arc_action_provenance_probe.py`) then produced the first direct number on
    tn36. This script generalizes that pilot into the measurement the question actually
    needs: SEVERAL games, REPLICATED, including at least one game the agent DOES bank a
    level on, so the accounting has a successful case to say what success looks like.

**What it runs, and what it never runs.**
    The SCORED policy `E3AgentPolicy` -- reached exactly as `make_carnot_agent` reaches it,
    via `arc_actions_to_progress.run_bounded_progress` -- against the OFFLINE arcade
    (`OperationMode.OFFLINE` over the local `environment_files/` tree). No scorecard is
    opened, no gateway is contacted, nothing is submitted, no online/scored game is played.
    Submission is operator-only and is not something this script can do even by accident:
    it never constructs an API client.

**The design, and the traps it is built around.**

    REPLICATES, NOT ONE EPISODE. A branch share read off a single episode is an anecdote.
    The live generator samples, so the policy is NOT deterministic at a fixed seed -- the
    pilot proved that directly (arms A and A' at the same seed diverged at action 50). So
    every game is run R times at the SAME seed, and the R runs ARE the A/A family: their
    spread is the measured noise floor for every per-game number reported, and no
    cross-game statement is made that is smaller than that floor.

    MISSING IS NOT ZERO. A cell that crashes, times out, or hangs in induction is recorded
    as a MISSING OBSERVATION and excluded from every aggregate, never folded in as a zero.
    A zero means "the agent spent its budget and none of it was plan-derived"; a crash
    means "we did not see what the agent would have done". Conflating the two is how a
    harness manufactures the null it was looking for.

    NOTHING IS TUNED. This is an autopsy. No knob is adjusted to make a game succeed, and
    the games are chosen from the PRIOR record (the 2026-07-28 banked-levels study), not
    from a peek at this run's output.

    THE TRACKED ENGINE STORE IS NOT WRITTEN. `LocalGGUFProposer.induce` writes
    `<E3_DIR>/<game>/world_model.py`, and the default `E3_DIR` is `results/arc_e3` --
    TRACKED, READ-ONLY EVIDENCE. Every cell therefore gets its OWN scratch store, set in
    the child's environment before its interpreter starts (E3_DIR is resolved at import
    time). A per-CELL store, not one shared one, because a shared store would let cell k's
    induced engine be loaded by cell k+1 and silently turn replicates into a sequence.

Usage:
    .venv/bin/python scripts/arc_action_provenance_census.py \
        --games vc33,tn36,ft09,tr87,lp85 --replicates 3 --budget 400

Spec: openspec/capabilities/arc-world-model-trust-energy/spec.md REQ-ARC-WMTE-6070
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from typing import Any, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER = os.path.join(REPO_ROOT, "scripts", "arc_action_provenance_worker.py")

sys.path.insert(0, os.path.join(REPO_ROOT, "python"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

# Reuse the pilot's GPU-citizenship helpers rather than re-typing them. They encode two
# lessons that cost real wall time to learn: reap ONLY the server this process started
# (matched on the port WE assigned, never on the process name -- the machine is shared),
# and WAIT for a card rather than evicting whoever holds it.
from arc_action_provenance_probe import _reap_my_generator, _wait_for_card  # noqa: E402

from carnot.agentic.arc_action_provenance import (  # noqa: E402
    EXPLORER_BRANCHES as _EXPLORER_BRANCHES,
)
from carnot.agentic.arc_action_provenance import (  # noqa: E402
    TOP_BRANCHES as _TOP_BRANCHES,
)

# The two top-level branches at which the induce->verify->plan pipeline is the thing that
# CHOSE the action. Imported concept, spelled out here because the headline number is
# defined by exactly this set and a reader must be able to see it without chasing a helper.
#
# `induce.plan_needs_reset` is deliberately NOT in this set. It is a RESET emitted so that
# a plan can be REPLAYED from the root -- an action spent BECAUSE of a plan, without being
# an action the plan chose. It is counted and reported separately; folding it in would
# flatter the pipeline by one action per plan.
PLAN_DERIVED_TOP = ("execute.plan_step", "induce.plan_from_current")

# PRE-REGISTERED TRUNCATION RULE (written 2026-08-01 BEFORE the first cell landed; the
# cells directory was empty). An episode that hits the WALL-CLOCK cap having spent less
# than this fraction of its ACTION budget is a MISSING OBSERVATION, not a data point: its
# branch shares describe a run that was cut off, and reporting them as the agent's
# behaviour is how a value never observed gets averaged in as if it were seen. This is
# exactly the recoding the 2026-07-28 banked-levels study had to apply after the fact, when
# a cell cut off at 26 of 400 actions was first written down as "a 0-0 tie". Fixing the
# threshold in advance is what stops it being chosen to suit the answer.
#
# NOT covered by this rule, deliberately: an episode that ENDS EARLY WITHOUT TIMING OUT
# (the explorer explored out, or the run finished). That is the agent's own behaviour and
# is a real observation, however short.
WALL_TRUNCATION_MIN_BUDGET_FRACTION = 0.5

# `induction_skipped` values that mean NO ENGINE WAS EVER PRODUCED, as opposed to "an
# engine was produced and then rejected". Collapsing those two into one boolean is exactly
# how a pipeline gets credited with work it did not do, so they are separated here and the
# separation is part of the record.
SKIP_NO_ENGINE = (
    "no_active_transitions",
    "disabled_by_env",
    "proposer_failed_or_missing_root",
    # REQ-ARC-WMTE-6610 (2026-08-21) split the conflated label above; rows recorded after
    # that date carry these three instead. A pattern list narrower than its concept is the
    # bug class the QA-layer discipline names -- widened additively.
    "proposer_failed",
    "missing_plan_start_grid",
    "proposer_failed_and_missing_plan_start_grid",
    "no_root",
    "no_transitions",
)


def _f(x: Any) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def run_cell(
    *,
    game: str,
    rep: int,
    seed: int,
    budget: int,
    max_inductions: int,
    explore_budget,
    wall_s: float,
    timeout: float,
    cuda_gpu: str,
    port: int,
    cells_dir: str,
    scratch_root: str,
) -> dict:
    """Run ONE armed episode in its own killable subprocess. Returns the worker's JSON.

    Its own process because inducing a world model EXECUTES LLM-authored engine code, which
    this repo never runs in the analysing interpreter, and because a cell that hangs must
    cost that cell and nothing else.
    """
    out_path = os.path.join(cells_dir, f"{game}_r{rep}.json")
    env = dict(os.environ)
    env["CARNOT_ARC_ACTION_PROVENANCE"] = "1"
    env.pop("CARNOT_ARC_ACTION_PROVENANCE_DIR", None)
    # PER-CELL scratch engine store. See the module docstring: this is what keeps the run
    # off `results/arc_e3/**` (tracked evidence) AND keeps replicates independent.
    env["CARNOT_ARC_E3_DIR"] = os.path.join(scratch_root, f"e3_{game}_r{rep}")
    env.pop("CARNOT_ARC_DISABLE_INDUCTION", None)
    # OUTER LOOP OWNS GPU 1 (2026-06-27 operator allocation). Never GPU 0 -- that is the
    # conductor's, and this session must not evict a process it did not start.
    #
    # `CARNOT_ARC_GENERATOR_CUDA_GPU` ONLY. Do NOT also set CUDA_VISIBLE_DEVICES: the
    # proposer builds its own launch env and pins the card itself, and pre-setting
    # CUDA_VISIBLE_DEVICES=1 renumbers the visible cards so physical card 1 is no longer AT
    # index 1, the headroom probe finds nothing, and the generator silently falls back to
    # the AMD iGPU HIP build. The worker refuses that fallback outright; this comment
    # records why the obvious-looking extra pin must not be re-added.
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["CARNOT_ARC_GENERATOR_CUDA_GPU"] = cuda_gpu
    cmd = [
        sys.executable,
        WORKER,
        "--game",
        game,
        "--seed",
        str(seed),
        "--budget",
        str(budget),
        "--max-inductions",
        str(max_inductions),
        "--explore-budget",
        str(explore_budget),
        "--wall-s",
        str(wall_s),
        "--generator",
        "live",
        # A DISTINCT port per cell. Same-port reuse across sequential cells is how a stale
        # server from a previous cell gets silently served instead of a fresh one.
        "--cuda-port",
        str(port),
        "--out",
        out_path,
        "--arm-label",
        f"{game}_r{rep}",
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, env=env, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return {
            "arm_label": f"{game}_r{rep}",
            "game": game,
            "replicate": rep,
            "missing_observation": True,
            "error": "timeout",
            "wall_s_measured": round(time.time() - t0, 3),
        }
    if proc.returncode != 0 or not os.path.exists(out_path):
        return {
            "arm_label": f"{game}_r{rep}",
            "game": game,
            "replicate": rep,
            "missing_observation": True,
            "error": f"worker_exit_{proc.returncode}",
            "stderr_tail": (proc.stderr or "")[-1500:],
            "wall_s_measured": round(time.time() - t0, 3),
        }
    with open(out_path, encoding="utf-8") as fh:
        cell = json.load(fh)
    cell["replicate"] = rep
    cell["cell_path"] = os.path.relpath(out_path, REPO_ROOT)
    # A policy crash INSIDE the run is also a missing observation for the accounting: the
    # episode did not get to spend its budget, so its branch shares describe a truncated
    # run, not the agent. Recorded, never silently averaged in.
    rr = cell.get("result_row") or {}
    cell["missing_observation"] = bool(rr.get("error"))
    return cell


def analyse_cell(cell: dict, budget: int) -> dict:
    """Turn one episode into the accounting: branch shares, plan-derivation, where it lost.

    Reads TWO sources and keeps them distinct, because they disagree in an instructive way:

      * the per-action provenance rows -- what the policy's state was AT THE MOMENT of each
        choice. `induction_skipped` in a row is the attempt dict AS IT WAS THEN, and the
        induce path mutates that dict afterwards, so a row can read `""` for an attempt
        that finally records `degenerate_goal_predicate`.
      * `result_row.induction_events` -- the FINAL state of every induction attempt, which
        is where the engine-trust verdicts actually settle.

    Engine/plan verdicts are therefore taken from the events; only the action accounting is
    taken from the rows. Silently preferring one would have made the trust numbers wrong in
    a way nothing downstream could see.
    """
    out: dict[str, Any] = {
        "game": cell.get("game"),
        "replicate": cell.get("replicate"),
        "missing_observation": bool(cell.get("missing_observation")),
        "error": cell.get("error") or (cell.get("result_row") or {}).get("error"),
        "wall_s": cell.get("wall_s_measured"),
    }
    if out["missing_observation"]:
        return out

    prov = cell.get("provenance") or {}
    rows = prov.get("rows") or []
    summ = prov.get("summary") or {}
    rr = cell.get("result_row") or {}
    events = rr.get("induction_events") or []
    n = len(rows)
    out["actions_recorded"] = n
    out["budget"] = budget
    out["budget_consumed_fraction"] = round(n / budget, 6) if budget else None
    out["levels_banked"] = rr.get("levels_gained")
    out["actions_to_first_solve"] = rr.get("actions_to_first_solve")
    out["timed_out"] = rr.get("timed_out")
    out["hit_induction_cap"] = rr.get("hit_induction_cap")
    out["revisit_frac"] = rr.get("revisit_frac")
    out["hv_progress"] = rr.get("hv_progress")
    # The PRE-REGISTERED wall-truncation rule. Applied here rather than in the driver so it
    # is visible in the analysis every reader runs, and so re-running the report can never
    # quietly use a different threshold from the one the census ran under.
    if bool(rr.get("timed_out")) and (out["budget_consumed_fraction"] or 0.0) < (
        WALL_TRUNCATION_MIN_BUDGET_FRACTION
    ):
        out["missing_observation"] = True
        out["error"] = (
            f"wall_truncated_at_{n}_of_{budget}_actions_below_prereg_floor_"
            f"{WALL_TRUNCATION_MIN_BUDGET_FRACTION}"
        )
        out["wall_truncated_below_prereg_floor"] = True
        return out

    # -- THE HEADLINE: what share of the spent budget did the pipeline choose? -----------
    by_top = summ.get("by_top_branch") or {}
    plan_derived = sum(int(by_top.get(b, 0)) for b in PLAN_DERIVED_TOP)
    reset_for_plan = int(by_top.get("induce.plan_needs_reset", 0))
    out["n_plan_derived"] = plan_derived
    out["plan_derived_fraction"] = round(plan_derived / n, 6) if n else None
    out["n_reset_for_plan_replay"] = reset_for_plan
    out["by_top_branch"] = by_top
    out["by_explorer_branch"] = summ.get("by_explorer_branch") or {}
    out["by_serve_kind"] = summ.get("by_serve_kind") or {}
    out["new_information_expansions"] = summ.get("new_information_expansions")
    out["new_information_expansion_fraction"] = summ.get("new_information_expansion_fraction")
    out["navigation_or_replay_actions"] = summ.get("navigation_or_replay_actions")
    out["navigation_or_replay_fraction"] = summ.get("navigation_or_replay_fraction")
    # An unknown branch label means the policy grew a decision path this accounting does not
    # know about and would have silently mis-attributed. Surfaced, never bucketed as "other".
    out["unknown_top_branches"] = sorted(set(by_top) - set(_TOP_BRANCHES))
    out["unknown_explorer_branches"] = sorted(
        set(out["by_explorer_branch"]) - set(_EXPLORER_BRANCHES)
    )
    out["recorder_errors"] = summ.get("recorder_errors") or []
    # SETUP COST, separated from planning share. The pipeline cannot choose an action before
    # a plan exists, so a low plan-derived share has two very different explanations: the
    # agent planned late (setup cost) or it planned and then stopped (abandonment). These
    # two fields separate them, and without that separation the headline is ambiguous in the
    # direction that matters most for deciding which stage to attack.
    first_plan = next((r.get("i") for r in rows if r.get("top_branch") in PLAN_DERIVED_TOP), None)
    out["actions_before_first_plan_step"] = first_plan
    out["actions_before_first_plan_step_fraction"] = (
        round(first_plan / n, 6) if (first_plan is not None and n) else None
    )
    if first_plan is not None:
        after = n - first_plan
        out["plan_derived_fraction_of_post_first_plan_budget"] = (
            round(plan_derived / after, 6) if after else None
        )
    else:
        out["plan_derived_fraction_of_post_first_plan_budget"] = None

    # -- did the pipeline run at all, and how far did it get? ---------------------------
    out["induction_fired"] = bool(events)
    out["n_inductions"] = len(events)
    n_engine_produced = 0
    n_engine_trusted = 0
    n_planned = 0
    ev_rows = []
    for ev in events:
        skipped = ev.get("skipped") or ""
        rounds = ev.get("refinement_rounds") or []
        # An engine EXISTS if any refinement round got a proposer response back.
        produced = any(bool(r.get("proposer_ok")) for r in rounds)
        # TRUSTED means the held-out verifier accepted it -- the engine passed the gate the
        # agent itself uses to decide whether to plan on it. Read off the rounds, which is
        # where the induce path writes it, not inferred from `skipped`.
        trusted = any(bool(r.get("accepted_by_heldout_verifier")) for r in rounds)
        planned = bool(ev.get("planned")) and not skipped
        n_engine_produced += int(produced)
        n_engine_trusted += int(trusted)
        n_planned += int(planned)
        best = None
        for r in rounds:
            if r.get("accepted_by_heldout_verifier"):
                best = r
                break
        ev_rows.append(
            {
                "reason": ev.get("reason"),
                "skipped": skipped or None,
                "planned": bool(ev.get("planned")),
                "plan_length": ev.get("plan_length"),
                "engine_produced": produced,
                "engine_trusted_by_heldout_verifier": trusted,
                "heldout_accuracy": _f((best or {}).get("heldout_accuracy")),
                "trust_energy": _f((best or {}).get("trust_energy")),
                "goal_predicate_satisfiable": (best or {}).get("goal_predicate_satisfiable"),
                "counterexample_kind": ((best or {}).get("counterexample") or {}).get("kind"),
                "plan_termination_reason": (ev.get("ttt_prior_engine_plan_diagnostics") or {}).get(
                    "termination_reason"
                ),
            }
        )
    out["n_engine_produced"] = n_engine_produced
    out["n_engine_trusted"] = n_engine_trusted
    out["n_inductions_that_planned"] = n_planned
    out["engine_ever_trusted"] = n_engine_trusted > 0
    out["plan_ever_executed"] = plan_derived > 0
    out["induction_events"] = ev_rows
    out["induction_skip_reasons"] = sorted({e["skipped"] for e in ev_rows if e["skipped"]})

    # -- the level trajectory, and WHICH branch emitted the banking action ---------------
    # `level_before` and `level_after` in a row are BOTH read off the frame the policy is
    # looking at when it chooses, so within one row they are equal by construction. A
    # level-up is therefore visible as an INCREASE BETWEEN CONSECUTIVE ROWS, and the action
    # that caused it is the PREVIOUS row's. Spelled out because reading `level_after` as
    # "the level after this action" is the obvious and wrong interpretation.
    levels = [r.get("level_before") for r in rows]
    banks = []
    prev = None
    for i, lv in enumerate(levels):
        if lv is None:
            continue
        if prev is not None and lv > prev:
            causing = rows[i - 1] if i else None
            banks.append(
                {
                    "observed_at_row": i,
                    "level_from": prev,
                    "level_to": lv,
                    "causing_action_row": (causing or {}).get("i"),
                    "causing_action_branch": (causing or {}).get("top_branch"),
                    "causing_action_explorer_branch": (causing or {}).get("explorer_branch"),
                    "causing_action_serve_kind": (causing or {}).get("explorer_serve_kind"),
                    "causing_action_plan_present": (causing or {}).get("plan_present"),
                }
            )
        prev = lv
    # THE CAUSAL ORDERING, which is stronger than the branch attribution and cheaper to
    # trust. Branch attribution says "an explorer action banked this level"; someone can
    # still argue the plan set up the state that made it possible. But a level banked
    # STRICTLY BEFORE the pipeline emitted its first action cannot have been caused by the
    # pipeline in any sense at all -- the pipeline had not acted yet. That argument needs no
    # model of the game, so it is recorded as its own field rather than left implicit in two
    # row indices a reader would have to compare by hand.
    for b in banks:
        b["before_first_plan_action"] = (
            None
            if first_plan is None
            else bool((b.get("causing_action_row") or 0) < int(first_plan))
        )
    out["level_up_events"] = banks
    out["n_level_ups_seen_in_rows"] = len(banks)
    out["level_ups_from_plan_branch"] = sum(
        1 for b in banks if b["causing_action_branch"] in PLAN_DERIVED_TOP
    )
    out["level_ups_strictly_before_the_pipelines_first_action"] = sum(
        1 for b in banks if b.get("before_first_plan_action") is True
    )
    out["level_ups_with_no_plan_action_anywhere_in_the_episode"] = (
        len(banks) if first_plan is None else 0
    )

    # -- plan segments: was a plan executed, and did anything come of it? ----------------
    segs = []
    cur: Optional[dict] = None
    for r in rows:
        if r.get("top_branch") in PLAN_DERIVED_TOP:
            if cur is None or r.get("plan_epoch") != cur["plan_epoch"]:
                cur = {
                    "plan_epoch": r.get("plan_epoch"),
                    "start_row": r.get("i"),
                    "end_row": r.get("i"),
                    "n_actions": 0,
                    "plan_len": r.get("plan_len"),
                    "level_at_start": r.get("level_before"),
                }
                segs.append(cur)
            cur["end_row"] = r.get("i")
            cur["n_actions"] += 1
        else:
            cur = None
    for s in segs:
        after = [r for r in rows if (r.get("i") or 0) > s["end_row"]]
        lv_after = next(
            (r.get("level_before") for r in after if r.get("level_before") is not None), None
        )
        s["level_after_segment"] = lv_after
        s["level_advanced_during_or_after_plan"] = bool(
            lv_after is not None
            and s["level_at_start"] is not None
            and lv_after > s["level_at_start"]
        )
    out["plan_segments"] = segs
    out["n_plan_segments"] = len(segs)
    out["n_plan_segments_that_advanced_level"] = sum(
        1 for s in segs if s["level_advanced_during_or_after_plan"]
    )
    out["plans_abandoned"] = summ.get("plans_abandoned")
    out["plans_consumed_fully"] = summ.get("plans_consumed_fully")

    # -- WHERE IS THE LEVEL LOST? --------------------------------------------------------
    # A single terminal label per episode, assigned by the FIRST stage that failed, so the
    # labels partition the episodes rather than overlapping. Deliberately blunt: the point
    # is to name the stage to attack, and a label that hedges names nothing.
    banked = int(out["levels_banked"] or 0)
    if banked > 0:
        where = "banked_level"
    elif not events:
        where = "no_induction_ever_fired"
    elif n_engine_produced == 0:
        where = "induction_fired_but_no_engine_produced"
    elif n_engine_trusted == 0:
        where = "engine_produced_but_never_trusted_by_heldout_verifier"
    elif plan_derived == 0:
        where = "engine_trusted_but_no_plan_ever_executed"
    else:
        where = "plan_executed_but_level_not_gained"
    out["where_it_is_lost"] = where
    # The budget-side story, orthogonal to the pipeline story and, on the pilot, the larger
    # effect: almost none of the explorer's actions test anything new.
    out["budget_story"] = {
        "explorer_fraction": summ.get("explorer_fraction"),
        "navigation_or_replay_fraction": summ.get("navigation_or_replay_fraction"),
        "new_information_expansion_fraction": summ.get("new_information_expansion_fraction"),
        "explored_out_reached": any(r.get("explorer_explored_out") for r in rows),
        "final_explorer_graph_nodes": (rows[-1].get("explorer_graph_nodes") if rows else None),
        "max_explorer_depth_seen": max(
            (r.get("explorer_cur_depth") or 0 for r in rows), default=None
        ),
    }
    return out


def _spread(vals: list[float]) -> dict:
    """min / median / max plus the raw list. No CI: R is 3, and a CI computed off three
    samples of a sampler-driven process would give the number a false authority."""
    clean = [v for v in vals if v is not None]
    if not clean:
        return {"n": 0, "min": None, "median": None, "max": None, "values": []}
    return {
        "n": len(clean),
        "min": round(min(clean), 6),
        "median": round(statistics.median(clean), 6),
        "max": round(max(clean), 6),
        "values": [round(v, 6) for v in clean],
    }


def aggregate(cells: list[dict], budget: int) -> dict:
    """Per-game aggregation over replicates, with missing observations kept OUT."""
    games: dict[str, list[dict]] = {}
    for c in cells:
        games.setdefault(str(c.get("game")), []).append(c)
    per_game = []
    for game, cs in games.items():
        ok = [c for c in cs if not c.get("missing_observation")]
        missing = [c for c in cs if c.get("missing_observation")]
        entry: dict[str, Any] = {
            "game": game,
            "replicates_attempted": len(cs),
            "replicates_observed": len(ok),
            "missing_observations": [
                {"replicate": c.get("replicate"), "error": c.get("error")} for c in missing
            ],
        }
        if not ok:
            entry["note"] = "NO OBSERVED EPISODE -- nothing is reported for this game."
            per_game.append(entry)
            continue
        entry["actions_recorded"] = _spread([c.get("actions_recorded") for c in ok])
        entry["plan_derived_fraction"] = _spread([c.get("plan_derived_fraction") for c in ok])
        entry["n_plan_derived"] = _spread([c.get("n_plan_derived") for c in ok])
        entry["new_information_expansion_fraction"] = _spread(
            [c.get("new_information_expansion_fraction") for c in ok]
        )
        entry["navigation_or_replay_fraction"] = _spread(
            [c.get("navigation_or_replay_fraction") for c in ok]
        )
        entry["levels_banked"] = _spread([c.get("levels_banked") for c in ok])
        entry["induction_fired_in_all_replicates"] = all(c.get("induction_fired") for c in ok)
        entry["engine_ever_trusted_in_any_replicate"] = any(
            c.get("engine_ever_trusted") for c in ok
        )
        entry["plan_ever_executed_in_any_replicate"] = any(c.get("plan_ever_executed") for c in ok)
        entry["where_it_is_lost"] = sorted({str(c.get("where_it_is_lost")) for c in ok})
        # The per-game branch census, summed over observed replicates. Shares are of the
        # SUM of the observed budgets, so a short replicate cannot dominate a long one.
        tot: dict[str, int] = {}
        tot_n = 0
        for c in ok:
            tot_n += int(c.get("actions_recorded") or 0)
            for k, v in (c.get("by_top_branch") or {}).items():
                tot[k] = tot.get(k, 0) + int(v)
        entry["pooled_actions"] = tot_n
        entry["pooled_by_top_branch"] = dict(sorted(tot.items(), key=lambda kv: -kv[1]))
        entry["pooled_by_top_branch_share"] = {
            k: round(v / tot_n, 6) for k, v in entry["pooled_by_top_branch"].items() if tot_n
        }
        exp_tot: dict[str, int] = {}
        for c in ok:
            for k, v in (c.get("by_explorer_branch") or {}).items():
                exp_tot[k] = exp_tot.get(k, 0) + int(v)
        entry["pooled_by_explorer_branch"] = dict(sorted(exp_tot.items(), key=lambda kv: -kv[1]))
        entry["level_up_events"] = [b for c in ok for b in (c.get("level_up_events") or [])]
        entry["induction_skip_reasons"] = sorted(
            {s for c in ok for s in (c.get("induction_skip_reasons") or [])}
        )
        per_game.append(entry)
    per_game.sort(key=lambda e: e["game"])
    return {"per_game": per_game}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--games",
        default="vc33,tn36,ft09,tr87,lp85",
        help=(
            "chosen from the PRIOR record (results/outer_loop_arc_heldout_31b_vs_9b_"
            "banked_levels_20260728.json), never from a peek at this run: vc33 and lp85 "
            "BANK levels under the frozen 31B generator; tn36/ft09/tr87 do not."
        ),
    )
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--budget", type=int, default=400, help="live scored MAX_ACTIONS is 400")
    ap.add_argument("--max-inductions", type=int, default=4)
    ap.add_argument(
        "--explore-budget",
        default="routed",
        help=(
            "'routed' (DEFAULT) lets each cell resolve this game's explore_budget exactly as "
            "the SCORED agent does; an integer pins one value across every game. Pinning is "
            "what the 2026-08-01 census did (24), and because `make_carnot_agent` passes no "
            "explore_budget and `_route_explore_budget` returns 24 for program_editor and 80 "
            "for graph_explore, that ran 4 of 5 games at a third of their scored budget. The "
            "budget is the induce stall threshold and, via `_active_transitions()`, the size "
            "of the induction prompt's evidence -- so pinning it changes what the pipeline is "
            "measured on, not just when it fires. Pass an integer only when holding the "
            "budget fixed across games IS the design, and say so in the artifact."
        ),
    )
    ap.add_argument("--wall-s", type=float, default=1200.0)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--cuda-gpu", default="1", help="outer loop owns GPU 1; never GPU 0")
    ap.add_argument("--cuda-port-base", type=int, default=8971)
    ap.add_argument(
        "--outdir",
        default=os.path.join(REPO_ROOT, "results", "arc_live_action_provenance_20260801"),
    )
    ap.add_argument(
        "--scratch",
        default=os.environ.get("CARNOT_ARC_PROV_SCRATCH", "/tmp/arc_action_provenance_census"),
        help="scratch root for the PER-CELL engine stores; never inside results/",
    )
    args = ap.parse_args()

    games = [g.strip() for g in args.games.split(",") if g.strip()]
    cells_dir = os.path.join(args.outdir, "cells")
    os.makedirs(cells_dir, exist_ok=True)
    os.makedirs(args.scratch, exist_ok=True)

    t0 = time.time()
    raw_cells: list[dict] = []
    card_waits: list[dict] = []
    port = args.cuda_port_base
    first = True
    # ROUND-ROBIN BY REPLICATE, not game-major. An interruption then leaves a COMPLETE
    # lower-replicate design across every game instead of full data on the first two games
    # and nothing on the rest -- the difference between a weaker measurement and no
    # cross-game measurement at all.
    for rep in range(args.replicates):
        for game in games:
            if not first:
                reaped = _reap_my_generator(port)
                wait = _wait_for_card(args.cuda_gpu, 23000, 600.0)
                wait.update({"before_cell": f"{game}_r{rep}", "reaped_servers": reaped})
                card_waits.append(wait)
                print(f"[census] reaped={reaped} card={wait}", flush=True)
            first = False
            port += 1
            print(f"[census] === {game} rep {rep} (port {port}) ===", flush=True)
            cell = run_cell(
                game=game,
                rep=rep,
                seed=args.seed,
                budget=args.budget,
                max_inductions=args.max_inductions,
                explore_budget=args.explore_budget,
                wall_s=args.wall_s,
                timeout=args.timeout,
                cuda_gpu=args.cuda_gpu,
                port=port,
                cells_dir=cells_dir,
                scratch_root=args.scratch,
            )
            raw_cells.append(cell)
            an = analyse_cell(cell, args.budget)
            print(
                f"[census] {game} r{rep}: missing={an['missing_observation']} "
                f"actions={an.get('actions_recorded')} plan_derived={an.get('n_plan_derived')} "
                f"({an.get('plan_derived_fraction')}) levels={an.get('levels_banked')} "
                f"where={an.get('where_it_is_lost')}",
                flush=True,
            )
            # Write the running analysis after EVERY cell. A two-hour run that is killed at
            # 90 minutes must leave a usable partial record, not an empty directory.
            with open(os.path.join(args.outdir, "_partial_cells.json"), "w", encoding="utf-8") as f:
                json.dump(
                    [analyse_cell(c, args.budget) for c in raw_cells], f, indent=1, default=str
                )
    # Reap the LAST cell's server. Leaving ~21 GB pinned on a shared card after exit is a
    # worse citizen than never having run.
    _reap_my_generator(port)

    analysed = [analyse_cell(c, args.budget) for c in raw_cells]
    agg = aggregate(analysed, args.budget)
    observed = [a for a in analysed if not a.get("missing_observation")]
    failed = [a for a in observed if not (a.get("levels_banked") or 0)]
    banked = [a for a in observed if (a.get("levels_banked") or 0)]

    def _pooled(subset: list[dict]) -> dict:
        tot_n = sum(int(a.get("actions_recorded") or 0) for a in subset)
        tot_plan = sum(int(a.get("n_plan_derived") or 0) for a in subset)
        return {
            "episodes": len(subset),
            "actions": tot_n,
            "plan_derived_actions": tot_plan,
            "plan_derived_fraction": round(tot_plan / tot_n, 6) if tot_n else None,
        }

    headline = {
        "question": (
            "In an episode the live agent FAILS, what share of the actions it spends did "
            "the induce->verify->plan pipeline choose?"
        ),
        "failed_episodes_pooled": _pooled(failed),
        "banked_episodes_pooled": _pooled(banked),
        "all_observed_pooled": _pooled(observed),
        "per_episode_plan_derived_fraction_failed": _spread(
            [a.get("plan_derived_fraction") for a in failed]
        ),
        "per_episode_plan_derived_fraction_banked": _spread(
            [a.get("plan_derived_fraction") for a in banked]
        ),
    }
    return _write(args, raw_cells, analysed, agg, headline, card_waits, t0, games)


def _write(args, raw_cells, analysed, agg, headline, card_waits, t0, games) -> int:
    observed = [a for a in analysed if not a.get("missing_observation")]
    missing = [a for a in analysed if a.get("missing_observation")]
    failed = [a for a in observed if not (a.get("levels_banked") or 0)]

    frac = headline["failed_episodes_pooled"]["plan_derived_fraction"]
    if not observed:
        verdict = "blocked_no_episode_observed_every_cell_was_a_missing_observation"
    elif not failed:
        verdict = "complete_no_failed_episode_observed_headline_undefined_see_banked_only"
    elif frac is not None and frac < 0.05:
        verdict = (
            "complete_induce_plan_pipeline_chose_under_5pct_of_the_actions_in_failed_"
            "episodes_budget_is_spent_by_the_explorer"
        )
    elif frac is not None and frac < 0.20:
        verdict = (
            "complete_induce_plan_pipeline_chose_a_small_minority_of_the_actions_in_"
            "failed_episodes_budget_is_spent_by_the_explorer"
        )
    else:
        verdict = (
            "complete_induce_plan_pipeline_chose_a_material_share_of_the_actions_in_"
            "failed_episodes_plans_are_executed_and_wrong"
        )

    artifact = {
        "experiment": "outer_loop_arc_live_action_provenance_census",
        "schema": "carnot.arc.action_provenance_census.v1",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "milestone": "2026.08.outer_loop",
        "honest_verdict": verdict,
        "duration_s": round(time.time() - t0, 3),
        "random_seed": args.seed,
        "inference_substrate": "live_llm_inference",
        "inference_substrate_note": (
            "Each episode loads the frozen live-submission generator (gemma-4-31B-it GGUF) "
            "on an RTX 3090 via llama-server and runs REAL autoregressive induction inside "
            "the agent's cascade, so this is live LLM inference, not the no-LLM offline "
            "arcade substrate. The ENVIRONMENT is the OFFLINE arcade "
            "(arc_solver_kit.offline_arcade, OperationMode.OFFLINE over local "
            "environment_files/): no scorecard, no gateway, no network game, no submission."
        ),
        "solve_provenance": "development_proxy",
        "solve_provenance_note": (
            "This is an INSTRUMENT run, not a solve attempt. It banks nothing it claims "
            "credit for, registers no new level, and tunes nothing to make a game succeed. "
            "Levels that happen to be banked during a census episode are OBSERVATIONS of "
            "the live path on the public development twin, not claimed solves."
        ),
        "verifier_is_oracle": {
            "value": False,
            "principle": (
                "no win oracle is consulted anywhere in the measurement. The recorded "
                "quantity is WHICH CODE BRANCH emitted each action -- a fact about the "
                "agent, not about whether the action was correct. The level counter is read "
                "from the environment frame as an observation, and it is the environment's "
                "own gate, not a heuristic this measurement invented."
            ),
        },
        "config": {
            "games": games,
            "game_selection_rationale": (
                "Chosen from the PRIOR record (2026-07-28 banked-levels study) BEFORE this "
                "run, never from its output: vc33 (banked 2) and lp85 (banked 1) are the "
                "contrast cases the brief requires; tn36 is the sharpest failure (an engine "
                "with held-out accuracy 1.0 and 0 levels); ft09 and tr87 spent ~396 actions "
                "for 0 levels with the engine rejected."
            ),
            "replicates_per_game": args.replicates,
            "budget_actions": args.budget,
            "max_inductions": args.max_inductions,
            # What was REQUESTED, which under the default is the string "routed" rather than
            # a number. The number each cell actually ran is resolved per-game in the worker
            # and recorded there as `explore_budget` + `explore_budget_provenance`; that is
            # the authoritative value, because with routing there is no single one.
            "explore_budget_requested": args.explore_budget,
            "wall_s_cap": args.wall_s,
            "policy": "E3AgentPolicy via arc_actions_to_progress.run_bounded_progress",
            "policy_game_id": "the real game id (NOT the anonymized held-out condition)",
            "cuda_gpu": args.cuda_gpu,
            "instrument": "CARNOT_ARC_ACTION_PROVENANCE=1",
        },
        "live_path_entrypoint": (
            "python/carnot/agentic/arc_competition_agent.py :: E3AgentPolicy.next_move -- "
            "entrypoint 1, the SCORED agent's own per-action cascade, reached through "
            "arc_actions_to_progress.run_bounded_progress. Not a bespoke solver."
        ),
        "headline": headline,
        "aggregate": agg,
        "episodes": analysed,
        "missing_observations": [
            {"game": m.get("game"), "replicate": m.get("replicate"), "error": m.get("error")}
            for m in missing
        ],
        "missing_observation_policy": (
            "A crash, a worker non-zero exit, a wall-clock timeout or a policy error inside "
            "the run is recorded as MISSING and excluded from every aggregate. It is never "
            "folded in as a zero: a zero means the agent spent its budget and none of it was "
            "plan-derived; a missing observation means we did not see what it would have done."
        ),
        "card_waits_between_cells": card_waits,
        "cells_dir": os.path.relpath(os.path.join(args.outdir, "cells"), REPO_ROOT),
    }
    checksum_src = json.dumps(
        {
            "episodes": [
                {
                    k: a.get(k)
                    for k in (
                        "game",
                        "replicate",
                        "actions_recorded",
                        "n_plan_derived",
                        "levels_banked",
                        "by_top_branch",
                        "where_it_is_lost",
                    )
                }
                for a in analysed
            ],
            "seed": args.seed,
            "games": games,
        },
        sort_keys=True,
        default=str,
    ).encode()
    artifact["reproducibility_checksum"] = "sha256:" + hashlib.sha256(checksum_src).hexdigest()

    out_path = os.path.join(args.outdir, "artifact_raw.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=1, default=str)
    print(json.dumps(headline, indent=1))
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Exp 6012 -- REQ-ARC-WMTE-6012: does the HIDDEN-STATE trust gate have the same hole?

WHY THIS EXPERIMENT EXISTS
==========================
REQ-ARC-WMTE-6011 (exp6011) shipped a change-weighted trust gate that rejects the
identity engine the legacy `accuracy >= 0.5` gate admits. It is wired into exactly ONE
of the live agent's two admission branches -- the `else` (non-hidden-state) branch of
`arc_competition_agent._induce_and_plan`.

The OTHER branch is taken when `self.short in HIDDEN_STATE_GAME_IDS` (11 games:
ar25 cd82 cn04 dc22 g50t ka59 m0r0 re86 sc25 sk48 wa30 -- which INCLUDES every one of
the 0.08-wall games cn04/ar25/sc25/sk48/wa30). That branch never calls
`change_gate_decision`. It admits on `trust_score.trust_pass`, which
`arc_world_model_trust_energy.score_change_weighted_consistency` defines as

    trust_pass := nondegenerate AND consistency >= 0.5
    consistency := correct_changed_cells / true_changed_cells

`consistency` masks to TRULY-CHANGED cells only (`pred[changed] == t_next[changed]`),
exactly like `WorldModelVerifier.cell_recall`. That module's own docstring, added
2026-07-27, states the limit and declines to fix it there:

    "HONEST LIMIT ... it masks to TRUE changes only, so it cannot see a cell the engine
     wrote that reality did not change. It is recall, not fidelity ... This function is
     left on its existing metric deliberately: it is the SHIPPED hidden-state gate and
     changing its meaning here would silently move a live gate."

That is a stated limit, not a measured one. NOTHING in exp6011 -- not the artifact, not
the 25-test suite, not the four-arm script -- calls `select_trusted_world_model` even
once (verified by grep: 0 hits in all three). So the hidden-state branch's behaviour
under the identity engine and under a spurious-writer is UNMEASURED. This experiment
measures it, on the real games, through the production function.

It also settles a second thing exp6011 could not: its must-not-fire control is a
hand-written **dc22** navigation engine -- and dc22 is a HIDDEN-STATE game. So the one
positive control proving the new gate's pass region is non-empty was measured on a game
whose live path takes the branch the new gate is not wired into. That does not make the
control wrong (the metric is game-agnostic), but it does mean the new gate has never been
shown to admit anything on a game that would actually route through it.

WHAT IS MEASURED, AND WHY EACH ARM EXISTS
=========================================
Per hidden-state game x seed x mask setting, every candidate engine is scored by BOTH
gates on the SAME FULL transition set -- which is what the live agent feeds each of them:

  * LIVE HIDDEN-STATE GATE -- via `select_trusted_world_model(transitions,
    hidden_state=True)` called with a ONE-CANDIDATE pool, so `selected_score` is that
    candidate and `trust_pass` is literally the boolean the live agent branches on. The
    function performs its own 1/3 held-out split INTERNALLY. Calling the production
    function rather than re-deriving its formula is deliberate: this project has twice had
    two independent reimplementations of the same wrong formula agree with each other and
    both be wrong. Agreement between reconstructions is not evidence.

  * THE REQ-6011 GATE -- `change_gate_decision(WorldModelVerifier(transitions).score(
    engine), enabled=True)` on the identical transitions.

An earlier version pre-split and passed the tail to the selector, which split the tail
AGAIN -- scoring the last 1/9 and rejecting even a perfect engine. Matching the production
call site is the fix; reconstructing what it "must" do is what produced the error.

BOTH MASK SETTINGS, because the two repairs interact on the positive control (measured:
REQ-6011 admits the hand-written engine on 3/3 rows with the mask and 0/3 without). Note
that SUPPLYING a mask is not ENABLING it -- `CARNOT_ARC_WM_HUD_MASK` must be set, or the
verifier records `hud_mask_status: "disabled"` and grades unmasked. This file shipped that
bug once: the mask=1 arm was byte-identical to mask=0 and would have been published as
"the mask changes nothing". There is now an acceptance gate on the arm being live.

The candidates:

  ondisk                     the real results/arc_e3/<game>/world_model.py, production loader.
  identity                   `return grid`. THE ORIGIN INCIDENT. Must be rejected.
  base                       the honest reference this game's attack arms perturb (below).
  base_plus_spurious         `base`, plus ONE write reality never made.
  base_plus_noop_hallucination
                             `base`, plus an invented change on every TRUE NO-OP.

The last two are the hole, if there is one: both are wrong about the world in a way that
`plan_in_model` would walk forward at every step, and both differ from `base` ONLY in
cells that a recall-style metric masks away.

WHAT `base` IS, AND WHY IT IS NOT A LOOKUP TABLE ON dc22
--------------------------------------------------------
On **dc22** `base` is the hand-written navigation engine from exp6011 -- a ~20-line
general rule (2x2 avatar, 2-cell step, blocked unless the destination footprint is free)
with no per-transition constants, imported unmodified rather than re-typed. dc22 is the
game exp6011's must-not-fire control was built for, AND it is in HIDDEN_STATE_GAME_IDS,
so it is exactly the right place to ask "does the new gate admit a genuinely good engine
on a game that actually routes through the hidden-state branch?"

On the other 10 games `base` is `visible_lookup`: the best predictor expressible as a
function of (visible grid, action), tabulated from the corpus.

MEASURED, NOT ASSUMED -- AND THE FIRST VERSION OF THIS FILE GOT IT WRONG: `visible_lookup`
was originally called `corpus_oracle` and asserted to be perfect by construction. It is
NOT, and the measurement said so (dc22 seed 0: legacy accuracy 0.85, no-op hallucination
rate 0.375). The reason is the entire premise of this branch -- these are HIDDEN-STATE
games, so the same visible grid under the same action has different successors depending
on state the grid does not show, and any table keyed on what is visible must be wrong on
the collisions. `visible_lookup_collision_rate` is reported per row so this is a published
number rather than a footnote. It is therefore used as an ATTACK BASELINE only, never as
the must-not-fire control.

MUST-NOT-FIRE: the dc22 hand-written `base` must be ADMITTED. Measured: REQ-6011 admits it
on 3/3 rows (mask on), but the LIVE hidden-state gate rejects it on 2/3 -- so that gate is
simultaneously too STRICT on an honest partial model and BLIND to a spurious writer. A gate
that rejects everything is not an improvement over a gate that admits identity engines.

NO GPU, NO MODEL. Pure numpy over cached transitions from the offline arcade:
inference_substrate = verifier_ensemble_against_cached_candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts"))

from carnot.agentic import arc_executable_world_model as e3  # noqa: E402
from carnot.agentic import arc_world_model_trust_energy as te  # noqa: E402
from analyze_scored_path_lever_ab import preserve_freshness_acknowledgements  # noqa: E402

OUT = REPO / "results" / "experiment_6012_hidden_state_trust_gate_hole.json"
E3_DIR = REPO / "results" / "arc_e3"


def _file_sha256(p) -> str | None:
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except Exception:
        return None


# REQ-ARC-WMTE-6016: the code this artifact's numbers depend on, repo-relative. Added
# 2026-07-27 after a review found this lane's artifacts were INVISIBLE to
# scripts/artifact_freshness_lint.py -- it reported 8 other artifacts as drifted and these as
# nothing, and their freshness was then inferred from that silence. A lint with no entry for
# an artifact is not evidence about it.
PROVENANCE_CODE_PATHS = (
    "python/carnot/agentic/arc_executable_world_model.py",
    "python/carnot/agentic/arc_world_model_trust_energy.py",
    "python/carnot/agentic/arc_competition_agent.py",
)


def _rel_or_abs(p) -> str:
    """Repo-relative path if possible, absolute otherwise.

    RESOLVE FIRST -- do not drop this. `REPO` is the RESOLVED repo root, but this
    environment exposes the repo under two names: the real path and a symlink alias
    (.../Carnot-EBM/carnot-ebm -> .../ianblenke/carnot). A caller who passes a path
    built from the alias (e.g. `CARNOT_ARC_E3_DIR=$PWD/...` from a shell sitting in
    the alias) hands us a string that points at the SAME directory but does not share
    a textual prefix with REPO, so `relative_to` raises and we silently publish an
    ABSOLUTE, machine-specific path into `provenance.engine_store` -- changing a
    published provenance string on a rebuild that should have been a no-op. Resolving
    the input first collapses both names to the same real path so the comparison
    works whichever alias the caller used. (Hit for real on 2026-07-28; the rebuild
    had to be redone.)
    """
    try:
        return str(pathlib.Path(p).resolve().relative_to(REPO))
    except ValueError:
        return str(pathlib.Path(p).resolve())


def _provenance() -> dict:
    import carnot.agentic.arc_executable_world_model as _e3prov

    paths = list(PROVENANCE_CODE_PATHS) + [str(pathlib.Path(__file__).resolve().relative_to(REPO))]
    return {
        "code": [
            {"path": rel, "sha256": _file_sha256(REPO / rel)}
            for rel in paths
            if _file_sha256(REPO / rel) is not None
        ],
        # Never `.relative_to` here: the store may point outside the repo (a per-arm
        # scratch dir), and a provenance helper must not crash a completed measurement.
        "engine_store": _rel_or_abs(_e3prov.E3_DIR),
        "engine_store_is_frozen_fixtures": _e3prov.E3_DIR.name == "arc_e3_origin_fixtures",
        "rebuild_command": (
            # NOT $PWD: from the .../Carnot-EBM/carnot-ebm symlink alias, $PWD
            # yields a path that points at this repo but shares no textual prefix
            # with the resolved REPO, which used to flip provenance.engine_store to
            # an absolute path on rebuild. `git rev-parse --show-toplevel` always
            # yields the canonical root, whichever alias the shell is sitting in.
            'CARNOT_ARC_E3_DIR="$(git rev-parse --show-toplevel)"/results/arc_e3_origin_fixtures .venv/bin/python '
            + str(pathlib.Path(__file__).resolve().relative_to(REPO))
        ),
    }


# The engine names whose ADMISSION is a defect (they are wrong about the world), as
# distinct from `identity` whose admission is the already-known origin defect.
SPURIOUS_ARMS = ("base_plus_spurious", "base_plus_noop_hallucination")
# The one game where `base` is a hand-written general rule rather than a lookup table, and
# therefore the only game whose `base` arm is a legitimate must-not-fire control.
HANDWRITTEN_GAME = "dc22"


def _key(grid: np.ndarray, action: int) -> bytes:
    """Content key for the lookup oracle. Includes the action: the same grid under two
    different actions has two different successors, and collapsing them would build an
    oracle that is silently wrong rather than perfect."""

    a = np.ascontiguousarray(np.asarray(grid, dtype=np.int64))
    return hashlib.sha256(a.tobytes() + b"|" + str(int(action)).encode()).digest()


def _free_cell(shape, mask) -> tuple[int, int]:
    """A cell the HUD mask does NOT cover, for the attack arms to write into.

    LOAD-BEARING, and found the hard way. The attacks originally wrote to `[0, 0]`. Under
    REQ-6010 the comparison COLLAPSES masked cells, so on any game whose HUD includes the
    top-left corner the spurious write was erased before scoring and the attack silently
    became the honest engine -- the measured `spurious_per_changing_transition` for the
    attack arm dropped to 0.0 and the separation "disappeared". That would have been
    published as "the proposed channel does not separate", which is the opposite of true.
    An attack must perturb a cell that is actually compared.
    """

    if mask is None:
        return (0, 0)
    m = np.asarray(mask)
    free = np.argwhere(~m)
    if len(free) == 0:
        return (0, 0)
    return (int(free[0][0]), int(free[0][1]))


def _make_arms(game: str, transitions, mask=None):
    """Build the candidate engines for one game's corpus, plus the base's own diagnostics.

    Returns (arms, base_info). `base_info` carries the visible-lookup collision rate, which
    is the published evidence that a function of the VISIBLE grid cannot be perfect on a
    hidden-state game -- the reason `visible_lookup` is an attack baseline and not a control.
    """

    table: dict[bytes, np.ndarray] = {}
    collisions = 0
    for t in transitions:
        k = _key(t.grid, t.action)
        nxt = np.asarray(t.next_grid)
        if k in table and not np.array_equal(table[k], nxt):
            # Same VISIBLE grid, same action, DIFFERENT successor. Only hidden state can
            # explain this, and it is the exact reason a visible-keyed table must be wrong.
            collisions += 1
        table[k] = nxt.copy()

    def identity(grid, action, data):
        return grid

    def visible_lookup(grid, action, data):
        hit = table.get(_key(grid, action))
        return np.asarray(grid).copy() if hit is None else hit.copy()

    if game == HANDWRITTEN_GAME:
        # Imported, not re-typed: two independent reimplementations of one formula agreeing
        # with each other is not evidence about the system (this project has been burned by
        # exactly that). The exp6011 module is the single definition.
        from experiment_6011_world_model_change_gate_four_arm import (  # noqa: E402
            dc22_navigation_engine,
        )

        base = dc22_navigation_engine
        base_kind = "handwritten_general_rule"
    else:
        base = visible_lookup
        base_kind = "visible_lookup_table"

    shape = np.asarray(transitions[0].grid).shape
    sy, sx = _free_cell(shape, mask)

    def base_plus_spurious(grid, action, data):
        g = np.asarray(base(grid, action, data)).copy()
        # One cell, set to a value the corpus never contains, so the write can never
        # coincide with a real change and be excused as correct -- and in a cell the HUD
        # mask does not erase, so the attack survives the masked comparison.
        g[sy, sx] = 999
        return g

    def base_plus_noop_hallucination(grid, action, data):
        g = np.asarray(base(grid, action, data)).copy()
        if np.array_equal(g, np.asarray(grid)):  # reality did nothing -> invent something
            g[sy, sx] = 998
        return g

    arms = {
        "identity": identity,
        "base": base,
        "base_plus_spurious": base_plus_spurious,
        "base_plus_noop_hallucination": base_plus_noop_hallucination,
    }
    base_info = {
        "base_kind": base_kind,
        "visible_lookup_collisions": int(collisions),
        "visible_lookup_collision_rate": round(collisions / max(1, len(transitions)), 6),
        "attack_write_cell": [int(sy), int(sx)],
        "attack_write_cell_is_masked": bool(mask is not None and bool(np.asarray(mask)[sy, sx])),
    }
    return arms, base_info


def _frame_hud_mask(game: str):
    """The live explorer's own HUD classifier, on this game's first real frame.

    Imported from the agent rather than re-derived: the whole point of REQ-6010 is that the
    comparison must use the SAME mask the explorer already computes.
    """

    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import _compute_hud_mask_from_frame

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    return _compute_hud_mask_from_frame(env.reset())


def _live_trust_pass(transitions, engine, hud_mask=None) -> dict:
    """The LIVE hidden-state admission decision, via the production function.

    One-candidate pool -> `selected_score` is this engine, so `trust_pass` is exactly the
    boolean `arc_competition_agent._induce_and_plan` branches on for a hidden-state game.
    """

    sel = te.select_trusted_world_model(
        list(transitions),
        [te.WorldModelCandidate(name="probe", engine=engine)],
        hidden_state=True,
        hud_mask=hud_mask,
    )
    s = sel.selected_score
    return {
        "trust_pass": bool(s.trust_pass),
        "nondegenerate": bool(s.nondegenerate),
        "heldout_change_consistency": round(float(s.heldout_change_consistency), 6),
        "heldout_accuracy": round(float(s.heldout_accuracy), 6),
        "correct_changed_cells": int(s.correct_changed_cells),
        "true_changed_cells": int(s.true_changed_cells),
        "binary_gate_pass": bool(s.binary_gate_pass),
    }


def _one(game: str, n: int, seed: int) -> dict:
    row: dict = {"game": game, "seed": seed, "n_requested": n}
    t0 = time.time()
    try:
        trans, cell = e3.collect_transitions(game, n=n, seed=seed)
    except Exception as exc:
        row["error"] = f"collect_transitions:{type(exc).__name__}:{exc!r}"[:300]
        row["elapsed_s"] = round(time.time() - t0, 3)
        return row
    row["n_transitions"] = len(trans)
    row["cell"] = int(cell)
    if len(trans) < 2:
        row["error"] = "too_few_transitions"
        row["elapsed_s"] = round(time.time() - t0, 3)
        return row

    # BOTH gates are fed the FULL transition set, because that is what the live agent does:
    # the hidden-state branch calls `select_trusted_world_model(active_transitions, ...)`
    # (which performs its own 1/3 held-out split INTERNALLY) and the non-hidden branch calls
    # `WorldModelVerifier(active_transitions)`. An earlier version of this file pre-split and
    # passed the tail to the selector, which then split the tail AGAIN -- scoring the last
    # 1/9 of the corpus and rejecting even the honest engine, because on this fixture the
    # tail-of-the-tail happened to contain no changing transition at all. Matching the
    # production call site exactly is the fix; reconstructing what it "must" do is what
    # produced the error.
    _prefix, heldout = te._split_prefix_heldout(trans)
    row["n_heldout"] = len(heldout)
    row["n_heldout_changing"] = int(
        sum(1 for t in heldout if not np.array_equal(np.asarray(t.grid), np.asarray(t.next_grid)))
    )

    # REQ-ARC-WMTE-6010 interaction. exp6011's own must-not-fire control passes the WHOLE
    # corpus only with the HUD mask ON (`mask_flips_reject_to_admit_all_seeds: true`), so
    # measuring this gate at one mask setting would confound "the gate rejects good engines"
    # with "the HUD makes exact match unattainable". Both settings, per row.
    try:
        fm = _frame_hud_mask(game)
        mask = e3.logical_hud_mask(fm, cell)
        row["hud_mask_cells"] = int(np.asarray(mask).sum()) if mask is not None else 0
        row["hud_mask_status"] = "resolved" if mask is not None else "unresolved"
    except Exception as exc:
        mask = None
        row["hud_mask_status"] = f"error:{type(exc).__name__}"[:80]
        row["hud_mask_error"] = repr(exc)[:200]

    arms, base_info = _make_arms(game, trans, mask=mask)
    row.update(base_info)
    try:
        ondisk, done = e3.load_engine(game)
        arms = {"ondisk": ondisk, **arms}
        row["ondisk_loaded"] = True
        # Item 5 (downstream): if the on-disk model's win predicate is never True on any
        # observed grid, then trusting it could not have produced a plan anyway, so
        # rejecting it costs nothing in solved levels. Measured, not assumed.
        if done is None:
            row["ondisk_done_ever_true"] = None
            row["ondisk_done_status"] = "absent"
        else:
            hits, errs = 0, 0
            for t in trans:
                try:
                    if bool(done(np.asarray(t.next_grid))):
                        hits += 1
                except Exception:
                    errs += 1
            row["ondisk_done_ever_true"] = bool(hits > 0)
            row["ondisk_done_true_count"] = int(hits)
            row["ondisk_done_error_count"] = int(errs)
            row["ondisk_done_status"] = "evaluated"
            # CONSTANT-ness, not just ever-True. The brief's downstream claim was that these
            # engines die at `no_reachable_plan_after_refinement` because `is_level_complete`
            # is constant-FALSE. Measured, that is true of most rows but not all -- and the
            # exceptions are constant-TRUE (the win predicate fires on EVERY observed grid),
            # which is the same defect mirrored: a goal that is never reachable and a goal
            # that is already satisfied everywhere are both non-discriminating, and neither
            # can steer a plan. Recording the direction keeps the two distinguishable instead
            # of collapsing them into "not constant-False".
            n_eval = len(trans) - errs
            row["ondisk_done_is_constant"] = bool(n_eval > 0 and hits in (0, n_eval))
            row["ondisk_done_constant_value"] = (
                None if not (n_eval > 0 and hits in (0, n_eval)) else bool(hits == n_eval)
            )
    except Exception as exc:
        row["ondisk_loaded"] = False
        row["ondisk_error"] = f"load_engine:{type(exc).__name__}:{exc!r}"[:300]

    out: dict[str, dict] = {}
    for mask_on, mval in ((0, None), (1, mask)):
        # SUPPLYING a mask is not the same as ENABLING masking: `WorldModelVerifier` honours
        # `world_model_hud_mask_enabled()` (env `CARNOT_ARC_WM_HUD_MASK`, default off) and
        # otherwise records `hud_mask_status: "disabled"` and grades unmasked. An earlier
        # version of this file passed the mask without setting the flag, so the "mask on" arm
        # silently produced byte-identical numbers to the "mask off" arm -- a structurally
        # dead arm that would have been published as "the mask changes nothing". The status
        # field is what exposed it, which is that field working as designed; it is asserted
        # below rather than trusted.
        os.environ["CARNOT_ARC_WM_HUD_MASK"] = "1" if mask_on else "0"
        for name, engine in arms.items():
            entry: dict = {}
            try:
                entry["live_hidden_state_gate"] = _live_trust_pass(trans, engine, hud_mask=mval)
            except Exception as exc:
                entry["live_hidden_state_gate"] = {"error": f"{type(exc).__name__}:{exc!r}"[:200]}
            try:
                vr = e3.WorldModelVerifier(list(trans), hud_mask=mval).score(engine)
                entry["req6011_change_gate"] = e3.change_gate_decision(vr, enabled=True)
            except Exception as exc:
                entry["req6011_change_gate"] = {"error": f"{type(exc).__name__}:{exc!r}"[:200]}
            out[f"mask={mask_on}|engine={name}"] = entry
    row["arms"] = out
    row["elapsed_s"] = round(time.time() - t0, 3)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--games", type=str, nargs="*", default=None)
    args = ap.parse_args()

    games = args.games or [
        g for g in te.HIDDEN_STATE_GAME_IDS if (E3_DIR / g / "world_model.py").exists()
    ]
    t0 = time.time()
    rows = [_one(g, args.n, s) for g in games for s in args.seeds]
    wall = time.time() - t0

    ok = [r for r in rows if "arms" in r]

    def _pass_set(engine: str, gate: str, mask_on: int) -> list[str]:
        """ADMITTED set for one engine under one gate at one mask setting, per game@seed.
        Sets, not totals -- a total cannot tell you WHICH rows moved."""

        key = "trust_pass" if gate == "live" else "passed"
        gate_key = "live_hidden_state_gate" if gate == "live" else "req6011_change_gate"
        arm = f"mask={mask_on}|engine={engine}"
        return sorted(
            f"{r['game']}@{r['seed']}"
            for r in ok
            if bool(r["arms"].get(arm, {}).get(gate_key, {}).get(key, False))
        )

    engines = ["ondisk", "identity", "base", *SPURIOUS_ARMS]
    admitted = {
        f"mask={m}|{gate}|{eng}": _pass_set(eng, gate, m)
        for m in (0, 1)
        for gate in ("live", "req6011")
        for eng in engines
    }
    n_rows = len(ok)

    # THE ASYMMETRY WITNESS, at the metric's own aggregation level, computed per row so a
    # single averaged number cannot hide a mixture.
    #
    # Both attack arms must be INDISTINGUISHABLE from `base` under the live gate's
    # `consistency` -- that is the blindness. They are then caught by DIFFERENT REQ-6011
    # channels, and separating them is what proves neither channel is decorative:
    #   * `base_plus_spurious` writes on every transition -> strictly LOWER union fidelity.
    #   * `base_plus_noop_hallucination` perturbs ONLY true no-ops, which `change_fidelity`
    #     does not score at all -> its fidelity is EQUAL to base's, and the ONLY thing that
    #     can catch it is the no-op hallucination channel. Fidelity alone would miss it.
    # Measured at mask=1, the setting under which the honest control is admitted at all --
    # comparing attack-vs-base at a setting where BOTH are rejected would prove nothing.
    ident_consistency_rows: list[str] = []
    spurious_worse_fidelity_rows: list[str] = []
    noop_equal_fidelity_higher_noop_rows: list[str] = []
    for r in ok:
        base = r["arms"].get("mask=1|engine=base", {})
        bc = base.get("live_hidden_state_gate", {}).get("heldout_change_consistency")
        bf = base.get("req6011_change_gate", {}).get("change_fidelity")
        bn = base.get("req6011_change_gate", {}).get("noop_hallucination_rate")
        for arm_name in SPURIOUS_ARMS:
            arm = arm_name
            a = r["arms"].get(f"mask=1|engine={arm_name}", {})
            ac = a.get("live_hidden_state_gate", {}).get("heldout_change_consistency")
            af = a.get("req6011_change_gate", {}).get("change_fidelity")
            an = a.get("req6011_change_gate", {}).get("noop_hallucination_rate")
            if None in (bc, ac, bf, af, bn, an):
                continue
            tag = f"{r['game']}@{r['seed']}|{arm}"
            if abs(float(ac) - float(bc)) < 1e-12:
                ident_consistency_rows.append(tag)
            if arm == "base_plus_spurious" and float(af) < float(bf) - 1e-12:
                spurious_worse_fidelity_rows.append(tag)
            if (
                arm == "base_plus_noop_hallucination"
                and abs(float(af) - float(bf)) < 1e-12
                and float(an) > float(bn) + 1e-12
            ):
                noop_equal_fidelity_higher_noop_rows.append(tag)

    # (c) THE PROPOSED FOURTH CHANNEL, CALIBRATED FROM THESE ROWS RATHER THAN FROM TASTE.
    #
    # re86 is the counterexample that motivates it: 40/40 held-out transitions CHANGE the
    # grid, so (i) `n_noop == 0` makes the no-op channel structurally unable to fire, and
    # (ii) the union denominator is huge (1296-1742 genuinely-changed cells), so ONE
    # spurious write per transition costs only ~0.08-0.12 of union fidelity. A near-perfect
    # base at fidelity 1.0 plus a spurious writer still lands at ~0.9, which clears an
    # ABSOLUTE threshold of 0.5 comfortably. The gate catches the spurious writer on dc22
    # only because dc22's honest base is ALREADY at ~0.5 and has no headroom to give away.
    # So union fidelity's sensitivity to spurious writes is proportional to how BAD the
    # engine already is -- exactly backwards.
    #
    # `spurious_changed_cells` is ALREADY computed and ALREADY in the witness dict; it is
    # simply not a gate condition. Normalised PER CHANGING TRANSITION it is scale-free in
    # the change density that defeats union fidelity.
    spur_rate = {}
    for arm_name in ("ondisk", "identity", "base", *SPURIOUS_ARMS):
        vals = []
        for r in ok:
            c = r["arms"].get(f"mask=1|engine={arm_name}", {}).get("req6011_change_gate", {})
            n = c.get("n_changing")
            if not n:
                continue
            vals.append(
                {
                    "row": f"{r['game']}@{r['seed']}",
                    "spurious_per_changing_transition": round(
                        float(c.get("spurious_changed_cells", 0)) / float(n), 6
                    ),
                }
            )
        xs = [v["spurious_per_changing_transition"] for v in vals]
        spur_rate[arm_name] = {
            "n": len(xs),
            "min": round(min(xs), 6) if xs else None,
            "max": round(max(xs), 6) if xs else None,
            "rows_above_0_25": sorted(
                v["row"] for v in vals if v["spurious_per_changing_transition"] > 0.25
            ),
        }
    honest_max = spur_rate["base"]["max"]
    attack_min = spur_rate["base_plus_spurious"]["min"]
    spur_rate["separation"] = {
        "honest_base_max": honest_max,
        "spurious_attack_min": attack_min,
        "separated": bool(
            honest_max is not None and attack_min is not None and honest_max < attack_min
        ),
        "proposed_threshold": 0.25,
        "why_0_25": (
            "well clear of the honest maximum and well clear of the attack minimum, rather "
            "than tight against either -- the same reasoning already used for "
            "WORLD_MODEL_MAX_NOOP_HALLUCINATION_RATE. NOT tuned to make this run pass: any "
            "threshold strictly between the two measured extremes separates all rows."
        ),
    }

    # Item 5, aggregated: on-disk win predicates that are never True on any observed grid.
    done_never = sorted(
        f"{r['game']}@{r['seed']}"
        for r in ok
        if r.get("ondisk_done_status") == "evaluated" and r.get("ondisk_done_ever_true") is False
    )
    done_ever = sorted(
        f"{r['game']}@{r['seed']}"
        for r in ok
        if r.get("ondisk_done_status") == "evaluated" and r.get("ondisk_done_ever_true") is True
    )

    # Everything below is at mask=1 (see the witness note above).
    live_bad = set(admitted["mask=1|live|base_plus_spurious"]) | set(
        admitted["mask=1|live|base_plus_noop_hallucination"]
    )
    req_bad = set(admitted["mask=1|req6011|base_plus_spurious"]) | set(
        admitted["mask=1|req6011|base_plus_noop_hallucination"]
    )
    hole = sorted(live_bad)
    caught = sorted(live_bad - req_bad)
    # The hand-written control's own rows, kept separate: it is the ONLY legitimate
    # must-not-fire arm here (see `base_kind`), so it must not be diluted into an average
    # over the 10 games whose `base` is an admittedly-imperfect lookup table.
    hw_rows = sorted(f"{r['game']}@{r['seed']}" for r in ok if r["game"] == HANDWRITTEN_GAME)
    hw_live_ok = sorted(set(admitted["mask=1|live|base"]) & set(hw_rows))
    hw_req_ok = sorted(set(admitted["mask=1|req6011|base"]) & set(hw_rows))
    hw_live_ok_mask_off = sorted(set(admitted["mask=0|live|base"]) & set(hw_rows))
    hw_req_ok_mask_off = sorted(set(admitted["mask=0|req6011|base"]) & set(hw_rows))
    hw_hole = sorted(live_bad & set(hw_rows))

    # ---- FINDINGS THAT ARE NOT ACCEPTANCE GATES -----------------------------------
    # Recorded as measurements rather than as pass/fail, because they are the SURPRISES.
    # Folding a surprise into a gate would force a choice between "the run failed" and
    # "quietly re-tune until it passes", and neither reports what was actually observed.
    #
    # (a) HOW MUCH ROOM DOES THE ONLY KNOWN-GOOD ENGINE HAVE? T is 0.5 and the control's
    #     union fidelity is barely above it. If the margin is ~0, the pass region is a
    #     knife edge and the constant is not really calibrated -- it is a coincidence.
    hw_margin = []
    for r in ok:
        if r["game"] != HANDWRITTEN_GAME:
            continue
        c = r["arms"].get("mask=1|engine=base", {}).get("req6011_change_gate", {})
        f = c.get("change_fidelity")
        if f is None:
            continue
        hw_margin.append(
            {
                "row": f"{r['game']}@{r['seed']}",
                "change_fidelity": float(f),
                "threshold": float(c.get("fidelity_threshold", 0.0)),
                "margin": round(float(f) - float(c.get("fidelity_threshold", 0.0)), 6),
                "passed": bool(c.get("passed")),
            }
        )
    # (b) DOES THE LIVE GATE ITSELF REJECT THE HONEST ENGINE? Its threshold is on `consistency`
    #     (recall), so it can be stricter than REQ-6011 on a partial-but-honest model even
    #     while being blind to a spurious writer. Both directions of wrong, on one engine.
    hw_live_rejects = sorted(set(hw_rows) - set(hw_live_ok))

    summary = {
        "n_rows": n_rows,
        "n_games": len(sorted({r["game"] for r in ok})),
        "games": sorted({r["game"] for r in ok}),
        "seeds": list(args.seeds),
        "admitted_sets": admitted,
        "rows_where_spurious_admitted_by_live_gate": hole,
        "rows_where_req6011_catches_what_live_gate_admits": caught,
        "handwritten_control_rows": hw_rows,
        "handwritten_control_admitted_by_live_gate_mask_on": hw_live_ok,
        "handwritten_control_admitted_by_req6011_mask_on": hw_req_ok,
        "handwritten_control_admitted_by_live_gate_mask_off": hw_live_ok_mask_off,
        "handwritten_control_admitted_by_req6011_mask_off": hw_req_ok_mask_off,
        "handwritten_control_rows_where_its_attack_arms_are_admitted_by_live": hw_hole,
        "attack_arm_consistency_identical_to_base_rows": sorted(ident_consistency_rows),
        "spurious_arm_fidelity_strictly_worse_rows": sorted(spurious_worse_fidelity_rows),
        "noop_arm_equal_fidelity_but_higher_noop_rate_rows": sorted(
            noop_equal_fidelity_higher_noop_rows
        ),
        "ondisk_done_never_true_rows": done_never,
        "ondisk_done_ever_true_rows": done_ever,
        "ondisk_done_constant_false_rows": sorted(
            f"{r['game']}@{r['seed']}"
            for r in ok
            if r.get("ondisk_done_is_constant") and r.get("ondisk_done_constant_value") is False
        ),
        "ondisk_done_constant_true_rows": sorted(
            f"{r['game']}@{r['seed']}"
            for r in ok
            if r.get("ondisk_done_is_constant") and r.get("ondisk_done_constant_value") is True
        ),
        "ondisk_done_state_discriminating_rows": sorted(
            f"{r['game']}@{r['seed']}"
            for r in ok
            if r.get("ondisk_done_status") == "evaluated"
            and r.get("ondisk_done_is_constant") is False
        ),
        "FINDING_handwritten_control_fidelity_margin_over_threshold": hw_margin,
        "FINDING_live_gate_rejects_the_handwritten_control_on_rows": hw_live_rejects,
        "FINDING_rows_where_a_spurious_writer_escapes_BOTH_gates": sorted(live_bad & req_bad),
        "PROPOSED_fourth_channel_spurious_per_changing_transition": spur_rate,
        "rows_with_zero_noop_transitions_where_the_noop_channel_cannot_fire": sorted(
            f"{r['game']}@{r['seed']}"
            for r in ok
            if r["arms"].get("mask=1|engine=base", {}).get("req6011_change_gate", {}).get("n_noop")
            == 0
        ),
        "errors": sorted(
            f"{r['game']}@{r['seed']}:{r.get('error') or r.get('ondisk_error')}"
            for r in rows
            if r.get("error") or r.get("ondisk_error")
        ),
    }

    gates = {
        # THE HOLE: an engine right about every real change that ALSO writes cells reality
        # never wrote is ADMITTED by the live hidden-state gate.
        "acceptance_gate_live_hidden_state_gate_admits_a_spurious_writer": bool(hole),
        # REQ-6011 rejects the great majority of the rows the live gate admits -- but NOT
        # all of them, and the exceptions are the point (see the re86 note above and
        # FINDING_rows_where_a_spurious_writer_escapes_BOTH_gates). Stated as "most", not
        # "all", because the honest measurement says most.
        "acceptance_gate_req6011_rejects_most_rows_the_live_gate_admits": bool(
            hole and len(live_bad & req_bad) < len(live_bad)
        ),
        # MUST-NOT-FIRE, on the ONLY legitimate positive control (the hand-written general
        # rule on dc22 -- a HIDDEN-STATE game, so it routes through this very branch).
        # REQ-6011 must admit it on EVERY row, or the gate is "reject everything".
        "acceptance_gate_req6011_admits_the_handwritten_control_all_rows": bool(
            hw_rows and len(hw_req_ok) == len(hw_rows)
        ),
        # THE TWO REPAIRS ARE NOT INDEPENDENT ON THE POSITIVE CONTROL: REQ-6011 admits the
        # hand-written engine on every row WITH the mask and on NO row without it. exp6011
        # recorded the same flip for its own non-hidden gate
        # (`mask_flips_reject_to_admit_all_seeds`); this reproduces it independently, on the
        # full production-matching corpus, and the fidelities land on exp6011's mask-on
        # numbers to 6 decimals (0.814815 / 0.760870 / 0.735849) -- a cross-check that this
        # harness and that one are measuring the same thing.
        #
        # OPERATOR CONSEQUENCE: flipping the change gate WITHOUT also flipping the HUD mask
        # would reject the one engine known to be genuinely good. The gate-only arm of the
        # four-arm matrix is not a safe partial ship.
        "acceptance_gate_mask_is_required_for_the_control_to_be_admitted": bool(
            hw_rows and not hw_req_ok_mask_off and len(hw_req_ok) == len(hw_rows)
        ),
        # THE PROPOSED FOURTH CHANNEL SEPARATES CLEANLY. Honest max strictly below attack
        # min across every matched row -- which is what makes a threshold calibrated rather
        # than chosen. If this ever goes false the constant must be re-derived, not nudged.
        "acceptance_gate_proposed_spurious_rate_channel_separates_honest_from_attack": bool(
            spur_rate["separation"]["separated"]
        ),
        # ...and it catches the rows that escape BOTH shipped gates, which is the entire
        # reason to propose it. An empty escape set would make the proposal unmotivated.
        "acceptance_gate_proposed_channel_catches_the_rows_that_escape_both_gates": bool(
            (live_bad & req_bad)
            and set(live_bad & req_bad) <= set(spur_rate["base_plus_spurious"]["rows_above_0_25"])
        ),
        # MATCHED DEMONSTRATION: on the very rows where the live gate admits the GOOD
        # engine, it also admits the attack arms. "Admits bad" and "admits good" shown on
        # one corpus, not on two conveniently different ones.
        "acceptance_gate_hole_demonstrated_on_the_rows_that_admit_the_control": bool(
            hw_live_ok and set(hw_live_ok) <= set(hw_hole)
        ),
        # The origin incident, on THIS branch: identity must be rejected by both gates.
        "acceptance_gate_identity_rejected_by_both_gates_all_rows": bool(
            n_rows > 0
            and not admitted["mask=1|live|identity"]
            and not admitted["mask=1|req6011|identity"]
            and not admitted["mask=0|live|identity"]
            and not admitted["mask=0|req6011|identity"]
        ),
        # The blindness itself: both attack arms score IDENTICALLY to base under the live
        # gate's consistency, on every row where both were measured.
        # (2 attack arms per measured row -- so equality here means it held EVERYWHERE, not
        # on average.)
        "acceptance_gate_live_consistency_cannot_see_either_attack": bool(
            n_rows > 0 and len(ident_consistency_rows) == 2 * n_rows
        ),
        # Both REQ-6011 channels are load-bearing and NON-INTERCHANGEABLE: the spurious arm
        # is caught by fidelity, and the no-op arm has EQUAL fidelity to base and is caught
        # only by the no-op channel. If either list is empty, one channel is decorative.
        "acceptance_gate_fidelity_channel_catches_the_spurious_arm": bool(
            spurious_worse_fidelity_rows
        ),
        "acceptance_gate_noop_channel_is_the_only_thing_that_catches_the_noop_arm": bool(
            noop_equal_fidelity_higher_noop_rows
        ),
        # ITEM 5 (the downstream cost of rejecting these engines), VERIFIED not assumed:
        # no on-disk win predicate discriminates between observed states, so none of them
        # could have steered a plan even if the gate had trusted its dynamics. Rejecting
        # them therefore costs nothing in solved levels -- which is the claim, now measured.
        "acceptance_gate_no_ondisk_win_predicate_is_state_discriminating": bool(
            n_rows > 0
            and not [
                r
                for r in ok
                if r.get("ondisk_done_status") == "evaluated"
                and r.get("ondisk_done_is_constant") is False
            ]
        ),
        # ...and every on-disk engine is rejected on its DYNAMICS anyway, independently of
        # its goal predicate -- so the two justifications do not lean on each other.
        "acceptance_gate_every_ondisk_engine_rejected_by_req6011": bool(
            n_rows > 0 and not admitted["mask=1|req6011|ondisk"]
        ),
        # THE MASK ARM IS ACTUALLY LIVE. Guards the exact dead-arm bug this file shipped
        # once: supplying a mask without enabling masking makes mask=1 byte-identical to
        # mask=0, and the run would have published "the mask changes nothing".
        "acceptance_gate_mask_arm_is_not_structurally_dead": bool(
            n_rows > 0
            # "applied" OR "unresolved": three games (cn04/sc25/sk48) have no edge-bar HUD
            # for the live classifier to find, so `unresolved` is the honest answer there and
            # not a dead arm. "disabled" is the failure -- it means masking was never turned
            # on, which is the bug this gate exists to catch.
            and all(
                r["arms"]["mask=1|engine=base"]["req6011_change_gate"]["hud_mask_status"]
                != "disabled"
                for r in ok
            )
            and all(
                r["arms"]["mask=0|engine=base"]["req6011_change_gate"]["hud_mask_status"]
                == "disabled"
                for r in ok
            )
        ),
        "acceptance_gate_nonempty_measurement": bool(n_rows > 0),
    }
    gates["acceptance_gate_passed"] = all(gates.values())

    payload = {
        "experiment": 6012,
        "experiment_id": 6012,
        "title": "REQ-ARC-WMTE-6012: the hidden-state trust gate admits the spurious writer "
        "REQ-6011 rejects",
        "provenance": _provenance(),
        "honest_verdict": (
            "complete_hidden_state_gate_hole_measured_req6011_not_wired_into_that_branch"
        ),
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "verifier_is_oracle": False,
        "random_seed": int(args.seeds[0]),
        "random_seeds_used": list(args.seeds),
        # TWO DIFFERENT CLOCKS, deliberately not the same number. `measurement_wall_s` is
        # the time actually spent MEASURING -- the sum of the per-row `elapsed_s`, which is
        # collection + scoring. `duration_s` is read HERE, at the end, so it additionally
        # covers aggregation and gate evaluation. Publishing one value under both names is
        # what `adversarial_verify`'s TAUTOLOGY check is for, and it correctly flagged this
        # artifact twice: first for literally assigning the same variable to both, then --
        # after a fix that only LOOKED different -- because the loop clock was stopped
        # before the aggregation it claimed to include, so the "two" numbers were still the
        # same measurement. The second catch is the more instructive one.
        # 6 dp, NOT 3. `duration_s` (this process's wall time) and `measurement_wall_s`
        # (the sum of per-row elapsed_s) are DIFFERENT quantities, but they differ only by
        # the analyser's own sub-millisecond overhead. Rounded to 3 dp they can land on the
        # SAME value -- which happened on 2026-07-28 (both 29.22) and tripped
        # adversarial_verify's TAUTOLOGY check ('two distinct metrics agreeing to >5 sig
        # figs is more likely a bug than a finding'), a CRITICAL flag that would have
        # quarantined a perfectly honest artifact via the fabrication gate. The fix is to
        # stop DESTROYING the information that distinguishes them, not to re-run until the
        # dice differ and not to exempt the check: at 6 dp the two genuinely-distinct
        # clocks are visibly distinct, which is the truth the detector needs to see.
        "measurement_wall_s": round(sum(float(r.get("elapsed_s") or 0.0) for r in rows), 6),
        "duration_s": round(time.time() - t0, 6),
        "model_specs": {
            "note": "no model invoked; pure numpy over cached offline-arcade transitions",
        },
        "preconditions_checked": [
            {"resource": "results/arc_e3/<game>/world_model.py", "available": True},
            {"resource": "offline arcade (arc_solver_kit.offline_arcade)", "available": True},
            {"resource": "gpu", "available": False, "note": "not required for this substrate"},
        ],
        **gates,
        "summary": summary,
        "rows": rows,
    }
    preserve_freshness_acknowledgements(payload, OUT)
    # Full merge-preserve supersedes the ack-only call above (kept;
    # idempotent): carries rebuild_note_* and any other hand-authored
    # top-level key (REQ-OPS-REBUILD-PRESERVE-1). Before the checksum so
    # the checksum covers the bytes written; the prior build's carried
    # checksum is analyzer-owned and dropped before rehashing.
    from artifact_merge_preserve import merge_preserve_with_file

    payload = merge_preserve_with_file(OUT, payload)
    payload.pop("reproducibility_checksum", None)
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    payload["reproducibility_checksum"] = hashlib.sha256(blob).hexdigest()
    OUT.write_text(json.dumps(payload, indent=1, default=str))
    print(
        json.dumps({k: v for k, v in payload.items() if k.startswith("acceptance_gate")}, indent=1)
    )
    print("rows:", n_rows, "wall_s:", round(wall, 1))
    print("HOLE rows (live admits spurious):", len(hole))
    print("REQ6011 catches:", len(caught))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
